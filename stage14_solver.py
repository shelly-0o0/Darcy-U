"""Stage 14 solver certification on the original double-precision system.

The public solver never reads a reference solution or a geometry.  Long-double
arithmetic evaluates residuals of a matrix assembled in double precision; it
neither reassembles the matrix nor represents a high-precision PDE solution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_info

from sequence_baselines import FixedAMGHierarchy

ROOT = Path(__file__).resolve().parent
CONFIG = dict(relative_residual_limit=1e-9, componentwise_backward_limit=1e-12,
              initial_cg_rtol=1e-10, correction_cg_rtol=1e-3,
              max_iterations=1000, max_refinements=2, amg_seed=51700,
              residual_arithmetic="longdouble", matrix_arithmetic="float64")


class ResidualEvaluator:
    """Reusable, charged residual and backward-error evaluation of stored A."""
    def __init__(self, a):
        self.a = sp.csr_matrix(a, dtype=np.float64, copy=True)
        self.extended = self.a.astype(np.longdouble)
        self.absolute = abs(self.extended)
        self.a_norm_inf = np.asarray(self.absolute.sum(axis=1)).ravel().max()

    def evaluate(self, u, b):
        u = np.asarray(u, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if u.ndim == 1:
            u, b = u[:, None], b[:, None]
        ue, be = u.astype(np.longdouble), b.astype(np.longdouble)
        r = be - self.extended @ ue
        rd = b - self.a @ u
        denominator = self.absolute @ abs(ue) + abs(be)
        tiny = np.finfo(np.longdouble).tiny
        bnorm = np.sqrt(np.sum(be * be, axis=0))
        rn = np.sqrt(np.sum(r * r, axis=0)) / np.maximum(bnorm, tiny)
        rnd = np.linalg.norm(rd, axis=0) / np.maximum(np.linalg.norm(b, axis=0), np.finfo(float).tiny)
        cw = np.max(abs(r) / np.maximum(denominator, tiny), axis=0)
        norm_denom = self.a_norm_inf * np.max(abs(ue), axis=0) + np.max(abs(be), axis=0)
        nw = np.max(abs(r), axis=0) / np.maximum(norm_denom, tiny)
        columns = [dict(relative_residual=float(rn[j]), relative_residual_double=float(rnd[j]),
                        componentwise_backward_error=float(cw[j]), normwise_backward_error=float(nw[j]))
                   for j in range(u.shape[1])]
        result = {name: max(c[name] for c in columns) for name in columns[0]}
        result["per_rhs"] = columns
        result["accepted"] = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(rn)) and
            np.all(rn <= CONFIG["relative_residual_limit"]) and
            np.all(cw <= CONFIG["componentwise_backward_limit"]))
        return result, r


def _validate(a, b):
    a = sp.csr_matrix(a, dtype=np.float64, copy=True)
    a.sum_duplicates(); a.sort_indices()
    b = np.array(b, dtype=np.float64, copy=True)
    vector = b.ndim == 1
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    if b.ndim not in (1, 2) or b.shape[0] != a.shape[0] or (b.ndim == 2 and b.shape[1] == 0):
        raise ValueError("b must have shape (n,) or (n,n_rhs)")
    if not np.all(np.isfinite(a.data)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must be finite")
    if np.any(a.diagonal() <= 0):
        raise ValueError("A must have a positive diagonal")
    asymmetry = a - a.T
    if asymmetry.nnz and abs(asymmetry.data).max() > 1e-13 * max(abs(a.data).max(), 1.):
        raise ValueError("A must be symmetric")
    return a, b[:, None] if vector else b, vector


def solve_system(A, b, method, *, max_refinements=2):
    """Return (u, metrics), charging validation, setup, solves and certification.

    ``b`` is either a vector or a matrix whose columns are right hand sides.
    ``direct`` uses one ordinary sparse LU; ``amg`` solves DAD y=Db, u=Dy,
    D=diag(A)^(-1/2), using one fixed symmetric AMG hierarchy. Up to two
    original-system residual corrections reuse only this call's factorization
    or hierarchy. Acceptance is identical for both methods and all RHS columns.
    Reference/QoI checks and PDE assembly are the caller's responsibility.
    """
    start = tick = perf_counter(); timings = {}
    if method not in ("direct", "amg"):
        raise ValueError("method must be direct or amg")
    if not isinstance(max_refinements, int) or not 0 <= max_refinements <= CONFIG["max_refinements"]:
        raise ValueError("max_refinements must be 0, 1 or 2")
    a, bb, vector = _validate(A, b)
    evaluator = ResidualEvaluator(a)
    now = perf_counter(); timings["validation_and_residual_setup_seconds"] = now - tick; tick = now
    if method == "direct":
        factor = sla.splu(a.tocsc())
        scale = np.ones(a.shape[0]); scaled = a; hierarchy = None
    else:
        scale = 1. / np.sqrt(a.diagonal())
        scaled = (sp.diags(scale) @ a @ sp.diags(scale)).tocsr()
        hierarchy = FixedAMGHierarchy(scaled, seed=CONFIG["amg_seed"])
        factor = None
    now = perf_counter(); timings["factor_or_scaled_amg_setup_seconds"] = now - tick; tick = now
    iterations = [0] * bb.shape[1]; infos = [[] for _ in iterations]
    refinements = [0] * bb.shape[1]

    def one_solve(rhs, j, correction=False):
        if factor is not None:
            infos[j].append(0)
            return factor.solve(rhs)
        def callback(_):
            iterations[j] += 1
        remaining = CONFIG["max_iterations"] - iterations[j]
        if remaining <= 0:
            infos[j].append(CONFIG["max_iterations"])
            return np.zeros_like(rhs)
        y, info = sla.cg(scaled, scale * rhs, M=hierarchy.preconditioner,
                         rtol=CONFIG["correction_cg_rtol"] if correction else CONFIG["initial_cg_rtol"],
                         atol=0., maxiter=remaining, callback=callback)
        infos[j].append(int(info))
        return scale * y

    u = np.column_stack([one_solve(bb[:, j], j) for j in range(bb.shape[1])])
    now = perf_counter(); timings["initial_solve_seconds"] = now - tick; tick = now
    check, residual = evaluator.evaluate(u, bb)
    initial = check
    history = [check]
    for _ in range(max_refinements):
        failing = [j for j, c in enumerate(check["per_rhs"])
                   if not np.isfinite(c["relative_residual"]) or
                   c["relative_residual"] > CONFIG["relative_residual_limit"] or
                   c["componentwise_backward_error"] > CONFIG["componentwise_backward_limit"]]
        if not failing:
            break
        for j in failing:
            if any(info < 0 for info in infos[j]):
                continue
            correction = one_solve(np.asarray(residual[:, j], dtype=np.float64), j, True)
            u[:, j] += correction
            refinements[j] += 1
        check, residual = evaluator.evaluate(u, bb)
        history.append(check)
    now = perf_counter(); timings["certification_and_refinement_seconds"] = now - tick
    all_info_zero = all(info == 0 for column in infos for info in column)
    metrics = dict(method=method, **check, initial=initial, residual_history=history,
                   converged=bool(check["accepted"] and all_info_zero), all_info_zero=all_info_zero,
                   info=0 if all_info_zero else next(info for column in infos for info in column if info != 0),
                   solver_info=infos, iterations=iterations, refinements=refinements,
                   total_iterations=sum(iterations), total_refinements=sum(refinements),
                   rhs_count=bb.shape[1], acceptance=dict(CONFIG),
                   amg=None if hierarchy is None else hierarchy.stats,
                   symmetric_diagonal_scaling=method == "amg",
                   extended_precision_bits=int(np.finfo(np.longdouble).nmant + 1),
                   original_matrix_dtype="float64", timings=timings)
    timings["total_seconds"] = perf_counter() - start
    return u[:, 0] if vector else u, metrics


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_development(output):
    """Eight independent development geometries; not a confirmation benchmark."""
    import stage10_experiment as old
    from stage10_goal_numerics import _vertices
    from stage13_local_fem import prepare_loads, assemble_local
    from numerics import _full_cell_matrix, solve_pcg
    if output.exists():
        raise FileExistsError("Use a new development directory; never overwrite recorded outputs")
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("Single thread pools required")
    output.mkdir(parents=True)
    inputs = ["stage14_solver.py", "test_stage14_solver.py", "stage13_local_fem.py",
              "stage10_experiment.py", "stage10_goal_numerics.py", "numerics.py", "sequence_baselines.py"]
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(), development_only=True,
        config=CONFIG, geometry_splits={"id": list(range(55000,55004)), "contrast": list(range(55100,55104))},
        nc=64, four_rhs_columns=True, no_confirmation_geometry_used=True,
        source_sha256={name: _sha(ROOT/name) for name in inputs},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), threadpools=threadpool_info(),
            thread_environment={n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant + 1)),
        timing_scope="Solver-only times include validation, conversions, factor/hierarchy setup, all four RHS solves, residual/backward-error checks and corrections. Matrix/load assembly separately recorded. Development times are descriptive, not a formal speed gate.")
    _dump(output/"freeze.json", freeze)
    rows=[]; start_all=perf_counter()
    for split, ids in freeze["geometry_splits"].items():
        for geometry in ids:
            assembly_start=perf_counter(); nc=64
            k=old.coefficient(geometry,2*nc,split=="contrast"); funcs=old.rhs_functions(geometry)
            _,free=_vertices(2*nc); a=_full_cell_matrix(k)[free][:,free].tocsr()
            loads=prepare_loads(nc,funcs).fine_free_loads; b=loads.T
            assembly_seconds=perf_counter()-assembly_start
            coarse=assemble_local(nc,k,funcs,[])
            uc=sla.splu(coarse.A).solve(coarse.loads[0]); qc=coarse.loads[1:]@uc
            scales=np.maximum(abs(qc),max(float(abs(qc).max())*1e-12,1e-30))
            outcomes={}
            # Plain historical LU and old AMG retain their original solve policy.
            for method in ("plain_lu", "old_amg"):
                start=perf_counter()
                evaluator=ResidualEvaluator(a)
                if method=="plain_lu":
                    factor=sla.splu(a.tocsc()); u=factor.solve(b); legacy=None
                else:
                    hierarchy=FixedAMGHierarchy(a,seed=CONFIG["amg_seed"])
                    answers=[solve_pcg(a,b[:,j],M=hierarchy.preconditioner,rtol=1e-10,maxiter=1000,max_restarts=3) for j in range(4)]
                    u=np.column_stack([x[0] for x in answers]); legacy=[x[1] for x in answers]
                checks,_=evaluator.evaluate(u,b)
                outcomes[method]=(u,dict(**checks,solver=legacy,total_seconds=perf_counter()-start))
            for method, base, refine in (("scaled_amg_no_refine","amg",0), ("unified_direct","direct",2), ("unified_amg","amg",2)):
                outcomes[method]=solve_system(a,b,base,max_refinements=refine)
            uf=outcomes["plain_lu"][0]; qf=loads[1:]@uf[:,0]
            denominator=float(np.linalg.norm((qf-qc)/scales))
            for method,(u,metrics) in outcomes.items():
                q=loads[1:]@u[:,0]
                e=float(np.linalg.norm((q-qf)/scales)/denominator)
                rows.append(dict(split=split,geometry=geometry,nc=nc,method=method,
                    coefficient_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
                    matrix_data_sha256=hashlib.sha256(a.data.tobytes()).hexdigest(),
                    loads_sha256=hashlib.sha256(loads.tobytes()).hexdigest(),
                    shared_fine_assembly_seconds=assembly_seconds,solver=metrics,
                    qoi=q.tolist(),fine_qoi=qf.tolist(),coarse_qoi=qc.tolist(),qoi_scales=scales.tolist(),
                    coarse_scaled_error=denominator,fine_qoi_discrepancy_ratio=e,
                    fine_qoi_accepted=bool(denominator>=1e-12 and e<=1e-5),
                    original_system_solution_relative_difference=float(np.linalg.norm(u-uf)/np.linalg.norm(uf)),
                    solution_sha256=hashlib.sha256(u.tobytes()).hexdigest()))
            _dump(output/"rows.json",rows)
            print(json.dumps(dict(event="solver_development",split=split,geometry=geometry,rows=len(rows))),flush=True)
    for name,digest in freeze["source_sha256"].items():
        if _sha(ROOT/name)!=digest:
            raise RuntimeError("Frozen development source changed: "+name)
    _dump(output/"complete.json",dict(complete=True,development_only=True,geometries=8,rows=len(rows),
        wall_seconds=perf_counter()-start_all,freeze_sha256=_sha(output/"freeze.json"),
        rows_sha256=_sha(output/"rows.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="results/stage14_solver_development")
    args=parser.parse_args();run_development((ROOT/args.output).resolve())
