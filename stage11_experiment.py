"""Frozen-weight, equal-DOF, one-step local refinement experiment.

Online pipelines accept geometry seeds and a frozen model only. Exact reference
solutions enter evaluation after those pipelines return. Fine direct solves all
three QoIs with ONE primal solve, without coarse or adjoint setup.
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
import torch
from threadpoolctl import threadpool_info

import stage10_experiment as old
from stage10_error_model import ErrorOperator
from stage10_goal_numerics import (assemble_nested, assemble_load, free_to_full,
    element_indicators, p1_prolongation, _vertices, _load_nodes)
from numerics import _full_cell_matrix, _cell_triangles
from sequence_baselines import FixedAMGHierarchy
from stage11_adaptive import budget_mark, build_space, solve_adaptive

ROOT = Path(__file__).resolve().parent
CONFIG = dict(nc=16, new_dofs=64, geometries_per_split=16, repetitions=5,
    splits={"id": 52000, "contrast": 52100}, source=0, goals=[1, 2, 3],
    torch_threads=1, method_order_seed=52991, uniform_seed=52900,
    near_zero_coarse_error=1e-12, paired_denominator_floor=1e-12,
    seed_aggregation="median within geometry",
    accuracy_ratio_gate=.90, minimum_wins=12, seed_ratio_limit=1.10,
    cheap_baseline_ratio_limit=1., time_ratio_limit=1.)
CONTROLS = ("jacobi1", "jacobi8", "amg_v1", "recovery", "geometry_uniform", "oracle_indicator")
METHODS = [(kind, seed) for kind in old.KINDS for seed in old.SEEDS] + [(x, None) for x in CONTROLS] + [("fine_direct", None)]


def quadratic_recovery_matrix(nc):
    """Tensor-product three-node quadratic interpolation of coarse nodal data.

    At fine coordinate t in coarse-cell units choose the nodes starting at
    clip(floor(t)-1,0,nc-2). Boundary values remain zero. This inexpensive
    polynomial recovery is not a conservative flux estimator or certificate.
    """
    nf = 2*nc
    rows, cols, vals = [], [], []
    for i in range(nf+1):
        t = i/2
        nodes = np.arange(min(max(int(np.floor(t))-1, 0), nc-2),
                          min(max(int(np.floor(t))-1, 0), nc-2)+3)
        for node in nodes:
            others = nodes[nodes != node]
            rows.append(i); cols.append(node)
            vals.append(float(np.prod((t-others)/(node-others))))
    one = sp.coo_matrix((vals, (rows, cols)), shape=(nf+1, nc+1)).tocsr()
    full = sp.kron(one, one, format="csr")
    _, fine = _vertices(nf); _, coarse = _vertices(nc)
    return full[fine][:, coarse].tocsr()


def assemble_fine_only(k, funcs):
    """Same matrix/quadrature as nested assembly; deliberately no P or Ac."""
    nf = len(k)
    coords, free = _vertices(nf)
    triangles = _cell_triangles(nf, nf)
    A = _full_cell_matrix(k)[free][:, free].tocsr()
    loads = []
    for fun in funcs:
        local = _load_nodes(nf, triangles, coords, fun)
        full = np.zeros((nf+1)**2)
        np.add.at(full, triangles.ravel(), local.ravel())
        loads.append(full[free])
    return A, np.stack(loads)


def predict_defects(case, coarse, residual, method, model):
    """Return three approximate adjoint defects, all reference-free except oracle."""
    r = residual[1:]; a = case.Af; d = a.diagonal()
    if method in old.KINDS:
        m = case.nf-1
        features = torch.from_numpy(old.features(case.k_fine, d.reshape(m,m), case.nf).astype("float32"))
        rr = torch.from_numpy(r.reshape(3,1,m,m).astype("float32"))
        diagonal = torch.from_numpy(d.reshape(1,1,m,m).astype("float32")).expand(3,-1,-1,-1)
        with torch.inference_mode():
            return model(features[None].expand(3,-1,-1,-1), rr, diagonal, case.nf).numpy().reshape(3,-1).astype("float64")
    if method == "jacobi1":
        return r/d
    if method == "jacobi8":
        value = np.zeros_like(r)
        for _ in range(8):
            value += (2/3)*(r-(a@value.T).T)/d
        return value
    if method == "amg_v1":
        hierarchy = FixedAMGHierarchy(a, seed=51700)
        return np.stack([hierarchy.preconditioner@v for v in r])
    if method == "recovery":
        recovery = quadratic_recovery_matrix(case.nc)
        return ((recovery-case.P)@coarse[:,1:]).T
    if method == "oracle_indicator":
        return sla.splu(a.tocsc()).solve(r.T).T
    raise ValueError(method)


def online_pipeline(geometry, nc, ood, method, model=None, budget=None):
    """Fresh online work, no exact solution argument or shared solve cache."""
    start = tick = perf_counter(); timings = {}
    def lap(name):
        nonlocal tick
        now = perf_counter(); timings[name] = now-tick; tick = now
    budget = nc*nc//4 if budget is None else budget
    k = old.coefficient(geometry, 2*nc, ood)
    funcs = old.rhs_functions(geometry)
    if method == "fine_direct":
        a, loads = assemble_fine_only(k, funcs)
        lap("geometry_assembly_seconds")
        u = sla.splu(a.tocsc()).solve(loads[0])
        lap("fine_factor_solve_seconds")
        qoi = loads[1:]@u
        residual = float(np.linalg.norm(a@u-loads[0])/np.linalg.norm(loads[0]))
        lap("qoi_diagnostics_seconds")
        timings["total_seconds"] = perf_counter()-start
        return dict(u=u, qoi=qoi, relative_residual=residual, timings=timings,
                    method=method, selected_cells=[], added_dofs=len(u)-(nc-1)**2,
                    budget_matched=False, solve_dofs=len(u))

    case = assemble_nested(nc, k, funcs[0])
    loads = np.stack([case.ff]+[assemble_load(case, f)[0] for f in funcs[1:]])
    lap("geometry_assembly_seconds")
    lu = sla.splu(case.Ac.tocsc())
    active = loads[:1] if method == "geometry_uniform" else loads
    coarse = lu.solve(np.asarray(case.P.T@active.T))
    lifted = np.asarray(case.P@coarse)
    coarse_qoi = loads[1:]@lifted[:,0]
    scales = np.maximum(abs(coarse_qoi), max(float(abs(coarse_qoi).max())*1e-12, 1e-30))
    lap("coarse_factor_solve_seconds")
    if method == "geometry_uniform":
        scores = np.random.default_rng(CONFIG["uniform_seed"]+nc).random(nc*nc)
        lap("geometry_only_score_seconds")
        projection_orthogonality = None
    else:
        residual = (active.T-case.Af@lifted).T
        primal_full = free_to_full(case, lifted[:,0])
        primal = dict(cell_residual_nodes=case.load_nodes-np.einsum("ctij,ctj->cti",
                      case.element_matrices, primal_full[case.triangles]), r_u=residual[0])
        lap("residual_seconds")
        defect = predict_defects(case, coarse, residual, method, model)
        lap("prediction_setup_apply_seconds")
        coarse_part = case.P@lu.solve(np.asarray(case.P.T@(case.Af@defect.T)))
        projected = defect-coarse_part.T
        lap("projection_seconds")
        scores = np.sum([abs(element_indicators(case, primal, v)["coarse_cell"].ravel())/scale
                         for v,scale in zip(projected, scales)],axis=0)
        action = case.Af@projected.T
        projection_orthogonality = float(np.max(np.linalg.norm(case.P.T@action,axis=0)/
                                               np.maximum(np.linalg.norm(action,axis=0),1e-30)))
        lap("indicator_diagnostics_seconds")
    selected, marking = budget_mark(scores, nc, budget)
    lap("marking_seconds")
    T, metadata = build_space(case, selected)
    lap("mesh_space_seconds")
    solution = solve_adaptive(case, T)
    lap("adaptive_solve_seconds")
    u = solution["lifted_u"]; qoi = loads[1:]@u
    # Independent coarse-embedding check uses the nodal degrees of freedom.
    free_ids = case.full_to_free[np.asarray(metadata["independent_free_node_ids"])]
    difference = T@case.P[free_ids]-case.P
    embedding = float(np.max(abs(difference.data),initial=0.))
    interpolation_hash = hashlib.sha256(T.data.tobytes()+T.indices.tobytes()+T.indptr.tobytes()).hexdigest()
    lap("qoi_diagnostics_seconds")
    timings["total_seconds"] = perf_counter()-start
    return dict(u=u, qoi=qoi, coarse_qoi=coarse_qoi, scales=scales,
        relative_residual=solution["relative_residual"], timings=timings,
        adaptive_solve_components=solution["timings"], method=method,
        selected_cells=selected.tolist(), scores=scores.tolist(), marking=marking,
        added_dofs=metadata["added_dofs"], budget_matched=True,
        solve_dofs=metadata["adaptive_dofs"], interpolation_sha256=interpolation_hash,
        coarse_embedding_max_abs=embedding, projection_orthogonality=projection_orthogonality,
        geometric_triangle_count=metadata["geometric_triangle_count"],
        hanging_nodes=len(metadata["hanging_constraints"]))


def reference_case(geometry, nc, ood):
    start = perf_counter()
    k = old.coefficient(geometry, 2*nc, ood)
    a, loads = assemble_fine_only(k, old.rhs_functions(geometry))
    p,_ = p1_prolongation(nc); ac = p.T@a@p
    uc = p@sla.splu(ac.tocsc()).solve(np.asarray(p.T@loads[0]))
    uf = sla.splu(a.tocsc()).solve(loads[0])
    qf = loads[1:]@uf; qc = loads[1:]@uc
    scales = np.maximum(abs(qc), max(float(abs(qc).max())*1e-12, 1e-30))
    delta = uf-uc; energy = float(delta@(a@delta))
    return dict(A=a, loads=loads, uf=uf, uc=uc, qf=qf, qc=qc, scales=scales,
        energy=energy, coarse_scaled_error=float(np.linalg.norm((qf-qc)/scales)),
        coefficient_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
        fine_relative_residual=float(np.linalg.norm(a@uf-loads[0])/np.linalg.norm(loads[0])),
        seconds=perf_counter()-start)


def evaluate(output, reference):
    for values in (output["u"],output["qoi"],reference["uf"],reference["qf"],reference["qc"],reference["scales"]):
        if not np.all(np.isfinite(values)):raise RuntimeError("Nonfinite solution or goal data")
    if not np.isfinite(reference["energy"]) or reference["energy"]<=0 or np.any(reference["scales"]<=0):
        raise RuntimeError("Invalid reference energy or QoI scales")
    error = reference["qf"]-output["qoi"]
    scaled = float(np.linalg.norm(error/reference["scales"]))
    denominator = reference["coarse_scaled_error"]
    near_zero = denominator < CONFIG["near_zero_coarse_error"]
    delta = reference["uf"]-output["u"]
    er = float(delta@(reference["A"]@delta)/reference["energy"])
    if not np.isfinite(er) or er < 0 or not np.isfinite(output["relative_residual"]) or er > 1+1e-8 or output["relative_residual"] > 1e-8:
        raise RuntimeError("Nested-space energy or linear solve verification failed")
    if output.get("coarse_embedding_max_abs",0) > 1e-12:
        raise RuntimeError("Coarse space not embedded in adaptive space")
    orth = output.get("projection_orthogonality")
    if orth is not None and orth > 1e-8:
        raise RuntimeError("Projected defects are not coarse-orthogonal")
    if "coarse_qoi" in output and not np.allclose(output["coarse_qoi"],reference["qc"],rtol=1e-10,atol=1e-15):
        raise RuntimeError("Online and offline coarse solves disagree")
    result = {k:v for k,v in output.items() if k not in ("u","qoi","coarse_qoi","scales","timings")}
    result.update(qoi=output["qoi"].tolist(), fine_qoi=reference["qf"].tolist(),
        coarse_qoi=reference["qc"].tolist(), qoi_scales=reference["scales"].tolist(),
        signed_qoi_errors=error.tolist(), signed_coarse_errors=(reference["qf"]-reference["qc"]).tolist(),
        absolute_error_l2=float(np.linalg.norm(error)), scaled_error_l2=scaled,
        coarse_scaled_error=denominator, coarse_error_near_zero=near_zero,
        error_reduction_ratio=None if near_zero else scaled/denominator,
        energy_error_ratio=er, solution_sha256=hashlib.sha256(output["u"].tobytes()).hexdigest())
    return result


def load_models():
    source = ROOT/"results/stage10_pilot"
    frozen = json.loads((source/"freeze.json").read_text())
    records = [json.loads((source/f"train_{kind}_{seed}.json").read_text())
               for kind in old.KINDS for seed in old.SEEDS]
    old.assert_frozen(frozen, records)
    models = {}; loading = []
    for rec in records:
        tick = perf_counter()
        payload = torch.load(ROOT/rec["checkpoint"], map_location="cpu",weights_only=False)
        model = ErrorOperator(rec["kind"],width=payload["config"]["width"],levels=payload["config"]["levels"])
        model.load_state_dict(payload["state_dict"]); model.eval()
        models[(rec["kind"],rec["seed"])] = model
        loading.append(dict(kind=rec["kind"],seed=rec["seed"],seconds=perf_counter()-tick,
                            checkpoint=rec["checkpoint"],sha256=rec["checkpoint_sha256"]))
    return models, loading, frozen, records


def run(output, smoke=False):
    if output.exists():
        raise FileExistsError("Use a new directory; frozen runs are never overwritten")
    models, loading, oldfreeze, records = load_models()
    config = dict(CONFIG)
    if smoke:
        config.update(nc=8,new_dofs=16,geometries_per_split=1,repetitions=1,
                      splits={"id":52970,"contrast":52980})
    files = ["stage11_experiment.py","stage11_adaptive.py","test_stage11_experiment.py",
             "test_stage11_adaptive.py","docs/stage11_protocol_zh.md",
             *oldfreeze["source_sha256"].keys()]
    files += ["results/stage10_pilot/"+name for name in
              ("freeze.json","complete.json","training_data.json","test_data.json","training_complete.json")]
    files += [f"results/stage10_pilot/train_{kind}_{seed}.json" for kind in old.KINDS for seed in old.SEEDS]
    input_hashes = {name:old.sha(ROOT/name) for name in files}
    input_hashes.update({rec["checkpoint"]:rec["checkpoint_sha256"] for rec in records})
    output.mkdir(parents=True)
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(),config=config,smoke=smoke,
        input_sha256=input_hashes,models=loading,methods=METHODS,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            torch=torch.__version__,device="cpu",torch_threads=torch.get_num_threads(),
            platform=platform.platform(),processor=platform.processor(),
            cpu_model=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")),"unknown") if Path("/proc/cpuinfo").exists() else platform.processor(),
            thread_environment={name:os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
            threadpools=threadpool_info()),
        timing_scope="Five fresh warm CPU pipelines per geometry/method; own setup; model loading and imports excluded and reported; no training; references/evaluation/serialization excluded. Fine-direct uses one primal solve for all three QoIs, no coarse setup.")
    old.dump(output/"freeze.json",freeze)
    # All runtime branches are warmed on a separate non-evaluation geometry.
    warm_start=perf_counter()
    for method,seed in METHODS:
        online_pipeline(52950,config["nc"],False,method,models.get((method,seed)),config["new_dofs"])
    warm_seconds=perf_counter()-warm_start
    seen = set()
    for fname in ("training_data.json","test_data.json"):
        obj=json.loads((ROOT/"results/stage10_pilot"/fname).read_text())
        entries=obj if isinstance(obj,list) else obj["train"]+obj["validation"]
        seen.update(x["coefficient_sha256"] for x in entries)
    rows=[];contexts=[];run_start=perf_counter()
    for split,first in config["splits"].items():
        for geometry in range(first,first+config["geometries_per_split"]):
            reference=reference_case(geometry,config["nc"],split=="contrast")
            digest=reference["coefficient_sha256"]
            if digest in seen: raise RuntimeError("Geometry collision with old or new data")
            seen.add(digest)
            if reference["fine_relative_residual"]>1e-8:raise RuntimeError("Invalid fine reference")
            samples={key:[] for key in METHODS}; first_outputs={}
            order_rng=np.random.default_rng(CONFIG["method_order_seed"]+geometry)
            for repeat in range(config["repetitions"]):
                for ix in order_rng.permutation(len(METHODS)):
                    method,seed=METHODS[ix];key=(method,seed)
                    value=online_pipeline(geometry,config["nc"],split=="contrast",method,
                                          models.get(key),config["new_dofs"])
                    samples[key].append(dict(repetition=repeat,**value["timings"]))
                    if repeat==0:
                        first_outputs[key]=evaluate(value,reference)
                    else:
                        oldvalue=first_outputs[key]
                        if value["selected_cells"]!=oldvalue["selected_cells"] or not np.allclose(
                            value["qoi"],oldvalue["qoi"],rtol=1e-12,atol=1e-15):
                            raise RuntimeError("Numerical repeat changed while measuring timing")
            for key in METHODS:
                method,seed=key
                rows.append(dict(split=split,geometry=geometry,training_seed=seed,
                    offline_only=method=="oracle_indicator",**first_outputs[key],
                    timings=samples[key],median_total_seconds=float(np.median([x["total_seconds"] for x in samples[key]]))))
            contexts.append(dict(split=split,geometry=geometry,nc=config["nc"],fine_dofs=(2*config["nc"]-1)**2,
                coarse_dofs=(config["nc"]-1)**2,new_dofs=config["new_dofs"],coefficient_sha256=digest,
                fine_relative_residual=reference["fine_relative_residual"],reference_seconds=reference["seconds"]))
            print(json.dumps(dict(event="refined",split=split,geometry=geometry,contexts=len(contexts),rows=len(rows))),flush=True)
            old.dump(output/"rows.json",rows);old.dump(output/"contexts.json",contexts)
    for name,digest in input_hashes.items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen input changed: {name}")
    old.assert_frozen(oldfreeze,records)
    old.dump(output/"complete.json",dict(complete=True,smoke=smoke,rows=len(rows),contexts=len(contexts),
        fresh_pipeline_runs=len(rows)*config["repetitions"],warmup_seconds=warm_seconds,
        measured_experiment_wall_seconds=perf_counter()-run_start,freeze_sha256=old.sha(output/"freeze.json"),
        output_sha256={name:old.sha(output/name) for name in ("rows.json","contexts.json")}))
    print(json.dumps(dict(event="complete",rows=len(rows),output=str(output))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",default="results/stage11_refinement")
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run((ROOT/args.output).resolve(),args.smoke)


if __name__=="__main__":main()
