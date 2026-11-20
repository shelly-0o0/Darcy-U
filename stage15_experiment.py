"""Execute the stage-15A manufactured-solution audit; never trains a model.

The deterministic matrix is a verification suite, not a statistical sample.
Analytic solutions appear only in offline error evaluation.  Local refinement
uses charged enriched DWR with fixed rounds, never the true error or true goal.
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
from threadpoolctl import threadpool_info

import stage15_fem as fem
from stage15_cases import verification_cases
import skfem

ROOT = Path(__file__).resolve().parent
CONFIG = dict(uniform_n=[8, 16, 32, 64, 128], intorder=10,
              quadrature_check_order=18, amg_n=64, adaptive_n=8,
              adaptive_refinements=3, marking_theta=.5,
              marking="sum_absolute_body_face_dwr_over_goals_stable_dorfler",
              tau_scale=1e-5, analytic_goal_orders=[64, 96],
              statistical_benchmark=False, train_models=0)
SOURCE_FILES = ["stage15_cases.py", "stage15_fem.py", "stage15_experiment.py",
                "report_stage15.py", "test_stage15_numerics.py", "stage14_solver.py",
                "sequence_baselines.py", "docs/stage15_numerical_protocol_zh.md",
                "requirements.txt"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n")


def mesh_check(mesh, case):
    verts = mesh.p[:, mesh.t]
    ab, ac = verts[:, 1]-verts[:, 0], verts[:, 2]-verts[:, 0]
    areas = abs(ab[0]*ac[1]-ab[1]*ac[0])*.5
    cuts = sum(int(np.sum((verts[0].min(axis=0) < a-1e-13) &
                         (verts[0].max(axis=0) > a+1e-13))) for a in case.interfaces)
    boundary = mesh.p[:, mesh.boundary_nodes()]
    on_square = np.any((abs(boundary) < 1e-13) | (abs(boundary-1) < 1e-13), axis=0)
    return dict(area=float(areas.sum()), min_area=float(areas.min()),
                material_crossing_elements=cuts,
                square_boundary=bool(np.all(on_square)),
                accepted=bool(cuts == 0 and np.all(areas > 0) and
                              abs(areas.sum()-1.) < 1e-12 and np.all(on_square)))


def build(case, mesh, degree, order):
    return fem.assemble(mesh, case.coefficient, [case.source, *case.goals],
                        degree=degree, intorder=order,
                        coefficient_alignment=case.metadata["coefficient_alignment"],
                        coefficient_grad=case.coefficient_grad)


def _solution_record(case, problem, solutions, metrics):
    q = problem.loads[:, 1:].T @ solutions[:, 0]
    error = case.exact_qoi-q
    return dict(degree=problem.degree, full_dofs=problem.basis.N,
                free_dofs=len(problem.free), qoi=q, signed_exact_qoi_error=error,
                q_ref=float(np.max(abs(error))/(CONFIG["tau_scale"]*case.metadata["scale_from_input"])),
                error_norms=fem.error_norms(problem, solutions[:, 0], case.exact, case.exact_grad),
                solver=metrics, assembly=problem.timings,
                solution_sha256=array_sha(solutions),
                matrix_data_sha256=array_sha(problem.A.data), load_sha256=array_sha(problem.loads))


def context(case, mesh, kind, key, *, with_amg=False):
    start = perf_counter()
    check = mesh_check(mesh, case)
    if not check["accepted"]:
        raise ValueError("Invalid or material-unaligned mesh")
    p1 = build(case, mesh, 1, CONFIG["intorder"])
    p2 = build(case, mesh, 2, CONFIG["intorder"])
    u1, m1 = fem.solve(p1)
    u2, m2 = fem.solve(p2)
    estimate = fem.dwr(p1, p2, u1[:, 0], u2[:, 1:], p2_primal=u2[:, 0])
    local = estimate.pop("signed_element_contributions")
    strong = estimate["strong"]
    if not strong["available"]:
        raise RuntimeError("Predefined body/face localization is unavailable")
    strong_local = strong.pop("signed_element_contributions")
    strong.pop("signed_interior_face_jumps")
    strong.pop("interior_facet_ids")
    estimate["local_contributions_sha256"] = array_sha(local)
    strong["local_contributions_sha256"] = array_sha(strong_local)
    r1 = _solution_record(case, p1, u1, m1)
    r2 = _solution_record(case, p2, u2, m2)
    true_error = np.array(r1["signed_exact_qoi_error"])
    resolved = abs(true_error) > 100*np.finfo(float).eps*case.metadata["scale_from_input"]
    effectivity = np.divide(estimate["total_estimates"], true_error,
                           out=np.zeros(3), where=resolved)
    row = dict(case=case.name, kind=kind, key=key, elements=mesh.nelements,
               mesh_check=check, mesh_coordinates_sha256=array_sha(mesh.p),
               mesh_connectivity_sha256=array_sha(mesh.t),
               p1=r1, p2=r2, dwr=estimate,
               effectivity=[float(effectivity[j]) if resolved[j] else None for j in range(3)],
               scale_from_input=case.metadata["scale_from_input"], exact_qoi=case.exact_qoi,
               analytic_goals_64_96_max_difference=float(np.max(abs(case.exact_qoi-case.analytic_goals(96)))))
    if with_amg:
        row["amg"] = []
        for problem, reference in ((p1, u1), (p2, u2)):
            ua, ma = fem.solve(problem, "amg")
            row["amg"].append(dict(degree=problem.degree, solver=ma,
                max_qoi_difference=float(np.max(abs(problem.loads[:, 1:].T@(ua[:, 0]-reference[:, 0])))),
                relative_solution_difference=float(np.linalg.norm(ua-reference)/np.linalg.norm(reference))))
    row["context_audit_wall_seconds"] = perf_counter()-start
    return row, strong_local, (p1, p2, u1, u2)


def mark(local, theta):
    if not np.isfinite(theta) or not 0. < theta <= 1.:
        raise ValueError("theta must lie in (0,1]")
    scores = np.sum(abs(np.asarray(local)), axis=1)
    total = float(scores.sum())
    if not np.all(np.isfinite(scores)) or total <= 0.:
        raise ValueError("Nonpositive or nonfinite marking mass")
    order = np.lexsort((np.arange(len(scores)), -scores))
    count = int(np.searchsorted(np.cumsum(scores[order]), theta*total, side="left"))+1
    selected = np.sort(order[:count])
    return selected, dict(theta=theta, selected_elements=len(selected),
                          captured_fraction=float(scores[selected].sum()/total),
                          scores=scores, selected_ids=selected,
                          selected_sha256=array_sha(selected), true_solution_not_used=True)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new directory; audit records cannot be overwritten")
    pools = threadpool_info()
    if any(p["num_threads"] != 1 for p in pools):
        raise RuntimeError("Single-thread numerical libraries are required")
    if skfem.__version__ != "12.0.2":
        raise RuntimeError("This audit is pinned to scikit-fem 12.0.2")
    cases = verification_cases()
    output.mkdir(parents=True)
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                  cases=[dict(name=c.name, metadata=c.metadata, exact_qoi=c.exact_qoi) for c in cases],
                  source_sha256={name: sha(ROOT/name) for name in SOURCE_FILES},
                  library_sha256={name: sha(ROOT/"vendor"/name) for name in (
                      "skfem/mesh/mesh_tri_1.py", "skfem/element/element_tri/element_tri_p1.py",
                      "skfem/element/element_tri/element_tri_p2.py", "skfem/quadrature.py")},
                  environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                   skfem=skfem.__version__,platform=platform.platform(),threadpools=pools,
                                   thread_environment={n: os.environ.get(n) for n in
                                       ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
                  scope="manufactured aligned/smooth numerical verification; G1/G2/G3 not evaluated")
    dump(output/"freeze.json", freeze)
    rows, quadrature_rows = [], []
    all_start = perf_counter()
    try:
        for case in cases:
            for n in CONFIG["uniform_n"]:
                row, _, result = context(case, fem.make_mesh(n), "uniform", n,
                                         with_amg=n == CONFIG["amg_n"])
                rows.append(row)
                if n == max(CONFIG["uniform_n"]):
                    mesh = result[0].mesh
                    high = build(case, mesh, 2, CONFIG["quadrature_check_order"])
                    uh, mh = fem.solve(high)
                    qh = high.loads[:, 1:].T@uh[:, 0]
                    quadrature_rows.append(dict(case=case.name,n=n,
                        low_order=CONFIG["intorder"],high_order=high.intorder,
                        qoi_low=row["p2"]["qoi"],qoi_high=qh,
                        max_qoi_difference=float(np.max(abs(qh-row["p2"]["qoi"]))),
                        source_load_relative_difference=float(np.linalg.norm(high.loads[:,0]-result[1].loads[:,0])/
                                                             np.linalg.norm(high.loads[:,0])),
                        goal_load_relative_difference=float(np.linalg.norm(high.loads[:,1:]-result[1].loads[:,1:])/
                                                           np.linalg.norm(high.loads[:,1:])),
                        solver=mh,assembly=high.timings))
                dump(output/"rows.json", rows)
                dump(output/"quadrature.json", quadrature_rows)
                print(json.dumps(dict(event="uniform",case=case.name,n=n,p1_l2=row["p1"]["error_norms"]["l2"],
                                      p2_l2=row["p2"]["error_norms"]["l2"])),flush=True)
                del result
            mesh = fem.make_mesh(CONFIG["adaptive_n"])
            for step in range(CONFIG["adaptive_refinements"]+1):
                row, local, result = context(case, mesh, "adaptive", step)
                row["mesh_coordinates"] = mesh.p
                row["mesh_connectivity"] = mesh.t
                row["mesh_sort_t"] = bool(mesh.sort_t)
                row["strong_element_contributions"] = local
                if step < CONFIG["adaptive_refinements"]:
                    marked, marking = mark(local, CONFIG["marking_theta"])
                    tick = perf_counter()
                    mesh = mesh.refined(marked)
                    marking["refinement_seconds"] = perf_counter()-tick
                    row["marking"] = marking
                rows.append(row)
                dump(output/"rows.json", rows)
                print(json.dumps(dict(event="adaptive",case=case.name,step=step,elements=row["elements"])),flush=True)
                del result
        for name, digest in freeze["source_sha256"].items():
            if sha(ROOT/name) != digest:
                raise RuntimeError("Source changed during audit: "+name)
        dump(output/"complete.json", dict(complete=True,wall_seconds=perf_counter()-all_start,
             rows=len(rows),quadrature_rows=len(quadrature_rows),
             file_sha256={name:sha(output/name) for name in ("freeze.json","rows.json","quadrature.json")},
             no_training=True,G1="not_evaluated",G2="not_evaluated",G3="not_evaluated"))
    except Exception as exc:
        dump(output/"rows.json", rows)
        dump(output/"quadrature.json", quadrature_rows)
        dump(output/"failure.json",dict(type=type(exc).__name__,message=str(exc),
                                       rows_completed=len(rows),wall_seconds=perf_counter()-all_start))
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/stage15_numerical")
    args=parser.parse_args()
    run(ROOT/args.output)
