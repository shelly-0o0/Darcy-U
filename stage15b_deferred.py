"""Post-hoc engineering variant: localize DWR only when marking is needed.

The immutable Stage-15B loop is the comparison control.  This separate copy of
its DWR path intentionally duplicates its control flow so the frozen experiment
is unchanged.  It is not a new estimator, model, stopping criterion or study.
The global estimate retains the lifted primal and dual defect; continuation
reuses them for strong localization without another matrix-vector product.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

import stage15b_adaptive as base


def global_dwr(p1, p2, u, z2):
    """Return (same global DWR fields, prepared vectors for optional marking)."""
    if p1.coefficient_alignment not in ("element_aligned", "smooth") or p2.coefficient_alignment != p1.coefficient_alignment:
        raise ValueError("matching verified alignment declarations required")
    if not np.array_equal(p1.basis.X, p2.basis.X) or not np.array_equal(p1.basis.W, p2.basis.W):
        raise ValueError("identical P1/P2 quadrature required")
    u, z2 = np.asarray(u, dtype=float), np.asarray(z2, dtype=float)
    if u.shape != (p1.basis.N,) or z2.shape != (p2.basis.N,) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(z2)):
        raise ValueError("invalid primal or dual solution")
    if np.max(abs(u[p1.boundary]), initial=0.) > 1e-13 or np.max(abs(z2[p2.boundary]), initial=0.) > 1e-13:
        raise ValueError("homogeneous boundary values required")
    embedding = base.prolongation_p1_p2(p1, p2)
    lifted = embedding @ u
    nodal = np.zeros(p1.basis.N)
    nodal[p1.basis.nodal_dofs[0]] = z2[p2.basis.nodal_dofs[0]]
    interpolated = embedding @ nodal
    defect = z2-interpolated
    residual = p2.loads[:, 0]-p2.A @ lifted
    result = dict(total_estimate=float(z2 @ residual),
                  discretization_estimate=float(defect @ residual),
                  algebraic_term=float(interpolated @ residual), local=None,
                  no_continuous_error_certificate=True, p2_primal_solved=False)
    if not all(np.isfinite(result[key]) for key in ("total_estimate", "discretization_estimate", "algebraic_term")):
        raise ValueError("nonfinite DWR estimate")
    return result, (lifted, defect)


def localize_dwr(p1, p2, u, z2, *, prepared=None):
    """Use cached vectors; the optional uncached branch supports diagnostics."""
    if prepared is None:
        embedding = base.prolongation_p1_p2(p1, p2)
        lifted = embedding @ u
        nodal = np.zeros(p1.basis.N)
        nodal[p1.basis.nodal_dofs[0]] = z2[p2.basis.nodal_dofs[0]]
        prepared = (lifted, z2-embedding @ nodal)
    lifted, defect = prepared
    strong = base._strong_contributions(p1, p2, lifted, defect[:, None], 0)
    if not strong["available"]:
        raise ValueError("strong localization unavailable: "+strong["reason"])
    return strong["signed_element_contributions"][:, 0]


def run_adaptive(coefficient, source, goal, *, initial_mesh, tau, marker="dwr", config=None,
                 coefficient_alignment="element_aligned", coefficient_grad=None, on_step=None):
    """The frozen DWR policy, with strong localization deferred until marking.

    Resource thresholds and algebraic acceptance match the original loop.
    Timing thresholds remain soft real-time limits, so a path at a time-budget
    boundary may naturally terminate differently when an implementation speeds
    up.  No exact or reference values are accepted by this interface.
    """
    cfg = base._config(config)
    if marker != "dwr":
        raise ValueError("the deferred comparison supports only the DWR marker")
    if not np.isfinite(tau) or tau <= 0.:
        raise ValueError("tau must be positive and finite")
    if not all(callable(f) for f in (coefficient, source, goal)):
        raise ValueError("coefficient/source/goal must be callable")
    if coefficient_alignment not in ("element_aligned", "smooth"):
        raise ValueError("explicit coefficient alignment required")
    if coefficient_alignment == "smooth" and not callable(coefficient_grad):
        raise ValueError("smooth coefficient requires its gradient")
    start = perf_counter(); io_seconds = 0.; rows = []
    mesh = initial_mesh; final_mesh = mesh; final_solution = None
    previous_qoi = None; status = "max_steps"

    def elapsed():
        return perf_counter()-start-io_seconds

    for step in range(cfg["max_steps"]):
        step_start = perf_counter()
        row = dict(step=step, marker=marker, tau=float(tau), verified=False,
                   verification_reasons=[], dwr=None, qoi=None, solves=[], timings={},
                   elements=int(mesh.nelements), vertices=int(mesh.nvertices),
                   mesh_coordinates_sha256=base.array_hash(mesh.p), mesh_connectivity_sha256=base.array_hash(mesh.t),
                   mesh_sort_t=bool(mesh.sort_t), selected_elements=[], scores=None,
                   score_definition=base.SCORE_DEFINITIONS[marker], status="running")
        timings = row["timings"]; next_mesh = None
        try:
            tick = perf_counter()
            p1 = base.assemble(mesh, coefficient, (source, goal), degree=1, intorder=cfg["intorder"],
                               coefficient_alignment=coefficient_alignment, coefficient_grad=coefficient_grad)
            timings["p1_assembly_seconds"] = perf_counter()-tick
            row["p1_free_dofs"] = int(len(p1.free)); row["p1_full_dofs"] = int(p1.basis.N)
            if len(p1.free) > cfg["max_free_dofs"]:
                status = "max_dofs"
            elif elapsed() > cfg["max_seconds"]:
                status = "time_limit"
            else:
                tick = perf_counter()
                solved, metrics = base.solve_columns(p1, (0,), cfg["solver"])
                timings["p1_solve_seconds"] = perf_counter()-tick
                row["solves"].append(dict(role="p1_primal", degree=1, metrics=metrics))
                u = solved[:, 0]
                final_mesh, final_solution = mesh, u
                row["solution_sha256"] = base.array_hash(u)
                row["qoi"] = float(p1.loads[:, 1] @ u)
                if not base.accepted(metrics):
                    status = "linear_solver_failed"
                elif elapsed() > cfg["max_seconds"]:
                    status = "time_limit"
                else:
                    row["verification_reasons"] = base.verification_reasons(step, row["qoi"], previous_qoi, tau, marker, cfg)
                    tick = perf_counter()
                    p2 = base.assemble(mesh, coefficient, (source, goal), degree=2, intorder=cfg["intorder"],
                                       coefficient_alignment=coefficient_alignment, coefficient_grad=coefficient_grad)
                    timings["p2_assembly_seconds"] = perf_counter()-tick
                    row["p2_free_dofs"] = int(len(p2.free)); row["p2_full_dofs"] = int(p2.basis.N)
                    tick = perf_counter()
                    dual, dual_metrics = base.solve_columns(p2, (1,), cfg["solver"])
                    timings["p2_dual_solve_seconds"] = perf_counter()-tick
                    row["solves"].append(dict(role="p2_dual_only", degree=2, metrics=dual_metrics))
                    if not base.accepted(dual_metrics):
                        status = "linear_solver_failed"
                    else:
                        tick = perf_counter()
                        estimate, prepared = global_dwr(p1, p2, u, dual[:, 0])
                        timings["dwr_evaluation_seconds"] = perf_counter()-tick
                        row["verified"] = True
                        row["dwr"] = dict(estimate)
                        row["stop_candidate"] = abs(estimate["total_estimate"]) <= cfg["safety"]*tau
                        if elapsed() > cfg["max_seconds"]:
                            status = "time_limit"
                        elif row["stop_candidate"]:
                            status = "estimated_converged"
                        elif step == cfg["max_steps"]-1:
                            status = "max_steps"
                        else:
                            tick = perf_counter()
                            local = localize_dwr(p1, p2, u, dual[:, 0], prepared=prepared)
                            timings["dwr_localization_seconds"] = perf_counter()-tick
                            row["dwr"]["local"] = local.tolist()
                            tick = perf_counter()
                            scores = abs(local)
                            timings["marker_indicator_seconds"] = perf_counter()-tick
                            tick = perf_counter()
                            selected = base.dorfler(scores, cfg["theta"])
                            row["scores"] = scores.tolist(); row["scores_sha256"] = base.array_hash(scores)
                            row["selected_elements"] = selected.tolist(); row["selected_sha256"] = base.array_hash(selected)
                            timings["marking_seconds"] = perf_counter()-tick
                            if not len(selected):
                                status = "zero_indicator"
                            else:
                                tick = perf_counter()
                                next_mesh = mesh.refined(selected)
                                timings["mesh_refinement_seconds"] = perf_counter()-tick
                                row["next_mesh_coordinates_sha256"] = base.array_hash(next_mesh.p)
                                row["next_mesh_connectivity_sha256"] = base.array_hash(next_mesh.t)
                                next_free = next_mesh.nvertices-len(next_mesh.boundary_nodes())
                                row["next_p1_free_dofs"] = int(next_free)
                                if next_free > cfg["max_free_dofs"]:
                                    status = "max_dofs"
                                elif elapsed() > cfg["max_seconds"]:
                                    status = "time_limit"
                                else:
                                    status = "continue"
        except Exception as exc:
            status = "exception"
            row["exception"] = dict(type=type(exc).__name__, message=str(exc))
        row["status"] = status
        row["algorithm_elapsed_seconds"] = elapsed()
        timings["step_total_seconds"] = perf_counter()-step_start
        timings["unattributed_seconds"] = max(0.,timings["step_total_seconds"]-sum(value for key,value in timings.items() if key != "step_total_seconds"))
        rows.append(row)
        if on_step is not None:
            tick = perf_counter()
            on_step(row,mesh,final_solution if final_mesh is mesh else None)
            io_seconds += perf_counter()-tick
        if status != "continue":
            break
        previous_qoi = row["qoi"]
        mesh = next_mesh
    total = perf_counter()-start
    return base.AdaptiveResult(status,rows,final_mesh,final_solution,total,total-io_seconds,io_seconds,cfg)
