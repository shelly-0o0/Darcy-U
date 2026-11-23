"""Stage-15C classical recovery controls; no learned model or reference input.

This independent loop preserves the immutable Stage-15B implementations.
Recovered adjoints are fixed-work marking vectors, never accepted solutions
or error certificates.  The damping 2/3 and three/eight sweeps are fixed
experimental choices; no contrast- or mesh-independent recovery quality is
claimed.  Only a separately solved and checked P2 adjoint can permit stopping.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

import stage15b_adaptive as base
from stage15b_deferred import global_dwr, localize_dwr


RECOVERY_STEPS = {"recovered_dwr3": 3, "recovered_dwr8": 8}
MARKERS = base.MARKERS + tuple(RECOVERY_STEPS)
OMEGA = 2./3.
SCORE_DEFINITIONS = dict(base.SCORE_DEFINITIONS, **{
    name: f"absolute strong local DWR residual weighted by P2 recovery defect; P1 dual lift plus {steps} Jacobi sweeps, omega=2/3; unverified marker only"
    for name, steps in RECOVERY_STEPS.items()})


def recover_dual(p1, p2, z1, steps):
    """Return a fixed-work P2 marking vector and a charged recovery ledger.

    Starting from E z1, perform exactly ``steps`` updates on free DOFs:
    z <- z + (2/3) diag(A2_ff)^(-1) (g2_f - A2_ff z).
    This performs precisely three or eight matrix-vector products.  It neither
    checks convergence nor performs an extra residual evaluation.  The vector
    is excluded from the loop's accepted linear-solve and RHS counts.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps not in (3, 8):
        raise ValueError("recovery steps must be exactly 3 or 8")
    z1 = np.asarray(z1, dtype=float)
    if z1.shape != (p1.basis.N,) or not np.all(np.isfinite(z1)):
        raise ValueError("invalid P1 adjoint")
    if np.max(abs(z1[p1.boundary]), initial=0.) > 1e-13:
        raise ValueError("homogeneous adjoint boundary values required")
    if p1.loads.shape[1] < 2 or p2.loads.shape[1] < 2:
        raise ValueError("source and goal loads are required")
    if not np.array_equal(p1.basis.X, p2.basis.X) or not np.array_equal(p1.basis.W, p2.basis.W):
        raise ValueError("identical P1/P2 quadrature required")
    start = tick = perf_counter(); timings = {}
    embedding = base.prolongation_p1_p2(p1, p2)
    recovered = embedding @ z1
    timings["embedding_seconds"] = perf_counter()-tick
    tick = perf_counter()
    matrix = p2.A[p2.free][:, p2.free]
    rhs = p2.loads[p2.free, 1]
    interior = recovered[p2.free].copy()
    timings["system_setup_seconds"] = perf_counter()-tick
    tick = perf_counter()
    diagonal = matrix.diagonal()
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.):
        raise ValueError("positive finite P2 diagonal required")
    inverse_diagonal = 1./diagonal
    timings["diagonal_seconds"] = perf_counter()-tick
    timings["matvec_seconds"] = 0.; timings["update_seconds"] = 0.
    for _ in range(int(steps)):
        tick = perf_counter()
        action = matrix @ interior
        timings["matvec_seconds"] += perf_counter()-tick
        tick = perf_counter()
        interior += OMEGA * inverse_diagonal * (rhs-action)
        timings["update_seconds"] += perf_counter()-tick
    tick = perf_counter()
    recovered[p2.free] = interior
    recovered[p2.boundary] = 0.
    if not np.all(np.isfinite(recovered)):
        raise ValueError("nonfinite recovered adjoint")
    boundary_maximum = float(np.max(abs(recovered[p2.boundary]), initial=0.))
    timings["check_seconds"] = perf_counter()-tick
    timings["total_seconds"] = perf_counter()-start
    return recovered, dict(steps=int(steps), omega=OMEGA, matvec_count=int(steps),
                           diagonal_extractions=1, goal_rhs_index=1,
                           unverified_marker=True, converged_solve_claim=False,
                           boundary_maximum=boundary_maximum, timings=timings,
                           source="accepted_P1_adjoint_lift_then_fixed_damped_Jacobi",
                           no_reference_or_true_p2_adjoint_used=True)


def run_adaptive(coefficient, source, goal, *, initial_mesh, tau, marker="dwr", config=None,
                 coefficient_alignment="element_aligned", coefficient_grad=None, on_step=None):
    """Full classical loop with paid stopping and deferred local indicators.

    All methods retain the Stage-15B stopping/resource policy.  Recovery methods
    use the same paid-verification schedule as the cheap classical markers.
    Even on a verified continuation step, recovery determines the marking:
    the true P2 adjoint cannot silently replace the recovered vector.  P2
    assembly is shared within a step when both operations require it.
    """
    cfg = base._config(config)
    if marker not in MARKERS:
        raise ValueError("unknown marker")
    if not np.isfinite(tau) or tau <= 0.:
        raise ValueError("tau must be positive and finite")
    if not all(callable(f) for f in (coefficient, source, goal)):
        raise ValueError("coefficient/source/goal must be callable")
    if coefficient_alignment not in ("element_aligned", "smooth"):
        raise ValueError("explicit coefficient alignment required")
    if coefficient_alignment == "smooth" and not callable(coefficient_grad):
        raise ValueError("smooth coefficient requires its gradient")
    recovering = marker in RECOVERY_STEPS
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
                   score_definition=SCORE_DEFINITIONS[marker], status="running",
                   recovery=None, recovered_local=None, p2_assembly_usage=[])
        timings = row["timings"]; next_mesh = None; p2 = None; prepared = None
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
                cols = (0, 1) if marker == "joint_residual" or recovering else (0,)
                solved, metrics = base.solve_columns(p1, cols, cfg["solver"])
                timings["p1_solve_seconds"] = perf_counter()-tick
                row["solves"].append(dict(role="p1_primal_and_dual" if len(cols) == 2 else "p1_primal", degree=1, metrics=metrics))
                u = solved[:, 0]
                final_mesh, final_solution = mesh, u
                row["solution_sha256"] = base.array_hash(u)
                row["qoi"] = float(p1.loads[:, 1] @ u)
                if not base.accepted(metrics):
                    status = "linear_solver_failed"
                elif elapsed() > cfg["max_seconds"]:
                    status = "time_limit"
                else:
                    reasons = base.verification_reasons(step, row["qoi"], previous_qoi, tau, marker, cfg)
                    row["verification_reasons"] = reasons
                    estimate = None
                    if reasons:
                        tick = perf_counter()
                        p2 = base.assemble(mesh, coefficient, (source, goal), degree=2, intorder=cfg["intorder"],
                                           coefficient_alignment=coefficient_alignment, coefficient_grad=coefficient_grad)
                        timings["p2_assembly_seconds"] = perf_counter()-tick
                        row["p2_assembly_usage"].append("verification")
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
                    if status != "linear_solver_failed":
                        row["stop_candidate"] = bool(row["verified"] and abs(estimate["total_estimate"]) <= cfg["safety"]*tau)
                        if elapsed() > cfg["max_seconds"]:
                            status = "time_limit"
                        elif row["stop_candidate"]:
                            status = "estimated_converged"
                        elif step == cfg["max_steps"]-1:
                            status = "max_steps"
                        else:
                            if recovering:
                                if p2 is None:
                                    tick = perf_counter()
                                    p2 = base.assemble(mesh, coefficient, (source, goal), degree=2, intorder=cfg["intorder"],
                                                       coefficient_alignment=coefficient_alignment, coefficient_grad=coefficient_grad)
                                    timings["p2_assembly_seconds"] = perf_counter()-tick
                                    row["p2_free_dofs"] = int(len(p2.free)); row["p2_full_dofs"] = int(p2.basis.N)
                                row["p2_assembly_usage"].append("recovery")
                                tick = perf_counter()
                                recovered, recovery_metrics = recover_dual(p1, p2, solved[:, 1], RECOVERY_STEPS[marker])
                                timings["recovery_seconds"] = perf_counter()-tick
                                row["recovery"] = recovery_metrics
                                tick = perf_counter()
                                local = localize_dwr(p1, p2, u, recovered)
                                timings["recovery_localization_seconds"] = perf_counter()-tick
                                row["recovered_local"] = local.tolist()
                            elif marker == "dwr":
                                tick = perf_counter()
                                local = localize_dwr(p1, p2, u, dual[:, 0], prepared=prepared)
                                timings["dwr_localization_seconds"] = perf_counter()-tick
                                row["dwr"]["local"] = local.tolist()
                            tick = perf_counter()
                            if marker == "uniform":
                                scores = np.ones(mesh.nelements)
                            elif marker == "coefficient":
                                scores = base.coefficient_scores(p1)
                            elif marker == "primal_residual":
                                indicators = base.residual_scores(p1, u, 0)
                                scores = indicators["scores"]
                                row["residual_components"] = dict(body=indicators["body"].tolist(), face=indicators["face"].tolist())
                            elif marker == "joint_residual":
                                primal = base.residual_scores(p1, u, 0)
                                dual_indicators = base.residual_scores(p1, solved[:, 1], 1)
                                scores = base.joint_scores(primal["scores"], dual_indicators["scores"])
                                row["residual_components"] = dict(primal_body=primal["body"].tolist(), primal_face=primal["face"].tolist(),
                                                                  dual_body=dual_indicators["body"].tolist(), dual_face=dual_indicators["face"].tolist())
                            else:
                                scores = abs(local)
                            timings["marker_indicator_seconds"] = perf_counter()-tick
                            tick = perf_counter()
                            selected = np.arange(mesh.nelements, dtype=int) if marker == "uniform" else base.dorfler(scores, cfg["theta"])
                            row["scores"] = scores.tolist(); row["scores_sha256"] = base.array_hash(scores)
                            row["selected_elements"] = selected.tolist(); row["selected_sha256"] = base.array_hash(selected)
                            timings["marking_seconds"] = perf_counter()-tick
                            if not len(selected):
                                status = "zero_indicator"
                            else:
                                tick = perf_counter()
                                next_mesh = mesh.refined() if marker == "uniform" else mesh.refined(selected)
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
