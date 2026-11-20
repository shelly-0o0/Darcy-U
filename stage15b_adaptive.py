"""Truth-free classical adaptive loops for the stage-15B development audit.

These are empirical stopping algorithms, not certified error bounds.  The
input coefficient must be smooth or piecewise constant and element aligned.
The only enriched solve is a P2 adjoint.  No exact or reference solution is
accepted by this module.  Stage-15A audit kernels remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter

import numpy as np

from stage14_solver import solve_system
from stage15_fem import (assemble, prolongation_p1_p2, _strong_contributions,
                         _scalar_values, _gradient_values)
from skfem import InteriorFacetBasis, ElementTriP1


MARKERS = ("uniform", "coefficient", "primal_residual", "joint_residual", "dwr")
DEFAULT_CONFIG = dict(theta=.5, safety=.5, max_steps=12, max_free_dofs=40000,
                      max_seconds=60., intorder=10, solver="direct",
                      verification_interval=3, verification_delta_factor=.5)
SCORE_DEFINITIONS = dict(
    uniform="all elements are marked for conforming uniform refinement",
    coefficient="h_T^2 * (1 + sum_over_interior_edges(abs(log(k_T/k_neighbor))))",
    primal_residual="h_T^2 integral_T(r_u^2/k) + sum_e 0.5*h_e/k_edge integral_e(jump_u^2); k_edge=max(k_left,k_right)",
    joint_residual="eta_T^2/sum(eta^2) + zeta_T^2/sum(zeta^2), zero terms omitted; normalized combined primal/dual residual marking, not native NGO",
    dwr="absolute signed strong body/face DWR contribution per element")


@dataclass
class AdaptiveResult:
    status: str
    rows: list
    final_mesh: object
    final_solution: np.ndarray | None
    total_seconds: float
    algorithm_seconds: float
    artifact_io_seconds: float
    config: dict

    def summary(self):
        calls = [s for row in self.rows for s in row.get("solves", [])]
        return dict(status=self.status, estimated_converged=self.status == "estimated_converged",
                    solved_states=len(self.rows), final_qoi=self.rows[-1].get("qoi") if self.rows else None,
                    solver_calls=len(calls), rhs_count=sum(s["metrics"]["rhs_count"] for s in calls),
                    peak_p1_free_dofs=max((r.get("p1_free_dofs", 0) for r in self.rows), default=0),
                    peak_p2_free_dofs=max((r.get("p2_free_dofs", 0) for r in self.rows), default=0),
                    total_seconds=self.total_seconds, algorithm_seconds=self.algorithm_seconds,
                    artifact_io_seconds=self.artifact_io_seconds, config=self.config,
                    no_continuous_error_certificate=True, no_reference_used_online=True)


def _config(config):
    result = dict(DEFAULT_CONFIG)
    if config is not None:
        extra = set(config) - set(result)
        if extra:
            raise ValueError("unknown configuration keys: " + ",".join(sorted(extra)))
        result.update(config)
    for name in ("max_steps", "max_free_dofs", "intorder", "verification_interval"):
        if isinstance(result[name], (bool, np.bool_)) or not isinstance(result[name], (int, np.integer)) or result[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if result["intorder"] < 2:
        raise ValueError("intorder must be at least two")
    for name in ("theta", "safety", "max_seconds", "verification_delta_factor"):
        if not np.isfinite(result[name]) or result[name] <= 0.:
            raise ValueError(name + " must be finite and positive")
    if result["theta"] > 1. or result["safety"] > 1.:
        raise ValueError("theta and safety must not exceed one")
    if result["solver"] not in ("direct", "amg"):
        raise ValueError("solver must be direct or amg")
    return result


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def solve_columns(problem, columns, method="direct"):
    """One common factorization/hierarchy for precisely the requested loads."""
    tick = perf_counter()
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns) or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= problem.loads.shape[1] for i in columns):
        raise ValueError("columns must be distinct valid load indices")
    a = problem.A[problem.free][:, problem.free]
    interior, metrics = solve_system(a, problem.loads[problem.free][:, columns], method)
    solutions = np.zeros((problem.basis.N, len(columns)))
    solutions[problem.free] = interior
    metrics["requested_columns"] = list(columns)
    metrics["fem_solve_total_seconds"] = perf_counter() - tick
    metrics["boundary_maximum"] = float(np.max(abs(solutions[problem.boundary]), initial=0.))
    return solutions, metrics


def accepted(metrics):
    return bool(metrics.get("accepted", False) and metrics.get("converged", False)
                and metrics.get("all_info_zero", False)
                and np.isfinite(metrics.get("relative_residual", np.inf))
                and metrics["relative_residual"] <= 1e-9
                and np.isfinite(metrics.get("componentwise_backward_error", np.inf))
                and metrics["componentwise_backward_error"] <= 1e-12
                and metrics.get("boundary_maximum", np.inf) <= 1e-13)


def verification_reasons(step, qoi, previous_qoi, tau, marker, config):
    """Fixed paid-verification schedule; a small QoI change is not acceptance."""
    reasons = []
    if marker == "dwr":
        reasons.append("dwr_marker")
    if step == 0:
        reasons.append("initial")
    elif step % config["verification_interval"] == 0:
        reasons.append("periodic")
    if previous_qoi is not None and abs(qoi - previous_qoi) <= config["verification_delta_factor"] * tau:
        reasons.append("qoi_delta")
    return reasons


def _h_squared(mesh):
    points = mesh.p[:, mesh.t]
    return np.max(np.stack([np.sum((points[:, i] - points[:, (i+1) % 3])**2, axis=0)
                            for i in range(3)]), axis=0)


def _element_data(problem):
    x = np.asarray(problem.basis.global_coordinates())
    k = _scalar_values(problem.coefficient, x, "coefficient")
    if np.any(k <= 0.):
        raise ValueError("coefficient must be positive")
    if problem.coefficient_alignment == "element_aligned":
        if np.any(np.ptp(k, axis=1) > 1e-12 * np.maximum(np.max(abs(k), axis=1), 1.)):
            raise ValueError("element_aligned requires elementwise constant material")
        gradk = np.zeros_like(x)
    elif problem.coefficient_alignment == "smooth" and problem.coefficient_grad is not None:
        gradk = _gradient_values(problem.coefficient_grad, x, "coefficient_grad")
    else:
        raise ValueError("explicit aligned constants or smooth coefficient with gradient required")
    return x, k, gradk


def residual_scores(problem, solution, rhs_index=0):
    """Squared classical residual indicators, without a robustness theorem.

    For P1, r=f+grad(k).grad(u).  An interior facet contribution is divided
    equally between its neighbors and uses max(k_left,k_right).  Boundary
    facets do not contribute for homogeneous Dirichlet data.  Data oscillation
    is not separately estimated, so these are not equilibrated-flux bounds.
    """
    if problem.degree != 1:
        raise ValueError("P1 residuals required")
    u = np.asarray(solution, dtype=float)
    if u.shape != (problem.basis.N,) or not np.all(np.isfinite(u)):
        raise ValueError("invalid solution")
    if not isinstance(rhs_index, (int, np.integer)) or not 0 <= rhs_index < len(problem.rhs_functions):
        raise ValueError("invalid rhs index")
    x, k, gradk = _element_data(problem)
    r = _scalar_values(problem.rhs_functions[rhs_index], x, "rhs") + np.sum(gradk * problem.basis.interpolate(u).grad, axis=0)
    body = _h_squared(problem.mesh) * np.sum(r*r / k * problem.basis.dx, axis=1)
    face = np.zeros(problem.mesh.nelements)
    interior = np.flatnonzero(problem.mesh.f2t[1] >= 0)
    fb0 = InteriorFacetBasis(problem.mesh, ElementTriP1(), facets=interior, side=0, intorder=problem.intorder)
    fb1 = InteriorFacetBasis(problem.mesh, ElementTriP1(), facets=interior, side=1, intorder=problem.intorder)
    if problem.coefficient_alignment == "element_aligned":
        k0 = k[problem.mesh.f2t[0, interior], :1]
        k1 = k[problem.mesh.f2t[1, interior], :1]
    else:
        k0 = _scalar_values(problem.coefficient, np.asarray(fb0.global_coordinates()), "facet coefficient")
        k1 = _scalar_values(problem.coefficient, np.asarray(fb1.global_coordinates()), "facet coefficient")
    jump = np.sum((k0[None] * fb0.interpolate(u).grad - k1[None] * fb1.interpolate(u).grad) * fb0.normals, axis=0)
    facet_points = problem.mesh.p[:, problem.mesh.facets[:, interior]]
    h_edge = np.linalg.norm(facet_points[:, 0] - facet_points[:, 1], axis=0)
    edge = h_edge * np.sum(jump*jump / np.maximum(k0, k1) * fb0.dx, axis=1)
    for side in (0, 1):
        np.add.at(face, problem.mesh.f2t[side, interior], .5 * edge)
    scores = body + face
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.):
        raise ValueError("nonfinite or negative residual score")
    return dict(scores=scores, body=body, face=face, interior_facet_ids=interior,
                facet_scores=edge, definition=SCORE_DEFINITIONS["primal_residual"],
                no_contrast_robust_bound_claim=True)


def coefficient_scores(problem):
    x, k, _ = _element_data(problem)
    # For smooth coefficients use the quadrature-weighted cell mean.  In the
    # aligned setting the mean is exactly the element material value.
    mean = np.sum(k * problem.basis.dx, axis=1) / np.sum(problem.basis.dx, axis=1)
    contribution = np.ones(problem.mesh.nelements)
    interior = np.flatnonzero(problem.mesh.f2t[1] >= 0)
    sides = problem.mesh.f2t[:, interior]
    difference = abs(np.log(mean[sides[0]]) - np.log(mean[sides[1]]))
    for side in (0, 1):
        np.add.at(contribution, sides[side], difference)
    return _h_squared(problem.mesh) * contribution


def joint_scores(primal, dual):
    """Normalized combined residual marker; no NGO/certification claim."""
    p, z = np.asarray(primal, dtype=float), np.asarray(dual, dtype=float)
    if p.shape != z.shape or p.ndim != 1 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(z)) or np.any(p < 0.) or np.any(z < 0.):
        raise ValueError("invalid primal/dual indicator arrays")
    return (p / p.sum() if p.sum() > 0. else np.zeros_like(p)) + (z / z.sum() if z.sum() > 0. else np.zeros_like(z))


def dorfler(scores, theta=.5):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)) or np.any(scores < 0.):
        raise ValueError("scores must be a nonempty finite nonnegative vector")
    if not np.isfinite(theta) or not 0. < theta <= 1.:
        raise ValueError("theta must lie in (0,1]")
    if scores.sum() <= 0.:
        return np.array([], dtype=int)
    order = np.lexsort((np.arange(len(scores)), -scores))
    count = min(int(np.searchsorted(np.cumsum(scores[order]), theta*scores.sum(), side="left"))+1, len(scores))
    return order[:count]


def online_dwr(p1, p2, u, z2, *, localize=False):
    """One residual product and optional strong localization; no P2 primal.

    The code does not form E.T A2 E or solve an audit reference.  P1/P2 are
    assembled by the same loop from the same input functions and quadrature.
    The algebraic term R(Iz2) is retained even for an imperfect P1 solve.
    """
    if p1.coefficient_alignment not in ("element_aligned", "smooth") or p2.coefficient_alignment != p1.coefficient_alignment:
        raise ValueError("matching verified alignment declarations required")
    if not np.array_equal(p1.basis.X, p2.basis.X) or not np.array_equal(p1.basis.W, p2.basis.W):
        raise ValueError("identical P1/P2 quadrature required")
    u, z2 = np.asarray(u, dtype=float), np.asarray(z2, dtype=float)
    if u.shape != (p1.basis.N,) or z2.shape != (p2.basis.N,) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(z2)):
        raise ValueError("invalid primal or dual solution")
    if np.max(abs(u[p1.boundary]), initial=0.) > 1e-13 or np.max(abs(z2[p2.boundary]), initial=0.) > 1e-13:
        raise ValueError("homogeneous boundary values required")
    E = prolongation_p1_p2(p1, p2)
    lifted = E @ u
    nodal = np.zeros(p1.basis.N)
    nodal[p1.basis.nodal_dofs[0]] = z2[p2.basis.nodal_dofs[0]]
    interpolated = E @ nodal
    defect = z2 - interpolated
    residual = p2.loads[:, 0] - p2.A @ lifted
    result = dict(total_estimate=float(z2 @ residual),
                  discretization_estimate=float(defect @ residual),
                  algebraic_term=float(interpolated @ residual),
                  local=None, no_continuous_error_certificate=True,
                  p2_primal_solved=False)
    if localize:
        strong = _strong_contributions(p1, p2, lifted, defect[:, None], 0)
        if not strong["available"]:
            raise ValueError("strong localization unavailable: " + strong["reason"])
        result["local"] = strong["signed_element_contributions"][:, 0]
    if not all(np.isfinite(result[key]) for key in ("total_estimate", "discretization_estimate", "algebraic_term")):
        raise ValueError("nonfinite DWR estimate")
    return result


def run_adaptive(coefficient, source, goal, *, initial_mesh, tau, marker="dwr", config=None,
                 coefficient_alignment="element_aligned", coefficient_grad=None, on_step=None):
    """Run one deterministic adaptive path with only paid online acceptance.

    A callback receives (row, current_mesh, current_P1_solution) after the
    row is complete. Its wall time is excluded from algorithm_seconds and
    resource limits, but included in total_seconds/artifact_io_seconds.
    Failed solves, exhausted resources and exceptions remain explicit rows.
    max_steps counts solved mesh states, including the initial state.
    """
    cfg = _config(config)
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
    start = perf_counter(); io_seconds = 0.; rows = []
    mesh = initial_mesh; final_mesh = mesh; final_solution = None
    previous_qoi = None; status = "max_steps"

    def elapsed():
        return perf_counter() - start - io_seconds

    for step in range(cfg["max_steps"]):
        step_start = perf_counter()
        row = dict(step=step, marker=marker, tau=float(tau), verified=False,
                   verification_reasons=[], dwr=None, qoi=None, solves=[], timings={},
                   elements=int(mesh.nelements), vertices=int(mesh.nvertices),
                   mesh_coordinates_sha256=array_hash(mesh.p), mesh_connectivity_sha256=array_hash(mesh.t),
                   mesh_sort_t=bool(mesh.sort_t), selected_elements=[], scores=None,
                   score_definition=SCORE_DEFINITIONS[marker], status="running")
        timings = row["timings"]
        next_mesh = None
        try:
            tick = perf_counter()
            p1 = assemble(mesh, coefficient, (source, goal), degree=1, intorder=cfg["intorder"],
                          coefficient_alignment=coefficient_alignment, coefficient_grad=coefficient_grad)
            timings["p1_assembly_seconds"] = perf_counter() - tick
            row["p1_free_dofs"] = int(len(p1.free)); row["p1_full_dofs"] = int(p1.basis.N)
            if len(p1.free) > cfg["max_free_dofs"]:
                status = "max_dofs"
            elif elapsed() > cfg["max_seconds"]:
                status = "time_limit"
            else:
                tick = perf_counter()
                cols = (0, 1) if marker == "joint_residual" else (0,)
                solved, metrics = solve_columns(p1, cols, cfg["solver"])
                timings["p1_solve_seconds"] = perf_counter() - tick
                row["solves"].append(dict(role="p1_primal_and_dual" if len(cols) == 2 else "p1_primal", degree=1, metrics=metrics))
                u = solved[:, 0]
                final_mesh, final_solution = mesh, u
                row["solution_sha256"] = array_hash(u)
                row["qoi"] = float(p1.loads[:, 1] @ u)
                if not accepted(metrics):
                    status = "linear_solver_failed"
                elif elapsed() > cfg["max_seconds"]:
                    status = "time_limit"
                else:
                    reasons = verification_reasons(step, row["qoi"], previous_qoi, tau, marker, cfg)
                    row["verification_reasons"] = reasons
                    estimate = None
                    if reasons:
                        tick = perf_counter()
                        p2 = assemble(mesh, coefficient, (source, goal), degree=2, intorder=cfg["intorder"],
                                      coefficient_alignment=coefficient_alignment, coefficient_grad=coefficient_grad)
                        timings["p2_assembly_seconds"] = perf_counter() - tick
                        row["p2_free_dofs"] = int(len(p2.free)); row["p2_full_dofs"] = int(p2.basis.N)
                        tick = perf_counter()
                        dual, dual_metrics = solve_columns(p2, (1,), cfg["solver"])
                        timings["p2_dual_solve_seconds"] = perf_counter() - tick
                        row["solves"].append(dict(role="p2_dual_only", degree=2, metrics=dual_metrics))
                        if not accepted(dual_metrics):
                            status = "linear_solver_failed"
                        else:
                            tick = perf_counter()
                            estimate = online_dwr(p1, p2, u, dual[:, 0], localize=marker == "dwr")
                            timings["dwr_evaluation_seconds"] = perf_counter() - tick
                            row["verified"] = True
                            row["dwr"] = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in estimate.items()}
                    if status != "linear_solver_failed":
                        row["stop_candidate"] = bool(row["verified"] and abs(estimate["total_estimate"]) <= cfg["safety"]*tau)
                        if elapsed() > cfg["max_seconds"]:
                            status = "time_limit"
                        elif row["stop_candidate"]:
                            status = "estimated_converged"
                        elif step == cfg["max_steps"] - 1:
                            status = "max_steps"
                        else:
                            tick = perf_counter()
                            if marker == "uniform":
                                scores = np.ones(mesh.nelements)
                            elif marker == "coefficient":
                                scores = coefficient_scores(p1)
                            elif marker == "primal_residual":
                                indicators = residual_scores(p1, u, 0)
                                scores = indicators["scores"]
                                row["residual_components"] = dict(body=indicators["body"].tolist(), face=indicators["face"].tolist())
                            elif marker == "joint_residual":
                                primal = residual_scores(p1, u, 0)
                                dual_indicators = residual_scores(p1, solved[:, 1], 1)
                                scores = joint_scores(primal["scores"], dual_indicators["scores"])
                                row["residual_components"] = dict(primal_body=primal["body"].tolist(), primal_face=primal["face"].tolist(),
                                                                  dual_body=dual_indicators["body"].tolist(), dual_face=dual_indicators["face"].tolist())
                            else:
                                scores = abs(estimate["local"])
                            timings["marker_indicator_seconds"] = perf_counter() - tick
                            tick = perf_counter()
                            selected = np.arange(mesh.nelements, dtype=int) if marker == "uniform" else dorfler(scores, cfg["theta"])
                            row["scores"] = scores.tolist(); row["scores_sha256"] = array_hash(scores)
                            row["selected_elements"] = selected.tolist(); row["selected_sha256"] = array_hash(selected)
                            timings["marking_seconds"] = perf_counter() - tick
                            if not len(selected):
                                status = "zero_indicator"
                            else:
                                tick = perf_counter()
                                next_mesh = mesh.refined() if marker == "uniform" else mesh.refined(selected)
                                timings["mesh_refinement_seconds"] = perf_counter() - tick
                                row["next_mesh_coordinates_sha256"] = array_hash(next_mesh.p)
                                row["next_mesh_connectivity_sha256"] = array_hash(next_mesh.t)
                                next_free = next_mesh.nvertices - len(next_mesh.boundary_nodes())
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
        timings["step_total_seconds"] = perf_counter() - step_start
        timings["unattributed_seconds"] = max(0., timings["step_total_seconds"] - sum(v for k, v in timings.items() if k != "step_total_seconds"))
        rows.append(row)
        if on_step is not None:
            tick = perf_counter()
            on_step(row, mesh, final_solution if final_mesh is mesh else None)
            io_seconds += perf_counter() - tick
        if status != "continue":
            break
        previous_qoi = row["qoi"]
        mesh = next_mesh
    total = perf_counter() - start
    return AdaptiveResult(status, rows, final_mesh, final_solution, total,
                          total - io_seconds, io_seconds, cfg)
