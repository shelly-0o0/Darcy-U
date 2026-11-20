"""Conforming P1/P2 kernels for the stage-15 numerical qualification audit.

Only homogeneous Dirichlet problems on triangular meshes are solved.  The
coefficient-alignment string is an explicit caller declaration, not a theorem
inferred from a handful of quadrature samples.  Unverified coefficients may be
assembled for diagnostics but are rejected by the DWR audit.  Neither a DWR
identity nor its element decomposition is a reliable error upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.sparse as sp

import sequence_baselines  # activates the repository-local vendor directory
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, MeshTri, asm
from skfem import InteriorFacetBasis, LinearForm
from skfem.helpers import dot, grad
from stage14_solver import ResidualEvaluator, solve_system


@dataclass
class FEMProblem:
    mesh: MeshTri
    basis: Basis
    A: sp.csr_matrix
    loads: np.ndarray
    free: np.ndarray
    boundary: np.ndarray
    degree: int
    intorder: int
    coefficient: Callable
    rhs_functions: tuple
    coefficient_alignment: str
    coefficient_grad: Callable | None = None
    timings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def make_mesh(n):
    """Return a conforming tensor-generated unit-square triangular mesh."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    axis = np.linspace(0., 1., int(n) + 1)
    return MeshTri.init_tensor(axis, axis)


def _scalar_values(function, x, name):
    values = np.asarray(function(x[0], x[1]), dtype=float)
    try:
        values = np.broadcast_to(values, x.shape[1:])
    except ValueError as exc:
        raise ValueError(f"{name} does not broadcast to quadrature coordinates") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")
    return values


def _gradient_values(function, x, name):
    values = np.asarray(function(x[0], x[1]), dtype=float)
    if values.shape == (2,):
        values = values.reshape((2,) + (1,) * (x.ndim - 1))
    try:
        values = np.broadcast_to(values, x.shape)
    except ValueError as exc:
        raise ValueError(f"{name} must return a leading dimension of length 2") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")
    return values


def _vector(values, n, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite vector with shape ({n},)")
    return values


def assemble(mesh, coefficient, rhs_functions, degree=1, intorder=10, *,
             coefficient_alignment="unverified", coefficient_grad=None):
    """Assemble full matrices; load columns use the same quadrature as A.

    Alignment declarations: ``smooth`` or ``element_aligned`` permit the DWR
    audit. ``unverified``/``unaligned`` remain explicitly labelled and are not
    accepted by that audit.  The caller must ensure that material interfaces
    do not cut cells when declaring ``element_aligned``.
    """
    start = perf_counter()
    if degree not in (1, 2) or isinstance(degree, (bool, np.bool_)):
        raise ValueError("only degree 1 or 2 is supported")
    if not isinstance(intorder, (int, np.integer)) or intorder < 2:
        raise ValueError("intorder must be an integer >= 2")
    if coefficient_alignment not in ("smooth", "element_aligned", "unverified", "unaligned"):
        raise ValueError("unknown coefficient_alignment declaration")
    funcs = tuple(rhs_functions)
    if not funcs or not callable(coefficient) or any(not callable(f) for f in funcs):
        raise ValueError("coefficient and a nonempty list of loads must be callable")
    if coefficient_grad is not None and not callable(coefficient_grad):
        raise ValueError("coefficient_grad must be callable or None")
    if not isinstance(mesh, MeshTri) or mesh.p.shape[0] != 2:
        raise ValueError("a two-dimensional MeshTri is required")
    if not np.all(np.isfinite(mesh.p)):
        raise ValueError("mesh coordinates must be finite")
    basis = Basis(mesh, ElementTriP1() if degree == 1 else ElementTriP2(), intorder=intorder)
    coordinates = np.asarray(basis.global_coordinates())
    kvals = _scalar_values(coefficient, coordinates, "coefficient")
    if np.any(kvals <= 0.):
        raise ValueError("coefficient must be strictly positive at quadrature points")

    @BilinearForm
    def stiffness(u, v, w):
        return w.k * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        return w.rhs * v

    A = sp.csr_matrix(asm(stiffness, basis, k=kvals))
    loads = np.column_stack([asm(load, basis, rhs=_scalar_values(f, coordinates, "load"))
                             for f in funcs])
    boundary = basis.get_dofs().all()
    free = np.setdiff1d(np.arange(basis.N), boundary)
    if not free.size:
        raise ValueError("mesh has no free degrees of freedom")
    return FEMProblem(mesh, basis, A, loads, free, boundary, int(degree), int(intorder),
        coefficient, funcs, coefficient_alignment, coefficient_grad,
        timings={"assembly_seconds": perf_counter() - start},
        metadata={"coefficient_alignment": coefficient_alignment,
                  "alignment_is_caller_declaration": True,
                  "coefficient_positive_at_quadrature": True,
                  "homogeneous_dirichlet": True, "elements": int(mesh.nelements),
                  "full_dofs": int(basis.N), "free_dofs": int(len(free)),
                  "no_continuous_error_certificate": True})


def solve(problem, method="direct"):
    """Return (full nodal solution columns, common stage-14 solver metrics)."""
    start = perf_counter()
    A = problem.A[problem.free][:, problem.free]
    interior, metrics = solve_system(A, problem.loads[problem.free], method)
    solutions = np.zeros_like(problem.loads)
    solutions[problem.free] = interior
    metrics["fem_solve_total_seconds"] = perf_counter() - start
    metrics["boundary_maximum"] = float(np.max(abs(solutions[problem.boundary]), initial=0.))
    return solutions, metrics


def _same_mesh(p1, p2):
    return np.array_equal(p1.mesh.p, p2.mesh.p) and np.array_equal(p1.mesh.t, p2.mesh.t)


def prolongation_p1_p2(coarse, enriched):
    """Exact same-mesh nodal embedding: vertices and averages on edge DOFs."""
    if coarse.degree != 1 or enriched.degree != 2 or not _same_mesh(coarse, enriched):
        raise ValueError("prolongation requires P1/P2 on the identical mesh")
    v1 = coarse.basis.nodal_dofs[0]
    v2 = enriched.basis.nodal_dofs[0]
    edge2 = enriched.basis.facet_dofs[0]
    facets = enriched.mesh.facets
    rows = np.concatenate((v2, edge2, edge2))
    cols = np.concatenate((v1, v1[facets[0]], v1[facets[1]]))
    data = np.concatenate((np.ones(len(v2)), np.full(2 * len(edge2), .5)))
    return sp.csr_matrix((data, (rows, cols)), shape=(enriched.basis.N, coarse.basis.N))


def error_norms(problem, u, exact, exact_grad, coefficient=None):
    """Quadrature estimates of L2, H1-seminorm and coefficient energy error."""
    u = _vector(u, problem.basis.N, "u")
    x = np.asarray(problem.basis.global_coordinates())
    field = problem.basis.interpolate(u)
    difference = np.asarray(field) - _scalar_values(exact, x, "exact")
    gradient = np.asarray(field.grad) - _gradient_values(exact_grad, x, "exact_grad")
    k = _scalar_values(problem.coefficient if coefficient is None else coefficient, x, "coefficient")
    if np.any(k <= 0.):
        raise ValueError("energy coefficient must be positive")
    l2_squared = float(np.sum(problem.basis.dx * difference ** 2))
    h1_squared = float(np.sum(problem.basis.dx * np.sum(gradient ** 2, axis=0)))
    energy_squared = float(np.sum(problem.basis.dx * k * np.sum(gradient ** 2, axis=0)))
    return dict(l2=float(np.sqrt(max(l2_squared, 0.))),
                h1_seminorm=float(np.sqrt(max(h1_squared, 0.))),
                energy=float(np.sqrt(max(energy_squared, 0.))),
                l2_squared=l2_squared, h1_seminorm_squared=h1_squared,
                energy_squared=energy_squared, quadrature_order=problem.intorder)


def _strong_contributions(p1, p2, u_lift, defects, source_index):
    """Optional body/face decomposition, with explicit one-sided k traces.

    Smooth k requires its gradient. For element_aligned k this implementation
    only supports elementwise constants; quadrature constancy is a diagnostic
    check and the caller's alignment declaration remains essential. Boundary
    terms vanish because each defect has homogeneous boundary values.
    """
    x = np.asarray(p2.basis.global_coordinates())
    k = _scalar_values(p2.coefficient, x, "coefficient")
    if p2.coefficient_alignment == "smooth":
        if p2.coefficient_grad is None:
            return {"available": False, "reason": "smooth_coefficient_gradient_missing"}
        gradk = _gradient_values(p2.coefficient_grad, x, "coefficient_grad")
        element_k = None
    else:
        variation = np.ptp(k, axis=1)
        if np.any(variation > 1e-12 * np.maximum(np.max(abs(k), axis=1), 1.)):
            return {"available": False, "reason": "aligned_nonconstant_trace_rule_not_implemented"}
        gradk = np.zeros_like(x)
        element_k = k[:, 0]
    uh = p2.basis.interpolate(u_lift)
    body = _scalar_values(p2.rhs_functions[source_index], x, "source") + np.sum(gradk * uh.grad, axis=0)
    local = np.column_stack([np.sum(body * np.asarray(p2.basis.interpolate(defects[:, j])) * p2.basis.dx,
                                    axis=1) for j in range(defects.shape[1])])
    interior = np.flatnonzero(p2.mesh.f2t[1] >= 0)
    fb0 = InteriorFacetBasis(p2.mesh, ElementTriP2(), facets=interior, side=0, intorder=p2.intorder)
    fb1 = InteriorFacetBasis(p2.mesh, ElementTriP2(), facets=interior, side=1, intorder=p2.intorder)
    grad0 = np.asarray(fb0.interpolate(u_lift).grad)
    grad1 = np.asarray(fb1.interpolate(u_lift).grad)
    if element_k is None:
        k0 = _scalar_values(p2.coefficient, np.asarray(fb0.global_coordinates()), "facet coefficient")
        k1 = _scalar_values(p2.coefficient, np.asarray(fb1.global_coordinates()), "facet coefficient")
    else:
        k0 = element_k[p2.mesh.f2t[0, interior]][:, None]
        k1 = element_k[p2.mesh.f2t[1, interior]][:, None]
    # Both facet bases use the normal outward from side 0.
    jump = np.sum((k0[None] * grad0 - k1[None] * grad1) * np.asarray(fb0.normals), axis=0)
    edges = np.column_stack([np.sum(jump * np.asarray(fb0.interpolate(defects[:, j])) * fb0.dx, axis=1)
                            for j in range(defects.shape[1])])
    for side in (0, 1):
        np.add.at(local, p2.mesh.f2t[side, interior], -.5 * edges)
    return {"available": True, "signed_element_contributions": local,
            "global_estimates": local.sum(axis=0), "signed_interior_face_jumps": edges,
            "interior_facet_ids": interior,
            "localization": "body_minus_half_interior_flux_jump_per_neighbor",
            "marking_quality_not_established": True}


def dwr(coarse, enriched, uH, z2matrix, *, source_index=0, goal_indices=None, p2_primal=None):
    """Audit enriched DWR, preserving algebraic residuals of inexact uH.

    ``global_estimates`` localize R_H(z2-I_H z2). ``algebraic_terms`` retain
    R_H(I_H z2), so ``total_estimates`` equal R_H(z2).  If the supplied enriched
    dual is an exact discrete solution, the total equals the enriched-vs-coarse
    QoI difference even for an inexact coarse solution.  ``p2_primal`` only adds
    an offline identity check and never changes the indicator or its labeling.
    """
    start = perf_counter()
    allowed = ("smooth", "element_aligned")
    if coarse.coefficient_alignment not in allowed or enriched.coefficient_alignment not in allowed:
        raise ValueError("DWR requires explicit smooth or element_aligned coefficients")
    E = prolongation_p1_p2(coarse, enriched)
    if not np.array_equal(coarse.basis.X, enriched.basis.X) or not np.array_equal(coarse.basis.W, enriched.basis.W):
        raise ValueError("P1 and P2 must use identical quadrature")
    if coarse.loads.shape[1] != enriched.loads.shape[1]:
        raise ValueError("P1/P2 must have the same number of load columns")
    if not isinstance(source_index, (int, np.integer)) or not 0 <= source_index < coarse.loads.shape[1]:
        raise ValueError("invalid source index")
    goals = tuple(range(1, coarse.loads.shape[1])) if goal_indices is None else tuple(goal_indices)
    if not goals or len(set(goals)) != len(goals) or any(not isinstance(j, (int, np.integer)) or
            j < 0 or j >= coarse.loads.shape[1] for j in goals):
        raise ValueError("goal_indices must be nonempty, distinct valid load columns")
    u = _vector(uH, coarse.basis.N, "uH")
    z2 = np.asarray(z2matrix, dtype=float)
    if z2.ndim == 1:
        z2 = z2[:, None]
    if z2.shape != (enriched.basis.N, len(goals)) or not np.all(np.isfinite(z2)):
        raise ValueError("z2matrix has incompatible shape or nonfinite values")
    if np.max(abs(u[coarse.boundary]), initial=0.) > 1e-13 or np.max(abs(z2[enriched.boundary]), initial=0.) > 1e-13:
        raise ValueError("DWR audit currently requires homogeneous boundary values")
    nested_matrix = E.T @ enriched.A @ E
    matrix_diff = nested_matrix - coarse.A
    matrix_error = float(np.max(abs(matrix_diff.data), initial=0.) / max(np.max(abs(coarse.A.data)), 1e-300))
    nested_loads = E.T @ enriched.loads
    load_error = float(np.linalg.norm(nested_loads - coarse.loads) / max(np.linalg.norm(coarse.loads), 1e-300))
    if matrix_error > 1e-11 or load_error > 1e-11:
        raise ValueError("P1/P2 represent different coefficient or load/quadrature problems")
    u_lift = E @ u
    z_vertices = np.zeros((coarse.basis.N, len(goals)))
    z_vertices[coarse.basis.nodal_dofs[0]] = z2[enriched.basis.nodal_dofs[0]]
    z_interp = E @ z_vertices
    defects = z2 - z_interp
    residual = enriched.loads[:, source_index] - enriched.A @ u_lift
    global_estimates = defects.T @ residual
    algebraic_terms = z_interp.T @ residual
    total_estimates = z2.T @ residual
    x = np.asarray(enriched.basis.global_coordinates())
    uh = enriched.basis.interpolate(u_lift)
    kvals = _scalar_values(enriched.coefficient, x, "coefficient")
    fvals = _scalar_values(enriched.rhs_functions[source_index], x, "source")
    local = []
    for j in range(len(goals)):
        w = enriched.basis.interpolate(defects[:, j])
        integrand = fvals * np.asarray(w) - kvals * np.sum(uh.grad * w.grad, axis=0)
        local.append(np.sum(integrand * enriched.basis.dx, axis=1))
    local = np.column_stack(local)
    dual_check, _ = ResidualEvaluator(enriched.A[enriched.free][:, enriched.free]).evaluate(
        z2[enriched.free], enriched.loads[enriched.free][:, goals])
    strong = _strong_contributions(coarse, enriched, u_lift, defects, source_index)
    if strong["available"]:
        strong["weak_global_difference"] = strong["global_estimates"] - global_estimates
    result = dict(global_estimates=global_estimates, algebraic_terms=algebraic_terms,
        total_estimates=total_estimates, signed_element_contributions=local,
        element_sum_errors=local.sum(axis=0) - global_estimates,
        algebraic_decomposition_errors=global_estimates + algebraic_terms - total_estimates,
        nested_matrix_relative_error=matrix_error, nested_load_relative_error=load_error,
        enriched_dual_metrics=dual_check, strong=strong,
        localization="element_weak_residual_of_enriched_minus_nodal_coarse_dual",
        no_continuous_error_certificate=True, marking_quality_not_established=True)
    if p2_primal is not None:
        uf = _vector(p2_primal, enriched.basis.N, "p2_primal")
        qdiff = enriched.loads[:, goals].T @ uf - coarse.loads[:, goals].T @ u
        result["qoi_differences"] = qdiff
        result["identity_errors"] = total_estimates - qdiff
    result["timings"] = {"dwr_audit_seconds": perf_counter() - start}
    return result
