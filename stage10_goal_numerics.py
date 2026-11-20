"""Nested P1 identities for the stage-10 goal-error representation pilot.

``nc`` and ``nf=2*nc`` count cells, not interior vertices.  Coarse operators
and loads are restricted from the fine discretization.  All identities in
this module concern that *two-level discrete error*, not the unknown error
to the continuous PDE solution.  Local signed contributions use the fixed
unintegrated element residual; they are not guaranteed error bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from numerics import _cell_triangles, _TRIANGLE_STIFFNESS


@dataclass
class NestedProblem:
    nc: int
    nf: int
    k_fine: np.ndarray
    Af: sp.csr_matrix
    Ac: sp.csr_matrix
    P: sp.csr_matrix
    P_full: sp.csr_matrix
    free: np.ndarray
    coarse_free: np.ndarray
    full_to_free: np.ndarray
    coordinates: np.ndarray
    full_coordinates: np.ndarray
    triangles: np.ndarray
    element_matrices: np.ndarray
    fine_diagonal: np.ndarray
    ff: np.ndarray
    fc: np.ndarray
    load_nodes: np.ndarray
    timings: dict
    cache: dict = field(default_factory=dict, repr=False)
    cache_statistics: dict = field(default_factory=lambda: {
        "coarse_factorizations": 0, "fine_factorizations": 0,
        "coarse_solves": 0, "fine_solves": 0,
        "coarse_factorization_seconds": 0., "fine_factorization_seconds": 0.,
        "coarse_solve_seconds": 0., "fine_solve_seconds": 0.})

    @property
    def dimension(self):
        return self.Af.shape[0]


def _vertices(n):
    yy, xx = np.mgrid[0:n+1, 0:n+1]
    coordinates = np.column_stack((xx.ravel()/n, yy.ravel()/n))
    iy, ix = np.mgrid[1:n, 1:n]
    free = (iy*(n+1)+ix).ravel()
    return coordinates, free


def p1_prolongation(n_coarse):
    """Return (free P, full P) for the SW-to-NE triangulation.

    Odd/odd fine nodes lie on the coarse diagonal: their two nonzero
    weights belong to SW and NE, rather than the four bilinear corners.
    The full map is useful for affine-reproduction diagnostics; solvers
    use the boundary-eliminated free map.
    """
    nc = int(n_coarse)
    if nc != n_coarse or nc < 2:
        raise ValueError("n_coarse must be an integer with at least two cells")
    nf = 2*nc
    rows, cols, values = [], [], []
    for fy in range(nf+1):
        for fx in range(nf+1):
            cy, cx = fy//2, fx//2
            if fy % 2 == 0 and fx % 2 == 0:
                vertices = [(cy, cx, 1.)]
            elif fy % 2 == 0:
                vertices = [(cy, cx, .5), (cy, cx+1, .5)]
            elif fx % 2 == 0:
                vertices = [(cy, cx, .5), (cy+1, cx, .5)]
            else:
                vertices = [(cy, cx, .5), (cy+1, cx+1, .5)]
            for vy, vx, weight in vertices:
                rows.append(fy*(nf+1)+fx)
                cols.append(vy*(nc+1)+vx)
                values.append(weight)
    full = sp.coo_matrix((values, (rows, cols)),
                         shape=((nf+1)**2, (nc+1)**2)).tocsr()
    _, fine_free = _vertices(nf)
    _, coarse_free = _vertices(nc)
    return full[fine_free][:, coarse_free].tocsr(), full


def _load_nodes(nf, triangles, coordinates, function):
    """Degree-two triangle quadrature, shared by source and goal loads."""
    bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6],
                     [1/6, 1/6, 2/3]], dtype=np.float64)
    points = np.einsum("qv,ctvd->ctqd", bary, coordinates[triangles])
    if function is None:
        values = np.ones(points.shape[:-1])
    elif callable(function):
        values = np.asarray(function(points[..., 0], points[..., 1]), dtype=float)
        values = np.broadcast_to(values, points.shape[:-1])
    elif np.isscalar(function):
        values = np.full(points.shape[:-1], float(function))
    else:
        raise TypeError("source/goal must be callable, scalar, or None (constant one)")
    if not np.all(np.isfinite(values)):
        raise ValueError("source/goal quadrature values must be finite")
    return np.einsum("ctq,qv->ctv", values, bary)/(6.*nf**2)


def assemble_load(case: NestedProblem, function):
    """Return (fine free load, per-triangle load vector, elapsed seconds)."""
    start = perf_counter()
    nodes = _load_nodes(case.nf, case.triangles, case.full_coordinates, function)
    full = np.zeros((case.nf+1)**2)
    np.add.at(full, case.triangles.ravel(), nodes.ravel())
    return full[case.free], nodes, perf_counter()-start


def assemble_nested(n_coarse, coefficient, source=None):
    """Assemble an SPD nested Darcy problem without factorizing its matrices.

    ``coefficient`` is positive scalar or a ``(2*nc,2*nc)`` cell array.
    An omitted source is the constant one.  Subsequent ``solve_primal``
    calls may replace it while sharing both cached factorizations.
    """
    start = perf_counter()
    nc = int(n_coarse)
    if nc != n_coarse or nc < 2:
        raise ValueError("n_coarse must be an integer with at least two cells")
    nf = 2*nc
    k = np.asarray(coefficient, dtype=np.float64)
    if k.ndim == 0:
        k = np.full((nf, nf), float(k))
    if k.shape != (nf, nf) or not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("coefficient must be finite, positive, and shaped (2*nc,2*nc)")
    k = k.copy()
    coordinates, free = _vertices(nf)
    _, coarse_free = _vertices(nc)
    full_to_free = np.full((nf+1)**2, -1, dtype=np.int64)
    full_to_free[free] = np.arange(free.size)
    triangles = _cell_triangles(nf, nf)
    element_matrices = k.reshape(-1, 1, 1, 1)*_TRIANGLE_STIFFNESS[None]
    t = perf_counter()
    rows = np.broadcast_to(triangles[..., :, None], (*triangles.shape, 3))
    cols = np.broadcast_to(triangles[..., None, :], (*triangles.shape, 3))
    full = sp.coo_matrix((element_matrices.ravel(), (rows.ravel(), cols.ravel())),
                         shape=((nf+1)**2, (nf+1)**2)).tocsr()
    Af = full[free][:, free].tocsr()
    matrix_seconds = perf_counter()-t
    t = perf_counter()
    P, P_full = p1_prolongation(nc)
    Ac = (P.T @ Af @ P).tocsr()
    hierarchy_seconds = perf_counter()-t
    t = perf_counter()
    load_nodes = _load_nodes(nf, triangles, coordinates, source)
    full_load = np.zeros((nf+1)**2)
    np.add.at(full_load, triangles.ravel(), load_nodes.ravel())
    ff = full_load[free]
    fc = np.asarray(P.T @ ff)
    load_seconds = perf_counter()-t
    return NestedProblem(nc=nc, nf=nf, k_fine=k, Af=Af, Ac=Ac, P=P,
        P_full=P_full, free=free, coarse_free=coarse_free,
        full_to_free=full_to_free, coordinates=coordinates[free],
        full_coordinates=coordinates, triangles=triangles,
        element_matrices=element_matrices, fine_diagonal=Af.diagonal(),
        ff=ff, fc=fc, load_nodes=load_nodes,
        timings=dict(matrix_seconds=matrix_seconds, hierarchy_seconds=hierarchy_seconds,
                     load_seconds=load_seconds, assembly_seconds=perf_counter()-start))


def free_to_full(case: NestedProblem, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != case.dimension:
        raise ValueError("fine free vector/matrix has incompatible shape")
    full = np.zeros(((case.nf+1)**2, *values.shape[1:]), dtype=np.float64)
    full[case.free] = values
    return full


def _solve(case, rhs, level, transpose=False):
    matrix = case.Ac if level == "coarse" else case.Af
    cache_hit = level in case.cache
    factor_seconds = 0.
    if not cache_hit:
        t = perf_counter()
        case.cache[level] = sla.splu(matrix.tocsc())
        factor_seconds = perf_counter()-t
        case.cache_statistics[level+"_factorizations"] += 1
        case.cache_statistics[level+"_factorization_seconds"] += factor_seconds
    t = perf_counter()
    value = case.cache[level].solve(np.asarray(rhs), trans="T" if transpose else "N")
    solve_seconds = perf_counter()-t
    case.cache_statistics[level+"_solves"] += 1
    case.cache_statistics[level+"_solve_seconds"] += solve_seconds
    return value, dict(factorization_seconds=factor_seconds,
                       solve_seconds=solve_seconds, cache_hit=cache_hit)


def _relative_residual(matrix, value, rhs, transpose=False):
    residual = rhs-(matrix.T @ value if transpose else matrix @ value)
    return float(np.linalg.norm(residual)/max(np.linalg.norm(rhs), np.finfo(float).tiny))


def _timings(start, load_seconds, coarse, fine, diagnostics_seconds):
    timing = dict(load_seconds=load_seconds,
        coarse_factorization_seconds=coarse["factorization_seconds"],
        coarse_solve_seconds=coarse["solve_seconds"],
        fine_factorization_seconds=fine["factorization_seconds"] if fine else 0.,
        fine_solve_seconds=fine["solve_seconds"] if fine else 0.,
        diagnostics_seconds=diagnostics_seconds)
    components_seconds = sum(timing.values())
    timing["total_seconds"] = perf_counter()-start
    timing["overhead_seconds"] = timing["total_seconds"]-components_seconds
    # Do not include cache flags in numeric cost-component sums.
    return timing, dict(coarse=coarse["cache_hit"], fine=fine["cache_hit"] if fine else None)


def solve_primal(case: NestedProblem, source=None, fine_reference=True):
    """Solve one source, with optional fine reference and charged LU reuse.

    ``r_u`` is the fine-space residual of the lifted *coarse* solution.
    It is intentionally nonzero; residual checks of the actual linear
    solves are reported separately.  No fine solve is needed online.
    """
    start = perf_counter()
    if source is None:
        ff, load_nodes, load_seconds = case.ff, case.load_nodes, 0.
    else:
        ff, load_nodes, load_seconds = assemble_load(case, source)
    fc = np.asarray(case.P.T @ ff)
    uc, coarse_time = _solve(case, fc, "coarse")
    uf, fine_time = _solve(case, ff, "fine") if fine_reference else (None, None)
    t = perf_counter()
    prolonged = np.asarray(case.P @ uc)
    r_u = ff-case.Af @ prolonged
    full_u = free_to_full(case, prolonged)
    residual_nodes = load_nodes-np.einsum("ctij,ctj->cti", case.element_matrices,
                                          full_u[case.triangles])
    result = dict(ff=ff, fc=fc, load_nodes=load_nodes, uc=uc, uf=uf,
        prolonged_uc=prolonged, r_u=np.asarray(r_u), cell_residual_nodes=residual_nodes,
        coarse_relative_residual=_relative_residual(case.Ac, uc, fc),
        fine_relative_residual=_relative_residual(case.Af, uf, ff) if uf is not None else None,
        restricted_residual_norm=float(np.linalg.norm(case.P.T @ r_u)))
    timing, hits = _timings(start, load_seconds, coarse_time, fine_time, perf_counter()-t)
    result.update(timings=timing, cache_hits=hits)
    return result


def element_indicators(case: NestedProblem, primal, delta_z):
    """Signed unintegrated local DWR contributions in one fixed gauge.

    No dense element-by-global-node operator is constructed.  Torch code
    can reproduce this using ``triangles``, ``cell_residual_nodes`` and
    the predicted vector padded with zero boundary nodes.
    """
    full_z = free_to_full(case, delta_z)
    if full_z.ndim != 1:
        raise ValueError("element_indicators expects a single fine free vector")
    triangle = np.einsum("ctv,ctv->ct", primal["cell_residual_nodes"],
                         full_z[case.triangles])
    fine_cell = triangle.sum(axis=1).reshape(case.nf, case.nf)
    coarse_cell = fine_cell.reshape(case.nc, 2, case.nc, 2).sum(axis=(1, 3))
    return dict(fine_triangle=triangle, fine_cell=fine_cell, coarse_cell=coarse_cell,
                total=float(coarse_cell.sum()),
                residual_pairing=float(np.dot(primal["r_u"], delta_z)))


def solve_goal(case: NestedProblem, goal, primal=None, fine_reference=True):
    """Solve a goal load independently of the source and optionally pair it.

    The exact enrichment defect is ``zf-P@zc``.  In the SPD setting it is
    linear in the goal and A-orthogonal to the coarse space.  Any coarse
    subtraction from a predicted defect leaves the global DWR pairing
    unchanged when the coarse primal solve is exact, but may change local
    signed contributions.  Such a projection is deliberately not an
    online layer in this module.
    """
    start = perf_counter()
    gf, goal_nodes, load_seconds = assemble_load(case, goal)
    gc = np.asarray(case.P.T @ gf)
    zc, coarse_time = _solve(case, gc, "coarse", transpose=True)
    zf, fine_time = _solve(case, gf, "fine", transpose=True) if fine_reference else (None, None)
    t = perf_counter()
    prolonged = np.asarray(case.P @ zc)
    r_z = gf-case.Af.T @ prolonged
    delta = zf-prolonged if zf is not None else None
    result = dict(gf=gf, gc=gc, goal_nodes=goal_nodes, zc=zc, zf=zf,
        prolonged_zc=prolonged, r_z=np.asarray(r_z), delta_z=delta,
        coarse_relative_residual=_relative_residual(case.Ac, zc, gc, transpose=True),
        fine_relative_residual=_relative_residual(case.Af, zf, gf, transpose=True) if zf is not None else None,
        restricted_residual_norm=float(np.linalg.norm(case.P.T @ r_z)))
    if primal is not None and delta is not None:
        result["dwr"] = float(primal["r_u"] @ delta)
        result["local_indicators"] = element_indicators(case, primal, delta)
        result["exact_goal_error"] = float(gf @ (primal["uf"]-primal["prolonged_uc"])) \
            if primal["uf"] is not None else None
    timing, hits = _timings(start, load_seconds, coarse_time, fine_time, perf_counter()-t)
    result.update(timings=timing, cache_hits=hits)
    return result


def build_problem(n_coarse, k_fine, f_callable=None, g_callable=None):
    """Convenience full-reference bundle; repeated loads should reuse case."""
    case = assemble_nested(n_coarse, k_fine, f_callable)
    primal = solve_primal(case)
    goal = solve_goal(case, g_callable, primal=primal)
    return dict(case=case, primal=primal, goal=goal)
