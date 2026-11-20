"""Actual one-level local P1 refinement inside the stage-10 fine reference.

Each marked coarse square has its two SW--NE triangles red-refined into
eight fine triangles.  An interface midpoint is constrained to the average
of its coarse-edge endpoints unless both adjacent squares are refined.
The resulting constrained finite-element space is conforming in H1, even
though the geometric mesh has hanging nodes.  The Galerkin operator uses
the already assembled fine coefficient and quadrature without resampling.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from stage10_goal_numerics import p1_prolongation


def _selection(selected_cells, nc):
    nc = int(nc)
    if nc < 2:
        raise ValueError("nc must be at least two")
    values = np.asarray(selected_cells)
    if values.dtype == np.bool_:
        if values.size != nc * nc:
            raise ValueError("boolean selection must contain nc*nc entries")
        return values.reshape(nc, nc).copy()
    if values.ndim != 1:
        raise ValueError("selected cells must be a one-dimensional index list")
    indices = values.astype(np.int64)
    if np.any(indices != values) or np.any(indices < 0) or np.any(indices >= nc * nc):
        raise ValueError("selected cell index is invalid")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("selected cell indices must not repeat")
    result = np.zeros(nc * nc, dtype=bool)
    result[indices] = True
    return result.reshape(nc, nc)


def count_added_dofs(selected_cells, nc):
    """Centers plus liberated shared-edge midpoints; boundary values are fixed."""
    selected = _selection(selected_cells, nc)
    return int(selected.sum() + (selected[:, :-1] & selected[:, 1:]).sum()
               + (selected[:-1, :] & selected[1:, :]).sum())


def _neighbors(cell, nc):
    y, x = divmod(int(cell), nc)
    if x:
        yield cell - 1
    if x + 1 < nc:
        yield cell + 1
    if y:
        yield cell - nc
    if y + 1 < nc:
        yield cell + nc


def budget_mark(scores, nc, budget):
    """Deterministic score-greedy marking with an exact added-DOF budget.

    Adding a square costs one center plus one midpoint per already selected
    neighbor.  A descending-score pass takes every feasible candidate.  If
    it stalls, budgets <=ceil(nc*nc/2) have a guaranteed exact fallback: the
    highest-scoring budget cells from the better feasible checkerboard.
    This fallback is explicitly recorded.  Larger arbitrary budgets are
    rejected rather than silently changed: some integers are impossible
    (a 2x2 mesh cannot add 4, 6, or 7 DOFs in this refinement family).
    Full refinement is also supported as an explicit special case.
    """
    nc = int(nc)
    raw_budget = budget
    budget = int(budget)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if nc < 2 or values.size != nc * nc or not np.all(np.isfinite(values)):
        raise ValueError("scores must contain nc*nc finite values, with nc>=2")
    if budget != raw_budget or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    maximum = 3 * nc * nc - 2 * nc
    guarantee_limit = (nc * nc + 1) // 2
    if budget > guarantee_limit and budget != maximum:
        raise ValueError("exact marking is guaranteed only up to ceil(nc*nc/2) "
                         "or at full refinement; the requested budget is unsupported")
    order = np.lexsort((np.arange(values.size), -values))
    if budget == maximum:
        cells = np.arange(nc * nc, dtype=np.int64)
        return cells, dict(rule="full_refinement", added_dofs=maximum,
                          requested_budget=budget, fallback=False,
                          selected_cell_count=len(cells), score_sum=float(values.sum()))
    selected = np.zeros(nc * nc, dtype=bool)
    spent = 0
    for cell in order:
        cost = 1 + sum(bool(selected[neighbor]) for neighbor in _neighbors(cell, nc))
        if spent + cost <= budget:
            selected[cell] = True
            spent += cost
        if spent == budget:
            break
    fallback = spent != budget
    if fallback:
        candidates = []
        yy, xx = np.mgrid[:nc, :nc]
        parity = ((xx + yy) % 2).ravel()
        for value in (0, 1):
            eligible = order[parity[order] == value]
            if len(eligible) >= budget:
                cells = eligible[:budget]
                candidates.append((float(values[cells].sum()), value, cells))
        _, parity_value, cells = max(candidates, key=lambda item: (item[0], -item[1]))
        selected[:] = False
        selected[cells] = True
    else:
        parity_value = None
    cells = np.flatnonzero(selected)
    actual = count_added_dofs(cells, nc)
    if actual != budget:
        raise RuntimeError("internal error: marking failed its exact DOF contract")
    return cells, dict(rule="descending_score_greedy_with_checkerboard_fallback",
        added_dofs=actual, requested_budget=budget, fallback=bool(fallback),
        fallback_parity=parity_value, greedy_spent_before_fallback=spent,
        selected_cell_count=len(cells), score_sum=float(values[cells].sum()))


def build_full_space(nc, selected_cells):
    """Full-node interpolation, including coarse Dirichlet nodes, for diagnostics."""
    nc = int(nc)
    nf = 2 * nc
    selected = _selection(selected_cells, nc)
    _, coarse_full = p1_prolongation(nc)
    independent = np.zeros((nf + 1, nf + 1), dtype=bool)
    independent[::2, ::2] = True
    independent[1::2, 1::2] = selected
    independent[1::2, 2:-1:2] = selected[:, :-1] & selected[:, 1:]
    independent[2:-1:2, 1::2] = selected[:-1, :] & selected[1:, :]
    independent_nodes = np.flatnonzero(independent.ravel())
    node_to_column = np.full((nf + 1) ** 2, -1, dtype=np.int64)
    node_to_column[independent_nodes] = np.arange(len(independent_nodes))
    cy, cx = np.mgrid[:nc + 1, :nc + 1]
    coarse_node_ids = (2 * cy * (nf + 1) + 2 * cx).ravel()
    old = coarse_full.tocoo()
    constrained = ~independent.ravel()[old.row]
    rows = np.concatenate((old.row[constrained], independent_nodes))
    cols = np.concatenate((node_to_column[coarse_node_ids[old.col[constrained]]],
                           node_to_column[independent_nodes]))
    data = np.concatenate((old.data[constrained], np.ones(len(independent_nodes))))
    full = sp.coo_matrix((data, (rows, cols)),
        shape=((nf + 1) ** 2, len(independent_nodes))).tocsr()

    triangles = []
    parents = []
    for cell in range(nc * nc):
        y, x = divmod(cell, nc)
        if selected[y, x]:
            origins = [(2*y+dy, 2*x+dx, 1) for dy in (0, 1) for dx in (0, 1)]
        else:
            origins = [(2*y, 2*x, 2)]
        for fy, fx, width in origins:
            sw = fy * (nf + 1) + fx
            se, nw, ne = sw+width, sw+width*(nf+1), sw+width*(nf+2)
            triangles.extend([[sw, se, ne], [sw, ne, nw]])
            parents.extend([cell, cell])

    hanging = []
    liberated_edges = []
    for y in range(nc):
        for x in range(1, nc):
            if selected[y, x-1] != selected[y, x]:
                middle = (2*y+1)*(nf+1)+2*x
                hanging.append(dict(node=middle,
                    endpoints=[middle-(nf+1), middle+(nf+1)], weights=[.5, .5]))
            elif selected[y, x-1] and selected[y, x]:
                middle = (2*y+1)*(nf+1)+2*x
                liberated_edges.append(dict(node=middle,
                    adjacent_cells=[y*nc+x-1, y*nc+x],
                    endpoints=[middle-(nf+1), middle+(nf+1)]))
    for y in range(1, nc):
        for x in range(nc):
            if selected[y-1, x] != selected[y, x]:
                middle = 2*y*(nf+1)+2*x+1
                hanging.append(dict(node=middle, endpoints=[middle-1, middle+1],
                                    weights=[.5, .5]))
            elif selected[y-1, x] and selected[y, x]:
                middle = 2*y*(nf+1)+2*x+1
                liberated_edges.append(dict(node=middle,
                    adjacent_cells=[(y-1)*nc+x, y*nc+x],
                    endpoints=[middle-1, middle+1]))
    metadata = dict(nc=nc, nf=nf, selected_cells=np.flatnonzero(selected.ravel()).tolist(),
        selected_cell_count=int(selected.sum()), added_dofs=count_added_dofs(selected, nc),
        independent_full_node_ids=independent_nodes.tolist(),
        adaptive_mesh_triangles_full_node_ids=triangles, triangle_parent_cells=parents,
        geometric_triangle_count=len(triangles), hanging_constraints=hanging,
        liberated_edges=liberated_edges,
        space_description="red-refined coarse triangles with constrained hanging edge midpoints")
    return full, metadata


def build_space(case, selected_cells):
    """Return fine-free interpolation T and serializable actual-mesh metadata.

    Columns follow fine-node order.  No marking yields exactly case.P;
    all squares marked yields exactly the fine identity matrix.
    """
    start = perf_counter()
    full, metadata = build_full_space(case.nc, selected_cells)
    nodes = np.asarray(metadata["independent_full_node_ids"])
    y, x = np.divmod(nodes, case.nf + 1)
    keep = (x > 0) & (x < case.nf) & (y > 0) & (y < case.nf)
    T = full[case.free][:, keep].tocsr()
    metadata.update(independent_free_node_ids=nodes[keep].tolist(),
        independent_fine_free_indices=case.full_to_free[nodes[keep]].tolist(),
        adaptive_dofs=int(T.shape[1]), coarse_dofs=int(case.P.shape[1]),
        fine_dofs=int(case.Af.shape[0]), interpolation_nnz=int(T.nnz),
        space_build_seconds=perf_counter()-start)
    if T.shape[1] - case.P.shape[1] != metadata["added_dofs"]:
        raise RuntimeError("refined-space dimension disagrees with the DOF count")
    return T, metadata


def solve_adaptive(case, T, ff=None):
    """Galerkin solve on the actual constrained local-refinement space.

    Fine-grid reference assembly is shared infrastructure, not hidden in
    these component timings.  The caller must charge it in complete costs.
    The returned residual is for the adaptive equations, not the nonzero
    fine residual of the adaptive approximation.
    """
    start = perf_counter()
    rhs_fine = np.asarray(case.ff if ff is None else ff, dtype=np.float64)
    if T.shape[0] != case.dimension or rhs_fine.ndim != 1 or rhs_fine.shape[0] != case.dimension:
        raise ValueError("adaptive interpolation or load has an incompatible shape")
    tick = perf_counter()
    matrix = (T.T @ case.Af @ T).tocsc()
    rhs = np.asarray(T.T @ rhs_fine)
    assembly_seconds = perf_counter()-tick
    tick = perf_counter()
    lu = sla.splu(matrix)
    factorization_seconds = perf_counter()-tick
    tick = perf_counter()
    coefficients = lu.solve(rhs)
    solve_seconds = perf_counter()-tick
    tick = perf_counter()
    lifted = np.asarray(T @ coefficients)
    residual = float(np.linalg.norm(matrix @ coefficients-rhs)
                     / max(np.linalg.norm(rhs), np.finfo(float).tiny))
    lift_diagnostics_seconds = perf_counter()-tick
    return dict(lifted_u=lifted, coefficients=coefficients, relative_residual=residual,
        adaptive_dofs=int(T.shape[1]), timings=dict(assembly_seconds=assembly_seconds,
        factorization_seconds=factorization_seconds, solve_seconds=solve_seconds,
        lift_diagnostics_seconds=lift_diagnostics_seconds, total_seconds=perf_counter()-start))
