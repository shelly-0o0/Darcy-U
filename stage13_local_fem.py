"""Direct assembly of the stage-11 constrained local-refinement space.

Only the requested space's sparse matrix is formed: there is no additional
fine reference matrix, fine-to-adaptive sparse interpolation, or Schur
complement. (Selecting every cell requests the fine matrix itself.)
Unrefined triangles integrate the original
piecewise-constant fine coefficient exactly. Loads retain the original
three-point degree-two quadrature on every fine triangle; they are not
resampled on coarse triangles. All boundary values are homogeneous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from numerics import _cell_triangles, _TRIANGLE_STIFFNESS
from stage11_adaptive import _selection


@dataclass
class LoadData:
    """One explicitly paid fine quadrature evaluation, reusable within a pipeline."""
    nf: int
    fine_loads_full: np.ndarray
    functions: tuple = field(repr=False)

    @property
    def fine_free_loads(self):
        nloads = self.fine_loads_full.shape[0]
        return self.fine_loads_full.reshape(nloads, self.nf + 1, self.nf + 1)[
            :, 1:-1, 1:-1].reshape(nloads, (self.nf - 1) ** 2)


@dataclass
class LocalProblem:
    nc: int
    nf: int
    A: sp.csc_matrix
    loads: np.ndarray
    independent_free_node_ids: np.ndarray
    timings: dict
    metadata: dict
    load_data: LoadData = field(repr=False)
    _columns: np.ndarray = field(repr=False)
    _weights: np.ndarray = field(repr=False)

    @property
    def dimension(self):
        return self.A.shape[0]


def _node_constraints(nc, selected):
    """At most two independent free-node values per fine vertex, no sparse T."""
    nf = 2 * nc
    independent = np.zeros((nf + 1, nf + 1), dtype=bool)
    independent[::2, ::2] = True
    independent[1::2, 1::2] = selected
    independent[1::2, 2:-1:2] = selected[:, :-1] & selected[:, 1:]
    independent[2:-1:2, 1::2] = selected[:-1, :] & selected[1:, :]
    independent[[0, -1], :] = False
    independent[:, [0, -1]] = False
    nodes = np.flatnonzero(independent.ravel())
    node_to_column = np.full((nf + 1) ** 2, -1, dtype=np.int64)
    node_to_column[nodes] = np.arange(nodes.size)

    fy, fx = np.mgrid[:nf + 1, :nf + 1]
    low = (fy // 2 * 2 * (nf + 1) + fx // 2 * 2).ravel()
    high = low + (fy % 2 * 2 * (nf + 1) + fx % 2 * 2).ravel()
    columns = np.column_stack((node_to_column[low], node_to_column[high]))
    weights = np.full(columns.shape, .5, dtype=np.float64)
    # Odd/odd unrefined nodes use the SW--NE diagonal, not bilinear weights.
    columns[nodes, 0] = node_to_column[nodes]
    columns[nodes, 1] = -1
    weights[nodes, 0] = 1.
    weights[nodes, 1] = 0.
    weights[columns < 0] = 0.
    return nodes, columns, weights


def _mixed_triangles(nc, coefficient, selected):
    """Actual coarse or red-refined triangles and their integrated diffusion."""
    nf = 2 * nc
    cy, cx = np.mgrid[:nc, :nc]
    coarse_sw = (2 * cy * (nf + 1) + 2 * cx).ravel()
    coarse_sw = coarse_sw[~selected.ravel()]
    coarse_tri = np.stack((np.column_stack((coarse_sw, coarse_sw + 2,
                                           coarse_sw + 2 * (nf + 2))),
                           np.column_stack((coarse_sw, coarse_sw + 2 * (nf + 2),
                                            coarse_sw + 2 * (nf + 1)))), axis=1)
    k00, k01 = coefficient[::2, ::2], coefficient[::2, 1::2]
    k10, k11 = coefficient[1::2, ::2], coefficient[1::2, 1::2]
    coarse_coefficients = np.stack(((k00 + 2 * k01 + k11) / 4,
                                    (k00 + 2 * k10 + k11) / 4), axis=-1)
    coarse_coefficients = coarse_coefficients[~selected]

    ys, xs = np.nonzero(selected)
    fy = (2 * ys[:, None] + np.array([0, 0, 1, 1])).ravel()
    fx = (2 * xs[:, None] + np.array([0, 1, 0, 1])).ravel()
    fine_sw = fy * (nf + 1) + fx
    fine_tri = np.stack((np.column_stack((fine_sw, fine_sw + 1, fine_sw + nf + 2)),
                        np.column_stack((fine_sw, fine_sw + nf + 2,
                                         fine_sw + nf + 1))), axis=1)
    fine_coefficients = np.repeat(coefficient[fy, fx, None], 2, axis=1)
    triangles = np.concatenate((coarse_tri, fine_tri), axis=0).reshape(-1, 3)
    integrated_k = np.concatenate((coarse_coefficients, fine_coefficients)).ravel()
    orientation = np.tile(np.arange(2), triangles.shape[0] // 2)
    return triangles, integrated_k, orientation


def _assemble_matrix(triangles, integrated_k, orientation, columns, weights, ndofs):
    local_columns = columns[triangles]
    local_weights = weights[triangles]
    element_matrices = integrated_k[:, None, None] * _TRIANGLE_STIFFNESS[orientation]
    ordinary = np.all(local_weights[:, :, 1] == 0, axis=1)
    row_parts, col_parts, data_parts = [], [], []
    if np.any(ordinary):
        ids = local_columns[ordinary, :, 0]
        w = local_weights[ordinary, :, 0]
        rows = np.broadcast_to(ids[:, :, None], (len(ids), 3, 3))
        cols = np.broadcast_to(ids[:, None, :], rows.shape)
        values = element_matrices[ordinary] * w[:, :, None] * w[:, None, :]
        keep = (rows >= 0) & (cols >= 0) & (values != 0)
        row_parts.append(rows[keep]); col_parts.append(cols[keep]); data_parts.append(values[keep])
    if np.any(~ordinary):
        ids = local_columns[~ordinary]
        w = local_weights[~ordinary]
        shape = (len(ids), 3, 2, 3, 2)
        rows = np.broadcast_to(ids[:, :, :, None, None], shape)
        cols = np.broadcast_to(ids[:, None, None, :, :], shape)
        values = (element_matrices[~ordinary, :, None, :, None]
                  * w[:, :, :, None, None] * w[:, None, None, :, :])
        keep = (rows >= 0) & (cols >= 0) & (values != 0)
        row_parts.append(rows[keep]); col_parts.append(cols[keep]); data_parts.append(values[keep])
    matrix = sp.coo_matrix((np.concatenate(data_parts),
                            (np.concatenate(row_parts), np.concatenate(col_parts))),
                           shape=(ndofs, ndofs)).tocsc()
    matrix.eliminate_zeros()
    return matrix


def prepare_loads(nc, functions):
    """Evaluate original fine quadrature once; callers must charge this work."""
    raw_nc = nc
    nc = int(nc)
    if nc != raw_nc or nc < 2:
        raise ValueError("nc must be an integer with at least two cells")
    nf = 2 * nc
    functions = tuple(functions)
    if not functions:
        raise ValueError("at least one load function is required")
    triangles = _cell_triangles(nf, nf)
    yy, xx = np.mgrid[:nf + 1, :nf + 1]
    coordinates = np.column_stack((xx.ravel() / nf, yy.ravel() / nf))
    bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    points = np.einsum("qv,ctvd->ctqd", bary, coordinates[triangles])
    all_nodes = (nf + 1) ** 2
    full_loads = np.empty((len(functions), all_nodes))
    for j, function in enumerate(functions):
        if function is None:
            values = np.ones(points.shape[:-1])
        elif callable(function):
            values = np.broadcast_to(np.asarray(function(points[..., 0], points[..., 1]),
                                                dtype=np.float64), points.shape[:-1])
        elif np.isscalar(function):
            values = np.full(points.shape[:-1], float(function))
        else:
            raise TypeError("each load must be callable, scalar, or None")
        if not np.all(np.isfinite(values)):
            raise ValueError("source/goal quadrature values must be finite")
        nodes = np.einsum("ctq,qv->ctv", values, bary) / (6. * nf ** 2)
        full_loads[j] = np.bincount(triangles.ravel(), weights=nodes.ravel(), minlength=all_nodes)
    full_loads.flags.writeable = False
    return LoadData(nf, full_loads, functions)


def _project_loads(load_data, columns, weights, ndofs):
    loads = np.zeros((len(load_data.functions), ndofs))
    slots = [(columns[:, slot] >= 0, columns[:, slot], weights[:, slot])
             for slot in range(2)]
    for j, fine_load in enumerate(load_data.fine_loads_full):
        for keep, ids, w in slots:
            loads[j] += np.bincount(ids[keep], weights=(fine_load * w)[keep], minlength=ndofs)
    return loads


def assemble_local(nc, coefficient, functions, selected_cells, *, load_data=None):
    """Assemble A and loads[nloads, ndofs] in the true constrained space.

    Independent unknowns follow ascending fine full-node IDs, matching
    stage11_adaptive.build_space. Timings include fresh geometry, constraint
    maps, quadrature, and matrix assembly. The input functions are evaluated
    at the same fine-triangle quadrature points for every chosen space.
    Pass a previous object's load_data to reuse that pipeline's paid fine
    quadrature. Constraint projection is still charged for the new space.
    """
    start = perf_counter()
    raw_nc = nc
    nc = int(nc)
    if nc != raw_nc or nc < 2:
        raise ValueError("nc must be an integer with at least two cells")
    nf = 2 * nc
    k = np.asarray(coefficient, dtype=np.float64)
    if k.ndim == 0:
        k = np.full((nf, nf), float(k))
    if k.shape != (nf, nf) or not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("coefficient must be positive, finite, and shaped (2*nc,2*nc)")
    functions = tuple(functions)
    if not functions:
        raise ValueError("at least one load function is required")
    if load_data is not None:
        if (not isinstance(load_data, LoadData) or load_data.nf != nf
                or len(load_data.functions) != len(functions)
                or load_data.fine_loads_full.shape != (len(functions), (nf + 1) ** 2)):
            raise ValueError("reused load_data has an incompatible grid or load count")
        for current, original in zip(functions, load_data.functions):
            if current is not original and not (np.isscalar(current) and np.isscalar(original)
                                                and current == original):
                raise ValueError("reused load_data must correspond to the same load functions")
    selected = _selection(selected_cells, nc)
    nodes, columns, weights = _node_constraints(nc, selected)
    triangles, integrated_k, orientation = _mixed_triangles(nc, k, selected)
    space_seconds = perf_counter() - start
    tick = perf_counter()
    matrix = _assemble_matrix(triangles, integrated_k, orientation, columns, weights, len(nodes))
    matrix_seconds = perf_counter() - tick
    tick = perf_counter()
    reused_load_data = load_data is not None
    if load_data is None:
        load_data = prepare_loads(nc, functions)
        load_prepare_seconds = perf_counter() - tick
    else:
        load_prepare_seconds = 0.
    tick = perf_counter()
    loads = _project_loads(load_data, columns, weights, len(nodes))
    load_projection_seconds = perf_counter() - tick
    load_seconds = load_prepare_seconds + load_projection_seconds
    metadata = dict(nc=nc, nf=nf, selected_cells=np.flatnonzero(selected.ravel()).tolist(),
                    selected_cell_count=int(selected.sum()), adaptive_dofs=len(nodes),
                    coarse_dofs=(nc - 1) ** 2, fine_dofs=(nf - 1) ** 2,
                    added_dofs=len(nodes) - (nc - 1) ** 2,
                    geometric_triangle_count=len(triangles),
                    fine_quadrature_triangle_count=2 * nf ** 2,
                    quadrature_rule="three-point degree-two on original fine triangles",
                    global_fine_sparse_operator_assembled=bool(selected.all()),
                    fine_reference_operator_assembled=False,
                    global_sparse_interpolation_assembled=False,
                    global_schur_assembled=False, reused_load_data=reused_load_data)
    return LocalProblem(nc, nf, matrix, loads, nodes,
                        dict(space_seconds=space_seconds, matrix_seconds=matrix_seconds,
                             load_seconds=load_seconds, load_prepare_seconds=load_prepare_seconds,
                             load_projection_seconds=load_projection_seconds,
                             total_seconds=perf_counter()-start),
                        metadata, load_data, columns, weights)


def lift_to_fine(problem, coefficients):
    """Evaluate after the timed solve; return fine-free values, no sparse T."""
    u = np.asarray(coefficients, dtype=np.float64)
    if u.ndim not in (1, 2) or u.shape[0] != problem.dimension:
        raise ValueError("coefficients must have shape (ndofs,) or (ndofs,nvectors)")
    full = np.zeros(((problem.nf + 1) ** 2, *u.shape[1:]), dtype=np.float64)
    for slot in range(2):
        columns, weights = problem._columns[:, slot], problem._weights[:, slot]
        keep = columns >= 0
        if u.ndim == 1:
            full[keep] += weights[keep] * u[columns[keep]]
        else:
            full[keep] += weights[keep, None] * u[columns[keep]]
    return full.reshape(problem.nf + 1, problem.nf + 1, *u.shape[1:])[1:-1, 1:-1].reshape(
        (problem.nf - 1) ** 2, *u.shape[1:])
