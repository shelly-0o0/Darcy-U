"""Implementation optimizations for the unchanged additive Schwarz method.

The pilot implementation in ``numerics.py`` is deliberately left intact.
This module changes matrix assembly reuse and the representation of local
direct solves, not the patches, partition of unity, or preconditioner formula.
Sparse LU is a direct solve of the same SPD local matrix used by dense
Cholesky; floating-point solutions may differ at roundoff level.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from numerics import Patch, _full_cell_matrix


def _geometry(n, core_cells, overlap):
    specs = []
    counts = np.zeros((n - 1) ** 2, dtype=np.float64)
    for cy in range(0, n, core_cells):
        for cx in range(0, n, core_cells):
            cxe, cye = min(cx + core_cells, n), min(cy + core_cells, n)
            x0, x1 = max(0, cx - overlap), min(n, cxe + overlap)
            y0, y1 = max(0, cy - overlap), min(n, cye + overlap)
            yy, xx = np.mgrid[max(y0, 1):min(y1, n - 1) + 1,
                              max(x0, 1):min(x1, n - 1) + 1]
            ids = ((yy - 1) * (n - 1) + (xx - 1)).ravel()
            pos = ((yy - y0).ravel(), (xx - x0).ravel())
            shape = (y1 - y0 + 1, x1 - x0 + 1)
            flat = np.ravel_multi_index(pos, shape)
            counts[ids] += 1
            specs.append((ids, (x0, x1, y0, y1), (cx, cxe, cy, cye),
                          shape, pos, flat))
    if np.any(counts == 0):
        raise RuntimeError("subdomains do not cover every free node")
    return [(*spec, 1.0 / counts[spec[0]]) for spec in specs]


def _same_sparse(a, b):
    return a.shape == b.shape and (a != b).nnz == 0


def build_patches_cached(k, A, core_cells=16, overlap=2,
                         previous_patches=None):
    """Return the same patches as ``numerics.build_patches``, with safe reuse.

    A patch's Neumann matrix depends on cells inside its bounds. Its global
    principal matrix also depends on incident cells one cell beyond its
    vertex bounds. Complete reuse therefore requires an unchanged clipped
    one-cell halo, identical geometry and partition weights, AND an exactly
    equal current principal A block. The last check also protects callers
    that supply an A not assembled from k. No approximate cache key is used.

    Patch arrays shared with the previous step are treated as immutable.
    Returned Patch objects themselves are distinct, so reuse diagnostics do
    not mutate previous-step metadata. Dynamic cache attributes are private.
    """
    k = np.asarray(k, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 2:
        raise ValueError("k must be a square array with at least two cells per side")
    n = k.shape[0]
    if core_cells < 1 or overlap < 0:
        raise ValueError("invalid subdomain size/overlap")
    if A.shape != ((n - 1) ** 2, (n - 1) ** 2):
        raise ValueError("A and coefficient grid have incompatible shapes")
    A = A.tocsr()
    geometry_key = (n, core_cells, overlap)
    can_reuse_geometry = bool(previous_patches) and all(
        getattr(p, "_geometry_key", None) == geometry_key
        for p in previous_patches)
    if can_reuse_geometry:
        specs = [(p.ids, p.bounds, p.core_bounds, p.grid_shape,
                  p.local_positions, p.local_flat_indices, p.weights)
                 for p in previous_patches]
    else:
        specs = _geometry(n, core_cells, overlap)
    result = []
    for index, (ids, bounds, core_bounds, shape, pos, flat, w) in enumerate(specs):
        old = (previous_patches[index] if previous_patches is not None
               and index < len(previous_patches) else None)
        same_ids = old is not None and np.array_equal(ids, old.ids)
        same_geometry = (same_ids and old.bounds == bounds
                         and old.core_bounds == core_bounds
                         and np.array_equal(w, old.weights))
        x0, x1, y0, y1 = bounds
        dependency = k[max(0, y0 - 1):min(n, y1 + 1),
                       max(0, x0 - 1):min(n, x1 + 1)]
        ad_sparse = A[ids][:, ids].tocsr()
        if same_ids:
            previous_ad = getattr(old, "_sparse_local_matrix", None)
            if previous_ad is None:
                previous_ad = sp.csr_matrix(old.A_D)
            same_ad = _same_sparse(ad_sparse, previous_ad)
        else:
            same_ad = False
        old_dependency = getattr(old, "_coefficient_dependency", None)
        full_reuse = (same_geometry and same_ad and old_dependency is not None
                      and np.array_equal(dependency, old_dependency))
        if full_reuse:
            patch = replace(old)
        else:
            coefficient = k[y0:y1, x0:x1].copy()
            local_full = _full_cell_matrix(coefficient)
            an = local_full[flat][:, flat].toarray()
            ad = old.A_D if same_ad else ad_sparse.toarray()
            bb = an + w[:, None] * ad * w[None, :]
            old_chol = old.local_cholesky if same_ad else None
            patch = Patch(index, ids, bounds, core_bounds, shape, pos, flat,
                          coefficient, w, an, ad, bb, old_chol)
        patch._geometry_key = geometry_key
        patch._coefficient_dependency = (old_dependency if full_reuse
                                         else dependency.copy())
        patch._sparse_local_matrix = ad_sparse
        patch._sparse_local_factor = (getattr(old, "_sparse_local_factor", None)
                                      if same_ad else None)
        patch._matrix_reused = bool(full_reuse)
        result.append(patch)
    return result


def build_preconditioner_sparse(A, patches, spaces, profile=False):
    """Build the identical fixed additive Schwarz operator with sparse solves.

    Each local factor is computed once and reused for exactly unchanged A_D.
    There is no factorization, adaptivity, or changing tolerance in matvec.
    The coarse construction matches the original implementation. Optional
    counters are updated in the returned stats dict; leave profiling off for
    performance comparisons to avoid timer overhead in every application.
    """
    start = perf_counter()
    if len(patches) != len(spaces):
        raise ValueError("one local space is required per patch")
    row, col, val = [], [], []
    ncols = newly_factored = 0
    for patch, z in zip(patches, spaces):
        local_factor = getattr(patch, "_sparse_local_factor", None)
        if local_factor is None:
            ad = getattr(patch, "_sparse_local_matrix", None)
            if ad is None:
                ad = sp.csr_matrix(patch.A_D)
                patch._sparse_local_matrix = ad
            patch._sparse_local_factor = sla.splu(
                ad.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.,
                options={"SymmetricMode": True})
            newly_factored += 1
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[0] != len(patch.ids):
            raise ValueError("basis has incompatible local node count")
        q = z.shape[1]
        if q:
            row.append(np.repeat(patch.ids, q))
            col.append(np.tile(np.arange(ncols, ncols + q), len(patch.ids)))
            val.append((patch.weights[:, None] * z).ravel())
        ncols += q
    local_seconds = perf_counter() - start
    coarse_start = perf_counter()
    if ncols:
        P = sp.coo_matrix((np.concatenate(val),
                           (np.concatenate(row), np.concatenate(col))),
                          shape=(A.shape[0], ncols)).tocsr()
        ac = (P.T @ (A @ P)).tocsc()
        diagonal = ac.diagonal()
        if np.any(diagonal <= 0):
            raise la.LinAlgError("coarse basis contains a zero-energy direction")
        scale = 1. / np.sqrt(diagonal)
        P = (P @ sp.diags(scale)).tocsr()
        ac = (sp.diags(scale) @ ac @ sp.diags(scale)).tocsc()
        ac = ((ac + ac.T) * .5).tocsc()
        coarse_lu = sla.splu(ac)
        pt = P.T.tocsr()
    else:
        P, ac, coarse_lu, pt = None, None, None, None
    coarse_seconds = perf_counter() - coarse_start
    stats = {"setup_seconds": perf_counter() - start,
             "local_factor_seconds": local_seconds,
             "coarse_setup_seconds": coarse_seconds,
             "new_local_factors": newly_factored,
             "coarse_dimension": ncols,
             "coarse_nnz": 0 if ac is None else int(ac.nnz),
             "interpolation_nnz": 0 if P is None else int(P.nnz),
             "profile_enabled": bool(profile), "apply_calls": 0,
             "apply_local_seconds": 0., "apply_coarse_seconds": 0.,
             "local_apply_calls": 0, "coarse_apply_calls": 0}

    # Capture factors here, not through mutable future Patch cache lookups.
    local_solves = [(p.ids, p._sparse_local_factor.solve) for p in patches]

    def matvec(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        out = np.zeros(A.shape[0], dtype=np.float64)
        if profile:
            local_start = perf_counter()
        for ids, solve in local_solves:
            out[ids] += solve(r[ids])
        if profile:
            stats["apply_local_seconds"] += perf_counter() - local_start
            stats["local_apply_calls"] += len(local_solves)
            coarse_apply_start = perf_counter()
        if coarse_lu is not None:
            out += P @ coarse_lu.solve(pt @ r)
        if profile:
            stats["apply_coarse_seconds"] += perf_counter() - coarse_apply_start
            stats["coarse_apply_calls"] += int(coarse_lu is not None)
            stats["apply_calls"] += 1
        return out

    op = sla.LinearOperator(A.shape, matvec=matvec, rmatvec=matvec,
                            dtype=np.float64)
    stats["setup_seconds"] = perf_counter() - start
    return op, stats
