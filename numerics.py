"""P1 finite elements and verified two-level overlapping Schwarz components.

The coefficient array is indexed [y-cell, x-cell].  An n by n array means
n by n squares, split along their southwest/northeast diagonals.  Free
unknowns are interior vertices flattened in C order (x varies fastest).
All numerical acceptance checks are floating-point diagnostics, not
interval-arithmetic certificates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


_TRIANGLE_STIFFNESS = np.array(
    [[[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]],
     [[1., 0., -1.], [0., 1., -1.], [-1., -1., 2.]]]
) * 0.5


def _cell_triangles(ny: int, nx: int) -> np.ndarray:
    yy, xx = np.mgrid[:ny, :nx]
    sw = (yy * (nx + 1) + xx).ravel()
    se, nw = sw + 1, sw + nx + 1
    ne = nw + 1
    return np.stack((np.stack((sw, se, ne), axis=-1),
                     np.stack((sw, ne, nw), axis=-1)), axis=1)


def _full_cell_matrix(k: np.ndarray) -> sp.csr_matrix:
    ny, nx = k.shape
    tri = _cell_triangles(ny, nx)
    rows = np.broadcast_to(tri[..., :, None], (*tri.shape, 3))
    cols = np.broadcast_to(tri[..., None, :], (*tri.shape, 3))
    vals = k.ravel()[:, None, None, None] * _TRIANGLE_STIFFNESS[None]
    nn = (ny + 1) * (nx + 1)
    return sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())),
                         shape=(nn, nn)).tocsr()


def assemble_system(k: np.ndarray, f: Callable | None = None):
    """Return (A CSR, b, free_coordinates) with homogeneous Dirichlet data.

    ``f(x,y)`` must accept NumPy arrays.  The default is f=1.  Three-point
    degree-two triangle quadrature is used for a supplied right-hand side.
    """
    k = np.asarray(k, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 2:
        raise ValueError("k must be a square array with at least two cells per side")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("diffusion coefficient must be positive and finite")
    n = k.shape[0]
    full = _full_cell_matrix(k)
    yy, xx = np.mgrid[1:n, 1:n]
    free = (yy * (n + 1) + xx).ravel()
    coords = np.column_stack((xx.ravel() / n, yy.ravel() / n))
    if f is None:
        # Every interior vertex belongs to six triangles, area h^2/2 each.
        rhs = np.full(free.size, 1.0 / n**2, dtype=np.float64)
    else:
        tri = _cell_triangles(n, n).reshape(-1, 3)
        fy, fx = np.mgrid[0:n + 1, 0:n + 1]
        allcoords = np.column_stack((fx.ravel() / n, fy.ravel() / n))
        bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6],
                         [1/6, 1/6, 2/3]], dtype=np.float64)
        q = np.einsum("qv,tvd->tqd", bary, allcoords[tri])
        fq = np.asarray(f(q[..., 0], q[..., 1]), dtype=np.float64)
        fq = np.broadcast_to(fq, q.shape[:2])
        local_rhs = np.einsum("tq,qv->tv", fq, bary) / (6.0 * n**2)
        rhs_full = np.zeros((n + 1)**2)
        np.add.at(rhs_full, tri.ravel(), local_rhs.ravel())
        rhs = rhs_full[free]
    return full[free][:, free].tocsr(), rhs, coords


def free_to_grid(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """Include zero Dirichlet boundary vertices around a free-node vector."""
    x = np.asarray(x)
    if n is None:
        n = int(round(np.sqrt(x.size))) + 1
    if x.size != (n - 1)**2:
        raise ValueError("free vector has incompatible length")
    out = np.zeros((n + 1, n + 1), dtype=x.dtype)
    out[1:-1, 1:-1] = x.reshape(n - 1, n - 1)
    return out


@dataclass
class Patch:
    index: int
    ids: np.ndarray
    bounds: tuple[int, int, int, int]  # x0,x1,y0,y1; cells [x0,x1), vertices inclusive
    core_bounds: tuple[int, int, int, int]
    grid_shape: tuple[int, int]
    local_positions: tuple[np.ndarray, np.ndarray]  # (row, col) for every free id
    local_flat_indices: np.ndarray
    coefficient: np.ndarray
    weights: np.ndarray
    A_N: np.ndarray
    A_D: np.ndarray
    B: np.ndarray
    local_cholesky: tuple | None = field(default=None, repr=False)

    @property
    def x0(self): return self.bounds[0]
    @property
    def x1(self): return self.bounds[1]
    @property
    def y0(self): return self.bounds[2]
    @property
    def y1(self): return self.bounds[3]

    def to_grid(self, values: np.ndarray, pad_to: int | None = None) -> np.ndarray:
        values = np.asarray(values)
        shape = self.grid_shape if pad_to is None else (pad_to, pad_to)
        if shape[0] < self.grid_shape[0] or shape[1] < self.grid_shape[1]:
            raise ValueError("padding size is smaller than patch")
        out = np.zeros((*shape, *values.shape[1:]), dtype=values.dtype)
        out[self.local_positions] = values
        return out

    def from_grid(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[self.local_positions]


def build_patches(k: np.ndarray, A: sp.spmatrix, core_cells: int = 16,
                  overlap: int = 2, previous_patches: list[Patch] | None = None
                  ) -> list[Patch]:
    """Build overlapping subdomains and partition-of-unity weights.

    A_N includes only elements inside the patch (Neumann at artificial
    boundaries), while physical homogeneous Dirichlet nodes are removed.
    A_D is the principal submatrix of the current global A.  The GenEO-like
    metric is B=A_N+Xi A_D Xi.  An unchanged A_D can reuse a previous dense
    Cholesky factor; no factorization happens during preconditioner matvecs.
    """
    k = np.asarray(k, dtype=np.float64)
    n = k.shape[0]
    if core_cells < 1 or overlap < 0:
        raise ValueError("invalid subdomain size/overlap")
    specs = []
    counts = np.zeros((n - 1)**2, dtype=np.float64)
    for cy in range(0, n, core_cells):
        for cx in range(0, n, core_cells):
            cxe, cye = min(cx + core_cells, n), min(cy + core_cells, n)
            x0, x1 = max(0, cx-overlap), min(n, cxe+overlap)
            y0, y1 = max(0, cy-overlap), min(n, cye+overlap)
            yy, xx = np.mgrid[max(y0, 1):min(y1, n-1)+1,
                              max(x0, 1):min(x1, n-1)+1]
            ids = ((yy-1)*(n-1)+(xx-1)).ravel()
            pos = ((yy-y0).ravel(), (xx-x0).ravel())
            shape = (y1-y0+1, x1-x0+1)
            flat = np.ravel_multi_index(pos, shape)
            counts[ids] += 1
            specs.append((ids, (x0,x1,y0,y1), (cx,cxe,cy,cye), shape, pos, flat))
    if np.any(counts == 0):
        raise RuntimeError("subdomains do not cover every free node")
    result = []
    for index, (ids, bounds, core_bounds, shape, pos, flat) in enumerate(specs):
        x0,x1,y0,y1 = bounds
        coeff = k[y0:y1, x0:x1].copy()
        local_full = _full_cell_matrix(coeff)
        an = local_full[flat][:,flat].toarray()
        ad = A[ids][:,ids].toarray()
        w = 1.0 / counts[ids]
        bb = an + w[:,None] * ad * w[None,:]
        old_chol = None
        if previous_patches is not None and index < len(previous_patches):
            old = previous_patches[index]
            if np.array_equal(ids, old.ids) and np.array_equal(ad, old.A_D):
                old_chol = old.local_cholesky
        result.append(Patch(index, ids, bounds, core_bounds, shape, pos, flat,
                            coeff, w, an, ad, bb, old_chol))
    return result


def b_orthonormalize(Y: np.ndarray, B: np.ndarray, existing: np.ndarray | None = None,
                     rtol: float = 1e-10) -> np.ndarray:
    """Return a B-orthonormal independent basis, optionally orthogonal to Z.

    Two projection passes reduce loss of orthogonality.  General old bases
    are accepted; they need not already be normalized in the current B.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:,None]
    if Y.shape[1] == 0:
        return np.empty((B.shape[0],0))
    Y = Y.copy()
    original_scale = max(float(np.max(np.sum(Y * (B @ Y), axis=0))), 0.)
    if existing is not None and existing.shape[1]:
        Z = b_orthonormalize(existing, B, rtol=rtol)
        for _ in range(2):
            Y -= Z @ (Z.T @ (B @ Y))
    gram = Y.T @ B @ Y
    gram = (gram + gram.T) * .5
    ev, q = la.eigh(gram, check_finite=False)
    cutoff = rtol * max(original_scale, float(ev[-1]), np.finfo(float).tiny)
    keep = ev > cutoff
    if not np.any(keep):
        return np.empty((B.shape[0],0))
    Z = (Y @ q[:,keep]) / np.sqrt(ev[keep])[None,:]
    # A final small Cholesky improves normalization after ill-scaled inputs.
    gg = Z.T @ B @ Z
    ll = la.cholesky((gg+gg.T)*.5, lower=True, check_finite=False)
    return la.solve_triangular(ll, Z.T, lower=True, check_finite=False).T


def patch_spectral_basis(patch: Patch, gamma: float = 0.03,
                         anchor: float | None = 0.06) -> np.ndarray:
    """All generalized modes below an anchor >= requested acceptance gamma."""
    threshold = gamma if anchor is None else anchor
    if threshold < gamma or not 0 < threshold < 1:
        raise ValueError("require 0 < gamma <= anchor < 1")
    _, z = la.eigh(patch.A_N, patch.B, subset_by_value=(-np.inf,threshold),
                   driver="gvx", check_finite=False)
    return b_orthonormalize(z, patch.B)


def certify_space(A_N: np.ndarray, B: np.ndarray, Z: np.ndarray,
                   gamma: float = .03, tolerance: float = 5e-10,
                   method: str = "eigen") -> dict:
    """Numerically check the complete best-approximation inequality.

    The tested matrix is L=A_N-gamma*(B-BZ(Z'BZ)^(-1)Z'B).
    Diagonal congruence scaling avoids an absolute tolerance proportional
    to the global high-contrast spectral radius.  Testing only a Rayleigh
    quotient on the complement of Z would NOT be valid for learned Z.
    """
    t0 = perf_counter()
    z = b_orthonormalize(Z, B)
    bz = B @ z
    ll = A_N - gamma * B
    if z.shape[1]:
        ll = ll + gamma * (bz @ bz.T)
    diag = np.maximum(np.diag(A_N)+np.diag(B), np.finfo(float).tiny)
    inv_scale = 1. / np.sqrt(diag)
    scaled = ll * inv_scale[:,None] * inv_scale[None,:]
    scaled = .5*(scaled+scaled.T)
    details = {"method": method, "eigen_fallback": False}
    value = None
    if method == "eigen":
        value = float(la.eigh(scaled, subset_by_index=(0,0),
                              eigvals_only=True, check_finite=False)[0])
        passed = value >= -tolerance
    elif method == "ldl":
        # Congruence preserves inertia, not eigenvalues.  Shift BEFORE the
        # factorization so tolerance has precisely its eigen-test meaning.
        shifted = scaled.copy()
        shifted.flat[::len(shifted)+1] += tolerance
        _, dd, _ = la.ldl(shifted, lower=True, check_finite=False,
                          overwrite_a=True)
        pivots = []
        j = 0
        while j < len(dd):
            if j + 1 < len(dd) and dd[j,j+1] != 0.:
                pivots.extend(la.eigvalsh(dd[j:j+2,j:j+2],
                                         check_finite=False).tolist())
                j += 2
            else:
                pivots.append(float(dd[j,j]))
                j += 1
        pivots = np.asarray(pivots)
        pivot_tol = 16*np.finfo(float).eps*len(dd)*max(1.,float(np.max(np.abs(dd))))
        details.update(min_pivot=float(np.min(pivots)),
                       negative_pivots=int(np.sum(pivots < -pivot_tol)),
                       ambiguous_pivots=int(np.sum(np.abs(pivots) <= pivot_tol)),
                       pivot_tolerance=float(pivot_tol))
        if details["ambiguous_pivots"]:
            # Avoid assigning a sign to a roundoff-scale pivot.  This also
            # handles tolerance=0 with exact Neumann null modes.
            value = float(la.eigh(scaled, subset_by_index=(0,0),
                                  eigvals_only=True, check_finite=False)[0])
            passed = value >= -tolerance
            details["eigen_fallback"] = True
        else:
            passed = details["negative_pivots"] == 0
    else:
        raise ValueError("method must be 'eigen' or 'ldl'")
    return {"passed": bool(passed), "min_eig": value,
            "tolerance": tolerance, "dimension": int(z.shape[1]),
            "seconds": perf_counter()-t0, **details}


def build_preconditioner(A: sp.spmatrix, patches: list[Patch],
                         spaces: list[np.ndarray]):
    """Construct fixed SPD two-level additive Schwarz LinearOperator.

    Coarse interpolation P is sparse, with local columns Xi*Z_i.  Its
    Galerkin matrix is sparse and factorized once.  All local factors are
    computed once and cached on Patch objects.  Returns (operator,stats).
    """
    start = perf_counter()
    if len(patches) != len(spaces):
        raise ValueError("one local space is required per patch")
    row, col, val = [], [], []
    ncols = 0
    newly_factored = 0
    for patch,z in zip(patches,spaces):
        if patch.local_cholesky is None:
            patch.local_cholesky = la.cho_factor(patch.A_D, lower=True,
                                                check_finite=False)
            newly_factored += 1
        z = np.asarray(z, dtype=np.float64)
        if z.shape[0] != len(patch.ids):
            raise ValueError("basis has incompatible local node count")
        q = z.shape[1]
        if q:
            row.append(np.repeat(patch.ids,q))
            col.append(np.tile(np.arange(ncols,ncols+q),len(patch.ids)))
            val.append((patch.weights[:,None]*z).ravel())
        ncols += q
    local_seconds = perf_counter()-start
    coarse_start = perf_counter()
    if ncols:
        P = sp.coo_matrix((np.concatenate(val),(np.concatenate(row),np.concatenate(col))),
                           shape=(A.shape[0],ncols)).tocsr()
        AP = A @ P
        ac = (P.T @ AP).tocsc()
        d = ac.diagonal()
        if np.any(d <= 0):
            raise la.LinAlgError("coarse basis contains a zero-energy direction")
        scale = 1/np.sqrt(d)
        P = (P @ sp.diags(scale)).tocsr()
        ac = (sp.diags(scale) @ ac @ sp.diags(scale)).tocsc()
        ac = ((ac+ac.T)*.5).tocsc()
        coarse_lu = sla.splu(ac)
        pt = P.T.tocsr()
    else:
        P, ac, coarse_lu, pt = None, None, None, None
    coarse_seconds = perf_counter()-coarse_start
    def matvec(r):
        out = np.zeros(A.shape[0],dtype=np.float64)
        for patch in patches:
            out[patch.ids] += la.cho_solve(patch.local_cholesky, r[patch.ids],
                                          check_finite=False)
        if coarse_lu is not None:
            out += P @ coarse_lu.solve(pt @ r)
        return out
    op = sla.LinearOperator(A.shape, matvec=matvec, rmatvec=matvec, dtype=np.float64)
    stats = {"setup_seconds":perf_counter()-start,
             "local_factor_seconds":local_seconds,
             "coarse_setup_seconds":coarse_seconds,
             "new_local_factors":newly_factored,
             "coarse_dimension":ncols,
             "coarse_nnz":0 if ac is None else int(ac.nnz),
             "interpolation_nnz":0 if P is None else int(P.nnz)}
    return op,stats


def solve_pcg(A, b, M=None, x0=None, rtol=1e-8, maxiter=1000,
               residual_refresh=True, max_restarts=3):
    """Solve with fixed M and judge success using the true residual.

    Optional residual-refresh continuations counter the recursive/true
    residual gap.  ``maxiter`` bounds the TOTAL iteration budget, and all
    refreshes and explicit residual calculations are included in timing.
    No refresh can turn an above-threshold true residual into success.
    """
    iterations = 0
    def callback(_):
        nonlocal iterations
        iterations += 1
    start = perf_counter()
    x,info = sla.cg(A,b,M=M,x0=x0,rtol=rtol,atol=0.,maxiter=maxiter,
                    callback=callback)
    initial_info = int(info)
    bnorm = max(np.linalg.norm(b),np.finfo(float).tiny)
    residual = float(np.linalg.norm(b-A@x)/bnorm)
    initial_residual = residual
    restarts = 0
    while (residual_refresh and residual > rtol and restarts < max_restarts
           and iterations < maxiter and info >= 0):
        restarts += 1
        x,info = sla.cg(A,b,M=M,x0=x,rtol=rtol*(.2**restarts),atol=0.,
                        maxiter=maxiter-iterations,callback=callback)
        residual = float(np.linalg.norm(b-A@x)/bnorm)
    elapsed = perf_counter()-start
    return x,{"iterations":iterations,"info":int(info),"seconds":elapsed,
              "initial_info":initial_info,"initial_relative_residual":initial_residual,
              "restarts":restarts,
              "relative_residual":residual,
              "converged":bool(residual <= rtol)}


def energy_relative_error(A, x, reference):
    delta = np.asarray(x)-reference
    return float(np.sqrt(max(float(delta @ (A @ delta)),0.) /
                         max(float(reference @ (A @ reference)),np.finfo(float).tiny)))
