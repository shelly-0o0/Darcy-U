"""Computable spectral increments and full-matrix capacity diagnostics.

``spectral_increment`` uses the actual current local matrices, so its cost
includes a local generalized eigensolve.  It is a reference for learned
directions, NOT an optimal rank-constrained correction or a free algorithm.
All statements about acceptance are floating-point diagnostics.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.linalg as la

from numerics import Patch, b_orthonormalize


def _validate_matrices(A_N, B):
    a, b = np.asarray(A_N, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("A_N and B must be square matrices of equal shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("local matrices must be finite")
    return a, b


def spectral_increment(patch: Patch, old_space: np.ndarray,
                       gamma: float = .03, budget: int | None = 4,
                       anchor: float | None = None,
                       rank_rtol: float = 1e-10) -> tuple[np.ndarray, dict]:
    """Retain the old space and append independent low-energy residual modes.

    We compute every generalized eigenvector with eigenvalue <= ``anchor``
    (default 2*gamma).  In increasing eigenvalue order, project each mode
    twice against the old B-orthonormal space and previously accepted
    residual directions.  Accept its normalized residual when its squared
    B norm exceeds ``rank_rtol`` (input modes have B norm one).  A finite
    budget caps the number of new directions, not the retained old rank.

    The order includes the Neumann zero mode before positive eigenvalues.
    Unlike a truncated SVD mixture of modes, this rule does not needlessly
    discard the exact kernel direction when the budget is sufficient.
    Degenerate eigenvectors use LAPACK's deterministic local ordering; the
    selected subspace is invariant to an invertible change of the old basis
    up to roundoff and numerical rank decisions.

    ``budget=None`` spans the union of the old space and all selected low
    modes, up to the stated rank tolerance.  It therefore satisfies the full
    PSD inequality in exact arithmetic when the low space is represented
    exactly and anchor >= gamma.  The caller must still check it numerically.

    A budgeted result is not claimed to minimize rank or maximize acceptance.
    Use its measured construction_seconds as real algorithm cost; subtract
    that cost only for a clearly marked hypothetical free-construction
    diagnostic, never for an actual speedup claim.
    """
    started = perf_counter()
    a, b = _validate_matrices(patch.A_N, patch.B)
    threshold = 2. * gamma if anchor is None else float(anchor)
    if not 0 < gamma <= threshold < 1:
        raise ValueError("require 0 < gamma <= anchor < 1")
    if budget is not None and (not isinstance(budget, (int, np.integer)) or budget < 0):
        raise ValueError("budget must be a nonnegative integer or None")
    if not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie between zero and one")
    old_input = np.asarray(old_space, dtype=np.float64)
    if old_input.ndim != 2 or old_input.shape[0] != len(a):
        raise ValueError("old_space must have shape (local_dimension, old_rank)")
    old = b_orthonormalize(old_input, b, rtol=rank_rtol)
    spectral_start = perf_counter()
    values, vectors = la.eigh(a, b, subset_by_value=(-np.inf, threshold),
                             driver="gvx", check_finite=False)
    spectral_seconds = perf_counter() - spectral_start
    added, selected_values, residual_norms = [], [], []
    visited = 0
    for j, value in enumerate(values):
        if budget is not None and len(added) >= budget:
            break
        visited += 1
        v = vectors[:, j].copy()
        basis = np.column_stack((old, *added)) if added else old
        for _ in range(2):
            if basis.shape[1]:
                v -= basis @ (basis.T @ (b @ v))
        norm_squared = float(v @ (b @ v))
        if norm_squared <= rank_rtol:
            continue
        v /= np.sqrt(norm_squared)
        # Fix signs for reproducible output without changing the space.
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
        added.append(v)
        selected_values.append(float(value))
        residual_norms.append(norm_squared)
    combined = np.column_stack((old, *added)) if added else old.copy()
    elapsed = perf_counter() - started
    return combined, {
        "reference_kind": "ordered_exact_spectral_residuals",
        "optimal_rank_q_claim": False,
        "rank_budget": None if budget is None else int(budget),
        "gamma": float(gamma),
        "anchor": threshold,
        "rank_rtol": float(rank_rtol),
        "old_rank": int(old.shape[1]),
        "added_rank": len(added),
        "candidate_rank": int(combined.shape[1]),
        "low_mode_count": int(len(values)),
        "visited_low_mode_count": visited,
        "selected_eigenvalues": selected_values,
        "selected_residual_norm_squared": residual_norms,
        "construction_seconds": elapsed,
        "spectral_seconds": spectral_seconds,
        "other_construction_seconds": elapsed - spectral_seconds,
    }


def diagnose_space(A_N: np.ndarray, B: np.ndarray, Z: np.ndarray,
                   gamma: float = .03, tolerance: float = 5e-10) -> dict:
    """Report the complete scaled PSD-check spectrum and capacity lower bound.

    The check is L=A_N-gamma*(B-BZ(Z'BZ)^(-1)Z'B), not a complementary
    Rayleigh test.  Scaling is the same diagonal congruence as certify_space.
    In exact arithmetic, adding q independent B-orthogonal directions adds
    a positive semidefinite matrix of rank at most q.  Such an update cannot
    eliminate more than q negative eigenvalues (congruence preserves inertia).
    Thus n_neg(L_old)>q proves an insufficient rank budget in exact arithmetic.
    Here eigenvalues below -tolerance are counted numerically; tolerance and
    roundoff mean this is a diagnostic, not an interval-arithmetic proof.
    Having at most q negative eigenvalues does NOT guarantee that a proposed
    q-dimensional correction passes the full PSD test.
    """
    started = perf_counter()
    a, b = _validate_matrices(A_N, B)
    if not 0 < gamma < 1 or tolerance < 0:
        raise ValueError("require 0 < gamma < 1 and nonnegative tolerance")
    z = b_orthonormalize(Z, b)
    bz = b @ z
    ll = a - gamma * b
    if z.shape[1]:
        ll = ll + gamma * (bz @ bz.T)
    diagonal = np.maximum(np.diag(a) + np.diag(b), np.finfo(float).tiny)
    inv_scale = 1. / np.sqrt(diagonal)
    scaled = ll * inv_scale[:, None] * inv_scale[None, :]
    values = la.eigvalsh(.5 * (scaled + scaled.T), check_finite=False)
    negative_count = int(np.count_nonzero(values < -tolerance))
    return {
        "passed": negative_count == 0,
        "min_eig": float(values[0]),
        "scaled_eigenvalues": values.tolist(),
        "significant_negative_count": negative_count,
        "negative_count": negative_count,
        "necessary_added_rank_lower_bound": negative_count,
        "ambiguous_eigenvalue_count": int(np.count_nonzero(np.abs(values) <= tolerance)),
        "tolerance": float(tolerance),
        "scaling": "diagonal_congruence_diag_A_N_plus_B",
        "floating_point_diagnostic": True,
        "seconds": perf_counter() - started,
    }
