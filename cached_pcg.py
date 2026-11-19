"""PCG recycling from actions already paid for by a fixed SPD solve.

For an iterate increment D_j=alpha_j*p_j, AD_j=alpha_j*A*p_j and
MAD_j=z_j-z_{j+1}, because z=M*r and r_{j+1}=r_j-AD_j.  The last
increment of each residual-refresh segment is omitted when the next M*r
was not needed by PCG.  No A/M action is added for collection or Ritz
extraction.  A and M must be real, fixed, symmetric positive definite;
using caches collected under a different M is not valid.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _pcg_segment(A, b, M, x, rtol, maxiter, collector, counts, segment):
    """One real PCG recurrence, with no residual refresh inside the segment."""
    bnorm = np.linalg.norm(b)
    if bnorm == 0.:
        return np.zeros_like(b), 0, 0
    if x.any():
        r = b - A @ x
        counts["a_applications"] += 1
    else:
        r = b.copy()
    rho_previous, p = None, None
    pending = None
    iterations = 0
    for iteration in range(maxiter):
        if np.linalg.norm(r) < rtol * bnorm:
            if pending is not None:
                collector["skipped_terminal_vectors"] += 1
            return x, 0, iterations
        z = np.asarray(M @ r, dtype=np.float64).reshape(-1)
        counts["m_applications"] += 1
        if pending is not None:
            d, ad, old_z = pending
            mad = old_z - z
            if np.all(np.isfinite(mad)) and np.any(d != 0.):
                collector["D"].append(d)
                collector["AD"].append(ad)
                collector["MAD"].append(mad)
                collector["segment_ids"].append(int(segment))
            else:
                collector["rejected_vectors"] += 1
            pending = None
        rho = float(r @ z)
        if not np.isfinite(rho) or rho <= 0.:
            return x, -1, iterations
        if iteration:
            p *= rho / rho_previous
            p += z
        else:
            p = z.copy()
        ap = np.asarray(A @ p, dtype=np.float64).reshape(-1)
        counts["a_applications"] += 1
        curvature = float(p @ ap)
        if not np.isfinite(curvature) or curvature <= 0.:
            return x, -2, iterations
        alpha = rho / curvature
        if len(collector["D"]) < collector["max_vectors"]:
            d, ad = alpha * p, alpha * ap
            if np.all(np.isfinite(d)) and np.all(np.isfinite(ad)):
                # M may return a view of its argument, so preserve z before
                # mutating r; D and AD already own their multiplication buffers.
                pending = (d, ad, z.copy())
            else:
                collector["rejected_vectors"] += 1
            x += d
            r -= ad
        else:
            x += alpha * p
            r -= alpha * ap
        rho_previous = rho
        iterations += 1
    if pending is not None:
        collector["skipped_terminal_vectors"] += 1
    return x, int(maxiter), iterations


def solve_pcg_cached(A, b, M=None, x0=None, rtol=1e-8, maxiter=500,
                     max_vectors=24):
    """Return ``(x, metrics, cache)`` using real double-precision PCG.

    ``cache['D']``, ``['AD']`` and ``['MAD']`` are (n,k) column arrays,
    limited to the first ``max_vectors`` valid completed pairs over all
    residual-refresh segments.  ``max_vectors=0`` runs the same recurrence
    without collection, providing a solver-implementation control.

    Success always uses a freshly computed true residual.  As in
    numerics.solve_pcg, at most three continuations use rtol*.2**restart
    within one total maxiter budget.  Each segment starts a new recurrence;
    no cached difference crosses a true-residual refresh.  Timing includes
    all input conversion, copies, cache construction, and true checks.

    Positive info means the latest segment exhausted its iteration budget;
    -1/-2 indicate a nonpositive/nonfinite preconditioned norm/curvature.
    The converged flag, not info alone, is the final acceptance criterion.
    """
    start = perf_counter()
    max_vectors = _integer(max_vectors, "max_vectors")
    maxiter = _integer(maxiter, "maxiter", 1)
    if not np.isfinite(rtol) or rtol <= 0.:
        raise ValueError("rtol must be finite and positive")
    if np.iscomplexobj(b) or (x0 is not None and np.iscomplexobj(x0)):
        raise ValueError("solve_pcg_cached supports real inputs only")
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    A = sla.aslinearoperator(A)
    M = sla.aslinearoperator(M) if M is not None else sla.LinearOperator(
        A.shape, matvec=lambda v: v, dtype=np.float64)
    if A.shape != (b.size, b.size) or M.shape != A.shape or x.shape != b.shape:
        raise ValueError("A, M, b, and x0 must have compatible shapes")
    if np.issubdtype(A.dtype, np.complexfloating) or np.issubdtype(M.dtype, np.complexfloating):
        raise ValueError("solve_pcg_cached supports real operators only")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x)):
        raise ValueError("b and x0 must be finite")
    collector = {"D": [], "AD": [], "MAD": [], "max_vectors": max_vectors,
                 "segment_ids": [], "skipped_terminal_vectors": 0,
                 "rejected_vectors": 0}
    counts = {"a_applications": 0, "m_applications": 0}
    iterations = 0
    restarts = 0
    x, info, used = _pcg_segment(A, b, M, x, rtol, maxiter,
                                  collector, counts, restarts)
    iterations += used
    initial_info = int(info)
    bnorm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residual = float(np.linalg.norm(b - A @ x) / bnorm)
    counts["a_applications"] += 1
    initial_residual = residual
    while (residual > rtol and restarts < 3 and iterations < maxiter and info >= 0):
        restarts += 1
        x, info, used = _pcg_segment(A, b, M, x, rtol * .2 ** restarts,
                                      maxiter - iterations, collector, counts, restarts)
        iterations += used
        residual = float(np.linalg.norm(b - A @ x) / bnorm)
        counts["a_applications"] += 1
    cache = {key: np.column_stack(collector[key]) if collector[key]
             else np.empty((b.size, 0), dtype=np.float64)
             for key in ("D", "AD", "MAD")}
    cache["stats"] = {key: value for key, value in collector.items()
                      if key not in ("D", "AD", "MAD")}
    cache["stats"].update(counts, captured_vectors=int(cache["D"].shape[1]),
                           extra_a_applications_for_cache=0,
                           extra_m_applications_for_cache=0,
                           solver="cached_real_spd_pcg")
    return x, {"iterations": int(iterations), "info": int(info),
               "initial_info": initial_info, "initial_relative_residual": initial_residual,
               "restarts": int(restarts), "relative_residual": residual,
               "converged": bool(residual <= rtol),
               "captured_vectors": int(cache["D"].shape[1]),
               "seconds": perf_counter() - start}, cache


def _asymmetry(matrix):
    return float(la.norm(matrix - matrix.T, ord="fro") /
                 max(la.norm(matrix, ord="fro"), np.finfo(float).tiny))


def cached_slow_ritz(D, AD, MAD, q=2, rtol=1e-10, symmetry_tol=1e-7):
    """Extract slow MA Ritz directions without any new A or M action.

    Solve H*y=lambda*G*y with G=D.T*AD and H=AD.T*MAD after consistent
    scaling and rank-revealing whitening of all three caches.  Retained
    columns are ordered by increasing strictly positive Ritz value.
    All input arrays must describe the *same fixed* A and M.

    Relative Gram/H asymmetry above symmetry_tol rejects the entire cache
    with an empty basis and explicit status.  Symmetrization only removes
    smaller floating-point deviations.  This is a diagnostic safeguard,
    not a certificate that arbitrary supplied caches equal true actions.
    Significant indefinite Gram/H matrices are likewise rejected.

    projected_ritz_residuals measure the A-inner-product Galerkin equations;
    operator_ritz_residuals measure Euclidean MA*u-lambda*u using MAD.
    A full-space A norm of this residual cannot be obtained from these
    caches alone and is deliberately not reported.
    """
    q = _integer(q, "q", 1)
    if not np.isfinite(rtol) or not 0. < rtol < 1.:
        raise ValueError("rtol must lie strictly between zero and one")
    if not np.isfinite(symmetry_tol) or not 0. < symmetry_tol < 1.:
        raise ValueError("symmetry_tol must lie strictly between zero and one")
    if any(np.iscomplexobj(a) for a in (D, AD, MAD)):
        raise ValueError("cached_slow_ritz supports real inputs only")
    D, AD, MAD = [np.asarray(a, dtype=np.float64) for a in (D, AD, MAD)]
    if D.ndim != 2 or AD.shape != D.shape or MAD.shape != D.shape:
        raise ValueError("D, AD, and MAD must have the same shape (n,k)")
    empty = np.empty((D.shape[0], 0), dtype=np.float64)
    stats = {"input_vectors": int(D.shape[1]), "valid_vectors": 0,
             "rejected_vectors": 0, "dependent_vectors": 0,
             "requested_rank": q, "basis_rank": 0, "rank": 0,
             "ritz_values": [], "projected_ritz_residuals": [],
             "operator_ritz_residuals": [], "orthogonality_error": 0.,
             "gram_relative_asymmetry": 0., "h_relative_asymmetry": 0.,
             "retained_condition": 0., "nonpositive_ritz_values": 0,
             "a_applications": 0, "m_applications": 0,
             "rtol": float(rtol), "symmetry_tol": float(symmetry_tol),
             "status": "empty_basis"}

    def reject(status):
        stats["status"] = status
        return empty.copy(), empty.copy(), stats

    triples = []
    for d, ad, mad in zip(D.T, AD.T, MAD.T):
        magnitude = np.max(np.abs(d), initial=0.)
        if magnitude <= 0. or not np.isfinite(magnitude) \
                or not all(np.all(np.isfinite(v)) for v in (d, ad, mad)):
            stats["rejected_vectors"] += 1
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            d, ad, mad = d / magnitude, ad / magnitude, mad / magnitude
            energy = float(d @ ad)
            if np.isfinite(energy) and energy > 0.:
                norm = np.sqrt(energy)
                d, ad, mad = d / norm, ad / norm, mad / norm
            else:
                stats["rejected_vectors"] += 1
                continue
        if not all(np.all(np.isfinite(v)) for v in (d, ad, mad)):
            stats["rejected_vectors"] += 1
            continue
        triples.append((d, ad, mad))
    stats["valid_vectors"] = len(triples)
    if not triples:
        return reject("no_valid_direction" if D.shape[1] else "empty_basis")
    V, AV, MAV = [np.column_stack([t[i] for t in triples]) for i in range(3)]
    raw_G, raw_H = V.T @ AV, AV.T @ MAV
    if not np.all(np.isfinite(raw_G)) or not np.all(np.isfinite(raw_H)):
        return reject("nonfinite_small_matrices")
    stats["gram_relative_asymmetry"] = _asymmetry(raw_G)
    stats["h_relative_asymmetry"] = _asymmetry(raw_H)
    if max(stats["gram_relative_asymmetry"], stats["h_relative_asymmetry"]) > symmetry_tol:
        return reject("cache_symmetry_mismatch")
    G = .5 * (raw_G + raw_G.T)
    values, modes = la.eigh(G, check_finite=False)
    cutoff = rtol * max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values[0] < -cutoff:
        return reject("indefinite_gram")
    keep = values > cutoff
    if not np.any(keep):
        return reject("no_positive_gram_mode")
    stats["retained_condition"] = float(values[keep][-1] / values[keep][0])
    transform = modes[:, keep] / np.sqrt(values[keep])[None, :]
    V, AV, MAV = V @ transform, AV @ transform, MAV @ transform
    refined_G = V.T @ AV
    refined_G = .5 * (refined_G + refined_G.T)
    gvalues, gmodes = la.eigh(refined_G, check_finite=False)
    gkeep = gvalues > rtol * max(float(gvalues[-1]), np.finfo(float).tiny)
    if not np.any(gkeep):
        return reject("no_positive_refined_mode")
    refine = gmodes[:, gkeep] / np.sqrt(gvalues[gkeep])[None, :]
    V, AV, MAV = V @ refine, AV @ refine, MAV @ refine
    stats["basis_rank"] = int(V.shape[1])
    stats["dependent_vectors"] = len(triples) - int(V.shape[1])
    H, G = AV.T @ MAV, V.T @ AV
    stats["whitened_gram_relative_asymmetry"] = _asymmetry(G)
    stats["whitened_h_relative_asymmetry"] = _asymmetry(H)
    if max(stats["whitened_gram_relative_asymmetry"],
           stats["whitened_h_relative_asymmetry"]) > symmetry_tol:
        return reject("whitened_cache_symmetry_mismatch")
    H, G = .5 * (H + H.T), .5 * (G + G.T)
    values, modes = la.eigh(H, G, check_finite=False)
    stats["nonpositive_ritz_values"] = int(np.count_nonzero(values <= 0.))
    cutoff = rtol * max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values[0] < -cutoff:
        return reject("indefinite_ritz_matrix")
    selected = np.flatnonzero(values > 0.)[:q]
    if not len(selected):
        return reject("no_positive_ritz_value")
    C, eigs = modes[:, selected], values[selected]
    U, AU, MAU = V @ C, AV @ C, MAV @ C
    projected = H @ C - (G @ C) * eigs[None, :]
    pscale = la.norm(H @ C, axis=0) + np.abs(eigs) * la.norm(G @ C, axis=0)
    residual = MAU - U * eigs[None, :]
    oscale = la.norm(MAU, axis=0) + np.abs(eigs) * la.norm(U, axis=0)
    tiny = np.finfo(float).tiny
    stats.update(status="accepted", rank=int(len(selected)), ritz_values=eigs.tolist(),
                 projected_ritz_residuals=(la.norm(projected, axis=0) /
                                           np.maximum(pscale, tiny)).tolist(),
                 operator_ritz_residuals=(la.norm(residual, axis=0) /
                                          np.maximum(oscale, tiny)).tolist(),
                 orthogonality_error=float(la.norm(U.T @ AU - np.eye(len(selected)), ord=2)))
    return U, AU, stats
