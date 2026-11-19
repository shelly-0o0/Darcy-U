"""Current-matrix balanced augmentation and causal Ritz recycling.

All direction arrays have shape ``(n, number_of_vectors)``; columns are
chronological when an explicit basis limit is used. These utilities accept
only matrices, preconditioners, right-hand sides, and supplied directions.
They neither solve for a reference solution nor read data from other steps.
Callers must time the complete setup, extraction, and application operations.

For SPD A and a fixed SPD inverse-preconditioner M, an A-orthonormal U gives
M_bal = U U.T + (I - U U.T A) M (I - A U U.T).
This is SPD, fixes the selected directions exactly (M_bal A U = U), and
remains a fixed linear operator throughout an ordinary PCG solve. Rebuild
U and A U with the *current* A whenever the system changes.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla


def _columns(directions, n):
    if directions is None:
        return np.empty((n, 0), dtype=np.float64)
    result = np.asarray(directions, dtype=np.float64)
    if result.ndim == 1:
        if result.size == 0:
            return np.empty((n, 0), dtype=np.float64)
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != n:
        raise ValueError("directions must have shape (n, k), with vectors in columns")
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def a_orthonormalize(A, directions, rtol=1e-10, max_vectors=None):
    """Return ``(U, AU, stats)`` after scaling and rank-revealing whitening.

    Invalid, zero, and nonpositive-energy columns are rejected individually.
    Max-magnitude and then A-energy normalization prevent dependence on the
    supplied vector units. Eigenvalues of the symmetrized small Gram matrix
    no greater than ``rtol * largest_eigenvalue`` are discarded. A second
    whitening pass removes roundoff left by a nearly dependent input basis;
    it reuses cached A actions rather than applying A again.

    ``max_vectors`` keeps the last columns, so an online caller can enforce
    a fixed memory limit. Truncation may remove parts of the original span.
    No claim is made that discarded near-dependent vectors are unimportant.
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.isfinite(rtol) or not 0. < rtol < 1.:
        raise ValueError("rtol must lie strictly between zero and one")
    n = A.shape[0]
    supplied = _columns(directions, n)
    if max_vectors is not None:
        max_vectors = _positive_integer(max_vectors, "max_vectors")
        selected = supplied[:, -max_vectors:]
    else:
        selected = supplied
    stats = {"input_vectors": int(supplied.shape[1]),
             "selected_vectors": int(selected.shape[1]), "valid_vectors": 0,
             "rejected_vectors": 0, "dependent_vectors": 0, "rank": 0,
             "retained_condition": 0., "orthogonality_error": 0.,
             "rtol": float(rtol), "status": "empty_basis"}
    empty = np.empty((n, 0), dtype=np.float64)
    vectors, actions = [], []
    for column in selected.T:
        magnitude = np.max(np.abs(column), initial=0.)
        if not np.all(np.isfinite(column)) or not np.isfinite(magnitude) \
                or magnitude <= 0.:
            stats["rejected_vectors"] += 1
            continue
        vector = column / magnitude
        action = np.asarray(A @ vector, dtype=np.float64).reshape(-1)
        with np.errstate(over="ignore", invalid="ignore"):
            energy = float(vector @ action)
        if action.shape != (n,) or not np.all(np.isfinite(action)) \
                or not np.isfinite(energy) or energy <= 0.:
            stats["rejected_vectors"] += 1
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            norm = np.sqrt(energy)
            normalized, normalized_action = vector / norm, action / norm
        if not np.all(np.isfinite(normalized)) \
                or not np.all(np.isfinite(normalized_action)):
            stats["rejected_vectors"] += 1
            continue
        vectors.append(normalized)
        actions.append(normalized_action)
    stats["valid_vectors"] = len(vectors)
    if not vectors:
        if selected.shape[1]:
            stats["status"] = "no_valid_direction"
        return empty.copy(), empty.copy(), stats
    V, AV = np.column_stack(vectors), np.column_stack(actions)
    gram = V.T @ AV
    gram = .5 * (gram + gram.T)
    if not np.all(np.isfinite(gram)):
        stats["status"] = "nonfinite_gram"
        return empty.copy(), empty.copy(), stats
    values, modes = la.eigh(gram, check_finite=False)
    keep = values > rtol * max(float(values[-1]), np.finfo(float).tiny)
    stats["dependent_vectors"] = int(len(values) - np.count_nonzero(keep))
    if not np.any(keep):
        stats["status"] = "no_positive_gram_mode"
        return empty.copy(), empty.copy(), stats
    retained = values[keep]
    transform = modes[:, keep] / np.sqrt(retained)[None, :]
    U, AU = V @ transform, AV @ transform
    refined_gram = U.T @ AU
    refined_gram = .5 * (refined_gram + refined_gram.T)
    if not np.all(np.isfinite(refined_gram)):
        stats["status"] = "nonfinite_whitening"
        return empty.copy(), empty.copy(), stats
    refined_values, refined_modes = la.eigh(refined_gram, check_finite=False)
    refined_keep = refined_values > rtol * max(float(refined_values[-1]),
                                              np.finfo(float).tiny)
    if not np.any(refined_keep):
        stats["status"] = "no_positive_refined_mode"
        return empty.copy(), empty.copy(), stats
    refine = refined_modes[:, refined_keep] / np.sqrt(refined_values[refined_keep])[None, :]
    U, AU = U @ refine, AU @ refine
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(AU)):
        stats["status"] = "nonfinite_whitening"
        return empty.copy(), empty.copy(), stats
    stats.update(rank=int(U.shape[1]), status="accepted",
                 retained_condition=float(retained[-1] / retained[0]),
                 dependent_vectors=int(len(vectors) - U.shape[1]),
                 orthogonality_error=float(la.norm(U.T @ AU - np.eye(U.shape[1]), ord=2)))
    return U, AU, stats


def _validate_basis(U, AU, n=None):
    U = np.asarray(U, dtype=np.float64)
    AU = np.asarray(AU, dtype=np.float64)
    if U.ndim != 2 or AU.shape != U.shape or (n is not None and U.shape[0] != n):
        raise ValueError("U and AU must have the same shape (n, rank)")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(AU)):
        raise ValueError("U and AU must be finite")
    if U.shape[1] and not np.allclose(U.T @ AU, np.eye(U.shape[1]),
                                    rtol=1e-7, atol=1e-7):
        raise ValueError("U must be A-orthonormal for the supplied AU")
    return U, AU


def balanced_preconditioner(M, U, AU):
    """Return a fixed SPD LinearOperator, caching owned U and A U arrays.

    M must itself be fixed, symmetric, and positive definite; these global
    properties cannot be certified cheaply and are the caller's contract.
    Each application uses one M application and dense rank-sized products,
    with no sparse A application. With rank zero, M is returned unchanged
    if it is already a LinearOperator. Caller mutation of input U/AU does
    not change the constructed operator.
    """
    M = sla.aslinearoperator(M)
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    U, AU = _validate_basis(U, AU, n=M.shape[0])
    if U.shape[1] == 0:
        return M
    U, AU = U.copy(), AU.copy()
    U.flags.writeable = AU.flags.writeable = False

    def apply(vector):
        coefficients = U.T @ vector
        projected = vector - AU @ coefficients
        smoothed = M @ projected
        return U @ coefficients + smoothed - U @ (AU.T @ smoothed)

    return sla.LinearOperator(M.shape, matvec=apply, matmat=apply,
                              rmatvec=apply, dtype=np.dtype(np.float64))


def galerkin_correction(A, b, x0, U, AU=None):
    """Apply x += U U.T (b-A x) using a current-A orthonormal basis.

    Return the corrected vector and diagnostics. The energy change is the
    change of J(x)=.5*x.T*A*x-b.T*x. No reference solution is needed. AU may
    be passed to reuse the setup cache; it must correspond to the current A.
    Nonfinite supplied b/x0 is an error, not a silent solution replacement.
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    if A.shape != (len(b), len(b)) or x.shape != b.shape:
        raise ValueError("A, b, and x0 must have compatible dimensions")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x)):
        raise ValueError("b and x0 must be finite")
    if AU is None:
        U = _columns(U, len(b))
        AU = A @ U
    U, AU = _validate_basis(U, AU, n=len(b))
    info = {"rank": int(U.shape[1]), "energy_change": 0., "status": "empty_basis"}
    if not U.shape[1]:
        return x, info
    residual = b - A @ x
    coefficients = U.T @ residual
    delta = U @ coefficients
    action = AU @ coefficients
    candidate = x + delta
    change = float(.5 * delta @ action - residual @ delta)
    if not np.all(np.isfinite(candidate)) or not np.isfinite(change) or change > 0.:
        info["status"] = "nonfinite_or_increased_energy"
        return x, info
    info.update(energy_change=change, status="accepted")
    return candidate, info


def extract_slow_ritz(A, M, P, q=4, max_basis=32, rtol=1e-10):
    """Extract up to q slow positive Ritz directions from supplied past P.

    This is Rayleigh-Ritz for M A in the A inner product. Equivalently, in
    the original P coordinates it solves H y=lambda G y with
    H=(A P).T M (A P), G=P.T A P. Rank-revealing A whitening stabilizes G
    before the small generalized symmetric eigenproblem is solved. The
    lowest *strictly positive* Ritz values are selected; a tiny positive
    value is not removed solely because another eigenvalue is large.

    P must come from an already computed solve. A/M define the fixed
    operator being probed; the solve may have used an augmented version of M.
    The caller must document that distinction and enforce chronological use.
    Re-orthonormalize the returned
    directions with the next current A before constructing augmentation.

    Return ``(U, AU, stats)``. ``projected_ritz_residuals`` measure the small
    Galerkin equations, while ``operator_ritz_residuals`` are Euclidean norms
    of M A u-lambda*u, scaled by ||M A u||+|lambda| ||u||. The latter need
    not be small for a restricted search span and require no new A action.
    """
    q = _positive_integer(q, "q")
    max_basis = _positive_integer(max_basis, "max_basis")
    M = sla.aslinearoperator(M)
    if M.shape != A.shape:
        raise ValueError("M and A must have the same dimensions")
    V, AV, basis_stats = a_orthonormalize(A, P, rtol=rtol, max_vectors=max_basis)
    info = {"basis": basis_stats, "requested_rank": q, "rank": 0,
            "ritz_values": [], "nonpositive_ritz_values": 0,
            "projected_ritz_residuals": [], "operator_ritz_residuals": [],
            "status": "empty_basis"}
    if V.shape[1] == 0:
        return V, AV, info
    MAV = np.asarray(M @ AV, dtype=np.float64)
    if MAV.shape != AV.shape or not np.all(np.isfinite(MAV)):
        raise ValueError("M returned an invalid or nonfinite basis action")
    H, G = AV.T @ MAV, V.T @ AV
    H, G = .5 * (H + H.T), .5 * (G + G.T)
    if not np.all(np.isfinite(H)):
        raise ValueError("Ritz matrix is nonfinite")
    values, modes = la.eigh(H, G, check_finite=False)
    positive = np.flatnonzero(np.isfinite(values) & (values > 0.))
    chosen = positive[:q]
    info["nonpositive_ritz_values"] = int(np.count_nonzero(values <= 0.))
    if not len(chosen):
        info["status"] = "no_positive_ritz_value"
        return V[:, :0], AV[:, :0], info
    coefficients, selected = modes[:, chosen], values[chosen]
    U, AU = V @ coefficients, AV @ coefficients
    action = MAV @ coefficients
    projected = H @ coefficients - (G @ coefficients) * selected[None, :]
    projected_scale = (la.norm(H @ coefficients, axis=0)
                       + np.abs(selected) * la.norm(G @ coefficients, axis=0))
    operator_residual = action - U * selected[None, :]
    operator_scale = la.norm(action, axis=0) + np.abs(selected) * la.norm(U, axis=0)
    tiny = np.finfo(float).tiny
    info.update(rank=int(len(chosen)), ritz_values=selected.tolist(), status="accepted",
                projected_ritz_residuals=(la.norm(projected, axis=0)
                                          / np.maximum(projected_scale, tiny)).tolist(),
                operator_ritz_residuals=(la.norm(operator_residual, axis=0)
                                         / np.maximum(operator_scale, tiny)).tolist())
    return U, AU, info
