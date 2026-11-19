"""A bounded classical update of a past slow space for the current system.

The retained old space is extended by its current M A image, and a small
Rayleigh--Ritz problem selects the lowest positive modes.  Every sparse
matrix or preconditioner vector action is explicitly counted.  No solution,
reference error, or current Krylov trajectory is required.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from balanced_augmentation import a_orthonormalize
from cached_pcg import cached_slow_ritz


def update_slow_space(A, M, old_U, q=4, rtol=1e-10, symmetry_tol=1e-7):
    """Return ``(new_U, new_AU, stats)`` for the current fixed real SPD A/M.

    This function owns the current-A orthonormalization: callers should not
    repeat it before this call.  If k old columns yield current rank r,
    the action budget is at most k+r A vector actions and 2r M vector
    actions.  In the usual k=r=q=4 case, that is A:8 and M:8.  Invalid or
    zero old columns are rejected before an A action when possible.

    After current-A orthonormalization gives U/AU, compute MAU=M@AU,
    AMAU=A@MAU and MAMAU=M@AMAU, each once.  Cached Rayleigh--Ritz uses
    W=[U,MAU], AW=[AU,AMAU], MAW=[MAU,MAMAU], with no further A/M action.
    The untruncated extension contains the entire retained old space.
    In exact arithmetic its first min(q,r) Ritz values cannot exceed the
    corresponding old-space values.  Floating-point rank truncation can
    weaken that guarantee, so both spectra and any increase are reported.

    Operator-action counts mean vector applications, not Python call count
    or wall time.  Dense work and all action costs must also be timed by
    the caller.  Empty input gives empty output with zero operator actions.
    Failed orthonormalization, invalid new actions, or rejected cached
    extraction give empty output with an explicit non-accepted status.
    Linear-algebra decomposition failures likewise return an empty basis,
    retaining action counts and the failing stage; other exceptions are
    not suppressed.  No silent fallback or extra solve is performed.
    """
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError("q must be a positive integer")
    if not np.isfinite(rtol) or not 0. < rtol < 1.:
        raise ValueError("rtol must lie strictly between zero and one")
    if not np.isfinite(symmetry_tol) or not 0. < symmetry_tol < 1.:
        raise ValueError("symmetry_tol must lie strictly between zero and one")
    A, M = sla.aslinearoperator(A), sla.aslinearoperator(M)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1] or M.shape != A.shape:
        raise ValueError("A and M must be square with the same shape")
    if np.issubdtype(A.dtype, np.complexfloating) or np.issubdtype(M.dtype, np.complexfloating):
        raise ValueError("update_slow_space supports real operators only")
    n = A.shape[0]
    if old_U is None:
        old_U = np.empty((n, 0), dtype=np.float64)
    if np.iscomplexobj(old_U):
        raise ValueError("old_U must be real")
    old_U = np.asarray(old_U, dtype=np.float64)
    if old_U.ndim == 1:
        old_U = old_U[:, None] if old_U.size else np.empty((n, 0), dtype=np.float64)
    if old_U.ndim != 2 or old_U.shape[0] != n:
        raise ValueError("old_U must have shape (n,k), with vectors in columns")
    k = int(old_U.shape[1])
    stats = {"input_vectors": k, "input_rank": 0, "requested_rank": int(q),
             "rank": 0, "output_rank": 0, "expanded_vectors": 0,
             "a_applications": 0, "m_applications": 0,
             "orthonormalization_a_applications": 0,
             "extension_a_applications": 0, "extension_m_applications": 0,
             "max_a_applications": k, "max_m_applications": 0,
             "basis": {}, "extraction": {}, "ritz_values": [],
             "old_ritz_values": [], "max_relative_ritz_increase": 0.,
             "orthogonality_error": 0., "status": "empty_basis"}
    empty = np.empty((n, 0), dtype=np.float64)

    def reject(status):
        stats["status"] = status
        return empty.copy(), empty.copy(), stats

    def decomposition_failure(stage):
        stats["numerical_failure_stage"] = stage
        return reject("numerical_decomposition_failure:" + stage)

    if not k:
        return reject("empty_basis")

    def apply_current_A(vector):
        stats["a_applications"] += 1
        return A @ vector

    counted_A = sla.LinearOperator(A.shape, matvec=apply_current_A, dtype=np.float64)
    try:
        U, AU, basis = a_orthonormalize(counted_A, old_U, rtol=rtol)
    except la.LinAlgError:
        stats["orthonormalization_a_applications"] = stats["a_applications"]
        return decomposition_failure("orthonormalization")
    stats["basis"] = basis
    stats["orthonormalization_a_applications"] = stats["a_applications"]
    r = int(U.shape[1])
    stats.update(input_rank=r, max_a_applications=k+r, max_m_applications=2*r)
    if not r:
        return reject("orthonormalization_rejected")

    def apply_columns(operator, block, which):
        columns = []
        for vector in block.T:
            stats[f"{which}_applications"] += 1
            stats[f"extension_{which}_applications"] += 1
            action = np.asarray(operator @ vector, dtype=np.float64).reshape(-1)
            if action.shape != (n,) or not np.all(np.isfinite(action)):
                return None
            columns.append(action)
        return np.column_stack(columns)

    MAU = apply_columns(M, AU, "m")
    if MAU is None:
        return reject("invalid_m_action")
    AMAU = apply_columns(A, MAU, "a")
    if AMAU is None:
        return reject("invalid_a_extension_action")
    MAMAU = apply_columns(M, AMAU, "m")
    if MAMAU is None:
        return reject("invalid_second_m_action")
    W, AW, MAW = (np.column_stack(pair) for pair in
                  ((U, MAU), (AU, AMAU), (MAU, MAMAU)))
    stats["expanded_vectors"] = int(W.shape[1])
    try:
        new_U, new_AU, extracted = cached_slow_ritz(
            W, AW, MAW, q=q, rtol=rtol, symmetry_tol=symmetry_tol)
    except la.LinAlgError:
        return decomposition_failure("extraction")
    stats["extraction"] = extracted
    if extracted["status"] != "accepted":
        return reject("extraction_rejected:" + extracted["status"])
    # These small-matrix diagnostics reuse the already paid U/AU/MAU;
    # they do not cause extra A or M applications.
    old_G, old_H = U.T @ AU, AU.T @ MAU
    try:
        old_values = la.eigvalsh(.5 * (old_H + old_H.T),
                               .5 * (old_G + old_G.T), check_finite=False)
    except la.LinAlgError:
        return decomposition_failure("old_spectrum")
    new_values = np.asarray(extracted["ritz_values"], dtype=np.float64)
    compared = min(len(old_values), len(new_values))
    increase = np.maximum(new_values[:compared] - old_values[:compared], 0.)
    increase /= np.maximum(np.abs(old_values[:compared]), np.finfo(float).tiny)
    stats.update(status="accepted", rank=int(new_U.shape[1]),
                 output_rank=int(new_U.shape[1]),
                 ritz_values=extracted["ritz_values"], old_ritz_values=old_values.tolist(),
                 max_relative_ritz_increase=float(np.max(increase, initial=0.)),
                 orthogonality_error=extracted["orthogonality_error"])
    return new_U, new_AU, stats
