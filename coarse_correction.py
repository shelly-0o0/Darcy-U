"""Classical current-matrix Galerkin correction on a fixed bilinear grid.

This baseline uses the present residual and a sparse coarse direct solve.
It never solves the fine system directly or accesses reference solutions.
"""

from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def _bilinear_prolongation(n, coarse_n):
    """Free-node interpolation; Dirichlet boundary coarse values are zero."""
    ratio = n // coarse_n
    fine = np.arange(1, n, dtype=np.int64)
    left = fine // ratio
    fraction = (fine % ratio) / float(ratio)
    rows, cols, values = [], [], []
    for coarse, weight in ((left, 1. - fraction), (left + 1, fraction)):
        valid = (coarse >= 1) & (coarse < coarse_n) & (weight != 0.)
        rows.append(fine[valid] - 1)
        cols.append(coarse[valid] - 1)
        values.append(weight[valid])
    one_dimensional = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n - 1, coarse_n - 1)).tocsr()
    # y-major free-node ordering agrees with numerics.assemble_system.
    return sp.kron(one_dimensional, one_dimensional, format="csr")


class CoarseGalerkinCorrection:
    """Add a fixed-grid, current-A energy-optimal coarse correction.

    P is built lazily during the first apply call and then reused. Each
    apply assembles and factors the current P.T*A*P, computes the current
    residual, and solves (P.T*A*P)c=P.T*(b-A*x0). No current coarse matrix,
    factor, residual, or solution is reused across calls. The correction
    minimizes energy on x0+range(P), up to numerical direct-solve error.
    """

    def __init__(self, n, coarse_n=32):
        if not isinstance(n, (int, np.integer)) or not isinstance(coarse_n, (int, np.integer)):
            raise ValueError("n and coarse_n must be integers")
        if coarse_n < 2 or n < coarse_n or n % coarse_n != 0:
            raise ValueError("n must be a positive multiple of coarse_n >= 2")
        self.n, self.coarse_n = int(n), int(coarse_n)
        self.P = None
        self._restriction = None
        self.calls = 0

    def apply(self, A, b, x0):
        start = perf_counter()
        size = (self.n - 1) ** 2
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        x = np.zeros(size) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        if A.shape != (size, size) or b.shape != (size,) or x.shape != (size,):
            raise ValueError("A, b, and initial guess must match the specified grid")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x)):
            raise ValueError("b and initial guess must be finite")
        interpolation_start = perf_counter()
        built = self.P is None
        if built:
            self.P = _bilinear_prolongation(self.n, self.coarse_n)
            self._restriction = self.P.T.tocsr()
        interpolation_seconds = perf_counter() - interpolation_start if built else 0.
        residual_start = perf_counter()
        residual = b - A @ x
        if not np.all(np.isfinite(residual)):
            raise ValueError("current residual must be finite")
        coarse_rhs = self._restriction @ residual
        residual_seconds = perf_counter() - residual_start
        assembly_start = perf_counter()
        AP = A @ self.P
        coarse_A = (self._restriction @ AP).tocsc()
        # Congruence is symmetric mathematically; remove asymmetric roundoff.
        coarse_A = ((coarse_A + coarse_A.T) * .5).tocsc()
        if not np.all(np.isfinite(coarse_A.data)) or np.any(coarse_A.diagonal() <= 0.):
            raise ValueError("coarse matrix must have finite entries and positive diagonal")
        assembly_seconds = perf_counter() - assembly_start
        factor_start = perf_counter()
        coarse_factor = sla.splu(coarse_A, permc_spec="MMD_AT_PLUS_A",
                                 diag_pivot_thresh=0., options={"SymmetricMode": True})
        factor_seconds = perf_counter() - factor_start
        solve_start = perf_counter()
        coarse_solution = coarse_factor.solve(coarse_rhs)
        solve_seconds = perf_counter() - solve_start
        correction_start = perf_counter()
        direction = self.P @ coarse_solution
        # Check the actual fine quadratic change; this extra matvec is timed.
        direction_energy = float(direction @ (A @ direction))
        projected_work = float(direction @ residual)
        energy_change = .5 * direction_energy - projected_work
        candidate = x + direction
        tolerance = 128. * np.finfo(float).eps * max(
            abs(.5 * direction_energy) + abs(projected_work), np.finfo(float).tiny)
        accepted = (np.all(np.isfinite(candidate)) and np.isfinite(energy_change)
                    and energy_change <= tolerance)
        if accepted:
            x = candidate
        else:
            energy_change = 0.
        correction_seconds = perf_counter() - correction_start
        self.calls += 1
        stats = {"seconds": perf_counter() - start,
                 "setup_seconds": interpolation_seconds + assembly_seconds + factor_seconds,
                 "interpolation_seconds": interpolation_seconds,
                 "residual_seconds": residual_seconds,
                 "coarse_assembly_seconds": assembly_seconds,
                 "coarse_factor_seconds": factor_seconds,
                 "coarse_solve_seconds": solve_seconds,
                 "correction_seconds": correction_seconds,
                 "interpolation_built": built, "coarse_matrix_rebuilt": True,
                 "coarse_dimension": int(coarse_A.shape[0]),
                 "coarse_nnz": int(coarse_A.nnz),
                 "interpolation_nnz": int(self.P.nnz),
                 "energy_change": float(energy_change),
                 "energy_roundoff_tolerance": float(tolerance),
                 "accepted": bool(accepted), "call": self.calls}
        return x, stats
