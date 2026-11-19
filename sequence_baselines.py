"""Leakage-free current-matrix initial guesses and fixed AMG baselines.

Only previous computed solution vectors, the current A, and the current b
are used. No reference solution or future sequence value is accepted by
these utilities. A reused hierarchy is a fixed SPD preconditioner built
from the first matrix, not a Galerkin hierarchy refreshed with current A.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

# Keep the repository-local optional dependency usable without installation.
_VENDOR = Path(__file__).resolve().parent / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


def _validate_system(A, b):
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.shape != (len(b), len(b)):
        raise ValueError("A must be square with the same dimension as b")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must be finite")
    return b


def galerkin_initial_guess(A, b, vectors, max_vectors=8, rtol=1e-10):
    """Minimize .5*x.T*A*x-b.T*x in a small stable history subspace.

    ``vectors`` is a chronological iterable of one-dimensional vectors;
    only the last ``max_vectors`` are used. Pass previous solutions,
    their differences, or a deliberate mixture. Dense 2D arrays are read
    row-by-row, consistently with a list of vectors. The caller must supply
    previous computed solutions rather than offline direct references.

    Columns are normalized twice: first by max magnitude to avoid overflow,
    then by current A energy. Small/negative Gram eigenvalues are discarded.
    The result minimizes energy over the retained numerical subspace;
    truncation can remove nearly dependent directions from the exact span.
    Invalid, zero, or nonpositive-energy vectors are discarded safely.
    """
    start = perf_counter()
    b = _validate_system(A, b)
    if not isinstance(max_vectors, (int, np.integer)) or not 1 <= max_vectors <= 8:
        raise ValueError("max_vectors must be an integer between 1 and 8")
    if not 0. < rtol < 1.:
        raise ValueError("rtol must lie strictly between zero and one")
    history = [] if vectors is None else list(vectors)
    selected = history[-max_vectors:]
    columns, actions = [], []
    rejected = 0
    for vector in selected:
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape != b.shape:
            raise ValueError("history vector has incompatible length")
        scale = np.max(np.abs(v), initial=0.)
        if not np.all(np.isfinite(v)) or not np.isfinite(scale) or scale <= 0.:
            rejected += 1
            continue
        v = v / scale
        av = A @ v
        energy = float(v @ av)
        if not np.isfinite(energy) or energy <= 0.:
            rejected += 1
            continue
        scale = np.sqrt(energy)
        columns.append(v / scale)
        actions.append(av / scale)
    x = np.zeros_like(b)
    rank, retained_condition, energy = 0, 0., 0.
    status = "empty_history" if not selected else "no_valid_direction"
    if columns:
        V, AV = np.column_stack(columns), np.column_stack(actions)
        gram = V.T @ AV
        gram = .5 * (gram + gram.T)
        vals, modes = la.eigh(gram, check_finite=False)
        cutoff = rtol * max(float(vals[-1]), np.finfo(float).tiny)
        keep = vals > cutoff
        rank = int(np.count_nonzero(keep))
        if rank:
            transform = modes[:, keep]
            coefficient = transform @ ((transform.T @ (V.T @ b)) / vals[keep])
            candidate = V @ coefficient
            a_candidate = AV @ coefficient
            candidate_energy = float(.5 * candidate @ a_candidate - b @ candidate)
            if np.all(np.isfinite(candidate)) and np.isfinite(candidate_energy) \
                    and candidate_energy <= 0.:
                x, energy = candidate, candidate_energy
                retained_condition = float(vals[keep][-1] / vals[keep][0])
                status = "accepted"
            else:
                rank, status = 0, "nonfinite_or_increased_energy"
    stats = {"seconds": perf_counter() - start, "input_vectors": len(history),
             "selected_vectors": len(selected), "valid_vectors": len(columns),
             "rejected_vectors": rejected, "rank": rank,
             "retained_condition": retained_condition, "energy": energy,
             "status": status, "rtol": float(rtol)}
    return x, stats


def project_previous_solution(A, b, previous):
    """Current-A optimal scalar multiple of the previous computed solution."""
    vectors = [] if previous is None else [previous]
    return galerkin_initial_guess(A, b, vectors, max_vectors=1)


def damped_initial_correction(A, b, x0, direction):
    """Apply the current-A optimal correction alpha*d to an initial guess.

    alpha=(d.T*(b-A*x0))/(d.T*A*d) can have either sign. This is an energy
    line search, not clipping to [0,1]. Nonfinite/zero/indefinite directions
    receive alpha=0. A nonfinite x0 falls back to zero before this check.
    Roundoff is guarded by checking the quadratic energy change. No direct
    solution is computed or accessed.
    """
    start = perf_counter()
    b = _validate_system(A, b)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    if x.shape != b.shape or d.shape != b.shape:
        raise ValueError("initial guess and direction must match b")
    reset = not np.all(np.isfinite(x))
    if reset:
        x.fill(0.)
    alpha, energy_change = 0., 0.
    status = "invalid_direction"
    scale = np.max(np.abs(d), initial=0.)
    if np.all(np.isfinite(d)) and np.isfinite(scale) and scale > 0.:
        d_scaled = d / scale
        r = b - A @ x
        ad = A @ d_scaled
        denominator = float(d_scaled @ ad)
        numerator = float(d_scaled @ r)
        if np.isfinite(denominator) and denominator > 0. and np.isfinite(numerator):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                scaled_alpha = numerator / denominator
                proposed_alpha = scaled_alpha / scale
                candidate = x + scaled_alpha * d_scaled
                change = scaled_alpha * (.5 * scaled_alpha * denominator - numerator)
            if np.isfinite(proposed_alpha) and np.all(np.isfinite(candidate)) \
                    and np.isfinite(change) and change <= 0.:
                x, alpha, energy_change = candidate, float(proposed_alpha), float(change)
                status = "accepted"
            else:
                status = "nonfinite_or_increased_energy"
        else:
            status = "nonpositive_energy_direction"
    return x, {"seconds": perf_counter() - start, "alpha": alpha,
               "energy_change": energy_change, "initial_guess_reset": reset,
               "status": status}


def jacobi_initial_correction(A, b, x0, sweeps=3, omega=2. / 3.):
    """Perform a prespecified number of weighted-Jacobi initial-guess sweeps.

    No stopping criterion, reference, adaptive parameter, or preconditioner
    change is involved. Defaults target the present scalar Darcy matrices.
    A nonfinite iterate is rejected; positive finite diagonal is required.
    """
    start = perf_counter()
    b = _validate_system(A, b)
    if not isinstance(sweeps, (int, np.integer)) or sweeps < 0:
        raise ValueError("sweeps must be a nonnegative integer")
    if not np.isfinite(omega) or omega <= 0.:
        raise ValueError("omega must be positive and finite")
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    if x.shape != b.shape:
        raise ValueError("initial guess must match b")
    reset = not np.all(np.isfinite(x))
    if reset:
        x.fill(0.)
    diagonal = np.asarray(A.diagonal(), dtype=np.float64)
    completed = 0
    status = "invalid_diagonal"
    if np.all(np.isfinite(diagonal)) and np.all(diagonal > 0.):
        status = "completed"
        for _ in range(sweeps):
            candidate = x + omega * (b - A @ x) / diagonal
            if not np.all(np.isfinite(candidate)):
                status = "nonfinite_iteration_rejected"
                break
            x = candidate
            completed += 1
    return x, {"seconds": perf_counter() - start, "sweeps": completed,
               "requested_sweeps": int(sweeps), "omega": float(omega),
               "initial_guess_reset": reset, "status": status}


class FixedAMGHierarchy:
    """An owned, immutable-in-use symmetric V-cycle hierarchy.

    Symmetric Gauss-Seidel pre/post smoothing and a pseudoinverse coarse
    solve preserve the usual SPD preconditioner construction for SPD A.
    A copy of A prevents later caller mutation changing the hierarchy.
    The build's NumPy random state is locally seeded and restored.
    """

    def __init__(self, A, seed=0, max_levels=10, max_coarse=10):
        import pyamg

        start = perf_counter()
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        owned_A = sp.csr_matrix(A, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(owned_A.data)):
            raise ValueError("A must be finite")
        state = np.random.get_state()
        try:
            np.random.seed(int(seed) % 2**32)
            self.hierarchy = pyamg.smoothed_aggregation_solver(
                owned_A, symmetry="symmetric", max_levels=max_levels,
                max_coarse=max_coarse,
                presmoother=("gauss_seidel", {"sweep": "symmetric"}),
                postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
                coarse_solver="pinv")
        finally:
            np.random.set_state(state)
        self.preconditioner = self.hierarchy.aspreconditioner(cycle="V")
        self.shape = owned_A.shape
        self.stats = {"setup_seconds": perf_counter() - start,
                      "levels": len(self.hierarchy.levels),
                      "coarse_dimension": int(self.hierarchy.levels[-1].A.shape[0]),
                      "operator_complexity": float(self.hierarchy.operator_complexity()),
                      "grid_complexity": float(self.hierarchy.grid_complexity()),
                      "hierarchy_seed": int(seed) % 2**32}

    @classmethod
    def from_interpolation(cls, A, template):
        """Rebuild every RAP matrix and smoother using frozen P/R operators.

        This creates independent Level objects and a fresh coarse-solver
        closure, so neither smoother nor pseudoinverse caches can come from
        earlier matrices. P/R are shared read-only. Every previous operator
        remains unchanged and usable. This is more than change_solve_matrix,
        which does not rebuild all coarse Galerkin matrices.
        """
        from pyamg.multilevel import MultilevelSolver
        from pyamg.relaxation.smoothing import change_smoothers

        start = perf_counter()
        if A.shape != template.shape:
            raise ValueError("A and template must have identical dimensions")
        current = sp.csr_matrix(A, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(current.data)):
            raise ValueError("A must be finite")
        levels = []
        for original in template.hierarchy.levels[:-1]:
            level = MultilevelSolver.Level()
            level.A, level.P, level.R = current, original.P, original.R
            level.A.symmetry = "symmetric"
            levels.append(level)
            current = (level.R @ (current @ level.P)).tocsr()
        coarsest = MultilevelSolver.Level()
        coarsest.A = current
        coarsest.A.symmetry = "symmetric"
        levels.append(coarsest)
        hierarchy = MultilevelSolver(levels, coarse_solver="pinv")
        change_smoothers(hierarchy,
                         presmoother=("gauss_seidel", {"sweep": "symmetric"}),
                         postsmoother=("gauss_seidel", {"sweep": "symmetric"}))
        result = cls.__new__(cls)
        result.hierarchy = hierarchy
        result.preconditioner = hierarchy.aspreconditioner(cycle="V")
        result.shape = A.shape
        result.stats = {"setup_seconds": perf_counter() - start,
                        "levels": len(levels),
                        "coarse_dimension": int(levels[-1].A.shape[0]),
                        "operator_complexity": float(hierarchy.operator_complexity()),
                        "grid_complexity": float(hierarchy.grid_complexity()),
                        "hierarchy_seed": template.stats["hierarchy_seed"]}
        return result


class AMGSequencePreconditioner:
    """Explicit fresh, reuse-first, or frozen-interpolation RAP-refresh policy.

    ``for_matrix`` is called once BEFORE a PCG solve. It returns a fixed
    LinearOperator and per-step setup statistics; it never rebuilds during
    PCG or uses an adaptive convergence-triggered threshold. Returned
    operators remain valid after later calls. A reused hierarchy can be
    a poor preconditioner after topology changes, which must be measured.
    """

    def __init__(self, mode="fresh", seed=0, max_levels=10, max_coarse=10):
        if mode not in ("fresh", "reuse_first", "refresh_galerkin"):
            raise ValueError("mode must be fresh, reuse_first, or refresh_galerkin")
        self.mode = mode
        self.seed = int(seed)
        self.max_levels, self.max_coarse = max_levels, max_coarse
        self.calls = 0
        self.fixed = None
        self._interpolation_template = None

    def for_matrix(self, A):
        start = perf_counter()
        if self.fixed is not None and A.shape != self.fixed.shape:
            raise ValueError("sequence matrices must have fixed dimension")
        rebuild = self.mode == "fresh" or self.fixed is None
        refresh = self.mode == "refresh_galerkin" and self.fixed is not None
        if rebuild:
            self.fixed = FixedAMGHierarchy(A, seed=self.seed + self.calls,
                                           max_levels=self.max_levels,
                                           max_coarse=self.max_coarse)
            if self._interpolation_template is None:
                self._interpolation_template = self.fixed
        elif refresh:
            self.fixed = FixedAMGHierarchy.from_interpolation(
                A, self._interpolation_template)
        stats = dict(self.fixed.stats)
        stats.update(mode=self.mode, hierarchy_rebuilt=bool(rebuild),
                     interpolation_rebuilt=bool(rebuild),
                     galerkin_refreshed=bool(refresh),
                     hierarchy_build_seconds=stats["setup_seconds"] if rebuild or refresh else 0.,
                     setup_seconds=perf_counter() - start, sequence_step=self.calls)
        self.calls += 1
        return self.fixed.preconditioner, stats
