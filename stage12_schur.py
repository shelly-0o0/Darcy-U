"""Exact action changes for the stage-11 constrained enrichment family.

This is a classical block-elimination diagnostic, not a learned solver.
Only the coefficient matrix, loads and coarse factorization are required;
no fine solution or fine-error target enters this module.  All matrices
describe the specified finite-dimensional fine discretization.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from stage11_adaptive import _selection


def _indices(values, size, name="extra indices"):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    ids = raw.astype(np.int64)
    if (np.any(raw != ids) or np.any(ids < 0) or np.any(ids >= size)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError(f"{name} must be distinct valid integer indices")
    return np.sort(ids)


@dataclass
class SchurState:
    """Exact conditioned action system after the extra ``indices`` were added."""

    indices: np.ndarray
    conditional_K: np.ndarray
    residuals: np.ndarray
    change: np.ndarray

    @property
    def total_change(self):
        """Descriptive alias used by the search runner."""
        return self.change


class SchurActions:
    """Eliminate all coarse variables once, then evaluate coupled enrichments.

    ``loads`` has shape (number of source/goal loads, fine free DOFs).
    Optional ``coarse_solutions`` has shape (number of loads, coarse DOFs).
    Every reported ``change[a,b]`` is the change in load-a's linear
    functional when solving with load b.  A usual source-0/multiple-goal
    experiment therefore reads ``change[1:,0]``.  A change can be negative;
    without an error target it is not itself an error reduction.

    Extra directions are canonical fine-node vectors at all non-coarse
    interior nodes.  ``extra_fine_indices`` maps extra IDs to fine free
    IDs, ``fine_to_extra`` maps backwards (-1 for coarse vertices), and
    ``center_extra_indices`` maps coarse cell IDs to center extra IDs.
    """

    def __init__(self, case, loads, coarse_lu=None, coarse_solutions=None):
        start = perf_counter()
        loads = np.asarray(loads, dtype=np.float64)
        if (loads.ndim != 2 or loads.shape[1] != case.dimension
                or not np.all(np.isfinite(loads)) or loads.shape[0] == 0):
            raise ValueError("loads must be finite with shape (nrhs, fine_dofs)")
        self.nc = case.nc
        self.nrhs = loads.shape[0]
        self.fine_dofs = case.dimension
        fy, fx = np.divmod(case.free, case.nf+1)
        self.extra_fine_indices = np.flatnonzero((fy % 2 != 0) | (fx % 2 != 0))
        self.nextra = len(self.extra_fine_indices)
        self.fine_to_extra = np.full(case.dimension, -1, dtype=np.int64)
        self.fine_to_extra[self.extra_fine_indices] = np.arange(self.nextra)
        self.full_to_extra = np.full((case.nf+1)**2, -1, dtype=np.int64)
        self.full_to_extra[case.free[self.extra_fine_indices]] = np.arange(self.nextra)
        yy, xx = np.mgrid[:case.nc, :case.nc]
        centers = ((2*yy+1)*(case.nf+1)+2*xx+1).ravel()
        self.center_extra_indices = self.full_to_extra[centers]

        tick = perf_counter()
        if coarse_lu is None:
            coarse_lu = sla.splu(case.Ac.tocsc())
            factor_seconds = perf_counter()-tick
        else:
            factor_seconds = 0.
        tick = perf_counter()
        coarse_loads = np.asarray(case.P.T @ loads.T)
        if coarse_solutions is None:
            coarse_u = coarse_lu.solve(coarse_loads).T
        else:
            coarse_u = np.asarray(coarse_solutions, dtype=np.float64)
            if (coarse_u.shape != (self.nrhs, case.Ac.shape[0])
                    or not np.all(np.isfinite(coarse_u))):
                raise ValueError("coarse_solutions must have shape (nrhs, coarse_dofs)")
        self.coarse_u = coarse_u.copy()
        coarse_residual = np.asarray(case.Ac @ coarse_u.T)-coarse_loads
        self.coarse_relative_residual = float(np.linalg.norm(coarse_residual)
            / max(np.linalg.norm(coarse_loads), np.finfo(float).tiny))
        if self.coarse_relative_residual > 1e-9:
            raise ValueError("supplied coarse solutions/factorization do not solve this problem")
        residual_full = loads - np.asarray(case.Af @ (case.P @ coarse_u.T)).T
        self.residuals = residual_full[:, self.extra_fine_indices]
        coarse_solve_seconds = perf_counter()-tick

        tick = perf_counter()
        coupling = (case.P.T @ case.Af[:, self.extra_fine_indices]).toarray()
        self.coupling = coupling
        raw_K = (case.Af[self.extra_fine_indices][:, self.extra_fine_indices].toarray()
                 - coupling.T @ coarse_lu.solve(coupling))
        self.symmetry_relative_defect = float(np.linalg.norm(raw_K-raw_K.T)
            / max(np.linalg.norm(raw_K), np.finfo(float).tiny))
        if self.symmetry_relative_defect > 1e-10:
            raise ValueError("coarse factorization produced a nonsymmetric Schur complement")
        # Remove only roundoff antisymmetry; no diagonal shift, clipping or fine inverse.
        self.K = .5*(raw_K+raw_K.T)
        schur_seconds = perf_counter()-tick
        self.timings = dict(coarse_factorization_seconds=factor_seconds,
            coarse_solve_residual_seconds=coarse_solve_seconds,
            schur_assembly_seconds=schur_seconds, total_seconds=perf_counter()-start)

    def extra_indices(self, selected_cells):
        """Centers plus shared interior edge midpoints liberated by selected cells."""
        selected = _selection(selected_cells, self.nc)
        nf = 2*self.nc
        independent = np.zeros((nf+1, nf+1), dtype=bool)
        independent[1::2, 1::2] = selected
        independent[1::2, 2:-1:2] = selected[:, :-1] & selected[:, 1:]
        independent[2:-1:2, 1::2] = selected[:-1, :] & selected[1:, :]
        return np.sort(self.full_to_extra[np.flatnonzero(independent.ravel())])

    def added_dofs(self, selected_cells):
        return len(self.extra_indices(selected_cells))

    def qoi_change(self, indices):
        """Exact coupled Gram change relative to the coarse solution."""
        ids = _indices(indices, self.nextra)
        if not len(ids):
            return np.zeros((self.nrhs, self.nrhs))
        rhs = self.residuals[:, ids]
        factor = la.cho_factor(self.K[np.ix_(ids, ids)], lower=True, check_finite=False)
        answer = rhs @ la.cho_solve(factor, rhs.T, check_finite=False)
        return .5*(answer+answer.T)

    def prepare(self, indices):
        """Condition on selected extra IDs, caching the full remaining system."""
        ids = _indices(indices, self.nextra)
        if not len(ids):
            return SchurState(ids, self.K.copy(), self.residuals.copy(),
                              np.zeros((self.nrhs, self.nrhs)))
        factor = la.cho_factor(self.K[np.ix_(ids, ids)], lower=True, check_finite=False)
        solved_rows = la.cho_solve(factor, self.K[ids, :], check_finite=False)
        conditional = self.K-self.K[:, ids] @ solved_rows
        residual = self.residuals-self.residuals[:, ids] @ solved_rows
        change = self.residuals[:, ids] @ la.cho_solve(
            factor, self.residuals[:, ids].T, check_finite=False)
        conditional[ids, :] = 0.
        conditional[:, ids] = 0.
        residual[:, ids] = 0.
        return SchurState(ids, .5*(conditional+conditional.T), residual, .5*(change+change.T))

    def _new_indices(self, state, added_indices):
        ids = _indices(added_indices, self.nextra, "added extra indices")
        if np.intersect1d(ids, state.indices).size:
            raise ValueError("added extra indices overlap the current state")
        return ids

    def incremental_change(self, state, added_indices):
        """Exact extra change for a coupled action after the current enrichment."""
        ids = self._new_indices(state, added_indices)
        if not len(ids):
            return np.zeros((self.nrhs, self.nrhs))
        rhs = state.residuals[:, ids]
        factor = la.cho_factor(state.conditional_K[np.ix_(ids, ids)],
                               lower=True, check_finite=False)
        answer = rhs @ la.cho_solve(factor, rhs.T, check_finite=False)
        return .5*(answer+answer.T)

    def incremental_changes(self, state, candidate_indices):
        """Batch exact action changes, grouping candidates by new-DOF count.

        The return order equals the input order; each candidate must already
        exclude currently selected IDs.  Grouping is only an algebraic
        acceleration and does not approximate interactions.
        """
        groups = {}
        result = np.zeros((len(candidate_indices), self.nrhs, self.nrhs))
        for position, values in enumerate(candidate_indices):
            ids = self._new_indices(state, values)
            if len(ids):
                groups.setdefault(len(ids), []).append((position, ids))
        for entries in groups.values():
            positions = [item[0] for item in entries]
            ids = np.stack([item[1] for item in entries])
            matrices = state.conditional_K[ids[:, :, None], ids[:, None, :]]
            # Shape: candidates, extra directions, loads.
            rhs = state.residuals[:, ids].transpose(1, 2, 0)
            lower = np.linalg.cholesky(matrices)
            whitened = np.linalg.solve(lower, rhs)
            result[positions] = np.einsum("bik,bil->bkl", whitened, whitened)
        return result

    def advance(self, state, added_indices):
        """Condition the cached system on an action, without mutating the input."""
        ids = self._new_indices(state, added_indices)
        if not len(ids):
            return SchurState(state.indices.copy(), state.conditional_K.copy(),
                              state.residuals.copy(), state.change.copy())
        factor = la.cho_factor(state.conditional_K[np.ix_(ids, ids)],
                               lower=True, check_finite=False)
        solved_rows = la.cho_solve(factor, state.conditional_K[ids, :], check_finite=False)
        conditional = state.conditional_K-state.conditional_K[:, ids] @ solved_rows
        residual = state.residuals-state.residuals[:, ids] @ solved_rows
        change = state.change + state.residuals[:, ids] @ la.cho_solve(
            factor, state.residuals[:, ids].T, check_finite=False)
        all_ids = np.union1d(state.indices, ids)
        conditional[all_ids, :] = 0.
        conditional[:, all_ids] = 0.
        residual[:, all_ids] = 0.
        return SchurState(all_ids, .5*(conditional+conditional.T), residual,
                          .5*(change+change.T))


def candidate_patches(nc):
    """Fixed singleton, adjacent-pair and 2×2 actions in deterministic order."""
    if int(nc) != nc or nc < 2:
        raise ValueError("nc must be an integer at least two")
    nc = int(nc)
    result = []
    shapes = [("single", ((0, 0),)),
              ("horizontal_pair", ((0, 0), (0, 1))),
              ("vertical_pair", ((0, 0), (1, 0))),
              ("block_2x2", ((0, 0), (0, 1), (1, 0), (1, 1)))]
    for kind, offsets in shapes:
        dy = max(item[0] for item in offsets)
        dx = max(item[1] for item in offsets)
        for y in range(nc-dy):
            for x in range(nc-dx):
                cells = tuple((y+oy)*nc+x+ox for oy, ox in offsets)
                result.append(dict(id=len(result), kind=kind, cells=cells))
    return result
