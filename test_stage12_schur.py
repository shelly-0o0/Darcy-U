"""Independent numerical checks against actual constrained Galerkin solves."""

import unittest

import numpy as np
import scipy.sparse.linalg as sla

from stage10_goal_numerics import assemble_nested
from stage11_adaptive import build_space, count_added_dofs, solve_adaptive
from stage12_schur import SchurActions, candidate_patches


class SchurActionsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12001)
        self.case = assemble_nested(4, np.exp(rng.normal(size=(8, 8))))
        # Arbitrary algebraic loads also exercise signed and non-smooth goals.
        self.loads = rng.normal(size=(4, self.case.dimension))
        self.model = SchurActions(self.case, self.loads)

    def actual_change(self, case, loads, selected):
        T, _ = build_space(case, selected)
        coarse = case.P @ sla.spsolve(case.Ac.tocsc(), case.P.T @ loads.T)
        refined = np.column_stack([solve_adaptive(case, T, load)["lifted_u"]
                                   for load in loads])
        return loads @ (refined-coarse)

    def test_extra_maps_and_exact_counts(self):
        model, case = self.model, self.case
        self.assertEqual(model.nextra, case.dimension-case.Ac.shape[0])
        np.testing.assert_array_equal(model.fine_to_extra[model.extra_fine_indices],
                                      np.arange(model.nextra))
        for cells in ([], [0], [0, 1], [0, 1, 4, 5], [0, 3, 12, 15], np.arange(16)):
            ids = model.extra_indices(cells)
            self.assertEqual(model.added_dofs(cells), count_added_dofs(cells, case.nc))
            self.assertEqual(len(ids), len(np.unique(ids)))
            self.assertTrue(np.all(ids >= 0))
            _, metadata = build_space(case, cells)
            fine_ids = np.array(metadata["independent_fine_free_indices"])
            expected = model.fine_to_extra[fine_ids]
            np.testing.assert_array_equal(ids, np.sort(expected[expected >= 0]))
        np.testing.assert_array_equal(model.extra_indices(np.arange(16)),
                                      np.arange(model.nextra))

    def test_schur_is_spd_and_augmented_basis_spans_fine_space(self):
        case, model = self.case, self.model
        E = np.eye(case.dimension)[:, model.extra_fine_indices]
        augmented = np.column_stack((case.P.toarray(), E))
        self.assertEqual(np.linalg.matrix_rank(augmented), case.dimension)
        np.testing.assert_allclose(model.K, model.K.T, atol=1e-14)
        self.assertGreater(float(np.linalg.eigvalsh(model.K)[0]), 0.)
        self.assertLess(model.symmetry_relative_defect, 1e-13)

    def test_actual_spaces_match_empty_partial_and_full_changes(self):
        for cells in ([], [5], [5, 6], [0, 1, 4, 5, 15], np.arange(16)):
            actual = self.actual_change(self.case, self.loads, cells)
            predicted = self.model.qoi_change(self.model.extra_indices(cells))
            np.testing.assert_allclose(predicted, actual, rtol=5e-12, atol=2e-12)

    def test_high_contrast_actual_solver_identity(self):
        yy, xx = np.mgrid[:8, :8]
        coefficient = np.where((xx < 4) & (yy > 1) & (yy < 7), 1000., 1.)
        case = assemble_nested(4, coefficient)
        model = SchurActions(case, self.loads)
        for cells in ([4, 5, 8, 9], [0, 1, 2, 3, 6, 7, 10], np.arange(16)):
            np.testing.assert_allclose(model.qoi_change(model.extra_indices(cells)),
                self.actual_change(case, self.loads, cells), rtol=2e-11, atol=5e-11)

    def test_multiple_signed_goals_and_positive_semidefinite_gram(self):
        f, g = self.loads[:2]
        loads = np.stack((f, g, -g, 2*f))
        model = SchurActions(self.case, loads)
        change = model.qoi_change(model.extra_indices([1, 2, 5, 6]))
        np.testing.assert_allclose(change[2], -change[1], atol=1e-13)
        np.testing.assert_allclose(change[3], 2*change[0], atol=1e-13)
        self.assertGreater(change[0, 0], 0.)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(change)[0]), -1e-12)
        np.testing.assert_allclose(change, self.actual_change(self.case, loads, [1, 2, 5, 6]),
                                   rtol=5e-12, atol=2e-12)

    def test_supplied_coarse_factorization_and_solutions_match(self):
        lu = sla.splu(self.case.Ac.tocsc())
        coarse_u = lu.solve(self.case.P.T @ self.loads.T).T
        reused = SchurActions(self.case, self.loads, lu, coarse_u)
        np.testing.assert_array_equal(reused.K, self.model.K)
        np.testing.assert_array_equal(reused.residuals, self.model.residuals)
        self.assertEqual(reused.timings["coarse_factorization_seconds"], 0.)
        with self.assertRaises(ValueError):
            SchurActions(self.case, self.loads, lu, coarse_u+1.)

    def test_prepare_and_incremental_changes_match_joint_solve(self):
        initial = self.model.extra_indices([5, 6])
        union = self.model.extra_indices([5, 6, 9, 10])
        new = np.setdiff1d(union, initial)
        state = self.model.prepare(initial)
        change = self.model.incremental_change(state, new)
        np.testing.assert_allclose(state.total_change+change,
            self.model.qoi_change(union), rtol=5e-13, atol=2e-13)
        np.testing.assert_allclose(state.total_change+change,
            self.actual_change(self.case, self.loads, [5, 6, 9, 10]), rtol=5e-12, atol=2e-12)

    def test_sequential_advance_matches_fresh_conditioning(self):
        state = self.model.prepare([])
        old_indices = state.indices.copy()
        old_matrix = state.conditional_K.copy()
        cells = []
        for action in ([5], [6], [9, 10], [0, 1], [2, 3, 7, 11, 15]):
            cells = sorted(set(cells).union(action))
            all_ids = self.model.extra_indices(cells)
            state = self.model.advance(state, np.setdiff1d(all_ids, state.indices))
            direct = self.model.prepare(all_ids)
            np.testing.assert_allclose(state.change, direct.change, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(state.conditional_K, direct.conditional_K,
                                       rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(state.residuals, direct.residuals,
                                       rtol=2e-12, atol=2e-12)
        # API is functional: advancing did not edit a previous state or the model.
        np.testing.assert_array_equal(old_indices, [])
        np.testing.assert_array_equal(old_matrix, self.model.K)

    def test_batched_incremental_matches_scalar_candidates(self):
        current = [5, 6]
        state = self.model.prepare(self.model.extra_indices(current))
        candidates = [np.array([], dtype=int)]
        for action in candidate_patches(4):
            union = self.model.extra_indices(sorted(set(current).union(action["cells"])))
            candidates.append(np.setdiff1d(union, state.indices))
        batch = self.model.incremental_changes(state, candidates)
        scalar = np.stack([self.model.incremental_change(state, ids) for ids in candidates])
        np.testing.assert_allclose(batch, scalar, rtol=2e-12, atol=2e-12)
        self.assertEqual(self.model.incremental_changes(state, []).shape, (0, 4, 4))

    def test_centers_have_no_fine_edges_but_coarse_elimination_couples_them(self):
        # Absence of a direct fine-center edge does not imply action independence.
        centers = self.model.center_extra_indices
        fine_ids = self.model.extra_fine_indices[centers]
        raw = self.case.Af[fine_ids][:, fine_ids].toarray()
        np.testing.assert_array_equal(raw, np.diag(np.diag(raw)))
        schur_centers = self.model.K[np.ix_(centers, centers)]
        self.assertGreater(np.max(np.abs(schur_centers-np.diag(np.diag(schur_centers)))), .01)
        separate = sum((self.model.qoi_change([idx]) for idx in centers), np.zeros((4, 4)))
        coupled = self.model.qoi_change(centers)
        self.assertGreater(np.linalg.norm(coupled-separate), .01)

    def test_joint_adjacent_patches_are_not_sum_of_empty_state_actions(self):
        first = self.model.extra_indices([5])
        joint = self.model.extra_indices([5, 6])
        second_new = np.setdiff1d(joint, first)
        empty = self.model.prepare([])
        uncoupled = (self.model.incremental_change(empty, first)
                     + self.model.incremental_change(empty, second_new))
        true = self.model.qoi_change(joint)
        self.assertGreater(np.linalg.norm(true-uncoupled), .001)
        coupled = (self.model.qoi_change(first)
                   + self.model.incremental_change(self.model.prepare(first), second_new))
        np.testing.assert_allclose(coupled, true, rtol=1e-12, atol=1e-12)

    def test_no_fine_solve_or_target_is_required(self):
        self.assertNotIn("fine", self.case.cache)
        self.model.qoi_change(self.model.extra_indices([0, 1, 4, 5]))
        self.assertNotIn("fine", self.case.cache)
        self.assertEqual(self.case.cache_statistics["fine_factorizations"], 0)

    def test_invalid_loads_extra_indices_and_overlapping_actions(self):
        for loads in (self.loads[0], np.ones((4, 3)), self.loads*np.nan):
            with self.assertRaises(ValueError):
                SchurActions(self.case, loads)
        for ids in ([0, 0], [-1], [self.model.nextra], [1.5], [[1]]):
            with self.assertRaises(ValueError):
                self.model.qoi_change(ids)
        state = self.model.prepare([0, 1])
        with self.assertRaises(ValueError):
            self.model.incremental_change(state, [1, 2])
        with self.assertRaises(ValueError):
            self.model.extra_indices([2, 2])


class PatchMetadataTests(unittest.TestCase):
    def test_fixed_patch_counts_shapes_order_and_deduplication(self):
        for nc in (2, 4, 16):
            actions = candidate_patches(nc)
            self.assertEqual(len(actions), nc*nc+2*nc*(nc-1)+(nc-1)**2)
            self.assertEqual([item["id"] for item in actions], list(range(len(actions))))
            self.assertEqual(len({item["cells"] for item in actions}), len(actions))
            self.assertEqual(actions, candidate_patches(nc))
            expected_dofs = dict(single=1, horizontal_pair=3, vertical_pair=3, block_2x2=8)
            for item in actions:
                self.assertEqual(count_added_dofs(item["cells"], nc), expected_dofs[item["kind"]])
                self.assertTrue(all(0 <= cell < nc*nc for cell in item["cells"]))
        for nc in (1, 2.5):
            with self.assertRaises(ValueError):
                candidate_patches(nc)


if __name__ == "__main__":
    unittest.main()
