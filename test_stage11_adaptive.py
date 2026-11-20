"""Constrained-mesh and fixed-budget checks independent of experimental scores."""

import itertools
import unittest

import numpy as np
import scipy.sparse.linalg as sla

from stage10_goal_numerics import assemble_nested
from stage11_adaptive import (budget_mark, build_full_space, build_space,
                              count_added_dofs, solve_adaptive)


class AdaptiveSpaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11001)
        self.case = assemble_nested(4, np.exp(rng.normal(size=(8, 8))))

    def test_empty_and_full_spaces_are_exact(self):
        case = self.case
        empty, meta = build_space(case, [])
        np.testing.assert_array_equal(empty.toarray(), case.P.toarray())
        self.assertEqual(meta["geometric_triangle_count"], 2*case.nc**2)
        full, meta = build_space(case, np.arange(case.nc**2))
        np.testing.assert_array_equal(full.toarray(), np.eye(case.dimension))
        self.assertEqual(meta["geometric_triangle_count"], 8*case.nc**2)
        self.assertEqual(meta["added_dofs"], case.dimension-case.P.shape[1])
        self.assertEqual(meta["hanging_constraints"], [])

    def test_full_affine_reproduction_and_partition(self):
        for selected in ([], [0], [0, 1, 4, 5, 10], np.arange(16)):
            full, metadata = build_full_space(4, selected)
            nodes = metadata["independent_full_node_ids"]
            xy = self.case.full_coordinates
            exact = 1.3 + 2.1*xy[:, 0] - .7*xy[:, 1]
            np.testing.assert_allclose(full @ exact[nodes], exact, atol=1e-15)
            np.testing.assert_allclose(np.asarray(full.sum(axis=1)).ravel(), 1.)

    def test_coarse_space_is_nested_and_boundary_is_zero(self):
        case = self.case
        rng = np.random.default_rng(11002)
        for selected in ([0], [0, 1, 4], [3, 7, 11, 15], np.arange(16)):
            T, metadata = build_space(case, selected)
            independent = case.full_to_free[metadata["independent_free_node_ids"]]
            coarse_injection = case.P[independent]
            np.testing.assert_array_equal((T @ coarse_injection).toarray(), case.P.toarray())
            full, fullmeta = build_full_space(case.nc, selected)
            independent_full = np.asarray(fullmeta["independent_full_node_ids"])
            ix = independent_full % (case.nf+1)
            iy = independent_full // (case.nf+1)
            coefficients = rng.normal(size=len(independent_full))
            coefficients[(ix == 0) | (ix == case.nf) | (iy == 0) | (iy == case.nf)] = 0.
            lifted = (full @ coefficients).reshape(case.nf+1, case.nf+1)
            np.testing.assert_array_equal(lifted[0], 0.)
            np.testing.assert_array_equal(lifted[-1], 0.)
            np.testing.assert_array_equal(lifted[:, 0], 0.)
            np.testing.assert_array_equal(lifted[:, -1], 0.)

    def test_shared_midpoints_hanging_constraints_and_mesh_area(self):
        case = self.case
        T, metadata = build_space(case, [0, 1])
        self.assertEqual(metadata["added_dofs"], 3)
        self.assertEqual(metadata["geometric_triangle_count"], 2*16+6*2)
        independent = set(metadata["independent_free_node_ids"])
        shared = (case.nf+1)+2
        self.assertIn(shared, independent)
        self.assertEqual([edge["node"] for edge in metadata["liberated_edges"]], [shared])
        for hanging in metadata["hanging_constraints"]:
            self.assertNotIn(hanging["node"], independent)
            row = case.full_to_free[hanging["node"]]
            endpoint_rows = case.full_to_free[hanging["endpoints"]]
            endpoint_sum = np.zeros(T.shape[1])
            for endpoint in endpoint_rows:
                if endpoint >= 0:
                    endpoint_sum += .5*T.getrow(endpoint).toarray().ravel()
            np.testing.assert_array_equal(T.getrow(row).toarray().ravel(), endpoint_sum)
        xy = case.full_coordinates[np.asarray(metadata["adaptive_mesh_triangles_full_node_ids"])]
        cross = np.cross(xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0])
        self.assertTrue(np.all(cross > 0))
        self.assertAlmostEqual(float(cross.sum()/2), 1.)

    def test_refined_space_is_nested_under_more_marking(self):
        first, first_meta = build_space(self.case, [0, 5, 7])
        second, second_meta = build_space(self.case, [0, 1, 5, 6, 7, 11])
        new_rows = self.case.full_to_free[second_meta["independent_free_node_ids"]]
        injection = first[new_rows]
        np.testing.assert_array_equal((second @ injection).toarray(), first.toarray())

    def test_energy_monotonicity_and_galerkin_orthogonality(self):
        case = self.case
        fine = sla.spsolve(case.Af, case.ff)
        errors = []
        for selection in ([], [5], [5, 6], [1, 5, 6, 9, 10], np.arange(16)):
            T, _ = build_space(case, selection)
            solution = solve_adaptive(case, T)
            error = fine-solution["lifted_u"]
            errors.append(float(error @ (case.Af @ error)))
            np.testing.assert_allclose(T.T @ (case.ff-case.Af @ solution["lifted_u"]),
                                       0., atol=1e-15)
            self.assertLess(solution["relative_residual"], 1e-12)
        self.assertTrue(np.all(np.diff(errors) < 0), errors)
        self.assertLess(errors[-1], 1e-25)

    def test_coarse_and_full_solves_match_reference(self):
        for selection, matrix, rhs, lift in (
                ([], self.case.Ac, self.case.fc, self.case.P),
                (np.arange(16), self.case.Af, self.case.ff, None)):
            T, _ = build_space(self.case, selection)
            actual = solve_adaptive(self.case, T)
            expected = sla.spsolve(matrix, rhs)
            if lift is not None:
                expected = lift @ expected
            np.testing.assert_allclose(actual["lifted_u"], expected, atol=1e-14)

    def test_invalid_space_selection_and_solver_shapes(self):
        for selection in ([1, 1], [-1], [16], [1.5], np.ones(15, dtype=bool)):
            with self.assertRaises(ValueError):
                build_space(self.case, selection)
        T, _ = build_space(self.case, [])
        with self.assertRaises(ValueError):
            solve_adaptive(self.case, T, np.ones(3))


class AdaptiveMarkingTests(unittest.TestCase):
    def test_budget_exact_for_every_supported_small_count(self):
        rng = np.random.default_rng(11003)
        for nc in (2, 3, 4, 7, 16):
            for budget in range((nc*nc+1)//2+1):
                for scores in (rng.random(nc*nc), np.ones(nc*nc)):
                    selected, metadata = budget_mark(scores, nc, budget)
                    self.assertEqual(count_added_dofs(selected, nc), budget)
                    self.assertEqual(metadata["added_dofs"], budget)
                    again, _ = budget_mark(scores, nc, budget)
                    np.testing.assert_array_equal(selected, again)

    def test_full_refinement_budget(self):
        cells, metadata = budget_mark(np.ones(16), 4, 40)
        np.testing.assert_array_equal(cells, np.arange(16))
        self.assertEqual(metadata["added_dofs"], 40)

    def test_nonattainable_budgets_are_not_silently_approximated(self):
        attainable = set()
        for bits in itertools.product((False, True), repeat=4):
            attainable.add(count_added_dofs(np.array(bits), 2))
        self.assertEqual(attainable, {0, 1, 2, 3, 5, 8})
        for budget in (4, 6, 7):
            with self.assertRaises(ValueError):
                budget_mark(np.ones(4), 2, budget)

    def test_fallback_is_explicit_and_exact(self):
        # A score ordering can first select a dominating set and leave only
        # candidates costing more than the last remaining DOF.
        nc = 4
        order = [6, 2, 0, 9, 12, 8, 13, 15, 14, 5, 10, 3, 7, 1, 4, 11]
        scores = np.empty(nc*nc)
        scores[order] = np.arange(nc*nc, 0, -1)
        cells, metadata = budget_mark(scores, nc, 8)
        self.assertTrue(metadata["fallback"])
        self.assertEqual(count_added_dofs(cells, nc), 8)
        y, x = np.divmod(cells, nc)
        self.assertTrue(np.all((x+y) % 2 == metadata["fallback_parity"]))

    def test_invalid_marking_arguments(self):
        for scores, budget in ((np.ones(15), 3), (np.full(16, np.nan), 3),
                               (np.ones(16), -1), (np.ones(16), 2.5)):
            with self.assertRaises(ValueError):
                budget_mark(scores, 4, budget)


if __name__ == "__main__":
    unittest.main()
