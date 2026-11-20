"""Numerical equivalence and cost-boundary tests for direct mixed assembly."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from stage10_goal_numerics import assemble_load, assemble_nested
from stage11_adaptive import budget_mark, build_space, count_added_dofs
from stage13_local_fem import assemble_local, lift_to_fine, prepare_loads


def functions():
    return (None, lambda x, y: x - 2 * y,
            lambda x, y: np.exp(-((x - .31) ** 2 + (y - .63) ** 2) / .013),
            lambda x, y: np.sin(5 * x) * np.cos(7 * y) + x ** 2)


class LocalAssemblyTests(unittest.TestCase):
    def assert_equivalent(self, nc, coefficient, selected):
        funcs = functions()
        reference = assemble_nested(nc, coefficient)
        fine_loads = np.stack([assemble_load(reference, f)[0] for f in funcs])
        T, metadata = build_space(reference, selected)
        direct = assemble_local(nc, coefficient, funcs, selected)
        expected_A = (T.T @ reference.Af @ T).tocsc()
        expected_loads = fine_loads @ T
        difference = direct.A - expected_A
        scale = np.max(np.abs(expected_A.data))
        self.assertLess(np.max(np.abs(difference.data), initial=0.) / scale, 5e-15)
        np.testing.assert_allclose(direct.loads, expected_loads, rtol=2e-14, atol=5e-17)
        np.testing.assert_array_equal(direct.independent_free_node_ids,
                                      metadata["independent_free_node_ids"])
        self.assertEqual(direct.dimension, T.shape[1])
        self.assertEqual(direct.metadata["geometric_triangle_count"],
                         metadata["geometric_triangle_count"])
        self.assertEqual(direct.metadata["added_dofs"], count_added_dofs(selected, nc))
        u = sla.splu(direct.A).solve(direct.loads.T)
        expected_u = sla.splu(expected_A).solve(expected_loads.T)
        np.testing.assert_allclose(u, expected_u, rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(lift_to_fine(direct, u), T @ u, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(direct.loads[1:] @ u[:, 0],
                                   fine_loads[1:] @ (T @ expected_u[:, 0]),
                                   rtol=2e-10, atol=1e-13)
        relative_residual = la.norm(direct.A @ u - direct.loads.T) / la.norm(direct.loads)
        self.assertLess(relative_residual, 1e-10)
        return direct

    def test_empty_selection_matches_coarse_restriction(self):
        rng = np.random.default_rng(1301)
        for nc in (2, 3, 8):
            with self.subTest(nc=nc):
                self.assert_equivalent(nc, np.exp(rng.normal(size=(2 * nc, 2 * nc))), [])

    def test_full_selection_matches_fine_operator(self):
        rng = np.random.default_rng(1302)
        for nc in (2, 5):
            with self.subTest(nc=nc):
                self.assert_equivalent(nc, rng.uniform(.1, 7, (2 * nc, 2 * nc)),
                                       np.arange(nc * nc))

    def test_random_partial_high_contrast_and_signed_loads(self):
        rng = np.random.default_rng(1303)
        for nc in (3, 5, 8):
            with self.subTest(nc=nc):
                k = rng.choice([1., 1000.], size=(2 * nc, 2 * nc))
                selected = rng.choice(nc * nc, nc * nc // 3, replace=False)
                self.assert_equivalent(nc, k, selected)

    def test_single_boundary_cell_and_shared_edge_constraints(self):
        nc = 4
        for selected in ([0], [0, 1], [0, 4], [5, 6, 9, 10], [0, 3, 12, 15]):
            with self.subTest(selected=selected):
                self.assert_equivalent(nc, 1., selected)

    def test_exact_budget_selection_dimension(self):
        nc = 8
        scores = np.random.default_rng(1304).normal(size=nc * nc)
        for budget in (0, 1, 16, 32):
            selected, marker = budget_mark(scores, nc, budget)
            with self.subTest(budget=budget):
                result = self.assert_equivalent(nc, 3., selected)
                self.assertEqual(result.dimension, (nc - 1) ** 2 + budget)
                self.assertEqual(result.metadata["added_dofs"], marker["added_dofs"])

    def test_boolean_selection_and_node_order(self):
        selection = np.zeros((4, 4), dtype=bool)
        selection[[0, 1, 1], [1, 1, 2]] = True
        direct = self.assert_equivalent(4, 2., selection)
        self.assertTrue(np.all(np.diff(direct.independent_free_node_ids) > 0))

    def test_effective_diffusion_formula_both_coarse_orientations(self):
        # Unequal values ensure exchanging lower/upper weights cannot pass.
        from stage13_local_fem import _mixed_triangles
        k = np.arange(1., 17.).reshape(4, 4) ** 2
        _, integrated, orientations = _mixed_triangles(2, k, np.zeros((2, 2), bool))
        blocks = k.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 2, 2)
        lower = (blocks[:, 0, 0] + 2 * blocks[:, 0, 1] + blocks[:, 1, 1]) / 4
        upper = (blocks[:, 0, 0] + 2 * blocks[:, 1, 0] + blocks[:, 1, 1]) / 4
        np.testing.assert_array_equal(integrated.reshape(-1, 2), np.column_stack((lower, upper)))
        np.testing.assert_array_equal(orientations, np.tile([0, 1], 4))
        self.assert_equivalent(2, k, [])

    def test_symmetry_and_positive_definiteness(self):
        rng = np.random.default_rng(1305)
        for selected in ([], [0, 1, 4, 5, 8], np.arange(16)):
            result = assemble_local(4, np.exp(rng.uniform(0, np.log(1000), (8, 8))),
                                    functions(), selected)
            matrix = result.A.toarray()
            np.testing.assert_allclose(matrix, matrix.T, rtol=1e-15, atol=1e-13)
            self.assertGreater(la.eigvalsh(matrix)[0], 0.)

    def test_load_preparation_reuse_is_exact_and_avoids_reevaluation(self):
        call_shapes = []
        def source(x, y):
            call_shapes.append(x.shape)
            return x ** 2 - y ** 3
        funcs = [source, -2., None]
        first = assemble_local(4, 1., funcs, [])
        self.assertEqual(call_shapes, [(64, 2, 3)])
        reused = assemble_local(4, 1., funcs, [0, 1, 5], load_data=first.load_data)
        self.assertEqual(call_shapes, [(64, 2, 3)])
        self.assertIs(reused.load_data, first.load_data)
        self.assertEqual(reused.timings["load_prepare_seconds"], 0.)
        self.assertGreater(reused.timings["load_projection_seconds"], 0.)
        self.assertTrue(reused.metadata["reused_load_data"])
        fresh = assemble_local(4, 1., funcs, [0, 1, 5])
        np.testing.assert_array_equal(reused.loads, fresh.loads)
        self.assertFalse(first.load_data.fine_loads_full.flags.writeable)
        reference = assemble_nested(4, 1.)
        expected = np.stack([assemble_load(reference, f)[0] for f in funcs])
        np.testing.assert_allclose(first.load_data.fine_free_loads, expected, atol=2e-17)

    def test_only_target_sparse_matrix_is_constructed(self):
        sizes = []
        original = sp.coo_matrix
        def record(*args, **kwargs):
            sizes.append(kwargs.get("shape"))
            return original(*args, **kwargs)
        with patch("stage13_local_fem.sp.coo_matrix", side_effect=record), \
             patch("stage10_goal_numerics.assemble_nested", side_effect=AssertionError("fine reference")), \
             patch("stage11_adaptive.build_space", side_effect=AssertionError("global T")):
            result = assemble_local(4, 1., functions(), [0, 1, 5])
        self.assertEqual(sizes, [(result.dimension, result.dimension)])
        self.assertLess(result.dimension, 49)

    def test_scalar_load_and_coefficient_agree_with_array(self):
        a = assemble_local(3, 7., [None, 1., -4.], [0, 1])
        b = assemble_local(3, np.full((6, 6), 7.), [1., 1., -4.], [0, 1])
        np.testing.assert_array_equal(a.A.toarray(), b.A.toarray())
        np.testing.assert_array_equal(a.loads, b.loads)
        np.testing.assert_allclose(a.loads[2], -4 * a.loads[0], rtol=0, atol=0)

    def test_invalid_inputs_and_reuse_contract(self):
        for nc in (1, 2.5):
            with self.assertRaises(ValueError):
                assemble_local(nc, 1., [None], [])
        for coefficient in (0., -1., np.nan, np.ones((3, 3))):
            with self.assertRaises(ValueError):
                assemble_local(3, coefficient, [None], [])
        for selection in ([0, 0], [-1], [9], [1.5], [[0]]):
            with self.assertRaises(ValueError):
                assemble_local(3, 1., [None], selection)
        with self.assertRaises(ValueError):
            assemble_local(3, 1., [], [])
        with self.assertRaises(ValueError):
            prepare_loads(3, [lambda x, y: np.nan])
        with self.assertRaises(TypeError):
            prepare_loads(3, [np.ones(3)])
        data = prepare_loads(3, [None, 1.])
        with self.assertRaises(ValueError):
            assemble_local(4, 1., [None, 1.], [], load_data=data)
        with self.assertRaises(ValueError):
            assemble_local(3, 1., [None, 2.], [], load_data=data)
        with self.assertRaises(ValueError):
            assemble_local(3, 1., [None], [], load_data=data)
        problem = assemble_local(3, 1., [None], [])
        with self.assertRaises(ValueError):
            lift_to_fine(problem, np.zeros(problem.dimension + 1))


if __name__ == "__main__":
    unittest.main()
