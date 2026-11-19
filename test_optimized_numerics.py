"""Equivalence and cache-invalidity tests for implementation-only changes."""

import unittest

import numpy as np

from numerics import (assemble_system, build_patches, build_preconditioner,
                      patch_spectral_basis, solve_pcg)
from optimized_numerics import (build_patches_cached,
                               build_preconditioner_sparse)


class SparseSchwarzTests(unittest.TestCase):
    def assert_patches_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.bounds, b.bounds)
            self.assertEqual(a.core_bounds, b.core_bounds)
            self.assertEqual(a.grid_shape, b.grid_shape)
            for name in ("ids", "weights", "coefficient", "A_N", "A_D", "B",
                         "local_flat_indices"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_matrix_equivalence_and_unchanged_reuse(self):
        rng = np.random.default_rng(28)
        for n in (32, 64):
            k = 10. ** rng.uniform(0., 4., size=(n, n))
            A, _, _ = assemble_system(k)
            original = build_patches(k, A)
            first = build_patches_cached(k, A)
            second = build_patches_cached(k, A, previous_patches=first)
            self.assert_patches_equal(first, original)
            self.assert_patches_equal(second, original)
            self.assertTrue(all(p._matrix_reused for p in second))
            self.assertTrue(all(a.A_N is b.A_N and a.B is b.B
                                for a, b in zip(first, second)))
            self.assertTrue(all(a is not b for a, b in zip(first, second)))

    def test_interior_and_halo_changes_invalidate_correct_blocks(self):
        k = np.ones((32, 32))
        A, _, _ = assemble_system(k)
        previous = build_patches_cached(k, A)
        # x=18 is outside patch 0's coefficient crop [0:18,0:18], but
        # contributes to A_D at the patch's rightmost vertex x=18.
        for changed_cell in ((10, 10), (10, 18)):
            changed = k.copy()
            changed[changed_cell] = 1e4
            current_A, _, _ = assemble_system(changed)
            cached = build_patches_cached(changed, current_A,
                                           previous_patches=previous)
            self.assert_patches_equal(cached, build_patches(changed, current_A))
            self.assertFalse(cached[0]._matrix_reused)
            self.assertFalse(np.array_equal(cached[0].A_D, previous[0].A_D))
            if changed_cell == (10, 18):
                np.testing.assert_array_equal(cached[0].A_N, previous[0].A_N)
                np.testing.assert_array_equal(cached[0].coefficient,
                                               previous[0].coefficient)
        # A caller changing A independently of k must not get a stale block.
        different_A = A.copy()
        different_A[0, 0] *= 2.
        cached = build_patches_cached(k, different_A, previous_patches=previous)
        self.assert_patches_equal(cached, build_patches(k, different_A))
        self.assertFalse(cached[0]._matrix_reused)

    def test_geometry_change_and_original_patch_interoperability(self):
        k = np.ones((32, 32))
        A, _, _ = assemble_system(k)
        old = build_patches(k, A)
        first = build_patches_cached(k, A, previous_patches=old)
        self.assert_patches_equal(first, old)
        changed = build_patches_cached(k, A, core_cells=8, overlap=1,
                                        previous_patches=first)
        self.assert_patches_equal(changed, build_patches(k, A, core_cells=8,
                                                       overlap=1))

    def test_factor_cache_counts(self):
        k = np.ones((32, 32))
        A, _, _ = assemble_system(k)
        first = build_patches_cached(k, A)
        empty = [np.empty((len(p.ids), 0)) for p in first]
        _, stats = build_preconditioner_sparse(A, first, empty)
        self.assertEqual(stats["new_local_factors"], 4)
        second = build_patches_cached(k, A, previous_patches=first)
        _, stats = build_preconditioner_sparse(A, second, empty)
        self.assertEqual(stats["new_local_factors"], 0)
        changed = k.copy()
        changed[2, 2] = 1e4
        changed_A, _, _ = assemble_system(changed)
        third = build_patches_cached(changed, changed_A, previous_patches=second)
        _, stats = build_preconditioner_sparse(changed_A, third, empty)
        self.assertEqual(stats["new_local_factors"], 1)
        self.assertEqual(sum(p._matrix_reused for p in third), 3)

    def test_operator_equivalence_symmetry_positivity_and_pcg(self):
        k = np.ones((32, 32))
        k[9:13, :] = 1e4
        k[:, 18:21] = 1e4
        A, b, _ = assemble_system(k)
        patches = build_patches_cached(k, A)
        spaces = [patch_spectral_basis(p) for p in patches]
        dense, _ = build_preconditioner(A, patches, spaces)
        sparse, stats = build_preconditioner_sparse(A, patches, spaces,
                                                     profile=True)
        rng = np.random.default_rng(29)
        for _ in range(3):
            x, y = rng.normal(size=(2, A.shape[0]))
            mx, my = sparse @ x, sparse @ y
            np.testing.assert_allclose(mx, dense @ x, rtol=2e-8, atol=2e-9)
            self.assertGreater(float(x @ mx), 0.)
            self.assertAlmostEqual(float(x @ my) / float(y @ mx), 1., places=8)
        self.assertEqual(stats["apply_calls"], 6)
        self.assertEqual(stats["local_apply_calls"], 24)
        self.assertEqual(stats["coarse_apply_calls"], 6)
        self.assertGreater(stats["apply_local_seconds"], 0.)
        self.assertGreater(stats["apply_coarse_seconds"], 0.)
        xd, dense_info = solve_pcg(A, b, dense, rtol=1e-8)
        xs, sparse_info = solve_pcg(A, b, sparse, rtol=1e-8)
        self.assertTrue(dense_info["converged"])
        self.assertTrue(sparse_info["converged"])
        np.testing.assert_allclose(xs, xd, rtol=2e-7, atol=1e-10)
        # Sparse symmetry and positivity also hold without a coarse space.
        no_coarse, no_coarse_stats = build_preconditioner_sparse(
            A, patches, [np.empty((len(p.ids), 0)) for p in patches])
        self.assertGreater(float(x @ (no_coarse @ x)), 0.)
        self.assertAlmostEqual(float(x @ (no_coarse @ y)) /
                               float(y @ (no_coarse @ x)), 1., places=8)
        self.assertEqual(no_coarse_stats["apply_calls"], 0)


if __name__ == "__main__":
    unittest.main()
