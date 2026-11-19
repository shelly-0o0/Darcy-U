"""Interpolation, current-matrix correctness, and energy checks."""

import unittest

import numpy as np

from coarse_correction import CoarseGalerkinCorrection, _bilinear_prolongation
from numerics import assemble_system, free_to_grid


class CoarseCorrectionTests(unittest.TestCase):
    def test_interpolation_injection_partition_and_zero_boundary(self):
        rng = np.random.default_rng(51)
        for n in (64, 128, 256):
            coarse_n, ratio = 32, n // 32
            P = _bilinear_prolongation(n, coarse_n)
            coarse = rng.normal(size=(coarse_n - 1) ** 2)
            fine = free_to_grid(P @ coarse, n)
            np.testing.assert_array_equal(fine[ratio:n:ratio, ratio:n:ratio].ravel(), coarse)
            self.assertTrue(np.all(fine[[0, -1], :] == 0.))
            self.assertTrue(np.all(fine[:, [0, -1]] == 0.))
            coordinate = np.arange(1, n) / ratio
            one_dim_sum = np.minimum(np.minimum(coordinate, coarse_n - coordinate), 1.)
            np.testing.assert_allclose((P @ np.ones_like(coarse)).reshape(n - 1, n - 1),
                                       one_dim_sum[:, None] * one_dim_sum[None, :],
                                       rtol=0., atol=0.)
            self.assertLessEqual(np.max(np.diff(P.indptr)), 4)

    def test_bilinear_basis_tent_values(self):
        n, m = 64, 8
        P = _bilinear_prolongation(n, m)
        coarse = np.zeros((m - 1, m - 1))
        coarse[2, 4] = 1.
        fine = free_to_grid(P @ coarse.ravel(), n)
        axis = np.arange(n + 1) / (n // m)
        expected = np.maximum(1. - abs(axis - 3.), 0.)[:, None] * \
            np.maximum(1. - abs(axis - 5.), 0.)[None, :]
        np.testing.assert_allclose(fine, expected, rtol=0., atol=0.)

    def test_high_contrast_coarse_orthogonality_and_energy(self):
        rng = np.random.default_rng(52)
        n = 64
        k = np.ones((n, n))
        k[21:25, :] = 1e4
        k[:, 35:39] = 1e4
        A, b, _ = assemble_system(k)
        x0 = .001 * rng.normal(size=len(b))
        correction = CoarseGalerkinCorrection(n)
        x, stats = correction.apply(A, b, x0)
        self.assertTrue(stats["accepted"])
        self.assertTrue(stats["interpolation_built"])
        self.assertEqual(stats["coarse_dimension"], 31 ** 2)
        coarse_residual_before = correction.P.T @ (b - A @ x0)
        coarse_residual_after = correction.P.T @ (b - A @ x)
        self.assertLess(np.linalg.norm(coarse_residual_after) /
                        np.linalg.norm(coarse_residual_before), 1e-10)
        energy = lambda z: .5 * z @ (A @ z) - b @ z
        self.assertLessEqual(energy(x), energy(x0) + 1e-10)
        np.testing.assert_allclose(stats["energy_change"], energy(x) - energy(x0), rtol=1e-11)

    def test_current_matrix_and_rhs_are_rebuilt_every_call(self):
        n = 64
        k = np.ones((n, n))
        A, b, _ = assemble_system(k)
        correction = CoarseGalerkinCorrection(n)
        first, _ = correction.apply(A, b, None)
        P = correction.P
        changed = k.copy()
        changed[13:20, :] = 1e4
        current_A, _, _ = assemble_system(changed)
        current_b = b * 1.7
        reused, stats = correction.apply(current_A, current_b, first)
        fresh, _ = CoarseGalerkinCorrection(n).apply(current_A, current_b, first)
        self.assertIs(correction.P, P)
        self.assertFalse(stats["interpolation_built"])
        self.assertEqual(stats["interpolation_seconds"], 0.)
        self.assertTrue(stats["coarse_matrix_rebuilt"])
        np.testing.assert_array_equal(reused, fresh)
        # With A scaled globally and zero initial guess the result scales inversely.
        scaled, _ = correction.apply(17. * A, b, None)
        np.testing.assert_allclose(scaled, first / 17., rtol=2e-12, atol=1e-14)

    def test_validation_and_equal_grid_identity(self):
        with self.assertRaises(ValueError):
            CoarseGalerkinCorrection(100, 32)
        P = _bilinear_prolongation(32, 32)
        np.testing.assert_array_equal(P.toarray(), np.eye(31 ** 2))
        A, b, _ = assemble_system(np.ones((8, 8)))
        x, stats = CoarseGalerkinCorrection(8, 8).apply(A, b, None)
        self.assertLess(np.linalg.norm(b - A @ x) / np.linalg.norm(b), 1e-12)
        self.assertTrue(stats["accepted"])


if __name__ == "__main__":
    unittest.main()
