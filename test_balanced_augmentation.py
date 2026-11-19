"""Algebraic and causal-input contracts of balanced augmentation."""

import unittest

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from balanced_augmentation import (a_orthonormalize, balanced_preconditioner,
                                   extract_slow_ritz, galerkin_correction)


class BalancedAugmentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(140)
        factor = rng.normal(size=(12, 12))
        self.A = factor.T @ factor + np.diag(np.linspace(.2, 2., 12))
        second = rng.normal(size=(12, 12))
        self.M = second.T @ second + np.eye(12)
        self.P = rng.normal(size=(12, 4))
        self.b, self.x0 = rng.normal(size=(2, 12))

    def test_scaled_dependent_invalid_columns_and_refined_orthogonality(self):
        u, v = self.P[:, 0], self.P[:, 1]
        basis = np.column_stack([1e250 * u, 1e-250 * v, u + v,
                                 np.zeros(12), np.full(12, np.nan)])
        U, AU, info = a_orthonormalize(self.A, basis)
        self.assertEqual(info["rank"], 2)
        self.assertEqual(info["valid_vectors"], 3)
        self.assertEqual(info["rejected_vectors"], 2)
        self.assertEqual(info["dependent_vectors"], 1)
        np.testing.assert_allclose(U.T @ AU, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(AU, self.A @ U, atol=1e-12)
        near = np.column_stack([u, u + 1e-3 * v])
        refined, action, info = a_orthonormalize(self.A, near)
        self.assertEqual(info["rank"], 2)
        np.testing.assert_allclose(refined.T @ action, np.eye(2), atol=2e-12)

    def test_balanced_formula_symmetry_positivity_and_exact_selected_modes(self):
        U, AU, _ = a_orthonormalize(self.A, self.P)
        operator = balanced_preconditioner(self.M, U, AU)
        dense = operator @ np.eye(12)
        left = np.eye(12) - U @ AU.T
        expected = U @ U.T + left @ self.M @ left.T
        np.testing.assert_allclose(dense, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dense, dense.T, rtol=1e-12, atol=1e-12)
        self.assertGreater(la.eigvalsh(dense)[0], 0.)
        np.testing.assert_allclose(operator @ AU, U, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(operator.T @ self.b, operator @ self.b,
                                   rtol=1e-13, atol=1e-13)
        saved = operator @ self.b
        U.fill(0.)
        AU.fill(0.)
        np.testing.assert_array_equal(operator @ self.b, saved)

    def test_preconditioner_application_reuses_cached_A_actions(self):
        counts = {"A": 0, "M": 0}

        def apply_A(x):
            counts["A"] += 1
            return self.A @ x

        def apply_M(x):
            counts["M"] += 1
            return self.M @ x

        A = sla.LinearOperator((12, 12), matvec=apply_A, dtype=float)
        M = sla.LinearOperator((12, 12), matvec=apply_M, dtype=float)
        U, AU, _ = a_orthonormalize(A, self.P)
        setup_actions = counts["A"]
        augmented = balanced_preconditioner(M, U, AU)
        augmented @ self.b
        augmented @ self.x0
        self.assertEqual(counts["A"], setup_actions)
        self.assertEqual(counts["M"], 2)

    def test_current_matrix_rebuild_and_galerkin_energy_identity(self):
        previous_U, _, _ = a_orthonormalize(self.A, self.P)
        current_A = self.A + np.diag(np.linspace(1., 20., 12))
        U, AU, _ = a_orthonormalize(current_A, previous_U)
        augmented = balanced_preconditioner(self.M, U, AU)
        np.testing.assert_allclose(augmented @ (current_A @ U), U, atol=1e-11)
        corrected, info = galerkin_correction(current_A, self.b, self.x0, U, AU)
        np.testing.assert_allclose(U.T @ (self.b - current_A @ corrected),
                                   np.zeros(4), atol=1e-12)
        energy = lambda x: .5 * x @ current_A @ x - self.b @ x
        self.assertLessEqual(energy(corrected), energy(self.x0))
        self.assertAlmostEqual(info["energy_change"], energy(corrected) - energy(self.x0),
                               places=10)
        coefficients = U.T @ (self.b - current_A @ self.x0)
        self.assertAlmostEqual(info["energy_change"], -.5 * coefficients @ coefficients,
                               places=10)
        without_cache, _ = galerkin_correction(current_A, self.b, self.x0, U)
        np.testing.assert_allclose(without_cache, corrected, atol=1e-13)

    def test_rank_zero_fallback_and_invalid_contracts(self):
        U, AU, info = a_orthonormalize(self.A, np.zeros((12, 3)))
        self.assertEqual(info["rank"], 0)
        M = sla.aslinearoperator(self.M)
        self.assertIs(balanced_preconditioner(M, U, AU), M)
        x, info = galerkin_correction(self.A, self.b, self.x0, U, AU)
        np.testing.assert_array_equal(x, self.x0)
        self.assertEqual(info["status"], "empty_basis")
        with self.assertRaises(ValueError):
            a_orthonormalize(self.A, self.P.T)
        with self.assertRaises(ValueError):
            a_orthonormalize(self.A, self.P, rtol=0.)
        with self.assertRaises(ValueError):
            balanced_preconditioner(self.M, self.P, self.A @ self.P)
        with self.assertRaises(ValueError):
            galerkin_correction(self.A, self.b * np.nan, self.x0, U, AU)
        U, AU, info = a_orthonormalize(np.diag([-1., 1.]), np.eye(2))
        self.assertEqual(info["rank"], 1)
        self.assertEqual(info["rejected_vectors"], 1)


class SlowRitzTests(unittest.TestCase):
    def test_known_diagonal_slow_modes_and_basis_limit(self):
        A = sp.diags([1., 2., 4., 8., 16.], format="csr")
        M = sp.diags([.01, .025, .25, .25, .25], format="csr")
        U, AU, info = extract_slow_ritz(A, M, np.eye(5), q=2)
        np.testing.assert_allclose(info["ritz_values"], [.01, .05], atol=1e-14)
        np.testing.assert_allclose(U[2:, :], 0., atol=1e-14)
        np.testing.assert_allclose(U.T @ AU, np.eye(2), atol=1e-14)
        self.assertLess(max(info["operator_ritz_residuals"]), 1e-13)
        self.assertLess(max(info["projected_ritz_residuals"]), 1e-13)
        _, _, limited = extract_slow_ritz(A, M, np.eye(5), q=8, max_basis=3)
        self.assertEqual(limited["basis"]["input_vectors"], 5)
        self.assertEqual(limited["basis"]["selected_vectors"], 3)
        self.assertEqual(limited["rank"], 3)
        np.testing.assert_allclose(limited["ritz_values"], [1., 2., 4.], atol=1e-13)

    def test_nondiagonal_full_span_generalized_ritz_equations(self):
        rng = np.random.default_rng(141)
        factor = rng.normal(size=(8, 8))
        A = factor.T @ factor + .2 * np.eye(8)
        factor = rng.normal(size=(8, 8))
        M = factor.T @ factor + .5 * np.eye(8)
        P = rng.normal(size=(8, 8))
        U, AU, info = extract_slow_ritz(A, M, P, q=4)
        expected = np.sort(la.eigvals(M @ A).real)[:4]
        np.testing.assert_allclose(info["ritz_values"], expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(M @ AU, U * expected[None, :], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(U.T @ AU, np.eye(4), rtol=1e-11, atol=1e-11)
        self.assertLess(max(info["operator_ritz_residuals"]), 1e-10)

    def test_restricted_span_projection_and_rank_degeneracy(self):
        rng = np.random.default_rng(142)
        A = np.diag(np.arange(1., 10.))
        M = np.diag(np.linspace(.1, 2., 9))
        P = rng.normal(size=(9, 3))
        P = np.column_stack([P, P[:, 0] * 1e240, np.zeros(9), np.full(9, np.inf)])
        U, AU, info = extract_slow_ritz(A, M, P, q=8)
        self.assertEqual(info["rank"], 3)
        self.assertEqual(info["basis"]["rejected_vectors"], 2)
        values = np.array(info["ritz_values"])
        # Ritz residuals are A-orthogonal to the supplied retained span;
        # they need not vanish outside that span.
        np.testing.assert_allclose(AU.T @ (M @ AU - U * values[None, :]),
                                   np.zeros((3, 3)), atol=1e-12)
        self.assertLess(max(info["projected_ritz_residuals"]), 1e-12)
        self.assertGreater(max(info["operator_ritz_residuals"]), 1e-3)

    def test_empty_and_tiny_positive_ritz_values(self):
        A = sp.eye(3, format="csr")
        M = sp.diags([1e-14, 1., 2.], format="csr")
        U, AU, info = extract_slow_ritz(A, M, np.empty((3, 0)), q=4)
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(info["rank"], 0)
        _, _, info = extract_slow_ritz(A, M, np.eye(3), q=1)
        self.assertAlmostEqual(info["ritz_values"][0], 1e-14, delta=1e-28)
        with self.assertRaises(ValueError):
            extract_slow_ritz(A, M, np.eye(3), q=0)


if __name__ == "__main__":
    unittest.main()
