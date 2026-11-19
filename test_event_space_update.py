"""Budget and Rayleigh--Ritz checks for current-system space expansion."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from balanced_augmentation import a_orthonormalize, extract_slow_ritz
from event_space_update import update_slow_space


class EventSpaceUpdateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6101)
        factor = rng.normal(size=(18, 18))
        self.A = factor.T @ factor + 2. * np.eye(18)
        factor = rng.normal(size=(18, 18))
        self.M = factor.T @ factor + np.eye(18)
        self.P = rng.normal(size=(18, 4))

    def test_noncommuting_spd_matches_explicit_extended_space(self):
        new_U, new_AU, stats = update_slow_space(self.A, self.M, self.P, q=4)
        old_U, old_AU, _ = a_orthonormalize(self.A, self.P)
        W = np.column_stack([old_U, self.M @ old_AU])
        expected, expected_AU, reference = extract_slow_ritz(self.A, self.M, W, q=4)
        self.assertEqual(stats["status"], "accepted")
        self.assertEqual(stats["input_rank"], 4)
        self.assertEqual(stats["expanded_vectors"], 8)
        self.assertEqual(stats["output_rank"], 4)
        np.testing.assert_allclose(stats["ritz_values"], reference["ritz_values"], rtol=2e-11)
        np.testing.assert_allclose(new_AU, self.A @ new_U, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(new_U.T @ new_AU, np.eye(4), atol=1e-11)
        # Sign and within-cluster rotations do not change the retained space.
        np.testing.assert_allclose(la.svdvals(new_U.T @ expected_AU), np.ones(4), atol=1e-10)

    def test_actual_action_counts_equal_budget_and_no_duplicate_actions(self):
        counts = {"a": 0, "m": 0}

        def apply_a(v):
            counts["a"] += 1
            return self.A @ v

        def apply_m(v):
            counts["m"] += 1
            return self.M @ v

        A = sla.LinearOperator(self.A.shape, matvec=apply_a, dtype=float)
        M = sla.LinearOperator(self.M.shape, matvec=apply_m, dtype=float)
        _, _, stats = update_slow_space(A, M, self.P, q=4)
        self.assertEqual(counts, {"a": 8, "m": 8})
        self.assertEqual(stats["a_applications"], counts["a"])
        self.assertEqual(stats["m_applications"], counts["m"])
        self.assertEqual(stats["orthonormalization_a_applications"], 4)
        self.assertEqual(stats["extension_a_applications"], 4)
        self.assertEqual(stats["extension_m_applications"], 8)
        self.assertEqual(stats["max_a_applications"], 8)
        self.assertEqual(stats["max_m_applications"], 8)
        self.assertEqual(stats["extraction"]["a_applications"], 0)
        self.assertEqual(stats["extraction"]["m_applications"], 0)

    def test_current_matrix_reorthogonalization_and_ritz_nonincrease(self):
        previous_A = self.A + np.diag(np.linspace(1., 20., 18))
        old_U, _, _ = a_orthonormalize(previous_A, self.P)
        self.assertGreater(la.norm(old_U.T @ self.A @ old_U - np.eye(4)), .1)
        new_U, new_AU, stats = update_slow_space(sp.csr_matrix(self.A),
                                                 sp.csr_matrix(self.M), old_U, q=4)
        np.testing.assert_allclose(new_U.T @ new_AU, np.eye(4), atol=1e-11)
        current_U, current_AU, _ = a_orthonormalize(self.A, old_U)
        old_H = current_AU.T @ self.M @ current_AU
        old_G = current_U.T @ current_AU
        old_values = la.eigvalsh(.5 * (old_H + old_H.T), .5 * (old_G + old_G.T))
        np.testing.assert_allclose(stats["old_ritz_values"], old_values, rtol=1e-12)
        self.assertTrue(np.all(np.array(stats["ritz_values"]) <= old_values * (1. + 1e-11)))
        self.assertLess(stats["max_relative_ritz_increase"], 1e-11)

    def test_exact_invariant_space_degeneracy_preserves_modes(self):
        A = sp.diags([1., 2., 4., 8., 16., 32.], format="csr")
        M = sp.diags([.01, .025, .25, .25, .25, .25], format="csr")
        old_U = np.eye(6)[:, :4]
        new_U, new_AU, stats = update_slow_space(A, M, old_U, q=4)
        self.assertEqual(stats["status"], "accepted")
        self.assertEqual(stats["extraction"]["basis_rank"], 4)
        np.testing.assert_allclose(stats["ritz_values"], [.01, .05, 1., 2.], atol=1e-12)
        np.testing.assert_allclose(M @ new_AU,
                                   new_U * np.array(stats["ritz_values"])[None, :], atol=1e-12)
        self.assertLess(stats["max_relative_ritz_increase"], 1e-12)

    def test_scaled_dependent_nonfinite_and_zero_old_columns(self):
        P = np.column_stack([self.P[:, 0] * 1e230, self.P[:, 1] * 1e-230,
                             self.P[:, 0], np.zeros(18), np.full(18, np.nan)])
        U, AU, stats = update_slow_space(self.A, self.M, P, q=4)
        self.assertEqual(stats["input_vectors"], 5)
        self.assertEqual(stats["input_rank"], 2)
        self.assertEqual(stats["basis"]["rejected_vectors"], 2)
        self.assertEqual(stats["basis"]["dependent_vectors"], 1)
        self.assertEqual(stats["orthonormalization_a_applications"], 3)
        self.assertEqual(stats["a_applications"], 5)
        self.assertEqual(stats["m_applications"], 4)
        self.assertEqual(stats["rank"], 4)
        np.testing.assert_allclose(U.T @ AU, np.eye(4), atol=1e-11)

    def test_empty_and_fully_invalid_inputs_take_no_actions(self):
        calls = []

        def forbidden(v):
            calls.append(1)
            raise AssertionError("an empty or invalid basis must not trigger actions")

        A = sla.LinearOperator((3, 3), matvec=forbidden, dtype=float)
        M = sla.LinearOperator((3, 3), matvec=forbidden, dtype=float)
        for directions in (None, np.empty((3, 0)), []):
            U, AU, stats = update_slow_space(A, M, directions)
            self.assertEqual(U.shape, (3, 0))
            self.assertEqual(AU.shape, (3, 0))
            self.assertEqual(stats["status"], "empty_basis")
            self.assertEqual(stats["a_applications"], 0)
            self.assertEqual(stats["m_applications"], 0)
        _, _, stats = update_slow_space(A, M, np.column_stack([np.zeros(3), np.full(3, np.inf)]))
        self.assertEqual(stats["status"], "orthonormalization_rejected")
        self.assertEqual(stats["basis"]["status"], "no_valid_direction")
        self.assertEqual(calls, [])

    def test_invalid_new_actions_and_asymmetry_have_explicit_rejections(self):
        A = sp.eye(3, format="csr")
        bad_M = sla.LinearOperator((3, 3), matvec=lambda v: np.full(3, np.nan), dtype=float)
        U, _, stats = update_slow_space(A, bad_M, np.eye(3)[:, :2], q=2)
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(stats["status"], "invalid_m_action")
        self.assertEqual(stats["a_applications"], 2)
        self.assertEqual(stats["m_applications"], 1)
        asymmetric = np.eye(3)
        asymmetric[0, 1] = .2
        _, _, stats = update_slow_space(A, asymmetric, np.eye(3)[:, :2], q=2)
        self.assertTrue(stats["status"].startswith("extraction_rejected:"))
        self.assertEqual(stats["extraction"]["status"], "cache_symmetry_mismatch")

    def test_bad_arguments(self):
        for q in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                update_slow_space(self.A, self.M, self.P, q=q)
        with self.assertRaises(ValueError):
            update_slow_space(self.A, self.M, self.P.T)
        with self.assertRaises(ValueError):
            update_slow_space(self.A, self.M, self.P, symmetry_tol=0.)
        with self.assertRaises(ValueError):
            update_slow_space(self.A, self.M, self.P.astype(complex))

    def test_orthonormalization_decomposition_failure_preserves_zero_or_paid_budget(self):
        for target, expected_a in (("event_space_update.a_orthonormalize", 0),
                                    ("balanced_augmentation.la.eigh", 4)):
            with self.subTest(target=target):
                with patch(target, side_effect=la.LinAlgError("injected failure")):
                    U, AU, stats = update_slow_space(self.A, self.M, self.P)
                self.assertEqual(U.shape, (18, 0))
                self.assertEqual(AU.shape, (18, 0))
                self.assertEqual(stats["status"],
                                 "numerical_decomposition_failure:orthonormalization")
                self.assertEqual(stats["numerical_failure_stage"], "orthonormalization")
                self.assertEqual(stats["a_applications"], expected_a)
                self.assertEqual(stats["orthonormalization_a_applications"], expected_a)
                self.assertEqual(stats["m_applications"], 0)
                self.assertEqual(stats["output_rank"], 0)

    def test_later_decomposition_failures_retain_complete_action_budget(self):
        for target, stage in (("event_space_update.cached_slow_ritz", "extraction"),
                              ("event_space_update.la.eigvalsh", "old_spectrum")):
            with self.subTest(stage=stage):
                with patch(target, side_effect=la.LinAlgError("injected failure")):
                    U, AU, stats = update_slow_space(self.A, self.M, self.P)
                self.assertEqual(U.shape, (18, 0))
                self.assertEqual(AU.shape, (18, 0))
                self.assertEqual(stats["status"], "numerical_decomposition_failure:" + stage)
                self.assertEqual(stats["numerical_failure_stage"], stage)
                self.assertEqual(stats["a_applications"], 8)
                self.assertEqual(stats["m_applications"], 8)
                self.assertEqual(stats["orthonormalization_a_applications"], 4)
                self.assertEqual(stats["extension_a_applications"], 4)
                self.assertEqual(stats["extension_m_applications"], 8)
                self.assertEqual(stats["input_rank"], 4)
                self.assertEqual(stats["output_rank"], 0)

    def test_programming_exceptions_are_not_suppressed(self):
        for target in ("event_space_update.a_orthonormalize",
                       "event_space_update.cached_slow_ritz",
                       "event_space_update.la.eigvalsh"):
            with self.subTest(target=target):
                with patch(target, side_effect=ValueError("injected programming failure")):
                    with self.assertRaisesRegex(ValueError, "programming failure"):
                        update_slow_space(self.A, self.M, self.P)


if __name__ == "__main__":
    unittest.main()
