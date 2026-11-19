"""Small SPD correctness, paid-action accounting, and cached Ritz checks."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from balanced_augmentation import extract_slow_ritz
from cached_pcg import cached_slow_ritz, solve_pcg_cached
from numerics import solve_pcg


class CachedPCGTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5101)
        factor = rng.normal(size=(23, 23))
        self.A = sp.csr_matrix(factor.T @ factor + 3. * np.eye(23))
        self.M = sp.diags(1. / self.A.diagonal(), format="csr")
        self.b, self.x0 = rng.normal(size=(2, 23))

    def test_solutions_match_scipy_and_capture_does_not_change_recurrence(self):
        expected, original = solve_pcg(self.A, self.b, M=self.M,
                                        x0=self.x0, rtol=1e-11, maxiter=100)
        plain, first, empty = solve_pcg_cached(self.A, self.b, M=self.M,
                                              x0=self.x0, rtol=1e-11,
                                              maxiter=100, max_vectors=0)
        captured, second, cache = solve_pcg_cached(self.A, self.b, M=self.M,
                                                  x0=self.x0, rtol=1e-11,
                                                  maxiter=100, max_vectors=12)
        np.testing.assert_array_equal(plain, expected)
        np.testing.assert_array_equal(captured, plain)
        for key in original:
            if key != "seconds":
                self.assertEqual(first[key], original[key], key)
                self.assertEqual(second[key], original[key], key)
        self.assertEqual(empty["D"].shape, (23, 0))
        self.assertEqual(cache["D"].shape, (23, 12))
        self.assertEqual(cache["stats"]["captured_vectors"], 12)
        self.assertTrue(second["converged"])

    def test_all_cached_actions_match_direct_actions_without_extra_operator_calls(self):
        def run(capacity):
            calls = {"A": 0, "M": 0}

            def a(v):
                calls["A"] += 1
                return self.A @ v

            def m(v):
                calls["M"] += 1
                return self.M @ v

            A = sla.LinearOperator(self.A.shape, matvec=a, dtype=float)
            M = sla.LinearOperator(self.M.shape, matvec=m, dtype=float)
            x, metrics, cache = solve_pcg_cached(A, self.b, M=M, x0=self.x0,
                                                rtol=1e-11, maxiter=100,
                                                max_vectors=capacity)
            self.assertEqual(calls["M"], metrics["iterations"])
            self.assertEqual(cache["stats"]["m_applications"], calls["M"])
            self.assertEqual(cache["stats"]["a_applications"], calls["A"])
            return x, metrics, cache, calls

        _, _, _, base_calls = run(0)
        _, metrics, cache, calls = run(32)
        self.assertEqual(calls, base_calls)
        np.testing.assert_allclose(cache["AD"], self.A @ cache["D"],
                                   rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(cache["MAD"], self.M @ cache["AD"],
                                   rtol=2e-11, atol=2e-14)
        self.assertEqual(cache["stats"]["skipped_terminal_vectors"], 1)
        self.assertEqual(cache["D"].shape[1], metrics["iterations"] - 1)
        for q in (1, 2, 4):
            cached_slow_ritz(cache["D"], cache["AD"], cache["MAD"], q=q)
        self.assertEqual(calls, base_calls)

    def test_terminal_step_is_skipped_and_identity_M_alias_is_safe(self):
        A = sp.diags([1., 2., 4., 7.], format="csr")
        b = np.array([1., 2., -1., 4.])
        x, metrics, cache = solve_pcg_cached(A, b, rtol=1e-12, max_vectors=20)
        self.assertTrue(metrics["converged"])
        self.assertEqual(cache["D"].shape[1], metrics["iterations"] - 1)
        np.testing.assert_allclose(cache["MAD"], cache["AD"], atol=1e-14)
        _, one, single = solve_pcg_cached(sp.eye(4), b, max_vectors=20)
        self.assertEqual(one["iterations"], 1)
        self.assertEqual(single["D"].shape, (4, 0))
        self.assertEqual(single["stats"]["m_applications"], 1)
        self.assertEqual(single["stats"]["skipped_terminal_vectors"], 1)

    def test_dense_noncommuting_spd_preconditioner(self):
        rng = np.random.default_rng(5103)
        factor = rng.normal(size=(10, 10))
        A = factor.T @ factor + 2. * np.eye(10)
        factor = rng.normal(size=(10, 10))
        M = factor.T @ factor + np.eye(10)
        b = rng.normal(size=10)
        x, metrics, cache = solve_pcg_cached(A, b, M=M, rtol=1e-11,
                                             maxiter=80, max_vectors=8)
        self.assertTrue(metrics["converged"])
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(cache["MAD"], M @ cache["AD"], atol=1e-12)
        _, _, info = cached_slow_ritz(cache["D"], cache["AD"], cache["MAD"], q=2)
        _, _, explicit = extract_slow_ritz(A, M, cache["D"], q=2)
        np.testing.assert_allclose(info["ritz_values"], explicit["ritz_values"], rtol=1e-10)

    def test_budget_exhaustion_and_last_step_no_extra_M(self):
        _, metrics, cache = solve_pcg_cached(self.A, self.b, M=self.M,
                                             rtol=1e-14, maxiter=3, max_vectors=20)
        self.assertEqual(metrics["iterations"], 3)
        self.assertEqual(metrics["restarts"], 0)
        self.assertFalse(metrics["converged"])
        self.assertEqual(cache["D"].shape[1], 2)
        self.assertEqual(cache["stats"]["m_applications"], 3)
        self.assertEqual(cache["stats"]["skipped_terminal_vectors"], 1)

    def test_refresh_total_budget_and_at_most_three_segments_after_initial(self):
        calls = []

        def premature(A, b, M, x, rtol, maxiter, collector, counts, segment):
            calls.append((maxiter, rtol, segment))
            used = min(maxiter, 2)
            return x + used * .01, 0, used

        with patch("cached_pcg._pcg_segment", side_effect=premature):
            _, metrics, cache = solve_pcg_cached(sp.eye(3), np.ones(3),
                                                 maxiter=5, max_vectors=20)
        self.assertEqual([v[0] for v in calls], [5, 3, 1])
        np.testing.assert_allclose([v[1] for v in calls], [1e-8, 2e-9, 4e-10])
        self.assertEqual([v[2] for v in calls], [0, 1, 2])
        self.assertEqual(metrics["iterations"], 5)
        self.assertFalse(metrics["converged"])
        self.assertEqual(cache["stats"]["a_applications"], 3)
        calls.clear()
        with patch("cached_pcg._pcg_segment", side_effect=premature):
            _, metrics, _ = solve_pcg_cached(sp.eye(3), np.ones(3),
                                              maxiter=100, max_vectors=20)
        self.assertEqual(len(calls), 4)
        self.assertEqual(metrics["restarts"], 3)
        self.assertEqual(metrics["iterations"], 8)

    def test_refresh_segments_never_pair_their_terminal_z(self):
        # Force a controlled true/recursive residual discrepancy by replacing
        # only the wrapper's segment result, preserving each real recurrence.
        # Each one-iteration identity solve terminates with an unpaired z.
        from cached_pcg import _pcg_segment
        calls = []

        def displaced(A, b, M, x, rtol, maxiter, collector, counts, segment):
            out, info, used = _pcg_segment(A, b, M, x, rtol, maxiter,
                                           collector, counts, segment)
            calls.append(segment)
            return out - .1, info, used

        with patch("cached_pcg._pcg_segment", side_effect=displaced):
            _, metrics, cache = solve_pcg_cached(sp.eye(4), np.ones(4),
                                                 maxiter=20, max_vectors=20)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(metrics["iterations"], 4)
        self.assertEqual(cache["D"].shape[1], 0)
        self.assertEqual(cache["stats"]["skipped_terminal_vectors"], 4)
        self.assertEqual(cache["stats"]["segment_ids"], [])
        self.assertEqual(cache["stats"]["m_applications"], 4)

    def test_exact_initial_guess_zero_rhs_invalid_inputs_and_spd_breakdown(self):
        A = sp.diags([1., 2., 4.], format="csr")
        _, metrics, cache = solve_pcg_cached(A, np.array([1., 2., 4.]),
                                             x0=np.ones(3), max_vectors=5)
        self.assertEqual(metrics["iterations"], 0)
        self.assertEqual(cache["D"].shape, (3, 0))
        self.assertEqual(cache["stats"]["m_applications"], 0)
        x, metrics, _ = solve_pcg_cached(A, np.zeros(3), x0=np.ones(3))
        np.testing.assert_array_equal(x, np.zeros(3))
        self.assertTrue(metrics["converged"])
        for bad in (-1, .5, True):
            with self.assertRaises(ValueError):
                solve_pcg_cached(A, np.ones(3), max_vectors=bad)
        _, metrics, _ = solve_pcg_cached(-A, np.ones(3))
        self.assertEqual(metrics["info"], -2)
        self.assertFalse(metrics["converged"])


class CachedRitzTests(unittest.TestCase):
    def test_cached_pcg_span_matches_explicit_ritz_for_q1_q2_q4(self):
        rng = np.random.default_rng(5102)
        factor = rng.normal(size=(31, 31))
        A = factor.T @ factor + .5 * np.eye(31)
        M = np.diag(1. / A.diagonal())
        b = rng.normal(size=31)
        _, metrics, cache = solve_pcg_cached(A, b, M=M, max_vectors=24,
                                             rtol=1e-11, maxiter=100)
        self.assertTrue(metrics["converged"])
        for q in (1, 2, 4):
            U, AU, info = cached_slow_ritz(cache["D"], cache["AD"], cache["MAD"], q=q)
            expected, expected_AU, reference = extract_slow_ritz(A, M, cache["D"], q=q)
            self.assertEqual(info["status"], "accepted")
            self.assertEqual(info["rank"], q)
            self.assertEqual(info["a_applications"], 0)
            self.assertEqual(info["m_applications"], 0)
            np.testing.assert_allclose(info["ritz_values"], reference["ritz_values"],
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(AU, A @ U, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(U.T @ AU, np.eye(q), atol=1e-11)
            np.testing.assert_allclose(np.linalg.svd(U.T @ expected_AU, compute_uv=False),
                                       np.ones(q), atol=1e-10)
            self.assertLess(max(info["projected_ritz_residuals"]), 1e-11)
            np.testing.assert_allclose(info["operator_ritz_residuals"],
                                       reference["operator_ritz_residuals"],
                                       rtol=1e-8, atol=1e-11)

    def test_rank_scaling_full_span_order_and_invalid_columns(self):
        A = np.diag([1., 2., 4., 8., 16.])
        M = np.diag([.01, .025, .25, .25, .25])
        D = np.eye(5)
        D[:, 0] *= 1e240
        D[:, 1] *= 1e-240
        D = np.column_stack([D, D[:, 0], np.zeros(5), np.full(5, np.nan)])
        U, AU, info = cached_slow_ritz(D, A @ D, M @ (A @ D), q=5)
        self.assertEqual(info["rank"], 5)
        self.assertEqual(info["rejected_vectors"], 2)
        self.assertEqual(info["dependent_vectors"], 1)
        np.testing.assert_allclose(info["ritz_values"], [.01, .05, 1., 2., 4.], atol=1e-13)
        np.testing.assert_allclose(M @ AU, U * np.array(info["ritz_values"])[None, :], atol=1e-13)
        np.testing.assert_allclose(U.T @ AU, np.eye(5), atol=1e-13)

    def test_cache_asymmetry_and_indefiniteness_are_rejected_not_hidden(self):
        D = np.eye(3)
        bad_H = np.eye(3)
        bad_H[0, 1] = .1
        U, _, info = cached_slow_ritz(D, D, bad_H, q=2)
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(info["status"], "cache_symmetry_mismatch")
        self.assertGreater(info["h_relative_asymmetry"], .01)
        bad_G = np.eye(3)
        bad_G[0, 1] = .1
        _, _, info = cached_slow_ritz(D, bad_G, bad_G, q=2)
        self.assertEqual(info["status"], "cache_symmetry_mismatch")
        _, _, info = cached_slow_ritz(D, D, np.diag([-1., 1., 2.]), q=2)
        self.assertEqual(info["status"], "indefinite_ritz_matrix")

    def test_empty_tiny_positive_and_validation(self):
        blank = np.empty((3, 0))
        U, AU, info = cached_slow_ritz(blank, blank, blank, q=2)
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(info["status"], "empty_basis")
        D = np.eye(3)
        _, _, info = cached_slow_ritz(D, D, np.diag([1e-14, 1., 2.]), q=1)
        self.assertAlmostEqual(info["ritz_values"][0], 1e-14, delta=1e-28)
        with self.assertRaises(ValueError):
            cached_slow_ritz(D, D, D[:, :1])
        with self.assertRaises(ValueError):
            cached_slow_ritz(D, D, D, q=0)


if __name__ == "__main__":
    unittest.main()
