"""Correctness and total-budget tests for optional completed-iterate capture."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from numerics import solve_pcg
from traced_pcg import solve_pcg_traced


class TracedPCGTests(unittest.TestCase):
    def test_disabled_observations_match_existing_solver(self):
        rng = np.random.default_rng(410)
        B = rng.normal(size=(37, 37))
        A = sp.csr_matrix(B.T @ B + 2. * np.eye(37))
        b, x0 = rng.normal(size=(2, 37))
        M = sla.LinearOperator(A.shape, matvec=lambda r: r / A.diagonal())
        expected, old = solve_pcg(A, b, M=M, x0=x0, rtol=1e-10, maxiter=80)
        actual, metrics, basis, trace = solve_pcg_traced(
            A, b, M=M, x0=x0, rtol=1e-10, maxiter=80)
        np.testing.assert_array_equal(actual, expected)
        for key in old:
            if key != "seconds":
                self.assertEqual(metrics[key], old[key], key)
        self.assertEqual(basis, [])
        self.assertEqual(trace, [])
        self.assertEqual(metrics["captured_vectors"], 0)

    def test_diagonal_increments_span_completed_correction_and_cap(self):
        A = sp.diags([1., 2., 3., 5., 8., 13.], format="csr")
        b = np.arange(1., 7.)
        x0 = np.linspace(-.2, .3, 6)
        x, metrics, basis, trace = solve_pcg_traced(
            A, b, x0=x0, rtol=1e-12, maxiter=20, max_vectors=32)
        self.assertTrue(metrics["converged"])
        self.assertEqual(len(basis), metrics["iterations"])
        np.testing.assert_allclose(np.sum(basis, axis=0), x - x0,
                                   rtol=1e-14, atol=1e-14)
        Z = np.column_stack(basis)
        gram = Z.T @ (A @ Z)
        scale = np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
        np.testing.assert_allclose(gram / scale, np.eye(len(basis)), atol=1e-10)
        capped, capped_metrics, first, _ = solve_pcg_traced(
            A, b, x0=x0, rtol=1e-12, maxiter=20, max_vectors=3)
        np.testing.assert_array_equal(capped, x)
        self.assertEqual(len(first), 3)
        for a, c in zip(first, basis):
            np.testing.assert_array_equal(a, c)
        self.assertEqual(capped_metrics["iterations"], metrics["iterations"])
        self.assertEqual(trace, [])

    def test_offline_trace_records_true_residual_and_preconditioned_norm(self):
        A = sp.diags([1., 3., 5., 7.], format="csr")
        M = sp.diags([1., .5, .25, .125], format="csr")
        b = np.array([1., -2., 3., 4.])
        x, metrics, basis, trace = solve_pcg_traced(
            A, b, M=M, rtol=1e-12, max_vectors=20, record_residuals=True)
        current = np.zeros_like(b)
        for i, (increment, point) in enumerate(zip(basis, trace), start=1):
            current += increment
            r = b - A @ current
            self.assertEqual(point["iteration"], i)
            self.assertEqual(point["restart"], 0)
            self.assertAlmostEqual(point["residual_norm"], np.linalg.norm(r))
            self.assertAlmostEqual(point["relative_residual"],
                                   np.linalg.norm(r) / np.linalg.norm(b))
            self.assertAlmostEqual(point["preconditioned_residual_squared"],
                                   float(r @ (M @ r)))
        self.assertEqual(len(trace), metrics["iterations"])
        np.testing.assert_allclose(current, x, atol=1e-15)

    def test_restart_budget_and_increment_origin(self):
        # Simulate recursive/true residual disagreement that real CG is
        # allowed to exhibit, without relying on platform-specific roundoff.
        A = sp.eye(3, format="csr")
        b = np.ones(3)
        calls = []

        def premature_cg(A, b, M, x0, rtol, atol, maxiter, callback):
            current = np.zeros_like(b) if x0 is None else x0.copy()
            calls.append((maxiter, rtol, current.copy()))
            current += .1
            callback(current)
            current += .1
            if maxiter > 1:
                callback(current)
            else:
                current -= .1
            return current, 0

        with patch("traced_pcg.sla.cg", side_effect=premature_cg):
            x, metrics, basis, trace = solve_pcg_traced(
                A, b, maxiter=5, max_vectors=32, record_residuals=True)
        self.assertEqual([c[0] for c in calls], [5, 3, 1])
        np.testing.assert_allclose([c[1] for c in calls], [1e-8, 2e-9, 4e-10])
        self.assertEqual(metrics["iterations"], 5)
        self.assertEqual(metrics["restarts"], 2)
        self.assertFalse(metrics["converged"])
        self.assertEqual([t["restart"] for t in trace], [0, 0, 1, 1, 2])
        np.testing.assert_allclose(np.asarray(basis), .1)
        np.testing.assert_allclose(np.sum(basis, axis=0), x)

    def test_at_most_three_refreshes_and_no_false_convergence(self):
        A = sp.eye(2, format="csr")
        b = np.ones(2)

        def premature_cg(A, b, M, x0, rtol, atol, maxiter, callback):
            current = np.zeros_like(b) if x0 is None else x0.copy()
            current += .01
            callback(current)
            return current, 0

        with patch("traced_pcg.sla.cg", side_effect=premature_cg) as mocked:
            _, metrics, basis, _ = solve_pcg_traced(
                A, b, maxiter=20, max_vectors=32)
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(metrics["restarts"], 3)
        self.assertEqual(metrics["iterations"], 4)
        self.assertEqual(len(basis), 4)
        self.assertFalse(metrics["converged"])

    def test_no_callback_exact_initial_guess_and_invalid_capacity(self):
        A = sp.diags([1., 2., 4.], format="csr")
        b = np.array([1., 2., 4.])
        x, metrics, basis, trace = solve_pcg_traced(
            A, b, x0=np.ones(3), max_vectors=32, record_residuals=True)
        np.testing.assert_array_equal(x, np.ones(3))
        self.assertEqual(metrics["iterations"], 0)
        self.assertTrue(metrics["converged"])
        self.assertEqual(basis, [])
        self.assertEqual(trace, [])
        for invalid in (-1, .5):
            with self.assertRaises(ValueError):
                solve_pcg_traced(A, b, max_vectors=invalid)


if __name__ == "__main__":
    unittest.main()
