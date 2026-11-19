"""Current-A projection and fixed/rebuilt-Galerkin AMG correctness tests."""

import unittest

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from numerics import assemble_system, solve_pcg
from sequence_baselines import (galerkin_initial_guess, project_previous_solution,
                                damped_initial_correction, jacobi_initial_correction,
                                FixedAMGHierarchy, AMGSequencePreconditioner)


class HistoryProjectionTests(unittest.TestCase):
    def test_constant_coefficient_scaling_is_exact(self):
        A, b, _ = assemble_system(np.ones((16, 16)))
        previous = sla.spsolve(A, b)
        current_A = 37. * A
        x, stats = project_previous_solution(current_A, b, previous)
        np.testing.assert_allclose(x, previous / 37., rtol=1e-12, atol=1e-14)
        self.assertEqual(stats["rank"], 1)
        self.assertLess(np.linalg.norm(b - current_A @ x) / np.linalg.norm(b), 1e-12)

    def test_rank_deficient_history_galerkin_orthogonality_and_energy(self):
        rng = np.random.default_rng(31)
        k = np.ones((16, 16))
        k[5:8, :] = 1e4
        A, b, _ = assemble_system(k)
        u, v = rng.normal(size=(2, len(b)))
        vectors = [u, 3. * u, v, v - u, np.zeros_like(u)]
        x, stats = galerkin_initial_guess(A, b, vectors)
        self.assertEqual(stats["rank"], 2)
        self.assertEqual(stats["rejected_vectors"], 1)
        np.testing.assert_allclose(np.array([u, v]) @ (b - A @ x), 0., atol=1e-12)
        energy = lambda z: .5 * z @ (A @ z) - b @ z
        for perturbation in (u, v, u - v, np.zeros_like(u)):
            self.assertLessEqual(energy(x), energy(x + .01 * perturbation) + 1e-14)
        # Scaling and rotating the same exact two-dimensional span changes no result.
        scaled, _ = galerkin_initial_guess(A, b, [1e250 * u, 1e-250 * v])
        np.testing.assert_allclose(scaled, x, rtol=1e-10, atol=1e-13)

    def test_empty_invalid_and_bounded_history(self):
        A = sp.eye(12, format="csr")
        b = np.ones(12)
        x, stats = galerkin_initial_guess(A, b, [])
        np.testing.assert_array_equal(x, np.zeros(12))
        self.assertEqual(stats["rank"], 0)
        history = list(np.eye(12))
        x, stats = galerkin_initial_guess(A, b, history, max_vectors=4)
        np.testing.assert_allclose(x, np.r_[np.zeros(8), np.ones(4)])
        self.assertEqual(stats["selected_vectors"], 4)
        x, stats = galerkin_initial_guess(A, b, [b, b * np.nan, b * np.inf])
        np.testing.assert_allclose(x, b)
        self.assertEqual(stats["rejected_vectors"], 2)
        with self.assertRaises(ValueError):
            galerkin_initial_guess(A, b, [b], max_vectors=9)

    def test_damped_line_optimum_and_finite_safeguards(self):
        rng = np.random.default_rng(32)
        A, b, _ = assemble_system(np.ones((16, 16)))
        x0, d = rng.normal(size=(2, len(b)))
        x, stats = damped_initial_correction(A, b, x0, d)
        expected = d @ (b - A @ x0) / (d @ (A @ d))
        self.assertAlmostEqual(stats["alpha"], expected, places=13)
        self.assertLessEqual(stats["energy_change"], 0.)
        self.assertAlmostEqual(float(d @ (b - A @ x)), 0., places=10)
        for bad in (np.zeros_like(d), d * np.nan, d * np.inf):
            safe, info = damped_initial_correction(A, b, x0, bad)
            np.testing.assert_array_equal(safe, x0)
            self.assertEqual(info["alpha"], 0.)
        safe, info = damped_initial_correction(A, b, x0 * np.nan, d)
        self.assertTrue(np.all(np.isfinite(safe)))
        self.assertTrue(info["initial_guess_reset"])

    def test_fixed_jacobi_matches_explicit_sweeps(self):
        A, b, _ = assemble_system(np.ones((16, 16)))
        x, stats = jacobi_initial_correction(A, b, None)
        expected = np.zeros_like(b)
        for _ in range(3):
            expected += (2. / 3.) * (b - A @ expected) / A.diagonal()
        np.testing.assert_allclose(x, expected, rtol=0., atol=0.)
        self.assertEqual(stats["sweeps"], 3)
        self.assertLess(.5 * x @ (A @ x) - b @ x, 0.)


class SequenceAMGTests(unittest.TestCase):
    def setUp(self):
        k = np.ones((24, 24))
        k[7:10, :] = 1e4
        self.A, self.b, _ = assemble_system(k)
        changed = k.copy()
        changed[7:10, 11:14] = 1.
        self.changed_A, _, _ = assemble_system(changed)
        changed[4:15, 17:19] = 1e4
        self.third_A, _, _ = assemble_system(changed)

    def assert_fixed_spd(self, M):
        rng = np.random.default_rng(33)
        x, y = rng.normal(size=(2, len(self.b)))
        mx, my = M @ x, M @ y
        self.assertGreater(float(x @ mx), 0.)
        np.testing.assert_allclose(x @ my, y @ mx, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(M @ (2. * x - .3 * y), 2. * mx - .3 * my,
                                   rtol=1e-10, atol=1e-10)

    def test_fresh_and_reuse_policies_are_fixed_and_solve(self):
        for mode in ("fresh", "reuse_first"):
            sequence = AMGSequencePreconditioner(mode, seed=34)
            first, initial = sequence.for_matrix(self.A)
            before = first @ self.b
            second, stats = sequence.for_matrix(self.changed_A)
            self.assertEqual(stats["hierarchy_rebuilt"], mode == "fresh")
            self.assertEqual(second is first, mode == "reuse_first")
            np.testing.assert_array_equal(first @ self.b, before)
            self.assert_fixed_spd(second)
            _, info = solve_pcg(self.changed_A, self.b, second, rtol=1e-8)
            self.assertTrue(info["converged"])

    def test_refresh_rebuilds_every_RAP_and_clears_old_solver_caches(self):
        sequence = AMGSequencePreconditioner("refresh_galerkin", seed=35)
        first, _ = sequence.for_matrix(self.A)
        first_result = first @ self.b  # Force lazy coarse-solver cache creation.
        template = sequence.fixed
        last_operator, last_result = first, first_result
        for current_A in (self.changed_A, self.third_A):
            current, stats = sequence.for_matrix(current_A)
            self.assertTrue(stats["galerkin_refreshed"])
            self.assertFalse(stats["interpolation_rebuilt"])
            levels = sequence.fixed.hierarchy.levels
            np.testing.assert_array_equal(levels[0].A.toarray(), current_A.toarray())
            for i, level in enumerate(levels[:-1]):
                self.assertIs(level.P, template.hierarchy.levels[i].P)
                np.testing.assert_allclose(levels[i + 1].A.toarray(),
                                           (level.R @ (level.A @ level.P)).toarray(),
                                           rtol=1e-13, atol=1e-13)
            self.assert_fixed_spd(current)
            independent = FixedAMGHierarchy.from_interpolation(current_A, template)
            np.testing.assert_allclose(current @ self.b,
                                       independent.preconditioner @ self.b,
                                       rtol=1e-13, atol=1e-13)
            np.testing.assert_array_equal(last_operator @ self.b, last_result)
            last_operator, last_result = current, current @ self.b
            _, info = solve_pcg(current_A, self.b, current, rtol=1e-8)
            self.assertTrue(info["converged"])
        np.testing.assert_array_equal(first @ self.b, first_result)

    def test_builder_does_not_mutate_input_or_numpy_random_state(self):
        A = self.A.copy()
        before = A.copy()
        np.random.seed(36)
        state = np.random.get_state()
        fixed = FixedAMGHierarchy(A, seed=999)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        result = fixed.preconditioner @ self.b
        A.data *= 2.
        np.testing.assert_array_equal(fixed.preconditioner @ self.b, result)
        np.testing.assert_array_equal(before.toarray(), self.A.toarray())


if __name__ == "__main__":
    unittest.main()
