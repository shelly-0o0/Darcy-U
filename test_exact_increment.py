"""Scientific checks for exact spectral reference and rank diagnostics."""

from types import SimpleNamespace
import unittest

import numpy as np
import scipy.linalg as la

from exact_increment import diagnose_space, spectral_increment
from numerics import (assemble_system, build_patches, b_orthonormalize,
                      certify_space, patch_spectral_basis)


class ExactIncrementTests(unittest.TestCase):
    def test_rotation_invariance_and_old_space_retention(self):
        rng = np.random.default_rng(83)
        q, _ = la.qr(rng.normal(size=(18, 18)))
        b = np.diag(np.geomspace(1, 50, 18))
        bhalf = np.diag(np.sqrt(np.diag(b)))
        a = bhalf @ (q * np.linspace(0, .3, 18)[None, :]) @ q.T @ bhalf
        p = SimpleNamespace(A_N=a, B=b)
        old = b_orthonormalize(rng.normal(size=(18, 3)), b)
        rotation, _ = la.qr(rng.normal(size=(3, 3)))
        z, info = spectral_increment(p, old, gamma=.08, budget=4)
        zz, _ = spectral_increment(p, old @ rotation, gamma=.08, budget=4)
        np.testing.assert_allclose(z @ z.T @ b, zz @ zz.T @ b, atol=2e-11)
        np.testing.assert_allclose(z @ (z.T @ b @ old), old, atol=2e-11)
        np.testing.assert_allclose(z.T @ b @ z, np.eye(z.shape[1]), atol=2e-11)
        self.assertEqual(info["added_rank"], 4)
        self.assertFalse(info["optimal_rank_q_claim"])

    def test_budget_four_cannot_repair_six_negative_directions(self):
        a = np.diag([0., .001, .002, .003, .004, .005, .1, .2])
        b = np.eye(8)
        old = np.empty((8, 0))
        p = SimpleNamespace(A_N=a, B=b)
        before = diagnose_space(a, b, old)
        self.assertEqual(before["significant_negative_count"], 6)
        z, info = spectral_increment(p, old, budget=4)
        self.assertEqual(info["added_rank"], 4)
        self.assertEqual(diagnose_space(a, b, z)["significant_negative_count"], 2)
        all_z, all_info = spectral_increment(p, old, budget=None)
        self.assertEqual(all_info["added_rank"], 6)
        self.assertTrue(certify_space(a, b, all_z)["passed"])
        # In particular the exactly zero mode was retained first.
        np.testing.assert_allclose(z @ z.T @ np.eye(8)[:, 0], np.eye(8)[:, 0])

    def test_all_reference_passes_on_changed_high_contrast_patches(self):
        k = np.ones((48, 48))
        k[10:14, :] = 1e4
        k[:, 28:34] = 1e4
        a, _, _ = assemble_system(k)
        previous = build_patches(k, a, core_cells=16, overlap=2)
        changed = k.copy()
        changed[12:24, 18:28] = 1e4
        aa, _, _ = assemble_system(changed)
        current = build_patches(changed, aa, core_cells=16, overlap=2)
        total_added = 0
        for p0, p in zip(previous, current):
            old = patch_spectral_basis(p0)
            z, info = spectral_increment(p, old, budget=None)
            self.assertTrue(certify_space(p.A_N, p.B, z)["passed"])
            self.assertTrue(diagnose_space(p.A_N, p.B, z)["passed"])
            total_added += info["added_rank"]
        self.assertGreater(total_added, 0)

    def test_full_matrix_diagnostic_and_count_are_not_sufficient(self):
        a = np.array([[1., .9], [.9, 1.]])
        b = np.eye(2)
        old = np.array([[1.], [0.]])
        result = diagnose_space(a, b, old, gamma=.5)
        self.assertEqual(result["negative_count"], 1)
        self.assertFalse(result["passed"])
        self.assertGreater(a[1, 1], .5)
        # An independent rank-one proposal can still be unhelpful: the count
        # lower bound alone is never a sufficient candidate acceptance test.
        aa = np.diag([.01, .2, .3])
        bb = np.eye(3)
        self.assertEqual(diagnose_space(aa, bb, np.empty((3, 0)))["negative_count"], 1)
        self.assertFalse(diagnose_space(aa, bb, np.eye(3)[:, 1:2])["passed"])
        self.assertTrue(diagnose_space(aa, bb, np.eye(3)[:, 0:1])["passed"])

    def test_zero_budget_and_no_missing_modes(self):
        a = np.diag([0., .01, .2])
        b = np.eye(3)
        p = SimpleNamespace(A_N=a, B=b)
        old = np.eye(3)[:, :2]
        z, info = spectral_increment(p, old, budget=0)
        self.assertEqual(info["added_rank"], 0)
        self.assertEqual(z.shape[1], 2)
        _, complete = spectral_increment(p, old, budget=None)
        self.assertEqual(complete["added_rank"], 0)
        self.assertTrue(diagnose_space(a, b, z)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
