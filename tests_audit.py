"""Independent numerical checks for the learned subspace objective."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from learning import coverage_loss, tensor_sample
from numerics import b_orthonormalize


class LearningObjectiveAudit(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(451)
        q = rng.normal(size=(12, 12))
        self.B = q.T @ q + np.eye(12)
        self.Z = b_orthonormalize(rng.normal(size=(12, 2)), self.B)
        self.T = b_orthonormalize(rng.normal(size=(12, 4)), self.B)
        self.Y = rng.normal(size=(12, 2))
        self.iy, self.ix = np.indices((4, 3)).reshape(2, -1)
        self.rng = rng

    def sample(self, Z=None, T=None):
        return tensor_sample({
            "B": sp.csr_matrix(self.B), "Z": self.Z if Z is None else Z,
            "target": self.T if T is None else T, "iy": self.iy,
            "ix": self.ix, "invdiag": np.ones(12),
        }, "cpu")

    def output(self, Y=None, grad=False):
        y = self.Y if Y is None else Y
        return torch.tensor(y.T.reshape(2, 4, 3), dtype=torch.float64,
                            requires_grad=grad)

    def test_loss_agrees_with_independent_best_projection(self):
        union = b_orthonormalize(np.column_stack([self.Z, self.Y]), self.B)
        remainder = self.T - union @ (union.T @ self.B @ self.T)
        exact = float(np.sum(remainder * (self.B @ remainder))) / self.T.shape[1]
        actual = float(coverage_loss(self.output(), self.sample()))
        # A 1e-6 ridge in the training objective slightly increases the loss.
        self.assertGreaterEqual(actual, exact - 1e-12)
        self.assertLess(abs(actual - exact), 1e-5)

    def test_old_and_target_basis_rotations_leave_loss_unchanged(self):
        qz, _ = np.linalg.qr(self.rng.normal(size=(2, 2)))
        qt, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        a = float(coverage_loss(self.output(), self.sample()))
        b = float(coverage_loss(self.output(), self.sample(self.Z @ qz, self.T @ qt)))
        self.assertAlmostEqual(a, b, places=11)

    def test_existing_space_additions_do_not_change_proposal(self):
        y = self.Y + self.Z @ self.rng.normal(size=(2, 2))
        a = float(coverage_loss(self.output(), self.sample()))
        b = float(coverage_loss(self.output(y), self.sample()))
        self.assertAlmostEqual(a, b, places=11)

    def test_gradient_is_finite_and_changes_loss(self):
        output = self.output(grad=True)
        loss = coverage_loss(output, self.sample())
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(output.grad)), 1e-8)
        updated = output.detach() - 1e-3 * output.grad
        self.assertLess(float(coverage_loss(updated, self.sample())), float(loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
