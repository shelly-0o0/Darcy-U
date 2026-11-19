"""Small checks for global correction physics, feature conventions, and network."""

import unittest

import numpy as np
import torch

from correction_model import (CorrectionUNet, make_correction_features, p1_apply,
                              p1_diagonal, p1_energy, relative_energy_loss,
                              weighted_residual_loss)
from numerics import assemble_system, free_to_grid


class CorrectionModelTests(unittest.TestCase):
    def test_energy_action_and_diagonal_match_assembled_matrix(self):
        rng = np.random.default_rng(18)
        n = 8
        k = 10 ** rng.uniform(0., 4., (n, n))
        matrix, _, _ = assemble_system(k)
        vector = rng.normal(size=(n - 1) ** 2)
        grid = torch.tensor(free_to_grid(vector, n), dtype=torch.float64, requires_grad=True)
        coefficient = torch.tensor(k, dtype=torch.float64)
        energy = p1_energy(grid, coefficient)
        np.testing.assert_allclose(energy.detach().numpy(), [vector @ (matrix @ vector)], rtol=2e-14)
        action = p1_apply(grid, coefficient).detach().numpy()
        np.testing.assert_allclose(action[1:-1, 1:-1].ravel(), matrix @ vector, atol=2e-11, rtol=2e-13)
        diagonal = p1_diagonal(coefficient).numpy()[1:-1, 1:-1].ravel()
        np.testing.assert_allclose(diagonal, matrix.diagonal(), rtol=2e-14)
        energy.sum().backward()
        np.testing.assert_allclose(grid.grad.numpy()[1:-1, 1:-1].ravel(), 2 * (matrix @ vector), atol=4e-11, rtol=2e-13)

    def test_batched_losses_and_cached_diagonal(self):
        torch.manual_seed(19)
        k = 1. + 20. * torch.rand(2, 6, 6, dtype=torch.float64)
        target = torch.nn.functional.pad(torch.randn(2, 1, 5, 5, dtype=torch.float64), (1, 1, 1, 1))
        zero = torch.zeros_like(target, requires_grad=True)
        np.testing.assert_allclose(relative_energy_loss(zero, target, k, reduction="none").detach().numpy(), 1.)
        np.testing.assert_allclose(weighted_residual_loss(zero, target, k, reduction="none").detach().numpy(), 1.)
        self.assertEqual(relative_energy_loss(target, target, k).item(), 0.)
        cached = weighted_residual_loss(.3 * target, target, k, diagonal=p1_diagonal(k))
        self.assertAlmostEqual(cached.item(), .49, places=12)
        loss = relative_energy_loss(zero, target, k) + .1 * weighted_residual_loss(zero, target, k)
        loss.backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        self.assertGreater(zero.grad.norm().item(), 0.)

    def test_feature_order_scales_and_available_input_only(self):
        k = np.full((4, 4), 1e4)
        previous = np.full((4, 4), 1e2)
        residual = np.full(9, 3.)
        base = np.full(9, 2.)
        features, metadata = make_correction_features(k, previous, residual, base, solution_scale=2.)
        self.assertEqual(features.shape, (6, 5, 5))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features[0], 1.)
        np.testing.assert_allclose(features[1], .5)
        np.testing.assert_allclose(features[2:4, 1:-1, 1:-1], 1.)
        np.testing.assert_allclose(features[4, 0], np.linspace(0, 1, 5))
        np.testing.assert_allclose(features[5, :, 0], np.linspace(0, 1, 5))
        self.assertEqual(metadata["residual_scale"], 3.)
        self.assertFalse(metadata["label_used_for_normalization"])
        with self.assertRaises(ValueError):
            make_correction_features(k, previous, residual, base, solution_scale=0.)

    def test_model_odd_grid_boundary_and_backward(self):
        # Keep this a shape/gradient test, not a training or timing benchmark.
        torch.manual_seed(20)
        torch.set_num_threads(1)
        model = CorrectionUNet(width=8, zero_init_output=False)
        inputs = torch.randn(1, 6, 17, 17)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (1, 1, 17, 17))
        self.assertTrue(torch.equal(output[..., 0, :], torch.zeros_like(output[..., 0, :])))
        self.assertTrue(torch.equal(output[..., -1, :], torch.zeros_like(output[..., -1, :])))
        self.assertTrue(torch.equal(output[..., :, 0], torch.zeros_like(output[..., :, 0])))
        self.assertTrue(torch.equal(output[..., :, -1], torch.zeros_like(output[..., :, -1])))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        initialized = CorrectionUNet(width=8)
        self.assertEqual(torch.count_nonzero(initialized(inputs)).item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
