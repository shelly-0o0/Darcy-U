"""Small, dependency-light tests of the coefficient experiment protocol."""

import unittest

import numpy as np

try:
    from .fields import FAMILIES, coefficient_sequence, smooth_rhs
except ImportError:
    from fields import FAMILIES, coefficient_sequence, smooth_rhs


def component_count(mask):
    """Four-connected component count, without an optional SciPy dependency."""
    unseen = set(map(tuple, np.argwhere(mask)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            iy, ix = stack.pop()
            for point in ((iy-1, ix), (iy+1, ix), (iy, ix-1), (iy, ix+1)):
                if point in unseen:
                    unseen.remove(point)
                    stack.append(point)
    return count


class TestFields(unittest.TestCase):
    def test_reproducible_and_disjoint_seed_geometries(self):
        a, meta_a = coefficient_sequence(64, seed=2)
        b, meta_b = coefficient_sequence(64, seed=2)
        c, _ = coefficient_sequence(64, seed=10002)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(meta_a, meta_b)
        self.assertFalse(np.array_equal(a, c))

    def test_shape_range_and_nested_geometry(self):
        for family in FAMILIES:
            a, metadata = coefficient_sequence(32, steps=8, seed=7, family=family)
            b, _ = coefficient_sequence(64, steps=8, seed=7, family=family)
            self.assertEqual(a.shape, (8, 32, 32))
            self.assertEqual(len(metadata), 8)
            self.assertTrue(np.isfinite(a).all())
            self.assertEqual(float(a.min()), 1.0)
            self.assertEqual(float(a.max()), 1e4)
            np.testing.assert_array_equal(b, np.repeat(np.repeat(a, 2, axis=1), 2, axis=2))
            for i in range(1, 8):
                changed = int(np.count_nonzero(a[i] != a[i-1]))
                self.assertEqual(changed, metadata[i]["changed_cells"])
                self.assertLess(metadata[i]["change_fraction"], 0.05)

    def test_topology_event_really_connects_and_disconnects(self):
        for seed in range(10):
            a, metadata = coefficient_sequence(32, steps=20, seed=seed, family="topology")
            self.assertEqual(component_count(a[0] > 1), 2)
            self.assertEqual(component_count(a[10] > 1), 1)
            self.assertEqual(component_count(a[15] > 1), 2)
            self.assertEqual(metadata[10]["event"], "bridge_connected")
            self.assertEqual(metadata[15]["event"], "bridge_blocked")
            np.testing.assert_array_equal(a[0], a[9])
            self.assertGreater(metadata[10]["changed_cells"], 0)
            self.assertGreater(metadata[15]["changed_cells"], 0)

    def test_global_rng_is_not_modified(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        coefficient_sequence(32, seed=4)
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_rhs_and_validation(self):
        x = np.linspace(0, 1, 33)[None, :]
        y = np.linspace(0, 1, 33)[:, None]
        self.assertEqual(smooth_rhs(x, y).shape, (33, 33))
        self.assertAlmostEqual(smooth_rhs(0.5, 0.5, "sine").item(), 2*np.pi**2)
        self.assertAlmostEqual(float(smooth_rhs(x, y, "source").sum()), 0.0, places=10)
        for kwargs in ({"N": 31}, {"N": 32, "steps": 3}, {"N": 32, "contrast": 0},
                       {"N": 32, "family": "bad"}, {"N": 32, "seed": -1}):
            with self.assertRaises(ValueError):
                coefficient_sequence(**kwargs)


if __name__ == "__main__":
    unittest.main()
