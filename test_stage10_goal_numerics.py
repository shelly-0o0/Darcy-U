"""Small numerical identity tests; no stage-10 experiment or timing selection."""

import unittest

import numpy as np
import scipy.sparse.linalg as sla

from numerics import assemble_system
from stage10_goal_numerics import (
    assemble_load, assemble_nested, build_problem, element_indicators,
    free_to_full, p1_prolongation, solve_goal, solve_primal)


def gaussian(cx, cy, width=.12):
    return lambda x, y: np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))


class NestedGoalNumericsTests(unittest.TestCase):
    def test_fine_assembly_matches_existing_fe(self):
        rng = np.random.default_rng(701)
        k = np.exp(rng.normal(size=(8, 8)))
        f = lambda x, y: 1.+2*x-y+3*x*y
        case = assemble_nested(4, k, f)
        old_a, old_f, xy = assemble_system(k, f)
        np.testing.assert_allclose(case.Af.toarray(), old_a.toarray(), atol=1e-14)
        np.testing.assert_allclose(case.ff, old_f, atol=1e-16)
        np.testing.assert_array_equal(case.coordinates, xy)
        np.testing.assert_array_equal(case.full_to_free[case.free], np.arange(case.dimension))
        self.assertEqual(case.triangles.shape, (64, 2, 3))

    def test_affine_full_prolongation_and_boundary(self):
        case = assemble_nested(5, 1.)
        yy, xx = np.mgrid[:6, :6]
        affine = (2.+3*xx/5.-.7*yy/5.).ravel()
        exact = 2.+3*case.full_coordinates[:, 0]-.7*case.full_coordinates[:, 1]
        np.testing.assert_allclose(case.P_full @ affine, exact, atol=1e-14)
        np.testing.assert_allclose(np.asarray(case.P_full.sum(axis=1)).ravel(), 1.)
        u = np.arange(16, dtype=float)
        coarse_full = np.zeros(36)
        coarse_full[case.coarse_free] = u
        lifted = case.P_full @ coarse_full
        np.testing.assert_allclose(lifted, free_to_full(case, case.P @ u))

    def test_diagonal_p1_is_not_bilinear_and_restricts_quadratics(self):
        nc = 4
        case = assemble_nested(nc, 1.)
        yy, xx = np.mgrid[:nc+1, :nc+1]
        quadratic = ((xx/nc)**2+(yy/nc)**2+xx*yy/nc**2).ravel()
        fine = (case.P_full @ quadratic).reshape(case.nf+1, case.nf+1)
        np.testing.assert_allclose(fine[::2, ::2].ravel(), quadratic)
        xy = (xx*yy/nc**2).ravel()
        p1_mid = (case.P_full @ xy).reshape(case.nf+1, case.nf+1)[1, 1]
        self.assertAlmostEqual(p1_mid, .5/nc**2)
        self.assertNotAlmostEqual(p1_mid, .25/nc**2)

    def test_galerkin_matrix_and_restricted_loads(self):
        k = np.arange(64, dtype=float).reshape(8, 8)/20.+1.
        case = assemble_nested(4, k, gaussian(.31, .43))
        np.testing.assert_allclose(case.Ac.toarray(), (case.P.T @ case.Af @ case.P).toarray())
        np.testing.assert_allclose(case.fc, case.P.T @ case.ff)
        self.assertTrue(np.all(np.linalg.eigvalsh(case.Ac.toarray()) > 0))
        self.assertTrue(np.all(case.fine_diagonal > 0))

    def test_manufactured_sine_spatial_convergence(self):
        errors = []
        for nc in [4, 8, 16]:
            f = lambda x, y: 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
            case = assemble_nested(nc, 1., f)
            result = solve_primal(case)
            exact = np.sin(np.pi*case.coordinates[:, 0])*np.sin(np.pi*case.coordinates[:, 1])
            errors.append(np.linalg.norm(result['uf']-exact)/np.linalg.norm(exact))
            self.assertLess(result['fine_relative_residual'], 1e-12)
        orders = np.log2(np.asarray(errors[:-1])/errors[1:])
        self.assertTrue(np.all((orders > 1.8) & (orders < 2.2)), orders)

    def test_two_level_goal_identity_and_orthogonality(self):
        rng = np.random.default_rng(702)
        case = assemble_nested(5, np.exp(rng.normal(scale=.4, size=(10, 10))),
                               gaussian(.23, .41))
        primal = solve_primal(case)
        goal = solve_goal(case, gaussian(.74, .67), primal)
        self.assertLess(primal['coarse_relative_residual'], 1e-12)
        self.assertLess(goal['fine_relative_residual'], 1e-12)
        np.testing.assert_allclose(case.P.T @ primal['r_u'], 0., atol=1e-16)
        np.testing.assert_allclose(case.P.T @ case.Af @ goal['delta_z'], 0., atol=1e-15)
        np.testing.assert_allclose(case.Af @ goal['delta_z'], goal['r_z'], atol=1e-15)
        np.testing.assert_allclose(goal['dwr'], goal['exact_goal_error'], rtol=1e-11, atol=1e-17)
        # This is enrichment error, not the residual of the converged fine solve.
        self.assertGreater(np.linalg.norm(primal['r_u']), 1e-4)

    def test_sparse_element_scatter_and_arbitrary_vector_identity(self):
        rng = np.random.default_rng(703)
        case = assemble_nested(4, 2., gaussian(.6, .2))
        primal = solve_primal(case, fine_reference=False)
        residual_full = np.zeros((case.nf+1)**2)
        np.add.at(residual_full, case.triangles.ravel(), primal['cell_residual_nodes'].ravel())
        np.testing.assert_allclose(residual_full[case.free], primal['r_u'], atol=1e-16)
        weight = rng.normal(size=case.dimension)
        local = element_indicators(case, primal, weight)
        self.assertEqual(local['coarse_cell'].shape, (case.nc, case.nc))
        self.assertAlmostEqual(local['total'], primal['r_u'] @ weight, places=13)
        np.testing.assert_allclose(local['coarse_cell'],
            local['fine_cell'][::2, ::2]+local['fine_cell'][1::2, ::2]
            +local['fine_cell'][::2, 1::2]+local['fine_cell'][1::2, 1::2])

    def test_error_operator_reciprocity_psd(self):
        case = assemble_nested(5, 1.)
        sources = [gaussian(.2, .3), gaussian(.7, .6), gaussian(.3, .8),
                   lambda x, y: 1.+x-2*y]
        primals = [solve_primal(case, source) for source in sources]
        residuals = np.column_stack([p['r_u'] for p in primals])
        defects = np.column_stack([p['uf']-p['prolonged_uc'] for p in primals])
        matrix = residuals.T @ defects
        loads = np.column_stack([p['ff'] for p in primals])
        np.testing.assert_allclose(matrix, loads.T @ defects, rtol=1e-11, atol=1e-16)
        np.testing.assert_allclose(matrix, matrix.T, rtol=1e-11, atol=1e-16)
        self.assertGreater(np.min(np.linalg.eigvalsh(matrix)), 0.)
        self.assertGreater(np.min(np.diag(matrix)), 0.)
        # Directly test the full discrete error operator on a tiny problem.
        tiny = assemble_nested(2, 1.)
        E = np.linalg.inv(tiny.Af.toarray())-tiny.P @ np.linalg.inv(tiny.Ac.toarray()) @ tiny.P.T
        np.testing.assert_allclose(E, E.T, atol=1e-14)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(E)), -1e-14)
        np.testing.assert_allclose(E @ tiny.Af @ tiny.P, 0., atol=1e-14)

    def test_coarse_projection_global_invariance_and_local_gauge(self):
        rng = np.random.default_rng(704)
        case = assemble_nested(5, 1., gaussian(.38, .47))
        primal = solve_primal(case)
        weight = rng.normal(size=case.dimension)
        coarse = sla.spsolve(case.Ac, case.P.T @ case.Af @ weight)
        projected = weight-case.P @ coarse
        np.testing.assert_allclose(case.P.T @ case.Af @ projected, 0., atol=2e-14)
        first = element_indicators(case, primal, weight)
        second = element_indicators(case, primal, projected)
        self.assertAlmostEqual(first['total'], second['total'], places=13)
        self.assertGreater(np.linalg.norm(first['coarse_cell']-second['coarse_cell']), 1e-4)
        self.assertAlmostEqual(np.sum(first['coarse_cell']-second['coarse_cell']), 0., places=13)

    def test_global_invariance_requires_small_coarse_algebraic_residual(self):
        case = assemble_nested(4, 1.)
        # Deliberately inaccurate coarse primal, unlike solve_primal's output.
        bad_r = case.ff.copy()
        shift = case.P @ np.ones(case.Ac.shape[0])
        self.assertGreater(abs(bad_r @ shift), 1e-3)
        # Coarse-shift pairing is exactly the missing coarse residual term.
        self.assertAlmostEqual(float(bad_r @ shift),
            float((case.P.T @ bad_r) @ np.ones(case.Ac.shape[0])), places=14)

    def test_goal_linearity_and_source_independence(self):
        case = assemble_nested(4, 1.)
        g1, g2 = gaussian(.3, .4), gaussian(.7, .8)
        z1, z2 = solve_goal(case, g1), solve_goal(case, g2)
        combination = solve_goal(case, lambda x, y: 2*g1(x, y)-.7*g2(x, y))
        np.testing.assert_allclose(combination['delta_z'],
            2*z1['delta_z']-.7*z2['delta_z'], rtol=1e-11, atol=1e-16)
        primal = solve_primal(case, gaussian(.1, .2))
        z_again = solve_goal(case, g1, primal)
        np.testing.assert_array_equal(z1['delta_z'], z_again['delta_z'])

    def test_cache_factorization_charged_once_and_nonoverlap(self):
        case = assemble_nested(4, 1.)
        primal = solve_primal(case)
        goal = solve_goal(case, gaussian(.5, .5), primal)
        self.assertEqual(primal['cache_hits'], dict(coarse=False, fine=False))
        self.assertEqual(goal['cache_hits'], dict(coarse=True, fine=True))
        self.assertGreater(primal['timings']['coarse_factorization_seconds'], 0.)
        self.assertEqual(goal['timings']['coarse_factorization_seconds'], 0.)
        self.assertEqual(case.cache_statistics['coarse_factorizations'], 1)
        self.assertEqual(case.cache_statistics['fine_factorizations'], 1)
        self.assertEqual(case.cache_statistics['coarse_solves'], 2)
        self.assertEqual(case.cache_statistics['fine_solves'], 2)
        for result in [primal, goal]:
            timing = result['timings']
            components = sum(v for k, v in timing.items() if k not in ['total_seconds', 'overhead_seconds'])
            self.assertGreaterEqual(timing['total_seconds'], components)
            self.assertAlmostEqual(timing['total_seconds'], components+timing['overhead_seconds'], places=14)

    def test_online_mode_does_not_create_fine_factorization(self):
        case = assemble_nested(4, 1.)
        primal = solve_primal(case, fine_reference=False)
        goal = solve_goal(case, gaussian(.2, .8), primal, fine_reference=False)
        self.assertIsNone(primal['uf'])
        self.assertIsNone(goal['zf'])
        self.assertIsNone(goal['delta_z'])
        self.assertEqual(case.cache_statistics['fine_factorizations'], 0)
        self.assertNotIn('fine', case.cache)
        self.assertEqual(primal['timings']['fine_solve_seconds'], 0.)

    def test_zero_loads_and_input_preservation(self):
        k = np.ones((6, 6))
        case = assemble_nested(3, k, 0.)
        k[:] = 5.
        self.assertTrue(np.all(case.k_fine == 1.))
        primal = solve_primal(case)
        goal = solve_goal(case, 0., primal)
        self.assertEqual(primal['fine_relative_residual'], 0.)
        self.assertEqual(goal['dwr'], 0.)
        np.testing.assert_array_equal(goal['delta_z'], 0.)

    def test_invalid_inputs_and_convenience_bundle(self):
        for nc, k in [(1, 1.), (2.5, 1.), (3, -1.), (3, np.ones((3, 3))), (3, np.nan)]:
            with self.assertRaises(ValueError):
                assemble_nested(nc, k)
        case = assemble_nested(3, 1.)
        with self.assertRaises(ValueError):
            assemble_load(case, lambda x, y: np.nan*x)
        with self.assertRaises(ValueError):
            free_to_full(case, np.ones(3))
        bundle = build_problem(3, 1., gaussian(.2, .4), gaussian(.6, .8))
        self.assertAlmostEqual(bundle['goal']['dwr'], bundle['goal']['exact_goal_error'], places=14)


if __name__ == '__main__':
    unittest.main()
