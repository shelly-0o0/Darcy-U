"""Small correctness checks for the stage-9 numerical core; no benchmarks."""
import unittest
from unittest.mock import patch

import numpy as np

import nonlinear_budget as core


class DiscretizationTests(unittest.TestCase):
    def test_upwind_stencil_boundary_and_no_row_wrap(self):
        problem = core.make_problem(3, dict(eps=.2, gamma=0., beta=(1., .5)))
        A = problem.linear_operator.toarray()
        h, diffusion = problem.h, .2/problem.h**2
        self.assertEqual(problem.dimension, 9)
        self.assertAlmostEqual(A[4, 4], 4*diffusion+1.5/h+1.)
        self.assertAlmostEqual(A[4, 3], -diffusion-1./h)
        self.assertAlmostEqual(A[4, 5], -diffusion)
        self.assertAlmostEqual(A[4, 1], -diffusion-.5/h)
        self.assertAlmostEqual(A[4, 7], -diffusion)
        self.assertEqual(A[2, 3], 0.)
        self.assertEqual(A[3, 2], 0.)
        self.assertEqual(A[5, 6], 0.)
        self.assertAlmostEqual((A@np.ones(9))[4], 1.)
        self.assertAlmostEqual((A@np.ones(9))[0], 1.+2*diffusion+1.5/h)
        off_diagonal = A-np.diag(np.diag(A))
        self.assertTrue(np.all(off_diagonal <= 0.))
        negative = core.make_problem(3, dict(eps=.2, beta=(-1., .5))).linear_operator
        self.assertAlmostEqual(negative[4, 3], -diffusion)
        self.assertAlmostEqual(negative[4, 5], -diffusion-1./h)

    def test_gaussian_rhs_and_analytic_manufactured_rhs(self):
        p = dict(eps=.13, gamma=2., beta=(1., .5), amplitude=3., cx=.3, cy=.6)
        regular = core.make_problem(4, p)
        x, y = regular.coordinates
        expected = 3*np.exp(-((x-.3)**2+(y-.6)**2)/(2*.15**2)) + .3*np.sin(np.pi*x)*np.sin(np.pi*y)
        np.testing.assert_allclose(regular.rhs, expected.ravel(), rtol=1e-15)
        manufactured = core.make_problem(4, p, manufactured=True)
        u = 3*np.sin(np.pi*x)*np.sin(np.pi*y)
        f = (2*.13*np.pi**2+1)*u + 3*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        f += 1.5*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y) + 2*u**3
        np.testing.assert_allclose(manufactured.rhs, f.ravel(), rtol=1e-14, atol=1e-14)
        self.assertGreater(np.linalg.norm(manufactured.F(manufactured.exact)), .1)

    def test_jacobian_matches_directional_finite_difference(self):
        problem = core.make_problem(5, dict(eps=.03, gamma=4., beta=(1., .5)))
        rng = np.random.default_rng(912)
        u, direction = rng.normal(size=25), rng.normal(size=25)
        delta = 1e-6
        finite_difference = (problem.F(u+delta*direction)-problem.F(u-delta*direction))/(2*delta)
        np.testing.assert_allclose(problem.jacobian(u)@direction, finite_difference,
                                   rtol=2e-8, atol=2e-8)

    def test_manufactured_spatial_orders(self):
        for beta, expected_order in [((1., .5), 1.), ((0., 0.), 2.)]:
            errors, widths = [], []
            for n in (7, 15, 31):
                problem = core.make_problem(n, dict(eps=.2, gamma=1., beta=beta), manufactured=True)
                result = core.solve(problem, 'direct', rtol=1e-12, atol=1e-13)
                self.assertTrue(result['converged'])
                errors.append(problem.h*np.linalg.norm(result['u']-problem.exact))
                widths.append(problem.h)
            orders = np.log(np.array(errors[:-1])/errors[1:])/np.log(np.array(widths[:-1])/widths[1:])
            self.assertTrue(np.all(orders > expected_order-.3), (beta, errors, orders))
            self.assertTrue(np.all(orders < expected_order+.3), (beta, errors, orders))


class ForcingTests(unittest.TestCase):
    def test_fixed_initial_and_capped_forcing_terms(self):
        self.assertEqual(core.forcing_eta('ew1', 10.), .5)
        self.assertEqual(core.forcing_eta('ew2', 10.), .5)
        self.assertIsNone(core.forcing_eta('direct', 10.))
        for method in ('fixed_0.01', 'fixed_0.1', 'fixed_0.5', 'fixed_0.9'):
            self.assertEqual(core.forcing_eta(method, 10.), float(method[6:]))
        self.assertEqual(core.forcing_eta('ew1', 0., 1., 0., .01), core.ETA_MIN)
        self.assertEqual(core.forcing_eta('ew2', 10., 1., 0., .01), core.ETA_MAX)

    def test_ew1_golden_safeguard_and_threshold(self):
        value = core.forcing_details('ew1', .2, 1., .1, .5)
        self.assertAlmostEqual(value['raw_eta'], .1)
        self.assertTrue(value['safeguard_applied'])
        self.assertAlmostEqual(value['native_eta'], .5**core.GOLDEN_POWER)
        inactive = core.forcing_details('ew1', .2, 1., .1, .1)
        self.assertFalse(inactive['safeguard_applied'])
        self.assertAlmostEqual(inactive['native_eta'], .1)

    def test_ew2_uses_prespecified_power_two_and_effective_eta_history(self):
        value = core.forcing_details('ew2', .2, 1., .1, .5)
        self.assertEqual(core.EW2_POWER, 2.)
        self.assertAlmostEqual(value['raw_eta'], .9*.2**2)
        self.assertAlmostEqual(value['native_eta'], .9*.5**2)
        self.assertAlmostEqual(core.forcing_eta('ew2', .2, 1., .1, .1), .9*.2**2)
        # Damping can make the accepted-step bound exceed the requested cap.
        self.assertAlmostEqual(core.forcing_eta('ew2', .2, 1., .1, .95), .9*.95**2)


class SolverTests(unittest.TestCase):
    def test_exact_linear_case_direct_has_no_ilu_and_iterative_is_accurate(self):
        problem = core.make_problem(7, dict(eps=.1, gamma=0.))
        with patch.object(core.sla, 'spilu', side_effect=AssertionError('direct must not construct ILU')):
            direct = core.solve(problem, 'direct')
        self.assertTrue(direct['converged'])
        self.assertEqual(direct['newton_iterations'], 1)
        self.assertEqual(direct['components']['preconditioner'], 0.)
        self.assertEqual(direct['linear_iterations'], 0)
        for method in ('ew1', 'ew2', 'fixed_0.01', 'fixed_0.1', 'fixed_0.5', 'fixed_0.9'):
            value = core.solve(problem, method)
            self.assertTrue(value['converged'], (method, value['status']))
            self.assertLessEqual(np.linalg.norm(problem.F(value['u'])), value['tolerance'])
            np.testing.assert_allclose(value['u'], direct['u'], rtol=2e-7, atol=2e-9)

    def test_zero_rhs_stops_without_any_linear_solve(self):
        problem = core.make_problem(3, dict(amplitude=0.))
        with patch.object(core.sla, 'gmres', side_effect=AssertionError('unnecessary linear solve')):
            value = core.solve(problem, 'ew1', capture=True)
        self.assertTrue(value['converged'])
        self.assertEqual(value['newton_iterations'], 0)
        self.assertEqual(value['relative_residual'], 0.)
        self.assertEqual(len(value['snapshots']), 1)

    def test_damped_step_preserves_actual_accepted_linearization_and_eta_bound(self):
        problem = core.make_problem(7, dict(eps=.02, gamma=100., amplitude=50.))
        result = core.solve(problem, 'ew1', capture=True)
        self.assertTrue(result['converged'], result['status'])
        self.assertGreater(result['line_search_rejections'], 0)
        by_iteration = {s['iteration']: s for s in result['snapshots']}
        damped = [r for r in result['trace'] if r['accepted_alpha'] < 1.]
        self.assertTrue(damped)
        for row in damped:
            before, after = by_iteration[row['iteration']], by_iteration[row['iteration']+1]
            residual = problem.F(before['u']) + problem.jacobian(before['u'])@(after['u']-before['u'])
            np.testing.assert_allclose(np.linalg.norm(residual), after['prevlinearizednorm'], rtol=1e-12)
            alpha, eta = row['accepted_alpha'], row['requested_eta']
            self.assertAlmostEqual(after['preveta'], (1.-alpha)+alpha*eta)
            self.assertAlmostEqual(row['effective_eta'], after['preveta'])
            self.assertLessEqual(row['accepted_actual_eta'], row['effective_eta']+1e-12)
            self.assertLessEqual(row['residual_after'], (1.-core.ARMIJO*alpha)*row['residual_before'])

    def test_snapshots_restart_equivalently_without_mutating_inputs(self):
        problem = core.make_problem(9, dict(eps=.02, gamma=25., amplitude=10.))
        source = core.solve(problem, 'ew2', capture=True)
        self.assertTrue(source['converged'])
        self.assertGreater(len(source['snapshots']), 3)
        for state in source['snapshots'][:4]:
            before = state['u'].copy()
            result = core.solve(problem, 'ew2', initial_state=state)
            self.assertTrue(result['converged'])
            np.testing.assert_array_equal(state['u'], before)
            np.testing.assert_array_equal(result['u'], source['u'])
            self.assertEqual(result['F0norm'], source['F0norm'])
            self.assertEqual(result['final_iteration'], source['final_iteration'])
            expected = source['trace'][state['iteration']:]
            self.assertEqual([r['requested_eta'] for r in result['trace']],
                             [r['requested_eta'] for r in expected])
            self.assertEqual([r['accepted_alpha'] for r in result['trace']],
                             [r['accepted_alpha'] for r in expected])

    def test_multiplier_is_applied_once_and_restart_keeps_global_budget(self):
        problem = core.make_problem(7, dict(eps=.02, gamma=25., amplitude=10.))
        source = core.solve(problem, 'ew1', capture=True)
        value = core.solve(problem, 'ew1', initial_state=source['snapshots'][0], first_multiplier=.5)
        self.assertTrue(value['converged'])
        self.assertEqual(value['trace'][0]['requested_eta'], .25)
        self.assertTrue(all(row['requested_eta'] == row['native_eta'] for row in value['trace'][1:]))
        state = source['snapshots'][1]
        no_budget = core.solve(problem, 'ew1', initial_state=state, max_newton=state['iteration'])
        self.assertEqual(no_budget['newton_iterations'], 0)
        self.assertEqual(no_budget['F0norm'], source['F0norm'])
        self.assertEqual(no_budget['status'], 'max_newton')

    def test_true_unpreconditioned_residual_rejects_false_gmres_success(self):
        problem = core.make_problem(4, dict(gamma=1.))
        with patch.object(core.sla, 'gmres', return_value=(np.zeros(problem.dimension), 0)), \
             patch.object(core.sla, 'spsolve', side_effect=AssertionError('no direct recovery')):
            result = core.solve(problem, 'ew1')
        self.assertFalse(result['converged'])
        self.assertEqual(result['linear_failures'], 1)
        self.assertEqual(result['newton_iterations'], 0)
        self.assertEqual(result['trace'][0]['linear_info'], 0)
        self.assertEqual(result['trace'][0]['actual_eta'], 1.)
        self.assertEqual(result['status'], 'linear_failure')

    def test_factorization_and_line_search_failures_are_retained(self):
        problem = core.make_problem(4, dict(gamma=1.))
        with patch.object(core.sla, 'spilu', side_effect=RuntimeError('injected factor failure')):
            result = core.solve(problem, 'ew1')
        self.assertFalse(result['converged'])
        self.assertEqual(result['linear_failures'], 1)
        self.assertEqual(result['status'], 'preconditioner_failure')
        self.assertIn('injected factor failure', result['trace'][0]['error'])
        F0 = problem.F(np.zeros(problem.dimension))
        with patch.object(problem, 'F', return_value=F0.copy()):
            rejected = core.solve(problem, 'direct')
        self.assertFalse(rejected['converged'])
        self.assertEqual(rejected['status'], 'line_search_failure')
        self.assertEqual(rejected['line_search_rejections'], core.MAX_BACKTRACKS+1)
        self.assertEqual(rejected['newton_iterations'], 0)

    def test_components_nonoverlap_and_snapshots_are_separate(self):
        problem = core.make_problem(7, dict(gamma=10., amplitude=5.))
        result = core.solve(problem, 'ew2', capture=True)
        self.assertTrue(result['converged'])
        self.assertEqual(set(result['components']), set(core.COMPONENTS))
        self.assertTrue(all(v >= 0. for v in result['components'].values()))
        self.assertLessEqual(sum(result['components'].values()), result['total_seconds'])
        self.assertGreaterEqual(result['overhead_seconds'], 0.)
        self.assertGreater(result['snapshot_seconds_excluded'], 0.)
        self.assertGreater(result['components']['residual'], 0.)
        self.assertEqual(result['assembly_seconds_excluded'], problem.assembly_seconds)


if __name__ == '__main__':
    unittest.main()
