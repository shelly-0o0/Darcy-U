"""Independent equivalence/expense-boundary checks for deferred localization.

The frozen Stage15B baseline is used for trajectory comparison.  This verifies
an engineering optimization, not a new DWR estimator or a continuous guarantee.
"""

import unittest
from itertools import count
from unittest.mock import patch

import numpy as np

import stage15_cases as cases
import stage15_fem as fem
import stage15b_adaptive as baseline
import stage15b_deferred as deferred


def constant(value):
    return lambda x, y: np.full(np.broadcast(x, y).shape, value, dtype=float)


class DeferredLocalizationTests(unittest.TestCase):
    def arguments(self, **updates):
        config = dict(max_steps=3, max_seconds=20.)
        config.update(updates.pop('config', {}))
        arguments = dict(coefficient=constant(1.), source=constant(1.),
                         goal=cases.manufactured_case('constant').goals[0],
                         initial_mesh=fem.make_mesh(4), tau=1e-14, marker='dwr', config=config)
        arguments.update(updates)
        return arguments

    def assert_same_trajectory(self, old, new):
        self.assertEqual(old.status, new.status)
        self.assertEqual(len(old.rows), len(new.rows))
        for before, after in zip(old.rows, new.rows):
            for key in ('step', 'status', 'qoi', 'verified', 'verification_reasons',
                        'mesh_coordinates_sha256', 'mesh_connectivity_sha256', 'selected_elements',
                        'p1_free_dofs', 'p2_free_dofs', 'solution_sha256'):
                self.assertEqual(before.get(key), after.get(key), key)
            for key in ('next_mesh_coordinates_sha256', 'next_mesh_connectivity_sha256'):
                self.assertEqual(before.get(key), after.get(key), key)
            if before['dwr'] is not None:
                for key in ('total_estimate', 'discretization_estimate', 'algebraic_term'):
                    self.assertAlmostEqual(before['dwr'][key], after['dwr'][key], delta=2e-14)
            if before['selected_elements']:
                np.testing.assert_allclose(before['dwr']['local'], after['dwr']['local'], atol=0., rtol=0.)
                np.testing.assert_allclose(before['scores'], after['scores'], atol=0., rtol=0.)
        np.testing.assert_array_equal(old.final_mesh.p, new.final_mesh.p)
        np.testing.assert_array_equal(old.final_mesh.t, new.final_mesh.t)
        np.testing.assert_allclose(old.final_solution, new.final_solution, atol=0., rtol=0.)
        self.assertEqual(old.summary()['solver_calls'], new.summary()['solver_calls'])
        self.assertEqual(old.summary()['rhs_count'], new.summary()['rhs_count'])

    def test_immediate_estimated_stop_never_localizes(self):
        arguments = self.arguments(tau=1.)
        with patch.object(deferred, 'localize_dwr', wraps=deferred.localize_dwr) as localize:
            new = deferred.run_adaptive(**arguments)
        old = baseline.run_adaptive(**arguments)
        self.assertEqual(new.status, 'estimated_converged')
        self.assertEqual(len(new.rows), 1)
        localize.assert_not_called()
        self.assertIsNone(new.rows[0]['dwr']['local'])
        self.assertEqual(new.rows[0]['selected_elements'], [])
        self.assert_same_trajectory(old, new)

    def test_continue_states_match_baseline_and_final_max_steps_skips_localization(self):
        arguments = self.arguments()
        old = baseline.run_adaptive(**arguments)
        with patch.object(deferred, 'localize_dwr', wraps=deferred.localize_dwr) as localize:
            new = deferred.run_adaptive(**arguments)
        self.assertEqual(new.status, 'max_steps')
        self.assertEqual(len(new.rows), 3)
        self.assertEqual(localize.call_count, 2)
        self.assertIsNone(new.rows[-1]['dwr']['local'])
        self.assert_same_trajectory(old, new)

    def test_layered_material_path_matches_baseline_and_continuing_local_signs(self):
        material = cases.manufactured_case('layered', contrast=1000.)
        arguments = self.arguments(coefficient=material.coefficient, source=material.source,
                                   goal=material.goals[1])
        old = baseline.run_adaptive(**arguments)
        new = deferred.run_adaptive(**arguments)
        self.assert_same_trajectory(old, new)
        self.assertTrue(any(row['selected_elements'] for row in new.rows))

    def test_global_estimate_reuses_prepared_vectors_for_optional_localization(self):
        original = deferred.global_dwr
        prepared_vectors = []
        def record_prepared(*args, **kwargs):
            estimate, prepared = original(*args, **kwargs)
            self.assertIsNone(estimate['local'])
            prepared_vectors.append(prepared)
            return estimate, prepared
        with patch.object(deferred, 'global_dwr', side_effect=record_prepared) as estimator, \
                patch.object(deferred, 'localize_dwr', wraps=deferred.localize_dwr) as localize, \
                patch.object(deferred.base, 'prolongation_p1_p2', wraps=deferred.base.prolongation_p1_p2) as embedding:
            result = deferred.run_adaptive(**self.arguments())
        self.assertEqual(estimator.call_count, len(result.rows))
        self.assertEqual(embedding.call_count, len(result.rows))
        self.assertEqual(localize.call_count, len(result.rows)-1)
        for call, prepared in zip(localize.call_args_list, prepared_vectors):
            self.assertIs(call.kwargs['prepared'], prepared)

    def test_p2_solves_remain_adjoint_only(self):
        result = deferred.run_adaptive(**self.arguments())
        calls = [call for row in result.rows for call in row['solves']]
        self.assertEqual(len(calls), 2*len(result.rows))
        self.assertTrue(all(call['metrics']['rhs_count'] == 1 for call in calls))
        self.assertTrue(all(call['metrics']['requested_columns'] == [1] for call in calls if call['degree'] == 2))
        self.assertTrue(all(call['metrics']['requested_columns'] == [0] for call in calls if call['degree'] == 1))
        self.assertTrue(all(row['dwr']['p2_primal_solved'] is False for row in result.rows))

    def test_one_state_budget_keeps_failure_without_localization(self):
        with patch.object(deferred, 'localize_dwr', wraps=deferred.localize_dwr) as localize:
            result = deferred.run_adaptive(**self.arguments(config={'max_steps': 1}))
        self.assertEqual(result.status, 'max_steps')
        self.assertFalse(result.rows[0]['stop_candidate'])
        self.assertIsNotNone(result.final_solution)
        localize.assert_not_called()

    def test_initial_dof_limit_prevents_solves_and_localization(self):
        with patch.object(deferred, 'localize_dwr') as localize, \
                patch.object(deferred.base, 'solve_columns') as solver:
            result = deferred.run_adaptive(**self.arguments(config={'max_free_dofs': 1}))
        self.assertEqual(result.status, 'max_dofs')
        self.assertIsNone(result.final_solution)
        self.assertEqual(result.summary()['rhs_count'], 0)
        localize.assert_not_called()
        solver.assert_not_called()

    def test_refinement_dof_limit_retains_last_solution_and_charged_localization(self):
        arguments = self.arguments(config={'max_free_dofs': 9})
        old = baseline.run_adaptive(**arguments)
        with patch.object(deferred, 'localize_dwr', wraps=deferred.localize_dwr) as localize:
            new = deferred.run_adaptive(**arguments)
        self.assertEqual(new.status, 'max_dofs')
        self.assertEqual(localize.call_count, 1)
        self.assertIsNotNone(new.final_solution)
        self.assertGreater(new.rows[0]['next_p1_free_dofs'], 9)
        self.assert_same_trajectory(old, new)

    def test_primal_linear_failure_prevents_localization_and_success(self):
        original = deferred.base.solve_columns
        def fail(problem, columns, method='direct'):
            solution, metrics = original(problem, columns, method)
            metrics['accepted'] = metrics['converged'] = False
            return solution, metrics
        with patch.object(deferred.base, 'solve_columns', side_effect=fail), \
                patch.object(deferred, 'localize_dwr') as localize:
            result = deferred.run_adaptive(**self.arguments(tau=1.))
        self.assertEqual(result.status, 'linear_solver_failed')
        self.assertEqual(result.summary()['rhs_count'], 1)
        self.assertIsNotNone(result.final_solution)
        self.assertFalse(result.rows[0]['verified'])
        localize.assert_not_called()

    def test_dual_linear_failure_prevents_localization_and_success(self):
        original = deferred.base.solve_columns
        def fail(problem, columns, method='direct'):
            solution, metrics = original(problem, columns, method)
            if problem.degree == 2:
                metrics['accepted'] = metrics['converged'] = False
            return solution, metrics
        with patch.object(deferred.base, 'solve_columns', side_effect=fail), \
                patch.object(deferred, 'localize_dwr') as localize:
            result = deferred.run_adaptive(**self.arguments(tau=1.))
        self.assertEqual(result.status, 'linear_solver_failed')
        self.assertEqual(result.summary()['rhs_count'], 2)
        self.assertFalse(result.rows[0]['verified'])
        localize.assert_not_called()

    def test_soft_time_limit_keeps_actual_elapsed_and_does_not_localize(self):
        clock = count(0., .1)
        with patch.object(deferred, 'perf_counter', side_effect=lambda: next(clock)), \
                patch.object(deferred, 'localize_dwr') as localize, \
                patch.object(deferred.base, 'solve_columns') as solver:
            result = deferred.run_adaptive(**self.arguments(config={'max_seconds': .15}))
        self.assertEqual(result.status, 'time_limit')
        self.assertGreater(result.algorithm_seconds, .15)
        localize.assert_not_called()
        solver.assert_not_called()

    def test_localization_exception_is_recorded_without_discarding_solve_cost(self):
        with patch.object(deferred, 'localize_dwr', side_effect=ValueError('independent localization failure')):
            result = deferred.run_adaptive(**self.arguments())
        self.assertEqual(result.status, 'exception')
        self.assertIn('independent localization failure', result.rows[0]['exception']['message'])
        self.assertEqual(result.summary()['rhs_count'], 2)
        self.assertIsNotNone(result.final_solution)

    def test_localization_is_timed_only_where_it_is_needed(self):
        result = deferred.run_adaptive(**self.arguments())
        for row in result.rows[:-1]:
            self.assertGreater(row['timings']['dwr_localization_seconds'], 0.)
            self.assertGreater(row['timings']['step_total_seconds'], row['timings']['dwr_localization_seconds'])
        self.assertEqual(result.rows[-1]['timings'].get('dwr_localization_seconds', 0.), 0.)

    def test_prepared_global_and_local_terms_preserve_inexact_primal_algebra(self):
        material = cases.manufactured_case('layered', contrast=100.)
        mesh = fem.make_mesh(4).refined(np.array([2, 9, 17]))
        kwargs = dict(coefficient_alignment='element_aligned')
        p1 = fem.assemble(mesh, material.coefficient, (material.source, material.goals[0]), degree=1, **kwargs)
        p2 = fem.assemble(mesh, material.coefficient, (material.source, material.goals[0]), degree=2, **kwargs)
        u, _ = baseline.solve_columns(p1, (0,))
        z, _ = baseline.solve_columns(p2, (1,))
        perturbed = u[:, 0].copy()
        perturbed[p1.free] += .003*np.cos(np.arange(len(p1.free)))
        old = baseline.online_dwr(p1, p2, perturbed, z[:, 0], localize=True)
        new, prepared = deferred.global_dwr(p1, p2, perturbed, z[:, 0])
        self.assertGreater(abs(new['algebraic_term']), 1e-8)
        for key in ('total_estimate', 'algebraic_term', 'discretization_estimate'):
            self.assertEqual(old[key], new[key])
        cached = deferred.localize_dwr(p1, p2, perturbed, z[:, 0], prepared=prepared)
        uncached = deferred.localize_dwr(p1, p2, perturbed, z[:, 0])
        np.testing.assert_array_equal(cached, old['local'])
        np.testing.assert_array_equal(cached, uncached)


if __name__ == '__main__':
    unittest.main()
