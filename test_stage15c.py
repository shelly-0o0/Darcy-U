"""Independent small-mesh tests for Stage15C classical recovery baselines.

Only old manufactured data are used here.  No new Stage15C experimental
geometry is solved.  Jacobi recovery is tested against a dense recurrence; it
provides marking information and is never a certified enriched dual solve.
"""

import unittest
from copy import deepcopy
from itertools import count
from unittest.mock import patch

import numpy as np
import scipy.sparse as sparse

import stage15_cases as manufactured
import stage15_fem as fem
import stage15b_adaptive as base
import stage15b_deferred as deferred
import stage15c_adaptive as current
import stage14_solver
import stage15c_cases as new_cases
import stage15c_experiment as experiment
import report_stage15c as report
from stage15b_cases import mesh_audit


def constant(value):
    return lambda x, y: np.full(np.broadcast(x, y).shape, value, dtype=float)


def independent_embedding(p1, p2, vector):
    """P1 vertex values and midpoint means, without a sparse embedding matrix."""
    values = np.zeros(p2.basis.N)
    for vertex in range(p1.mesh.nvertices):
        values[p2.basis.nodal_dofs[0, vertex]] = vector[p1.basis.nodal_dofs[0, vertex]]
    for facet, (first, second) in enumerate(p1.mesh.facets.T):
        values[p2.basis.facet_dofs[0, facet]] = .5*(
            vector[p1.basis.nodal_dofs[0, first]] + vector[p1.basis.nodal_dofs[0, second]])
    return values


class FixedRecoveryAlgebraTests(unittest.TestCase):
    def problem(self):
        case = manufactured.manufactured_case('layered', contrast=100.)
        mesh = fem.make_mesh(4).refined(np.array([2, 9, 17]))
        rhs = (case.source, case.goals[0])
        p1 = fem.assemble(mesh, case.coefficient, rhs, degree=1, coefficient_alignment='element_aligned')
        p2 = fem.assemble(mesh, case.coefficient, rhs, degree=2, coefficient_alignment='element_aligned')
        solved, _ = base.solve_columns(p1, (0, 1))
        return p1, p2, solved[:, 1]

    def test_three_and_eight_iterations_match_independent_dense_recurrence(self):
        p1, p2, z1 = self.problem()
        a = p2.A[p2.free][:, p2.free].toarray()
        diagonal = np.diag(a)
        rhs = p2.loads[p2.free, 1]
        for steps in (3, 8):
            with self.subTest(steps=steps):
                expected = independent_embedding(p1, p2, z1)
                interior = expected[p2.free].copy()
                for _ in range(steps):
                    interior += (2./3.)*(rhs-a @ interior)/diagonal
                expected[p2.free] = interior
                recovered, metrics = current.recover_dual(p1, p2, z1, steps)
                np.testing.assert_allclose(recovered, expected, rtol=3e-13, atol=3e-14)
                np.testing.assert_array_equal(recovered[p2.boundary], 0.)
                self.assertEqual(metrics['steps'], steps)
                self.assertEqual(metrics['omega'], 2./3.)
                self.assertEqual(metrics['matvec_count'], steps)
                self.assertTrue(metrics['unverified_marker'])
                self.assertFalse(metrics['converged_solve_claim'])
                self.assertEqual(metrics['boundary_maximum'], 0.)

    def test_reported_matvec_count_matches_actual_square_sparse_products(self):
        p1, p2, z1 = self.problem()
        original = sparse.csr_matrix.__matmul__
        observed = []
        def counted(matrix, operand):
            if matrix.shape[0] == matrix.shape[1] and np.asarray(operand).ndim == 1:
                observed.append(matrix.shape)
            return original(matrix, operand)
        for steps in (3, 8):
            observed.clear()
            with patch.object(sparse.csr_matrix, '__matmul__', counted):
                _, metrics = current.recover_dual(p1, p2, z1, steps)
            self.assertEqual(len(observed), steps)
            self.assertEqual(metrics['matvec_count'], len(observed))

    def test_recovery_does_not_call_a_linear_solver_or_certify_its_residual(self):
        p1, p2, z1 = self.problem()
        with patch.object(base, 'solve_columns', side_effect=AssertionError('No solve allowed in fixed recovery')):
            recovered, metrics = current.recover_dual(p1, p2, z1, 3)
        relative = np.linalg.norm(p2.loads[p2.free, 1]-p2.A[p2.free][:, p2.free] @ recovered[p2.free]) / np.linalg.norm(p2.loads[p2.free, 1])
        self.assertGreater(relative, 1e-9)
        self.assertFalse(metrics['converged_solve_claim'])
        self.assertTrue(metrics['unverified_marker'])

    def test_recovery_rejects_invalid_iteration_count(self):
        p1, p2, z1 = self.problem()
        for steps in (0, 1, 2, 4, 9, -3, 3.5, True, np.nan):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                current.recover_dual(p1, p2, z1, steps)

    def test_recovery_rejects_nonfinite_or_wrong_shape_guess(self):
        p1, p2, z1 = self.problem()
        bad = z1.copy(); bad[p1.free[0]] = np.nan
        for guess in (z1[:-1], z1[:, None], bad):
            with self.assertRaises(ValueError):
                current.recover_dual(p1, p2, guess, 3)

    def test_recovery_rejects_nonhomogeneous_boundary_or_mismatched_quadrature(self):
        p1, p2, z1 = self.problem()
        bad = z1.copy(); bad[p1.boundary[0]] = .1
        with self.assertRaises(ValueError):
            current.recover_dual(p1, p2, bad, 3)
        other = fem.assemble(p2.mesh, p2.coefficient, p2.rhs_functions, degree=2,
                             intorder=18, coefficient_alignment='element_aligned')
        with self.assertRaises(ValueError):
            current.recover_dual(p1, other, z1, 3)

    def test_recovery_timing_subparts_are_contained_in_total(self):
        p1, p2, z1 = self.problem()
        _, metrics = current.recover_dual(p1, p2, z1, 8)
        parts = metrics['timings']
        self.assertTrue(all(np.isfinite(value) and value >= 0. for value in parts.values()))
        self.assertGreaterEqual(parts['total_seconds'] + 1e-9,
                                sum(value for key, value in parts.items() if key != 'total_seconds'))
        self.assertEqual(metrics['diagonal_extractions'], 1)
        self.assertEqual(metrics['goal_rhs_index'], 1)
        self.assertTrue(metrics['no_reference_or_true_p2_adjoint_used'])


class RecoveryLoopTests(unittest.TestCase):
    def arguments(self, **updates):
        config = dict(max_steps=4, max_seconds=20., verification_delta_factor=1e-10)
        config.update(updates.pop('config', {}))
        arguments = dict(coefficient=constant(1.), source=constant(1.),
                         goal=manufactured.manufactured_case('constant').goals[0],
                         initial_mesh=fem.make_mesh(4), tau=1e-14,
                         marker='recovered_dwr3', config=config)
        arguments.update(updates)
        return arguments

    def assert_same_numerical_trajectory(self, first, second):
        self.assertEqual(first.status, second.status)
        self.assertEqual(len(first.rows), len(second.rows))
        for before, after in zip(first.rows, second.rows):
            for key in ('status', 'qoi', 'mesh_coordinates_sha256', 'mesh_connectivity_sha256',
                        'solution_sha256', 'selected_elements', 'scores', 'verified',
                        'verification_reasons', 'next_mesh_coordinates_sha256',
                        'next_mesh_connectivity_sha256'):
                self.assertEqual(before.get(key), after.get(key), key)
        np.testing.assert_array_equal(first.final_mesh.p, second.final_mesh.p)
        np.testing.assert_array_equal(first.final_mesh.t, second.final_mesh.t)
        np.testing.assert_array_equal(first.final_solution, second.final_solution)

    @staticmethod
    def nonconverging_check(p1, p2, u, z2):
        # Test double only: force a paid check to continue so marking inputs can
        # be changed independently of stopping without using a reference value.
        return dict(total_estimate=1., discretization_estimate=1., algebraic_term=0.,
                    local=None, no_continuous_error_certificate=True, p2_primal_solved=False), None

    def test_unchanged_classical_markers_match_frozen_15b_or_deferred_dwr(self):
        for marker in ('uniform', 'coefficient', 'primal_residual', 'joint_residual', 'dwr'):
            with self.subTest(marker=marker):
                arguments = self.arguments(marker=marker, config={'max_steps': 3})
                reference = deferred if marker == 'dwr' else base
                old = reference.run_adaptive(**arguments)
                new = current.run_adaptive(**arguments)
                self.assert_same_numerical_trajectory(old, new)
                self.assertEqual(old.summary()['solver_calls'], new.summary()['solver_calls'])
                self.assertEqual(old.summary()['rhs_count'], new.summary()['rhs_count'])

    def test_uniform_amg_keeps_old_policy_and_numerical_trajectory(self):
        arguments = self.arguments(marker='uniform', config={'max_steps': 3, 'solver': 'amg'})
        self.assert_same_numerical_trajectory(base.run_adaptive(**arguments), current.run_adaptive(**arguments))

    def test_recovery_uses_fixed_work_and_exact_checks_only_at_original_schedule(self):
        for marker, steps in (('recovered_dwr3', 3), ('recovered_dwr8', 8)):
            with self.subTest(marker=marker):
                result = current.run_adaptive(**self.arguments(marker=marker))
                self.assertEqual(result.status, 'max_steps')
                self.assertEqual([row['verified'] for row in result.rows], [True, False, False, True])
                self.assertEqual([row['recovery'] is not None for row in result.rows], [True, True, True, False])
                self.assertEqual(sum(row['recovery']['matvec_count'] for row in result.rows if row['recovery']), 3*steps)
                for row in result.rows[1:3]:
                    self.assertIsNone(row['dwr'])
                    self.assertFalse(row['stop_candidate'])
                    self.assertTrue(row['recovery']['unverified_marker'])
                    self.assertFalse(row['recovery']['converged_solve_claim'])
                    np.testing.assert_array_equal(row['scores'], abs(np.asarray(row['recovered_local'])))
                self.assertEqual(result.summary()['solver_calls'], 6)
                self.assertEqual(result.summary()['rhs_count'], 10)

    def test_p1_two_rhs_share_one_factor_and_recovery_adds_no_factorization(self):
        original = stage14_solver.sla.splu
        with patch.object(stage14_solver.sla, 'splu', wraps=original) as factor:
            result = current.run_adaptive(**self.arguments())
        self.assertEqual(factor.call_count, 6)
        self.assertEqual(result.summary()['solver_calls'], 6)
        for row in result.rows:
            self.assertEqual(row['solves'][0]['metrics']['requested_columns'], [0, 1])
            self.assertEqual(row['solves'][0]['metrics']['rhs_count'], 2)
            for call in row['solves'][1:]:
                self.assertEqual(call['degree'], 2)
                self.assertEqual(call['metrics']['requested_columns'], [1])
                self.assertEqual(call['metrics']['rhs_count'], 1)

    def test_p2_assembly_is_reused_between_paid_verification_and_recovery(self):
        original = base.assemble
        with patch.object(base, 'assemble', wraps=original) as assemble:
            result = current.run_adaptive(**self.arguments())
        self.assertEqual(sum(call.kwargs.get('degree') == 2 for call in assemble.call_args_list), 4)
        self.assertEqual([row['p2_assembly_usage'] for row in result.rows],
                         [['verification', 'recovery'], ['recovery'], ['recovery'], ['verification']])

    def test_paid_exact_dual_never_substitutes_for_the_recovered_marking_vector(self):
        arguments = self.arguments(config={'max_steps': 2})
        with patch.object(current, 'global_dwr', side_effect=self.nonconverging_check):
            normal = current.run_adaptive(**arguments)
        original = base.solve_columns
        def different_paid_dual(problem, columns, method='direct'):
            values, metrics = original(problem, columns, method)
            # Controlled dependency intervention, not a valid solver record:
            # the genuine residual fields are intentionally left unchanged.
            if problem.degree == 2:
                values = 7.*values
            return values, metrics
        with patch.object(current, 'global_dwr', side_effect=self.nonconverging_check), \
                patch.object(base, 'solve_columns', side_effect=different_paid_dual):
            intervened = current.run_adaptive(**arguments)
        self.assert_same_numerical_trajectory(normal, intervened)
        for before, after in zip(normal.rows, intervened.rows):
            self.assertEqual(before['recovered_local'], after['recovered_local'])

    def test_successful_paid_stop_does_not_compute_recovery_or_marking(self):
        with patch.object(current, 'recover_dual') as recover, patch.object(current, 'localize_dwr') as localize:
            result = current.run_adaptive(**self.arguments(tau=1.))
        self.assertEqual(result.status, 'estimated_converged')
        self.assertTrue(result.rows[0]['verified'])
        self.assertIsNone(result.rows[0]['recovery'])
        self.assertIsNone(result.rows[0]['recovered_local'])
        self.assertEqual(result.rows[0]['p2_assembly_usage'], ['verification'])
        recover.assert_not_called(); localize.assert_not_called()

    def test_zero_recovered_indicator_does_not_authorize_success(self):
        with patch.object(current, 'global_dwr', side_effect=self.nonconverging_check), \
                patch.object(current, 'localize_dwr', side_effect=lambda p1, *args, **kwargs: np.zeros(p1.mesh.nelements)):
            result = current.run_adaptive(**self.arguments())
        self.assertEqual(result.status, 'zero_indicator')
        self.assertFalse(result.rows[0]['stop_candidate'])
        self.assertFalse(result.summary()['estimated_converged'])
        self.assertEqual(result.rows[0]['selected_elements'], [])

    def test_linear_failure_blocks_recovery_and_success(self):
        original = base.solve_columns
        for failed_degree in (1, 2):
            def failure(problem, columns, method='direct'):
                values, metrics = original(problem, columns, method)
                if problem.degree == failed_degree:
                    metrics['accepted'] = metrics['converged'] = False
                return values, metrics
            with self.subTest(degree=failed_degree), \
                    patch.object(base, 'solve_columns', side_effect=failure), \
                    patch.object(current, 'recover_dual') as recover:
                result = current.run_adaptive(**self.arguments(tau=1.))
            self.assertEqual(result.status, 'linear_solver_failed')
            self.assertFalse(result.rows[0]['verified'])
            self.assertIsNotNone(result.final_solution)
            recover.assert_not_called()

    def test_initial_resource_limits_preserve_failure_without_recovery(self):
        with patch.object(current, 'recover_dual') as recover:
            result = current.run_adaptive(**self.arguments(config={'max_free_dofs': 1}))
        self.assertEqual(result.status, 'max_dofs')
        self.assertIsNone(result.final_solution)
        recover.assert_not_called()
        clock = count(0., .1)
        with patch.object(current, 'perf_counter', side_effect=lambda: next(clock)), \
                patch.object(current, 'recover_dual') as recover:
            result = current.run_adaptive(**self.arguments(config={'max_seconds': .15}))
        self.assertEqual(result.status, 'time_limit')
        self.assertGreater(result.algorithm_seconds, .15)
        recover.assert_not_called()

    def test_refinement_dof_limit_and_recovery_exception_keep_paid_solve_cost(self):
        result = current.run_adaptive(**self.arguments(config={'max_free_dofs': 9}))
        self.assertEqual(result.status, 'max_dofs')
        self.assertIsNotNone(result.rows[0]['recovery'])
        self.assertGreater(result.rows[0]['next_p1_free_dofs'], 9)
        self.assertEqual(result.final_mesh.nvertices, 25)
        with patch.object(current, 'recover_dual', side_effect=ValueError('recovery diagnostic failure')):
            failed = current.run_adaptive(**self.arguments())
        self.assertEqual(failed.status, 'exception')
        self.assertEqual(failed.summary()['rhs_count'], 3)
        self.assertIn('recovery diagnostic failure', failed.rows[0]['exception']['message'])

    def test_recovery_timing_is_charged_once_in_outer_ledger(self):
        result = current.run_adaptive(**self.arguments())
        for row in result.rows:
            times = row['timings']
            self.assertGreaterEqual(times['step_total_seconds']+1e-9,
                sum(value for name, value in times.items() if name not in ('step_total_seconds', 'unattributed_seconds')))
            if row['recovery'] is not None:
                self.assertGreaterEqual(times['recovery_seconds'], row['recovery']['timings']['total_seconds'])
                self.assertNotIn('matvec_seconds', times)
                self.assertGreater(times['recovery_localization_seconds'], 0.)

    def test_reference_or_exact_solution_configuration_cannot_enter_the_loop(self):
        for config in ({'reference_qoi': 1.}, {'exact_solution': lambda x, y: x+y}):
            with self.assertRaises(ValueError):
                current.run_adaptive(**self.arguments(config=config))


class PreSolveGeometryContractTests(unittest.TestCase):
    """Inspect only frozen design metadata and meshes; solve no new PDE."""
    def test_stratified_without_replacement_sample_is_reproducible_and_new(self):
        first, second = new_cases.specifications(), new_cases.specifications()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(len({item['name'] for item in first}), 24)
        old = {'layers': (2, 4, 6), 'barrier': (4, 5),
               'channel': (3, 5), 'inclusion': (4, 6, 2, 6)}
        for family in new_cases.FAMILIES:
            subset = [item for item in first if item['family'] == family]
            self.assertEqual(len(subset), 6)
            geometry = [tuple(item['geometry_eighths']) for item in subset]
            self.assertEqual(len(set(geometry)), 6)
            self.assertNotIn(old[family], geometry)
            self.assertEqual(sum(item['contrast'] == 10 for item in subset), 3)
            self.assertEqual(sum(item['contrast'] == 1000 for item in subset), 3)

    def test_family_geometry_constraints_match_the_predeclared_finite_pools(self):
        for spec in new_cases.specifications():
            geometry = spec['geometry_eighths']
            self.assertTrue(all(isinstance(value, int) and 1 <= value <= 7 for value in geometry))
            if spec['family'] == 'layers':
                self.assertEqual(len(geometry), 3)
                self.assertEqual(geometry, sorted(set(geometry)))
            elif spec['family'] == 'barrier':
                self.assertIn(geometry[1]-geometry[0], (1, 2))
            elif spec['family'] == 'channel':
                self.assertIn(geometry[1]-geometry[0], (1, 2, 3))
            else:
                self.assertIn(geometry[1]-geometry[0], (1, 2, 3))
                self.assertIn(geometry[3]-geometry[2], (1, 2, 3, 4))

    def test_new_geometries_are_initially_aligned_and_keep_fixed_source_and_goal(self):
        mesh = fem.make_mesh(8)
        refined = mesh.refined(np.array([1, 8, 17, 39, 65, 101]))
        items = new_cases.cases()
        centers = refined.p[:, refined.t].mean(axis=1)
        for case in items:
            with self.subTest(case=case.name):
                self.assertTrue(mesh_audit(mesh, case)['accepted'])
                self.assertTrue(mesh_audit(refined, case)['accepted'])
                self.assertIs(case.source, items[0].source)
                self.assertIs(case.goal, items[0].goal)
                self.assertFalse(hasattr(case, 'exact'))
                self.assertFalse(hasattr(case, 'reference_qoi'))
                k = case.coefficient(centers[0], centers[1])
                self.assertTrue(np.all(np.isin(k, [case.metadata['k_min'], 1.])))
                self.assertTrue(np.any(k == case.metadata['k_min']))
                self.assertTrue(np.any(k == 1.))


class ScreeningContractTests(unittest.TestCase):
    """Synthetic cost records only; no new PDE or measured speed claim."""
    def cohort(self):
        freeze = dict(config=deepcopy(experiment.CONFIG), cases=[
            dict(name=f'layers_{i:02d}', metadata=dict(family='layers', contrast=10))
            for i in range(6)])
        pairs = []
        for case in freeze['cases']:
            for tau in freeze['config']['taus']:
                methods = []
                for i, method in enumerate(experiment.METHODS):
                    dwr = method == 'dwr'
                    methods.append(dict(method=method, all_success=True,
                        median_seconds=8. if dwr else 10.+i,
                        median_final_p1_dofs=50 if dwr else 100,
                        median_nonterminal_seconds=4. if dwr else 0.,
                        median_nonterminal_fraction=.5 if dwr else 0.,
                        median_hypothetical_remaining_seconds=4. if dwr else 10.+i))
                pairs.append(dict(case=case['name'], tau=tau, reference_qualified=True,
                                  methods=methods))
        return freeze, pairs

    @staticmethod
    def dwr(pair):
        return next(row for row in pair['methods'] if row['method'] == 'dwr')

    def test_screening_thresholds_match_frozen_design_and_never_approve_ai(self):
        freeze, pairs = self.cohort()
        cfg = freeze['config']['screening']
        self.assertEqual(cfg, dict(min_nonterminal_fraction=.30, max_hypothetical_vs_control=.80,
            max_dwr_time_vs_control=.90, max_dwr_dofs_vs_fastest_control=.75,
            min_geometries_per_family=4, require_both_taus=True, require_all_methods_success=True))
        result = report.screen_opportunity(freeze, pairs)
        self.assertEqual(result['families'][0]['passed'], 6)
        self.assertEqual(result['decision'], 'LIMITED_DESIGN_ONLY')
        self.assertEqual(result['G1'], 'NOT_FULLY_APPROVED')
        self.assertEqual(result['G3'], 'NOT_EVALUATED')
        self.assertTrue(result['no_ai_model'])
        self.assertTrue(result['retrospective_nonterminal_labels'])
        self.assertTrue(result['hypothetical_cost_not_measured_speedup'])

    def test_one_failed_tau_blocks_geometry_even_when_other_tau_passes(self):
        freeze, pairs = self.cohort()
        self.dwr(pairs[1])['median_nonterminal_fraction'] = .299
        result = report.screen_opportunity(freeze, pairs)
        first = result['geometries'][0]
        self.assertEqual(first['settings'], 2)
        self.assertEqual(first['passed_settings'], 1)
        self.assertFalse(first['passed'])

    def test_failed_control_cannot_promote_geometry_by_being_excluded(self):
        freeze, pairs = self.cohort()
        pairs[0]['methods'][0]['all_success'] = False
        result = report.screen_opportunity(freeze, pairs)
        first = result['settings'][0]
        self.assertEqual(first['fastest_control'], 'uniform_amg')
        self.assertTrue(first['cost_necessary'])
        self.assertTrue(first['goal_information'])
        self.assertFalse(first['numerical'])
        self.assertFalse(first['passed'])
        self.assertFalse(result['geometries'][0]['passed'])

    def test_reference_or_all_control_failure_prevents_acceptance(self):
        for mode in ('reference', 'controls'):
            with self.subTest(mode=mode):
                freeze, pairs = self.cohort()
                if mode == 'reference':
                    pairs[0]['reference_qualified'] = False
                else:
                    for row in pairs[0]['methods']:
                        if row['method'] != 'dwr':
                            row['all_success'] = False
                first = report.screen_opportunity(freeze, pairs)['settings'][0]
                self.assertFalse(first['passed'])
                self.assertFalse(first['numerical'])
                self.assertFalse(first['cost_necessary'])
                if mode == 'controls':
                    self.assertIsNone(first['fastest_control'])

    def test_three_of_six_stops_family_but_four_retains_limited_design(self):
        for successes in (3, 4):
            with self.subTest(successes=successes):
                freeze, pairs = self.cohort()
                for pair in pairs[2*successes:]:
                    self.dwr(pair)['median_nonterminal_fraction'] = .1
                result = report.screen_opportunity(freeze, pairs)
                self.assertEqual(result['families'][0]['passed'], successes)
                self.assertEqual(result['families'][0]['retain_for_limited_design'], successes >= 4)
                self.assertEqual(result['retained_families'], ['layers'] if successes >= 4 else [])
                self.assertEqual(result['decision'], 'LIMITED_DESIGN_ONLY' if successes >= 4
                                 else 'STOP_CURRENT_ESTIMATOR_REPLACEMENT_BRANCH')

    def test_cost_necessary_requires_both_fraction_and_hypothetical_time(self):
        for fraction, remaining, expected in ((.30, 8., True), (.299999, 8., False), (.30, 8.00001, False)):
            with self.subTest(fraction=fraction, remaining=remaining):
                freeze, pairs = self.cohort()
                dwr = self.dwr(pairs[0])
                dwr.update(median_nonterminal_fraction=fraction,
                           median_hypothetical_remaining_seconds=remaining)
                first = report.screen_opportunity(freeze, pairs)['settings'][0]
                self.assertEqual(first['cost_necessary'], expected)
                self.assertEqual(first['passed'], expected)

    def test_goal_information_uses_time_or_dofs_but_does_not_waive_cost_gate(self):
        for seconds, dofs, expected in ((9., 100, True), (12., 75, True), (12., 76, False)):
            with self.subTest(seconds=seconds, dofs=dofs):
                freeze, pairs = self.cohort()
                self.dwr(pairs[0]).update(median_seconds=seconds, median_final_p1_dofs=dofs)
                first = report.screen_opportunity(freeze, pairs)['settings'][0]
                self.assertEqual(first['goal_information'], expected)
                self.assertEqual(first['passed'], expected)


if __name__ == '__main__':
    unittest.main()
