"""Independent small-mesh checks for the Stage 15B baseline comparison.

These tests check implemented algebra and online information boundaries.  They
do not turn residual indicators, DWR stopping, or reference stability into a
continuous-error certificate.  The residual reference uses affine triangle
geometry and exact constant-load edge/volume integrals, independently of skfem.
"""

import unittest
from copy import deepcopy
from itertools import count
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import leggauss

import stage15_cases as cases
import stage15_fem as fem
import stage15b_adaptive as adaptive
import stage14_solver
import stage15b_cases as applications
import stage15b_reference as reference
import stage15b_experiment as experiment


def constant(value):
    return lambda x, y: np.full(np.broadcast(x, y).shape, value, dtype=float)


def independent_residual(mesh, nodal_solution, coefficient, source):
    """Exact P1 residual pieces for element-constant k and constant source."""
    count = mesh.t.shape[1]
    body = np.zeros(count)
    face = np.zeros(count)
    gradients = np.zeros((count, 2))
    coefficients = np.zeros(count)
    edges = {}
    for element, nodes in enumerate(mesh.t.T):
        points = mesh.p[:, nodes].T
        affine = np.column_stack((np.ones(3), points))
        area = abs(np.linalg.det(affine)) / 2.
        gradients[element] = np.linalg.solve(affine, nodal_solution[nodes])[1:]
        center = points.mean(axis=0)
        coefficients[element] = float(coefficient(center[0], center[1]))
        diameter = max(np.linalg.norm(points[i] - points[j])
                       for i, j in ((0, 1), (1, 2), (2, 0)))
        body[element] = diameter ** 2 * area * source ** 2 / coefficients[element]
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(element)
    for (a, b), elements in edges.items():
        if len(elements) != 2:
            continue
        left, right = elements
        tangent = mesh.p[:, b] - mesh.p[:, a]
        length = float(np.linalg.norm(tangent))
        normal = np.array((tangent[1], -tangent[0])) / length
        jump = float((coefficients[left] * gradients[left] -
                      coefficients[right] * gradients[right]) @ normal)
        contribution = .5 * length ** 2 * jump ** 2 / max(coefficients[left], coefficients[right])
        face[left] += contribution
        face[right] += contribution
    return body, face


class ResidualIndicatorTests(unittest.TestCase):
    def test_constant_coefficient_affine_field_has_only_exact_body_piece(self):
        mesh = fem.make_mesh(4)
        problem = fem.assemble(mesh, constant(3.5), (constant(2.),),
                               coefficient_alignment='element_aligned')
        x, y = problem.basis.doflocs
        result = adaptive.residual_scores(problem, .7 + 2.*x - 3.*y)
        body, face = independent_residual(mesh, .7 + 2.*x - 3.*y, constant(3.5), 2.)
        np.testing.assert_allclose(result['body'], body, rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(result['face'], face, atol=1e-25)
        np.testing.assert_allclose(result['scores'], body + face, rtol=2e-13)

    def test_material_flux_jumps_match_independent_edge_integrals(self):
        mesh = fem.make_mesh(4).refined(np.array([2, 9, 17]))
        coefficient = cases.manufactured_case('layered', contrast=100.).coefficient
        problem = fem.assemble(mesh, coefficient, (constant(1.),),
                               coefficient_alignment='element_aligned')
        x, y = problem.basis.doflocs
        u = x + .3*y
        result = adaptive.residual_scores(problem, u)
        body, face = independent_residual(mesh, u, coefficient, 1.)
        self.assertGreater(float(face.max()), 0.)
        np.testing.assert_allclose(result['body'], body, rtol=3e-12, atol=2e-15)
        np.testing.assert_allclose(result['face'], face, rtol=3e-12, atol=2e-13)
        np.testing.assert_allclose(result['scores'], body + face, rtol=3e-12, atol=2e-13)

    def test_general_nodal_field_matches_independent_residual(self):
        mesh = fem.make_mesh(4).refined(np.array([0, 5, 18]))
        coefficient = cases.manufactured_case('layered', contrast=10., barrier=True).coefficient
        problem = fem.assemble(mesh, coefficient, (constant(1.7),),
                               coefficient_alignment='element_aligned')
        u = np.sin(np.arange(problem.basis.N) * .7)
        expected_body, expected_face = independent_residual(mesh, u, coefficient, 1.7)
        result = adaptive.residual_scores(problem, u)
        np.testing.assert_allclose(result['body'], expected_body, rtol=3e-12, atol=1e-14)
        np.testing.assert_allclose(result['face'], expected_face, rtol=3e-12, atol=1e-11)

    def test_residual_changes_when_right_hand_side_changes(self):
        problem = fem.assemble(fem.make_mesh(4), constant(1.),
                               (constant(1.), constant(3.)), coefficient_alignment='element_aligned')
        u = np.zeros(problem.basis.N)
        first = adaptive.residual_scores(problem, u, rhs_index=0)
        second = adaptive.residual_scores(problem, u, rhs_index=1)
        np.testing.assert_allclose(second['body'], 9.*first['body'], rtol=3e-13)
        np.testing.assert_allclose(first['face'], 0., atol=1e-28)
        np.testing.assert_allclose(second['face'], 0., atol=1e-28)


class OnlineDWRTests(unittest.TestCase):
    def paired(self, goal_index=0):
        case = cases.manufactured_case('layered', contrast=100.)
        mesh = fem.make_mesh(4).refined(np.array([2, 9, 17]))
        rhs = (case.source, case.goals[goal_index])
        p1 = fem.assemble(mesh, case.coefficient, rhs, degree=1,
                          coefficient_alignment='element_aligned')
        p2 = fem.assemble(mesh, case.coefficient, rhs, degree=2,
                          coefficient_alignment='element_aligned')
        u1, _ = fem.solve(p1)
        u2, _ = fem.solve(p2)
        return p1, p2, u1[:, 0], u2

    def test_online_dwr_agrees_with_full_stage15a_audit(self):
        p1, p2, u, u2 = self.paired()
        online = adaptive.online_dwr(p1, p2, u, u2[:, 1], localize=True)
        audit = fem.dwr(p1, p2, u, u2[:, 1], goal_indices=(1,), p2_primal=u2[:, 0])
        self.assertAlmostEqual(online['total_estimate'], float(audit['total_estimates'][0]), delta=2e-14)
        self.assertAlmostEqual(online['algebraic_term'], float(audit['algebraic_terms'][0]), delta=2e-14)
        self.assertAlmostEqual(online['discretization_estimate'], float(audit['global_estimates'][0]), delta=2e-14)
        np.testing.assert_allclose(online['local'], audit['strong']['signed_element_contributions'][:, 0],
                                   rtol=2e-11, atol=2e-14)
        qdiff = float(p2.loads[:, 1] @ u2[:, 0] - p1.loads[:, 1] @ u)
        self.assertAlmostEqual(online['total_estimate'], qdiff, delta=2e-13)

    def test_inexact_primal_preserves_nonzero_algebraic_term(self):
        p1, p2, u, u2 = self.paired()
        perturbed = u.copy()
        perturbed[p1.free] += .003 * np.cos(np.arange(len(p1.free)))
        online = adaptive.online_dwr(p1, p2, perturbed, u2[:, 1], localize=True)
        audit = fem.dwr(p1, p2, perturbed, u2[:, 1], goal_indices=(1,), p2_primal=u2[:, 0])
        self.assertGreater(abs(online['algebraic_term']), 1e-8)
        self.assertAlmostEqual(online['total_estimate'], float(audit['qoi_differences'][0]), delta=2e-13)
        self.assertAlmostEqual(online['total_estimate'], online['discretization_estimate'] +
                               online['algebraic_term'], delta=2e-14)

    def test_goal_sign_reverses_signed_dwr_and_not_magnitude(self):
        p1, p2, u, u2 = self.paired()
        positive = adaptive.online_dwr(p1, p2, u, u2[:, 1], localize=True)
        negative = adaptive.online_dwr(p1, p2, u, -u2[:, 1], localize=True)
        for key in ('total_estimate', 'algebraic_term', 'discretization_estimate'):
            self.assertAlmostEqual(positive[key], -negative[key], delta=2e-14)
        np.testing.assert_allclose(positive['local'], -negative['local'], rtol=2e-13, atol=2e-14)

    def test_goal_switch_changes_dual_and_indicator_with_fixed_primal(self):
        p1a, p2a, ua, u2a = self.paired(goal_index=0)
        p1b, p2b, ub, u2b = self.paired(goal_index=1)
        np.testing.assert_allclose(p1a.A.toarray(), p1b.A.toarray(), atol=0.)
        np.testing.assert_allclose(p1a.loads[:, 0], p1b.loads[:, 0], atol=0.)
        np.testing.assert_allclose(ua, ub, atol=2e-14)
        first = adaptive.online_dwr(p1a, p2a, ua, u2a[:, 1], localize=True)
        second = adaptive.online_dwr(p1b, p2b, ub, u2b[:, 1], localize=True)
        self.assertGreater(float(np.linalg.norm(u2a[:, 1] - u2b[:, 1])), 1e-5)
        self.assertGreater(float(np.linalg.norm(first['local'] - second['local'])), 1e-7)


class AccountingAndOnlineControlTests(unittest.TestCase):
    def run_small(self, **kwargs):
        config = dict(max_steps=4, max_seconds=20., verification_delta_factor=1e-10)
        config.update(kwargs.pop('config', {}))
        return adaptive.run_adaptive(constant(1.), constant(1.),
            cases.manufactured_case('constant').goals[0], initial_mesh=fem.make_mesh(4),
            tau=kwargs.pop('tau', 1e-12), config=config, **kwargs)

    @staticmethod
    def nonconverging_dwr(p1, p2, u, z2, *, localize=False):
        return dict(total_estimate=1., discretization_estimate=1., algebraic_term=0.,
                    local=np.ones(p1.mesh.nelements) if localize else None,
                    no_continuous_error_certificate=True, p2_primal_solved=False)

    def test_requested_p1_columns_share_one_factorization(self):
        problem = fem.assemble(fem.make_mesh(4), constant(1.), (constant(1.), constant(3.)),
                               coefficient_alignment='element_aligned')
        real_factor = stage14_solver.sla.splu
        with patch.object(stage14_solver.sla, 'splu', wraps=real_factor) as factor:
            solution, metrics = adaptive.solve_columns(problem, (0, 1), 'direct')
        self.assertEqual(factor.call_count, 1)
        self.assertEqual(metrics['rhs_count'], 2)
        self.assertEqual(metrics['requested_columns'], [0, 1])
        self.assertTrue(adaptive.accepted(metrics))
        np.testing.assert_allclose(solution[:, 1], 3. * solution[:, 0], rtol=2e-13, atol=1e-14)
        self.assertGreater(metrics['timings']['factor_or_scaled_amg_setup_seconds'], 0.)
        self.assertGreater(metrics['fem_solve_total_seconds'], metrics['timings']['initial_solve_seconds'])

    def test_p2_adjoint_request_does_not_solve_enriched_primal(self):
        problem = fem.assemble(fem.make_mesh(4), constant(1.), (constant(1.), constant(3.)),
                               degree=2, coefficient_alignment='element_aligned')
        solution, metrics = adaptive.solve_columns(problem, (1,), 'direct')
        self.assertEqual(solution.shape, (problem.basis.N, 1))
        self.assertEqual(metrics['rhs_count'], 1)
        self.assertEqual(metrics['requested_columns'], [1])

    def test_cheap_path_verifies_initial_and_periodic_states_only(self):
        with patch.object(adaptive, 'online_dwr', side_effect=self.nonconverging_dwr):
            result = self.run_small(marker='primal_residual')
        self.assertEqual(result.status, 'max_steps')
        self.assertEqual([r['verified'] for r in result.rows], [True, False, False, True])
        for row in result.rows[1:3]:
            self.assertIsNone(row['dwr'])
            self.assertFalse(row['stop_candidate'])
            self.assertEqual(row['verification_reasons'], [])
        self.assertEqual(result.summary()['solver_calls'], 6)
        self.assertEqual(result.summary()['rhs_count'], 6)
        self.assertTrue(all(s['metrics']['requested_columns'] == [1]
                            for r in result.rows for s in r['solves'] if s['degree'] == 2))

    def test_dwr_path_verifies_each_state_and_only_one_p2_rhs(self):
        with patch.object(adaptive, 'online_dwr', side_effect=self.nonconverging_dwr):
            result = self.run_small(marker='dwr')
        self.assertEqual(result.status, 'max_steps')
        self.assertTrue(all(r['verified'] for r in result.rows))
        self.assertEqual(result.summary()['solver_calls'], 8)
        self.assertEqual(result.summary()['rhs_count'], 8)
        self.assertTrue(all('dwr_marker' in r['verification_reasons'] for r in result.rows))

    def test_joint_path_shares_p1_setup_and_charges_extra_dual_rhs(self):
        with patch.object(adaptive, 'online_dwr', side_effect=self.nonconverging_dwr):
            result = self.run_small(marker='joint_residual')
        self.assertEqual(result.status, 'max_steps')
        self.assertEqual(result.summary()['solver_calls'], 6)
        self.assertEqual(result.summary()['rhs_count'], 10)
        self.assertTrue(all(r['solves'][0]['metrics']['requested_columns'] == [0, 1]
                            for r in result.rows))

    def test_small_qoi_delta_triggers_paid_check_but_is_not_acceptance(self):
        config = adaptive._config(None)
        self.assertEqual(adaptive.verification_reasons(1, .2, .2, 1e-4, 'uniform', config), ['qoi_delta'])
        self.assertEqual(adaptive.verification_reasons(1, .3, .2, 1e-4, 'uniform', config), [])
        with patch.object(adaptive, 'online_dwr', side_effect=self.nonconverging_dwr):
            result = self.run_small(marker='uniform', config={'max_steps': 2, 'verification_delta_factor': 1e20})
        self.assertEqual(result.status, 'max_steps')
        self.assertEqual(result.rows[1]['verification_reasons'], ['qoi_delta'])
        self.assertFalse(result.rows[1]['stop_candidate'])

    def test_only_current_paid_dwr_check_can_stop_and_no_reference_config_is_accepted(self):
        calls = []
        def verifier(p1, p2, u, z2, *, localize=False):
            answer = self.nonconverging_dwr(p1, p2, u, z2, localize=localize)
            calls.append(p1.mesh.nelements)
            if len(calls) == 2:
                answer['total_estimate'] = answer['discretization_estimate'] = 0.
            return answer
        with patch.object(adaptive, 'online_dwr', side_effect=verifier):
            result = self.run_small(marker='primal_residual', config={'max_steps': 5})
        self.assertEqual(result.status, 'estimated_converged')
        self.assertEqual(len(result.rows), 4)
        self.assertEqual(result.rows[-1]['step'], 3)
        self.assertTrue(result.rows[-1]['verified'])
        self.assertTrue(result.rows[-1]['stop_candidate'])
        with self.assertRaises(ValueError):
            self.run_small(config={'reference_qoi': result.rows[-1]['qoi']})

    def test_linear_failure_preserves_solution_record_and_cannot_pass_zero_estimate(self):
        original = adaptive.solve_columns
        def failing(problem, columns, method='direct'):
            solution, metrics = original(problem, columns, method)
            metrics['accepted'] = metrics['converged'] = False
            return solution, metrics
        with patch.object(adaptive, 'solve_columns', side_effect=failing), \
                patch.object(adaptive, 'online_dwr') as verifier:
            result = self.run_small(marker='dwr', tau=1.)
        self.assertEqual(result.status, 'linear_solver_failed')
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(len(result.rows[0]['solves']), 1)
        self.assertIsNotNone(result.final_solution)
        self.assertIn('solution_sha256', result.rows[0])
        self.assertFalse(result.rows[0]['verified'])
        verifier.assert_not_called()

    def test_enriched_dual_failure_is_retained_and_prevents_acceptance(self):
        original = adaptive.solve_columns
        def failing(problem, columns, method='direct'):
            solution, metrics = original(problem, columns, method)
            if problem.degree == 2:
                metrics['accepted'] = metrics['converged'] = False
            return solution, metrics
        with patch.object(adaptive, 'solve_columns', side_effect=failing), \
                patch.object(adaptive, 'online_dwr') as verifier:
            result = self.run_small(marker='dwr', tau=1.)
        self.assertEqual(result.status, 'linear_solver_failed')
        self.assertEqual(len(result.rows[0]['solves']), 2)
        self.assertFalse(result.rows[0]['verified'])
        verifier.assert_not_called()

    def test_dof_limit_retains_last_solved_mesh_and_pending_refinement(self):
        with patch.object(adaptive, 'online_dwr', side_effect=self.nonconverging_dwr):
            result = self.run_small(marker='uniform', config={'max_free_dofs': 9})
        self.assertEqual(result.status, 'max_dofs')
        self.assertEqual(result.final_mesh.nvertices, 25)
        self.assertEqual(result.rows[-1]['p1_free_dofs'], 9)
        self.assertGreater(result.rows[-1]['next_p1_free_dofs'], 9)
        self.assertTrue(result.rows[-1]['selected_elements'])
        self.assertIsNotNone(result.final_solution)
        self.assertEqual(len(result.rows), 1)

    def test_initial_dof_limit_is_failure_before_any_solve(self):
        with patch.object(adaptive, 'solve_columns') as solver:
            result = self.run_small(config={'max_free_dofs': 1})
        self.assertEqual(result.status, 'max_dofs')
        self.assertIsNone(result.final_solution)
        self.assertEqual(result.summary()['rhs_count'], 0)
        solver.assert_not_called()

    def test_soft_time_limit_records_actual_elapsed_time_without_solving(self):
        clock = count(0., .1)
        with patch.object(adaptive, 'perf_counter', side_effect=lambda: next(clock)), \
                patch.object(adaptive, 'solve_columns') as solver:
            result = self.run_small(config={'max_seconds': .15})
        self.assertEqual(result.status, 'time_limit')
        self.assertGreater(result.algorithm_seconds, .15)
        self.assertFalse(result.summary()['estimated_converged'])
        solver.assert_not_called()

    def test_estimator_exception_is_a_retained_failure(self):
        with patch.object(adaptive, 'online_dwr', side_effect=ValueError('deliberate diagnostic failure')):
            result = self.run_small()
        self.assertEqual(result.status, 'exception')
        self.assertEqual(result.rows[0]['exception']['type'], 'ValueError')
        self.assertIn('deliberate diagnostic failure', result.rows[0]['exception']['message'])
        self.assertEqual(result.summary()['solver_calls'], 2)

    def test_stop_safety_factor_is_applied_to_signed_magnitude(self):
        def verifier(p1, p2, u, z2, *, localize=False):
            answer = self.nonconverging_dwr(p1, p2, u, z2, localize=localize)
            answer['total_estimate'] = answer['discretization_estimate'] = -.75
            return answer
        with patch.object(adaptive, 'online_dwr', side_effect=verifier):
            rejected = self.run_small(tau=1., config={'max_steps': 1, 'safety': .5})
            accepted = self.run_small(tau=1., config={'max_steps': 1, 'safety': 1.})
        self.assertEqual(rejected.status, 'max_steps')
        self.assertEqual(accepted.status, 'estimated_converged')


class MarkingAndInputValidationTests(unittest.TestCase):
    def test_dorfler_minimal_prefix_has_deterministic_ties(self):
        np.testing.assert_array_equal(adaptive.dorfler(np.array([1., 2., 2., 1.]), .5), [1, 2])
        np.testing.assert_array_equal(adaptive.dorfler(np.ones(4), .5), [0, 1])
        np.testing.assert_array_equal(adaptive.dorfler(np.zeros(4), .5), [])
        for theta in (0., -1., 1.1, np.nan):
            with self.assertRaises(ValueError):
                adaptive.dorfler(np.ones(4), theta)
        for scores in ([1., -1.], [1., np.nan], [], [[1., 2.]]):
            with self.assertRaises(ValueError):
                adaptive.dorfler(scores)

    def test_joint_residual_is_scale_invariant_and_handles_zero_terms(self):
        first = np.array([1., 3., 0.])
        second = np.array([0., 1., 2.])
        expected = np.array([.25, .75 + 1./3., 2./3.])
        np.testing.assert_allclose(adaptive.joint_scores(first, second), expected)
        np.testing.assert_allclose(adaptive.joint_scores(first * 1000., second * .01), expected)
        np.testing.assert_allclose(adaptive.joint_scores(first, np.zeros(3)), first / 4.)
        np.testing.assert_allclose(adaptive.joint_scores(np.zeros(3), np.zeros(3)), 0.)

    def test_invalid_resources_and_unknown_inputs_are_rejected(self):
        for config in ({'max_steps': 0}, {'max_steps': True}, {'max_free_dofs': -1},
                       {'verification_interval': 0}, {'safety': 1.1}, {'max_seconds': np.inf},
                       {'solver': 'unknown'}, {'reference': 0.}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                adaptive._config(config)
        for tau in (0., -1., np.inf, np.nan):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                adaptive.run_adaptive(constant(1.), constant(1.), constant(1.),
                                      initial_mesh=fem.make_mesh(4), tau=tau)


class ApplicationAndReferenceTests(unittest.TestCase):
    def synthetic_reference(self, qoi=None):
        config = dict(reference.REFERENCE_CONFIG)
        qoi = [0.999, 0.9999, 0.99999, 0.999999] if qoi is None else qoi
        rows = [dict(n=n, degree=2, intorder=config['intorder'], qoi=q, accepted=True)
                for n, q in zip(config['p2_n'], qoi)]
        rows.append(dict(n=max(config['p2_n']), degree=2, intorder=config['high_order'],
                         qoi=qoi[-1] + 1e-7, accepted=True))
        rows.append(dict(n=config['p1_cross_n'], degree=1, intorder=config['intorder'],
                         qoi=qoi[-1] - 1e-6, accepted=True))
        return rows, config

    def test_application_source_and_goal_are_normalized_and_fixed_across_materials(self):
        inputs = applications.cases()
        self.assertEqual(len(inputs), 6)
        nodes, weights = leggauss(64)
        for function, center in ((inputs[0].source, (.25, .5)), (inputs[0].goal, (.75, .5))):
            x, y = center[0] + .18*nodes[:, None], center[1] + .18*nodes[None, :]
            mass = float(.18**2 * np.einsum('i,ij,j->', weights, function(x, y), weights))
            self.assertAlmostEqual(mass, 1., delta=2e-13)
        self.assertTrue(all(case.source is inputs[0].source and case.goal is inputs[0].goal for case in inputs))
        self.assertTrue(all(not case.metadata['manufactured'] for case in inputs))
        self.assertTrue(all(not hasattr(case, 'exact') and not hasattr(case, 'reference_qoi') for case in inputs))

    def test_all_application_interfaces_remain_aligned_after_local_refinement(self):
        mesh = fem.make_mesh(8)
        refined = mesh.refined(np.array([1, 8, 17, 39, 65, 101]))
        for case in applications.cases():
            with self.subTest(case=case.name):
                self.assertTrue(applications.mesh_audit(mesh, case)['accepted'])
                self.assertTrue(applications.mesh_audit(refined, case)['accepted'])
                midpoint = refined.p[:, refined.t].mean(axis=1)
                k = case.coefficient(midpoint[0], midpoint[1])
                self.assertGreaterEqual(float(k.min()), case.metadata['k_min'])
                self.assertLessEqual(float(k.max()), case.metadata['k_max'])
        barrier = next(case for case in applications.cases() if case.name == 'barrier_1000')
        self.assertFalse(applications.mesh_audit(fem.make_mesh(6), barrier)['accepted'])

    def test_stable_reference_passes_only_as_empirical_check(self):
        rows, config = self.synthetic_reference()
        result = reference.assess_reference(rows, config)
        self.assertTrue(result['accepted'])
        self.assertTrue(all(result['checks'].values()))
        self.assertTrue(result['no_continuous_error_bound'])
        self.assertEqual(result['qoi'], rows[-2]['qoi'])
        np.testing.assert_allclose(result['signed_hierarchy_differences'], [9e-4, 9e-5, 9e-6], rtol=2e-11)

    def test_small_final_difference_does_not_hide_sign_cancellation(self):
        rows, config = self.synthetic_reference([.999, .9999, .99989, .999891])
        result = reference.assess_reference(rows, config)
        self.assertTrue(result['checks']['finest_difference'])
        self.assertTrue(result['checks']['differences_not_growing'])
        self.assertFalse(result['checks']['stable_difference_sign'])
        self.assertFalse(result['accepted'])

    def test_small_final_difference_does_not_hide_growing_sequence(self):
        rows, config = self.synthetic_reference([.999, .9999, .999909, .9999195])
        result = reference.assess_reference(rows, config)
        self.assertTrue(result['checks']['finest_difference'])
        self.assertTrue(result['checks']['stable_difference_sign'])
        self.assertFalse(result['checks']['differences_not_growing'])
        self.assertFalse(result['accepted'])

    def test_reference_failure_quadrature_and_cross_order_disagreement_are_retained(self):
        rows, config = self.synthetic_reference()
        for change, key in ((lambda data: data[1].update(accepted=False), 'all_linear_and_mesh_checks'),
                            (lambda data: data[-2].update(qoi=1.02), 'quadrature'),
                            (lambda data: data[-1].update(qoi=.98), 'independent_order')):
            data = deepcopy(rows)
            change(data)
            result = reference.assess_reference(data, config)
            self.assertFalse(result['checks'][key])
            self.assertFalse(result['accepted'])

    def test_incomplete_reference_hierarchy_is_rejected(self):
        rows, config = self.synthetic_reference()
        wrong_order = deepcopy(rows)
        wrong_order[-1]['intorder'] = config['high_order']
        for data in (rows[1:], rows[:-1], rows[:-2] + rows[-1:], rows + [rows[0]], wrong_order):
            with self.assertRaises(ValueError):
                reference.assess_reference(data, config)

    def test_finite_precision_linear_acceptance_uses_actual_thresholds(self):
        metrics = dict(accepted=True, converged=True, all_info_zero=True,
                       relative_residual=1e-9, componentwise_backward_error=1e-12,
                       boundary_maximum=0.)
        self.assertTrue(adaptive.accepted(metrics))
        for key in ('relative_residual', 'componentwise_backward_error'):
            altered = dict(metrics)
            altered[key] = np.nextafter(altered[key], np.inf)
            self.assertFalse(adaptive.accepted(altered))
            altered[key] = np.nan
            self.assertFalse(adaptive.accepted(altered))


class OfflineAcceptanceTests(unittest.TestCase):
    def make_reference(self, qoi=1., accepted=True):
        return dict(assessment=dict(accepted=accepted, qoi=qoi))

    def test_online_estimate_stop_but_failed_reference_accuracy_stays_failure(self):
        summary = dict(status='estimated_converged', final_qoi=.9, algorithm_seconds=.01)
        result = experiment.assess_output(summary, self.make_reference(), .01)
        self.assertTrue(result['online_completed'])
        self.assertEqual(result['status'], 'accuracy_failed')
        self.assertFalse(result['success'])
        self.assertGreater(result['q'], 1.)

    def test_actual_threshold_has_no_hidden_accuracy_slack(self):
        reference_value = self.make_reference(qoi=0.)
        for qoi, expected in ((.01, True), (np.nextafter(.01, np.inf), False), (-.01, True)):
            result = experiment.assess_output(dict(status='estimated_converged', final_qoi=qoi), reference_value, .01)
            self.assertEqual(result['success'], expected)
        self.assertFalse(result['continuous_error_bound'])

    def test_resource_failures_do_not_become_success_with_accidentally_good_qoi(self):
        for status in ('max_steps', 'max_dofs', 'time_limit', 'linear_solver_failed', 'exception', 'zero_indicator'):
            result = experiment.assess_output(dict(status=status, final_qoi=1.), self.make_reference(), .01)
            self.assertEqual(result['status'], status)
            self.assertFalse(result['success'])
            self.assertFalse(result['online_completed'])
            self.assertEqual(result['q'], 0.)

    def test_unqualified_reference_blocks_claim_even_when_values_match(self):
        result = experiment.assess_output(dict(status='estimated_converged', final_qoi=1.),
                                          self.make_reference(accepted=False), .01)
        self.assertFalse(result['reference_qualified'])
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'reference_unqualified')

    def test_missing_or_nonfinite_qoi_is_explicit_accuracy_failure(self):
        for value in (None, np.nan, np.inf, -np.inf):
            result = experiment.assess_output(dict(status='estimated_converged', final_qoi=value),
                                              self.make_reference(), .01)
            self.assertFalse(result['success'])
            self.assertEqual(result['status'], 'accuracy_failed')
            self.assertIsNone(result['q'])
            self.assertIsNone(result['absolute_qoi_error'])

    def test_changing_offline_reference_does_not_mutate_online_summary(self):
        summary = dict(status='estimated_converged', final_qoi=.999, algorithm_seconds=1.23)
        original = deepcopy(summary)
        passing = experiment.assess_output(summary, self.make_reference(1.), .01)
        failing = experiment.assess_output(summary, self.make_reference(2.), .01)
        self.assertTrue(passing['success'])
        self.assertFalse(failing['success'])
        self.assertEqual(summary, original)

    def test_fixed_mesh_completion_has_same_offline_accuracy_requirement(self):
        for value, success in ((.999, True), (.9, False)):
            result = experiment.assess_output(dict(status='fixed_mesh_completed', final_qoi=value),
                                              self.make_reference(), .01)
            self.assertEqual(result['success'], success)
            self.assertTrue(result['acceptance_is_relative_to_empirically_checked_reference'])
            self.assertFalse(result['continuous_error_bound'])

    def test_missing_reference_assessment_cannot_silently_pass(self):
        with self.assertRaises(KeyError):
            experiment.assess_output(dict(status='estimated_converged', final_qoi=1.), {}, .01)


if __name__ == '__main__':
    unittest.main()
