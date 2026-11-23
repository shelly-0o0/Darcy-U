"""Detached ledger tests for the post-run verifier; no PDE is assembled."""
from copy import deepcopy
import hashlib
import unittest

import numpy as np

import stage15c_experiment as experiment
import stage15c_cases as cases
import report_stage15c as report
import verify_stage15c as verify
import test_stage15c as screening_fixtures


class VerifierContractTests(unittest.TestCase):
    def fixture(self):
        """Deliberately synthetic recovery-only resource-stop ledger."""
        metrics = dict(accepted=True, converged=True, all_info_zero=True,
            requested_columns=[0, 1], rhs_count=2, boundary_maximum=0.,
            per_rhs=[dict(relative_residual=1e-14, componentwise_backward_error=1e-16)]*2)
        recovery = dict(steps=3, matvec_count=3, diagonal_extractions=1, goal_rhs_index=1,
            omega=2./3., unverified_marker=True, converged_solve_claim=False,
            no_reference_or_true_p2_adjoint_used=True, boundary_maximum=0.,
            source='accepted_P1_adjoint_lift_then_fixed_damped_Jacobi',
            timings=dict(total_seconds=.1, matvec_seconds=.05))
        row = dict(status='max_dofs', verified=False, dwr=None, recovery=recovery,
            elements=2, recovered_local=[1., -2.], scores=[1., 2.],
            p2_assembly_usage=['recovery'], solves=[dict(degree=1, metrics=metrics)],
            timings=dict(recovery_seconds=.11, p2_assembly_seconds=.03))
        summary = dict(status='max_dofs', final_qoi=.05, config={})
        payload = dict(p=np.zeros((2, 3)), t=np.zeros((3, 2), dtype=int),
                       u=np.zeros(3), sort_t=True)
        hashes = [verify.ahash(payload[key]) for key in ('p', 't', 'u')]
        content = hashlib.sha256(('|'.join(hashes)+'True').encode()).hexdigest()
        entry = dict(case='synthetic', family='synthetic', tau=.1, method='recovered_dwr3', repeat=0,
            summary=deepcopy(summary), final_mesh_check=dict(accepted=True),
            offline=dict(success=False, q=.5), payload_file='payloads/'+content+'.npz')
        record = dict(summary=summary, rows=[row])
        freeze = dict(config=dict(methods=['recovered_dwr3'], taus=[.1], repeats=1,
                                  reference={}, adaptive={}),
                      cases=[dict(name='synthetic', metadata=dict(family='synthetic'))])
        references = [dict(case='synthetic', config={}, assessment=dict(accepted=True, qoi=0.))]
        return freeze, references, [(entry, record, payload)], dict(paths=1)

    def test_consistent_recovery_resource_failure_is_valid_but_never_successful(self):
        fixture = self.fixture()
        verify.require(verify.contract_checks(*fixture))
        self.assertFalse(fixture[2][0][0]['offline']['success'])

    def test_actual_matvec_contract_corruption_is_rejected(self):
        fixture = self.fixture()
        fixture[2][0][1]['rows'][0]['recovery']['matvec_count'] = 4
        with self.assertRaisesRegex(ValueError, 'recovery_steps'):
            verify.require(verify.contract_checks(*fixture))

    def test_removing_the_shared_p1_goal_column_is_rejected(self):
        fixture = self.fixture()
        fixture[2][0][1]['rows'][0]['solves'][0]['metrics']['requested_columns'] = [0]
        with self.assertRaisesRegex(ValueError, 'p1_columns'):
            verify.require(verify.contract_checks(*fixture))

    def test_noncertified_recovery_cannot_be_forged_into_success(self):
        fixture = self.fixture()
        entry, record, _ = fixture[2][0]
        record['rows'][0]['status'] = record['summary']['status'] = 'estimated_converged'
        entry['summary'] = deepcopy(record['summary']); entry['offline']['success'] = True
        with self.assertRaisesRegex(ValueError, 'deployable_stop'):
            verify.require(verify.contract_checks(*fixture))

    def test_content_address_rejects_changed_payload_even_without_pde_resolve(self):
        fixture = self.fixture()
        fixture[2][0][2]['u'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'content_addressed_payload'):
            verify.require(verify.contract_checks(*fixture))


class RawScreenTests(unittest.TestCase):
    def cohort(self):
        freeze, pairs = screening_fixtures.ScreeningContractTests().cohort()
        references = [dict(case=case['name'], assessment=dict(accepted=True, qoi=0.)) for case in freeze['cases']]
        paths = []
        for pair in pairs:
            for method in pair['methods']:
                dwr = method['method'] == 'dwr'
                for repeat in range(3):
                    entry = dict(case=pair['case'], tau=pair['tau'], method=method['method'], repeat=repeat,
                                 final_mesh_check=dict(accepted=True))
                    summary = dict(status='fixed_mesh_completed' if method['method']=='fixed_direct' else 'estimated_converged',
                                   final_qoi=0., algorithm_seconds=method['median_seconds'])
                    rows = [dict(status='continue', timings=dict(p2_assembly_seconds=4.))] if dwr else []
                    rows += [dict(status=summary['status'], p1_free_dofs=method['median_final_p1_dofs'], timings={})]
                    paths.append((entry, dict(summary=summary, rows=rows), {}))
        return freeze, pairs, references, paths

    def test_raw_recomputation_matches_report_gate_on_detached_cohort(self):
        freeze, pairs, references, paths = self.cohort()
        independent = verify.independent_screen(freeze, references, paths)
        verify.require(verify.compare_nested(independent, report.screen_opportunity(freeze, pairs)))

    def test_missing_tau_is_rejected_before_any_family_can_be_retained(self):
        freeze, _, references, paths = self.cohort()
        paths = [item for item in paths if item[0]['tau'] == freeze['config']['taus'][0]]
        with self.assertRaisesRegex(ValueError, 'entire frozen repeat matrix'):
            verify.independent_screen(freeze, references, paths)

    def test_hypothetical_time_is_median_of_differences_not_difference_of_medians(self):
        freeze, _, references, paths = self.cohort()
        subset = [item for item in paths if item[0]['case'] == freeze['cases'][0]['name']
                  and item[0]['tau'] == freeze['config']['taus'][0] and item[0]['method'] == 'dwr']
        for item, total, paid in zip(subset, [10., 20., 12.], [9., 1., 10.]):
            item[1]['summary']['algorithm_seconds'] = total
            item[1]['rows'][0]['timings']['p2_assembly_seconds'] = paid
        first = verify.independent_screen(freeze, references, paths)['settings'][0]
        self.assertEqual(first['hypothetical_remaining_over_control'], .2)
        # median(total-paid)=median(1,19,2)=2, whereas the wrong order is 3.
        self.assertNotEqual(first['hypothetical_remaining_over_control'], (12.-9.)/10.)

    def test_reference_failure_cannot_be_hidden_in_a_fast_screen(self):
        freeze, _, references, paths = self.cohort()
        references[0]['assessment']['accepted'] = False
        result = verify.independent_screen(freeze, references, paths)
        self.assertFalse(result['geometries'][0]['passed'])
        self.assertIsNone(result['settings'][0]['dwr_time_over_control'])


class SamplingVerificationTests(unittest.TestCase):
    def test_new_geometry_sampling_matches_independently_built_pools(self):
        freeze = dict(config=deepcopy(experiment.CONFIG),
                      cases=[dict(name=case.name, metadata=case.metadata) for case in cases.cases()])
        verify.require(verify.geometry_checks(freeze))
        freeze['cases'][0]['metadata']['geometry_eighths'][0] = 7
        with self.assertRaisesRegex(ValueError, 'sample'):
            verify.require(verify.geometry_checks(freeze))


if __name__ == '__main__':
    unittest.main()
