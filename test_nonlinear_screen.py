"""Selection and snapshot tests using synthetic data, without PDE solves.

The only test that writes artifacts replaces both ROOT and OUT with a
TemporaryDirectory. Numerical solves and problem construction are mocked.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import nonlinear_screen_experiment as experiment


def baseline_records():
    """EW1 wins globally, while fixed_0.1 wins most fine-grid cases.

    The nominally fastest direct method fails one fine-grid repeat and
    must be ineligible. These deliberate conflicts expose both per-case
    baseline selection and selection using only the branch grid.
    """
    rows = []
    for n in (32, 64):
        for case in range(12):
            for method in experiment.METHODS:
                seconds = 14.
                if method == 'ew1':
                    seconds = 8.
                elif method == 'ew2':
                    seconds = 10.
                elif method == 'fixed_0.1':
                    seconds = 11. if n == 32 else (5. if case < 9 else 12.)
                elif method == 'direct':
                    seconds = .1
                for repeat, delta in enumerate((-.02, 0., .02)):
                    value = seconds + delta
                    rows.append(dict(n=n, case=case, method=method, repeat=repeat,
                        total_seconds=value, accurate=not (method == 'direct' and
                            n == 64 and case == 11 and repeat == 2),
                        newton_iterations=4, linear_iterations=20,
                        components={'residual': value/10.},
                        assembly_seconds=value/100.))
    return rows


def branch_records(step0_seconds=7., deep_variant=None):
    rows = []
    iterations = (0,) if deep_variant is None else (0, 1, 2, 3)
    for case in range(12):
        for iteration in iterations:
            for multiplier in (.5, 1.):
                if iteration == 0:
                    # The outlier must not become a candidate time; the
                    # median must be taken before choosing an action.
                    values = ((step0_seconds-1., step0_seconds, 100.)
                              if multiplier == .5 else (7., 8., 9.))
                elif deep_variant == 'large_gain':
                    values = (1e-6, 1e-6, 1e-6) if multiplier == .5 else (100.,)*3
                else:
                    values = (2e6,)*3 if multiplier == .5 else (1e6,)*3
                for repeat, value in enumerate(values):
                    rows.append(dict(case=case, iteration=iteration, repeat=repeat,
                        multiplier=multiplier, total_seconds=value, accurate=True,
                        expected_eta=.25 if multiplier == .5 else .5,
                        newton_iterations=4, linear_iterations=20))
    return rows


def preparation_record():
    return {'sequences': [{'case': case, 'source_accurate': True}
                           for case in range(12)]}


class NonlinearScreenTests(unittest.TestCase):
    def test_one_primary_uses_both_grids_and_remains_fixed_for_step0_pairs(self):
        rows = baseline_records()
        selection = experiment.aggregate_baselines(rows)
        self.assertEqual(selection['primary'], 'ew1')
        self.assertEqual(selection['continuation'], 'ew1')
        self.assertFalse(selection['methods']['direct']['all_valid'])
        self.assertEqual(len(selection['methods']['ew1']['points']), 24)

        fine_only = experiment.aggregate_baselines([r for r in rows if r['n'] == 64])
        self.assertEqual(fine_only['primary'], 'fixed_0.1',
                         'Fixture must distinguish global from fine-grid selection')
        decision = experiment.assess(branch_records(), selection, preparation_record())
        self.assertTrue(decision['numerically_valid'])
        self.assertEqual({p['case'] for p in decision['step0_pairs']}, set(range(12)))
        for pair in decision['step0_pairs']:
            self.assertEqual(pair['baseline_seconds'], 8.)
            self.assertEqual(pair['best_action']['seconds'], 7.)
            self.assertAlmostEqual(pair['ratio'], 7./8.)
        self.assertAlmostEqual(decision['step0_median_ratio'], .875)
        self.assertFalse(decision['opportunity_gate_passed'])

    def test_deeper_costs_cannot_change_step0_gate_but_failure_still_vetoes(self):
        selection = experiment.aggregate_baselines(baseline_records())
        gain_rows = branch_records(step0_seconds=6., deep_variant='large_gain')
        slow_rows = branch_records(step0_seconds=6., deep_variant='no_gain')
        gain = experiment.assess(gain_rows, selection, preparation_record())
        slow = experiment.assess(slow_rows, selection, preparation_record())
        self.assertTrue(gain['opportunity_gate_passed'])
        self.assertAlmostEqual(gain['step0_median_ratio'], .75)
        for key in ('step0_pairs', 'step0_median_ratio', 'step0_faster_instances',
                    'opportunity_gate_passed', 'status', 'numerically_valid'):
            self.assertEqual(gain[key], slow[key], key)
        gain_deep = next(s for s in gain['states'] if s['iteration'] == 1)
        slow_deep = next(s for s in slow['states'] if s['iteration'] == 1)
        self.assertNotEqual(gain_deep['remaining_ratio_to_continuation'],
                            slow_deep['remaining_ratio_to_continuation'])

        failed_rows = deepcopy(gain_rows)
        next(r for r in failed_rows if r['iteration'] == 2)['accurate'] = False
        failed = experiment.assess(failed_rows, selection, preparation_record())
        self.assertEqual(failed['step0_median_ratio'], gain['step0_median_ratio'])
        self.assertFalse(failed['numerically_valid'])
        self.assertFalse(failed['opportunity_gate_passed'],
                         'The protocol requires accuracy for every measured branch')

    def test_early_convergence_terminal_snapshot_is_counted_as_missing(self):
        truth = np.ones(4, dtype=float)
        snapshots = []
        for iteration in (0, 1, 2):
            snapshots.append(dict(u=np.full(4, iteration/2.), iteration=iteration,
                F0norm=2., prevnorm=None if iteration == 0 else 1.,
                prevlinearizednorm=None if iteration == 0 else .1,
                preveta=None if iteration == 0 else .5,
                native_eta=.5 if iteration == 0 else .2,
                converged=iteration == 2, absolute_residual=0. if iteration == 2 else 1.,
                residual_norm=0. if iteration == 2 else 1.,
                problem_signature='synthetic', method='ew1'))
        source = dict(u=truth.copy(), snapshots=snapshots, converged=True,
            status='converged', absolute_residual=0., tolerance=1e-8,
            trace=[{'iteration': 0, 'native_eta': .5},
                   {'iteration': 1, 'native_eta': .2}])
        problem = SimpleNamespace(assembly_seconds=.01)
        with TemporaryDirectory(prefix='stage9-screen-test-') as directory:
            root = Path(directory)
            (root/'data').mkdir()
            out = root/'results'/'stage9_screen'
            with patch.object(experiment, 'ROOT', root), \
                    patch.object(experiment, 'OUT', out), \
                    patch.object(experiment, 'make_problem', return_value=problem) as make, \
                    patch.object(experiment, 'solve', return_value=source) as solve:
                states, record = experiment.prepare_snapshots(
                    [{'case': 0, 'params': {}}], 'ew1', {(64, 0): truth})
            make.assert_called_once_with(64, {})
            solve.assert_called_once_with(problem, 'ew1', capture=True)
            self.assertEqual([s['state']['iteration'] for s in states], [0, 1])
            self.assertEqual(record['snapshot_count'], 2)
            sequence = record['sequences'][0]
            self.assertTrue(sequence['source_accurate'])
            self.assertEqual(sequence['missing_iterations'], [2, 3])
            self.assertEqual(len(list((root/'data').glob('*.npz'))), 2)
            self.assertFalse((root/'data'/'stage9_case00_step2.npz').exists())
            self.assertTrue((out/'snapshot_preparation.json').is_file())
            for item in sequence['snapshots']:
                self.assertTrue((root/item['archive']).is_file())
                self.assertEqual(experiment.sha(root/item['archive']), item['archive_sha256'])

    def test_clipped_actions_are_unique_and_keep_native_representative(self):
        expected = {
            1e-10: {1e-10, 2e-10, 4e-10},
            .5: {.125, .25, .5, .9},
            .9: {.225, .45, .9},
        }
        for native, etas in expected.items():
            with self.subTest(native=native):
                actions = experiment.unique_actions(native)
                self.assertEqual({a['eta'] for a in actions}, etas)
                self.assertEqual(len(actions), len(etas))
                controls = [a for a in actions if a['multiplier'] == 1.]
                self.assertEqual(len(controls), 1)
                self.assertEqual(controls[0]['eta'], native)
                self.assertEqual(next(a for a in actions if a['eta'] == native)['multiplier'], 1.)


if __name__ == '__main__':
    unittest.main()
