"""Small-grid stage-8 wiring, independent residual and stopping checks.

No formal split/result files are created. The one synthetic geometry is
the explicitly excluded seed 75004. Reference preparation is completed
before any of the five measured-method smoke paths.
"""
import contextlib
import hashlib
import inspect
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import oracle_budget_experiment as experiment


class OracleBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields, cls.events = experiment.coefficient_sequence(32, 20, 75004,
                                                               experiment.CONTRAST, 'mixed')
        cls.results = {}
        with experiment.threadpool_limits(limits=1, user_api='blas'):
            cls.directions, cls.truth, cls.quality = experiment.prepare_sequence(cls.fields, cls.events)
            cls.systems = [experiment.assemble_system(k)[:2] for k in cls.fields]
            for setting in experiment.specs():
                state = dict(base_operators=[], balanced_operators={}, calls=[])
                real_matrix = experiment.AMGSequencePreconditioner.for_matrix
                real_balanced = experiment.balanced_preconditioner
                real_cached = experiment.solve_pcg_cached
                real_scipy = experiment.solve_pcg

                def matrix_spy(instance, A, *args, **kwargs):
                    M, info = real_matrix(instance, A, *args, **kwargs)
                    state['base_operators'].append(M)
                    return M, info

                def balanced_spy(M, U, AU):
                    B = real_balanced(M, U, AU)
                    state['balanced_operators'][len(state['base_operators'])-1] = B
                    return B

                def solver_spy(real, kind, *args, **kwargs):
                    bound = inspect.signature(real).bind(*args, **kwargs)
                    bound.apply_defaults()
                    state['calls'].append(dict(kind=kind, operator=bound.arguments['M'],
                        max_vectors=bound.arguments.get('max_vectors', 0)))
                    return real(*args, **kwargs)

                def cached_spy(*args, **kwargs):
                    return solver_spy(real_cached, 'cached', *args, **kwargs)

                def scipy_spy(*args, **kwargs):
                    return solver_spy(real_scipy, 'scipy', *args, **kwargs)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if setting['policy'] == 'current_reference':
                        stack.enter_context(patch.object(experiment.AMGSequencePreconditioner,
                            'for_matrix', autospec=True, side_effect=matrix_spy))
                        stack.enter_context(patch.object(experiment, 'balanced_preconditioner', side_effect=balanced_spy))
                        stack.enter_context(patch.object(experiment, 'solve_pcg_cached', side_effect=cached_spy))
                        stack.enter_context(patch.object(experiment, 'solve_pcg', side_effect=scipy_spy))
                        stack.enter_context(patch.object(experiment, 'cached_slow_ritz',
                            side_effect=AssertionError('Reference extraction entered timed path')))
                        stack.enter_context(patch.object(experiment.sla, 'spsolve',
                            side_effect=AssertionError('Direct truth entered timed path')))
                        rows, xs = experiment.run_reference_sequence(cls.fields, cls.events,
                            setting, 0, cls.directions[setting['rank']])
                    else:
                        rows, xs = experiment.previous.run_sequence(cls.fields, cls.events, setting, 0)
                cls.results[setting['name']] = dict(rows=rows, solutions=xs, **state)

    def test_all_five_methods_complete_and_independent_true_residuals_pass(self):
        self.assertEqual(len(self.results), 5)
        for name, result in self.results.items():
            with self.subTest(method=name):
                self.assertEqual(len(result['rows']), 20)
                self.assertTrue(all(r['converged'] and r['final_converged']
                                    and not r['solver_fallback'] for r in result['rows']))
                # This fixed small geometry converges in five base iterations
                # at step 2, leaving only four paid columns. q8 must fail the
                # structural gate rather than being silently relabelled q4.
                self.assertEqual(experiment.rows_valid(result['rows']), name != 'current_reference8')
                for (A, b), x in zip(self.systems, result['solutions']):
                    self.assertLessEqual(float(experiment.la.norm(b-A@x)/experiment.la.norm(b)),
                                         experiment.RTOL)

    def test_every_reference_step_passes_the_actual_balanced_operator(self):
        event_steps = {0, 3, 6, 9, 10, 12, 15, 18}
        for q in experiment.RANKS:
            result = self.results[f'current_reference{q}']
            self.assertEqual(len(result['base_operators']), 20)
            expected = {i for i, r in enumerate(result['rows']) if r['reference_space_valid']}
            self.assertEqual(set(result['balanced_operators']), expected)
            self.assertEqual(len(result['calls']), 20)
            for step, call in enumerate(result['calls']):
                with self.subTest(q=q, step=step):
                    if step in expected:
                        self.assertIs(call['operator'], result['balanced_operators'][step])
                        self.assertIsNot(call['operator'], result['base_operators'][step])
                    else:
                        self.assertIs(call['operator'], result['base_operators'][step])
                        self.assertTrue(result['rows'][step]['cache_rejected'])
                    self.assertEqual(call['kind'], 'cached' if step in event_steps else 'scipy')
                    self.assertEqual(call['max_vectors'], 0)
                    row = result['rows'][step]
                    self.assertEqual(row['rank'] == q, step in expected)
                    self.assertFalse(row['capture_requested'])
                    self.assertFalse(row['extraction_requested'])
                    self.assertEqual(row['extraction_seconds'], 0.)
                    self.assertEqual(row['capture_stats'].get('captured_vectors', 0), 0)

    def test_preparation_uses_fixed_base_cache_and_independent_full_space_checks(self):
        self.assertEqual(len(self.quality), 20)
        for step, row in enumerate(self.quality):
            with self.subTest(step=step):
                self.assertEqual(row['numerically_valid'], step != 2)
                self.assertEqual(row['capture_stats']['max_vectors'], 64)
                self.assertLessEqual(row['captured_vectors'], 64)
                self.assertEqual(len(row['capture_stats']['segment_ids']), row['captured_vectors'])
                self.assertEqual(row['segment_count'], len(set(row['capture_stats']['segment_ids'])))
                self.assertEqual(row['cache_audit'] is not None, step in (0, 10))
                if row['cache_audit']:
                    self.assertTrue(row['cache_audit']['valid'])
                for q in experiment.RANKS:
                    value = row['quality'][str(q)]
                    self.assertEqual(value['valid'], not (q == 8 and step == 2))
                    self.assertEqual(value['rank'], 4 if step == 2 else q)
                    self.assertEqual(len(value['a_relative_ritz_residuals']), value['rank'])
                    self.assertLessEqual(value['orthogonality_error'], 1e-7)
                    self.assertTrue(all(v > 0 for v in value['ritz_values']))
                    self.assertLessEqual(value['projected_energy_remaining_ratio'], 1.+1e-7)
                self.assertEqual(row['frozen4_projection']['rank'], 4)

    def test_reference_arrays_remain_read_only_and_unchanged(self):
        for q in experiment.RANKS:
            for step, U in enumerate(self.directions[q]):
                self.assertFalse(U.flags.writeable)
                self.assertEqual(experiment.array_sha(U), self.quality[step]['space_sha256'][str(q)])
                self.assertEqual(experiment.array_sha(U),
                                 self.results[f'current_reference{q}']['rows'][step]['reference_space_sha256'])

    def test_archive_loader_preserves_read_only_arrays_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'reference.npz'
            arrays = {f'U{q}_{step:02d}': self.directions[q][step]
                      for q in experiment.RANKS for step in range(20)}
            arrays.update({f'truth_{step:02d}': x for step, x in enumerate(self.truth)})
            experiment.np.savez(path, **arrays)
            metadata = dict(sequences=[dict(n=32, seed=75004, archive=path.name,
                archive_sha256=experiment.sha(path), systems=self.quality,
                field_sha256=hashlib.sha256(self.fields.tobytes()).hexdigest())])
            with patch.object(experiment, 'ROOT', root):
                loaded = experiment.load_group(metadata)[75004]
                for q in experiment.RANKS:
                    for actual, expected in zip(loaded['directions'][q], self.directions[q]):
                        self.assertFalse(actual.flags.writeable)
                        experiment.np.testing.assert_array_equal(actual, expected)
                with path.open('ab') as handle:
                    handle.write(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'archive changed'):
                    experiment.load_group(metadata)

    def test_cost_components_are_inside_total_and_extraction_not_added_twice(self):
        components = ('assembly_seconds', 'guess_seconds', 'setup_seconds',
                      'correction_seconds', 'solve_seconds', 'recovery_seconds', 'maintenance_seconds')
        for result in self.results.values():
            for row in result['rows']:
                self.assertLessEqual(sum(row[k] for k in components), row['total_seconds']+1e-9)
                self.assertLessEqual(row['extraction_seconds'], row['maintenance_seconds']+1e-9)

    @staticmethod
    def fake_data(q4=(.7, .7), q8=(.7, .7), valid=True):
        return {str(n): dict(numerically_valid=valid, summary={
            'baseline_scipy': dict(by_seed={'1': 1., '2': 1.}),
            'current_reference4': dict(by_seed={'1': q4[index], '2': q4[index]}),
            'current_reference8': dict(by_seed={'1': q8[index], '2': q8[index]})})
            for index, n in enumerate((128, 256))}

    def test_gate_never_switches_rank_between_grids_or_after_invalid_run(self):
        split_winner = experiment.evaluate_gate(self.fake_data(q4=(.7, .9), q8=(.9, .7)))
        self.assertFalse(split_winner['proceed_to_confirmation'])
        self.assertIsNone(split_winner['selected_rank'])
        self.assertFalse(experiment.evaluate_gate(self.fake_data(valid=False))['proceed_to_confirmation'])
        all_pass = experiment.evaluate_gate(self.fake_data())
        self.assertTrue(all_pass['proceed_to_confirmation'])
        self.assertEqual(all_pass['selected_rank'], 4)

    def test_gate_checks_every_geometry_and_zero_budget_is_not_training(self):
        data = self.fake_data(q4=(.8, .8), q8=(.9, .9))
        result = experiment.evaluate_gate(data)
        self.assertTrue(result['proceed_to_confirmation'])
        self.assertFalse(result['training_authorized'])
        self.assertTrue(all(c['construction_budget_seconds'] == 0.
                            for c in result['candidates']['4']['cases']))
        data['128']['summary']['current_reference4']['by_seed']['2'] = .81
        self.assertFalse(experiment.evaluate_gate(data)['proceed_to_confirmation'])

    def test_rank_deficiency_and_nonfinite_quality_are_not_promoted_to_success(self):
        A, _ = self.systems[0]
        U = self.directions[4][0]
        info = self.quality[0]['extraction']['4']['stats']
        M = experiment.sla.aslinearoperator(experiment.np.eye(A.shape[0]))
        short = experiment.space_quality(A, M, U, info, 8)
        self.assertFalse(short['valid'])
        poisoned = dict(info, ritz_values=[float('nan')]*4)
        bad = experiment.space_quality(A, M, U, poisoned, 4)
        self.assertFalse(bad['valid'])
        self.assertEqual(bad['ritz_values'], [None]*4)
        self.assertIsNone(bad['max_a_relative_ritz_residual'])

    def test_failed_source_never_recovers_history_from_the_direct_reference(self):
        real_solver, real_initial = experiment.solve_pcg_cached, experiment.initial
        histories, calls = [], []

        def solver_spy(*args, **kwargs):
            x, metrics, cache = real_solver(*args, **kwargs)
            calls.append(1)
            if len(calls) == 1:
                # Keep a known failed iterate. The independent direct truth
                # is nonzero and must not replace this history entry.
                x = experiment.np.zeros_like(x)
                metrics = dict(metrics, converged=False)
            return x, metrics, cache

        def initial_spy(A, b, history):
            histories.append([x.copy() for x in history])
            return real_initial(A, b, history)

        with experiment.threadpool_limits(limits=1, user_api='blas'):
            with patch.object(experiment, 'solve_pcg_cached', side_effect=solver_spy), \
                 patch.object(experiment, 'initial', side_effect=initial_spy):
                _, truth, quality = experiment.prepare_sequence(self.fields[:2], self.events[:2])
        self.assertFalse(quality[0]['numerically_valid'])
        self.assertTrue(quality[0]['source_failed'])
        self.assertFalse(quality[0]['source_solver_fallback'])
        self.assertEqual(len(histories[1]), 1)
        self.assertFalse(histories[1][0].any())
        self.assertTrue(truth[0].any())


if __name__ == '__main__':
    unittest.main()
