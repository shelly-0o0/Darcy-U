"""Post-run Stage15C audit without solving another PDE.

The frozen report is recomputed.  Separately, geometry sampling, recovery work,
shared-solve ledgers, repeat trajectories and the screening decision are
checked.  Mesh replay reuses the independent Stage15B replay routine, which
rebuilds meshes from the initial grid and saved marks.  Neither this program
nor the empirical reference checks provide a continuous-error certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import report_stage15c as report
import verify_stage15b as replay


ROOT = Path(__file__).resolve().parent
RECOVERY = {'recovered_dwr3': 3, 'recovered_dwr8': 8}
METHODS = ['uniform_direct', 'uniform_amg', 'coefficient', 'primal_residual',
           'joint_residual', 'dwr', 'recovered_dwr3', 'recovered_dwr8', 'fixed_direct']
SCREEN = dict(min_nonterminal_fraction=.30, max_hypothetical_vs_control=.80,
              max_dwr_time_vs_control=.90, max_dwr_dofs_vs_fastest_control=.75,
              min_geometries_per_family=4, require_both_taus=True,
              require_all_methods_success=True)
sha, ahash, jsonable = replay.sha, replay.ahash, replay.jsonable
check, require, near = replay._check, replay.require, replay.close_number


def geometry_checks(freeze):
    """Recreate the predeclared finite pools without importing their generator."""
    checks = []
    pairs = list(combinations(range(1, 8), 2))
    pools = {
        'layers': [list(v) for v in combinations(range(1, 8), 3) if v != (2, 4, 6)],
        'barrier': [list(v) for v in pairs if v[1]-v[0] in (1, 2) and v != (4, 5)],
        'channel': [list(v) for v in pairs if v[1]-v[0] in (1, 2, 3) and v != (3, 5)],
        'inclusion': [list(x+y) for x, y in product(pairs, repeat=2)
                      if 1 <= x[1]-x[0] <= 3 and 1 <= y[1]-y[0] <= 4 and x+y != (4, 6, 2, 6)],
    }
    expected = []
    for fi, (family, pool) in enumerate(pools.items()):
        indices = np.random.default_rng(1520261123+fi).choice(len(pool), 6, replace=False)
        for j, index in enumerate(indices):
            expected.append(dict(name=f'{family}_{j:02d}', family=family,
                geometry_eighths=pool[int(index)], contrast=10 if j % 2 == 0 else 1000,
                seed=1520261123+fi, pool_size=len(pool)))
    check(checks, 'design/case_count', len(freeze['cases']) == 24)
    check(checks, 'design/methods', freeze['config']['methods'] == METHODS)
    check(checks, 'design/taus_repeats', freeze['config']['taus'] == [1e-3, 3e-4]
          and freeze['config']['repeats'] == 3)
    check(checks, 'design/screen_thresholds', freeze['config']['screening'] == SCREEN)
    check(checks, 'design/no_training', freeze['config']['models_trained'] == 0)
    check(checks, 'design/total_soft_budget', freeze['config']['experiment_soft_seconds'] == 5400.)
    check(checks, 'design/initial_and_fixed_mesh', freeze['config']['initial_n'] == 8
          and freeze['config']['fixed_n'] == 256)
    for index, spec in enumerate(expected):
        if index >= len(freeze['cases']):
            break
        case = freeze['cases'][index]
        metadata = case['metadata']; label = spec['name']
        check(checks, label+'/sample', case['name'] == spec['name']
              and all(metadata.get(key) == value for key, value in spec.items()))
        cuts = [v/8 for v in spec['geometry_eighths']]
        interfaces = (cuts, []) if spec['family'] in ('layers', 'barrier') else (
            ([], cuts) if spec['family'] == 'channel' else (cuts[:2], cuts[2:]))
        check(checks, label+'/interfaces', metadata['interface_x'] == interfaces[0]
              and metadata['interface_y'] == interfaces[1])
        check(checks, label+'/fixed_physics', metadata['k_min'] == 1./spec['contrast']
              and metadata['k_max'] == 1. and metadata['source_center'] == [.25, .5]
              and metadata['goal_center'] == [.75, .5] and metadata['radius'] == .18
              and metadata['source_integral'] == metadata['goal_integral'] == 1.
              and metadata['manufactured'] is False
              and metadata['coefficient_alignment'] == 'element_aligned')
    return checks


def contract_checks(freeze, references, paths, complete):
    checks = replay.contract_checks(freeze, references, paths, complete)
    metadata = {case['name']: case['metadata'] for case in freeze['cases']}
    expected_order = []
    methods = freeze['config']['methods']
    for ci, case in enumerate(freeze['cases']):
        for ti, tau in enumerate(freeze['config']['taus']):
            for repeat in range(freeze['config']['repeats']):
                offset = (ci+ti+repeat) % len(methods)
                expected_order.extend((case['name'], tau, method, repeat)
                    for method in methods[offset:]+methods[:offset])
    actual_order = [(entry['case'], entry['tau'], entry['method'], entry['repeat']) for entry, _, _ in paths]
    check(checks, 'frozen_method_rotation', actual_order == expected_order)
    for entry, record, payload in paths:
        label = f"{entry['case']}/{entry['tau']}/{entry['method']}/{entry['repeat']}"
        method, rows = entry['method'], record['rows']
        fixed, recovering = method.startswith('fixed_'), method in RECOVERY
        check(checks, label+'/family_identity', entry.get('family') == metadata[entry['case']]['family'])
        hashes = [ahash(payload[key]) for key in ('p', 't', 'u')]
        content_id = hashlib.sha256(('|'.join(hashes)+str(bool(payload['sort_t']))).encode()).hexdigest()
        check(checks, label+'/content_addressed_payload', entry['payload_file'] == f'payloads/{content_id}.npz')
        for index, row in enumerate(rows):
            prefix = label+f'/state{index}'
            solves = row['solves']
            p1 = [call for call in solves if call['degree'] == 1]
            p2 = [call for call in solves if call['degree'] == 2]
            check(checks, prefix+'/one_shared_p1_factor', len(p1) <= 1)
            check(checks, prefix+'/at_most_one_paid_p2', len(p2) <= 1)
            for call in p1:
                expected = [0, 1] if recovering or method == 'joint_residual' else [0]
                check(checks, prefix+'/p1_columns', call['metrics']['requested_columns'] == expected
                      and call['metrics']['rhs_count'] == len(expected))
            for call in p2:
                check(checks, prefix+'/p2_never_primal', call['metrics']['requested_columns'] == [1]
                      and call['metrics']['rhs_count'] == 1 and call['role'] == 'p2_dual_only')
            if any(not replay.paid_solver_ok(call) for call in solves):
                check(checks, prefix+'/linear_failure_status', row['status'] == 'linear_solver_failed')
            if row['status'] == 'fixed_mesh_completed':
                check(checks, prefix+'/fixed_success_paid_primal', bool(p1) and not p2
                      and all(replay.paid_solver_ok(call) for call in p1))
            if fixed:
                continue
            for key, value in freeze['config']['adaptive'].items():
                check(checks, prefix+'/frozen_config/'+key, record['summary']['config'][key] == value)
            recovery = row.get('recovery')
            local = row.get('recovered_local')
            if not recovering:
                check(checks, prefix+'/ordinary_method_no_recovery', recovery is None and local is None)
            if row['status'] == 'continue' and recovering:
                check(checks, prefix+'/continuation_requires_recovery', recovery is not None and local is not None)
            if row['status'] in ('estimated_converged', 'max_steps', 'linear_solver_failed'):
                check(checks, prefix+'/terminal_does_not_recover', recovery is None and local is None)
            if recovery is not None:
                check(checks, prefix+'/recovery_steps', recovering and recovery['steps'] == RECOVERY.get(method)
                      and recovery['matvec_count'] == RECOVERY.get(method)
                      and recovery['diagonal_extractions'] == 1 and recovery['goal_rhs_index'] == 1
                      and recovery['omega'] == 2./3.)
                check(checks, prefix+'/recovery_noncertifying', recovery['unverified_marker'] is True
                      and recovery['converged_solve_claim'] is False
                      and recovery['no_reference_or_true_p2_adjoint_used'] is True
                      and recovery['source'] == 'accepted_P1_adjoint_lift_then_fixed_damped_Jacobi')
                check(checks, prefix+'/recovery_accepted_p1_dual', len(p1) == 1
                      and replay.paid_solver_ok(p1[0]) and p1[0]['metrics']['requested_columns'] == [0, 1])
                check(checks, prefix+'/recovery_boundary', np.isfinite(recovery['boundary_maximum'])
                      and 0. <= recovery['boundary_maximum'] <= 1e-13)
                times = recovery['timings']
                check(checks, prefix+'/recovery_times', all(np.isfinite(v) and v >= 0. for v in times.values())
                      and sum(v for key, v in times.items() if key != 'total_seconds') <= times['total_seconds']+1e-9
                      and times['total_seconds'] <= row['timings']['recovery_seconds']+1e-9)
                if row['status'] != 'exception':
                    local_array = np.asarray(local, dtype=float)
                    check(checks, prefix+'/recovery_local_array', local_array.shape == (row['elements'],)
                          and np.all(np.isfinite(local_array)))
                    check(checks, prefix+'/recovery_absolute_marking', np.array_equal(abs(local_array), row['scores']))
                # A paid true dual, if present, remains a global check only.
                check(checks, prefix+'/true_dual_not_local_marker', row['dwr'] is None or row['dwr']['local'] is None)
            if row['status'] != 'exception':
                usage = (['verification'] if p2 else []) + (['recovery'] if recovery is not None else [])
                check(checks, prefix+'/p2_single_assembly_usage', row.get('p2_assembly_usage') == usage)
                check(checks, prefix+'/p2_assembly_charged', bool(usage) == ('p2_assembly_seconds' in row['timings']))
    return checks


def independent_screen(freeze, references, paths):
    """Recompute all medians and Boolean gates directly from raw paths."""
    refs = {ref['case']: ref for ref in references}
    groups = defaultdict(list)
    for entry, record, _ in paths:
        summary = record['summary']
        valid, _ = replay.independent_success(summary, refs[entry['case']], entry['tau'])
        valid = valid and entry['final_mesh_check']['accepted']
        elapsed = summary['algorithm_seconds']
        spent = 0.
        if entry['method'] == 'dwr':
            for row in record['rows']:
                if row['status'] == 'continue':
                    spent += sum(row['timings'].get(key, 0.) for key in (
                        'p2_assembly_seconds', 'p2_dual_solve_seconds',
                        'dwr_evaluation_seconds', 'dwr_localization_seconds'))
        groups[(entry['case'], entry['tau'], entry['method'])].append(dict(
            valid=valid, seconds=elapsed, dofs=record['rows'][-1].get('p1_free_dofs', 0),
            fraction=spent/elapsed, remaining=elapsed-spent, spent=spent))
    cfg = freeze['config']; thresholds = cfg['screening']; settings = []
    for case in freeze['cases']:
        for tau in cfg['taus']:
            statistics = {}
            for method in cfg['methods']:
                group = groups[(case['name'], tau, method)]
                if len(group) != cfg['repeats']:
                    raise ValueError('Screening requires the entire frozen repeat matrix')
                statistics[method] = dict(valid=all(row['valid'] for row in group), **{
                    key: median(row[key] for row in group)
                    for key in ('seconds', 'dofs', 'fraction', 'remaining', 'spent')})
            dwr = statistics['dwr']
            controls = [name for name in cfg['methods'] if name != 'dwr' and statistics[name]['valid']]
            winner = min(controls, key=lambda name: statistics[name]['seconds']) if controls else None
            qualified = bool(refs[case['name']]['assessment']['accepted'])
            numerical = qualified and all(row['valid'] for row in statistics.values())
            time_ratio = dof_ratio = remaining_ratio = None
            cost = information = False
            if winner is not None and dwr['valid'] and qualified:
                control = statistics[winner]
                time_ratio, dof_ratio = dwr['seconds']/control['seconds'], dwr['dofs']/control['dofs']
                remaining_ratio = dwr['remaining']/control['seconds']
                cost = not (dwr['fraction'] < thresholds['min_nonterminal_fraction']
                            or remaining_ratio > thresholds['max_hypothetical_vs_control'])
                information = not (time_ratio > thresholds['max_dwr_time_vs_control']
                                   and dof_ratio > thresholds['max_dwr_dofs_vs_fastest_control'])
            settings.append(dict(case=case['name'], tau=tau, numerical=numerical,
                cost_necessary=cost, goal_information=information, passed=all((numerical, cost, information)),
                fastest_control=winner, dwr_time_over_control=time_ratio, dwr_dofs_over_control=dof_ratio,
                hypothetical_remaining_over_control=remaining_ratio,
                dwr_nonterminal_fraction=dwr['fraction'], dwr_nonterminal_ms=1000*dwr['spent']))
    geometries = []
    for case in freeze['cases']:
        cohort = [row for row in settings if row['case'] == case['name']]
        geometries.append(dict(case=case['name'], family=case['metadata']['family'],
            contrast=case['metadata']['contrast'], passed=len(cohort) == len(cfg['taus']) and all(row['passed'] for row in cohort),
            passed_settings=sum(row['passed'] for row in cohort), settings=len(cohort)))
    families = []
    for family in dict.fromkeys(case['metadata']['family'] for case in freeze['cases']):
        cohort = [row for row in geometries if row['family'] == family]
        passed = sum(row['passed'] for row in cohort)
        families.append(dict(family=family, geometries=len(cohort), passed=passed,
            retain_for_limited_design=passed >= thresholds['min_geometries_per_family']))
    retained = [row['family'] for row in families if row['retain_for_limited_design']]
    return dict(thresholds=thresholds, settings=settings, geometries=geometries, families=families,
        retained_families=retained, decision='LIMITED_DESIGN_ONLY' if retained else 'STOP_CURRENT_ESTIMATOR_REPLACEMENT_BRANCH',
        G1='NOT_FULLY_APPROVED', G3='NOT_EVALUATED', no_ai_model=True,
        retrospective_nonterminal_labels=True, hypothetical_cost_not_measured_speedup=True)


def compare_nested(expected, saved, path='screening'):
    checks = []
    if isinstance(expected, dict):
        check(checks, path+'/keys', isinstance(saved, dict) and set(expected) == set(saved))
        if isinstance(saved, dict):
            for key, value in expected.items():
                checks.extend(compare_nested(value, saved.get(key), path+'/'+str(key)))
    elif isinstance(expected, list):
        check(checks, path+'/length', isinstance(saved, list) and len(expected) == len(saved))
        if isinstance(saved, list):
            for index, (first, second) in enumerate(zip(expected, saved)):
                checks.extend(compare_nested(first, second, path+'/'+str(index)))
    elif isinstance(expected, (float, np.floating)):
        check(checks, path, near(expected, saved))
    else:
        check(checks, path, expected == saved)
    return checks


def repeat_checks(freeze, paths):
    result = replay.repeat_audit(freeze, paths)
    groups = defaultdict(list)
    for entry, record, _ in paths:
        groups[(entry['case'], entry['tau'], entry['method'])].append(record)
    checks = []
    fields = ('status', 'qoi', 'mesh_coordinates_sha256', 'mesh_connectivity_sha256',
              'solution_sha256', 'scores', 'selected_elements', 'verified',
              'verification_reasons', 'dwr', 'recovered_local', 'p2_assembly_usage')
    exact = []
    for key, records in groups.items():
        reference = records[0]
        trajectory = lambda record: [{field: row.get(field) for field in fields} for row in record['rows']]
        same = all(trajectory(record) == trajectory(reference) for record in records)
        time_limited = any(record['summary']['status'] == 'time_limit' for record in records)
        check(checks, '/'.join(map(str, key))+'/deterministic_or_time_limited', same or time_limited)
        exact.append(dict(case=key[0], tau=key[1], method=key[2],
                          exact_numerical_trajectory=same, time_limit_present=time_limited))
    result['exact_trajectory_groups'] = sum(row['exact_numerical_trajectory'] for row in exact)
    result['extended_trajectory_checks'] = exact
    result['timings_excluded_from_equality'] = True
    return result, checks


def corruption_controls(freeze, references, paths, complete, inputdir):
    # The original controls already cover missing paths, loosened reference
    # gates, removed paid duals, false q>1 success, and retained linear failure.
    controls = replay.corruption_controls(freeze, references, paths, complete, inputdir)
    index = next(i for i, (entry, record, _) in enumerate(paths)
                 if entry['method'] in RECOVERY and any(row.get('recovery') for row in record['rows']))
    original_entry, original_record, payload = paths[index]

    def rejected(name, entry, record, suffix):
        altered = list(paths); altered[index] = (entry, record, payload)
        failures = [row['name'] for row in contract_checks(freeze, references, altered, complete) if not row['passed']]
        if not any(value.endswith(suffix) for value in failures):
            raise AssertionError('Negative control did not reject '+name)
        controls.append(dict(name=name, rejected=True, targeted_failure=suffix,
                             production_data_modified=False))

    entry, record = deepcopy(original_entry), deepcopy(original_record)
    recovery = next(row['recovery'] for row in record['rows'] if row.get('recovery'))
    recovery['matvec_count'] += 1
    rejected('recovery_fixed_work_counter_changed', entry, record, '/recovery_steps')
    entry, record = deepcopy(original_entry), deepcopy(original_record)
    recovery = next(row['recovery'] for row in record['rows'] if row.get('recovery'))
    recovery['converged_solve_claim'] = True
    rejected('recovery_forged_as_converged_dual', entry, record, '/recovery_noncertifying')
    entry, record = deepcopy(original_entry), deepcopy(original_record)
    row = next(row for row in record['rows'] if row.get('recovery'))
    row['p2_assembly_usage'] = []
    rejected('recovery_assembly_charge_removed', entry, record, '/p2_single_assembly_usage')
    entry, record = deepcopy(original_entry), deepcopy(original_record)
    row = next(row for row in record['rows'] if row.get('recovery'))
    row['solves'][0]['metrics']['requested_columns'] = [0]
    rejected('recovery_p1_adjoint_column_removed', entry, record, '/p1_columns')
    entry, record = deepcopy(original_entry), deepcopy(original_record)
    row = next(row for row in record['rows'] if row.get('recovery'))
    row['recovered_local'][0] += 1.
    rejected('recovered_indicator_changed_without_marks', entry, record, '/recovery_absolute_marking')

    # A consistently recorded resource stop remains valid failed data.  The
    # synthetic row keeps its paid calculations; its offline flag cannot pass.
    index = next(i for i, (entry, _, _) in enumerate(paths) if entry['method'] == 'fixed_direct')
    original_entry, original_record, payload = paths[index]
    entry, record = deepcopy(original_entry), deepcopy(original_record)
    record['rows'][-1]['status'] = 'time_limit'; record['summary']['status'] = 'time_limit'
    entry['summary'] = deepcopy(record['summary'])
    reference = next(ref for ref in references if ref['case'] == entry['case'])
    entry['offline'] = report.offline_assessment(record['summary'], reference, entry['tau'])
    altered = list(paths); altered[index] = (entry, record, payload)
    require(contract_checks(freeze, references, altered, complete))
    controls.append(dict(name='consistent_resource_failure_retained', accepted_as_valid_failure=True,
                         success=False, production_data_modified=False))
    entry = deepcopy(entry); entry['offline'].update(success=True, status='success')
    rejected('resource_failure_forged_as_success', entry, record, '/resource_or_linear_failure_retained')
    return controls


def run(inputdir, reportdir, output, *, check_sources=True):
    inputdir, reportdir, output = Path(inputdir), Path(reportdir), Path(output)
    if output.exists():
        raise FileExistsError('Use a new verification output; never overwrite results')
    started = perf_counter(); own = sha(__file__)
    dependencies = {name: sha(ROOT/name) for name in ('verify_stage15b.py', 'report_stage15c.py')}
    raw_digest, report_digest = sha(inputdir/'complete.json'), sha(reportdir/'complete.json')
    report_complete = json.loads((reportdir/'complete.json').read_text())
    if report_complete.get('complete') is not True or report_complete['raw_complete_sha256'] != raw_digest:
        raise ValueError('Report is not linked to this complete raw run')
    if report_complete.get('report_source_sha256') != dependencies['report_stage15c.py']:
        raise ValueError('Report source digest does not match the audited implementation')
    for needed in ('summary.json', 'integrity_checks.json', 'REPORT_zh.md'):
        if needed not in report_complete['file_sha256']:
            raise ValueError('Report artifact is not protected by manifest: '+needed)
    for name, digest in report_complete['file_sha256'].items():
        path = (reportdir/name).resolve()
        if reportdir.resolve() not in path.parents or sha(path) != digest:
            raise ValueError('Report artifact hash mismatch or unsafe path: '+name)
    freeze, references, paths, complete = report.read_checked(inputdir, check_sources=check_sources)
    summary, report_checks = report.summarize(freeze, references, paths, complete)
    saved = json.loads((reportdir/'summary.json').read_text())
    saved_checks = json.loads((reportdir/'integrity_checks.json').read_text())
    if jsonable(summary) != saved or jsonable(report_checks) != saved_checks:
        raise ValueError('Saved report differs from its raw-data recomputation')
    geometry = geometry_checks(freeze); require(geometry)
    contracts = contract_checks(freeze, references, paths, complete); require(contracts)
    screen = independent_screen(freeze, references, paths)
    screen_checks = compare_nested(screen, saved['screening']); require(screen_checks)
    meshes, _ = replay.mesh_replay(freeze, paths)
    repeats, repeat_tests = repeat_checks(freeze, paths); require(repeat_tests)
    controls = corruption_controls(freeze, references, paths, complete, inputdir)
    recovery_counts = Counter()
    for entry, record, _ in paths:
        if entry['method'] in RECOVERY:
            for row in record['rows']:
                if row.get('recovery'):
                    recovery_counts[entry['method']+'/calls'] += 1
                    recovery_counts[entry['method']+'/matvecs'] += row['recovery']['matvec_count']
    if sha(__file__) != own or any(sha(ROOT/name) != digest for name, digest in dependencies.items()) \
            or sha(inputdir/'complete.json') != raw_digest or sha(reportdir/'complete.json') != report_digest:
        raise ValueError('An audit input or verifier changed during verification')
    result = dict(verified=True, created_utc=datetime.now(timezone.utc).isoformat(), paths=len(paths),
        report_recomputed=True, report_integrity_checks=len(report_checks),
        geometry_contract_checks=len(geometry), independent_contract_checks=len(contracts),
        screening_checks=len(screen_checks), screening_recomputed_from_raw_paths=True,
        screening=screen, mesh_replay=meshes, repeats=repeats,
        recovery_work_counts=dict(recovery_counts), negative_and_failure_controls=controls,
        actual_online_failures_retained=dict(Counter(record['summary']['status'] for _, record, _ in paths
                                                     if record['summary']['status'] in replay.FAILURES)),
        accepted_failures_are_not_successes=True, source_hashes_checked=check_sources,
        verifier_source_sha256=own, verifier_dependencies_sha256=dependencies,
        raw_complete_sha256=raw_digest, report_complete_sha256=report_digest,
        report_summary_sha256=sha(reportdir/'summary.json'), raw_manifest_files=len(complete['file_sha256']),
        report_manifest_files=len(report_complete['file_sha256']), verification_seconds=perf_counter()-started,
        limitations=['Report and reference formulas are reused; no independent second FEM solver is run.',
            'Mesh replay rebuilds saved refinements, while solution hashes protect stored arrays without re-solving.',
            'Recovery counters and data-flow metadata are audited; the full production Jacobi calculation is not repeated.',
            'Dense-recurrence and dependency-intervention tests separately validate the fixed recovery implementation.',
            'The screening medians and decision are independently recomputed from raw records, not an AI speed claim.',
            'Reference stability, recovered markers and paid DWR stopping are not continuous-error certificates.',
            'Timing is excluded from repeat equality; resource-limit variations are explicitly preserved.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'results/stage15c_classical')
    parser.add_argument('--report', type=Path, default=ROOT/'results/stage15c_report')
    parser.add_argument('--output', type=Path, default=ROOT/'results/stage15c_verification.json')
    args = parser.parse_args()
    value = run(args.input, args.report, args.output)
    print(json.dumps(dict(verified=value['verified'], paths=value['paths'],
        replay_checks=value['mesh_replay']['checks'], controls=len(value['negative_and_failure_controls']),
        screening=value['screening']['decision'])))
