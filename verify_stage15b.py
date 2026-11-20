"""Post-run integrity and mesh-replay audit, without another PDE experiment.

The report aggregation and reference qualification formulas are reused and are
not an independent second FEM solver.  Mesh trajectories are reconstructed from
the frozen initial resolution and saved marks, separately from report checks.
Synthetic corruption controls use detached records or a small temporary copy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

import report_stage15b as report
from stage15_fem import make_mesh
from stage15b_reference import assess_reference


ROOT = Path(__file__).resolve().parent
FAILURES = {'max_steps', 'max_dofs', 'time_limit', 'linear_solver_failed',
            'exception', 'zero_indicator'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check(checks, name, condition):
    checks.append(dict(name=name, passed=bool(condition)))


def require(checks):
    failed = [check['name'] for check in checks if not check['passed']]
    if failed:
        raise ValueError('Independent verification failed: ' + '; '.join(failed[:15]))


def close_number(first, second):
    if first is None or second is None:
        return first is second
    return bool(np.isfinite(first) and np.isfinite(second) and
                abs(first-second) <= 1e-12 + 1e-10*max(abs(first), abs(second)))


def independent_success(summary, reference, tau):
    """Separate Boolean acceptance, used also by synthetic failure controls."""
    value = summary.get('final_qoi')
    target = reference['assessment']['qoi']
    finite = value is not None and np.isfinite(value) and np.isfinite(target)
    qualified = bool(reference['assessment']['accepted'])
    complete = summary['status'] in ('estimated_converged', 'fixed_mesh_completed')
    ratio = None if not finite else float(abs(value-target)/tau)
    return bool(qualified and complete and ratio is not None and ratio <= 1.), ratio


def paid_solver_ok(call):
    metrics = call['metrics']
    columns = metrics.get('per_rhs', [])
    return bool(columns and metrics.get('converged') and metrics.get('accepted') and
                metrics.get('all_info_zero') and all(
                    np.isfinite(column['relative_residual']) and column['relative_residual'] <= 1e-9 and
                    np.isfinite(column['componentwise_backward_error']) and column['componentwise_backward_error'] <= 1e-12
                    for column in columns) and metrics.get('boundary_maximum', np.inf) <= 1e-13)


def contract_checks(freeze, references, paths, complete):
    """Cheap, independent information-flow/acceptance checks before replay."""
    checks = []
    cfg = freeze['config']
    names = [case['name'] for case in freeze['cases']]
    expected = {(case, tau, method, repeat) for case in names for tau in cfg['taus']
                for method in cfg['methods'] for repeat in range(cfg['repeats'])}
    keys = [(entry['case'], entry['tau'], entry['method'], entry['repeat']) for entry, _, _ in paths]
    _check(checks, 'complete_path_matrix', len(keys) == len(expected) and set(keys) == expected)
    _check(checks, 'complete_path_count', len(paths) == complete['paths'])
    refs = {ref['case']: ref for ref in references}
    _check(checks, 'complete_reference_matrix', len(refs) == len(references) == len(names) and set(refs) == set(names))
    for name, ref in refs.items():
        _check(checks, name+'/frozen_reference_config', ref['config'] == cfg['reference'])
    for entry, record, _ in paths:
        label = f"{entry['case']}/{entry['tau']}/{entry['method']}/{entry['repeat']}"
        summary, rows = record['summary'], record['rows']
        _check(checks, label+'/saved_summary', summary == entry['summary'])
        _check(checks, label+'/final_status', bool(rows) and rows[-1]['status'] == summary['status'])
        _check(checks, label+'/only_continue_before_final', all(row['status'] == 'continue' for row in rows[:-1]))
        ref = refs[entry['case']]
        expected_success, ratio = independent_success(summary, ref, entry['tau'])
        if not entry['final_mesh_check']['accepted']:
            expected_success = False
        _check(checks, label+'/offline_success', entry['offline']['success'] == expected_success)
        _check(checks, label+'/offline_ratio', close_number(entry['offline']['q'], ratio))
        if summary['status'] in FAILURES:
            _check(checks, label+'/resource_or_linear_failure_retained', not entry['offline']['success'])
        for index, row in enumerate(rows):
            prefix = label+f'/state{index}'
            if row['verified']:
                _check(checks, prefix+'/paid_p2_dual', any(call['degree'] == 2 and
                    call['metrics'].get('requested_columns') == [1] and paid_solver_ok(call)
                    for call in row['solves']))
                _check(checks, prefix+'/current_estimate', row['dwr'] is not None and
                       np.isfinite(row['dwr']['total_estimate']))
            if row['status'] == 'estimated_converged':
                _check(checks, prefix+'/deployable_stop', row['verified'] and row['dwr'] is not None and
                    abs(row['dwr']['total_estimate']) <= summary['config']['safety']*entry['tau'] and
                    bool(row['solves']) and all(paid_solver_ok(call) for call in row['solves']))
    return checks


def mesh_replay(freeze, paths):
    """Rebuild all unique trajectories; every repeated record is compared."""
    checks, cache = [], {}
    states, rebuilt, cached, solution_hashes = 0, 0, 0, 0
    cfg = freeze['config']
    for entry, record, payload in paths:
        fixed = entry['method'].startswith('fixed_')
        n = cfg['fixed_n'] if fixed else cfg['initial_n']
        label = f"{entry['case']}/{entry['tau']}/{entry['method']}/{entry['repeat']}"
        rows = record['rows']
        # The cache stores fingerprints, not meshes.  A repeated trajectory must
        # have identical marks and coordinates at every state to reuse replay.
        skeleton = json.dumps(dict(n=n, fixed=fixed, rows=[{key: row.get(key) for key in (
            'step', 'elements', 'vertices', 'mesh_coordinates_sha256', 'mesh_connectivity_sha256',
            'mesh_sort_t', 'p1_free_dofs', 'p1_full_dofs', 'p2_free_dofs', 'p2_full_dofs',
            'selected_elements', 'selected_sha256', 'next_mesh_coordinates_sha256',
            'next_mesh_connectivity_sha256', 'next_p1_free_dofs', 'status', 'solution_sha256')}
            for row in rows]), sort_keys=True)
        key = hashlib.sha256(skeleton.encode()).hexdigest()
        if key in cache:
            expected = cache[key]
            cached += 1
        else:
            mesh = make_mesh(n)
            final_mesh = mesh
            last_solution_hash = None
            for index, row in enumerate(rows):
                states += 1
                prefix = label+f'/state{index}'
                _check(checks, prefix+'/coordinates_rebuilt', ahash(mesh.p) == row['mesh_coordinates_sha256'])
                _check(checks, prefix+'/connectivity_rebuilt', ahash(mesh.t) == row['mesh_connectivity_sha256'])
                _check(checks, prefix+'/element_count', mesh.nelements == row['elements'])
                _check(checks, prefix+'/vertex_count', mesh.nvertices == row['vertices'])
                _check(checks, prefix+'/sort_flag', bool(mesh.sort_t) == row['mesh_sort_t'])
                free = mesh.nvertices-len(mesh.boundary_nodes())
                if 'p1_free_dofs' in row:
                    _check(checks, prefix+'/p1_free_dofs', free == row['p1_free_dofs'])
                if 'p1_full_dofs' in row:
                    _check(checks, prefix+'/p1_full_dofs', mesh.nvertices == row['p1_full_dofs'])
                if row.get('p2_free_dofs', 0):
                    p2_full = mesh.nvertices+mesh.nfacets
                    p2_free = p2_full-len(mesh.boundary_nodes())-len(mesh.boundary_facets())
                    _check(checks, prefix+'/p2_free_dofs', p2_free == row['p2_free_dofs'])
                    _check(checks, prefix+'/p2_full_dofs', p2_full == row['p2_full_dofs'])
                if fixed or row.get('solution_sha256') is not None:
                    final_mesh = mesh
                    last_solution_hash = row.get('solution_sha256')
                raw_selected = row['selected_elements']
                legal = (isinstance(raw_selected, list) and all(isinstance(v, int) and
                         not isinstance(v, bool) and 0 <= v < mesh.nelements for v in raw_selected) and
                         len(set(raw_selected)) == len(raw_selected))
                _check(checks, prefix+'/legal_selected_elements', legal)
                if not legal:
                    require(checks)
                selected = np.asarray(raw_selected, dtype=int)
                if 'selected_sha256' in row:
                    _check(checks, prefix+'/selected_hash', ahash(selected) == row['selected_sha256'])
                next_mesh = None
                if 'next_mesh_coordinates_sha256' in row:
                    _check(checks, prefix+'/nonempty_refinement', len(selected) > 0)
                    uniform = entry['method'].startswith('uniform_')
                    if uniform:
                        _check(checks, prefix+'/uniform_all_selected', np.array_equal(selected, np.arange(mesh.nelements)))
                    next_mesh = mesh.refined() if uniform else mesh.refined(selected)
                    _check(checks, prefix+'/next_coordinates_rebuilt', ahash(next_mesh.p) == row['next_mesh_coordinates_sha256'])
                    _check(checks, prefix+'/next_connectivity_rebuilt', ahash(next_mesh.t) == row['next_mesh_connectivity_sha256'])
                    _check(checks, prefix+'/next_free_dofs', next_mesh.nvertices-len(next_mesh.boundary_nodes()) == row['next_p1_free_dofs'])
                if index < len(rows)-1:
                    _check(checks, prefix+'/continued_mesh_exists', row['status'] == 'continue' and next_mesh is not None)
                    if next_mesh is None:
                        require(checks)
                    mesh = next_mesh
            require(checks)
            expected = dict(p=ahash(final_mesh.p), t=ahash(final_mesh.t), sort_t=bool(final_mesh.sort_t),
                            vertices=final_mesh.nvertices, solution_sha256=last_solution_hash)
            cache[key] = expected
            rebuilt += 1
        _check(checks, label+'/payload_coordinates_rebuilt', ahash(payload['p']) == expected['p'])
        _check(checks, label+'/payload_connectivity_rebuilt', ahash(payload['t']) == expected['t'])
        _check(checks, label+'/payload_sort_flag', bool(payload['sort_t']) == expected['sort_t'])
        _check(checks, label+'/payload_solution_shape', payload['u'].shape in ((0,), (expected['vertices'],)))
        _check(checks, label+'/payload_solution_finite', np.all(np.isfinite(payload['u'])))
        if expected['solution_sha256'] is not None:
            _check(checks, label+'/payload_solution_hash', ahash(payload['u']) == expected['solution_sha256'])
            solution_hashes += 1
    require(checks)
    return dict(passed=True, checks=len(checks), paths_checked=len(paths), unique_trajectories_rebuilt=rebuilt,
                repeated_trajectories_checked_against_rebuilt_fingerprints=cached,
                unique_states_rebuilt=states, solution_hash_checked_paths=solution_hashes,
                all_payloads_protected_by_raw_manifest=True), checks


def repeat_audit(freeze, paths):
    groups = defaultdict(list)
    for entry, record, _ in paths:
        groups[(entry['case'], entry['tau'], entry['method'])].append((entry, record))
    records = []
    for key, values in groups.items():
        values.sort(key=lambda value: value[0]['repeat'])
        baseline = values[0][1]
        def structure(record):
            return [{key: row.get(key) for key in ('status', 'mesh_coordinates_sha256',
                     'mesh_connectivity_sha256', 'selected_elements', 'verification_reasons', 'verified')}
                    for row in record['rows']]
        same_structure = all(structure(record) == structure(baseline) for _, record in values)
        same_q = all(len(record['rows']) == len(baseline['rows']) and all(
                     close_number(row.get('qoi'), original.get('qoi'))
                     for row, original in zip(record['rows'], baseline['rows'])) for _, record in values)
        resource_variation = any(record['summary']['status'] == 'time_limit' for _, record in values)
        records.append(dict(case=key[0], tau=key[1], method=key[2], repeats=len(values),
                            expected_repeats=freeze['config']['repeats'], structure_consistent=same_structure,
                            qoi_compatible=same_q, time_limit_present=resource_variation,
                            timing_values_not_required_to_match=True))
    return dict(groups=len(records), all_groups_have_frozen_repeat_count=all(
                record['repeats'] == record['expected_repeats'] for record in records),
                structurally_consistent_groups=sum(record['structure_consistent'] for record in records),
                numerically_compatible_groups=sum(record['qoi_compatible'] for record in records),
                disagreements_preserved=[record for record in records if not
                    record['structure_consistent'] or not record['qoi_compatible']], records=records)


def corruption_controls(freeze, references, paths, complete, inputdir):
    controls = []
    def rejected(name, altered_refs, altered_paths, expected_suffix):
        checks = contract_checks(freeze, altered_refs, altered_paths, complete)
        failed = [check['name'] for check in checks if not check['passed']]
        rejected_for_reason = any(value.endswith(expected_suffix) for value in failed)
        if not rejected_for_reason:
            raise AssertionError('Negative control did not reject expected defect: '+name)
        controls.append(dict(name=name, rejected=True, targeted_failure=expected_suffix,
                             rejection_examples=failed[:3], production_data_modified=False))

    rejected('missing_path', references, paths[:-1], 'complete_path_matrix')
    altered_refs = deepcopy(references)
    altered_refs[0]['config']['finest_difference_fraction'] *= 100.
    altered_refs[0]['assessment'] = assess_reference(altered_refs[0]['rows'], altered_refs[0]['config'])
    rejected('relaxed_reference_gate_and_consistent_assessment', altered_refs, paths, '/frozen_reference_config')

    index = next(index for index, (_, record, _) in enumerate(paths)
                 if any(row['verified'] for row in record['rows']))
    entry, record, payload = paths[index]
    altered_entry, altered_record = deepcopy(entry), deepcopy(record)
    row = next(row for row in altered_record['rows'] if row['verified'])
    row['solves'] = [call for call in row['solves'] if call['degree'] != 2]
    calls = [call for state in altered_record['rows'] for call in state['solves']]
    altered_record['summary']['solver_calls'] = len(calls)
    altered_record['summary']['rhs_count'] = sum(call['metrics']['rhs_count'] for call in calls)
    altered_entry['summary'] = deepcopy(altered_record['summary'])
    altered_paths = list(paths)
    altered_paths[index] = (altered_entry, altered_record, payload)
    rejected('verified_but_paid_p2_adjoint_removed', references, altered_paths, '/paid_p2_dual')

    # Make the false success explicit even if the actual run has no false stops.
    index = next(index for index, (entry, _, _) in enumerate(paths) if not entry['method'].startswith('fixed_'))
    entry, record, payload = paths[index]
    altered_entry, altered_record = deepcopy(entry), deepcopy(record)
    target = next(ref['assessment']['qoi'] for ref in references if ref['case'] == entry['case'])
    altered_record['summary'].update(status='estimated_converged', final_qoi=target+2.*entry['tau'])
    altered_record['rows'][-1].update(status='estimated_converged', qoi=target+2.*entry['tau'])
    altered_entry['summary'] = deepcopy(altered_record['summary'])
    altered_entry['offline'].update(success=True, status='success', q=2.)
    altered_paths = list(paths)
    altered_paths[index] = (altered_entry, altered_record, payload)
    rejected('estimated_stop_with_q_above_one_forged_as_success', references, altered_paths, '/offline_success')

    # A consistently recorded failed solve is data to keep, not corruption.
    index = next(index for index, (entry, _, _) in enumerate(paths) if entry['method'] == 'fixed_direct')
    entry, record, payload = paths[index]
    failed_entry, failed_record = deepcopy(entry), deepcopy(record)
    row = failed_record['rows'][-1]
    metrics = row['solves'][0]['metrics']
    metrics['accepted'] = metrics['converged'] = False
    metrics['relative_residual'] = metrics['per_rhs'][0]['relative_residual'] = 1e-8
    row['status'] = 'linear_solver_failed'
    failed_record['summary']['status'] = 'linear_solver_failed'
    failed_entry['summary'] = deepcopy(failed_record['summary'])
    ref = next(ref for ref in references if ref['case'] == entry['case'])
    failed_entry['offline'] = report.offline_assessment(failed_record['summary'], ref, entry['tau'])
    altered_paths = list(paths)
    altered_paths[index] = (failed_entry, failed_record, payload)
    require(contract_checks(freeze, references, altered_paths, complete))
    controls.append(dict(name='consistent_linear_failure_retained', accepted_as_valid_failure=True,
                         success=False, production_data_modified=False))
    failed_entry = deepcopy(failed_entry)
    failed_entry['offline'].update(success=True, status='success')
    altered_paths[index] = (failed_entry, failed_record, payload)
    rejected('failed_solver_forged_as_success', references, altered_paths, '/resource_or_linear_failure_retained')

    # Copy only three small JSON files; no trajectory/payload copies are needed
    # to prove that the original raw reader rejects a digest mismatch.
    with tempfile.TemporaryDirectory(prefix='darcy-stage15b-hash-control-') as directory:
        directory = Path(directory)
        manifest = {}
        for name in ('freeze.json', 'references.json', 'index.json'):
            (directory/name).write_bytes((inputdir/name).read_bytes())
            manifest[name] = sha(directory/name)
        (directory/'complete.json').write_text(json.dumps(dict(complete=True, file_sha256=manifest)))
        with (directory/'freeze.json').open('ab') as stream:
            stream.write(b'\n')
        try:
            report.read_checked(directory, check_sources=False)
        except ValueError as error:
            if 'Raw file hash mismatch: freeze.json' not in str(error):
                raise
        else:
            raise AssertionError('Raw hash corruption was not rejected')
    controls.append(dict(name='raw_file_digest_tamper', rejected=True, production_data_modified=False,
                         copied_payload_files=0))
    return controls


def run(inputdir, reportdir, output, *, check_sources=True):
    inputdir, reportdir, output = Path(inputdir), Path(reportdir), Path(output)
    if output.exists():
        raise FileExistsError('Use a new verification output; never overwrite results')
    started = perf_counter()
    own_digest = sha(__file__)
    raw_digest = sha(inputdir/'complete.json')
    report_digest = sha(reportdir/'complete.json')
    report_complete = json.loads((reportdir/'complete.json').read_text())
    if not report_complete['complete'] or report_complete['raw_complete_sha256'] != raw_digest:
        raise ValueError('Report is not linked to this complete raw run')
    for name in ('summary.json', 'integrity_checks.json', 'REPORT_zh.md'):
        if name not in report_complete['file_sha256']:
            raise ValueError('Required report file missing from manifest: '+name)
    for name, expected in report_complete['file_sha256'].items():
        path = (reportdir/name).resolve()
        if reportdir.resolve() not in path.parents or sha(path) != expected:
            raise ValueError('Report file hash mismatch or unsafe path: '+name)
    freeze, references, paths, complete = report.read_checked(inputdir, check_sources=check_sources)
    summary, report_checks = report.summarize(freeze, references, paths, complete)
    saved_summary = json.loads((reportdir/'summary.json').read_text())
    saved_checks = json.loads((reportdir/'integrity_checks.json').read_text())
    if jsonable(summary) != saved_summary or jsonable(report_checks) != saved_checks:
        raise ValueError('Saved report differs from recomputed summary/checks')
    contracts = contract_checks(freeze, references, paths, complete)
    require(contracts)
    replay, replay_checks = mesh_replay(freeze, paths)
    repeats = repeat_audit(freeze, paths)
    controls = corruption_controls(freeze, references, paths, complete, inputdir)
    if sha(__file__) != own_digest or sha(inputdir/'complete.json') != raw_digest or sha(reportdir/'complete.json') != report_digest:
        raise ValueError('A verification input or verifier changed during the audit')
    result = dict(verified=True, created_utc=datetime.now(timezone.utc).isoformat(),
        paths=len(paths), report_recomputed=True, report_integrity_checks=len(report_checks),
        independent_contract_checks=len(contracts), mesh_replay=replay, repeats=repeats,
        negative_and_failure_controls=controls,
        actual_online_failures_retained=dict(Counter(record['summary']['status'] for _, record, _ in paths
                                                     if record['summary']['status'] in FAILURES)),
        accepted_failures_are_not_successes=True, source_hashes_checked=check_sources,
        verifier_source_sha256=own_digest, raw_complete_sha256=raw_digest,
        report_complete_sha256=report_digest, report_summary_sha256=sha(reportdir/'summary.json'),
        raw_manifest_files=len(complete['file_sha256']), report_manifest_files=len(report_complete['file_sha256']),
        verification_seconds=perf_counter()-started,
        limitations=['Report aggregation and reference formulas are reused; this is not an independent second FEM solver.',
                     'Replay validates meshes and stored selections; it does not re-solve every finite-element system.',
                     'Saved nodal-solution hashes and the raw manifest protect payload integrity; they do not constitute a new numerical solve.',
                     'Reference stability and DWR stopping remain empirical and do not certify continuous PDE error.',
                     'Repeat disagreements are retained explicitly; timing values are never required to match.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'results/stage15b_classical')
    parser.add_argument('--report', type=Path, default=ROOT/'results/stage15b_report')
    parser.add_argument('--output', type=Path, default=ROOT/'results/stage15b_verification.json')
    args = parser.parse_args()
    result = run(args.input, args.report, args.output)
    print(json.dumps(dict(verified=result['verified'], paths=result['paths'],
                         replay_checks=result['mesh_replay']['checks'],
                         controls=len(result['negative_and_failure_controls']))))
