#!/usr/bin/env python3
"""Stage 8: a bounded, retrospective direction-budget diagnostic.

Current-system directions come from a completed base-M solve. They are not
causally available directions or optimal spectral oracles. Preparation of
the entire phase, including both grids, finishes before any timed method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
import time

import persistent_recycling_experiment as previous
from persistent_recycling_experiment import (
    np, la, sla, threadpool_limits, coefficient_sequence, assemble_system,
    AMGSequencePreconditioner, a_orthonormalize, galerkin_correction,
    balanced_preconditioner, solve_pcg, solve_pcg_cached, cached_slow_ritz,
    energy_relative_error, initial, is_event, read, sha, stamp,
)

ROOT = Path(__file__).resolve().parent
RTOL, STEPS, CONTRAST = previous.RTOL, previous.STEPS, previous.CONTRAST
RANKS, LIMIT, REPEATS, ORTH_TOL, CACHE_TOL = (4, 8), 64, 3, 1e-7, 1e-7
SOURCES = [*previous.SOURCES, 'oracle_budget_experiment.py']
PROTOCOL, SPLITS = 'docs/stage8_protocol_zh.md', 'data/stage8_splits.json'
FREEZE = 'results/stage8_freeze.json'
DECISION = 'results/stage8_decision.json'
COMPLETE = 'results/stage8_complete.json'


def json_safe(value):
    """Retain failed records without emitting nonstandard NaN JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False))


def array_sha(value):
    value = np.asarray(value, dtype=np.float64)
    digest = hashlib.sha256(str(value.shape).encode())
    digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def specs():
    old = {s['name']: s for s in previous.specs()}
    return [old['baseline_scipy'], old['baseline_event_custom'],
            old['frozen_all_balanced4'],
            *[dict(name=f'current_reference{q}', policy='current_reference',
                   event_custom=True, rank=q) for q in RANKS]]


def splits():
    path = ROOT / SPLITS
    if path.exists():
        return read(path)
    old = read(ROOT / 'data/stage7_splits.json')
    excluded = sorted(set(old['excluded_seeds'] +
                          sum(old['groups'].values(), []) + [75004]))
    seen = {previous.geometry_hash(seed) for seed in excluded}
    value = dict(excluded_seeds=excluded, groups={}, geometry_sha256={},
                 skipped_duplicates=[])
    for name, start, count in [('development', 46001, 3),
                                ('confirmation', 46011, 5)]:
        seeds = []
        while len(seeds) < count:
            digest = previous.geometry_hash(start)
            if digest in seen:
                value['skipped_duplicates'].append(dict(seed=start, group=name))
            else:
                seen.add(digest)
                seeds.append(start)
                value['geometry_sha256'][str(start)] = digest
            start += 1
        value['groups'][name] = seeds
    save(path, value)
    return value


def provenance():
    return dict(source_sha256={f: sha(ROOT / f) for f in SOURCES},
                protocol_sha256=sha(ROOT / PROTOCOL),
                splits_sha256=sha(ROOT / SPLITS))


def check_provenance(record):
    for key, value in provenance().items():
        if record[key] != value:
            raise RuntimeError(f'Frozen {key} changed')


def finite(value):
    return value is not None and bool(np.isfinite(value))


def _safe_ritz(cache, q, limit=None):
    arrays = [cache[key][:, :limit] for key in ('D', 'AD', 'MAD')]
    try:
        return cached_slow_ritz(*arrays, q=q)
    except (la.LinAlgError, ValueError, FloatingPointError) as error:
        empty = np.empty((arrays[0].shape[0], 0))
        return empty, empty.copy(), dict(status='numerical_exception', rank=0,
            requested_rank=q, input_vectors=arrays[0].shape[1], ritz_values=[],
            error=f'{type(error).__name__}: {error}')


def _a_norms(A, columns):
    squared = np.sum(columns * (A @ columns), axis=0)
    # Negative/nonfinite energies remain invalid rather than being hidden.
    with np.errstate(invalid='ignore'):
        return np.sqrt(squared)


def space_quality(A, M, U, extraction, q):
    """Independent full-operator checks; no eigen-residual cutoff is fitted."""
    rank = U.shape[1]
    values = np.asarray(extraction.get('ritz_values', []), dtype=float)
    result = dict(requested_rank=q, rank=int(rank), ritz_values=values.tolist(),
                  orthogonality_error=None, a_relative_ritz_residuals=[],
                  max_a_relative_ritz_residual=None,
                  projected_energy_remaining_ratio=None, valid=False)
    if not rank or len(values) != rank or not np.all(np.isfinite(U)):
        return result
    AU = A @ U
    MAU = np.column_stack([M @ column for column in AU.T])
    residual = MAU - U * values[None, :]
    denominator = _a_norms(A, MAU) + np.abs(values) * _a_norms(A, U)
    residuals = _a_norms(A, residual) / np.maximum(denominator, np.finfo(float).tiny)
    orth = float(la.norm(U.T @ AU - np.eye(rank), ord=2))
    result.update(orthogonality_error=orth,
                  a_relative_ritz_residuals=residuals.tolist(),
                  max_a_relative_ritz_residual=float(np.max(residuals)))
    valid = (rank == q and extraction.get('status') == 'accepted' and
             np.all(np.isfinite(values)) and np.all(values > 0.) and
             finite(orth) and orth <= ORTH_TOL and
             np.all(np.isfinite(MAU)) and np.all(np.isfinite(residuals)))
    result['valid'] = bool(valid)
    return json_safe(result)


def projection_quality(A, b, base, truth, directions):
    """Compare every supplied space at the same base-PCG initial guess."""
    try:
        U, AU, info = a_orthonormalize(A, directions)
        corrected, _ = galerkin_correction(A, b, base, U, AU)
        error, after = truth - base, truth - corrected
        energy = float(error @ (A @ error))
        ratio = float(after @ (A @ after) / energy) if energy > np.finfo(float).tiny else None
        return json_safe(dict(rank=int(U.shape[1]), basis_stats=info,
                             valid=bool(finite(info['orthogonality_error']) and
                                        info['orthogonality_error'] <= ORTH_TOL and
                                        finite(energy) and energy >= 0. and
                                        (ratio is None or finite(ratio))),
                             base_error_energy=energy,
                             projected_energy_remaining_ratio=ratio))
    except (la.LinAlgError, ValueError, FloatingPointError) as error:
        return dict(rank=0, valid=False, projected_energy_remaining_ratio=None,
                    error=f'{type(error).__name__}: {error}')


def audit_cache(A, M, cache):
    actual_ad = A @ cache['D']
    actual_mad = np.column_stack([M @ column for column in actual_ad.T]) \
        if actual_ad.shape[1] else actual_ad.copy()
    tiny = np.finfo(float).tiny
    ad_error = float(la.norm(cache['AD'] - actual_ad) / max(la.norm(actual_ad), tiny))
    mad_error = float(la.norm(cache['MAD'] - actual_mad) / max(la.norm(actual_mad), tiny))
    return json_safe(dict(ad_relative_error=ad_error, mad_relative_error=mad_error,
                         threshold=CACHE_TOL, valid=bool(finite(ad_error) and finite(mad_error)
                         and ad_error <= CACHE_TOL and mad_error <= CACHE_TOL)))


def prepare_sequence(fields, events):
    """Create retrospective directions; never called by a timed method."""
    history, rows, truth = [], [], []
    directions = {q: [] for q in RANKS}
    policy = AMGSequencePreconditioner('refresh_galerkin', seed=events[0]['seed'] * 100)
    frozen = np.empty(((fields.shape[-1] - 1) ** 2, 0))
    for step, k in enumerate(fields):
        acquisition_start = time.perf_counter()
        A, b, _ = assemble_system(k)
        base = initial(A, b, history)
        M, _ = policy.for_matrix(A)
        x, metrics, cache = solve_pcg_cached(A, b, M, base, RTOL, 500, max_vectors=LIMIT)
        # Direct references must never feed subsequent source histories or
        # direction extraction, even after a failed source solve.
        history_reset = not bool(np.all(np.isfinite(x)))
        if history_reset:
            x = np.zeros_like(b)
        extraction = {}
        for q in RANKS:
            t = time.perf_counter()
            U, _, info = _safe_ritz(cache, q)
            elapsed = time.perf_counter() - t
            U.flags.writeable = False
            directions[q].append(U)
            extraction[str(q)] = dict(seconds=elapsed, stats=info)
        frozen_extraction = None
        if step == 0:
            t = time.perf_counter()
            frozen, _, info = _safe_ritz(cache, 4, previous.LIMIT)
            frozen.flags.writeable = False
            frozen_extraction = dict(seconds=time.perf_counter() - t, stats=info)
        history.append(x.copy())
        history = history[-4:]
        acquisition_seconds = time.perf_counter() - acquisition_start
        t = time.perf_counter()
        exact = sla.spsolve(A, b)
        direct_residual = float(la.norm(b - A @ exact) / la.norm(b))
        truth.append(exact)
        direct_seconds = time.perf_counter() - t
        t = time.perf_counter()
        quality = {}
        for q in RANKS:
            value = space_quality(A, M, directions[q][-1], extraction[str(q)]['stats'], q)
            value['projection'] = projection_quality(A, b, base, exact, directions[q][-1])
            value['projected_energy_remaining_ratio'] = value['projection']['projected_energy_remaining_ratio']
            quality[str(q)] = value
        frozen_quality = projection_quality(A, b, base, exact, frozen)
        cache_audit = audit_cache(A, M, cache) if step in (0, 10) else None
        quality_seconds = time.perf_counter() - t
        numerically_valid = bool(metrics['converged'] and finite(direct_residual)
            and direct_residual <= RTOL and all(v['valid'] for v in quality.values())
            and all(v['projection']['valid'] and v['projection']['rank'] == int(q)
                    for q, v in quality.items())
            and frozen_quality['rank'] == 4 and frozen_quality['valid']
            and (cache_audit is None or cache_audit['valid']))
        rows.append(dict(n=fields.shape[-1], seed=events[0]['seed'], step=step,
            acquisition_seconds=acquisition_seconds, direct_seconds=direct_seconds,
            quality_seconds=quality_seconds, source_pcg=metrics,
            source_solver_fallback=False, source_recovery_seconds=0.,
            source_failed=not metrics['converged'], source_history_reset=history_reset,
            capture_stats=cache['stats'], captured_vectors=int(cache['D'].shape[1]),
            segment_count=len(set(cache['stats']['segment_ids'])),
            extraction=extraction, frozen_initial_extraction=frozen_extraction,
            quality=quality, frozen4_projection=frozen_quality, cache_audit=cache_audit,
            direct_relative_residual=direct_residual, numerically_valid=numerically_valid,
            space_sha256={str(q): array_sha(directions[q][-1]) for q in RANKS}))
    return directions, truth, json_safe(rows)


def archive_path(role, n, seed):
    # Flat data/*.npz is already ignored by the repository.
    return ROOT / 'data' / f'stage8_{role}{n}_seed{seed}.npz'


def prepare_group(out, role, n, seeds):
    sequences = []
    for seed in seeds:
        fields, events = coefficient_sequence(n, STEPS, seed, CONTRAST, 'mixed')
        directions, truth, quality = prepare_sequence(fields, events)
        archive = archive_path(role, n, seed)
        if archive.exists():
            raise FileExistsError(archive)
        arrays = {f'U{q}_{step:02d}': directions[q][step]
                  for q in RANKS for step in range(STEPS)}
        arrays.update({f'truth_{step:02d}': value for step, value in enumerate(truth)})
        t = time.perf_counter()
        with archive.open('xb') as handle:
            np.savez(handle, **arrays)
        archive_digest = sha(archive)
        storage_seconds = time.perf_counter() - t
        sequences.append(dict(seed=seed, n=n, archive=str(archive.relative_to(ROOT)),
            archive_sha256=archive_digest, storage_seconds=storage_seconds,
            field_sha256=hashlib.sha256(fields.tobytes()).hexdigest(), systems=quality))
        save(out / 'reference_metadata.json', dict(role=role, n=n, sequences=sequences,
             reference_scope='Completed base-M trajectories; retrospective, RHS-accessible Ritz spaces, not optimal or causal.',
             acquisition_in_online_time=False, limit=LIMIT, ranks=list(RANKS)))
        print(json.dumps(dict(stage='reference_prepared', role=role, n=n, seed=seed,
                              valid=all(r['numerically_valid'] for r in quality))), flush=True)
    return read(out / 'reference_metadata.json')


def load_group(metadata):
    """Load all arrays before timing; mark spaces and truths read-only."""
    loaded = {}
    for sequence in metadata['sequences']:
        path = ROOT / sequence['archive']
        if sha(path) != sequence['archive_sha256']:
            raise RuntimeError(f'Reference archive changed: {path}')
        n, seed = sequence['n'], sequence['seed']
        fields, events = coefficient_sequence(n, STEPS, seed, CONTRAST, 'mixed')
        if hashlib.sha256(fields.tobytes()).hexdigest() != sequence['field_sha256']:
            raise RuntimeError('Reference coefficient sequence changed')
        directions, truth = {q: [] for q in RANKS}, []
        with np.load(path, allow_pickle=False) as archive:
            expected = {f'U{q}_{step:02d}' for q in RANKS for step in range(STEPS)}
            expected.update(f'truth_{step:02d}' for step in range(STEPS))
            if set(archive.files) != expected:
                raise RuntimeError('Reference archive array keys changed')
            for step in range(STEPS):
                for q in RANKS:
                    U = archive[f'U{q}_{step:02d}']
                    if U.ndim != 2 or U.shape[0] != (n-1)**2 or U.dtype != np.float64:
                        raise RuntimeError('Reference space has an invalid shape/dtype')
                    if array_sha(U) != sequence['systems'][step]['space_sha256'][str(q)]:
                        raise RuntimeError('Reference space digest changed')
                    U.flags.writeable = False
                    directions[q].append(U)
                value = archive[f'truth_{step:02d}']
                if value.shape != ((n-1)**2,) or not np.all(np.isfinite(value)):
                    raise RuntimeError('Invalid direct-reference vector')
                value.flags.writeable = False
                truth.append(value)
        loaded[seed] = dict(fields=fields, events=events, directions=directions,
                            truth=truth, metadata=sequence)
    return loaded


def run_reference_sequence(fields, events, setting, repeat, directions):
    """Time a complete path with preloaded retrospective U; accepts no truth."""
    history, rows, solutions = [], [], []
    q = setting['rank']
    if len(directions) != len(fields):
        raise ValueError('Exactly one supplied space per current system is required')
    policy = AMGSequencePreconditioner('refresh_galerkin', seed=events[0]['seed'] * 100)
    old = None
    for step, k in enumerate(fields):
        start = time.perf_counter()
        row = dict(method=setting['name'], n=fields.shape[-1], seed=events[0]['seed'],
            step=step, repeat=repeat, event=events[step]['event'], initial_step=step == 0,
            recovery_seconds=0., solver_fallback=False, cache_rejected=False,
            rank=0, requested_rank=q, orthogonality_error=0., capture_stats={},
            extraction={}, extraction_seconds=0., capture_requested=False,
            extraction_requested=False, solver_operator='base', basis_stats={},
            projection_info={}, old_space_rank=0, stored_space_rank=0,
            initial_space_sha256=None, reference_space_valid=False,
            retrospective_reference=True)
        t = time.perf_counter()
        A, b, _ = assemble_system(k)
        row['assembly_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        base = initial(A, b, history)
        row['guess_seconds'] = time.perf_counter() - t
        M, setup = policy.for_matrix(A)
        row['setup_seconds'] = setup['setup_seconds']
        event = is_event(k, old)
        row['refresh_event'] = event
        t = time.perf_counter()
        x0, operator = base, M
        U, AU = None, None
        try:
            U, AU, info = a_orthonormalize(A, directions[step])
            row['basis_stats'] = info
            orth = float(la.norm(U.T @ AU - np.eye(U.shape[1]), ord=2)) if U.shape[1] else 0.
            row.update(rank=int(U.shape[1]), orthogonality_error=orth)
            row['reference_space_valid'] = bool(U.shape[1] == q and finite(orth)
                and orth <= ORTH_TOL and info['status'] == 'accepted')
            if row['reference_space_valid']:
                x0, projection = galerkin_correction(A, b, base, U, AU)
                operator = balanced_preconditioner(M, U, AU)
                row.update(projection_info=projection, solver_operator='balanced')
            else:
                row['cache_rejected'] = True
        except (la.LinAlgError, ValueError, FloatingPointError) as error:
            row['cache_rejected'] = True
            row['space_error'] = f'{type(error).__name__}: {error}'
        row['correction_seconds'] = time.perf_counter() - t
        row['base_relative_residual'] = float(la.norm(b - A @ base) / la.norm(b))
        row['corrected_relative_residual'] = float(la.norm(b - A @ x0) / la.norm(b))
        cache = None
        if event:
            x, metrics, cache = solve_pcg_cached(A, b, operator, x0, RTOL, 500, max_vectors=0)
            row['capture_stats'] = cache['stats']
        else:
            x, metrics = solve_pcg(A, b, operator, x0, RTOL, 500)
        row.update({('solve_seconds' if key == 'seconds' else key): value
                    for key, value in metrics.items()})
        if not metrics['converged']:
            t = time.perf_counter()
            x = sla.spsolve(A, b)
            row['recovery_seconds'] = time.perf_counter() - t
            row['solver_fallback'] = True
        row['final_relative_residual'] = float(la.norm(b - A @ x) / la.norm(b))
        row['final_converged'] = row['final_relative_residual'] <= RTOL
        t = time.perf_counter()
        cache = None
        U, AU, operator = None, None, None
        history.append(x.copy())
        history = history[-4:]
        old = k
        row['maintenance_seconds'] = time.perf_counter() - t
        row['total_seconds'] = time.perf_counter() - start
        rows.append(row)
        solutions.append(x.copy())
    # Provenance checks are outside all measured paths and never affect U.
    for step, row in enumerate(rows):
        row['reference_space_sha256'] = array_sha(directions[step])
    print(json.dumps(dict(stage='sequence', method=setting['name'], n=fields.shape[-1],
        seed=events[0]['seed'], repeat=repeat, seconds=sum(r['total_seconds'] for r in rows),
        iterations=sum(r['iterations'] for r in rows),
        pcg_failures=sum(not r['converged'] for r in rows))), flush=True)
    return json_safe(rows), solutions


def rows_valid(rows):
    for row in rows:
        if not (row['converged'] and row['final_converged'] and
                not row['solver_fallback'] and not row['cache_rejected'] and
                finite(row['final_relative_residual']) and row['final_relative_residual'] <= RTOL):
            return False
        if row['method'].startswith('current_reference'):
            if not row['reference_space_valid'] or row['rank'] != row['requested_rank']:
                return False
        elif row['method'] == 'frozen_all_balanced4':
            if row['stored_space_rank'] != 4 or (row['step'] > 0 and row['rank'] != 4):
                return False
        if not finite(row['orthogonality_error']) or row['orthogonality_error'] > ORTH_TOL:
            return False
    return True


def complete_rows(rows, seeds, repeats=REPEATS):
    expected = {(s['name'], seed, repeat, step) for s in specs()
                for seed in seeds for repeat in range(repeats) for step in range(STEPS)}
    actual = {(r['method'], r['seed'], r['repeat'], r['step']) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError('Incomplete or duplicate prescribed measurements')


def evaluate_gate(stage_data):
    """One rank across both grids, every geometry; no pooled ratio gate."""
    all_valid = all(stage['numerically_valid'] for stage in stage_data.values())
    candidates = {}
    for q in RANKS:
        cases = []
        for n, stage in stage_data.items():
            summary = stage['summary']
            base = summary['baseline_scipy']['by_seed']
            ref = summary[f'current_reference{q}']['by_seed']
            for seed, base_seconds in base.items():
                ref_seconds = ref[seed]
                cases.append(dict(n=int(n), seed=int(seed), baseline_seconds=base_seconds,
                    reference_path_seconds=ref_seconds, time_ratio=ref_seconds / base_seconds,
                    construction_budget_seconds=.8 * base_seconds - ref_seconds,
                    construction_budget_seconds_per_system=(.8 * base_seconds - ref_seconds) / STEPS,
                    passes_time_gate=ref_seconds <= .8 * base_seconds))
        candidates[str(q)] = dict(cases=cases, passes_all_geometries=bool(all_valid and
            all(case['passes_time_gate'] for case in cases)))
    passing = [q for q in RANKS if candidates[str(q)]['passes_all_geometries']]
    return dict(numerically_valid=all_valid, candidates=candidates,
        selected_rank=passing[0] if passing else None,
        proceed_to_confirmation=bool(passing), training_authorized=False,
        rule='All five methods and references numerically valid; one fixed q passes <=0.8 SciPy on every geometry of both grids; prefer q4 if both pass. Zero construction budget is not training eligibility.')


def benchmark_group(out, role, n, seeds, loaded, freeze_hash, decision_hash=None):
    configuration = dict(n=n, seeds=seeds, repeats=REPEATS, specs=specs(), role=role,
                         freeze_sha256=freeze_hash, decision_sha256=decision_hash)
    value = dict(configuration=configuration, created_utc=stamp(),
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        scipy=previous.scipy.__version__, pyamg=previous.pyamg.__version__,
        device='cpu', blas_threads=1, rtol=RTOL, maxiter=500, steps=STEPS,
        contrast=CONTRAST, **provenance())
    value.update(rank=list(RANKS), reference_limit=LIMIT,
        timing='Full assembly through history maintenance with independent AMG/history; retrospective direction acquisition, all quality checks, direct truth, archive loading and hashes outside measured paths.',
        reference_metadata_sha256=sha(out / 'reference_metadata.json'),
        phase_preparation_sha256=sha(ROOT / 'results' / f'stage8_{role}_preparation.json'))
    save(out / 'manifest.json', value)
    rows, order = [], []
    candidates = specs()
    for seed in seeds:
        item = loaded[seed]
        for repeat in range(REPEATS):
            for index in np.random.default_rng(seed + repeat + n).permutation(len(candidates)):
                setting = candidates[int(index)]
                order.append(dict(seed=seed, repeat=repeat, method=setting['name']))
                if setting['policy'] == 'current_reference':
                    measured, solutions = run_reference_sequence(item['fields'], item['events'],
                        setting, repeat, item['directions'][setting['rank']])
                else:
                    measured, solutions = previous.run_sequence(item['fields'], item['events'], setting, repeat)
                for step, row in enumerate(measured):
                    A, _, _ = assemble_system(item['fields'][step])
                    row['energy_relative_error'] = energy_relative_error(A, solutions[step], item['truth'][step])
                rows.extend(measured)
                save(out / 'measurements.json', rows)
                save(out / 'execution_order.json', order)
    complete_rows(rows, seeds)
    check_provenance(value)
    summary = previous.aggregate(rows)
    metadata = read(out / 'reference_metadata.json')
    numeric = bool(rows_valid(rows) and all(system['numerically_valid']
        for sequence in metadata['sequences'] for system in sequence['systems']))
    save(out / 'summary.json', dict(numerically_valid=numeric, summary=summary))
    save(out / 'complete.json', dict(complete=True, rows=len(rows), finished_utc=stamp()))
    return dict(numerically_valid=numeric, summary=summary)


def cohorts(role):
    seeds = splits()['groups']['development' if role == 'development' else 'confirmation']
    return [(128, seeds), (256, seeds[:2])]


def preflight(role):
    for n, seeds in cohorts(role):
        out = ROOT / 'results' / f'stage8_{role}{n}'
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(out)
        for seed in seeds:
            if archive_path(role, n, seed).exists():
                raise FileExistsError(archive_path(role, n, seed))
    marker = ROOT / 'results' / f'stage8_{role}_preparation.json'
    if marker.exists():
        raise FileExistsError(marker)


def run_phase(role, freeze_hash, decision_hash=None):
    """Both grid preparations and all loads precede the first measurement."""
    preflight(role)
    groups, metadata = cohorts(role), {}
    for n, seeds in groups:
        out = ROOT / 'results' / f'stage8_{role}{n}'
        out.mkdir(parents=True, exist_ok=True)
        metadata[n] = prepare_group(out, role, n, seeds)
    prepared_utc = stamp()
    t = time.perf_counter()
    loaded = {n: load_group(metadata[n]) for n, _ in groups}
    load_seconds = time.perf_counter() - t
    save(ROOT / 'results' / f'stage8_{role}_preparation.json', dict(
        prepared_utc=prepared_utc, all_loaded_utc=stamp(), loading_seconds=load_seconds,
        all_grids_prepared_and_loaded_before_measurement=True,
        reference_metadata_sha256={str(n): sha(ROOT / 'results' / f'stage8_{role}{n}' / 'reference_metadata.json') for n, _ in groups}))
    check_provenance(read(ROOT / FREEZE))
    return {str(n): benchmark_group(ROOT / 'results' / f'stage8_{role}{n}', role,
        n, seeds, loaded[n], freeze_hash, decision_hash) for n, seeds in groups}


def phase_hashes(role):
    paths = [ROOT / 'results' / f'stage8_{role}_preparation.json']
    for n, _ in cohorts(role):
        paths.extend(sorted((ROOT / 'results' / f'stage8_{role}{n}').glob('*.json')))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def development():
    for path in (ROOT / FREEZE, ROOT / DECISION, ROOT / COMPLETE):
        if path.exists():
            raise FileExistsError(path)
    preflight('development')
    frozen = dict(created_utc=stamp(), specs=specs(), ranks=list(RANKS), reference_limit=LIMIT,
        repeats=REPEATS, orthogonality_threshold=ORTH_TOL, cache_threshold=CACHE_TOL,
        development_cohorts=dict(cohorts('development')),
        confirmation_cohorts=dict(cohorts('confirm')),
        eigen_residual_gate=None, fixed_baseline='baseline_scipy', **provenance())
    save(ROOT / FREEZE, frozen)
    stage_data = run_phase('development', sha(ROOT / FREEZE))
    decision = dict(created_utc=stamp(), freeze_sha256=sha(ROOT / FREEZE),
                    development_sha256=phase_hashes('development'), **evaluate_gate(stage_data))
    save(ROOT / DECISION, decision)
    if not decision['proceed_to_confirmation']:
        save(ROOT / COMPLETE, dict(complete=True, status='stopped_after_development',
            finished_utc=stamp(), confirmation_run=False, selected_rank=None,
            training_authorized=False, freeze_sha256=sha(ROOT / FREEZE),
            decision_sha256=sha(ROOT / DECISION)))
    print(json.dumps(dict(stage='development_decision', **evaluate_gate(stage_data))), flush=True)


def verify_decision():
    frozen, decision = read(ROOT / FREEZE), read(ROOT / DECISION)
    check_provenance(frozen)
    if frozen['specs'] != specs() or frozen['ranks'] != list(RANKS):
        raise RuntimeError('Frozen method specification changed')
    if decision['freeze_sha256'] != sha(ROOT / FREEZE):
        raise RuntimeError('Development freeze changed')
    if decision['development_sha256'] != phase_hashes('development'):
        raise RuntimeError('Development artifacts changed')
    stage_data = {}
    for n, seeds in cohorts('development'):
        out = ROOT / 'results' / f'stage8_development{n}'
        rows = read(out / 'measurements.json')
        complete_rows(rows, seeds)
        metadata = read(out / 'reference_metadata.json')
        stage_data[str(n)] = dict(summary=previous.aggregate(rows),
            numerically_valid=bool(rows_valid(rows) and all(system['numerically_valid']
                for sequence in metadata['sequences'] for system in sequence['systems'])))
    for key, value in evaluate_gate(stage_data).items():
        if decision[key] != value:
            raise RuntimeError(f'Development decision mismatch: {key}')
    return decision


def confirm():
    if (ROOT / COMPLETE).exists():
        raise FileExistsError(ROOT / COMPLETE)
    decision = verify_decision()
    if not decision['proceed_to_confirmation']:
        raise RuntimeError('Prespecified stopping rule forbids confirmation')
    stage_data = run_phase('confirm', sha(ROOT / FREEZE), sha(ROOT / DECISION))
    assessment = evaluate_gate(stage_data)
    q = decision['selected_rank']
    passed = assessment['candidates'][str(q)]['passes_all_geometries']
    save(ROOT / COMPLETE, dict(complete=True, status='confirmation_finished',
        finished_utc=stamp(), confirmation_run=True, selected_rank=q,
        selected_rank_confirmed=passed, training_authorized=False,
        assessment=assessment, confirmation_sha256=phase_hashes('confirm'),
        freeze_sha256=sha(ROOT / FREEZE), decision_sha256=sha(ROOT / DECISION)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['splits', 'development', 'confirm'])
    args = parser.parse_args()
    splits()
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'development':
            development()
        elif args.stage == 'confirm':
            confirm()


if __name__ == '__main__':
    main()
