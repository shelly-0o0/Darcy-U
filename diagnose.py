#!/usr/bin/env python3
"""Validation-only performance and spectral-capacity diagnostics.

The original pilot runner and numerical implementation are unchanged.
All actual candidate construction/check/repair costs are included online.
The separately labelled zero-cost arithmetic is a diagnostic counterfactual,
not a measured deployable solver and not a bound over all possible spaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / 'vendor').exists():
    sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import torch
import pyamg
from threadpoolctl import threadpool_limits

from fields import coefficient_sequence
from numerics import (assemble_system, build_patches, build_preconditioner,
    patch_spectral_basis, certify_space, solve_pcg, energy_relative_error)
from optimized_numerics import build_patches_cached, build_preconditioner_sparse
from exact_increment import spectral_increment, diagnose_space
from learning import load_model, predict_directions, field_hash
from run_experiment import relevant_coefficient

METHODS = ['amg', 'dense_spectral_reuse', 'sparse_spectral_reuse',
           'sparse_spectral_increment4', 'sparse_spectral_increment_all',
           'sparse_unet']


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def save_records(path, records):
    save_json(Path(path).with_suffix('.json'), records)
    if not records:
        return
    keys = sorted(set().union(*(r.keys() for r in records)))
    with Path(path).with_suffix('.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, keys, lineterminator='\n')
        w.writeheader()
        for row in records:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})


def geometry_hash(seed, steps):
    # Compare geometric occupancy independently of contrast and fine resolution.
    k, _ = coefficient_sequence(32, steps, seed, 1e4, 'mixed')
    return hashlib.sha256((k > 1).tobytes()).hexdigest()


def validate_splits(args):
    data = json.loads((ROOT / 'data/pilot.manifest.json').read_text())
    prior = json.loads((ROOT / 'results/pilot_cpu/manifest.json').read_text())
    excluded = data['train_seeds'] + data['val_seeds'] + prior['configuration']['test_seeds']
    new_seeds = sorted(set(args.validation_seeds + args.event_seeds))
    if set(excluded) & set(new_seeds):
        raise ValueError('New validation seeds overlap with previous data')
    old_hashes = {geometry_hash(s, args.steps) for s in excluded}
    new_hashes = {str(s): geometry_hash(s, args.steps) for s in new_seeds}
    if old_hashes & set(new_hashes.values()):
        raise ValueError('Repeated geometric occupancy across old/new datasets')
    if len(set(new_hashes.values())) != len(new_seeds):
        raise ValueError('Repeated geometric occupancy within new validation')
    return {'role': 'development validation, never a new held-out test',
            'prior_seeds_excluded': excluded, 'new_geometry_sha256': new_hashes,
            'timed_seeds_are_subset_of_event_seeds': set(args.validation_seeds) <= set(args.event_seeds)}


def make_manifest(args):
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    return {'configuration': vars(args), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'python': sys.version, 'platform': platform.platform(), 'cpu': cpu,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
            'pyamg': pyamg.__version__, 'cuda_available': torch.cuda.is_available(),
            'blas_threads': 1, 'torch_threads': args.torch_threads,
            'splits': validate_splits(args),
            'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                ['fields.py', 'numerics.py', 'learning.py', 'run_experiment.py',
                 'optimized_numerics.py', 'exact_increment.py', 'diagnose.py']},
            'checkpoint_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for s in args.audit_model_seeds for p in [ROOT / f'checkpoints/unet_seed{s}.pt']},
            'protocol_sha256': hashlib.sha256((ROOT / 'docs/diagnostic_protocol_zh.md').read_bytes()).hexdigest(),
            'acceptance_tolerance': 5e-10,
            'gate': 'Measured full online time <= 0.8 times the fastest classical comparator on each timed validation geometry. Three geometries are a screen, not statistical evidence.',
            'timing': 'All online construction, checks, factorization, assembly, PCG and recovery included. Reference solves, event-only auditing, plotting and profiling are separate.',
            'counterfactual': 'Subtracting measured construction/check costs holds the observed spaces and solve trajectory fixed. Not a deployable method or a bound on all possible learned spaces.'}


def run_sequence(fields, events, method, args, model, repeat):
    old_patches = None
    spaces = anchors = anchor_cells = bounds = None
    previous_solution = None
    records, solutions = [], []
    sparse = method.startswith('sparse_')
    for step, k in enumerate(fields):
        start = time.perf_counter()
        row = {'method': method, 'sequence_seed': events[0]['seed'], 'repeat': repeat,
               'step': step, 'event': events[step]['event'], 'change_fraction': events[step]['change_fraction'],
               'model_seed': args.model_seed if method == 'sparse_unet' else -1,
               'assembly_seconds': 0., 'patch_seconds': 0., 'reuse_seconds': 0.,
               'spectral_seconds': 0., 'candidate_seconds': 0., 'check_seconds': 0.,
               'fallback_seconds': 0., 'setup_seconds': 0., 'direct_fallback_seconds': 0.,
               'reused_patches': 0, 'cached_patch_matrices': 0, 'old_checks': 0,
               'old_check_passes': 0, 'candidate_checks': 0, 'candidate_passes': 0,
               'spectral_repairs': 0, 'added_directions': 0, 'coarse_dimension': 0,
               'solver_fallback': False, 'check_diagnostics': [], 'candidate_metadata': [],
               'rtol': args.rtol}
        t = time.perf_counter()
        A, b, _ = assemble_system(k)
        row['assembly_seconds'] = time.perf_counter() - t
        if method == 'amg':
            t = time.perf_counter()
            # Same deterministic setup on repeated runs; no tuning on timings.
            np.random.seed(events[0]['seed'] * 100 + step)
            ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric',
                presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                postsmoother=('gauss_seidel', {'sweep': 'symmetric'}))
            pre = ml.aspreconditioner(cycle='V')
            row['setup_seconds'] = time.perf_counter() - t
            row['coarse_dimension'] = int(ml.levels[-1].A.shape[0])
        else:
            t = time.perf_counter()
            builder = build_patches_cached if sparse else build_patches
            patches = builder(k, A, args.core, args.overlap, old_patches)
            row['patch_seconds'] = time.perf_counter() - t
            if old_patches is not None:
                row['cached_patch_matrices'] = sum(bool(getattr(p, '_matrix_reused', False)) for p in patches)
            if spaces is None:
                t = time.perf_counter()
                spaces = [patch_spectral_basis(p, args.gamma, 2 * args.gamma) for p in patches]
                row['spectral_seconds'] = time.perf_counter() - t
                anchors = [relevant_coefficient(k, p) for p in patches]
                anchor_cells = [p.coefficient.copy() for p in patches]
                bounds = [2 * args.gamma] * len(patches)
            else:
                needs = []
                for i, p in enumerate(patches):
                    t = time.perf_counter()
                    region = relevant_coefficient(k, p)
                    ratio = region / anchors[i]
                    lower = bounds[i] * float(ratio.min() / ratio.max())
                    row['reuse_seconds'] += time.perf_counter() - t
                    if lower >= args.gamma * (1 - 1e-12):
                        row['reused_patches'] += 1
                        continue
                    t = time.perf_counter()
                    verdict = certify_space(p.A_N, p.B, spaces[i], args.gamma)
                    row['check_seconds'] += time.perf_counter() - t
                    row['old_checks'] += 1
                    row['check_diagnostics'].append({'kind': 'old', 'patch': i, **verdict})
                    if verdict['passed']:
                        row['old_check_passes'] += 1
                        anchors[i], anchor_cells[i], bounds[i] = region, p.coefficient.copy(), args.gamma
                    else:
                        needs.append(i)
                if needs and method in ('sparse_spectral_increment4', 'sparse_spectral_increment_all', 'sparse_unet'):
                    t = time.perf_counter()
                    if method == 'sparse_unet':
                        candidates = predict_directions(model, [patches[i] for i in needs],
                            [anchor_cells[i] for i in needs], [spaces[i] for i in needs], 'cpu')
                    else:
                        budget = 4 if method.endswith('4') else None
                        proposals = [spectral_increment(patches[i], spaces[i], args.gamma, budget=budget) for i in needs]
                        candidates = [v[0] for v in proposals]
                        row['candidate_metadata'] = [{'patch': i, **v[1]} for i, v in zip(needs, proposals)]
                    row['candidate_seconds'] = time.perf_counter() - t
                    for i, candidate in zip(needs, candidates):
                        p = patches[i]
                        t = time.perf_counter()
                        verdict = certify_space(p.A_N, p.B, candidate, args.gamma)
                        row['check_seconds'] += time.perf_counter() - t
                        row['candidate_checks'] += 1
                        row['check_diagnostics'].append({'kind': 'candidate', 'patch': i, **verdict})
                        if verdict['passed']:
                            row['candidate_passes'] += 1
                            row['added_directions'] += candidate.shape[1] - spaces[i].shape[1]
                            spaces[i] = candidate
                            anchors[i], anchor_cells[i], bounds[i] = relevant_coefficient(k, p), p.coefficient.copy(), args.gamma
                        else:
                            t = time.perf_counter()
                            spaces[i] = patch_spectral_basis(p, args.gamma, 2 * args.gamma)
                            row['fallback_seconds'] += time.perf_counter() - t
                            row['spectral_repairs'] += 1
                            anchors[i], anchor_cells[i], bounds[i] = relevant_coefficient(k, p), p.coefficient.copy(), 2 * args.gamma
                else:
                    for i in needs:
                        p = patches[i]
                        t = time.perf_counter()
                        spaces[i] = patch_spectral_basis(p, args.gamma, 2 * args.gamma)
                        row['fallback_seconds'] += time.perf_counter() - t
                        row['spectral_repairs'] += 1
                        anchors[i], anchor_cells[i], bounds[i] = relevant_coefficient(k, p), p.coefficient.copy(), 2 * args.gamma
            factory = build_preconditioner_sparse if sparse else build_preconditioner
            pre, setup = factory(A, patches, spaces)
            row.update(setup)
            old_patches = patches
        solution, metrics = solve_pcg(A, b, pre, previous_solution, args.rtol, args.maxiter)
        row.update({('solve_seconds' if key == 'seconds' else key): val for key, val in metrics.items()})
        if not metrics['converged']:
            t = time.perf_counter()
            solution = sla.spsolve(A, b)
            row['direct_fallback_seconds'] = time.perf_counter() - t
            row['solver_fallback'] = True
        row['final_relative_residual'] = float(np.linalg.norm(b - A @ solution) / np.linalg.norm(b))
        row['final_converged'] = row['final_relative_residual'] <= args.rtol
        row['total_seconds'] = time.perf_counter() - start
        # Both numbers are explicitly hypothetical, with this solve trajectory fixed.
        row['hypothetical_free_candidate_seconds'] = row['total_seconds'] - row['candidate_seconds']
        row['hypothetical_free_all_space_work_seconds'] = row['total_seconds'] - sum(
            row[key] for key in ['spectral_seconds', 'candidate_seconds', 'check_seconds', 'fallback_seconds'])
        records.append(row)
        solutions.append(solution.copy())
        previous_solution = solution.copy()
    print(json.dumps({'stage': 'sequence_complete', 'method': method, 'seed': events[0]['seed'],
        'repeat': repeat, 'seconds': sum(r['total_seconds'] for r in records),
        'candidate_passes': sum(r['candidate_passes'] for r in records),
        'max_residual': max(r['final_relative_residual'] for r in records)}), flush=True)
    return records, solutions


def benchmark(args, out, model):
    all_records = []
    for seed in args.validation_seeds:
        fields, events = coefficient_sequence(args.n, args.steps, seed, args.contrast, 'mixed')
        np.savez_compressed(out / f'fields_seed{seed}.npz', k=fields)
        save_json(out / f'events_seed{seed}.json', events)
        references = []
        t = time.perf_counter()
        for k in fields:
            A, b, _ = assemble_system(k)
            x = sla.spsolve(A, b)
            references.append((x, float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))))
        save_json(out / f'reference_seed{seed}.json', {'seconds': time.perf_counter() - t,
            'max_residual': max(r[1] for r in references), 'sequence_sha256': field_hash(fields)})
        for repeat in range(args.repeats):
            order = list(args.methods)
            np.random.default_rng(seed + repeat).shuffle(order)
            for method in order:
                records, solutions = run_sequence(fields, events, method, args, model, repeat)
                for step, (row, x) in enumerate(zip(records, solutions)):
                    A, _, _ = assemble_system(fields[step])
                    row['energy_relative_error'] = energy_relative_error(A, x, references[step][0])
                    row['reference_relative_residual'] = references[step][1]
                all_records.extend(records)
                save_records(out / 'measurements', all_records)
    return all_records


def event_audit(args, out, models):
    """Shared traditional-reuse trajectory; no measured solve timing claims."""
    audit, counts = [], []
    start = time.perf_counter()
    for seed in args.event_seeds:
        fields, events = coefficient_sequence(args.n, args.steps, seed, args.contrast, 'mixed')
        previous_patches = None
        spaces = anchors = anchor_cells = bounds = None
        count = {'sequence_seed': seed, 'steps': args.steps, 'patch_visits': 0,
                 'old_checks': 0, 'trigger_events': 0, 'sequence_sha256': field_hash(fields)}
        for step, k in enumerate(fields):
            A, _, _ = assemble_system(k)
            patches = build_patches_cached(k, A, args.core, args.overlap, previous_patches)
            count['patch_visits'] += len(patches)
            if spaces is None:
                spaces = [patch_spectral_basis(p, args.gamma, 2 * args.gamma) for p in patches]
                anchors = [relevant_coefficient(k, p) for p in patches]
                anchor_cells = [p.coefficient.copy() for p in patches]
                bounds = [2 * args.gamma] * len(patches)
            else:
                for i, p in enumerate(patches):
                    region = relevant_coefficient(k, p)
                    ratio = region / anchors[i]
                    if bounds[i] * float(ratio.min() / ratio.max()) >= args.gamma * (1 - 1e-12):
                        continue
                    count['old_checks'] += 1
                    verdict = certify_space(p.A_N, p.B, spaces[i], args.gamma)
                    if verdict['passed']:
                        anchors[i], anchor_cells[i], bounds[i] = region, p.coefficient.copy(), args.gamma
                        continue
                    count['trigger_events'] += 1
                    record = {'sequence_seed': seed, 'step': step, 'patch': i, 'event': events[step]['event'],
                              'old_dimension': spaces[i].shape[1], 'old_check': verdict,
                              'old_spectrum': diagnose_space(p.A_N, p.B, spaces[i], args.gamma),
                              'candidates': {}}
                    for name, budget in [('spectral4', 4), ('spectral_all', None)]:
                        candidate, metadata = spectral_increment(p, spaces[i], args.gamma, budget=budget)
                        check = certify_space(p.A_N, p.B, candidate, args.gamma)
                        record['candidates'][name] = {'metadata': metadata, 'check': check}
                    for model_seed, model in models.items():
                        t = time.perf_counter()
                        candidate = predict_directions(model, [p], [anchor_cells[i]], [spaces[i]], 'cpu')[0]
                        prediction_seconds = time.perf_counter() - t
                        record['candidates'][f'unet_{model_seed}'] = {'prediction_seconds': prediction_seconds,
                            'added_rank': candidate.shape[1] - spaces[i].shape[1],
                            'check': certify_space(p.A_N, p.B, candidate, args.gamma)}
                    audit.append(record)
                    # Every candidate sees the same old space. None changes the audit trajectory.
                    spaces[i] = patch_spectral_basis(p, args.gamma, 2 * args.gamma)
                    anchors[i], anchor_cells[i], bounds[i] = region, p.coefficient.copy(), 2 * args.gamma
            previous_patches = patches
        counts.append(count)
        save_json(out / 'event_audit.json', {'role': 'offline shared-trajectory diagnosis',
            'elapsed_seconds': time.perf_counter() - start, 'sequence_counts': counts, 'events': audit})
        print(json.dumps({'stage': 'event_audit', **count}), flush=True)
    return audit


def profile_preconditioner(args, out):
    """Separate microprofile; instrumentation is absent from timed benchmarks."""
    k, _ = coefficient_sequence(args.n, args.steps, args.validation_seeds[0], args.contrast, 'mixed')
    A, b, _ = assemble_system(k[0])
    p_dense = build_patches(k[0], A, args.core, args.overlap)
    p_sparse = build_patches_cached(k[0], A, args.core, args.overlap)
    z = [patch_spectral_basis(p, args.gamma, 2 * args.gamma) for p in p_dense]
    dense, d_setup = build_preconditioner(A, p_dense, z)
    sparse, s_setup = build_preconditioner_sparse(A, p_sparse, z, profile=True)
    rng = np.random.default_rng(1234)
    vectors = rng.standard_normal((12, A.shape[0]))
    dense @ vectors[0]; sparse @ vectors[0]
    records = []
    for repeat in range(5):
        order = [('dense', dense), ('sparse', sparse)]
        if repeat % 2:
            order.reverse()
        for name, op in order:
            t = time.perf_counter()
            outputs = [op @ v for v in vectors]
            elapsed = time.perf_counter() - t
            records.append({'backend': name, 'repeat': repeat, 'calls': len(vectors), 'seconds': elapsed})
    x = dense @ vectors[0]
    y = sparse @ vectors[0]
    save_json(out / 'preconditioner_profile.json', {'role': 'separate instrumented microprofile',
        'n': args.n, 'coarse_dimension': d_setup['coarse_dimension'], 'records': records,
        'relative_application_difference': float(np.linalg.norm(x - y) / np.linalg.norm(x)),
        'dense_setup': d_setup, 'sparse_setup_and_profile': s_setup})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='diagnostic_cpu')
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--core', type=int, default=16)
    p.add_argument('--overlap', type=int, default=2)
    p.add_argument('--contrast', type=float, default=1e4)
    p.add_argument('--gamma', type=float, default=.03)
    p.add_argument('--rtol', type=float, default=1e-8)
    p.add_argument('--maxiter', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--validation-seeds', type=int, nargs='+', default=[70001, 70002, 70003])
    p.add_argument('--event-seeds', type=int, nargs='+', default=list(range(70001, 70021)))
    p.add_argument('--model-seed', type=int, default=7)
    p.add_argument('--audit-model-seeds', type=int, nargs='+', default=[7, 17, 27])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--torch-threads', type=int, default=2)
    args = p.parse_args()
    out = ROOT / 'results' / args.run_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh run name; previous measurements will not be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api='blas'):
        torch.set_num_threads(args.torch_threads)
        models = {seed: load_model(ROOT / f'checkpoints/unet_seed{seed}.pt', 'cpu')
                  for seed in sorted(set(args.audit_model_seeds + [args.model_seed]))}
        with torch.no_grad():
            for model in models.values():
                model(torch.zeros(2, 6, 32, 32))
        save_json(out / 'manifest.json', make_manifest(args))
        benchmark(args, out, models[args.model_seed])
        # Run this only AFTER all primary timings have completed.
        profile_preconditioner(args, out)
        event_audit(args, out, models)
    save_json(out / 'complete.json', {'benchmark_complete': True, 'event_audit_complete': True,
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(json.dumps({'stage': 'complete', 'output': str(out)}), flush=True)


if __name__ == '__main__':
    main()
