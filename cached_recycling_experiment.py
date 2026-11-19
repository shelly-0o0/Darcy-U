#!/usr/bin/env python3
"""Stage 5: causal low-rank recycling from already-paid fixed-PCG actions."""
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

ROOT = Path(__file__).resolve().parent
if (ROOT / 'vendor').exists():
    sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
import pyamg
from threadpoolctl import threadpool_limits
from fields import coefficient_sequence
from numerics import assemble_system, solve_pcg, energy_relative_error
from sequence_baselines import AMGSequencePreconditioner, galerkin_initial_guess
from balanced_augmentation import a_orthonormalize, balanced_preconditioner, galerkin_correction, extract_slow_ritz
from cached_pcg import solve_pcg_cached, cached_slow_ritz

RTOL, STEPS, CONTRAST, LIMIT = 1e-8, 20, 1e4, 24
SOURCES = ['fields.py', 'numerics.py', 'sequence_baselines.py', 'balanced_augmentation.py',
           'cached_pcg.py', 'cached_recycling_experiment.py']
PROTOCOL = 'docs/stage5_protocol_zh.md'
SPLITS = 'data/stage5_splits.json'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def fresh(name):
    p = ROOT / 'results' / name
    if p.exists() and any(p.iterdir()): raise FileExistsError(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def geometry_hash(seed):
    f, _ = coefficient_sequence(32, STEPS, seed, CONTRAST, 'mixed')
    return hashlib.sha256((f > 1).tobytes()).hexdigest()


def splits():
    p = ROOT / SPLITS
    if p.exists(): return read(p)
    old = read(ROOT / 'data/stage4_splits.json')
    excluded = sorted(set(old['excluded_seeds'] + sum(old['groups'].values(), [])))
    seen = {geometry_hash(s) for s in excluded}
    result = dict(excluded_seeds=excluded, groups={}, geometry_sha256={}, skipped_duplicates=[])
    for name, start, count in [('development', 43001, 3), ('confirmation', 43011, 5)]:
        seeds = []
        while len(seeds) < count:
            digest = geometry_hash(start)
            if digest in seen: result['skipped_duplicates'].append(dict(seed=start, group=name))
            else:
                seen.add(digest); seeds.append(start)
                result['geometry_sha256'][str(start)] = digest
            start += 1
        result['groups'][name] = seeds
    save(p, result)
    return result


def provenance():
    return dict(source_sha256={f: sha(ROOT/f) for f in SOURCES},
                protocol_sha256=sha(ROOT/PROTOCOL), splits_sha256=sha(ROOT/SPLITS))


def check_provenance(record):
    for key, value in provenance().items():
        if record[key] != value: raise RuntimeError(f'Frozen {key} changed')


def manifest(configuration):
    return dict(configuration=configuration, created_utc=stamp(), python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__, pyamg=pyamg.__version__,
        device='cpu', blas_threads=1, rtol=RTOL, maxiter=500, steps=STEPS, contrast=CONTRAST,
        timing='All assembly, history, hierarchy, directions, capture/copies, solve, true checks, recovery and maintenance included. Offline references and diagnostics excluded.',
        **provenance())


def spec(name, source='none', rank=0, interface='warm', custom=True):
    return dict(name=name, source=source, rank=rank, interface=interface, event_custom=custom)


def specs():
    return [spec('baseline_scipy', custom=False), spec('baseline_event_custom'),
        spec('capture_only', 'capture'),
        *[spec(f'cached_balanced{q}', 'cached', q, 'balanced') for q in [1, 2, 4]],
        spec('explicit_balanced2', 'explicit', 2, 'balanced'), spec('cached_warm2', 'cached', 2)]


def is_event(k, old):
    return bool(old is None or float(np.max(np.abs(np.log(k / old)))) >= np.log(10.))


def initial(A, b, history):
    return galerkin_initial_guess(A, b, history, max_vectors=4)[0] if history else np.zeros_like(b)


def run_sequence(fields, events, setting, repeat):
    history, rows, solutions = [], [], []
    recycle = np.empty(((fields.shape[-1]-1)**2, 0))
    policy = AMGSequencePreconditioner('refresh_galerkin', seed=events[0]['seed']*100)
    old = None
    for step, k in enumerate(fields):
        start = time.perf_counter()
        row = dict(method=setting['name'], n=fields.shape[-1], seed=events[0]['seed'], step=step,
            repeat=repeat, event=events[step]['event'], recovery_seconds=0., solver_fallback=False,
            cache_rejected=False, rank=0, orthogonality_error=0., capture_stats={}, extraction={}, extraction_seconds=0.)
        t = time.perf_counter(); A, b, _ = assemble_system(k)
        row['assembly_seconds'] = time.perf_counter()-t
        t = time.perf_counter(); base = initial(A, b, history)
        row['guess_seconds'] = time.perf_counter()-t
        M, setup = policy.for_matrix(A); row['setup_seconds'] = setup['setup_seconds']
        event = is_event(k, old); row['refresh_event'] = event
        t = time.perf_counter(); x0, operator = base, M
        if not event and recycle.shape[1]:
            U, AU, info = a_orthonormalize(A, recycle)
            x0, _ = galerkin_correction(A, b, base, U, AU)
            if setting['interface'] == 'balanced': operator = balanced_preconditioner(M, U, AU)
            row.update(rank=info['rank'], orthogonality_error=info['orthogonality_error'])
        row['correction_seconds'] = time.perf_counter()-t
        row['base_relative_residual'] = float(la.norm(b-A@base)/la.norm(b))
        row['corrected_relative_residual'] = float(la.norm(b-A@x0)/la.norm(b))
        cache = None
        if event and setting['event_custom']:
            capacity = LIMIT if setting['source'] != 'none' else 0
            x, metrics, cache = solve_pcg_cached(A,b,M,x0,RTOL,500,max_vectors=capacity)
            row['capture_stats'] = cache['stats']
        else:
            x, metrics = solve_pcg(A,b,operator,x0,RTOL,500)
        row.update({('solve_seconds' if key=='seconds' else key):value for key,value in metrics.items()})
        if not metrics['converged']:
            t = time.perf_counter(); x = sla.spsolve(A,b)
            row['recovery_seconds'] = time.perf_counter()-t; row['solver_fallback'] = True
        row['final_relative_residual'] = float(la.norm(b-A@x)/la.norm(b))
        row['final_converged'] = row['final_relative_residual'] <= RTOL
        t = time.perf_counter()
        if event and setting['source'] in ('cached','explicit'):
            extraction_started = time.perf_counter()
            if setting['source'] == 'cached':
                recycle, _, info = cached_slow_ritz(cache['D'],cache['AD'],cache['MAD'],q=setting['rank'])
            else:
                recycle, _, info = extract_slow_ritz(A,M,cache['D'],q=setting['rank'],max_basis=LIMIT)
            row['extraction'] = info
            row['cache_rejected'] = info['status'] not in ('accepted','empty_basis')
            row['extraction_seconds'] = time.perf_counter()-extraction_started
        # Include releasing captured arrays and previous vectors in online cost.
        cache = None
        history.append(x.copy()); history = history[-4:]; old = k
        row['maintenance_seconds'] = time.perf_counter()-t
        row['total_seconds'] = time.perf_counter()-start
        rows.append(row); solutions.append(x.copy())
    print(json.dumps(dict(stage='sequence',method=setting['name'],n=fields.shape[-1],seed=events[0]['seed'],
        repeat=repeat,seconds=sum(r['total_seconds'] for r in rows),iterations=sum(r['iterations'] for r in rows),
        pcg_failures=sum(not r['converged'] for r in rows),cache_rejections=sum(r['cache_rejected'] for r in rows))),flush=True)
    return rows, solutions


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows: grouped[(r['method'],r['seed'],r['repeat'])].append(r)
    result = {}
    for method in sorted({r['method'] for r in rows}):
        local = [r for r in rows if r['method']==method]
        by_seed = {str(s):median(sum(r['total_seconds'] for r in v) for (m,seed,_),v in grouped.items()
                    if m==method and seed==s) for s in sorted({r['seed'] for r in local})}
        result[method] = dict(by_seed=by_seed,median_seconds=median(by_seed.values()),
            all_accurate=all(r['final_converged'] for r in local),pcg_failures=sum(not r['converged'] for r in local),
            cache_rejections=sum(r['cache_rejected'] for r in local))
    return result


def benchmark(name,n,seeds,candidates,repeats,role,selection_hash=None):
    out = fresh(name)
    save(out/'manifest.json',manifest(dict(n=n,seeds=seeds,specs=candidates,repeats=repeats,
         role=role,selection_sha256=selection_hash)))
    rows, references = [], []
    for seed in seeds:
        fields, events = coefficient_sequence(n,STEPS,seed,CONTRAST,'mixed')
        t = time.perf_counter(); truth, residuals = [], []
        for k in fields:
            A,b,_=assemble_system(k); x=sla.spsolve(A,b)
            truth.append(x); residuals.append(float(la.norm(b-A@x)/la.norm(b)))
        references.append(dict(seed=seed,seconds=time.perf_counter()-t,field_sha256=hashlib.sha256(fields.tobytes()).hexdigest(),relative_residuals=residuals))
        save(out/'references.json',references)
        for repeat in range(repeats):
            for index in np.random.default_rng(seed+repeat+n).permutation(len(candidates)):
                measured,xs=run_sequence(fields,events,candidates[int(index)],repeat)
                for step,row in enumerate(measured):
                    A,_,_=assemble_system(fields[step]); row['energy_relative_error']=energy_relative_error(A,xs[step],truth[step])
                rows.extend(measured);save(out/'measurements.json',rows)
    check_provenance(read(out/'manifest.json'))
    save(out/'summary.json',aggregate(rows))
    save(out/'complete.json',dict(complete=True,rows=len(rows),finished_utc=stamp()))
    return out


def audit():
    out=fresh('stage5_cache_audit'); seeds=splits()['groups']['development']
    save(out/'manifest.json',manifest(dict(n=128,seeds=seeds,steps_observed=[0,6,10,15],role='offline cache identities and same-span Ritz audit')))
    rows=[]
    for seed in seeds:
        fields,events=coefficient_sequence(128,STEPS,seed,CONTRAST,'mixed')
        policy=AMGSequencePreconditioner('refresh_galerkin',seed=seed*100); history=[]
        for step,k in enumerate(fields):
            A,b,_=assemble_system(k);base=initial(A,b,history);M,_=policy.for_matrix(A)
            if step in [0,6,10,15]:
                x,metrics,cache=solve_pcg_cached(A,b,M,base,RTOL,500,max_vectors=LIMIT)
                D,AD,MAD=cache['D'],cache['AD'],cache['MAD']
                exactAD=A@D;exactMAD=M@exactAD
                rel=lambda v,w:float(la.norm(v-w)/max(la.norm(w),np.finfo(float).tiny))
                row=dict(seed=seed,step=step,converged=metrics['converged'],relative_residual=metrics['relative_residual'],
                    captured_vectors=D.shape[1],capture_stats=cache['stats'],ad_relative_error=rel(AD,exactAD),
                    mad_relative_error=rel(MAD,exactMAD),ranks={})
                for q in [1,2,4]:
                    U,AU,cached=cached_slow_ritz(D,AD,MAD,q=q)
                    V,AV,explicit=extract_slow_ritz(A,M,D,q=q,max_basis=LIMIT)
                    rank_ok=U.shape[1]==V.shape[1]==q
                    eig_error=rel(np.asarray(cached['ritz_values']),np.asarray(explicit['ritz_values'])) if rank_ok else None
                    # Re-orthonormalize under actual A for an independent space comparison.
                    Ua,AUa,_=a_orthonormalize(A,U)
                    Va,AVa,_=a_orthonormalize(A,V)
                    rank_ok=rank_ok and Ua.shape[1]==Va.shape[1]==q
                    singular=la.svdvals(Ua.T@AVa) if rank_ok else np.empty(0)
                    sine=float(np.sqrt(max(0.,1.-min(float(singular[-1]),1.)**2))) if rank_ok else None
                    row['ranks'][str(q)]=dict(cached=cached,explicit=explicit,rank_matches=rank_ok,
                        relative_ritz_value_error=eig_error,max_principal_angle_sine=sine,
                        passed=bool(rank_ok and eig_error<=1e-5 and sine<=1e-3))
                row['passed']=bool(metrics['converged'] and row['ad_relative_error']<=1e-7 and row['mad_relative_error']<=1e-7
                    and all(v['passed'] for v in row['ranks'].values()))
                rows.append(row);save(out/'measurements.json',rows)
                print(json.dumps(dict(stage='cache_audit',seed=seed,step=step,passed=row['passed'],ad_error=row['ad_relative_error'],mad_error=row['mad_relative_error'])),flush=True)
            else: x,metrics=solve_pcg(A,b,M,base,RTOL,500)
            if not metrics['converged']: x=sla.spsolve(A,b)
            history.append(x.copy());history=history[-4:]
    passed=all(r['passed'] for r in rows)
    save(out/'complete.json',dict(complete=True,cases=len(rows),all_passed=passed,finished_utc=stamp()))
    if not passed: raise RuntimeError('Cache equivalence failed; retain audit and stop before screen')


def check_audit():
    out=ROOT/'results/stage5_cache_audit'
    check_provenance(read(out/'manifest.json'))
    rows=read(out/'measurements.json');complete=read(out/'complete.json')
    expected={(s,t) for s in splits()['groups']['development'] for t in [0,6,10,15]}
    if not complete['all_passed'] or len(rows)!=12 or {(r['seed'],r['step']) for r in rows}!=expected or not all(r['passed'] for r in rows):
        raise RuntimeError('Cache audit incomplete or failed')
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(out.glob('*.json'))}


def screen():
    check_audit()
    out=benchmark('stage5_screen',128,splits()['groups']['development'],specs(),2,'development low-cost recycling screen')
    summary=read(out/'summary.json')
    eligible=[s for s in specs() if s['source']=='cached' and s['interface']=='balanced' and summary[s['name']]['all_accurate']
        and summary[s['name']]['pcg_failures']==0 and summary[s['name']]['cache_rejections']==0]
    if not eligible: raise RuntimeError('No eligible cached balanced candidate; preserve negative screen')
    chosen=min(eligible,key=lambda s:summary[s['name']]['median_seconds']);q=chosen['rank']
    selected=dict(created_utc=stamp(),candidate=chosen,confirmation_specs=[specs()[0],specs()[1],chosen,
        spec(f'explicit_balanced{q}','explicit',q,'balanced'),spec(f'cached_warm{q}','cached',q)],
        audit_sha256=check_audit(),screen_measurements_sha256=sha(out/'measurements.json'),screen_manifest_sha256=sha(out/'manifest.json'),
        rule='Fastest accurate failure-free nonrejected cached balanced rank 1/2/4 on development only; does not imply faster than baseline.',**provenance())
    save(out/'frozen_selection.json',selected)
    print(json.dumps(dict(stage='screen_selected',candidate=chosen)),flush=True)


def confirm():
    p=ROOT/'results/stage5_screen/frozen_selection.json';selected=read(p);check_provenance(selected)
    if check_audit()!=selected['audit_sha256']: raise RuntimeError('Frozen audit changed')
    for key,file in [('screen_measurements_sha256','measurements.json'),('screen_manifest_sha256','manifest.json')]:
        if sha(p.parent/file)!=selected[key]: raise RuntimeError('Frozen screen changed')
    rows=read(p.parent/'measurements.json')
    expected={(s['name'],seed,r,t) for s in specs() for seed in splits()['groups']['development'] for r in range(2) for t in range(STEPS)}
    if len(rows)!=len(expected) or {(r['method'],r['seed'],r['repeat'],r['step']) for r in rows}!=expected: raise RuntimeError('Incomplete screen')
    seeds=splits()['groups']['confirmation']
    for n,cohort in [(128,seeds),(256,seeds[:2])]:
        benchmark(f'stage5_confirm{n}',n,cohort,selected['confirmation_specs'],3,'frozen mechanism confirmation',sha(p))
    save(ROOT/'results/stage5_complete.json',dict(complete=True,finished_utc=stamp(),selection_sha256=sha(p)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['splits','audit','screen','confirm'])
    args=parser.parse_args();splits()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='audit': audit()
        elif args.stage=='screen': screen()
        elif args.stage=='confirm': confirm()


if __name__=='__main__': main()
