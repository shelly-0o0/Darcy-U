#!/usr/bin/env python3
"""Stage 6: causal current-event reuse, bounded updates and refresh timing."""
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
from event_space_update import update_slow_space

RTOL, STEPS, CONTRAST, LIMIT, RANK = 1e-8, 20, 1e4, 24, 4
SOURCES = ['fields.py', 'numerics.py', 'sequence_baselines.py', 'balanced_augmentation.py',
           'cached_pcg.py', 'event_space_update.py', 'event_recycling_experiment.py']
PROTOCOL, SPLITS = 'docs/stage6_protocol_zh.md', 'data/stage6_splits.json'


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
    fields, _ = coefficient_sequence(32, STEPS, seed, CONTRAST, 'mixed')
    return hashlib.sha256((fields > 1).tobytes()).hexdigest()


def splits():
    p = ROOT / SPLITS
    if p.exists(): return read(p)
    old = read(ROOT / 'data/stage5_splits.json')
    excluded = sorted(set(old['excluded_seeds'] + sum(old['groups'].values(), []) + [75001,75002]))
    seen = {geometry_hash(s) for s in excluded}
    result = dict(excluded_seeds=excluded, groups={}, geometry_sha256={}, skipped_duplicates=[])
    for name, start, count in [('development',44001,3),('confirmation',44011,5)]:
        seeds=[]
        while len(seeds)<count:
            digest=geometry_hash(start)
            if digest in seen: result['skipped_duplicates'].append(dict(seed=start,group=name))
            else:
                seen.add(digest);seeds.append(start);result['geometry_sha256'][str(start)]=digest
            start+=1
        result['groups'][name]=seeds
    save(p,result)
    return result


def provenance():
    return dict(source_sha256={f:sha(ROOT/f) for f in SOURCES},
                protocol_sha256=sha(ROOT/PROTOCOL),splits_sha256=sha(ROOT/SPLITS))


def check_provenance(record):
    for key,value in provenance().items():
        if record[key]!=value: raise RuntimeError(f'Frozen {key} changed')


def manifest(configuration):
    return dict(configuration=configuration,created_utc=stamp(),python=sys.version,
        platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,pyamg=pyamg.__version__,
        device='cpu',blas_threads=1,rtol=RTOL,maxiter=500,steps=STEPS,contrast=CONTRAST,rank=RANK,
        timing='Complete assembly through history/space maintenance; update is independent, extraction is inside maintenance. References and offline probes excluded.',**provenance())


def specs():
    return [dict(name='baseline_scipy',policy='none',event_custom=False),
        dict(name='baseline_event_custom',policy='none',event_custom=True),
        *[dict(name=name,policy=policy,event_custom=True) for name,policy in [
            ('lagged_balanced4','lagged'),('event_reuse_refresh4','reuse_refresh'),
            ('event_update_refresh4','update_refresh'),('event_reuse_frozen4','reuse_frozen'),
            ('event_update_transport4','update_transport')]]]


def is_event(k,old):
    return bool(old is None or float(np.max(np.abs(np.log(k/old))))>=np.log(10.))


def initial(A,b,history):
    return galerkin_initial_guess(A,b,history,max_vectors=4)[0] if history else np.zeros_like(b)


def run_sequence(fields,events,setting,repeat):
    history,rows,solutions=[],[],[]
    recycle=np.empty(((fields.shape[-1]-1)**2,0))
    policy=AMGSequencePreconditioner('refresh_galerkin',seed=events[0]['seed']*100)
    old=None; mode=setting['policy']
    for step,k in enumerate(fields):
        start=time.perf_counter()
        row=dict(method=setting['name'],n=fields.shape[-1],seed=events[0]['seed'],step=step,
            repeat=repeat,event=events[step]['event'],initial_step=step==0,
            recovery_seconds=0.,solver_fallback=False,cache_rejected=False,update_rejected=False,
            rank=0,old_space_rank=int(recycle.shape[1]),orthogonality_error=0.,
            capture_stats={},extraction={},extraction_seconds=0.,update_stats={},update_seconds=0.,
            basis_stats={},projection_info={})
        t=time.perf_counter();A,b,_=assemble_system(k);row['assembly_seconds']=time.perf_counter()-t
        t=time.perf_counter();base=initial(A,b,history);row['guess_seconds']=time.perf_counter()-t
        M,setup=policy.for_matrix(A);row['setup_seconds']=setup['setup_seconds']
        event=is_event(k,old);row['refresh_event']=event
        do_update=bool(event and step>0 and mode in ('update_refresh','update_transport') and recycle.shape[1])
        U,AU=None,None
        if do_update:
            t=time.perf_counter();U,AU,info=update_slow_space(A,M,recycle,q=RANK)
            row['update_seconds']=time.perf_counter()-t;row['update_stats']=info
            row['update_rejected']=info['status'] not in ('accepted','empty_basis')
            if mode=='update_transport' and info['status']=='accepted':
                # Keep the owned returned array; its storage is paid by update.
                recycle=U
        t=time.perf_counter();x0,operator=base,M
        apply_space=bool(recycle.shape[1] and (not event or mode!='lagged'))
        if apply_space:
            if U is None:
                U,AU,info=a_orthonormalize(A,recycle);row['basis_stats']=info
            x0,projection=galerkin_correction(A,b,base,U,AU);row['projection_info']=projection
            if not event: operator=balanced_preconditioner(M,U,AU)
            row.update(rank=int(U.shape[1]),orthogonality_error=float(la.norm(U.T@AU-np.eye(U.shape[1]),ord=2)) if U.shape[1] else 0.)
        row['correction_seconds']=time.perf_counter()-t
        row['base_relative_residual']=float(la.norm(b-A@base)/la.norm(b))
        row['corrected_relative_residual']=float(la.norm(b-A@x0)/la.norm(b))
        do_extract=bool(mode!='none' and event and (step==0 or mode in ('lagged','reuse_refresh','update_refresh')))
        row['extraction_requested']=do_extract
        cache=None
        if event and setting['event_custom']:
            x,metrics,cache=solve_pcg_cached(A,b,M,x0,RTOL,500,max_vectors=LIMIT if do_extract else 0)
            row['capture_stats']=cache['stats']
        else: x,metrics=solve_pcg(A,b,operator,x0,RTOL,500)
        row.update({('solve_seconds' if key=='seconds' else key):value for key,value in metrics.items()})
        if not metrics['converged']:
            t=time.perf_counter();x=sla.spsolve(A,b);row['recovery_seconds']=time.perf_counter()-t;row['solver_fallback']=True
        row['final_relative_residual']=float(la.norm(b-A@x)/la.norm(b))
        row['final_converged']=row['final_relative_residual']<=RTOL
        t=time.perf_counter()
        if do_extract:
            et=time.perf_counter();recycle,_,info=cached_slow_ritz(cache['D'],cache['AD'],cache['MAD'],q=RANK)
            row['extraction_seconds']=time.perf_counter()-et;row['extraction']=info
            row['cache_rejected']=info['status'] not in ('accepted','empty_basis')
        cache=None;U=None;AU=None
        history.append(x.copy());history=history[-4:];old=k
        row['stored_space_rank']=int(recycle.shape[1])
        row['maintenance_seconds']=time.perf_counter()-t;row['total_seconds']=time.perf_counter()-start
        rows.append(row);solutions.append(x.copy())
    print(json.dumps(dict(stage='sequence',method=setting['name'],n=fields.shape[-1],seed=events[0]['seed'],repeat=repeat,
        seconds=sum(r['total_seconds'] for r in rows),iterations=sum(r['iterations'] for r in rows),
        pcg_failures=sum(not r['converged'] for r in rows),cache_rejections=sum(r['cache_rejected'] for r in rows),
        update_rejections=sum(r['update_rejected'] for r in rows))),flush=True)
    return rows,solutions


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[(r['method'],r['seed'],r['repeat'])].append(r)
    result={}
    for method in sorted({r['method'] for r in rows}):
        local=[r for r in rows if r['method']==method]
        by_seed={str(s):median(sum(r['total_seconds'] for r in v) for (m,seed,_),v in grouped.items() if m==method and seed==s)
                 for s in sorted({r['seed'] for r in local})}
        result[method]=dict(by_seed=by_seed,median_seconds=median(by_seed.values()),all_accurate=all(r['final_converged'] for r in local),
            pcg_failures=sum(not r['converged'] for r in local),cache_rejections=sum(r['cache_rejected'] for r in local),
            update_rejections=sum(r['update_rejected'] for r in local),
            update_budget_valid=all(r['update_stats'].get('a_applications',0)<=8 and r['update_stats'].get('m_applications',0)<=8 for r in local))
    return result


def benchmark(name,n,seeds,candidates,repeats,role,selection_hash=None):
    out=fresh(name);save(out/'manifest.json',manifest(dict(n=n,seeds=seeds,specs=candidates,repeats=repeats,role=role,selection_sha256=selection_hash)))
    rows,references=[],[]
    for seed in seeds:
        fields,events=coefficient_sequence(n,STEPS,seed,CONTRAST,'mixed')
        t=time.perf_counter();truth,residuals=[],[]
        for k in fields:
            A,b,_=assemble_system(k);x=sla.spsolve(A,b);truth.append(x);residuals.append(float(la.norm(b-A@x)/la.norm(b)))
        references.append(dict(seed=seed,seconds=time.perf_counter()-t,field_sha256=hashlib.sha256(fields.tobytes()).hexdigest(),relative_residuals=residuals))
        save(out/'references.json',references)
        for repeat in range(repeats):
            for index in np.random.default_rng(seed+repeat+n).permutation(len(candidates)):
                measured,xs=run_sequence(fields,events,candidates[int(index)],repeat)
                for step,row in enumerate(measured):
                    A,_,_=assemble_system(fields[step]);row['energy_relative_error']=energy_relative_error(A,xs[step],truth[step])
                rows.extend(measured);save(out/'measurements.json',rows)
    check_provenance(read(out/'manifest.json'));save(out/'summary.json',aggregate(rows))
    save(out/'complete.json',dict(complete=True,rows=len(rows),finished_utc=stamp()))
    return out


def probe():
    out=fresh('stage6_probe');seeds=splits()['groups']['development'];rows=[]
    save(out/'manifest.json',manifest(dict(n=128,seeds=seeds,role='offline common baseline history; noninitial events only',
        directions=['previous_event','updated_previous','initial_frozen','current_retrospective'],interfaces=['warm','balanced'],
        timing_scope='Offline component measurements only, not complete online costs: historical acquisition excluded; current extraction is separate; current retrospective direction is unavailable before the baseline solve.')))
    for seed in seeds:
        fields,events=coefficient_sequence(128,STEPS,seed,CONTRAST,'mixed')
        policy=AMGSequencePreconditioner('refresh_galerkin',seed=seed*100)
        history=[];old=None;recycle=np.empty(((128-1)**2,0));first=recycle;source_step=None
        for step,k in enumerate(fields):
            A,b,_=assemble_system(k);base=initial(A,b,history);M,_=policy.for_matrix(A);event=is_event(k,old)
            if event:
                x,metrics,cache=solve_pcg_cached(A,b,M,base,RTOL,500,max_vectors=LIMIT)
                et=time.perf_counter();current,_,extraction=extract_slow_ritz(A,M,cache['D'],q=RANK,max_basis=LIMIT)
                current_extraction_seconds=time.perf_counter()-et
                if step>0:
                    truth=sla.spsolve(A,b);e=truth-base;Ae=A@e;energy=float(e@Ae)
                    C,AC,_=a_orthonormalize(A,current);slow_energy=float(la.norm(C.T@Ae)**2)
                    row=dict(seed=seed,step=step,event=events[step]['event'],previous_source_step=source_step,
                        initial_source_step=0,baseline=metrics,reference_relative_residual=float(la.norm(b-A@truth)/la.norm(b)),
                        base_error_energy=energy,current_slow_error_energy=slow_energy,current_extraction=extraction,
                        current_extraction_seconds=current_extraction_seconds,directions={})
                    for name in ['previous_event','updated_previous','initial_frozen','current_retrospective']:
                        t=time.perf_counter()
                        if name=='updated_previous': U,AU,info=update_slow_space(A,M,recycle,q=RANK)
                        else: U,AU,info=a_orthonormalize(A,{'previous_event':recycle,'initial_frozen':first,'current_retrospective':current}[name])
                        setup_seconds=time.perf_counter()-t
                        pt=time.perf_counter();corrected,projection=galerkin_correction(A,b,base,U,AU)
                        projection_seconds=time.perf_counter()-pt
                        ce=truth-corrected;Ace=A@ce
                        value=dict(rank=int(U.shape[1]),setup_seconds=setup_seconds,setup_info=info,projection=projection,projection_seconds=projection_seconds,
                            error_energy_ratio=float(ce@Ace/max(energy,np.finfo(float).tiny)),
                            slow_error_energy_ratio=float(la.norm(C.T@Ace)**2/max(slow_energy,np.finfo(float).tiny)),
                            reference_space_overlap=float(la.norm(U.T@AC,ord='fro')**2/max(C.shape[1],1)),solves={})
                        for interface in ['warm','balanced']:
                            bt=time.perf_counter();operator=M if interface=='warm' else balanced_preconditioner(M,U,AU)
                            preconditioner_setup_seconds=time.perf_counter()-bt
                            _,solved=solve_pcg(A,b,operator,corrected,RTOL,500)
                            solved['preconditioner_setup_seconds']=preconditioner_setup_seconds;value['solves'][interface]=solved
                        row['directions'][name]=value
                    rows.append(row);save(out/'measurements.json',rows)
                    print(json.dumps(dict(stage='probe',seed=seed,step=step,baseline_iterations=metrics['iterations'],
                        warm_iterations={name:v['solves']['warm']['iterations'] for name,v in row['directions'].items()})),flush=True)
                recycle=current;source_step=step
                if step==0: first=current.copy()
            else: x,metrics=solve_pcg(A,b,M,base,RTOL,500)
            if not metrics['converged']: x=sla.spsolve(A,b)
            history.append(x.copy());history=history[-4:];old=k
    check_provenance(read(out/'manifest.json'))
    save(out/'complete.json',dict(complete=True,cases=len(rows),finished_utc=stamp()))


def check_probe():
    out=ROOT/'results/stage6_probe';check_provenance(read(out/'manifest.json'))
    rows=read(out/'measurements.json');complete=read(out/'complete.json');expected=set()
    for seed in splits()['groups']['development']:
        fields,_=coefficient_sequence(32,STEPS,seed,CONTRAST,'mixed')
        expected.update((seed,t) for t in range(1,STEPS) if is_event(fields[t],fields[t-1]))
    if not complete['complete'] or complete['cases']!=len(expected) or len(rows)!=len(expected) or {(r['seed'],r['step']) for r in rows}!=expected:
        raise RuntimeError('Incomplete common-history probe')
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(out.glob('*.json'))}


def screen():
    check_probe();out=benchmark('stage6_screen',128,splits()['groups']['development'],specs(),2,'development event-space and refresh screen')
    summary=read(out/'summary.json')
    eligible=[s for s in specs()[3:] if summary[s['name']]['all_accurate'] and summary[s['name']]['pcg_failures']==0
        and summary[s['name']]['cache_rejections']==0 and summary[s['name']]['update_rejections']==0 and summary[s['name']]['update_budget_valid']]
    if not eligible: raise RuntimeError('No eligible causal event candidate; preserve negative screen')
    chosen=min(eligible,key=lambda s:summary[s['name']]['median_seconds'])
    confirmation=specs()[:5]
    if chosen['name'] not in {s['name'] for s in confirmation}: confirmation.append(chosen)
    selected=dict(created_utc=stamp(),candidate=chosen,confirmation_specs=confirmation,probe_sha256=check_probe(),
        screen_measurements_sha256=sha(out/'measurements.json'),screen_manifest_sha256=sha(out/'manifest.json'),
        rule='Fastest accurate zero-failure zero-space-rejection budget-valid method among four causal event strategies on development only; selection does not imply gain.',**provenance())
    save(out/'frozen_selection.json',selected);print(json.dumps(dict(stage='screen_selected',candidate=chosen)),flush=True)


def confirm():
    p=ROOT/'results/stage6_screen/frozen_selection.json';selected=read(p);check_provenance(selected)
    if check_probe()!=selected['probe_sha256']: raise RuntimeError('Frozen probe changed')
    for key,file in [('screen_measurements_sha256','measurements.json'),('screen_manifest_sha256','manifest.json')]:
        if sha(p.parent/file)!=selected[key]: raise RuntimeError('Frozen screen changed')
    rows=read(p.parent/'measurements.json')
    expected={(s['name'],seed,r,t) for s in specs() for seed in splits()['groups']['development'] for r in range(2) for t in range(STEPS)}
    if len(rows)!=len(expected) or {(r['method'],r['seed'],r['repeat'],r['step']) for r in rows}!=expected: raise RuntimeError('Incomplete screen')
    seeds=splits()['groups']['confirmation']
    for n,cohort in [(128,seeds),(256,seeds[:2])]:
        benchmark(f'stage6_confirm{n}',n,cohort,selected['confirmation_specs'],3,'frozen event-space mechanism confirmation',sha(p))
    save(ROOT/'results/stage6_complete.json',dict(complete=True,finished_utc=stamp(),selection_sha256=sha(p)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['splits','probe','screen','confirm'])
    args=parser.parse_args();splits()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='probe': probe()
        elif args.stage=='screen': screen()
        elif args.stage=='confirm': confirm()


if __name__=='__main__': main()
