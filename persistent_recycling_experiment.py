#!/usr/bin/env python3
"""Stage 7: use a first-system recycled space throughout later solves."""
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
from balanced_augmentation import a_orthonormalize, balanced_preconditioner, galerkin_correction
from cached_pcg import solve_pcg_cached, cached_slow_ritz

RTOL, STEPS, CONTRAST, LIMIT, RANK = 1e-8, 20, 1e4, 24, 4
SOURCES = ['fields.py','numerics.py','sequence_baselines.py','balanced_augmentation.py',
           'cached_pcg.py','persistent_recycling_experiment.py']
PROTOCOL, SPLITS = 'docs/stage7_protocol_zh.md','data/stage7_splits.json'
HYPOTHESIS = 'frozen_all_balanced4'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def fresh(name):
    p=ROOT/'results'/name
    if p.exists() and any(p.iterdir()): raise FileExistsError(p)
    p.mkdir(parents=True,exist_ok=True)
    return p


def geometry_hash(seed):
    fields,_=coefficient_sequence(32,STEPS,seed,CONTRAST,'mixed')
    return hashlib.sha256((fields>1).tobytes()).hexdigest()


def splits():
    p=ROOT/SPLITS
    if p.exists(): return read(p)
    old=read(ROOT/'data/stage6_splits.json')
    excluded=sorted(set(old['excluded_seeds']+sum(old['groups'].values(),[])+[75003]))
    seen={geometry_hash(s) for s in excluded}
    result=dict(excluded_seeds=excluded,groups={},geometry_sha256={},skipped_duplicates=[])
    for name,start,count in [('development',45001,3),('confirmation',45011,5)]:
        seeds=[]
        while len(seeds)<count:
            digest=geometry_hash(start)
            if digest in seen:result['skipped_duplicates'].append(dict(seed=start,group=name))
            else:
                seen.add(digest);seeds.append(start);result['geometry_sha256'][str(start)]=digest
            start+=1
        result['groups'][name]=seeds
    save(p,result);return result


def provenance():
    return dict(source_sha256={f:sha(ROOT/f) for f in SOURCES},protocol_sha256=sha(ROOT/PROTOCOL),splits_sha256=sha(ROOT/SPLITS))


def check_provenance(record):
    for key,value in provenance().items():
        if record[key]!=value:raise RuntimeError(f'Frozen {key} changed')


def manifest(configuration):
    return dict(configuration=configuration,created_utc=stamp(),python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,pyamg=pyamg.__version__,device='cpu',blas_threads=1,
        rtol=RTOL,maxiter=500,steps=STEPS,contrast=CONTRAST,rank=RANK,
        timing='Complete assembly through space/history maintenance; extraction inside maintenance. References, energy errors and source-space hashes outside timing.',**provenance())


def specs():
    return [dict(name='baseline_scipy',policy='none',event_custom=False),
        dict(name='baseline_event_custom',policy='none',event_custom=True),
        *[dict(name=name,policy=policy,event_custom=True) for name,policy in [
            ('frozen_event_warm4','event_warm'),('frozen_all_balanced4','all_balanced'),('frozen_all_warm4','all_warm')]]]


def is_event(k,old):
    return bool(old is None or float(np.max(np.abs(np.log(k/old))))>=np.log(10.))


def initial(A,b,history):
    return galerkin_initial_guess(A,b,history,max_vectors=4)[0] if history else np.zeros_like(b)


def run_sequence(fields,events,setting,repeat):
    history,rows,solutions=[],[],[]
    recycle=np.empty(((fields.shape[-1]-1)**2,0))
    policy=AMGSequencePreconditioner('refresh_galerkin',seed=events[0]['seed']*100)
    old=None;mode=setting['policy']
    for step,k in enumerate(fields):
        start=time.perf_counter()
        row=dict(method=setting['name'],n=fields.shape[-1],seed=events[0]['seed'],step=step,repeat=repeat,
            event=events[step]['event'],initial_step=step==0,recovery_seconds=0.,solver_fallback=False,
            cache_rejected=False,rank=0,old_space_rank=int(recycle.shape[1]),orthogonality_error=0.,
            capture_stats={},extraction={},extraction_seconds=0.,basis_stats={},projection_info={},solver_operator='base')
        t=time.perf_counter();A,b,_=assemble_system(k);row['assembly_seconds']=time.perf_counter()-t
        t=time.perf_counter();base=initial(A,b,history);row['guess_seconds']=time.perf_counter()-t
        M,setup=policy.for_matrix(A);row['setup_seconds']=setup['setup_seconds']
        event=is_event(k,old);row['refresh_event']=event
        t=time.perf_counter();x0,operator=base,M;U,AU=None,None
        if recycle.shape[1]:
            U,AU,info=a_orthonormalize(A,recycle);row['basis_stats']=info
            x0,projection=galerkin_correction(A,b,base,U,AU);row['projection_info']=projection
            use_balanced=mode=='all_balanced' or (mode=='event_warm' and not event)
            if use_balanced and U.shape[1]:
                operator=balanced_preconditioner(M,U,AU);row['solver_operator']='balanced'
            row.update(rank=int(U.shape[1]),orthogonality_error=float(la.norm(U.T@AU-np.eye(U.shape[1]),ord=2)) if U.shape[1] else 0.)
        row['correction_seconds']=time.perf_counter()-t
        row['base_relative_residual']=float(la.norm(b-A@base)/la.norm(b))
        row['corrected_relative_residual']=float(la.norm(b-A@x0)/la.norm(b))
        do_extract=bool(mode!='none' and step==0)
        row['extraction_requested']=do_extract;row['capture_requested']=do_extract
        cache=None
        if event and setting['event_custom']:
            # Later event calls must receive the actual B/M, with no collection.
            x,metrics,cache=solve_pcg_cached(A,b,operator,x0,RTOL,500,max_vectors=LIMIT if do_extract else 0)
            row['capture_stats']=cache['stats']
        else:x,metrics=solve_pcg(A,b,operator,x0,RTOL,500)
        row.update({('solve_seconds' if key=='seconds' else key):value for key,value in metrics.items()})
        if not metrics['converged']:
            t=time.perf_counter();x=sla.spsolve(A,b);row['recovery_seconds']=time.perf_counter()-t;row['solver_fallback']=True
        row['final_relative_residual']=float(la.norm(b-A@x)/la.norm(b));row['final_converged']=row['final_relative_residual']<=RTOL
        t=time.perf_counter()
        if do_extract:
            et=time.perf_counter();recycle,_,info=cached_slow_ritz(cache['D'],cache['AD'],cache['MAD'],q=RANK)
            row['extraction_seconds']=time.perf_counter()-et;row['extraction']=info
            row['cache_rejected']=info['status'] not in ('accepted','empty_basis')
            recycle.flags.writeable=False
        cache=None;U=None;AU=None
        history.append(x.copy());history=history[-4:];old=k
        row['stored_space_rank']=int(recycle.shape[1]);row['maintenance_seconds']=time.perf_counter()-t
        row['total_seconds']=time.perf_counter()-start
        rows.append(row);solutions.append(x.copy())
    # Immutable first-space audit only: not consulted by any online decision.
    digest=hashlib.sha256(recycle.tobytes()).hexdigest() if mode!='none' else None
    for row in rows:row['initial_space_sha256']=digest
    print(json.dumps(dict(stage='sequence',method=setting['name'],n=fields.shape[-1],seed=events[0]['seed'],repeat=repeat,
        seconds=sum(r['total_seconds'] for r in rows),iterations=sum(r['iterations'] for r in rows),
        pcg_failures=sum(not r['converged'] for r in rows),cache_rejections=sum(r['cache_rejected'] for r in rows))),flush=True)
    return rows,solutions


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['method'],r['seed'],r['repeat'])].append(r)
    result={}
    for method in sorted({r['method'] for r in rows}):
        local=[r for r in rows if r['method']==method]
        by_seed={str(s):median(sum(r['total_seconds'] for r in v) for (m,seed,_),v in grouped.items() if m==method and seed==s)
                 for s in sorted({r['seed'] for r in local})}
        result[method]=dict(by_seed=by_seed,median_seconds=median(by_seed.values()),all_accurate=all(r['final_converged'] for r in local),
            pcg_failures=sum(not r['converged'] for r in local),cache_rejections=sum(r['cache_rejected'] for r in local))
    return result


def benchmark(name,n,seeds,repeats,role,freeze_hash=None):
    candidates=specs();out=fresh(name)
    save(out/'manifest.json',manifest(dict(n=n,seeds=seeds,specs=candidates,repeats=repeats,role=role,freeze_sha256=freeze_hash)))
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


def development():
    out=benchmark('stage7_development',128,splits()['groups']['development'],2,'prespecified development; no performance selection')
    summary=read(out/'summary.json')
    valid=all(v['all_accurate'] and v['pcg_failures']==0 and v['cache_rejections']==0 for v in summary.values())
    if not valid:raise RuntimeError('Prespecified numerical development gate failed; retain all data and stop')
    frozen=dict(created_utc=stamp(),fixed_hypothesis=HYPOTHESIS,confirmation_specs=specs(),
        development_measurements_sha256=sha(out/'measurements.json'),development_manifest_sha256=sha(out/'manifest.json'),
        rule='All five methods numerically valid; no timing-based eligibility or method selection. Primary hypothesis fixed before development.',**provenance())
    save(out/'freeze.json',frozen);print(json.dumps(dict(stage='development_frozen',fixed_hypothesis=HYPOTHESIS)),flush=True)


def confirm():
    p=ROOT/'results/stage7_development/freeze.json';frozen=read(p);check_provenance(frozen)
    if frozen['fixed_hypothesis']!=HYPOTHESIS or frozen['confirmation_specs']!=specs():raise RuntimeError('Fixed hypothesis changed')
    for key,file in [('development_measurements_sha256','measurements.json'),('development_manifest_sha256','manifest.json')]:
        if sha(p.parent/file)!=frozen[key]:raise RuntimeError('Frozen development changed')
    rows=read(p.parent/'measurements.json')
    expected={(s['name'],seed,r,t) for s in specs() for seed in splits()['groups']['development'] for r in range(2) for t in range(STEPS)}
    if len(rows)!=len(expected) or {(r['method'],r['seed'],r['repeat'],r['step']) for r in rows}!=expected:raise RuntimeError('Incomplete development')
    if not all(r['final_converged'] and r['converged'] and not r['cache_rejected'] for r in rows):raise RuntimeError('Numerical development gate failed')
    seeds=splits()['groups']['confirmation']
    for n,cohort in [(128,seeds),(256,seeds[:2])]:
        benchmark(f'stage7_confirm{n}',n,cohort,3,'frozen confirmation; fixed five methods',sha(p))
    save(ROOT/'results/stage7_complete.json',dict(complete=True,finished_utc=stamp(),freeze_sha256=sha(p)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['splits','development','confirm'])
    args=parser.parse_args();splits()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='development':development()
        elif args.stage=='confirm':confirm()


if __name__=='__main__':main()
