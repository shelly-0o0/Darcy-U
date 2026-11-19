#!/usr/bin/env python3
"""Frozen, CPU-only nonlinear forcing opportunity screen; no learning."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
from statistics import median
import time
import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info
from nonlinear_budget import make_problem, solve

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/stage9_screen'
METHODS = ['ew1','ew2','fixed_0.01','fixed_0.1','fixed_0.5','fixed_0.9','direct']
MULTIPLIERS = [.25,.5,1.,2.,4.]
SOURCES = ['nonlinear_budget.py','nonlinear_screen_experiment.py']
PROTOCOL = 'docs/stage9_protocol_zh.md'
CASE_FILE = 'data/stage9_cases.json'


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a): return hashlib.sha256(np.asarray(a,dtype=np.float64).tobytes()).hexdigest()
def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,np.ndarray): return safe(value.tolist())
    if isinstance(value,np.generic): return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(path): return json.loads(Path(path).read_text())


def unique_actions(native):
    actions=[];seen={}
    for multiplier in MULTIPLIERS:
        eta=float(np.clip(multiplier*native,1e-10,.9))
        if eta not in seen:
            seen[eta]=len(actions);actions.append(dict(multiplier=multiplier,eta=eta))
        elif multiplier==1.:
            actions[seen[eta]]['multiplier']=1.
    return actions


def check_freeze():
    record=read(OUT/'freeze.json')
    for name,digest in record['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/PROTOCOL)==record['protocol_sha256']
    assert sha(ROOT/CASE_FILE)==record['cases_sha256']


def cases():
    result=[]
    for index,(eps,gamma,amplitude) in enumerate(itertools.product([.003,.03,.3],[10.,1000.],[5.,50.])):
        cx,cy=np.random.default_rng(47000+index).uniform(.3,.7,2)
        result.append(dict(case=index,seed=47000+index,params=dict(eps=eps,gamma=gamma,
            amplitude=amplitude,cx=float(cx),cy=float(cy),beta=[1.,.5])))
    return result


def freeze():
    if OUT.exists(): raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    case_data=dict(created_utc=stamp(),grids=[32,64],cases=cases(),
                   role='development only; 12 shared parameter instances across grids')
    if (ROOT/CASE_FILE).exists():
        assert read(ROOT/CASE_FILE)['cases']==case_data['cases']
    else:save(ROOT/CASE_FILE,case_data)
    record=dict(created_utc=stamp(),source_sha256={p:sha(ROOT/p) for p in SOURCES},
        protocol_sha256=sha(ROOT/PROTOCOL),cases_sha256=sha(ROOT/CASE_FILE),methods=METHODS,
        repeats=3,grids=[32,64],branch_grid=64,snapshot_iterations=[0,1,2,3],
        multipliers=MULTIPLIERS,threshold=.85,faster_required=9,training_authorized=False,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),blas=threadpool_info())
    save(OUT/'freeze.json',record)
    return case_data['cases']


def pack(result):
    return {k:v for k,v in result.items() if k not in ['u','snapshots']}


def accurate(result,truth):
    error=float(np.linalg.norm(result['u']-truth)/max(np.linalg.norm(truth),1e-300))
    valid=bool(result['converged'] and np.isfinite(error) and error<=1e-6
        and np.isfinite(result['absolute_residual']) and result['absolute_residual']<=result['tolerance'])
    return error,valid


def numerical_verification(instances):
    records=[];mms=[];truth={}
    for label,beta,order_range in [('upwind',[1.,.5],[.75,1.35]),('diffusion',[0.,0.],[1.7,2.3])]:
        group=[]
        for n in [15,31,63,127]:
            problem=make_problem(n,dict(eps=.05,gamma=2.,amplitude=.4,beta=beta),manufactured=True)
            result=solve(problem,'direct',rtol=1e-11,atol=1e-12)
            error=float(np.linalg.norm(result['u']-problem.exact)/np.linalg.norm(problem.exact))
            group.append(dict(n=n,h=problem.h,error=error,solver=pack(result)))
        orders=[math.log(a['error']/b['error'])/math.log(a['h']/b['h']) for a,b in zip(group,group[1:])]
        valid=bool(all(r['solver']['converged'] for r in group) and order_range[0]<=orders[-1]<=order_range[1])
        mms.append(dict(label=label,records=group,orders=orders,required_last_order=order_range,valid=valid))
    for n in [32,64]:
        for case in instances:
            problem=make_problem(n,case['params'])
            result=solve(problem,'direct',rtol=1e-11,atol=1e-12)
            truth[(n,case['case'])]=result['u'].copy()
            records.append(dict(n=n,case=case['case'],rhs_sha256=array_sha(problem.rhs),
                solution_sha256=array_sha(result['u']),assembly_seconds=problem.assembly_seconds,solver=pack(result)))
    valid=bool(all(v['valid'] for v in mms) and all(v['solver']['converged'] for v in records))
    record=dict(created_utc=stamp(),manufactured=mms,direct_references=records,valid=valid)
    save(OUT/'verification.json',record)
    return truth,valid


def aggregate_baselines(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['n'],row['case'],row['method'])].append(row)
    by_method={}
    for method in METHODS:
        points=[]
        for n,case in sorted({(r['n'],r['case']) for r in rows}):
            group=grouped[(n,case,method)]
            values=[r['total_seconds'] for r in group]
            seconds=median(values)
            base=median(r['total_seconds'] for r in grouped[(n,case,'ew2')])
            points.append(dict(n=n,case=case,seconds=seconds,
                time_mad=median(abs(v-seconds) for v in values),time_min=min(values),time_max=max(values),
                relative_to_ew2=seconds/base,valid=all(r['accurate'] for r in group),
                newton=median(r['newton_iterations'] for r in group),
                linear=median(r['linear_iterations'] for r in group),
                components={k:median(r['components'][k] for r in group) for k in group[0]['components']},
                assembly_seconds=median(r['assembly_seconds'] for r in group)))
        by_method[method]=dict(points=points,all_valid=all(p['valid'] for p in points),
            score=median(p['relative_to_ew2'] for p in points))
    eligible=[m for m in METHODS if by_method[m]['all_valid']]
    ew=[m for m in ['ew1','ew2'] if m in eligible]
    return dict(methods=by_method,
        primary=min(eligible,key=lambda m:by_method[m]['score']) if eligible else None,
        continuation=min(ew,key=lambda m:by_method[m]['score']) if ew else None)


def run_baselines(instances,truth):
    rows=[];execution=[]
    warm=make_problem(16,dict(eps=.07,gamma=3.,amplitude=2.,beta=[1.,.5]))
    warmup=[dict(method=method,**pack(solve(warm,method))) for method in METHODS]
    save(OUT/'warmup.json',dict(n=16,params=warm.params,runs=warmup,
        scope='Excluded deterministic warmup; not a formal parameter instance'))
    for n in [32,64]:
        for case in instances:
            for repeat in range(3):
                order=list(np.random.default_rng(590000+n*100+case['case']*10+repeat).permutation(METHODS))
                execution.append(dict(n=n,case=case['case'],repeat=repeat,methods=order))
                for method in order:
                    t=time.perf_counter()
                    problem=make_problem(n,case['params'])
                    result=solve(problem,method)
                    total=time.perf_counter()-t
                    error,valid=accurate(result,truth[(n,case['case'])])
                    row=dict(**pack(result),n=n,case=case['case'],repeat=repeat,method=method,
                        assembly_seconds=problem.assembly_seconds,solve_seconds=result['total_seconds'],
                        relative_solution_error=error,accurate=valid,rhs_sha256=array_sha(problem.rhs),
                        solution_sha256=array_sha(result['u']))
                    row['total_seconds']=total
                    rows.append(row)
            save(OUT/'baselines.json',rows)
            print(json.dumps(dict(phase='baseline',n=n,case=case['case'],rows=len(rows))),flush=True)
    save(OUT/'baseline_order.json',execution)
    decision=aggregate_baselines(rows)
    decision['baseline_sha256']=sha(OUT/'baselines.json')
    decision['created_utc']=stamp()
    save(OUT/'baseline_selection.json',decision)
    return decision


def prepare_snapshots(instances,method,truth):
    states=[];metadata=[]
    start=time.perf_counter()
    for case in instances:
        problem=make_problem(64,case['params'])
        source=solve(problem,method,capture=True)
        error,valid=accurate(source,truth[(64,case['case'])])
        snapshots=[]
        trace_by_iteration={r['iteration']:r for r in source['trace']}
        for state in source['snapshots']:
            iteration=state['iteration']
            if iteration not in [0,1,2,3] or iteration not in trace_by_iteration:continue
            trace=trace_by_iteration[iteration]
            native=trace['native_eta']
            actions=unique_actions(native)
            path=ROOT/'data'/f"stage9_case{case['case']:02d}_step{iteration}.npz"
            path.parent.mkdir(parents=True,exist_ok=True)
            storage=time.perf_counter()
            # Scalar history is JSON metadata; arrays require no pickle.
            np.savez_compressed(path,u=state['u'])
            storage=time.perf_counter()-storage
            info=dict(iteration=iteration,native_eta=native,actions=actions,
                archive=str(path.relative_to(ROOT)),archive_sha256=sha(path),
                state_u_sha256=array_sha(state['u']),storage_seconds=storage,
                scalar_state={k:v for k,v in state.items() if k!='u'})
            snapshots.append(info)
            states.append(dict(case=case['case'],params=case['params'],state=state,info=info,problem=problem))
        metadata.append(dict(case=case['case'],source=pack(source),source_accurate=valid,
            relative_solution_error=error,source_assembly_seconds=problem.assembly_seconds,
            snapshots=snapshots,missing_iterations=sorted(set([0,1,2,3])-{s['iteration'] for s in snapshots})))
    record=dict(created_utc=stamp(),continuation=method,sequences=metadata,
        preparation_seconds=time.perf_counter()-start,all_snapshots_prepared_before_branch_timing=True,
        snapshot_count=len(states))
    save(OUT/'snapshot_preparation.json',record)
    return states,record


def run_branches(states,method,truth):
    rows=[];execution=[]
    for item in states:
        case=item['case'];iteration=item['state']['iteration']
        for repeat in range(3):
            actions=item['info']['actions']
            order=np.random.default_rng(690000+case*100+iteration*10+repeat).permutation(len(actions))
            execution.append(dict(case=case,iteration=iteration,repeat=repeat,action_order=order.tolist()))
            for index in order:
                action=actions[int(index)]
                start=time.perf_counter()
                problem=make_problem(64,item['params']) if iteration==0 else item['problem']
                result=solve(problem,method,initial_state=item['state'],first_multiplier=action['multiplier'])
                total=time.perf_counter()-start
                error,valid=accurate(result,truth[(64,case)])
                row=dict(**pack(result),n=64,case=case,iteration=iteration,repeat=repeat,
                    method=method,multiplier=action['multiplier'],expected_eta=action['eta'],
                    timing_scope='complete_with_assembly' if iteration==0 else 'remaining_solve_with_resident_operator',
                    assembly_seconds=problem.assembly_seconds if iteration==0 else 0.,
                    solve_seconds=result['total_seconds'],relative_solution_error=error,accurate=valid,
                    source_state_sha256=item['info']['state_u_sha256'],solution_sha256=array_sha(result['u']))
                row['total_seconds']=total
                assert array_sha(item['state']['u'])==item['info']['state_u_sha256'],'Snapshot mutated'
                rows.append(row)
        save(OUT/'branches.json',rows)
        print(json.dumps(dict(phase='branch',case=case,iteration=iteration,rows=len(rows))),flush=True)
    save(OUT/'branch_order.json',execution)
    return rows


def assess(rows,selection,preparation):
    groups=defaultdict(list)
    for row in rows:groups[(row['case'],row['iteration'],row['multiplier'])].append(row)
    state_points=[]
    for case,iteration in sorted({(r['case'],r['iteration']) for r in rows}):
        actions=[]
        for (c,k,m),group in sorted(groups.items()):
            if (c,k)!=(case,iteration):continue
            values=[r['total_seconds'] for r in group];seconds=median(values)
            actions.append(dict(multiplier=m,seconds=seconds,
                time_mad=median(abs(v-seconds) for v in values),valid=all(r['accurate'] for r in group),
                eta=group[0]['expected_eta'],newton=median(r['newton_iterations'] for r in group),
                linear=median(r['linear_iterations'] for r in group)))
        eligible=[a for a in actions if a['valid']]
        best=min(eligible,key=lambda a:a['seconds']) if eligible else None
        continuation=next(a for a in actions if a['multiplier']==1.)
        state_points.append(dict(case=case,iteration=iteration,actions=actions,best=best,
            remaining_ratio_to_continuation=best['seconds']/continuation['seconds'] if best else None))
    pairs=[]
    for case in range(12):
        point=next((p for p in state_points if p['case']==case and p['iteration']==0),None)
        base=next(p for p in selection['methods'][selection['primary']]['points'] if p['n']==64 and p['case']==case)
        best=point['best'] if point else None
        pairs.append(dict(case=case,baseline_seconds=base['seconds'],baseline_mad=base['time_mad'],
            best_action=best,ratio=best['seconds']/base['seconds'] if best else None))
    valid=bool(len(pairs)==12 and all(r['accurate'] for r in rows)
        and all(s['source_accurate'] for s in preparation['sequences'])
        and all(p['ratio'] is not None for p in pairs))
    ratios=[p['ratio'] for p in pairs if p['ratio'] is not None]
    med=median(ratios) if ratios else None
    faster=sum(r<1 for r in ratios)
    passed=bool(valid and med<=.85 and faster>=9)
    return dict(created_utc=stamp(),numerically_valid=valid,states=state_points,step0_pairs=pairs,
        step0_median_ratio=med,step0_faster_instances=faster,opportunity_gate_passed=passed,
        status='proceed_to_baseline_and_novelty_validation' if passed else 'stop_current_action_family',
        training_authorized=False,
        scope='Development retrospective first-action selection; no learned or deployable policy, no combined multi-state oracle.',
        missing_baselines_and_novelty=['Botti2015 full formula not yet verified','Oladokun2020 full text not yet verified'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run']);parser.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        instances=freeze()
        truth,valid=numerical_verification(instances)
        if not valid:
            save(OUT/'complete.json',dict(complete=True,status='stopped_numerical_verification',training_authorized=False))
            raise RuntimeError('Numerical verification failed; formal timing not started')
        check_freeze()
        selection=run_baselines(instances,truth)
        if selection['continuation'] is None or selection['primary'] is None:
            save(OUT/'complete.json',dict(complete=True,status='stopped_invalid_baselines',training_authorized=False))
            return
        states,preparation=prepare_snapshots(instances,selection['continuation'],truth)
        check_freeze()
        rows=run_branches(states,selection['continuation'],truth)
        check_freeze()
        assert selection['baseline_sha256']==sha(OUT/'baselines.json')
        decision=assess(rows,selection,preparation)
        decision['freeze_sha256']=sha(OUT/'freeze.json')
        decision['input_sha256']={p.name:sha(p) for p in sorted(OUT.glob('*.json'))}
        save(OUT/'decision.json',decision)
        save(OUT/'complete.json',dict(complete=True,finished_utc=stamp(),status=decision['status'],
            decision_sha256=sha(OUT/'decision.json'),training_authorized=False,
            baseline_rows=504,branch_rows=len(rows),source_runs=12,direct_references=24,manufactured_runs=8,warmup_runs=7))
        print(json.dumps({k:decision[k] for k in ['status','step0_median_ratio','step0_faster_instances','opportunity_gate_passed']}),flush=True)


if __name__=='__main__':main()
