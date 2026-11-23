"""One frozen new-geometry classical confirmation cohort, no learning."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import argparse
import json
import platform
import numpy as np
import scipy
from threadpoolctl import threadpool_info
from stage15_fem import make_mesh
from stage15b_cases import cases as old_cases, mesh_audit
from stage15b_adaptive import array_hash
from stage15b_experiment import sha, dump, fixed_path, assess_output
from stage15b_reference import REFERENCE_CONFIG, build_reference
from stage15c_cases import cases
from stage15c_adaptive import run_adaptive
import skfem

ROOT=Path(__file__).resolve().parent
METHODS=['uniform_direct','uniform_amg','coefficient','primal_residual','joint_residual',
         'dwr','recovered_dwr3','recovered_dwr8','fixed_direct']
CONFIG=dict(taus=[1e-3,3e-4],methods=METHODS,repeats=3,initial_n=8,fixed_n=256,
            adaptive=dict(theta=.5,safety=.5,max_steps=13,max_free_dofs=70000,max_seconds=60.,
                          intorder=10,verification_interval=3,verification_delta_factor=.5),
            reference=REFERENCE_CONFIG,experiment_soft_seconds=5400.,models_trained=0,
            screening=dict(min_nonterminal_fraction=.30,max_hypothetical_vs_control=.80,
                           max_dwr_time_vs_control=.90,max_dwr_dofs_vs_fastest_control=.75,
                           min_geometries_per_family=4,require_both_taus=True,
                           require_all_methods_success=True),
            sampling='24 geometries from finite aligned pools, stratified without replacement')
SOURCES=['stage15c_cases.py','stage15c_adaptive.py','stage15c_experiment.py','report_stage15c.py','test_stage15c.py',
         'docs/stage15c_protocol_zh.md','stage15b_cases.py','stage15b_adaptive.py','stage15b_deferred.py',
         'stage15b_reference.py','stage15b_experiment.py','stage15_cases.py','stage15_fem.py',
         'stage14_solver.py','sequence_baselines.py','requirements.txt','vendor/skfem/quadrature.py']


def execute_path(case,method,tau,config):
    if method=='fixed_direct':
        return fixed_path(case,'direct',config)
    tick=perf_counter(); mesh=make_mesh(config['initial_n']); creation=perf_counter()-tick
    marker='uniform' if method.startswith('uniform_') else method
    solver='amg' if method=='uniform_amg' else 'direct'
    result=run_adaptive(case.coefficient,case.source,case.goal,initial_mesh=mesh,tau=tau,
                        marker=marker,config=dict(config['adaptive'],solver=solver))
    summary=result.summary()
    summary['kernel_algorithm_seconds']=summary['algorithm_seconds']
    summary['initial_mesh_seconds']=creation
    summary['algorithm_seconds']+=creation; summary['total_seconds']+=creation
    return summary,result.rows,result.final_mesh,result.final_solution


def run(output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Choose a new directory; frozen results must not be overwritten')
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools): raise RuntimeError('Set all BLAS threads to one')
    if skfem.__version__!='12.0.2': raise RuntimeError('scikit-fem 12.0.2 required')
    output.mkdir(parents=True); (output/'paths').mkdir(); (output/'payloads').mkdir()
    materials=cases()
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,
                cases=[dict(name=c.name,metadata=c.metadata) for c in materials],
                source_sha256={name:sha(ROOT/name) for name in SOURCES},
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                 skfem=skfem.__version__,threadpools=pools),
                scope='One new finite-pool aligned single-goal cohort; no AI or continuous certificate',
                payload_storage='Identical final p/t/u arrays share a content-addressed payload')
    dump(output/'freeze.json',freeze)
    start=perf_counter(); references=[]; index=[]; warmups=[]

    def check_budget():
        if perf_counter()-start>CONFIG['experiment_soft_seconds']:
            raise TimeoutError('Predeclared total soft time budget reached; retain partial cohort')

    try:
        for method in METHODS:
            warm=dict(CONFIG,initial_n=4,fixed_n=8,adaptive=dict(CONFIG['adaptive'],max_steps=2))
            summary,_,_,_=execute_path(old_cases()[0],method,1e-6,warm)
            warmups.append(dict(method=method,summary=summary))
        dump(output/'warmups.json',warmups)
        for case in materials:
            check_budget()
            references.append(build_reference(case,CONFIG['reference']))
            dump(output/'references.json',references)
            print(json.dumps(dict(event='reference',case=case.name,assessment=references[-1]['assessment'])),flush=True)
        for ci,case in enumerate(materials):
            for ti,tau in enumerate(CONFIG['taus']):
                for repeat in range(CONFIG['repeats']):
                    offset=(ci+ti+repeat)%len(METHODS)
                    for method in METHODS[offset:]+METHODS[:offset]:
                        check_budget()
                        summary,rows,mesh,u=execute_path(case,method,tau,CONFIG)
                        stem=f'{case.name}_t{ti}_{method}_r{repeat}'
                        path_file='paths/'+stem+'.json.gz'
                        u=np.asarray([] if u is None else u)
                        hashes=[array_hash(a) for a in (mesh.p,mesh.t,u)]
                        payload_id=__import__('hashlib').sha256(('|'.join(hashes)+str(bool(mesh.sort_t))).encode()).hexdigest()
                        payload_file='payloads/'+payload_id+'.npz'
                        audit=mesh_audit(mesh,case)
                        entry=dict(case=case.name,family=case.metadata['family'],tau=tau,method=method,
                                   repeat=repeat,path_file=path_file,payload_file=payload_file,summary=summary,
                                   offline=assess_output(summary,references[ci],tau),final_mesh_check=audit)
                        if not audit['accepted']: entry['offline'].update(success=False,status='mesh_failed')
                        dump(output/path_file,dict(case=case.name,tau=tau,method=method,repeat=repeat,summary=summary,rows=rows))
                        if not (output/payload_file).exists():
                            np.savez_compressed(output/payload_file,p=mesh.p,t=mesh.t,u=u,sort_t=bool(mesh.sort_t))
                        index.append(entry)
                        print(json.dumps(dict(event='path',count=len(index),case=case.name,tau=tau,method=method,
                            repeat=repeat,status=entry['offline']['status'],q=entry['offline']['q'],
                            seconds=summary['algorithm_seconds'])),flush=True)
                    dump(output/'index.json',index)
        for name,digest in freeze['source_sha256'].items():
            if sha(ROOT/name)!=digest: raise RuntimeError('Frozen source changed: '+name)
        files=sorted(p for p in output.rglob('*') if p.is_file())
        dump(output/'complete.json',dict(complete=True,paths=len(index),reference_cases=len(references),
             wall_seconds=perf_counter()-start,file_sha256={str(p.relative_to(output)):sha(p) for p in files},
             models_trained=0,screening='Recompute using frozen thresholds; not an AI speed claim'))
    except Exception as exc:
        dump(output/'index.json',index)
        dump(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),paths_completed=len(index),
                                       reference_cases_completed=len(references),wall_seconds=perf_counter()-start))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/stage15c_classical')
    args=parser.parse_args();run(ROOT/args.output)
