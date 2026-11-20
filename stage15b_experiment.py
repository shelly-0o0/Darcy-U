"""Predeclared single-goal classical development screen; no learning."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import argparse
import gzip
import hashlib
import json
import platform
import numpy as np
import scipy
from threadpoolctl import threadpool_info
from stage15_fem import make_mesh,assemble
from stage15b_cases import cases,mesh_audit
from stage15b_adaptive import run_adaptive,solve_columns,accepted,array_hash
from stage15b_reference import REFERENCE_CONFIG,build_reference
import skfem

ROOT=Path(__file__).resolve().parent
METHODS=['uniform_direct','uniform_amg','coefficient','primal_residual','joint_residual','dwr','fixed_direct','fixed_amg']
CONFIG=dict(taus=[1e-3,3e-4],methods=METHODS,repeats=3,initial_n=8,fixed_n=256,
            adaptive=dict(theta=.5,safety=.5,max_steps=13,max_free_dofs=70000,max_seconds=60.,
                          intorder=10,verification_interval=3,verification_delta_factor=.5),
            reference=REFERENCE_CONFIG,randomized_statistical_sample=False,models_trained=0)
SOURCES=['stage15b_cases.py','stage15b_adaptive.py','stage15b_reference.py','stage15b_experiment.py',
         'test_stage15b.py','report_stage15b.py','stage15_cases.py','stage15_fem.py','stage14_solver.py',
         'sequence_baselines.py','requirements.txt','docs/stage15b_protocol_zh.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n'
    if str(path).endswith('.gz'):
        with gzip.open(path,'wt') as stream: stream.write(data)
    else: Path(path).write_text(data)


def assess_output(summary,reference,tau):
    qualified=reference['assessment']['accepted']
    qoi=summary.get('final_qoi')
    finite=qoi is not None and bool(np.isfinite(qoi))
    error=abs(qoi-reference['assessment']['qoi']) if finite else None
    q=error/tau if finite else None
    completed=summary['status'] in ('estimated_converged','fixed_mesh_completed')
    success=bool(qualified and completed and finite and q<=1.)
    if not qualified: status='reference_unqualified'
    elif not completed: status=summary['status']
    elif not finite or q>1.: status='accuracy_failed'
    else: status='success'
    return dict(reference_qualified=qualified,online_completed=completed,
                absolute_qoi_error=error,q=q,success=success,status=status,
                acceptance_is_relative_to_empirically_checked_reference=True,
                continuous_error_bound=False)


def fixed_path(case,method,config):
    start=perf_counter(); tick=start
    mesh=make_mesh(config['fixed_n']); mesh_seconds=perf_counter()-tick
    tick=perf_counter()
    p=assemble(mesh,case.coefficient,(case.source,case.goal),degree=1,intorder=config['adaptive']['intorder'],
               coefficient_alignment='element_aligned')
    assembly=perf_counter()-tick
    tick=perf_counter()
    u,metrics=solve_columns(p,(0,),method)
    solve_seconds=perf_counter()-tick
    qoi=float(p.loads[:,1]@u[:,0])
    status='fixed_mesh_completed' if accepted(metrics) else 'linear_solver_failed'
    elapsed=perf_counter()-start
    if elapsed>config['adaptive']['max_seconds']: status='time_limit'
    if len(p.free)>config['adaptive']['max_free_dofs']: status='max_dofs'
    row=dict(step=0,marker='fixed',qoi=qoi,status=status,verified=False,verification_reasons=[],dwr=None,
             elements=int(mesh.nelements),vertices=int(mesh.nvertices),p1_free_dofs=len(p.free),p2_free_dofs=0,
             mesh_coordinates_sha256=array_hash(mesh.p),mesh_connectivity_sha256=array_hash(mesh.t),
             mesh_sort_t=bool(mesh.sort_t),solution_sha256=array_hash(u[:,0]),selected_elements=[],scores=None,
             solves=[dict(role='p1_primal',degree=1,metrics=metrics)],
             timings=dict(p1_assembly_seconds=assembly,p1_solve_seconds=solve_seconds,
                          step_total_seconds=elapsed-mesh_seconds),algorithm_elapsed_seconds=elapsed-mesh_seconds)
    summary=dict(status=status,estimated_converged=False,solved_states=1,final_qoi=qoi,solver_calls=1,rhs_count=1,
                 peak_p1_free_dofs=len(p.free),peak_p2_free_dofs=0,algorithm_seconds=elapsed,total_seconds=elapsed,
                 initial_mesh_seconds=mesh_seconds,kernel_algorithm_seconds=elapsed-mesh_seconds,artifact_io_seconds=0.,
                 config=dict(solver=method,**config['adaptive']),fixed_n=config['fixed_n'],
                 no_continuous_error_certificate=True,no_reference_used_online=True)
    return summary,[row],mesh,u[:,0]


def execute_path(case,method,tau,config):
    if method.startswith('fixed_'):
        return fixed_path(case,method.split('_')[1],config)
    tick=perf_counter(); mesh=make_mesh(config['initial_n']); creation=perf_counter()-tick
    marker='uniform' if method.startswith('uniform_') else method
    solver='amg' if method=='uniform_amg' else 'direct'
    result=run_adaptive(case.coefficient,case.source,case.goal,initial_mesh=mesh,tau=tau,marker=marker,
                        config=dict(config['adaptive'],solver=solver))
    summary=result.summary()
    summary['kernel_algorithm_seconds']=summary['algorithm_seconds']
    summary['initial_mesh_seconds']=creation
    summary['algorithm_seconds']+=creation
    summary['total_seconds']+=creation
    return summary,result.rows,result.final_mesh,result.final_solution


def run(output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a new directory; never overwrite audit results')
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools): raise RuntimeError('Set BLAS thread counts to one')
    if skfem.__version__!='12.0.2': raise RuntimeError('scikit-fem 12.0.2 required')
    output.mkdir(parents=True); (output/'paths').mkdir(); (output/'payloads').mkdir()
    materials=cases()
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,
                cases=[dict(name=c.name,metadata=c.metadata) for c in materials],
                source_sha256={name:sha(ROOT/name) for name in SOURCES},
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                 skfem=skfem.__version__,threadpools=pools),
                scope='Six fixed aligned single-goal non-manufactured problems; development screen only')
    dump(output/'freeze.json',freeze)
    start=perf_counter(); references=[]; index=[]; warmups=[]
    try:
        for method in METHODS:
            warm=dict(CONFIG,initial_n=4,fixed_n=8,adaptive=dict(CONFIG['adaptive'],max_steps=2))
            summary,_,_,_=execute_path(materials[0],method,1e-6,warm)
            warmups.append(dict(method=method,summary=summary))
        dump(output/'warmups.json',warmups)
        for case in materials:
            references.append(build_reference(case,CONFIG['reference']))
            dump(output/'references.json',references)
            print(json.dumps(dict(event='reference',case=case.name,assessment=references[-1]['assessment'])),flush=True)
        for ci,case in enumerate(materials):
            reference=references[ci]
            for ti,tau in enumerate(CONFIG['taus']):
                for repeat in range(CONFIG['repeats']):
                    offset=(ci+ti+repeat)%len(METHODS)
                    order=METHODS[offset:]+METHODS[:offset]
                    for method in order:
                        summary,rows,mesh,u=execute_path(case,method,tau,CONFIG)
                        stem=f'{case.name}_t{ti}_{method}_r{repeat}'
                        path_file='paths/'+stem+'.json.gz'; payload_file='payloads/'+stem+'.npz'
                        entry=dict(case=case.name,tau=tau,method=method,repeat=repeat,
                                   path_file=path_file,payload_file=payload_file,summary=summary,
                                   offline=assess_output(summary,reference,tau))
                        audit=mesh_audit(mesh,case)
                        entry['final_mesh_check']=audit
                        if not audit['accepted']:
                            entry['offline'].update(success=False,status='mesh_failed')
                        dump(output/path_file,dict(case=case.name,tau=tau,method=method,repeat=repeat,summary=summary,rows=rows))
                        np.savez_compressed(output/payload_file,p=mesh.p,t=mesh.t,
                                            u=np.asarray([] if u is None else u),sort_t=bool(mesh.sort_t))
                        index.append(entry); dump(output/'index.json',index)
                        print(json.dumps(dict(event='path',count=len(index),case=case.name,tau=tau,method=method,
                            repeat=repeat,status=entry['offline']['status'],q=entry['offline']['q'],
                            seconds=summary['algorithm_seconds'])),flush=True)
        for name,digest in freeze['source_sha256'].items():
            if sha(ROOT/name)!=digest: raise RuntimeError('Frozen source changed: '+name)
        files=sorted(p for p in output.rglob('*') if p.is_file())
        dump(output/'complete.json',dict(complete=True,paths=len(index),reference_cases=len(references),
             wall_seconds=perf_counter()-start,file_sha256={str(p.relative_to(output)):sha(p) for p in files},
             models_trained=0,G1='not_approved',G2='not_evaluated',G3='not_evaluated'))
    except Exception as exc:
        dump(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),paths_completed=len(index),
                                       reference_cases_completed=len(references),wall_seconds=perf_counter()-start))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/stage15b_classical')
    args=parser.parse_args(); run(ROOT/args.output)
