#!/usr/bin/env python3
"""Train and measure a complete Darcy-U CPU/GPU research pilot.

Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 for timings.
No reference pressure, test label or future frame is used by any online method.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if (ROOT/'vendor').exists(): sys.path.insert(0,str(ROOT/'vendor'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
import torch
import pyamg
from threadpoolctl import threadpool_limits

from fields import coefficient_sequence,smooth_rhs
from numerics import (assemble_system,build_patches,patch_spectral_basis,b_orthonormalize,
    certify_space,build_preconditioner,solve_pcg,energy_relative_error)
from learning import build_dataset,train_model,load_model,predict_directions,field_hash

METHODS=['amg','spectral_full','spectral_reuse','smooth_increment','unet_increment']


def relevant_coefficient(k,p):
    # A_D boundary diagonals include elements just outside the patch.
    # The anchor region therefore includes one extra cell in each direction.
    n=k.shape[0]
    return k[max(0,p.y0-1):min(n,p.y1+1),max(0,p.x0-1):min(n,p.x1+1)].copy()


def smooth_candidate(p,z,seed,directions=4,steps=8):
    rng=np.random.default_rng(seed)
    y=rng.standard_normal((len(p.ids),directions))/np.sqrt(np.maximum(np.diag(p.A_N),1e-30))[:,None]
    invdiag=1/np.maximum(np.diag(p.A_N),1e-30)
    for _ in range(steps): y-=(2/3)*invdiag[:,None]*(p.A_N@y)
    delta=b_orthonormalize(y,p.B,existing=z)
    return np.column_stack([z,delta])


def sequence_run(fields,events,method,args,model=None,model_seed=7,repeat=0):
    """Return online records and solutions; post-run error assessment is separate."""
    previous_patches=None; spaces=None; anchors=None; anchor_cells=None; bounds=None
    previous_solution=None; records=[]; solutions=[]
    for t,k in enumerate(fields):
        start=time.perf_counter()
        row={'method':method,'sequence_seed':events[0]['seed'],'model_seed':model_seed if method=='unet_increment' else -1,
             'repeat':repeat,'step':t,'event':events[t]['event'],'change_fraction':events[t]['change_fraction'],
             'n':args.n,'contrast':args.contrast,'gamma':args.gamma,'rtol':args.rtol,
             'assembly_seconds':0.,'patch_seconds':0.,'reuse_seconds':0.,'spectral_seconds':0.,
             'candidate_seconds':0.,'check_seconds':0.,'fallback_seconds':0.,'setup_seconds':0.,
             'direct_fallback_seconds':0.,'reused_patches':0,'old_checks':0,'old_check_passes':0,
             'candidate_checks':0,'candidate_passes':0,'spectral_repairs':0,'added_directions':0,
             'solver_fallback':False,'coarse_dimension':0,'check_diagnostics':[]}
        ts=time.perf_counter()
        f=None if args.rhs=='one' else lambda x,y:smooth_rhs(x,y,args.rhs)
        A,b,_=assemble_system(k,f)
        row['assembly_seconds']=time.perf_counter()-ts
        if method=='amg':
            ts=time.perf_counter()
            ml=pyamg.smoothed_aggregation_solver(A,symmetry='symmetric',
                presmoother=('gauss_seidel',{'sweep':'symmetric'}),
                postsmoother=('gauss_seidel',{'sweep':'symmetric'}))
            pre=ml.aspreconditioner(cycle='V')
            row['setup_seconds']=time.perf_counter()-ts
            row['coarse_dimension']=int(ml.levels[-1].A.shape[0])
        else:
            ts=time.perf_counter()
            patches=build_patches(k,A,args.core,args.overlap,previous_patches)
            row['patch_seconds']=time.perf_counter()-ts
            if spaces is None or method=='spectral_full':
                ts=time.perf_counter()
                spaces=[patch_spectral_basis(p,args.gamma,2*args.gamma) for p in patches]
                row['spectral_seconds']=time.perf_counter()-ts
                anchors=[relevant_coefficient(k,p) for p in patches]
                anchor_cells=[p.coefficient.copy() for p in patches]
                bounds=[2*args.gamma]*len(patches)
            else:
                needs=[]
                for i,p in enumerate(patches):
                    ts=time.perf_counter()
                    region=relevant_coefficient(k,p)
                    ratio=region/anchors[i]
                    lower=bounds[i]*float(ratio.min()/ratio.max())
                    row['reuse_seconds']+=time.perf_counter()-ts
                    if lower>=args.gamma*(1-1e-12):
                        row['reused_patches']+=1
                        continue
                    ts=time.perf_counter()
                    verdict=certify_space(p.A_N,p.B,spaces[i],args.gamma,method=args.check)
                    row['check_seconds']+=time.perf_counter()-ts; row['old_checks']+=1
                    row['check_diagnostics'].append({'kind':'old','patch':i,**verdict})
                    if verdict['passed']:
                        row['old_check_passes']+=1
                        anchors[i]=region; anchor_cells[i]=p.coefficient.copy(); bounds[i]=args.gamma
                    else: needs.append(i)
                if method in ('smooth_increment','unet_increment') and needs:
                    ts=time.perf_counter()
                    if method=='unet_increment':
                        candidates=predict_directions(model,[patches[i] for i in needs],
                            [anchor_cells[i] for i in needs],[spaces[i] for i in needs],args.device)
                    else:
                        candidates=[smooth_candidate(patches[i],spaces[i],
                            events[0]['seed']*10000+t*100+i,args.directions,args.smooth_steps) for i in needs]
                    row['candidate_seconds']=time.perf_counter()-ts
                    for i,candidate in zip(needs,candidates):
                        p=patches[i]
                        ts=time.perf_counter()
                        verdict=certify_space(p.A_N,p.B,candidate,args.gamma,method=args.check)
                        row['check_seconds']+=time.perf_counter()-ts; row['candidate_checks']+=1
                        row['check_diagnostics'].append({'kind':'candidate','patch':i,**verdict})
                        if verdict['passed']:
                            row['candidate_passes']+=1
                            row['added_directions']+=candidate.shape[1]-spaces[i].shape[1]
                            spaces[i]=candidate
                            anchors[i]=relevant_coefficient(k,p)
                            anchor_cells[i]=p.coefficient.copy(); bounds[i]=args.gamma
                        else:
                            ts=time.perf_counter()
                            spaces[i]=patch_spectral_basis(p,args.gamma,2*args.gamma)
                            row['fallback_seconds']+=time.perf_counter()-ts; row['spectral_repairs']+=1
                            anchors[i]=relevant_coefficient(k,p)
                            anchor_cells[i]=p.coefficient.copy(); bounds[i]=2*args.gamma
                elif method=='spectral_reuse':
                    for i in needs:
                        p=patches[i]; ts=time.perf_counter()
                        spaces[i]=patch_spectral_basis(p,args.gamma,2*args.gamma)
                        row['fallback_seconds']+=time.perf_counter()-ts; row['spectral_repairs']+=1
                        anchors[i]=relevant_coefficient(k,p)
                        anchor_cells[i]=p.coefficient.copy(); bounds[i]=2*args.gamma
            pre,setup=build_preconditioner(A,patches,spaces)
            row.update(setup)
            previous_patches=patches
        solution,metrics=solve_pcg(A,b,pre,previous_solution,args.rtol,args.maxiter)
        row.update({('solve_seconds' if key=='seconds' else key):value for key,value in metrics.items()})
        if not metrics['converged']:
            # A failed iterative attempt is retained, and the recovery time is counted.
            ts=time.perf_counter(); solution=sla.spsolve(A,b)
            row['direct_fallback_seconds']=time.perf_counter()-ts
            row['solver_fallback']=True
        row['final_relative_residual']=float(np.linalg.norm(b-A@solution)/np.linalg.norm(b))
        row['final_converged']=row['final_relative_residual']<=args.rtol
        row['total_seconds']=time.perf_counter()-start
        previous_solution=solution.copy(); solutions.append(solution.copy()); records.append(row)
        print(json.dumps({'stage':'benchmark','method':method,'seed':events[0]['seed'],'repeat':repeat,
            'step':t,'seconds':row['total_seconds'],'iterations':row['iterations'],
            'coarse_dim':row['coarse_dimension'],'candidate_passes':row['candidate_passes'],
            'repairs':row['spectral_repairs'],'residual':row['final_relative_residual']}),flush=True)
    return records,solutions


def write_records(records,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(records,indent=2))
    keys=sorted(set().union(*(row.keys() for row in records)))
    with path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(records)


def benchmark(args):
    torch.set_num_threads(args.torch_threads)
    out=ROOT/'results'/args.run_name; out.mkdir(parents=True,exist_ok=True)
    data_manifest=json.loads((ROOT/'data/pilot.manifest.json').read_text()) if (ROOT/'data/pilot.manifest.json').exists() else None
    if data_manifest:
        used=set(data_manifest['train_seeds']+data_manifest['val_seeds'])
        if used.intersection(args.test_seeds): raise ValueError('Train/test seed overlap')
    models={}
    if 'unet_increment' in args.methods:
        for seed in args.model_seeds:
            models[seed]=load_model(ROOT/f'checkpoints/unet_seed{seed}.pt',args.device)
            with torch.no_grad(): models[seed](torch.zeros(2,6,32,32,device=args.device))
    manifest={'configuration':vars(args),'python':sys.version,'platform':platform.platform(),
        'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'numpy':np.__version__,'scipy':scipy.__version__,
        'torch':torch.__version__,'pyamg':pyamg.__version__,'cuda_available':torch.cuda.is_available(),
        'blas_threads':1,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        'checkpoint_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for s in args.model_seeds
            for p in [ROOT/f'checkpoints/unet_seed{s}.pt',ROOT/f'checkpoints/unet_seed{s}.training.json'] if p.exists()},
        'dataset_manifest':data_manifest,'timing_note':'reference solves and energy assessment outside online timings; initial construction included; no timing runs concurrent with training',
        'limitations':['pilot synthetic family','numerical floating-point checks, not interval certificates',
            'AMG configuration is fixed, not exhaustively tuned','spectral fallback uses direct local eigensolves, not an optimized recycled eigensolver']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    all_records=[]
    for sequence_seed in args.test_seeds:
        fields,events=coefficient_sequence(args.n,args.steps,sequence_seed,args.contrast,args.family)
        digest=field_hash(fields)
        if data_manifest and digest in data_manifest['hashes'].values(): raise ValueError('Identical test/training sequence')
        np.savez_compressed(out/f'fields_seed{sequence_seed}.npz',k=fields)
        (out/f'events_seed{sequence_seed}.json').write_text(json.dumps(events,indent=2))
        refs=[]; ref_start=time.perf_counter()
        f=None if args.rhs=='one' else lambda x,y:smooth_rhs(x,y,args.rhs)
        for k in fields:
            a,b,_=assemble_system(k,f); reference=sla.spsolve(a,b)
            residual=float(np.linalg.norm(b-a@reference)/np.linalg.norm(b))
            refs.append((reference,residual))
        reference_info={'stage':'references','seed':sequence_seed,'seconds':time.perf_counter()-ref_start,
                        'max_residual':max(x[1] for x in refs),'sequence_sha256':digest}
        (out/f'reference_seed{sequence_seed}.json').write_text(json.dumps(reference_info,indent=2))
        print(json.dumps(reference_info),flush=True)
        for repeat in range(args.repeats):
            variants=[(m,s) for m in args.methods for s in (args.model_seeds if m=='unet_increment' else [-1])]
            # Deterministic random order reduces order/thermal bias.
            np.random.default_rng(sequence_seed+repeat).shuffle(variants)
            for method,model_seed in variants:
                records,solutions=sequence_run(fields,events,method,args,models.get(model_seed),model_seed,repeat)
                for t,(row,solution) in enumerate(zip(records,solutions)):
                    a,b,_=assemble_system(fields[t],f)
                    row['energy_relative_error']=energy_relative_error(a,solution,refs[t][0])
                    row['reference_relative_residual']=refs[t][1]
                all_records.extend(records)
                write_records(all_records,out/'measurements')
    print(json.dumps({'stage':'benchmark_complete','rows':len(all_records),'output':str(out)}),flush=True)
    return all_records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['data','train','benchmark'])
    p.add_argument('--n',type=int,default=128); p.add_argument('--steps',type=int,default=20)
    p.add_argument('--core',type=int,default=16); p.add_argument('--overlap',type=int,default=2)
    p.add_argument('--gamma',type=float,default=.03); p.add_argument('--contrast',type=float,default=1e4)
    p.add_argument('--family',default='mixed'); p.add_argument('--rhs',default='one',choices=['one','sine','source'])
    p.add_argument('--rtol',type=float,default=1e-8); p.add_argument('--maxiter',type=int,default=1000)
    p.add_argument('--directions',type=int,default=4); p.add_argument('--smooth-steps',type=int,default=8)
    p.add_argument('--check',default='eigen',choices=['eigen','ldl'])
    p.add_argument('--device',default='cpu'); p.add_argument('--torch-threads',type=int,default=2)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--model-seeds',type=int,nargs='+',default=[7])
    p.add_argument('--test-seeds',type=int,nargs='+',default=[90001])
    p.add_argument('--repeats',type=int,default=1); p.add_argument('--methods',nargs='+',default=METHODS,choices=METHODS)
    p.add_argument('--run-name',default='pilot_cpu'); p.add_argument('--train-sequences',type=int,default=48)
    p.add_argument('--val-sequences',type=int,default=12)
    args=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='data':
            build_dataset(ROOT/'data/pilot.pkl',args.train_sequences,args.val_sequences,args.n,args.steps,args.gamma,args.core,args.overlap)
        elif args.action=='train':
            with (ROOT/'data/pilot.pkl').open('rb') as f: data=pickle.load(f)
            for seed in args.model_seeds:
                train_model(data,ROOT/f'checkpoints/unet_seed{seed}.pt',args.epochs,args.batch_size,seed,
                            directions=args.directions,device=args.device,threads=args.torch_threads)
        else: benchmark(args)


if __name__=='__main__': main()
