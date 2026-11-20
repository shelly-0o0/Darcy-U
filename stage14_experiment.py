"""New solver confirmation and controlled finite-bank mesh intervention.

Candidate evaluation and reference-based selection are explicitly OFFLINE.
Three fresh complete baselines are timed separately with identical acceptance.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import traceback
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info

import stage10_experiment as old
import stage13_experiment as previous
from numerics import _full_cell_matrix
from stage10_goal_numerics import _vertices
from stage11_adaptive import budget_mark,count_added_dofs,_neighbors
from stage13_local_fem import assemble_local,prepare_loads,lift_to_fine
from stage14_solver import solve_system,CONFIG as SOLVER_CONFIG
from stage14_mechanism import BANK_NAMES,build_bank,error_metrics

ROOT=Path(__file__).resolve().parent
BASELINES=("fine_direct","fine_amg","coefficient_local")
CONFIG=dict(nc=64,budget=1024,geometries_per_split=16,
    splits={"id":55200,"contrast":55300},repetitions=3,
    primary_mode="locked_ring",modes=["locked_ring","coverage"],
    oracle_ratio_limit=.8,minimum_wins=12,improvement_ratio_limit=.9,
    robust_ratio_limit=1.05,near_zero=1e-12,fine_qoi_limit=1e-5,
    energy_limit=1+1e-8,matching_qoi_limit=1e-8,
    warmup_geometry=55950,order_seed=55991,audit_rhs_seed_offset=2000000)


def digest_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def baseline_pipeline(geometry,ood,nc,budget,method):
    if method not in BASELINES:raise ValueError(method)
    start=perf_counter();k=old.coefficient(geometry,2*nc,ood);fs=old.rhs_functions(geometry)
    cells=[]
    if method.startswith("fine_"):
        _,free=_vertices(2*nc);a=_full_cell_matrix(k)[free][:,free].tocsr()
        loads=prepare_loads(nc,fs).fine_free_loads
        u,solver=solve_system(a,loads[0],"direct" if method=="fine_direct" else "amg")
        private=dict(_u=u,_A=a)
    else:
        cells,_=budget_mark(previous.coefficient_scores(k),nc,budget)
        problem=assemble_local(nc,k,fs,cells)
        u,solver=solve_system(problem.A,problem.loads[0],"direct");loads=problem.loads
        private=dict(_problem=problem,_coefficients=u)
    q=loads[1:]@u
    seconds=perf_counter()-start
    return dict(method=method,qoi=q.tolist(),solver=solver,full_seconds=seconds,
        selected_cells=np.asarray(cells,dtype=int).tolist(),solve_dofs=len(u),**private)


def valid_solver(s):
    return bool(s["converged"] and s["info"]==0 and
        s["relative_residual"]<=SOLVER_CONFIG["relative_residual_limit"] and
        s["componentwise_backward_error"]<=SOLVER_CONFIG["componentwise_backward_limit"])


def public(v):return {k:x for k,x in v.items() if not k.startswith("_")}


def input_files():
    frozen=json.loads((ROOT/"results/stage13_cost/freeze.json").read_text())
    for name,digest in frozen["input_sha256"].items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError("Historical input modified: "+name)
    return sorted(set([*frozen["input_sha256"],"stage13_selection_diagnostic.py",
        "stage14_solver.py","test_stage14_solver.py","stage14_mechanism.py",
        "test_stage14_mechanism.py","stage14_experiment.py","test_stage14_experiment.py",
        "docs/stage14_protocol_zh.md","results/stage13_cost/freeze.json",
        "results/stage13_cost/complete.json","results/stage13_cost/contexts.json",
        "results/stage14_solver_development/freeze.json","results/stage14_solver_development/complete.json",
        "results/stage14_solver_development/rows.json"]))


def run(output,development=False,smoke=False):
    if output.exists():raise FileExistsError("Use a fresh output directory")
    config=dict(CONFIG)
    if development:config.update(splits={"id":55020,"contrast":55120},geometries_per_split=4,repetitions=1)
    if smoke:config.update(nc=32,budget=256,splits={"id":55970,"contrast":55980},geometries_per_split=1,repetitions=1)
    if any(p["num_threads"]!=1 for p in threadpool_info()):raise RuntimeError("Single thread pools required")
    if np.finfo(np.longdouble).nmant<=np.finfo(float).nmant:
        raise RuntimeError("This protocol requires extended precision residual arithmetic")
    hashes={name:old.sha(ROOT/name) for name in input_files()}
    if not (development or smoke):
        for name in ("freeze.json","complete.json","contexts.json","rows.json.gz"):
            path="results/stage14_mechanism_development/"+name
            hashes[path]=old.sha(ROOT/path)
    output.mkdir(parents=True)
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=config,
        solver_config=SOLVER_CONFIG,bank_names=list(BANK_NAMES),baseline_methods=list(BASELINES),
        development=development,smoke=smoke,no_training=True,input_sha256=hashes,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            torch=torch.__version__,threadpools=threadpool_info(),platform=platform.platform(),
            longdouble_bits=int(np.finfo(np.longdouble).nmant+1),
            cpu_model=next((s.split(":",1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
            thread_environment={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        timing_scope="Baseline full_seconds includes fresh geometry, four loads, assembly, selection when needed, solver setup/certification/refinement and three QoIs. Bank seconds are OFFLINE conditional solve/evaluation with prepaid seven-load quadrature; not end-to-end deployable times. All references, candidate building and evaluation paid in case offline wall time; no speed or AI gate.")
    old.dump(output/"freeze.json",freeze)
    seen=previous.historical_hashes()
    seen.update(c['coefficient_sha256'] for c in json.loads((ROOT/'results/stage13_cost/contexts.json').read_text()))
    if not (development or smoke):
        seen.update(c['coefficient_sha256'] for c in json.loads((ROOT/'results/stage14_mechanism_development/contexts.json').read_text()))
        seen.update(c['coefficient_sha256'] for c in json.loads((ROOT/'results/stage14_solver_development/rows.json').read_text()))
    tick=perf_counter()
    for method in BASELINES:baseline_pipeline(config['warmup_geometry'],False,config['nc'],config['budget'],method)
    warmup=perf_counter()-tick;start=perf_counter();contexts=[];rows=[]
    for split,first in config['splits'].items():
        for geometry in range(first,first+config['geometries_per_split']):
            case_start=perf_counter();nc=config['nc'];budget=config['budget'];ood=split=='contrast'
            k=old.coefficient(geometry,2*nc,ood);digest=digest_array(k)
            if digest in seen:raise RuntimeError("Coefficient collision with historical/development/confirmation data")
            seen.add(digest);fs=old.rhs_functions(geometry)
            # Both banks are built before reference solutions; audit functions
            # cannot enter their API (it requires exactly four given functions).
            bank_start=perf_counter();banks={}
            for mode in config['modes']:banks[mode],mask=build_bank(k,fs,geometry,budget,mode)
            bank_seconds=perf_counter()-bank_start
            ring=mask.copy()
            for cell in np.flatnonzero(mask):
                for other in _neighbors(cell,nc):ring[other]=True
            values=[];orders=[];rng=np.random.default_rng(config['order_seed']+geometry+nc)
            for repetition in range(config['repetitions']):
                methods=[BASELINES[i] for i in rng.permutation(len(BASELINES))];orders.append(methods)
                for method in methods:
                    value=baseline_pipeline(geometry,ood,nc,budget,method)
                    value['repetition']=repetition;values.append(value)
            fine=next(v for v in values if v['method']=='fine_direct' and v['repetition']==0)
            uf=fine['_u'];a=fine['_A']
            audit_fs=old.rhs_functions(geometry+config['audit_rhs_seed_offset'])[1:]
            all_fs=fs+audit_fs;load_start=perf_counter();load_data=prepare_loads(nc,all_fs)
            reference_load_seconds=perf_counter()-load_start
            coarse=assemble_local(nc,k,all_fs,[],load_data=load_data)
            uc,coarse_solver=solve_system(coarse.A,coarse.loads[0],'direct')
            qc=coarse.loads[1:]@uc;qf=load_data.fine_free_loads[1:]@uf
            delta=uf-lift_to_fine(coarse,uc);coarse_energy=float(delta@(a@delta))
            references=dict(known=error_metrics(qf[:3],qf[:3],qc[:3]),
                            audit=error_metrics(qf[3:],qf[3:],qc[3:]))
            baseline_runs=[]
            for value in values:
                known=error_metrics(value['qoi'],qf[:3],qc[:3])
                valid=valid_solver(value['solver']) and known['E'] is not None
                if value['method'].startswith('fine_'):valid=valid and known['E']<=config['fine_qoi_limit']
                run=public(value);run.update(known=known,numerical_valid=bool(valid));baseline_runs.append(run)
            repeat_deterministic=True
            for method in BASELINES:
                repeat=[v for v in values if v['method']==method]
                for value in repeat[1:]:
                    repeat_deterministic&=value['selected_cells']==repeat[0]['selected_cells'] and np.allclose(value['qoi'],repeat[0]['qoi'],rtol=1e-12,atol=1e-15)
            context=dict(split=split,geometry=geometry,nc=nc,budget=budget,
                coefficient_sha256=digest,known_load_sha256=digest_array(load_data.fine_free_loads[:4]),
                audit_load_sha256=digest_array(load_data.fine_free_loads[4:]),
                audit_rhs_seed=geometry+config['audit_rhs_seed_offset'],
                interface_cells=np.flatnonzero(mask).tolist(),ring_cells=np.flatnonzero(ring).tolist(),
                baseline_cells=banks['locked_ring'][0]['selected_cells'],references=references,
                coarse_solver=coarse_solver,coarse_energy=coarse_energy,
                method_orders=orders,baseline_runs=baseline_runs,
                repeat_deterministic=bool(repeat_deterministic),candidate_build_seconds=bank_seconds,
                reference_load_seconds=reference_load_seconds)
            baseline_q=np.asarray(next(v for v in values if v['method']=='coefficient_local')['qoi'])
            for mode,bank in banks.items():
                for item in bank:
                    t=perf_counter();cells=item['selected_cells']
                    problem=assemble_local(nc,k,all_fs,cells,load_data=load_data)
                    u,solver=solve_system(problem.A,problem.loads[0],'direct')
                    q=problem.loads[1:]@u
                    e=uf-lift_to_fine(problem,u);energy=float(e@(a@e)/coarse_energy) if coarse_energy>0 else None
                    known=error_metrics(q[:3],qf[:3],qc[:3]);audit=error_metrics(q[3:],qf[3:],qc[3:])
                    match=float(np.linalg.norm((q[:3]-baseline_q)/np.asarray(known['scales']))) if item['index']==0 else None
                    numerical_valid=bool(valid_solver(solver) and valid_solver(coarse_solver)
                        and item['interface_covered'] and (mode!='locked_ring' or item['ring_state_matched'])
                        and count_added_dofs(cells,nc)==budget and problem.dimension==(nc-1)**2+budget
                        and known['E'] is not None and audit['E'] is not None
                        and energy is not None and -1e-12<=energy<=config['energy_limit']
                        and (match is None or match<=config['matching_qoi_limit']))
                    rows.append(dict(split=split,geometry=geometry,nc=nc,budget=budget,**item,
                        solver=solver,known=known,audit=audit,energy_error_ratio=energy,
                        solve_dofs=problem.dimension,numerical_valid=numerical_valid,
                        baseline_scaled_qoi_difference=match,
                        selection_sha256=digest_array(np.asarray(cells,dtype=np.int64)),
                        solution_sha256=digest_array(u),seconds=perf_counter()-t,offline_only=True))
            context['offline_case_seconds']=perf_counter()-case_start;contexts.append(context)
            old.dump(output/'contexts.json',contexts)
            with gzip.open(output/'rows.json.gz','wt',encoding='utf-8') as stream:
                json.dump(rows,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            print(json.dumps(dict(event='stage14_case',split=split,geometry=geometry,contexts=len(contexts),rows=len(rows))),flush=True)
    for name,digest in hashes.items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError("Frozen input changed: "+name)
    old.dump(output/'complete.json',dict(complete=True,development=development,smoke=smoke,no_training=True,
        contexts=len(contexts),rows=len(rows),independent_geometries=len(contexts),
        baseline_pipeline_runs=len(contexts)*len(BASELINES)*config['repetitions'],
        offline_candidate_solves=len(rows),warmup_seconds=warmup,wall_seconds=perf_counter()-start,
        freeze_sha256=old.sha(output/'freeze.json'),
        output_sha256={n:old.sha(output/n) for n in ('contexts.json','rows.json.gz')}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/stage14_confirmation')
    parser.add_argument('--development',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=(ROOT/args.output).resolve()
    try:run(output,args.development,args.smoke)
    except Exception:
        if output.exists() and not (output/'complete.json').exists():
            old.dump(output/'failure.json',dict(complete=False,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
