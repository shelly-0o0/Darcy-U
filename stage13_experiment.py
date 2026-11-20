"""Cost feasibility of reference-free mesh proposals without a global Schur matrix.

No training or AI inference. Local paths assemble only their actual constrained
operators; fine-scale coefficient sampling and RHS quadrature remain charged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import traceback
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from threadpoolctl import threadpool_info

import stage10_experiment as old
import stage11_experiment as prior
from numerics import _full_cell_matrix, solve_pcg
from sequence_baselines import FixedAMGHierarchy
from stage10_goal_numerics import _vertices
from stage11_adaptive import budget_mark, build_space, count_added_dofs
from stage13_local_fem import assemble_local, lift_to_fine, prepare_loads

ROOT=Path(__file__).resolve().parent
METHODS=("fine_direct","fine_amg_pcg","jacobi8","coefficient_local",
         "coarse_goal_local","coarse_goal_galerkin")
CONFIG=dict(grids=[16,32,64],budget_divisor=4,geometries_per_split=16,repetitions=3,
    splits={"id":54000,"contrast":54100},source=0,goals=[1,2,3],
    order_seed=54991,warmup_geometry=54950,primary_method="coarse_goal_local",
    gate_nc=64,time_ratio_limit=.8,minimum_wins=12,pcg_rtol=1e-10,
    pcg_maxiter=1000,pcg_max_restarts=3,amg_seed=51700,
    residual_limit=1e-8,near_zero_coarse_error=1e-12,
    fine_pcg_error_ratio_limit=1e-5,matched_solution_tolerance=1e-8,
    matched_scaled_qoi_tolerance=1e-8)


def coefficient_statistics(k):
    nc=len(k)//2
    cells=np.asarray(k).reshape(nc,2,nc,2).transpose(0,2,1,3)
    return cells.mean(axis=(2,3)),np.log(cells).mean(axis=(2,3)),np.log(cells).std(axis=(2,3))


def coefficient_scores(k):
    _,mean,variation=coefficient_statistics(k)
    value=variation.copy()
    dx=abs(mean[:,1:]-mean[:,:-1]);dy=abs(mean[1:]-mean[:-1])
    value[:,1:]+=dx;value[:,:-1]+=dx
    value[1:]+=dy;value[:-1]+=dy
    return value.ravel()


def coarse_goal_scores(k,coarse,solutions):
    """Fixed unsigned gradient proxy, not a goal-error estimator or certificate."""
    nc=coarse.nc;nodes=coarse.independent_free_node_ids
    yy,xx=np.divmod(nodes,2*nc+1)
    full=np.zeros((nc+1,nc+1,4))
    full[yy//2,xx//2]=solutions
    sw,se,ne,nw=full[:-1,:-1],full[:-1,1:],full[1:,1:],full[1:,:-1]
    gx=np.stack((se-sw,ne-nw),axis=2)*nc
    gy=np.stack((ne-se,nw-sw),axis=2)*nc
    norms=np.sqrt(gx*gx+gy*gy)
    qc=coarse.loads[1:]@solutions[:,0]
    scales=np.maximum(abs(qc),max(float(abs(qc).max())*1e-12,1e-30))
    mean,_,variation=coefficient_statistics(k)
    score=mean*(1+variation)/nc**2*np.mean(norms[...,0]*
        np.sum(norms[...,1:]/scales,axis=-1),axis=2)
    return score.ravel(),qc


def _direct_solve(a,loads):
    u=sla.splu(a.tocsc()).solve(loads[0])
    qoi=loads[1:]@u
    residual=float(np.linalg.norm(a@u-loads[0])/max(np.linalg.norm(loads[0]),1e-300))
    return u,qoi,residual


def pipeline(geometry,nc,ood,method,budget):
    if method not in METHODS:raise ValueError(method)
    if method=="jacobi8":
        value=prior.online_pipeline(geometry,nc,ood,"jacobi8",budget=budget)
        value.update(solver_converged=value["relative_residual"]<=CONFIG["residual_limit"],
                     assembly_backend="historical_fine_galerkin",uses_global_fine_matrix=True)
        return value
    start=tick=perf_counter();timings={}
    def lap(name):
        nonlocal tick
        now=perf_counter();timings[name]=now-tick;tick=now
    k=old.coefficient(geometry,2*nc,ood);funcs=old.rhs_functions(geometry)
    lap("geometry_seconds")
    if method.startswith("fine_"):
        _,free=_vertices(2*nc)
        a=_full_cell_matrix(k)[free][:,free].tocsr()
        loads=prepare_loads(nc,funcs).fine_free_loads
        lap("fine_assembly_seconds")
        stats={}
        if method=="fine_direct":
            u,qoi,residual=_direct_solve(a,loads)
            converged=residual<=CONFIG["residual_limit"]
            lap("fine_solve_qoi_seconds")
        else:
            hierarchy=FixedAMGHierarchy(a,seed=CONFIG["amg_seed"])
            lap("amg_setup_seconds")
            u,stats=solve_pcg(a,loads[0],M=hierarchy.preconditioner,
                rtol=CONFIG["pcg_rtol"],maxiter=CONFIG["pcg_maxiter"],
                max_restarts=CONFIG["pcg_max_restarts"])
            qoi=loads[1:]@u;residual=stats["relative_residual"]
            converged=stats["converged"] and stats["info"]==0
            lap("fine_pcg_qoi_seconds")
        timings["total_seconds"]=perf_counter()-start
        return dict(method=method,u=u,qoi=qoi,selected_cells=[],
            added_dofs=len(u)-(nc-1)**2,solve_dofs=len(u),budget_matched=False,
            relative_residual=residual,solver_converged=bool(converged),solver=stats,
            uses_global_fine_matrix=True,assembly_backend="fine",timings=timings)

    coarse=None;qc=None;coarse_residual=None
    if method.startswith("coarse_goal"):
        coarse=assemble_local(nc,k,funcs,[])
        lap("coarse_assembly_seconds")
        solutions=sla.splu(coarse.A).solve(coarse.loads.T)
        coarse_residual=float(np.max(np.linalg.norm(coarse.A@solutions-coarse.loads.T,axis=0)/
                              np.maximum(np.linalg.norm(coarse.loads,axis=1),1e-300)))
        lap("coarse_factor_four_solve_seconds")
        scores,qc=coarse_goal_scores(k,coarse,solutions)
    else:scores=coefficient_scores(k)
    lap("proposal_features_seconds")
    cells,marking=budget_mark(scores,nc,budget)
    lap("budget_decode_seconds")
    if method=="coarse_goal_galerkin":
        nf=2*nc;_,free=_vertices(nf)
        a=_full_cell_matrix(k)[free][:,free].tocsr()
        loads=coarse.load_data.fine_loads_full[:,free]
        lap("fine_matrix_seconds")
        full_to_free=np.full((nf+1)**2,-1,dtype=np.int64)
        full_to_free[free]=np.arange(len(free))
        shape_only=sp.csr_matrix((len(free),(nc-1)**2))
        minimal=SimpleNamespace(nc=nc,nf=nf,Af=a,P=shape_only,free=free,
            full_to_free=full_to_free,dimension=len(free),ff=loads[0])
        t,metadata=build_space(minimal,cells)
        lap("mesh_space_seconds")
        aa=(t.T@a@t).tocsc();bb=np.asarray(t.T@loads.T).T
        lap("adaptive_assembly_seconds")
        u,qoi,residual=_direct_solve(aa,bb)
        lap("adaptive_solve_qoi_seconds")
        private={"_T":t,"_coefficients":u}
        assembly={"matrix_nnz":int(aa.nnz),"geometric_triangle_count":metadata["geometric_triangle_count"]}
        dofs=aa.shape[0]
        independent_nodes=metadata["independent_free_node_ids"]
    else:
        problem=assemble_local(nc,k,funcs,cells,
            load_data=None if coarse is None else coarse.load_data)
        lap("adaptive_assembly_seconds")
        u,qoi,residual=_direct_solve(problem.A,problem.loads)
        lap("adaptive_solve_qoi_seconds")
        private={"_problem":problem,"_coefficients":u}
        assembly=dict(timings=problem.timings,metadata=problem.metadata)
        dofs=problem.A.shape[0]
        independent_nodes=problem.independent_free_node_ids
    timings["total_seconds"]=perf_counter()-start
    marking["scores"]=scores.tolist()
    result=dict(method=method,qoi=qoi,selected_cells=cells.tolist(),added_dofs=count_added_dofs(cells,nc),
        solve_dofs=dofs,budget_matched=dofs-(nc-1)**2==budget,relative_residual=residual,
        solver_converged=bool(residual<=CONFIG["residual_limit"]),marking=marking,
        coarse_relative_residual=coarse_residual,local_assembly=assembly,
        independent_node_sha256=hashlib.sha256(np.asarray(independent_nodes,dtype=np.int64).tobytes()).hexdigest(),
        assembly_backend="fine_galerkin" if method.endswith("galerkin") else "direct_local",
        uses_global_fine_matrix=method.endswith("galerkin"),timings=timings,**private)
    if qc is not None:result["coarse_qoi"]=qc
    return result


def lifted_solution(value):
    if "u" in value:return value["u"]
    if "_T" in value:return np.asarray(value["_T"]@value["_coefficients"])
    return lift_to_fine(value["_problem"],value["_coefficients"])


def evaluate(value,reference):
    u=lifted_solution(value);qoi=np.asarray(value["qoi"])
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(qoi)):
        raise RuntimeError("Nonfinite field or QoI; retain interrupted output for diagnosis")
    error=reference["qf"]-qoi
    scaled=float(np.linalg.norm(error/reference["scales"]))
    denom=reference["coarse_scaled_error"];near=denom<CONFIG["near_zero_coarse_error"]
    delta=reference["uf"]-u
    energy=float(delta@(reference["A"]@delta)/reference["energy"])
    qc_discrepancy=float(np.linalg.norm((np.asarray(value.get("coarse_qoi",reference["qc"]))-
                                       reference["qc"])/reference["scales"]))
    valid=bool(value["solver_converged"] and np.isfinite(value["relative_residual"])
        and value["relative_residual"]<=CONFIG["residual_limit"] and -1e-12<=energy<=1+1e-8
        and qc_discrepancy<=1e-8 and
        (value.get("coarse_relative_residual") is None or value["coarse_relative_residual"]<=1e-8)
        and value.get("coarse_embedding_max_abs",0)<=1e-12
        and value.get("projection_orthogonality",0)<=1e-8)
    if value["method"]=="fine_amg_pcg":
        valid=bool(valid and value["solver"]["info"]==0 and value["relative_residual"]<=CONFIG["pcg_rtol"]
                   and not near and scaled/denom<=CONFIG["fine_pcg_error_ratio_limit"])
    result={k:v for k,v in value.items() if not k.startswith("_") and
            k not in ("u","qoi","coarse_qoi","scales","timings")}
    result.update(qoi=qoi.tolist(),fine_qoi=reference["qf"].tolist(),
        coarse_qoi=reference["qc"].tolist(),qoi_scales=reference["scales"].tolist(),
        signed_qoi_errors=error.tolist(),signed_coarse_errors=(reference["qf"]-reference["qc"]).tolist(),
        absolute_error_l2=float(np.linalg.norm(error)),scaled_error_l2=scaled,
        coarse_scaled_error=denom,coarse_error_near_zero=near,
        error_reduction_ratio=None if near else scaled/denom,energy_error_ratio=energy,
        coarse_scaled_qoi_discrepancy=qc_discrepancy,numerical_valid=valid,
        solution_sha256=hashlib.sha256(u.tobytes()).hexdigest())
    return result


def input_files():
    prior_freeze=json.loads((ROOT/"results/stage12_actions/freeze.json").read_text())
    for name,digest in prior_freeze["input_sha256"].items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError(f"Historical frozen input changed: {name}")
    return sorted(set([*prior_freeze["input_sha256"],"stage13_experiment.py","stage13_local_fem.py",
        "test_stage13_experiment.py","test_stage13_local_fem.py","docs/stage13_protocol_zh.md",
        "results/stage12_actions/freeze.json","results/stage12_actions/complete.json",
        "results/stage12_actions/contexts.json"]))


def historical_hashes():
    seen=set()
    for name in ("training_data.json","test_data.json"):
        data=json.loads((ROOT/"results/stage10_pilot"/name).read_text())
        entries=data if isinstance(data,list) else data["train"]+data["validation"]
        seen.update(x["coefficient_sha256"] for x in entries)
    for stage in ("stage11_refinement","stage12_actions"):
        seen.update(x["coefficient_sha256"] for x in json.loads((ROOT/f"results/{stage}/contexts.json").read_text()))
    return seen


def run(output,smoke=False):
    if output.exists():raise FileExistsError("Use a fresh directory; never overwrite frozen results")
    config=dict(CONFIG)
    if smoke:config.update(grids=[8],geometries_per_split=1,repetitions=1,
                           splits={"id":54970,"contrast":54980},gate_nc=8,minimum_wins=1)
    hashes={name:old.sha(ROOT/name) for name in input_files()}
    if any(p["num_threads"]!=1 for p in threadpool_info()):raise RuntimeError("Single-thread pools required")
    output.mkdir(parents=True)
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),smoke=smoke,config=config,
        methods=METHODS,input_sha256=hashes,no_training=True,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            torch=torch.__version__,device="cpu",platform=platform.platform(),threadpools=threadpool_info(),
            cpu_model=next((x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                            if x.startswith("model name")),"unknown"),
            thread_environment={n:os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        timing_scope="Three fresh single-thread CPU warm pipelines; coefficient sampling, all needed quadrature, coarse features, budget decoding, matrix assembly, factor/solve and QoIs/residual checks included. Local/Galerkin outputs adaptive coefficients and QoIs; lifting to fine representation and reference/error evaluation outside timing. No training or inference; no same-accuracy speedup claim.")
    old.dump(output/"freeze.json",freeze)
    tick=perf_counter()
    for nc in config["grids"]:
        for method in METHODS:pipeline(config["warmup_geometry"],nc,False,method,nc*nc//4)
    warm=perf_counter()-tick;rows=[];contexts=[];seen=historical_hashes();start=perf_counter()
    for nc in config["grids"]:
        for split,first in config["splits"].items():
            for geometry in range(first,first+config["geometries_per_split"]):
                budget=nc*nc//config["budget_divisor"]
                ref=prior.reference_case(geometry,nc,split=="contrast")
                if ref["fine_relative_residual"]>1e-8:raise RuntimeError("Reference solve failed")
                digest=ref["coefficient_sha256"]
                if digest in seen:raise RuntimeError("Historical or same-resolution coefficient collision")
                seen.add(digest)
                values={};first_rows={};samples={m:[] for m in METHODS};repeat_valid={m:True for m in METHODS}
                repeat_checks={m:[] for m in METHODS};method_orders=[]
                order=np.random.default_rng(config["order_seed"]+geometry+nc)
                for rep in range(config["repetitions"]):
                    permutation=order.permutation(len(METHODS))
                    method_orders.append([METHODS[i] for i in permutation])
                    for idx in permutation:
                        method=METHODS[idx];v=pipeline(geometry,nc,split=="contrast",method,budget)
                        samples[method].append(dict(repetition=rep,**v["timings"]))
                        evaluated=evaluate(v,ref)
                        repeat_valid[method]&=evaluated["numerical_valid"]
                        repeat_checks[method].append(dict(repetition=rep,numerical_valid=evaluated["numerical_valid"],
                            relative_residual=v["relative_residual"],solver_converged=v["solver_converged"],
                            solver_info=v.get("solver",{}).get("info"),error_reduction_ratio=evaluated["error_reduction_ratio"]))
                        if rep==0:values[method]=v;first_rows[method]=evaluated
                        elif v["selected_cells"]!=first_rows[method]["selected_cells"] or not np.allclose(
                                v["qoi"],first_rows[method]["qoi"],rtol=1e-12,atol=1e-15):
                            raise RuntimeError("Timing repeats changed selected mesh or numerical goals")
                local=values["coarse_goal_local"];galerkin=values["coarse_goal_galerkin"]
                ul=lifted_solution(local);ug=lifted_solution(galerkin)
                match=dict(same_mesh=local["selected_cells"]==galerkin["selected_cells"],
                    same_dofs=local["solve_dofs"]==galerkin["solve_dofs"],
                    same_independent_nodes=local["independent_node_sha256"]==galerkin["independent_node_sha256"],
                    relative_solution_difference=float(np.linalg.norm(ul-ug)/max(np.linalg.norm(ug),1e-300)),
                    scaled_qoi_difference=float(np.linalg.norm((local["qoi"]-galerkin["qoi"])/ref["scales"])))
                match["passed"]=bool(match["same_mesh"] and match["same_dofs"] and match["same_independent_nodes"] and
                    match["relative_solution_difference"]<=config["matched_solution_tolerance"] and
                    match["scaled_qoi_difference"]<=config["matched_scaled_qoi_tolerance"])
                for method in METHODS:
                    rows.append(dict(split=split,geometry=geometry,nc=nc,training_seed=None,offline_only=False,
                        **first_rows[method],all_repeats_numerically_valid=bool(repeat_valid[method]),
                        repeat_numerical_checks=repeat_checks[method],
                        matched_implementation=match if method.startswith("coarse_goal") else None,
                        timings=samples[method],median_total_seconds=float(np.median([t["total_seconds"] for t in samples[method]]))))
                contexts.append(dict(split=split,geometry=geometry,nc=nc,new_dofs=budget,
                    coarse_dofs=(nc-1)**2,fine_dofs=(2*nc-1)**2,coefficient_sha256=digest,
                    reference_seconds=ref["seconds"],fine_relative_residual=ref["fine_relative_residual"],
                    method_orders=method_orders))
                old.dump(output/"rows.json",rows);old.dump(output/"contexts.json",contexts)
                print(json.dumps(dict(event="cost_case",nc=nc,split=split,geometry=geometry,contexts=len(contexts),rows=len(rows))),flush=True)
    for name,digest in hashes.items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen input changed: {name}")
    old.dump(output/"complete.json",dict(complete=True,smoke=smoke,no_training=True,rows=len(rows),
        contexts=len(contexts),independent_geometries=2*config["geometries_per_split"],
        fresh_pipeline_runs=len(rows)*config["repetitions"],warmup_seconds=warm,
        experiment_wall_seconds=perf_counter()-start,freeze_sha256=old.sha(output/"freeze.json"),
        output_sha256={n:old.sha(output/n) for n in ("rows.json","contexts.json")}))
    print(json.dumps(dict(event="complete",rows=len(rows),output=str(output))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",default="results/stage13_cost")
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=(ROOT/args.output).resolve()
    try:run(output,args.smoke)
    except Exception:
        if output.exists() and not (output/"complete.json").exists():
            old.dump(output/"failure.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
                traceback=traceback.format_exc(),complete=False))
        raise


if __name__=="__main__":main()
