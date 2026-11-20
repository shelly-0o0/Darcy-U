"""Non-learning feasibility of coupled, budgeted local-refinement actions.

The two oracle paths solve a fine primal to obtain their error target and are
explicitly offline. The AMG-target path never receives a fine solution. All
final reported errors are evaluated after an actual constrained FEM solve.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
import scipy.sparse.linalg as sla
import torch
from threadpoolctl import threadpool_info

import stage10_experiment as old
import stage11_experiment as previous
from stage10_goal_numerics import assemble_nested, assemble_load
from stage11_adaptive import build_space, solve_adaptive, count_added_dofs
from stage12_schur import SchurActions, candidate_patches
from sequence_baselines import FixedAMGHierarchy

ROOT=Path(__file__).resolve().parent
SEARCH_METHODS=("oracle_coupled","oracle_uncoupled","amg_coupled")
METHODS=(*SEARCH_METHODS,"amg_v1","jacobi8","recovery","oracle_indicator","fine_direct")
CONFIG=dict(nc=16,new_dofs=64,geometries_per_split=16,repetitions=3,
    splits={"id":53000,"contrast":53100},source=0,goals=[1,2,3],
    order_seed=53991,warmup_geometry=53950,near_zero_coarse_error=1e-12,
    paired_denominator_floor=1e-12,identity_tolerance=1e-8,
    accuracy_ratio_gate=.90,minimum_wins=12,cheap_baseline_ratio_limit=1.,
    primary_method="oracle_coupled",primary_fallback_invalidates_gate=True)


def neighbor_table(nc):
    return [tuple(n for allowed,n in ((c%nc>0,c-1),(c%nc<nc-1,c+1),
                    (c>=nc,c-nc),(c<nc*(nc-1),c+nc)) if allowed) for c in range(nc*nc)]


def available_actions(schur, patches, selected, neighbors, center_full, remaining):
    """Enumerate distinct unions, counting both centers and new shared edges."""
    seen=set(); choices=[]; extra=[]
    for patch in patches:
        new=tuple(c for c in patch["cells"] if not selected[c])
        if not new or new in seen:continue
        seen.add(new);newset=set(new)
        ids={int(schur.center_extra_indices[c]) for c in new}
        for cell in new:
            for other in neighbors[cell]:
                if selected[other] or other in newset:
                    ids.add(int(schur.full_to_extra[(center_full[cell]+center_full[other])//2]))
        if len(ids)>remaining:continue
        if min(ids)<0:raise RuntimeError("An action liberated a boundary/non-extra node")
        choices.append(dict(patch_id=patch["id"],kind=patch["kind"],new_cells=list(new)))
        extra.append(np.array(sorted(ids),dtype=np.int64))
    return choices,extra


def search_actions(schur, target, scales, budget, coupled):
    """Greedy gain-per-added-DOF search, retaining unfavorable steps and fallback."""
    target=np.asarray(target,dtype=float);scales=np.asarray(scales,dtype=float)
    if target.shape!=(schur.nrhs-1,) or scales.shape!=target.shape or not np.all(np.isfinite(target)) \
       or not np.all(np.isfinite(scales)) or np.any(scales<=0):raise ValueError("Invalid search target/scales")
    if not 0<budget<=schur.nc*schur.nc//2:raise ValueError("Unsupported exact-budget search")
    nc=schur.nc;nf=2*nc;patches=candidate_patches(nc);neighbors=neighbor_table(nc)
    yy,xx=np.mgrid[:nc,:nc];center_full=((2*yy+1)*(nf+1)+2*xx+1).ravel()
    selected=np.zeros(nc*nc,dtype=bool);chosen_extra=np.zeros(schur.nextra,dtype=bool)
    empty=schur.prepare([]);state=empty
    predicted=np.zeros_like(target);trace=[];evaluated=0
    singleton_changes=schur.incremental_changes(empty,[np.array([i]) for i in schur.center_extra_indices])[:,1:,0]
    base_objective=float(np.sum((target/scales)**2))
    singleton_scores=base_objective-np.sum(((target[None]-singleton_changes)/scales[None])**2,axis=1)
    while int(chosen_extra.sum())<budget:
        choices,extra=available_actions(schur,patches,selected,neighbors,center_full,budget-int(chosen_extra.sum()))
        if not choices:break
        if any(np.any(chosen_extra[ids]) for ids in extra):raise RuntimeError("Candidate double-counts an extra DOF")
        changes=schur.incremental_changes(state if coupled else empty,extra)[:,1:,0]
        before=float(np.sum(((target-predicted)/scales)**2))
        after=np.sum(((target[None]-predicted[None]-changes)/scales[None])**2,axis=1)
        costs=np.array([len(ids) for ids in extra])
        rates=(before-after)/costs
        if not np.all(np.isfinite(rates)):raise RuntimeError("Nonfinite candidate score")
        best=int(np.argmax(rates));chosen=choices[best];ids=extra[best]
        prior=predicted.copy()
        if coupled:
            state=schur.advance(state,ids);predicted=state.change[1:,0].copy()
            if not np.allclose(predicted,prior+changes[best],rtol=1e-9,atol=1e-16):
                raise RuntimeError("Rank update disagrees with selected candidate")
        else:predicted=prior+changes[best]
        selected[chosen["new_cells"]]=True;chosen_extra[ids]=True;evaluated+=len(choices)
        if not np.array_equal(np.flatnonzero(chosen_extra),schur.extra_indices(np.flatnonzero(selected))):
            raise RuntimeError("Action bookkeeping differs from actual mesh DOFs")
        trace.append(dict(step=len(trace),**chosen,new_extra_indices=ids.tolist(),cost=int(costs[best]),
            spent=int(chosen_extra.sum()),candidate_count=len(choices),gain_per_dof=float(rates[best]),
            predicted_qoi_change_before=prior.tolist(),predicted_increment=changes[best].tolist(),
            predicted_qoi_change_after=predicted.tolist(),objective_before=before,
            objective_after=float(np.sum(((target-predicted)/scales)**2)),
            negative_gain=bool(rates[best]<0)))
    fallback=int(chosen_extra.sum())!=budget
    partial=dict(selected_cells=np.flatnonzero(selected).tolist(),added_dofs=int(chosen_extra.sum()),
                 predicted_qoi_change=predicted.tolist())
    fallback_options=[]
    if fallback:
        parity=((yy+xx)%2).ravel();rank=np.lexsort((np.arange(nc*nc),-singleton_scores))
        for parity_value in (0,1):
            cells=rank[parity[rank]==parity_value][:budget]
            if len(cells)!=budget:raise RuntimeError("Checkerboard fallback cannot fill budget")
            change=(schur.qoi_change(schur.extra_indices(cells))[1:,0] if coupled
                    else singleton_changes[cells].sum(axis=0))
            objective=float(np.sum(((target-change)/scales)**2))
            fallback_options.append(dict(parity=parity_value,selected_cells=sorted(cells.tolist()),
                                         predicted_qoi_change=change.tolist(),objective=objective))
        pick=min(fallback_options,key=lambda x:(x["objective"],x["parity"]))
        selected[:]=False;selected[pick["selected_cells"]]=True
        predicted=np.array(pick["predicted_qoi_change"])
    final_cells=np.flatnonzero(selected);ids=schur.extra_indices(final_cells)
    if len(ids)!=budget:raise RuntimeError("Search did not meet exact DOF budget")
    exact_change=schur.qoi_change(ids)[1:,0]
    path_difference=float(np.linalg.norm((predicted-exact_change)/scales)/
                           max(np.linalg.norm(exact_change/scales),1e-12))
    if coupled and path_difference>CONFIG["identity_tolerance"]:
        raise RuntimeError("Incremental coupled search differs from direct Schur solve")
    return dict(selected_cells=final_cells.tolist(),trace=trace,added_dofs=len(ids),
        final_extra_indices=ids.tolist(),candidate_catalogue_size=len(patches),
        greedy_candidate_evaluations=evaluated,initial_singleton_evaluations=nc*nc,
        fallback_collection_evaluations=len(fallback_options),terminal_exact_evaluations=1,
        initial_singleton_scores=singleton_scores.tolist(),
        fallback=fallback,partial_before_fallback=partial,fallback_options=fallback_options,
        predicted_qoi_change=predicted.tolist(),exact_schur_qoi_change=exact_change.tolist(),
        ignored_coupling_relative_change_error=path_difference,
        target_error=target.tolist(),scales=scales.tolist(),coupled=coupled,
        negative_predicted_gain_steps=sum(x["negative_gain"] for x in trace))


def action_pipeline(geometry,nc,ood,method,budget):
    if method not in SEARCH_METHODS:raise ValueError(method)
    start=tick=perf_counter();timings={}
    def lap(name):
        nonlocal tick
        now=perf_counter();timings[name]=now-tick;tick=now
    k=old.coefficient(geometry,2*nc,ood);funcs=old.rhs_functions(geometry)
    case=assemble_nested(nc,k,funcs[0])
    loads=np.stack([case.ff]+[assemble_load(case,f)[0] for f in funcs[1:]])
    lap("geometry_assembly_seconds")
    lu=sla.splu(case.Ac.tocsc());coarse=lu.solve(np.asarray(case.P.T@loads.T))
    lifted=np.asarray(case.P@coarse);residual=(loads.T-case.Af@lifted).T
    qc=loads[1:]@lifted[:,0]
    scales=np.maximum(abs(qc),max(float(abs(qc).max())*1e-12,1e-30))
    lap("coarse_solve_residual_seconds")
    if method.startswith("oracle"):
        fine=sla.splu(case.Af.tocsc()).solve(loads[0])
        target=loads[1:]@(fine-lifted[:,0])
        label_residual=float(np.linalg.norm(case.Af@fine-loads[0])/np.linalg.norm(loads[0]))
        if label_residual>1e-8:raise RuntimeError("Oracle fine-primal solve failed")
        lap("oracle_fine_primal_target_seconds")
    else:
        hierarchy=FixedAMGHierarchy(case.Af,seed=51700)
        target=residual[1:]@(hierarchy.preconditioner@residual[0])
        label_residual=None
        lap("amg_target_setup_apply_seconds")
    schur=SchurActions(case,loads,coarse_lu=lu,coarse_solutions=coarse.T)
    lap("schur_setup_seconds")
    search=search_actions(schur,target,scales,budget,coupled=method!="oracle_uncoupled")
    lap("candidate_search_seconds")
    T,metadata=build_space(case,search["selected_cells"])
    lap("mesh_space_seconds")
    solved=solve_adaptive(case,T);u=solved["lifted_u"]
    lap("adaptive_solve_seconds")
    qoi=loads[1:]@u;actual_change=qoi-qc;schur_change=np.array(search["exact_schur_qoi_change"])
    identity=float(np.linalg.norm((actual_change-schur_change)/scales)/
                   max(np.linalg.norm(actual_change/scales),1e-12))
    if identity>CONFIG["identity_tolerance"]:raise RuntimeError("Schur change does not match actual FEM QoIs")
    difference=T@case.P[metadata["independent_fine_free_indices"]]-case.P
    embedding=float(np.max(abs(difference.data),initial=0.))
    interpolation_hash=hashlib.sha256(T.data.tobytes()+T.indices.tobytes()+T.indptr.tobytes()).hexdigest()
    lap("qoi_identity_diagnostics_seconds")
    timings["total_seconds"]=perf_counter()-start
    return dict(u=u,qoi=qoi,coarse_qoi=qc,scales=scales,method=method,
        selected_cells=search["selected_cells"],added_dofs=metadata["added_dofs"],
        budget_matched=True,solve_dofs=metadata["adaptive_dofs"],
        relative_residual=solved["relative_residual"],timings=timings,
        adaptive_solve_components=solved["timings"],search=search,
        coarse_embedding_max_abs=embedding,interpolation_sha256=interpolation_hash,
        geometric_triangle_count=metadata["geometric_triangle_count"],
        hanging_nodes=len(metadata["hanging_constraints"]),
        schur_identity_relative_error=identity,actual_qoi_change=actual_change.tolist(),
        oracle_label_relative_residual=label_residual,schur_components=schur.timings,
        schur_dimension=schur.nextra,schur_symmetry_defect=schur.symmetry_relative_defect,
        coarse_relative_residual=schur.coarse_relative_residual)


def pipeline(geometry,nc,ood,method,budget):
    if method in SEARCH_METHODS:return action_pipeline(geometry,nc,ood,method,budget)
    return previous.online_pipeline(geometry,nc,ood,method,budget=budget)


def input_files():
    prior=ROOT/"results/stage11_refinement/freeze.json"
    frozen=json.loads(prior.read_text())
    paths=["stage12_experiment.py","stage12_schur.py","test_stage12_experiment.py",
           "test_stage12_schur.py","docs/stage12_protocol_zh.md",
           "stage11_experiment.py","stage11_adaptive.py",*frozen["input_sha256"].keys(),
           "results/stage11_refinement/freeze.json","results/stage11_refinement/complete.json",
           "results/stage11_refinement/contexts.json"]
    for name,digest in frozen["input_sha256"].items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError(f"Prior frozen input changed: {name}")
    return sorted(set(paths))


def run(output,smoke=False):
    if output.exists():raise FileExistsError("Use a new output directory")
    config=dict(CONFIG)
    if smoke:config.update(nc=8,new_dofs=16,geometries_per_split=1,repetitions=1,
                           splits={"id":53970,"contrast":53980})
    files=input_files();hashes={name:old.sha(ROOT/name) for name in files}
    output.mkdir(parents=True)
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),smoke=smoke,config=config,
        methods=METHODS,input_sha256=hashes,no_training=True,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            torch=torch.__version__,device="cpu",torch_threads=torch.get_num_threads(),
            platform=platform.platform(),processor=platform.processor(),
            cpu_model=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")),"unknown") if Path("/proc/cpuinfo").exists() else platform.processor(),
            thread_environment={n:os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
            threadpools=threadpool_info()),
        timing_scope="Three fresh CPU warm runs; full setup, oracle target where used, Schur construction, candidate search, actual adaptive solve, QoI/identity checks; imports and offline evaluation/serialization excluded; no solver speedup claim.")
    old.dump(output/"freeze.json",freeze)
    tick=perf_counter()
    for method in METHODS:pipeline(config["warmup_geometry"],config["nc"],False,method,config["new_dofs"])
    warmup_seconds=perf_counter()-tick
    historical=set()
    for name in ("training_data.json","test_data.json"):
        data=json.loads((ROOT/"results/stage10_pilot"/name).read_text())
        entries=data if isinstance(data,list) else data["train"]+data["validation"]
        historical.update(x["coefficient_sha256"] for x in entries)
    historical.update(x["coefficient_sha256"] for x in json.loads((ROOT/"results/stage11_refinement/contexts.json").read_text()))
    rows=[];contexts=[];start=perf_counter()
    for split,first in config["splits"].items():
        for geometry in range(first,first+config["geometries_per_split"]):
            reference=previous.reference_case(geometry,config["nc"],split=="contrast")
            if not np.isfinite(reference["fine_relative_residual"]) or reference["fine_relative_residual"]>1e-8:
                raise RuntimeError("Invalid independent fine reference")
            digest=reference["coefficient_sha256"]
            if digest in historical:raise RuntimeError("Historical or current geometry collision")
            historical.add(digest)
            first_outputs={};samples={method:[] for method in METHODS}
            order=np.random.default_rng(config["order_seed"]+geometry)
            for repeat in range(config["repetitions"]):
                for index in order.permutation(len(METHODS)):
                    method=METHODS[index]
                    value=pipeline(geometry,config["nc"],split=="contrast",method,config["new_dofs"])
                    samples[method].append(dict(repetition=repeat,**value["timings"]))
                    if repeat==0:first_outputs[method]=previous.evaluate(value,reference)
                    else:
                        first_value=first_outputs[method]
                        if value["selected_cells"]!=first_value["selected_cells"] or not np.allclose(value["qoi"],first_value["qoi"],rtol=1e-12,atol=1e-15):
                            raise RuntimeError("Timing repeat changed the selected mesh or actual QoIs")
                        if method in SEARCH_METHODS and [x["patch_id"] for x in value["search"]["trace"]]!=[x["patch_id"] for x in first_value["search"]["trace"]]:
                            raise RuntimeError("Timing repeat changed search trajectory")
            for method in METHODS:
                rows.append(dict(split=split,geometry=geometry,training_seed=None,
                    offline_only=method.startswith("oracle"),**first_outputs[method],
                    timings=samples[method],median_total_seconds=float(np.median([x["total_seconds"] for x in samples[method]]))))
            contexts.append(dict(split=split,geometry=geometry,nc=config["nc"],new_dofs=config["new_dofs"],
                coefficient_sha256=digest,coarse_dofs=(config["nc"]-1)**2,fine_dofs=(2*config["nc"]-1)**2,
                reference_seconds=reference["seconds"],fine_relative_residual=reference["fine_relative_residual"]))
            old.dump(output/"rows.json",rows);old.dump(output/"contexts.json",contexts)
            print(json.dumps(dict(event="action_search",split=split,geometry=geometry,contexts=len(contexts),rows=len(rows))),flush=True)
    for name,digest in hashes.items():
        if old.sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen input changed: {name}")
    old.dump(output/"complete.json",dict(complete=True,smoke=smoke,no_training=True,rows=len(rows),
        contexts=len(contexts),fresh_pipeline_runs=len(rows)*config["repetitions"],
        warmup_seconds=warmup_seconds,experiment_wall_seconds=perf_counter()-start,
        freeze_sha256=old.sha(output/"freeze.json"),
        output_sha256={n:old.sha(output/n) for n in ("rows.json","contexts.json")}))
    print(json.dumps(dict(event="complete",rows=len(rows),output=str(output))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",default="results/stage12_actions")
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run((ROOT/args.output).resolve(),args.smoke)


if __name__=="__main__":main()
