"""Rebuild the Stage-15C descriptive report from frozen raw trajectories.

Reference-stability formulas are reused from stage15b_reference; solver,
schedule, stopping, offline acceptance, counters and manifest checks below are
recomputed.  This is not a second FEM implementation or a continuous certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from stage15b_reference import assess_reference


ROOT = Path(__file__).resolve().parent
LIMITATIONS = ["Finite aligned-geometry development cohort, not an AI speed claim."]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _inside(directory, name):
    path = (directory / name).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError("Manifest path escapes input directory: " + str(name))
    return path


def read_checked(directory, check_sources=True):
    directory = Path(directory)
    complete = json.loads((directory / "complete.json").read_text())
    if complete.get("complete") is not True:
        raise ValueError("Incomplete experiment")
    manifest = complete["file_sha256"]
    for needed in ("freeze.json", "references.json", "index.json"):
        if needed not in manifest:
            raise ValueError("Required raw file missing from manifest: " + needed)
    for name, digest in manifest.items():
        if sha(_inside(directory, name)) != digest:
            raise ValueError("Raw file hash mismatch: " + name)
    freeze = json.loads((directory / "freeze.json").read_text())
    if check_sources:
        for name, digest in freeze["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("Frozen source changed: " + name)
    references = json.loads((directory / "references.json").read_text())
    index = json.loads((directory / "index.json").read_text())
    paths = []
    for entry in index:
        for key in ("path_file", "payload_file"):
            if entry[key] not in manifest:
                raise ValueError("Indexed artifact missing from manifest: " + entry[key])
        with gzip.open(_inside(directory, entry["path_file"]), "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        with np.load(_inside(directory, entry["payload_file"]), allow_pickle=False) as arrays:
            payload = {name: arrays[name] for name in arrays.files}
        paths.append((entry, record, payload))
    return freeze, references, paths, complete


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed):
        self.checks.append(dict(name=name, passed=bool(passed)))

    def close(self):
        failed = [x["name"] for x in self.checks if not x["passed"]]
        if failed:
            raise ValueError("Integrity/recomputation failed: " + "; ".join(failed[:12]))


def _near(a, b, rtol=1e-10, atol=1e-12):
    if a is None or b is None:
        return a is b
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= atol+rtol*max(abs(a),abs(b)))


def _solver_accepted(metrics):
    per_rhs = metrics.get("per_rhs", [])
    finite = bool(per_rhs) and all(np.isfinite(v.get("relative_residual", np.inf)) and
                                   np.isfinite(v.get("componentwise_backward_error", np.inf)) for v in per_rhs)
    numerical = finite and all(v["relative_residual"] <= 1e-9 and
                               v["componentwise_backward_error"] <= 1e-12 for v in per_rhs)
    solver_ok = all(info == 0 for column in metrics.get("solver_info", [[1]]) for info in column)
    return bool(numerical and solver_ok and metrics.get("boundary_maximum", np.inf) <= 1e-13)


def offline_assessment(summary, reference, tau):
    """Independent implementation of output acceptance for report checks."""
    qualified = bool(reference["assessment"]["accepted"])
    completed = summary["status"] in ("estimated_converged", "fixed_mesh_completed")
    qoi = summary.get("final_qoi")
    error = None if qoi is None or not np.isfinite(qoi) else float(abs(qoi-reference["assessment"]["qoi"]))
    ratio = None if error is None else error/tau
    success = bool(qualified and completed and ratio is not None and ratio <= 1.)
    if not qualified:
        status = "reference_unqualified"
    elif not completed:
        status = summary["status"]
    elif not success:
        status = "accuracy_failed"
    else:
        status = "success"
    return dict(reference_qualified=qualified, online_completed=completed,
                absolute_qoi_error=error, q=ratio, success=success, status=status,
                acceptance_is_relative_to_empirically_checked_reference=True,
                continuous_error_bound=False)


def _marker(method):
    return "uniform" if method.startswith("uniform_") else method


def _expected_reasons(step, qoi, previous, tau, marker, cfg):
    result = []
    if marker == "dwr":
        result.append("dwr_marker")
    if step == 0:
        result.append("initial")
    elif step % cfg["verification_interval"] == 0:
        result.append("periodic")
    if previous is not None and abs(qoi-previous) <= cfg["verification_delta_factor"]*tau:
        result.append("qoi_delta")
    return result


def final_mesh_check(payload, metadata):
    """Recompute domain/material checks from saved coordinates and triangles."""
    from skfem import MeshTri
    p, t = payload["p"], payload["t"]
    vertices = p[:, t]
    first, second = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    areas = abs(first[0]*second[1]-first[1]*second[0])/2.
    crossings = sum(int(np.count_nonzero((vertices[axis].min(axis=0)<value-1e-13) &
                                         (vertices[axis].max(axis=0)>value+1e-13)))
                    for axis,key in enumerate(("interface_x","interface_y")) for value in metadata[key])
    mesh = MeshTri(p,t,sort_t=bool(payload.get("sort_t", True)))
    boundary = p[:,mesh.boundary_nodes()]
    square = bool(np.all(np.any((abs(boundary)<1e-13)|(abs(boundary-1.)<1e-13),axis=0)))
    return dict(accepted=bool(np.all(areas>0.) and abs(areas.sum()-1.)<1e-12 and crossings==0 and square),
                crossing_elements=crossings,area=float(areas.sum()),square_boundary=square)


def audit_path(entry, record, payload, reference, audit, metadata):
    label = f"{entry['case']}/{entry['tau']}/{entry['method']}/{entry['repeat']}"
    for key in ("case", "tau", "method", "repeat"):
        audit.check(label+"/identity/"+key, entry[key] == record[key])
    summary, rows = record["summary"], record["rows"]
    audit.check(label+"/index_summary", summary == entry["summary"])
    audit.check(label+"/nonempty_rows", bool(rows))
    if not rows:
        return None
    fixed = entry["method"].startswith("fixed_")
    cfg = summary["config"]
    calls = [s for row in rows for s in row.get("solves", [])]
    audit.check(label+"/solver_calls", summary["solver_calls"] == len(calls))
    audit.check(label+"/rhs_count", summary["rhs_count"] == sum(s["metrics"]["rhs_count"] for s in calls))
    audit.check(label+"/recorded_states", summary["solved_states"] == len(rows))
    audit.check(label+"/peak_p1", summary["peak_p1_free_dofs"] == max(r.get("p1_free_dofs", 0) for r in rows))
    audit.check(label+"/peak_p2", summary["peak_p2_free_dofs"] == max(r.get("p2_free_dofs", 0) for r in rows))
    audit.check(label+"/final_qoi", _near(summary["final_qoi"], rows[-1].get("qoi")))
    audit.check(label+"/status", summary["status"] == rows[-1]["status"])
    audit.check(label+"/time_identity", _near(summary["total_seconds"], summary["algorithm_seconds"]+summary["artifact_io_seconds"]))
    initial_seconds = summary.get("initial_mesh_seconds", 0.)
    kernel_seconds = summary.get("kernel_algorithm_seconds", summary["algorithm_seconds"]-initial_seconds)
    audit.check(label+"/initial_mesh_time", initial_seconds >= 0. and _near(summary["algorithm_seconds"], initial_seconds+kernel_seconds))
    step_seconds = sum(r["timings"]["step_total_seconds"] for r in rows)
    audit.check(label+"/step_time_containment", kernel_seconds+1e-9 >= step_seconds)
    for i, row in enumerate(rows):
        prefix = label+f"/step{i}"
        audit.check(prefix+"/index", row["step"] == i)
        accepted = []
        for j, call in enumerate(row.get("solves", [])):
            metrics = call["metrics"]
            ok = _solver_accepted(metrics)
            accepted.append(ok)
            audit.check(prefix+f"/solve{j}/acceptance", bool(metrics["accepted"]) == bool(all(
                v["relative_residual"] <= 1e-9 and v["componentwise_backward_error"] <= 1e-12
                for v in metrics["per_rhs"])))
            audit.check(prefix+f"/solve{j}/converged", bool(metrics["converged"]) == ok)
            audit.check(prefix+f"/solve{j}/rhs_count", metrics["rhs_count"] == len(metrics["per_rhs"]))
            if "requested_columns" in metrics:
                expected_columns = [1] if call["degree"] == 2 else ([0, 1] if entry["method"] in ("joint_residual", "recovered_dwr3", "recovered_dwr8") else [0])
                audit.check(prefix+f"/solve{j}/requested_columns", metrics["requested_columns"] == expected_columns)
            audit.check(prefix+f"/solve{j}/method", metrics["method"] == ("amg" if entry["method"].endswith("_amg") else "direct"))
        if any(not value for value in accepted):
            audit.check(prefix+"/failed_solve_preserved", row["status"] == "linear_solver_failed")
        if fixed:
            audit.check(prefix+"/fixed_no_goal_verify", not row["verified"] and all(c["degree"] == 1 for c in row.get("solves", [])))
        else:
            recovery=row.get('recovery')
            recovering=entry['method'] in ('recovered_dwr3','recovered_dwr8')
            if row['status']!='exception':
                usage=(['verification'] if any(c['degree']==2 for c in row['solves']) else [])
                if recovery is not None: usage.append('recovery')
                audit.check(prefix+'/p2_shared_usage',row.get('p2_assembly_usage',[])==usage)
            if recovery is not None:
                expected_steps=3 if entry['method']=='recovered_dwr3' else 8
                audit.check(prefix+'/recovery_fixed_work',recovering and recovery['steps']==expected_steps
                    and recovery['matvec_count']==expected_steps and recovery['omega']==2./3.)
                audit.check(prefix+'/recovery_not_a_solve',recovery['unverified_marker'] is True
                    and recovery['converged_solve_claim'] is False
                    and recovery['no_reference_or_true_p2_adjoint_used'] is True)
                audit.check(prefix+'/recovery_boundary',recovery['boundary_maximum']<=1e-13)
                if row.get('scores') is not None:
                    audit.check(prefix+'/recovery_scores',np.array_equal(np.asarray(row['scores']),abs(np.asarray(row['recovered_local']))))
                rt=recovery['timings']
                audit.check(prefix+'/recovery_time_containment',row['timings']['recovery_seconds']+1e-9>=rt['total_seconds']
                    and rt['total_seconds']+1e-9>=sum(v for k,v in rt.items() if k!='total_seconds'))
            if row['status']=='estimated_converged':
                audit.check(prefix+'/no_unused_localization',row.get('scores') is None and recovery is None
                    and (row.get('dwr') is None or row['dwr']['local'] is None))
            if "stop_candidate" in row:
                expected = _expected_reasons(i, row["qoi"], None if i == 0 else rows[i-1]["qoi"], entry["tau"], _marker(entry["method"]), cfg)
                audit.check(prefix+"/verification_schedule", row["verification_reasons"] == expected)
            verified = row["verified"]
            if verified:
                audit.check(prefix+"/verified_has_paid_dual", any(c["degree"] == 2 and _solver_accepted(c["metrics"]) for c in row["solves"]))
                audit.check(prefix+"/verified_estimate", row["dwr"] is not None and np.isfinite(row["dwr"]["total_estimate"]))
                audit.check(prefix+"/algebraic_decomposition", _near(row["dwr"]["total_estimate"], row["dwr"]["discretization_estimate"]+row["dwr"]["algebraic_term"], atol=1e-11))
            if "stop_candidate" in row:
                candidate = bool(verified and abs(row["dwr"]["total_estimate"]) <= cfg["safety"]*entry["tau"])
                audit.check(prefix+"/stop_candidate", row["stop_candidate"] == candidate)
            if row["status"] == "estimated_converged":
                audit.check(prefix+"/paid_success", verified and bool(accepted) and all(accepted) and abs(row["dwr"]["total_estimate"]) <= cfg["safety"]*entry["tau"])
            if row.get("scores") is not None:
                scores = np.asarray(row["scores"], dtype=float)
                chosen = np.asarray(row["selected_elements"], dtype=int)
                audit.check(prefix+"/scores_hash", array_hash(scores) == row["scores_sha256"])
                audit.check(prefix+"/selected_hash", array_hash(chosen) == row["selected_sha256"])
                if _marker(entry["method"]) == "uniform":
                    expected_chosen = np.arange(row["elements"])
                elif scores.sum() <= 0.:
                    expected_chosen = np.array([], dtype=int)
                else:
                    order = np.lexsort((np.arange(len(scores)), -scores))
                    count = min(1+np.searchsorted(np.cumsum(scores[order]), cfg["theta"]*scores.sum()), len(scores))
                    expected_chosen = order[:count]
                audit.check(prefix+"/marking", np.array_equal(chosen, expected_chosen))
        times = row["timings"]
        audit.check(prefix+"/finite_times", all(np.isfinite(t) and t >= 0. for t in times.values()))
        parts = sum(value for key,value in times.items() if key not in ("step_total_seconds", "unattributed_seconds"))
        audit.check(prefix+"/time_parts", times["step_total_seconds"]+1e-9 >= parts)
    if all(key in payload for key in ("p", "t", "u")) and payload["u"].size:
        last_solved = next((r for r in reversed(rows) if r.get("solution_sha256") is not None), None)
        if last_solved is not None:
            audit.check(label+"/final_p_hash", array_hash(payload["p"]) == last_solved["mesh_coordinates_sha256"])
            audit.check(label+"/final_t_hash", array_hash(payload["t"]) == last_solved["mesh_connectivity_sha256"])
            audit.check(label+"/final_u_hash", array_hash(payload["u"]) == last_solved["solution_sha256"])
    mesh_check = final_mesh_check(payload, metadata)
    audit.check(label+"/final_mesh_recomputation", mesh_check == entry["final_mesh_check"])
    offline = offline_assessment(summary, reference, entry["tau"])
    if not mesh_check["accepted"]:
        offline.update(success=False,status="mesh_failed")
    for key, value in offline.items():
        saved = entry["offline"].get(key)
        audit.check(label+"/offline/"+key, _near(value, saved) if key in ("q", "absolute_qoi_error") else value == saved)
    breakdown = Counter()
    for row in rows:
        for key, value in row["timings"].items():
            if key not in ("step_total_seconds", "unattributed_seconds"):
                breakdown[key] += value
    # Shared P2 assembly belongs to verification once; recovery-only assembly
    # must not be mislabelled as an exact adjoint verification charge.
    dwr_cost=sum(sum(r['timings'].get(k,0.) for k in (
        'p2_assembly_seconds','p2_dual_solve_seconds','dwr_evaluation_seconds'))
        for r in rows if 'verification' in r.get('p2_assembly_usage',[]))
    recovery_cost=sum(r['timings'].get('recovery_seconds',0.)+r['timings'].get('recovery_localization_seconds',0.)
        +(r['timings'].get('p2_assembly_seconds',0.) if r.get('p2_assembly_usage')==['recovery'] else 0.) for r in rows)
    nonterminal = sum(sum(r["timings"].get(k,0.) for k in (
        "p2_assembly_seconds","p2_dual_solve_seconds","dwr_evaluation_seconds","dwr_localization_seconds"))
        for r in rows if r["status"]=="continue") if entry["method"]=="dwr" else 0.
    return dict(case=entry["case"], tau=entry["tau"], method=entry["method"], repeat=entry["repeat"],
                final_p1_dofs=rows[-1].get("p1_free_dofs",0), states=len(rows),
                nonterminal_seconds=nonterminal, nonterminal_fraction=nonterminal/summary["algorithm_seconds"],
                hypothetical_remaining_seconds=summary["algorithm_seconds"]-nonterminal,
                status=offline["status"], online_status=summary["status"], success=offline["success"],
                reference_qualified=offline["reference_qualified"], q=offline["q"],
                algorithm_seconds=summary["algorithm_seconds"], total_seconds=summary["total_seconds"],
                initial_mesh_seconds=initial_seconds, solver_calls=summary["solver_calls"], rhs_count=summary["rhs_count"],
                dwr_paid_seconds=dwr_cost, dwr_paid_fraction=dwr_cost/summary["algorithm_seconds"],
                paid_cost_definition='exact P2 verification only; shared assembly charged here once; excludes localization',
                recovery_extra_seconds=recovery_cost,dwr_localization_seconds=breakdown['dwr_localization_seconds'],
                timings=dict(breakdown))


def summarize(freeze, references, paths, complete):
    audit = Audit()
    cfg = freeze["config"]
    cases = [c["name"] for c in freeze["cases"]]
    methods = cfg["methods"]
    taus = cfg["taus"]
    repeats = cfg["repeats"]
    expected = {(case,tau,method,repeat) for case in cases for tau in taus for method in methods for repeat in range(repeats)}
    keys = [(e["case"],e["tau"],e["method"],e["repeat"]) for e,_,_ in paths]
    audit.check("complete_path_matrix", len(keys) == len(expected) and set(keys) == expected)
    audit.check("complete_path_count", complete["paths"] == len(paths))
    refs = {reference["case"]: reference for reference in references}
    audit.check("reference_case_matrix", len(refs) == len(references) == len(cases) and set(refs) == set(cases))
    reference_summary = []
    for case in cases:
        ref = refs[case]
        audit.check(case+"/reference_frozen_config", ref["config"] == cfg["reference"])
        recomputed = assess_reference(ref["rows"], ref["config"])
        audit.check(case+"/reference_recomputation", recomputed == ref["assessment"])
        for i,row in enumerate(ref["rows"]):
            audit.check(case+f"/reference_solver{i}", row["accepted"] == bool(row["mesh_check"]["accepted"] and _solver_accepted(row["solver"])))
        reference_summary.append(dict(case=case, **recomputed, total_seconds=ref["total_seconds"]))
    metadata = {case["name"]:case["metadata"] for case in freeze["cases"]}
    records = [audit_path(entry,record,payload,refs[entry["case"]],audit,metadata[entry["case"]]) for entry,record,payload in paths]
    audit.close()
    per_method = []
    for method in methods:
        subset = [row for row in records if row["method"] == method]
        statuses = dict(Counter(row["status"] for row in subset))
        per_method.append(dict(method=method, paths=len(subset), success=sum(row["success"] for row in subset),
                               failure=sum(not row["success"] for row in subset), statuses=statuses,
                               median_path_seconds=median(row["algorithm_seconds"] for row in subset),
                               total_algorithm_seconds=sum(row["algorithm_seconds"] for row in subset),
                               median_dwr_paid_fraction=median(row["dwr_paid_fraction"] for row in subset)))
    pairs = []
    for case in cases:
        for tau in taus:
            group = []
            for method in methods:
                subset = [r for r in records if (r["case"],r["tau"],r["method"]) == (case,tau,method)]
                qs = [r["q"] for r in subset if r["q"] is not None]
                group.append(dict(method=method, repeats=len(subset), success=sum(r["success"] for r in subset),
                                  all_success=all(r["success"] for r in subset),
                                  median_seconds=median(r["algorithm_seconds"] for r in subset),
                                  median_final_p1_dofs=median(r["final_p1_dofs"] for r in subset),
                                  median_states=median(r["states"] for r in subset),
                                  median_nonterminal_seconds=median(r["nonterminal_seconds"] for r in subset),
                                  median_nonterminal_fraction=median(r["nonterminal_fraction"] for r in subset),
                                  median_hypothetical_remaining_seconds=median(r["hypothetical_remaining_seconds"] for r in subset),
                                  min_seconds=min(r["algorithm_seconds"] for r in subset),
                                  max_seconds=max(r["algorithm_seconds"] for r in subset),
                                  median_q=median(qs) if qs else None, maximum_q=max(qs) if qs else None,
                                  median_dwr_paid_fraction=median(r["dwr_paid_fraction"] for r in subset),
                                  statuses=dict(Counter(r["status"] for r in subset))))
            eligible = [row for row in group if row["all_success"]]
            fastest = min(eligible,key=lambda row:row["median_seconds"]) if eligible else None
            for row in group:
                row["time_over_fastest_successful"] = row["median_seconds"]/fastest["median_seconds"] if row["all_success"] and fastest else None
            pairs.append(dict(case=case,tau=tau, reference_qualified=refs[case]["assessment"]["accepted"],
                              methods=group, fastest_successful=None if fastest is None else fastest["method"]))
    summary = dict(integrity_passed=True, integrity_check_count=len(audit.checks), paths=len(records),
                   successful_paths=sum(r["success"] for r in records), failed_paths=sum(not r["success"] for r in records),
                   cases=cases, taus=taus, methods=methods, repeats=repeats, reference_summary=reference_summary,
                   per_method=per_method, pairs=pairs, records=records,
                   total_algorithm_seconds=sum(r["algorithm_seconds"] for r in records),
                   experiment_wall_seconds=complete["wall_seconds"],
                   reference_formula_reused=True, no_independent_second_fem_solver=True,
                   limitations=LIMITATIONS, gates={"G1":"NOT_APPROVED","G2":"NOT_EVALUATED","G3":"NOT_EVALUATED"})
    summary["screening"]=screen_opportunity(freeze,pairs)
    summary["limitations"]=[
        "24个几何来自有限对齐池的分层无放回抽样；容限与计时重复不是独立几何。",
        "参考稳定性和DWR安全系数不构成连续误差证书。",
        "事后最快对照不是部署选择器；中间层识别和免费估计仅是固定轨迹算术情景。",
        "未训练AI；固定Jacobi恢复不是高对比可靠估计，也未证明学习必要性或新颖性。",
        "本轮不覆盖一般未对齐材料、多目标循环和通量认证。"]
    return summary, audit.checks


def screen_opportunity(freeze,pairs):
    """Frozen necessary screening only; never declares deployed AI benefit."""
    cfg=freeze['config']['screening']; settings=[]
    for pair in pairs:
        dwr=next(r for r in pair['methods'] if r['method']=='dwr')
        controls=[r for r in pair['methods'] if r['method']!='dwr' and r['all_success']]
        fastest=min(controls,key=lambda r:r['median_seconds']) if controls else None
        numerical=bool(pair['reference_qualified'] and all(r['all_success'] for r in pair['methods']))
        opportunity=information=False; time_ratio=dof_ratio=ideal_ratio=None
        if fastest and dwr['all_success'] and pair['reference_qualified']:
            time_ratio=dwr['median_seconds']/fastest['median_seconds']
            dof_ratio=dwr['median_final_p1_dofs']/fastest['median_final_p1_dofs']
            ideal_ratio=dwr['median_hypothetical_remaining_seconds']/fastest['median_seconds']
            opportunity=bool(dwr['median_nonterminal_fraction']>=cfg['min_nonterminal_fraction']
                             and ideal_ratio<=cfg['max_hypothetical_vs_control'])
            information=bool(time_ratio<=cfg['max_dwr_time_vs_control']
                             or dof_ratio<=cfg['max_dwr_dofs_vs_fastest_control'])
        settings.append(dict(case=pair['case'],tau=pair['tau'],numerical=numerical,
            cost_necessary=opportunity,goal_information=information,passed=numerical and opportunity and information,
            fastest_control=None if fastest is None else fastest['method'],dwr_time_over_control=time_ratio,
            dwr_dofs_over_control=dof_ratio,hypothetical_remaining_over_control=ideal_ratio,
            dwr_nonterminal_fraction=dwr['median_nonterminal_fraction'],
            dwr_nonterminal_ms=1000*dwr['median_nonterminal_seconds']))
    geometries=[]
    for c in freeze['cases']:
        rows=[r for r in settings if r['case']==c['name']]
        geometries.append(dict(case=c['name'],family=c['metadata']['family'],contrast=c['metadata']['contrast'],
            passed=all(r['passed'] for r in rows),passed_settings=sum(r['passed'] for r in rows),settings=len(rows)))
    families=[]
    for family in dict.fromkeys(c['family'] for c in geometries):
        group=[c for c in geometries if c['family']==family]
        n=sum(c['passed'] for c in group)
        families.append(dict(family=family,geometries=len(group),passed=n,
                             retain_for_limited_design=n>=cfg['min_geometries_per_family']))
    retained=[f['family'] for f in families if f['retain_for_limited_design']]
    return dict(thresholds=cfg,settings=settings,geometries=geometries,families=families,
        retained_families=retained,decision='LIMITED_DESIGN_ONLY' if retained else 'STOP_CURRENT_ESTIMATOR_REPLACEMENT_BRANCH',
        G1='NOT_FULLY_APPROVED',G3='NOT_EVALUATED',no_ai_model=True,
        retrospective_nonterminal_labels=True,hypothetical_cost_not_measured_speedup=True)


def make_plots(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    families=[f['family'] for f in summary['screening']['families']]
    for family in families:
        pairs=[p for p in summary['pairs'] if p['case'].startswith(family+'_')]
        values=np.array([[r['median_seconds'] for r in p['methods']] for p in pairs])*1000
        fig,ax=plt.subplots(figsize=(12,7.3))
        chart=ax.imshow(values,cmap='viridis',norm=LogNorm(vmin=values.min(),vmax=values.max()),aspect='auto')
        for i,p in enumerate(pairs):
            for j,r in enumerate(p['methods']):
                txt=f'{values[i,j]:.2f}'
                if not r['all_success']:
                    txt+=f"\nFAIL {3-r['success']}/3"
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='#f94444',lw=2))
                ax.text(j,i,txt,ha='center',va='center',fontsize=8,color='white',
                        bbox=dict(facecolor='black',alpha=.3,edgecolor='none',pad=1.1))
        ax.set_xticks(range(len(summary['methods'])),summary['methods'],rotation=30,ha='right')
        ax.set_yticks(range(len(pairs)),[f"{p['case']} / {p['tau']:g}" for p in pairs])
        ax.set_title(f'{family}: full online time, median of 3 repeats (ms)')
        fig.colorbar(chart,ax=ax,label='milliseconds')
        fig.text(.02,.018,'Red borders preserve failures. Six geometries per family; two tolerances are repeated measurements.',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1))
        fig.savefig(output/f'time_{family}.png',dpi=160)
        fig.savefig(output/f'time_{family}.pdf')
        plt.close(fig)


def markdown_report(s):
    lines=['# 第十五阶段C：新几何经典确认','',
        f"24个新几何、两个容限、九方法、三次重复，共{s['paths']}条路径；成功{s['successful_paths']}，失败{s['failed_paths']}。数据契约检查{s['integrity_check_count']}项通过。",'',
        '这是同批次的经典方法比较，没有训练AI。参考精度为经验稳定性，失败样本始终保留。','',
        '## 预定投入筛查','',f"决定：`{s['screening']['decision']}`。",'',
        '| 材料族 | 两容限均通过的几何 | 保留作有限设计 |','|---|---:|---|']
    for f in s['screening']['families']:
        lines.append(f"| {f['family']} | {f['passed']}/{f['geometries']} | {'是' if f['retain_for_limited_design'] else '否'} |")
    lines+=['','门槛来自运行前协议：每族至少4/6几何、每几何两个容限均通过数值、成本必要条件和目标信息条件。事后最快对照是严格筛查参照，不是在线选择策略。该筛查即使通过也不是G1完整批准或AI净收益。','',
        '| 方法 | 成功/总数 | 失败原因 |','|---|---:|---|']
    for m in s['per_method']:
        failures={k:v for k,v in m['statuses'].items() if k!='success'}
        lines.append(f"| {m['method']} | {m['success']}/{m['paths']} | {failures or '无'} |")
    lines+=['','## 逐设置筛查','',
        '中间费用免费是假设轨迹不变的算术情景，保留最终核验；实际算法不能免费预知终止层。时间/自由度比只与同一设置中最快的成功非DWR对照比较。全部方法成功才允许批准设置。','',
        '| 几何 | τ | 最快对照 | DWR时间比 | DWR自由度比 | 中间费用占比 | 假设剩余时间比 | 数值/成本/信息 | 通过 |',
        '|---|---:|---|---:|---:|---:|---:|---|---|']
    fmt=lambda x:'—' if x is None else f'{x:.3f}'
    for r in s['screening']['settings']:
        lines.append(f"| {r['case']} | {r['tau']:g} | {r['fastest_control']} | {fmt(r['dwr_time_over_control'])} | {fmt(r['dwr_dofs_over_control'])} | {r['dwr_nonterminal_fraction']:.1%} | {fmt(r['hypothetical_remaining_over_control'])} | {r['numerical']}/{r['cost_necessary']}/{r['goal_information']} | {r['passed']} |")
    lines+=['','## 参考稳定性','',
        '| 几何 | 准入 | 最细级差/τmin | 求积差/τmin | P1交叉差/τmin | 失败项 |',
        '|---|---|---:|---:|---:|---|']
    for r in s['reference_summary']:
        lines.append(f"| {r['case']} | {r['accepted']} | {r['finest_difference_over_tau']:.3g} | {r['quadrature_difference_over_tau']:.3g} | {r['independent_order_difference_over_tau']:.3g} | {','.join(k for k,v in r['checks'].items() if not v) or '无'} |")
    lines+=['','## 完整时间图','']
    for f in s['screening']['families']:
        lines += [f"![{f['family']}](time_{f['family']}.png)",'']
    lines+=['## 全部配对记录','',
        '| 几何 | τ | 方法 | 成功/3 | 时间中位ms | 范围ms | 最大q | 最终P1自由度中位 |',
        '|---|---:|---|---:|---:|---|---:|---:|']
    for p in s['pairs']:
        for r in p['methods']:
            lines.append(f"| {p['case']} | {p['tau']:g} | {r['method']} | {r['success']}/3 | {r['median_seconds']*1000:.3f} | {r['min_seconds']*1000:.3f}–{r['max_seconds']*1000:.3f} | {fmt(r['maximum_q'])} | {r['median_final_p1_dofs']:g} |")
    lines+=['','## 边界','']+['- '+x for x in s['limitations']]+['']
    return '\n'.join(lines)


def run(inputdir,outputdir):
    inputdir,outputdir=Path(inputdir),Path(outputdir)
    if outputdir.exists():raise FileExistsError('Choose a new report directory')
    summary,checks=summarize(*read_checked(inputdir))
    outputdir.mkdir(parents=True)
    dump(outputdir/'summary.json',summary);dump(outputdir/'integrity_checks.json',checks)
    (outputdir/'REPORT_zh.md').write_text(markdown_report(summary))
    make_plots(summary,outputdir)
    dump(outputdir/'complete.json',dict(complete=True,raw_complete_sha256=sha(inputdir/'complete.json'),
         report_source_sha256=sha(__file__),file_sha256={p.name:sha(p) for p in outputdir.iterdir() if p.is_file()}))
    print(json.dumps(dict(paths=summary['paths'],success=summary['successful_paths'],
          failed=summary['failed_paths'],screening=summary['screening']['decision'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/stage15c_classical')
    parser.add_argument('--output',default='results/stage15c_report');args=parser.parse_args()
    run(ROOT/args.input,ROOT/args.output)
