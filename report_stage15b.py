"""Rebuild the Stage-15B descriptive report from frozen raw trajectories.

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
METHODS = ("uniform_direct", "uniform_amg", "coefficient", "primal_residual",
           "joint_residual", "dwr", "fixed_direct", "fixed_amg")
LIMITATIONS = [
    "六个固定材料不是独立随机样本；三次重复仅描述本机计时波动，不支持统计显著性或泛化结论。",
    "参考资格只表示预定网格/阶次/求积稳定性，不是连续目标误差上界。",
    "DWR停止是带安全系数的经验核验；固定网格基线不使用在线目标误差核验，两者都需退出后的相同精度验收。",
    "缺少带数据振荡项的平衡通量对照、一般未对齐材料几何及多目标原生循环算法。",
    "P2成本并不自动成为可替代成本：廉价方法的最终付费核验仍须保留，学习还会引入推断与回退成本。",
    "G1成本资格、G2可用信息与G3学习确认均未批准；本轮没有训练模型，也不证明AI优势或论文新颖性。",
]


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
                expected_columns = [1] if call["degree"] == 2 else ([0, 1] if entry["method"] == "joint_residual" else [0])
                audit.check(prefix+f"/solve{j}/requested_columns", metrics["requested_columns"] == expected_columns)
            audit.check(prefix+f"/solve{j}/method", metrics["method"] == ("amg" if entry["method"].endswith("_amg") else "direct"))
        if any(not value for value in accepted):
            audit.check(prefix+"/failed_solve_preserved", row["status"] == "linear_solver_failed")
        if fixed:
            audit.check(prefix+"/fixed_no_goal_verify", not row["verified"] and all(c["degree"] == 1 for c in row.get("solves", [])))
        else:
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
    dwr_cost = sum(breakdown[key] for key in ("p2_assembly_seconds", "p2_dual_solve_seconds", "dwr_evaluation_seconds"))
    return dict(case=entry["case"], tau=entry["tau"], method=entry["method"], repeat=entry["repeat"],
                status=offline["status"], online_status=summary["status"], success=offline["success"],
                reference_qualified=offline["reference_qualified"], q=offline["q"],
                algorithm_seconds=summary["algorithm_seconds"], total_seconds=summary["total_seconds"],
                initial_mesh_seconds=initial_seconds, solver_calls=summary["solver_calls"], rhs_count=summary["rhs_count"],
                dwr_paid_seconds=dwr_cost, dwr_paid_fraction=dwr_cost/summary["algorithm_seconds"],
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
    return summary, audit.checks


def make_plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    labels = [f"{p['case']} / tau={p['tau']:g}" for p in summary["pairs"]]
    methods = summary["methods"]
    for key,stem,title,unit in (("median_seconds","complete_time",f"Complete online algorithm time (median of {summary['repeats']} local repeats)","seconds"),
                                 ("median_q","goal_error","Offline goal discrepancy / tau (reference stability only)","q = |J - J_ref| / tau")):
        values = np.array([[row[key] if row[key] is not None else np.nan for row in p["methods"]] for p in summary["pairs"]],dtype=float)
        positive = values[np.isfinite(values) & (values > 0.)]
        lo,hi = (max(float(positive.min()),1e-12),float(positive.max())) if len(positive) else (1e-12,1.)
        if hi <= lo:
            hi = lo*10.
        fig, ax = plt.subplots(figsize=(13,max(4.,2.8+.46*len(labels))))
        mesh = ax.imshow(np.ma.masked_invalid(np.maximum(values,lo)), cmap="viridis", norm=LogNorm(vmin=lo,vmax=hi), aspect="auto")
        for i,pair in enumerate(summary["pairs"]):
            for j,row in enumerate(pair["methods"]):
                value=values[i,j]
                annotation = "NA" if not np.isfinite(value) else f"{value:.3g}"
                if not row["all_success"]:
                    annotation += f"\nFAIL {summary['repeats']-row['success']}/{summary['repeats']}"
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor="#ff5252",lw=1.8))
                ax.text(j,i,annotation,ha="center",va="center",fontsize=8,color="white",
                        bbox=dict(facecolor="black",alpha=.28,edgecolor="none",pad=1.2))
        ax.set_xticks(range(len(methods)),methods,rotation=35,ha="right")
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title(title,fontsize=12)
        fig.colorbar(mesh,ax=ax,label=unit,pad=.02)
        fig.text(.01,.015,f"Red borders retain every failure. {len(summary['cases'])} fixed materials; repeated timings are not independent geometry samples.",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,1))
        fig.savefig(output/(stem+".png"),dpi=170)
        fig.savefig(output/(stem+".pdf"))
        plt.close(fig)


def markdown_report(summary):
    lines = ["# 第十五阶段B：经典基线同精度成本核查", "",
             f"原始记录完整性核查通过：{summary['integrity_check_count']}项。共{summary['paths']}条路径，退出后同精度验收成功{summary['successful_paths']}条，失败{summary['failed_paths']}条。失败记录全部保留。", "",
             "本轮六个确定性材料使用相同固定源和单目标观测。自适应方法采用预先规定的付费核验计划；固定P1网格方法的分辨率提前固定，无在线目标误差核验。所有输出均在退出后按同一目标容限与参考资格检查验收。", "",
             "## 参考稳定性", "",
             "资格重算复用已冻结的参考评估公式；它不是第二套独立有限元实现，也不提供连续误差严格上界。参考不合格时，相应路径全部不计为同精度成功。", "",
             "| 材料 | 参考资格 | 最细差/最小τ | 求积差/最小τ | P1阶次交叉差/最小τ |", "|---|---|---:|---:|---:|"]
    for r in summary["reference_summary"]:
        lines.append(f"| {r['case']} | {'通过' if r['accepted'] else '不通过'} | {r['finest_difference_over_tau']:.3g} | {r['quadrature_difference_over_tau']:.3g} | {r['independent_order_difference_over_tau']:.3g} |")
    lines += ["", "## 成功与失败计数", "", "跨材料的时间中位数仅作描述，不是同精度加速比；比较应查看下方同一材料、同一容限的三次重复中位数。", "",
              "| 方法 | 成功/总数 | 失败 | 所有路径时间中位(s) | P2装配+伴随+估计占比中位 | 退出后状态 |", "|---|---:|---:|---:|---:|---|"]
    for r in summary["per_method"]:
        statuses = "; ".join(f"{key}:{value}" for key,value in r["statuses"].items())
        lines.append(f"| {r['method']} | {r['success']}/{r['paths']} | {r['failure']} | {r['median_path_seconds']:.4g} | {r['median_dwr_paid_fraction']:.1%} | {statuses} |")
    lines += ["", "## 配对结果", "", "只对同一材料、同一容限下三次均成功的方法给出相对最快成功方法的时间比。没有通过参考资格或有任何重复失败的方法不参与该比值。时间包括初始网格、装配、全部实际线性验收、伴随/估计、标记、细化及算法记录准备；输出文件写入与离线参考另计。", "",
              "| 材料 | τ | 方法 | 成功/3 | 时间中位(s) | 时间范围(s) | 最大q | 时间/最快成功 |", "|---|---:|---|---:|---:|---|---:|---:|"]
    for pair in summary["pairs"]:
        for r in pair["methods"]:
            q = "—" if r["maximum_q"] is None else f"{r['maximum_q']:.4g}"
            ratio = "—" if r["time_over_fastest_successful"] is None else f"{r['time_over_fastest_successful']:.3g}"
            lines.append(f"| {pair['case']} | {pair['tau']:g} | {r['method']} | {r['success']}/{r['repeats']} | {r['median_seconds']:.4g} | {r['min_seconds']:.4g}–{r['max_seconds']:.4g} | {q} | {ratio} |")
    lines += ["", "![完整算法时间](complete_time.png)", "", "![退出后目标误差](goal_error.png)", "",
              "## 决策边界", ""] + ["- "+value for value in LIMITATIONS]
    lines += ["", "G1：NOT_APPROVED；G2、G3：NOT_EVALUATED。P2成本占比只能定位待研究成本；不能据此承诺可替代比例或AI加速。", ""]
    return "\n".join(lines)


def run(inputdir, outputdir):
    inputdir, outputdir = Path(inputdir), Path(outputdir)
    if outputdir.exists():
        raise FileExistsError("Use a new report directory; do not overwrite reports")
    frozen = read_checked(inputdir)
    summary, checks = summarize(*frozen)
    outputdir.mkdir(parents=True)
    dump(outputdir/"summary.json",summary)
    dump(outputdir/"integrity_checks.json",checks)
    (outputdir/"REPORT_zh.md").write_text(markdown_report(summary))
    make_plots(summary,outputdir)
    dump(outputdir/"complete.json",dict(complete=True,raw_complete_sha256=sha(inputdir/"complete.json"),
                                       file_sha256={p.name:sha(p) for p in sorted(outputdir.iterdir()) if p.is_file()}))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=ROOT/"results/stage15b_classical")
    parser.add_argument("--output",type=Path,default=ROOT/"results/stage15b_report")
    args=parser.parse_args()
    result=run(args.input,args.output)
    print(json.dumps(dict(paths=result["paths"],successful_paths=result["successful_paths"],
                          failed_paths=result["failed_paths"],integrity_checks=result["integrity_check_count"])))
