#!/usr/bin/env python3
"""Report the completed, preregistered diagnostic run without rerunning PDEs.

Consumes saved JSON only. Refuses incomplete/duplicate measurements. Figures
use English labels for portable fonts; prose is Chinese. No training or model
selection is performed here, and hypothetical costs never replace real costs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
METHODS = ["amg", "dense_spectral_reuse", "sparse_spectral_reuse",
           "sparse_spectral_increment4", "sparse_spectral_increment_all",
           "sparse_unet"]
CLASSICAL = METHODS[:-1]
LABELS = {
    "amg": "AMG", "dense_spectral_reuse": "Dense spectral reuse",
    "sparse_spectral_reuse": "Sparse spectral reuse",
    "sparse_spectral_increment4": "Sparse spectral increment (4)",
    "sparse_spectral_increment_all": "Sparse spectral increment (all)",
    "sparse_unet": "Sparse U-Net (seed 7)",
}
GEOMETRIES = [70001, 70002, 70003]
EVENT_GEOMETRIES = list(range(70001, 70021))
COMPONENTS = {
    "assembly_seconds": "Assembly", "patch_seconds": "Patch construction",
    "reuse_seconds": "Reuse decisions", "spectral_seconds": "Initial spectrum",
    "candidate_seconds": "Candidates", "check_seconds": "All checks",
    "fallback_seconds": "Spectral repair", "setup_seconds": "Preconditioner setup",
    "solve_seconds": "PCG", "direct_fallback_seconds": "Direct recovery",
}
HYPOTHETICAL = ["hypothetical_free_candidate_seconds",
                "hypothetical_free_all_space_work_seconds"]
INPUTS = ["manifest.json", "measurements.json", "event_audit.json",
          "preconditioner_profile.json", "complete.json"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def descriptive(values):
    values = [float(v) for v in values]
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or missing numeric field {name}: {value!r}")
    return float(value)


def validate(out, manifest, records, audit, complete):
    if not complete.get("benchmark_complete") or not complete.get("event_audit_complete"):
        raise ValueError("The benchmark and event audit must both be marked complete")
    config = manifest["configuration"]
    expected_config = {"n": 128, "steps": 20, "core": 16, "overlap": 2,
                       "contrast": 1e4, "gamma": .03, "repeats": 3,
                       "rtol": 1e-8, "model_seed": 7, "torch_threads": 2}
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"Not the preregistered run: {key}={config.get(key)!r}, expected {expected}")
    if sorted(config["validation_seeds"]) != GEOMETRIES:
        raise ValueError("Unexpected timed validation seeds")
    if sorted(config["event_seeds"]) != EVENT_GEOMETRIES:
        raise ValueError("Unexpected event-audit seeds")
    if sorted(config["methods"]) != sorted(METHODS):
        raise ValueError("All six preregistered methods are required")
    expected_keys = {(method, seed, repeat, step) for method in METHODS
                     for seed in GEOMETRIES for repeat in range(3) for step in range(20)}
    observed = [(r["method"], r["sequence_seed"], r["repeat"], r["step"]) for r in records]
    if len(records) != 1080 or len(set(observed)) != 1080 or set(observed) != expected_keys:
        raise ValueError("Expected exactly 1,080 distinct method/geometry/repeat/step records")
    for row in records:
        key = (row["method"], row["sequence_seed"], row["repeat"], row["step"])
        expected_model = 7 if row["method"] == "sparse_unet" else -1
        if row.get("model_seed") != expected_model:
            raise ValueError(f"Unexpected model seed in {key}")
        for field in ["total_seconds", *COMPONENTS, *HYPOTHETICAL,
                      "relative_residual", "final_relative_residual", "energy_relative_error", "rtol"]:
            value = finite_number(row.get(field), f"{key}/{field}")
            if value < 0:
                raise ValueError(f"Negative timing/error in {key}/{field}")
        real = row["total_seconds"]
        if not math.isclose(row[HYPOTHETICAL[0]], real - row["candidate_seconds"], abs_tol=1e-8):
            raise ValueError(f"Free-candidate arithmetic mismatch: {key}")
        removed = sum(row[f] for f in ["spectral_seconds", "candidate_seconds", "check_seconds", "fallback_seconds"])
        if not math.isclose(row[HYPOTHETICAL[1]], real - removed, abs_tol=1e-8):
            raise ValueError(f"Free-all-space-work arithmetic mismatch: {key}")
    counts = audit["sequence_counts"]
    if sorted(c["sequence_seed"] for c in counts) != EVENT_GEOMETRIES:
        raise ValueError("The event audit must include all 20 geometries exactly once")
    events = audit["events"]
    event_keys = [(e["sequence_seed"], e["step"], e["patch"]) for e in events]
    if len(set(event_keys)) != len(event_keys):
        raise ValueError("Duplicate shared-trajectory event")
    actual_counts = Counter(e["sequence_seed"] for e in events)
    required_candidates = {"spectral4", "spectral_all", "unet_7", "unet_17", "unet_27"}
    for c in counts:
        if c["steps"] != 20 or c["trigger_events"] != actual_counts[c["sequence_seed"]]:
            raise ValueError("Incomplete or inconsistent event-audit sequence")
    for e in events:
        if e["sequence_seed"] not in EVENT_GEOMETRIES or not 0 <= e["step"] < 20 or not 0 <= e["patch"] < 64:
            raise ValueError("Unexpected event coordinates")
        if e["old_check"]["passed"]:
            raise ValueError("A recorded trigger did not fail the old-space check")
        if not required_candidates <= set(e["candidates"]):
            raise ValueError("Missing reference or frozen-model checks on a shared event")
    source_checks = {}
    for name, expected in manifest["source_sha256"].items():
        path = ROOT / name
        source_checks[name] = path.exists() and digest(path) == expected
    protocol = ROOT / "docs/diagnostic_protocol_zh.md"
    source_checks["docs/diagnostic_protocol_zh.md"] = digest(protocol) == manifest["protocol_sha256"]
    for name, expected in manifest["checkpoint_sha256"].items():
        path = ROOT / "checkpoints" / name
        source_checks[f"checkpoints/{name}"] = path.exists() and digest(path) == expected
    if not all(source_checks.values()):
        raise ValueError(f"Frozen source/model mismatch: {[k for k, v in source_checks.items() if not v]}")
    return {"expected_rows": 1080, "actual_rows": len(records), "unique_rows": len(set(observed)),
            "timed_independent_geometries": 3, "event_independent_geometries": 20,
            "source_and_checkpoint_hashes": source_checks,
            "input_sha256": {name: digest(out / name) for name in INPUTS}}


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["method"], row["sequence_seed"], row["repeat"])].append(row)
    units = []
    for (method, seed, repeat), rows in sorted(groups.items()):
        unit = {"method": method, "sequence_seed": seed, "repeat": repeat, "steps": len(rows)}
        for field in ["total_seconds", *COMPONENTS, *HYPOTHETICAL]:
            unit[field] = sum(r[field] for r in rows)
        unit["other_seconds"] = unit["total_seconds"] - sum(unit[field] for field in COMPONENTS)
        for count in ["old_checks", "old_check_passes", "candidate_checks", "candidate_passes",
                      "spectral_repairs", "added_directions", "cached_patch_matrices", "new_local_factors"]:
            unit[count] = sum(r.get(count, 0) for r in rows)
        unit["max_iterations"] = max(r["iterations"] for r in rows)
        unit["max_final_relative_residual"] = max(r["final_relative_residual"] for r in rows)
        unit["max_energy_relative_error"] = max(r["energy_relative_error"] for r in rows)
        unit["pcg_failures"] = sum(not r["converged"] for r in rows)
        unit["solver_fallbacks"] = sum(bool(r["solver_fallback"]) for r in rows)
        unit["final_failures"] = sum(not r["final_converged"] for r in rows)
        unit["residual_exceeds_rtol"] = sum(r["final_relative_residual"] > r["rtol"] for r in rows)
        units.append(unit)
    methods = {}
    for method in METHODS:
        selected = [u for u in units if u["method"] == method]
        geometry = {}
        for seed in GEOMETRIES:
            local = [u for u in selected if u["sequence_seed"] == seed]
            geometry[str(seed)] = {field: descriptive(u[field] for u in local)
                for field in ["total_seconds", *HYPOTHETICAL]}
        result = {"geometry": geometry,
            "total_seconds": descriptive(g["total_seconds"]["median"] for g in geometry.values()),
            "hypothetical_free_candidate_seconds": descriptive(g[HYPOTHETICAL[0]]["median"] for g in geometry.values()),
            "hypothetical_free_all_space_work_seconds": descriptive(g[HYPOTHETICAL[1]]["median"] for g in geometry.values()),
            "mean_breakdown_seconds": {field: statistics.mean(u[field] for u in selected)
                                        for field in [*COMPONENTS, "other_seconds"]},
            "max_iterations": max(u["max_iterations"] for u in selected),
            "max_final_relative_residual": max(u["max_final_relative_residual"] for u in selected),
            "max_energy_relative_error": max(u["max_energy_relative_error"] for u in selected)}
        for field in ["pcg_failures", "solver_fallbacks", "final_failures", "residual_exceeds_rtol",
                      "candidate_checks", "candidate_passes", "spectral_repairs", "cached_patch_matrices"]:
            result[field] = sum(u[field] for u in selected)
        checks = result["candidate_checks"]
        result["candidate_pass_rate_including_timing_repeats"] = result["candidate_passes"] / checks if checks else None
        methods[method] = result
    dense_sparse, gate = {}, {}
    for seed in GEOMETRIES:
        key = str(seed)
        timings = {m: methods[m]["geometry"][key]["total_seconds"]["median"] for m in METHODS}
        best = min(CLASSICAL, key=timings.get)
        dense_sparse[key] = {"dense_seconds": timings["dense_spectral_reuse"],
            "sparse_seconds": timings["sparse_spectral_reuse"],
            "dense_over_sparse": timings["dense_spectral_reuse"] / timings["sparse_spectral_reuse"]}
        ai = timings["sparse_unet"]
        accuracy_by_method = {method: all(
            r["final_converged"] and r["final_relative_residual"] <= r["rtol"]
            for r in records if r["method"] == method and r["sequence_seed"] == seed)
            for method in METHODS}
        accuracy = accuracy_by_method["sparse_unet"]
        classical_accuracy = all(accuracy_by_method[method] for method in CLASSICAL)
        comparison_valid = accuracy and classical_accuracy
        gate[key] = {"fastest_classical_method": best, "classical_seconds": timings[best],
            "unet_seconds": ai, "unet_over_classical": ai / timings[best],
            "full_cost_20_percent_gate": bool(ai <= .8 * timings[best] and comparison_valid),
            "unet_accuracy_passed": accuracy,
            "classical_accuracy_passed": classical_accuracy,
            "comparison_valid": comparison_valid,
            "accuracy_by_method": accuracy_by_method,
            "invalid_accuracy_methods": [m for m, valid in accuracy_by_method.items() if not valid],
            "hypothetical_ratios": {m: {field: methods[m]["geometry"][key][field]["median"] / timings[best]
                for field in HYPOTHETICAL} for m in METHODS[3:]}}
    return units, methods, dense_sparse, gate


def summarize_events(audit):
    events = audit["events"]
    candidate_names = ["spectral4", "spectral_all", "unet_7", "unet_17", "unet_27"]
    candidates = {}
    for name in candidate_names:
        passes = sum(bool(e["candidates"][name]["check"]["passed"]) for e in events)
        candidates[name] = {"passes": passes, "unique_event_checks": len(events),
            "pass_rate": passes / len(events) if events else None,
            "by_geometry": {str(seed): {"passes": sum(bool(e["candidates"][name]["check"]["passed"])
                for e in events if e["sequence_seed"] == seed),
                "events": sum(e["sequence_seed"] == seed for e in events)} for seed in EVENT_GEOMETRIES}}
    negative_counts = [e["old_spectrum"]["significant_negative_count"] for e in events]
    return {"elapsed_seconds": audit["elapsed_seconds"], "independent_geometries": len(audit["sequence_counts"]),
        "geometries_with_events": sum(c["trigger_events"] > 0 for c in audit["sequence_counts"]),
        "unique_events": len(events), "sequence_counts": audit["sequence_counts"],
        "candidates": candidates,
        "negative_count_histogram": {str(k): v for k, v in sorted(Counter(negative_counts).items())},
        "negative_count_gt4_events": sum(n > 4 for n in negative_counts),
        "spectral4_failures_with_negative_count_le4": sum(
            e["old_spectrum"]["significant_negative_count"] <= 4 and not e["candidates"]["spectral4"]["check"]["passed"]
            for e in events),
        "spectral_all_failures": sum(not e["candidates"]["spectral_all"]["check"]["passed"] for e in events),
        "old_full_spectrum_disagrees_with_trigger": sum(bool(e["old_spectrum"]["passed"]) for e in events),
        "event_types": dict(Counter(e["event"] for e in events)),
        "interpretation": "Shared traditional-reuse trajectory; the same cases are reused across models, not extra independent samples."}


def summarize_profile(profile):
    return {"relative_application_difference": profile["relative_application_difference"],
        "microseconds_per_application": {name: descriptive(
            r["seconds"] / r["calls"] * 1e6 for r in profile["records"] if r["backend"] == name)
            for name in ["dense", "sparse"]},
        "dense_setup": profile["dense_setup"],
        "sparse_setup_and_profile": profile["sparse_setup_and_profile"],
        "scope": "Separate instrumented microprofile at the first geometry's first step; not included in online sequence timing."}


def equivalent_reuse(records):
    keyed = {(r["method"], r["sequence_seed"], r["repeat"], r["step"]): r for r in records}
    fields = ["coarse_dimension", "reused_patches", "old_checks", "old_check_passes", "spectral_repairs"]
    mismatches = []
    for seed in GEOMETRIES:
        for repeat in range(3):
            for step in range(20):
                a = keyed[("dense_spectral_reuse", seed, repeat, step)]
                b = keyed[("sparse_spectral_reuse", seed, repeat, step)]
                diff = {f: [a[f], b[f]] for f in fields if a[f] != b[f]}
                signatures = lambda r: [(d["kind"], d["patch"], d["passed"], d["dimension"])
                                         for d in r["check_diagnostics"]]
                if signatures(a) != signatures(b):
                    diff["check_decisions"] = [signatures(a), signatures(b)]
                if diff:
                    mismatches.append({"sequence_seed": seed, "repeat": repeat, "step": step, "differences": diff})
    return {"step_pairs": 180, "branch_or_dimension_mismatches": mismatches,
            "claim_limit": "Matching recorded branches/dimensions does not itself prove identical basis projections; use numerical tests and operator comparison as well."}


def create_figures(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    methods = summary["methods"]
    y = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(10.5, 5.1), constrained_layout=True)
    med = np.array([methods[m]["total_seconds"]["median"] for m in METHODS])
    low = np.array([methods[m]["total_seconds"]["min"] for m in METHODS])
    high = np.array([methods[m]["total_seconds"]["max"] for m in METHODS])
    colors = ["#477695", "#839aab", "#477695", "#749a74", "#749a74", "#d28b42"]
    ax.errorbar(med, y, xerr=np.stack([med - low, high - med]), fmt="none", ecolor="black", capsize=3)
    for row, method in enumerate(METHODS):
        ax.scatter(med[row], row, color=colors[row], s=105, zorder=2)
        ax.scatter([methods[method]["geometry"][str(s)]["total_seconds"]["median"] for s in GEOMETRIES],
                   [row] * 3, color="black", s=15, zorder=3)
        ax.annotate(f"{med[row]:.3f} s", (high[row], row), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color="black")
    ax.set_yticks(y, [LABELS[m] for m in METHODS]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(max(min(low) * .65, .001), max(high) * 1.5)
    ax.set_xlabel("Full online seconds per 20-step sequence (log scale)")
    ax.set_title("Actual measured costs; 3 validation geometries, 3 timing repeats")
    ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    for ext in ["png", "pdf"]:
        fig.savefig(figures / f"timing.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    bottom = np.zeros(len(METHODS))
    keys = list(COMPONENTS) + ["other_seconds"]
    palette = plt.get_cmap("tab20")
    for i, field in enumerate(keys):
        values = np.array([methods[m]["mean_breakdown_seconds"][field] for m in METHODS])
        if np.max(np.abs(values)) < 1e-9:
            continue
        ax.bar(y, values, bottom=bottom, label=COMPONENTS.get(field, "Other"), color=palette(i), width=.7)
        bottom += values
    ax.set_xticks(y, ["AMG", "Dense\nreuse", "Sparse\nreuse", "Spectral\n4", "Spectral\nall", "U-Net\nseed 7"])
    ax.set_ylabel("Mean full-sequence online seconds")
    ax.set_title("Measured cost breakdown (equal-weight means preserve additivity)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    for ext in ["png", "pdf"]:
        fig.savefig(figures / f"cost_breakdown.{ext}", dpi=180)
    plt.close(fig)


def fmt(value, digits=3):
    return "不适用" if value is None else f"{value:.{digits}f}"


def report_text(summary):
    methods, gate, event = summary["methods"], summary["gate_by_geometry"], summary["event_audit"]
    lines = ["# 第二轮诊断报告：稀疏实现、真实触发与增量方向容量", "",
        "本报告仅使用保存的实测记录生成。本轮没有重新训练模型，原始先导实验保留。CPU；128×128 单元；每条序列 20 步；对比度 1e4；64 个局部块；BLAS 1 线程、PyTorch 2 线程。", "",
        "完整计时覆盖三个新验证几何 70001–70003、六种方法、三次计时重复，共 54 次完整序列、1,080 次线性求解。离线事件审计覆盖 70001–70020 共二十个验证几何，包含前三个计时几何，不重复计数。它们是开发验证集，不能被当作最终独立测试集。", "",
        "## 完整在线时间", "",
        "先将每次完整序列的 20 步时间相加，再对同一几何的三次计时取中位数，最后汇总三个几何。范围是几何间观测范围，不是置信区间。所有构造、检查、修复、预条件器设置和求解成本均计入；包含初始化。", "",
        "| 方法 | 总时间中位数 / s | 几何范围 / s | 最大 PCG 次数 | 最大真残差 | 最大相对能量误差 |",
        "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        r = methods[method]; d = r["total_seconds"]
        lines.append(f"| {LABELS[method]} | {d['median']:.3f} | {d['min']:.3f}–{d['max']:.3f} | {r['max_iterations']} | {r['max_final_relative_residual']:.3e} | {r['max_energy_relative_error']:.3e} |")
    lines += ["", "![完整在线时间](figures/timing.png)", "", "![实测成本分解](figures/cost_breakdown.png)", "",
        "成本分解使用等权平均，以保证各项可加；因此不必等于主表的中位数。局部分解和粗设置是 `setup_seconds` 的子项，没有重复相加。能量误差相对离线直接解，描述离散线性系统求解误差，并非连续 PDE 离散误差。", "",
        "## 稠密与稀疏实现的配对比较", "",
        "| 几何 | 稠密谱复用 / s | 稀疏谱复用 / s | 稠密时间 / 稀疏时间 |", "|---|---:|---:|---:|"]
    for seed, r in summary["dense_sparse_by_geometry"].items():
        lines.append(f"| {seed} | {r['dense_seconds']:.3f} | {r['sparse_seconds']:.3f} | {r['dense_over_sparse']:.3f} |")
    mismatch = len(summary["reuse_equivalence"]["branch_or_dimension_mismatches"])
    lines += ["", f"检查了 180 对逐步记录，粗维数、复用与修复分支、检查决定存在 {mismatch} 对差异。记录一致性本身不能证明基投影完全相同，需结合数值单元测试及算子作用比较。此处改善应归因于局部稀疏求解与缓存实现，不能归因于神经网络。", "",
        "## 冻结网络对最强传统方法的工程门槛", "",
        "每个几何的传统基线从 AMG、稠密/稀疏谱复用、完整计时的四方向/全部方向谱增量中取最快者。门槛为网络完整在线时间不超过该基线的 80%，且全部比较方法的每次求解都满足相同最终真残差要求。任何方法最终精度不达标，该几何的速度门槛比较无效，不因剔除失败对照而重新选基线。该 20% 是预先约定的工程筛选线，不是发表标准或统计显著性证明。", "",
        "| 几何 | 最快传统方法 | 传统时间 / s | 网络时间 / s | 网络 / 传统 | 该几何是否过线 |", "|---|---|---:|---:|---:|---|"]
    for seed, r in gate.items():
        ratio = f"{r['unet_over_classical']:.3f}" if r["comparison_valid"] else "无效"
        status = ("是" if r["full_cost_20_percent_gate"] else "否") if r["comparison_valid"] else "无效：精度未通过"
        lines.append(f"| {seed} | {LABELS[r['fastest_classical_method']]} | {r['classical_seconds']:.3f} | {r['unet_seconds']:.3f} | {ratio} | {status} |")
    overall = summary["actual_unet_gate_all_geometries"]
    overall_status = ("是" if overall else "否") if all(r["comparison_valid"] for r in gate.values()) else "比较无效：存在最终精度未达标的方法"
    lines += ["", f"三个验证几何全部过线：**{overall_status}**。该判断仅用于本轮资源投入筛选。精度无效时仍保留原始耗时及算术比值供审计，不能将它们解释为有效的加速结论。", "",
        "## 假设免费成本情景：仅用于诊断", "",
        "以下两列均由实际时间做减法得到，保持已发生的粗空间与 PCG 路径不变，未实际实现对应求解器。不能作为真实加速比，也不是所有可能 AI 方法的普适下界。", "",
        "- 候选免费：总时间减去 `candidate_seconds`，包含候选谱计算或网络计算及正交化。",
        "- 全部空间工作免费：再将首步谱构造、全部旧空间/候选检查、谱回退修复设为免费；具体为总时间减去 `spectral_seconds + candidate_seconds + check_seconds + fallback_seconds`。",
        "- 局部矩阵组装、复用逻辑、局部与粗矩阵设置、PCG、直接恢复仍然计入。全部方向参考可能比网络有更多方向，不能把它的迭代表现直接当作四输出网络的性能。", "",
        "| 方法 | 实测完整时间 / s | 假设候选免费 / s | 假设全部空间工作免费 / s |", "|---|---:|---:|---:|"]
    for method in METHODS[3:]:
        r = methods[method]
        lines.append(f"| {LABELS[method]} | {r['total_seconds']['median']:.3f} | {r[HYPOTHETICAL[0]]['median']:.3f} | {r[HYPOTHETICAL[1]]['median']:.3f} |")
    lines += ["", "表中每列分别按完整序列、重复、几何聚合，不能将列中位数之差解释为某个单独阶段的中位耗时。逐几何的完整结果及假设时间/最快传统时间比均保存在 `summary.json`。只有假设情景过线时，只能说存在潜在空间，不能宣称可部署方法已经取得收益。", "",
        "## 正确性、候选与失败", "",
        "下表计数包括全部计时重复，只用于审计实际运行工作量，不作为独立候选样本数。", "",
        "| 方法 | 候选通过/检查 | 谱修复 | PCG 未达标 | 直接恢复 | 最终未收敛 | 最终真残差超限 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        r = methods[method]
        lines.append(f"| {LABELS[method]} | {r['candidate_passes']}/{r['candidate_checks']} | {r['spectral_repairs']} | {r['pcg_failures']} | {r['solver_fallbacks']} | {r['final_failures']} | {r['residual_exceeds_rtol']} |")
    lines += ["", "直接恢复后的准确解不自动表示 PCG 原本成功；两者分列且恢复成本计入在线时间。非有限记录会阻止报告生成，不能以缺失值跳过失败。", "",
        "## 二十条共同经典轨迹上的事件审计", "",
        f"审计用时 **{event['elapsed_seconds']:.3f} 秒**，属于单独的离线诊断成本。{event['independent_geometries']} 个几何中，{event['geometries_with_events']} 个触发了需要补充方向的事件，共 **{event['unique_events']} 个唯一“几何—时刻—局部块”案例**。同一几何内事件可能相关；三种网络检查同一案例，不增加独立样本量。", "",
        "只有廉价复用未通过、且旧空间完整数值检查失败才触发候选。每个事件的所有候选共享同一旧空间，随后仍以经典谱修复推进轨迹；这与完整计时中各方法自身的在线轨迹不同。", "",
        "| 候选 | 通过案例数 | 唯一检查案例数 | 通过率 |", "|---|---:|---:|---:|"]
    for name, r in event["candidates"].items():
        rate = "不适用" if r["pass_rate"] is None else f"{r['pass_rate']:.1%}"
        lines.append(f"| {name} | {r['passes']} | {r['unique_event_checks']} | {rate} |")
    lines += ["", "| 验证几何 | 旧空间检查次数 | 唯一触发事件数 | U-Net 7 通过事件数 |", "|---|---:|---:|---:|"]
    for c in event["sequence_counts"]:
        seed = str(c["sequence_seed"])
        passes = event["candidates"]["unet_7"]["by_geometry"][seed]["passes"]
        lines.append(f"| {seed} | {c['old_checks']} | {c['trigger_events']} | {passes} |")
    lines += ["", f"旧完整检查矩阵存在超过四个显著负特征值的案例数：**{event['negative_count_gt4_events']}**。显著负模式数不超过四个、但四方向谱参考失败的案例数：**{event['spectral4_failures_with_negative_count_le4']}**。全部方向谱参考失败数：**{event['spectral_all_failures']}**。", "",
        "显著负特征值按原对角合同缩放及 -5e-10 阈值计数。在精确算术下，超过四个负模式表示秩至多四的半正定增量不足；本轮是带容差的浮点诊断。反之，负模式数不超过四个并不保证某个四方向候选通过。四方向参考按低特征值模式顺序补充，并非全局最优四维补空间；其失败不能单独证明所有四方向方法都不可能。", "",
        "若全部方向参考仍有失败，应先检查数值秩截断、零模与参考谱计算，不应直接归因于网络。各候选新增维数和原完整检查谱保留在 `event_audit.json`。", "",
        "## 独立预条件器微观计时", ""]
    p = summary["profile"]
    lines += [f"第一计时几何首步的稠密/稀疏算子作用相对差异为 **{p['relative_application_difference']:.3e}**。微观计时单独执行，不计入序列时间；稀疏端含分项计时器开销，仅用于定位瓶颈。", "",
        "| 局部后端 | 每次预条件器作用中位时间 / μs | 重复范围 / μs |", "|---|---:|---:|"]
    for name, r in p["microseconds_per_application"].items():
        lines.append(f"| {name} | {r['median']:.1f} | {r['min']:.1f}–{r['max']:.1f} |")
    sp = p["sparse_setup_and_profile"]
    calls = sp.get("apply_calls", 0)
    if calls:
        lines += ["", f"稀疏剖析累计 {calls} 次作用：局部求解 {sp['apply_local_seconds']:.6f} 秒，粗求解 {sp['apply_coarse_seconds']:.6f} 秒。包括预热及作用差异检查，不能直接与主计时的求解次数相乘当作实测序列总时间。"]
    lines += ["", "## 解释边界与可追溯性", "",
        "本轮仅检验固定问题分布与当前实现。三条计时几何不足以支持普遍加速、统计显著性或一区论文创新已经成立。数值谱检查不等于带舍入误差控制的严格认证，所用粗空间也没有由本轮计时自动获得全局鲁棒收敛定理。", "",
        "协议、算法源码及冻结权重哈希均已核对。报告输入、每次完整序列聚合、每个几何的比较、假设成本字段和失败计数保存在 `summary.json`。原始逐步记录为 `measurements.json`；事件审计、微观剖析和完成标记分别保存在 `event_audit.json`、`preconditioner_profile.json`、`complete.json`。", ""]
    if summary["warnings"]:
        lines += ["本次审计提醒：", ""] + [f"- {w}" for w in summary["warnings"]] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=ROOT / "results/diagnostic_cpu")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = args.run_dir.resolve()
    data = {name: load(out / name) for name in INPUTS}
    manifest, records = data["manifest.json"], data["measurements.json"]
    check = validate(out, manifest, records, data["event_audit.json"], data["complete.json"])
    units, methods, pairs, gate = aggregate(records)
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": True, "report_source_sha256": digest(__file__), "validation": check,
        "aggregation": "Sum all steps per complete sequence; median 3 timing repeats per geometry; report median/range over 3 geometries. Ranges are observations, not confidence intervals.",
        "configuration": manifest["configuration"], "methods": methods, "sequence_units": units,
        "dense_sparse_by_geometry": pairs, "gate_by_geometry": gate,
        "actual_unet_gate_all_geometries": all(g["full_cost_20_percent_gate"] for g in gate.values()),
        "comparison_valid_all_geometries": all(g["comparison_valid"] for g in gate.values()),
        "gate_definition": "Full-cost U-Net <= 0.8 times fastest full-cost classical comparator on every timed geometry, with every solve of all compared methods satisfying final true residual criteria. Any final accuracy failure invalidates that geometry's comparison. Engineering screen only.",
        "event_audit": summarize_events(data["event_audit.json"]),
        "profile": summarize_profile(data["preconditioner_profile.json"]),
        "reuse_equivalence": equivalent_reuse(records), "warnings": []}
    for method, result in methods.items():
        if result["final_failures"] or result["residual_exceeds_rtol"]:
            summary["warnings"].append(
                f"{LABELS[method]} 存在最终精度失败：最终未收敛 {result['final_failures']} 次、真残差超限 {result['residual_exceeds_rtol']} 次；涉及几何的速度门槛比较无效。")
    if summary["reuse_equivalence"]["branch_or_dimension_mismatches"]:
        summary["warnings"].append("稠密/稀疏复用存在记录分支或维数差异，须解释后才能归因于等价实现改进。")
    if any(u["other_seconds"] < -1e-6 for u in units):
        summary["warnings"].append("部分时间分解超过总时间，检查是否存在重叠计时。")
    if summary["event_audit"]["old_full_spectrum_disagrees_with_trigger"]:
        summary["warnings"].append("个别旧空间触发检查与全谱负模式诊断不一致，可能接近数值阈值，需要查看原谱。")
    if summary["event_audit"]["spectral_all_failures"]:
        summary["warnings"].append("存在全部方向参考失败；应核查数值秩阈值与谱检查。")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not args.no_figures:
        create_figures(out, summary)
    (out / "REPORT_zh.md").write_text(report_text(summary))
    print(json.dumps({"output": str(out), "rows": 1080,
        "unique_event_cases": summary["event_audit"]["unique_events"],
        "actual_unet_gate_all_geometries": summary["actual_unet_gate_all_geometries"],
        "warnings": summary["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
