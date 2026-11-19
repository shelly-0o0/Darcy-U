#!/usr/bin/env python3
"""Audit and report stage-4 raw measurements without running any PDE solver.

Online statistics use complete sequences, repeat medians within a geometry,
then summaries across independent geometries. Retrospective current-trajectory
Ritz probes are kept separate from online algorithms and online timings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
RTOL, STEPS = 1e-8, 20
SOURCES = ["fields.py", "numerics.py", "sequence_baselines.py", "sequence_experiment.py",
           "correction_model.py", "balanced_augmentation.py", "traced_pcg.py", "mechanism_experiment.py"]
COMPONENTS = ["assembly_seconds", "guess_seconds", "setup_seconds", "correction_seconds",
              "solve_seconds", "maintenance_seconds", "recovery_seconds"]
QUANTITIES = ["total_seconds", *COMPONENTS, "other_seconds", "inference_seconds", "iterations"]
SCREEN_METHODS = ["baseline", "network_warm", "network_balanced", "history_balanced4",
                  "delta_warm4", "delta_balanced4", "ritz_warm4", "ritz_balanced4", "ritz_balanced8"]
DIAGNOSTIC_METHODS = ["baseline", "network_warm", "network_balanced", "current_ritz4_warm",
                      "current_ritz4_balanced", "current_ritz8_balanced"]
PAIRS = [("network_warm", "network_balanced"), ("delta_warm4", "delta_balanced4"),
         ("ritz_warm4", "ritz_balanced4")]
LABELS = {"baseline": "AMG + history", "network_warm": "Network warm", "network_balanced": "Network balanced",
          "history_balanced4": "History balanced-4", "delta_warm4": "Delta warm-4",
          "delta_balanced4": "Delta balanced-4", "ritz_warm4": "Past Ritz warm-4",
          "ritz_balanced4": "Past Ritz balanced-4", "ritz_balanced8": "Past Ritz balanced-8",
          "current_ritz4_warm": "Current Ritz-4 warm*", "current_ritz4_balanced": "Current Ritz-4 balanced*",
          "current_ritz8_balanced": "Current Ritz-8 balanced*"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = list(values)
    valid = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {"count": len(values), "finite_count": len(valid),
            "median": median(valid) if valid else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None}


def ratio(a, b):
    return float(a / b) if a is not None and b is not None and b > 0 else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_hashes(root, value, where):
    require(set(value["source_sha256"]) == set(SOURCES), f"Unexpected source inventory: {where}")
    results = {}
    for key in ["source_sha256", "checkpoint_sha256"]:
        results[key] = {p: (root / p).is_file() and sha(root / p) == digest for p, digest in value[key].items()}
        require(all(results[key].values()), f"Frozen {key} mismatch: {where}: {results[key]}")
    for key, path in [("protocol_sha256", "docs/stage4_protocol_zh.md"),
                      ("splits_sha256", "data/stage4_splits.json")]:
        results[key] = sha(root / path) == value[key]
        require(results[key], f"Frozen {key} mismatch: {where}")
    return results


def check_problem(manifest, where):
    for field, expected in [("steps", 20), ("rtol", 1e-8), ("contrast", 1e4), ("maxiter", 500),
                            ("blas_threads", 1), ("torch_threads", 2)]:
        require(manifest.get(field) == expected, f"Unexpected {field} in {where}")


def check_splits(root, splits):
    groups = splits["groups"]
    require(len(groups["development"]) == 3 and len(groups["confirmation"]) == 5, "Unexpected cohort size")
    seeds = groups["development"] + groups["confirmation"]
    require(len(set(seeds)) == 8, "Duplicate stage-4 seed")
    hashes = [splits["geometry_sha256"][str(s)] for s in seeds]
    require(len(set(hashes)) == 8, "Duplicate stage-4 geometry hash")
    require(not set(seeds).intersection(splits["excluded_seeds"]), "Excluded seed reused")
    prior = read(root / "data/stage3_splits.json")
    prior_seeds = set(prior["excluded_prior_seeds"] + sum(prior["groups"].values(), []))
    require(prior_seeds.issubset(splits["excluded_seeds"]), "Earlier seed exclusion list incomplete")
    prior_hashes = set(prior.get("geometry_sha256", {}).values())
    require(not set(hashes).intersection(prior_hashes), "Geometry hash overlaps stage 3")
    return {"development_geometries": 3, "confirmation_geometries": 5,
            "confirmation_256_uses_same_first_two_geometries": True,
            "seed_and_available_saved_hash_exclusions_checked": True,
            "note": "Reporter checks saved split metadata; it does not regenerate coefficient fields."}


def load_online(root, name, splits, frozen, frozen_hash):
    directory = root / "results" / name
    filenames = ["manifest.json", "measurements.json", "complete.json", "references.json"]
    data = {f: read(directory / f) for f in filenames}
    manifest, rows, complete = data["manifest.json"], data["measurements.json"], data["complete.json"]
    config = manifest["configuration"]
    is_screen = name == "stage4_screen"
    n = 256 if name == "stage4_confirm256" else 128
    seeds = splits["groups"]["development" if is_screen else "confirmation"]
    if n == 256:
        seeds = seeds[:2]
    repeats = 2 if is_screen else 3
    names = [s["name"] for s in config["specs"]]
    require(config["n"] == n and config["seeds"] == seeds and config["repeats"] == repeats,
            f"Unexpected experiment schedule: {name}")
    require(len(names) == len(set(names)), f"Duplicate method names: {name}")
    if is_screen:
        require(names == SCREEN_METHODS, "Screen must retain all nine prespecified methods")
    else:
        require(config["specs"] == frozen["confirmation_specs"], f"Confirmation configurations differ: {name}")
        require(config.get("selection_sha256") == frozen_hash, f"Selection hash mismatch: {name}")
    expected = {(n, m, s, repeat, step) for m in names for s in seeds
                for repeat in range(repeats) for step in range(STEPS)}
    keys = [(r["n"], r["method"], r["seed"], r["repeat"], r["step"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, f"Missing, duplicate or unexpected rows: {name}")
    require(complete.get("complete") is True and complete.get("rows") == len(rows), f"Completion marker invalid: {name}")
    check_problem(manifest, name)
    hashes = check_hashes(root, manifest, name)
    for key in ["source_sha256", "protocol_sha256", "splits_sha256", "checkpoint_sha256"]:
        require(manifest[key] == frozen[key], f"Artifacts changed between screen/diagnosis/confirmation: {name}/{key}")
    for r in rows:
        for field in ["total_seconds", *COMPONENTS, "inference_seconds", "relative_residual", "final_relative_residual",
                      "energy_relative_error", "base_relative_residual", "corrected_relative_residual", "orthogonality_error"]:
            require(finite(r.get(field)) and r[field] >= 0, f"Invalid/missing {field}: {name}/{r['seed']}/{r['step']}")
        require(r["final_converged"] == (r["final_relative_residual"] <= RTOL), f"Inconsistent final accuracy flag: {name}")
        require(r["converged"] == (r["relative_residual"] <= RTOL), f"Inconsistent PCG accuracy flag: {name}")
        require(r["solver_fallback"] == (not r["converged"]), f"Missing or unexpected recovery record: {name}")
        require(isinstance(r["iterations"], int) and 0 <= r["iterations"] <= 500, f"Iteration budget violated: {name}")
        require(isinstance(r["restarts"], int) and 0 <= r["restarts"] <= 3, f"Restart budget violated: {name}")
        require(not r["record_residuals"], f"Offline trace enabled during online timing: {name}")
        require(sum(r[c] for c in COMPONENTS) <= r["total_seconds"] + 1e-8, f"Component costs exceed total: {name}")
        require(r["inference_seconds"] <= r["correction_seconds"] + 1e-8, f"Inference not nested in correction: {name}")
        require(0 <= r["captured_vectors"] <= 24, f"Online capture cap violated: {name}")
    references = data["references.json"]
    require(len(references) == len(seeds) and {r["seed"] for r in references} == set(seeds), f"Reference inventory invalid: {name}")
    for ref in references:
        require(len(ref["relative_residuals"]) == STEPS and finite(ref["seconds"]) and ref["seconds"] >= 0,
                f"Reference provenance incomplete: {name}")
    return manifest, rows, {"rows_expected": len(expected), "rows_actual": len(rows), "complete": True,
        "duplicate_rows": 0, "frozen_hash_checks": hashes,
        "input_sha256": {f: sha(directory / f) for f in filenames},
        "offline_reference_seconds": sum(r["seconds"] for r in references),
        "offline_reference_relative_residual": distribution(x for r in references for x in r["relative_residuals"]),
        "all_reference_residuals_pass": all(finite(x) and x <= RTOL for r in references for x in r["relative_residuals"])}


def aggregate_online(manifest, rows):
    config = manifest["configuration"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"], row["repeat"])].append(row)
    sequences = []
    for (method, seed, repeat), group in sorted(grouped.items()):
        sequence = {"method": method, "seed": seed, "repeat": repeat}
        for key in ["total_seconds", *COMPONENTS, "inference_seconds", "iterations"]:
            sequence[key] = sum(r[key] for r in group)
        sequence["other_seconds"] = sequence["total_seconds"] - sum(sequence[c] for c in COMPONENTS)
        for key, predicate in [("pcg_failures", lambda r: not r["converged"]),
                               ("recoveries", lambda r: r["solver_fallback"]),
                               ("final_failures", lambda r: not r["final_converged"]),
                               ("network_calls", lambda r: r["network_called"])]:
            sequence[key] = sum(predicate(r) for r in group)
        sequences.append(sequence)
    methods = {}
    for spec in config["specs"]:
        method = spec["name"]
        ss = [s for s in sequences if s["method"] == method]
        rr = [r for r in rows if r["method"] == method]
        geometry = {}
        for seed in config["seeds"]:
            local = [s for s in ss if s["seed"] == seed]
            geometry[str(seed)] = {key: distribution(s[key] for s in local) for key in QUANTITIES}
            geometry[str(seed)].update({key: sum(s[key] for s in local)
                for key in ["pcg_failures", "recoveries", "final_failures", "network_calls"]})
        methods[method] = {"spec": spec, "by_geometry": geometry,
            **{key: distribution(g[key]["median"] for g in geometry.values()) for key in QUANTITIES},
            **{key: sum(s[key] for s in ss) for key in ["pcg_failures", "recoveries", "final_failures", "network_calls"]},
            "rank": distribution(r["rank"] for r in rr), "captured_vectors": distribution(r["captured_vectors"] for r in rr),
            "max_iterations_one_solve": max(r["iterations"] for r in rr),
            "final_relative_residual": distribution(r["final_relative_residual"] for r in rr),
            "pcg_relative_residual_before_recovery": distribution(r["relative_residual"] for r in rr),
            "energy_relative_error": distribution(r["energy_relative_error"] for r in rr),
            "all_final_accurate": all(r["final_converged"] for r in rr),
            "all_pcg_accurate_without_recovery": all(r["converged"] for r in rr)}
    comparisons = {}
    for method, summary in methods.items():
        if method == "baseline":
            continue
        paired = []
        for seed, values in summary["by_geometry"].items():
            base = methods["baseline"]["by_geometry"][seed]
            paired.append({"seed": int(seed), "time_over_baseline": ratio(values["total_seconds"]["median"], base["total_seconds"]["median"]),
                "iterations_over_baseline": ratio(values["iterations"]["median"], base["iterations"]["median"]),
                "iteration_difference": values["iterations"]["median"] - base["iterations"]["median"],
                "accurate_without_recovery": values["pcg_failures"] == values["final_failures"] == 0})
        time_ratios = distribution(p["time_over_baseline"] for p in paired)
        comparisons[method] = {"by_geometry": paired, "time_over_baseline": time_ratios,
            "iterations_over_baseline": distribution(p["iterations_over_baseline"] for p in paired),
            "iteration_difference": distribution(p["iteration_difference"] for p in paired),
            "geometries_faster": sum(p["time_over_baseline"] < 1 for p in paired),
            "geometries_at_least_20_percent_faster": sum(p["time_over_baseline"] <= .8 for p in paired),
            "passes_median_20_percent_engineering_gate": bool(time_ratios["median"] <= .8
                and summary["all_pcg_accurate_without_recovery"] and summary["all_final_accurate"])}
    matched = {}
    for warm, balanced in PAIRS:
        if warm not in methods or balanced not in methods:
            continue
        paired = []
        for seed, values in methods[balanced]["by_geometry"].items():
            base = methods[warm]["by_geometry"][seed]
            paired.append({"seed": int(seed), "balanced_over_warm_time": ratio(values["total_seconds"]["median"], base["total_seconds"]["median"]),
                "balanced_over_warm_iterations": ratio(values["iterations"]["median"], base["iterations"]["median"]),
                "balanced_minus_warm_iterations": values["iterations"]["median"] - base["iterations"]["median"]})
        matched[f"{warm}__{balanced}"] = {"warm": warm, "balanced": balanced, "by_geometry": paired,
            **{key: distribution(p[key] for p in paired) for key in ["balanced_over_warm_time", "balanced_over_warm_iterations", "balanced_minus_warm_iterations"]}}
    maintenance_audit = {}
    for name, result in methods.items():
        if result["spec"]["source"] != "ritz":
            continue
        by_geometry = []
        for seed in config["seeds"]:
            current = [s for s in sequences if s["method"] == name and s["seed"] == seed]
            free = median(s["total_seconds"] - s["maintenance_seconds"] for s in current)
            free_without_correction = median(s["total_seconds"] - s["maintenance_seconds"] - s["correction_seconds"] for s in current)
            base_time = methods["baseline"]["by_geometry"][str(seed)]["total_seconds"]["median"]
            by_geometry.append({"seed": seed,
                "measured_total_seconds": result["by_geometry"][str(seed)]["total_seconds"]["median"],
                "measured_maintenance_seconds": result["by_geometry"][str(seed)]["maintenance_seconds"]["median"],
                "fixed_trajectory_time_excluding_maintenance": free,
                "fixed_trajectory_time_excluding_maintenance_and_correction": free_without_correction,
                "fixed_trajectory_without_maintenance_over_baseline": ratio(free, base_time),
                "fixed_trajectory_without_maintenance_and_correction_over_baseline": ratio(free_without_correction, base_time),
                "maintenance_budget_to_match_baseline_seconds": base_time - free})
        maintenance_audit[name] = {"by_geometry": by_geometry,
            "description": "Retrospective cost subtraction on fixed measured trajectories only. Sum each complete repeat first, subtract its component totals, median repeats within geometry, pair with the baseline median, then summarize geometries. Neither an executable zero-cost algorithm, a speedup prediction, nor a universal lower bound.",
            **{key: distribution(g[key] for g in by_geometry) for key in ["measured_maintenance_seconds",
                "fixed_trajectory_time_excluding_maintenance", "fixed_trajectory_time_excluding_maintenance_and_correction",
                "fixed_trajectory_without_maintenance_over_baseline", "fixed_trajectory_without_maintenance_and_correction_over_baseline",
                "maintenance_budget_to_match_baseline_seconds"]}}
    return {"n": config["n"], "independent_geometries": len(config["seeds"]), "repeats_per_geometry": config["repeats"],
            "methods": methods, "versus_baseline": comparisons, "matched_interfaces": matched,
            "retrospective_maintenance_budget_audit": maintenance_audit,
            "complete_sequences": sequences}


def load_diagnosis(root, splits, frozen):
    directory = root / "results/stage4_diagnosis"
    names = ["manifest.json", "measurements.json", "complete.json"]
    data = {f: read(directory / f) for f in names}
    manifest, rows, complete = data["manifest.json"], data["measurements.json"], data["complete.json"]
    config = manifest["configuration"]
    seeds = splits["groups"]["development"]
    require(config["n"] == 128 and config["seeds"] == seeds and config["steps_observed"] == [0, 6, 10, 15], "Diagnostic schedule changed")
    expected = {(s, step) for s in seeds for step in [0, 6, 10, 15]}
    keys = [(r["seed"], r["step"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Diagnostic cases missing, repeated or unexpected")
    require(complete.get("complete") is True and complete.get("cases") == 12, "Diagnostic completion marker invalid")
    check_problem(manifest, "stage4_diagnosis")
    hashes = check_hashes(root, manifest, "stage4_diagnosis")
    for key in ["source_sha256", "protocol_sha256", "splits_sha256", "checkpoint_sha256"]:
        require(manifest[key] == frozen[key], f"Diagnostic frozen artifact mismatch: {key}")
    cases, variant_rows = [], defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["seed"], r["step"])):
        expected_variants = set(DIAGNOSTIC_METHODS) if r["step"] else set(DIAGNOSTIC_METHODS) - {"network_warm", "network_balanced"}
        require(set(r["variants"]) == expected_variants, "Unexpected diagnostic variants")
        for field in ["initial_error_energy", "reference_relative_residual", "slow4_error_energy_fraction", "retrospective_extraction_seconds"]:
            require(finite(r[field]) and r[field] >= 0, f"Invalid diagnostic {field}")
        case = {key: value for key, value in r.items() if key != "variants"}
        case["variants"] = {}
        base_iters = r["variants"]["baseline"]["iterations"]
        for method, variant in r["variants"].items():
            require(finite(variant["final_relative_residual"]) and variant["final_relative_residual"] >= 0, "Invalid diagnostic residual")
            require(variant["converged"] == (variant["final_relative_residual"] <= RTOL), "Diagnostic convergence flag inconsistent")
            require(0 <= variant["iterations"] <= 500 and len(variant["trace"]) == variant["iterations"], "Diagnostic trace incomplete")
            for index, point in enumerate(variant["trace"], start=1):
                require(point["iteration"] == index and 0 <= point["restart"] <= 3, "Diagnostic trace index invalid")
                require(all(finite(point[f]) for f in ["residual_norm", "relative_residual", "preconditioned_residual_squared"]), "Nonfinite diagnostic trace")
            v = {key: value for key, value in variant.items() if key != "trace"}
            if method == "baseline":
                v.update(error_energy_over_base=1., slow4_energy_over_base=1.)
            v.update(iterations_over_base=ratio(v["iterations"], base_iters), iteration_difference=v["iterations"] - base_iters)
            case["variants"][method] = v
            variant_rows[method].append({"seed": r["seed"], "step": r["step"], **v})
        cases.append(case)
    variants = {}
    for method in DIAGNOSTIC_METHODS:
        vv = variant_rows[method]
        geometry = {str(seed): {key: distribution(v[key] for v in vv if v["seed"] == seed) for key in
            ["iterations", "iterations_over_base", "iteration_difference", "error_energy_over_base", "slow4_energy_over_base"]} for seed in seeds}
        variants[method] = {"case_count": len(vv), "by_geometry": geometry,
            **{key: distribution(v[key] for v in vv) for key in ["iterations", "iterations_over_base", "iteration_difference", "error_energy_over_base", "slow4_energy_over_base"]},
            "geometric_medians": {key: distribution(g[key]["median"] for g in geometry.values()) for key in
                ["iterations", "iterations_over_base", "iteration_difference", "error_energy_over_base", "slow4_energy_over_base"]},
            "pcg_failures": sum(not v["converged"] for v in vv),
            "final_relative_residual": distribution(v["final_relative_residual"] for v in vv)}
    matched = {}
    for warm, balanced in [("network_warm", "network_balanced"), ("current_ritz4_warm", "current_ritz4_balanced")]:
        pairs = [{"seed": c["seed"], "step": c["step"],
            "balanced_over_warm_iterations": ratio(c["variants"][balanced]["iterations"], c["variants"][warm]["iterations"]),
            "balanced_minus_warm_iterations": c["variants"][balanced]["iterations"] - c["variants"][warm]["iterations"],
            "initial_energy_ratio_difference": c["variants"][balanced]["error_energy_over_base"] - c["variants"][warm]["error_energy_over_base"]}
            for c in cases if warm in c["variants"] and balanced in c["variants"]]
        matched[f"{warm}__{balanced}"] = {"pairs": pairs,
            "iteration_ratio": distribution(p["balanced_over_warm_iterations"] for p in pairs),
            "iteration_difference": distribution(p["balanced_minus_warm_iterations"] for p in pairs),
            "geometric_median_iteration_ratio": distribution(median(p["balanced_over_warm_iterations"] for p in pairs if p["seed"] == s) for s in seeds),
            "geometric_median_iteration_difference": distribution(median(p["balanced_minus_warm_iterations"] for p in pairs if p["seed"] == s) for s in seeds),
            "max_abs_initial_energy_ratio_difference": max(abs(p["initial_energy_ratio_difference"]) for p in pairs)}
    geometry = {str(seed): {field: distribution(r[field] for r in rows if r["seed"] == seed and field in r)
        for field in ["initial_error_energy", "slow4_error_energy_fraction", "network_direction_slow4_overlap",
                      "retrospective_extraction_seconds"]} for seed in seeds}
    summary = {"independent_geometries": 3, "paired_cases": 12, "network_cases": 9,
        "by_geometry": geometry,
        "geometric_medians": {field: distribution(g[field]["median"] for g in geometry.values()) for field in
            ["initial_error_energy", "slow4_error_energy_fraction", "network_direction_slow4_overlap", "retrospective_extraction_seconds"]},
        "retrospective_only": True, "variants": variants, "matched_interfaces": matched, "cases": cases,
        "initial_error_energy": distribution(r["initial_error_energy"] for r in rows),
        "slow4_error_energy_fraction": distribution(r["slow4_error_energy_fraction"] for r in rows),
        "network_direction_slow4_overlap": distribution(r["network_direction_slow4_overlap"] for r in rows if "network_direction_slow4_overlap" in r),
        "ritz_projected_residual": distribution(x for r in rows for x in r["ritz"]["projected_ritz_residuals"]),
        "ritz_full_operator_residual": distribution(x for r in rows for x in r["ritz"]["operator_ritz_residuals"]),
        "ritz_selected_rank": distribution(r["ritz"]["rank"] for r in rows),
        "retrospective_extraction_seconds": distribution(r["retrospective_extraction_seconds"] for r in rows),
        "pcg_failures": sum(v["pcg_failures"] for v in variants.values()),
        "reference_relative_residual": distribution(r["reference_relative_residual"] for r in rows)}
    audit = {"cases_expected": 12, "cases_actual": len(rows), "complete": True, "duplicates": 0,
        "frozen_hash_checks": hashes, "input_sha256": {f: sha(directory / f) for f in names},
        "note": "Diagnostic residuals are pre-recovery PCG outcomes; failed probes remain failures. Direct references are offline only."}
    return summary, audit


def f(x, digits=3):
    return "不适用" if x is None else f"{x:.{digits}f}"


def sci(x):
    return "不适用" if x is None else f"{x:.3e}"


def markdown(summary):
    stages, diagnosis = summary["stages"], summary["diagnosis"]
    lines = ["# 第四阶段机制探索结果", "", f"生成时间：{summary['created_utc']}。", "",
        "本阶段检验固定平衡粗空间能否改善 AMG-PCG 的全过程；没有重新训练模型。在线计时包含完整成本。",
        "当前轨迹 Ritz 由同一个系统求解完成后获得，仅用于回顾性潜力诊断，不能作为在线算法或端到端加速证据。", "",
        "## 完整性与统计口径", "",
        f"开发筛选共 1080 次求解，确认共 1680 次求解；确认独立几何为 5 个，256 网格复用其中前 2 个。诊断为 3 个开发几何上的 12 个预定时刻，网络有 9 个配对 case。",
        f"全部在线求解的最终真残差最大值为 {sci(summary['online_accuracy']['max_final_relative_residual'])}；PCG 失败 {summary['online_accuracy']['pcg_failures']} 次，直接恢复 {summary['online_accuracy']['recoveries']} 次，恢复后仍失败 {summary['online_accuracy']['final_failures']} 次。",
        "源码、协议、分组、权重和确认选择文件哈希均核对通过；完整行键与预定日程逐一匹配，没有缺失或重复行。",
        "时间先按 20 步求和，再取同几何重复中位数，最后汇总独立几何。范围是几何间最小值—最大值，并非置信区间。诊断多个时刻共享几何，不能当成 12 个独立样本。",
        "推理及传输已包含在方向构造/校正时间中，不能重复相加。组件分别取中位数通常不会严格加回总时间中位数。失败与恢复全部保留；恢复时间包含在总时间中。", ""]
    for key, title in [("stage4_screen", "开发筛选：128 网格、3 个几何、每项 2 次"),
                       ("stage4_confirm128", "冻结确认：128 网格、5 个几何、每项 3 次"),
                       ("stage4_confirm256", "冻结确认：256 网格、2 个已有确认几何、每项 3 次")]:
        stage = stages[key]
        lines += [f"## {title}", "", "| 方法 | 20 步总秒数：中位数 [范围] | 迭代总数中位数 | 配对时间/基线 | PCG失败/恢复/最终失败 |", "|---|---:|---:|---:|---:|"]
        for name, result in stage["methods"].items():
            timing = result["total_seconds"]
            paired = 1. if name == "baseline" else stage["versus_baseline"][name]["time_over_baseline"]["median"]
            lines.append(f"| {name} | {f(timing['median'])} [{f(timing['min'])}, {f(timing['max'])}] | {f(result['iterations']['median'], 1)} | {f(paired)} | {result['pcg_failures']}/{result['recoveries']}/{result['final_failures']} |")
        lines += ["", "组件为每条完整序列的几何中位秒数：", "",
            "| 方法 | 组装 | 历史初值 | AMG | 方向/校正 | PCG | 回收维护 | 恢复 | 其余 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, result in stage["methods"].items():
            lines.append("| " + name + " | " + " | ".join(f(result[c]["median"], 4) for c in [*COMPONENTS, "other_seconds"]) + " |")
        lines += ["", "同来源 warm / balanced 配对（小于 1 表示 balanced 更小）：", "",
            "| 方向来源 | 配对时间比中位数 | 配对迭代比中位数 | 迭代差中位数 |", "|---|---:|---:|---:|"]
        for pair in stage["matched_interfaces"].values():
            lines.append(f"| {pair['warm']} → {pair['balanced']} | {f(pair['balanced_over_warm_time']['median'])} | {f(pair['balanced_over_warm_iterations']['median'])} | {f(pair['balanced_minus_warm_iterations']['median'], 1)} |")
        lines += ["", "各方法后续沿自己的实际计算历史推进，在线配对控制的是方向来源及接口，并非强制所有后续方向逐位相同。诊断部分才使用完全相同的当前空间和初值直接比较接口。", ""]
    lines += ["## 预定 12 个系统的离线诊断", "",
        "下表先逐 case 相对同一基础初值配对，再在每个几何内取时刻中位数，最后取 3 个独立几何的中位数。网络项有 9 个 case，当前轨迹 Ritz 项有 12 个。解误差能量是 eᵀAe；slow4 是前 4 个低端 Ritz 方向的 A 投影能量，并非完整慢特征空间。", "",
        "| 变体 | case数 | 迭代数几何中位数 | 迭代/基线几何中位数 | 初始误差能量/基线 | slow4剩余能量/基线 | PCG失败 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in DIAGNOSTIC_METHODS:
        v = diagnosis["variants"][name]
        g = v["geometric_medians"]
        lines.append(f"| {name} | {v['case_count']} | {f(g['iterations']['median'], 1)} | {f(g['iterations_over_base']['median'])} | {sci(g['error_energy_over_base']['median'])} | {sci(g['slow4_energy_over_base']['median'])} | {v['pcg_failures']} |")
    geometric = diagnosis["geometric_medians"]
    nn = diagnosis["variants"]["network_warm"]
    lines += ["", f"两种聚合口径须区分：把 9 个网络事件合并，初始误差能量比的中位数为 {f(nn['error_energy_over_base']['median'], 6)}（下降约 {f(100*(1-nn['error_energy_over_base']['median']), 1)}%）；先在每个几何内取事件中位数、再在 3 个几何间取中位数，则为 {f(nn['geometric_medians']['error_energy_over_base']['median'], 6)}（下降约 {f(100*(1-nn['geometric_medians']['error_energy_over_base']['median']), 1)}%）。差异来自中位数的聚合顺序，不是新增实验或改动结果；本报告主表使用后一口径，独立开发几何只有 3 个。",
        f"按同一几何优先口径，初始误差能量中位数为 {sci(geometric['initial_error_energy']['median'])}；slow4 在基础误差中的能量占比中位数为 {f(geometric['slow4_error_energy_fraction']['median'])}。",
        f"网络单方向与该 slow4 空间的 A 能量重合度几何中位数为 {f(geometric['network_direction_slow4_overlap']['median'])}，几何中位数范围 [{f(geometric['network_direction_slow4_overlap']['min'])}, {f(geometric['network_direction_slow4_overlap']['max'])}]。这个比值不是预测误差，也不能独立证明方向无效。",
        f"Ritz 投影方程相对残差最大值为 {sci(diagnosis['ritz_projected_residual']['max'])}，全算子 Ritz 相对残差最大值为 {sci(diagnosis['ritz_full_operator_residual']['max'])}。小型投影方程准确不代表已经得到完整算子的精确特征向量。",
        f"回顾性 Ritz 提取耗时几何中位数为 {f(geometric['retrospective_extraction_seconds']['median'], 4)} 秒；这不包含获得当前完整轨迹的先行求解，因而不是在线方向生成成本。诊断中的每步真残差及 rᵀMr 记录也会增加诊断成本，诊断耗时不参与在线排序。",
        "汇总 JSON 同时保留逐 case 统计及几何优先统计，避免用一种中位数冒充另一种口径。没有逐迭代 slow4 能量记录，因此本实验没有直接观测或证明慢方向误差在迭代中重新出现；只能报告同空间接口消融差异。", "",
        "同空间、同初值的诊断接口比较，迭代比和迭代差也采用几何优先口径：", "", "| 对照 | balanced/warm 迭代中位比 | 迭代差中位数 | 初始能量比最大绝对差 |", "|---|---:|---:|---:|"]
    for name, pair in diagnosis["matched_interfaces"].items():
        lines.append(f"| {name} | {f(pair['geometric_median_iteration_ratio']['median'])} | {f(pair['geometric_median_iteration_difference']['median'], 1)} | {sci(pair['max_abs_initial_energy_ratio_difference'])} |")
    lines += ["", "完整配对系统如下；当前轨迹列均为回顾性。", "",
        "| 几何/步 | 基础 | 网络warm | 网络balanced | 当前Ritz4 warm | 当前Ritz4 balanced | 当前Ritz8 balanced | 初始能量 | slow4占比 | 网络重合度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in diagnosis["cases"]:
        values = [str(case["variants"][m]["iterations"]) if m in case["variants"] else "—" for m in DIAGNOSTIC_METHODS]
        lines.append(f"| {case['seed']}/{case['step']} | " + " | ".join(values) + f" | {sci(case['initial_error_energy'])} | {f(case['slow4_error_energy_fraction'])} | {f(case.get('network_direction_slow4_overlap'))} |")
    lines += ["", "## 过去 Ritz 回收的维护成本审计", "",
        "以下仅为事后成本核算：固定已测迭代轨迹，先对每个完整重复合计 T、维护时间和方向/校正时间，再分别算 T−维护及 T−维护−方向/校正；对每个几何取重复中位数，除以同几何基础方法时间中位数，最后汇总几何配对比。匹配基础方法所容许的维护预算则为基础时间减去 T−维护的重复中位数。",
        "这些计算不是可执行的免费回收算法，也不是所有未来算法的时间下界；替换提取方式可能改变方向、迭代数和其他成本，不能假定实测轨迹保持不变。", "",
        "| 开发候选 | 实测维护秒数几何中位数 | 匹配基础方法可用维护预算几何中位数 | 配对(T−维护)/T基础 | 配对(T−维护−方向/校正)/T基础 |", "|---|---:|---:|---:|---:|"]
    for name, budget in stages["stage4_screen"]["retrospective_maintenance_budget_audit"].items():
        lines.append(f"| {name} | {f(budget['measured_maintenance_seconds']['median'], 4)} | {f(budget['maintenance_budget_to_match_baseline_seconds']['median'], 4)} | {f(budget['fixed_trajectory_without_maintenance_over_baseline']['median'], 4)} | {f(budget['fixed_trajectory_without_maintenance_and_correction_over_baseline']['median'], 4)} |")
    budget4 = stages["stage4_screen"]["retrospective_maintenance_budget_audit"]["ritz_balanced4"]
    lines += ["", f"对 ritz_balanced4，固定当前已测轨迹并扣除全部维护时间时，配对时间比中位数仍为 {f(budget4['fixed_trajectory_without_maintenance_over_baseline']['median'], 4)}；进一步扣除方向/校正构造时间后为 {f(budget4['fixed_trajectory_without_maintenance_and_correction_over_baseline']['median'], 4)}。这两个删成本比值应与 0.8 的工程线比较，不能仅凭减少维护成本便承诺达到 20% 加速。新提取机制是否改善轨迹、是否达到完整时间目标，仍需实际测量。"]
    lines += ["", "## 工程筛选与可支持的结论", "",
        f"开发集冻结的经典候选为 `{summary['frozen_selection']['classic_candidate']['name']}`；它按开发集完整时间选定，没有根据确认结果重选。", ""]
    for n in [128, 256]:
        stage = stages[f"stage4_confirm{n}"]
        for name, comparison in stage["versus_baseline"].items():
            lines.append(f"- {n} 网格 `{name}`：几何配对时间比中位数 {f(comparison['time_over_baseline']['median'])}；{comparison['geometries_faster']}/{stage['independent_geometries']} 个几何更快；至少节省 20% 的几何为 {comparison['geometries_at_least_20_percent_faster']}/{stage['independent_geometries']}。")
    lines += ["", "20% 仅为投入下一轮实验的工程线：报告同时列逐几何计数和配对中位数，不以少量几何概括一般性。更低的能量、残差或迭代数都不能替代完整时间收益；经典回收的收益不能归给 AI。",
        "若当前轨迹 Ritz 减少迭代而过去轨迹没有完整时间收益，只能保留‘当前预条件慢空间预测可能有价值’这一待检验假设；不能由本实验推断已有可训练的新模型。冻结的单方向网络失败也不能证明全部学习粗空间无效。",
        "本轮确认仍是机制探索，尚无多模型随机种子、广泛方程或论文级最终测试。后续算法改变需使用新分配的测试几何并重新冻结方案。", "",
        "## 文献边界和复核入口", "",
        "平衡两层、deflation 和 Krylov 回收是已有数值方法。直接参考 [Nabben–Vuik (2006)](https://epubs.siam.org/doi/10.1137/040608246)、[Tang 等 (2009)](https://link.springer.com/article/10.1007/s10915-009-9272-6) 与 [Parks 等 (2006)](https://epubs.siam.org/doi/10.1137/040607277)。",
        "学习 deflation 空间已有 [Kopaničáková 等 (2025)](https://link.springer.com/article/10.1007/s44379-025-00039-7)；学习谱空间亦有 [NeuraLSP](https://arxiv.org/html/2601.20174v3)。此处不把这些经典结构或近邻已有内容宣称为本项目的新颖性。",
        "理论与完整近邻界限见 [机制理论文档](../../docs/stage4_mechanism_theory_zh.md)，预定选择与计时规则见 [协议](../../docs/stage4_protocol_zh.md)。",
        "原始记录：[诊断](../stage4_diagnosis/measurements.json)、[筛选](../stage4_screen/measurements.json)、[128确认](../stage4_confirm128/measurements.json)、[256确认](../stage4_confirm256/measurements.json)。",
        "逐几何配对、全部组件、所有完整序列和输入哈希保存在 [summary.json](summary.json)。", "",
        "![在线时间与迭代](figures/online_comparison.png)", "", "![确认组件分解](figures/confirmation_components.png)", "",
        "![离线机制诊断](figures/retrospective_diagnosis.png)", ""]
    return "\n".join(lines)


def plots(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    target = out / "figures"
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for extension in ["png", "pdf"]:
            fig.savefig(target / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for column, (key, title) in enumerate([("stage4_screen", "Development 128 (3 geometries)"),
        ("stage4_confirm128", "Frozen confirmation 128 (5 geometries)"),
        ("stage4_confirm256", "Frozen confirmation 256 (same first 2)")]):
        methods = summary["stages"][key]["methods"]
        names = list(methods)
        x = np.arange(len(names))
        for row, field, ylabel in [(0, "total_seconds", "Complete 20-step sequence (s)"), (1, "iterations", "PCG iterations / sequence")]:
            ax = axes[row, column]
            values = [methods[n][field]["median"] for n in names]
            low = [methods[n][field]["min"] for n in names]
            high = [methods[n][field]["max"] for n in names]
            colors = ["#dc8055" if "network" in n else "#6597b0" for n in names]
            ax.bar(x, values, color=colors)
            ax.errorbar(x, values, yerr=[np.array(values)-low, np.array(high)-values], fmt="none", capsize=3, color="#333333")
            ax.set_xticks(x, [LABELS[n] for n in names], rotation=55, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title if row == 0 else "Geometry medians; whiskers show geometry range")
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Online mechanisms: all costs and recovery retained; current-trajectory Ritz excluded", fontsize=12)
    save(fig, "online_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    component_labels = ["Assembly", "History guess", "AMG setup", "Direction / projection", "PCG", "Recycle maintenance", "Recovery", "Other"]
    for ax, n in zip(axes, [128, 256]):
        methods = summary["stages"][f"stage4_confirm{n}"]["methods"]
        names = list(methods)
        bottom = np.zeros(len(names))
        for component, label in zip([*COMPONENTS, "other_seconds"], component_labels):
            values = np.array([methods[m][component]["median"] for m in names])
            ax.bar(np.arange(len(names)), values, bottom=bottom, label=label)
            bottom += values
        ax.set_xticks(np.arange(len(names)), [LABELS[m] for m in names], rotation=30, ha="right")
        ax.set_ylabel("Component median (s / complete sequence)")
        ax.set_title(f"Grid {n}; inference is inside direction cost")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Separately aggregated components; their sum need not equal the total-time median", fontsize=11)
    save(fig, "confirmation_components")

    diagnosis = summary["diagnosis"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), constrained_layout=True)
    methods = DIAGNOSTIC_METHODS[1:]
    for ax, field, title, logarithmic in [(axes[0], "iterations_over_base", "Paired PCG iterations / baseline", False),
        (axes[1], "error_energy_over_base", "Initial error energy / baseline", True),
        (axes[2], "slow4_energy_over_base", "Initial slow-4 energy / baseline", True)]:
        for index, method in enumerate(methods):
            observations = [case["variants"][method][field] for case in diagnosis["cases"] if method in case["variants"]]
            displayed = np.maximum(observations, 1e-30) if logarithmic else observations
            offsets = np.linspace(-.13, .13, len(observations))
            ax.scatter(index + offsets, displayed, s=19, alpha=.6, color="#dc8055" if "network" in method else "#6597b0")
            value = diagnosis["variants"][method]["geometric_medians"][field]["median"]
            ax.plot([index-.22, index+.22], [max(value, 1e-30) if logarithmic else value]*2, color="black", linewidth=2)
        ax.axhline(1., linestyle="--", color="grey", linewidth=1)
        ax.set_xticks(np.arange(len(methods)), [LABELS[m] for m in methods], rotation=40, ha="right")
        if logarithmic:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Retrospective diagnosis only: *current Ritz needs the already-completed current solve\n12 cases from 3 geometries (network: 9); points are cases, black bars are medians of geometry medians", fontsize=11)
    save(fig, "retrospective_diagnosis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve() if args.output else root / "results/stage4_report"
    splits = read(root / "data/stage4_splits.json")
    split_audit = check_splits(root, splits)
    selection_path = root / "results/stage4_screen/frozen_selection.json"
    frozen, frozen_hash = read(selection_path), sha(selection_path)
    check_hashes(root, frozen, "frozen_selection")
    require(sha(root / "results/stage4_screen/measurements.json") == frozen["screen_measurements_sha256"], "Screen measurements changed after selection")
    require(sha(root / "results/stage4_screen/manifest.json") == frozen["screen_manifest_sha256"], "Screen manifest changed after selection")
    require(frozen["confirmation_specs"] == [next(s for s in frozen["confirmation_specs"] if s["name"] == "baseline"),
        frozen["classic_candidate"], next(s for s in frozen["confirmation_specs"] if s["name"] == "network_warm"),
        next(s for s in frozen["confirmation_specs"] if s["name"] == "network_balanced")], "Confirmation order or candidate mismatch")
    finished = read(root / "results/stage4_complete.json")
    require(finished.get("complete") is True and finished.get("selection_sha256") == frozen_hash, "Final completion marker invalid")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "schema_version": 1,
        "aggregation": "sum 20 steps, median repeats within a geometry, then median/range across independent geometries; no failed rows discarded",
        "frozen_selection": frozen, "frozen_selection_sha256": frozen_hash, "split_audit": split_audit,
        "stages": {}, "audit": {}}
    all_rows = []
    for name in ["stage4_screen", "stage4_confirm128", "stage4_confirm256"]:
        manifest, rows, audit = load_online(root, name, splits, frozen, frozen_hash)
        summary["stages"][name] = aggregate_online(manifest, rows)
        summary["audit"][name] = audit
        all_rows.extend(rows)
    require(summary["audit"]["stage4_screen"]["rows_actual"] == 1080, "Expected exactly 1080 screen rows")
    require(sum(summary["audit"][s]["rows_actual"] for s in ["stage4_confirm128", "stage4_confirm256"]) == 1680,
            "Expected exactly 1680 confirmation rows")
    screen = summary["stages"]["stage4_screen"]["methods"]
    eligible = [name for name, value in screen.items() if value["spec"]["source"] not in ("none", "network")
                and value["all_pcg_accurate_without_recovery"] and value["all_final_accurate"]]
    chosen = frozen["classic_candidate"]["name"]
    require(chosen in eligible and screen[chosen]["total_seconds"]["median"] == min(screen[m]["total_seconds"]["median"] for m in eligible),
            "Frozen candidate violates the prespecified screen rule")
    summary["diagnosis"], summary["audit"]["stage4_diagnosis"] = load_diagnosis(root, splits, frozen)
    summary["online_accuracy"] = {"solves": len(all_rows), "pcg_failures": sum(not r["converged"] for r in all_rows),
        "recoveries": sum(r["solver_fallback"] for r in all_rows), "final_failures": sum(not r["final_converged"] for r in all_rows),
        "all_final_accurate": all(r["final_converged"] for r in all_rows),
        "max_final_relative_residual": max(r["final_relative_residual"] for r in all_rows)}
    summary["reporter_sha256"] = sha(Path(__file__))
    summary["theory_document_sha256"] = sha(root / "docs/stage4_mechanism_theory_zh.md")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    (out / "REPORT_zh.md").write_text(markdown(summary))
    plots(summary, out)
    print(json.dumps({"output": str(out), "online_solves": len(all_rows), "diagnostic_cases": 12,
        "all_final_accurate": summary["online_accuracy"]["all_final_accurate"], "selected_classic": chosen}, ensure_ascii=False))


if __name__ == "__main__":
    main()
