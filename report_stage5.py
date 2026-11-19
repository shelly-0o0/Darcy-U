#!/usr/bin/env python3
"""Audit cached-recycling measurements and report complete online costs.

This program never assembles or solves a PDE. It checks saved metadata and
reports failures without silently dropping observations or changing thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
SOURCES = ["fields.py", "numerics.py", "sequence_baselines.py", "balanced_augmentation.py",
           "cached_pcg.py", "cached_recycling_experiment.py"]
COMPONENTS = ["assembly_seconds", "guess_seconds", "setup_seconds", "correction_seconds",
              "solve_seconds", "maintenance_seconds", "recovery_seconds"]
QUANTITIES = ["total_seconds", *COMPONENTS, "other_seconds", "extraction_seconds",
              "solve_plus_maintenance_seconds", "event_solve_seconds", "event_solve_fraction", "iterations"]
BASELINES = ["baseline_scipy", "baseline_event_custom"]
SCREEN = [*BASELINES, "capture_only", "cached_balanced1", "cached_balanced2", "cached_balanced4",
          "explicit_balanced2", "cached_warm2"]
RTOL, STEPS = 1e-8, 20


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def dist(values):
    values = list(values)
    good = [float(v) for v in values if finite(v)]
    return dict(count=len(values), finite_count=len(good), median=median(good) if good else None,
                min=min(good) if good else None, max=max(good) if good else None)


def ratio(a, b):
    return a / b if finite(a) and finite(b) and b > 0 else None


def f(x, digits=3):
    return "不适用" if x is None else f"{x:.{digits}f}"


def sci(x):
    return "不适用" if x is None else f"{x:.3e}"


def check_hashes(root, record, where):
    require(set(record["source_sha256"]) == set(SOURCES), f"Source inventory differs: {where}")
    result = {"source_sha256": {name: sha(root / name) == digest
                              for name, digest in record["source_sha256"].items()}}
    require(all(result["source_sha256"].values()), f"Frozen sources changed: {where}")
    for key, name in [("protocol_sha256", "docs/stage5_protocol_zh.md"),
                      ("splits_sha256", "data/stage5_splits.json")]:
        result[key] = sha(root / name) == record[key]
        require(result[key], f"Frozen {key} differs: {where}")
    return result


def check_problem(manifest, where):
    for key, expected in [("steps", STEPS), ("rtol", RTOL), ("contrast", 1e4),
                          ("maxiter", 500), ("blas_threads", 1), ("device", "cpu")]:
        require(manifest.get(key) == expected, f"Unexpected {key}: {where}")


def check_splits(root, splits):
    groups = splits["groups"]
    require(set(groups) == {"development", "confirmation"}, "Unexpected split groups")
    require(len(groups["development"]) == 3 and len(groups["confirmation"]) == 5, "Unexpected split sizes")
    seeds = groups["development"] + groups["confirmation"]
    require(len(set(seeds)) == 8, "Duplicate stage-5 seed")
    hashes = [splits["geometry_sha256"][str(s)] for s in seeds]
    require(len(set(hashes)) == 8, "Duplicate stage-5 geometry")
    require(not set(seeds).intersection(splits["excluded_seeds"]), "Excluded seed reused")
    prior4 = read(root / "data/stage4_splits.json")
    prior3 = read(root / "data/stage3_splits.json")
    require(set(prior4["excluded_seeds"] + sum(prior4["groups"].values(), [])).issubset(splits["excluded_seeds"]),
            "Stage-4 exclusions are incomplete")
    require(set(prior3["excluded_prior_seeds"] + sum(prior3["groups"].values(), [])).issubset(splits["excluded_seeds"]),
            "Stage-3 exclusions are incomplete")
    for old in [prior3, prior4]:
        require(not set(hashes).intersection(old.get("geometry_sha256", {}).values()), "Saved geometry hash overlaps a prior stage")
    return dict(development_geometries=3, confirmation_geometries=5,
                confirmation_256_reuses_first_two=True,
                prior_split_sha256={f"data/stage{n}_splits.json": sha(root / f"data/stage{n}_splits.json") for n in [3, 4]},
                note="Checks saved metadata and available prior hashes; does not regenerate any coefficient fields.")


def load_audit(root, splits):
    folder = root / "results/stage5_cache_audit"
    manifest, rows, complete = [read(folder / n) for n in ["manifest.json", "measurements.json", "complete.json"]]
    check_problem(manifest, "cache_audit")
    checks = check_hashes(root, manifest, "cache_audit")
    config = manifest["configuration"]
    require(config["n"] == 128 and config["seeds"] == splits["groups"]["development"]
            and config["steps_observed"] == [0, 6, 10, 15], "Audit schedule differs")
    expected = {(s, step) for s in config["seeds"] for step in config["steps_observed"]}
    keys = [(r["seed"], r["step"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Audit cases missing, duplicate or unexpected")
    require(complete.get("complete") is True and complete.get("cases") == 12, "Audit completion marker invalid")
    rank_rows = defaultdict(list)
    for row in rows:
        for key in ["ad_relative_error", "mad_relative_error", "relative_residual"]:
            require(finite(row[key]) and row[key] >= 0, f"Invalid audit {key}")
        require(row["converged"] == (row["relative_residual"] <= RTOL), "Audit convergence flag inconsistent")
        require(0 <= row["captured_vectors"] <= 24 and set(row["ranks"]) == {"1", "2", "4"}, "Audit cache inventory differs")
        for q, value in row["ranks"].items():
            raw_rank_matches = value["cached"]["rank"] == value["explicit"]["rank"] == int(q)
            # The runner also checks ranks after independent A-orthogonalization.
            # Those arrays are not saved, so a failed independent check cannot be
            # reconstructed from the extraction's rank alone.
            rank_matches = value["rank_matches"]
            require(not rank_matches or raw_rank_matches, "Audit rank comparison flag inconsistent")
            ev, angle = value["relative_ritz_value_error"], value["max_principal_angle_sine"]
            if rank_matches:
                require(finite(ev) and ev >= 0 and finite(angle) and 0 <= angle <= 1, "Invalid audit eigenspace error")
            passed = bool(rank_matches and ev <= 1e-5 and angle <= 1e-3)
            require(value["passed"] == passed, "Audit rank threshold flag inconsistent")
            rank_rows[q].append(dict(seed=row["seed"], step=row["step"], **value))
        passed = bool(row["converged"] and row["ad_relative_error"] <= 1e-7 and row["mad_relative_error"] <= 1e-7
                      and all(v["passed"] for v in row["ranks"].values()))
        require(row["passed"] == passed, "Audit system threshold flag inconsistent")
    all_passed = all(r["passed"] for r in rows)
    require(complete["all_passed"] == all_passed, "Audit completion outcome inconsistent")
    hashes = {str(p.relative_to(root)): sha(p) for p in sorted(folder.glob("*.json"))}
    return dict(cases=12, independent_geometries=3, all_passed=all_passed,
        failed_cases=[dict(seed=r["seed"], step=r["step"]) for r in rows if not r["passed"]],
        thresholds=dict(ad_relative_error=1e-7, mad_relative_error=1e-7, relative_ritz_value_error=1e-5,
                        max_principal_angle_sine=1e-3),
        ad_relative_error=dist(r["ad_relative_error"] for r in rows),
        mad_relative_error=dist(r["mad_relative_error"] for r in rows),
        ranks={q: dict(passed=sum(r["passed"] for r in rr),
                      relative_ritz_value_error=dist(r["relative_ritz_value_error"] for r in rr),
                      max_principal_angle_sine=dist(r["max_principal_angle_sine"] for r in rr)) for q, rr in rank_rows.items()},
        captured_rejected_vectors=sum(r["capture_stats"]["rejected_vectors"] for r in rows),
        captured_terminal_vectors_skipped=sum(r["capture_stats"]["skipped_terminal_vectors"] for r in rows),
        raw_cases=rows, input_sha256=hashes, frozen_hash_checks=checks)


def load_online(root, name, splits, frozen=None):
    folder = root / "results" / name
    manifest, rows, complete, refs = [read(folder / n) for n in ["manifest.json", "measurements.json", "complete.json", "references.json"]]
    is_screen, n = name == "stage5_screen", 256 if name == "stage5_confirm256" else 128
    seeds = splits["groups"]["development" if is_screen else "confirmation"]
    if n == 256:
        seeds = seeds[:2]
    repeats = 2 if is_screen else 3
    config = manifest["configuration"]
    names = [s["name"] for s in config["specs"]]
    require(config["n"] == n and config["seeds"] == seeds and config["repeats"] == repeats, f"Schedule differs: {name}")
    require(len(names) == len(set(names)), f"Duplicate method: {name}")
    if is_screen:
        require(names == SCREEN, "All eight prespecified screen methods must be retained")
        expected_specs = [
            dict(name="baseline_scipy", source="none", rank=0, interface="warm", event_custom=False),
            dict(name="baseline_event_custom", source="none", rank=0, interface="warm", event_custom=True),
            dict(name="capture_only", source="capture", rank=0, interface="warm", event_custom=True),
            *[dict(name=f"cached_balanced{q}", source="cached", rank=q, interface="balanced", event_custom=True) for q in [1, 2, 4]],
            dict(name="explicit_balanced2", source="explicit", rank=2, interface="balanced", event_custom=True),
            dict(name="cached_warm2", source="cached", rank=2, interface="warm", event_custom=True)]
        require(config["specs"] == expected_specs, "Screen algorithm settings differ")
    else:
        require(frozen is not None and config["specs"] == frozen["confirmation_specs"], "Confirmation settings differ")
        require(config["selection_sha256"] == sha(root / "results/stage5_screen/frozen_selection.json"), "Selection hash differs")
    expected = {(n, method, seed, repeat, step) for method in names for seed in seeds
                for repeat in range(repeats) for step in range(STEPS)}
    keys = [(r["n"], r["method"], r["seed"], r["repeat"], r["step"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, f"Missing, duplicate or unexpected rows: {name}")
    require(complete.get("complete") is True and complete.get("rows") == len(rows), f"Completion marker differs: {name}")
    check_problem(manifest, name)
    hashes = check_hashes(root, manifest, name)
    settings = {s["name"]: s for s in config["specs"]}
    for row in rows:
        for key in ["total_seconds", *COMPONENTS, "relative_residual", "final_relative_residual", "energy_relative_error",
                    "base_relative_residual", "corrected_relative_residual", "orthogonality_error"]:
            require(finite(row[key]) and row[key] >= 0, f"Invalid {key}: {name}")
        require(row["final_converged"] == (row["final_relative_residual"] <= RTOL), "Final accuracy flag inconsistent")
        require(row["converged"] == (row["relative_residual"] <= RTOL), "PCG accuracy flag inconsistent")
        require(row["solver_fallback"] == (not row["converged"]), "Recovery flag inconsistent")
        require(isinstance(row["iterations"], int) and 0 <= row["iterations"] <= 500, "Iteration budget violated")
        require(isinstance(row["restarts"], int) and 0 <= row["restarts"] <= 3, "Refresh budget violated")
        require(sum(row[key] for key in COMPONENTS) <= row["total_seconds"] + 1e-8, "Component cost exceeds total")
        require(finite(row["extraction_seconds"]) and 0 <= row["extraction_seconds"] <= row["maintenance_seconds"] + 1e-8,
                "Extraction time is not nested in maintenance")
        setting = settings[row["method"]]
        require(0 <= row["rank"] <= setting["rank"], "Applied rank exceeds requested rank")
        if row["refresh_event"]:
            require(row["rank"] == 0, "Old recycling space used on a refresh event")
        cap = row["capture_stats"]
        if row["refresh_event"] and setting["event_custom"]:
            require(0 <= cap["captured_vectors"] <= 24, "Capture capacity exceeded")
            require(cap["extra_a_applications_for_cache"] == cap["extra_m_applications_for_cache"] == 0,
                    "Capture claims extra operator applications")
            require(len(cap["segment_ids"]) == cap["captured_vectors"], "Capture segment metadata incomplete")
            require(all(0 <= segment <= row["restarts"] for segment in cap["segment_ids"]), "Invalid cache segment id")
            if setting["source"] == "none":
                require(cap["captured_vectors"] == 0, "No-capture baseline collected vectors")
        else:
            require(not cap, "Capture unexpectedly enabled outside the prescribed event")
        extraction = row["extraction"]
        expected_extraction = row["refresh_event"] and setting["source"] in ("cached", "explicit")
        require(bool(extraction) == expected_extraction, "Extraction event differs from protocol")
        rejected = bool(extraction and extraction["status"] not in ("accepted", "empty_basis"))
        require(row["cache_rejected"] == rejected, "Cache rejection flag inconsistent")
        if extraction and setting["source"] == "cached":
            require(extraction["a_applications"] == extraction["m_applications"] == 0, "Cached extraction uses new operators")
    require(len(refs) == len(seeds) and {r["seed"] for r in refs} == set(seeds), "Reference inventory differs")
    for ref in refs:
        require(len(ref["relative_residuals"]) == STEPS and finite(ref["seconds"]) and ref["seconds"] >= 0, "Invalid reference record")
        require(all(finite(v) and v >= 0 for v in ref["relative_residuals"]), "Invalid reference residual")
    return manifest, rows, dict(expected_rows=len(expected), actual_rows=len(rows), duplicate_rows=0,
        frozen_hash_checks=hashes,
        input_sha256={p.name: sha(p) for p in sorted(folder.glob("*.json"))},
        offline_reference_seconds=sum(r["seconds"] for r in refs),
        reference_relative_residual=dist(v for r in refs for v in r["relative_residuals"]),
        all_reference_accurate=all(v <= RTOL for r in refs for v in r["relative_residuals"]))


def comparison(methods, numerator, denominator):
    a, b = methods[numerator], methods[denominator]
    pairs = []
    for seed, va in a["by_geometry"].items():
        vb = b["by_geometry"][seed]
        pair = dict(seed=int(seed))
        for key in ["total_seconds", "iterations", "solve_plus_maintenance_seconds", "maintenance_seconds", "extraction_seconds"]:
            pair[key + "_ratio"] = ratio(va[key]["median"], vb[key]["median"])
            av, bv = va[key]["median"], vb[key]["median"]
            pair[key + "_difference"] = av - bv if av is not None and bv is not None else None
        pairs.append(pair)
    return dict(numerator=numerator, denominator=denominator, by_geometry=pairs,
        **{key: dist(p[key] for p in pairs) for key in pairs[0] if key != "seed"})


def aggregate(manifest, rows):
    config = manifest["configuration"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"], row["repeat"])].append(row)
    sequences = []
    for (method, seed, repeat), group in sorted(grouped.items()):
        item = dict(method=method, seed=seed, repeat=repeat)
        for key in ["total_seconds", *COMPONENTS, "iterations"]:
            item[key] = sum(r[key] for r in group)
        item["other_seconds"] = item["total_seconds"] - sum(item[k] for k in COMPONENTS)
        item["solve_plus_maintenance_seconds"] = item["solve_seconds"] + item["maintenance_seconds"]
        item["extraction_seconds"] = sum(r["extraction_seconds"] for r in group) if all("extraction_seconds" in r for r in group) else None
        item["event_solve_seconds"] = sum(r["solve_seconds"] for r in group if r["refresh_event"])
        item["event_solve_fraction"] = ratio(item["event_solve_seconds"], item["solve_seconds"])
        item.update(pcg_failures=sum(not r["converged"] for r in group), recoveries=sum(r["solver_fallback"] for r in group),
                    final_failures=sum(not r["final_converged"] for r in group), cache_rejections=sum(r["cache_rejected"] for r in group))
        sequences.append(item)
    methods = {}
    for setting in config["specs"]:
        name = setting["name"]
        ss = [r for r in sequences if r["method"] == name]
        rr = [r for r in rows if r["method"] == name]
        geometry = {str(seed): {key: dist(r[key] for r in ss if r["seed"] == seed) for key in QUANTITIES}
                    for seed in config["seeds"]}
        failures = {key: sum(r[key] for r in ss) for key in ["pcg_failures", "recoveries", "final_failures", "cache_rejections"]}
        extraction = [r["extraction"] for r in rr if r["extraction"]]
        methods[name] = dict(spec=setting, by_geometry=geometry,
            **{key: dist(v[key]["median"] for v in geometry.values()) for key in QUANTITIES}, **failures,
            eligible=all(v == 0 for v in failures.values()),
            final_relative_residual=dist(r["final_relative_residual"] for r in rr),
            pcg_relative_residual_before_recovery=dist(r["relative_residual"] for r in rr),
            energy_relative_error=dist(r["energy_relative_error"] for r in rr),
            applied_rank=dist(r["rank"] for r in rr if not r["refresh_event"]),
            captured_vectors=dist(r["capture_stats"]["captured_vectors"] for r in rr if r["capture_stats"]),
            extraction_status_counts=dict(Counter(r["status"] for r in extraction)),
            capture_rejected_vectors=sum(r["capture_stats"].get("rejected_vectors", 0) for r in rr),
            capture_terminal_vectors_skipped=sum(r["capture_stats"].get("skipped_terminal_vectors", 0) for r in rr),
            extraction_rejected_vectors=sum(r.get("rejected_vectors", r.get("basis", {}).get("rejected_vectors", 0)) for r in extraction),
            extraction_dependent_vectors=sum(r.get("dependent_vectors", r.get("basis", {}).get("dependent_vectors", 0)) for r in extraction),
            extraction_selected_rank=dist(r["rank"] for r in extraction),
            max_iterations_one_solve=max(r["iterations"] for r in rr))
    pairs = {}
    for name in methods:
        for baseline in BASELINES:
            if name != baseline:
                pairs[f"{name}__over__{baseline}"] = comparison(methods, name, baseline)
    # Compare the same rank and the same event strategy; never compare stage-4 timings here.
    for q in [1, 2, 4]:
        cached, explicit, warm = f"cached_balanced{q}", f"explicit_balanced{q}", f"cached_warm{q}"
        for numerator, denominator in [(cached, explicit), (cached, warm)]:
            if numerator in methods and denominator in methods:
                pairs[f"{numerator}__over__{denominator}"] = comparison(methods, numerator, denominator)
    if "capture_only" in methods:
        pairs["capture_only__over__baseline_event_custom"] = comparison(methods, "capture_only", "baseline_event_custom")
    valid_baselines = [b for b in BASELINES if methods[b]["eligible"]]
    fastest = min(valid_baselines, key=lambda b: methods[b]["total_seconds"]["median"]) if valid_baselines else None
    conservative = {}
    fixed_comparisons = {}
    for name, value in methods.items():
        if name in BASELINES:
            continue
        geo = []
        for seed, result in value["by_geometry"].items():
            base = min(valid_baselines, key=lambda b: methods[b]["by_geometry"][seed]["total_seconds"]["median"]) if valid_baselines else None
            denominator = methods[base]["by_geometry"][seed]["total_seconds"]["median"] if base else None
            geo.append(dict(seed=int(seed), denominator=base, numerator_seconds=result["total_seconds"]["median"],
                            denominator_seconds=denominator, time_ratio=ratio(result["total_seconds"]["median"], denominator)))
        ratios = dist(g["time_ratio"] for g in geo)
        conservative[name] = dict(by_geometry=geo, time_ratio=ratios,
            geometries_faster=sum(g["time_ratio"] is not None and g["time_ratio"] < 1 for g in geo),
            geometries_at_least_20_percent_faster=sum(g["time_ratio"] is not None and g["time_ratio"] <= .8 for g in geo),
            passes_median_20_percent_engineering_gate=bool(value["eligible"] and ratios["median"] is not None and ratios["median"] <= .8))
        fixed_geo = []
        for seed, result in value["by_geometry"].items():
            denominator = methods[fastest]["by_geometry"][seed]["total_seconds"]["median"] if fastest else None
            fixed_geo.append(dict(seed=int(seed), denominator=fastest, numerator_seconds=result["total_seconds"]["median"],
                                  denominator_seconds=denominator, time_ratio=ratio(result["total_seconds"]["median"], denominator)))
        fixed_ratios = dist(g["time_ratio"] for g in fixed_geo)
        fixed_comparisons[name] = dict(denominator=fastest, by_geometry=fixed_geo, time_ratio=fixed_ratios,
            geometries_faster=sum(g["time_ratio"] is not None and g["time_ratio"] < 1 for g in fixed_geo),
            geometries_at_least_20_percent_faster=sum(g["time_ratio"] is not None and g["time_ratio"] <= .8 for g in fixed_geo),
            passes_median_20_percent_engineering_gate=bool(value["eligible"] and fixed_ratios["median"] is not None and fixed_ratios["median"] <= .8))
    budgets = {}
    for name, value in methods.items():
        if value["spec"]["source"] not in ("cached", "explicit"):
            continue
        geo = []
        for seed in config["seeds"]:
            local = [s for s in sequences if s["method"] == name and s["seed"] == seed]
            extraction_free = median(s["total_seconds"] - s["extraction_seconds"] for s in local)
            entry = dict(seed=seed, measured_total_seconds=value["by_geometry"][str(seed)]["total_seconds"]["median"],
                         time_minus_extraction_seconds=extraction_free)
            for baseline in BASELINES:
                base_time = methods[baseline]["by_geometry"][str(seed)]["total_seconds"]["median"]
                entry[f"time_minus_extraction_over_{baseline}"] = ratio(extraction_free, base_time)
            geo.append(entry)
        budgets[name] = dict(by_geometry=geo, **{key: dist(g[key] for g in geo) for key in geo[0] if key != "seed"},
            interpretation="Retrospective subtraction on fixed measured trajectories only. Not an executable zero-cost extraction algorithm, a speedup forecast, or a universal lower bound.")
    return dict(n=config["n"], independent_geometries=len(config["seeds"]), repeats_per_geometry=config["repeats"],
        methods=methods, paired_comparisons=pairs, fastest_valid_baseline_by_overall_median=fastest,
        versus_fastest_valid_baseline=fixed_comparisons,
        retrospective_per_geometry_baseline_envelope=conservative,
        retrospective_extraction_budget=budgets, complete_sequences=sequences)


def markdown(summary):
    audit, stages = summary["cache_audit"], summary["stages"]
    lines = ["# 第五阶段：复用 PCG 缓存的低秩回收实验", "", f"生成时间：{summary['created_utc']}。状态：`{summary['status']}`。", "",
        "本阶段没有训练或调用 AI 模型。任何收益首先归于经典数值缓存、回收策略及实现，不能据此声称 AI 加速或新的学习算法。", "",
        "## 预定缓存等价核验", "", "三个开发几何、四个固定时刻，共 12 个系统；同一几何内时刻不独立。核验门槛在正式运行前固定，没有按结果放宽。", "",
        "| 指标 | 最大值 | 预定门槛 |", "|---|---:|---:|",
        f"| 缓存 AD 相对偏差 | {sci(audit['ad_relative_error']['max'])} | 1e-7 |",
        f"| 缓存 MAD 相对偏差 | {sci(audit['mad_relative_error']['max'])} | 1e-7 |"]
    for q, value in audit["ranks"].items():
        lines += [f"| 秩 {q} Ritz 值相对偏差 | {sci(value['relative_ritz_value_error']['max'])} | 1e-5 |",
                  f"| 秩 {q} 最大主角正弦 | {sci(value['max_principal_angle_sine']['max'])} | 1e-3 |"]
    lines += ["", f"核验通过 {12 - len(audit['failed_cases'])}/12。缓存提取使用已支付的 D、AD、MAD；最后一项若没有已支付的下一次 Mr，则跳过；真残差刷新前后不相减。", "",
        "| 几何/步 | AD 偏差 | MAD 偏差 | 秩1/2/4通过 | PCG真残差 |", "|---|---:|---:|---|---:|"]
    for row in audit["raw_cases"]:
        passed = "/".join("是" if row["ranks"][q]["passed"] else "否" for q in ["1", "2", "4"])
        lines.append(f"| {row['seed']}/{row['step']} | {sci(row['ad_relative_error'])} | {sci(row['mad_relative_error'])} | {passed} | {sci(row['relative_residual'])} |")
    if not audit["all_passed"]:
        lines += ["", "离线等价核验未通过，按协议停止进入正式筛选。失败记录完整保留；本轮不能给出在线收益结论。", "", "![缓存等价核验](figures/cache_equivalence.png)", ""]
        return "\n".join(lines)
    lines += ["", "## 在线成本与统计口径", "",
        "每条完整序列先合计 20 步，再取同几何重复中位数，最后取几何中位数与范围；图中范围不是置信区间。所有失败、恢复和缓存拒绝均保留。确认仅有五个独立几何，256 网格复用前两个。",
        "总时间包括组装、历史四解初值、AMG 刷新、当前方向投影、求解、缓存采集与复制、真残差、恢复、提取及历史维护。直接参考解只用于计时外误差核验。采集开销可能位于求解内，因此必须同时检查 solve+maintenance，不能只比较维护时间。",
        "提取秒数若单独记录，是维护时间的子项，不能再加到总时间中；组件各自取中位数，其和不必等于总时间中位数。显式对照与缓存对照采集相同的三份缓存，仅在提取阶段重新施加 A/M，以隔离提取机制；这不证明显式方案已经存储最优。", ""]
    acc = summary.get("online_accuracy", {})
    if acc:
        lines += [f"已核验在线记录 {acc['solves']} 条，PCG 失败 {acc['pcg_failures']}，恢复 {acc['recoveries']}，最终失败 {acc['final_failures']}，整个缓存空间拒绝 {acc['cache_rejections']}；最终真残差最大值 {sci(acc['max_final_relative_residual'])}。",
            f"逐列统计另计：采集时拒绝 {acc['capture_rejected_vectors']} 列，末步缺少下一次已支付 Mr 而跳过 {acc['capture_terminal_vectors_skipped']} 列，提取时拒绝 {acc['extraction_rejected_vectors']} 列，因线性相关筛去 {acc['extraction_dependent_vectors']} 列。末步跳过及相关列筛选不等于整个空间被拒绝。", ""]
    titles = {"stage5_screen": "开发筛选：128网格、3几何、2重复", "stage5_confirm128": "冻结确认：128网格、5几何、3重复",
              "stage5_confirm256": "冻结确认：256网格、2已有几何、3重复"}
    for key, stage in stages.items():
        lines += [f"## {titles[key]}", "", "| 方法 | 总秒数中位数 [几何范围] | 迭代中位数 | 对SciPy配对时间比 | 对自实现配对时间比 | 失败/恢复/最终失败/拒绝 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, value in stage["methods"].items():
            timing = value["total_seconds"]
            rr = [1. if name == b else stage["paired_comparisons"][f"{name}__over__{b}"]["total_seconds_ratio"]["median"] for b in BASELINES]
            failures = "/".join(str(value[k]) for k in ["pcg_failures", "recoveries", "final_failures", "cache_rejections"])
            lines.append(f"| {name} | {f(timing['median'])} [{f(timing['min'])}, {f(timing['max'])}] | {f(value['iterations']['median'],1)} | {f(rr[0])} | {f(rr[1])} | {failures} |")
        lines += ["", "| 方法 | PCG秒数 | 维护秒数 | 其中提取秒数 | PCG+维护秒数 | 当前方向/投影秒数 |", "|---|---:|---:|---:|---:|---:|"]
        for name, value in stage["methods"].items():
            lines.append("| " + name + " | " + " | ".join(f(value[k]["median"],4) for k in ["solve_seconds", "maintenance_seconds", "extraction_seconds", "solve_plus_maintenance_seconds", "correction_seconds"]) + " |")
        lines += ["", "同秩同策略对照及实现/采集对照：比值小于 1 表示分子较小。", "", "| 分子 / 分母 | 总时间配对比 | 迭代配对比 | PCG+维护配对比 | 维护配对比 | 提取配对比 |", "|---|---:|---:|---:|---:|---:|"]
        for name, pair in stage["paired_comparisons"].items():
            selected = (pair["numerator"] == "baseline_event_custom" and pair["denominator"] == "baseline_scipy") or (pair["numerator"] == "capture_only" and pair["denominator"] == "baseline_event_custom") or pair["denominator"] not in BASELINES
            if selected:
                lines.append(f"| {pair['numerator']} / {pair['denominator']} | " + " | ".join(f(pair[k+"_ratio"]["median"]) for k in ["total_seconds", "iterations", "solve_plus_maintenance_seconds", "maintenance_seconds", "extraction_seconds"]) + " |")
        lines += ["", f"按本组全部几何的中位时间选出的最快有效基线为 `{stage['fastest_valid_baseline_by_overall_median']}`。下表固定使用这一种基线，再逐几何配对；不在每个几何上另选基线。JSON 另外保留逐几何两基线取小的事后保守包络，它不可作为可执行选择器或主结论。", "",
            "| 候选 | 对固定最快有效基线时间比中位数 | 更快几何数 | 至少节省20%几何数 | 配对中位20%工程线 |", "|---|---:|---:|---:|---|"]
        for name, value in stage["versus_fastest_valid_baseline"].items():
            lines.append(f"| {name} | {f(value['time_ratio']['median'])} | {value['geometries_faster']}/{stage['independent_geometries']} | {value['geometries_at_least_20_percent_faster']}/{stage['independent_geometries']} | {'通过' if value['passes_median_20_percent_engineering_gate'] else '未通过'} |")
        lines += ["", "以下为事后固定轨迹成本核算：每次完整重复计算 T−提取时间，再在同几何取重复中位数并对基线配对。不是免费提取算法、加速预测或普遍时间下界。", "",
            "| 回收候选 | (T−提取)/SciPy 配对中位比 | (T−提取)/自实现 配对中位比 |", "|---|---:|---:|"]
        for name, value in stage["retrospective_extraction_budget"].items():
            lines.append(f"| {name} | {f(value['time_minus_extraction_over_baseline_scipy']['median'],4)} | {f(value['time_minus_extraction_over_baseline_event_custom']['median'],4)} |")
        event_shares = [f"`{b}` 为 {f(100*stage['methods'][b]['event_solve_fraction']['median'],1)}%" for b in BASELINES]
        lines += ["", "预定事件步占基础求解时间的份额（每个完整重复先算比例，再取几何优先中位数）：" + "，".join(event_shares) + "。本阶段事件步禁用旧空间，因此事件求解成本仍需通过其他机制改善。"]
        lines += [""]
    if "frozen_selection" in summary:
        selected = summary["frozen_selection"]["candidate"]["name"]
        lines += ["## 冻结选择与全部确认几何配对", "", f"开发集预定规则选中 `{selected}`。选择只在秩 1/2/4 的合格缓存 balanced 候选间进行，即使它们均慢于基线也会选一个确认；选中本身不代表收益。", "",
                  "| 网格/方法/几何 | 总秒数 | SciPy秒数 | 自实现秒数 | 对该网格固定最快有效基线比 |", "|---|---:|---:|---:|---:|"]
        for key in ["stage5_confirm128", "stage5_confirm256"]:
            if key not in stages:
                continue
            stage = stages[key]
            for name, values in stage["versus_fastest_valid_baseline"].items():
                for g in values["by_geometry"]:
                    seed = str(g["seed"])
                    times = [stage["methods"][b]["by_geometry"][seed]["total_seconds"]["median"] for b in BASELINES]
                    lines.append(f"| {stage['n']}/{name}/{seed} | {f(g['numerator_seconds'],4)} | {f(times[0],4)} | {f(times[1],4)} | {f(g['time_ratio'],4)} |")
    if summary["status"] != "complete":
        lines += ["", f"本轮按协议停止：{summary.get('stop_reason','尚未完成确认')}。尚无完整确认结论。"]
    lines += ["", "## 可支持的结论与边界", "",
        "缓存等价核验通过只说明在这批已检查轨迹上，缓存和显式提取一致；它不证明所有系统上的舍入误差均可忽略。全算子 Ritz 方向仍是有限空间的近似，不能把小型投影方程精度当成完整特征向量精度。",
        "判断收益应先看自实现无采集基线与 SciPy 的差异，再看采集开销、同秩缓存/显式差异和 warm/balanced 差异；迭代减少、维护减少或某个单次测量更快都不等价于完整时间改善。每种方法沿自己的数值历史推进，在线空间并非逐位相同。",
        "20% 是工程投入筛选线，不是统计显著性或论文录用标准。当前只有单右端、单对比度及少量新几何；不能直接外推到其他 PDE 或声称达到国际一区发表条件。本阶段不包含学习模型，不产生 AI 创新或 AI 加速证据。",
        "本阶段事件步禁用旧空间，只用本次缓存更新供下一步使用；第四阶段事件策略不同，因此不能用跨阶段计时差直接归因缓存优化。后续任何方案改变都需重新冻结实验并使用新的确认几何。", "",
        "逐步原始记录见 [开发集](../stage5_screen/measurements.json)、[128 确认集](../stage5_confirm128/measurements.json)、[256 确认集](../stage5_confirm256/measurements.json)，离线核验见 [缓存核验记录](../stage5_cache_audit/measurements.json)。逐几何配对、完整重复序列成本、失败及哈希审计见 [summary.json](summary.json)；预定规则见 [协议](../../docs/stage5_protocol_zh.md)。", "",
        "![在线完整时间与迭代](figures/online_comparison.png)", "", "![组件及采集提取成本](figures/cost_breakdown.png)", "",
        "![缓存等价核验](figures/cache_equivalence.png)", ""]
    return "\n".join(lines)


def plots(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(folder / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    rows = summary["cache_audit"]["raw_cases"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for field, label in [("ad_relative_error", "AD"), ("mad_relative_error", "MAD")]:
        axes[0].plot(range(12), [max(r[field], 1e-18) for r in rows], marker="o", markersize=3, label=label)
    axes[0].axhline(1e-7, color="firebrick", linestyle="--", label="Prespecified limit")
    axes[0].set_title("Cached operator-action identities")
    for ax, field, limit, title in [(axes[1], "relative_ritz_value_error", 1e-5, "Ritz values: cached versus explicit"),
                                   (axes[2], "max_principal_angle_sine", 1e-3, "A-inner-product subspace angle")]:
        for q in ["1", "2", "4"]:
            ax.plot(range(12), [max(r["ranks"][q][field], 1e-18) if r["ranks"][q][field] is not None else np.nan for r in rows],
                    marker="o", markersize=3, label=f"Rank {q}")
        ax.axhline(limit, color="firebrick", linestyle="--", label="Prespecified limit")
        ax.set_title(title)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xticks(range(12), [f"{r['seed']}\nstep {r['step']}" for r in rows], rotation=90, fontsize=6)
        ax.set_ylabel("Relative error / angle sine (zero displayed at 1e-18)")
        ax.grid(axis="y", alpha=.15)
        ax.legend(fontsize=7)
    fig.suptitle("Offline equivalence audit: 12 systems from 3 geometries; thresholds fixed before measurement")
    save(fig, "cache_equivalence")
    stages = summary["stages"]
    if not stages:
        return
    fig, axes = plt.subplots(2, len(stages), figsize=(5.4*len(stages), 8.5), squeeze=False, constrained_layout=True)
    for col, (name, stage) in enumerate(stages.items()):
        methods = stage["methods"]
        names = list(methods)
        for row, field, label in [(0, "total_seconds", "Complete 20-step sequence (s)"), (1, "iterations", "PCG iterations / sequence")]:
            ax = axes[row, col]
            values = np.array([methods[n][field]["median"] for n in names])
            low = np.array([methods[n][field]["min"] for n in names])
            high = np.array([methods[n][field]["max"] for n in names])
            ax.bar(range(len(names)), values, color=["#a2a7ae" if n in BASELINES else "#377c9b" if "cached" in n else "#d69e57" for n in names])
            ax.errorbar(range(len(names)), values, yerr=[values-low, high-values], fmt="none", capsize=3, color="#333333")
            ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
            ax.set_ylabel(label)
            ax.set_title(f"{'Development' if name.endswith('screen') else 'Frozen confirmation'} {stage['n']}\n{stage['independent_geometries']} geometries; {stage['repeats_per_geometry']} repeats")
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("All online costs and all failures retained; geometry medians with geometry ranges")
    save(fig, "online_comparison")
    selected_stages = [(k, v) for k, v in stages.items() if "confirm" in k] or list(stages.items())
    fig, axes = plt.subplots(2, len(selected_stages), figsize=(7*len(selected_stages), 9), squeeze=False, constrained_layout=True)
    labels = ["Assembly", "History guess", "AMG setup", "Directions / projection", "PCG (capture included)", "Maintenance", "Recovery", "Other"]
    for col, (name, stage) in enumerate(selected_stages):
        methods, ax = stage["methods"], axes[0, col]
        names = list(methods)
        bottom = np.zeros(len(names))
        for field, label in zip([*COMPONENTS, "other_seconds"], labels):
            values = np.array([methods[n][field]["median"] for n in names])
            ax.bar(range(len(names)), values, bottom=bottom, label=label)
            bottom += values
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Component median (s / complete sequence)")
        ax.set_title(f"Grid {stage['n']}: separately aggregated components")
        if col == len(selected_stages)-1:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax = axes[1, col]
        for offset, field, label in [(-.25, "solve_plus_maintenance_seconds", "PCG + maintenance"), (0., "maintenance_seconds", "Maintenance"), (.25, "extraction_seconds", "Extraction (inside maintenance)")]:
            values = [methods[n][field]["median"] for n in names]
            if any(v is not None for v in values):
                ax.bar(np.arange(len(names))+offset, [v if v is not None else np.nan for v in values], width=.23, label=label)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Median seconds / complete sequence")
        ax.set_title("Capture copies remain included; extraction is not added twice")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Cost audit: compare complete totals and solve + maintenance; component medians need not sum to the total median")
    save(fig, "cost_breakdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve() if args.output else root / "results/stage5_report"
    splits = read(root / "data/stage5_splits.json")
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), schema_version=1,
        aggregation="sum 20 steps, median repeats within geometry, then median/range across geometries; retain all failed rows",
        split_audit=check_splits(root, splits), cache_audit=load_audit(root, splits), stages={}, audit={},
        reporter_sha256=sha(Path(__file__)), status="audit_failed")
    all_rows = []
    if summary["cache_audit"]["all_passed"]:
        manifest, rows, checks = load_online(root, "stage5_screen", splits)
        summary["stages"]["stage5_screen"] = aggregate(manifest, rows)
        summary["audit"]["stage5_screen"] = checks
        all_rows.extend(rows)
        require(len(rows) == 960, "Expected exactly 960 development rows")
        methods = summary["stages"]["stage5_screen"]["methods"]
        eligible = [n for n in ["cached_balanced1", "cached_balanced2", "cached_balanced4"] if methods[n]["eligible"]]
        selection_path = root / "results/stage5_screen/frozen_selection.json"
        if not eligible:
            require(not selection_path.exists(), "Ineligible candidate was frozen")
            summary.update(status="screen_no_eligible_candidate", stop_reason="开发集没有同时准确、零求解失败、零缓存拒绝的缓存 balanced 候选")
        else:
            frozen = read(selection_path)
            check_hashes(root, frozen, "selection")
            require(frozen["audit_sha256"] == summary["cache_audit"]["input_sha256"], "Frozen audit files changed")
            require(frozen["screen_measurements_sha256"] == sha(selection_path.parent / "measurements.json"), "Frozen development measurements changed")
            require(frozen["screen_manifest_sha256"] == sha(selection_path.parent / "manifest.json"), "Frozen development manifest changed")
            candidate = frozen["candidate"]
            name, q = candidate["name"], candidate["rank"]
            expected = min(eligible, key=lambda n: methods[n]["total_seconds"]["median"])
            require(name == expected and candidate == methods[name]["spec"], "Selection violates the prespecified rule")
            expected_names = [*BASELINES, name, f"explicit_balanced{q}", f"cached_warm{q}"]
            require([s["name"] for s in frozen["confirmation_specs"]] == expected_names, "Confirmation methods differ")
            require(frozen["confirmation_specs"][:2] == manifest["configuration"]["specs"][:2], "Baseline settings changed at confirmation")
            require(frozen["confirmation_specs"][2] == candidate, "Selected rank settings changed")
            require(frozen["confirmation_specs"][3] == dict(name=f"explicit_balanced{q}", source="explicit", rank=q, interface="balanced", event_custom=True), "Explicit same-rank control differs")
            require(frozen["confirmation_specs"][4] == dict(name=f"cached_warm{q}", source="cached", rank=q, interface="warm", event_custom=True), "Warm same-rank control differs")
            summary.update(frozen_selection=frozen, frozen_selection_sha256=sha(selection_path))
            for n, expected_rows in [(128, 1500), (256, 600)]:
                key = f"stage5_confirm{n}"
                m, rr, aa = load_online(root, key, splits, frozen)
                require(len(rr) == expected_rows, f"Expected {expected_rows} confirmation rows")
                summary["stages"][key] = aggregate(m, rr)
                summary["audit"][key] = aa
                all_rows.extend(rr)
            finished = read(root / "results/stage5_complete.json")
            require(finished.get("complete") is True and finished.get("selection_sha256") == sha(selection_path), "Final completion marker differs")
            require(len(all_rows) == 3060, "Expected exactly 3060 online rows")
            summary.update(status="complete", completion_sha256=sha(root / "results/stage5_complete.json"))
    if all_rows:
        extractions = [r["extraction"] for r in all_rows if r["extraction"]]
        summary["online_accuracy"] = dict(solves=len(all_rows), pcg_failures=sum(not r["converged"] for r in all_rows),
            recoveries=sum(r["solver_fallback"] for r in all_rows), final_failures=sum(not r["final_converged"] for r in all_rows),
            cache_rejections=sum(r["cache_rejected"] for r in all_rows), max_final_relative_residual=max(r["final_relative_residual"] for r in all_rows),
            capture_rejected_vectors=sum(r["capture_stats"].get("rejected_vectors", 0) for r in all_rows),
            capture_terminal_vectors_skipped=sum(r["capture_stats"].get("skipped_terminal_vectors", 0) for r in all_rows),
            extraction_rejected_vectors=sum(r.get("rejected_vectors", r.get("basis", {}).get("rejected_vectors", 0)) for r in extractions),
            extraction_dependent_vectors=sum(r.get("dependent_vectors", r.get("basis", {}).get("dependent_vectors", 0)) for r in extractions))
    verification = root / "results/stage5_verification.json"
    if verification.is_file():
        record = read(verification)
        require(record["new_test_sha256"] == sha(root / "test_cached_pcg.py"), "Verification test source changed")
        summary["prior_verification"] = dict(record=record, sha256=sha(verification))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    (out / "REPORT_zh.md").write_text(markdown(summary))
    plots(summary, out)
    print(json.dumps(dict(output=str(out), status=summary["status"], audit_passed=summary["cache_audit"]["all_passed"], online_rows=len(all_rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
