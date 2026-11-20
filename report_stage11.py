#!/usr/bin/env python3
"""Independently audit and report the frozen one-step refinement experiment.

This reporter never imports the experiment, PDE solver, or model. It verifies
saved bytes and reconstructs all QoI-derived metrics from the raw values.
Energy and solver residuals can be range-checked, not reconstructed without
the solution fields. Training seeds and timing repetitions are not PDE samples.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("id", "contrast")
KINDS = ("symmetric", "untied", "nonlinear")
SEEDS = (51001, 51002, 51003)
CONTROLS = ("jacobi1", "jacobi8", "amg_v1", "recovery", "geometry_uniform", "oracle_indicator", "fine_direct")
METHODS = KINDS + CONTROLS
LABELS = dict(symmetric="Symmetric", untied="Untied", nonlinear="Nonlinear", jacobi1="Jacobi 1",
              jacobi8="Jacobi 8", amg_v1="AMG V1", recovery="Recovery", geometry_uniform="Geometry only",
              oracle_indicator="Exact indicator*", fine_direct="Fine direct")
BASELINES = ("amg_v1", "jacobi8", "recovery")
METRICS = ("error_reduction_ratio", "energy_error_ratio", "scaled_error_l2", "absolute_error_l2", "median_total_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value}: {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(actual, expected, name, rtol=1e-9, atol=1e-14):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"Recomputed metric disagrees: {name}")


def rowkey(row):
    return row["split"], row["geometry"], row["method"], row["training_seed"]


def stat(values):
    values = list(values)
    valid = [float(v) for v in values if v is not None]
    return dict(median=median(valid) if valid else None, minimum=min(valid) if valid else None,
                maximum=max(valid) if valid else None, valid_count=len(valid), missing_count=len(values)-len(valid))


def guarded_ratio(numerator, denominator):
    return None if numerator is None or denominator is None or denominator <= 0 else numerator/denominator


def added_dofs(cells, nc):
    chosen = set(cells)
    shared = sum((c+1 in chosen and c % nc != nc-1)+(c+nc in chosen) for c in chosen)
    return len(chosen)+shared


def independently_mark(scores, nc, budget):
    """Reconstruct the deterministic selector without importing its code."""
    order = sorted(range(nc*nc), key=lambda x: (-scores[x], x))
    selected = set(); spent = 0
    for cell in order:
        y, x = divmod(cell, nc)
        neighbors = [v for allowed, v in ((x > 0, cell-1), (x+1 < nc, cell+1),
                     (y > 0, cell-nc), (y+1 < nc, cell+nc)) if allowed]
        cost = 1+sum(n in selected for n in neighbors)
        if spent+cost <= budget:
            selected.add(cell); spent += cost
        if spent == budget:
            break
    fallback = spent != budget
    if fallback:
        options = []
        for parity in (0, 1):
            candidate = [c for c in order if sum(divmod(c, nc)) % 2 == parity][:budget]
            require(len(candidate) == budget, "Checkerboard fallback cannot exhaust budget")
            options.append((sum(scores[c] for c in candidate), -parity, candidate))
        selected = set(max(options, key=lambda x: (x[0], x[1]))[2])
    return sorted(selected), fallback


def audit(source):
    frozen = read(source/"freeze.json"); complete = read(source/"complete.json")
    config = frozen["config"]
    require(complete["complete"] and complete["smoke"] == frozen["smoke"], "Incomplete run or smoke mismatch")
    require(sha(source/"freeze.json") == complete["freeze_sha256"], "Frozen manifest changed")
    require(set(complete["output_sha256"]) == {"rows.json", "contexts.json"}, "Incomplete output hashes")
    for name, digest in complete["output_sha256"].items():
        require(sha(source/name) == digest, f"Output changed: {name}")
    for name, digest in frozen["input_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Nonportable input path: {name}")
        require(sha(ROOT/path) == digest, f"Frozen input changed: {name}")
    expected_methods = {(m, s) for m in KINDS for s in SEEDS} | {(m, None) for m in CONTROLS}
    require({tuple(x) for x in frozen["methods"]} == expected_methods and len(frozen["methods"]) == 16, "Wrong method set")
    require(set(config["splits"]) == set(SPLITS), "Wrong split set")
    if not frozen["smoke"]:
        required = dict(nc=16, new_dofs=64, geometries_per_split=16, repetitions=5,
                        splits={"id": 52000, "contrast": 52100}, source=0, goals=[1, 2, 3],
                        accuracy_ratio_gate=.90, minimum_wins=12, seed_ratio_limit=1.10,
                        cheap_baseline_ratio_limit=1., time_ratio_limit=1., paired_denominator_floor=1e-12,
                        near_zero_coarse_error=1e-12)
        for k, v in required.items():
            require(config[k] == v, f"Formal configuration changed: {k}")
    require(frozen["environment"]["device"] == "cpu" and frozen["environment"]["torch_threads"] == 1,
            "Timing environment is not single-thread CPU")
    require(all(p["num_threads"] == 1 for p in frozen["environment"]["threadpools"]), "Numerical-library threading changed")
    # Independently anchor the frozen model records and historical geometry data.
    olddir = ROOT/"results/stage10_pilot"
    oldfreeze = read(olddir/"freeze.json")
    require(not oldfreeze["smoke"], "Weights came from a smoke run")
    for name, digest in oldfreeze["source_sha256"].items():
        require(sha(ROOT/name) == digest, f"Stage-10 source changed: {name}")
    require(sha(ROOT/"docs/stage10_protocol_zh.md") == oldfreeze["protocol_sha256"], "Stage-10 protocol changed")
    model_records = {(r["kind"], r["seed"]): r for r in frozen["models"]}
    require(set(model_records) == {(m,s) for m in KINDS for s in SEEDS}, "Wrong checkpoint record set")
    for key, record in model_records.items():
        train = read(olddir/f"train_{key[0]}_{key[1]}.json")
        require(record["checkpoint"] == train["checkpoint"] and record["sha256"] == train["checkpoint_sha256"],
                "Stage-10 checkpoint provenance changed")
        require(sha(ROOT/record["checkpoint"]) == record["sha256"], "Checkpoint bytes changed")
    historical = set()
    for filename in ("training_data.json", "test_data.json"):
        obj = read(olddir/filename)
        entries = obj if isinstance(obj, list) else obj["train"]+obj["validation"]
        historical.update(r["coefficient_sha256"] for r in entries)
    rows = read(source/"rows.json"); contexts = read(source/"contexts.json")
    require(finite(rows) and finite(contexts) and finite(frozen) and finite(complete), "Nonfinite saved value")
    expected_contexts = {(split, g) for split, first in config["splits"].items()
                         for g in range(first, first+config["geometries_per_split"])}
    expected_keys = {(split, g, m, s) for split, g in expected_contexts for m, s in expected_methods}
    require(len(rows) == len(expected_keys) == complete["rows"] and {rowkey(r) for r in rows} == expected_keys,
            "Duplicate or missing method/seed/geometry rows")
    require(len(contexts) == len(expected_contexts) == complete["contexts"] and
            {(r["split"], r["geometry"]) for r in contexts} == expected_contexts, "Duplicate or missing contexts")
    require(complete["fresh_pipeline_runs"] == len(rows)*config["repetitions"], "Wrong timing repeat count")
    hashes = [r["coefficient_sha256"] for r in contexts]
    require(len(set(hashes)) == len(hashes) and not set(hashes) & historical, "Coefficient geometry leakage/repetition")
    nc = config["nc"]; budget = config["new_dofs"]; threshold = config["near_zero_coarse_error"]
    maxima = dict(solve_relative_residual=0., projection_orthogonality=0., coarse_embedding=0.,
                  energy_ratio=0., timing_unclassified_seconds=0.)
    near_zero = set(); reference = {}; repeat_variability = []; timing_remainders=[]; space_hashes={}
    for context in contexts:
        require(context["nc"] == nc and context["fine_dofs"] == (2*nc-1)**2 and
                context["coarse_dofs"] == (nc-1)**2 and context["new_dofs"] == budget, "Wrong mesh dimensions")
        require(context["fine_relative_residual"] <= 1e-8, "Invalid fine reference residual")
    for row in rows:
        method = row["method"]; key = rowkey(row)
        require(len(row["solution_sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["solution_sha256"]),
                "Invalid solution fingerprint")
        require(row["offline_only"] == (method == "oracle_indicator"), "Offline diagnostic misclassified")
        q, qf, qc, scales = (np.asarray(row[k], dtype=float) for k in ("qoi", "fine_qoi", "coarse_qoi", "qoi_scales"))
        require(all(x.shape == (3,) for x in (q, qf, qc, scales)) and np.all(scales > 0), "Invalid QoI/scales shape")
        context_key = key[:2]
        if context_key in reference:
            for actual, saved in zip((qf,qc,scales), reference[context_key]):
                close(actual, saved, "shared reference and scales")
        else:
            reference[context_key] = (qf,qc,scales)
        expect_scales = np.maximum(abs(qc), max(float(abs(qc).max())*1e-12, 1e-30))
        close(scales, expect_scales, "online coarse QoI scales", atol=1e-25)
        e = qf-q; ec = qf-qc
        scaled = float(np.linalg.norm(e/scales)); coarse = float(np.linalg.norm(ec/scales))
        close(e, row["signed_qoi_errors"], "signed QoI error", atol=1e-20)
        close(ec, row["signed_coarse_errors"], "signed coarse error", atol=1e-20)
        close(np.linalg.norm(e), row["absolute_error_l2"], "absolute error", atol=1e-20)
        close(scaled, row["scaled_error_l2"], "scaled error")
        close(coarse, row["coarse_scaled_error"], "scaled coarse error")
        degenerate = coarse < threshold
        require(row["coarse_error_near_zero"] == degenerate, "Near-zero flag mismatch")
        if degenerate:
            near_zero.add(context_key)
            require(row["error_reduction_ratio"] is None, "Degenerate coarse ratio must be null")
        else:
            close(scaled/coarse, row["error_reduction_ratio"], "actual error reduction")
        require(0 <= row["energy_error_ratio"] <= 1+1e-8 and row["relative_residual"] <= 1e-8,
                "Energy monotonicity or solve residual failed")
        maxima["solve_relative_residual"] = max(maxima["solve_relative_residual"], row["relative_residual"])
        maxima["energy_ratio"] = max(maxima["energy_ratio"], row["energy_error_ratio"])
        if method == "fine_direct":
            require(row["selected_cells"] == [] and row["solve_dofs"] == (2*nc-1)**2 and
                    row["added_dofs"] == (2*nc-1)**2-(nc-1)**2 and not row["budget_matched"],
                    "Fine direct uses unexpected mesh work")
            close(q, qf, "Fine direct/reference", atol=1e-16)
        else:
            cells = row["selected_cells"]
            require(cells == sorted(set(cells)) and all(0 <= c < nc*nc for c in cells), "Invalid selected cells")
            require(added_dofs(cells,nc) == row["added_dofs"] == budget and row["solve_dofs"] == (nc-1)**2+budget,
                    "Unequal added DOF budget")
            require(row["budget_matched"], "Adaptive method is not budget-matched")
            digest=row["interpolation_sha256"]
            require(len(digest)==64 and all(c in "0123456789abcdef" for c in digest), "Invalid interpolation fingerprint")
            if tuple(cells) in space_hashes:
                require(space_hashes[tuple(cells)] == digest, "Identical selected cells have different interpolation fingerprints")
            else:
                space_hashes[tuple(cells)]=digest
            require(row["geometric_triangle_count"] == 2*nc*nc+6*len(cells), "Wrong geometric triangle count")
            scores = row["scores"]
            require(len(scores) == nc*nc and min(scores) >= 0, "Invalid priority scores")
            selected, fallback = independently_mark(scores,nc,budget)
            require(selected == cells and fallback == row["marking"]["fallback"], "Marking rule mismatch")
            require(row["marking"]["selected_cell_count"] == len(cells) and row["marking"]["added_dofs"] == budget,
                    "Marking metadata mismatch")
            close(sum(scores[c] for c in cells), row["marking"]["score_sum"], "Selected priority sum")
            maxima["coarse_embedding"] = max(maxima["coarse_embedding"],row["coarse_embedding_max_abs"])
            require(row["coarse_embedding_max_abs"] <= 1e-12, "Coarse space embedding failed")
            orth = row["projection_orthogonality"]
            if orth is not None:
                maxima["projection_orthogonality"] = max(maxima["projection_orthogonality"],orth)
                require(orth <= 1e-8, "Coarse orthogonality failed")
        times = row["timings"]
        require(len(times) == config["repetitions"] and {t["repetition"] for t in times} == set(range(config["repetitions"])),
                "Missing or duplicate timing repetitions")
        totals = []
        for timing in times:
            require(all(v >= 0 for k,v in timing.items() if k != "repetition"), "Negative timing")
            total = timing["total_seconds"]; totals.append(total)
            components = sum(v for k,v in timing.items() if k.endswith("_seconds") and k not in
                             ("total_seconds", "component_sum_seconds", "unclassified_seconds"))
            require(total+1e-10 >= components, "Timing components exceed total")
            maxima["timing_unclassified_seconds"] = max(maxima["timing_unclassified_seconds"],total-components)
            timing_remainders.append(dict(split=key[0],geometry=key[1],method=key[2],training_seed=key[3],
                repetition=timing["repetition"],component_sum_seconds=components,unclassified_seconds=total-components))
            if "component_sum_seconds" in timing:
                close(components,timing["component_sum_seconds"],"Timing component sum")
            if "unclassified_seconds" in timing:
                close(total-components,timing["unclassified_seconds"],"Timing unclassified remainder")
        close(median(totals),row["median_total_seconds"],"Median fresh pipeline time")
        repeat_variability.append(max(totals)/min(totals))
    audit_result = dict(passed=True, rows=len(rows), contexts=len(contexts), independent_geometries=len(hashes),
        fresh_pipeline_runs=complete["fresh_pipeline_runs"], frozen_input_count=len(frozen["input_sha256"]),
        zero_train_validation_history_overlap=True, exact_dof_budget=True, reconstructed_qoi_metrics=True,
        reconstructed_marking=True, interpolation_fingerprint_consistency=True,
        coarse_denominator_degenerate_contexts=[list(k) for k in sorted(near_zero)],
        numerical_maxima=maxima, within_row_timing_max_over_min=stat(repeat_variability),
        timing_component_sums_and_unclassified=timing_remainders,
        limitation="Energy and solver-residual scalars are range-checked; full solution fields are not stored and are not independently recomputed here.")
    return rows, contexts, frozen, complete, audit_result


def aggregate(rows, frozen, audited):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"],row["geometry"],row["method"])].append(row)
    geometry = []
    for (split, number, method), records in sorted(grouped.items()):
        r = dict(split=split,geometry=number,method=method,training_seed_count=len(records))
        r.update({m:None if any(x[m] is None for x in records) else median(x[m] for x in records) for m in METRICS})
        r["selected_cell_count"] = median(len(x["selected_cells"]) for x in records)
        r["fallback_rows"] = sum(bool(x.get("marking",{}).get("fallback",False)) for x in records)
        r["worse_than_coarse_rows"] = sum(x["error_reduction_ratio"] is not None and x["error_reduction_ratio"] > 1 for x in records)
        r["goal_abs_error_over_coarse"] = [stat(guarded_ratio(abs(x["signed_qoi_errors"][j]),abs(x["signed_coarse_errors"][j])) for x in records)["median"] for j in range(3)]
        timing_keys = sorted(set.intersection(*(set(x["timings"][0]) for x in records))-{ "repetition" })
        r["timing_components_seconds"] = {k:median(median(t[k] for t in x["timings"]) for x in records) for k in timing_keys}
        r["preparation_seconds"] = median(median(t["geometry_assembly_seconds"]+t.get("coarse_factor_solve_seconds",0.) for t in x["timings"]) for x in records)
        r["after_preparation_seconds"] = median(median(t["total_seconds"]-t["geometry_assembly_seconds"]-t.get("coarse_factor_solve_seconds",0.) for t in x["timings"]) for x in records)
        geometry.append(r)
    lookup = {(r["split"],r["geometry"],r["method"]):r for r in geometry}
    threshold = frozen["config"]["paired_denominator_floor"]
    degenerate_baselines=[]; totals={}; per_seed={}; pairs=[]
    for split in SPLITS:
        totals[split]={};per_seed[split]={}
        numbers = sorted({r["geometry"] for r in geometry if r["split"]==split})
        for method in METHODS:
            entries=[lookup[(split,g,method)] for g in numbers]
            raw=[r for r in rows if r["split"]==split and r["method"]==method]
            totals[split][method]={m:stat(r[m] for r in entries) for m in METRICS}
            totals[split][method].update(selected_cell_count=stat(r["selected_cell_count"] for r in entries),
                raw_selected_cell_count=stat(len(r["selected_cells"]) for r in raw),
                fallback_rows=sum(r["fallback_rows"] for r in entries),raw_rows=len(raw),
                worse_than_coarse_rows=sum(r["worse_than_coarse_rows"] for r in entries),
                seed_median_worse_than_coarse_geometries=sum(r["error_reduction_ratio"] is not None and r["error_reduction_ratio"]>1 for r in entries),
                preparation_seconds=stat(r["preparation_seconds"] for r in entries),
                after_preparation_seconds=stat(r["after_preparation_seconds"] for r in entries),
                timing_components_seconds={k:stat(r["timing_components_seconds"][k] for r in entries) for k in entries[0]["timing_components_seconds"]},
                goal_abs_error_over_coarse=[stat(r["goal_abs_error_over_coarse"][j] for r in entries) for j in range(3)], paired={})
            for baseline in (*BASELINES,"fine_direct"):
                ratios=[];time_ratios=[];wins=0
                for r in entries:
                    b=lookup[(split,r["geometry"],baseline)]
                    degenerate=b["scaled_error_l2"]<threshold
                    value=None if baseline=="fine_direct" or degenerate else guarded_ratio(r["error_reduction_ratio"],b["error_reduction_ratio"])
                    tr=r["median_total_seconds"]/b["median_total_seconds"]
                    ratios.append(value);time_ratios.append(tr)
                    wins+=int(value is not None and value<1)
                    if degenerate and baseline!="fine_direct":
                        item=[split,r["geometry"],baseline]
                        if item not in degenerate_baselines:degenerate_baselines.append(item)
                    pairs.append(dict(split=split,geometry=r["geometry"],method=method,baseline=baseline,
                        error_ratio=value,time_ratio=tr,baseline_error_degenerate=bool(degenerate)))
                totals[split][method]["paired"][baseline]=dict(error_ratio=stat(ratios),time_ratio=stat(time_ratios),strict_error_wins=wins)
        for method in KINDS:
            per_seed[split][method]={}
            for seed in SEEDS:
                records=[r for r in rows if r["split"]==split and r["method"]==method and r["training_seed"]==seed]
                paired={}
                for baseline in BASELINES:
                    ratios=[]
                    for r in records:
                        b=lookup[(split,r["geometry"],baseline)]
                        ratios.append(None if b["scaled_error_l2"]<threshold else guarded_ratio(r["error_reduction_ratio"],b["error_reduction_ratio"]))
                    paired[baseline]=dict(error_ratio=stat(ratios),strict_error_wins=sum(x is not None and x<1 for x in ratios))
                per_seed[split][method][str(seed)]={**{m:stat(r[m] for r in records) for m in METRICS},"paired":paired}
    # Mesh overlap is descriptive; the three separately chosen seed meshes remain separate.
    overlap_summary={};oracle_overlap={}
    raw_lookup={rowkey(r):r for r in rows}
    for split in SPLITS:
        overlap_summary[split]={};oracle_overlap[split]={}
        for method in KINDS:
            values=[];oracle_values=[]
            for r in rows:
                if r["split"]!=split or r["method"]!=method:continue
                a=set(r["selected_cells"]);b=set(raw_lookup[(split,r["geometry"],"amg_v1",None)]["selected_cells"])
                values.append(len(a&b)/len(a|b))
                b=set(raw_lookup[(split,r["geometry"],"oracle_indicator",None)]["selected_cells"])
                oracle_values.append(len(a&b)/len(a|b))
            overlap_summary[split][method]=stat(values)
            oracle_overlap[split][method]=stat(oracle_values)
    config=frozen["config"]; criteria={}
    enough=not audited["coarse_denominator_degenerate_contexts"] and not any(x[2] in ("amg_v1","jacobi8") for x in degenerate_baselines)
    for split in SPLITS:
        symmetric=totals[split]["symmetric"];amg=symmetric["paired"]["amg_v1"];cheap=symmetric["paired"]["jacobi8"]
        def below(record,limit):return record["missing_count"]==0 and record["median"] is not None and record["median"]<=limit
        criteria[split]=dict(amg_accuracy=below(amg["error_ratio"],config["accuracy_ratio_gate"]),
            amg_wins=amg["strict_error_wins"]>=config["minimum_wins"],
            every_seed=all(below(per_seed[split]["symmetric"][str(seed)]["paired"]["amg_v1"]["error_ratio"],config["seed_ratio_limit"]) for seed in SEEDS),
            cheap_accuracy=below(cheap["error_ratio"],config["cheap_baseline_ratio_limit"]),
            complete_cost=below(amg["time_ratio"],config["time_ratio_limit"]))
    accuracy=enough and all(v[k] for v in criteria.values() for k in ("amg_accuracy","amg_wins","every_seed","cheap_accuracy"))
    efficiency=accuracy and all(v["complete_cost"] for v in criteria.values())
    decision=dict(formal=not frozen["smoke"], numerical_audit_passed=audited["passed"],
        nondegenerate_required_denominators=enough,criteria=criteria,accuracy_gate_passed=accuracy,
        complete_efficiency_gate_passed=efficiency,continue_current_amr_branch=efficiency and not frozen["smoke"],
        note="Smoke results never establish a formal conclusion." if frozen["smoke"] else
             ("Continue only the frozen primary candidate; this is a development gate, not a significance test." if efficiency else
              "Current AI local-marking AMR branch fails its prespecified continuation gate; do not enlarge training to chase this test set."))
    return dict(aggregation="Within each geometry median of three training seeds, then across independent geometries; all ratios paired before split median. Timing is median of fresh repetitions before seed aggregation.",
        totals=totals,per_geometry=geometry,paired_geometry=pairs,per_training_seed=per_seed,
        baseline_denominator_degenerate_contexts=degenerate_baselines,baseline_scaled_error_threshold=threshold,
        selected_cell_jaccard_against_amg=overlap_summary,selected_cell_jaccard_against_exact_indicator=oracle_overlap,
        mesh_overlap_analysis_scope="Post-hoc descriptive mechanism probe; raw training-seed rows, not independent geometries; never part of the continuation gate.",decision=decision)


def fmt(value, digits=4):
    return "未定义" if value is None else f"{value:.{digits}g}"


def report_text(summary, frozen, complete, audited, source):
    totals=summary["totals"];decision=summary["decision"];config=frozen["config"]
    lines=["# 第十一阶段：相同新增自由度的一次真实局部细化", "",
        "本报告由独立程序读取冻结结果生成，未导入实验运行器，也未重新训练或重跑求解。",
        "",f"正式实验：{'否（smoke）' if frozen['smoke'] else '是'}；独立几何 **{audited['contexts']}**；方法/种子结果 **{audited['rows']}**；每项 **{config['repetitions']} 次**全新问题对象暖运行，共 **{complete['fresh_pipeline_runs']}** 次。",
        f"每个适应网格恰新增 **{config['new_dofs']}** 个内部自由度；粗/适应/细参考维数为 **{(config['nc']-1)**2}/{(config['nc']-1)**2+config['new_dofs']}/{(2*config['nc']-1)**2}**。三个目标共用同一个细化网格和一次主问题求解。",
        "", "## 事前门槛", "",
        f"准确性门槛：**{'通过' if decision['accuracy_gate_passed'] else '未通过'}**；准确性与完整成本联合门槛：**{'通过' if decision['complete_efficiency_gate_passed'] else '未通过'}**。",
        "对称模型是事前主候选，其他网络只作为消融。失败后不把表现更好的消融临时换成主候选，也不增加训练量追逐本测试集。10% 是开发决策门槛，不是显著性检验或论文录用条件。",
        "", "| 测试组 | S/AMG 配对 E 比中位数 | 严格改善数 | S/Jacobi8 配对 E 比 | S/AMG 完整时间比 | 准确性/成本门槛 |",
        "|---|---:|---:|---:|---:|---|" ]
    for split in SPLITS:
        s=totals[split]["symmetric"];a=s["paired"]["amg_v1"];c=decision["criteria"][split]
        lines.append(f"| {split} | {fmt(a['error_ratio']['median'],6)} | {a['strict_error_wins']}/{config['geometries_per_split']} | {fmt(s['paired']['jacobi8']['error_ratio']['median'],6)} | {fmt(a['time_ratio']['median'],6)} | {all(c[k] for k in ('amg_accuracy','amg_wins','every_seed','cheap_accuracy'))} / {c['complete_cost']} |")
    lines += ["", "每个配对比先在同一几何上计算，再跨几何取中位数；不能用下面两列跨几何中位数相除替代。三个训练种子先在同一几何上取中位数，不是 48 个独立 PDE 样本。", "", "## 全部方法的真实解误差与完整成本", "",
        "E 是三个目标误差经粗目标值归一化后的二范数，相对粗解该范数的比例；粗解 E=1。小于 1 才代表实际目标量改善，能量误差降低不保证每个目标量都改善。所有误差只相对指定细离散参考解。",
        "", "| 方法 | ID E | 高对比 E | ID 能量误差比 | 高对比能量误差比 | ID 完整耗时 ms | 高对比完整耗时 ms |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        a=totals["id"][method];b=totals["contrast"][method]
        lines.append(f"| {LABELS[method]} | {fmt(a['error_reduction_ratio']['median'])} | {fmt(b['error_reduction_ratio']['median'])} | {fmt(a['energy_error_ratio']['median'])} | {fmt(b['energy_error_ratio']['median'])} | {fmt(1000*a['median_total_seconds']['median'])} | {fmt(1000*b['median_total_seconds']['median'])} |")
    lines += ["", "Exact indicator* 使用三个精确细伴随缺陷，是离线诊断；其分解与求解成本按实际计入。它采用同一固定预算选择器，不是实际目标误差的全局最优网格。Fine direct 只做一次细主问题求解便回答全部三个目标，误差为零，绝不作为误差比的分母。", "", "## 每个训练种子的配对结果", "",
        "| 测试组 | 模型 | 训练种子 | /AMG E 比 | /Jacobi8 E 比 | /Recovery E 比 |", "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in KINDS:
            for seed in SEEDS:
                p=summary["per_training_seed"][split][method][str(seed)]["paired"]
                lines.append(f"| {split} | {LABELS[method]} | {seed} | "+" | ".join(fmt(p[b]["error_ratio"]["median"],6) for b in BASELINES)+" |")
    lines += ["", "## 标记预算、网格与目标误差退化", "",
        "表内方格数是逐几何种子中位数的范围；回退和 E>1 均按原始方法/种子行计数，分母含训练种子，不能当作独立几何比例。", "",
        "| 测试组 | 方法 | 选中方格数范围 | 棋盘回退行数 | E>1 行数 | 完整时间/Fine direct 配对中位数 |", "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in METHODS:
            v=totals[split][method];n=v["selected_cell_count"]
            lines.append(f"| {split} | {LABELS[method]} | {fmt(n['minimum'])}–{fmt(n['maximum'])} | {v['fallback_rows']}/{v['raw_rows']} | {v['worse_than_coarse_rows']}/{v['raw_rows']} | {fmt(v['paired']['fine_direct']['time_ratio']['median'])} |")
    lines += ["",f"粗误差分母退化几何：**{len(audited['coarse_denominator_degenerate_contexts'])}**；经典分母退化记录：**{len(summary['baseline_denominator_degenerate_contexts'])}**。记录完整保留，不补抽、不悄悄删除；所需分母退化会使正式门槛判为证据不足。", "",
        "各目标的误差比例、学习方法与 AMG/精确指标的网格 Jaccard 重合度、全部逐几何配对值及范围均保存在 summary.json。网格重合是事后描述性机制诊断，不进入过线标准；原始训练种子行不作为独立几何。网格重合或单一诊断不能确定失败的唯一原因。", "", "## 完整计时与复核边界", "",
        "CPU 单线程；每次都重新生成系数、装配、构造自己的粗层和预条件器/恢复算子，执行预测、投影、指标、预算选择、适应空间构造、矩阵分解与主问题求解。跨方法不借用设置缓存。三目标可在同一次方法运行内共享分解。",
        "进程内模型载入、库导入、暖机、离线参考求解、结果核验和序列化不计入每几何在线时间；模型载入单列，训练不在本阶段发生。每个几何先取五次暖运行中位数，随后才做种子及几何聚合。没有 GPU 推理，也不据此外推其他硬件或规模。",
        f"九个模型载入合计 **{sum(r['seconds'] for r in frozen['models']):.6g} s**；所有分支暖机 **{complete['warmup_seconds']:.6g} s**；实验墙钟 **{complete['measured_experiment_wall_seconds']:.6g} s**。墙钟包含非在线参考、核验及序列化，与在线时间之和不是同一口径。", "",
        "| 测试组 | 方法 | 装配+粗求解 ms | 此后完整路径 ms |", "|---|---|---:|---:|"]
    for split in SPLITS:
        for method in ("symmetric","jacobi8","amg_v1","recovery","fine_direct"):
            v=totals[split][method]
            lines.append(f"| {split} | {LABELS[method]} | {fmt(1000*v['preparation_seconds']['median'])} | {fmt(1000*v['after_preparation_seconds']['median'])} |")
    lines += ["", "两个分量各自取中位数，因此不要求相加恰好等于总耗时中位数。逐重复阶段分量在原始结果，独立重算的分量之和及未归类差额保存在 summary.json 的 audit 字段；报告不把网络前向时间当作完整求解成本。", "",
        f"独立复核通过：**{audited['frozen_input_count']} 个冻结输入哈希**、全部结果哈希、模型权重来源、历史几何哈希无重叠、全部 3 目标误差重建、精确新增自由度、标记规则重建、五次计时中位数。",
        f"最大求解相对残差 **{audited['numerical_maxima']['solve_relative_residual']:.3e}**；最大投影粗正交残差 **{audited['numerical_maxima']['projection_orthogonality']:.3e}**；最大粗空间嵌入差 **{audited['numerical_maxima']['coarse_embedding']:.3e}**。",
        "独立报告程序未重跑 PDE：能量与残差保存值核验范围，完整解向量未持久化，不能声称在报告阶段独立重算能量或解字段。实际求解与重复一致性由冻结运行器及数值单元测试核验。", "",
        "本轮只是一轮、固定网格层级、固定三个目标、固定自由度预算的检验，不证明多轮 AMR 收敛、连续误差认证、跨尺度泛化、模型新颖性或论文可发表。", "",
        "![Actual refinement error and full cost](figures/comparison.png)", "",
        f"原始结果目录：`{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`；完整逐几何聚合见 `summary.json`。", ""]
    return "\n".join(lines)


def plot(summary, output):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/darcy-u-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={"id":"#2478a9","contrast":"#d57725"};offset={"id":-.16,"contrast":.16}
    fig,axes=plt.subplots(2,2,figsize=(13.7,10.5),gridspec_kw={"width_ratios":[1.15,1]})
    xs=np.arange(len(METHODS))
    for split in SPLITS:
        records=summary["totals"][split]
        for ax,metric,factor in ((axes[0,0],"error_reduction_ratio",1),(axes[0,1],"median_total_seconds",1000)):
            vals=[records[m][metric]["median"] for m in METHODS]
            ax.plot(xs+offset[split],[np.nan if v is None else factor*v for v in vals],"o",color=colors[split],label=split.upper(),ms=6)
            ax.set_xticks(xs);ax.set_xticklabels([LABELS[m] for m in METHODS],rotation=52,ha="right",fontsize=9)
            ax.grid(axis="y",alpha=.22)
        for ax,ratio,title in ((axes[1,0],"error_ratio","C  Paired actual-error ratio to AMG"),(axes[1,1],"time_ratio","D  Paired complete-time ratio to AMG")):
            for i,method in enumerate(KINDS):
                values=[r[ratio] for r in summary["paired_geometry"] if r["split"]==split and r["method"]==method and r["baseline"]=="amg_v1" and r[ratio] is not None]
                x=i+offset[split]
                jitter=np.linspace(-.055,.055,len(values))
                ax.scatter(x+jitter,values,s=17,alpha=.45,color=colors[split],edgecolors="none")
                if values:ax.plot([x-.09,x+.09],[median(values)]*2,color=colors[split],lw=3)
            ax.axhline(1,color="#555",ls="--",lw=1)
            ax.set_xticks(range(3));ax.set_xticklabels([LABELS[m] for m in KINDS]);ax.set_title(title,loc="left",fontweight="bold")
            ax.grid(axis="y",alpha=.22)
    axes[0,0].axhline(1,color="#555",ls="--",lw=1);axes[0,0].set_title("A  Actual QoI error after one local refinement",loc="left",fontweight="bold")
    axes[0,0].set_ylabel("Scaled three-QoI error / coarse error (lower is better)")
    axes[0,1].set_title("B  Complete online cost",loc="left",fontweight="bold");axes[0,1].set_ylabel("Median fresh pipeline time (ms)")
    axes[1,0].axhline(.9,color="#777",ls=":",lw=1);axes[1,0].set_ylabel("Geometry-paired error ratio (lower is better)")
    axes[1,1].set_ylabel("Geometry-paired time ratio (lower is better)")
    handles=[Line2D([0],[0],marker="o",linestyle="",color=colors[s],label="ID" if s=="id" else "High contrast") for s in SPLITS]
    fig.legend(handles=handles,loc="upper center",ncol=2,bbox_to_anchor=(.5,.953),frameon=False)
    fig.suptitle("Frozen error operators: equal-DOF, one-step local refinement",fontsize=16,y=.99)
    fig.text(.5,.015,"Each dot in C/D is one independent geometry after median of 3 training seeds.  *Exact indicator is offline, not an optimal mesh.\nFull cost includes assembly, setup, prediction, projection, marking and adaptive solve. Fine direct uses one primal solve for all goals.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.095,top=.90,hspace=.65,wspace=.30)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):fig.savefig(output/f"comparison.{ext}",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",default="results/stage11_refinement")
    parser.add_argument("--output",default="results/stage11_report");parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args();source=(ROOT/args.source).resolve();output=(ROOT/args.output).resolve()
    rows,contexts,frozen,complete,audited=audit(source)
    summary=aggregate(rows,frozen,audited);summary["audit"]=audited
    summary["provenance"]=dict(source=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        freeze_sha256=sha(source/"freeze.json"),complete_sha256=sha(source/"complete.json"),reporter_sha256=sha(__file__))
    output.mkdir(parents=True,exist_ok=True)
    dump(output/"summary.json",summary)
    (output/"REPORT_zh.md").write_text(report_text(summary,frozen,complete,audited,source))
    if not args.no_plot:plot(summary,output/"figures")
    files=[p for p in output.rglob("*") if p.is_file() and p.name!="complete.json"]
    dump(output/"complete.json",dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        reporter_sha256=sha(__file__),source_freeze_sha256=sha(source/"freeze.json"),source_complete_sha256=sha(source/"complete.json"),
        output_sha256={str(p.relative_to(output)):sha(p) for p in sorted(files)}))
    compact_audit={k:v for k,v in audited.items() if k!="timing_component_sums_and_unclassified"}
    print(json.dumps(dict(output=str(output),audit=compact_audit,decision=summary["decision"]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
