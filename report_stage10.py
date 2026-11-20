#!/usr/bin/env python3
"""Independently audit and aggregate stage 10 without importing its runner.

Only saved JSON, checkpoint bytes and frozen source bytes are read.  No PDE,
network inference, training, or benchmark is rerun.  NumPy is used only for
small saved 4-by-4 matrices and descriptive aggregation.
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
SPLITS = ("id", "contrast", "narrow_load", "resolution")
LEARNED = ("symmetric", "untied", "nonlinear")
CLASSICAL = ("jacobi1", "jacobi8", "amg_v1")
METHODS = LEARNED + CLASSICAL
SEEDS = (51001, 51002, 51003)
WEIGHTS = np.array([[1., -1., 0., 0.], [.5, .5, -.5, -.5],
                    [-3., 0., 0., 0.], [0., 10., -5., 1.]])
METRICS = (
    "matrix_error_relative", "normalized_matrix_rmse", "defect_energy_error_mean",
    "reciprocity_defect", "minimum_symmetric_eigenvalue",
    "relative_minimum_symmetric_eigenvalue", "marker_mass_ratio", "indicator_relative_l1",
    "mixture_linearity_defect", "mixture_cross_normalized_rmse",
    "mixture_matrix_error_relative", "mixed_symmetric_eigenvalue_relative")
COMBINATION_METRICS = ("mixture_linearity_by_combination",
                       "mixture_cross_normalized_rmse_by_combination",
                       "mixture_diagonal_relative_error")
LABELS = {"id": "ID", "contrast": "高对比", "narrow_load": "窄源项–目标载荷",
          "resolution": "更细网格"}
METHOD_LABELS = {"symmetric": "Symmetric", "untied": "Untied", "nonlinear": "Nonlinear",
                 "jacobi1": "Jacobi 1", "jacobi8": "Jacobi 8", "amg_v1": "AMG V-cycle"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, wanted, name, rtol=2e-8, atol=1e-13):
    require(np.allclose(actual, wanted, rtol=rtol, atol=atol),
            f"Saved matrix-derived metric mismatch: {name}")


def finite_tree(value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite value: {label}")


def matrix(value, name):
    array = np.asarray(value, dtype=float)
    require(array.shape == (4, 4) and np.isfinite(array).all(), f"Invalid matrix: {name}")
    return array


def audit_matrix_metrics(row):
    """Reconstruct every scalar available from the saved small matrices."""
    e = matrix(row["reference_matrix"], "reference")
    a = matrix(row["estimated_matrix"], "estimate")
    w = matrix(row["mixture_weights"], "weights")
    exact = matrix(row["mixture_reference_matrix"], "mixture reference")
    mixed = matrix(row["mixture_estimated_matrix"], "mixture estimate")
    cross = matrix(row["mixture_cross_estimated"], "cross estimate")
    cross_exact = matrix(row["mixture_cross_reference"], "cross reference")
    require(np.array_equal(w, WEIGHTS), "Mixture weights changed")
    require(np.all(e.diagonal() > 0) and np.all(exact.diagonal() > 0),
            "Nonpositive reference goal energy")
    scale_e = np.linalg.norm(e, 2)
    require(np.linalg.eigvalsh((e+e.T)/2).min() >= -1e-10*scale_e,
            "Reference error matrix is not PSD within the frozen tolerance")
    require(np.linalg.norm(e-e.T)/np.linalg.norm(e) <= 1e-9,
            "Reference reciprocity tolerance exceeded")
    close(exact, w @ e @ w.T, "mixture reference", atol=1e-18)
    close(cross_exact, e @ w.T, "cross reference", atol=1e-18)
    close(mixed, w @ cross, "mixture/cross consistency", atol=1e-18)
    eig = np.linalg.eigvalsh((a+a.T)/2)
    normalized = (a-e)**2/np.outer(e.diagonal(), e.diagonal())
    derived = dict(
        matrix_error_relative=np.linalg.norm(a-e)/np.linalg.norm(e),
        normalized_matrix_rmse=np.sqrt(np.mean(normalized)),
        reciprocity_defect=np.linalg.norm(a-a.T)/max(np.linalg.norm(a), 1e-30),
        minimum_symmetric_eigenvalue=eig.min(),
        relative_minimum_symmetric_eigenvalue=eig.min()/max(np.linalg.norm(a, 2), 1e-30),
        mixture_cross_normalized_rmse=np.sqrt(np.mean(
            (cross-cross_exact)**2/np.outer(e.diagonal(), exact.diagonal()))),
        mixture_matrix_error_relative=np.linalg.norm(mixed-exact)/np.linalg.norm(exact),
        mixed_symmetric_eigenvalue_relative=np.linalg.eigvalsh((mixed+mixed.T)/2).min()
            / max(np.linalg.norm(mixed, 2), 1e-30))
    for name, value in derived.items():
        close(row[name], value, name)
    close(row["mixture_cross_normalized_rmse_by_combination"],
          np.sqrt(np.mean((cross-cross_exact)**2/np.outer(e.diagonal(), exact.diagonal()), axis=0)),
          "per-combination normalized error")
    close(row["mixture_diagonal_relative_error"],
          abs(mixed.diagonal()-exact.diagonal())/exact.diagonal(), "mixed diagonal errors")
    require(row["diagonal_positive"] == bool(np.all(a.diagonal() > 0)), "Diagonal sign flag differs")
    require(-1e-12 <= row["marker_mass_ratio"] <= 1.+1e-12, "Invalid marker mass ratio")
    require(row["defect_energy_error_mean"] >= -1e-12 and row["indicator_relative_l1"] >= 0,
            "Invalid recorded energy or local indicator error")
    for name in COMBINATION_METRICS:
        require(len(row[name]) == 4 and min(row[name]) >= 0, f"Invalid combination metric {name}")
    require(row["mixture_linearity_defect"] >= 0, "Negative linearity defect")
    return e


def audit(source):
    complete = read(source/"complete.json")
    frozen = read(source/"freeze.json")
    require(complete["complete"] and not frozen["smoke"], "Only a completed formal run is supported")
    require(complete["freeze_sha256"] == sha(source/"freeze.json"), "Freeze hash changed")
    require(complete["source_sha256"] == frozen["source_sha256"], "Source hashes changed during the run")
    for name, digest in frozen["source_sha256"].items():
        require(sha(ROOT/name) == digest, f"Current frozen source changed: {name}")
    require(sha(ROOT/"docs/stage10_protocol_zh.md") == frozen["protocol_sha256"], "Protocol changed")
    expected_config = dict(train_geometries=64, validation_geometries=12, test_geometries=16,
                           goals=4, coarse_cells=16, epochs=60, seeds=list(SEEDS), kinds=list(LEARNED))
    require(all(frozen["config"][k] == value for k, value in expected_config.items()), "Unexpected formal configuration")
    required_results = {"evaluation.json", "test_data.json", "training_data.json", "training_complete.json"}
    require(set(complete["result_sha256"]) == required_results, "Result hash manifest is incomplete")
    for name, digest in complete["result_sha256"].items():
        require(sha(source/name) == digest, f"Result hash changed: {name}")

    training_complete = read(source/"training_complete.json")
    checkpoints = training_complete["checkpoints"]
    expected_models = {(kind, seed) for kind in LEARNED for seed in SEEDS}
    require(len(checkpoints) == 9 == complete["models"], "Expected nine checkpoints")
    require({(x["kind"], x["seed"]) for x in checkpoints} == expected_models, "Checkpoint combinations differ")
    expected_training_files = {f"train_{kind}_{seed}.json" for kind, seed in expected_models}
    require({p.name for p in source.glob("train_*.json")} == expected_training_files, "Training logs differ")
    training = []
    for item in checkpoints:
        kind, seed = item["kind"], item["seed"]
        record = read(source/f"train_{kind}_{seed}.json")
        finite_tree(record, "training")
        require((record["kind"], record["seed"]) == (kind, seed), "Training metadata mismatch")
        filename = f"{kind}_{seed}.pt"
        require(Path(record["checkpoint"]).name == filename, "Unexpected checkpoint filename")
        require(sha(source/filename) == record["checkpoint_sha256"] == item["sha256"], "Checkpoint hash mismatch")
        history = record["history"]
        require([r["epoch"] for r in history] == list(range(1, 61)), "Expected all 60 training epochs")
        best = min(history, key=lambda r: r["validation_loss"])
        require(best["epoch"] == record["best_epoch"] and best["validation_loss"] == record["validation_loss"],
                "Checkpoint was not selected by the recorded validation minimum")
        training.append(record)
    close(complete["training_seconds"], sum(r["training_seconds"] for r in training), "training seconds")

    train_data, test_data = read(source/"training_data.json"), read(source/"test_data.json")
    for name, wanted in [("train", set(range(50000, 50064))), ("validation", set(range(50100, 50112)))]:
        records = train_data[name]
        require(len(records) == len(wanted) and {r["seed"] for r in records} == wanted, f"Unexpected {name} split")
        require(all(r["nc"] == 16 and not r["ood"] and not r["narrow"] for r in records), f"Wrong {name} configuration")
    expected_cases = {(split, case) for split in SPLITS
                      for case in range(50300 if split == "contrast" else 50200,
                                        50316 if split == "contrast" else 50216)}
    require(len(test_data) == 64 and {(r["split"], r["seed"]) for r in test_data} == expected_cases,
            "Test manifest combinations differ")
    for r in test_data:
        require(r["nc"] == (32 if r["split"] == "resolution" else 16)
                and r["ood"] == (r["split"] == "contrast")
                and r["narrow"] == (r["split"] == "narrow_load"), "Wrong test case configuration")
    all_data = train_data["train"]+train_data["validation"]+test_data
    for r in all_data:
        finite_tree(r, "reference diagnostics")
        require(r["reference_symmetry"] <= 1e-9 and r["max_galerkin_residual"] <= 1e-8
                and r["max_reference_relative_residual"] <= 1e-8, "Reference quality gate failed")
    seen = [r["coefficient_sha256"] for r in train_data["train"]+train_data["validation"]]
    require(len(set(seen)) == len(seen), "Training/validation coefficient collision")
    require(not (set(seen) & {r["coefficient_sha256"] for r in test_data}), "Train/test coefficient collision")
    test_lookup = {(r["split"], r["seed"]): r for r in test_data}
    for case in range(50200, 50216):
        require(test_lookup["id", case]["coefficient_sha256"] == test_lookup["narrow_load", case]["coefficient_sha256"],
                "Narrow loads must retain the ID coefficient")

    rows = read(source/"evaluation.json")
    expected_rows = {(split, case, method, seed) for split, case in expected_cases for method in METHODS
                     for seed in (SEEDS if method in LEARNED else (None,))}
    actual_rows = [(r["split"], r["geometry"], r["method"], r["training_seed"]) for r in rows]
    require(len(rows) == 768 == complete["rows"] and len(set(actual_rows)) == len(rows)
            and set(actual_rows) == expected_rows, "Expected exactly 768 unique evaluation combinations")
    references = {}
    for row in rows:
        finite_tree(row, "evaluation")
        e = audit_matrix_metrics(row)
        key = (row["split"], row["geometry"])
        if key in references:
            require(np.array_equal(references[key], e), "Methods do not share the same reference matrix")
        else:
            references[key] = e
            close(test_lookup[key]["minimum_reference_eigenvalue"], np.linalg.eigvalsh((e+e.T)/2).min(),
                  "manifest reference eigenvalue", atol=1e-18)
    audit_record = dict(passed=True, source_files=len(frozen["source_sha256"]), checkpoints=9,
        evaluation_rows=768, learned_rows=576, classical_rows=192, test_cases=64,
        independent_test_geometry_seeds=32, train_geometries=64, validation_geometries=12,
        max_reference_reciprocity=max(r["reference_symmetry"] for r in all_data),
        max_galerkin_residual=max(r["max_galerkin_residual"] for r in all_data),
        max_reference_relative_residual=max(r["max_reference_relative_residual"] for r in all_data),
        matrix_derived_metrics_recomputed=True,
        scope="Recomputed saved 4x4 matrix metrics and hashes; field energy, local marking and vector-linearity metrics are checked for finite values and valid ranges, not recomputed from absent solution arrays.")
    return rows, training, train_data, test_data, frozen, complete, audit_record


def describe(values):
    values = list(values)
    return dict(median=float(median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), minimum=float(min(values)), maximum=float(max(values)))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["geometry"], row["method"]].append(row)
    by_split, paired = {}, {}
    for split in SPLITS:
        cases = sorted({r["geometry"] for r in rows if r["split"] == split})
        by_split[split] = {}
        for method in METHODS:
            points = []
            for case in cases:
                group = groups[split, case, method]
                points.append(dict(geometry=case,
                    metrics={name: float(median(r[name] for r in group)) for name in METRICS},
                    combination_metrics={name: [float(median(r[name][j] for r in group)) for j in range(4)]
                                         for name in COMBINATION_METRICS}))
            metrics = {name: describe(p["metrics"][name] for p in points) for name in METRICS}
            per_seed = {}
            if method in LEARNED:
                for seed in SEEDS:
                    selected = [r for r in rows if r["split"] == split and r["method"] == method and r["training_seed"] == seed]
                    per_seed[str(seed)] = dict(
                        metrics={name: describe(r[name] for r in selected) for name in METRICS},
                        geometry_points=[dict(geometry=r["geometry"], metrics={name: r[name] for name in METRICS},
                            combination_metrics={name: r[name] for name in COMBINATION_METRICS})
                            for r in sorted(selected, key=lambda x: x["geometry"])])
            by_split[split][method] = dict(metrics=metrics,
                medians={name: value["median"] for name, value in metrics.items()}, geometry_points=points,
                combination_metrics={name: [describe(p["combination_metrics"][name][j] for p in points) for j in range(4)]
                                     for name in COMBINATION_METRICS}, per_seed=per_seed)
        paired[split] = {}
        for other in ("untied", "nonlinear", "amg_v1"):
            values, seed_values = [], defaultdict(list)
            for case in cases:
                first = median(r["matrix_error_relative"] for r in groups[split, case, "symmetric"])
                second = median(r["matrix_error_relative"] for r in groups[split, case, other])
                require(second > 0, "A zero comparator error requires explicit ratio handling")
                values.append(dict(geometry=case, ratio=float(first/second)))
                for seed in SEEDS:
                    ss = next(r["matrix_error_relative"] for r in groups[split, case, "symmetric"] if r["training_seed"] == seed)
                    rr = next(r["matrix_error_relative"] for r in groups[split, case, other]
                              if r["training_seed"] == (seed if other in LEARNED else None))
                    require(rr > 0, "A zero seed comparator error requires explicit ratio handling")
                    seed_values[str(seed)].append(dict(geometry=case, ratio=float(ss/rr)))
            paired[split][f"symmetric_over_{other}"] = dict(**describe(p["ratio"] for p in values),
                lower_error_geometries=sum(p["ratio"] < 1 for p in values), geometry_points=values,
                per_seed={seed: dict(**describe(p["ratio"] for p in points), geometry_points=points)
                          for seed, points in seed_values.items()})
    symmetric = [r for r in rows if r["method"] == "symmetric"]
    structure = dict(
        maximum_reciprocity_defect=max(r["reciprocity_defect"] for r in symmetric),
        maximum_mixture_linearity_defect=max(r["mixture_linearity_defect"] for r in symmetric),
        maximum_per_combination_linearity_defect=max(max(r["mixture_linearity_by_combination"]) for r in symmetric),
        minimum_relative_eigenvalue=min(r["relative_minimum_symmetric_eigenvalue"] for r in symmetric),
        minimum_mixed_relative_eigenvalue=min(r["mixed_symmetric_eigenvalue_relative"] for r in symmetric),
        all_positive_diagonals=all(r["diagonal_positive"] for r in symmetric))
    structure["passed"] = bool(structure["maximum_reciprocity_defect"] <= 1e-5
        and max(structure["maximum_mixture_linearity_defect"], structure["maximum_per_combination_linearity_defect"]) <= 1e-5
        and min(structure["minimum_relative_eigenvalue"], structure["minimum_mixed_relative_eigenvalue"]) >= -1e-5)
    ratios = {split: paired[split]["symmetric_over_untied"]["median"] for split in SPLITS}
    clauses = dict(id_at_most_0_9=ratios["id"] <= .9,
        at_least_one_ood_at_most_0_9=any(ratios[s] <= .9 for s in SPLITS[1:]),
        all_other_ood_at_most_1_1=all(ratios[s] <= 1.1 for s in SPLITS[1:]))
    accuracy = all(clauses.values())
    decision = dict(symmetric_over_untied_paired_medians=ratios, accuracy_clauses=clauses,
        accuracy_gate_passed=accuracy, structure=structure,
        structure_accuracy_hypothesis_supported=bool(accuracy and structure["passed"]),
        scope="Prespecified representation hypothesis only; no significance, runtime, AMR, continuum certification or novelty claim.",
        structure_check_scope="All symmetric raw rows, including every seed, signed mixture and per-combination linearity; no median hides a violation.")
    return by_split, paired, decision


def fmt(value):
    return f"{value:.6f}" if abs(value) >= 1e-4 or value == 0 else f"{value:.3e}"


def relative_link(path, output):
    return os.path.relpath(path, output).replace(os.sep, "/")


def write_markdown(summary, output, source):
    decision, data = summary["decision"], summary["by_split"]
    status = "通过" if decision["structure_accuracy_hypothesis_supported"] else "未通过"
    lines = ["# 第十阶段：离散误差算子结构消融", "",
        f"**预定的结构准确率假设{status}。本轮检验表示与一致性，没有完整求解加速、自适应网格闭环或新颖性结论。**", "",
        "正式完成9个模型（3种结构×3个初始化），768条测试记录。64个训练几何、12个验证几何；"
        "测试有16个ID和16个独立高对比几何。窄载荷与细网格复用ID的16个几何，共64个测试配置，独立测试几何仍为32个。", "",
        "源码、协议、完成标记、结果文件和9个checkpoint哈希均通过核验，768条方法/几何/种子组合无缺漏或重复。"
        "报告独立重算已保存4×4矩阵可确定的指标；场能量、局部标记和向量线性误差仅核验记录的有限性与范围，未重新求解。", "",
        "## 冻结假设的判断", "",
        "主指标为误差矩阵相对Frobenius误差。先在同一几何内对三个训练种子取中位，再逐几何计算"
        "symmetric/untied之比，最后汇总几何；不是两个汇总中位误差相除。", "",
        "| 测试 | S/untied配对中位比 | S误差更低的几何 | 达到≤0.9 |", "|---|---:|---:|---|"]
    for split in SPLITS:
        p = summary["paired_matrix_error"][split]["symmetric_over_untied"]
        lines.append(f"| {LABELS[split]} | {fmt(p['median'])} | {p['lower_error_geometries']}/16 | {'是' if p['median'] <= .9 else '否'} |")
    lines += ["", "预定准确率条件为ID≤0.9，至少一个分布外测试≤0.9，其余分布外测试≤1.1；"
        f"准确率门槛：`{decision['accuracy_gate_passed']}`。结构检查：`{decision['structure']['passed']}`。"
        "这是工程研究判据，不是统计显著性检验；不能从某个单独种子或测试子集挑选有利结论。", ""]
    st = decision["structure"]
    lines += [f"symmetric的全部192条原始行中，最大互易缺陷为{fmt(st['maximum_reciprocity_defect'])}，"
        f"最大混合线性缺陷为{fmt(st['maximum_mixture_linearity_defect'])}，逐组合最大为{fmt(st['maximum_per_combination_linearity_defect'])}；"
        f"原始及混合预测矩阵的相对最小特征值分别至少为{fmt(st['minimum_relative_eigenvalue'])}、"
        f"{fmt(st['minimum_mixed_relative_eigenvalue'])}。结构保持与误差预测准确是两个不同命题。", "",
        "## 所有方法的同口径结果", "",
        "以下每项先取同几何三个模型种子的中位，再取16个几何的中位；经典方法每几何只有一行。"
        "能量误差为平方能量范数之比，未开平方。标记质量越大越好，其余表列误差越小越好。"]
    for split in SPLITS:
        lines += ["", f"### {LABELS[split]}", "",
            "| 方法 | 矩阵相对误差 | 混合交叉归一RMSE | 缺陷相对能量误差 | 标记质量比 |",
            "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            m = data[split][method]["medians"]
            lines.append(f"| {method} | {fmt(m['matrix_error_relative'])} | {fmt(m['mixture_cross_normalized_rmse'])} | "
                         f"{fmt(m['defect_energy_error_mean'])} | {fmt(m['marker_mass_ratio'])} |")
    lines += ["", "![四项表示指标](figures/comparison.png)", "",
        "图中点为几何中位数，竖线是几何四分位区间，不是置信区间。训练种子先在几何内聚合。"
        "完整逐几何、逐种子及逐组合统计保存在summary.json，原始4×4矩阵仍保留在evaluation.json。", "",
        "## 逐种子透明性", "",
        "| 初始化 | ID S/U | 高对比 S/U | 窄载荷 S/U | 细网格 S/U |", "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        values = [summary["paired_matrix_error"][split]["symmetric_over_untied"]["per_seed"][str(seed)]["median"] for split in SPLITS]
        lines.append(f"| {seed} | " + " | ".join(fmt(x) for x in values) + " |")
    lines += ["", "以上逐种子结果用于暴露初始化变化，不用于测试后挑选模型。三个模型宽度和训练预算相同，"
        "untied/nonlinear含额外合成参数，因此不是等参数量消融。", "",
        "| 模型 | 参数量 | 验证选择的epoch（按种子顺序） |", "|---|---:|---|"]
    for method in LEARNED:
        records = sorted((r for r in summary["training"] if r["kind"] == method), key=lambda r: r["seed"])
        lines.append(f"| {method} | {', '.join(str(x) for x in sorted({r['parameters'] for r in records}))} | "
                     + ", ".join(str(r["best_epoch"]) for r in records) + " |")
    lines += ["", f"记录的数据生成合计{summary['recorded_costs']['data_generation_seconds']:.3f}秒，"
        f"9个训练/验证/保存循环合计{summary['recorded_costs']['training_loop_seconds']:.3f}秒。"
        "这些是本轮工作记录，不是在线速度对照，也未覆盖完整进程初始化等全部离线费用。", "",
        "## 结论边界", "",
        "- 学习目标是同一嵌套有限元系统的粗细网格离散误差，不是连续PDE真误差或严格误差上界。",
        "- 窄载荷测试同时改变四个源项与目标；符号/幅值组合检验结构和组合预测，不单独建立一般目标量泛化。",
        "- 局部标记固定源项为第一个载荷，使用统一的元素弱残差分解；最高25%单元的质量捕获不能替代实际细化后的误差下降。",
        "- 经典AMG、Jacobi与全部神经消融均保留。即使某个全局指标更好，也不能忽略能量误差、局部指标或新增计算成本。",
        "- 未执行网格适应、完整成本比较或最近方法的完整复现；结构一致性与本次误差结果均不足以证明创新成立。", "",
        f"[完整聚合](summary.json) · [原始逐例结果]({relative_link(source/'evaluation.json', output)}) · "
        f"[冻结协议]({relative_link(ROOT/'docs/stage10_protocol_zh.md', output)}) · "
        f"[原始完成记录]({relative_link(source/'complete.json', output)})"]
    (output/"REPORT_zh.md").write_text("\n".join(lines)+"\n")


def plot(by_split, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#207765", "#367eb0", "#b6594c", "#9a8350", "#8064a2", "#222222"]
    panels = [("matrix_error_relative", "Relative matrix error (lower is better)"),
              ("mixture_cross_normalized_rmse", "Mixed-goal normalized RMSE (lower is better)"),
              ("marker_mass_ratio", "Top-25% marker mass ratio (higher is better)"),
              ("defect_energy_error_mean", "Relative squared energy error (lower is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    xs = np.arange(4)
    labels = ["ID", "High contrast", "Narrow loads", "Finer grid"]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for index, (method, color) in enumerate(zip(METHODS, colors)):
            values = [by_split[s][method]["metrics"][metric] for s in SPLITS]
            yy = np.array([v["median"] for v in values])
            low, high = np.array([v["q25"] for v in values]), np.array([v["q75"] for v in values])
            ax.errorbar(xs+(index-2.5)*.075, yy, yerr=np.vstack((yy-low, high-yy)),
                        fmt="o-", markersize=4, linewidth=1.1, elinewidth=.85,
                        capsize=2, alpha=.95, color=color, label=METHOD_LABELS[method])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(xs, labels)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        if metric != "marker_mass_ratio":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0., 1.035)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, .065))
    fig.suptitle("Stage 10: discrete error-operator representation pilot", fontsize=14, y=.975)
    fig.text(.5, .025, "Geometry medians and interquartile ranges; trained models first aggregate three seeds per geometry.\n"
             "No inference-time, adaptive-mesh, continuum-certification or novelty claim.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.91, wspace=.23, hspace=.3)
    folder = output/"figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder/f"comparison.{extension}", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/stage10_pilot")
    parser.add_argument("--output", default="results/stage10_report")
    args = parser.parse_args()
    source = (ROOT/args.source).resolve()
    output = (ROOT/args.output).resolve()
    require(source != output and source not in output.parents, "Report output must not be inside the frozen source directory")
    rows, training, train_data, test_data, frozen, complete, verification = audit(source)
    by_split, paired, decision = aggregate(rows)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), reporter_sha256=sha(__file__),
        source_directory=str(source), freeze=frozen, complete=complete, verification=verification,
        aggregation="Median across three training seeds within each geometry, then median/quantiles across 16 geometries. Paired ratios are formed within geometry; classical rows are never replicated as seeds.",
        by_split=by_split, paired_matrix_error=paired, decision=decision,
        training=[{k: v for k, v in r.items() if k != "history"} for r in training],
        recorded_costs=dict(data_generation_seconds=sum(r["generation_seconds"] for r in train_data["train"]+train_data["validation"]+test_data),
                            training_loop_seconds=sum(r["training_seconds"] for r in training)),
        input_sha256={p.name: sha(p) for p in sorted(source.glob("*.json"))})
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    write_markdown(summary, output, source)
    plot(by_split, output)
    print(json.dumps(dict(rows=len(rows), audit_passed=True, decision=decision, output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
