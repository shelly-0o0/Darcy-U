"""Audit and report the post-hoc stage-10 projection probe; no solver calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("id", "contrast", "narrow_load", "resolution")
KINDS = ("symmetric", "untied", "nonlinear")
METHODS = (*KINDS, "jacobi1", "jacobi8", "amg_v1")
PLOT_METHODS = (*KINDS, "jacobi8", "amg_v1")
SEEDS = (51001, 51002, 51003)
METRICS = ("matrix_error_relative", "normalized_matrix_rmse",
           "defect_energy_error_mean", "marker_mass_ratio", "indicator_relative_l1")
SPLIT_ZH = {"id": "同分布", "contrast": "高对比度", "narrow_load": "窄载荷", "resolution": "倍分辨率"}
METHOD_LABEL = {"symmetric": "Symmetric", "untied": "Untied", "nonlinear": "Nonlinear",
                "jacobi1": "Jacobi(1)", "jacobi8": "Jacobi(8)", "amg_v1": "AMG(1 V)"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def key(row):
    return row["split"], row["geometry"], row["method"], row["training_seed"]


def close(actual, recorded, name):
    require(math.isclose(float(actual), float(recorded), rel_tol=1e-9, abs_tol=1e-12),
            f"Recomputed value disagrees: {name}")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"], row["geometry"])].append(row)
    geometry = []
    for (split, method, number), records in sorted(groups.items()):
        record = dict(split=split, method=method, geometry=number, seeds=len(records))
        record.update({state: {metric: median(r[state][metric] for r in records) for metric in METRICS}
                       for state in ("before", "after")})
        record["coarse_fraction_of_error_mean"] = median(r["energy"]["coarse_fraction_of_error_mean"] for r in records)
        geometry.append(record)
    lookup = {(r["split"], r["method"], r["geometry"]): r for r in geometry}
    totals = {}
    for split in SPLITS:
        totals[split] = {}
        for method in METHODS:
            records = [r for r in geometry if r["split"] == split and r["method"] == method]
            entry = {state: {metric: median(r[state][metric] for r in records) for metric in METRICS}
                     for state in ("before", "after")}
            entry["coarse_fraction_of_error_mean"] = median(r["coarse_fraction_of_error_mean"] for r in records)
            entry["paired_after_over_before"] = {
                metric: median(r["after"][metric]/r["before"][metric] for r in records) for metric in METRICS}
            entry["paired_marker_mass_gain"] = median(r["after"]["marker_mass_ratio"]-r["before"]["marker_mass_ratio"] for r in records)
            entry["geometries_marker_improved"] = sum(r["after"]["marker_mass_ratio"] > r["before"]["marker_mass_ratio"] for r in records)
            entry["paired_after_over_projected_amg"] = {
                metric: median(r["after"][metric]/lookup[(split, "amg_v1", r["geometry"])]["after"][metric] for r in records)
                for metric in METRICS}
            entry["geometries"] = len(records)
            totals[split][method] = entry
    return totals, geometry


def verify(source):
    complete = read(source/"complete.json"); manifest = read(source/"manifest.json")
    require(complete["complete"] and complete["post_hoc"] and complete["formal_gate_unchanged"], "Probe is not complete/post-hoc")
    require(manifest["post_hoc"] and manifest["zero_training"] and manifest["formal_gate_unchanged"], "Wrong probe scope")
    require(complete["rows"] == 768 and complete["contexts"] == 64, "Wrong completion counts")
    require(set(complete["output_sha256"]) == {"manifest.json", "rows.json", "contexts.json", "summary.json"}, "Incomplete probe output hash set")
    for name, digest in complete["output_sha256"].items():
        require(sha(source/name) == digest, f"Probe output changed: {name}")
    require(complete["input_sha256"] == manifest["input_sha256"], "Probe input manifests disagree")
    recorded_sources = [Path(p) for p in manifest["input_sha256"] if Path(p).name == "stage10_projection_probe.py"]
    require(len(recorded_sources) == 1, "Cannot identify the recorded repository root")
    recorded_root = recorded_sources[0].parent
    mapped_paths = {}
    for path, digest in manifest["input_sha256"].items():
        original_path = Path(path)
        require(original_path.is_relative_to(recorded_root), f"Recorded input is outside its repository: {path}")
        current_path = ROOT/original_path.relative_to(recorded_root)
        mapped_paths[path] = str(current_path)
        require(sha(current_path) == digest, f"Current mapped input path has changed: {current_path}")
    for name, digest in manifest["formal_source_sha256"].items():
        require(sha(ROOT/name) == digest, f"Frozen formal source changed: {name}")
    require(sha(ROOT/"docs/stage10_protocol_zh.md") == manifest["formal_protocol_sha256"], "Frozen formal protocol changed")
    formal_freezes = [Path(mapped_paths[p]) for p in manifest["input_sha256"] if Path(p).name == "freeze.json"]
    require(len(formal_freezes) == 1, "Formal freeze path is ambiguous")
    formal_dir = formal_freezes[0].parent
    require(sha(formal_freezes[0]) == manifest["formal_freeze_sha256"], "Formal freeze changed")
    frozen = read(formal_freezes[0]); formal_complete = read(formal_dir/"complete.json")
    require(frozen["source_sha256"] == manifest["formal_source_sha256"] and frozen["protocol_sha256"] == manifest["formal_protocol_sha256"], "Frozen source/protocol manifests disagree")
    require(formal_complete["complete"] and not frozen["smoke"], "Formal run unavailable")
    for name, digest in formal_complete["result_sha256"].items():
        require(sha(formal_dir/name) == digest, f"Formal result changed: {name}")
    trained = read(formal_dir/"training_complete.json")["checkpoints"]
    require({(r["kind"], r["seed"]) for r in trained} == {(kind, seed) for kind in KINDS for seed in SEEDS}, "Wrong model set")
    for record in trained:
        name = f"{record['kind']}_{record['seed']}.pt"
        require(sha(formal_dir/name) == record["sha256"], f"Checkpoint changed: {name}")
    raw_formal = read(formal_dir/"evaluation.json"); formal = {key(r): r for r in raw_formal}
    expected = {(split, geometry, method, seed)
                for split in SPLITS
                for geometry in range(50300 if split == "contrast" else 50200,
                                      (50300 if split == "contrast" else 50200)+16)
                for method in METHODS for seed in (SEEDS if method in KINDS else (None,))}
    rows = read(source/"rows.json"); contexts = read(source/"contexts.json"); saved_summary = read(source/"summary.json")
    require(len(rows) == len(raw_formal) == 768 and len(contexts) == 64, "Wrong raw record count")
    require({key(r) for r in rows} == set(formal) == expected, "Missing or duplicate record keys")
    require({(r["split"], r["geometry"]) for r in contexts} == {(k[0], k[1]) for k in expected}, "Missing context keys")
    require(all(finite(x) for x in (rows, contexts, saved_summary)), "Nonfinite numeric values")
    maxima = dict(cpu_replay_matrix_difference=0., gram_change_relative_to_reference=0.,
                  gram_change_relative_to_before=0., energy_identity_error=0.,
                  projected_coarse_orthogonality=0., target_coarse_orthogonality=0.,
                  matrix_error_absolute_change=0.)
    energy_increases = 0
    tolerance = manifest["identity_tolerance"]
    for row in rows:
        original = formal[key(row)]
        require(row["formal_metrics"] == {m: original[m] for m in METRICS}, "Formal metrics changed in replay record")
        reference = np.array(row["before"]["reference_matrix"])
        before = np.array(row["before"]["estimated_matrix"]); after = np.array(row["after"]["estimated_matrix"])
        require(reference.shape == before.shape == after.shape == (4, 4), "Invalid matrix shape")
        require(np.allclose(reference, original["reference_matrix"], rtol=1e-10, atol=1e-25), "Reference replay mismatch")
        require(np.array_equal(reference, row["after"]["reference_matrix"]), "Projection changed reference matrix")
        replay = float(np.linalg.norm(before-np.array(original["estimated_matrix"]))/np.linalg.norm(reference))
        gram_ref = float(np.linalg.norm(before-after)/np.linalg.norm(reference))
        gram_pred = float(np.linalg.norm(before-after)/np.linalg.norm(before))
        close(replay, row["cpu_replay_matrix_difference"], "CPU Gram replay")
        close(gram_ref, row["gram_invariance_relative_to_reference"], "Gram invariance/reference")
        close(gram_pred, row["gram_invariance_relative_to_before"], "Gram invariance/prediction")
        require(replay <= manifest["cpu_replay_tolerance"] and max(gram_ref, gram_pred) <= tolerance, "Replay/invariance tolerance failed")
        e = row["energy"]
        before_e = np.array(e["error_energy_before"]); after_e = np.array(e["error_energy_after"])
        removed = np.array(e["coarse_energy_removed"]); true_e = np.array(e["reference_energy"])
        require(all(v.shape == (4,) for v in (before_e, after_e, removed, true_e)), "Invalid energy vector")
        require(np.all(true_e > 0) and np.all(before_e > 0) and np.all(after_e >= 0) and np.all(removed >= 0), "Invalid energy signs")
        identity = np.abs(before_e-after_e-removed)/np.maximum.reduce((abs(before_e), abs(after_e), abs(removed), np.full(4, 1e-30)))
        require(np.allclose(identity, e["energy_identity_relative_error"], rtol=1e-8, atol=1e-14), "Energy identity summary mismatch")
        close(float(np.mean(removed/before_e)), e["coarse_fraction_of_error_mean"], "Removed energy fraction")
        close(float(np.mean(before_e/true_e)), row["before"]["defect_energy_error_mean"], "Pre-projection energy metric")
        close(float(np.mean(after_e/true_e)), row["after"]["defect_energy_error_mean"], "Projected energy metric")
        require(max(identity) <= tolerance and max(e["projected_coarse_orthogonality"]) <= tolerance and max(e["target_coarse_orthogonality"]) <= tolerance, "Projection identity/orthogonality failed")
        energy_increases += int(np.any(after_e > before_e*(1+tolerance)))
        for state, value in (("before", before), ("after", after)):
            close(float(np.linalg.norm(value-reference)/np.linalg.norm(reference)), row[state]["matrix_error_relative"], "Matrix-error metric")
            require(0 <= row[state]["marker_mass_ratio"] <= 1+1e-12 and row[state]["indicator_relative_l1"] >= 0, "Invalid local metric")
        observations = dict(cpu_replay_matrix_difference=replay, gram_change_relative_to_reference=gram_ref,
                            gram_change_relative_to_before=gram_pred, energy_identity_error=float(max(identity)),
                            projected_coarse_orthogonality=max(e["projected_coarse_orthogonality"]),
                            target_coarse_orthogonality=max(e["target_coarse_orthogonality"]),
                            matrix_error_absolute_change=abs(row["before"]["matrix_error_relative"]-row["after"]["matrix_error_relative"]))
        for name, value in observations.items():
            maxima[name] = max(maxima[name], value)
    totals, geometry = aggregate(rows)
    for split in SPLITS:
        for method in METHODS:
            fresh = totals[split][method]; saved = saved_summary["by_split_method"][split][method]
            require(fresh == saved, f"Independent aggregation disagrees: {split}/{method}")
    require(energy_increases == 0, "Projection increased an error energy beyond tolerance")
    formal_groups = defaultdict(list)
    for row in raw_formal:
        formal_groups[(row["split"], row["geometry"], row["method"])].append(row)
    formal_ratios = {}
    for split in SPLITS:
        geometries = sorted({row["geometry"] for row in raw_formal if row["split"] == split})
        formal_ratios[split] = median(
            median(r["matrix_error_relative"] for r in formal_groups[(split, g, "symmetric")]) /
            median(r["matrix_error_relative"] for r in formal_groups[(split, g, "untied")]) for g in geometries)
    symmetric_rows = [r for r in raw_formal if r["method"] == "symmetric"]
    structure_passed = (max(r["reciprocity_defect"] for r in symmetric_rows) <= 1e-5
                        and max(r["mixture_linearity_defect"] for r in symmetric_rows) <= 1e-5
                        and min(r["relative_minimum_symmetric_eigenvalue"] for r in symmetric_rows) >= -1e-5)
    formal_gate = (formal_ratios["id"] <= .9 and any(formal_ratios[s] <= .9 for s in SPLITS[1:])
                   and all(formal_ratios[s] <= 1.1 for s in SPLITS[1:]) and structure_passed)
    marker_replay = [abs(r["before"]["marker_mass_ratio"]-r["formal_metrics"]["marker_mass_ratio"]) for r in rows]
    audit = dict(passed=True, rows=768, contexts=64, independent_geometries=32, frozen_models=9,
                 classical_methods=3, all_method_keys_complete=True, input_hashes_verified=len(manifest["input_sha256"]),
                 frozen_source_hashes_verified=len(manifest["formal_source_sha256"]), checkpoint_hashes_verified=9,
                 energy_increase_rows=energy_increases, maxima=maxima,
                 cpu_replay_scope="Global Gram matrices are checked, not equality of full prediction vectors between CPU and GPU",
                 aggregation="Median over training seeds within geometry, then median over geometries; paired ratios are formed per geometry before aggregation",
                 input_sha256=manifest["input_sha256"], probe_complete_sha256=sha(source/"complete.json"),
                 input_path_mapping=mapped_paths, recorded_repository_root=str(recorded_root),
                 current_repository_root=str(ROOT), formal_structure_passed=structure_passed,
                 formal_symmetric_over_untied_paired= formal_ratios, formal_gate_passed=formal_gate,
                 cpu_marker_replay_changed_rows=sum(v > 1e-12 for v in marker_replay),
                 maximum_cpu_marker_replay_difference=max(marker_replay),
                 formal_freeze_sha256=manifest["formal_freeze_sha256"], reporter_sha256=sha(__file__))
    return audit, totals, geometry, saved_summary, formal_dir


def plot(totals, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/darcy-u-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.subplots_adjust(left=.09, right=.945, top=.76, bottom=.23, wspace=.34)
    panels = [("marker_mass_ratio", "a  Marker mass captured (higher is better)", .65, 1., "viridis"),
              ("indicator_relative_l1", "b  Relative local L1 error (lower is better)", 0., 1., "magma_r")]
    for ax, (metric, title, vmin, vmax, cmap) in zip(axes, panels):
        values = np.array([[totals[split][method][state][metric] for split in SPLITS for state in ("before", "after")]
                           for method in PLOT_METHODS])
        im = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", fontsize=12, pad=38)
        ax.set_yticks(range(len(PLOT_METHODS)), [METHOD_LABEL[m] for m in PLOT_METHODS])
        ax.set_xticks(range(8), ["Before", "After"]*4, fontsize=9)
        ax.tick_params(length=0, pad=6)
        for group, label in enumerate(("ID", "High contrast", "Narrow load", "Finer mesh")):
            ax.text(2*group+.5, -.72, label, ha="center", va="bottom", fontsize=9.5)
        for x in (1.5, 3.5, 5.5):
            ax.axvline(x, color="white", lw=2)
        for y in (.5, 1.5, 2.5, 3.5):
            ax.axhline(y, color="white", lw=.5, alpha=.4)
        for (y, x), value in np.ndenumerate(values):
            color = im.cmap(im.norm(value))
            luminance = .2126*color[0]+.7152*color[1]+.0722*color[2]
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="black" if luminance > .56 else "white")
        fig.colorbar(im, ax=ax, fraction=.028, pad=.025, shrink=.95)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Coarse-complement projection: a post-hoc diagnostic, no training", fontsize=15, y=.96)
    fig.text(.09, .105, "All methods receive the same Q. Marker mass approaches saturation after projection.", fontsize=10)
    fig.text(.09, .055, "64 test contexts; 768 paired records including Jacobi(1) in the tables. Seed median within geometry, then geometry median.\nNo measured adaptive-refinement benefit, novelty or speed claim.", fontsize=9, color="#444444")
    figure_dir = output/"figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir/"projection.png", dpi=200, facecolor="white")
    fig.savefig(figure_dir/"projection.pdf", facecolor="white")
    plt.close(fig)


def report(audit, totals, timing, source, output, formal_dir):
    relative_source = os.path.relpath(source, output)
    relative_formal = os.path.relpath(formal_dir, output)
    relative_protocol = os.path.relpath(ROOT/"docs/stage10_projection_probe_zh.md", output)
    a = audit["maxima"]
    lines = ["# 第十阶段投影诊断：全局误差不变，局部分解明显改善", "",
             "这是在正式结果揭示局部标记偏弱后制定并执行的**事后、零训练机制诊断**。所有学习模型和经典方法接受同一粗空间正交补投影；正式训练、测试记录和结构收益判据不变。它不构成独立确认实验，也不声称新的算法创新、端到端加速或真实自适应细化收益。", "",
             "## 核心发现", "",
             "预测中的粗空间分量对全局误差矩阵不可见，却会改变固定单元分解。去除它后，多种方法的局部指标大幅改善，包括经典 Jacobi 与 AMG。因此，这首先暴露了表示及评估的一致性问题，不能将改进归因于 AI。", "",
             "| 测试组 | Symmetric 移除的误差能量份额 | 标记质量：前 → 后 | 局部 L1：前 → 后 | 投影后能量误差 / 投影 AMG，逐几何配对中位 |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        r = totals[split]["symmetric"]
        lines.append(f"| {SPLIT_ZH[split]} | {r['coarse_fraction_of_error_mean']:.2%} | {r['before']['marker_mass_ratio']:.4f} → {r['after']['marker_mass_ratio']:.4f} | {r['before']['indicator_relative_l1']:.4f} → {r['after']['indicator_relative_l1']:.4f} | {r['paired_after_over_projected_amg']['defect_energy_error_mean']:.4f} |")
    lines += ["", "“移除份额”先对四右端分别计算粗分量能量 / 原误差能量并取均值，再按训练种子、几何逐层取中位；不是两个总体中位数相除。能量误差指相对 A 范数的**平方**。投影后 Symmetric 的剩余误差能量仍高于同样投影的 AMG；标记质量则对多个方法已接近 1，难以据此区分实际细化价值。", "",
              "![投影前后局部指标](figures/projection.png)", "",
              "图中每个值按同几何三个训练种子中位、再跨几何中位汇总；经典方法只有一个确定性记录。左图越高越好，右图越低越好。图为五个主要对照，以下完整表保留 Jacobi(1)。", "",
              "## 同一投影与数学边界", "",
              r"定义 $A_c=P^TAP$、$Q=I-PA_c^{-1}P^TA$，对每个当前预测使用 $\widehat\delta_Q=Q\widehat\delta$。真实两层缺陷满足 $P^TA\delta=0$。令 $c=(I-Q)\widehat\delta$，则", "",
              "$$", r"r_i^T\widehat\delta_{Q,j}=r_i^T\widehat\delta_j,\qquad \|\widehat\delta-\delta\|_A^2-\|\widehat\delta_Q-\delta\|_A^2=\|c\|_A^2.", "$$", "",
              "全局矩阵不变和能量不增来自经典正交投影；局部标记改善没有由这两个恒等式保证。该处理也未消除细空间正交补内的预测误差。对两个线性学习模型，它相当于输出 QHθr；非线性模型则先计算 Nθ(r)，再施加 Q。不能把 QHθ 本身一般性地称作欧氏对称正定矩阵。原先残差对上的双线性与互易性结论应通过完整误差映射讨论。", "",
              "## 全方法结果", ""]
    for split in SPLITS:
        lines += [f"### {SPLIT_ZH[split]}", "",
                  "| 方法 | 全局矩阵误差，前后不变 | 相对能量误差平方：前 → 后 | 标记质量：前 → 后 | 局部 L1：前 → 后 | 标记改善几何数 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            r = totals[split][method]; b = r["before"]; p = r["after"]
            lines.append(f"| {METHOD_LABEL[method]} | {b['matrix_error_relative']:.4f} | {b['defect_energy_error_mean']:.4f} → {p['defect_energy_error_mean']:.4f} | {b['marker_mass_ratio']:.4f} → {p['marker_mass_ratio']:.4f} | {b['indicator_relative_l1']:.4f} → {p['indicator_relative_l1']:.4f} | {r['geometries_marker_improved']}/16 |")
        lines.append("")
    lines += ["## 完整性与证据范围", "",
              f"已独立核验全部 **{audit['rows']} 条唯一匹配记录、64 个 context、9 个冻结模型和3个经典方法**。四组各16个 context，但窄载荷和倍分辨率复用同分布几何，所以只有32个独立系数几何。窄载荷同时改变四个源项与目标；局部指标固定源项0对四个目标。", "",
              f"核验了 probe complete 的输出哈希、{audit['input_hashes_verified']} 个输入当前路径哈希、{audit['frozen_source_hashes_verified']} 个冻结源码、正式协议及9个权重哈希。报告从原始记录重新聚合，和 probe 汇总一致。最大 CPU/GPU 全局 Gram 重放差为 {a['cpu_replay_matrix_difference']:.3e}；**这只核验全局矩阵，不证明完整预测向量相等**。所有投影前后比较采用本次同一 CPU 预测。", "",
              f"与正式 GPU 记录比较，CPU 投影前的标记质量在 {audit['cpu_marker_replay_changed_rows']}/768 条记录上变化超过10⁻¹²，最大绝对差为 {audit['maximum_cpu_marker_replay_difference']:.3e}。该差异保留在原始记录中，未将 CPU 重放冒充逐位相同的预测。", "",
              "| 数值检查 | 最大相对差/缺陷 |", "|---|---:|",
              f"| 全局 Gram 变化 / 参考 Gram 范数 | {a['gram_change_relative_to_reference']:.3e} |",
              f"| 全局 Gram 变化 / 投影前 Gram 范数 | {a['gram_change_relative_to_before']:.3e} |",
              f"| 误差能量分解恒等式 | {a['energy_identity_error']:.3e} |",
              f"| 投影后粗正交残差 | {a['projected_coarse_orthogonality']:.3e} |",
              f"| 参考缺陷粗正交残差 | {a['target_coarse_orthogonality']:.3e} |", "",
              f"检查容差为10⁻⁸，未发现能量上升记录。完整矩阵主指标最大绝对变化为 {a['matrix_error_absolute_change']:.3e}。从正式记录复算的结构贡献判据为**{'通过' if audit['formal_gate_passed'] else '未通过'}**；此事后投影不更改该判据。", "",
              "## 成本与下一步限制", "",
              f"本次 CPU 诊断记录的全部投影时间合计 {timing['summed_projection_seconds']:.4f} 秒，数据生成 {timing['summed_case_generation_seconds']:.4f} 秒、预测 {timing['summed_prediction_seconds']:.4f} 秒、AMG准备 {timing['summed_amg_setup_seconds']:.4f} 秒。每次投影包含细矩阵乘、限制、已有粗 LU 求解、延拓及减法；同一 context 的因子分解由参考数据生成准备并复用。模型加载和指标评估另有开销，以上组件并非全部端到端成本。**这些数值只说明诊断工作量，不用于比较速度或声称加速。**", "",
              "合理的后续验证是：保持同一 PDE 和固定模型，在独立几何上给全部方法同样的投影，并检验固定标记预算的一次真实细化后 QoI 误差和完整成本，保留投影 AMG 与一次细主问题求解覆盖多目标的强对照。现有 top-25% 标记质量接近饱和，不能代替这种验证；在此之前没有证据支持扩大训练或声称 AMR 优势。", "",
              "参考标签仍是两个离散网格间的误差，不是连续 PDE 的真误差证书。正交投影、结构正确性和局部指标改善也不建立论文新颖性。", "",
              "## 数据与复现", "",
              f"[原始768条记录]({relative_source}/rows.json)、[诊断运行前方案]({relative_protocol})、[正式原始评价]({relative_formal}/evaluation.json)、[本报告审计及汇总](summary.json)、[矢量图](figures/projection.pdf)。", "",
              "```bash", "python report_stage10_projection.py --source results/stage10_projection_probe --output results/stage10_projection_report", "```", "",
              "报告仅读取既有数据并绘图，不运行模型、PDE求解或训练。通过记录中的唯一 stage10_projection_probe.py 路径识别历史仓库根目录，再把其下相对路径映射到当前仓库，因此支持 GitHub 克隆目录变更。映射只改变读取位置，不修改原始 manifest；当前文件不匹配冻结哈希时停止。", ""]
    (output/"REPORT_zh.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/stage10_projection_probe")
    parser.add_argument("--output", default="results/stage10_projection_report")
    args = parser.parse_args()
    source = (ROOT/args.source).resolve(); output = (ROOT/args.output).resolve()
    require(source != output and source not in output.parents, "Report must not write inside raw diagnostic data")
    audit, totals, geometry, timing, formal_dir = verify(source)
    output.mkdir(parents=True, exist_ok=True)
    dump(output/"summary.json", dict(audit=audit, post_hoc=True, zero_training=True, formal_gate_unchanged=True,
                                     by_split_method=totals, geometry_summaries=geometry))
    plot(totals, output)
    report(audit, totals, timing, source, output, formal_dir)
    dump(output/"complete.json", dict(complete=True, audit_passed=True, rows=768,
                                      reporter_sha256=sha(__file__), source_complete_sha256=sha(source/"complete.json"),
                                      output_sha256={name: sha(output/name) for name in ("summary.json", "REPORT_zh.md", "figures/projection.png", "figures/projection.pdf")}))
    print(json.dumps(dict(complete=True, rows=768, output=str(output))))


if __name__ == "__main__":
    main()
