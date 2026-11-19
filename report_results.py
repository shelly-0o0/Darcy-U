#!/usr/bin/env python3
"""Summarize measured sequence runs without treating repeats as geometries.

Only stdlib, NumPy and Matplotlib are required. This script never launches
solves or training. Usage: python report_results.py results/pilot_cpu
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["amg", "spectral_full", "spectral_reuse", "smooth_increment", "unet_increment"]
LABELS = {
    "amg": "AMG", "spectral_full": "Full spectral",
    "spectral_reuse": "Spectral reuse", "smooth_increment": "Smoothed increment",
    "unet_increment": "U-Net increment",
}
COLORS = dict(zip(METHODS, ["#7e8b96", "#d9a441", "#3583a8", "#7d5ba6", "#df6854"]))
COMPONENTS = {
    "Assembly + patches": ["assembly_seconds", "patch_seconds"],
    "Reuse": ["reuse_seconds"],
    "Spectral construction": ["spectral_seconds"],
    "Candidates": ["candidate_seconds"],
    "Checks": ["check_seconds"],
    "Spectral repair": ["fallback_seconds"],
    "Preconditioner setup": ["setup_seconds"],
    "PCG": ["solve_seconds"],
    "Direct recovery": ["direct_fallback_seconds"],
}


def number(value, digits=3):
    return "不适用" if value is None else f"{value:.{digits}f}"


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0, "median": None, "min": None, "max": None}
    return {"count": len(a), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def range_text(value, digits=3):
    return f"{number(value['min'], digits)}–{number(value['max'], digits)}"


def load_and_summarize(run):
    measurements = run / "measurements.json"
    manifest_path = run / "manifest.json"
    records = json.loads(measurements.read_text())
    manifest = json.loads(manifest_path.read_text())
    config = manifest["configuration"]
    if not records:
        raise ValueError("No measurements to summarize")
    groups = defaultdict(list)
    for row in records:
        key = (row["method"], row["sequence_seed"], row["model_seed"], row["repeat"])
        groups[key].append(row)
    expected = {(method, seq, model, repeat)
                for method in config["methods"] for seq in config["test_seeds"]
                for model in (config["model_seeds"] if method == "unet_increment" else [-1])
                for repeat in range(config["repeats"])}
    missing = expected - groups.keys()
    if missing:
        raise ValueError(f"Benchmark incomplete: {len(missing)} missing sequence/model/repeat groups")
    if set(groups) != expected:
        raise ValueError("Measurements contain variants not present in the manifest")
    runs, warnings = [], []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x["step"])
        if [row["step"] for row in rows] != list(range(config["steps"])):
            raise ValueError(f"Incomplete or duplicate steps in {key}")
        breakdown = {name: sum(sum(float(row.get(k, 0)) for k in keys) for row in rows)
                     for name, keys in COMPONENTS.items()}
        total = sum(float(row["total_seconds"]) for row in rows)
        other = total - sum(breakdown.values())
        if other < -1e-6:
            warnings.append(f"Timing components exceed wall time in {key}: {other:.3e} seconds")
        breakdown["Other overhead"] = max(0.0, other)
        runs.append({"method": key[0], "sequence_seed": key[1], "model_seed": key[2], "repeat": key[3],
                     "steps": len(rows), "total_seconds": total, "breakdown_seconds": breakdown,
                     "iterations_max": max(row["iterations"] for row in rows),
                     "candidate_checks": sum(row["candidate_checks"] for row in rows),
                     "candidate_passes": sum(row["candidate_passes"] for row in rows),
                     "spectral_repairs": sum(row["spectral_repairs"] for row in rows),
                     "solver_fallbacks": sum(bool(row["solver_fallback"]) for row in rows)})
    units = defaultdict(list)
    for result in runs:
        units[(result["method"], result["sequence_seed"], result["model_seed"])].append(result)
    unit_results = []
    for (method, seq, model), repeats in sorted(units.items()):
        total = stats([x["total_seconds"] for x in repeats])
        unit_results.append({"method": method, "sequence_seed": seq, "model_seed": model,
                             "repeats": len(repeats), "repeat_total_seconds": total,
                             "total_seconds": total["median"],
                             "mean_breakdown_seconds": {name: float(np.mean([x["breakdown_seconds"][name] for x in repeats]))
                                                        for name in repeats[0]["breakdown_seconds"]}})
    method_results = {}
    variants = []
    for method in METHODS:
        own = [u for u in unit_results if u["method"] == method]
        if not own:
            continue
        geom = defaultdict(list)
        for u in own:
            geom[u["sequence_seed"]].append(u)
        geometry_totals = {str(seed): float(np.median([u["total_seconds"] for u in us]))
                           for seed, us in sorted(geom.items())}
        geometry_means = [{name: float(np.mean([u["mean_breakdown_seconds"][name] for u in us]))
                          for name in own[0]["mean_breakdown_seconds"]} for us in geom.values()]
        own_rows = [r for r in records if r["method"] == method]
        candidate_checks = sum(r["candidate_checks"] for r in own_rows)
        candidate_passes = sum(r["candidate_passes"] for r in own_rows)
        method_results[method] = {
            "independent_geometries": len(geom),
            "model_seeds": sorted(set(u["model_seed"] for u in own)),
            "geometry_total_seconds": geometry_totals,
            "sequence_total_seconds": stats(list(geometry_totals.values())),
            "sequence_model_total_seconds": stats([u["total_seconds"] for u in own]),
            "mean_breakdown_seconds": {name: float(np.mean([g[name] for g in geometry_means])) for name in geometry_means[0]},
            "max_iterations": max(r["iterations"] for r in own_rows),
            "max_relative_residual": max(r["final_relative_residual"] for r in own_rows),
            "max_energy_relative_error": max(r["energy_relative_error"] for r in own_rows),
            "candidate_checks_all_timed_runs": candidate_checks,
            "candidate_passes_all_timed_runs": candidate_passes,
            "candidate_pass_rate": candidate_passes / candidate_checks if candidate_checks else None,
            "spectral_repairs_all_timed_runs": sum(r["spectral_repairs"] for r in own_rows),
            "solver_fallbacks_all_timed_runs": sum(bool(r["solver_fallback"]) for r in own_rows),
            "coarse_dimension_min": min(r["coarse_dimension"] for r in own_rows),
            "coarse_dimension_max": max(r["coarse_dimension"] for r in own_rows),
        }
        for model in sorted(set(u["model_seed"] for u in own)):
            vals = [u["total_seconds"] for u in own if u["model_seed"] == model]
            variants.append({"method": method, "model_seed": model, "sequence_total_seconds": stats(vals)})
    paired = {}
    if "unet_increment" in method_results:
        learned = method_results["unet_increment"]["geometry_total_seconds"]
        for method, result in method_results.items():
            if method == "unet_increment":
                continue
            ratios = {s: result["geometry_total_seconds"][s] / learned[s] for s in learned}
            paired[method] = {"definition": "baseline seconds / U-Net seconds; >1 favors U-Net",
                              "ratios_by_geometry": ratios, "summary": stats(list(ratios.values()))}
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [measurements, manifest_path]},
        "aggregation": "Sum all steps including initialization. Median timing repeats per geometry/model. Median models within geometry, then median independent geometries. Ranges are observed ranges, not confidence intervals. Breakdown uses equal-weight means to preserve additivity.",
        "methods": method_results, "variants": variants, "sequence_model_units": unit_results,
        "timed_sequence_runs": runs, "paired_baseline_over_unet": paired, "warnings": warnings,
    }
    return records, manifest, summary


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(summary, figures):
    methods = list(summary["methods"])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, method in enumerate(methods):
        result = summary["methods"][method]
        s = result["sequence_total_seconds"]
        ax.bar(i, s["median"], color=COLORS[method], alpha=.85, width=.62)
        ax.errorbar(i, s["median"], yerr=[[s["median"]-s["min"]], [s["max"]-s["median"]]],
                    color="#22313d", capsize=5, fmt="none", lw=1.4)
        units = [u for u in summary["sequence_model_units"] if u["method"] == method]
        jitter = np.linspace(-.15, .15, len(units)) if len(units) > 1 else [0.0]
        ax.scatter(i + np.asarray(jitter), [u["total_seconds"] for u in units],
                   s=20, facecolor="white", edgecolor="#273746", zorder=4)
        ax.annotate(f"{s['median']:.2f}", (i, s["max"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=14, ha="right")
    ax.set_ylabel("Wall time per full sequence (s)")
    ax.set_title("Sequence time, including initialization")
    ax.text(.01, .99, "Bars: median across geometries; whiskers: observed geometry range\nDots: repeat-median geometry/model measurements", transform=ax.transAxes,
            va="top", fontsize=8.5, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    ax.margins(y=.25); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    save_figure(fig, figures / "sequence_timing")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    names = list(next(iter(summary["methods"].values()))["mean_breakdown_seconds"])
    bottom = np.zeros(len(methods))
    palette = plt.get_cmap("tab20").colors
    for j, name in enumerate(names):
        values = np.array([summary["methods"][m]["mean_breakdown_seconds"][name] for m in methods])
        if not np.any(values > 0):
            continue
        ax.bar(range(len(methods)), values, bottom=bottom, label=name, color=palette[j], width=.62)
        bottom += values
    ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=14, ha="right")
    ax.set_ylabel("Mean wall time per sequence (s)")
    ax.set_title("Online cost breakdown (equal geometry weights)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    save_figure(fig, figures / "cost_breakdown")


def plot_sequences(records, summary, figures):
    files = []
    for seed in sorted(set(r["sequence_seed"] for r in records)):
        selected = [r for r in records if r["sequence_seed"] == seed]
        steps = sorted(set(r["step"] for r in selected))
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for method in summary["methods"]:
            own = [r for r in selected if r["method"] == method]
            if not own:
                continue
            totals, iterations = [], []
            for step in steps:
                values = [r for r in own if r["step"] == step]
                totals.append(float(np.median([r["total_seconds"] for r in values])))
                iterations.append(float(np.median([r["iterations"] for r in values])))
            axes[0].plot(steps, np.cumsum(totals), label=LABELS[method], color=COLORS[method], lw=1.8)
            axes[1].plot(steps, iterations, label=LABELS[method], color=COLORS[method], marker=".", lw=1.3)
            if method in ("smooth_increment", "unet_increment"):
                tried, passed = [], []
                for step in steps:
                    rows = [r for r in own if r["step"] == step]
                    tried.append(float(np.median([r["candidate_checks"] for r in rows])))
                    passed.append(float(np.median([r["candidate_passes"] for r in rows])))
                axes[2].plot(steps, tried, "--", color=COLORS[method], label=f"{LABELS[method]}: checked")
                axes[2].plot(steps, passed, "-o", ms=3, color=COLORS[method], label=f"{LABELS[method]}: passed")
        event_rows = {r["step"]: r["event"] for r in selected}
        for step, event in event_rows.items():
            if "bridge_" in event:
                for ax in axes:
                    ax.axvline(step, color="#263746", alpha=.25, lw=1)
                axes[0].annotate("connect" if "connected" in event else "block", (step, .98),
                                 xycoords=("data", "axes fraction"), xytext=(4, -2),
                                 textcoords="offset points", va="top", fontsize=9)
        axes[0].set_ylabel("Cumulative time (s)"); axes[1].set_ylabel("PCG iterations")
        axes[2].set_ylabel("Candidate patches"); axes[2].set_xlabel("Sequence step (0 = initialization)")
        axes[0].set_title(f"Sequence seed {seed}: per-step medians over repeats/models")
        axes[0].legend(ncols=3, fontsize=8, frameon=False)
        if axes[2].lines:
            axes[2].legend(ncols=2, fontsize=8, frameon=False)
        for ax in axes:
            ax.grid(alpha=.18); ax.set_xlim(min(steps)-.2, max(steps)+.2)
        fig.tight_layout()
        name = f"sequence_seed{seed}"
        save_figure(fig, figures / name); files.append(name)
    return files


def load_training(manifest, run):
    training = []
    for seed in manifest["configuration"].get("model_seeds", []):
        # Prefer benchmark-local snapshots if the runner provides them.
        candidates = [run / f"unet_seed{seed}.training.json", ROOT / "checkpoints" / f"unet_seed{seed}.training.json"]
        for path in candidates:
            if path.exists():
                value = json.loads(path.read_text())
                value["source_path"] = str(path)
                value["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                value["benchmark_local_snapshot"] = path.parent == run
                training.append(value)
                break
    return training


def plot_training(training, figures):
    if not training or not any(t.get("history") for t in training):
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in training:
        h = t.get("history", [])
        if not h:
            continue
        line, = ax.plot([v["epoch"] for v in h], [v["val_loss"] for v in h], label=f"seed {t['seed']} validation")
        ax.plot([v["epoch"] for v in h], [v["train_loss"] for v in h], ls="--", alpha=.6,
                color=line.get_color(), label=f"seed {t['seed']} training")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Subspace coverage loss")
    ax.set_title("Training history (not online benchmark time)")
    ax.legend(fontsize=8, ncols=2, frameon=False); ax.grid(alpha=.18)
    save_figure(fig, figures / "training_curves")
    return True


def write_report(run, manifest, summary, training, series, has_training_plot):
    c = manifest["configuration"]
    methods = summary["methods"]
    device = "CPU" if c.get("device", "cpu") == "cpu" else c["device"]
    lines = [f"# Darcy 增量粗空间实验报告：{run.name}", "",
             f"本报告由实际保存的测量记录生成。运行设备：{device}；网格 {c['n']}×{c['n']}；"
             f"每条序列 {c['steps']} 步；对比度 {c['contrast']:.0e}；右端 `{c['rhs']}`；系数类型 `{c['family']}`。", "",
             f"独立测试几何种子：{c['test_seeds']}；每个配置计时 {c['repeats']} 次；网络种子：{c.get('model_seeds', [])}。"
             f"BLAS 线程 {manifest.get('blas_threads', '未知')}，PyTorch 线程 {c.get('torch_threads', '未知')}。", "",
             "## 统计口径", "",
             "先将每个“几何—模型—重复”配置的全部时刻耗时相加，包含初始构造；再对计时重复取中位数。"
             "总表先对同一几何的模型种子取中位数，再对独立几何取中位数。几何范围与模型范围均是实际观测范围，"
             "不是置信区间。重复计时和多个模型种子没有被算作新的独立几何。", "",
             "耗时分解使用等权平均，以保证各项可相加，因此与总表中位数可以不同。"
             "`setup_seconds` 已包括局部分解与粗矩阵设置，未再次叠加其子项。", "",
             "## 整条序列的在线时间与误差", "",
             "| 方法 | 中位总时间 / s | 几何间范围 / s | 几何×模型范围 / s | 最大 PCG 次数 | 最大相对残差 | 最大相对能量误差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, s in methods.items():
        lines.append(f"| {LABELS[method]} | {s['sequence_total_seconds']['median']:.3f} | {range_text(s['sequence_total_seconds'])} | "
                     f"{range_text(s['sequence_model_total_seconds'])} | {s['max_iterations']} | {s['max_relative_residual']:.3e} | {s['max_energy_relative_error']:.3e} |")
    lines += ["", "![整条序列耗时](figures/sequence_timing.png)", "", "![在线成本分解](figures/cost_breakdown.png)", "",
              "PCG 次数包含迭代尝试的实际工作量；若迭代失败后使用直接求解恢复，恢复耗时已计入在线总时间，"
              "最终误差可能来自恢复后的解。恢复次数见下表。", "", "## 候选接纳与修复", "",
              "| 方法 | 候选通过 / 检查 | 通过率 | 谱修复次数 | 直接求解恢复次数 | 粗空间维数范围 |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, s in methods.items():
        rate = "不适用（没有候选检查）" if s["candidate_pass_rate"] is None else f"{100*s['candidate_pass_rate']:.1f}%"
        lines.append(f"| {LABELS[method]} | {s['candidate_passes_all_timed_runs']} / {s['candidate_checks_all_timed_runs']} | {rate} | "
                     f"{s['spectral_repairs_all_timed_runs']} | {s['solver_fallbacks_all_timed_runs']} | {s['coarse_dimension_min']}–{s['coarse_dimension_max']} |")
    lines += ["", "计数覆盖全部实际计时运行，包含重复与模型种子，不能直接把不同运行数量的方法按总计数排名。"
              "没有触发候选检查时通过率为不适用，不能报告为 100%。AMG 的维数是最粗层大小，"
              "Schwarz 的维数是全部拼接粗基列数，两者不是同一指标。", "", "## 与 U-Net 的配对时间比较", "",
              "比值定义为“基线总时间 / U-Net 总时间”：大于 1 表示本次观测中 U-Net 用时更少，小于 1 表示用时更多。"
              "配对先在同一几何内进行；下列数值不外推到未测场景。", "",
              "| 基线 | 配对比值中位数 | 几何间观测范围 |", "|---|---:|---:|"]
    for method, paired in summary["paired_baseline_over_unet"].items():
        s = paired["summary"]
        lines.append(f"| {LABELS[method]} | {s['median']:.3f} | {range_text(s)} |")
    if not summary["paired_baseline_over_unet"]:
        lines.append("| 未运行 U-Net | 不适用 | 不适用 |")
    lines += ["", "## 分模型结果", "", "| 方法 | 模型种子 | 独立几何数量 | 序列时间中位数 / s | 范围 / s |", "|---|---:|---:|---:|---:|"]
    for v in summary["variants"]:
        s = v["sequence_total_seconds"]
        lines.append(f"| {LABELS[v['method']]} | {'—' if v['model_seed'] == -1 else v['model_seed']} | {s['count']} | {s['median']:.3f} | {range_text(s)} |")
    lines += ["", "## 逐步行为", "", "下图每步先对重复和模型取中位数，再累加作趋势展示；其累计曲线终点不必等于整条序列先求和后的中位数。"]
    for name in series:
        lines += ["", f"![{name}](figures/{name}.png)"]
    lines += ["", "## 离线成本与参考解", ""]
    data = manifest.get("dataset_manifest") or {}
    lines.append(f"数据和标签生成用时：{number(data.get('generation_seconds'))} 秒；局部样本数：{data.get('samples', '未记录')}。")
    for t in training:
        lines.append(f"- 模型种子 {t['seed']}：训练 {number(t.get('train_seconds'))} 秒，选择第 {t.get('best_epoch')} 轮，"
                     f"验证损失 {t.get('best_validation_loss', float('nan')):.6g}，参数量 {t.get('parameters', '未记录')}。")
    if training and not all(t["benchmark_local_snapshot"] for t in training):
        lines += ["", "部分训练历史读取自当前检查点目录，并记录了文件哈希；若该目录后续发生覆盖，需对照原始基准检查点确认归属。"]
    if has_training_plot:
        lines += ["", "![训练历史](figures/training_curves.png)"]
    refs = [json.loads(p.read_text()) for p in sorted(run.glob("reference_seed*.json"))]
    if refs:
        lines += ["", f"离线直接参考解与对应组装共计 {sum(r['seconds'] for r in refs):.3f} 秒；"
                  f"参考解最大相对残差 {max(r['max_residual'] for r in refs):.3e}。"]
    lines += ["", "参考解和能量误差评估在在线计时之外，未提供给网络或求解器。参考解计时不是在线方法对照项。"
              "训练、标签生成与参考解成本未被隐入在线加速比；尚未据此声称离线成本已摊销。", "",
              "## 解释范围与待完成工作", "",
              f"这是本机 {device} 合成问题先导实验。它仅支持上述实测配置内的时间和精度结论。"
              "几何数量较少时不能据观测范围声称统计显著性、普遍加速或论文创新已经成立。", "",
              "- 谱检查为浮点数值检查，不是带舍入误差控制的严格认证。",
              "- AMG 配置固定，未穷尽调参；传统谱修复使用直接局部特征求解，尚非优化后的回收特征求解器。",
              "- 需要更多独立几何、多个对比度、不同右端和尺度实验，以及最近邻学习方法对照。",
              "- 网络收益必须相对于共享检查与回退流程的平滑误差方法单独成立；复用收益不能全部归因于 AI。",
              "- 粗空间检查成本若抵消了候选生成的节省，应如实保留负结果并调整算法。"]
    for warning in summary["warnings"]:
        lines += ["", f"数据审计提示：{warning}"]
    lines += ["", "## 可追溯文件", "", "- `measurements.json` / `measurements.csv`：逐时刻原始记录。",
              "- `manifest.json`：硬件、配置、软件版本、数据种子和源代码哈希。",
              "- `summary.json`：完整聚合规则、每次完整序列和每个几何/模型结果。",
              "- `figures/`：PNG 预览与 PDF 科学绘图。", ""]
    (run / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory with measurements.json and manifest.json")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    if not run.exists():
        run = (ROOT / args.run_dir).resolve()
    records, manifest, summary = load_and_summarize(run)
    figures = run / "figures"; figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    plot_summary(summary, figures)
    series = plot_sequences(records, summary, figures)
    training = load_training(manifest, run)
    has_training_plot = plot_training(training, figures)
    summary["training"] = [{k: v for k, v in t.items() if k != "history"} for t in training]
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(run, manifest, summary, training, series, has_training_plot)
    print(json.dumps({"report": str(run / "REPORT_zh.md"), "summary": str(run / "summary.json"),
                      "figures": str(figures), "methods": list(summary["methods"]), "warnings": summary["warnings"]}))


if __name__ == "__main__":
    main()
