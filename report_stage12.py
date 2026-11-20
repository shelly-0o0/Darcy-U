#!/usr/bin/env python3
"""Independent audit of saved Stage-12 action-search results.

No experiment, PDE solver, model or Schur module is imported. This report
reconstructs QoI metrics, combinatorial action costs and recorded objectives.
Unstored candidate predictions and solution fields cannot be independently
reconstructed by this reader; their verification remains a runner/test claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("id", "contrast")
SEARCH = ("oracle_coupled", "oracle_uncoupled", "amg_coupled")
MARKERS = ("amg_v1", "jacobi8", "recovery", "oracle_indicator")
METHODS = SEARCH + MARKERS + ("fine_direct",)
LABELS = dict(oracle_coupled="Coupled oracle*", oracle_uncoupled="Uncoupled oracle*",
    amg_coupled="AMG target + coupled", amg_v1="AMG indicator", jacobi8="Jacobi 8 indicator",
    recovery="Recovery indicator", oracle_indicator="Exact indicator*", fine_direct="Fine direct")
BASELINES = ("amg_v1", "jacobi8", "recovery", "oracle_indicator", "oracle_uncoupled", "fine_direct")
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
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"Recomputed value disagrees: {name}")


def rowkey(row):
    return row["split"], row["geometry"], row["method"]


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


@lru_cache(None)
def extra_coordinate_index(nc):
    nf=2*nc
    return {p:i for i,p in enumerate((y,x) for y in range(1,nf) for x in range(1,nf) if y%2 or x%2)}


def extra_ids(cells, nc):
    """Independent fine-interior coordinate enumeration of enrichment DOFs."""
    indices=extra_coordinate_index(nc)
    chosen=set(cells);points=set()
    for cell in chosen:
        y,x=divmod(cell,nc)
        points.add((2*y+1,2*x+1))
        if x+1<nc and cell+1 in chosen:points.add((2*y+1,2*x+2))
        if y+1<nc and cell+nc in chosen:points.add((2*y+2,2*x+1))
    return sorted(indices[p] for p in points)


def catalogue(nc):
    entries=[]
    for kind, offsets in (("single",((0,0),)),("horizontal_pair",((0,0),(0,1))),
            ("vertical_pair",((0,0),(1,0))),("block_2x2",((0,0),(0,1),(1,0),(1,1)))):
        for y in range(nc-max(p[0] for p in offsets)):
            for x in range(nc-max(p[1] for p in offsets)):
                entries.append(dict(id=len(entries),kind=kind,cells=sorted((y+dy)*nc+x+dx for dy,dx in offsets)))
    return entries


def feasible_candidates(cells, nc, budget, patches):
    before=set(cells);old=added_dofs(cells,nc);seen=set();result=[]
    for patch in patches:
        new=tuple(sorted(set(patch["cells"])-before))
        if not new or new in seen:continue
        union=sorted(before|set(new));cost=added_dofs(union,nc)-old
        if cost<=0 or old+cost>budget:continue
        seen.add(new)
        result.append(dict(id=patch["id"],new_cells=list(new),selected_cells=union,cost=cost))
    return result


def independently_mark(scores,nc,budget):
    order=sorted(range(nc*nc),key=lambda c:(-scores[c],c));chosen=set();spent=0
    for cell in order:
        y,x=divmod(cell,nc)
        neighbors=[n for allowed,n in ((x>0,cell-1),(x+1<nc,cell+1),(y>0,cell-nc),(y+1<nc,cell+nc)) if allowed]
        cost=1+sum(n in chosen for n in neighbors)
        if spent+cost<=budget:chosen.add(cell);spent+=cost
        if spent==budget:break
    fallback=spent!=budget
    if fallback:
        options=[]
        for parity in (0,1):
            cells=[c for c in order if sum(divmod(c,nc))%2==parity][:budget]
            require(len(cells)==budget,"Insufficient checkerboard cells")
            options.append((sum(scores[c] for c in cells),-parity,cells))
        chosen=set(max(options,key=lambda x:(x[0],x[1]))[2])
    return sorted(chosen),fallback


def posthoc_cost(rows):
    """Describe recorded cost after arithmetically removing the search stage.

    This is not a runnable substitute, a speed measurement, or a lower bound.
    All search-internal work (including conditional state updates) disappears;
    no replacement predictor inference or changed setup costs are charged.
    """
    lookup={rowkey(r):r for r in rows};records=[]
    for row in rows:
        if row["method"] not in SEARCH:continue
        fine=lookup[(row["split"],row["geometry"],"fine_direct")]["median_total_seconds"]
        timings=row["timings"]
        remaining=median(t["total_seconds"]-t["candidate_search_seconds"] for t in timings)
        require(remaining>=0,"Search exceeds total in cost diagnostic")
        components=sorted(k for k in timings[0] if k.endswith("_seconds") and k!="total_seconds")
        records.append(dict(split=row["split"],geometry=row["geometry"],method=row["method"],
            search_seconds=median(t["candidate_search_seconds"] for t in timings),
            remaining_seconds=remaining,remaining_over_fine=remaining/fine,
            search_fraction=median(t["candidate_search_seconds"]/t["total_seconds"] for t in timings),
            components_median={k:median(t[k] for t in timings) for k in components}))
    totals={split:{method:{metric:stat(r[metric] for r in records if r["split"]==split and r["method"]==method)
        for metric in ("search_seconds","remaining_seconds","remaining_over_fine","search_fraction")}
        for method in SEARCH} for split in SPLITS}
    return dict(post_hoc=True,no_extra_runs=True,changes_formal_gate=False,records=records,totals=totals,
        aggregation="Subtract search from total within each recorded repeat, then take median repeats within geometry. Divide that remainder by the same geometry's median fine-direct time, then aggregate across geometries. Search fraction is likewise formed per repeat before geometry medians.",
        interpretation="Hypothetical bookkeeping only: remove the entire measured candidate-search stage including conditional-state updates while retaining all other measured stages. No replacement learned inference or altered architecture/setup costs are charged. This is not an achievable acceleration, a numerical-method or hardware-independent lower bound, or evidence of a speedup.")


def aggregate(rows,frozen,audited):
    config=frozen["config"];threshold=config["paired_denominator_floor"]
    lookup={rowkey(r):r for r in rows};totals={};pairs=[];degenerate=[]
    action_lookup={(r["split"],r["geometry"],r["method"]):r for r in audited["action_rows"]}
    for split in SPLITS:
        totals[split]={}
        numbers=sorted({r["geometry"] for r in rows if r["split"]==split})
        for method in METHODS:
            entries=[lookup[(split,g,method)] for g in numbers]
            actions=[action_lookup[(split,g,method)] for g in numbers] if method in SEARCH else []
            timing_keys=sorted(set.intersection(*(set(r["timings"][0]) for r in entries))-{ "repetition" })
            record={m:stat(r[m] for r in entries) for m in METRICS}
            record.update(independent_geometries=len(entries),selected_cell_count=stat(len(r["selected_cells"]) for r in entries),
                fallback_count=sum(a["fallback"] for a in actions) if actions else sum(bool(r.get("marking",{}).get("fallback",False)) for r in entries),
                worse_than_coarse_count=sum(r["error_reduction_ratio"] is not None and r["error_reduction_ratio"]>1 for r in entries),
                timing_components_seconds={k:stat(median(t[k] for t in r["timings"]) for r in entries) for k in timing_keys},
                paired={})
            if actions:
                record["action_search"]={k:stat(a[k] for a in actions) for k in (
                    "steps","candidate_evaluations","initial_singleton_evaluations","fallback_collection_evaluations","terminal_exact_evaluations","negative_gain_steps","predicted_final_objective",
                    "actual_final_objective","prediction_actual_change_discrepancy","target_relative_error")}
                record["action_search"]["geometries_with_negative_gain_steps"]=sum(a["negative_gain_steps"]>0 for a in actions)
                record["action_search"]["final_objective_discrepancy"]=stat(a["predicted_final_objective"]-a["actual_final_objective"] for a in actions)
            for baseline in BASELINES:
                ratios=[];times=[];wins=0
                for r in entries:
                    b=lookup[(split,r["geometry"],baseline)]
                    bad=b["scaled_error_l2"]<threshold
                    value=None if baseline=="fine_direct" or bad else guarded_ratio(r["error_reduction_ratio"],b["error_reduction_ratio"])
                    tr=r["median_total_seconds"]/b["median_total_seconds"]
                    ratios.append(value);times.append(tr);wins+=int(value is not None and value<1)
                    if bad and baseline!="fine_direct" and [split,r["geometry"],baseline] not in degenerate:
                        degenerate.append([split,r["geometry"],baseline])
                    pairs.append(dict(split=split,geometry=r["geometry"],method=method,baseline=baseline,
                        error_ratio=value,time_ratio=tr,baseline_error_degenerate=bool(bad)))
                record["paired"][baseline]=dict(error_ratio=stat(ratios),time_ratio=stat(times),strict_error_wins=wins)
            totals[split][method]=record
    enough=not audited["coarse_denominator_degenerate_contexts"] and not degenerate
    criteria={}
    for split in SPLITS:
        p=totals[split]["oracle_coupled"];amg=p["paired"]["amg_v1"];cheap=p["paired"]["jacobi8"]
        def below(value,limit):return value["missing_count"]==0 and value["median"] is not None and value["median"]<=limit
        criteria[split]=dict(amg_accuracy=below(amg["error_ratio"],config["accuracy_ratio_gate"]),
            amg_wins=amg["strict_error_wins"]>=config["minimum_wins"],
            cheap_accuracy=below(cheap["error_ratio"],config["cheap_baseline_ratio_limit"]),
            no_primary_fallback=p["fallback_count"]==0)
    passed=audited["passed"] and enough and all(x for v in criteria.values() for x in v.values())
    decision=dict(formal=not frozen["smoke"],numerical_audit_passed=audited["passed"],
        nondegenerate_required_denominators=enough,criteria=criteria,headroom_gate_passed=passed,
        continue_action_information_research=passed and not frozen["smoke"],
        efficiency_gate_defined=False,ai_training_performed=False,
        note="Smoke results do not establish a formal conclusion." if frozen["smoke"] else
            ("The limited greedy action family passes the prespecified accuracy headroom gate. This is not deployability, efficiency, AI necessity or novelty evidence."
             if passed else "The limited greedy action family fails the prespecified headroom gate. Stop training an AI imitator of this oracle at this mesh and budget; this does not rule out all action families."))
    overlap={}
    for split in SPLITS:
        overlap[split]={}
        numbers=sorted({r["geometry"] for r in rows if r["split"]==split})
        for method in SEARCH:
            overlap[split][method]={}
            for baseline in ("amg_v1","oracle_indicator","oracle_uncoupled"):
                vals=[]
                for number in numbers:
                    a=set(lookup[(split,number,method)]["selected_cells"])
                    b=set(lookup[(split,number,baseline)]["selected_cells"])
                    vals.append(len(a&b)/len(a|b))
                overlap[split][method][baseline]=stat(vals)
    return dict(aggregation="One row per independent geometry and method; timing is median of three fresh repetitions. All ratios are computed within geometry before split medians. No learned models or training-seed aggregation.",
        totals=totals,paired_geometry=pairs,baseline_denominator_degenerate_contexts=degenerate,posthoc_cost=posthoc_cost(rows),
        baseline_scaled_error_threshold=threshold,mesh_jaccard=overlap,
        mechanism_comparison_scope="Coupled and uncoupled methods can follow different candidate paths; their comparison is descriptive and is not a uniquely isolated causal effect.",decision=decision)


def fmt(value,digits=4):
    return "未定义" if value is None else f"{value:.{digits}g}"


def report_text(summary,frozen,complete,audited,source):
    totals=summary["totals"];decision=summary["decision"];config=frozen["config"]
    lines=["# 第十二阶段：局部补丁动作与相互作用", "",
        "本报告由独立程序读取冻结结果生成，不导入实验运行器、Schur 实现、PDE 求解器或 AI 模型。阶段内没有训练或使用 AI 模型。", "",
        f"正式实验：**{'否（smoke）' if frozen['smoke'] else '是'}**；独立几何 **{audited['contexts']}**；方法/几何结果 **{audited['rows']}**；每项 **{config['repetitions']} 次**全新对象暖运行，共 **{complete['fresh_pipeline_runs']}** 次。",
        f"粗/适应/细参考自由度为 **{(config['nc']-1)**2}/{(config['nc']-1)**2+config['new_dofs']}/{(2*config['nc']-1)**2}**。每个适应方法恰新增 **{config['new_dofs']}** 自由度，三个目标共用一个网格和一次适应主问题求解。",
        "", "## 事前可达性门槛", "",
        f"主候选 **Coupled oracle** 的准确性可达性门槛：**{'通过' if decision['headroom_gate_passed'] else '未通过'}**。",
        "预定要求为：两组相对 AMG 的配对 E 比中位数均不超过 0.90、每组至少 12/16 个严格改善；两组相对 Jacobi8 均不超过 1；无必需分母退化、全部数值核验通过，且主候选无预算完成回退。这里的 10% 与 12/16 是开发决策阈值，不是显著性检验或发表条件。",
        "", "| 测试组 | Oracle/AMG 配对 E 比 | 严格改善数 | Oracle/Jacobi8 配对 E 比 | 主候选回退 | 门槛 |", "|---|---:|---:|---:|---:|---|"]
    for split in SPLITS:
        v=totals[split]["oracle_coupled"];a=v["paired"]["amg_v1"]
        lines.append(f"| {split} | {fmt(a['error_ratio']['median'],6)} | {a['strict_error_wins']}/{config['geometries_per_split']} | {fmt(v['paired']['jacobi8']['error_ratio']['median'],6)} | {v['fallback_count']} | {all(decision['criteria'][split].values())} |")
    lines += ["", "每个比值先在同一几何上计算，再跨独立几何取中位数，不能用两列跨几何中位数相除替代。三个重复仅用于计时与确定性核验，不增加独立样本量。",
        "", "**该门槛没有效率条件。** 主 oracle 已求得精确细主解来知道真实粗目标误差；其网格即使更好，也没有节省当前实例的求解工作。通过仅表示当前有限贪心动作族中存在值得进一步检验的信息，不能据此主张可部署、AI 必要性或方法新颖性。", "",
        "## 实际解误差与完整成本", "",
        "E 为三个目标误差按粗目标值归一化后二范数，相对于粗解同一范数的比例；粗解 E=1。能量改善不能代替目标误差改善，全部误差仅相对于固定细离散参考。",
        "", "| 方法 | ID E 中位数 [范围] | 高对比 E 中位数 [范围] | ID 能量误差比 | 高对比能量误差比 | ID 完整时间 ms | 高对比完整时间 ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        a=totals["id"][method];b=totals["contrast"][method]
        lines.append(f"| {LABELS[method]} | {fmt(a['error_reduction_ratio']['median'])} [{fmt(a['error_reduction_ratio']['minimum'])}, {fmt(a['error_reduction_ratio']['maximum'])}] | {fmt(b['error_reduction_ratio']['median'])} [{fmt(b['error_reduction_ratio']['minimum'])}, {fmt(b['error_reduction_ratio']['maximum'])}] | {fmt(a['energy_error_ratio']['median'])} | {fmt(b['energy_error_ratio']['median'])} | {fmt(1000*a['median_total_seconds']['median'])} | {fmt(1000*b['median_total_seconds']['median'])} |")
    lines += ["", "带 * 的方法使用精确细参考信息且全部实际收费。Coupled oracle 是固定候选族上的贪心搜索，不是全局最优网格；Exact indicator 是精确指标配合同一预算排序器，也不是最优动作。Fine direct 用一次细主解回答全部三个目标，零离散误差且不匹配 64 自由度预算，不能作为误差比分母。", "",
        "## 相互作用消融与可部署目标估计", "",
        "Uncoupled oracle 对每一步新出现的独立方向使用原始 Schur 主子块求增量并累加，避免重叠补丁重复计数，但忽略不同步骤之间的交叉耦合。它的末端误差仍来自真实联合网格求解。AMG target + coupled 用一次 AMG 主残差修正估计粗目标误差，其选择函数不接受精确参考；它是经典探索性诊断，不是学习模型。",
        "", "| 测试组 | 方法 | /AMG 配对 E 比 | /AMG 改善数 | /Uncoupled 配对 E 比 | /Fine direct 完整时间比 |", "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in SEARCH:
            p=totals[split][method]["paired"]
            lines.append(f"| {split} | {LABELS[method]} | {fmt(p['amg_v1']['error_ratio']['median'],6)} | {p['amg_v1']['strict_error_wins']}/{config['geometries_per_split']} | {fmt(p['oracle_uncoupled']['error_ratio']['median'],6)} | {fmt(p['fine_direct']['time_ratio']['median'],6)} |")
    lines += ["", "| 测试组 | 方法 | 目标估计相对误差 | 预测/实际变化差 | 预测末端 F | 实际末端 F |", "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in SEARCH:
            a=totals[split][method]["action_search"]
            lines.append(f"| {split} | {LABELS[method]} | {fmt(a['target_relative_error']['median'],6)} | {fmt(a['prediction_actual_change_discrepancy']['median'],6)} | {fmt(a['predicted_final_objective']['median'],6)} | {fmt(a['actual_final_objective']['median'],6)} |")
    lines += ["", "目标估计相对误差为 `||(target−e_c)/s||/max(||e_c/s||,1e-12)`；预测/实际变化差为 `||(predicted_change−actual_change)/s||/max(||actual_change/s||,1e-12)`。F 是未再除以粗误差的缩放误差平方，不能与 E 混用。AMG 行的预测 F 同时包含目标估计差异；uncoupled 的真实目标已知，其末端预测差描述被忽略的交互作用。",
        "两种 oracle 可能走不同的贪心路径、得到不同新增方向。完整策略的末端比较与同终端集合的预测偏差属于描述性机制检验，不足以确定唯一因果机制。所有逐几何值、范围与网格重合度保存在 summary.json。", "",
        "## 搜索轨迹、预算与全部负结果", "",
        "| 测试组 | 方法 | 步数中位数（范围） | 候选评价数中位数 | 含负增益步骤的几何 | 回退几何 | E>1 几何 |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in SEARCH:
            r=totals[split][method];a=r["action_search"];n=a["steps"]
            lines.append(f"| {split} | {LABELS[method]} | {fmt(n['median'])} ({fmt(n['minimum'])}–{fmt(n['maximum'])}) | {fmt(a['candidate_evaluations']['median'],6)} | {a['geometries_with_negative_gain_steps']}/{config['geometries_per_split']} | {r['fallback_count']} | {r['worse_than_coarse_count']} |")
    lines += ["", "原始动作目录含单格、横/纵相邻双格和 2×2 块；16×16 粗网格共 961 项。每步与当前集合取并集，按新增加的中心与释放的内部共享边自由度收费；去重后只评价可行候选。预算是 64 个新增自由度，不是 64 个动作或方格。所有负增益步骤和回退均保留。表中候选评价数只统计贪心循环；每次搜索另外评价初始 256 个单格和一次终端精确集合，触发回退时再评价两个棋盘集合。这些均计入完整搜索成本，并在原始结果与 summary.json 单列。", "",
        f"粗误差退化几何 **{len(audited['coarse_denominator_degenerate_contexts'])}**；经典误差分母退化记录 **{len(summary['baseline_denominator_degenerate_contexts'])}**。这些记录不丢弃、不重抽；必需分母退化会使正式门槛不通过。", "",
        "## 时间收费与独立审计边界", "",
        "CPU 与数值库均单线程。每次从新对象开始，包括系数/载荷装配、必要粗求解、完整 Schur 设置与存储、候选结构、所有搜索、真实约束网格、适应装配求解及目标评价；oracle 自己构造参考并收费，不免费借用其他方法的细分解。非 oracle 的离线评价参考不计入其在线路径。初始化、独立预热、报告与序列化单列，不把微小暖运行差异解释为速度优势。",
        f"八个分支独立预热合计 **{complete['warmup_seconds']:.6g} s**；正式循环墙钟 **{complete['experiment_wall_seconds']:.6g} s**。墙钟含离线评价与序列化，与在线路径时间之和不是同一口径。",
        "全局 Schur 是本次小网格诊断工具，不能直接外推到大规模网格；准确性门槛不含成本，但完整成本始终展示。", "",
        "| 测试组 | 方法 | 目标构建 ms | Schur 设置 ms | 搜索 ms | 适应求解 ms |", "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in SEARCH:
            timing=totals[split][method]["timing_components_seconds"]
            target_key="oracle_fine_primal_target_seconds" if method.startswith("oracle") else "amg_target_setup_apply_seconds"
            components=[target_key,"schur_setup_seconds","candidate_search_seconds","adaptive_solve_seconds"]
            lines.append(f"| {split} | {LABELS[method]} | "+" | ".join(fmt(1000*timing[k]["median"]) for k in components)+" |")
    lines += ["", "表内各分量分别取中位数，不要求相加等于总时间中位数；装配、粗求解、网格构造和诊断等其余成本也计入完整路径。全部重复的分量之和与未归类余量在 summary.json 的 audit 字段。", "",
        "### 事后成本诊断：仅删除已测搜索时间", "",
        "以下只对已有计时做算术，不新增运行、不影响正式门槛：每次重复先计算 `total_seconds−candidate_search_seconds`，再取逐几何三次中位数；与同几何细直接解的时间中位数相除后，才跨几何汇总。搜索占比也先逐重复计算，再依次取几何及组中位数。", "",
        "| 测试组 | 方法 | 搜索占比 % | 删除搜索后的余量 ms | 余量/Fine direct 配对比 |", "|---|---|---:|---:|---:|"]
    for split in SPLITS:
        for method in SEARCH:
            v=summary["posthoc_cost"]["totals"][split][method]
            lines.append(f"| {split} | {LABELS[method]} | {fmt(100*v['search_fraction']['median'],6)} | {fmt(1000*v['remaining_seconds']['median'],6)} | {fmt(v['remaining_over_fine']['median'],6)} |")
    lines += ["", "这是假设性记账：整段搜索及其内部条件状态更新等工作一起被删除，尚未加入任何替代预测器推理或新架构成本。因此，它不是实测加速、可实现的性能下界或速度优势证据。当前 Schur 设置本身约 11 ms，而细直接解完整路径约 3 ms；仅压缩搜索，仍留下明显的设置与求解成本。其他算法或规模可以改变这些成本，不能把本表当成普遍不可能性结论。逐几何记录与聚合规则在 summary.json 的 posthoc_cost 字段，可由同一报告程序重建。", "",
        f"独立审计：**{audited['frozen_input_count']} 个输入哈希**、全部输出哈希、**{audited['contexts']} 个独立几何**及历史无重叠、全部三目标误差与尺度、精确预算、指标排序和动作轨迹中的候选合法性/新增方向/目标函数/收费均通过。",
        "独立报告未重跑 PDE；精确候选数值排名、Schur 正定性、保存的能量/求解残差及完整解字段不能由仅存摘要重新证明。报告重建记录的选择轨迹与最终 Schur/实际目标变化一致性，完整数值核验由冻结运行器和数值测试完成。报告不会把记录的获胜得分当成对所有未保存候选值的独立最优性验证。", "",
        "![Actual error, complete cost and paired headroom](figures/comparison.png)", "",
        f"原始结果目录：`{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`；全部逐几何配对、动作审计与计时分量见 `summary.json`。", ""]
    return "\n".join(lines)


def plot(summary,output):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/darcy-u-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={"id":"#2478a9","contrast":"#d57725"};offset={"id":-.16,"contrast":.16}
    fig,axes=plt.subplots(2,2,figsize=(14,10.5))
    xs=np.arange(len(METHODS))
    for split in SPLITS:
        records=summary["totals"][split]
        for ax,metric,factor in ((axes[0,0],"error_reduction_ratio",1),(axes[0,1],"median_total_seconds",1000)):
            vals=[records[m][metric]["median"] for m in METHODS]
            ax.plot(xs+offset[split],[np.nan if v is None else factor*v for v in vals],"o",color=colors[split],ms=6)
            ax.set_xticks(xs);ax.set_xticklabels([LABELS[m] for m in METHODS],rotation=46,ha="right",fontsize=9)
            ax.grid(axis="y",alpha=.2)
        for ax,baseline,title in ((axes[1,0],"amg_v1","C  Paired actual-error ratio to AMG indicator"),
                (axes[1,1],"oracle_uncoupled","D  Paired actual-error ratio to uncoupled oracle")):
            for i,method in enumerate(SEARCH):
                values=[r["error_ratio"] for r in summary["paired_geometry"] if r["split"]==split and r["method"]==method and r["baseline"]==baseline and r["error_ratio"] is not None]
                x=i+offset[split];jitter=np.linspace(-.055,.055,len(values))
                ax.scatter(x+jitter,values,s=18,alpha=.50,color=colors[split],edgecolors="none")
                if values:ax.plot([x-.09,x+.09],[median(values)]*2,color=colors[split],lw=3)
            ax.axhline(1,color="#555",ls="--",lw=1);ax.grid(axis="y",alpha=.2)
            ax.set_xticks(range(len(SEARCH)));ax.set_xticklabels([LABELS[m] for m in SEARCH],fontsize=9)
            ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
            ax.set_ylabel("Within-geometry error ratio (lower is better)")
    axes[0,0].axhline(1,color="#555",ls="--",lw=1)
    axes[0,0].set_title("A  Actual QoI error after equal-DOF refinement",loc="left",fontweight="bold",fontsize=11)
    axes[0,0].set_ylabel("Scaled three-QoI error / coarse error")
    axes[0,1].set_title("B  Complete cost, including oracle reference",loc="left",fontweight="bold",fontsize=11)
    axes[0,1].set_ylabel("Median fresh pipeline time (ms; log scale)");axes[0,1].set_yscale("log")
    axes[1,0].axhline(.9,color="#777",ls=":",lw=1)
    handles=[Line2D([0],[0],marker="o",linestyle="",color=colors[s],label="ID" if s=="id" else "High contrast") for s in SPLITS]
    fig.legend(handles=handles,loc="upper center",ncol=2,bbox_to_anchor=(.5,.953),frameon=False)
    fig.suptitle("Patch-action feasibility: fixed 64 new DOFs, no learned model",fontsize=16,y=.99)
    fig.text(.5,.015,"Each dot in C/D is one independent geometry. *Oracle methods use and pay for fine reference information.\nA greedy path is not a globally optimal mesh. Accuracy headroom is not efficiency or deployability.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.095,top=.90,hspace=.67,wspace=.28)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):fig.savefig(output/f"comparison.{ext}",dpi=180)
    plt.close(fig)


def audit_search(row,nc,budget,tolerance):
    search=row["search"];method=row["method"];patches=catalogue(nc)
    require(search["coupled"]==(method!="oracle_uncoupled"),"Wrong coupling mode")
    require(search["candidate_catalogue_size"]==len(patches),"Wrong action catalogue size")
    target=np.asarray(search["target_error"]);scales=np.asarray(search["scales"])
    close(scales,row["qoi_scales"],"search QoI scales")
    require(target.shape==(3,),"Wrong action target shape")
    ec=np.asarray(row["signed_coarse_errors"])
    if method.startswith("oracle"):
        close(target,ec,"oracle target is true coarse error",atol=1e-16)
        require(row["oracle_label_relative_residual"] is not None and row["oracle_label_relative_residual"]<=1e-8,
            "Oracle target fine solve residual failed")
    else:
        require(row["oracle_label_relative_residual"] is None,"AMG target includes oracle label")
    chosen=[];ids=[];predicted=np.zeros(3);negative=0;evaluated=0
    for number,step in enumerate(search["trace"]):
        require(step["step"]==number,"Nonconsecutive search steps")
        candidates=feasible_candidates(chosen,nc,budget,patches)
        require(step["candidate_count"]==len(candidates),"Wrong feasible distinct candidate count")
        byid={c["id"]:c for c in candidates}
        require(step["patch_id"] in byid,"Chosen patch is not a valid distinct feasible action")
        winning=byid[step["patch_id"]]
        require(step["kind"]==patches[step["patch_id"]]["kind"] and step["new_cells"]==winning["new_cells"],
            "Chosen patch kind/cells mismatch")
        union=winning["selected_cells"];newids=sorted(set(extra_ids(union,nc))-set(ids))
        require(step["new_extra_indices"]==newids and step["cost"]==winning["cost"]==len(newids),
            "Action extra-DOF IDs or marginal cost mismatch")
        require(not set(newids)&set(ids),"Extra direction double-counted")
        next_prediction=predicted+np.asarray(step["predicted_increment"])
        close(predicted,step["predicted_qoi_change_before"],"prior cumulative change",atol=1e-16)
        close(next_prediction,step["predicted_qoi_change_after"],"incremental/cumulative change",atol=1e-16)
        before=float(np.sum(((target-predicted)/scales)**2))
        after=float(np.sum(((target-next_prediction)/scales)**2))
        close(before,step["objective_before"],"action objective before")
        close(after,step["objective_after"],"action objective after")
        close((before-after)/len(newids),step["gain_per_dof"],"signed gain per new DOF")
        require(step["negative_gain"]==(step["gain_per_dof"]<0),"Negative-gain step misclassified")
        chosen=union;ids=extra_ids(chosen,nc);predicted=np.asarray(step["predicted_qoi_change_after"])
        require(step["spent"]==len(ids)==added_dofs(chosen,nc)<=budget,"Cumulative budget mismatch")
        negative+=step["negative_gain"];evaluated+=len(candidates)
    partial=search["partial_before_fallback"]
    require(partial["selected_cells"]==chosen and partial["added_dofs"]==len(ids),"Partial search endpoint changed")
    close(partial["predicted_qoi_change"],predicted,"partial search prediction",atol=1e-16)
    require(search["negative_predicted_gain_steps"]==negative and search["greedy_candidate_evaluations"]==evaluated,
        "Wrong negative-gain or candidate evaluation count")
    require(search["initial_singleton_evaluations"]==nc*nc and search["terminal_exact_evaluations"]==1 and
        search["fallback_collection_evaluations"]==(2 if search["fallback"] else 0),"Wrong additional search evaluation counts")
    require(search["fallback"]==(len(ids)!=budget),"Wrong completion fallback flag")
    require(len(search["initial_singleton_scores"])==nc*nc,"Missing initial singleton scores")
    if search["fallback"]:
        require(not feasible_candidates(chosen,nc,budget,patches),"Fallback despite a feasible next action")
        scores=search["initial_singleton_scores"]
        rank=sorted(range(nc*nc),key=lambda c:(-scores[c],c))
        options=search["fallback_options"]
        require(len(options)==2 and {o["parity"] for o in options}=={0,1},"Wrong fallback options")
        for option in options:
            expect=sorted([c for c in rank if sum(divmod(c,nc))%2==option["parity"]][:budget])
            require(option["selected_cells"]==expect and added_dofs(expect,nc)==budget,"Wrong fallback ranking/budget")
            close(option["objective"],np.sum(((target-np.asarray(option["predicted_qoi_change"]))/scales)**2),"fallback objective")
        picked=min(options,key=lambda o:(o["objective"],o["parity"]))
        chosen=picked["selected_cells"];predicted=np.asarray(picked["predicted_qoi_change"])
    else:
        require(search["fallback_options"]==[],"Unexpected fallback options")
    require(chosen==row["selected_cells"]==search["selected_cells"],"Search/actual mesh endpoint mismatch")
    require(search["added_dofs"]==budget and search["final_extra_indices"]==extra_ids(chosen,nc),"Final Schur DOFs differ from mesh")
    close(search["predicted_qoi_change"],predicted,"final prediction",atol=1e-16)
    actual=np.asarray(row["qoi"])-np.asarray(row["coarse_qoi"])
    exact=np.asarray(search["exact_schur_qoi_change"])
    close(row["actual_qoi_change"],actual,"actual QoI change",atol=1e-18)
    identity=float(np.linalg.norm((actual-exact)/scales)/max(np.linalg.norm(actual/scales),1e-12))
    close(identity,row["schur_identity_relative_error"],"Schur/FEM identity error")
    require(identity<=tolerance,"Schur prediction does not agree with actual refinement")
    path_error=float(np.linalg.norm((predicted-exact)/scales)/max(np.linalg.norm(exact/scales),1e-12))
    close(path_error,search["ignored_coupling_relative_change_error"],"path/direct Schur discrepancy")
    if search["coupled"]:require(path_error<=tolerance,"Coupled incremental/direct formula failed")
    require(row["schur_dimension"]==(2*nc-1)**2-(nc-1)**2 and row["schur_symmetry_defect"]<=1e-10 and
        row["coarse_relative_residual"]<=1e-9,"Schur dimension/symmetry/coarse residual failed")
    result=dict(split=row["split"],geometry=row["geometry"],method=method,
        steps=len(search["trace"]),candidate_evaluations=evaluated,
        initial_singleton_evaluations=search["initial_singleton_evaluations"],
        fallback_collection_evaluations=search["fallback_collection_evaluations"],
        terminal_exact_evaluations=search["terminal_exact_evaluations"],negative_gain_steps=negative,fallback=search["fallback"],
        schur_identity_relative_error=identity,incremental_exact_change_discrepancy=path_error,
        predicted_final_objective=float(np.sum(((target-predicted)/scales)**2)),
        actual_final_objective=row["scaled_error_l2"]**2,
        prediction_actual_change_discrepancy=float(np.linalg.norm((predicted-actual)/scales)/max(np.linalg.norm(actual/scales),1e-12)),
        target_relative_error=float(np.linalg.norm((target-ec)/scales)/max(np.linalg.norm(ec/scales),1e-12)))
    if method=="oracle_coupled":close(result["predicted_final_objective"],result["actual_final_objective"],"oracle search/final objective")
    return result


def audit(source):
    frozen=read(source/"freeze.json");complete=read(source/"complete.json");config=frozen["config"]
    require(complete["complete"] and complete["smoke"]==frozen["smoke"],"Incomplete run or smoke mismatch")
    require(frozen["no_training"] and complete["no_training"],"Stage-12 unexpectedly trained a model")
    require(sha(source/"freeze.json")==complete["freeze_sha256"],"Frozen manifest changed")
    require(set(complete["output_sha256"])=={"rows.json","contexts.json"},"Incomplete result hashes")
    for name,digest in complete["output_sha256"].items():require(sha(source/name)==digest,f"Output changed: {name}")
    for name,digest in frozen["input_sha256"].items():
        path=Path(name)
        require(not path.is_absolute() and ".." not in path.parts,f"Nonportable input path: {name}")
        require(sha(ROOT/path)==digest,f"Frozen input changed: {name}")
    require(frozen["methods"]==list(METHODS),"Wrong method set/order")
    require(set(config["splits"])==set(SPLITS),"Wrong split set")
    if not frozen["smoke"]:
        required=dict(nc=16,new_dofs=64,geometries_per_split=16,repetitions=3,splits={"id":53000,"contrast":53100},
            source=0,goals=[1,2,3],order_seed=53991,warmup_geometry=53950,near_zero_coarse_error=1e-12,
            paired_denominator_floor=1e-12,identity_tolerance=1e-8,accuracy_ratio_gate=.90,minimum_wins=12,
            cheap_baseline_ratio_limit=1.,primary_method="oracle_coupled",primary_fallback_invalidates_gate=True)
        for key,value in required.items():require(config[key]==value,f"Formal configuration changed: {key}")
    require(frozen["environment"]["device"]=="cpu" and frozen["environment"]["torch_threads"]==1,"Not CPU single-thread")
    require(all(p["num_threads"]==1 for p in frozen["environment"]["threadpools"]),"Numerical library threading changed")
    history=set()
    for name in ("training_data.json","test_data.json"):
        data=read(ROOT/"results/stage10_pilot"/name)
        entries=data if isinstance(data,list) else data["train"]+data["validation"]
        history.update(r["coefficient_sha256"] for r in entries)
    history.update(r["coefficient_sha256"] for r in read(ROOT/"results/stage11_refinement/contexts.json"))
    rows=read(source/"rows.json");contexts=read(source/"contexts.json")
    require(finite(rows) and finite(contexts) and finite(frozen) and finite(complete),"Nonfinite saved value")
    expected={(split,g) for split,first in config["splits"].items() for g in range(first,first+config["geometries_per_split"])}
    expected_keys={(s,g,m) for s,g in expected for m in METHODS}
    require(len(rows)==len(expected_keys)==complete["rows"] and {rowkey(r) for r in rows}==expected_keys,"Missing/duplicate method or geometry")
    require(len(contexts)==len(expected)==complete["contexts"] and {(r["split"],r["geometry"]) for r in contexts}==expected,"Missing/duplicate context")
    require(complete["fresh_pipeline_runs"]==len(rows)*config["repetitions"],"Wrong number of fresh timing runs")
    hashes=[c["coefficient_sha256"] for c in contexts]
    require(len(set(hashes))==len(hashes) and not set(hashes)&history,"Current/history geometry overlap")
    nc=config["nc"];budget=config["new_dofs"];near_zero=set();reference={};space_hashes={};actions=[]
    maxima=dict(solve_relative_residual=0.,projection_orthogonality=0.,coarse_embedding=0.,energy_ratio=0.,
        schur_identity_relative_error=0.,coupled_path_error=0.,schur_symmetry_defect=0.,coarse_relative_residual=0.)
    variability=[];remainders=[]
    for c in contexts:
        require(c["nc"]==nc and c["fine_dofs"]==(2*nc-1)**2 and c["coarse_dofs"]==(nc-1)**2 and c["new_dofs"]==budget,"Wrong context dimensions")
        require(c["fine_relative_residual"]<=1e-8,"Fine-reference residual failed")
    for row in rows:
        method=row["method"];key=rowkey(row)
        require(row["training_seed"] is None and row["offline_only"]==method.startswith("oracle"),"Training/oracle classification changed")
        require(len(row["solution_sha256"])==64 and all(c in "0123456789abcdef" for c in row["solution_sha256"]),"Invalid solution fingerprint")
        q,qf,qc,scales=(np.asarray(row[k],dtype=float) for k in ("qoi","fine_qoi","coarse_qoi","qoi_scales"))
        require(all(v.shape==(3,) for v in (q,qf,qc,scales)) and np.all(scales>0),"Invalid QoI data")
        if key[:2] in reference:
            for a,b in zip((qf,qc,scales),reference[key[:2]]):close(a,b,"shared reference and scales")
        else:reference[key[:2]]=(qf,qc,scales)
        close(scales,np.maximum(abs(qc),max(float(abs(qc).max())*1e-12,1e-30)),"online coarse scales",atol=1e-25)
        error=qf-q;coarse=qf-qc;scaled=float(np.linalg.norm(error/scales));denom=float(np.linalg.norm(coarse/scales))
        close(error,row["signed_qoi_errors"],"signed actual QoI errors",atol=1e-20)
        close(coarse,row["signed_coarse_errors"],"signed coarse QoI errors",atol=1e-20)
        close(np.linalg.norm(error),row["absolute_error_l2"],"absolute error",atol=1e-20)
        close(scaled,row["scaled_error_l2"],"scaled error");close(denom,row["coarse_scaled_error"],"scaled coarse error")
        degenerate=denom<config["near_zero_coarse_error"]
        require(row["coarse_error_near_zero"]==degenerate,"Wrong near-zero denominator flag")
        if degenerate:
            near_zero.add(key[:2]);require(row["error_reduction_ratio"] is None,"Near-zero ratio must be null")
        else:close(scaled/denom,row["error_reduction_ratio"],"actual error reduction")
        require(0<=row["energy_error_ratio"]<=1+1e-8 and row["relative_residual"]<=1e-8,"Energy/solve verification failed")
        maxima["solve_relative_residual"]=max(maxima["solve_relative_residual"],row["relative_residual"])
        maxima["energy_ratio"]=max(maxima["energy_ratio"],row["energy_error_ratio"])
        if method=="fine_direct":
            require(row["selected_cells"]==[] and row["solve_dofs"]==(2*nc-1)**2 and not row["budget_matched"] and
                row["added_dofs"]==(2*nc-1)**2-(nc-1)**2,"Fine direct incorrectly budget-matched")
            close(q,qf,"fine direct/reference",atol=1e-16)
        else:
            cells=row["selected_cells"]
            require(cells==sorted(set(cells)) and all(0<=c<nc*nc for c in cells),"Invalid selected cells")
            require(added_dofs(cells,nc)==row["added_dofs"]==budget and row["solve_dofs"]==(nc-1)**2+budget and row["budget_matched"],"Unequal DOF budget")
            digest=row["interpolation_sha256"]
            require(len(digest)==64 and all(c in "0123456789abcdef" for c in digest),"Invalid interpolation fingerprint")
            if tuple(cells) in space_hashes:require(space_hashes[tuple(cells)]==digest,"Same cells gave different interpolation fingerprints")
            else:space_hashes[tuple(cells)]=digest
            require(row["geometric_triangle_count"]==2*nc*nc+6*len(cells),"Wrong triangle count")
            maxima["coarse_embedding"]=max(maxima["coarse_embedding"],row["coarse_embedding_max_abs"])
            require(row["coarse_embedding_max_abs"]<=1e-12,"Coarse space not embedded")
            if method in SEARCH:
                action=audit_search(row,nc,budget,config["identity_tolerance"]);actions.append(action)
                maxima["schur_identity_relative_error"]=max(maxima["schur_identity_relative_error"],action["schur_identity_relative_error"])
                if row["search"]["coupled"]:maxima["coupled_path_error"]=max(maxima["coupled_path_error"],action["incremental_exact_change_discrepancy"])
                for metric in ("schur_symmetry_defect","coarse_relative_residual"):maxima[metric]=max(maxima[metric],row[metric])
            else:
                scores=row["scores"]
                require(len(scores)==nc*nc and min(scores)>=0,"Invalid marker scores")
                chosen,fallback=independently_mark(scores,nc,budget)
                require(chosen==cells and fallback==row["marking"]["fallback"],"Indicator ranking rule mismatch")
                require(row["marking"]["selected_cell_count"]==len(cells) and row["marking"]["added_dofs"]==budget,"Wrong marker metadata")
                close(sum(scores[c] for c in cells),row["marking"]["score_sum"],"marker score sum")
                orth=row["projection_orthogonality"]
                require(orth is not None and orth<=1e-8,"Marker coarse projection failed")
                maxima["projection_orthogonality"]=max(maxima["projection_orthogonality"],orth)
        times=row["timings"]
        require(len(times)==config["repetitions"] and {t["repetition"] for t in times}==set(range(config["repetitions"])),"Missing/duplicate timings")
        totals=[]
        for t in times:
            require(all(v>=0 for k,v in t.items() if k!="repetition"),"Negative stage timing")
            total=t["total_seconds"];totals.append(total)
            components=sum(v for k,v in t.items() if k.endswith("_seconds") and k!="total_seconds")
            require(total+1e-10>=components,"Timing stage sum exceeds complete path")
            if method in SEARCH:
                require(t["schur_setup_seconds"]>0 and t["candidate_search_seconds"]>0,"Search cost uncharged")
                target_key="oracle_fine_primal_target_seconds" if method.startswith("oracle") else "amg_target_setup_apply_seconds"
                require(t[target_key]>0,"Target construction uncharged")
            remainders.append(dict(split=key[0],geometry=key[1],method=method,repetition=t["repetition"],
                component_sum_seconds=components,unclassified_seconds=total-components))
        close(median(totals),row["median_total_seconds"],"median fresh complete path time")
        variability.append(max(totals)/min(totals))
    audited=dict(passed=True,rows=len(rows),contexts=len(contexts),independent_geometries=len(hashes),
        fresh_pipeline_runs=complete["fresh_pipeline_runs"],frozen_input_count=len(frozen["input_sha256"]),
        zero_train_validation_history_overlap=True,exact_dof_budget=True,reconstructed_qoi_metrics=True,
        reconstructed_marker_sorting=True,reconstructed_action_catalogue_and_candidates=True,
        reconstructed_action_dof_ids_and_costs=True,reconstructed_recorded_objectives_and_endpoints=True,
        candidate_numeric_argmax_independently_verified=False,interpolation_fingerprint_consistency=True,
        coarse_denominator_degenerate_contexts=[list(k) for k in sorted(near_zero)],numerical_maxima=maxima,
        within_row_timing_max_over_min=stat(variability),action_rows=actions,timing_component_sums_and_unclassified=remainders,
        limitation="No PDE or Schur recomputation: stored scalar energy/residual values are range-checked, final QoI/Schur changes are cross-checked, recorded trajectories and combinatorial feasibility are reconstructed. Non-winning candidate values and full solution fields were not stored, so independent numerical argmax and energy re-evaluation are not claimed.")
    return rows,contexts,frozen,complete,audited


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",default="results/stage12_actions")
    parser.add_argument("--output",default="results/stage12_report");parser.add_argument("--no-plot",action="store_true")
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
    compact={k:v for k,v in audited.items() if k not in ("timing_component_sums_and_unclassified","action_rows")}
    print(json.dumps(dict(output=str(output),audit=compact,decision=summary["decision"]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
