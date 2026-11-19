#!/usr/bin/env python3
"""Audit a prespecified persistent-space hypothesis from real saved PDE runs.

No PDE is assembled or solved here. Development is a numerical gate, not a
performance selector. All observations and full online costs are retained.
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

ROOT=Path(__file__).resolve().parent
RTOL,STEPS=1e-8,20
SOURCES=["fields.py","numerics.py","sequence_baselines.py","balanced_augmentation.py","cached_pcg.py","persistent_recycling_experiment.py"]
BASELINES=["baseline_scipy","baseline_event_custom"]
METHODS=[*BASELINES,"frozen_event_warm4","frozen_all_balanced4","frozen_all_warm4"]
HYPOTHESIS="frozen_all_balanced4"
EVENT_STEPS=[3,6,9,10,12,15,18]
CATEGORIES=["initial","event","non_event"]
COMPONENTS=["assembly_seconds","guess_seconds","setup_seconds","correction_seconds",
            "solve_seconds","maintenance_seconds","recovery_seconds"]
QUANTITIES=["total_seconds",*COMPONENTS,"other_seconds","extraction_seconds","solve_plus_maintenance_seconds","iterations"]
COUNTS=["pcg_failures","recoveries","final_failures","cache_rejections"]
MECHANISM_PAIRS=[("frozen_all_balanced4","frozen_event_warm4"),
                 ("frozen_all_balanced4","frozen_all_warm4"),
                 ("frozen_event_warm4","frozen_all_warm4")]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def dist(values):
    values=list(values);valid=[float(v) for v in values if finite(v)]
    return dict(count=len(values),finite_count=len(valid),median=median(valid) if valid else None,
                min=min(valid) if valid else None,max=max(valid) if valid else None)


def ratio(a,b): return a/b if finite(a) and finite(b) and b>0 else None
def f(value,digits=3): return "不适用" if value is None else f"{value:.{digits}f}"
def sci(value): return "不适用" if value is None else f"{value:.3e}"


def expected_specs():
    return [dict(name=name,policy=policy,event_custom=name!="baseline_scipy") for name,policy in zip(METHODS,
            ["none","none","event_warm","all_balanced","all_warm"])]


def check_hashes(root,value,where):
    require(set(value["source_sha256"])==set(SOURCES),f"Source inventory differs: {where}")
    checks={"source_sha256":{name:sha(root/name)==digest for name,digest in value["source_sha256"].items()}}
    require(all(checks["source_sha256"].values()),f"Frozen source differs: {where}")
    for key,name in [("protocol_sha256","docs/stage7_protocol_zh.md"),("splits_sha256","data/stage7_splits.json")]:
        checks[key]=sha(root/name)==value[key]
        require(checks[key],f"Frozen {key} differs: {where}")
    return checks


def check_splits(root,splits):
    groups=splits["groups"]
    require(set(groups)=={"development","confirmation"},"Split groups differ")
    require(len(groups["development"])==3 and len(groups["confirmation"])==5,"Split sizes differ")
    seeds=groups["development"]+groups["confirmation"]
    require(len(set(seeds))==8,"Repeated stage-7 seed")
    hashes=[splits["geometry_sha256"][str(seed)] for seed in seeds]
    require(len(set(hashes))==8,"Repeated stage-7 geometry hash")
    require(not set(seeds).intersection(splits["excluded_seeds"]),"Excluded seed reused")
    require(75003 in splits["excluded_seeds"],"Functional-check geometry is not excluded")
    previous=read(root/"data/stage6_splits.json")
    require(set(previous["excluded_seeds"]+sum(previous["groups"].values(),[])).issubset(splits["excluded_seeds"]),"Earlier exclusions incomplete")
    require(not set(hashes).intersection(previous.get("geometry_sha256",{}).values()),"Saved geometry overlaps stage 6")
    return dict(development_geometries=3,confirmation_geometries=5,confirmation_256_reuses_first_two=True,
                previous_split_sha256=sha(root/"data/stage6_splits.json"),
                note="Checks saved split metadata and available prior geometry hashes; no fields or PDEs are regenerated.")


def load_online(root,name,splits,freeze_hash=None):
    folder=root/"results"/name
    manifest,rows,complete,refs=[read(folder/p) for p in ["manifest.json","measurements.json","complete.json","references.json"]]
    development=name=="stage7_development";n=256 if name=="stage7_confirm256" else 128
    seeds=splits["groups"]["development" if development else "confirmation"]
    if n==256: seeds=seeds[:2]
    repeats=2 if development else 3;config=manifest["configuration"]
    require(config["n"]==n and config["seeds"]==seeds and config["repeats"]==repeats,f"Schedule differs: {name}")
    require(config["specs"]==expected_specs(),f"Prespecified five methods differ: {name}")
    require(config["freeze_sha256"]==(None if development else freeze_hash),f"Freeze hash differs: {name}")
    for key,expected in [("steps",20),("rtol",RTOL),("contrast",1e4),("maxiter",500),("blas_threads",1),("device","cpu"),("rank",4)]:
        require(manifest.get(key)==expected,f"Unexpected {key}: {name}")
    hashes=check_hashes(root,manifest,name)
    expected={(n,method,seed,repeat,step) for method in METHODS for seed in seeds for repeat in range(repeats) for step in range(STEPS)}
    keys=[(r["n"],r["method"],r["seed"],r["repeat"],r["step"]) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)==expected,f"Missing, duplicate or unexpected row: {name}")
    require(complete.get("complete") is True and complete.get("rows")==len(rows),f"Completion marker differs: {name}")
    grouped=defaultdict(list);event_labels=defaultdict(set)
    for row in rows:
        method=row["method"];recycled=method in METHODS[2:]
        for key in ["total_seconds",*COMPONENTS,"extraction_seconds","relative_residual","final_relative_residual","energy_relative_error",
                    "base_relative_residual","corrected_relative_residual","orthogonality_error"]:
            require(finite(row[key]) and row[key]>=0,f"Invalid {key}: {name}")
        require(row["initial_step"]==(row["step"]==0),"Initial-step flag differs")
        require(row["refresh_event"]==(row["step"]==0 or row["step"] in EVENT_STEPS),"Input event schedule differs")
        require(row["final_converged"]==(row["final_relative_residual"]<=RTOL),"Final convergence flag inconsistent")
        require(row["converged"]==(row["relative_residual"]<=RTOL),"PCG convergence flag inconsistent")
        require(row["solver_fallback"]==(not row["converged"]),"Recovery flag inconsistent")
        require(isinstance(row["iterations"],int) and 0<=row["iterations"]<=500,"Iteration budget exceeded")
        require(isinstance(row["restarts"],int) and 0<=row["restarts"]<=3,"Refresh budget exceeded")
        require(sum(row[key] for key in COMPONENTS)<=row["total_seconds"]+1e-8,"Overlapping/excess component times")
        require(row["extraction_seconds"]<=row["maintenance_seconds"]+1e-8,"Extraction is not inside maintenance")
        require(all(isinstance(row[key],int) and 0<=row[key]<=4 for key in ["rank","old_space_rank","stored_space_rank"]),"Rank budget exceeded")
        if row["initial_step"]: require(row["old_space_rank"]==row["rank"]==0,"First step uses historical space")
        if not recycled:
            require(row["old_space_rank"]==row["rank"]==row["stored_space_rank"]==0,"Baseline uses a space")
            require(row["initial_space_sha256"] is None,"Baseline unexpectedly has a space hash")
        else:
            digest=row["initial_space_sha256"]
            require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),"Invalid first-space hash")
        expected_operator="balanced" if row["rank"]>0 and (method=="frozen_all_balanced4" or (method=="frozen_event_warm4" and not row["refresh_event"])) else "base"
        require(row["solver_operator"]==expected_operator,"Recorded operator does not match prescribed interface")
        capture_expected=bool(recycled and row["initial_step"])
        require(row["capture_requested"]==row["extraction_requested"]==capture_expected,"Capture/extraction outside first solve")
        cap=row["capture_stats"]
        custom_expected=row["refresh_event"] and method!="baseline_scipy"
        require(bool(cap)==custom_expected,"Missing/unexpected custom-solver metadata")
        if cap:
            require(cap["max_vectors"]==(24 if capture_expected else 0),"Capture capacity differs")
            require(0<=cap["captured_vectors"]<=cap["max_vectors"],"Captured vectors exceed capacity")
            require(cap["extra_a_applications_for_cache"]==cap["extra_m_applications_for_cache"]==0,"Collection adds operator actions")
            require(len(cap["segment_ids"])==cap["captured_vectors"],"Capture segment metadata incomplete")
            require(all(0<=segment<=row["restarts"] for segment in cap["segment_ids"]),"Invalid capture segment id")
            if capture_expected: require(row["solver_operator"]=="base","Only M-trajectory collection is permitted")
        extraction=row["extraction"]
        require(bool(extraction)==capture_expected,"Extraction record differs from first-step policy")
        require(row["cache_rejected"]==bool(extraction and extraction["status"] not in ("accepted","empty_basis")),"Cache rejection flag inconsistent")
        if extraction:
            require(extraction["requested_rank"]==4 and extraction["a_applications"]==extraction["m_applications"]==0,"Extraction rank/operator budget differs")
        else: require(row["extraction_seconds"]==0,"Unrequested extraction time")
        grouped[(method,row["seed"],row["repeat"])].append(row)
        event_labels[(row["seed"],row["step"])].add((row["initial_step"],row["refresh_event"],row["event"]))
    require(all(len(values)==1 for values in event_labels.values()),"Input event metadata differs across methods")
    source_records=[]
    for (method,seed,repeat),group in grouped.items():
        sequence=sorted(group,key=lambda r:r["step"])
        require(len({r["initial_space_sha256"] for r in sequence})==1,"Space audit hash changes within sequence")
        previous=0
        for row in sequence:
            require(row["old_space_rank"]==previous,"Space rank history is discontinuous")
            next_rank=row["extraction"]["rank"] if row["extraction"] else previous
            require(row["stored_space_rank"]==next_rank,"Stored rank changed without first-step extraction")
            previous=next_rank
        if method in METHODS[2:]:
            source_records.append(dict(n=n,method=method,seed=seed,repeat=repeat,sha256=sequence[0]["initial_space_sha256"],rank=sequence[0]["stored_space_rank"]))
    shared=[]
    for seed in seeds:
        for repeat in range(repeats):
            local=[r for r in source_records if r["seed"]==seed and r["repeat"]==repeat]
            require(len(local)==3,"Initial-space source inventory incomplete")
            shared.append(dict(seed=seed,repeat=repeat,all_three_byte_identical=len({r["sha256"] for r in local})==1,
                               method_sha256={r["method"]:r["sha256"] for r in local}))
    require(len(refs)==len(seeds) and {r["seed"] for r in refs}==set(seeds),"Reference inventory differs")
    for ref in refs:
        require(len(ref["relative_residuals"])==20 and finite(ref["seconds"]) and ref["seconds"]>=0,"Reference provenance invalid")
        require(all(finite(v) and v>=0 for v in ref["relative_residuals"]),"Reference residual invalid")
    return manifest,rows,dict(expected_rows=len(expected),actual_rows=len(rows),duplicate_rows=0,frozen_hash_checks=hashes,
        operator_audit_scope="Checks recorded solver_operator against policy; actual object passing is supported by source review and 100 small-grid identity-check solves, not per-call identity instrumentation in timed runs.",
        input_sha256={p.name:sha(p) for p in sorted(folder.glob("*.json"))},
        reference_relative_residual=dist(v for r in refs for v in r["relative_residuals"]),
        all_reference_accurate=all(v<=RTOL for r in refs for v in r["relative_residuals"]),
        offline_reference_seconds=sum(r["seconds"] for r in refs),
        first_space_audit=dict(sequences=source_records,matched_seed_repeat_groups=shared,
            all_three_methods_byte_identical=all(g["all_three_byte_identical"] for g in shared),
            byte_mismatch_groups=sum(not g["all_three_byte_identical"] for g in shared),
            note="Hashes are computed after all sequence timing and never used in decisions. Byte disagreement does not prove different mathematical subspaces; retain it for sign/rotation equivalence review."))


def category(row):
    return "initial" if row["initial_step"] else "event" if row["refresh_event"] else "non_event"


def totals(rows):
    values={key:sum(r[key] for r in rows) for key in ["total_seconds",*COMPONENTS,"extraction_seconds","iterations"]}
    values["other_seconds"]=values["total_seconds"]-sum(values[key] for key in COMPONENTS)
    values["solve_plus_maintenance_seconds"]=values["solve_seconds"]+values["maintenance_seconds"]
    values.update(steps=len(rows),pcg_failures=sum(not r["converged"] for r in rows),recoveries=sum(r["solver_fallback"] for r in rows),
                  final_failures=sum(not r["final_converged"] for r in rows),cache_rejections=sum(r["cache_rejected"] for r in rows))
    return values


def compare(methods,numerator,denominator):
    pairs=[]
    for seed,left in methods[numerator]["by_geometry"].items():
        right=methods[denominator]["by_geometry"][seed]
        pair=dict(seed=int(seed),categories={})
        for key in QUANTITIES:
            a,b=left[key]["median"],right[key]["median"]
            pair[key+"_ratio"]=ratio(a,b)
            pair[key+"_difference"]=a-b
        for group in CATEGORIES:
            pair["categories"][group]={}
            for key in QUANTITIES:
                a,b=left["categories"][group][key]["median"],right["categories"][group][key]["median"]
                pair["categories"][group][key+"_ratio"]=ratio(a,b)
                pair["categories"][group][key+"_difference"]=a-b
        pairs.append(pair)
    keys=[key for key in pairs[0] if key not in ("seed","categories")]
    return dict(numerator=numerator,denominator=denominator,by_geometry=pairs,
        **{key:dist(p[key] for p in pairs) for key in keys},
        categories={group:{key:dist(p["categories"][group][key] for p in pairs) for key in keys} for group in CATEGORIES})


def fixed_trajectory_cost_scenario(sequences,methods,fastest):
    """Post hoc accounting only; never changes an algorithm or selection rule."""
    scenarios={}
    for name in METHODS[2:]:
        local=[dict(seed=s["seed"],repeat=s["repeat"],total_seconds=s["total_seconds"],
                    correction_seconds=s["correction_seconds"],extraction_seconds=s["extraction_seconds"],
                    adjusted_seconds=s["total_seconds"]-s["correction_seconds"]-s["extraction_seconds"])
               for s in sequences if s["method"]==name]
        geometry={seed:dist(s["adjusted_seconds"] for s in local if str(s["seed"])==seed)
                  for seed in methods[name]["by_geometry"]}
        comparisons={}
        for baseline in BASELINES:
            points=[dict(seed=int(seed),numerator_seconds=value["median"],
                         denominator_seconds=methods[baseline]["by_geometry"][seed]["total_seconds"]["median"],
                         time_ratio=ratio(value["median"],methods[baseline]["by_geometry"][seed]["total_seconds"]["median"]))
                    for seed,value in geometry.items()]
            comparisons[baseline]=dict(denominator=baseline,by_geometry=points,
                                      time_ratio=dist(p["time_ratio"] for p in points))
        scenarios[name]=dict(complete_sequences=local,by_geometry=geometry,
            adjusted_seconds=dist(value["median"] for value in geometry.values()),
            versus_each_baseline=comparisons,
            versus_fixed_fastest_valid_baseline=comparisons[fastest] if fastest else None)
    return dict(post_hoc=True,participates_in_selection=False,fixed_primary=HYPOTHESIS,
        formula="per complete sequence: total_seconds - correction_seconds - extraction_seconds",
        aggregation="subtract within each full sequence, median repeats within geometry, then pair with each baseline's geometry repeat median",
        fastest_valid_baseline_by_overall_median=fastest,
        scope="Measured trajectories, iteration counts and all other costs remain fixed. This is neither an executable algorithm nor a general lower bound. It assumes no reduction in the cost of each B application; it removes only recorded correction and extraction components.",
        methods=scenarios)


def aggregate(manifest,rows):
    config=manifest["configuration"];grouped=defaultdict(list)
    for row in rows: grouped[(row["method"],row["seed"],row["repeat"])].append(row)
    sequences=[]
    for (method,seed,repeat),group in sorted(grouped.items()):
        value=dict(method=method,seed=seed,repeat=repeat,**totals(group))
        value["categories"]={kind:totals([r for r in group if category(r)==kind]) for kind in CATEGORIES}
        for values in value["categories"].values():
            values["solve_fraction"]=ratio(values["solve_seconds"],value["solve_seconds"])
            values["total_fraction"]=ratio(values["total_seconds"],value["total_seconds"])
        sequences.append(value)
    methods={}
    for setting in config["specs"]:
        name=setting["name"]
        ss=[s for s in sequences if s["method"]==name];rr=[r for r in rows if r["method"]==name]
        geometry={}
        for seed in config["seeds"]:
            local=[s for s in ss if s["seed"]==seed]
            geometry[str(seed)]={key:dist(s[key] for s in local) for key in QUANTITIES}
            geometry[str(seed)]["categories"]={kind:{key:dist(s["categories"][kind][key] for s in local)
                for key in [*QUANTITIES,"steps","solve_fraction","total_fraction"]} for kind in CATEGORIES}
        failures={key:sum(s[key] for s in ss) for key in COUNTS}
        methods[name]=dict(spec=setting,by_geometry=geometry,**failures,eligible=all(v==0 for v in failures.values()),
            **{key:dist(g[key]["median"] for g in geometry.values()) for key in QUANTITIES},
            categories={kind:{key:dist(g["categories"][kind][key]["median"] for g in geometry.values())
                for key in [*QUANTITIES,"steps","solve_fraction","total_fraction"]} for kind in CATEGORIES},
            final_relative_residual=dist(r["final_relative_residual"] for r in rr),
            pcg_relative_residual_before_recovery=dist(r["relative_residual"] for r in rr),
            energy_relative_error=dist(r["energy_relative_error"] for r in rr),
            applied_rank=dist(r["rank"] for r in rr),old_space_rank=dist(r["old_space_rank"] for r in rr),
            solver_operator_counts=dict(Counter(r["solver_operator"] for r in rr)),
            extraction_status_counts=dict(Counter(r["extraction"]["status"] for r in rr if r["extraction"])),
            basis_status_counts=dict(Counter(r["basis_stats"]["status"] for r in rr if r["basis_stats"])),
            projection_status_counts=dict(Counter(r["projection_info"]["status"] for r in rr if r["projection_info"])))
    pairs={}
    for name in METHODS:
        for baseline in BASELINES:
            if name!=baseline: pairs[f"{name}__over__{baseline}"]=compare(methods,name,baseline)
    for numerator,denominator in MECHANISM_PAIRS:
        pairs[f"{numerator}__over__{denominator}"]=compare(methods,numerator,denominator)
    valid=[b for b in BASELINES if methods[b]["eligible"]]
    fastest=min(valid,key=lambda b:methods[b]["total_seconds"]["median"]) if valid else None
    main={}
    for name in METHODS[2:]:
        points=[]
        for seed,value in methods[name]["by_geometry"].items():
            base=methods[fastest]["by_geometry"][seed]["total_seconds"]["median"] if fastest else None
            points.append(dict(seed=int(seed),denominator=fastest,numerator_seconds=value["total_seconds"]["median"],
                denominator_seconds=base,time_ratio=ratio(value["total_seconds"]["median"],base)))
        ratios=dist(p["time_ratio"] for p in points)
        main[name]=dict(denominator=fastest,by_geometry=points,time_ratio=ratios,
            geometries_faster=sum(p["time_ratio"] is not None and p["time_ratio"]<1 for p in points),
            geometries_at_least_20_percent_faster=sum(p["time_ratio"] is not None and p["time_ratio"]<=.8 for p in points),
            passes_median_20_percent_engineering_gate=bool(methods[name]["eligible"] and ratios["median"] is not None and ratios["median"]<=.8))
    return dict(n=config["n"],independent_geometries=len(config["seeds"]),repeats_per_geometry=config["repeats"],
        methods=methods,paired_comparisons=pairs,fastest_valid_baseline_by_overall_median=fastest,
        versus_fastest_valid_baseline=main,complete_sequences=sequences,
        fixed_trajectory_cost_scenario=fixed_trajectory_cost_scenario(sequences,methods,fastest))


def markdown(summary):
    accuracy=summary["online_accuracy"]
    lines=["# 第七阶段：冻结历史空间的持续平衡作用", "",
        f"生成时间：{summary['created_utc']}。状态：`{summary['status']}`。", "",
        "本阶段主假设在运行前固定为 `frozen_all_balanced4`：仅在首步采集基础 M 的轨迹并提取四个方向，之后所有步均使用当前 A 下的固定平衡接口。开发阶段只检查预定数值门槛，不按性能选择候选；确认始终保留同一套五种方法。没有训练或调用 AI。", "",
        "## 完整性、计时与聚合", "",
        f"已核验 {accuracy['solves']} 条在线记录。PCG 失败 {accuracy['pcg_failures']}、直接恢复 {accuracy['recoveries']}、最终失败 {accuracy['final_failures']}、整个缓存空间拒绝 {accuracy['cache_rejections']}；最终显式相对真残差最大值 {sci(accuracy['max_final_relative_residual'])}。",
        "每次完整 20 步先求和，再取同几何重复中位数，最后取几何中位数或逐几何配对比。范围为几何间最小值至最大值，不是置信区间。首步无历史空间，后续事件和非事件分别列示；不能挑选更快的某个步代替完整序列。确认仅有五个独立几何，256 网格复用其中前两个。",
        "组装、历史初值、AMG 刷新、当前方向处理、投影、PCG、真残差、恢复、缓存采集/复制、提取及维护均计入总时间。提取是维护的子项，不能重复相加。各组件分别取中位数，其和不必等于总时间中位数。直接参考解和误差核验在计时外，首步空间哈希在每条完整序列计时结束后计算，仅作来源审计，不参与在线决策。",
        "所有回收方案仅在首步使用基础 M 采集至多 24 列并提取秩4，此后不更新空间或采集新轨迹。后续事件中的自实现 PCG 接收实际使用的 B 或 M，采集容量为零，因此不会把 BA 缓存解释成 MA。每次求解内预条件器固定。",
        "`solver_operator` 审核仅核对正式记录与预定策略的一致性；实际算子对象的传递由源码审查和 100 次小网格 identity 检测求解验证。正式计时没有加入逐次 identity 检测，不能将记录审核当成逐次对象身份验证。",
        "首步空间哈希核对每条序列记录的一致性及同网格/几何/重复下三种回收方法的来源。哈希不同不自动意味着数学子空间不同，需要检查符号或基旋转等价；不据此换样本。相同初始空间也不代表后续求解历史和浮点轨迹逐位相同。", "",
        f"首步来源审计中，共 {summary['first_space_audit']['matched_groups']} 组同网格/几何/重复比较，字节哈希不一致 {summary['first_space_audit']['byte_mismatch_groups']} 组。所有哈希记录保留在 JSON。", ""]
    titles={"stage7_development":"开发数值门槛：128网格、3个几何、2重复",
            "stage7_confirm128":"冻结确认：128网格、5个几何、3重复",
            "stage7_confirm256":"冻结确认：256网格、同组前2个几何、3重复"}
    for key,stage in summary["stages"].items():
        lines += [f"## {titles[key]}", "", "| 方法 | 完整秒数中位数 [范围] | 迭代中位数 | 对SciPy配对时间比 | 对自实现配对时间比 | PCG失败/恢复/最终失败/缓存拒绝 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name,value in stage["methods"].items():
            timing=value["total_seconds"]
            ratios=[1. if name==b else stage["paired_comparisons"][f"{name}__over__{b}"]["total_seconds_ratio"]["median"] for b in BASELINES]
            counts="/".join(str(value[c]) for c in COUNTS)
            lines.append(f"| {name} | {f(timing['median'])} [{f(timing['min'])}, {f(timing['max'])}] | {f(value['iterations']['median'],1)} | {f(ratios[0])} | {f(ratios[1])} | {counts} |")
        lines += ["", "| 方法 | 首步总秒数 | 后续事件总秒数 | 非事件总秒数 | 首步迭代 | 后续事件迭代 | 非事件迭代 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name,value in stage["methods"].items():
            values=[value["categories"][group][field]["median"] for field in ["total_seconds","iterations"] for group in CATEGORIES]
            lines.append("| "+name+" | "+" | ".join(f(v,4 if i<3 else 1) for i,v in enumerate(values))+" |")
        lines += ["", "| 方法 | 方向/投影秒数 | PCG秒数 | 维护秒数 | 其中提取秒数 | PCG+维护秒数 |", "|---|---:|---:|---:|---:|---:|"]
        for name,value in stage["methods"].items():
            lines.append("| "+name+" | "+" | ".join(f(value[field]["median"],4) for field in ["correction_seconds","solve_seconds","maintenance_seconds","extraction_seconds","solve_plus_maintenance_seconds"])+" |")
        lines += ["", "同阶段机制配对；小于 1 表示分子更少。all_balanced/event_warm 只增加后续事件中的持续平衡作用；all_balanced/all_warm 比较所有后续步的持续作用；event_warm/all_warm 只增加非事件中的持续作用。", "",
                  "| 分子 / 分母 | 完整时间比 | 完整迭代比 | 后续事件时间比 | 后续事件迭代比 | 非事件时间比 | 非事件迭代比 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for numerator,denominator in [("baseline_event_custom","baseline_scipy"),*MECHANISM_PAIRS]:
            pair=stage["paired_comparisons"][f"{numerator}__over__{denominator}"]
            values=[pair["total_seconds_ratio"]["median"],pair["iterations_ratio"]["median"]]
            values += [pair["categories"][kind][field]["median"] for kind in ["event","non_event"] for field in ["total_seconds_ratio","iterations_ratio"]]
            lines.append(f"| {numerator} / {denominator} | "+" | ".join(f(v) for v in values)+" |")
        lines += ["", f"按本组全部几何的中位时间选出的最快有效基础方法为 `{stage['fastest_valid_baseline_by_overall_median']}`。以下固定使用这一种基础方法，再逐几何配对；不逐几何或逐次挑选最快基线。预定主候选始终是 `{HYPOTHESIS}`，不随结果更换。", "",
                  "| 回收方法 | 对固定最快基线配对比 | 更快几何数 | 至少节省20%几何数 | 配对中位20%工程线 |", "|---|---:|---:|---:|---|"]
        for name,value in stage["versus_fastest_valid_baseline"].items():
            lines.append(f"| {name} | {f(value['time_ratio']['median'])} | {value['geometries_faster']}/{stage['independent_geometries']} | {value['geometries_at_least_20_percent_faster']}/{stage['independent_geometries']} | {'通过' if value['passes_median_20_percent_engineering_gate'] else '未通过'} |")
        lines += [""]
    lines += ["## 全部确认几何配对", "", "| 网格/方法/几何 | 完整秒数 | SciPy秒数 | 自实现秒数 | 对本组固定最快基线比 |", "|---|---:|---:|---:|---:|"]
    for key in ["stage7_confirm128","stage7_confirm256"]:
        if key not in summary["stages"]: continue
        stage=summary["stages"][key]
        for name,comparison in stage["versus_fastest_valid_baseline"].items():
            for value in comparison["by_geometry"]:
                seed=str(value["seed"])
                basetimes=[stage["methods"][b]["by_geometry"][seed]["total_seconds"]["median"] for b in BASELINES]
                lines.append(f"| {stage['n']}/{name}/{seed} | {f(value['numerator_seconds'],4)} | {f(basetimes[0],4)} | {f(basetimes[1],4)} | {f(value['time_ratio'],4)} |")
    lines += ["", "## 事后固定实测轨迹的成本情景", "",
        "对主候选每次完整序列先计算 `total − correction − extraction`，再取同几何重复中位数并与基线逐几何配对。两基线都保留，固定最快基线仍按该网格整组实际完整时间选出。该情景不参与选法；JSON 另保留所有回收方法的逐重复、逐几何值。", "",
        "| 阶段/网格 | 对SciPy配对时间比 | 对自实现配对时间比 | 固定最快基线 | 对固定最快基线比 |",
        "|---|---:|---:|---|---:|"]
    for key,stage in summary["stages"].items():
        scenario=stage["fixed_trajectory_cost_scenario"]
        value=scenario["methods"][HYPOTHESIS]
        ratios=[value["versus_each_baseline"][b]["time_ratio"]["median"] for b in BASELINES]
        fastest=value["versus_fixed_fastest_valid_baseline"]
        label="开发" if key=="stage7_development" else "确认"
        lines.append(f"| {label}/{stage['n']} | {f(ratios[0],6)} | {f(ratios[1],6)} | {scenario['fastest_valid_baseline_by_overall_median']} | {f(fastest['time_ratio']['median'] if fastest else None,6)} |")
    lines += ["", "这只是固定实测轨迹、迭代数及其余成本的事后核算，不是实际可运行算法，也不是一般性能下界。它没有假设降低每次 B 应用的代价，仅移除记录中的方向处理/投影和提取组件；维护中的其他成本仍保留。在这些测量下，即便把这两个组件视为零，确认配对中位时间仍未达到节省 20% 的工程线；不能据此断言其他轨迹或其他实现永远达不到。"]
    if summary["status"]!="complete":
        lines += ["",f"数值门槛未满足，按协议停止：{summary.get('stop_reason','未进入确认')}。不对缺失确认结果作推断。"]
    lines += ["", "## 可支持的结论与边界", "",
        "完整时间、对应迭代和组件成本必须同时解释。持续平衡若只减少迭代而没有完整时间优势，也应保留；若出现完整收益，首先归于经典冻结空间与预条件接口，不能归为 AI 收益或创新。只使用本阶段同协议的真实 PDE 结果比较，不能将跨阶段时间差直接归因于接口变化。",
        "20% 仍是预定工程筛选线，不是统计显著性或论文录用标准。少量几何、单右端和单对比度不足以建立普遍结论；后续改变预算、实现或方程范围需重新冻结并分配新的确认几何。",
        "源代码、协议、分组、开发数值门槛与冻结文件均核对哈希；每条预定记录精确匹配，没有按性能删样本。逐重复成本、逐几何配对、数值失败、空间哈希和输入哈希见 [summary.json](summary.json)，设计见 [运行前协议](../../docs/stage7_protocol_zh.md)。", "",
        "![在线完整时间与迭代](figures/online_comparison.png)", "", "![首步事件非事件作用](figures/event_coverage.png)", "",
        "![完整成本组件](figures/component_costs.png)", ""]
    return "\n".join(lines)


def plots(summary,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    stages=summary["stages"]
    folder=out/"figures";folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        for extension in ["png","pdf"]: fig.savefig(folder/f"{name}.{extension}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(2,len(stages),figsize=(5.2*len(stages),8.7),squeeze=False,constrained_layout=True)
    for col,(name,stage) in enumerate(stages.items()):
        methods=stage["methods"];names=list(methods)
        for row,field,label in [(0,"total_seconds","Complete 20-step sequence (s)"),(1,"iterations","PCG iterations per sequence")]:
            ax=axes[row,col]
            values=np.array([methods[m][field]["median"] for m in names])
            low=np.array([methods[m][field]["min"] for m in names]);high=np.array([methods[m][field]["max"] for m in names])
            ax.bar(range(len(names)),values,color=["#9da4aa" if m in BASELINES else "#3b7f9c" if m==HYPOTHESIS else "#ce9a59" for m in names])
            ax.errorbar(range(len(names)),values,yerr=[values-low,high-values],fmt="none",capsize=3,color="#333333")
            ax.set_xticks(range(len(names)),names,rotation=45,ha="right",fontsize=7)
            ax.set_title(f"{'Development gate' if name.endswith('development') else 'Frozen confirmation'} {stage['n']}\n{stage['independent_geometries']} geometries; {stage['repeats_per_geometry']} repeats")
            ax.set_ylabel(label);ax.grid(axis="y",alpha=.15)
    fig.suptitle("Prespecified persistent-space hypothesis: full costs, geometry medians and ranges")
    save(fig,"online_comparison")
    fig,axes=plt.subplots(2,len(stages),figsize=(5.2*len(stages),8.7),squeeze=False,constrained_layout=True)
    for col,(name,stage) in enumerate(stages.items()):
        methods=stage["methods"];names=list(methods)
        for row,field,label in [(0,"total_seconds","Category time (s / complete sequence)"),(1,"iterations","Iterations by step category")]:
            ax=axes[row,col];bottom=np.zeros(len(names))
            for category_name,display,color in [("initial","Initial step (no history)","#9da4aa"),("event","Later events","#ce9a59"),("non_event","Non-events","#3b7f9c")]:
                values=np.array([methods[m]["categories"][category_name][field]["median"] for m in names])
                ax.bar(range(len(names)),values,bottom=bottom,label=display,color=color);bottom+=values
            ax.set_xticks(range(len(names)),names,rotation=45,ha="right",fontsize=7)
            ax.set_title(f"Grid {stage['n']}; {'development' if name.endswith('development') else 'confirmation'}")
            ax.set_ylabel(label)
            if col==len(stages)-1: ax.legend(fontsize=7,loc="upper left",bbox_to_anchor=(1.01,1))
    fig.suptitle("Initial step separated from later events; separately aggregated category medians need not sum to total median")
    save(fig,"event_coverage")
    fig,axes=plt.subplots(1,len(stages),figsize=(5.2*len(stages),5.3),squeeze=False,constrained_layout=True)
    labels=["Assembly","History guess","AMG setup","Directions / projection","PCG","Maintenance","Recovery","Other"]
    for col,(name,stage) in enumerate(stages.items()):
        ax=axes[0,col];methods=stage["methods"];names=list(methods);bottom=np.zeros(len(names))
        for field,label in zip([*COMPONENTS,"other_seconds"],labels):
            values=np.array([methods[m][field]["median"] for m in names])
            ax.bar(range(len(names)),values,bottom=bottom,label=label);bottom+=values
        ax.set_xticks(range(len(names)),names,rotation=45,ha="right",fontsize=7)
        ax.set_ylabel("Component median (s / complete sequence)")
        ax.set_title(f"Grid {stage['n']}; {'development' if name.endswith('development') else 'confirmation'}")
        if col==len(stages)-1: ax.legend(fontsize=7,loc="upper left",bbox_to_anchor=(1.01,1))
    fig.suptitle("All online components retained; first-step extraction is inside maintenance, never added twice")
    save(fig,"component_costs")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    root=args.root.resolve();out=args.output.resolve() if args.output else root/"results/stage7_report"
    splits=read(root/"data/stage7_splits.json")
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),schema_version=1,fixed_hypothesis=HYPOTHESIS,
        aggregation="sum all twenty steps; median repeats within geometry; then geometry medians/ranges and paired ratios; retain all failures",
        split_audit=check_splits(root,splits),stages={},audit={},reporter_sha256=sha(Path(__file__)))
    manifest,rows,audit=load_online(root,"stage7_development",splits)
    require(len(rows)==600,"Expected exactly 600 development rows")
    summary["stages"]["stage7_development"]=aggregate(manifest,rows)
    summary["audit"]["stage7_development"]=audit
    all_rows=list(rows)
    numeric_gate=all(m["eligible"] for m in summary["stages"]["stage7_development"]["methods"].values())
    summary["development_gate"]=dict(passed=numeric_gate,performance_selection=False,
        rule="All fixed five methods must be finally accurate with zero PCG failures and zero whole-space cache rejections; timing never changes the primary hypothesis or confirmation methods.")
    freeze_path=root/"results/stage7_development/freeze.json"
    if not numeric_gate:
        require(not freeze_path.exists(),"Freeze exists despite a failed numerical development gate")
        summary.update(status="development_numerical_gate_failed",stop_reason="预定五种方法未全部满足最终准确、零PCG失败及零整空间缓存拒绝")
    else:
        frozen=read(freeze_path)
        check_hashes(root,frozen,"freeze")
        require(frozen["fixed_hypothesis"]==HYPOTHESIS,"Prespecified primary hypothesis changed")
        require(frozen["confirmation_specs"]==expected_specs(),"Fixed confirmation methods changed")
        for key,name in [("development_measurements_sha256","measurements.json"),("development_manifest_sha256","manifest.json")]:
            require(frozen[key]==sha(freeze_path.parent/name),"Frozen development files changed")
        summary.update(frozen_design=frozen,freeze_sha256=sha(freeze_path))
        for n,expected_rows in [(128,1500),(256,600)]:
            name=f"stage7_confirm{n}"
            m,rr,aa=load_online(root,name,splits,sha(freeze_path))
            require(len(rr)==expected_rows,f"Expected {expected_rows} confirmation rows")
            summary["stages"][name]=aggregate(m,rr);summary["audit"][name]=aa;all_rows.extend(rr)
        complete=read(root/"results/stage7_complete.json")
        require(complete.get("complete") is True and complete.get("freeze_sha256")==sha(freeze_path),"Final completion marker differs")
        require(len(all_rows)==2700,"Expected exactly 2700 online rows")
        summary.update(status="complete",completion_sha256=sha(root/"results/stage7_complete.json"))
    extraction_rows=[r["extraction"] for r in all_rows if r["extraction"]]
    summary["online_accuracy"]=dict(solves=len(all_rows),pcg_failures=sum(not r["converged"] for r in all_rows),
        recoveries=sum(r["solver_fallback"] for r in all_rows),final_failures=sum(not r["final_converged"] for r in all_rows),
        cache_rejections=sum(r["cache_rejected"] for r in all_rows),max_final_relative_residual=max(r["final_relative_residual"] for r in all_rows),
        capture_rejected_vectors=sum(r["capture_stats"].get("rejected_vectors",0) for r in all_rows),
        capture_terminal_vectors_skipped=sum(r["capture_stats"].get("skipped_terminal_vectors",0) for r in all_rows),
        extraction_rejected_vectors=sum(r.get("rejected_vectors",0) for r in extraction_rows),
        extraction_dependent_vectors=sum(r.get("dependent_vectors",0) for r in extraction_rows),
        first_step_extraction_count=len(extraction_rows),later_step_extraction_count=sum(bool(r["extraction"]) and r["step"]>0 for r in all_rows),
        later_step_captured_vectors=sum(r["capture_stats"].get("captured_vectors",0) for r in all_rows if r["step"]>0))
    first_space=[value["first_space_audit"] for value in summary["audit"].values()]
    summary["first_space_audit"]=dict(matched_groups=sum(len(v["matched_seed_repeat_groups"]) for v in first_space),
        byte_mismatch_groups=sum(v["byte_mismatch_groups"] for v in first_space),
        all_three_methods_byte_identical=all(v["all_three_methods_byte_identical"] for v in first_space),
        byte_mismatch_is_not_automatic_subspace_failure=True,
        timing_scope="Computed after all sequence timing; source audit only, never an online input.")
    verification=root/"results/stage7_verification.json"
    if verification.exists():
        record=read(verification)
        if "new_test_sha256" in record:
            require(record["new_test_sha256"]==sha(root/"test_persistent_recycling.py"),"Verification test source changed")
        summary["prior_verification"]=dict(record=record,sha256=sha(verification))
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    (out/"REPORT_zh.md").write_text(markdown(summary))
    plots(summary,out)
    print(json.dumps(dict(output=str(out),status=summary["status"],fixed_hypothesis=HYPOTHESIS,online_rows=len(all_rows)),ensure_ascii=False))


if __name__=="__main__":main()
