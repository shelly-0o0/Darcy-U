#!/usr/bin/env python3
"""Audit causal event-space experiments without running a PDE solver.

Each observation is a complete twenty-step sequence. Aggregate repeats within
one geometry before aggregating or pairing geometries. Offline current-system
probes remain separate from causal online timings and conclusions.
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
RTOL, STEPS = 1e-8, 20
SOURCES = ["fields.py", "numerics.py", "sequence_baselines.py", "balanced_augmentation.py",
           "cached_pcg.py", "event_space_update.py", "event_recycling_experiment.py"]
BASELINES = ["baseline_scipy", "baseline_event_custom"]
SCREEN = [*BASELINES, "lagged_balanced4", "event_reuse_refresh4", "event_update_refresh4",
          "event_reuse_frozen4", "event_update_transport4"]
NEW_METHODS = SCREEN[3:]
COMPONENTS = ["assembly_seconds", "guess_seconds", "setup_seconds", "update_seconds",
              "correction_seconds", "solve_seconds", "maintenance_seconds", "recovery_seconds"]
QUANTITIES = ["total_seconds", *COMPONENTS, "other_seconds", "extraction_seconds",
              "solve_plus_maintenance_seconds", "iterations"]
CATEGORIES = ["initial", "event", "non_event"]
COUNTS = ["pcg_failures", "recoveries", "final_failures", "cache_rejections", "update_rejections"]
PROBE_DIRECTIONS = ["previous_event", "updated_previous", "initial_frozen", "current_retrospective"]
EVENT_STEPS = [3, 6, 9, 10, 12, 15, 18]
MECHANISM_PAIRS = [("event_reuse_refresh4", "lagged_balanced4"),
                   ("event_update_refresh4", "event_reuse_refresh4"),
                   ("event_reuse_frozen4", "event_reuse_refresh4"),
                   ("event_update_transport4", "event_update_refresh4"),
                   ("event_update_transport4", "event_reuse_frozen4")]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dist(values):
    values = list(values)
    valid = [float(v) for v in values if finite(v)]
    return dict(count=len(values), finite_count=len(valid), median=median(valid) if valid else None,
                min=min(valid) if valid else None, max=max(valid) if valid else None)


def ratio(a, b):
    return a / b if finite(a) and finite(b) and b > 0 else None


def f(value, digits=3):
    return "不适用" if value is None else f"{value:.{digits}f}"


def sci(value):
    return "不适用" if value is None else f"{value:.3e}"


def check_problem(manifest, where):
    for key, expected in [("steps", STEPS), ("rtol", RTOL), ("contrast", 1e4),
                          ("maxiter", 500), ("blas_threads", 1), ("device", "cpu"), ("rank", 4)]:
        require(manifest.get(key) == expected, f"Unexpected {key}: {where}")


def check_hashes(root, record, where):
    require(set(record["source_sha256"]) == set(SOURCES), f"Source inventory differs: {where}")
    result = {"source_sha256": {name: sha(root/name) == digest for name,digest in record["source_sha256"].items()}}
    require(all(result["source_sha256"].values()), f"Frozen source differs: {where}")
    for key,name in [("protocol_sha256","docs/stage6_protocol_zh.md"),("splits_sha256","data/stage6_splits.json")]:
        result[key] = sha(root/name) == record[key]
        require(result[key],f"Frozen {key} differs: {where}")
    return result


def check_splits(root, splits):
    groups = splits["groups"]
    require(set(groups) == {"development","confirmation"}, "Unexpected split groups")
    require(len(groups["development"]) == 3 and len(groups["confirmation"]) == 5, "Unexpected split sizes")
    seeds=groups["development"]+groups["confirmation"]
    require(len(set(seeds))==8,"Repeated stage-6 seed")
    hashes=[splits["geometry_sha256"][str(seed)] for seed in seeds]
    require(len(set(hashes))==8,"Repeated stage-6 geometry hash")
    require(not set(seeds).intersection(splits["excluded_seeds"]),"Excluded seed reused")
    require({75001,75002}.issubset(splits["excluded_seeds"]),"Smoke geometry exclusions incomplete")
    for n in [4,5]:
        old=read(root/f"data/stage{n}_splits.json")
        require(set(old["excluded_seeds"]+sum(old["groups"].values(),[])).issubset(splits["excluded_seeds"]), f"Stage {n} exclusion incomplete")
        require(not set(hashes).intersection(old.get("geometry_sha256",{}).values()),f"Saved geometry overlaps stage {n}")
    return dict(development_geometries=3, confirmation_geometries=5, confirmation_256_reuses_first_two=True,
        prior_split_sha256={f"data/stage{n}_splits.json":sha(root/f"data/stage{n}_splits.json") for n in [4,5]},
        note="Checks saved split metadata and available prior hashes; no coefficient fields are regenerated.")


def load_online(root,name,splits,frozen=None):
    folder=root/"results"/name
    manifest,rows,complete,refs=[read(folder/p) for p in ["manifest.json","measurements.json","complete.json","references.json"]]
    is_screen,n=name=="stage6_screen",256 if name=="stage6_confirm256" else 128
    seeds=splits["groups"]["development" if is_screen else "confirmation"]
    if n==256:
        seeds=seeds[:2]
    repeats=2 if is_screen else 3
    config=manifest["configuration"]
    names=[s["name"] for s in config["specs"]]
    require(config["n"]==n and config["seeds"]==seeds and config["repeats"]==repeats,f"Schedule differs: {name}")
    require(len(names)==len(set(names)),f"Duplicate methods: {name}")
    if is_screen:
        require(names==SCREEN,"All seven prespecified screen methods must remain")
        expected_specs=[dict(name=name,policy=policy,event_custom=name!="baseline_scipy") for name,policy in zip(SCREEN,
                        ["none","none","lagged","reuse_refresh","update_refresh","reuse_frozen","update_transport"])]
        require(config["specs"]==expected_specs,"Screen settings differ from protocol")
    else:
        require(frozen is not None and config["specs"]==frozen["confirmation_specs"],"Confirmation methods differ from freeze")
        require(config["selection_sha256"]==sha(root/"results/stage6_screen/frozen_selection.json"),"Confirmation selection hash differs")
    expected={(n,method,seed,repeat,step) for method in names for seed in seeds for repeat in range(repeats) for step in range(STEPS)}
    keys=[(r["n"],r["method"],r["seed"],r["repeat"],r["step"]) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)==expected,f"Missing, duplicate or unexpected row: {name}")
    require(complete.get("complete") is True and complete.get("rows")==len(rows),f"Completion marker differs: {name}")
    check_problem(manifest,name)
    hashes=check_hashes(root,manifest,name)
    settings={s["name"]:s for s in config["specs"]}
    for row in rows:
        for key in ["total_seconds",*COMPONENTS,"extraction_seconds","relative_residual","final_relative_residual",
                    "energy_relative_error","base_relative_residual","corrected_relative_residual","orthogonality_error"]:
            require(finite(row[key]) and row[key]>=0,f"Invalid {key}: {name}")
        require(row["initial_step"]==(row["step"]==0),"Initial step flag inconsistent")
        require(not row["initial_step"] or row["refresh_event"],"Initial step must be a refresh event")
        require(row["refresh_event"]==(row["step"]==0 or row["step"] in EVENT_STEPS),"Event schedule differs from protocol")
        require(row["final_converged"]==(row["final_relative_residual"]<=RTOL),"Final convergence flag inconsistent")
        require(row["converged"]==(row["relative_residual"]<=RTOL),"PCG convergence flag inconsistent")
        require(row["solver_fallback"]==(not row["converged"]),"Recovery flag inconsistent")
        require(isinstance(row["iterations"],int) and 0<=row["iterations"]<=500,"Iteration budget exceeded")
        require(isinstance(row["restarts"],int) and 0<=row["restarts"]<=3,"Refresh budget exceeded")
        require(sum(row[c] for c in COMPONENTS)<=row["total_seconds"]+1e-8,"Component costs exceed total")
        require(row["extraction_seconds"]<=row["maintenance_seconds"]+1e-8,"Extraction is not inside maintenance")
        require(0<=row["rank"]<=4 and 0<=row["old_space_rank"]<=4 and 0<=row["stored_space_rank"]<=4,"Space rank exceeds budget")
        policy=settings[row["method"]]["policy"]
        if policy=="none":
            require(row["rank"]==row["old_space_rank"]==row["stored_space_rank"]==0,"Baseline unexpectedly uses a space")
        if policy=="lagged" and row["refresh_event"]:
            require(row["rank"]==0,"Lagged method uses old space on event")
        if row["initial_step"]:
            require(row["old_space_rank"]==row["rank"]==0,"First step uses a past space")
        cap=row["capture_stats"]
        require(bool(cap)==bool(row["refresh_event"] and settings[row["method"]]["event_custom"]),"Unexpected/missing custom solver metadata")
        expected_extraction=bool(policy!="none" and row["refresh_event"] and (row["initial_step"] or policy in ("lagged","reuse_refresh","update_refresh")))
        require(row["extraction_requested"]==expected_extraction,"Extraction request differs from policy")
        if cap:
            require(0<=cap["captured_vectors"]<=24,"Capture exceeds 24 columns")
            require(cap["max_vectors"]==(24 if expected_extraction else 0),"Capture budget differs from policy")
            if not expected_extraction:
                require(cap["captured_vectors"]==0,"Unrequested trajectory columns collected")
            require(cap["extra_a_applications_for_cache"]==cap["extra_m_applications_for_cache"]==0,"Cache added operator actions")
            require(len(cap["segment_ids"])==cap["captured_vectors"],"Incomplete capture segment metadata")
            require(all(0<=s<=row["restarts"] for s in cap["segment_ids"]),"Invalid capture segment")
        update=row["update_stats"]
        expected_update=bool(row["refresh_event"] and not row["initial_step"] and policy in ("update_refresh","update_transport") and row["old_space_rank"])
        require(bool(update)==expected_update,"Update request differs from policy")
        if update:
            require(row["refresh_event"] and not row["initial_step"],"Update outside a later event")
            require(all(isinstance(update[key],int) and update[key]>=0 for key in ["a_applications","m_applications","max_a_applications","max_m_applications"]),"Invalid update action counts")
            require(update["a_applications"]<=update["max_a_applications"]<=8,"Update A action budget exceeded")
            require(update["m_applications"]<=update["max_m_applications"]<=8,"Update M action budget exceeded")
            require(row["update_rejected"]==(update["status"] not in ("accepted","empty_basis")),"Update rejection flag inconsistent")
        else:
            require(row["update_seconds"]==0 and not row["update_rejected"],"Unrequested update cost/rejection")
        extraction=row["extraction"]
        require(bool(extraction)==expected_extraction,"Extraction record differs from policy")
        require(row["cache_rejected"]==bool(extraction and extraction["status"] not in ("accepted","empty_basis")),"Space rejection flag inconsistent")
        if extraction:
            require(extraction["a_applications"]==extraction["m_applications"]==0,"Cached extraction adds operators")
    refs_by_seed={r["seed"]:r for r in refs}
    require(len(refs)==len(seeds) and set(refs_by_seed)==set(seeds),"Reference inventory differs")
    for ref in refs:
        require(len(ref["relative_residuals"])==STEPS and finite(ref["seconds"]) and ref["seconds"]>=0,"Invalid reference provenance")
        require(all(finite(v) and v>=0 for v in ref["relative_residuals"]),"Invalid reference residual")
    # Event classification is an input-only schedule and must agree across methods/repeats.
    labels=defaultdict(set)
    sequences=defaultdict(list)
    for row in rows:
        labels[(row["seed"],row["step"])].add((row["initial_step"],row["refresh_event"],row["event"]))
        sequences[(row["method"],row["seed"],row["repeat"])].append(row)
    require(all(len(values)==1 for values in labels.values()),"Event schedule differs across methods/repeats")
    for (method,_,_),sequence in sequences.items():
        previous_rank=0
        policy=settings[method]["policy"]
        for row in sorted(sequence,key=lambda r:r["step"]):
            require(row["old_space_rank"]==previous_rank,"Stored-space history is discontinuous")
            next_rank=previous_rank
            if policy=="update_transport" and row["update_stats"].get("status")=="accepted":
                next_rank=row["update_stats"]["rank"]
            if row["extraction"]:
                next_rank=row["extraction"]["rank"]
            require(row["stored_space_rank"]==next_rank,"Stored space violates refresh/transport policy")
            previous_rank=next_rank
    return manifest,rows,dict(expected_rows=len(expected),actual_rows=len(rows),duplicate_rows=0,
        frozen_hash_checks=hashes,input_sha256={p.name:sha(p) for p in sorted(folder.glob("*.json"))},
        offline_reference_seconds=sum(r["seconds"] for r in refs),
        reference_relative_residual=dist(v for r in refs for v in r["relative_residuals"]),
        all_reference_accurate=all(v<=RTOL for r in refs for v in r["relative_residuals"]))


def check_solve(solve, where):
    require(finite(solve["relative_residual"]) and solve["relative_residual"]>=0,f"Invalid true residual: {where}")
    require(solve["converged"]==(solve["relative_residual"]<=RTOL),f"Convergence flag differs: {where}")
    require(isinstance(solve["iterations"],int) and 0<=solve["iterations"]<=500,f"Iteration budget differs: {where}")
    require(isinstance(solve["restarts"],int) and 0<=solve["restarts"]<=3,f"Restart budget differs: {where}")
    require(finite(solve["seconds"]) and solve["seconds"]>=0,f"Invalid probe time: {where}")


def load_probe(root,splits):
    folder=root/"results/stage6_probe"
    manifest,rows,complete=[read(folder/p) for p in ["manifest.json","measurements.json","complete.json"]]
    check_problem(manifest,"probe")
    hashes=check_hashes(root,manifest,"probe")
    config=manifest["configuration"]
    require(config["n"]==128 and config["seeds"]==splits["groups"]["development"],"Probe cohort differs")
    require(config["directions"]==PROBE_DIRECTIONS and config["interfaces"]==["warm","balanced"],"Probe variants differ")
    expected={(seed,step) for seed in config["seeds"] for step in EVENT_STEPS}
    keys=[(r["seed"],r["step"]) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)==expected,"Missing, duplicate or unexpected probe systems")
    require(complete.get("complete") is True and complete.get("cases")==21,"Probe completion marker differs")
    for row in rows:
        prior=[0,*EVENT_STEPS][EVENT_STEPS.index(row["step"])]
        require(row["previous_source_step"]==prior and row["initial_source_step"]==0,"Probe direction causality metadata differs")
        require(set(row["directions"])==set(PROBE_DIRECTIONS),"Probe direction inventory differs")
        for key in ["reference_relative_residual","base_error_energy","current_slow_error_energy"]:
            require(finite(row[key]) and row[key]>=0,f"Invalid probe {key}")
        check_solve(row["baseline"],"probe baseline")
        for name,value in row["directions"].items():
            require(0<=value["rank"]<=4,"Probe rank exceeds budget")
            for key in ["setup_seconds","error_energy_ratio","slow_error_energy_ratio","reference_space_overlap"]:
                require(finite(value[key]) and value[key]>=0,f"Invalid probe {name}/{key}")
            require(value["reference_space_overlap"]<=1+1e-6,"Probe space overlap exceeds normalized bound")
            require(set(value["solves"])=={"warm","balanced"},"Probe interfaces missing")
            for interface,solved in value["solves"].items():
                check_solve(solved,f"probe {name}/{interface}")
            if name=="updated_previous":
                info=value["setup_info"]
                require(info["a_applications"]<=info["max_a_applications"]<=8 and info["m_applications"]<=info["max_m_applications"]<=8,
                        "Probe update budget exceeded")
    direction_summaries={}
    quantities=["error_energy_ratio","slow_error_energy_ratio","reference_space_overlap","setup_seconds","rank"]
    for name in PROBE_DIRECTIONS:
        cases=[]
        for row in rows:
            value=row["directions"][name]
            case=dict(seed=row["seed"],step=row["step"],**{key:value[key] for key in quantities},solves={})
            for interface,solved in value["solves"].items():
                case["solves"][interface]=dict(**solved,iterations_over_baseline=ratio(solved["iterations"],row["baseline"]["iterations"]),
                    iteration_difference=solved["iterations"]-row["baseline"]["iterations"])
            case["balanced_over_warm_iterations"]=ratio(value["solves"]["balanced"]["iterations"],value["solves"]["warm"]["iterations"])
            case["balanced_minus_warm_iterations"]=value["solves"]["balanced"]["iterations"]-value["solves"]["warm"]["iterations"]
            cases.append(case)
        matched_fields=["balanced_over_warm_iterations","balanced_minus_warm_iterations"]
        def matched_counts(values):
            comparable=[c for c in values if c["solves"]["warm"]["converged"] and c["solves"]["balanced"]["converged"]]
            return dict(events=len(values),both_converged=len(comparable),failed_pairs=len(values)-len(comparable),
                balanced_fewer_iterations=sum(c["balanced_minus_warm_iterations"]<0 for c in comparable),
                equal_iterations=sum(c["balanced_minus_warm_iterations"]==0 for c in comparable),
                balanced_more_iterations=sum(c["balanced_minus_warm_iterations"]>0 for c in comparable))
        geometry={}
        for seed in config["seeds"]:
            local=[c for c in cases if c["seed"]==seed]
            geometry[str(seed)]={key:dist(c[key] for c in local) for key in [*quantities,*matched_fields]}
            geometry[str(seed)]["matched_interface_counts"]=matched_counts(local)
            geometry[str(seed)]["solves"]={interface:{key:dist(c["solves"][interface][key] for c in local)
                for key in ["iterations","iterations_over_baseline","iteration_difference","relative_residual"]} for interface in ["warm","balanced"]}
        direction_summaries[name]=dict(cases=cases,by_geometry=geometry,
            geometric_medians={key:dist(g[key]["median"] for g in geometry.values()) for key in [*quantities,*matched_fields]},
            matched_interface_counts=dict(**matched_counts(cases),geometries_with_fewer_median_iterations=sum(
                g["matched_interface_counts"]["failed_pairs"]==0 and g["balanced_minus_warm_iterations"]["median"]<0 for g in geometry.values())),
            solves={interface:dict(
                **{key:dist(g["solves"][interface][key]["median"] for g in geometry.values()) for key in ["iterations","iterations_over_baseline","iteration_difference","relative_residual"]},
                failures=sum(not c["solves"][interface]["converged"] for c in cases),
                max_relative_residual=max(c["solves"][interface]["relative_residual"] for c in cases)) for interface in ["warm","balanced"]},
            setup_status_counts=dict(Counter(r["directions"][name]["setup_info"]["status"] for r in rows)))
    update_pairs=[]
    for row in rows:
        updated,previous=row["directions"]["updated_previous"],row["directions"]["previous_event"]
        item=dict(seed=row["seed"],step=row["step"])
        for key in ["error_energy_ratio","slow_error_energy_ratio","reference_space_overlap"]:
            item[key+"_difference"]=updated[key]-previous[key]
        for interface in ["warm","balanced"]:
            item[interface+"_iteration_ratio"]=ratio(updated["solves"][interface]["iterations"],previous["solves"][interface]["iterations"])
            item[interface+"_iteration_difference"]=updated["solves"][interface]["iterations"]-previous["solves"][interface]["iterations"]
        update_pairs.append(item)
    pair_keys=[key for key in update_pairs[0] if key not in ("seed","step")]
    paired_geo={str(seed):{key:dist(p[key] for p in update_pairs if p["seed"]==seed) for key in pair_keys} for seed in config["seeds"]}
    return dict(cases=21,independent_geometries=3,retrospective_reference_only=True,raw_cases=rows,directions=direction_summaries,
        timing_scope="setup_seconds is current-space processing only; it excludes historical collection, current-retrospective extraction, projection and preconditioner construction. It is not a complete comparable online cost.",
        update_versus_reuse=dict(cases=update_pairs,by_geometry=paired_geo,
            geometric_medians={key:dist(g[key]["median"] for g in paired_geo.values()) for key in pair_keys}),
        baseline_failures=sum(not r["baseline"]["converged"] for r in rows),
        probe_solve_failures=sum(not v["converged"] for r in rows for d in r["directions"].values() for v in d["solves"].values()),
        reference_relative_residual=dist(r["reference_relative_residual"] for r in rows),
        update_a_applications=dist(r["directions"]["updated_previous"]["setup_info"]["a_applications"] for r in rows),
        update_m_applications=dist(r["directions"]["updated_previous"]["setup_info"]["m_applications"] for r in rows),
        update_max_relative_ritz_increase=dist(r["directions"]["updated_previous"]["setup_info"]["max_relative_ritz_increase"] for r in rows),
        input_sha256={str(p.relative_to(root)):sha(p) for p in sorted(folder.glob("*.json"))},frozen_hash_checks=hashes)


def category(row):
    return "initial" if row["initial_step"] else "event" if row["refresh_event"] else "non_event"


def sequence_values(rows):
    result = {key: sum(r[key] for r in rows) for key in ["total_seconds", *COMPONENTS, "extraction_seconds", "iterations"]}
    result["other_seconds"] = result["total_seconds"] - sum(result[k] for k in COMPONENTS)
    result["solve_plus_maintenance_seconds"] = result["solve_seconds"] + result["maintenance_seconds"]
    result.update(steps=len(rows), pcg_failures=sum(not r["converged"] for r in rows),
        recoveries=sum(r["solver_fallback"] for r in rows), final_failures=sum(not r["final_converged"] for r in rows),
        cache_rejections=sum(r["cache_rejected"] for r in rows),update_rejections=sum(r["update_rejected"] for r in rows))
    return result


def paired(methods, numerator, denominator):
    pairs = []
    for seed, va in methods[numerator]["by_geometry"].items():
        vb = methods[denominator]["by_geometry"][seed]
        pair = dict(seed=int(seed), categories={})
        for key in QUANTITIES:
            a, b = va[key]["median"], vb[key]["median"]
            pair[key + "_ratio"] = ratio(a, b)
            pair[key + "_difference"] = a - b if a is not None and b is not None else None
        for group in CATEGORIES:
            left, right = va["categories"][group], vb["categories"][group]
            pair["categories"][group] = {}
            for key in QUANTITIES:
                a, b = left[key]["median"], right[key]["median"]
                pair["categories"][group][key + "_ratio"] = ratio(a, b)
                pair["categories"][group][key + "_difference"] = a - b if a is not None and b is not None else None
        pairs.append(pair)
    keys = [key for key in pairs[0] if key not in ("seed", "categories")]
    return dict(numerator=numerator, denominator=denominator, by_geometry=pairs,
        **{key: dist(p[key] for p in pairs) for key in keys},
        categories={group: {key: dist(p["categories"][group][key] for p in pairs) for key in keys} for group in CATEGORIES})


def aggregate(manifest, rows):
    config = manifest["configuration"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["seed"], row["repeat"])].append(row)
    sequences = []
    for (method, seed, repeat), values in sorted(groups.items()):
        item = dict(method=method, seed=seed, repeat=repeat, **sequence_values(values))
        item["categories"] = {group: sequence_values([r for r in values if category(r) == group]) for group in CATEGORIES}
        for group, totals in item["categories"].items():
            totals["solve_fraction"] = ratio(totals["solve_seconds"], item["solve_seconds"])
            totals["total_fraction"] = ratio(totals["total_seconds"], item["total_seconds"])
        sequences.append(item)
    methods = {}
    for setting in config["specs"]:
        name = setting["name"]
        ss = [s for s in sequences if s["method"] == name]
        rr = [r for r in rows if r["method"] == name]
        geometry = {}
        for seed in config["seeds"]:
            local = [s for s in ss if s["seed"] == seed]
            geometry[str(seed)] = {key: dist(s[key] for s in local) for key in QUANTITIES}
            geometry[str(seed)]["categories"] = {group: {
                key: dist(s["categories"][group][key] for s in local)
                for key in [*QUANTITIES, "steps", "solve_fraction", "total_fraction"]} for group in CATEGORIES}
        failures = {key: sum(s[key] for s in ss) for key in COUNTS}
        methods[name] = dict(spec=setting, by_geometry=geometry,
            **{key: dist(g[key]["median"] for g in geometry.values()) for key in QUANTITIES},
            categories={group: {key: dist(g["categories"][group][key]["median"] for g in geometry.values())
                               for key in [*QUANTITIES, "steps", "solve_fraction", "total_fraction"]} for group in CATEGORIES},
            **failures, eligible=all(v == 0 for v in failures.values()),
            final_relative_residual=dist(r["final_relative_residual"] for r in rr),
            pcg_relative_residual_before_recovery=dist(r["relative_residual"] for r in rr),
            energy_relative_error=dist(r["energy_relative_error"] for r in rr),
            old_space_rank=dist(r["old_space_rank"] for r in rr), applied_rank=dist(r["rank"] for r in rr),
            extraction_status_counts=dict(Counter(r["extraction"]["status"] for r in rr if r["extraction"])),
            update_status_counts=dict(Counter(r["update_stats"]["status"] for r in rr if r["update_stats"])),
            projection_status_counts=dict(Counter(r["projection_info"]["status"] for r in rr if r["projection_info"])),
            basis_status_counts=dict(Counter(r["basis_stats"]["status"] for r in rr if r["basis_stats"])),
            update_budget_valid=all(r["update_stats"].get("a_applications",0)<=8 and r["update_stats"].get("m_applications",0)<=8 for r in rr),
            update_a_applications=dist(r["update_stats"]["a_applications"] for r in rr if r["update_stats"]),
            update_m_applications=dist(r["update_stats"]["m_applications"] for r in rr if r["update_stats"]),
            capture_rejected_vectors=sum(r["capture_stats"].get("rejected_vectors", 0) for r in rr),
            capture_terminal_vectors_skipped=sum(r["capture_stats"].get("skipped_terminal_vectors", 0) for r in rr),
            extraction_rejected_vectors=sum(r["extraction"].get("rejected_vectors", 0) for r in rr),
            extraction_dependent_vectors=sum(r["extraction"].get("dependent_vectors", 0) for r in rr),
            update_basis_rejected_vectors=sum(r["update_stats"].get("basis",{}).get("rejected_vectors",0) for r in rr),
            update_extraction_rejected_vectors=sum(r["update_stats"].get("extraction",{}).get("rejected_vectors",0) for r in rr),
            update_extraction_dependent_vectors=sum(r["update_stats"].get("extraction",{}).get("dependent_vectors",0) for r in rr))
    comparisons = {}
    for name in methods:
        for baseline in [*BASELINES, "lagged_balanced4"]:
            if name != baseline:
                comparisons[f"{name}__over__{baseline}"] = paired(methods, name, baseline)
    for numerator, denominator in MECHANISM_PAIRS:
        if numerator in methods and denominator in methods:
            comparisons[f"{numerator}__over__{denominator}"] = paired(methods, numerator, denominator)
    valid = [name for name in BASELINES if methods[name]["eligible"]]
    fastest = min(valid, key=lambda n: methods[n]["total_seconds"]["median"]) if valid else None
    main = {}
    for name, value in methods.items():
        if name in BASELINES:
            continue
        by_geometry = []
        for seed, result in value["by_geometry"].items():
            denominator = methods[fastest]["by_geometry"][seed]["total_seconds"]["median"] if fastest else None
            by_geometry.append(dict(seed=int(seed), denominator=fastest, numerator_seconds=result["total_seconds"]["median"],
                denominator_seconds=denominator, time_ratio=ratio(result["total_seconds"]["median"], denominator)))
        ratios = dist(g["time_ratio"] for g in by_geometry)
        main[name] = dict(denominator=fastest, by_geometry=by_geometry, time_ratio=ratios,
            geometries_faster=sum(g["time_ratio"] is not None and g["time_ratio"] < 1 for g in by_geometry),
            geometries_at_least_20_percent_faster=sum(g["time_ratio"] is not None and g["time_ratio"] <= .8 for g in by_geometry),
            passes_median_20_percent_engineering_gate=bool(value["eligible"] and ratios["median"] is not None and ratios["median"] <= .8))
    return dict(n=config["n"], independent_geometries=len(config["seeds"]), repeats_per_geometry=config["repeats"],
        methods=methods, paired_comparisons=comparisons, fastest_valid_baseline_by_overall_median=fastest,
        versus_fastest_valid_baseline=main, complete_sequences=sequences)


def online_markdown(summary):
    lines = ["## 在线统计口径", "",
        "每条完整序列先合计 20 步，再按同几何重复取中位数，最后汇总独立几何及逐几何配对比。几何范围不是置信区间；重复次数和第二种网格不增加独立几何数。所有失败、恢复和方向拒绝保留。",
        "将首步（无历史方向）、后续事件步和非事件步分别统计；后续事件部分不包含首步。每次求解内预条件器固定。离线直接解及当前系统事后轨迹只用于诊断，不能作为在线输入或加速证据。",
        "总计时包含组装、历史初值、AMG、更新、投影、求解、缓存采集/复制、真残差、恢复和维护。update_seconds 是独立组件；extraction_seconds 是维护的子项，不能重复相加。组件分别取中位数，其和不必等于总时间中位数。", ""]
    acc = summary.get("online_accuracy")
    if acc:
        lines += [f"在线记录共 {acc['solves']} 条；PCG 失败 {acc['pcg_failures']}，直接恢复 {acc['recoveries']}，最终失败 {acc['final_failures']}，整个缓存空间拒绝 {acc['cache_rejections']}，整个更新空间拒绝 {acc['update_rejections']}。最终显式相对真残差最大值 {sci(acc['max_final_relative_residual'])}。",
            f"更新预算核验：单次空间更新的 A/M 向量作用最多为 {acc['max_update_a_applications']}/{acc['max_update_m_applications']}，预定上限均为 8。该预算不包括历史初值、投影、残差检查及 PCG 本身的作用，全部真实成本仍计入总时间。", ""]
    titles = {"stage6_screen": "开发筛选：128网格、3个几何、2次重复", "stage6_confirm128": "冻结确认：128网格、5个几何、3次重复",
              "stage6_confirm256": "冻结确认：256网格、同一确认组前2个几何、3次重复"}
    for key, stage in summary["stages"].items():
        lines += [f"## {titles[key]}", "", "| 方法 | 完整序列秒数中位数 [范围] | 迭代中位数 | 对SciPy配对时间比 | 对自实现配对时间比 | PCG失败/恢复/最终失败/缓存拒绝/更新拒绝 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, value in stage["methods"].items():
            timing = value["total_seconds"]
            ratios = [1. if name == b else stage["paired_comparisons"][f"{name}__over__{b}"]["total_seconds_ratio"]["median"] for b in BASELINES]
            counts = "/".join(str(value[c]) for c in COUNTS)
            lines.append(f"| {name} | {f(timing['median'])} [{f(timing['min'])}, {f(timing['max'])}] | {f(value['iterations']['median'],1)} | {f(ratios[0])} | {f(ratios[1])} | {counts} |")
        lines += ["", "| 方法 | 首步总秒数 | 后续事件总秒数 | 非事件总秒数 | 首步迭代 | 后续事件迭代 | 非事件迭代 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, value in stage["methods"].items():
            values = [value["categories"][c][field]["median"] for field in ["total_seconds", "iterations"] for c in CATEGORIES]
            lines.append("| " + name + " | " + " | ".join(f(v,4 if i < 3 else 1) for i,v in enumerate(values)) + " |")
        lines += ["", "每条完整序列的成本几何中位数：", "", "| 方法 | 更新秒数 | 投影/方向秒数 | PCG秒数 | 维护秒数 | 其中提取秒数 | PCG+维护秒数 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, value in stage["methods"].items():
            fields = ["update_seconds", "correction_seconds", "solve_seconds", "maintenance_seconds", "extraction_seconds", "solve_plus_maintenance_seconds"]
            lines.append("| " + name + " | " + " | ".join(f(value[field]["median"],4) for field in fields) + " |")
        lines += ["", "同一阶段内的机制对照；配对比小于 1 表示分子较小。不同在线方法沿自己的数值历史推进，不能称后续方向或轨迹逐位相同。", "",
                  "| 分子 / 分母 | 完整时间比 | 完整迭代比 | 后续事件时间比 | 后续事件迭代比 | 非事件时间比 |", "|---|---:|---:|---:|---:|---:|"]
        comparisons = [("baseline_event_custom", "baseline_scipy"), *MECHANISM_PAIRS]
        for numerator, denominator in comparisons:
            pair = stage["paired_comparisons"].get(f"{numerator}__over__{denominator}")
            if pair is None:
                continue
            vals = [pair["total_seconds_ratio"]["median"], pair["iterations_ratio"]["median"],
                    pair["categories"]["event"]["total_seconds_ratio"]["median"], pair["categories"]["event"]["iterations_ratio"]["median"],
                    pair["categories"]["non_event"]["total_seconds_ratio"]["median"]]
            lines.append(f"| {numerator} / {denominator} | " + " | ".join(f(v) for v in vals) + " |")
        lines += ["", f"按这一网格整组几何中位时间选择的最快有效基础方法为 `{stage['fastest_valid_baseline_by_overall_median']}`。以下固定使用这一种基础方法，再逐几何配对，不逐几何挑选最快方法。", "",
                  "| 方法 | 对固定最快基线配对时间比 | 更快几何数 | 节省至少20%几何数 | 配对中位20%工程线 |", "|---|---:|---:|---:|---|"]
        for name, value in stage["versus_fastest_valid_baseline"].items():
            lines.append(f"| {name} | {f(value['time_ratio']['median'])} | {value['geometries_faster']}/{stage['independent_geometries']} | {value['geometries_at_least_20_percent_faster']}/{stage['independent_geometries']} | {'通过' if value['passes_median_20_percent_engineering_gate'] else '未通过'} |")
        lines += [""]
    if "frozen_selection" in summary:
        selection = summary["frozen_selection"]
        lines += ["## 冻结选择及全部确认配对", "", f"开发集选中 `{selection['candidate']['name']}`；只在预定新候选中按开发全成本排序，并不表示它已经赢过基础方法。确认仍固定保留两基线、lagged、reuse_refresh、update_refresh；如选中的新方法已在其中则去重。没有根据确认结果重选。", "",
                  "| 网格/方法/几何 | 完整秒数 | SciPy秒数 | 自实现秒数 | 对本组固定最快基线比 |", "|---|---:|---:|---:|---:|"]
        for key in ["stage6_confirm128", "stage6_confirm256"]:
            if key not in summary["stages"]:
                continue
            stage = summary["stages"][key]
            for name, comp in stage["versus_fastest_valid_baseline"].items():
                for pair in comp["by_geometry"]:
                    seed = str(pair["seed"])
                    bt = [stage["methods"][b]["by_geometry"][seed]["total_seconds"]["median"] for b in BASELINES]
                    lines.append(f"| {stage['n']}/{name}/{seed} | {f(pair['numerator_seconds'],4)} | {f(bt[0],4)} | {f(bt[1],4)} | {f(pair['time_ratio'],4)} |")
    return lines


def online_plots(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    stages = summary["stages"]
    if not stages:
        return
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(folder / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, len(stages), figsize=(5.8*len(stages),9), squeeze=False, constrained_layout=True)
    for col, (name, stage) in enumerate(stages.items()):
        methods, names = stage["methods"], list(stage["methods"])
        for row, field, ylabel in [(0, "total_seconds", "Complete 20-step sequence (s)"), (1, "iterations", "PCG iterations per sequence")]:
            values = np.array([methods[m][field]["median"] for m in names])
            low = np.array([methods[m][field]["min"] for m in names])
            high = np.array([methods[m][field]["max"] for m in names])
            ax = axes[row,col]
            ax.bar(range(len(names)), values, color=["#9da4aa" if m in BASELINES else "#d49a58" if m == "lagged_balanced4" else "#4b88a8" for m in names])
            ax.errorbar(range(len(names)), values, yerr=[values-low,high-values], fmt="none", capsize=3, color="#333333")
            ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{'Development' if name.endswith('screen') else 'Frozen confirmation'} {stage['n']}\n{stage['independent_geometries']} geometries; {stage['repeats_per_geometry']} repeats")
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Causal event-space mechanisms: complete online costs; geometry medians and ranges")
    save(fig,"online_comparison")
    fig, axes = plt.subplots(3,len(stages),figsize=(5.8*len(stages),12.5),squeeze=False,constrained_layout=True)
    for col,(name,stage) in enumerate(stages.items()):
        methods,names = stage["methods"],list(stage["methods"])
        for row,field,ylabel in [(0,"total_seconds","Time by step category (s / sequence)"), (1,"iterations","Iterations by step category")]:
            ax,bottom = axes[row,col],np.zeros(len(names))
            for group,label,color in [("initial","Initial step (no past space)","#9da4aa"),("event","Later event steps","#d49a58"),("non_event","Non-event steps","#4b88a8")]:
                values=np.array([methods[m]["categories"][group][field]["median"] for m in names])
                ax.bar(range(len(names)),values,bottom=bottom,label=label,color=color)
                bottom+=values
            ax.set_xticks(range(len(names)),names,rotation=50,ha="right",fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Grid {stage['n']}; {'development' if name.endswith('screen') else 'confirmation'}")
            if col==len(stages)-1:
                ax.legend(fontsize=7,loc="upper left",bbox_to_anchor=(1.01,1))
        ax=axes[2,col]
        for offset,field,label in [(-.25,"update_seconds","Pre-solve update"),(0.,"extraction_seconds","Post-solve extraction"),(.25,"correction_seconds","Projection / directions")]:
            ax.bar(np.arange(len(names))+offset,[methods[m][field]["median"] for m in names],width=.23,label=label)
        ax.set_xticks(range(len(names)),names,rotation=50,ha="right",fontsize=7)
        ax.set_ylabel("Direction cost (s / complete sequence)")
        ax.set_title("Update is separate; extraction lies inside maintenance")
        ax.legend(fontsize=7)
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("Timing and iteration coverage: initial step separated from later events\nEach component is aggregated separately; stacked medians need not equal the whole-sequence median")
    save(fig,"event_cost_breakdown")


def probe_markdown(probe):
    lines=["## 共同基础轨迹上的离线诊断", "",
        "三个开发几何各七个非首事件，共 21 个系统。所有探针对同一当前 A、M 和基础历史初值进行比较；历史轨迹始终由基础求解推进，直接参考解不驱动历史。上一事件和首步方向在时间上可得；current_retrospective 需要当前系统已完成轨迹，只是事后有限空间参考，不能用于在线性能声明。",
        "下面先在每个几何内对七个事件取中位数，再对三个几何取中位数。error 为 eᵀAe 相对基础初值的比值；slow 为当前事后四维参考空间中的 A 投影误差能量比；空间重合度为当前参考空间被候选空间覆盖的平均平方主角余弦。当前参考只有至多 24 个已完成增量，不能当成精确完整慢特征空间。", "",
        "| 方向 | error能量比 | slow剩余能量比 | 参考空间重合度 | warm迭代/基础 | balanced迭代/基础 | balanced/warm迭代 | warm/balanced失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,value in probe["directions"].items():
        geom=value["geometric_medians"]
        vals=[geom["error_energy_ratio"]["median"],geom["slow_error_energy_ratio"]["median"],geom["reference_space_overlap"]["median"],
              value["solves"]["warm"]["iterations_over_baseline"]["median"],value["solves"]["balanced"]["iterations_over_baseline"]["median"],
              geom["balanced_over_warm_iterations"]["median"]]
        lines.append(f"| {name} | "+" | ".join(f(v,4) for v in vals)+f" | {value['solves']['warm']['failures']}/{value['solves']['balanced']['failures']} |")
    update=probe["update_versus_reuse"]["geometric_medians"]
    lines += ["", "同空间、同初值的接口配对：改善事件仅计两接口均准确的系统；任何失败对另行保留。事件数是相关观测，不作为 21 个独立样本。", "",
        "| 方向 | balanced/warm迭代几何优先中位比 [范围] | 迭代差几何优先中位数 | balanced更少/相同/更多事件 | 失败对 | 中位迭代差改善几何数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,value in probe["directions"].items():
        geom=value["geometric_medians"]
        ratios=geom["balanced_over_warm_iterations"]
        counts=value["matched_interface_counts"]
        lines.append(f"| {name} | {f(ratios['median'],4)} [{f(ratios['min'],4)}, {f(ratios['max'],4)}] | {f(geom['balanced_minus_warm_iterations']['median'],1)} | {counts['balanced_fewer_iterations']}/{counts['equal_iterations']}/{counts['balanced_more_iterations']} | {counts['failed_pairs']} | {counts['geometries_with_fewer_median_iterations']}/3 |")
    lines += ["", f"有限更新相对直接复用的几何优先配对结果：warm 迭代比 {f(update['warm_iteration_ratio']['median'],4)}，balanced 迭代比 {f(update['balanced_iteration_ratio']['median'],4)}；误差能量比之差 {f(update['error_energy_ratio_difference']['median'],4)}，参考空间重合度之差 {f(update['reference_space_overlap_difference']['median'],4)}。这些差值与比值分别在同系统内计算后再按几何汇总，不能用上表中位数相减或相除替代。",
        f"基准探针 PCG 失败 {probe['baseline_failures']} 次，全部方向/接口探针失败 {probe['probe_solve_failures']} 次；直接参考解真残差最大值 {sci(probe['reference_relative_residual']['max'])}。单次更新的最大 A/M 向量作用数 {f(probe['update_a_applications']['max'],0)}/{f(probe['update_m_applications']['max'],0)}，预定预算 8/8。",
        "八维更新候选包含旧四维空间，但截取四个低端 Ritz 方向后不保证包含旧空间，也不保证对当前右端更有效。谱量、空间重合度、误差及迭代需要共同解释，不能用其中一项替代完整在线时间。",
        "离线 setup_seconds 仅为当前空间处理组件，不包括历史采集、当前事后空间提取、投影或平衡算子构造；该时间不作为可比较的完整在线成本，原始组件仅在 JSON 中保留。离线 balanced 接口探针也不是线上事件策略：本轮在线事件使用一次投影后仍由基础 M 求解，避免把 BA 缓存误读为 MA。", "",
        "完整配对系统如下，当前事后方向仅用于诊断：", "",
        "| 几何/步 | 上一方向来源步 | 基础迭代 | 上一事件warm | 更新warm | 首步冻结warm | 当前事后warm | 上一事件balanced | 更新balanced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in probe["raw_cases"]:
        vals=[row["directions"][name]["solves"]["warm"]["iterations"] for name in PROBE_DIRECTIONS]
        vals += [row["directions"][name]["solves"]["balanced"]["iterations"] for name in PROBE_DIRECTIONS[:2]]
        lines.append(f"| {row['seed']}/{row['step']} | {row['previous_source_step']} | {row['baseline']['iterations']} | "+" | ".join(str(v) for v in vals)+" |")
    return lines


def probe_plots(summary,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder=out/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    probe=summary["probe"]
    labels=["Previous event","Updated previous","Initial frozen","Current retrospective*"]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,field,title in [(axes[0,0],"error_energy_ratio","Initial error energy / baseline"),
                          (axes[0,1],"slow_error_energy_ratio","Initial slow-reference error / baseline"),
                          (axes[1,0],"reference_space_overlap","Overlap with current retrospective reference")]:
        for index,name in enumerate(PROBE_DIRECTIONS):
            value=probe["directions"][name]
            cases=[r[field] for r in value["cases"]]
            ax.scatter(index+np.linspace(-.16,.16,len(cases)),cases,s=16,alpha=.5,color="#4b88a8" if index<3 else "#d49a58")
            med=value["geometric_medians"][field]["median"]
            ax.plot([index-.22,index+.22],[med,med],color="black",linewidth=2)
        if field != "reference_space_overlap":
            ax.axhline(1.,color="grey",linestyle="--",linewidth=1)
        ax.set_xticks(range(4),labels,rotation=20,ha="right",fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y",alpha=.15)
    ax=axes[1,1]
    for offset,interface,color in [(-.16,"warm","#4b88a8"),(.16,"balanced","#d49a58")]:
        values=[probe["directions"][name]["solves"][interface]["iterations_over_baseline"]["median"] for name in PROBE_DIRECTIONS]
        ax.bar(np.arange(4)+offset,values,width=.29,label=interface,color=color)
    ax.axhline(1.,color="grey",linestyle="--",linewidth=1)
    ax.set_xticks(range(4),labels,rotation=20,ha="right",fontsize=8)
    ax.set_title("Paired iterations / baseline: median of geometry medians")
    ax.legend()
    ax.grid(axis="y",alpha=.15)
    fig.suptitle("Offline common-history diagnosis: 21 later events from 3 geometries\n*Current retrospective space requires an already-completed current solve; points are cases, black bars are geometry-first medians",fontsize=11)
    for ext in ["png","pdf"]:
        fig.savefig(folder/f"common_history_probe.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def markdown(summary):
    lines=["# 第六阶段：跳变当步的历史方向、预算更新与刷新时机", "",
        f"生成时间：{summary['created_utc']}。状态：`{summary['status']}`。", "",
        "本阶段是经典数值机制探索，没有训练或调用 AI。只使用本阶段同协议实际测得的 PDE 结果进行比较，不把跨阶段时间差归因于新算法，也不将经典收益归为 AI 创新。", ""]
    lines += probe_markdown(summary["probe"])
    lines += [""]+online_markdown(summary)
    if summary["status"] != "complete":
        lines += ["", f"按协议停止：{summary.get('stop_reason','确认尚未完成')}。不对缺失确认结果作推断。"]
    lines += ["", "## 解释边界与复核入口", "",
        "使用时机、有限更新、轨迹采集和刷新频率都可能影响迭代与成本。reuse_refresh/lagged 保持事件末刷新而改变当步初值；update_refresh/reuse_refresh 检验预算更新；transport/frozen 在两者都不做后续轨迹刷新的条件下检验持久预算更新。跨 refresh 与 frozen/transport 的比较同时改变刷新成本与空间陈旧程度，不能将复合变化完全归因于某一个组件。",
        "事件覆盖改善若未带来完整时间收益，应保留这一负结果；若经典冻结或更新已足够有效，也不能据此假定需要学习模型。不能仅因某个事后参考方向好、某个事件快或某个谱指标改善就扩大训练。",
        "20% 是预定工程投入筛选线，不是统计显著性或论文录用保证。本轮仍是单右端、单对比度和少量几何。后续改变算法、预算或协议需重新冻结并使用新的确认几何。",
        "源码、协议、分组、共同历史诊断、开发测量及选择哈希均在报告生成时核对；全部准确性、恢复、更新和缓存拒绝、逐几何配对、重复序列和计时组件见 [summary.json](summary.json)。预定设计见 [协议](../../docs/stage6_protocol_zh.md)。", "",
        "![在线完整时间与迭代](figures/online_comparison.png)", "", "![首步、事件与非事件成本](figures/event_cost_breakdown.png)", "",
        "![共同历史离线探针](figures/common_history_probe.png)", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    out=args.output.resolve() if args.output else root/"results/stage6_report"
    splits=read(root/"data/stage6_splits.json")
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),schema_version=1,
        aggregation="sum twenty steps; median repeats within geometry; then median/range and paired ratios across geometries; all failures retained",
        split_audit=check_splits(root,splits),probe=load_probe(root,splits),stages={},audit={},reporter_sha256=sha(Path(__file__)))
    manifest,rows,checks=load_online(root,"stage6_screen",splits)
    require(len(rows)==840,"Expected exactly 840 development rows")
    summary["stages"]["stage6_screen"]=aggregate(manifest,rows)
    summary["audit"]["stage6_screen"]=checks
    all_rows=list(rows)
    methods=summary["stages"]["stage6_screen"]["methods"]
    eligible=[name for name in NEW_METHODS if methods[name]["eligible"] and methods[name]["update_budget_valid"]]
    selection_path=root/"results/stage6_screen/frozen_selection.json"
    if not eligible:
        require(not selection_path.exists(),"Ineligible candidate frozen")
        summary.update(status="screen_no_eligible_candidate",stop_reason="四种新因果候选均未同时满足准确、零求解失败、零整空间拒绝及更新预算要求")
    else:
        frozen=read(selection_path)
        check_hashes(root,frozen,"frozen selection")
        require(frozen["probe_sha256"]==summary["probe"]["input_sha256"],"Frozen probe files changed")
        for key,name in [("screen_measurements_sha256","measurements.json"),("screen_manifest_sha256","manifest.json")]:
            require(frozen[key]==sha(selection_path.parent/name),"Frozen screen changed")
        expected=min(eligible,key=lambda name:methods[name]["total_seconds"]["median"])
        candidate=frozen["candidate"]
        require(candidate["name"]==expected and candidate==methods[expected]["spec"],"Selection violates prespecified rule")
        expected_specs=manifest["configuration"]["specs"][:5].copy()
        if candidate["name"] not in {s["name"] for s in expected_specs}:
            expected_specs.append(candidate)
        require(frozen["confirmation_specs"]==expected_specs and len(expected_specs) in (5,6),"Confirmation controls or de-duplication differ")
        summary.update(frozen_selection=frozen,frozen_selection_sha256=sha(selection_path))
        count=len(expected_specs)
        for n,expected_rows in [(128,300*count),(256,120*count)]:
            name=f"stage6_confirm{n}"
            m,rr,aa=load_online(root,name,splits,frozen)
            require(len(rr)==expected_rows,f"Expected {expected_rows} confirmation rows")
            summary["stages"][name]=aggregate(m,rr)
            summary["audit"][name]=aa
            all_rows.extend(rr)
        complete=read(root/"results/stage6_complete.json")
        require(complete.get("complete") is True and complete.get("selection_sha256")==sha(selection_path),"Final completion marker differs")
        require(len(all_rows)==840+420*count,"Final online inventory differs")
        summary.update(status="complete",completion_sha256=sha(root/"results/stage6_complete.json"),confirmation_methods=count)
    updates=[r["update_stats"] for r in all_rows if r["update_stats"]]
    summary["online_accuracy"]=dict(solves=len(all_rows),pcg_failures=sum(not r["converged"] for r in all_rows),
        recoveries=sum(r["solver_fallback"] for r in all_rows),final_failures=sum(not r["final_converged"] for r in all_rows),
        cache_rejections=sum(r["cache_rejected"] for r in all_rows),update_rejections=sum(r["update_rejected"] for r in all_rows),
        max_final_relative_residual=max(r["final_relative_residual"] for r in all_rows),
        max_update_a_applications=max((r["a_applications"] for r in updates),default=0),
        max_update_m_applications=max((r["m_applications"] for r in updates),default=0),
        capture_rejected_vectors=sum(r["capture_stats"].get("rejected_vectors",0) for r in all_rows),
        capture_terminal_vectors_skipped=sum(r["capture_stats"].get("skipped_terminal_vectors",0) for r in all_rows),
        extraction_rejected_vectors=sum(r["extraction"].get("rejected_vectors",0) for r in all_rows),
        extraction_dependent_vectors=sum(r["extraction"].get("dependent_vectors",0) for r in all_rows),
        update_basis_rejected_vectors=sum(r.get("basis",{}).get("rejected_vectors",0) for r in updates),
        update_extraction_rejected_vectors=sum(r.get("extraction",{}).get("rejected_vectors",0) for r in updates),
        update_extraction_dependent_vectors=sum(r.get("extraction",{}).get("dependent_vectors",0) for r in updates))
    verification=root/"results/stage6_verification.json"
    if verification.exists():
        record=read(verification)
        if "new_test_sha256" in record:
            require(record["new_test_sha256"]==sha(root/"test_event_space_update.py"),"Verification test hash differs")
        summary["prior_verification"]=dict(record=record,sha256=sha(verification))
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    (out/"REPORT_zh.md").write_text(markdown(summary))
    online_plots(summary,out)
    probe_plots(summary,out)
    print(json.dumps(dict(output=str(out),status=summary["status"],probe_cases=21,online_rows=len(all_rows)),ensure_ascii=False))


if __name__=="__main__":
    main()
