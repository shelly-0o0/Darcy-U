#!/usr/bin/env python3
"""Independently audit saved stage-13 results without importing PDE code.

Reconstructs saved QoI errors, mesh budgets, score decoding, paired costs,
and the frozen cost decision. It does not regenerate coefficient fields,
solve PDEs, or recover solution fields from fingerprints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ("fine_direct", "fine_amg_pcg", "jacobi8", "coefficient_local",
           "coarse_goal_local", "coarse_goal_galerkin")
SPLITS = ("id", "contrast")
PRIMARY = "coarse_goal_local"
BASELINES = ("fine_direct", "fine_amg_pcg", "jacobi8", "coarse_goal_galerkin")
LABELS = dict(fine_direct="Fine direct", fine_amg_pcg="Fine AMG-PCG",
              jacobi8="Jacobi 8", coefficient_local="Coefficient proposal",
              coarse_goal_local="Coarse-goal local", coarse_goal_galerkin="Coarse-goal Galerkin")


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
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(actual, expected, label, rtol=1e-9, atol=1e-14):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"Recomputed value differs: {label}")


def key(row):
    return row["nc"], row["split"], row["geometry"], row["method"]


def stat(values):
    values = list(values)
    valid = [float(v) for v in values if v is not None]
    return dict(median=median(valid) if valid else None,
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
                valid_count=len(valid), missing_count=len(values)-len(valid))


def added_dofs(cells, nc):
    chosen = set(cells)
    return len(chosen) + sum((c % nc + 1 < nc and c + 1 in chosen) + (c + nc in chosen)
                             for c in chosen)


def independent_node_hash(cells,nc):
    nf=2*nc;chosen=set(cells)
    nodes={y*(nf+1)+x for y in range(2,nf,2) for x in range(2,nf,2)}
    for c in cells:
        y,x=divmod(c,nc)
        nodes.add((2*y+1)*(nf+1)+2*x+1)
        if x+1<nc and c+1 in chosen:
            nodes.add((2*y+1)*(nf+1)+2*x+2)
        if y+1<nc and c+nc in chosen:
            nodes.add((2*y+2)*(nf+1)+2*x+1)
    return hashlib.sha256(np.asarray(sorted(nodes),dtype=np.int64).tobytes()).hexdigest()


def independently_mark(scores, nc, budget):
    order = sorted(range(nc * nc), key=lambda c: (-scores[c], c))
    chosen = set(); spent = 0
    for cell in order:
        y, x = divmod(cell, nc)
        neighbors = [n for valid, n in ((x > 0, cell-1), (x+1 < nc, cell+1),
                                        (y > 0, cell-nc), (y+1 < nc, cell+nc)) if valid]
        cost = 1 + sum(n in chosen for n in neighbors)
        if spent + cost <= budget:
            chosen.add(cell); spent += cost
        if spent == budget:
            break
    fallback = spent != budget
    parity = None
    if fallback:
        options = []
        for p in (0, 1):
            cells = [c for c in order if sum(divmod(c, nc)) % 2 == p][:budget]
            require(len(cells) == budget, "Insufficient checkerboard budget")
            options.append((sum(scores[c] for c in cells), -p, cells))
        _, negative_parity, cells = max(options, key=lambda entry: (entry[0], entry[1]))
        chosen = set(cells); parity = -negative_parity
    return sorted(chosen), fallback, spent, parity


def audit(source):
    frozen = read(source / "freeze.json"); complete = read(source / "complete.json")
    config = frozen["config"]
    require(complete["complete"] and complete["smoke"] == frozen["smoke"], "Incomplete/mismatched run")
    require(frozen["no_training"] and complete["no_training"], "Unexpected AI training")
    require(sha(source / "freeze.json") == complete["freeze_sha256"], "Frozen manifest changed")
    require(set(complete["output_sha256"]) == {"rows.json", "contexts.json"}, "Missing output hashes")
    for name, digest in complete["output_sha256"].items():
        require(sha(source / name) == digest, f"Raw output changed: {name}")
    for name, digest in frozen["input_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Nonportable frozen path: {name}")
        require(sha(ROOT / path) == digest, f"Frozen input changed: {name}")
    require(frozen["methods"] == list(METHODS), "Wrong method catalogue")
    require(set(config["splits"]) == set(SPLITS), "Wrong split catalogue")
    if not frozen["smoke"]:
        expected_config = dict(grids=[16,32,64], budget_divisor=4, geometries_per_split=16,
            repetitions=3, splits={"id":54000,"contrast":54100}, source=0, goals=[1,2,3],
            order_seed=54991, warmup_geometry=54950, primary_method=PRIMARY, gate_nc=64,
            time_ratio_limit=.8, minimum_wins=12, pcg_rtol=1e-10, pcg_maxiter=1000,
            pcg_max_restarts=3, amg_seed=51700, residual_limit=1e-8,
            near_zero_coarse_error=1e-12, fine_pcg_error_ratio_limit=1e-5,
            matched_solution_tolerance=1e-8, matched_scaled_qoi_tolerance=1e-8)
        for name, value in expected_config.items():
            require(config[name] == value, f"Formal configuration changed: {name}")
    require(frozen["environment"]["device"] == "cpu", "Expected CPU")
    require(all(p["num_threads"] == 1 for p in frozen["environment"]["threadpools"]),
            "Threadpool is not single threaded")
    require(all(frozen["environment"]["thread_environment"][v] == "1" for v in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")), "Thread environment changed")
    rows = read(source / "rows.json"); contexts = read(source / "contexts.json")
    require(finite(rows) and finite(contexts) and finite(frozen), "Nonfinite saved values")
    expected_contexts = {(nc, s, g) for nc in config["grids"] for s, first in config["splits"].items()
                         for g in range(first, first + config["geometries_per_split"])}
    expected_rows = {(*c, m) for c in expected_contexts for m in METHODS}
    require({key(r) for r in rows} == expected_rows and len(rows) == len(expected_rows) == complete["rows"],
            "Missing/duplicate method-context rows")
    require({(c["nc"], c["split"], c["geometry"]) for c in contexts} == expected_contexts
            and len(contexts) == len(expected_contexts) == complete["contexts"], "Missing/duplicate contexts")
    geometries = {(s, g) for _, s, g in expected_contexts}
    require(len(geometries) == complete["independent_geometries"] == 2 * config["geometries_per_split"],
            "Scale repeats miscounted as independent geometries")
    require(complete["fresh_pipeline_runs"] == len(rows) * config["repetitions"], "Wrong repetition count")
    historical_hashes = set(); historical_ids = set()
    for name in ("training_data.json", "test_data.json"):
        data = read(ROOT / "results/stage10_pilot" / name)
        entries = data if isinstance(data, list) else data["train"] + data["validation"]
        historical_hashes.update(r["coefficient_sha256"] for r in entries)
        historical_ids.update(r["seed"] for r in entries)
    for stage in ("stage11_refinement", "stage12_actions"):
        entries = read(ROOT / "results" / stage / "contexts.json")
        historical_hashes.update(r["coefficient_sha256"] for r in entries)
        historical_ids.update(r["geometry"] for r in entries)
    hashes = [c["coefficient_sha256"] for c in contexts]
    require(len(set(hashes)) == len(hashes) and not set(hashes) & historical_hashes,
            "Coefficient fingerprint collision")
    require(not {g for _, g in geometries} & historical_ids, "Historical geometry identity overlap")
    lookup = {key(r): r for r in rows}
    failures = []; near_zero = set(); reference = {}; remainders = []; timing_variability = []
    maxima = dict(relative_residual=0., coarse_relative_residual=0., energy_error_ratio=0.,
                  matched_solution_difference=0., matched_scaled_qoi_difference=0.,
                  pcg_relative_residual=0., pcg_error_reduction_ratio=0.)
    def numerical(condition, context, reason):
        if not condition:
            failures.append(dict(context=list(context), reason=reason))
    for c in contexts:
        nc = c["nc"]; budget = nc * nc // config["budget_divisor"]
        require(c["fine_dofs"] == (2*nc-1)**2 and c["coarse_dofs"] == (nc-1)**2
                and c["new_dofs"] == budget, "Wrong reference dimensions")
        numerical(c["fine_relative_residual"] <= config["residual_limit"],
                  (nc, c["split"], c["geometry"]), "reference_residual")
        rng=np.random.default_rng(config["order_seed"]+c["geometry"]+nc)
        orders=[[METHODS[i] for i in rng.permutation(len(METHODS))] for _ in range(config["repetitions"])]
        require(c["method_orders"]==orders,"Randomized method order differs from frozen seed")
    for row in rows:
        rk = key(row); context = rk[:3]; nc = row["nc"]; method = row["method"]
        budget = nc * nc // config["budget_divisor"]
        require(row["training_seed"] is None and not row["offline_only"], "Unexpected training/oracle")
        require(len(row["solution_sha256"]) == 64 and all(c in "0123456789abcdef"
                for c in row["solution_sha256"]), "Invalid solution hash")
        q, qf, qc, scales = (np.asarray(row[name]) for name in
                            ("qoi", "fine_qoi", "coarse_qoi", "qoi_scales"))
        require(all(x.shape == (3,) for x in (q,qf,qc,scales)) and np.all(scales > 0), "Wrong QoI data")
        close(scales, np.maximum(abs(qc), max(float(abs(qc).max())*1e-12, 1e-30)), "QoI scales", atol=1e-25)
        if context in reference:
            for a, b in zip((qf,qc,scales), reference[context]):
                close(a,b,"shared reference",atol=1e-25)
        else:
            reference[context] = qf,qc,scales
        error=qf-q; coarse=qf-qc
        scaled=float(np.linalg.norm(error/scales)); denom=float(np.linalg.norm(coarse/scales))
        close(row["signed_qoi_errors"],error,"signed target errors",atol=1e-22)
        close(row["signed_coarse_errors"],coarse,"signed coarse errors",atol=1e-22)
        close(row["absolute_error_l2"],np.linalg.norm(error),"absolute error",atol=1e-22)
        close(row["scaled_error_l2"],scaled,"scaled error")
        close(row["coarse_scaled_error"],denom,"coarse scaled error")
        degenerate=denom < config["near_zero_coarse_error"]
        require(row["coarse_error_near_zero"] == degenerate,"Wrong degenerate denominator flag")
        if degenerate:
            near_zero.add(context)
            require(row["error_reduction_ratio"] is None,"Degenerate ratio must be null")
        else:
            close(row["error_reduction_ratio"],scaled/denom,"error reduction ratio")
        numerical(row["solver_converged"] and row["numerical_valid"]
                  and row["all_repeats_numerically_valid"],rk,"solver_or_repeat_validity")
        numerical(row["relative_residual"] <= config["residual_limit"],rk,"solve_residual")
        numerical(-1e-12 <= row["energy_error_ratio"] <= 1+1e-8,rk,"energy_monotonicity")
        numerical(row["coarse_scaled_qoi_discrepancy"] <= 1e-8,rk,"coarse_qoi_discrepancy")
        maxima["relative_residual"] = max(maxima["relative_residual"],row["relative_residual"])
        maxima["energy_error_ratio"] = max(maxima["energy_error_ratio"],row["energy_error_ratio"])
        if row.get("coarse_relative_residual") is not None:
            numerical(row["coarse_relative_residual"] <= 1e-8,rk,"coarse_solve_residual")
            maxima["coarse_relative_residual"] = max(maxima["coarse_relative_residual"],row["coarse_relative_residual"])
        if method.startswith("fine_"):
            require(row["selected_cells"] == [] and row["solve_dofs"] == (2*nc-1)**2
                    and not row["budget_matched"] and row["added_dofs"] == (2*nc-1)**2-(nc-1)**2,
                    "Fine baseline incorrectly budget matched")
            require(row["uses_global_fine_matrix"],"Fine baseline lacks fine matrix")
            if method == "fine_direct":
                close(q,qf,"direct reference QoIs",atol=1e-16)
            else:
                solver = row["solver"]
                numerical(solver["converged"] and solver["info"] == 0
                          and row["relative_residual"] <= config["pcg_rtol"],rk,"fine_pcg_convergence")
                numerical(row["error_reduction_ratio"] is not None and
                          row["error_reduction_ratio"] <= config["fine_pcg_error_ratio_limit"],rk,"fine_pcg_qoi_accuracy")
                close(solver["relative_residual"],row["relative_residual"],"PCG actual residual")
                maxima["pcg_relative_residual"] = max(maxima["pcg_relative_residual"],row["relative_residual"])
                maxima["pcg_error_reduction_ratio"] = max(maxima["pcg_error_reduction_ratio"],row["error_reduction_ratio"] or 0.)
        else:
            cells = row["selected_cells"]
            require(cells == sorted(set(cells)) and all(isinstance(c,int) and 0 <= c < nc*nc for c in cells),
                    "Invalid selected cell IDs")
            require(added_dofs(cells,nc) == row["added_dofs"] == budget and row["budget_matched"]
                    and row["solve_dofs"] == (nc-1)**2+budget,"Wrong true DOF budget")
            scores = row["scores"] if method == "jacobi8" else row["marking"]["scores"]
            require(len(scores) == nc*nc and min(scores) >= 0,"Invalid proposal scores")
            chosen,fallback,spent,parity = independently_mark(scores,nc,budget)
            mark = row["marking"]
            require(chosen == cells and fallback == mark["fallback"]
                    and parity == mark["fallback_parity"] and spent == mark["greedy_spent_before_fallback"],
                    "Budget decoder differs from saved scores")
            require(mark["added_dofs"] == mark["requested_budget"] == budget
                    and mark["selected_cell_count"] == len(cells),"Wrong decoder metadata")
            close(mark["score_sum"],sum(scores[c] for c in cells),"selected score sum")
            expected_triangles = 2*nc*nc + 6*len(cells)
            if method == "jacobi8":
                require(row["geometric_triangle_count"] == expected_triangles,"Wrong historical triangle count")
                numerical(row["coarse_embedding_max_abs"] <= 1e-12,rk,"coarse_embedding")
            elif method.endswith("galerkin"):
                require(row["local_assembly"]["geometric_triangle_count"] == expected_triangles,"Wrong Galerkin triangle count")
            else:
                metadata=row["local_assembly"]["metadata"]
                require(metadata["geometric_triangle_count"] == expected_triangles
                        and metadata["added_dofs"] == budget and metadata["selected_cells"] == cells,
                        "Local assembly mesh metadata mismatch")
                require(not metadata["global_fine_sparse_operator_assembled"]
                        and not metadata["fine_reference_operator_assembled"]
                        and not metadata["global_sparse_interpolation_assembled"]
                        and not metadata["global_schur_assembled"],"Local path recorded a global reference operator")
                require(metadata["fine_quadrature_triangle_count"] == 2*(2*nc)**2,
                        "Fine quadrature work missing")
                require(metadata["reused_load_data"] == (method == PRIMARY),"Wrong within-pipeline quadrature reuse")
            if method!="jacobi8":
                require(row["independent_node_sha256"]==independent_node_hash(cells,nc),
                        "Independent node IDs differ from reconstructed true space")
            require(row["uses_global_fine_matrix"] == (method in ("jacobi8","coarse_goal_galerkin")),
                    "Wrong fine operator accounting")
        repeat_checks=row["repeat_numerical_checks"]
        require(len(repeat_checks)==config["repetitions"] and {c["repetition"] for c in repeat_checks}
                ==set(range(config["repetitions"])),"Missing repeated numerical checks")
        require(row["all_repeats_numerically_valid"]==all(c["numerical_valid"] for c in repeat_checks),
                "Repeated numerical aggregate flag mismatch")
        for check in repeat_checks:
            numerical(check["numerical_valid"] and check["solver_converged"]
                      and check["relative_residual"]<=config["residual_limit"],(*rk,check["repetition"]),"repeat_solver")
            if method=="fine_amg_pcg":
                numerical(check["solver_info"]==0 and check["relative_residual"]<=config["pcg_rtol"]
                          and check["error_reduction_ratio"] is not None
                          and check["error_reduction_ratio"]<=config["fine_pcg_error_ratio_limit"],
                          (*rk,check["repetition"]),"repeat_fine_pcg")
        first=next(c for c in repeat_checks if c["repetition"]==0)
        close(first["relative_residual"],row["relative_residual"],"first-repeat residual")
        require(first["numerical_valid"]==row["numerical_valid"],"First-repeat validity mismatch")
        timings = row["timings"]
        require(len(timings) == config["repetitions"] and {t["repetition"] for t in timings}
                == set(range(config["repetitions"])),"Missing timing repeats")
        totals = [t["total_seconds"] for t in timings]
        require(min(totals) > 0,"Nonpositive full pipeline time")
        close(row["median_total_seconds"],median(totals),"repeat median",atol=1e-15)
        timing_variability.append(dict(context=list(rk),minimum=min(totals),maximum=max(totals),
                                       max_over_min=max(totals)/min(totals)))
        for t in timings:
            components = {name:value for name,value in t.items() if name.endswith("_seconds") and name != "total_seconds"}
            require(min(components.values()) >= 0,"Negative component time")
            remainder=t["total_seconds"]-sum(components.values())
            require(remainder >= -1e-8,"Double-counted top-level timing components")
            remainders.append(remainder)
    for context in sorted(expected_contexts):
        local=lookup[(*context,PRIMARY)]; gal=lookup[(*context,"coarse_goal_galerkin")]
        require(local["selected_cells"] == gal["selected_cells"] and local["solve_dofs"] == gal["solve_dofs"],
                "Matched implementations selected different spaces")
        close(local["marking"]["scores"],gal["marking"]["scores"],"same proposal scores")
        require(local["matched_implementation"] == gal["matched_implementation"],"Different matched audit records")
        match=local["matched_implementation"]
        qdifference=float(np.linalg.norm((np.asarray(local["qoi"])-np.asarray(gal["qoi"]))/local["qoi_scales"]))
        close(match["scaled_qoi_difference"],qdifference,"matched QoI discrepancy")
        numerical(match["same_mesh"] and match["same_dofs"] and match["same_independent_nodes"] and match["passed"]
                  and match["relative_solution_difference"] <= config["matched_solution_tolerance"]
                  and qdifference <= config["matched_scaled_qoi_tolerance"],context,"matched_implementation")
        maxima["matched_solution_difference"] = max(maxima["matched_solution_difference"],match["relative_solution_difference"])
        maxima["matched_scaled_qoi_difference"] = max(maxima["matched_scaled_qoi_difference"],qdifference)
    audited=dict(passed=not failures,frozen_input_count=len(frozen["input_sha256"]),rows=len(rows),
        contexts=len(contexts),independent_geometries=len(geometries),
        budget_matched_rows=sum(not r["method"].startswith("fine_") for r in rows),
        fresh_pipeline_runs=complete["fresh_pipeline_runs"],numeric_failures=failures,
        failed_fine_pcg_contexts=[dict(nc=r["nc"],split=r["split"],geometry=r["geometry"],
            **r["solver"],error_reduction_ratio=r["error_reduction_ratio"],
            failed_repeats=sum(not c["numerical_valid"] for c in r["repeat_numerical_checks"]))
            for r in rows if r["method"]=="fine_amg_pcg" and not r["all_repeats_numerically_valid"]],
        coarse_denominator_degenerate_contexts=[list(c) for c in sorted(near_zero)],maxima=maxima,
        timing_variability=timing_variability,unclassified_timing_remainder_seconds=stat(remainders),
        independent_checks=["frozen and raw output hashes", "historical geometry identities and coefficient fingerprints",
            "three-QoI scales and errors", "same-context paired timing aggregation", "true added DOFs",
            "saved-score deterministic budget decoder including fallback", "matched proposal scores and chosen cells",
            "matched QoI difference", "recorded convergence, energy, residual and repeat flags"],
        limits=["No PDE solver or numerical experiment module is imported.",
            "Raw coefficient arrays and coarse solution fields are not persisted; score formula values are not regenerated.",
            "Full solution fields are not persisted; their hashes cannot reconstruct energy norms, residuals, or field agreement.",
            "All repetitions retain solver/residual/error validity summaries; full repeated fields, selected meshes and energies are not persisted, so their determinism remains a frozen runner check.",
            "Independent sparse-assembly equivalence is established by frozen numerical tests, not reconstructed by this report."])
    return frozen,complete,rows,audited


def aggregate(rows,frozen,audited):
    config=frozen["config"];lookup={key(r):r for r in rows};totals={};pairs=[];remaining=[]
    for nc in config["grids"]:
        totals[str(nc)]={}
        for split in SPLITS:
            numbers=sorted({r["geometry"] for r in rows if r["nc"]==nc and r["split"]==split})
            totals[str(nc)][split]={}
            for method in METHODS:
                entries=[lookup[(nc,split,g,method)] for g in numbers]
                metrics=("error_reduction_ratio","energy_error_ratio","absolute_error_l2",
                         "scaled_error_l2","median_total_seconds","relative_residual")
                item={m:stat(r[m] for r in entries) for m in metrics}
                item.update(independent_geometries=len(numbers),
                    worse_than_coarse_count=sum(r["error_reduction_ratio"] is not None and r["error_reduction_ratio"]>1 for r in entries),
                    fallback_count=sum(r.get("marking",{}).get("fallback",False) for r in entries),
                    selected_cell_count=stat(len(r["selected_cells"]) for r in entries),
                    timing_components_seconds={name:stat(median(t[name] for t in r["timings"]) for r in entries)
                        for name in sorted(set.intersection(*(set(r["timings"][0]) for r in entries))-{ "repetition" })},
                    paired={})
                if method=="fine_amg_pcg":
                    item["pcg"]={name:stat(r["solver"][name] for r in entries)
                                 for name in ("iterations","restarts","relative_residual")}
                    item["pcg"]["failed_count"]=sum(not r["solver"]["converged"] or r["solver"]["info"]!=0 for r in entries)
                for baseline in BASELINES:
                    selected=[]
                    for r in entries:
                        b=lookup[(nc,split,r["geometry"],baseline)]
                        ratio=r["median_total_seconds"]/b["median_total_seconds"]
                        er=None if baseline.startswith("fine_") or b["error_reduction_ratio"] in (None,0) or r["error_reduction_ratio"] is None else r["error_reduction_ratio"]/b["error_reduction_ratio"]
                        record=dict(nc=nc,split=split,geometry=r["geometry"],method=method,baseline=baseline,
                                    time_ratio=ratio,error_ratio=er,
                                    baseline_numerically_valid=bool(b["numerical_valid"] and b["all_repeats_numerically_valid"]))
                        selected.append(record);pairs.append(record)
                    item["paired"][baseline]=dict(time_ratio=stat(r["time_ratio"] for r in selected),
                        strict_time_wins=sum(r["time_ratio"]<1 for r in selected),error_ratio=stat(r["error_ratio"] for r in selected),
                        strict_error_wins=sum(r["error_ratio"] is not None and r["error_ratio"]<1 for r in selected),
                        invalid_baseline_count=sum(not r["baseline_numerically_valid"] for r in selected))
                totals[str(nc)][split][method]=item
            for g in numbers:
                p=lookup[(nc,split,g,PRIMARY)]
                fastest=min(lookup[(nc,split,g,b)]["median_total_seconds"] for b in ("fine_direct","fine_amg_pcg"))
                remaining.append(dict(nc=nc,split=split,geometry=g,
                    faster_fine_baseline_seconds=fastest,primary_seconds=p["median_total_seconds"],
                    remaining_to_equal_seconds=fastest-p["median_total_seconds"],
                    remaining_to_80pct_seconds=.8*fastest-p["median_total_seconds"]))
    criteria={}
    for split in SPLITS:
        criteria[split]={}
        primary=totals[str(config["gate_nc"])][split][PRIMARY]
        for baseline in ("fine_direct","fine_amg_pcg"):
            p=primary["paired"][baseline]
            criteria[split][baseline]=dict(median_ratio=p["time_ratio"]["median"],strict_time_wins=p["strict_time_wins"],
                median_passed=p["time_ratio"]["median"]<=config["time_ratio_limit"],
                wins_passed=p["strict_time_wins"]>=config["minimum_wins"])
    nondegenerate=not audited["coarse_denominator_degenerate_contexts"]
    passed=audited["passed"] and nondegenerate and all(v["median_passed"] and v["wins_passed"]
                        for group in criteria.values() for v in group.values())
    return dict(aggregation="Median three fresh runs within geometry/method/scale, then ratios within identical geometry and scale, then split medians. Scales and timing repeats do not increase the 32 independent geometries.",
        totals=totals,paired_geometry=pairs,remaining_time_records=remaining,
        remaining_time_summary={str(nc):{s:{metric:stat(r[metric] for r in remaining if r["nc"]==nc and r["split"]==s)
            for metric in ("remaining_to_equal_seconds","remaining_to_80pct_seconds")}
            for s in SPLITS} for nc in config["grids"]},
        decision=dict(formal=not frozen["smoke"],primary_method=PRIMARY,gate_nc=config["gate_nc"],
            criteria=criteria,numerical_audit_passed=audited["passed"],nondegenerate_required_denominators=nondegenerate,
            cost_gate_passed=passed,continue_cost_structure_research=passed and not frozen["smoke"],
            same_accuracy_speedup_established=False,ai_training_performed=False,
            note="Smoke only; no formal conclusion." if frozen["smoke"] else
            ("The frozen cost-only gate passes. Actual trained proposal quality and inference cost remain untested."
             if passed else "The primary coarse-solve plus single-refinement pipeline fails the frozen cost-only gate. Do not replace it with a secondary method or claim learned acceleration.")))


def fmt(value,digits=5):
    return "未定义" if value is None else f"{value:.{digits}g}"


def report_text(summary,frozen,complete,audited,source):
    config=frozen["config"];decision=summary["decision"];totals=summary["totals"]
    lines=["# 第十三阶段：移除全局 Schur 后的成本可行性", "",
        "本报告仅读取冻结输出，不导入 PDE 求解器、数值装配器或实验运行器。没有训练、加载或测试 AI 模型。", "",
        f"正式实验：**{'否（烟雾检查）' if frozen['smoke'] else '是'}**；**{audited['independent_geometries']} 个独立几何**，在 {len(config['grids'])} 个尺度上形成 **{audited['contexts']} 个上下文**、**{audited['rows']} 条方法记录**和 **{audited['fresh_pipeline_runs']} 次完整路径计时**。尺度重复、三目标和计时重复均不是额外独立样本。", "",
        f"主候选 `coarse_goal_local` 在 nc={config['gate_nc']} 的预定**成本门槛：{'通过' if decision['cost_gate_passed'] else '未通过'}**。数值审计：{'通过' if audited['passed'] else '未通过'}；粗误差分母退化上下文 {len(audited['coarse_denominator_degenerate_contexts'])} 个。", "",
        "## 预定成本判据", "",
        f"每组分别要求主候选相对细直接和细 AMG-PCG 的逐几何时间比中位数≤{config['time_ratio_limit']:.2f}，并至少 {config['minimum_wins']}/{config['geometries_per_split']} 个几何严格更快。所有预算、数值、重复和同网格实现核验均须通过。该开发门槛不包含等精度条件。", "",
        "| 测试组 | 细基线 | 配对完整时间比中位数 | 严格更快 | 时间比条件 | 个数条件 |",
        "|---|---|---:|---:|---|---|"]
    for split in SPLITS:
        for baseline, value in decision["criteria"][split].items():
            lines.append(f"| {split} | {LABELS[baseline]} | {fmt(value['median_ratio'],7)} | {value['strict_time_wins']}/{config['geometries_per_split']} | {value['median_passed']} | {value['wins_passed']} |")
    lines += ["", "所有比值先在同一几何、同一尺度内计算，再跨几何取中位数；不能用两个组中位数相除替代。表内时间条件只检查测得的时间；失败基线的计时被保留，但相关时间比仅作描述，不能作为对有效基线的速度优势。完整门槛同时要求全部数值通过。未通过时不能换成次要系数代理重新判定，通过时也不能继承第十二阶段联合搜索的准确性或宣称未来网络已获得加速。", "",
              "## 全部方法的实际误差和完整成本", "",
              "E 为三个目标误差按粗目标值缩放后二范数，相对于同一粗解误差范数的比例；粗解 E=1，指定细直接参考 E=0。各尺度的细参考不同，这些 E 不是相对于共同连续真解的网格收敛误差。Fine AMG-PCG 的目标偏差据实保留，没有强行设为0。", ""]
    for nc in config["grids"]:
        lines += [f"### nc={nc}，粗/适应/细自由度 {(nc-1)**2}/{(nc-1)**2+nc*nc//4}/{(2*nc-1)**2}", "",
            "| 方法 | ID E中位数 [范围] | 高对比 E中位数 [范围] | ID完整时间 ms | 高对比完整时间 ms | ID/高对比 E>1个数 | ID/高对比预算回退 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            a=totals[str(nc)]["id"][method];b=totals[str(nc)]["contrast"][method]
            error=lambda v:f"{fmt(v['median'])} [{fmt(v['minimum'])}, {fmt(v['maximum'])}]"
            lines.append(f"| {LABELS[method]} | {error(a['error_reduction_ratio'])} | {error(b['error_reduction_ratio'])} | {fmt(1000*a['median_total_seconds']['median'])} | {fmt(1000*b['median_total_seconds']['median'])} | {a['worse_than_coarse_count']}/{b['worse_than_coarse_count']} | {a['fallback_count']}/{b['fallback_count']} |")
        lines += [""]
    lines += ["细基线求解全部细自由度，不匹配适应预算。适应方法均恰好新增 nc²/4 个独立自由度：中心加被释放的共享边中点，不能把选中方格数当作预算。Coarse-goal local 与 Galerkin 使用完全相同的代理、网格和积分，误差应重合；其差别是最终装配实现。", "",
        "## 相同网格的实现一致性与强基线", "",
        f"同网格直接/全域 Galerkin 路径的最大已记录解相对差 **{fmt(audited['maxima']['matched_solution_difference'],7)}**，可由保存目标值独立重建的最大缩放目标差 **{fmt(audited['maxima']['matched_scaled_qoi_difference'],7)}**。", "",
        "| nc | 组 | Local/Galerkin 配对时间比 | Coarse-goal/Jacobi8 配对 E比 | 改善数 | PCG迭代中位数 [范围] | PCG最大 E | PCG最大残差 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for nc in config["grids"]:
        for split in SPLITS:
            p=totals[str(nc)][split][PRIMARY];pcg=totals[str(nc)][split]["fine_amg_pcg"];it=pcg["pcg"]["iterations"]
            lines.append(f"| {nc} | {split} | {fmt(p['paired']['coarse_goal_galerkin']['time_ratio']['median'])} | {fmt(p['paired']['jacobi8']['error_ratio']['median'])} | {p['paired']['jacobi8']['strict_error_wins']}/{config['geometries_per_split']} | {fmt(it['median'])} [{fmt(it['minimum'])}, {fmt(it['maximum'])}] | {fmt(pcg['error_reduction_ratio']['maximum'])} | {fmt(pcg['relative_residual']['maximum'])} |")
    lines += ["", "PCG 必须实际残差≤1e-10、converged 为真且 info=0，并且 E≤1e-5；任何不合格记录均使成本门槛不通过。预算回退、误差变差和未通过记录均保留。能量单调性是嵌套空间性质，不能代替目标误差改善。", "",
        f"未收敛 AMG 上下文共 **{len(audited['failed_fine_pcg_contexts'])} 个**；以下列首个重复的诊断，失败重复数逐项保存。库退出状态 info=0 不能代替实际残差判据，目标误差很小也不允许事后放宽门槛。", "",
        "| nc/组/几何 | 初始实际残差 | 最终实际残差 | 迭代 | 重启 | info | converged | E | 失败重复 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in audited["failed_fine_pcg_contexts"]:
        lines.append(f"| {r['nc']}/{r['split']}/{r['geometry']} | {fmt(r['initial_relative_residual'],8)} | {fmt(r['relative_residual'],8)} | {r['iterations']} | {r['restarts']} | {r['info']} | {r['converged']} | {fmt(r['error_reduction_ratio'],8)} | {r['failed_repeats']}/{config['repetitions']} |")
    if not audited["failed_fine_pcg_contexts"]:
        lines.append("| 无 | — | — | — | — | — | — | — | — |")
    lines += ["",
        "## 全成本及剩余时间", "",
        "直接局部路径省去全域细稀疏矩阵、稀疏 T 和 Schur，并未省去输入系数的细采样和全域细三角三点二阶求积。首次粗装配支付四载荷求积，适应装配只在同一条新建 pipeline 内复用这份已收费数据。两种同网格后端使用相同复用规则；不跨方法、重复、几何共享。细直接/AMG 基线与直接局部路径共用同一载荷求积实现，避免把通用载荷装配优化冒充局部网格优势。", "",
        "所有计时包含系数生成、所需积分、粗装配/分解及四解、特征、预算解码、适应装配/分解/主解、三个目标及实际残差；fine AMG 还包含层级设置和全部迭代。交付物为适应系数和三个目标。细场提升、独立参考与误差评价在两条同网格路径的计时外。历史 Jacobi8 路径保留其原有完整实现成本。", "",
        "| nc | 组 | 粗装配 ms | 粗分解＋四解 ms | 特征 ms | 预算解码 ms | 适应装配 ms | 适应求解＋目标 ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    component_names=("coarse_assembly_seconds","coarse_factor_four_solve_seconds","proposal_features_seconds",
                     "budget_decode_seconds","adaptive_assembly_seconds","adaptive_solve_qoi_seconds")
    for nc in config["grids"]:
        for split in SPLITS:
            t=totals[str(nc)][split][PRIMARY]["timing_components_seconds"]
            lines.append(f"| {nc} | {split} | "+" | ".join(fmt(1000*t[n]["median"]) for n in component_names)+" |")
    lines += ["", "上表仅解释主候选构成；各分量单独取中位数，其和不必等于完整时间中位数。完整路径还含系数/载荷函数生成与计时余量。全部逐重复分量和范围保存在 summary.json。", "",
        "| nc | 组 | 相对逐几何更快细基线的余量 ms | 达到该基线80%时间的余量 ms |",
        "|---:|---|---:|---:|"]
    for nc in config["grids"]:
        for split in SPLITS:
            v=summary["remaining_time_summary"][str(nc)][split]
            lines.append(f"| {nc} | {split} | {fmt(1000*v['remaining_to_equal_seconds']['median'])} | {fmt(1000*v['remaining_to_80pct_seconds']['median'])} |")
    lines += ["", f"余量在每个尺度保留全部 {audited['independent_geometries']} 个几何，以两种已测细基线时间的逐几何最小值减主候选完整时间；负数表示当前路径已超时。当前计时集合含 {len(audited['failed_fine_pcg_contexts'])} 个 AMG 未收敛上下文，它们没有从上述记账中删除。因此包含不合格基线的组，其余量只作诊断，不能解释为合格速度优势或可用推理预算。这也尚未支付任何新模型推理、额外特征或训练成本，不能当成已实现的网络加速。", "",
        f"独立预热 **{fmt(complete['warmup_seconds'],7)} s**；正式循环墙钟 **{fmt(complete['experiment_wall_seconds'],7)} s**。墙钟含离线参考、评价和序列化，不等于在线时间之和。", "",
        "## 独立审计和限制", "",
        f"验证了 **{audited['frozen_input_count']} 个冻结输入哈希**、全部原始输出哈希、历史几何身份/系数指纹无重叠、全部目标误差、真实预算、保存分数对应的确定性选择，以及同网格目标一致性。数值失败记录 **{len(audited['numeric_failures'])} 条**，全部保存在 summary.json 的 audit.numeric_failures。", "",
        "报告没有重跑 PDE。完整系数场、粗解和细解未存储，不能从字段哈希重建代理分数公式、能量范数、残差或解场相等性；这些由冻结装配测试与运行器核查。报告可重算已保存目标和分数的相关逻辑，并检查保存的重复有效性标记，但不把标记等同于再次独立求解。", "",
        "![Actual error, complete cost and paired time ratios](figures/comparison.png)", "",
        f"原始目录：`{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`。复现报告：`python report_stage13.py --source <原始目录> --output <新报告目录>`。", ""]
    return "\n".join(lines)


def plot(summary,frozen,output):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/darcy-u-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    grids=frozen["config"]["grids"];x=np.arange(len(grids));total=summary["totals"]
    colors=dict(zip(METHODS,["#303030","#7a6aa9","#d68b26","#3a9c83","#216faf","#cc5270"]))
    styles={"id":"-","contrast":"--"};split_colors={"id":"#216faf","contrast":"#d68b26"}
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for split in SPLITS:
        for method in METHODS:
            time=[1000*total[str(nc)][split][method]["median_total_seconds"]["median"] for nc in grids]
            axes[0,1].plot(x,time,styles[split],marker="o",color=colors[method],lw=1.6,ms=4)
            if method!="coarse_goal_galerkin":
                err=[total[str(nc)][split][method]["error_reduction_ratio"]["median"] for nc in grids]
                axes[0,0].plot(x,err,styles[split],marker="o",color=colors[method],lw=1.6,ms=4)
        for ax,baseline in ((axes[1,0],"fine_direct"),(axes[1,1],"fine_amg_pcg")):
            offset=-.07 if split=="id" else .07
            for index,nc in enumerate(grids):
                records=[r for r in summary["paired_geometry"] if r["nc"]==nc
                         and r["split"]==split and r["method"]==PRIMARY and r["baseline"]==baseline]
                values=[r["time_ratio"] for r in records]
                pos=index+offset
                positions=pos+np.linspace(-.035,.035,len(records))
                ax.scatter(positions,values,s=17,alpha=.55,
                           color=split_colors[split],edgecolors="none")
                failed=[i for i,r in enumerate(records) if not r["baseline_numerically_valid"]]
                if failed:
                    ax.scatter(positions[failed],np.asarray(values)[failed],s=55,marker="x",color="#be2633",lw=1.8,zorder=5)
                ax.plot([pos-.06,pos+.06],[median(values)]*2,lw=3,color=split_colors[split])
    for ax in axes.ravel():
        ax.set_xticks(x);ax.set_xticklabels(grids);ax.set_xlabel("Coarse cells per side, nc")
        ax.grid(axis="y",alpha=.2)
    axes[0,0].axhline(1,color="#666",ls=":",lw=1)
    axes[0,0].set_title("A  Actual QoI error (same-mesh controls coincide)",loc="left",fontsize=11,fontweight="bold")
    axes[0,0].set_ylabel("Three-QoI error / coarse error (median)")
    axes[0,1].set_title("B  Complete online cost, including fine quadrature",loc="left",fontsize=11,fontweight="bold")
    axes[0,1].set_ylabel("Fresh pipeline time (ms, median; log scale)");axes[0,1].set_yscale("log")
    for ax,title in ((axes[1,0],"C  Primary / fine direct"),(axes[1,1],"D  Primary / fine AMG-PCG")):
        ax.axhline(1,color="#555",ls="--",lw=1);ax.axhline(.8,color="#888",ls=":",lw=1)
        ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
        ax.set_ylabel("Within-geometry complete time ratio")
        gate_index=grids.index(frozen["config"]["gate_nc"])
        ax.axvspan(gate_index-.23,gate_index+.23,color="#e5f0e3",alpha=.45,zorder=-5)
    if summary["audit"]["failed_fine_pcg_contexts"]:
        axes[1,1].legend(handles=[Line2D([0],[0],marker="x",color="#be2633",linestyle="",label="Failed AMG baseline")],
                         loc="upper left",bbox_to_anchor=(.01,.94),fontsize=9,
                         frameon=True,framealpha=1,facecolor="white",edgecolor="none")
    handles=[Line2D([0],[0],color=colors[m],marker="o",lw=1.5,ms=4,label=LABELS[m]) for m in METHODS]
    handles += [Line2D([0],[0],color="#666",ls=styles[s],label="ID" if s=="id" else "High contrast") for s in SPLITS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=9)
    fig.suptitle("No global Schur: actual accuracy and complete cost",y=.99,fontsize=16)
    fig.text(.5,.018,"Bottom: each dot is one geometry; bars are paired medians. Shaded scale is the frozen cost gate.\nFailed-baseline timings are retained as diagnostics; their ratios do not establish a valid speedup.\nE uses a different fine reference at each scale. A cost-only pass does not establish equal-accuracy speedup or AI benefit.",
             ha="center",fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,bottom=.12,top=.83,hspace=.40,wspace=.28)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):
        fig.savefig(output/f"comparison.{ext}",dpi=180)
    plt.close(fig)


def run(source,output):
    require(source.resolve()!=output.resolve(),"Report must not overwrite raw results")
    frozen,complete,rows,audited=audit(source)
    summary=aggregate(rows,frozen,audited);summary["audit"]=audited
    output.mkdir(parents=True,exist_ok=True)
    dump(output/"summary.json",summary)
    (output/"REPORT_zh.md").write_text(report_text(summary,frozen,complete,audited,source))
    plot(summary,frozen,output/"figures")
    files=("summary.json","REPORT_zh.md","figures/comparison.png","figures/comparison.pdf")
    manifest=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),smoke=frozen["smoke"],
        source=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        reporter_sha256=sha(Path(__file__)),source_sha256={n:sha(source/n) for n in ("freeze.json","rows.json","contexts.json","complete.json")},
        output_sha256={n:sha(output/n) for n in files},decision=summary["decision"],
        no_pde_solver_imported=True,no_ai_model_used=True)
    dump(output/"complete.json",manifest)
    print(json.dumps(dict(output=str(output),audit_passed=audited["passed"],decision=summary["decision"]),ensure_ascii=False))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="results/stage13_cost")
    parser.add_argument("--output",default="results/stage13_report")
    args=parser.parse_args()
    run((ROOT/args.source).resolve(),(ROOT/args.output).resolve())


if __name__=="__main__":
    main()
