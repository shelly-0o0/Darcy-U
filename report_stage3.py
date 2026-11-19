#!/usr/bin/env python3
"""Audit and report completed stage-3 artifacts, without executing any solver.

All raw inputs remain read-only. The report separates development selection
from held-out geometry tests and never pools grids as independent tasks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import time

ROOT = Path(__file__).resolve().parent
RUNS = ["stage3_screen", "stage3_calibration_initial", "stage3_calibration_optimized",
        "stage3_test128", "stage3_test256"]
FINAL_METHODS = ["fresh_previous", "selected_classical", "best_classical",
                 "initial_network", "budget_control_network", "optimized_network"]
LABELS = {"fresh_previous": "Fresh AMG + previous", "selected_classical": "Screened classical",
          "best_classical": "Calibrated classical", "initial_network": "Initial network",
          "budget_control_network": "Equal-budget energy", "optimized_network": "Mixed-loss optimized"}
COMPONENTS = ["assembly_seconds", "guess_seconds", "correction_seconds", "setup_seconds",
              "solve_seconds", "recovery_seconds"]
TRAINING = ["correction_energy", "correction_energy_continued", "correction_mixed"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = list(values)
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {"count": len(values), "finite_count": len(finite),
            "median": median(finite) if finite else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None}


def identity(spec):
    return json.dumps({k: v for k, v in spec.items() if k != "name"}, sort_keys=True)


def actual_accuracy(row, rtol):
    value = row.get("final_relative_residual")
    return bool(row.get("final_converged")) and isinstance(value, (int, float)) and math.isfinite(value) and value <= rtol


def check_stage(root, name, splits, frozen_hash):
    out = root / "results" / name
    required = ["manifest.json", "measurements.json", "complete.json", "checkpoint_sha256.json", "references.json"]
    inputs = {f: read(out / f) for f in required}
    manifest, rows, done = inputs["manifest.json"], inputs["measurements.json"], inputs["complete.json"]
    config = manifest["configuration"]
    expected_grids = [128, 256] if name == "stage3_screen" else [256] if name == "stage3_test256" else [128]
    expected_repeats = 3 if name.startswith("stage3_test") else 2
    if config["grids"] != expected_grids or config["repeats"] != expected_repeats:
        raise ValueError(f"Unexpected grid/repetition schedule in {name}")
    expected = set()
    for n in config["grids"]:
        specs = config["specs_by_n"][str(n)]
        names = [s["name"] for s in specs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate method labels in {name}/{n}")
        expected.update((n, method, seed, repeat, step) for method in names
            for seed in config["seeds"] for repeat in range(config["repeats"]) for step in range(manifest["steps"]))
    observed = [(r["n"], r["method"], r["seed"], r["repeat"], r["step"]) for r in rows]
    if not done.get("complete") or done.get("rows") != len(rows) or len(set(observed)) != len(rows) or set(observed) != expected:
        raise ValueError(f"Incomplete, duplicate or unexpected rows: {name}; actual={len(rows)}, expected={len(expected)}")
    if manifest["steps"] != 20 or manifest["rtol"] != 1e-8 or manifest["contrast"] != 1e4:
        raise ValueError(f"Unexpected problem/accuracy settings in {name}")
    expected_group = "screen" if name == "stage3_screen" else "calibration" if "calibration" in name else "test"
    expected_seeds = splits["groups"][expected_group]
    if name == "stage3_test256":
        expected_seeds = expected_seeds[:3]
    if config["seeds"] != expected_seeds:
        raise ValueError(f"Seed group mismatch in {name}")
    if name.startswith("stage3_test"):
        if config.get("frozen_selection_sha256") != frozen_hash:
            raise ValueError(f"Frozen selection mismatch in {name}")
        for n in config["grids"]:
            if [s["name"] for s in config["specs_by_n"][str(n)]] != FINAL_METHODS:
                raise ValueError(f"The six frozen final configurations are required in {name}")
    for row in rows:
        for field in ["total_seconds", *COMPONENTS, "inference_seconds", "safeguard_seconds"]:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing/nonfinite/negative cost {name}/{field}; do not silently exclude failed costs")
    checks = {"sources": {f: (root / f).exists() and sha(root / f) == v for f, v in manifest["source_sha256"].items()},
        "protocol": sha(root / "docs/stage3_protocol_zh.md") == manifest["protocol_sha256"],
        "splits": sha(root / "data/stage3_splits.json") == manifest["splits_sha256"],
        "checkpoints": {f: (root / f).exists() and sha(root / f) == v for f, v in inputs["checkpoint_sha256.json"].items()}}
    matches = all(checks["sources"].values()) and checks["protocol"] and checks["splits"] and all(checks["checkpoints"].values())
    if name.startswith("stage3_test") and not matches:
        raise ValueError(f"Final frozen artifacts do not match current files: {name}: {checks}")
    referenced = {(r["n"], r["seed"]) for r in inputs["references.json"]}
    if referenced != {(n, s) for n in config["grids"] for s in config["seeds"]}:
        raise ValueError(f"Missing reference provenance in {name}")
    provenance = {"rows_expected": len(expected), "rows_actual": len(rows), "complete": True,
        "input_sha256": {f: sha(out / f) for f in required}, "current_artifact_matches": checks,
        "all_current_artifacts_match": matches,
        "offline_reference_seconds": sum(r["seconds"] for r in inputs["references.json"])}
    return manifest, rows, provenance


def aggregate(manifest, rows):
    config, rtol = manifest["configuration"], manifest["rtol"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n"], row["method"], row["seed"], row["repeat"])].append(row)
    units = []
    for (n, method, seed, repeat), values in sorted(grouped.items()):
        u = {"n": n, "method": method, "seed": seed, "repeat": repeat, "steps": len(values)}
        for f in ["total_seconds", *COMPONENTS, "inference_seconds", "safeguard_seconds", "iterations"]:
            u[f] = sum(r[f] for r in values)
        u["other_seconds"] = u["total_seconds"] - sum(u[f] for f in COMPONENTS)
        u["accurate"] = all(actual_accuracy(r, rtol) for r in values)
        u["pcg_failures"] = sum(not r["converged"] for r in values)
        u["solver_fallbacks"] = sum(bool(r["solver_fallback"]) for r in values)
        u["final_failures"] = sum(not actual_accuracy(r, rtol) for r in values)
        u["network_calls"] = sum(bool(r["network_called"]) for r in values)
        units.append(u)
    grids = {}
    for n in config["grids"]:
        methods = {}
        for spec in config["specs_by_n"][str(n)]:
            name = spec["name"]
            uu = [u for u in units if u["n"] == n and u["method"] == name]
            rr = [r for r in rows if r["n"] == n and r["method"] == name]
            by_seed = {}
            for seed in config["seeds"]:
                local = [u for u in uu if u["seed"] == seed]
                by_seed[str(seed)] = {f: stats(u[f] for u in local)
                    for f in ["total_seconds", "iterations", "inference_seconds", "network_calls"]}
                by_seed[str(seed)]["accurate"] = all(u["accurate"] for u in local)
                by_seed[str(seed)]["pcg_failures"] = sum(u["pcg_failures"] for u in local)
            calls = [r for r in rr if r["network_called"]]
            residual_cases = defaultdict(list)
            for r in calls:
                base = r["base_relative_residual"]
                ratio = r["corrected_relative_residual"] / base if base > 0 else None
                residual_cases[(r["seed"], r["step"])].append(ratio)
            ratios = [stats(v)["median"] for v in residual_cases.values()]
            result = {"spec": spec, "by_seed": by_seed,
                "total_seconds": stats(v["total_seconds"]["median"] for v in by_seed.values()),
                "iterations_per_sequence": stats(v["iterations"]["median"] for v in by_seed.values()),
                "all_accurate": all(u["accurate"] for u in uu),
                "mean_breakdown_seconds": {f: mean(u[f] for u in uu) for f in [*COMPONENTS, "other_seconds"]},
                "max_iterations_one_solve": max(r["iterations"] for r in rr),
                "final_relative_residual": stats(r.get("final_relative_residual") for r in rr),
                "energy_relative_error": stats(r.get("energy_relative_error") for r in rr),
                "network_inference_seconds_per_call": stats(r["inference_seconds"] for r in calls),
                "network_unique_seed_step_cases": len(residual_cases),
                "corrected_over_base_residual_by_unique_case": stats(ratios),
                "residual_reduced_unique_cases": sum(x is not None and x < 1 for x in ratios),
                "zero_alpha_calls": sum(r["correction_alpha"] == 0 for r in calls),
                "positive_correction_energy_calls": sum(r["correction_energy_change"] > 0 for r in calls)}
            for f in ["pcg_failures", "solver_fallbacks", "final_failures", "network_calls"]:
                result[f] = sum(u[f] for u in uu)
            methods[name] = result
        aliases = defaultdict(list)
        for spec in config["specs_by_n"][str(n)]:
            aliases[identity(spec)].append(spec["name"])
        grids[str(n)] = {"independent_geometries": len(config["seeds"]), "methods": methods,
            "identical_configuration_labels": [names for names in aliases.values() if len(names) > 1]}
    return units, grids


def test_comparisons(manifest, rows, units, grids):
    result = {}
    for n in manifest["configuration"]["grids"]:
        methods = grids[str(n)]["methods"]
        classical = [name for name, r in methods.items() if r["spec"].get("correction") != "network"]
        aliases = defaultdict(list)
        for name in classical:
            aliases[identity(methods[name]["spec"])].append(name)
        per_seed = {}
        for seed in manifest["configuration"]["seeds"]:
            skey = str(seed)
            eligible = [name for name in classical if methods[name]["by_seed"][skey]["accurate"]]
            fastest = min(eligible, key=lambda name: methods[name]["by_seed"][skey]["total_seconds"]["median"]) if eligible else None
            best_seconds = methods[fastest]["by_seed"][skey]["total_seconds"]["median"] if fastest else None
            pooled = []
            for names in aliases.values():
                selected = [u for u in units if u["n"] == n and u["seed"] == seed and u["method"] in names]
                if all(u["accurate"] for u in selected):
                    pooled.append({"labels": names, "complete_runs": len(selected),
                                   "seconds": median(u["total_seconds"] for u in selected)})
            pooled_best = min(pooled, key=lambda p: p["seconds"]) if pooled else None
            primary_seconds = pooled_best["seconds"] if pooled_best else None
            comparison = {"fastest_accurate_classical_label_sensitivity": fastest,
                "label_min_classical_seconds_sensitivity": best_seconds,
                "classical_seconds": primary_seconds, "pooled_identical_classical_best": pooled_best,
                "networks": {}}
            for name in FINAL_METHODS[3:]:
                r = methods[name]["by_seed"][skey]
                value = r["total_seconds"]["median"]
                ratio = value / primary_seconds if primary_seconds else None
                comparison["networks"][name] = {"seconds": value, "accurate": r["accurate"],
                    "network_over_fastest_classical": ratio,
                    "passes_20_percent_gate": bool(r["accurate"] and ratio is not None and ratio <= .8),
                    "network_over_label_min_classical_sensitivity": value / best_seconds if best_seconds else None,
                    "passes_label_min_20_percent_gate_sensitivity": bool(r["accurate"] and best_seconds is not None and value <= .8 * best_seconds)}
            initial = comparison["networks"]["initial_network"]["seconds"]
            control = comparison["networks"]["budget_control_network"]["seconds"]
            optimized = comparison["networks"]["optimized_network"]["seconds"]
            comparison["optimized_over_initial"] = optimized / initial
            comparison["optimized_over_equal_budget_energy"] = optimized / control
            per_seed[skey] = comparison
        result[str(n)] = {"by_seed": per_seed,
            "gate_definition": "Canonicalize complete classical configuration excluding name; pool identical labels' complete-run timings within each geometry, take median; compare actual network full time to 0.8 times fastest accurate distinct classical configuration. Label-min comparison is sensitivity only.",
            "networks": {name: {"paired_ratio": stats(c["networks"][name]["network_over_fastest_classical"] for c in per_seed.values()),
                "geometries_passing_20_percent_gate": sum(c["networks"][name]["passes_20_percent_gate"] for c in per_seed.values()),
                "total_geometries": len(per_seed),
                "all_geometries_pass_gate": all(c["networks"][name]["passes_20_percent_gate"] for c in per_seed.values()),
                "label_min_sensitivity_paired_ratio": stats(c["networks"][name]["network_over_label_min_classical_sensitivity"] for c in per_seed.values()),
                "label_min_sensitivity_gate_count": sum(c["networks"][name]["passes_label_min_20_percent_gate_sensitivity"] for c in per_seed.values())}
                for name in FINAL_METHODS[3:]},
            "optimized_over_initial": stats(c["optimized_over_initial"] for c in per_seed.values()),
            "optimized_over_equal_budget_energy": stats(c["optimized_over_equal_budget_energy"] for c in per_seed.values())}
    return result


def validate_frozen_specs(stage, screen, frozen):
    config = stage["manifest"]["configuration"]
    for n in config["grids"]:
        actual = {s["name"]: s for s in config["specs_by_n"][str(n)]}
        base = screen["selected_by_grid"][str(n)]
        for name in FINAL_METHODS[3:]:
            expected = {**base, **{k: frozen[name][k] for k in ["correction", "checkpoint", "device", "smooth_prediction"]}, "name": name}
            if actual[name] != expected:
                raise ValueError(f"Frozen network deployment changed at grid {n}: {name}")
        expected = ({**frozen["classical"], "name": "best_classical"}
            if n == 128 or frozen["classical"]["name"] == "fresh_previous" else
            {**base, "name": "best_classical", "correction": frozen["classical"]["correction"]})
        if actual["best_classical"] != expected:
            raise ValueError(f"Frozen classical deployment changed at grid {n}")


def training_summary(root, frozen):
    data_manifest_path = root / "data/stage3_correction.manifest.json"
    data_manifest = read(data_manifest_path)
    results = {}
    for name in TRAINING:
        path = root / "checkpoints" / f"{name}.training.json"
        state = read(path)
        config, history = state["configuration"], state["history"]
        if config["epochs"] != 40 or config["seed"] != 7:
            raise ValueError(f"Unexpected registered training budget/seed: {name}")
        if len(history) != config["epochs"] or [r["epoch"] for r in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError(f"Incomplete training history: {name}")
        if config["dataset_sha256"] != data_manifest["dataset_sha256"]:
            raise ValueError(f"Training data hash mismatch: {name}")
        checkpoint = f"checkpoints/{name}.pt"
        if sha(root / checkpoint) != frozen["checkpoint_sha256"][checkpoint]:
            raise ValueError(f"Training checkpoint differs from frozen selection: {name}")
        best = min(history, key=lambda r: r["selection_loss"])
        if not math.isclose(best["selection_loss"], state["best_loss"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Best training loss mismatch: {name}")
        results[name] = {"configuration": config, "best_epoch_row": best,
            "stage_training_seconds": state["elapsed_seconds"], "parameter_count": state["parameter_count"],
            "history": history, "training_history_sha256": sha(path), "checkpoint_sha256": sha(root / checkpoint)}
    first = results[TRAINING[0]]["configuration"]
    a, b = results[TRAINING[1]]["configuration"], results[TRAINING[2]]["configuration"]
    for key in ["epochs", "initialize", "seed", "width", "batch_size", "initial_lr", "dataset_sha256"]:
        if a[key] != b[key]:
            raise ValueError(f"Energy/mixed continuation budgets differ: {key}")
    return {"models": results, "label_generation_seconds": data_manifest["generation_seconds"],
        "data_manifest_sha256": sha(data_manifest_path), "dataset_sha256": data_manifest["dataset_sha256"],
        "sample_counts": {group: len(v) for group, v in data_manifest["sample_metadata"].items()},
        "independent_training_geometries": len(data_manifest["groups"]["train"]),
        "independent_training_validation_geometries": len(data_manifest["groups"]["validation"]),
        "single_training_seed": first["seed"], "matched_continuation_budget_verified": True,
        "total_executed_training_seconds": sum(r["stage_training_seconds"] for r in results.values()),
        "history_provenance": data_manifest["configuration"]["label_source"]}


def verify_selection(stages, screen, frozen):
    for n, grid in stages["stage3_screen"]["grids"].items():
        methods = grid["methods"]
        eligible = [r for r in methods.values() if r["all_accurate"] and not r["pcg_failures"]]
        minimum = min(r["total_seconds"]["median"] for r in eligible)
        selected = methods[screen["selected_by_grid"][n]["name"]]
        if not selected["all_accurate"] or selected["pcg_failures"] or not math.isclose(selected["total_seconds"]["median"], minimum, abs_tol=1e-12):
            raise ValueError(f"Screen selection does not obey its declared rule at grid {n}")
    methods = stages["stage3_calibration_optimized"]["grids"]["128"]["methods"]
    predicates = {"classical": lambda name, r: r["spec"]["correction"] != "network",
        "initial_network": lambda name, r: name.startswith("energy_"),
        "budget_control_network": lambda name, r: name.startswith("continued_"),
        "optimized_network": lambda name, r: name.startswith("mixed")}
    for category, predicate in predicates.items():
        eligible = [r for name, r in methods.items() if predicate(name, r) and r["all_accurate"] and not r["pcg_failures"]]
        chosen = methods[frozen[category]["name"]]
        if chosen["spec"] != frozen[category] or not chosen["all_accurate"] or chosen["pcg_failures"] or not math.isclose(
                chosen["total_seconds"]["median"], min(r["total_seconds"]["median"] for r in eligible), abs_tol=1e-12):
            raise ValueError(f"Calibration selection does not obey its declared rule: {category}")
    return True


def controlled_calibration(stages):
    methods = stages["stage3_calibration_optimized"]["grids"]["128"]["methods"]
    result = {}
    for device in ["cpu", "cuda"]:
        mixed, continued = methods[f"mixed_{device}"], methods[f"continued_{device}"]
        pairs = {seed: mixed["by_seed"][seed]["total_seconds"]["median"] / r["total_seconds"]["median"]
                 for seed, r in continued["by_seed"].items()}
        result[device] = {"mixed_over_equal_budget_energy": stats(pairs.values()), "by_seed": pairs,
            "both_accurate": mixed["all_accurate"] and continued["all_accurate"],
            "scope": "Development calibration, same device and no prediction smoothing; same continuation start, budget and nominal data order. Does not include retraining-seed uncertainty."}
    return result


def verify_pretest_freeze(root, stages, frozen_hash, frozen):
    path = root / "results/stage3_test_freeze.json"
    lock = read(path)
    expected = {"protocol_sha256": sha(root / "docs/stage3_protocol_zh.md"),
        "splits_sha256": sha(root / "data/stage3_splits.json"),
        "selection_sha256": frozen_hash,
        "screen_selection_sha256": sha(root / "results/stage3_screen/selection.json"),
        "reporting_rules_sha256": sha(root / "docs/stage3_reporting_rules_zh.md")}
    for key, value in expected.items():
        if lock[key] != value:
            raise ValueError(f"Pretest freeze mismatch: {key}")
    if lock["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
        raise ValueError("Pretest frozen checkpoint set differs from model selection")
    if lock["freeze_script_sha256"] != sha(root / "freeze_stage3.py"):
        raise ValueError("Freeze script differs from recorded provenance")
    if lock["development_manifest_sha256"] != {
            name: stages[name]["provenance"]["input_sha256"]["manifest.json"] for name in RUNS[:3]}:
        raise ValueError("Development manifests differ from pretest freeze")
    for key in ["source_sha256", "checkpoint_sha256"]:
        if not lock[key] or any(not (root / f).exists() or sha(root / f) != v for f, v in lock[key].items()):
            raise ValueError(f"Current files differ from pretest freeze: {key}")
    def utc(value):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Freeze/test timestamps must specify their UTC offset")
        return stamp.astimezone(timezone.utc)
    frozen_time = utc(lock["frozen_utc"])
    for name in RUNS[3:]:
        stage = stages[name]; manifest = stage["manifest"]
        for key in ["source_sha256", "protocol_sha256", "splits_sha256"]:
            if manifest[key] != lock[key]:
                raise ValueError(f"Final run and pretest freeze differ: {name}/{key}")
        weights = read(root / "results" / name / "checkpoint_sha256.json")
        if weights != lock["checkpoint_sha256"]:
            raise ValueError(f"Final checkpoint set differs from pretest freeze: {name}")
        # Runner manifests have one-second precision; its start may share the
        # freeze second even when the freeze retains fractional seconds.
        if frozen_time.replace(microsecond=0) > utc(manifest["created_utc"]):
            raise ValueError(f"Freeze timestamp is later than test start: {name}")
    return {"verified": True, "sha256": sha(path), "manifest": lock}


def reference_audit_summary(root, stages):
    """Validate saved offline recomputation; never assemble or solve a system."""
    path = root / "results/stage3_reference_audit.json"
    audit = read(path)
    if set(audit["configuration"]["stages"]) != set(RUNS) or set(audit["stages"]) != set(RUNS):
        raise ValueError("Reference audit must cover all five completed stages")
    if not audit.get("finished_utc"):
        raise ValueError("Reference audit has not finished")
    for source in ["fields.py", "numerics.py", "audit_stage3_references.py"]:
        if audit["source_sha256"].get(source) != sha(root / source):
            raise ValueError(f"Reference audit source differs: {source}")
    expected = defaultdict(dict)
    for name, stage in stages.items():
        evidence = audit["stages"][name]
        for key in ["manifest", "references", "complete"]:
            if evidence["input_sha256"][key] != stage["provenance"]["input_sha256"][key + ".json"]:
                raise ValueError(f"Reference audit used different inputs: {name}/{key}")
        if evidence["configuration"] != stage["manifest"]["configuration"]:
            raise ValueError(f"Reference audit configuration differs: {name}")
        done = read(root / "results" / name / "complete.json")
        started = datetime.fromisoformat(audit["created_utc"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(done["finished_utc"].replace("Z", "+00:00"))
        if started < finished:
            raise ValueError(f"Reference audit started before online timing ended: {name}")
        for ref in read(root / "results" / name / "references.json"):
            expected[(ref["n"], ref["seed"])][name] = ref["field_sha256"]
    sequences = audit["sequences"]
    keys = [(row["n"], row["seed"]) for row in sequences]
    if len(set(keys)) != len(keys) or set(keys) != set(expected) or audit["unique_sequences"] != len(keys):
        raise ValueError("Reference audit has missing, duplicate or unexpected sequences")
    values, stage_rows = [], defaultdict(list)
    failures, nonfinite, exceptions, warnings_count = [], 0, 0, 0
    hashes_match = True
    for row in sequences:
        key = (row["n"], row["seed"]); original = expected[key]
        if set(row["stage_field_hash_matches"]) != set(original) or set(row["source_stages"]) != set(original):
            raise ValueError(f"Reference audit omitted a shared stage: {key}")
        checks = {name: row["field_sha256"] == digest for name, digest in original.items()}
        if checks != row["stage_field_hash_matches"] or any(
                row["source_stages"][name]["expected_field_sha256"] != value for name, value in original.items()):
            raise ValueError(f"Reference audit hash assertions are inconsistent: {key}")
        hashes_match = hashes_match and all(checks.values())
        records = row["records"]
        required_steps = set(range(20)) if all(checks.values()) else set()
        if len(records) != len(required_steps) or {r["step"] for r in records} != required_steps:
            raise ValueError(f"Reference audit solve records are incomplete: {key}")
        for record in records:
            value = record["explicit_relative_residual"]
            valid = isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            if valid:
                values.append(value)
            nonfinite += bool(record["nonfinite"])
            exceptions += "exception" in record
            warnings_count += len(record["warnings"])
            for name in original:
                passed = bool(valid and value <= stages[name]["manifest"]["rtol"])
                if record["passes_stage_tolerances"].get(name) != passed:
                    raise ValueError(f"Reference audit tolerance assertion inconsistent: {key}/{record['step']}")
                stage_rows[name].append((key, record, passed))
            if not valid or any(not record["passes_stage_tolerances"][name] for name in original):
                failures.append({"n": key[0], "seed": key[1], "step": record["step"]})
    successful = hashes_match and not failures and len(values) == len(keys) * 20
    maximum = max(values) if values else None
    if audit["successful_reference_audit"] != successful or audit["max_explicit_relative_residual"] != maximum:
        raise ValueError("Reference audit global accuracy assertions are inconsistent")
    if audit["nonfinite_step_count"] != nonfinite or audit["exception_step_count"] != exceptions:
        raise ValueError("Reference audit failure counts are inconsistent")
    by_stage = {}
    for name, evidence in audit["stages"].items():
        records = stage_rows[name]
        stage_values = [r["explicit_relative_residual"] for _, r, _ in records if r["explicit_relative_residual"] is not None]
        match = all(row["stage_field_hash_matches"][name] for row in sequences if name in row["source_stages"])
        computed = {"field_hashes_match": match, "max_explicit_relative_residual": max(stage_values) if stage_values else None,
            "attempted_direct_solves": len(records), "finite_residual_steps": len(stage_values),
            "failed_steps": [{"n": key[0], "seed": key[1], "step": r["step"]} for key, r, passed in records if not passed]}
        if any(evidence[key] != value for key, value in computed.items()):
            raise ValueError(f"Reference audit stage summary is inconsistent: {name}")
        by_stage[name] = computed
    return {"sha256": sha(path), "metadata_complete": True, "unique_sequences": len(keys),
        "unique_direct_solves": sum(len(row["records"]) for row in sequences),
        "field_hashes_match": hashes_match, "successful_reference_audit": successful,
        "max_explicit_relative_residual": maximum, "nonfinite_step_count": nonfinite,
        "exception_step_count": exceptions, "warning_count": warnings_count, "failed_steps": failures,
        "elapsed_seconds": audit["elapsed_seconds"], "created_utc": audit["created_utc"],
        "finished_utc": audit["finished_utc"], "stages": by_stage,
        "scope": "Post-hoc direct-reference recomputation; matching original coefficient bytes does not prove bitwise equality with original solutions because original solution hashes were not saved."}


def figures(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    target = out / "figures"
    target.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for ax, n in zip(axes, [128, 256]):
        methods = summary["stages"][f"stage3_test{n}"]["grids"][str(n)]["methods"]
        x = np.arange(6)
        values = [methods[m]["total_seconds"]["median"] for m in FINAL_METHODS]
        lows = [methods[m]["total_seconds"]["min"] for m in FINAL_METHODS]
        highs = [methods[m]["total_seconds"]["max"] for m in FINAL_METHODS]
        ax.bar(x, values, color=["#8394a5"] * 3 + ["#8cb2b0", "#c1a16f", "#c56f4d"])
        ax.errorbar(x, values, yerr=[np.array(values) - lows, np.array(highs) - values], fmt="none", ecolor="black", capsize=3)
        for i, method in enumerate(FINAL_METHODS):
            ax.scatter([i] * len(methods[method]["by_seed"]),
                [v["total_seconds"]["median"] for v in methods[method]["by_seed"].values()], s=12, color="black", zorder=3)
        ax.set_xticks(x, ["Fresh", "Screened", "Calibrated", "Initial NN", "Energy +40", "Mixed +40"], rotation=32, ha="right")
        ax.set_ylabel("Full online seconds per 20-step sequence")
        ax.set_title(f"Grid {n}: {len(methods[FINAL_METHODS[0]]['by_seed'])} geometries")
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("Held-out geometry timing; ranges are observed, not confidence intervals")
    for ext in ["png", "pdf"]:
        fig.savefig(target / f"test_timing.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for name, model in summary["training"]["models"].items():
        history = model["history"]
        for ax, key in zip(axes, ["validation_energy", "validation_residual"]):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=name.replace("correction_", ""))
            ax.set_yscale("log"); ax.set_xlabel("Epoch within this stage")
            ax.set_ylabel(key.replace("_", " ")); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Training validation objectives; continued models share the initial checkpoint")
    for ext in ["png", "pdf"]:
        fig.savefig(target / f"training.{ext}", dpi=180)
    plt.close(fig)


def f(value, digits=3):
    return "不适用" if value is None else f"{value:.{digits}f}"


def scientific(value):
    return "无有限值" if value is None or not math.isfinite(value) else f"{value:.3e}"


def text_report(summary):
    training = summary["training"]
    audit = summary["reference_audit"]
    audit_maximum = scientific(audit["max_explicit_relative_residual"])
    lines = ["# 第三阶段报告：残差条件热启动与强经典序列基线", "",
        "本报告由保存的完整运行生成，未重新执行求解器或挑选模型。问题仍为二维 Darcy/P1、对比度 1e4、恒定右端、每条序列 20 步，最终显式相对真残差要求为 1e-8。网络只改变 PCG 初值，每次 PCG 使用固定的经典 AMG 预条件器。", "",
        "最终 128 网格测试包含 12 个未用于选择的物理几何；256 网格使用其中前三个，是分辨率迁移检查，不是新增独立几何。每个标签三次计时重复，共 5400 次求解测量（128 网格 4320 次，256 网格 1080 次），独立物理几何仍为 12 个。先求完整序列总时间，再对同一几何的重复取中位数，最后汇总几何中位数和范围。范围不是置信区间。", "",
        "## 最终完整在线时间", ""]
    for n in [128, 256]:
        stage = summary["stages"][f"stage3_test{n}"]
        grid = stage["grids"][str(n)]
        lines += [f"### 网格 {n}，{grid['independent_geometries']} 个几何", "",
            "| 方法标签 | 时间中位数 / s | 几何范围 / s | 序列迭代数中位数 | 最大最终真残差 | 最大相对能量误差 |", "|---|---:|---:|---:|---:|---:|"]
        for name in FINAL_METHODS:
            r = grid["methods"][name]; t = r["total_seconds"]
            lines.append(f"| {name} | {f(t['median'])} | {f(t['min'])}–{f(t['max'])} | {f(r['iterations_per_sequence']['median'],1)} | {scientific(r['final_relative_residual']['max'])} | {scientific(r['energy_relative_error']['max'])} |")
        aliases = grid["identical_configuration_labels"]
        if aliases:
            lines += ["", "相同算法的重复标签：" + "；".join("、".join(x) for x in aliases) + "。保留各标签原始测量以复现运行，但它们不是额外独立算法或物理样本。"]
        lines += [""]
    lines += ["![最终测试耗时](figures/test_timing.png)", "",
        "在线成本包含组装、历史维护、层次构造或刷新、特征准备、传输、推理、校正、保护、PCG、真实残差验证及失败恢复。`correction_seconds` 已包含推理和保护子项；这些子项没有重复加入总时间。模型加载和按实际尺寸预热在在线计时外，且没有单独保存耗时；结果采用模型常驻进程口径，不代表进程冷启动后的完整部署时间。", "",
        "## 配对比较与 20% 工程筛选线", "",
        "先按完整算法配置（忽略标签名称）合并完全相同的经典方法；在每个几何内汇集这些别名的完整运行时间取中位数，再从不同且所有重复最终精度合格的经典配置中取最快者。网络完整时间除以该时间，小于 1 表示本次测量中更快，不超过 0.8 为预定 20% 工程门槛。即使经典方法使用了恢复也保留其完整成本与失败记录；未达共同最终精度的经典配置不作为有效基线。这个门槛不是论文发表标准。", "",
        "| 网格 | 网络 | 网络/最快经典的配对中位比 | 几何比值范围 | 过20%线的几何数 |", "|---|---|---:|---:|---:|"]
    for n, comp in summary["test_comparisons"].items():
        for name, r in comp["networks"].items():
            ratio = r["paired_ratio"]
            lines.append(f"| {n} | {name} | {f(ratio['median'])} | {f(ratio['min'])}–{f(ratio['max'])} | {r['geometries_passing_20_percent_gate']}/{r['total_geometries']} |")
    lines += ["", "上述主口径按测试前制定的 `docs/stage3_reporting_rules_zh.md` 合并相同经典配置，只增加同一算法的计时观测，不增加独立几何。`summary.json` 另外保留按原标签最小中位数选基线的敏感性比较；后者更易受相同算法重复计时的噪声影响，不与主口径混用。", "",
        "| 网格 | 优化/初始网络配对比中位数 | 优化/等预算纯能量续训配对比中位数 |", "|---|---:|---:|"]
    for n, comp in summary["test_comparisons"].items():
        lines.append(f"| {n} | {f(comp['optimized_over_initial']['median'])} | {f(comp['optimized_over_equal_budget_energy']['median'])} |")
    lines += ["", "优化模型相对初始模型增加了训练预算；等预算纯能量续训用于区分这一因素。部署设备及是否进行 Jacobi 平滑也经过校准选择，因此最终优化与等预算对照的完整时间差是所选部署方案的联合差异；不能自动全部归因于损失函数。若选中 mixed_smooth3，即便优于 continued，也只能归因于混合损失与平滑后处理组合。单独考察损失差异只使用优化校准中同设备、无平滑的 mixed/continued 配对结果。", "",
        "## 最终部署配置", "", "| 网格 | 标签 | AMG策略 | 初值 | 校正 | 设备 | 预测平滑次数 |", "|---|---|---|---|---|---|---:|"]
    for n in [128, 256]:
        methods = summary["stages"][f"stage3_test{n}"]["grids"][str(n)]["methods"]
        for name in FINAL_METHODS:
            s = methods[name]["spec"]
            lines.append(f"| {n} | {name} | {s['mode']} | {s['guess']} | {s['correction']} | {s.get('device','CPU')} | {s.get('smooth_prediction',0)} |")
    lines += ["", "`reuse_first` 固定使用首步层次；`refresh_galerkin` 固定传递算子但使用当前矩阵逐层重算 RAP，并重建平滑器和粗解缓存；`fresh` 每步完整重建。历史 Galerkin 初值使用最近四个已计算的解，不能称为完整复现某个回收 Krylov 算法。", "",
        "## 推理开销、残差与失败", "",
        "以下计数覆盖实际计时运行，包含重复，不是独立样本数。残差比例先对同一几何/时刻的重复取中位，再作为描述性事件统计；同一几何内事件相关。", "",
        "| 网格 | 网络 | 实际调用数 | 唯一几何/时刻数 | 每次推理中位 ms | 校正后/前残差中位比 | 残差下降案例数 |", "|---|---|---:|---:|---:|---:|---:|"]
    for n in [128, 256]:
        methods = summary["stages"][f"stage3_test{n}"]["grids"][str(n)]["methods"]
        for name in FINAL_METHODS[3:]:
            r = methods[name]; timing = r["network_inference_seconds_per_call"]["median"]
            lines.append(f"| {n} | {name} | {r['network_calls']} | {r['network_unique_seed_step_cases']} | {f(None if timing is None else timing*1000)} | {f(r['corrected_over_base_residual_by_unique_case']['median'])} | {r['residual_reduced_unique_cases']} |")
    lines += ["", "能量线搜索只保护离散二次能量；欧氏残差、迭代数和总时间仍可能增加。混合物理损失下降也不保证在线求解更快。零步长调用数及能量变化保留在汇总中，零步长不能自动解释为某种特定拒绝原因。", "",
        "| 阶段 | 网格 | 方法 | PCG失败 | 直接恢复 | 最终未达精度 | 非有限最终残差数 |", "|---|---|---|---:|---:|---:|---:|"]
    for name, stage in summary["stages"].items():
        for n, grid in stage["grids"].items():
            for method, r in grid["methods"].items():
                nr = r["final_relative_residual"]
                lines.append(f"| {name} | {n} | {method} | {r['pcg_failures']} | {r['solver_fallbacks']} | {r['final_failures']} | {nr['count']-nr['finite_count']} |")
    lines += ["", "失败不会从总时间或分母中删除。直接恢复后的准确解不意味着 PCG 原本成功。能量误差相对离线离散参考解，不代表连续 PDE 的离散误差。", "",
        "## 训练与离线成本", "",
        f"训练几何 {training['independent_training_geometries']} 个，训练验证几何 {training['independent_training_validation_geometries']} 个；全局校正事件样本数 {training['sample_counts']}。标签生成 {training['label_generation_seconds']:.3f} 秒；三段训练所记录的训练/验证循环及保存计时合计 {training['total_executed_training_seconds']:.3f} 秒。计时起点晚于数据加载、数据传入 GPU、模型和优化器初始化，因此不包含这些阶段或训练进程启动成本。训练种子仅 {training['single_training_seed']}，不能据此声称跨初始化鲁棒性。", "",
        "| 模型 | 本段轮数 | 最佳轮数 | 验证相对能量平方损失 | 验证加权残差损失 | 本段循环及保存秒 |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in training["models"].items():
        best = r["best_epoch_row"]
        lines.append(f"| {name} | {r['configuration']['epochs']} | {best['epoch']} | {best['validation_energy']:.6f} | {best['validation_residual']:.6f} | {r['stage_training_seconds']:.3f} |")
    lines += ["", "![训练曲线](figures/training.png)", "",
        "初始能量模型训练 40 轮；两个续训分支从同一初始检查点出发，各增加 40 轮，种子、批大小、学习率及数据哈希已核对一致。等预算指相同更新轮数和名义批次顺序，不是相同训练秒数。两续训分支各按自身验证目标选检查点，因此是训练流程对照，目标值不能跨不同损失直接排序。", "",
        "训练标签使用旧离线直接解近似已收敛历史，在线使用实际 PCG 历史；真残差达到 1e-8 不能推出状态或特征差异也不超过 1e-8。训练 hard_change 样本可能包含线上因初值残差已达容差而跳过的项，这些训练与部署分布差异仍需后续量化。", "",
        "## 事后离线直接参考解审计", "",
        f"在线计时全部结束后，另行重算 {audit['unique_sequences']} 条唯一网格/几何序列、{audit['unique_direct_solves']} 次直接求解，额外离线成本 {audit['elapsed_seconds']:.3f} 秒，最大显式相对残差 {audit_maximum}。输入系数序列哈希全部匹配：{'是' if audit['field_hashes_match'] else '否'}；重算参考解全部通过相应残差要求：{'是' if audit['successful_reference_audit'] else '否'}。非有限步骤 {audit['nonfinite_step_count']}，异常步骤 {audit['exception_step_count']}，求解器警告 {audit['warning_count']}。", "",
        "| 原运行阶段 | 重算次数 | 最大显式相对残差 | 系数哈希全部匹配 | 未达精度步骤 |", "|---|---:|---:|---|---:|"]
    for name, evidence in audit["stages"].items():
        value = evidence["max_explicit_relative_residual"]
        maximum = scientific(value)
        lines.append(f"| {name} | {evidence['attempted_direct_solves']} | {maximum} | {'是' if evidence['field_hashes_match'] else '否'} | {len(evidence['failed_steps'])} |")
    lines += ["", "不同阶段可能共享同一序列，因此表中重算次数不能相加作为独立求解数。原运行只存系数序列哈希，未保存原参考解的逐比特哈希；本审计证明输入一致并检验事后重算解的残差，不能证明重算解与原参考解逐比特相同。报告中的能量误差仍是相对当时计算的离散直接参考解，不是精确算术解或连续 PDE 真解的误差证书。", "",
        "## 开发筛选与校准", "",
        "开发筛选与两次校准不属于最终测试。以下展示完整时间中位数与精度标记，所有逐几何结果、失败计数和实际选择在 `summary.json` 可查。", "",
        "| 阶段 | 网格 | 方法 | 序列时间中位数 / s | 全部最终精度合格 |", "|---|---|---|---:|---|"]
    for name in RUNS[:3]:
        for n, grid in summary["stages"][name]["grids"].items():
            for method, r in grid["methods"].items():
                lines.append(f"| {name} | {n} | {method} | {f(r['total_seconds']['median'])} | {'是' if r['all_accurate'] else '否'} |")
    lines += ["", "相同续训预算、相同部署设备且无预测平滑的开发校准比较：", "",
        "| 设备 | mixed/纯能量续训的完整时间配对中位比 | 几何范围 |", "|---|---:|---:|"]
    for device, r in summary["controlled_calibration_comparisons"].items():
        p = r["mixed_over_equal_budget_energy"]
        lines.append(f"| {device} | {f(p['median'])} | {f(p['min'])}–{f(p['max'])} |")
    lines += ["", "该对照缩小了额外训练、设备和平滑因素的混淆，但仍只是一个训练种子和开发校准几何上的结果。", "",
        "筛选分别确定各网格 AMG/历史策略；128 网格校准锁定经典校正、模型、设备及预测平滑选择，随后按固定规则迁移至 256。测试结果没有用于再次选配置。报告分别保存已有训练循环、数据生成、开发校准和参考解计时；未覆盖所有离线成本，也未测量完整进程启动成本，不能据此声称完整研究或部署投入已经摊销。", "",
        "## 可追溯性与研究结论边界", "",
        "各阶段行数由清单中的网格、方法、种子、重复和 20 步推导，逐键检查完整性与唯一性；所有原始文件只读。最终源码、协议、数据划分、权重、冻结选择及报告聚合规则哈希已与测试前冻结清单核验，冻结时间不晚于两个最终运行的创建时间。最终测试共有 12 个独立几何，不能将 256 子集、多个方法标签或重复计时累加为更多独立问题。", "",
        "已有文献包含神经热启动、残差条件校正、柔性神经预条件和经典序列回收。本轮即使通过工程门槛，也不自动建立新颖性、一般化加速或期刊录用结论；若没有通过，应保留负结果而非在同一测试集上继续选模。", ""]
    if summary["warnings"]:
        lines += ["本次审计提醒：", ""] + [f"- {w}" for w in summary["warnings"]] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(); root = args.root.resolve()
    split_path = root / "data/stage3_splits.json"
    splits = read(split_path)
    all_seeds = [s for group in splits["groups"].values() for s in group]
    if len(set(all_seeds)) != len(all_seeds) or set(all_seeds) & set(splits["excluded_prior_seeds"]):
        raise ValueError("Stage-3 seed groups overlap with one another or prior data")
    hashes = list(splits["geometry_sha256"].values())
    if len(hashes) != len(set(hashes)):
        raise ValueError("Stage-3 geometry hashes overlap")
    frozen_path = root / "results/stage3_calibration_optimized/frozen_selection.json"
    frozen, frozen_hash = read(frozen_path), sha(frozen_path)
    final_path = root / "results/stage3_final_complete.json"
    final = read(final_path)
    if final["selection_sha256"] != frozen_hash or final["test128"] != "results/stage3_test128" or final["test256"] != "results/stage3_test256":
        raise ValueError("Final completion marker does not match the frozen test runs")
    summary = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "report_source_sha256": sha(__file__), "frozen_selection": frozen,
        "frozen_selection_sha256": frozen_hash, "final_marker_sha256": sha(final_path),
        "splits_sha256": sha(split_path),
        "reporting_rules_sha256": sha(root / "docs/stage3_reporting_rules_zh.md"),
        "independent_final_geometries": 12,
        "grid256_is_subset_not_new_geometries": True, "stages": {}, "test_comparisons": {}, "warnings": []}
    screen = read(root / "results/stage3_screen/selection.json")
    for name in RUNS:
        manifest, rows, provenance = check_stage(root, name, splits, frozen_hash)
        units, grids = aggregate(manifest, rows)
        stage = {"manifest": manifest, "provenance": provenance, "sequence_units": units, "grids": grids}
        summary["stages"][name] = stage
        if not provenance["all_current_artifacts_match"]:
            summary["warnings"].append(f"{name} 开发阶段的部分源码/协议/权重与当前版本不同，原始哈希保留；不能将不同阶段时间当作固定实现重复。")
        if any(u["other_seconds"] < -1e-6 for u in units):
            summary["warnings"].append(f"{name} 存在时间分解超过总时间的记录，须检查重叠计时。")
        if name.startswith("stage3_test"):
            validate_frozen_specs(stage, screen, frozen)
            summary["test_comparisons"].update(test_comparisons(manifest, rows, units, grids))
    if len(splits["groups"]["test"]) != 12:
        raise ValueError("The preregistered final cohort must contain 12 geometries")
    summary["training"] = training_summary(root, frozen)
    summary["selection_rule_verified"] = verify_selection(summary["stages"], screen, frozen)
    summary["controlled_calibration_comparisons"] = controlled_calibration(summary["stages"])
    summary["pretest_freeze"] = verify_pretest_freeze(root, summary["stages"], frozen_hash, frozen)
    summary["reference_audit"] = reference_audit_summary(root, summary["stages"])
    if not summary["reference_audit"]["successful_reference_audit"]:
        summary["warnings"].append("事后直接参考解审计存在输入哈希不匹配或残差失败，必须保留这些结果；相对直接参考解的能量误差不能当作已核验的精确解误差。")
    summary["complete"] = True
    out = root / "results/stage3_report"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not args.no_figures:
        figures(out, summary)
    (out / "REPORT_zh.md").write_text(text_report(summary))
    print(json.dumps({"output": str(out), "complete": True,
        "optimized_gate": {n: c["networks"]["optimized_network"] for n, c in summary["test_comparisons"].items()},
        "warnings": summary["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
