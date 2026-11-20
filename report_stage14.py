#!/usr/bin/env python3
"""Audit saved Stage 14 outcomes without importing or running a PDE solver.

The finite-bank oracle is an offline existence probe. Its selection does use
fine-reference errors; no trained policy or deployment speed is inferred.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
MODES = ("locked_ring", "coverage")
SPLITS = ("id", "contrast")
BANK_NAMES = ("coefficient", "seed_coefficient", "source_weighted",
              "goal_weighted", "source_goal_weighted", "seed_uniform",
              *tuple(f"random_{seed}_beta_{beta}" for seed in range(4) for beta in (0.5, 1.5)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value}: {path}")
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream, parse_constant=reject)
    return json.loads(path.read_text(), parse_constant=reject)


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


def stat(values):
    values = list(values)
    valid = [float(v) for v in values if v is not None]
    return dict(median=median(valid) if valid else None,
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
                valid_count=len(valid), missing_count=len(values) - len(valid))


def added_dofs(cells, nc):
    chosen = set(cells)
    return len(chosen) + sum((c % nc + 1 < nc and c + 1 in chosen) + (c + nc in chosen)
                             for c in chosen)


def neighbors(cell, nc):
    y, x = divmod(cell, nc)
    return [n for valid, n in ((x > 0, cell - 1), (x + 1 < nc, cell + 1),
                              (y > 0, cell - nc), (y + 1 < nc, cell + nc)) if valid]


def regenerate_inputs(geometry, nc, contrast):
    """Pure NumPy reconstruction from frozen generation rules; no PDE imports."""
    rng = np.random.default_rng(geometry)
    nf = 2 * nc; yy, xx = np.mgrid[:nf, :nf]; x, y = (xx + .5) / nf, (yy + .5) / nf
    center = rng.uniform(.25, .75, 2); radius = rng.uniform(.12, .25)
    origin = rng.uniform(.15, .6, 2); sizes = rng.uniform(.15, .3, 2)
    jump = float(rng.choice([100., 1000.] if contrast else [5., 20.]))
    high = (x - center[0])**2 + (y - center[1])**2 < radius**2
    high |= (abs(x - origin[0]) < sizes[0]/2) & (abs(y - origin[1]) < sizes[1]/2)
    k = np.exp(.25*np.sin(2*np.pi*x+rng.uniform(0, 6))*np.cos(2*np.pi*y+rng.uniform(0, 6))) * np.where(high, jump, 1.)
    cells = k.reshape(nc, 2, nc, 2).transpose(0, 2, 1, 3)
    counts = (cells > 2).sum(axis=(2, 3))
    interface = set(np.flatnonzero((counts > 0) & (counts < 4)).tolist())
    logs = np.log(cells); means = logs.mean(axis=(2, 3)); base = logs.std(axis=(2, 3))
    dx, dy = abs(means[:, 1:] - means[:, :-1]), abs(means[1:] - means[:-1])
    base[:, 1:] += dx; base[:, :-1] += dx; base[1:] += dy; base[:-1] += dy
    base = base.ravel()
    parameters = {}
    for name, seed in (("known_four", geometry), ("audit_four_generator", geometry+2000000)):
        gen = np.random.default_rng(seed + 700000)
        parameters[name] = dict(centers=gen.uniform(.2, .8, (4, 2)).tolist(),
                                widths=gen.uniform(.08, .18, 4).tolist())
    yy, xx = np.mgrid[:nc, :nc]; x, y = (xx+.5)/nc, (yy+.5)/nc
    p = parameters["known_four"]
    fields = np.stack([np.exp(-((x-c[0])**2+(y-c[1])**2)/(2*w*w))
                       for c, w in zip(p["centers"], p["widths"])]).reshape(4, -1)
    fields /= np.maximum(abs(fields).max(axis=1, keepdims=True), 1e-300)
    source, goals = fields[0], fields[1:].mean(axis=0)
    scores = [base, base, base*(.25+source), base*(.25+goals),
              base*(.25+.5*(source+goals)), np.zeros(nc*nc)]
    basis = np.stack((np.sin(2*np.pi*x), np.cos(2*np.pi*y),
                      np.sin(2*np.pi*(x+y)), np.cos(4*np.pi*(x-y))))
    for seed in range(4):
        weights = np.random.default_rng(55600+17*geometry+seed).normal(size=4)
        field = np.einsum("i,ijk->jk", weights, basis)
        field = np.clip((field-field.mean())/max(field.std(), 1e-300), -3, 3).ravel()
        scores.extend(base*np.exp(beta*field) for beta in (.5, 1.5))
    return dict(coefficient_sha256=hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest(),
                interface=interface, scores=scores, load_parameters=parameters)


def decode(scores, nc, budget, mandatory=(), forbidden=()):
    """Independent exact-budget implementation, including seeded fallback."""
    order = sorted(range(nc * nc), key=lambda c: (-scores[c], c))
    seed, blocked = set(mandatory), set(forbidden)
    chosen = set(seed); initial = spent = added_dofs(seed, nc)
    require(not seed & blocked and initial <= budget, "Impossible seed or blocked seed")
    for cell in order:
        if spent == budget:
            break
        if cell in chosen or cell in blocked:
            continue
        cost = 1 + sum(n in chosen for n in neighbors(cell, nc))
        if spent + cost <= budget:
            chosen.add(cell); spent += cost
    fallback = spent != budget
    if fallback:
        unsafe = seed | blocked | {n for c in seed for n in neighbors(c, nc)}
        options = []
        for parity in (0, 1):
            cells = [c for c in order if c not in unsafe and sum(divmod(c, nc)) % 2 == parity]
            if len(cells) >= budget - initial:
                cells = cells[:budget - initial]
                options.append((sum(scores[c] for c in cells), -parity, cells))
        require(options, "Seeded checkerboard cannot complete budget")
        _, _, cells = max(options, key=lambda v: (v[0], v[1]))
        chosen = seed | set(cells)
    return sorted(chosen), fallback


def metrics(saved, reference, label):
    q = np.asarray(saved["qoi"], dtype=float)
    fine_q = np.asarray(reference["fine_qoi"], dtype=float)
    coarse_q = np.asarray(reference["coarse_qoi"], dtype=float)
    require(q.shape == fine_q.shape == coarse_q.shape == (3,), f"Wrong QoI shape: {label}")
    scales = np.maximum(abs(coarse_q), max(float(abs(coarse_q).max()) * 1e-12, 1e-30))
    signed = (fine_q - q) / scales
    denominator = float(np.linalg.norm((fine_q - coarse_q) / scales))
    norm = float(np.linalg.norm(signed))
    expected = dict(fine_qoi=fine_q.tolist(), coarse_qoi=coarse_q.tolist(), scales=scales.tolist(),
                    signed_scaled_error=signed.tolist(), coarse_scaled_norm=denominator,
                    scaled_norm=norm, maximum_scaled_error=float(abs(signed).max()),
                    E=None if denominator < 1e-12 else norm / denominator)
    for name, value in expected.items():
        if value is None:
            require(saved[name] is None, f"Degenerate normalization not null: {label}")
        else:
            close(saved[name], value, f"{label}.{name}")
    if "scales" in reference:
        close(reference["scales"], scales, f"{label}.reference_scales")
    return expected


def certified(solver):
    checks = solver["per_rhs"]
    acceptance = solver["acceptance"]
    require(acceptance["relative_residual_limit"] == 1e-9, "Unexpected common residual rule")
    require(acceptance["componentwise_backward_limit"] == 1e-12, "Unexpected backward error rule")
    valid = all(c["relative_residual"] <= 1e-9 and c["componentwise_backward_error"] <= 1e-12
                for c in checks) and all(info == 0 for rhs in solver["solver_info"] for info in rhs)
    require(solver["converged"] == valid, "Solver certification flag differs from saved diagnostics")
    require(solver["rhs_count"] == len(checks), "Saved RHS count differs")
    for name in ("relative_residual", "componentwise_backward_error", "normwise_backward_error"):
        close(solver[name], max(c[name] for c in checks), f"Maximum {name}")
    return valid


def audit(source):
    frozen = read(source / "freeze.json"); complete = read(source / "complete.json")
    require(complete["complete"], "Incomplete experiment")
    require(sha(source / "freeze.json") == complete["freeze_sha256"], "Frozen manifest changed")
    for name, digest in complete["output_sha256"].items():
        require(sha(source / name) == digest, f"Output hash mismatch: {name}")
    inputs = frozen.get("input_sha256", frozen.get("source_sha256", {}))
    require(inputs, "Missing input hashes")
    for name, digest in inputs.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Nonportable input path {name}")
        require(sha(ROOT / path) == digest, f"Frozen input changed: {name}")
    row_name = "rows.json.gz" if "rows.json.gz" in complete["output_sha256"] else "rows.json"
    contexts = read(source / "contexts.json"); rows = read(source / row_name)
    require(finite(contexts) and finite(rows), "Nonfinite saved data")
    lookup = {(c["split"], c["geometry"]): c for c in contexts}
    require(len(lookup) == len(contexts), "Repeated context identity")
    smoke = bool(frozen.get("smoke", False) or frozen.get("development", False))
    config = frozen["config"]
    require(frozen["no_training"] and complete["no_training"], "Unexpected learning run")
    require(frozen["bank_names"] == list(BANK_NAMES), "Bank catalogue changed")
    require(frozen["baseline_methods"] == ["fine_direct", "fine_amg", "coefficient_local"], "Baseline catalogue changed")
    require(set(complete["output_sha256"]) == {row_name, "contexts.json"}, "Missing raw outputs")
    require(all(p["num_threads"] == 1 for p in frozen["environment"]["threadpools"]), "Multithreaded experiment")
    require(all(v == "1" for v in frozen["environment"]["thread_environment"].values()), "Thread environment changed")
    require(frozen["environment"]["longdouble_bits"] > 53, "Residual arithmetic is not extended precision")
    expected_config = dict(primary_mode="locked_ring", modes=["locked_ring", "coverage"],
        oracle_ratio_limit=.8, minimum_wins=12, improvement_ratio_limit=.9,
        robust_ratio_limit=1.05, near_zero=1e-12, fine_qoi_limit=1e-5,
        energy_limit=1+1e-8, matching_qoi_limit=1e-8,
        warmup_geometry=55950, order_seed=55991, audit_rhs_seed_offset=2000000)
    for name, value in expected_config.items():
        require(config[name] == value, f"Frozen decision configuration changed: {name}")
    if not smoke:
        require(set(lookup) == {(s, g) for s, base in (("id", 55200), ("contrast", 55300))
                                 for g in range(base, base + 16)}, "Confirmation geometry identities differ")
        require(config["nc"] == 64 and config["budget"] == 1024 and config["repetitions"] == 3,
                "Confirmation size or repetitions changed")
    require(len(contexts) == 2 * config["geometries_per_split"], "Context count differs from frozen configuration")
    hashes = [c["coefficient_sha256"] for c in contexts]
    require(len(set(hashes)) == len(hashes), "Repeated independent coefficient field")
    for name in inputs:
        if name.endswith("contexts.json") or name == "results/stage14_solver_development/rows.json":
            prior = read(ROOT / name)
            if isinstance(prior, list):
                historical = {r["coefficient_sha256"] for r in prior if "coefficient_sha256" in r}
                require(not historical & set(hashes), f"Geometry overlaps with {name}")
    by_context = {key: {mode: [] for mode in MODES} for key in lookup}
    regenerated = {}; load_descriptors = []
    failures = []; numerical_failures = []; decoder_checks = 0; baseline_checks = 0
    for key, c in lookup.items():
        nc, budget = c["nc"], c["budget"]
        require(nc == config["nc"] and budget == config["budget"], "Unexpected mesh or DOF budget")
        regenerated[key] = regen = regenerate_inputs(key[1], nc, key[0] == "contrast")
        require(regen["coefficient_sha256"] == c["coefficient_sha256"], "Independent coefficient generation differs")
        interface, ring, base = map(set, (c["interface_cells"], c["ring_cells"], c["baseline_cells"]))
        require(interface == regen["interface"], "Independently regenerated interface mask differs")
        load_descriptors.append(dict(split=key[0], geometry=key[1], **regen["load_parameters"],
            audit_goal_indices=[1, 2, 3], known_goals_indices=[1, 2, 3], source_index=0))
        require(ring == interface | {n for cell in interface for n in neighbors(cell, nc)}, "Wrong pinned ring")
        require(added_dofs(base, nc) == budget, "Wrong baseline DOF budget")
        if not interface <= base:
            failures.append(dict(split=key[0], geometry=key[1], reason="baseline_missing_interface"))
        if not certified(c["coarse_solver"]):
            numerical_failures.append(dict(split=key[0], geometry=key[1], reference="coarse"))
        if c["coarse_energy"] <= 0:
            numerical_failures.append(dict(split=key[0], geometry=key[1], reference="degenerate_coarse_energy"))
        require(c["audit_rhs_seed"] == key[1] + 2000000, "Audit load identity changed")
        seen = set(); repeated = {}
        order_rng = np.random.default_rng(55991 + key[1] + nc)
        expected_orders = [[frozen["baseline_methods"][i] for i in order_rng.permutation(3)]
                           for _ in range(config["repetitions"])]
        require(c["method_orders"] == expected_orders, "Randomized timing order changed")
        require([(r["method"], r["repetition"]) for r in c["baseline_runs"]] ==
                [(m, repetition) for repetition, order in enumerate(expected_orders) for m in order],
                "Saved baseline run order differs")
        for run in c["baseline_runs"]:
            run_key = run["method"], run["repetition"]
            require(run_key not in seen, "Repeated timed baseline run"); seen.add(run_key)
            m = metrics(run["known"], c["references"]["known"], f"baseline {key} {run_key}")
            valid = certified(run["solver"]) and m["E"] is not None
            if run["method"].startswith("fine_"):
                valid = valid and m["E"] <= 1e-5
            if run["method"] == "coefficient_local":
                require(run["selected_cells"] == c["baseline_cells"], "Timed coefficient mesh differs")
            close(run["qoi"], run["known"]["qoi"], "Timed QoI differs from its saved metric")
            repeated.setdefault(run["method"], []).append(run)
            require(run["full_seconds"] > 0, "Invalid complete cost")
            require(run["numerical_valid"] == valid, "Baseline validity flag differs")
            if not valid:
                numerical_failures.append(dict(split=key[0], geometry=key[1], baseline=run["method"],
                                               repetition=run["repetition"]))
            baseline_checks += 1
        deterministic = all(v["selected_cells"] == runs[0]["selected_cells"] and
            np.allclose(v["qoi"], runs[0]["qoi"], rtol=1e-12, atol=1e-15)
            for runs in repeated.values() for v in runs[1:])
        require(c["repeat_deterministic"] == deterministic, "Repeated run consistency flag differs")
        if not deterministic:
            failures.append(dict(split=key[0], geometry=key[1], reason="nondeterministic_repeat"))
    seen_rows = set()
    for row in rows:
        key = row["split"], row["geometry"]
        require(key in lookup and row["mode"] in MODES, "Unknown bank context or mode")
        c = lookup[key]; nc, budget = c["nc"], c["budget"]
        rid = (*key, row["mode"], row["index"])
        require(rid not in seen_rows, "Repeated bank slot"); seen_rows.add(rid)
        require(row["name"] == BANK_NAMES[row["index"]], "Bank slot name changed")
        close(row["scores"], regenerated[key]["scores"][row["index"]], "Independent candidate score formula", rtol=1e-13, atol=1e-15)
        cells = row["selected_cells"]; chosen = set(cells)
        require(cells == sorted(chosen) and all(0 <= i < nc * nc for i in cells), "Malformed cell set")
        require(row["selection_sha256"] == hashlib.sha256(np.asarray(cells, dtype=np.int64).tobytes()).hexdigest(),
                "Selection fingerprint differs")
        require(row["offline_only"] and row["seconds"] > 0, "Candidate timing boundary changed")
        require(added_dofs(cells, nc) == budget, "Candidate DOF budget differs")
        require(row["marking"]["added_dofs"] == budget, "Saved budget differs")
        interface, ring, base = map(set, (c["interface_cells"], c["ring_cells"], c["baseline_cells"]))
        covered, matched = interface <= chosen, (chosen & ring) == (base & ring)
        require(row["interface_covered"] == covered, "Coverage flag differs")
        require(row["ring_state_matched"] == matched, "Pinned ring flag differs")
        if not covered or (row["mode"] == "locked_ring" and not matched):
            failures.append(dict(split=key[0], geometry=key[1], mode=row["mode"], index=row["index"],
                                 reason="interface_or_ring_contract"))
        if row["index"] == 0:
            require(chosen == base, "Original coefficient baseline changed across banks")
            mandatory, forbidden = (), ()
        elif row["mode"] == "locked_ring":
            mandatory, forbidden = base & ring, ring - base
        else:
            mandatory, forbidden = interface, ()
        decoded, fallback = decode(row["scores"], nc, budget, mandatory, forbidden)
        require(decoded == cells and fallback == row["marking"]["fallback"], "Independent decoder differs")
        decoder_checks += 1
        for goal in ("known", "audit"):
            metrics(row[goal], c["references"][goal], f"{rid} {goal}")
        match = None
        if row["index"] == 0:
            base_q = next(r["qoi"] for r in c["baseline_runs"] if r["method"] == "coefficient_local")
            match = float(np.linalg.norm((np.asarray(row["known"]["qoi"])-base_q) / row["known"]["scales"]))
            close(row["baseline_scaled_qoi_difference"], match, "Offline/timed coefficient QoI check")
        else:
            require(row["baseline_scaled_qoi_difference"] is None, "Unexpected baseline-match diagnostic")
        valid = bool(certified(row["solver"]) and certified(c["coarse_solver"]) and covered and
            (row["mode"] != "locked_ring" or matched) and row["solve_dofs"] == (nc - 1)**2 + budget and
            row["known"]["E"] is not None and row["audit"]["E"] is not None and
            row["energy_error_ratio"] is not None and -1e-12 <= row["energy_error_ratio"] <= 1+1e-8 and
            (match is None or match <= 1e-8))
        require(row["numerical_valid"] == valid, "Candidate validity flag differs")
        if not valid:
            numerical_failures.append(dict(split=key[0], geometry=key[1], mode=row["mode"], index=row["index"]))
        by_context[key][row["mode"]].append(row)
    for key, modes in by_context.items():
        for mode, bank in modes.items():
            bank.sort(key=lambda v: v["index"])
            require([r["index"] for r in bank] == list(range(14)), f"Incomplete bank {key} {mode}")
    require(complete["contexts"] == len(contexts) and complete["rows"] == len(rows), "Completion counts differ")
    require(complete["baseline_pipeline_runs"] == baseline_checks, "Timed run count differs")
    require(complete["offline_candidate_solves"] == len(rows), "Offline solve count differs")
    return frozen, complete, contexts, rows, by_context, dict(
        passed=not failures and not numerical_failures, frozen_input_count=len(inputs),
        independent_geometries=len(contexts), candidate_rows=len(rows),
        decoded_rows=decoder_checks, baseline_run_checks=baseline_checks,
        structural_failures=failures, numerical_failures=numerical_failures,
        regenerated_coefficient_count=len(regenerated), regenerated_score_rows=len(rows),
        regenerated_function_parameters=load_descriptors,
        smoke=smoke, limitations=["No PDE solver imported or executed by the reporter.",
            "Saved fields do not contain full solution vectors: residuals, backward errors and energies cannot be recomputed here.",
            "Gaussian function parameters are regenerated; load quadrature and persisted load-array hashes are not recomputed."])


def ratio(a, b):
    return None if a is None or b is None or abs(b) < 1e-12 else float(a / b)


def aggregate(contexts, banks, audited):
    paired = []; totals = {}; cost_rows = []
    for c in contexts:
        key = c["split"], c["geometry"]
        for mode in MODES:
            bank = banks[key][mode]; base = bank[0]
            select = lambda ids, metric: min(ids, key=lambda i: (metric(bank[i]), i))
            selectors = dict(coefficient=0, seeded_coefficient=1, uniform=5,
                known_oracle=select(range(14), lambda r: r["known"]["scaled_norm"]),
                random_best=select(range(6, 14), lambda r: r["known"]["scaled_norm"]),
                energy_oracle=select(range(14), lambda r: float("inf") if r["energy_error_ratio"] is None else r["energy_error_ratio"]),
                heuristic_best=select((2, 3, 4), lambda r: r["known"]["scaled_norm"]),
                equal3_random_best=select((6, 8, 10), lambda r: r["known"]["scaled_norm"]))
            unique = {}
            for r in bank:
                unique.setdefault(tuple(r["selected_cells"]), r)
            for name, index in selectors.items():
                row = bank[index]
                errors, base_errors = abs(np.asarray(row["known"]["signed_scaled_error"])), abs(np.asarray(base["known"]["signed_scaled_error"]))
                paired.append(dict(split=key[0], geometry=key[1], mode=mode, selector=name,
                    selected_index=index, selected_name=row["name"], known_E=row["known"]["E"],
                    known_norm=row["known"]["scaled_norm"], audit_E=row["audit"]["E"],
                    known_ratio=ratio(row["known"]["scaled_norm"], base["known"]["scaled_norm"]),
                    audit_ratio=ratio(row["audit"]["scaled_norm"], base["audit"]["scaled_norm"]),
                    max_goal_ratio=ratio(row["known"]["maximum_scaled_error"], base["known"]["maximum_scaled_error"]),
                    energy_ratio=ratio(row["energy_error_ratio"], base["energy_error_ratio"]),
                    improved_known_goal_count=int(np.sum(errors < base_errors)),
                    energy_error_ratio=row["energy_error_ratio"], unique_mesh_count=len(unique),
                    fallback=bool(row["marking"]["fallback"])))
            for name, choices in (("expected_random_slots", bank), ("expected_random_unique", list(unique.values())),
                                   ("expected_random_subbank", bank[6:]),
                                   ("expected_random_subbank_unique", list({tuple(r['selected_cells']):r for r in bank[6:]}.values()))):
                known = float(np.mean([r["known"]["scaled_norm"] for r in choices]))
                audit_value = float(np.mean([r["audit"]["scaled_norm"] for r in choices]))
                paired.append(dict(split=key[0], geometry=key[1], mode=mode, selector=name,
                    selected_index=None, selected_name=None, known_E=None if any(r["known"]["E"] is None for r in choices)
                    else float(np.mean([r["known"]["E"] for r in choices])),
                    known_norm=known, audit_E=None if any(r["audit"]["E"] is None for r in choices)
                    else float(np.mean([r["audit"]["E"] for r in choices])),
                    known_ratio=ratio(known, base["known"]["scaled_norm"]),
                    audit_ratio=ratio(audit_value, base["audit"]["scaled_norm"]),
                    max_goal_ratio=None, energy_ratio=None, improved_known_goal_count=None,
                    energy_error_ratio=None, unique_mesh_count=len(unique), fallback=None))
        methods = sorted(set(r["method"] for r in c["baseline_runs"]))
        times = {m: median(r["full_seconds"] for r in c["baseline_runs"] if r["method"] == m) for m in methods}
        cost_rows.append(dict(split=key[0], geometry=key[1], seconds=times,
                              all_numerically_valid=all(r["numerical_valid"] for r in c["baseline_runs"])))
    names = sorted(set(r["selector"] for r in paired))
    numeric = ("known_E", "audit_E", "known_ratio", "audit_ratio", "max_goal_ratio", "energy_ratio",
               "improved_known_goal_count", "energy_error_ratio", "unique_mesh_count")
    for mode in MODES:
        totals[mode] = {}
        for split in SPLITS:
            totals[mode][split] = {}
            for name in names:
                group = [r for r in paired if (r["mode"], r["split"], r["selector"]) == (mode, split, name)]
                totals[mode][split][name] = {k: stat(r[k] for r in group) for k in numeric}
                ratios = [r["known_ratio"] for r in group if r["known_ratio"] is not None]
                totals[mode][split][name].update(count=len(group), wins=sum(r < 1 for r in ratios),
                    ten_percent_improvements=sum(r <= .9 for r in ratios))
    potential = {}; robustness = {}
    for split in SPLITS:
        value = totals["locked_ring"][split]["known_oracle"]
        potential[split] = bool(value["known_ratio"]["missing_count"] == 0 and
            value["known_ratio"]["median"] <= .8 and value["ten_percent_improvements"] >= 12)
        robustness[split] = bool(all(value[k]["missing_count"] == 0 and value[k]["median"] <= 1.05
                                      for k in ("audit_ratio", "energy_ratio")))
    potential_pass = audited["passed"] and not audited["smoke"] and all(potential.values())
    robust_pass = potential_pass and all(robustness.values())
    costs = {}
    for split in SPLITS:
        group = [c for c in cost_rows if c["split"] == split]
        methods = sorted(group[0]["seconds"])
        costs[split] = dict(method_seconds={m: stat(r["seconds"][m] for r in group) for m in methods},
                            valid_contexts=sum(r["all_numerically_valid"] for r in group), total_contexts=len(group))
    return dict(totals=totals, paired_geometry=paired, baseline_costs=costs, baseline_geometry=cost_rows,
        decision=dict(primary_mode="locked_ring", numerical_gate_passed=not audited["numerical_failures"],
                      structural_gate_passed=not audited["structural_failures"],
                      accuracy_condition_per_split=potential, accuracy_condition_passed=all(potential.values()),
                      potential_per_split=potential,
                      potential_passed=potential_pass, robustness_per_split=robustness,
                      robust_potential_passed=robust_pass, primary_ratio_limit=.8,
                      required_ten_percent_improvements=12, audit_and_energy_median_limit=1.05,
                      interpretation="Only finite-bank offline potential; not learnability, AI benefit, global mesh optimality, or a speedup."))


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}g}"


def report_text(summary, audited, source):
    d = summary["decision"]; totals = summary["totals"]
    lines = ["# 第十四阶段：统一数值验收与界面之外的有限候选潜力", "",
        *(["**本目录仅为开发/烟雾实现复核，不给出正式确认结论。**", ""] if audited["smoke"] else []),
        f"主实验固定界面闭一环内的选中/未选状态，精确保持新增自由度 1,024。共 {audited['independent_geometries']} 个独立几何；两个各14位置的候选集合共 {audited['candidate_rows']} 条候选求解记录。重复、候选与两种约束不增加独立样本量。", "",
        f"数值与结构验收：**{'通过' if audited['passed'] else '未通过'}**。主有限候选潜力门槛：**{'通过' if d['potential_passed'] else '未通过'}**；包含独立目标/能量护栏的稳健潜力总门槛：**{'通过' if d['robust_potential_passed'] else '未通过'}**。", "",
        "候选构造只读系数、源项与三个已知目标描述。已知目标 oracle 在14个预先固定候选中，使用细参考误差选择最小者；它明确支付离线参考及全候选求解，不是在线算法、全局最优网格或已训练模型。集合含原系数基线，因此 oracle 不劣于该基线是构造事实，只有非平凡改善幅度才有研究意义。", "",
        "## 主实验：固定界面一环的同预算比较", "",
        "E 是三个已知目标的缩放误差二范数除以同几何粗解误差范数，粗解为1。先平方再求和，三个目标之间不会直接正负抵消。每个目标内部的空间误差抵消仍可能存在。所有比值先在同几何计算再取中位数。", "",
        "| 选择规则 | ID E | 高对比 E | ID /系数基线 | 高对比 /系数基线 |",
        "|---|---:|---:|---:|---:|"]
    labels = dict(coefficient="原系数基线", seeded_coefficient="固定一环＋系数排序", uniform="固定一环＋确定性均匀分数",
                  known_oracle="已知目标 oracle（14位置）", random_best="随机分数子集 oracle（8位置）",
                  energy_oracle="离线能量 oracle（14位置）", heuristic_best="目标/源项启发式 oracle（3位置）",
                  equal3_random_best="固定随机子集 oracle（3位置）", expected_random_slots="均匀随机14位置的期望",
                  expected_random_unique="均匀随机14位置去重网格的期望", expected_random_subbank="随机分数8位置的期望",
                  expected_random_subbank_unique="随机分数8位置去重网格的期望")
    for name, label in labels.items():
        values = [totals["locked_ring"][s][name] for s in SPLITS]
        lines.append(f"| {label} | {fmt(values[0]['known_E']['median'])} | {fmt(values[1]['known_E']['median'])} | {fmt(values[0]['known_ratio']['median'])} | {fmt(values[1]['known_ratio']['median'])} |")
    lines += ["", "随机8位置与全14位置的最小值不是相同候选数控制；另列3位置启发式与预先固定的3位置随机子集，两者候选数相同。四个随机场各带两个强度，位置并不独立。随机期望为确定性已保存候选上的算术期望，没有额外随机试验。去重网格数与所有选择索引保存在 summary.json。", "",
        "| 组 | 主 oracle 至少改善10% | audit误差 /基线 | 能量平方误差 /基线 | 最大已知目标误差 /基线 |",
        "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        v = totals["locked_ring"][split]["known_oracle"]
        lines.append(f"| {split} | {v['ten_percent_improvements']}/{v['count']} | {fmt(v['audit_ratio']['median'])} | {fmt(v['energy_ratio']['median'])} | {fmt(v['max_goal_ratio']['median'])} |")
    lines += ["", "主门槛要求两组配对误差比中位数均≤0.8，且各至少12/16几何改善≥10%，并通过全部数值、预算、覆盖与固定一环检查。额外护栏要求同一已知目标赢家的三个独立audit目标误差比和能量平方误差比中位数均≤1.05，分母不得退化。audit目标从未参与候选构造或选择；护栏失败不否定已知目标有限候选空间，但阻止称为稳健、可转移的潜力。", "",
        "## 次实验与机制边界", "",
        "次实验只固定全部粗格内部界面覆盖，允许一环其他状态改变。其 oracle /原系数基线比：", "",
        "| 组 | 固定一环（主） | 仅固定覆盖（次） |",
        "|---|---:|---:|"]
    for split in SPLITS:
        lines.append(f"| {split} | {fmt(totals['locked_ring'][split]['known_oracle']['known_ratio']['median'])} | {fmt(totals['coverage'][split]['known_oracle']['known_ratio']['median'])} |")
    lines += ["", "界面标签仅指2×2细系数在粗格内部同时出现两相，不包括恰沿粗格边界的界面。固定闭一环还固定了这些界面格与邻格共享边中点的自由度状态；不等于固定所有物理界面、通量误差或全域影响。改变其他格后的椭圆解仍全局耦合。若只有次实验改善，不能解释为已排除界面邻近细化收益。", "",
        "## 求解器验收及成本边界", "",
        "直接法与缩放 AMG-PCG 对同一原始双精度矩阵使用统一验收：扩展精度计算的实际相对残差≤1e-9，分量后向误差≤1e-12，所有求解器状态成功。扩展精度仅用于残差评估，不代表高精度 PDE 装配或高精度真实解。设置、全部修正与验收计入完整基线成本。第十三阶段原有失败结论保持不变。", "",
        "| 组 | 基线方法 | 完整路径 ms 中位数 | 全部数值合格上下文 |",
        "|---|---|---:|---:|"]
    for split in SPLITS:
        v = summary["baseline_costs"][split]
        for name, time in v["method_seconds"].items():
            lines.append(f"| {split} | {name} | {fmt(1000*time['median'])} | {v['valid_contexts']}/{v['total_contexts']} |")
    lines += ["", "候选集合重复使用离线已经支付的载荷/参考数据，其 seconds 字段不能与完整基线路径当作在线速度比较。本轮没有训练、推理或等精度加速证明。", "",
        "## 独立复核", "",
        f"核对 {audited['frozen_input_count']} 个冻结输入哈希、原始输出哈希，以独立NumPy公式重建 {audited['regenerated_coefficient_count']} 个系数哈希、界面及一环、全部高斯函数参数、{audited['regenerated_score_rows']} 个候选分数；独立重建分数解码、自由度预算，以及全部已知/audit目标误差和 oracle 选择。独立检查完整基线 {audited['baseline_run_checks']} 次。结构失败 {len(audited['structural_failures'])} 条，数值失败 {len(audited['numerical_failures'])} 条；全部记录保留。", "",
        "报告不导入 PDE 代码、不重新求解；保存数据没有完整解场，无法独立重算残差、后向误差或能量。这里重建其验收逻辑，相关数值证据来自冻结运行器与独立数值测试。高斯函数参数已重建，载荷求积及保存的载荷数组哈希没有重新计算；不将函数参数复核等同于复核载荷装配。", "",
        "![Stage 14 finite-bank potential and safeguards](figures/comparison.png)", "",
        f"原始数据：`{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`。", ""]
    return "\n".join(lines)


def plot(summary, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/darcy-u-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(id="#216faf", contrast="#d68b26")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    panels = [("known_ratio", "A  Fixed-ring known-goal oracle / coefficient"),
              ("audit_ratio", "B  Same winner: independent audit-goal ratio"),
              ("energy_ratio", "C  Same winner: squared energy-error ratio")]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for j, split in enumerate(SPLITS):
            values = [r[metric] for r in summary["paired_geometry"] if r["mode"] == "locked_ring"
                      and r["selector"] == "known_oracle" and r["split"] == split and r[metric] is not None]
            ax.scatter(j + np.linspace(-.09, .09, len(values)), values, color=colors[split], s=23, alpha=.7)
            if values:
                ax.plot([j-.15, j+.15], [median(values)] * 2, color=colors[split], lw=3)
        ax.axhline(1, color="#666", ls="--", lw=1)
        ax.axhline(.8 if metric == "known_ratio" else 1.05, color="#888", ls=":", lw=1)
        ax.set_xticks([0, 1], ["ID", "High contrast"]); ax.set_ylabel("Within-geometry ratio")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=.2)
    ax = axes[1, 1]
    selectors = ("known_oracle", "random_best", "energy_oracle", "expected_random_slots")
    for j, split in enumerate(SPLITS):
        values = [summary["totals"]["locked_ring"][split][n]["known_ratio"]["median"] for n in selectors]
        ax.plot(np.arange(len(selectors)), values, "o-", color=colors[split], label="ID" if split == "id" else "High contrast")
    ax.set_xticks(range(4), ["Known best\n14 slots", "Random best\n8 slots", "Energy best\n14 slots", "Random mean\n14 slots"])
    ax.set_xlim(-.35, 3.35); ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("Median within-geometry known-goal ratio"); ax.axhline(1, color="#666", ls="--", lw=1)
    ax.set_title("D  Offline selection controls (unequal pool sizes)", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fixed interface-ring states: finite-bank offline potential", fontsize=16, y=.98)
    fig.text(.5, .025, "Each dot is one geometry. Bars are paired medians. Audit goals do not enter construction or selection.\nThe bank contains the coefficient baseline; best-of-bank dominance alone is automatic.\nNo trained model, global mesh optimum, learnability result, or equal-accuracy speedup is claimed.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.15, hspace=.34, wspace=.25)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(source, output):
    require(not output.exists(), "Use a new report directory; never overwrite recorded reports")
    frozen, complete, contexts, rows, banks, audited = audit(source)
    summary = aggregate(contexts, banks, audited); summary["audit"] = audited
    output.mkdir(parents=True)
    dump(output / "summary.json", summary)
    (output / "REPORT_zh.md").write_text(report_text(summary, audited, source))
    plot(summary, output / "figures")
    files = ("summary.json", "REPORT_zh.md", "figures/comparison.png", "figures/comparison.pdf")
    dump(output / "complete.json", dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        reporter_sha256=sha(Path(__file__)), source_sha256={name: sha(source / name) for name in
            ("freeze.json", "complete.json", *sorted(complete["output_sha256"]))},
        output_sha256={name: sha(output / name) for name in files}, decision=summary["decision"],
        no_pde_solver_imported=True, no_ai_model_used=True))
    print(json.dumps(dict(output=str(output), audit_passed=audited["passed"], decision=summary["decision"]), ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/stage14_confirmation")
    parser.add_argument("--output", default="results/stage14_report")
    args = parser.parse_args()
    run((ROOT / args.source).resolve(), (ROOT / args.output).resolve())
