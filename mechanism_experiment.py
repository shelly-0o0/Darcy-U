#!/usr/bin/env python3
"""Stage 4: diagnose slow-direction coverage and test causal augmentation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from statistics import median

ROOT = Path(__file__).resolve().parent
if (ROOT / "vendor").exists():
    sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import scipy.sparse.linalg as sla
import torch
from threadpoolctl import threadpool_limits
from fields import coefficient_sequence
from numerics import assemble_system, energy_relative_error
from sequence_baselines import AMGSequencePreconditioner
from sequence_experiment import guess, hard_change, load_network, predict, manifest as prior_manifest
from balanced_augmentation import a_orthonormalize, balanced_preconditioner, galerkin_correction, extract_slow_ritz
from traced_pcg import solve_pcg_traced

RTOL, STEPS, CONTRAST, LIMIT = 1e-8, 20, 1e4, 24
CHECKPOINT = "checkpoints/correction_mixed.pt"
SOURCES = ["fields.py", "numerics.py", "sequence_baselines.py", "sequence_experiment.py",
           "correction_model.py", "balanced_augmentation.py", "traced_pcg.py", "mechanism_experiment.py"]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fresh(name):
    p = ROOT / "results" / name
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def geometry_hash(seed):
    fields, _ = coefficient_sequence(32, STEPS, seed, CONTRAST, "mixed")
    return hashlib.sha256((fields > 1).tobytes()).hexdigest()


def splits():
    p = ROOT / "data/stage4_splits.json"
    if p.exists():
        return read(p)
    old = read(ROOT / "data/stage3_splits.json")
    excluded = sorted(set(old["excluded_prior_seeds"] + sum(old["groups"].values(), [])))
    seen = {geometry_hash(s) for s in excluded}
    result = {"excluded_seeds": excluded, "groups": {}, "geometry_sha256": {}, "skipped_duplicates": []}
    for name, start, count in [("development", 42001, 3), ("confirmation", 42011, 5)]:
        seeds = []
        while len(seeds) < count:
            digest = geometry_hash(start)
            if digest in seen:
                result["skipped_duplicates"].append({"seed": start, "group": name})
            else:
                seen.add(digest); seeds.append(start)
                result["geometry_sha256"][str(start)] = digest
            start += 1
        result["groups"][name] = seeds
    save(p, result)
    return result


def manifest(configuration):
    result = prior_manifest(configuration)
    result.update(source_sha256={f: sha(ROOT / f) for f in SOURCES},
                  protocol_sha256=sha(ROOT / "docs/stage4_protocol_zh.md"),
                  splits_sha256=sha(ROOT / "data/stage4_splits.json"),
                  checkpoint_sha256={CHECKPOINT: sha(ROOT / CHECKPOINT)})
    return result


def columns(vectors, n):
    return np.column_stack(vectors) if len(vectors) else np.empty((n, 0))


def check_provenance(record):
    for key in ["source_sha256", "checkpoint_sha256"]:
        for file, expected in record[key].items():
            if sha(ROOT / file) != expected:
                raise RuntimeError(f"Frozen {file} changed")
    for key, file in [("protocol_sha256", "docs/stage4_protocol_zh.md"),
                      ("splits_sha256", "data/stage4_splits.json")]:
        if sha(ROOT / file) != record[key]:
            raise RuntimeError(f"Frozen {file} changed")


def specs():
    return [{"name": "baseline", "source": "none", "interface": "warm", "rank": 0},
        {"name": "network_warm", "source": "network", "interface": "warm", "rank": 1},
        {"name": "network_balanced", "source": "network", "interface": "balanced", "rank": 1},
        {"name": "history_balanced4", "source": "history", "interface": "balanced", "rank": 4},
        {"name": "delta_warm4", "source": "delta", "interface": "warm", "rank": 4},
        {"name": "delta_balanced4", "source": "delta", "interface": "balanced", "rank": 4},
        {"name": "ritz_warm4", "source": "ritz", "interface": "warm", "rank": 4},
        {"name": "ritz_balanced4", "source": "ritz", "interface": "balanced", "rank": 4},
        {"name": "ritz_balanced8", "source": "ritz", "interface": "balanced", "rank": 8}]


def run_sequence(fields, events, spec, repeat, model):
    history, deltas, rows, solutions = [], [], [], []
    recycle = np.empty(((fields.shape[-1] - 1) ** 2, 0))
    policy = AMGSequencePreconditioner("refresh_galerkin", seed=events[0]["seed"] * 100)
    old_k = None
    for step, k in enumerate(fields):
        started = time.perf_counter()
        row = {"method": spec["name"], "n": fields.shape[-1], "seed": events[0]["seed"],
               "step": step, "repeat": repeat, "event": events[step]["event"],
               "inference_seconds": 0., "maintenance_seconds": 0., "solver_fallback": False,
               "recovery_seconds": 0., "network_called": False}
        t = time.perf_counter(); A, b, _ = assemble_system(k)
        row["assembly_seconds"] = time.perf_counter() - t
        t = time.perf_counter(); base = guess(A, b, history, "history4")
        row["guess_seconds"] = time.perf_counter() - t
        row["base_relative_residual"] = float(np.linalg.norm(b - A @ base) / np.linalg.norm(b))
        M, setup = policy.for_matrix(A)
        row["setup_seconds"] = setup["setup_seconds"]
        event = old_k is None or hard_change(k, old_k)
        t = time.perf_counter()
        W = np.empty((len(b), 0))
        if spec["source"] == "network" and old_k is not None and event and row["base_relative_residual"] > RTOL:
            it = time.perf_counter()
            W = predict(model, "cuda", k, old_k, A, b, base)[:, None]
            row["inference_seconds"] = time.perf_counter() - it
            row["network_called"] = True
        elif spec["source"] == "history":
            W = columns(history[-spec["rank"]:], len(b))
        elif spec["source"] == "delta":
            W = columns(deltas[-spec["rank"]:], len(b))
        elif spec["source"] == "ritz":
            W = recycle
        if W.shape[1]:
            U, AU, info = a_orthonormalize(A, W)
            x0, _ = galerkin_correction(A, b, base, U, AU)
            operator = balanced_preconditioner(M, U, AU) if spec["interface"] == "balanced" else M
            row["rank"] = info["rank"]
            row["orthogonality_error"] = info["orthogonality_error"]
        else:
            x0, operator = base, M
            row.update(rank=0, orthogonality_error=0.)
        row["correction_seconds"] = time.perf_counter() - t
        row["corrected_relative_residual"] = float(np.linalg.norm(b - A @ x0) / np.linalg.norm(b))
        capture = LIMIT if spec["source"] == "ritz" and event else 0
        x, metrics, captured, _ = solve_pcg_traced(A, b, operator, x0, RTOL, 500, max_vectors=capture)
        row.update({("solve_seconds" if key == "seconds" else key): value for key, value in metrics.items()})
        if not metrics["converged"]:
            t = time.perf_counter(); x = sla.spsolve(A, b)
            row["recovery_seconds"] = time.perf_counter() - t
            row["solver_fallback"] = True
        row["final_relative_residual"] = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
        row["final_converged"] = row["final_relative_residual"] <= RTOL
        t = time.perf_counter()
        if spec["source"] == "ritz" and event:
            P = np.column_stack((recycle, columns(captured, len(b))))
            recycle, _, extraction = extract_slow_ritz(A, M, P, q=spec["rank"], max_basis=32)
            row["extraction"] = extraction
        if spec["source"] == "delta":
            deltas.append(x - base); deltas = deltas[-spec["rank"]:]
        row["maintenance_seconds"] = time.perf_counter() - t
        history.append(x.copy()); history = history[-4:]
        old_k = k
        row["total_seconds"] = time.perf_counter() - started
        rows.append(row); solutions.append(x.copy())
    print(json.dumps({"stage": "sequence", "method": spec["name"], "n": fields.shape[-1],
        "seed": events[0]["seed"], "repeat": repeat, "seconds": sum(r["total_seconds"] for r in rows),
        "iterations": sum(r["iterations"] for r in rows), "pcg_failures": sum(not r["converged"] for r in rows)}), flush=True)
    return rows, solutions


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["method"], r["seed"], r["repeat"])].append(r)
    sequences = [{"n": n, "method": method, "seed": seed, "repeat": repeat,
        "seconds": sum(r["total_seconds"] for r in group), "iterations": sum(r["iterations"] for r in group),
        "accurate": all(r["final_converged"] for r in group), "pcg_failures": sum(not r["converged"] for r in group)}
        for (n, method, seed, repeat), group in grouped.items()]
    summary = {}
    for n in sorted({s["n"] for s in sequences}):
        summary[str(n)] = {}
        for name in sorted({s["method"] for s in sequences if s["n"] == n}):
            local = [s for s in sequences if s["n"] == n and s["method"] == name]
            by_seed = {str(seed): median(s["seconds"] for s in local if s["seed"] == seed)
                       for seed in sorted({s["seed"] for s in local})}
            summary[str(n)][name] = {"by_seed": by_seed, "median_seconds": median(by_seed.values()),
                "all_accurate": all(s["accurate"] for s in local), "pcg_failures": sum(s["pcg_failures"] for s in local)}
    return sequences, summary


def benchmark(name, n, seeds, candidates, repeats, role, selection_hash=None):
    out = fresh(name)
    config = {"n": n, "seeds": seeds, "specs": candidates, "repeats": repeats, "role": role}
    if selection_hash:
        config["selection_sha256"] = selection_hash
    save(out / "manifest.json", manifest(config))
    model = load_network(ROOT / CHECKPOINT, "cuda", n)
    rows, references = [], []
    for seed in seeds:
        fields, events = coefficient_sequence(n, STEPS, seed, CONTRAST, "mixed")
        t = time.perf_counter(); truth, residuals = [], []
        for k in fields:
            A, b, _ = assemble_system(k); x = sla.spsolve(A, b)
            truth.append(x); residuals.append(float(np.linalg.norm(b - A @ x) / np.linalg.norm(b)))
        references.append({"seed": seed, "seconds": time.perf_counter() - t,
            "field_sha256": hashlib.sha256(fields.tobytes()).hexdigest(), "relative_residuals": residuals})
        save(out / "references.json", references)
        for repeat in range(repeats):
            for index in np.random.default_rng(seed + repeat + n).permutation(len(candidates)):
                measured, xs = run_sequence(fields, events, candidates[int(index)], repeat, model)
                for step, row in enumerate(measured):
                    A, _, _ = assemble_system(fields[step])
                    row["energy_relative_error"] = energy_relative_error(A, xs[step], truth[step])
                rows.extend(measured); save(out / "measurements.json", rows)
    _, summary = aggregate(rows)
    save(out / "summary.json", summary)
    save(out / "complete.json", {"complete": True, "rows": len(rows), "finished_utc": stamp()})
    return out


def diagnose():
    out = fresh("stage4_diagnosis")
    seeds = splits()["groups"]["development"]
    save(out / "manifest.json", manifest({"n": 128, "seeds": seeds, "steps_observed": [0, 6, 10, 15],
        "role": "offline retrospective current-trajectory Ritz probes; not an online timing comparison"}))
    model = load_network(ROOT / CHECKPOINT, "cuda", 128)
    rows = []
    for seed in seeds:
        fields, events = coefficient_sequence(128, STEPS, seed, CONTRAST, "mixed")
        policy = AMGSequencePreconditioner("refresh_galerkin", seed=seed * 100)
        history, old_k = [], None
        for step, k in enumerate(fields):
            A, b, _ = assemble_system(k); base = guess(A, b, history, "history4")
            M, _ = policy.for_matrix(A)
            observed = step in [0, 6, 10, 15]
            x, stats, captured, trace = solve_pcg_traced(A, b, M, base, RTOL, 500,
                max_vectors=32 if observed else 0, record_residuals=observed)
            if not stats["converged"]:
                x = sla.spsolve(A, b)
            if observed:
                reference = sla.spsolve(A, b)
                e = reference - base; initial_energy = float(e @ (A @ e))
                t = time.perf_counter()
                U8, AU8, ritz = extract_slow_ritz(A, M, columns(captured, len(b)), q=8, max_basis=32)
                extraction_seconds = time.perf_counter() - t
                U4, AU4 = U8[:, :4], AU8[:, :4]
                rbase = b - A @ base
                slow_initial = float(np.sum((U4.T @ rbase) ** 2))
                record = {"seed": seed, "step": step, "event": events[step]["event"],
                    "initial_error_energy": initial_energy,
                    "reference_relative_residual": float(np.linalg.norm(b - A @ reference) / np.linalg.norm(b)),
                    "slow4_error_energy_fraction": slow_initial / max(initial_energy, 1e-300),
                    "ritz": ritz, "retrospective_extraction_seconds": extraction_seconds,
                    "variants": {"baseline": {"iterations": stats["iterations"], "converged": stats["converged"],
                        "final_relative_residual": stats["relative_residual"], "trace": trace,
                        "initial_relative_residual": float(np.linalg.norm(rbase) / np.linalg.norm(b)),
                        "error_energy_over_base": initial_energy / max(initial_energy, 1e-300),
                        "slow4_energy_over_base": slow_initial / max(slow_initial, 1e-300),
                        "solve_seconds": stats["seconds"], "captured_vectors": stats["captured_vectors"],
                        "restarts": stats["restarts"]}}}
                probes = [("current_ritz4_warm", U4, AU4, False),
                    ("current_ritz4_balanced", U4, AU4, True), ("current_ritz8_balanced", U8, AU8, True)]
                if old_k is not None:
                    d = predict(model, "cuda", k, old_k, A, b, base)
                    UN, AUN, _ = a_orthonormalize(A, d)
                    probes.extend([("network_warm", UN, AUN, False), ("network_balanced", UN, AUN, True)])
                    denergy = float(d @ (A @ d))
                    record["network_direction_slow4_overlap"] = float(np.sum((AU4.T @ d) ** 2)) / max(denergy, 1e-300)
                for label, U, AU, balanced in probes:
                    initial, guard = galerkin_correction(A, b, base, U, AU)
                    operator = balanced_preconditioner(M, U, AU) if balanced else M
                    solution, metrics, _, trajectory = solve_pcg_traced(A, b, operator, initial, RTOL, 500,
                        record_residuals=True)
                    error = reference - initial; residual = b - A @ initial
                    record["variants"][label] = {"iterations": metrics["iterations"], "converged": metrics["converged"],
                        "final_relative_residual": metrics["relative_residual"],
                        "initial_relative_residual": float(np.linalg.norm(residual) / np.linalg.norm(b)),
                        "error_energy_over_base": float(error @ (A @ error)) / max(initial_energy, 1e-300),
                        "slow4_energy_over_base": float(np.sum((U4.T @ residual) ** 2)) / max(slow_initial, 1e-300),
                        "energy_change": guard["energy_change"], "trace": trajectory}
                rows.append(record); save(out / "measurements.json", rows)
                print(json.dumps({"stage": "diagnosis", "seed": seed, "step": step,
                    "iterations": {name: v["iterations"] for name, v in record["variants"].items()}}), flush=True)
            history.append(x.copy()); history = history[-4:]; old_k = k
    save(out / "complete.json", {"complete": True, "cases": len(rows), "finished_utc": stamp()})


def screen():
    out = benchmark("stage4_screen", 128, splits()["groups"]["development"], specs(), 2, "development mechanism screen")
    check_provenance(read(out / "manifest.json"))
    result = read(out / "summary.json")["128"]
    eligible = [s for s in specs() if s["source"] not in ("none", "network") and
        result[s["name"]]["all_accurate"] and not result[s["name"]]["pcg_failures"]]
    if not eligible:
        raise RuntimeError("No accurate failure-free classical mechanism; keep screen records")
    chosen = min(eligible, key=lambda s: result[s["name"]]["median_seconds"])
    frozen = {"created_utc": stamp(), "classic_candidate": chosen,
        "confirmation_specs": [specs()[0], chosen, specs()[1], specs()[2]],
        "source_sha256": {f: sha(ROOT / f) for f in SOURCES},
        "protocol_sha256": sha(ROOT / "docs/stage4_protocol_zh.md"),
        "splits_sha256": sha(ROOT / "data/stage4_splits.json"),
        "checkpoint_sha256": {CHECKPOINT: sha(ROOT / CHECKPOINT)},
        "screen_measurements_sha256": sha(out / "measurements.json"),
        "screen_manifest_sha256": sha(out / "manifest.json"),
        "rule": "Fastest accurate no-failure non-baseline classical mechanism on development only; not an AI model selection."}
    save(out / "frozen_selection.json", frozen)
    print(json.dumps({"stage": "screen_selected", "candidate": chosen}), flush=True)


def confirm():
    p = ROOT / "results/stage4_screen/frozen_selection.json"; selected = read(p)
    check_provenance(selected)
    screen_dir = p.parent
    for key, file in [("screen_manifest_sha256", "manifest.json"),
                      ("screen_measurements_sha256", "measurements.json")]:
        if sha(screen_dir / file) != selected[key]:
            raise RuntimeError(f"Frozen screen {file} changed")
    check_provenance(read(screen_dir / "manifest.json"))
    complete = read(screen_dir / "complete.json")
    measured = read(screen_dir / "measurements.json")
    expected = {(128, s["name"], seed, repeat, step) for s in specs()
                for seed in splits()["groups"]["development"] for repeat in range(2) for step in range(STEPS)}
    actual = {(r["n"], r["method"], r["seed"], r["repeat"], r["step"]) for r in measured}
    if not complete["complete"] or complete["rows"] != len(expected) or len(measured) != len(expected) or actual != expected:
        raise RuntimeError("Incomplete or duplicated development screen")
    seeds = splits()["groups"]["confirmation"]
    for n, cohort in [(128, seeds), (256, seeds[:2])]:
        benchmark(f"stage4_confirm{n}", n, cohort, selected["confirmation_specs"], 3,
            "frozen mechanism confirmation; not a paper-level final test", sha(p))
    save(ROOT / "results/stage4_complete.json", {"complete": True, "finished_utc": stamp(), "selection_sha256": sha(p)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["splits", "diagnose", "screen", "confirm"])
    args = parser.parse_args(); splits(); torch.set_num_threads(2)
    with threadpool_limits(limits=1, user_api="blas"):
        if args.stage == "diagnose": diagnose()
        elif args.stage == "screen": screen()
        elif args.stage == "confirm": confirm()


if __name__ == "__main__":
    main()
