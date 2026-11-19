#!/usr/bin/env python3
"""Stage 3: strong classical sequence baselines and learned initial corrections.

Network output changes the initial iterate only. Each following PCG has a
fixed classical preconditioner and the original explicit true-residual test.
Offline references are never passed to the online sequence function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
if (ROOT / "vendor").exists():
    sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import torch
from threadpoolctl import threadpool_limits
from fields import coefficient_sequence
from numerics import assemble_system, free_to_grid, solve_pcg, energy_relative_error
from sequence_baselines import (AMGSequencePreconditioner, galerkin_initial_guess,
    damped_initial_correction, jacobi_initial_correction)
from coarse_correction import CoarseGalerkinCorrection
from correction_model import (CorrectionUNet, make_correction_features,
    relative_energy_loss, weighted_residual_loss, p1_energy)

RTOL = 1e-8
MAXITER = 500
STEPS = 20
CONTRAST = 1e4
CORE_SOURCES = ["fields.py", "numerics.py", "sequence_baselines.py",
    "coarse_correction.py", "correction_model.py", "sequence_experiment.py"]


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_digest(seed):
    k, _ = coefficient_sequence(32, STEPS, seed, CONTRAST, "mixed")
    return hashlib.sha256((k > 1).tobytes()).hexdigest()


def splits():
    path = ROOT / "data/stage3_splits.json"
    if path.exists():
        return read(path)
    old = read(ROOT / "data/pilot.manifest.json")
    excluded = sorted(set(old["train_seeds"] + old["val_seeds"] +
        [90001, 90002, 90003, 75001] + list(range(70001, 70021))))
    seen = {geometry_digest(s) for s in excluded}
    result = {"excluded_prior_seeds": excluded, "groups": {}, "geometry_sha256": {},
        "skipped_duplicates": [], "test_role": "Never used for model/backend selection"}
    for name, start, count in [("screen", 72001, 3), ("train", 20000, 64),
            ("validation", 30000, 12), ("calibration", 74001, 3), ("test", 80001, 12)]:
        chosen = []
        seed = start
        while len(chosen) < count:
            key = geometry_digest(seed)
            if key in seen:
                result["skipped_duplicates"].append({"seed": seed, "group": name})
            else:
                seen.add(key)
                chosen.append(seed)
                result["geometry_sha256"][str(seed)] = key
            seed += 1
        result["groups"][name] = chosen
    save(path, result)
    return result


def manifest(configuration):
    import pyamg
    return {"configuration": configuration, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "pyamg": pyamg.__version__,
        "cuda_available": torch.cuda.is_available(), "blas_threads": 1, "torch_threads": 2,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rtol": RTOL, "maxiter": MAXITER, "steps": STEPS, "contrast": CONTRAST,
        "source_sha256": {f: digest(ROOT / f) for f in CORE_SOURCES},
        "protocol_sha256": digest(ROOT / "docs/stage3_protocol_zh.md"),
        "splits_sha256": digest(ROOT / "data/stage3_splits.json"),
        "timing": "All current-system assembly, history, hierarchy, features, transfer, network, safeguards, PCG and recovery are included. References, training and model load/prewarm are separate."}


def fresh_output(name):
    out = ROOT / "results" / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def guess(A, b, history, kind):
    if not history:
        return np.zeros_like(b)
    if kind == "previous":
        return history[-1].copy()
    if kind == "history4":
        return galerkin_initial_guess(A, b, history, max_vectors=4)[0]
    raise ValueError(kind)


def hard_change(k, old_k):
    # A fixed input-only trigger: ratio at least 10 in either direction.
    # The network is not invoked on the small drift-only transitions.
    return old_k is not None and float(np.max(np.abs(np.log(k / old_k)))) >= np.log(10.)


def load_network(path, device, n=128):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = CorrectionUNet(state["width"])
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    with torch.inference_mode():
        model(torch.zeros(1, 6, n + 1, n + 1, device=device))
    if device == "cuda":
        torch.cuda.synchronize()
    return model


def predict(model, device, k, old_k, A, b, x):
    scale = max(float(np.sqrt(np.mean(x*x))), 1e-8)
    features, _ = make_correction_features(k, old_k, b - A @ x, x, solution_scale=scale)
    with torch.inference_mode():
        output = model(torch.from_numpy(features)[None].to(device))
        direction = output[0, 0, 1:-1, 1:-1].detach().cpu().numpy().astype(np.float64).ravel() * scale
    return direction


def run_sequence(fields, events, spec, repeat, model=None):
    policy = AMGSequencePreconditioner(spec["mode"], seed=events[0]["seed"] * 100)
    coarse = None
    history, records, solutions = [], [], []
    old_k = None
    n = fields.shape[-1]
    for step, k in enumerate(fields):
        started = time.perf_counter()
        row = {"method": spec["name"], "n": n, "seed": events[0]["seed"],
            "step": step, "repeat": repeat, "event": events[step]["event"],
            "assembly_seconds": 0., "guess_seconds": 0., "correction_seconds": 0.,
            "inference_seconds": 0., "safeguard_seconds": 0., "setup_seconds": 0.,
            "solve_seconds": 0., "recovery_seconds": 0.,
            "network_called": False, "solver_fallback": False,
            "correction_alpha": 0., "correction_energy_change": 0.}
        t = time.perf_counter()
        A, b, _ = assemble_system(k)
        row["assembly_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        x0 = guess(A, b, history, spec["guess"])
        row["guess_seconds"] = time.perf_counter() - t
        rnorm = float(np.linalg.norm(b - A @ x0) / np.linalg.norm(b))
        row["base_relative_residual"] = rnorm
        correction = spec.get("correction", "none")
        trigger = hard_change(k, old_k) and rnorm > RTOL
        t = time.perf_counter()
        if trigger and correction == "jacobi":
            x0, _ = jacobi_initial_correction(A, b, x0, sweeps=3)
        elif trigger and correction == "coarse":
            if coarse is None:
                coarse = CoarseGalerkinCorrection(n, coarse_n=32)
            x0, _ = coarse.apply(A, b, x0)
        elif trigger and correction == "network":
            it = time.perf_counter()
            direction = predict(model, spec["device"], k, old_k, A, b, x0)
            row["inference_seconds"] = time.perf_counter() - it
            row["network_called"] = True
            st = time.perf_counter()
            if spec.get("smooth_prediction", 0):
                # Smooth the proposed full initial iterate, with the same
                # current-system Jacobi operation used in the classical control.
                smoothed, _ = jacobi_initial_correction(A, b, x0 + direction,
                    sweeps=spec["smooth_prediction"])
                direction = smoothed - x0
            x0, guard = damped_initial_correction(A, b, x0, direction)
            row["correction_alpha"] = guard["alpha"]
            row["correction_energy_change"] = guard["energy_change"]
            row["safeguard_seconds"] = time.perf_counter() - st
        row["correction_seconds"] = time.perf_counter() - t
        row["corrected_relative_residual"] = float(np.linalg.norm(b - A @ x0) / np.linalg.norm(b))
        M, setup = policy.for_matrix(A)
        row.update(setup)
        x, metrics = solve_pcg(A, b, M, x0, RTOL, MAXITER)
        row.update({("solve_seconds" if key == "seconds" else key): val for key, val in metrics.items()})
        if not metrics["converged"]:
            t = time.perf_counter()
            x = sla.spsolve(A, b)
            row["recovery_seconds"] = time.perf_counter() - t
            row["solver_fallback"] = True
        row["final_relative_residual"] = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
        row["final_converged"] = row["final_relative_residual"] <= RTOL
        # Only completed computed solutions, never current references, enter history.
        history.append(x.copy())
        history = history[-4:]
        old_k = k
        row["total_seconds"] = time.perf_counter() - started
        records.append(row)
        solutions.append(x.copy())
    print(json.dumps({"stage": "sequence", "method": spec["name"], "n": n,
        "seed": events[0]["seed"], "repeat": repeat,
        "seconds": sum(r["total_seconds"] for r in records),
        "iterations": sum(r["iterations"] for r in records),
        "pcg_failures": sum(not r["converged"] for r in records)}), flush=True)
    return records, solutions


def benchmark(name, grids, seeds, specs_by_n, repeats, role):
    out = fresh_output(name)
    config = {"grids": grids, "seeds": seeds, "specs_by_n": specs_by_n,
        "repeats": repeats, "role": role}
    if role.startswith("held-out") or role.startswith("resolution transfer"):
        config["frozen_selection_sha256"] = digest(ROOT / "results/stage3_calibration_optimized/frozen_selection.json")
    save(out / "manifest.json", manifest(config))
    models = {}
    for n, specs in specs_by_n.items():
        for spec in specs:
            if spec.get("correction") == "network":
                key = (spec["checkpoint"], spec["device"])
                if key not in models:
                    models[key] = load_network(ROOT / spec["checkpoint"], spec["device"], int(n))
    weights = {key[0]: digest(ROOT / key[0]) for key in models}
    save(out / "checkpoint_sha256.json", weights)
    rows = []
    reference_info = []
    for n in grids:
        for seed in seeds:
            fields, events = coefficient_sequence(n, STEPS, seed, CONTRAST, "mixed")
            t = time.perf_counter()
            references = []
            for k in fields:
                A, b, _ = assemble_system(k)
                references.append(sla.spsolve(A, b))
            reference_info.append({"n": n, "seed": seed, "seconds": time.perf_counter() - t,
                "field_sha256": hashlib.sha256(fields.tobytes()).hexdigest()})
            save(out / "references.json", reference_info)
            for repeat in range(repeats):
                order = np.random.default_rng(seed + repeat + n).permutation(len(specs_by_n[str(n)]))
                for index in order:
                    spec = specs_by_n[str(n)][int(index)]
                    model = models.get((spec.get("checkpoint"), spec.get("device")))
                    measured, xs = run_sequence(fields, events, spec, repeat, model)
                    for step, (row, x) in enumerate(zip(measured, xs)):
                        A, _, _ = assemble_system(fields[step])
                        row["energy_relative_error"] = energy_relative_error(A, x, references[step])
                    rows.extend(measured)
                    save(out / "measurements.json", rows)
    save(out / "complete.json", {"complete": True, "rows": len(rows), "finished_utc":
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return out


def classical_specs():
    return [{"name": f"{mode}_{g}", "mode": mode, "guess": g, "correction": "none"}
        for mode in ("fresh", "reuse_first", "refresh_galerkin")
        for g in ("previous", "history4")]


def aggregate_rows(rows):
    from collections import defaultdict
    from statistics import median
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["method"], r["seed"], r["repeat"])].append(r)
    units = []
    for (n, method, seed, repeat), rr in sorted(grouped.items()):
        units.append({"n": n, "method": method, "seed": seed, "repeat": repeat,
            "seconds": sum(r["total_seconds"] for r in rr),
            "iterations": sum(r["iterations"] for r in rr),
            "accuracy": all(r["final_converged"] and r["final_relative_residual"] <= RTOL for r in rr),
            "pcg_failures": sum(not r["converged"] for r in rr)})
    summaries = {}
    for n in sorted({u["n"] for u in units}):
        summaries[str(n)] = {}
        for method in sorted({u["method"] for u in units if u["n"] == n}):
            selected = [u for u in units if u["n"] == n and u["method"] == method]
            by_seed = {str(seed): median(u["seconds"] for u in selected if u["seed"] == seed)
                for seed in sorted({u["seed"] for u in selected})}
            summaries[str(n)][method] = {"median_seconds": median(by_seed.values()),
                "by_seed": by_seed, "all_accurate": all(u["accuracy"] for u in selected),
                "pcg_failures": sum(u["pcg_failures"] for u in selected),
                "mean_iterations": float(np.mean([u["iterations"] for u in selected]))}
    return units, summaries


def screen():
    split = splits()
    out = benchmark("stage3_screen", [128, 256], split["groups"]["screen"],
        {str(n): classical_specs() for n in (128, 256)}, 2, "development screen")
    _, summary = aggregate_rows(read(out / "measurements.json"))
    chosen = {}
    for n, methods in summary.items():
        valid = [name for name, r in methods.items() if r["all_accurate"] and not r["pcg_failures"]]
        name = min(valid, key=lambda m: methods[m]["median_seconds"])
        chosen[n] = next(spec for spec in classical_specs() if spec["name"] == name)
    save(out / "selection.json", {"rule": "Minimum complete-sequence median on screen geometries; accurate and no PCG failures required.",
        "summary": summary, "selected_by_grid": chosen})
    print(json.dumps({"stage": "screen_selected", "selected": chosen}), flush=True)


def build_data():
    split = splits()
    spec = read(ROOT / "results/stage3_screen/selection.json")["selected_by_grid"]["128"]
    dest = ROOT / "data/stage3_correction.npz"
    if dest.exists():
        raise FileExistsError(dest)
    data, meta = {}, {"configuration": {"n": 128, "steps": STEPS, "contrast": CONTRAST,
        "base_spec": spec, "trigger": "max abs log coefficient ratio >= log(10)",
        "label_source": "offline direct solutions; previous exact solutions approximate deployment history to solver tolerance",
        "solution_scale": "max(RMS available base solution),1e-8"},
        "groups": split["groups"], "sample_metadata": {}, "source_sha256": {f: digest(ROOT/f) for f in CORE_SOURCES}}
    start = time.perf_counter()
    for group in ("train", "validation"):
        features, targets, coefficients, metadata = [], [], [], []
        for seed in split["groups"][group]:
            fields, events = coefficient_sequence(128, STEPS, seed, CONTRAST, "mixed")
            history, previous_k = [], None
            for step, k in enumerate(fields):
                A, b, _ = assemble_system(k)
                base = guess(A, b, history, spec["guess"])
                truth = sla.spsolve(A, b)
                if hard_change(k, previous_k):
                    scale = max(float(np.sqrt(np.mean(base * base))), 1e-8)
                    f, normalization = make_correction_features(k, previous_k, b - A @ base, base,
                        solution_scale=scale)
                    target = free_to_grid((truth - base) / scale, 128).astype(np.float32)
                    features.append(f); targets.append(target[None]); coefficients.append(k.astype(np.float32))
                    metadata.append({"seed": seed, "step": step, "event": events[step]["event"],
                        "normalization": normalization,
                        "reference_relative_residual": float(np.linalg.norm(b - A @ truth) / np.linalg.norm(b))})
                history.append(truth)
                history = history[-4:]
                previous_k = k
            print(json.dumps({"stage": "data", "group": group, "seed": seed,
                "samples": len(features)}), flush=True)
        data[f"{group}_features"] = np.stack(features)
        data[f"{group}_targets"] = np.stack(targets)
        data[f"{group}_coefficients"] = np.stack(coefficients)
        meta["sample_metadata"][group] = metadata
    np.savez_compressed(dest, **data)
    meta["generation_seconds"] = time.perf_counter() - start
    meta["dataset_sha256"] = digest(dest)
    save(dest.with_suffix(".manifest.json"), meta)


def train(name, loss_kind="energy", initialize=None, epochs=40):
    torch.manual_seed(7)
    np.random.seed(7)
    if not torch.cuda.is_available():
        raise RuntimeError("The authorized training configuration requires the local CUDA GPU")
    device = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = np.load(ROOT / "data/stage3_correction.npz")
    data_hash = digest(ROOT / "data/stage3_correction.npz")
    tensor = {key: torch.from_numpy(data[key]).to(device) for key in data.files}
    model = CorrectionUNet(width=8).to(device)
    if initialize:
        old = torch.load(ROOT / initialize, map_location="cpu", weights_only=False)
        model.load_state_dict(old["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if not initialize else 3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    checkpoint = ROOT / "checkpoints" / f"{name}.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    best, history = float("inf"), []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tensor["train_features"]), device=device)
        sums, count = np.zeros(3), 0
        for indices in order.split(16):
            f, target, k = (tensor[f"train_{suffix}"][indices] for suffix in ("features", "targets", "coefficients"))
            pred = model(f)
            energy = relative_energy_loss(pred, target, k)
            residual = weighted_residual_loss(pred, target, k)
            loss = energy + (0.1 * residual if loss_kind == "mixed" else 0.)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            sums += np.array([loss.item(), energy.item(), residual.item()]) * len(indices)
            count += len(indices)
        scheduler.step()
        model.eval()
        vals = []
        with torch.inference_mode():
            for begin in range(0, len(tensor["validation_features"]), 16):
                sl = slice(begin, begin + 16)
                f, target, k = (tensor[f"validation_{suffix}"][sl] for suffix in ("features", "targets", "coefficients"))
                pred = model(f)
                e = relative_energy_loss(pred, target, k, reduction="none")
                r = weighted_residual_loss(pred, target, k, reduction="none")
                vals.append(torch.stack((e, r), dim=1).cpu().numpy())
        validation = np.concatenate(vals).mean(axis=0)
        selection_loss = float(validation[0] + (0.1 * validation[1] if loss_kind == "mixed" else 0.))
        epoch_row = {"epoch": epoch + 1, "train_loss": float(sums[0] / count),
            "train_energy": float(sums[1] / count), "train_residual": float(sums[2] / count),
            "validation_energy": float(validation[0]), "validation_residual": float(validation[1]),
            "selection_loss": selection_loss}
        history.append(epoch_row)
        if selection_loss < best:
            best = selection_loss
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "width": 8, "seed": 7, "epoch": epoch + 1, "loss_kind": loss_kind,
                "selection_loss": best, "initialize": initialize}, checkpoint)
        save(checkpoint.with_suffix(".training.json"), {"configuration": {"epochs": epochs,
            "loss_kind": loss_kind, "initialize": initialize, "seed": 7, "width": 8,
            "batch_size": 16, "device": device, "gpu": torch.cuda.get_device_name(0),
            "optimizer": "AdamW", "initial_lr": 1e-3 if not initialize else 3e-4,
            "checkpoint_rule": "minimum validation objective of this training stage",
            "dataset_sha256": data_hash},
            "history": history, "best_loss": best, "elapsed_seconds": time.perf_counter() - start,
            "parameter_count": sum(p.numel() for p in model.parameters())})
        print(json.dumps({"stage": "train", "name": name, **epoch_row}), flush=True)
    print(json.dumps({"stage": "trained", "checkpoint": str(checkpoint),
        "seconds": time.perf_counter() - start, "best_loss": best}), flush=True)


def base_specs(n):
    chosen = read(ROOT / "results/stage3_screen/selection.json")["selected_by_grid"][str(n)]
    base = {**chosen, "name": "selected_classical"}
    return [{"name": "fresh_previous", "mode": "fresh", "guess": "previous", "correction": "none"},
        base, {**base, "name": "selected_jacobi3", "correction": "jacobi"},
        {**base, "name": "selected_coarse32", "correction": "coarse"}]


def network_spec(n, checkpoint, device, name, smooth=0):
    base = base_specs(n)[1]
    return {**base, "name": name, "correction": "network", "checkpoint": checkpoint,
        "device": device, "smooth_prediction": smooth}


def calibrate(phase):
    specs = base_specs(128)
    if phase == "initial":
        variants = [("energy", "checkpoints/correction_energy.pt", 0)]
    else:
        variants = [("energy", "checkpoints/correction_energy.pt", 0),
            ("continued", "checkpoints/correction_energy_continued.pt", 0),
            ("mixed", "checkpoints/correction_mixed.pt", 0),
            ("mixed_smooth3", "checkpoints/correction_mixed.pt", 3)]
    for label, checkpoint, smooth in variants:
        for device in ("cpu", "cuda"):
            specs.append(network_spec(128, checkpoint, device, f"{label}_{device}", smooth))
    out = benchmark(f"stage3_calibration_{phase}", [128], splits()["groups"]["calibration"],
        {"128": specs}, 2, "development calibration; never a held-out test")
    _, summary = aggregate_rows(read(out / "measurements.json"))
    save(out / "summary.json", summary)
    if phase == "optimized":
        eligible = {name: r for name, r in summary["128"].items()
            if r["all_accurate"] and not r["pcg_failures"]}
        def valid_specs(predicate):
            candidates = [s for s in specs if predicate(s) and s["name"] in eligible]
            if not candidates:
                save(out / "selection_failed.json", {"reason": "No valid method in a required category"})
                raise RuntimeError("No accurate PCG-successful candidate in a required category")
            return candidates
        klass = min(valid_specs(lambda s: s["correction"] != "network"),
            key=lambda s: eligible[s["name"]]["median_seconds"])
        original = min(valid_specs(lambda s: s["name"].startswith("energy_")),
            key=lambda s: eligible[s["name"]]["median_seconds"])
        continued = min(valid_specs(lambda s: s["name"].startswith("continued_")),
            key=lambda s: eligible[s["name"]]["median_seconds"])
        optimized = min(valid_specs(lambda s: s["name"].startswith("mixed")),
            key=lambda s: eligible[s["name"]]["median_seconds"])
        save(out / "frozen_selection.json", {"classical": klass, "initial_network": original,
            "budget_control_network": continued,
            "optimized_network": optimized, "rule": "Minimum full-sequence median on calibration only; optimization candidate restricted to mixed loss variants, same seed and pretrained energy weights.",
            "all_specs": specs, "checkpoint_sha256": {s["checkpoint"]: digest(ROOT/s["checkpoint"])
                for s in specs if s["correction"] == "network"}})


def final_benchmark():
    selected = read(ROOT / "results/stage3_calibration_optimized/frozen_selection.json")
    for checkpoint, expected in selected["checkpoint_sha256"].items():
        if digest(ROOT / checkpoint) != expected:
            raise RuntimeError(f"Frozen checkpoint changed: {checkpoint}")
    specs_by_n = {}
    for n in (128, 256):
        # The AMG/history pair for each grid was fixed by the screen, while
        # correction/device choices were fixed at 128 by calibration.
        base = base_specs(n)[1]
        klass = ({**selected["classical"], "name": "best_classical"}
            if n == 128 or selected["classical"]["name"] == "fresh_previous" else
            {**base, "name": "best_classical", "correction": selected["classical"]["correction"]})
        initial = {**base, **{k: selected["initial_network"][k] for k in
            ("correction", "checkpoint", "device", "smooth_prediction")}, "name": "initial_network"}
        improved = {**base, **{k: selected["optimized_network"][k] for k in
            ("correction", "checkpoint", "device", "smooth_prediction")}, "name": "optimized_network"}
        continued = {**base, **{k: selected["budget_control_network"][k] for k in
            ("correction", "checkpoint", "device", "smooth_prediction")}, "name": "budget_control_network"}
        specs_by_n[str(n)] = [base_specs(n)[0], base, klass, initial, continued, improved]
    test = splits()["groups"]["test"]
    out128 = benchmark("stage3_test128", [128], test, {"128": specs_by_n["128"]}, 3,
        "held-out geometry test after frozen selection")
    out256 = benchmark("stage3_test256", [256], test[:3], {"256": specs_by_n["256"]}, 3,
        "resolution transfer on a subset of the same held-out geometries; not additional independent geometries")
    save(ROOT / "results/stage3_final_complete.json", {"test128": str(out128.relative_to(ROOT)),
        "test256": str(out256.relative_to(ROOT)), "selection_sha256":
        digest(ROOT / "results/stage3_calibration_optimized/frozen_selection.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen", "data", "train_energy", "calibrate_initial",
        "train_energy_control", "train_mixed", "calibrate_optimized", "test"])
    args = parser.parse_args()
    splits()
    torch.set_num_threads(2)
    with threadpool_limits(limits=1, user_api="blas"):
        if args.stage == "screen": screen()
        elif args.stage == "data": build_data()
        elif args.stage == "train_energy": train("correction_energy")
        elif args.stage == "calibrate_initial": calibrate("initial")
        elif args.stage == "train_energy_control": train("correction_energy_continued", "energy", "checkpoints/correction_energy.pt")
        elif args.stage == "train_mixed": train("correction_mixed", "mixed", "checkpoints/correction_energy.pt")
        elif args.stage == "calibrate_optimized": calibrate("optimized")
        elif args.stage == "test": final_benchmark()


if __name__ == "__main__":
    main()
