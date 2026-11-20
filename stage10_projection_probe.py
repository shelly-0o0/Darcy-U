"""Post-hoc, zero-training coarse-complement diagnostic for stage 10.

Frozen experiment files and checkpoints are read only.  This diagnostic cannot
change the formal gate.  All methods receive the same A-orthogonal projection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import median
import time

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch

import stage10_experiment as experiment
from stage10_error_model import ErrorOperator
from sequence_baselines import FixedAMGHierarchy

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/stage10_projection_probe_zh.md"
METHODS = (*experiment.KINDS, "jacobi1", "jacobi8", "amg_v1")
SPLITS = ("id", "contrast", "narrow_load", "resolution")
METRICS = ("matrix_error_relative", "normalized_matrix_rmse",
           "defect_energy_error_mean", "marker_mass_ratio", "indicator_relative_l1")
IDENTITY_TOLERANCE = 1e-8
CPU_REPLAY_TOLERANCE = 1e-5


def read(path):
    return json.loads(Path(path).read_text())


def array_hash(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def row_key(row):
    return row["split"], row["geometry"], row["method"], row["training_seed"]


def quadratic(a, vectors):
    return np.einsum("ij,ij->i", vectors, (a @ vectors.T).T)


def project(case, prediction):
    """Four RHS, existing LU; all projection operations are inside the timer."""
    start = time.perf_counter()
    coarse_rhs = case.P.T @ (case.Af @ prediction.T)
    coefficients = case.cache["coarse"].solve(np.asarray(coarse_rhs))
    coarse = np.asarray(case.P @ coefficients).T
    projected = prediction - coarse
    seconds = time.perf_counter() - start
    return projected, coarse, seconds


def energy_diagnostic(case, target, prediction, projected, coarse):
    true_energy = quadratic(case.Af, target)
    before = quadratic(case.Af, prediction-target)
    after = quadratic(case.Af, projected-target)
    removed = quadratic(case.Af, coarse)
    scale = np.maximum.reduce((abs(before), abs(after), abs(removed), np.full(4, 1e-30)))
    identity = abs(before-after-removed)/scale
    projected_action = case.Af @ projected.T
    orthogonality = np.linalg.norm(case.P.T @ projected_action, axis=0)/np.maximum(
        np.linalg.norm(projected_action, axis=0), 1e-30)
    target_action = case.Af @ target.T
    target_orthogonality = np.linalg.norm(case.P.T @ target_action, axis=0)/np.maximum(
        np.linalg.norm(target_action, axis=0), 1e-30)
    if max(identity) > IDENTITY_TOLERANCE or max(orthogonality) > IDENTITY_TOLERANCE:
        raise RuntimeError("Projection energy identity or coarse orthogonality failed")
    if max(target_orthogonality) > IDENTITY_TOLERANCE:
        raise RuntimeError("Reference is not A-orthogonal to the coarse space")
    return dict(reference_energy=true_energy.tolist(), error_energy_before=before.tolist(),
                error_energy_after=after.tolist(), coarse_energy_removed=removed.tolist(),
                coarse_fraction_of_error=(removed/np.maximum(before, 1e-30)).tolist(),
                coarse_energy_relative_to_reference=(removed/true_energy).tolist(),
                energy_identity_relative_error=identity.tolist(),
                projected_coarse_orthogonality=orthogonality.tolist(),
                target_coarse_orthogonality=target_orthogonality.tolist(),
                coarse_fraction_of_error_mean=float(np.mean(removed/np.maximum(before, 1e-30))))


def self_test():
    """A nontrivial dense SPD example, unrelated to the formal PDE dataset."""
    from types import SimpleNamespace
    rng = np.random.default_rng(7411)
    b = rng.normal(size=(7, 7)); a = sp.csr_matrix(b.T @ b + np.eye(7))
    p = sp.csr_matrix(rng.normal(size=(7, 3))); ac = p.T @ a @ p
    case = SimpleNamespace(Af=a, P=p, cache={"coarse": sla.splu(ac.tocsc())})
    target, _, _ = project(case, rng.normal(size=(4, 7)))
    prediction = rng.normal(size=(4, 7))
    projected, coarse, _ = project(case, prediction)
    diagnostic = energy_diagnostic(case, target, prediction, projected, coarse)
    residual = (a @ target.T).T
    assert np.allclose(residual @ prediction.T, residual @ projected.T, atol=1e-11)
    assert np.all(np.array(diagnostic["error_energy_after"]) <=
                  np.array(diagnostic["error_energy_before"])*(1+1e-12))
    assert np.max(np.linalg.norm((project(case, projected)[0]-projected), axis=1)) < 1e-12
    return dict(passed=True, dimension=7, right_hand_sides=4,
                max_energy_identity_error=max(diagnostic["energy_identity_relative_error"]))


def verify_inputs(source):
    frozen = read(source/"freeze.json"); complete = read(source/"complete.json")
    if frozen["smoke"] or not complete["complete"] or complete["rows"] != 768 or complete["models"] != 9:
        raise RuntimeError("This diagnostic requires the complete formal 768-row run")
    if experiment.sha(source/"freeze.json") != complete["freeze_sha256"]:
        raise RuntimeError("Formal freeze hash mismatch")
    for name, digest in complete["result_sha256"].items():
        if experiment.sha(source/name) != digest:
            raise RuntimeError(f"Formal result changed: {name}")
    records = [read(source/f"train_{kind}_{seed}.json")
               for kind in experiment.KINDS for seed in experiment.SEEDS]
    experiment.assert_frozen(frozen, records)
    trained = {(r["kind"], r["seed"]): r["sha256"]
               for r in read(source/"training_complete.json")["checkpoints"]}
    for record in records:
        if trained[(record["kind"], record["seed"])] != record["checkpoint_sha256"]:
            raise RuntimeError("Training completion and checkpoint records disagree")
    evaluation = read(source/"evaluation.json")
    formal = {row_key(r): r for r in evaluation}
    expected = {(split, geometry, method, seed)
                for split in SPLITS
                for geometry in range(50300 if split == "contrast" else 50200,
                                      (50300 if split == "contrast" else 50200)+16)
                for method in METHODS
                for seed in (experiment.SEEDS if method in experiment.KINDS else (None,))}
    if set(formal) != expected or len(evaluation) != len(expected):
        raise RuntimeError("Formal row keys are not exactly the expected 768 keys")
    contexts = read(source/"test_data.json")
    if {(x["split"], x["seed"]) for x in contexts} != {(x[0], x[1]) for x in expected} or len(contexts) != 64:
        raise RuntimeError("Formal test manifest is incomplete or duplicated")
    return frozen, complete, records, formal, contexts


def load_models(records):
    start = time.perf_counter(); models = {}
    for record in records:
        payload = torch.load(ROOT/record["checkpoint"], map_location="cpu", weights_only=False)
        model = ErrorOperator(record["kind"], width=payload["config"]["width"],
                              levels=payload["config"]["levels"])
        model.load_state_dict(payload["state_dict"]); model.eval()
        models[(record["kind"], record["seed"])] = model
    return models, time.perf_counter()-start


def classical_predictions(item):
    a = item["case"].Af; r = item["r"]; d = a.diagonal()
    times = {}; predictions = {}
    start = time.perf_counter(); predictions["jacobi1"] = r/d
    times["jacobi1"] = time.perf_counter()-start
    start = time.perf_counter(); value = np.zeros_like(r)
    for _ in range(8):
        value += (2./3)*(r-(a @ value.T).T)/d
    predictions["jacobi8"] = value; times["jacobi8"] = time.perf_counter()-start
    start = time.perf_counter(); hierarchy = FixedAMGHierarchy(a, seed=51700)
    setup_seconds = time.perf_counter()-start
    start = time.perf_counter()
    predictions["amg_v1"] = np.stack([hierarchy.preconditioner @ v for v in r])
    times["amg_v1"] = time.perf_counter()-start
    return predictions, times, setup_seconds


def make_probe_row(item, prediction, formal, prediction_seconds):
    before = experiment.metrics(item, prediction)
    formal_matrix = np.array(formal["estimated_matrix"])
    reference = item["reference"]
    replay = np.linalg.norm(np.array(before["estimated_matrix"])-formal_matrix)/np.linalg.norm(reference)
    reference_difference = np.linalg.norm(reference-np.array(formal["reference_matrix"]))/np.linalg.norm(reference)
    if replay > CPU_REPLAY_TOLERANCE or reference_difference > 1e-10:
        raise RuntimeError("CPU replay or regenerated reference differs from the formal experiment")
    projected, coarse, seconds = project(item["case"], prediction)
    after = experiment.metrics(item, projected)
    gram_before = np.array(before["estimated_matrix"]); gram_after = np.array(after["estimated_matrix"])
    difference = np.linalg.norm(gram_before-gram_after)
    invariant_reference = difference/np.linalg.norm(reference)
    invariant_before = difference/max(np.linalg.norm(gram_before), 1e-30)
    if max(invariant_reference, invariant_before) > IDENTITY_TOLERANCE:
        raise RuntimeError("Projection changed the global Gram matrix beyond tolerance")
    diagnostic = energy_diagnostic(item["case"], item["target"], prediction, projected, coarse)
    row = {k: formal[k] for k in ("split", "geometry", "method", "training_seed")}
    row.update(before=before, after=after, formal_metrics={k: formal[k] for k in METRICS},
               cpu_replay_matrix_difference=replay, reference_matrix_difference=reference_difference,
               gram_invariance_relative_to_reference=invariant_reference,
               gram_invariance_relative_to_before=invariant_before,
               matrix_error_absolute_change=abs(before["matrix_error_relative"]-after["matrix_error_relative"]),
               energy=diagnostic, prediction_seconds=prediction_seconds, projection_seconds=seconds,
               arrays_sha256={"prediction": array_hash(prediction), "projected": array_hash(projected),
                              "removed_coarse": array_hash(coarse)})
    return row


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["method"], row["geometry"])].append(row)
    geometries = []
    for (split, method, geometry), records in sorted(grouped.items()):
        entry = dict(split=split, method=method, geometry=geometry, seeds=len(records))
        for state in ("before", "after"):
            entry[state] = {m: median(r[state][m] for r in records) for m in METRICS}
        entry["coarse_fraction_of_error_mean"] = median(r["energy"]["coarse_fraction_of_error_mean"] for r in records)
        entry["projection_seconds"] = median(r["projection_seconds"] for r in records)
        geometries.append(entry)
    summary = {}
    lookup = {(r["split"], r["method"], r["geometry"]): r for r in geometries}
    for split in SPLITS:
        summary[split] = {}
        for method in METHODS:
            rr = [r for r in geometries if r["split"] == split and r["method"] == method]
            result = {state: {m: median(r[state][m] for r in rr) for m in METRICS}
                      for state in ("before", "after")}
            result["coarse_fraction_of_error_mean"] = median(r["coarse_fraction_of_error_mean"] for r in rr)
            result["paired_after_over_before"] = {m: median(r["after"][m]/r["before"][m] for r in rr) for m in METRICS}
            result["paired_marker_mass_gain"] = median(r["after"]["marker_mass_ratio"]-r["before"]["marker_mass_ratio"] for r in rr)
            result["geometries_marker_improved"] = sum(r["after"]["marker_mass_ratio"] > r["before"]["marker_mass_ratio"] for r in rr)
            result["paired_after_over_projected_amg"] = {
                m: median(r["after"][m]/lookup[(split, "amg_v1", r["geometry"])]["after"][m] for r in rr)
                for m in METRICS}
            result["geometries"] = len(rr)
            summary[split][method] = result
    return dict(aggregation="Within geometry median across training seeds, then across geometries; paired ratios use those geometry summaries",
                by_split_method=summary, geometry_summaries=geometries)


def run(source, output):
    if (output/"manifest.json").exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    frozen, complete, records, formal, contexts = verify_inputs(source)
    input_files = [source/name for name in ("freeze.json", "complete.json", "evaluation.json", "test_data.json",
                                           "training_data.json", "training_complete.json")]
    input_files += [source/f"train_{kind}_{seed}.json" for kind in experiment.KINDS for seed in experiment.SEEDS]
    input_files += [ROOT/record["checkpoint"] for record in records]
    input_files += [Path(__file__), PROTOCOL]
    input_hashes = {str(path.resolve()): experiment.sha(path) for path in input_files}
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = dict(post_hoc=True, zero_training=True, formal_gate_unchanged=True,
                    created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=input_hashes,
                    formal_freeze_sha256=complete["freeze_sha256"], formal_source_sha256=frozen["source_sha256"],
                    formal_protocol_sha256=frozen["protocol_sha256"],
                    identity_tolerance=IDENTITY_TOLERANCE, cpu_replay_tolerance=CPU_REPLAY_TOLERANCE,
                    expected_rows=768, expected_contexts=64, independent_coefficient_geometries=32,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     torch=torch.__version__, device="cpu", torch_threads=2),
                    timing_scope="Diagnostic only; projection includes A multiply, restriction, cached coarse LU solve, prolongation and subtraction; shared factorization/data generation, checkpoint loading, inference and metric evaluation are separate; no acceleration claim",
                    self_test=self_test())
    experiment.dump(output/"manifest.json", manifest)
    models, model_load_seconds = load_models(records)
    rows = []; case_records = []
    for index, context in enumerate(contexts, 1):
        item = experiment.make_case(context["seed"], context["nc"], context["ood"], context["narrow"])
        if item["diagnostics"]["coefficient_sha256"] != context["coefficient_sha256"]:
            raise RuntimeError("Regenerated coefficient mismatch")
        case = item["case"]
        if case.cache_statistics["coarse_factorizations"] != 1 or case.cache_statistics["fine_factorizations"] != 1:
            raise RuntimeError("Expected one reused fine/coarse factorization per context")
        checkpoint = dict(case.cache_statistics)
        predicted, prediction_times, amg_setup_seconds = classical_predictions(item)
        batch = experiment.stacked([item], torch.device("cpu"))
        for method in METHODS:
            for seed in (experiment.SEEDS if method in experiment.KINDS else (None,)):
                if seed is None:
                    value = predicted[method]; prediction_seconds = prediction_times[method]
                else:
                    inference_start = time.perf_counter()
                    with torch.no_grad():
                        value = experiment.prediction(models[(method, seed)], batch, case.nf).numpy().reshape(4, -1).astype(np.float64)
                    prediction_seconds = time.perf_counter()-inference_start
                key = (context["split"], context["seed"], method, seed)
                rows.append(make_probe_row(item, value, formal[key], prediction_seconds))
        if case.cache_statistics != checkpoint:
            raise RuntimeError("Projection unexpectedly changed the reference solve cache counters")
        case_records.append(dict(split=context["split"], geometry=context["seed"], nc=context["nc"],
                                 coefficient_sha256=context["coefficient_sha256"],
                                 residual_sha256=array_hash(item["r"]), target_sha256=array_hash(item["target"]),
                                 generation_seconds=item["diagnostics"]["generation_seconds"],
                                 reference_cache_statistics=checkpoint, projection_coarse_solve_batches=12,
                                 projection_right_hand_sides_per_batch=4, amg_setup_seconds=amg_setup_seconds))
        if index % 16 == 0:
            print(json.dumps(dict(event="projection_probe",contexts=index,rows=len(rows))), flush=True)
    if len(rows) != 768 or {row_key(r) for r in rows} != set(formal):
        raise RuntimeError("Diagnostic row completeness failure")
    experiment.assert_frozen(frozen, records)
    for path, digest in input_hashes.items():
        if experiment.sha(path) != digest:
            raise RuntimeError(f"Input changed during diagnostic: {path}")
    summary = aggregate(rows)
    summary.update(post_hoc=True, formal_gate_unchanged=True, rows=len(rows), contexts=len(case_records),
                   model_load_seconds=model_load_seconds,
                   maximum_cpu_replay_difference=max(r["cpu_replay_matrix_difference"] for r in rows),
                   maximum_gram_change=max(max(r["gram_invariance_relative_to_reference"], r["gram_invariance_relative_to_before"]) for r in rows),
                   maximum_energy_identity_error=max(max(r["energy"]["energy_identity_relative_error"]) for r in rows),
                   maximum_projection_coarse_orthogonality=max(max(r["energy"]["projected_coarse_orthogonality"]) for r in rows),
                   summed_projection_seconds=sum(r["projection_seconds"] for r in rows),
                   summed_prediction_seconds=sum(r["prediction_seconds"] for r in rows),
                   summed_case_generation_seconds=sum(c["generation_seconds"] for c in case_records),
                   summed_amg_setup_seconds=sum(c["amg_setup_seconds"] for c in case_records),
                   diagnostic_elapsed_seconds=time.perf_counter()-start)
    experiment.dump(output/"rows.json", rows)
    experiment.dump(output/"contexts.json", case_records)
    experiment.dump(output/"summary.json", summary)
    experiment.dump(output/"complete.json", dict(complete=True, post_hoc=True, formal_gate_unchanged=True,
                    rows=len(rows), contexts=len(case_records), input_sha256=input_hashes,
                    output_sha256={name: experiment.sha(output/name) for name in ("manifest.json", "rows.json", "contexts.json", "summary.json")}))
    print(json.dumps(dict(event="complete",rows=len(rows),contexts=len(case_records),output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/stage10_pilot")
    parser.add_argument("--output", default="results/stage10_projection_probe")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        run((ROOT/args.source).resolve(), (ROOT/args.output).resolve())


if __name__ == "__main__":
    main()
