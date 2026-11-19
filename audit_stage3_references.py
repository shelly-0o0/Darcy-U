#!/usr/bin/env python3
"""Recompute stage-3 direct references after every online timing run has ended.

This is a separate offline audit. It never edits algorithm, protocol, checkpoint,
or original run files. Existing runner references.json files contain coefficient
sequence hashes, not solution hashes: matching those hashes establishes identical
inputs, not bitwise equality with the original direct-solver outputs.
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
import warnings

# Set these before importing scientific libraries. The explicit threadpool
# context below also limits libraries in environments that already set them.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_info, threadpool_limits

from fields import coefficient_sequence
from numerics import assemble_system


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGES = ["stage3_screen", "stage3_calibration_initial",
                  "stage3_calibration_optimized", "stage3_test128", "stage3_test256"]
REFERENCE_SOURCES = ["fields.py", "numerics.py"]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_inputs(stage_names):
    """Read and check metadata only; this function never assembles a PDE."""
    if len(set(stage_names)) != len(stage_names):
        raise ValueError("Stage names must be unique")
    stages, pairs = {}, {}
    for name in stage_names:
        if Path(name).name != name:
            raise ValueError("Use stage directory names beneath results/, not paths")
        directory = ROOT / "results" / name
        paths = {key: directory / f"{key}.json" for key in ("manifest", "references", "complete")}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Completed stage input is missing: {path}")
        manifest, references, complete = (read_json(paths[key]) for key in ("manifest", "references", "complete"))
        if complete.get("complete") is not True:
            raise ValueError(f"Stage {name} is not marked complete; wait for online timings to finish")
        configuration = manifest["configuration"]
        grids, seeds = configuration["grids"], configuration["seeds"]
        if len(set(grids)) != len(grids) or len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated grids or seeds in {name}")
        steps, contrast, rtol = int(manifest["steps"]), float(manifest["contrast"]), float(manifest["rtol"])
        if steps != 20 or contrast != 1e4 or not 0 < rtol < 1:
            raise ValueError(f"Unexpected fixed stage-3 problem configuration in {name}")
        source_checks = {source: sha256(ROOT / source) == manifest["source_sha256"][source]
                         for source in REFERENCE_SOURCES}
        if not all(source_checks.values()):
            raise ValueError(f"Reference-generating source changed since {name}: {source_checks}")
        expected = {(int(n), int(seed)) for n in grids for seed in seeds}
        reference_keys = [(int(row["n"]), int(row["seed"])) for row in references]
        if len(set(reference_keys)) != len(reference_keys) or set(reference_keys) != expected:
            raise ValueError(f"references.json does not cover each configured (n,seed) exactly once in {name}")
        stages[name] = {
            "configuration": configuration,
            "steps": steps, "contrast": contrast, "rtol": rtol,
            "input_sha256": {key: sha256(path) for key, path in paths.items()},
            "reference_source_matches": source_checks,
            "original_runner_source_sha256": manifest["source_sha256"].get("sequence_experiment.py"),
            "expected_pairs": len(expected),
        }
        for row in references:
            key = (int(row["n"]), int(row["seed"]))
            specification = (steps, contrast, "mixed", "one")
            if key in pairs and pairs[key]["specification"] != specification:
                raise ValueError(f"Shared (n,seed) has conflicting problem specifications: {key}")
            if key not in pairs:
                pairs[key] = {"specification": specification, "stages": {}}
            expected_hash = row["field_sha256"]
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"Invalid original field hash for {name}/{key}")
            pairs[key]["stages"][name] = {"expected_field_sha256": expected_hash, "rtol": rtol}
    return stages, pairs


def audit_pair(n, seed, specification):
    """Recreate one unique input sequence, then audit all direct solves."""
    steps, contrast, family, rhs = specification["specification"]
    started = time.perf_counter()
    fields, _ = coefficient_sequence(n, steps, seed, contrast, family)
    # This is deliberately identical to sequence_experiment.benchmark.
    field_hash = hashlib.sha256(fields.tobytes()).hexdigest()
    matches = {name: field_hash == value["expected_field_sha256"]
               for name, value in specification["stages"].items()}
    result = {"n": n, "seed": seed, "steps": steps, "contrast": contrast,
              "family": family, "rhs": rhs, "field_sha256": field_hash,
              "stage_field_hash_matches": matches, "source_stages": specification["stages"],
              "records": [], "nonfinite_steps": [], "exception_steps": [],
              "computed_reference_solution_sha256": None}
    if not all(matches.values()):
        result.update(skipped_reason="Coefficient sequence hash differs from an original stage; direct solves skipped",
                      max_explicit_relative_residual=None, elapsed_seconds=time.perf_counter() - started,
                      successful_reference_audit=False)
        return result
    solution_hasher = hashlib.sha256()
    for step, k in enumerate(fields):
        row = {"step": step, "explicit_relative_residual": None,
               "nonfinite": False, "warnings": []}
        solve_start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                matrix, b, _ = assemble_system(k)
                x = np.asarray(sla.spsolve(matrix, b), dtype=np.float64)
                residual = b - matrix @ x
                bnorm = float(np.linalg.norm(b))
                rnorm = float(np.linalg.norm(residual))
            row["warnings"] = [{"category": warning.category.__name__, "message": str(warning.message)}
                               for warning in captured]
            finite = {"matrix": bool(np.all(np.isfinite(matrix.data))),
                      "rhs": bool(np.all(np.isfinite(b))),
                      "solution": bool(np.all(np.isfinite(x))),
                      "residual": bool(np.all(np.isfinite(residual))),
                      "rhs_norm": bool(np.isfinite(bnorm)),
                      "residual_norm": bool(np.isfinite(rnorm))}
            row["finite_checks"] = finite
            row["nonfinite"] = not all(finite.values())
            row["rhs_norm_positive"] = bool(np.isfinite(bnorm) and bnorm > 0.)
            if not row["nonfinite"] and row["rhs_norm_positive"]:
                relative = rnorm / bnorm
                if np.isfinite(relative):
                    row["explicit_relative_residual"] = float(relative)
                else:
                    row["nonfinite"] = True
            solution_hasher.update(np.ascontiguousarray(x).tobytes())
            row["computed_solution_sha256"] = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        except Exception as error:
            row["exception"] = {"type": type(error).__name__, "message": str(error)}
            result["exception_steps"].append(step)
        row["elapsed_seconds"] = time.perf_counter() - solve_start
        if row["nonfinite"]:
            result["nonfinite_steps"].append(step)
        row["passes_stage_tolerances"] = {name: row["explicit_relative_residual"] is not None
            and row["explicit_relative_residual"] <= value["rtol"]
            for name, value in specification["stages"].items()}
        result["records"].append(row)
    values = [row["explicit_relative_residual"] for row in result["records"]
              if row["explicit_relative_residual"] is not None]
    result["max_explicit_relative_residual"] = max(values) if values else None
    if not result["exception_steps"]:
        result["computed_reference_solution_sha256"] = solution_hasher.hexdigest()
    result["successful_reference_audit"] = all(all(row["passes_stage_tolerances"].values())
                                               for row in result["records"])
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES,
                        help="Completed directory names under results/; default is all five stage-3 runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage3_reference_audit.json")
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="Validate JSON/source hashes only; do not generate fields, assemble, solve, or write the audit")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing standalone audit output")
    args = parser.parse_args()
    stages, pairs = inspect_inputs(args.stages)
    if args.check_inputs_only:
        print(json.dumps({"metadata_valid": True, "stages": list(stages),
                          "unique_sequences": len(pairs), "planned_direct_solves": 20 * len(pairs)}, ensure_ascii=False))
        return
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Audit already exists; choose another output or use --overwrite: {output}")
    started = time.perf_counter()
    audit = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "scope": "Separate offline recomputation of direct reference solutions after all online timings; never included in online algorithm cost.",
             "interpretation": "Measured solution-energy errors in the original runs are relative to computed direct references, not certified errors against exact arithmetic solutions.",
             "input_hash_scope": "Runner field_sha256 hashes coefficient-sequence bytes. Original solution hashes were not saved, so recomputed solution hashes cannot establish equality with original solution outputs.",
             "solution_hash_scope": "Computed solution hashes cover float64 C-order bytes in time order. Exact equality across independent runs can depend on SciPy/SuperLU, BLAS versions, platform and thread settings.",
             "configuration": {"stages": args.stages, "blas_threads": 1},
             "source_sha256": {name: sha256(ROOT / name) for name in
                               [*REFERENCE_SOURCES, "audit_stage3_references.py"]},
             "environment": {"python": sys.version, "platform": platform.platform(),
                             "numpy": np.__version__, "scipy": scipy.__version__},
             "stages": stages, "unique_sequences": len(pairs), "sequences": []}
    with threadpool_limits(limits=1, user_api="blas"):
        audit["environment"]["threadpools"] = [{key: pool.get(key) for key in
            ("user_api", "internal_api", "num_threads", "version", "architecture")}
            for pool in threadpool_info()]
        for (n, seed), specification in sorted(pairs.items()):
            row = audit_pair(n, seed, specification)
            audit["sequences"].append(row)
            print(json.dumps({"stage": "reference_audit", "n": n, "seed": seed,
                              "max_relative_residual": row["max_explicit_relative_residual"],
                              "passed": row["successful_reference_audit"]}, ensure_ascii=False), flush=True)
    for name, stage in audit["stages"].items():
        relevant = [row for row in audit["sequences"] if name in row["source_stages"]]
        values = [record["explicit_relative_residual"] for row in relevant for record in row["records"]
                  if record["explicit_relative_residual"] is not None]
        stage["field_hashes_match"] = all(row["stage_field_hash_matches"][name] for row in relevant)
        stage["max_explicit_relative_residual"] = max(values) if values else None
        stage["failed_steps"] = [{"n": row["n"], "seed": row["seed"], "step": record["step"]}
            for row in relevant for record in row["records"] if not record["passes_stage_tolerances"][name]]
        stage["attempted_direct_solves"] = sum(len(row["records"]) for row in relevant)
        stage["finite_residual_steps"] = len(values)
    values = [row["max_explicit_relative_residual"] for row in audit["sequences"]
              if row["max_explicit_relative_residual"] is not None]
    audit["max_explicit_relative_residual"] = max(values) if values else None
    audit["successful_reference_audit"] = all(row["successful_reference_audit"] for row in audit["sequences"])
    audit["nonfinite_step_count"] = sum(len(row["nonfinite_steps"]) for row in audit["sequences"])
    audit["exception_step_count"] = sum(len(row["exception_steps"]) for row in audit["sequences"])
    audit["elapsed_seconds"] = time.perf_counter() - started
    audit["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "successful_reference_audit": audit["successful_reference_audit"],
                      "elapsed_seconds": audit["elapsed_seconds"]}, ensure_ascii=False))
    if not audit["successful_reference_audit"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
