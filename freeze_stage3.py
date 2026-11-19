#!/usr/bin/env python3
"""Verify development provenance and freeze all inputs before held-out runs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    destination = ROOT / "results/stage3_test_freeze.json"
    if destination.exists():
        raise FileExistsError(destination)
    for name in ("stage3_test128", "stage3_test256"):
        if (ROOT / "results" / name).exists():
            raise RuntimeError("Held-out output already exists; freeze must precede testing")
    manifests = {}
    for name in ("stage3_screen", "stage3_calibration_initial", "stage3_calibration_optimized"):
        folder = f"results/{name}"
        read(f"{folder}/complete.json")
        manifest = read(f"{folder}/manifest.json")
        config = manifest["configuration"]
        expected = sum(len(config["specs_by_n"][str(n)]) for n in config["grids"])
        expected *= len(config["seeds"]) * config["repeats"] * manifest["steps"]
        if len(read(f"{folder}/measurements.json")) != expected:
            raise RuntimeError(f"Incomplete development run: {name}")
        for file, expected_hash in manifest["source_sha256"].items():
            if sha(file) != expected_hash:
                raise RuntimeError(f"Changed source: {file}")
        for key, file in (("protocol_sha256", "docs/stage3_protocol_zh.md"),
                          ("splits_sha256", "data/stage3_splits.json")):
            if sha(file) != manifest[key]:
                raise RuntimeError(f"Changed {key} in {name}")
        manifests[name] = sha(f"{folder}/manifest.json")
    selection_path = "results/stage3_calibration_optimized/frozen_selection.json"
    selection = read(selection_path)
    for file, expected_hash in selection["checkpoint_sha256"].items():
        if sha(file) != expected_hash:
            raise RuntimeError(f"Changed model: {file}")
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": manifest["source_sha256"],
        "protocol_sha256": sha("docs/stage3_protocol_zh.md"),
        "splits_sha256": sha("data/stage3_splits.json"),
        "selection_sha256": sha(selection_path),
        "screen_selection_sha256": sha("results/stage3_screen/selection.json"),
        "reporting_rules_sha256": sha("docs/stage3_reporting_rules_zh.md"),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "development_manifest_sha256": manifests,
        "freeze_script_sha256": sha("freeze_stage3.py"),
        "rule": "Frozen before any held-out test; final outcomes cannot select or retrain these candidates."
    }
    destination.write_text(json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
