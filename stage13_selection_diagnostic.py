"""Posthoc interface-coverage description of frozen stage-13 selections.

This reads saved outcomes and rebuilds only coefficient samples. It never
assembles a PDE operator, solves a PDE, chooses a new mesh, or changes a gate.
The interface mask detects a binary material interface passing through a
coarse cell's four fine coefficient samples; interfaces lying exclusively
on a coarse-cell edge are not included.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from stage10_experiment import coefficient


ROOT = Path(__file__).resolve().parent
METHODS = ("coefficient_local", "coarse_goal_local", "jacobi8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Use a fresh diagnostic output; do not overwrite results")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def interior_interface_mask(k):
    """Threshold 2 separates both coefficient phases for the frozen generator."""
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2:
        raise ValueError("Expected an even square coefficient grid")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("Coefficient samples must be finite and positive")
    nc = k.shape[0] // 2
    high = k > 2.
    blocks = high.reshape(nc, 2, nc, 2).transpose(0, 2, 1, 3)
    counts = blocks.sum(axis=(2, 3))
    return (counts > 0) & (counts < 4)


def summarize(values):
    finite = [float(v) for v in values if v is not None]
    return dict(count=len(values), nonnull_count=len(finite),
                median=None if not finite else float(np.median(finite)),
                minimum=None if not finite else min(finite),
                maximum=None if not finite else max(finite))


def analyze(source):
    source = Path(source)
    completion = json.loads((source / "complete.json").read_text())
    freeze = json.loads((source / "freeze.json").read_text())
    if not completion["complete"] or completion["smoke"]:
        raise RuntimeError("Diagnostic requires completed formal stage-13 data")
    if sha(source / "freeze.json") != completion["freeze_sha256"]:
        raise RuntimeError("Frozen metadata hash mismatch")
    for name in ("rows.json", "contexts.json"):
        if sha(source / name) != completion["output_sha256"][name]:
            raise RuntimeError(f"Frozen formal output changed: {name}")
    generator = ROOT / "stage10_experiment.py"
    if sha(generator) != freeze["input_sha256"]["stage10_experiment.py"]:
        raise RuntimeError("Coefficient generator changed after formal freezing")
    contexts = [v for v in json.loads((source / "contexts.json").read_text()) if v["nc"] == 64]
    rows = json.loads((source / "rows.json").read_text())
    lookup = {(v["split"], v["geometry"], v["method"]): v for v in rows if v["nc"] == 64}
    if len(contexts) != 32 or len({(v["split"], v["geometry"]) for v in contexts}) != 32:
        raise RuntimeError("Expected 32 distinct nc=64 contexts")
    records = []
    geometries = []
    for context in sorted(contexts, key=lambda v: (v["split"], v["geometry"])):
        split, geometry = context["split"], context["geometry"]
        k = coefficient(geometry, 128, split == "contrast")
        digest = hashlib.sha256(k.tobytes()).hexdigest()
        if digest != context["coefficient_sha256"]:
            raise RuntimeError(f"Coefficient reconstruction mismatch for {geometry}")
        # The low phase is exp(s), |s|<=.25; the high phase is at least
        # 5*exp(s). Thus 2 is structurally between the two phase ranges.
        high = k > 2.
        if np.any(k[~high] > np.exp(.25) * (1 + 1e-12)) or np.any(
                k[high] < 5 * np.exp(-.25) * (1 - 1e-12)):
            raise RuntimeError("Threshold-separation assumption violated")
        mask = interior_interface_mask(k).ravel()
        interface_cells = np.flatnonzero(mask)
        geometries.append(dict(split=split, geometry=geometry, nc=64,
            coefficient_sha256=digest, interface_cell_count=int(mask.sum()),
            interface_cell_ids=interface_cells.tolist(),
            high_fine_cell_count=int(high.sum())))
        for method in METHODS:
            row = lookup[split, geometry, method]
            cells = np.asarray(row["selected_cells"], dtype=np.int64)
            if len(np.unique(cells)) != len(cells) or np.any(cells < 0) or np.any(cells >= 4096):
                raise RuntimeError("Invalid saved selected-cell indices")
            covered = int(mask[cells].sum())
            scales = np.asarray(row["qoi_scales"])
            denominator = float(np.linalg.norm((np.asarray(row["fine_qoi"]) - row["coarse_qoi"]) / scales))
            error = float(np.linalg.norm((np.asarray(row["fine_qoi"]) - row["qoi"]) / scales))
            e = None if denominator < freeze["config"]["near_zero_coarse_error"] else error / denominator
            if e is None or not np.isclose(e, row["error_reduction_ratio"], rtol=1e-12, atol=1e-15):
                raise RuntimeError("Reconstructed saved error metric is inconsistent")
            records.append(dict(split=split, geometry=geometry, nc=64, method=method,
                coefficient_sha256=digest, selected_cell_count=len(cells),
                interface_cell_count=int(mask.sum()), covered_interface_cell_count=covered,
                missed_interface_cell_count=int(mask.sum()) - covered,
                selected_noninterface_cell_count=len(cells) - covered,
                recall=None if not mask.any() else covered / int(mask.sum()),
                precision=None if not len(cells) else covered / len(cells),
                E=e, added_dofs=row["added_dofs"], solve_dofs=row["solve_dofs"],
                saved_selection_sha256=hashlib.sha256(cells.tobytes()).hexdigest()))
    groups = []
    for split in ("id", "contrast"):
        for method in METHODS:
            group = [v for v in records if v["split"] == split and v["method"] == method]
            groups.append(dict(split=split, method=method, geometries=len(group),
                statistics={name: summarize([v[name] for v in group]) for name in (
                    "interface_cell_count", "selected_cell_count", "covered_interface_cell_count",
                    "missed_interface_cell_count", "selected_noninterface_cell_count",
                    "recall", "precision", "E")},
                full_interior_interface_coverage_count=sum(v["missed_interface_cell_count"] == 0 for v in group)))
    return dict(metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        analysis="posthoc_descriptive_selection_coverage", pre_registered=False,
        changes_formal_gate=False, performs_new_pde_solves=False, trains_or_calls_ai=False,
        formal_source=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        source_sha256={name: sha(source / name) for name in ("freeze.json", "complete.json", "rows.json", "contexts.json")},
        generator_sha256=sha(generator), analysis_script_sha256=sha(__file__),
        nc=64, independent_geometries=32, geometry_method_records=len(records), methods=list(METHODS),
        high_phase_threshold=2.,
        threshold_reason="low phase <= exp(.25) < 2; high phase >= 5*exp(-.25) > 2 in the frozen two-inclusion generator",
        interface_definition="A coarse cell is included iff its 2x2 fine coefficient samples contain both binary phases; interfaces only on coarse-cell edges are excluded.",
        interpretation_limits=[
            "Posthoc description of already frozen selections; no mesh or threshold was optimized using these results.",
            "Recall/precision count coarse cells, not interface lengths, flux errors, or degrees of freedom.",
            "Selections have equal added-DOF budgets but need not contain equal numbers of coarse cells.",
            "High recall and lower E occurring together do not establish causation or a unique mechanism.",
            "Precision need not be high for a good method: noninterface refinement can contribute to the three goals.",
            "This binary mask is specific to the frozen simple two-inclusion coefficient family, not a general a posteriori estimator."]),
        geometries=geometries, records=records, groups=groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/stage13_cost")
    parser.add_argument("--output", default="results/stage13_selection_diagnostic.json")
    args = parser.parse_args()
    result = analyze((ROOT / args.source).resolve())
    output = (ROOT / args.output).resolve()
    dump(output, result)
    for group in result["groups"]:
        print(json.dumps(dict(split=group["split"], method=group["method"],
            recall_median=group["statistics"]["recall"]["median"],
            precision_median=group["statistics"]["precision"]["median"],
            E_median=group["statistics"]["E"]["median"],
            full_coverage=group["full_interior_interface_coverage_count"])))
    print(json.dumps(dict(output=str(output), records=len(result["records"]))))


if __name__ == "__main__":
    main()
