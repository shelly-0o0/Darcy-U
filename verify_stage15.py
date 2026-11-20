"""Independent integrity checks for the frozen Stage 15A numerical audit.

This adds checks without editing the experiment, its report, or its thresholds.
Negative controls mutate deep copies or temporary files only.  Moving raw JSON
to a temporary directory tests path portability, not cross-platform numerics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

import report_stage15


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def canonical(value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=convert)


def check_file_manifest(directory, completion_name="complete.json"):
    directory = Path(directory)
    completion = json.loads((directory/completion_name).read_text())
    if completion.get("complete") is not True:
        raise ValueError("Incomplete artifact: " + str(directory))
    for name, expected in completion["file_sha256"].items():
        if sha(directory/name) != expected:
            raise ValueError("File digest mismatch: " + name)
    return completion


def supplemental_checks(freeze, rows, quadrature):
    """Checks deliberately absent from the frozen report's original contract."""
    checks = []

    def record(name, passed, observed=None):
        entry = {"name": name, "passed": bool(passed)}
        if observed is not None:
            entry["observed"] = observed
        checks.append(entry)

    config = freeze["config"]
    frozen_cases = {case["name"]: case for case in freeze["cases"]}
    record("unique_frozen_cases", len(frozen_cases) == len(freeze["cases"]))
    amg_count = 0
    adaptive_count = 0
    selection_count = 0
    for row in rows:
        prefix = f'{row["case"]}/{row["kind"]}/{row["key"]}'
        frozen = frozen_cases[row["case"]]
        scale = frozen["metadata"]["scale_from_input"]
        record(prefix+"/frozen_scale", np.isfinite(row["scale_from_input"]) and
               row["scale_from_input"] == scale and scale > 0.0)
        record(prefix+"/frozen_analytic_goals", np.array_equal(row["exact_qoi"], frozen["exact_qoi"]))
        for degree in ("p1", "p2"):
            truth = np.array(frozen["exact_qoi"]) - np.array(row[degree]["qoi"])
            q = float(np.max(abs(truth)) / (config["tau_scale"] * scale))
            record(prefix+"/"+degree+"/recomputed_q", np.isclose(q, row[degree]["q_ref"],
                   rtol=2e-14, atol=0.0))
        expected_degrees = [1, 2] if (row["kind"] == "uniform" and
                                              row["key"] == config["amg_n"]) else []
        amg = row.get("amg", [])
        record(prefix+"/scheduled_amg_degrees", sorted(x["degree"] for x in amg) == expected_degrees)
        amg_count += len(amg)
        if row["kind"] != "adaptive":
            continue
        adaptive_count += 1
        p = np.asarray(row["mesh_coordinates"], dtype=np.float64)
        t = np.asarray(row["mesh_connectivity"], dtype=np.int32)
        local = np.asarray(row["strong_element_contributions"], dtype=np.float64)
        shape_ok = p.ndim == 2 and p.shape[0] == 2 and t.shape == (3, row["elements"]) and \
            local.shape == (row["elements"], 3)
        record(prefix+"/payload_shapes", shape_ok)
        if not shape_ok:
            continue
        record(prefix+"/coordinate_hash", array_sha(p, np.float64) == row["mesh_coordinates_sha256"])
        record(prefix+"/connectivity_hash", array_sha(t, np.int32) == row["mesh_connectivity_sha256"])
        record(prefix+"/strong_local_hash", array_sha(local, np.float64) ==
               row["dwr"]["strong"]["local_contributions_sha256"])
        record(prefix+"/connectivity_indices", np.all((t >= 0) & (t < p.shape[1])))
        vertices = p[:, t]
        edge_a, edge_b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
        areas = abs(edge_a[0]*edge_b[1]-edge_a[1]*edge_b[0]) / 2.0
        crossings = sum(int(np.sum((vertices[0].min(axis=0) < interface-1e-13) &
                                   (vertices[0].max(axis=0) > interface+1e-13)))
                        for interface in frozen["metadata"]["interfaces"])
        all_edges = np.concatenate((t[[0, 1]].T, t[[1, 2]].T, t[[2, 0]].T))
        edges, multiplicities = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
        boundary_nodes = np.unique(edges[multiplicities == 1])
        boundary = p[:, boundary_nodes]
        on_square = np.any((abs(boundary) < 1e-13) | (abs(boundary-1.0) < 1e-13), axis=0)
        geometry_ok = np.all(np.isfinite(p)) and np.all(areas > 0) and \
            abs(float(areas.sum())-1.0) < 1e-12 and crossings == 0 and \
            np.all(on_square) and np.all((p >= -1e-13) & (p <= 1+1e-13)) and \
            np.all((multiplicities == 1) | (multiplicities == 2))
        record(prefix+"/recomputed_geometry", geometry_ok)
        recorded = row["mesh_check"]
        record(prefix+"/stored_geometry", bool(recorded["accepted"]) == bool(geometry_ok) and
               recorded["material_crossing_elements"] == crossings and
               recorded["square_boundary"] == bool(np.all(on_square)) and
               abs(recorded["area"]-float(areas.sum())) < 1e-14 and
               abs(recorded["min_area"]-float(areas.min())) < 1e-14)
        record(prefix+"/recomputed_local_sum", np.allclose(local.sum(axis=0),
               row["dwr"]["strong"]["global_estimates"], rtol=0.0, atol=1e-10*scale))
        if row["key"] == 0:
            initial = next(r for r in rows if r["case"] == row["case"] and r["kind"] == "uniform"
                           and r["key"] == config["adaptive_n"])
            record(prefix+"/initial_mesh_matches_uniform", all(row[key] == initial[key] for key in
                   ("mesh_coordinates_sha256", "mesh_connectivity_sha256")))
        if "marking" in row:
            selection_count += 1
            mark = row["marking"]
            selected = np.asarray(mark["selected_ids"], dtype=np.int64)
            scores = np.sum(abs(local), axis=1)
            record(prefix+"/selection_hash", array_sha(selected, np.int64) == mark["selected_sha256"])
            record(prefix+"/selection_count", len(selected) == mark["selected_elements"])
            record(prefix+"/captured_fraction", np.isclose(float(scores[selected].sum()/scores.sum()),
                   mark["captured_fraction"], rtol=2e-14, atol=0.0))
    record("scheduled_amg_total", amg_count == 2*len(frozen_cases), amg_count)
    record("adaptive_payload_total", adaptive_count == len(frozen_cases)*(config["adaptive_refinements"]+1),
           adaptive_count)
    record("selection_payload_total", selection_count == len(frozen_cases)*config["adaptive_refinements"],
           selection_count)
    for case_name in frozen_cases:
        matches = [q for q in quadrature if q["case"] == case_name]
        record(case_name+"/unique_quadrature_check", len(matches) == 1)
        if len(matches) != 1:
            continue
        q = matches[0]
        finest = next(r for r in rows if r["case"] == case_name and r["kind"] == "uniform"
                      and r["key"] == max(config["uniform_n"]))
        record(case_name+"/quadrature_schedule", q["n"] == max(config["uniform_n"]) and
               q["low_order"] == config["intorder"] and q["high_order"] == config["quadrature_check_order"])
        record(case_name+"/quadrature_low_matches_finest", np.array_equal(q["qoi_low"], finest["p2"]["qoi"]))
    return checks


def negative_controls(freeze, rows, quadrature, done, directory, expected_summary):
    controls = []
    for name in ("delete_amg", "change_row_scale", "change_quadrature_low_qoi"):
        changed_rows, changed_q = copy.deepcopy(rows), copy.deepcopy(quadrature)
        if name == "delete_amg":
            next(row for row in changed_rows if row.get("amg"))["amg"].pop()
        elif name == "change_row_scale":
            changed_rows[0]["scale_from_input"] *= 2.0
        else:
            changed_q[0]["qoi_low"][0] += 1e-3
        failed = [item["name"] for item in supplemental_checks(freeze, changed_rows, changed_q)
                  if not item["passed"]]
        controls.append(dict(name=name, passed=bool(failed), rejected_by=failed))

    changed_rows = copy.deepcopy(rows)
    failed_solver = changed_rows[0]["p1"]["solver"]
    failed_solver.update(relative_residual=1e-3, relative_residual_double=1e-3,
                         componentwise_backward_error=1e-6, normwise_backward_error=1e-6,
                         accepted=False, converged=False)
    failed_solver["per_rhs"][0].update(relative_residual=1e-3, relative_residual_double=1e-3,
                                     componentwise_backward_error=1e-6, normwise_backward_error=1e-6)
    # Keep the redundant diagnostic views consistent in this synthetic failure.
    for view in [failed_solver["initial"], *failed_solver["residual_history"]]:
        for field in ("relative_residual", "relative_residual_double", "componentwise_backward_error",
                      "normwise_backward_error", "accepted", "per_rhs"):
            view[field] = copy.deepcopy(failed_solver[field])
    summary = report_stage15.summarize(freeze, changed_rows, quadrature, done)
    retained = summary["uniform_contexts"] + summary["adaptive_contexts"] == len(rows)
    controls.append(dict(name="solver_failure_retained", passed=not summary["numerical_audit_pass"] and
                         retained and bool(summary["failed_checks"]), rows_retained=retained,
                         failed_checks=[item["name"] for item in summary["failed_checks"]]))

    with tempfile.TemporaryDirectory(prefix="stage15_integrity_") as temporary:
        copied = Path(temporary)/"relocated_raw"
        copied.mkdir()
        for name in ("freeze.json", "rows.json", "quadrature.json", "complete.json"):
            shutil.copy2(Path(directory)/name, copied/name)
        relocated = report_stage15.read_checked(copied)
        migrated_summary = report_stage15.summarize(*relocated)
        controls.append(dict(name="filesystem_relocation", passed=canonical(migrated_summary) ==
                             canonical(expected_summary), scope="same runtime, raw JSON moved to /tmp"))
        with (copied/"rows.json").open("a") as stream:
            stream.write("\n")
        rejected = False
        try:
            check_file_manifest(copied)
        except ValueError as exc:
            rejected = "digest mismatch" in str(exc)
        controls.append(dict(name="raw_file_content_hash_rejection", passed=rejected,
                             mutation="One newline appended to temporary rows.json only"))
    return controls


def run(directory, report_directory, output):
    directory, report_directory, output = map(Path, (directory, report_directory, output))
    if output.exists():
        raise FileExistsError("Verification output already exists; choose a new file")
    raw_completion = check_file_manifest(directory)
    report_completion = check_file_manifest(report_directory)
    if report_completion["source_complete_sha256"] != sha(directory/"complete.json"):
        raise ValueError("Report belongs to a different raw completion manifest")
    freeze, rows, quadrature, done = report_stage15.read_checked(directory)
    checks = supplemental_checks(freeze, rows, quadrature)
    recomputed = report_stage15.summarize(freeze, rows, quadrature, done)
    saved = json.loads((report_directory/"summary.json").read_text())
    matches = canonical(recomputed) == canonical(saved)
    checks.append(dict(name="all_saved_summary_fields_recomputed", passed=matches))
    controls = negative_controls(freeze, rows, quadrature, done, directory, recomputed)
    result = dict(accepted=all(x["passed"] for x in checks+controls),
                  raw_numerical_audit_pass=recomputed["numerical_audit_pass"],
                  scope="Integrity and reproducibility; no new numerical or publication claim",
                  raw_directory=str(directory), report_directory=str(report_directory),
                  verifier_sha256=sha(__file__), raw_complete_sha256=sha(directory/"complete.json"),
                  report_complete_sha256=sha(report_directory/"complete.json"),
                  summary_sha256=sha(report_directory/"summary.json"),
                  checks=checks, failed_checks=[x for x in checks if not x["passed"]],
                  negative_controls=controls, recomputed_counts={name:recomputed[name] for name in
                    ("uniform_contexts", "adaptive_contexts", "adaptive_decisions_rebuilt", "linear_solver_calls",
                     "rhs_solutions", "amg_calls", "quadrature_checks")},
                  raw_rows=raw_completion["rows"], frozen_files_modified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name:result[name] for name in ("accepted", "raw_numerical_audit_pass",
                     "recomputed_counts", "negative_controls", "failed_checks")}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/stage15_numerical")
    parser.add_argument("--report", default="results/stage15_report")
    parser.add_argument("--output", default="results/stage15_verification.json")
    args = parser.parse_args()
    outcome = run(ROOT/args.input, ROOT/args.report, ROOT/args.output)
    raise SystemExit(0 if outcome["accepted"] else 1)
