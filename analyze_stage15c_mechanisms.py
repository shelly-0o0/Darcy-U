"""Post-run descriptive mechanism audit; no PDE solve or threshold changes.

This script independently reconstructs the frozen screening gates from raw
path records.  It never imports the report or numerical solver.  Additional
mechanism descriptors do not change the predeclared retain/stop decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import sys


ROOT = Path(__file__).resolve().parent
NONTERMINAL_ITEMS = ("p2_assembly_seconds", "p2_dual_solve_seconds",
                     "dwr_evaluation_seconds", "dwr_localization_seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def reference_checks(reference, cfg):
    if reference["config"] != cfg:
        raise ValueError("Reference thresholds differ from freeze")
    rows = reference["rows"]
    standard = [row for row in rows if row["degree"] == 2 and row["intorder"] == cfg["intorder"]]
    higher = [row for row in rows if row["degree"] == 2 and row["intorder"] == cfg["high_order"]]
    linear = [row for row in rows if row["degree"] == 1]
    if [row["n"] for row in standard] != cfg["p2_n"] or len(higher) != 1 or len(linear) != 1:
        raise ValueError("Incomplete reference hierarchy")
    if higher[0]["n"] != max(cfg["p2_n"]) or linear[0]["n"] != cfg["p1_cross_n"]:
        raise ValueError("Incorrect reference cross-check resolutions")
    q = [row["qoi"] for row in standard]
    differences = [b-a for a,b in zip(q[:-1],q[1:])]
    tiny = 100*sys.float_info.epsilon*max(1.,max(abs(value) for value in q))
    resolved = [value for value in differences if abs(value) > tiny]
    minimum_tau = cfg["min_tau"]
    checks = dict(all_linear_and_mesh_checks=all(row["accepted"] for row in rows),
                  finest_difference=abs(differences[-1]) <= cfg["finest_difference_fraction"]*minimum_tau,
                  quadrature=abs(higher[0]["qoi"]-q[-1]) <= cfg["quadrature_fraction"]*minimum_tau,
                  independent_order=abs(higher[0]["qoi"]-linear[0]["qoi"]) <= cfg["independent_order_fraction"]*minimum_tau,
                  differences_not_growing=all(abs(b) <= cfg["growth_allowance"]*abs(a)+tiny for a,b in zip(differences[:-1],differences[1:])),
                  stable_difference_sign=all(value > 0. for value in resolved) or all(value < 0. for value in resolved))
    if checks != reference["assessment"]["checks"] or all(checks.values()) != reference["assessment"]["accepted"]:
        raise ValueError("Independent reference assessment mismatch")
    return dict(accepted=all(checks.values()), qoi=higher[0]["qoi"], checks=checks,
                failed_checks=[key for key,value in checks.items() if not value])


def _offline(entry, reference, tau):
    summary = entry["summary"]
    value = summary.get("final_qoi")
    q = None if value is None else abs(value-reference["qoi"])/tau
    completed = summary["status"] in ("estimated_converged", "fixed_mesh_completed")
    success = bool(reference["accepted"] and completed and q is not None and q <= 1.)
    if not reference["accepted"]:
        status = "reference_unqualified"
    elif not completed:
        status = summary["status"]
    elif q is None or q > 1.:
        status = "accuracy_failed"
    else:
        status = "success"
    if not entry["final_mesh_check"]["accepted"]:
        success, status = False, "mesh_failed"
    if entry["offline"]["success"] != success or entry["offline"]["status"] != status:
        raise ValueError("Offline acceptance mismatch")
    saved_q = entry["offline"]["q"]
    if (q is None) != (saved_q is None) or (q is not None and abs(q-saved_q) > 1e-12):
        raise ValueError("Offline q mismatch")
    return success, status, q


def analyze(directory, reported_summary):
    directory = Path(directory)
    completion = load(directory/"complete.json")
    if completion["complete"] is not True:
        raise ValueError("Wait for a completed formal run")
    manifest = completion["file_sha256"]
    for filename in ("freeze.json", "references.json", "index.json"):
        if digest(directory/filename) != manifest[filename]:
            raise ValueError("Raw manifest mismatch: "+filename)
    freeze = load(directory/"freeze.json"); cfg = freeze["config"]; thresholds = cfg["screening"]
    if not thresholds["require_both_taus"] or not thresholds["require_all_methods_success"]:
        raise ValueError("This analysis implements the declared both-tolerance/all-method gate")
    references = {r["case"]:reference_checks(r,cfg["reference"]) for r in load(directory/"references.json")}
    metadata = {case["name"]:case["metadata"] for case in freeze["cases"]}
    index = load(directory/"index.json")
    expected = {(case,tau,method,repeat) for case in metadata for tau in cfg["taus"] for method in cfg["methods"] for repeat in range(cfg["repeats"])}
    keys = [(e["case"],e["tau"],e["method"],e["repeat"]) for e in index]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing or duplicate paths")
    groups = defaultdict(list); raw_paths = {}; status_counts = Counter()
    for entry in index:
        path = directory/entry["path_file"]
        if digest(path) != manifest[entry["path_file"]]:
            raise ValueError("Path hash mismatch: "+entry["path_file"])
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            record = json.load(stream)
        if record["summary"] != entry["summary"]:
            raise ValueError("Summary duplication mismatch")
        rows = record["rows"]; summary = record["summary"]
        success,status,q = _offline(entry,references[entry["case"]],entry["tau"])
        total = summary["algorithm_seconds"]
        middle = sum(sum(row["timings"].get(key,0.) for key in NONTERMINAL_ITEMS)
                     for row in rows if row["status"] == "continue") if entry["method"] == "dwr" else 0.
        first_q = rows[0].get("qoi")
        initial_q = None if first_q is None else abs(first_q-references[entry["case"]]["qoi"])/entry["tau"]
        datum = dict(repeat=entry["repeat"], success=success, status=status, q=q,
                     total_seconds=total, nonterminal_seconds=middle,
                     nonterminal_fraction=middle/total, hypothetical_remaining_seconds=total-middle,
                     states=len(rows), final_dofs=rows[-1].get("p1_free_dofs",0), initial_q=initial_q,
                     initial_stopped=bool(len(rows)==1 and summary["status"]=="estimated_converged"),
                     initial_accuracy_adequate=bool(references[entry["case"]]["accepted"] and initial_q is not None and initial_q <= 1.))
        key = (entry["case"],entry["tau"],entry["method"])
        groups[key].append(datum)
        if entry["method"] in ("dwr","recovered_dwr3","recovered_dwr8"):
            trace = [(row["mesh_coordinates_sha256"],row["mesh_connectivity_sha256"],row.get("qoi"),
                      hashlib.sha256(json.dumps(row["selected_elements"],separators=(",",":")).encode()).hexdigest())
                     for row in rows]
            final = tuple(rows[-1].get(k) for k in ("mesh_coordinates_sha256","mesh_connectivity_sha256","solution_sha256"))
            raw_paths[key+(entry["repeat"],)] = dict(trace=trace,final=final)
        status_counts[status] += 1
    aggregated = {}
    for key, rows in groups.items():
        aggregated[key] = dict(all_success=all(row["success"] for row in rows),
            failed_statuses=dict(Counter(row["status"] for row in rows if not row["success"])),
            seconds=median(row["total_seconds"] for row in rows),
            dofs=median(row["final_dofs"] for row in rows),
            states=median(row["states"] for row in rows),
            fraction=median(row["nonterminal_fraction"] for row in rows),
            nonterminal_seconds=median(row["nonterminal_seconds"] for row in rows),
            remainder=median(row["hypothetical_remaining_seconds"] for row in rows),
            initial_q=median(row["initial_q"] for row in rows) if all(row["initial_q"] is not None for row in rows) else None,
            initial_stopped=all(row["initial_stopped"] for row in rows),
            initial_accuracy_adequate=all(row["initial_accuracy_adequate"] for row in rows))
    settings = []; recoveries = []; gate_failure_counts = Counter(); descriptive_counts = Counter()
    for case in metadata:
        for tau in cfg["taus"]:
            methods = {method:aggregated[(case,tau,method)] for method in cfg["methods"]}
            dwr = methods["dwr"]
            eligible = [method for method in cfg["methods"] if method != "dwr" and methods[method]["all_success"]]
            fastest = min(eligible,key=lambda method:methods[method]["seconds"]) if eligible else None
            numerical = references[case]["accepted"] and all(value["all_success"] for value in methods.values())
            comparable = bool(references[case]["accepted"] and dwr["all_success"] and fastest is not None)
            share = dwr["fraction"] >= thresholds["min_nonterminal_fraction"]
            time_ratio=dof_ratio=ideal_ratio=None; cost=information=False
            if comparable:
                control = methods[fastest]
                time_ratio = dwr["seconds"]/control["seconds"]
                dof_ratio = dwr["dofs"]/control["dofs"]
                ideal_ratio = dwr["remainder"]/control["seconds"]
                cost = bool(share and ideal_ratio <= thresholds["max_hypothetical_vs_control"])
                information = bool(time_ratio <= thresholds["max_dwr_time_vs_control"] or
                                   dof_ratio <= thresholds["max_dwr_dofs_vs_fastest_control"])
            passed = bool(numerical and cost and information)
            reasons = []
            if not references[case]["accepted"]: reasons.append("reference_unqualified")
            if any(not value["all_success"] for value in methods.values()): reasons.append("one_or_more_methods_not_successful")
            if comparable and not share: reasons.append("nonterminal_share_below_frozen_threshold")
            if comparable and ideal_ratio > thresholds["max_hypothetical_vs_control"]: reasons.append("free_middle_work_still_not_enough")
            if comparable and not information: reasons.append("paid_goal_information_gate_failed")
            gate_failure_counts.update(reasons)
            descriptive_counts["initial_accuracy_adequate_settings"] += int(dwr["initial_accuracy_adequate"])
            descriptive_counts["dwr_stopped_on_initial_mesh_settings"] += int(dwr["initial_stopped"])
            descriptive_counts["zero_nonterminal_work_settings"] += int(dwr["nonterminal_seconds"] == 0.)
            settings.append(dict(case=case,tau=tau,family=metadata[case]["family"],
                numerical=bool(numerical),cost_necessary=cost,goal_information=information,passed=passed,
                fastest_control=fastest,dwr_time_over_control=time_ratio,dwr_dofs_over_control=dof_ratio,
                hypothetical_remaining_over_control=ideal_ratio,dwr_nonterminal_fraction=dwr["fraction"],
                dwr_nonterminal_ms=1000*dwr["nonterminal_seconds"],frozen_gate_failure_reasons=reasons,
                initial_accuracy_adequate=dwr["initial_accuracy_adequate"],initial_q=dwr["initial_q"],
                dwr_stopped_on_initial_mesh=dwr["initial_stopped"],methods=methods))
            for recovered in ("recovered_dwr3","recovered_dwr8"):
                value = methods[recovered]
                valid = bool(references[case]["accepted"] and dwr["all_success"] and value["all_success"])
                exact_paths = exact_final = 0
                for repeat in range(cfg["repeats"]):
                    a=raw_paths[(case,tau,"dwr",repeat)]; b=raw_paths[(case,tau,recovered,repeat)]
                    exact_paths += int(a["trace"] == b["trace"])
                    exact_final += int(a["final"] == b["final"])
                recoveries.append(dict(case=case,tau=tau,method=recovered,comparable=valid,
                    time_over_dwr=value["seconds"]/dwr["seconds"] if valid else None,
                    dofs_over_dwr=value["dofs"]/dwr["dofs"] if valid else None,
                    exact_path_repeats=exact_paths,exact_final_mesh_solution_repeats=exact_final,
                    no_slower_and_no_more_dofs=bool(valid and value["seconds"] <= dwr["seconds"] and value["dofs"] <= dwr["dofs"])))
    geometries=[]
    for case,meta in metadata.items():
        rows=[row for row in settings if row["case"]==case]
        geometries.append(dict(case=case,family=meta["family"],contrast=meta["contrast"],
            passed=all(row["passed"] for row in rows),passed_settings=sum(row["passed"] for row in rows),settings=len(rows)))
    families=[]
    for family in dict.fromkeys(meta["family"] for meta in metadata.values()):
        geometries_in_family=[row for row in geometries if row["family"]==family]
        passed=sum(row["passed"] for row in geometries_in_family)
        families.append(dict(family=family,geometries=len(geometries_in_family),passed=passed,
            retain_for_limited_design=passed >= thresholds["min_geometries_per_family"]))
    retained=[row["family"] for row in families if row["retain_for_limited_design"]]
    decision="LIMITED_DESIGN_ONLY" if retained else "STOP_CURRENT_ESTIMATOR_REPLACEMENT_BRANCH"
    reported=load(reported_summary)["screening"]
    if families != reported["families"] or geometries != reported["geometries"] or decision != reported["decision"]:
        raise ValueError("Independent family decision differs from formal report")
    by_setting={(row["case"],row["tau"]):row for row in reported["settings"]}
    report_fields=("numerical","cost_necessary","goal_information","passed","fastest_control",
                   "dwr_time_over_control","dwr_dofs_over_control","hypothetical_remaining_over_control",
                   "dwr_nonterminal_fraction","dwr_nonterminal_ms")
    for row in settings:
        other=by_setting[(row["case"],row["tau"])]
        for key in report_fields:
            a,b=row[key],other[key]
            if isinstance(a,float) and isinstance(b,float):
                if abs(a-b) > 1e-11*max(1.,abs(a),abs(b)):
                    raise ValueError("Independent setting mismatch: "+key)
            elif a != b:
                raise ValueError("Independent setting mismatch: "+key)
    recovery_summary=[]
    for method in ("recovered_dwr3","recovered_dwr8"):
        rows=[row for row in recoveries if row["method"]==method]; valid=[row for row in rows if row["comparable"]]
        recovery_summary.append(dict(method=method,settings=len(rows),comparable_settings=len(valid),
            median_time_over_dwr=median(row["time_over_dwr"] for row in valid) if valid else None,
            median_dofs_over_dwr=median(row["dofs_over_dwr"] for row in valid) if valid else None,
            no_slower_and_no_more_dofs_settings=sum(row["no_slower_and_no_more_dofs"] for row in rows),
            all_repeats_identical_path_settings=sum(row["exact_path_repeats"]==cfg["repeats"] for row in rows),
            all_repeats_identical_final_settings=sum(row["exact_final_mesh_solution_repeats"]==cfg["repeats"] for row in rows)))
    return dict(independent_raw_gate_recomputation_matches_report=True,decision=decision,
        thresholds=thresholds,retained_families=retained,families=families,geometries=geometries,settings=settings,
        reference_assessments=references,reference_failure_cases=[case for case,value in references.items() if not value["accepted"]],
        path_status_counts=dict(status_counts),overlapping_frozen_gate_failure_counts=dict(gate_failure_counts),
        descriptive_counts=dict(descriptive_counts),recovery_summary=recovery_summary,recovery_comparisons=recoveries,
        scope=dict(no_new_pde_solved=True,no_threshold_changes=True,postrun_descriptive_analysis=True,
            ratio_summaries_not_statistical_significance=True,fastest_controls_retrospective_not_deployable=True,
            initial_accuracy_adequacy_not_an_added_gate=True,
            no_slower_and_no_more_dofs_is_descriptive_not_an_added_gate=True,
            recovery_comparisons_mix_marking_and_verification_schedule=True,
            no_causal_attribution_to_marker_quality_from_final_dofs_alone=True,
            raw_path_hashes_checked=True,payload_arrays_not_recomputed_here=True),
        provenance=dict(raw_complete_sha256=digest(directory/"complete.json"),
                        report_summary_sha256=digest(reported_summary),analysis_source_sha256=digest(__file__)))


def run(directory, reported_summary, output):
    output=Path(output)
    if output.exists(): raise FileExistsError("Do not overwrite mechanism audits")
    result=analyze(directory,reported_summary)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("decision","families","path_status_counts",
        "overlapping_frozen_gate_failure_counts","descriptive_counts","recovery_summary")},indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=ROOT/"results/stage15c_classical")
    parser.add_argument("--summary",type=Path,default=ROOT/"results/stage15c_report/summary.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/stage15c_mechanisms.json")
    args=parser.parse_args();run(args.input,args.summary,args.output)
