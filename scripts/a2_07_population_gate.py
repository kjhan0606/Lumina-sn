#!/usr/bin/env python3
"""A2-07 L-2ion/L-2level deterministic gate and negative controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

NEG_MARKERS = {
    "N1": "A2_07_NEG_STAGE_SWAP",
    "N2": "A2_07_NEG_NEIGHBOR_NE",
    "N3": "A2_07_NEG_TRAD_FOR_TE",
    "N4": "A2_07_NEG_LEVEL_SHUFFLE",
}
FROZEN = {"SiVI": 5, "SVI": 5, "CaVI": 5, "FeVI": 5, "FeVII": 6,
          "NiVI": 5, "NiVII": 6, "CoVI": 5, "CoVII": 6}
FROZEN_ELEMENT = {"SiVI": "Si", "SVI": "S", "CaVI": "Ca",
                  "FeVI": "Fe", "FeVII": "Fe", "NiVI": "Ni",
                  "NiVII": "Ni", "CoVI": "Co", "CoVII": "Co"}
CMFGEN_REQUIRED = ("POPCAL", "POPCOB", "POPIRON", "POPNICK", "POPSIL",
                    "POPSUL", "RVTJ", "SUPERLEVEL_MEMBERSHIP")
CROSSWALK_REQUIRED = ("Z", "spectroscopic_ion", "normalized_label",
                      "excitation_energy_eV", "g", "parent_core_id",
                      "superlevel_membership_id", "status")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.inf
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)


def esym(a: float, b: float) -> float:
    den = abs(a) + abs(b)
    return 0.0 if den == 0 else 2.0 * abs(a - b) / den


def sha_ok(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def synthetic() -> dict:
    frozen = []
    for shell in range(9):
        for ion, charge in FROZEN.items():
            frozen.append({
                "element": FROZEN_ELEMENT[ion], "ion": ion,
                "charge": charge, "shell": shell,
                "n_ion_truth": 1.0e4, "n_element_truth": 1.0e6,
                "population_share": 0.01, "bf_outflow": 2.0e2,
                "bb_incident_flow": 8.0e2, "total_radiative_flow": 1.0e3,
                "rate_flow_share": 0.01, "population_dominant": False,
                "rate_dominant": False, "exclusion_reason": "",
                "source_file": "synthetic-POP", "source_generation": 7,
                "crosswalk_status": "MATCHED"})
    shells = []
    for s in range(9):
        shells.append({
            "shell": s, "velocity": 1000.0 * (s + 1),
            # Adjacent depths are deliberately separated enough that N2's
            # *real neighbouring-depth borrow* is a strong negative witness.
            # The baseline remains a uniform one-percent perturbation.
            "ne_truth": 1.0e8 * (1.30 ** s),
            "ne_lumina": 1.01e8 * (1.30 ** s),
            "elements": [{"Z": 26, "element": "Fe", "eligible": True,
                          "truth": [0.1, 0.8, 0.1],
                          "lumina": [0.11, 0.78, 0.11],
                          "n_element": 1.0e7,
                          "lumina_stage_sum": 1.0e7}],
            "levels": [{"ion": "FeII", "eligible": True,
                        "element": "Fe", "unit_type": "identity",
                        "truth_ion": 8.0e6, "lumina_ion": 7.8e6,
                        "truth": [4.0e6, 2.4e6, 1.6e6],
                        "lumina": [3.9e6, 2.34e6, 1.56e6],
                        "matched": [True, True, True],
                        "crosswalk": [{
                            "Z": 26, "spectroscopic_ion": "II",
                            "normalized_label": f"feii-{i}",
                            "excitation_energy_eV": float(i), "g": i + 1,
                            "parent_core_id": "ground",
                            "superlevel_membership_id": i,
                            "status": "MATCHED"} for i in range(3)]}],
        })
    cmfgen = {name: hashlib.sha256(name.encode()).hexdigest()
              for name in CMFGEN_REQUIRED}
    immutable = {
        "cmfgen_files": cmfgen,
        "cmfgen_out_files": {
            "SUPERLEVEL_MEMBERSHIP.OUT": hashlib.sha256(b"membership-out").hexdigest()},
        "bf_rate_input_sha256": hashlib.sha256(b"bf").hexdigest(),
        "line_jbar_input_sha256": hashlib.sha256(b"bb").hexdigest(),
        "geometry_sha256": hashlib.sha256(b"geometry").hexdigest(),
        "shell_mapping_sha256": hashlib.sha256(b"shellmap").hexdigest(),
        "level_crosswalk_sha256": hashlib.sha256(b"crosswalk").hexdigest(),
        "binary_sha256": hashlib.sha256(b"binary").hexdigest(),
        "source_tree_sha256": hashlib.sha256(b"tree").hexdigest(),
        "environment_sha256": hashlib.sha256(b"env").hexdigest(),
        "te_sha256": hashlib.sha256(b"te").hexdigest(),
        "rng_seed": 1707, "rng_stream": "synthetic",
        "cmfgen_run_token": "synthetic-run", "cmfgen_iteration": 7,
        "source_generation": 7,
        "file_generations": {name: 7 for name in CMFGEN_REQUIRED},
        "out_file_generations": {"SUPERLEVEL_MEMBERSHIP.OUT": 7},
    }
    return {
        "schema": "A2_07_GATE_INPUT_V1", "lane": "ORACLE_INPUT",
        "upstream_status": "PASS", "generation": 7,
        "radfield_generation": 7, "line_cache_generation": 7,
        "te_manifest_sha256": hashlib.sha256(b"te").hexdigest(),
        "atomic_model_sha256": hashlib.sha256(b"atom").hexdigest(),
        "truth_manifest_sha256": hashlib.sha256(b"truth").hexdigest(),
        "truth_active_manifest_sha256": hashlib.sha256(b"active").hexdigest(),
        "truth_active_fixed_before_lumina": True,
        "truth_f_cov": 0.99, "truth_f_cov_lcl": 0.99,
        "truth_active": [
            {"id": "truth-major", "contribution": 99.0, "usable_matched": True},
            {"id": "truth-tail", "contribution": 1.0, "usable_matched": False},
        ],
        "mc_confidence": {"ion_tv_halfwidth": 0.0,
                          "ne_median_halfwidth": 0.0,
                          "ne_p95_halfwidth": 0.0,
                          "level_sum_halfwidth": 0.0,
                          "level_log_p95_halfwidth": 0.0,
                          "dominant_stable": True},
        "population_counters": {
            "pop_generation_required": 7, "pop_generation_committed": 7,
            "pop_shells_attempted": 9, "pop_shells_published": 9,
            "pop_bf_terms": 81, "pop_bb_terms": 81, "pop_exact_zero_terms": 0,
            "pop_blocked_stale": 0, "pop_blocked_unsampled": 0,
            "pop_blocked_oog": 0, "pop_blocked_miss": 0,
            "pop_blocked_profile": 0, "pop_blocked_qhash": 0,
            "pop_blocked_te": 0, "pop_blocked_partition": 0,
            "pop_rank_incomplete": 0, "pop_ne_not_converged": 0,
            "pop_solve_failed": 0, "pop_nonfinite": 0,
            "pop_generation_mismatch": 0, "pop_fallback_attempts": 0,
            "pop_partial_publish_attempts": 0
        },
        "immutable_manifest": immutable, "shells": shells,
        "partition_cases": [
            {"id": "real-s0", "class": "actual_s0_s8", "prod": 4.25,
             "ref": 4.25, "te": 8000.0, "trad": 16000.0,
             "trad_value": 2.0},
            {"id": "fixed-low", "class": "fixed_range", "prod": 2.0,
             "ref": 2.0, "te": 1000.0, "trad": 9000.0,
             "trad_value": 1.0},
            {"id": "underflow", "class": "synthetic_edge", "prod": 1.0,
             "ref": 1.0, "te": 50.0, "trad": 50000.0,
             "trad_value": 5.0}],
        "frozen_ion_contrib": frozen,
    }


def poison(data: dict, neg: str) -> None:
    if neg == "N1":
        for shell in data["shells"]:
            shell["elements"][0]["lumina"] = [0.8, 0.1, 0.1]
    elif neg == "N2":
        truth = [s["ne_truth"] for s in data["shells"]]
        for i, shell in enumerate(data["shells"]):
            shell["ne_lumina"] = truth[i + 1] if i < 8 else truth[7]
    elif neg == "N3":
        for case in data["partition_cases"]:
            case["prod"] = case["trad_value"]
        for shell in data["shells"]:
            shell["levels"][0]["lumina"] = [7.0e6, 0.7e6, 0.1e6]
    elif neg == "N4":
        for shell in data["shells"]:
            values = shell["levels"][0]["lumina"]
            shell["levels"][0]["lumina"] = values[1:] + values[:1]


def evaluate(data: dict) -> tuple[int, dict]:
    required = ("schema", "lane", "upstream_status", "generation", "shells",
                "partition_cases", "frozen_ion_contrib", "immutable_manifest",
                "truth_active_manifest_sha256", "truth_active_fixed_before_lumina",
                "truth_active", "mc_confidence", "population_counters")
    missing = [key for key in required if key not in data]
    if missing:
        return 2, {"status": "FAIL_INPUT", "reason_code": "MISSING_KEYS",
                   "missing": missing}
    if data["upstream_status"] != "PASS":
        return 3, {"status": f"BLOCKED_UPSTREAM_{data['upstream_status']}",
                   "reason_code": "UPSTREAM_NOT_QUALIFIED"}
    if [s.get("shell") for s in data["shells"]] != list(range(9)):
        return 2, {"status": "FAIL_INPUT", "reason_code": "SHELL_DOMAIN_NOT_S0_S8"}
    if not all(sha_ok(data.get(k)) for k in
               ("te_manifest_sha256", "atomic_model_sha256", "truth_manifest_sha256",
                "truth_active_manifest_sha256")):
        return 2, {"status": "FAIL_INPUT", "reason_code": "BAD_MANIFEST_HASH"}
    if not data["truth_active_fixed_before_lumina"]:
        return 5, {"status": "FAIL_CONTRACT",
                   "reason_code": "TRUTH_ACTIVE_NOT_PREFROZEN"}
    immutable = data["immutable_manifest"]
    hash_fields = ("bf_rate_input_sha256", "line_jbar_input_sha256",
                   "geometry_sha256", "shell_mapping_sha256",
                   "level_crosswalk_sha256", "binary_sha256",
                   "source_tree_sha256", "environment_sha256", "te_sha256")
    if not isinstance(immutable, dict) or not all(
            sha_ok(immutable.get(k)) for k in hash_fields):
        return 2, {"status": "FAIL_INPUT", "reason_code": "BAD_IMMUTABLE_MANIFEST"}
    cmfgen = immutable.get("cmfgen_files", {})
    if set(cmfgen) != set(CMFGEN_REQUIRED) or not all(sha_ok(v) for v in cmfgen.values()):
        return 2, {"status": "FAIL_INPUT", "reason_code": "CMFGEN_FILE_SET_OR_HASH"}
    generations = immutable.get("file_generations", {})
    if (set(generations) != set(CMFGEN_REQUIRED) or
            set(generations.values()) != {immutable.get("source_generation")}):
        return 2, {"status": "FAIL_INPUT", "reason_code": "MIXED_SOURCE_GENERATION"}
    out_files = immutable.get("cmfgen_out_files", {})
    if not isinstance(out_files, dict) or not out_files or not all(
            name.endswith("OUT") or ".OUT" in name for name in out_files) or not all(
            sha_ok(value) for value in out_files.values()):
        return 2, {"status": "FAIL_INPUT", "reason_code": "CMFGEN_OUT_SET_OR_HASH"}
    out_generations = immutable.get("out_file_generations", {})
    if (set(out_generations) != set(out_files) or
            set(out_generations.values()) != {immutable.get("source_generation")}):
        return 2, {"status": "FAIL_INPUT", "reason_code": "MIXED_OUT_GENERATION"}
    generation_set = {data.get("generation"), data.get("radfield_generation"),
                      data.get("line_cache_generation"),
                      immutable.get("source_generation")}
    if len(generation_set) != 1:
        return 5, {"status": "FAIL_GENERATION_MISMATCH",
                   "reason_code": "BF_BB_GENERATION_MISMATCH"}
    if data.get("te_manifest_sha256") != immutable.get("te_sha256"):
        return 5, {"status": "FAIL_GENERATION_MISMATCH",
                   "reason_code": "TE_MANIFEST_BINDING_MISMATCH"}

    active = data["truth_active"]
    if (not isinstance(active, list) or not active or
            any(not isinstance(row, dict) or not math.isfinite(float(
                row.get("contribution", math.nan))) or
                float(row.get("contribution", 0.0)) < 0.0 for row in active)):
        return 2, {"status": "FAIL_INPUT", "reason_code": "BAD_TRUTH_ACTIVE"}
    ordered = sorted(active, key=lambda row: (-float(row["contribution"]), str(row.get("id"))))
    if [row.get("id") for row in active] != [row.get("id") for row in ordered]:
        return 5, {"status": "FAIL_CONTRACT", "reason_code": "TRUTH_ACTIVE_NOT_PREFROZEN_ORDER"}
    total = sum(float(row["contribution"]) for row in active)
    cutoff, selected = 0.999 * total, []
    cumulative = 0.0
    for row in active:
        selected.append(row)
        cumulative += float(row["contribution"])
        if cumulative >= cutoff:
            boundary = float(row["contribution"])
            selected.extend(r for r in active[len(selected):]
                            if float(r["contribution"]) == boundary)
            break
    active_den = sum(float(row["contribution"]) for row in selected)
    active_num = sum(float(row["contribution"]) for row in selected
                     if bool(row.get("usable_matched", False)))
    computed_f_cov = active_num / active_den if active_den > 0.0 else 0.0
    if abs(computed_f_cov - float(data.get("truth_f_cov", -1.0))) > 1e-12:
        return 5, {"status": "FAIL_CONTRACT", "reason_code": "TRUTH_COVERAGE_REDEFINED"}

    counters = data["population_counters"]
    zero_counter_names = (
        "pop_blocked_stale", "pop_blocked_unsampled", "pop_blocked_oog",
        "pop_blocked_miss", "pop_blocked_profile", "pop_blocked_qhash",
        "pop_blocked_te", "pop_blocked_partition", "pop_rank_incomplete",
        "pop_ne_not_converged", "pop_solve_failed", "pop_nonfinite",
        "pop_generation_mismatch", "pop_fallback_attempts",
        "pop_partial_publish_attempts")
    counter_ok = (isinstance(counters, dict) and
                  counters.get("pop_generation_required") == data.get("generation") and
                  counters.get("pop_generation_committed") == data.get("generation") and
                  counters.get("pop_shells_attempted") == len(data["shells"]) and
                  counters.get("pop_shells_published") == len(data["shells"]) and
                  int(counters.get("pop_bf_terms", 0)) > 0 and
                  int(counters.get("pop_bb_terms", 0)) > 0 and
                  all(counters.get(name) == 0 for name in zero_counter_names))
    if not counter_ok:
        return 5, {"status": "FAIL_CONTRACT", "reason_code": "POP_COUNTER_INVARIANT"}

    frozen = data["frozen_ion_contrib"]
    required_frozen_fields = ("element", "ion", "charge", "shell",
                              "n_ion_truth", "n_element_truth",
                              "population_share", "bf_outflow",
                              "bb_incident_flow", "rate_flow_share",
                              "source_file", "source_generation",
                              "crosswalk_status")
    if any(any(k not in row for k in required_frozen_fields) for row in frozen):
        return 3, {"status": "BLOCKED_FROZEN_RATE_CONTRIBUTION",
                   "reason_code": "MISSING_FROZEN_FLOW_FIELD"}
    source_generation = immutable.get("source_generation")
    for row in frozen:
        numeric = ("n_ion_truth", "n_element_truth", "population_share",
                   "bf_outflow", "bb_incident_flow", "total_radiative_flow",
                   "rate_flow_share")
        if (any(not math.isfinite(float(row.get(name, math.nan))) or
                float(row.get(name, -1.0)) < 0.0 for name in numeric) or
                row.get("source_generation") != source_generation or
                row.get("crosswalk_status") != "MATCHED" or
                abs(float(row["total_radiative_flow"]) -
                    (float(row["bf_outflow"]) + float(row["bb_incident_flow"]))) >
                1e-12 * max(1.0, float(row["total_radiative_flow"]))):
            return 3, {"status": "BLOCKED_FROZEN_RATE_CONTRIBUTION",
                       "reason_code": "INVALID_FROZEN_FLOW_OR_GENERATION"}
    dominant_elements = {
        row["element"] for row in frozen
        if float(row["population_share"]) >= 0.5 or
           float(row["rate_flow_share"]) >= 0.5}

    ion_tv, dominant, closure, ne_errors = [], [], [], []
    level_sum, level_log, level_cov, hard_zero = [], [], [], 0
    excluded_level_truth = excluded_level_matched = 0.0
    for shell in data["shells"]:
        ne_errors.append(esym(float(shell["ne_lumina"]), float(shell["ne_truth"])))
        for element in shell["elements"]:
            if not element.get("eligible", False):
                continue
            truth, lumina = element["truth"], element["lumina"]
            closure.append(abs(float(element["lumina_stage_sum"]) -
                               float(element["n_element"])) / float(element["n_element"]))
            if element.get("element") in dominant_elements:
                continue
            ion_tv.append(0.5 * sum(abs(a - b) for a, b in zip(truth, lumina)))
            tmax = max(truth)
            truth_set = {i for i, x in enumerate(truth) if abs(x - tmax) <= 1e-12}
            dominant.append(max(range(len(lumina)), key=lumina.__getitem__) in truth_set)
        for unit in shell["levels"]:
            if not unit.get("eligible", False):
                continue
            crosswalk = unit.get("crosswalk", [])
            if (len(crosswalk) != len(unit.get("truth", [])) or
                    any(any(k not in row for k in CROSSWALK_REQUIRED)
                        for row in crosswalk)):
                return 2, {"status": "FAIL_INPUT",
                           "reason_code": "BAD_LEVEL_CROSSWALK"}
            derived_matched = [row["status"] == "MATCHED" for row in crosswalk]
            if derived_matched != unit.get("matched"):
                return 5, {"status": "FAIL_CONTRACT",
                           "reason_code": "CROSSWALK_STATUS_SUBSTITUTION"}
            truth, lumina, matched = unit["truth"], unit["lumina"], unit["matched"]
            truth_ion, lumina_ion = float(unit["truth_ion"]), float(unit["lumina_ion"])
            if unit.get("element") in dominant_elements:
                excluded_level_truth += truth_ion
                excluded_level_matched += sum(x for x, m in zip(truth, matched) if m)
                continue
            level_cov.append(sum(x for x, m in zip(truth, matched) if m) / truth_ion)
            level_sum.append(sum(abs(a - b) for a, b, m in zip(truth, lumina, matched) if m) /
                             truth_ion)
            for a, b, m in zip(truth, lumina, matched):
                if not m:
                    continue
                if a > 0.0 and (b <= 0.0 or not math.isfinite(b)):
                    hard_zero += 1
                elif a > 0.0 and b > 0.0:
                    level_log.append(abs(math.log10((b / lumina_ion) / (a / truth_ion))))

    frozen_keys = {(r.get("ion"), r.get("charge"), r.get("shell")) for r in frozen}
    expected = {(ion, charge, shell) for ion, charge in FROZEN.items() for shell in range(9)}
    frozen_ok = len(frozen) == 81 and frozen_keys == expected and all(
        r["charge"] == FROZEN[r["ion"]] for r in frozen)
    if not ion_tv or not level_cov:
        return 3, {"status": "BLOCKED_NO_ELIGIBLE_ELEMENT",
                   "reason_code": "FROZEN_DOMINANCE_EXCLUDED_ALL"}
    zerr = [abs(float(c["prod"]) - float(c["ref"])) / abs(float(c["ref"]))
            for c in data["partition_cases"] if float(c["ref"]) != 0.0]
    partition_classes = {c.get("class") for c in data["partition_cases"]}
    if partition_classes != {"actual_s0_s8", "fixed_range", "synthetic_edge"}:
        return 2, {"status": "FAIL_INPUT",
                   "reason_code": "PARTITION_FIXTURE_CLASSES_INCOMPLETE"}
    metrics = {
        "ion_tv_max": max(ion_tv, default=math.inf),
        "dominant_stage_all": all(dominant) and bool(dominant),
        "ne_esym_median": statistics.median(ne_errors),
        "ne_esym_p95": percentile(ne_errors, 0.95),
        "closure_max": max(closure, default=math.inf),
        "level_coverage_min": min(level_cov, default=0.0),
        "level_sum_max": max(level_sum, default=math.inf),
        "level_log_p95": percentile(level_log, 0.95),
        "level_hard_zero": hard_zero,
        "Z_relerr_max": max(zerr, default=math.inf),
        "truth_f_cov": computed_f_cov,
        "frozen_9x9_complete": frozen_ok,
        "excluded_frozen_dominant_elements": sorted(dominant_elements),
        "excluded_frozen_level_coverage": (excluded_level_matched / excluded_level_truth
                                            if excluded_level_truth > 0.0 else None),
    }
    confidence = data["mc_confidence"]
    ci_ok = (float(confidence.get("ion_tv_halfwidth", math.inf)) <= 0.0333 and
             float(confidence.get("ne_median_halfwidth", math.inf)) <= 0.0333 and
             float(confidence.get("ne_p95_halfwidth", math.inf)) <= 0.0667 and
             float(confidence.get("level_sum_halfwidth", math.inf)) <= 0.0333 and
             float(confidence.get("level_log_p95_halfwidth", math.inf)) <= 0.10 and
             bool(confidence.get("dominant_stable", False)) and
             float(data.get("truth_f_cov_lcl", 0.0)) >= 0.95)
    checks = {
        "ion_tv": metrics["ion_tv_max"] <= 0.10,
        "dominant": metrics["dominant_stage_all"],
        "ne": metrics["ne_esym_median"] <= 0.10 and metrics["ne_esym_p95"] <= 0.20,
        "closure": metrics["closure_max"] <= 1e-10,
        "level_coverage": metrics["level_coverage_min"] >= 0.95,
        "level_sum": metrics["level_sum_max"] <= 0.10,
        "level_log": metrics["level_log_p95"] <= 0.30 and hard_zero == 0,
        "partition": metrics["Z_relerr_max"] <= 1e-10,
        "truth_coverage": metrics["truth_f_cov"] >= 0.95,
        "frozen_disclosure": frozen_ok,
        "mc_uncertainty": ci_ok,
    }
    passed = all(checks.values())
    ion_checks = {k: checks[k] for k in
                  ("ion_tv", "dominant", "ne", "closure",
                   "truth_coverage", "frozen_disclosure", "mc_uncertainty")}
    level_checks = {k: checks[k] for k in
                    ("level_coverage", "level_sum", "level_log",
                     "truth_coverage", "mc_uncertainty")}
    partition_checks = {"partition": checks["partition"]}
    blocked_reason = None
    if not checks["truth_coverage"] or not checks["level_coverage"]:
        blocked_reason = "BLOCKED_COVERAGE"
    elif not checks["mc_uncertainty"]:
        blocked_reason = "BLOCKED_MC_UNCERTAINTY"
    result_rc = 0 if passed else (3 if blocked_reason else 4)
    return result_rc, {
        "schema": "A2_07_GATE_RESULT_V1", "lane": data["lane"],
        "status": "PASS" if passed else (blocked_reason or "FAIL_METRIC"),
        "reason_code": "OK" if passed else (
            blocked_reason or "THRESHOLD_EXCEEDED"),
        "metrics": metrics, "checks": checks,
        "population_counters": counters,
        "new_layer_status": {
            "L2ION": "PASS" if all(ion_checks.values()) else
                     (blocked_reason or "FAIL_METRIC"),
            "L2LEVEL": "PASS" if all(level_checks.values()) else
                       (blocked_reason or "FAIL_METRIC"),
            "PARTITION_CPU": "PASS" if all(partition_checks.values()) else "FAIL_METRIC",
        },
        "mc_confidence": confidence,
    }


def write_result(path: Path | None, report: dict, child_rc: int, wrapper_rc: int) -> None:
    report = dict(report, child_rc=child_rc, wrapper_rc=wrapper_rc)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def write_layer_results(path: Path, report: dict, child_rc: int) -> None:
    """Keep the two physical layers and the numerical partition gate distinct."""
    lane = "".join(c if c.isalnum() else "_" for c in str(report.get("lane", "UNKNOWN")))
    statuses = report.get("new_layer_status", {})
    checks = report.get("checks", {})
    partitions = {
        "L2ION": ("ion_tv", "dominant", "ne", "closure", "truth_coverage",
                  "frozen_disclosure", "mc_uncertainty"),
        "L2LEVEL": ("level_coverage", "level_sum", "level_log",
                    "truth_coverage", "mc_uncertainty"),
        "PARTITION_CPU": ("partition",),
    }
    for layer, names in partitions.items():
        status = statuses.get(layer, report.get("status", "FAIL_INPUT"))
        doc = {
            "schema": "A2_07_LAYER_RESULT_V1", "lane": lane,
            "layer": layer, "status": status,
            "reason_code": report.get("reason_code", "UNKNOWN"),
            "checks": {name: checks.get(name) for name in names},
            "metrics": report.get("metrics", {}),
            "child_rc": child_rc, "wrapper_rc": child_rc,
        }
        target = path.parent / f"A2_07_{lane}_{layer}_RESULT.json"
        target.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def write_frozen_csv(path: Path, data: dict) -> None:
    fields = ["element", "ion", "charge", "shell", "velocity",
              "n_ion_truth", "n_element_truth", "population_share",
              "bf_outflow", "bb_incident_flow", "total_radiative_flow",
              "rate_flow_share", "population_dominant", "rate_dominant",
              "exclusion_reason", "source_file", "source_generation",
              "crosswalk_status"]
    velocities = {s["shell"]: s["velocity"] for s in data["shells"]}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in data["frozen_ion_contrib"]:
            row = {key: source.get(key, "") for key in fields}
            row["element"] = source.get("element", "".join(filter(str.isalpha, source["ion"])))
            row["velocity"] = source.get("velocity", velocities[source["shell"]])
            row["total_radiative_flow"] = source.get(
                "total_radiative_flow",
                float(source.get("bf_outflow", 0.0)) +
                float(source.get("bb_incident_flow", 0.0)))
            row["population_dominant"] = source.get(
                "population_dominant", float(source.get("population_share", 0.0)) >= 0.5)
            row["rate_dominant"] = source.get(
                "rate_dominant", float(source.get("rate_flow_share", 0.0)) >= 0.5)
            row["crosswalk_status"] = source.get("crosswalk_status", "MATCHED")
            writer.writerow(row)


def self_check(output: Path | None) -> int:
    base = synthetic()
    rc, baseline = evaluate(base)
    negative = {}
    if rc != 0:
        write_result(output, {"status": "FAIL_SELF_CHECK_BASELINE", "baseline": baseline}, rc, 4)
        return 4
    with tempfile.TemporaryDirectory(prefix="a2_07_gate_") as tmp:
        fixture = Path(tmp) / "fixture.json"
        fixture.write_text(json.dumps(base), encoding="utf-8")
        baseline_path = Path(tmp) / "A2_07_ORACLE_INPUT_RESULT.json"
        baseline_proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--input", str(fixture),
             "--output", str(baseline_path)], text=True, capture_output=True, check=False)
        expected_outputs = [
            baseline_path, Path(tmp) / "A2_07_FROZEN_ION_CONTRIB.csv",
            Path(tmp) / "A2_07_ORACLE_INPUT_L2ION_RESULT.json",
            Path(tmp) / "A2_07_ORACLE_INPUT_L2LEVEL_RESULT.json",
            Path(tmp) / "A2_07_ORACLE_INPUT_PARTITION_CPU_RESULT.json",
        ]
        artifact_writer_ok = baseline_proc.returncode == 0 and all(
            path.is_file() and path.stat().st_size > 0 for path in expected_outputs)
        for neg, marker in NEG_MARKERS.items():
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--input", str(fixture), "--negative", neg],
                                  text=True, capture_output=True, check=False)
            try:
                poisoned = json.loads(proc.stdout[proc.stdout.index("{"):])
            except (ValueError, json.JSONDecodeError):
                poisoned = {}
            checks = poisoned.get("checks", {})
            witness = ((neg == "N1" and (checks.get("ion_tv") is False or
                                          checks.get("dominant") is False)) or
                       (neg == "N2" and checks.get("ne") is False) or
                       (neg == "N3" and checks.get("partition") is False and
                        (checks.get("level_sum") is False or
                         checks.get("level_log") is False)) or
                       (neg == "N4" and (checks.get("level_sum") is False or
                                         checks.get("level_log") is False)))
            negative[neg] = {"marker": marker, "marker_seen": marker in proc.stdout,
                             "metric_witness_seen": witness,
                             "child_rc": proc.returncode, "wrapper_rc": 0}
    ok = artifact_writer_ok and all(
        v["marker_seen"] and v["metric_witness_seen"] and v["child_rc"] == 4
        for v in negative.values())
    report = {"schema": "A2_07_GATE_SELF_CHECK_V1",
              "status": "PASS" if ok else "FAIL_NEGATIVE_CONTROL",
              "reason_code": "OK" if ok else "NEGATIVE_RC_OR_MARKER",
              "baseline": baseline, "artifact_writer_status": {
                  "status": "PASS" if artifact_writer_ok else "FAIL",
                  "expected_outputs": 5},
              "negative_control_status": negative}
    write_result(output, report, 0 if ok else 4, 0 if ok else 4)
    return 0 if ok else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negative", choices=NEG_MARKERS)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--write-self-fixture", type=Path)
    args = parser.parse_args()
    if args.write_self_fixture:
        args.write_self_fixture.parent.mkdir(parents=True, exist_ok=True)
        args.write_self_fixture.write_text(json.dumps(synthetic(), indent=2) + "\n",
                                            encoding="utf-8")
        return 0
    if args.self_check:
        return self_check(args.output)
    if not args.input:
        parser.error("--input is required unless --self-check is used")
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        write_result(args.output, {"status": "FAIL_INPUT", "reason_code": str(exc)}, 2, 2)
        return 2
    if args.negative:
        poison(data, args.negative)
        print(NEG_MARKERS[args.negative])
    rc, report = evaluate(data)
    if args.output and not args.negative:
        write_frozen_csv(args.output.parent / "A2_07_FROZEN_ION_CONTRIB.csv", data)
        write_layer_results(args.output, report, rc)
    write_result(args.output, report, rc, 0 if args.negative else rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
