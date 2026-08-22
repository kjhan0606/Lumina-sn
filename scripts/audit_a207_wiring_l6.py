#!/usr/bin/env python3
"""Offline, fail-closed attribution audit for DET-A207-WIRING."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


A207_WIRING_L6_ATTRIBUTION = "DET-A207-WIRING"
SOURCE_REVISION = "8c2aace"
SOURCE_PATH = "src/lumina_plasma.c"
ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
R7_PREFIX = "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED"
KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
SLOT_LINE = re.compile(
    r"^\s*Z=(26|27|28)\s+ion=(\d+):\s+\d+\s+levels\s*$"
)
H_PLANCK = 6.62607015e-27
C_LIGHT = 2.99792458e10
K_BOLTZMANN = 1.380649e-16
EV_TO_ERG = 1.602176634e-12
TEMPERATURE_K = 10020.0
EXPECTED_ROWS = 594
EXPECTED_ROWS_PER_ITERATION = 297
COMMON_MODE_TOLERANCE = 1.0e-14
SOURCE_STABILITY_TOLERANCE = 1.0e-12
THERMAL_SOURCE_TOLERANCE = 2.0e-5
TARGET_Z = {26, 27, 28}


def fields(line: str) -> dict[str, str]:
    return dict(KEY_VALUE.findall(line))


def gate(status: str, reasons: Iterable[str], **details: Any) -> dict[str, Any]:
    result = {"status": status, "reasons": list(dict.fromkeys(reasons))}
    result.update(details)
    return result


def read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def git_blob(repo: Path) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{SOURCE_REVISION}:{SOURCE_PATH}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except (OSError, UnicodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"git show rc={completed.returncode}"
        return None, message
    return completed.stdout, None


def audit_w1(run_footer: str, stdout: str) -> dict[str, Any]:
    env_names = (
        "LUMINA_NLTE_STAGE4",
        "LUMINA_NLTE_ELEMENT_WIDE",
        "LUMINA_SUPER_LEVELS",
        "LUMINA_A210_LINE_SATURATION_TARGET_ION",
    )
    env_values: dict[str, list[str]] = {}
    for name in env_names:
        pattern = re.compile(rf"^{re.escape(name)}=(.*)$", re.MULTILINE)
        env_values[name] = pattern.findall(run_footer)

    slots: dict[str, list[int]] = {str(z): [] for z in sorted(TARGET_Z)}
    for line in stdout.splitlines():
        match = SLOT_LINE.match(line)
        if match:
            slots[match.group(1)].append(int(match.group(2)))
    slot_sets = {z: sorted(set(values)) for z, values in slots.items()}
    super_lines = [
        line.strip()
        for line in stdout.splitlines()
        if re.match(r"^\s*\[NLTE\] Super-levels: off \(identity\)", line)
    ]
    mapped_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip()
        == "[NLTE] Lines mapped to NLTE ions: 1777859 / 2588798"
    ]

    reasons: list[str] = []
    if any(3 in values for values in slots.values()):
        reasons.append("COVERAGE_SLOT_PRESENT")
    forbidden = env_names[:3]
    if any(env_values[name] for name in forbidden):
        reasons.append("FORBIDDEN_COVERAGE_ENV_PRESENT")
    if env_values[env_names[3]] != ["3"]:
        reasons.append("TARGET_ION_ENV_MISMATCH")
    if any(set(slots[str(z)]) != {1, 2} for z in TARGET_Z):
        reasons.append("NLTE_SLOT_SET_MISMATCH")
    if len(super_lines) != 1:
        reasons.append("SUPER_LEVEL_FOSSIL_MISMATCH")
    if len(mapped_lines) != 1:
        reasons.append("MAPPED_LINES_FOSSIL_MISMATCH")
    return gate(
        "PASS" if not reasons else "FAIL",
        reasons,
        env_occurrences=env_values,
        slot_ions=slot_sets,
        slot_line_counts={z: len(values) for z, values in slots.items()},
        super_level_lines=super_lines,
        mapped_line_fossils=mapped_lines,
    )


def mask_c_lexical(source: str) -> str:
    """Mask comments and literals while preserving offsets and newlines."""
    output = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code" and char == "/" and following == "/":
            output[index] = output[index + 1] = " "
            index += 2
            state = "line_comment"
            continue
        if state == "code" and char == "/" and following == "*":
            output[index] = output[index + 1] = " "
            index += 2
            state = "block_comment"
            continue
        if state == "code" and char in ('"', "'"):
            quote = char
            output[index] = " "
            index += 1
            state = "literal"
            continue
        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                output[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                output[index] = output[index + 1] = " "
                index += 2
                state = "code"
            else:
                if char != "\n":
                    output[index] = " "
                index += 1
            continue
        if state == "literal":
            output[index] = "\n" if char == "\n" else " "
            if char == "\\" and following:
                output[index + 1] = "\n" if following == "\n" else " "
                index += 2
                continue
            if char == quote:
                state = "code"
            index += 1
            continue
        index += 1
    return "".join(output)


def function_span(source: str, name: str) -> tuple[int, int] | None:
    masked = mask_c_lexical(source)
    definition = re.search(
        rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", masked, re.DOTALL
    )
    if not definition:
        return None
    opening = masked.find("{", definition.start())
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return definition.start(), index + 1
    return None


def parse_c_array(source: str, name: str) -> tuple[list[int] | None, str | None]:
    masked = mask_c_lexical(source)
    match = re.search(
        rf"\b{re.escape(name)}\s*(?:\[[^\]]*\])?\s*=\s*\{{(.*?)\}}\s*;",
        masked,
        re.DOTALL,
    )
    if not match:
        return None, f"ARRAY_NOT_FOUND_{name}"
    body = match.group(1)
    residues = re.sub(r"[-+]?\d+|[\s,]", "", body)
    if residues:
        return None, f"ARRAY_TOKEN_INVALID_{name}"
    values = [int(token) for token in re.findall(r"[-+]?\d+", body)]
    if not values:
        return None, f"ARRAY_EMPTY_{name}"
    return values, None


def paired_table(
    source: str, z_name: str, ion_name: str
) -> tuple[set[tuple[int, int]] | None, dict[str, Any]]:
    z_values, z_error = parse_c_array(source, z_name)
    ion_values, ion_error = parse_c_array(source, ion_name)
    detail: dict[str, Any] = {
        "z_array": z_name,
        "ion_array": ion_name,
        "z_count": None if z_values is None else len(z_values),
        "ion_count": None if ion_values is None else len(ion_values),
        "parse_errors": [x for x in (z_error, ion_error) if x],
    }
    if z_values is None or ion_values is None:
        return None, detail
    if len(z_values) != len(ion_values):
        detail["parse_errors"].append("ARRAY_LENGTH_MISMATCH")
        return None, detail
    pairs = set(zip(z_values, ion_values))
    detail["pairs"] = [[z, ion] for z, ion in sorted(pairs)]
    detail["ion_values"] = sorted(set(ion_values))
    return pairs, detail


def source_line(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def audit_w2(source: str) -> tuple[dict[str, Any], set[tuple[int, int]] | None,
                                    set[tuple[int, int]] | None]:
    anchor_specs = (
        (
            "unmapped_skip",
            "nlte_update_tau_sobolev_with_authority",
            re.compile(
                r"if\s*\(\s*element_inactive\s*\|\|\s*!authority\.mapped\s*\)"
                r"\s*continue\s*;",
                re.DOTALL,
            ),
            19568,
        ),
        (
            "lte_population_branch",
            "a209_upper_population_for_tau",
            re.compile(
                r"if\s*\(\s*!use_nlte\s*&&\s*lte_level_density\s*\)\s*\{",
                re.DOTALL,
            ),
            8973,
        ),
        (
            "candidate_tau_rebuild",
            "nlte_population_candidate_produce_tau_source",
            re.compile(r"\bcompute_tau_sobolev\s*\(", re.DOTALL),
            20809,
        ),
    )
    anchors: dict[str, Any] = {}
    missing: list[str] = []
    for label, function_name, pattern, expected_line in anchor_specs:
        span = function_span(source, function_name)
        match = None if span is None else pattern.search(source, span[0], span[1])
        line = None if match is None else source_line(source, match.start())
        anchors[label] = {
            "function": function_name,
            "present": match is not None,
            "line": line,
            "registered_line": expected_line,
            "line_moved": line is not None and line != expected_line,
        }
        if match is None:
            missing.append(label)

    base_pairs, base_detail = paired_table(
        source, "NLTE_TARGET_Z", "NLTE_TARGET_ION"
    )
    stage4_pairs, stage4_detail = paired_table(
        source, "NLTE_TARGET_Z4", "NLTE_TARGET_ION4"
    )
    reasons: list[str] = []
    classifications: list[str] = []
    if missing:
        reasons.append("ANCHOR_MISSING")
        classifications.append("BLOCKER")
    if base_pairs is None or stage4_pairs is None:
        reasons.append("SOURCE_TABLE_PARSE_FAILURE")
        classifications.append("BLOCKER")
    if base_pairs is not None and any(ion == 3 for _, ion in base_pairs):
        reasons.append("SOURCE_MAPPING_CONTRADICTION")
        classifications.append("COUNTEREXAMPLE")
    if stage4_pairs is not None and not any(ion == 3 for _, ion in stage4_pairs):
        reasons.append("SOURCE_MAPPING_CONTRADICTION")
        classifications.append("COUNTEREXAMPLE")
    return (
        gate(
            "PASS" if not reasons else "FAIL",
            reasons,
            failure_classifications=list(dict.fromkeys(classifications)),
            blob=f"{SOURCE_REVISION}:{SOURCE_PATH}",
            anchors=anchors,
            missing_anchors=missing,
            base_table=base_detail,
            stage4_table=stage4_detail,
            base_contains_ion3=(
                None if base_pairs is None
                else any(ion == 3 for _, ion in base_pairs)
            ),
            stage4_contains_ion3=(
                None if stage4_pairs is None
                else any(ion == 3 for _, ion in stage4_pairs)
            ),
        ),
        base_pairs,
        stage4_pairs,
    )


def bracket_rows(stderr: str) -> dict[str, Any]:
    all_rows: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    summaries = 0
    iterations: dict[int, list[dict[str, str]]] = {}
    issues: list[str] = []
    markers: list[dict[str, Any]] = []
    for line_number, line in enumerate(stderr.splitlines(), 1):
        if line.startswith(ROW_PREFIX):
            row = fields(line)
            row["_log_line"] = str(line_number)
            all_rows.append(row)
            pending.append(row)
        elif line.startswith(SUMMARY_PREFIX):
            summaries += 1
        elif line.startswith(R7_PREFIX):
            marker = fields(line)
            marker_record: dict[str, Any] = {
                "log_line": line_number,
                "lane": marker.get("lane"),
                "phase": marker.get("phase"),
                "iter": marker.get("iter"),
                "te_generation": marker.get("te_generation"),
            }
            markers.append(marker_record)
            if not pending:
                continue
            try:
                iteration = int(marker["iter"])
            except (KeyError, ValueError):
                issues.append("INVALID_R7_ITERATION")
                pending = []
                summaries = 0
                continue
            if marker.get("lane") != "DET" or marker.get("phase") != "A2-10":
                issues.append("INVALID_R7_MARKER")
            if marker.get("te_generation") != f"{iteration + 1}->{iteration + 2}":
                issues.append("INVALID_R7_GENERATION")
            if summaries != 1:
                issues.append("SUMMARY_COUNT_MISMATCH")
            if iteration in iterations:
                issues.append("DUPLICATE_ITERATION_BLOCK")
            else:
                iterations[iteration] = pending
                for row in pending:
                    row["_iteration"] = str(iteration)
            pending = []
            summaries = 0
    if pending:
        issues.append("UNATTRIBUTED_TRAILING_ROWS")
    if summaries:
        issues.append("UNATTRIBUTED_SUMMARY")
    if set(iterations) != {0, 1}:
        issues.append("ITERATION_SET_MISMATCH")
    return {
        "all_rows": all_rows,
        "iterations": iterations,
        "issues": list(dict.fromkeys(issues)),
        "markers": markers,
        "pending_count": len(pending),
    }


def row_integer(row: dict[str, str], name: str) -> tuple[int | None, str]:
    if name not in row:
        return None, "missing"
    try:
        return int(row[name]), "valid"
    except ValueError:
        return None, "invalid"


def row_float(row: dict[str, str], name: str) -> tuple[float | None, str]:
    if name not in row:
        return None, "missing"
    try:
        value = float(row[name])
    except ValueError:
        return None, "invalid"
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, "valid"


def mapping_census(
    rows: list[dict[str, str]], pairs: set[tuple[int, int]] | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(rows),
        "mapped": 0,
        "unmapped": 0,
        "unknown": 0,
        "mapped_examples": [],
        "unknown_examples": [],
    }
    for row in rows:
        z_value, z_status = row_integer(row, "Z")
        ion_value, ion_status = row_integer(row, "ion")
        identifier = {
            "log_line": row.get("_log_line"),
            "line": row.get("line"),
            "Z": row.get("Z"),
            "ion": row.get("ion"),
        }
        if pairs is None or z_status != "valid" or ion_status != "valid":
            result["unknown"] += 1
            if len(result["unknown_examples"]) < 20:
                result["unknown_examples"].append(identifier)
        elif (z_value, ion_value) in pairs:
            result["mapped"] += 1
            if len(result["mapped_examples"]) < 20:
                result["mapped_examples"].append(identifier)
        else:
            result["unmapped"] += 1
    result["f_mapped"] = (
        result["mapped"] / result["total"]
        if result["total"] and result["unknown"] == 0
        else None
    )
    return result


def audit_w3(bracket: dict[str, Any], base_pairs: set[tuple[int, int]] | None
             ) -> dict[str, Any]:
    rows = bracket["all_rows"]
    integer_fields = ("line", "Z", "ion", "producer_raw_defined")
    field_census = {
        name: Counter(row_integer(row, name)[1] for row in rows)
        for name in integer_fields
    }
    target_counter: Counter[str] = Counter()
    target_examples: list[dict[str, Any]] = []
    raw_flag_counter: Counter[str] = Counter()
    by_iteration = {
        str(iteration): len(iteration_rows)
        for iteration, iteration_rows in sorted(bracket["iterations"].items())
    }
    attributed = sum(by_iteration.values())
    by_iteration["UNATTRIBUTED"] = len(rows) - attributed
    for row in rows:
        z_value, z_status = row_integer(row, "Z")
        ion_value, ion_status = row_integer(row, "ion")
        if z_status == "valid" and ion_status == "valid":
            key = f"Z={z_value},ion={ion_value}"
            target_counter[key] += 1
            if (z_value not in TARGET_Z or ion_value != 3) and len(target_examples) < 20:
                target_examples.append({
                    "log_line": row.get("_log_line"),
                    "line": row.get("line"),
                    "Z": z_value,
                    "ion": ion_value,
                })
        raw_value, raw_status = row_integer(row, "producer_raw_defined")
        raw_flag_counter[
            str(raw_value) if raw_status == "valid" else raw_status.upper()
        ] += 1

    mapped = mapping_census(rows, base_pairs)
    reasons: list[str] = []
    classifications: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        reasons.append("ROW_COUNT_MISMATCH")
        classifications.append("BLOCKER")
    if bracket["issues"] or by_iteration.get("0") != EXPECTED_ROWS_PER_ITERATION \
            or by_iteration.get("1") != EXPECTED_ROWS_PER_ITERATION:
        reasons.append("ROW_ATTRIBUTION_BLOCKED")
        classifications.append("BLOCKER")
    if any(field_census[name].get("valid", 0) != len(rows) for name in integer_fields):
        reasons.append("ROW_FIELD_CENSUS_FAILURE")
        classifications.append("BLOCKER")
    if any(
        (row_integer(row, "producer_raw_defined")[0] != 1)
        for row in rows
    ):
        reasons.append("PRODUCER_RAW_UNDEFINED")
        classifications.append("BLOCKER")
    if mapped["unknown"]:
        reasons.append("MAPPING_CENSUS_INCOMPLETE")
        classifications.append("BLOCKER")
    if target_examples or mapped["mapped"]:
        reasons.append("MAPPED_COUNTEREXAMPLE")
        classifications.append("COUNTEREXAMPLE")
    return gate(
        "PASS" if not reasons else "FAIL",
        reasons,
        failure_classifications=list(dict.fromkeys(classifications)),
        row_count=len(rows),
        expected_row_count=EXPECTED_ROWS,
        rows_by_iteration=by_iteration,
        bracket_issues=bracket["issues"],
        r7_markers=bracket["markers"],
        field_census={name: dict(counts) for name, counts in field_census.items()},
        producer_raw_defined=dict(raw_flag_counter),
        z_ion_census=dict(sorted(target_counter.items())),
        target_counterexamples=target_examples,
        mapping=mapped,
    )


def planck_nu(frequency_hz: float, temperature_K: float) -> float:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency")
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature")
    exponent = H_PLANCK * frequency_hz / (K_BOLTZMANN * temperature_K)
    try:
        denominator = math.expm1(exponent)
    except OverflowError as exc:
        raise ValueError("planck_overflow") from exc
    value = 2.0 * H_PLANCK * frequency_hz ** 3 / (C_LIGHT ** 2 * denominator)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("planck_value")
    return value


def model_time(config_text: str) -> tuple[float | None, str | None]:
    try:
        payload = json.loads(config_text)
        value = float(payload["time_explosion_s"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not math.isfinite(value) or value <= 0.0:
        return None, "time_explosion_s must be finite and positive"
    return value, None


def index_rows(rows: list[dict[str, str]]) -> tuple[dict[int, dict[str, str]],
                                                    list[dict[str, Any]]]:
    indexed: dict[int, dict[str, str]] = {}
    issues: list[dict[str, Any]] = []
    for row in rows:
        line_value, status = row_integer(row, "line")
        if status != "valid" or line_value is None:
            issues.append({"kind": "INVALID_LINE_ID", "log_line": row.get("_log_line")})
        elif line_value in indexed:
            issues.append({"kind": "DUPLICATE_LINE_ID", "line": line_value})
        else:
            indexed[line_value] = row
    return indexed, issues


def audit_w4(bracket: dict[str, Any], time_explosion_s: float | None,
             config_error: str | None = None
             ) -> tuple[dict[str, Any], dict[tuple[int, int], float]]:
    rows = bracket["all_rows"]
    numeric_fields = ("producer_eta", "producer_tau_eff", "nu")
    field_census = {
        name: Counter(row_float(row, name)[1] for row in rows)
        for name in numeric_fields
    }
    reasons: list[str] = []
    if time_explosion_s is None:
        reasons.append("MODEL_CONFIG_PARSE_FAILURE")
    if any(field_census[name].get("valid", 0) != len(rows) for name in numeric_fields):
        reasons.append("W4_FIELD_CENSUS_FAILURE")

    indexed: dict[int, dict[int, dict[str, str]]] = {}
    pairing_issues: list[dict[str, Any]] = []
    for iteration in (0, 1):
        table, issues = index_rows(bracket["iterations"].get(iteration, []))
        indexed[iteration] = table
        pairing_issues.extend({"iteration": iteration, **issue} for issue in issues)
    line_zero = set(indexed[0])
    line_one = set(indexed[1])
    missing_in_zero = sorted(line_one - line_zero)
    missing_in_one = sorted(line_zero - line_one)
    if pairing_issues or len(line_zero & line_one) != EXPECTED_ROWS_PER_ITERATION:
        reasons.append("PAIRING_FAILURE")

    common_errors: list[float] = []
    stability_errors: list[float] = []
    thermal_errors: list[float] = []
    measured_by_row: dict[tuple[int, int], float] = {}
    domain_failures: list[dict[str, Any]] = []
    evaluated_rows = 0
    evaluated_pairs = 0

    if time_explosion_s is not None:
        row_ratios: dict[tuple[int, int], float] = {}
        for iteration, table in indexed.items():
            for line_id, row in table.items():
                eta, eta_status = row_float(row, "producer_eta")
                tau, tau_status = row_float(row, "producer_tau_eff")
                nu, nu_status = row_float(row, "nu")
                if (eta_status, tau_status, nu_status) != ("valid", "valid", "valid"):
                    continue
                assert eta is not None and tau is not None and nu is not None
                if eta <= 0.0 or tau == 0.0 or nu <= 0.0:
                    domain_failures.append({
                        "iteration": iteration,
                        "line": line_id,
                        "eta": eta,
                        "tau": tau,
                        "nu": nu,
                    })
                    continue
                try:
                    b_nu = planck_nu(nu, TEMPERATURE_K)
                except ValueError as exc:
                    domain_failures.append({
                        "iteration": iteration,
                        "line": line_id,
                        "error": str(exc),
                    })
                    continue
                chi = tau * nu / (C_LIGHT * time_explosion_s)
                source_over_b = (eta / chi) / b_nu
                if not math.isfinite(source_over_b) or source_over_b <= 0.0:
                    domain_failures.append({
                        "iteration": iteration,
                        "line": line_id,
                        "source_over_b": source_over_b,
                    })
                    continue
                row_ratios[(iteration, line_id)] = source_over_b
                thermal_errors.append(abs(source_over_b - 1.0))
                evaluated_rows += 1
                measured_by_row[(iteration, line_id)] = source_over_b

        for line_id in sorted(line_zero & line_one):
            row0 = indexed[0][line_id]
            row1 = indexed[1][line_id]
            eta0, eta0_status = row_float(row0, "producer_eta")
            eta1, eta1_status = row_float(row1, "producer_eta")
            tau0, tau0_status = row_float(row0, "producer_tau_eff")
            tau1, tau1_status = row_float(row1, "producer_tau_eff")
            if (eta0_status, eta1_status, tau0_status, tau1_status) != (
                "valid", "valid", "valid", "valid"
            ):
                continue
            assert eta0 is not None and eta1 is not None
            assert tau0 is not None and tau1 is not None
            if eta0 == 0.0 or tau0 == 0.0 or tau1 == 0.0:
                continue
            common = abs((eta1 / eta0) / (tau1 / tau0) - 1.0)
            ratio0 = row_ratios.get((0, line_id))
            ratio1 = row_ratios.get((1, line_id))
            if ratio0 is None or ratio1 is None:
                continue
            stability = abs(ratio1 / ratio0 - 1.0)
            if not math.isfinite(common) or not math.isfinite(stability):
                domain_failures.append({
                    "line": line_id, "error": "NONFINITE_PAIR_METRIC"
                })
                continue
            common_errors.append(common)
            stability_errors.append(stability)
            evaluated_pairs += 1

    if domain_failures:
        reasons.append("W4_DOMAIN_FAILURE")
    if evaluated_rows != EXPECTED_ROWS or evaluated_pairs != EXPECTED_ROWS_PER_ITERATION:
        reasons.append("W4_CENSUS_INCOMPLETE")
    max_common = max(common_errors) if common_errors else None
    max_stability = max(stability_errors) if stability_errors else None
    max_thermal = max(thermal_errors) if thermal_errors else None
    if max_common is not None and max_common > COMMON_MODE_TOLERANCE:
        reasons.append("COMMON_MODE_IDENTITY_FAIL")
    if max_stability is not None and max_stability > SOURCE_STABILITY_TOLERANCE:
        reasons.append("SOURCE_RATIO_STABILITY_FAIL")
    if max_thermal is not None and max_thermal > THERMAL_SOURCE_TOLERANCE:
        reasons.append("THERMAL_SOURCE_BOUND_FAIL")
    return gate(
        "PASS" if not reasons else "FAIL",
        reasons,
        calculation=(
            "S_over_B=(producer_eta/(producer_tau_eff*nu/(c*time_explosion_s)))"
            "/Planck_nu(nu,10020K)"
        ),
        relative_stability_formula="abs((S_over_B_iter1/S_over_B_iter0)-1)",
        constants={
            "h_planck_erg_s": H_PLANCK,
            "c_light_cm_s": C_LIGHT,
            "k_boltzmann_erg_K": K_BOLTZMANN,
            "temperature_K": TEMPERATURE_K,
            "time_explosion_s": time_explosion_s,
        },
        thresholds={
            "common_mode": COMMON_MODE_TOLERANCE,
            "source_ratio_stability": SOURCE_STABILITY_TOLERANCE,
            "thermal_source": THERMAL_SOURCE_TOLERANCE,
        },
        maxima={
            "common_mode": max_common,
            "source_ratio_stability": max_stability,
            "thermal_source": max_thermal,
        },
        field_census={name: dict(counts) for name, counts in field_census.items()},
        pair_census={
            "iter0_unique_lines": len(line_zero),
            "iter1_unique_lines": len(line_one),
            "paired_lines": len(line_zero & line_one),
            "evaluated_pairs": evaluated_pairs,
            "missing_in_iter0_count": len(missing_in_zero),
            "missing_in_iter1_count": len(missing_in_one),
            "missing_in_iter0_examples": missing_in_zero[:20],
            "missing_in_iter1_examples": missing_in_one[:20],
            "pairing_issues": pairing_issues[:20],
        },
        calculation_census={
            "rows_total": len(rows),
            "rows_evaluated": evaluated_rows,
            "rows_unevaluated": len(rows) - evaluated_rows,
            "domain_failure_count": len(domain_failures),
            "domain_failure_examples": domain_failures[:20],
        },
        config_error=config_error,
    ), measured_by_row


def locate_level_energy(group: Any) -> tuple[Any | None, str | None, str | None]:
    candidates = (
        ("levels/energy_eV", "eV"),
        ("levels/energies_eV", "eV"),
        ("level_energy_eV", "eV"),
        ("level_energies_eV", "eV"),
        ("energies_eV", "eV"),
        ("energy_eV", "eV"),
        ("levels/energy_erg", "erg"),
        ("level_energy_erg", "erg"),
    )
    for name, unit in candidates:
        if name in group:
            dataset = group[name]
            if getattr(dataset, "ndim", None) == 1:
                return dataset, name, unit
    return None, None, None


def observe_w5(h5_path: Path, rows: list[dict[str, str]],
               measured_by_row: dict[tuple[int, int], float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "TOOLING_LIMIT",
        "gate_weight": 0,
        "path": str(h5_path),
        "rows_total": len(rows),
        "rows_predicted": 0,
        "rows_unpredicted": len(rows),
        "reasons": [],
        "dataset_census": {},
        "row_failure_census": {},
        "comparison": None,
    }
    try:
        import h5py
    except Exception as exc:  # observation only: environment tooling can vary
        result["reasons"].append(f"H5PY_UNAVAILABLE: {type(exc).__name__}: {exc}")
        return result
    if not h5_path.is_file():
        result["reasons"].append("H5_DECK_UNAVAILABLE")
        return result

    failures: Counter[str] = Counter()
    residuals: list[float] = []
    relative_residuals: list[float] = []
    samples: list[dict[str, Any]] = []
    try:
        with h5py.File(h5_path, "r") as deck:
            energy_tables: dict[tuple[int, int], tuple[Any, str]] = {}
            for z in sorted(TARGET_Z):
                key = f"Z{z}_ion3"
                if key not in deck:
                    result["dataset_census"][key] = "GROUP_MISSING"
                    continue
                dataset, dataset_name, unit = locate_level_energy(deck[key])
                if dataset is None or unit is None:
                    result["dataset_census"][key] = "LEVEL_ENERGY_DATASET_NOT_FOUND"
                    continue
                energy_tables[(z, 3)] = (dataset, unit)
                result["dataset_census"][key] = {
                    "dataset": dataset_name,
                    "unit": unit,
                    "count": int(dataset.shape[0]),
                }

            for row in rows:
                z, z_status = row_integer(row, "Z")
                ion, ion_status = row_integer(row, "ion")
                lower, lower_status = row_integer(row, "lower_level")
                upper, upper_status = row_integer(row, "upper_level")
                nu, nu_status = row_float(row, "nu")
                line_id, line_status = row_integer(row, "line")
                iteration, iteration_status = row_integer(row, "_iteration")
                statuses = (
                    z_status, ion_status, lower_status, upper_status,
                    nu_status, line_status, iteration_status,
                )
                if any(status != "valid" for status in statuses):
                    failures["ROW_FIELD_UNAVAILABLE"] += 1
                    continue
                assert z is not None and ion is not None
                assert lower is not None and upper is not None
                assert nu is not None and line_id is not None and iteration is not None
                table = energy_tables.get((z, ion))
                if table is None:
                    failures["LEVEL_ENERGY_DATASET_NOT_FOUND"] += 1
                    continue
                dataset, unit = table
                if lower < 0 or upper < 0 or lower >= len(dataset) or upper >= len(dataset):
                    failures["LEVEL_INDEX_OUT_OF_RANGE"] += 1
                    continue
                lower_energy = float(dataset[lower])
                upper_energy = float(dataset[upper])
                delta = upper_energy - lower_energy
                if unit == "eV":
                    delta *= EV_TO_ERG
                if not math.isfinite(delta) or delta <= 0.0:
                    failures["LEVEL_ENERGY_DOMAIN"] += 1
                    continue
                try:
                    prediction = math.expm1(H_PLANCK * nu / (K_BOLTZMANN * TEMPERATURE_K)) \
                        / math.expm1(delta / (K_BOLTZMANN * TEMPERATURE_K))
                except (OverflowError, ZeroDivisionError):
                    failures["PREDICTION_DOMAIN"] += 1
                    continue
                measured = measured_by_row.get((iteration, line_id))
                if measured is None or not math.isfinite(prediction):
                    failures["MEASUREMENT_OR_PREDICTION_UNAVAILABLE"] += 1
                    continue
                residual = abs(prediction - measured)
                relative = abs(prediction / measured - 1.0)
                residuals.append(residual)
                relative_residuals.append(relative)
                if len(samples) < 20:
                    samples.append({
                        "line": line_id,
                        "Z": z,
                        "ion": ion,
                        "predicted_S_over_B": prediction,
                        "measured_S_over_B": measured,
                        "absolute_residual": residual,
                    })
    except Exception as exc:  # observation only; report, never promote to a gate
        result["reasons"].append(f"H5_PARSE_FAILURE: {type(exc).__name__}: {exc}")
        return result

    result["rows_predicted"] = len(residuals)
    result["rows_unpredicted"] = len(rows) - len(residuals)
    result["row_failure_census"] = dict(failures)
    if residuals:
        result["status"] = "OBSERVED"
        result["comparison"] = {
            "max_absolute_residual": max(residuals),
            "max_relative_residual": max(relative_residuals),
            "samples": samples,
        }
    else:
        result["reasons"].append("LEVEL_ENERGY_SCHEMA_UNAVAILABLE")
    return result


def blocked_gate(reason: str, detail: str | None = None) -> dict[str, Any]:
    return gate("FAIL", [reason], failure_classifications=["BLOCKER"], detail=detail)


def determine_branch(w1: dict[str, Any], w2: dict[str, Any], w3: dict[str, Any],
                     w4: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    for result in (w1, w2, w3):
        classifications = result.get("failure_classifications", [])
        if "BLOCKER" in classifications:
            blockers.extend(result["reasons"])
    if w1["status"] != "PASS":
        blockers.extend(w1["reasons"])
    if w4["status"] != "PASS":
        blockers.extend(w4["reasons"])
    if blockers:
        return "W-D", list(dict.fromkeys(blockers))

    counterexamples: list[str] = []
    if "COUNTEREXAMPLE" in w2.get("failure_classifications", []):
        counterexamples.extend(w2["reasons"])
    mapped = w3.get("mapping", {}).get("mapped")
    if mapped is not None and mapped > 0:
        counterexamples.append("MAPPED_COUNTEREXAMPLE")
    if counterexamples:
        return "W-B", list(dict.fromkeys(counterexamples))
    if all(result["status"] == "PASS" for result in (w1, w2, w3, w4)):
        return "W-A", ["ATTRIBUTION_CLOSED"]
    return "W-D", ["UNCLASSIFIED_GATE_STATE"]


def run_audit(run_root: Path, repo: Path) -> dict[str, Any]:
    input_paths = {
        "run_footer": run_root / "RUN_FOOTER.txt",
        "stdout": run_root / "stdout.log",
        "stderr": run_root / "stderr.log",
        "config": run_root / "input" / "model" / "config.json",
        "h5": run_root / "input" / "model" / "atomic_data_cmfgen.h5",
    }
    texts: dict[str, str | None] = {}
    access_errors: dict[str, str] = {}
    for name in ("run_footer", "stdout", "stderr", "config"):
        text, error = read_text(input_paths[name])
        texts[name] = text
        if error:
            access_errors[name] = error
    source, source_error = git_blob(repo)
    if source_error:
        access_errors["source_blob"] = source_error

    if texts["run_footer"] is None or texts["stdout"] is None:
        w1 = blocked_gate("SEALED_ACCESS_UNAVAILABLE", json.dumps(access_errors))
    else:
        w1 = audit_w1(texts["run_footer"], texts["stdout"])

    base_pairs: set[tuple[int, int]] | None = None
    if source is None:
        w2 = blocked_gate("SOURCE_BLOB_UNAVAILABLE", source_error)
        stage4_pairs = None
    else:
        w2, base_pairs, stage4_pairs = audit_w2(source)

    if texts["stderr"] is None:
        bracket = {"all_rows": [], "iterations": {}, "issues": ["STDERR_UNAVAILABLE"],
                   "markers": [], "pending_count": 0}
        w3 = blocked_gate("SEALED_ACCESS_UNAVAILABLE", access_errors.get("stderr"))
    else:
        bracket = bracket_rows(texts["stderr"])
        w3 = audit_w3(bracket, base_pairs)

    time_value: float | None = None
    time_error: str | None = access_errors.get("config")
    if texts["config"] is not None:
        time_value, time_error = model_time(texts["config"])
    w4, measured = audit_w4(bracket, time_value, time_error)
    w5 = observe_w5(input_paths["h5"], bracket["all_rows"], measured)
    branch, branch_reasons = determine_branch(w1, w2, w3, w4)
    return {
        "schema": "DET_A207_WIRING_L6_ATTRIBUTION_V1",
        "audit": A207_WIRING_L6_ATTRIBUTION,
        "source_blob": f"{SOURCE_REVISION}:{SOURCE_PATH}",
        "run_root": str(run_root),
        "sealed_access_mode": "READ_ONLY",
        "access_errors": access_errors,
        "gates": {"W1": w1, "W2": w2, "W3": w3, "W4": w4},
        "W5_observation": w5,
        "f_mapped": w3.get("mapping", {}).get("f_mapped"),
        "branch": branch,
        "branch_reasons": branch_reasons,
        "physical_values_modified": False,
        "oracle_values_used": False,
        "value_clamping_used": False,
    }


def synthetic_source() -> str:
    return """
static const int NLTE_TARGET_Z[] = {26,26,27,27,28,28};
static const int NLTE_TARGET_ION[] = {1,2,1,2,1,2};
static const int NLTE_TARGET_Z4[] = {26,26,26,27,27,27,28,28,28};
static const int NLTE_TARGET_ION4[] = {1,2,3,1,2,3,1,2,3};
static void nlte_update_tau_sobolev_with_authority(void) {
    if (element_inactive || !authority.mapped)
        continue;
}
static int a209_upper_population_for_tau(void) {
    if(!use_nlte&&lte_level_density){ return 1; }
    return 0;
}
static int nlte_population_candidate_produce_tau_source(void) {
    compute_tau_sobolev(&candidate->atom, &candidate->plasma, &candidate->opacity,
                        time_explosion);
    return 0;
}
""".lstrip()


def synthetic_footer() -> str:
    return "LUMINA_A210_LINE_SATURATION_TARGET_ION=3\n"


def synthetic_stdout() -> str:
    lines = []
    for z in sorted(TARGET_Z):
        lines.extend((f"    Z={z} ion=1: 10 levels", f"    Z={z} ion=2: 10 levels"))
    lines.extend((
        "  [NLTE] Super-levels: off (identity) (60 FL -> 60 SL across ions)",
        "  [NLTE] Lines mapped to NLTE ions: 1777859 / 2588798",
    ))
    return "\n".join(lines) + "\n"


def synthetic_stderr(time_explosion_s: float = 1683072.0) -> str:
    lines: list[str] = []
    for iteration in (0, 1):
        population_scale = 1.0 if iteration == 0 else 1.004
        for offset in range(EXPECTED_ROWS_PER_ITERATION):
            line_id = 100000 + offset
            z = (26, 27, 28)[offset % 3]
            nu = 5.0e14 + offset * 1.0e10
            tau0 = 0.2 + offset * 1.0e-4
            tau = tau0 * population_scale
            eta = planck_nu(nu, TEMPERATURE_K) * tau * nu \
                / (C_LIGHT * time_explosion_s)
            lines.append(
                f"{ROW_PREFIX} phase=REQUESTED_TE shell=0 line={line_id} "
                f"Z={z} ion=3 lower_level={offset % 40} upper_level={(offset % 40) + 1} "
                f"nu={nu:.17g} producer_eta={eta:.17g} "
                f"producer_tau_eff={tau:.17g} producer_raw_defined=1"
            )
        lines.append(f"{SUMMARY_PREFIX} complete=1")
        lines.append(
            f"{R7_PREFIX} lane=DET iter={iteration} phase=A2-10 "
            f"te_generation={iteration + 1}->{iteration + 2}"
        )
    return "\n".join(lines) + "\n"


def replace_first_eta(text: str, factor: float) -> str:
    marker = f"{R7_PREFIX} lane=DET iter=0"
    split = text.find(marker)
    if split < 0:
        raise RuntimeError("SELFTEST_ITER0_MARKER_MISSING")
    match = re.search(r"producer_eta=([^\s]+)", text[split:])
    if not match:
        raise RuntimeError("SELFTEST_ETA_MISSING")
    start = split + match.start(1)
    end = split + match.end(1)
    return text[:start] + f"{float(text[start:end]) * factor:.17g}" + text[end:]


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="a207_wiring_selftest_") as temporary:
        scratch = Path(temporary)
        footer_path = scratch / "RUN_FOOTER.txt"
        stdout_path = scratch / "stdout.log"
        stderr_path = scratch / "stderr.log"
        source_path = scratch / "lumina_plasma.c"
        footer_path.write_text(synthetic_footer(), encoding="utf-8")
        stdout_clean = synthetic_stdout()
        stdout_path.write_text(stdout_clean, encoding="utf-8")
        stderr_clean = synthetic_stderr()
        stderr_path.write_text(stderr_clean, encoding="utf-8")
        source_clean = synthetic_source()
        source_path.write_text(source_clean, encoding="utf-8")

        stdout_path.write_text(
            stdout_clean + "    Z=26 ion=3: 100 levels\n", encoding="utf-8"
        )
        w1_bad = audit_w1(footer_path.read_text(), stdout_path.read_text())
        require("COVERAGE_SLOT_PRESENT" in w1_bad["reasons"], "NC-W1_NOT_DETECTED")
        print("NC-W1 inject=Z26_ION3_SLOT status=FAIL reason=COVERAGE_SLOT_PRESENT")
        stdout_path.write_text(stdout_clean, encoding="utf-8")
        require(audit_w1(footer_path.read_text(), stdout_path.read_text())["status"] == "PASS",
                "NC-W1_RESTORE_FAILED")
        print("NC-W1 remove=Z26_ION3_SLOT status=PASS")

        w2_clean, base_pairs, stage4_pairs = audit_w2(source_path.read_text())
        require(w2_clean["status"] == "PASS" and base_pairs is not None
                and stage4_pairs is not None, "NC-W2_SOURCE_FIXTURE_INVALID")
        discriminator_rows = [
            {"line": str(index), "Z": str((26, 27, 28)[index % 3]), "ion": "3"}
            for index in range(EXPECTED_ROWS_PER_ITERATION)
        ]
        stage4_mapping = mapping_census(discriminator_rows, stage4_pairs)
        require(stage4_mapping["mapped"] == EXPECTED_ROWS_PER_ITERATION,
                "NC-W2A_STAGE4_NOT_MAPPED")
        print("NC-W2a inject=NLTE_TARGET_ION4 status=FAIL "
              "reason=MAPPED_COUNTEREXAMPLE mapped=297/297 branch=W-B")
        base_mapping = mapping_census(discriminator_rows, base_pairs)
        require(base_mapping["mapped"] == 0, "NC-W2A_BASE_NOT_UNMAPPED")
        print("NC-W2a remove=NLTE_TARGET_ION status=PASS mapped=0/297 branch=W-A")

        source_bad = source_clean.replace(
            "if (element_inactive || !authority.mapped)",
            "if (element_inactive)",
            1,
        )
        source_path.write_text(source_bad, encoding="utf-8")
        w2_bad, _, _ = audit_w2(source_path.read_text())
        require("ANCHOR_MISSING" in w2_bad["reasons"], "NC-W2B_NOT_DETECTED")
        print("NC-W2b inject=DELETE_UNMAPPED_SKIP status=FAIL reason=ANCHOR_MISSING")
        source_path.write_text(source_clean, encoding="utf-8")
        require(audit_w2(source_path.read_text())[0]["status"] == "PASS",
                "NC-W2B_RESTORE_FAILED")
        print("NC-W2b remove=RESTORE_UNMAPPED_SKIP status=PASS")

        forged = stderr_clean.replace("ion=3", "ion=2", 1)
        stderr_path.write_text(forged, encoding="utf-8")
        w3_bad = audit_w3(bracket_rows(stderr_path.read_text()), base_pairs)
        require(w3_bad["mapping"]["mapped"] == 1
                and "MAPPED_COUNTEREXAMPLE" in w3_bad["reasons"], "NC-W3_NOT_DETECTED")
        print("NC-W3 inject=ONE_ION2_ROW status=FAIL reason=MAPPED_COUNTEREXAMPLE "
              "f_mapped=1/594 branch=W-B")
        stderr_path.write_text(stderr_clean, encoding="utf-8")
        w3_clean = audit_w3(bracket_rows(stderr_path.read_text()), base_pairs)
        require(w3_clean["status"] == "PASS", "NC-W3_RESTORE_FAILED")
        print("NC-W3 remove=RESTORE_ION3 status=PASS f_mapped=0/594 branch=W-A")

        perturbed = replace_first_eta(stderr_clean, 1.0 + 1.0e-9)
        stderr_path.write_text(perturbed, encoding="utf-8")
        w4_bad, _ = audit_w4(bracket_rows(stderr_path.read_text()), 1683072.0)
        require("COMMON_MODE_IDENTITY_FAIL" in w4_bad["reasons"]
                and "SOURCE_RATIO_STABILITY_FAIL" in w4_bad["reasons"],
                "NC-W4_NOT_DETECTED")
        print("NC-W4 inject=ONE_ETA_REL_PLUS_1E-9 status=FAIL "
              "reason=COMMON_MODE_IDENTITY_FAIL,SOURCE_RATIO_STABILITY_FAIL")
        stderr_path.write_text(stderr_clean, encoding="utf-8")
        w4_clean, _ = audit_w4(bracket_rows(stderr_path.read_text()), 1683072.0)
        require(w4_clean["status"] == "PASS", "NC-W4_RESTORE_FAILED")
        print("NC-W4 remove=RESTORE_ETA status=PASS")

    print("DET_A207_WIRING_SELFTEST_PASS NC-W1,NC-W2a,NC-W2b,NC-W3,NC-W4=PASS")
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run all five scratch negative controls")
    parser.add_argument("--run-root", type=Path, help="sealed L6 run root (opened read-only)")
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1],
        help="repository used only for git show 8c2aace:src/lumina_plasma.c",
    )
    parser.add_argument("--output", type=Path, help="verdict JSON; stdout when omitted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    if args.run_root is None:
        print("DET_A207_WIRING_FAIL reason=RUN_ROOT_REQUIRED", file=sys.stderr)
        return 4
    if args.output is not None and is_below(args.output, args.run_root):
        print("DET_A207_WIRING_FAIL reason=SEALED_WRITE_FORBIDDEN", file=sys.stderr)
        return 4
    report = run_audit(args.run_root, args.repo)
    if args.output is None:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
    else:
        atomic_write_json(args.output, report)
    for name, result in report["gates"].items():
        reasons = ",".join(result["reasons"]) or "NONE"
        print(f"{name} status={result['status']} reasons={reasons}", file=sys.stderr)
    w5 = report["W5_observation"]
    print(
        f"W5 status={w5['status']} gate_weight=0 predicted="
        f"{w5['rows_predicted']}/{w5['rows_total']}",
        file=sys.stderr,
    )
    print(
        f"DET_A207_WIRING branch={report['branch']} "
        f"f_mapped={report['f_mapped']} reasons={','.join(report['branch_reasons'])}",
        file=sys.stderr,
    )
    return {"W-A": 0, "W-B": 3, "W-D": 4}[report["branch"]]


if __name__ == "__main__":
    raise SystemExit(main())
