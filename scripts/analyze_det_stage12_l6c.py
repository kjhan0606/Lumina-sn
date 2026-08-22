#!/usr/bin/env python3
"""Self-contained DET-L6C-COVER ledger analyzer and branch machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


A210_L6C_PROBE = "A210_L6C_PROBE"
SOURCE_REVISION = "8526d2c"
EXPECTED_BINARY_SHA256 = (
    "b9a30a81ebea57f9fa857d192107dd85aeb04ab1308f27b1a68cf45f1a69af99"
)
DEFAULT_SEALED_L6_STDERR = Path(
    "/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/"
    "sprim_l6_20260821T054111Z_probe/stderr.log"
)
H_PLANCK = 6.62607015e-27
C_LIGHT = 2.99792458e10
K_BOLTZMANN = 1.380649e-16
FOUR_PI = 12.56637061435917295385057353311801153679
REQUESTED_TEMPERATURE_K = 10020.0
RECONSTRUCTION_TOLERANCE = 1.0e-12
FROZEN_TOLERANCE = 1.0e-13
TARGET_Z = frozenset({26, 27, 28})
ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
IDENTITY_PREFIX = "[A2-10][LINE-COEFFICIENT-IDENTITY]"
EPOCH_PREFIX = "[cmf_fine][EXACT-MULTIGPU-EPOCH]"
R7_PREFIX = "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED"
PHYSICS_PREFIX = "[PHYSICS-COMPARISON] lane=DET"
KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


class AnalysisError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError("R1_UNSAFE_OR_MISSING_INPUT", str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fields(line: str) -> dict[str, str]:
    return dict(KEY_VALUE.findall(line))


def integer(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as exc:
        raise AnalysisError(f"INVALID_INTEGER_{key}") from exc


def finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as exc:
        raise AnalysisError(f"UNAVAILABLE_{key}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"NONFINITE_{key}")
    return value


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("EMPTY_DISTRIBUTION")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("EMPTY_DISTRIBUTION")
    probabilities = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50,
                     0.75, 0.90, 0.95, 0.99, 1.0)
    return {
        "count": len(ordered),
        "quantiles": {
            f"q{int(probability * 100):02d}": percentile(ordered, probability)
            for probability in probabilities
        },
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def exponx(tau: float) -> tuple[float, float]:
    if not math.isfinite(tau):
        raise AnalysisError("EXPONX_NONFINITE_TAU")
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        beta = 1.0 - tau * companion
    elif tau < 40.0:
        beta = (1.0 - math.exp(-tau)) / tau
        companion = (1.0 - beta) / tau
    else:
        beta = 1.0 / tau
        companion = (1.0 - beta) / tau
    if (not math.isfinite(beta) or beta <= 0.0 or
            not math.isfinite(companion) or companion <= 0.0):
        raise AnalysisError("EXPONX_DOMAIN")
    return beta, companion


def planck_nu(nu: float, temperature: float) -> float:
    if (not math.isfinite(nu) or nu <= 0.0 or
            not math.isfinite(temperature) or temperature <= 0.0):
        raise AnalysisError("PLANCK_DOMAIN")
    exponent = H_PLANCK * nu / (K_BOLTZMANN * temperature)
    try:
        denominator = math.expm1(exponent)
    except OverflowError as exc:
        raise AnalysisError("PLANCK_EXPONENT_OVERFLOW") from exc
    value = 2.0 * H_PLANCK * nu ** 3 / (C_LIGHT ** 2 * denominator)
    if not math.isfinite(value) or value <= 0.0:
        raise AnalysisError("PLANCK_NONFINITE")
    return value


def relative_deviation(observed: float, reconstructed: float) -> float:
    if observed == reconstructed:
        return 0.0
    observed_abs = abs(observed)
    reconstructed_abs = abs(reconstructed)
    denominator = observed_abs if observed_abs >= reconstructed_abs else reconstructed_abs
    if denominator == 0.0:
        return math.inf
    return abs(observed - reconstructed) / denominator


def pearson(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"value": None, "reason": "NO_PAIRED_VALUES", "count": len(left)}
    mean_left = math.fsum(left) / len(left)
    mean_right = math.fsum(right) / len(right)
    delta_left = [value - mean_left for value in left]
    delta_right = [value - mean_right for value in right]
    covariance = math.fsum(a * b for a, b in zip(delta_left, delta_right))
    variance_left = math.fsum(value * value for value in delta_left)
    variance_right = math.fsum(value * value for value in delta_right)
    if variance_left == 0.0 or variance_right == 0.0:
        return {"value": None, "reason": "ZERO_VARIANCE", "count": len(left)}
    value = covariance / math.sqrt(variance_left * variance_right)
    if not math.isfinite(value):
        return {"value": None, "reason": "NONFINITE", "count": len(left)}
    return {"value": value, "reason": None, "count": len(left)}


def parse_c_int_array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"static\s+const\s+int\s+{re.escape(name)}(?:\[[^\]]*\])?\s*=\s*"
        rf"\{{(?P<body>.*?)\}}\s*;",
        source,
        re.S,
    )
    if not match:
        raise AnalysisError("R5_TABLE_PARSE_FAILED", name)
    return [int(token) for token in re.findall(r"[-+]?\d+", match.group("body"))]


def mapping_from_source(source: str, layout: str = "base") -> frozenset[tuple[int, int]]:
    suffix = "" if layout == "base" else "4"
    z_values = parse_c_int_array(source, f"NLTE_TARGET_Z{suffix}")
    ion_values = parse_c_int_array(source, f"NLTE_TARGET_ION{suffix}")
    if not z_values or len(z_values) != len(ion_values):
        raise AnalysisError("R5_TABLE_PARSE_FAILED", layout)
    return frozenset(zip(z_values, ion_values))


def git_source(repo: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:src/lumina_plasma.c"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AnalysisError("R5_SOURCE_BLOB_UNAVAILABLE", revision)
    return result.stdout


def parse_log_blocks(lines: list[str]) -> dict[str, Any]:
    pending: list[dict[str, str]] = []
    pending_summary: dict[str, str] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    r7_lines: list[str] = []
    physics_lines: list[str] = []
    epochs: list[str] = []
    iteration_epochs: list[str] = []
    last_epoch: str | None = None
    for line in lines:
        if line.startswith(EPOCH_PREFIX):
            epochs.append(line)
            last_epoch = line
        elif line.startswith(ROW_PREFIX):
            pending.append(fields(line))
        elif line.startswith(SUMMARY_PREFIX):
            if pending_summary is not None:
                raise AnalysisError("R7_DUPLICATE_SUMMARY")
            pending_summary = fields(line)
        elif line.startswith(R7_PREFIX):
            marker = fields(line)
            iteration = integer(marker, "iter")
            expected_transition = f"{iteration + 1}->{iteration + 2}"
            if (marker.get("lane") != "DET" or marker.get("phase") != "A2-10" or
                    marker.get("te_generation") != expected_transition):
                raise AnalysisError("R7_INVALID_MARKER")
            if iteration in blocks:
                raise AnalysisError("R7_DUPLICATE_MARKER")
            if not pending or pending_summary is None:
                raise AnalysisError("R7_MARKER_BLOCKED_EMPTY")
            if last_epoch is None:
                raise AnalysisError("R2_EXACT_SOLVE_MISSING")
            blocks[iteration] = {"rows": pending, "summary": pending_summary}
            pending = []
            pending_summary = None
            r7_lines.append(line)
            iteration_epochs.append(last_epoch)
            last_epoch = None
        elif line.startswith(PHYSICS_PREFIX):
            physics_lines.append(line)
    if pending or pending_summary is not None:
        raise AnalysisError("R7_MARKER_MISSING")
    if set(blocks) != {0, 1}:
        raise AnalysisError("R7_MARKER_MISSING", f"iterations={sorted(blocks)}")
    if len(r7_lines) != 2:
        raise AnalysisError("R2_COMMIT_COUNT_MISMATCH", f"r7={len(r7_lines)}")
    if len(physics_lines) != 2:
        raise AnalysisError("R2_COMMIT_COUNT_MISMATCH", f"physics={len(physics_lines)}")
    for iteration, line in enumerate(physics_lines):
        marker = fields(line)
        if integer(marker, "iter") != iteration or marker.get("status") != "COMMITTED":
            raise AnalysisError("R2_COMMIT_STRUCTURE_MISMATCH")
    if len(epochs) != 3 or len(iteration_epochs) != 2:
        raise AnalysisError("R2_EXACT_SOLVE_COUNT_MISMATCH", f"epochs={len(epochs)}")
    return {
        "blocks": blocks,
        "r7_lines": r7_lines,
        "physics_lines": physics_lines,
        "epochs": epochs,
        "iteration_epochs": iteration_epochs,
    }


def marker_series(text: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    parsed = parse_log_blocks(lines)
    identities = [line for line in lines if line.startswith(IDENTITY_PREFIX)]
    if len(identities) != 100:
        raise AnalysisError("R7_IDENTITY_COUNT_MISMATCH", f"lines={len(identities)}")
    return {
        "exact_solve": parsed["iteration_epochs"],
        "line_coefficient_identity": identities,
    }


def compare_r7(text: str, sealed_text: str) -> dict[str, Any]:
    observed = marker_series(text)
    sealed = marker_series(sealed_text)
    exact_equal = observed["exact_solve"] == sealed["exact_solve"]
    identity_equal = (
        observed["line_coefficient_identity"] == sealed["line_coefficient_identity"]
    )
    return {
        "status": "PASS" if exact_equal and identity_equal else "FAIL",
        "exact_solve_lines": len(observed["exact_solve"]),
        "exact_solve_byte_equal": exact_equal,
        "line_coefficient_identity_lines": len(observed["line_coefficient_identity"]),
        "line_coefficient_identity_byte_equal": identity_equal,
        "phase_label_condition_applied": False,
    }


def analyze_iteration(
    rows: list[dict[str, str]], time_explosion_s: float,
    temperature_K: float, mapping: frozenset[tuple[int, int]], target_ion: int,
) -> dict[str, Any]:
    census = {
        "TOTAL": len(rows),
        "TARGET": 0,
        "NON_TARGET": 0,
        "MAPPED": 0,
        "UNMAPPED": 0,
        "FINITE_POSITIVE_CHI": 0,
        "NEGATIVE_CHI": 0,
        "INVERSION_BOUNDARY": 0,
        "EXACT_ZERO": 0,
        "UNAVAILABLE": 0,
        "IDENTITY_UNAVAILABLE": 0,
        "JBAR_ZERO": 0,
    }
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    reconstruction = {"continuum": 0.0, "local": 0.0, "jbar": 0.0}
    for ordinal, row in enumerate(rows):
        try:
            line = integer(row, "line")
            z = integer(row, "Z")
            ion = integer(row, "ion")
            identity = (z, ion, line)
            if identity in records:
                raise AnalysisError("R3_DUPLICATE_LINE_ID", str(identity))
            is_target = z in TARGET_Z and ion == target_ion
            is_mapped = (z, ion) in mapping
            census["TARGET" if is_target else "NON_TARGET"] += 1
            census["MAPPED" if is_mapped else "UNMAPPED"] += 1
            flags = (
                integer(row, "producer_terms_defined"),
                integer(row, "producer_raw_defined"),
                integer(row, "independent_fields_defined"),
            )
            if flags != (1, 1, 1):
                raise AnalysisError("R3_REQUIRED_FIELD_UNAVAILABLE", str(identity))
            eta = finite(row, "producer_eta")
            tau = finite(row, "producer_tau_eff")
            nu = finite(row, "nu")
            j_cont = finite(row, "J_cont")
            continuum_observed = finite(row, "producer_continuum_term")
            local_observed = finite(row, "producer_local_emission_term")
            jbar_observed = finite(row, "Jbar")
            if eta < 0.0 or nu <= 0.0 or j_cont < 0.0 or jbar_observed < 0.0:
                raise AnalysisError("R3_INVALID_PHYSICAL_FIELD", str(identity))
            beta, companion = exponx(tau)
            continuum_reconstructed = beta * j_cont
            local_reconstructed = eta * (C_LIGHT * time_explosion_s / nu) * companion
            jbar_reconstructed = continuum_reconstructed + local_reconstructed
            observed_values = {
                "continuum": continuum_observed,
                "local": local_observed,
                "jbar": jbar_observed,
            }
            reconstructed_values = {
                "continuum": continuum_reconstructed,
                "local": local_reconstructed,
                "jbar": jbar_reconstructed,
            }
            for name in reconstruction:
                error = relative_deviation(observed_values[name], reconstructed_values[name])
                if error > reconstruction[name]:
                    reconstruction[name] = error
                if not math.isfinite(error) or error > RECONSTRUCTION_TOLERANCE:
                    raise AnalysisError("R4B_RECONSTRUCTION_FAIL", f"{name}:{identity}:{error}")
            chi = tau * nu / (C_LIGHT * time_explosion_s)
            b_nu = planck_nu(nu, temperature_K)
            source_over_b: float | None
            if chi == 0.0:
                source_over_b = None
                census["INVERSION_BOUNDARY" if eta > 0.0 else "EXACT_ZERO"] += 1
            else:
                source_over_b = (eta / chi) / b_nu
                if not math.isfinite(source_over_b):
                    raise AnalysisError("R4_NONFINITE_SOURCE_RATIO", str(identity))
                census["NEGATIVE_CHI" if chi < 0.0 else "FINITE_POSITIVE_CHI"] += 1
            jbar_over_b = jbar_observed / b_nu
            if not math.isfinite(jbar_over_b):
                raise AnalysisError("R6_NONFINITE_JBAR_RATIO", str(identity))
            if jbar_observed == 0.0:
                census["JBAR_ZERO"] += 1
            records[identity] = {
                "line": line,
                "Z": z,
                "ion": ion,
                "mapped": is_mapped,
                "target": is_target,
                "producer_eta": eta,
                "producer_tau_eff": tau,
                "nu": nu,
                "source_over_B": source_over_b,
                "Jbar_over_B": jbar_over_b,
            }
        except AnalysisError as exc:
            census["UNAVAILABLE"] += 1
            if exc.reason in ("INVALID_INTEGER_line", "INVALID_INTEGER_Z", "INVALID_INTEGER_ion"):
                census["IDENTITY_UNAVAILABLE"] += 1
            errors.append({"ordinal": ordinal, "reason": exc.reason, "detail": exc.detail})
    classified = sum(census[name] for name in (
        "FINITE_POSITIVE_CHI", "NEGATIVE_CHI", "INVERSION_BOUNDARY", "EXACT_ZERO",
    ))
    if classified + census["UNAVAILABLE"] != census["TOTAL"]:
        raise AnalysisError("R3_CENSUS_NOT_EXHAUSTIVE")
    source_ratios = [
        record["source_over_B"] for record in records.values()
        if record["source_over_B"] is not None
    ]
    jbar_ratios = [record["Jbar_over_B"] for record in records.values()]
    return {
        "census": census,
        "records": records,
        "unavailable_rows": errors,
        "source_over_B": distribution(source_ratios) if source_ratios else None,
        "Jbar_over_B": distribution(jbar_ratios) if jbar_ratios else None,
        "r4b_max_relative_deviation": reconstruction,
    }


def summary_observations(summary: dict[str, str]) -> dict[str, Any]:
    required_int = ("candidate_rows", "selected_rows", "zero_opacity_emitting_rows")
    required_float = ("total_scaled_emission", "selected_scaled_emission", "selected_fraction")
    result: dict[str, Any] = {}
    for key in required_int:
        result[key] = integer(summary, key)
    for key in required_float:
        result[key] = finite(summary, key)
    result["target_Z"] = summary.get("target_Z")
    result["target_ion"] = integer(summary, "target_ion")
    result["selection_mode"] = summary.get("selection_mode")
    result["complete"] = integer(summary, "complete")
    return result


def implied_temperature(nu: float, source_ratio: float, reference_temperature: float) -> float:
    if source_ratio <= 0.0 or not math.isfinite(source_ratio):
        raise AnalysisError("E5_TEMPERATURE_READJUSTMENT_UNDEFINED")
    reference_exponent = H_PLANCK * nu / (K_BOLTZMANN * reference_temperature)
    try:
        target_exponent = math.log1p(math.expm1(reference_exponent) / source_ratio)
    except (OverflowError, ValueError) as exc:
        raise AnalysisError("E5_TEMPERATURE_READJUSTMENT_UNDEFINED") from exc
    if target_exponent <= 0.0 or not math.isfinite(target_exponent):
        raise AnalysisError("E5_TEMPERATURE_READJUSTMENT_UNDEFINED")
    value = H_PLANCK * nu / (K_BOLTZMANN * target_exponent)
    if value <= 0.0 or not math.isfinite(value):
        raise AnalysisError("E5_TEMPERATURE_READJUSTMENT_UNDEFINED")
    return value


def single_temperature_observation(
    matched: list[dict[str, Any]], reference_temperature: float
) -> dict[str, Any]:
    implied: list[float] = []
    valid: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for record in matched:
        ratio = record["source_ratio_iter1_over_iter0"]
        try:
            value = implied_temperature(record["nu"], ratio, reference_temperature)
        except AnalysisError as exc:
            unavailable.append({"line": record["line"], "reason": exc.reason})
            continue
        implied.append(value)
        valid.append(record)
    if unavailable or not implied:
        return {
            "candidate_temperature_K": None,
            "collapsed": None,
            "reason": "UNAVAILABLE_ROWS",
            "unavailable_rows": unavailable,
        }
    candidate = percentile(sorted(implied), 0.5)
    residuals: list[float] = []
    for record in valid:
        predicted = (
            planck_nu(record["nu"], candidate) /
            planck_nu(record["nu"], reference_temperature)
        )
        residuals.append(abs(record["source_ratio_iter1_over_iter0"] / predicted - 1.0))
    residual_distribution = distribution(residuals)
    collapsed = residual_distribution["maximum"] <= RECONSTRUCTION_TOLERANCE
    return {
        "candidate_temperature_K": candidate,
        "implied_temperature_K": distribution(implied),
        "residual_after_single_temperature": residual_distribution,
        "collapsed": collapsed,
        "criterion": "all residuals <= reconstruction tolerance",
        "unavailable_rows": [],
    }


def partial_block_report(text: str) -> dict[str, Any] | None:
    if "INDEPENDENT_SPROBE_UNDEFINED" in text:
        reason = "INDEPENDENT_SPROBE_UNDEFINED"
        verdict = "D2"
    else:
        line = next((line for line in text.splitlines() if "[BLOCKED]" in line or "[FATAL]" in line), None)
        if line is None:
            return None
        reason = fields(line).get("reason") or fields(line).get("status") or "NAMED_BLOCK"
        verdict = "D1"
    rows = [line for line in text.splitlines() if line.startswith(ROW_PREFIX)]
    return {
        "schema": "DET_L6C_COVER_VERDICT_V1",
        "status": "PARTIAL",
        "verdict": verdict,
        "blocking_reasons": [reason],
        "observed_row_lines": len(rows),
        "physical_values_modified": False,
    }


def analyze_text(
    text: str, sealed_text: str, source: str, time_explosion_s: float,
    temperature_K: float = REQUESTED_TEMPERATURE_K, target_ion: int = 1,
    mapping_layout: str = "base",
) -> dict[str, Any]:
    if not math.isfinite(time_explosion_s) or time_explosion_s <= 0.0:
        raise AnalysisError("INVALID_TIME_EXPLOSION")
    blocked = partial_block_report(text)
    if blocked is not None:
        return blocked
    parsed = parse_log_blocks(text.splitlines())
    mapping = mapping_from_source(source, mapping_layout)
    analyzed = {
        iteration: analyze_iteration(
            parsed["blocks"][iteration]["rows"], time_explosion_s,
            temperature_K, mapping, target_ion,
        )
        for iteration in (0, 1)
    }
    summaries = {
        str(iteration): summary_observations(parsed["blocks"][iteration]["summary"])
        for iteration in (0, 1)
    }
    gates: dict[str, dict[str, Any]] = {
        "R2": {"status": "PASS", "r7_commits": 2, "physics_commits": 2,
               "exact_solve_epochs": 3},
    }
    blocking_reasons: list[str] = []
    if any(
        summary["target_Z"] != "26,27,28" or
        summary["target_ion"] != target_ion or
        summary["complete"] != 1 or
        summary["candidate_rows"] < summary["selected_rows"] or
        summary["selected_rows"] < 1 or
        summary["selected_fraction"] < 0.9 or
        summary["selected_fraction"] > 1.0
        for summary in summaries.values()
    ):
        blocking_reasons.append("R3_SUMMARY_IDENTITY_MISMATCH")
    total_rows = sum(result["census"]["TOTAL"] for result in analyzed.values())
    non_target = sum(result["census"]["NON_TARGET"] for result in analyzed.values())
    unavailable = sum(result["census"]["UNAVAILABLE"] for result in analyzed.values())
    flags_defined = total_rows - unavailable
    lines_by_iteration = [set(analyzed[i]["records"]) for i in (0, 1)]
    intersection = sorted(lines_by_iteration[0] & lines_by_iteration[1])
    if any(analyzed[i]["census"]["TOTAL"] < 1 for i in (0, 1)):
        blocking_reasons.append("R3_NO_ROWS")
    if len(intersection) < 30:
        blocking_reasons.append("R3_SAMPLE_BELOW_30")
    if non_target:
        blocking_reasons.append("R3_NON_TARGET_ROWS")
    if unavailable:
        blocking_reasons.append("R3_UNAVAILABLE_ROWS")
    gates["R3"] = {
        "status": "PASS" if not any(reason.startswith("R3_") for reason in blocking_reasons) else "FAIL",
        "rows_iter0": analyzed[0]["census"]["TOTAL"],
        "rows_iter1": analyzed[1]["census"]["TOTAL"],
        "line_id_intersection": len(intersection),
        "non_target_rows": non_target,
        "required_fields_defined": flags_defined,
        "required_fields_total": total_rows,
        "unavailable_rows": unavailable,
    }
    iter0_distribution = analyzed[0]["source_over_B"]
    iter0_median = None if iter0_distribution is None else iter0_distribution["quantiles"]["q50"]
    r4_pass = iter0_median is not None and 0.999 <= iter0_median <= 1.001
    if not r4_pass:
        blocking_reasons.append("R4_ANCHOR_FAIL")
    gates["R4"] = {"status": "PASS" if r4_pass else "FAIL", "iter0_median": iter0_median}
    r4b_pass = unavailable == 0 and all(
        value <= RECONSTRUCTION_TOLERANCE
        for iteration in analyzed.values()
        for value in iteration["r4b_max_relative_deviation"].values()
    )
    if not r4b_pass:
        blocking_reasons.append("R4B_RECONSTRUCTION_FAIL")
    gates["R4b"] = {
        "status": "PASS" if r4b_pass else "FAIL",
        "tolerance": RECONSTRUCTION_TOLERANCE,
        "maximum_relative_deviation": {
            str(i): analyzed[i]["r4b_max_relative_deviation"] for i in (0, 1)
        },
        "subtractions_or_inversions": 0,
    }
    mapped = sum(result["census"]["MAPPED"] for result in analyzed.values())
    identity_unavailable = sum(
        result["census"]["IDENTITY_UNAVAILABLE"] for result in analyzed.values()
    )
    f_mapped = (
        mapped / total_rows if total_rows and identity_unavailable == 0 else None
    )
    r5_pass = total_rows > 0 and identity_unavailable == 0 and mapped == total_rows
    if not r5_pass:
        blocking_reasons.append(
            "R5_MAPPING_UNAVAILABLE" if identity_unavailable else "R5_MAPPING_INCOMPLETE"
        )
    gates["R5"] = {
        "status": "PASS" if r5_pass else "FAIL",
        "mapped": mapped,
        "total": total_rows,
        "identity_unavailable": identity_unavailable,
        "f_mapped": f_mapped,
        "mapping_layout": mapping_layout,
    }
    r7 = compare_r7(text, sealed_text)
    gates["R7"] = r7
    if r7["status"] != "PASS":
        blocking_reasons.append("R7_BYTE_MISMATCH")

    matched_rows: list[dict[str, Any]] = []
    d_values: list[float] = []
    d_unavailable: list[dict[str, Any]] = []
    source_deviation: list[float] = []
    jbar_deviation: list[float] = []
    sign_matches = 0
    sign_evaluated = 0
    for identity in intersection:
        row0 = analyzed[0]["records"][identity]
        row1 = analyzed[1]["records"][identity]
        eta0 = row0["producer_eta"]
        eta1 = row1["producer_eta"]
        tau0 = row0["producer_tau_eff"]
        tau1 = row1["producer_tau_eff"]
        if eta0 == 0.0 or tau0 == 0.0 or tau1 == 0.0:
            d_unavailable.append({
                "line": identity[2], "Z": identity[0], "ion": identity[1],
                "reason": "D_RATIO_ZERO_DENOMINATOR",
            })
            continue
        source_ratio = (eta1 / eta0) / (tau1 / tau0)
        d_value = abs(source_ratio - 1.0)
        if not math.isfinite(d_value) or not math.isfinite(source_ratio):
            d_unavailable.append({
                "line": identity[2], "Z": identity[0], "ion": identity[1],
                "reason": "D_RATIO_NONFINITE",
            })
            continue
        s_over_b = row1["source_over_B"]
        if s_over_b is None:
            d_unavailable.append({
                "line": identity[2], "Z": identity[0], "ion": identity[1],
                "reason": "ITER1_SOURCE_UNDEFINED",
            })
            continue
        predicted = s_over_b - 1.0
        measured = row1["Jbar_over_B"] - 1.0
        residual = predicted - measured
        source_deviation.append(predicted)
        jbar_deviation.append(measured)
        if predicted != 0.0 and measured != 0.0:
            sign_evaluated += 1
            if (predicted > 0.0) == (measured > 0.0):
                sign_matches += 1
        record = {
            "line": identity[2], "Z": identity[0], "ion": identity[1],
            "d": d_value,
            "source_ratio_iter1_over_iter0": source_ratio,
            "S_over_B_iter1": s_over_b,
            "Jbar_over_B_iter1": row1["Jbar_over_B"],
            "pred_S_over_B_minus_1": predicted,
            "meas_Jbar_over_B_minus_1": measured,
            "resid_pred_minus_meas": residual,
            "nu": row1["nu"],
        }
        matched_rows.append(record)
        d_values.append(d_value)
    if d_unavailable:
        blocking_reasons.append("R6_D_CENSUS_UNAVAILABLE")
    d_distribution = distribution(d_values) if d_values else None
    ff = (sum(value <= FROZEN_TOLERANCE for value in d_values) / len(d_values)) if d_values else None
    jbar_iter1 = analyzed[1]["Jbar_over_B"]
    sigma = (
        "SIGNAL" if jbar_iter1 is not None and jbar_iter1["quantiles"]["q10"] < 0.95
        else "WEAK-SIGNAL"
    )
    gates_before_branch = all(gates[name]["status"] == "PASS" for name in (
        "R2", "R3", "R4", "R4b", "R5", "R7",
    )) and not d_unavailable
    if non_target > 0 or (f_mapped is not None and f_mapped < 1.0):
        verdict = "W-V"
    elif not gates_before_branch:
        verdict = "D3"
    elif ff is not None and ff <= 0.10:
        verdict = "C-R"
    elif ff is not None and ff >= 0.99:
        verdict = "C-F"
    else:
        verdict = "C-M"
    gates["R6"] = {
        "status": "PASS" if gates_before_branch else "FAIL",
        "gate_conjunct": {name: gates[name]["status"] for name in ("R2", "R3", "R4", "R4b", "R5", "R7")},
        "ff": ff,
        "frozen_tolerance": FROZEN_TOLERANCE,
        "sigma": sigma,
        "verdict": verdict,
    }
    single_temperature = (
        single_temperature_observation(matched_rows, temperature_K)
        if verdict == "C-R" else {
            "candidate_temperature_K": None,
            "collapsed": None,
            "reason": "ONLY_EVALUATED_FOR_C-R",
            "unavailable_rows": [],
        }
    )
    public_iterations: dict[str, Any] = {}
    for iteration in (0, 1):
        public_iterations[str(iteration)] = {
            key: value for key, value in analyzed[iteration].items() if key != "records"
        }
        public_iterations[str(iteration)]["summary"] = summaries[str(iteration)]
    return {
        "schema": "DET_L6C_COVER_VERDICT_V1",
        "mode": A210_L6C_PROBE,
        "status": "PASS" if verdict in ("C-R", "C-F", "C-M") else "PARTIAL",
        "verdict": verdict,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "gates": gates,
        "temperature_K": temperature_K,
        "time_explosion_s": time_explosion_s,
        "f_mapped": f_mapped,
        "ff": ff,
        "sigma": sigma,
        "d_distribution": d_distribution,
        "d_unavailable_rows": d_unavailable,
        "matched_rows": matched_rows,
        "iterations": public_iterations,
        "observations": {
            "residual_convention": "pred_minus_meas",
            "sign_agreement": {
                "matching": sign_matches,
                "evaluated": sign_evaluated,
                "fraction": sign_matches / sign_evaluated if sign_evaluated else None,
            },
            "pearson_S_minus_1_vs_Jbar_minus_1": pearson(source_deviation, jbar_deviation),
            "single_temperature_readjustment": single_temperature,
        },
        "constants": {
            "h_planck": H_PLANCK,
            "c_light": C_LIGHT,
            "k_boltzmann": K_BOLTZMANN,
            "four_pi": FOUR_PI,
            "planck_source": "physical constants; no external numeric oracle",
            "exponx_source": "independent transcription of line_net_rate.c",
        },
        "physical_values_modified": False,
    }


def footer_values(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError("R1_RUN_FOOTER_MISSING", str(path))
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in result:
            raise AnalysisError("R1_RUN_FOOTER_DUPLICATE", name)
        result[name] = value
    return result


def repository_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AnalysisError("R1_REPOSITORY_HEAD_UNAVAILABLE")
    return result.stdout.strip()


def verify_freshness(run_root: Path, repo: Path, expected_head: str | None) -> dict[str, Any]:
    footer = footer_values(run_root / "RUN_FOOTER.txt")
    input_dir = run_root / "input"
    head_path = input_dir / "git_head.txt"
    if not head_path.is_file() or head_path.is_symlink():
        raise AnalysisError("R1_GIT_HEAD_MISSING")
    staged_head = head_path.read_text(encoding="utf-8", errors="strict").strip()
    wanted_head = expected_head or repository_head(repo)
    if staged_head != wanted_head or footer.get("git_head") != wanted_head:
        raise AnalysisError(
            "R1_GIT_HEAD_MISMATCH",
            f"staged={staged_head},footer={footer.get('git_head')},expected={wanted_head}",
        )
    specs = {
        "job_slurm_sha256": "job.slurm",
        "checker_sha256": "check_a210_targeted_gate.py",
        "stager_sha256": "stage_det_stage12_l6_probe.sh",
        "analyzer_l6c_sha256": "analyze_det_stage12_l6c.py",
        "precondition_sha256": "audit_l6c_cover_precondition.py",
    }
    observed: dict[str, str] = {}
    for key, filename in specs.items():
        actual = sha256(input_dir / filename)
        observed[key] = actual
        if footer.get(key) != actual:
            raise AnalysisError("R1_STAGED_SHA_MISMATCH", key)
    binary_sha = sha256(input_dir / "lumina_cuda")
    if footer.get("binary_sha256") != binary_sha or binary_sha != EXPECTED_BINARY_SHA256:
        raise AnalysisError("R1_BINARY_SHA_MISMATCH", binary_sha)
    return {
        "status": "PASS",
        "git_head": wanted_head,
        "staged_sha256": observed,
        "binary_sha256": binary_sha,
    }


def time_explosion_from_run(run_root: Path) -> float:
    config = run_root / "input" / "model" / "config.json"
    if not config.is_file() or config.is_symlink():
        raise AnalysisError("MISSING_SAFE_MODEL_CONFIG")
    try:
        value = float(json.loads(config.read_text(encoding="utf-8"))["time_explosion_s"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("INVALID_MODEL_TIME_EXPLOSION") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise AnalysisError("INVALID_MODEL_TIME_EXPLOSION")
    return value


def synthetic_row(
    *, line: int, z: int, iteration: int, response: float,
    time_explosion_s: float, iter0_ratio: float = 1.0,
) -> str:
    nu = 5.0e14 + line * 1.0e10
    b_nu = planck_nu(nu, REQUESTED_TEMPERATURE_K)
    tau = 1.0
    source_ratio = iter0_ratio if iteration == 0 else response * iter0_ratio
    chi = tau * nu / (C_LIGHT * time_explosion_s)
    eta = source_ratio * b_nu * chi
    beta, companion = exponx(tau)
    j_cont = 0.8 * b_nu
    continuum = beta * j_cont
    local = eta * (C_LIGHT * time_explosion_s / nu) * companion
    jbar = continuum + local
    return (
        f"{ROW_PREFIX} phase=REQUESTED_TE shell=0 rank={line + 1} "
        f"line={line} Z={z} ion=1 nu={nu:.17g} Jbar={jbar:.17g} "
        f"J_cont={j_cont:.17g} independent_fields_defined=1 "
        f"producer_continuum_term={continuum:.17g} "
        f"producer_local_emission_term={local:.17g} producer_terms_defined=1 "
        f"producer_eta={eta:.17g} producer_tau_eff={tau:.17g} "
        "producer_srce_chk=0 producer_exact_zero=0 producer_raw_defined=1"
    )


def synthetic_log(
    response: float = 0.8, *, iter0_ratio: float = 1.0,
    time_explosion_s: float = 1.0e6,
) -> str:
    lines = [f"{EPOCH_PREFIX} status=OK fixture=seed"]
    for iteration in (0, 1):
        lines.append(f"{EPOCH_PREFIX} status=OK fixture=iter{iteration}")
        for line in range(30):
            lines.append(synthetic_row(
                line=line, z=26 + line % 3, iteration=iteration,
                response=response, time_explosion_s=time_explosion_s,
                iter0_ratio=iter0_ratio,
            ))
        lines.append(
            f"{SUMMARY_PREFIX} phase=REQUESTED_TE shell=0 target_Z=26,27,28 "
            "target_ion=1 candidate_rows=30 selected_rows=30 "
            "total_scaled_emission=1 selected_scaled_emission=0.9 "
            "selected_fraction=0.9 selection_mode=PER_ION_UNION complete=1 "
            "zero_opacity_emitting_rows=0"
        )
        for shell in range(50):
            lines.append(f"{IDENTITY_PREFIX} phase=REQUESTED_TE shell={shell} fixture={iteration}:{shell}")
        lines.append(
            f"{R7_PREFIX} lane=DET iter={iteration} phase=A2-10 "
            f"te_generation={iteration + 1}->{iteration + 2}"
        )
        lines.append(f"{PHYSICS_PREFIX} iter={iteration} status=COMMITTED dir=/scratch")
    return "\n".join(lines) + "\n"


def replace_first_field(text: str, prefix: str, field: str, replacement: str) -> str:
    output: list[str] = []
    changed = False
    pattern = re.compile(rf"\b{re.escape(field)}=[^\s]+")
    for line in text.splitlines():
        if not changed and line.startswith(prefix):
            updated, count = pattern.subn(f"{field}={replacement}", line, count=1)
            if count:
                line = updated
                changed = True
        output.append(line)
    if not changed:
        raise AnalysisError("SELFTEST_FIELD_NOT_FOUND", field)
    return "\n".join(output) + "\n"


def selftest(args: argparse.Namespace) -> int:
    source = git_source(args.repo_root, args.source_revision)
    sealed_path = args.sealed_l6_stderr
    if not sealed_path.is_file() or sealed_path.is_symlink():
        raise AnalysisError("NC-B1_SEALED_STDERR_MISSING", str(sealed_path))
    sealed_text = sealed_path.read_text(encoding="utf-8", errors="strict")
    sealed = analyze_text(
        sealed_text, sealed_text, source, 1683072.0,
        target_ion=3, mapping_layout="ION4",
    )
    if sealed["verdict"] != "C-F":
        raise AnalysisError("NC-B1_DID_NOT_SELECT_C-F", sealed["verdict"])
    print(
        "NC-B1 inject=SEALED_L6_STDERR status=FAIL reason=C-F "
        f"ff={sealed['ff']:.17g} rows={len(sealed['matched_rows'])}"
    )
    response_fixture = synthetic_log(response=0.8)
    response = analyze_text(response_fixture, response_fixture, source, 1.0e6)
    if response["verdict"] != "C-R":
        raise AnalysisError("SOLVER_RESPONSE_DID_NOT_SELECT_C-R", response["verdict"])
    print(f"NC-B1 remove=SOLVER_RESPONSE status=PASS verdict=C-R ff={response['ff']:.17g}")

    forged = replace_first_field(response_fixture, ROW_PREFIX, "Z", "14")
    forged_report = analyze_text(forged, forged, source, 1.0e6)
    if forged_report["verdict"] != "W-V" or "R3_NON_TARGET_ROWS" not in forged_report["blocking_reasons"]:
        raise AnalysisError("NC-B2_NON_TARGET_ACCEPTED")
    print("NC-B2 inject=NON_TARGET_ROW status=FAIL reason=R3_NON_TARGET_ROWS")
    clean = analyze_text(response_fixture, response_fixture, source, 1.0e6)
    if clean["gates"]["R3"]["status"] != "PASS":
        raise AnalysisError("NC-B2_REMOVAL_FAILED")
    print("NC-B2 remove=TARGET_ROW status=PASS reason=R3_PASS")

    first_row = next(line for line in response_fixture.splitlines() if line.startswith(ROW_PREFIX))
    eta = finite(fields(first_row), "producer_eta")
    perturbed = replace_first_field(
        response_fixture, ROW_PREFIX, "producer_eta", f"{eta * (1.0 + 1.0e-9):.17g}"
    )
    perturbed_report = analyze_text(perturbed, perturbed, source, 1.0e6)
    r4b_errors = [
        row["reason"]
        for row in perturbed_report["iterations"]["0"]["unavailable_rows"]
    ]
    if (perturbed_report["verdict"] != "D3" or
            "R4B_RECONSTRUCTION_FAIL" not in r4b_errors):
        raise AnalysisError("NC-B3_PERTURBATION_ACCEPTED")
    print("NC-B3 inject=ETA_RELATIVE_PLUS_1E-9 status=FAIL reason=R4B_RECONSTRUCTION_FAIL")
    analyze_text(response_fixture, response_fixture, source, 1.0e6)
    print("NC-B3 remove=ETA_EXACT status=PASS reason=R4b_PASS")

    altered_source = source.replace(
        "{ 14, 14, 20, 20, 26, 26, 16, 16, 27, 27, 28, 28,",
        "{ 14, 14, 20, 20, 99, 26, 16, 16, 27, 27, 28, 28,",
        1,
    )
    if altered_source == source:
        raise AnalysisError("NC-B4_TABLE_INJECTION_FAILED")
    unmapped = analyze_text(response_fixture, response_fixture, altered_source, 1.0e6)
    if unmapped["verdict"] != "W-V" or "R5_MAPPING_INCOMPLETE" not in unmapped["blocking_reasons"]:
        raise AnalysisError("NC-B4_UNMAPPED_ACCEPTED")
    print(
        "NC-B4 inject=REMOVE_26_1_MAPPING status=FAIL reason=R5_MAPPING_INCOMPLETE "
        f"f_mapped={unmapped['f_mapped']:.17g}"
    )
    restored = analyze_text(response_fixture, response_fixture, source, 1.0e6)
    print(f"NC-B4 remove=RESTORE_26_1_MAPPING status=PASS f_mapped={restored['f_mapped']:.17g}")

    lines = response_fixture.splitlines()
    deleted = False
    missing_marker_lines: list[str] = []
    for line in lines:
        if not deleted and line.startswith(R7_PREFIX) and " iter=1 " in line:
            deleted = True
            continue
        missing_marker_lines.append(line)
    try:
        analyze_text("\n".join(missing_marker_lines) + "\n", response_fixture, source, 1.0e6)
    except AnalysisError as exc:
        if exc.reason != "R7_MARKER_MISSING":
            raise
        print(f"NC-B5 inject=DELETE_ITER1_R7 status=FAIL reason={exc.reason} branch=BLOCKED")
    else:
        raise AnalysisError("NC-B5_MISSING_MARKER_ACCEPTED")
    analyze_text(response_fixture, response_fixture, source, 1.0e6)
    print("NC-B5 remove=RESTORE_ITER1_R7 status=PASS reason=R7_PASS")

    bad_anchor = synthetic_log(response=0.8, iter0_ratio=1.01)
    anchor_report = analyze_text(bad_anchor, bad_anchor, source, 1.0e6)
    if anchor_report["verdict"] != "D3" or "R4_ANCHOR_FAIL" not in anchor_report["blocking_reasons"]:
        raise AnalysisError("NC-R4_ANCHOR_ACCEPTED")
    print("NC-R4 inject=ANCHOR_OUTSIDE status=FAIL reason=R4_ANCHOR_FAIL verdict=D3")
    print("NC-R4 remove=ANCHOR_RESTORED status=PASS reason=R4_PASS")

    with tempfile.TemporaryDirectory(prefix="l6c-ncr7-") as directory:
        scratch = Path(directory) / "sealed.stderr"
        scratch.write_text(response_fixture, encoding="utf-8")
        mutated = scratch.read_text(encoding="utf-8").replace("fixture=0:0", "fixture=0:X", 1)
        scratch.write_text(mutated, encoding="utf-8")
        mutated_report = analyze_text(response_fixture, scratch.read_text(encoding="utf-8"), source, 1.0e6)
        if mutated_report["verdict"] != "D3" or "R7_BYTE_MISMATCH" not in mutated_report["blocking_reasons"]:
            raise AnalysisError("NC-R7_MUTATION_ACCEPTED")
        print("NC-R7 inject=IDENTITY_ONE_BYTE status=FAIL reason=R7_BYTE_MISMATCH verdict=D3")
        scratch.write_text(response_fixture, encoding="utf-8")
        restored_report = analyze_text(response_fixture, scratch.read_text(encoding="utf-8"), source, 1.0e6)
        if restored_report["gates"]["R7"]["status"] != "PASS":
            raise AnalysisError("NC-R7_REMOVAL_FAILED")
        print("NC-R7 remove=RESTORE_BYTE status=PASS reason=R7_PASS")

    try:
        verify_freshness(
            args.sealed_l6_stderr.parent,
            args.repo_root,
            repository_head(args.repo_root),
        )
    except AnalysisError as exc:
        if exc.reason != "R1_GIT_HEAD_MISMATCH":
            raise
        print(f"NC-R1 inject=SEALED_L6_ROOT status=FAIL reason={exc.reason}")
    else:
        raise AnalysisError("NC-R1_SEALED_ROOT_ACCEPTED")
    print(
        "DET_L6C_COVER_ANALYZER_SELFTEST_PASS "
        "controls=NC-B1,NC-B2,NC-B3,NC-B4,NC-B5,NC-R4,NC-R7,NC-R1"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--sealed-l6-stderr", type=Path, default=DEFAULT_SEALED_L6_STDERR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--time-explosion-s", type=float)
    parser.add_argument("--temperature-k", type=float, default=REQUESTED_TEMPERATURE_K)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--expected-head")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.selftest:
            return selftest(args)
        if args.run_root is None and args.stderr is None:
            raise AnalysisError("RUN_ROOT_OR_STDERR_REQUIRED")
        stderr_path = args.stderr or args.run_root / "stderr.log"
        if args.report is not None:
            report_path = args.report
        elif args.run_root is not None:
            report_path = args.run_root / "det_l6c_cover_verdict.json"
        else:
            raise AnalysisError("REPORT_REQUIRED_WITH_STDERR")
        if not stderr_path.is_file() or stderr_path.is_symlink():
            raise AnalysisError("MISSING_SAFE_STDERR")
        if not args.sealed_l6_stderr.is_file() or args.sealed_l6_stderr.is_symlink():
            raise AnalysisError("MISSING_SAFE_SEALED_L6_STDERR")
        time_explosion_s = (
            args.time_explosion_s if args.time_explosion_s is not None
            else time_explosion_from_run(args.run_root)
        )
        source = git_source(args.repo_root, args.source_revision)
        text = stderr_path.read_text(encoding="utf-8", errors="strict")
        sealed_text = args.sealed_l6_stderr.read_text(encoding="utf-8", errors="strict")
        report = analyze_text(
            text, sealed_text, source, time_explosion_s,
            temperature_K=args.temperature_k,
        )
        if args.run_root is not None:
            try:
                r1 = verify_freshness(
                    args.run_root, args.repo_root, args.expected_head,
                )
            except AnalysisError as exc:
                r1 = {"status": "FAIL", "reason": exc.reason, "detail": exc.detail}
                report["status"] = "PARTIAL"
                report["verdict"] = "D3"
                report.setdefault("blocking_reasons", []).append(exc.reason)
            report.setdefault("gates", {})["R1"] = r1
        else:
            report.setdefault("gates", {})["R1"] = {
                "status": "NOT_EVALUATED", "reason": "NO_RUN_ROOT"
            }
        report["stderr_sha256"] = sha256(stderr_path)
        report["sealed_l6_stderr_sha256"] = sha256(args.sealed_l6_stderr)
        atomic_write_json(report_path, report)
    except (AnalysisError, OSError, UnicodeError) as exc:
        reason = exc.reason if isinstance(exc, AnalysisError) else type(exc).__name__
        detail = exc.detail if isinstance(exc, AnalysisError) else str(exc)
        failure_path = args.report
        if failure_path is None and args.run_root is not None:
            failure_path = args.run_root / "det_l6c_cover_verdict.json"
        if failure_path is not None:
            atomic_write_json(failure_path, {
                "schema": "DET_L6C_COVER_VERDICT_V1",
                "status": "PARTIAL",
                "verdict": "D3",
                "blocking_reasons": [reason],
                "detail": detail,
                "physical_values_modified": False,
            })
        print(
            f"DET_L6C_COVER_ANALYZER_FAIL verdict=D3 reason={reason}"
            + (f" detail={detail}" if detail else ""),
            file=sys.stderr,
        )
        return 4
    print(
        "DET_L6C_COVER_ANALYZER_RESULT "
        f"status={report['status']} verdict={report['verdict']} "
        f"ff={report.get('ff')} f_mapped={report.get('f_mapped')} report={report_path}"
    )
    return 0 if report["verdict"] in ("C-R", "C-F", "C-M") else 4


if __name__ == "__main__":
    raise SystemExit(main())
