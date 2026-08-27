#!/usr/bin/env python3
"""Fail-closed reader for DET-L6C-COVER-R sealed R-II output.

The production path is deliberately split in two.  PR-0 reads only repository
documents and ``sacct`` output.  Sealed run, census, and deck paths are opened
only after PR-0 passes.  ``--selftest`` uses scratch fixtures exclusively.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


A210_L6C_REDISPATCH = "A210_L6C_REDISPATCH"
SCHEMA = "A210_L6C_REDISPATCH_V1"
ROW_PREFIX = b"[A2-10][LINE-SATURATION-ROW]"
ZERO_PREFIX = b"[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"
SUMMARY_PREFIX = b"[A2-10][LINE-SATURATION-SUMMARY]"
BLOCKED_PREFIX = b"[A2-10][LINE-SATURATION-BLOCKED]"
ISOLATED_PREFIX = b"[A2-10][LINE-SATURATION-ISOLATED]"
R7_PREFIX = b"[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED"
TARGET_Z = frozenset({26, 27, 28})
TARGET_ION = 1
SUPER_CUTOFF = 100
RII_JOB_ID = "324557"
TEMPERATURE_K = 10020.0
H_PLANCK = 6.62607015e-27
C_LIGHT = 2.99792458e10
K_BOLTZMANN = 1.380649e-16
RECONSTRUCTION_TOLERANCE = 1.0e-12
FROZEN_TOLERANCE = 1.0e-13
FF_RESPONSE_MAX = 0.10
FF_FROZEN_MIN = 0.99
ANCHOR_LINE = 669_992
ANCHOR_BITS = {
    "tau_validity": "2",
    "n_upper": "0x1.d1534dba20c26p-12",
    "A_ul": "0x1.0b6516044e436p-1",
    "emission_per_sr": "0x1.3adcec9caadc0p-52",
}


class ReaderError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Expected:
    selected: tuple[int, int]
    intersection: int
    classes: tuple[dict[str, int], dict[str, int]]
    matched_fpc: dict[str, int]
    matched_neg: int
    zero_stream: tuple[int, int]
    candidate_rows: int


REGISTERED = Expected(
    selected=(1_201, 992),
    intersection=992,
    classes=(
        {"EXP": 323, "MIX": 621, "BL": 164, "HAZ": 93},
        {"EXP": 284, "MIX": 506, "BL": 141, "HAZ": 61},
    ),
    matched_fpc={"BL": 95, "EXP": 284, "MIX": 506},
    matched_neg=46,
    zero_stream=(221_382, 221_382),
    candidate_rows=749_644,
)


@dataclass
class Block:
    rows: list[dict[str, str]] = field(default_factory=list)
    row_line_numbers: list[int] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)
    zero_count: int = 0
    zero_digest: str = ""
    anchors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ParsedLog:
    blocks: list[Block]
    blocked: list[dict[str, str]]
    isolated: list[dict[str, str]]
    r7_iter1_good: int
    r7_iter1_total: int
    issues: list[dict[str, Any]]


def issue(reason: str, detail: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {"reason": reason}
    if detail not in (None, ""):
        result["detail"] = detail
    return result


def add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def parse_fields(raw: bytes, log_line: int) -> dict[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ReaderError("LOG_RECORD_NOT_UTF8", f"line={log_line}") from exc
    result: dict[str, str] = {}
    for token in text.split()[1:]:
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if name in result:
            raise ReaderError("DUPLICATE_FIELD", f"line={log_line},field={name}")
        result[name] = value
    result["_log_line"] = str(log_line)
    return result


def exact_int(row: dict[str, str], name: str, reason: str = "FIELD_INVALID") -> int:
    raw = row.get(name)
    if raw is None:
        raise ReaderError("FIELD_MISSING", name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ReaderError(reason, f"{name}={raw}") from exc
    if str(value) != raw:
        raise ReaderError(reason, f"{name}={raw}")
    return value


def finite_float(row: dict[str, str], name: str) -> float:
    raw = row.get(name)
    if raw is None:
        raise ReaderError("FIELD_MISSING", name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ReaderError("FIELD_INVALID", f"{name}={raw}") from exc
    if not math.isfinite(value):
        raise ReaderError("FIELD_NONFINITE", f"{name}={raw}")
    return value


def regular_file(path: Path, reason: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ReaderError(reason, str(path))


def stream_sha_update(digest: Any, raw: bytes) -> None:
    # This exactly represents splitlines()-style record bytes without allowing
    # concatenation ambiguity.  Newline spelling is intentionally not a field.
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def parse_log(path: Path) -> ParsedLog:
    regular_file(path, "LOG_MISSING_OR_UNSAFE")
    blocks: list[Block] = []
    rows: list[dict[str, str]] = []
    row_lines: list[int] = []
    zero_count = 0
    zero_digest = hashlib.sha256()
    anchors: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    isolated: list[dict[str, str]] = []
    issues: list[dict[str, Any]] = []
    r7_good = 0
    r7_total = 0
    try:
        with path.open("rb") as stream:
            for log_line, with_newline in enumerate(stream, 1):
                raw = with_newline.rstrip(b"\r\n")
                try:
                    if raw.startswith(ZERO_PREFIX):
                        zero_count += 1
                        stream_sha_update(zero_digest, raw)
                        values = parse_fields(raw, log_line)
                        if values.get("line") == str(ANCHOR_LINE):
                            anchors.append(values)
                    elif raw.startswith(ROW_PREFIX):
                        rows.append(parse_fields(raw, log_line))
                        row_lines.append(log_line)
                    elif raw.startswith(SUMMARY_PREFIX):
                        summary = parse_fields(raw, log_line)
                        blocks.append(Block(
                            rows=rows,
                            row_line_numbers=row_lines,
                            summary=summary,
                            zero_count=zero_count,
                            zero_digest=zero_digest.hexdigest(),
                            anchors=anchors,
                        ))
                        rows, row_lines = [], []
                        zero_count = 0
                        zero_digest = hashlib.sha256()
                        anchors = []
                    elif raw.startswith(BLOCKED_PREFIX):
                        blocked.append(parse_fields(raw, log_line))
                    elif raw.startswith(ISOLATED_PREFIX):
                        isolated.append(parse_fields(raw, log_line))
                    elif raw.startswith(R7_PREFIX):
                        values = parse_fields(raw, log_line)
                        if values.get("iter") == "1":
                            r7_total += 1
                            if (values.get("lane") == "DET" and
                                    values.get("te_generation") == "2->3"):
                                r7_good += 1
                except ReaderError as exc:
                    issues.append(issue(exc.reason, {
                        "log_line": log_line, "detail": exc.detail,
                    }))
    except (OSError, UnicodeError) as exc:
        raise ReaderError("LOG_READ_FAILED", f"{path}:{exc}") from exc
    if rows or zero_count or anchors:
        issues.append(issue("ORPHAN_RECORDS_AFTER_FINAL_SUMMARY", {
            "selected_rows": len(rows), "zero_rows": zero_count,
        }))
    return ParsedLog(blocks, blocked, isolated, r7_good, r7_total, issues)


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ReaderError("EMPTY_DISTRIBUTION")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def distribution(values: Iterable[float]) -> dict[str, Any] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q10": percentile(ordered, 0.10),
        "median": percentile(ordered, 0.50),
        "q90": percentile(ordered, 0.90),
        "maximum": ordered[-1],
        "exact_zero": sum(value == 0.0 for value in ordered),
    }


def relative_deviation(observed: float, reconstructed: float) -> float:
    if observed == reconstructed:
        return 0.0
    denominator = max(abs(observed), abs(reconstructed))
    if denominator == 0.0:
        return math.inf
    return abs(observed - reconstructed) / denominator


def exponx(tau: float) -> tuple[float, float]:
    if not math.isfinite(tau):
        raise ReaderError("EXPONX_NONFINITE_TAU")
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        beta = 1.0 - tau * companion
    elif tau < 40.0:
        beta = (1.0 - math.exp(-tau)) / tau
        companion = (1.0 - beta) / tau
    else:
        beta = 1.0 / tau
        companion = (1.0 - beta) / tau
    if not all(math.isfinite(value) for value in (beta, companion)):
        raise ReaderError("EXPONX_DOMAIN")
    return beta, companion


def planck_nu(nu: float) -> float:
    if not math.isfinite(nu) or nu <= 0.0:
        raise ReaderError("PLANCK_DOMAIN")
    exponent = H_PLANCK * nu / (K_BOLTZMANN * TEMPERATURE_K)
    try:
        denominator = math.expm1(exponent)
    except OverflowError as exc:
        raise ReaderError("PLANCK_DOMAIN") from exc
    value = 2.0 * H_PLANCK * nu ** 3 / (C_LIGHT ** 2 * denominator)
    if not math.isfinite(value) or value <= 0.0:
        raise ReaderError("PLANCK_DOMAIN")
    return value


def load_deck(deck: Path, wanted_lines: set[int]) -> tuple[
        dict[int, tuple[int, int, int, int]], set[tuple[int, int, int]]]:
    lines_path = deck / "line_list.csv"
    levels_path = deck / "levels.csv"
    regular_file(lines_path, "DECK_LINE_LIST_MISSING")
    regular_file(levels_path, "DECK_LEVELS_MISSING")
    line_map: dict[int, tuple[int, int, int, int]] = {}
    required_lines = {
        "line_id", "atomic_number", "ion_number",
        "level_number_lower", "level_number_upper",
    }
    try:
        with lines_path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required_lines.issubset(reader.fieldnames):
                raise ReaderError("DECK_LINE_FIELDS_MISSING", sorted(required_lines))
            for row_number, row in enumerate(reader, 2):
                try:
                    line_id = int(row["line_id"])
                    if line_id not in wanted_lines:
                        continue
                    value = (
                        int(row["atomic_number"]), int(row["ion_number"]),
                        int(row["level_number_lower"]), int(row["level_number_upper"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReaderError("DECK_LINE_FIELD_INVALID", f"row={row_number}") from exc
                if line_id in line_map:
                    raise ReaderError("DECK_LINE_DUPLICATE", str(line_id))
                line_map[line_id] = value
        needed_levels = {
            (z, ion, lower) for z, ion, lower, _upper in line_map.values()
        } | {
            (z, ion, upper) for z, ion, _lower, upper in line_map.values()
        }
        level_keys: set[tuple[int, int, int]] = set()
        required_levels = {"atomic_number", "ion_number", "level_number"}
        with levels_path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required_levels.issubset(reader.fieldnames):
                raise ReaderError("DECK_LEVEL_FIELDS_MISSING", sorted(required_levels))
            for row_number, row in enumerate(reader, 2):
                try:
                    key = (int(row["atomic_number"]), int(row["ion_number"]),
                           int(row["level_number"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReaderError("DECK_LEVEL_FIELD_INVALID", f"row={row_number}") from exc
                if key in needed_levels:
                    if key in level_keys:
                        raise ReaderError("DECK_LEVEL_DUPLICATE", str(key))
                    level_keys.add(key)
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, ReaderError):
            raise
        raise ReaderError("DECK_READ_FAILED", str(exc)) from exc
    return line_map, level_keys


def parse_c_int_array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"static\s+const\s+int\s+{re.escape(name)}(?:\[[^\]]*\])?\s*=\s*"
        rf"\{{(?P<body>.*?)\}}\s*;", source, re.S,
    )
    if not match:
        raise ReaderError("MAPPING_TABLE_PARSE_FAILED", name)
    return [int(token) for token in re.findall(r"[-+]?\d+", match.group("body"))]


def mapping_from_source(source: str) -> frozenset[tuple[int, int]]:
    z_values = parse_c_int_array(source, "NLTE_TARGET_Z")
    ion_values = parse_c_int_array(source, "NLTE_TARGET_ION")
    if not z_values or len(z_values) != len(ion_values):
        raise ReaderError("MAPPING_TABLE_PARSE_FAILED", "length")
    return frozenset(zip(z_values, ion_values))


def source_at_head(repo: Path, expected_head: str) -> str:
    current = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if current.returncode != 0:
        raise ReaderError("REPOSITORY_HEAD_UNAVAILABLE")
    if current.stdout.strip() != expected_head:
        raise ReaderError(
            "EXPECTED_HEAD_MISMATCH",
            f"current={current.stdout.strip()},expected={expected_head}",
        )
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{expected_head}:src/lumina_plasma.c"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ReaderError("MAPPING_SOURCE_BLOB_UNAVAILABLE", expected_head)
    return result.stdout


def time_explosion(deck: Path) -> float:
    path = deck / "config.json"
    regular_file(path, "MODEL_CONFIG_MISSING")
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["time_explosion_s"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ReaderError("MODEL_CONFIG_INVALID", str(exc)) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ReaderError("MODEL_CONFIG_INVALID", "time_explosion_s")
    return value


def anchor_ok(block: Block) -> tuple[bool, str]:
    if len(block.anchors) != 1:
        return False, "ANCHOR_MISSING" if not block.anchors else "ANCHOR_AMBIGUOUS"
    row = block.anchors[0]
    for name, expected in ANCHOR_BITS.items():
        raw = row.get(name)
        if raw is None:
            return False, "ANCHOR_FIELD_MISSING"
        if name == "tau_validity":
            if raw != expected:
                return False, "ANCHOR_FIELD_MISMATCH"
        else:
            try:
                if float(raw).hex() != expected:
                    return False, "ANCHOR_FIELD_MISMATCH"
            except ValueError:
                return False, "ANCHOR_FIELD_INVALID"
    return True, "PASS"


def summary_valid(summary: dict[str, str], selected: int,
                  expected: Expected, require_sprobe: bool) -> list[str]:
    reasons: list[str] = []
    expected_fields = {
        "phase": "REQUESTED_TE", "shell": "0", "target_Z": "26,27,28",
        "target_ion": "1", "candidate_rows": str(expected.candidate_rows),
        "selected_rows": str(selected), "complete": "1",
        "zero_opacity_emitting_rows": str(expected.zero_stream[0]),
    }
    if require_sprobe:
        expected_fields["sprobe_undefined_rows"] = str(expected.zero_stream[0])
    for name, wanted in expected_fields.items():
        if summary.get(name) != wanted:
            add_reason(reasons, "SUMMARY_TOKEN_MISMATCH")
    return reasons


def classify_rows(
    parsed: ParsedLog,
    line_map: dict[int, tuple[int, int, int, int]],
    level_keys: set[tuple[int, int, int]],
) -> tuple[list[dict[int, dict[str, Any]]], list[Counter[str]], list[dict[str, Any]]]:
    indexed: list[dict[int, dict[str, Any]]] = []
    censuses: list[Counter[str]] = []
    issues: list[dict[str, Any]] = []
    for iteration, block in enumerate(parsed.blocks):
        records: dict[int, dict[str, Any]] = {}
        counts: Counter[str] = Counter()
        for row in block.rows:
            log_line = row.get("_log_line")
            try:
                line = exact_int(row, "line")
                rank = exact_int(row, "rank")
                z = exact_int(row, "Z")
                ion = exact_int(row, "ion")
                if line in records:
                    raise ReaderError("SELECTED_LINE_DUPLICATE", str(line))
                if z not in TARGET_Z or ion != TARGET_ION:
                    class_name = "NON_TARGET"
                else:
                    deck_identity = line_map.get(line)
                    if deck_identity is None:
                        raise ReaderError("DECK_LINE_JOIN_MISSING", str(line))
                    dz, dion, lower, upper = deck_identity
                    if (dz, dion) != (z, ion):
                        raise ReaderError("DECK_ROW_IDENTITY_MISMATCH", str(line))
                    if row.get("lower_level") != str(lower) or row.get("upper_level") != str(upper):
                        raise ReaderError("DECK_ROW_LEVEL_MISMATCH", str(line))
                    if ((z, ion, lower) not in level_keys or
                            (z, ion, upper) not in level_keys):
                        raise ReaderError("DECK_LEVEL_JOIN_MISSING", str(line))
                    validity = exact_int(row, "tau_validity")
                    if validity == 2:
                        class_name = "HAZ"
                    elif validity == 1:
                        lower_lumped = lower >= SUPER_CUTOFF
                        upper_lumped = upper >= SUPER_CUTOFF
                        class_name = (
                            "BL" if lower_lumped and upper_lumped else
                            "EXP" if not lower_lumped and not upper_lumped else "MIX"
                        )
                    else:
                        raise ReaderError("TAU_VALIDITY_INVALID", str(validity))
                record: dict[str, Any] = {
                    "iteration": iteration, "line": line, "rank": rank,
                    "Z": z, "ion": ion, "class": class_name, "raw": row,
                }
                records[line] = record
                counts[class_name] += 1
            except ReaderError as exc:
                issues.append(issue(exc.reason, {
                    "iteration": iteration, "log_line": log_line,
                    "detail": exc.detail,
                }))
        indexed.append(records)
        censuses.append(counts)
    return indexed, censuses, issues


def capture_values(row: dict[str, str]) -> dict[str, float]:
    flags = (
        exact_int(row, "producer_terms_defined"),
        exact_int(row, "producer_raw_defined"),
        exact_int(row, "independent_fields_defined"),
    )
    if flags != (1, 1, 1):
        raise ReaderError("CAPTURE_FIELDS_UNDEFINED", str(flags))
    return {
        name: finite_float(row, name) for name in (
            "producer_eta", "producer_tau_eff", "producer_continuum_term",
            "producer_local_emission_term", "nu", "J_cont", "Jbar",
            "chi_effective", "eta_per_sr", "tau_effective",
        )
    }


def hazard_observation(records: list[dict[int, dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    decode_counts: Counter[str] = Counter()
    violations: list[dict[str, Any]] = []
    decode_violations: list[dict[str, Any]] = []
    for table in records:
        for record in table.values():
            if record["class"] != "HAZ":
                continue
            raw = record["raw"]
            independent = raw.get("independent_fields_defined")
            s_probe = raw.get("S_probe")
            correct_representation = independent == "0" and s_probe == "UNAVAILABLE"
            if not correct_representation:
                violations.append({
                    "iteration": record["iteration"], "line": record["line"],
                    "reason": "HAZARD_REPRESENTATION_INVALID",
                })
            raw_defined = raw.get("producer_raw_defined", "MISSING")
            terms_defined = raw.get("producer_terms_defined", "MISSING")
            decode_counts[f"raw={raw_defined},terms={terms_defined}"] += 1
            try:
                if raw_defined != "1":
                    raise ReaderError("HAZARD_CAPTURE_RAW_UNDEFINED")
                eta = finite_float(raw, "producer_eta")
                tau = finite_float(raw, "producer_tau_eff")
                srce = exact_int(raw, "producer_srce_chk")
                exact_zero = exact_int(raw, "producer_exact_zero")
                if eta < 0.0 or tau != 0.0 or srce not in (0, 1) or exact_zero not in (0, 1):
                    raise ReaderError("HAZARD_CAPTURE_DECODE_INVALID")
            except ReaderError as exc:
                decode_violations.append({
                    "iteration": record["iteration"], "line": record["line"],
                    "reason": exc.reason, "detail": exc.detail,
                })
            rows.append({
                "iteration": record["iteration"], "line": record["line"],
                "Z": record["Z"], "class": "HAZ",
                "S_probe": s_probe, "independent_fields_defined": independent,
                "producer_raw_defined": raw_defined,
                "producer_terms_defined": terms_defined,
                "producer_eta": raw.get("producer_eta"),
                "producer_tau_eff": raw.get("producer_tau_eff"),
                "producer_srce_chk": raw.get("producer_srce_chk"),
                "producer_exact_zero": raw.get("producer_exact_zero"),
            })
    return {
        "count": len(rows), "rows": rows,
        "capture_decode_census": dict(sorted(decode_counts.items())),
        "capture_decode_violations": decode_violations,
        "representation_violations": violations,
        "interpretation": "observation; capture decode violations belong to Z-2A ZI-X",
    }


def reconstruct(values: dict[str, float], time_s: float) -> dict[str, float]:
    beta, companion = exponx(values["producer_tau_eff"])
    continuum = beta * values["J_cont"]
    local = (values["producer_eta"] * (C_LIGHT * time_s / values["nu"]) * companion)
    jbar = continuum + local
    return {"continuum": continuum, "local": local, "jbar": jbar}


def analyze(
    run_log: Path, census_log: Path, deck: Path, source: str,
    expected: Expected = REGISTERED,
) -> dict[str, Any]:
    run = parse_log(run_log)
    census = parse_log(census_log)
    time_s = time_explosion(deck)
    reasons: list[str] = []
    details: list[dict[str, Any]] = [*run.issues, *census.issues]
    for entry in details:
        add_reason(reasons, entry["reason"])
    if len(run.blocks) != 2 or len(census.blocks) != 2:
        add_reason(reasons, "SUMMARY_COUNT_MISMATCH")
    if run.blocked:
        add_reason(reasons, "BLOCKED_PRESENT")
    if run.isolated:
        add_reason(reasons, "ISOLATED_PRESENT")
    if run.r7_iter1_total != 1 or run.r7_iter1_good != 1:
        add_reason(reasons, "R7_COMMIT_MARKER_MISMATCH")

    wanted_lines = {
        exact_int(row, "line")
        for parsed in (run, census) for block in parsed.blocks
        for row in block.rows if row.get("line", "").isdigit()
    }
    line_map, level_keys = load_deck(deck, wanted_lines)
    run_records, run_classes, class_issues = classify_rows(run, line_map, level_keys)
    census_records, census_classes, census_class_issues = classify_rows(
        census, line_map, level_keys,
    )
    details.extend(class_issues)
    details.extend(census_class_issues)
    for entry in (*class_issues, *census_class_issues):
        add_reason(reasons, entry["reason"])

    rd1_reasons: list[str] = []
    rd6_reasons: list[str] = []
    for iteration in range(min(2, len(run.blocks), len(census.blocks))):
        actual = run.blocks[iteration]
        reference = census.blocks[iteration]
        if len(actual.rows) != expected.selected[iteration]:
            add_reason(rd1_reasons, "SELECTED_ROW_COUNT_MISMATCH")
        if actual.zero_count != expected.zero_stream[iteration]:
            add_reason(rd1_reasons, "ZERO_STREAM_COUNT_MISMATCH")
        if actual.zero_digest != reference.zero_digest:
            add_reason(rd1_reasons, "ZERO_STREAM_BIT_MISMATCH")
        actual_sequence = [(row.get("rank"), row.get("line")) for row in actual.rows]
        reference_sequence = [(row.get("rank"), row.get("line")) for row in reference.rows]
        if actual_sequence != reference_sequence:
            add_reason(rd1_reasons, "SELECTION_SEQUENCE_MISMATCH")
        for reason in summary_valid(actual.summary, expected.selected[iteration], expected, True):
            add_reason(rd6_reasons, reason)
        for reason in summary_valid(reference.summary, expected.selected[iteration], expected, False):
            add_reason(rd6_reasons, f"CENSUS_{reason}")
        shared_actual = {key: value for key, value in actual.summary.items()
                         if key not in ("sprobe_undefined_rows", "_log_line")}
        shared_reference = {key: value for key, value in reference.summary.items()
                            if key != "_log_line"}
        if shared_actual != shared_reference:
            add_reason(rd6_reasons, "SUMMARY_BIT_MISMATCH")
        for block in (actual, reference):
            ok, anchor_reason = anchor_ok(block)
            if not ok:
                add_reason(rd6_reasons, anchor_reason)
    if run.r7_iter1_total != 1 or run.r7_iter1_good != 1:
        add_reason(rd1_reasons, "R7_COMMIT_MARKER_MISMATCH")
    if run.blocked:
        add_reason(rd1_reasons, "BLOCKED_PRESENT")
    if run.isolated:
        add_reason(rd1_reasons, "ISOLATED_PRESENT")
    if len(run.blocks) != 2 or len(census.blocks) != 2:
        add_reason(rd1_reasons, "SUMMARY_COUNT_MISMATCH")

    intersection: set[int] = set()
    if len(run_records) == 2:
        intersection = set(run_records[0]) & set(run_records[1])
        if len(intersection) != expected.intersection:
            add_reason(rd1_reasons, "INTERSECTION_COUNT_MISMATCH")

    rd2_reasons: list[str] = []
    for iteration in range(min(2, len(run_classes))):
        observed = {name: run_classes[iteration].get(name, 0)
                    for name in ("EXP", "MIX", "BL", "HAZ")}
        wanted = {name: expected.classes[iteration].get(name, 0)
                  for name in ("EXP", "MIX", "BL", "HAZ")}
        if observed != wanted:
            add_reason(rd2_reasons, "CLASS_CENSUS_MISMATCH")

    rd3_reasons: list[str] = []
    hazards = hazard_observation(run_records)
    if hazards["representation_violations"]:
        add_reason(rd3_reasons, "HAZARD_REPRESENTATION_INVALID")
    capture_cache: dict[tuple[int, int], dict[str, float]] = {}
    sign_axis: dict[int, str] = {}
    matched_fpc: Counter[str] = Counter()
    matched_neg: Counter[str] = Counter()
    for iteration, table in enumerate(run_records):
        for line, record in table.items():
            if record["class"] in ("HAZ", "NON_TARGET"):
                continue
            try:
                capture_cache[(iteration, line)] = capture_values(record["raw"])
            except ReaderError as exc:
                add_reason(rd3_reasons, exc.reason)
                details.append(issue(exc.reason, {
                    "iteration": iteration, "line": line, "detail": exc.detail,
                }))
    for line in sorted(intersection):
        left = run_records[0][line]
        right = run_records[1][line]
        if left["class"] == "HAZ" or right["class"] == "HAZ":
            if left["class"] != right["class"]:
                add_reason(rd2_reasons, "MATCHED_CLASS_MISMATCH")
            continue
        if left["class"] != right["class"]:
            add_reason(rd2_reasons, "MATCHED_CLASS_MISMATCH")
            continue
        values0 = capture_cache.get((0, line))
        values1 = capture_cache.get((1, line))
        if values0 is None or values1 is None:
            continue
        signs = (values0["chi_effective"], values1["chi_effective"])
        if signs[0] > 0.0 and signs[1] > 0.0:
            sign_axis[line] = "FPC"
            matched_fpc[left["class"]] += 1
        elif signs[0] < 0.0 and signs[1] < 0.0:
            sign_axis[line] = "NEG"
            matched_neg[left["class"]] += 1
        else:
            add_reason(rd3_reasons, "CHI_SIGN_CLASS_INVALID")
            details.append(issue("CHI_SIGN_CLASS_INVALID", {"line": line, "chi": signs}))
    if dict(matched_fpc) != expected.matched_fpc:
        add_reason(rd2_reasons, "MATCHED_FPC_CLASS_CENSUS_MISMATCH")
    if sum(matched_neg.values()) != expected.matched_neg or any(
            name != "BL" and count for name, count in matched_neg.items()):
        add_reason(rd2_reasons, "MATCHED_NEG_CENSUS_MISMATCH")

    reconstruction_max = {"continuum": 0.0, "local": 0.0, "jbar": 0.0}
    reconstruction_failures: list[dict[str, Any]] = []
    for line, axis in sign_axis.items():
        if axis != "FPC":
            continue
        for iteration in (0, 1):
            values = capture_cache[(iteration, line)]
            try:
                rebuilt = reconstruct(values, time_s)
                observed = {
                    "continuum": values["producer_continuum_term"],
                    "local": values["producer_local_emission_term"],
                    "jbar": values["Jbar"],
                }
                for name in reconstruction_max:
                    deviation = relative_deviation(observed[name], rebuilt[name])
                    reconstruction_max[name] = max(reconstruction_max[name], deviation)
                    if not math.isfinite(deviation) or deviation > RECONSTRUCTION_TOLERANCE:
                        reconstruction_failures.append({
                            "iteration": iteration, "line": line,
                            "component": name, "relative_deviation": deviation,
                        })
            except ReaderError as exc:
                reconstruction_failures.append({
                    "iteration": iteration, "line": line,
                    "reason": exc.reason, "detail": exc.detail,
                })
    rd4_reasons: list[str] = []
    if reconstruction_failures:
        add_reason(rd4_reasons, "RECONSTRUCTION_IDENTITY_FAIL")

    mapping = mapping_from_source(source)
    mapping_total = 0
    mapping_mapped = 0
    non_target_rows: list[dict[str, int]] = []
    for table in run_records:
        for record in table.values():
            mapping_total += 1
            if (record["Z"], record["ion"]) in mapping:
                mapping_mapped += 1
            if record["Z"] not in TARGET_Z or record["ion"] != TARGET_ION:
                non_target_rows.append({
                    "iteration": record["iteration"], "line": record["line"],
                    "Z": record["Z"], "ion": record["ion"],
                })
    f_mapped = mapping_mapped / mapping_total if mapping_total else None
    rd5_reasons: list[str] = []
    if mapping_total == 0 or mapping_mapped != mapping_total:
        add_reason(rd5_reasons, "MAPPING_INCOMPLETE")

    matched_rows: list[dict[str, Any]] = []
    d_values: list[float] = []
    d_by_class: dict[str, list[float]] = {"BL": [], "EXP": [], "MIX": []}
    proxy_by_class: dict[str, list[float]] = {"BL": [], "EXP": [], "MIX": []}
    proxy_capture_mismatches: list[dict[str, Any]] = []
    d_failures: list[dict[str, Any]] = []
    sign_counts: dict[str, Counter[str]] = {
        "BL": Counter(), "EXP": Counter(), "MIX": Counter(),
    }
    for line in sorted(intersection):
        record0 = run_records[0][line]
        record1 = run_records[1][line]
        class_name = record0["class"]
        axis = sign_axis.get(line)
        output: dict[str, Any] = {
            "line": line, "Z": record0["Z"], "ion": record0["ion"],
            "class": class_name, "chi_axis": axis,
            "capture_d": None, "proxy_d": None,
        }
        if axis in ("FPC", "NEG"):
            values0 = capture_cache.get((0, line))
            values1 = capture_cache.get((1, line))
            try:
                if values0 is None or values1 is None:
                    raise ReaderError("D_FIELDS_UNAVAILABLE")
                eta0, eta1 = values0["producer_eta"], values1["producer_eta"]
                tau0, tau1 = values0["producer_tau_eff"], values1["producer_tau_eff"]
                if eta0 == 0.0 or tau0 == 0.0 or tau1 == 0.0:
                    raise ReaderError("D_ZERO_DENOMINATOR", str(line))
                source_ratio = (eta1 / eta0) / (tau1 / tau0)
                capture_d = abs(source_ratio - 1.0)
                if not math.isfinite(capture_d):
                    raise ReaderError("D_NONFINITE", str(line))
                output["capture_d"] = capture_d
                output["capture_frozen"] = capture_d <= FROZEN_TOLERANCE
                proxy_eta0 = values0["eta_per_sr"]
                proxy_eta1 = values1["eta_per_sr"]
                proxy_tau0 = values0["tau_effective"]
                proxy_tau1 = values1["tau_effective"]
                if proxy_eta0 == 0.0 or proxy_tau0 == 0.0 or proxy_tau1 == 0.0:
                    raise ReaderError("PROXY_D_ZERO_DENOMINATOR", str(line))
                proxy_ratio = (proxy_eta1 / proxy_eta0) / (proxy_tau1 / proxy_tau0)
                proxy_d = abs(proxy_ratio - 1.0)
                if not math.isfinite(proxy_d):
                    raise ReaderError("PROXY_D_NONFINITE", str(line))
                output["proxy_d"] = proxy_d
                if axis == "FPC":
                    d_values.append(capture_d)
                    d_by_class[class_name].append(capture_d)
                    proxy_by_class[class_name].append(proxy_d)
                    delta_s = source_ratio - 1.0
                    j0, j1 = values0["Jbar"], values1["Jbar"]
                    delta_j = j1 - j0
                    if delta_s == 0.0 or delta_j == 0.0:
                        sign_counts[class_name]["unevaluated_zero"] += 1
                    elif (delta_s > 0.0) == (delta_j > 0.0):
                        sign_counts[class_name]["match"] += 1
                    else:
                        sign_counts[class_name]["mismatch"] += 1
            except ReaderError as exc:
                d_failures.append({"line": line, "reason": exc.reason, "detail": exc.detail})
        matched_rows.append(output)

    # E-5(1) is an all-defined-row observation, not merely a matched-row one.
    for (iteration, line), values in sorted(capture_cache.items()):
        for capture_name, proxy_name in (
            ("producer_eta", "eta_per_sr"),
            ("producer_tau_eff", "tau_effective"),
        ):
            if values[capture_name].hex() != values[proxy_name].hex():
                proxy_capture_mismatches.append({
                    "iteration": iteration, "line": line,
                    "capture_field": capture_name,
                    "proxy_field": proxy_name,
                    "capture": values[capture_name],
                    "proxy": values[proxy_name],
                })

    rd7_reasons: list[str] = []
    expected_fpc_total = sum(expected.matched_fpc.values())
    if len(d_values) != expected_fpc_total or d_failures:
        add_reason(rd7_reasons, "D_CENSUS_INCOMPLETE")
    ff = (sum(value <= FROZEN_TOLERANCE for value in d_values) / len(d_values)
          if len(d_values) == expected_fpc_total and d_values else None)

    iter0_source_over_b: list[float] = []
    iter0_jbar_over_b: list[float] = []
    iter0_positive_rows: list[dict[str, Any]] = []
    if run_records:
        for line, record in run_records[0].items():
            if record["class"] in ("HAZ", "NON_TARGET"):
                continue
            values = capture_cache.get((0, line))
            if values is None or values["chi_effective"] <= 0.0:
                continue
            b_nu = planck_nu(values["nu"])
            source_over_b = (values["producer_eta"] / values["chi_effective"]) / b_nu
            jbar_over_b = values["Jbar"] / b_nu
            if not math.isfinite(source_over_b) or not math.isfinite(jbar_over_b):
                details.append(issue("E5_NONFINITE_RATIO", {"line": line}))
                continue
            iter0_source_over_b.append(source_over_b)
            iter0_jbar_over_b.append(jbar_over_b)
            iter0_positive_rows.append({
                "line": line, "Z": record["Z"], "ion": record["ion"],
                "class": record["class"], "S_over_B": source_over_b,
                "Jbar_over_B": jbar_over_b,
            })
    jbar_distribution = distribution(iter0_jbar_over_b)
    sigma = ("SIGNAL" if jbar_distribution is not None and
             jbar_distribution["q10"] < 0.95 else "WEAK-SIGNAL")

    gates = {
        "RD-1": {
            "status": "PASS" if not rd1_reasons else "FAIL",
            "reasons": rd1_reasons,
            "selected_rows": [len(block.rows) for block in run.blocks],
            "all_summaries": [block.summary for block in run.blocks],
            "intersection": len(intersection),
            "zero_stream_rows": [block.zero_count for block in run.blocks],
            "blocked": len(run.blocked), "isolated": len(run.isolated),
            "r7_iter1_markers": run.r7_iter1_total,
            "r7_iter1_exact_markers": run.r7_iter1_good,
        },
        "RD-2": {
            "status": "PASS" if not rd2_reasons else "FAIL",
            "reasons": rd2_reasons,
            "class_census": [dict(counts) for counts in run_classes],
            "matched_FPC": dict(matched_fpc),
            "matched_NEG": dict(matched_neg),
        },
        "RD-3": {
            "status": "PASS" if not rd3_reasons else "FAIL",
            "reasons": rd3_reasons,
            "defined_nonhazard_rows": len(capture_cache),
            "hazards": hazards,
        },
        "RD-4": {
            "status": "PASS" if not rd4_reasons else "FAIL",
            "reasons": rd4_reasons,
            "tolerance": RECONSTRUCTION_TOLERANCE,
            "maximum_relative_deviation": reconstruction_max,
            "failures": reconstruction_failures,
            "subtractions_or_inversions_for_reconstruction": 0,
        },
        "RD-5": {
            "status": "PASS" if not rd5_reasons else "FAIL",
            "reasons": rd5_reasons,
            "mapped": mapping_mapped, "total": mapping_total,
            "f_mapped": f_mapped,
        },
        "RD-6": {
            "status": "PASS" if not rd6_reasons else "FAIL",
            "reasons": rd6_reasons,
            "anchor_line": ANCHOR_LINE,
            "anchor_bits": ANCHOR_BITS,
        },
    }
    for gate in gates.values():
        for reason in gate["reasons"]:
            add_reason(reasons, reason)

    gate_failures = [name for name, gate in gates.items() if gate["status"] != "PASS"]
    authority_violation = bool(non_target_rows) or (f_mapped is not None and f_mapped < 1.0)
    if non_target_rows:
        add_reason(reasons, "NON_TARGET_ROWS")
    # W-V is the registered dedicated disposition for authority violations.
    # If another RD gate also fails, the registered X-D-before-W-V order wins.
    derivative_authority_reasons = {
        "CLASS_CENSUS_MISMATCH", "MATCHED_FPC_CLASS_CENSUS_MISMATCH",
        "INTERSECTION_COUNT_MISMATCH", "SELECTED_ROW_COUNT_MISMATCH",
    }
    non_authority_gate_failure = False
    for name in gate_failures:
        if name == "RD-5" and authority_violation:
            continue
        gate_reasons = set(gates[name]["reasons"])
        if authority_violation and gate_reasons and gate_reasons <= derivative_authority_reasons:
            continue
        non_authority_gate_failure = True
    if non_authority_gate_failure:
        branch = "X-D"
    elif authority_violation:
        branch = "W-V"
    elif ff is None:
        branch = "X-D"
        add_reason(reasons, "D_CENSUS_INCOMPLETE")
    elif ff <= FF_RESPONSE_MAX:
        branch = "C-R"
    elif ff >= FF_FROZEN_MIN:
        branch = "C-F"
    else:
        frozen = {row["line"] for row in matched_rows
                  if row.get("chi_axis") == "FPC" and row.get("capture_frozen")}
        bl = {row["line"] for row in matched_rows
              if row.get("chi_axis") == "FPC" and row.get("class") == "BL"}
        responsive_frozen = {row["line"] for row in matched_rows
                             if row.get("chi_axis") == "FPC" and
                             row.get("class") in ("EXP", "MIX") and
                             row.get("capture_frozen")}
        branch = "C-M-S" if frozen == bl and not responsive_frozen else "C-M-U"

    gates["RD-7"] = {
        "status": "PASS" if branch in ("C-R", "C-F", "C-M-S", "C-M-U") else "FAIL",
        "gate_conjunct": {name: gates[name]["status"] for name in (
            "RD-1", "RD-2", "RD-3", "RD-4", "RD-5", "RD-6",
        )},
        "ff": ff, "frozen_tolerance": FROZEN_TOLERANCE,
        "response_threshold": FF_RESPONSE_MAX,
        "frozen_threshold": FF_FROZEN_MIN,
        "sigma": sigma, "branch": branch,
    }
    primary_branch = "C-M" if branch in ("C-M-S", "C-M-U") else branch
    selected_inventory = [
        [
            {
                "iteration": iteration, "line": record["line"],
                "rank": record["rank"], "Z": record["Z"], "ion": record["ion"],
                "class": record["class"],
                "chi_axis_if_matched": sign_axis.get(record["line"]),
                "capture_defined": (iteration, record["line"]) in capture_cache,
            }
            for record in sorted(table.values(), key=lambda item: item["rank"])
        ]
        for iteration, table in enumerate(run_records)
    ]
    return {
        "schema": SCHEMA,
        "mode": A210_L6C_REDISPATCH,
        "status": "PASS" if gates["RD-7"]["status"] == "PASS" else "FAIL",
        "branch": branch,
        "verdict": branch,
        "primary_branch": primary_branch,
        "blocking_reasons": reasons,
        "error_details": details,
        "gates": gates,
        "registered_formula": "abs((eta1/eta0)/(tau1/tau0)-1)",
        "registered_denominator": "matched FPC rows including BL",
        "ff": ff,
        "sigma": sigma,
        "f_mapped": f_mapped,
        "d_distribution": distribution(d_values),
        "d_distribution_by_class": {
            name: distribution(values) for name, values in d_by_class.items()
        },
        "proxy_d_distribution_by_class": {
            name: distribution(values) for name, values in proxy_by_class.items()
        },
        "ff_responsive": (
            sum(value <= FROZEN_TOLERANCE
                for name in ("EXP", "MIX") for value in d_by_class[name]) /
            sum(len(d_by_class[name]) for name in ("EXP", "MIX"))
            if sum(len(d_by_class[name]) for name in ("EXP", "MIX")) else None
        ),
        "matched_rows": matched_rows,
        "selected_row_inventory": selected_inventory,
        "negative_rows": [row for row in matched_rows if row.get("chi_axis") == "NEG"],
        "non_target_rows": non_target_rows,
        "d_failures": d_failures,
        "observations": {
            "proxy_capture_bit_mismatches": proxy_capture_mismatches,
            "proxy_capture_bit_equal": not proxy_capture_mismatches,
            "sign_agreement_by_class": {
                name: dict(counts) for name, counts in sign_counts.items()
            },
            "iter0_source_over_B": distribution(iter0_source_over_b),
            "iter0_Jbar_over_B": jbar_distribution,
            "iter0_positive_chi_rows": iter0_positive_rows,
            "hazard_capture_decode": hazards,
        },
        "physical_values_modified": False,
        "clamp": 0, "floor": 0, "cap": 0, "repair": 0,
    }


def verdict_candidates(docs: Path) -> list[Path]:
    if not docs.is_dir() or docs.is_symlink():
        return []
    return sorted(path for path in docs.iterdir()
                  if path.is_file() and not path.is_symlink() and
                  path.suffix == ".md" and "VERDICT" in path.name.upper() and
                  "Z2A" in path.name.upper())


def verdict_has_pr0_anchors(text: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    zi_a_lines = [line for line in text.splitlines() if "ZI-A" in line]
    if not any(re.search(r"(R-II|분기|판정|착지)", line) for line in zi_a_lines):
        missing.append("ZI-A")
    for name in ("E-I3", "E-I4"):
        lines = [line for line in text.splitlines() if name in line]
        if not any(re.search(r"\bPASS\b", line) for line in lines):
            missing.append(f"{name}_PASS")
    audit_joint = re.search(r"감리\s*A[·ㆍ・/]B\s*통과", text)
    audit_separate = (
        re.search(r"감리[- ]?A[^\n]{0,120}\bPASS\b", text) and
        re.search(r"감리[- ]?B[^\n]{0,120}\bPASS\b", text)
    )
    if not (audit_joint or audit_separate):
        missing.append("AUDIT_A_B_PASS")
    return not missing, missing


def check_pr0(docs: Path, sacct_text: str | None = None) -> dict[str, Any]:
    candidates = verdict_candidates(docs)
    if not candidates:
        raise ReaderError("Z2A_VERDICT_ABSENT", str(docs))
    passing: list[Path] = []
    failures: dict[str, list[str]] = {}
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            failures[str(path)] = [f"READ:{type(exc).__name__}"]
            continue
        ok, missing = verdict_has_pr0_anchors(text)
        if ok:
            passing.append(path)
        else:
            failures[str(path)] = missing
    if not passing:
        raise ReaderError("Z2A_VERDICT_NOT_ZI_A_AUDITED", json.dumps(failures, sort_keys=True))
    if len(passing) != 1:
        raise ReaderError("Z2A_VERDICT_AMBIGUOUS", ",".join(map(str, passing)))
    if sacct_text is None:
        result = subprocess.run(
            ["sacct", "-j", RII_JOB_ID, "-X", "-n", "-P", "-o", "JobIDRaw,State,ExitCode"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ReaderError("SACCT_UNAVAILABLE", result.stderr.strip())
        sacct_text = result.stdout
    completed = False
    records: list[str] = []
    for line in sacct_text.splitlines():
        tokens = line.strip().split("|")
        if not tokens or tokens[0] != RII_JOB_ID:
            continue
        records.append(line.strip())
        if len(tokens) >= 2 and tokens[1] == "COMPLETED":
            completed = True
    if not completed:
        raise ReaderError("RII_NOT_COMPLETED", ";".join(records) or "record_absent")
    return {
        "status": "PASS", "z2a_verdict": str(passing[0]),
        "required_anchors": ["ZI-A", "E-I3 PASS", "E-I4 PASS", "audit A/B PASS"],
        "sacct_job": RII_JOB_ID, "sacct_records": records,
        "sealed_inputs_opened": False,
    }


def report_outside_inputs(report: Path, inputs: Iterable[Path]) -> None:
    report_real = Path(os.path.realpath(report))
    for path in inputs:
        root_real = Path(os.path.realpath(path))
        try:
            report_real.relative_to(root_real)
        except ValueError:
            continue
        raise ReaderError("REPORT_INSIDE_SEALED_INPUT", f"{report_real} inside {root_real}")
    if report.exists() and report.is_symlink():
        raise ReaderError("REPORT_PATH_SYMLINK", str(report))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise ReaderError("REPORT_PATH_SYMLINK", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.tmp.", delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def model_rc(run_root: Path) -> int:
    path = run_root / "model.rc"
    regular_file(path, "MODEL_RC_MISSING")
    try:
        raw = path.read_text(encoding="ascii", errors="strict").strip()
        value = int(raw)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReaderError("MODEL_RC_INVALID", str(exc)) from exc
    if str(value) != raw:
        raise ReaderError("MODEL_RC_INVALID", raw)
    return value


def synthetic_source() -> str:
    return (
        "static const int NLTE_TARGET_Z[] = {26,26,27,27,28,28};\n"
        "static const int NLTE_TARGET_ION[] = {1,2,1,2,1,2};\n"
    )


def synthetic_zero(anchor_value: str = "0.0004437688873013011") -> str:
    return (
        f"{ZERO_PREFIX.decode()} phase=REQUESTED_TE shell=0 line={ANCHOR_LINE} "
        "Z=26 ion=1 lower_global=1 upper_global=2 tau_validity=2 tau_raw=0 "
        f"effective_tau=0 n_upper={anchor_value} A_ul=0.52225560000000004 "
        "nu=5e14 emission_per_sr=2.7310010625545125e-16 srce_chk=0 "
        "exact_zero_provenance=1 source_defined=0"
    )


def synthetic_row(line: int, rank: int, z: int, lower: int, upper: int,
                  response: float, iteration: int, time_s: float,
                  hazard: bool = False) -> str:
    nu = 5.0e14 + (line % 1000) * 1.0e9
    if hazard:
        return (
            f"{ROW_PREFIX.decode()} phase=REQUESTED_TE shell=0 rank={rank} line={line} "
            f"Z={z} ion=1 lower_level={lower} upper_level={upper} nu={nu:.17g} "
            "tau_validity=2 chi_effective=0 eta_per_sr=1e-20 tau_effective=0 "
            "Jbar=1e-20 J_cont=UNAVAILABLE S_probe=UNAVAILABLE "
            "independent_fields_defined=0 producer_continuum_term=UNAVAILABLE "
            "producer_local_emission_term=UNAVAILABLE producer_terms_defined=0 "
            "producer_eta=UNAVAILABLE producer_tau_eff=UNAVAILABLE "
            "producer_srce_chk=UNAVAILABLE producer_exact_zero=UNAVAILABLE "
            "producer_raw_defined=0"
        )
    tau = 0.5
    b_nu = planck_nu(nu)
    chi = tau * nu / (C_LIGHT * time_s)
    eta0 = b_nu * chi * 0.5
    eta = eta0 if iteration == 0 else eta0 * response
    beta, companion = exponx(tau)
    j_cont = b_nu * 0.5
    continuum = beta * j_cont
    local = eta * (C_LIGHT * time_s / nu) * companion
    jbar = continuum + local
    return (
        f"{ROW_PREFIX.decode()} phase=REQUESTED_TE shell=0 rank={rank} line={line} "
        f"Z={z} ion=1 lower_level={lower} upper_level={upper} nu={nu:.17g} "
        f"tau_validity=1 chi_effective={chi:.17g} eta_per_sr={eta:.17g} "
        f"tau_effective={tau:.17g} Jbar={jbar:.17g} J_cont={j_cont:.17g} "
        f"S_probe={eta / chi:.17g} independent_fields_defined=1 "
        f"producer_continuum_term={continuum:.17g} "
        f"producer_local_emission_term={local:.17g} producer_terms_defined=1 "
        f"producer_eta={eta:.17g} producer_tau_eff={tau:.17g} "
        "producer_srce_chk=0 producer_exact_zero=0 producer_raw_defined=1"
    )


def fixture_spec(composition: tuple[int, int, int] = (0, 3, 0),
                 response: float = 0.8, hazard: bool = True
                 ) -> list[tuple[int, int, int, int, float, bool]]:
    bl, exp, mix = composition
    result: list[tuple[int, int, int, int, float, bool]] = []
    line = 1000
    for _ in range(bl):
        result.append((line, 26 + line % 3, 100, 101, 1.0, False)); line += 1
    for _ in range(exp):
        result.append((line, 26 + line % 3, 1, 2, response, False)); line += 1
    for _ in range(mix):
        result.append((line, 26 + line % 3, 1, 100, response, False)); line += 1
    if hazard:
        result.append((ANCHOR_LINE, 26, 100, 101, 1.0, True))
    return result


def write_fixture(root: Path, spec: list[tuple[int, int, int, int, float, bool]],
                  *, anchor_value: str = "0.0004437688873013011",
                  blocked: bool = False, remove_r7: bool = False,
                  non_target: bool = False) -> tuple[Path, Path, Path, Expected]:
    run = root / "run"
    census = root / "census"
    deck = root / "deck"
    run.mkdir(parents=True); census.mkdir(); deck.mkdir()
    (run / "model.rc").write_text("0\n", encoding="ascii")
    time_s = 1.0e6
    (deck / "config.json").write_text(json.dumps({"time_explosion_s": time_s}), encoding="utf-8")
    level_keys: set[tuple[int, int, int]] = set()
    line_rows = ["line_id,atomic_number,ion_number,level_number_lower,level_number_upper\n"]
    for index, (line, z, lower, upper, _response, _hazard) in enumerate(spec):
        if non_target and index == 0:
            z = 14
            spec[index] = (line, z, lower, upper, _response, _hazard)
        line_rows.append(f"{line},{z},1,{lower},{upper}\n")
        level_keys.add((z, 1, lower)); level_keys.add((z, 1, upper))
    (deck / "line_list.csv").write_text("".join(line_rows), encoding="utf-8")
    levels = ["atomic_number,ion_number,level_number,energy_eV,g\n"]
    for z, ion, level in sorted(level_keys):
        levels.append(f"{z},{ion},{level},0,1\n")
    (deck / "levels.csv").write_text("".join(levels), encoding="utf-8")
    class_counts = Counter()
    for _line, _z, lower, upper, _response, is_hazard in spec:
        class_counts[
            "HAZ" if is_hazard else
            "BL" if lower >= 100 and upper >= 100 else
            "EXP" if lower < 100 and upper < 100 else "MIX"
        ] += 1
    expected = Expected(
        selected=(len(spec), len(spec)), intersection=len(spec),
        classes=(dict(class_counts), dict(class_counts)),
        matched_fpc={name: class_counts.get(name, 0) for name in ("BL", "EXP", "MIX")
                     if class_counts.get(name, 0)},
        matched_neg=0, zero_stream=(1, 1), candidate_rows=10,
    )
    run_lines: list[str] = []
    census_lines: list[str] = []
    if blocked:
        run_lines.append(f"{BLOCKED_PREFIX.decode()} reason=FIXTURE_BLOCKED")
    for iteration in (0, 1):
        run_lines.append(synthetic_zero(anchor_value))
        census_lines.append(synthetic_zero())
        for rank, (line, z, lower, upper, response_i, is_hazard) in enumerate(spec, 1):
            row = synthetic_row(line, rank, z, lower, upper, response_i,
                                iteration, time_s, is_hazard)
            run_lines.append(row)
            # Census is capture-off in production.  Removing the suffix here
            # is unnecessary for the reader; only shared summary tokens and
            # selected identity order are compared.
            census_lines.append(row)
        common = (
            f"{SUMMARY_PREFIX.decode()} phase=REQUESTED_TE shell=0 target_Z=26,27,28 "
            f"target_ion=1 candidate_rows=10 selected_rows={len(spec)} "
            "total_scaled_emission=1 selected_scaled_emission=1 selected_fraction=1 "
            "selection_target_fraction=0.9 selected_reaches_target=1 complete=1 "
            "interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 clamp=0 floor=0 "
            "cap=0 jitter=0 repair=0 zero_opacity_emitting_rows=1"
        )
        run_lines.append(common + " sprobe_undefined_rows=1")
        census_lines.append(common)
        if iteration == 1 and not remove_r7:
            run_lines.append(
                f"{R7_PREFIX.decode()} lane=DET iter=1 phase=A2-10 te_generation=2->3"
            )
    (run / "stderr.log").write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    (census / "stderr.log").write_text("\n".join(census_lines) + "\n", encoding="utf-8")
    return run, census, deck, expected


def mutate_first(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ReaderError("SELFTEST_MUTATION_TARGET_MISSING", old)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def control(name: str, action: str, report: dict[str, Any]) -> str:
    reasons = ",".join(report.get("blocking_reasons", []))
    measured_reason = reasons or (
        report["branch"] if report["branch"] in ("C-F", "C-M-S", "C-M-U") else "PASS"
    )
    return (f"{name} {action} status={report['status']} reason={measured_reason} "
            f"branch={report['branch']}")


def negative_landing_control(name: str, action: str, report: dict[str, Any],
    negative_branch: str) -> str:
    landed = report["branch"] == negative_branch
    reason = negative_branch if landed else f"EXPECTED_{negative_branch}_GOT_{report['branch']}"
    return (f"{name} {action} status=FAIL reason={reason} "
            f"branch={report['branch']} analyzer_status={report['status']}")


def run_fixture(root: Path, spec: list[tuple[int, int, int, int, float, bool]],
                **kwargs: Any) -> tuple[dict[str, Any], tuple[Path, Path, Path, Expected]]:
    paths = write_fixture(root, spec, **kwargs)
    run, census, deck, expected = paths
    report = analyze(run / "stderr.log", census / "stderr.log", deck,
                     synthetic_source(), expected)
    return report, paths


def selftest() -> int:
    failures = 0

    def measured(spec: list[tuple[int, int, int, int, float, bool]],
                 **kwargs: Any) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="l6cr-restored-") as directory:
            report, _paths = run_fixture(Path(directory), spec, **kwargs)
            return report

    def expect(name: str, report: dict[str, Any], branch: str,
               reason: str | None = None) -> None:
        nonlocal failures
        if report["branch"] != branch or (reason and reason not in report["blocking_reasons"]):
            failures += 1

    def report_gate_cli(run: Path, census: Path, deck: Path,
                        report: Path, repo: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--run-root", str(run), "--census-root", str(census),
             "--deck", str(deck), "--report", str(report),
             "--expected-head", "SYNTHETIC", "--repo-root", str(repo)],
            capture_output=True, text=True,
        )

    clean_spec = fixture_spec()
    with tempfile.TemporaryDirectory(prefix="l6cr-r0-") as directory:
        report, _paths = run_fixture(Path(directory), clean_spec.copy())
        expect("NC-R0", report, "C-R")
        print(control("NC-R0", "inject=NONE", report))

    with tempfile.TemporaryDirectory(prefix="l6cr-r1-") as directory:
        report, _paths = run_fixture(Path(directory), fixture_spec((3, 0, 0)))
        expect("NC-R1", report, "C-F")
        print(negative_landing_control("NC-R1", "inject=FROZEN", report, "C-F"))
        restored = measured(clean_spec.copy())
        expect("NC-R1-restore", restored, "C-R")
        print(control("NC-R1", "remove=RESPONSE", restored))

    with tempfile.TemporaryDirectory(prefix="l6cr-r2-") as directory:
        report, _paths = run_fixture(Path(directory), clean_spec.copy())
        expect("NC-R2", report, "C-R")
        print(control("NC-R2", "inject=RESPONSE", report))
        restored = measured(fixture_spec((3, 0, 0)))
        expect("NC-R2-restore", restored, "C-F")
        print(control("NC-R2", "remove=FROZEN", restored))

    with tempfile.TemporaryDirectory(prefix="l6cr-r3-") as directory:
        report, _paths = run_fixture(
            Path(directory), fixture_spec((95, 284, 506), hazard=False),
        )
        expect("NC-R3", report, "C-M-S")
        print(control("NC-R3", "inject=PROXY_COMPOSITION", report))
        restored = measured(fixture_spec((0, 885, 0), hazard=False))
        expect("NC-R3-restore", restored, "C-R")
        print(control("NC-R3", "remove=ALL_RESPONSE", restored))

    boundary_response = measured(fixture_spec((1, 9, 0), hazard=False))
    expect("NC-R2-boundary-0.10", boundary_response, "C-R")
    print(control("NC-R2", "boundary=FF_EXACT_0.10", boundary_response))
    boundary_frozen = measured(fixture_spec((99, 1, 0), hazard=False))
    expect("NC-R1-boundary-0.99", boundary_frozen, "C-F")
    print(control("NC-R1", "boundary=FF_EXACT_0.99", boundary_frozen))

    with tempfile.TemporaryDirectory(prefix="l6cr-r4a-") as directory:
        report, _paths = run_fixture(Path(directory), clean_spec.copy(), remove_r7=True)
        expect("NC-R4", report, "X-D", "R7_COMMIT_MARKER_MISMATCH")
        print(control("NC-R4", "inject=DELETE_R7", report))
        restored = measured(clean_spec.copy())
        expect("NC-R4-restore-r7", restored, "C-R")
        print(control("NC-R4", "remove=RESTORE_R7", restored))
    with tempfile.TemporaryDirectory(prefix="l6cr-r4b0-") as directory:
        report, _paths = run_fixture(Path(directory), clean_spec.copy(), blocked=True)
        expect("NC-R4", report, "X-D", "BLOCKED_PRESENT")
        print(control("NC-R4", "inject=BLOCKED", report))
        restored = measured(clean_spec.copy())
        expect("NC-R4-restore-blocked", restored, "C-R")
        print(control("NC-R4", "remove=REMOVE_BLOCKED", restored))

    with tempfile.TemporaryDirectory(prefix="l6cr-r4b-") as directory:
        report, paths = run_fixture(Path(directory), clean_spec.copy())
        run, _census, deck, expected = paths
        hazard_spec = next(item for item in clean_spec if item[-1])
        hazard_rank = clean_spec.index(hazard_spec) + 1
        hazard_line = synthetic_row(
            hazard_spec[0], hazard_rank, hazard_spec[1], hazard_spec[2],
            hazard_spec[3], hazard_spec[4], 0, 1.0e6, True,
        )
        mutate_first(run / "stderr.log", hazard_line + "\n", "")
        injected = analyze(run / "stderr.log", paths[1] / "stderr.log", deck,
                           synthetic_source(), expected)
        expect("NC-R4b", injected, "X-D", "SELECTED_ROW_COUNT_MISMATCH")
        print(control("NC-R4b", "inject=DELETE_HAZARD_ROW", injected))
        print(control("NC-R4b", "remove=RESTORE_HAZARD_ROW", report))

    with tempfile.TemporaryDirectory(prefix="l6cr-r5-") as directory:
        clean, paths = run_fixture(Path(directory), clean_spec.copy())
        run, census, deck, expected = paths
        text = (run / "stderr.log").read_text(encoding="utf-8")
        match = re.search(r"producer_eta=([^\s]+)", text)
        if match is None:
            raise ReaderError("SELFTEST_MUTATION_TARGET_MISSING", "producer_eta")
        old = match.group(1)
        mutate_first(run / "stderr.log", f"producer_eta={old}",
                     f"producer_eta={float(old) * (1.0 + 1.0e-9):.17g}")
        injected = analyze(run / "stderr.log", census / "stderr.log", deck,
                           synthetic_source(), expected)
        expect("NC-R5", injected, "X-D", "RECONSTRUCTION_IDENTITY_FAIL")
        print(control("NC-R5", "inject=ETA_PLUS_1E-9", injected))
        print(control("NC-R5", "remove=RESTORE_ETA", clean))

    with tempfile.TemporaryDirectory(prefix="l6cr-r6-") as directory:
        clean, paths = run_fixture(Path(directory), clean_spec.copy())
        run, census, deck, expected = paths
        source = synthetic_source().replace("26,26,27,27,28,28", "99,26,27,27,28,28", 1)
        injected = analyze(run / "stderr.log", census / "stderr.log", deck, source, expected)
        expect("NC-R6", injected, "W-V", "MAPPING_INCOMPLETE")
        print(control("NC-R6", "inject=REMOVE_26_1_MAPPING", injected))
        print(control("NC-R6", "remove=RESTORE_MAPPING", clean))

    with tempfile.TemporaryDirectory(prefix="l6cr-r7-") as directory:
        clean, paths = run_fixture(Path(directory), clean_spec.copy())
        run, census, deck, expected = paths
        mutate_first(run / "stderr.log", "A_ul=0.52225560000000004",
                     "A_ul=0.52225560000000015")
        injected = analyze(run / "stderr.log", census / "stderr.log", deck,
                           synthetic_source(), expected)
        expect("NC-R7", injected, "X-D", "ANCHOR_FIELD_MISMATCH")
        print(control("NC-R7", "inject=ANCHOR_FIELD", injected))
        print(control("NC-R7", "remove=RESTORE_ANCHOR", clean))

    with tempfile.TemporaryDirectory(prefix="l6cr-r8-") as directory:
        injected, _paths = run_fixture(Path(directory), clean_spec.copy(), non_target=True)
        expect("NC-R8", injected, "W-V")
        print(control("NC-R8", "inject=NON_TARGET", injected))
        restored = measured(clean_spec.copy())
        expect("NC-R8-restore", restored, "C-R")
        print(control("NC-R8", "remove=RESTORE_TARGET", restored))

    with tempfile.TemporaryDirectory(prefix="l6cr-r9-") as directory:
        clean, paths = run_fixture(Path(directory), clean_spec.copy())
        run, census, deck, expected = paths
        mutate_first(deck / "levels.csv", ",1,1,0,1\n", ",1,999,0,1\n")
        injected = analyze(run / "stderr.log", census / "stderr.log", deck,
                           synthetic_source(), expected)
        expect("NC-R9", injected, "X-D", "DECK_LEVEL_JOIN_MISSING")
        print(control("NC-R9", "inject=DECK_LEVEL_NUMBER", injected))
        print(control("NC-R9", "remove=RESTORE_DECK", clean))

    with tempfile.TemporaryDirectory(prefix="l6cr-r10-") as directory:
        docs = Path(directory) / "docs"
        docs.mkdir()
        try:
            check_pr0(docs, f"{RII_JOB_ID}|COMPLETED|0:0\n")
        except ReaderError as exc:
            if exc.reason != "Z2A_VERDICT_ABSENT":
                failures += 1
            print("NC-R10 inject=Z2A_VERDICT_ABSENT status=FAIL "
                  f"reason={exc.reason} branch=X-P sealed_access=0")
        else:
            failures += 1
        verdict = docs / "VERDICT_A210_Z2A_FIXTURE.md"
        verdict.write_text(
            "R-II branch ZI-A\n| E-I3 | PASS |\n| E-I4 | PASS |\n감리 A·B 통과\n",
            encoding="utf-8",
        )
        restored = check_pr0(docs, f"{RII_JOB_ID}|COMPLETED|0:0\n")
        print("NC-R10 remove=RESTORE_Z2A_VERDICT status=PASS reason=PASS "
              f"branch=PR-0 sealed_access={int(restored['sealed_inputs_opened'])}")

    with tempfile.TemporaryDirectory(prefix="l6cr-r11-") as directory:
        root = Path(directory)
        run, census, deck = root / "run", root / "census", root / "deck"
        run.mkdir(); census.mkdir(); deck.mkdir()
        sealed_file = run / "model.rc"
        original = b"SEALED_MODEL_RC_SENTINEL\n"
        sealed_file.write_bytes(original)
        injected = report_gate_cli(run, census, deck, sealed_file, root)
        match = re.search(r"reason=([A-Z0-9_]+)", injected.stderr)
        stderr_reason = match.group(1) if match else "ABSENT"
        byte_unchanged = sealed_file.read_bytes() == original
        if (injected.returncode != 4 or
                stderr_reason != "REPORT_INSIDE_SEALED_INPUT" or not byte_unchanged):
            failures += 1
        print("NC-R11 inject=REPORT_EXISTING_SEALED_FILE status=FAIL "
              f"reason={stderr_reason} branch=X-P rc={injected.returncode} "
              f"byte_unchanged={int(byte_unchanged)}")
        outside_report = root / "report.json"
        report_outside_inputs(outside_report, (run, census, deck))
        print("NC-R11 remove=REPORT_OUTSIDE_INPUTS status=PASS reason=PASS "
              "branch=PR-0 sealed_access=0")

    with tempfile.TemporaryDirectory(prefix="l6cr-r12-") as directory:
        root = Path(directory)
        run, census, deck = root / "run", root / "census", root / "deck"
        run.mkdir(); census.mkdir(); deck.mkdir()
        nested_report = run / "subdir" / "r.json"
        injected = report_gate_cli(run, census, deck, nested_report, root)
        match = re.search(r"reason=([A-Z0-9_]+)", injected.stderr)
        stderr_reason = match.group(1) if match else "ABSENT"
        path_created = nested_report.exists()
        parent_created = nested_report.parent.exists()
        if (injected.returncode != 4 or
                stderr_reason != "REPORT_INSIDE_SEALED_INPUT" or
                path_created or parent_created):
            failures += 1
        print("NC-R12 inject=REPORT_NEW_SEALED_SUBPATH status=FAIL "
              f"reason={stderr_reason} branch=X-P rc={injected.returncode} "
              f"path_created={int(path_created)} parent_created={int(parent_created)}")
        outside_report = root / "report.json"
        report_outside_inputs(outside_report, (run, census, deck))
        print("NC-R12 remove=REPORT_OUTSIDE_INPUTS status=PASS reason=PASS "
              "branch=PR-0 sealed_access=0")

    print(f"A210_L6C_REDISPATCH_SELFTEST {'PASS' if failures == 0 else 'FAIL'} "
          f"controls=NC-R0,NC-R1,NC-R2,NC-R3,NC-R4,NC-R4b,NC-R5,NC-R6,NC-R7,NC-R8,NC-R9,NC-R10,NC-R11,NC-R12 "
          f"failed={failures} synthetic_fixtures=1 sealed_access=0")
    return 0 if failures == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--census-root", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    return parser.parse_args()


def failure_report(reason: str, detail: str, branch: str,
                   pr0: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "mode": A210_L6C_REDISPATCH,
        "status": "FAIL", "branch": branch, "verdict": branch,
        "blocking_reasons": [reason], "detail": detail,
        "PR-0": pr0 or {"status": "FAIL", "sealed_inputs_opened": False},
        "physical_values_modified": False,
        "clamp": 0, "floor": 0, "cap": 0, "repair": 0,
    }


def main() -> int:
    args = parse_args()
    if args.selftest:
        try:
            return selftest()
        except (ReaderError, OSError, UnicodeError) as exc:
            reason = exc.reason if isinstance(exc, ReaderError) else type(exc).__name__
            detail = exc.detail if isinstance(exc, ReaderError) else str(exc)
            print(f"A210_L6C_REDISPATCH_SELFTEST FAIL reason={reason} detail={detail}",
                  file=sys.stderr)
            return 4
    required = {
        "run-root": args.run_root, "census-root": args.census_root,
        "deck": args.deck, "report": args.report,
        "expected-head": args.expected_head,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        print("A210_L6C_REDISPATCH_FAIL branch=X-P reason=ARGUMENT_REQUIRED "
              f"detail={','.join(missing)}", file=sys.stderr)
        return 4
    assert args.run_root is not None and args.census_root is not None
    assert args.deck is not None and args.report is not None
    assert args.expected_head is not None
    pr0: dict[str, Any] | None = None
    try:
        report_outside_inputs(args.report, (args.run_root, args.census_root, args.deck))
        # Source blob/head and repository documents are unsealed preflight inputs.
        source = source_at_head(args.repo_root, args.expected_head)
        pr0 = check_pr0(args.repo_root / "docs")
    except (ReaderError, OSError, UnicodeError) as exc:
        reason = exc.reason if isinstance(exc, ReaderError) else type(exc).__name__
        detail = exc.detail if isinstance(exc, ReaderError) else str(exc)
        if reason != "REPORT_INSIDE_SEALED_INPUT":
            payload = failure_report(reason, detail, "X-P", pr0)
            try:
                atomic_write_json(args.report, payload)
            except (ReaderError, OSError) as report_exc:
                report_reason = (report_exc.reason if isinstance(report_exc, ReaderError)
                                 else type(report_exc).__name__)
                print("A210_L6C_REDISPATCH_FAIL branch=X-P "
                      f"reason=REPORT_WRITE_FAILED detail={report_reason}", file=sys.stderr)
        print(f"A210_L6C_REDISPATCH_FAIL branch=X-P reason={reason}" +
              (f" detail={detail}" if detail else ""), file=sys.stderr)
        return 4

    try:
        # This is the first production point at which sealed inputs may open.
        rc = model_rc(args.run_root)
        report = analyze(
            args.run_root / "stderr.log", args.census_root / "stderr.log",
            args.deck, source, REGISTERED,
        )
        if rc != 0:
            report["status"] = "FAIL"
            report["branch"] = "X-D"
            add_reason(report["blocking_reasons"], "MODEL_NOT_COMPLETE")
        report["PR-0"] = {**pr0, "sealed_inputs_opened": True}
        report["expected_head"] = args.expected_head
        report["inputs"] = {
            "run_root": str(args.run_root), "census_root": str(args.census_root),
            "deck": str(args.deck), "streaming_log_reader": True,
        }
        atomic_write_json(args.report, report)
    except (ReaderError, OSError, UnicodeError, csv.Error) as exc:
        reason = exc.reason if isinstance(exc, ReaderError) else type(exc).__name__
        detail = exc.detail if isinstance(exc, ReaderError) else str(exc)
        payload = failure_report(reason, detail, "X-D",
                                 {**pr0, "sealed_inputs_opened": True})
        atomic_write_json(args.report, payload)
        print(f"A210_L6C_REDISPATCH_FAIL branch=X-D reason={reason}" +
              (f" detail={detail}" if detail else ""), file=sys.stderr)
        return 4
    print("A210_L6C_REDISPATCH_RESULT "
          f"status={report['status']} branch={report['branch']} "
          f"ff={report.get('ff')} sigma={report.get('sigma')} "
          f"f_mapped={report.get('f_mapped')} report={args.report}")
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
