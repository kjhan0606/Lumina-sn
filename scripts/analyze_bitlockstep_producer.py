#!/usr/bin/env python3
"""Read the sealed BIT-LOCKSTEP evidence without assigning a producer.

This reader implements the numerical D1--D5 discriminants and the G-B2/G-B3
integrity gates registered in RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md.  Its
``branch`` is a numerical partition only: source-audit closure and every
producer attribution remain the adjudicator's work.

The normal mode opens sealed inputs only for reading.  ``--selftest`` creates
small, disposable fixtures in a temporary directory and never opens a sealed
path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


MODE = "A210_BITLOCKSTEP_PRODUCER"
RESULT_SYMBOL = "A210_BITLOCKSTEP_PRODUCER_RESULT"

DEFAULT_RUN_ROOT = Path(
    "/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/"
    "zo_20260826T050021Z_census_ion1"
)
DEFAULT_PAIR_LIST = Path(
    "/gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv"
)
EXPECTED_PAIRLIST_SHA256 = (
    "ac7f3ca6def9738acc452121e844af08c4aba80bc83c90c14eb7683db7f75f01"
)
EXPECTED_TOTAL = 221_382
TARGET_Z = (26, 27, 28)
EXPECTED_ROWS_PER_Z = {26: 106_502, 27: 105_685, 28: 9_195}
EXPECTED_D3_ABS_FORMAT = ("1.1686e-02", "1.4416e-02", "1.9996e-02")
ANCHOR_LINE = 669_992
ANCHOR_FIELDS = {
    "tau_validity": 2,
    "n_upper": "0x1.d1534dba20c26p-12",
    "A_ul": "0x1.0b6516044e436p-1",
    "emission_per_sr": "0x1.3adcec9caadc0p-52",
}

ZERO_ROW_PREFIX = "[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"
SELECTED_ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


@dataclass(frozen=True)
class Citation:
    path: str
    line: int
    text: str


# G-B3 is deliberately an existence check, not the A1 writer audit.  Each item
# is an exact contractual source citation in §2-3 or §9.
CITATIONS = (
    Citation("src/lumina_atomic.c", 1468, 'getenv("LUMINA_SUPER_CUTOFF")'),
    Citation("src/lumina_atomic.c", 1473, "atom->level_super[l] = K"),
    Citation("src/lumina_atomic.c", 1474, "atom->level_super[l] = atom->level_num[l]"),
    Citation("src/lumina_atomic.c", 1476, "[ARTIS super-cutoff] K=%d"),
    Citation("src/lumina_plasma.c", 16336, "atom->level_Z[l] != target_Z[i]"),
    Citation("src/lumina_plasma.c", 16344, "nlte->nlte_ion_super_offset[i + 1]"),
    Citation("src/lumina_plasma.c", 16350, "nlte->super_mode = super_requested"),
    Citation("src/lumina_plasma.c", 16383, "nlte->nlte_ion_level_offset[i] + cursor[i]++"),
    Citation("src/lumina_plasma.c", 16384, "atom->level_super[l]"),
    Citation("src/lumina_plasma.c", 16387, "nlte->fl_to_super[g] = sl"),
    Citation("src/lumina_plasma.c", 16456, 'getenv("LUMINA_SUPER_LEVELS")'),
    Citation("src/lumina_plasma.c", 16483, "[NLTE] Super-levels:"),
    Citation("src/population_contract.c", 172, "population_within_superlevel_build"),
    Citation("src/population_contract.c", 199, "if (!super_mode)"),
    Citation("src/population_contract.c", 200, "work[i] = 1.0"),
    Citation("src/population_contract.c", 248, "(double)atomic->g[global_level]"),
    Citation("src/lumina_plasma.c", 16993, "#define SOLVE_OF(fl_g)"),
    Citation("src/lumina_plasma.c", 16994, "#define FRAC_OF(fl_g)"),
    Citation("src/lumina_plasma.c", 19025, "nlte_solve_ion_shell"),
    Citation("src/lumina_plasma.c", 19042, "nlte->nlte_ion_super_offset[ion_idx_hi + 1]"),
    Citation("src/lumina_plasma.c", 19043, "nlte->nlte_ion_level_offset[ion_idx_hi + 1]"),
    Citation("src/lumina_plasma.c", 19386, "Redistribute super-level solution to full levels"),
    Citation("src/lumina_plasma.c", 19397, "nlte->within_sl_frac"),
    Citation("src/lumina_plasma.c", 19398, "xfl[i] = b[sl] * f"),
    Citation("src/lumina_plasma.c", 19418, "nlte->nlte_level_populations"),
    Citation("src/lumina_plasma.c", 19422, "nlte->nlte_level_populations"),
    Citation("src/lumina_plasma.c", 19433, "nlte->nlte_level_populations"),
    Citation("src/lumina_plasma.c", 19633, "nlte_precompute_within_sl_frac_checked"),
    Citation("src/lumina_plasma.c", 19831, "nlte_precompute_within_sl_frac_checked"),
    Citation("src/lumina.h", 723, "When super_mode==0"),
    Citation("src/nlte_population_candidate.c", 226, "candidate->nlte = *public_nlte"),
    Citation("src/nlte_population_candidate.c", 267, "candidate->nlte.nlte_level_populations"),
    Citation("src/nlte_population_candidate.c", 385, "population_within_superlevel_build"),
    Citation("src/lumina_plasma.c", 20641, "nlte_solve_all_impl("),
    Citation("src/lumina_plasma.c", 20043, "nlte_solve_ion_shell("),
    Citation("src/lumina_plasma.c", 16060, "lumina_init_seed_material_predictor"),
    Citation("src/lumina_plasma.c", 16202, "nlte_population_candidate_commit_seed_material"),
    Citation("src/lumina_cmfgen.c", 7525, "lumina_init_seed_material_predictor"),
    Citation("src/lumina_cuda.cu", 1049, "nlte_solve_all_gpu"),
    Citation("src/lumina_cuda.cu", 1075, "[NLTE-GPU][FAIL-CLOSED-ROUTE]"),
    Citation("src/lumina_cuda.cu", 1080, "return nlte_solve_all("),
    Citation("src/lumina_cuda.cu", 1083, "Forensic-only legacy implementation"),
    Citation("src/lumina_plasma.c", 8925, "global_to_nlte_level[a.lower_global]"),
    Citation("src/lumina_plasma.c", 8926, "global_to_nlte_level[a.upper_global]"),
    Citation("src/lumina_plasma.c", 8927, "if(a.nlte_lo<0||a.nlte_up<0)return a"),
    Citation("src/lumina_plasma.c", 6893, "tau_sobolev_require_refresh"),
    Citation("src/lumina_plasma.c", 6903, "tau_sobolev_mark_computed"),
    Citation("src/lumina_plasma.c", 6913, "tau_sobolev_assert_fresh"),
    Citation("src/lumina_plasma.c", 19471, "tau_sobolev_require_refresh"),
    Citation("src/lumina_plasma.c", 19607, "tau_sobolev_mark_computed"),
    Citation("src/lumina_plasma.c", 20792, "candidate->opacity.tau_required_generation == 0"),
    Citation("src/lumina_plasma.c", 20876, "nlte_population_candidate_produce_publications"),
    Citation("src/lumina_plasma.c", 20884, "candidate->opacity.tau_required_generation"),
    Citation("src/lumina_plasma.c", 21175, "candidate->bf.activation_level"),
    Citation("src/lumina_plasma.c", 21178, "candidate->opacity.tau_required_generation)return 0"),
)


class ReaderError(RuntimeError):
    def __init__(self, reason: str, detail: str = "", metrics: "Metrics | None" = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.metrics = metrics


@dataclass
class Metrics:
    pairlist_sha256: str = "NA"
    membership_good: int | None = None
    membership_total: int | None = None
    dmax: int | None = None
    d2: dict[tuple[int, int], int] = field(default_factory=dict)
    d3_unique: dict[int, int] = field(default_factory=dict)
    d3_abs_min: str = "NA"
    d3_abs_median: str = "NA"
    d3_abs_max: str = "NA"
    d4_samples: int | None = None
    d4_matches: int | None = None
    d4_mismatches: int | None = None
    d4_label: str = "NA"
    pdeck: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Pair:
    line: int
    z: int
    ion: int
    lower_global: int
    upper_global: int


@dataclass(frozen=True)
class Level:
    z: int
    ion: int
    number: int
    energy_ev: float
    g: float


@dataclass
class ReaderConfig:
    expected_hash: str = EXPECTED_PAIRLIST_SHA256
    expected_total: int = EXPECTED_TOTAL
    expected_rows_per_z: dict[int, int] = field(
        default_factory=lambda: dict(EXPECTED_ROWS_PER_Z)
    )
    expected_d3_abs_format: tuple[str, str, str] = EXPECTED_D3_ABS_FORMAT


@dataclass
class ParsedLog:
    values: list[dict[int, float]]
    d2_values: dict[tuple[int, int], set[str]]
    group_counts: Counter[tuple[int, int]]
    selected: list[tuple[int, int, int, float]]
    anchor_rows: list[dict[str, str]]


def fail(metrics: Metrics, reason: str, detail: str = "") -> None:
    raise ReaderError(reason, detail, metrics)


def regular_file(path: Path, reason: str, metrics: Metrics) -> None:
    if not path.is_file() or path.is_symlink():
        fail(metrics, reason, str(path))


def sha256_file(path: Path, metrics: Metrics) -> str:
    regular_file(path, "PAIRLIST_MISSING", metrics)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def field_tokens(text: str, metrics: Metrics, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in result:
            fail(metrics, "ROW_FIELD_INVALID", f"{label}:duplicate={key}")
        result[key] = value
    return result


def parse_int(raw: str | None, name: str, metrics: Metrics, reason: str) -> int:
    if raw is None:
        fail(metrics, "ROW_FIELD_MISSING", name)
    try:
        value = int(raw)
    except ValueError:
        fail(metrics, reason, f"{name}={raw}")
    if str(value) != raw:
        fail(metrics, reason, f"{name}={raw}")
    return value


def parse_float(raw: str | None, name: str, metrics: Metrics, reason: str) -> float:
    if raw is None:
        fail(metrics, "ROW_FIELD_MISSING", name)
    try:
        value = float(raw)
    except ValueError:
        fail(metrics, reason, f"{name}={raw}")
    if not math.isfinite(value):
        fail(metrics, reason, f"{name}={raw}")
    return value


def csv_columns(reader: csv.DictReader, needed: Iterable[str], reason: str,
                metrics: Metrics) -> None:
    names = set(reader.fieldnames or ())
    missing = [name for name in needed if name not in names]
    if missing:
        fail(metrics, reason, ",".join(missing))


def load_pairs(path: Path, config: ReaderConfig, metrics: Metrics) -> dict[int, Pair]:
    metrics.pairlist_sha256 = sha256_file(path, metrics)
    if metrics.pairlist_sha256 != config.expected_hash:
        fail(metrics, "PAIRLIST_HASH_MISMATCH",
             f"expected={config.expected_hash} actual={metrics.pairlist_sha256}")
    pairs: dict[int, Pair] = {}
    count = 0
    with path.open("rt", encoding="utf-8", errors="strict", newline="") as stream:
        for number, raw in enumerate(stream, 1):
            text = raw.rstrip("\n")
            columns = text.split("\t")
            if len(columns) != 5:
                fail(metrics, "PAIRLIST_PARSE_ERROR", f"line={number} fields={len(columns)}")
            try:
                values = [int(value) for value in columns]
            except ValueError:
                fail(metrics, "PAIRLIST_PARSE_ERROR", f"line={number}")
            if any(str(value) != raw_value for value, raw_value in zip(values, columns)):
                fail(metrics, "PAIRLIST_PARSE_ERROR", f"line={number}")
            pair = Pair(*values)
            if pair.line in pairs:
                fail(metrics, "PAIRLIST_DUPLICATE_LINE", f"line={pair.line}")
            if pair.z not in TARGET_Z or pair.ion != 1:
                fail(metrics, "PAIRLIST_SCOPE_MISMATCH", f"line={pair.line}")
            pairs[pair.line] = pair
            count += 1
    metrics.membership_total = count
    if count != config.expected_total:
        fail(metrics, "PAIRLIST_COUNT_MISMATCH",
             f"expected={config.expected_total} actual={count}")
    return pairs


def load_levels(path: Path, metrics: Metrics) -> tuple[list[Level], dict[tuple[int, int, int], Level]]:
    regular_file(path, "LEVELS_MISSING", metrics)
    global_levels: list[Level] = []
    by_key: dict[tuple[int, int, int], Level] = {}
    with path.open("rt", encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        csv_columns(reader, ("atomic_number", "ion_number", "level_number", "energy_eV", "g"),
                    "LEVELS_COLUMN_MISSING", metrics)
        for number, row in enumerate(reader, 2):
            z = parse_int(row.get("atomic_number"), "atomic_number", metrics, "LEVELS_FIELD_INVALID")
            ion = parse_int(row.get("ion_number"), "ion_number", metrics, "LEVELS_FIELD_INVALID")
            level_number = parse_int(row.get("level_number"), "level_number", metrics,
                                     "LEVELS_FIELD_INVALID")
            energy = parse_float(row.get("energy_eV"), "energy_eV", metrics, "LEVELS_FIELD_INVALID")
            g = parse_float(row.get("g"), "g", metrics, "LEVELS_FIELD_INVALID")
            value = Level(z, ion, level_number, energy, g)
            key = (z, ion, level_number)
            if key in by_key:
                fail(metrics, "LEVELS_DUPLICATE_KEY", f"line={number} key={key}")
            global_levels.append(value)
            by_key[key] = value
    if not global_levels:
        fail(metrics, "LEVELS_EMPTY")
    return global_levels, by_key


def check_d1(pairs: dict[int, Pair], levels: list[Level], metrics: Metrics) -> None:
    membership_good = 0
    membership_bad = 0
    energy_bad = 0
    map_bad = 0
    for pair in pairs.values():
        if pair.lower_global < 0 or pair.upper_global < 0 or \
           pair.lower_global >= len(levels) or pair.upper_global >= len(levels):
            map_bad += 1
            continue
        lower = levels[pair.lower_global]
        upper = levels[pair.upper_global]
        if (lower.z, lower.ion) != (pair.z, pair.ion) or \
           (upper.z, upper.ion) != (pair.z, pair.ion):
            map_bad += 1
            continue
        if lower.number >= 100 and upper.number >= 100:
            membership_good += 1
        else:
            membership_bad += 1
        if not upper.energy_ev - lower.energy_ev > 0.0:
            energy_bad += 1
    metrics.membership_good = membership_good
    metrics.membership_total = len(pairs)
    if map_bad:
        fail(metrics, "GLOBAL_MAP_MISMATCH", f"count={map_bad}")
    if membership_bad:
        fail(metrics, "MEMBERSHIP_VIOLATION", f"count={membership_bad}")
    if energy_bad:
        fail(metrics, "ENERGY_ORDER_VIOLATION", f"count={energy_bad}")


def parse_zero_row(data: dict[str, str], iteration: int, pairs: dict[int, Pair],
                   parsed: ParsedLog, metrics: Metrics) -> None:
    phase = data.get("phase")
    if phase != "REQUESTED_TE":
        fail(metrics, "ROW_SCOPE_MISMATCH", f"phase={phase}")
    shell = parse_int(data.get("shell"), "shell", metrics, "ROW_FIELD_INVALID")
    line = parse_int(data.get("line"), "line", metrics, "ROW_FIELD_INVALID")
    z = parse_int(data.get("Z"), "Z", metrics, "ROW_FIELD_INVALID")
    ion = parse_int(data.get("ion"), "ion", metrics, "ROW_FIELD_INVALID")
    n_upper = parse_float(data.get("n_upper"), "n_upper", metrics, "ROW_FIELD_INVALID")
    if shell != 0 or z not in TARGET_Z or ion != 1:
        fail(metrics, "ROW_SCOPE_MISMATCH", f"shell={shell} Z={z} ion={ion}")
    pair = pairs.get(line)
    if pair is None or (pair.z, pair.ion) != (z, ion):
        fail(metrics, "ROW_HAZARD_MISMATCH", f"line={line}")
    if line in parsed.values[iteration]:
        fail(metrics, "DUPLICATE_HAZARD_ROW", f"iter={iteration} line={line}")
    parsed.values[iteration][line] = n_upper
    parsed.d2_values[(iteration, z)].add(n_upper.hex())
    parsed.group_counts[(iteration, z)] += 1
    if iteration == 0 and line == ANCHOR_LINE:
        parsed.anchor_rows.append(data)


def parse_selected_row(data: dict[str, str], iteration: int, parsed: ParsedLog,
                       metrics: Metrics) -> None:
    # Other ions can emit ordinary selected rows.  They are outside this reader's
    # registered scope and are intentionally ignored.
    try:
        shell_raw, z_raw, ion_raw = data.get("shell"), data.get("Z"), data.get("ion")
        if shell_raw is None or z_raw is None or ion_raw is None:
            return
        shell, z, ion = int(shell_raw), int(z_raw), int(ion_raw)
    except ValueError:
        fail(metrics, "D4_FIELD_INVALID", "identity")
    if shell != 0 or z not in TARGET_Z or ion != 1:
        return
    upper_level = parse_int(data.get("upper_level"), "upper_level", metrics, "D4_FIELD_INVALID")
    n_upper = parse_float(data.get("n_upper"), "n_upper", metrics, "D4_FIELD_INVALID")
    if upper_level >= 100:
        parsed.selected.append((iteration, z, upper_level, n_upper))


def parse_log(path: Path, pairs: dict[int, Pair], config: ReaderConfig,
              metrics: Metrics) -> ParsedLog:
    regular_file(path, "STDERR_MISSING", metrics)
    parsed = ParsedLog(
        values=[{}, {}], d2_values=defaultdict(set), group_counts=Counter(),
        selected=[], anchor_rows=[],
    )
    iteration = 0
    summaries = 0
    zero_rows_since_summary = 0
    with path.open("rt", encoding="utf-8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, 1):
            text = raw.rstrip("\n")
            if text.startswith(ZERO_ROW_PREFIX):
                if iteration >= 2:
                    fail(metrics, "INCOMPLETE_STREAM", f"row_after_iter2={line_number}")
                data = field_tokens(text[len(ZERO_ROW_PREFIX):], metrics, f"line={line_number}")
                parse_zero_row(data, iteration, pairs, parsed, metrics)
                zero_rows_since_summary += 1
            elif text.startswith(SELECTED_ROW_PREFIX):
                if iteration < 2:
                    data = field_tokens(text[len(SELECTED_ROW_PREFIX):], metrics,
                                        f"line={line_number}")
                    parse_selected_row(data, iteration, parsed, metrics)
            elif text.startswith(SUMMARY_PREFIX):
                data = field_tokens(text[len(SUMMARY_PREFIX):], metrics, f"line={line_number}")
                if data.get("phase") != "REQUESTED_TE" or data.get("target_ion") != "1":
                    continue
                shell = parse_int(data.get("shell"), "shell", metrics, "SUMMARY_FIELD_INVALID")
                if shell != 0:
                    fail(metrics, "SUMMARY_SCOPE_MISMATCH", f"shell={shell}")
                complete = parse_int(data.get("complete"), "complete", metrics,
                                     "SUMMARY_FIELD_INVALID")
                announced = parse_int(data.get("zero_opacity_emitting_rows"),
                                     "zero_opacity_emitting_rows", metrics,
                                     "SUMMARY_FIELD_INVALID")
                if complete != 1:
                    fail(metrics, "INCOMPLETE_STREAM", f"summary={summaries} complete={complete}")
                if announced != zero_rows_since_summary:
                    fail(metrics, "ROW_COUNT_MISMATCH",
                         f"iter={summaries} summary={announced} rows={zero_rows_since_summary}")
                summaries += 1
                iteration += 1
                zero_rows_since_summary = 0
    if summaries != 2 or iteration != 2 or zero_rows_since_summary != 0:
        fail(metrics, "INCOMPLETE_STREAM", f"summaries={summaries}")
    expected_lines = set(pairs)
    for iteration in (0, 1):
        observed = set(parsed.values[iteration])
        if len(observed) != config.expected_total:
            fail(metrics, "ROW_COUNT_MISMATCH",
                 f"iter={iteration} expected={config.expected_total} actual={len(observed)}")
        if observed != expected_lines:
            fail(metrics, "ROW_IDENTITY_MISMATCH", f"iter={iteration}")
        for z in TARGET_Z:
            actual = parsed.group_counts[(iteration, z)]
            expected = config.expected_rows_per_z[z]
            if actual != expected:
                fail(metrics, "ROW_COUNT_MISMATCH",
                     f"iter={iteration} Z={z} expected={expected} actual={actual}")
    return parsed


def check_anchor(anchor_rows: list[dict[str, str]], metrics: Metrics) -> None:
    if len(anchor_rows) != 1:
        fail(metrics, "ANCHOR_ROW_MISMATCH", f"line={ANCHOR_LINE} count={len(anchor_rows)}")
    anchor = anchor_rows[0]
    tau = parse_int(anchor.get("tau_validity"), "tau_validity", metrics, "ANCHOR_FIELD_MISMATCH")
    n_upper = parse_float(anchor.get("n_upper"), "n_upper", metrics, "ANCHOR_FIELD_MISMATCH")
    a_ul = parse_float(anchor.get("A_ul"), "A_ul", metrics, "ANCHOR_FIELD_MISMATCH")
    emission = parse_float(anchor.get("emission_per_sr"), "emission_per_sr", metrics,
                           "ANCHOR_FIELD_MISMATCH")
    actual = {
        "tau_validity": tau,
        "n_upper": n_upper.hex(),
        "A_ul": a_ul.hex(),
        "emission_per_sr": emission.hex(),
    }
    mismatches = [key for key, expected in ANCHOR_FIELDS.items() if actual[key] != expected]
    if mismatches:
        fail(metrics, "ANCHOR_FIELD_MISMATCH", ",".join(mismatches))


def check_d2_d3(parsed: ParsedLog, pairs: dict[int, Pair], config: ReaderConfig,
                metrics: Metrics) -> None:
    for iteration in (0, 1):
        for z in TARGET_Z:
            metrics.d2[(iteration, z)] = len(parsed.d2_values[(iteration, z)])
    metrics.dmax = max(metrics.d2.values())

    histograms: dict[int, Counter[str]] = {z: Counter() for z in TARGET_Z}
    values: dict[int, dict[str, float]] = {z: {} for z in TARGET_Z}
    for pair in pairs.values():
        n0, n1 = parsed.values[0][pair.line], parsed.values[1][pair.line]
        if n0 == 0.0:
            fail(metrics, "D3_RELATIVE_MOVEMENT_INVALID", f"line={pair.line} n0=0")
        relative = n1 / n0 - 1.0
        if not math.isfinite(relative):
            fail(metrics, "D3_RELATIVE_MOVEMENT_INVALID", f"line={pair.line}")
        key = relative.hex()
        histograms[pair.z][key] += 1
        values[pair.z][key] = relative
    for z in TARGET_Z:
        metrics.d3_unique[z] = len(histograms[z])
    if any(metrics.d3_unique[z] != 1 for z in TARGET_Z):
        detail = ",".join(f"Z={z}:{metrics.d3_unique[z]}" for z in TARGET_Z)
        fail(metrics, "D3_RELATIVE_MOVEMENT_MISMATCH", detail)

    weighted: list[float] = []
    for z in TARGET_Z:
        for key, count in histograms[z].items():
            weighted.extend([abs(values[z][key])] * count)
    weighted.sort()
    if not weighted:
        fail(metrics, "D3_RELATIVE_MOVEMENT_MISMATCH", "empty")
    metrics.d3_abs_min = format(weighted[0], ".4e")
    metrics.d3_abs_median = format(statistics.median(weighted), ".4e")
    metrics.d3_abs_max = format(weighted[-1], ".4e")
    actual = (metrics.d3_abs_min, metrics.d3_abs_median, metrics.d3_abs_max)
    if actual != config.expected_d3_abs_format:
        fail(metrics, "D3_RELATIVE_MOVEMENT_MISMATCH",
             f"expected={','.join(config.expected_d3_abs_format)} actual={','.join(actual)}")


def check_d4(parsed: ParsedLog, metrics: Metrics) -> None:
    matches = 0
    mismatches = 0
    for iteration, z, _upper_level, n_upper in parsed.selected:
        if n_upper.hex() in parsed.d2_values[(iteration, z)]:
            matches += 1
        else:
            mismatches += 1
    metrics.d4_samples = matches + mismatches
    metrics.d4_matches = matches
    metrics.d4_mismatches = mismatches
    if metrics.d4_samples == 0:
        metrics.d4_label = "D4_EMPTY"
    elif mismatches == 0:
        metrics.d4_label = "D4_ALL_MATCH"
    else:
        metrics.d4_label = "D4_OUTSIDE_D2"


def check_d5(path: Path, pairs: dict[int, Pair], by_key: dict[tuple[int, int, int], Level],
             metrics: Metrics) -> None:
    regular_file(path, "LINE_LIST_MISSING", metrics)
    predicted: dict[int, set[int]] = {z: set() for z in TARGET_Z}
    with path.open("rt", encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        csv_columns(reader, ("atomic_number", "ion_number", "level_number_lower",
                             "level_number_upper", "line_id"), "LINE_LIST_COLUMN_MISSING", metrics)
        for number, row in enumerate(reader, 2):
            z = parse_int(row.get("atomic_number"), "atomic_number", metrics,
                          "LINE_LIST_FIELD_INVALID")
            ion = parse_int(row.get("ion_number"), "ion_number", metrics, "LINE_LIST_FIELD_INVALID")
            if z not in TARGET_Z or ion != 1:
                continue
            lower_number = parse_int(row.get("level_number_lower"), "level_number_lower", metrics,
                                     "LINE_LIST_FIELD_INVALID")
            upper_number = parse_int(row.get("level_number_upper"), "level_number_upper", metrics,
                                     "LINE_LIST_FIELD_INVALID")
            line = parse_int(row.get("line_id"), "line_id", metrics, "LINE_LIST_FIELD_INVALID")
            lower = by_key.get((z, ion, lower_number))
            upper = by_key.get((z, ion, upper_number))
            if lower is None or upper is None:
                fail(metrics, "DECK_LINE_LEVEL_MISSING", f"line={number} id={line}")
            if lower.number >= 100 and upper.number >= 100 and lower.g == upper.g:
                predicted[z].add(line)
    metrics.pdeck = {z: len(predicted[z]) for z in TARGET_Z}
    missing = {z: sorted(pair.line for pair in pairs.values()
                         if pair.z == z and pair.line not in predicted[z]) for z in TARGET_Z}
    missing_count = sum(len(lines) for lines in missing.values())
    if missing_count:
        detail = ",".join(f"Z={z}:{len(missing[z])}" for z in TARGET_Z if missing[z])
        fail(metrics, "PREDICTION_SET_VIOLATION", detail)


def check_citations(root: Path, metrics: Metrics, citations: Iterable[Citation] = CITATIONS,
                    check_diff: bool = True) -> None:
    for citation in citations:
        path = root / citation.path
        regular_file(path, "CITATION_MISMATCH", metrics)
        with path.open("rt", encoding="utf-8", errors="strict") as stream:
            for _ in range(citation.line - 1):
                if stream.readline() == "":
                    fail(metrics, "CITATION_MISMATCH",
                         f"{citation.path}:{citation.line}:{citation.text}")
            line = stream.readline()
        if citation.text not in line:
            fail(metrics, "CITATION_MISMATCH",
                 f"{citation.path}:{citation.line}:{citation.text}")
    if not check_diff:
        return
    names = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "e500e29", "HEAD", "--", "src"],
        text=True, capture_output=True,
    )
    if names.returncode != 0:
        fail(metrics, "SOURCE_DIFF_UNAVAILABLE", names.stderr.strip())
    changed = [line for line in names.stdout.splitlines() if line]
    if changed != ["src/lumina_plasma.c"]:
        fail(metrics, "SOURCE_DIFF_SCOPE_MISMATCH", ",".join(changed))
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--unified=0", "e500e29", "HEAD", "--", "src"],
        text=True, capture_output=True,
    )
    if diff.returncode != 0:
        fail(metrics, "SOURCE_DIFF_UNAVAILABLE", diff.stderr.strip())
    hunks = [match for line in diff.stdout.splitlines() if (match := HUNK_RE.match(line))]
    if not hunks:
        fail(metrics, "SOURCE_DIFF_SCOPE_MISMATCH", "no-hunks")
    for match in hunks:
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        end = start + max(count, 1) - 1
        if start < 13_904 or end > 15_392:
            fail(metrics, "SOURCE_DIFF_SCOPE_MISMATCH", f"old={start}-{end}")


def branch_for(metrics: Metrics) -> str:
    assert metrics.dmax is not None
    if metrics.dmax <= 1.5:
        return "PB-A"
    if metrics.dmax <= 64.5:
        return "PB-B"
    return "PB-C"


def analyze(run_root: Path, pair_list: Path, root: Path, config: ReaderConfig,
            citations: Iterable[Citation] = CITATIONS, check_diff: bool = True) -> Metrics:
    metrics = Metrics()
    check_citations(root, metrics, citations, check_diff)
    pairs = load_pairs(pair_list, config, metrics)
    levels_path = run_root / "input" / "model" / "levels.csv"
    lines_path = run_root / "input" / "model" / "line_list.csv"
    stderr_path = run_root / "stderr.log"
    levels, by_key = load_levels(levels_path, metrics)
    check_d1(pairs, levels, metrics)
    parsed = parse_log(stderr_path, pairs, config, metrics)
    check_anchor(parsed.anchor_rows, metrics)
    check_d2_d3(parsed, pairs, config, metrics)
    check_d4(parsed, metrics)
    check_d5(lines_path, pairs, by_key, metrics)
    return metrics


def d2_text(metrics: Metrics) -> str:
    if not metrics.d2:
        return "NA"
    return ",".join(
        f"Z={z}/iter={iteration}:{metrics.d2[(iteration, z)]}"
        for iteration in (0, 1) for z in TARGET_Z if (iteration, z) in metrics.d2
    )


def d3_text(metrics: Metrics) -> str:
    if not metrics.d3_unique:
        return "NA"
    return ",".join(f"Z={z}:{metrics.d3_unique.get(z, 'NA')}" for z in TARGET_Z)


def pdeck_text(metrics: Metrics) -> str:
    if not metrics.pdeck:
        return "NA"
    return ",".join(f"Z={z}:{metrics.pdeck.get(z, 'NA')}" for z in TARGET_Z)


def format_result(status: str, branch: str, metrics: Metrics, reasons: Iterable[str]) -> str:
    membership = "NA" if metrics.membership_good is None or metrics.membership_total is None else \
        f"{metrics.membership_good}/{metrics.membership_total}"
    dmax = "NA" if metrics.dmax is None else str(metrics.dmax)
    reason_text = ",".join(reasons) or "NONE"
    return (
        f"{RESULT_SYMBOL} status={status} branch={branch} layer=NUMERIC_ONLY "
        f"dmax={dmax} membership={membership} d2={d2_text(metrics)} "
        f"d3_unique={d3_text(metrics)} d3_abs_min={metrics.d3_abs_min} "
        f"d3_abs_median={metrics.d3_abs_median} d3_abs_max={metrics.d3_abs_max} "
        f"d4_label={metrics.d4_label} d4_samples={metrics.d4_samples if metrics.d4_samples is not None else 'NA'} "
        f"d4_match={metrics.d4_matches if metrics.d4_matches is not None else 'NA'} "
        f"d4_mismatch={metrics.d4_mismatches if metrics.d4_mismatches is not None else 'NA'} "
        f"pdeck={pdeck_text(metrics)} pairlist_sha256={metrics.pairlist_sha256} "
        f"reasons={reason_text}"
    )


def emit_analysis(run_root: Path, pair_list: Path, root: Path, config: ReaderConfig,
                  citations: Iterable[Citation] = CITATIONS, check_diff: bool = True) -> tuple[int, str]:
    try:
        metrics = analyze(run_root, pair_list, root, config, citations, check_diff)
        return 0, format_result("PASS", branch_for(metrics), metrics, ())
    except ReaderError as exc:
        metrics = exc.metrics or Metrics()
        reason = exc.reason if not exc.detail else f"{exc.reason}:{exc.detail}"
        return 4, format_result("FAIL", "PB-X", metrics, (reason,))
    except (OSError, UnicodeError, csv.Error) as exc:
        return 4, format_result("FAIL", "PB-X", Metrics(), (f"IO_OR_PARSE_ERROR:{exc}",))


def fixture_paths(root: Path) -> tuple[Path, Path, Path]:
    run_root = root / "run"
    model = run_root / "input" / "model"
    model.mkdir(parents=True)
    pair_list = root / "pairs.tsv"
    source = root / "source"
    (source / "src").mkdir(parents=True)
    return run_root, pair_list, source


def fixture_row(line: int, z: int, n_upper: str, lower_global: int, upper_global: int) -> str:
    anchor = ""
    if line == ANCHOR_LINE:
        anchor = " tau_validity=2 A_ul=0.5222556 emission_per_sr=2.7310010625545125e-16"
    return (
        f"{ZERO_ROW_PREFIX} phase=REQUESTED_TE shell=0 line={line} Z={z} ion=1 "
        f"lower_global={lower_global} upper_global={upper_global} n_upper={n_upper}{anchor}"
    )


def fixture_selected(z: int, n_upper: str) -> str:
    return (
        f"{SELECTED_ROW_PREFIX} phase=REQUESTED_TE shell=0 line={900000 + z} "
        f"Z={z} ion=1 upper_level=101 n_upper={n_upper}"
    )


def fixture_summary(count: int) -> str:
    return (
        f"{SUMMARY_PREFIX} phase=REQUESTED_TE shell=0 target_ion=1 complete=1 "
        f"zero_opacity_emitting_rows={count}"
    )


def write_fixture(root: Path) -> tuple[Path, Path, Path, ReaderConfig, tuple[Citation, ...]]:
    run_root, pair_list, source = fixture_paths(root)
    records = [
        (ANCHOR_LINE, 26, 0, 1, "0.0004437688873013011", "0.00045264235457262295"),
        (669993, 26, 2, 3, "0.0004437688873013011", "0.00045264235457262295"),
        (669994, 27, 4, 5, "0.003129366625290828", "0.0031744804538243829"),
        (669995, 27, 6, 7, "0.003129366625290828", "0.0031744804538243829"),
        (669996, 28, 8, 9, "0.0011886643746207774", "0.0011747739839675887"),
    ]
    pair_list.write_text("".join(
        f"{line}\t{z}\t1\t{lower}\t{upper}\n"
        for line, z, lower, upper, _n0, _n1 in records
    ), encoding="utf-8")
    levels = ["atomic_number,ion_number,level_number,energy_eV,g\n"]
    line_rows = [
        "atomic_number,ion_number,level_number_lower,level_number_upper,line_id\n"
    ]
    for index, (line, z, lower, upper, _n0, _n1) in enumerate(records):
        lower_number = 100 + 2 * index
        upper_number = lower_number + 1
        levels.append(f"{z},1,{lower_number},{float(lower + 1):.1f},8\n")
        levels.append(f"{z},1,{upper_number},{float(upper + 2):.1f},8\n")
        line_rows.append(f"{z},1,{lower_number},{upper_number},{line}\n")
    model = run_root / "input" / "model"
    (model / "levels.csv").write_text("".join(levels), encoding="utf-8")
    (model / "line_list.csv").write_text("".join(line_rows), encoding="utf-8")
    log: list[str] = []
    for iteration in (0, 1):
        for line, z, lower, upper, n0, n1 in records:
            value = n0 if iteration == 0 else n1
            log.append(fixture_row(line, z, value, lower, upper))
        for z, value in ((26, records[0][4 + iteration]),
                         (27, records[2][4 + iteration]),
                         (28, records[4][4 + iteration])):
            log.append(fixture_selected(z, value))
        log.append(fixture_summary(len(records)))
    (run_root / "stderr.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    (source / "src" / "fixture.c").write_text("line1\nfixture-token\nline3\n", encoding="utf-8")
    config = ReaderConfig(
        expected_hash=hashlib.sha256(pair_list.read_bytes()).hexdigest(),
        expected_total=len(records), expected_rows_per_z={26: 2, 27: 2, 28: 1},
    )
    return run_root, pair_list, source, config, (Citation("src/fixture.c", 2, "fixture-token"),)


def mutate_k1(run_root: Path, _pairs: Path, _source: Path) -> None:
    path = run_root / "stderr.log"
    rows = path.read_text(encoding="utf-8").splitlines()
    for index, row in enumerate(rows):
        if row.startswith(ZERO_ROW_PREFIX):
            rows.pop(index)
            break
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def mutate_k2(run_root: Path, _pairs: Path, _source: Path) -> None:
    path = run_root / "stderr.log"
    old = "line=669993 Z=26 ion=1 lower_global=2 upper_global=3 n_upper=0.0004437688873013011"
    new = "line=669993 Z=26 ion=1 lower_global=2 upper_global=3 n_upper=0.0005"
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError("selftest K2 target absent")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def mutate_k3(run_root: Path, _pairs: Path, _source: Path) -> None:
    path = run_root / "input" / "model" / "levels.csv"
    text = path.read_text(encoding="utf-8")
    old = "26,1,100,1.0,8\n"
    if old not in text:
        raise RuntimeError("selftest K3 target absent")
    path.write_text(text.replace(old, "26,1,99,1.0,8\n", 1), encoding="utf-8")


def mutate_k4(_run_root: Path, pairs: Path, _source: Path) -> None:
    text = pairs.read_text(encoding="utf-8")
    pairs.write_text(text.replace("669996", "669997", 1), encoding="utf-8")


def mutate_k5(run_root: Path, _pairs: Path, _source: Path) -> None:
    path = run_root / "stderr.log"
    rows = path.read_text(encoding="utf-8").splitlines()
    for index in range(len(rows) - 1, -1, -1):
        if rows[index].startswith(SUMMARY_PREFIX):
            rows.pop(index)
            break
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def selftest_case(name: str, injection: str, expected_reason: str | None,
                  mutate=None, bad_citation: bool = False) -> bool:
    with tempfile.TemporaryDirectory(prefix="bitlockstep-reader-") as directory:
        root = Path(directory)
        run_root, pair_list, source, config, citations = write_fixture(root)
        active_citations = citations
        if mutate is not None:
            mutate(run_root, pair_list, source)
        if bad_citation:
            active_citations = (Citation("src/fixture.c", 2, "missing-fixture-token"),)
        rc, output = emit_analysis(run_root, pair_list, source, config,
                                   active_citations, check_diff=False)
        print(f"[BITLOCKSTEP-SELFTEST] name={name} inject={injection} {output}")
        if expected_reason is None:
            observed = rc == 0 and "status=PASS" in output and "branch=PB-A" in output
        elif name == "NC-K2":
            observed = rc == 4 and "dmax=2" in output and "branch=PB-X" in output
        else:
            observed = rc == 4 and f"reasons={expected_reason}" in output

        if mutate is not None or bad_citation:
            run_root, pair_list, source, config, citations = write_fixture(root / "restored")
            restore_rc, restore = emit_analysis(run_root, pair_list, source, config,
                                                citations, check_diff=False)
            print(f"[BITLOCKSTEP-SELFTEST] name={name} remove=RESTORE_FIXTURE {restore}")
            observed = observed and restore_rc == 0 and "status=PASS" in restore and \
                "branch=PB-A" in restore
        return observed


def selftest() -> int:
    cases = (
        ("NC-K0", "NONE", None, None, False),
        ("NC-K1", "DELETE_ONE_ZERO_ROW", "ROW_COUNT_MISMATCH", mutate_k1, False),
        ("NC-K2", "MUTATE_ONE_N_UPPER", "D3_RELATIVE_MOVEMENT_MISMATCH", mutate_k2, False),
        ("NC-K3", "LEVEL_NUMBER_BELOW_100", "MEMBERSHIP_VIOLATION", mutate_k3, False),
        ("NC-K4", "TAMPER_PAIRLIST", "PAIRLIST_HASH_MISMATCH", mutate_k4, False),
        ("NC-K5", "REMOVE_SUMMARY", "INCOMPLETE_STREAM", mutate_k5, False),
        ("NC-K6", "BAD_CITATION_AT_LINE", "CITATION_MISMATCH", None, True),
    )
    failed = 0
    for case in cases:
        if not selftest_case(*case):
            failed += 1
    print(f"{MODE}_SELFTEST status={'PASS' if failed == 0 else 'FAIL'} "
          f"controls={','.join(case[0] for case in cases)} failed={failed}")
    return 0 if failed == 0 else 1


def citation_only(root: Path) -> int:
    metrics = Metrics()
    try:
        check_citations(root, metrics)
    except ReaderError as exc:
        print(f"A210_BITLOCKSTEP_PRODUCER_CITATION_RESULT status=FAIL reason={exc.reason}"
              + (f" detail={exc.detail}" if exc.detail else ""))
        return 4
    print("A210_BITLOCKSTEP_PRODUCER_CITATION_RESULT status=PASS "
          f"citations={len(CITATIONS)} hunk_range=13904-15392")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--pair-list", type=Path, default=DEFAULT_PAIR_LIST)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="repository root for G-B3 source citations")
    parser.add_argument("--check-citations", action="store_true",
                        help="run G-B3 only; sealed inputs are not opened")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    if args.check_citations:
        return citation_only(args.root)
    rc, output = emit_analysis(args.run_root, args.pair_list, args.root, ReaderConfig())
    print(output)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
