#!/usr/bin/env python3
"""Read-only d-proxy runtime-super-level covariation readout.

The program streams stderr.log exactly once, reads only levels.csv from the
run inputs, and writes descriptive products under --out-dir.  Validation is
fail-closed: errors are retained in readout.json, all safe calculations keep
running, and the command exits 2 after writing the products when any
validation error was observed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator


SCHEMA = "dproxy-sl-covariation-readout-v2"
ROW_MARKER = "[A2-10][LINE-SATURATION-ROW]"
OWNER_MARKER = "[A2-10][LINE-ION-OWNER]"
BOUNDARY_MARKER = "[R7][PHASE]"
H_EV_S = 4.135667696e-15
NU_ENERGY_RELATIVE_TOLERANCE = 1.0e-6
SPREAD_REPORT_THRESHOLD = 1.0e-10

CORE_REQUIRED_ROW_FIELDS = (
    "phase",
    "shell",
    "line",
    "Z",
    "ion",
    "lower_global",
    "upper_global",
    "lower_level",
    "upper_level",
    "nu",
    "A_ul",
    "n_upper",
    "Jbar",
)

PROVENANCE_REQUIRED_ROW_FIELDS = (
    "n_upper_hex",
    "within_sl_frac",
    "within_sl_frac_hex",
    "runtime_sl_id",
    "super_anchor_global",
    "population_source",
    "used_nlte",
    "super_mode",
    "outer_iter",
    "te_generation",
    "nlte_population_generation",
    "population_provenance_status",
    "ledger_join_key",
)

REQUIRED_ROW_FIELDS = CORE_REQUIRED_ROW_FIELDS + PROVENANCE_REQUIRED_ROW_FIELDS

CORE_INTEGER_ROW_FIELDS = (
    "shell",
    "line",
    "Z",
    "ion",
    "lower_global",
    "upper_global",
    "lower_level",
    "upper_level",
)

FLOAT_ROW_FIELDS = (
    "nu",
    "A_ul",
    "n_upper",
    "Jbar",
)

PAIR_INVARIANT_FIELDS = (
    "runtime_sl_id",
    "super_anchor_global",
    "upper_global",
    "lower_global",
    "nu",
    "A_ul",
)

# These identity fields are needed to make the direct levels.csv join and the
# runtime-SL grouping unambiguous.  They supplement the ordered invariants.
PAIR_JOIN_IDENTITY_FIELDS = (
    "Z",
    "ion",
    "lower_level",
    "upper_level",
)

PAIR_CSV_FIELDS = (
    "eligible_for_sl",
    "sl_map_ok",
    "sl_map_status_0",
    "sl_map_status_1",
    "shell",
    "line",
    "Z",
    "ion",
    "runtime_sl_id_0",
    "runtime_sl_id_1",
    "super_anchor_global_0",
    "super_anchor_global_1",
    "lower_global",
    "upper_global",
    "lower_level",
    "upper_level",
    "nu_0",
    "nu_1",
    "A_ul_0",
    "A_ul_1",
    "n_upper_0",
    "n_upper_1",
    "Jbar_0",
    "Jbar_1",
    "within_sl_frac_0",
    "within_sl_frac_1",
    "B_0",
    "B_1",
    "dlogJ",
    "dlogn",
    "dlogB",
    "g_lower",
    "g_upper",
    "weight",
    "nu_energy_relative_error_0",
    "nu_energy_relative_error_1",
    "invariants_ok",
    "invariant_errors",
    "domain_ok",
    "deck_join_ok",
    "exclusion_reasons",
    "ledger_join_key_0",
    "ledger_join_key_1",
    "log_line_0",
    "log_line_1",
)

SL_MAP_CSV_FIELDS = (
    "upper_global",
    "runtime_sl_id",
    "super_anchor_global",
    "within_sl_frac_hex",
)

CONTRAST_CSV_FIELDS = (
    "line",
    "upper_global",
    "runtime_sl_id",
    "dlogJ_prom",
    "dlogJ_semi",
    "dlogn_prom",
    "dlogn_semi",
)

SL_CSV_FIELDS = (
    "shell",
    "Z",
    "ion",
    "runtime_sl_id",
    "member_pair_count",
    "distinct_upper_global_count",
    "distinct_line_count",
    "within_sl_frac_eq_1_member_count",
    "dlogB_spread_distinct_upper",
    "dlogn_spread",
    "within_sl_frac_log_change_max",
    "weight_sum",
    "X_s",
    "Y_s",
)

EXCLUSION_CSV_FIELDS = (
    "scope",
    "publication",
    "shell",
    "line",
    "Z",
    "ion",
    "runtime_sl_id",
    "reason",
    "detail",
)


class ReadoutError(RuntimeError):
    """Fatal setup/output error for which products cannot safely be written."""


class RowParseError(ValueError):
    def __init__(self, code: str, detail: Any) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass
class ErrorCollector:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, message: str, **context: Any) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if context:
            item["context"] = context
        self.entries.append(item)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_detail(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def float_text(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot be serialized")
        return format(value, ".17g")
    if value is None:
        return ""
    return value


def parse_key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key:
            continue
        if key in result:
            raise RowParseError("DUPLICATE_KEY_VALUE_FIELD", {"field": key})
        result[key] = value
    return result


def parse_int(raw: str, field_name: str) -> int:
    try:
        return int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise RowParseError(
            "NUMERIC_INTEGER_INVALID", {"field": field_name, "value": raw}
        ) from exc


def ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(first: float, second: float) -> int:
    if first == second:
        return 0
    return abs(ordered_float_bits(first) - ordered_float_bits(second))


def parse_finite_decimal(raw: str, field_name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RowParseError(
            "NUMERIC_FLOAT_INVALID", {"field": field_name, "value": raw}
        ) from exc
    if not math.isfinite(value):
        raise RowParseError(
            "NUMERIC_NONFINITE", {"field": field_name, "value": raw}
        )
    return value


def parse_finite_hex(raw: str, field_name: str) -> float:
    try:
        value = float.fromhex(raw)
    except (TypeError, ValueError) as exc:
        raise RowParseError(
            "NUMERIC_HEX_INVALID", {"field": field_name, "value": raw}
        ) from exc
    if not math.isfinite(value):
        raise RowParseError(
            "NUMERIC_NONFINITE", {"field": field_name, "value": raw}
        )
    return value


def parse_float_prefer_hex(fields: dict[str, str], field_name: str) -> float:
    decimal = parse_finite_decimal(fields[field_name], field_name)
    hex_name = f"{field_name}_hex"
    if hex_name not in fields:
        return decimal
    hexadecimal = parse_finite_hex(fields[hex_name], hex_name)
    distance = ulp_distance(decimal, hexadecimal)
    if distance > 1:
        raise RowParseError(
            "HEX_DECIMAL_ULP_MISMATCH",
            {
                "field": field_name,
                "decimal": fields[field_name],
                "hex": fields[hex_name],
                "ulp_distance": distance,
            },
        )
    return hexadecimal


def parse_row(
    fields: dict[str, str],
    publication: int,
    log_line: int,
    *,
    semi_state: bool,
    sl_map: dict[int, dict[str, Any]] | None,
    errors: ErrorCollector,
    mapping_stats: Counter[str],
) -> dict[str, Any]:
    required = CORE_REQUIRED_ROW_FIELDS if semi_state else REQUIRED_ROW_FIELDS
    missing = [name for name in required if name not in fields]
    if missing:
        raise RowParseError("ROW_REQUIRED_FIELDS_MISSING", {"fields": missing})
    empty = [name for name in required if fields[name] == ""]
    if empty:
        raise RowParseError("ROW_REQUIRED_FIELDS_EMPTY", {"fields": empty})

    row: dict[str, Any] = {
        "publication": publication,
        "log_line": log_line,
        "phase": fields["phase"],
        "population_source": fields.get("population_source"),
        "population_provenance_status": fields.get(
            "population_provenance_status"
        ),
        "ledger_join_key": fields.get("ledger_join_key", ""),
    }
    for name in CORE_INTEGER_ROW_FIELDS:
        row[name] = parse_int(fields[name], name)
    for name in FLOAT_ROW_FIELDS:
        row[name] = parse_float_prefer_hex(fields, name)
    mapping_stats[
        "n_upper_hex_present" if "n_upper_hex" in fields else "n_upper_hex_absent"
    ] += 1

    optional_integer_fields = (
        "used_nlte",
        "super_mode",
        "outer_iter",
        "te_generation",
        "nlte_population_generation",
    )
    for name in optional_integer_fields:
        raw = fields.get(name)
        row[name] = parse_int(raw, name) if raw not in (None, "") else None

    if fields.get("within_sl_frac") not in (None, ""):
        mapping_stats["row_within_sl_frac_present"] += 1
    if semi_state:
        supplied_fraction: float | None = None
        if fields.get("within_sl_frac") not in (None, ""):
            supplied_fraction = parse_float_prefer_hex(fields, "within_sl_frac")
            if supplied_fraction != 1.0:
                errors.add(
                    "SEMI_STATE_WITHIN_SL_FRAC_NOT_ONE",
                    "--semi-state requires an exact within_sl_frac of 1 when supplied",
                    publication=publication,
                    log_line=log_line,
                    upper_global=row["upper_global"],
                    observed=supplied_fraction,
                )
        supplied_super_mode = row["super_mode"]
        if supplied_super_mode not in (None, 0):
            errors.add(
                "SEMI_STATE_SUPER_MODE_NOT_ZERO",
                "--semi-state requires super_mode=0 when the field is supplied",
                publication=publication,
                log_line=log_line,
                observed=supplied_super_mode,
            )
        row["within_sl_frac"] = 1.0
        row["within_sl_frac_source"] = "semi_state_exact_one"
        row["super_mode"] = 0
        row["outer_iter"] = (
            publication if row["outer_iter"] is None else row["outer_iter"]
        )
        mapping_stats["within_sl_frac_set_to_one"] += 1
    else:
        row["within_sl_frac"] = parse_float_prefer_hex(fields, "within_sl_frac")
        row["within_sl_frac_source"] = "row_provenance"

    runtime_present = fields.get("runtime_sl_id") not in (None, "")
    anchor_present = fields.get("super_anchor_global") not in (None, "")
    if runtime_present != anchor_present:
        raise RowParseError(
            "ROW_PROVENANCE_FIELDS_PARTIAL",
            {
                "runtime_sl_id_present": runtime_present,
                "super_anchor_global_present": anchor_present,
            },
        )
    if runtime_present:
        row["runtime_sl_id"] = parse_int(fields["runtime_sl_id"], "runtime_sl_id")
        row["super_anchor_global"] = parse_int(
            fields["super_anchor_global"], "super_anchor_global"
        )
        mapping_stats["row_provenance_present"] += 1
    else:
        row["runtime_sl_id"] = None
        row["super_anchor_global"] = None
        mapping_stats["row_provenance_absent"] += 1

    row["sl_map_ok"] = True
    row["sl_map_status"] = "not_requested"
    if sl_map is not None:
        mapped = sl_map.get(row["upper_global"])
        if mapped is None:
            row["sl_map_ok"] = False
            row["sl_map_status"] = "upper_global_missing"
            mapping_stats["upper_global_missing"] += 1
            if not semi_state:
                errors.add(
                    "SL_MAP_UPPER_GLOBAL_MISSING",
                    "provenance row upper_global is absent from --sl-map",
                    publication=publication,
                    log_line=log_line,
                    upper_global=row["upper_global"],
                )
        elif runtime_present:
            mismatches: list[str] = []
            for name in ("runtime_sl_id", "super_anchor_global"):
                if row[name] != mapped[name]:
                    mismatches.append(
                        f"{name}:row={row[name]!r},map={mapped[name]!r}"
                    )
            if not semi_state and ulp_distance(
                row["within_sl_frac"], mapped["within_sl_frac"]
            ) > 1:
                mismatches.append(
                    "within_sl_frac:"
                    f"row={row['within_sl_frac'].hex()},"
                    f"map={mapped['within_sl_frac'].hex()}"
                )
            if mismatches:
                row["sl_map_ok"] = False
                row["sl_map_status"] = "provenance_mismatch"
                mapping_stats["provenance_mismatch"] += 1
                errors.add(
                    "SL_MAP_PROVENANCE_MISMATCH",
                    "--sl-map disagrees with ROW provenance",
                    publication=publication,
                    log_line=log_line,
                    upper_global=row["upper_global"],
                    mismatches=mismatches,
                )
            else:
                row["sl_map_status"] = "provenance_match"
                mapping_stats["provenance_match"] += 1
        else:
            row["runtime_sl_id"] = mapped["runtime_sl_id"]
            row["super_anchor_global"] = mapped["super_anchor_global"]
            row["sl_map_status"] = "transplanted"
            mapping_stats["transplanted"] += 1

    if not semi_state and not runtime_present:
        # Defensive guard: normal-mode required-field validation should already
        # make this unreachable.
        raise RowParseError(
            "ROW_REQUIRED_FIELDS_MISSING",
            {"fields": ["runtime_sl_id", "super_anchor_global"]},
        )

    mapping_stats["parsed_row_count"] += 1
    return row


def ensure_input_regular_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ReadoutError(f"{label} is not a regular file: {path}")
    if path.is_symlink():
        raise ReadoutError(f"{label} must not be a symlink: {path}")


def ensure_safe_output_dir(run_root: Path, out_dir: Path) -> Path:
    root_resolved = run_root.resolve(strict=True)
    out_resolved = out_dir.resolve(strict=False)
    if out_resolved == root_resolved or root_resolved in out_resolved.parents:
        raise ReadoutError(f"--out-dir is inside the read-only run root: {out_resolved}")

    # Reject existing symlinks along the destination path before creating it.
    absolute = Path(os.path.abspath(out_dir))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise ReadoutError(f"symlink in --out-dir path: {cursor}")
    absolute.mkdir(parents=True, exist_ok=True)
    if not absolute.is_dir() or absolute.is_symlink():
        raise ReadoutError(f"unsafe --out-dir: {absolute}")
    return absolute


def discover_levels_files(run_root: Path) -> tuple[list[Path], str]:
    primary_root = run_root / "input" / "global_atomic"
    primary = sorted(
        path for path in primary_root.glob("**/levels.csv") if path.is_file()
    )
    if primary:
        return primary, "input/global_atomic/**/levels.csv"

    # The preserved 331050 deck resolves the single table under input/model.
    # This fallback still selects only files named exactly levels.csv.
    fallback_root = run_root / "input"
    fallback = sorted(
        path for path in fallback_root.glob("**/levels.csv") if path.is_file()
    )
    if fallback:
        return fallback, "input/**/levels.csv (global_atomic pattern empty)"
    raise ReadoutError(f"no levels.csv below {fallback_root}")


def binary_text_lines(
    path: Path, digest: Any, byte_counter: list[int]
) -> Iterator[str]:
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            byte_counter[0] += len(raw)
            try:
                yield raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ReadoutError(f"non-UTF-8 input in {path}: {exc}") from exc


def read_levels(
    paths: list[Path], errors: ErrorCollector
) -> tuple[dict[tuple[int, int, int], dict[str, float]], list[dict[str, Any]]]:
    required = {
        "atomic_number",
        "ion_number",
        "level_number",
        "energy_eV",
        "g",
        "metastable",
        "super_level",
        "configuration",
    }
    levels: dict[tuple[int, int, int], dict[str, float]] = {}
    manifests: list[dict[str, Any]] = []

    for path in paths:
        ensure_input_regular_file(path, "levels.csv")
        digest = hashlib.sha256()
        byte_counter = [0]
        row_count = 0
        lines = binary_text_lines(path, digest, byte_counter)
        try:
            reader = csv.DictReader(lines)
            header = reader.fieldnames or []
            if len(header) != len(set(header)):
                errors.add(
                    "LEVELS_DUPLICATE_HEADER",
                    "levels.csv contains duplicate column names",
                    path=str(path),
                    header=header,
                )
            missing = sorted(required - set(header))
            if missing:
                errors.add(
                    "LEVELS_REQUIRED_COLUMNS_MISSING",
                    "levels.csv is missing required columns",
                    path=str(path),
                    fields=missing,
                )
            for item in reader:
                row_count += 1
                if missing:
                    continue
                try:
                    key = (
                        parse_int(item["atomic_number"], "atomic_number"),
                        parse_int(item["ion_number"], "ion_number"),
                        parse_int(item["level_number"], "level_number"),
                    )
                    energy = parse_finite_decimal(item["energy_eV"], "energy_eV")
                    statistical_weight = parse_finite_decimal(item["g"], "g")
                except (RowParseError, TypeError, KeyError) as exc:
                    if isinstance(exc, RowParseError):
                        detail: Any = exc.detail
                        code = exc.code
                    else:
                        detail = str(exc)
                        code = "LEVELS_ROW_INVALID"
                    errors.add(
                        code,
                        "invalid numeric field in levels.csv",
                        path=str(path),
                        csv_line=reader.line_num,
                        detail=detail,
                    )
                    continue
                record = {"energy_eV": energy, "g": statistical_weight}
                if key in levels:
                    errors.add(
                        "LEVELS_DUPLICATE_KEY",
                        "direct (Z, ion, level_number) key occurs more than once",
                        key=list(key),
                        path=str(path),
                        csv_line=reader.line_num,
                    )
                    continue
                levels[key] = record
        except csv.Error as exc:
            errors.add(
                "LEVELS_CSV_PARSE_ERROR",
                "CSV parser rejected levels.csv",
                path=str(path),
                detail=str(exc),
            )
            # Complete the input/hash stream even after a CSV syntax error.
            for _ in lines:
                pass
        manifests.append(
            {
                "path": str(path.resolve()),
                "sha256": digest.hexdigest(),
                "bytes": byte_counter[0],
                "data_rows": row_count,
            }
        )
    return levels, manifests


def read_sl_map(
    path: Path, errors: ErrorCollector
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Read either canonical sl_map.csv or a pairs.csv provenance source."""
    ensure_input_regular_file(path, "--sl-map")
    digest = hashlib.sha256()
    byte_counter = [0]
    lines = binary_text_lines(path, digest, byte_counter)
    mapping: dict[int, dict[str, Any]] = {}
    ambiguous: set[int] = set()
    source_rows = 0
    accepted_rows = 0
    source_format = "unknown"
    try:
        reader = csv.DictReader(lines)
        header = reader.fieldnames or []
        canonical_required = {
            "upper_global",
            "runtime_sl_id",
            "super_anchor_global",
            "within_sl_frac_hex",
        }
        pairs_required = {
            "upper_global",
            "runtime_sl_id_0",
            "super_anchor_global_0",
            "within_sl_frac_0",
        }
        if canonical_required <= set(header):
            source_format = "canonical_sl_map"
        elif pairs_required <= set(header):
            source_format = "pairs_csv"
        else:
            errors.add(
                "SL_MAP_REQUIRED_COLUMNS_MISSING",
                "--sl-map is neither canonical sl_map.csv nor supported pairs.csv",
                path=str(path),
                header=header,
            )

        for item in reader:
            source_rows += 1
            if source_format == "unknown":
                continue
            try:
                upper_global = parse_int(item["upper_global"], "upper_global")
                if source_format == "canonical_sl_map":
                    runtime_sl_id = parse_int(
                        item["runtime_sl_id"], "runtime_sl_id"
                    )
                    super_anchor_global = parse_int(
                        item["super_anchor_global"], "super_anchor_global"
                    )
                    fraction = parse_finite_hex(
                        item["within_sl_frac_hex"], "within_sl_frac_hex"
                    )
                else:
                    runtime_sl_id = parse_int(
                        item["runtime_sl_id_0"], "runtime_sl_id_0"
                    )
                    super_anchor_global = parse_int(
                        item["super_anchor_global_0"],
                        "super_anchor_global_0",
                    )
                    fraction = parse_finite_decimal(
                        item["within_sl_frac_0"], "within_sl_frac_0"
                    )
                    paired_values = (
                        ("runtime_sl_id_1", runtime_sl_id, parse_int),
                        (
                            "super_anchor_global_1",
                            super_anchor_global,
                            parse_int,
                        ),
                    )
                    for name, expected, parser in paired_values:
                        if item.get(name) not in (None, ""):
                            observed = parser(item[name], name)
                            if observed != expected:
                                raise RowParseError(
                                    "SL_MAP_PAIR_PROVENANCE_MISMATCH",
                                    {
                                        "field": name,
                                        "publication_0": expected,
                                        "publication_1": observed,
                                    },
                                )
                    if item.get("within_sl_frac_1") not in (None, ""):
                        fraction_1 = parse_finite_decimal(
                            item["within_sl_frac_1"], "within_sl_frac_1"
                        )
                        if ulp_distance(fraction, fraction_1) > 1:
                            raise RowParseError(
                                "SL_MAP_PAIR_PROVENANCE_MISMATCH",
                                {
                                    "field": "within_sl_frac_1",
                                    "publication_0": fraction.hex(),
                                    "publication_1": fraction_1.hex(),
                                },
                            )
                if not (0.0 < fraction <= 1.0):
                    raise RowParseError(
                        "SL_MAP_FRACTION_DOMAIN_INVALID",
                        {
                            "upper_global": upper_global,
                            "within_sl_frac": fraction,
                        },
                    )
            except (KeyError, RowParseError, TypeError) as exc:
                if isinstance(exc, RowParseError):
                    code = exc.code
                    detail: Any = exc.detail
                else:
                    code = "SL_MAP_ROW_INVALID"
                    detail = str(exc)
                errors.add(
                    code,
                    "invalid --sl-map row",
                    path=str(path),
                    csv_line=reader.line_num,
                    detail=detail,
                )
                continue

            candidate = {
                "runtime_sl_id": runtime_sl_id,
                "super_anchor_global": super_anchor_global,
                "within_sl_frac": fraction,
            }
            if upper_global in ambiguous:
                continue
            previous = mapping.get(upper_global)
            if previous is not None and (
                previous["runtime_sl_id"] != runtime_sl_id
                or previous["super_anchor_global"] != super_anchor_global
                or ulp_distance(previous["within_sl_frac"], fraction) > 1
            ):
                errors.add(
                    "SL_MAP_CONFLICTING_UPPER_GLOBAL",
                    "upper_global has more than one mapping",
                    path=str(path),
                    csv_line=reader.line_num,
                    upper_global=upper_global,
                    first={
                        **previous,
                        "within_sl_frac": previous["within_sl_frac"].hex(),
                    },
                    conflicting={
                        **candidate,
                        "within_sl_frac": fraction.hex(),
                    },
                )
                ambiguous.add(upper_global)
                del mapping[upper_global]
                continue
            mapping[upper_global] = candidate
            accepted_rows += 1
    except csv.Error as exc:
        errors.add(
            "SL_MAP_CSV_PARSE_ERROR",
            "CSV parser rejected --sl-map",
            path=str(path),
            detail=str(exc),
        )
        for _ in lines:
            pass

    manifest = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": byte_counter[0],
        "source_format": source_format,
        "data_rows": source_rows,
        "accepted_source_rows": accepted_rows,
        "unique_upper_global_count": len(mapping),
        "ambiguous_upper_global_count": len(ambiguous),
    }
    return mapping, manifest


def read_promoted_pairs(
    path: Path, errors: ErrorCollector
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    ensure_input_regular_file(path, "--promoted-pairs")
    digest = hashlib.sha256()
    byte_counter = [0]
    lines = binary_text_lines(path, digest, byte_counter)
    records: dict[tuple[int, int], dict[str, Any]] = {}
    data_rows = 0
    eligible_rows = 0
    try:
        reader = csv.DictReader(lines)
        header = reader.fieldnames or []
        required = {"shell", "line", "upper_global", "dlogJ", "dlogn"}
        runtime_field = (
            "runtime_sl_id"
            if "runtime_sl_id" in header
            else "runtime_sl_id_0"
        )
        missing = sorted((required | {runtime_field}) - set(header))
        if missing:
            errors.add(
                "PROMOTED_PAIRS_REQUIRED_COLUMNS_MISSING",
                "comparison pairs CSV is missing required columns",
                path=str(path),
                fields=missing,
            )
        for item in reader:
            data_rows += 1
            if missing:
                continue
            try:
                eligible = (
                    True
                    if item.get("eligible_for_sl") in (None, "")
                    else parse_int(item["eligible_for_sl"], "eligible_for_sl") == 1
                )
                if not eligible:
                    continue
                shell = parse_int(item["shell"], "shell")
                line = parse_int(item["line"], "line")
                record = {
                    "shell": shell,
                    "line": line,
                    "upper_global": parse_int(
                        item["upper_global"], "upper_global"
                    ),
                    "runtime_sl_id": parse_int(
                        item[runtime_field], runtime_field
                    ),
                    "dlogJ": parse_finite_decimal(item["dlogJ"], "dlogJ"),
                    "dlogn": parse_finite_decimal(item["dlogn"], "dlogn"),
                }
            except (KeyError, RowParseError, TypeError) as exc:
                if isinstance(exc, RowParseError):
                    code = exc.code
                    detail: Any = exc.detail
                else:
                    code = "PROMOTED_PAIRS_ROW_INVALID"
                    detail = str(exc)
                errors.add(
                    code,
                    "invalid comparison pairs row",
                    path=str(path),
                    csv_line=reader.line_num,
                    detail=detail,
                )
                continue
            key = (shell, line)
            if key in records:
                errors.add(
                    "PROMOTED_PAIRS_DUPLICATE_KEY",
                    "comparison pairs contain duplicate (shell,line)",
                    path=str(path),
                    shell=shell,
                    line=line,
                )
                continue
            records[key] = record
            eligible_rows += 1
    except csv.Error as exc:
        errors.add(
            "PROMOTED_PAIRS_CSV_PARSE_ERROR",
            "CSV parser rejected comparison pairs",
            path=str(path),
            detail=str(exc),
        )
        for _ in lines:
            pass
    manifest = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": byte_counter[0],
        "data_rows": data_rows,
        "eligible_pair_rows": eligible_rows,
    }
    return records, manifest


def verify_run_seal(run_root: Path, phase: str) -> dict[str, Any]:
    manifest = run_root / "RII_RUN_SEAL.sha"
    ensure_input_regular_file(manifest, "RII_RUN_SEAL.sha")
    entry_count = 0
    with manifest.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if len(line) < 67 or line[64:66] not in ("  ", " *"):
                raise ReadoutError(
                    f"invalid seal manifest record at line {line_number}: {manifest}"
                )
            try:
                int(line[:64], 16)
            except ValueError as exc:
                raise ReadoutError(
                    f"invalid seal digest at line {line_number}: {manifest}"
                ) from exc
            relative = Path(line[66:])
            if relative.is_absolute():
                raise ReadoutError(
                    f"absolute path in seal manifest at line {line_number}: {relative}"
                )
            resolved = (run_root / relative).resolve(strict=True)
            if resolved != run_root and run_root not in resolved.parents:
                raise ReadoutError(
                    f"seal target escapes run root at line {line_number}: {relative}"
                )
            entry_count += 1
    completed = subprocess.run(
        ["sha256sum", "-c", manifest.name],
        cwd=run_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_digest = hashlib.sha256(completed.stdout).hexdigest()
    stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise ReadoutError(
            f"R-II seal verification failed during {phase}: "
            f"exit={completed.returncode} stderr={stderr_text!r}"
        )
    return {
        "phase": phase,
        "command": ["sha256sum", "-c", manifest.name],
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": output_hash(manifest),
        "manifest_entry_count": entry_count,
        "verified_entry_count": entry_count,
        "exit_status": completed.returncode,
        "stdout_sha256": stdout_digest,
        "stderr": stderr_text,
    }


def parse_stderr(
    path: Path,
    errors: ErrorCollector,
    *,
    semi_state: bool = False,
    sl_map: dict[int, dict[str, Any]] | None = None,
    mapping_stats: Counter[str] | None = None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[str]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    ensure_input_regular_file(path, "stderr.log")
    if mapping_stats is None:
        mapping_stats = Counter()
    rows: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    temperatures: dict[int, list[str]] = {0: [], 1: []}
    boundaries: list[dict[str, Any]] = []
    encountered_rows: Counter[int] = Counter()
    encountered_unassigned_rows = 0
    digest = hashlib.sha256()
    byte_count = 0
    current_publication: int | None = None

    # This is the sole open/read traversal of stderr.log.
    with path.open("rb") as handle:
        for log_line, raw in enumerate(handle, 1):
            digest.update(raw)
            byte_count += len(raw)
            try:
                line = raw.decode("utf-8", errors="strict").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                errors.add(
                    "LOG_ENCODING_ERROR",
                    "stderr.log contains non-UTF-8 bytes",
                    log_line=log_line,
                    detail=str(exc),
                )
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

            if BOUNDARY_MARKER in line:
                try:
                    boundary_fields = parse_key_values(line)
                except RowParseError as exc:
                    errors.add(
                        exc.code,
                        "duplicate token on publication-boundary line",
                        log_line=log_line,
                        detail=exc.detail,
                    )
                    continue
                if (
                    boundary_fields.get("lane") == "DET"
                    and boundary_fields.get("phase") == "view"
                ):
                    raw_iter = boundary_fields.get("iter")
                    if raw_iter is None:
                        errors.add(
                            "SCHEMA_PUBLICATION_BOUNDARY_ITER_MISSING",
                            "DET view boundary has no iter token",
                            log_line=log_line,
                        )
                        current_publication = None
                    else:
                        try:
                            iteration = parse_int(raw_iter, "iter")
                        except RowParseError as exc:
                            errors.add(
                                exc.code,
                                "invalid publication iter",
                                log_line=log_line,
                                detail=exc.detail,
                            )
                            current_publication = None
                        else:
                            boundaries.append(
                                {"publication": iteration, "log_line": log_line}
                            )
                            current_publication = (
                                iteration if iteration in (0, 1) else None
                            )
                continue

            if ROW_MARKER in line:
                if current_publication not in (0, 1):
                    encountered_unassigned_rows += 1
                    errors.add(
                        "ROW_OUTSIDE_PUBLICATION",
                        "ROW occurred before a valid DET view publication boundary",
                        log_line=log_line,
                    )
                    continue
                publication = current_publication
                encountered_rows[publication] += 1
                try:
                    fields = parse_key_values(line)
                    row = parse_row(
                        fields,
                        publication,
                        log_line,
                        semi_state=semi_state,
                        sl_map=sl_map,
                        errors=errors,
                        mapping_stats=mapping_stats,
                    )
                except RowParseError as exc:
                    errors.add(
                        exc.code,
                        "invalid LINE-SATURATION-ROW",
                        publication=publication,
                        log_line=log_line,
                        detail=exc.detail,
                    )
                    continue
                if row["outer_iter"] != publication:
                    errors.add(
                        "ROW_OUTER_ITER_MISMATCH",
                        "ROW outer_iter does not equal its publication boundary",
                        publication=publication,
                        outer_iter=row["outer_iter"],
                        log_line=log_line,
                    )
                    continue
                rows[publication].append(row)
                continue

            if OWNER_MARKER in line:
                if current_publication not in (0, 1):
                    errors.add(
                        "OWNER_OUTSIDE_PUBLICATION",
                        "LINE-ION-OWNER occurred outside a publication",
                        log_line=log_line,
                    )
                    continue
                try:
                    owner_fields = parse_key_values(line)
                    shell = parse_int(owner_fields.get("shell", ""), "shell")
                except RowParseError as exc:
                    errors.add(
                        exc.code,
                        "invalid LINE-ION-OWNER shell",
                        publication=current_publication,
                        log_line=log_line,
                        detail=exc.detail,
                    )
                    continue
                if shell != 0:
                    continue
                raw_temperature = owner_fields.get("T_e_K")
                if raw_temperature is None or raw_temperature == "":
                    errors.add(
                        "OWNER_TEMPERATURE_MISSING",
                        "shell=0 LINE-ION-OWNER has no T_e_K",
                        publication=current_publication,
                        log_line=log_line,
                    )
                    continue
                try:
                    parse_finite_decimal(raw_temperature, "T_e_K")
                except RowParseError as exc:
                    errors.add(
                        exc.code,
                        "invalid shell=0 owner temperature",
                        publication=current_publication,
                        log_line=log_line,
                        detail=exc.detail,
                    )
                    continue
                temperatures[current_publication].append(raw_temperature)

    iterations = [item["publication"] for item in boundaries]
    if len(boundaries) != 2:
        errors.add(
            "SCHEMA_PUBLICATION_BOUNDARY_COUNT",
            "exactly two DET view publication boundaries are required",
            observed_count=len(boundaries),
            observed_iterations=iterations,
            required_iterations=[0, 1],
        )
    elif iterations != [0, 1]:
        errors.add(
            "SCHEMA_PUBLICATION_BOUNDARY_ITERATIONS",
            "the two publication boundaries must occur once each in order 0,1",
            observed_iterations=iterations,
            required_iterations=[0, 1],
        )
    for publication in (0, 1):
        if not temperatures[publication]:
            errors.add(
                "OWNER_TEMPERATURE_SET_EMPTY",
                "publication has no finite shell=0 LINE-ION-OWNER T_e_K",
                publication=publication,
            )

    stream = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "stream_passes": 1,
        "encountered_rows": {
            "0": encountered_rows[0],
            "1": encountered_rows[1],
            "outside_publication": encountered_unassigned_rows,
        },
    }
    return rows, temperatures, boundaries, stream


def counter_records(counter: Counter[Any], names: tuple[str, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in sorted(counter):
        parts = key if isinstance(key, tuple) else (key,)
        record = {name: part for name, part in zip(names, parts)}
        record["count"] = counter[key]
        records.append(record)
    return records


def publication_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def present_or_absent(value: Any) -> Any:
        return "<absent>" if value is None else value

    return {
        "row_count": len(rows),
        "shell_Z_ion_distribution": counter_records(
            Counter((row["shell"], row["Z"], row["ion"]) for row in rows),
            ("shell", "Z", "ion"),
        ),
        "population_source_counts": dict(
            sorted(
                Counter(
                    present_or_absent(row["population_source"]) for row in rows
                ).items()
            )
        ),
        "population_provenance_status_counts": dict(
            sorted(
                Counter(
                    present_or_absent(row["population_provenance_status"])
                    for row in rows
                ).items()
            )
        ),
        "super_mode_counts": {
            str(key): value
            for key, value in sorted(
                Counter(row["super_mode"] for row in rows).items()
            )
        },
        "generation_value_sets": {
            "te_generation": sorted(
                {
                    row["te_generation"]
                    for row in rows
                    if row["te_generation"] is not None
                }
            ),
            "nlte_population_generation": sorted(
                {
                    row["nlte_population_generation"]
                    for row in rows
                    if row["nlte_population_generation"] is not None
                }
            ),
        },
        "metadata_absent_row_counts": {
            name: sum(row[name] is None for row in rows)
            for name in (
                "population_source",
                "population_provenance_status",
                "used_nlte",
                "te_generation",
                "nlte_population_generation",
            )
        },
    }


def exclusion_record(
    scope: str,
    reason: str,
    *,
    publication: Any = "",
    shell: Any = "",
    line: Any = "",
    Z: Any = "",
    ion: Any = "",
    runtime_sl_id: Any = "",
    detail: Any = "",
) -> dict[str, Any]:
    return {
        "scope": scope,
        "publication": publication,
        "shell": shell,
        "line": line,
        "Z": Z,
        "ion": ion,
        "runtime_sl_id": runtime_sl_id,
        "reason": reason,
        "detail": json_detail(detail),
    }


def level_check(
    row: dict[str, Any], levels: dict[tuple[int, int, int], dict[str, float]]
) -> dict[str, Any]:
    lower_key = (row["Z"], row["ion"], row["lower_level"])
    upper_key = (row["Z"], row["ion"], row["upper_level"])
    missing = [list(key) for key in (lower_key, upper_key) if key not in levels]
    if missing:
        return {"ok": False, "reason": "LEVEL_JOIN_MISSING", "missing": missing}
    lower = levels[lower_key]
    upper = levels[upper_key]
    delta_energy = upper["energy_eV"] - lower["energy_eV"]
    if not math.isfinite(delta_energy) or delta_energy <= 0.0:
        return {
            "ok": False,
            "reason": "LEVEL_ENERGY_DIFFERENCE_INVALID",
            "lower_key": list(lower_key),
            "upper_key": list(upper_key),
            "delta_energy_eV": delta_energy,
        }
    expected_nu = delta_energy / H_EV_S
    if not math.isfinite(expected_nu) or expected_nu <= 0.0:
        return {
            "ok": False,
            "reason": "EXPECTED_NU_INVALID",
            "expected_nu": expected_nu,
        }
    relative_error = abs(row["nu"] - expected_nu) / abs(expected_nu)
    return {
        "ok": relative_error <= NU_ENERGY_RELATIVE_TOLERANCE,
        "reason": (
            "NONE"
            if relative_error <= NU_ENERGY_RELATIVE_TOLERANCE
            else "NU_LEVEL_ENERGY_MISMATCH"
        ),
        "expected_nu": expected_nu,
        "relative_error": relative_error,
        "g_lower": lower["g"],
        "g_upper": upper["g"],
        "lower_key": list(lower_key),
        "upper_key": list(upper_key),
    }


def build_pairs(
    rows: dict[int, list[dict[str, Any]]],
    levels: dict[tuple[int, int, int], dict[str, float]],
    errors: ErrorCollector,
    exclusions: list[dict[str, Any]],
    *,
    semi_state: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed: dict[int, dict[tuple[int, int], list[dict[str, Any]]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for publication in (0, 1):
        for row in rows[publication]:
            indexed[publication][(row["shell"], row["line"])].append(row)

    pair_records: list[dict[str, Any]] = []
    energy_checked_pairs = 0
    energy_within_tolerance_pairs = 0
    energy_max_error: float | None = None
    energy_max_location: dict[str, Any] | None = None

    for key in sorted(set(indexed[0]) | set(indexed[1])):
        shell, line = key
        left = indexed[0].get(key, [])
        right = indexed[1].get(key, [])
        if len(left) > 1 or len(right) > 1:
            errors.add(
                "DUPLICATE_PAIR_KEY",
                "(shell,line) must occur at most once in each publication",
                shell=shell,
                line=line,
                publication_0_count=len(left),
                publication_1_count=len(right),
            )
            continue
        if not left or not right:
            present_publication = 0 if left else 1
            row = (left or right)[0]
            missing_publication = 1 - present_publication
            exclusions.append(
                exclusion_record(
                    "pair_key",
                    f"missing_publication_{missing_publication}",
                    publication=present_publication,
                    shell=shell,
                    line=line,
                    Z=row["Z"],
                    ion=row["ion"],
                    runtime_sl_id=row["runtime_sl_id"],
                    detail={"present_log_line": row["log_line"]},
                )
            )
            continue

        row0, row1 = left[0], right[0]
        sl_map_ok = row0["sl_map_ok"] and row1["sl_map_ok"]
        pair_exclusion_reasons: list[str] = []
        if not sl_map_ok:
            statuses = [row0["sl_map_status"], row1["sl_map_status"]]
            if "upper_global_missing" in statuses:
                mapping_reason = "sl_map_upper_global_missing"
            else:
                mapping_reason = "sl_map_provenance_mismatch"
            pair_exclusion_reasons.append(mapping_reason)
            exclusions.append(
                exclusion_record(
                    "pair",
                    mapping_reason,
                    shell=shell,
                    line=line,
                    Z=row0["Z"],
                    ion=row0["ion"],
                    runtime_sl_id=(
                        row0["runtime_sl_id"]
                        if row0["runtime_sl_id"] is not None
                        else ""
                    ),
                    detail={
                        "upper_global": row0["upper_global"],
                        "publication_0_status": row0["sl_map_status"],
                        "publication_1_status": row1["sl_map_status"],
                    },
                )
            )
        invariant_problems: list[str] = []
        for name in PAIR_INVARIANT_FIELDS + PAIR_JOIN_IDENTITY_FIELDS:
            if row0[name] != row1[name]:
                invariant_problems.append(
                    f"{name}:{row0[name]!r}!={row1[name]!r}"
                )
        for publication, row in ((0, row0), (1, row1)):
            if row["population_source"] is not None and (
                row["population_source"] != "NLTE_COMMITTED"
            ):
                invariant_problems.append(
                    f"publication_{publication}.population_source="
                    f"{row['population_source']!r}"
                )
            if row["used_nlte"] is not None and row["used_nlte"] != 1:
                invariant_problems.append(
                    f"publication_{publication}.used_nlte={row['used_nlte']!r}"
                )
            if row["population_provenance_status"] is not None and (
                row["population_provenance_status"] != "OK"
            ):
                invariant_problems.append(
                    f"publication_{publication}.population_provenance_status="
                    f"{row['population_provenance_status']!r}"
                )
            if not semi_state:
                for name in (
                    "population_source",
                    "used_nlte",
                    "population_provenance_status",
                ):
                    if row[name] is None:
                        invariant_problems.append(
                            f"publication_{publication}.{name}=<absent>"
                        )
        invariants_ok = not invariant_problems
        if not invariants_ok:
            errors.add(
                "PAIR_INVARIANT_VIOLATION",
                "paired rows do not satisfy the required identity/provenance invariants",
                shell=shell,
                line=line,
                violations=invariant_problems,
            )
            pair_exclusion_reasons.append("pair_invariant_violation")
            exclusions.append(
                exclusion_record(
                    "pair",
                    "pair_invariant_violation",
                    shell=shell,
                    line=line,
                    Z=row0["Z"],
                    ion=row0["ion"],
                    runtime_sl_id=row0["runtime_sl_id"],
                    detail=invariant_problems,
                )
            )

        domain_problems: list[str] = []
        for publication, row in ((0, row0), (1, row1)):
            if row["Jbar"] <= 0.0:
                domain_problems.append(f"publication_{publication}.Jbar<=0")
            if row["n_upper"] <= 0.0:
                domain_problems.append(f"publication_{publication}.n_upper<=0")
            if not (0.0 < row["within_sl_frac"] <= 1.0):
                domain_problems.append(
                    f"publication_{publication}.within_sl_frac_not_in_(0,1]"
                )
        domain_ok = not domain_problems
        B0: float | None = None
        B1: float | None = None
        dlogJ: float | None = None
        dlogn: float | None = None
        dlogB: float | None = None
        if domain_ok:
            B0 = row0["n_upper"] / row0["within_sl_frac"]
            B1 = row1["n_upper"] / row1["within_sl_frac"]
            if not (math.isfinite(B0) and math.isfinite(B1) and B0 > 0 and B1 > 0):
                domain_ok = False
                domain_problems.append("derived_B_nonfinite_or_nonpositive")
            else:
                dlogJ = math.log(row1["Jbar"]) - math.log(row0["Jbar"])
                dlogn = math.log(row1["n_upper"]) - math.log(row0["n_upper"])
                dlogB = math.log(B1) - math.log(B0)
                if not all(math.isfinite(value) for value in (dlogJ, dlogn, dlogB)):
                    domain_ok = False
                    domain_problems.append("derived_log_nonfinite")
        if not domain_ok:
            pair_exclusion_reasons.append("invalid_pair_domain")
            exclusions.append(
                exclusion_record(
                    "pair",
                    "invalid_pair_domain",
                    shell=shell,
                    line=line,
                    Z=row0["Z"],
                    ion=row0["ion"],
                    runtime_sl_id=row0["runtime_sl_id"],
                    detail=domain_problems,
                )
            )
            B0 = B1 = dlogJ = dlogn = dlogB = None

        checks = [level_check(row0, levels), level_check(row1, levels)]
        check_errors = [check for check in checks if not check["ok"]]
        relative_errors = [
            check.get("relative_error")
            for check in checks
            if isinstance(check.get("relative_error"), float)
        ]
        if len(relative_errors) == 2:
            energy_checked_pairs += 1
            local_max = max(relative_errors)
            if energy_max_error is None or local_max > energy_max_error:
                energy_max_error = local_max
                energy_max_location = {"shell": shell, "line": line}
        deck_join_ok = not check_errors
        if deck_join_ok:
            energy_within_tolerance_pairs += 1
        else:
            reasons = sorted({check["reason"] for check in check_errors})
            for reason in reasons:
                matching = [check for check in check_errors if check["reason"] == reason]
                errors.add(
                    reason,
                    "direct levels.csv join or nu=(E_u-E_l)/h verification failed",
                    shell=shell,
                    line=line,
                    checks=matching,
                    relative_tolerance=NU_ENERGY_RELATIVE_TOLERANCE,
                )
                exclusion_reason = reason.lower()
                if exclusion_reason not in pair_exclusion_reasons:
                    pair_exclusion_reasons.append(exclusion_reason)
                    exclusions.append(
                        exclusion_record(
                            "pair",
                            exclusion_reason,
                            shell=shell,
                            line=line,
                            Z=row0["Z"],
                            ion=row0["ion"],
                            runtime_sl_id=row0["runtime_sl_id"],
                            detail=matching,
                        )
                    )

        g_lower: float | None = None
        g_upper: float | None = None
        weight: float | None = None
        if deck_join_ok and invariants_ok:
            g_lower = checks[0]["g_lower"]
            g_upper = checks[0]["g_upper"]
            weight_problem: str | None = None
            if not (g_lower > 0.0 and g_upper > 0.0):
                weight_problem = "level_statistical_weight_nonpositive"
            elif row0["A_ul"] <= 0.0 or row0["nu"] <= 0.0:
                weight_problem = "A_ul_or_nu_nonpositive"
            else:
                try:
                    frequency_cube = row0["nu"] ** 3
                    weight = (g_upper / g_lower) * row0["A_ul"] / frequency_cube
                except OverflowError:
                    weight_problem = "weight_frequency_cube_overflow"
                if weight_problem is None and (
                    weight is None or not math.isfinite(weight) or weight <= 0.0
                ):
                    weight_problem = "derived_weight_nonfinite_or_nonpositive"
            if weight_problem is not None:
                weight = None
                deck_join_ok = False
                errors.add(
                    "WEIGHT_DOMAIN_INVALID",
                    "w=(g_u/g_l)*A_ul/nu^3 is not finite and positive",
                    shell=shell,
                    line=line,
                    detail=weight_problem,
                )
                pair_exclusion_reasons.append("weight_domain_invalid")
                exclusions.append(
                    exclusion_record(
                        "pair",
                        "weight_domain_invalid",
                        shell=shell,
                        line=line,
                        Z=row0["Z"],
                        ion=row0["ion"],
                        runtime_sl_id=row0["runtime_sl_id"],
                        detail=weight_problem,
                    )
                )

        eligible = (
            sl_map_ok
            and invariants_ok
            and domain_ok
            and deck_join_ok
            and weight is not None
        )
        pair_records.append(
            {
                "eligible_for_sl": eligible,
                "sl_map_ok": sl_map_ok,
                "sl_map_status_0": row0["sl_map_status"],
                "sl_map_status_1": row1["sl_map_status"],
                "shell": shell,
                "line": line,
                "Z": row0["Z"],
                "ion": row0["ion"],
                "runtime_sl_id_0": row0["runtime_sl_id"],
                "runtime_sl_id_1": row1["runtime_sl_id"],
                "super_anchor_global_0": row0["super_anchor_global"],
                "super_anchor_global_1": row1["super_anchor_global"],
                "lower_global": row0["lower_global"],
                "upper_global": row0["upper_global"],
                "lower_level": row0["lower_level"],
                "upper_level": row0["upper_level"],
                "nu_0": row0["nu"],
                "nu_1": row1["nu"],
                "A_ul_0": row0["A_ul"],
                "A_ul_1": row1["A_ul"],
                "n_upper_0": row0["n_upper"],
                "n_upper_1": row1["n_upper"],
                "Jbar_0": row0["Jbar"],
                "Jbar_1": row1["Jbar"],
                "within_sl_frac_0": row0["within_sl_frac"],
                "within_sl_frac_1": row1["within_sl_frac"],
                "B_0": B0,
                "B_1": B1,
                "dlogJ": dlogJ,
                "dlogn": dlogn,
                "dlogB": dlogB,
                "g_lower": g_lower,
                "g_upper": g_upper,
                "weight": weight,
                "nu_energy_relative_error_0": checks[0].get("relative_error"),
                "nu_energy_relative_error_1": checks[1].get("relative_error"),
                "invariants_ok": invariants_ok,
                "invariant_errors": invariant_problems,
                "domain_ok": domain_ok,
                "deck_join_ok": deck_join_ok,
                "exclusion_reasons": pair_exclusion_reasons,
                "ledger_join_key_0": row0["ledger_join_key"],
                "ledger_join_key_1": row1["ledger_join_key"],
                "log_line_0": row0["log_line"],
                "log_line_1": row1["log_line"],
            }
        )

    energy_summary = {
        "relation": "nu == (E_upper_eV - E_lower_eV) / h_eV_s",
        "h_eV_s": H_EV_S,
        "relative_tolerance": NU_ENERGY_RELATIVE_TOLERANCE,
        "ion_join_convention": "ROW ion equals levels.csv ion_number directly (no offset)",
        "matched_pair_count": len(pair_records),
        "pairs_with_two_computable_relative_errors": energy_checked_pairs,
        "pairs_with_both_publications_within_tolerance": energy_within_tolerance_pairs,
        "maximum_relative_error": energy_max_error,
        "maximum_relative_error_pair": energy_max_location,
    }
    return pair_records, energy_summary


def derive_sl_map_rows(
    rows: dict[int, list[dict[str, Any]]], errors: ErrorCollector
) -> list[dict[str, Any]]:
    derived: dict[int, dict[str, Any]] = {}
    conflicting: set[int] = set()
    for publication in (0, 1):
        for row in rows[publication]:
            if row["runtime_sl_id"] is None or row["super_anchor_global"] is None:
                continue
            upper_global = row["upper_global"]
            candidate = {
                "upper_global": upper_global,
                "runtime_sl_id": row["runtime_sl_id"],
                "super_anchor_global": row["super_anchor_global"],
                "within_sl_frac": row["within_sl_frac"],
            }
            if upper_global in conflicting:
                continue
            previous = derived.get(upper_global)
            if previous is not None and (
                previous["runtime_sl_id"] != candidate["runtime_sl_id"]
                or previous["super_anchor_global"]
                != candidate["super_anchor_global"]
                or ulp_distance(
                    previous["within_sl_frac"], candidate["within_sl_frac"]
                )
                > 1
            ):
                errors.add(
                    "SL_MAP_DERIVATION_CONFLICT",
                    "ROW provenance does not define a function of upper_global",
                    publication=publication,
                    log_line=row["log_line"],
                    upper_global=upper_global,
                    first={
                        "runtime_sl_id": previous["runtime_sl_id"],
                        "super_anchor_global": previous["super_anchor_global"],
                        "within_sl_frac_hex": previous[
                            "within_sl_frac"
                        ].hex(),
                    },
                    conflicting={
                        "runtime_sl_id": candidate["runtime_sl_id"],
                        "super_anchor_global": candidate[
                            "super_anchor_global"
                        ],
                        "within_sl_frac_hex": candidate[
                            "within_sl_frac"
                        ].hex(),
                    },
                )
                conflicting.add(upper_global)
                del derived[upper_global]
                continue
            derived[upper_global] = candidate

    return [
        {
            "upper_global": upper_global,
            "runtime_sl_id": item["runtime_sl_id"],
            "super_anchor_global": item["super_anchor_global"],
            "within_sl_frac_hex": item["within_sl_frac"].hex(),
        }
        for upper_global, item in sorted(derived.items())
    ]


def energy_summary_for_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    within = 0
    maximum: float | None = None
    location: dict[str, Any] | None = None
    for pair in pairs:
        values = (
            pair["nu_energy_relative_error_0"],
            pair["nu_energy_relative_error_1"],
        )
        if not all(isinstance(value, float) for value in values):
            continue
        checked += 1
        local_max = max(values)
        if all(value <= NU_ENERGY_RELATIVE_TOLERANCE for value in values):
            within += 1
        if maximum is None or local_max > maximum:
            maximum = local_max
            location = {"shell": pair["shell"], "line": pair["line"]}
    return {
        "relation": "nu == (E_upper_eV - E_lower_eV) / h_eV_s",
        "h_eV_s": H_EV_S,
        "relative_tolerance": NU_ENERGY_RELATIVE_TOLERANCE,
        "ion_join_convention": (
            "ROW ion equals levels.csv ion_number directly (no offset)"
        ),
        "matched_pair_count": len(pairs),
        "pairs_with_two_computable_relative_errors": checked,
        "pairs_with_both_publications_within_tolerance": within,
        "maximum_relative_error": maximum,
        "maximum_relative_error_pair": location,
    }


def filter_exclusions_for_pair_keys(
    exclusions: list[dict[str, Any]], keys: set[tuple[int, int]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in exclusions:
        if item["scope"] == "block":
            continue
        if item["shell"] == "" or item["line"] == "":
            continue
        if (int(item["shell"]), int(item["line"])) in keys:
            selected.append(dict(item))
    return selected


def build_contrast_rows(
    semi_pairs: list[dict[str, Any]],
    promoted: dict[tuple[int, int], dict[str, Any]],
    errors: ErrorCollector,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for pair in sorted(semi_pairs, key=lambda item: (item["line"], item["shell"])):
        key = (pair["shell"], pair["line"])
        prom = promoted[key]
        violations: list[str] = []
        if pair["line"] in seen_lines:
            violations.append("line_not_unique_across_shells")
        seen_lines.add(pair["line"])
        if pair["upper_global"] != prom["upper_global"]:
            violations.append(
                "upper_global:"
                f"prom={prom['upper_global']},semi={pair['upper_global']}"
            )
        if pair["runtime_sl_id_0"] != prom["runtime_sl_id"]:
            violations.append(
                "runtime_sl_id:"
                f"prom={prom['runtime_sl_id']},semi={pair['runtime_sl_id_0']}"
            )
        if pair["dlogJ"] is None or pair["dlogn"] is None:
            violations.append("semi_pair_log_difference_unavailable")
        if violations:
            errors.add(
                "CONTRAST_PAIR_INVARIANT_VIOLATION",
                "promoted and semi-state pair records cannot be contrasted exactly",
                shell=pair["shell"],
                line=pair["line"],
                violations=violations,
            )
        records.append(
            {
                "line": pair["line"],
                "upper_global": pair["upper_global"],
                "runtime_sl_id": pair["runtime_sl_id_0"],
                "dlogJ_prom": prom["dlogJ"],
                "dlogJ_semi": pair["dlogJ"],
                "dlogn_prom": prom["dlogn"],
                "dlogn_semi": pair["dlogn"],
            }
        )
    return records


def midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def pearson(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    mean_first = math.fsum(first) / len(first)
    mean_second = math.fsum(second) / len(second)
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_first)
        * math.fsum(value * value for value in centered_second)
    )
    if denominator == 0.0:
        return None
    return math.fsum(
        left * right for left, right in zip(centered_first, centered_second)
    ) / denominator


def midrank_spearman(first: list[float], second: list[float]) -> float | None:
    if len(set(first)) < 2 or len(set(second)) < 2:
        return None
    return pearson(midranks(first), midranks(second))


def sl_key_record(key: tuple[int, int, int, int]) -> dict[str, int]:
    return {
        "shell": key[0],
        "Z": key[1],
        "ion": key[2],
        "runtime_sl_id": key[3],
    }


def build_sl_groups(
    pairs: list[dict[str, Any]], exclusions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    eligible_pairs = [pair for pair in pairs if pair["eligible_for_sl"]]
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in eligible_pairs:
        key = (
            pair["shell"],
            pair["Z"],
            pair["ion"],
            pair["runtime_sl_id_0"],
        )
        grouped[key].append(pair)

    sl_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda item: item["line"])
        distinct_upper = {member["upper_global"] for member in members}
        dlogB_values = [member["dlogB"] for member in members]
        dlogn_values = [member["dlogn"] for member in members]
        frac_changes = [
            abs(
                math.log(member["within_sl_frac_1"])
                - math.log(member["within_sl_frac_0"])
            )
            for member in members
        ]
        weights = [member["weight"] for member in members]
        weight_sum = math.fsum(weights)
        X_s = math.fsum(
            member["weight"] * member["dlogJ"] for member in members
        ) / weight_sum
        Y_s = median(dlogB_values)
        record: dict[str, Any] = {
            **sl_key_record(key),
            "member_pair_count": len(members),
            "distinct_upper_global_count": len(distinct_upper),
            "distinct_line_count": len({member["line"] for member in members}),
            "within_sl_frac_eq_1_member_count": sum(
                member["within_sl_frac_0"] == 1.0
                and member["within_sl_frac_1"] == 1.0
                for member in members
            ),
            "dlogB_spread_distinct_upper": (
                max(dlogB_values) - min(dlogB_values)
                if len(distinct_upper) >= 2
                else None
            ),
            "dlogn_spread": max(dlogn_values) - min(dlogn_values),
            "within_sl_frac_log_change_max": max(frac_changes),
            "weight_sum": weight_sum,
            "X_s": X_s,
            "Y_s": Y_s,
        }
        sl_rows.append(record)

    dlogB_candidates = [
        row for row in sl_rows if row["dlogB_spread_distinct_upper"] is not None
    ]
    dlogn_candidates = list(sl_rows)
    max_B_row = (
        max(dlogB_candidates, key=lambda item: item["dlogB_spread_distinct_upper"])
        if dlogB_candidates
        else None
    )
    max_n_row = (
        max(dlogn_candidates, key=lambda item: item["dlogn_spread"])
        if dlogn_candidates
        else None
    )
    group_summary = {
        "valid_runtime_sl_count": len(sl_rows),
        "member_pair_count_ge_2": sum(
            row["member_pair_count"] >= 2 for row in sl_rows
        ),
        "distinct_upper_global_count_ge_2": sum(
            row["distinct_upper_global_count"] >= 2 for row in sl_rows
        ),
        "dlogB_spread_gt_1e-10_count": sum(
            row["dlogB_spread_distinct_upper"] is not None
            and row["dlogB_spread_distinct_upper"] > SPREAD_REPORT_THRESHOLD
            for row in sl_rows
        ),
        "maximum_dlogB_spread_distinct_upper": (
            max_B_row["dlogB_spread_distinct_upper"] if max_B_row else None
        ),
        "maximum_dlogB_spread_runtime_sl": (
            sl_key_record(
                (
                    max_B_row["shell"],
                    max_B_row["Z"],
                    max_B_row["ion"],
                    max_B_row["runtime_sl_id"],
                )
            )
            if max_B_row
            else None
        ),
        "dlogn_spread_gt_1e-10_count": sum(
            row["dlogn_spread"] > SPREAD_REPORT_THRESHOLD for row in sl_rows
        ),
        "maximum_dlogn_spread": (
            max_n_row["dlogn_spread"] if max_n_row else None
        ),
        "maximum_dlogn_spread_runtime_sl": (
            sl_key_record(
                (
                    max_n_row["shell"],
                    max_n_row["Z"],
                    max_n_row["ion"],
                    max_n_row["runtime_sl_id"],
                )
            )
            if max_n_row
            else None
        ),
        "spread_threshold": SPREAD_REPORT_THRESHOLD,
        "within_sl_frac_eq_1_definition": (
            "fraction equals 1.0 in both publications"
        ),
    }

    block_members: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sl_rows:
        block_members[(row["shell"], row["Z"], row["ion"])].append(row)

    block_rows: list[dict[str, Any]] = []
    pooled_X: list[float] = []
    pooled_Y: list[float] = []
    for key in sorted(block_members):
        members = sorted(block_members[key], key=lambda item: item["runtime_sl_id"])
        X = [member["X_s"] for member in members]
        Y = [member["Y_s"] for member in members]
        distinct_X = len(set(X))
        distinct_Y = len(set(Y))
        included = distinct_X >= 2 and distinct_Y >= 2
        item = {
            "shell": key[0],
            "Z": key[1],
            "ion": key[2],
            "valid_sl_count": len(members),
            "distinct_X_s_count": distinct_X,
            "distinct_Y_s_count": distinct_Y,
            "tie_ratio": {
                "X_s": (len(members) - distinct_X) / len(members),
                "Y_s": (len(members) - distinct_Y) / len(members),
            },
            "included_in_rho_SL": included,
        }
        block_rows.append(item)
        if not included:
            exclusions.append(
                exclusion_record(
                    "block",
                    "block_distinct_values_lt_2",
                    shell=key[0],
                    Z=key[1],
                    ion=key[2],
                    detail={
                        "valid_sl_count": len(members),
                        "distinct_X_s_count": distinct_X,
                        "distinct_Y_s_count": distinct_Y,
                    },
                )
            )
            continue
        ranks_X = midranks(X)
        ranks_Y = midranks(Y)
        mean_X = math.fsum(ranks_X) / len(ranks_X)
        mean_Y = math.fsum(ranks_Y) / len(ranks_Y)
        pooled_X.extend(rank - mean_X for rank in ranks_X)
        pooled_Y.extend(rank - mean_Y for rank in ranks_Y)

    rho_SL = pearson(pooled_X, pooled_Y)
    block_summary = {
        "block_count": len(block_rows),
        "included_block_count": sum(
            row["included_in_rho_SL"] for row in block_rows
        ),
        "excluded_block_count": sum(
            not row["included_in_rho_SL"] for row in block_rows
        ),
        "pooled_centered_midrank_observation_count": len(pooled_X),
        "rho_SL": rho_SL,
        "definition": (
            "within-block midranks of X_s and Y_s, block means removed, "
            "then pooled Pearson correlation"
        ),
        "blocks": block_rows,
    }

    pair_X = [pair["dlogJ"] for pair in eligible_pairs]
    pair_Y = [pair["dlogn"] for pair in eligible_pairs]
    pair_summary = {
        "eligible_pair_count": len(eligible_pairs),
        "distinct_dlogJ_count": len(set(pair_X)),
        "distinct_dlogn_count": len(set(pair_Y)),
        "dlogJ_tie_ratio": (
            (len(pair_X) - len(set(pair_X))) / len(pair_X) if pair_X else None
        ),
        "dlogn_tie_ratio": (
            (len(pair_Y) - len(set(pair_Y))) / len(pair_Y) if pair_Y else None
        ),
        "midrank_spearman_dlogJ_dlogn": midrank_spearman(pair_X, pair_Y),
    }
    return sl_rows, group_summary, block_summary, pair_summary


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            record: dict[str, Any] = {}
            for key in writer.fieldnames:
                value = source.get(key, "")
                if isinstance(value, list):
                    value = "|".join(str(item) for item in value)
                elif isinstance(value, bool):
                    value = int(value)
                record[key] = float_text(value)
            writer.writerow(record)


def output_target(out_dir: Path, name: str) -> Path:
    path = out_dir / name
    if path.is_symlink():
        raise ReadoutError(f"output target must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ReadoutError(f"output target is not a regular file: {path}")
    if path.parent.resolve(strict=True) != out_dir.resolve(strict=True):
        raise ReadoutError(f"output target escaped --out-dir: {path}")
    return path


def exclusion_summary(exclusions: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason = Counter(item["reason"] for item in exclusions)
    by_scope = Counter(item["scope"] for item in exclusions)
    return {
        "record_count": len(exclusions),
        "by_reason": dict(sorted(by_reason.items())),
        "by_scope": dict(sorted(by_scope.items())),
    }


def output_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_optional(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def mapping_application_summary(
    stats: Counter[str],
    manifest: dict[str, Any] | None,
    *,
    semi_state: bool,
) -> dict[str, Any]:
    names = (
        "parsed_row_count",
        "row_provenance_present",
        "row_provenance_absent",
        "provenance_match",
        "provenance_mismatch",
        "transplanted",
        "upper_global_missing",
        "within_sl_frac_set_to_one",
        "row_within_sl_frac_present",
        "n_upper_hex_present",
        "n_upper_hex_absent",
    )
    counts = {name: stats[name] for name in names}
    self_validation = None
    if manifest is not None and not semi_state:
        self_validation = {
            "required": True,
            "checked_provenance_row_count": stats["row_provenance_present"],
            "matched_row_count": stats["provenance_match"],
            "missing_upper_global_count": stats["upper_global_missing"],
            "mismatch_count": stats["provenance_mismatch"],
            "all_rows_match": (
                stats["row_provenance_present"] == stats["parsed_row_count"]
                and stats["provenance_match"] == stats["parsed_row_count"]
                and stats["upper_global_missing"] == 0
                and stats["provenance_mismatch"] == 0
            ),
        }
    return {
        "enabled": manifest is not None,
        "input": manifest,
        "application_counts": counts,
        "within_sl_frac_transplanted": False,
        "self_validation": self_validation,
    }


def make_document(
    *,
    started: str,
    run_root: Path,
    rows: dict[int, list[dict[str, Any]]],
    temperatures: dict[int, list[str]],
    boundaries: list[dict[str, Any]],
    stderr_stream: dict[str, Any],
    levels_discovery: str,
    level_manifests: list[dict[str, Any]],
    level_count: int,
    pairs: list[dict[str, Any]],
    energy_summary: dict[str, Any],
    sl_rows: list[dict[str, Any]],
    group_summary: dict[str, Any],
    block_summary: dict[str, Any],
    pair_correlation: dict[str, Any],
    exclusions: list[dict[str, Any]],
    errors: ErrorCollector,
    artifacts: dict[str, Any],
    scope: dict[str, Any],
    mapping_summary: dict[str, Any],
    semi_state: bool,
    promoted_manifest: dict[str, Any] | None = None,
    seal_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    temperature_sets = {
        "0": sorted(set(temperatures[0])),
        "1": sorted(set(temperatures[1])),
    }
    eligible_pairs = sum(pair["eligible_for_sl"] for pair in pairs)
    method: dict[str, Any] = {
        "descriptive_only": True,
        "data_adjustments": [],
        "stderr_stream_passes": 1,
        "pair_provenance_policy": (
            "validate recorded fields; explicitly report unavailable legacy fields"
            if semi_state
            else "require and validate all provenance fields"
        ),
    }
    if semi_state:
        method["data_adjustments"] = [
            "within_sl_frac set to exactly 1.0 for every --semi-state ROW"
        ]
        method["semi_state_contract"] = {
            "enabled": True,
            "super_mode": 0,
            "within_sl_frac": 1.0,
            "within_sl_frac_source": "--semi-state exact-one contract",
            "within_sl_frac_taken_from_sl_map": False,
            "B_definition": "B = n_upper because within_sl_frac is exactly 1",
            "unrecorded_provenance_values_fabricated": False,
        }

    inputs: dict[str, Any] = {
        "stderr_log": stderr_stream,
        "levels_discovery": levels_discovery,
        "levels_csv": level_manifests,
        "levels_join_row_count": level_count,
        "line_list_csv_read": False,
        "sl_map": mapping_summary["input"],
    }
    if promoted_manifest is not None:
        inputs["promoted_pairs"] = promoted_manifest

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "reader_role": "descriptive runtime-SL covariation readout",
        "sealed": semi_state,
        "seal_note": (
            "RII_RUN_SEAL.sha verified before and after read-only analysis"
            if semi_state
            else "post-model seal is FAIL; run root is intentionally treated as unsealed"
        ),
        "run_root": str(run_root),
        "analysis_scope": scope,
        "execution": {
            "started_utc": started,
            "finished_utc": utc_now(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "argv": list(sys.argv),
        },
        "method": method,
        "inputs": inputs,
        "sl_map_application": mapping_summary,
        "publication_boundaries": boundaries,
        "publications": {
            "0": publication_summary(rows[0]),
            "1": publication_summary(rows[1]),
        },
        "D3_temperature": {
            "shell": 0,
            "temperature_string_sets": temperature_sets,
            "entry_counts": {
                "0": len(temperatures[0]),
                "1": len(temperatures[1]),
            },
            "strings_identical": temperature_sets["0"] == temperature_sets["1"],
        },
        "pairs": {
            "matched_pair_count": len(pairs),
            "eligible_pair_count": eligible_pairs,
            "ineligible_matched_pair_count": len(pairs) - eligible_pairs,
            "pair_level_midrank_spearman": pair_correlation,
        },
        "nu_energy_verification": energy_summary,
        "runtime_sl_summary": group_summary,
        "runtime_sl_groups": sl_rows,
        "sl_block_covariation": block_summary,
        "exclusions": exclusion_summary(exclusions),
        "validation": {
            "error_count": len(errors.entries),
            "errors": errors.entries,
        },
        "artifacts": artifacts,
    }
    if seal_verification is not None:
        document["seal_verification"] = seal_verification
    return document


def write_json(path: Path, document: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def print_run_summary(document: dict[str, Any], out_dir: Path) -> None:
    pairs = document["pairs"]
    groups = document["runtime_sl_summary"]
    blocks = document["sl_block_covariation"]
    temperature = document["D3_temperature"]
    print("files=readout.json,pairs.csv,sl_groups.csv,exclusions.csv,sl_map.csv")
    print(f"out_dir={out_dir}")
    print(
        f"pairs_matched={pairs['matched_pair_count']} "
        f"pairs_eligible={pairs['eligible_pair_count']}"
    )
    print(
        f"runtime_sl={groups['valid_runtime_sl_count']} "
        f"member_ge_2={groups['member_pair_count_ge_2']} "
        f"distinct_upper_ge_2={groups['distinct_upper_global_count_ge_2']}"
    )
    print(
        "dlogB_spread_max="
        f"{format_optional(groups['maximum_dlogB_spread_distinct_upper'])} "
        f"at={json_detail(groups['maximum_dlogB_spread_runtime_sl'])}"
    )
    print(
        "dlogn_spread_max="
        f"{format_optional(groups['maximum_dlogn_spread'])} "
        f"at={json_detail(groups['maximum_dlogn_spread_runtime_sl'])}"
    )
    print(f"D3_temperature_strings_identical={int(temperature['strings_identical'])}")
    print(f"rho_SL={format_optional(blocks['rho_SL'])}")
    print(
        f"blocks={blocks['block_count']} "
        f"blocks_included={blocks['included_block_count']}"
    )
    print(f"validation_errors={document['validation']['error_count']}")


def print_semi_state_summary(result: dict[str, Any], out_dir: Path) -> None:
    all_doc = result["all"]
    common_doc = result["common"]
    pre = all_doc["seal_verification"]["pre"]
    post = all_doc["seal_verification"]["post"]
    print(
        "files=readout_rii_all.json,readout_rii_common231.json,"
        "pairs_rii_all.csv,pairs_rii_common231.csv,"
        "sl_groups_rii_all.csv,sl_groups_rii_common231.csv,"
        "exclusions_rii_all.csv,exclusions_rii_common231.csv,"
        "contrast_common231.csv"
    )
    print(f"out_dir={out_dir}")
    print(
        f"seal_pre=PASS entries={pre['verified_entry_count']} "
        f"exit={pre['exit_status']}"
    )
    for label, document in (("rii_all", all_doc), ("rii_common231", common_doc)):
        pairs = document["pairs"]
        groups = document["runtime_sl_summary"]
        blocks = document["sl_block_covariation"]
        print(
            f"{label} pairs={pairs['matched_pair_count']} "
            f"eligible={pairs['eligible_pair_count']} "
            f"runtime_sl={groups['valid_runtime_sl_count']}"
        )
        print(
            f"{label} dlogB_spread_max="
            f"{format_optional(groups['maximum_dlogB_spread_distinct_upper'])} "
            f"dlogn_spread_max={format_optional(groups['maximum_dlogn_spread'])}"
        )
        print(
            f"{label} rho_SL={format_optional(blocks['rho_SL'])} "
            f"blocks={blocks['block_count']} "
            f"blocks_included={blocks['included_block_count']}"
        )
    print(f"contrast_rows={result['contrast_row_count']}")
    print(
        f"seal_post=PASS entries={post['verified_entry_count']} "
        f"exit={post['exit_status']}"
    )
    print(f"validation_errors={all_doc['validation']['error_count']}")


def analyze(
    run_root: Path,
    out_dir: Path,
    *,
    sl_map_path: Path | None = None,
    semi_state: bool = False,
    promoted_pairs_path: Path | None = None,
) -> dict[str, Any]:
    started = utc_now()
    run_root = run_root.resolve(strict=True)
    if not run_root.is_dir():
        raise ReadoutError(f"--run-root is not a directory: {run_root}")
    if semi_state and sl_map_path is None:
        raise ReadoutError("--semi-state requires --sl-map")
    if not semi_state and promoted_pairs_path is not None:
        raise ReadoutError("--promoted-pairs is valid only with --semi-state")

    seal_pre = verify_run_seal(run_root, "pre-analysis") if semi_state else None
    out_dir = ensure_safe_output_dir(run_root, out_dir)
    stderr_path = run_root / "stderr.log"
    ensure_input_regular_file(stderr_path, "stderr.log")

    errors = ErrorCollector()
    sl_map: dict[int, dict[str, Any]] | None = None
    sl_map_manifest: dict[str, Any] | None = None
    if sl_map_path is not None:
        sl_map, sl_map_manifest = read_sl_map(sl_map_path.resolve(strict=True), errors)
    levels_paths, levels_discovery = discover_levels_files(run_root)
    levels, level_manifests = read_levels(levels_paths, errors)
    mapping_stats: Counter[str] = Counter()
    rows, temperatures, boundaries, stderr_stream = parse_stderr(
        stderr_path,
        errors,
        semi_state=semi_state,
        sl_map=sl_map,
        mapping_stats=mapping_stats,
    )

    base_exclusions: list[dict[str, Any]] = []
    pairs, energy_summary = build_pairs(
        rows,
        levels,
        errors,
        base_exclusions,
        semi_state=semi_state,
    )
    mapping_summary = mapping_application_summary(
        mapping_stats, sl_map_manifest, semi_state=semi_state
    )

    if not semi_state:
        exclusions = list(base_exclusions)
        sl_rows, group_summary, block_summary, pair_correlation = build_sl_groups(
            pairs, exclusions
        )
        sl_map_rows = derive_sl_map_rows(rows, errors)

        pair_path = output_target(out_dir, "pairs.csv")
        sl_path = output_target(out_dir, "sl_groups.csv")
        exclusion_path = output_target(out_dir, "exclusions.csv")
        map_path = output_target(out_dir, "sl_map.csv")
        readout_path = output_target(out_dir, "readout.json")
        write_csv(pair_path, PAIR_CSV_FIELDS, pairs)
        write_csv(sl_path, SL_CSV_FIELDS, sl_rows)
        write_csv(exclusion_path, EXCLUSION_CSV_FIELDS, exclusions)
        write_csv(map_path, SL_MAP_CSV_FIELDS, sl_map_rows)
        artifacts = {
            "pairs.csv": {"sha256": output_hash(pair_path)},
            "sl_groups.csv": {"sha256": output_hash(sl_path)},
            "exclusions.csv": {"sha256": output_hash(exclusion_path)},
            "sl_map.csv": {
                "sha256": output_hash(map_path),
                "unique_upper_global_count": len(sl_map_rows),
            },
        }
        document = make_document(
            started=started,
            run_root=run_root,
            rows=rows,
            temperatures=temperatures,
            boundaries=boundaries,
            stderr_stream=stderr_stream,
            levels_discovery=levels_discovery,
            level_manifests=level_manifests,
            level_count=len(levels),
            pairs=pairs,
            energy_summary=energy_summary,
            sl_rows=sl_rows,
            group_summary=group_summary,
            block_summary=block_summary,
            pair_correlation=pair_correlation,
            exclusions=exclusions,
            errors=errors,
            artifacts=artifacts,
            scope={
                "name": "all_matched_pairs",
                "definition": "all (shell,line) keys present once in both publications",
            },
            mapping_summary=mapping_summary,
            semi_state=False,
        )
        write_json(readout_path, document)
        return document

    comparison_path = promoted_pairs_path
    if comparison_path is None and sl_map_manifest is not None:
        if sl_map_manifest["source_format"] == "pairs_csv":
            comparison_path = sl_map_path
    if comparison_path is None:
        raise ReadoutError(
            "--semi-state requires --promoted-pairs when --sl-map is canonical sl_map.csv"
        )
    promoted, promoted_manifest = read_promoted_pairs(
        comparison_path.resolve(strict=True), errors
    )
    all_pair_keys = {(pair["shell"], pair["line"]) for pair in pairs}
    common_keys = all_pair_keys & set(promoted)
    common_pairs = [
        pair for pair in pairs if (pair["shell"], pair["line"]) in common_keys
    ]
    contrast_rows = build_contrast_rows(common_pairs, promoted, errors)

    exclusions_all = list(base_exclusions)
    (
        sl_rows_all,
        group_summary_all,
        block_summary_all,
        pair_correlation_all,
    ) = build_sl_groups(pairs, exclusions_all)
    exclusions_common = filter_exclusions_for_pair_keys(base_exclusions, common_keys)
    (
        sl_rows_common,
        group_summary_common,
        block_summary_common,
        pair_correlation_common,
    ) = build_sl_groups(common_pairs, exclusions_common)
    common_rows = {
        publication: [
            row
            for row in rows[publication]
            if (row["shell"], row["line"]) in common_keys
        ]
        for publication in (0, 1)
    }

    # This is deliberately after all reads/calculations and before any output
    # is replaced, so a seal failure leaves no newly published products.
    seal_post = verify_run_seal(run_root, "post-analysis")
    seal_verification = {"pre": seal_pre, "post": seal_post}

    paths = {
        name: output_target(out_dir, name)
        for name in (
            "readout_rii_all.json",
            "readout_rii_common231.json",
            "pairs_rii_all.csv",
            "pairs_rii_common231.csv",
            "sl_groups_rii_all.csv",
            "sl_groups_rii_common231.csv",
            "exclusions_rii_all.csv",
            "exclusions_rii_common231.csv",
            "contrast_common231.csv",
        )
    }
    write_csv(paths["pairs_rii_all.csv"], PAIR_CSV_FIELDS, pairs)
    write_csv(paths["pairs_rii_common231.csv"], PAIR_CSV_FIELDS, common_pairs)
    write_csv(paths["sl_groups_rii_all.csv"], SL_CSV_FIELDS, sl_rows_all)
    write_csv(
        paths["sl_groups_rii_common231.csv"], SL_CSV_FIELDS, sl_rows_common
    )
    write_csv(
        paths["exclusions_rii_all.csv"], EXCLUSION_CSV_FIELDS, exclusions_all
    )
    write_csv(
        paths["exclusions_rii_common231.csv"],
        EXCLUSION_CSV_FIELDS,
        exclusions_common,
    )
    write_csv(paths["contrast_common231.csv"], CONTRAST_CSV_FIELDS, contrast_rows)

    artifacts_all = {
        name: {"sha256": output_hash(paths[name])}
        for name in (
            "pairs_rii_all.csv",
            "sl_groups_rii_all.csv",
            "exclusions_rii_all.csv",
        )
    }
    artifacts_common = {
        name: {"sha256": output_hash(paths[name])}
        for name in (
            "pairs_rii_common231.csv",
            "sl_groups_rii_common231.csv",
            "exclusions_rii_common231.csv",
            "contrast_common231.csv",
        )
    }
    all_document = make_document(
        started=started,
        run_root=run_root,
        rows=rows,
        temperatures=temperatures,
        boundaries=boundaries,
        stderr_stream=stderr_stream,
        levels_discovery=levels_discovery,
        level_manifests=level_manifests,
        level_count=len(levels),
        pairs=pairs,
        energy_summary=energy_summary_for_pairs(pairs),
        sl_rows=sl_rows_all,
        group_summary=group_summary_all,
        block_summary=block_summary_all,
        pair_correlation=pair_correlation_all,
        exclusions=exclusions_all,
        errors=errors,
        artifacts=artifacts_all,
        scope={
            "name": "rii_all",
            "definition": "all R-II keys present once in both publications",
            "matched_pair_key_count": len(all_pair_keys),
        },
        mapping_summary=mapping_summary,
        semi_state=True,
        promoted_manifest=promoted_manifest,
        seal_verification=seal_verification,
    )
    common_document = make_document(
        started=started,
        run_root=run_root,
        rows=common_rows,
        temperatures=temperatures,
        boundaries=boundaries,
        stderr_stream=stderr_stream,
        levels_discovery=levels_discovery,
        level_manifests=level_manifests,
        level_count=len(levels),
        pairs=common_pairs,
        energy_summary=energy_summary_for_pairs(common_pairs),
        sl_rows=sl_rows_common,
        group_summary=group_summary_common,
        block_summary=block_summary_common,
        pair_correlation=pair_correlation_common,
        exclusions=exclusions_common,
        errors=errors,
        artifacts=artifacts_common,
        scope={
            "name": "rii_common231",
            "definition": (
                "intersection of R-II matched (shell,line) keys and eligible "
                "331050 promoted pairs"
            ),
            "RII_matched_pair_key_count": len(all_pair_keys),
            "promoted_pair_key_count": len(promoted),
            "intersection_pair_key_count": len(common_keys),
        },
        mapping_summary=mapping_summary,
        semi_state=True,
        promoted_manifest=promoted_manifest,
        seal_verification=seal_verification,
    )
    common_document["cross_run_contrast"] = {
        "row_count": len(contrast_rows),
        "artifact": "contrast_common231.csv",
        "columns": list(CONTRAST_CSV_FIELDS),
    }
    write_json(paths["readout_rii_all.json"], all_document)
    write_json(paths["readout_rii_common231.json"], common_document)
    return {
        "mode": "semi_state",
        "all": all_document,
        "common": common_document,
        "contrast_row_count": len(contrast_rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only d-proxy runtime-SL covariation readout"
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--sl-map",
        type=Path,
        help="canonical sl_map.csv or a supported 331050 pairs.csv",
    )
    parser.add_argument(
        "--semi-state",
        action="store_true",
        help="use the R-II semi-state contract (super_mode=0, frac exactly 1)",
    )
    parser.add_argument(
        "--promoted-pairs",
        type=Path,
        help="331050 pairs.csv used for the common-pair subset and contrast",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = analyze(
            args.run_root,
            args.out_dir,
            sl_map_path=args.sl_map,
            semi_state=args.semi_state,
            promoted_pairs_path=args.promoted_pairs,
        )
    except (ReadoutError, OSError, ValueError) as exc:
        print(f"readout setup error: {exc}", file=sys.stderr)
        return 2
    if args.semi_state:
        print_semi_state_summary(result, Path(args.out_dir).resolve())
        return 2 if result["all"]["validation"]["error_count"] else 0
    print_run_summary(result, Path(args.out_dir).resolve())
    return 2 if result["validation"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
