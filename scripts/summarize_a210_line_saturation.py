#!/usr/bin/env python3
"""Seal the REQUESTED_TE shell-0 Co/Fe/Ni IV saturation diagnostic.

This is a read-only arithmetic/provenance validator.  Its tolerances cover
only serialization and floating-point evaluation identities; none is a
physical acceptance tolerance and no input value is repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW] "
UNION_META_PREFIX = "[A2-10][LINE-SATURATION-UNION-META] "
UNION_ION_SUMMARY_PREFIX = (
    "[A2-10][LINE-SATURATION-UNION-ION-SUMMARY] "
)
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY] "
BLOCKED_PREFIX = "[A2-10][LINE-SATURATION-BLOCKED] "
TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
FOUR_PI = 12.56637061435917295385057353311801153679

ROW_FLOAT_FIELDS = (
    "nu", "tau_raw", "tau_effective", "chi_raw", "chi_effective",
    "n_upper", "A_ul", "eta_per_sr", "Jbar", "Jbar_absolute_bound",
    "beta", "one_minus_beta_over_tau", "one_minus_beta",
    "jbar_over_source", "deck_scale", "absorption_per_sr", "net_per_sr",
    "signed_rate", "uncertainty", "cancellation_condition",
)
ROW_DECIMAL_FIELDS = (
    "scaled_emission", "scaled_absorption", "cumulative_scaled_emission",
    "cumulative_fraction", "selection_target_fraction",
)
ROW_INTEGER_FIELDS = (
    "shell", "rank", "line", "Z", "ion", "ion_label", "ion_slot",
    "lower_global", "upper_global", "lower_level", "upper_level",
    "tau_validity", "srce_chk", "source_function_defined",
    "scan_complete", "physical_values_modified", "clamp", "floor", "cap",
    "jitter", "repair",
)
ZERO_FIELDS = (
    "physical_values_modified", "clamp", "floor", "cap", "jitter", "repair",
)


class SaturationError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finite(fields: dict[str, str], name: str, line_number: int) -> float:
    try:
        value = float(fields[name])
    except (KeyError, ValueError) as exc:
        raise SaturationError(
            f"line {line_number}: missing or invalid float {name}"
        ) from exc
    if not math.isfinite(value):
        raise SaturationError(f"line {line_number}: nonfinite {name}")
    return value


def decimal_value(fields: dict[str, str], name: str, line_number: int) -> Decimal:
    try:
        value = Decimal(fields[name])
    except (KeyError, InvalidOperation) as exc:
        raise SaturationError(
            f"line {line_number}: missing or invalid decimal {name}"
        ) from exc
    if not value.is_finite():
        raise SaturationError(f"line {line_number}: nonfinite {name}")
    return value


def arithmetic_close(observed: float, expected: float, *operands: float) -> bool:
    """Binary64 serialization/evaluation bound, never a physics tolerance."""
    if not all(math.isfinite(value) for value in (observed, expected, *operands)):
        return False
    magnitude = abs(observed) + abs(expected) + math.fsum(
        abs(value) for value in operands
    )
    scale = max(
        (abs(observed), abs(expected), *(abs(value) for value in operands),
         sys.float_info.min)
    )
    bound = 128.0 * sys.float_info.epsilon * magnitude + 16.0 * math.ulp(scale)
    return abs(observed - expected) <= bound


def cmfgen_exponx(tau: float) -> tuple[float, float]:
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        beta = 1.0 - tau * companion
    elif tau < 40.0:
        beta = (1.0 - math.exp(-tau)) / tau
        companion = (1.0 - beta) / tau
    else:
        beta = 1.0 / tau
        companion = (1.0 - beta) / tau
    if not all(math.isfinite(value) and value > 0.0
               for value in (beta, companion)):
        raise SaturationError("invalid CMFGEN EXPONX reproduction")
    return beta, companion


def parse_row(text: str, line_number: int) -> dict[str, Any]:
    fields = dict(TOKEN.findall(text[len(ROW_PREFIX):]))
    required = {
        "phase", "interpretation", "source_function",
        *ROW_FLOAT_FIELDS, *ROW_DECIMAL_FIELDS, *ROW_INTEGER_FIELDS,
    }
    missing = required.difference(fields)
    if missing:
        raise SaturationError(
            f"line {line_number}: missing fields {sorted(missing)}"
        )
    row: dict[str, Any] = {
        "phase": fields["phase"],
        "interpretation": fields["interpretation"],
    }
    try:
        for name in ROW_INTEGER_FIELDS:
            row[name] = int(fields[name])
    except ValueError as exc:
        raise SaturationError(f"line {line_number}: invalid integer") from exc
    for name in ROW_FLOAT_FIELDS:
        row[name] = finite(fields, name, line_number)
    decimals = {
        name: decimal_value(fields, name, line_number)
        for name in ROW_DECIMAL_FIELDS
    }
    for name, value in decimals.items():
        row[name] = float(value)
        row[f"{name}_serialized"] = str(value)

    if row["phase"] != "REQUESTED_TE" or row["shell"] != 0:
        raise SaturationError(f"line {line_number}: wrong phase/shell")
    if row["interpretation"] != "DIAGNOSTIC_ONLY":
        raise SaturationError(f"line {line_number}: non-diagnostic record")
    if any(row[name] != 0 for name in ZERO_FIELDS):
        raise SaturationError(f"line {line_number}: forbidden mutation/repair")
    if row["scan_complete"] != 1:
        raise SaturationError(f"line {line_number}: incomplete scan marker")
    if row["Z"] not in (26, 27, 28) or row["ion"] < 0 or \
       row["ion_label"] != row["ion"] + 1:
        raise SaturationError(f"line {line_number}: wrong target ion")
    if row["line"] < 0 or row["rank"] <= 0 or row["ion_slot"] < 0:
        raise SaturationError(f"line {line_number}: invalid identity")
    if row["lower_global"] < 0 or row["upper_global"] < 0 or \
       row["lower_level"] < 0 or row["upper_level"] < 0:
        raise SaturationError(f"line {line_number}: invalid level identity")
    if row["tau_validity"] not in (1, 2) or row["srce_chk"] not in (0, 1):
        raise SaturationError(f"line {line_number}: invalid tau provenance")
    if row["source_function_defined"] not in (0, 1):
        raise SaturationError(f"line {line_number}: invalid source flag")
    if row["nu"] <= 0.0 or row["n_upper"] < 0.0 or row["A_ul"] < 0.0 or \
       row["eta_per_sr"] <= 0.0 or row["Jbar"] < 0.0 or \
       row["Jbar_absolute_bound"] < 0.0 or row["deck_scale"] <= 0.0 or \
       row["uncertainty"] < 0.0 or row["cancellation_condition"] < 0.0:
        raise SaturationError(f"line {line_number}: invalid component domain")
    if decimals["scaled_emission"] <= 0 or \
       decimals["cumulative_scaled_emission"] <= 0 or \
       decimals["cumulative_fraction"] <= 0:
        raise SaturationError(f"line {line_number}: invalid selected emission")
    if decimals["selection_target_fraction"] != Decimal("0.9"):
        raise SaturationError(f"line {line_number}: changed selection target")

    if row["srce_chk"] == 1:
        if not (row["tau_raw"] < -0.5 and row["tau_effective"] > 0.0):
            raise SaturationError(f"line {line_number}: SRCE_CHK provenance failed")
    elif row["tau_effective"] != row["tau_raw"]:
        raise SaturationError(f"line {line_number}: undeclared tau mutation")

    beta, companion = cmfgen_exponx(row["tau_effective"])
    if not arithmetic_close(row["beta"], beta, row["tau_effective"]):
        raise SaturationError(f"line {line_number}: beta identity failed")
    if not arithmetic_close(
        row["one_minus_beta_over_tau"], companion, row["tau_effective"]
    ):
        raise SaturationError(f"line {line_number}: companion identity failed")
    expected_one_minus = row["tau_effective"] * companion
    if not arithmetic_close(
        row["one_minus_beta"], expected_one_minus,
        row["tau_effective"], companion
    ):
        raise SaturationError(f"line {line_number}: one-minus-beta identity failed")

    expected_absorption = row["chi_effective"] * row["Jbar"]
    if not arithmetic_close(
        row["absorption_per_sr"], expected_absorption,
        row["chi_effective"], row["Jbar"]
    ):
        raise SaturationError(f"line {line_number}: absorption identity failed")
    expected_net = row["eta_per_sr"] - row["absorption_per_sr"]
    if not arithmetic_close(
        row["net_per_sr"], expected_net,
        row["eta_per_sr"], row["absorption_per_sr"]
    ):
        raise SaturationError(f"line {line_number}: net identity failed")
    expected_ratio = row["absorption_per_sr"] / row["eta_per_sr"]
    if not arithmetic_close(
        row["jbar_over_source"], expected_ratio,
        row["absorption_per_sr"], row["eta_per_sr"]
    ):
        raise SaturationError(f"line {line_number}: Jbar/source identity failed")

    if row["source_function_defined"]:
        source = finite(fields, "source_function", line_number)
        if row["chi_effective"] == 0.0:
            raise SaturationError(f"line {line_number}: false source definition")
        expected_source = row["eta_per_sr"] / row["chi_effective"]
        if not arithmetic_close(
            source, expected_source, row["eta_per_sr"], row["chi_effective"]
        ):
            raise SaturationError(f"line {line_number}: source identity failed")
        row["source_function"] = source
    else:
        if fields["source_function"] != "UNDEFINED_CHI_ZERO" or \
           row["chi_effective"] != 0.0:
            raise SaturationError(f"line {line_number}: undefined source mismatch")
        row["source_function"] = None

    factor = FOUR_PI * row["deck_scale"]
    expected_scaled_emission = row["eta_per_sr"] * factor
    expected_scaled_absorption = row["absorption_per_sr"] * factor
    expected_signed = row["net_per_sr"] * factor
    if not arithmetic_close(
        float(decimals["scaled_emission"]), expected_scaled_emission,
        row["eta_per_sr"], factor
    ):
        raise SaturationError(f"line {line_number}: scaled emission failed")
    if not arithmetic_close(
        float(decimals["scaled_absorption"]), expected_scaled_absorption,
        row["absorption_per_sr"], factor
    ):
        raise SaturationError(f"line {line_number}: scaled absorption failed")
    if not arithmetic_close(
        row["signed_rate"], expected_signed, row["net_per_sr"], factor
    ):
        raise SaturationError(f"line {line_number}: signed-rate identity failed")
    expected_condition = (
        (abs(row["eta_per_sr"]) + abs(row["absorption_per_sr"])) /
        abs(row["net_per_sr"])
    )
    if not arithmetic_close(
        row["cancellation_condition"], expected_condition,
        row["eta_per_sr"], row["absorption_per_sr"], row["net_per_sr"]
    ):
        raise SaturationError(f"line {line_number}: condition identity failed")
    return row


def parse_summary(text: str, line_number: int) -> dict[str, Any]:
    fields = dict(TOKEN.findall(text[len(SUMMARY_PREFIX):]))
    required = {
        "phase", "shell", "target_Z", "target_ion", "candidate_rows",
        "selected_rows", "total_scaled_emission", "selected_scaled_emission",
        "selected_fraction", "selection_target_fraction",
        "selected_reaches_target", "complete", "interpretation",
        *ZERO_FIELDS,
    }
    missing = required.difference(fields)
    if missing:
        raise SaturationError(
            f"line {line_number}: summary missing {sorted(missing)}"
        )
    try:
        integer = {
            name: int(fields[name])
            for name in (
                "shell", "target_ion", "candidate_rows", "selected_rows",
                "selected_reaches_target", "complete", *ZERO_FIELDS,
            )
        }
    except ValueError as exc:
        raise SaturationError(f"line {line_number}: invalid summary integer") from exc
    decimal = {
        name: decimal_value(fields, name, line_number)
        for name in (
            "total_scaled_emission", "selected_scaled_emission",
            "selected_fraction", "selection_target_fraction",
        )
    }
    if fields["phase"] != "REQUESTED_TE" or integer["shell"] != 0 or \
       fields["target_Z"] != "26,27,28" or not 0 <= integer["target_ion"] <= 10:
        raise SaturationError(f"line {line_number}: summary scope mismatch")
    if fields["interpretation"] != "DIAGNOSTIC_ONLY" or \
       any(integer[name] != 0 for name in ZERO_FIELDS):
        raise SaturationError(f"line {line_number}: summary mutation/repair")
    if integer["complete"] != 1 or integer["selected_reaches_target"] != 1:
        raise SaturationError(f"line {line_number}: incomplete summary")
    if integer["candidate_rows"] <= 0 or integer["selected_rows"] <= 0 or \
       integer["selected_rows"] > integer["candidate_rows"]:
        raise SaturationError(f"line {line_number}: invalid summary counts")
    if decimal["total_scaled_emission"] <= 0 or \
       decimal["selected_scaled_emission"] <= 0 or \
       decimal["selected_scaled_emission"] > decimal["total_scaled_emission"]:
        raise SaturationError(f"line {line_number}: invalid summary emission")
    selection_mode = fields.get("selection_mode", "COMBINED_PREFIX")
    if selection_mode not in ("COMBINED_PREFIX", "PER_ION_UNION"):
        raise SaturationError(f"line {line_number}: invalid selection mode")
    if decimal["selection_target_fraction"] != Decimal("0.9") or \
       (selection_mode == "COMBINED_PREFIX" and
        decimal["selected_fraction"] < decimal["selection_target_fraction"]):
        raise SaturationError(f"line {line_number}: selection below 90 percent")
    expected_fraction = (
        float(decimal["selected_scaled_emission"]) /
        float(decimal["total_scaled_emission"])
    )
    if not arithmetic_close(
        float(decimal["selected_fraction"]), expected_fraction,
        float(decimal["selected_scaled_emission"]),
        float(decimal["total_scaled_emission"]),
    ):
        raise SaturationError(f"line {line_number}: summary fraction failed")
    result = {
        "phase": fields["phase"],
        "shell": integer["shell"],
        "target_atomic_numbers": [26, 27, 28],
        "target_ion_zero_based": integer["target_ion"],
        "candidate_rows": integer["candidate_rows"],
        "selected_rows": integer["selected_rows"],
        **{name: float(value) for name, value in decimal.items()},
        **{f"{name}_serialized": str(value) for name, value in decimal.items()},
        "complete": True,
    }
    if "selection_mode" in fields:
        result["selection_mode"] = selection_mode
    return result


def parse_union_meta(text: str, line_number: int) -> dict[str, Any]:
    fields = dict(TOKEN.findall(text[len(UNION_META_PREFIX):]))
    integer_names = (
        "shell", "line", "Z", "ion", "global_rank", "ion_rank",
        "ion_candidate_rows", "scan_complete", *ZERO_FIELDS,
    )
    decimal_names = (
        "ion_total_scaled_emission", "ion_cumulative_scaled_emission",
        "ion_cumulative_fraction", "selection_target_fraction",
    )
    required = {
        "phase", "selection_mode", "interpretation",
        *integer_names, *decimal_names,
    }
    missing = required.difference(fields)
    if missing:
        raise SaturationError(
            f"line {line_number}: union metadata missing {sorted(missing)}"
        )
    try:
        integer = {name: int(fields[name]) for name in integer_names}
    except ValueError as exc:
        raise SaturationError(
            f"line {line_number}: invalid union metadata integer"
        ) from exc
    decimals = {
        name: decimal_value(fields, name, line_number)
        for name in decimal_names
    }
    if fields["phase"] != "REQUESTED_TE" or integer["shell"] != 0 or \
       fields["selection_mode"] != "PER_ION_UNION" or \
       fields["interpretation"] != "DIAGNOSTIC_ONLY":
        raise SaturationError(f"line {line_number}: union metadata scope")
    if integer["Z"] not in (26, 27, 28) or integer["ion"] < 0 or \
       integer["line"] < 0 or integer["global_rank"] <= 0 or \
       integer["ion_rank"] <= 0 or integer["ion_candidate_rows"] <= 0:
        raise SaturationError(f"line {line_number}: union metadata identity")
    if integer["scan_complete"] != 1 or \
       any(integer[name] != 0 for name in ZERO_FIELDS):
        raise SaturationError(f"line {line_number}: union metadata mutation")
    if decimals["ion_total_scaled_emission"] <= 0 or \
       decimals["ion_cumulative_scaled_emission"] <= 0 or \
       decimals["ion_cumulative_fraction"] <= 0 or \
       decimals["selection_target_fraction"] != Decimal("0.9"):
        raise SaturationError(f"line {line_number}: union metadata domain")
    return {
        **integer,
        **{name: float(value) for name, value in decimals.items()},
        **{f"{name}_serialized": str(value)
           for name, value in decimals.items()},
        "selection_mode": fields["selection_mode"],
    }


def parse_union_ion_summary(text: str, line_number: int) -> dict[str, Any]:
    fields = dict(TOKEN.findall(text[len(UNION_ION_SUMMARY_PREFIX):]))
    integer_names = (
        "shell", "Z", "ion", "candidate_rows", "selected_rows",
        "selected_reaches_target", "prefix_minimal", "complete",
        *ZERO_FIELDS,
    )
    decimal_names = (
        "total_scaled_emission", "selected_scaled_emission",
        "selected_fraction", "selection_target_fraction",
    )
    required = {
        "phase", "selection_mode", "interpretation",
        *integer_names, *decimal_names,
    }
    missing = required.difference(fields)
    if missing:
        raise SaturationError(
            f"line {line_number}: union ion summary missing {sorted(missing)}"
        )
    try:
        integer = {name: int(fields[name]) for name in integer_names}
    except ValueError as exc:
        raise SaturationError(
            f"line {line_number}: invalid union ion summary integer"
        ) from exc
    decimals = {
        name: decimal_value(fields, name, line_number)
        for name in decimal_names
    }
    if fields["phase"] != "REQUESTED_TE" or integer["shell"] != 0 or \
       fields["selection_mode"] != "PER_ION_UNION" or \
       fields["interpretation"] != "DIAGNOSTIC_ONLY":
        raise SaturationError(f"line {line_number}: union ion summary scope")
    if integer["Z"] not in (26, 27, 28) or integer["ion"] < 0 or \
       integer["candidate_rows"] <= 0 or integer["selected_rows"] <= 0 or \
       integer["selected_rows"] > integer["candidate_rows"]:
        raise SaturationError(f"line {line_number}: union ion summary counts")
    if integer["selected_reaches_target"] != 1 or \
       integer["prefix_minimal"] != 1 or integer["complete"] != 1 or \
       any(integer[name] != 0 for name in ZERO_FIELDS):
        raise SaturationError(f"line {line_number}: union ion summary incomplete")
    if decimals["total_scaled_emission"] <= 0 or \
       decimals["selected_scaled_emission"] <= 0 or \
       decimals["selected_scaled_emission"] > \
       decimals["total_scaled_emission"] or \
       decimals["selection_target_fraction"] != Decimal("0.9") or \
       decimals["selected_fraction"] < Decimal("0.9"):
        raise SaturationError(f"line {line_number}: union ion summary domain")
    return {
        **integer,
        **{name: float(value) for name, value in decimals.items()},
        **{f"{name}_serialized": str(value)
           for name, value in decimals.items()},
        "selection_mode": fields["selection_mode"],
    }


def validate_per_ion_union(
        rows: list[dict[str, Any]], metas: list[dict[str, Any]],
        ion_summaries: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if len(metas) != len(rows):
        raise SaturationError("union metadata count does not match selected rows")
    if len(ion_summaries) != 3 or \
       {item["Z"] for item in ion_summaries} != {26, 27, 28}:
        raise SaturationError("union ion summaries are incomplete")
    row_by_line = {row["line"]: row for row in rows}
    if len(row_by_line) != len(rows):
        raise SaturationError("duplicate union row line identity")
    meta_by_line = {meta["line"]: meta for meta in metas}
    if len(meta_by_line) != len(metas) or set(meta_by_line) != set(row_by_line):
        raise SaturationError("union row/metadata identity mismatch")

    previous_rank = 0
    previous_emission: Decimal | None = None
    previous_cumulative: float | None = None
    selected_values: list[float] = []
    total_emission = float(summary["total_scaled_emission"])
    by_ion: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        26: [], 27: [], 28: [],
    }
    for row in rows:
        rank = row["rank"]
        if rank <= previous_rank:
            raise SaturationError("union global ranks are not increasing")
        emission = Decimal(row["scaled_emission_serialized"])
        if previous_emission is not None and emission > previous_emission:
            raise SaturationError("union global emission order violated")
        cumulative = float(row["cumulative_scaled_emission"])
        fraction = float(row["cumulative_fraction"])
        if cumulative < float(emission) or \
           (previous_cumulative is not None and cumulative <= previous_cumulative):
            raise SaturationError("union candidate cumulative did not increase")
        if not arithmetic_close(
            fraction, cumulative / total_emission, cumulative, total_emission
        ):
            raise SaturationError("union candidate cumulative fraction failed")
        if previous_cumulative is not None and rank == previous_rank + 1 and \
           not arithmetic_close(
               cumulative, previous_cumulative + float(emission),
               previous_cumulative, float(emission)
           ):
            raise SaturationError("union consecutive candidate sum failed")
        meta = meta_by_line[row["line"]]
        if meta["global_rank"] != rank or meta["Z"] != row["Z"] or \
           meta["ion"] != row["ion"]:
            raise SaturationError("union row/metadata provenance mismatch")
        by_ion[row["Z"]].append((row, meta))
        selected_values.append(float(emission))
        previous_rank = rank
        previous_emission = emission
        previous_cumulative = cumulative

    selected_sum = math.fsum(selected_values)
    summary_selected = float(summary["selected_scaled_emission"])
    if not arithmetic_close(
        selected_sum, summary_selected, *selected_values, summary_selected
    ):
        raise SaturationError("union selected rows do not close to summary")

    ion_summary_by_z = {item["Z"]: item for item in ion_summaries}
    candidate_count = 0
    ion_totals: list[float] = []
    for z in (26, 27, 28):
        records = by_ion[z]
        ion_summary = ion_summary_by_z[z]
        if len(records) != ion_summary["selected_rows"]:
            raise SaturationError(f"Z={z}: selected row count mismatch")
        previous_ion_cumulative = 0.0
        previous_ion_fraction: Decimal | None = None
        for ion_rank, (row, meta) in enumerate(records, 1):
            if meta["ion_rank"] != ion_rank or \
               meta["ion_candidate_rows"] != ion_summary["candidate_rows"]:
                raise SaturationError(f"Z={z}: ion rank/candidate mismatch")
            if meta["ion_total_scaled_emission_serialized"] != \
               ion_summary["total_scaled_emission_serialized"]:
                raise SaturationError(f"Z={z}: ion total serialization mismatch")
            expected = previous_ion_cumulative + row["scaled_emission"]
            observed = meta["ion_cumulative_scaled_emission"]
            if not arithmetic_close(
                observed, expected, previous_ion_cumulative,
                row["scaled_emission"]
            ):
                raise SaturationError(f"Z={z}: ion cumulative identity failed")
            ion_fraction = Decimal(meta["ion_cumulative_fraction_serialized"])
            if not arithmetic_close(
                float(ion_fraction), observed / meta["ion_total_scaled_emission"],
                observed, meta["ion_total_scaled_emission"]
            ):
                raise SaturationError(f"Z={z}: ion fraction identity failed")
            if previous_ion_fraction is not None and \
               ion_fraction <= previous_ion_fraction:
                raise SaturationError(f"Z={z}: ion fraction did not increase")
            previous_ion_cumulative = observed
            previous_ion_fraction = ion_fraction
        if len(records) > 1 and Decimal(
            records[-2][1]["ion_cumulative_fraction_serialized"]
        ) >= Decimal("0.9"):
            raise SaturationError(f"Z={z}: prefix is not minimal")
        if Decimal(
            records[-1][1]["ion_cumulative_fraction_serialized"]
        ) < Decimal("0.9"):
            raise SaturationError(f"Z={z}: selected prefix is below target")
        if records[-1][1]["ion_cumulative_scaled_emission_serialized"] != \
           ion_summary["selected_scaled_emission_serialized"]:
            raise SaturationError(f"Z={z}: selected serialization mismatch")
        candidate_count += ion_summary["candidate_rows"]
        ion_totals.append(ion_summary["total_scaled_emission"])
    if candidate_count != summary["candidate_rows"]:
        raise SaturationError("union candidate counts do not close")
    if not arithmetic_close(
        math.fsum(ion_totals), total_emission, *ion_totals, total_emission
    ):
        raise SaturationError("union ion totals do not close to target total")


def summarize(log: Path) -> dict[str, Any]:
    if not log.is_file() or log.is_symlink():
        raise SaturationError(f"missing or unsafe log: {log}")
    rows: list[dict[str, Any]] = []
    union_metas: list[dict[str, Any]] = []
    union_ion_summaries: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    blocked: list[str] = []
    with log.open("rt", encoding="utf-8", errors="strict") as stream:
        for line_number, text in enumerate(stream, 1):
            if text.startswith(ROW_PREFIX):
                rows.append(parse_row(text, line_number))
            elif text.startswith(UNION_META_PREFIX):
                union_metas.append(parse_union_meta(text, line_number))
            elif text.startswith(UNION_ION_SUMMARY_PREFIX):
                union_ion_summaries.append(
                    parse_union_ion_summary(text, line_number)
                )
            elif text.startswith(SUMMARY_PREFIX):
                summaries.append(parse_summary(text, line_number))
            elif text.startswith(BLOCKED_PREFIX):
                blocked.append(text.rstrip("\n"))
    if blocked:
        raise SaturationError(f"blocked diagnostic records present: {len(blocked)}")
    if len(summaries) != 1:
        raise SaturationError(f"expected one complete summary, found {len(summaries)}")
    summary = summaries[0]
    target_ion = summary["target_ion_zero_based"]
    for row in rows:
        if row["ion"] != target_ion or row["ion_label"] != target_ion + 1:
            raise SaturationError(
                "selected row ion does not match summary target ion"
            )
    for meta in union_metas:
        if meta["ion"] != target_ion:
            raise SaturationError(
                "union metadata ion does not match summary target ion"
            )
    for item in union_ion_summaries:
        if item["ion"] != target_ion:
            raise SaturationError(
                "union ion summary ion does not match summary target ion"
            )
    if len(rows) != summary["selected_rows"]:
        raise SaturationError(
            f"selected row count mismatch: {len(rows)} != {summary['selected_rows']}"
        )
    selection_mode = summary.get("selection_mode", "COMBINED_PREFIX")
    if selection_mode == "PER_ION_UNION":
        validate_per_ion_union(
            rows, union_metas, union_ion_summaries, summary
        )
    elif union_metas or union_ion_summaries:
        raise SaturationError("combined prefix contains union-only records")

    lines: set[int] = set()
    previous_emission: Decimal | None = None
    previous_fraction: Decimal | None = None
    previous_cumulative_value: float | None = None
    total_emission = float(summary["total_scaled_emission"])
    for index, row in enumerate(rows, 1):
        if selection_mode == "COMBINED_PREFIX" and row["rank"] != index:
            raise SaturationError(f"non-contiguous rank at {index}")
        if row["line"] in lines:
            raise SaturationError(f"duplicate line identity: {row['line']}")
        lines.add(row["line"])
        emission = Decimal(row["scaled_emission_serialized"])
        emission_value = float(emission)
        expected_cumulative = (
            emission_value if previous_cumulative_value is None else
            previous_cumulative_value + emission_value
        )
        observed_cumulative = float(row["cumulative_scaled_emission"])
        if selection_mode == "COMBINED_PREFIX" and not arithmetic_close(
            observed_cumulative, expected_cumulative,
            *(value for value in (previous_cumulative_value, emission_value)
              if value is not None),
        ):
            raise SaturationError(
                f"cumulative emission identity failed at rank {index}"
            )
        fraction = Decimal(row["cumulative_fraction_serialized"])
        if not arithmetic_close(
            float(fraction), observed_cumulative / total_emission,
            observed_cumulative, total_emission,
        ):
            raise SaturationError(
                f"cumulative fraction identity failed at rank {index}"
            )
        previous_cumulative_value = observed_cumulative
        if previous_emission is not None and emission > previous_emission:
            raise SaturationError(f"emission order violated at rank {index}")
        if previous_fraction is not None and fraction <= previous_fraction:
            raise SaturationError(f"cumulative fraction did not increase at rank {index}")
        previous_emission = emission
        previous_fraction = fraction
    if selection_mode == "COMBINED_PREFIX":
        target = Decimal(summary["selection_target_fraction_serialized"])
        if len(rows) > 1 and Decimal(
            rows[-2]["cumulative_fraction_serialized"]
        ) >= target:
            raise SaturationError("selected prefix is not minimal")
        if Decimal(rows[-1]["cumulative_fraction_serialized"]) < target:
            raise SaturationError("last selected row remains below target")
        if Decimal(rows[-1]["cumulative_scaled_emission_serialized"]) != Decimal(
            summary["selected_scaled_emission_serialized"]
        ):
            raise SaturationError("last cumulative emission != summary selection")

    result = {
        "schema": "lumina-a210-line-saturation-summary-v1",
        "status": "PASS",
        "verdict": "READ_ONLY_TOP_90_PERCENT_TARGET_ION_EMISSION_SEALED",
        "source_log": str(log.resolve()),
        "source_log_sha256": digest(log),
        "summary": summary,
        "rows": rows,
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }
    if selection_mode == "PER_ION_UNION":
        result["union_metadata"] = union_metas
        result["union_ion_summaries"] = union_ion_summaries
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = summarize(args.log)
        atomic_write(args.report, payload)
        summary = payload["summary"]
        print(
            "PASS A210_LINE_SATURATION "
            f"candidates={summary['candidate_rows']} "
            f"selected={summary['selected_rows']} "
            f"fraction={summary['selected_fraction']:.17g} repair=0"
        )
        return 0
    except (SaturationError, OSError, UnicodeError) as exc:
        atomic_write(args.report, {
            "schema": "lumina-a210-line-saturation-summary-v1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_LINE_SATURATION reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
