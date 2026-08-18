#!/usr/bin/env python3
"""Seal complete diagnostic-only A2-10 signed line totals by ion owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal, localcontext
from pathlib import Path


OWNER = "[A2-10][LINE-ION-OWNER]"
SUMMARY = "[A2-10][LINE-ION-OWNER-SUMMARY]"
BLOCKED = "[A2-10][LINE-ION-OWNER-BLOCKED]"
CELL_BLOCKED = "[A2-10][LINE-NET-CELL-BLOCKED]"
INTERIOR = "[A2-10][VECTOR-INTERIOR-SCAN]"
LONG_DOUBLE_EPS = Decimal(2) ** Decimal(-63)


def fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def print_half_ulp(value: Decimal, digits: int = 21) -> Decimal:
    if not value:
        return Decimal(0)
    return Decimal(5).scaleb(value.copy_abs().adjusted() - digits)


def grouped_bound(values: list[Decimal], grouped: Decimal) -> Decimal:
    absolute = sum((abs(value) for value in values), Decimal(0))
    arithmetic = Decimal(4 * (len(values) + 1)) * LONG_DOUBLE_EPS * absolute
    transcription = sum((print_half_ulp(value) for value in values), Decimal(0))
    return arithmetic + transcription + print_half_ulp(grouped)


def signed_line_order_bound(summary: dict[str, str]) -> Decimal:
    """Bound only the change in summation order, never a physical sign/value."""
    cells = int(summary["eligible_cells"])
    owners = int(summary["ion_records"])
    line_order = Decimal(summary["line_order_signed_rate"])
    grouped = Decimal(summary["grouped_signed_rate"])
    absolute = Decimal(summary["line_order_absolute_sum"])
    if cells < 0 or owners < 0 or absolute < 0:
        raise SystemExit("invalid signed grouping metadata")
    scale = max(absolute, abs(line_order), abs(grouped))
    # Both paths traverse the same accepted double-valued cells.  Cover the
    # line-order accumulation, the within-owner accumulations, the final owner
    # grouping, and their decimal serialization with a deliberately
    # conservative long-double operation count.  This bound is an evidence
    # check only: it is never used to accept, replace, or repair a physical
    # rate or its sign.
    operations = 2 * cells + owners + 8
    arithmetic = Decimal(4 * operations) * LONG_DOUBLE_EPS * scale
    transcription = (
        print_half_ulp(line_order) + print_half_ulp(grouped) +
        print_half_ulp(absolute)
    )
    return arithmetic + transcription


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--phase", default="REQUESTED_TE")
    parser.add_argument("--expected-shells", type=int, required=True)
    parser.add_argument("--expected-temperature-K", type=Decimal, required=True)
    args = parser.parse_args()

    if args.expected_shells <= 0 or not args.log.is_file():
        raise SystemExit(2)

    records: dict[tuple[int, int], dict[str, str]] = {}
    summaries: dict[int, dict[str, str]] = {}
    owner_blocked: list[str] = []
    cell_blocked: list[dict[str, str]] = []
    requested_valid = False
    with args.log.open(errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if BLOCKED in line:
                owner_blocked.append(line)
            if CELL_BLOCKED in line:
                item = fields(line)
                if item.get("phase") == args.phase:
                    cell_blocked.append(item)
            if INTERIOR in line:
                item = fields(line)
                if item.get("phase") == args.phase and item.get("valid") == "1":
                    requested_valid = True
            if OWNER in line and SUMMARY not in line:
                item = fields(line)
                if item.get("phase") != args.phase:
                    continue
                key = (int(item["shell"]), int(item["ion_slot"]))
                if key in records:
                    raise SystemExit(f"duplicate owner record: {key}")
                records[key] = item
            elif SUMMARY in line:
                item = fields(line)
                if item.get("phase") != args.phase:
                    continue
                shell = int(item["shell"])
                if shell in summaries:
                    raise SystemExit(f"duplicate owner summary: shell={shell}")
                summaries[shell] = item

    report: dict[str, object] = {
        "schema": "a210-line-ion-owner-diagnostic-v1",
        "phase": args.phase,
        "expected_shells": args.expected_shells,
        "expected_temperature_K": str(args.expected_temperature_K),
        "source_log": str(args.log.resolve()),
        "source_log_sha256": sha256(args.log),
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }

    if not records or len(summaries) != args.expected_shells:
        report.update({
            "status": "BLOCKED_INCOMPLETE_CALLBACK",
            "complete": False,
            "owner_records": len(records),
            "summary_shells": sorted(summaries),
            "owner_blocked": owner_blocked,
            "first_requested_cell_blocked": cell_blocked[0] if cell_blocked else None,
        })
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        return 4

    numeric_pairs = (
        ("signed_rate", "grouped_signed_rate"),
        ("absolute_signed_sum", "grouped_absolute_sum"),
        ("uncertainty", "grouped_uncertainty"),
        ("scaled_emission", "grouped_emission"),
        ("scaled_absorption", "grouped_absorption"),
    )
    shell_reports: list[dict[str, object]] = []
    with localcontext() as context:
        context.prec = 90
        for shell in range(args.expected_shells):
            summary = summaries.get(shell)
            if summary is None:
                raise SystemExit(f"missing shell summary: {shell}")
            owned = [item for (record_shell, _), item in records.items()
                     if record_shell == shell]
            if len(owned) != int(summary["ion_records"]):
                raise SystemExit(f"ion record count mismatch: shell={shell}")
            if Decimal(summary["T_e_K"]) != args.expected_temperature_K:
                raise SystemExit(f"temperature mismatch: shell={shell}")
            for item in owned:
                required = {
                    "complete": "1", "interpretation": "DIAGNOSTIC_ONLY",
                    "physical_values_modified": "0", "clamp": "0",
                    "floor": "0", "jitter": "0", "repair": "0",
                }
                if any(item.get(key) != value for key, value in required.items()):
                    raise SystemExit(f"unsafe owner record: shell={shell}")
            if any(summary.get(key) != value for key, value in {
                "complete": "1", "interpretation": "DIAGNOSTIC_ONLY",
                "physical_values_modified": "0", "clamp": "0",
                "floor": "0", "jitter": "0", "repair": "0",
            }.items()):
                raise SystemExit(f"unsafe owner summary: shell={shell}")

            checks: dict[str, object] = {}
            for record_key, summary_key in numeric_pairs:
                values = [Decimal(item[record_key]) for item in owned]
                expected = sum(values, Decimal(0))
                observed = Decimal(summary[summary_key])
                difference = abs(expected - observed)
                bound = grouped_bound(values, observed)
                if difference > bound:
                    raise SystemExit(
                        f"grouped {record_key} mismatch: shell={shell} "
                        f"difference={difference} bound={bound}")
                checks[record_key] = {
                    "record_decimal_sum": str(expected),
                    "summary_grouped": str(observed),
                    "difference": str(difference),
                    "roundoff_bound": str(bound),
                }
            eligible = sum(int(item["eligible_cells"]) for item in owned)
            if eligible != int(summary["eligible_cells"]):
                raise SystemExit(f"eligible-cell closure mismatch: shell={shell}")
            signed_line_order = Decimal(summary["line_order_signed_rate"])
            signed_grouped = Decimal(summary["grouped_signed_rate"])
            signed_grouping_difference = abs(
                signed_grouped - signed_line_order)
            signed_grouping_bound = signed_line_order_bound(summary)
            if signed_grouping_difference > signed_grouping_bound:
                raise SystemExit(
                    f"signed line-order grouping mismatch: shell={shell} "
                    f"difference={signed_grouping_difference} "
                    f"bound={signed_grouping_bound}")
            ranked = sorted(
                owned,key=lambda item: abs(Decimal(item["signed_rate"])),
                reverse=True)
            rendered_owners = [
                {key: item[key] for key in (
                    "ion_slot", "Z", "ion_stage", "ion_label",
                    "signed_rate", "absolute_signed_sum", "uncertainty",
                    "scaled_emission", "scaled_absorption", "eligible_cells",
                    "cooling_cells", "heating_cells", "exact_zero_cells",
                    "srce_chk_cells")}
                for item in ranked
            ]
            shell_reports.append({
                "shell": shell,
                "temperature_K": summary["T_e_K"],
                "electron_density_cm3": summary["n_e_cm3"],
                "ion_records": len(owned),
                "eligible_cells": eligible,
                "line_order_signed_rate": summary["line_order_signed_rate"],
                "grouped_signed_rate": summary["grouped_signed_rate"],
                "signed_grouping_delta": summary["signed_grouping_delta"],
                "signed_line_order_grouping_check": {
                    "difference": str(signed_grouping_difference),
                    "roundoff_bound": str(signed_grouping_bound),
                    "physical_acceptance_or_repair": False,
                },
                "line_order_absolute_sum": summary["line_order_absolute_sum"],
                "line_order_uncertainty": summary["line_order_uncertainty"],
                "line_order_emission": summary["line_order_emission"],
                "line_order_absorption": summary["line_order_absorption"],
                "grouping_checks": checks,
                "owners_by_abs_signed_ion_total": rendered_owners,
                "top8_by_abs_signed_ion_total": rendered_owners[:8],
            })

    if owner_blocked or cell_blocked or not requested_valid:
        raise SystemExit("complete owner records conflict with blocked/invalid callback")
    report.update({
        "status": "PASS",
        "complete": True,
        "owner_records": len(records),
        "requested_vector_scan_valid": True,
        "shells": shell_reports,
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "owner_records": len(records),
                      "report": str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
