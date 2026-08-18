#!/usr/bin/env python3
"""Validate and summarize an opt-in A2-10 all-cell cancellation census."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from check_a210_cancellation_witnesses import (
    AuditError,
    MARKER,
    PAIR_RE,
    audit_record,
)


SUMMARY_MARKER = "[A2-10][CANCELLATION-CENSUS]"
CSV_FIELDS = (
    "phase",
    "line",
    "shell",
    "status",
    "signed_direction",
    "eta_per_sr",
    "chi_effective",
    "jbar",
    "jbar_bound",
    "absorption_per_sr",
    "net_per_sr",
    "signed_rate",
    "absolute_uncertainty",
    "uncertainty_to_abs_rate",
    "required_symmetric_jbar_bound",
    "current_to_required_bound_ratio",
    "one_sided_jbar_threshold_eta_over_chi",
    "identity_max_relative_error",
    "clamp",
    "floor",
    "jitter",
)


def integer(fields: dict[str, str], name: str) -> int:
    try:
        return int(fields[name])
    except (KeyError, ValueError) as exc:
        raise AuditError(f"invalid or missing integer {name}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row(item: dict[str, Any]) -> dict[str, Any]:
    inputs = item["inputs"]
    reconstructed = item["reconstructed"]
    proof = item["proof_requirement"]
    repair = item["repair_counters"]
    current_to_required = proof["current_to_required_bound_ratio"]
    return {
        "phase": item["phase"],
        "line": item["line"],
        "shell": item["shell"],
        "status": item["status"],
        "signed_direction": item["signed_direction"],
        "eta_per_sr": inputs["eta_per_sr"],
        "chi_effective": inputs["chi_effective"],
        "jbar": inputs["jbar"],
        "jbar_bound": inputs["jbar_bound"],
        "absorption_per_sr": reconstructed["absorption_per_sr"],
        "net_per_sr": reconstructed["net_per_sr"],
        "signed_rate": reconstructed["signed_rate"],
        "absolute_uncertainty": reconstructed["absolute_uncertainty"],
        "uncertainty_to_abs_rate": current_to_required,
        "required_symmetric_jbar_bound": (
            proof["required_symmetric_jbar_bound_strictly_below"]
        ),
        "current_to_required_bound_ratio": current_to_required,
        "one_sided_jbar_threshold_eta_over_chi": (
            proof["one_sided_jbar_threshold_eta_over_chi"]
        ),
        "identity_max_relative_error": item["identity_max_relative_error"],
        "clamp": repair["clamp"],
        "floor": repair["floor"],
        "jitter": repair["jitter"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expect-phase", action="append", default=[])
    parser.add_argument("--maximum-relative-error", default="1e-12")
    args = parser.parse_args()

    try:
        if not args.log.is_file():
            raise AuditError(f"missing or non-regular log: {args.log}")
        maximum_relative_error = Decimal(args.maximum_relative_error)
        if not maximum_relative_error.is_finite() or maximum_relative_error <= 0:
            raise AuditError("maximum relative error must be finite and positive")

        summaries: dict[str, dict[str, str]] = {}
        rows: list[dict[str, Any]] = []
        blocked_per_phase: dict[str, int] = {}
        maximum_identity_error = Decimal(0)
        with args.log.open("r", encoding="utf-8", errors="strict") as stream:
            for line_number, text in enumerate(stream, start=1):
                if MARKER in text:
                    fields = dict(PAIR_RE.findall(text))
                    phase = fields.get("phase")
                    if not phase:
                        raise AuditError(
                            f"{args.log}:{line_number}: census cell lacks phase"
                        )
                    if fields.get("status") == "INVALID_INPUT":
                        raise AuditError(
                            f"{args.log}:{line_number}: non-finite/invalid cell in census"
                        )
                    item = audit_record(fields, maximum_relative_error)
                    blocked_per_phase[phase] = blocked_per_phase.get(phase, 0) + 1
                    maximum_identity_error = max(
                        maximum_identity_error,
                        Decimal(str(item["identity_max_relative_error"])),
                    )
                    rows.append(csv_row(item))
                if SUMMARY_MARKER in text:
                    fields = dict(PAIR_RE.findall(text))
                    phase = fields.get("phase")
                    if not phase:
                        raise AuditError(
                            f"{args.log}:{line_number}: census summary lacks phase"
                        )
                    if phase in summaries:
                        raise AuditError(f"duplicate census summary phase={phase}")
                    summaries[phase] = fields

        expected_phases = set(args.expect_phase)
        observed_phases = set(summaries)
        if expected_phases and observed_phases != expected_phases:
            raise AuditError(
                f"phase mismatch expected={sorted(expected_phases)} "
                f"observed={sorted(observed_phases)}"
            )
        if not summaries:
            raise AuditError("no cancellation census summary records")

        phase_reports: dict[str, Any] = {}
        for phase, fields in sorted(summaries.items()):
            complete = integer(fields, "complete")
            unresolved = integer(fields, "unresolved")
            invalid = integer(fields, "invalid")
            bins = {
                "ratio_le_2": integer(fields, "ratio_le_2"),
                "ratio_2_10": integer(fields, "ratio_2_10"),
                "ratio_10_100": integer(fields, "ratio_10_100"),
                "ratio_gt_100": integer(fields, "ratio_gt_100"),
                "ratio_infinite": integer(fields, "ratio_infinite"),
            }
            observed_blocked = blocked_per_phase.get(phase, 0)
            if complete != 1:
                raise AuditError(f"phase={phase} scan incomplete")
            if invalid != 0:
                raise AuditError(f"phase={phase} invalid={invalid}")
            if sum(bins.values()) != unresolved:
                raise AuditError(f"phase={phase} ratio bins do not sum to unresolved")
            if observed_blocked != unresolved + invalid:
                raise AuditError(
                    f"phase={phase} cell/summary mismatch "
                    f"cells={observed_blocked} unresolved={unresolved} invalid={invalid}"
                )
            for flag in ("physical_values_modified", "clamp", "floor", "jitter"):
                if integer(fields, flag) != 0:
                    raise AuditError(f"phase={phase} forbidden {flag}={fields[flag]}")
            phase_reports[phase] = {
                "evaluated_cells": integer(fields, "evaluated_cells"),
                "unresolved": unresolved,
                "invalid": invalid,
                "ratio_bins": bins,
                "max_finite_uncertainty_to_abs_rate": fields.get(
                    "max_finite_uncertainty_to_abs_rate"
                ),
                "first_unresolved_line": integer(fields, "first_unresolved_line"),
                "first_unresolved_shell": integer(fields, "first_unresolved_shell"),
                "physical_values_modified": False,
                "publication": fields.get("publication"),
            }

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "schema": "lumina-a2-10-cancellation-census-v1",
            "status": "PASS",
            "reason_code": "ALL_CELL_CANCELLATION_CENSUS_COMPLETE",
            "source_log": str(args.log.resolve()),
            "source_sha256": sha256_file(args.log),
            "csv": str(args.csv.resolve()),
            "csv_sha256": sha256_file(args.csv),
            "phases": phase_reports,
            "unresolved_cell_rows": len(rows),
            "maximum_identity_relative_error": str(maximum_identity_error),
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            "PASS A2_10_CANCELLATION_CENSUS "
            f"phases={','.join(sorted(phase_reports))} rows={len(rows)} "
            "invalid=0 repair=0"
        )
        return 0
    except (AuditError, OSError, ValueError) as exc:
        print(f"FAIL A2_10_CANCELLATION_CENSUS reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
