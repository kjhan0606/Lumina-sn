#!/usr/bin/env python3
"""Verify that the selected saturation rows cover 90% of each target ion.

The line-saturation diagnostic selects a combined Fe/Co/Ni IV emission prefix.
This read-only check binds that prefix to the complete REQUESTED_TE shell-0 ion
owner totals from the same stderr log.  It fails closed if any individual ion
has less than 90% coverage.  Arithmetic bounds below validate serialized
long-double sums only; they are never physical tolerances or value repairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any


TARGETS = {26: "Fe IV", 27: "Co IV", 28: "Ni IV"}
TARGET_FRACTION = Decimal("0.9")
LONG_DOUBLE_EPS = Decimal(2) ** Decimal(-63)
REPAIR_KEYS = ("floor", "cap", "clamp", "jitter", "repair")


class CoverageError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoverageError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoverageError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"non-object {label}")
    return value


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CoverageError(f"invalid decimal {label}") from exc
    require(parsed.is_finite(), f"nonfinite decimal {label}")
    return parsed


def validate_no_repairs(document: dict[str, Any], label: str) -> None:
    require(document.get("physical_values_modified") in (False, 0),
            f"{label}: physical mutation present")
    for key in REPAIR_KEYS:
        require(document.get(key) == 0, f"{label}: forbidden {key}")


def print_half_ulp(value: Decimal, digits: int = 21) -> Decimal:
    if value == 0:
        return Decimal(0)
    return Decimal(5).scaleb(abs(value).adjusted() - digits)


def sum_proof_bound(values: list[Decimal], observed: Decimal,
                    operations: int) -> Decimal:
    magnitude = sum((abs(value) for value in values), Decimal(0))
    magnitude = max(magnitude, abs(observed))
    arithmetic = Decimal(8 * max(operations, 1)) * LONG_DOUBLE_EPS * magnitude
    serialization = sum((print_half_ulp(value) for value in values), Decimal(0))
    return arithmetic + serialization + print_half_ulp(observed)


def same_source(saturation: dict[str, Any], owner: dict[str, Any]) -> dict[str, str]:
    sat_source = Path(str(saturation.get("source_log", "")))
    owner_source = Path(str(owner.get("source_log", "")))
    require(sat_source.is_absolute() and owner_source.is_absolute(),
            "source log path is not absolute")
    require(sat_source.resolve() == owner_source.resolve(),
            "saturation and owner reports use different logs")
    require(sat_source.is_file() and not sat_source.is_symlink(),
            "shared source log is missing or unsafe")
    actual = digest(sat_source)
    require(saturation.get("source_log_sha256") == actual and
            owner.get("source_log_sha256") == actual,
            "shared source log SHA mismatch")
    return {"path": str(sat_source.resolve()), "sha256": actual}


def check(saturation_path: Path, owner_path: Path) -> dict[str, Any]:
    saturation = load(saturation_path, "saturation summary")
    owner = load(owner_path, "ion-owner report")
    require(saturation.get("schema") == "lumina-a210-line-saturation-summary-v1" and
            saturation.get("status") == "PASS",
            "saturation summary did not pass")
    require(owner.get("schema") == "a210-line-ion-owner-diagnostic-v1" and
            owner.get("status") == "PASS" and owner.get("complete") is True and
            owner.get("phase") == "REQUESTED_TE",
            "ion-owner report did not pass")
    validate_no_repairs(saturation, "saturation summary")
    validate_no_repairs(owner, "ion-owner report")
    source = same_source(saturation, owner)

    summary = saturation.get("summary")
    rows = saturation.get("rows")
    shells = owner.get("shells")
    require(isinstance(summary, dict) and isinstance(rows, list) and rows,
            "saturation summary has no selected rows")
    try:
        target_ion = int(summary.get("target_ion_zero_based", 3))
    except (KeyError, TypeError, ValueError) as exc:
        raise CoverageError("saturation summary target ion is malformed") from exc
    require(0 <= target_ion <= 10,
            "saturation summary target ion is outside schema")
    target_label = target_ion + 1
    element_names = {26: "Fe", 27: "Co", 28: "Ni"}
    target_species = {
        z: f"{element_names[z]} {target_label}" for z in TARGETS
    }
    require(isinstance(shells, list), "owner report has no shell records")
    shell_zero = [item for item in shells
                  if isinstance(item, dict) and item.get("shell") == 0]
    require(len(shell_zero) == 1, "owner report lacks unique shell zero")
    owners = shell_zero[0].get("owners_by_abs_signed_ion_total")
    require(isinstance(owners, list), "shell-zero owner list missing")

    owner_emission: dict[int, Decimal] = {}
    for item in owners:
        if not isinstance(item, dict):
            raise CoverageError("malformed owner record")
        try:
            z = int(item["Z"])
            stage = int(item["ion_stage"])
            label = int(item["ion_label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CoverageError("malformed owner identity") from exc
        if z in TARGETS and stage == target_ion and label == target_label:
            require(z not in owner_emission, f"duplicate {TARGETS[z]} owner")
            value = decimal(item.get("scaled_emission"),
                            f"{target_species[z]} owner emission")
            require(value > 0, f"nonpositive {target_species[z]} owner emission")
            owner_emission[z] = value
    require(set(owner_emission) == set(TARGETS),
            f"target ion {target_label} owner set incomplete")

    selected: dict[int, list[Decimal]] = {z: [] for z in TARGETS}
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict), f"malformed saturation row {index}")
        try:
            z = int(row["Z"])
            ion = int(row["ion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CoverageError(f"malformed saturation identity {index}") from exc
        require(z in TARGETS and ion == target_ion,
                f"saturation row {index} outside target scope")
        serialized = row.get("scaled_emission_serialized")
        value = decimal(serialized if serialized is not None else
                        row.get("scaled_emission"),
                        f"saturation row {index} emission")
        require(value > 0, f"nonpositive saturation row {index} emission")
        selected[z].append(value)

    selected_values = [value for values in selected.values() for value in values]
    selected_sum = sum(selected_values, Decimal(0))
    summary_selected = decimal(
        summary.get("selected_scaled_emission_serialized",
                    summary.get("selected_scaled_emission")),
        "summary selected emission",
    )
    selected_bound = sum_proof_bound(
        selected_values, summary_selected, len(selected_values) + 2
    )
    require(abs(selected_sum - summary_selected) <= selected_bound,
            "selected row emissions do not close to summary")

    owner_values = list(owner_emission.values())
    owner_sum = sum(owner_values, Decimal(0))
    summary_total = decimal(
        summary.get("total_scaled_emission_serialized",
                    summary.get("total_scaled_emission")),
        "summary target-ion total emission",
    )
    candidate_rows = int(summary.get("candidate_rows", 0))
    require(candidate_rows >= len(rows) and candidate_rows > 0,
            "invalid saturation candidate count")
    owner_bound = sum_proof_bound(
        owner_values, summary_total, 2 * candidate_rows + len(owner_values) + 8
    )
    require(abs(owner_sum - summary_total) <= owner_bound,
            "owner target total does not close to saturation total")

    coverage: list[dict[str, Any]] = []
    all_pass = True
    with localcontext() as context:
        context.prec = 90
        for z in sorted(TARGETS):
            selected_z = sum(selected[z], Decimal(0))
            total_z = owner_emission[z]
            fraction = selected_z / total_z
            passed = fraction >= TARGET_FRACTION
            all_pass = all_pass and passed
            coverage.append({
                "Z": z,
                "species": target_species[z],
                "selected_line_count": len(selected[z]),
                "selected_scaled_emission": str(selected_z),
                "owner_total_scaled_emission": str(total_z),
                "selected_fraction": str(fraction),
                "required_fraction": str(TARGET_FRACTION),
                "coverage_pass": passed,
            })

    return {
        "schema": "lumina-a210-line-saturation-per-ion-coverage-v1",
        "status": "PASS" if all_pass else "FAIL",
        "verdict": (
            f"EACH_FE_CO_NI_{target_label}_SELECTED_EMISSION_AT_LEAST_90_PERCENT"
            if all_pass else
            "COMBINED_PREFIX_UNDERCOVERS_AT_LEAST_ONE_TARGET_ION"
        ),
        "same_source_log": source,
        "saturation_summary": str(saturation_path.resolve()),
        "saturation_summary_sha256": digest(saturation_path),
        "owner_report": str(owner_path.resolve()),
        "owner_report_sha256": digest(owner_path),
        "selected_sum_closure": {
            "row_decimal_sum": str(selected_sum),
            "summary_selected": str(summary_selected),
            "difference": str(abs(selected_sum - summary_selected)),
            "arithmetic_proof_bound": str(selected_bound),
        },
        "owner_total_closure": {
            "owner_decimal_sum": str(owner_sum),
            "saturation_total": str(summary_total),
            "difference": str(abs(owner_sum - summary_total)),
            "arithmetic_proof_bound": str(owner_bound),
        },
        "per_ion": coverage,
        "physical_cause_claim": False,
        "arithmetic_proof_bound_is_physical_tolerance": False,
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saturation-summary", type=Path, required=True)
    parser.add_argument("--owner-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = check(args.saturation_summary, args.owner_report)
        atomic_write(args.report, payload)
        print(
            f"{payload['status']} A210_LINE_SATURATION_PER_ION_COVERAGE "
            + " ".join(
                f"Z{item['Z']}={item['selected_fraction']}"
                for item in payload["per_ion"]
            )
            + " repair=0"
        )
        return 0 if payload["status"] == "PASS" else 4
    except (CoverageError, OSError, UnicodeError, ValueError) as exc:
        atomic_write(args.report, {
            "schema": "lumina-a210-line-saturation-per-ion-coverage-v1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_LINE_SATURATION_PER_ION_COVERAGE reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
