#!/usr/bin/env python3
"""Compare finite Lumina/CMFGEN signed line owners without claiming parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from cmfgen_ion_identity import parse_cmfgen_ion_id


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_number(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"nonfinite {label}")
    return number


def require_zero_repairs(payload: dict, label: str) -> None:
    for key in ("floor", "cap", "clamp", "jitter", "repair"):
        if payload.get(key) != 0:
            raise SystemExit(f"{label} lacks an explicit zero {key} marker")


def cmfgen_key(row: dict, label: str = "CMFGEN owner") -> tuple[int, int]:
    raw = row.get("cmfgen_label")
    normalized = row.get("normalized_species")
    if not isinstance(raw, str) or not isinstance(normalized, str):
        raise SystemExit(f"{label} lacks raw/normalized ion provenance")
    try:
        atomic_number, stage, expected = parse_cmfgen_ion_id(raw)
    except ValueError as exc:
        raise SystemExit(f"{label}: {exc}") from exc
    if normalized != expected:
        raise SystemExit(
            f"{label} normalization mismatch for {raw}: "
            f"expected {expected!r}, found {normalized!r}")
    return atomic_number, stage


def keyed_cmfgen_rows(rows: list[dict], label: str) -> dict[tuple[int, int], dict]:
    keyed: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = cmfgen_key(row, label)
        if key in keyed:
            raise SystemExit(f"duplicate CMFGEN ion owner in {label}: {key}")
        signed = finite_number(row["signed_cgs_erg_cm3_s"],
                               f"{label} signed owner")
        if "absolute_cgs_erg_cm3_s" in row:
            absolute = finite_number(row["absolute_cgs_erg_cm3_s"],
                                     f"{label} absolute owner")
            if absolute < 0.0 or absolute < abs(signed):
                raise SystemExit(f"invalid CMFGEN absolute owner in {label}: {key}")
        keyed[key] = row
    return keyed


def keyed_lumina_rows(rows: list[dict]) -> dict[tuple[int, int], dict]:
    keyed: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = (int(row["Z"]), int(row["ion_label"]))
        if key[0] <= 0 or key[1] <= 0 or key in keyed:
            raise SystemExit(f"invalid or duplicate Lumina ion owner: {key}")
        signed = finite_number(row["signed_rate"], "Lumina signed owner")
        absolute = finite_number(
            row["absolute_signed_sum"], "Lumina absolute owner")
        if absolute < 0.0 or absolute < abs(signed):
            raise SystemExit(f"invalid Lumina absolute owner: {key}")
        keyed[key] = row
    return keyed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumina-owner", type=Path, required=True)
    parser.add_argument("--cmfgen-owner", type=Path, required=True)
    parser.add_argument("--cmfgen-finite", type=Path, required=True)
    parser.add_argument("--shell", type=int, default=0)
    parser.add_argument("--depth-lo", type=int, default=67)
    parser.add_argument("--depth-hi", type=int, default=68)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    lumina = json.loads(args.lumina_owner.read_text())
    cmfgen = json.loads(args.cmfgen_owner.read_text())
    finite = json.loads(args.cmfgen_finite.read_text())
    if (lumina.get("schema") != "a210-line-ion-owner-diagnostic-v1" or
            lumina.get("status") != "PASS" or
            lumina.get("complete") is not True or
            lumina.get("phase") != "REQUESTED_TE"):
        raise SystemExit("Lumina owner callback is not a complete requested-Te record")
    if lumina.get("physical_values_modified") is not False:
        raise SystemExit("Lumina owner report lacks the no-mutation contract")
    require_zero_repairs(lumina, "Lumina owner report")
    if (cmfgen.get("schema") != "cmfgen-lineheat-ion-owner-summary-v1" or
            cmfgen.get("verdict") !=
            "COMPLETE_DIAGNOSTIC_OWNER_DECOMPOSITION"):
        raise SystemExit("CMFGEN signed owner decomposition is not complete")
    if cmfgen.get("physical_mutation") not in (0, False):
        raise SystemExit("CMFGEN signed owners lack the no-mutation contract")
    require_zero_repairs(cmfgen, "CMFGEN signed owners")
    if (finite.get("schema") != "lumina-cmfgen-lineheat-finite-reference-v1" or
            finite.get("verdict") !=
            "FINITE_REFERENCE_REPRODUCED_MATCHED_STATE_PENDING"):
        raise SystemExit("CMFGEN finite reference is not sealed")
    if finite.get("physical_mutation") not in (0, False):
        raise SystemExit("CMFGEN finite reference lacks the no-mutation contract")
    require_zero_repairs(finite, "CMFGEN finite reference")

    shells = {int(item["shell"]): item for item in lumina["shells"]}
    shell = shells.get(args.shell)
    if shell is None or "owners_by_abs_signed_ion_total" not in shell:
        raise SystemExit("Lumina report lacks the requested complete owner rows")
    try:
        low = cmfgen["depths"][str(args.depth_lo)]
        high = cmfgen["depths"][str(args.depth_hi)]
    except KeyError as exc:
        raise SystemExit("requested CMFGEN interpolation depth is absent") from exc
    for label, depth in (("low depth", low), ("high depth", high)):
        check = depth.get("finite_reference_check", {})
        if (check.get("signed_bit_exact") is not True or
                check.get("absolute_bit_exact") is not True):
            raise SystemExit(f"unsealed signed owner decomposition at {label}")
    low_rows = keyed_cmfgen_rows(
        low["top_by_abs_signed_ion_total"], "low depth")
    high_rows = keyed_cmfgen_rows(
        high["top_by_abs_signed_ion_total"], "high depth")
    if set(low_rows) != set(high_rows):
        raise SystemExit("CMFGEN interpolation depths have different ion owners")
    lumina_rows = keyed_lumina_rows(shell["owners_by_abs_signed_ion_total"])

    state = finite["shell_zero_velocity_interpolation"]
    fraction = finite_number(
        state["fraction_from_depth_67_to_68"], "interpolation fraction")
    if not 0.0 <= fraction <= 1.0:
        raise SystemExit("CMFGEN interpolation fraction is outside its bracket")
    lumina_temperature = finite_number(shell["temperature_K"],
                                       "Lumina temperature")
    lumina_ne = finite_number(shell["electron_density_cm3"],
                              "Lumina electron density")
    cmfgen_temperature = finite_number(state["temperature_K"],
                                       "CMFGEN temperature")
    cmfgen_ne = finite_number(state["electron_density_cm3"],
                              "CMFGEN electron density")
    if lumina_temperature <= 0.0 or cmfgen_temperature <= 0.0:
        raise SystemExit("nonpositive comparison temperature")
    if lumina_ne <= 0.0 or cmfgen_ne <= 0.0:
        raise SystemExit("nonpositive comparison electron density")
    if lumina_temperature != cmfgen_temperature:
        raise SystemExit("requested Lumina temperature is not CMFGEN-aligned")

    low_total = finite_number(
        low["line_order_signed_cgs_erg_cm3_s"], "CMFGEN low total")
    high_total = finite_number(
        high["line_order_signed_cgs_erg_cm3_s"], "CMFGEN high total")
    interpolated_total = low_total + fraction * (high_total - low_total)
    finite_total = finite_number(
        state["signed_cgs_erg_cm3_s"], "CMFGEN finite total")
    if interpolated_total != finite_total:
        raise SystemExit("CMFGEN signed owner interpolation drifted from finite reference")

    comparisons = []
    for key, lumina_row in lumina_rows.items():
        if key not in low_rows:
            continue
        lo_row, hi_row = low_rows[key], high_rows[key]
        lo_rate = finite_number(
            lo_row["signed_cgs_erg_cm3_s"], "CMFGEN low owner")
        hi_rate = finite_number(
            hi_row["signed_cgs_erg_cm3_s"], "CMFGEN high owner")
        cmfgen_rate = lo_rate + fraction * (hi_rate - lo_rate)
        lumina_rate = finite_number(lumina_row["signed_rate"],
                                    "Lumina signed owner")
        comparisons.append({
            "Z": key[0],
            "ion_label": key[1],
            "cmfgen_label_lo": lo_row["cmfgen_label"],
            "normalized_species": lo_row["normalized_species"],
            "lumina_signed_rate_erg_cm3_s": lumina_rate,
            "cmfgen_interpolated_signed_rate_erg_cm3_s": cmfgen_rate,
            "lumina_to_cmfgen_signed_rate_ratio": (
                lumina_rate / cmfgen_rate if cmfgen_rate != 0.0 else None),
            "lumina_absolute_signed_sum": finite_number(
                lumina_row["absolute_signed_sum"], "Lumina absolute owner"),
            "lumina_cancellation_condition": (
                finite_number(lumina_row["absolute_signed_sum"],
                              "Lumina absolute owner") / abs(lumina_rate)
                if lumina_rate != 0.0 else None),
            "cmfgen_cancellation_condition_depth_lo": lo_row[
                "cancellation_condition"],
            "cmfgen_cancellation_condition_depth_hi": hi_row[
                "cancellation_condition"],
        })
    comparisons.sort(
        key=lambda row: abs(row["lumina_signed_rate_erg_cm3_s"]),
        reverse=True)
    if not comparisons:
        raise SystemExit("Lumina and CMFGEN have no common ion owner")

    report = {
        "schema": "a210-cmfgen-ion-owner-comparison-v1",
        "status": "FINITE_COMPARISON_STATE_UNMATCHED",
        "lumina_owner_source": {
            "path": str(args.lumina_owner.resolve()),
            "sha256": digest(args.lumina_owner),
        },
        "cmfgen_owner_source": {
            "path": str(args.cmfgen_owner.resolve()),
            "sha256": digest(args.cmfgen_owner),
        },
        "cmfgen_finite_source": {
            "path": str(args.cmfgen_finite.resolve()),
            "sha256": digest(args.cmfgen_finite),
        },
        "mapping": {
            "lumina_shell": args.shell,
            "cmfgen_depths": [args.depth_lo, args.depth_hi],
            "velocity_interpolation_fraction": fraction,
            "same_signed_observable": True,
            "units": "erg cm^-3 s^-1",
        },
        "state": {
            "lumina_temperature_K": lumina_temperature,
            "cmfgen_interpolated_temperature_K": cmfgen_temperature,
            "temperature_exact_match": True,
            "lumina_electron_density_cm3": lumina_ne,
            "cmfgen_interpolated_electron_density_cm3": cmfgen_ne,
            "cmfgen_to_lumina_electron_density_ratio": cmfgen_ne / lumina_ne,
            "mass_density_ratio_cmfgen_to_lumina": finite_number(
                state["cmfgen_to_lumina_mass_density_ratio"],
                "mass density ratio"),
            "matched_electron_density": lumina_ne == cmfgen_ne,
            "matched_ion_and_level_populations": False,
            "matched_radiation_field_or_Jbar": False,
            "matched_state": False,
        },
        "totals": {
            "lumina_signed_rate_erg_cm3_s": finite_number(
                shell["line_order_signed_rate"], "Lumina total"),
            "cmfgen_interpolated_signed_rate_erg_cm3_s": finite_total,
        },
        "common_ion_owners": comparisons,
        "common_owner_count": len(comparisons),
        "parity_claim": False,
        "interpretation": (
            "Temperature is aligned only to isolate the owner structure.  "
            "Electron density, ion/level populations, and Jbar remain unmatched; "
            "rate ratios diagnose the next state-matching branch and are not a "
            "CMFGEN reproduction verdict."),
        "physical_values_modified": 0,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "common_owner_count": len(comparisons),
                      "report": str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
