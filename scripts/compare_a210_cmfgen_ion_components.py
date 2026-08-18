#!/usr/bin/env python3
"""Compare matched-temperature ion net/emission/absorption diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compare_a210_cmfgen_ion_owners import (
    cmfgen_key,
    finite_number,
    require_zero_repairs,
)
from extract_cmfgen_line_net_fixture import CMFGEN_RE_INTERNAL_TO_CGS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolate(lo: float, hi: float, fraction: float) -> float:
    return lo + fraction * (hi - lo)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def keyed_rows(rows: list[dict], label: str) -> dict[tuple[int, int], dict]:
    keyed: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = cmfgen_key(row, label)
        if key in keyed:
            raise SystemExit(f"duplicate CMFGEN ion owner in {label}: {key}")
        keyed[key] = row
    return keyed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumina-owner", required=True, type=Path)
    parser.add_argument("--cmfgen-components", required=True, type=Path)
    parser.add_argument("--cmfgen-finite", required=True, type=Path)
    parser.add_argument("--shell", default=0, type=int)
    parser.add_argument("--depth-lo", default=67, type=int)
    parser.add_argument("--depth-hi", default=68, type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    lumina = json.loads(args.lumina_owner.read_text())
    components = json.loads(args.cmfgen_components.read_text())
    finite = json.loads(args.cmfgen_finite.read_text())
    if lumina.get("status") != "PASS" or not lumina.get("complete"):
        raise SystemExit("Lumina owner callback is not complete")
    if lumina.get("physical_values_modified") is not False:
        raise SystemExit("Lumina owner report lacks the no-mutation contract")
    require_zero_repairs(lumina, "Lumina owner report")
    if (components.get("schema") != "cmfgen-line-components-ion-owner-v1" or
            components.get("verdict") !=
            "COMPLETE_FINITE_COMPONENT_DECOMPOSITION"):
        raise SystemExit("CMFGEN component decomposition is not complete")
    if components.get("physical_values_modified") not in (0, False):
        raise SystemExit("CMFGEN components lack the no-mutation contract")
    require_zero_repairs(components, "CMFGEN components")
    if (finite.get("schema") != "lumina-cmfgen-lineheat-finite-reference-v1" or
            finite.get("verdict") !=
            "FINITE_REFERENCE_REPRODUCED_MATCHED_STATE_PENDING"):
        raise SystemExit("CMFGEN finite reference is not sealed")
    if finite.get("physical_mutation") not in (0, False):
        raise SystemExit("CMFGEN finite reference lacks the no-mutation contract")
    require_zero_repairs(finite, "CMFGEN finite reference")
    shells = {int(row["shell"]): row for row in lumina["shells"]}
    shell = shells.get(args.shell)
    if shell is None or "owners_by_abs_signed_ion_total" not in shell:
        raise SystemExit("requested Lumina shell is absent")
    lo = components["depths"][str(args.depth_lo)]
    hi = components["depths"][str(args.depth_hi)]
    for label, depth in (("low depth", lo), ("high depth", hi)):
        finite_check = depth.get("finite_reference_check", {})
        if (not depth.get("cellwise_component_closure_verified") or
                finite_check.get("signed_bit_exact") is not True or
                finite_check.get("absolute_bit_exact") is not True):
            raise SystemExit(f"unsealed component closure at {label}")
    lo_rows = keyed_rows(lo["owners_by_abs_signed_ion_total"], "low depth")
    hi_rows = keyed_rows(hi["owners_by_abs_signed_ion_total"], "high depth")
    if set(lo_rows) != set(hi_rows):
        raise SystemExit("CMFGEN interpolation depths have different ion owners")
    lumina_rows: dict[tuple[int, int], dict] = {}
    for row in shell["owners_by_abs_signed_ion_total"]:
        key = (int(row["Z"]), int(row["ion_label"]))
        if key in lumina_rows:
            raise SystemExit(f"duplicate Lumina ion owner: {key}")
        lumina_rows[key] = row
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

    def interpolated_internal(field: str) -> float:
        return interpolate(
            finite_number(lo[field], f"CMFGEN low {field}"),
            finite_number(hi[field], f"CMFGEN high {field}"), fraction)

    cmfgen_total_net = (
        interpolated_internal("line_order_signed_internal") *
        CMFGEN_RE_INTERNAL_TO_CGS)
    cmfgen_total_emission = (
        interpolated_internal("line_order_scaled_emission_internal") *
        CMFGEN_RE_INTERNAL_TO_CGS)
    cmfgen_total_absorption = (
        interpolated_internal("line_order_scaled_absorption_internal") *
        CMFGEN_RE_INTERNAL_TO_CGS)
    finite_total_net = finite_number(
        state["signed_cgs_erg_cm3_s"], "CMFGEN finite signed total")
    if cmfgen_total_net != finite_total_net:
        raise SystemExit("CMFGEN component net drifted from finite reference")
    if cmfgen_total_emission < 0.0 or cmfgen_total_absorption < 0.0:
        raise SystemExit("negative CMFGEN total line component")
    lumina_total_net = finite_number(
        shell["line_order_signed_rate"], "Lumina signed total")
    lumina_total_emission = finite_number(
        shell["line_order_emission"], "Lumina emission total")
    lumina_total_absorption = finite_number(
        shell["line_order_absorption"], "Lumina absorption total")
    if lumina_total_emission < 0.0 or lumina_total_absorption < 0.0:
        raise SystemExit("negative Lumina total line component")

    comparisons = []
    for key, lumina_row in lumina_rows.items():
        if key not in lo_rows or key not in hi_rows:
            continue
        low = lo_rows[key]
        high = hi_rows[key]
        if low["normalized_species"] != high["normalized_species"]:
            raise SystemExit(f"CMFGEN ion label mismatch across depths: {key}")
        cmfgen_net = interpolate(
            finite_number(low["signed_cgs_erg_cm3_s"], "CMFGEN net"),
            finite_number(high["signed_cgs_erg_cm3_s"], "CMFGEN net"),
            fraction)
        cmfgen_emission = interpolate(
            finite_number(low["scaled_emission_cgs_erg_cm3_s"],
                          "CMFGEN emission"),
            finite_number(high["scaled_emission_cgs_erg_cm3_s"],
                          "CMFGEN emission"), fraction)
        cmfgen_absorption = interpolate(
            finite_number(low["scaled_absorption_cgs_erg_cm3_s"],
                          "CMFGEN absorption"),
            finite_number(high["scaled_absorption_cgs_erg_cm3_s"],
                          "CMFGEN absorption"), fraction)
        lumina_net = finite_number(lumina_row["signed_rate"], "Lumina net")
        lumina_emission = finite_number(
            lumina_row["scaled_emission"], "Lumina emission")
        lumina_absorption = finite_number(
            lumina_row["scaled_absorption"], "Lumina absorption")
        if (cmfgen_emission < 0.0 or cmfgen_absorption < 0.0 or
                lumina_emission < 0.0 or lumina_absorption < 0.0):
            raise SystemExit(f"negative line component for ion owner {key}")
        comparisons.append({
            "Z": key[0], "ion_label": key[1],
            "normalized_species": low["normalized_species"],
            "lumina": {
                "signed_rate_erg_cm3_s": lumina_net,
                "scaled_emission_erg_cm3_s": lumina_emission,
                "scaled_absorption_erg_cm3_s": lumina_absorption,
                "component_closure_erg_cm3_s": (
                    lumina_emission - lumina_absorption - lumina_net),
            },
            "cmfgen_interpolated": {
                "signed_rate_erg_cm3_s": cmfgen_net,
                "scaled_emission_erg_cm3_s": cmfgen_emission,
                "scaled_absorption_erg_cm3_s": cmfgen_absorption,
                "component_closure_erg_cm3_s": (
                    cmfgen_emission - cmfgen_absorption - cmfgen_net),
            },
            "lumina_to_cmfgen": {
                "signed_rate_ratio": ratio(lumina_net, cmfgen_net),
                "scaled_emission_ratio": ratio(
                    lumina_emission, cmfgen_emission),
                "scaled_absorption_ratio": ratio(
                    lumina_absorption, cmfgen_absorption),
            },
        })
    comparisons.sort(
        key=lambda row: abs(row["lumina"]["signed_rate_erg_cm3_s"]),
        reverse=True)
    if not comparisons:
        raise SystemExit("Lumina and CMFGEN have no common ion owner")

    report = {
        "schema": "a210-cmfgen-ion-component-comparison-v1",
        "status": "FINITE_COMPONENT_COMPARISON_STATE_UNMATCHED",
        "sources": {
            "lumina_owner": {"path": str(args.lumina_owner.resolve()),
                             "sha256": digest(args.lumina_owner)},
            "cmfgen_components": {
                "path": str(args.cmfgen_components.resolve()),
                "sha256": digest(args.cmfgen_components)},
            "cmfgen_finite": {"path": str(args.cmfgen_finite.resolve()),
                              "sha256": digest(args.cmfgen_finite)},
        },
        "mapping": {
            "lumina_shell": args.shell,
            "cmfgen_depths": [args.depth_lo, args.depth_hi],
            "velocity_interpolation_fraction": fraction,
            "same_signed_component_equation": True,
            "units": "erg cm^-3 s^-1",
        },
        "state": {
            "lumina_temperature_K": lumina_temperature,
            "cmfgen_temperature_K": cmfgen_temperature,
            "temperature_exact_match": True,
            "lumina_electron_density_cm3": lumina_ne,
            "cmfgen_electron_density_cm3": cmfgen_ne,
            "cmfgen_to_lumina_electron_density_ratio": cmfgen_ne / lumina_ne,
            "matched_electron_density": lumina_ne == cmfgen_ne,
            "matched_ion_and_level_populations": False,
            "matched_radiation_field_or_Jbar": False,
            "matched_state": False,
        },
        "totals": {
            "lumina": {
                "signed_rate_erg_cm3_s": lumina_total_net,
                "scaled_emission_erg_cm3_s": lumina_total_emission,
                "scaled_absorption_erg_cm3_s": lumina_total_absorption,
                "component_closure_erg_cm3_s": (
                    lumina_total_emission - lumina_total_absorption -
                    lumina_total_net),
            },
            "cmfgen_interpolated": {
                "signed_rate_erg_cm3_s": cmfgen_total_net,
                "scaled_emission_erg_cm3_s": cmfgen_total_emission,
                "scaled_absorption_erg_cm3_s": cmfgen_total_absorption,
                "separately_summed_component_closure_erg_cm3_s": (
                    cmfgen_total_emission - cmfgen_total_absorption -
                    cmfgen_total_net),
            },
            "lumina_to_cmfgen": {
                "signed_rate_ratio": ratio(
                    lumina_total_net, cmfgen_total_net),
                "scaled_emission_ratio": ratio(
                    lumina_total_emission, cmfgen_total_emission),
                "scaled_absorption_ratio": ratio(
                    lumina_total_absorption, cmfgen_total_absorption),
            },
        },
        "common_ion_components": comparisons,
        "common_owner_count": len(comparisons),
        "cmfgen_component_precision": (
            "Diagnostic only: components are reconstructed from serialized "
            "LINEHEAT (5 significant digits) and NETRATE (7 significant "
            "digits); the original finite net LINEHEAT remains authoritative."),
        "parity_claim": False,
        "interpretation": (
            "Emission-ratio structure diagnoses population/material mismatch; "
            "absorption additionally contains Jbar. Temperature alone is "
            "aligned, so none of these ratios is a reproduction verdict."),
        "physical_values_modified": 0,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "common_owner_count": len(comparisons),
        "report": str(args.report),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
