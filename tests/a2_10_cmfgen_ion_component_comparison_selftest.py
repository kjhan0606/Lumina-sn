#!/usr/bin/env python3
from __future__ import annotations

import json
import copy
import math
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/compare_a210_cmfgen_ion_components.py"


def main() -> int:
    zero_repairs = {
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
    }
    lumina = {"status": "PASS", "complete": True,
              "physical_values_modified": False, **zero_repairs, "shells": [{
        "shell": 0, "temperature_K": "10000", "electron_density_cm3": "2e9",
        "line_order_signed_rate": "1", "line_order_emission": "5",
        "line_order_absorption": "4",
        "owners_by_abs_signed_ion_total": [{
            "Z": "27", "ion_label": "4", "signed_rate": "1",
            "scaled_emission": "5", "scaled_absorption": "4"}],
    }]}
    def depth(net: float, emission: float, absorption: float) -> dict:
        internal_to_cgs = 4.0 * math.pi * 1.0e-10
        return {
            "line_order_signed_internal": net / internal_to_cgs,
            "line_order_scaled_emission_internal": (
                emission / internal_to_cgs),
            "line_order_scaled_absorption_internal": (
                absorption / internal_to_cgs),
            "cellwise_component_closure_verified": True,
            "finite_reference_check": {
                "signed_bit_exact": True, "absolute_bit_exact": True},
            "owners_by_abs_signed_ion_total": [{
            "cmfgen_label": "CoIV", "normalized_species": "Co IV",
            "signed_cgs_erg_cm3_s": net,
            "scaled_emission_cgs_erg_cm3_s": emission,
            "scaled_absorption_cgs_erg_cm3_s": absorption,
        }]}
    components = {
        "schema": "cmfgen-line-components-ion-owner-v1",
        "verdict": "COMPLETE_FINITE_COMPONENT_DECOMPOSITION",
        "physical_values_modified": 0, **zero_repairs,
        "depths": {"67": depth(2, 8, 6), "68": depth(4, 12, 8)},
    }
    internal_to_cgs = 4.0 * math.pi * 1.0e-10
    low_internal = components["depths"]["67"]["line_order_signed_internal"]
    high_internal = components["depths"]["68"]["line_order_signed_internal"]
    finite_signed_total = (
        low_internal + 0.25 * (high_internal - low_internal)) * internal_to_cgs
    finite = {
        "schema": "lumina-cmfgen-lineheat-finite-reference-v1",
        "verdict": "FINITE_REFERENCE_REPRODUCED_MATCHED_STATE_PENDING",
        "physical_mutation": 0, **zero_repairs,
        "shell_zero_velocity_interpolation": {
        "fraction_from_depth_67_to_68": .25,
        "temperature_K": 10000.0, "electron_density_cm3": 5e9,
        "signed_cgs_erg_cm3_s": finite_signed_total,
    }}
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        def run_case(name: str, lumina_case: dict, components_case: dict,
                     finite_case: dict) -> tuple[subprocess.CompletedProcess, Path]:
            case = root / name
            case.mkdir()
            paths = [case / item for item in
                     ("lumina.json", "components.json", "finite.json",
                      "report.json")]
            for path, payload in zip(
                    paths[:3], (lumina_case, components_case, finite_case)):
                path.write_text(json.dumps(payload))
            result = subprocess.run([
                "python3", str(SCRIPT), "--lumina-owner", str(paths[0]),
                "--cmfgen-components", str(paths[1]), "--cmfgen-finite",
                str(paths[2]), "--report", str(paths[3]),
            ], cwd=ROOT, text=True, capture_output=True)
            return result, paths[3]

        result, report_path = run_case(
            "positive", lumina, components, finite)
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        report = json.loads(report_path.read_text())
        bad_temperature = copy.deepcopy(lumina)
        bad_temperature["shells"][0]["temperature_K"] = "10001"
        bad_closure = copy.deepcopy(components)
        bad_closure["depths"]["67"][
            "cellwise_component_closure_verified"] = False
        bad_repair = copy.deepcopy(components)
        bad_repair["repair"] = 1
        bad_normalization = copy.deepcopy(components)
        bad_normalization["depths"]["67"][
            "owners_by_abs_signed_ion_total"][0][
                "normalized_species"] = "Co VI"
        for name, lumina_case, components_case in (
                ("temperature_mismatch", bad_temperature, components),
                ("unsealed_closure", lumina, bad_closure),
                ("repair_marker", lumina, bad_repair),
                ("normalization_drift", lumina, bad_normalization)):
            negative, negative_report = run_case(
                name, lumina_case, components_case, finite)
            if negative.returncode == 0 or negative_report.exists():
                raise SystemExit(f"negative case accepted: {name}")
    row = report["common_ion_components"][0]
    if (report["status"] != "FINITE_COMPONENT_COMPARISON_STATE_UNMATCHED" or
            report["parity_claim"] or report["state"]["matched_state"] or
            row["cmfgen_interpolated"]["signed_rate_erg_cm3_s"] != 2.5 or
            row["cmfgen_interpolated"]["scaled_emission_erg_cm3_s"] != 9.0 or
            row["cmfgen_interpolated"]["scaled_absorption_erg_cm3_s"] != 6.5 or
            report["totals"]["lumina_to_cmfgen"][
                "scaled_emission_ratio"] != 5.0 / 9.0 or
            report["totals"]["lumina_to_cmfgen"][
                "scaled_absorption_ratio"] != 4.0 / 6.5):
        raise SystemExit("component comparison contract mismatch")
    print("PASS a2_10_cmfgen_ion_component_comparison positive+4_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
