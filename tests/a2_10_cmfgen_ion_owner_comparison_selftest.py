#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/compare_a210_cmfgen_ion_owners.py"


def main() -> int:
    zero_repairs = {
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
    }
    lumina = {
        "schema": "a210-line-ion-owner-diagnostic-v1",
        "status": "PASS", "complete": True, "phase": "REQUESTED_TE",
        "physical_values_modified": False, **zero_repairs,
        "shells": [{
            "shell": 0, "temperature_K": "10000",
            "electron_density_cm3": "2e9", "line_order_signed_rate": "0.5",
            "owners_by_abs_signed_ion_total": [
                {"Z": "27", "ion_label": "4", "signed_rate": "1",
                 "absolute_signed_sum": "3"},
                {"Z": "26", "ion_label": "6", "signed_rate": "-.5",
                 "absolute_signed_sum": "2"},
            ],
        }],
    }

    def depth(co: float, fe: float) -> dict:
        return {
            "line_order_signed_cgs_erg_cm3_s": co + fe,
            "finite_reference_check": {
                "signed_bit_exact": True, "absolute_bit_exact": True},
            "top_by_abs_signed_ion_total": [
                {"normalized_species": "Co IV", "cmfgen_label": "CoIV",
                 "signed_cgs_erg_cm3_s": co,
                 "absolute_cgs_erg_cm3_s": abs(co),
                 "cancellation_condition": 10},
                {"normalized_species": "Fe VI", "cmfgen_label": "FeSIX",
                 "signed_cgs_erg_cm3_s": fe,
                 "absolute_cgs_erg_cm3_s": abs(fe),
                 "cancellation_condition": 20},
            ],
        }

    cmfgen = {
        "schema": "cmfgen-lineheat-ion-owner-summary-v1",
        "verdict": "COMPLETE_DIAGNOSTIC_OWNER_DECOMPOSITION",
        "physical_mutation": 0, **zero_repairs,
        "depths": {"67": depth(2, -1), "68": depth(4, -3)},
    }
    finite = {
        "schema": "lumina-cmfgen-lineheat-finite-reference-v1",
        "verdict": "FINITE_REFERENCE_REPRODUCED_MATCHED_STATE_PENDING",
        "physical_mutation": 0, **zero_repairs,
        "shell_zero_velocity_interpolation": {
            "fraction_from_depth_67_to_68": 0.25,
            "temperature_K": 10000.0,
            "electron_density_cm3": 5e9,
            "cmfgen_to_lumina_mass_density_ratio": 1.001,
            "signed_cgs_erg_cm3_s": 1.0,
        },
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def run_case(name: str, lumina_case: dict, cmfgen_case: dict,
                     finite_case: dict) -> tuple[subprocess.CompletedProcess, Path]:
            case = root / name
            case.mkdir()
            paths = [case / item for item in
                     ("lumina.json", "cmfgen.json", "finite.json", "report.json")]
            for path, payload in zip(paths[:3],
                                     (lumina_case, cmfgen_case, finite_case)):
                path.write_text(json.dumps(payload))
            result = subprocess.run([
                "python3", str(SCRIPT), "--lumina-owner", str(paths[0]),
                "--cmfgen-owner", str(paths[1]), "--cmfgen-finite",
                str(paths[2]), "--report", str(paths[3]),
            ], cwd=ROOT, text=True, capture_output=True)
            return result, paths[3]

        result, report_path = run_case("positive", lumina, cmfgen, finite)
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        report = json.loads(report_path.read_text())

        bad_normalization = copy.deepcopy(cmfgen)
        bad_normalization["depths"]["67"]["top_by_abs_signed_ion_total"][1][
            "normalized_species"] = "FeS IX"
        bad_unknown = copy.deepcopy(cmfgen)
        bad_unknown["depths"]["67"]["top_by_abs_signed_ion_total"][1][
            "cmfgen_label"] = "FeSIXjunk"
        bad_temperature = copy.deepcopy(lumina)
        bad_temperature["shells"][0]["temperature_K"] = "10001"
        bad_repair = copy.deepcopy(cmfgen)
        bad_repair["repair"] = 1
        for name, lumina_case, cmfgen_case in (
                ("normalization_drift", lumina, bad_normalization),
                ("unknown_raw_label", lumina, bad_unknown),
                ("temperature_mismatch", bad_temperature, cmfgen),
                ("repair_marker", lumina, bad_repair)):
            negative, negative_report = run_case(
                name, lumina_case, cmfgen_case, finite)
            if negative.returncode == 0 or negative_report.exists():
                raise SystemExit(f"negative case accepted: {name}")

    if (report["status"] != "FINITE_COMPARISON_STATE_UNMATCHED" or
            report["common_owner_count"] != 2 or report["parity_claim"] or
            not report["state"]["temperature_exact_match"] or
            report["state"]["matched_state"]):
        raise SystemExit("comparison contract mismatch")
    owners = {row["normalized_species"]: row
              for row in report["common_ion_owners"]}
    if (owners["Co IV"]["cmfgen_interpolated_signed_rate_erg_cm3_s"] != 2.5 or
            owners["Fe VI"]["cmfgen_interpolated_signed_rate_erg_cm3_s"] != -1.5):
        raise SystemExit("owner interpolation mismatch")
    print("PASS a2_10_cmfgen_ion_owner_comparison positive+4_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
