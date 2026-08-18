#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MONITOR = ROOT / "scripts/monitor_a210_line_owner_components.sh"
COMPARATOR = ROOT / "scripts/compare_a210_cmfgen_ion_components.py"
OWNER_DEP = ROOT / "scripts/compare_a210_cmfgen_ion_owners.py"
RE_CGS = 4.0 * math.pi * 1.0e-10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    zero = {"floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0}
    lumina = {
        "status": "PASS", "complete": True,
        "physical_values_modified": False, **zero,
        "shells": [{
            "shell": 0, "temperature_K": "10000",
            "electron_density_cm3": "2e9",
            "line_order_signed_rate": "1",
            "line_order_emission": "5",
            "line_order_absorption": "4",
            "owners_by_abs_signed_ion_total": [{
                "Z": "27", "ion_label": "4", "signed_rate": "1",
                "scaled_emission": "5", "scaled_absorption": "4",
            }],
        }],
    }

    def depth(net: float, emission: float, absorption: float) -> dict:
        return {
            "cellwise_component_closure_verified": True,
            "line_order_signed_internal": net,
            "line_order_scaled_emission_internal": emission,
            "line_order_scaled_absorption_internal": absorption,
            "finite_reference_check": {
                "signed_bit_exact": True, "absolute_bit_exact": True},
            "owners_by_abs_signed_ion_total": [{
                "cmfgen_label": "CoIV",
                "normalized_species": "Co IV",
                "signed_cgs_erg_cm3_s": net,
                "scaled_emission_cgs_erg_cm3_s": emission,
                "scaled_absorption_cgs_erg_cm3_s": absorption,
            }],
        }

    components = {
        "schema": "cmfgen-line-components-ion-owner-v1",
        "verdict": "COMPLETE_FINITE_COMPONENT_DECOMPOSITION",
        "physical_values_modified": 0, **zero,
        "depths": {"67": depth(2, 8, 6), "68": depth(4, 12, 8)},
    }
    finite = {
        "schema": "lumina-cmfgen-lineheat-finite-reference-v1",
        "verdict": "FINITE_REFERENCE_REPRODUCED_MATCHED_STATE_PENDING",
        "physical_mutation": 0, **zero,
        "shell_zero_velocity_interpolation": {
            "fraction_from_depth_67_to_68": .25,
            "temperature_K": 10000.0,
            "electron_density_cm3": 5e9,
            "signed_cgs_erg_cm3_s": 2.5 * RE_CGS,
        },
    }

    gpfs_parent = Path("/gpfs/kjhan/lumina")
    with tempfile.TemporaryDirectory(
            prefix="a210-owner-component-monitor-selftest-",
            dir=gpfs_parent) as raw:
        run = Path(raw)
        control = run / "manual_control"
        control.mkdir()
        (control / "line_owner_closure_monitor.log").write_text(
            "LINE_OWNER_CLOSURE_MONITOR status=PASS "
            "comparison=FINITE_COMPARISON_STATE_UNMATCHED\n")
        (run / "a210_line_ion_owner_report_strict.json").write_text(
            json.dumps(lumina))
        component_path = run / "components.json"
        finite_path = run / "finite.json"
        component_path.write_text(json.dumps(components))
        finite_path.write_text(json.dumps(finite))
        command = [
            str(MONITOR), str(run), sha(COMPARATOR), sha(OWNER_DEP),
            str(component_path), sha(component_path),
            str(finite_path), sha(finite_path),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True,
                                capture_output=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                result.stdout + result.stderr +
                (control / "line_owner_component_monitor.log").read_text())
        report = json.loads(
            (run / "a210_cmfgen_ion_component_comparison.json").read_text())
        if (report["status"] !=
                "FINITE_COMPONENT_COMPARISON_STATE_UNMATCHED" or
                report["parity_claim"] or report["common_owner_count"] != 1):
            raise SystemExit("positive component monitor mismatch")

        negative = run / "negative"
        negative_control = negative / "manual_control"
        negative_control.mkdir(parents=True)
        (negative_control / "line_owner_closure_monitor.log").write_text(
            "LINE_OWNER_CLOSURE_MONITOR status=PASS "
            "comparison=FINITE_COMPARISON_STATE_UNMATCHED\n")
        (negative / "a210_line_ion_owner_report_strict.json").write_bytes(
            (run / "a210_line_ion_owner_report_strict.json").read_bytes())
        bad = list(command)
        bad[1] = str(negative)
        bad[2] = "0" * 64
        result = subprocess.run(bad, cwd=ROOT, text=True,
                                capture_output=True, check=False)
        if (result.returncode != 4 or
                "COMPONENT_COMPARATOR_SHA_MISMATCH" not in
                (negative_control /
                 "line_owner_component_monitor.log").read_text()):
            raise SystemExit("component-monitor SHA drift accepted")

    print("PASS a2_10_line_owner_component_monitor positive+sha_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
