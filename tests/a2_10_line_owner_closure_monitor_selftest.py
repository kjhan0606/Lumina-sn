#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MONITOR = ROOT / "scripts/monitor_a210_line_owner_closure.sh"
COMPARATOR = ROOT / "scripts/compare_a210_targeted_reference.py"
SUMMARIZER = ROOT / "scripts/summarize_a210_line_ion_owners.py"
OWNER_COMPARE = ROOT / "scripts/compare_a210_cmfgen_ion_owners.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proof_fixture(refinements: int, component_upper: str,
                  profile_upper: str) -> str:
    return "\n".join((
        "[cmf_fine][SIGNED-MATERIAL-CENSUS] line_shells=10 raw_negative=0 "
        "mild_negative=0 raw_preserved=1 floor=0 clamp=0 jitter=0 repair=0",
        "[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK devices=2/2 "
        "iterations=45 residual=9e-9 tolerance=1e-8 "
        f"refinements={refinements} component_error=[1e-9,{component_upper}] "
        "floor=0 clamp=0 jitter=0 repair=0 domain_hash=abc",
        "[R6][LINE-IDENTITY] lane=DET generation=1 q_lines=3 e_lines=4 "
        "domain_hash=abc exact_residual=9e-9 "
        f"refinements={refinements} component_error=[1e-9,{component_upper}] "
        f"profile_error=[1e-9,{profile_upper}] repair=0",
        "[R6][LINE-COVERAGE] generation=1 q_lines=3 e_lines=4 valid_lines=4 "
        "partial_lines=0 unsampled_lines=0 valid_cells=200 exact_zero_cells=0 "
        "repair=0",
    ))


def owner_fixture() -> str:
    common = (
        "phase=REQUESTED_TE shell=0 T_e_K=10000 n_e_cm3=2e9 "
        "complete=1 interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        "clamp=0 floor=0 jitter=0 repair=0"
    )
    return "\n".join((
        "[A2-10][LINE-ION-OWNER] " + common + " ion_slot=1 Z=27 "
        "ion_stage=3 ion_label=4 signed_rate=1 absolute_signed_sum=1 "
        "uncertainty=.01 scaled_emission=2 scaled_absorption=1 "
        "eligible_cells=6 cooling_cells=6 heating_cells=0 exact_zero_cells=0 "
        "srce_chk_cells=0",
        "[A2-10][LINE-ION-OWNER] " + common + " ion_slot=2 Z=26 "
        "ion_stage=2 ion_label=3 signed_rate=-.5 absolute_signed_sum=.5 "
        "uncertainty=.02 scaled_emission=3 scaled_absorption=2 "
        "eligible_cells=4 cooling_cells=0 heating_cells=4 exact_zero_cells=0 "
        "srce_chk_cells=0",
        "[A2-10][LINE-ION-OWNER-SUMMARY] " + common + " ion_records=2 "
        "eligible_cells=10 line_order_signed_rate=.5 grouped_signed_rate=.5 "
        "signed_grouping_delta=0 line_order_absolute_sum=1.5 "
        "grouped_absolute_sum=1.5 absolute_grouping_delta=0 "
        "line_order_uncertainty=.03 grouped_uncertainty=.03 "
        "uncertainty_grouping_delta=0 line_order_emission=5 "
        "grouped_emission=5 emission_grouping_delta=0 line_order_absorption=3 "
        "grouped_absorption=3 absorption_grouping_delta=0",
        "[A2-10][VECTOR-INTERIOR-SCAN] phase=REQUESTED_TE valid=1 "
        "endpoint_no_bracket=1 interior_bracket=0 still_same_sign=1 "
        "action=DIAGNOSTIC_ONLY solver_result=RADEQ_NO_BRACKET",
    ))


def main() -> int:
    gpfs_parent = Path("/gpfs/kjhan/lumina")
    with tempfile.TemporaryDirectory(
            prefix="a210-owner-monitor-selftest-", dir=gpfs_parent) as raw:
        run = Path(raw)
        control = run / "manual_control"
        control.mkdir()
        (control / "FAILED").write_text("1\n")
        (control / "supervisor.log").write_text(
            "MANUAL_DET_TRIPWIRE status=CHILD_EXITED child_rc=70\n")
        (run / "model.rc").write_text("1\n")
        (run / "stdout.log").write_text("physical_values_modified=0 repair=0\n")
        reference = run / "reference.log"
        reference.write_text(proof_fixture(30, "2e-9", "4e-9") + "\n")
        (run / "stderr.log").write_text(
            proof_fixture(36, "1e-9", "2e-9") + "\n" +
            owner_fixture() + "\n")

        cmfgen_owner = run / "cmfgen_owner.json"
        cmfgen_owner.write_text(json.dumps({
            "schema": "cmfgen-lineheat-ion-owner-summary-v1",
            "verdict": "COMPLETE_DIAGNOSTIC_OWNER_DECOMPOSITION",
            "physical_mutation": 0,
            "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
            "depths": {
            "67": {
                "line_order_signed_cgs_erg_cm3_s": 1.0,
                "finite_reference_check": {
                    "signed_bit_exact": True, "absolute_bit_exact": True},
                "top_by_abs_signed_ion_total": [
                {"normalized_species": "Co IV", "cmfgen_label": "CoIV",
                 "signed_cgs_erg_cm3_s": 2.0, "cancellation_condition": 10},
                {"normalized_species": "Fe III", "cmfgen_label": "FeIII",
                 "signed_cgs_erg_cm3_s": -1.0, "cancellation_condition": 20}]},
            "68": {
                "line_order_signed_cgs_erg_cm3_s": 1.0,
                "finite_reference_check": {
                    "signed_bit_exact": True, "absolute_bit_exact": True},
                "top_by_abs_signed_ion_total": [
                {"normalized_species": "Co IV", "cmfgen_label": "CoIV",
                 "signed_cgs_erg_cm3_s": 4.0, "cancellation_condition": 11},
                {"normalized_species": "Fe III", "cmfgen_label": "FeIII",
                 "signed_cgs_erg_cm3_s": -3.0, "cancellation_condition": 21}]},
        }}))
        cmfgen_finite = run / "cmfgen_finite.json"
        cmfgen_finite.write_text(json.dumps({
            "schema": "lumina-cmfgen-lineheat-finite-reference-v1",
            "verdict": "FINITE_REFERENCE_REPRODUCED_MATCHED_STATE_PENDING",
            "physical_mutation": 0,
            "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
            "shell_zero_velocity_interpolation": {
                "fraction_from_depth_67_to_68": .25,
                "temperature_K": 10000.0,
                "electron_density_cm3": 5e9,
                "cmfgen_to_lumina_mass_density_ratio": 1.001,
                "signed_cgs_erg_cm3_s": 1.0,
            }}))

        command = [
            str(MONITOR), str(run), str(reference), sha(reference), "10000", "1",
            sha(COMPARATOR), sha(SUMMARIZER), sha(OWNER_COMPARE),
            str(cmfgen_owner), sha(cmfgen_owner),
            str(cmfgen_finite), sha(cmfgen_finite),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True,
                                capture_output=True, check=False)
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr +
                             (control / "line_owner_closure_monitor.log").read_text())
        comparison = json.loads(
            (run / "a210_cmfgen_ion_owner_comparison.json").read_text())
        if comparison["status"] != "FINITE_COMPARISON_STATE_UNMATCHED" or \
                comparison["parity_claim"] or comparison["common_owner_count"] != 2:
            raise SystemExit("positive closure comparison mismatch")

        # Independent hash-drift control: remove the monitor's completed log
        # and use a distinct run root so no active-marker state is reused.
        negative = run / "negative"
        negative_control = negative / "manual_control"
        negative_control.mkdir(parents=True)
        (negative_control / "FAILED").write_text("1\n")
        (negative_control / "supervisor.log").write_text(
            "MANUAL_DET_TRIPWIRE status=CHILD_EXITED child_rc=70\n")
        for name in ("model.rc", "stdout.log", "stderr.log"):
            (negative / name).write_bytes((run / name).read_bytes())
        bad = list(command)
        bad[1] = str(negative)
        bad[3] = "0" * 64
        result = subprocess.run(bad, cwd=ROOT, text=True,
                                capture_output=True, check=False)
        if result.returncode != 4 or \
                "REFERENCE_STDERR_SHA_MISMATCH" not in \
                (negative_control / "line_owner_closure_monitor.log").read_text():
            raise SystemExit("hash-drift control accepted")

    print("PASS a2_10_line_owner_closure_monitor positive+sha_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
