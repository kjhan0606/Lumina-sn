#!/usr/bin/env python3
"""Controls for the targeted-gate/census reference comparison."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/compare_a210_targeted_reference.py"


def fixture(
    refinements: int = 18,
    component_upper: str = "2e-9",
    profile_upper: str = "4e-9",
    residual: str = "9e-9",
) -> str:
    return "\n".join((
        "[cmf_fine][SIGNED-MATERIAL-CENSUS] line_shells=10 raw_negative=0 "
        "mild_negative=0 raw_preserved=1 floor=0 clamp=0 jitter=0",
        "[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK devices=2/2 iterations=45 "
        f"residual={residual} tolerance=1e-8 refinements={refinements} "
        f"component_error=[1e-9,{component_upper}] "
        "floor=0 clamp=0 jitter=0 domain_hash=abc",
        "[R6][LINE-IDENTITY] lane=DET generation=1 q_lines=3 e_lines=4 "
        f"domain_hash=abc exact_residual={residual} refinements={refinements} "
        f"component_error=[1e-9,{component_upper}] profile_error=[1e-9,{profile_upper}]",
        "[R6][LINE-COVERAGE] generation=1 q_lines=3 e_lines=4 valid_lines=4 "
        "partial_lines=0 unsampled_lines=0 valid_cells=200 exact_zero_cells=0",
        "",
    ))


def run(
    directory: Path,
    candidate_text: str,
    candidate_occurrence: int = 0,
    proof_refinements: tuple[int, int] | None = None,
) -> tuple[int, dict[str, object]]:
    reference = directory / "reference.log"
    candidate = directory / "candidate.log"
    report = directory / "report.json"
    reference.write_text(fixture(), encoding="utf-8")
    candidate.write_text(candidate_text, encoding="utf-8")
    command = [
            sys.executable, str(CHECKER),
            "--reference-stderr", str(reference),
            "--candidate-stderr", str(candidate),
            "--candidate-occurrence", str(candidate_occurrence),
            "--report", str(report),
    ]
    if proof_refinements is not None:
        command.extend((
            "--reference-refinements", str(proof_refinements[0]),
            "--candidate-refinements", str(proof_refinements[1]),
        ))
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-target-reference-") as raw:
        directory = Path(raw)
        rc, report = run(directory, fixture())
        if rc != 0 or report.get("status") != "PASS":
            print("FAIL A2_10_TARGETED_REFERENCE_SELFTEST positive")
            return 4
        rc, report = run(directory, fixture() + fixture(), 1)
        if rc != 0 or report.get("status") != "PASS":
            print("FAIL A2_10_TARGETED_REFERENCE_SELFTEST occurrence")
            return 4
        proof_candidate = fixture(
            refinements=24, component_upper="1e-9", profile_upper="2e-9"
        )
        rc, report = run(directory, proof_candidate, proof_refinements=(18, 24))
        if (rc != 0 or report.get("status") != "PASS" or
                report.get("comparison_mode") != "PROOF_REFINEMENT_ONLY" or
                not report.get("proof_bounds_nonincreasing")):
            print("FAIL A2_10_TARGETED_REFERENCE_SELFTEST proof_positive")
            return 4
        controls = {
            "envelope": fixture().replace("[1e-9,2e-9]", "[1e-9,3e-9]"),
            "coverage": fixture().replace("valid_cells=200", "valid_cells=199"),
            "repair": fixture().replace("floor=0", "floor=1", 1),
            "missing_record": fixture().replace("[R6][LINE-COVERAGE]", "[R6][OTHER]"),
            "missing_occurrence": fixture(),
        }
        for name, text in controls.items():
            rc, report = run(
                directory, text, 1 if name == "missing_occurrence" else 0
            )
            if rc != 4 or report.get("status") != "FAIL":
                print(f"FAIL A2_10_TARGETED_REFERENCE_SELFTEST negative={name}")
                return 4
        proof_controls = {
            "proof_worse_bound": fixture(
                refinements=24, component_upper="3e-9", profile_upper="2e-10"
            ),
            "proof_physical_drift": fixture(
                refinements=24, component_upper="1e-9", profile_upper="2e-9",
                residual="8e-9",
            ),
            "proof_wrong_refinement": fixture(
                refinements=23, component_upper="1e-9", profile_upper="2e-9"
            ),
        }
        for name, text in proof_controls.items():
            rc, report = run(directory, text, proof_refinements=(18, 24))
            if rc != 4 or report.get("status") != "FAIL":
                print(f"FAIL A2_10_TARGETED_REFERENCE_SELFTEST negative={name}")
                return 4
    print(
        "PASS A2_10_TARGETED_REFERENCE_SELFTEST positive=1 proof_positive=1 "
        "occurrence_selection=1 negative_controls=8 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
