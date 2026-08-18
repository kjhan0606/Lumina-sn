#!/usr/bin/env python3
"""Positive and fail-closed controls for the A2-10 targeted gate judge."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_a210_targeted_gate.py"
HASH = "a" * 64


def r6_block(generation: int) -> tuple[str, ...]:
    if generation == 1:
        signed = 27_748_410
        exact_zero = 81_265_890
        raw_negative = mild_negative = srce_chk = 0
        operator = "INIT_SHARED_GAUSSIAN"
    else:
        signed = 22_866_166
        exact_zero = 86_148_134
        raw_negative = 4_246_581
        mild_negative = 4_246_577
        srce_chk = 4
        operator = "CMFGEN_NONOVERLAP_SOBOLEV"
    return (
        f"[cmf_fine][SIGNED-MATERIAL-CENSUS] line_shells={signed} "
        f"exact_zero_tau={exact_zero} raw_negative={raw_negative} "
        f"mild_negative={mild_negative} srce_chk={srce_chk} "
        "raw_preserved=1 floor=0 clamp=0 jitter=0",
        f"[cmf_fine][SIGNED-MATERIAL-POLICY] operator={operator} "
        f"srce_chk_expected={srce_chk} srce_chk_material={srce_chk} "
        "raw_preserved=1 floor=0 clamp=0 jitter=0 repair=0",
        "[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK devices=2/2 "
        "iterations=45 residual=9.6e-9 tolerance=1e-8 component_envelope=1 "
        "refinements=18 cap=64 failure_phase=0 failure_iteration=-1 "
        "floor=0 clamp=0 jitter=0",
        *(() if generation == 1 else (
            "[cmf_fine][SOBOLEV-LINE-OPERATOR] status=PASS "
            "mode=CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0 "
            "continuum_sampling=GAUSSIAN_PROFILE jbar_cells=109014300 "
            "raw_negative=4246581 mild_negative=4246577 "
            "srce_chk_expected=4 srce_chk_applied=4 beta_min=0.1 "
            "beta_max=1.3 all_jbar_finite=1 raw_preserved=1 floor=0 "
            "cap=0 clamp=0 jitter=0 repair=0",
        )),
        f"[R6][LINE-IDENTITY] lane=DET generation={generation} "
        f"q_set_hash={HASH} e_set_hash={HASH} "
        f"domain_hash={HASH} profile_hash={HASH} exact_residual=9.6e-9 "
        "exact_tolerance=1e-8 component_envelope=1 refinements=18",
        f"[R6][LINE-COVERAGE] generation={generation} q_lines=1391131 "
        "e_lines=2180286 "
        "valid_lines=2180286 partial_lines=0 unsampled_lines=0 "
        "valid_cells=109014300 exact_zero_cells=0",
    )


PREDICTOR_LINE = (
    "[A2-INIT][SEED-MATERIAL] event=INIT_SEED_MATERIAL_PREDICTOR lane=DET "
    "r1_generation=1 te_generation=1->1 population_generation=1->2 "
    "te_manifest_preserved=1 te_publication_preserved=1 "
    "floor=0 cap=0 clamp=0 jitter=0 repair=0"
)


def stderr_fixture() -> str:
    return "\n".join((
        *r6_block(1),
        PREDICTOR_LINE,
        *r6_block(2),
        "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=0 "
        "phase=A2-10 te_generation=1->2",
        "[PHYSICS-COMPARISON] lane=DET iter=0 status=COMMITTED dir=/tmp/dump",
        "",
    ))


def run(directory: Path, stderr_text: str) -> tuple[int, dict[str, object]]:
    stdout = directory / "stdout.log"
    stderr = directory / "stderr.log"
    report = directory / "report.json"
    stdout.write_text("targeted gate fixture\n", encoding="utf-8")
    stderr.write_text(stderr_text, encoding="utf-8")
    result = subprocess.run(
        (
            sys.executable, str(CHECKER),
            "--stdout", str(stdout),
            "--stderr", str(stderr),
            "--expected-devices", "2",
            "--expected-refinements", "18",
            "--report", str(report),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-targeted-selftest-") as raw:
        directory = Path(raw)
        positive = stderr_fixture()
        rc, report = run(directory, positive)
        if rc != 0 or report.get("status") != "PASS":
            print("FAIL A2_10_TARGETED_GATE_SELFTEST positive")
            return 4

        single_r6 = "\n".join((
            *r6_block(1),
            PREDICTOR_LINE,
            "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=0 "
            "phase=A2-10 te_generation=1->2",
            "[PHYSICS-COMPARISON] lane=DET iter=0 status=COMMITTED "
            "dir=/tmp/dump",
            "",
        ))
        predictor_after_r2 = "\n".join((
            *r6_block(1),
            *r6_block(2),
            PREDICTOR_LINE,
            "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=0 "
            "phase=A2-10 te_generation=1->2",
            "[PHYSICS-COMPARISON] lane=DET iter=0 status=COMMITTED "
            "dir=/tmp/dump",
            "",
        ))
        negative_controls = {
            "floor": positive.replace("floor=0", "floor=1", 1),
            "cap": positive.replace("cap=0", "cap=1", 1),
            "coverage": positive.replace("valid_lines=2180286", "valid_lines=2180285"),
            "missing_commit": positive.replace(
                "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED", "[R7][PHASE] event=OTHER"
            ),
            "census": positive + "[A2-10][CANCELLATION-CENSUS] phase=LOWER\n",
            "blocked": positive + "[A2-10][BLOCKED] reason=test\n",
            "missing_predictor": positive.replace(PREDICTOR_LINE + "\n", ""),
            "duplicate_predictor": positive.replace(
                PREDICTOR_LINE, PREDICTOR_LINE + "\n" + PREDICTOR_LINE
            ),
            "single_r6_publication": single_r6,
            "predictor_te_transition": positive.replace(
                "te_generation=1->1", "te_generation=1->2", 1
            ),
            "predictor_population_stall": positive.replace(
                "population_generation=1->2", "population_generation=1->1"
            ),
            "predictor_after_r2": predictor_after_r2,
            "generation_not_monotonic": positive.replace(
                "[R6][LINE-COVERAGE] generation=2",
                "[R6][LINE-COVERAGE] generation=3",
            ),
            "r2_negative_census": positive.replace(
                "raw_negative=4246581", "raw_negative=4246580", 1
            ),
            "srce_chk_runtime_mismatch": positive.replace(
                "srce_chk_applied=4", "srce_chk_applied=3", 1
            ),
            "sobolev_nonfinite": positive.replace(
                "beta_max=1.3", "beta_max=nan", 1
            ),
        }
        for name, fixture in negative_controls.items():
            rc, report = run(directory, fixture)
            if rc != 4 or report.get("status") != "FAIL":
                print(f"FAIL A2_10_TARGETED_GATE_SELFTEST negative={name}")
                return 4

    print(
        "PASS A2_10_TARGETED_GATE_SELFTEST positive=1 negative_controls=16 "
        "floor=0 cap=0 clamp=0 jitter=0 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
