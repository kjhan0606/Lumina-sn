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


def iteration_block(iteration: int) -> tuple[str, ...]:
    generation = iteration + 2
    return (
        *r6_block(generation),
        f"[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter={iteration} "
        f"phase=A2-10 te_generation={iteration + 1}->{iteration + 2}",
        f"[PHYSICS-COMPARISON] lane=DET iter={iteration} status=COMMITTED "
        "dir=/tmp/dump",
    )


def relabel_iteration_one(iteration_zero: tuple[str, ...]) -> tuple[str, ...]:
    """Apply the P5 fixture rule: duplicate iter0, relabel only its identities."""
    relabeled = []
    for line in iteration_zero:
        line = line.replace(
            "[R6][LINE-IDENTITY] lane=DET generation=2",
            "[R6][LINE-IDENTITY] lane=DET generation=3",
        )
        line = line.replace(
            "[R6][LINE-COVERAGE] generation=2",
            "[R6][LINE-COVERAGE] generation=3",
        )
        line = line.replace(
            "lane=DET iter=0 phase=A2-10 te_generation=1->2",
            "lane=DET iter=1 phase=A2-10 te_generation=2->3",
        )
        line = line.replace(
            "[PHYSICS-COMPARISON] lane=DET iter=0 status=COMMITTED",
            "[PHYSICS-COMPARISON] lane=DET iter=1 status=COMMITTED",
        )
        relabeled.append(line)
    return tuple(relabeled)


def stderr_fixture() -> str:
    """F1: the in-tree one-outer-iteration normal fixture."""
    return "\n".join((
        *r6_block(1),
        PREDICTOR_LINE,
        *iteration_block(0),
        "",
    ))


def stderr_fixture_two_iterations() -> str:
    """F2: F1 plus a duplicated/relabelled second outer-iteration block."""
    iteration_zero = iteration_block(0)
    return "\n".join((
        *r6_block(1),
        PREDICTOR_LINE,
        *iteration_zero,
        *relabel_iteration_one(iteration_zero),
        "",
    ))


def run(
    directory: Path,
    stderr_text: str,
    expected_outer_iterations: int | None = None,
) -> tuple[int, dict[str, object], bytes]:
    stdout = directory / "stdout.log"
    stderr = directory / "stderr.log"
    report = directory / "report.json"
    stdout.write_text("targeted gate fixture\n", encoding="utf-8")
    stderr.write_text(stderr_text, encoding="utf-8")
    command = [
        sys.executable, str(CHECKER),
        "--stdout", str(stdout),
        "--stderr", str(stderr),
        "--expected-devices", "2",
        "--expected-refinements", "18",
        "--report", str(report),
    ]
    if expected_outer_iterations is not None:
        command.extend((
            "--expected-outer-iterations", str(expected_outer_iterations),
        ))
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report_bytes = report.read_bytes()
    return result.returncode, json.loads(report_bytes), report_bytes


def require_result(
    name: str,
    rc: int,
    report: dict[str, object],
    expected_rc: int,
    expected_status: str,
    expected_error: str | None = None,
) -> bool:
    error = str(report.get("error", ""))
    if (
        rc != expected_rc
        or report.get("status") != expected_status
        or (expected_error is not None and error != expected_error)
    ):
        print(
            f"FAIL A2_10_TARGETED_GATE_SELFTEST case={name} rc={rc} "
            f"status={report.get('status')} reason={error or 'PASS'}"
        )
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-targeted-selftest-") as raw:
        directory = Path(raw)
        positive = stderr_fixture()
        rc, report, _ = run(directory, positive)
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
            rc, report, _ = run(directory, fixture)
            if rc != 4 or report.get("status") != "FAIL":
                print(f"FAIL A2_10_TARGETED_GATE_SELFTEST negative={name}")
                return 4

        # Registered extension (i): the default and explicit N=1 paths must
        # produce byte-identical reports for F1.
        default_rc, default_report, default_bytes = run(directory, positive)
        explicit_rc, explicit_report, explicit_bytes = run(
            directory, positive, expected_outer_iterations=1
        )
        if not require_result("i-default", default_rc, default_report, 0, "PASS"):
            return 4
        if not require_result("i-explicit-n1", explicit_rc, explicit_report, 0, "PASS"):
            return 4
        if default_bytes != explicit_bytes:
            print(
                "FAIL A2_10_TARGETED_GATE_SELFTEST case=i "
                "reason=report-byte-mismatch"
            )
            return 4
        print(
            "PASS A2_10_TARGETED_GATE_EXTENDED case=i "
            "default_rc=0 explicit_n1_rc=0 reason=report-byte-identical"
        )

        # Registered extension (ii): F1 cannot satisfy N=2.
        f1_n2_error = (
            "expected exactly 3 '[cmf_fine][EXACT-MULTIGPU-EPOCH]' lines, found 2"
        )
        rc, report, _ = run(directory, positive, expected_outer_iterations=2)
        if not require_result("ii-f1-n2", rc, report, 4, "FAIL", f1_n2_error):
            return 4
        print(
            f"PASS A2_10_TARGETED_GATE_EXTENDED case=ii rc={rc} "
            f"reason={report['error']}"
        )

        # Registered extension (iii): F2 is coupled symmetrically to N=2.
        two_iterations = stderr_fixture_two_iterations()
        n2_rc, n2_report, _ = run(
            directory, two_iterations, expected_outer_iterations=2
        )
        default_rc, default_report, _ = run(directory, two_iterations)
        f2_default_error = (
            "expected exactly 2 '[cmf_fine][EXACT-MULTIGPU-EPOCH]' lines, found 3"
        )
        if not require_result("iii-f2-n2", n2_rc, n2_report, 0, "PASS"):
            return 4
        if not require_result(
            "iii-f2-default", default_rc, default_report, 4, "FAIL",
            f2_default_error,
        ):
            return 4
        print(
            "PASS A2_10_TARGETED_GATE_EXTENDED case=iii "
            f"n2_rc={n2_rc} default_rc={default_rc} "
            f"reason_n2=PASS reason_default={default_report['error']}"
        )

        # Registered extension (iv): only the iter1 region is contaminated;
        # the whole-log repair audit must still reject it.
        iter1_marker = (
            "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1 "
        )
        contaminated = two_iterations.replace(
            iter1_marker, iter1_marker + "repair=1 ", 1
        )
        if contaminated.count("repair=1") != 1:
            print(
                "FAIL A2_10_TARGETED_GATE_SELFTEST case=iv "
                "reason=iter1-repair-injection-count"
            )
            return 4
        repair_error = "nonzero numerical repair field repair=1"
        rc, report, _ = run(
            directory, contaminated, expected_outer_iterations=2
        )
        if not require_result("iv-iter1-repair", rc, report, 4, "FAIL", repair_error):
            return 4
        print(
            f"PASS A2_10_TARGETED_GATE_EXTENDED case=iv rc={rc} "
            f"reason={report['error']}"
        )

    print(
        "PASS A2_10_TARGETED_GATE_SELFTEST positive=1 negative_controls=16 "
        "extended_controls=4 "
        "floor=0 cap=0 clamp=0 jitter=0 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
