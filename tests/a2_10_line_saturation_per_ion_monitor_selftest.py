#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MONITOR = ROOT / "scripts/monitor_a210_line_saturation_per_ion_coverage_v4.sh"
OWNER = ROOT / "scripts/summarize_a210_line_ion_owners.py"
COVERAGE = ROOT / "scripts/check_a210_line_saturation_per_ion_coverage.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owner_line(z: int, slot: int, signed: str, uncertainty: str,
               emission: str, absorption: str) -> str:
    return (
        "[A2-10][LINE-ION-OWNER] phase=REQUESTED_TE shell=0 "
        "T_e_K=19059.411196903675 n_e_cm3=5e9 "
        f"ion_slot={slot} Z={z} ion_stage=3 ion_label=4 "
        f"signed_rate={signed} absolute_signed_sum={signed} "
        f"uncertainty={uncertainty} scaled_emission={emission} "
        f"scaled_absorption={absorption} eligible_cells=1 "
        "cooling_cells=1 heating_cells=0 exact_zero_cells=0 srce_chk_cells=0 "
        "complete=1 interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        "clamp=0 floor=0 jitter=0 repair=0"
    )


def source_log(path: Path) -> None:
    lines = [
        owner_line(26, 1, "1", ".1", "10", "9"),
        owner_line(27, 2, "2", ".2", "20", "18"),
        owner_line(28, 3, "3", ".3", "30", "27"),
        (
            "[A2-10][LINE-ION-OWNER-SUMMARY] phase=REQUESTED_TE shell=0 "
            "T_e_K=19059.411196903675 n_e_cm3=5e9 ion_records=3 "
            "eligible_cells=3 line_order_signed_rate=6 grouped_signed_rate=6 "
            "signed_grouping_delta=0 line_order_absolute_sum=6 "
            "grouped_absolute_sum=6 absolute_grouping_delta=0 "
            "line_order_uncertainty=.6 grouped_uncertainty=.6 "
            "uncertainty_grouping_delta=0 line_order_emission=60 "
            "grouped_emission=60 emission_grouping_delta=0 "
            "line_order_absorption=54 grouped_absorption=54 "
            "absorption_grouping_delta=0 complete=1 interpretation=DIAGNOSTIC_ONLY "
            "physical_values_modified=0 clamp=0 floor=0 jitter=0 repair=0"
        ),
        (
            "[A2-10][VECTOR-INTERIOR-SCAN] phase=REQUESTED_TE valid=1 "
            "endpoint_no_bracket=2 interior_bracket=0 still_same_sign=2 "
            "action=DIAGNOSTIC_ONLY solver_result=RADEQ_NO_BRACKET"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def saturation(path: Path, log: Path, selected: dict[int, float]) -> None:
    rows = []
    total_selected = 0.0
    for rank, z in enumerate((26, 27, 28), 1):
        total_selected += selected[z]
        rows.append({
            "rank": rank, "line": 100 + rank, "Z": z, "ion": 3,
            "scaled_emission": selected[z],
            "scaled_emission_serialized": str(selected[z]),
        })
    path.write_text(json.dumps({
        "schema": "lumina-a210-line-saturation-summary-v1",
        "status": "PASS",
        "source_log": str(log.resolve()),
        "source_log_sha256": sha(log),
        "summary": {
            "candidate_rows": 3,
            "selected_rows": 3,
            "total_scaled_emission": 60.0,
            "total_scaled_emission_serialized": "60",
            "selected_scaled_emission": total_selected,
            "selected_scaled_emission_serialized": str(total_selected),
        },
        "rows": rows,
        "physical_values_modified": False,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
    }) + "\n", encoding="utf-8")


def fixture(root: Path, selected: dict[int, float],
            bad_v3: bool = False) -> tuple[Path, Path]:
    run = root / "run"
    control = run / "manual_control"
    bundle = run / "postprocess_per_ion_coverage_v4"
    control.mkdir(parents=True)
    bundle.mkdir()
    log = run / "stderr.log"
    source_log(log)
    summary = run / "a210_line_saturation_summary_v2.json"
    saturation(summary, log, selected)
    summary_sha = sha(summary)

    (run / "LINE_SATURATION_VERDICT_V2.txt").write_text("\n".join([
        "status=PASS", f"summary_sha256={summary_sha}", "model_rc=1",
        "natural_result=RADEQ_NO_BRACKET", "physical_mutation=0",
        "floor=0", "cap=0", "clamp=0", "jitter=0", "repair=0", "",
    ]), encoding="utf-8")
    v3_lines = [
        "status=PASS", f"v2_summary_sha256={summary_sha}",
        "execution_order=AFTER_V2_PASS",
        "arithmetic_bound_is_physical_tolerance=0", "physical_mutation=0",
        "floor=0", "cap=0", "clamp=0", "jitter=0", "repair=0",
    ]
    if bad_v3:
        v3_lines.remove("repair=0")
    (run / "LINE_SATURATION_ROUNDOFF_VERDICT_V3.txt").write_text(
        "\n".join(v3_lines) + "\n", encoding="utf-8"
    )

    for source, name in (
        (OWNER, "summarize_a210_line_ion_owners.py"),
        (COVERAGE, "check_a210_line_saturation_per_ion_coverage.py"),
        (MONITOR, "monitor_a210_line_saturation_per_ion_coverage_v4.sh"),
    ):
        shutil.copy2(source, bundle / name)
    (bundle / "requested_diag_te_K.txt").write_text(
        "19059.411196903675\n", encoding="utf-8"
    )
    (bundle / "line_ion_owner_shells.txt").write_text("1\n", encoding="utf-8")
    (bundle / "PER_ION_COVERAGE_CONTRACT.txt").write_text(
        "fixture=1\n", encoding="utf-8"
    )
    (bundle / "READY").write_text("READY\n", encoding="utf-8")
    members = [
        "summarize_a210_line_ion_owners.py",
        "check_a210_line_saturation_per_ion_coverage.py",
        "monitor_a210_line_saturation_per_ion_coverage_v4.sh",
        "requested_diag_te_K.txt", "line_ion_owner_shells.txt",
        "PER_ION_COVERAGE_CONTRACT.txt", "READY",
    ]
    (bundle / "POSTPROCESS_MANIFEST.sha256").write_text(
        "".join(f"{sha(bundle / name)}  {name}\n" for name in members),
        encoding="utf-8",
    )
    return run, bundle


def execute(run: Path, bundle: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(bundle / "monitor_a210_line_saturation_per_ion_coverage_v4.sh"),
         str(run), str(bundle)],
        cwd=ROOT, text=True, capture_output=True, timeout=20,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)

        passed = base / "passed"
        passed.mkdir()
        run, bundle = fixture(passed, {26: 9.0, 27: 18.0, 28: 27.0})
        result = execute(run, bundle)
        if result.returncode != 0 or not (run /
                "LINE_SATURATION_PER_ION_COVERAGE_VERDICT_V4.txt").read_text(
                    encoding="utf-8").startswith("status=PASS\n"):
            raise SystemExit(f"monitor PASS fixture failed: {result.stderr}")

        under = base / "under"
        under.mkdir()
        run, bundle = fixture(under, {26: 8.0, 27: 20.0, 28: 30.0})
        result = execute(run, bundle)
        verdict = (run / "LINE_SATURATION_PER_ION_COVERAGE_VERDICT_V4.txt").read_text(
            encoding="utf-8"
        )
        if result.returncode != 0 or not verdict.startswith("status=UNDERCOVERED\n") or \
                "rerun_with_per_ion_union_required=1" not in verdict:
            raise SystemExit("monitor undercoverage outcome was not preserved")

        drift = base / "drift"
        drift.mkdir()
        run, bundle = fixture(drift, {26: 9.0, 27: 18.0, 28: 27.0})
        with (bundle / "check_a210_line_saturation_per_ion_coverage.py").open(
                "a", encoding="utf-8") as stream:
            stream.write("# drift\n")
        if execute(run, bundle).returncode != 4 or "POSTPROCESS_BUNDLE_SHA_DRIFT" not in (
                run / "manual_control/line_saturation_per_ion_coverage_v4.log"
        ).read_text(encoding="utf-8"):
            raise SystemExit("monitor accepted bundle SHA drift")

        bad = base / "bad_v3"
        bad.mkdir()
        run, bundle = fixture(
            bad, {26: 9.0, 27: 18.0, 28: 27.0}, bad_v3=True
        )
        if execute(run, bundle).returncode != 4 or "V3_VERDICT_CONTRACT_MISMATCH" not in (
                run / "manual_control/line_saturation_per_ion_coverage_v4.log"
        ).read_text(encoding="utf-8"):
            raise SystemExit("monitor accepted incomplete V3 verdict")

    print("PASS a2_10_line_saturation_per_ion_monitor positive+undercoverage+2_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
