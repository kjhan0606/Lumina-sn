#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/summarize_a210_line_ion_owners.py"


def owner(shell: int, ip: int, signed: str, absolute: str, uncertainty: str,
          emission: str, absorption: str, cells: int, repair: int = 0) -> str:
    return (
        "[A2-10][LINE-ION-OWNER] phase=REQUESTED_TE "
        f"shell={shell} T_e_K=19059.411196903675 n_e_cm3=5e9 "
        f"ion_slot={ip} Z={26 + ip} ion_stage=3 ion_label=4 "
        f"signed_rate={signed} absolute_signed_sum={absolute} "
        f"uncertainty={uncertainty} scaled_emission={emission} "
        f"scaled_absorption={absorption} eligible_cells={cells} "
        "cooling_cells=1 heating_cells=1 exact_zero_cells=0 srce_chk_cells=0 "
        "complete=1 interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        f"clamp=0 floor=0 jitter=0 repair={repair}"
    )


def summary(shell: int, records: int, cells: int, signed: str, absolute: str,
            uncertainty: str, emission: str, absorption: str) -> str:
    return (
        "[A2-10][LINE-ION-OWNER-SUMMARY] phase=REQUESTED_TE "
        f"shell={shell} T_e_K=19059.411196903675 n_e_cm3=5e9 "
        f"ion_records={records} eligible_cells={cells} "
        f"line_order_signed_rate={signed} grouped_signed_rate={signed} "
        "signed_grouping_delta=0 "
        f"line_order_absolute_sum={absolute} grouped_absolute_sum={absolute} "
        "absolute_grouping_delta=0 "
        f"line_order_uncertainty={uncertainty} grouped_uncertainty={uncertainty} "
        "uncertainty_grouping_delta=0 "
        f"line_order_emission={emission} grouped_emission={emission} "
        "emission_grouping_delta=0 "
        f"line_order_absorption={absorption} grouped_absorption={absorption} "
        "absorption_grouping_delta=0 complete=1 interpretation=DIAGNOSTIC_ONLY "
        "physical_values_modified=0 clamp=0 floor=0 jitter=0 repair=0"
    )


def run(log: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), "--log", str(log), "--report", str(report),
         "--phase", "REQUESTED_TE", "--expected-shells", "2",
         "--expected-temperature-K", "19059.411196903675"],
        cwd=ROOT, text=True, capture_output=True,
    )


def main() -> int:
    lines = [
        owner(0, 1, "1", "1", ".01", "2", "1.25", 5),
        owner(0, 2, "-.25", ".25", ".02", "3", "2.75", 3),
        summary(0, 2, 8, ".75", "1.25", ".03", "5", "4"),
        owner(1, 3, ".5", ".5", ".04", "1.5", "1", 4),
        summary(1, 1, 4, ".5", ".5", ".04", "1.5", "1"),
        "[A2-10][VECTOR-INTERIOR-SCAN] phase=REQUESTED_TE valid=1 "
        "endpoint_no_bracket=2 interior_bracket=0 still_same_sign=2 "
        "action=DIAGNOSTIC_ONLY solver_result=RADEQ_NO_BRACKET",
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = root / "positive.log"
        report = root / "positive.json"
        log.write_text("\n".join(lines) + "\n")
        result = run(log, report)
        if result.returncode != 0:
            raise SystemExit(f"positive failed: {result.stderr}")
        payload = json.loads(report.read_text())
        if payload["status"] != "PASS" or payload["owner_records"] != 3:
            raise SystemExit("positive report mismatch")

        incomplete = root / "incomplete.log"
        incomplete.write_text("\n".join(lines[:3]) + "\n")
        result = run(incomplete, root / "incomplete.json")
        if result.returncode != 4:
            raise SystemExit("incomplete callback accepted")

        bad_group = list(lines)
        bad_group[2] = bad_group[2].replace(
            "grouped_signed_rate=.75", "grouped_signed_rate=.8")
        path = root / "bad_group.log"
        path.write_text("\n".join(bad_group) + "\n")
        if run(path, root / "bad_group.json").returncode == 0:
            raise SystemExit("grouping mismatch accepted")

        bad_line_order = list(lines)
        bad_line_order[2] = bad_line_order[2].replace(
            "line_order_signed_rate=.75", "line_order_signed_rate=.8")
        path = root / "bad_line_order.log"
        path.write_text("\n".join(bad_line_order) + "\n")
        if run(path, root / "bad_line_order.json").returncode == 0:
            raise SystemExit("line-order grouping mismatch accepted")

        unsafe = list(lines)
        unsafe[0] = unsafe[0].replace("repair=0", "repair=1")
        path = root / "unsafe.log"
        path.write_text("\n".join(unsafe) + "\n")
        if run(path, root / "unsafe.json").returncode == 0:
            raise SystemExit("repair marker accepted")

    print("PASS a2_10_line_ion_owner_summary positive+4_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
