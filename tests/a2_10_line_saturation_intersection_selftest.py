#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/compare_a210_line_saturation_intersection.py"


def row(line: int, z: int, rank: int, value: str) -> str:
    return (
        "[A2-10][LINE-SATURATION-ROW] phase=REQUESTED_TE shell=0 "
        f"rank={rank} line={line} Z={z} ion=3 scaled_emission={value} "
        "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def meta(line: int, z: int, rank: int) -> str:
    return (
        "[A2-10][LINE-SATURATION-UNION-META] phase=REQUESTED_TE shell=0 "
        f"line={line} Z={z} ion=3 global_rank={rank} ion_rank=1 "
        "selection_mode=PER_ION_UNION physical_values_modified=0 "
        "clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def summary(selected: int, union: bool) -> str:
    mode = " selection_mode=PER_ION_UNION" if union else ""
    return (
        "[A2-10][LINE-SATURATION-SUMMARY] phase=REQUESTED_TE shell=0 "
        f"selected_rows={selected}{mode} physical_values_modified=0 "
        "clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def run(reference: Path, candidate: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        "python3", str(SCRIPT),
        "--reference-log", str(reference.resolve()),
        "--candidate-log", str(candidate.resolve()),
        "--report", str(report),
    ], cwd=ROOT, text=True, capture_output=True)


def main() -> int:
    common = [
        row(101, 26, 1, "9.0"),
        row(202, 27, 2, "8.0"),
        row(303, 28, 3, "7.0"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = root / "reference.log"
        candidate = root / "candidate.log"
        reference.write_text(
            "\n".join([*common, summary(3, False)]) + "\n",
            encoding="utf-8",
        )
        candidate_lines: list[str] = []
        for rank, text in enumerate(common, 1):
            candidate_lines.extend([text, meta((101, 202, 303)[rank - 1],
                                               (26, 27, 28)[rank - 1], rank)])
        for z in (26, 27, 28):
            candidate_lines.append(
                "[A2-10][LINE-SATURATION-UNION-ION-SUMMARY] "
                f"phase=REQUESTED_TE shell=0 Z={z} ion=3 "
                "selection_mode=PER_ION_UNION complete=1"
            )
        candidate_lines.append(summary(3, True))
        candidate.write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")
        report = root / "pass.json"
        result = run(reference, candidate, report)
        if result.returncode != 0:
            raise SystemExit(f"positive failed: {result.stdout} {result.stderr}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("intersection_row_count") != 3 or \
           payload.get("intersection_rows_byte_identical") is not True:
            raise SystemExit("positive report mismatch")

        changed = root / "changed.log"
        changed.write_text(
            candidate.read_text(encoding="utf-8").replace(
                "scaled_emission=8.0", "scaled_emission=8.1", 1
            ),
            encoding="utf-8",
        )
        if run(reference, changed, root / "changed.json").returncode != 4:
            raise SystemExit("shared row perturbation accepted")

        missing_meta = root / "missing_meta.log"
        missing_meta.write_text(
            "\n".join(candidate_lines[1:]) + "\n", encoding="utf-8"
        )
        if run(reference, missing_meta,
               root / "missing_meta.json").returncode != 4:
            raise SystemExit("union row/metadata deletion accepted")

    print("PASS a2_10_line_saturation_intersection positive+2_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
