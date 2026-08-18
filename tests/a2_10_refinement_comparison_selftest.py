#!/usr/bin/env python3
"""Controls for the refinement-only A2-10 census comparison."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_a210_cancellation_censuses.py"
FIELDS = (
    "phase", "line", "shell", "eta_per_sr", "chi_effective", "jbar",
    "jbar_bound", "absorption_per_sr", "net_per_sr", "signed_rate",
    "current_to_required_bound_ratio",
)


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(base: Path, candidate: Path, report: Path) -> int:
    return subprocess.run(
        (
            sys.executable, str(SCRIPT),
            "--baseline-csv", str(base),
            "--candidate-csv", str(candidate),
            "--baseline-refinements", "8",
            "--candidate-refinements", "10",
            "--report", str(report),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
    ).returncode


def row(phase: str, line: str, bound: str) -> dict[str, str]:
    return {
        "phase": phase,
        "line": line,
        "shell": "1",
        "eta_per_sr": "3",
        "chi_effective": "2",
        "jbar": "1",
        "jbar_bound": bound,
        "absorption_per_sr": "2",
        "net_per_sr": "1",
        "signed_rate": "12.566370614359172",
        "current_to_required_bound_ratio": bound,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-refine-selftest-") as raw:
        directory = Path(raw)
        base = directory / "base.csv"
        candidate = directory / "candidate.csv"
        report = directory / "report.json"
        base_rows = [row("LOWER", "15", "7"), row("UPPER", "20", "1.5")]
        write(base, base_rows)
        write(candidate, [row("LOWER", "15", "3")])
        if run(base, candidate, report) != 0:
            print("FAIL A2_10_REFINEMENT_COMPARISON_SELFTEST positive")
            return 4

        changed = row("LOWER", "15", "3")
        changed["jbar"] = "1.1"
        write(candidate, [changed])
        if run(base, candidate, report) != 4:
            print("FAIL A2_10_REFINEMENT_COMPARISON_SELFTEST identity mutation")
            return 4

        write(candidate, [row("LOWER", "15", "8")])
        if run(base, candidate, report) != 4:
            print("FAIL A2_10_REFINEMENT_COMPARISON_SELFTEST bound increase")
            return 4

        write(candidate, [row("LOWER", "15", "3"), row("UPPER", "99", "1")])
        if run(base, candidate, report) != 4:
            print("FAIL A2_10_REFINEMENT_COMPARISON_SELFTEST new unresolved")
            return 4

    print(
        "PASS A2_10_REFINEMENT_COMPARISON_SELFTEST positive=1 "
        "identity_mutation_rejected=1 bound_increase_rejected=1 "
        "new_unresolved_rejected=1 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
