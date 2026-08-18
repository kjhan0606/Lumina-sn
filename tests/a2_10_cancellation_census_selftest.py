#!/usr/bin/env python3
"""Positive and fail-closed controls for the A2-10 census summarizer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/a210_cancellation_census.log"
SCRIPT = ROOT / "scripts/summarize_a210_cancellation_census.py"


def run(log: Path, directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            "--log",
            str(log),
            "--csv",
            str(directory / "census.csv"),
            "--report",
            str(directory / "census.json"),
            "--expect-phase",
            "LOWER",
            "--expect-phase",
            "UPPER",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-census-selftest-") as raw:
        directory = Path(raw)
        positive = run(FIXTURE, directory)
        if positive.returncode != 0:
            print(positive.stdout, end="")
            print(positive.stderr, end="", file=sys.stderr)
            return 4
        report = json.loads((directory / "census.json").read_text())
        if (
            report["status"] != "PASS"
            or report["unresolved_cell_rows"] != 3
            or report["phases"]["LOWER"]["ratio_bins"]["ratio_2_10"] != 1
            or report["phases"]["LOWER"]["ratio_bins"]["ratio_10_100"] != 1
            or report["phases"]["UPPER"]["ratio_bins"]["ratio_le_2"] != 1
        ):
            print("FAIL A2_10_CANCELLATION_CENSUS_SELFTEST positive artifact")
            return 4

        original = FIXTURE.read_text(encoding="utf-8")
        incomplete = directory / "incomplete.log"
        incomplete.write_text(
            original.replace("phase=LOWER complete=1", "phase=LOWER complete=0", 1),
            encoding="utf-8",
        )
        if run(incomplete, directory).returncode != 4:
            print("FAIL A2_10_CANCELLATION_CENSUS_SELFTEST incomplete accepted")
            return 4

        corrupted = directory / "corrupted.log"
        corrupted.write_text(
            original.replace(
                "uncertainty=6.9194822653875239e-57",
                "uncertainty=6.0e-57",
                1,
            ),
            encoding="utf-8",
        )
        if run(corrupted, directory).returncode != 4:
            print("FAIL A2_10_CANCELLATION_CENSUS_SELFTEST corrupt identity accepted")
            return 4

    print(
        "PASS A2_10_CANCELLATION_CENSUS_SELFTEST positive=1 "
        "incomplete_rejected=1 corrupt_identity_rejected=1 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
