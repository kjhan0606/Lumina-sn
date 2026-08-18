#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/compare_a210_phase_baseline_streams.py"
PHASES = (
    "LOWER", "UPPER", "INTERIOR", "PUBLIC_SEED", "REQUESTED_TE",
    "GEOMETRIC_MID",
)


def baseline() -> list[str]:
    records = [
        "[A2-10][PRE] lane=DET iter=0 te_gen=1 rad=2 line=2 "
        "opacity=2 emissivity=2 population=2"
    ]
    for index, phase in enumerate(PHASES, 1):
        records.append(
            "[A2-10][ENDPOINT-FINITE] "
            f"phase={phase} shell=0 value={index}.0 repair=0"
        )
    return records


def run(reference: Path, candidate: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        "python3", str(SCRIPT),
        "--reference-log", str(reference.resolve()),
        "--candidate-log", str(candidate.resolve()),
        "--report", str(report),
    ], cwd=ROOT, text=True, capture_output=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = root / "reference.log"
        candidate = root / "candidate.log"
        records = baseline()
        reference.write_text("\n".join(records) + "\n", encoding="utf-8")
        candidate.write_text("\n".join([
            *records,
            "[A2-10][LINE-SATURATION-ROW] phase=REQUESTED_TE line=1",
            "[A2-10][LINE-SATURATION-UNION-META] "
            "phase=REQUESTED_TE line=1",
        ]) + "\n", encoding="utf-8")
        report = root / "pass.json"
        result = run(reference, candidate, report)
        if result.returncode != 0:
            raise SystemExit(f"positive failed: {result.stdout} {result.stderr}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS" or len(payload.get("phases", [])) != 6:
            raise SystemExit("positive report mismatch")

        changed = root / "changed.log"
        changed.write_text(
            candidate.read_text(encoding="utf-8").replace("value=2.0", "value=2.1"),
            encoding="utf-8",
        )
        if run(reference, changed, root / "changed.json").returncode != 4:
            raise SystemExit("baseline byte change accepted")

        missing = root / "missing.log"
        missing.write_text(
            "\n".join(records[:-1]) + "\n", encoding="utf-8"
        )
        if run(reference, missing, root / "missing.json").returncode != 4:
            raise SystemExit("missing phase accepted")

        repaired = root / "repaired.log"
        repaired.write_text(
            candidate.read_text(encoding="utf-8").replace("repair=0", "repair=1", 1),
            encoding="utf-8",
        )
        if run(reference, repaired, root / "repaired.json").returncode != 4:
            raise SystemExit("nonzero repair accepted")

        blocked = root / "blocked.log"
        blocked.write_text(
            candidate.read_text(encoding="utf-8") +
            "[A2-10][LINE-SATURATION-BLOCKED] reason=TEST\n",
            encoding="utf-8",
        )
        if run(reference, blocked, root / "blocked.json").returncode != 4:
            raise SystemExit("saturation block accepted")

    print("PASS a2_10_phase_baseline_streams positive+4_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
