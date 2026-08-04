#!/usr/bin/env python3
"""Run the 18 order-D cases plus the canonical-deck control (CPU only)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


FATAL_IDS = [
    "D1", "D2", "D3", "D4", "D7a", "D7b", "D7c", "D8",
    "D9", "D10", "D12", "D13", "D14", "D15", "D16", "D17",
]
WARN_IDS = ["D5", "D6"]


def invoke(harness: Path, deck: Path, n_shells: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(harness), str(deck), str(n_shells)],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    if manifest["case_count"] != 18 or len(FATAL_IDS) != 16 or len(WARN_IDS) != 2:
        raise SystemExit("invalid gate arithmetic: expected 16 FATAL + 2 WARN = 18")

    passed = 0
    failed = 0
    for case_id in FATAL_IDS + WARN_IDS:
        proc = invoke(args.harness.resolve(), args.fixtures / case_id,
                      int(manifest["n_shells"]))
        combined = proc.stdout + proc.stderr
        marker = f"[{case_id}][{'FATAL' if case_id in FATAL_IDS else 'WARN'}]"
        checks = [marker in combined]
        if case_id in FATAL_IDS:
            checks.append(proc.returncode != 0)
        else:
            checks.extend([proc.returncode == 0, marker in proc.stdout])

        if case_id == "D8":
            checks.append(combined.count("[D8][FATAL]") >= 3)
        elif case_id == "D9":
            checks.extend([
                combined.count("[D9][FATAL]") >= 2,
                "atom_masses.csv" in combined,
                "abundances.csv" in combined,
            ])
        elif case_id == "D14":
            checks.append("row-count mismatch" in combined)
        elif case_id == "D17":
            checks.extend(["strtol" in combined, "strtod" in combined])
        elif case_id == "D5":
            checks.extend([
                proc.stdout.count("[D5][WARN]") == 1,
                "missing Z: 8" in proc.stdout,
            ])

        ok = all(checks)
        print(f"{'PASS' if ok else 'FAIL'} {case_id} rc={proc.returncode}")
        if not ok:
            print("--- stdout ---")
            print(proc.stdout.rstrip())
            print("--- stderr ---")
            print(proc.stderr.rstrip())
            failed += 1
        else:
            passed += 1

    canonical = invoke(args.harness.resolve(), args.canonical.resolve(), 50)
    canonical_ok = all([
        canonical.returncode == 0,
        canonical.stdout.count("[D5][WARN]") == 1,
        "missing Z: 12,13,21,22,23,24,25" in canonical.stdout,
        "[D6][WARN]" not in canonical.stdout,
    ])
    print(f"{'PASS' if canonical_ok else 'FAIL'} canonical rc={canonical.returncode}")
    if canonical_ok:
        passed += 1
    else:
        failed += 1
        print("--- canonical stdout ---")
        print(canonical.stdout.rstrip())
        print("--- canonical stderr ---")
        print(canonical.stderr.rstrip())

    print(f"SUMMARY cases=18 controls=1 PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
