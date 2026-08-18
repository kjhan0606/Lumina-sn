#!/usr/bin/env python3
"""Run the 18 order-D cases plus the canonical-deck control (CPU only)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gate_parallel import run_cases, worker_count


FATAL_IDS = [
    "D1", "D2", "D3", "D4", "D7a", "D7b", "D7c", "D8",
    "D9", "D10", "D12", "D13", "D14", "D15", "D16", "D17",
]
WARN_IDS = ["D5", "D6"]


@dataclass(frozen=True)
class Case:
    case_id: str
    harness: Path
    deck: Path
    n_shells: int
    scratch: Path


@dataclass(frozen=True)
class Result:
    case_id: str
    returncode: int
    stdout: str
    stderr: str
    ok: bool


def invoke(
    harness: Path, deck: Path, n_shells: int, scratch: Path
) -> subprocess.CompletedProcess[str]:
    scratch.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["TMPDIR"] = str(scratch)
    return subprocess.run(
        [str(harness), str(deck), str(n_shells)],
        cwd=scratch,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def run_case(case: Case) -> Result:
    proc = invoke(case.harness, case.deck, case.n_shells, case.scratch)
    combined = proc.stdout + proc.stderr
    case_id = case.case_id
    if case_id == "canonical":
        # A sealed active-only quarantine deck deliberately removes every
        # zero/unlinked element from the loader topology.  Its positive control
        # must therefore load cleanly with no D5 missing-element warning.  The
        # historical full-topology deck retains its pinned D5 expectation.
        manifest_path = case.deck / "quarantine/manifest.json"
        sealed_active_only = False
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                sealed_active_only = (
                    manifest.get("state") == "sealed" and
                    "active_ions.csv" in manifest.get("active_files", {}) and
                    (case.deck / "quarantine/source_deck_snapshot").is_dir()
                )
            except (OSError, ValueError):
                sealed_active_only = False
        if sealed_active_only:
            ok = all([
                proc.returncode == 0,
                "[D5][WARN]" not in proc.stdout,
                "[D6][WARN]" not in proc.stdout,
                "[D" not in proc.stderr,
            ])
        else:
            ok = all([
                proc.returncode == 0,
                proc.stdout.count("[D5][WARN]") == 1,
                "missing Z: 12,13,21,22,23,24,25" in proc.stdout,
                "[D6][WARN]" not in proc.stdout,
            ])
        return Result(case_id, proc.returncode, proc.stdout, proc.stderr, ok)

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
    return Result(case_id, proc.returncode, proc.stdout, proc.stderr, all(checks))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    if manifest["case_count"] != 18 or len(FATAL_IDS) != 16 or len(WARN_IDS) != 2:
        raise SystemExit("invalid gate arithmetic: expected 16 FATAL + 2 WARN = 18")

    scratch_context = None
    if args.scratch_root is None:
        scratch_context = tempfile.TemporaryDirectory(prefix="lumina-d-gate-")
        scratch_root = Path(scratch_context.name)
    else:
        scratch_root = args.scratch_root.resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)

    case_ids = FATAL_IDS + WARN_IDS + ["canonical"]
    tasks = [
        Case(
            case_id,
            args.harness.resolve(),
            (args.canonical if case_id == "canonical"
             else args.fixtures / case_id).resolve(),
            50 if case_id == "canonical" else int(manifest["n_shells"]),
            scratch_root / case_id,
        )
        for case_id in case_ids
    ]
    print(
        f"D_GATE mode={'serial' if args.serial else 'parallel'} "
        f"workers={worker_count(args.serial)} scratch={scratch_root}"
    )
    try:
        results = run_cases(
            "D", run_case, tasks, serial=args.serial,
            case_name=lambda case: case.case_id,
        )
    finally:
        if scratch_context is not None:
            scratch_context.cleanup()

    passed = 0
    failed = 0
    for result in results:
        print(
            f"{'PASS' if result.ok else 'FAIL'} {result.case_id} "
            f"rc={result.returncode}"
        )
        print(
            f"RESULT battery=D case={result.case_id} "
            f"verdict={'PASS' if result.ok else 'FAIL'} rc={result.returncode}"
        )
        if not result.ok:
            if result.case_id == "canonical":
                print("--- canonical stdout ---")
                print(result.stdout.rstrip())
                print("--- canonical stderr ---")
                print(result.stderr.rstrip())
            else:
                print("--- stdout ---")
                print(result.stdout.rstrip())
                print("--- stderr ---")
                print(result.stderr.rstrip())
            failed += 1
        else:
            passed += 1

    print(f"SUMMARY cases=18 controls=1 PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
