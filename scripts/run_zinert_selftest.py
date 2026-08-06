#!/usr/bin/env python3
"""Run the Z-INERT cases with isolated scratch directories."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gate_parallel import run_cases, worker_count

CANONICAL_TAU_EXPECTATION = (
    "active_lines=2211572 active_tau_bit_differences=0 "
    "active_tau_fnv64=4a80c65d9c37fad9"
)


@dataclass(frozen=True)
class Case:
    case_id: str
    command: tuple[str, ...]
    expect_nonzero: bool
    scratch: Path


@dataclass(frozen=True)
class Result:
    case_id: str
    returncode: int
    output: str
    ok: bool


def run_case(case: Case) -> Result:
    case.scratch.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["TMPDIR"] = str(case.scratch)
    proc = subprocess.run(
        case.command,
        cwd=case.scratch,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    ok = proc.returncode != 0 if case.expect_nonzero else proc.returncode == 0
    if case.case_id == "canonical-tau":
        ok = ok and CANONICAL_TAU_EXPECTATION in proc.stdout
    return Result(case.case_id, proc.returncode, proc.stdout, ok)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--tau", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--canonical-tau", type=Path, required=True)
    parser.add_argument("--a2-08", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()

    paths = [args.validator, args.tau, args.population, args.canonical_tau, args.a2_08, args.verify]
    if any(not path.resolve().is_file() for path in paths) or not args.deck.resolve().is_dir():
        parser.error("all Z-INERT binaries, verifier, and deck must exist")
    scratch_context = None
    if args.scratch_root:
        scratch_root = args.scratch_root.resolve()
    else:
        scratch_context = tempfile.TemporaryDirectory(
            prefix="lumina-zinert-cases-"
        )
        scratch_root = Path(scratch_context.name)
    scratch_root.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("validator", (str(args.validator.resolve()),), False),
        ("negative", (str(args.validator.resolve()), "--inject-phantom"), True),
        ("tau", (str(args.tau.resolve()),), False),
        ("population", (str(args.population.resolve()),), False),
        ("canonical-tau", (str(args.canonical_tau.resolve()), str(args.deck.resolve())), False),
        ("a2-08-signed-opacity", (str(args.a2_08.resolve()),), False),
        ("verify", (sys.executable, str(args.verify.resolve()), "--deck", str(args.deck.resolve())), False),
    )
    tasks = [
        Case(case_id, command, expect_nonzero, scratch_root / case_id)
        for case_id, command, expect_nonzero in definitions
    ]
    print(
        f"Z_INERT_GATE mode={'serial' if args.serial else 'parallel'} "
        f"workers={worker_count(args.serial)} scratch={scratch_root}"
    )
    results = run_cases(
        "Z", run_case, tasks, serial=args.serial,
        case_name=lambda case: case.case_id,
    )
    failed = 0
    for result in results:
        if result.output:
            print(result.output.rstrip())
        if result.case_id == "negative" and result.ok:
            print(
                f"[Z-INERT-NEGATIVE] phantom population rejected "
                f"rc={result.returncode} PASS"
            )
        print(
            f"RESULT battery=Z case={result.case_id} "
            f"verdict={'PASS' if result.ok else 'FAIL'} rc={result.returncode}"
        )
        failed += not result.ok
    print(f"Z_INERT_SUMMARY PASS={len(results) - failed} FAIL={failed} total={len(results)}")
    if scratch_context is not None:
        scratch_context.cleanup()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
