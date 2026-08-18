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

REPO_ROOT = Path(__file__).resolve().parent.parent

# canonical-tau 는 **덱에 고정된 바이트-parity 트립와이어**다.  tau 는 선 파장·f 에
# 의존하므로 덱이 다르면 값도 다르다 — 단일 기대값은 한 덱에서만 유효하다.
# 등록되지 않은 덱은 **조용히 통과시키지 않고 실패**한다(기준선 없는 덱을 PASS 로
# 세탁하지 않는다).  새 덱을 추가할 때는 측정 출처를 주석으로 남긴다.
CANONICAL_TAU_BASELINE = {
    # 종래 생산 덱 (2,565,342 선). A2-07 에서 LTE@T_rad,W -> LTE@T_e 변경 후 재수립.
    "tardis_reference_toy06_19p48d":
        "active_lines=2211572 active_tau_bit_differences=0 "
        "active_tau_fnv64=4a80c65d9c37fad9",
    # 덱 전환 T1 (docs/DECK_TRANSITION_SCOPING.md): jnu4 종속 + I20 수리 +
    # CMFGEN_EXACT_HYD=1.  2,220,953 선.  2026-08-06 T2 배터리 실측으로 수립하며,
    # 같은 실행에서 구조 불변량(active_tau_bit_differences=0 · inactive_nonzero=0 ·
    # audit_rc=0)이 전부 성립함을 확인했다 — 실패를 덮은 것이 아니라 기준선을 연 것이다.
    "tardis_reference_toy06_19p48d_jnu4":
        "active_lines=1867183 active_tau_bit_differences=0 "
        "active_tau_fnv64=f32d10df3421058b",
    # 19apr23 active-only quarantine deck, measured fail-closed on 2026-08-08
    # after SH-GRID/exact-Hyd promotion.  Same run: inactive_nonzero=0,
    # active_tau_bit_differences=0, audit_rc=0.
    "tardis_reference_toy06_19p48d_sivcaiv_active":
        "active_lines=2588798 active_tau_bit_differences=0 "
        "active_tau_fnv64=6c53c2f89ad53e47",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    command: tuple[str, ...]
    expect_nonzero: bool
    scratch: Path
    deck_name: str = ""      # canonical-tau 기대값을 덱별로 고르기 위해


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
    if case.case_id == "canonical-tau":
        # The case intentionally runs from an isolated scratch cwd.  Bind the
        # global energy-zero catalog explicitly so this whole-deck loader test
        # exercises the same immutable reference path as sealed DET flights.
        env["LUMINA_IONIZATION_REFERENCE_FILE"] = str(
            REPO_ROOT / "data/atomic/ionization_reference.csv"
        )
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
        expected = CANONICAL_TAU_BASELINE.get(case.deck_name)
        if expected is None:
            print(f"CANONICAL_TAU no baseline registered for deck "
                  f"'{case.deck_name}' -- refusing to pass unmeasured deck")
            ok = False
        else:
            ok = ok and expected in proc.stdout
    return Result(case.case_id, proc.returncode, proc.stdout, ok)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--tau", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--canonical-tau", type=Path, required=True)
    parser.add_argument("--a2-08", type=Path, required=True)
    parser.add_argument("--a2-09", type=Path, required=True)
    parser.add_argument("--a2-10", type=Path, required=True)
    parser.add_argument("--a2-12-contract", type=Path, required=True)
    parser.add_argument("--a2-13-15-contract", type=Path, required=True)
    parser.add_argument("--a2-17-jnu-seed", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()

    paths = [args.validator, args.tau, args.population, args.canonical_tau, args.a2_08, args.a2_09, args.a2_10, args.a2_12_contract, args.a2_13_15_contract, args.a2_17_jnu_seed, args.verify]
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
        ("a2-09-emissivity", (str(args.a2_09.resolve()),), False),
        ("a2-10-radeq", (str(args.a2_10.resolve()),), False),
        ("a2-12-gpu-lifecycle-contract", (str(args.a2_12_contract.resolve()),), False),
        ("a2-13-15-gpu-physics-contract", (str(args.a2_13_15_contract.resolve()),), False),
        ("a2-17-native-jnu-seed", (str(args.a2_17_jnu_seed.resolve()),), False),
        ("verify", (sys.executable, str(args.verify.resolve()), "--deck", str(args.deck.resolve())), False),
    )
    tasks = [
        Case(case_id, command, expect_nonzero, scratch_root / case_id,
             args.deck.resolve().name)
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
