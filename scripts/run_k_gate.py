#!/usr/bin/env python3
"""Run the established K-SHAPE/K-FRESH positive + six-negative battery."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gate_parallel import run_cases, worker_count


@dataclass(frozen=True)
class Case:
    case_id: str
    description: str
    expected: str
    mutation: str
    source: Path
    loader: Path
    scratch: Path


@dataclass(frozen=True)
class Result:
    case_id: str
    description: str
    expected: str
    returncode: int
    observed: str
    output: str
    ok: bool


def mirror_deck(source: Path, target: Path) -> None:
    target.mkdir()
    for entry in source.iterdir():
        os.symlink(entry.resolve(), target / entry.name)


def replace_contract(deck: Path, source: Path, old: str, new: str) -> None:
    path = deck / "kshape_contract.txt"
    path.unlink()
    text = (source / "kshape_contract.txt").read_text(encoding="ascii")
    path.write_text(text.replace(old, new), encoding="ascii")


def prepare(case: Case) -> Path:
    case.scratch.mkdir(parents=True, exist_ok=False)
    deck = case.scratch / "deck"
    mirror_deck(case.source, deck)
    if case.mutation == "none":
        return deck
    if case.mutation == "no-contract":
        (deck / "kshape_contract.txt").unlink()
    elif case.mutation == "bad-hash":
        replace_contract(
            deck,
            case.source,
            next(
                line for line in (case.source / "kshape_contract.txt")
                .read_text(encoding="ascii").splitlines()
                if line.startswith("line_list_sha256=")
            ),
            "line_list_sha256=" + "0" * 64,
        )
    elif case.mutation == "bad-shells":
        replace_contract(deck, case.source, "n_shells=50", "n_shells=30")
    elif case.mutation == "truncated-tau":
        target = deck / "tau_sobolev.npy"
        target.unlink()
        with (case.source / "tau_sobolev.npy").open("rb") as src:
            target.write_bytes(src.read(100_000))
    elif case.mutation == "npy-30col":
        # ★C3 수리 (Fable L3 Q3-2): 음성1 은 **살아있는 결함 덱**을 픽스처로 쓰고 있었다
        # (data/tardis_reference_toy06_19p48d 의 geometry=50 / npy=30열).
        # 그 덱이 고쳐지는 순간 대조가 조용히 무력해진다 — 그것이 C3 의 정의 그 자체다.
        # 음성 픽스처는 **찾는 것이 아니라 만드는 것**이다. 나머지 5종처럼 변조로 만든다:
        # 행 수는 유지하고 열만 30 으로 줄여 npy/geometry 불일치를 재현한다.
        target = deck / "tau_sobolev.npy"
        target.unlink()
        src_arr = np.load(case.source / "tau_sobolev.npy", mmap_mode="r",
                          allow_pickle=False)
        arr = np.lib.format.open_memmap(
            target, mode="w+", dtype=src_arr.dtype, shape=(src_arr.shape[0], 30))
        arr[:] = 0.0
        del arr
        replace_contract(deck, case.source, "n_shells=50", "n_shells=30")
    elif case.mutation == "stale-sentinel":
        target = deck / "tau_sobolev.npy"
        target.unlink()
        source_array = np.load(
            case.source / "tau_sobolev.npy", mmap_mode="r", allow_pickle=False
        )
        array = np.lib.format.open_memmap(
            target, mode="w+", dtype=source_array.dtype, shape=source_array.shape
        )
        array[:] = 0.0
        array[0, 0] = 12345.678
        array.flush()
        del array
    else:
        raise ValueError(f"unknown mutation: {case.mutation}")
    return deck


def run_case(case: Case) -> Result:
    deck = prepare(case)
    env = os.environ.copy()
    env["TMPDIR"] = str(case.scratch)
    proc = subprocess.run(
        [str(case.loader), str(deck)],
        cwd=case.scratch,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    observed = "OK" if proc.returncode == 0 else "FATAL"
    return Result(
        case.case_id,
        case.description,
        case.expected,
        proc.returncode,
        observed,
        proc.stdout + proc.stderr,
        observed == case.expected,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument(
        "--positive-deck",
        type=Path,
        default="data/tardis_reference_toy06_19p48d_sivcaiv",
    )
    parser.add_argument(
        "--negative-deck",
        type=Path,
        default="data/tardis_reference_toy06_19p48d",
    )
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()

    loader = args.loader.resolve()
    positive = args.positive_deck.resolve()
    negative = args.negative_deck.resolve()
    if not loader.is_file() or not positive.is_dir() or not negative.is_dir():
        parser.error("loader and both K deck directories must exist")

    scratch_context = None
    if args.scratch_root:
        scratch_root = args.scratch_root.resolve()
    else:
        scratch_context = tempfile.TemporaryDirectory(prefix="lumina-k-gate-")
        scratch_root = Path(scratch_context.name)
    scratch_root.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("positive", "양성: 정상 _sivcaiv (계약 있음, 50열)", "OK", "none", positive),
        ("canonical-30", "음성1: npy 30열 대 geometry 50셸(구성)", "FATAL", "npy-30col", positive),
        ("no-contract", "음성2: 계약 파일 없음", "FATAL", "no-contract", positive),
        ("bad-line-hash", "음성3: line_list 해시 불일치", "FATAL", "bad-hash", positive),
        ("bad-shells", "음성4: 계약 n_shells=30", "FATAL", "bad-shells", positive),
        ("truncated-tau", "음성5: tau_sobolev 절단", "FATAL", "truncated-tau", positive),
        ("stale-sentinel", "음성6: stale sentinel (형상 정상, 값 오염)", "FATAL", "stale-sentinel", positive),
    )
    tasks = [
        Case(*definition, loader, scratch_root / definition[0])
        for definition in definitions
    ]
    print(
        f"K_GATE mode={'serial' if args.serial else 'parallel'} "
        f"workers={worker_count(args.serial)} scratch={scratch_root}"
    )
    results = run_cases(
        "K", run_case, tasks, serial=args.serial,
        case_name=lambda case: case.case_id,
    )

    passed = 0
    failed = 0
    for result in results:
        marker = "  ok " if result.ok else "★MISS"
        print(
            f"{marker} rc={result.returncode:<3d} 기대={result.expected:<5s} "
            f"관측={result.observed:<5s}  {result.description}"
        )
        print(
            f"RESULT battery=K case={result.case_id} "
            f"verdict={'PASS' if result.ok else 'FAIL'} rc={result.returncode}"
        )
        if result.ok:
            passed += 1
        else:
            failed += 1
            print(result.output.rstrip())
    print(f"K_GATE_SUMMARY PASS={passed} MISS={failed} total={len(results)}")
    if scratch_context is not None:
        scratch_context.cleanup()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
