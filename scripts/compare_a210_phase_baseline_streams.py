#!/usr/bin/env python3
"""Compare non-saturation A2-10 phase records as strict byte streams.

This proves diagnostic lineage only.  It excludes the explicitly new
LINE-SATURATION-* records and applies no numerical or physical tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PHASES = (
    "LOWER", "UPPER", "INTERIOR", "PUBLIC_SEED", "REQUESTED_TE",
    "GEOMETRIC_MID",
)
A210_PREFIX = "[A2-10]"
SATURATION_PREFIX = "[A2-10][LINE-SATURATION-"
SATURATION_BLOCKED = "[A2-10][LINE-SATURATION-BLOCKED]"
PHASE = re.compile(r"(?:^|\s)phase=([^\s]+)")
FORBIDDEN_NONZERO = re.compile(
    r"(?:physical_values_modified|floor|cap|clamp|jitter|repair)="
    r"([1-9][0-9]*)"
)


class BaselineError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest_records(records: list[str]) -> str:
    payload = ("\n".join(records) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def extract(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            f"missing or unsafe log: {path}")
    streams = {phase: [] for phase in PHASES}
    pre: list[str] = []
    with path.open("rt", encoding="utf-8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, 1):
            text = raw.rstrip("\n")
            if text.startswith(SATURATION_BLOCKED):
                raise BaselineError(
                    f"line {line_number}: saturation diagnostic blocked"
                )
            if not text.startswith(A210_PREFIX) or \
               text.startswith(SATURATION_PREFIX):
                continue
            if FORBIDDEN_NONZERO.search(text):
                raise BaselineError(
                    f"line {line_number}: forbidden mutation/repair marker"
                )
            match = PHASE.search(text)
            if match:
                phase = match.group(1)
                if phase in streams:
                    streams[phase].append(text)
            elif text.startswith("[A2-10][PRE] "):
                pre.append(text)
    require(len(pre) == 1, f"expected one PRE record, found {len(pre)}")
    for phase in PHASES:
        require(streams[phase], f"missing phase stream: {phase}")
    return streams, pre


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_streams, reference_pre = extract(reference)
    candidate_streams, candidate_pre = extract(candidate)
    require(reference_pre == candidate_pre, "PRE record is not bit-exact")
    phases: list[dict[str, Any]] = []
    for phase in PHASES:
        expected = reference_streams[phase]
        observed = candidate_streams[phase]
        if len(expected) != len(observed):
            raise BaselineError(
                f"{phase}: record count {len(observed)} != {len(expected)}"
            )
        for index, (left, right) in enumerate(zip(expected, observed), 1):
            if left != right:
                raise BaselineError(
                    f"{phase}: byte mismatch at record {index}"
                )
        phases.append({
            "phase": phase,
            "record_count": len(expected),
            "stream_sha256": digest_records(expected),
            "strict_bit_exact": True,
        })
    return {
        "schema": "lumina-a210-phase-baseline-stream-comparison-v1",
        "status": "PASS",
        "verdict": "NON_SATURATION_A210_PHASE_STREAMS_STRICT_BIT_EXACT",
        "reference_log": str(reference.resolve()),
        "reference_log_sha256": digest_file(reference),
        "candidate_log": str(candidate.resolve()),
        "candidate_log_sha256": digest_file(candidate),
        "pre_record_sha256": digest_records(reference_pre),
        "phases": phases,
        "excluded_record_family": "[A2-10][LINE-SATURATION-*]",
        "physical_tolerance_used": False,
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = compare(args.reference_log, args.candidate_log)
        atomic_write(args.report, payload)
        print(
            "PASS A210_PHASE_BASELINE_STREAMS "
            f"phases={len(payload['phases'])} repair=0"
        )
        return 0
    except (BaselineError, OSError, UnicodeError) as exc:
        atomic_write(args.report, {
            "schema": "lumina-a210-phase-baseline-stream-comparison-v1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_PHASE_BASELINE_STREAMS reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
