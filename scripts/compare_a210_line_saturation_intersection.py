#!/usr/bin/env python3
"""Prove byte identity of rows shared by sealed combined/union diagnostics.

The comparison is exact text identity, not a physical tolerance.  Selection
metadata is carried on separate records, so a shared LINE-SATURATION-ROW must
be byte-for-byte equal when both runs consumed the same physical state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW] "
META_PREFIX = "[A2-10][LINE-SATURATION-UNION-META] "
ION_SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-UNION-ION-SUMMARY] "
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY] "
BLOCKED_PREFIX = "[A2-10][LINE-SATURATION-BLOCKED] "
TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
TARGETS = (26, 27, 28)


class IntersectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntersectionError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(text: str, label: str) -> tuple[int, int, int]:
    fields = dict(TOKEN.findall(text))
    try:
        line = int(fields["line"])
        z = int(fields["Z"])
        ion = int(fields["ion"])
    except (KeyError, ValueError) as exc:
        raise IntersectionError(f"{label}: malformed identity") from exc
    require(line >= 0 and z in TARGETS and ion >= 0,
            f"{label}: identity outside target scope")
    return line, z, ion


def load_log(path: Path, expected_mode: str) -> dict[str, Any]:
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            f"missing or unsafe {expected_mode} log: {path}")
    rows: dict[int, tuple[int, str]] = {}
    metadata: dict[int, tuple[int, str]] = {}
    ion_summaries: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    blocked = 0
    with path.open("rt", encoding="utf-8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, 1):
            text = raw.rstrip("\n")
            if text.startswith(ROW_PREFIX):
                line, _, _ = identity(text, f"line {line_number}")
                require(line not in rows, f"duplicate row line={line}")
                rows[line] = (line_number, text)
            elif text.startswith(META_PREFIX):
                line, _, _ = identity(text, f"metadata line {line_number}")
                require(line not in metadata, f"duplicate metadata line={line}")
                metadata[line] = (line_number, text)
            elif text.startswith(ION_SUMMARY_PREFIX):
                ion_summaries.append(dict(TOKEN.findall(text)))
            elif text.startswith(SUMMARY_PREFIX):
                summaries.append(dict(TOKEN.findall(text)))
            elif text.startswith(BLOCKED_PREFIX):
                blocked += 1
    require(blocked == 0, f"{expected_mode}: blocked saturation record")
    require(len(summaries) == 1, f"{expected_mode}: summary count mismatch")
    summary = summaries[0]
    try:
        target_ion = int(summary.get("target_ion", 3))
    except (KeyError, ValueError) as exc:
        raise IntersectionError(
            f"{expected_mode}: summary target ion is malformed"
        ) from exc
    require(0 <= target_ion <= 10,
            f"{expected_mode}: summary target ion is outside schema")
    for line, (_, text) in rows.items():
        _, _, ion = identity(text, f"{expected_mode} row line={line}")
        require(ion == target_ion,
                f"{expected_mode}: row ion does not match summary target")
    for line, (_, text) in metadata.items():
        _, _, ion = identity(text, f"{expected_mode} metadata line={line}")
        require(ion == target_ion,
                f"{expected_mode}: metadata ion does not match summary target")
    require(int(summary.get("selected_rows", "-1")) == len(rows),
            f"{expected_mode}: selected row count mismatch")
    if expected_mode == "COMBINED_PREFIX":
        require("selection_mode" not in summary,
                "reference is not the sealed legacy combined mode")
        require(not metadata and not ion_summaries,
                "reference contains union-only records")
    else:
        require(summary.get("selection_mode") == "PER_ION_UNION",
                "candidate lacks per-ion union marker")
        require(set(metadata) == set(rows),
                "candidate row/union-metadata identity mismatch")
        require(len(ion_summaries) == 3 and
                {int(item.get("Z", "-1")) for item in ion_summaries} ==
                set(TARGETS),
                "candidate union ion summaries are incomplete")
        require(all(int(item.get("ion", "-1")) == target_ion
                    for item in ion_summaries),
                "candidate union summaries do not match target ion")
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "target_ion": target_ion,
        "rows": rows,
    }


def compare(reference_log: Path, candidate_log: Path) -> dict[str, Any]:
    reference = load_log(reference_log, "COMBINED_PREFIX")
    candidate = load_log(candidate_log, "PER_ION_UNION")
    require(reference["target_ion"] == candidate["target_ion"],
            "combined/union target ion mismatch")
    reference_rows = reference.pop("rows")
    candidate_rows = candidate.pop("rows")
    shared = sorted(set(reference_rows).intersection(candidate_rows))
    require(shared, "combined/union row intersection is empty")
    per_ion = {z: 0 for z in TARGETS}
    mismatches: list[dict[str, Any]] = []
    for line in shared:
        reference_number, reference_text = reference_rows[line]
        candidate_number, candidate_text = candidate_rows[line]
        _, z, _ = identity(reference_text, f"reference line={line}")
        per_ion[z] += 1
        if reference_text != candidate_text:
            mismatches.append({
                "line": line,
                "Z": z,
                "reference_line_number": reference_number,
                "candidate_line_number": candidate_number,
            })
    require(all(per_ion[z] > 0 for z in TARGETS),
            "intersection does not cover every target ion")
    require(not mismatches,
            f"shared row byte mismatch count={len(mismatches)}")
    return {
        "schema": "lumina-a210-line-saturation-intersection-v1",
        "status": "PASS",
        "verdict": "SHARED_PRINTED_ROWS_STRICT_BIT_EXACT",
        "reference": reference,
        "candidate": candidate,
        "reference_row_count": len(reference_rows),
        "candidate_row_count": len(candidate_rows),
        "intersection_row_count": len(shared),
        "intersection_by_atomic_number": {
            str(z): per_ion[z] for z in TARGETS
        },
        "intersection_rows_byte_identical": True,
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
            "PASS A210_LINE_SATURATION_INTERSECTION "
            f"shared={payload['intersection_row_count']} repair=0"
        )
        return 0
    except (IntersectionError, OSError, UnicodeError, ValueError) as exc:
        atomic_write(args.report, {
            "schema": "lumina-a210-line-saturation-intersection-v1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_LINE_SATURATION_INTERSECTION reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
