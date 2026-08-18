#!/usr/bin/env python3
"""Compare a non-census A2-10 gate against its sealed census reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


class ComparisonError(RuntimeError):
    pass


KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
PREFIXES = (
    "[cmf_fine][SIGNED-MATERIAL-CENSUS]",
    "[cmf_fine][EXACT-MULTIGPU-EPOCH]",
    "[R6][LINE-IDENTITY]",
    "[R6][LINE-COVERAGE]",
)
PROOF_ONLY_FIELDS = {
    "[cmf_fine][EXACT-MULTIGPU-EPOCH]": {
        "refinements", "component_error",
    },
    "[R6][LINE-IDENTITY]": {
        "refinements", "component_error", "profile_error",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def records(path: Path, occurrence: int) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ComparisonError(f"missing or unsafe log: {path}")
    if occurrence < 0:
        raise ComparisonError(f"negative record occurrence: {occurrence}")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    result: dict[str, dict[str, str]] = {}
    for prefix in PREFIXES:
        found = [line for line in lines if line.startswith(prefix)]
        if len(found) <= occurrence:
            raise ComparisonError(
                f"{path}: missing occurrence {occurrence} of {prefix!r}; "
                f"found {len(found)}"
            )
        result[prefix] = dict(KEY_VALUE.findall(found[occurrence]))
    return result


def bound_pair(text: str | None, label: str) -> tuple[float, float]:
    if text is None:
        raise ComparisonError(f"missing proof bound {label}")
    match = re.fullmatch(r"\[([^,]+),([^\]]+)\]", text)
    if match is None:
        raise ComparisonError(f"malformed proof bound {label}={text!r}")
    try:
        lower, upper = (float(value) for value in match.groups())
    except ValueError as exc:
        raise ComparisonError(f"non-numeric proof bound {label}={text!r}") from exc
    if (not math.isfinite(lower) or not math.isfinite(upper) or
            lower < 0.0 or upper < lower):
        raise ComparisonError(f"invalid proof bound {label}={text!r}")
    return lower, upper


def proof_refinement_changes(
    reference: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
    reference_refinements: int,
    candidate_refinements: int,
) -> list[dict[str, Any]]:
    if not (1 <= reference_refinements < candidate_refinements <= 64):
        raise ComparisonError("proof refinement rung must satisfy 1 <= reference < candidate <= 64")
    changes: list[dict[str, Any]] = []
    for prefix, fields in PROOF_ONLY_FIELDS.items():
        ref = reference[prefix]
        cand = candidate[prefix]
        if ref.get("refinements") != str(reference_refinements) or \
                cand.get("refinements") != str(candidate_refinements):
            raise ComparisonError(f"proof refinement count differs in {prefix}")
        for field in sorted(fields - {"refinements"}):
            ref_pair = bound_pair(ref.get(field), f"reference {prefix} {field}")
            cand_pair = bound_pair(cand.get(field), f"candidate {prefix} {field}")
            if cand_pair[0] > ref_pair[0] or cand_pair[1] >= ref_pair[1]:
                raise ComparisonError(
                    f"proof bound did not contract record={prefix} field={field} "
                    f"reference={ref.get(field)} candidate={cand.get(field)}"
                )
            changes.append({
                "record": prefix,
                "field": field,
                "reference": ref[field],
                "candidate": cand[field],
                "upper_bound_ratio": cand_pair[1] / ref_pair[1]
                    if ref_pair[1] > 0.0 else 0.0,
            })
        changes.append({
            "record": prefix,
            "field": "refinements",
            "reference": str(reference_refinements),
            "candidate": str(candidate_refinements),
        })
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-stderr", type=Path, required=True)
    parser.add_argument("--candidate-stderr", type=Path, required=True)
    parser.add_argument("--reference-occurrence", type=int, default=0)
    parser.add_argument("--candidate-occurrence", type=int, default=0)
    parser.add_argument("--reference-refinements", type=int)
    parser.add_argument("--candidate-refinements", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        reference = records(args.reference_stderr, args.reference_occurrence)
        candidate = records(args.candidate_stderr, args.candidate_occurrence)
        proof_mode = (args.reference_refinements is not None or
                      args.candidate_refinements is not None)
        if proof_mode and (args.reference_refinements is None or
                           args.candidate_refinements is None):
            raise ComparisonError("both proof refinement counts are required")
        proof_changes = proof_refinement_changes(
            reference, candidate, args.reference_refinements,
            args.candidate_refinements,
        ) if proof_mode else []
        differences: list[dict[str, str | None]] = []
        for prefix in PREFIXES:
            ref_fields = reference[prefix]
            candidate_fields = candidate[prefix]
            for key in sorted(set(ref_fields) | set(candidate_fields)):
                if proof_mode and key in PROOF_ONLY_FIELDS.get(prefix, set()):
                    continue
                if ref_fields.get(key) != candidate_fields.get(key):
                    differences.append({
                        "record": prefix,
                        "field": key,
                        "reference": ref_fields.get(key),
                        "candidate": candidate_fields.get(key),
                    })
        if differences:
            first = differences[0]
            raise ComparisonError(
                "reference identity changed "
                f"record={first['record']} field={first['field']} "
                f"reference={first['reference']} candidate={first['candidate']}"
            )
        payload = {
            "schema": "LUMINA_A210_TARGETED_REFERENCE_COMPARISON_V1",
            "status": "PASS",
            "reason": (
                "PHYSICAL_AND_SOLVER_FIELDS_BIT_EXACT_PROOF_BOUNDS_CONTRACTED"
                if proof_mode else "EXACT_AND_R6_FIELDS_BIT_EXACT"
            ),
            "comparison_mode": (
                "PROOF_REFINEMENT_ONLY" if proof_mode else "STRICT_BIT_EXACT"
            ),
            "reference_stderr": str(args.reference_stderr.resolve()),
            "reference_sha256": digest(args.reference_stderr),
            "candidate_stderr": str(args.candidate_stderr.resolve()),
            "candidate_sha256": digest(args.candidate_stderr),
            "reference_occurrence": args.reference_occurrence,
            "candidate_occurrence": args.candidate_occurrence,
            "records_compared": list(PREFIXES),
            "differences": [],
            "reference_refinements": args.reference_refinements,
            "candidate_refinements": args.candidate_refinements,
            "proof_bounds_nonincreasing": True if proof_mode else None,
            "proof_field_changes": proof_changes,
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
            "repair": 0,
        }
        atomic_write(args.report, payload)
        print(
            "PASS A210_TARGETED_REFERENCE_COMPARISON "
            f"records=4 identity={'PHYSICAL_SOLVER_BIT_EXACT' if proof_mode else 'BIT_EXACT'} "
            f"proof={'CONTRACTED' if proof_mode else 'UNCHANGED'} repair=0"
        )
        return 0
    except (ComparisonError, OSError, UnicodeError) as exc:
        payload = {
            "schema": "LUMINA_A210_TARGETED_REFERENCE_COMPARISON_V1",
            "status": "FAIL",
            "error": str(exc),
        }
        atomic_write(args.report, payload)
        print(f"FAIL A210_TARGETED_REFERENCE_COMPARISON reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
