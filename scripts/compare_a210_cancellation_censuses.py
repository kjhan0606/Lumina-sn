#!/usr/bin/env python3
"""Compare two A2-10 census CSVs from refinement-only experiments.

Changing the proof-envelope refinement count must not change any physical
input.  A candidate census may remove previously unresolved cells, but it may
not add new ones or enlarge a surviving Jbar bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "eta_per_sr",
    "chi_effective",
    "jbar",
    "absorption_per_sr",
    "net_per_sr",
    "signed_rate",
)


class ComparisonError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    if not path.is_file():
        raise ComparisonError(f"missing CSV: {path}")
    result: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                key = (row["phase"], int(row["line"]), int(row["shell"]))
            except (KeyError, ValueError) as exc:
                raise ComparisonError(f"{path}:{line_number}: invalid key") from exc
            if key in result:
                raise ComparisonError(f"{path}:{line_number}: duplicate key={key}")
            result[key] = row
    return result


def decimal(row: dict[str, str], field: str, key: tuple[str, int, int]) -> Decimal:
    try:
        value = Decimal(row[field])
    except (KeyError, InvalidOperation) as exc:
        raise ComparisonError(f"key={key}: invalid {field}") from exc
    if not value.is_finite():
        raise ComparisonError(f"key={key}: non-finite {field}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--baseline-refinements", type=int, required=True)
    parser.add_argument("--candidate-refinements", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.baseline_refinements < 1:
            raise ComparisonError("baseline refinements must be positive")
        if args.candidate_refinements <= args.baseline_refinements:
            raise ComparisonError("candidate refinements must be larger than baseline")
        baseline = load(args.baseline_csv)
        candidate = load(args.candidate_csv)
        new_keys = set(candidate) - set(baseline)
        if new_keys:
            raise ComparisonError(
                f"refinement introduced {len(new_keys)} unresolved cells; "
                f"first={sorted(new_keys)[0]}"
            )

        surviving: list[dict[str, Any]] = []
        maximum_bound_ratio = Decimal(0)
        minimum_bound_ratio: Decimal | None = None
        for key in sorted(candidate):
            before = baseline[key]
            after = candidate[key]
            changed = [field for field in IDENTITY_FIELDS if before[field] != after[field]]
            if changed:
                raise ComparisonError(
                    f"key={key}: physical identity changed fields={changed}"
                )
            before_bound = decimal(before, "jbar_bound", key)
            after_bound = decimal(after, "jbar_bound", key)
            if before_bound <= 0:
                raise ComparisonError(f"key={key}: baseline bound is not positive")
            if after_bound > before_bound:
                raise ComparisonError(
                    f"key={key}: bound increased {before_bound} -> {after_bound}"
                )
            ratio = after_bound / before_bound
            maximum_bound_ratio = max(maximum_bound_ratio, ratio)
            minimum_bound_ratio = ratio if minimum_bound_ratio is None \
                else min(minimum_bound_ratio, ratio)
            surviving.append(
                {
                    "phase": key[0],
                    "line": key[1],
                    "shell": key[2],
                    "baseline_jbar_bound": str(before_bound),
                    "candidate_jbar_bound": str(after_bound),
                    "candidate_to_baseline_bound_ratio": str(ratio),
                    "candidate_current_to_required_ratio": after[
                        "current_to_required_bound_ratio"
                    ],
                }
            )

        resolved = sorted(set(baseline) - set(candidate))
        payload = {
            "schema": "lumina-a2-10-refinement-only-comparison-v1",
            "status": "PASS",
            "reason_code": "PHYSICAL_IDENTITY_PRESERVED_AND_BOUND_NONINCREASING",
            "baseline": {
                "csv": str(args.baseline_csv.resolve()),
                "sha256": digest(args.baseline_csv),
                "refinements": args.baseline_refinements,
                "unresolved": len(baseline),
            },
            "candidate": {
                "csv": str(args.candidate_csv.resolve()),
                "sha256": digest(args.candidate_csv),
                "refinements": args.candidate_refinements,
                "unresolved": len(candidate),
            },
            "resolved_count": len(resolved),
            "resolved_line_shell": [list(key) for key in resolved],
            "surviving_count": len(surviving),
            "candidate_to_baseline_bound_ratio": {
                "minimum": str(minimum_bound_ratio) if minimum_bound_ratio is not None else None,
                "maximum": str(maximum_bound_ratio) if surviving else None,
            },
            "surviving": surviving,
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            "PASS A2_10_REFINEMENT_ONLY_COMPARISON "
            f"baseline={len(baseline)} candidate={len(candidate)} "
            f"resolved={len(resolved)} physical_identity=BIT_EXACT repair=0"
        )
        return 0
    except (ComparisonError, OSError, ValueError) as exc:
        print(f"FAIL A2_10_REFINEMENT_ONLY_COMPARISON reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
