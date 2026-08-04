#!/usr/bin/env python3
"""Compare Wave-3.2 Fe EW provenance before/after a frozen replay.

The comparison is keyed by channel/row/column/source/target so it can prove that
the shared field-source refactor is byte-invariant apart from the explicitly
expected R5 Kramers continuum rows.  It also reports the measured Fe II->Fe III
radiative photoionization-rate change without applying a tuning verdict.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


IDENTITY = "lumina_ew_iter0011_z26_s008_identity.csv"
PROVENANCE = "lumina_ew_iter0011_z26_s008_provenance.csv"
KEY_FIELDS = ("channel", "row", "column", "source_identity", "target_identity")


def read_identity(directory: Path) -> dict[int, int]:
    with (directory / IDENTITY).open(newline="") as handle:
        return {
            int(row["matrix_index"]): int(row["spectroscopic_stage"])
            for row in csv.DictReader(handle)
        }


def read_provenance(
    directory: Path,
) -> tuple[bytes, dict[tuple[str, ...], tuple[bytes, float]]]:
    """Keep each original CSV row as bytes; do not normalize float spelling."""
    with (directory / PROVENANCE).open("rb") as handle:
        header = handle.readline()
        names = next(csv.reader([header.decode("utf-8")]))
        index = {name: position for position, name in enumerate(names)}
        values: dict[tuple[str, ...], tuple[bytes, float]] = {}
        for raw in handle:
            columns = next(csv.reader([raw.decode("utf-8")]))
            key = tuple(columns[index[field]] for field in KEY_FIELDS)
            values[key] = (raw, float(columns[index["aggregated_rate"]]))
    return header, values


def fe2_rad_bf_gamma(
    values: dict[tuple[str, ...], tuple[bytes, float]], stages: dict[int, int]
) -> float:
    total = 0.0
    for key, (_raw, rate) in values.items():
        channel, row, column, _source, _target = key
        if channel != "rad_bf":
            continue
        if stages[int(column)] == 2 and stages[int(row)] == 3:
            total += rate
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", required=True, type=Path)
    parser.add_argument("--post", required=True, type=Path)
    parser.add_argument(
        "--expect-r5-only",
        action="store_true",
        help="fail if changed provenance channels are not rad_bf/coll_bf",
    )
    args = parser.parse_args()

    pre_stage = read_identity(args.pre)
    post_stage = read_identity(args.post)
    if pre_stage != post_stage:
        raise SystemExit("FAIL: EW identity/stage maps differ")

    pre_header, pre = read_provenance(args.pre)
    post_header, post = read_provenance(args.post)
    if pre_header != post_header:
        raise SystemExit("FAIL: provenance CSV headers differ by bytes")
    union = set(pre) | set(post)
    changed = {
        key for key in union
        if key not in pre or key not in post or pre[key][0] != post[key][0]
    }
    added = set(post) - set(pre)
    removed = set(pre) - set(post)
    changed_by_channel: dict[str, int] = {}
    for key in changed:
        changed_by_channel[key[0]] = changed_by_channel.get(key[0], 0) + 1

    gamma_pre = fe2_rad_bf_gamma(pre, pre_stage)
    gamma_post = fe2_rad_bf_gamma(post, post_stage)
    delta = gamma_post - gamma_pre
    ratio = gamma_post / gamma_pre if gamma_pre else math.inf
    delta_dex = math.log10(ratio) if ratio > 0.0 else math.nan

    print(f"provenance_union={len(union)}")
    print(f"provenance_unchanged={len(union) - len(changed)}")
    print(f"provenance_changed={len(changed)}")
    print(f"provenance_added={len(added)}")
    print(f"provenance_removed={len(removed)}")
    print(
        "changed_by_channel="
        + ",".join(f"{name}:{changed_by_channel[name]}" for name in sorted(changed_by_channel))
    )
    print(f"fe2_to_fe3_rad_bf_gamma_pre_s-1={gamma_pre:.17g}")
    print(f"fe2_to_fe3_rad_bf_gamma_post_s-1={gamma_post:.17g}")
    print(f"fe2_to_fe3_rad_bf_gamma_delta_s-1={delta:.17g}")
    print(f"fe2_to_fe3_rad_bf_gamma_ratio={ratio:.17g}")
    print(f"fe2_to_fe3_rad_bf_gamma_delta_dex={delta_dex:.17g}")

    if args.expect_r5_only:
        unexpected = changed - {
            key for key in changed if key[0] in {"rad_bf", "coll_bf"}
        }
        if unexpected or removed:
            print(f"FAIL: unexpected_changed_or_removed={len(unexpected) + len(removed)}")
            return 1
        print("PASS: provenance differences are confined to R5 bf fallback channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
