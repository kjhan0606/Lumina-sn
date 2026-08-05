#!/usr/bin/env python3
"""Byte-compare two fixed-RNG output trees for the A2-01 OFF gate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, digest(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("trace_gate_off", type=Path)
    args = parser.parse_args()
    left = inventory(args.baseline.resolve())
    right = inventory(args.trace_gate_off.resolve())
    if not left:
        # 2026-08-05 driver fix: comparing zero files produced a vacuous PASS
        # (tree_sha256 of the empty string).  A parity verdict needs evidence.
        print(f"FAIL A2_01_GATE_OFF empty_baseline={args.baseline}")
        return 2
    trace_artifacts = [name for name in right if "a2_01_read_trace" in name]
    if trace_artifacts:
        print(f"FAIL A2_01_GATE_OFF unexpected_trace={','.join(trace_artifacts)}")
        return 2
    if left != right:
        left_only = sorted(set(left) - set(right))
        right_only = sorted(set(right) - set(left))
        changed = sorted(name for name in set(left) & set(right) if left[name] != right[name])
        print(
            "FAIL A2_01_GATE_OFF "
            f"left_only={left_only} right_only={right_only} changed={changed}"
        )
        return 2
    aggregate = hashlib.sha256()
    for name, (size, file_hash) in sorted(left.items()):
        aggregate.update(f"{name}\0{size}\0{file_hash}\n".encode("utf-8"))
    print(
        f"PASS A2_01_GATE_OFF files={len(left)} bytes_identical=true "
        f"tree_sha256={aggregate.hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
