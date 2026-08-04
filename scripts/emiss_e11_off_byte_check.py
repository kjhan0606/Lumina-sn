#!/usr/bin/env python3
"""Compare operator-supplied OFF captures byte-for-byte (no runs launched)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path,
                        help="capture with LUMINA_FLUOR_MATRIX_DUMP unset")
    parser.add_argument("off", type=Path,
                        help="same seeded capture with the gate explicitly empty")
    parser.add_argument("--artifact", action="append", required=True,
                        help="relative artifact path; repeat for every claimed output")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    all_equal = True
    for relative in args.artifact:
        left, right = args.baseline / relative, args.off / relative
        if not left.is_file() or not right.is_file():
            rows.append({"artifact": relative, "status": "MISSING"})
            all_equal = False
            continue
        left_hash, right_hash = digest(left), digest(right)
        equal = left.stat().st_size == right.stat().st_size and left_hash == right_hash
        rows.append({"artifact": relative, "bytes": left.stat().st_size,
                     "baseline_sha256": left_hash, "off_sha256": right_hash,
                     "byte_equal": equal})
        all_equal &= equal
    result = {"schema": "lumina-emiss-e11-off-byte-check-v1",
              "all_byte_equal": all_equal, "artifacts": rows}
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(raw)
    print(raw, end="")
    return 0 if all_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
