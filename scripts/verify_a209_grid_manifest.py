#!/usr/bin/env python3
"""Independently verify an A2-09 grid manifest against a spectral CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


HEX64 = re.compile(r"^[0-9a-f]{64}$")
GRID_DOMAIN = b"A2-09:grid-manifest:Hz:bin-edges:IEEE754:v1"


class VerificationError(RuntimeError):
    pass


def integer(value: Any, label: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise VerificationError(f"invalid {label}: {value!r}")
    return value


def csv_integer(value: str | None, label: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"invalid {label}: {value!r}") from exc
    return parsed


def finite(value: str | None, label: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"invalid {label}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise VerificationError(f"non-finite {label}: {value!r}")
    return parsed


def bits(value: float) -> bytes:
    return struct.pack(">d", value)


def recover_edges(spectral_path: Path, n_shells: int,
                  n_bins: int) -> tuple[float, ...]:
    try:
        stream = spectral_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"cannot open spectral CSV: {exc}") from exc
    with stream:
        reader = csv.DictReader(stream)
        required = {"shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise VerificationError(f"spectral CSV missing columns: {missing}")
        cells: dict[tuple[int, int], tuple[float, float]] = {}
        for row_number, row in enumerate(reader, 2):
            shell = csv_integer(row.get("shell_id"), f"row {row_number} shell_id")
            bin_id = csv_integer(row.get("bin_id"), f"row {row_number} bin_id")
            if not (0 <= shell < n_shells and 0 <= bin_id < n_bins):
                raise VerificationError(
                    f"row {row_number} index out of range: shell={shell} bin={bin_id}"
                )
            key = (shell, bin_id)
            if key in cells:
                raise VerificationError(f"duplicate spectral cell: {key}")
            cells[key] = (
                finite(row.get("nu_lo_Hz"), f"row {row_number} nu_lo_Hz"),
                finite(row.get("nu_hi_Hz"), f"row {row_number} nu_hi_Hz"),
            )
    if len(cells) != n_shells * n_bins:
        raise VerificationError(
            f"spectral cell count mismatch: {len(cells)}/{n_shells * n_bins}"
        )

    reference: tuple[float, ...] | None = None
    for shell in range(n_shells):
        shell_edges: list[float] = []
        prior_hi: float | None = None
        for bin_id in range(n_bins):
            try:
                lo, hi = cells[(shell, bin_id)]
            except KeyError as exc:
                raise VerificationError(
                    f"missing spectral cell: shell={shell} bin={bin_id}"
                ) from exc
            if not (lo > 0.0 and hi > lo):
                raise VerificationError(
                    f"invalid grid cell: shell={shell} bin={bin_id} lo={lo} hi={hi}"
                )
            if prior_hi is not None and bits(lo) != bits(prior_hi):
                raise VerificationError(
                    f"discontinuous grid: shell={shell} bin={bin_id}"
                )
            shell_edges.append(lo)
            prior_hi = hi
        assert prior_hi is not None
        shell_edges.append(prior_hi)
        current = tuple(shell_edges)
        if reference is None:
            reference = current
        elif any(bits(left) != bits(right)
                 for left, right in zip(reference, current)):
            raise VerificationError(f"frequency grid differs at shell={shell}")
    assert reference is not None
    return reference


def appendix_a_grid_sha256(edges: tuple[float, ...], n_bins: int) -> str:
    digest = hashlib.sha256()
    digest.update(GRID_DOMAIN)
    digest.update(struct.pack(">Q", n_bins))
    for edge in edges:
        digest.update(struct.pack(">d", edge))
    return digest.hexdigest()


def verify(manifest_path: Path, spectral_path: Path) -> tuple[int, int, str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise VerificationError("manifest JSON root is not an object")
    n_shells = integer(manifest.get("n_shells"), "n_shells", 1)
    n_bins = integer(manifest.get("n_bins"), "n_bins", 1)
    recorded = manifest.get("grid_manifest_sha256")
    if not isinstance(recorded, str) or not HEX64.fullmatch(recorded):
        raise VerificationError(f"invalid manifest grid hash: {recorded!r}")
    spectral_name = manifest.get("spectral_file")
    if not isinstance(spectral_name, str) or Path(spectral_name).name != spectral_name:
        raise VerificationError(f"unsafe manifest spectral_file: {spectral_name!r}")
    if spectral_path.name != spectral_name:
        raise VerificationError(
            f"spectral filename mismatch: manifest={spectral_name!r} "
            f"argument={spectral_path.name!r}"
        )
    edges = recover_edges(spectral_path, n_shells, n_bins)
    calculated = appendix_a_grid_sha256(edges, n_bins)
    if calculated != recorded:
        raise VerificationError(
            f"grid hash mismatch: manifest={recorded} calculated={calculated}"
        )
    return n_shells, n_bins, calculated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("spectral_csv", type=Path)
    args = parser.parse_args()
    try:
        n_shells, n_bins, calculated = verify(
            args.manifest.resolve(), args.spectral_csv.resolve()
        )
    except VerificationError as exc:
        print(f"A209_GRID_MANIFEST_VERIFY FAIL reason={exc}", file=sys.stderr)
        return 3
    print(
        "A209_GRID_MANIFEST_VERIFY PASS "
        f"n_shells={n_shells} n_bins={n_bins} "
        f"grid_manifest_sha256={calculated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
