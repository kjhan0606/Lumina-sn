#!/usr/bin/env python3
"""MC-EVT ME2 measurement and E4 byte-parity checker.

ME2 measures the end-to-end GPU spectrum difference between the named event
measure producers.  It intentionally does not assign a physical PASS threshold.
E4 is stricter: the two supplied spectra must be byte-identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_spectrum(path: Path) -> tuple[bytes, list[str], list[float], list[float]]:
    raw = path.read_bytes()
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    if not rows or len(rows[0]) != 2:
        raise ValueError(f"{path}: expected two-column spectrum CSV")
    if rows[0] != ["wavelength_angstrom", "flux"]:
        raise ValueError(f"{path}: unexpected header {rows[0]!r}")
    axis_text: list[str] = []
    axis: list[float] = []
    flux: list[float] = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise ValueError(f"{path}:{number}: expected two columns")
        x, y = float(row[0]), float(row[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{path}:{number}: nonfinite spectrum value")
        if axis and x <= axis[-1]:
            raise ValueError(f"{path}:{number}: wavelength axis is not increasing")
        axis_text.append(row[0])
        axis.append(x)
        flux.append(y)
    if not axis:
        raise ValueError(f"{path}: empty spectrum")
    return raw, axis_text, axis, flux


def metrics(left: list[float], right: list[float], axis: list[float]) -> dict[str, float | int]:
    relative: list[float] = []
    different = 0
    for a, b in zip(left, right):
        denominator = max(abs(a), abs(b))
        value = abs(a - b) / denominator if denominator else 0.0
        relative.append(value)
        if a != b:
            different += 1
    abs_integral = 0.0
    scale_integral = 0.0
    for i in range(1, len(axis)):
        width = axis[i] - axis[i - 1]
        abs_integral += 0.5 * width * (
            abs(left[i - 1] - right[i - 1]) + abs(left[i] - right[i])
        )
        scale_integral += 0.5 * width * (
            max(abs(left[i - 1]), abs(right[i - 1])) +
            max(abs(left[i]), abs(right[i]))
        )
    return {
        "different_flux_bins": different,
        "max_symmetric_relative_difference": max(relative),
        "median_symmetric_relative_difference": statistics.median(relative),
        "wavelength_weighted_l1_relative_difference": (
            abs_integral / scale_integral if scale_integral else 0.0
        ),
    }


def compare(left_path: Path, right_path: Path, gate: str) -> tuple[dict[str, object], int]:
    left_raw, left_axis_text, left_axis, left_flux = load_spectrum(left_path)
    right_raw, right_axis_text, right_axis, right_flux = load_spectrum(right_path)
    if left_axis_text != right_axis_text:
        raise ValueError("spectrum wavelength axes are not byte-identical")
    payload: dict[str, object] = {
        "schema": "lumina-mc-evt-spectrum-compare-v1",
        "gate": gate,
        "left": str(left_path),
        "right": str(right_path),
        "left_sha256": digest(left_raw),
        "right_sha256": digest(right_raw),
        "byte_identical": left_raw == right_raw,
        "axis_byte_identical": True,
        "bins": len(left_axis),
    }
    payload.update(metrics(left_flux, right_flux, left_axis))
    if gate == "E4":
        payload["verdict"] = "PASS" if left_raw == right_raw else "FAIL"
        return payload, 0 if left_raw == right_raw else 1
    payload["verdict"] = "MEASURED"
    return payload, 0


def selftest() -> int:
    axis = [1.0, 2.0, 3.0]
    same = metrics([0.0, 2.0, 4.0], [0.0, 2.0, 4.0], axis)
    changed = metrics([0.0, 2.0, 4.0], [0.0, 1.0, 8.0], axis)
    ok = (
        same["different_flux_bins"] == 0 and
        same["max_symmetric_relative_difference"] == 0.0 and
        changed["different_flux_bins"] == 2 and
        changed["max_symmetric_relative_difference"] == 0.5 and
        changed["wavelength_weighted_l1_relative_difference"] > 0.0
    )
    print("[E-SPECTRUM-COMPARE][SELFTEST][%s]" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("ME2", "E4"))
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.gate is None or args.left is None or args.right is None:
        parser.error("--gate, --left, and --right are required")
    try:
        payload, status = compare(args.left, args.right, args.gate)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"[E-{args.gate}][BLOCKED] reason={error}", file=sys.stderr)
        return 2
    print(f"[E-{args.gate}] " + json.dumps(payload, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
