#!/usr/bin/env python3
"""Fail-closed timing comparison for sealed H200x1 and A40x4 lane reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h200", type=Path, required=True)
    parser.add_argument("--a40", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    h200 = json.loads(args.h200.read_text())
    a40 = json.loads(args.a40.read_text())
    for name, lane, count in (("H200", h200, 1), ("A40", a40, 4)):
        if lane.get("verdict") != "PASS" or lane.get("devices") != count:
            raise ValueError(f"{name} lane is not the expected PASS lane")
    if h200["binary_sha256"] != a40["binary_sha256"]:
        raise ValueError("lanes did not run the same fat binary")
    if h200["iterations"] != a40["iterations"]:
        raise ValueError("lane convergence iteration counts differ")
    if h200["finite_cpu_J"] != a40["finite_cpu_J"]:
        raise ValueError("CPU baselines differ across lanes")
    for lane in (h200, a40):
        if lane["max_relative_J"] > 1.0e-12:
            raise ValueError("CPU/GPU J mismatch exceeds contract")
        if lane["max_distance_over_combined_envelope"] > 1.0:
            raise ValueError("CPU/GPU estimates have disjoint envelopes")
    h_owner = float(h200["gpu_owner_seconds"])
    a_owner = float(a40["gpu_owner_seconds"])
    h_sweep = float(h200["timing"]["device_sweep_s"])
    a_sweep = float(a40["timing"]["device_sweep_s"])
    if not all(
        value > 0.0 and math.isfinite(value)
        for value in (h_owner, a_owner, h_sweep, a_sweep)
    ):
        raise ValueError("nonpositive timing in fair comparison")
    report = {
        "schema": "lumina-cmf-exact-h200x1-vs-a40x4-v1",
        "verdict": "PASS",
        "binary_sha256": h200["binary_sha256"],
        "iterations": h200["iterations"],
        "h200x1_gpu_owner_seconds": h_owner,
        "a40x4_gpu_owner_seconds": a_owner,
        "gpu_owner_speedup_h200x1_over_a40x4": a_owner / h_owner,
        "h200x1_device_sweep_seconds": h_sweep,
        "a40x4_device_sweep_seconds": a_sweep,
        "device_sweep_speedup_h200x1_over_a40x4": a_sweep / h_sweep,
        "h200x1_host_reduction_seconds": h200["timing"][
            "host_reduction_s"
        ],
        "a40x4_host_reduction_seconds": a40["timing"][
            "host_reduction_s"
        ],
        "finite_cpu_J": h200["finite_cpu_J"],
        "finite_h200_J": h200["finite_gpu_J"],
        "finite_a40_J": a40["finite_gpu_J"],
        "max_relative_J": {
            "h200x1": h200["max_relative_J"],
            "a40x4": a40["max_relative_J"],
        },
        "gpu_sampling": {
            "h200x1": h200["gpu_sampling"],
            "a40x4": a40["gpu_sampling"],
        },
        "repair": {"floor": 0, "cap": 0, "clamp": 0, "jitter": 0},
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
