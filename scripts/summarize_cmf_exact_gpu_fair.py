#!/usr/bin/env python3
"""Summarize one sealed H200x1/A40x4 production exact flight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def only_line(text: str, marker: str) -> str:
    lines = [line for line in text.splitlines() if marker in line]
    if len(lines) != 1:
        raise ValueError(f"expected one {marker!r}, found {len(lines)}")
    return lines[0]


def field(line: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise ValueError(f"missing {name} in {line}")
    return match.group(1)


def finite(line: str, name: str) -> float:
    value = float(field(line, name))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid {name}={value}")
    return value


def finite_range(line: str, name: str) -> tuple[float, float]:
    match = re.search(
        rf"(?:^| ){re.escape(name)}=\[([^,\]]+),([^\]]+)\]", line
    )
    if not match:
        raise ValueError(f"missing range {name}")
    lo, hi = float(match.group(1)), float(match.group(2))
    if not all(math.isfinite(value) for value in (lo, hi)) or lo < 0.0 or lo > hi:
        raise ValueError(f"invalid range {name}")
    return lo, hi


def gpu_samples(path: Path, expected_devices: int) -> list[dict[str, object]]:
    by_index: dict[int, dict[str, list[float] | str]] = {}
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 11:
                continue
            try:
                index = int(row[1].strip())
                used = float(row[4].strip())
                total = float(row[5].strip())
                gpu_util = float(row[6].strip())
                memory_util = float(row[7].strip())
                power = float(row[8].strip())
                sm_clock = float(row[9].strip())
                memory_clock = float(row[10].strip())
            except ValueError:
                continue
            item = by_index.setdefault(
                index,
                {
                    "uuid": row[2].strip(),
                    "name": row[3].strip(),
                    "used": [],
                    "total": [],
                    "gpu_util": [],
                    "memory_util": [],
                    "power": [],
                    "sm_clock": [],
                    "memory_clock": [],
                },
            )
            for key, value in (
                ("used", used),
                ("total", total),
                ("gpu_util", gpu_util),
                ("memory_util", memory_util),
                ("power", power),
                ("sm_clock", sm_clock),
                ("memory_clock", memory_clock),
            ):
                cast = item[key]
                assert isinstance(cast, list)
                cast.append(value)
    if len(by_index) != expected_devices:
        raise ValueError(
            f"GPU sample device count {len(by_index)} != {expected_devices}"
        )
    output: list[dict[str, object]] = []
    for index, item in sorted(by_index.items()):
        util = np.asarray(item["gpu_util"], dtype=np.float64)
        used = np.asarray(item["used"], dtype=np.float64)
        active = util[util > 0.0]
        output.append(
            {
                "index": index,
                "uuid": item["uuid"],
                "name": item["name"],
                "samples": int(util.size),
                "gpu_util_mean_percent_all_samples": float(util.mean()),
                "gpu_util_p95_percent_all_samples": float(np.percentile(util, 95)),
                "gpu_util_max_percent": float(util.max()),
                "gpu_util_positive_samples": int(active.size),
                "gpu_util_mean_percent_when_positive": (
                    float(active.mean()) if active.size else 0.0
                ),
                "memory_used_min_MiB": float(used.min()),
                "memory_used_peak_MiB": float(used.max()),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    if (root / "model.rc").read_text().strip() != "0":
        raise ValueError("model process did not exit zero")
    ab_report = json.loads((root / "ab_report.json").read_text())
    if ab_report.get("verdict") != "PASS":
        raise ValueError("A/B report is not PASS")
    stderr = (root / "stderr.log").read_text(errors="replace")
    epoch = only_line(stderr, "[cmf_fine][EXACT-MULTIGPU-EPOCH]")
    timing = only_line(stderr, "[cmf_fine][EXACT-MULTIGPU-TIMING]")
    ab = only_line(stderr, "[cmf_fine][EXACT-MULTIGPU-AB]")
    devices = [
        line
        for line in stderr.splitlines()
        if "[cmf_fine][EXACT-MULTIGPU-DEVICE]" in line
    ]
    if len(devices) != args.expected_devices:
        raise ValueError("device partition line count mismatch")
    timing_names = (
        "initialization_s",
        "fixed_point_s",
        "source_assembly_s",
        "h2d_s",
        "device_sweep_s",
        "d2h_s",
        "host_reduction_s",
        "convergence_check_s",
        "envelope_context_setup_s",
        "bounds_s",
        "envelope_residual_s",
        "envelope_verify_s",
        "envelope_refine_s",
        "publication_s",
        "cleanup_s",
        "reported_total_s",
        "caller_total_s",
    )
    report = {
        "schema": "lumina-cmf-exact-gpu-fair-lane-v1",
        "verdict": "PASS",
        "run_root": str(root),
        "binary_sha256": (root / "input/binary.sha256").read_text().strip(),
        "devices": args.expected_devices,
        "iterations": int(field(epoch, "iterations")),
        "residual": finite(epoch, "residual"),
        "finite_cpu_J": finite_range(ab, "finite_cpu_J"),
        "finite_gpu_J": finite_range(ab, "finite_gpu_J"),
        "max_relative_J": finite(ab, "max_relative_J"),
        "max_distance_over_combined_envelope": finite(
            ab, "max_distance_over_combined_envelope"
        ),
        "cpu_baseline_seconds": finite(ab, "cpu_baseline_s"),
        "gpu_owner_seconds": finite(ab, "gpu_owner_s"),
        "comparison_seconds": finite(ab, "comparison_s"),
        "timing": {name: finite(timing, name) for name in timing_names},
        "partition": [
            {
                "index": int(field(line, "index")),
                "rays": field(line, "rays"),
                "owned_segment_work": int(field(line, "owned_segment_work")),
                "computed_segment_work": int(
                    field(line, "computed_segment_work")
                ),
                "allocated_bytes": int(field(line, "allocated_bytes")),
            }
            for line in devices
        ],
        "gpu_sampling": gpu_samples(
            root / "gpu_samples.csv", args.expected_devices
        ),
        "repair": {"floor": 0, "cap": 0, "clamp": 0, "jitter": 0},
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
