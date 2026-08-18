#!/usr/bin/env python3
"""Summarize per-GPU memory traces produced by nvidia-smi --loop-ms."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-devices", type=int, default=4)
    args = parser.parse_args()
    if args.expected_devices <= 0:
        parser.error("--expected-devices must be positive")
    samples: dict[tuple[int, str], list[float]] = defaultdict(list)
    with args.trace.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 4:
                continue
            try:
                index = int(row[1].strip())
                used = float(row[3].strip())
            except ValueError:
                continue
            samples[(index, row[2].strip())].append(used)
    if len(samples) < args.expected_devices:
        raise SystemExit(
            f"fewer than {args.expected_devices} GPU traces: {len(samples)}"
        )
    for (index, uuid), values in sorted(samples.items()):
        baseline = min(values)
        peak = max(values)
        print(
            f"VRAM_PEAK index={index} uuid={uuid} samples={len(values)} "
            f"baseline_min_MiB={baseline:.0f} peak_MiB={peak:.0f} "
            f"delta_MiB={peak - baseline:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
