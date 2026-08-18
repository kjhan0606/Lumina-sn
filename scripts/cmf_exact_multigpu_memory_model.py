#!/usr/bin/env python3
"""Exact allocation model for the compact ray-sharded CUDA prototype.

This models only src/cmf_exact_multigpu.cu.  It intentionally does not claim
that the rest of Lumina's CUDA state fits on the same devices.  The positive
window input is a conservative explicit bound; the actual solver derives it
from every segment beta.
"""

from __future__ import annotations

import argparse
import json


def ray_work(n_shells: int) -> list[int]:
    return [n_shells] * 16 + list(range(n_shells - 1, -1, -1))


def partition_boundaries(work: list[int], devices: int,
                         weighted: bool) -> list[int]:
    rays = len(work)
    if devices <= 0 or devices > rays:
        raise ValueError("invalid device count")
    if not weighted:
        return [rays * device // devices for device in range(devices)] + [rays]
    prefix = [0]
    for value in work:
        prefix.append(prefix[-1] + value)
    total = prefix[-1]
    quotient, remainder = divmod(total, devices)
    boundary = [0] * (devices + 1)
    boundary[-1] = rays
    for device in range(1, devices):
        minimum = boundary[device - 1] + 1
        maximum = rays - (devices - device)
        target = quotient * device + remainder * device // devices
        upper = minimum
        while upper < maximum and prefix[upper] < target:
            upper += 1
        chosen = upper
        if upper > minimum:
            lower = upper - 1
            if abs(prefix[lower] - target) <= abs(prefix[upper] - target):
                chosen = lower
        boundary[device] = chosen
    return boundary


def shard_bytes(
    n_shells: int,
    n_bins: int,
    devices: int,
    max_positive_window: int | None = None,
    *,
    weighted: bool = True,
    epoch_replay_max_window: int | None = None,
) -> list[int]:
    n_rays = n_shells + 16
    cells = n_shells * n_bins
    if n_shells <= 0 or n_bins < 2 or devices <= 0 or devices > n_rays:
        raise ValueError("invalid shape or device count")
    work = ray_work(n_shells)
    boundary = partition_boundaries(work, devices, weighted)
    result: list[int] = []
    for device in range(devices):
        ray_begin = boundary[device]
        ray_end = boundary[device + 1]
        compute_end = min(n_rays, ray_end + 1)
        local_rays = compute_end - ray_begin
        local_slots = sum(work[ray_begin:compute_end])
        segment_cells = local_slots * n_bins
        allocated = (
            local_rays * 4                 # rn
            + (local_rays + 1) * 4         # compact segment offsets
            + local_slots * 4              # shell
            + local_rays * 4               # core
            + local_slots * 8              # beta
            + n_rays * 8                   # impact
            + n_shells * 8                 # rmid
            + cells * 8                    # dt1
            + cells * 8                    # source
            + n_bins * 8                   # inner boundary
            + segment_cells * 8            # inward intensity
            + segment_cells * 8            # outward intensity
            + cells * 8                    # partial J
        )
        if max_positive_window is not None:
            if max_positive_window < 0:
                raise ValueError("negative positive-window extent")
            allocated += (
                cells * 8                  # t1
                + cells * 8                # source_cell
                + 6 * 4                    # device failure record
            )
            if epoch_replay_max_window is None:
                allocated += local_rays * 2 * max_positive_window * 32
            elif max_positive_window > epoch_replay_max_window:
                workspace_span = n_bins + max_positive_window
                allocated += local_rays * 2 * workspace_span * 16
        result.append(allocated)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shells", type=int, default=50)
    parser.add_argument("--bins", type=int, default=2_013_113)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--max-positive-window",
        type=int,
        default=47_649,
        help="conservative maximum qtop-1 over every segment",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Compact-layout small-grid allocation contract.
    fixture = shard_bytes(3, 96, 4)
    if max(fixture) != 35_796 or sum(fixture) != 124_584:
        raise RuntimeError(f"allocation-model self-check failed: {fixture}")
    positive_fixture = shard_bytes(3, 96, 4, 10)
    if max(positive_fixture) != 44_268 or sum(positive_fixture) != 157_192:
        raise RuntimeError(
            f"positive allocation-model self-check failed: {positive_fixture}"
        )

    rows = []
    for count in args.devices:
        allocations = shard_bytes(args.shells, args.bins, count)
        positive_allocations = shard_bytes(
            args.shells, args.bins, count, args.max_positive_window
        )
        epoch_allocations = shard_bytes(
            args.shells, args.bins, count, args.max_positive_window,
            epoch_replay_max_window=32,
        )
        rows.append({
            "devices": count,
            "max_device_bytes": max(allocations),
            "max_device_GiB": max(allocations) / 2**30,
            "total_device_bytes": sum(allocations),
            "total_device_GiB": sum(allocations) / 2**30,
            "per_device_GiB": [value / 2**30 for value in allocations],
            "fits_A10_23028_MiB": max(allocations) <= 23_028 * 2**20,
            "fits_A40_46068_MiB": max(allocations) <= 46_068 * 2**20,
            "positive_max_device_bytes": max(positive_allocations),
            "positive_max_device_GiB": max(positive_allocations) / 2**30,
            "positive_total_device_bytes": sum(positive_allocations),
            "positive_total_device_GiB": sum(positive_allocations) / 2**30,
            "positive_per_device_GiB": [
                value / 2**30 for value in positive_allocations
            ],
            "positive_fits_A10_23028_MiB": (
                max(positive_allocations) <= 23_028 * 2**20
            ),
            "positive_fits_A40_46068_MiB": (
                max(positive_allocations) <= 46_068 * 2**20
            ),
            "epoch_max_device_bytes": max(epoch_allocations),
            "epoch_max_device_GiB": max(epoch_allocations) / 2**30,
            "epoch_total_device_GiB": sum(epoch_allocations) / 2**30,
            "epoch_per_device_GiB": [value / 2**30 for value in epoch_allocations],
            "epoch_fits_A40_46068_MiB": (
                max(epoch_allocations) <= 46_068 * 2**20
            ),
        })
    payload = {
        "scope": "isolated cmf_exact_multigpu direct and positive allocations",
        "positive_max_window_bins": args.max_positive_window,
        "componentwise_envelope_device_operator_included": True,
        "persistent_production_integration_included": False,
        "full_lumina_cuda_state_included": False,
        "n_shells": args.shells,
        "n_bins": args.bins,
        "n_rays": args.shells + 16,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "devices direct_max_GiB serial_positive_max_GiB "
            "epoch_max_GiB epoch_total_GiB epoch_fits_A40"
        )
        for row in rows:
            print(
                f"{row['devices']:7d} {row['max_device_GiB']:14.3f} "
                f"{row['positive_max_device_GiB']:23.3f} "
                f"{row['epoch_max_device_GiB']:13.3f} "
                f"{row['epoch_total_device_GiB']:15.3f} "
                f"{str(row['epoch_fits_A40_46068_MiB']):>15s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
