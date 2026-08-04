#!/usr/bin/env python3
"""L1-A(v3) read-only atomic-data instrument: JSONL stdout, table stderr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import l1a_collision
import l1a_lines
import l1a_sigma
from l1a_common import (
    ContractError, RSS_LIMIT, THRESHOLD_MODES, compare_golden, finalize_run,
    load_golden, peak_rss_bytes, validate_record,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "docs/L1_GOLDEN_MANIFEST.json"
ENGINES = {
    "collision": l1a_collision.run,
    "lines": l1a_lines.run,
    "sigma": l1a_sigma.run,
}


def _engine_list(value: str) -> list[str]:
    if value == "all":
        return list(ENGINES)
    result = value.split(",")
    unknown = set(result)-set(ENGINES)
    if unknown or not result:
        raise argparse.ArgumentTypeError(f"unknown engines: {sorted(unknown)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--epoch-peer", type=Path, required=True,
                        help="explicit peer used only by I19 legacy_vs_current metrics")
    parser.add_argument("--cmfgen-tree", type=Path, required=True)
    parser.add_argument("--cmfgen-run", type=Path, required=True)
    parser.add_argument("--super-cutoff", type=int, required=True)
    parser.add_argument("--engine", required=True,
                        help="collision, lines, sigma, comma list, or all")
    parser.add_argument("--chunk-points", type=int, required=True)
    parser.add_argument("--threshold-mode", choices=sorted(THRESHOLD_MODES), required=True)
    return parser.parse_args()


def _command(args: argparse.Namespace) -> str:
    values = [
        sys.executable, str(Path(__file__).resolve()), "--deck", str(args.deck),
        "--epoch-peer", str(args.epoch_peer),
        "--cmfgen-tree", str(args.cmfgen_tree), "--cmfgen-run", str(args.cmfgen_run),
        "--super-cutoff", str(args.super_cutoff),
        "--engine", args.engine, "--chunk-points", str(args.chunk_points),
        "--threshold-mode", args.threshold_mode,
    ]
    return shlex.join(values)


def main() -> int:
    args = parse_args()
    try:
        golden = load_golden(GOLDEN)
        args.deck = args.deck.resolve(strict=True)
        args.epoch_peer = args.epoch_peer.resolve(strict=True)
        args.cmfgen_tree = args.cmfgen_tree.resolve(strict=True)
        args.cmfgen_run = args.cmfgen_run.resolve(strict=True)
        if not all(path.is_dir() for path in
                   (args.deck, args.epoch_peer, args.cmfgen_tree, args.cmfgen_run)):
            raise ContractError("deck, epoch-peer, cmfgen-tree, and cmfgen-run must be directories")
        if args.chunk_points <= 0:
            raise ContractError("chunk-points must be positive")
        if args.deck == args.epoch_peer:
            raise ContractError("epoch-peer must differ from selected deck")
        if args.super_cutoff <= 0:
            raise ContractError("super-cutoff must be positive")
        engines = _engine_list(args.engine)
        threshold = float(golden["thresholds"][args.threshold_mode])
    except (OSError, KeyError, ValueError, ContractError) as exc:
        print(f"L1A INPUT ERROR: {exc}", file=sys.stderr)
        return 2

    command = _command(args)
    output: list[dict] = []
    failures: list[str] = []
    warnings: list[str] = []
    for engine in engines:
        started = time.perf_counter()
        try:
            kwargs = dict(
                deck=args.deck, peer=args.epoch_peer, cmfgen_tree=args.cmfgen_tree,
                cmfgen_run=args.cmfgen_run, threshold_mode=args.threshold_mode,
                threshold=threshold, command=command, super_cutoff=args.super_cutoff,
            )
            if engine == "sigma":
                kwargs["chunk_points"] = args.chunk_points
            records = ENGINES[engine](**kwargs)
            wall = time.perf_counter()-started
            rss = peak_rss_bytes()
            if rss > RSS_LIMIT:
                raise ContractError(f"C11 peak RSS {rss} exceeds 2^30")
            for record in records:
                record["resources"]["peak_rss_bytes"] = rss
                record["resources"]["wall_seconds"] = wall
                record["resources"]["chunk_points"] = args.chunk_points
                warnings.extend(f"{engine}/{record['id']}: {item}"
                                for item in validate_record(record))
                compare_golden(record, golden)
            output.extend(records)
        except Exception as exc:  # independent engine boundary is deliberate
            failures.append(f"{engine}: {type(exc).__name__}: {exc}")

    finalize_run(output, args.engine == "all", failures)
    for record in output:
        print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))

    print("ENGINE     ID    OUTCOME          DENOMINATOR    MISSING  UNSUPPORTED  GOLDEN        ELIGIBLE RSS(MiB)  WALL(s)",
          file=sys.stderr)
    print("---------- ----- ---------------- -------------- ---------- ------------ ------------- -------- --------- --------",
          file=sys.stderr)
    for record in output:
        engine = next((name for name in ENGINES if name in record["evidence"]["validator"]), "-")
        print(f"{engine:10s} {record['id']:5s} {record['verdict']['outcome']:16s} "
              f"{record['universe']['denominator']:14d} {record['states']['missing']:10d} "
              f"{record['states']['unsupported']:12d} "
              f"{record['golden']['status']:13s} {str(record['judgment_eligible']):8s} "
              f"{record['resources']['peak_rss_bytes']/2**20:9.1f} "
              f"{record['resources']['wall_seconds']:8.3f}", file=sys.stderr)
    for warning in warnings:
        print(f"L1A WARNING: {warning}", file=sys.stderr)
    for failure in failures:
        print(f"L1A ENGINE FAILURE: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
