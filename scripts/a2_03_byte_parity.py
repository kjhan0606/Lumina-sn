#!/usr/bin/env python3
"""Run fixed-RNG A2-03 OFF/ON lanes and compare every non-shadow byte."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def files(directory: Path, excluded: set[str]) -> dict[str, str]:
    return {
        item.name: digest(item)
        for item in sorted(directory.iterdir())
        if item.is_file() and item.name not in excluded
    }


def inspect_diagnostic(path: Path) -> dict[str, int | bool]:
    out_of_grid = -1
    unsampled = 0
    unsampled_nonzero = 0
    floor_values = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#out_of_grid_contributions="):
            out_of_grid = int(line.split("=", 1)[1])
        if not line or line.startswith("#") or line.startswith("shell,"):
            continue
        columns = line.split(",")
        value = float(columns[4])
        validity = int(columns[5])
        count = int(columns[6])
        if value == 1e-30:
            floor_values += 1
        if validity == 3:
            unsampled += 1
            if value != 0.0 or count != 0:
                unsampled_nonzero += 1
    return {
        "out_of_grid_contributions": out_of_grid,
        "unsampled_bins": unsampled,
        "unsampled_nonzero_bins": unsampled_nonzero,
        "historical_1e_30_floor_values": floor_values,
    }


def lane(name: str, directory: Path, command: list[str], environment: dict[str, str]) -> int:
    print(f"[a2-03 parity] lane={name} start", file=sys.stderr, flush=True)
    with (directory / "stdout.log").open("wb") as stdout, \
         (directory / "stderr.log").open("wb") as stderr:
        completed = subprocess.run(
            command, cwd=directory, env=environment, stdout=stdout, stderr=stderr,
            check=False,
        )
    print(f"[a2-03 parity] lane={name} rc={completed.returncode}",
          file=sys.stderr, flush=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--packets", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--parallel", action="store_true",
                        help="run OFF/ON as a two-chunk subprocess pool")
    parser.add_argument("--progress-seconds", type=int, default=30)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error("--binary must be a file")
    if args.fixture:
        command = [str(binary)]
    else:
        if args.data is None or not args.data.resolve().is_dir():
            parser.error("--data must be a directory unless --fixture is used")
        data = args.data.resolve()
        command = [str(binary), str(data), str(args.packets), str(args.iterations),
                   "spectrum", "nlte"]
    base_env = os.environ.copy()
    base_env.pop("LUMINA_RADFIELD_SHADOW", None)
    base_env.pop("LUMINA_RADFIELD_SHADOW_DUMP", None)
    base_env["LUMINA_LINE_INTERACTION"] = "scatter"
    base_env["OMP_NUM_THREADS"] = "1"

    with tempfile.TemporaryDirectory(prefix="a2_03_parity_") as temporary:
        root = Path(temporary)
        off_dir, on_dir = root / "off", root / "on"
        off_dir.mkdir()
        on_dir.mkdir()
        if not args.fixture:
            # The executable also resolves optional shared atomic tables through
            # ./data; bind both isolated lanes to the same read-only source tree.
            (off_dir / "data").symlink_to(data.parent, target_is_directory=True)
            (on_dir / "data").symlink_to(data.parent, target_is_directory=True)
        on_env = base_env.copy()
        on_env["LUMINA_RADFIELD_SHADOW"] = "1"
        on_env["LUMINA_RADFIELD_SHADOW_DUMP"] = "radiation_field_shadow.csv"
        if args.parallel:
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(lane, "OFF", off_dir, command, base_env): "OFF",
                    pool.submit(lane, "ON", on_dir, command, on_env): "ON",
                }
                exit_status: dict[str, int] = {}
                pending = set(futures)
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending, timeout=max(1, args.progress_seconds),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        exit_status[futures[future]] = future.result()
                    print(
                        f"[a2-03 parity] pool complete={len(exit_status)}/2 "
                        f"elapsed_s={time.monotonic() - started:.1f}",
                        file=sys.stderr, flush=True,
                    )
            off_rc, on_rc = exit_status["OFF"], exit_status["ON"]
        else:
            off_rc = lane("OFF", off_dir, command, base_env)
            on_rc = lane("ON", on_dir, command, on_env)

        off_hashes = files(off_dir, set())
        on_hashes = files(on_dir, {"radiation_field_shadow.csv"})
        same_names = set(off_hashes) == set(on_hashes)
        differing = sorted(
            name for name in set(off_hashes) & set(on_hashes)
            if off_hashes[name] != on_hashes[name]
        )
        diagnostic = on_dir / "radiation_field_shadow.csv"
        diagnostic_ok = diagnostic.is_file() and diagnostic.stat().st_size > 0
        diagnostic_metrics = inspect_diagnostic(diagnostic) if diagnostic_ok else {}
        verdict = (
            off_rc == 0 and on_rc == 0 and same_names and not differing and
            diagnostic_ok and diagnostic_metrics.get("out_of_grid_contributions") == 0 and
            diagnostic_metrics.get("unsampled_nonzero_bins") == 0 and
            diagnostic_metrics.get("historical_1e_30_floor_values") == 0
        )
        payload = {
            "schema": "lumina-a2-03-byte-parity-v1",
            "command": command,
            "fixed_rng": True,
            "fixture": args.fixture,
            "omp_num_threads": 1,
            "chunk_pool_workers": 2 if args.parallel else 0,
            "gate": "LUMINA_RADFIELD_SHADOW",
            "gate_default": "OFF",
            "off_exit_status": off_rc,
            "on_exit_status": on_rc,
            "compared_file_count": len(off_hashes),
            "same_file_set": same_names,
            "differing_files": differing,
            "changed_output_allowlist": ["radiation_field_shadow.csv"],
            "shadow_diagnostic_sha256": digest(diagnostic) if diagnostic_ok else None,
            "shadow_diagnostic_metrics": diagnostic_metrics,
            "guard_hits": diagnostic_metrics.get("out_of_grid_contributions"),
            "fallback_hits": 0,
            "verdict": "PASS" if verdict else "FAIL",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
