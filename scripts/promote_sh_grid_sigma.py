#!/usr/bin/env python3
"""Atomically promote a verified SH-GRID sigma asset into a sealed deck."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np


N_BINS = 1178
NU_MIN = 5.8412785919616062e13
NU_MAX = 3.0e16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sigma_header(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(32)
        if len(raw) != 32:
            raise SystemExit(f"short sigma header: {path}")
        magic, version, n_levels, n_freq, nu_min, nu_max = struct.unpack(
            "<IIiidd", raw
        )
        flags = stream.read(n_levels)
        pad = (8 - n_levels % 8) % 8
        padding = stream.read(pad)
    offset = 32 + n_levels + pad
    return magic, version, n_levels, n_freq, nu_min, nu_max, flags, padding, offset


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".sh-grid.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    deck = args.deck.resolve()
    candidate = args.candidate.resolve()
    canonical = deck / "cmfgen_sigma_bf.bin"
    manifest_path = deck / "quarantine/manifest.json"
    provenance_path = deck / "DECK_PROVENANCE.json"
    if not candidate.is_file() or not canonical.is_file():
        raise SystemExit("candidate/canonical sigma asset missing")
    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if manifest.get("state") != "sealed":
        raise SystemExit(f"deck is not sealed: state={manifest.get('state')!r}")

    # Establish that the sealed deck has not drifted before changing it.
    for name, record in manifest.get("active_files", {}).items():
        path = deck / name
        if (not path.is_file() or path.stat().st_size != record["bytes"] or
                sha256(path) != record["sha256"]):
            raise SystemExit(f"sealed active file drifted before migration: {path}")

    old = sigma_header(canonical)
    new = sigma_header(candidate)
    n_levels_csv = 0
    with (deck / "levels.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        n_levels_csv = sum(1 for row in reader if row)
    if (new[0], new[1], new[2], new[3], new[4], new[5]) != (
            0x434D4644, 1, n_levels_csv, N_BINS, NU_MIN, NU_MAX):
        raise SystemExit(f"candidate header violates SH-GRID: {new[:6]}")
    if old[2] != new[2] or old[6] != new[6]:
        raise SystemExit("candidate changed level count or has_cmfgen flags")
    if any(new[7]):
        raise SystemExit("candidate has nonzero alignment padding")
    expected_size = new[8] + new[2] * new[3] * 8
    if candidate.stat().st_size != expected_size:
        raise SystemExit(
            f"candidate size={candidate.stat().st_size}, expected={expected_size}"
        )
    grid = np.memmap(candidate, dtype="<f8", mode="r", offset=new[8],
                     shape=(new[2], new[3]))
    if not np.isfinite(grid).all() or not (grid >= 0.0).all():
        raise SystemExit("candidate sigma contains nonfinite or negative values")
    del grid

    migration_dir = deck / "quarantine/sh_grid_migration_2026-08-08"
    backup = migration_dir / "cmfgen_sigma_bf.pre_sh_grid_1000.bin"
    if migration_dir.exists() or backup.exists():
        raise SystemExit(f"refusing to reuse migration archive: {migration_dir}")
    migration_dir.mkdir()

    old_record = {
        "path": str(backup.relative_to(deck)),
        "bytes": canonical.stat().st_size,
        "sha256": sha256(canonical),
        "header": {
            "n_levels": old[2], "n_freq_bins": old[3],
            "nu_min_hz": old[4], "nu_max_hz": old[5],
        },
    }
    new_record = {
        "path": canonical.name,
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
        "header": {
            "n_levels": new[2], "n_freq_bins": new[3],
            "nu_min_hz": new[4], "nu_max_hz": new[5],
        },
        "production": "fresh evaluation from linked CMFGEN phot data",
        "old_grid_padding": False,
        "first_bin_fill": False,
    }

    # Preserve the old asset inside the loader-forbidden archive, then make the
    # fully written candidate the one canonical root file.
    os.replace(canonical, backup)
    os.replace(candidate, canonical)

    provenance["frequency_grid"] = {
        "schema": "lumina-sh-grid-v1",
        "n_freq_bins": N_BINS,
        "nu_min_hz": NU_MIN,
        "nu_max_hz": NU_MAX,
        "sigma_sha256": new_record["sha256"],
        "migration_contract": "docs/SH_GRID_REOPEN_CONTRACT_2026-08-08.md",
    }
    write_json_atomic(provenance_path, provenance)

    manifest.setdefault("grid_migrations", []).append({
        "schema": "lumina-sh-grid-migration-v1",
        "date": "2026-08-08",
        "old": old_record,
        "new": new_record,
    })
    manifest["active_files"]["cmfgen_sigma_bf.bin"] = {
        "bytes": canonical.stat().st_size,
        "sha256": sha256(canonical),
    }
    manifest["active_files"]["DECK_PROVENANCE.json"] = {
        "bytes": provenance_path.stat().st_size,
        "sha256": sha256(provenance_path),
    }
    write_json_atomic(manifest_path, manifest)
    print(
        "[SH-GRID][PROMOTE][PASS] "
        f"canonical={canonical} sha256={new_record['sha256']} "
        f"recoverable_old={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
