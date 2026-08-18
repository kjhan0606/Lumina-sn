#!/usr/bin/env python3
"""Atomically promote the verified 1234-bin SH-GRID upper-closure asset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np


H_PLANCK = 6.62607015e-27
EV_TO_ERG = 1.602176634e-12
OLD_N_BINS = 1178
OLD_NU_MAX = 3.0e16
N_BINS = 1234
NU_MIN = 5.8412785919616062e13
NU_MAX = 4.0362581455823112e16


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
    temporary = path.with_suffix(path.suffix + ".upper-grid.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
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

    for name, record in manifest.get("active_files", {}).items():
        path = deck / name
        if (not path.is_file() or path.stat().st_size != record["bytes"] or
                sha256(path) != record["sha256"]):
            raise SystemExit(f"sealed active file drifted before migration: {path}")

    old = sigma_header(canonical)
    new = sigma_header(candidate)
    with (deck / "levels.csv").open(newline="") as stream:
        levels = list(csv.DictReader(stream))
    n_levels_csv = len(levels)
    if (old[0], old[1], old[2], old[3], old[4], old[5]) != (
            0x434D4644, 1, n_levels_csv, OLD_N_BINS, NU_MIN, OLD_NU_MAX):
        raise SystemExit(f"canonical is not the sealed 1178-bin predecessor: {old[:6]}")
    if (new[0], new[1], new[2], new[3], new[4], new[5]) != (
            0x434D4644, 1, n_levels_csv, N_BINS, NU_MIN, NU_MAX):
        raise SystemExit(f"candidate header violates upper SH-GRID: {new[:6]}")
    with (deck / "ionization_energies.csv").open(newline="") as stream:
        ionization = {
            (int(row["atomic_number"]), int(row["ion_number"])):
                float(row["ionization_energy_eV"])
            for row in csv.DictReader(stream)
        }
    newly_covered_flags = []
    for index, (was, now) in enumerate(zip(old[6], new[6])):
        if was == now:
            continue
        row = levels[index]
        key = (int(row["atomic_number"]), int(row["ion_number"]))
        threshold_eV = ionization[key] - float(row["energy_eV"])
        nu_threshold = threshold_eV * EV_TO_ERG / H_PLANCK
        if not (was == 0 and now == 1 and
                OLD_NU_MAX <= nu_threshold < NU_MAX):
            raise SystemExit(
                "candidate changed has_cmfgen outside newly covered band: "
                f"global={index} old={was} new={now} nu={nu_threshold:.17g}"
            )
        newly_covered_flags.append({
            "global_level": index,
            "atomic_number": key[0],
            "ion_number": key[1],
            "level_number": int(row["level_number"]),
            "nu_threshold_hz": nu_threshold,
        })
    if any(old[7]) or any(new[7]):
        raise SystemExit("sigma asset has nonzero alignment padding")
    for path, header in ((canonical, old), (candidate, new)):
        expected_size = header[8] + header[2] * header[3] * 8
        if path.stat().st_size != expected_size:
            raise SystemExit(f"bad sigma extent: {path}")

    old_grid = np.memmap(canonical, dtype="<f8", mode="r", offset=old[8],
                         shape=(old[2], old[3]))
    new_grid = np.memmap(candidate, dtype="<f8", mode="r", offset=new[8],
                         shape=(new[2], new[3]))
    if not np.isfinite(new_grid).all() or not (new_grid >= 0.0).all():
        raise SystemExit("candidate sigma contains nonfinite or negative values")
    prefix = new_grid[:, :OLD_N_BINS]
    if not np.array_equal(old_grid > 0.0, prefix > 0.0):
        changed = int(np.count_nonzero((old_grid > 0.0) != (prefix > 0.0)))
        raise SystemExit(f"candidate changed {changed} predecessor support cells")
    common = (old_grid > 0.0) & (prefix > 0.0)
    predecessor_changed = int(np.count_nonzero(old_grid != prefix))
    predecessor_max_rel = float(np.max(
        np.abs(old_grid[common] - prefix[common]) /
        np.maximum(np.abs(old_grid[common]), np.abs(prefix[common]))
    )) if np.any(common) else 0.0
    if predecessor_max_rel > 1.0e-7:
        raise SystemExit(
            "candidate predecessor-grid drift exceeds coordinate-repair bound: "
            f"max_rel={predecessor_max_rel:.17g}"
        )
    high_nonzero = int(np.count_nonzero(new_grid[:, OLD_N_BINS:] > 0.0))
    del old_grid, new_grid

    migration_dir = deck / "quarantine/sh_grid_upper_closure_2026-08-08"
    backup = migration_dir / "cmfgen_sigma_bf.pre_upper_closure_1178.bin"
    if migration_dir.exists() or backup.exists():
        raise SystemExit(f"refusing to reuse migration archive: {migration_dir}")
    migration_dir.mkdir()

    old_record = {
        "path": str(backup.relative_to(deck)),
        "bytes": canonical.stat().st_size,
        "sha256": sha256(canonical),
        "header": {"n_levels": old[2], "n_freq_bins": old[3],
                   "nu_min_hz": old[4], "nu_max_hz": old[5]},
    }
    new_record = {
        "path": canonical.name,
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
        "header": {"n_levels": new[2], "n_freq_bins": new[3],
                   "nu_min_hz": new[4], "nu_max_hz": new[5]},
        "production": "fresh evaluation from linked CMFGEN phot data",
        "predecessor_support_bit_identical": True,
        "predecessor_changed_cells": predecessor_changed,
        "predecessor_max_relative_change": predecessor_max_rel,
        "newly_covered_cmfgen_rows": newly_covered_flags,
        "high_band_nonzero_cells": high_nonzero,
        "row_padding_or_tail_fill": False,
    }

    os.replace(canonical, backup)
    os.replace(candidate, canonical)

    provenance["frequency_grid"] = {
        "schema": "lumina-sh-grid-upper-closure-v1",
        "n_freq_bins": N_BINS,
        "nu_min_hz": NU_MIN,
        "nu_max_hz": NU_MAX,
        "sigma_sha256": new_record["sha256"],
        "sigma_evaluator":
            "CMFGEN SUB_PHOT_GEN exact types 1/2/3/7/8/9/20/21/22",
        "migration_contract":
            "docs/SH_GRID_UPPER_CLOSURE_CONTRACT_2026-08-08.md",
    }
    write_json_atomic(provenance_path, provenance)

    manifest.setdefault("grid_migrations", []).append({
        "schema": "lumina-sh-grid-upper-closure-migration-v1",
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
        "[SH-GRID][UPPER-PROMOTE][PASS] "
        f"canonical={canonical} sha256={new_record['sha256']} "
        f"predecessor_support_bit_identical=1 "
        f"predecessor_changed={predecessor_changed} "
        f"predecessor_max_rel={predecessor_max_rel:.9g} "
        f"newly_covered_cmfgen_rows={len(newly_covered_flags)} "
        f"high_band_nonzero={high_nonzero} "
        f"recoverable_old={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
