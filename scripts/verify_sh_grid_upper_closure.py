#!/usr/bin/env python3
"""Fail-closed static gate for the 1234-bin SH-GRID upper closure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct

import numpy as np


H_PLANCK = 6.62607015e-27
EV_TO_ERG = 1.602176634e-12
C_ANGSTROM_HZ = 2.99792458e18
OLD_MIN = 1.5e14
OLD_MAX = 3.0e16
OLD_N = 1000
PREDECESSOR_N = 1178
NEW_MIN = 5.8412785919616062e13
NEW_MAX = 4.0362581455823112e16
NEW_N = 1234
MAGIC = 0x434D4644
VERSION = 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_sigma(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(32)
        if len(raw) != 32:
            raise ValueError(f"short sigma header: {path}")
        header = struct.unpack("<IIiidd", raw)
        flags = np.frombuffer(stream.read(header[2]), dtype="i1").copy()
        pad = (8 - header[2] % 8) % 8
        padding = stream.read(pad)
    offset = 32 + header[2] + pad
    expected = offset + header[2] * header[3] * 8
    if path.stat().st_size != expected:
        raise ValueError(f"bad sigma extent: got={path.stat().st_size} expected={expected}")
    grid = np.memmap(path, dtype="<f8", mode="r", offset=offset,
                     shape=(header[2], header[3]))
    return header, flags, padding, grid


def active_elements(rows: list[dict[str, str]]) -> set[int]:
    active = set()
    for row in rows:
        z = int(row["atomic_number"])
        if any(float(value) > 0.0 for key, value in row.items()
               if key != "atomic_number"):
            active.add(z)
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck", type=Path,
        default=Path("data/tardis_reference_toy06_19p48d_sivcaiv_active")
    )
    args = parser.parse_args()
    deck = args.deck
    try:
        levels = read_csv(deck / "levels.csv")
        ions = read_csv(deck / "ionization_energies.csv")
        active = active_elements(read_csv(deck / "abundances.csv"))
        header, flags, padding, grid = read_sigma(deck / "cmfgen_sigma_bf.bin")
        provenance = json.loads((deck / "DECK_PROVENANCE.json").read_text())
        manifest = json.loads((deck / "quarantine/manifest.json").read_text())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[SH-GRID][UPPER-VERIFY][ERROR] {exc}")
        return 2

    expected_header = (MAGIC, VERSION, len(levels), NEW_N, NEW_MIN, NEW_MAX)
    if header != expected_header or any(padding):
        print(f"[SH-GRID][UPPER-VERIFY][FAIL] header={header} expected={expected_header}")
        return 3
    if not np.isfinite(grid).all() or not (grid >= 0.0).all():
        print("[SH-GRID][UPPER-VERIFY][FAIL] nonfinite_or_negative_sigma")
        return 3

    old_dlog = math.log(OLD_MAX / OLD_MIN) / OLD_N
    new_dlog = math.log(NEW_MAX / NEW_MIN) / NEW_N
    if struct.pack("<d", old_dlog) != struct.pack("<d", new_dlog):
        print("[SH-GRID][UPPER-VERIFY][FAIL] dlog_not_bit_identical")
        return 3
    aligned_max = NEW_MIN * math.exp(NEW_N * old_dlog)
    if aligned_max != NEW_MAX:
        print("[SH-GRID][UPPER-VERIFY][FAIL] upper_edge_not_aligned")
        return 3

    ion_eV = {
        (int(row["atomic_number"]), int(row["ion_number"])):
            float(row["ionization_energy_eV"])
        for row in ions
    }
    above_old = []
    outside_new = []
    for index, row in enumerate(levels):
        z = int(row["atomic_number"])
        ion = int(row["ion_number"])
        if z not in active or ion < 1:
            continue
        energy = ion_eV.get((z, ion), math.nan) - float(row["energy_eV"])
        if not math.isfinite(energy) or energy <= 0.0:
            continue
        nu = energy * EV_TO_ERG / H_PLANCK
        witness = (index, z, ion, int(row["level_number"]), energy, nu)
        if nu >= OLD_MAX:
            above_old.append(witness)
        if nu >= NEW_MAX:
            outside_new.append(witness)
    if len(above_old) != 1 or outside_new:
        print("[SH-GRID][UPPER-VERIFY][FAIL] "
              f"above_old={len(above_old)} outside_new={len(outside_new)}")
        return 3
    witness = above_old[0]
    if witness[:4] != (333, 14, 4, 0) or flags[333] != 1:
        print(f"[SH-GRID][UPPER-VERIFY][FAIL] witness={witness} flag={flags[333]}")
        return 3
    si_high = np.asarray(grid[333, PREDECESSOR_N:])
    if not np.any(si_high > 0.0) or np.any(np.asarray(grid[333, :PREDECESSOR_N]) != 0.0):
        print("[SH-GRID][UPPER-VERIFY][FAIL] SiV_support_not_confined_to_new_band")
        return 3

    high = np.asarray(grid[:, PREDECESSOR_N:])
    high_nonzero = int(np.count_nonzero(high > 0.0))
    migrations = manifest.get("grid_migrations", [])
    upper = [item for item in migrations
             if item.get("schema") == "lumina-sh-grid-upper-closure-migration-v1"]
    freq = provenance.get("frequency_grid", {})
    if (len(upper) != 1 or
            upper[0]["new"].get("high_band_nonzero_cells") != high_nonzero or
            len(upper[0]["new"].get("newly_covered_cmfgen_rows", [])) != 1 or
            freq.get("n_freq_bins") != NEW_N or
            freq.get("nu_min_hz") != NEW_MIN or
            freq.get("nu_max_hz") != NEW_MAX):
        print("[SH-GRID][UPPER-VERIFY][FAIL] manifest_or_provenance_mismatch")
        return 3

    print(
        "[SH-GRID][UPPER-VERIFY][PASS] "
        f"levels={len(levels)} bins={NEW_N} dlog={new_dlog:.17g} "
        f"range=[{NEW_MIN:.17g},{NEW_MAX:.17g}] "
        f"active_above_old={len(above_old)} active_outside_new=0 "
        f"high_nonzero={high_nonzero} newly_cmfgen=1"
    )
    print(
        "[SH-GRID][UPPER-VERIFY][WITNESS] "
        f"global={witness[0]} Z={witness[1]} ion={witness[2]} "
        f"level={witness[3]} threshold_eV={witness[4]:.13g} "
        f"nu={witness[5]:.17g} lambda_A={C_ANGSTROM_HZ / witness[5]:.17g} "
        f"sigma_high_max={float(np.max(si_high)):.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
