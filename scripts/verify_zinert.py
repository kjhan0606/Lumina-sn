#!/usr/bin/env python3
"""Canonical metadata census for L0-CLOSE-R2 section 3.7 Z-INERT."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = ROOT / "data/tardis_reference_toy06_19p48d"


def rows(path: Path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    args = parser.parse_args()
    deck = args.deck.resolve()

    topology = [int(row["atomic_number"]) for row in rows(deck / "atom_masses.csv")]
    abundance_by_z: dict[int, list[float]] = {}
    for row in rows(deck / "abundances.csv"):
        z = int(row.pop("atomic_number"))
        abundance_by_z[z] = [float(value) for value in row.values()]
    n_shells = len(next(iter(abundance_by_z.values())))
    inactive = {
        z for z in topology
        if all(value == 0.0 for value in abundance_by_z.get(z, [0.0] * n_shells))
    }
    active = set(topology) - inactive

    ionization = Counter(int(row["atomic_number"])
                         for row in rows(deck / "ionization_energies.csv"))
    ion_slots = {z: ionization[z] + 1 for z in topology}
    levels: dict[tuple[int, int], set[int]] = defaultdict(set)
    level_count = Counter()
    for row in rows(deck / "levels.csv"):
        z = int(row["atomic_number"])
        ion = int(row["ion_number"])
        level = int(row["level_number"])
        levels[(z, ion)].add(level)
        level_count[z] += 1

    line_count = Counter()
    missing_ion = Counter()
    missing_level = Counter()
    for row in rows(deck / "line_list.csv"):
        z = int(row["atomic_number"])
        ion = int(row["ion_number"])
        lower = int(row["level_number_lower"])
        upper = int(row["level_number_upper"])
        key = (z, ion)
        line_count[z] += 1
        if key not in levels:
            missing_ion[z] += 1
        elif lower not in levels[key] or upper not in levels[key]:
            missing_level[z] += 1

    print(f"[Z-INERT-CANONICAL] deck={deck} n_shells={n_shells} "
          f"inactive_Z={','.join(map(str, sorted(inactive)))} "
          f"active_Z={','.join(map(str, sorted(active)))}")
    for z in sorted(inactive):
        print(f"[Z-INERT-CANONICAL] Z={z} ion_slots={ion_slots[z]} "
              f"levels={level_count[z]} lines={line_count[z]} "
              f"input_nonzero=0 missing_ion_lines={missing_ion[z]} "
              f"missing_level_lines={missing_level[z]}")

    active_missing = sum(missing_ion[z] + missing_level[z] for z in active)
    inactive_missing = sum(missing_ion[z] + missing_level[z] for z in inactive)
    manifest_path = deck / "quarantine/manifest.json"
    sealed_active_only = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            sealed_active_only = (
                manifest.get("state") == "sealed" and
                "active_ions.csv" in manifest.get("active_files", {}) and
                (deck / "quarantine/source_deck_snapshot").is_dir()
            )
        except (OSError, ValueError):
            sealed_active_only = False
    print(f"[Z-INERT-ACTIVE-MAP] active_lines={sum(line_count[z] for z in active)} "
          f"missing_ion_or_level={active_missing} verdict="
          f"{'PASS' if active_missing == 0 else 'FAIL'}")
    print(f"[Z-INERT-INACTIVE-MAP] inactive_lines="
          f"{sum(line_count[z] for z in inactive)} "
          f"missing_ion_or_level={inactive_missing}")
    print(f"[Z-INERT-DECK-MODE] mode="
          f"{'sealed-active-only' if sealed_active_only else 'full-topology'}")
    return 0 if (inactive or sealed_active_only) and active_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
