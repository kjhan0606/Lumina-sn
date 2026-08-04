#!/usr/bin/env python3
"""Create the CMFGEN full-coverage deck without touching the provenance deck.

This driver must be run on a CPU compute node with both
CMFGEN_FULL_LEVELS=1 and CMFGEN_SUPER_LEVELS=1.  It deliberately refuses to
reuse an existing output directory.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"
NEW = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_fullcov"
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"

# Produced again by expand/finalize/collision/recombination builders.
REBUILT = {
    "atom_masses.csv",
    "atomic_data_cmfgen.h5",
    "cmfgen_sigma_bf.bin",
    "coldata_cmfgen_manifest.csv",
    "ionization_energies.csv",
    "level_multiplicity.csv",
    "levels.csv",
    "line2macro_level_upper.npy",
    "line_list.csv",
    "kshape_contract.txt",
    "ma_radrecomb_target.bin",
    "ma_radrecomb_target_manifest.csv",
    "macro_atom_data.csv",
    "macro_atom_references.csv",
    "transition_probabilities.npy",
    "tau_sobolev.npy",
    "zeta_data.npy",
    "zeta_ions.csv",
    "zeta_temps.csv",
}


def validate_composition_shape(deck: Path) -> None:
    with (deck / "abundances.csv").open(newline="") as stream:
        header = next(csv.reader(stream), [])
    abundance_columns = max(len(header) - 1, 0)
    with (deck / "geometry.csv").open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        geometry_rows = sum(1 for row in rows if row)
    if abundance_columns != geometry_rows:
        raise SystemExit(
            f"abundances/geometry shape mismatch in {deck}: "
            f"abundance_columns={abundance_columns}, geometry_rows={geometry_rows}"
        )


def load_expand_module():
    spec = importlib.util.spec_from_file_location("deck_expand", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_companions() -> None:
    """Copy, never symlink, companion files not rebuilt by the pipeline."""
    for source in sorted(OLD.iterdir()):
        if source.name in REBUILT or source.name.startswith("ige_col_"):
            continue
        target = NEW / source.name
        if target.exists():
            continue
        resolved = source.resolve(strict=True)
        if resolved.is_file():
            shutil.copy2(resolved, target)


def main() -> None:
    if os.environ.get("CMFGEN_FULL_LEVELS") != "1":
        raise SystemExit("CMFGEN_FULL_LEVELS=1 is required")
    if os.environ.get("CMFGEN_SUPER_LEVELS") != "1":
        raise SystemExit("CMFGEN_SUPER_LEVELS=1 is required")
    if NEW.exists():
        raise SystemExit(f"refusing to overwrite existing output: {NEW}")
    if not OLD.is_dir():
        raise SystemExit(f"missing provenance deck: {OLD}")

    module = load_expand_module()
    module.OUT_DIR = NEW
    module.OUT_H5 = NEW / "atomic_data_cmfgen.h5"
    module.OUT_SIGMA_BIN = NEW / "cmfgen_sigma_bf.bin"
    module.main()
    copy_companions()
    validate_composition_shape(NEW)
    print(f"created core full-coverage deck: {NEW}")


if __name__ == "__main__":
    main()
