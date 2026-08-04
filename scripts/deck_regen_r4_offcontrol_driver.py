#!/usr/bin/env python3
"""Build an ephemeral gate-OFF control for byte comparison with _links."""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"

REBUILT = {
    "atom_masses.csv", "atomic_data_cmfgen.h5",
    "atomic_vintage_manifest.csv", "cmfgen_sigma_bf.bin",
    "coldata_cmfgen_manifest.csv", "ionization_energies.csv",
    "level_multiplicity.csv", "levels.csv", "line2macro_level_upper.npy",
    "line_list.csv", "kshape_contract.txt", "ma_radrecomb_target.bin",
    "ma_radrecomb_target_manifest.csv", "macro_atom_data.csv",
    "macro_atom_references.csv", "transition_probabilities.npy",
    "tau_sobolev.npy", "verification.log", "zeta_data.npy",
    "zeta_ions.csv", "zeta_temps.csv",
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
    spec = importlib.util.spec_from_file_location("deck_expand_r4_off", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    output_text = os.environ.get("R4_OFF_DIR", "")
    if not output_text:
        raise SystemExit("R4_OFF_DIR is required")
    output = Path(output_text)
    if output.exists():
        raise SystemExit(f"refusing to overwrite OFF control: {output}")
    if os.environ.get("CMFGEN_LINK_FTOS", "0").strip().lower() \
            not in ("", "0", "false"):
        raise SystemExit("OFF control requires CMFGEN_LINK_FTOS=0/unset")
    if os.environ.get("CMFGEN_FULL_LEVELS") != "1" or \
            os.environ.get("CMFGEN_SUPER_LEVELS") != "1":
        raise SystemExit("OFF control requires FULL_LEVELS=1 and SUPER_LEVELS=1")
    links = os.environ.get("CMFGEN_LINKS", "")
    if not links or not Path(links).is_file():
        raise SystemExit(f"CMFGEN_LINKS must exist: {links}")
    if not OLD.is_dir():
        raise SystemExit(f"missing _links byte oracle: {OLD}")

    module = load_expand_module()
    if module.LINK_FTOS_ENABLED:
        raise SystemExit("OFF control unexpectedly armed LINK_FTOS_ENABLED")
    module.ROOT = ROOT
    module.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    module.OUT_DIR = output
    module.OUT_H5 = output / "atomic_data_cmfgen.h5"
    module.OUT_SIGMA_BIN = output / "cmfgen_sigma_bf.bin"
    module.main()

    for source in sorted(OLD.iterdir()):
        if source.name in REBUILT or source.name.startswith("ige_col_"):
            continue
        target = output / source.name
        if target.exists():
            continue
        resolved = source.resolve(strict=True)
        if resolved.is_file():
            shutil.copy2(resolved, target)
    validate_composition_shape(output)
    print(f"created ephemeral R4 gate-OFF control: {output}")


if __name__ == "__main__":
    main()
