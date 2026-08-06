#!/usr/bin/env python3
"""I20 수리 판정런 — 진공파장 덱을 생성한다 (기존 덱은 덮어쓰지 않는다).

`deck_regen_r4_ftos_driver.py` 와 **입력이 완전히 같다**: 같은 provenance 덱
(`_sivcaiv_links`), 같은 `CMFGEN_LINKS`(jnu4 런의 atomic_links.txt), 같은 env.
달라지는 것은 `expand_atomic_data_cmfgen.build_lines` 와
`finalize_cmfgen_ref_npy` 의 I20 수리분뿐이다 —
즉 이 덱과 `_ftos` 의 차이는 **I20 하나로 격리**된다.

계약·기대 변경집합: docs/I20_AIR_WAVELENGTH_REPAIR_CONTRACT.md

주의: O-PHYS 런은 20개 이온에서 다른 vintage(19apr23)를 쓴다. 그것은 별도 판정
사안이므로 여기서 섞지 않는다 — 이 덱은 `_ftos` 와 동일 vintage 집합이다.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"
REF = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"   # A/B 상대
NEW = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_vac"
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"

REBUILT = {
    "atom_masses.csv",
    "atomic_data_cmfgen.h5",
    "atomic_vintage_manifest.csv",
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
    "verification.log",
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
    spec = importlib.util.spec_from_file_location("deck_expand_r5", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_companions() -> None:
    """Copy immutable model companions; runtime atomic sidecars are rebuilt."""
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
    required = {
        "CMFGEN_FULL_LEVELS": "1",
        "CMFGEN_SUPER_LEVELS": "1",
        "CMFGEN_LINK_FTOS": "1",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise SystemExit(f"{name}={expected} is required")
    links = os.environ.get("CMFGEN_LINKS", "")
    if not links or not Path(links).is_file():
        raise SystemExit(f"CMFGEN_LINKS must name an existing atomic_links.txt: {links}")
    if NEW.exists():
        raise SystemExit(f"refusing to overwrite existing output: {NEW}")
    if not OLD.is_dir():
        raise SystemExit(f"missing R1 link-pinned provenance deck: {OLD}")
    if not REF.is_dir():
        raise SystemExit(f"missing A/B reference deck: {REF}")

    module = load_expand_module()
    if not module.LINK_FTOS_ENABLED:
        raise SystemExit("expand module did not arm CMFGEN_LINK_FTOS")
    # I20 수리분이 실제로 적재됐는지 확인 — 구 코드로 덱을 만들면 안 된다.
    if not hasattr(module, "A_PREFACTOR"):
        raise SystemExit("expand module lacks A_PREFACTOR — I20 repair not present")
    module.ROOT = ROOT
    module.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    module.OUT_DIR = NEW
    module.OUT_H5 = NEW / "atomic_data_cmfgen.h5"
    module.OUT_SIGMA_BIN = NEW / "cmfgen_sigma_bf.bin"
    module.main()
    copy_companions()
    validate_composition_shape(NEW)
    print(f"created I20 vacuum-wavelength deck: {NEW}")


if __name__ == "__main__":
    main()
