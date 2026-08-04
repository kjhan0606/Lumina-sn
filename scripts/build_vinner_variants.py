#!/usr/bin/env python3
"""Create v_inner sensitivity variants of the champion ref dir.

Strategy: keep r_inner/r_outer, density, abundances, atomic data unchanged.
Scale all per-shell velocities by `target_kms / 10000` (current v_inner_min).
This is an approximate Doppler-only sensitivity probe; not strict homologous.
"""
import json, shutil
import numpy as np, pandas as pd
from pathlib import Path

ROOT  = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
BASE  = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19"
TARGETS_KMS = [11500, 13000]  # 10000 = baseline (use existing BASE)

ATOMIC_FILES = [  # symlink to BASE — atomic data identical
    "abundances.csv", "abundances.npy", "atom_masses.csv",
    "electron_densities.csv", "ion_number_density.npy",
    "ionization_energies.csv", "j_blues.npy", "levels.csv",
    "line2macro_level_upper.npy", "line_interaction_id.npy",
    "line_list.csv", "macro_atom_data.csv", "macro_atom_references.csv",
    "mc_estimators.csv", "plasma_state.csv", "spectrum_real.csv",
    "spectrum_virtual.csv", "tau_sobolev.npy", "tau_sobolev_stats.csv",
    "kshape_contract.txt",
    "transition_probabilities.npy", "zeta_data.npy", "zeta_ions.csv",
    "zeta_temps.csv", "density.csv",
]


def make_variant(target_kms: int) -> Path:
    out = ROOT / f"data/tardis_reference_strat6_higherL_aulboost_L19_v{target_kms}k"
    out.mkdir(exist_ok=True)
    scale = target_kms / 10000.0
    print(f"\n=== v_inner = {target_kms} km/s  (scale={scale:.4f})  ->  {out.name}")

    # symlink atomic + structural-density files (unchanged)
    for f in ATOMIC_FILES:
        src = BASE / f
        if not src.exists():
            print(f"  WARN missing source {src}")
            continue
        dst = out / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # resolve through BASE's own symlinks so the new variant points at the
        # ultimate file (BASE itself uses symlinks into strat6_higherL_aulboost)
        dst.symlink_to(src.resolve())

    # rewrite geometry.csv: scale r AND v (LUMINA uses r_inner/r_outer for ray
    # tracing; v_inner/v_outer columns are read but only used in rescale_epoch.
    # v=r/t with t fixed → scaling r is equivalent to scaling v.)
    g = pd.read_csv(BASE / "geometry.csv")
    g["r_inner"] = g["r_inner"].astype(float) * scale
    g["r_outer"] = g["r_outer"].astype(float) * scale
    g["v_inner"] = g["v_inner"].astype(float) * scale
    g["v_outer"] = g["v_outer"].astype(float) * scale
    g.to_csv(out / "geometry.csv", index=False)
    print(f"  geometry.csv: shell 0 r_inner={g['r_inner'].iloc[0]:.3e}  v_inner={g['v_inner'].iloc[0]:.3e}")
    print(f"               shell 29 r_outer={g['r_outer'].iloc[-1]:.3e}  v_outer={g['v_outer'].iloc[-1]:.3e}")

    # rewrite config.json with scaled v_inner_min/v_outer_max
    cfg = json.loads((BASE / "config.json").read_text())
    cfg["v_inner_min_cm_s"] = float(cfg["v_inner_min_cm_s"]) * scale
    cfg["v_outer_max_cm_s"] = float(cfg["v_outer_max_cm_s"]) * scale
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  config.json: v_inner_min={cfg['v_inner_min_cm_s']:.2e}  "
          f"v_outer_max={cfg['v_outer_max_cm_s']:.2e}")
    return out


for t in TARGETS_KMS:
    make_variant(t)
print("\nDone. Variants created. Baseline v_inner=10000 km/s = existing BASE.")
