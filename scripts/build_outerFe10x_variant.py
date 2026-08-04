#!/usr/bin/env python3
"""outerFe 10× variant — column density hypothesis test.

기존 outerFe (X_Fe=0.002-0.005) 가 UVbl을 오히려 0.96→1.31로 악화시킴.
가설 (A): 외곽 ρ ∝ r⁻⁷이므로 0.005 mass fraction 으로는 τ<1, blanketing 발생 못함.
검증: X_Fe를 10× 올려 (0.02-0.05) τ≥1 가능한 column density 확보.

기존: 0.005,0.005,0.005,0.004,0.004,0.003,0.002 (shells 8-14)
신규: 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
BASE = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19"
OUT  = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19_outerFe10x"

FE_PROFILE = {
    8:  0.05,
    9:  0.05,
    10: 0.05,
    11: 0.04,
    12: 0.04,
    13: 0.03,
    14: 0.02,
}

SYMLINK = [
    "atom_masses.csv", "electron_densities.csv", "ion_number_density.npy",
    "ionization_energies.csv", "j_blues.npy", "levels.csv",
    "line2macro_level_upper.npy", "line_interaction_id.npy",
    "line_list.csv", "macro_atom_data.csv", "macro_atom_references.csv",
    "mc_estimators.csv", "plasma_state.csv", "spectrum_real.csv",
    "spectrum_virtual.csv", "tau_sobolev.npy", "tau_sobolev_stats.csv",
    "kshape_contract.txt",
    "transition_probabilities.npy", "zeta_data.npy", "zeta_ions.csv",
    "zeta_temps.csv", "density.csv", "geometry.csv",
]


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Building {OUT.name}")
    print(f"  Base: {BASE.name}")

    for f in SYMLINK:
        src = BASE / f
        if not src.exists():
            print(f"  WARN missing {src}")
            continue
        dst = OUT / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    ab = pd.read_csv(BASE / "abundances.csv").set_index("atomic_number")
    ab.columns = ab.columns.astype(int)

    print("\n  Adding outer Fe 10× (subtracting from O):")
    print(f"  {'shell':>5} {'v_kms':>7} {'X_Fe_old':>10} {'X_Fe_new':>10} {'X_O_old':>10} {'X_O_new':>10}")
    g = pd.read_csv(BASE / "geometry.csv")
    for s, dfe in FE_PROFILE.items():
        v = g.v_inner.iloc[s] / 1e5
        x_fe_old = ab.loc[26, s]
        x_o_old  = ab.loc[8, s]
        x_fe_new = x_fe_old + dfe
        x_o_new  = x_o_old - dfe
        if x_o_new < 0.05:
            raise ValueError(f"shell {s}: O too low ({x_o_new:.3f}) after Fe addition")
        ab.loc[26, s] = x_fe_new
        ab.loc[8, s]  = x_o_new
        print(f"  {s:>5d} {v:>7.0f} {x_fe_old:>10.2e} {x_fe_new:>10.2e} {x_o_old:>10.2e} {x_o_new:>10.2e}")

    ab.reset_index().to_csv(OUT / "abundances.csv", index=False)
    print(f"\n  Wrote {OUT / 'abundances.csv'}")

    src_npy = BASE / "abundances.npy"
    if src_npy.exists():
        ab_arr = np.load(src_npy)
        ab_full = pd.read_csv(OUT / "abundances.csv").set_index("atomic_number")
        ab_full.columns = ab_full.columns.astype(int)
        n_shells = ab_arr.shape[1]
        n_z = ab_arr.shape[0]
        new_arr = np.zeros_like(ab_arr)
        for Z in ab_full.index:
            if 1 <= Z <= n_z:
                new_arr[Z-1, :] = ab_full.loc[Z].values[:n_shells]
        np.save(OUT / "abundances.npy", new_arr)
        print(f"  Wrote {OUT / 'abundances.npy'}  (regenerated from CSV)")

    cfg_src = BASE / "config.json"
    if cfg_src.exists():
        shutil.copy(cfg_src, OUT / "config.json")
    print(f"\nDone. Variant ready: {OUT}")


if __name__ == "__main__":
    main()
