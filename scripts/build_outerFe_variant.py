#!/usr/bin/env python3
"""HST 진단선 분석에서 도출된 GAP에 따라 champion ref에 outer Fe를 추가.

GAP (project_hst_diagnostic_zone_unwind.md):
  Fe II 2382/2600 v=15-16k → shells 10-12, X_Fe=1e-6 → 데이터 요구 ~10⁴×
  Fe II 5169 v=17k → shell 14, X_Fe=1e-6
  Fe III 5129 v=15k → shell 10

전략: shells 8-14에 Fe 추가, outward로 감소. O에서 mass 차감 (O가 filler 0.55).
다른 모든 데이터 (geometry, levels, line_list, atomic, etc.) 는 BASE에서 symlink.

기존 stratV2D ("Fe-both 0.06") 와의 차이: stratV2D는 X_Fe=0.06 일정 추가, 본 변형은
HST 데이터 depth 비례 점진 감소 (0.005 → 0.002).
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd, json

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
BASE = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19"
OUT  = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19_outerFe"

# Shell index → X_Fe to add (depth-motivated profile, not flat)
# Fe II 2382 (depth 0.68 at v=16.5k) is strongest → put more around shell 12-13
FE_PROFILE = {
    8:  0.005,   # v=14k
    9:  0.005,   # v=14.5k
    10: 0.005,   # v=15k Fe II 2600 absorber
    11: 0.004,
    12: 0.004,   # v=16k Fe II 2382 absorber
    13: 0.003,
    14: 0.002,   # v=17k Fe II 5169 absorber
}

# Atomic-data files: symlink unchanged
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

    # Symlink unchanged files
    for f in SYMLINK:
        src = BASE / f
        if not src.exists():
            print(f"  WARN missing {src}")
            continue
        dst = OUT / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    # Modify abundances.csv
    ab = pd.read_csv(BASE / "abundances.csv")
    ab = ab.set_index("atomic_number")
    ab.columns = ab.columns.astype(int)

    print("\n  Adding outer Fe (subtracting from O):")
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

    # Copy/update abundances.npy if exists (TARDIS sometimes uses npy)
    src_npy = BASE / "abundances.npy"
    if src_npy.exists():
        ab_arr = np.load(src_npy)
        # ab_arr typically shape (n_elements, n_shells); index alignment matters
        # Safer: regenerate from CSV
        ab_full = pd.read_csv(OUT / "abundances.csv").set_index("atomic_number")
        ab_full.columns = ab_full.columns.astype(int)
        # Match original npy shape — assume (Z_max, n_shells) with Z=1..30 rows
        n_shells = ab_arr.shape[1]
        n_z = ab_arr.shape[0]
        new_arr = np.zeros_like(ab_arr)
        for Z in ab_full.index:
            if 1 <= Z <= n_z:
                new_arr[Z-1, :] = ab_full.loc[Z].values[:n_shells]
        np.save(OUT / "abundances.npy", new_arr)
        print(f"  Wrote {OUT / 'abundances.npy'}  (regenerated from CSV)")

    # Copy config.json
    cfg_src = BASE / "config.json"
    if cfg_src.exists():
        shutil.copy(cfg_src, OUT / "config.json")
    print(f"\nDone. Variant ready: {OUT}")


if __name__ == "__main__":
    main()
