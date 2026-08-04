#!/usr/bin/env python3
"""Task #219 opt 3: AUTOSTRUCTURE-derived ζ_k(T) for iron-peak III placeholders.

Parses AS plan-D `adasout` target-level tables for Cr III, Co III, Ni III and
computes ζ(T) = g_ground / Σ g_α exp(-E_α / kT) at the 20 LUMINA temperatures.

Patches rows (Z,ion=2) for Z ∈ {24, 27, 28} in `zeta_data.npy`. Cr III is patched
for completeness (its current value 0.7415 is not the 0.2908 Kurucz placeholder
but is a Carsus extrapolation that AS should improve on). Co III and Ni III at
ion=2 currently sit at the 0.2908 Ti-III placeholder — main correction target.

Writes a new reference dir alongside the input.
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

RY_TO_K = 157887.66  # 1 Ry / k_B in Kelvin

AS_RUNS = {
    24: "/gpfs/kjhan/autostructure_runs/cr3_dr_planD_mxconf5/adasout",
    27: "/gpfs/kjhan/autostructure_runs/co3_dr_planD_mxconf5/adasout",
    28: "/gpfs/kjhan/autostructure_runs/ni3_dr_planD_mxconf5/adasout",
}

def parse_adasout_levels(path):
    """Return list of (g, E_Ry, P, S2p1, L) tuples from the target-level table in adasout.

    Columns in adasout: `2J  P  (2S+1)  L  E1C(RYD)`. g = 2J + 1.
    """
    rows = []
    in_table = False
    with open(path) as f:
        for line in f:
            s = line.rstrip()
            if "2J P" in s and "E1C(RYD)" in s:
                in_table = True
                continue
            if in_table:
                if s.strip() == "":
                    break
                parts = s.split()
                if len(parts) < 5:
                    continue
                try:
                    twoJ = int(parts[0])
                    P    = int(parts[1])
                    S2p1 = int(parts[2])
                    L    = int(parts[3])
                    E_ry = float(parts[-1])
                except ValueError:
                    continue
                g = twoJ + 1
                rows.append((g, E_ry, P, S2p1, L))
    return rows

def zeta_of_T(levels, T_grid_K):
    """ζ_k(T) = Σ_{α ∈ ground_term} g_α exp(-ΔE_α/kT) / Σ_α g_α exp(-ΔE_α/kT)

    Ground term defined as all levels sharing (P, 2S+1, L) with the lowest-E level
    (i.e., the J-splitting of the ground term is collapsed into the numerator).
    """
    g     = np.array([r[0] for r in levels], dtype=float)
    E     = np.array([r[1] for r in levels], dtype=float)
    P     = np.array([r[2] for r in levels], dtype=int)
    S2p1  = np.array([r[3] for r in levels], dtype=int)
    L     = np.array([r[4] for r in levels], dtype=int)
    i_gs  = int(np.argmin(E))
    P0, S0, L0 = P[i_gs], S2p1[i_gs], L[i_gs]
    in_gt = (P == P0) & (S2p1 == S0) & (L == L0)
    # Restrict ground-term tag to levels close in energy (avoid distant
    # accidental term collisions): within 0.1 Ry of ground.
    in_gt &= (E - E[i_gs] < 0.10)
    dE = E - E[i_gs]
    z = np.empty_like(T_grid_K, dtype=float)
    for i, T in enumerate(T_grid_K):
        boltz = np.exp(-dE * RY_TO_K / T)
        U = (g * boltz).sum()
        num = (g[in_gt] * boltz[in_gt]).sum()
        z[i] = num / U
    return z, int(in_gt.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tardis_reference_strat6_2011fe_physical")
    ap.add_argument("--dst", default="data/tardis_reference_strat6_aszeta")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    temps = pd.read_csv(src/"zeta_temps.csv")["temperature"].values
    ions = pd.read_csv(src/"zeta_ions.csv")
    zdata = np.load(src/"zeta_data.npy")

    print(f"Source: {src}")
    print(f"temps grid: {len(temps)} points [{temps[0]:.0f}, {temps[-1]:.0f}]K")
    print(f"zeta_data: {zdata.shape}")
    print()

    patched = zdata.copy()
    i8k = int(np.argmin(np.abs(temps - 8000)))
    i12k = int(np.argmin(np.abs(temps - 12000)))

    summary = []
    for Z, path in AS_RUNS.items():
        if not Path(path).exists():
            print(f"  SKIP Z={Z}: missing {path}")
            continue
        levels = parse_adasout_levels(path)
        if len(levels) < 2:
            print(f"  SKIP Z={Z}: parsed only {len(levels)} levels")
            continue
        z_new, n_gt = zeta_of_T(levels, temps)
        m = (ions["atomic_number"] == Z) & (ions["ion_number"] == 2)
        if not m.any():
            print(f"  SKIP Z={Z}: row (Z,ion=2) not in zeta_ions.csv")
            continue
        idx = ions.index[m][0]
        z_old = zdata[idx]
        print(f"  Z={Z:2d} ion=2 (idx={idx}, NTAR2={len(levels)}):")
        print(f"      OLD ζ@8000K = {z_old[i8k]:.4f}  ζ@12000K = {z_old[i12k]:.4f}")
        print(f"      NEW ζ@8000K = {z_new[i8k]:.4f}  ζ@12000K = {z_new[i12k]:.4f}")
        gs = levels[int(np.argmin([l[1] for l in levels]))]
        gt_g_total = sum(l[0] for l in levels
                         if l[2]==gs[2] and l[3]==gs[3] and l[4]==gs[4] and l[1]-gs[1]<0.10)
        print(f"      ground term: P={gs[2]} 2S+1={gs[3]} L={gs[4]}  "
              f"n_levels_in_term={n_gt}  Σg(term)={gt_g_total}  g_lowest_J={gs[0]}")
        summary.append((Z, idx, z_old.copy(), z_new.copy()))
        patched[idx] = z_new

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    if dst.exists():
        print(f"\nDest {dst} exists — refusing overwrite. Remove it manually first.")
        sys.exit(1)

    # Symlink all original files except zeta_data.npy
    dst.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        name = f.name
        if name == "zeta_data.npy":
            continue
        (dst/name).symlink_to(f.resolve())
    np.save(dst/"zeta_data.npy", patched)
    print(f"\nWrote: {dst}/zeta_data.npy  (other files symlinked from {src})")

    # Diagnostic CSV alongside
    diag_path = dst/"zeta_patch_diag.csv"
    rows = []
    for Z, idx, z_old, z_new in summary:
        for i, T in enumerate(temps):
            rows.append((Z, 2, idx, T, z_old[i], z_new[i], z_new[i]-z_old[i]))
    pd.DataFrame(rows, columns=["Z","ion","row_idx","T_K","zeta_old","zeta_new","delta"]).to_csv(diag_path, index=False)
    print(f"Wrote: {diag_path}")

if __name__ == "__main__":
    main()
