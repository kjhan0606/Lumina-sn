#!/usr/bin/env python3
"""Task #219c: Recompute Co III + Ni III ζ from the clean LUMINA partition functions.

Uses the full level structure in `data/tardis_reference/levels.csv` (cleaned per
#253: CMFGEN 19apr23 Co III + CMFGEN 27aug12 Ni III).  Replaces the Kurucz
0.2908 placeholder and the AS plan-D ground-term ζ (which used only the small
AUTOSTRUCTURE target-level subset, ~10-30 levels).

ζ(T) convention (matches Carsus / TARDIS):
    ζ(T) = Σ_{α ∈ ground term}  g_α exp(−ΔE_α/kT)
           / Σ_α                 g_α exp(−ΔE_α/kT)

Ground-term cluster = all levels within 1.36 eV (≈0.10 Ry) of the lowest level,
which captures the fine-structure splittings of the ground term without
contaminating with the first excited term (usually >1.5 eV above).

Writes a new reference dir alongside the input (default
`data/tardis_reference_zeta_clean/`) with `zeta_data.npy` patched at rows for
Co III + Ni III; all other files symlinked from the source.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

K_B_EV = 8.617333e-5   # eV / K
GT_WIN_EV = 1.36       # ≈ 0.10 Ry — ground-term cluster width

PATCH_IONS = [
    (27, 2, "Co III"),
    (28, 2, "Ni III"),
]


def zeta_of_T(levels, T_grid_K, gt_win_eV=GT_WIN_EV):
    """Return ζ(T) over the temperature grid plus diagnostic counts."""
    g = levels["g"].to_numpy(dtype=float)
    E = levels["energy_eV"].to_numpy(dtype=float)
    E_gs = E.min()
    dE = E - E_gs
    in_gt = dE < gt_win_eV
    z = np.empty_like(T_grid_K, dtype=float)
    for i, T in enumerate(T_grid_K):
        boltz = np.exp(-dE / (K_B_EV * T))
        U = (g * boltz).sum()
        U_gt = (g[in_gt] * boltz[in_gt]).sum()
        z[i] = U_gt / U
    return z, int(in_gt.sum()), float(g[in_gt].sum()), float(g.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tardis_reference")
    ap.add_argument("--dst", default="data/tardis_reference_zeta_clean")
    ap.add_argument("--gt-win-eV", type=float, default=GT_WIN_EV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    temps = pd.read_csv(src/"zeta_temps.csv")["temperature"].to_numpy()
    ions = pd.read_csv(src/"zeta_ions.csv")
    zdata = np.load(src/"zeta_data.npy")
    levels = pd.read_csv(src/"levels.csv")

    print(f"Source: {src}")
    print(f"  temps grid: {len(temps)} points [{temps[0]:.0f}, {temps[-1]:.0f}] K")
    print(f"  zeta_data shape: {zdata.shape}")
    print(f"  ground-term cluster window: {args.gt_win_eV:.3f} eV")
    print()

    patched = zdata.copy()
    i8k = int(np.argmin(np.abs(temps - 8000)))
    i12k = int(np.argmin(np.abs(temps - 12000)))

    summary = []
    for Z, ion, name in PATCH_IONS:
        sub = levels[(levels.atomic_number == Z) & (levels.ion_number == ion)]
        if len(sub) < 2:
            print(f"  SKIP {name}: only {len(sub)} levels in levels.csv")
            continue
        z_new, n_gt, sum_g_gt, sum_g_tot = zeta_of_T(sub, temps, args.gt_win_eV)
        m = (ions.atomic_number == Z) & (ions.ion_number == ion)
        if not m.any():
            print(f"  SKIP {name}: no row in zeta_ions.csv")
            continue
        idx = ions.index[m][0]
        z_old = zdata[idx]
        print(f"  {name:7s} (Z={Z} ion={ion}, idx={idx}, N_levels={len(sub)}):")
        print(f"      OLD ζ@8000K = {z_old[i8k]:.4f}   ζ@12000K = {z_old[i12k]:.4f}")
        print(f"      NEW ζ@8000K = {z_new[i8k]:.4f}   ζ@12000K = {z_new[i12k]:.4f}")
        print(f"      ground term: {n_gt} levels, Σg(GT) = {sum_g_gt:.0f}, "
              f"Σg(all) = {sum_g_tot:.0f}, g_gs = {sub.iloc[0].g:.0f}")
        summary.append((Z, ion, idx, name, z_old.copy(), z_new.copy()))
        patched[idx] = z_new

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    if dst.exists():
        print(f"\nDest {dst} exists — refusing overwrite. Remove it manually first.")
        sys.exit(1)
    dst.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        name = f.name
        if name == "zeta_data.npy":
            continue
        (dst/name).symlink_to(f.resolve())
    np.save(dst/"zeta_data.npy", patched)
    print(f"\nWrote: {dst}/zeta_data.npy  (other files symlinked from {src})")

    rows = []
    for Z, ion, idx, name, z_old, z_new in summary:
        for i, T in enumerate(temps):
            rows.append((Z, ion, idx, name, T, z_old[i], z_new[i], z_new[i] - z_old[i]))
    diag = pd.DataFrame(rows, columns=["Z","ion","row_idx","name","T_K",
                                       "zeta_old","zeta_new","delta"])
    diag.to_csv(dst/"zeta_patch_diag.csv", index=False)
    print(f"Wrote: {dst}/zeta_patch_diag.csv")


if __name__ == "__main__":
    main()
