#!/usr/bin/env python3
"""Composition stratification V2: address user's peak position requests.

User wishes (peak fine positions):
  D [4600,4800] -200Å BLUE  -> Fe II should form at HIGHER v (outer shells)
  F [5900,6100] +150Å RED   -> Si II should form at LOWER v (inner shells)

Current strat6 (champion 152761):
  Fe peak at shell 4 (v~12500 km/s), drops to ~1e-6 in outer half
  Si peak at shell 6 (v~13500 km/s), inner shells 0-2 already high (0.38)

Strategy: redistribute Z=14 (Si) and Z=26,27,28 (Fe-peak) by shell.
Track Co (27) and Ni (28) with Fe (decay daughters of 56Ni).
Keep all other species unchanged. Renormalize each shell to sum=1.

Variants:
  stratV2A: Fe-peak (Fe/Co/Ni) shifted outward (shell 12-22 instead of 0-7)
  stratV2B: Si concentrated inner (shells 0-4)
  stratV2C: A + B combined
"""
import json, shutil
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
BASE = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19"

SYMLINK_FILES = [
    "abundances.npy", "atom_masses.csv",
    "electron_densities.csv", "ion_number_density.npy",
    "ionization_energies.csv", "j_blues.npy", "levels.csv",
    "line2macro_level_upper.npy", "line_interaction_id.npy",
    "line_list.csv", "macro_atom_data.csv", "macro_atom_references.csv",
    "mc_estimators.csv", "plasma_state.csv", "spectrum_real.csv",
    "spectrum_virtual.csv", "tau_sobolev.npy", "tau_sobolev_stats.csv",
    "kshape_contract.txt",
    "transition_probabilities.npy", "zeta_data.npy", "zeta_ions.csv",
    "zeta_temps.csv", "density.csv", "geometry.csv", "config.json",
]

# atomic_number column header is shell index 0..29 (NOT Z)
N_SHELLS = 30


def load_abund():
    df = pd.read_csv(BASE / "abundances.csv")
    df = df.set_index("atomic_number")
    return df  # rows = Z, cols = shell index "0".."29"


def write_abund(df, out: Path):
    df.reset_index().to_csv(out / "abundances.csv", index=False)


def renormalize(df):
    # each shell column should sum to ~1.0
    for col in df.columns:
        s = df[col].astype(float).sum()
        if s > 0:
            df[col] = df[col].astype(float) / s
    return df


def shift_fe_outward(df):
    """Move Fe-peak (Z=26,27,28) outward.
       Original: Fe peaks shell 4 (0.10), drops to 1e-6 from shell 8.
       New: spread Fe across shells 10-22 (mid-outer), small inner residue.
    """
    cols = [str(i) for i in range(N_SHELLS)]
    for Z, fe_inner_residue, fe_outer_peak in [
        (26, 0.005, 0.080),  # Fe
        (27, 0.001, 0.015),  # Co
        (28, 0.001, 0.012),  # Ni (decoupled from Co/Fe ratio for simplicity)
    ]:
        if Z not in df.index:
            continue
        new = np.zeros(N_SHELLS)
        # inner residue (shells 0-4)
        for s in range(5):
            new[s] = fe_inner_residue
        # main outer plateau (shells 10-22): smooth Gaussian-ish
        peak_shell = 16
        sigma = 4.0
        for s in range(N_SHELLS):
            x = (s - peak_shell) / sigma
            new[s] += fe_outer_peak * np.exp(-0.5 * x * x)
        for i, c in enumerate(cols):
            df.loc[Z, c] = new[i]
    return df


def shift_si_inward(df):
    """Concentrate Si (Z=14) at shells 0-4.
       Original: peak 0.45 at shell 6, 0.38 at shell 0.
       New: peak 0.55 at shell 1-3, drop to 0.10 at shell 8+.
    """
    cols = [str(i) for i in range(N_SHELLS)]
    Z = 14
    new = np.zeros(N_SHELLS)
    peak_shell = 2
    sigma = 3.0
    for s in range(N_SHELLS):
        x = (s - peak_shell) / sigma
        # Gaussian core for shells 0-7, exponential decay outer
        if s <= 7:
            new[s] = 0.55 * np.exp(-0.5 * x * x)
        else:
            new[s] = 0.10 * np.exp(-(s - 7) / 12.0)
    for i, c in enumerate(cols):
        df.loc[Z, c] = new[i]
    return df


def make_variant(name: str, transform):
    out = ROOT / f"data/tardis_reference_strat6_higherL_aulboost_L19_{name}"
    out.mkdir(exist_ok=True)
    print(f"\n=== {name} -> {out.name}")
    # symlink shared files (atomic data + density + geometry + config unchanged)
    for f in SYMLINK_FILES:
        src = BASE / f
        if not src.exists():
            print(f"  WARN missing {f}")
            continue
        dst = out / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    # transform abundances + renormalize + write
    df = load_abund()
    df = transform(df)
    df = renormalize(df)
    write_abund(df, out)
    # report key Z profile
    for Z, label in [(14, "Si"), (26, "Fe"), (27, "Co"), (28, "Ni"), (12, "Mg"), (8, "O")]:
        if Z in df.index:
            row = df.loc[Z].astype(float)
            print(f"  Z={Z:2d} {label}:  s0={row['0']:.3f}  s4={row['4']:.3f}  "
                  f"s10={row['10']:.3f}  s16={row['16']:.3f}  s22={row['22']:.3f}  s29={row['29']:.3f}")


def add_fe_outer_keep_inner(df, outer_peak_fe=0.060):
    """Keep champion inner Fe profile (preserves UVbl), ADD outer Fe.
       inner: shells 0-7 unchanged
       outer: add Fe/Co/Ni Gaussian peaked at shell 14, eats from O+He budget.
    """
    co_scale = 0.010 / 0.060
    ni_scale = 0.008 / 0.060
    for Z, peak in [(26, outer_peak_fe),
                    (27, outer_peak_fe * co_scale),
                    (28, outer_peak_fe * ni_scale)]:
        if Z not in df.index:
            continue
        for s in range(N_SHELLS):
            if s < 8:
                continue
            x = (s - 14) / 5.0
            current = float(df.loc[Z, str(s)])
            df.loc[Z, str(s)] = current + peak * np.exp(-0.5 * x * x)
    return df


if __name__ == "__main__":
    make_variant("stratV2A_FeOut", shift_fe_outward)
    make_variant("stratV2B_SiIn",  shift_si_inward)
    make_variant("stratV2C_combo", lambda df: shift_si_inward(shift_fe_outward(df)))
    make_variant("stratV2D_FeBoth", add_fe_outer_keep_inner)
    make_variant("stratV2E_FeBoth03", lambda df: add_fe_outer_keep_inner(df, 0.030))
    make_variant("stratV2F_FeBoth01", lambda df: add_fe_outer_keep_inner(df, 0.012))
    print("\nDone. 6 stratification variants created.")
