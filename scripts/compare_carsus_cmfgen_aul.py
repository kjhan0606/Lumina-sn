#!/usr/bin/env python3
"""Compare carsus vs CMFGEN A_ul values for the user's 6 problem peak bands.

For each (Z, ion) of interest, match transitions between the two line lists
by wavelength (within ±0.5 Å). Report A_ul ratio CMFGEN/carsus, flag those
that differ by >2x or <0.5x. Output per-band summaries.

Bands of interest (P-Cygni emission peak position):
  A [3400,3600]  lower height  — Ca II / Ti II
  B [3800,4100]  raise (fluo)  — Fe II / Ca II H
  C [4250,4400]  +50Å red       — Fe II 4233 / Cr II
  D [4600,4800]  -200Å BLUE     — Fe II 4924/5018/5169 (target shells)
  E [5400,5600]  lower          — Fe II / Ti II
  F [5900,6100]  +150Å Si II    — Si II 6347/6371
Plus the documented residual: [3000,3100] iron-peak III bump.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
CARSUS = ROOT / "data/tardis_reference_strat6_higherL_aulboost_L19/line_list.csv"
CMFGEN = ROOT / "data/tardis_reference_cmfgen/line_list.csv"

BANDS = [
    ("UVbl_iron3", 3000, 3100, "Co/Fe/Cr III bump (memory)"),
    ("A_3500",     3400, 3600, "lower"),
    ("B_4000",     3800, 4100, "raise fluorescence"),
    ("C_4300",     4250, 4400, "+50A red lower"),
    ("D_4700",     4600, 4800, "-200A BLUE (Fe II 4924/5018/5169)"),
    ("E_5500",     5400, 5600, "lower"),
    ("F_6000",     5900, 6100, "+150A Si II"),
]

# ions that drive iron blanketing + IME peaks
ION_FILTER = [
    (14, 1),  # Si II
    (20, 1),  # Ca II
    (22, 1), (22, 2),  # Ti I/II
    (24, 1), (24, 2),  # Cr I/II
    (26, 1), (26, 2),  # Fe I/II
    (27, 1), (27, 2),  # Co I/II
    (28, 1), (28, 2),  # Ni I/II
]


def load_lines(path):
    print(f"Loading {path.name} ...", end=" ", flush=True)
    df = pd.read_csv(path, usecols=["atomic_number", "ion_number",
                                     "wavelength", "A_ul"])
    print(f"{len(df):,} rows")
    return df


def match_band(carsus_df, cmfgen_df, lo_A, hi_A, ions, tol_A=0.5):
    """For each carsus line in [lo,hi] of given ions, find nearest CMFGEN match.
    Returns dataframe with both A_ul values + ratio."""
    rows = []
    for Z, ion in ions:
        c = carsus_df[(carsus_df.atomic_number == Z) &
                      (carsus_df.ion_number == ion) &
                      (carsus_df.wavelength >= lo_A) &
                      (carsus_df.wavelength <= hi_A)].copy()
        m = cmfgen_df[(cmfgen_df.atomic_number == Z) &
                      (cmfgen_df.ion_number == ion) &
                      (cmfgen_df.wavelength >= lo_A - tol_A) &
                      (cmfgen_df.wavelength <= hi_A + tol_A)].copy()
        if c.empty or m.empty:
            continue
        m_lams = m.wavelength.values
        m_aul = m.A_ul.values
        for _, r in c.iterrows():
            lam_c = r.wavelength
            j = int(np.argmin(np.abs(m_lams - lam_c)))
            dlam = abs(m_lams[j] - lam_c)
            if dlam > tol_A:
                continue
            a_c = r.A_ul
            a_m = m_aul[j]
            if a_c <= 0 or a_m <= 0:
                continue
            rows.append(dict(Z=Z, ion=ion, lam_carsus=lam_c, lam_cmfgen=m_lams[j],
                             dlam=dlam, A_carsus=a_c, A_cmfgen=a_m,
                             ratio=a_m / a_c))
    return pd.DataFrame(rows)


def summarize(df, label):
    if df.empty:
        print(f"  {label}: NO MATCHES")
        return
    print(f"\n--- {label}: {len(df)} matched transitions ---")
    # group by (Z, ion)
    for (Z, ion), g in df.groupby(["Z", "ion"]):
        med = g.ratio.median()
        n_div = (g.ratio > 2).sum() + (g.ratio < 0.5).sum()
        # top 5 strongest carsus lines in this band
        top = g.nlargest(5, "A_carsus")
        print(f"  Z={Z:2d} ion={ion}: n={len(g):4d}  median(CMFGEN/carsus)={med:.3f}  "
              f"|ratio>2|+|<0.5|={n_div}")
        for _, r in top.iterrows():
            flag = ""
            if r.ratio > 2 or r.ratio < 0.5:
                flag = "  *DIFFER*"
            print(f"     λ={r.lam_carsus:7.2f}  A_carsus={r.A_carsus:.3e}  "
                  f"A_cmfgen={r.A_cmfgen:.3e}  ratio={r.ratio:.3f}{flag}")


def main():
    car = load_lines(CARSUS)
    cmf = load_lines(CMFGEN)
    all_rows = []
    for label, lo, hi, note in BANDS:
        print(f"\n=== {label}  [{lo},{hi}]Å  ({note}) ===")
        df = match_band(car, cmf, lo, hi, ION_FILTER)
        summarize(df, label)
        df["band"] = label
        all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out = ROOT / "logs/carsus_vs_cmfgen_Aul.csv"
    full.to_csv(out, index=False)
    print(f"\n\nWrote {out}  ({len(full)} matched transitions across all bands)")

    if not full.empty:
        print("\n=== Global stats per ion (all bands combined) ===")
        for (Z, ion), g in full.groupby(["Z", "ion"]):
            ratios = g.ratio.values
            print(f"  Z={Z:2d} ion={ion}: n={len(g):5d}  median={np.median(ratios):.3f}  "
                  f"geomean={np.exp(np.mean(np.log(ratios))):.3f}  "
                  f"min={ratios.min():.3e}  max={ratios.max():.3e}")


if __name__ == "__main__":
    main()
