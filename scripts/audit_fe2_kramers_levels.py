#!/usr/bin/env python3
"""Identify Fe II levels currently using Kramers σ_bf fallback.

Per project #301, Fe II has ~27% CMFGEN coverage. Fe II is the candidate UV
opacity carrier for the priority #2 fix (project #302): unlike Si II/Mg II/Ca II
it is not heavily over-ionized at SN photosphere, so its σ_bf could absorb J_ν
near 1808 Å where Si II level 5 is pumped.

Binary layout (from lumina_atomic.c:710-787):
   uint32 magic = 0x434D4644 ('CMFD')
   uint32 version = 1
   int32  n_levels, n_freq_bins
   double nu_min, nu_max
   int8   has_cmfgen[n_levels]   (padded to 8-byte alignment)
   double sigma[n_levels * n_freq_bins]
"""
import numpy as np
import pandas as pd
from pathlib import Path

REF = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/"
    "tardis_reference_v3_femerge_capraise_bbfix_metafix_asE"
)
DUMP = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/"
    "paperDDC15v3asEfloor_bbfix_2002bo_vi9019_L1p0_nltedump_158780/"
    "nlte_levels_iter003.csv"
)
OUT_DIR = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/layer2_bl_diagnosis"
)


def load_has_cmfgen(path):
    with open(path, "rb") as f:
        magic = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        assert magic == 0x434D4644, f"magic mismatch 0x{magic:08x}"
        version = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        n_lev = int(np.frombuffer(f.read(4), dtype="<i4")[0])
        n_freq = int(np.frombuffer(f.read(4), dtype="<i4")[0])
        nu_min = float(np.frombuffer(f.read(8), dtype="<f8")[0])
        nu_max = float(np.frombuffer(f.read(8), dtype="<f8")[0])
        has = np.frombuffer(f.read(n_lev), dtype=np.int8).astype(bool)
    return n_lev, n_freq, nu_min, nu_max, has


def main():
    levels = pd.read_csv(REF / "levels.csv")
    print(f"levels.csv: {len(levels)} rows")

    n_lev, n_freq, nu_min, nu_max, has_cmf = load_has_cmfgen(REF / "cmfgen_sigma_bf.bin")
    print(f"cmfgen_sigma_bf.bin: n_lev={n_lev}, n_freq={n_freq}, "
          f"ν∈[{nu_min:.3e}, {nu_max:.3e}] Hz")
    assert n_lev == len(levels), f"n_lev mismatch {n_lev} vs {len(levels)}"

    levels["has_cmfgen"] = has_cmf
    fe2 = levels[(levels["atomic_number"] == 26) & (levels["ion_number"] == 1)].copy()
    print(f"\nFe II: {len(fe2)} levels, {int(fe2['has_cmfgen'].sum())} CMFGEN, "
          f"{int((~fe2['has_cmfgen']).sum())} Kramers")

    IP_FE2 = 16.188  # NIST
    fe2["E_thresh_eV"] = IP_FE2 - fe2["energy_eV"]
    fe2["lam_thresh_A"] = np.where(
        fe2["E_thresh_eV"] > 0, 12397.6 / fe2["E_thresh_eV"], np.nan
    )

    bands = [
        ("Lyα/Si II ion 800-1310",   800, 1310),
        ("UV middle 1310-1700",     1310, 1700),
        ("Si II pump 1700-1900",    1700, 1900),
        ("UV mid 2000-2700",        2000, 2700),
        ("Mg II resonance 2700-2900", 2700, 2900),
        ("Optical 3000-5000",       3000, 5000),
        ("Long λ >5000",            5000, 1e6),
    ]

    print("\nFe II level COUNT per threshold band:")
    print(f"  {'band':<28} {'N_total':>7} {'N_CMF':>6} {'N_Kram':>6} {'g_sum_K':>8}")
    for label, lo, hi in bands:
        sel = (fe2["lam_thresh_A"] >= lo) & (fe2["lam_thresh_A"] < hi)
        sub = fe2[sel]
        n_cmf = int(sub["has_cmfgen"].sum())
        n_kram = int((~sub["has_cmfgen"]).sum())
        g_sum_k = float(sub.loc[~sub["has_cmfgen"], "g"].sum())
        print(f"  {label:<28} {len(sub):>7} {n_cmf:>6} {n_kram:>6} {g_sum_k:>8.0f}")

    # Cross with NLTE dump to weight by actual n_pop at shell 0
    print(f"\nLoading NLTE dump for Fe II shell-0 populations ...")
    df = pd.read_csv(DUMP, usecols=["Z", "ion", "shell", "global_idx", "n_pop"])
    fe2_dump = df[(df["Z"] == 26) & (df["ion"] == 1) & (df["shell"] == 0)].copy()
    print(f"  Fe II shell-0 rows: {len(fe2_dump)}")
    fe2_dump = fe2_dump.merge(
        fe2.reset_index().rename(columns={"index": "lev_csv_idx"})[
            ["lev_csv_idx", "g", "energy_eV", "has_cmfgen", "lam_thresh_A"]
        ],
        left_on="global_idx",
        right_on="lev_csv_idx",
        how="left",
    )
    matched = fe2_dump["lev_csv_idx"].notna().sum()
    print(f"  Merge matched: {matched}/{len(fe2_dump)}")

    print("\nFe II shell-0 n_pop sum by threshold band (CMFGEN vs Kramers):")
    print(f"  {'band':<28} {'n_pop_CMF':>12} {'n_pop_Kram':>12} {'frac_K':>7}")
    total_npop = fe2_dump["n_pop"].sum()
    for label, lo, hi in bands:
        sel = (fe2_dump["lam_thresh_A"] >= lo) & (fe2_dump["lam_thresh_A"] < hi)
        sub = fe2_dump[sel]
        n_cmf = float(sub.loc[sub["has_cmfgen"] == True, "n_pop"].sum())
        n_kram = float(sub.loc[sub["has_cmfgen"] == False, "n_pop"].sum())
        frac_k = n_kram / max(n_cmf + n_kram, 1e-30)
        print(f"  {label:<28} {n_cmf:>12.3e} {n_kram:>12.3e} {frac_k:>7.1%}")
    print(f"\n  Total Fe II shell-0 n_pop: {total_npop:.3e}")

    out_path = OUT_DIR / "fe2_kramers_levels_audit.csv"
    fe2_dump_out = fe2_dump.sort_values("lam_thresh_A").reset_index(drop=True)
    fe2_dump_out.to_csv(out_path, index=False)
    print(f"\nDetailed Fe II shell-0 dump → {out_path}")


if __name__ == "__main__":
    main()
