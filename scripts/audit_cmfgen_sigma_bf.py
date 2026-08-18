#!/usr/bin/env python3
"""Path-1 audit: CMFGEN σ_bf magnitudes for iron-peak III ions (Z=24/26/27/28, ion=2).

Per project_sigmabf_saturation_evidence.md, R_rec/R_bf=1e-10 across all iron-peak
III at shell 0 — photoionization 10⁹-10¹⁰× beyond plan-C DR ceiling. If σ_bf
itself is anomalously large (10²-10⁴× too big), fixing the CMFGEN bake closes
the saturation gap. If σ_bf magnitudes look physical, saturation is real and
Path 3 (T_e self-consistent) is the next lever.

Outputs:
  1. Per-(Z,ion) histogram of σ_bf(ν_threshold) — flag outliers > 1e-16 cm²
  2. Boltzmann-weighted ⟨σ_bf⟩ at T_e=8000 K (representative inner shell)
  3. Per-level σ_bf(ν_threshold) vs hydrogenic σ_H = 7.91e-18 / Z_eff^2 [cm²]
  4. Top-20 outliers by σ_bf magnitude per ion
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
LEVELS_CSV = ROOT / "data" / "tardis_reference" / "levels.csv"
BIN_PATH   = ROOT / "data" / "atomic" / "cmfgen_sigma_bf.bin"

C_CGS, H_CGS, K_CGS = 2.99792458e10, 6.62607015e-27, 1.380649e-16
EV_TO_ERG = 1.602176634e-12

# BF grid (must match lumina.h)
BF_NU_MIN, BF_NU_MAX, BF_N_FREQ_BIN = (
    5.8412785919616062e13, 4.0362581455823112e16, 1234
)
log_min = np.log(BF_NU_MIN)
d_log_nu = np.log(BF_NU_MAX / BF_NU_MIN) / BF_N_FREQ_BIN
NU_GRID = BF_NU_MIN * np.exp((np.arange(BF_N_FREQ_BIN) + 0.5) * d_log_nu)

# Iron-peak III ions of interest (carsus ion_number = ion_charge = stage-1)
TARGETS = [(24, 2, "Cr III"), (26, 2, "Fe III"),
           (27, 2, "Co III"), (28, 2, "Ni III")]
# Reference: also dump iron-peak II for comparison
EXTRA = [(24, 1, "Cr II"), (26, 1, "Fe II"),
         (27, 1, "Co II"), (28, 1, "Ni II")]

# Ionization energies (eV) for threshold computation per (Z, ion).
# Source: NIST ASD for the species (II=loses 1 more, III=loses 2 more, etc).
# We need the ionization potential FROM the listed ion stage TO the next.
IONIZATION_eV = {
    (24, 1): 16.486,  (24, 2): 30.96,   # Cr II→III, Cr III→IV
    (26, 1): 16.199,  (26, 2): 30.652,  # Fe II→III, Fe III→IV
    (27, 1): 17.084,  (27, 2): 33.50,   # Co II→III, Co III→IV
    (28, 1): 18.169,  (28, 2): 35.187,  # Ni II→III, Ni III→IV
}


def load_cmfgen_bin(path: Path):
    """Return (n_levels, n_freq, has, sigma) where sigma[lev, freq] in cm²."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        assert magic == 0x434D4644 and version == 1, f"bad header {magic:x}"
        n_levels, n_freq = struct.unpack("<ii", f.read(8))
        nu_min, nu_max = struct.unpack("<dd", f.read(16))
        has = np.frombuffer(f.read(n_levels), dtype=np.int8).copy()
        pad = (8 - (n_levels % 8)) % 8
        f.read(pad)
        sigma = np.frombuffer(f.read(n_levels * n_freq * 8),
                              dtype=np.float64).reshape(n_levels, n_freq).copy()
    return n_levels, n_freq, nu_min, nu_max, has, sigma


def sigma_at_threshold(sigma_row, nu_thresh):
    """σ_bf at the per-level threshold frequency (first non-zero bin near ν_t)."""
    idx = int(np.searchsorted(NU_GRID, nu_thresh))
    if idx >= BF_N_FREQ_BIN:
        return 0.0, idx
    return float(sigma_row[idx]), idx


def hydrogenic_sigma_thresh(Z_eff):
    """σ_H(ν_t) for a hydrogenic ion of effective charge Z_eff [cm²].
    Ground state: σ = 7.91e-18 / Z_eff² (Karzas-Latter)."""
    return 7.91e-18 / (Z_eff ** 2)


def main():
    print("=" * 80)
    print("PATH-1 AUDIT: CMFGEN σ_bf magnitudes (iron-peak II/III)")
    print("=" * 80)

    levels = pd.read_csv(LEVELS_CSV)
    print(f"levels.csv  : {len(levels)} rows")

    n_lev, n_freq, nu_min, nu_max, has, sigma = load_cmfgen_bin(BIN_PATH)
    print(f"binary      : {n_lev} levels × {n_freq} freq bins, has=1 for {int(has.sum())}")
    assert n_lev == len(levels), f"size mismatch {n_lev} vs {len(levels)}"
    assert abs(nu_min - BF_NU_MIN) < 1 and abs(nu_max - BF_NU_MAX) < 1

    rows = []
    for (Z, ion, label) in EXTRA + TARGETS:
        sub = levels[(levels.atomic_number == Z) &
                     (levels.ion_number == ion)].copy()
        if sub.empty:
            print(f"\n[{label}] no carsus levels"); continue
        gids = sub.index.values
        h = has[gids]; n_total = len(sub); n_has = int(h.sum())
        E_ion = IONIZATION_eV.get((Z, ion))
        if E_ion is None:
            print(f"\n[{label}] missing IP — skip"); continue

        # Per-level σ_bf at ν_threshold
        thr_vals, edge_idx = [], []
        for gi in gids[h.astype(bool)]:
            E_lev = float(levels.iloc[gi].energy_eV)
            E_thr_eV = E_ion - E_lev
            if E_thr_eV <= 0:
                thr_vals.append(0.0); edge_idx.append(-1); continue
            nu_t = E_thr_eV * EV_TO_ERG / H_CGS
            s, idx = sigma_at_threshold(sigma[gi], nu_t)
            thr_vals.append(s); edge_idx.append(idx)

        thr = np.array([v for v in thr_vals if v > 0])
        Z_eff = ion + 1  # hydrogenic effective charge
        sig_H = hydrogenic_sigma_thresh(Z_eff)

        print(f"\n[{label}] Z={Z} ion={ion} carsus_levels={n_total} cmfgen_baked={n_has}")
        print(f"  E_ion = {E_ion:.3f} eV  →  ν_thresh ~ {E_ion*EV_TO_ERG/H_CGS:.2e} Hz")
        print(f"  hydrogenic σ_H(thr) ~ {sig_H:.2e} cm² (Z_eff={Z_eff})")
        if len(thr) == 0:
            print("  no non-zero σ_bf at threshold"); continue
        print(f"  σ_bf(ν_t) stats over {len(thr)} levels:")
        print(f"    min      {thr.min():.3e} cm²   (× σ_H = {thr.min()/sig_H:.2e})")
        print(f"    median   {np.median(thr):.3e} cm²   (× σ_H = {np.median(thr)/sig_H:.2e})")
        print(f"    mean     {thr.mean():.3e} cm²   (× σ_H = {thr.mean()/sig_H:.2e})")
        print(f"    max      {thr.max():.3e} cm²   (× σ_H = {thr.max()/sig_H:.2e})")
        # Outlier flagging — anything > 1e-16 cm² is suspect for non-H-like ions
        OUTLIER = 1e-16
        n_out = int((thr > OUTLIER).sum())
        if n_out:
            print(f"  ** OUTLIERS > 1e-16 cm² : {n_out}/{len(thr)} ({100*n_out/len(thr):.1f}%)")
        else:
            print(f"  no outliers > 1e-16 cm² — distribution physical")

        # Boltzmann-weighted ⟨σ_bf⟩ at T_e=8000 K (inner shell)
        T_e = 8000.0
        E_arr = sub["energy_eV"].values[h.astype(bool)] * EV_TO_ERG
        g_arr = sub["g"].values[h.astype(bool)]
        boltz = g_arr * np.exp(-E_arr / (K_CGS * T_e))
        if boltz.sum() > 0:
            valid = np.array(thr_vals) > 0
            wsig = (np.array(thr_vals)[valid] * boltz[valid]).sum() / boltz[valid].sum()
            print(f"  ⟨σ_bf⟩ Boltzmann-weighted (T_e=8000K): {wsig:.3e} cm²"
                  f"  (× σ_H = {wsig/sig_H:.2e})")

        # Top-10 outliers
        idx_sorted = np.argsort([-v for v in thr_vals])[:10]
        valid_gids = gids[h.astype(bool)]
        print(f"  Top-10 σ_bf(ν_t) levels:")
        print(f"    {'global_idx':>10s} {'lev':>4s} {'E_eV':>8s} {'g':>5s} {'σ_bf':>11s} {'×σ_H':>8s}")
        for k in idx_sorted:
            gi = valid_gids[k]
            row = levels.iloc[gi]
            sb = thr_vals[k]
            print(f"    {gi:>10d} {int(row.level_number):>4d} "
                  f"{row.energy_eV:>8.3f} {int(row.g):>5d} "
                  f"{sb:>11.3e} {sb/sig_H:>8.2e}")

        rows.append({"label": label, "Z": Z, "ion": ion,
                     "n_levels": n_total, "n_baked": n_has,
                     "median_sigma": np.median(thr) if len(thr) else 0,
                     "max_sigma": thr.max() if len(thr) else 0,
                     "outliers_1e-16": n_out,
                     "wsig_T8000": wsig if boltz.sum() > 0 else 0,
                     "sigma_H": sig_H})

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    df = pd.DataFrame(rows)
    if len(df):
        df["median/H"]  = df.median_sigma / df.sigma_H
        df["max/H"]     = df.max_sigma / df.sigma_H
        df["wsig/H"]    = df.wsig_T8000 / df.sigma_H
        cols = ["label", "n_levels", "n_baked", "median_sigma",
                "max_sigma", "outliers_1e-16", "median/H", "max/H", "wsig/H"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3e}"))


if __name__ == "__main__":
    main()
