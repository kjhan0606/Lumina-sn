#!/usr/bin/env python3
"""Diagnostic (2): per-line source-function dump for the W-dip carrier lines.

For a wavelength window and shell, joins the NLTE level dump (per-(Z,ion,level)
n_pop, g, E_eV, T_e, T_rad, W, n_ion_total) with line_list.csv (Z,ion,lo,up,
wavelength,f_lu) and computes, per line:
  S_l   = 2hv^3/c^2 / ((g_up n_lo)/(g_lo n_up) - 1)   (two-level NLTE source)
  WB    = W * B_nu(T_rad)                              (diluted-continuum proxy)
  b_lo, b_up : departure coeffs n_pop / n_LTE (LTE = n_ion*g e^{-E/kT_e}/U_NLTE)
  S_l/WB : >1 => line source brighter than continuum => fills trough;
           <0.5 => can carve a >50% trough.

Disentangles A/B (source-function) vs C (carrier/unsuppressed upper level):
  - b_up ~ 1 (unsuppressed) + S_l/WB >~ 1  => carrier-driven fill (Agent C)
  - b_up << 1 but S_l/WB still >~ 1        => two-level formula structural (A/B)

Usage: diag_line_source_wdip.py <nlte_levels_iterNNN.csv> [line_list.csv]
"""
import sys, numpy as np, pandas as pd
from pathlib import Path

H = 6.62607015e-27      # erg s
C = 2.99792458e10       # cm/s
K = 1.380649e-16        # erg/K

dump_path = Path(sys.argv[1])
if len(sys.argv) > 2:
    ll_path = Path(sys.argv[2])
else:
    ll_path = dump_path.parent / "ref" / "line_list.csv"

# W-dip carrier windows (obs 2002bo B-max): Fe/S-blend 4800-4900, S II doublet 5300-5500
WINDOWS = [(4600, 4800, "4700 bump / blue W-trough"),
           (4850, 4950, "4896 W-trough"),
           (5300, 5500, "S II 5454/5640 W-troughs")]
SHELLS = [0, 4, 8, 12, 16, 20]

dump = pd.read_csv(dump_path)
ll = pd.read_csv(ll_path)
print(f"dump: {dump_path}  ({len(dump)} rows)  line_list: {len(ll)} lines")

# partition fn per (ion,shell) from the NLTE-tracked levels (approx, lower bound)
def add_lte(df):
    out = []
    for (Z, ion, s), g in df.groupby(["Z", "ion", "shell"]):
        Te = g["T_e"].iloc[0]
        U = (g["g"] * np.exp(-g["E_eV"] * 1.602176634e-12 / (K * Te))).sum()
        nl = g["n_ion_total"].iloc[0] * g["g"] * \
             np.exp(-g["E_eV"] * 1.602176634e-12 / (K * Te)) / U
        gg = g.copy(); gg["n_lte"] = nl.values; gg["U"] = U
        out.append(gg)
    return pd.concat(out)

dump = add_lte(dump)

for lo_w, hi_w, name in WINDOWS:
    sub = ll[(ll["wavelength"] >= lo_w) & (ll["wavelength"] <= hi_w)].copy()
    print(f"\n=== {name}  [{lo_w},{hi_w}] A : {len(sub)} lines ===")
    for s in SHELLS:
        d = dump[dump["shell"] == s].set_index(["Z", "ion", "level_idx"])
        if d.empty:
            continue
        T_rad = d["T_rad"].iloc[0]; W = d["W"].iloc[0]
        rows = []
        for _, L in sub.iterrows():
            key_lo = (L["atomic_number"], L["ion_number"], L["level_number_lower"])
            key_up = (L["atomic_number"], L["ion_number"], L["level_number_upper"])
            if key_lo not in d.index or key_up not in d.index:
                continue
            n_lo = d.loc[key_lo, "n_pop"]; n_up = d.loc[key_up, "n_pop"]
            g_lo = d.loc[key_lo, "g"];     g_up = d.loc[key_up, "g"]
            if n_up <= 0 or n_lo <= 0:
                continue
            nu = C / (L["wavelength"] * 1e-8)
            denom = (g_up * n_lo) / (g_lo * n_up) - 1.0
            if denom <= 0:   # inversion -> formally negative/masing, skip
                S_l = np.inf
            else:
                S_l = (2 * H * nu**3 / C**2) / denom
            B = (2 * H * nu**3 / C**2) / (np.exp(H * nu / (K * T_rad)) - 1.0)
            WB = W * B
            b_lo = n_lo / d.loc[key_lo, "n_lte"] if d.loc[key_lo, "n_lte"] > 0 else np.nan
            b_up = n_up / d.loc[key_up, "n_lte"] if d.loc[key_up, "n_lte"] > 0 else np.nan
            rows.append((L["wavelength"], int(L["atomic_number"]), int(L["ion_number"]),
                         L["f_lu"], n_lo, n_up, b_lo, b_up, S_l, WB,
                         S_l / WB if WB > 0 else np.nan))
        if not rows:
            continue
        r = pd.DataFrame(rows, columns=["lam", "Z", "ion", "f_lu", "n_lo", "n_up",
                                        "b_lo", "b_up", "S_l", "WB", "S/WB"])
        r = r.sort_values("f_lu", ascending=False).head(6)
        print(f"  shell {s:2d}  T_rad={T_rad:7.0f}  W={W:.3f}")
        with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
            print(r.to_string(index=False))
