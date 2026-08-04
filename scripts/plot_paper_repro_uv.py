#!/usr/bin/env python3
"""
#279 Blondin+2013 Fig 7-style UV+opt overlay for capraise champion 157631
against HST stitched p+3.7d (B-max ≈ p+0.4d but we use p+3.7d formal).

This is where the iron-peak III bb cascades (the lever of capraise) should
show their largest effect: the [1700,3000]Å UV blanketing region.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from score_nw import band_int

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"

# HST stitched (B-max)
HST = pd.read_csv(f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv", comment="#")
hst_wl = HST["wavelength_angstrom"].values
hst_fl = HST["flux_erg_s_cm2_angstrom"].values

BASE = f"{ROOT}/logs/nlteO_recalP_fe1p00_co0p3_niuv0p3_n2o0p05_p+3.7d_157611/lumina_spectrum_formal.csv"
CAP  = f"{ROOT}/logs/nlteO_recalP_fe1p0_co0p3_niuv0p3_n2o0p05_ni30p0_p+3.7d_157631/lumina_spectrum_formal.csv"

gO_hst = band_int(hst_wl, hst_fl, 4500, 5800)

def load_mod(p):
    sp = pd.read_csv(p); w=sp.iloc[:,0].values; f=sp.iloc[:,1].values
    gM = band_int(w, f, 4500, 5800)
    return w, f * (gO_hst/gM)

w_b, f_b = load_mod(BASE)
w_c, f_c = load_mod(CAP)

# Two panels: UV [1700, 3300] and opt [3000, 8000]
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

for ax, (lo, hi, title) in zip(axes, [
    (1700, 3300, "UV [1700,3300] — iron-peak III blanketing region"),
    (3000, 8000, "Optical [3000,8000] — bb cascade signatures"),
]):
    m_h = (hst_wl >= lo) & (hst_wl <= hi)
    ax.plot(hst_wl[m_h], hst_fl[m_h], color="k", lw=1.0,
            label="HST stitched (SN 2011fe B-max)", zorder=3)
    m_b = (w_b >= lo) & (w_b <= hi)
    ax.plot(w_b[m_b], f_b[m_b], color="#888", lw=1.0, alpha=0.75,
            label="LUMINA baseline (Fe III 800)", zorder=2)
    m_c = (w_c >= lo) & (w_c <= hi)
    ax.plot(w_c[m_c], f_c[m_c], color="#D97757", lw=1.2,
            label="LUMINA capraise (Co III 3917)", zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylabel(r"$F_\lambda$ [erg/s/cm$^2$/Å]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel(r"Wavelength [Å]")
plt.suptitle("LUMINA capraise vs HST p+3.7d (cf. Blondin+2013 DDC15, Fig 6/7 style)",
             y=0.995)
plt.tight_layout()
out = f"{ROOT}/figures/paper_repro/uv_opt_hst_capraise_vs_baseline.png"
plt.savefig(out, dpi=150)
print(f"wrote {out}")

# Per-band integrals (model/obs ratios)
def ratio(wl, fl, lo, hi):
    mod = band_int(wl, fl, lo, hi)
    obs = band_int(hst_wl, hst_fl, lo, hi)
    return mod/obs, mod, obs

print()
print(f"{'='*64}")
print(f"{'Band [Å]':<14} {'<F_mod>/<F_obs>':>16}  {'<F_obs>':>10}")
print(f"{'-'*64}")
for lo, hi in [(1700,2400), (2400,3000), (3000,3800), (3800,5500),
               (5500,7000), (7000,8000)]:
    rb, _, fo = ratio(w_b, f_b, lo, hi)
    rc, _, _  = ratio(w_c, f_c, lo, hi)
    print(f"{lo}-{hi:<8} base={rb:>6.3f}  cap={rc:>6.3f}    {fo:>.2e}")
print(f"{'='*64}")
