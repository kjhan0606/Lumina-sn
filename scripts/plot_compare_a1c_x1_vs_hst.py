#!/usr/bin/env python3
"""Spectrum comparison: A1c_e1p00 (J-champion) + X1_b3p0 (RMS-champion) vs HST 2011fe B-max.

Generates a multi-panel overlay (full range + zoom-ins on diagnostic troughs)
to help identify remaining residual structure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"

HST = os.path.join(ROOT, "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv")
A1C = os.path.join(ROOT, "logs/ddc15A1c_156567_ddc15A1c_e1p00/lumina_spectrum.csv")
X1  = os.path.join(ROOT, "logs/ddc15X1_156525_ddc15X1_b3p0/lumina_spectrum.csv")

# Continuum band for amplitude normalization (line-poor between Fe and Na/Si forest)
NORM_LO, NORM_HI = 5500.0, 6000.0
WL_LO, WL_HI = 2400.0, 9200.0

def load_lumina(path):
    df = pd.read_csv(path)
    return df["wavelength_angstrom"].to_numpy(), df["flux"].to_numpy()

def load_hst(path):
    df = pd.read_csv(path)
    return df["wavelength_angstrom"].to_numpy(), df["flux_erg_s_cm2_angstrom"].to_numpy()

def integrate_band(wl, fl, lo, hi):
    m = (wl >= lo) & (wl <= hi) & np.isfinite(fl)
    if m.sum() < 3:
        return np.nan
    return np.trapz(fl[m], wl[m])

def normalize_to_hst(wl_m, fl_m, wl_h, fl_h):
    I_m = integrate_band(wl_m, fl_m, NORM_LO, NORM_HI)
    I_h = integrate_band(wl_h, fl_h, NORM_LO, NORM_HI)
    if not np.isfinite(I_m) or I_m <= 0:
        return fl_m
    return fl_m * (I_h / I_m)

wl_h, fl_h = load_hst(HST)
wl_a, fl_a = load_lumina(A1C)
wl_x, fl_x = load_lumina(X1)

fl_a_n = normalize_to_hst(wl_a, fl_a, wl_h, fl_h)
fl_x_n = normalize_to_hst(wl_x, fl_x, wl_h, fl_h)

# ---- plot ----
features = [
    ("Ca H&K",    3700, 4000),
    ("Mg II",     4300, 4600),
    ("Fe II 5169",4800, 5300),
    ("Si II 5972",5650, 5950),
    ("Si II 6355",6000, 6350),
    ("O I 7773",  7400, 7800),
    ("Ca IR trip",8200, 8800),
]

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
              height_ratios=[1.4, 1.0, 1.0])

# Top: full-range overlay
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(wl_h, fl_h,  color="black", lw=1.2, label="HST 2011fe B-max", zorder=5)
ax0.plot(wl_a, fl_a_n, color="#3898EC", lw=1.0, alpha=0.85,
         label="LUMINA A1c ε=1.00 (J=0.220, RMS_bn=0.179)")
ax0.plot(wl_x, fl_x_n, color="#D97757", lw=1.0, alpha=0.85,
         label="LUMINA X1_b3p0 (J=0.237, RMS_bn=0.171)")
ax0.set_xlim(WL_LO, WL_HI)
y_max = np.nanmax(fl_h[(wl_h >= WL_LO) & (wl_h <= WL_HI)]) * 1.15
ax0.set_ylim(0, y_max)
ax0.set_xlabel("Wavelength (Å)")
ax0.set_ylabel(r"$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
ax0.set_title(f"SN 2011fe B-max — LUMINA vs HST  (norm. on {NORM_LO:.0f}–{NORM_HI:.0f} Å)")
ax0.legend(loc="upper right", fontsize=9)
ax0.grid(alpha=0.25)
for name, lo, hi in features:
    ax0.axvspan(lo, hi, alpha=0.06, color="gray")
    ax0.text((lo+hi)/2, y_max*0.95, name, ha="center", va="top", fontsize=7, color="dimgray")

# Six zoom panels
zoom = [
    ("UV blanketing 2400–3500",   2400, 3500),
    ("Ca H&K 3700–4200",          3700, 4200),
    ("Mg II / Fe II 4200–5400",   4200, 5400),
    ("Si II / Na I 5500–6400",    5500, 6400),
    ("O I + emission 6500–8000",  6500, 8000),
    ("Ca IR triplet 8000–9200",   8000, 9200),
]
for i, (title, lo, hi) in enumerate(zoom):
    ax = fig.add_subplot(gs[1 + i // 4, (i % 4)] if i < 4 else gs[2, i - 4])
    mh = (wl_h >= lo) & (wl_h <= hi)
    ma = (wl_a >= lo) & (wl_a <= hi)
    mx = (wl_x >= lo) & (wl_x <= hi)
    ax.plot(wl_h[mh], fl_h[mh], color="black", lw=1.0, label="HST")
    ax.plot(wl_a[ma], fl_a_n[ma], color="#3898EC", lw=0.9, alpha=0.85, label="A1c ε=1.00")
    ax.plot(wl_x[mx], fl_x_n[mx], color="#D97757", lw=0.9, alpha=0.85, label="X1_b3p0")
    ax.set_xlim(lo, hi)
    yh = fl_h[mh]
    if yh.size and np.isfinite(yh).any():
        ax.set_ylim(0, np.nanmax(yh) * 1.25)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

out = os.path.join(ROOT, "figures/compare_a1c_x1_vs_hst.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
