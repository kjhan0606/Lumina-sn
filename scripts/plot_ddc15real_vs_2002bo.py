#!/usr/bin/env python3
"""Overlay the REAL DDC15-composition runs (17.76d, scatter + macroatom) against
SN 2002bo (+0d obs), with the best baseline 162212 (scatter+binnedJ on the
einsteinFix DDC15-strat ref) as a reference line. norm [3500,9000].
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
BASE = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_dbj_A_162212/lumina_spectrum.csv"
SCAT = f"{ROOT}/logs/paperDDC15real_2002bo_ddc15real_scatter_162467/lumina_spectrum.csv"
MACR = f"{ROOT}/logs/paperDDC15real_2002bo_ddc15real_macroatom_162468/lumina_spectrum.csv"
LO, HI = 3500.0, 9000.0
BANDS = [(3500, 4500, "blue"), (4500, 5500, "green"),
         (5500, 7000, "red"), (7000, 9000, "NIR")]


def integ(w, f):
    s = (w >= LO) & (w <= HI)
    return np.trapezoid(f[s], w[s])


def load(p):
    d = pd.read_csv(p, comment='#')
    return d.iloc[:, 0].values, d.iloc[:, 1].values


def bandscore(w, f, ow, ofn):
    fi = np.interp(ow, w, f)
    fi = fi / integ(ow, fi)
    out = {}
    for a, b, nm in BANDS:
        m = (ow >= a) & (ow <= b)
        out[nm] = np.trapezoid(fi[m], ow[m]) / np.trapezoid(ofn[m], ow[m])
    return out


ow, of = load(OBS)
ofn = of / integ(ow, of)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ow, ofn, color="#222", lw=1.5, label="SN 2002bo (+0d, obs)", zorder=5)

for path, color, name in [(BASE, "#999999", "162212 baseline (scatter, DDC15-strat)"),
                          (SCAT, "#3898EC", "ddc15real scatter (162467)"),
                          (MACR, "#D94545", "ddc15real macroatom (162468)")]:
    w, f = load(path)
    fn = f / integ(w, f)
    bs = bandscore(w, f, ow, ofn)
    l1 = sum(abs(v - 1) for v in bs.values())
    lab = f"{name}  [" + " ".join(f"{k}{v:.2f}" for k, v in bs.items()) + f" L1={l1:.2f}]"
    ax.plot(w, fn, color=color, lw=1.1, alpha=0.9, label=lab)
    print(f"{name}: " + "  ".join(f"{k}={v:.3f}" for k, v in bs.items()) + f"  L1={l1:.3f}")

for a, b, nm in BANDS:
    ax.axvspan(a, b, alpha=0.04, color="gray")
ax.set_xlim(3000, 9500)
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Normalized flux [3500,9000]")
ax.set_title("Real DDC15 composition (17.76d) vs SN 2002bo +0d")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.2)
out = f"{ROOT}/figures/2026-06-03_ddc15real_scatter_macroatom_vs_2002bo.png"
fig.tight_layout()
fig.savefig(out, dpi=130)
print("saved", out)
