#!/usr/bin/env python3
"""Overlay best baseline (162212 scatter+binnedJ) vs SN 2002bo (+0d), norm [3500,9000]."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
RUN = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_dbj_A_162212/lumina_spectrum.csv"
LO, HI = 3500.0, 9000.0

def integ(w, f):
    s = (w >= LO) & (w <= HI)
    return np.trapezoid(f[s], w[s])

obs = pd.read_csv(OBS, comment='#')
ow, of = obs.iloc[:, 0].values, obs.iloc[:, 1].values
m = pd.read_csv(RUN)
mw, mf = m.iloc[:, 0].values, m.iloc[:, 1].values

ofn = of / integ(ow, of)
mfn = mf / integ(mw, mf)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(ow, ofn, color="#222", lw=1.3, label="SN 2002bo (+0d, obs)")
ax.plot(mw, mfn, color="#3898EC", lw=1.3, label="LUMINA 162212 (scatter+binnedJ)")
for a, b, nm in [(3500,4500,"blue 0.845"), (4500,5500,"green 1.098"),
                 (5500,7000,"red 1.030"), (7000,9000,"NIR 1.029")]:
    ax.axvspan(a, b, alpha=0.05, color="gray")
    ax.text((a+b)/2, ax.get_ylim()[1]*0.92 if False else 0, nm, fontsize=7,
            ha="center", va="bottom", rotation=90, color="#555")
ax.set_xlim(3000, 9500)
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Normalized flux [3500,9000]")
ax.set_title("Best baseline: scatter + binned-J + diffusive-BC + DDC15-strat  vs  SN 2002bo +0d")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)
out = f"{ROOT}/figures/2026-06-03_162212_scatter_binnedj_vs_2002bo.png"
fig.tight_layout(); fig.savefig(out, dpi=130)
print("saved", out)
