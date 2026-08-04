#!/usr/bin/env python3
"""Clean 3-way overlay vs SN 2002bo (+0d): obs, scatter+binnedJ (162212, sane),
macroatom+binnedJ (162300, fluorescence collapse). Norm [3500,9000].
Models lightly smoothed to suppress MC noise."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
SCAT = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_dbj_A_162212/lumina_spectrum.csv"
MACRO = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_maj_B_162300/lumina_spectrum.csv"
LO, HI = 3500.0, 9000.0

def norm(w, f):
    s = (w >= LO) & (w <= HI)
    return f / np.trapezoid(f[s], w[s])

def smooth(f, win=9):
    if win < 3: return f
    k = np.ones(win) / win
    return np.convolve(f, k, mode="same")

obs = pd.read_csv(OBS, comment='#'); ow, of = obs.iloc[:,0].values, obs.iloc[:,1].values
sc = pd.read_csv(SCAT); sw, sf = sc.iloc[:,0].values, sc.iloc[:,1].values
ma = pd.read_csv(MACRO); mw, mf = ma.iloc[:,0].values, ma.iloc[:,1].values

ofn = norm(ow, of)
sfn = smooth(norm(sw, sf)); mfn = smooth(norm(mw, mf))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.13})

# --- top: the sane best baseline vs obs ---
ax1.fill_between(ow, ofn, color="0.85", zorder=0)
ax1.plot(ow, ofn, color="#222", lw=1.6, label="SN 2002bo  (+0d, observed)")
ax1.plot(sw, sfn, color="#1f77b4", lw=1.7,
         label="162212  scatter + binned-J   (L1 = 2.44)")
ax1.set_title("Best baseline (scatter + binned-J) reproduces per-band flux; "
              "blue[3500-4500]=0.85 + feature positions are the residual defects",
              fontsize=12, pad=8)
ax1.legend(loc="upper right", fontsize=11, framealpha=0.9)
for a, b in [(3500, 4500), (7000, 9000)]:
    ax1.axvspan(a, b, color="#1f77b4", alpha=0.05)

# --- bottom: the fluorescence collapse ---
ax2.fill_between(ow, ofn, color="0.85", zorder=0)
ax2.plot(ow, ofn, color="#222", lw=1.6, label="SN 2002bo  (+0d, observed)")
ax2.plot(mw, mfn, color="#d62728", lw=1.7,
         label="162300  macroatom + binned-J   (L1 = 4.33,  collapse)")
ax2.set_title("Fluorescence (macroatom) catastrophically over-redshifts: blue crushed → NIR pile-up "
              "(T_rad estimator collapses ~4000K despite T_inner = 10074K)",
              fontsize=12, pad=8)
ax2.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax2.axvspan(3500, 4500, color="#1f77b4", alpha=0.06)
ax2.axvspan(7000, 9000, color="#d62728", alpha=0.06)
ax2.annotate("blue\ncrushed", xy=(4000, 0.00012), color="#1f77b4",
             fontsize=10, ha="center", fontweight="bold")
ax2.annotate("NIR pile-up", xy=(8000, 0.00030), color="#d62728",
             fontsize=10, ha="center", fontweight="bold")

for ax in (ax1, ax2):
    ax.set_xlim(3300, 9500)
    ax.set_ylim(0, 0.00052)
    ax.set_ylabel("Normalized flux\n(area-normalized [3500,9000])", fontsize=10)
    ax.grid(alpha=0.18)
ax2.set_xlabel("Wavelength (Å)", fontsize=11)

out = f"{ROOT}/figures/2026-06-03_fluorescence_3way_162212_162300_vs_2002bo.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
