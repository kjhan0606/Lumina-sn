#!/usr/bin/env python3
"""Blue/NIR comparison: 162212 (scatter+binnedJ, best) vs fixtrad macroatom
(162323, fluorescence on FROZEN 162212 thermal structure) vs SN 2002bo (+0d).
Decisive negative result: with T_rad/W frozen identical, macroatom STILL collapses
(blue 0.062, NIR 2.34) -> over-redshift is in the redistribution itself, not the
(W,T_rad) estimator feedback loop. Norm [3500,9000]; models lightly smoothed."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS   = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
SCAT  = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_dbj_A_162212/lumina_spectrum.csv"
MACRO = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_fixtrad_B_162323/lumina_spectrum.csv"
LO, HI = 3500.0, 9000.0

def norm(w, f):
    s = (w >= LO) & (w <= HI)
    return f / np.trapezoid(f[s], w[s])

def smooth(f, win=9):
    if win < 3: return f
    k = np.ones(win) / win
    return np.convolve(f, k, mode="same")

obs = pd.read_csv(OBS, comment='#'); ow, of = obs.iloc[:,0].values, obs.iloc[:,1].values
sc  = pd.read_csv(SCAT);  sw, sf = sc.iloc[:,0].values, sc.iloc[:,1].values
ma  = pd.read_csv(MACRO); mw, mf = ma.iloc[:,0].values, ma.iloc[:,1].values

ofn = norm(ow, of)
sfn = smooth(norm(sw, sf)); mfn = smooth(norm(mw, mf))

fig, (axb, axn) = plt.subplots(1, 2, figsize=(15, 6))

for ax, (a, b), title in [
    (axb, (3300, 4700), "BLUE [3500-4500]:  162212=0.845   macroatom(frozen-T)=0.062"),
    (axn, (7000, 9500), "NIR [7000-9000]:  162212=1.029   macroatom(frozen-T)=2.342")]:
    sel_o = (ow >= a) & (ow <= b)
    ax.fill_between(ow[sel_o], ofn[sel_o], color="0.85", zorder=0)
    ax.plot(ow, ofn, color="#222", lw=1.7, label="SN 2002bo (+0d, obs)")
    ax.plot(sw, sfn, color="#1f77b4", lw=1.8, label="162212  scatter+binnedJ  (best, L1=2.44)")
    ax.plot(mw, mfn, color="#d62728", lw=1.8,
            label="162323  macroatom + FROZEN 162212 T_rad/W  (L1=3.92)")
    ax.set_xlim(a, b)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Wavelength (Å)", fontsize=11)
    ax.grid(alpha=0.2)
    if (a, b) == (3300, 4700):
        ax.axvspan(3500, 4500, color="#1f77b4", alpha=0.06)
        ax.set_ylabel("Normalized flux (area-norm [3500,9000])", fontsize=11)
        ax.annotate("blue STILL crushed\n(0.062) despite\nidentical T_rad/W",
                    xy=(4000, 0.00006), color="#d62728", fontsize=10,
                    ha="center", fontweight="bold")
    else:
        ax.axvspan(7000, 9000, color="#d62728", alpha=0.06)
        ax.annotate("NIR STILL piles up\n(2.34) — redistribution,\nNOT estimator feedback",
                    xy=(8200, 0.00045), color="#d62728", fontsize=10,
                    ha="center", fontweight="bold")

axb.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
fig.suptitle("Freezing thermal structure does NOT cure fluorescence collapse: "
             "over-redshift is intrinsic to macroatom redistribution",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
out = f"{ROOT}/figures/2026-06-03_fixtrad_macroatom_bluenir_vs_162212.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
