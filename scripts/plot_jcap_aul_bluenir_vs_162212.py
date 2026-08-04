#!/usr/bin/env python3
"""Blue/NIR comparison vs SN 2002bo (+0d): 162212 scatter(best) vs the macroatom
fix attempts. 162323 = 1/N table + uncapped J-pump (collapse). 162390/162391 =
J_CAP_FACTOR 1.0/5.0 (option 1). 162401 = A_ul-weighted static table + env-bug
fix, dynamic OFF (option 2/3). Decisive: the J-cap runs STILL collapse
(blue~0.07, NIR~2.3) AND show p_iup/p_idn=16.08 identical to uncapped -> the cap
did not engage; the A_ul-static run ran T_inner away (120885K) emptying the
optical band. Norm [3500,9000]; models lightly smoothed."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS  = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
runs = [
 ("SN 2002bo (+0d obs)", OBS, "#222", 1.9, True),
 ("162212 scatter (best, L1=0.31)",
  f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_dbj_A_162212/lumina_spectrum.csv", "#1f77b4", 1.8, False),
 ("162323 1/N + J-pump (L1=3.38)",
  f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_fixtrad_B_162323/lumina_spectrum.csv", "#d62728", 1.5, False),
 ("162390 J-cap=1.0 (L1=3.31)",
  f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_jcap1p0_162390/lumina_spectrum.csv", "#ff7f0e", 1.5, False),
 ("162391 J-cap=5.0 (L1=3.28)",
  f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_jcap5p0_162391/lumina_spectrum.csv", "#9467bd", 1.5, False),
]
LO, HI = 3500.0, 9000.0

def load(p, obs):
    df = pd.read_csv(p, comment='#') if obs else pd.read_csv(p)
    return df.iloc[:,0].values.astype(float), df.iloc[:,1].values.astype(float)
def norm(w, f):
    s = (w >= LO) & (w <= HI); return f / np.trapezoid(f[s], w[s])
def smooth(f, win=9):
    k = np.ones(win)/win; return np.convolve(f, k, mode="same")

fig, (axb, axn) = plt.subplots(1, 2, figsize=(15, 6))
for ax, (a, b), ttl in [
    (axb, (3300, 4700), "BLUE [3500-4500]:  scatter=0.845  vs  J-cap 1.0/5.0 = 0.074/0.068  (STILL crushed)"),
    (axn, (7000, 9500), "NIR [7000-9000]:  scatter=1.029  vs  J-cap 1.0/5.0 = 2.33/2.25  (STILL piled)")]:
    for name, p, c, lw, obs in runs:
        w, f = load(p, obs); fn = norm(w, f)
        if not obs: fn = smooth(fn)
        ax.plot(w, fn, color=c, lw=lw, label=name)
    ax.set_xlim(a, b); ax.set_title(ttl, fontsize=11, pad=8)
    ax.set_xlabel("Wavelength (Å)", fontsize=11); ax.grid(alpha=0.2)
    if a == 3300:
        ax.axvspan(3500, 4500, color="#1f77b4", alpha=0.06)
        ax.set_ylabel("Normalized flux (area-norm [3500,9000])", fontsize=11)
        ax.annotate("J-cap (orange/purple)\nlands on the collapsed\nred curve — cap did\nNOT engage\n(p_iup/p_idn=16.08\nidentical at cap 1 & 5)",
                    xy=(4050, 0.00006), color="#d62728", fontsize=9.5, ha="center", fontweight="bold")
    else:
        ax.axvspan(7000, 9000, color="#d62728", alpha=0.06)
axb.legend(loc="upper right", fontsize=8.8, framealpha=0.92)
fig.suptitle("J-cap (option 1) does NOT cure the macroatom over-redshift — cap never binds; "
             "A_ul-static (option 2/3) ran T_inner away to 120885K (optical empty)",
             fontsize=12.5, fontweight="bold", y=1.00)
fig.tight_layout()
out = f"{ROOT}/figures/2026-06-03_jcap_aul_bluenir_vs_162212.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
