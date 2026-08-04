#!/usr/bin/env python3
"""Plot top-5 MC-ground-truth champions vs HST + TARDIS, using MC packet spectrum
(lumina_spectrum.csv). Validates that the new ranking is physically real."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TAR  = f"{ROOT}/data/sn2011fe/tardis_spectrum.csv"
OUT  = f"{ROOT}/figures/MC_top5_vs_hst.png"
C_KMS = 2.998e5

CHAMPIONS = [
    ("H2 ε=0.7 redonly",     0.182, "logs/ddc15H2_156031_ddc15H2_epsUV0.7_redonly/lumina_spectrum.csv", "crimson"),
    ("H1b ε=0.10",            0.187, "logs/ddc15H1b_156018_ddc15H1b_epsUV0.10/lumina_spectrum.csv",     "darkorange"),
    ("C2 xFeO=0.15",          0.188, "logs/ddc15C2_155756_ddc15C2_xFeO0.15/lumina_spectrum.csv",        "tab:green"),
    ("H2 ε=0.0 redonly",      0.189, "logs/ddc15H2_156031_ddc15H2_epsUV0.0_redonly/lumina_spectrum.csv","tab:purple"),
    ("S2_f015 (FI champ)",    0.200, "logs/ddc15S2_156421_ddc15S2_S2_f015/lumina_spectrum.csv",         "tab:gray"),
]

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return float(np.trapezoid(flu[sel], lam[sel]))

def smooth(lam, flu, fwhm=20000.0):
    dlam = np.median(np.diff(lam)); mid = 0.5*(lam[0]+lam[-1])
    sigma = (fwhm/C_KMS)*mid/2.355/dlam
    return gaussian_filter1d(flu, sigma, mode='nearest')

h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
gH = band_int(hlam, hflu, 4500, 5800)

t = pd.read_csv(TAR)
tlam, tflu = t["wavelength_A"].values, t["flux_erg_s_A"].values
tflu *= gH / band_int(tlam, tflu, 4500, 5800)

TROUGHS = [
    ("Si II 4130", 3950, 4250),
    ("Si II 5972", 5700, 6050),
    ("Si II 6355", 5950, 6450),
    ("Ca II H&K", 3650, 4000),
    ("Mg II 4481", 4300, 4600),
    ("Fe II 5169", 4900, 5250),
    ("O I 7773",  7400, 7900),
    ("Ca II IR",  7900, 8800),
]

fig = plt.figure(figsize=(16, 12), facecolor="white")
gs  = fig.add_gridspec(4, 2, height_ratios=[1.8, 0.9, 1.0, 1.0], hspace=0.40, wspace=0.18)

# Top: 4-panel comparison
ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu, color="black", lw=1.4, label="HST B-max")
ax.plot(tlam, tflu, color="tab:blue", lw=0.9, alpha=0.85, label="TARDIS (MC=0.164)")
for name, mc_rms, path, color in CHAMPIONS:
    df = pd.read_csv(f"{ROOT}/{path}")
    lam, flu = df["wavelength_angstrom"].values, df["flux"].values
    g = band_int(lam, flu, 4500, 5800)
    flu_n = flu * (gH/g)
    ax.plot(lam, flu_n, color=color, lw=1.0, alpha=0.9,
            label=f"{name}  MC={mc_rms:.3f}")
ax.set_xlim(2800, 9300)
ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("Top-5 MC-ground-truth champions vs HST  (MC packet spectrum, NOT formal integral)",
             fontsize=12, weight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.92)
ax.grid(alpha=0.25)

# Middle: smooth baseline / HST ratio
selH = (hlam>=3000) & (hlam<=9000); shl = hlam[selH]
sH = smooth(shl, hflu[selH])
ax = fig.add_subplot(gs[1, :])
ax.axhline(1.0, color='black', lw=0.5)
for name, mc_rms, path, color in CHAMPIONS:
    df = pd.read_csv(f"{ROOT}/{path}")
    lam, flu = df["wavelength_angstrom"].values, df["flux"].values
    g = band_int(lam, flu, 4500, 5800)
    flu_n = flu * (gH/g)
    selM = (lam>=3000) & (lam<=9000)
    ml, mf = lam[selM], flu_n[selM]
    smM = smooth(ml, mf)
    common = (shl>=ml[0]) & (shl<=ml[-1])
    himi = np.interp(shl[common], ml, smM)
    ax.plot(shl[common], himi/sH[common], color=color, lw=1.2, alpha=0.85, label=name)
selT_ = (tlam>=3000) & (tlam<=9000)
smT = smooth(tlam[selT_], tflu[selT_])
common = (shl>=tlam[selT_][0]) & (shl<=tlam[selT_][-1])
himi = np.interp(shl[common], tlam[selT_], smT)
ax.plot(shl[common], himi/sH[common], color="tab:blue", lw=1.0, alpha=0.7, label="TARDIS")
ax.set_xlim(2800, 9300); ax.set_ylim(0.4, 3.6)
ax.set_ylabel("smooth ratio / HST")
ax.set_title("Continuum-only excess (MC) — TARDIS stays near 1, LUMINA all stretch red 1.5-3×",
             fontsize=10)
ax.legend(loc="upper left", fontsize=9, ncol=3)
ax.grid(alpha=0.25)

# Bottom: 8 trough zoom
for i, (nm, lo, hi) in enumerate(TROUGHS):
    r = 2 + i//4; c = i % 4
    ax = fig.add_subplot(gs[r:r+1, c // 2:c // 2 + 1])  # ignore, just do 4 per row
for i, (nm, lo, hi) in enumerate(TROUGHS):
    ax = fig.add_subplot(gs[2 + i // 4, i % 4 if (i % 4) < 2 else (i % 4) - 2])

# Reset bottom layout — make 2 rows x 4 cols of zooms
fig.clear()
gs2 = fig.add_gridspec(4, 4, height_ratios=[1.8, 0.9, 1.0, 1.0], hspace=0.40, wspace=0.20)
ax = fig.add_subplot(gs2[0, :])
ax.plot(hlam, hflu, color="black", lw=1.4, label="HST")
ax.plot(tlam, tflu, color="tab:blue", lw=0.9, alpha=0.85, label="TARDIS (MC=0.164)")
for name, mc_rms, path, color in CHAMPIONS:
    df = pd.read_csv(f"{ROOT}/{path}")
    lam, flu = df["wavelength_angstrom"].values, df["flux"].values
    g = band_int(lam, flu, 4500, 5800)
    ax.plot(lam, flu*(gH/g), color=color, lw=1.0, alpha=0.9,
            label=f"{name}  MC={mc_rms:.3f}")
ax.set_xlim(2800, 9300)
ax.set_ylabel(r"$F_\lambda$")
ax.set_title("Top-5 MC champions vs HST (MC packet spectrum)", fontsize=12, weight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs2[1, :])
ax.axhline(1.0, color='black', lw=0.5)
for name, mc_rms, path, color in CHAMPIONS:
    df = pd.read_csv(f"{ROOT}/{path}")
    lam, flu = df["wavelength_angstrom"].values, df["flux"].values
    g = band_int(lam, flu, 4500, 5800)
    flu_n = flu * (gH/g)
    selM = (lam>=3000) & (lam<=9000)
    ml, mf = lam[selM], flu_n[selM]
    smM = smooth(ml, mf)
    common = (shl>=ml[0]) & (shl<=ml[-1])
    himi = np.interp(shl[common], ml, smM)
    ax.plot(shl[common], himi/sH[common], color=color, lw=1.2, alpha=0.85, label=name)
common = (shl>=tlam[selT_][0]) & (shl<=tlam[selT_][-1])
himi = np.interp(shl[common], tlam[selT_], smT)
ax.plot(shl[common], himi/sH[common], color="tab:blue", lw=1.0, alpha=0.7, label="TARDIS")
ax.set_xlim(2800, 9300); ax.set_ylim(0.4, 3.6)
ax.set_ylabel("smooth/HST")
ax.set_title("Continuum ratio", fontsize=10)
ax.legend(loc="upper left", fontsize=8, ncol=4)
ax.grid(alpha=0.25)

for i, (nm, lo, hi) in enumerate(TROUGHS):
    r = 2 + i // 4
    c = i % 4
    ax = fig.add_subplot(gs2[r, c])
    selH_ = (hlam>=lo) & (hlam<=hi)
    selT_ = (tlam>=lo) & (tlam<=hi)
    ax.plot(hlam[selH_], hflu[selH_], color="black", lw=1.0, label="HST")
    ax.plot(tlam[selT_], tflu[selT_], color="tab:blue", lw=0.7, alpha=0.7, label="TARDIS")
    for name, mc_rms, path, color in CHAMPIONS:
        df = pd.read_csv(f"{ROOT}/{path}")
        lam, flu = df["wavelength_angstrom"].values, df["flux"].values
        g = band_int(lam, flu, 4500, 5800)
        flu_n = flu * (gH/g)
        sel = (lam>=lo) & (lam<=hi)
        ax.plot(lam[sel], flu_n[sel], color=color, lw=0.8, alpha=0.85)
    ax.set_title(nm, fontsize=9)
    ax.set_xlim(lo, hi)
    ax.grid(alpha=0.25)

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
