#!/usr/bin/env python3
"""FI ablation overview — single-panel comparison + flux-balance residual.

Goal: show side-by-side that FI_cont fixes line-shape (RMS_bn↓) while
absolute red continuum stays high or even rises — the line-only formal
integral was masking, not creating, the red excess.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TAR  = f"{ROOT}/data/sn2011fe/tardis_spectrum.csv"
FIBASE = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_FI_base/lumina_spectrum_formal.csv"
FICONT = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_FI_cont/lumina_spectrum_formal.csv"
FIBOTH = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_FI_both/lumina_spectrum_formal.csv"
OUT  = f"{ROOT}/figures/FI_overview.png"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return np.trapezoid(flu[sel], lam[sel])

def load(p):
    df = pd.read_csv(p)
    if "flux" in df.columns:
        return df["wavelength_angstrom"].values, df["flux"].values
    if "flux_erg_s_A" in df.columns:
        return df["wavelength_A"].values, df["flux_erg_s_A"].values
    return df["wavelength_angstrom"].values, df["flux_erg_s_cm2_angstrom"].values

def smooth(lam, flu, fwhm=20000.0):
    dlam = np.median(np.diff(lam)); mid = 0.5*(lam[0]+lam[-1])
    sigma = (fwhm/C_KMS)*mid/2.355/dlam
    return gaussian_filter1d(flu, sigma, mode='nearest')

# Load
h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
gH = band_int(hlam, hflu, 4500, 5800)

def norm(p):
    lam, flu = load(p)
    g = band_int(lam, flu, 4500, 5800)
    return lam, flu * (gH/g)

tlam, tflu = norm(TAR)
blam, bflu = norm(FIBASE)
clam, cflu = norm(FICONT)
xlam, xflu = norm(FIBOTH)

# Smooth baselines on each
selH = (hlam>=2900) & (hlam<=9300); sHl = hlam[selH]; sH = smooth(sHl, hflu[selH])
def baseline_on(lam, flu):
    sel = (lam>=2900) & (lam<=9300)
    return lam[sel], smooth(lam[sel], flu[sel])
slT, smT = baseline_on(tlam, tflu)
slB, smB = baseline_on(blam, bflu)
slC, smC = baseline_on(clam, cflu)
slX, smX = baseline_on(xlam, xflu)

fig = plt.figure(figsize=(15, 9), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.9, 0.9], hspace=0.20)

# --- Top: raw flux overlay ---
ax = fig.add_subplot(gs[0])
ax.plot(hlam, hflu, color="black",    lw=1.4, label="HST B-max")
ax.plot(tlam, tflu, color="tab:blue", lw=0.9, alpha=0.8, label="TARDIS  RMS_bn=0.140")
ax.plot(blam, bflu, color="dimgray",  lw=0.9, alpha=0.85, label="FI_base  RMS_bn=0.218  (line-only)")
ax.plot(clam, cflu, color="crimson",  lw=1.1, alpha=0.95, label="FI_cont  RMS_bn=0.146  (+ e-scatter + W·B(T_rad))")
ax.plot(xlam, xflu, color="darkorange", lw=1.1, alpha=0.9,
        label="FI_both  RMS_bn=0.140  (+ inner Planck W=0.5)")
ax.set_xlim(2800, 9300)
ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("S2_f015 base + formal-integral algorithmic ablation  (job 156433, 600K×10)",
             fontsize=12, weight="bold")
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
RED_BANDS = [(5800,6500,"opt"),(6500,7400,"redI"),(7400,8200,"redII"),(8200,9000,"redIII")]
for lo, hi, nm in RED_BANDS:
    ax.axvspan(lo, hi, color="gold", alpha=0.05)

# --- Middle: smooth baseline ratio to HST ---
ax = fig.add_subplot(gs[1])
ax.axhline(1.0, color='black', lw=0.5)
def ratio(slM, smM, color, label):
    common = (slM >= sHl[0]) & (slM <= sHl[-1])
    hi = np.interp(slM[common], sHl, sH)
    ax.plot(slM[common], smM[common]/hi, color=color, lw=1.3, label=label)
ratio(slT, smT, "tab:blue", "TARDIS")
ratio(slB, smB, "dimgray",  "FI_base")
ratio(slC, smC, "crimson",  "FI_cont")
ratio(slX, smX, "darkorange","FI_both")
ax.set_xlim(2800, 9300); ax.set_ylim(0.5, 2.6)
ax.set_ylabel("baseline / HST")
ax.set_title("Smooth-baseline ratio (Gauss 20k km/s) — continuum-only excess",
             fontsize=10)
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=9, ncol=4)
for lo, hi, nm in RED_BANDS:
    ax.axvspan(lo, hi, color="gold", alpha=0.05)

# --- Bottom: band-ratio bar chart ---
ax = fig.add_subplot(gs[2])
band_centers = [(lo+hi)/2 for lo, hi, _ in RED_BANDS]
band_names   = [nm for _,_,nm in RED_BANDS]
def band_ratios(lam, flu):
    rs = []
    for lo, hi, _ in RED_BANDS:
        rs.append(band_int(lam, flu, lo, hi) / band_int(hlam, hflu, lo, hi))
    return rs
rT = band_ratios(tlam, tflu); rB = band_ratios(blam, bflu)
rC = band_ratios(clam, cflu); rX = band_ratios(xlam, xflu)
x = np.arange(len(RED_BANDS)); w = 0.18
ax.axhline(1.0, color='black', lw=0.6)
ax.bar(x-1.5*w, rT, w, color="tab:blue",  label="TARDIS")
ax.bar(x-0.5*w, rB, w, color="dimgray",   label="FI_base")
ax.bar(x+0.5*w, rC, w, color="crimson",   label="FI_cont")
ax.bar(x+1.5*w, rX, w, color="darkorange",label="FI_both")
for xi, ratios in zip(x, zip(rT, rB, rC, rX)):
    for j, r in enumerate(ratios):
        ax.text(xi + (j-1.5)*w, r+0.04, f"{r:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{nm}\n[{lo},{hi}]Å" for lo,hi,nm in RED_BANDS])
ax.set_ylim(0, 2.6)
ax.set_ylabel("band flux / HST")
ax.set_title("Red-band absolute flux ratio — FI_cont/both make red MORE excess vs HST",
             fontsize=10)
ax.legend(loc="upper left", fontsize=9, ncol=4)
ax.grid(alpha=0.25, axis='y')

fig.suptitle("FI ablation overview — line-shape gains (RMS_bn↓36%) hide real red excess",
             y=0.995, fontsize=12.5, weight="bold")
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"\nRMS_bn vs baseline-norm metric:")
print(f"  FI_base  0.218   FI_cont  0.146   FI_both  0.140   TARDIS  0.140")
print(f"\nRed-band absolute flux (model/HST):")
print(f"{'band':>10s}  {'TARDIS':>7s}  {'FI_base':>8s}  {'FI_cont':>8s}  {'FI_both':>8s}")
for i, (lo, hi, nm) in enumerate(RED_BANDS):
    print(f"{nm:>10s}  {rT[i]:>7.3f}  {rB[i]:>8.3f}  {rC[i]:>8.3f}  {rX[i]:>8.3f}")
