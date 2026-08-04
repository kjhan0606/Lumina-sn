#!/usr/bin/env python3
"""Honest comparison: no self-normalization, no FWHM=20k smooth.

Three panels, all on absolute flux (linear or log):
  1. Linear flux full range — sees the UV catastrophe directly
  2. Log10(flux) full range — UV gap visible AND optical detail readable
  3. Log10(model/obs) — direct ratio per wavelength
"""
import glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
TAG = "phase+03.7"
HST_GRATS = [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]
JOB = 157643
RUN_DIR = f"{ROOT}/logs/oTri_prod_vi11600_L2p45_de10p5_p+3.7d_{JOB}"
LABEL = f"LUMINA #281 oTri (job {JOB})"

def band_int(lam, flu, lo, hi):
    s = (lam >= lo) & (lam <= hi)
    return float(np.trapezoid(flu[s], lam[s]))

# --- load ---
m = pd.read_csv(f"{RUN_DIR}/lumina_spectrum_formal.csv")
mlam, mflu = m["wavelength_angstrom"].values, m["flux"].values

sn = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/sn2011fe_p4d2d.csv", comment="#")
slam, sflu = sn["wavelength_angstrom"].values, sn["flux_erg_s_cm2_angstrom"].values

parts = []
for grat, w_lo, w_hi in HST_GRATS:
    files = sorted(glob.glob(f"{ROOT}/data/sn2011fe/hst_uv/CCD_{grat}_*{TAG}*sx1.csv"))
    if not files:
        continue
    d = pd.read_csv(files[0])
    sel = (d["wavelength_angstrom"] >= w_lo) & (d["wavelength_angstrom"] <= w_hi)
    parts.append(d.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom"]])
hst = pd.concat(parts).sort_values("wavelength_angstrom").reset_index(drop=True)
hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
hlam, hflu = hst["wavelength_angstrom"].values, hst["flux_erg_s_cm2_angstrom"].values

# --- ONE anchor normalization at [4500,5800] (same as RMS_bn uses internally) ---
gH = band_int(hlam, hflu, 4500, 5800)
gM = band_int(mlam, mflu, 4500, 5800)
mn = mflu * (gH / gM)

# --- figure ---
fig, axes = plt.subplots(3, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [3, 3, 2.4]})

ax = axes[0]
ax.plot(hlam, hflu, lw=0.9, color="black", alpha=0.85, label="HST stitched (obs)")
ax.plot(slam, sflu, lw=0.6, color="goldenrod", alpha=0.75, label="Snifs p+4.2d (obs)")
ax.plot(mlam, mn, lw=0.9, color="crimson", alpha=0.85, label=LABEL + " (anchor: 4500-5800Å)")
ax.set_xlim(1600, 10000)
ax.set_ylim(0, max(hflu.max(), sflu.max()) * 1.15)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("Honest linear flux — UV catastrophe visible")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

ax = axes[1]
ax.semilogy(hlam, hflu, lw=0.9, color="black", alpha=0.85, label="HST obs")
ax.semilogy(slam, sflu, lw=0.6, color="goldenrod", alpha=0.75, label="Snifs obs")
ax.semilogy(mlam, mn, lw=0.9, color="crimson", alpha=0.85, label="LUMINA #281 (anchor-normalized)")
ax.set_xlim(1600, 10000)
all_flu = np.concatenate([hflu, sflu, mn[(mlam>1600)&(mlam<10000)]])
all_flu = all_flu[all_flu > 0]
ax.set_ylim(all_flu.min() * 0.5, all_flu.max() * 3)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$] (log)")
ax.set_title("Log flux — same data, UV gap is ~1.5 dex (30×) over HST")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25, which="both")

ax = axes[2]
common = (hlam >= mlam[0]) & (hlam <= mlam[-1])
mi = np.interp(hlam[common], mlam, mn)
ratio = mi / np.maximum(hflu[common], 1e-30)
ax.semilogy(hlam[common], ratio, lw=0.7, color="crimson", alpha=0.85)
ax.axhline(1.0, lw=1.0, color="black", ls="--", alpha=0.6, label="model = obs")
ax.axhline(2.0, lw=0.5, color="orange", ls=":", alpha=0.5, label="2× / 0.5×")
ax.axhline(0.5, lw=0.5, color="orange", ls=":", alpha=0.5)
ax.axhline(10., lw=0.5, color="red", ls=":", alpha=0.5, label="10× / 0.1×")
ax.axhline(0.1, lw=0.5, color="red", ls=":", alpha=0.5)
ax.set_xlim(1600, 10000)
ax.set_ylim(0.05, 200)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("model / obs (log)")
ax.set_title("Direct model/obs ratio — Panel 3 done honestly. RMS_bn divides this away.")
ax.axvspan(1700, 3000, alpha=0.08, color="purple")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
out = f"{ROOT}/figures/2026-05-25_best_157643_HONEST_vs_sn2011fe_p3p7d.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")
