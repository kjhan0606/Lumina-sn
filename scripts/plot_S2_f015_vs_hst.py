#!/usr/bin/env python3
"""S2_f015 (Si II opt A_ul × 0.15 on Q1b r03 stack) vs HST stitched B-max + TARDIS."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TAR  = f"{ROOT}/data/sn2011fe/tardis_spectrum.csv"
LUM  = f"{ROOT}/logs/ddc15S2_156421_ddc15S2_S2_f015/lumina_spectrum_formal.csv"
OUT  = f"{ROOT}/figures/S2_f015_vs_hst.png"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return np.trapz(flu[sel], lam[sel])

def baseline_norm(lam, flu, fwhm=20000.0, lo=3000.0, hi=8000.0):
    sel = (lam>=lo) & (lam<=hi)
    sl, sf = lam[sel], flu[sel]
    dlam = np.median(np.diff(sl)); mid = 0.5*(lo+hi)
    sigma = (fwhm/C_KMS)*mid/2.355/dlam
    base = gaussian_filter1d(sf, sigma, mode='nearest')
    return sl, sf/base

h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
t = pd.read_csv(TAR)
tlam, tflu = t["wavelength_A"].values, t["flux_erg_s_A"].values
l = pd.read_csv(LUM)
llam, lflu = l["wavelength_angstrom"].values, l["flux"].values

# Green-band [4500, 5800] integral normalization to HST units
gH = band_int(hlam, hflu, 4500, 5800)
gT = band_int(tlam, tflu, 4500, 5800)
gL = band_int(llam, lflu, 4500, 5800)
tflu_n = tflu * (gH/gT)
lflu_n = lflu * (gH/gL)

# Baseline-norm RMS @ 20k FWHM in [3000, 8000]
def rms_bn(lam, flu):
    # Common grid: use HST grid in band as target
    selH = (hlam>=3000) & (hlam<=8000)
    hl_b, hf_b = baseline_norm(hlam[selH], hflu[selH])
    # Interp model onto hl_b
    selM = (lam>=2900) & (lam<=8100)
    ml_b, mf_b = baseline_norm(lam[selM], flu[selM])
    common = (hl_b >= ml_b[0]) & (hl_b <= ml_b[-1])
    mi = np.interp(hl_b[common], ml_b, mf_b)
    return float(np.sqrt(np.mean((hf_b[common] - mi)**2)))

rms_L = rms_bn(llam, lflu_n)
rms_T = rms_bn(tlam, tflu_n)

# Trough zoom panels
TROUGHS = [
    ("Si II 4130", 3950, 4250),
    ("Si II 5972", 5700, 6050),
    ("Si II 6355", 5950, 6450),
    ("Ca II H&K", 3650, 4000),
    ("Mg II 4481", 4300, 4600),
    ("Fe II m42 5018", 4750, 5100),
    ("Fe II 5169", 4900, 5250),
    ("O I 7773",  7400, 7900),
    ("Ca II IR",  7900, 8800),
]

fig = plt.figure(figsize=(16, 11), facecolor="white")
gs  = fig.add_gridspec(4, 3, height_ratios=[1.6, 1.0, 1.0, 1.0], hspace=0.42, wspace=0.22)

# Top: full overlay
ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu, color="black",   lw=1.2, label="HST B-max (stitched)")
ax.plot(tlam, tflu_n, color="tab:blue", lw=1.1, alpha=0.85, label=f"TARDIS  RMS_bn={rms_T:.3f}")
ax.plot(llam, lflu_n, color="crimson", lw=1.2, alpha=0.95, label=f"LUMINA S2_f015 RMS_bn={rms_L:.3f}")
ax.set_xlim(2800, 9200); ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title(f"S2_f015 (Si II opt A_ul ×0.15) vs HST  —  RMS_bn@F20k = {rms_L:.4f}  (T1 ≤0.20 target)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax.grid(alpha=0.3)
# Mark trough windows
for nm, lo, hi in TROUGHS:
    ax.axvspan(lo, hi, alpha=0.06, color="gold")

# 9 zoom panels (3x3 grid in rows 1..3)
for i, (nm, lo, hi) in enumerate(TROUGHS):
    r = 1 + i//3; c = i%3
    ax = fig.add_subplot(gs[r, c])
    selH = (hlam>=lo) & (hlam<=hi)
    selL = (llam>=lo) & (llam<=hi)
    selT = (tlam>=lo) & (tlam<=hi)
    ax.plot(hlam[selH], hflu[selH],     color="black",   lw=1.1, label="HST")
    ax.plot(tlam[selT], tflu_n[selT],   color="tab:blue", lw=0.9, alpha=0.8, label="TARDIS")
    ax.plot(llam[selL], lflu_n[selL],   color="crimson", lw=1.1, alpha=0.95, label="S2_f015")
    ax.set_title(nm, fontsize=10)
    ax.set_xlim(lo, hi)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

fig.suptitle("SN 2011fe B-max — S2_f015 LUMINA model vs HST stitched + TARDIS",
             y=0.995, fontsize=12, weight="bold")
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"  rms_bn LUMINA S2_f015 = {rms_L:.4f}")
print(f"  rms_bn TARDIS         = {rms_T:.4f}")
print(f"  green-band int: HST={gH:.3e}  TARDIS={gT:.3e}  LUMINA={gL:.3e}")
