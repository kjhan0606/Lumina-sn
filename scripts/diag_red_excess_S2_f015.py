#!/usr/bin/env python3
"""Decompose >6000Å red excess of S2_f015 vs HST into:
  (1) Smooth continuum offset (Gauss baseline ratio)
  (2) Trough-fill from cascade (line-residual structure)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TAR  = f"{ROOT}/data/sn2011fe/tardis_spectrum.csv"
LUM  = f"{ROOT}/logs/ddc15S2_156421_ddc15S2_S2_f015/lumina_spectrum_formal.csv"
OUT  = f"{ROOT}/figures/S2_f015_red_excess_diag.png"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return np.trapezoid(flu[sel], lam[sel])

def smooth_baseline(lam, flu, fwhm_kms=20000.0):
    dlam = np.median(np.diff(lam))
    mid  = 0.5*(lam[0]+lam[-1])
    sigma = (fwhm_kms/C_KMS)*mid/2.355/dlam
    return gaussian_filter1d(flu, sigma, mode='nearest')

h = pd.read_csv(HST); hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
t = pd.read_csv(TAR); tlam, tflu = t["wavelength_A"].values, t["flux_erg_s_A"].values
l = pd.read_csv(LUM); llam, lflu = l["wavelength_angstrom"].values, l["flux"].values

# Green-band normalize TARDIS, LUMINA to HST in [4500,5800]
gH = band_int(hlam, hflu, 4500, 5800)
gT = band_int(tlam, tflu, 4500, 5800)
gL = band_int(llam, lflu, 4500, 5800)
tflu *= gH/gT
lflu *= gH/gL

# Resample LUMINA, TARDIS onto HST grid in [3000, 9200]
band_mask = (hlam>=3000) & (hlam<=9200)
xg = hlam[band_mask]
yH = hflu[band_mask]
yL = np.interp(xg, llam, lflu)
yT = np.interp(xg, tlam, tflu)

# Smooth (Gauss FWHM 20k km/s) — baseline continuum
sH = smooth_baseline(xg, yH)
sL = smooth_baseline(xg, yL)
sT = smooth_baseline(xg, yT)

# Band integrals
def report(lo, hi):
    iH = band_int(xg, yH, lo, hi)
    iT = band_int(xg, yT, lo, hi)
    iL = band_int(xg, yL, lo, hi)
    sH_= band_int(xg, sH, lo, hi)
    sT_= band_int(xg, sT, lo, hi)
    sL_= band_int(xg, sL, lo, hi)
    return dict(lo=lo, hi=hi,
                full_LH=iL/iH, full_TH=iT/iH,
                base_LH=sL_/sH_, base_TH=sT_/sH_)

bands = [
    ("green ref", 4500, 5800),
    ("optical",   5800, 6500),
    ("red I",     6500, 7400),
    ("red II",    7400, 8200),
    ("red III",   8200, 9000),
]
print(f"{'band':12s}  {'λ-range':14s}  {'full L/H':>9s}  {'base L/H':>9s}  {'trough-fill':>11s}  {'(TARDIS base T/H)':>17s}")
for nm, lo, hi in bands:
    r = report(lo, hi)
    fill = r['full_LH'] - r['base_LH']   # positive ⇒ troughs less deep ⇒ cascade fill
    print(f"{nm:12s}  [{lo:>4},{hi:>4}]  {r['full_LH']:>9.3f}  {r['base_LH']:>9.3f}  {fill:>+11.3f}  {r['base_TH']:>17.3f}")

# === Plot ===
fig, axs = plt.subplots(3, 1, figsize=(13, 10), facecolor="white",
                        gridspec_kw=dict(hspace=0.32, height_ratios=[1.3, 1, 1]))

ax = axs[0]
ax.plot(xg, yH, 'k',  lw=0.8, alpha=0.6, label='HST raw')
ax.plot(xg, sH, 'k',  lw=1.6, label='HST baseline (Gauss 20k)')
ax.plot(xg, yL, color='crimson', lw=0.7, alpha=0.6, label='LUMINA raw')
ax.plot(xg, sL, color='crimson', lw=1.6, label='LUMINA baseline')
ax.plot(xg, sT, color='tab:blue', lw=1.3, alpha=0.9, label='TARDIS baseline')
ax.set_xlim(3000, 9200); ax.grid(alpha=0.3)
ax.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
ax.set_title('S2_f015 red excess diagnosis  —  smooth baseline (continuum) vs raw (lines)')
ax.legend(fontsize=9, ncol=3, loc='upper right')
for nm, lo, hi in bands[1:]:
    ax.axvspan(lo, hi, color='gold', alpha=0.06)

# Middle: baseline ratio LUMINA/HST  (continuum-only excess)
ax = axs[1]
ax.axhline(1.0, color='black', lw=0.5)
ax.plot(xg, sL/sH, color='crimson', lw=1.4, label='LUMINA/HST  (smooth baseline)')
ax.plot(xg, sT/sH, color='tab:blue', lw=1.2, alpha=0.9, label='TARDIS/HST  (smooth baseline)')
ax.set_ylim(0.5, 1.8)
ax.set_xlim(3000, 9200); ax.grid(alpha=0.3)
ax.set_ylabel('baseline ratio')
ax.set_title('(continuum-only excess: this is the smooth-Planck level)')
ax.legend(fontsize=9, loc='upper right')
for nm, lo, hi in bands[1:]:
    ax.axvspan(lo, hi, color='gold', alpha=0.06)

# Bottom: raw - baseline, comparing structure (line residual)
ax = axs[2]
ax.axhline(0.0, color='black', lw=0.5)
ax.plot(xg, (yH-sH)/sH, 'k',         lw=0.9, alpha=0.85, label='HST (raw-base)/base')
ax.plot(xg, (yL-sL)/sL, color='crimson', lw=0.9, label='LUMINA (raw-base)/base')
ax.set_ylim(-0.7, 0.5)
ax.set_xlim(3000, 9200); ax.grid(alpha=0.3)
ax.set_xlabel('Wavelength [Å]')
ax.set_ylabel('(raw - base) / base')
ax.set_title('(line-residual: trough depth — if LUMINA shallower in red, cascade is filling troughs)')
ax.legend(fontsize=9, loc='upper right')
for nm, lo, hi in bands[1:]:
    ax.axvspan(lo, hi, color='gold', alpha=0.06)

fig.savefig(OUT, dpi=130, bbox_inches='tight')
print(f"\nsaved: {OUT}")
