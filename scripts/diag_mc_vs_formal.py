#!/usr/bin/env python3
"""Compare MC packet spectrum (lumina_spectrum.csv) vs formal integral
(lumina_spectrum_formal.csv) for FI ablation cells.

If MC packet spectrum is similar across cells (and matches HST flux balance),
then the formal-integral is the artifact — switch reference spectrum.
If MC packet spectrum also shows red excess, the model has true red excess
and we need physics work (cascade / cooling).

Also test energy conservation: total integrated flux should equal L_emitted.
"""
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return float(np.trapezoid(flu[sel], lam[sel]))

def baseline_norm_rms(mod_lam, mod_flu, hlam, hflu, fwhm=20000.0):
    selH = (hlam>=3000) & (hlam<=8000)
    hl, hf = hlam[selH], hflu[selH]
    dl = np.median(np.diff(hl)); mid = 0.5*(hl[0]+hl[-1])
    sig = (fwhm/C_KMS)*mid/2.355/dl
    sH = gaussian_filter1d(hf, sig, mode='nearest')
    selM = (mod_lam>=2900) & (mod_lam<=8100)
    ml, mf = mod_lam[selM], mod_flu[selM]
    dlM = np.median(np.diff(ml)); midM = 0.5*(ml[0]+ml[-1])
    sigM = (fwhm/C_KMS)*midM/2.355/dlM
    sM = gaussian_filter1d(mf, sigM, mode='nearest')
    common = (hl>=ml[0]) & (hl<=ml[-1])
    mb = np.interp(hl[common], ml, mf/sM)
    return float(np.sqrt(np.mean((hf[common]/sH[common] - mb)**2)))

h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
gH = band_int(hlam, hflu, 4500, 5800)

BANDS = [("UV", 2900, 4500), ("opt", 4500, 5800), ("yel", 5800, 6500),
         ("redI", 6500, 7400), ("redII", 7400, 8200), ("redIII", 8200, 9000)]

def report(name, lam, flu):
    g = band_int(lam, flu, 4500, 5800)
    if g > 0:
        flu_n = flu * (gH/g)
    else:
        flu_n = flu
    rms = baseline_norm_rms(lam, flu_n, hlam, hflu)
    tot = band_int(lam, flu_n, 2900, 9000)
    band_rats = []
    for nm, lo, hi in BANDS:
        m = band_int(lam, flu_n, lo, hi)
        h_ = band_int(hlam, hflu, lo, hi)
        band_rats.append(m/h_ if h_ > 0 else 0)
    return rms, tot, band_rats, g

H_tot = band_int(hlam, hflu, 2900, 9000)
print(f"{'cell':>22s}  {'RMS_bn':>7s}  {'tot/HST':>7s}  " +
      "  ".join(f"{nm:>6s}" for nm,_,_ in BANDS) + f"  {'green_raw':>11s}")
print(f"{'HST':>22s}  {'-':>7s}  {1.000:>7.3f}  " +
      "  ".join(f"{1.000:>6.3f}" for _ in BANDS) + f"  {gH:>11.3e}")

for cell in ["FI_base", "FI_cont", "FI_both"]:
    for kind in ["spectrum", "spectrum_formal"]:
        path = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_{cell}/lumina_{kind}.csv"
        df = pd.read_csv(path)
        col_lam = "wavelength_angstrom"
        col_flu = "flux"
        lam = df[col_lam].values
        flu = df[col_flu].values
        rms, tot, rats, graw = report(cell, lam, flu)
        kindshort = "MC " if kind == "spectrum" else "FI "
        print(f"{kindshort+cell:>22s}  {rms:>7.3f}  {tot/H_tot:>7.3f}  " +
              "  ".join(f"{r:>6.3f}" for r in rats) + f"  {graw:>11.3e}")

# Energy conservation check from log files
print("\n--- L_emitted from stdout (final iter) ---")
import re, subprocess
for cell in ["FI_base", "FI_cont", "FI_both"]:
    path = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_{cell}/stdout.log"
    txt = open(path).read()
    matches = re.findall(r'L_emitted\s*=\s*([\d.eE+-]+)', txt)
    if matches:
        print(f"  {cell}  L_emitted_final = {float(matches[-1]):.3e}")
