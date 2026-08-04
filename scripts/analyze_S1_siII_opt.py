#!/usr/bin/env python3
"""S1 (Task #221) analysis: Si II opt A_ul sweep vs Q1b r03 baseline.

Compares ref/f05/f03/f01 vs HST + TARDIS. Reports baseline-norm RMS @
FWHM=20k/40k, per-trough metrics on the 3 Si II features (4130/5972/6355),
plus residual cost decomposition for #220 ranked features.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST    = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458

JOB = int(sys.argv[1]) if len(sys.argv) > 1 else 156411

CELLS = [
    ("S1_ref", f"logs/ddc15S1_{JOB}_ddc15S1_S1_ref/lumina_spectrum_formal.csv"),
    ("S1_f05", f"logs/ddc15S1_{JOB}_ddc15S1_S1_f05/lumina_spectrum_formal.csv"),
    ("S1_f03", f"logs/ddc15S1_{JOB}_ddc15S1_S1_f03/lumina_spectrum_formal.csv"),
    ("S1_f01", f"logs/ddc15S1_{JOB}_ddc15S1_S1_f01/lumina_spectrum_formal.csv"),
]

FEATURES = [
    ("Ca II H&K",   3945.0, 3700, 3900),
    ("Si II 4130",  4130.0, 3990, 4130),
    ("Mg II 4481",  4481.0, 4280, 4430),
    ("Fe II m42",   4924.0, 4700, 4900),
    ("Fe II 5018",  5018.0, 4830, 4990),
    ("S II 5454",   5454.0, 5200, 5400),
    ("S II 5640",   5640.0, 5400, 5620),
    ("Si II 5972",  5972.0, 5750, 5950),
    ("Si II 6355",  6355.0, 6050, 6300),
    ("O I 7773",    7773.0, 7400, 7750),
    ("Ca II IR",    8542.0, 8000, 8420),
]

def load(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    flu = d.iloc[:,1].values
    return lam, flu

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def gauss_baseline(lam, flu, fwhm_kms, mask_lo=3000, mask_hi=8000):
    sel = (lam >= mask_lo) & (lam <= mask_hi)
    sub_lam = lam[sel]; sub_flu = flu[sel]
    if len(sub_lam) < 50: return None, None
    dlam = np.median(np.diff(sub_lam))
    lam_mid = 0.5*(mask_lo+mask_hi)
    sigma_aa = (fwhm_kms/C_KMS) * lam_mid / 2.355
    sigma_pix = sigma_aa / dlam
    base = gaussian_filter1d(sub_flu, sigma_pix, mode='nearest')
    return sub_lam, sub_flu / base

def rms_bn(mlam, mflu, rlam, rflu, fwhm):
    mlam_n, mflu_n = gauss_baseline(mlam, mflu, fwhm)
    rlam_n, rflu_n = gauss_baseline(rlam, rflu, fwhm)
    if mlam_n is None or rlam_n is None: return np.nan
    fm = interp1d(mlam_n, mflu_n, kind='linear', bounds_error=False, fill_value=np.nan)
    return float(np.sqrt(np.nanmean((fm(rlam_n) - rflu_n)**2)))

def trough_metrics(lam, flu, lab, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 5: return (np.nan,)*3
    sl, sf = lam[m], flu[m]
    imin = int(np.argmin(sf))
    lam_min = float(sl[imin]); f_min = float(sf[imin])
    n = len(sl); edge = max(3, n//20)
    f_cont = max(float(np.median(sf[:edge])), float(np.median(sf[-edge:])))
    depth = 1.0 - f_min/f_cont if f_cont > 0 else np.nan
    v_kms = (lab - lam_min)/lab * C_KMS
    f_half = 0.5*(f_min + f_cont)
    j = imin
    while j > 0 and sf[j] < f_half: j -= 1
    k = imin
    while k < n-1 and sf[k] < f_half: k += 1
    fwhm_kms = (sl[min(k,n-1)] - sl[max(j,0)]) / lab * C_KMS
    return v_kms, depth, fwhm_kms

hlam, hflu = load(HST)
tlam, tflu = load(TARDIS)
gH = band_int(hlam, hflu, 4500, 5800)

data = {}
for name, rel in CELLS:
    p = ROOT/rel
    if not p.exists():
        print(f"MISSING {name}: {p}")
        continue
    lam, flu = load(p)
    g = band_int(lam, flu, 4500, 5800)
    data[name] = (lam, flu*(gH/g))

print("="*100)
print(f"=== S1 (#221) Si II optical band A_ul sweep, job {JOB} ===")
print("    SCALE6 Si II λ∈[4000,7000]Å on Q1b r03 stack")
print("="*100)
print(f"{'cell':<10s}  {'HST_20k':>9s} {'HST_40k':>9s} {'TAR_20k':>9s} {'TAR_40k':>9s}")
print("-"*100)
rms = {}
for name, (lam, fH) in data.items():
    r_h20 = rms_bn(lam, fH, hlam, hflu, 20000)
    r_h40 = rms_bn(lam, fH, hlam, hflu, 40000)
    r_t20 = rms_bn(lam, fH, tlam, tflu, 20000)
    r_t40 = rms_bn(lam, fH, tlam, tflu, 40000)
    rms[name] = (r_h20, r_h40, r_t20, r_t40)
    print(f"{name:<10s}  {r_h20:>9.4f} {r_h40:>9.4f} {r_t20:>9.4f} {r_t40:>9.4f}")
print()
print(f"  Q1b r03 floor (job 156281):  HST_20k = 0.2375")
print(f"  Task #172 target:            HST_20k ≤ 0.2000")

# Per-trough Si II depth tracking
print()
print("="*100)
print("=== Si II trough depths (depth, v_kms, FWHM_kms) per cell ===")
print(f"{'feature':<13}{'lab':>5} | {'HST_d':>6} | "
      f"{'ref_d':>6}{'05_d':>6}{'03_d':>6}{'01_d':>6} | {'ref_v':>7}{'01_v':>7}")
print("-"*100)
for name, lab, lo, hi in FEATURES:
    H = trough_metrics(hlam, hflu, lab, lo, hi)
    row = f"{name:<13}{lab:>5.0f} | {H[1]:>6.2f} |"
    vrow = {}
    for cell, (lam, fH) in data.items():
        L = trough_metrics(lam, fH, lab, lo, hi)
        row += f" {L[1]:>5.2f}"
        vrow[cell] = L[0]
    row += f" | {vrow.get('S1_ref',np.nan):>7.0f}{vrow.get('S1_f01',np.nan):>7.0f}"
    print(row)

# Band ratios
print()
print("="*100)
print("=== Band ratios (model/HST, green-normalized) ===")
BANDS = [(2300,3500,"UV-mid"), (3500,4500,"blue"), (4500,5800,"green"),
         (5800,6800,"Si-red"), (6800,8000,"OI/cont"), (8000,9500,"Ca-IR")]
hdr = f"{'cell':<10s}"
for _,_,lab in BANDS: hdr += f"{lab:>10s}"
print(hdr)
for name, (lam, fH) in data.items():
    row = f"{name:<10s}"
    for lo, hi, _ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, fH, lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)

print()
print("="*100)
print("=== DECISION ===")
if "S1_ref" in rms:
    best_cell = min(rms, key=lambda k: rms[k][0])
    print(f"  Best HST_20k cell: {best_cell} = {rms[best_cell][0]:.4f}")
    print(f"  Δ(best − ref): {rms[best_cell][0] - rms['S1_ref'][0]:+.4f}")
    if rms[best_cell][0] <= 0.20:
        print(f"  🎯 Task #172 ACHIEVED at {best_cell}")
    elif rms[best_cell][0] < 0.2375:
        print(f"  ✓ Beats Q1b r03 floor 0.2375 by {0.2375-rms[best_cell][0]:.4f}")
    else:
        print(f"  ✗ Does not beat floor. Floor still 0.2375.")
