#!/usr/bin/env python3
"""I1: Ni II-targeted UV internal-down suppression analysis (job 156044).
Compares 4 cells {factor=1.0, 0.7, 0.5, 0.3} with same metrics as H1p production
(UV/HST, red/HST, cost, band-log10-rms, RMS_bn)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "156044"

CELLS = [
    ("NiII_f1p0_ctrl", "factor=1.0 (control)"),
    ("NiII_f0p7",      "factor=0.7 (mild)"),
    ("NiII_f0p5",      "factor=0.5 (moderate)"),
    ("NiII_f0p3",      "factor=0.3 (aggressive)"),
]
C_KMS = 299792.458
SUB_BANDS = [(3000,5500,"UV+blue"),(5500,5800,"5500-5800"),(5800,6800,"Si red"),
             (6800,8000,"OI/cont"),(8000,9500,"Ca IR")]
RED_TOT = (5500, 9500)
UV_TOT  = (3300, 3700)

def load(p):
    if not p.exists(): return None, None
    d = pd.read_csv(p); return d.iloc[:,0].values, d.iloc[:,1].values

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def scale(p, hlam, hflu):
    lam, flu = load(p)
    if lam is None: return None
    gH = band_int(hlam, hflu, 4500, 5800)
    return lam, flu * (gH / band_int(lam, flu, 4500, 5800))

def band_log10_rms(lam_m, flu_m, lam_h, flu_h, bands):
    diffs = []
    for lo, hi, name in bands:
        im = band_int(lam_m, flu_m, lo, hi)
        ih = band_int(lam_h, flu_h, lo, hi)
        if im > 0 and ih > 0:
            diffs.append(np.log10(im/ih))
    return np.sqrt(np.mean(np.array(diffs)**2))

def rms_baseline_norm(lam_m, flu_m, lam_h, flu_h, lo=3000, hi=8000, fwhm_kms=40000):
    log_lam_m = np.log(lam_m); dlog_m = np.median(np.diff(log_lam_m))
    sig_m = (fwhm_kms/C_KMS)/(2.355*dlog_m)
    cont_m = gaussian_filter1d(flu_m, sigma=sig_m)
    log_lam_h = np.log(lam_h); dlog_h = np.median(np.diff(log_lam_h))
    sig_h = (fwhm_kms/C_KMS)/(2.355*dlog_h)
    cont_h = gaussian_filter1d(flu_h, sigma=sig_h)
    m = (lam_m>=lo)&(lam_m<=hi)
    lam_c = lam_m[m]
    res_m = (flu_m - cont_m)[m]
    res_h = np.interp(lam_c, lam_h, flu_h - cont_h)
    norm = np.interp(lam_c, lam_h, cont_h)
    norm = np.maximum(norm, 0.01*np.max(norm))
    return np.sqrt(np.mean(((res_m - res_h)/norm)**2))

hlam, hflu = load(HST)
rtH = band_int(hlam, hflu, *RED_TOT)
uvH = band_int(hlam, hflu, *UV_TOT)

print(f"\n=== K1 Ni II A_ul scale sweep (job {JOB}, 800K × 12 iter) ===")
print(f"  H3 motivation: Ni II = 68% of UV→red cascade attribution\n")
print(f"  {'cell':<26}  {'UV/HST':<7}  {'red/HST':<8}  {'cost':<7}  {'band-log10':<11}  {'RMS_bn':<8}")
print("-"*80)

rows = []
for label, desc in CELLS:
    p = ROOT/f"logs/ddc15K1_{JOB}_ddc15K1_{label}/lumina_spectrum_formal.csv"
    d = scale(p, hlam, hflu)
    if d is None:
        print(f"  {desc:<26}  MISSING ({p.parent.name})")
        rows.append((label, None)); continue
    rt = band_int(*d, *RED_TOT) / rtH
    uv = band_int(*d, *UV_TOT)  / uvH
    cost = np.sqrt(((uv-1)**2 + (rt-1)**2)/2)
    blrms = band_log10_rms(*d, hlam, hflu, SUB_BANDS)
    try:    bnrms = rms_baseline_norm(*d, hlam, hflu)
    except: bnrms = float('nan')
    rows.append((label, dict(uv=uv, rt=rt, cost=cost, blrms=blrms, bnrms=bnrms)))
    print(f"  {desc:<26}  {uv:6.3f}   {rt:7.3f}  {cost:6.3f}   {blrms:8.4f}     {bnrms:6.4f}")

# Best-cell selection (lowest RMS_bn)
print()
valid = [(l,r) for l,r in rows if r is not None]
if valid:
    best_bn = min(valid, key=lambda x: x[1]["bnrms"])
    best_co = min(valid, key=lambda x: x[1]["cost"])
    bn_val  = best_bn[1]["bnrms"]
    target  = bn_val <= 0.20
    verdict = "HIT ≤0.20" if target else f"MISS (+{bn_val-0.20:.4f})"
    print(f"  --- Best cells ---")
    print(f"  Lowest RMS_bn: {best_bn[0]:<22} = {bn_val:.4f}  → {verdict}")
    print(f"  Lowest cost:    {best_co[0]:<22} = {best_co[1]['cost']:.4f}")

    # vs H1p control (RMS_bn 0.392, cost 0.435)
    if "NiII_f1p0_ctrl" in dict(rows):
        ctrl = dict(rows)["NiII_f1p0_ctrl"]
        print(f"\n  --- vs control (factor=1.0, same run) ---")
        for label, r in valid:
            if label == "NiII_f1p0_ctrl": continue
            dbn = r["bnrms"] - ctrl["bnrms"]
            dco = r["cost"]  - ctrl["cost"]
            print(f"  {label:<22}  ΔRMS_bn = {dbn:+.4f}   Δcost = {dco:+.4f}")
print()

# Per-sub-band breakdown
print(f"  --- Per-sub-band flux ratio (cell / HST) ---")
hdr2 = f"  {'band':<14}  " + "  ".join(f"{l[:11]:<11}" for l,_ in CELLS)
print(hdr2)
for lo, hi, name in SUB_BANDS:
    iH = band_int(hlam, hflu, lo, hi)
    row = f"  {name:<14}  "
    for label,_ in CELLS:
        p = ROOT/f"logs/ddc15K1_{JOB}_ddc15K1_{label}/lumina_spectrum_formal.csv"
        d = scale(p, hlam, hflu)
        row += (f"{band_int(*d, lo, hi)/iH:<11.3f}" if d is not None else f"{'---':<11}") + "  "
    print(row)
print()
