#!/usr/bin/env python3
"""H1: LUMINA_EPS_UV sweep diagnostic — does UV macro-atom cascade bypass
reduce red excess? Compares 4 variants {0.0, 0.3, 0.6, 0.9} vs HST + C2 baseline.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "156009"
C2_REF = ROOT/"logs/ddc15C2_155756_ddc15C2_xFeO0.05/lumina_spectrum_formal.csv"

EPS_UVS = ["0.0", "0.3", "0.6", "0.9"]
DESCS   = ["EPS_UV=0.0 (control)", "EPS_UV=0.3", "EPS_UV=0.6", "EPS_UV=0.9"]
C_KMS = 299792.458

SUB_BANDS = [(3000,5500,"UV+blue"),(5500,5800,"5500-5800"),(5800,6800,"Si red"),
             (6800,8000,"OI/cont"),(8000,9500,"Ca IR")]
RED_TOT = (5500, 9500)

def load(p):
    if not p.exists(): return None, None
    d = pd.read_csv(p)
    return d.iloc[:,0].values, d.iloc[:,1].values

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
gH  = band_int(hlam, hflu, 4500, 5800)

print(f"\n=== H1 EPS_UV sweep (job {JOB}) ===")
print(f"  Mechanism: prob d_eps_uv replaces UV-entry macro-atom cascade with Planck(T_rad)")
print(f"  C2 baseline (EPS_UV=0, no short-circuit): {C2_REF.parent.name}")
print(f"  HST red-tot [5500,9500]Å = {rtH:.3e}\n")

hdr = f"  {'variant':<24}  {'red-tot':<10}  {'red/HST':<8}  {'Δred%':<8}  {'band-log10':<11}  {'RMS_bn':<8}"
print(hdr); print("-"*len(hdr))

# C2 ref
c2 = scale(C2_REF, hlam, hflu)
base_red = None
if c2 is not None:
    rt = band_int(*c2, *RED_TOT)
    blrms = band_log10_rms(*c2, hlam, hflu, SUB_BANDS)
    bnrms = rms_baseline_norm(*c2, hlam, hflu)
    base_red = rt / rtH
    print(f"  {'C2-ref (155756)':<24}  {rt:.3e}  {rt/rtH:7.3f}   {'0.0':<8}  {blrms:8.4f}     {bnrms:6.4f}")

# H1 variants
red_ratios = []
ctrl_red = None
for eps, desc in zip(EPS_UVS, DESCS):
    p = ROOT/f"logs/ddc15H1_{JOB}_ddc15H1_epsUV{eps}/lumina_spectrum_formal.csv"
    d = scale(p, hlam, hflu)
    if d is None:
        print(f"  {desc:<24}  MISSING ({p.name})")
        red_ratios.append(None); continue
    rt = band_int(*d, *RED_TOT)
    blrms = band_log10_rms(*d, hlam, hflu, SUB_BANDS)
    try: bnrms = rms_baseline_norm(*d, hlam, hflu)
    except: bnrms = float('nan')
    r = rt/rtH
    if eps == "0.0": ctrl_red = r
    delta = 100*(r - (ctrl_red if ctrl_red is not None else r))
    red_ratios.append(r)
    print(f"  {desc:<24}  {rt:.3e}  {r:7.3f}   {delta:+7.1f}  {blrms:8.4f}     {bnrms:6.4f}")

# Per-sub-band
print(f"\n  --- Per-sub-band integrated flux ratio (variant / HST) ---")
hdr2 = f"  {'band':<14}  " + "  ".join(f"{x:<7}" for x in EPS_UVS)
print(hdr2)
for lo, hi, name in SUB_BANDS:
    iH = band_int(hlam, hflu, lo, hi)
    row = f"  {name:<14}  "
    for eps in EPS_UVS:
        p = ROOT/f"logs/ddc15H1_{JOB}_ddc15H1_epsUV{eps}/lumina_spectrum_formal.csv"
        d = scale(p, hlam, hflu)
        if d is None: row += " ---     "
        else: row += f"{band_int(*d, lo, hi)/iH:<7.3f}  "
    print(row)

# Verdict
print()
valid = [r for r in red_ratios if r is not None]
if len(valid) >= 2:
    print(f"  Verdict (red/HST):")
    mono_down = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
    mono_up   = all(valid[i] <= valid[i+1] for i in range(len(valid)-1))
    print(f"    monotone: {'DOWN (cascade IS source)' if mono_down else ('UP (wrong direction)' if mono_up else 'NO')}")
    print(f"    range: {min(valid):.3f} — {max(valid):.3f}  (Δ={max(valid)-min(valid):+.3f})")
    if max(valid) - min(valid) < 0.03:
        print(f"    → cascade-bypass NULL (<3% effect). Red excess is NOT UV macro-atom cascade.")
    elif mono_down and (valid[0] - valid[-1]) > 0.05:
        print(f"    → cascade IS the lever (monotone↓ by Δ≥5%). H1 confirmed.")
    elif mono_up:
        print(f"    → cascade bypass makes red WORSE — direction-flipping like F1. Mechanism is opposite.")
    else:
        print(f"    → weak/non-monotone. Inspect per-band, may saturate.")
print()
