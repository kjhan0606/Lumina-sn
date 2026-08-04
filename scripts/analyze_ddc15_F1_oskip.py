#!/usr/bin/env python3
"""F1: Compare base/Z8/Z6_8 oskip variants vs HST + C2 baseline (155756).
Outputs red-tot ratio, band-log10-RMS, baseline-norm RMS for each variant.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "156005"
C2_REF_JOB = "155756"
C2_REF_PATH = ROOT/f"logs/ddc15C2_{C2_REF_JOB}_ddc15C2_xFeO0.05/lumina_spectrum_formal.csv"

LABELS = ["base", "Z8", "Z6_8"]
SKIPS  = ["", "8", "6,8"]
DESCS  = ["no mask (reprod)", "Z=8 (O zero-out)", "Z=6,8 (C+O zero-out)"]
C_KMS = 299792.458

# Band definitions
SUB_BANDS = [(3000,5500,"UV+blue"),(5500,5800,"5500-5800"),(5800,6800,"Si red"),
             (6800,8000,"OI/cont"),(8000,9500,"Ca IR")]
RED_TOT_BAND = (5500, 9500)

def load(p):
    if not p.exists(): return None, None
    d = pd.read_csv(p)
    lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def scale_to_hst(p, hlam, hflu, gH):
    lam, flu = load(p)
    if lam is None: return None
    return lam, flu * (gH / band_int(lam, flu, 4500, 5800))

def baseline_norm(lam, flu, lo=3000, hi=8000, fwhm_kms=40000):
    """Pseudo-continuum via Gaussian FWHM=40k km/s in log-lambda."""
    m = (lam>=lo)&(lam<=hi)
    lam_c, flu_c = lam[m], flu[m]
    log_lam = np.log(lam_c)
    dlog = np.median(np.diff(log_lam))
    fwhm_log = (fwhm_kms/C_KMS)
    sigma = fwhm_log/(2.355*dlog)
    cont = gaussian_filter1d(flu_c, sigma=sigma)
    return lam_c, flu_c - cont, cont

def rms_baseline_norm(lam_m, flu_m, lam_h, flu_h, lo=3000, hi=8000):
    """RMS of (model - HST) after baseline normalization, both on common λ grid."""
    lam_c_m, res_m, cont_m = baseline_norm(lam_m, flu_m, lo, hi)
    cont_h_interp = np.interp(lam_c_m, lam_h, gaussian_filter1d_interp(lam_h, flu_h, lam_c_m))
    flu_h_interp = np.interp(lam_c_m, lam_h, flu_h)
    res_h = flu_h_interp - cont_h_interp
    # normalize by HST continuum
    norm = np.maximum(cont_h_interp, 0.01*np.max(cont_h_interp))
    diff = (res_m - res_h) / norm
    return np.sqrt(np.mean(diff**2))

def gaussian_filter1d_interp(lam, flu, target_lam, fwhm_kms=40000):
    """Compute Gaussian continuum and return it on target_lam grid."""
    log_lam = np.log(lam)
    dlog = np.median(np.diff(log_lam))
    sigma = (fwhm_kms/C_KMS)/(2.355*dlog)
    cont = gaussian_filter1d(flu, sigma=sigma)
    return np.interp(target_lam, lam, cont)

def band_log10_rms(lam_m, flu_m, lam_h, flu_h, bands):
    """RMS of log10(model/HST) in band-integrated flux."""
    diffs = []
    for lo, hi, name in bands:
        im = band_int(lam_m, flu_m, lo, hi)
        ih = band_int(lam_h, flu_h, lo, hi)
        if im > 0 and ih > 0:
            diffs.append(np.log10(im/ih))
    return np.sqrt(np.mean(np.array(diffs)**2))

# Load HST
hlam, hflu = load(HST)
gH = band_int(hlam, hflu, 4500, 5800)

# Load C2 baseline (reference)
c2 = scale_to_hst(C2_REF_PATH, hlam, hflu, gH)

# Load F1 variants
print(f"\n=== F1 oskip diagnostic (job {JOB}) ===")
print(f"  C2 baseline = {C2_REF_JOB} (X_Fe_outer=0.05, no mask)")
print(f"  Reference HST integrated [4500,5800] = {gH:.3e}\n")

print(f"  HST red-tot [5500,9500]Å integrated  = {band_int(hlam,hflu,*RED_TOT_BAND[:2]):.3e}")
print()

header = f"  {'label':<6} {'desc':<22} {'red-tot':<10} {'red/HST':<8} {'band-log10':<12} {'RMS_bn':<8}"
print(header)
print("-"*len(header))

# C2 baseline
if c2 is not None:
    lam_c2, flu_c2 = c2
    rt = band_int(lam_c2, flu_c2, *RED_TOT_BAND[:2])
    rtH = band_int(hlam, hflu, *RED_TOT_BAND[:2])
    blrms = band_log10_rms(lam_c2, flu_c2, hlam, hflu, SUB_BANDS)
    try:
        bnrms = rms_baseline_norm(lam_c2, flu_c2, hlam, hflu)
    except Exception as e:
        bnrms = float('nan')
    print(f"  {'C2-ref':<6} {'X_Fe_out=0.05':<22} {rt:.3e} {rt/rtH:6.3f}   {blrms:8.4f}     {bnrms:6.4f}")

# F1 variants
for lab, skip, desc in zip(LABELS, SKIPS, DESCS):
    p = ROOT/f"logs/ddc15F1_{JOB}_{lab}/lumina_spectrum_formal.csv"
    d = scale_to_hst(p, hlam, hflu, gH)
    if d is None:
        print(f"  {lab:<6} {desc:<22} MISSING ({p.name})")
        continue
    lam, flu = d
    rt = band_int(lam, flu, *RED_TOT_BAND[:2])
    rtH = band_int(hlam, hflu, *RED_TOT_BAND[:2])
    blrms = band_log10_rms(lam, flu, hlam, hflu, SUB_BANDS)
    try:
        bnrms = rms_baseline_norm(lam, flu, hlam, hflu)
    except Exception as e:
        bnrms = float('nan')
    print(f"  {lab:<6} {desc:<22} {rt:.3e} {rt/rtH:6.3f}   {blrms:8.4f}     {bnrms:6.4f}")

# Per-sub-band breakdown vs HST + vs C2
print(f"\n  --- Per-sub-band integrated flux ratio (variant / HST) ---")
print(f"  {'band':<14}  {'C2-ref':<8}  " + "  ".join(f"{l:<8}" for l in LABELS))
for lo, hi, name in SUB_BANDS:
    iH = band_int(hlam, hflu, lo, hi)
    row = f"  {name:<14}  "
    if c2 is not None:
        row += f"{band_int(*c2, lo, hi)/iH:<8.3f}  "
    else:
        row += " ---     "
    for lab in LABELS:
        p = ROOT/f"logs/ddc15F1_{JOB}_{lab}/lumina_spectrum_formal.csv"
        d = scale_to_hst(p, hlam, hflu, gH)
        if d is None:
            row += " ---     "
        else:
            row += f"{band_int(*d, lo, hi)/iH:<8.3f}  "
    print(row)

print()
