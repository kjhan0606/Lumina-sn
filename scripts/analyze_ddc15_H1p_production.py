#!/usr/bin/env python3
"""H1p: production-fidelity (800K × 12) validation of the H1/H1b/H2 champion triad.
Computes red/HST, band-log10-rms, and RMS_bn (baseline-norm) for 4 cells and
compares to 200K results from H1 156014 / H1b 156018 / H2 156031."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "156038"
C2_REF = ROOT/"logs/ddc15C2_155756_ddc15C2_xFeO0.05/lumina_spectrum_formal.csv"

CELLS = [
    ("h1b_eps0p4",        "H1b ε=0.4 (joint)",     "200K ref: cost 0.216 RMS 0.245"),
    ("h1_eps0p9",         "H1  ε=0.9 (shape)",     "200K ref: cost 0.275 RMS 0.202"),
    ("h2_eps0p9_redonly", "H2  ε=0.9 RED_ONLY",    "200K ref: cost 0.215 RMS 0.285"),
    ("control_eps0p0",    "control ε=0.0",         "200K ref: cost 0.291 RMS 0.291"),
]
C_KMS = 299792.458
SUB_BANDS = [(3000,5500,"UV+blue"),(5500,5800,"5500-5800"),(5800,6800,"Si red"),
             (6800,8000,"OI/cont"),(8000,9500,"Ca IR")]
RED_TOT = (5500, 9500)
UV_TOT  = (3300, 3700)

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
uvH = band_int(hlam, hflu, *UV_TOT)
gH  = band_int(hlam, hflu, 4500, 5800)

print(f"\n=== H1p production validation (job {JOB}, 800K × 12 iter) ===")
print(f"  HST red-tot [5500,9500]Å = {rtH:.3e}")
print(f"  HST UVtgt   [3300,3700]Å = {uvH:.3e}\n")

hdr = f"  {'cell':<22}  {'UV/HST':<7}  {'red/HST':<8}  {'cost':<7}  {'band-log10':<11}  {'RMS_bn':<8}"
print(hdr); print("-"*len(hdr))

rows = []
for label, desc, ref200 in CELLS:
    p = ROOT/f"logs/ddc15H1p_{JOB}_ddc15H1p_{label}/lumina_spectrum_formal.csv"
    d = scale(p, hlam, hflu)
    if d is None:
        print(f"  {desc:<22}  MISSING")
        rows.append((label, None)); continue
    rt = band_int(*d, *RED_TOT) / rtH
    uv = band_int(*d, *UV_TOT)  / uvH
    cost = np.sqrt(((uv-1)**2 + (rt-1)**2)/2)
    blrms = band_log10_rms(*d, hlam, hflu, SUB_BANDS)
    try:    bnrms = rms_baseline_norm(*d, hlam, hflu)
    except: bnrms = float('nan')
    rows.append((label, dict(uv=uv, rt=rt, cost=cost, blrms=blrms, bnrms=bnrms)))
    print(f"  {desc:<22}  {uv:6.3f}   {rt:7.3f}  {cost:6.3f}   {blrms:8.4f}     {bnrms:6.4f}")
    print(f"    {'└─ 200K ref:':<22}  {ref200}")

# Verdict
print()
named = {l:r for l,r in rows if r is not None}
if "h1_eps0p9" in named:
    r = named["h1_eps0p9"]
    target_hit = r["bnrms"] <= 0.20
    miss = r["bnrms"] - 0.20
    verdict = "HIT" if target_hit else f"MISS by {miss:+.4f}"
    print(f"  --- Task #172 verdict (target RMS_bn ≤ 0.20) ---")
    print(f"  H1 ε=0.9 production RMS_bn = {r['bnrms']:.4f}  →  {verdict}")
if "h1b_eps0p4" in named and "h2_eps0p9_redonly" in named:
    print(f"  H1b ε=0.4 RMS_bn = {named['h1b_eps0p4']['bnrms']:.4f}  cost = {named['h1b_eps0p4']['cost']:.3f}")
    print(f"  H2  ε=0.9 RMS_bn = {named['h2_eps0p9_redonly']['bnrms']:.4f}  cost = {named['h2_eps0p9_redonly']['cost']:.3f}")

# Per-sub-band
print(f"\n  --- Per-sub-band integrated flux ratio (cell / HST) ---")
hdr2 = f"  {'band':<14}  " + "  ".join(f"{l[:8]:<8}" for l,_,_ in CELLS)
print(hdr2)
for lo, hi, name in SUB_BANDS:
    iH = band_int(hlam, hflu, lo, hi)
    row = f"  {name:<14}  "
    for label,_,_ in CELLS:
        p = ROOT/f"logs/ddc15H1p_{JOB}_ddc15H1p_{label}/lumina_spectrum_formal.csv"
        d = scale(p, hlam, hflu)
        if d is None: row += " ---      "
        else: row += f"{band_int(*d, lo, hi)/iH:<8.3f}  "
    print(row)
print()
