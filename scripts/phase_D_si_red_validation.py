#!/usr/bin/env python3
"""Phase D (#133): joint validation — Si II 6355 trough depth + position + red excess.
Pass criteria:
   (A) Si II λ_min  ≈ 6154 Å (HST), tolerance ±50 Å (±2400 km/s)
   (B) Si II depth  (1 - F_min/F_cont) ≥ 0.40  (HST B-max ~0.75; need at least half)
   (C) red/HST < 1.30   (red excess no worse than +30%)
   (D) RMS_bn ≤ 0.30   (production-fidelity working floor; ≤0.20 = TARDIS-class)

Operates on whatever CSV path is passed as argv (one or many). Single-cell verdict
plus comparative table when multiple cells supplied.

Usage:
   python3 phase_D_si_red_validation.py path1 [path2 ...]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"

C_KMS = 299792.458
SI_REST  = 6355.0
RED_TOT  = (5500, 9500)

def load(p):
    p = Path(p)
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

def trough_metrics(lam, flu, lo=5800, hi=6300, smooth_pts=15):
    """Si II 6355 trough: (λ_min, v_blue, depth)."""
    m = (lam>=lo)&(lam<=hi)
    lam_w = lam[m]; flu_w = flu[m]
    flu_s = gaussian_filter1d(flu_w, sigma=smooth_pts)
    i_min = np.argmin(flu_s)
    lam_min = lam_w[i_min]
    F_min = flu_s[i_min]
    # local continuum: average of edges (5500-5700 and 6300-6500)
    cl = (lam>=5500)&(lam<=5700); cr = (lam>=6300)&(lam<=6500)
    F_cont = 0.5 * (np.median(flu[cl]) + np.median(flu[cr]))
    depth = 1.0 - F_min/F_cont if F_cont > 0 else float('nan')
    v_blue = C_KMS * (SI_REST - lam_min)/SI_REST
    return lam_min, v_blue, depth, F_cont

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

PASS = {
    "λ_min vs HST 6154±50Å": ("lam_min", lambda v: abs(v - 6154) <= 50),
    "Si II depth ≥ 0.40":    ("depth",   lambda v: v >= 0.40),
    "red/HST ≤ 1.30":        ("red",     lambda v: v <= 1.30),
    "RMS_bn ≤ 0.30":         ("rms_bn",  lambda v: v <= 0.30),
}

hlam, hflu = load(HST)
rtH = band_int(hlam, hflu, *RED_TOT)
hL, hV, hD, hC = trough_metrics(hlam, hflu)

print(f"\n=== Phase D (#133) joint Si II + red excess validation ===")
print(f"  HST B-max: Si II λ_min={hL:.1f}Å, v={hV:.0f} km/s, depth={hD:.3f}, "
      f"red-tot={rtH:.3e}\n")

paths = sys.argv[1:]
if not paths:
    print("  no input paths — pass one or more lumina_spectrum_formal.csv")
    sys.exit(1)

print(f"  {'cell':<48}  {'λ_min':<8}  {'v(km/s)':<8}  {'depth':<7}  {'red/HST':<8}  {'RMS_bn':<7}  pass")
print("-"*120)

rows = []
for path in paths:
    p = Path(path)
    d = scale(p, hlam, hflu)
    if d is None:
        print(f"  {p.parent.name:<48}  MISSING"); continue
    lam, flu = d
    lm, vb, dp, fc = trough_metrics(lam, flu)
    rt = band_int(lam, flu, *RED_TOT) / rtH
    bn = rms_baseline_norm(lam, flu, hlam, hflu)
    metrics = dict(lam_min=lm, v_blue=vb, depth=dp, red=rt, rms_bn=bn)
    passed = [name for name, (key, fn) in PASS.items() if fn(metrics[key])]
    rows.append((p.parent.name, metrics, passed))
    tag = ("A" if PASS["λ_min vs HST 6154±50Å"][1](lm) else ".") + \
          ("B" if PASS["Si II depth ≥ 0.40"][1](dp) else ".") + \
          ("C" if PASS["red/HST ≤ 1.30"][1](rt) else ".") + \
          ("D" if PASS["RMS_bn ≤ 0.30"][1](bn) else ".")
    print(f"  {p.parent.name:<48}  {lm:6.1f}    {vb:6.0f}    {dp:5.3f}    {rt:6.3f}    {bn:5.3f}    {tag}  ({len(passed)}/4)")

print()
print("  pass flags:")
print("    A = Si II λ_min within ±50 Å of HST 6154.2")
print("    B = Si II depth ≥ 0.40")
print("    C = red/HST ≤ 1.30")
print("    D = RMS_bn ≤ 0.30  (≤0.20 = TARDIS-class)")
print()

# Verdict
if rows:
    full_pass = [r for r in rows if len(r[2]) == 4]
    print(f"  --- Phase D verdict ---")
    if full_pass:
        print(f"  ✓ {len(full_pass)}/{len(rows)} cells pass ALL 4 criteria.")
        for name, m, _ in full_pass:
            print(f"    • {name}  (RMS_bn={m['rms_bn']:.3f}, v={m['v_blue']:.0f} km/s)")
    else:
        best = max(rows, key=lambda r: len(r[2]))
        print(f"  ✗ no cell passes all 4. Best: {best[0]} ({len(best[2])}/4)")
        print(f"    passed: {', '.join(best[2]) if best[2] else '(none)'}")
print()
