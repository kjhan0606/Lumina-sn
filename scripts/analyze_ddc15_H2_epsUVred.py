#!/usr/bin/env python3
"""H2 analysis: wavelength-conditional EPS_UV (red-only post-cascade gate).
Compares H2 (EPS_UV_RED_ONLY=1) to H1/H1b reference points at the same EPS_UV.

Hypothesis: red-only mode preserves UV+blue while still killing red excess.
If true, the H2 trade-off curve should sit below H1's (lower joint cost)
at EPS_UV=0.4..0.9 — UV should NOT collapse the way it did in H1.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB    = sys.argv[1] if len(sys.argv) > 1 else "PLACEHOLDER"
H1_JOB = sys.argv[2] if len(sys.argv) > 2 else "156014"

H2_EPS = ["0.0", "0.4", "0.7", "0.9"]
# H1 reference: 0.0/0.3/0.6/0.9 — pair (0.0, 0.9) for direct red-only vs full comparison
H1_EPS = ["0.0", "0.3", "0.6", "0.9"]

C_KMS = 299792.458
SUB_BANDS = [(3000,5500,"UV+blue"),(5500,5800,"5500-5800"),(5800,6800,"Si red"),
             (6800,8000,"OI/cont"),(8000,9500,"Ca IR")]
RED_TOT = (5500, 9500)
UV_TOT  = (3000, 5500)

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

def metrics_for(path):
    d = scale(path, hlam, hflu)
    if d is None: return None
    lam, flu = d
    rt = band_int(lam, flu, *RED_TOT) / rtH
    uv = band_int(lam, flu, *UV_TOT) / uvH
    bl = band_log10_rms(lam, flu, hlam, hflu, SUB_BANDS)
    try: bn = rms_baseline_norm(lam, flu, hlam, hflu)
    except: bn = float('nan')
    cost = np.sqrt(((uv-1)**2 + (rt-1)**2)/2)
    sub = {name: band_int(lam, flu, lo, hi)/band_int(hlam, hflu, lo, hi)
           for lo, hi, name in SUB_BANDS}
    return uv, rt, cost, bl, bn, sub

print(f"\n=== H2 epsUV red-only sweep (job {JOB}, H1 ref {H1_JOB}) ===")
print(f"  Mechanism: cascade runs normally; if UV-entry AND exit in [5500,10000)A,")
print(f"             prob eps_uv replace exit with Planck(T_rad).")
print(f"  HST red-tot [5500,9500]A = {rtH:.3e}")
print(f"  HST UV+blue [3000,5500]A = {uvH:.3e}\n")

print(f"  --- H2 (red-only) sweep ---")
hdr = f"  {'EPS_UV':<7} {'UV/HST':<7} {'red/HST':<8} {'cost':<7} {'bl-RMS':<8} {'RMS_bn':<7}"
print(hdr); print("-"*len(hdr))
h2_rows = []
for eps in H2_EPS:
    p = ROOT/f"logs/ddc15H2_{JOB}_ddc15H2_epsUV{eps}_redonly/lumina_spectrum_formal.csv"
    r = metrics_for(p)
    if r is None:
        print(f"  {eps:<7}  MISSING ({p.name})")
        continue
    uv, rt, cost, bl, bn, sub = r
    h2_rows.append((float(eps), uv, rt, cost, bl, bn, sub))
    print(f"  {eps:<7} {uv:<7.3f} {rt:<8.3f} {cost:<7.3f} {bl:<8.4f} {bn:<7.4f}")

print(f"\n  --- H1 reference (full pre-cascade bypass) ---")
print(hdr); print("-"*len(hdr))
h1_rows = []
for eps in H1_EPS:
    p = ROOT/f"logs/ddc15H1_{H1_JOB}_ddc15H1_epsUV{eps}/lumina_spectrum_formal.csv"
    r = metrics_for(p)
    if r is None: continue
    uv, rt, cost, bl, bn, sub = r
    h1_rows.append((float(eps), uv, rt, cost, bl, bn, sub))
    print(f"  {eps:<7} {uv:<7.3f} {rt:<8.3f} {cost:<7.3f} {bl:<8.4f} {bn:<7.4f}")

# Direct head-to-head at matched eps
print(f"\n  --- Head-to-head (H2 red-only vs H1 full, matched EPS_UV) ---")
print(f"  {'eps':<5} {'H2_UV':<7} {'H1_UV':<7} {'ΔUV':<7}   {'H2_red':<7} {'H1_red':<7} {'Δred':<7}   {'H2_cost':<7} {'H1_cost':<7}")
for eps2, uv2, rt2, c2, *_ in h2_rows:
    h1_match = next((r for r in h1_rows if abs(r[0]-eps2) < 0.05), None)
    if h1_match is None: continue
    eps1, uv1, rt1, c1, *_ = h1_match
    print(f"  {eps2:<5.2f} {uv2:<7.3f} {uv1:<7.3f} {uv2-uv1:<+7.3f}   "
          f"{rt2:<7.3f} {rt1:<7.3f} {rt2-rt1:<+7.3f}   "
          f"{c2:<7.3f} {c1:<7.3f}")

# Verdict
print()
if len(h2_rows) >= 2:
    uv_pres = [(eps, uv) for eps, uv, *_ in h2_rows]
    red_drop = [(eps, rt) for eps, _, rt, *_ in h2_rows]
    print(f"  Verdict:")
    uv_at_max = max([uv for _, uv in uv_pres if uv is not None])
    uv_at_high = next((uv for eps, uv in uv_pres if eps >= 0.8), None)
    red_at_high = next((rt for eps, rt in red_drop if eps >= 0.8), None)
    if uv_at_high is not None:
        # H1 ε=0.9 UV was 0.777. If H2 UV stays >0.88, hypothesis confirmed.
        if uv_at_high >= 0.88:
            print(f"    UV preservation: ε=0.9 UV/HST={uv_at_high:.3f} (>=0.88, H1 was 0.777) → preserved")
        elif uv_at_high >= 0.83:
            print(f"    UV preservation: ε=0.9 UV/HST={uv_at_high:.3f} → partially preserved (H1 0.777)")
        else:
            print(f"    UV preservation: ε=0.9 UV/HST={uv_at_high:.3f} → NOT preserved (H1 0.777, no gain)")
    if red_at_high is not None:
        if red_at_high <= 1.30:
            print(f"    Red suppression: ε=0.9 red/HST={red_at_high:.3f} (<=1.30, H1 was 1.255) → still works")
        else:
            print(f"    Red suppression: ε=0.9 red/HST={red_at_high:.3f} → weakened (H1 1.255)")
print()
