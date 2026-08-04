#!/usr/bin/env python3
"""H1b: Dense LUMINA_EPS_UV knee sweep analysis. Quantifies trade-off between
red/HST reduction and UV+blue collapse. Adds composite score:
  cost = sqrt( ((UV+blue/HST - 1)**2 + (red/HST - 1)**2)/2 )
plus per-band ratios and baseline-norm RMS. Includes H1 reference points
(0.0/0.3/0.6/0.9 from job 156014) for continuity.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "156015"
H1_JOB = sys.argv[2] if len(sys.argv) > 2 else "156014"

# H1b 8-cell + H1 4-cell merge (sorted EPS_UV)
H1B_EPS = ["0.10","0.15","0.20","0.25","0.30","0.40","0.50","0.70"]
H1_EPS  = ["0.0","0.3","0.6","0.9"]

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
gH  = band_int(hlam, hflu, 4500, 5800)

print(f"\n=== H1b knee sweep (job {JOB}, merged with H1 {H1_JOB}) ===")
print(f"  HST red-tot [5500,9500]Å = {rtH:.3e}")
print(f"  HST UV+blue [3000,5500]Å = {uvH:.3e}\n")

# Collect all variants
rows = []
def add_variant(eps, path):
    d = scale(path, hlam, hflu)
    if d is None:
        return None
    lam, flu = d
    rt = band_int(lam, flu, *RED_TOT) / rtH
    uv = band_int(lam, flu, *UV_TOT) / uvH
    bl = band_log10_rms(lam, flu, hlam, hflu, SUB_BANDS)
    try: bn = rms_baseline_norm(lam, flu, hlam, hflu)
    except: bn = float('nan')
    cost = np.sqrt(((uv-1)**2 + (rt-1)**2)/2)
    sub = {name: band_int(lam, flu, lo, hi)/band_int(hlam, hflu, lo, hi)
           for lo, hi, name in SUB_BANDS}
    rows.append((float(eps), uv, rt, cost, bl, bn, sub))
    return True

for eps in H1_EPS:
    add_variant(eps, ROOT/f"logs/ddc15H1_{H1_JOB}_ddc15H1_epsUV{eps}/lumina_spectrum_formal.csv")
for eps in H1B_EPS:
    add_variant(eps, ROOT/f"logs/ddc15H1b_{JOB}_ddc15H1b_epsUV{eps}/lumina_spectrum_formal.csv")

rows.sort(key=lambda r: r[0])

hdr = f"  {'EPS_UV':<7} {'UV/HST':<7} {'red/HST':<8} {'cost':<7} {'bl-RMS':<8} {'RMS_bn':<7}"
print(hdr); print("-"*len(hdr))
best_bn = (None, 1e9)
best_cost = (None, 1e9)
for eps, uv, rt, cost, bl, bn, sub in rows:
    flag = ""
    if bn < best_bn[1]: best_bn = (eps, bn)
    if cost < best_cost[1]: best_cost = (eps, cost)
    print(f"  {eps:<7.2f} {uv:<7.3f} {rt:<8.3f} {cost:<7.3f} {bl:<8.4f} {bn:<7.4f}")

print(f"\n  best RMS_bn: EPS_UV={best_bn[0]:.2f} → {best_bn[1]:.4f}")
print(f"  best cost  : EPS_UV={best_cost[0]:.2f} → {best_cost[1]:.4f}")

print(f"\n  --- Per-sub-band ratio (variant / HST) ---")
hdr2 = f"  {'EPS_UV':<7}  " + "  ".join(f"{name:<10}" for _,_,name in SUB_BANDS)
print(hdr2)
for eps, uv, rt, cost, bl, bn, sub in rows:
    row = f"  {eps:<7.2f}  "
    for _,_,name in SUB_BANDS:
        row += f"{sub[name]:<10.3f}  "
    print(row)

# Knee identification: largest 2nd derivative of red/HST vs EPS_UV
xs = np.array([r[0] for r in rows])
rs = np.array([r[2] for r in rows])
uvs = np.array([r[1] for r in rows])
if len(xs) >= 3:
    print(f"\n  --- Knee detection (red/HST vs EPS_UV) ---")
    d1 = np.gradient(rs, xs)
    d2 = np.gradient(d1, xs)
    print(f"  {'EPS_UV':<7} {'red/HST':<8} {'d(red)/dε':<11} {'d²(red)/dε²':<11}")
    for i, (e, r_, d, d2_) in enumerate(zip(xs, rs, d1, d2)):
        print(f"  {e:<7.2f} {r_:<8.3f} {d:<+11.3f} {d2_:<+11.3f}")
    knee_idx = np.argmax(np.abs(d2[1:-1])) + 1  # exclude edges
    print(f"  → strongest curvature at EPS_UV={xs[knee_idx]:.2f} (|d²|={abs(d2[knee_idx]):.3f})")
print()
