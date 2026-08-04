#!/usr/bin/env python3
"""#180: find Si II 6355 trough λ_min in skipSi runs, compare to HST + empirical ML."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
SI_REST = 6355.0          # absorption rest wavelength (effective)
C_KMS   = 299792.458

CELLS = [
    ("skipSi_rp0p00_n9p0_v10400", "skipSi v_in=10400"),
    ("skipSi_rp0p00_n9p0_v12500", "skipSi v_in=12500"),
]

def load(p):
    d = pd.read_csv(p); return d.iloc[:,0].values, d.iloc[:,1].values

def trough_lambda(lam, flu, lo=5800, hi=6300, smooth_pts=15):
    """Find minimum of smoothed flux within the Si II 6355 absorption window."""
    m = (lam>=lo)&(lam<=hi)
    if m.sum() < 5: return None, None
    lam_w = lam[m]; flu_w = flu[m]
    flu_s = gaussian_filter1d(flu_w, sigma=smooth_pts)
    i = np.argmin(flu_s)
    return lam_w[i], flu_s[i]

def vel_kms(lam_obs, lam_rest=SI_REST):
    return C_KMS * (lam_rest - lam_obs)/lam_rest   # blueshift positive

hlam, hflu = load(HST)
lam_h, fmin_h = trough_lambda(hlam, hflu)
v_h = vel_kms(lam_h)

print(f"\n=== Si II 6355 trough position (job 155590) ===")
print(f"  rest λ = {SI_REST:.1f} Å")
print(f"  HST B-max:                λ_min = {lam_h:.1f} Å,  v_blue = {v_h:7.0f} km/s")
print(f"  Empirical ML (X_emp):                                       v_form = 9611 km/s")
print()
print(f"  {'cell':<28}  {'λ_min':<10}  {'v_blue':<8}  {'Δv vs HST':<10}")
print("-"*68)

for sub, desc in CELLS:
    p = ROOT/f"logs/skipSi_phys_155590_{sub}/lumina_spectrum_formal.csv"
    if not p.exists():
        print(f"  {desc:<28}  MISSING"); continue
    lam, flu = load(p)
    lm, fm = trough_lambda(lam, flu)
    v = vel_kms(lm)
    dv = v - v_h
    print(f"  {desc:<28}  {lm:7.1f}     {v:7.0f}   {dv:+7.0f}")
print()
