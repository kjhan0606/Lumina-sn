#!/usr/bin/env python3
"""V0 — score X1_b3p0 B-max LUMINA spectrum against multiple observers at near-B-max.

Tests whether the 0.171 baseline-norm RMS is robust to the choice of comparison
spectrum, or specific to HST stitched.

Observation set (all near phase 0):
  HST B-max stitched      (phase  0.0, λ 1660-10300 Å, UV+optical, 152 pts)
  Snifs p+0.2d            (phase +0.2, λ 3301-9700 Å, optical only)
  Snifs p-0.8d            (phase -0.8, λ 3301-9700 Å)
  Snifs p+1.2d            (phase +1.2, λ 3301-9700 Å)

Common comparison band is constrained by Snifs cutoff: λ ∈ [3300, 8000] Å.
For the HST-vs-LUMINA score we also include the full canonical [3000,8000] band
that the rerank script uses (the 0.171 reference).
"""
import os, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
MODEL = f"{ROOT}/logs/ddc15X1_156525_ddc15X1_b3p0/lumina_spectrum.csv"
HST_STITCH = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
SNIFS_DIR = f"{ROOT}/data/sn2011fe/epochs"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return float(np.trapezoid(flu[sel], lam[sel]))

def rms_bn(mod_lam, mod_flu, obs_lam, obs_flu, gO, fwhm=20000.0,
           wl_lo=3000.0, wl_hi=8000.0):
    """Baseline-norm RMS: divide each spectrum by Gaussian-smoothed continuum,
    then RMS of difference on common wavelength grid."""
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0 or not np.isfinite(g): return np.nan
    mod_flu = mod_flu * (gO/g)
    selO = (obs_lam>=wl_lo) & (obs_lam<=wl_hi)
    ol, of = obs_lam[selO], obs_flu[selO]
    if len(ol) < 10: return np.nan
    dl = np.median(np.diff(ol)); mid = 0.5*(ol[0]+ol[-1])
    sig = (fwhm/C_KMS)*mid/2.355/dl
    sO = gaussian_filter1d(of, sig, mode='nearest')
    selM = (mod_lam>=wl_lo-100) & (mod_lam<=wl_hi+100)
    ml, mf = mod_lam[selM], mod_flu[selM]
    if len(ml) < 10: return np.nan
    dlM = np.median(np.diff(ml)); midM = 0.5*(ml[0]+ml[-1])
    sigM = (fwhm/C_KMS)*midM/2.355/dlM
    sM = gaussian_filter1d(mf, sigM, mode='nearest')
    common = (ol>=ml[0]) & (ol<=ml[-1])
    if common.sum() < 10: return np.nan
    mb = np.interp(ol[common], ml, mf/sM)
    return float(np.sqrt(np.mean((of[common]/sO[common] - mb)**2)))

# load LUMINA model
mdf = pd.read_csv(MODEL)
m_lam, m_flu = mdf["wavelength_angstrom"].values, mdf["flux"].values

# load observations
obs = {}
hdf = pd.read_csv(HST_STITCH)
obs["HST_Bmax_stitched"]  = (hdf["wavelength_angstrom"].values,
                             hdf["flux_erg_s_cm2_angstrom"].values)
for tag, fn in [("Snifs_p-0.8d", "sn2011fe_m0d8d.csv"),
                ("Snifs_p+0.2d", "sn2011fe_p0d2d.csv"),
                ("Snifs_p+1.2d", "sn2011fe_p1d2d.csv")]:
    sdf = pd.read_csv(f"{SNIFS_DIR}/{fn}", comment='#')
    obs[tag] = (sdf["wavelength_angstrom"].values,
                sdf["flux_erg_s_cm2_angstrom"].values)

print("V0 — X1_b3p0 B-max LUMINA spectrum vs near-B-max observations")
print("=" * 74)
print(f"  {'observation':28s}  {'phase':>6s}  {'λ-band':>12s}  {'RMS_bn':>8s}")
print("-" * 74)
# Snifs starts at 3301 — use 3300-8000 for fair Snifs comparison.
# HST also score on canonical [3000,8000] and shared [3300,8000].
results = []
for tag, (lam, flu) in obs.items():
    # Snifs has only optical → both runs use the same shared band for fairness
    if "HST" in tag:
        phase = "+0.0"
        # canonical band (the one the 0.171 reference uses)
        gO_can = band_int(lam, flu, 4500, 5800)
        rms_can = rms_bn(m_lam, m_flu, lam, flu, gO_can, wl_lo=3000, wl_hi=8000)
        print(f"  {tag:28s}  {phase:>6s}  {'[3000,8000]':>12s}  {rms_can:>8.4f}")
        results.append((tag, "+0.0", "[3000,8000]", rms_can))
        # shared band for cross-instrument comparison
        rms_sh  = rms_bn(m_lam, m_flu, lam, flu, gO_can, wl_lo=3300, wl_hi=8000)
        print(f"  {tag:28s}  {phase:>6s}  {'[3300,8000]':>12s}  {rms_sh:>8.4f}")
        results.append((tag, "+0.0", "[3300,8000]", rms_sh))
    else:
        phase = tag.split("_")[1].replace("p", "+").replace("m", "-").replace("d","")
        # canonical 4500-5800 normalization band — verify it's covered
        gO = band_int(lam, flu, 4500, 5800)
        rms = rms_bn(m_lam, m_flu, lam, flu, gO, wl_lo=3300, wl_hi=8000)
        print(f"  {tag:28s}  {phase:>6s}  {'[3300,8000]':>12s}  {rms:>8.4f}")
        results.append((tag, phase, "[3300,8000]", rms))

print("=" * 74)
out = f"{ROOT}/figures/v0_x1_multi_observer.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(results, columns=["observation","phase","band","RMS_bn"]).to_csv(out, index=False)
print(f"saved: {out}")
