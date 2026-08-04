#!/usr/bin/env python3
"""Re-rank all historical champion candidates by MC packet spectrum RMS_bn.

The formal-integral spectrum has algorithmic artifacts (hard Planck leak +
under-counted continuum opacity) that distort the metric. The MC packet
spectrum (lumina_spectrum.csv) is ground truth.

Walks logs/ddc15*/  for both lumina_spectrum.csv and
lumina_spectrum_formal.csv, computes baseline-norm RMS@FWHM=20k against HST
B-max stitched, and reports the top candidates with side-by-side MC vs FI.
"""
import os, glob, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
LOGS = f"{ROOT}/logs"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return float(np.trapezoid(flu[sel], lam[sel]))

def baseline_norm_rms(mod_lam, mod_flu, hlam, hflu, gH, fwhm=20000.0):
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0 or not np.isfinite(g):
        return np.nan
    mod_flu = mod_flu * (gH/g)
    selH = (hlam>=3000) & (hlam<=8000)
    hl, hf = hlam[selH], hflu[selH]
    dl = np.median(np.diff(hl)); mid = 0.5*(hl[0]+hl[-1])
    sig = (fwhm/C_KMS)*mid/2.355/dl
    sH = gaussian_filter1d(hf, sig, mode='nearest')
    selM = (mod_lam>=2900) & (mod_lam<=8100)
    ml, mf = mod_lam[selM], mod_flu[selM]
    if len(ml) < 10: return np.nan
    dlM = np.median(np.diff(ml)); midM = 0.5*(ml[0]+ml[-1])
    sigM = (fwhm/C_KMS)*midM/2.355/dlM
    sM = gaussian_filter1d(mf, sigM, mode='nearest')
    common = (hl>=ml[0]) & (hl<=ml[-1])
    if common.sum() < 10: return np.nan
    mb = np.interp(hl[common], ml, mf/sM)
    return float(np.sqrt(np.mean((hf[common]/sH[common] - mb)**2)))

def band_ratios(lam, flu, hlam, hflu, gH):
    g = band_int(lam, flu, 4500, 5800)
    if g <= 0: return None
    flu_n = flu * (gH/g)
    BANDS = [("UV", 2900, 4500), ("opt", 4500, 5800), ("yel", 5800, 6500),
             ("redI", 6500, 7400), ("redII", 7400, 8200), ("redIII", 8200, 9000)]
    rats = []
    for nm, lo, hi in BANDS:
        m = band_int(lam, flu_n, lo, hi)
        h_ = band_int(hlam, hflu, lo, hi)
        rats.append(m/h_ if h_>0 else np.nan)
    return rats

h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
gH = band_int(hlam, hflu, 4500, 5800)

dirs = sorted(glob.glob(f"{LOGS}/ddc15*/"))
rows = []
for d in dirs:
    tag = os.path.basename(d.rstrip('/'))
    mc_path = f"{d}lumina_spectrum.csv"
    fi_path = f"{d}lumina_spectrum_formal.csv"
    if not (os.path.exists(mc_path) and os.path.exists(fi_path)):
        continue
    try:
        mcd = pd.read_csv(mc_path)
        fid = pd.read_csv(fi_path)
    except Exception:
        continue
    if "wavelength_angstrom" not in mcd.columns or "flux" not in mcd.columns:
        continue
    mc_lam, mc_flu = mcd["wavelength_angstrom"].values, mcd["flux"].values
    fi_lam, fi_flu = fid["wavelength_angstrom"].values, fid["flux"].values
    mc_rms = baseline_norm_rms(mc_lam, mc_flu, hlam, hflu, gH)
    fi_rms = baseline_norm_rms(fi_lam, fi_flu, hlam, hflu, gH)
    mc_rats = band_ratios(mc_lam, mc_flu, hlam, hflu, gH)
    if mc_rats is None:
        continue
    rows.append((tag, mc_rms, fi_rms, *mc_rats))

cols = ["tag", "MC_RMS", "FI_RMS", "UV", "opt", "yel", "redI", "redII", "redIII"]
df = pd.DataFrame(rows, columns=cols).sort_values("MC_RMS")
df.to_csv(f"{ROOT}/figures/champions_mc_rerank.csv", index=False)
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 200
pd.options.display.max_colwidth = 60

print(f"=== Top 25 by MC RMS_bn (MC = ground truth, FI = formal integral) ===")
print(df.head(25).to_string(index=False))

print(f"\n=== Tier breakdown (MC) ===")
for tier, lim in [("T1 ≤0.20", 0.20), ("T2 ≤0.18", 0.18), ("T3 ≤0.17", 0.17), ("T4 ≤0.164", 0.164)]:
    sub = df[df["MC_RMS"] <= lim]
    print(f"  {tier}: {len(sub)} cells")
    for _, r in sub.head(5).iterrows():
        print(f"    {r['tag']:55s}  MC={r['MC_RMS']:.4f}  FI={r['FI_RMS']:.4f}  redII={r['redII']:.2f}")

print(f"\nsaved CSV: {ROOT}/figures/champions_mc_rerank.csv")
print(f"  N total = {len(df)}")
