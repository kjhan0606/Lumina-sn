#!/usr/bin/env python3
"""P1: Fe II / Co II A_ul factor sweep around N1+Co champion."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458
SI_LAB = 6355.0
JOB = 156163

CELLS = [
    ("champ_repl",  f"logs/ddc15P1_{JOB}_ddc15P1_champ_repl/lumina_spectrum_formal.csv"),
    ("Fe0.3",       f"logs/ddc15P1_{JOB}_ddc15P1_Fe0p3/lumina_spectrum_formal.csv"),
    ("Fe0.2",       f"logs/ddc15P1_{JOB}_ddc15P1_Fe0p2/lumina_spectrum_formal.csv"),
    ("Co0.3",       f"logs/ddc15P1_{JOB}_ddc15P1_Co0p3/lumina_spectrum_formal.csv"),
]

def load(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def gauss_baseline(lam, flu, fwhm_kms, mask_lo=3000, mask_hi=8000):
    sel = (lam >= mask_lo) & (lam <= mask_hi)
    sub_lam = lam[sel]; sub_flu = flu[sel]
    if len(sub_lam) < 50: return None, None
    dlam = np.median(np.diff(sub_lam))
    lam_mid = 0.5*(mask_lo+mask_hi)
    sigma_aa = (fwhm_kms/C_KMS) * lam_mid / 2.355
    sigma_pix = sigma_aa / dlam
    base = gaussian_filter1d(sub_flu, sigma_pix, mode='nearest')
    return sub_lam, sub_flu / base

def rms_bn(mlam, mflu, rlam, rflu, fwhm):
    mlam_n, mflu_n = gauss_baseline(mlam, mflu, fwhm)
    rlam_n, rflu_n = gauss_baseline(rlam, rflu, fwhm)
    if mlam_n is None or rlam_n is None: return np.nan
    fm = interp1d(mlam_n, mflu_n, kind='linear', bounds_error=False, fill_value=np.nan)
    return float(np.sqrt(np.nanmean((fm(rlam_n) - rflu_n)**2)))

def trough_v_depth(lam, flu, lab=SI_LAB, half_aa=200):
    m = (lam>=lab-half_aa)&(lam<=lab+half_aa)
    sl, sf = lam[m], flu[m]
    if len(sl)<5: return np.nan, np.nan
    imin = np.argmin(sf); lam_min = sl[imin]; f_min = sf[imin]
    f_cont = sf.max()
    depth = 1.0 - f_min/f_cont if f_cont>0 else np.nan
    v_kms = (lab - lam_min)/lab * C_KMS
    return v_kms, depth

hlam, hflu = load(HST)
tlam, tflu = load(TARDIS)
gH = band_int(hlam, hflu, 4500, 5800)
gT = band_int(tlam, tflu, 4500, 5800)

data = {}
for name, rel in CELLS:
    p = ROOT/rel
    if not p.exists(): print(f"MISSING {name}"); continue
    lam, flu = load(p)
    g = band_int(lam, flu, 4500, 5800)
    data[name] = (lam, flu*(gH/g), flu*(gT/g))

print("="*88)
print(f"=== P1 Fe/Co push around N1+Co champion, job {JOB} ===")
print("    Baked: Ni II f=0.3, Si II f=0.05, λ<4000Å")
print(f"{'cell':<14s}  {'HST_20k':>9s} {'HST_40k':>9s} {'TAR_20k':>9s} {'TAR_40k':>9s}")
print("-"*88)
rms = {}
for name, (lam, fH, fT) in data.items():
    r_h20 = rms_bn(lam, fH, hlam, hflu, 20000)
    r_h40 = rms_bn(lam, fH, hlam, hflu, 40000)
    r_t20 = rms_bn(lam, fT, tlam, tflu, 20000)
    r_t40 = rms_bn(lam, fT, tlam, tflu, 40000)
    rms[name] = (r_h20, r_h40, r_t20, r_t40)
    print(f"{name:<14s}  {r_h20:>9.4f} {r_h40:>9.4f} {r_t20:>9.4f} {r_t40:>9.4f}")

print()
print("="*88)
print("=== Δ RMS_bn vs champ_repl (local champion baseline) @ FWHM=20k & 40k ===")
print(f"{'comparison':<28s}  {'Δ HST_20k':>11s} {'Δ HST_40k':>11s} {'Δ TAR_20k':>11s} {'Δ TAR_40k':>11s}")
print("-"*88)
ch = rms["champ_repl"]
for name in ["Fe0.3", "Fe0.2", "Co0.3"]:
    d = tuple(rms[name][i] - ch[i] for i in range(4))
    print(f"{name+' − champ':<28s}  {d[0]:>+11.4f} {d[1]:>+11.4f} {d[2]:>+11.4f} {d[3]:>+11.4f}")

# Cross-job MC noise
print()
print("="*88)
print("=== MC noise envelope (Co=0.5/Fe=0.5 reproducibility) ===")
print(f"    job 156155 O1_Co (same config):  HST_20k=0.2310")
print(f"    job 156163 champ_repl:           HST_20k={ch[0]:.4f}")
print(f"    Cross-seed Δ: {ch[0]-0.2310:+.4f}  (MC noise σ~0.02)")

# Si II trough
print()
print("="*88)
print("=== Si II 6355 trough velocity, depth (HST-scaled) ===")
hv, hd = trough_v_depth(hlam, hflu)
print(f"  {'HST':<14s}  v={hv:+8.0f} km/s  depth={hd:.3f}")
for name, (lam, fH, fT) in data.items():
    v, d = trough_v_depth(lam, fH)
    print(f"  {name:<14s}  v={v:+8.0f}  Δv={v-hv:+7.0f}  depth={d:.3f}")

# Band ratios
print()
print("="*88)
print("=== Band ratios (model/HST, green-normalized) ===")
BANDS = [(2300,3500,"UV-mid"), (3500,4500,"blue"), (4500,5800,"green"),
         (5800,6800,"Si-red"), (6800,8000,"OI/cont"), (8000,9500,"Ca-IR")]
hdr = f"{'cell':<14s}"
for _,_,lab in BANDS: hdr += f"{lab:>10s}"
print(hdr)
for name, (lam, fH, fT) in data.items():
    row = f"{name:<14s}"
    for lo,hi,_ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, fH, lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)

print()
print("="*88)
print("=== DECISION LOGIC ===")
best = min(rms.items(), key=lambda x: x[1][0])
print(f"  Best cell @ FWHM=20k vs HST: {best[0]} = {best[1][0]:.4f}")
target = 0.20
print(f"  Task #172 target ≤ {target} vs HST @ FWHM=20k")
print(f"  Gap: {best[1][0]-target:+.4f}")
print()
print("  Trends:")
fe_axis = [(0.5, ch[0]), (0.3, rms['Fe0.3'][0]), (0.2, rms['Fe0.2'][0])]
print(f"    Fe II axis (Co=0.5): " + " | ".join([f"f={f:.1f}→{r:.4f}" for f,r in fe_axis]))
print(f"    Co II axis (Fe=0.5): f=0.5→{ch[0]:.4f} | f=0.3→{rms['Co0.3'][0]:.4f}")
