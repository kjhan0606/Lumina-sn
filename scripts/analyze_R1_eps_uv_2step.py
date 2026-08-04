#!/usr/bin/env python3
"""R1: True 2-step UV→opt→red macro-atom cascade.

Compares EPS_UV={0,0.2,0.4,0.6,0.9} with band-constrained re-emit into
[3500,5500]Å against Q1b r03 baseline (0.2375 floor) and H1 full-thermal
production result (0.288 at ε=0.9, 800K). Goal: prove that constraining
the Planck draw to the optical band preserves P-Cygni shape while still
redirecting UV flux to red.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458
SI_LAB = 6355.0
JOB = 156298

CELLS = [
    ("R1_ref",  f"logs/ddc15R1_{JOB}_ddc15R1_R1_ref/lumina_spectrum_formal.csv"),
    ("R1_e02",  f"logs/ddc15R1_{JOB}_ddc15R1_R1_e02/lumina_spectrum_formal.csv"),
    ("R1_e04",  f"logs/ddc15R1_{JOB}_ddc15R1_R1_e04/lumina_spectrum_formal.csv"),
    ("R1_e06",  f"logs/ddc15R1_{JOB}_ddc15R1_R1_e06/lumina_spectrum_formal.csv"),
    ("R1_e09",  f"logs/ddc15R1_{JOB}_ddc15R1_R1_e09/lumina_spectrum_formal.csv"),
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

print("="*92)
print(f"=== R1 true 2-step UV→opt→red cascade [3500,5500]Å, job {JOB} ===")
print("    Q1b r03 base: Ni 0.3 + Si 0.05 + Fe 0.3 + Co 0.3 (λ<4000Å, ion=1)")
print("                  + Cr/Fe/Co III f=0.3 λ∈[4500,7500] (Q1b winner, RMS_bn=0.2375)")
print(f"{'cell':<14s}  {'HST_20k':>9s} {'HST_40k':>9s} {'TAR_20k':>9s} {'TAR_40k':>9s}")
print("-"*92)
rms = {}
for name, (lam, fH, fT) in data.items():
    r_h20 = rms_bn(lam, fH, hlam, hflu, 20000)
    r_h40 = rms_bn(lam, fH, hlam, hflu, 40000)
    r_t20 = rms_bn(lam, fT, tlam, tflu, 20000)
    r_t40 = rms_bn(lam, fT, tlam, tflu, 40000)
    rms[name] = (r_h20, r_h40, r_t20, r_t40)
    print(f"{name:<14s}  {r_h20:>9.4f} {r_h40:>9.4f} {r_t20:>9.4f} {r_t40:>9.4f}")

print()
print("="*92)
print("=== Δ RMS_bn vs R1_ref (intra-job EPS_UV=0 anchor) @ FWHM=20k & 40k ===")
print(f"{'comparison':<28s}  {'Δ HST_20k':>11s} {'Δ HST_40k':>11s} {'Δ TAR_20k':>11s} {'Δ TAR_40k':>11s}")
print("-"*92)
ch = rms["R1_ref"]
for name in ["R1_e02", "R1_e04", "R1_e06", "R1_e09"]:
    if name not in rms: continue
    d = tuple(rms[name][i] - ch[i] for i in range(4))
    print(f"{name+' − R1_ref':<28s}  {d[0]:>+11.4f} {d[1]:>+11.4f} {d[2]:>+11.4f} {d[3]:>+11.4f}")

print()
print("="*92)
print("=== Cross-job: R1 vs full-thermal H1 vs Q1b r03 vs Q1c ref ===")
print(f"    job 156281 Q1b_r03 [4500,7500] f=0.3:                HST_20k=0.2375  ← Q1b winner / floor")
print(f"    job 156285 Q1c_ref base:                              HST_20k=0.2358")
print(f"    job 156038 H1p ε=0.9 full-thermal (800K production):  HST_20k=0.288 (shape destroyed)")
print(f"    job 156018 H1b ε=0.4 200K screen:                     HST_20k=0.245 (joint-cost min)")
print(f"    job {JOB} R1_ref EPS_UV=0:                            HST_20k={ch[0]:.4f}")
best_r = min([(n, rms[n][0]) for n in ['R1_e02','R1_e04','R1_e06','R1_e09'] if n in rms],
             key=lambda x: x[1])
print(f"    job {JOB} best R1 {best_r[0]} (2-step):              HST_20k={best_r[1]:.4f}  ← R1 winner")

# Si II trough
print()
print("="*92)
print("=== Si II 6355 trough velocity, depth (HST-scaled) ===")
hv, hd = trough_v_depth(hlam, hflu)
print(f"  {'HST':<14s}  v={hv:+8.0f} km/s  depth={hd:.3f}")
for name, (lam, fH, fT) in data.items():
    v, d = trough_v_depth(lam, fH)
    print(f"  {name:<14s}  v={v:+8.0f}  Δv={v-hv:+7.0f}  depth={d:.3f}")

# Band ratios
print()
print("="*92)
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
print("="*92)
print("=== DECISION LOGIC ===")
best = min(rms.items(), key=lambda x: x[1][0])
print(f"  Best cell @ FWHM=20k vs HST: {best[0]} = {best[1][0]:.4f}")
target = 0.20
print(f"  Task #172 target ≤ {target}")
print(f"  Gap: {best[1][0]-target:+.4f}")
print()
non_ref = [n for n in ['R1_e02','R1_e04','R1_e06','R1_e09'] if n in rms]
deltas = {name: rms[name][0] - ch[0] for name in non_ref}
if deltas:
    best_q = min(deltas.items(), key=lambda x: x[1])
    print("  R1 2-step verdict:")
    if best_q[1] <= -0.005:
        print(f"    → LEVER ACTIVE: best {best_q[0]} Δ={best_q[1]:+.4f}")
    elif abs(best_q[1]) < 0.005:
        print(f"    → LEVER WITHIN MC NOISE (best Δ={best_q[1]:+.4f}) — INCONCLUSIVE")
    else:
        print(f"    → LEVER WORSE (best deviation {best_q[1]:+.4f}) — direction CLOSED")

print()
print("  Comparison vs H1 full-thermal (800K production):")
print("    H1p ε=0.9 full-thermal: 0.288 (shape destroyed by Planck smear)")
if 'R1_e09' in rms:
    print(f"    R1   ε=0.9 2-step [3500,5500]: {rms['R1_e09'][0]:.4f}")
    h1p = 0.288
    delta_vs_h1p = rms['R1_e09'][0] - h1p
    if delta_vs_h1p <= -0.01:
        print(f"    → 2-step IMPROVES on full-thermal by Δ={delta_vs_h1p:+.4f}")
    else:
        print(f"    → 2-step matches/worsens full-thermal (Δ={delta_vs_h1p:+.4f})")
