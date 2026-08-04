#!/usr/bin/env python3
"""Q1c: widened LAMBDA_MIN band [5500,9500] for full red coverage."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458
SI_LAB = 6355.0
JOB = 156285

CELLS = [
    ("Q1c_ref",  f"logs/ddc15Q1c_{JOB}_ddc15Q1c_Q1c_ref/lumina_spectrum_formal.csv"),
    ("Q1c_w05",  f"logs/ddc15Q1c_{JOB}_ddc15Q1c_Q1c_w05/lumina_spectrum_formal.csv"),
    ("Q1c_w03",  f"logs/ddc15Q1c_{JOB}_ddc15Q1c_Q1c_w03/lumina_spectrum_formal.csv"),
    ("Q1c_w015", f"logs/ddc15Q1c_{JOB}_ddc15Q1c_Q1c_w015/lumina_spectrum_formal.csv"),
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
print(f"=== Q1c LAMBDA_MIN widened [5500,9500]Å for Cr/Fe/Co III, job {JOB} ===")
print("    Stack: Ni 0.3 + Si 0.05 + Fe 0.3 + Co 0.3 (λ<4000Å, ion=1)")
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
print("=== Δ RMS_bn vs Q1c_ref @ FWHM=20k & 40k ===")
print(f"{'comparison':<28s}  {'Δ HST_20k':>11s} {'Δ HST_40k':>11s} {'Δ TAR_20k':>11s} {'Δ TAR_40k':>11s}")
print("-"*88)
ch = rms["Q1c_ref"]
for name in ["Q1c_w05", "Q1c_w03", "Q1c_w015"]:
    d = tuple(rms[name][i] - ch[i] for i in range(4))
    print(f"{name+' − Q1c_ref':<28s}  {d[0]:>+11.4f} {d[1]:>+11.4f} {d[2]:>+11.4f} {d[3]:>+11.4f}")

# Cross-job vs Q1b
print()
print("="*88)
print("=== Cross-job: Q1c vs Q1b vs prior floors ===")
print(f"    job 156183 P2 Fe0.3+Co0.3:       HST_20k=0.2384")
print(f"    job 156190 Q1_ref (stack):       HST_20k=0.2379")
print(f"    job 156281 Q1b_ref:              HST_20k=0.2452")
print(f"    job 156281 Q1b_r03 [4500,7500]:  HST_20k=0.2375  ← Q1b winner")
print(f"    job 156285 Q1c_ref:              HST_20k={ch[0]:.4f}")
best_w = min([(n, rms[n][0]) for n in ['Q1c_w05','Q1c_w03','Q1c_w015']], key=lambda x: x[1])
print(f"    job 156285 best Q1c {best_w[0]} [5500,9500]: HST_20k={best_w[1]:.4f}  ← Q1c winner")

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
print(f"  Task #172 target ≤ {target}")
print(f"  Gap: {best[1][0]-target:+.4f}")
print()
deltas = {name: rms[name][0] - ch[0] for name in ["Q1c_w05", "Q1c_w03", "Q1c_w015"]}
best_q = min(deltas.items(), key=lambda x: x[1])
print("  Q1c band-widened verdict:")
if best_q[1] <= -0.005:
    print(f"    → LEVER ACTIVE: best {best_q[0]} Δ={best_q[1]:+.4f}")
elif abs(best_q[1]) < 0.005:
    print(f"    → LEVER WITHIN MC NOISE (best Δ={best_q[1]:+.4f}) — INCONCLUSIVE")
else:
    print(f"    → LEVER WORSE (best deviation {best_q[1]:+.4f}) — direction CLOSED")
print()
print("  Red-band ratios (HST relative):")
BANDS_RED = [(5800,6800,"Si-red"), (6800,8000,"OI/cont"), (8000,9500,"Ca-IR")]
for name, (lam, fH, fT) in data.items():
    parts = []
    for lo,hi,lab in BANDS_RED:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, fH, lo, hi)
        parts.append(f"{lab} {rM/rH:.3f}")
    print(f"    {name:<12s}  " + " | ".join(parts))
