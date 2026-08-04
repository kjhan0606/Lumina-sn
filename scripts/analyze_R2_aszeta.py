#!/usr/bin/env python3
"""R2: AUTOSTRUCTURE-derived ζ vs placeholder ζ on Q1b r03 stack.

Compares R2_ref (placeholder ζ, Co/Ni III = 0.2908) against R2_zfix (AS plan-D
ζ, Co III 0.94 / Ni III 0.97 / Cr III 0.82) on the Q1b r03 production base.
Targets baseline-norm RMS @ FWHM=20k vs HST 0.2375 production floor.
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
JOB = 156311

CELLS = [
    ("R2_ref",  f"logs/ddc15R2_{JOB}_ddc15R2_R2_ref/lumina_spectrum_formal.csv"),
    ("R2_zfix", f"logs/ddc15R2_{JOB}_ddc15R2_R2_zfix/lumina_spectrum_formal.csv"),
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
    if not p.exists(): print(f"MISSING {name}: {p}"); continue
    lam, flu = load(p)
    g = band_int(lam, flu, 4500, 5800)
    data[name] = (lam, flu*(gH/g), flu*(gT/g))

print("="*100)
print(f"=== R2: AS-derived ζ (Co/Ni/Cr III) on Q1b r03 stack, job {JOB} ===")
print("    Patches @ T=8000K:  Co III 0.291→0.94  |  Ni III 0.291→0.97  |  Cr III 0.74→0.82")
print(f"{'cell':<10s}  {'HST_20k':>9s} {'HST_40k':>9s} {'TAR_20k':>9s} {'TAR_40k':>9s}")
print("-"*100)
rms = {}
for name, (lam, fH, fT) in data.items():
    r_h20 = rms_bn(lam, fH, hlam, hflu, 20000)
    r_h40 = rms_bn(lam, fH, hlam, hflu, 40000)
    r_t20 = rms_bn(lam, fT, tlam, tflu, 20000)
    r_t40 = rms_bn(lam, fT, tlam, tflu, 40000)
    rms[name] = (r_h20, r_h40, r_t20, r_t40)
    print(f"{name:<10s}  {r_h20:>9.4f} {r_h40:>9.4f} {r_t20:>9.4f} {r_t40:>9.4f}")

if "R2_ref" in rms and "R2_zfix" in rms:
    d = [rms["R2_zfix"][i] - rms["R2_ref"][i] for i in range(4)]
    print()
    print(f"  Δ(zfix − ref):  HST_20k={d[0]:+.4f}  HST_40k={d[1]:+.4f}  "
          f"TAR_20k={d[2]:+.4f}  TAR_40k={d[3]:+.4f}")

print()
print("="*100)
print(f"    Q1b r03 production floor (job 156281):           HST_20k = 0.2375")
print(f"    R1 best (job 156298, 2-step ε_UV=0.2):           HST_20k = 0.2388")
print(f"    Task #172 target:                                HST_20k ≤ 0.2000")

# Si II trough
print()
print("="*100)
print("=== Si II 6355 trough velocity, depth (HST-scaled) ===")
hv, hd = trough_v_depth(hlam, hflu)
print(f"  {'HST':<10s}  v={hv:+8.0f} km/s  depth={hd:.3f}")
for name, (lam, fH, fT) in data.items():
    v, d = trough_v_depth(lam, fH)
    print(f"  {name:<10s}  v={v:+8.0f}  Δv={v-hv:+7.0f}  depth={d:.3f}")

# Band ratios
print()
print("="*100)
print("=== Band ratios (model/HST, green-normalized) ===")
BANDS = [(2300,3500,"UV-mid"), (3500,4500,"blue"), (4500,5800,"green"),
         (5800,6800,"Si-red"), (6800,8000,"OI/cont"), (8000,9500,"Ca-IR")]
hdr = f"{'cell':<10s}"
for _,_,lab in BANDS: hdr += f"{lab:>10s}"
print(hdr)
for name, (lam, fH, fT) in data.items():
    row = f"{name:<10s}"
    for lo,hi,_ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, fH, lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)

print()
print("="*100)
print("=== DECISION LOGIC ===")
if "R2_ref" in rms and "R2_zfix" in rms:
    delta = rms["R2_zfix"][0] - rms["R2_ref"][0]
    print(f"  R2_zfix HST_20k = {rms['R2_zfix'][0]:.4f}")
    print(f"  R2_ref  HST_20k = {rms['R2_ref'][0]:.4f}")
    print(f"  Δ(zfix − ref)   = {delta:+.4f}")
    print()
    if delta <= -0.005:
        print(f"  → AS-derived ζ IMPROVES baseline-norm RMS")
        print(f"    Adopt aszeta as new reference dir. Stack with Q1b r03 + R1 2-step possible.")
    elif abs(delta) < 0.005:
        print(f"  → AS-derived ζ matches placeholder within MC noise")
        print(f"    No improvement on RMS_bn but the patch is data hygiene — keep it.")
    else:
        print(f"  → AS-derived ζ WORSENS RMS by {delta:+.4f}")
        print(f"    Placeholder was load-bearing on Q1b r03. Revert. Confirms P4 #166 direction.")
