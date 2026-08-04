#!/usr/bin/env python3
"""O1: Co II + Cr II A_ul stack on N1 baked (Ni 0.3 + Si 0.05 + Fe 0.5)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458
SI_LAB = 6355.0
JOB = 156155

CELLS = [
    ("N1_repl",   f"logs/ddc15O1_{JOB}_ddc15O1_N1_repl/lumina_spectrum_formal.csv"),
    ("O1_Co",     f"logs/ddc15O1_{JOB}_ddc15O1_O1_Co/lumina_spectrum_formal.csv"),
    ("O1_Cr",     f"logs/ddc15O1_{JOB}_ddc15O1_O1_Cr/lumina_spectrum_formal.csv"),
    ("O1_CoCr",   f"logs/ddc15O1_{JOB}_ddc15O1_O1_CoCr/lumina_spectrum_formal.csv"),
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
print(f"=== O1 5-ion stack (Co + Cr on N1 baked), job {JOB} ===")
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

# Delta vs N1_repl (local baseline)
print()
print("="*88)
print("=== Δ RMS_bn vs N1_repl (local baseline) @ FWHM=20k & 40k ===")
print(f"{'comparison':<28s}  {'Δ HST_20k':>11s} {'Δ HST_40k':>11s} {'Δ TAR_20k':>11s} {'Δ TAR_40k':>11s}")
print("-"*88)
n1 = rms["N1_repl"]
for name in ["O1_Co", "O1_Cr", "O1_CoCr"]:
    d = tuple(rms[name][i] - n1[i] for i in range(4))
    print(f"{name+' − N1_repl':<28s}  {d[0]:>+11.4f} {d[1]:>+11.4f} {d[2]:>+11.4f} {d[3]:>+11.4f}")

# Reference: N1 triple from job 156141
print()
print("="*88)
print("=== Reference: N1 triple from job 156141 (different seed, MC σ~0.02) ===")
print(f"    job 156141 N1_triple:  HST_20k=0.2423, HST_40k=0.2933, TAR_20k=0.2458, TAR_40k=0.2905")
print(f"    job 156155 N1_repl:    HST_20k={n1[0]:.4f}, HST_40k={n1[1]:.4f}, TAR_20k={n1[2]:.4f}, TAR_40k={n1[3]:.4f}")
print(f"    Cross-seed Δ (156155 − 156141): {n1[0]-0.2423:+.4f}, {n1[1]-0.2933:+.4f}, {n1[2]-0.2458:+.4f}, {n1[3]-0.2905:+.4f}")

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

# Decision
print()
print("="*88)
print("=== DECISION LOGIC ===")
print()
print(f"  Best cell @ FWHM=20k vs HST: ", end="")
best = min(rms.items(), key=lambda x: x[1][0])
print(f"{best[0]} = {best[1][0]:.4f}")
target = 0.20
print(f"  Task #172 target: ≤ {target} vs HST @ FWHM=20k")
print(f"  Gap: {best[1][0]-target:+.4f}")
print()
co_d = rms["O1_Co"][0] - n1[0]
cr_d = rms["O1_Cr"][0] - n1[0]
cocr_d = rms["O1_CoCr"][0] - n1[0]
if cocr_d <= -0.005 and (co_d <= -0.005 or cr_d <= -0.005):
    print(f"  → Co II / Cr II A_ul lever ACTIVE")
elif abs(cocr_d) < 0.005:
    print(f"  → Co + Cr combined within MC noise — lever WEAK or CLOSED")
elif cocr_d > 0.005:
    print(f"  → Co + Cr WORSENS — antagonism")
else:
    print(f"  → mixed signal: Δ_Co={co_d:+.4f}, Δ_Cr={cr_d:+.4f}, Δ_combined={cocr_d:+.4f}")
