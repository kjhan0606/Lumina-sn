#!/usr/bin/env python3
"""L1: Si II A_ul λ<4000Å direct scaling sweep — does Si II UV-line cut affect red excess?"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "156085"

FACTORS = ["f1p0_ctrl", "f0p5", "f0p2", "f0p05"]
CELLS = [(f, ROOT/f"logs/ddc15L1_{JOB}_ddc15L1_SiII_{f}/lumina_spectrum_formal.csv")
         for f in FACTORS]
OUT = ROOT/f"figures/ddc15_L1_{JOB}_SiII_Aul_sweep.png"
C_KMS = 299792.458
SI_LAB = 6355.0  # Si II rest λ

def load(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

hlam, hflu = load(HST)
gH = band_int(hlam, hflu, 4500, 5800)

def scale_to_hst(p):
    if not p.exists(): return None
    lam, flu = load(p)
    return lam, flu * (gH / band_int(lam, flu, 4500, 5800))

data = {}
for f, p in CELLS:
    d = scale_to_hst(p)
    if d is None: print(f"  MISSING: {p}"); continue
    data[f] = d

fig, axes = plt.subplots(3, 1, figsize=(15, 13))
cmap = plt.cm.cividis(np.linspace(0.15, 0.85, len(data)))

ax = axes[0]
ax.plot(hlam, hflu*1e14, 'k-', lw=1.2, label='HST 2011fe B-max', alpha=0.95)
for (f, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=0.9, alpha=0.85, label=f'L1 {f}')
ax.set_xlim(1700, 9500)
ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title(f'L1 Si II A_ul (λ<4000Å) sweep on C2-base (job {JOB})')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
m = (hlam>=5500)&(hlam<=6800)
ymax = hflu[m].max()*1e14*1.20
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
for (f, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.3, alpha=0.9, label=f'{f}')
ax.set_xlim(5500, 6800); ax.set_ylim(0, ymax)
ax.axvline(SI_LAB, color='gray', ls=':', alpha=0.4, lw=0.7)
ax.text(SI_LAB, ymax*0.92, 'Si II 6355', rotation=90, fontsize=7, va='top', ha='right', alpha=0.5)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Si II 6355 trough')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

ax = axes[2]
m = (hlam>=5500)&(hlam<=9500)
ymax = hflu[m].max()*1e14*1.20
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
for (f, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.3, alpha=0.9, label=f'{f}')
ax.set_xlim(5500, 9500); ax.set_ylim(0, ymax)
for lam_m, lab in [(5972,'Si II'),(6355,'Si II'),(7773,'O I'),
                    (8498,'Ca II IR'),(8542,'Ca II IR'),(8662,'Ca II IR')]:
    ax.axvline(lam_m, color='gray', ls=':', alpha=0.4, lw=0.7)
    ax.text(lam_m, ymax*0.92, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.5)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Red continuum [5500-9500]Å')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

print(f"\n=== Band ratios (model/HST, green-normalized) ===")
BANDS = [(2300,3500,"UV-mid"), (3200,3500,"Ti II"), (3500,4500,"blue"),
         (4500,5800,"green"), (5500,9500,"red-tot"),
         (5800,6800,"Si red"), (6800,8000,"OI/cont"), (8000,9500,"Ca IR")]
hdr = f"{'factor':<12s}"
for _,_,lab in BANDS: hdr += f"{lab:>10s}"
print(hdr)
for f, (lam, flu) in data.items():
    row = f"{f:<12s}"
    for lo,hi,_ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, flu, lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)

def trough_v_depth(lam, flu, lab=SI_LAB, half_aa=200):
    """Return (v_kms_blueshift, depth_rel_continuum) of trough nearest lab."""
    m = (lam>=lab-half_aa)&(lam<=lab+half_aa)
    sl, sf = lam[m], flu[m]
    if len(sl)<5: return np.nan, np.nan
    imin = np.argmin(sf)
    lam_min = sl[imin]; f_min = sf[imin]
    # continuum estimate = max in window
    f_cont = sf.max()
    depth = 1.0 - f_min/f_cont if f_cont>0 else np.nan
    v_kms = (lab - lam_min)/lab * C_KMS
    return v_kms, depth

print(f"\n=== Si II 6355 trough velocity + depth ===")
hv, hd = trough_v_depth(hlam, hflu)
print(f"  HST                v={hv:+8.0f} km/s  λ_min={SI_LAB*(1-hv/C_KMS):.1f}Å  depth={hd:.3f}")
for f, (lam, flu) in data.items():
    v, d = trough_v_depth(lam, flu)
    dv = v - hv
    print(f"  L1 {f:<10s}  v={v:+8.0f} km/s  λ_min={SI_LAB*(1-v/C_KMS):.1f}Å  depth={d:.3f}  Δv={dv:+6.0f}")

print(f"\n=== band-log10-RMS (global cascade reddening, target ≤ 0.05) ===")
LOG_BANDS = [(2300,3500),(3500,4500),(4500,5800),(5500,6800),(6800,8000),(8000,9500)]
def band_log_rms(lam, flu):
    ds = []
    for lo,hi in LOG_BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, flu, lo, hi)
        if rH>0 and rM>0: ds.append(np.log10(rM/rH))
    return np.sqrt(np.mean([d*d for d in ds])) if ds else float('nan')
for f, (lam, flu) in data.items():
    print(f"  L1 {f:<10s}  band-log10-RMS = {band_log_rms(lam,flu):.4f}")

def gauss_baseline(lam, flu, fwhm_kms=40000, mask_lo=3000, mask_hi=8000):
    from scipy.ndimage import gaussian_filter1d
    sel = (lam >= mask_lo) & (lam <= mask_hi)
    sub_lam = lam[sel]; sub_flu = flu[sel]
    if len(sub_lam) < 50: return None, None
    dlam = np.median(np.diff(sub_lam))
    lam_mid = 0.5*(mask_lo+mask_hi)
    sigma_aa = (fwhm_kms/C_KMS) * lam_mid / 2.355
    sigma_pix = sigma_aa / dlam
    base = gaussian_filter1d(sub_flu, sigma_pix, mode='nearest')
    return sub_lam, sub_flu / base

print(f"\n=== Baseline-norm RMS [3000,8000]Å (PRIMARY, target ≤ 0.20) ===")
hlam_n, hflu_n = gauss_baseline(hlam, hflu)
from scipy.interpolate import interp1d
for f, (lam, flu) in data.items():
    mlam, mflu = gauss_baseline(lam, flu)
    if mlam is None: continue
    fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
    rms = np.sqrt(np.nanmean((fm(hlam_n) - hflu_n)**2))
    print(f"  L1 {f:<10s}  RMS_bn = {rms:.4f}")
