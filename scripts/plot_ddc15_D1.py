#!/usr/bin/env python3
"""D1: DDC15 v10400 X_Fe_out=0.05 L_inner sweep — red continuum thermal-origin test."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "155764"

L_VALS = ["7e42", "8.5e42", "1.0e43", "1.15e43", "1.3e43"]
CELLS = [(L, ROOT/f"logs/ddc15D1_{JOB}_ddc15D1_L{L}/lumina_spectrum_formal.csv")
         for L in L_VALS]
OUT = ROOT/f"figures/ddc15_D1_{JOB}_Linner_sweep.png"
C_KMS = 299792.458
C2_CHAMP = ROOT/"logs/ddc15C2_155756_ddc15C2_xFeO0.05/lumina_spectrum_formal.csv"

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
for L, p in CELLS:
    d = scale_to_hst(p)
    if d is None: print(f"  MISSING: {p}"); continue
    data[L] = d

c2 = scale_to_hst(C2_CHAMP)

fig, axes = plt.subplots(3, 1, figsize=(15, 13))
cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(data)))

ax = axes[0]
ax.plot(hlam, hflu*1e14, 'k-', lw=1.2, label='HST 2011fe B-max', alpha=0.95)
for (L, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=0.9, alpha=0.85, label=f'D1 L={L}')
if c2 is not None:
    ax.plot(c2[0], c2[1]*1e14, color='tab:cyan', lw=0.7, alpha=0.55, ls='--',
            label='C2 champ X_Fe_out=0.05 (L=1.0e43)')
ax.set_xlim(1700, 9500)
ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title(f'D1 L_inner sweep on DDC15 X_Fe_out=0.05 v_in=10400 (job {JOB})')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
m = (hlam>=5500)&(hlam<=9500)
ymax = hflu[m].max()*1e14*1.20
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
for (L, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.3, alpha=0.9, label=f'L={L}')
ax.set_xlim(5500, 9500); ax.set_ylim(0, ymax)
for lam_m, lab in [(5972,'Si II'),(6355,'Si II'),(7773,'O I'),
                    (8498,'Ca II IR'),(8542,'Ca II IR'),(8662,'Ca II IR')]:
    ax.axvline(lam_m, color='gray', ls=':', alpha=0.4, lw=0.7)
    ax.text(lam_m, ymax*0.92, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.5)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Red continuum [5500-9500]Å — should drop with L_inner')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

ax = axes[2]
m = (hlam>=3000)&(hlam<=4500)
ymax = hflu[m].max()*1e14*1.20
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
for (L, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.3, alpha=0.9, label=f'L={L}')
ax.set_xlim(3000, 4500); ax.set_ylim(0, ymax)
for lam_m, lab in [(3242,'Ti II'),(3349,'Ti II'),(3613,'Fe II'),(3727,'Fe II'),
                    (3938,'Ca II K'),(3968,'Ca II H'),(4233,'Fe II')]:
    ax.axvline(lam_m, color='gray', ls=':', alpha=0.4, lw=0.7)
    ax.text(lam_m, ymax*0.92, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.5)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Ti II curtain [3000-4500]Å — risk: green band drop with low L')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

print(f"\n=== Band ratios (model/HST, green-normalized) ===")
BANDS = [(2300,3500,"UV-mid"), (3200,3500,"Ti II"), (3500,4500,"blue"),
         (4500,5800,"green"), (5500,9500,"red-tot"),
         (5800,6800,"Si red"), (6800,8000,"OI/cont"), (8000,9500,"Ca IR")]
hdr = f"{'L_inner':<10s}"
for _,_,lab in BANDS: hdr += f"{lab:>10s}"
print(hdr)
for L, (lam, flu) in data.items():
    row = f"{L:<10s}"
    for lo,hi,_ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, flu, lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)
if c2 is not None:
    row = f"{'C2 champ':<10s}"
    for lo,hi,_ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(c2[0], c2[1], lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)

print(f"\n=== Energy flow: ΔF_UV [3000,4500] vs ΔF_red [5500,9500] ===")
I_HST_uv  = band_int(hlam, hflu, 3000, 4500)
I_HST_red = band_int(hlam, hflu, 5500, 9500)
print(f"  HST  UV={I_HST_uv:.3e}  red={I_HST_red:.3e}")
print(f"  {'L_inner':<10s} {'ΔF_UV':>11s} {'ΔF_red':>11s} {'red/uv':>8s}")
for L, (lam, flu) in data.items():
    dF_uv = I_HST_uv - band_int(lam, flu, 3000, 4500)
    dF_red = band_int(lam, flu, 5500, 9500) - I_HST_red
    r = dF_red/dF_uv if dF_uv > 0 else float('nan')
    print(f"  {L:<10s} {dF_uv:>11.3e} {dF_red:>11.3e} {r:>8.2f}")

print(f"\n=== band-log10-RMS (global cascade reddening metric, target ≤ 0.05) ===")
LOG_BANDS = [(2300,3500),(3500,4500),(4500,5800),(5500,6800),(6800,8000),(8000,9500)]
def band_log_rms(lam, flu):
    ds = []
    for lo,hi in LOG_BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, flu, lo, hi)
        if rH>0 and rM>0: ds.append(np.log10(rM/rH))
    return np.sqrt(np.mean([d*d for d in ds])) if ds else float('nan')
for L, (lam, flu) in data.items():
    print(f"  D1 L={L:10s}  band-log10-RMS = {band_log_rms(lam,flu):.4f}")
if c2 is not None:
    print(f"  C2 champ X_Fe_out=0.05      band-log10-RMS = {band_log_rms(c2[0],c2[1]):.4f}")

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

print(f"\n=== Baseline-norm RMS [3000,8000]Å (line-shape secondary) ===")
hlam_n, hflu_n = gauss_baseline(hlam, hflu)
from scipy.interpolate import interp1d
for L, (lam, flu) in data.items():
    mlam, mflu = gauss_baseline(lam, flu)
    if mlam is None: continue
    fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
    rms = np.sqrt(np.nanmean((fm(hlam_n) - hflu_n)**2))
    print(f"  D1 L={L:10s}  RMS_bn = {rms:.4f}")
if c2 is not None:
    mlam, mflu = gauss_baseline(c2[0], c2[1])
    if mlam is not None:
        fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
        rms = np.sqrt(np.nanmean((fm(hlam_n) - hflu_n)**2))
        print(f"  C2 champ X_Fe_out=0.05      RMS_bn = {rms:.4f}")
