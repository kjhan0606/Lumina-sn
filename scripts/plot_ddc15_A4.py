#!/usr/bin/env python3
"""A4: DDC15-style abundance probe + v_in scan — vs HST + Path A champion."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "155722"

VINS = [10000, 10400, 10800, 11200]
CELLS = [(vi, ROOT/f"logs/ddc15A4_{JOB}_ddc15_v{vi:05d}/lumina_spectrum_formal.csv")
         for vi in VINS]
OUT = ROOT/f"figures/ddc15_A4_{JOB}_vs_hst.png"
C_KMS = 299792.458

PATH_A_CHAMP = ROOT/"logs/rho11feA_155702_rho11fe_fullNLTE/lumina_spectrum_formal.csv"
PATH_A_SKIPSI = ROOT/"logs/rho11feA_155702_rho11fe_skipSi/lumina_spectrum_formal.csv"

def load_csv(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo=4500, hi=5800):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def trough_min(lam, flu, lo=5800, hi=6300):
    m = (lam>=lo)&(lam<=hi)
    if m.sum()<5: return None
    i = np.argmin(flu[m]); return float(lam[m][i])

hlam, hflu = load_csv(HST)
hi_grn = band_int(hlam, hflu)

def load_and_scale(p):
    if not p.exists(): return None
    lam, flu = load_csv(p)
    li = band_int(lam, flu)
    return lam, flu * (hi_grn/li)

data = {}
for vi, p in CELLS:
    d = load_and_scale(p)
    if d is None: print(f"  MISSING: {p}"); continue
    data[vi] = d

pa_champ = load_and_scale(PATH_A_CHAMP)
pa_skipsi = load_and_scale(PATH_A_SKIPSI)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

ax = axes[0]
ax.plot(hlam, hflu*1e14, 'k-', lw=1.2, label='HST 2011fe B-max', alpha=0.95)
cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(data)))
for (vi, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=0.9, alpha=0.85, label=f'A4 DDC15 v_in={vi}')
if pa_champ is not None:
    ax.plot(pa_champ[0], pa_champ[1]*1e14, color='tab:green', lw=0.7, alpha=0.65, ls='--',
            label='Path A fullNLTE v10400 (champ 0.218)')
if pa_skipsi is not None:
    ax.plot(pa_skipsi[0], pa_skipsi[1]*1e14, color='tab:red', lw=0.7, alpha=0.55, ls=':',
            label='Path A SkipSi v10400 (0.252)')
ax.set_xlim(1700, 9000)
ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title(f'A4: DDC15-style + v_in scan (job {JOB}) vs HST + Path A champion')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
m = (hlam>=5500)&(hlam<=6800)
maxima = [hflu[m].max()]
for (vi, (lam, flu)) in data.items():
    msel = (lam>=5500)&(lam<=6800)
    if msel.sum() > 0: maxima.append(flu[msel].max())
ymax = max(maxima) * 1e14 * 1.1
ax.plot(hlam, hflu*1e14, 'k-', lw=2.5, label='HST', alpha=0.95)
for (vi, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.5, alpha=0.9, label=f'DDC15 v_in={vi}')
if pa_skipsi is not None:
    ax.plot(pa_skipsi[0], pa_skipsi[1]*1e14, color='tab:red', lw=1.0, alpha=0.5, ls=':',
            label='Path A SkipSi (Si II locked)')
ax.set_xlim(5500, 6800); ax.set_ylim(0, ymax)
for v_kms, ls, lab in [(0, ':', 'rest'), (9934, '-.', 'HST 9934'), (10080, ':', 'P+13 10080')]:
    lo = 6355*(1-v_kms/C_KMS)
    ax.axvline(lo, color='gray', ls=ls, alpha=0.5, lw=0.8)
    ax.text(lo, ymax*0.95, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.7)
for i, ((vi, (lam, flu)), c) in enumerate(zip(data.items(), cmap)):
    tm = trough_min(lam, flu)
    if tm:
        v_obs = (1-tm/6355.0)*C_KMS
        ax.axvline(tm, color=c, alpha=0.5, lw=1)
        ax.text(tm, ymax*(0.20 + i*0.10), f'v_in={vi}\nλ={tm:.0f}\nv={v_obs:.0f}',
                color=c, fontsize=7, ha='center', va='bottom', alpha=0.9, fontweight='bold')
tm = trough_min(hlam, hflu)
if tm:
    v_obs = (1-tm/6355.0)*C_KMS
    ax.axvline(tm, color='k', alpha=0.7, lw=1.5)
    ax.text(tm, ymax*0.05, f'HST {tm:.0f}\nv={v_obs:.0f}', color='k', fontsize=9,
            ha='center', va='bottom', fontweight='bold')
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Si II 6355 zoom — DDC15 v_in vs HST 9934 km/s')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

ax = axes[2]
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
for (vi, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.3, alpha=0.9, label=f'DDC15 v_in={vi}')
if pa_champ is not None:
    ax.plot(pa_champ[0], pa_champ[1]*1e14, color='tab:green', lw=1.0, alpha=0.55, ls='--',
            label='Path A champ')
ax.set_xlim(2300, 4000)
m = (hlam>=2300)&(hlam<=4000)
ymax_uv = max(hflu[m].max(),
              *[d[1][(d[0]>=2300)&(d[0]<=4000)].max() for d in data.values()]) * 1e14 * 1.15
ax.set_ylim(0, ymax_uv)
for lam_marker, lab in [(2382, 'Fe II 2382'), (2600, 'Fe II 2600'),
                         (2796, 'Mg II 2796'), (3934, 'Ca II K')]:
    ax.axvline(lam_marker, color='gray', ls=':', alpha=0.4, lw=0.7)
    ax.text(lam_marker, ymax_uv*0.92, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.5)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('UV blanketing 2300-4000Å — DDC15 vs Path A')
ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

print(f"\n=== Si II 6355 trough position (A4 DDC15) ===")
tm = trough_min(hlam, hflu)
print(f"{'HST 2011fe B-max':40s}  λ={tm:.1f}  v={(1-tm/6355)*C_KMS:.0f} km/s")
for vi, (lam, flu) in data.items():
    tm = trough_min(lam, flu)
    if tm:
        v = (1-tm/6355)*C_KMS
        print(f"{'A4 DDC15 v_in='+str(vi):40s}  λ={tm:.1f}  v={v:.0f} km/s  (HST diff {v-9934:+.0f})")

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

print(f"\n=== Baseline-norm RMS [3000,8000]Å (target ≤ 0.20) ===")
hlam_n, hflu_n = gauss_baseline(hlam, hflu)
from scipy.interpolate import interp1d
for vi, (lam, flu) in data.items():
    mlam, mflu = gauss_baseline(lam, flu)
    if mlam is None: continue
    fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
    mflu_h = fm(hlam_n)
    diff = mflu_h - hflu_n
    rms = np.sqrt(np.nanmean(diff**2))
    print(f"A4 DDC15 v_in={vi}                       RMS_bn = {rms:.4f}")

if pa_champ is not None:
    mlam, mflu = gauss_baseline(pa_champ[0], pa_champ[1])
    if mlam is not None:
        fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
        mflu_h = fm(hlam_n)
        rms = np.sqrt(np.nanmean((mflu_h - hflu_n)**2))
        print(f"Path A fullNLTE v10400 (champion)         RMS_bn = {rms:.4f}")
if pa_skipsi is not None:
    mlam, mflu = gauss_baseline(pa_skipsi[0], pa_skipsi[1])
    if mlam is not None:
        fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
        mflu_h = fm(hlam_n)
        rms = np.sqrt(np.nanmean((mflu_h - hflu_n)**2))
        print(f"Path A SkipSi v10400                      RMS_bn = {rms:.4f}")
