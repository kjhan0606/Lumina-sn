#!/usr/bin/env python3
"""Path (A) ρ-11fe composition probe (155702) vs HST + W7-strat6 SkipSi baseline."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
JOB  = sys.argv[1] if len(sys.argv) > 1 else "155702"

CELLS = [
    ("W7-strat6 +SkipSi v_in=10400 (155590)",
     ROOT/"logs/skipSi_phys_155590_skipSi_rp0p00_n9p0_v10400/lumina_spectrum_formal.csv",
     "tab:blue"),
    (f"ρ-11fe +SkipSi v_in=10400 ({JOB})",
     ROOT/f"logs/rho11feA_{JOB}_rho11fe_skipSi/lumina_spectrum_formal.csv",
     "tab:red"),
    (f"ρ-11fe full-NLTE v_in=10400 ({JOB})",
     ROOT/f"logs/rho11feA_{JOB}_rho11fe_fullNLTE/lumina_spectrum_formal.csv",
     "tab:green"),
]
OUT = ROOT/f"figures/rho11fe_pathA_{JOB}_vs_hst.png"
C_KMS = 299792.458

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
    return lam, flu * (hi_grn / li)

data = {}
for label, path, color in CELLS:
    d = load_and_scale(path)
    if d is None:
        print(f"  MISSING: {path}")
        continue
    data[label] = (d, color)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Panel 1: full spectrum
ax = axes[0]
ax.plot(hlam, hflu*1e14, 'k-', lw=1.2, label='HST 2011fe B-max', alpha=0.95)
for label, ((lam, flu), color) in data.items():
    ax.plot(lam, flu*1e14, color=color, lw=0.9, alpha=0.85, label=label)
ax.set_xlim(1700, 9000)
ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title(f'Path (A) ρ-11fe composition probe (job {JOB}) — full spectrum')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

# Panel 2: Si II 6355 zoom
ax = axes[1]
m = (hlam>=5500)&(hlam<=6800)
maxima = [hflu[m].max()]
for label, ((lam, flu), color) in data.items():
    msel = (lam>=5500)&(lam<=6800)
    if msel.sum() > 0: maxima.append(flu[msel].max())
ymax = max(maxima) * 1e14 * 1.1
ax.plot(hlam, hflu*1e14, 'k-', lw=2.5, label='HST', alpha=0.95)
for label, ((lam, flu), color) in data.items():
    ax.plot(lam, flu*1e14, color=color, lw=1.5, alpha=0.9, label=label.split(' +')[0]+' '+label.split(' ')[-1])
ax.set_xlim(5500, 6800); ax.set_ylim(0, ymax)
for v_kms, ls, lab in [(0, ':', 'Si II rest'), (9611, '--', 'ML 9611'),
                        (9934, '-.', 'HST 9934'), (10080, ':', 'Pereira 10080')]:
    lo = 6355*(1-v_kms/C_KMS)
    ax.axvline(lo, color='gray', ls=ls, alpha=0.5, lw=0.8)
    ax.text(lo, ymax*0.95, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.7)
# mark trough mins
for label, ((lam, flu), color) in data.items():
    tm = trough_min(lam, flu)
    if tm:
        v_obs = (1 - tm/6355.0)*C_KMS
        ax.axvline(tm, color=color, alpha=0.5, lw=1)
        ax.text(tm, ymax*0.55, f'{tm:.0f}Å\nv={v_obs:.0f}', color=color, fontsize=8,
                ha='center', va='bottom', alpha=0.85, fontweight='bold')
tm = trough_min(hlam, hflu)
if tm:
    v_obs = (1 - tm/6355.0)*C_KMS
    ax.axvline(tm, color='k', alpha=0.6, lw=1.2)
    ax.text(tm, ymax*0.35, f'HST {tm:.0f}Å\nv={v_obs:.0f}', color='k', fontsize=9,
            ha='center', va='bottom', fontweight='bold')
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Si II 6355 zoom — trough position vs HST')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

# Panel 3: UV blanketing zone
ax = axes[2]
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
for label, ((lam, flu), color) in data.items():
    ax.plot(lam, flu*1e14, color=color, lw=1.3, alpha=0.9, label=label.split(' +')[0]+' '+label.split(' ')[-1])
ax.set_xlim(2300, 4000)
m = (hlam>=2300)&(hlam<=4000)
ymax_uv = max(hflu[m].max(),
              *[d[0][1][(d[0][0]>=2300)&(d[0][0]<=4000)].max() for d in data.values()]) * 1e14 * 1.15
ax.set_ylim(0, ymax_uv)
for lam_marker, lab in [(2382, 'Fe II 2382'), (2600, 'Fe II 2600'),
                         (2796, 'Mg II 2796'), (3934, 'Ca II K')]:
    ax.axvline(lam_marker, color='gray', ls=':', alpha=0.4, lw=0.7)
    ax.text(lam_marker, ymax_uv*0.92, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.5)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('UV blanketing 2300-4000Å (composition sensitive)')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

# Trough table
print(f"\n=== Si II 6355 trough position ===")
tm = trough_min(hlam, hflu)
print(f"{'HST 2011fe B-max':45s}  λ={tm:.1f}  v={(1-tm/6355)*C_KMS:.0f} km/s")
for label, ((lam, flu), color) in data.items():
    tm = trough_min(lam, flu)
    if tm:
        v = (1-tm/6355)*C_KMS
        print(f"{label:45s}  λ={tm:.1f}  v={v:.0f} km/s  (HST diff {v-9934:+.0f})")

# Baseline-norm RMS [3000,8000]
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
for label, ((lam, flu), color) in data.items():
    mlam, mflu = gauss_baseline(lam, flu)
    if mlam is None: continue
    # interp model to HST grid
    from scipy.interpolate import interp1d
    fm = interp1d(mlam, mflu, kind='linear', bounds_error=False, fill_value=np.nan)
    mflu_h = fm(hlam_n)
    diff = mflu_h - hflu_n
    rms = np.sqrt(np.nanmean(diff**2))
    print(f"{label:45s}  RMS_bn = {rms:.4f}")
