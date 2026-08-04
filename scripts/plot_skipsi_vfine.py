#!/usr/bin/env python3
"""SkipSi v_in fine sweep (155687) — Si II 6355 trough velocity scan."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"

CELLS = [
    (9000,  ROOT/"logs/skipSi_vfine_155687_skipSi_rp0p00_n9p0_v09000/lumina_spectrum_formal.csv"),
    (9500,  ROOT/"logs/skipSi_vfine_155687_skipSi_rp0p00_n9p0_v09500/lumina_spectrum_formal.csv"),
    (10000, ROOT/"logs/skipSi_vfine_155687_skipSi_rp0p00_n9p0_v10000/lumina_spectrum_formal.csv"),
    (10400, ROOT/"logs/skipSi_phys_155590_skipSi_rp0p00_n9p0_v10400/lumina_spectrum_formal.csv"),
    (12500, ROOT/"logs/skipSi_phys_155590_skipSi_rp0p00_n9p0_v12500/lumina_spectrum_formal.csv"),
]
OUT = ROOT/"figures/skipsi_vfine_155687_vs_hst.png"
C_KMS = 299792.458

def load_csv(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo=4500, hi=5800):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

hlam, hflu = load_csv(HST)
hi_grn = band_int(hlam, hflu)

def load_and_scale(p):
    lam, flu = load_csv(p)
    li = band_int(lam, flu)
    return lam, flu * (hi_grn/li)

def trough_min(lam, flu, lo=5800, hi=6300):
    m = (lam>=lo)&(lam<=hi)
    if m.sum()<5: return None
    i = np.argmin(flu[m]); return float(lam[m][i])

data = {v: load_and_scale(p) for v,p in CELLS}

fig, axes = plt.subplots(2, 1, figsize=(15, 9))

# Full spectrum
ax = axes[0]
ax.plot(hlam, hflu*1e14, 'k-', lw=1.2, label='HST 2011fe B-max', alpha=0.95)
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(data)))
for (vi, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=0.9, alpha=0.85, label=f'+SkipSi v_in={vi}')
ax.set_xlim(1700, 9000); ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('SkipSi v_in fine sweep (155687 + 155590) — full spectrum')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

# Si II 6355 zoom
ax = axes[1]
m = (hlam>=5500)&(hlam<=6800)
ymax = max(hflu[m].max(), *[d[1][(d[0]>=5500)&(d[0]<=6800)].max() for d in data.values()]) * 1e14 * 1.1
ax.plot(hlam, hflu*1e14, 'k-', lw=2.5, label='HST', alpha=0.95)
for (vi, (lam, flu)), c in zip(data.items(), cmap):
    ax.plot(lam, flu*1e14, color=c, lw=1.5, alpha=0.9, label=f'v_in={vi}')
ax.set_xlim(5500, 6800); ax.set_ylim(0, ymax)
for v_kms, ls, lab in [(0, ':', 'Si II rest'), (9611, '--', 'ML 9611 km/s'),
                        (9934, '-.', 'HST 9934 km/s')]:
    lo = 6355*(1-v_kms/C_KMS)
    ax.axvline(lo, color='gray', ls=ls, alpha=0.5, lw=0.8)
    ax.text(lo, ymax*0.95, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.7)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Si II 6355 zoom')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

print("\n=== Si II 6355 trough position ===")
tm = trough_min(hlam, hflu); print(f"{'HST':16s}  λ={tm:.1f}  v={(1-tm/6355)*C_KMS:.0f} km/s")
for vi, (lam, flu) in data.items():
    tm = trough_min(lam, flu)
    if tm:
        v_obs = (1-tm/6355)*C_KMS
        delta = v_obs - 9934
        print(f"v_in={vi:5d}      λ={tm:.1f}  v={v_obs:.0f} km/s  (HST diff {delta:+.0f})")
