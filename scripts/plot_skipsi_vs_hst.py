#!/usr/bin/env python3
"""SkipSi probe (155590) vs HST + baseline (155524) — Si II 6355 trough fix?"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"

BASE = {
    10400: ROOT/"logs/struct3d_phys_155524_rp0p00_n9p0_v10400/lumina_spectrum_formal.csv",
    12500: ROOT/"logs/struct3d_phys_155524_rp0p00_n9p0_v12500/lumina_spectrum_formal.csv",
}
SKIP = {
    10400: ROOT/"logs/skipSi_phys_155590_skipSi_rp0p00_n9p0_v10400/lumina_spectrum_formal.csv",
    12500: ROOT/"logs/skipSi_phys_155590_skipSi_rp0p00_n9p0_v12500/lumina_spectrum_formal.csv",
}
OUT = ROOT/"figures/skipsi_155590_vs_hst.png"
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

bdata = {v: load_and_scale(p) for v,p in BASE.items()}
sdata = {v: load_and_scale(p) for v,p in SKIP.items()}

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

for col, vi in enumerate([10400, 12500]):
    # Top: full spectrum
    ax = axes[0, col]
    blam, bflu = bdata[vi]; slam, sflu = sdata[vi]
    ax.plot(hlam, hflu*1e14, 'k-', lw=1.0, label='HST', alpha=0.9)
    ax.plot(blam, bflu*1e14, 'b-', lw=0.9, alpha=0.75, label=f'baseline v_in={vi}')
    ax.plot(slam, sflu*1e14, 'r-', lw=0.9, alpha=0.85, label=f'+SkipSi v_in={vi}')
    ax.set_xlim(1700, 9000); ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
    ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
    ax.set_title(f'v_in={vi} km/s — full spectrum')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

    # Bottom: Si II 6355 zoom
    ax = axes[1, col]
    m = (hlam>=5500)&(hlam<=6800)
    ymax = max(hflu[m].max(), bflu[(blam>=5500)&(blam<=6800)].max(),
               sflu[(slam>=5500)&(slam<=6800)].max()) * 1e14 * 1.1
    ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
    ax.plot(blam, bflu*1e14, 'b-', lw=1.5, alpha=0.85, label=f'baseline')
    ax.plot(slam, sflu*1e14, 'r-', lw=1.5, alpha=0.9, label=f'+SkipSi')
    ax.set_xlim(5500, 6800); ax.set_ylim(0, ymax)
    for v_kms, ls, lab in [(0, ':', 'rest'), (9611, '--', 'ML 9611 km/s'),
                            (9934, '-.', 'HST 9934 km/s')]:
        lo = 6355*(1-v_kms/C_KMS)
        ax.axvline(lo, color='gray', ls=ls, alpha=0.5, lw=0.8)
        ax.text(lo, ymax*0.95, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.7)
    # mark each trough min
    for (lam, flu, c, ls, lbl) in [(hlam, hflu, 'k', 'k.', 'HST'),
                                     (blam, bflu, 'b', 'b.', 'base'),
                                     (slam, sflu, 'r', 'r.', 'skip')]:
        tm = trough_min(lam, flu)
        if tm is not None:
            v_obs = (1 - tm/6355.0)*C_KMS
            ax.axvline(tm, color=c, alpha=0.4, lw=0.5)
            ax.text(tm, ymax*0.05+(0.10 if lbl=='HST' else 0.05 if lbl=='base' else 0),
                    f'{lbl} {tm:.0f}Å v={v_obs:.0f}', color=c, fontsize=7,
                    ha='center', va='bottom', alpha=0.85)
    ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
    ax.set_title(f'Si II 6355 zoom — v_in={vi}')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

print("\n=== Si II 6355 trough position ===")
print(f"{'cell':28s}  λ_min (Å)   v (km/s)")
tm = trough_min(hlam, hflu); print(f"{'HST 2011fe B-max':28s}  {tm:.1f}     {(1-tm/6355)*C_KMS:.0f}")
for vi in [10400, 12500]:
    for label, data in [(f'baseline v_in={vi}', bdata[vi]), (f'+SkipSi v_in={vi}', sdata[vi])]:
        tm = trough_min(*data)
        if tm:
            print(f"{label:28s}  {tm:.1f}     {(1-tm/6355)*C_KMS:.0f}")
