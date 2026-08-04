#!/usr/bin/env python3
"""v_in sweep (155583) vs HST + physical-ref champion (155524 cell v12500).
Uses green-band [4500,5800] integral matching for LUMINA→HST flux scale."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT/"logs/struct3d_phys_155524_rp0p00_n9p0_v12500/lumina_spectrum_formal.csv"
VINS = {
    12500: ROOT/"logs/outerFevin_155583_outerFe_rp0p00_n9p0_v12500/lumina_spectrum_formal.csv",
    15500: ROOT/"logs/outerFevin_155583_outerFe_rp0p00_n9p0_v15500/lumina_spectrum_formal.csv",
    17500: ROOT/"logs/outerFevin_155583_outerFe_rp0p00_n9p0_v17500/lumina_spectrum_formal.csv",
}
OUT = ROOT/"figures/vin_sweep_155583_vs_hst.png"
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
print(f"HST green-band integral: {hi_grn:.3e}")

def load_and_scale(p):
    lam, flu = load_csv(p)
    li = band_int(lam, flu)
    return lam, flu * (hi_grn/li)

clam, cflu = load_and_scale(CHAMP)
vin_data = {v: load_and_scale(p) for v, p in VINS.items()}

# Plot
fig, axes = plt.subplots(2, 1, figsize=(15, 9))

ax = axes[0]
ax.plot(hlam, hflu*1e14, 'k-', lw=1.0, label='HST 2011fe B-max', alpha=0.9)
ax.plot(clam, cflu*1e14, 'b-', lw=0.9, alpha=0.7, label='champion physical v_in=10400')
colors = ['tab:orange', 'tab:red', 'tab:purple']
for (v, (lam, flu)), c in zip(vin_data.items(), colors):
    ax.plot(lam, flu*1e14, color=c, lw=0.9, alpha=0.8, label=f'v_in={v} (outerFe)')
ax.set_xlim(1700, 9000); ax.set_ylim(0, max(hflu[(hlam>2900)&(hlam<9000)])*1e14*1.15)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴  [erg/s/cm²/Å]')
ax.set_title('v_in sweep (155583, outerFe ref ρ=0 n=9) vs HST + physical champion')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

ax = axes[1]
m = (hlam>=5500)&(hlam<=6800)
ymax = max(hflu[m].max(), cflu[(clam>=5500)&(clam<=6800)].max(),
           *[d[1][(d[0]>=5500)&(d[0]<=6800)].max() for d in vin_data.values()]) * 1e14 * 1.1
ax.plot(hlam, hflu*1e14, 'k-', lw=2.0, label='HST', alpha=0.95)
ax.plot(clam, cflu*1e14, 'b-', lw=1.6, alpha=0.85, label='champion v_in=10400')
for (v, (lam, flu)), c in zip(vin_data.items(), colors):
    ax.plot(lam, flu*1e14, color=c, lw=1.6, alpha=0.9, label=f'v_in={v}')
ax.set_xlim(5500, 6800); ax.set_ylim(0, ymax)
for v_kms, ls, lab in [(0, ':', 'Si II rest'), (9611, '--', 'Si II 9611 km/s (ML)'),
                        (12500, '-.', 'Si II 12500 km/s')]:
    lo = 6355*(1-v_kms/C_KMS)
    ax.axvline(lo, color='gray', ls=ls, alpha=0.5, lw=0.8)
    ax.text(lo, ymax*0.95, lab, rotation=90, fontsize=7, va='top', ha='right', alpha=0.7)
ax.set_xlabel('λ (Å)'); ax.set_ylabel('F_λ × 10¹⁴')
ax.set_title('Si II 6355 zoom — empirical v_form=9611 km/s vs models')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.2)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110)
print(f"Wrote {OUT}")

def trough_min(lam, flu, lo=5800, hi=6300):
    m = (lam>=lo)&(lam<=hi)
    if m.sum()<5: return None, None
    i = np.argmin(flu[m]); return float(lam[m][i]), float(flu[m][i])

print("\nSi II 6355 trough position:")
for name, (lam, flu) in [('HST', (hlam, hflu)), ('champion v=10400', (clam, cflu)),
                          *[(f'outerFe v_in={v}', d) for v,d in vin_data.items()]]:
    lo, _ = trough_min(lam, flu)
    if lo:
        v_obs = (1 - lo/6355.0)*C_KMS
        print(f"  {name:22s}: λ_min={lo:.1f} Å  →  v_obs={v_obs:.0f} km/s")
