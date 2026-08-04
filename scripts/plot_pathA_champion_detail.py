#!/usr/bin/env python3
"""Path A champion (ρ-11fe full-NLTE v_in=10400, job 155702) — detailed vs HST 2011fe B-max.

Single-model deep-dive: 6 zoom panels covering all major SN Ia diagnostic lines.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
MODEL = ROOT/"logs/rho11feA_155702_rho11fe_fullNLTE/lumina_spectrum_formal.csv"
OUT  = ROOT/"figures/pathA_champion_155702_detail.png"
C_KMS = 299792.458

def load_csv(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo=4500, hi=5800):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def trough_min(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    if m.sum()<5: return None, None
    i = np.argmin(flu[m])
    return float(lam[m][i]), float(flu[m][i])

hlam, hflu = load_csv(HST)
mlam, mflu = load_csv(MODEL)
mflu = mflu * (band_int(hlam, hflu) / band_int(mlam, mflu))

# 6 panels
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.18, top=0.95, bottom=0.05,
                       left=0.06, right=0.97)

PANELS = [
    # (row, col, xlo, xhi, title, lines)
    (0, 0, 1700, 9500, "Full spectrum 1700-9500Å",
     [(2796, 'Mg II'), (3934, 'Ca II K'), (3969, 'Ca II H'),
      (4471, 'He I'), (5169, 'Fe II'), (6355, 'Si II'), (8542, 'Ca II IR')]),
    (0, 1, 2300, 4000, "UV blanketing 2300-4000Å",
     [(2382, 'Fe II 2382'), (2600, 'Fe II 2600'), (2796, 'Mg II 2796'),
      (3242, 'Ti II 3242'), (3934, 'Ca II K')]),
    (1, 0, 4000, 5400, "Fe II/Mg II 4000-5400Å",
     [(4233, 'Fe II 4233'), (4549, 'Fe II 4549'), (4924, 'Fe II 4924'),
      (5018, 'Fe II 5018'), (5169, 'Fe II 5169')]),
    (1, 1, 5400, 6800, "S II / Si II 5400-6800Å",
     [(5454, 'S II W'), (5640, 'S II W'), (5972, 'Si II 5972'), (6355, 'Si II 6355')]),
    (2, 0, 6800, 8000, "O I / continuum 6800-8000Å",
     [(7065, 'He I'), (7773, 'O I 7773')]),
    (2, 1, 8000, 9500, "Ca II IR triplet 8000-9500Å",
     [(8498, 'Ca II IR'), (8542, 'Ca II IR'), (8662, 'Ca II IR')]),
]

for (r, c, xlo, xhi, title, lines) in PANELS:
    ax = fig.add_subplot(gs[r, c])
    sel_h = (hlam>=xlo)&(hlam<=xhi)
    sel_m = (mlam>=xlo)&(mlam<=xhi)
    if sel_h.sum() < 2 or sel_m.sum() < 2: continue
    ymax = max(hflu[sel_h].max(), mflu[sel_m].max()) * 1e14 * 1.1

    ax.plot(hlam[sel_h], hflu[sel_h]*1e14, 'k-', lw=1.8, label='HST 2011fe B-max', alpha=0.95)
    ax.plot(mlam[sel_m], mflu[sel_m]*1e14, color='tab:green', lw=1.3,
            label='LUMINA ρ-11fe fullNLTE v10400', alpha=0.9)

    # Velocity markers for Si II in main panel
    if 6355 in [l[0] for l in lines]:
        for v_kms, ls, lab in [(0, ':', 'rest'), (9934, '-.', 'HST 9934'),
                                (10080, ':', 'P+13 10080')]:
            lo_lam = 6355*(1-v_kms/C_KMS)
            if xlo <= lo_lam <= xhi:
                ax.axvline(lo_lam, color='gray', ls=ls, alpha=0.4, lw=0.7)

    # Annotate lines
    for lam_marker, lab in lines:
        if xlo <= lam_marker <= xhi:
            ax.axvline(lam_marker, color='red', ls=':', alpha=0.3, lw=0.6)
            ax.text(lam_marker, ymax*0.95, lab, rotation=90, fontsize=8,
                    va='top', ha='right', alpha=0.6)

    # Si II 6355 trough mark for both HST and model
    if 6355 in [l[0] for l in lines]:
        for lam_arr, flu_arr, color, label in [(hlam, hflu, 'k', 'HST'),
                                                  (mlam, mflu, 'tab:green', 'LUMINA')]:
            tm, fm_v = trough_min(lam_arr, flu_arr, 5800, 6300)
            if tm and xlo <= tm <= xhi:
                v = (1 - tm/6355)*C_KMS
                ax.axvline(tm, color=color, alpha=0.6, lw=1.2)
                ax.text(tm, ymax*(0.45 if color=='k' else 0.30),
                        f'{label}\n{tm:.0f}\nv={v:.0f}',
                        color=color, fontsize=8, ha='center', va='bottom',
                        fontweight='bold', alpha=0.95)

    ax.set_xlim(xlo, xhi); ax.set_ylim(0, ymax)
    ax.set_xlabel('λ (Å)', fontsize=9); ax.set_ylabel('F_λ × 10¹⁴', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    if r == 0 and c == 0:
        ax.legend(loc='upper right', fontsize=9)

# Baseline-norm residual at bottom would be useful
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

from scipy.interpolate import interp1d
hlam_n, hflu_n = gauss_baseline(hlam, hflu)
mlam_n, mflu_n = gauss_baseline(mlam, mflu)
fm_b = interp1d(mlam_n, mflu_n, kind='linear', bounds_error=False, fill_value=np.nan)
mflu_h_n = fm_b(hlam_n)
diff = mflu_h_n - hflu_n
rms = np.sqrt(np.nanmean(diff**2))

# Si II trough metrics
tm_h, _ = trough_min(hlam, hflu, 5800, 6300)
tm_m, _ = trough_min(mlam, mflu, 5800, 6300)
v_h = (1-tm_h/6355)*C_KMS if tm_h else 0
v_m = (1-tm_m/6355)*C_KMS if tm_m else 0

fig.suptitle(f'Path A Champion: ρ-11fe fullNLTE v_in=10400 (job 155702) vs HST 2011fe B-max\n'
             f'RMS_bn[3000,8000]={rms:.4f} (target ≤0.20)  |  '
             f'Si II 6355: HST v={v_h:.0f}, LUMINA v={v_m:.0f} (Δ={v_m-v_h:+.0f} km/s)',
             fontsize=12, y=0.99)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=110, bbox_inches='tight')
print(f"Wrote {OUT}")
print(f"RMS_bn[3000,8000] = {rms:.4f}")
print(f"Si II 6355: HST λ={tm_h:.1f} v={v_h:.0f}, LUMINA λ={tm_m:.1f} v={v_m:.0f}")
print(f"  diff: v_LUMINA - v_HST = {v_m-v_h:+.0f} km/s")
