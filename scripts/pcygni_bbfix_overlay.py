#!/usr/bin/env python3
"""#298 bbfix vs old-asE P-Cygni 6-panel trough overlay vs SN 2002bo Bmax.

Overlays:
  - bbfix prod 158558  (Fe III bb +7×, Co III bb +162× post-#298 patch)
  - old asE 157921     (#298-tainted base; Fe III 14% / Co III 0.6% bb)
  - SN 2002bo m0d0     (rest-frame, CCM-dereddened)

Both models pinned to obs via integrated flux over Blondin F_scl [4000,6000] Å.
Six panels: Ca H&K, Si II 6355, O I 7774, Ni II 7378, Mg II 9218, Ca II IR triplet.

Per-panel: model/obs ratio + min-position blueshift v (km/s) printed.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
BBFIX = ROOT / "logs/paperDDC15v3asEfloor_bbfix_2002bo_vi9019_L1p0_prod_158558/lumina_spectrum_formal.csv"
ASE   = ROOT / "logs/paperDDC15v3asEfloorEPS0p0_2002bo_vi9019_L1p0_prod_157921/lumina_spectrum_formal.csv"
OBS   = ROOT / "data/sn2002bo/epochs/sn2002bo_m0d0.csv"

EBV, RV, Z_HEL = 0.41, 3.1, 0.0042
C_KMS = 2.998e5

def ccm_a_over_av(wave_aa):
    x = 1e4 / wave_aa
    a = np.zeros_like(x); b = np.zeros_like(x)
    sel = (x >= 1.1) & (x <= 3.3)
    y = x[sel] - 1.82
    a[sel] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[sel] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    sel = (x >= 0.3) & (x < 1.1)
    a[sel] =  0.574 * x[sel]**1.61
    b[sel] = -0.527 * x[sel]**1.61
    sel = (x > 3.3) & (x <= 8.0)
    xs = x[sel]
    Fa = np.where(xs >= 5.9, -0.04473*(xs-5.9)**2 - 0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs >= 5.9,  0.2130*(xs-5.9)**2 + 0.1207*(xs-5.9)**3, 0.0)
    a[sel] =  1.752 - 0.316*xs - 0.104/((xs - 4.67)**2 + 0.341) + Fa
    b[sel] = -3.090 + 1.825*xs + 1.206/((xs - 4.62)**2 + 0.263) + Fb
    return a + b / RV

A_V = RV * EBV
deredden = lambda w, f: f * 10**(0.4 * A_V * ccm_a_over_av(w))

# --- Load + deredden + de-redshift observed ---
obs = pd.read_csv(OBS, comment='#')
olam = obs['wavelength_angstrom'].values / (1.0 + Z_HEL)
oflu = deredden(obs['wavelength_angstrom'].values, obs['flux_erg_s_cm2_angstrom'].values) * (1.0 + Z_HEL)

# --- Load both models (rest-frame λ) ---
m_bb = pd.read_csv(BBFIX); mlam_bb = m_bb['wavelength_angstrom'].values; mflu_bb_raw = m_bb['flux'].values
m_ae = pd.read_csv(ASE);   mlam_ae = m_ae['wavelength_angstrom'].values; mflu_ae_raw = m_ae['flux'].values

# --- Anchor each model on [4000,6000] (Blondin F_scl band) ---
ANCHOR_LO, ANCHOR_HI = 4000., 6000.
def anchor(mlam, mflu_raw):
    sel_m = (mlam >= ANCHOR_LO) & (mlam <= ANCHOR_HI)
    sel_o = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
    K = float(np.trapezoid(oflu[sel_o], olam[sel_o])) / float(np.trapezoid(mflu_raw[sel_m], mlam[sel_m]))
    return mflu_raw * K, K

mflu_bb, K_bb = anchor(mlam_bb, mflu_bb_raw)
mflu_ae, K_ae = anchor(mlam_ae, mflu_ae_raw)
print(f"K_bbfix = {K_bb:.3e}    K_asE = {K_ae:.3e}    ratio = {K_bb/K_ae:.4f}")

# --- P-Cygni 6 panels: (label, λ_lab Å, win_lo Å, win_hi Å, trough_lo Å, trough_hi Å) ---
PANELS = [
    ("Ca II H&K",   3945, 3500, 4100, 3700, 3900),
    ("Si II 6355",  6355, 5950, 6500, 6050, 6300),
    ("O I 7774",    7774, 7300, 8000, 7400, 7750),
    ("Ni II 7378",  7378, 7100, 7600, 7200, 7400),
    ("Mg II 9218",  9218, 8850, 9400, 8950, 9200),
    ("Ca II IR trip", 8542, 7900, 8800, 8000, 8500),
]

def trough_min(lam, flu, lo, hi, lam_lab):
    """Return (lam_min, depth, v_blueshift_km_s) over [lo, hi]."""
    sel = (lam >= lo) & (lam <= hi)
    if sel.sum() < 5:
        return np.nan, np.nan, np.nan
    L = lam[sel]; F = flu[sel]
    i = int(np.argmin(F))
    v = (lam_lab - L[i]) / lam_lab * C_KMS
    return float(L[i]), float(F[i]), float(v)

def band_ratio(mlam, mflu, lo, hi):
    sel_o = (olam >= lo) & (olam <= hi)
    mi = np.interp(olam[sel_o], mlam, mflu)
    F_o = float(np.trapezoid(oflu[sel_o], olam[sel_o]))
    F_m = float(np.trapezoid(mi,         olam[sel_o]))
    return F_m / F_o if F_o > 0 else np.nan

# --- Figure ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flat

print("\n=== Per-panel trough metrics (window-min) ===")
print(f"  {'feature':14s}  {'src':6s}  {'λ_min':>7s}  {'v (km/s)':>9s}  {'F/F_obs':>8s}")
for ax, (label, lam_lab, lo, hi, trough_lo, trough_hi) in zip(axes, PANELS):
    # plot all three (observed + bbfix + old asE)
    sel_o = (olam >= lo) & (olam <= hi)
    sel_bb = (mlam_bb >= lo) & (mlam_bb <= hi)
    sel_ae = (mlam_ae >= lo) & (mlam_ae <= hi)
    ax.plot(olam[sel_o], oflu[sel_o], lw=1.4, color='black', alpha=0.9, label='SN 2002bo Bmax')
    ax.plot(mlam_ae[sel_ae], mflu_ae[sel_ae], lw=1.2, color='steelblue', alpha=0.85,
            label='old asE (#298-tainted, 157921)')
    ax.plot(mlam_bb[sel_bb], mflu_bb[sel_bb], lw=1.4, color='crimson', alpha=0.9,
            label='bbfix prod (158558)')
    # blueshift markers at v=11000 km/s photospheric
    v_phot = 11000.0
    lam_blue = lam_lab * (1 - v_phot/C_KMS)
    ax.axvline(lam_lab, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.axvline(lam_blue, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.axvspan(trough_lo, trough_hi, color='yellow', alpha=0.08)
    ax.set_xlim(lo, hi)
    ax.set_xlabel('λ_rest [Å]'); ax.set_ylabel('F_λ [erg/s/cm²/Å]')
    ax.set_title(f'{label}  (λ_lab={lam_lab}Å)', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.25)
    # metrics over trough window
    for src, lam, flu in [('obs', olam, oflu), ('asE', mlam_ae, mflu_ae), ('bbfix', mlam_bb, mflu_bb)]:
        lmin, fmin, v = trough_min(lam, flu, trough_lo, trough_hi, lam_lab)
        print(f"  {label:14s}  {src:6s}  {lmin:7.1f}  {v:9.0f}  {band_ratio(lam, flu, trough_lo, trough_hi) if src!='obs' else 1.0:8.3f}")

plt.tight_layout()
DATE = "2026-05-28"
out = ROOT / f"figures/{DATE}_pcygni_bbfix_vs_ase_vs_sn2002bo.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nsaved: {out}")

# --- Per-panel band-int ratios (trough window) ---
print("\n=== Per-feature band-int ratio (trough window) — >1 = excess, <1 = deficit ===")
print(f"  {'feature':14s}  {'asE/obs':>9s}  {'bbfix/obs':>11s}  {'Δ (bb-asE)':>11s}")
for label, lam_lab, lo, hi, trough_lo, trough_hi in PANELS:
    r_ae = band_ratio(mlam_ae, mflu_ae, trough_lo, trough_hi)
    r_bb = band_ratio(mlam_bb, mflu_bb, trough_lo, trough_hi)
    arrow = "  ←DEEPER ✓" if (r_bb < r_ae and r_ae > 1) else ("  ←SHALLOWER ✗" if (r_bb > r_ae and r_ae < 1) else "")
    print(f"  {label:14s}  {r_ae:9.3f}  {r_bb:11.3f}  {r_bb - r_ae:+11.3f}{arrow}")
