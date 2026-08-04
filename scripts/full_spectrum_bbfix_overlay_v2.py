#!/usr/bin/env python3
"""#298 bbfix overlay v2 — emphasize P-Cygni structure with mid-range zoom.

Three panels, each tuned to make line structure visible:
  (1) [3500, 6000] Å — blue side (Ca H&K, Fe II m42, Mg II 4481, Fe II 4924/5018/5169)
  (2) [5800, 7500] Å — mid (Si II 5972/6355, [Fe II]7155, Ni II 7378)
  (3) [7300, 10000] Å — red (O I 7774, Ca II IR triplet, Mg II 9218)

Uses RAW formal integration (no smoothing). All three panels anchored on [4000,6000].
"""
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

obs = pd.read_csv(OBS, comment='#')
olam = obs['wavelength_angstrom'].values / (1.0 + Z_HEL)
oflu = deredden(obs['wavelength_angstrom'].values, obs['flux_erg_s_cm2_angstrom'].values) * (1.0 + Z_HEL)

m_bb = pd.read_csv(BBFIX); mlam_bb = m_bb['wavelength_angstrom'].values; mflu_bb_raw = m_bb['flux'].values
m_ae = pd.read_csv(ASE);   mlam_ae = m_ae['wavelength_angstrom'].values; mflu_ae_raw = m_ae['flux'].values

ANCHOR_LO, ANCHOR_HI = 4000., 6000.
def anchor(mlam, mflu_raw):
    sel_m = (mlam >= ANCHOR_LO) & (mlam <= ANCHOR_HI)
    sel_o = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
    K = float(np.trapezoid(oflu[sel_o], olam[sel_o])) / float(np.trapezoid(mflu_raw[sel_m], mlam[sel_m]))
    return mflu_raw * K, K

mflu_bb, _ = anchor(mlam_bb, mflu_bb_raw)
mflu_ae, _ = anchor(mlam_ae, mflu_ae_raw)

LINES = [
    (3934, "Ca II H"),  (3968, "Ca II K"),
    (4128, "Si II"),    (4481, "Mg II"),
    (4555, "Fe II"),    (4924, "Fe II"),
    (5018, "Fe II"),    (5169, "Fe II"),
    (5454, "S II W"),   (5640, "S II W"),
    (5972, "Si II"),    (6355, "Si II"),
    (7155, "[Fe II]"),  (7378, "Ni II"),
    (7774, "O I"),
    (8498, "Ca II IR"), (8542, "Ca II IR"), (8662, "Ca II IR"),
    (9218, "Mg II"),    (9244, "Mg II"),
]

PANELS = [
    ("Blue [3500-6000] Å", 3500, 6000),
    ("Mid [5800-7500] Å",  5800, 7500),
    ("Red [7300-10000] Å", 7300, 10000),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 13))

for ax, (title, lo, hi) in zip(axes, PANELS):
    sel_o = (olam >= lo) & (olam <= hi)
    sel_bb = (mlam_bb >= lo) & (mlam_bb <= hi)
    sel_ae = (mlam_ae >= lo) & (mlam_ae <= hi)
    ax.plot(olam[sel_o], oflu[sel_o], lw=1.4, color='black', alpha=0.9, label='SN 2002bo Bmax')
    ax.plot(mlam_ae[sel_ae], mflu_ae[sel_ae], lw=1.2, color='steelblue', alpha=0.85, label='old asE 157921')
    ax.plot(mlam_bb[sel_bb], mflu_bb[sel_bb], lw=1.4, color='crimson', alpha=0.9, label='bbfix 158558')
    ax.set_xlim(lo, hi)
    # auto y-scale per panel — this is the key to seeing P-Cygni
    yvals = np.concatenate([oflu[sel_o], mflu_bb[sel_bb], mflu_ae[sel_ae]])
    ymax = np.percentile(yvals, 99.5) * 1.10
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Rest-frame λ [Å]'); ax.set_ylabel('F_λ [erg/s/cm²/Å]')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=10); ax.grid(True, alpha=0.25)
    for lc, name in LINES:
        if lo <= lc <= hi:
            ax.axvline(lc, color='gray', lw=0.4, ls=':', alpha=0.5)
            ax.text(lc, ymax*0.95, name, rotation=90, fontsize=7,
                    ha='right', va='top', color='gray', alpha=0.7)

plt.tight_layout()
out = ROOT / "figures/2026-05-28_full_spectrum_bbfix_3panel_zoom.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")

# Line depth diagnostic
def line_depth(lam, flu, lam_lo, lam_hi, cont_lam):
    """% trough depth relative to local continuum at cont_lam."""
    sel = (lam >= lam_lo) & (lam <= lam_hi)
    if sel.sum() < 3: return None
    Fmin = flu[sel].min()
    Fcont = np.interp(cont_lam, lam, flu)
    return 100. * (1 - Fmin/Fcont)

print("\n=== P-Cygni trough depth %  (lower = shallower) ===")
print(f"  {'feature':14s}  {'obs':>6s}  {'asE':>6s}  {'bbfix':>6s}")
FEATURES = [
    ("Ca II H&K",  3700, 3900, 4050),
    ("Si II 6355", 6050, 6300, 6700),
    ("O I 7774",   7400, 7750, 8000),
    ("Ca II IR",   8000, 8400, 8800),
    ("Mg II 9218", 8950, 9200, 9400),
]
for label, lo, hi, cl in FEATURES:
    do = line_depth(olam, oflu, lo, hi, cl)
    da = line_depth(mlam_ae, mflu_ae, lo, hi, cl)
    db = line_depth(mlam_bb, mflu_bb, lo, hi, cl)
    print(f"  {label:14s}  {do:6.1f}  {da:6.1f}  {db:6.1f}")
