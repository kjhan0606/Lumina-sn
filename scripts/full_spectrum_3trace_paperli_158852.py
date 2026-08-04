#!/usr/bin/env python3
"""3-trace overlay: obs / bbfix-macroatom 158558 / paperli-scatter 158852.

Tests user directive (2026-05-28): "paper 비교에서 macro-atom 사용 금지".
158852 reruns the same DDC15 / 2002bo / vi=9019 / L=1.22e43 / N_PKT=200k
config but with LUMINA_LINE_INTERACTION=scatter (no macro-atom cascade).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
BB   = ROOT / "logs/paperDDC15v3asEfloor_bbfix_2002bo_vi9019_L1p0_prod_158558/lumina_spectrum_formal.csv"
SCT  = ROOT / "logs/paperDDC15_paperli_scatter_2002bo_vi9019_L1p0_prod_158852/lumina_spectrum_formal.csv"
OBS  = ROOT / "data/sn2002bo/epochs/sn2002bo_m0d0.csv"

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

def load(p):
    df = pd.read_csv(p)
    return df['wavelength_angstrom'].values, df['flux'].values

mlam_bb, mflu_bb_raw = load(BB)
mlam_sc, mflu_sc_raw = load(SCT)

ANCHOR_LO, ANCHOR_HI = 4000., 6000.
def anchor(mlam, mflu_raw):
    sel_m = (mlam >= ANCHOR_LO) & (mlam <= ANCHOR_HI)
    sel_o = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
    K = float(np.trapezoid(oflu[sel_o], olam[sel_o])) / float(np.trapezoid(mflu_raw[sel_m], mlam[sel_m]))
    return mflu_raw * K, K

mflu_bb, K_bb = anchor(mlam_bb, mflu_bb_raw)
mflu_sc, K_sc = anchor(mlam_sc, mflu_sc_raw)
print(f"K_bb={K_bb:.3e}  K_sc={K_sc:.3e}  (sc/bb={K_sc/K_bb:.4f})")

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
    sel_o  = (olam >= lo) & (olam <= hi)
    sel_bb = (mlam_bb >= lo) & (mlam_bb <= hi)
    sel_sc = (mlam_sc >= lo) & (mlam_sc <= hi)
    ax.plot(olam[sel_o], oflu[sel_o], lw=1.6, color='black', alpha=0.9, label='SN 2002bo Bmax (obs)')
    ax.plot(mlam_bb[sel_bb], mflu_bb[sel_bb], lw=1.3, color='crimson', alpha=0.85, label='bbfix MACROATOM 158558')
    ax.plot(mlam_sc[sel_sc], mflu_sc[sel_sc], lw=1.5, color='steelblue', alpha=0.9, label='paperli SCATTER 158852 (no macro-atom)')
    ax.set_xlim(lo, hi)
    yvals = np.concatenate([oflu[sel_o], mflu_bb[sel_bb], mflu_sc[sel_sc]])
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
out = ROOT / "figures/2026-05-28_spectrum_3trace_158852_paperli_scatter.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130)
print(f"saved: {out}")

# Line-depth diagnostic
def line_depth(lam, flu, lam_lo, lam_hi, cont_lam):
    sel = (lam >= lam_lo) & (lam <= lam_hi)
    if sel.sum() < 3: return None
    Fmin = flu[sel].min()
    Fcont = np.interp(cont_lam, lam, flu)
    return 100. * (1 - Fmin/Fcont)

print("\n=== P-Cygni trough depth %  (lower = shallower) ===")
print(f"  {'feature':14s}  {'obs':>7s}  {'bbfix-MA':>9s}  {'paperli-SC':>11s}  {'Δ(SC-MA)':>10s}")
FEATURES = [
    ("Ca II H&K",  3700, 3900, 4050),
    ("Si II 6355", 6050, 6300, 6700),
    ("O I 7774",   7400, 7750, 8000),
    ("Ca II IR",   8000, 8400, 8800),
    ("Mg II 9218", 8950, 9200, 9400),
]
for label, lo, hi, cl in FEATURES:
    do = line_depth(olam, oflu, lo, hi, cl)
    db = line_depth(mlam_bb, mflu_bb, lo, hi, cl)
    ds = line_depth(mlam_sc, mflu_sc, lo, hi, cl)
    delta = ds - db if (ds is not None and db is not None) else float('nan')
    print(f"  {label:14s}  {do:7.1f}  {db:9.1f}  {ds:11.1f}  {delta:+10.1f}")
