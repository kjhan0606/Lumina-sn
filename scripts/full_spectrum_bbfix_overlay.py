#!/usr/bin/env python3
"""#298 bbfix vs old-asE FULL spectrum overlay vs SN 2002bo Bmax.

Three traces over full UVOIR range:
  - bbfix prod 158558  (post-#298 patch: Fe III bb +7×, Co III bb +162×)
  - old asE 157921     (#298-tainted base)
  - SN 2002bo m0d0     (rest-frame, CCM-dereddened)

Both models pinned to obs via integrated flux over Blondin F_scl [4000,6000] Å.

Two-panel:
  (1) λ²F_λ over [1500,10500] (paper Fig 6 convention)
  (2) F_λ raw over [3000,10500] (line shapes)
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

# obs
obs = pd.read_csv(OBS, comment='#')
olam = obs['wavelength_angstrom'].values / (1.0 + Z_HEL)
oflu = deredden(obs['wavelength_angstrom'].values, obs['flux_erg_s_cm2_angstrom'].values) * (1.0 + Z_HEL)

# models
m_bb = pd.read_csv(BBFIX); mlam_bb = m_bb['wavelength_angstrom'].values; mflu_bb_raw = m_bb['flux'].values
m_ae = pd.read_csv(ASE);   mlam_ae = m_ae['wavelength_angstrom'].values; mflu_ae_raw = m_ae['flux'].values

ANCHOR_LO, ANCHOR_HI = 4000., 6000.
def anchor(mlam, mflu_raw):
    sel_m = (mlam >= ANCHOR_LO) & (mlam <= ANCHOR_HI)
    sel_o = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
    K = float(np.trapezoid(oflu[sel_o], olam[sel_o])) / float(np.trapezoid(mflu_raw[sel_m], mlam[sel_m]))
    return mflu_raw * K, K

mflu_bb, K_bb = anchor(mlam_bb, mflu_bb_raw)
mflu_ae, K_ae = anchor(mlam_ae, mflu_ae_raw)
print(f"K_bbfix = {K_bb:.3e}    K_asE = {K_ae:.3e}    ratio = {K_bb/K_ae:.4f}")

# Line ID for annotation
LINES = [
    (3934, "Ca II H",   "Ca"), (3968, "Ca II K",   "Ca"),
    (4128, "Si II 4128","Si"), (4481, "Mg II 4481","Mg"),
    (4555, "Fe II 4555","Fe"), (4924, "Fe II m42", "Fe"),
    (5018, "Fe II 5018","Fe"), (5169, "Fe II 5169","Fe"),
    (5454, "S II W",    "S"),  (5640, "S II W",    "S"),
    (5972, "Si II 5972","Si"), (6355, "Si II 6355","Si"),
    (7155, "[Fe II]7155","Fe"),(7378, "Ni II 7378","Ni"),
    (7774, "O I 7774",  "O"),
    (8498, "Ca IR 8498","Ca"), (8542, "Ca IR 8542","Ca"), (8662, "Ca IR 8662","Ca"),
    (9218, "Mg II 9218","Mg"), (9244, "Mg II 9244","Mg"),
]
COLOR = {"Ca":"#1f77b4","Si":"#d62728","Fe":"#8c564b","Mg":"#9467bd","O":"#2ca02c","S":"#ff7f0e","Ni":"#7f7f7f"}

# --- Figure: 3 panels stacked ---
fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                         gridspec_kw={"height_ratios":[3, 3, 2]})

# Panel 1: λ²F_λ full UVOIR (Blondin Fig 6 convention)
ax = axes[0]
o_l2f = olam**2 * oflu
m_l2f_bb = mlam_bb**2 * mflu_bb
m_l2f_ae = mlam_ae**2 * mflu_ae
ax.plot(olam, o_l2f, lw=1.2, color='black', alpha=0.9, label='SN 2002bo Bmax (dereddened, rest-frame)')
ax.plot(mlam_ae, m_l2f_ae, lw=1.0, color='steelblue', alpha=0.8, label='old asE (#298-tainted, job 157921)')
ax.plot(mlam_bb, m_l2f_bb, lw=1.2, color='crimson', alpha=0.85, label='bbfix prod (#298 patched, job 158558)')
ax.set_xlim(1500, 10500)
ymax = max(np.percentile(o_l2f[(olam>=3000)&(olam<=10200)], 99.5),
           np.percentile(m_l2f_bb[(mlam_bb>=3000)&(mlam_bb<=10200)], 99.5),
           np.percentile(m_l2f_ae[(mlam_ae>=3000)&(mlam_ae<=10200)], 99.5)) * 1.20
ax.set_ylim(0, ymax)
ax.set_xlabel('Rest-frame λ [Å]'); ax.set_ylabel('λ² F_λ  [Å² · erg s⁻¹ cm⁻² Å⁻¹]')
ax.set_title('λ² F_λ (Blondin Fig 6 convention) — anchor [4000,6000] Å pinned to obs')
ax.legend(loc='upper left', fontsize=10); ax.grid(True, alpha=0.25)
for lc, name, sp in LINES:
    ax.axvline(lc, color=COLOR.get(sp,"gray"), lw=0.5, ls=':', alpha=0.4)

# Panel 2: F_λ full UVOIR
ax = axes[1]
ax.plot(olam, oflu, lw=1.2, color='black', alpha=0.9, label='SN 2002bo Bmax')
ax.plot(mlam_ae, mflu_ae, lw=1.0, color='steelblue', alpha=0.8, label='old asE 157921')
ax.plot(mlam_bb, mflu_bb, lw=1.2, color='crimson', alpha=0.85, label='bbfix 158558')
ax.set_xlim(3000, 10500)
sel = (olam >= 3000) & (olam <= 10500)
ymax2 = max(np.percentile(oflu[sel], 99.5),
            np.percentile(mflu_bb[(mlam_bb>=3000)&(mlam_bb<=10500)], 99.5)) * 1.15
ax.set_ylim(0, ymax2)
ax.set_xlabel('Rest-frame λ [Å]'); ax.set_ylabel('F_λ  [erg s⁻¹ cm⁻² Å⁻¹]')
ax.set_title('F_λ raw (line shapes, optical+NIR)')
ax.legend(loc='upper right', fontsize=10); ax.grid(True, alpha=0.25)
for lc, name, sp in LINES:
    if 3000 <= lc <= 10500:
        ax.axvline(lc, color=COLOR.get(sp,"gray"), lw=0.5, ls=':', alpha=0.4)
        ax.text(lc, ymax2*0.96, name, rotation=90, fontsize=7,
                ha='right', va='top', color=COLOR.get(sp,"gray"), alpha=0.7)

# Panel 3: ratio (model/obs) - shows where each model is over/under
ax = axes[2]
common_lo, common_hi = 2500., 10500.
sel_o = (olam >= common_lo) & (olam <= common_hi)
mi_bb = np.interp(olam[sel_o], mlam_bb, mflu_bb)
mi_ae = np.interp(olam[sel_o], mlam_ae, mflu_ae)
mask = oflu[sel_o] > 0
ratio_bb = mi_bb[mask] / oflu[sel_o][mask]
ratio_ae = mi_ae[mask] / oflu[sel_o][mask]
lam_r = olam[sel_o][mask]
ax.plot(lam_r, ratio_ae, lw=0.8, color='steelblue', alpha=0.7, label='old asE / obs')
ax.plot(lam_r, ratio_bb, lw=1.0, color='crimson', alpha=0.85, label='bbfix / obs')
ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.7)
ax.set_xlim(common_lo, common_hi); ax.set_ylim(0, 4)
ax.set_xlabel('Rest-frame λ [Å]'); ax.set_ylabel('model / obs ratio')
ax.set_title('Model/obs ratio (1.0 = perfect; >1 = LUMINA excess, <1 = deficit)')
ax.legend(loc='upper right', fontsize=10); ax.grid(True, alpha=0.25)
for lc, name, sp in LINES:
    if common_lo <= lc <= common_hi:
        ax.axvline(lc, color=COLOR.get(sp,"gray"), lw=0.4, ls=':', alpha=0.35)

plt.tight_layout()
out = ROOT / "figures/2026-05-28_full_spectrum_bbfix_vs_ase_vs_sn2002bo.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nsaved: {out}")

# --- Band-int ratios over UVOIR bands ---
BANDS = [
    ("UV[2000-3000]", 2000., 3000.),
    ("blue[3000-4500]", 3000., 4500.),
    ("opt[4500-6500]", 4500., 6500.),
    ("red[6500-8000]", 6500., 8000.),
    ("NIR[8000-10000]", 8000., 10000.),
    ("ALL[3000-10000]", 3000., 10000.),
]
print(f"\n=== Per-band integrated flux ratio (model/obs) ===")
print(f"  {'band':22s}  {'asE/obs':>9s}  {'bbfix/obs':>11s}  {'Δ (bb−asE)':>12s}")
for name, lo, hi in BANDS:
    sel_o = (olam >= lo) & (olam <= hi)
    if sel_o.sum() < 5:
        print(f"  {name:22s}  obs gap")
        continue
    F_o = float(np.trapezoid(oflu[sel_o], olam[sel_o]))
    F_bb = float(np.trapezoid(np.interp(olam[sel_o], mlam_bb, mflu_bb), olam[sel_o]))
    F_ae = float(np.trapezoid(np.interp(olam[sel_o], mlam_ae, mflu_ae), olam[sel_o]))
    r_bb = F_bb / F_o; r_ae = F_ae / F_o
    print(f"  {name:22s}  {r_ae:9.3f}  {r_bb:11.3f}  {r_bb - r_ae:+12.3f}")
