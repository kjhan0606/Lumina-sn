#!/usr/bin/env python3
"""λ²F_λ residual map of LUMINA(plain DDC15) vs SN 2002bo with line IDs.

Paper Fig 6 convention: λ²F_λ vs λ on linear axis (highlights Si II 6355 trough + UV).
Plus NIR zoom [6500,10500] with vertical line IDs to confirm/reject the
'NIR over-emission is Ca II IR triplet' hypothesis vs 'Fe II forest'.

Usage: python3 scripts/residual_map_sn2002bo.py <JOB> [TAG]
       default JOB=157921 (α v1 baseline = EPS_IR=0.0 cell from #288)
"""
import sys, glob, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
JOB  = int(sys.argv[1]) if len(sys.argv) > 1 else 157921
TAG  = sys.argv[2] if len(sys.argv) > 2 else "prod"

cand = sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{TAG}_{JOB}"))
if not cand:
    cand = sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{JOB}"))
if not cand:
    print(f"[err] no run dir for job {JOB}", file=sys.stderr); sys.exit(1)
RUN = cand[0]
print(f"RUN = {RUN}")

MU, EBV, RV = 31.90, 0.41, 3.1
DIST_CM = 10**((MU+5)/5) * 3.0857e18
V_PHOT_KMS = 11000.0          # 2002bo Bmax photospheric v from Benetti+2004 Si II 6355
Z_HEL = 0.0042                # Hamuy+2002

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
def deredden(wave_aa, flux):
    return flux * 10**(0.4 * A_V * ccm_a_over_av(wave_aa))

# --- Load model (rest-frame λ, internal flux unit) ---
m = pd.read_csv(f"{RUN}/lumina_spectrum_formal.csv")
mlam = m["wavelength_angstrom"].values
mflu_raw = m["flux"].values

# --- Load obs (observer frame; deredden then deredshift to rest frame for direct λ-comparison) ---
obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam_obs = obs["wavelength_angstrom"].values
oflu_obs = deredden(olam_obs, obs["flux_erg_s_cm2_angstrom"].values)
olam = olam_obs / (1.0 + Z_HEL)
oflu = oflu_obs * (1.0 + Z_HEL)   # F_λ ∝ 1/(1+z); intrinsic = obs × (1+z)

# --- Anchor on [4000,6000] (Blondin F_scl band) ---
ANCHOR_LO, ANCHOR_HI = 4000., 6000.
sel_m = (mlam >= ANCHOR_LO) & (mlam <= ANCHOR_HI)
sel_o = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
K = float(np.trapezoid(oflu[sel_o], olam[sel_o])) / float(np.trapezoid(mflu_raw[sel_m], mlam[sel_m]))
mflu = mflu_raw * K
print(f"  K = {K:.3e}   (F_scl anchor: model [{ANCHOR_LO:.0f},{ANCHOR_HI:.0f}] pinned to obs)")

# --- λ²F_λ ---
m_l2f = mlam**2 * mflu
o_l2f = olam**2 * oflu

# --- Line ID list (rest-frame Å, species, role) ---
LINES = [
    (3934, "Ca II H",       "Ca",   "P-Cygni"),
    (3968, "Ca II K",       "Ca",   "P-Cygni"),
    (4128, "Si II λ4128",   "Si",   "absorption"),
    (4481, "Mg II λ4481",   "Mg",   "absorption"),
    (4555, "Fe II 4555",    "Fe",   "blend"),
    (4924, "Fe II 4924 (Mult 42)", "Fe", "blend"),
    (5018, "Fe II 5018 (Mult 42)", "Fe", "blend"),
    (5169, "Fe II 5169 (Mult 42)", "Fe", "blend"),
    (5454, "S II W (5454)",  "S",   "emission/W"),
    (5640, "S II W (5640)",  "S",   "emission/W"),
    (5972, "Si II λ5972",    "Si",  "absorption"),
    (6355, "Si II λ6355",    "Si",  "absorption"),
    (7155, "[Fe II] 7155",   "Fe",  "nebular tracer"),
    (7378, "[Ni II] 7378",   "Ni",  "nebular tracer"),
    (7774, "O I λ7774",      "O",   "absorption"),
    (8498, "Ca II IR 8498",  "Ca",  "P-Cygni"),
    (8542, "Ca II IR 8542",  "Ca",  "P-Cygni"),
    (8662, "Ca II IR 8662",  "Ca",  "P-Cygni"),
    (9218, "Mg II 9218",     "Mg",  "absorption"),
    (9244, "Mg II 9244",     "Mg",  "absorption"),
]
COLOR = {"Ca":"#1f77b4","Si":"#d62728","Fe":"#8c564b","Mg":"#9467bd","O":"#2ca02c","S":"#ff7f0e","Ni":"#7f7f7f"}

# --- Per-line residual budget over ±150 Å windows ---
budget = []
for lc, name, sp, role in LINES:
    lo, hi = lc - 150., lc + 150.
    sel = (olam >= lo) & (olam <= hi)
    if sel.sum() < 5: continue
    mi = np.interp(olam[sel], mlam, mflu)
    F_o = float(np.trapezoid(oflu[sel], olam[sel]))
    F_m = float(np.trapezoid(mi,        olam[sel]))
    if F_o <= 0: continue
    ratio = F_m / F_o
    budget.append((lc, name, sp, role, ratio))

# --- Figure ---
fig, axes = plt.subplots(3, 1, figsize=(15, 13), gridspec_kw={"height_ratios":[3, 3, 3]})

def annotate_lines(ax, lo, hi, ymin, ymax, only_species=None):
    for lc, name, sp, role, *_ in [(*L, None) for L in LINES]:
        if lc < lo or lc > hi: continue
        if only_species is not None and sp not in only_species: continue
        ax.axvline(lc, color=COLOR.get(sp,"gray"), lw=0.6, ls="--", alpha=0.55)
        ax.text(lc, ymax*0.93, name, rotation=90, fontsize=7,
                ha="right", va="top", color=COLOR.get(sp,"gray"), alpha=0.85)

ax = axes[0]  # λ²F_λ full UVOIR (paper Fig 6 style)
ax.plot(olam, o_l2f, lw=1.0, color="black",   alpha=0.85, label="SN 2002bo Bmax (rest-frame, dereddened)")
ax.plot(mlam, m_l2f, lw=1.0, color="crimson", alpha=0.85, label=f"LUMINA plain DDC15 EPS_IR=0.0 (job {JOB})")
ax.set_xlim(1500, 10500)
ymax_full = max(np.percentile(o_l2f[(olam>=3300)&(olam<=10200)], 99),
                np.percentile(m_l2f[(mlam>=3300)&(mlam<=10200)], 99)) * 1.25
ax.set_ylim(0, ymax_full)
ax.set_xlabel("Rest-frame wavelength [Å]"); ax.set_ylabel("λ² F_λ  [Å² · erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("Blondin Fig 6 convention: λ² F_λ (paper anchor [4000,6000] Å pinned)")
ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.25)
annotate_lines(ax, 1500, 10500, 0, ymax_full)

ax = axes[1]  # NIR zoom [6500,10500]
sel_o2 = (olam >= 6500) & (olam <= 10500)
sel_m2 = (mlam >= 6500) & (mlam <= 10500)
ax.plot(olam[sel_o2], o_l2f[sel_o2], lw=1.1, color="black",   alpha=0.9, label="SN 2002bo (rest-frame)")
ax.plot(mlam[sel_m2], m_l2f[sel_m2], lw=1.1, color="crimson", alpha=0.9, label="LUMINA")
ymax_nir = max(o_l2f[sel_o2].max(), m_l2f[sel_m2].max()) * 1.20
ax.set_xlim(6500, 10500); ax.set_ylim(0, ymax_nir)
ax.set_xlabel("Rest-frame wavelength [Å]"); ax.set_ylabel("λ² F_λ")
ax.set_title("NIR zoom — Ca II IR triplet (blue) vs O I 7774 (green) vs Mg II (purple) vs Fe forest")
ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.25)
annotate_lines(ax, 6500, 10500, 0, ymax_nir)
# Photospheric-velocity blueshift markers for the dominant Ca/O/Mg P-Cygni absorptions
v = V_PHOT_KMS / 2.998e5
for lc in [7774, 8498, 8542, 8662, 9218, 9244]:
    lc_blue = lc * (1 - v)
    ax.axvline(lc_blue, color="dimgray", lw=0.4, ls=":", alpha=0.5)

ax = axes[2]  # residual = model - obs in λ²Fλ
common = (olam >= mlam[0]) & (olam <= mlam[-1])
mi_resid = np.interp(olam[common], mlam, mflu)
resid = (olam[common]**2) * (mi_resid - oflu[common])
ax.plot(olam[common], resid, lw=0.7, color="navy", alpha=0.9)
ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
ax.set_xlim(1500, 10500)
yrng = np.percentile(np.abs(resid[(olam[common]>=2000)&(olam[common]<=10200)]), 99) * 1.4
ax.set_ylim(-yrng, yrng)
ax.set_xlabel("Rest-frame wavelength [Å]"); ax.set_ylabel("λ² · (F_mod − F_obs)")
ax.set_title("Residual λ² ΔF — positive = LUMINA excess, negative = LUMINA deficit")
ax.grid(True, alpha=0.25)
annotate_lines(ax, 1500, 10500, -yrng, yrng)

plt.tight_layout()
DATE = "2026-05-26"
out = f"{ROOT}/figures/{DATE}_residualmap_{JOB}_vs_sn2002bo_bmax.png"
os.makedirs(f"{ROOT}/figures", exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nsaved: {out}")

# --- Per-line residual budget table ---
print("\n=== Per-line model/obs ratio (±150Å window) — >1 = LUMINA excess, <1 = deficit ===")
print(f"  {'λ_rest':>6s}  {'species':10s}  {'mod/obs':>8s}  {'name':28s}  role")
for lc, name, sp, role, ratio in sorted(budget, key=lambda x: -abs(np.log10(x[4]))):
    flag = "  ←excess" if ratio > 1.5 else ("  ←deficit" if ratio < 0.7 else "")
    print(f"  {lc:6d}  {sp:10s}  {ratio:8.2f}  {name:28s}  {role}{flag}")
