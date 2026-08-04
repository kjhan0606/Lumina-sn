#!/usr/bin/env python3
"""Compute Blondin+2013 Fig 6 / Table 3 F_scl for LUMINA(plain DDC15) vs SN 2002bo at Bmax.

F_scl definition (Blondin Sect 6, eq footnote):
    F_scl = < F_synth(λ) / F_obs(λ) > averaged over λ ∈ [4000, 6000] Å
    Q_uvoir = L_synth_UVOIR / L_obs_UVOIR  (UVOIR = total integrated 1500-25000Å)
    |F_scl − Q_uvoir| is the honest SED-agreement metric (paper got 0.01 for DDC15 vs 2002bo).

Distance: μ = 31.90 → d = 10^((μ+5)/5) pc = 2.398e7 pc = 7.401e25 cm  (Blondin Table 3 NIR-derived).
Extinction: E(B-V) = 0.41, R_V = 3.1 (CCM 1989). Apply to obs (deredden) OR to model (redden); we deredden obs.
"""
import sys, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
JOB = int(sys.argv[1]) if len(sys.argv) > 1 else 157708
TAG = sys.argv[2] if len(sys.argv) > 2 else "smoke"

# Locate the model run dir
cand = sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{TAG}_{JOB}"))
if not cand:
    cand = sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{JOB}"))
if not cand:
    print(f"[err] no run dir for job {JOB}", file=sys.stderr); sys.exit(1)
RUN = cand[0]
print(f"RUN = {RUN}")

# Paper params for SN 2002bo (Blondin+2013 Table 3)
MU      = 31.90
EBV     = 0.41
RV      = 3.1
DIST_CM = 10**((MU + 5)/5) * 3.0857e18  # parsec → cm

# CCM 1989 extinction A(lambda)/A(V); we use the analytic CCM formula for 1250-33333Å.
def ccm_a_over_av(wave_aa):
    x = 1e4 / wave_aa  # x in 1/μm
    a = np.zeros_like(x); b = np.zeros_like(x)
    # Optical/NIR 1.1 <= x <= 3.3
    sel = (x >= 1.1) & (x <= 3.3)
    y = x[sel] - 1.82
    a[sel] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[sel] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    # NIR 0.3 <= x < 1.1
    sel = (x >= 0.3) & (x < 1.1)
    a[sel] =  0.574 * x[sel]**1.61
    b[sel] = -0.527 * x[sel]**1.61
    # UV 3.3 <= x <= 8.0
    sel = (x > 3.3) & (x <= 8.0)
    xs = x[sel]
    Fa = np.where(xs >= 5.9, -0.04473*(xs-5.9)**2 - 0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs >= 5.9,  0.2130*(xs-5.9)**2 + 0.1207*(xs-5.9)**3, 0.0)
    a[sel] =  1.752 - 0.316*xs - 0.104/((xs - 4.67)**2 + 0.341) + Fa
    b[sel] = -3.090 + 1.825*xs + 1.206/((xs - 4.62)**2 + 0.263) + Fb
    return a + b / RV  # A(λ)/A(V) for given R_V

A_V = RV * EBV  # mag
def deredden(wave_aa, flux):
    a_over_av = ccm_a_over_av(wave_aa)
    A_lambda = A_V * a_over_av
    # F_intrinsic = F_obs × 10^(0.4 × A_λ)
    return flux * 10**(0.4 * A_lambda)

# --- Load model ---
m = pd.read_csv(f"{RUN}/lumina_spectrum_formal.csv")
mlam = m["wavelength_angstrom"].values
mflu_raw = m["flux"].values  # arbitrary LUMINA internal scale, NOT erg/s/Å
print(f"  model raw flux peak {mflu_raw.max():.3e}  ∫dλ over [1500,10200] = {np.trapezoid(mflu_raw[(mlam>=1500)&(mlam<=10200)], mlam[(mlam>=1500)&(mlam<=10200)]):.3e}")

# --- Load obs ---
obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu_raw = obs["flux_erg_s_cm2_angstrom"].values
oflu = deredden(olam, oflu_raw)  # intrinsic flux at SN (dereddened)
print(f"  obs raw  F_λ median {np.median(oflu_raw):.3e}  →  dereddened median {np.median(oflu):.3e} (A_V={A_V:.2f}, gain factor median {np.median(oflu/np.maximum(oflu_raw,1e-30)):.2f})")

# --- Optical-anchor normalization (Blondin Sect 6: F_scl band = [4000,6000]Å) ---
# Pin ∫F_mod = ∫F_obs over [4000,6000]Å. F_scl is then ≡ 1 by construction in that band.
# Per-band ratios elsewhere reveal SED-shape disagreement honestly.
# Q_uvoir then becomes a derived number: how much extra/less UVOIR luminosity model has vs obs.
ANCHOR_LO, ANCHOR_HI = 4000., 6000.
sel_m_a = (mlam >= ANCHOR_LO) & (mlam <= ANCHOR_HI)
sel_o_a = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
I_mod_anchor_raw = float(np.trapezoid(mflu_raw[sel_m_a], mlam[sel_m_a]))
I_obs_anchor     = float(np.trapezoid(oflu[sel_o_a],     olam[sel_o_a]))
K = I_obs_anchor / I_mod_anchor_raw
mflu = mflu_raw * K
print(f"  optical anchor [{ANCHOR_LO:.0f},{ANCHOR_HI:.0f}]Å match: K = {K:.3e} (LUMINA → erg/s/cm²/Å)")

# UVOIR comparison AFTER anchor normalization — reveals how much "extra" UV flux model carries.
UVOIR_LO, UVOIR_HI = 1500., 10200.
sel_m = (mlam >= UVOIR_LO) & (mlam <= UVOIR_HI)
sel_o2 = (olam >= max(UVOIR_LO, olam[0])) & (olam <= UVOIR_HI)
I_mod_UVOIR = float(np.trapezoid(mflu[sel_m], mlam[sel_m]))
I_obs_UVOIR = float(np.trapezoid(oflu[sel_o2], olam[sel_o2]))

F_scl = 1.000  # by construction (anchor pin)
Q_uvoir = I_mod_UVOIR / I_obs_UVOIR
print(f"\n=== Blondin SED-agreement metrics (optical anchor [{ANCHOR_LO:.0f},{ANCHOR_HI:.0f}]Å pinned) ===")
print(f"  F_scl = {F_scl:.3f}   (paper DDC15 vs 2002bo = 1.05; pinned, so identically 1.0 here)")
print(f"  Q_uvoir = {Q_uvoir:.3f}   (paper = 1.06; UVOIR over [{UVOIR_LO:.0f},{UVOIR_HI:.0f}]Å, obs starts at {olam[0]:.0f}Å)")
print(f"  |F_scl - Q_uvoir| = {abs(F_scl - Q_uvoir):.3f}   (paper = 0.01 — Blondin's honest SED metric)")
# Also include unobserved FUV [1500,3356] over total model UVOIR — fraction of model flux in unobserved band
sel_m_uv = (mlam >= 1500) & (mlam <= 3356)
I_mod_UV_only = float(np.trapezoid(mflu[sel_m_uv], mlam[sel_m_uv]))
I_mod_all     = float(np.trapezoid(mflu[(mlam>=1500)&(mlam<=10200)], mlam[(mlam>=1500)&(mlam<=10200)]))
print(f"  Model flux fraction in unobserved [1500,3356]Å: {I_mod_UV_only/I_mod_all*100:.1f}%   (suggests blue-shifted SED if >>15%)")

# --- Per-band ratio table ---
print(f"\n=== Per-band model/obs ratios (dereddened obs) ===")
BANDS = [(1700,2400,"UV mid"),(2400,3000,"UV near"),(3000,4000,"UV/blue"),
         (4000,5500,"blue/green"),(5500,7000,"red"),(7000,9000,"NIR I"),(9000,10200,"NIR II")]
print(f"  {'band':12s}  {'λ range':>16s}  {'mod/obs':>8s}")
for lo, hi, name in BANDS:
    sel = (olam >= lo) & (olam <= hi)
    if sel.sum() < 5: continue
    mi_b = np.interp(olam[sel], mlam, mflu)
    rb = float(np.trapezoid(mi_b, olam[sel])) / max(float(np.trapezoid(oflu[sel], olam[sel])), 1e-30)
    print(f"  {name:12s}  [{lo:5d},{hi:5d}]Å  {rb:>8.2f}")

# --- Figure: λ²F_λ display per paper Fig 6 + raw overlay + ratio ---
fig, axes = plt.subplots(3, 1, figsize=(13, 12), gridspec_kw={"height_ratios":[3, 3, 2.4]})

ax = axes[0]
ax.plot(olam, oflu_raw, lw=0.8, color="goldenrod", alpha=0.5, label=f"SN 2002bo raw (E(B-V)={EBV})")
ax.plot(olam, oflu,     lw=0.9, color="black",     alpha=0.85, label="SN 2002bo dereddened")
ax.plot(mlam, mflu,     lw=0.9, color="crimson",   alpha=0.85, label=f"LUMINA plain DDC15 (#283 job {JOB}, F_scl={F_scl:.2f})")
ax.set_xlim(1500, 10500); ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title(f"LUMINA plain DDC15 vs SN 2002bo at B-max  —  paper F_scl=1.05, Q_uvoir=1.06")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)

ax = axes[1]
sel_o_lam = (olam >= 1500) & (olam <= 10500)
sel_m_lam = (mlam >= 1500) & (mlam <= 10500)
ax.semilogy(olam[sel_o_lam], oflu[sel_o_lam], lw=0.9, color="black",   alpha=0.85, label="2002bo dereddened (log F)")
ax.semilogy(mlam[sel_m_lam], mflu[sel_m_lam], lw=0.9, color="crimson", alpha=0.85, label="LUMINA plain DDC15 (log F)")
ax.set_xlim(1500, 10500); ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("F_λ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$] (log)")
ax.set_title("Log F_λ — UVOIR-pinned (full UV→red span on a single scale)")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)

ax = axes[2]
common = (olam >= mlam[0]) & (olam <= mlam[-1])
mi = np.interp(olam[common], mlam, mflu)
ratio = mi / np.maximum(oflu[common], 1e-30)
ax.semilogy(olam[common], ratio, lw=0.7, color="crimson", alpha=0.85)
ax.axhline(1.0, lw=1.0, color="black", ls="--", alpha=0.6, label="model = obs")
ax.axhline(F_scl, lw=0.7, color="navy", ls="--", alpha=0.6, label=f"F_scl = {F_scl:.2f}")
ax.axhline(2.0, lw=0.5, color="orange", ls=":", alpha=0.4); ax.axhline(0.5, lw=0.5, color="orange", ls=":", alpha=0.4)
ax.axhline(10., lw=0.5, color="red", ls=":", alpha=0.4); ax.axhline(0.1, lw=0.5, color="red", ls=":", alpha=0.4)
ax.set_xlim(1500, 10500); ax.set_ylim(0.05, 200); ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("model / obs (log)")
ax.set_title("model/obs ratio (honest, dereddened obs at d=μ31.90)")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
import os
RUN_TAG_NAME = os.path.basename(RUN).split("_2002bo")[0]  # "plainDDC15" or "paperDDC15v2"
DATE_TAG = "2026-05-26"
out = f"{ROOT}/figures/{DATE_TAG}_{RUN_TAG_NAME}_{JOB}_vs_sn2002bo_bmax.png"
os.makedirs(f"{ROOT}/figures", exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nsaved: {out}")
