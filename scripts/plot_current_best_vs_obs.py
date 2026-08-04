#!/usr/bin/env python3
"""Current best (157643 oTri_prod) vs SN 2011fe p+3.7d Snifs + HST stitched."""
import glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
C_KMS = 2.998e5
FWHM = 20000.0
TAG = "phase+03.7"
HST_GRATS = [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]

JOB = 157643
RUN_DIR = f"{ROOT}/logs/oTri_prod_vi11600_L2p45_de10p5_p+3.7d_{JOB}"
LABEL = f"LUMINA #281 oTri vi11600 L2.45 de10.5 FE2.0 CO0.3 (job {JOB})"

def band_int(lam, flu, lo, hi):
    sel = (lam >= lo) & (lam <= hi)
    return float(np.trapezoid(flu[sel], lam[sel]))

def smooth(lam, flu, fwhm=FWHM):
    dl = np.median(np.diff(lam))
    mid = 0.5 * (lam[0] + lam[-1])
    sig = (fwhm / C_KMS) * mid / 2.355 / dl
    return gaussian_filter1d(flu, sig, mode="nearest")

# --- load LUMINA formal spectrum ---
m = pd.read_csv(f"{RUN_DIR}/lumina_spectrum_formal.csv")
mlam = m["wavelength_angstrom"].values
mflu = m["flux"].values

# --- load Snifs p+4.2d (closest match to p+3.7d t_exp) ---
sn = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/sn2011fe_p4d2d.csv", comment="#")
slam = sn["wavelength_angstrom"].values
sflu = sn["flux_erg_s_cm2_angstrom"].values

# --- load HST stitched p+3.7d ---
hst_dir = f"{ROOT}/data/sn2011fe/hst_uv"
parts = []
for grat, w_lo, w_hi in HST_GRATS:
    files = sorted(glob.glob(f"{hst_dir}/CCD_{grat}_*{TAG}*sx1.csv"))
    if not files:
        continue
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0].copy()
    if len(dfs) > 1:
        stack = np.column_stack([
            np.interp(base["wavelength_angstrom"].values,
                      d["wavelength_angstrom"].values,
                      d["flux_erg_s_cm2_angstrom"].values) for d in dfs])
        base["flux_erg_s_cm2_angstrom"] = np.nanmean(stack, axis=1)
    sel = (base["wavelength_angstrom"] >= w_lo) & (base["wavelength_angstrom"] <= w_hi)
    parts.append(base.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom"]])
hst = pd.concat(parts, ignore_index=True).sort_values("wavelength_angstrom").reset_index(drop=True)
hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
hlam = hst["wavelength_angstrom"].values
hflu = hst["flux_erg_s_cm2_angstrom"].values

# --- normalize model to HST [4500,5800] band ---
gH = band_int(hlam, hflu, 4500, 5800)
gM = band_int(mlam, mflu, 4500, 5800)
mflu_norm = mflu * (gH / gM)

# --- figure ---
fig, axes = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [3, 3, 2]})

ax = axes[0]
ax.plot(hlam, hflu, lw=1.0, color="black", alpha=0.78, label="HST stitched G230LB+G430L+G750L (obs)")
ax.plot(slam, sflu, lw=0.7, color="goldenrod", alpha=0.78, label="Snifs p+4.2d (obs)")
ax.plot(mlam, mflu_norm, lw=1.0, color="crimson", alpha=0.85, label=LABEL)
ax.set_xlim(1600, 10000)
ax.set_ylim(0, max(hflu.max(), sflu.max()) * 1.15)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title(f"Current best model vs SN 2011fe p+3.7d — full range  [Snifs RMS_bn 0.1860]")
ax.axvspan(1700, 3000, alpha=0.08, color="purple", label="UV [1700,3000]")
ax.axvspan(3000, 8000, alpha=0.05, color="green", label="HST [3000,8000]")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, alpha=0.25)

ax = axes[1]
ax.plot(hlam, hflu, lw=1.2, color="black", alpha=0.85, label="HST obs")
ax.plot(mlam, mflu_norm, lw=1.2, color="crimson", alpha=0.85, label="LUMINA #281")
ax.set_xlim(1700, 3500)
sel = (hlam >= 1700) & (hlam <= 3500)
ax.set_ylim(0, hflu[sel].max() * 1.4)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("UV [1700,3500] zoom — iron-peak III blanketing (#282 AS extension target)")
for xv, lab in [(2382, "Fe II"), (2600, "Fe II"), (2796, "Mg II"), (2803, "Mg II")]:
    ax.axvline(xv, lw=0.6, color="gray", ls=":")
    ax.text(xv, ax.get_ylim()[1] * 0.97, lab, fontsize=7.5, ha="center", color="gray")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)

ax = axes[2]
slam_sm = smooth(slam, sflu)
hlam_sm = smooth(hlam, hflu)
mlam_sm = smooth(mlam, mflu_norm)
ax.plot(hlam, hflu / hlam_sm, lw=0.7, color="black", alpha=0.65, label="HST obs (bn)")
ax.plot(slam, sflu / slam_sm, lw=0.6, color="goldenrod", alpha=0.55, label="Snifs obs (bn)")
ax.plot(mlam, mflu_norm / mlam_sm, lw=0.7, color="crimson", alpha=0.85, label="LUMINA #281 (bn)")
ax.set_xlim(1600, 10000)
ax.set_ylim(0, 2.5)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("flux / baseline (FWHM=20k)")
ax.set_title("Baseline-normalized line shapes — Snifs[3300,8000] RMS_bn 0.186 · UV[1700,3000] residual gap")
ax.axhline(1.0, lw=0.5, color="gray", ls="--")
ax.axvspan(1700, 3000, alpha=0.08, color="purple")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

# trough labels on residual panel
for xv, lab in [(3933, "Ca II K"), (3968, "Ca II H"), (4471, "Mg II"),
                (5169, "Fe II"), (5876, "He I/Na D"), (6355, "Si II"),
                (7774, "O I"), (8500, "Ca II IR")]:
    ax.axvline(xv, lw=0.4, color="navy", ls=":", alpha=0.4)

plt.tight_layout()
out = f"{ROOT}/figures/2026-05-25_best_157643_oTri_vs_sn2011fe_p3p7d.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")

# --- scores ---
def rms_bn(mod_lam, mod_flu, obs_lam, obs_flu, gO, fwhm=FWHM, wl_lo=3300., wl_hi=8000.):
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0: return float("nan")
    mod_flu = mod_flu * (gO/g)
    selO = (obs_lam>=wl_lo) & (obs_lam<=wl_hi); ol, of = obs_lam[selO], obs_flu[selO]
    if len(ol) < 10: return float("nan")
    sO = smooth(ol, of)
    selM = (mod_lam>=wl_lo-100) & (mod_lam<=wl_hi+100); ml, mf = mod_lam[selM], mod_flu[selM]
    if len(ml) < 10: return float("nan")
    sM = smooth(ml, mf)
    common = (ol>=ml[0]) & (ol<=ml[-1])
    mb = np.interp(ol[common], ml, mf/sM)
    return float(np.sqrt(np.mean((of[common]/sO[common] - mb)**2)))

gS = band_int(slam, sflu, 4500, 5800)
r_snifs   = rms_bn(mlam, mflu, slam, sflu, gS, wl_lo=3300, wl_hi=8000)
r_hst_can = rms_bn(mlam, mflu, hlam, hflu, gH, wl_lo=3000, wl_hi=8000)
r_hst_uv  = rms_bn(mlam, mflu, hlam, hflu, gH, wl_lo=1700, wl_hi=3000)
b_uv_obs   = band_int(hlam, hflu, 1700, 2400)
b_uv_modN  = band_int(mlam, mflu_norm, 1700, 2400)
uv_ratio   = b_uv_modN / b_uv_obs
print()
print(f"=== job {JOB} oTri_prod scores (p+3.7d) ===")
print(f"  Snifs [3300,8000]  = {r_snifs:.4f}")
print(f"  HST   [3000,8000]  = {r_hst_can:.4f}  (Tier-4 ≤0.164)")
print(f"  HST   [1700,3000]  = {r_hst_uv:.4f}")
print(f"  UV[1700,2400] L/HST ratio = {uv_ratio:.2f}  (#279 baseline 34×)")
