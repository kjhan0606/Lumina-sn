#!/usr/bin/env python3
"""LUMINA r070 with artificial red continuum × 0.5 above 6700Å (smooth 6500-6700 transition).
Compare to SN 2011fe p+3.7d obs."""
import glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
C_KMS = 2.998e5
FWHM = 20000.0
TAG = "phase+03.7"

def band_int(lam, flu, lo, hi):
    sel = (lam >= lo) & (lam <= hi)
    return float(np.trapezoid(flu[sel], lam[sel]))

def smooth(lam, flu, fwhm=FWHM):
    dl = np.median(np.diff(lam))
    mid = 0.5 * (lam[0] + lam[-1])
    sig = (fwhm / C_KMS) * mid / 2.355 / dl
    return gaussian_filter1d(flu, sig, mode="nearest")

def rms_bn(mod_lam, mod_flu, obs_lam, obs_flu, gO, wl_lo, wl_hi):
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0: return float("nan")
    mod_flu = mod_flu * (gO/g)
    selO = (obs_lam>=wl_lo) & (obs_lam<=wl_hi)
    ol, of = obs_lam[selO], obs_flu[selO]
    if len(ol) < 10: return float("nan")
    sO = smooth(ol, of)
    selM = (mod_lam>=wl_lo-100) & (mod_lam<=wl_hi+100)
    ml, mf = mod_lam[selM], mod_flu[selM]
    if len(ml) < 10: return float("nan")
    sM = smooth(ml, mf)
    common = (ol>=ml[0]) & (ol<=ml[-1])
    mb = np.interp(ol[common], ml, mf/sM)
    return float(np.sqrt(np.mean((of[common]/sO[common] - mb)**2)))

# --- load LUMINA r070 ---
m = pd.read_csv(f"{ROOT}/logs/w1_156893_w1_r070/lumina_spectrum.csv")
mlam = m["wavelength_angstrom"].values.astype(float)
mflu = m["flux"].values.astype(float)

# --- red-half transformation: cosine ramp 6500->6700, then 0.5 ---
WL_LO, WL_HI, FAC = 6500.0, 6700.0, 0.5
fac = np.ones_like(mlam)
ramp = (mlam >= WL_LO) & (mlam <= WL_HI)
fac[ramp] = 1.0 - (1.0 - FAC) * 0.5 * (1 - np.cos(np.pi * (mlam[ramp] - WL_LO) / (WL_HI - WL_LO)))
fac[mlam > WL_HI] = FAC
mflu_half = mflu * fac

# --- load obs ---
sn = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/sn2011fe_p3d2d.csv", comment="#")
slam = sn["wavelength_angstrom"].values
sflu = sn["flux_erg_s_cm2_angstrom"].values

hst_dir = f"{ROOT}/data/sn2011fe/hst_uv"
parts = []
for grat, w_lo, w_hi in [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]:
    files = sorted(glob.glob(f"{hst_dir}/CCD_{grat}_*{TAG}*sx1.csv"))
    if not files: continue
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0].copy()
    if len(dfs) > 1:
        stack = np.column_stack([
            np.interp(base["wavelength_angstrom"].values, d["wavelength_angstrom"].values,
                      d["flux_erg_s_cm2_angstrom"].values) for d in dfs])
        base["flux_erg_s_cm2_angstrom"] = np.nanmean(stack, axis=1)
    sel = (base["wavelength_angstrom"] >= w_lo) & (base["wavelength_angstrom"] <= w_hi)
    parts.append(base.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom"]])
hst = pd.concat(parts, ignore_index=True).sort_values("wavelength_angstrom").reset_index(drop=True)
hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
hlam = hst["wavelength_angstrom"].values
hflu = hst["flux_erg_s_cm2_angstrom"].values

# --- normalize both LUMINA versions to HST [4500,5800] band ---
gH = band_int(hlam, hflu, 4500, 5800)
gM_orig = band_int(mlam, mflu, 4500, 5800)
gM_half = band_int(mlam, mflu_half, 4500, 5800)
mflu_norm = mflu * (gH / gM_orig)
mflu_half_norm = mflu_half * (gH / gM_half)

# --- figure ---
fig, axes = plt.subplots(3, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [3, 3, 2]})

# Panel 1: full range
ax = axes[0]
ax.plot(hlam, hflu, lw=1.0, color="black", alpha=0.75, label="HST stitched (obs)")
ax.plot(slam, sflu, lw=0.7, color="goldenrod", alpha=0.65, label="Snifs p+3.2d (obs)")
ax.plot(mlam, mflu_norm, lw=0.9, color="crimson", alpha=0.6,
        label="LUMINA r070 original (norm to HST 4500-5800)")
ax.plot(mlam, mflu_half_norm, lw=1.2, color="navy", alpha=0.85,
        label="LUMINA r070 × ramp[6500-6700]→0.5 (norm to HST 4500-5800)")
ax.axvspan(6500, 6700, alpha=0.18, color="orange", label="ramp region 6500-6700Å (×1→×0.5)")
ax.axvspan(6700, 10000, alpha=0.08, color="orange", label="halved region 6700-10000Å (×0.5)")
ax.set_xlim(1600, 10000)
ax.set_ylim(0, max(hflu.max(), sflu.max()) * 1.15)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("LUMINA r070 with artificial red-half transformation — full range")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, alpha=0.25)

# Panel 2: red zoom 6000-10000Å
ax = axes[1]
ax.plot(hlam, hflu, lw=1.3, color="black", alpha=0.85, label="HST obs")
ax.plot(slam, sflu, lw=1.0, color="goldenrod", alpha=0.70, label="Snifs obs")
ax.plot(mlam, mflu_norm, lw=1.0, color="crimson", alpha=0.55, label="LUMINA r070 original")
ax.plot(mlam, mflu_half_norm, lw=1.4, color="navy", alpha=0.85, label="LUMINA r070 × ramp→0.5")
ax.axvspan(6500, 6700, alpha=0.18, color="orange")
ax.axvspan(6700, 10000, alpha=0.08, color="orange")
ax.set_xlim(6000, 10000)
sel = (hlam >= 6000) & (hlam <= 10000)
ax.set_ylim(0, hflu[sel].max() * 1.4)
for xv, lab in [(6355, "Si II"), (7773, "O I"), (8542, "Ca II IR"), (8662, "Ca II IR")]:
    ax.axvline(xv, lw=0.6, color="gray", ls=":")
    ax.text(xv, ax.get_ylim()[1] * 0.97, lab, fontsize=7.5, ha="center", color="gray")
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("Red region 6000-10000Å — ramp transition + halving")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

# Panel 3: baseline-normalized residual full range
ax = axes[2]
slam_sm = smooth(slam, sflu)
hlam_sm = smooth(hlam, hflu)
mlam_sm = smooth(mlam, mflu_norm)
mhalf_sm = smooth(mlam, mflu_half_norm)
ax.plot(hlam, hflu / hlam_sm, lw=0.7, color="black", alpha=0.55, label="HST (bn)")
ax.plot(slam, sflu / slam_sm, lw=0.6, color="goldenrod", alpha=0.45, label="Snifs (bn)")
ax.plot(mlam, mflu_norm / mlam_sm, lw=0.7, color="crimson", alpha=0.55, label="LUMINA r070 orig (bn)")
ax.plot(mlam, mflu_half_norm / mhalf_sm, lw=0.9, color="navy", alpha=0.85, label="LUMINA r070 × ramp (bn)")
ax.set_xlim(1600, 10000)
ax.set_ylim(0, 2.5)
ax.axvspan(6500, 6700, alpha=0.18, color="orange")
ax.axvspan(6700, 10000, alpha=0.08, color="orange")
ax.axhline(1.0, lw=0.5, color="gray", ls="--")
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("flux / baseline (FWHM=20k)")
ax.set_title("Baseline-normalized line shapes — note the ramp folds into baseline trivially")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = f"{ROOT}/figures/2026-05-22_W1_r070_redhalf_vs_obs.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")

# --- recompute scores ---
print()
print("=== scoring (rms_bn, FWHM=20k) ===")
gS = band_int(slam, sflu, 4500, 5800)
for label, lam, flu in [("original ", mlam, mflu), ("red-half ", mlam, mflu_half)]:
    rs = rms_bn(lam, flu, slam, sflu, gS, 3300, 8000)
    rh_can = rms_bn(lam, flu, hlam, hflu, gH, 3000, 8000)
    rh_uv = rms_bn(lam, flu, hlam, hflu, gH, 1700, 3000)
    rh_nir = rms_bn(lam, flu, hlam, hflu, gH, 6500, 9700)
    print(f"  {label}  Snifs[3300,8000]={rs:.4f}  HST[3000,8000]={rh_can:.4f}  HST[1700,3000]={rh_uv:.4f}  HST[6500,9700]={rh_nir:.4f}")

print()
print("=== band-flux ratio (LUMINA / HST integrated) ===")
def br(lam, flu):
    return band_int(lam, flu * (gH / band_int(lam, flu, 4500, 5800)), 6500, 10000) / band_int(hlam, hflu, 6500, 10000)
print(f"  original    red[6500,10000] LUMINA/HST = {br(mlam, mflu):.3f}")
print(f"  red-halved  red[6500,10000] LUMINA/HST = {br(mlam, mflu_half):.3f}")
