#!/usr/bin/env python3
"""
#279 Blondin+2013 DDC15 Fig 6-style overlay for capraise champion 157631.

Plots:
  Top:    SN 2011fe p+3.7d (Snifs) vs LUMINA capraise champion 157631 formal,
          with baseline 157611 also overplotted for visual A/B.
  Bottom: residual (model - obs)/obs in same band, with key SN Ia line IDs
          annotated.

Per-trough gap quantification table printed to stdout.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from score_nw import band_int

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/sn2011fe_p4d2d.csv", comment="#")
obs_wl   = OBS["wavelength_angstrom"].values
obs_flux = OBS["flux_erg_s_cm2_angstrom"].values

BASE = f"{ROOT}/logs/nlteO_recalP_fe1p00_co0p3_niuv0p3_n2o0p05_p+3.7d_157611/lumina_spectrum_formal.csv"
CAP  = f"{ROOT}/logs/nlteO_recalP_fe1p0_co0p3_niuv0p3_n2o0p05_ni30p0_p+3.7d_157631/lumina_spectrum_formal.csv"

def load_mod(path):
    sp = pd.read_csv(path)
    wl = sp.iloc[:,0].values
    fl = sp.iloc[:,1].values
    # rescale to green-band integral of obs (4500-5800)
    gO = band_int(obs_wl, obs_flux, 4500, 5800)
    gM = band_int(wl, fl, 4500, 5800)
    return wl, fl * (gO/gM)

w_b, f_b = load_mod(BASE)
w_c, f_c = load_mod(CAP)

# Trough line IDs (SN Ia, p+3.7d): ion, lambda_rest_A, lambda_obs_A (after blueshift), label
# v_phot ~ 11500 km/s at p+3.7d for LUMINA Phase 9b cell
LINES = [
    ("Ca II H&K",   "Ca II",  3950, 3700),
    ("S II W blue", "S II",   5454, 5340),
    ("S II W red",  "S II",   5640, 5520),
    ("Si II 5972",  "Si II",  5972, 5850),
    ("Si II 6355",  "Si II",  6355, 6225),
    ("O I 7774",    "O I",    7774, 7620),
    ("Mg II 4481",  "Mg II",  4481, 4395),
    ("Fe II 5169",  "Fe II",  5169, 5070),
    ("Ca II IR",    "Ca II",  8540, 8370),
]

# ============ Plot ============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5),
                                gridspec_kw=dict(height_ratios=[2.4, 1.0]),
                                sharex=True)

# Top: spectra overlay
mask_obs = (obs_wl >= 3300) & (obs_wl <= 9700)
ax1.plot(obs_wl[mask_obs], obs_flux[mask_obs], color="k", lw=1.0,
         label="SN 2011fe p+3.7d (Snifs)", zorder=3)

m_b = (w_b >= 3300) & (w_b <= 9700)
ax1.plot(w_b[m_b], f_b[m_b], color="#888", lw=1.0, alpha=0.7,
         label="LUMINA baseline (Fe III 800 cap, RMS_bn=0.228)", zorder=2)

m_c = (w_c >= 3300) & (w_c <= 9700)
ax1.plot(w_c[m_c], f_c[m_c], color="#D97757", lw=1.2,
         label="LUMINA capraise (Co III 3917 lev, RMS_bn=0.191)", zorder=4)

# Annotate troughs
ymin, ymax = ax1.get_ylim()
y_label = ymax * 0.95
for name, ion, lam_rest, lam_obs in LINES:
    ax1.axvline(lam_obs, color="#3898EC", ls=":", lw=0.7, alpha=0.5)
    ax1.text(lam_obs, y_label, name, rotation=90, fontsize=8,
             color="#3898EC", ha="right", va="top", alpha=0.85)

ax1.set_ylabel(r"$F_\lambda$ [erg/s/cm$^2$/Å] (green-band normalized)")
ax1.set_title("Blondin+2013 Fig 6-style overlay — LUMINA capraise vs SN 2011fe p+3.7d (DDC15-like base)")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)

# Bottom: residual (capraise - obs)/obs
obs_interp_c = np.interp(w_c[m_c], obs_wl, obs_flux)
res_c = (f_c[m_c] - obs_interp_c) / np.maximum(obs_interp_c, 1e-30)
obs_interp_b = np.interp(w_b[m_b], obs_wl, obs_flux)
res_b = (f_b[m_b] - obs_interp_b) / np.maximum(obs_interp_b, 1e-30)

ax2.axhline(0, color="k", lw=0.8)
ax2.plot(w_b[m_b], res_b, color="#888", lw=0.7, alpha=0.6, label="baseline residual")
ax2.plot(w_c[m_c], res_c, color="#D97757", lw=0.9, label="capraise residual")
ax2.set_ylim(-1.0, 1.5)
ax2.set_xlim(3300, 9700)
ax2.set_xlabel(r"Wavelength [Å]")
ax2.set_ylabel("(model − obs) / obs")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.25)

# Per-trough gap table (printed)
print(f"{'='*78}")
print(f"{'Trough':<14} {'λ_obs':>6}   {'<obs>':>8}  {'<base>':>8}  {'<cap>':>8}  {'Δbase':>7}  {'Δcap':>7}")
print(f"{'-'*78}")
def band_mean(wl, fl, lam, half=50):
    m = (wl >= lam-half) & (wl <= lam+half)
    return float(fl[m].mean()) if m.any() else float("nan")
for name, ion, lam_rest, lam_obs in LINES:
    o = band_mean(obs_wl, obs_flux, lam_obs)
    b = band_mean(w_b, f_b, lam_obs)
    c = band_mean(w_c, f_c, lam_obs)
    db = (b-o)/o if o else float("nan")
    dc = (c-o)/o if o else float("nan")
    print(f"{name:<14} {lam_obs:>6}   {o:>.2e}  {b:>.2e}  {c:>.2e}  {db:>+7.3f}  {dc:>+7.3f}")
print(f"{'='*78}")

out = f"{ROOT}/figures/paper_repro/fig6_capraise_vs_sn2011fe_p+3.7d.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"\nwrote {out}")
