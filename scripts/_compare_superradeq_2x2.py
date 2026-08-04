#!/usr/bin/env python3
"""Overlay the super x radeq 2x2 (N_ITER=12) vs dereddened SN 2002bo.
Reuses the score_blondin deredden + [4000,6000] anchor so every curve is on
the identical physical scale."""
import glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
MU, EBV, RV = 31.90, 0.41, 3.1
A_V = RV * EBV

def ccm_a_over_av(w):
    x = 1e4 / w; a = np.zeros_like(x); b = np.zeros_like(x)
    s = (x >= 1.1) & (x <= 3.3); y = x[s] - 1.82
    a[s] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[s] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    s = (x >= 0.3) & (x < 1.1); a[s] = 0.574*x[s]**1.61; b[s] = -0.527*x[s]**1.61
    s = (x > 3.3) & (x <= 8.0); xs = x[s]
    Fa = np.where(xs >= 5.9, -0.04473*(xs-5.9)**2 - 0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs >= 5.9,  0.2130*(xs-5.9)**2 + 0.1207*(xs-5.9)**3, 0.0)
    a[s] = 1.752 - 0.316*xs - 0.104/((xs-4.67)**2 + 0.341) + Fa
    b[s] = -3.090 + 1.825*xs + 1.206/((xs-4.62)**2 + 0.263) + Fb
    return a + b/RV

def deredden(w, f): return f * 10**(0.4 * A_V * ccm_a_over_av(w))

obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = deredden(olam, obs["flux_erg_s_cm2_angstrom"].values)

ALO, AHI = 4000., 6000.
I_obs_a = float(np.trapezoid(oflu[(olam>=ALO)&(olam<=AHI)], olam[(olam>=ALO)&(olam<=AHI)]))

RUNS = [
    (161427, "radeq_off_n12",        "trunc, radeq OFF",  "#1f77b4", "|F-Q|=0.291 Werr=151%"),
    (161421, "radeq_milne_n12",      "trunc, radeq ON",   "#2ca02c", "|F-Q|=0.144 Werr=153%"),
    (161616, "superSE_n12",          "super, radeq OFF",  "#ff7f0e", "|F-Q|=0.161 Werr=412%"),
    (161617, "superSE_radeq_n12",    "super, radeq ON",   "#d62728", "|F-Q|=0.068 Werr=574%"),
]

curves = []
for job, tag, lbl, col, ann in RUNS:
    cand = sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{tag}_{job}"))
    m = pd.read_csv(f"{cand[0]}/lumina_spectrum_formal.csv")
    mlam = m["wavelength_angstrom"].values; mraw = m["flux"].values
    K = I_obs_a / float(np.trapezoid(mraw[(mlam>=ALO)&(mlam<=AHI)], mlam[(mlam>=ALO)&(mlam<=AHI)]))
    curves.append((mlam, mraw*K, lbl, col, ann))

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios":[3, 2]})

ax = axes[0]
ax.plot(olam, oflu, lw=1.6, color="black", alpha=0.9, label="SN 2002bo (dereddened, B-max)")
for mlam, mflu, lbl, col, ann in curves:
    s = (mlam >= 1500) & (mlam <= 10500)
    ax.plot(mlam[s], mflu[s], lw=1.0, color=col, alpha=0.85, label=f"{lbl}  [{ann}]")
ax.axvspan(ALO, AHI, color="grey", alpha=0.10)
ax.text(5000, ax.get_ylim()[1]*0.92, "anchor\n[4000,6000]", ha="center", fontsize=8, color="grey")
ax.set_xlim(1500, 10500); ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("F_λ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("super-level × radeq 2×2 (N_ITER=12) vs SN 2002bo — anchor-pinned [4000,6000]Å")
ax.legend(loc="upper right", fontsize=8.5); ax.grid(True, alpha=0.25)

ax = axes[1]
ax.axhline(1.0, lw=1.0, color="black", ls="--", alpha=0.6)
ax.axhline(2.0, lw=0.5, color="orange", ls=":", alpha=0.5)
ax.axhline(0.5, lw=0.5, color="orange", ls=":", alpha=0.5)
common = (olam >= 3356) & (olam <= 10200)
for mlam, mflu, lbl, col, ann in curves:
    mi = np.interp(olam[common], mlam, mflu)
    ax.semilogy(olam[common], mi/np.maximum(oflu[common],1e-30), lw=1.0, color=col, alpha=0.85, label=lbl)
for lo, hi, nm in [(3000,4000,"UV/bl"),(4000,5500,"bl/gr"),(5500,7000,"red"),(7000,9000,"NIR I"),(9000,10200,"NIR II")]:
    ax.axvline(hi, lw=0.4, color="grey", ls=":", alpha=0.4)
ax.set_xlim(1500, 10500); ax.set_ylim(0.1, 6)
ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("model / obs (log)")
ax.set_title("model/obs ratio — over-flux (>1) shifts blue→red as super-levels add blanketing; radeq lowers continuum")
ax.legend(loc="upper left", fontsize=8.5); ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
out = f"{ROOT}/figures/2026-06-01_superlevel_x_radeq_2x2_vs_sn2002bo.png"
plt.savefig(out, dpi=140)
print(f"saved: {out}")
