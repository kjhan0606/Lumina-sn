#!/usr/bin/env python3
"""LUMINA runs vs dereddened SN 2002bo at Bmax, optical-anchor pinned [4000,6000]Å.
Annotates the user's feature defect map."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
EBV, RV = 0.41, 3.1
A_V = RV * EBV

def ccm_a_over_av(wave_aa):
    x = 1e4 / wave_aa; a = np.zeros_like(x); b = np.zeros_like(x)
    sel = (x >= 1.1) & (x <= 3.3); y = x[sel] - 1.82
    a[sel] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[sel] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    sel = (x >= 0.3) & (x < 1.1)
    a[sel] = 0.574 * x[sel]**1.61; b[sel] = -0.527 * x[sel]**1.61
    sel = (x > 3.3) & (x <= 8.0); xs = x[sel]
    Fa = np.where(xs >= 5.9, -0.04473*(xs-5.9)**2 - 0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs >= 5.9,  0.2130*(xs-5.9)**2 + 0.1207*(xs-5.9)**3, 0.0)
    a[sel] = 1.752 - 0.316*xs - 0.104/((xs-4.67)**2+0.341) + Fa
    b[sel] = -3.090 + 1.825*xs + 1.206/((xs-4.62)**2+0.263) + Fb
    return a + b/RV

def deredden(w, f): return f * 10**(0.4 * A_V * ccm_a_over_av(w))

# obs
obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = deredden(olam, obs["flux_erg_s_cm2_angstrom"].values)

runs = {
    "LUMINA radeq-off (161362)": (glob.glob(f"{ROOT}/logs/*radeq_off_161362")[0], "#3898EC"),
    "LUMINA super-level (161350)": (glob.glob(f"{ROOT}/logs/*superlev_ab_super_161350")[0], "#4EC9B0"),
}

def anchor_scale(mlam, mflu_raw):
    sm = (mlam >= 4000) & (mlam <= 6000); so = (olam >= 4000) & (olam <= 6000)
    K = np.trapezoid(oflu[so], olam[so]) / np.trapezoid(mflu_raw[sm], mlam[sm])
    return mflu_raw * K

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(olam, oflu, color="black", lw=1.3, alpha=0.9, label="SN 2002bo (dereddened, Bmax)", zorder=5)
for lab, (run, c) in runs.items():
    m = pd.read_csv(f"{run}/lumina_spectrum_formal.csv")
    ml = m["wavelength_angstrom"].values
    mf = anchor_scale(ml, m["flux"].values)
    sel = (ml >= 2900) & (ml <= 10300)
    ax.plot(ml[sel], mf[sel], color=c, lw=0.9, alpha=0.85, label=lab)

# user defect annotations
feats = [
    (3650, "3500-3800\nover-abs", "v"),
    (3800, "3800 dip\n+60%", "^"),
    (3900, "3900 pk\n+40%", "^"),
    (4100, "4100\npk->DIP", "x"),
    (4700, "4700 pk\n+70%", "^"),
    (5100, "5100 pk\nMISSING", "x"),
    (5600, "5600 pk\n-50%", "v"),
    (5900, "5900 pk\n-30%", "v"),
    (6355, "Si II\nOK", "o"),
    (7600, "6500-8800\n+50%", "^"),
    (8200, "8200 dip\nOK", "o"),
    (9200, "8300-10000\n+20%", "^"),
]
ymax = np.nanmax(oflu[(olam>=4000)&(olam<=6000)]) * 1.15
for wl, txt, mk in feats:
    ax.axvline(wl, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.annotate(txt, xy=(wl, ymax*0.93), fontsize=6.5, ha="center", va="top",
                color="#D97757" if mk in ("x","^","v") else "#2E7D32")

ax.set_xlim(2900, 10300)
ax.set_ylim(0, ymax)
ax.set_xlabel("wavelength (Å)")
ax.set_ylabel(r"F$_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$, anchor-pinned [4000,6000])")
ax.set_title("LUMINA DDC15 vs SN 2002bo (Bmax) — user defect map annotated")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.15)
out = f"{ROOT}/figures/lumina_vs_2002bo_obs_161362_161350.png"
plt.tight_layout(); plt.savefig(out, dpi=140); print("saved:", out)
