#!/usr/bin/env python3
"""O-fix A/B overlay vs SN 2002bo B-max.
160663 = partial O fix (O I 483x over, O II collapsed); 160756 = full fix (O I/II/III = 1.0).
Both Sobolev formal spectra, anchor-normalized on [4000,6000]A against dereddened obs.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUNS = [("160663", "partial O-fix (O I 483x)", "darkorange"),
        ("160756", "full O-fix (O I/II/III=1.0)", "navy")]
MU, EBV, RV = 31.90, 0.41, 3.1
A_V = RV * EBV

def ccm_a_over_av(wave_aa):
    x = 1e4 / wave_aa
    a = np.zeros_like(x); b = np.zeros_like(x)
    sel = (x >= 1.1) & (x <= 3.3); y = x[sel] - 1.82
    a[sel] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[sel] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    sel = (x >= 0.3) & (x < 1.1)
    a[sel] = 0.574 * x[sel]**1.61; b[sel] = -0.527 * x[sel]**1.61
    sel = (x > 3.3) & (x <= 8.0); xs = x[sel]
    Fa = np.where(xs >= 5.9, -0.04473*(xs-5.9)**2 - 0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs >= 5.9, 0.2130*(xs-5.9)**2 + 0.1207*(xs-5.9)**3, 0.0)
    a[sel] = 1.752 - 0.316*xs - 0.104/((xs-4.67)**2 + 0.341) + Fa
    b[sel] = -3.090 + 1.825*xs + 1.206/((xs-4.62)**2 + 0.263) + Fb
    return a + b / RV

def deredden(w, f): return f * 10**(0.4 * A_V * ccm_a_over_av(w))

obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = deredden(olam, obs["flux_erg_s_cm2_angstrom"].values)
ALO, AHI = 4000., 6000.
I_obs = float(np.trapezoid(oflu[(olam>=ALO)&(olam<=AHI)], olam[(olam>=ALO)&(olam<=AHI)]))

def load(job):
    p = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_nltedump_{job}/lumina_spectrum_formal.csv"
    m = pd.read_csv(p); lam = m["wavelength_angstrom"].values; fr = m["flux"].values
    sa = (lam>=ALO)&(lam<=AHI)
    K = I_obs / float(np.trapezoid(fr[sa], lam[sa]))
    return lam, fr*K

fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios":[3,2]})
ax = axes[0]
ax.plot(olam, oflu, lw=1.0, color="black", alpha=0.85, label="SN 2002bo dereddened")
data = {}
for job, lbl, c in RUNS:
    lam, flu = load(job); data[job] = (lam, flu)
    ax.plot(lam, flu, lw=1.0, color=c, alpha=0.85, label=lbl)
ax.set_xlim(1500,10500); ax.set_xlabel("Wavelength [A]"); ax.set_ylabel("F_lambda [erg/s/cm2/A]")
ax.set_title("O-fix A/B vs SN 2002bo B-max (anchor [4000,6000]A pinned)")
ax.legend(loc="upper right", fontsize=10); ax.grid(True, alpha=0.25)

ax = axes[1]
ax.axhline(1.0, lw=1.0, color="black", ls="--", alpha=0.6)
for job, lbl, c in RUNS:
    lam, flu = data[job]
    cm = (olam>=lam[0])&(olam<=lam[-1])
    r = np.interp(olam[cm], lam, flu) / np.maximum(oflu[cm], 1e-30)
    ax.semilogy(olam[cm], r, lw=0.9, color=c, alpha=0.85, label=f"{lbl} / obs")
for h in (2.0,0.5): ax.axhline(h, lw=0.5, color="orange", ls=":", alpha=0.4)
ax.set_xlim(1500,10500); ax.set_ylim(0.05,50); ax.set_xlabel("Wavelength [A]"); ax.set_ylabel("model/obs (log)")
ax.set_title("model/obs ratio"); ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
out = f"{ROOT}/figures/2026-05-30_ofix_ab_160663_vs_160756_sn2002bo_bmax.png"
plt.savefig(out, dpi=130); print(f"saved: {out}")
