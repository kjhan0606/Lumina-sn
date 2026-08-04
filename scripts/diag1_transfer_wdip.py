#!/usr/bin/env python3
"""Diagnostic (1): does the line-interaction mode (scatter vs downbranch vs
macroatom) carve the W-troughs that scatter fills?

Compares the MC spectrum (lumina_spectrum.csv) of the keeper scatter run
(161737) vs downbranch (161865) vs macroatom (161866), all keeper config
(superlev_ionfix + wcap + radeq-OFF). Dereddened SN2002bo overlaid; each model
optical-anchored on [4000,6000]. Reports trough depth (model/obs) at the W-dip
minima 4896/5307/5474 and the 5600-5900 deficit band.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
EBV, RV = 0.41, 3.1; A_V = RV * EBV

def ccm(wave_aa):
    x = 1e4 / wave_aa; a = np.zeros_like(x); b = np.zeros_like(x)
    s = (x >= 1.1) & (x <= 3.3); y = x[s]-1.82
    a[s] = 1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b[s] = 1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    s = (x >= 0.3) & (x < 1.1); a[s] = 0.574*x[s]**1.61; b[s] = -0.527*x[s]**1.61
    s = (x > 3.3) & (x <= 8.0); xs = x[s]
    Fa = np.where(xs>=5.9, -0.04473*(xs-5.9)**2-0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs>=5.9,  0.2130*(xs-5.9)**2+0.1207*(xs-5.9)**3, 0.0)
    a[s] = 1.752-0.316*xs-0.104/((xs-4.67)**2+0.341)+Fa
    b[s] = -3.090+1.825*xs+1.206/((xs-4.62)**2+0.263)+Fb
    return a + b/RV

obs = pd.read_csv(ROOT/"data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = obs["flux_erg_s_cm2_angstrom"].values * 10**(0.4*A_V*ccm(olam))

def load(job, label, which="lumina_spectrum.csv"):
    wd = sorted(ROOT.glob(f"logs/*_{job}"))[-1]
    m = pd.read_csv(wd/which)
    lam = m["wavelength_angstrom"].values; flu = m["flux"].values
    sa = (lam>=4000)&(lam<=6000); oa=(olam>=4000)&(olam<=6000)
    K = np.trapezoid(oflu[oa],olam[oa])/np.trapezoid(flu[sa],lam[sa])
    return lam, flu*K, label

runs = [load("161737","scatter (keeper 161737)"),
        load("161865","downbranch (161865)"),
        load("161866","macroatom (161866)")]

def at(lam, flu, w):  # model/obs flux ratio at wavelength w (interp)
    return np.interp(w, lam, flu) / np.interp(w, olam, oflu)
def band(lam, flu, lo, hi):
    sa=(lam>=lo)&(lam<=hi); oa=(olam>=lo)&(olam<=hi)
    return np.trapezoid(flu[sa],lam[sa])/np.trapezoid(oflu[oa],olam[oa])

print(f"{'run':30s} {'4896':>7s} {'5307':>7s} {'5474':>7s} {'5600-5900':>10s}")
print("(model/obs; <1 = trough present like obs, >1 = filled/over)")
for lam,flu,lab in runs:
    print(f"{lab:30s} {at(lam,flu,4896):7.2f} {at(lam,flu,5307):7.2f} "
          f"{at(lam,flu,5474):7.2f} {band(lam,flu,5600,5900):10.2f}")

fig, ax = plt.subplots(figsize=(13,6))
ax.plot(olam, oflu, color="black", lw=1.1, label="SN 2002bo (dereddened, B-max)")
for lam,flu,lab in runs: ax.plot(lam, flu, lw=1.0, alpha=0.9, label=lab)
for w in (4896,5307,5474): ax.axvline(w, color="gray", ls=":", lw=0.7)
ax.axvspan(5600,5900,color="gold",alpha=0.10)
ax.set_xlim(4500,6000); ax.set_ylim(0,None)
ax.set_xlabel("wavelength (Å)"); ax.set_ylabel("F_λ (optical-anchored)")
ax.set_title("W-trough: line-interaction mode (scatter/downbranch/macroatom) — SN 2002bo")
ax.legend(fontsize=9)
out = ROOT/"figures/2026-06-01_wdip_transfer_modes_sn2002bo.png"
out.parent.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=130)
print("saved:", out)
