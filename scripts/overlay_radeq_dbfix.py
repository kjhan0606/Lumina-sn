#!/usr/bin/env python3
"""3-way overlay vs SN2002bo (dereddened): radeq-OFF keeper (161737) vs old
cooling-fix radeq-ON that oscillated/ended-hot (161796) vs detailed-balance bf
fix radeq-ON (161811). Tests the FUV<3400 hypothesis: radeq's net spectral effect
should be confined to <3400A (over-ionization band); >3400 ~ radeq-insensitive.
Each model optical-anchor-pinned on [4000,6000]A (same as the scorer)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
EBV, RV = 0.41, 3.1
A_V = RV * EBV

def ccm(wave_aa):
    x = 1e4 / wave_aa
    a = np.zeros_like(x); b = np.zeros_like(x)
    s = (x >= 1.1) & (x <= 3.3); y = x[s]-1.82
    a[s] = 1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b[s] = 1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    s = (x >= 0.3) & (x < 1.1)
    a[s] = 0.574*x[s]**1.61; b[s] = -0.527*x[s]**1.61
    s = (x > 3.3) & (x <= 8.0); xs = x[s]
    Fa = np.where(xs>=5.9, -0.04473*(xs-5.9)**2-0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs>=5.9,  0.2130*(xs-5.9)**2+0.1207*(xs-5.9)**3, 0.0)
    a[s] = 1.752-0.316*xs-0.104/((xs-4.67)**2+0.341)+Fa
    b[s] = -3.090+1.825*xs+1.206/((xs-4.62)**2+0.263)+Fb
    return a + b/RV

obs = pd.read_csv(ROOT/"data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = obs["flux_erg_s_cm2_angstrom"].values * 10**(0.4*A_V*ccm(obs["wavelength_angstrom"].values))

def load_pinned(job, label):
    wd = sorted(ROOT.glob(f"logs/*_{job}"))[-1]
    m = pd.read_csv(wd/"lumina_spectrum_formal.csv")
    lam = m["wavelength_angstrom"].values; flu = m["flux"].values
    sa = (lam>=4000)&(lam<=6000); oa=(olam>=4000)&(olam<=6000)
    K = np.trapezoid(oflu[oa],olam[oa])/np.trapezoid(flu[sa],lam[sa])
    return lam, flu*K, label

runs = [load_pinned("161737","radeq OFF keeper (161737)"),
        load_pinned("161796","coolfix radeq ON, ended HOT (161796)"),
        load_pinned("161811","detailed-balance bf fix (161811)")]

# --- Quantitative FUV<3400 vs >3400 band test (model/obs integrated ratio) ---
def band_ratio(lam, flu, lo, hi):
    sa=(lam>=lo)&(lam<=hi); oa=(olam>=lo)&(olam<=hi)
    if sa.sum()<2 or oa.sum()<2: return float("nan")
    return np.trapezoid(flu[sa],lam[sa])/np.trapezoid(oflu[oa],olam[oa])
print("band model/obs ratios (1.0 = perfect):")
print(f"{'job':36s} {'FUV<3400':>9s} {'3400-6000':>10s} {'>6000':>8s}")
for lam,flu,lab in runs:
    print(f"{lab:36s} {band_ratio(lam,flu,1800,3400):9.3f} "
          f"{band_ratio(lam,flu,3400,6000):10.3f} {band_ratio(lam,flu,6000,10000):8.3f}")

fig, ax = plt.subplots(figsize=(13,6.5))
ax.plot(olam, oflu, color="black", lw=1.0, alpha=0.85, label="SN 2002bo (dereddened, B-max)")
for lam,flu,lab in runs:
    ax.plot(lam, flu, lw=1.1, alpha=0.9, label=lab)
for lo,hi,name in [(1800,3400,"FUV (radeq-sensitive)"),(3500,5500,"P-Cygni band"),(9000,10200,"NIR II")]:
    ax.axvspan(lo,hi,color="gold",alpha=0.10)
    ax.text((lo+hi)/2, 0, name, ha="center", va="bottom", fontsize=8, color="darkgoldenrod")
ax.axvline(3400, color="gray", ls=":", lw=0.8)
ax.set_xlim(1800,10200); ax.set_ylim(0, None)
ax.set_xlabel("wavelength (Å)"); ax.set_ylabel("F_λ (optical-anchored)")
ax.set_title("radeq detailed-balance bf fix (161811) vs coolfix-hot (161796) vs radeq-OFF (161737) — SN 2002bo")
ax.legend(fontsize=9, loc="upper right")
out = ROOT/"figures/2026-06-01_radeq_dbfix_3way_sn2002bo.png"
out.parent.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("saved:", out)
