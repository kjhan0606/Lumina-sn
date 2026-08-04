#!/usr/bin/env python3
"""Doppler obs emergent (with P-Cygni structure) vs gold, AND the featureless static one,
so the honest state is clear: obs march HAS features (too-red); static is smooth color only."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gold = np.genfromtxt("data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat")
gl, gf = gold[:, 0], gold[:, 1]
obs = np.genfromtxt("logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169874/lumina_spectrum_freqres_obs.csv",
                    delimiter=",", names=True)
ol, of = obs["wavelength_angstrom"], obs["flux"]
stat = np.genfromtxt("logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169758/lumina_spectrum_freqres.csv",
                     delimiter=",", names=True)
sl, sf = stat["wavelength_angstrom"], stat["flux"]

def normed(lam, flux, lo=4000, hi=8000):
    m = (lam >= lo) & (lam <= hi)
    return flux / np.trapezoid(flux[m], lam[m])

def peak(lam, flux, lo=3000, hi=12000):
    m = (lam >= lo) & (lam <= hi)
    return lam[m][np.argmax(flux[m])]

fig, ax = plt.subplots(figsize=(12.5, 6.2))
ax.plot(gl, normed(gl, gf), color="#FFC107", lw=2.2, label=f"DDC15 GOLD (rich P-Cygni, peak {peak(gl,gf):.0f}A)")
ax.plot(ol, normed(ol, of), color="#D97757", lw=1.7,
        label=f"LUMINA obs Doppler march (HAS P-Cygni, too-red peak {peak(ol,of):.0f}A)")
ax.plot(sl, normed(sl, sf), color="#3898EC", lw=1.3, ls="--", alpha=0.7,
        label=f"LUMINA static freqres (FEATURELESS, color only, peak {peak(sl,sf):.0f}A)")
ax.set_xlabel("wavelength (A)"); ax.set_ylabel("flux (normalized 4000-8000A)")
ax.set_title("Deterministic emergent — honest state (2026-06-25)\n"
             "obs march DOES produce P-Cygni features but too-red; static = smooth color only; both lack fluorescence green")
ax.legend(loc="upper right", fontsize=9.5); ax.grid(alpha=0.3); ax.set_xlim(3000, 12000)
ax.set_ylim(0, None)
fig.tight_layout()
out = "figures/2026-06-25_obs_emergent_vs_gold.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print("wrote", out)
for nm, lam, flx in [("GOLD", gl, gf), ("obs-Doppler", ol, of), ("static", sl, sf)]:
    print(f"{nm:12s} peak {peak(lam,flx):.0f}A")
