#!/usr/bin/env python3
"""Band-by-band decomposition of obs emergent vs gold (user's 2026-06-25 read):
blue absorption deficit + green emission deficit = ONE fluorescence mechanism (blue->green)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gold = np.genfromtxt("data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat")
gl, gf = gold[:, 0], gold[:, 1]
obs = np.genfromtxt("logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169874/lumina_spectrum_freqres_obs.csv",
                    delimiter=",", names=True)
ol, of = obs["wavelength_angstrom"], obs["flux"]

def N(lam, flux, lo=4000, hi=8000):
    m = (lam >= lo) & (lam <= hi); return flux / np.trapezoid(flux[m], lam[m])
gN, oN = N(gl, gf), N(ol, of)

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.plot(gl, gN, color="#FFC107", lw=2.3, label="DDC15 GOLD", zorder=3)
ax.plot(ol, oN, color="#D97757", lw=1.8, label="LUMINA obs Doppler march", zorder=3)

bands = [("blue\nABSORB too low\n6.5x excess", 3000, 5000, "#3898EC"),
         ("green\nEMIT too low\n0.57x deficit", 5600, 7300, "#4EC9B0"),
         ("7700 dip\ntoo shallow\n(0.44 vs 0.24)", 7600, 7900, "#9B59B6"),
         ("NIR\n~matches\n0.94x", 8500, 9500, "#7F8C8D")]
ymax = max(gN.max(), oN.max()) * 1.05
for nm, lo, hi, c in bands:
    ax.axvspan(lo, hi, color=c, alpha=0.13, zorder=1)
    ax.text((lo+hi)/2, ymax*0.80, nm, ha="center", va="top", fontsize=8.5,
            color=c, weight="bold", zorder=4)

# annotate the blue->green redistribution arrow
ax.annotate("", xy=(6400, ymax*0.55), xytext=(4000, ymax*0.55),
            arrowprops=dict(arrowstyle="->", color="black", lw=2, alpha=0.6))
ax.text(5200, ymax*0.58, "fluorescence: blue→green\n(MISSING in model)", ha="center",
        fontsize=9, style="italic", zorder=4)

ax.set_xlabel("wavelength (A)"); ax.set_ylabel("flux (normalized 4000-8000A)")
ax.set_title("It is NOT a uniform red shift — it is localized feature errors (user decomposition, 2026-06-25)\n"
             "blue excess + green deficit = ONE missing mechanism (blue→green fluorescence); NIR scatter already OK")
ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=0.3)
ax.set_xlim(3000, 12000); ax.set_ylim(0, ymax)
fig.tight_layout()
out = "figures/2026-06-25_band_decomposition_vs_gold.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print("wrote", out)
