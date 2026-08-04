#!/usr/bin/env python3
"""Current best deterministic emergent vs DDC15 gold (2026-06-25).
static-freqres = converged thermal champion (169758); obs = Doppler P-Cygni march (169882 CONTONLY)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gold = np.genfromtxt("data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat")
gl, gf = gold[:, 0], gold[:, 1]

stat = np.genfromtxt("logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169758/lumina_spectrum_freqres.csv",
                     delimiter=",", names=True)
sl, sf = stat["wavelength_angstrom"], stat["flux"]

def band(lam, flux, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    return np.trapz(flux[m], lam[m])

def grnnir(lam, flux):
    return band(lam, flux, 5000, 7000) / band(lam, flux, 7000, 12000)

def norm(lam, flux, lo=4000, hi=8000):
    m = (lam >= lo) & (lam <= hi)
    return flux / np.trapz(flux[m], lam[m])

m = (sl >= 3000) & (sl <= 12000)
mg = (gl >= 3000) & (gl <= 12000)
sln, sfn = sl[m], norm(sl, sf)[m]
gln, gfn = gl[mg], norm(gl, gf)[mg]

peak_s = sln[np.argmax(sfn)]
peak_g = gln[np.argmax(gfn)]
gr_s, gr_g = grnnir(sl, sf), grnnir(gl, gf)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(gln, gfn, color="#FFC107", lw=2.2, label=f"DDC15 GOLD (peak {peak_g:.0f}A, grn/nir {gr_g:.2f})")
ax.plot(sln, sfn, color="#3898EC", lw=1.6, alpha=0.9,
        label=f"LUMINA deterministic emergent (peak {peak_s:.0f}A, grn/nir {gr_s:.2f})")
ax.axvspan(5000, 7000, color="#4EC9B0", alpha=0.08)
ax.axvspan(7000, 12000, color="#D97757", alpha=0.06)
ax.text(6000, ax.get_ylim()[1]*0.92, "green\n5000-7000", ha="center", fontsize=8, color="#2a8")
ax.text(9000, ax.get_ylim()[1]*0.92, "NIR 7000-12000", ha="center", fontsize=8, color="#a64")
ax.set_xlabel("wavelength (A)"); ax.set_ylabel("flux (normalized 4000-8000A)")
ax.set_title("Current deterministic emergent vs DDC15 gold (0.976d)\n"
             "COLOR fixed (peak ~6785 vs gold ~6610); remaining gap = green deficit (fluorescence, in progress)")
ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=0.3); ax.set_xlim(3000, 12000)
fig.tight_layout()
out = "figures/2026-06-25_current_emergent_vs_gold.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print("wrote", out)
print(f"GOLD  : peak {peak_g:.0f}A  grn/nir {gr_g:.3f}")
print(f"MODEL : peak {peak_s:.0f}A  grn/nir {gr_s:.3f}")
