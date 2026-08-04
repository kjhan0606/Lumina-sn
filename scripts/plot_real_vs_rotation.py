#!/usr/bin/env python3
"""Overlay real (4pi binning) vs rotation (user's variance-reduction) spectra
for the DDC15 0.976d self-test, scatter and macroatom. DDC15 reference shown
for context. Expectation: for the symmetric DDC15 model rotation ~= real
(mean weight ~0.98), so the two curves should nearly coincide."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
RUNS = {
    "scatter":   "logs/paperDDC15init_ddc15init_scatter_cro0_cfe0_kp0_ste0_rte0_smrotation_163063",
    "macroatom": "logs/paperDDC15init_ddc15init_macroatom_cro0_cfe0_kp0_ste0_rte0_smrotation_163064",
}
WMIN, WMAX = 3000.0, 11000.0
NWIN = (3500.0, 9000.0)


def loadcsv(p):
    d = np.genfromtxt(p, delimiter=",", names=True)
    return d["wavelength_angstrom"], d["flux"]


def norm(l, f):
    m = (l >= NWIN[0]) & (l <= NWIN[1])
    return f / np.trapezoid(f[m], l[m])


lr = np.loadtxt(REF)
lref, fref = lr[:, 0], lr[:, 1]
fref_n = norm(lref, fref)

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
for ax, (mode, d) in zip(axes, RUNS.items()):
    lreal, freal = loadcsv(f"{d}/lumina_spectrum.csv")
    lrot, frot = loadcsv(f"{d}/lumina_spectrum_rotation.csv")
    fr_n = norm(lreal, freal)
    ro_n = norm(lrot, frot)
    ax.plot(lref, fref_n, color="0.55", lw=1.4, label="DDC15 0.976d (ref)")
    ax.plot(lreal, fr_n, color="#3898EC", lw=1.3, label="real (4π binning)")
    ax.plot(lrot, ro_n, color="#D97757", lw=1.3, ls="--", label="rotation (user RT)")
    # band-integrated relative difference rotation vs real
    m = (lreal >= NWIN[0]) & (lreal <= NWIN[1])
    rms = np.sqrt(np.mean((ro_n[m] - fr_n[m]) ** 2)) / np.mean(fr_n[m])
    ax.set_title(f"{mode}  —  rotation vs real RMS diff = {rms*100:.1f}% of mean",
                 fontsize=11)
    ax.set_xlim(WMIN, WMAX)
    ax.set_ylabel("normalized flux")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("wavelength (Å)")
fig.suptitle("DDC15 0.976d self-test: real vs rotation spectrum (symmetric → expect coincidence)",
             fontsize=12)
fig.tight_layout()
out = "figures/2026-06-04_ddc15_real_vs_rotation.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

# numeric band-ratio table
BANDS = [("blue", 3500, 4500), ("green", 4500, 5500),
         ("red", 5500, 7000), ("NIR", 7000, 9000)]


def bandratios(l, f, lr2, fr2):
    def bm(ll, ff, a, b):
        m = (ll >= a) & (ll <= b)
        n = (ll >= NWIN[0]) & (ll <= NWIN[1])
        return np.trapezoid(ff[m], ll[m]) / np.trapezoid(ff[n], ll[n])
    return {n: bm(l, f, a, b) / bm(lr2, fr2, a, b) for n, a, b in BANDS}


for mode, d in RUNS.items():
    lreal, freal = loadcsv(f"{d}/lumina_spectrum.csv")
    lrot, frot = loadcsv(f"{d}/lumina_spectrum_rotation.csv")
    rr = bandratios(lreal, freal, lref, fref)
    ro = bandratios(lrot, frot, lref, fref)
    print(f"\n{mode}: band ratios vs DDC15")
    print(f"  real    : " + "  ".join(f"{n} {rr[n]:.2f}" for n, _, _ in BANDS))
    print(f"  rotation: " + "  ".join(f"{n} {ro[n]:.2f}" for n, _, _ in BANDS))
