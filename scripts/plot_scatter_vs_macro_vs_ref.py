#!/usr/bin/env python3
"""Overlay LUMINA scatter and macroatom spectra against the DDC15 0.976d
reference (the self-test 'observation'). Both LUMINA spectra use real-mode
binning from the rotation run dirs. Normalized over [3500,9000] A."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
SCAT = "logs/paperDDC15init_ddc15init_scatter_cro0_cfe0_kp0_ste0_rte0_smrotation_163063/lumina_spectrum.csv"
MACR = "logs/paperDDC15init_ddc15init_macroatom_cro0_cfe0_kp0_ste0_rte0_smrotation_163064/lumina_spectrum.csv"
NWIN = (3500.0, 9000.0)
WMIN, WMAX = 3000.0, 11000.0


def loadcsv(p):
    d = np.genfromtxt(p, delimiter=",", names=True)
    return d["wavelength_angstrom"], d["flux"]


def norm(l, f):
    m = (l >= NWIN[0]) & (l <= NWIN[1])
    return f / np.trapezoid(f[m], l[m])


def peak(l, f, a=4000, b=8000):
    m = (l >= a) & (l <= b)
    return l[m][np.argmax(f[m])]


lr = np.loadtxt(REF)
lref, fref = lr[:, 0], lr[:, 1]
ls, fs = loadcsv(SCAT)
lm, fm = loadcsv(MACR)
fref_n, fs_n, fm_n = norm(lref, fref), norm(ls, fs), norm(lm, fm)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lref, fref_n, color="0.35", lw=2.0, label=f"DDC15 0.976d (ref)  peak {peak(lref,fref):.0f}A")
ax.plot(ls, fs_n, color="#3898EC", lw=1.4, label=f"LUMINA scatter  peak {peak(ls,fs):.0f}A")
ax.plot(lm, fm_n, color="#D97757", lw=1.4, label=f"LUMINA macroatom  peak {peak(lm,fm):.0f}A")
for a, b, nm in [(3500, 4500, "blue"), (4500, 5500, "green"),
                 (5500, 7000, "red"), (7000, 9000, "NIR")]:
    ax.axvspan(a, b, color="0.9", alpha=0.15)
ax.axvline(peak(lref, fref), color="0.35", ls=":", lw=1)
ax.set_xlim(WMIN, WMAX)
ax.set_ylim(0, None)
ax.set_xlabel("wavelength (A)")
ax.set_ylabel("normalized flux")
ax.set_title("DDC15 0.976d self-test: scatter vs macroatom vs reference 'observation'")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
out = "figures/2026-06-04_ddc15_scatter_vs_macro_vs_ref.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

BANDS = [("blue", 3500, 4500), ("green", 4500, 5500),
         ("red", 5500, 7000), ("NIR", 7000, 9000)]


def bandratio(l, f, name, a, b):
    m = (l >= a) & (l <= b)
    n = (l >= NWIN[0]) & (l <= NWIN[1])
    mr = (lref >= a) & (lref <= b)
    nr = (lref >= NWIN[0]) & (lref <= NWIN[1])
    lum = np.trapezoid(f[m], l[m]) / np.trapezoid(f[n], l[n])
    ref = np.trapezoid(fref[mr], lref[mr]) / np.trapezoid(fref[nr], lref[nr])
    return lum / ref


for nm2, l, f in [("scatter", ls, fs), ("macroatom", lm, fm)]:
    print(f"{nm2:10s} vs DDC15: " +
          "  ".join(f"{n} {bandratio(l,f,n,a,b):.2f}" for n, a, b in BANDS) +
          f"   peak {peak(l,f):.0f}A (ref {peak(lref,fref):.0f}A)")
