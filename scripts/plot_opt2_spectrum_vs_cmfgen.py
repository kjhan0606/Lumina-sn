#!/usr/bin/env python3
"""Option-2 pure-CMFGEN emergent spectrum (tangent-ray surface integral) vs the
CMFGEN ground-truth DDC15 0.976d spectrum (Blondin et al. 2015), overlaying the
T_e_T_rad_ratio=0.9 (default seed) and ratio=1.0 (thick-limit thermalization)
arms to expose the seed-anchor effect on the emergent SED."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
R09 = "logs/ddc15_pc_phase3_lstar1_linere1_165118/lumina_spectrum.csv"
R10 = "logs/ddc15_pc_phase3_lstar1_linere1_ratio1.0_165124/lumina_spectrum.csv"
NWIN = (3500.0, 9000.0)
WMIN, WMAX = 2500.0, 11000.0


def loadcsv(p):
    d = np.genfromtxt(p, delimiter=",", names=True)
    return d["wavelength_angstrom"], d["flux"]


def norm(l, f):
    m = (l >= NWIN[0]) & (l <= NWIN[1])
    a = np.trapezoid(f[m], l[m])
    return f / a if a > 0 else f


def peak(l, f, a=4000, b=8000):
    m = (l >= a) & (l <= b)
    return l[m][np.argmax(f[m])] if m.any() else np.nan


lr = np.loadtxt(REF)
lref, fref = lr[:, 0], lr[:, 1]
l9, f9 = loadcsv(R09)
l10, f10 = loadcsv(R10)
fref_n, f9_n, f10_n = norm(lref, fref), norm(l9, f9), norm(l10, f10)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lref, fref_n, color="0.20", lw=2.4,
        label=f"CMFGEN truth (Blondin+2015)  peak {peak(lref,fref):.0f}A")
ax.plot(l9, f9_n, color="#707E9A", lw=1.4,
        label=f"Option-2  T_e/T_rad seed=0.9  peak {peak(l9,f9):.0f}A")
ax.plot(l10, f10_n, color="#D97757", lw=1.5,
        label=f"Option-2  T_e/T_rad seed=1.0  peak {peak(l10,f10):.0f}A")
for a, b in [(3500, 4500), (4500, 5500), (5500, 7000), (7000, 9000)]:
    ax.axvspan(a, b, color="0.9", alpha=0.12)
ax.axvline(peak(lref, fref), color="0.20", ls=":", lw=1)
ax.set_xlim(WMIN, WMAX)
ax.set_ylim(0, None)
ax.set_xlabel("wavelength (A)")
ax.set_ylabel("normalized flux  (1/integral over 3500-9000A)")
ax.set_title("DDC15 0.976d: Option-2 emergent spectrum vs CMFGEN truth "
             "(seed 0.9 vs 1.0 arms)")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
out = "figures/2026-06-10_ddc15_opt2_spectrum_vs_cmfgen.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

BANDS = [("blue", 3500, 4500), ("green", 4500, 5500),
         ("red", 5500, 7000), ("NIR", 7000, 9000)]


def bandratio(l, f, a, b):
    m = (l >= a) & (l <= b)
    n = (l >= NWIN[0]) & (l <= NWIN[1])
    mr = (lref >= a) & (lref <= b)
    nr = (lref >= NWIN[0]) & (lref <= NWIN[1])
    return (np.trapezoid(f[m], l[m]) / np.trapezoid(f[n], l[n])) / \
           (np.trapezoid(fref[mr], lref[mr]) / np.trapezoid(fref[nr], lref[nr]))


for nm, l, f in [("seed 0.9", l9, f9), ("seed 1.0", l10, f10)]:
    print(f"{nm:10s}: " + "  ".join(f"{n} {bandratio(l,f,a,b):.2f}" for n, a, b in BANDS) +
          f"   peak {peak(l,f):.0f}A")
