#!/usr/bin/env python3
"""Self-consistent T_e (full radiative-equilibrium balance) macroatom vs the
0.9*T_rad baseline, both against the DDC15 0.976d reference 'observation'.
NOTE: cap settings differ (selfTe run cro0, baseline cro1); a cap-matched
self-Te run (163163, cro1) is the clean A/B and supersedes this preview."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
BASE = "logs/paperDDC15init_ddc15init_macroatom_cro1_cfe1_kp0_162997/lumina_spectrum.csv"
SELF = "logs/paperDDC15init_ddc15init_macroatom_cro0_cfe0_kp0_ste1_rte0_smspectrum_163088/lumina_spectrum.csv"
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
lb, fb = loadcsv(BASE)
ls, fs = loadcsv(SELF)
fref_n, fb_n, fs_n = norm(lref, fref), norm(lb, fb), norm(ls, fs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lref, fref_n, color="0.30", lw=2.2, label=f"DDC15 0.976d (ref)  peak {peak(lref,fref):.0f}A")
ax.plot(lb, fb_n, color="#707E9A", lw=1.4, label=f"baseline T_e=0.9 T_rad  peak {peak(lb,fb):.0f}A")
ax.plot(ls, fs_n, color="#D97757", lw=1.5, label=f"self-consistent T_e (full balance)  peak {peak(ls,fs):.0f}A")
for a, b in [(3500, 4500), (4500, 5500), (5500, 7000), (7000, 9000)]:
    ax.axvspan(a, b, color="0.9", alpha=0.12)
ax.axvline(peak(lref, fref), color="0.30", ls=":", lw=1)
ax.set_xlim(WMIN, WMAX)
ax.set_ylim(0, None)
ax.set_xlabel("wavelength (A)")
ax.set_ylabel("normalized flux")
ax.set_title("DDC15 0.976d: self-consistent T_e vs 0.9 T_rad baseline vs reference "
             "(PREVIEW — cap settings differ; 163163 cap-matched A/B pending)")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
out = "figures/2026-06-04_ddc15_selfte_vs_baseline_vs_ref.png"
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


for nm, l, f in [("baseline 0.9Tr", lb, fb), ("self-Te full", ls, fs)]:
    print(f"{nm:16s}: " + "  ".join(f"{n} {bandratio(l,f,a,b):.2f}" for n, a, b in BANDS) +
          f"   peak {peak(l,f):.0f}A")
