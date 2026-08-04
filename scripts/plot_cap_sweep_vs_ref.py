#!/usr/bin/env python3
"""Cap-scaling sensitivity sweep: DDC15 0.976d macroatom (cro1/cfe1) emergent
SED at LUMINA_MAX_INTERACTIONS = 200/1000/2000/20000, plus scatter endpoints,
all vs the DDC15 0.976d reference. Tests the survivorship-bias hypothesis: if
raising the cap revives the 4500A peak and reduces the NIR pile, the 200-cap was
scissoring the redward-cascaded (heavily-reprocessed) packets."""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
NWIN = (3500.0, 9000.0)
WMIN, WMAX = 3000.0, 11000.0

# (label, line_int, cap, jobid)
RUNS = [
    ("macro cap=200", "macroatom", 200, 163164),
    ("macro cap=1000", "macroatom", 1000, 163165),
    ("macro cap=2000", "macroatom", 2000, 163166),
    ("macro cap=20000", "macroatom", 20000, 163167),
    ("scatter cap=200", "scatter", 200, 163168),
    ("scatter cap=20000", "scatter", 20000, 163169),
]
COLORS = ["#D97757", "#E0A030", "#4EC9B0", "#3898EC", "#A0A0A0", "#404040"]
BANDS = [("blue", 3500, 4500), ("green", 4500, 5500),
         ("red", 5500, 7000), ("NIR", 7000, 9000)]


def find_csv(line_int, cap, jobid):
    pat = f"logs/paperDDC15init_ddc15init_{line_int}_cro1_cfe1_kp0_ste0_rte0_mi{cap}_sm*_{jobid}/lumina_spectrum.csv"
    g = glob.glob(pat)
    return g[0] if g else None


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
fref_n = norm(lref, fref)


def bandratio(l, f, a, b):
    m = (l >= a) & (l <= b)
    n = (l >= NWIN[0]) & (l <= NWIN[1])
    mr = (lref >= a) & (lref <= b)
    nr = (lref >= NWIN[0]) & (lref <= NWIN[1])
    return (np.trapezoid(f[m], l[m]) / np.trapezoid(f[n], l[n])) / \
           (np.trapezoid(fref[mr], lref[mr]) / np.trapezoid(fref[nr], lref[nr]))


fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(lref, fref_n, color="0.15", lw=2.4, label=f"DDC15 0.976d ref  peak {peak(lref,fref):.0f}A")
for (lab, li, cap, jid), c in zip(RUNS, COLORS):
    p = find_csv(li, cap, jid)
    if p is None:
        print(f"MISSING: {lab} ({jid})")
        continue
    l, f = loadcsv(p)
    fn = norm(l, f)
    ls = "-" if li == "macroatom" else "--"
    ax.plot(l, fn, color=c, lw=1.5, ls=ls, label=f"{lab}  peak {peak(l,f):.0f}A")
    print(f"{lab:18s}: " + "  ".join(f"{n} {bandratio(l,f,a,b):.2f}" for n, a, b in BANDS) +
          f"   peak {peak(l,f):.0f}A")
for a, b in [(3500, 4500), (4500, 5500), (5500, 7000), (7000, 9000)]:
    ax.axvspan(a, b, color="0.9", alpha=0.10)
ax.axvline(peak(lref, fref), color="0.15", ls=":", lw=1)
ax.set_xlim(WMIN, WMAX)
ax.set_ylim(0, None)
ax.set_xlabel("wavelength (A)")
ax.set_ylabel("normalized flux")
ax.set_title("DDC15 0.976d cap-scaling sweep: does lifting the interaction cap revive blue / reduce NIR?")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
out = "figures/2026-06-04_ddc15_cap_sweep_vs_ref.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
