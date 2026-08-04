#!/usr/bin/env python3
"""DDC15 0.976d: emergent spectrum, CMFGEN vs LUMINA frozen-zone closures
color(B2) / nebular / adiab. Formal-integral spectra, optical-normalized."""
import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
CMFGEN_SPEC = f"{ROOT}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"

def rd(p):
    r = csv.reader(open(p)); h = next(r); c = {k: [] for k in h}
    for row in r:
        for k, v in zip(h, row): c[k].append(float(v))
    return {k: np.array(v) for k, v in c.items()}

def norm(w, f, lo=3500, hi=9000):
    m = (w >= lo) & (w <= hi); a = np.trapz(f[m], w[m]); return f / a if a > 0 else f

def peak(w, f, lo=4000, hi=9000):
    m = (w >= lo) & (w <= hi); ww, ff = w[m], f[m]; return ww[np.argmax(ff)]

cm = np.loadtxt(CMFGEN_SPEC); cm_w, cm_f = cm[:, 0], cm[:, 1]

runs = {
    "color (B2)": ("logs/ddc15_radeqB2_163887",        "#D97757"),
    "nebular":    ("logs/ddc15_radeq_nebular_164006",  "#4EC9B0"),
    "adiab":      ("logs/ddc15_radeq_adiab_164007",    "#FFC107"),
}

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(cm_w, norm(cm_w, cm_f), "k", lw=1.6, label=f"CMFGEN (peak {peak(cm_w,cm_f):.0f}Å)")
for k, (p, col) in runs.items():
    s = rd(f"{ROOT}/{p}/lumina_spectrum_formal.csv")
    w, f = s["wavelength_angstrom"], s["flux"]
    ax.plot(w, norm(w, f), col, lw=1.1, alpha=0.85,
            label=f"LUMINA {k} (peak {peak(w,f):.0f}Å)")

ax.set_xlim(2500, 12000)
ax.set_xlabel("wavelength [Å]")
ax.set_ylabel("flux (normalized over optical 3500–9000 Å)")
ax.set_title("DDC15 0.976d: emergent spectrum vs CMFGEN — frozen-zone T_e closures")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
out = f"{ROOT}/figures/2026-06-06_ddc15_radeq_ab_spectrum.png"
plt.savefig(out, dpi=130); print("saved", out)

print(f"\npeak [Å]: CMFGEN {peak(cm_w,cm_f):.0f}", end="")
for k, (p, _) in runs.items():
    s = rd(f"{ROOT}/{p}/lumina_spectrum_formal.csv")
    print(f"   {k} {peak(s['wavelength_angstrom'], s['flux']):.0f}", end="")
print()
