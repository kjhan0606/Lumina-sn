"""DDC15 0.976d self-test: 4-way comparison of the Ti-II-fix + cap-fix matrix.
Shows that macroatom (not scatter) blankets green, but the energy-conserving (#5)
macroatom run reveals a severe over-redshift cascade (blue-starve / NIR-pile)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DDC15 = Path("data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat")
BANDS = {"blue": (3500, 4500), "green": (4500, 5500),
         "red": (5500, 7000), "NIR": (7000, 9000)}

def load_dat(p):
    a = np.loadtxt(p); return a[:, 0], a[:, 1]

def load_csv(p):
    a = np.genfromtxt(p, delimiter=",", names=True)
    return a["wavelength_angstrom"], a["flux"]

def norm(w, f, rw, rf):
    m = (w >= 4000) & (w <= 9000)
    rm = (rw >= 4000) & (rw <= 9000)
    return f * (np.trapz(rf[rm], rw[rm]) / np.trapz(f[m], w[m]))

runs = [
    ("scatter+Ti (162748)",            "logs/paperDDC15init_ddc15init_scatter_162748/lumina_spectrum.csv",           "#3898EC"),
    ("macroatom+Ti legacy-cap (162749)","logs/paperDDC15init_ddc15init_macroatom_162749/lumina_spectrum.csv",        "#D97757"),
    ("macroatom+Ti #5-clean (162755)", "logs/paperDDC15init_ddc15init_macroatom_cro1_cfe1_162755/lumina_spectrum.csv","#4EC9B0"),
]

dw, df = load_dat(DDC15)
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(dw, df, color="k", lw=2.2, label="DDC15 own 0.976d (target)", zorder=5)
for lab, path, c in runs:
    w, f = load_csv(path)
    ax.plot(w, norm(w, f, dw, df), color=c, lw=1.2, alpha=0.9, label=lab)

for b, (lo, hi) in BANDS.items():
    ax.axvspan(lo, hi, color="gray", alpha=0.05)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.96, b, ha="center", fontsize=9, color="gray")

ax.set_xlim(3000, 9500)
ax.set_xlabel("wavelength (Å)"); ax.set_ylabel("flux (normalized 4000–9000 Å)")
ax.set_title("DDC15 0.976d self-test — Ti II fix + #5 cap fix: green blanketing is macroatom-only,\n"
             "but energy-conserving macroatom reveals severe over-redshift (blue 0.22 / NIR 1.67)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)
out = "figures/2026-06-04_combo_TiII_cap_green_4way.png"
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)
