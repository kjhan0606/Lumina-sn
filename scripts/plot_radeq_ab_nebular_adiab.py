#!/usr/bin/env python3
"""DDC15 0.976d: frozen-zone T_e closure A/B — color(B2) vs nebular(b) vs adiab(a)
vs CMFGEN gas temperature. Verdict: nebular wins (outer RMS 42->24%)."""
import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"

def rd(p):
    r = csv.reader(open(p)); h = next(r); c = {k: [] for k in h}
    for row in r:
        for k, v in zip(h, row): c[k].append(float(v))
    return {k: np.array(v) for k, v in c.items()}

geo = rd(f"{REF}/geometry.csv")
v = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5
ref_te = rd(f"{REF}/plasma_state.csv")["T_rad"]   # CMFGEN gas temperature

runs = {
    "B2 color (ratio·T_rad)": ("logs/ddc15_radeqB2_163887",   "#D97757", "^"),
    "nebular (W·B_ν balance)": ("logs/ddc15_radeq_nebular_164006", "#4EC9B0", "o"),
    "adiab ((t_0/t)^2)":       ("logs/ddc15_radeq_adiab_164007",   "#FFC107", "s"),
}
data = {k: rd(f"{ROOT}/{p}/lumina_plasma_state.csv")["T_e"] for k, (p, _, _) in runs.items()}

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DDC15 0.976d: frozen-zone T_e closure A/B vs CMFGEN gas temperature",
             fontsize=14, weight="bold")

a = ax[0]
a.plot(v, ref_te, "k", marker="o", ms=3, lw=1.6, label="CMFGEN gas T (target)")
for k, (p, col, mk) in runs.items():
    a.plot(v, data[k], col, marker=mk, ms=3, lw=1.0, alpha=0.85, label=k)
a.axhline(2505, ls=":", color="gray", lw=0.8)
a.set_xlabel("velocity [km/s]"); a.set_ylabel("T_e [K]"); a.set_ylim(0, 7000)
a.set_title("A. T_e(v): nebular tracks CMFGEN, adiab collapses")
a.legend(fontsize=9); a.grid(alpha=0.3)

b = ax[1]
o = slice(24, 49)
names, rmss, cols = [], [], []
for k, (p, col, mk) in runs.items():
    d = data[k][o]; r = ref_te[o]
    rms = np.sqrt(np.mean(((d - r) / r) ** 2)) * 100
    names.append(k.split(" ")[0]); rmss.append(rms); cols.append(col)
bars = b.bar(names, rmss, color=cols)
for bar, val in zip(bars, rmss):
    b.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha="center", fontsize=11)
b.set_ylabel("outer sh24-48 RMS% vs CMFGEN")
b.set_title("B. Outer frozen-zone error (lower = better)")
b.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{ROOT}/figures/2026-06-06_ddc15_radeq_ab_nebular_adiab.png"
plt.savefig(out, dpi=130); print("saved", out)
