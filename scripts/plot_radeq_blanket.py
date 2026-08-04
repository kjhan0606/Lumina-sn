#!/usr/bin/env python3
"""DDC15 0.976d Stage-0 de-risk: does a BLANKETED expansion-opacity heating
source J_eff(ν) move outer T_e toward CMFGEN's gas temp vs the bare W·B_ν
(nebular)? NOT a winner-picking A/B — all closures are approximate; this asks
whether blanketing alone closes nebular's residual +8% over-heat, which gates
whether the full deterministic-J solver is worth building."""
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
    "color (ratio·T_rad)":        ("logs/ddc15_radeqB2_163887",       "#D97757", "^"),
    "nebular (bare W·B_ν)":       ("logs/ddc15_radeq_nebular_164006", "#4EC9B0", "o"),
    "blanket (J_eff exp-opac)":   ("logs/ddc15_radeq_blanket_164032", "#3898EC", "D"),
}
data = {k: rd(f"{ROOT}/{p}/lumina_plasma_state.csv")["T_e"] for k, (p, _, _) in runs.items()}

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DDC15 0.976d Stage-0: blanketed expansion-opacity heating vs bare W·B_ν",
             fontsize=14, weight="bold")

a = ax[0]
a.plot(v, ref_te, "k", marker="o", ms=3, lw=1.6, label="CMFGEN gas T (target)")
for k, (p, col, mk) in runs.items():
    a.plot(v, data[k], col, marker=mk, ms=3, lw=1.0, alpha=0.85, label=k)
a.axhline(2505, ls=":", color="gray", lw=0.8)
a.set_xlabel("velocity [km/s]"); a.set_ylabel("T_e [K]"); a.set_ylim(0, 7000)
a.set_title("A. T_e(v): does blanketing pull the outer toward CMFGEN?")
a.legend(fontsize=9); a.grid(alpha=0.3)

b = ax[1]
o = slice(24, 49)
names, means, rmss, cols = [], [], [], []
print(f"\nouter sh24-48 vs CMFGEN mean {np.mean(ref_te[o]):.0f}K:")
for k, (p, col, mk) in runs.items():
    d = data[k][o]; r = ref_te[o]
    rms = np.sqrt(np.mean(((d - r) / r) ** 2)) * 100
    mn = np.mean(d)
    names.append(k.split(" ")[0]); means.append(mn); rmss.append(rms); cols.append(col)
    print(f"  {k:28s} mean={mn:6.0f}K ({100*(mn/np.mean(r)-1):+5.1f}%)  RMS={rms:5.1f}%")
bars = b.bar(names, rmss, color=cols)
for bar, val in zip(bars, rmss):
    b.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha="center", fontsize=11)
b.set_ylabel("outer sh24-48 RMS% vs CMFGEN")
b.set_title("B. Outer error — blanket vs nebular vs color")
b.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{ROOT}/figures/2026-06-06_ddc15_radeq_blanket.png"
plt.savefig(out, dpi=130); print("saved", out)
