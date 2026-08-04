#!/usr/bin/env python3
"""DDC15 0.976d Stage-0 verdict: blanketed expansion-opacity heating CURES the
systematic outer over-heat (mean on CMFGEN), and the damping test (η0.5 vs η0.3)
shows the residual scatter is INTRINSIC MC NOISE — under-relaxation barely moves
RMS (22.7%->20.9%), so the flat CMFGEN plateau is unreachable by a per-shell
balance fed noisy MC inputs. The deterministic two-stream J solver is NOT the lever."""
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
    "nebular (bare W·B_ν)":          ("logs/ddc15_radeq_nebular_164006",     "#4EC9B0", "o"),
    "blanket η=0.5 (10 it)":         ("logs/ddc15_radeq_blanket_164032",     "#3898EC", "D"),
    "blanket η=0.3 (15 it)":         ("logs/ddc15_radeq_blanketdamp_164105", "#D97757", "s"),
}
data = {k: rd(f"{ROOT}/{p}/lumina_plasma_state.csv")["T_e"] for k, (p, _, _) in runs.items()}

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DDC15 0.976d Stage-0: blanketed heating fixes the MEAN; residual scatter is MC noise (damping can't kill it)",
             fontsize=13, weight="bold")

a = ax[0]
a.plot(v, ref_te, "k", marker="o", ms=3, lw=1.8, label="CMFGEN gas T (flat plateau)")
for k, (p, col, mk) in runs.items():
    a.plot(v, data[k], col, marker=mk, ms=3, lw=1.0, alpha=0.85, label=k)
a.axhline(2505, ls=":", color="gray", lw=0.8)
a.axvspan(v[24], v[31], color="orange", alpha=0.08)
a.axvspan(v[40], v[48], color="purple", alpha=0.08)
a.set_xlabel("velocity [km/s]"); a.set_ylabel("T_e [K]"); a.set_ylim(0, 7000)
a.set_title("A. T_e(v): blanket tracks CMFGEN mean but scatters; η0.3 barely tighter")
a.legend(fontsize=9); a.grid(alpha=0.3)
a.text(v[27], 6500, "freeze-gate\ncrossover\n(sh24-31 HIGH)", fontsize=8, color="#a06000", ha="center")
a.text(v[44], 700, "packet-starved\n(sh40-48 LOW)", fontsize=8, color="#603090", ha="center")

b = ax[1]
o = slice(24, 49); r = ref_te[o]
names, means, rmss, cols = [], [], [], []
print(f"\nouter sh24-48 vs CMFGEN mean {np.mean(r):.0f}K:")
for k, (p, col, mk) in runs.items():
    d = data[k][o]
    rms = np.sqrt(np.mean(((d - r) / r) ** 2)) * 100
    names.append(k.split(" ")[0] + "\n" + k.split("(")[-1].rstrip(")") if "(" in k else k.split(" ")[0])
    means.append(np.mean(d)); rmss.append(rms); cols.append(col)
    print(f"  {k:24s} mean={np.mean(d):6.0f}K ({100*(np.mean(d)/np.mean(r)-1):+5.1f}%)  RMS={rms:5.1f}%")
x = np.arange(len(names))
bars = b.bar(x, rmss, color=cols, width=0.55)
for bar, val, mn in zip(bars, rmss, means):
    b.text(bar.get_x() + bar.get_width()/2, val + 0.6, f"{val:.1f}%", ha="center", fontsize=11, weight="bold")
    b.text(bar.get_x() + bar.get_width()/2, val/2, f"mean\n{mn:.0f}K", ha="center", fontsize=9, color="white")
b.set_xticks(x); b.set_xticklabels(names, fontsize=9)
b.set_ylabel("outer sh24-48 RMS% vs CMFGEN")
b.set_title("B. Stronger damping (0.5→0.3) barely moves RMS → residual = MC noise")
b.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{ROOT}/figures/2026-06-07_ddc15_radeq_blanket_damp.png"
plt.savefig(out, dpi=130); print("saved", out)
