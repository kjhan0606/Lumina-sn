#!/usr/bin/env python3
"""H2 de-risk: does REMOVING the tau_rec/t_exp zone-switch (TAUREC=1e9, every shell
on the single bisection) cure the sh24-31 crossover kink that the blanket run shows?
Overlays noswitch vs blanket vs CMFGEN; zooms the transition shells 22-32 and prints
the sh24-31 RMS for each. If noswitch flattens sh24-31 -> the switch made the kink
(H2 confirmed); if it persists -> the bias is in the inputs (H3/H1 mandatory)."""
import numpy as np, csv, glob, sys

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"

def rd(p):
    r = csv.reader(open(p)); h = next(r); c = {k: [] for k in h}
    for row in r:
        for k, v in zip(h, row): c[k].append(float(v))
    return {k: np.array(v) for k, v in c.items()}

geo = rd(f"{REF}/geometry.csv")
v = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5
ref_te = rd(f"{REF}/plasma_state.csv")["T_rad"]
nsh = len(ref_te)

# auto-find the latest noswitch run
import os
ns_dirs = sorted(d for d in glob.glob(f"{ROOT}/logs/ddc15_radeq_noswitch_*")
                 if os.path.isdir(d))
if not ns_dirs:
    sys.exit("no ddc15_radeq_noswitch_* run found yet")
NS = ns_dirs[-1]
print(f"noswitch run: {NS}")

runs = {
    "blanket (zone-switch, TAUREC=1.0)": (f"{ROOT}/logs/ddc15_radeq_blanket_164032", "#3898EC", "D"),
    "noswitch (single bisection, 1e9)":  (f"{NS}",                                   "#D97757", "s"),
}
data = {k: rd(f"{p}/lumina_plasma_state.csv")["T_e"] for k, (p, _, _) in runs.items()}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DDC15 0.976d H2 de-risk: does removing the zone-switch cure the sh24-31 kink?",
             fontsize=13, weight="bold")

a = ax[0]
a.plot(v, ref_te, "k", marker="o", ms=3, lw=1.8, label="CMFGEN gas T (flat plateau)")
for k, (p, col, mk) in runs.items():
    a.plot(v, data[k], col, marker=mk, ms=3, lw=1.0, alpha=0.85, label=k)
a.axvspan(v[24], v[31], color="orange", alpha=0.10)
a.set_xlabel("velocity [km/s]"); a.set_ylabel("T_e [K]"); a.set_ylim(0, 7000)
a.set_title("A. full T_e(v); orange = transition sh24-31"); a.legend(fontsize=9); a.grid(alpha=0.3)

b = ax[1]
zo = slice(22, 33)
b.plot(np.arange(nsh)[zo], ref_te[zo], "k", marker="o", ms=5, lw=2, label="CMFGEN")
for k, (p, col, mk) in runs.items():
    b.plot(np.arange(nsh)[zo], data[k][zo], col, marker=mk, ms=5, lw=1.2, label=k)
b.axvspan(24, 31, color="orange", alpha=0.10)
b.set_xlabel("shell index"); b.set_ylabel("T_e [K]")
b.set_title("B. ZOOM sh22-32 (the crossover)"); b.legend(fontsize=9); b.grid(alpha=0.3)

# quantify
o2431 = slice(24, 32); oall = slice(24, 49)
print(f"\nsh24-31 (transition) and sh24-48 (full outer) RMS vs CMFGEN:")
for k, (p, col, mk) in runs.items():
    d = data[k]
    r1 = np.sqrt(np.mean(((d[o2431]-ref_te[o2431])/ref_te[o2431])**2))*100
    r2 = np.sqrt(np.mean(((d[oall]-ref_te[oall])/ref_te[oall])**2))*100
    print(f"  {k:38s} sh24-31 RMS={r1:5.1f}%  sh24-48 RMS={r2:5.1f}%  "
          f"sh24-31 mean={np.mean(d[o2431]):5.0f}K (CMFGEN {np.mean(ref_te[o2431]):.0f}K)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{ROOT}/figures/2026-06-07_ddc15_radeq_noswitch.png"
plt.savefig(out, dpi=130); print("saved", out)
