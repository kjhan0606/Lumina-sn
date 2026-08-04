#!/usr/bin/env python3
"""Pure-CMFGEN first-light: overlay the deterministic tangent-ray formal-solver
T_e(v) vs CMFGEN gas T and the blanket MC run. Pure CMFGEN replaces ONLY the
radiation field (J_nu) with a deterministic comoving-frame solve, reusing the
same RADEQ/plasma/NLTE downstream. Prints inner (photosphere) + outer RMS.
NOTE: solver physics is UNVALIDATED — the thick(J->S->B) and thin(W*B) limits
must be checked before trusting absolute numbers; this is a sanity overlay."""
import numpy as np, csv, glob, os, sys

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
nsh = len(ref_te)

pc_dirs = sorted(d for d in glob.glob(f"{ROOT}/logs/ddc15_pure_cmfgen_*")
                 if os.path.isdir(d))
if not pc_dirs:
    sys.exit("no ddc15_pure_cmfgen_* run found")
PC = pc_dirs[-1]
print(f"pure-CMFGEN run: {PC}")

runs = {
    "blanket MC (zone-switch)":      (f"{ROOT}/logs/ddc15_radeq_blanket_164032", "#3898EC", "D"),
    "pure CMFGEN (deterministic J)": (f"{PC}",                                   "#D97757", "s"),
}
data = {}
for k, (p, _, _) in runs.items():
    d = rd(f"{p}/lumina_plasma_state.csv")
    data[k] = d["T_e"]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DDC15 0.976d pure-CMFGEN first light: deterministic tangent-ray J vs CMFGEN gas T "
             "(UNVALIDATED — sanity overlay)", fontsize=12, weight="bold")

a = ax[0]
a.plot(v, ref_te, "k", marker="o", ms=3, lw=1.8, label="CMFGEN gas T")
for k, (p, col, mk) in runs.items():
    a.plot(v, data[k], col, marker=mk, ms=3, lw=1.0, alpha=0.85, label=k)
a.axvspan(v[24], v[48], color="purple", alpha=0.07)
a.set_xlabel("velocity [km/s]"); a.set_ylabel("T_e [K]"); a.set_ylim(0, 7000)
a.set_title("A. full T_e(v); purple = outer sh24-48"); a.legend(fontsize=9); a.grid(alpha=0.3)

b = ax[1]
zo = slice(0, nsh)
b.plot(np.arange(nsh)[zo], ref_te[zo], "k", marker="o", ms=3, lw=1.5, label="CMFGEN")
for k, (p, col, mk) in runs.items():
    b.plot(np.arange(nsh)[zo], data[k][zo], col, marker=mk, ms=3, lw=1.0, label=k)
b.set_xlabel("shell index"); b.set_ylabel("T_e [K]")
b.set_title("B. by shell index"); b.legend(fontsize=9); b.grid(alpha=0.3); b.set_ylim(0, 7000)

inn = slice(0, 1); oo = slice(24, 49)
print(f"\nphotosphere sh0 + outer sh24-48 RMS vs CMFGEN:")
for k, (p, col, mk) in runs.items():
    d = data[k]
    r_out = np.sqrt(np.mean(((d[oo]-ref_te[oo])/ref_te[oo])**2))*100
    print(f"  {k:32s} T_e[0]={d[0]:5.0f}K (CMFGEN {ref_te[0]:.0f}K, {100*(d[0]/ref_te[0]-1):+5.1f}%)  "
          f"outer sh24-48 mean={np.mean(d[oo]):5.0f}K (CMFGEN {np.mean(ref_te[oo]):.0f}K)  RMS={r_out:5.1f}%")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{ROOT}/figures/2026-06-07_ddc15_pure_cmfgen_firstlight.png"
plt.savefig(out, dpi=130); print("saved", out)
