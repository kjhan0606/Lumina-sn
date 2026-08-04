#!/usr/bin/env python3
"""DDC15 0.976d RADEQ T_e A/B (+cooling fix) vs CMFGEN reference.

Arms:
  OFF      job 163554_0  : T_e = 0.9*T_rad (production baseline)
  ON       job 163554_1  : LUMINA_RADEQ_TE=1 (full thermal-balance T_e; over-cools
                           photosphere because collisional line cooling summed the
                           full 2.5M-line census with lagged dilute-Boltzmann pops)
  COOLFIX  job 163580     : RADEQ_TE=1 + RADEQ_COOL_NLTE_ONLY=1 (collisional cooling
                           restricted to SE-satisfying NLTE-tracked levels)

Panels: A spectrum, B n_e(v), C T_e(v) [the RADEQ target], D T_rad(v).
CMFGEN plasma_state.csv carries t_rad; t_electron column is the CMFGEN T_e target.
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"
OFF  = f"{ROOT}/logs/ddc15_radeq_off_163554_0"
ON   = f"{ROOT}/logs/ddc15_radeqesc_163604"        # escape cooling, NLTE-only
COOL = f"{ROOT}/logs/ddc15_radeqescall_163606"     # escape cooling, all-lines
CMFGEN_SPEC = f"{ROOT}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"

def read_csv_cols(path):
    with open(path) as f:
        r = csv.reader(f); hdr = next(r)
        cols = {h: [] for h in hdr}
        for row in r:
            for h, v in zip(hdr, row):
                cols[h].append(float(v))
    return {h: np.array(v) for h, v in cols.items()}

geo = read_csv_cols(f"{REF}/geometry.csv")
v_mid = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5

ref_ne = read_csv_cols(f"{REF}/electron_densities.csv")["n_e"]
ref_ps = read_csv_cols(f"{REF}/plasma_state.csv")
# CMFGEN solves a single gas temperature (radiative equilibrium). The TARDIS
# reference plasma_state.csv "T_rad" column was populated from that CMFGEN gas
# temperature (verified: matches the hydro Temperature block interpolated onto the
# 49-shell grid, 4374K shell0 -> 2505K outer), so it IS the CMFGEN T_e target.
ref_te = ref_ps["T_rad"]

def load_arm(d):
    p = f"{d}/lumina_plasma_state.csv"
    s = f"{d}/lumina_spectrum_formal.csv"
    out = {}
    out["ps"] = read_csv_cols(p) if os.path.exists(p) else None
    out["sp"] = read_csv_cols(s) if os.path.exists(s) else None
    return out

arms = {"OFF": load_arm(OFF), "ESC-NLTE": load_arm(ON), "ESC-ALL": load_arm(COOL)}

cm = np.loadtxt(CMFGEN_SPEC)
cm_w, cm_f = cm[:, 0], cm[:, 1]

def norm_optical(w, f, lo=3500, hi=9000):
    m = (w >= lo) & (w <= hi)
    area = np.trapz(f[m], w[m])
    return f / area if area > 0 else f

C = {"REF": "k", "OFF": "#3898EC", "ESC-NLTE": "#D97757", "ESC-ALL": "#4EC9B0"}
MK = {"OFF": "s", "ESC-NLTE": "^", "ESC-ALL": "D"}

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("DDC15 0.976d: RADEQ T_e A/B (+collisional-cooling fix) vs CMFGEN",
             fontsize=15, weight="bold")

a = ax[0, 0]
a.plot(cm_w, norm_optical(cm_w, cm_f), C["REF"], lw=1.4, label="CMFGEN DDC15 0.976d")
for k, arm in arms.items():
    if arm["sp"] is None: continue
    w = arm["sp"]["wavelength_angstrom"]; f = arm["sp"]["flux"]
    a.plot(w, norm_optical(w, f), C[k], lw=1.0, alpha=0.85, label=f"LUMINA {k}")
a.set_xlim(2500, 12000); a.set_xlabel("wavelength [A]")
a.set_ylabel("flux (norm optical area 3500-9000A)")
a.set_title("A. Spectrum"); a.legend(fontsize=9)

b = ax[0, 1]
b.semilogy(v_mid, ref_ne, C["REF"], marker="o", ms=3, label="CMFGEN")
for k, arm in arms.items():
    if arm["ps"] is None: continue
    b.semilogy(v_mid, arm["ps"]["n_e"], C[k], marker=MK[k], ms=3, label=k)
b.set_xlabel("velocity [km/s]"); b.set_ylabel("n_e [cm^-3]")
b.set_title("B. Electron density"); b.legend(fontsize=9); b.grid(alpha=0.3)

c = ax[1, 0]
if ref_te is not None:
    c.plot(v_mid, ref_te, C["REF"], marker="o", ms=3, label="CMFGEN T_e")
for k, arm in arms.items():
    if arm["ps"] is None: continue
    te = arm["ps"].get("T_e", arm["ps"].get("t_electron"))
    if te is not None:
        c.plot(v_mid, te, C[k], marker=MK[k], ms=3, label=f"{k} T_e")
c.set_xlabel("velocity [km/s]"); c.set_ylabel("T_e [K]")
c.set_title("C. Electron temperature (RADEQ target)")
c.legend(fontsize=9); c.grid(alpha=0.3)

d = ax[1, 1]
d.plot(v_mid, ref_ps["t_rad"] if "t_rad" in ref_ps else ref_ps["T_rad"],
       C["REF"], marker="o", ms=3, label="CMFGEN T_rad")
for k, arm in arms.items():
    if arm["ps"] is None: continue
    d.plot(v_mid, arm["ps"]["T_rad"], C[k], marker=MK[k], ms=3, label=f"{k} T_rad")
d.set_xlabel("velocity [km/s]"); d.set_ylabel("T_rad [K]")
d.set_title("D. Radiation temperature"); d.legend(fontsize=9); d.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{ROOT}/figures/2026-06-05_ddc15_radeq_te_vs_cmfgen.png"
plt.savefig(out, dpi=130)
print("saved", out)

print("\n shell  v[km/s]   T_e CMFGEN", end="")
for k in arms: print(f"   T_e {k:7s}", end="")
print()
for s in [0, 6, 12, 24, 36, 48]:
    line = f"  {s:3d}  {v_mid[s]:8.0f}   {ref_te[s] if ref_te is not None else float('nan'):8.1f}"
    for k, arm in arms.items():
        if arm["ps"] is None:
            line += f"   {'--':>9}"; continue
        te = arm["ps"].get("T_e", arm["ps"].get("t_electron"))
        line += f"   {te[s]:9.1f}" if te is not None else f"   {'--':>9}"
    print(line)
