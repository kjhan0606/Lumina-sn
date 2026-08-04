#!/usr/bin/env python3
"""DDC15 0.976d: B2-hybrid RADEQ T_e vs path-B vs CMFGEN reference.
Panel A spectrum, B T_e(v), C n_e(v), D T_rad(v)."""
import numpy as np, csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"
PB   = f"{ROOT}/logs/ddc15_radeqB_163822"
B2   = f"{ROOT}/logs/ddc15_radeqB2_163887"
CMFGEN_SPEC = f"{ROOT}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"

def rd(p):
    r = csv.reader(open(p)); h = next(r); c = {k: [] for k in h}
    for row in r:
        for k, v in zip(h, row): c[k].append(float(v))
    return {k: np.array(v) for k, v in c.items()}

geo = rd(f"{REF}/geometry.csv")
v = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5
ref_ps = rd(f"{REF}/plasma_state.csv")
ref_te = ref_ps["T_rad"]                 # CMFGEN gas temperature = T_e target
ref_ne = rd(f"{REF}/electron_densities.csv")["n_e"]
pb = rd(f"{PB}/lumina_plasma_state.csv")
b2 = rd(f"{B2}/lumina_plasma_state.csv")

cm = np.loadtxt(CMFGEN_SPEC); cm_w, cm_f = cm[:, 0], cm[:, 1]
def norm(w, f, lo=3500, hi=9000):
    m = (w >= lo) & (w <= hi); a = np.trapz(f[m], w[m]); return f / a if a > 0 else f

C = {"REF": "k", "PB": "#3898EC", "B2": "#D97757"}
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("DDC15 0.976d: B2-hybrid RADEQ T_e vs path-B vs CMFGEN", fontsize=15, weight="bold")

a = ax[0, 0]
a.plot(cm_w, norm(cm_w, cm_f), C["REF"], lw=1.4, label="CMFGEN (reference)")
for d, k, lab in [(PB, "PB", "path-B (diverged)"), (B2, "B2", "B2 hybrid")]:
    s = rd(f"{d}/lumina_spectrum_formal.csv")
    a.plot(s["wavelength_angstrom"], norm(s["wavelength_angstrom"], s["flux"]),
           C[k], lw=1.0, alpha=0.85, label=f"LUMINA {lab}")
a.set_xlim(2500, 12000); a.set_xlabel("wavelength [A]")
a.set_ylabel("flux (norm optical 3500-9000A)"); a.set_title("A. Spectrum"); a.legend(fontsize=9)

b = ax[0, 1]
b.plot(v, ref_te, C["REF"], marker="o", ms=3, label="CMFGEN T_e")
b.plot(v, pb["T_e"], C["PB"], marker="s", ms=3, label="path-B T_e")
b.plot(v, b2["T_e"], C["B2"], marker="^", ms=3, label="B2 hybrid T_e")
b.set_xlabel("velocity [km/s]"); b.set_ylabel("T_e [K]"); b.set_ylim(1500, 9000)
b.set_title("B. Electron temperature (RADEQ target)"); b.legend(fontsize=9); b.grid(alpha=0.3)

c = ax[1, 0]
c.semilogy(v, ref_ne, C["REF"], marker="o", ms=3, label="CMFGEN")
c.semilogy(v, b2["n_e"], C["B2"], marker="^", ms=3, label="B2 hybrid")
c.set_xlabel("velocity [km/s]"); c.set_ylabel("n_e [cm^-3]")
c.set_title("C. Electron density"); c.legend(fontsize=9); c.grid(alpha=0.3)

d = ax[1, 1]
d.plot(v, ref_te, C["REF"], marker="o", ms=3, label="CMFGEN T_e (gas)")
d.plot(v, b2["T_rad"], C["B2"], marker="^", ms=3, label="B2 T_rad (color)")
d.set_xlabel("velocity [km/s]"); d.set_ylabel("T [K]")
d.set_title("D. CMFGEN gas-T vs LUMINA radiation color-T (decoupling)")
d.legend(fontsize=9); d.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{ROOT}/figures/2026-06-06_ddc15_radeq_B2_vs_cmfgen.png"
plt.savefig(out, dpi=130); print("saved", out)

# spectral metrics
def peak(w, f, lo=4000, hi=9000):
    m = (w >= lo) & (w <= hi); ww, ff = w[m], f[m]; return ww[np.argmax(ff)]
print(f"\npeak wavelength [A]: CMFGEN {peak(cm_w, cm_f):.0f}", end="")
for d, lab in [(PB, "path-B"), (B2, "B2")]:
    s = rd(f"{d}/lumina_spectrum_formal.csv")
    print(f"   {lab} {peak(s['wavelength_angstrom'], s['flux']):.0f}", end="")
print()
