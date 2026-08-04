#!/usr/bin/env python3
"""DDC15 0.976d frozen-in A/B (job 163486) vs CMFGEN reference.

Panels:
  A  spectrum    : LUMINA OFF / ON (formal) vs CMFGEN DDC15 0.976d
  B  n_e(v)      : LUMINA OFF / ON vs CMFGEN electron_densities.csv (flat outer plateau)
  C  T_rad(v)    : LUMINA OFF / ON vs CMFGEN plasma_state.csv
  D  W(v)        : LUMINA OFF / ON vs CMFGEN plasma_state.csv

tau is NOT plotted vs reference: the reference tau_sobolev.npy is an all-zero
placeholder and LUMINA only dumps shell-0 tau. The convergence caveat (LUMINA
T_inner oscillated, outer T_rad 2.5-3.6x too hot) is annotated on panel C.
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"
OFF  = f"{ROOT}/logs/ddc15_frozenin_off_163486_0"
ON   = f"{ROOT}/logs/ddc15_frozenin_on_163486_1"
CMFGEN_SPEC = f"{ROOT}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"

def read_csv_cols(path):
    with open(path) as f:
        r = csv.reader(f); hdr = next(r)
        cols = {h: [] for h in hdr}
        for row in r:
            for h, v in zip(hdr, row):
                cols[h].append(float(v))
    return {h: np.array(v) for h, v in cols.items()}

# --- velocity grid (km/s, shell midpoint) ---
geo = read_csv_cols(f"{REF}/geometry.csv")
v_mid = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5   # cm/s -> km/s

# --- plasma profiles ---
ref_ne = read_csv_cols(f"{REF}/electron_densities.csv")["n_e"]
ref_ps = read_csv_cols(f"{REF}/plasma_state.csv")
off_ps = read_csv_cols(f"{OFF}/lumina_plasma_state.csv")
on_ps  = read_csv_cols(f"{ON}/lumina_plasma_state.csv")

# --- spectra ---
cm = np.loadtxt(CMFGEN_SPEC)
cm_w, cm_f = cm[:, 0], cm[:, 1]
off_sp = read_csv_cols(f"{OFF}/lumina_spectrum_formal.csv")
on_sp  = read_csv_cols(f"{ON}/lumina_spectrum_formal.csv")

def norm_optical(w, f, lo=3500, hi=9000):
    m = (w >= lo) & (w <= hi)
    area = np.trapz(f[m], w[m])
    return f / area if area > 0 else f

cm_fn  = norm_optical(cm_w, cm_f)
off_fn = norm_optical(off_sp["wavelength_angstrom"], off_sp["flux"])
on_fn  = norm_optical(on_sp["wavelength_angstrom"],  on_sp["flux"])

C_REF, C_OFF, C_ON = "k", "#3898EC", "#D97757"
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("DDC15 0.976d: LUMINA frozen-in A/B (job 163486) vs CMFGEN reference",
             fontsize=15, weight="bold")

# A: spectrum
a = ax[0, 0]
a.plot(cm_w, cm_fn, C_REF, lw=1.4, label="CMFGEN DDC15 0.976d")
a.plot(off_sp["wavelength_angstrom"], off_fn, C_OFF, lw=1.0, alpha=0.85,
       label="LUMINA OFF (steady-state)")
a.plot(on_sp["wavelength_angstrom"], on_fn, C_ON, lw=1.0, alpha=0.85,
       label="LUMINA ON (frozen-in)")
a.set_xlim(2500, 12000); a.set_xlabel("wavelength [Å]")
a.set_ylabel("flux (normalized to optical area 3500-9000Å)")
a.set_title("A. Spectrum"); a.legend(fontsize=9)

# B: n_e
b = ax[0, 1]
b.semilogy(v_mid, ref_ne, C_REF, marker="o", ms=3, label="CMFGEN")
b.semilogy(v_mid, off_ps["n_e"], C_OFF, marker="s", ms=3, label="LUMINA OFF")
b.semilogy(v_mid, on_ps["n_e"], C_ON, marker="^", ms=3, label="LUMINA ON")
b.set_xlabel("velocity [km/s]"); b.set_ylabel("n_e [cm$^{-3}$]")
b.set_title("B. Electron density  (CMFGEN flat outer plateau ~4.7e4)")
b.legend(fontsize=9); b.grid(alpha=0.3)

# C: T_rad
c = ax[1, 0]
c.plot(v_mid, ref_ps["T_rad"], C_REF, marker="o", ms=3, label="CMFGEN")
c.plot(v_mid, off_ps["T_rad"], C_OFF, marker="s", ms=3, label="LUMINA OFF")
c.plot(v_mid, on_ps["T_rad"], C_ON, marker="^", ms=3, label="LUMINA ON")
c.set_xlabel("velocity [km/s]"); c.set_ylabel("T_rad [K]")
c.set_title("C. Radiation temperature")
c.legend(fontsize=9); c.grid(alpha=0.3)
c.annotate("CAVEAT: LUMINA T_inner oscillated (4430→10.3kK→~4.8kK, unconverged)\n"
           "→ outer T_rad 2.5-3.6x hotter than CMFGEN 2505K.\n"
           "n_e agreement in panel B is partly coincidental (hot field compensates).",
           xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
           bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#B0AEA5"))

# D: W
d = ax[1, 1]
d.plot(v_mid, ref_ps["W"], C_REF, marker="o", ms=3, label="CMFGEN")
d.plot(v_mid, off_ps["W"], C_OFF, marker="s", ms=3, label="LUMINA OFF")
d.plot(v_mid, on_ps["W"], C_ON, marker="^", ms=3, label="LUMINA ON")
d.set_xlabel("velocity [km/s]"); d.set_ylabel("W (dilution factor)")
d.set_title("D. Geometric dilution factor")
d.legend(fontsize=9); d.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{ROOT}/figures/2026-06-05_ddc15_frozenin_ab_vs_cmfgen.png"
plt.savefig(out, dpi=130)
print("saved", out)

# --- print numeric comparison table (outer shells) ---
print("\n shell   v[km/s]   n_e CMFGEN   n_e OFF    n_e ON    Trad CMFGEN  Trad OFF  Trad ON")
for s in [0, 12, 24, 36, 42, 48]:
    print(f"  {s:3d}  {v_mid[s]:8.0f}  {ref_ne[s]:.3e}  {off_ps['n_e'][s]:.3e} "
          f"{on_ps['n_e'][s]:.3e}   {ref_ps['T_rad'][s]:7.1f}   {off_ps['T_rad'][s]:7.1f}  {on_ps['T_rad'][s]:7.1f}")
