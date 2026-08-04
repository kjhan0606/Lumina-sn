#!/usr/bin/env python3
"""Time-dependent ionization (option A) A/B vs CMFGEN gold (DDC15 0.976d).

Overlays LUMINA n_e and mean ionization <Z>=n_e/n_atom for the steady-state
control (td0) and the backward-Euler frozen-in run (td1) against the CMFGEN
hydro-file gold, with the recombination/expansion timescale ratio tau_rec/t_exp
drawn on a twin axis. The prediction under test: td1 should lift the OUTER <Z>
toward CMFGEN's frozen-in 0.53 exactly where tau_rec/t_exp >> 1, while leaving
the inner (equilibrium) shells unchanged vs td0.
"""
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYDRO = "data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"
REF = "data/tardis_reference_ddc15_0p976d"
TEXP = 84326.4
TD0 = "logs/paperDDC15init_*_td0_*_163383"
TD1 = "logs/paperDDC15init_*_td1_*_163384"


def hydro(key, n=115):
    f = open(HYDRO).read()
    i = f.find(key)
    blk = f[i:].split("\n", 1)[1]
    return np.array([float(x) for x in re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]])


def lum(pat):
    g = glob.glob(f"{pat}/lumina_plasma_state.csv")
    return pd.read_csv(g[0]) if g else None


v_h = hydro("Velocity (km/s)")
o = np.argsort(v_h)
v_h = v_h[o]
ne_h = hydro("Electron density")[o]
na_h = hydro("Atom density")[o]
T_h = (hydro("Temperature (10^4 K)") * 1e4)[o]

geo = pd.read_csv(f"{REF}/geometry.csv")
vc = 0.5 * (geo.v_inner + geo.v_outer).values / 1e5
nsh = len(vc)
ne_C = np.interp(vc, v_h, ne_h)
na_C = np.interp(vc, v_h, na_h)
T_C = np.interp(vc, v_h, T_h)
Zbar_C = ne_C / na_C

alpha = 2.6e-13 * (T_h / 1e4) ** -0.8
tau_ratio_h = 1.0 / (ne_h * alpha) / TEXP
tau_ratio = np.interp(vc, v_h, tau_ratio_h)

td0 = lum(TD0)
td1 = lum(TD1)
sh = np.arange(nsh)

print("=== time-dependent ionization A/B vs CMFGEN (DDC15 0.976d) ===")
print(f"{'sh':>3} {'v':>6} {'T_C':>6} {'tau/texp':>9} {'<Z>_C':>6} "
      f"{'<Z>td0':>7} {'<Z>td1':>7} {'ne_td0/C':>9} {'ne_td1/C':>9}")
for i in range(0, nsh, 3):
    z0 = td0.n_e[i] / na_C[i] if td0 is not None else np.nan
    z1 = td1.n_e[i] / na_C[i] if td1 is not None else np.nan
    r0 = td0.n_e[i] / ne_C[i] if td0 is not None else np.nan
    r1 = td1.n_e[i] / ne_C[i] if td1 is not None else np.nan
    print(f"{i:>3} {vc[i]:>6.0f} {T_C[i]:>6.0f} {tau_ratio[i]:>9.2f} "
          f"{Zbar_C[i]:>6.3f} {z0:>7.3f} {z1:>7.3f} {r0:>9.3e} {r1:>9.3e}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
CK, C0, C1, CT = "k", "#D97757", "#4EC9B0", "#7080C0"

ax[0].semilogy(sh, ne_C, CK, lw=2.5, label="CMFGEN gold")
if td0 is not None: ax[0].semilogy(sh, td0.n_e, C0, lw=1.8, label="td0 steady-state")
if td1 is not None: ax[0].semilogy(sh, td1.n_e, C1, lw=1.8, label="td1 time-dependent")
ax[0].set_title("n_e [cm^-3]"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
ax[0].set_xlabel("shell (0=photosphere)")

ax[1].plot(sh, Zbar_C, CK, lw=2.5, label="CMFGEN <Z>")
if td0 is not None: ax[1].plot(sh, td0.n_e / na_C, C0, lw=1.8, label="td0 steady-state")
if td1 is not None: ax[1].plot(sh, td1.n_e / na_C, C1, lw=1.8, label="td1 time-dependent")
ax[1].set_title("mean ionization <Z> = n_e / n_atom"); ax[1].set_xlabel("shell")
ax[1].grid(alpha=0.3); ax[1].legend(loc="upper left")
axt = ax[1].twinx()
axt.semilogy(sh, tau_ratio, CT, lw=1.2, ls=":", label="tau_rec/t_exp")
axt.axhline(1.0, color=CT, lw=0.8, ls="--")
axt.set_ylabel("tau_rec / t_exp (frozen-in where >>1)", color=CT)
axt.tick_params(axis="y", colors=CT)

fig.suptitle("Time-dependent (frozen-in) ionization vs CMFGEN-gold — does the outer <Z> recover where tau_rec >> t_exp?")
fig.tight_layout()
out = "figures/2026-06-05_ddc15_timedep_ion_vs_cmfgen.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
