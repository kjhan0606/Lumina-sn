#!/usr/bin/env python3
"""COMPREHENSIVE CMFGEN-gold validation of LUMINA's plasma state (DDC15 0.976d).

Strategy: faithfully reproduce Blondin et al. (2015) and compare EVERY available
transport-independent plasma quantity shell-by-shell, so each mismatch can be
classified as MISSING physics (enable/implement) or WRONG physics (fix).

CMFGEN gold (115-pt hydro file): v, T_e(gas), n_atom, n_e, Rosseland chi, kappa.
LUMINA: geometry.csv (v,r), density.csv (rho), lumina_plasma_state.csv (W,T_rad,n_e).

Quantities compared: n_e, mean ionization <Z>=n_e/n_atom, T_e, electron-scattering
tau_es, dilution W (vs geometric), with CMFGEN Rosseland-tau photosphere as anchor.
"""
import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/tardis_reference_ddc15_0p976d"
HYDRO = "data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"
P_OFF = "logs/paperDDC15init_ddc15init_macroatom_cro1_cfe1_kp0_ste0_rte0_mi200_gd0_sm*_163304"
P_ON = "logs/paperDDC15init_ddc15init_macroatom_cro1_cfe1_kp0_ste0_rte0_mi200_gd1_sm*_163305"
SIGMA_TH = 6.6524587e-25  # cm^2
TE_RATIO = 0.9            # LUMINA hardwired T_e = 0.9*T_rad


def hydro(key, n=115):
    f = open(HYDRO).read()
    i = f.find(key)
    blk = f[i:].split("\n", 1)[1]
    vals = re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]
    return np.array([float(x) for x in vals])


def lum(pat):
    g = glob.glob(f"{pat}/lumina_plasma_state.csv")
    return pd.read_csv(g[0]) if g else None


# ---- CMFGEN gold (sorted ascending in velocity for interpolation) ----
v_h = hydro("Velocity (km/s)")
order = np.argsort(v_h)
v_h = v_h[order]
T_h = (hydro("Temperature (10^4 K)") * 1e4)[order]
na_h = hydro("Atom density")[order]
ne_h = hydro("Electron density")[order]
chi_h = (hydro("Rosseland mean opacity") * 1e-10)[order]  # cm^-1

# ---- LUMINA shell geometry ----
geo = pd.read_csv(f"{REF}/geometry.csv")
vc = 0.5 * (geo.v_inner + geo.v_outer).values / 1e5          # km/s shell center
r_in, r_out = geo.r_inner.values, geo.r_outer.values         # cm
dr = r_out - r_in
nsh = len(vc)

# CMFGEN quantities interpolated onto LUMINA shell-center velocities
ne_C = np.interp(vc, v_h, ne_h)
na_C = np.interp(vc, v_h, na_h)
T_C = np.interp(vc, v_h, T_h)
Zbar_C = ne_C / na_C

# CMFGEN Rosseland tau (from outer edge inward) -> photosphere velocity
Rh = (hydro("Radius grid") * 1e10)
chih = hydro("Rosseland mean opacity") * 1e-10
tau_R = np.zeros_like(Rh)
for i in range(1, len(Rh)):
    tau_R[i] = tau_R[i - 1] + 0.5 * (chih[i - 1] + chih[i]) * abs(Rh[i - 1] - Rh[i])
iph = int(np.searchsorted(tau_R, 2.0 / 3.0))
v_phot = hydro("Velocity (km/s)")[iph]


def tau_es(ne):
    """electron-scattering optical depth from outer boundary inward (LUMINA shells)."""
    t = np.zeros(nsh)
    acc = 0.0
    for i in range(nsh - 1, -1, -1):
        acc += ne[i] * SIGMA_TH * dr[i]
        t[i] = acc
    return t


off = lum(P_OFF)
on = lum(P_ON)

# geometric dilution factor at shell center (photosphere = inner boundary r_in[0])
r_phot = r_in[0]
rc = 0.5 * (r_in + r_out)
W_geo = 0.5 * (1.0 - np.sqrt(np.clip(1.0 - (r_phot / rc) ** 2, 0, 1)))

print("=== CMFGEN-gold validation (DDC15 0.976d) ===")
print(f"  CMFGEN photosphere tau_Ross=2/3 at v={v_phot:.0f} km/s "
      f"(LUMINA inner boundary v={vc[0]:.0f} km/s)")
print(f"\n{'sh':>3} {'v':>6} {'<Z>_C':>6} {'<Z>off':>7} {'<Z>on':>7} "
      f"{'ne_off/C':>8} {'ne_on/C':>8} {'T_C':>6} {'Te_off':>6}")
for i in range(0, nsh, 4):
    zo = off.n_e[i] / na_C[i] if off is not None else np.nan
    zn = on.n_e[i] / na_C[i] if on is not None else np.nan
    ro = off.n_e[i] / ne_C[i] if off is not None else np.nan
    rn = on.n_e[i] / ne_C[i] if on is not None else np.nan
    teo = TE_RATIO * off.T_rad[i] if off is not None else np.nan
    print(f"{i:>3} {vc[i]:>6.0f} {Zbar_C[i]:>6.3f} {zo:>7.3f} {zn:>7.3f} "
          f"{ro:>8.3f} {rn:>8.3f} {T_C[i]:>6.0f} {teo:>6.0f}")

# ---- figure: 2x3 panels ----
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
sh = np.arange(nsh)
COFF, CON, CK = "#D97757", "#4EC9B0", "k"

ax[0, 0].semilogy(sh, ne_C, CK, lw=2.5, label="CMFGEN gold")
if off is not None: ax[0, 0].semilogy(sh, off.n_e, COFF, lw=1.8, label="LUMINA gamma-OFF")
if on is not None: ax[0, 0].semilogy(sh, on.n_e, CON, lw=1.8, label="LUMINA gamma-ON")
ax[0, 0].set_title("n_e [cm^-3]"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3, which="both")

ax[0, 1].plot(sh, Zbar_C, CK, lw=2.5, label="CMFGEN <Z>")
if off is not None: ax[0, 1].plot(sh, off.n_e / na_C, COFF, lw=1.8, label="LUMINA OFF")
if on is not None: ax[0, 1].plot(sh, on.n_e / na_C, CON, lw=1.8, label="LUMINA ON")
ax[0, 1].set_title("mean ionization <Z> = n_e / n_atom"); ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

if off is not None: ax[0, 2].semilogy(sh, off.n_e / ne_C, COFF, lw=1.8, label="OFF / CMFGEN")
if on is not None: ax[0, 2].semilogy(sh, on.n_e / ne_C, CON, lw=1.8, label="ON / CMFGEN")
ax[0, 2].axhline(1, color=CK, ls=":")
ax[0, 2].set_title("n_e ratio LUMINA/CMFGEN (<1 = under-ionized)"); ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3, which="both")

ax[1, 0].plot(sh, T_C, CK, lw=2.5, label="CMFGEN T_e")
if off is not None:
    ax[1, 0].plot(sh, off.T_rad, COFF, lw=1.2, ls="--", label="LUMINA T_rad")
    ax[1, 0].plot(sh, TE_RATIO * off.T_rad, COFF, lw=1.8, label="LUMINA T_e=0.9T_rad")
ax[1, 0].set_title("temperature [K]"); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

ax[1, 1].semilogy(sh, tau_es(ne_C), CK, lw=2.5, label="CMFGEN n_e")
if off is not None: ax[1, 1].semilogy(sh, tau_es(off.n_e.values), COFF, lw=1.8, label="LUMINA OFF")
if on is not None: ax[1, 1].semilogy(sh, tau_es(on.n_e.values), CON, lw=1.8, label="LUMINA ON")
ax[1, 1].axhline(2 / 3, color="gray", ls=":", label="tau=2/3")
ax[1, 1].set_title("electron-scattering tau (outer->in)"); ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3, which="both")

ax[1, 2].plot(sh, W_geo, CK, lw=2.5, label="geometric W")
if off is not None: ax[1, 2].plot(sh, off.W, COFF, lw=1.8, label="LUMINA W")
ax[1, 2].axhline(0.5, color="gray", ls=":", label="W=0.5 (phot)")
ax[1, 2].set_title("dilution factor W"); ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)

for a in ax.flat:
    a.set_xlabel("shell (0=photosphere)")
fig.suptitle("LUMINA vs CMFGEN-gold plasma validation (DDC15 0.976d) — classify each mismatch: missing vs wrong physics",
             fontsize=13)
fig.tight_layout()
out = "figures/2026-06-04_ddc15_plasma_validation_vs_cmfgen.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
