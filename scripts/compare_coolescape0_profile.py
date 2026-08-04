#!/usr/bin/env python3
"""Per-shell T_e/n_e profile of the COOL_ESCAPE=0 (collisional-net) run vs CMFGEN gold."""
import re, sys
import numpy as np
import pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rtruth_collis/lumina_plasma_state.csv"
HYDRO = f"{ROOT}/data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"
REF = f"{ROOT}/data/tardis_reference_ddc15_0p976d"


def hydro(key, n=115):
    f = open(HYDRO).read()
    i = f.find(key)
    blk = f[i:].split("\n", 1)[1]
    vals = re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]
    return np.array([float(x) for x in vals])


v_h = hydro("Velocity (km/s)")
order = np.argsort(v_h)
v_h = v_h[order]
T_h = (hydro("Temperature (10^4 K)") * 1e4)[order]
ne_h = hydro("Electron density")[order]

geo = pd.read_csv(f"{REF}/geometry.csv")
vc = 0.5 * (geo.v_inner + geo.v_outer).values / 1e5
nsh = len(vc)
T_C = np.interp(vc, v_h, ne_h * 0 + T_h) if False else np.interp(vc, v_h, T_h)
ne_C = np.interp(vc, v_h, ne_h)

run = pd.read_csv(RUN)
Te = run.T_e.values
ne = run.n_e.values

dexne = np.log10(ne / ne_C)
dTe = (Te - T_C) / T_C * 100

print(f"=== COOL_ESCAPE=0 profile vs CMFGEN gold  ({RUN}) ===")
print(f"{'sh':>3} {'v':>6} {'T_C':>6} {'T_e':>6} {'dT%':>7} {'ne_C':>10} {'n_e':>10} {'dex':>6}")
for i in range(nsh):
    print(f"{i:>3} {vc[i]:>6.0f} {T_C[i]:>6.0f} {Te[i]:>6.0f} {dTe[i]:>+7.1f} "
          f"{ne_C[i]:>10.3e} {ne[i]:>10.3e} {dexne[i]:>+6.3f}")

def rms(a): return float(np.sqrt(np.mean(a**2)))
inner = slice(0, 13); trans = slice(13, 28); outer = slice(28, nsh)
print("\n--- n_e dex-RMS ---")
print(f"  all   {rms(dexne):.3f}   inner0-12 {rms(dexne[inner]):.3f}  "
      f"trans13-27 {rms(dexne[trans]):.3f}  outer28-48 {rms(dexne[outer]):.3f}")
print("--- T_e %RMS ---")
print(f"  all   {rms(dTe):.1f}%  inner0-12 {rms(dTe[inner]):.1f}%  "
      f"trans13-27 {rms(dTe[trans]):.1f}%  outer28-48 {rms(dTe[outer]):.1f}%")
print(f"\n  T_e[0]={Te[0]:.0f} (CMFGEN {T_C[0]:.0f}, {dTe[0]:+.1f}%)   "
      f"n_e[0]={ne[0]:.3e} (CMFGEN {ne_C[0]:.3e}, {dexne[0]:+.3f} dex)")

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sh = np.arange(nsh)
CK, CL = "k", "#4EC9B0"
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("COOL_ESCAPE=0 (collisional-net) vs CMFGEN gold — DDC15 0.976d\n"
             "root restored + photosphere fixed, but outer T_e runs away hot",
             fontsize=13, fontweight="bold")

ax[0, 0].plot(sh, T_C, CK, lw=2.5, label="CMFGEN gold")
ax[0, 0].plot(sh, Te, CL, lw=1.8, marker="o", ms=3, label="LUMINA COOL_ESCAPE=0")
ax[0, 0].set_title(f"T_e [K]   (%RMS all {rms(dTe):.0f}%, outer28-48 {rms(dTe[outer]):.0f}%)")
ax[0, 0].set_xlabel("shell"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

ax[0, 1].semilogy(sh, ne_C, CK, lw=2.5, label="CMFGEN gold")
ax[0, 1].semilogy(sh, ne, CL, lw=1.8, marker="o", ms=3, label="LUMINA")
ax[0, 1].set_title(f"n_e [cm^-3]   (dex-RMS all {rms(dexne):.3f}, outer {rms(dexne[outer]):.3f})")
ax[0, 1].set_xlabel("shell"); ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3, which="both")

ax[1, 0].axhline(0, color="0.6", lw=1)
ax[1, 0].plot(sh, dTe, CL, lw=1.8, marker="o", ms=3)
ax[1, 0].fill_between(sh, 0, dTe, color=CL, alpha=0.2)
ax[1, 0].set_title("T_e residual (T_e-T_C)/T_C [%]  — outer over-heating")
ax[1, 0].set_xlabel("shell"); ax[1, 0].grid(alpha=0.3)

ax[1, 1].axhline(0, color="0.6", lw=1)
ax[1, 1].plot(sh, dexne, "#D97757", lw=1.8, marker="o", ms=3)
ax[1, 1].fill_between(sh, 0, dexne, color="#D97757", alpha=0.2)
ax[1, 1].set_title("n_e residual log10(n_e/n_C) [dex]  — sh7-13 over-ion from hot T_e")
ax[1, 1].set_xlabel("shell"); ax[1, 1].grid(alpha=0.3)

for a in ax.flat:
    a.axvspan(13, 28, color="0.85", alpha=0.3, zorder=0)
out = f"{ROOT}/figures/2026-06-09_ddc15_coolescape0_profile.png"
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out, dpi=110)
print(f"\nfigure -> {out}")
