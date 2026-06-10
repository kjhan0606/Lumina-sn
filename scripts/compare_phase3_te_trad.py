#!/usr/bin/env python3
"""Phase-3 J_nu->ionization A/B, judged by the T_e/T_rad view (the diagnostic
that was MISSING when this method was first tested and shelved as dormant).

The discriminator for whether J_nu photoionization is faithful is NOT n_e
dex-RMS alone but the relation between the gas temperature T_e and the field
color temperature T_rad that DRIVES the photoionization:
  - thick / faithful : deterministic J -> J=B(T_e) -> T_rad ~ T_e  -> photoion
    field matches the gas, ionization is self-consistent.
  - thin  / failure  : J anchored to photospheric W*B(T_phot) -> T_rad >> T_e
    -> photoion off a hotter-than-gas field over-ionizes (the A2 failure).
So (T_rad - T_e) per shell tells us WHERE the method is right vs wrong.

Usage: compare_phase3_te_trad.py LABEL=csv [LABEL=csv ...]
"""
import re, sys
import numpy as np
import pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
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
T_C = np.interp(vc, v_h, T_h)
ne_C = np.interp(vc, v_h, ne_h)

# parse LABEL=csv args
runs = []
for a in sys.argv[1:]:
    lab, path = a.split("=", 1)
    df = pd.read_csv(path)
    runs.append((lab, df))

if not runs:
    print("usage: compare_phase3_te_trad.py LABEL=csv [LABEL=csv ...]")
    sys.exit(1)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


inner = slice(0, 13)
trans = slice(13, 28)
outer = slice(28, nsh)

# CMFGEN gold is a single temperature (T_e ~ T_rad in the thick interior).
for lab, df in runs:
    Te = df.T_e.values
    Tr = df.T_rad.values
    W = df.W.values
    ne = df.n_e.values
    dex = np.log10(ne / ne_C)
    dmis = Tr - Te                       # field-vs-gas mismatch
    print(f"\n================ {lab} ================")
    print(f"{'sh':>3} {'v':>6} {'T_C':>6} {'T_rad':>6} {'T_e':>6} "
          f"{'Tr-Te':>7} {'W':>6} {'ne_C':>10} {'n_e':>10} {'dex':>6}")
    for i in range(nsh):
        flag = ""
        if dmis[i] > 300:
            flag = "  <== Tr>>Te (over-ion risk)"
        print(f"{i:>3} {vc[i]:>6.0f} {T_C[i]:>6.0f} {Tr[i]:>6.0f} {Te[i]:>6.0f} "
              f"{dmis[i]:>+7.0f} {W[i]:>6.3f} {ne_C[i]:>10.3e} {ne[i]:>10.3e} "
              f"{dex[i]:>+6.3f}{flag}")
    print(f"  n_e dex-RMS : all {rms(dex):.3f}  inner {rms(dex[inner]):.3f}  "
          f"trans {rms(dex[trans]):.3f}  outer {rms(dex[outer]):.3f}")
    print(f"  (Tr-Te) RMS : inner {rms(dmis[inner]):.0f}K  "
          f"trans {rms(dmis[trans]):.0f}K  outer {rms(dmis[outer]):.0f}K")
    print(f"  T_e %RMS    : all {rms((Te-T_C)/T_C*100):.1f}%  "
          f"outer {rms((Te[outer]-T_C[outer])/T_C[outer]*100):.1f}%")

# ---- figure: T_e & T_rad overlaid per arm + n_e dex + (Tr-Te) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sh = np.arange(nsh)
cols = ["#4EC9B0", "#D97757", "#3898EC", "#FFC107"]
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("Phase-3 J_nu -> ionization, judged by T_e vs T_rad (DDC15 0.976d)",
             fontsize=13, fontweight="bold")

# Key finding: T_rad of EVERY arm sits on the gold curve, so we draw gold once
# (= all arms' T_rad) and only spread the T_e lines. One faint dotted T_rad is
# overlaid to prove the coincidence without cluttering.
ax[0, 0].plot(sh, T_C, "k", lw=3.0, label="CMFGEN gold T  (= T_rad of all arms)")
ax[0, 0].plot(sh, runs[-1][1].T_rad.values, color="0.5", lw=1.0, ls=":",
              label="T_rad (sample arm, lies on gold)")
for k, (lab, df) in enumerate(runs):
    ax[0, 0].plot(sh, df.T_e.values, cols[k % 4], lw=1.8, marker="o", ms=3,
                  label=f"T_e {lab}")
ax[0, 0].set_title("T_e per arm vs gold  (T_rad collapses onto gold)")
ax[0, 0].set_xlabel("shell"); ax[0, 0].set_ylabel("T [K]")
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

ax[0, 1].semilogy(sh, ne_C, "k", lw=2.5, label="CMFGEN gold")
for k, (lab, df) in enumerate(runs):
    ax[0, 1].semilogy(sh, df.n_e.values, cols[k % 4], lw=1.6, marker="o", ms=3,
                      label=lab)
ax[0, 1].set_title("n_e [cm^-3]"); ax[0, 1].set_xlabel("shell")
ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3, which="both")

ax[1, 0].axhline(0, color="0.6", lw=1)
for k, (lab, df) in enumerate(runs):
    ax[1, 0].plot(sh, df.T_rad.values - df.T_e.values, cols[k % 4], lw=1.6,
                  marker="o", ms=3, label=lab)
ax[1, 0].set_title("T_rad - T_e [K]  (>0 = field hotter than gas -> over-ion driver)")
ax[1, 0].set_xlabel("shell"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

ax[1, 1].axhline(0, color="0.6", lw=1)
for k, (lab, df) in enumerate(runs):
    ax[1, 1].plot(sh, np.log10(df.n_e.values / ne_C), cols[k % 4], lw=1.6,
                  marker="o", ms=3, label=lab)
ax[1, 1].set_title("n_e residual log10(n_e/n_C) [dex]")
ax[1, 1].set_xlabel("shell"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

for a in ax.flat:
    a.axvspan(13, 28, color="0.85", alpha=0.3, zorder=0)
out = f"{ROOT}/figures/2026-06-09_ddc15_phase3_te_trad.png"
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(out, dpi=110)
print(f"\nfigure -> {out}")
