#!/usr/bin/env python3
"""Frozen-in ionization vs CMFGEN as a function of freeze-out time t_0 (DDC15 0.976d).

Decision-support for the open t_0 question (see memory
project_timedep_backwardeuler_swamped_by_detailedbalance.md): the backward-Euler
matrix term is a no-op, but a physical-alpha recombination ODE over homologous
expansion (n_e ∝ t^-3) reproduces CMFGEN's outer <Z> -- IF a freeze-out start t_0
is chosen. Single shell (sh36) matched at t_0~0.10d. This sweeps t_0 across ALL
shells to answer: does ONE global t_0 fit the whole outer profile, or must t_0 be
shell-dependent?

Self-consistent frozen-in fixed point (fully-ionized IC, x_0=1):
    x = exp(-K x),  K = alpha(T)*n_atom*t_exp*0.5*((t_exp/t_0)^2 - 1)
with n_e = x*n_atom (Z_eff ~ 1 per ionization), alpha = 2.6e-13*(T/1e4)^-0.8.
Solved by damped iteration (stable for K>1, where naive x<-exp(-Kx) oscillates).
CAVEAT printed on the figure: T held at T(t_exp); early T was higher so alpha was
smaller -> this OVERESTIMATES recombination (true frozen <Z> a bit higher).
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYDRO = "data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"
REF = "data/tardis_reference_ddc15_0p976d"
TEXP_D = 0.976
TEXP = 84326.4
TD0 = "logs/paperDDC15init_ddc15init_macroatom_cro1_cfe1_kp0_ste0_rte0_mi200_gd0_td0_smspectrum_163383"


def hydro(key, n=115):
    f = open(HYDRO).read()
    i = f.find(key)
    blk = f[i:].split("\n", 1)[1]
    return np.array([float(x) for x in re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]])


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

alpha = 2.6e-13 * (T_C / 1e4) ** -0.8
tau_ratio = 1.0 / (ne_C * alpha) / TEXP  # using CMFGEN n_e


def frozen_x(t0_d, n_atom, T):
    """Self-consistent x = exp(-K x) per shell, damped fixed point."""
    a = 2.6e-13 * (T / 1e4) ** -0.8
    x0 = t0_d / TEXP_D
    K = a * n_atom * TEXP * 0.5 * (x0 ** -2 - 1.0)
    x = np.full_like(n_atom, 0.5)
    for _ in range(500):
        x_new = np.exp(-K * x)
        x = 0.5 * x + 0.5 * x_new  # damping
    return x, K


t0_list = [0.05, 0.10, 0.15, 0.20, 0.30]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(t0_list)))
sh = np.arange(nsh)

# LUMINA steady-state (td0) overlay
td0 = None
try:
    td0 = pd.read_csv(f"{TD0}/lumina_plasma_state.csv")
except Exception:
    pass

print("=== frozen-in <Z> vs CMFGEN: t_0 sweep (all shells) ===")
print(f"CMFGEN outer plateau <Z> ~ 0.53-0.55 (sh24-48)")
for t0 in t0_list:
    x, K = frozen_x(t0, na_C, T_C)
    o24 = x[24:].mean()
    print(f"  t_0={t0:.2f}d : <Z>_frozen[outer sh24-48] mean = {o24:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Panel 0: <Z>(shell) overlays
ax[0].plot(sh, Zbar_C, "k", lw=2.8, label="CMFGEN gold <Z>", zorder=5)
if td0 is not None:
    ax[0].plot(sh, td0.n_e / na_C, color="#D97757", lw=1.6, ls="--",
               label="LUMINA steady-state (td0)")
for t0, c in zip(t0_list, colors):
    x, _ = frozen_x(t0, na_C, T_C)
    ax[0].plot(sh, x, color=c, lw=1.8, label=f"frozen-in t_0={t0:.2f}d")
ax[0].axvline(np.argmin(np.abs(tau_ratio - 1.0)), color="gray", ls=":",
              label="tau_rec=t_exp crossover")
ax[0].set_xlabel("shell (0=photosphere)")
ax[0].set_ylabel("mean ionization <Z>")
ax[0].set_ylim(0, 1.05)
ax[0].set_title("Frozen-in <Z> vs CMFGEN: does ONE t_0 fit the whole profile?")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].grid(alpha=0.3)

# Panel 1: per-shell t_0 that EXACTLY matches CMFGEN <Z>, to expose shell-dependence
t0_match = np.full(nsh, np.nan)
t0_grid = np.linspace(0.02, 0.9, 400)
for i in range(nsh):
    if Zbar_C[i] >= 0.999 or Zbar_C[i] <= 1e-3:
        continue
    xs = np.array([frozen_x(t0, np.array([na_C[i]]), np.array([T_C[i]]))[0][0]
                   for t0 in t0_grid])
    j = np.argmin(np.abs(xs - Zbar_C[i]))
    t0_match[i] = t0_grid[j]
ax[1].plot(sh, t0_match, "o-", color="#4EC9B0", ms=4, lw=1.4)
ax[1].set_xlabel("shell")
ax[1].set_ylabel("t_0 [d] that reproduces CMFGEN <Z> at this shell")
ax[1].set_title("Required freeze-out t_0 per shell\n(flat => single global t_0 works; sloped => shell-dependent)")
ax[1].grid(alpha=0.3)
axt = ax[1].twinx()
axt.semilogy(sh, tau_ratio, color="#7080C0", lw=1.0, ls=":")
axt.axhline(1.0, color="#7080C0", lw=0.7, ls="--")
axt.set_ylabel("tau_rec/t_exp (CMFGEN n_e)", color="#7080C0")
axt.tick_params(axis="y", colors="#7080C0")

fig.suptitle("Frozen-in recombination ODE (homologous n_e∝t^-3) vs CMFGEN — t_0 decision support  "
             "[CAVEAT: T held at T(t_exp), overestimates recomb]")
fig.tight_layout()
out = "figures/2026-06-05_ddc15_frozen_in_t0_profile.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
print("\nPer-shell matching t_0 (sh: t_0[d]):")
for i in range(0, nsh, 3):
    print(f"  sh{i:>2} v={vc[i]:>6.0f}  <Z>_C={Zbar_C[i]:.3f}  tau/texp={tau_ratio[i]:>7.2f}  t0_match={t0_match[i]:.3f}")
