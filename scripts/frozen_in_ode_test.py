#!/usr/bin/env python3
"""Standalone numerical test of the Chugai/Potashov single-epoch freeze-out ODE
BEFORE touching the C code (DDC15 0.976d).

Question: does the PARAMETER-FREE freeze-out prescription actually reproduce
CMFGEN's outer <Z>~0.53 plateau + inner monotonic rise, when we integrate the
real recombination ODE over the homologous density history n_e(t) ∝ t^-3?

Method (per shell, fully self-contained):
  1. t_0 from the decoupling criterion  tau_rec(t_0) = t_0
       tau_rec(t) = 1/(n_e(t)*alpha),  n_e(t) = n_e(t_exp)*(t_exp/t)^3
       =>  t_0 = sqrt(alpha * n_e(t_exp) * t_exp^3)
     (n_e(t_exp) = CMFGEN observed electron density)
  2. seed x(t_0) = 1  (fully ionized near explosion: before t_0 recomb is faster
     than expansion so the gas tracks equilibrium, which is ~fully ionized hot)
  3. integrate  dx/dt = -alpha(T) * n_e(t) * x   from t_0 to t_exp,
     with n_e(t) = x(t) * n_atom(t),  n_atom(t) = n_atom(t_exp)*(t_exp/t)^3
     (Gamma photoionization neglected in the frozen outer ejecta; recomb integral
      is early-time/high-density dominated)
  4. compare x(t_exp) to CMFGEN <Z>.

If t_0 >= t_exp (inner, dense) the shell never freezes -> report steady-state
(x -> equilibrium), which is the auto-recovery the memory predicts.

CAVEAT (same as frozen_in_t0_profile.py): T held at T(t_exp). Early T was higher
=> alpha smaller => this OVERestimates recombination (true frozen <Z> a bit higher).
A second pass with a homologous T(t) ~ T(t_exp)*(t_exp/t) (radiation-T scaling)
is run as a sensitivity bracket.
"""
import re
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYDRO = "data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"
REF = "data/tardis_reference_ddc15_0p976d"
TEXP_D = 0.976
TEXP = 84326.4  # s
DAY = 86400.0


def hydro(key, n=115):
    f = open(HYDRO).read()
    i = f.find(key)
    blk = f[i:].split("\n", 1)[1]
    return np.array([float(x) for x in re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]])


# --- load hydro, sort by velocity (shell0 = innermost after sort) ---
v_h = hydro("Velocity (km/s)")
o = np.argsort(v_h)
v_h = v_h[o]
ne_h = hydro("Electron density")[o]
na_h = hydro("Atom density")[o]
T_h = (hydro("Temperature (10^4 K)") * 1e4)[o]

# --- CMFGEN reference on LUMINA geometry grid ---
geo = pd.read_csv(f"{REF}/geometry.csv")
vc = 0.5 * (geo.v_inner + geo.v_outer).values / 1e5  # km/s
nsh = len(vc)
ne_C = np.interp(vc, v_h, ne_h)
na_C = np.interp(vc, v_h, na_h)
T_C = np.interp(vc, v_h, T_h)
Zbar_C = ne_C / na_C


def alpha_rec(T):
    return 2.6e-13 * (T / 1e4) ** -0.8


def solve_shell(i, T_homologous=False):
    """Return (t0_d, x_final, froze).  x_final = ionization fraction at t_exp."""
    ne0 = ne_C[i]            # n_e at t_exp (CMFGEN)
    na0 = na_C[i]            # n_atom at t_exp
    Texp = T_C[i]
    a_exp = alpha_rec(Texp)

    # criterion t_0 = sqrt(alpha * n_e(t_exp) * t_exp^3)
    t0 = np.sqrt(a_exp * ne0 * TEXP ** 3)
    t0_d = t0 / DAY

    if t0 >= TEXP:
        # never freezes -> equilibrium (steady-state) recovers; report CMFGEN-like
        # equilibrium is whatever steady-state gives; here mark as "equilibrium"
        return t0_d, np.nan, False

    def T_of_t(t):
        if T_homologous:
            return Texp * (TEXP / t)            # radiation-T ~ 1/t scaling
        return Texp

    def rhs(t, y):
        x = max(y[0], 0.0)
        na_t = na0 * (TEXP / t) ** 3
        ne_t = x * na_t
        a = alpha_rec(T_of_t(t))
        return [-a * ne_t * x]

    sol = solve_ivp(rhs, [t0, TEXP], [1.0], method="LSODA",
                    rtol=1e-8, atol=1e-12, dense_output=False)
    xf = float(np.clip(sol.y[0, -1], 0.0, 1.0))
    return t0_d, xf, True


for label, homo in [("T=T(texp) [overest. recomb]", False),
                    ("T~1/t homologous [bracket]", True)]:
    t0s = np.zeros(nsh)
    xf = np.full(nsh, np.nan)
    froze = np.zeros(nsh, bool)
    for i in range(nsh):
        t0s[i], xf[i], froze[i] = solve_shell(i, T_homologous=homo)
    o24 = np.nanmean(xf[24:])
    print(f"=== {label} ===")
    print(f"  outer sh24-48 mean x_frozen = {o24:.3f}  (CMFGEN plateau ~0.53)")
    print(f"  t_0 range: inner sh0={t0s[0]/1:.3f}d ... outer sh48={t0s[-1]:.3f}d")
    nfroze = froze.sum()
    print(f"  shells that froze (t_0<t_exp): {nfroze}/{nsh}; "
          f"steady-state (t_0>=t_exp): {nsh-nfroze}")
    if not homo:
        t0s_save, xf_save, froze_save = t0s.copy(), xf.copy(), froze.copy()

# --- figure ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sh = np.arange(nsh)

ax[0].plot(sh, Zbar_C, "k", lw=2.8, label="CMFGEN gold <Z>", zorder=5)
# baseline (T fixed)
xf_plot = xf_save.copy()
# fill steady-state (unfrozen) shells with CMFGEN value as the "equilibrium recovers" marker
ss = ~froze_save
ax[0].plot(sh, xf_plot, "o-", color="#D97757", ms=4, lw=1.6,
           label="frozen-in ODE (param-free t_0, T fixed)")
if ss.any():
    ax[0].plot(sh[ss], Zbar_C[ss], "s", color="#4EC9B0", ms=7, mfc="none",
               label="t_0>=t_exp: steady-state (equilibrium)")
# homologous-T bracket
t0h = np.zeros(nsh); xfh = np.full(nsh, np.nan); frh = np.zeros(nsh, bool)
for i in range(nsh):
    t0h[i], xfh[i], frh[i] = solve_shell(i, T_homologous=True)
ax[0].plot(sh, xfh, "--", color="#3898EC", lw=1.4,
           label="frozen-in ODE (T~1/t bracket)")
ax[0].axvline(np.argmax(froze_save), color="gray", ls=":",
              label="freeze-out onset (t_0<t_exp)")
ax[0].set_xlabel("shell (0=innermost)")
ax[0].set_ylabel("ionization fraction x  (≈ <Z> for single-stage)")
ax[0].set_ylim(0, 1.05)
ax[0].set_title("Param-free freeze-out ODE vs CMFGEN <Z>\n(does the criterion reproduce the plateau?)")
ax[0].legend(fontsize=8, loc="center left")
ax[0].grid(alpha=0.3)

# panel 1: t_0 profile from the criterion + tau_rec/t_exp
ax[1].plot(sh, t0s_save, "o-", color="#FFC107", ms=4, lw=1.4,
           label="t_0 (criterion) [d]")
ax[1].axhline(TEXP_D, color="k", ls="--", lw=0.8, label="t_exp=0.976d")
ax[1].set_xlabel("shell")
ax[1].set_ylabel("freeze-out t_0 [d]  (param-free)")
ax[1].set_title("Parameter-free t_0 = sqrt(alpha*n_e*t_exp^3) per shell")
ax[1].legend(fontsize=8, loc="upper right")
ax[1].grid(alpha=0.3)
axt = ax[1].twinx()
tau_ratio = 1.0 / (ne_C * alpha_rec(T_C)) / TEXP
axt.semilogy(sh, tau_ratio, color="#7080C0", lw=1.0, ls=":")
axt.axhline(1.0, color="#7080C0", lw=0.7, ls="--")
axt.set_ylabel("tau_rec/t_exp", color="#7080C0")
axt.tick_params(axis="y", colors="#7080C0")

fig.suptitle("Frozen-in recombination ODE numerical test (pre-implementation) — DDC15 0.976d")
fig.tight_layout()
out = "figures/2026-06-05_ddc15_frozen_in_ode_test.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)

print("\nPer-shell (sh: v  <Z>_C  t0[d]  x_frozen  froze):")
for i in range(0, nsh, 3):
    fr = "froze" if froze_save[i] else "STEADY"
    xs = f"{xf_save[i]:.3f}" if froze_save[i] else "  -  "
    print(f"  sh{i:>2} v={vc[i]:>6.0f}  <Z>_C={Zbar_C[i]:.3f}  t0={t0s_save[i]:>6.3f}d  x={xs}  {fr}")
