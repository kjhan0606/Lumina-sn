#!/usr/bin/env python3
"""Multi-stage frozen-in freeze-out prototype (DDC15 0.976d), pre-C-implementation.

Extends scripts/frozen_in_ode_test.py from a SCALAR ionization fraction to a
per-element, per-ION-STAGE recombination cascade. The scalar test reproduced
CMFGEN's outer <Z>~0.53 plateau (0.553) but cannot say WHICH ion (II vs III)
carries the opacity. This prototype tracks the stage partition so we can:
  (1) check the self-consistent multi-stage n_e (= sum over elements/stages of
      k*n_{Z,k}) STILL reproduces CMFGEN <Z> (the only directly observable),
  (2) report the predicted ion partition per element (spectral-opacity relevant),
  (3) confirm the parameter-free t_0 still controls the freeze cleanly.

Physics (fraction-space, recombination-only freeze-out; photoionization neglected
in the frozen ejecta, exact in the cascade limit):
  state y_{Z,k} = fraction of element Z in ion stage k (sum_k y=1 per element).
  dy_{Z,k}/dt = alpha_k * n_e * y_{Z,k+1}  -  alpha_{k-1} * n_e * y_{Z,k}
  alpha_k (recomb from stage k+1, ion charge k+1, to stage k)
        = 2.6e-13 * (k+1) * (T/1e4)^-0.8   [hydrogenic, charge-linear]
  n_e(t) = n_atom(t) * sum_Z f_Z * <Z>_Z,   n_atom(t)=n_atom(texp)*(texp/t)^3
  <Z>_Z = sum_k k*y_{Z,k},  f_Z = number fraction of element Z.
  (Homologous t^-3 dilution cancels in fraction-space: element number conserved,
   only the n_e prefactor in the rates carries it.)

Per shell: t_0 = sqrt(alpha_1 * n_e(texp) * texp^3)  [singly-ionized criterion].
  If t_0 >= texp: shell never freezes -> steady-state (NLTE handles it), skip.
  Else: seed fully ionized (top tracked stage) at t_0, integrate t_0 -> texp,
  fixed T=T(texp) (adequate per scalar test: recomb integral concentrates near t_0).

NOTE: CMFGEN per-ion populations are NOT in the reference (plasma_state.csv has
only W,T_rad), so we validate the AGGREGATE <Z>=n_e/n_atom (observable) and
present the predicted partition as a physical sanity check, not a gold compare.
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
TEXP = 84326.4
TEXP_D = 0.976
DAY = 86400.0
NSTAGE = 4  # track stages 0..3 (I..IV); enough for outer <Z><~0.6


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

# --- composition: mass fractions -> number fractions per shell ---
ab = pd.read_csv(f"{REF}/abundances.csv")
Zlist = ab.atomic_number.values
Xmass = ab.iloc[:, 1:].values  # (nelem, nsh) mass fractions
masses = pd.read_csv(f"{REF}/atom_masses.csv").set_index("atomic_number").mass_amu
A = np.array([masses[z] for z in Zlist])
nfrac = (Xmass / A[:, None])
nfrac /= nfrac.sum(axis=0, keepdims=True)  # number fraction per shell, sums to 1
maxstage = pd.read_csv(f"{REF}/ionization_energies.csv").groupby(
    "atomic_number").ion_number.max()
# top ion stage available per element (stage index = max chi ion_number + 1)
topstage = np.array([min(NSTAGE - 1, int(maxstage.get(z, 0)) + 1) for z in Zlist])
nelem = len(Zlist)


def alpha_charge(T):
    """alpha_k for k=0..NSTAGE-2: recomb from stage k+1 (charge k+1)."""
    base = 2.6e-13 * (T / 1e4) ** -0.8
    return np.array([base * (k + 1) for k in range(NSTAGE - 1)])


def solve_shell(i):
    """Return (t0_d, Zbar_final, froze, partition[nelem,NSTAGE])."""
    Texp = T_C[i]
    na0 = na_C[i]
    f = nfrac[:, i]
    a = alpha_charge(Texp)
    a1 = 2.6e-13 * (Texp / 1e4) ** -0.8  # singly-ionized (charge 1)
    t0 = np.sqrt(a1 * ne_C[i] * TEXP ** 3)
    t0_d = t0 / DAY
    if t0 >= TEXP:
        return t0_d, np.nan, False, None

    # IC at t_0 = EQUILIBRIUM partition, NOT max-ionized. Physics: each
    # transition (k+1->k) freezes at its own t_0(k)=sqrt(alpha_k*n_e*t_exp^3);
    # higher-charge alpha is larger -> larger t_0 -> those transitions stay
    # coupled (in equilibrium) PAST the II->I freeze, so by the dominant
    # singly-ionized freeze epoch everything above stage k_seed has recombined
    # down to it. Seed at the highest stage k whose t_0(k) still >= t_exp/?-...
    # for DDC15 outer the dominant freeze is II->I, so k_seed=1 (singly ionized).
    # Generalized: k_seed = highest stage k>=1 with t_0(k-1)=sqrt(alpha_{k-1}*ne*texp^3) < t_exp
    #   (i.e. that transition froze in-flight); else the top stage that stays coupled.
    a_all = alpha_charge(Texp)  # a_all[k] = recomb from stage k+1
    k_seed = 1
    for k in range(1, NSTAGE):
        t0k = np.sqrt(a_all[k - 1] * ne_C[i] * TEXP ** 3)  # freeze of k->k-1
        if t0k < TEXP:
            k_seed = k  # this transition froze before t_exp -> stage k can survive
            break
    y0 = np.zeros((nelem, NSTAGE))
    for e in range(nelem):
        ks = min(k_seed, topstage[e])
        y0[e, ks] = 1.0
    y0 = y0.ravel()

    def rhs(t, yflat):
        y = yflat.reshape(nelem, NSTAGE)
        na_t = na0 * (TEXP / t) ** 3
        Zbar_e = (y * np.arange(NSTAGE)[None, :]).sum(axis=1)
        ne = na_t * (f * Zbar_e).sum()
        dy = np.zeros_like(y)
        for k in range(NSTAGE):
            inflow = a[k] * ne * y[:, k + 1] if k + 1 < NSTAGE else 0.0
            outflow = a[k - 1] * ne * y[:, k] if k - 1 >= 0 else 0.0
            dy[:, k] = inflow - outflow
        return dy.ravel()

    sol = solve_ivp(rhs, [t0, TEXP], y0, method="LSODA",
                    rtol=1e-7, atol=1e-10)
    y = sol.y[:, -1].reshape(nelem, NSTAGE)
    y = np.clip(y, 0, 1)
    y /= y.sum(axis=1, keepdims=True)
    Zbar_e = (y * np.arange(NSTAGE)[None, :]).sum(axis=1)
    Zbar = (f * Zbar_e).sum()
    return t0_d, Zbar, True, y


t0s = np.zeros(nsh)
Zf = np.full(nsh, np.nan)
froze = np.zeros(nsh, bool)
parts = [None] * nsh
for i in range(nsh):
    t0s[i], Zf[i], froze[i], parts[i] = solve_shell(i)

om = np.nanmean(Zf[froze])
print("=== Multi-stage frozen-in prototype (DDC15 0.976d) ===")
print(f"outer-frozen mean <Z> = {om:.3f}   (CMFGEN plateau ~0.53; scalar test gave 0.553)")
print(f"shells frozen (t_0<t_exp): {froze.sum()}/{nsh}")
print("\nsh  v[km/s]  <Z>_C  <Z>_ODE  t0[d]   dominant-ion partition (frozen shells)")
EL = {6:"C",8:"O",12:"Mg",13:"Al",14:"Si",16:"S",20:"Ca",21:"Sc",22:"Ti",
      23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni"}
ROM = ["I", "II", "III", "IV"]
for i in range(0, nsh, 3):
    if not froze[i]:
        print(f"sh{i:>2} v={vc[i]:>6.0f}  <Z>_C={Zbar_C[i]:.3f}  STEADY (t0={t0s[i]:.2f}d>t_exp)")
        continue
    y = parts[i]
    # report O and Si partition (outer opacity carriers)
    bits = []
    for z in (8, 14, 26):
        e = list(Zlist).index(z)
        kdom = int(np.argmax(y[e]))
        bits.append(f"{EL[z]}{ROM[kdom]}={y[e,kdom]:.2f}")
    print(f"sh{i:>2} v={vc[i]:>6.0f}  <Z>_C={Zbar_C[i]:.3f}  <Z>_ODE={Zf[i]:.3f}"
          f"  t0={t0s[i]:.3f}d  " + " ".join(bits))

# --- figure ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sh = np.arange(nsh)
ax[0].plot(sh, Zbar_C, "k", lw=2.8, label="CMFGEN gold <Z>", zorder=5)
ax[0].plot(sh, Zf, "o-", color="#D97757", ms=4, lw=1.6,
           label="multi-stage frozen ODE <Z>")
ss = ~froze
if ss.any():
    ax[0].plot(sh[ss], Zbar_C[ss], "s", color="#4EC9B0", ms=7, mfc="none",
               label="t_0>=t_exp: steady-state (NLTE)")
ax[0].set_xlabel("shell (0=innermost)")
ax[0].set_ylabel("mean ionization <Z> = n_e/n_atom")
ax[0].set_ylim(0, 1.05)
ax[0].set_title("Multi-stage frozen-in <Z> vs CMFGEN (self-consistent n_e)")
ax[0].legend(fontsize=8, loc="center left")
ax[0].grid(alpha=0.3)

# panel 1: O and Si ion partition vs shell (frozen region)
for z, col in [(8, "#3898EC"), (14, "#FFC107")]:
    e = list(Zlist).index(z)
    for k, ls in [(0, "-"), (1, "--"), (2, ":")]:
        yk = np.array([parts[i][e, k] if froze[i] else np.nan for i in range(nsh)])
        ax[1].plot(sh, yk, ls, color=col, lw=1.4,
                   label=f"{EL[z]} {ROM[k]}")
ax[1].set_xlabel("shell")
ax[1].set_ylabel("ion fraction (frozen shells)")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("Predicted frozen ion partition: O (blue) & Si (gold)\nsolid=neutral dash=singly dot=doubly")
ax[1].legend(fontsize=7, ncol=2)
ax[1].grid(alpha=0.3)

fig.suptitle("Multi-stage frozen-in freeze-out prototype — DDC15 0.976d (pre-implementation)")
fig.tight_layout()
out = "figures/2026-06-05_ddc15_frozen_in_multistage.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
