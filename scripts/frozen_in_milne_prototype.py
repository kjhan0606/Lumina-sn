#!/usr/bin/env python3
"""FAITHFUL frozen-in prototype: per-ion alpha_rec from CMFGEN sigma_bf via the
Milne relation (NO hydrogenic approximation), fed into the multi-stage freeze-out
ODE. DDC15 0.976d. Pre-C-implementation validation.

Previous prototypes (frozen_in_multistage_prototype.py) used a hydrogenic
alpha = 2.6e-13*(T/1e4)^-0.8 for every ion -> reproduced <Z>=0.553 but gave an
element-UNIFORM partition (all 0.53 II), an artifact of one alpha for all species.
This computes the REAL per-ion radiative recombination coefficient from the same
CMFGEN photoionization cross-sections the C code already carries
(cmfgen_sigma_bf.bin), via the Milne relation:

  alpha_l(T) = [ sum_nu 4*pi*B(nu,T)*sigma_l(nu)/(h*nu) d_nu ]
               * lambda_dB^3 * g_l/(2 g_ion) * exp(chi_l/kT)
  alpha_RR(Z,i,T) = sum over levels l of ion i (recomb from i+1 lands in ion i)

with B = Planck, lambda_dB^3 = (h^2/(2 pi m_e k T))^1.5, chi_l = chi_ion(Z,i) - E_l.
This is EXACTLY the integral the C NLTE assembly does against J_nu (lumina_plasma.c
~3719-3760), but against B(T_e) -> the LTE radiative recomb rate (Milne).

Question answered: does faithful Milne-alpha (a) still reproduce CMFGEN's outer
<Z>~0.53 plateau, and (b) SPLIT the partition across elements (O vs Si vs Fe)?

CAVEAT: radiative recomb only (RR). Dielectronic (DR) not included here (the C
code has a separate DR table); DR is subdominant at the 2500K O/Si-dominated
outer. T held at T_e(t_exp) (adequate per scalar test). Self-consistent n_e by
outer iteration.
"""
import re
import struct
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYDRO = "data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"
REF = "data/tardis_reference_ddc15_0p976d"
SIGMA = f"{REF}/cmfgen_sigma_bf.bin"
LEVELS = f"{REF}/levels.csv"
TEXP = 84326.4
TEXP_D = 0.976
DAY = 86400.0
NSTAGE = 4

# physical constants (CGS)
H = 6.62607015e-27
KB = 1.380649e-16
ME = 9.1093837015e-28
C = 2.99792458e10
EV = 1.602176634e-12


def hydro(key, n=115):
    f = open(HYDRO).read()
    i = f.find(key)
    blk = f[i:].split("\n", 1)[1]
    return np.array([float(x) for x in re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]])


# --- load sigma_bf grid ---
with open(SIGMA, "rb") as fp:
    magic, ver, nlev, nfreq = struct.unpack("<IIii", fp.read(16))
    numin, numax = struct.unpack("<dd", fp.read(16))
    flag8 = np.frombuffer(fp.read(nlev), dtype=np.int8).astype(bool)
    pad = (8 - (nlev % 8)) % 8
    fp.read(pad)
    sigma = np.frombuffer(fp.read(nlev * nfreq * 8), dtype=np.float64).reshape(nlev, nfreq)
assert magic == 0x434D4644 and ver == 1

# freq grid (log-spaced, matches C: bin center & width)
dlog = (np.log(numax) - np.log(numin)) / nfreq
edges = np.exp(np.log(numin) + np.arange(nfreq + 1) * dlog)
nu_c = np.sqrt(edges[:-1] * edges[1:])          # geometric bin center ~ exp(lo+0.5dlog)
nu_c = np.exp(np.log(numin) + (np.arange(nfreq) + 0.5) * dlog)
dnu = edges[1:] - edges[:-1]

lev = pd.read_csv(LEVELS)
assert len(lev) == nlev, (len(lev), nlev)
lev_Z = lev.atomic_number.values
lev_ion = lev.ion_number.values
lev_E = lev.energy_eV.values            # eV above that ion's ground
lev_g = lev.g.values.astype(float)

ioniz = pd.read_csv(f"{REF}/ionization_energies.csv")
chi_map = {(r.atomic_number, r.ion_number): r.ionization_energy_eV
           for r in ioniz.itertuples()}
# ground g per (Z,ion) = g of level_number 0
gnd_g = {}
for (z, i), grp in lev.groupby(["atomic_number", "ion_number"]):
    g0 = grp.loc[grp.level_number == 0, "g"]
    gnd_g[(z, i)] = float(g0.iloc[0]) if len(g0) else 1.0


def planck_nu(T):
    x = H * nu_c / (KB * T)
    x = np.clip(x, 1e-8, 700)
    return (2 * H * nu_c**3 / C**2) / (np.expm1(x))


# Precompute, per (Z, ion i), the Milne RR alpha as a function of T via a small
# T-grid then interpolate (alpha depends on T only).
def alpha_RR_table(Z, i, T):
    """Total radiative recomb coeff [cm^3/s] producing ion i (from ion i+1)."""
    chi_ion = chi_map.get((Z, i))
    if chi_ion is None:
        return 0.0
    g_ion = gnd_g.get((Z, i + 1), 1.0)
    lam3 = (H * H / (2 * np.pi * ME * KB * T)) ** 1.5
    sel = np.where((lev_Z == Z) & (lev_ion == i) & flag8)[0]
    if len(sel) == 0:
        return 0.0
    B = planck_nu(T)
    a_tot = 0.0
    for l in sel:
        chi_l = (chi_ion - lev_E[l]) * EV
        if chi_l <= 0:
            continue
        nu_th = chi_l / H
        m = nu_c >= nu_th
        if not m.any():
            continue
        Rbf_planck = np.sum(4 * np.pi * B[m] * sigma[l, m] / (H * nu_c[m]) * dnu[m])
        a_tot += Rbf_planck * lam3 * lev_g[l] / (2.0 * g_ion) * np.exp(chi_l / (KB * T))
    return a_tot


# --- geometry / CMFGEN ref ---
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

ab = pd.read_csv(f"{REF}/abundances.csv")
Zlist = ab.atomic_number.values
Xmass = ab.iloc[:, 1:].values
masses = pd.read_csv(f"{REF}/atom_masses.csv").set_index("atomic_number").mass_amu
A = np.array([masses[z] for z in Zlist])
nfrac = (Xmass / A[:, None])
nfrac /= nfrac.sum(axis=0, keepdims=True)
nelem = len(Zlist)
maxstage = ioniz.groupby("atomic_number").ion_number.max()
topstage = np.array([min(NSTAGE - 1, int(maxstage.get(z, 0)) + 1) for z in Zlist])

EL = {6:"C",8:"O",12:"Mg",13:"Al",14:"Si",16:"S",20:"Ca",21:"Sc",22:"Ti",
      23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni"}
ROM = ["I", "II", "III", "IV"]

# Precompute per-element per-stage alpha at each shell's T (T fixed = T_e(texp))
print("Precomputing Milne RR alpha per (element,stage,shell)... ", flush=True)
# alpha_es[e,k,s] = recomb producing stage k of element e at shell s temperature
alpha_es = np.zeros((nelem, NSTAGE - 1, nsh))
# cache by (Z,i,Tround) to cut cost
cache = {}
for s in range(nsh):
    T = T_C[s]
    Tk = round(T, -1)
    for e in range(nelem):
        Z = int(Zlist[e])
        for k in range(NSTAGE - 1):
            key = (Z, k, Tk)
            if key not in cache:
                cache[key] = alpha_RR_table(Z, k, T)
            alpha_es[e, k, s] = cache[key]
print("done.", flush=True)


def solve_shell(s, ne_texp):
    """Multi-stage frozen ODE for shell s, given the t_exp electron density.
    Returns (t0_d, Zbar, froze, partition[nelem,NSTAGE])."""
    na0 = na_C[s]
    f = nfrac[:, s]
    aek = alpha_es[:, :, s]          # [nelem, NSTAGE-1]
    # per-element t_0 from the neutral-producing (lowest) transition criterion
    # use element-representative alpha = recomb into its lowest tracked stage
    a_rep = np.array([aek[e, 0] if aek[e, 0] > 0 else 2.6e-13 * (T_C[s]/1e4)**-0.8
                      for e in range(nelem)])
    # single shell t_0 (overall): use abundance-weighted alpha
    a_bar = np.sum(f * a_rep)
    t0 = np.sqrt(a_bar * ne_texp * TEXP ** 3)
    t0_d = t0 / DAY
    if t0 >= TEXP:
        return t0_d, np.nan, False, None

    # seed: equilibrium at t_0 ~ singly ionized (per multistage lesson)
    y0 = np.zeros((nelem, NSTAGE))
    for e in range(nelem):
        ks = min(1, topstage[e])
        y0[e, ks] = 1.0
    y0 = y0.ravel()

    def rhs(t, yflat):
        y = yflat.reshape(nelem, NSTAGE)
        na_t = na0 * (TEXP / t) ** 3
        Zbar_e = (y * np.arange(NSTAGE)[None, :]).sum(axis=1)
        ne = na_t * (f * Zbar_e).sum()
        dy = np.zeros_like(y)
        for k in range(NSTAGE):
            inflow = aek[:, k] * ne * y[:, k + 1] if k + 1 < NSTAGE else 0.0
            outflow = aek[:, k - 1] * ne * y[:, k] if k - 1 >= 0 else 0.0
            dy[:, k] = inflow - outflow
        return dy.ravel()

    sol = solve_ivp(rhs, [t0, TEXP], y0, method="LSODA", rtol=1e-7, atol=1e-10)
    y = np.clip(sol.y[:, -1].reshape(nelem, NSTAGE), 0, 1)
    y /= y.sum(axis=1, keepdims=True)
    Zbar_e = (y * np.arange(NSTAGE)[None, :]).sum(axis=1)
    Zbar = (f * Zbar_e).sum()
    return t0_d, Zbar, True, y


# self-consistent n_e(t_exp): iterate
ne_sc = ne_C.copy()
t0s = np.zeros(nsh); Zf = np.full(nsh, np.nan); froze = np.zeros(nsh, bool)
parts = [None] * nsh
for it in range(12):
    new_ne = ne_sc.copy()
    for s in range(nsh):
        t0s[s], Zf[s], froze[s], parts[s] = solve_shell(s, ne_sc[s])
        if froze[s]:
            new_ne[s] = Zf[s] * na_C[s]
    drel = np.nanmax(np.abs(new_ne[froze] - ne_sc[froze]) / (ne_sc[froze] + 1))
    ne_sc = 0.5 * ne_sc + 0.5 * new_ne
    if drel < 1e-3:
        break

om = np.nanmean(Zf[froze])
print(f"\n=== FAITHFUL Milne-alpha multi-stage (self-consistent n_e, {it+1} iters) ===")
print(f"outer-frozen mean <Z> = {om:.3f}   (CMFGEN ~0.53; hydrogenic gave 0.553)")
print("\nsh  v       <Z>_C  <Z>_ODE t0[d]  O / Si / Fe dominant ion (faithful split)")
for i in range(0, nsh, 3):
    if not froze[i]:
        print(f"sh{i:>2} v={vc[i]:>6.0f}  <Z>_C={Zbar_C[i]:.3f}  STEADY")
        continue
    y = parts[i]
    bits = []
    for z in (8, 14, 26):
        e = list(Zlist).index(z)
        kdom = int(np.argmax(y[e]))
        zb = (y[e] * np.arange(NSTAGE)).sum()
        bits.append(f"{EL[z]}{ROM[kdom]}={y[e,kdom]:.2f}(Z{zb:.2f})")
    print(f"sh{i:>2} v={vc[i]:>6.0f}  <Z>_C={Zbar_C[i]:.3f}  <Z>={Zf[i]:.3f}"
          f"  t0={t0s[i]:.3f}  " + "  ".join(bits))

# figure
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sh = np.arange(nsh)
ax[0].plot(sh, Zbar_C, "k", lw=2.8, label="CMFGEN gold <Z>", zorder=5)
ax[0].plot(sh, Zf, "o-", color="#D97757", ms=4, lw=1.6, label="faithful Milne-α frozen <Z>")
ss = ~froze
if ss.any():
    ax[0].plot(sh[ss], Zbar_C[ss], "s", color="#4EC9B0", ms=7, mfc="none",
               label="t0>=t_exp: steady-state (NLTE)")
ax[0].set_xlabel("shell (0=innermost)"); ax[0].set_ylabel("<Z>=n_e/n_atom")
ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8, loc="center left"); ax[0].grid(alpha=0.3)
ax[0].set_title("Faithful Milne-α frozen-in <Z> vs CMFGEN (self-consistent n_e)")

# per-element <Z> to show the SPLIT
for z, col in [(8,"#3898EC"),(14,"#FFC107"),(26,"#D97757"),(20,"#4EC9B0")]:
    e = list(Zlist).index(z)
    zb = np.array([(parts[i][e]*np.arange(NSTAGE)).sum() if froze[i] else np.nan
                   for i in range(nsh)])
    ax[1].plot(sh, zb, "-", color=col, lw=1.6, label=f"{EL[z]} <Z>")
ax[1].plot(sh, Zf, "k--", lw=1.0, label="total <Z>")
ax[1].set_xlabel("shell"); ax[1].set_ylabel("per-element <Z> (frozen)")
ax[1].set_ylim(0, 1.6); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].set_title("Per-element ionization SPLIT (faithful α resolves it;\nhydrogenic gave all-uniform)")
fig.suptitle("FAITHFUL frozen-in: Milne RR α from CMFGEN σ_bf — DDC15 0.976d (pre-implementation)")
fig.tight_layout()
out = "figures/2026-06-05_ddc15_frozen_in_milne.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("\nwrote", out)
