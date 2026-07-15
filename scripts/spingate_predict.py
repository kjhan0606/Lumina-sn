#!/usr/bin/env python3
"""[ALPHA-SPINGATE] PRE-REGISTRATION: gated/full Milne recombination ratios.

Predicts what the C gate (LUMINA_ALPHA_SPINGATE=1, frozenin_alpha_rr) will do,
using the SAME inputs the C code uses: the baked bin sigma_bf and the companion
multiplicity table (level_multiplicity.csv). For each target ion it restricts
the Milne recombination sum to spin-allowed daughter terms M in {M_core-1,
M_core+1} and reports alpha_gated/alpha_full and the fraction of alpha lost.

M_core = ground-term multiplicity of the RECOMBINING ion (Z, ion+1):
  - from the companion table if that ion's ground level is present (Si/S/Ca III);
  - else from the hardcoded NIST table (Fe/Co/Ni IV are not in CMFGEN config).

This is the number to hold the future gated run to.  For Fe III it must match
the one-time C banner "[ALPHA-SPINGATE] Fe III: alpha_gated/alpha_full = ...".

Reuses the bin machinery in scripts/fe3_alpha_inflation_audit.py.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np

import fe3_alpha_inflation_audit as A   # bin SIG, levZ/levI/levN/levG, milne_bin, CHI

ROOT = A.ROOT
REF = A.REF
MULT_CSV = REF / "level_multiplicity.csv"

# ---- companion multiplicity, keyed by (Z, ion, level_num) -> mult ----
MULT = {}
for r in csv.DictReader(open(MULT_CSV)):
    MULT[(int(r["atomic_number"]), int(r["ion_number"]),
          int(r["level_number"]))] = int(r["multiplicity"])

# per-global-bin-level multiplicity array (aligned with the bin / levels.csv)
LEVMULT = np.zeros(A.NLEV, dtype=np.int32)
for i in range(A.NLEV):
    LEVMULT[i] = MULT.get((int(A.levZ[i]), int(A.levI[i]), int(A.levN[i])), 0)


# ---- NIST fallback table (mirror of spingate_core_mult in lumina_plasma.c) ----
CORE_TABLE = {
    (14, 0): 3, (14, 1): 2, (14, 2): 1, (14, 3): 2,
    (16, 0): 3, (16, 1): 4, (16, 2): 3, (16, 3): 2,
    (20, 0): 1, (20, 1): 2, (20, 2): 1, (20, 3): 2,
    (26, 0): 5, (26, 1): 6, (26, 2): 5, (26, 3): 6, (26, 4): 5, (26, 5): 4,
    (27, 0): 4, (27, 1): 3, (27, 2): 4, (27, 3): 5, (27, 4): 6,
    (28, 0): 3, (28, 1): 2, (28, 2): 3, (28, 3): 4, (28, 4): 5,
}


def core_mult(Z, ion):
    """M_core = ground multiplicity of the recombining ion (Z, ion+1)."""
    m = MULT.get((Z, ion + 1, 0), 0)          # companion ground of recombining ion
    if m > 0:
        return m, "data"
    m = CORE_TABLE.get((Z, ion + 1), 0)       # NIST fallback
    return m, ("table" if m > 0 else "none")


def milne_gated(Z, ion, T, M_core):
    """alpha_full, alpha_gated, n_skip, n_lev over the bin (mirrors the C loop)."""
    chi0 = A.CHI[(Z, ion)]; kT = A.KB * T
    U_ion, _ = A.U_upper(Z, ion, T)
    lam3 = (A.H * A.H / (2 * A.PI * A.ME * A.KB * T)) ** 1.5
    idx = np.where((A.levZ == Z) & (A.levI == ion))[0]
    a_full = a_gate = 0.0; n_skip = n_lev = 0
    for gl in idx:
        if not A.flags[gl]:
            continue
        chi_l = chi0 * A.EV - A.levE[gl] * A.EV
        if chi_l <= 0.0:
            continue
        nu_th = chi_l / A.H
        sig = np.asarray(A.SIG[gl])
        x = A.H * A.nu_c / kT
        m = (A.nu_c >= nu_th) & (sig > 0.0) & (x < 700.0)
        if not m.any():
            continue
        B = 2 * A.H * A.nu_c[m] ** 3 / A.C ** 2 / np.expm1(x[m])
        Rbf = float(np.sum(4 * A.PI * B * sig[m] / (A.H * A.nu_c[m]) * A.dnu[m]))
        a_l = Rbf * lam3 * A.levG[gl] / (2 * U_ion) * math.exp(chi_l / kT)
        Ml = int(LEVMULT[gl])
        spin_skip = (M_core > 0 and Ml > 0 and
                     Ml != M_core - 1 and Ml != M_core + 1)
        a_full += a_l; n_lev += 1
        if spin_skip:
            n_skip += 1
        else:
            a_gate += a_l
    return a_full, a_gate, n_skip, n_lev


# DR(27,3) Co IV -> Co III (DR_TABLE entry, lumina_plasma.c:4130; boost=1)
DR_CO3_C = [2.7336e-06, 9.9735e-06, 1.2109e-05]
DR_CO3_E = [6.1307e+02, 4.3869e+03, 9.7077e+03]


def dr_co3(T):
    s = sum(c * math.exp(-E / T) for c, E in zip(DR_CO3_C, DR_CO3_E)
            if -E / T >= -700.0)
    return s * T ** -1.5


TARGETS = [
    (26, 2, "Fe III"), (27, 2, "Co III"), (28, 2, "Ni III"),
    (14, 1, "Si II"),  (16, 1, "S II"),   (20, 1, "Ca II"),
]
T0 = 13120


def main():
    print(f"bin NLEV={A.NLEV}  companion levels with known M: "
          f"{int((LEVMULT>0).sum())}/{A.NLEV}")
    print("\n" + "=" * 96)
    print(f"[ALPHA-SPINGATE] PRE-REGISTRATION  (T = {T0} K)")
    print("=" * 96)
    print(f"{'ion':7} {'M_core':>6} {'src':>5} {'allowed':>9} "
          f"{'alpha_full':>12} {'alpha_gated':>12} {'gated/full':>10} "
          f"{'lost%':>7} {'skip/lev':>10}")
    results = {}
    for Z, ion, name in TARGETS:
        Mc, src = core_mult(Z, ion)
        allowed = sorted({Mc - 1, Mc + 1} & set(range(1, 12))) if Mc > 0 else []
        a_full, a_gate, nsk, nlev = milne_gated(Z, ion, T0, Mc)
        ratio = a_gate / a_full if a_full > 0 else float("nan")
        lost = 100.0 * (1 - ratio) if a_full > 0 else float("nan")
        results[name] = (Mc, src, a_full, a_gate, ratio)
        print(f"{name:7} {Mc:6d} {src:>5} {str(allowed):>9} "
              f"{a_full:12.4e} {a_gate:12.4e} {ratio:10.3f} "
              f"{lost:7.1f} {nsk:4d}/{nlev:<4d}")

    # temperature dependence of the Fe III ratio (bracket the diagnostic banner)
    print("\n  Fe III gated/full vs T (the C banner reports the run's T_e):")
    Mc, _ = core_mult(26, 2)
    for T in A.T_LIST:
        a_full, a_gate, nsk, nlev = milne_gated(26, 2, T, Mc)
        print(f"    T={T:6d} K : gated/full = {a_gate/a_full:.3f} "
              f"(skipped {nsk}/{nlev})")

    # Co III: gated-RR + DIE-DR  vs  current full-RR
    print("\n" + "=" * 96)
    print("Co III combined channel: current full-RR  vs  gated-RR + DR(27,3)")
    print("=" * 96)
    Mc, src = core_mult(27, 2)
    a_full, a_gate, _, _ = milne_gated(27, 2, T0, Mc)
    dr = dr_co3(T0)
    print(f"  M_core(Co IV) = {Mc} [{src}]   (allowed daughter M in {{4,6}})")
    print(f"  full-RR                : {a_full:.4e}")
    print(f"  gated-RR               : {a_gate:.4e}  ({a_gate/a_full:.3f} x full)")
    print(f"  DR(27,3) @ {T0}K        : {dr:.4e}")
    print(f"  gated-RR + DR          : {a_gate+dr:.4e}")
    print(f"  NET (gatedRR+DR)/fullRR: {(a_gate+dr)/a_full:.3f}   "
          f"(DR replaces {100.0*dr/(a_full-a_gate):.0f}% of the removed forbidden RR)")


if __name__ == "__main__":
    main()
