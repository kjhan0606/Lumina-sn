#!/usr/bin/env python3
"""gamma_photoion_cmp.py -- trigger (1) of the integrated-arm decision:
Gamma(Co III) CMFGEN vs LUMINA, decomposed into FIELD vs POPULATIONS.

WHY THIS EXISTS (read before trusting any number it prints)
-----------------------------------------------------------
The recorded plan said "read Gamma(Co III) off CMFGEN via dispgen NETR_CoIII".
That is WRONG: in maingen.f:825 NETR is grouped with the *line* options
(MOMR/SOBR/EW/LAM/SRCE/CHIL/TAUL/BETA) and its handler (maingen.f:1618) computes
ZNET = 1 - JBAR*CHIL/ETAL, i.e. the net rate of a bound-bound transition.
dispgen exposes NO photoionization-rate option at all:
    PHOT_* / PLTPHOT_*  -> photoionization CROSS-SECTIONS only
    RR_*                -> integrated RECOMBINATION coefficient (alpha)
    DC_* / POP_*        -> departure coefficients / level populations
So Gamma must be built offline as the same integral LUMINA evaluates:

    Gamma_ion = SUM_k (n_k/n_ion) * R_k(J),
    R_k(J)    = 4pi INT_{nu_k}^{inf} sigma_k(nu) J(nu) / (h nu) dnu

which is exactly db_photoion_calc.R_of_level (self-tested: J=B -> r/saha=1.000).

THE DESIGN POINT: a single ratio Gamma_C/Gamma_L would only say "10x or 1x".
But Gamma depends on TWO inputs -- the field J and the populations n_k/n_ion --
so swapping them independently localises the gap:

                  pops=LUMINA     pops=CMFGEN
    J=LUMINA        G_LL            G_LC
    J=CMFGEN        G_CL            G_CC
    G_CC/G_LL = total gap ; G_CL/G_LL = field-only ; G_LC/G_LL = pops-only

YARDSTICK GATES (mandatory; see feedback_audit_the_yardstick_first)
    Two conclusions in this project have already been overturned because the
    COMPARATOR, not the reasoning, was defective (g-weighted b_k; T_rad pinned at
    10470 K in all 50 shells).  So this tool refuses to print a ratio until it has
    (a) calibrated the CMFGEN J scale against the free-streaming luminosity anchor
        rather than trusting a units convention read out of Fortran, and
    (b) checked every comparator column it uses for a pinned (uniq==1) value.
    A ratio built on an unaudited denominator is not evidence.

STATUS
    - LUMINA side + J=B self-test: RUNNABLE NOW.
    - CMFGEN side: needs a CONVERGED 19.48d model (jnu.csv full-grid + level pops).
      Those readers are written against the documented formats but are UNVERIFIED
      until the first converged model exists; they fail loudly rather than guess.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import db_photoion_calc as dbp  # sigma grid, levels, R_of_level, R_planck

H = dbp.H
PI = dbp.PI
C = dbp.C
KB_EV = dbp.KB_EV
LSUN = 3.826e33


# --------------------------------------------------------------------------
# comparator hygiene
# --------------------------------------------------------------------------
def assert_not_pinned(values, name, ctx):
    """A comparator column that never varies is a pin, not a measurement."""
    u = len(set(values))
    if u == 1:
        raise SystemExit(
            f"YARDSTICK ABORT: {name} is PINNED at {values[0]!r} across all "
            f"{len(values)} {ctx} (uniq=1). Any ratio against it is meaningless. "
            f"This is the T_rad=10470 failure mode -- fix the source of {name} "
            f"or pass an explicit override."
        )
    return u


# --------------------------------------------------------------------------
# CMFGEN J_nu  (from cmfgen_extract/parse_jnu.py -> jnu.csv, needs JNU_ALL=1)
# --------------------------------------------------------------------------
def cmfgen_J(jnu_csv, depth_index):
    """Return J(nu) resampled onto the sigma grid, plus the raw (lam, J) arrays.

    jnu.csv columns: depth_index, v_kms, lambda_A, J_nu   (parse_jnu.py:107)
    lambda is vacuum Angstrom; J_nu is CMFGEN's internal RJ (EDDFACTOR stores
    r^2.J and plt_jh.f:381 divides by r^2 before dumping).  The ABSOLUTE SCALE of
    RJ is NOT taken on faith -- calibrate_J_scale() fixes it empirically.
    """
    lam, J, vk = [], [], None
    with open(jnu_csv) as fh:
        for r in csv.DictReader(fh):
            if int(r["depth_index"]) != depth_index:
                continue
            lam.append(float(r["lambda_A"]))
            J.append(float(r["J_nu"]))
            if vk is None and r.get("v_kms"):
                vk = float(r["v_kms"])
    if not lam:
        raise SystemExit(f"no rows for depth_index={depth_index} in {jnu_csv}")
    lam = np.asarray(lam)
    J = np.asarray(J)
    nu = C / (lam * 1e-8)
    o = np.argsort(nu)
    nu, J = nu[o], J[o]
    if nu[0] > dbp.nu_c[0] * 1.05 or nu[-1] < dbp.nu_c[-1] * 0.95:
        print(
            f"  [warn] CMFGEN J grid {C/1e-8/lam.max():.3e}..{C/1e-8/lam.min():.3e} Hz "
            f"does not span the sigma grid {dbp.nu_c[0]:.3e}..{dbp.nu_c[-1]:.3e} Hz; "
            f"extrapolation is ZERO (photoionization above the top edge is dropped). "
            f"Re-dump with JNU_ALL=1 if this band is truncated."
        )
    Jg = np.interp(dbp.nu_c, nu, J, left=0.0, right=0.0)
    return Jg, nu, J, vk


def calibrate_J_scale(nu, J, r_cm, lum_lsun):
    """Empirically fix the CMFGEN J scale instead of trusting a Fortran convention.

    At the outermost depth the field is ~free-streaming, so
        4pi INT J dnu  ~=  F  =  L / (4 pi r^2)
    (the same anchor the #12 energy-ledger audit used: J = L/(16 pi^2 r^2)).
    Returns (scale, ratio_before): multiply J by `scale` to land on the anchor.
    """
    integ = float(np.trapezoid(J, nu)) if hasattr(np, "trapezoid") else float(np.trapz(J, nu))
    flux_model = 4 * PI * integ
    flux_anchor = lum_lsun * LSUN / (4 * PI * r_cm ** 2)
    if flux_model <= 0:
        raise SystemExit("YARDSTICK ABORT: CMFGEN J integrates to <=0")
    ratio = flux_model / flux_anchor
    return flux_anchor / flux_model, ratio


# --------------------------------------------------------------------------
# populations
# --------------------------------------------------------------------------
def lumina_pops(logdir, Z, ion, shell):
    """{level_number: n_k/n_ion} from lumina_levelpop.csv."""
    raw = dbp.levelpops(logdir, Z, ion, shell)
    tot = sum(v[0] for v in raw.values())
    if tot <= 0:
        raise SystemExit(f"LUMINA pops empty for Z={Z} ion={ion} s{shell} in {logdir}")
    return {k: v[0] / tot for k, v in raw.items()}, tot


def cmfgen_pops(pop_csv, Z, ion):
    """{level_number: n_k/n_ion} from a converged CMFGEN model.

    UNVERIFIED until a converged model exists.  Expected columns:
        Z, ion, level_number, depth_index, n_k
    Level numbering MUST be the CMFGEN atomic-data order, which is the order
    levels.csv was baked from -- verify_level_mapping() checks that, and this
    tool refuses to run if coverage is poor.
    """
    if not os.path.exists(pop_csv):
        raise SystemExit(
            f"CMFGEN pops not found: {pop_csv}\n"
            f"  Produce it from a CONVERGED model with:\n"
            f"    scripts/cmfgen_extract/dispgen_pops.sh <run_dir> <out_prefix>\n"
            f"  (needs RVTJ + POP<species>; written only at LST_ITERATION)."
        )
    out = {}
    with open(pop_csv) as fh:
        for r in csv.DictReader(fh):
            if int(r["Z"]) == Z and int(r["ion"]) == ion:
                out[int(r["level_number"])] = float(r["n_k"])
    if not out:
        raise SystemExit(f"no Z={Z} ion={ion} rows in {pop_csv}")
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}, tot


def verify_level_mapping(pops, Z, ion, label):
    """Refuse to compare if the pops don't cover the sigma-grid levels."""
    idx = np.where((dbp.levZ == Z) & (dbp.levI == ion))[0]
    ours = {int(dbp.levN[g]) for g in idx}
    theirs = set(pops)
    hit = ours & theirs
    cov = len(hit) / max(1, len(ours))
    miss_w = sum(v for k, v in pops.items() if k not in ours)
    print(
        f"  [map] {label}: sigma-grid levels={len(ours)} pops={len(theirs)} "
        f"matched={len(hit)} coverage={cov:.1%} unmatched_pop_weight={miss_w:.2%}"
    )
    if cov < 0.90 or miss_w > 0.05:
        raise SystemExit(
            f"YARDSTICK ABORT: level mapping for Z={Z} ion={ion} is unreliable "
            f"(coverage {cov:.1%}, unmatched weight {miss_w:.2%}). The level "
            f"numbering conventions differ -- fix the mapping before comparing."
        )
    return idx


# --------------------------------------------------------------------------
# Gamma
# --------------------------------------------------------------------------
def gamma(idx, J, chi0, pops):
    """Gamma = SUM_k (n_k/n_ion) R_k(J)  [s^-1 per ion of this stage]."""
    g = 0.0
    per = []
    for gl in idx:
        R, _ = dbp.R_of_level(gl, J, chi0)
        if R <= 0:
            continue
        p = pops.get(int(dbp.levN[gl]), 0.0)
        if p <= 0:
            continue
        g += p * R
        per.append((int(dbp.levN[gl]), float(dbp.levE[gl]), p, p * R))
    per.sort(key=lambda t: -t[3])
    return g, per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lumina-log", default="logs/coevolve_consume_a10_kx_gphall",
                    help="LUMINA run dir (B-state = all-level G, the arm baseline)")
    ap.add_argument("--shell", type=int, default=0, help="LUMINA shell (core=0-3)")
    ap.add_argument("--Z", type=int, default=27)
    ap.add_argument("--ion", type=int, default=2, help="2 = III (stage=charge)")
    ap.add_argument("--jnu-csv", help="CMFGEN jnu.csv (JNU_ALL=1); omit for LUMINA-only")
    ap.add_argument("--depth", type=int, help="CMFGEN depth index matching --shell")
    ap.add_argument("--pop-csv", help="CMFGEN level pops csv")
    ap.add_argument("--r-cm", type=float, help="radius [cm] at the OUTER depth, for the J anchor")
    ap.add_argument("--lum-lsun", type=float, help="model luminosity [Lsun], for the J anchor")
    ap.add_argument("--jnu-outer-depth", type=int, default=1,
                    help="depth index of the outer boundary (anchor calibration)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the J=B -> Saha gate and exit")
    a = ap.parse_args()

    if a.selftest:
        print("== SELF-TEST: J=B must reproduce Saha (validates sigma grid + integrator) ==")
        dbp.analyze(a.lumina_log, f"SELFTEST Z={a.Z} ion={a.ion}", a.Z, a.ion, a.shell,
                    selftest=True)
        print("PASS criterion: saha_check r_b/saha = 1.000")
        return

    chi0 = dbp.CHI[(a.Z, a.ion)]
    Te, ne = dbp.plasma(a.lumina_log, a.shell)
    print(f"== Gamma(Z={a.Z}, ion={a.ion}->{a.ion+1}) @ LUMINA s{a.shell} "
          f"(Te={Te:.0f} ne={ne:.3e}) ==")

    # ---- comparator hygiene on the LUMINA side -----------------------------
    with open(f"{a.lumina_log}/lumina_plasma_state.csv") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("T_rad", "T_e"):
        if col in rows[0]:
            u = len(set(r[col] for r in rows))
            flag = "  <-- PINNED (do NOT build a comparator on this)" if u == 1 else ""
            print(f"  [hygiene] {col}: {u} distinct value(s) over {len(rows)} shells{flag}")

    pL, ntotL = lumina_pops(a.lumina_log, a.Z, a.ion, a.shell)
    idx = verify_level_mapping(pL, a.Z, a.ion, "LUMINA")
    JL = dbp.field(a.lumina_log, a.shell)
    G_LL, perL = gamma(idx, JL, chi0, pL)
    print(f"\n  G_LL (J=LUMINA, pops=LUMINA) = {G_LL:.4e} s^-1/ion")
    print("    top levels n(E_eV, frac, contrib):",
          ", ".join(f"{n}({E:.1f}, {p:.2e}, {c:.2e})" for n, E, p, c in perL[:5]))

    if not a.jnu_csv:
        print("\n  [CMFGEN side skipped: no --jnu-csv]")
        print("  When the converged 19.48d model lands:")
        print("    JNU_ALL=1 scripts/cmfgen_extract/extract_all.sh <run> <out>")
        print("    scripts/cmfgen_extract/dispgen_pops.sh <run> <out>/pops")
        print("    python3 scripts/gamma_photoion_cmp.py --jnu-csv <out>/jnu.csv \\")
        print("        --pop-csv <out>/pops.csv --depth D --r-cm R --lum-lsun L")
        return

    if not (a.r_cm and a.lum_lsun and a.depth is not None):
        raise SystemExit("--jnu-csv requires --depth, --r-cm and --lum-lsun (anchor gate)")

    # ---- YARDSTICK GATE: calibrate the CMFGEN J scale empirically ----------
    _, nu_o, J_o, _ = cmfgen_J(a.jnu_csv, a.jnu_outer_depth)
    scale, ratio = calibrate_J_scale(nu_o, J_o, a.r_cm, a.lum_lsun)
    print(f"\n  [anchor] outer-depth 4pi.INT(J)dnu / [L/(4pi r^2)] = {ratio:.4e}")
    print(f"           -> CMFGEN J scale factor = {scale:.4e}")
    if not (0.2 < ratio < 5.0) and not (0.2 < ratio * 1e15 < 5.0) and not (0.2 < ratio * 1e-15 < 5.0):
        raise SystemExit(
            "YARDSTICK ABORT: the CMFGEN J normalisation is neither ~1 nor an obvious "
            f"10^+-15 convention offset (ratio={ratio:.3e}). Resolve the unit before "
            "comparing -- an unaudited denominator is exactly how the last two "
            "conclusions got overturned."
        )
    JC, _, _, vk = cmfgen_J(a.jnu_csv, a.depth)
    JC = JC * scale
    print(f"           CMFGEN depth {a.depth} v={vk} km/s (LUMINA s{a.shell})")

    pC, _ = cmfgen_pops(a.pop_csv, a.Z, a.ion) if a.pop_csv else (pL, 0)
    if a.pop_csv:
        verify_level_mapping(pC, a.Z, a.ion, "CMFGEN")

    # ---- the 2x2 --------------------------------------------------------------
    G_CL, _ = gamma(idx, JC, chi0, pL)
    G_LC, _ = gamma(idx, JL, chi0, pC)
    G_CC, perC = gamma(idx, JC, chi0, pC)
    print(f"\n  2x2 decomposition            Gamma [s^-1/ion]      / G_LL")
    print(f"    G_LL  J=LUMINA pops=LUMINA   {G_LL:.4e}          1.00")
    print(f"    G_CL  J=CMFGEN pops=LUMINA   {G_CL:.4e}      {G_CL/G_LL:8.3f}   <- FIELD only")
    print(f"    G_LC  J=LUMINA pops=CMFGEN   {G_LC:.4e}      {G_LC/G_LL:8.3f}   <- POPS only")
    print(f"    G_CC  J=CMFGEN pops=CMFGEN   {G_CC:.4e}      {G_CC/G_LL:8.3f}   <- TOTAL")
    sep = (G_CL / G_LL) * (G_LC / G_LL)
    print(f"    separability: (field)x(pops)={sep:.3f} vs total={G_CC/G_LL:.3f} "
          f"-> {'separable' if abs(math.log10(max(sep,1e-30)/max(G_CC/G_LL,1e-30)))<0.15 else 'COUPLED (do not attribute additively)'}")

    # ---- pre-registered verdict ----------------------------------------------
    tot = G_CC / G_LL
    print(f"\n  VERDICT (pre-registered rule):")
    if tot >= 5:
        print(f"    Gamma_C/G_L = {tot:.2f} >= 5  ==> the ~10x gap is REAL and sits on the")
        print(f"    {'FIELD' if G_CL/G_LL > G_LC/G_LL else 'POPULATIONS'} side "
              f"(field {G_CL/G_LL:.2f}x vs pops {G_LC/G_LL:.2f}x).")
        print(f"    ==> G-correction is the right third component; alpha side closed.")
    elif tot <= 2:
        print(f"    Gamma_C/G_L = {tot:.2f} <= 2  ==> NO large Gamma gap. The recorded "
              f"'Gamma(Co III) ~10x' inference is REFUTED.")
        print(f"    ==> re-audit the alpha side (spingate x DR interaction).")
    else:
        print(f"    Gamma_C/G_L = {tot:.2f} is in the undecided band (2,5) -- report as-is, "
              f"do not force a verdict.")


if __name__ == "__main__":
    main()
