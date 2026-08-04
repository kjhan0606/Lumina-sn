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


def cmfgen_pops(pop_csv, Z, ion, depth=None):
    """{level_number: n_k/n_ion} from a converged CMFGEN model, at one depth.

    UNVERIFIED until a converged model exists.  Columns (parse_pops.py output):
        Z, ion, level_number, depth_index, n_k    with depth_index = 1..ND.
    A POP<species> file carries ALL depths, so a depth MUST be selected -- the
    comparison is at the single depth matching --shell.  If depth is None (pops
    plumbing / fixture) the smallest available depth is used and announced.
    Level numbering is the CMFGEN atomic-data order (= levels.csv order, verified
    by verify_index_mapping); level_number is 0-based to match the sigma grid.
    """
    if not os.path.exists(pop_csv):
        raise SystemExit(
            f"CMFGEN pops not found: {pop_csv}\n"
            f"  Produce it from a CONVERGED model with:\n"
            f"    scripts/cmfgen_extract/dispgen_pops.sh <run_dir> <out_prefix>\n"
            f"  (needs RVTJ + POP<species>; written only at LST_ITERATION)."
        )
    by_depth = {}
    for r in csv.DictReader(open(pop_csv)):
        if int(r["Z"]) == Z and int(r["ion"]) == ion:
            d = int(r["depth_index"])
            by_depth.setdefault(d, {})[int(r["level_number"])] = float(r["n_k"])
    if not by_depth:
        raise SystemExit(f"no Z={Z} ion={ion} rows in {pop_csv}")
    if depth is None:
        depth = min(by_depth)
        if len(by_depth) > 1:
            print(f"  [cmfgen_pops] no --depth given: using depth_index={depth} of "
                  f"{sorted(by_depth)[0]}..{sorted(by_depth)[-1]} (plumbing only).")
    if depth not in by_depth:
        raise SystemExit(f"depth_index={depth} absent in {pop_csv} "
                         f"(have {sorted(by_depth)[0]}..{sorted(by_depth)[-1]})")
    out = by_depth[depth]
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}, tot


def verify_level_mapping(pops, Z, ion, label, fatal=True):
    """Check whether pops cover the sigma-grid levels.

    strict mode (fatal=True): refuse to compare below 90% coverage -- correct
    default, because a comparison over mismatched level sets is meaningless.
    intersection mode (fatal=False): report coverage but return so the caller
    can restrict the comparison to the common levels (see build_common_levels).
    """
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
    if (cov < 0.90 or miss_w > 0.05):
        if fatal:
            raise SystemExit(
                f"YARDSTICK ABORT: level mapping for Z={Z} ion={ion} is unreliable "
                f"(coverage {cov:.1%}, unmatched weight {miss_w:.2%}). The level "
                f"numbering conventions differ -- fix the mapping before comparing.\n"
                f"  If this is the EXPECTED CMFGEN-vs-LUMINA atom-size mismatch "
                f"(CMFGEN models a 1000-level subset of LUMINA's larger sigma grid),\n"
                f"  re-run with --level-mode intersection to compare over the common "
                f"levels with explicit coverage accounting."
            )
        print(f"       [intersection] coverage below strict gate -- comparing over "
              f"the {len(hit)} common levels only (accounting printed below).")
    return idx


# --------------------------------------------------------------------------
# CMFGEN oscillator-file level list  (verified index mapping, task 1)
# --------------------------------------------------------------------------
CM2EV = 1.239841984e-4  # same factor carsus used to bake levels.csv


def parse_cmfgen_osc_levels(osc_path):
    """Parse a CMFGEN F_OSCDAT level list -> [(name, g, E_eV), ...] in file order.

    File order == CMFGEN internal level order == the order POP<SPECIES> and the
    carsus levels.csv are written in (verified: 100% (g,E) match, no index shift,
    for CoIII 18oct00 coiii_osc.dat vs levels.csv rows 0..999).  Level line:
        name  g  E(cm^-1)  freq[1e15Hz]  eV_thresh  Lam(A)  ID  ARAD  GAM2  GAM4
    with |ID| the 1-based sequential level index.
    """
    import re
    lines = open(osc_path).read().split("\n")
    nlev = None
    hdr_end = None
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(\d+)\s+!Number of energy levels", ln)
        if m:
            nlev = int(m.group(1)); hdr_end = i; break
    if nlev is None:
        raise SystemExit(f"{osc_path}: no 'Number of energy levels' header line")
    lev = []
    numre = re.compile(r"^[-+]?\d*\.?\d+$")
    for ln in lines[hdr_end + 1:]:
        s = ln.split()
        if len(s) < 7:
            continue
        try:
            g = float(s[1]); Ecm = float(s[2]); int(float(s[6]))
        except ValueError:
            continue
        if numre.match(s[0]):        # name must not be a bare number
            continue
        lev.append((s[0], g, Ecm * CM2EV))
        if len(lev) >= nlev:
            break
    if len(lev) != nlev:
        raise SystemExit(f"{osc_path}: parsed {len(lev)} levels, header says {nlev}")
    return lev


def verify_index_mapping(osc_levels, Z, ion, tol_abs=0.01, tol_rel=1e-3):
    """Confirm sigma-grid levels 0..N-1 ARE the CMFGEN osc levels, same order.

    Returns (clean, nmatch, N, common_level_nums, first_mismatch).  clean=True
    means the identity mapping (levels.csv level_number k == CMFGEN level k) is
    valid, so POP<SPECIES> level_numbers can be trusted without re-derivation.
    """
    idx = np.where((dbp.levZ == Z) & (dbp.levI == ion))[0]
    grid_n = [int(dbp.levN[g]) for g in idx]
    grid_g = {int(dbp.levN[g]): float(dbp.levG[g]) for g in idx}
    grid_E = {int(dbp.levN[g]): float(dbp.levE[g]) for g in idx}
    N = min(len(osc_levels), len(idx))
    nmatch = 0
    first_mismatch = None
    common = []
    for k in range(N):
        _, g_osc, E_osc = osc_levels[k]
        if k not in grid_g:
            if first_mismatch is None:
                first_mismatch = (k, "sigma-grid has no level_number %d" % k)
            continue
        dg = abs(g_osc - grid_g[k])
        dE = abs(E_osc - grid_E[k])
        tolE = max(tol_abs, tol_rel * max(abs(E_osc), abs(grid_E[k])))
        if dg < 0.5 and dE <= tolE:
            nmatch += 1
            common.append(k)
        elif first_mismatch is None:
            first_mismatch = (k, f"g {g_osc} vs {grid_g[k]} | E {E_osc:.5f} vs "
                                 f"{grid_E[k]:.5f} (dE={dE:.2e}, tol={tolE:.2e})")
    clean = (nmatch == N) and first_mismatch is None
    print(f"  [index-map] CMFGEN osc vs sigma-grid Z={Z} ion={ion}: "
          f"{nmatch}/{N} match (g exact, E<=max({tol_abs}eV,{tol_rel}rel)) "
          f"-> {'CLEAN identity mapping' if clean else 'MISMATCH'}")
    if first_mismatch:
        print(f"              first mismatch @ index {first_mismatch[0]}: {first_mismatch[1]}")
    return clean, nmatch, N, common, first_mismatch


def build_common_levels(Z, ion, cmf_osc=None, cmf_pops=None):
    """Common level set = sigma-grid CoIII levels that CMFGEN also models.

    Preference order (task 2: use the mapping result, do not assume):
      1. --cmfgen-osc : parse the osc level list and VERIFY the index mapping.
         Common = the verified-matched sigma-grid level_numbers.
      2. --pop-csv    : trust the parse_pops level_numbers (0-based, = osc order
         by construction) intersected with the sigma grid.
    Returns (common_set, source_str, clean_or_None).
    """
    grid = {int(dbp.levN[g]) for g in np.where((dbp.levZ == Z) & (dbp.levI == ion))[0]}
    if cmf_osc is not None:
        osc = parse_cmfgen_osc_levels(cmf_osc)
        clean, nmatch, N, common, _ = verify_index_mapping(osc, Z, ion)
        return set(common), f"osc:{os.path.basename(cmf_osc)} ({len(osc)} levels)", clean
    if cmf_pops is not None:
        common = grid & set(cmf_pops)
        print(f"  [index-map] no --cmfgen-osc: trusting parse_pops level_numbers "
              f"(0-based osc order); {len(common)} common with sigma grid. "
              f"Index identity was verified separately (task 1).")
        return common, "pop-csv level_numbers", None
    raise SystemExit("intersection mode needs --cmfgen-osc or --pop-csv to define "
                     "the CMFGEN level set")


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


def restrict_idx(idx, level_nums):
    """Subset the sigma-grid idx array to the given set of level_numbers."""
    keep = {int(x) for x in level_nums}
    return np.array([gl for gl in idx if int(dbp.levN[gl]) in keep], dtype=idx.dtype)


def detect_floor(pops, min_count=10, max_rel=0.1):
    """Find the modal byte-identical n_frac cluster = the levelpop FLOOR artifact.

    The NLTE solve clamps every negatively-solved population to 1e-30
    (src/lumina_cuda.cu:1206), then rescales by n_total/sum (:1265-1274), so all
    clamped levels land at ONE identical absolute n_k -> identical n_frac.  A
    genuine NLTE solve never gives thousands of distinct levels byte-identical
    populations.  Returns (floor_level_numbers, value, count).
    """
    from collections import Counter
    vals = [v for v in pops.values() if v > 0]
    if not vals:
        return set(), None, 0
    val, cnt = Counter(vals).most_common(1)[0]
    if cnt >= min_count and val <= max_rel * max(vals):
        return {k for k, v in pops.items() if v == val}, val, cnt
    return set(), None, 0


def apply_exclude_floor(pops, label):
    """Zero the floor-pinned levels in a pops dict; report the count. Returns pops."""
    fset, fval, fcnt = detect_floor(pops)
    if not fset:
        print(f"  [exclude-floor] {label}: no byte-identical floor cluster detected.")
        return pops
    print(f"  [exclude-floor] {label}: excluded {fcnt} levels pinned at n_frac={fval:.4e} "
          f"(1e-30 clamp artifact, lumina_cuda.cu:1206). Kept {sum(1 for v in pops.values() if v>0)-fcnt} genuine levels.")
    return {k: (0.0 if k in fset else v) for k, v in pops.items()}


def report_coverage(side, idx_full, idx_common, J, chi0, pops, G_full, src):
    """Print how much of the full-set Gamma lives inside the intersection.

    This is THE reframing number: if most of LUMINA's Gamma comes from levels
    CMFGEN does not model, a full-atom ratio is not a like-for-like comparison.
    """
    G_common, per = gamma(idx_common, J, chi0, pops)
    frac = G_common / G_full if G_full > 0 else float("nan")
    print(f"\n  [coverage:{side}] common set = {len(idx_common)} of {len(idx_full)} "
          f"sigma-grid levels  (source: {src})")
    print(f"    G_{side[0]}{side[0]}_full   = {G_full:.4e} s^-1/ion   (all levels)")
    print(f"    G_{side[0]}{side[0]}_common = {G_common:.4e} s^-1/ion   "
          f"(intersection = CMFGEN-modelled levels)")
    print(f"    intersection captures {100*frac:.2f}% of the full-set Gamma; "
          f"{100*(1-frac):.2f}% lives in levels CMFGEN does NOT model.")
    # top contributors, flagged inside/outside the intersection
    common_nums = {int(dbp.levN[g]) for g in idx_common}
    G_out = G_full - G_common
    per_full = gamma(idx_full, J, chi0, pops)[1]
    print(f"    top-10 {side} contributors (level, E_eV, n_frac, contrib, in_CMFGEN?):")
    for n, E, p, c in per_full[:10]:
        tag = "yes" if n in common_nums else "NO (>=CMFGEN atom size)"
        print(f"      {n:>5}  E={E:6.2f}  n_frac={p:.3e}  contrib={c:.4e}  {tag}")


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
    ap.add_argument("--level-mode", choices=["strict", "intersection"], default="strict",
                    help="strict (default): abort if the CMFGEN pops do not cover the "
                         "sigma grid. intersection: compare over the common level set "
                         "(CMFGEN's 1000-level atom is a subset of LUMINA's) with "
                         "explicit coverage accounting.")
    ap.add_argument("--cmfgen-osc", help="CMFGEN F_OSCDAT level file (e.g. coiii_osc.dat); "
                    "defines + VERIFIES the common level set for intersection mode "
                    "even before converged pops exist.")
    ap.add_argument("--unconverged", action="store_true",
                    help="print the UNCONVERGED-MODEL banner (plumbing/fixture test only, "
                         "no physics conclusions).")
    ap.add_argument("--exclude-floor", action="store_true",
                    help="exclude the levelpop FLOOR artifact (byte-identical modal n_k "
                         "cluster from the 1e-30 clamp) from both sides before summing "
                         "Gamma -- gives the GENUINE photoion rate.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the J=B -> Saha gate and exit")
    a = ap.parse_args()

    if a.unconverged:
        print("=" * 74)
        print("  UNCONVERGED MODEL -- numbers are plumbing-test only, no physics")
        print("  conclusions.  (Level mapping + coverage plumbing verification.)")
        print("=" * 74)

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

    strict = (a.level_mode == "strict")
    if dbp.ion_is_kramers(a.Z, a.ion):
        print(f"  *** WARNING: sigma_bf for Z={a.Z} ion={a.ion} is KRAMERS FALLBACK "
              f"(nu^-3, no CMFGEN edge structure) -- this Gamma is DATA-INSENSITIVE. ***")
    pL, ntotL = lumina_pops(a.lumina_log, a.Z, a.ion, a.shell)
    if a.exclude_floor:
        pL = apply_exclude_floor(pL, "LUMINA")
    idx = verify_level_mapping(pL, a.Z, a.ion, "LUMINA", fatal=strict)
    JL = dbp.field(a.lumina_log, a.shell)
    G_LL, perL = gamma(idx, JL, chi0, pL)
    print(f"\n  G_LL_full (J=LUMINA, pops=LUMINA, all {len(idx)} sigma-grid levels) "
          f"= {G_LL:.4e} s^-1/ion")
    print("    top levels n(E_eV, frac, contrib):",
          ", ".join(f"{n}({E:.1f}, {p:.2e}, {c:.2e})" for n, E, p, c in perL[:5]))

    # ---- intersection coverage accounting -----------------------------------
    # (needs the CMFGEN level set; from --cmfgen-osc now, else from --pop-csv later)
    common, idx_use, cmf_src = None, idx, None
    if not strict:
        if a.cmfgen_osc:
            common, cmf_src, _ = build_common_levels(a.Z, a.ion, cmf_osc=a.cmfgen_osc)
        elif not a.pop_csv:
            print("  [intersection] no --cmfgen-osc and no --pop-csv: cannot define "
                  "the CMFGEN level set for coverage accounting.")
        if common is not None:
            idx_use = restrict_idx(idx, common)
            report_coverage("LUMINA", idx, idx_use, JL, chi0, pL, G_LL, cmf_src)

    if not a.jnu_csv:
        # POPS-SIDE PLUMBING: with real CMFGEN pops but no J yet (J needs a live
        # dispgen run), still exercise the pops parse + mapping + intersection
        # coverage.  This is the fixture path against a (possibly unconverged) POP file.
        if a.pop_csv:
            pC, _ = cmfgen_pops(a.pop_csv, a.Z, a.ion, a.depth)
            if a.exclude_floor:
                pC = apply_exclude_floor(pC, "CMFGEN")
            verify_level_mapping(pC, a.Z, a.ion, "CMFGEN", fatal=strict)
            if not strict and common is None:
                common, cmf_src, _ = build_common_levels(a.Z, a.ion, cmf_pops=pC)
                idx_use = restrict_idx(idx, common)
                report_coverage("LUMINA", idx, idx_use, JL, chi0, pL, G_LL, cmf_src)
            print("\n  [pops plumbing OK: parse + mapping + coverage exercised. "
                  "The 2x2 (field vs pops) needs the CMFGEN J field -- rerun with "
                  "--jnu-csv/--depth/--r-cm/--lum-lsun once a converged dispgen J exists.]")
            return
        print("\n  [CMFGEN side skipped: no --jnu-csv]")
        print("  When the converged 19.48d model lands:")
        print("    JNU_ALL=1 scripts/cmfgen_extract/extract_all.sh <run> <out>")
        print("    scripts/cmfgen_extract/dispgen_pops.sh <run> <out>/pops")
        print("    python3 scripts/gamma_photoion_cmp.py --jnu-csv <out>/jnu.csv \\")
        print("        --pop-csv <out>/pops.csv --depth D --r-cm R --lum-lsun L \\")
        print("        --level-mode intersection --cmfgen-osc <run>/CoIII_F_OSCDAT")
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

    pC, _ = cmfgen_pops(a.pop_csv, a.Z, a.ion, a.depth) if a.pop_csv else (pL, 0)
    if a.pop_csv:
        if a.exclude_floor:
            pC = apply_exclude_floor(pC, "CMFGEN")
        verify_level_mapping(pC, a.Z, a.ion, "CMFGEN", fatal=strict)
        # if intersection set wasn't defined via --cmfgen-osc, build it from pops now
        if not strict and common is None:
            common, cmf_src, _ = build_common_levels(a.Z, a.ion, cmf_pops=pC)
            idx_use = restrict_idx(idx, common)
            report_coverage("LUMINA", idx, idx_use, JL, chi0, pL, G_LL, cmf_src)

    # ---- the 2x2 (over idx_use: full grid in strict, common set in intersection)
    basis = "full sigma grid" if strict else f"{len(idx_use)} COMMON levels [{cmf_src}]"
    G_LLb, _ = gamma(idx_use, JL, chi0, pL)   # same-basis denominator for the ratios
    G_CL, _ = gamma(idx_use, JC, chi0, pL)
    G_LC, _ = gamma(idx_use, JL, chi0, pC)
    G_CC, perC = gamma(idx_use, JC, chi0, pC)
    if G_LLb <= 0:
        raise SystemExit("YARDSTICK ABORT: G_LL over the comparison basis is <=0")
    print(f"\n  2x2 decomposition [basis: {basis}]   Gamma [s^-1/ion]   / G_LL(basis)")
    print(f"    G_LL  J=LUMINA pops=LUMINA   {G_LLb:.4e}          1.00")
    print(f"    G_CL  J=CMFGEN pops=LUMINA   {G_CL:.4e}      {G_CL/G_LLb:8.3f}   <- FIELD only")
    print(f"    G_LC  J=LUMINA pops=CMFGEN   {G_LC:.4e}      {G_LC/G_LLb:8.3f}   <- POPS only")
    print(f"    G_CC  J=CMFGEN pops=CMFGEN   {G_CC:.4e}      {G_CC/G_LLb:8.3f}   <- TOTAL")
    if not strict:
        print(f"    [CMFGEN atom has NO levels outside the intersection, so 100% of the "
              f"CMFGEN-side Gamma is captured here by construction.]")
    sep = (G_CL / G_LLb) * (G_LC / G_LLb)
    print(f"    separability: (field)x(pops)={sep:.3f} vs total={G_CC/G_LLb:.3f} "
          f"-> {'separable' if abs(math.log10(max(sep,1e-30)/max(G_CC/G_LLb,1e-30)))<0.15 else 'COUPLED (do not attribute additively)'}")

    # ---- pre-registered verdict ----------------------------------------------
    tot = G_CC / G_LLb
    basis_note = (f"over {len(idx_use)} common levels capturing "
                  f"{100*G_LLb/G_LL:.1f}% of LUMINA's full Gamma" if not strict
                  else "over the full sigma grid")
    print(f"\n  VERDICT (pre-registered rule; {basis_note}):")
    if tot >= 5:
        print(f"    Gamma_C/G_L = {tot:.2f} >= 5  ==> the ~10x gap is REAL and sits on the")
        print(f"    {'FIELD' if G_CL/G_LLb > G_LC/G_LLb else 'POPULATIONS'} side "
              f"(field {G_CL/G_LLb:.2f}x vs pops {G_LC/G_LLb:.2f}x).")
        print(f"    ==> G-correction is the right third component; alpha side closed.")
    elif tot <= 2:
        print(f"    Gamma_C/G_L = {tot:.2f} <= 2  ==> NO large Gamma gap. The recorded "
              f"'Gamma(Co III) ~10x' inference is REFUTED.")
        print(f"    ==> re-audit the alpha side (spingate x DR interaction).")
    else:
        print(f"    Gamma_C/G_L = {tot:.2f} is in the undecided band (2,5) -- report as-is, "
              f"do not force a verdict.")
    if not strict:
        print(f"    NOTE: this verdict is over the CMFGEN-modelled level subset only. "
              f"{100*(1-G_LLb/G_LL):.1f}% of LUMINA's full Co III Gamma lives in levels "
              f"CMFGEN does not model -- see the coverage accounting above.")


if __name__ == "__main__":
    main()
