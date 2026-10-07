#!/usr/bin/env python3
"""DET-SIIII-REPAIR-OFFLINE readout (prereg docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md, rev 1).

Subcommands
  selftest --production-source DIR --out DIR
  analyze  --e-run DIR --r-run DIR --prior-mechanism DIR --writer-root DIR --public-spectrum FILE
           --production-source DIR --selftest DIR --out DIR

Offline verification of the repair candidates for the generation-zero seed of the thick
Si III 1206.50 A resonance line (prereg 2.3):
  A    NRB armed coefficients: the measured R' response is re-cited, with the isolated
       armed ratio (radiative and radiative+collision) as the reduced model
  B-T  initial material at the sealed FIXED_T profile: conditional NRB-off seed
       B_l(T_pin,s), iterated through calls 2..7 on the stored E' (and R') background
  B-W  contrast column of B-T only: seed W_s B_l(T_inner)
Construction facts (seed memory q^6 ~ 1, C = A on a fixed background, the D iteration
count) are recorded once and never become result codes (prereg 4.5 / 6.2).
Every number is an interval [lo, hi] with a middle value; 1e-3, 0.05 dex and one half are
registered science criteria, never numerical tolerances.
"""
import argparse
import json
import math
import os
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_siiii_pop_readout as POP                                              # noqa: E402
import det_siiii_rates_readout as RATES                                          # noqa: E402
import det_siiii_1206mech_readout as MECH                                        # noqa: E402
import nt_eta_readout as NTE                                                     # noqa: E402
from det_siiii_pop_readout import I, recover, sha, REC, C, H, C_SOB             # noqa: E402

K_B, EV_TO_ERG, FOUR_PI = MECH.K_B, MECH.EV_TO_ERG, MECH.FOUR_PI
U = 2.0 ** -53
Z, ION = 14, 2
CRIT = 1e-3
TARGET = {"r1206": 600693, "lo": 0, "up": 4, "r2542": 1097363, "u5": 5, "w5_shells": [10, 15, 19, 24, 30],
          "main": 19, "spatial_2542": [17, 18, 19, 20, 21, 22], "n_fixed": 122,
          "ion_counts": {"16_2": 67, "14_2": 32, "20_2": 9, "16_1": 8, "14_1": 6}}
CODE_NAME = {0: "OFFLINE_HYPOTHESES_SURVIVE", 1: "INIT_PREMISE_REFUTED", 3: "REDUCED_MODEL_REFUTED",
             4: "UNRESOLVED", 5: "NOT_IDENTIFIABLE"}
BANDS8 = [(2500, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000),
          (8000, 10000)]
NUV, UV2500, OPT = (2500, 4000), (2500, 3000), (4000, 6000)
KEEP4 = [(3000, 3500), (3500, 4000), (4000, 5000), (5000, 6000)]
PUBLIC_TIME = 19.48
SRC_FILES = ("src/lumina_atomic.c", "src/lumina_cmfgen.c", "src/lumina_plasma.c", "src/lumina_cuda.cu")
SEED_SRC_RE = re.compile(r"Generation-zero material-temperature seed.*?"
                         r"for \(int i = 0; i < geo->n_shells; i\+\+\)\s*(?:/\*.*?\*/\s*)?\{\s*(?:/\*.*?\*/\s*)?"
                         r"opacity->t_electrons\[i\]\s*=\s*config->T_inner;", re.S)
WARM_RE = re.compile(r"fs\.J\[idx\]\s*=\s*Bnu;")
FAULTS = ("warmstart_as_material", "claim_origin", "beta_cancel_with_coll", "omit_n0_ratio", "drop_kappa", "beta_only")

status = MECH.det_1206mech_status
allst = MECH.det_1206mech_all
absdev = MECH.det_1206mech_absdev
ivj = MECH.ivj
up, dn = MECH.up, MECH.dn


def det_siiii_repair_log10(a):
    """log10 of a positive interval (outward by a few ulps)."""
    if a is None or not (a.lo > 0.0):
        return None
    lo, mid, hi = math.log10(a.lo), math.log10(a.mid), math.log10(a.hi)
    return I(dn(lo - 4 * U * abs(lo)), mid, up(hi + 4 * U * abs(hi)))


def det_siiii_repair_direction(l):
    """sign of a predicted log change: UP / DOWN, or ZERO_NOT_SEPARATED when the interval holds 0."""
    if l is None:
        return "NOT_IDENTIFIABLE"
    return "UP" if l.lo > 0.0 else "DOWN" if l.hi < 0.0 else "ZERO_NOT_SEPARATED"


def det_siiii_repair_cols(row, key, v):
    row[f"{key}_lo"] = None if v is None else v.lo
    row[f"{key}_mid"] = None if v is None else v.mid
    row[f"{key}_hi"] = None if v is None else v.hi


# ------------------------------------------------------------------ inputs
def det_siiii_repair_fixed_te(run):
    """sealed fixed_te_profile: shell -> float(token) (the production parses the token)."""
    prof = {}
    for ln in open(Path(run) / "input/fixed_te_profile.txt"):
        p = ln.split()
        if len(p) >= 2 and p[0].isdigit():
            prof[int(p[0])] = float(p[1])
    return prof


def det_siiii_repair_geometry(run):
    """W_s = (1/2)[1 - sqrt(1 - x^2)], x = R_in/r_s, evaluated in the cancellation-free form
    x^2 / (2 (1 + sqrt(1 - x^2))); interval (24 + 8/(1-x^2)) u."""
    g = pd.read_csv(Path(run) / "input/model/geometry.csv", float_precision="round_trip").set_index("shell_id")
    R_in = float(g.loc[g.index.min(), "r_inner"])
    W = {}
    for s in g.index:
        r = 0.5 * (float(g.loc[s, "r_inner"]) + float(g.loc[s, "r_outer"]))
        x = R_in / r
        if not (x < 1.0):
            W[int(s)] = None
            continue
        w = 0.5 * (x * x) / (1.0 + math.sqrt(1.0 - x * x))
        W[int(s)] = I.rel(w, 24.0 + 8.0 / (1.0 - x * x))
    return W


def det_siiii_repair_atom_rows(run, rows):
    """exact floats of the given 0-based line_list data rows and their level data."""
    want = sorted(set(int(r) for r in rows))
    ws = set(want)
    ll = pd.read_csv(Path(run) / "input/model/line_list.csv", skiprows=lambda i: i > 0 and (i - 1) not in ws,
                     float_precision="round_trip")
    if len(ll) != len(want):
        return None
    ll.index = want
    lv = pd.read_csv(Path(run) / "input/model/levels.csv", float_precision="round_trip")
    lv = lv.set_index(["atomic_number", "ion_number", "level_number"])
    out = {}
    for row in want:
        r = ll.loc[row]
        z, ion, lo, upl = int(r.atomic_number), int(r.ion_number), int(r.level_number_lower), int(r.level_number_upper)
        out[row] = {"Z": z, "ion": ion, "lo": lo, "up": upl, "nu": float(r.nu), "lam_cm": float(r.wavelength_cm),
                    "f_lu": float(r.f_lu), "A_ul": float(r.A_ul), "B_lu": float(r.B_lu), "B_ul": float(r.B_ul),
                    "g_l": float(lv.loc[(z, ion, lo), "g"]), "g_u": float(lv.loc[(z, ion, upl), "g"]),
                    "E_l": float(lv.loc[(z, ion, lo), "energy_eV"]), "E_u": float(lv.loc[(z, ion, upl), "energy_eV"])}
    return out


def det_siiii_repair_w5_load(run, keys):
    """W5 history of the requested (line, shell) keys only, one pass (MECH loader semantics)."""
    keys = set(keys)
    lines = {k[0] for k in keys}
    hist, contra = {}, []
    cols = ["call", "operator", "rad_generation", "material_generation", "shell", "line_id", "Z", "ion", "srce_chk",
            "status"] + MECH.W5_NUM
    for ch in pd.read_csv(Path(run) / "work/w5_lines.csv", usecols=cols, na_values=["NOT_APPLICABLE"],
                          dtype={"operator": str, "status": str, "srce_chk": str, "call": int, "shell": int,
                                 "line_id": int, "Z": int, "ion": int, "rad_generation": int,
                                 "material_generation": int}, float_precision="round_trip", chunksize=1 << 18):
        ch = ch[ch.line_id.isin(lines)]
        for r in ch.itertuples(index=False):
            d = r._asdict()
            k = (d["line_id"], d["shell"])
            if k not in keys:
                continue
            h = hist.setdefault(k, {})
            if d["call"] in h:
                if h[d["call"]] != d:
                    contra.append((k, d["call"]))
                continue
            h[d["call"]] = d
    return hist, contra


def det_siiii_repair_key_check(h, atom, z, ion):
    """per-key admissibility: every present call row passes the MECH row check, the off form
    (srce_chk 0) was consumed from call 2 on, and frequency and ion match the sealed row."""
    why = []
    for c, r in h.items():
        why += [w for w in MECH.det_1206mech_check_row(r, c) if w != "W5_CALL_MISSING"]
        if float(r["nu_hz"]) != atom["nu"] or int(r["Z"]) != z or int(r["ion"]) != ion:
            why.append(f"W5_IDENTITY:call{c}")
        if c >= 2 and str(r["srce_chk"]) not in ("0", "0.0"):
            why.append(f"SRCE_CHK_APPLIED:call{c}")
    return why


# ------------------------------------------------------------------ INIT origin (prereg 3)
def det_siiii_repair_S_boot(atom, T):
    """S_boot(T) = K_nu r_LTE/(1 - a r_LTE), K_nu = h nu A/(4 pi k_tau), k_tau = C_Sob f lam nu/c,
    r_LTE = (g_u/g_l) exp(-(E_u-E_l)/kT) (prereg 3.2).  Interval (8x + 32 + (8x+8) a r/den) u."""
    nu, lam, f, A = atom["nu"], atom["lam_cm"], atom["f_lu"], atom["A_ul"]
    a = atom["g_l"] / atom["g_u"]
    k_tau = C_SOB * f * lam * nu / C
    K = H * nu * A / (FOUR_PI * k_tau)
    x = (atom["E_u"] - atom["E_l"]) * EV_TO_ERG / (K_B * T)
    r = atom["g_u"] / atom["g_l"] * math.exp(-x)
    den = 1.0 - a * r
    if not (den > 0.0 and r > 0.0):
        return None, r
    return I.rel(K * r / den, 8.0 * x + 32.0 + (8.0 * x + 8.0) * a * r / den), r


def det_siiii_repair_run_logs(run, T_inner):
    """within-stream order evidence only (stdout and stderr line numbers are never compared)."""
    so, se = Path(run) / "stdout.log", Path(run) / "stderr.log"
    lo = so.read_text(errors="replace").splitlines() if so.exists() else []
    le = se.read_text(errors="replace").splitlines() if se.exists() else []

    def first(lines, pred):
        for i, l in enumerate(lines):
            if pred(l):
                return i
        return None
    cfg = first(lo, lambda l: "[CONFIG-PREC] effective T_inner=" in l)
    boot = first(lo, lambda l: "[A2-10][SEED] bootstrap T_e published generation=1" in l)
    cons = first(lo, lambda l: "first consumer=" in l and "transport" in l)
    supp = first(lo, lambda l: "[BOOTSTRAP] iteration-0 material supplier OPEN" in l)
    init = first(le, lambda l: "[cmf_fine][W5-CONT] call=1 operator=INIT_SHARED_GAUSSIAN" in l
                 and "material_generation=0" in l)
    fixed = first(le, lambda l: "te_lane=FIXED_T" in l)
    out = {"stdout": str(so), "stderr": str(se), "stdout_sha256": sha(so) if so.exists() else None,
           "stderr_sha256": sha(se) if se.exists() else None}
    for nm, v in (("config_T_inner_line", cfg), ("bootstrap_published_line", boot), ("first_transport_consumer_line", cons),
                  ("bootstrap_supplier_line", supp), ("init_call1_line_stderr", init), ("first_fixed_T_line_stderr", fixed)):
        out[nm] = None if v is None else v + 1
    if cfg is None:
        out["config"] = "NOT_IDENTIFIABLE"
    else:
        m = re.search(r"effective T_inner=([0-9.eE+-]+) K", lo[cfg])
        out["config_T_inner"] = None if m is None else float(m.group(1))
        out["config"] = ("NOT_IDENTIFIABLE" if m is None else "SURVIVES" if float(m.group(1)) == T_inner else "REFUTED")
    out["order_stdout"] = ("NOT_IDENTIFIABLE" if (boot is None or cons is None) else
                           "SURVIVES" if boot < cons else "REFUTED")
    out["order_stderr"] = ("NOT_IDENTIFIABLE" if init is None else
                           "REFUTED" if (fixed is not None and fixed < init) else "SURVIVES")
    if supp is None or init is None:
        out["supplier"] = "NOT_IDENTIFIABLE"
    else:
        out["supplier"] = "SURVIVES" if ("LTE/Saha" in lo[supp] and "provenance=BOOTSTRAP_LTE_SAHA" in lo[supp]) else "REFUTED"
    return out


def det_siiii_repair_init_origin(e_run, r_run, src, T_inner, fault=None):
    srcdir = Path(src)
    ev = {"production_source": str(srcdir)}
    atomic = srcdir / "src/lumina_atomic.c"
    cmf = srcdir / "src/lumina_cmfgen.c"
    atomic_c = atomic.read_text(errors="replace") if atomic.exists() else ""
    cmf_c = cmf.read_text(errors="replace") if cmf.exists() else ""
    seed_src = bool(SEED_SRC_RE.search(atomic_c))
    warm_src = bool(WARM_RE.search(cmf_c))
    ev["source_seed_assignment_T_inner"] = seed_src
    ev["source_warm_start_B_Te_unchanged"] = warm_src
    ev["source_sha256"] = {f: (sha(srcdir / f) if (srcdir / f).exists() else None) for f in SRC_FILES}
    runs = {tag: det_siiii_repair_run_logs(run, T_inner) for tag, run in (("E", e_run), ("R", r_run))}
    ev["runs"] = runs
    st_src = "NOT_IDENTIFIABLE" if not atomic_c else ("SURVIVES" if seed_src else "REFUTED")
    IT = allst(st_src, *[runs[t][k] for t in ("E", "R") for k in ("config", "order_stdout", "order_stderr")])
    if fault == "warmstart_as_material" and not warm_src:
        IT = "REFUTED"                     # fault: a changed warm start counted as a material change
    IP = allst(runs["E"]["supplier"], runs["R"]["supplier"])
    origin = "LOCAL_LINE_EMISSION" if fault == "claim_origin" else "ORIGIN_NOT_SEPARATED"
    return {"I_T": IT, "I_T_source": st_src, "I_P": IP, "evidence": ev, "origin_attribution": origin,
            "T_boot": T_inner if st_src == "SURVIVES" else None, "T_inner": T_inner,
            "warm_start_role": "initial estimate fs.J = B_nu(T_e) of the exact transfer solve; not a material or boundary input",
            "not_specified": ["source-by-source contribution of the INIT transfer solution (initial coefficient and field arrays "
                              "are not stored)", "cross-stream ordering between stdout and stderr"]}


# ------------------------------------------------------------------ 1206 candidates (prereg 4)
def det_siiii_repair_bg(row):
    beta1 = I.rel(float(row["beta"]), 2)
    jc, je = float(row["J_c"]), float(row["J_c_err_upper"])
    return beta1, I(dn(jc - je), jc, up(jc + je))


def det_siiii_repair_map(J1, rows, atom, kappa, dB, fault=None):
    """iterate the conditional off map F_k from J1 through calls 2..7 on the stored background
    rows (beta_{k+1}, J_c,{k+1}).  Returns J = {1..7}, M7 = |prod q_k| and q_k (intervals)."""
    if J1 is None:
        return None, None, None
    J = {1: J1}
    M = I.x(1.0)
    qs = []
    for k in range(1, 7):
        r1 = rows.get(k + 1)
        if r1 is None:
            return None, None, None
        beta1, Jc1 = det_siiii_repair_bg(r1)
        kap = I.x(1.0) if fault == "drop_kappa" else kappa
        F, q, F0, q0 = MECH.det_1206mech_F(J[k], beta1, Jc1, atom, kap, dB)
        if fault == "beta_only":
            F, q = F0, q0
        J[k + 1] = F
        M = M * q
        qs.append(q)
    return J, M.abs(), qs


def det_siiii_repair_rC(J, atom, Cu, Cd):
    """r_off+C(J) = (B_lu J + C_lu)/(A + B_ul J + C_ul)."""
    return (I.x(atom["B_lu"]) * J + Cu).div(I.x(atom["A_ul"]) + I.x(atom["B_ul"]) * J + Cd)


def det_siiii_repair_rAC(Jc, beta, atom, Cu, Cd, fault=None):
    """r_A+C = (beta B_lu J_c + C_lu)/(beta (A + B_ul J_c) + C_ul); with collisions beta does not cancel."""
    if fault == "beta_cancel_with_coll":
        beta = I.x(1.0)
    return (beta * I.x(atom["B_lu"]) * Jc + Cu).div(beta * (I.x(atom["A_ul"]) + I.x(atom["B_ul"]) * Jc) + Cd)


def det_siiii_repair_iters(q):
    """D cell: m = ceil(log(1e-3)/log q) for 0 < q < 1 (interval ends); undefined otherwise."""
    if q is None or not (q.lo > 0.0 and q.hi < 1.0):
        return None, "UNDEFINED_Q_NOT_IN_(0,1)"
    m = [math.ceil(math.log(1e-3) / math.log(v)) for v in (q.lo, q.hi)]
    return [min(m), max(m)], "OK"


def det_siiii_repair_code(premise, model):
    """prereg 6.2 priority: premise refuted -> 1; model refuted -> 3; all survive -> 0;
    missing -> 5; else 4.  Code 2 is never produced."""
    if "REFUTED" in premise:
        return 1
    if "REFUTED" in model:
        return 3
    allv = list(premise) + list(model)
    if all(v == "SURVIVES" for v in allv):
        return 0
    if "NOT_IDENTIFIABLE" in allv:
        return 5
    return 4


# ------------------------------------------------------------------ 2542 response (prereg 5)
def det_siiii_repair_P(alpha4, q, tau, mode):
    """local-injection ratio Q'/Q for positive tau: P1 (alpha5 = alpha4) or P2 (alpha5 = 1).
    Returns (interval or None, flag); a non-positive alpha4 - alpha5 q is recorded, not truncated."""
    if not (tau.lo > 0.0):
        return None, "TAU_NOT_POSITIVE"
    if not (q.hi < 1.0):
        return None, "INVERSION_Q_GE_1" if q.lo >= 1.0 else "Q_UNRESOLVED_AT_1"
    e = lambda t: -math.expm1(-t)                                      # noqa: E731
    if mode == "P1":
        d = alpha4 * (I.x(1.0) - q)
        if not (d.lo > 0.0):
            return None, ("SIGN_CHANGE" if d.hi <= 0.0 else "SIGN_UNRESOLVED")
        ends = [e(a * t1) / e(t0) for a in (alpha4.lo, alpha4.hi) for t1 in (tau.lo, tau.hi) for t0 in (tau.lo, tau.hi)]
        mid = e(alpha4.mid * tau.mid) / e(tau.mid)
    else:
        d = alpha4 - q
        if not (d.lo > 0.0):
            return None, ("SIGN_CHANGE" if d.hi <= 0.0 else "SIGN_UNRESOLVED")
        one_q = I.x(1.0) - q
        taup = (tau * d).div(one_q)
        Sr = one_q.div(d)
        ends = [sr * e(tp) / e(t0) for sr in (Sr.lo, Sr.hi) for tp in (taup.lo, taup.hi) for t0 in (tau.lo, tau.hi)]
        mid = Sr.mid * e(taup.mid) / e(tau.mid)
    return I(dn(min(ends) * (1 - 32 * U)), mid, up(max(ends) * (1 + 32 * U))), "OK"


# ------------------------------------------------------------------ public comparison (prereg 5.5)
def det_siiii_repair_public(path, time=PUBLIC_TIME):
    """public CMFGEN column at exactly 19.48 d (no nearest-time substitution)."""
    times, rows = None, []
    for ln in open(path):
        if ln.startswith("#TIMES"):
            times = [float(x) for x in ln.split(":")[1].split()]
        elif ln.startswith("#") or not ln.strip():
            continue
        else:
            rows.append([float(x) for x in ln.split()])
    if times is None or time not in times:
        return None
    k = times.index(time)
    return sorted((r[0], r[1 + k]) for r in rows)


def det_siiii_repair_spec(path):
    out = []
    with open(path) as f:
        next(f)
        for ln in f:
            a, b = ln.split(",")[:2]
            out.append((float(a), float(b)))
    return sorted(out)


def det_siiii_repair_band_err(pts, lo, hi):
    """bound of the floating error of NTE.band_integral over [lo, hi]: every overlapping segment adds
    a term (b-a)(fa+fb)/2 with fa, fb within 2u(|f0|+|f1|) of exact and <= 8 more roundings, the sum
    adds <= n u of the running total: (n + 16) u sum (b-a) max(|f0|,|f1|)."""
    n, s = 0, 0.0
    for (l0, f0), (l1, f1) in zip(pts[:-1], pts[1:]):
        a, b = max(l0, lo), min(l1, hi)
        if b <= a:
            continue
        n += 1
        s += (b - a) * max(abs(f0), abs(f1))
    return (n + 16) * U * s * (1.0 + 1e-9)


def det_siiii_repair_bands(pts):
    """band luminosities as intervals (value from NTE.band_integral, the cmp_pub_paired.py integral)."""
    if pts is None:
        return None
    out = {}
    for b in BANDS8 + [NUV, OPT]:
        v, st = NTE.band_integral(pts, float(b[0]), float(b[1]))
        if st != "OK" or not (v > 0.0):
            out[b] = None
            continue
        e = det_siiii_repair_band_err(pts, float(b[0]), float(b[1]))
        out[b] = I(dn(v - e), v, up(v + e))
    return out


def det_siiii_repair_le(x, t):
    """registered criterion x <= t on intervals."""
    if x is None or t is None:
        return "NOT_IDENTIFIABLE"
    return "SURVIVES" if x.hi <= t.lo else "REFUTED" if x.lo > t.hi else "UNRESOLVED"


def det_siiii_repair_lt(x, t):
    """registered criterion x < t on intervals."""
    if x is None or t is None:
        return "NOT_IDENTIFIABLE"
    return "SURVIVES" if x.hi < t.lo else "REFUTED" if x.lo >= t.hi else "UNRESOLVED"


def det_siiii_repair_spectrum_table(specs):
    """cmp_pub_paired.py definitions: C_b = L_b/L_OPT, d_b = log10(C_b/C_b^pub), R8 = rms of the 8 bands;
    every quantity carries its floating-error interval so a boundary crossing stays UNRESOLVED."""
    B = {nm: det_siiii_repair_bands(p) for nm, p in specs.items()}
    pub = B.get("public")
    if pub is None or pub[OPT] is None:
        return [], {"status": "PUBLIC_NOT_AVAILABLE"}
    rows, summ = [], {}
    for nm, b in B.items():
        if b is None or b[OPT] is None:
            summ[nm] = {"status": "NOT_IDENTIFIABLE"}
            continue
        d = {}
        for band in BANDS8 + [NUV, OPT]:
            if b[band] is None or pub[band] is None:
                d[band] = None
                rows.append({"spectrum": nm, "band": f"{band[0]}-{band[1]}", "status": "UNSUPPORTED"})
                continue
            c, cp = b[band].div(b[OPT]), pub[band].div(pub[OPT])
            d[band] = det_siiii_repair_log10(c.div(cp))
            row = {"spectrum": nm, "band": f"{band[0]}-{band[1]}", "status": "OK", "L_band": b[band].mid,
                   "L_band_public": pub[band].mid, "L_band_over_public": b[band].mid / pub[band].mid,
                   "L_OPT": b[OPT].mid, "L_OPT_public": pub[OPT].mid, "colour": c.mid, "colour_public": cp.mid}
            det_siiii_repair_cols(row, "d_dex", d[band])
            rows.append(row)
        r8 = None
        if all(d[x] is not None for x in BANDS8):
            sq = I.x(0.0)
            for x in BANDS8:
                sq = sq + d[x].abs() * d[x].abs()
            # a sum of squares is >= 0 exactly; a negative lower end is only the outward rounding of 0
            lo8 = max(sq.lo, 0.0) / 8.0
            r8 = I(max(0.0, dn(math.sqrt(lo8))), math.sqrt(sq.mid / 8.0), up(math.sqrt(sq.hi / 8.0)))
        summ[nm] = {"status": "OK", "R8": ivj(r8), "NUV_over_OPT": None if b[NUV] is None else b[NUV].mid / b[OPT].mid,
                    "d_NUV": ivj(d[NUV]), "d_UV2500": ivj(d[UV2500]),
                    "d_keep4": {f"{x[0]}-{x[1]}": ivj(d[x]) for x in KEEP4},
                    "L_OPT": b[OPT].mid, "L_OPT_over_public": b[OPT].mid / pub[OPT].mid,
                    "d_8bands": {f"{x[0]}-{x[1]}": ivj(d[x]) for x in BANDS8}}
    return rows, summ


def det_siiii_repair_targets(summ):
    """registered secondary expectations (prereg 5.6) from E'N, applied as interval criteria to every
    stored spectrum: status SURVIVES / REFUTED / UNRESOLVED / NOT_IDENTIFIABLE."""
    def iv(x):
        return None if x is None else I(x["lo"], x["mid"], x["hi"])
    eN = summ.get("EprimeN", {})
    if eN.get("status") != "OK" or None in (eN.get("d_NUV"), eN.get("d_UV2500"), eN.get("R8")):
        return None
    tN, tU, tR = iv(eN["d_NUV"]).abs().half(), iv(eN["d_UV2500"]).abs().half(), iv(eN["R8"])
    t = {"abs_d_NUV_max": tN.j(), "abs_d_UV2500_max": tU.j(), "abs_d_keep4_max": 0.05, "R8_below": tR.j()}
    for nm, s_ in summ.items():
        if not isinstance(s_, dict) or s_.get("status") != "OK":
            continue
        dN, dU, r8 = iv(s_["d_NUV"]), iv(s_["d_UV2500"]), iv(s_["R8"])
        st = {"NUV": det_siiii_repair_le(None if dN is None else dN.abs(), tN),
              "UV2500": det_siiii_repair_le(None if dU is None else dU.abs(), tU),
              "R8": det_siiii_repair_lt(r8, tR)}
        for k_, v in s_["d_keep4"].items():
            st[k_] = det_siiii_repair_le(None if v is None else iv(v).abs(), I.x(0.05))
        s_["target_status"] = st
        s_["meets_secondary_targets"] = allst(*st.values())
    return t


# ------------------------------------------------------------------ analyze
def det_siiii_repair_analyze(e_run, r_run, prior_mech, writer_root, public_spectrum, src, out, target=TARGET,
                             fault=None, n0_ratio=1.0):
    """n0_ratio is the registered conditional assumption n0'/n0 = 1 (prereg 5.2); other values and `target`
    are selftest-only function arguments, not CLI options."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    why = []
    mE, mR = MECH.det_1206mech_meta(e_run), MECH.det_1206mech_meta(r_run)
    why += [f"E:{w}" for w in mE["why"]] + [f"R:{w}" for w in mR["why"]]
    for k in ("t_exp", "T_inner", "b_inner", "vdop", "ppd", "lam_hi"):
        if mE[k] != mR[k]:
            why.append(f"LINK_META_E_R_DIFFER:{k}")
    if mE["nrb"] or not mR["nrb"]:
        why.append("LINK_NRB_BRANCH")
    profE, profR = det_siiii_repair_fixed_te(e_run), det_siiii_repair_fixed_te(r_run)
    if profE != profR:
        why.append("LINK_FIXED_TE_E_R_DIFFER")
    srcdir = Path(src)
    for tag, run in (("E", e_run), ("R", r_run)):
        rb = Path(run) / "input/run_build_identity.json"
        if not rb.exists():
            why.append(f"{tag}:LINK_RUN_BUILD_IDENTITY_MISSING")
            continue
        ent = {f["path"]: f["sha256"] for f in json.load(open(rb)).get("source_files", [])}
        for f in SRC_FILES:
            if f not in ent or not (srcdir / f).exists() or ent[f] != sha(srcdir / f):
                why.append(f"{tag}:LINK_SOURCE_NOT_RUN_BUILD:{f}")
        if POP.deck_sha_of(run, "line_list.csv") != sha(Path(run) / "input/model/line_list.csv"):
            why.append(f"{tag}:LINK_LINE_LIST_NOT_SEALED")
    # fixed extension set (prereg 2.2): selected from the prior census by the registered criteria only
    cen = pd.read_csv(Path(prior_mech) / "census.csv", float_precision="round_trip")
    fixed = cen[(cen.set == "H") & (cen.uv == True) & (cen.E_seed_and_retained == "SURVIVES")]   # noqa: E712
    if len(fixed) != target["n_fixed"]:
        why.append(f"LINK_FIXED_SET_COUNT:{len(fixed)}")
    if target.get("ion_counts"):
        got = {f"{z}_{i}": int(len(g)) for (z, i), g in fixed.groupby(["Z", "ion"])}
        if got != target["ion_counts"]:
            why.append(f"LINK_FIXED_SET_IONS:{got}")
    fixed_keys = [(int(k.line), int(k.shell)) for k in fixed.itertuples()]
    rows_needed = sorted(set([target["r1206"], target["r2542"]] + [k[0] for k in fixed_keys]))
    atoms = det_siiii_repair_atom_rows(e_run, rows_needed)
    atomsR = det_siiii_repair_atom_rows(r_run, rows_needed)
    if atoms is None or atoms != atomsR:
        why.append("LINK_ATOM_ROWS")
        atoms = atoms or {}
    a1206, a2542 = atoms.get(target["r1206"]), atoms.get(target["r2542"])
    if a1206 is None or (a1206["Z"], a1206["ion"], a1206["lo"], a1206["up"]) != (Z, ION, target["lo"], target["up"]):
        why.append("LINK_TARGET_ROW_1206")
    if a2542 is None or (a2542["Z"], a2542["ion"], a2542["lo"], a2542["up"]) != (Z, ION, target["up"], target["u5"]):
        why.append("LINK_TARGET_ROW_2542")
    for k in fixed.itertuples():
        a = atoms.get(int(k.line))
        if a is None or (a["Z"], a["ion"]) != (int(k.Z), int(k.ion)):
            why.append(f"LINK_FIXED_KEY_ATOM:{int(k.line)}")
    w5keys = set(fixed_keys) | {(target["r1206"], s) for s in target["w5_shells"]}
    hE, cE = det_siiii_repair_w5_load(e_run, w5keys)
    hR, cR = det_siiii_repair_w5_load(r_run, w5keys)
    contra_E = {k for k, _ in cE}
    if a1206 is not None and not why:
        for s in target["w5_shells"]:
            k = (target["r1206"], s)
            for tag, h, cc, run, m in (("E", hE, contra_E, e_run, mE), ("R", hR, {k_ for k_, _ in cR}, r_run, mR)):
                if k in cc:
                    why.append(f"{tag}:W5_CONTRADICTORY_DUPLICATE:{s}")
                hk = h.get(k, {})
                why += [f"{tag}:LINK_{w}:{s}" for w in det_siiii_repair_key_check(hk, a1206, Z, ION)]
                # cross-link: call-7 tau_eff of material generation 7 equals the captured tau of this shell
                cap_, _ = MECH.det_1206mech_capture_ratio(run, target["r1206"], s, a1206, m["t_exp"])
                if cap_ is not None and 7 in hk and float(hk[7]["tau_eff"]) != cap_["tau"]:
                    why.append(f"{tag}:LINK_TAU_W5_CALL7_VS_CAPTURE:{s}")
    if why:
        res = {"schema": "DET_SIIII_REPAIR_RESULT_V1", "validity": "X", "reasons": why, "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res

    T_in, b_in = mE["T_inner"], mE["b_inner"]
    # ---- INIT origin (prereg 3)
    init = det_siiii_repair_init_origin(e_run, r_run, src, T_in, fault)
    grid = MECH.det_1206mech_grid(mE["vdop"], mE["ppd"], mE["lam_hi"])
    W = det_siiii_repair_geometry(e_run)
    cap = {tag: {nm: np.fromfile(Path(run) / f"work/det_obs_capture/{nm}.f64", "<f8") for nm in ("Te", "ne")}
           for tag, run in (("E", e_run), ("R", r_run))}
    kappa, dB, _ = MECH.det_1206mech_kappa(a1206)
    B_in, _ = MECH.det_1206mech_profile_mean(grid, a1206["nu"], T_in, b_in)          # inner-boundary comparison
    B_boot, _ = MECH.det_1206mech_profile_mean(grid, a1206["nu"], T_in, 1.0)        # local thermal at T_boot
    S_boot, r_lte = det_siiii_repair_S_boot(a1206, T_in)
    dIS = absdev(S_boot, B_boot) if S_boot is not None else None
    init.update({"T_pin": {str(s): profE.get(s) for s in target["w5_shells"]}, "r_LTE_T_boot": r_lte,
                 "S_boot": ivj(S_boot), "B_profile_T_boot": ivj(B_boot), "B_profile_T_inner_b_inner": ivj(B_in),
                 "I_S_dev": ivj(dIS), "I_S": status(dIS, CRIT)})
    init["I_J"] = {}
    for s in target["w5_shells"]:
        r1 = hE.get((target["r1206"], s), {}).get(1)
        J1 = MECH.det_1206mech_jiv(r1) if r1 is not None else None
        d1 = absdev(J1, S_boot) if (J1 is not None and S_boot is not None) else None
        d2 = absdev(J1, B_in) if J1 is not None else None
        init["I_J"][str(s)] = {"J1": ivj(J1), "dev_vs_S_boot": ivj(d1), "dev_vs_B_profile": ivj(d2),
                               "status": allst(status(d1, CRIT), status(d2, CRIT))}
    # ---- five-shell candidate table (prereg 4.1-4.5)
    cand_rows, codes, D_cell, construction = [], {}, None, {}
    for s in target["w5_shells"]:
        rowsE, rowsR = hE.get((target["r1206"], s), {}), hR.get((target["r1206"], s), {})
        capE, wE = MECH.det_1206mech_capture_ratio(e_run, target["r1206"], s, a1206, mE["t_exp"])
        capR, wR = MECH.det_1206mech_capture_ratio(r_run, target["r1206"], s, a1206, mR["t_exp"])
        CuE, CdE = RATES.det_siiii_rates_coll(float(cap["E"]["ne"][s]), float(cap["E"]["Te"][s]), a1206["E_l"],
                                              a1206["E_u"], a1206["g_l"], a1206["g_u"], a1206["f_lu"])
        CuR, CdR = RATES.det_siiii_rates_coll(float(cap["R"]["ne"][s]), float(cap["R"]["Te"][s]), a1206["E_l"],
                                              a1206["E_u"], a1206["g_l"], a1206["g_u"], a1206["f_lu"])
        rE = None if capE is None else capE["r"]
        rR = None if capR is None else capR["r"]
        row = {"shell": s, "role": "MAIN" if s == target["main"] else "COMPARISON", "T_inner": T_in,
               "T_pin": profE.get(s), "T_e7_E": float(cap["E"]["Te"][s]), "n_e7_E": float(cap["E"]["ne"][s]),
               "T_e7_R": float(cap["R"]["Te"][s]), "n_e7_R": float(cap["R"]["ne"][s]),
               "W_mid": None if W.get(s) is None else W[s].mid, "capture_E": wE or "OK", "capture_R": wR or "OK"}
        det_siiii_repair_cols(row, "E_r7_capture", rE)
        # existing E' state: off model applicability on the baseline (material 7 consumes radiation 6)
        J1E = MECH.det_1206mech_jiv(rowsE[1]) if 1 in rowsE else None
        J6E = MECH.det_1206mech_jiv(rowsE[6]) if 6 in rowsE else None
        r_off6 = MECH.det_1206mech_r_off(J6E, a1206) if J6E is not None else None
        d_base = absdev(rE, r_off6) if (rE is not None and r_off6 is not None) else None
        base_off = status(d_base, CRIT)
        det_siiii_repair_cols(row, "E_r_off_J6", r_off6)
        row["E_baseline_off_dev_mid"] = None if d_base is None else d_base.mid
        row["BT_baseline_applicability"] = base_off
        # A: re-cited R' measurement and the isolated armed ratio at the radiation it consumed (call 6)
        rA = rAC = None
        if 6 in rowsR:
            b6, Jc6 = det_siiii_repair_bg(rowsR[6])
            rA = MECH.det_1206mech_r_off(Jc6, a1206)                       # radiative armed: beta cancels
            rAC = det_siiii_repair_rAC(Jc6, b6, a1206, CuR, CdR, fault)
        dA = absdev(rR, rAC) if (rR is not None and rAC is not None) else None
        dArad = absdev(rAC, rA) if (rA is not None and rAC is not None) else None
        base_A = status(dA, CRIT)
        det_siiii_repair_cols(row, "A_r7_R_capture", rR)
        det_siiii_repair_cols(row, "A_r_A_rad_call6", rA)
        det_siiii_repair_cols(row, "A_r_A_coll_call6", rAC)
        row["A_baseline_dev_mid"] = None if dA is None else dA.mid
        row["A_baseline_applicability"] = base_A
        row["A_radiative_only_dev_mid"] = None if dArad is None else dArad.mid
        row["A_radiative_only_applicability"] = status(dArad, CRIT)
        lA = det_siiii_repair_log10(rR.div(rE)) if (rR is not None and rE is not None) else None
        det_siiii_repair_cols(row, "A_log10_r7_over_E", lA)
        tA = MECH.det_1206mech_Texc(a1206, rR) if rR is not None else None
        det_siiii_repair_cols(row, "A_Texc", tA)
        JR1 = MECH.det_1206mech_jiv(rowsR[1]) if 1 in rowsR else None
        JR7 = MECH.det_1206mech_jiv(rowsR[7]) if 7 in rowsR else None
        row["A_R_endpoint_J7_over_J1_mid"] = (JR7.div(JR1).mid if (JR1 is not None and JR7 is not None) else None)
        # B-T and B-W conditional seeds
        Tp = profE.get(s)
        BT1 = MECH.det_1206mech_profile_mean(grid, a1206["nu"], Tp, 1.0)[0] if Tp is not None else None
        BW1 = (W[s] * B_in) if W.get(s) is not None else None
        res_c = {}
        for cname, key, J1c in (("B-T", "BT", BT1), ("B-W", "BW", BW1)):
            JsE, ME, _ = det_siiii_repair_map(J1c, rowsE, a1206, kappa, dB, fault)
            JsR, _, _ = det_siiii_repair_map(J1c, rowsR, a1206, kappa, dB, fault)
            r1 = MECH.det_1206mech_r_off(J1c, a1206) if J1c is not None else None
            r7 = MECH.det_1206mech_r_off(JsE[6], a1206) if JsE else None          # material 7 <- radiation 6
            r7R = MECH.det_1206mech_r_off(JsR[6], a1206) if JsR else None
            rC1 = det_siiii_repair_rC(J1c, a1206, CuE, CdE) if J1c is not None else None
            rC7 = det_siiii_repair_rC(JsE[6], a1206, CuE, CdE) if JsE else None
            dC1 = absdev(rC1, r1) if (rC1 is not None and r1 is not None) else None
            dC7 = absdev(rC7, r7) if (rC7 is not None and r7 is not None) else None
            st_coll = allst(status(dC1, CRIT), status(dC7, CRIT))
            res_c[cname] = {"J1": J1c, "Js": JsE, "M7": ME}
            det_siiii_repair_cols(row, f"{key}_J1", J1c)
            det_siiii_repair_cols(row, f"{key}_J7_Ebg", None if not JsE else JsE[7])
            det_siiii_repair_cols(row, f"{key}_J7_Rbg", None if not JsR else JsR[7])
            det_siiii_repair_cols(row, f"{key}_M7_Ebg", ME)
            det_siiii_repair_cols(row, f"{key}_r1", r1)
            det_siiii_repair_cols(row, f"{key}_r7_Ebg", r7)
            det_siiii_repair_cols(row, f"{key}_r7_Rbg", r7R)
            row[f"{key}_coll_dev_J1_mid"] = None if dC1 is None else dC1.mid
            row[f"{key}_coll_dev_J6_mid"] = None if dC7 is None else dC7.mid
            row[f"{key}_radiative_only_applicability"] = st_coll
            l1 = det_siiii_repair_log10(J1c.div(J1E)) if (J1c is not None and J1E is not None) else None
            det_siiii_repair_cols(row, f"{key}_log10_seed_over_existing", l1)
            row[f"{key}_seed_direction"] = det_siiii_repair_direction(l1)
            lr = det_siiii_repair_log10(r7.div(rE)) if (r7 is not None and rE is not None) else None
            det_siiii_repair_cols(row, f"{key}_log10_r7_over_E", lr)
            row[f"{key}_r7_direction"] = det_siiii_repair_direction(lr)
            lb = det_siiii_repair_log10(r7R.div(r7)) if (r7 is not None and r7R is not None) else None
            det_siiii_repair_cols(row, f"{key}_log10_Rbg_over_Ebg", lb)
            t = MECH.det_1206mech_Texc(a1206, r7) if r7 is not None else None
            det_siiii_repair_cols(row, f"{key}_Texc", t)
        dJ1 = (res_c["B-W"]["J1"] - res_c["B-T"]["J1"]) if (BW1 is not None and BT1 is not None) else None
        dJ7 = ((res_c["B-W"]["Js"][7] - res_c["B-T"]["Js"][7]) if (res_c["B-W"]["Js"] and res_c["B-T"]["Js"]) else None)
        MWT = dJ7.abs().div(dJ1.abs()) if (dJ7 is not None and dJ1 is not None and dJ1.abs().lo > 0.0) else None
        det_siiii_repair_cols(row, "M7_WT", MWT)
        row["M7_WT_status"] = "OK" if MWT is not None else "UNSEPARATED_DENOMINATOR"
        # codes (prereg 6.2): B-T needs the INIT premises; A only its reduced model
        premise = [init["I_T"], init["I_P"], init["I_S"], init["I_J"][str(s)]["status"]]
        model_BT = [base_off, row["BT_radiative_only_applicability"]]
        model_A = [base_A]
        cBT, cA = det_siiii_repair_code(premise, model_BT), det_siiii_repair_code([], model_A)
        codes[str(s)] = {"A": cA, "A_name": CODE_NAME[cA], "B-T": cBT, "B-T_name": CODE_NAME[cBT],
                         "premise_BT": premise, "model_BT": model_BT, "model_A": model_A}
        row["code_A"], row["code_BT"] = cA, cBT
        cand_rows.append(row)
        if s == target["main"]:
            qD = None
            if J1E is not None and 2 in rowsE:
                b2, Jc2 = det_siiii_repair_bg(rowsE[2])
                qD = MECH.det_1206mech_F(J1E, b2, Jc2, a1206, kappa, dB)[1]
            m, mflag = det_siiii_repair_iters(qD)
            D_cell = {"shell": s, "local_q_at_existing_call1": ivj(qD), "required_iterations_to_1e-3": m, "flag": mflag,
                      "note": "construction arithmetic; not a proof of non-convergence; no iteration extension"}
            construction = {"memory_M7_BT_Ebg_main": ivj(res_c["B-T"]["M7"]),
                            "memory_note": "registered NRB-off background retains the seed (q^6 ~ 1): construction, not a prediction",
                            "C_equiv_A": "on a fixed background C has the isolated ratio of A: construction, no separate column"}
    # ---- 122-key extension (prereg 4.6)
    ext_rows = []
    for k in fixed.itertuples():
        line, s = int(k.line), int(k.shell)
        a = atoms[line]
        h = hE.get((line, s), {})
        kall = det_siiii_repair_key_check(h, a, int(k.Z), int(k.ion))
        # SRCE_CHK (negative-opacity replacement) is a production branch, not a mislink: recorded, not inadmissible
        ksrce = [w for w in kall if w.startswith("SRCE_CHK_APPLIED")]
        kwhy = [w for w in kall if not w.startswith("SRCE_CHK_APPLIED")]
        if (line, s) in contra_E:
            kwhy.append("W5_CONTRADICTORY_DUPLICATE")
        J1 = MECH.det_1206mech_jiv(h[1]) if 1 in h else None
        Bk = MECH.det_1206mech_profile_mean(grid, a["nu"], T_in, b_in)[0]
        Bk1 = MECH.det_1206mech_profile_mean(grid, a["nu"], T_in, 1.0)[0]
        Sk, _ = det_siiii_repair_S_boot(a, T_in)
        dS = absdev(Sk, Bk1) if (Sk is not None and Bk1 is not None) else None
        dJ = absdev(J1, Bk) if (J1 is not None and Bk is not None) else None
        dJS = absdev(J1, Sk) if (J1 is not None and Sk is not None) else None
        st = "INADMISSIBLE" if kwhy else allst(status(dS, CRIT), status(dJ, CRIT), status(dJS, CRIT))
        Tp = profE.get(s)
        BTk = MECH.det_1206mech_profile_mean(grid, a["nu"], Tp, 1.0)[0] if Tp is not None else None
        shift = det_siiii_repair_log10(BTk.div(J1)) if (BTk is not None and J1 is not None) else None
        kap_k, dB_k, _ = MECH.det_1206mech_kappa(a)
        _, Mk, _ = det_siiii_repair_map(BTk, h, a, kap_k, dB_k)
        rr = (MECH.det_1206mech_r_off(BTk, a).div(MECH.det_1206mech_r_off(J1, a))
              if (BTk is not None and J1 is not None) else None)
        er = {"line": line, "shell": s, "Z": a["Z"], "ion": a["ion"], "lambda_A": C / a["nu"] * 1e8,
              "beta_bin": k.beta_bin, "T_pin": Tp, "source_vs_B_dev_mid": None if dS is None else dS.mid,
              "J1_vs_B_dev_mid": None if dJ is None else dJ.mid, "J1_vs_S_dev_mid": None if dJS is None else dJS.mid,
              "initial_source_status": st, "why": ";".join(kwhy), "srce_chk_calls": ";".join(ksrce),
              "BT_r_off_ratio_mid": None if rr is None else rr.mid, "BT_M7_mid": None if Mk is None else Mk.mid}
        det_siiii_repair_cols(er, "BT_seed_shift_log10", shift)
        er["BT_seed_direction"] = det_siiii_repair_direction(shift)
        ext_rows.append(er)
    ext = pd.DataFrame(ext_rows)

    def agg(sub):
        n = int(len(sub))
        if n == 0:
            return {"n": 0}
        d = {"n": n, "initial_source": {k_: int(v) for k_, v in sub.initial_source_status.value_counts().items()},
             "BT_seed_direction": {k_: int(v) for k_, v in sub.BT_seed_direction.value_counts().items()}}
        sh = sub.BT_seed_shift_log10_mid.dropna()
        mk = sub.BT_M7_mid.dropna()
        d.update({"BT_seed_shift_log10_median": float(sh.median()) if len(sh) else None,
                  "BT_seed_shift_log10_min": float(sh.min()) if len(sh) else None,
                  "BT_seed_shift_log10_max": float(sh.max()) if len(sh) else None,
                  "BT_M7_min": float(mk.min()) if len(mk) else None, "BT_M7_max": float(mk.max()) if len(mk) else None})
        surv = int((sub.initial_source_status == "SURVIVES").sum())
        unres = int(sub.initial_source_status.isin(["UNRESOLVED", "NOT_IDENTIFIABLE", "INADMISSIBLE"]).sum())
        d["universal_initial_source"] = "HOLDS" if surv == n else "REFUTED" if surv + unres < n else "UNRESOLVED"
        d["subset_count_bounds"] = [surv, surv + unres]
        d["refuted_keys"] = [[int(r.line), int(r.shell)] for r in sub.itertuples() if r.initial_source_status == "REFUTED"]
        return d
    ext_agg = {"all": agg(ext)}
    if len(ext):
        ext_agg["by_ion"] = {f"{z}_{i}": agg(g) for (z, i), g in ext.groupby(["Z", "ion"])}
        ext_agg["by_shell"] = {str(s_): agg(g) for s_, g in ext.groupby("shell")}
        ext_agg["by_beta_bin"] = {str(b): agg(g) for b, g in ext.groupby("beta_bin")}
        ext_agg["S_III"] = agg(ext[(ext.Z == 16) & (ext.ion == 2)])
        ext_agg["S_II"] = agg(ext[(ext.Z == 16) & (ext.ion == 1)])
    # ---- 2542 response (prereg 5)
    by_shell = {r["shell"]: r for r in cand_rows}
    resp = []
    for s in sorted(set(target["spatial_2542"] + target["w5_shells"])):
        row = {"shell": s, "role": "MAIN" if s == target["main"] else ("W5" if s in target["w5_shells"] else "SPATIAL")}
        st = {}
        for tag, run, m in (("E", e_run, mE), ("R", r_run, mR)):
            c12, _ = MECH.det_1206mech_capture_ratio(run, target["r1206"], s, a1206, m["t_exp"])
            rec = POP.capture_records(Path(run) / "work/det_obs_capture", [target["r2542"]], {s}).get((target["r2542"], s))
            rv = (recover(rec["tau"], rec["S"], a2542["g_l"], a2542["g_u"], a2542["f_lu"], a2542["lam_cm"], m["t_exp"])
                  if (rec is not None and rec["mapped"] == 1) else None)
            if c12 is None or rv is None:
                row[f"{tag}_status"] = "NOT_IDENTIFIABLE"
                continue
            ok = c12["n_u"].lo <= rv["n_l"].hi and rv["n_l"].lo <= c12["n_u"].hi
            row[f"{tag}_status"] = "OK" if ok else "N4_INCONSISTENT"
            for nm, v in (("n0", c12["n_l"]), ("n4_from_1206", c12["n_u"]), ("n4", rv["n_l"]), ("n5", rv["n_u"]),
                          ("q45", rv["q"])):
                det_siiii_repair_cols(row, f"{tag}_{nm}", v)
            row[f"{tag}_S45"], row[f"{tag}_tau45"] = rec["S"], rec["tau"]
            row[f"{tag}_Q45"] = rec["S"] * (-math.expm1(-rec["tau"])) if rec["tau"] > 0 else None
            st[tag] = (c12, rv, rec)
        if row.get("E_status") == "OK" and row.get("R_status") == "OK":
            vE, vR = st["E"][1], st["R"][1]
            det_siiii_repair_cols(row, "A_alpha4", vR["n_l"].div(vE["n_l"]))
            det_siiii_repair_cols(row, "A_alpha5", vR["n_u"].div(vE["n_u"]))
            row["A_Q_ratio"] = (row["R_Q45"] / row["E_Q45"]) if (row["R_Q45"] and row["E_Q45"]) else None
        if row.get("E_status") == "OK":
            c12E, vE, recE = st["E"]
            cr = by_shell.get(s)
            rBT = None
            if cr is not None and cr.get("BT_r7_Ebg_mid") is not None:
                rBT = I(cr["BT_r7_Ebg_lo"], cr["BT_r7_Ebg_mid"], cr["BT_r7_Ebg_hi"])
                row["BT_ratio_source"] = "CONDITIONAL_MATERIAL7_W5_HISTORY"
            elif profE.get(s) is not None:
                BTs = MECH.det_1206mech_profile_mean(grid, a1206["nu"], profE[s], 1.0)[0]
                rBT = MECH.det_1206mech_r_off(BTs, a1206)
                row["BT_ratio_source"] = "SEED_RATIO_ONLY_NO_W5_HISTORY"
            if rBT is not None:
                # alpha4 = (n0'/n0) * (n4/n0)'/(n4/n0); n0'/n0 is a conditional assumption, never an observation
                alpha4 = I.x(n0_ratio) * rBT.div(c12E["r"])
                if fault == "omit_n0_ratio":
                    alpha4 = rBT.div(c12E["r"])
                det_siiii_repair_cols(row, "BT_alpha4", alpha4)
                row["BT_n0_ratio_assumed"] = n0_ratio
                row["BT_alpha4_assumption"] = "n0'/n0 fixed by assumption (conditional, not an observation)"
                tau, q = I.x(recE["tau"]), vE["q"]
                for mode in ("P1", "P2"):
                    p, fl = det_siiii_repair_P(alpha4, q, tau, mode)
                    det_siiii_repair_cols(row, f"{mode}_Q_ratio", p)
                    row[f"{mode}_flag"] = fl
                    a5 = alpha4 if mode == "P1" else I.x(1.0)
                    num = alpha4 - a5 * q                                   # tau'/tau = (a4 - a5 q)/(1 - q)
                    one_q = I.x(1.0) - q
                    det_siiii_repair_cols(row, f"{mode}_alpha5", a5)
                    det_siiii_repair_cols(row, f"{mode}_tau_ratio", num.div(one_q) if one_q.lo > 0.0 else None)
                    det_siiii_repair_cols(row, f"{mode}_S_ratio",
                                          (a5 * one_q).div(num) if (num.lo > 0.0 and one_q.lo > 0.0) else None)
                    det_siiii_repair_cols(row, f"{mode}_n4", alpha4 * vE["n_l"])
                    det_siiii_repair_cols(row, f"{mode}_n5", a5 * vE["n_u"])
        resp.append(row)
    # ---- public comparison (prereg 5.5)
    N = Path(writer_root)
    paths = {"EprimeN": N / "Eprime/run1/work/lumina_spectrum_det_obs.csv",
             "RprimeN": N / "Rprime/run2/work/lumina_spectrum_det_obs.csv",
             "capture_E": Path(e_run) / "work/lumina_spectrum_det_obs.csv",
             "capture_R": Path(r_run) / "work/lumina_spectrum_det_obs.csv"}
    specs = {"public": det_siiii_repair_public(public_spectrum)}
    for nm, p in paths.items():
        specs[nm] = det_siiii_repair_spec(p) if p.exists() else None
    spec_rows, spec_summ = det_siiii_repair_spectrum_table(specs)
    targets = det_siiii_repair_targets(spec_summ) if spec_summ.get("status") != "PUBLIC_NOT_AVAILABLE" else None
    ident = {}
    for a_, b_ in (("capture_E", "EprimeN"), ("capture_R", "RprimeN")):
        pa, pb = paths[a_], paths[b_]
        ident[f"{a_}_vs_{b_}"] = {"byte_identical": bool(pa.exists() and pb.exists() and sha(pa) == sha(pb)),
                                  "sha256": [sha(pa) if pa.exists() else None, sha(pb) if pb.exists() else None]}
    # ---- expectations (prereg 5.6 / 6.4)
    alg = "SELFTEST_PASS_OF_THIS_READOUT (checked by cmd_analyze before the analysis)"
    expectations = {
        "schema": "DET_SIIII_REPAIR_EXPECTATIONS_V1",
        "status": "INPUT_TO_NEXT_USER_DECISION (not a run authorisation)",
        "A": {"definition": "existing NRB armed coefficients (R'), re-cited measurement",
              "readiness": {"algebra_verified": alg,
                            "baseline_applicability": {str(r["shell"]): r["A_baseline_applicability"] for r in cand_rows},
                            "intervention_defined": "LUMINA_BB_NRB=1 (existing production branch)",
                            "seed_delivery_verified": "NOT_APPLICABLE (A prescribes no seed)",
                            "expectations_registered": True, "future_run_authorized": False},
              "re_cited": {str(r["shell"]): {"r7_R": r["A_r7_R_capture_mid"], "R_endpoint_J7_over_J1": r["A_R_endpoint_J7_over_J1_mid"]}
                           for r in cand_rows},
              "public_secondary_targets_met_by_existing_RprimeN": spec_summ.get("RprimeN", {}).get("meets_secondary_targets")},
        "B-T": {"definition": "generation-zero material temperature from the sealed fixed_te_profile before the first "
                              "transport (bootstrap LTE/Saha populations, partition functions, tau, emission, publication)",
                "readiness": {"algebra_verified": alg,
                              "baseline_applicability": {str(r["shell"]): r["BT_baseline_applicability"] for r in cand_rows},
                              "intervention_defined": ("warm start fs.J initial estimate" if fault == "warmstart_as_material"
                                                       else "initial material temperature owner (lumina_atomic.c generation-zero seed) "
                                                            "set from fixed_te_profile, before bootstrap publication"),
                              "seed_delivery_verified": False, "expectations_registered": True,
                              "future_run_authorized": False},
                "primary_expectation": {
                    "status": "FUTURE_TEST",
                    "quantity": "published W5 call-1 Jbar of Si III 1206.50 A (row 600693) at shells 10,15,19,24,30",
                    "criterion": "|Jbar_1 / B_profile(T_pin,s) - 1| <= 1e-3 at every shell",
                    "B_profile_T_pin": {str(r["shell"]): [r["BT_J1_lo"], r["BT_J1_mid"], r["BT_J1_hi"]] for r in cand_rows},
                    "log10_seed_over_existing": {str(r["shell"]): [r["BT_log10_seed_over_existing_lo"],
                                                                   r["BT_log10_seed_over_existing_mid"],
                                                                   r["BT_log10_seed_over_existing_hi"]] for r in cand_rows},
                    "seed_direction": {str(r["shell"]): r["BT_seed_direction"] for r in cand_rows},
                    "extension_seed_shift_by_shell_median_log10": {
                        s_: v.get("BT_seed_shift_log10_median") for s_, v in ext_agg.get("by_shell", {}).items()},
                    "note": "shell 10 moves up (reverse direction); seed-memory removal is not a success condition"},
                "mechanism_expectation": {
                    "status": "FUTURE_TEST",
                    "criterion": "material-7 n4/n0 within 1e-3 of the registered conditional value where the reduced model applies",
                    "registered": {str(r["shell"]): {"r7_Ebg": [r["BT_r7_Ebg_lo"], r["BT_r7_Ebg_mid"], r["BT_r7_Ebg_hi"]],
                                                     "r7_Rbg_mid": r["BT_r7_Rbg_mid"],
                                                     "applicable": r["BT_baseline_applicability"] == "SURVIVES"
                                                     and r["BT_radiative_only_applicability"] == "SURVIVES"} for r in cand_rows},
                    "condition": "stored E' background (beta, J_c) held fixed; R' background shown as sensitivity"},
                "2542_boundary_records": {str(r["shell"]): {"P1": r.get("P1_Q_ratio_mid"), "P1_flag": r.get("P1_flag"),
                                                             "P2": r.get("P2_Q_ratio_mid"), "P2_flag": r.get("P2_flag"),
                                                             "ratio_source": r.get("BT_ratio_source")} for r in resp},
                "secondary_spectral_expectation": {"status": "FUTURE_TEST", "targets": targets,
                                                   "note": "B-T seed delivery does not guarantee these"},
                "spectrum": "NOT_PREDICTED_OFFLINE (transfer, overlap, redistribution and other sources change)"},
        "B-W": {"role": "contrast column of B-T only; no candidate readiness"},
        "construction_facts": {**construction, "D_cell": D_cell}}
    # ---- outputs
    pd.DataFrame(cand_rows).to_csv(out / "candidate_1206.csv", index=False)
    ext.to_csv(out / "extension.csv", index=False)
    pd.DataFrame(resp).to_csv(out / "response_2542.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(out / "spectrum_reference.csv", index=False)
    (out / "init_origin.json").write_text(json.dumps(init, indent=1, default=str))
    (out / "expectations.json").write_text(json.dumps(expectations, indent=1, default=str))
    res = {"schema": "DET_SIIII_REPAIR_RESULT_V1", "validity": "VALID", "reasons": [], "fault": fault,
           "codes": codes, "main_shell": target["main"], "main_codes": codes[str(target["main"])],
           "construction_facts": construction, "D_local_q_required_iterations": D_cell, "extension": ext_agg,
           "spectrum_summary": spec_summ, "spectrum_secondary_targets": targets, "spectrum_identity": ident,
           "init": {k: init[k] for k in ("I_T", "I_T_source", "I_P", "I_S", "origin_attribution", "T_boot")},
           "kappa": ivj(kappa), "Delta_B": ivj(dB), "B_profile_T_inner": ivj(B_in),
           "claim_ceiling": "conditional seed / level-ratio algebra on stored backgrounds, re-cited A, 2542 boundary records, "
                            "public comparison of existing spectra and expectation registration; no spectral residual "
                            "reduction, no production promotion, B-T seed delivery unverified (run requirement 2 open)",
           "unresolved": ["B-T seed delivery (primary expectation of a future run)", "2542 level-5 and n0 response",
                          "spectral response NOT_PREDICTED_OFFLINE", "generation-7 production solution reproduction (inherited)",
                          "INIT transfer source contributions (ORIGIN_NOT_SEPARATED)"]}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def cmd_analyze(a):
    out = Path(a.out)
    stj = json.load(open(Path(a.selftest) / "selftest.json"))
    if stj.get("readout_sha256") != sha(Path(__file__).resolve()) or stj.get("failures") != 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": ["SELFTEST_NOT_THIS_READOUT_OR_FAILED"]}))
        print(json.dumps({"validity": "X"}))
        return 2
    res = det_siiii_repair_analyze(a.e_run, a.r_run, a.prior_mechanism, a.writer_root, a.public_spectrum,
                                   a.production_source, out)
    here = Path(__file__).resolve().parent
    deps = [POP.__file__, RATES.__file__, MECH.__file__, NTE.__file__, here / "nrb_e_readout.py", here / "w5_cont_readout.py",
            "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/figs/cmp_pub_paired.py"]
    prior_man = Path(a.prior_mechanism) / "manifest.json"
    prereg = here.parent / "docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md"
    man = {"schema": "DET_SIIII_REPAIR_MANIFEST_V1", "prereg": {"path": str(prereg), "sha256": sha(prereg)},
           "readout": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
           "dependencies": {str(Path(d).resolve()): (sha(Path(d)) if Path(d).exists() else None) for d in deps},
           "imported_functions": {
               "det_siiii_1206mech_readout": ["det_1206mech_meta", "det_1206mech_check_row", "det_1206mech_jiv",
                                              "det_1206mech_grid", "det_1206mech_profile_mean", "det_1206mech_kappa",
                                              "det_1206mech_r_off", "det_1206mech_F", "det_1206mech_capture_ratio",
                                              "det_1206mech_Texc", "det_1206mech_status", "det_1206mech_all",
                                              "det_1206mech_absdev", "W5_NUM"],
               "det_siiii_pop_readout": ["I", "recover", "capture_records", "deck_sha_of", "sha"],
               "det_siiii_rates_readout": ["det_siiii_rates_coll", "det_siiii_rates_exports (via meta)",
                                           "det_siiii_rates_exponx (via check_row)"],
               "nt_eta_readout": ["band_integral (the object nrb_e_readout re-exports to cmp_pub_paired.py)"]},
           "mech_readout_identity_at_t7": (json.load(open(prior_man)).get("readout") if prior_man.exists() else None),
           "mech_readout_identity_now": {"path": str(Path(MECH.__file__).resolve()), "sha256": sha(Path(MECH.__file__))},
           "selftest": {"path": str(Path(a.selftest) / "selftest.json"), "sha256": sha(Path(a.selftest) / "selftest.json")},
           "production_source": {f: sha(Path(a.production_source) / f) for f in SRC_FILES},
           "argv": sys.argv, "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
           "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "inputs": {}, "outputs": {}}
    for tag, run in (("E", a.e_run), ("R", a.r_run)):
        for rel in ("input/resolved_lumina.exports", "input/deck.sha256", "input/run_build_identity.json",
                    "input/model/line_list.csv", "input/model/levels.csv", "input/model/geometry.csv",
                    "input/fixed_te_profile.txt", "work/det_obs_capture/manifest.json", "work/det_obs_capture/meta.txt",
                    "work/det_obs_capture/lines.bin", "work/det_obs_capture/Te.f64", "work/det_obs_capture/ne.f64",
                    "work/w5_lines.csv", "work/lumina_spectrum_det_obs.csv", "stdout.log", "stderr.log"):
            p = Path(run) / rel
            man["inputs"][f"{tag}:{rel}"] = {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
    for rel in ("census.csv", "result.json", "manifest.json"):
        p = Path(a.prior_mechanism) / rel
        man["inputs"][f"prior:{rel}"] = {"path": str(p), "sha256": sha(p)}
    for p in (Path(a.writer_root) / "Eprime/run1/work/lumina_spectrum_det_obs.csv",
              Path(a.writer_root) / "Rprime/run2/work/lumina_spectrum_det_obs.csv", Path(a.public_spectrum)):
        man["inputs"][str(p)] = {"sha256": sha(p)}
    for p in sorted(out.iterdir()):
        man["outputs"][p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "main": res.get("main_codes"), "reasons": res.get("reasons")}, default=str))
    return 0 if res["validity"] == "VALID" else 2


# ------------------------------------------------------------------ fixture
FXT = {"r1206": 3, "lo": 0, "up": 4, "r2542": 1, "u5": 5, "w5_shells": [10, 15, 19, 24, 30], "main": 19,
       "spatial_2542": [17, 18, 19, 20, 21, 22], "n_fixed": 5, "ion_counts": None}
FX_TEXP = 1683072.0
MOCK_ATOMIC = ("/* Generation-zero material-temperature seed; radiation is not consulted. */\n"
               "opacity->t_electrons = (double *)malloc(geo->n_shells * sizeof(double)); /* Phase 2 */\n"
               "for (int i = 0; i < geo->n_shells; i++) { /* Phase 2 */\n"
               "    opacity->t_electrons[i] = ASSIGN;\n"
               "}\n")


def det_siiii_repair_make_source(root, seed="inner", warm="std"):
    root = Path(root)
    (root / "src").mkdir(parents=True)
    (root / "src/lumina_atomic.c").write_text(MOCK_ATOMIC.replace("ASSIGN", "config->T_inner" if seed == "inner" else "fixed_te[i]"))
    (root / "src/lumina_cmfgen.c").write_text("            fs.J[idx] = Bnu;   /* warm ALI start */\n" if warm == "std" else
                                              "            fs.J[idx] = Bnu_pin;   /* warm ALI start from the pinned profile */\n")
    (root / "src/lumina_plasma.c").write_text("/* mock plasma */\n")
    (root / "src/lumina_cuda.cu").write_text("/* mock cuda */\n")
    return root


def det_siiii_repair_make_fixture(root, w5, pops, pops5, nrb, src, T_inner=10020.0, te_prof=None, ne=3.0e9, Te=9000.0,
                                  logs="ok", A_scale=1.0, cfg_T=None, **kw):
    """1206-MECH fixture + 2542 and 1206 capture rows at the 2542 shells, ne, geometry, fixed_te_profile,
    run_build_identity (of the mock source), stdout/stderr logs and a capture spectrum."""
    root = Path(root)
    run = MECH.det_1206mech_make_fixture(root, w5, pops, nrb, T_inner=T_inner, Te=Te, **kw)
    ll = run / "input/model/line_list.csv"
    if A_scale != 1.0:
        lines = ll.read_text().splitlines()
        f = lines[1 + MECH.FX["row"]].split(",")
        f[11] = repr(float(f[11]) * A_scale)
        lines[1 + MECH.FX["row"]] = ",".join(f)
        ll.write_text("\n".join(lines) + "\n")
        (run / "input/deck.sha256").write_text(f"{sha(ll)}  line_list.csv\n")
    capd = run / "work/det_obs_capture"
    recs = [tuple(r) for r in np.fromfile(capd / "lines.bin", dtype=REC)]
    have = {(int(r[0]), int(r[1])) for r in recs}
    at5, at = MECH.det_1206mech_fx_atom(1), MECH.det_1206mech_fx_atom(3)
    n0_19, n4_19 = pops[19]
    for s, (n4, n5) in pops5.items():
        tau, S, *_ = POP.prod_tau_S(n4, n5, at5["g_l"], at5["g_u"], at5["f_lu"], at5["lam_cm"], FX_TEXP)
        recs.append((1, s, tau, S, 0.0, 1))
        if (3, s) not in have:
            n0 = n4 * (n0_19 / n4_19)
            tau, S, *_ = POP.prod_tau_S(n0, n4, at["g_l"], at["g_u"], at["f_lu"], at["lam_cm"], FX_TEXP)
            recs.append((3, s, tau, S, 0.0, 1))
    np.array(sorted(recs, key=lambda r: (r[0], r[1])), dtype=REC).tofile(capd / "lines.bin")
    np.full(50, ne).tofile(capd / "ne.f64")
    with open(run / "input/model/geometry.csv", "w") as f:
        f.write("shell_id,r_inner,r_outer,v_inner,v_outer\n")
        for s in range(50):
            f.write(f"{s},{1e15 * (1 + 0.2 * s)!r},{1e15 * (1 + 0.2 * (s + 1))!r},{1e8 * (1 + s)!r},{1e8 * (2 + s)!r}\n")
    te_prof = te_prof or {s: 8000.0 + 50.0 * s for s in range(50)}
    with open(run / "input/fixed_te_profile.txt", "w") as f:
        for s in range(50):
            f.write(f"{s} {te_prof[s]!r}\n")
    rb = {"schema": "CMFGEN_TWIN_RUN_BUILD_V1",
          "source_files": [{"path": p, "sha256": sha(Path(src) / p)} for p in SRC_FILES]}
    (run / "input/run_build_identity.json").write_text(json.dumps(rb))
    cT = T_inner if cfg_T is None else cfg_T
    so = ["header", f"  [CONFIG-PREC] effective T_inner={cT:.9f} K source=config.json:T_inner_K (argv channel not defined)"]
    if logs != "no_boot":
        so += ["[A2-10][SEED] bootstrap T_e published generation=1 n_shells=50 reason=deck seed T_e",
               "[BOOTSTRAP] iteration-0 material supplier OPEN (seed-T_e LTE/Saha, provenance=BOOTSTRAP_LTE_SAHA)"]
    so += ["[K-FRESH] first consumer=CUDA upload/CMFGEN/transport computed_generation=2"]
    (run / "stdout.log").write_text("\n".join(so) + "\n")
    init = "[cmf_fine][W5-CONT] call=1 operator=INIT_SHARED_GAUSSIAN rad_generation=1 material_generation=0 assemble_pass=1"
    fx = "[A2-10][LANE] te_lane=FIXED_T"
    (run / "stderr.log").write_text("\n".join(["x"] + ([fx, init] if logs == "fixed_first" else [init, fx])) + "\n")
    (run / "work/lumina_spectrum_det_obs.csv").write_text("wavelength_angstrom,flux\n1000.0,1.0\n")
    return run


def det_siiii_repair_make_shared(root, census_keys, spectra, public_time="1.94800e+01"):
    """prior census, writer spectra and the public spectrum shared by E and R."""
    root = Path(root)
    (root / "prior").mkdir(parents=True)
    with open(root / "prior/census.csv", "w") as f:
        f.write("line,shell,Z,ion,lambda_A,uv,set,beta_bin,E_seed_and_retained\n")
        for (line, s, z, ion, bb, st) in census_keys:
            lam = C / MECH.det_1206mech_fx_atom(line)["nu"] * 1e8
            f.write(f'{line},{s},{z},{ion},{lam!r},True,H,"{bb}",{st}\n')
    (root / "prior/result.json").write_text("{}")
    (root / "prior/manifest.json").write_text(json.dumps({"readout": {"sha256": "fixture"}}))
    for sub, nm in (("writer/Eprime/run1/work", "E"), ("writer/Rprime/run2/work", "R")):
        (root / sub).mkdir(parents=True)
        with open(root / sub / "lumina_spectrum_det_obs.csv", "w") as f:
            f.write("wavelength_angstrom,flux\n")
            for lam, fl in spectra[nm]:
                f.write(f"{lam!r},{fl!r}\n")
    with open(root / "public.txt", "w") as f:
        f.write(f"#NTIMES:  2\n#NWAVE: {len(spectra['public'])}\n#TIMES[d]: 1.77100e+01 {public_time}\n"
                "#wavelength[Ang] flux_t0 flux_t1\n")
        for lam, fl in spectra["public"]:
            f.write(f"{lam!r} {fl * 0.5!r} {fl!r}\n")          # full precision: ratios to writer spectra exact
    return root


# exact truths (selftest only)
FI, _fr, _dexp = MECH.FI, MECH._fr, MECH._dexp


def det_siiii_repair_rout(x, down, digits=70):
    """outward rounding of a rational to `digits` significant digits (keeps exact enclosures small)."""
    if x == 0:
        return x
    e = math.floor(math.log10(abs(float(x))))
    sc = Fraction(10) ** (digits - 1 - e)
    v = x * sc
    return Fraction(math.floor(v) if down else math.ceil(v)) / sc


def det_siiii_repair_truth_ln(x):
    """ln of a positive rational at 80 digits, bracketed by 1e-70 relative plus 1e-70 absolute."""
    from decimal import Decimal, localcontext
    with localcontext() as c:
        c.prec = 80
        v = Fraction((Decimal(x.numerator) / Decimal(x.denominator)).ln())
    e = Fraction(1, 10 ** 70)
    return FI(v - abs(v) * e - e, v + abs(v) * e + e)


def det_siiii_repair_truth_log10(r):
    """exact enclosure of log10 over a positive rational interval r (FI)."""
    L10 = det_siiii_repair_truth_ln(Fraction(10))
    a, b = det_siiii_repair_truth_ln(r.lo).div(L10), det_siiii_repair_truth_ln(r.hi).div(L10)
    return FI(min(a.lo, b.lo), max(a.hi, b.hi))


def det_siiii_repair_truth_Sboot(at, T):
    nu, lam, f, A = _fr(at["nu"]), _fr(at["lam_cm"]), _fr(at["f_lu"]), _fr(at["A_ul"])
    a = _fr(at["g_l"]) / _fr(at["g_u"])
    K = _fr(H) * nu * A / (_fr(FOUR_PI) * (_fr(C_SOB) * f * lam * nu / _fr(C)))
    x = (_fr(at["E_u"]) - _fr(at["E_l"])) * _fr(EV_TO_ERG) / (_fr(K_B) * _fr(T))
    e = _dexp(-x)
    g = _fr(at["g_u"]) / _fr(at["g_l"])
    return FI(K * g * e.lo / (1 - a * g * e.lo), K * g * e.hi / (1 - a * g * e.hi))


def det_siiii_repair_truth_map(J1, rows, at):
    """exact iteration of F_k from the rational interval J1 (F increasing in J while A + dB J > 0),
    endpoints rounded outward to 70 digits after each step; returns {1..7: FI}."""
    lo, hi = det_siiii_repair_rout(J1.lo, True), det_siiii_repair_rout(J1.hi, False)
    out = {1: FI(lo, hi)}
    for k in range(1, 7):
        r1 = rows[k + 1]
        b, jc = float(r1["beta"]), float(r1["J_c"])
        lo = det_siiii_repair_rout(MECH.det_1206mech_truth_F(lo, b, jc, at)[0], True)
        hi = det_siiii_repair_rout(MECH.det_1206mech_truth_F(hi, b, jc, at)[0], False)
        out[k + 1] = FI(lo, hi)
    return out


def det_siiii_repair_truth_roff(J, at):
    """exact r_off = B_lu J/(A + B_ul J) over a rational interval (increasing in J)."""
    Bl, Bu, A = _fr(at["B_lu"]), _fr(at["B_ul"]), _fr(at["A_ul"])
    return FI(Bl * J.lo / (A + Bu * J.lo), Bl * J.hi / (A + Bu * J.hi))


def det_siiii_repair_truth_Texc(at, r):
    """exact T_exc over a rational ratio interval (increasing in r)."""
    return FI(MECH.det_1206mech_truth_Texc(at, r.lo).lo, MECH.det_1206mech_truth_Texc(at, r.hi).hi)


def det_siiii_repair_truth_P(alpha, q, tau, mode):
    """exact Q'/Q: P1 (1-e^{-alpha tau})/(1-e^{-tau}) increasing in alpha; P2 ((1-q)/(alpha-q))
    (1-e^{-tau (alpha-q)/(1-q)})/(1-e^{-tau}) decreasing in alpha (q, tau exact rationals)."""
    def em(x):
        e = _dexp(-x)
        return FI(1 - e.hi, 1 - e.lo)
    den = em(tau)
    if mode == "P1":
        return FI(em(alpha.lo * tau).div(den).lo, em(alpha.hi * tau).div(den).hi)

    def g(a):
        return (FI((1 - q) / (a - q)) * em(tau * (a - q) / (1 - q))).div(den)
    return FI(g(alpha.hi).lo, g(alpha.lo).hi)


SELFTEST_REQUIRED = (
    "link: clean fixture VALID; INIT rows carry no Sobolev-only columns; I-T and I-P survive; code 2 never produced",
    "link: INIT row labelled material generation 1, shifted call, shell swap, frequency mismatch, contradictory duplicate are X",
    "link: missing authority, source fallback, source not the run build, wrong target row and fixed-set count change are X",
    "init: seed source from a fixed profile, FIXED_T before INIT and a config T_inner mismatch refute I-T (B-T code 1, A unchanged)",
    "init: missing bootstrap evidence is NOT_IDENTIFIABLE (code 5); origin stays ORIGIN_NOT_SEPARATED; claiming it is detected",
    "init: atomic data off the Planck relation refute I-S; stored S_boot encloses the exact truth on the clean fixture",
    "init: a warm-start-only source change leaves I-T surviving; counting it as a material change is detected",
    "algebra: B-T seed and iterated J7 enclose the exact truth with kappa != 1 and Delta B != 0; drop-kappa and beta-only break it",
    "algebra: stored log10 (seed, B-T and A level ratios) and T_exc intervals enclose 80-digit exact references; drop-kappa breaks them",
    "algebra: off model keeps the seed difference (M7_WT ~ 1); armed ratio ignores the R' seed; equal seeds -> unseparated denominator",
    "algebra: beta does not cancel with collisions (exact r_A+C enclosed); cancelling beta is detected",
    "algebra: q >= 1 leaves the D iteration count undefined instead of filling it",
    "algebra: a zero predicted seed change is ZERO_NOT_SEPARATED, not a direction (clean fixture DOWN)",
    "2542: recovered n4/n5, alpha4 and the P1/P2 injection ratios enclose exact references; P1 lowers and P2 raises at alpha4 < 1",
    "2542: n0'/n0 is carried explicitly (0.5 halves alpha4) and omitting it is detected; alpha4 below q45 records the sign, not truncated",
    "spectrum (analyze path): OPT-only change moves colours with the absolute UV unchanged; band gain with worse R8 fails; "
    "out-of-range band unsupported",
    "spectrum (analyze path): a residual exactly at the half-target is UNRESOLVED; public 19.48 d missing -> PUBLIC_NOT_AVAILABLE; "
    "capture vs writer identity recorded",
    "extension: a refuted key and a missing key are kept; universal REFUTED with subset bounds",
    "codes: 0, 1, 3, 4 and 5 through the real analyze path, never 2; expectations FUTURE_TEST and seed delivery unverified",
)


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []
    T = out / "fx"

    def case(name, got, want, inj=None, reason=None):
        ok = bool(got == want)
        res.append({"case": name, "pass": ok, "got": str(got)[:400], "want": str(want)[:400], "injection": inj,
                    "reason": reason})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:160]} want {str(want)[:160]}", flush=True)

    def enc(lo, hi, t):
        return Fraction(lo) <= t.lo and t.hi <= Fraction(hi)

    at = MECH.det_1206mech_fx_atom(3)
    grid = MECH.det_1206mech_grid(1.0e6, 12.0, 20000.0)
    Tin = 10020.0
    Bprof, _ = MECH.det_1206mech_profile_mean(grid, at["nu"], Tin, 1.0)
    J1, tau, Jc = Bprof.mid, 1.4e7, 3.0e-8
    shells = FXT["w5_shells"]

    def hist(J1_, tau_, Jc_, atom=None, Jerr=0.0, mat1=0):
        atm = atom or at
        k_, d_, _ = MECH.det_1206mech_kappa(atm)
        h = {1: ("INIT_SHARED_GAUSSIAN", 1, mat1, None, None, None, J1_, Jerr)}
        beta = RATES.det_siiii_rates_exponx(tau_)
        J = J1_
        for c in range(2, 8):
            J = MECH.det_1206mech_F(I.x(J), I.x(beta), I.x(Jc_), atm, k_, d_)[0].mid
            h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc_, tau_, J - beta * Jc_, None, 0.0)
        return h

    def hist_dep(J1_, tau_, Jc_):
        h = {1: ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1_, 0.0)}
        beta = RATES.det_siiii_rates_exponx(tau_)
        for c in range(2, 8):
            h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc_, tau_, J1_ * 0.3 ** (c - 1) - beta * Jc_, None, 0.0)
        return h

    r_seed = MECH.det_1206mech_r_off(I.x(J1), at).mid

    def mkw5E(h=None, drop=()):
        w = {(3, s): (h or hist(J1, tau, Jc)) for s in shells}
        w.update({(0, s): hist_dep(J1 * 0.2, 300.0, Jc) for s in shells})
        for k in drop:
            del w[k]
        return w

    def mkw5R(J1R=None):
        return {**{(3, s): hist_dep(J1R or J1, tau, Jc) for s in shells}, **{(0, s): hist_dep(J1 * 0.2, 300.0, Jc) for s in shells}}

    popsE = {s: (1.0e8 * (1 + 0.01 * s), 1.0e8 * (1 + 0.01 * s) * r_seed) for s in shells}
    popsR = {s: (1.0e8 * (1 + 0.01 * s), 1.0e8 * (1 + 0.01 * s) * r_seed / 150.0) for s in shells}
    all5 = sorted(set(FXT["spatial_2542"] + shells))
    a45 = MECH.FX_LEVELS[4][1] / MECH.FX_LEVELS[5][1]

    def p5(pops, q45=0.02):
        return {s: ((pops[s][1] if s in pops else pops[19][1]), (pops[s][1] if s in pops else pops[19][1]) * q45 / a45)
                for s in all5}
    lam_grid = [2000.0 + 10.0 * i for i in range(901)]

    def spec(fn):
        return [(l_, fn(l_)) for l_ in lam_grid]
    base_f = lambda l_: 1e38 * (l_ / 5000.0) ** -1.0                     # noqa: E731
    pub = spec(base_f)
    sE = spec(lambda l_: base_f(l_) * (1.6 if l_ < 3000 else 1.3 if l_ < 4000 else 1.0))
    sR = spec(lambda l_: base_f(l_) * (0.5 if l_ < 4000 else 1.0))
    keys = [(3, s, 14, 2, "(0,1e-6]", "SURVIVES") for s in shells]

    def run(name, wE=None, wR=None, pE=None, pR=None, q45=0.02, ekw=None, rkw=None, fault=None, ck=None,
            pub_time="1.94800e+01", src_kw=None, target=None, bkw=None, sp=None, n0_ratio=1.0):
        base = T / name
        src = det_siiii_repair_make_source(base / "T2", **(src_kw or {}))
        sh = det_siiii_repair_make_shared(base / "shared", ck or keys, sp or {"public": pub, "E": sE, "R": sR}, pub_time)
        pE_, pR_ = pE or popsE, pR or popsR
        e = det_siiii_repair_make_fixture(base / "E", wE or mkw5E(), pE_, p5(pE_, q45), False, src,
                                          **{**(bkw or {}), **(ekw or {})})
        r = det_siiii_repair_make_fixture(base / "R", wR or mkw5R(), pR_, p5(pR_, q45), True, src,
                                          **{**(bkw or {}), **(rkw or {})})
        tg = dict(target) if target else dict(FXT, n_fixed=len([k for k in (ck or keys) if k[5] == "SURVIVES"]))
        rr = det_siiii_repair_analyze(e, r, sh / "prior", sh / "writer", sh / "public.txt", src, base / "out",
                                      target=tg, fault=fault, n0_ratio=n0_ratio)
        return rr, base / "out", e, r

    def csvr(o, f):
        return pd.read_csv(o / f, float_precision="round_trip")

    def anyr(rr, s):
        return any(s in x for x in rr.get("reasons", []))

    def encc(df, s, col, t):
        return enc(df.loc[s, f"{col}_lo"], df.loc[s, f"{col}_hi"], t)

    # 0 clean
    rr, o, eC, rC = run("clean")
    cand = csvr(o, "candidate_1206.csv").set_index("shell") if rr["validity"] == "VALID" else None
    w5c = pd.read_csv(eC / "work/w5_lines.csv", dtype=str)
    init_na = bool((w5c[w5c.call == "1"][["J_c", "beta", "L_l", "tau_eff"]] == "NOT_APPLICABLE").all().all())
    codes_all = {v[k] for v in rr["codes"].values() for k in ("A", "B-T")} if rr["validity"] == "VALID" else {None}
    case(SELFTEST_REQUIRED[0], (rr["validity"], None if cand is None else len(cand), init_na, rr.get("init", {}).get("I_T"),
                                rr.get("init", {}).get("I_P"), 2 in codes_all),
         ("VALID", 5, True, "SURVIVES", "SURVIVES", False), reason=str(rr.get("reasons"))[:300])
    # 1 W5 linkage
    r_mat1, *_ = run("mat1", wE=mkw5E(h=hist(J1, tau, Jc, mat1=1)))
    wsh = mkw5E()
    h = dict(wsh[(3, 19)])
    h[4] = ("CMFGEN_NONOVERLAP_SOBOLEV", 5, 5) + h[4][3:]
    wsh[(3, 19)] = h
    r_shift, *_ = run("shift", wE=wsh)
    r_swap, *_ = run("swap", ekw={"swap_shells": (19, 24)})
    r_nu, *_ = run("nu", ekw={"w5_nu_scale": 1.0 + 1e-12})
    r_dup, *_ = run("dup", ekw={"w5_contra": (3, 19, 3)})
    case(SELFTEST_REQUIRED[1], (r_mat1["validity"], anyr(r_mat1, "W5_INIT_IDENTITY"), r_shift["validity"],
                                anyr(r_shift, "W5_GENERATION_MISMATCH"), r_swap["validity"],
                                anyr(r_swap, "LINK_TAU_W5_CALL7_VS_CAPTURE"), r_nu["validity"], anyr(r_nu, "W5_IDENTITY"),
                                r_dup["validity"], anyr(r_dup, "CONTRADICTORY")),
         ("X", True, "X", True, "X", True, "X", True, "X", True),
         inj="INIT material_generation=1 / call 4 labelled 5 / shells 19,24 swapped / nu x(1+1e-12) / duplicate call 3")
    # 2 authority, source, row, count
    r_auth, *_ = run("auth", ekw={"manifest_drop": "transaction"}, rkw={"cap_fallback": 2})
    base = T / "srcmis"
    src_a = det_siiii_repair_make_source(base / "T2a")
    src_b = det_siiii_repair_make_source(base / "T2b", warm="pin")
    sh = det_siiii_repair_make_shared(base / "shared", keys, {"public": pub, "E": sE, "R": sR})
    e_ = det_siiii_repair_make_fixture(base / "E", mkw5E(), popsE, p5(popsE), False, src_a)
    r_ = det_siiii_repair_make_fixture(base / "R", mkw5R(), popsR, p5(popsR), True, src_a)
    r_src = det_siiii_repair_analyze(e_, r_, sh / "prior", sh / "writer", sh / "public.txt", src_b, base / "out", target=FXT)
    r_row, *_ = run("row", target=dict(FXT, r1206=4))
    r_cnt, *_ = run("count", target=dict(FXT, n_fixed=99))
    case(SELFTEST_REQUIRED[2], (r_auth["validity"], anyr(r_auth, "AUTHORITY_MISSING:transaction"), anyr(r_auth, "SOURCE_FALLBACK"),
                                r_src["validity"], anyr(r_src, "LINK_SOURCE_NOT_RUN_BUILD"), r_row["validity"],
                                anyr(r_row, "LINK_TARGET_ROW_1206"), r_cnt["validity"], anyr(r_cnt, "FIXED_SET_COUNT")),
         ("X", True, True, "X", True, "X", True, "X", True),
         inj="transaction dropped + fallback 2 / analysis source differs from run build / target row 4 / count 99")
    # 3 I-T
    r_sf, *_ = run("seedfix", src_kw={"seed": "fixed"})
    r_ff, *_ = run("fixedfirst", ekw={"logs": "fixed_first"})
    r_cf, *_ = run("cfg", rkw={"cfg_T": 10021.0})
    case(SELFTEST_REQUIRED[3], (r_sf["init"]["I_T"], r_sf["main_codes"]["B-T"], r_sf["main_codes"]["A"] == rr["main_codes"]["A"],
                                r_ff["init"]["I_T"], r_ff["main_codes"]["B-T"], r_cf["init"]["I_T"], rr["main_codes"]["B-T"]),
         ("REFUTED", 1, True, "REFUTED", 1, "REFUTED", 0),
         inj="seed = fixed_te[i] / FIXED_T lane before INIT / stdout effective T_inner 10021 K")
    # 4 I-P missing; origin
    r_nb, *_ = run("noboot", ekw={"logs": "no_boot"})
    r_or, *_ = run("origin", fault="claim_origin")
    case(SELFTEST_REQUIRED[4], (r_nb["init"]["I_P"], r_nb["init"]["I_T"], r_nb["main_codes"]["B-T"],
                                rr["init"]["origin_attribution"], r_or["init"]["origin_attribution"] != "ORIGIN_NOT_SEPARATED"),
         ("NOT_IDENTIFIABLE", "NOT_IDENTIFIABLE", 5, "ORIGIN_NOT_SEPARATED", True), inj="stdout without bootstrap / claim_origin")
    # 5 I-S
    r_pl, *_ = run("planck", bkw={"A_scale": 1.01})
    io = json.loads((o / "init_origin.json").read_text())
    tS = det_siiii_repair_truth_Sboot(at, Tin)
    case(SELFTEST_REQUIRED[5], (enc(io["S_boot"]["lo"], io["S_boot"]["hi"], tS), rr["init"]["I_S"], r_pl["init"]["I_S"],
                                r_pl["main_codes"]["B-T"]), (True, "SURVIVES", "REFUTED", 1), inj="A_ul x 1.01 on the target row")
    # 6 warm start only
    r_w, *_ = run("warmonly", src_kw={"warm": "pin"})
    r_wf, *_ = run("warmfault", src_kw={"warm": "pin"}, fault="warmstart_as_material")
    case(SELFTEST_REQUIRED[6], (r_w["init"]["I_T"], r_wf["init"]["I_T"]), ("SURVIVES", "REFUTED"),
         inj="warm start changed to the pinned profile; warmstart_as_material fault")
    # 7 exact map, kappa != 1, Delta B != 0
    at2 = dict(at, B_ul=at["B_ul"] * 1.01)
    w2E = mkw5E(h=hist(J1, tau, Jc, atom=at2))
    r_db, o_db, e_db, _ = run("dB", wE=w2E, bkw={"bul_scale": 1.01})
    c_db = csvr(o_db, "candidate_1206.csv").set_index("shell")
    hE_db, _ = det_siiii_repair_w5_load(e_db, {(3, 19)})
    T19 = 8000.0 + 50.0 * 19
    tB = MECH.det_1206mech_truth_profile(grid, at2["nu"], T19, 1.0)
    tJ_db = det_siiii_repair_truth_map(tB, hE_db[(3, 19)], at2)
    r_dk, o_dk, *_ = run("dropk", wE=w2E, bkw={"bul_scale": 1.01}, fault="drop_kappa")
    r_bo, o_bo, *_ = run("betaonly", wE=w2E, bkw={"bul_scale": 1.01}, fault="beta_only")
    c_dk = csvr(o_dk, "candidate_1206.csv").set_index("shell")
    c_bo = csvr(o_bo, "candidate_1206.csv").set_index("shell")
    case(SELFTEST_REQUIRED[7], (r_db["validity"], encc(c_db, 19, "BT_J1", tB), encc(c_db, 19, "BT_J7_Ebg", tJ_db[7]),
                                encc(c_dk, 19, "BT_J7_Ebg", tJ_db[7]), encc(c_bo, 19, "BT_J7_Ebg", tJ_db[7])),
         ("VALID", True, True, False, False), inj="B_ul x 1.01; drop_kappa / beta_only faults")
    # 8 log10 and T_exc against independent 80-digit references (dB fixture)
    rE_t = Fraction(popsE[19][1]) / Fraction(popsE[19][0])
    rR_t = Fraction(popsR[19][1]) / Fraction(popsR[19][0])
    r7_t = det_siiii_repair_truth_roff(tJ_db[6], at2)
    t_seed = det_siiii_repair_truth_log10(FI(tB.lo / Fraction(J1), tB.hi / Fraction(J1)))
    t_r7 = det_siiii_repair_truth_log10(FI(r7_t.lo / rE_t, r7_t.hi / rE_t))
    t_A = det_siiii_repair_truth_log10(FI(rR_t / rE_t))
    t_Tx = det_siiii_repair_truth_Texc(at2, r7_t)
    t_TA = det_siiii_repair_truth_Texc(at2, FI(rR_t))
    case(SELFTEST_REQUIRED[8], (encc(c_db, 19, "BT_log10_seed_over_existing", t_seed), encc(c_db, 19, "BT_log10_r7_over_E", t_r7),
                                encc(c_db, 19, "A_log10_r7_over_E", t_A), encc(c_db, 19, "BT_Texc", t_Tx),
                                encc(c_db, 19, "A_Texc", t_TA), encc(c_dk, 19, "BT_log10_r7_over_E", t_r7),
                                encc(c_dk, 19, "BT_Texc", t_Tx)),
         (True, True, True, True, True, False, False), inj="drop_kappa fault",
         reason="truths: exact iterated map, exact r_off, captured n4/n0 of the fixture floats, ln at 80 digits")
    # 9 memory difference, armed seed independence, unseparated denominator
    MWT = I(cand.loc[19, "M7_WT_lo"], cand.loc[19, "M7_WT_mid"], cand.loc[19, "M7_WT_hi"])
    r_rs, o_rs, *_ = run("rseed", wR=mkw5R(J1R=J1 * 0.5))
    c_rs = csvr(o_rs, "candidate_1206.csv").set_index("shell")
    W19 = det_siiii_repair_geometry(eC)[19]
    target_B = W19.mid * Bprof.mid
    lo_T, hi_T = 3000.0, 10020.0
    for _ in range(200):
        mid_T = 0.5 * (lo_T + hi_T)
        if MECH.det_1206mech_profile_mean(grid, at["nu"], mid_T, 1.0)[0].mid > target_B:
            hi_T = mid_T
        else:
            lo_T = mid_T
    prof = {s: 8000.0 + 50.0 * s for s in range(50)}
    prof[19] = 0.5 * (lo_T + hi_T)
    r_eq, o_eq, *_ = run("wtsame", bkw={"te_prof": prof})
    c_eq = csvr(o_eq, "candidate_1206.csv").set_index("shell")
    case(SELFTEST_REQUIRED[9], (MWT.lo >= 1 - 1e-3 and MWT.hi <= 1.0 + 1e-3,
                                c_rs.loc[19, "A_r_A_coll_call6_mid"] == cand.loc[19, "A_r_A_coll_call6_mid"],
                                c_eq.loc[19, "M7_WT_status"]), (True, True, "UNSEPARATED_DENOMINATOR"),
         inj="R' call-1 seed x0.5 / T_pin(19) solved so that B(T_pin) = W B(T_inner)")
    # 10 beta with collisions
    hR0, _ = det_siiii_repair_w5_load(rC, {(3, 19)})
    r6 = hR0[(3, 19)][6]
    bR, jcR = _fr(float(r6["beta"])), _fr(float(r6["J_c"]))
    Cu, Cd = RATES.det_siiii_rates_coll(3.0e9, 9000.0, at["E_l"], at["E_u"], at["g_l"], at["g_u"], at["f_lu"])
    tAC = [(bR * _fr(at["B_lu"]) * jcR + _fr(cu)) / (bR * (_fr(at["A_ul"]) + _fr(at["B_ul"]) * jcR) + _fr(cd))
           for cu, cd in ((Cu.lo, Cd.hi), (Cu.hi, Cd.lo))]
    r_bc, o_bc, *_ = run("betacancel", fault="beta_cancel_with_coll")
    c_bc = csvr(o_bc, "candidate_1206.csv").set_index("shell")
    case(SELFTEST_REQUIRED[10], (Fraction(cand.loc[19, "A_r_A_coll_call6_lo"]) <= min(tAC) and
                                 max(tAC) <= Fraction(cand.loc[19, "A_r_A_coll_call6_hi"]),
                                 Fraction(c_bc.loc[19, "A_r_A_coll_call6_lo"]) <= min(tAC) and
                                 max(tAC) <= Fraction(c_bc.loc[19, "A_r_A_coll_call6_hi"])),
         (True, False), inj="beta_cancel_with_coll fault")
    # 11 q >= 1
    at_h = dict(at, B_ul=at["B_ul"] * 0.5)
    B30 = MECH.det_1206mech_profile_mean(grid, at["nu"], 30000.0, 1.0)[0].mid
    r_q, *_ = run("qge1", wE=mkw5E(h=hist(B30, tau, Jc, atom=at_h)), bkw={"bul_scale": 0.5, "T_inner": 30000.0})
    Dq = r_q.get("D_local_q_required_iterations") or {}
    Dc = rr["D_local_q_required_iterations"]
    case(SELFTEST_REQUIRED[11], (r_q["validity"], (Dq.get("local_q_at_existing_call1") or {}).get("lo", 0) > 1.0,
                                 Dq.get("required_iterations_to_1e-3"), Dq.get("flag"), Dc["flag"]),
         ("VALID", True, None, "UNDEFINED_Q_NOT_IN_(0,1)", "OK"), inj="T_inner 30000 K, B_ul x 0.5 -> q > 1")
    # 12 zero predicted change
    prof0 = {s: 8000.0 + 50.0 * s for s in range(50)}
    prof0[19] = Tin
    r_z, o_z, *_ = run("zero", bkw={"te_prof": prof0})
    c_z = csvr(o_z, "candidate_1206.csv").set_index("shell")
    ex_z = json.loads((o_z / "expectations.json").read_text())
    case(SELFTEST_REQUIRED[12], (c_z.loc[19, "BT_seed_direction"], bool(c_z.loc[19, "BT_log10_seed_over_existing_lo"] <= 0.0
                                                                        <= c_z.loc[19, "BT_log10_seed_over_existing_hi"]),
                                 ex_z["B-T"]["primary_expectation"]["seed_direction"]["19"], cand.loc[19, "BT_seed_direction"]),
         ("ZERO_NOT_SEPARATED", True, "ZERO_NOT_SEPARATED", "DOWN"), inj="T_pin(19) = T_inner")
    # 13 2542 exact references on the clean fixture
    resp = csvr(o, "response_2542.csv").set_index("shell")
    x = resp.loc[19]
    hE_c, _ = det_siiii_repair_w5_load(eC, {(3, 19)})
    tBc = MECH.det_1206mech_truth_profile(grid, at["nu"], T19, 1.0)
    tJc = det_siiii_repair_truth_map(tBc, hE_c[(3, 19)], at)
    r7c = det_siiii_repair_truth_roff(tJc[6], at)
    t_alpha = FI(r7c.lo / rE_t, r7c.hi / rE_t)
    n4t, n5t = p5(popsE)[19]
    q_t = Fraction(3, 5) * Fraction(n5t) / Fraction(n4t)
    tau_t = Fraction(float(x.E_tau45))
    tP1 = det_siiii_repair_truth_P(t_alpha, q_t, tau_t, "P1")
    tP2 = det_siiii_repair_truth_P(t_alpha, q_t, tau_t, "P2")
    case(SELFTEST_REQUIRED[13], (Fraction(x.E_n4_lo) <= Fraction(n4t) <= Fraction(x.E_n4_hi),
                                 Fraction(x.E_n5_lo) <= Fraction(n5t) <= Fraction(x.E_n5_hi), x.E_status,
                                 encc(resp, 19, "BT_alpha4", t_alpha), encc(resp, 19, "P1_Q_ratio", tP1),
                                 encc(resp, 19, "P2_Q_ratio", tP2), float(x.P1_Q_ratio_hi) < 1.0, float(x.P2_Q_ratio_lo) > 1.0),
         (True, True, "OK", True, True, True, True, True), reason=f"2542 tau {x.E_tau45}; q45 = 3/5 n5/n4 exact")
    # 14 n0'/n0 explicit; sign change
    r_n0, o_n0, *_ = run("n0half", n0_ratio=0.5)
    r_no, o_no, *_ = run("n0omit", n0_ratio=0.5, fault="omit_n0_ratio")
    t_half = FI(t_alpha.lo / 2, t_alpha.hi / 2)
    x_n0 = csvr(o_n0, "response_2542.csv").set_index("shell")
    x_no = csvr(o_no, "response_2542.csv").set_index("shell")
    r_sg, o_sg, *_ = run("sign", q45=0.9, bkw={"te_prof": {s: 6000.0 for s in range(50)}})
    x_sg = csvr(o_sg, "response_2542.csv").set_index("shell").loc[19]
    case(SELFTEST_REQUIRED[14], (encc(x_n0, 19, "BT_alpha4", t_half), float(x_n0.loc[19, "BT_n0_ratio_assumed"]),
                                 encc(x_no, 19, "BT_alpha4", t_half), x_sg.P2_flag, bool(pd.isna(x_sg.P2_Q_ratio_mid)),
                                 bool(float(x_sg.P2_tau_ratio_hi) < 0.0), bool(pd.isna(x_sg.P2_S_ratio_mid)), x_sg.P1_flag),
         (True, 0.5, False, "SIGN_CHANGE", True, True, True, "OK"),
         inj="n0'/n0 = 0.5 / omit_n0_ratio fault / q45 = 0.9 with T_pin 6000 K")
    # 15 spectrum through analyze: OPT-only, band gain with worse R8, out of range
    sOPT = spec(lambda l_: base_f(l_) * (2.0 if 4000 <= l_ <= 6000 else 1.0))
    sP = spec(lambda l_: base_f(l_) * (1.1 if l_ < 3000 else 3.0 if l_ > 7000 else 1.0))
    sOut = [(l_, f_) for l_, f_ in pub if l_ >= 2600.0]
    r_so, o_so, *_ = run("specopt", sp={"public": pub, "E": sE, "R": sOPT})
    r_sp, *_ = run("specgain", sp={"public": pub, "E": sE, "R": sP})
    r_sx, o_sx, *_ = run("specout", sp={"public": pub, "E": sE, "R": sOut})
    so = csvr(o_so, "spectrum_reference.csv")
    so = so[so.spectrum == "RprimeN"].set_index("band")
    moved = all(abs(float(so.loc[b, "d_dex_mid"]) + math.log10(2.0)) < 1e-9 for b in
                ("2500-3000", "3000-3500", "7000-8000", "8000-10000"))
    sx = csvr(o_sx, "spectrum_reference.csv")
    sx = sx[sx.spectrum == "RprimeN"].set_index("band")
    sR8 = r_sp["spectrum_summary"]["RprimeN"]["R8"]
    eR8 = r_sp["spectrum_summary"]["EprimeN"]["R8"]
    case(SELFTEST_REQUIRED[15], (moved, abs(float(so.loc["2500-3000", "L_band_over_public"]) - 1.0) < 1e-12,
                                 round(r_so["spectrum_summary"]["RprimeN"]["L_OPT_over_public"], 9),
                                 r_sp["spectrum_summary"]["RprimeN"]["target_status"]["UV2500"], sR8["lo"] > eR8["hi"],
                                 r_sp["spectrum_summary"]["RprimeN"]["target_status"]["R8"],
                                 r_sp["spectrum_summary"]["RprimeN"]["meets_secondary_targets"],
                                 sx.loc["2500-3000", "status"], r_sx["spectrum_summary"]["RprimeN"]["d_UV2500"],
                                 r_sx["spectrum_summary"]["RprimeN"]["meets_secondary_targets"]),
         (True, True, 2.0, "SURVIVES", True, "REFUTED", "REFUTED", "UNSUPPORTED", None, "NOT_IDENTIFIABLE"),
         inj="writer R'N spectra: OPT x2 / UV x1.1 + red x3 / starts at 2600 A")
    # 16 boundary, public time, identity
    sE4 = spec(lambda l_: base_f(l_) * (4.0 if 2500 <= l_ <= 3000 else 1.0))
    sY2 = spec(lambda l_: base_f(l_) * (2.0 if 2500 <= l_ <= 3000 else 1.0))
    r_bd, *_ = run("specbound", sp={"public": pub, "E": sE4, "R": sY2})
    r_pt, *_ = run("pubtime", pub_time="1.94900e+01")
    case(SELFTEST_REQUIRED[16], (r_bd["spectrum_summary"]["RprimeN"]["target_status"]["UV2500"],
                                 r_bd["spectrum_summary"]["RprimeN"]["meets_secondary_targets"] != "SURVIVES",
                                 r_pt["validity"], r_pt["spectrum_summary"].get("status"),
                                 rr["spectrum_identity"]["capture_E_vs_EprimeN"]["byte_identical"]),
         ("UNRESOLVED", True, "VALID", "PUBLIC_NOT_AVAILABLE", False),
         inj="E'N UV2500 x4 and R'N x2 (|d| = half-target up to rounding) / public times 17.71, 19.49 d")
    # 17 extension
    ck = keys + [(0, 30, 14, 2, "(1e-4,1e-2]", "SURVIVES"), (0, 24, 14, 2, "(1e-4,1e-2]", "SURVIVES")]
    r_ex, o_ex, *_ = run("ext", wE=mkw5E(drop=[(0, 24)]), ck=ck)
    e_ex = csvr(o_ex, "extension.csv").set_index(["line", "shell"])
    case(SELFTEST_REQUIRED[17], (len(e_ex), e_ex.loc[(0, 30), "initial_source_status"], e_ex.loc[(0, 24), "initial_source_status"],
                                 r_ex["extension"]["all"]["universal_initial_source"], r_ex["extension"]["all"]["subset_count_bounds"]),
         (7, "REFUTED", "NOT_IDENTIFIABLE", "REFUTED", [5, 6]), inj="key (0,30) seed 0.2 B(1206); key (0,24) W5 history removed")
    # 18 codes
    pE3 = {s: (v[0], v[1] * 1.01) for s, v in popsE.items()}
    r_c3, *_ = run("code3", pE=pE3)
    r_c4, *_ = run("code4", wE=mkw5E(h=hist(J1, tau, Jc, Jerr=J1 * 2e-3)))
    got = sorted({r_["main_codes"]["B-T"] for r_ in (rr, r_sf, r_c3, r_c4, r_nb)})
    allc = set()
    for r_ in (rr, r_sf, r_c3, r_c4, r_nb, r_pl, r_w, r_db, r_rs, r_eq, r_q, r_z):
        if r_.get("validity") == "VALID":
            allc |= {v[k] for v in r_["codes"].values() for k in ("A", "B-T")}
    ex0 = json.loads((o / "expectations.json").read_text())
    case(SELFTEST_REQUIRED[18], (got, 2 in allc, ex0["B-T"]["primary_expectation"]["status"],
                                 ex0["B-T"]["secondary_spectral_expectation"]["status"],
                                 ex0["B-T"]["readiness"]["seed_delivery_verified"], ex0["B-T"]["readiness"]["future_run_authorized"]),
         ([0, 1, 3, 4, 5], False, "FUTURE_TEST", "FUTURE_TEST", False, False),
         inj="clean / seed fixed / capture n4 x1.01 / INIT Jbar error 2e-3 / no bootstrap")
    names = [r["case"] for r in res]
    missing = [n for n in SELFTEST_REQUIRED if n not in names]
    fails = sum(1 for r in res if not r["pass"]) + len(missing)
    summary = {"schema": "DET_SIIII_REPAIR_SELFTEST_V2", "readout_sha256": sha(Path(__file__).resolve()),
               "dependencies": {str(Path(d).resolve()): sha(Path(d)) for d in (POP.__file__, RATES.__file__, MECH.__file__,
                                                                               NTE.__file__)},
               "production_source_arg": str(a.production_source), "cases": res, "missing_required": missing,
               "n_cases": len(res), "failures": fails,
               "scope_note": "missing_required checks only that every listed case ran; coverage of prereg 8.2 is the case list",
               "truth": "exact rational arithmetic on the fixture doubles; exp at 60 digits bracketed 1e-58; ln at 80 digits "
                        "bracketed 1e-70; iterated map endpoints rounded outward to 70 digits; interval containment"}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1))
    print(f"selftest: {len(res) - sum(1 for r in res if not r['pass'])}/{len(res)} pass, missing {len(missing)}, failures {fails}")
    return 0 if fails == 0 else 1


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--production-source", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("analyze")
    for k in ("--e-run", "--r-run", "--prior-mechanism", "--writer-root", "--public-spectrum", "--production-source",
              "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit(cmd_selftest(a) if a.cmd == "selftest" else cmd_analyze(a))


if __name__ == "__main__":
    main()
