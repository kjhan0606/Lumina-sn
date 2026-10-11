#!/usr/bin/env python3
"""DET-LINE-NET-PROVENANCE (prereg docs/PREREG_DET_LINE_NET_PROVENANCE_2026-10-11.md).

Definition and generation trace of the fixed-T line net-rate ledger L = V (line_abs_heat - line_emit_cool) of the
reference configuration O, against the same dump's binned transport Q_bb:
  - all 100 shell rows (A10: material 13 / radiation 12, A11: material 14 / radiation 13): L, A_bb, E_bb, Q_bb,
    L - Q_bb, the line set, population ownership, e / k / d / Jbar definitions with their generations, the W5 support,
    the producing call, the line-set registration of the logs and the registered sameness conditions;
  - the W5 check (calls of radiation 12 / 13, the five stored shells): Jbar = beta J_c + l and beta = beta_N(tau_eff);
  - the registered six representative lines of A10 shell 10: the ordered arithmetic decomposition F0 .. F4 at fixed
    J_c^p, and D_rest = (L - Q_bb) - sum (F0 - F4);
  - two global rows (per state).

Subcommands
  selftest                                  negative controls of the new code through toy files (prereg 9)
  analyze --out DIR [--selftest-log PATH]   G-INPUT, G-CALC, G-RUN and the main table
Exit codes: 0 pass, 2 refused or gate finding, 3 existing output path.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_line_energy_corr as LEC                                                # noqa: E402
import det_si_line_input as LI                                                    # noqa: E402
import det_si_trace_readout as TR                                                 # noqa: E402
import det_si_excitation_budget as EB                                             # noqa: E402

PREREG = "docs/PREREG_DET_LINE_NET_PROVENANCE_2026-10-11.md"
H_PLANCK, C_LIGHT = 6.62607015e-27, 2.99792458e10
EV_TO_ERG = 1.602176634e-12
FOUR_PI = LEC.FOUR_PI
TOL = LEC.TOL
DELTA = LEC.DELTA
BB_NU = (1.49896229e14, 2.99792458e16)            # line_jbar.h BB domain
SCL_LN_FAC, DENSITY_LIMIT = 0.5, 1.0e30           # lumina_plasma.c call of line_net_cmfgen_scl_ln
SOBOLEV_OP = "CMFGEN_NONOVERLAP_SOBOLEV"
REAL = {"O": Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1"),
        "T": Path("/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1"),
        "shell": 10, "ns": LEC.NS, "nb": LEC.NB, "lines": LI.LINES, "w5_shells": (10, 15, 19, 24, 30),
        "adopted": EB.ADOPTED,
        "states": {"A10": {"it": 10, "material": 13, "radiation": 12},
                   "A11": {"it": 11, "material": 14, "radiation": 13}}}
SOURCE = {   # driver's reading of the sealed sources (prereg 3-5); data for the table, not a string-match verdict
    "line_set": "energy set = BB-domain lines regardless of mapping (line_jbar_eset_build); summed only strictly "
                "inside the publication frequency bounds; inactive elements are explicit zero contributions "
                "(lumina_plasma.c ledger loop)",
    "population": "a209_upper_population_for_tau: NLTE level population where ownership and published generation "
                  "match, otherwise LTE level population of the same material (same ownership as the opacity)",
    "e": "e = n_u A_ul h nu / (4 pi) of the ledger material m",
    "k": "k_raw = tau_raw nu / (c t) of the ledger material m; CMFGEN_SRCE_CHK: tau_raw < -0.5 -> k_eff = 1e-10",
    "d": "deck_scale = (E_u - E_l)_super-average eV / (h nu), 1 if n_atom >= 1e30 or |r - 1| > 0.5",
    "Jbar": "Jbar^p = beta_N(tau_eff^p) J_c^p + e^p (c t / nu) C_N(tau_eff^p), J_c = Gaussian (|x| <= 4) mean of the "
            "fine field (line_jbar.c), registered before the material update (producer material p)",
    "transport": "chi_bb dnu = sum nu (1 - exp(-tau_raw)) / (c t), eta_bb dnu = sum e beta_T(tau_raw); raw tau, "
                 "no SRCE_CHK"}


# ------------------------------------------------------------------ small arithmetic (undefined-preserving)
def D(f, *deps, scaling=False):
    return LEC.lec_derived(f, *deps, scaling=scaling)


def lnp_beta_T(tau):
    """transport escape factor: 1 at tau = 0, -expm1(-tau) / tau otherwise (finite limit, no division at 0)."""
    if tau is None or not math.isfinite(tau):
        return None
    if tau == 0.0:
        return 1.0
    return D(lambda t: -math.expm1(-t) / t, tau)


def lnp_F(four_pi_V, w, a, b, c):
    """F = 4 pi V w (a b - c) with the pre-cancellation scale S = 4 pi V |w| (|a b| + |c|); None when undefined."""
    P = D(lambda x, y: x * y, a, b, scaling=True)
    diff = D(lambda x, y: x - y, P, c)
    F = D(lambda x, y, z: x * y * z, four_pi_V, w, diff, scaling=True)
    S = D(lambda x, y, p, q: x * abs(y) * (abs(p) + abs(q)), four_pi_V, w, P, c)
    return F, S


def lnp_d(E_u, E_l, nu, n_atom):
    """line_net_cmfgen_scl_ln: ratio and deck scale (1 when the density limit or the |r-1| > 0.5 rule applies)."""
    r = D(lambda a, b, n: (a - b) * EV_TO_ERG / (H_PLANCK * n), E_u, E_l, nu)
    if r is None or n_atom is None or not math.isfinite(n_atom):
        return r, None, None
    if n_atom >= DENSITY_LIMIT:
        return r, 1.0, "density_limit"
    if abs(r - 1.0) > SCL_LN_FAC:
        return r, 1.0, "|r-1|>0.5"
    return r, r, "ratio"


# ------------------------------------------------------------------ readers
def lnp_hash_lines(path, keep):
    """stream a large text file once: sha256 of the bytes and the decoded lines accepted by keep(bytes line)."""
    h, out, n = hashlib.sha256(), [], 0
    with open(path, "rb") as f:
        for ln in f:
            h.update(ln)
            n += len(ln)
            if keep(ln):
                out.append(ln.decode().rstrip("\r\n"))
    return h.hexdigest(), n, out


def lnp_scan_w5(path, cfg, rad_gens):
    """W5 production ledger: (a) the Sobolev rows of the calls of the registered radiation generations (relation
    check per call and shell), (b) the rows of the registered lines at the registered shell, all operators."""
    line_ids = {str(lid) for _, lid, _, _ in cfg["lines"]}
    w5_shells = {str(s) for s in cfg["w5_shells"]}

    def keep(ln):
        return b"," + SOBOLEV_OP.encode() + b"," in ln[:40] or any(
            b"," + i.encode() + b"," in ln for i in line_ids)
    sha, nbytes, lines = lnp_hash_lines(path, keep)
    head = ["call", "operator", "rad_generation", "material_generation", "shell", "line_id", "Z", "ion", "nu_hz",
            "J_c", "J_c_err_upper", "beta", "L_l", "Jbar", "Jbar_err_upper", "tau_eff", "srce_chk", "status"]
    check, line_rows, gens, notes = {}, {}, {}, []
    for ln in lines:
        t = ln.split(",")
        if len(t) != len(head) or t[0] == "call":
            if t[0] != "call":
                notes.append("W5_ROW_WIDTH")
            continue
        r = dict(zip(head, t))
        if r["line_id"] in line_ids and r["shell"] == str(cfg["shell"]):
            line_rows.setdefault(r["line_id"], []).append(r)
        if r["operator"] != SOBOLEV_OP:
            continue
        gens.setdefault(r["call"], set()).add((r["rad_generation"], r["material_generation"]))
        if r["rad_generation"] not in rad_gens:
            continue
        c = check.setdefault(r["call"], {"used": 0, "excluded": Counter(), "max_jbar_residual": 0.0,
                                         "max_beta_residual": 0.0, "by_shell": {}})
        bs = c["by_shell"].setdefault(r["shell"], {"used": 0, "excluded": Counter(), "max_jbar_residual": 0.0,
                                                   "max_beta_residual": 0.0})
        if r["shell"] not in w5_shells:
            c["excluded"]["OUTSIDE_SUPPORT"] += 1
            bs["excluded"]["OUTSIDE_SUPPORT"] += 1
            continue
        if r["status"] != "OK":
            c["excluded"][r["status"]] += 1
            bs["excluded"][r["status"]] += 1
            continue
        why = []
        J_c, beta, l, Jbar, tau = (LEC.lec_num(r[k], why, k) for k in ("J_c", "beta", "L_l", "Jbar", "tau_eff"))
        e = LI.det_si_line_input_exponx(tau) if not why else None
        if why or e is None:
            c["excluded"]["UNDEFINED_VALUE"] += 1
            bs["excluded"]["UNDEFINED_VALUE"] += 1
            continue
        c1 = D(lambda b, j: b * j, beta, J_c)
        sc = D(lambda x, y: abs(x) + abs(y), c1, l)
        rj = D(lambda a, b, s: abs(a - b) / s if s > 0 else (0.0 if a == b else float("inf")),
               Jbar, D(lambda x, y: x + y, c1, l), sc)
        rb = D(lambda a, b: abs(a - b) / abs(a) if a != 0 else (0.0 if b == 0 else float("inf")), beta, e[0])
        if rj is None or rb is None:
            c["excluded"]["RESIDUAL_UNDEFINED"] += 1
            bs["excluded"]["RESIDUAL_UNDEFINED"] += 1
            continue
        for x in (c, bs):
            x["used"] += 1
            x["max_jbar_residual"] = max(x["max_jbar_residual"], rj)
            x["max_beta_residual"] = max(x["max_beta_residual"], rb)
    return {"sha256": sha, "bytes": nbytes, "check": check, "line_rows": line_rows,
            "generations": {k: sorted(v) for k, v in gens.items()}, "notes": notes}


def lnp_registration(log_path, gens):
    """[R6][LINE-IDENTITY] / [LINE-COVERAGE] rows of the registered radiation generations (DET lane)."""
    out = {g: {} for g in gens}
    with open(log_path, errors="replace") as f:
        for ln in f:
            if "[R6][LINE-IDENTITY]" in ln and "lane=DET" in ln or "[R6][LINE-COVERAGE]" in ln:
                kv = dict(x.split("=", 1) for x in ln.split() if "=" in x)
                g = kv.get("generation")
                if g in out:
                    kind = "identity" if "LINE-IDENTITY" in ln else "coverage"
                    out[g].setdefault(kind, []).append(kv)
    return out


def lnp_super_average(levels):
    """statistical-weight average energy of each superlevel (lumina_plasma.c super average), keyed by level."""
    acc = {}
    for r in levels:
        acc.setdefault(r["super_level"], [0.0, 0.0])
        acc[r["super_level"]][0] += float(r["g"]) * float(r["energy_eV"])
        acc[r["super_level"]][1] += float(r["g"])
    return {int(r["level_number"]): (acc[r["super_level"]][0] / acc[r["super_level"]][1]
                                     if acc[r["super_level"]][1] > 0 else None) for r in levels}


def lnp_trace(cfg, why):
    """populations of material 12 (call 1 PRE_FL) and material 13 (the adopted call's commit, kind=level, equal level
    by level to the adopted CE pass 'after'; prereg 6, revision 1) with their links.  FINAL_FL is recorded only."""
    tr = cfg["T"] / "si_trace"
    calls = TR.read_trace(tr / "si10_calls.csv")
    cevec = TR.read_trace(tr / "si10_cevec.csv")
    commit_rows = TR.read_trace(tr / "si10_commit.csv")
    call, w = EB.det_si_exc_adopt(TR.read_trace(tr / "si10_ne.csv"), calls, cevec, commit_rows, cfg["adopted"])
    if call is None:
        why += [f"TRACE_ADOPTION:{x}" for x in w]
        return None
    cid = call["call"]
    pcall = LI.det_si_line_input_production_call(calls)
    if pcall is None:
        why.append("TRACE_PRODUCTION_CALL")
        return None
    a10 = cfg["states"]["A10"]
    if int(call["rad_generation"]) != a10["radiation"] or int(pcall["core_generation"]) != a10["material"]:
        why.append(f"TRACE_GENERATIONS:{call['rad_generation']}:{pcall['core_generation']}")
        return None
    vec = {}
    for r in TR.read_trace(tr / "si10_vec.csv"):
        if (r["call"], r["phase"]) in (("1", "PRE_FL"), (cid, "FINAL_FL")):
            vec.setdefault((r["call"], r["phase"]), {})[int(r["i"])] = float(r["value"])
    sm = TR.read_trace(tr / "si10_slmap.csv")
    sm1 = sorted((r for r in sm if r["call"] == "1"), key=lambda r: int(r["i_fl"]))
    smA = sorted((r for r in sm if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    N = int(call["N_fl"])
    key = ("fl_index", "level_num", "upper_ion", "g")
    if len(sm1) != N or [int(r["i_fl"]) for r in sm1] != list(range(N)) or \
            [tuple(r[k] for k in key) for r in sm1] != [tuple(r[k] for k in key) for r in smA]:
        why.append("TRACE_LEVEL_MAP_CALL1_VS_ADOPTED")
        return None
    pre = [vec.get(("1", "PRE_FL"), {}).get(i) for i in range(N)]
    if any(v is None or not math.isfinite(v) for v in pre):
        why.append("TRACE_PRE_FL_VECTOR")
        return None
    lev = [r for r in commit_rows if r["kind"] == "level"]
    if sorted(int(r["i"]) for r in lev) != list(range(N)):
        why.append("TRACE_COMMIT_INDEX")
        return None
    keys = {k: str(cfg["adopted"][k]) for k in ("bundle", "ne_iter", "ce_pass")}
    ce = [r for r in cevec if all(r[k] == v for k, v in keys.items())]
    if sorted(int(r["i_fl"]) for r in ce) != list(range(N)):
        why.append("TRACE_CE_AFTER_INDEX")
        return None
    commit = {int(r["i"]): float(r["value"]) for r in lev}
    after = {int(r["i_fl"]): float(r["after"]) for r in ce}
    if any(commit[i] != after[i] for i in range(N)) or not all(math.isfinite(commit[i]) for i in range(N)):
        why.append("TRACE_COMMIT_VS_CE_AFTER")
        return None
    mat13 = [commit[i] for i in range(N)]
    final_fl = [vec.get((cid, "FINAL_FL"), {}).get(i) for i in range(N)]
    lmap = EB.det_si_exc_level_map(sm1, {x for _, _, lo, up in cfg["lines"] for x in (lo, up)})
    if lmap is None:
        why.append("TRACE_LEVEL_MAP")
        return None
    upper = [r["upper_ion"] == "1" for r in sm1]

    def sums(v):
        if any(x is None for x in v):
            return None
        return (sum(x for x, u in zip(v, upper) if not u), sum(x for x, u in zip(v, upper) if u))
    return {"call": call, "pcall": pcall, "pre": pre, "mat13": mat13, "final_fl": final_fl, "lmap": lmap,
            "sums": {"pre": sums(pre), "mat13": sums(mat13), "final_fl": sums(final_fl)},
            "g": {int(r["level_num"]): int(r["g"]) for r in sm1 if r["upper_ion"] == "0"}}


def lnp_ion_row(path, shell):
    rows = [r for r in TR.read_trace(path) if int(r["shell_id"]) == shell]
    return rows[0] if len(rows) == 1 else None


def lnp_atomic(path, line_ids):
    """registered line rows of Si II (atomic 14, ion 1) from the full line list, streamed once."""
    pref = b"14,1,"
    sha, nbytes, lines = lnp_hash_lines(path, lambda ln: ln.startswith(pref) or ln.startswith(b"atomic_number"))
    head = lines[0].split(",") if lines and lines[0].startswith("atomic_number") else None
    out = {}
    for ln in lines[1:] if head else []:
        r = dict(zip(head, ln.split(",")))
        if r.get("line_id") and int(r["line_id"]) in line_ids:
            out.setdefault(int(r["line_id"]), []).append(r)
    return sha, nbytes, out, head


# ------------------------------------------------------------------ the six-line decomposition
def lnp_line(lid, lo, up, atom, levels, sup, tr, w5row, cfg, V, n_atom, J_b, t):
    """one registered line of A10 at the registered shell: material m = adopted commit (13), producer p = call-1
    PRE_FL (12) and the W5 call of radiation 12; F0 .. F4 as registered (prereg 6)."""
    row = {"row_kind": "제한 선", "state": "A10", "shell": cfg["shell"], "scope": "제한 선 (등록 6선 중 하나)",
           "line_id": lid, "lower_level": lo, "upper_level": up}
    why = []
    if len(atom) != 1:
        why.append(f"ATOMIC_ROWS:{len(atom)}")
    if len(w5row) != 1:
        why.append(f"W5_ROWS:{len(w5row)}")
    if why:
        row.update(status="미산출", reason=";".join(why))
        return row
    a, w = atom[0], w5row[0]
    if (int(a["level_number_lower"]), int(a["level_number_upper"])) != (lo, up):
        why.append("ATOMIC_LEVELS")
    if (w["operator"], w["rad_generation"], w["material_generation"]) != (SOBOLEV_OP, "12", "12") or \
            w["status"] != "OK" or w["Z"] != "14" or w["ion"] != "1":
        why.append(f"W5_LINK:{w['operator']}:{w['rad_generation']}:{w['material_generation']}:{w['status']}")
    if lo not in tr["lmap"] or up not in tr["lmap"] or lo not in levels or up not in levels:
        why.append("LEVEL_LINK")
    if why:
        row.update(status="미산출", reason=";".join(why))
        return row
    vals = []
    f_lu, lam, nu, A = (LEC.lec_num(a[k], vals, k) for k in ("f_lu", "wavelength_cm", "nu", "A_ul"))
    if LEC.lec_num(w["nu_hz"], vals, "nu_hz") != nu:
        why.append("W5_FREQUENCY")
    J_c, beta_p, l_p, Jbar_p, taueff_p = (LEC.lec_num(w[k], vals, k)
                                          for k in ("J_c", "beta", "L_l", "Jbar", "tau_eff"))
    g_l, g_u = int(levels[lo]["g"]), int(levels[up]["g"])
    if (g_l, g_u) != (tr["g"].get(lo), tr["g"].get(up)):
        why.append("STATISTICAL_WEIGHTS")
    if not (BB_NU[0] <= nu <= BB_NU[1]):
        why.append("OUTSIDE_BB_DOMAIN")
    if vals:
        why += [f"VALUE:{v}" for v in vals]
    if why:
        row.update(status="미산출", reason=";".join(why))
        return row
    il, iu = tr["lmap"][lo], tr["lmap"][up]
    nl_m, nu_m, nl_p, nu_p = tr["mat13"][il], tr["mat13"][iu], tr["pre"][il], tr["pre"][iu]
    tau_m = LI.det_si_line_input_tau(f_lu, lam, t, nl_m, nu_m, g_l, g_u)[0]
    tau_p = LI.det_si_line_input_tau(f_lu, lam, t, nl_p, nu_p, g_l, g_u)[0]
    e_m, taueff_m, srce_m = LI.det_si_line_input_material(nu_m, A, nu, tau_m, t)
    e_p, taueff_p_calc, srce_p = LI.det_si_line_input_material(nu_p, A, nu, tau_p, t)
    k_raw = D(lambda x, n: x * n / (C_LIGHT * t), tau_m, nu, scaling=True)
    lost = []
    for name, value, factors in (("tau_m13", tau_m, (nl_m - g_l / g_u * nu_m,)),
                                 ("tau_p12", tau_p, (nl_p - g_l / g_u * nu_p,)),
                                 ("e_m13", e_m, (nu_m, A)), ("e_p12", e_p, (nu_p, A))):
        if not math.isfinite(value):
            lost.append(f"{name}:NONFINITE")
        elif value == 0.0 and all(f_ != 0.0 for f_ in factors):
            lost.append(f"{name}:LOST_TO_ZERO")
    if lost or k_raw is None:
        row.update(status="부분", reason="UNDEFINED:" + ",".join(lost or ["k_raw"]))
        return row
    k_eff = LI.INTERNAL_OPACITY_TO_CGS / 1.0 if srce_m else k_raw
    r_d, d, d_rule = lnp_d(sup.get(up), sup.get(lo), nu, n_atom)
    radp = LI.det_si_line_input_radiation(e_p, taueff_p_calc, J_c, nu, t)
    radm = LI.det_si_line_input_radiation(e_m, taueff_m, J_c, nu, t)
    Jstar = radm["Jbar"] if radm else None
    bT = lnp_beta_T(tau_m)
    fpv = D(lambda v: FOUR_PI * v, V, scaling=True)
    F = {}
    F["F0"] = lnp_F(fpv, d, k_eff, Jbar_p, e_m)
    F["F1"] = lnp_F(fpv, 1.0, k_eff, Jbar_p, e_m)
    F["F2"] = lnp_F(fpv, 1.0, k_eff, Jstar, e_m)
    F["F3"] = lnp_F(fpv, bT, k_raw, J_c, e_m)
    F["F4"] = lnp_F(fpv, bT, k_raw, J_b, e_m)
    row.update(ion="Si II", nu_Hz=nu, wavelength_cm=lam, f_lu=f_lu, A_ul=A, g_l=g_l, g_u=g_u, t_s=t,
               n_l_m13=nl_m, n_u_m13=nu_m, n_l_p12=nl_p, n_u_p12=nu_p, tau_raw_m13=tau_m, tau_eff_m13=taueff_m,
               srce_chk_m13=srce_m, k_raw=k_raw, k_eff=k_eff, e_m13=e_m, e_p12=e_p, d_ratio=r_d, d=d, d_rule=d_rule,
               n_atom_cm3=n_atom, J_c_p=J_c, Jbar_p_W5=Jbar_p, beta_p_W5=beta_p, l_p_W5=l_p, tau_eff_p_W5=taueff_p,
               Jbar_star=Jstar, beta_N_m13=radm["beta"] if radm else None, beta_T_m13=bT, J_bin_rad12=J_b,
               volume_cm3=V,
               producer_check_tau_eff_rel=LI.det_si_line_input_match(taueff_p_calc, taueff_p)[2],
               producer_check_l_rel=LI.det_si_line_input_match(radp["l"] if radp else None, l_p)[2],
               producer_check_beta_rel=LI.det_si_line_input_match(radp["beta"] if radp else None, beta_p)[2])
    for k, (f, s) in F.items():
        row[k], row[f"{k}_S"] = f, s
    seps = []
    for a_, b_ in (("F0", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "F4")):
        diff = D(lambda x, y: x - y, row[a_], row[b_])
        sc = D(lambda x, y: x + y, row[f"{a_}_S"], row[f"{b_}_S"])
        sp = LEC.lec_separated(row[a_], row[b_], sc)
        row.update({f"{a_}_minus_{b_}": diff, f"{a_}_minus_{b_}_separated": sp,
                    f"{a_}_minus_{b_}_over_delta": D(lambda x: x / DELTA, diff, scaling=True)})
        seps.append(sp)
    tot = D(lambda x, y: x - y, row["F0"], row["F4"])
    parts = [row.get(f"{a_}_minus_{b_}") for a_, b_ in (("F0", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "F4"))]
    psum = D(lambda *x: sum(x), *parts)
    pscale = D(lambda *x: sum(abs(v) for v in x), *parts)
    row.update(F0_minus_F4=tot, F0_minus_F4_over_delta=D(lambda x: x / DELTA, tot, scaling=True),
               telescoping_separated=LEC.lec_separated(psum, tot, pscale))
    row["attribution"] = ("F0-F1 가중 d · F1-F2 국소항·beta 의 물질 이력(12→13) · F2-F3 소광 처방·beta 표현 · F3-F4 장 표현"
                          "(J_c 대 거친 구간 J) — 고정 J_c^p 에서의 등록 순서 산술 분해")
    und = [k for k in ("F0", "F1", "F2", "F3", "F4", "F0_minus_F4", "F0_minus_F4_over_delta", "d", "Jbar_star")
           if row.get(k) is None] + [f"{a_}_minus_{b_}" for (a_, b_), s in
                                    zip((("F0", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "F4")), seps) if s is None]
    und += [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
    if row["telescoping_separated"] is not False:
        und.append("TELESCOPING")
    row["status"] = "산출" if not und else "부분"
    row["reason"] = ("UNDEFINED:" + ",".join(und)) if und else ""
    return row


# ------------------------------------------------------------------ pipeline
def lnp_state_rows(cfg, st, info, why_common):
    """the 50 shell rows of one state from the comparison dump (existing reader), with provenance columns."""
    pc = cfg["O"] / "work/physics_comparison"
    why = []
    d = LEC.lec_read_dump(pc, info["it"], why, cfg["ns"], cfg["nb"])
    man = d["man"]
    if man.get("epoch_s") is None:
        why.append(f"{st}:MISSING:epoch_s")
    why_common[st] = why
    if why:
        return d, [{"row_kind": "전체 셸", "state": st, "shell": s, "scope": "전체 셸", "status": "미산출",
                    "reason": "COMMON_INPUT_DEFECT:" + ";".join(why[:3])} for s in range(cfg["ns"])]
    rows, _, _ = LEC.lec_d_rows(d, LEC.lec_dump_q(d), f"D{info['it']}", "dump grid")
    out = []
    for r in rows:
        o = {"row_kind": "전체 셸", "state": st, "shell": r["shell"], "scope": "전체 셸",
             "material_generation": man["population_generation"], "radiation_generation": man["radiation_generation"],
             "volume_cm3": r.get("volume_cm3"), "v_inner_cm_s": r.get("v_inner_cm_s"),
             "v_outer_cm_s": r.get("v_outer_cm_s"), "L": r.get("L"), "line_abs_heat_V": r.get("line_abs_heat"),
             "line_emit_cool_V": r.get("line_emit_cool"), "A_bb": r.get("bb_A"), "E_bb": r.get("bb_E"),
             "Q_bb": r.get("bb_Q"), "S_bb": r.get("bb_S"), "L_minus_Q": r.get("line_diff"),
             "L_minus_Q_over_delta": r.get("line_diff_over_delta"), "L_minus_Q_separated": r.get("line_separated"),
             "line_set": SOURCE["line_set"], "population_ownership": SOURCE["population"],
             "e_definition": SOURCE["e"] + f" (m = {man['population_generation']})",
             "k_definition": SOURCE["k"] + f" (m = {man['population_generation']})", "d_definition": SOURCE["d"],
             "Jbar_definition": SOURCE["Jbar"], "transport_definition": SOURCE["transport"],
             "raw_line_e_k_d": None, "raw_line_e_k_d_reason": "선별 e·k·d 배열과 선별 총흡수·총방출 미저장(구조적)"}
        und = [k for k in ("L", "A_bb", "E_bb", "Q_bb", "L_minus_Q", "L_minus_Q_over_delta", "L_minus_Q_separated")
               if o.get(k) is None]
        und += [k for k, v in o.items() if isinstance(v, float) and not math.isfinite(v)]
        o["status"] = "산출" if not und else "부분"
        o["reason"] = ("UNDEFINED:" + ",".join(und)) if und else ""
        out.append(o)
    return d, out


def lnp_conditions(st, m, p, reg):
    """registered sameness conditions (prereg 5) judged from the stored records and the sealed sources."""
    mh = "미확인" if p is None else ("충족" if p == m else "불충족")
    return {"cond_material_history": f"{mh} (장부 물질 {m}, J̄ 생산 물질 {p})",
            "cond_field_definition": "불충족(정의 차: J̄ 는 미세장 가우스 평균 J_c, 전달은 거친 구간 J; 소스)",
            "cond_weight_d": "미확인(선별 d 미저장; 제한 선만 계산)",
            "cond_negative_opacity": "처방 차(장부 SRCE_CHK, 전달 원 tau; 소스) — 적용 선 수 미확인",
            "cond_beta_form": "형식 차(beta_N 근사 분기 대 beta_T; 소스) — 수치 영향 미확인",
            "cond_line_set": "미확인(전달 구간의 선 구성 미저장; 장부는 에너지 집합·엄격 내부)",
            "cond_population_generation": "기록상 같은 덤프 물질 세대(충족), 선별 인구 소유권 동일성 미확인",
            "registration": reg}


def lnp_pipeline(cfg):
    O, T, sh = cfg["O"], cfg["T"], cfg["shell"]
    pc = O / "work/physics_comparison"
    common, rows = {}, []
    dumps = {}
    for st, info in cfg["states"].items():
        d, srows = lnp_state_rows(cfg, st, info, common)
        dumps[st] = d
        rows += srows
    rad_gens = {str(i["radiation"]) for i in cfg["states"].values()}
    w5 = lnp_scan_w5(O / "work/w5_lines.csv", cfg, rad_gens)
    reg = lnp_registration(O / "stderr.log", rad_gens)
    # producer material p of each state: the W5 call(s) of its radiation generation
    prod = {}
    for st, info in cfg["states"].items():
        calls = [c for c, gs in w5["generations"].items() if any(g[0] == str(info["radiation"]) for g in gs)]
        mats = {g[1] for c in calls for g in w5["generations"][c] if g[0] == str(info["radiation"])}
        prod[st] = {"calls": calls, "material": int(next(iter(mats))) if len(calls) == 1 and len(mats) == 1 else None}
        if prod[st]["material"] is None:
            common.setdefault("W5", []).append(f"{st}:W5_PRODUCER_LINK:{calls}:{sorted(mats)}")
    for st, info in cfg["states"].items():
        g = str(info["radiation"])
        ident, cov = reg[g].get("identity", []), reg[g].get("coverage", [])
        r = {"identity_rows": len(ident), "coverage_rows": len(cov)}
        if ident:
            r.update({k: ident[-1].get(k) for k in ("q_lines", "e_lines", "e_set_hash")})
        if cov:
            r.update({k: cov[-1].get(k) for k in ("valid_lines", "partial_lines", "unsampled_lines")})
        reg[g]["summary"] = r
        if not ident or not cov:
            common.setdefault("LOG", []).append(f"{st}:REGISTRATION_ROWS_MISSING")
    cond = {st: lnp_conditions(st, info["material"], prod[st]["material"], reg[str(info["radiation"])]["summary"])
            for st, info in cfg["states"].items()}
    for r in rows:
        st = r["state"]
        c = w5["check"].get(prod[st]["calls"][0], {}) if prod[st]["calls"] else {}
        bs = c.get("by_shell", {}).get(str(r["shell"]))
        r.update({k: v for k, v in cond[st].items() if k != "registration"})
        r.update(producing_call=",".join(prod[st]["calls"]) or None, producer_material=prod[st]["material"],
                 registration=json.dumps(cond[st]["registration"], ensure_ascii=False),
                 w5_support="예" if r["shell"] in cfg["w5_shells"] else "아니오",
                 w5_rows_used=bs["used"] if bs else None,
                 w5_excluded=json.dumps(dict(bs["excluded"])) if bs else None,
                 w5_max_jbar_residual=bs["max_jbar_residual"] if bs else None,
                 w5_max_beta_residual=bs["max_beta_residual"] if bs else None,
                 attributed_terms=None, attributed_terms_reason="전체 셸 선별 항 미저장 — 제한 선(A10 셸10)만 분해",
                 unattributed_residual=r.get("L_minus_Q"))
    # ---- the registered lines of A10 at the registered shell
    lrows, why_lines = [], []
    tr = None
    t = None
    if not common.get("A10"):
        tr = lnp_trace(cfg, why_lines)
        try:
            t = float(json.loads((O / "input/model/config.json").read_text())["time_explosion_s"])
        except (KeyError, ValueError, OSError):
            why_lines.append("CONFIG_TIME")
        if t is not None and t != float(dumps["A10"]["man"].get("epoch_s", "nan")):
            why_lines.append("TIME_EXPLOSION")
        ion9 = lnp_ion_row(pc / "physics_DET_iter0009.ion.csv", sh)
        ion10 = lnp_ion_row(pc / "physics_DET_iter0010.ion.csv", sh)
        if tr is not None:
            m13 = cfg["states"]["A10"]["material"]
            for tag, ion, key, gen in (("PRE_FL_VS_MATERIAL12", ion9, "pre", m13 - 1),
                                       ("COMMIT_VS_MATERIAL13", ion10, "mat13", m13)):
                if ion is None or int(ion["population_generation"]) != gen:
                    why_lines.append(f"{tag}:ION_ROW")
                    continue
                ref = (float(ion["Z14_nlte_q1_cm3"]), float(ion["Z14_nlte_q2_cm3"]))
                if any(LEC.lec_separated(a, b, abs(a) + abs(b)) is not False for a, b in zip(tr["sums"][key], ref)):
                    why_lines.append(tag)
    if common.get("A10"):
        why_lines.append("A10_COMMON_DEFECT")
    line_ids = {lid for _, lid, _, _ in cfg["lines"]}
    asha, abytes, atom, ahead = lnp_atomic(O / "input/model/line_list.csv", line_ids)
    lev_rows = [r for r in TR.read_trace(O / "input/model/levels.csv")
                if r["atomic_number"] == "14" and r["ion_number"] == "1"]
    levels = {int(r["level_number"]): r for r in lev_rows}
    sup = lnp_super_average(lev_rows)
    if not why_lines:
        A10 = dumps["A10"]
        V = float(A10["V"][sh])
        n_atom = None
        for r in TR.read_trace(pc / "physics_DET_iter0010.shell.csv"):
            if int(r["shell_id"]) == sh:
                w_ = []
                n_atom = LEC.lec_num(r.get("n_atom_cm3"), w_, "n_atom_cm3")
                if w_:
                    n_atom = None
        edges = A10["edges"]
        for grp, lid, lo, up in cfg["lines"]:
            nu_l = LEC.lec_num(atom[lid][0]["nu"], [], "nu") if len(atom.get(lid, [])) == 1 else None
            nu_l = nu_l if nu_l is not None and math.isfinite(nu_l) else None
            b = None
            if nu_l is not None:
                bb = [i for i in range(len(edges) - 1) if edges[i] <= nu_l < edges[i + 1]]
                b = bb[0] if len(bb) == 1 else None
            J_b = None
            if b is not None and not A10["bad"]["J"][sh, b]:
                J_b = float(A10["arr"]["J_nu"][sh, b])
            row = lnp_line(lid, lo, up, atom.get(lid, []), levels, sup, tr,
                           [r for r in w5["line_rows"].get(str(lid), []) if r["rad_generation"] ==
                            str(cfg["states"]["A10"]["radiation"])], cfg, V, n_atom, J_b, t)
            row["group"] = grp
            row["bin"] = b
            if J_b is None and row["status"] != "미산출":
                row["status"], row["reason"] = "부분", (row["reason"] + ";J_BIN").strip(";")
            lrows.append(row)
    else:
        lrows = [{"row_kind": "제한 선", "state": "A10", "shell": sh, "line_id": lid, "group": grp,
                  "status": "미산출", "reason": "LINE_INPUT_DEFECT:" + ";".join(why_lines[:4])}
                 for grp, lid, _, _ in cfg["lines"]]
    # ---- D_rest on the A10 registered shell row
    srow = [r for r in rows if r["state"] == "A10" and r["shell"] == sh][0]
    tot = LEC.lec_global([r.get("F0_minus_F4") for r in lrows])
    srow["six_line_F0_minus_F4_sum"] = tot["sum"] if tot else None
    srow["D_rest"] = lnp_d_rest(srow.get("L_minus_Q"), tot["sum"] if tot else None)
    srow["D_rest_label"] = "미귀속 잔여(부분집합의 장부·전달 구성원 연결 미확인)"
    srow["D_rest_over_delta"] = D(lambda x: x / DELTA, srow["D_rest"], scaling=True)
    if (srow["D_rest"] is None or srow["D_rest_over_delta"] is None) and srow.get("status") == "산출":
        srow["status"] = "부분"
        srow["reason"] = (srow.get("reason", "") + ";UNDEFINED:D_rest").strip(";")
    # ---- global rows
    grows = []
    for st, info in cfg["states"].items():
        g = LEC.lec_summary([r for r in rows if r["state"] == st], st, ["L", "Q_bb", "L_minus_Q"])
        c = w5["check"].get(prod[st]["calls"][0], {}) if prod[st]["calls"] else {}
        g.update(row_kind="전역", scope="상태 전역(전체 셸)", w5_call=",".join(prod[st]["calls"]) or None,
                 w5_scope="W5 = Z 14·16·20, ion 1·2 매핑 선과 저장 5셸만 — 전체 에너지 집합 아님",
                 w5_rows_used=c.get("used"), w5_excluded=json.dumps(dict(c.get("excluded", {}))),
                 w5_max_jbar_residual=c.get("max_jbar_residual"), w5_max_beta_residual=c.get("max_beta_residual"),
                 **{k: v for k, v in cond[st].items() if k != "registration"},
                 registration=json.dumps(cond[st]["registration"], ensure_ascii=False))
        if not c.get("used"):
            g["status"] = "부분"
            g["reason"] = (g.get("reason", "") + ";W5_CHECK_EMPTY").strip(";")
        grows.append(g)
    return {"rows": rows, "line_rows": lrows, "global_rows": grows, "common": common, "why_lines": why_lines,
            "w5": w5, "reg": reg, "prod": prod, "cond": cond, "trace": tr, "atomic": (asha, abytes),
            "dumps": dumps}


def lnp_rows_kept(p):
    """the O-only full-shell values survive: every shell row has L, Q_bb and L - Q defined (D_rest may be undefined
    when a registered line is undefined)."""
    return all(r.get("L") is not None and r.get("Q_bb") is not None and r.get("L_minus_Q") is not None and
               (r["status"] == "산출" or r["reason"] == "UNDEFINED:D_rest") for r in p["rows"])


def lnp_d_rest(l_minus_q, six_line_sum):
    """D_rest = (L - Q_bb) of the registered shell minus the sum of F0 - F4 over the registered lines."""
    return D(lambda a, b: a - b, l_minus_q, six_line_sum)


def lnp_produced(p):
    """every mandatory part valid: 100 shell rows, the registered lines, D_rest and both global rows."""
    ok = all(r.get("status") == "산출" for r in p["rows"] + p["line_rows"] + p["global_rows"])
    return ok and not any(p["common"].values()) and not p["why_lines"]


def lnp_branch(produced, conditions_unmet, separated):
    if not produced or any(s is None for s in separated):
        return 2
    return 0 if (conditions_unmet or any(separated)) else 1


# ------------------------------------------------------------------ toy files (selftest)
TOY_LINES = (("T1", 11, 0, 2), ("T2", 12, 1, 2), ("T3", 13, 0, 3))
TOY_T = 1.683072e6
TOY_NU = {11: 1.05e15, 12: 1.15e15, 13: 1.3e15}
TOY_POP = {"pre": [3.0e2, 5.0e1, 4.0e1, 6.0e1, 9.0e2], "fin": [3.3e2, 4.0e1, 2.0e2, 3.3e2, 9.5e2]}


def lnp_toy_levels():
    """levels: level 2 energy gives r = 1.2 for line 11 (d = r), level 3 energy r = 1 for line 13; line 12 gets
    r far from 1 (d = 1); level 1 at 4 eV."""
    hv = lambda lid: H_PLANCK * TOY_NU[lid] / EV_TO_ERG                                   # noqa: E731
    E2, E3 = 1.2 * hv(11), 1.0 * hv(13)
    return [(0, 0.0, 2, 0), (1, 4.0, 2, 1), (2, E2, 4, 2), (3, E3, 2, 3)]


def lnp_toy_atomic(lid, lo, up):
    """f_lu chosen so that the populations give tau ~ 2 (line 11), tau < -0.5 (line 12, maser) and tau = 0 (13)."""
    nu = TOY_NU[lid]
    lam = C_LIGHT / nu
    f = {11: 4.0e-3, 12: 3.0e-2, 13: 5.0e-3}[lid]
    A = {11: 2.0e8, 12: 5.0e7, 13: 1.0e8}[lid]
    return {"atomic_number": "14", "ion_number": "1", "level_number_lower": str(lo), "level_number_upper": str(up),
            "line_id": str(lid), "wavelength": repr(lam * 1e8), "f_ul": "0", "f_lu": repr(f), "nu": repr(nu),
            "B_lu": "1.0", "B_ul": "1.0", "A_ul": repr(A), "wavelength_cm": repr(lam)}


def lnp_write_toy(root, mutate=None, record=None):
    """a toy O and T: the LEC toy dumps (2 shells, 3 bins; shell 1 is the registered shell) plus ion files,
    n_atom, epoch, W5, logs, config, line list, levels and the Si shell trace of a registered adopted call 2."""
    mutate = mutate or (lambda kind, obj: obj)
    record = record if record is not None else {}

    def lmut(kind, obj):
        if kind[0] == "manifest":
            obj = dict(obj, epoch_s=TOY_T)
        if kind[0] == "shell":
            obj = [dict(r, n_atom_cm3=repr(1.0e9 * (r["shell_id"] + 1))) for r in obj]
        return mutate(kind, obj)
    pc, cap, cells, bridge = LEC.lec_write_toy(root, mutate=lmut, record=record)
    O, T = root / "O", root / "T"
    (O / "work").mkdir(parents=True)
    (O / "input/model").mkdir(parents=True)
    pc.rename(O / "work/physics_comparison")
    pc = O / "work/physics_comparison"
    (T / "si_trace").mkdir(parents=True)
    levels = lnp_toy_levels()
    lev_rows = [{"atomic_number": "14", "ion_number": "1", "level_number": str(n), "energy_eV": repr(E), "g": str(g),
                 "metastable": "0", "super_level": str(s), "configuration": f"L{n}"} for n, E, g, s in levels]
    lev_rows = mutate(("levels", 0), lev_rows)
    with open(O / "input/model/levels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(lev_rows[0]))
        wr.writeheader()
        wr.writerows(lev_rows)
    atoms = mutate(("atomic", 0), [lnp_toy_atomic(lid, lo, up) for _, lid, lo, up in TOY_LINES])
    with open(O / "input/model/line_list.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(lnp_toy_atomic(11, 0, 2)))
        wr.writeheader()
        wr.writerows(atoms)
    (O / "input/model/config.json").write_text(json.dumps(mutate(("config", 0), {"time_explosion_s": TOY_T})))
    logs = []
    for g in ("12", "13"):
        logs.append(f"[R6][LINE-IDENTITY] lane=DET generation={g} q_lines=5 q_set_hash=aa e_lines=7 e_set_hash=bb")
        logs.append(f"[R6][LINE-COVERAGE] generation={g} all_lines=9 q_lines=5 e_lines=7 valid_lines=7 "
                    "partial_lines=0 unsampled_lines=0")
    (O / "stderr.log").write_text("\n".join(mutate(("log", 0), logs)) + "\n")
    # Si shell trace: call 1 (production, material 12) and adopted call 2 (material 13)
    pre, fin = list(TOY_POP["pre"]), list(TOY_POP["fin"])
    g = {n: gg for n, _, gg, _ in levels}
    # line 13 (0 -> 3, g 2 / 2): equal populations of material 13 give tau = 0 exactly
    fin[0] = fin[3] = 3.3e2
    # the adopted call's internal FINAL_FL differs level by level from the commit (= the published material 13)
    final_fl = [v * (1.0 + 0.01 * (i + 1)) for i, v in enumerate(fin)]
    trf = {"si10_ne.csv": [{"bundle": "1", "ne_iter": "1", "accepted": "1"}],
           "si10_calls.csv": [{"call": "1", "bundle": "1", "ne_iter": "1", "ce_pass": "0", "n_e": "1e9", "T_e": "8000",
                               "N_fl": "5", "n_lo_super": "4", "rad_generation": "12", "core_generation": "13"},
                              {"call": "2", "bundle": "1", "ne_iter": "1", "ce_pass": "1", "n_e": "1.1e9",
                               "T_e": "8000", "N_fl": "5", "n_lo_super": "4", "rad_generation": "12",
                               "core_generation": "13"}],
           "si10_cevec.csv": [{"bundle": "1", "ne_iter": "1", "ce_pass": "1", "i_fl": str(i), "old": repr(pre[i]),
                               "after": repr(fin[i])} for i in range(5)],
           "si10_commit.csv": [{"kind": "level", "i": str(i), "value": repr(fin[i])} for i in range(5)] +
                              [{"kind": "n_e", "i": "", "value": "1.1e9"}, {"kind": "T_e", "i": "", "value": "8000"}],
           "si10_vec.csv": [{"call": "1", "phase": "PRE_FL", "i": str(i), "value": repr(pre[i])} for i in range(5)] +
                           [{"call": "2", "phase": "FINAL_FL", "i": str(i), "value": repr(final_fl[i])}
                            for i in range(5)],
           "si10_slmap.csv": [{"call": c, "i_fl": str(i), "fl_index": str(i), "global_level": str(i),
                               "level_num": str(i if i < 4 else 0), "upper_ion": "0" if i < 4 else "1",
                               "E_eV": "0", "g": str(g[i] if i < 4 else 1), "sl_local": str(i)}
                              for c in ("1", "2") for i in range(5)]}
    for name, rows in trf.items():
        rows = mutate(("trace", name), rows)
        with open(T / "si_trace" / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    # ion files of material 12 (iter 9) and 13 (iter 10) at both shells (shell 1 = registered)
    for it, pops in ((9, pre), (10, fin)):
        rows = [{"population_generation": str(it + 3), "shell_id": str(s),
                 "Z14_nlte_q1_cm3": repr(sum(pops[:4]) if s == 1 else 1.0),
                 "Z14_nlte_q2_cm3": repr(pops[4] if s == 1 else 1.0)} for s in range(2)]
        rows = mutate(("ion", it), rows)
        with open(pc / f"physics_DET_iter{it:04d}.ion.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    # W5: calls 12 (radiation 12 / material 12) and 13 (13 / 13) for both shells; values follow the production
    # relation with the call's material populations (12 = PRE_FL, 13 = FINAL_FL at the registered shell)
    head = ["call", "operator", "rad_generation", "material_generation", "shell", "line_id", "Z", "ion", "nu_hz",
            "J_c", "J_c_err_upper", "beta", "L_l", "Jbar", "Jbar_err_upper", "tau_eff", "srce_chk", "status"]
    w5 = []
    for _, lid, lo, up in TOY_LINES:
        a = lnp_toy_atomic(lid, lo, up)
        for call, pops in (("12", pre), ("13", fin)):
            for s in range(2):
                tau = LI.det_si_line_input_tau(float(a["f_lu"]), float(a["wavelength_cm"]), TOY_T, pops[lo], pops[up],
                                               g[lo], g[up])[0]
                e, te, sc = LI.det_si_line_input_material(pops[up], float(a["A_ul"]), TOY_NU[lid], tau, TOY_T)
                J_c = 1.0e-7 * (1 + s) * (1.05 if call == "13" else 1.0) * (1 + 0.1 * lid)
                rad = LI.det_si_line_input_radiation(e, te, J_c, TOY_NU[lid], TOY_T)
                w5.append({"call": call, "operator": SOBOLEV_OP, "rad_generation": call, "material_generation": call,
                           "shell": str(s), "line_id": str(lid), "Z": "14", "ion": "1", "nu_hz": a["nu"],
                           "J_c": repr(J_c), "J_c_err_upper": "0", "beta": repr(rad["beta"]), "L_l": repr(rad["l"]),
                           "Jbar": repr(rad["Jbar"]), "Jbar_err_upper": "0", "tau_eff": repr(te),
                           "srce_chk": str(sc), "status": "OK"})
        w5.append({"call": "1", "operator": "INIT_SHARED_GAUSSIAN", "rad_generation": "1", "material_generation": "0",
                   "shell": "1", "line_id": str(lid), "Z": "14", "ion": "1", "nu_hz": a["nu"],
                   "J_c": "NOT_APPLICABLE", "J_c_err_upper": "NOT_APPLICABLE", "beta": "NOT_APPLICABLE",
                   "L_l": "NOT_APPLICABLE", "Jbar": "2e-7", "Jbar_err_upper": "0", "tau_eff": "NOT_APPLICABLE",
                   "srce_chk": "NOT_APPLICABLE", "status": "OK_INIT_GAUSSIAN_PROFILE_MEAN"})
    w5 = mutate(("w5", 0), w5)
    with open(O / "work/w5_lines.csv", "w", newline="") as f:
        f.write(",".join(head) + "\n")
        for r in w5:
            f.write(",".join(str(r[k]) for k in head) + "\n")
    record.update(pre=pre, fin=fin, final_fl=final_fl, g=g, levels=levels)
    return {"O": O, "T": T, "shell": 1, "ns": 2, "nb": 3, "lines": TOY_LINES, "w5_shells": (0, 1),
            "adopted": {"bundle": 1, "ne_iter": 1, "ce_pass": 1, "call": 2}, "states": REAL["states"]}


def lnp_toy_hand(rec, cfg, p):
    """hand values of F0 .. F4 for each toy line from the written arrays (independent closed forms)."""
    out = {}
    sh = cfg["shell"]
    V = 4.0 * math.pi / 3.0 * (rec["ro"][sh] ** 3 - rec["ri"][sh] ** 3)
    edges = rec["edges"]
    Jb = {r["bin_id"]: float(r["J_nu"]) for r in rec[10] if r["shell_id"] == sh}
    lev = {n: (E, g) for n, E, g, _ in rec["levels"]}
    for _, lid, lo, up in TOY_LINES:
        a = lnp_toy_atomic(lid, lo, up)
        nu, lam, f, A = TOY_NU[lid], float(a["wavelength_cm"]), float(a["f_lu"]), float(a["A_ul"])
        t = TOY_T
        coef = LI.SOBOLEV_COEFF * f * lam * t

        def tau(p_):
            return coef * (p_[lo] - lev[lo][1] / lev[up][1] * p_[up])
        tm, tp = tau(rec["fin"]), tau(rec["pre"])
        em, ep = rec["fin"][up] * A * H_PLANCK * nu / FOUR_PI, rec["pre"][up] * A * H_PLANCK * nu / FOUR_PI
        kraw = tm * nu / (C_LIGHT * t)
        keff = 1e-10 if tm < -0.5 else kraw
        teff_m = 1e-10 * C_LIGHT * t / nu if tm < -0.5 else tm
        teff_p = 1e-10 * C_LIGHT * t / nu if tp < -0.5 else tp

        def bc(x):
            if abs(x) < 1e-3:
                c = 0.5 - x / 6.0 * (1.0 - x / 4.0)
                return 1.0 - x * c, c
            b = (1.0 - math.exp(-x)) / x if x < 40 else 1.0 / x
            return b, (1.0 - b) / x
        J_c = 1.0e-7 * (1 + sh) * (1 + 0.1 * lid)
        bp, cp = bc(teff_p)
        Jbar_p = bp * J_c + ep * (C_LIGHT * t / nu) * cp
        bm, cm = bc(teff_m)
        Jstar = bm * J_c + em * (C_LIGHT * t / nu) * cm
        bT = 1.0 if tm == 0 else -math.expm1(-tm) / tm
        r = (lev[up][0] - lev[lo][0]) * EV_TO_ERG / (H_PLANCK * nu)
        d = 1.0 if abs(r - 1.0) > 0.5 else r
        b = [i for i in range(len(edges) - 1) if edges[i] <= nu < edges[i + 1]][0]
        out[lid] = {"F0": FOUR_PI * V * d * (keff * Jbar_p - em), "F1": FOUR_PI * V * (keff * Jbar_p - em),
                    "F2": FOUR_PI * V * (keff * Jstar - em), "F3": FOUR_PI * V * bT * (kraw * J_c - em),
                    "F4": FOUR_PI * V * bT * (kraw * Jb[b] - em), "tau_m": tm, "d": d}
    return out


def cmd_selftest():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    close = lambda a, b: a is not None and b is not None and abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)  # noqa
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or None))
    try:
        def run(name, mutate=None):
            root = tmp / name
            root.mkdir()
            rec = {}
            cfg = lnp_write_toy(root, mutate=mutate, record=rec)
            return lnp_pipeline(cfg), rec, cfg

        def lines(p):
            return {r["line_id"]: r for r in p["line_rows"]}

        def shells(p, st="A10"):
            return {r["shell"]: r for r in p["rows"] if r["state"] == st}
        p, rec, cfg = run("clean")
        hand = lnp_toy_hand(rec, cfg, p)
        bad = [(r.get("row_kind"), r.get("state"), r.get("shell"), r.get("line_id"), r.get("reason"))
               for r in p["rows"] + p["line_rows"] + p["global_rows"] if r.get("status") != "산출"]
        if bad or p["why_lines"] or any(p["common"].values()):
            print("[DIAG] clean toy: " + json.dumps({"bad": bad[:8], "why_lines": p["why_lines"],
                                                    "common": p["common"]}, ensure_ascii=False, default=str))
        L = lines(p)
        expect(all(r["status"] == "산출" for r in p["rows"] + p["line_rows"] + p["global_rows"]) and
               not any(p["common"].values()) and not p["why_lines"] and lnp_produced(p),
               "toy files through the reader path: all full-shell rows, the registered lines, D_rest and global "
               "rows produced (positive control)")
        expect(hand[12]["tau_m"] < -0.5 and hand[13]["tau_m"] == 0.0 and 0.5 < hand[11]["d"] < 1.5 and
               hand[11]["d"] != 1.0 and hand[12]["d"] == 1.0, "toy covers a maser line (SRCE_CHK), an exact tau = 0 "
               "line and a d != 1 line (fixture control)")

        def hand_ok(pp):
            Lr = lines(pp)
            return all(all(close(Lr.get(lid, {}).get(k), hand[lid][k]) for k in ("F0", "F1", "F2", "F3", "F4"))
                       for lid in hand)
        expect(hand_ok(p), "F0 .. F4 through the reader path equal the independent hand values for all toy lines")
        r1 = shells(p)[1]
        expect(all(L[l].get("F0_minus_F4") is not None for l in L) and r1.get("D_rest") is not None and close(
            r1["D_rest"], r1["L_minus_Q"] - sum(L[l]["F0_minus_F4"] for l in L)) and
               r1["L"] != sum(L[l]["F0"] for l in L) and r1["scope"] == "전체 셸" and
               "전체 에너지 집합 아님" in p["global_rows"][0]["w5_scope"],
               "scope: the shell row keeps the dump L, D_rest = (L - Q) - sum (F0 - F4), W5 labelled as not the "
               "full energy set (positive control)")

        # ---- wrong arithmetic / wrong connections injected into the pipeline: the hand check must fail
        orig = globals()["lnp_F"]

        def variant(fn):
            globals()["lnp_F"] = fn
            try:
                return hand_ok(run(f"v{len(results)}")[0])
            finally:
                globals()["lnp_F"] = orig
        expect(not variant(lambda fpv, w, a, b, c: orig(fpv, w, a, b, -c)), "inject a sign error (absorption minus "
               "emission reversed for the emission term): hand check fails")
        expect(not variant(lambda fpv, w, a, b, c: orig(fpv / FOUR_PI, w, a, b, c)), "inject 4 pi omitted: fails")
        expect(not variant(lambda fpv, w, a, b, c: orig(FOUR_PI, w, a, b, c)), "inject volume omitted: fails")
        orig_line = globals()["lnp_line"]

        def lvariant(patch):
            globals()["lnp_line"] = patch
            try:
                return hand_ok(run(f"l{len(results)}")[0])
            finally:
                globals()["lnp_line"] = orig_line

        def with_w5(change):
            def f(lid, lo, up, atom, levels, sup, tr, w5row, cfg_, V, n_atom, J_b, t):
                return orig_line(lid, lo, up, atom, levels, sup, tr, [change(dict(w5row[0]))] if w5row else w5row,
                                 cfg_, V, n_atom, J_b, t)
            return f
        expect(not lvariant(with_w5(lambda r: dict(r, Jbar=r["L_l"]))), "inject W5 L_l used as Jbar (the local "
               "term taken as the field): fails")
        expect(not lvariant(with_w5(lambda r: dict(r, Jbar=repr(float(r["beta"]) * float(r["Jbar"]))))),
               "inject beta applied twice to Jbar: fails")
        expect(not lvariant(lambda lid, lo, up, atom, levels, sup, tr, w5row, cfg_, V, n_atom, J_b, t: orig_line(
            lid, lo, up, atom, levels, sup, dict(tr, mat13=tr["pre"]), w5row, cfg_, V, n_atom, J_b, t)),
               "inject the old material (12) as the ledger material of F2 (and the line): fails")
        expect(not lvariant(lambda lid, lo, up, atom, levels, sup, tr, w5row, cfg_, V, n_atom, J_b, t: orig_line(
            lid, lo, up, atom, levels, sup, dict(tr, mat13=tr["final_fl"]), w5row, cfg_, V, n_atom, J_b, t)) and
               rec["final_fl"] != rec["fin"],
               "inject the adopted FINAL_FL (differing from the commit in the clean toy) as the ledger material: "
               "fails against the hand values from the commit")
        expect(not lvariant(lambda lid, lo, up, atom, levels, sup, tr, w5row, cfg_, V, n_atom, J_b, t: orig_line(
            lid, lo, up, atom, levels, sup, tr, w5row, cfg_, V, n_atom, J_b * 1.1 if J_b else J_b, t)),
               "inject the radiation-13 bin J (another generation) into F4: fails")
        orig_bT = globals()["lnp_beta_T"]
        globals()["lnp_beta_T"] = lambda tau: orig_bT(LI.det_si_line_input_material(1.0, 1.0, 1e15, tau, TOY_T)[1])
        try:
            bad_eff = hand_ok(run("taueff")[0])
        finally:
            globals()["lnp_beta_T"] = orig_bT
        expect(not bad_eff, "inject the effective (SRCE_CHK) tau into the transport beta of F3 / F4: fails")
        orig_d = globals()["lnp_d"]
        globals()["lnp_d"] = lambda Eu, El, nu, n: (orig_d(Eu, El, nu, n)[0], 1.0, "forced")
        try:
            bad_d = hand_ok(run("dforce")[0])
        finally:
            globals()["lnp_d"] = orig_d
        expect(not bad_d, "inject d = 1 unconditionally: fails")
        globals()["lnp_beta_T"] = lambda tau: orig_bT(abs(tau)) if tau is not None else None
        try:
            bad_neg = hand_ok(run("negtau")[0])
        finally:
            globals()["lnp_beta_T"] = orig_bT
        expect(not bad_neg, "inject the valid negative tau removed (absolute value): fails")
        globals()["lnp_beta_T"] = lambda tau: D(lambda t_: -math.expm1(-t_) / t_, tau)
        try:
            pz = run("zerodiv")[0]
        finally:
            globals()["lnp_beta_T"] = orig_bT
        expect(lines(pz)[13]["status"] != "산출" and lines(pz)[11]["status"] == "산출",
               "inject a division at tau = 0 (no finite limit): that line undefined, the other lines kept")

        # ---- links, rows, supports, common definitions through the files
        pa, _, _ = run("w5gen", lambda k, o: [dict(r, material_generation="13") if (r["call"], r["line_id"],
                                                    r["shell"]) == ("12", "11", "1") else r for r in o]
                       if k == ("w5", 0) else o)
        expect(lines(pa)[11]["status"] == "미산출" and "W5_LINK" in lines(pa)[11]["reason"] and
               lines(pa)[12]["status"] == "산출" and lnp_rows_kept(pa),
               "inject a W5 row of the line with another material generation: that line refused, others and the "
               "shell rows kept")
        pa, _, _ = run("w5miss", lambda k, o: [r for r in o if not (r["call"] == "12" and r["line_id"] == "12" and
                                                                   r["shell"] == "1")] if k == ("w5", 0) else o)
        r1 = shells(pa)[1]
        expect(lines(pa)[12]["status"] == "미산출" and lines(pa)[11]["status"] == "산출" and r1["D_rest"] is None and
               r1.get("L") is not None and not lnp_produced(pa) and
               lnp_branch(lnp_produced(pa), True, []) == 2,
               "inject a missing W5 row of one line: that line, the six-line sum and D_rest undefined; the shell L, "
               "other lines kept; branch 2")
        pa, _, _ = run("w5dup", lambda k, o: o + [r for r in o if (r["call"], r["line_id"], r["shell"]) ==
                                                 ("12", "11", "1")] if k == ("w5", 0) else o)
        expect(lines(pa)[11]["status"] == "미산출" and "W5_ROWS:2" in lines(pa)[11]["reason"],
               "inject a duplicated W5 row: that line refused")
        pa, _, _ = run("w5init", lambda k, o: [dict(r, operator="INIT_SHARED_GAUSSIAN") if (
            r["call"], r["line_id"], r["shell"]) == ("12", "11", "1") else r for r in o] if k == ("w5", 0) else o)
        expect(lines(pa)[11]["status"] == "미산출", "inject an initialisation row used as the Sobolev row: refused")
        pa, _, _ = run("w5na", lambda k, o: [dict(r, J_c="NOT_APPLICABLE") if (r["call"], r["line_id"], r["shell"])
                                             == ("12", "11", "1") else r for r in o] if k == ("w5", 0) else o)
        expect(lines(pa)[11]["status"] == "미산출" and lines(pa)[11].get("F0") is None,
               "inject a not-applicable W5 value: undefined, not zero")
        pa, _, _ = run("w5out", lambda k, o: o + [dict(o[0], shell="7")] if k == ("w5", 0) else o)
        c12 = pa["w5"]["check"]["12"]
        expect(c12["excluded"].get("OUTSIDE_SUPPORT") == 1 and "7" not in [str(s) for s in (0, 1)],
               "inject a W5 row outside the stored support: excluded from the check, not counted as used")
        pa, _, _ = run("lev", lambda k, o: [dict(r, level_num="9") if (r["call"], r["i_fl"]) == ("1", "2") else r
                                            for r in o] if k == ("trace", "si10_slmap.csv") else o)
        expect(all(r["status"] == "미산출" for r in pa["line_rows"]) and
               lnp_rows_kept(pa),
               "inject a level map mismatch between call 1 and the adopted call: every registered line stopped, "
               "shell rows kept")
        pa, _, _ = run("ion13", lambda k, o: [dict(r, Z14_nlte_q1_cm3="1.5") if r["shell_id"] == "1" else r
                                              for r in o] if k == ("ion", 10) else o)
        expect("COMMIT_VS_MATERIAL13" in pa["why_lines"] and not lnp_produced(pa) and
               all(r["status"] == "미산출" for r in pa["line_rows"])
               and lnp_rows_kept(pa),
               "inject a material-13 ion sum that the commit (= CE after) does not reproduce (public link only): "
               "registered lines stopped, O-only L and Q kept, not produced")

        def commit_mut(fn):
            return lambda k, o: fn(o) if k == ("trace", "si10_commit.csv") else o

        def swap01(o):
            v = {r["i"]: r["value"] for r in o if r["kind"] == "level"}
            return [dict(r, value=v["1"] if r["i"] == "0" else v["0"]) if r["kind"] == "level" and r["i"] in ("0", "1")
                    else r for r in o]
        cases = {"redistribution (ion sums preserved)": swap01,
                 "duplicated commit row": lambda o: o + [r for r in o if r["kind"] == "level" and r["i"] == "0"],
                 "missing commit row": lambda o: [r for r in o if not (r["kind"] == "level" and r["i"] == "4")],
                 "commit index mislink": lambda o: [dict(r, i="5") if r["kind"] == "level" and r["i"] == "4" else r
                                                    for r in o]}
        ok = []
        for i, (name, fn) in enumerate(cases.items()):
            pa, _, _ = run(f"commit{i}", commit_mut(fn))
            ok.append(any(w.startswith("TRACE_") for w in pa["why_lines"]) and
                      all(r["status"] == "미산출" for r in pa["line_rows"]) and lnp_rows_kept(pa) and
                      not lnp_produced(pa))
        expect(all(ok) and len(ok) == 4, "inject a commit that differs from the adopted CE after level by level "
               "(incl. an ion-sum-preserving redistribution), a duplicated / missing commit row and a commit index "
               "mislink: registered lines stopped, O-only L and Q kept, not produced (each)")
        pa, _, _ = run("time", lambda k, o: {"time_explosion_s": TOY_T * 1.001} if k == ("config", 0) else o)
        expect("TIME_EXPLOSION" in pa["why_lines"], "inject another explosion time: refused")
        pa, _, _ = run("man", lambda k, o: {kk: v for kk, v in o.items() if kk != "opacity_units"}
                       if k == ("manifest", 10) else o)
        expect(pa["common"].get("A10") and all(r["status"] == "미산출" for r in pa["rows"] if r["state"] == "A10") and
               all(r["status"] == "산출" for r in pa["rows"] if r["state"] == "A11") and
               all(r["status"] == "미산출" for r in pa["line_rows"]),
               "inject a missing unit key into the A10 manifest: A10 rows and lines stopped, A11 kept")
        pa, _, _ = run("epoch", lambda k, o: {kk: v for kk, v in o.items() if kk != "epoch_s"}
                       if k == ("manifest", 11) else o)
        expect(pa["common"].get("A11") and all(r["status"] == "미산출" for r in pa["rows"] if r["state"] == "A11"),
               "inject a missing epoch in the A11 manifest: A11 stopped")
        pa, _, _ = run("w5prod", lambda k, o: [dict(r, material_generation="11") if r["call"] == "13" and r["shell"]
                                               == "0" else r for r in o] if k == ("w5", 0) else o)
        expect(any("A11:W5_PRODUCER_LINK" in w for w in pa["common"].get("W5", [])) and not lnp_produced(pa),
               "inject two producer materials for one radiation generation: producer link refused, not produced")
        pa, _, _ = run("log", lambda k, o: [x for x in o if "generation=13" not in x] if k == ("log", 0) else o)
        expect(any("A11:REGISTRATION_ROWS_MISSING" in w for w in pa["common"].get("LOG", [])),
               "inject missing registration rows of radiation 13: refused")

        # ---- non-finite values, text and product loss, missing shell
        pa, _, _ = run("nan", lambda k, o: [dict(r, J_c="nan") if (r["call"], r["line_id"], r["shell"]) ==
                                            ("12", "11", "1") else r for r in o] if k == ("w5", 0) else o)
        expect(lines(pa)[11]["status"] == "미산출" and lines(pa)[12]["status"] == "산출" and
               lnp_rows_kept(pa),
               "inject a non-finite W5 value of one line: only that line undefined")
        pa, _, _ = run("text", lambda k, o: [dict(r, Jbar="1e-400") if (r["call"], r["line_id"], r["shell"]) ==
                                             ("12", "11", "1") else r for r in o] if k == ("w5", 0) else o)
        expect(lines(pa)[11]["status"] == "미산출" and "TEXT_LOST_TO_ZERO" in lines(pa)[11]["reason"],
               "inject a non-zero W5 text lost to 0: undefined")
        pa, _, _ = run("big", lambda k, o: [dict(r, A_ul="1e300") if r["line_id"] == "11" else r for r in o]
                       if k == ("atomic", 0) else o)
        expect(lines(pa)[11]["status"] != "산출" and lines(pa)[12]["status"] == "산출",
               "inject an atomic value whose products overflow: that line undefined, others kept")
        pa, _, _ = run("ledger", lambda k, o: [dict(r, line_abs_heat="bad") if r["shell_id"] == 0 else r for r in o]
                       if k == ("shell", 10) else o)
        s0 = shells(pa)[0]
        g10 = [g for g in pa["global_rows"] if g["relation"] == "A10"][0]
        expect(s0["status"] == "부분" and s0.get("L") is None and shells(pa)[1]["status"] == "산출" and
               "L_sum" not in g10 and "Q_bb_sum" in g10 and not lnp_produced(pa),
               "inject one undefined ledger term in one shell: that shell's L and the L global sum undefined, Q_bb "
               "global sum and other shells kept, not produced")
        pa, _, _ = run("shellgone", lambda k, o: [r for r in o if r["shell_id"] != 0] if k == ("spectral", 11) else o)
        expect(pa["common"].get("A11") and all(r["status"] == "미산출" for r in pa["rows"] if r["state"] == "A11") and
               all(r["status"] == "산출" for r in pa["rows"] if r["state"] == "A10"),
               "inject one missing shell of the A11 spectral table: A11 stopped (coverage), A10 kept")
        # ---- common frame / grid definitions
        pa, _, _ = run("frame", lambda k, o: dict(o, frame="LAB") if k == ("manifest", 11) else o)
        expect(pa["common"].get("A11") and all(r["status"] == "미산출" for r in pa["rows"] if r["state"] == "A11") and
               all(r["status"] == "산출" for r in pa["line_rows"]),
               "inject another frame into the A11 manifest: A11 stopped, A10 and the registered lines kept")
        pa, _, _ = run("grid", lambda k, o: [dict(r, nu_lo_Hz=repr(float(r["nu_lo_Hz"]) * 1.01))
                                             if (r["shell_id"], r["bin_id"]) == (0, 1) else r for r in o]
                       if k == ("spectral", 10) else o)
        expect(pa["common"].get("A10") and all(r["status"] == "미산출" for r in pa["rows"] if r["state"] == "A10") and
               all(r["status"] == "미산출" for r in pa["line_rows"]) and
               all(r["status"] == "산출" for r in pa["rows"] if r["state"] == "A11"),
               "inject a per-shell frequency grid mismatch into the A10 dump: A10 rows and lines stopped, A11 kept")

        # ---- scope: the shell value is the dump ledger, D_rest subtracts the registered lines
        def scope_ok(pp, root):
            src = [r for r in TR.read_trace(root / "O/work/physics_comparison/physics_DET_iter0010.shell.csv")
                   if int(r["shell_id"]) == 1][0]
            V = 4.0 * math.pi / 3.0 * (float(src["r_outer_cm"]) ** 3 - float(src["r_inner_cm"]) ** 3)
            Ldump = V * (float(src["line_abs_heat"]) - float(src["line_emit_cool"]))
            r1_ = shells(pp)[1]
            s = sum(r["F0_minus_F4"] for r in pp["line_rows"])
            return close(r1_["L"], Ldump) and close(r1_["D_rest"], r1_["L_minus_Q"] - s) and \
                "전체 에너지 집합 아님" in pp["global_rows"][0]["w5_scope"]
        expect(scope_ok(p, tmp / "clean"), "scope check on the clean toy: shell L = the dump ledger, D_rest = "
               "(L - Q) - sum (F0 - F4) (positive control)")
        orig_dr = globals()["lnp_d_rest"]
        caught = []
        for i, fn in enumerate((lambda lq, s: lq, lambda lq, s: s)):
            globals()["lnp_d_rest"] = fn
            try:
                caught.append(not scope_ok(run(f"scope{i}")[0], tmp / f"scope{i}"))
            finally:
                globals()["lnp_d_rest"] = orig_dr
        expect(all(caught) and len(caught) == 2, "inject the registered lines not subtracted / the six-line sum used "
               "as the shell value: the scope check fails for each")

        # ---- overflow of a global sum of finite shells; a Delta ratio lost to 0 while the raw values stay
        def huge(k, o):
            if k != ("shell", 10):
                return o
            out = []
            for r in o:
                s = int(r["shell_id"])
                V = 4.0 * math.pi / 3.0 * ((1.0e15 * (s + 2)) ** 3 - (1.0e15 * (s + 1)) ** 3)
                out.append(dict(r, line_abs_heat=repr(1.0e308 / V), line_emit_cool="0"))
            return out
        pa, _, _ = run("sumflow", huge)
        g10 = [g for g in pa["global_rows"] if g["relation"] == "A10"][0]
        expect(all(r.get("L") is not None for r in pa["rows"]) and "L_sum" not in g10 and "Q_bb_sum" in g10 and
               g10["status"] != "산출" and not lnp_produced(pa),
               "inject two finite shell ledgers whose global sum overflows: the L global sum not formed, shell L "
               "and the Q_bb global sum kept, not produced")
        hv13 = H_PLANCK * TOY_NU[13] / EV_TO_ERG

        def tiny(k, o):
            if k == ("atomic", 0):
                return [dict(r, A_ul="1e-280") if r["line_id"] == "13" else r for r in o]
            if k == ("levels", 0):
                return [dict(r, energy_eV=repr(1.3 * hv13)) if r["level_number"] == "3" else r for r in o]
            return o
        old_delta = globals()["DELTA"]
        globals()["DELTA"] = 1.0e100
        try:
            pa, _, _ = run("ratio", tiny)
        finally:
            globals()["DELTA"] = old_delta
        l13 = lines(pa)[13]
        expect(l13["status"] == "부분" and "F0_minus_F4_over_delta" in l13["reason"] and
               all(l13.get(k) is not None for k in ("F0", "F4", "F0_minus_F4")) and lines(pa)[11]["status"] == "산출"
               and not lnp_produced(pa) and lnp_branch(lnp_produced(pa), True, [False]) == 2,
               "inject a line whose F0 - F4 over Delta is lost to 0 (Delta fixture 1e100): that row partial with "
               "its raw F values kept, not produced, branch 2")
        expect(lnp_branch(True, False, [False, False]) == 1 and lnp_branch(True, True, [False]) == 0 and
               lnp_branch(True, False, [True]) == 0 and lnp_branch(False, True, [True]) == 2 and
               lnp_branch(True, True, [None]) == 2, "branch fixtures (positive control)")
    finally:
        shutil.rmtree(tmp)
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_LINE_NET_PROVENANCE_SELFTEST_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
def cmd_analyze(out, selftest_log=None):
    out = Path(out)
    if out.exists():
        print(f"EXISTS {out}")
        return 3
    gates = {k: [] for k in ("G-INPUT", "G-CALC", "G-RUN")}
    part, cvd = os.environ.get("SLURM_JOB_PARTITION"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if part != "a10" or cvd not in ("", "NoDevFiles"):
        gates["G-RUN"].append(f"NOT_A10_CPU:{part}:{cvd}")
    cfg = REAL
    O, T = cfg["O"], cfg["T"]
    pc = O / "work/physics_comparison"
    small = [pc / f"physics_DET_iter{it:04d}.{k}" for it in (10, 11) for k in ("manifest.json", "shell.csv",
                                                                               "spectral.csv")]
    small += [pc / "physics_DET_iter0009.ion.csv", pc / "physics_DET_iter0010.ion.csv", O / "stderr.log",
              O / "input/model/config.json", O / "input/model/levels.csv"]
    small += [T / "si_trace" / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_cevec.csv", "si10_commit.csv",
                                           "si10_vec.csv", "si10_slmap.csv")]
    before = {str(q): TR.sha(q) for q in small}
    p = lnp_pipeline(cfg)
    for k, v in p["common"].items():
        gates["G-INPUT"] += v[:6]
    gates["G-INPUT"] += [f"LINES:{w}" for w in p["why_lines"][:6]]
    if p["w5"]["notes"]:
        gates["G-INPUT"].append("W5_ROW_WIDTH")
    after = {str(q): TR.sha(q) for q in small}
    if after != before:
        gates["G-INPUT"].append("INPUT_CHANGED_DURING_RUN")
    if any(r.get("telescoping_separated") for r in p["line_rows"]):
        gates["G-CALC"].append("TELESCOPING_IDENTITY")
    st = None
    if selftest_log:
        ls = [ln for ln in Path(selftest_log).read_text().splitlines() if ln.startswith("{")] \
            if Path(selftest_log).is_file() else []
        st = json.loads(ls[-1]) if ls else None
        if not st or st.get("failures") != 0:
            gates["G-CALC"].append("SELFTEST_NOT_PASSED")
    produced = lnp_produced(p)
    unmet = any(str(c["cond_material_history"]).startswith("불충족") for c in p["cond"].values())
    seps = [r.get(f"{a}_minus_{b}_separated") for r in p["line_rows"]
            for a, b in (("F0", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "F4"))]
    all_gates = [f for v in gates.values() for f in v]
    code = lnp_branch(produced and not all_gates, unmet, seps)
    label = {0: "RESOLVED_SCOPED_DIFFERENCE", 1: "NO_RESOLVED_SCOPED_DIFFERENCE", 2: "INDETERMINATE"}[code]
    out.mkdir(parents=True)
    table = p["rows"] + p["line_rows"] + p["global_rows"]
    fields = list(dict.fromkeys(k for r in table for k in r))
    LEC.write_csv(out / "line_net_provenance.csv", table, fields)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    cols = {"shell": ["state", "shell", "L", "A_bb", "E_bb", "Q_bb", "L_minus_Q", "L_minus_Q_over_delta",
                      "L_minus_Q_separated", "producer_material", "w5_support", "w5_rows_used",
                      "w5_max_jbar_residual", "D_rest", "status", "reason"],
            "line": ["line_id", "group", "tau_raw_m13", "srce_chk_m13", "d", "J_c_p", "Jbar_p_W5", "Jbar_star",
                     "J_bin_rad12", "F0", "F1", "F2", "F3", "F4", "F0_minus_F1", "F1_minus_F2", "F2_minus_F3",
                     "F3_minus_F4", "F0_minus_F4_over_delta", "status", "reason"]}
    with open(out / "line_net_provenance.md", "x") as f:
        f.write(f"# Line net-rate ledger provenance, O 19.48 d (branch {code} {label})\n\n")
        f.write(f"Context: Delta (diag9 2500-6000 A deficit) = {DELTA:.6e} erg/s. Positive = radiation to matter. "
                "Full-shell rows use the comparison dumps; the line rows are the registered six lines of A10 shell "
                f"{cfg['shell']} at fixed J_c^p (arithmetic decomposition in the registered order, not causal "
                "shares). D_rest is an unattributed residual. Every definition, generation and condition column is "
                "in the CSV.\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        for st in cfg["states"]:
            f.write(f"## {st} — full shells\n\n| " + " | ".join(cols["shell"]) + " |\n|" +
                    "---|" * len(cols["shell"]) + "\n")
            for r in [r for r in p["rows"] if r["state"] == st]:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols["shell"]) + " |\n")
            f.write("\n")
        f.write(f"## A10 shell {cfg['shell']} — registered lines\n\n| " + " | ".join(cols["line"]) + " |\n|" +
                "---|" * len(cols["line"]) + "\n")
        for r in p["line_rows"]:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols["line"]) + " |\n")
        f.write("\n## Global rows\n\n")
        for g in p["global_rows"]:
            f.write("- " + ", ".join(f"{k}={fmt(v)}" for k, v in g.items()) + "\n")
        f.write("\n## Sameness conditions (prereg 5)\n\n")
        for st, c in p["cond"].items():
            f.write(f"- {st}: " + json.dumps(c, ensure_ascii=False) + "\n")
    big = {str(O / "work/w5_lines.csv"): {"bytes": p["w5"]["bytes"], "sha256": p["w5"]["sha256"]},
           str(O / "input/model/line_list.csv"): {"bytes": p["atomic"][1], "sha256": p["atomic"][0]}}
    tr = p["trace"]
    evidence = {"schema": "DET_LINE_NET_PROVENANCE_V1", "prereg": PREREG, "argv": sys.argv,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "branch": code, "label": label, "gates": gates,
                "produced": produced, "conditions": p["cond"], "producer": p["prod"],
                "registration": p["reg"], "source_reading": SOURCE,
                "w5_check": {c: {"used": v["used"], "excluded": dict(v["excluded"]),
                                 "max_jbar_residual": v["max_jbar_residual"],
                                 "max_beta_residual": v["max_beta_residual"],
                                 "by_shell": {s: {"used": b["used"], "excluded": dict(b["excluded"]),
                                                  "max_jbar_residual": b["max_jbar_residual"],
                                                  "max_beta_residual": b["max_beta_residual"]}
                                              for s, b in v["by_shell"].items()}}
                             for c, v in p["w5"]["check"].items()},
                "w5_generations": p["w5"]["generations"],
                "trace": ({"adopted_call": tr["call"]["call"], "production_call": tr["pcall"]["call"],
                           "ion_sums": tr["sums"],
                           "material13_source": "si10_commit.csv kind=level (= adopted CE after, level by level)"}
                          if tr else None),
                "line_input_findings": p["why_lines"], "common_defects": p["common"],
                "undefined": [{"row": r.get("row_kind"), "state": r.get("state"), "shell": r.get("shell"),
                               "line_id": r.get("line_id"), "reason": r.get("reason")} for r in table
                              if r.get("status") != "산출"],
                "structural_nulls": {"raw_line_e_k_d": "선별 e·k·d 배열·선별 총흡수·총방출 미저장",
                                     "fine_field": "미세장 원배열 미저장 (J_c 재계산 없음)",
                                     "observed_ray_exchange": "관측 광선 교환 미저장"},
                "selftest": {"log": selftest_log, "summary": st},
                "inputs": dict({q: {"bytes": Path(q).stat().st_size, "sha256": h} for q, h in before.items()}, **big),
                "inputs_unchanged": after == before,
                "code": {str(Path(m.__file__).resolve()): TR.sha(Path(m.__file__))
                         for m in (sys.modules[__name__], LEC, LI, TR, EB)}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str, ensure_ascii=False) + "\n")
    print(f"BRANCH {code} {label} produced={produced} lines=" +
          json.dumps({str(r["line_id"]): r["status"] for r in p["line_rows"]}, ensure_ascii=False))
    if all_gates:
        print("GATE FINDINGS: " + "; ".join(all_gates))
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    s = sp.add_parser("analyze")
    s.add_argument("--out", required=True)
    s.add_argument("--selftest-log")
    a = ap.parse_args()
    return cmd_selftest() if a.cmd == "selftest" else cmd_analyze(a.out, a.selftest_log)


if __name__ == "__main__":
    sys.exit(main())
