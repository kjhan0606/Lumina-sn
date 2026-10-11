#!/usr/bin/env python3
"""DET-SI-BF-SENS (prereg docs/PREREG_DET_SI_BF_SENS_2026-10-11.md).

The recorded bound-free photoionization coefficient R_bf of Si II excited levels times q (0.1 or 10) on the recorded
FL operator of the adopted consumption state (material 13, Si shell 10, call 36); the recorded recombination
coefficient (stimulated part included) and every other rate fixed, the steady state re-solved.  Ranges (Si II; the
ground level 0 is never a target):
  M  level_num 6, 7, 9, 10        H  recorded E_eV >= 14.0        A  level_num > 0
States F0, M-0.1, M-10, H-0.1, H-10, A-0.1, A-10, each assembled from the recorded statements.

Subcommands
  negative                    negative controls of the new code (prereg 9)
  analyze --run T --out DIR   G-INPUT, G-RATE, G-SOLVE, G-CLAIM, G-RUN and the main table
Exit codes: 0 pass, 2 refused or gate finding, 3 existing output path.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_si_trace_readout as TR                                                 # noqa: E402
import det_si_excitation_budget as EB                                             # noqa: E402
import det_si_bb_counterfactual as CF                                             # noqa: E402

PREREG = "docs/PREREG_DET_SI_BF_SENS_2026-10-11.md"
ADOPTED, SHELL = EB.ADOPTED, EB.SHELL
M_LEVELS, H_EMIN, N_SI_II = frozenset({6, 7, 9, 10}), 14.0, 125
STATES = ("F0", "M-0.1", "M-10", "H-0.1", "H-10", "A-0.1", "A-10")
PLAN = {"F0": (None, 1.0), "M-0.1": ("M", 0.1), "M-10": ("M", 10.0), "H-0.1": ("H", 0.1), "H-10": ("H", 10.0),
        "A-0.1": ("A", 0.1), "A-10": ("A", 10.0)}
RULES = {"M": "Si II level_num in {6,7,9,10}", "H": "Si II recorded E_eV >= 14.0",
         "A": "Si II level_num > 0 (ground level 0 excluded)"}
R0, FINAL_FL = 3.5597568125822305, 3.563683
PUBLIC = {"si_II": 0.0092499, "si_III": 0.97769}
RTOL, REF_DEX, DEX = 1e-12, 0.001, 0.01
EFF = ("이동 분리", "이동 미분리")
DIST = ("공개 접근", "공개 이격", "공개 거리 변화 미분리")
CHANGED = {"BF ionization"}
NOTE = ("diagnostic finite intervention: the recorded BF photoionization coefficient R_bf of the target Si II levels "
        "times q with the recorded recombination coefficient (stimulated part included) and every other rate fixed; "
        "populations re-solved on the fixed operator, environment and rates not updated; q is not an error estimate, "
        "a CMFGEN correction, a cross-section repair or a radiation-field change")


def close(a, b):
    """prereg 6: relative difference within 1e-12, exact zeros only equal to exact zeros."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    M = max(abs(a), abs(b))
    return abs(a - b) / M <= RTOL if M > 0 else (a == 0.0 and b == 0.0)


# ------------------------------------------------------------------ targets and statements
def bf_ranges(sm, n_si2=N_SI_II, m_levels=M_LEVELS):
    """registered target FL sets from the adopted level map (prereg 3); (ranges, findings).  A finding means the
    membership itself is uncertain, which stops all six interventions (prereg 5)."""
    rows = [r for r in sm if r["upper_ion"] == "0"]
    nums = [int(r["level_num"]) for r in rows]
    why = [] if len(rows) == n_si2 else [f"SI_II_COUNT:{len(rows)}"]
    if len(set(nums)) != len(nums):
        why.append("LEVEL_NUM_REPEATED")
    if nums.count(0) != 1:
        why.append("GROUND_NOT_UNIQUE")
    if not all(math.isfinite(TR.num(r["E_eV"])) for r in rows):
        why.append("ENERGY_NONFINITE")
    rng = {"M": {int(r["i_fl"]) for r in rows if int(r["level_num"]) in m_levels},
           "H": {int(r["i_fl"]) for r in rows if TR.num(r["E_eV"]) >= H_EMIN and int(r["level_num"]) > 0},
           "A": {int(r["i_fl"]) for r in rows if int(r["level_num"]) > 0}}
    if len(rng["M"]) != len(m_levels):
        why.append(f"M_COUNT:{len(rng['M'])}")
    if len(rng["A"]) != n_si2 - 1:
        why.append(f"A_COUNT:{len(rng['A'])}")
    return rng, why


def bf_check(r, x, m, gh):
    """recorded identity and arithmetic of one BF statement (prereg 5); None when its substitution is defined."""
    if int(r["lev_a"]) != int(m["global_level"]) or int(r["src"]) != int(m["sl_local"]) or int(r["dst"]) != gh:
        return "IDENTITY"
    v1, v2 = TR.num(r["v1"]), TR.num(r["v2"])
    if not all(math.isfinite(v) for v in list(x) + [v1, v2]):
        return "NONFINITE"
    R_bf, f_lev, R_rec, I_rec, nstar, nu = x
    if R_bf < 0.0 or R_rec < 0.0 or I_rec < 0.0 or not (0.0 < f_lev <= 1.0):
        return "NEGATIVE_OR_WEIGHT"
    if not (nstar > 0.0 and nu > 0.0):
        return "NSTAR_OR_THRESHOLD"
    # a product of positive factors lost to 0 is a representation failure, not an exact zero
    if (R_bf > 0.0 and R_bf * f_lev == 0.0) or (I_rec > 0.0 and nstar * I_rec == 0.0):
        return "REPRESENTATION"
    if not close(f_lev, TR.num(m["within_sl_frac"])):
        return "WEIGHT_VS_LEVEL_MAP"
    if not (close(v1, R_bf * f_lev) and close(v2, R_rec) and close(R_rec, nstar * I_rec)):
        return "RECORD_ARITHMETIC"
    return None


def bf_link(ops, sm, gh, call):
    """one recorded BF statement of the call per Si II FL level: ({i_fl: (r, x)}, {i_fl: reason}, findings).
    Statements of another call or of no Si II FL level are findings; a missing or repeated level is undefined."""
    fl = {int(r["fl_index"]): r for r in sm}
    si2 = sorted(int(r["i_fl"]) for r in sm if r["upper_ion"] == "0")
    found, why = {}, []
    for r in ops:
        if r["site"] != "5":
            continue
        if r["call"] != call:
            why.append(f"OTHER_CALL:op{r['op']}")
            continue
        m = fl.get(int(r["lev_b"]))
        if m is None or m["upper_ion"] != "0":
            why.append(f"NOT_SI_II_LEVEL:op{r['op']}")
            continue
        found.setdefault(int(m["i_fl"]), []).append((r, [TR.num(r[f"x{k}"]) for k in range(1, 7)]))
    links, undefined = {}, {}
    for i in si2:
        got = found.get(i, [])
        if len(got) != 1:
            undefined[i] = "MISSING" if not got else f"DUPLICATE:{len(got)}"
            continue
        links[i] = got[0]
        w = bf_check(got[0][0], got[0][1], fl[int(got[0][0]["lev_b"])], gh)
        if w:
            undefined[i] = w
    return links, undefined, why


def bf_scaled(r, x, q):
    """a scaled copy of one BF statement: x1' = q x1 and v1' = x1' x2, every other field the recorded string;
    (copy, None) or (None, reason) when a positive rate is lost or not finite (an exact zero stays zero)."""
    R = q * x[0]
    v1 = R * x[1]
    if not (math.isfinite(R) and math.isfinite(v1)):
        return None, "NONFINITE_SUBSTITUTION"
    if x[0] > 0.0 and (R == 0.0 or v1 == 0.0):
        return None, "REPRESENTATION"
    return dict(r, x1=repr(R), v1=repr(v1)), None


def bf_state_ops(ops, links, targets, q):
    """the state's statements: scaled copies for the targets, the recorded objects elsewhere; (ops', undefined)."""
    new, und = {}, {}
    for i in targets:
        r, x = links[i]
        c, w = bf_scaled(r, x, q)
        if c is None:
            und[i] = w
        else:
            new[r["op"]] = c
    return [new.get(r["op"], r) for r in ops], und


def bf_run_state(ops, sm, N, gh, total, links, undefined, targets, q, stop, solve=CF.det_si_bb_solve):
    """one registered state (prereg 5): with a range-level stop or an undefined target substitution nothing is
    assembled or solved and no number is returned; otherwise the scaled operator, its channels and steady state."""
    und = {i: undefined[i] for i in targets if i in undefined}
    und.update({i: "NOT_LINKED" for i in targets if i not in links and i not in und})
    if stop or und:
        return {"produced": False, "undefined": und, "why": list(stop) + (["UNDEFINED_SUBSTITUTION"] if und else [])}
    ops1, und = bf_state_ops(ops, links, targets, q)
    if und:
        return {"produced": False, "undefined": und, "why": ["UNDEFINED_SUBSTITUTION"]}
    A, _, fl_und, ch, maps = TR.det_si_exc_build_fl(ops1, sm, N, gh)
    if fl_und or maps["gh_fl"] is None:
        return {"produced": False, "undefined": {}, "why": ["FL_OPERATOR_UNDEFINED"] + sorted(set(fl_und))}
    x, why, eta = solve(A, total)
    return {"produced": x is not None, "undefined": {}, "why": list(why), "ops": ops1, "A": A, "ch": ch, "x": x,
            "eta_max": eta}


def bf_states(ops, sm, N, gh, total, call, rng, why_r, solve=CF.det_si_bb_solve):
    """the seven registered states from the recorded statements, starting at the BF linkage (prereg 5): a common
    BF defect (any undefined Si II BF statement or a BF statement outside the Si II levels) leaves the F0 operator
    undefined and stops every state, F0 included; an uncertain membership stops the six interventions."""
    links, undefined, why_l = bf_link(ops, sm, gh, call)
    common = ["COMMON_BF_UNDEFINED"] if (undefined or why_l) else []
    stop = common + (["MEMBERSHIP_UNCERTAIN"] if why_r else [])
    st = {"F0": bf_run_state(ops, sm, N, gh, total, links, {}, set(), 1.0, common, solve)}
    for s in STATES[1:]:
        rname, q = PLAN[s]
        st[s] = bf_run_state(ops, sm, N, gh, total, links, undefined, rng[rname], q, stop, solve)
    return st, links, undefined, why_l


def bf_prescription_check(ops, ops1, links, targets, q):
    """statement copies (outputs only): targets carry x1' = q x1 and v1' = x1' x2 (1e-12) with every other field
    the recorded string; every other statement is unchanged.  Returns the offending ops."""
    tops = {links[i][0]["op"] for i in targets}
    bad = []
    for r0, r1 in zip(ops, ops1):
        if r0["op"] not in tops:
            if r1 != r0:
                bad.append(r0["op"])
            continue
        x0, x1 = TR.num(r0["x1"]), TR.num(r1["x1"])
        same = all(r1.get(k) == r0[k] for k in r0 if k not in ("x1", "v1"))
        if not (same and close(x1, q * x0) and close(TR.num(r1["v1"]), x1 * TR.num(r1["x2"]))):
            bad.append(r0["op"])
    return bad + (["LENGTH"] if len(ops) != len(ops1) else [])


def bf_channel_check(ch0, ch1, targets, q):
    """operators: in the BF ionization channel every target column is q times F0 (1e-12, off-diagonal and
    diagonal) and every other column bit-identical; every other channel bit-identical to F0."""
    a0, a1 = ch0.get("BF ionization"), ch1.get("BF ionization")
    if a0 is None or a1 is None or a0.shape != a1.shape:
        return ["BF_CHANNEL_MISSING"]
    why = []
    for j in range(a0.shape[1]):
        if j in targets:
            ok = all(close(float(a1[i, j]), q * float(a0[i, j])) for i in range(a0.shape[0]))
        else:
            ok = np.array_equal(a1[:, j], a0[:, j])
        if not ok:
            why.append(f"BF_COLUMN:{j}")
    return why + [f"INVARIANCE:{k}" for k in CF.det_si_bb_invariance(ch0, ch1, CHANGED)]


# ------------------------------------------------------------------ summaries, classes, branch
def bf_rates(ops1, links):
    """per Si II FL level the photoionization coefficient R_bf consumed by the state."""
    by_op = {r["op"]: r for r in ops1}
    return {i: TR.num(by_op[r["op"]]["x1"]) for i, (r, _) in links.items()}


def bf_summary(x, upper, rates, S):
    """prereg 6 with f_i = n_i / N_II: Gamma_S = sum_S f R', w_S = Gamma_S / sum_SiII f R', P_S = sum_S f.
    None where the Si II total or a needed rate is missing; w None for a zero denominator."""
    lo = [i for i in range(len(x)) if not upper[i]]
    N2 = float(sum(x[i] for i in lo))
    if not N2 > 0.0 or any(i not in rates for i in lo):
        return None
    f = {i: float(x[i]) / N2 for i in lo}
    tot = sum(f[i] * rates[i] for i in lo)
    G = sum(f[i] * rates[i] for i in S)
    return {"Gamma": G, "w": G / tot if tot > 0.0 else None, "P": sum(f[i] for i in S), "total": tot}


def bf_C(x_s, x_0, upper, rates0, S):
    """rate-weighted population change C_S = sum_S f_s R_0 / sum_S f_0 R_0 (None for a zero denominator)."""
    lo = [i for i in range(len(x_s)) if not upper[i]]
    n_s, n_0 = float(sum(x_s[i] for i in lo)), float(sum(x_0[i] for i in lo))
    num = sum(float(x_s[i]) / n_s * rates0[i] for i in S)
    den = sum(float(x_0[i]) / n_0 * rates0[i] for i in S)
    return num / den if den > 0.0 else None


def bf_summary_identity(G_s, G_0, q, C):
    """Gamma_S,s / Gamma_S,0 = q C_S,s for one common factor (prereg 6); None when undefined."""
    if G_0 is None or C is None or not G_0 > 0.0:
        return None
    return close(G_s / G_0, q * C)


def bf_classify(delta, G):
    """prereg 7: boundary values belong to the unresolved class."""
    eff = EFF[0] if abs(delta) > DEX else EFF[1]
    dist = DIST[0] if G > DEX else DIST[1] if G < -DEX else DIST[2]
    return f"{eff}; {dist}"


def bf_branch(valid_all, G_a01, G_a10):
    """0 RESOLVED_APPROACH / 1 NO_RESOLVED_APPROACH need every registered state valid; otherwise 2 INDETERMINATE."""
    if not valid_all or any(g is None or not math.isfinite(g) for g in (G_a01, G_a10)):
        return 2
    return 0 if (G_a01 > DEX or G_a10 > DEX) else 1


def bf_claim_check(rows):
    """a state that is not produced carries no ratio or summary; labels stay in the registered vocabulary; the
    small-response sentence only when every intervention is unresolved."""
    bad = []
    for r in rows:
        if r.get("kind") != "state":
            continue
        if not r.get("produced") and any(r.get(k) is not None for k in ("III_II", "Gamma_S", "P_S", "C_S")):
            bad.append(f"PARTIAL_AS_FULL:{r['state']}")
        c = r.get("cls")
        p = c.split("; ") if isinstance(c, str) else []
        if not (c in ("기준", "미산출") or (len(p) == 2 and p[0] in EFF and p[1] in DIST)):
            bad.append(f"LABEL:{r['state']}")
        if r.get("summary_identity") is False:
            bad.append(f"SUMMARY_IDENTITY:{r['state']}")
    return bad


# ------------------------------------------------------------------ negative controls (prereg 9)
def bf_toy():
    """Si II FL 0..5 (level_num 0, 1, 6, 7, 40, 41; E 0, 0.036, 6.86, 13.999, 14.0, 14.5 eV; FL 4 and 5 share SL 4
    with weights 0.6 / 0.4) and Si III FL 6..7 (upper ground SL 5); collisional links keep the operator connected;
    one consistent BF statement per Si II level."""
    lv = ((0, 0, 0.0, 2, 0, 1.0), (1, 0, 0.036, 4, 1, 1.0), (6, 0, 6.86, 2, 2, 1.0), (7, 0, 13.999, 4, 3, 1.0),
          (40, 0, 14.0, 6, 4, 0.6), (41, 0, 14.5, 4, 4, 0.4), (0, 1, 0.0, 1, 5, 1.0), (1, 1, 6.6, 3, 6, 1.0))
    sm = [{"call": "1", "i_fl": str(i), "fl_index": str(i), "sl_local": str(sl), "within_sl_frac": repr(fr),
           "global_level": str(100 + i), "level_num": str(ln), "upper_ion": str(u), "E_eV": repr(E), "g": str(g)}
          for i, (ln, u, E, g, sl, fr) in enumerate(lv)]
    N, gh, ops = 7, 5, []

    def mk(site, dst, src, v1, v2, la, lb, x):
        r = {"call": "1", "op": str(len(ops) + 1), "site": str(site), "dst": str(dst), "src": str(src),
             "v1": repr(float(v1)), "v2": repr(float(v2)), "lev_a": str(la), "lev_b": str(lb), "aux": "0"}
        r.update({f"x{j + 1}": (repr(float(x[j])) if j < len(x) else "") for j in range(14)})
        ops.append(r)
    for up, lo, cu, cd in ((1, 0, 3.0, 2.0), (2, 0, 1.0, 4.0), (3, 2, 2.0, 5.0), (4, 3, 1.5, 2.5), (5, 4, 2.0, 1.0),
                           (7, 6, 0.5, 3.0)):
        mk(2, int(sm[up]["sl_local"]), int(sm[lo]["sl_local"]), cu, cd, int(sm[up]["level_num"]),
           int(sm[lo]["level_num"]), (cu, cd))
    for i, (R_bf, I_rec, nstar) in enumerate(((1e-3, 0.2, 0.5), (2e-3, 0.1, 0.5), (0.5, 0.3, 0.25), (1.0, 0.2, 0.25),
                                              (3.0, 0.1, 0.5), (5.0, 0.05, 0.5))):
        f = float(sm[i]["within_sl_frac"])
        R_rec = nstar * I_rec
        mk(5, gh, int(sm[i]["sl_local"]), R_bf * f, R_rec, 100 + i, i, (R_bf, f, R_rec, I_rec, nstar, 1e15))
    return {"N": N, "gh": gh, "sm": sm, "ops": ops, "total": 10.0, "upper": np.array([r["upper_ion"] == "1"
                                                                                      for r in sm])}


def cmd_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    t = bf_toy()
    N, gh, sm, ops, total, upper = t["N"], t["gh"], t["sm"], t["ops"], t["total"], t["upper"]
    counts = {"n_si2": 6, "m_levels": frozenset({6, 7})}
    rng, why_r = bf_ranges(sm, **counts)
    links, und, why_l = bf_link(ops, sm, gh, "1")
    expect(not why_r and not und and not why_l and len(links) == 6 and rng["M"] == {2, 3} and rng["H"] == {4, 5}
           and rng["A"] == {1, 2, 3, 4, 5}, "toy ranges and links (positive control; H boundary 14.0 inclusive, "
           "13.999 outside, ground 0 excluded, level 1 in A)")
    f0 = bf_run_state(ops, sm, N, gh, total, links, und, set(), 1.0, [])
    ok = f0["produced"]
    for s in STATES[1:]:
        rname, q = PLAN[s]
        res = bf_run_state(ops, sm, N, gh, total, links, und, rng[rname], q, [])
        ok = ok and res["produced"] and not bf_prescription_check(ops, res["ops"], links, rng[rname], q) and \
            not bf_channel_check(f0["ch"], res["ch"], rng[rname], q) and CF.det_si_bb_channel_sum_ok(res["A"],
                                                                                                    res["ch"])
    expect(ok, "toy F0 and six states produced, prescription and channel checks hold (positive control)")
    one = bf_run_state(ops, sm, N, gh, total, links, und, rng["A"], 1.0, [])
    expect(np.array_equal(one["A"], f0["A"]) and not bf_channel_check(f0["ch"], one["ch"], rng["A"], 1.0),
           "factor 1 leaves the operator bit-identical (positive control)")

    # 1. other call, ion, level, direction, duplicate and missing statements
    bf2 = [r for r in ops if r["site"] == "5"][2]                     # FL 2 (level_num 6)
    other_call = ops + [dict(bf2, call="2", op="90")]
    expect(any("OTHER_CALL" in w for w in bf_link(other_call, sm, gh, "1")[2]), "inject a statement of another "
           "call: refused")
    expect(any("NOT_SI_II_LEVEL" in w for w in bf_link(ops + [dict(bf2, op="91", lev_b="6")], sm, gh, "1")[2]),
           "inject a statement on a Si III level: refused")
    expect(bf_link([dict(r, lev_a="999") if r is bf2 else r for r in ops], sm, gh, "1")[1].get(2) == "IDENTITY",
           "inject another atomic level: undefined")
    expect(bf_link([dict(r, dst="6") if r is bf2 else r for r in ops], sm, gh, "1")[1].get(2) == "IDENTITY",
           "inject a direction to another upper-ion level: undefined")
    expect(bf_link([dict(r, src="3") if r is bf2 else r for r in ops], sm, gh, "1")[1].get(2) == "IDENTITY",
           "inject another SL: undefined")
    expect(bf_link(ops + [dict(bf2, op="92")], sm, gh, "1")[1].get(2) == "DUPLICATE:2", "inject a duplicate "
           "statement: undefined")
    expect(bf_link([r for r in ops if r is not bf2], sm, gh, "1")[1].get(2) == "MISSING", "inject a missing "
           "statement: undefined (not a zero rate)")
    # 2. range membership
    rep = [dict(r, level_num="6") if r["i_fl"] == "5" else r for r in sm]
    expect("LEVEL_NUM_REPEATED" in bf_ranges(rep, **counts)[1], "inject a repeated level_num: membership uncertain")
    expect(bool(bf_ranges([r for r in sm if r["i_fl"] != "3"], **counts)[1]), "inject a missing Si II level: "
           "membership uncertain")
    bad_rng = bf_run_state(ops, sm, N, gh, total, links, und, rng["H"], 10.0, [])
    expect(bool(bf_channel_check(f0["ch"], bad_rng["ch"], rng["M"], 10.0)), "inject an operator scaled on another "
           "range than the registered one: refused")
    # 3. recombination scaled, f_lev applied twice, one matrix term, non-target, accumulation
    m10 = bf_run_state(ops, sm, N, gh, total, links, und, rng["M"], 10.0, [])
    rec = [dict(r, x3=repr(10 * TR.num(r["x3"])), v2=repr(10 * TR.num(r["v2"])))
           if r["op"] in {links[i][0]["op"] for i in rng["M"]} else r for r in m10["ops"]]
    _, _, _, ch_rec, _ = TR.det_si_exc_build_fl(rec, sm, N, gh)
    expect(any("INVARIANCE" in w for w in bf_channel_check(f0["ch"], ch_rec, rng["M"], 10.0)),
           "inject recombination scaled with photoionization: refused")
    h10 = bf_run_state(ops, sm, N, gh, total, links, und, rng["H"], 10.0, [])
    dbl = [dict(r, x1=repr(TR.num(r["x1"]) * TR.num(r["x2"]))) if r["op"] in {links[i][0]["op"] for i in rng["H"]}
           else r for r in h10["ops"]]
    _, _, _, ch_dbl, _ = TR.det_si_exc_build_fl(dbl, sm, N, gh)
    expect(bool(bf_channel_check(f0["ch"], ch_dbl, rng["H"], 10.0)) and
           bool(bf_prescription_check(ops, dbl, links, rng["H"], 10.0)), "inject the SL weight f_lev applied twice "
           "(FL 4, 5 with 0.6 / 0.4): refused")
    half = {k: v.copy() for k, v in m10["ch"].items()}
    j = sorted(rng["M"])[0]
    half["BF ionization"][j, j] = f0["ch"]["BF ionization"][j, j]
    expect(any("BF_COLUMN" in w for w in bf_channel_check(f0["ch"], half, rng["M"], 10.0)), "inject only the "
           "off-diagonal term scaled (diagonal left): refused")
    over = bf_run_state(ops, sm, N, gh, total, links, und, rng["M"] | {1}, 10.0, [])
    expect(bool(bf_channel_check(f0["ch"], over["ch"], rng["M"], 10.0)) and
           bool(bf_prescription_check(ops, over["ops"], links, rng["M"], 10.0)), "inject a change on a non-target "
           "level: refused")
    acc_ops, _ = bf_state_ops(m10["ops"], {i: (next(r for r in m10["ops"] if r["op"] == links[i][0]["op"]),
                                               links[i][1]) for i in links}, rng["H"], 10.0)
    _, _, _, ch_acc, _ = TR.det_si_exc_build_fl(acc_ops, sm, N, gh)
    expect(bool(bf_channel_check(f0["ch"], ch_acc, rng["H"], 10.0)) and
           bool(bf_prescription_check(ops, acc_ops, links, rng["H"], 10.0)), "inject H-10 built on M-10 "
           "(accumulated states): refused")
    # 4. negative, non-finite, representation, exact zero
    r5, x5 = links[2]
    m2 = sm[2]
    expect(bf_check(r5, [-1.0] + x5[1:], m2, gh) == "NEGATIVE_OR_WEIGHT", "inject a negative rate: undefined")
    expect(bf_check(r5, [float("nan")] + x5[1:], m2, gh) == "NONFINITE", "inject a non-finite rate: undefined")
    expect(bf_check(r5, x5[:4] + [0.0] + x5[5:], m2, gh) == "NSTAR_OR_THRESHOLD", "inject n* = 0: undefined")
    expect(bf_check(r5, x5[:2] + [2.0 * x5[2]] + x5[3:], m2, gh) == "RECORD_ARITHMETIC", "inject R_rec != n* I_rec: "
           "undefined")
    expect(bf_check(dict(r5, v1=repr(2.0 * TR.num(r5["v1"]))), x5, m2, gh) == "RECORD_ARITHMETIC",
           "inject v1 != R_bf f_lev: undefined")
    expect(bf_check(r5, x5, dict(m2, within_sl_frac="0.5"), gh) == "WEIGHT_VS_LEVEL_MAP", "inject a weight other than "
           "the level map: undefined")
    expect(bf_scaled(r5, [5e-324] + x5[1:], 0.1)[1] == "REPRESENTATION", "inject a positive rate lost to 0 by the "
           "factor: undefined")
    z, w = bf_scaled(dict(r5, x1="0.0", v1="0.0"), [0.0] + x5[1:], 10.0)
    expect(w is None and TR.num(z["x1"]) == 0.0 and TR.num(z["v1"]) == 0.0, "exact recorded zero stays zero "
           "(positive control)")
    expect(bf_scaled(r5, [1e308] + x5[1:], 10.0)[1] == "NONFINITE_SUBSTITUTION", "inject an overflowing product: "
           "undefined")
    # 5. the state execution path does not solve or return numbers when undefined or stopped
    calls = []

    def spy(A, tot):
        calls.append(1)
        return CF.det_si_bb_solve(A, tot)
    und_t = {**und, 2: "IDENTITY"}
    r_u = bf_run_state(ops, sm, N, gh, total, links, und_t, rng["M"], 10.0, [], solve=spy)
    r_s = bf_run_state(ops, sm, N, gh, total, links, und, rng["A"], 10.0, ["MEMBERSHIP_UNCERTAIN"], solve=spy)
    r_ok = bf_run_state(ops, sm, N, gh, total, links, und, rng["M"], 10.0, [], solve=spy)
    expect(not r_u["produced"] and not r_s["produced"] and "x" not in r_u and "x" not in r_s and "A" not in r_u
           and len(calls) == 1 and r_ok["produced"], "inject an undefined target / an uncertain membership: the state "
           "function neither assembles nor solves nor returns numbers (the defined state solves once)")
    calls.clear()
    good_st = bf_states(ops, sm, N, gh, total, "1", rng, [], solve=spy)[0]
    n_good = len(calls)
    calls.clear()
    bf0 = [r for r in ops if r["site"] == "5"][0]                     # ground level 0: never a target
    bad_ops = [dict(r, dst="6") if r is bf0 else r for r in ops]
    bad_st, _, bad_und, _ = bf_states(bad_ops, sm, N, gh, total, "1", rng, [], solve=spy)
    expect(all(v["produced"] for v in good_st.values()) and n_good == 7 and bad_und.get(0) == "IDENTITY" and
           not calls and not any(v["produced"] or "x" in v or "A" in v for v in bad_st.values()),
           "inject a ground-level BF direction defect at the linkage (non-target): every state, F0 included, "
           "neither assembles nor solves nor returns numbers")
    r_g, x_g = links[5]                                               # FL 5, weight 0.4
    lost = bf_check(dict(r_g, v1="0.0"), [5e-324] + x_g[1:], sm[5], gh)
    lost_rec = bf_check(dict(r_g, v2="0.0"), x_g[:2] + [0.0, 1e-200, 1e-200] + x_g[5:], sm[5], gh)
    exact = bf_check(dict(r_g, v2="0.0"), x_g[:2] + [0.0, 0.0] + x_g[4:], sm[5], gh)
    expect(lost == "REPRESENTATION" and lost_rec == "REPRESENTATION" and exact is None,
           "inject recorded products of positive factors lost to 0 (R_bf f_lev, n* I_rec): undefined; an exact zero "
           "I_rec stays valid")
    # 6. summaries: P against the rate-weighted C
    x0 = np.array([8.0, 1.0, 0.5, 0.01, 0.001, 0.001, 0.4, 0.088])
    xs = np.array([8.0, 1.0, 0.25, 0.02, 0.001, 0.001, 0.6, 0.128])
    r0 = {0: 1e-3, 1: 2e-3, 2: 0.5, 3: 100.0, 4: 3.0, 5: 5.0}
    S = {2, 3}
    Pr = bf_summary(xs, upper, r0, S)["P"] / bf_summary(x0, upper, r0, S)["P"]
    C = bf_C(xs, x0, upper, r0, S)
    rs = {i: (10.0 * v if i in S else v) for i, v in r0.items()}
    Gs, G0 = bf_summary(xs, upper, rs, S)["Gamma"], bf_summary(x0, upper, r0, S)["Gamma"]
    expect(not close(Pr, C) and bf_summary_identity(Gs, G0, 10.0, C) and not bf_summary_identity(Gs, G0, 10.0, Pr),
           "inject the population ratio P in place of the rate-weighted C: the summary identity refuses it")
    # 7. classes, branch, claim
    expect(bf_classify(0.01, 0.01) == "이동 미분리; 공개 거리 변화 미분리" and
           bf_classify(-0.02, -0.02) == "이동 분리; 공개 이격", "boundary values unresolved (positive control)")
    expect([bf_branch(True, -0.5, 0.02), bf_branch(True, -0.5, 0.01), bf_branch(False, 0.5, 0.5),
            bf_branch(True, None, 0.5)] == [0, 1, 2, 2], "branch fixtures (positive control)")
    rows = [{"kind": "state", "state": "M-10", "produced": False, "III_II": 3.0, "cls": "미산출"},
            {"kind": "state", "state": "H-10", "produced": True, "III_II": 3.0, "cls": "effect resolved"},
            {"kind": "state", "state": "A-10", "produced": True, "III_II": 3.0, "cls": "기준",
             "summary_identity": False}]
    expect(bf_claim_check(rows) == ["PARTIAL_AS_FULL:M-10", "LABEL:H-10", "SUMMARY_IDENTITY:A-10"],
           "inject a stopped state with a ratio, an unregistered label and a broken summary identity: refused")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_SI_BF_SENS_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
def cmd_analyze(run, out):
    run = Path(run)
    tr, pc = run / "si_trace", run / "work/physics_comparison"
    out = TR.new_dir(out)
    gates = {k: [] for k in ("G-INPUT", "G-RATE", "G-SOLVE", "G-CLAIM", "G-RUN")}
    part, cvd = os.environ.get("SLURM_JOB_PARTITION"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if part != "a10" or cvd not in ("", "NoDevFiles"):
        gates["G-RUN"].append(f"NOT_A10_CPU:{part}:{cvd}")
    footer = dict(ln.split("=", 1) for ln in (run / "RUN_FOOTER.txt").read_text().splitlines() if "=" in ln)
    stop_iter = int(footer.get("stop_iteration", "-1"))
    if footer.get("terminal_state") != "TRACE_TARGET_COMPLETE" or stop_iter < 0:
        gates["G-INPUT"].append("TRACE_RUN_NOT_COMPLETE")

    # ---- G-INPUT: adoption, replay, level map, ranges, BF statements, public row
    call, why = EB.det_si_exc_adopt(TR.read_trace(tr / "si10_ne.csv"), TR.read_trace(tr / "si10_calls.csv"),
                                    TR.read_trace(tr / "si10_cevec.csv"), TR.read_trace(tr / "si10_commit.csv"),
                                    ADOPTED)
    if call is None:
        return TR.refuse(why)
    cid = call["call"]
    N, N_fl, gh = int(call["N"]), int(call["N_fl"]), int(call["ground_hi"])
    if int(call["rad_generation"]) != 12 or int(call["core_generation"]) != 13:
        gates["G-INPUT"].append("GENERATIONS")
    ops = [r for r in TR.read_trace(tr / "si10_ops.csv") if r["call"] == cid]
    g_a, g_b, _ = TR.read_matrix(tr / f"si10_mat_{int(cid):06d}_gen.bin")
    r_a, r_b = TR.replay(ops, N)
    if not (np.array_equal(r_a, g_a) and np.array_equal(r_b, g_b)):
        gates["G-INPUT"].append("REPLAY_NOT_GENERATOR")
    sm = sorted((r for r in TR.read_trace(tr / "si10_slmap.csv") if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    if len(sm) != N_fl:
        return TR.refuse(["LEVEL_MAP"])
    total = float(next(r for r in TR.read_trace(tr / "si10_final.csv") if r["call"] == cid)["n_total"])
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    rng, why_r = bf_ranges(sm)
    if why_r:
        gates["G-INPUT"].append("MEMBERSHIP:" + ";".join(why_r))
    # ---- states from the BF linkage on (a common BF defect stops every state, F0 included)
    st, links, undefined, why_l = bf_states(ops, sm, N, gh, total, cid, rng, why_r)
    if why_l or undefined:
        gates["G-INPUT"].append(f"BF_STATEMENTS:{len(why_l)}:{len(undefined)}:" +
                                ";".join(why_l[:3] + [f"fl{i}:{w}" for i, w in list(undefined.items())[:3]]))
    t_shell = TR.dump_row(run, stop_iter, "shell", SHELL)[0]
    vel = (float(t_shell["v_inner_cm_s"]) + float(t_shell["v_outer_cm_s"])) / 2e5
    pub, pub_vel, pub_ii, pub_iii = EB.det_si_exc_public(vel)
    if not (math.isclose(vel, TR.REGISTERED_VEL["lumina_vel_mid_kms"]) and
            math.isclose(pub_vel, TR.REGISTERED_VEL["public_vel_mid_kms"]) and pub == pub_iii / pub_ii and
            (pub_ii, pub_iii) == (PUBLIC["si_II"], PUBLIC["si_III"])):
        gates["G-INPUT"].append(f"PUBLIC_CORRESPONDENCE:{vel}->{pub_vel}:{pub_ii}:{pub_iii}")

    # ---- G-RATE, G-SOLVE (each state assembled from the recorded statements)
    for s in STATES[1:]:
        rname, q = PLAN[s]
        if "ch" in st[s] and "ch" in st["F0"]:
            bad = bf_prescription_check(ops, st[s]["ops"], links, rng[rname], q)
            if bad:
                gates["G-RATE"].append(f"PRESCRIPTION:{s}:{len(bad)}")
            bad = bf_channel_check(st["F0"]["ch"], st[s]["ch"], rng[rname], q)
            if bad:
                gates["G-RATE"].append(f"CHANNELS:{s}:" + ",".join(bad[:5]))
    for s, res in st.items():
        if "A" in res:
            if not CF.det_si_bb_channel_sum_ok(res["A"], res["ch"]):
                gates["G-RATE"].append(f"CHANNEL_SUM:{s}")
            if EB.det_si_exc_conservation(res["A"]):
                gates["G-RATE"].append(f"CONSERVATION:{s}")
            if res["why"]:
                gates["G-SOLVE"].append(f"{s}:" + ",".join(res["why"]))
    R = {}
    for s, res in st.items():
        if res["produced"]:
            ratio = TR.ratio(res["x"], upper)
            if math.isfinite(ratio) and ratio > 0.0:
                R[s] = ratio
            else:
                res["produced"] = False
                res["why"] = res["why"] + ["RATIO_UNDEFINED"]
                gates["G-SOLVE"].append(f"{s}:RATIO_UNDEFINED")
    ref_dev = TR.dex(R.get("F0", float("nan")), R0)
    if not (abs(ref_dev) <= REF_DEX):
        gates["G-SOLVE"].append(f"F0_REFERENCE:{R.get('F0')}")

    # ---- the table
    rates0 = bf_rates(ops, links)
    x0 = st["F0"].get("x")
    d0 = TR.dex(R["F0"], pub) if "F0" in R else float("nan")
    rows = [{"kind": "public", "state": "CMFGEN toy06 19.480 d", "range": "public Si II / Si III ion fractions",
             "III_II": pub, "si_II": pub_ii, "si_III": pub_iii, "vel_kms": pub_vel}]
    G = {}
    for s in STATES:
        rname, q = PLAN[s]
        res = st[s]
        tg = rng[rname] if rname else set()
        und_s = res.get("undefined", {})
        row = {"kind": "state", "state": s, "range": RULES[rname] if rname else "none (recorded operator)", "q": q,
               "planned": len(tg), "definable": len(tg) - len(und_s), "substituted": len(tg) if "A" in res else 0,
               "undefined": len(und_s), "stop_reasons": ";".join(res["why"]) if not res["produced"] else "",
               "produced": res["produced"], "eta_max": res.get("eta_max")}
        if s in R:
            d = TR.dex(R[s], pub)
            row.update(III_II=R[s], delta_F0=TR.dex(R[s], R["F0"]) if "F0" in R else None, d_pub=d,
                       position="위" if d > DEX else "아래" if d < -DEX else "일치(보고 분해능 이내)")
            if s == "F0":
                row.update(cls="기준", reference_dex=ref_dev)
                for k in ("M", "H", "A"):
                    sm0 = bf_summary(res["x"], upper, rates0, rng[k])
                    row.update({f"Gamma_{k}": sm0 and sm0["Gamma"], f"w_{k}": sm0 and sm0["w"],
                                f"P_{k}": sm0 and sm0["P"], f"n_{k}": len(rng[k])})
                row["Gamma_total"] = (bf_summary(res["x"], upper, rates0, set()) or {}).get("total")
            elif "F0" in R:
                G[s] = abs(d0) - abs(d)
                ss = bf_summary(res["x"], upper, bf_rates(res["ops"], links), tg)
                s0 = bf_summary(x0, upper, rates0, tg)
                C = bf_C(res["x"], x0, upper, rates0, tg)
                row.update(G=G[s], cls=bf_classify(row["delta_F0"], G[s]))
                if ss and s0:
                    row.update(Gamma_S=ss["Gamma"], w_S=ss["w"], P_S=ss["P"], Gamma_total=ss["total"],
                               Gamma_ratio=ss["Gamma"] / s0["Gamma"] if s0["Gamma"] > 0.0 else None,
                               P_ratio=ss["P"] / s0["P"] if s0["P"] > 0.0 else None, C_S=C,
                               summary_identity=bf_summary_identity(ss["Gamma"], s0["Gamma"], q, C))
        else:
            row.update(cls="미산출")
        rows.append(row)
    gates["G-CLAIM"] += bf_claim_check(rows)
    sub = [r for r in rows if r["kind"] == "state" and r["state"] != "F0"]
    produced = all(r["state"] in R for r in sub)
    small = produced and all(r["cls"].startswith(EFF[1]) for r in sub)
    a_small = produced and all(r["cls"].startswith(EFF[1]) for r in sub if r["state"].startswith("A-"))
    offset = a_small and any(r["cls"].startswith(EFF[0]) for r in sub if not r["state"].startswith("A-"))
    summary = ("등록한 범위와 두 배율에서 III/II 반응이 작다(여섯 개입 모두 이동 미분리)" if small else
               "A 반응은 작으나 일부 범위가 이동 분리 — 상쇄·재분배 가능성" if offset else
               "III/II 반응이 작다고 보고하지 않음(이동 분리 또는 미산출 상태 있음)")
    all_gates = [f for v in gates.values() for f in v]
    valid_all = not all_gates and all(s in R for s in STATES)
    code = bf_branch(valid_all, G.get("A-0.1"), G.get("A-10"))
    label = {0: "RESOLVED_APPROACH", 1: "NO_RESOLVED_APPROACH", 2: "INDETERMINATE"}[code]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "si10_bf_sensitivity_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    context = {"F0_R0": R0, "FINAL_FL_III_II_context_only": FINAL_FL, "public_si_II": pub_ii, "public_si_III": pub_iii,
               "public_III_II": pub, "time_d": 19.48, "vel_kms": f"Lumina {vel:.0f} <-> public {pub_vel}",
               "H_levels": len(rng["H"]), "H_rule": f"E_eV >= {H_EMIN}"}
    with open(out / "si10_bf_sensitivity_table.md", "x") as f:
        f.write(f"# Si II excited-level BF photoionization sensitivity, Si shell 10 (branch {code} {label})\n\n")
        f.write("Context (quoted): " + json.dumps(context, ensure_ascii=False) + "\n\n")
        f.write("Scope: " + NOTE + "\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        cols = ["state", "range", "q", "planned", "definable", "substituted", "undefined", "III_II", "delta_F0",
                "d_pub", "position", "G", "Gamma_S", "w_S", "P_S", "Gamma_ratio", "P_ratio", "C_S", "Gamma_total",
                "cls", "stop_reasons"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
        f0row = next(r for r in rows if r.get("state") == "F0")
        f.write("\nF0 baselines: " + ", ".join(f"{k}: n {f0row.get(f'n_{k}')}, Gamma {fmt(f0row.get(f'Gamma_{k}'))}, "
                                              f"w {fmt(f0row.get(f'w_{k}'))}, P {fmt(f0row.get(f'P_{k}'))}"
                                              for k in ("M", "H", "A")) + "\n")
        f.write(f"\nSummary: {summary}\n\nNotes: delta_F0 = log10(R/R_F0); d_pub = log10(R/R_pub); G = |d_F0| - |d|; "
                "classes at 0.01 dex (reporting resolution, boundary unresolved); Gamma_S = sum_S f R' per Si II "
                "(f = n / N_II), w_S = Gamma_S / Gamma_total, P_S = sum_S f, C_S = sum_S f_s R_0 / sum_S f_0 R_0, "
                "Gamma_ratio = q C_S.\n")
    used = [tr / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_commit.csv", "si10_cevec.csv", "si10_ops.csv",
                             "si10_slmap.csv", "si10_final.csv", f"si10_mat_{int(cid):06d}_gen.bin")]
    used += [run / "RUN_FOOTER.txt", pc / f"physics_DET_iter{stop_iter:04d}.shell.csv", TR.PUBLIC_SI, TR.DIAG8_PUBLIC]
    fl_row = {int(r["i_fl"]): r for r in sm}
    state_ev = {}
    for s, res in st.items():
        rname, q = PLAN[s]
        tg = sorted(rng[rname]) if rname else []
        e = {"produced": res["produced"], "why": res["why"], "eta_max": res.get("eta_max"), "q": q, "targets": tg}
        if "ops" in res:
            by_op = {r["op"]: r for r in res["ops"]}
            e["substituted"] = [{"i_fl": i, "op": links[i][0]["op"], "x1": TR.num(by_op[links[i][0]["op"]]["x1"]),
                                 "v1": TR.num(by_op[links[i][0]["op"]]["v1"])} for i in tg]
        if res.get("x") is not None:
            e["populations"] = [float(v) for v in res["x"]]
            e["N_II"], e["N_III"] = float(res["x"][~upper].sum()), float(res["x"][upper].sum())
        state_ev[s] = e
    evidence = {"schema": "DET_SI_BF_SENS_V1", "prereg": PREREG, "argv": sys.argv,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "branch": code, "label": label, "gates": gates,
                "summary": summary, "note": NOTE, "context": context,
                "adopted": {k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call", "rad_generation",
                                                       "core_generation")},
                "ranges": {k: sorted(v) for k, v in rng.items()}, "membership_findings": why_r,
                "bf_findings": why_l, "undefined": {str(i): w for i, w in undefined.items()},
                "ratios": R, "reference_dex": ref_dev, "states": state_ev,
                "fl_levels": [{"i_fl": int(r["i_fl"]), "level_num": int(r["level_num"]), "upper_ion": r["upper_ion"],
                               "E_eV": TR.num(r["E_eV"]), "sl_local": int(r["sl_local"]),
                               "within_sl_frac": TR.num(r["within_sl_frac"])} for r in sm],
                "statements": [{"i_fl": i, "level_num": int(fl_row[i]["level_num"]), "E_eV": TR.num(fl_row[i]["E_eV"]),
                                "op": r["op"], "R_bf": x[0], "f_lev": x[1], "R_rec": x[2], "I_rec": x[3],
                                "n_star_ratio": x[4], "nu_thresh": x[5],
                                "targets": [s for s in STATES[1:] if i in rng[PLAN[s][0]]]}
                               for i, (r, x) in sorted(links.items())],
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(p).resolve()): TR.sha(Path(p)) for p in (__file__, TR.__file__, EB.__file__,
                                                                           CF.__file__)}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str, ensure_ascii=False) + "\n")
    print(f"BRANCH {code} {label} R={json.dumps(R)} G={json.dumps(G)}")
    if all_gates:
        print("GATE FINDINGS: " + "; ".join(all_gates))
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("negative")
    s = sp.add_parser("analyze")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_negative() if a.cmd == "negative" else cmd_analyze(a.run, a.out)


if __name__ == "__main__":
    sys.exit(main())
