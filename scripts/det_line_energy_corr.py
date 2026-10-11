#!/usr/bin/env python3
"""DET-LINE-ENERGY-CORR (prereg docs/PREREG_DET_LINE_ENERGY_CORR_2026-10-11.md).

Shell-by-shell correspondence, reference configuration O at 19.48 d, between the fixed-T thermal ledger of the
comparison dumps and the radiative energy exchange of the stored transport coefficients:
  A10  iter0010 dump (material 13 / radiation 12)      A11  iter0011 dump (material 14 / radiation 13)
  X13  A10 coefficients with the A11 J (mixed history, arithmetic substitution)
  C13  observation-capture bin representation (diag9 cells.csv; material 13 / radiation 13)
  Q_x = 4 pi V sum_b (chi_x J - eta_x) dnu (x = bb, bf, ff; positive = radiation to matter), L = V (line_abs_heat -
  line_emit_cool); relations D10 (L10 vs Q_bb A10), D11, J13 (A10 vs X13), O13 (X13 vs C13), plus the mandatory
  BF / FF identities V (photo_heat - recomb_cool) = Q_bf, V (ff_abs_heat - ff_emit_cool) = Q_ff.

Subcommands
  selftest                negative controls of the new code (prereg 9)
  analyze --out DIR       G-INPUT, G-CALC, G-RUN and the main table
Exit codes: 0 pass, 2 refused or gate finding, 3 existing output path.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

D9 = Path("/gpfs/kjhan/lumina/cmfgen_map/diag9_absdef_20261011")
sys.path.insert(0, str(D9 / "tools"))
from r1_common import read_json, sha, write_csv, write_json                       # noqa: E402

PREREG = "docs/PREREG_DET_LINE_ENERGY_CORR_2026-10-11.md"
O = Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1")
PC = O / "work/physics_comparison"
CAP = O / "work/det_obs_capture"
CELLS = D9 / "out/r1_bridge_419630/cells.csv"
BRIDGE = D9 / "out/r1_bridge_419630/result.json"
DELTA = 8.308031852615979e42          # diag9 target deficit 2500-6000 A (erg/s), quoted context
TOL = 1e-10
NS, NB = 50, 1234
FOUR_PI = 4.0 * math.pi
GEN = {"A10": {"population_generation": 13, "radiation_generation": 12},
       "A11": {"population_generation": 14, "radiation_generation": 13}}
GRID = {"frame": "SHELL_COMOVING", "frequency_coordinate": "HZ",
        "frequency_regrid": "INTEGRAL_PRESERVING_PIECEWISE_CONSTANT", "n_bins": NB, "n_shells": NS}
LEDGER = ("q_ad_heating", "photo_heat", "line_abs_heat", "ff_abs_heat", "compton_heat", "gamma_heat",
          "nonthermal_heat", "q_ad_cooling", "recomb_cool", "line_emit_cool", "coll_line_cool", "ff_emit_cool",
          "compton_cool", "sum_heating", "sum_cooling", "residual")
COMP = {"bb": ("chi_bb_cm1", "eta_bb"), "bf": ("chi_bf_cm1", "eta_bf"), "ff": ("chi_ff_cm1", "eta_ff")}
ZERO_TEXT = re.compile(r"^[+-]?0*\.?0*([eE][+-]?\d+)?$")


# ------------------------------------------------------------------ parsing
def lec_num(text, why, where):
    """a stored decimal; non-finite, unparsable or a non-zero text read as 0 (representation loss) is recorded."""
    try:
        v = float(text)
    except (TypeError, ValueError):
        why.append(f"UNPARSABLE:{where}")
        return float("nan")
    if not math.isfinite(v):
        why.append(f"NONFINITE:{where}")
    elif v == 0.0 and not ZERO_TEXT.match(text.strip()):
        why.append(f"TEXT_LOST_TO_ZERO:{where}")
        return float("nan")
    return v


def lec_volume(ri, ro):
    return FOUR_PI / 3.0 * (ro ** 3 - ri ** 3)


# ------------------------------------------------------------------ integrals
def lec_integrate(V, dnu, chi, eta, J, bad):
    """per shell: A = 4 pi V sum chi J dnu, E = 4 pi V sum eta dnu, Q = A - E and the pre-cancellation scale
    S = 4 pi V sum (|chi J| + |eta|) dnu.  Signed chi is kept.  A shell with an undefined bin (bad), a negative
    J or emissivity, a non-positive width or volume, a non-finite value or a product of non-zero factors lost to
    zero is None with its reason (prereg 7); no partial shell sum is returned."""
    out = []
    for s in range(len(V)):
        c, e, j = chi[s], eta[s], J[s]
        w = dnu[s] if dnu.ndim == 2 else dnu
        reasons = []
        if not (math.isfinite(V[s]) and V[s] > 0.0):
            reasons.append("VOLUME")
        if bad is not None and bad[s].any():
            reasons.append(f"UNDEFINED_BINS:{int(bad[s].sum())}")
        if not (np.isfinite(c).all() and np.isfinite(e).all() and np.isfinite(j).all() and np.isfinite(w).all()):
            reasons.append("NONFINITE")
        elif (j < 0).any() or (e < 0).any():
            reasons.append("NEGATIVE_J_OR_EMISSIVITY")
        elif (w <= 0).any():
            reasons.append("WIDTH")
        else:
            pa, pe = c * j * w, e * w
            if (((pa == 0) & (c != 0) & (j != 0)) | ((pe == 0) & (e != 0))).any():
                reasons.append("PRODUCT_LOST_TO_ZERO")
        if reasons:
            out.append({"value": None, "reason": ";".join(reasons)})
            continue
        A = FOUR_PI * V[s] * float(np.sum(pa))
        E = FOUR_PI * V[s] * float(np.sum(pe))
        S = FOUR_PI * V[s] * float(np.sum(np.abs(pa)) + np.sum(np.abs(pe)))
        out.append({"value": {"A": A, "E": E, "Q": A - E, "S": S}, "reason": None})
    return out


def lec_separated(a, b, scale):
    """prereg 7: |a - b| <= 1e-10 S is unresolved; S = 0 accepts only an exact zero difference."""
    d = a - b
    return (d != 0.0) if scale == 0.0 else abs(d) > TOL * scale


def lec_sign(Q, scale):
    eps = TOL * scale
    return "순흡수" if Q > eps else "순방출" if Q < -eps else "부호 미분리"


def lec_global(values):
    """sum, positive sum, negative sum and absolute sum over shells; None when any shell is missing."""
    if any(v is None for v in values):
        return None
    pos = sum(v for v in values if v > 0)
    neg = sum(v for v in values if v < 0)
    return {"sum": pos + neg, "positive": pos, "negative": neg, "abs": pos - neg}


# ------------------------------------------------------------------ inputs
def lec_read_dump(it, why):
    """one comparison dump: manifest, shell ledger and spectral arrays on its own bin edges.  Common defects
    (generation, grid, table structure, coverage) go to why as '<tag>:...'; value-level parse failures stay
    per shell / per bin (notes, bad masks) and are handled by the integrals and ledger rows."""
    man = read_json(PC / f"physics_DET_iter{it:04d}.manifest.json")
    tag = f"A{it}"
    for k, v in GEN[tag].items():
        if man.get(k) != v:
            why.append(f"{tag}:GENERATION:{k}={man.get(k)}")
    for k, v in GRID.items():
        if k in man and man.get(k) != v:
            why.append(f"{tag}:GRID:{k}={man.get(k)}")
    with open(PC / f"physics_DET_iter{it:04d}.shell.csv") as f:
        shells = list(csv.DictReader(f))
    if [int(r["shell_id"]) for r in shells] != list(range(NS)) or any(k not in shells[0] for k in LEDGER):
        why.append(f"{tag}:SHELL_TABLE")
    notes = []
    ri = np.array([lec_num(r["r_inner_cm"], notes, f"r_inner:{r['shell_id']}") for r in shells])
    ro = np.array([lec_num(r["r_outer_cm"], notes, f"r_outer:{r['shell_id']}") for r in shells])
    ledger = {k: np.array([lec_num(r[k], notes, f"{k}:{r['shell_id']}") for r in shells]) for k in LEDGER}
    arr = {k: np.full((NS, NB), np.nan) for k in ("J_nu", "chi_es_cm1") + sum(COMP.values(), ())}
    lo, hi = np.full((NS, NB), np.nan), np.full((NS, NB), np.nan)
    seen = np.zeros((NS, NB), dtype=int)
    bad = np.zeros((NS, NB), dtype=bool)
    bad_J = np.zeros((NS, NB), dtype=bool)
    with open(PC / f"physics_DET_iter{it:04d}.spectral.csv") as f:
        for r in csv.DictReader(f):
            s, b = int(r["shell_id"]), int(r["bin_id"])
            if not (0 <= s < NS and 0 <= b < NB):
                why.append(f"{tag}:BIN_ID")
                continue
            seen[s, b] += 1
            w, wj = [], []
            lo[s, b] = lec_num(r["nu_lo_Hz"], w, "nu_lo")
            hi[s, b] = lec_num(r["nu_hi_Hz"], w, "nu_hi")
            for k in arr:
                arr[k][s, b] = lec_num(r[k], wj if k == "J_nu" else w, k)
            bad[s, b] = bool(w or wj)
            bad_J[s, b] = bool(wj)
    if (seen != 1).any():
        why.append(f"{tag}:BIN_COVERAGE:{int((seen != 1).sum())}")
    edges = np.append(lo[0], hi[0, -1])
    if not ((lo == lo[0]).all() and (hi == hi[0]).all() and (hi[0, :-1] == lo[0, 1:]).all() and
            (np.diff(edges) > 0).all()):
        why.append(f"{tag}:GRID_EDGES")
    return {"tag": tag, "man": man, "ri": ri, "ro": ro, "V": lec_volume(ri, ro), "ledger": ledger, "edges": edges,
            "dnu": np.diff(edges), "arr": arr, "bad": bad, "bad_J": bad_J, "notes": notes}


def lec_read_capture(why):
    """the capture identity (meta, geometry) and the diag9 cells.csv bin representation."""
    meta = dict(ln.split("=", 1) for ln in (CAP / "meta.txt").read_text().splitlines() if "=" in ln)
    if (meta.get("material_generation"), meta.get("rad_generation"), meta.get("NS"), meta.get("NB")) != \
            ("13", "13", str(NS), str(NB)):
        why.append("C13:CAPTURE_META")
    ri = np.fromfile(CAP / "r_inner.f64", dtype="<f8")
    ro = np.fromfile(CAP / "r_outer.f64", dtype="<f8")
    bridge = read_json(BRIDGE)
    cap_sha = sha(CAP / "manifest.json")
    if cap_sha not in json.dumps(bridge):
        why.append("C13:CELLS_NOT_FROM_THIS_CAPTURE")
    cols = ("J", "chi_line_exp", "chi_abs", "eta_line_used", "eta_bfff", "eta_dep")
    arr = {k: np.full((NS, NB), np.nan) for k in cols}
    nu, dnu = np.full(NB, np.nan), np.full(NB, np.nan)
    seen = np.zeros((NS, NB), dtype=int)
    bad = np.zeros((NS, NB), dtype=bool)
    with open(CELLS) as f:
        for r in csv.DictReader(f):
            s, b = int(r["shell"]), int(r["bin"])
            if not (0 <= s < NS and 0 <= b < NB):
                why.append("C13:BIN_ID")
                continue
            seen[s, b] += 1
            w = []
            n, d = lec_num(r["nu_Hz"], w, "nu"), lec_num(r["dnu_Hz"], w, "dnu")
            if s == 0:
                nu[b], dnu[b] = n, d
            elif n != nu[b] or d != dnu[b]:
                w.append("GRID_PER_SHELL")
            for k in cols:
                arr[k][s, b] = lec_num(r[k], w, k)
            if w:
                bad[s, b] = True
    if (seen != 1).any():
        why.append(f"C13:BIN_COVERAGE:{int((seen != 1).sum())}")
    return {"tag": "C13", "meta": meta, "ri": ri, "ro": ro, "V": lec_volume(ri, ro), "nu": nu, "dnu": dnu,
            "arr": arr, "bad": bad, "nu_min": float.fromhex(meta["nu_min"]), "nu_max": float.fromhex(meta["nu_max"])}


# ------------------------------------------------------------------ states and relations
def lec_dump_q(d, J=None):
    """Q per component on the dump's own coefficients; J replaced for X13 (emission coefficients kept)."""
    JJ = d["arr"]["J_nu"] if J is None else J
    return {x: lec_integrate(d["V"], d["dnu"], d["arr"][c], d["arr"][e], JJ, d["bad"]) for x, (c, e) in COMP.items()}


def lec_capture_q(c):
    a = c["arr"]
    zero = np.zeros((NS, NB))
    return {"bb": lec_integrate(c["V"], c["dnu"], a["chi_line_exp"], a["eta_line_used"], a["J"], c["bad"]),
            "cont": lec_integrate(c["V"], c["dnu"], a["chi_abs"], a["eta_bfff"] + a["eta_dep"], a["J"], c["bad"]),
            "dep": lec_integrate(c["V"], c["dnu"], zero, a["eta_dep"], a["J"], c["bad"])}


def lec_cont(q):
    """Q_cont = Q_bf + Q_ff per shell (None if either is missing)."""
    out = []
    for a, b in zip(q["bf"], q["ff"]):
        if a["value"] is None or b["value"] is None:
            out.append({"value": None, "reason": a["reason"] or b["reason"]})
        else:
            out.append({"value": {k: a["value"][k] + b["value"][k] for k in ("A", "E", "Q", "S")}, "reason": None})
    return out


def lec_ledger_row(d, s):
    """V times every registered ledger term, H, C, R, L and R/(H+C) for one shell (None-free only if all finite)."""
    V = d["V"][s]
    t = {k: V * float(d["ledger"][k][s]) for k in LEDGER}
    t["L"] = t["line_abs_heat"] - t["line_emit_cool"]
    t["BF_ledger"] = t["photo_heat"] - t["recomb_cool"]
    t["FF_ledger"] = t["ff_abs_heat"] - t["ff_emit_cool"]
    hc = t["sum_heating"] + t["sum_cooling"]
    t["R_over_HC"] = t["residual"] / hc if hc else None
    t["line_scale"] = t["line_abs_heat"] + t["line_emit_cool"]
    t["bf_scale"] = t["photo_heat"] + t["recomb_cool"]
    t["ff_scale"] = t["ff_abs_heat"] + t["ff_emit_cool"]
    return t if all(math.isfinite(v) for v in t.values() if v is not None) else None


def lec_d_rows(d, q, rel):
    """D10 / D11: ledger against the dump transport integrals, with the BF / FF mandatory identities."""
    rows, sep, ident = [], [], []
    for s in range(len(d["V"])):
        t = lec_ledger_row(d, s)
        row = {"relation": rel, "shell": s, "v_inner_cm_s": None, "volume_cm3": d["V"][s],
               "left": f"{d['tag']} ledger", "right": f"{d['tag']} transport",
               "history": f"material {d['man'].get('population_generation')} / radiation "
                          f"{d['man'].get('radiation_generation')}"}
        miss = [x for x in ("bb", "bf", "ff") if q[x][s]["value"] is None]
        if t is None or miss:
            row.update(status="미산출", reason=";".join(([] if t else ["LEDGER"]) +
                                                       [f"{x}:{q[x][s]['reason']}" for x in miss]))
            rows.append(row)
            sep.append(None)
            ident.append(None)
            continue
        qb, qf, qff = q["bb"][s]["value"], q["bf"][s]["value"], q["ff"][s]["value"]
        row.update({k: t[k] for k in LEDGER + ("L", "R_over_HC")})
        row.update({f"{x}_{k}": q[x][s]["value"][k] for x in ("bb", "bf", "ff") for k in ("A", "E", "Q")})
        Qrad = qb["Q"] + qf["Q"] + qff["Q"]
        sl, sb, sf = t["line_scale"] + qb["S"], t["bf_scale"] + qf["S"], t["ff_scale"] + qff["S"]
        dl, db, df = t["L"] - qb["Q"], t["BF_ledger"] - qf["Q"], t["FF_ledger"] - qff["Q"]
        s_line, s_bf, s_ff = lec_separated(t["L"], qb["Q"], sl), lec_separated(t["BF_ledger"], qf["Q"], sb), \
            lec_separated(t["FF_ledger"], qff["Q"], sf)
        row.update(Q_rad=Qrad, Q_rad_sign=lec_sign(Qrad, qb["S"] + qf["S"] + qff["S"]), line_diff=dl,
                   line_diff_over_delta=dl / DELTA, line_tolerance=TOL * sl, line_separated=s_line,
                   bf_identity_diff=db, bf_tolerance=TOL * sb, bf_identity_separated=s_bf,
                   ff_identity_diff=df, ff_tolerance=TOL * sf, ff_identity_separated=s_ff,
                   status="산출", cls=("선 장부–전달 차 분리" if s_line else "선 장부–전달 차 미분리"))
        rows.append(row)
        sep.append(s_line)
        ident.append(s_bf or s_ff)
    return rows, sep, ident


def lec_pair_rows(rel, left_name, right_name, left, right, extra=None):
    """J13 / O13: bb and continuum Q of two representations per shell and their differences."""
    rows = []
    for s in range(len(left["bb"])):
        row = {"relation": rel, "shell": s, "left": left_name, "right": right_name}
        vals = {}
        for x in ("bb", "cont"):
            a, b = left[x][s], right[x][s]
            if a["value"] is None or b["value"] is None:
                vals = None
                row.update(status="미산출", reason=f"{x}:{a['reason'] or b['reason']}")
                break
            sc = a["value"]["S"] + b["value"]["S"]
            vals[x] = (a["value"]["Q"], b["value"]["Q"], sc)
        if vals is None:
            rows.append(row)
            continue
        for x, (qa, qb, sc) in vals.items():
            row.update({f"{x}_Q_left": qa, f"{x}_Q_right": qb, f"{x}_diff": qa - qb, f"{x}_diff_over_delta":
                        (qa - qb) / DELTA, f"{x}_tolerance": TOL * sc, f"{x}_separated": lec_separated(qa, qb, sc),
                        f"{x}_right_sign": lec_sign(qb, sc)})
        if extra is not None:
            row.update(extra(s))
        row["status"] = "산출"
        rows.append(row)
    return rows


def lec_summary(rows, rel, keys):
    """one global row per relation: sums of the listed per-shell keys, None when any shell is missing."""
    out = {"relation": rel, "shell": "전역", "status": "산출"}
    for k in keys:
        g = lec_global([r.get(k) if r.get("status") == "산출" else None for r in rows])
        if g is None:
            out.update(status="미산출", reason="incomplete shells: global sum not formed")
            return out
        out.update({f"{k}_{m}": v for m, v in g.items()})
        out[f"{k}_sum_over_delta"] = g["sum"] / DELTA
        out[f"{k}_abs_over_delta"] = g["abs"] / DELTA
    return out


def lec_branch(valid_all, separated):
    """0 PROJECTED_GAP / 1 NO_RESOLVED_PROJECTED_GAP need every mandatory calculation valid; otherwise 2."""
    if not valid_all or any(v is None for v in separated):
        return 2
    return 0 if any(separated) else 1


# ------------------------------------------------------------------ selftest (prereg 9)
def lec_toy():
    """two shells x three bins with signed bb opacity, a scattering opacity and a hand-checked integral."""
    V = np.array([2.0, 3.0])
    dnu = np.array([1.0, 2.0, 4.0])
    chi = np.array([[1.0, -0.5, 0.25], [0.0, 2.0, 1.0]])
    eta = np.array([[0.5, 0.0, 1.0], [1.0, 0.0, 0.0]])
    J = np.array([[2.0, 1.0, 4.0], [3.0, 1.0, 0.5]])
    # shell 0: A = 4pi 2 (2 - 1 + 4) = 40 pi, E = 4pi 2 (0.5 + 4) = 36 pi; shell 1: A = 4pi 3 (4 + 2) = 72 pi,
    # E = 4pi 3 (1) = 12 pi
    return V, dnu, chi, eta, J, [(40 * math.pi, 36 * math.pi), (72 * math.pi, 12 * math.pi)]


def cmd_selftest():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    V, dnu, chi, eta, J, want = lec_toy()
    q = lec_integrate(V, dnu, chi, eta, J, None)
    close = lambda a, b: abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)     # noqa: E731
    expect(all(close(q[s]["value"]["A"], want[s][0]) and close(q[s]["value"]["E"], want[s][1]) for s in range(2)),
           "toy integrals A, E, Q with signed opacity (positive control)")
    no4pi = [r["value"]["Q"] / FOUR_PI for r in q]
    expect(not close(no4pi[0], want[0][0] - want[0][1]), "inject 4 pi omitted: differs")
    noV = lec_integrate(np.ones(2), dnu, chi, eta, J, None)
    expect(not close(noV[1]["value"]["Q"], want[1][0] - want[1][1]), "inject volume omitted: differs")
    nodnu = lec_integrate(V, np.ones(3), chi, eta, J, None)
    expect(not close(nodnu[0]["value"]["Q"], want[0][0] - want[0][1]), "inject bin width omitted: differs")
    logw = lec_integrate(V, np.log(np.array([2.0, 3.0, 5.0])) - np.log(np.array([1.0, 2.0, 3.0])), chi, eta, J, None)
    expect(not close(logw[0]["value"]["Q"], want[0][0] - want[0][1]), "inject a log-frequency width: differs")
    expect(q[0]["value"]["Q"] > 0 and lec_sign(-q[0]["value"]["Q"], q[0]["value"]["S"]) == "순방출" and
           lec_sign(q[0]["value"]["Q"], q[0]["value"]["S"]) == "순흡수", "Q sign convention (positive = absorption)")
    noneg = lec_integrate(V, dnu, np.abs(chi), eta, J, None)
    expect(not close(noneg[0]["value"]["Q"], q[0]["value"]["Q"]), "inject the valid negative opacity removed "
           "(absolute value): differs")
    # scattering: chi_total J - eta_true_total omits the scattering source
    chi_es = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    naive = lec_integrate(V, dnu, chi + chi_es, eta, J, None)
    expect(not close(naive[0]["value"]["Q"], q[0]["value"]["Q"]), "inject chi_total J - eta_true_total (scattering "
           "source missing): differs")
    # a toy dump whose sign-split line ledger equals the transport net Q: the registered net comparison is
    # unresolved, while reading line_abs_heat as the total absorption A would report a separation
    led = {k: np.zeros(2) for k in LEDGER}
    for s in range(2):
        Qs = q[s]["value"]["Q"] / V[s]
        led["line_abs_heat"][s], led["line_emit_cool"][s] = (Qs, 0.0) if Qs > 0 else (0.0, -Qs)
    zero = np.zeros((2, 3))
    dt = {"tag": "T", "man": {}, "V": V, "dnu": dnu, "ledger": led, "bad": None,
          "arr": {"J_nu": J, "chi_bb_cm1": chi, "eta_bb": eta, "chi_bf_cm1": zero, "eta_bf": zero,
                  "chi_ff_cm1": zero, "eta_ff": zero}}
    trows, tsep, tident = lec_d_rows(dt, lec_dump_q(dt), "DT")
    wrong = lec_separated(V[0] * led["line_abs_heat"][0], q[0]["value"]["A"], q[0]["value"]["S"])
    expect(tsep == [False, False] and tident == [False, False] and wrong,
           "net ledger vs net transport unresolved (positive control); inject the sign-split line heat read as the "
           "total absorption A: separated")
    # X13: emission changed or another shell / bin linked
    J2 = J * np.array([[1.5], [0.5]])
    x13 = lec_integrate(V, dnu, chi, eta, J2, None)
    x13_bad = lec_integrate(V, dnu, chi, eta * 2.0, J2, None)
    expect(close(x13[0]["value"]["E"], q[0]["value"]["E"]) and not close(x13_bad[0]["value"]["E"],
                                                                          q[0]["value"]["E"]),
           "X13 keeps the emission integral; inject emission coefficients replaced: detected")
    swapped = lec_integrate(V, dnu, chi, eta, J2[::-1], None)
    expect(not close(swapped[0]["value"]["A"], x13[0]["value"]["A"]), "inject J of another shell: differs")
    rolled = lec_integrate(V, dnu, chi, eta, np.roll(J2, 1, axis=1), None)
    expect(not close(rolled[0]["value"]["A"], x13[0]["value"]["A"]), "inject J of another bin: differs")
    # exact zero vs representation loss
    tiny = lec_integrate(V, dnu, np.array([[5e-324, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.zeros((2, 3)),
                         np.array([[1e-300, 1.0, 1.0], [1.0, 1.0, 1.0]]), None)
    exact = lec_integrate(V, dnu, np.zeros((2, 3)), np.zeros((2, 3)), J, None)
    expect(tiny[0]["value"] is None and "PRODUCT_LOST_TO_ZERO" in tiny[0]["reason"] and exact[0]["value"]["Q"] == 0.0,
           "inject a product of non-zero factors lost to 0: undefined; exact zero stays a valid zero")
    w = []
    v0 = lec_num("1e-400", w, "x")
    expect(math.isnan(v0) and any("TEXT_LOST_TO_ZERO" in s for s in w) and lec_num("0.0", [], "x") == 0.0,
           "inject a non-zero stored text read as 0: undefined; a stored zero stays zero")
    # one bin undefined -> that shell undefined; global sums refuse partial shells
    bad = np.zeros((2, 3), dtype=bool)
    bad[1, 2] = True
    qb = lec_integrate(V, dnu, chi, eta, J, bad)
    expect(qb[0]["value"] is not None and qb[1]["value"] is None, "inject one undefined bin: that shell undefined, "
           "the other shell kept")
    expect(lec_global([1.0, None]) is None and lec_global([1.0, -2.0]) == {"sum": -1.0, "positive": 1.0,
                                                                          "negative": -2.0, "abs": 3.0},
           "inject a partial shell set into a global sum: refused (complete set positive control)")
    neg_j = lec_integrate(V, dnu, chi, eta, -J, None)
    expect(all(r["value"] is None for r in neg_j), "inject a negative J: undefined")
    # common defect through the real calculation path: a dump with a broken grid stops D, J13 and O13
    d = {"tag": "A10", "man": {"population_generation": 13, "radiation_generation": 12}, "V": np.ones(NS),
         "dnu": np.ones(NB), "ledger": {k: np.zeros(NS) for k in LEDGER},
         "arr": {k: np.zeros((NS, NB)) for k in ("J_nu",) + sum(COMP.values(), ())},
         "bad": np.ones((NS, NB), dtype=bool)}
    qd = lec_dump_q(d)
    rows, sep, ident = lec_d_rows(d, qd, "D10")
    j13 = lec_pair_rows("J13", "A10", "X13", {"bb": qd["bb"], "cont": lec_cont(qd)},
                        {"bb": qd["bb"], "cont": lec_cont(qd)})
    expect(all(r["status"] == "미산출" for r in rows + j13) and lec_branch(True, sep) == 2 and
           lec_summary(rows, "D10", ["line_diff"])["status"] == "미산출",
           "inject a common dump defect: D10 and J13 rows, the global row and the branch are not produced")
    # an unrecorded ledger is not copied into X13 / C13 rows
    expect(all("line_abs_heat" not in r for r in j13), "X13 / C13 rows carry no copied ledger columns")
    expect([lec_branch(True, [False, True]), lec_branch(True, [False, False]), lec_branch(False, [True]),
            lec_branch(True, [None, True])] == [0, 1, 2, 2], "branch fixtures (positive control)")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_LINE_ENERGY_CORR_SELFTEST_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
def cmd_analyze(out):
    out = Path(out)
    if out.exists():
        print(f"EXISTS {out}")
        return 3
    gates = {k: [] for k in ("G-INPUT", "G-CALC", "G-RUN")}
    part, cvd = os.environ.get("SLURM_JOB_PARTITION"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if part != "a10" or cvd not in ("", "NoDevFiles"):
        gates["G-RUN"].append(f"NOT_A10_CPU:{part}:{cvd}")
    used = [PC / f"physics_DET_iter{it:04d}.{k}" for it in (10, 11) for k in ("manifest.json", "shell.csv",
                                                                               "spectral.csv")]
    used += [CAP / "manifest.json", CAP / "meta.txt", CAP / "r_inner.f64", CAP / "r_outer.f64", CELLS, BRIDGE]
    before = {str(p): sha(p) for p in used}
    why10, why11, whyc = [], [], []
    A10, A11, C = lec_read_dump(10, why10), lec_read_dump(11, why11), lec_read_capture(whyc)
    for w in (why10, why11, whyc):
        gates["G-INPUT"] += sorted(set(x.split(":")[0] + ":" + x.split(":")[1] for x in w))[:8]
    same_geom = np.array_equal(A10["ri"], A11["ri"]) and np.array_equal(A10["ro"], A11["ro"])
    same_grid = np.array_equal(A10["edges"], A11["edges"])
    cap_geom = np.array_equal(C["ri"], A10["ri"]) and np.array_equal(C["ro"], A10["ro"])
    cap_support = (C["nu_min"] == A10["edges"][0], C["nu_max"] == A10["edges"][-1])
    if not same_geom:
        gates["G-INPUT"].append("A10_A11_GEOMETRY")
    if not same_grid:
        gates["G-INPUT"].append("A10_A11_GRID")
    if not cap_geom:
        gates["G-INPUT"].append("C13_GEOMETRY")
    common10 = any(w.startswith("A10:") for w in gates["G-INPUT"])
    common11 = any(w.startswith("A11:") for w in gates["G-INPUT"])
    commonc = any(w.startswith("C13:") for w in gates["G-INPUT"]) or not cap_geom

    q10 = lec_dump_q(A10) if not common10 else None
    q11 = lec_dump_q(A11) if not common11 else None
    stop_x = common10 or common11 or not (same_geom and same_grid)
    qx = None if stop_x else lec_dump_q(A10, J=A11["arr"]["J_nu"])
    if qx is not None and A11["bad_J"].any():
        qx = {x: [r if not A11["bad_J"][s].any() else {"value": None, "reason": "A11_J_UNDEFINED"}
                  for s, r in enumerate(v)] for x, v in qx.items()}
    qc = None if commonc else lec_capture_q(C)
    stop_rows = [{"relation": rel, "shell": s, "status": "미산출", "reason": "COMMON_INPUT_DEFECT"}
                 for rel in ("D10", "D11", "J13", "O13") for s in range(NS)]
    rows, sep, ident = [], [], []
    for rel, d, q in (("D10", A10, q10), ("D11", A11, q11)):
        if q is None:
            rr = [r for r in stop_rows if r["relation"] == rel]
            rows += rr
            sep += [None] * NS
            ident += [None] * NS
        else:
            rr, ss, ii = lec_d_rows(d, q, rel)
            rows += rr
            sep += ss
            ident += ii
    for r in rows:
        r["v_inner_cm_s"] = None
    if q10 is None or qx is None:
        rows += [r for r in stop_rows if r["relation"] == "J13"]
    else:
        rows += lec_pair_rows("J13", "A10 (J radiation 12)", "X13 (A10 coefficients, J radiation 13)",
                              {"bb": q10["bb"], "cont": lec_cont(q10)}, {"bb": qx["bb"], "cont": lec_cont(qx)})
    support_ok = all(cap_support)
    if qx is None or qc is None or not support_ok:
        rr = [dict(r, reason=r["reason"] if (qx is None or qc is None) else "SUPPORT_ENDS_DIFFER")
              for r in stop_rows if r["relation"] == "O13"]
        rows += rr
    else:
        rows += lec_pair_rows("O13", "X13", "C13 (capture representation)", {"bb": qx["bb"], "cont": lec_cont(qx)},
                              {"bb": qc["bb"], "cont": qc["cont"]},
                              extra=lambda s: {"C13_E_dep": qc["dep"][s]["value"]["E"] if qc["dep"][s]["value"]
                                               else None})
    for r in rows:
        if r.get("status") == "산출":
            r["v_inner_cm_s"] = float(A10["ri"][r["shell"]] / 1683072.0)
    if any(v for v in ident if v):
        gates["G-CALC"].append(f"BF_FF_IDENTITY:{sum(1 for v in ident if v)}")
    summaries = [lec_summary([r for r in rows if r["relation"] == "D10"], "D10", ["L", "bb_Q", "line_diff"]),
                 lec_summary([r for r in rows if r["relation"] == "D11"], "D11", ["L", "bb_Q", "line_diff"]),
                 lec_summary([r for r in rows if r["relation"] == "J13"], "J13", ["bb_diff", "cont_diff"]),
                 lec_summary([r for r in rows if r["relation"] == "O13"], "O13",
                             ["bb_Q_left", "bb_Q_right", "bb_diff", "cont_Q_right", "cont_diff"])]
    produced = all(r.get("status") == "산출" for r in rows) and all(s["status"] == "산출" for s in summaries)
    after = {str(p): sha(p) for p in used}
    if after != before:
        gates["G-INPUT"].append("INPUT_CHANGED_DURING_RUN")
    all_gates = [f for v in gates.values() for f in v]
    valid_all = produced and not all_gates
    code = lec_branch(valid_all, sep)
    label = {0: "PROJECTED_GAP", 1: "NO_RESOLVED_PROJECTED_GAP", 2: "INDETERMINATE"}[code]
    n_sep = {rel: sum(1 for r in rows if r["relation"] == rel and r.get("line_separated")) for rel in ("D10", "D11")}
    out.mkdir(parents=True)
    allrows = rows + summaries
    fields = list(dict.fromkeys(k for r in allrows for k in r))
    write_csv(out / "shell_energy_correspondence.csv", allrows, fields)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    cols = {"D": ["relation", "shell", "history", "L", "bb_A", "bb_E", "bb_Q", "line_diff", "line_diff_over_delta",
                  "line_separated", "bf_identity_separated", "ff_identity_separated", "Q_rad", "Q_rad_sign",
                  "R_over_HC", "status", "reason"],
            "J": ["relation", "shell", "bb_Q_left", "bb_Q_right", "bb_diff", "bb_separated", "cont_Q_left",
                  "cont_Q_right", "cont_diff", "cont_separated", "status", "reason"],
            "O": ["relation", "shell", "bb_Q_left", "bb_Q_right", "bb_diff", "bb_separated", "bb_right_sign",
                  "cont_Q_right", "cont_diff", "cont_separated", "C13_E_dep", "status", "reason"]}
    with open(out / "shell_energy_correspondence.md", "x") as f:
        f.write(f"# Thermal ledger vs transport energy exchange, O 19.48 d (branch {code} {label})\n\n")
        f.write(f"Context: Delta (diag9 2500-6000 A deficit) = {DELTA:.6e} erg/s. Positive Q = radiation to matter. "
                "D rows compare the sign-split line ledger L = V(line_abs_heat - line_emit_cool) with "
                "Q_bb = 4 pi V sum (chi_bb J - eta_bb) dnu of the same dump; J13 = A10 vs X13 (coefficients fixed, "
                "J radiation 12 -> 13, mixed history); O13 = X13 vs the capture bin representation C13 (its positive "
                "Q is a net absorption of the stored bin representation, not unreemitted energy of observed rays). "
                "Arithmetic decomposition, not causal shares.\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        for rel, key in (("D10", "D"), ("D11", "D"), ("J13", "J"), ("O13", "O")):
            f.write(f"## {rel}\n\n| " + " | ".join(cols[key]) + " |\n|" + "---|" * len(cols[key]) + "\n")
            for r in [r for r in rows if r["relation"] == rel]:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols[key]) + " |\n")
            f.write("\n")
        f.write("## Global rows\n\n")
        for s in summaries:
            f.write("- " + ", ".join(f"{k}={fmt(v)}" for k, v in s.items()) + "\n")
    evidence = {"schema": "DET_LINE_ENERGY_CORR_V1", "prereg": PREREG, "argv": sys.argv,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "branch": code, "label": label, "gates": gates,
                "D_separated_shells": n_sep, "BF_FF_identity_failures": sum(1 for v in ident if v),
                "histories": {"A10": {k: A10["man"].get(k) for k in list(GEN["A10"]) + ["opacity_generation",
                                                                                     "emissivity_generation"]},
                              "A11": {k: A11["man"].get(k) for k in list(GEN["A11"]) + ["opacity_generation",
                                                                                     "emissivity_generation"]},
                              "X13": "A10 coefficients with the A11 J (mixed, not a complete 13/13 ledger)",
                              "C13": {k: C["meta"].get(k) for k in ("material_generation", "rad_generation",
                                                                    "tau_required_generation", "iter")}},
                "checks": {"A10_A11_geometry_equal": same_geom, "A10_A11_grid_equal": same_grid,
                           "capture_geometry_equal": cap_geom, "capture_support_ends_equal": [bool(x) for x in cap_support],
                           "capture_dnu_over_dump_dnu": [float(np.min(C["dnu"] / A10["dnu"])),
                                                         float(np.max(C["dnu"] / A10["dnu"]))],
                           "capture_J_equals_A11_J": bool(np.array_equal(C["arr"]["J"], A11["arr"]["J_nu"]))},
                "input_findings": {"A10": why10[:20], "A11": why11[:20], "C13": whyc[:20]},
                "value_notes": {"A10": A10["notes"][:20], "A11": A11["notes"][:20]},
                "undefined_cells": {"A10": int(A10["bad"].sum()), "A11": int(A11["bad"].sum()),
                                    "A11_J": int(A11["bad_J"].sum()), "C13": int(C["bad"].sum())},
                "inputs": {p: {"bytes": Path(p).stat().st_size, "sha256": h} for p, h in before.items()},
                "inputs_unchanged": after == before,
                "code": {str(Path(__file__).resolve()): sha(Path(__file__)),
                         str((D9 / "tools/r1_common.py").resolve()): sha(D9 / "tools/r1_common.py")}}
    write_json(out / "evidence.json", evidence)
    print(f"BRANCH {code} {label} D_separated={json.dumps(n_sep)} "
          f"identity_failures={evidence['BF_FF_identity_failures']}")
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
    a = ap.parse_args()
    return cmd_selftest() if a.cmd == "selftest" else cmd_analyze(a.out)


if __name__ == "__main__":
    sys.exit(main())
