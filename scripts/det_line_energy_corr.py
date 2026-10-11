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
  selftest                                  negative controls of the new code (prereg 9), incl. reader-path
                                            injections on toy files written to a job-local temporary directory
  analyze --out DIR [--selftest-log PATH]   G-INPUT, G-CALC, G-RUN and the main table
Exit codes: 0 pass, 2 refused or gate finding, 3 existing output path.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
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
EPOCH_S = 1683072.0
FOUR_PI = 4.0 * math.pi
GEN = {"A10": {"population_generation": 13, "radiation_generation": 12, "te_generation": 12,
               "opacity_generation": 23, "emissivity_generation": 23},
       "A11": {"population_generation": 14, "radiation_generation": 13, "te_generation": 13,
               "opacity_generation": 25, "emissivity_generation": 25}}
DEF = {"frame": "SHELL_COMOVING", "frequency_coordinate": "HZ",
       "frequency_regrid": "INTEGRAL_PRESERVING_PIECEWISE_CONSTANT", "opacity_units": "CM^-1",
       "emissivity_units": "ERG_S^-1_CM^-3_HZ^-1_SR^-1", "eta_is_per_sr": True}
LEDGER = ("q_ad_heating", "photo_heat", "line_abs_heat", "ff_abs_heat", "compton_heat", "gamma_heat",
          "nonthermal_heat", "q_ad_cooling", "recomb_cool", "line_emit_cool", "coll_line_cool", "ff_emit_cool",
          "compton_cool", "sum_heating", "sum_cooling", "residual")
SHELL_COLS = ("shell_id", "r_inner_cm", "r_outer_cm", "v_inner_cm_s", "v_outer_cm_s") + LEDGER
COMP = {"bb": ("chi_bb_cm1", "eta_bb"), "bf": ("chi_bf_cm1", "eta_bf"), "ff": ("chi_ff_cm1", "eta_ff")}
SPEC_COLS = ("shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz", "J_nu", "chi_es_cm1") + sum(COMP.values(), ())
CELL_COLS = ("shell", "bin", "nu_Hz", "dnu_Hz", "J", "chi_line_exp", "chi_abs", "eta_line_used", "eta_bfff", "eta_dep")
ZERO_TEXT = re.compile(r"^[+-]?0*\.?0*([eE][+-]?\d+)?$")
STRUCTURAL = {"ledger_raw_line_A": "원 선 총흡수 미저장 — 장부 선 열은 셸 순율의 부호 분리(§4·§6)",
              "ledger_raw_line_E": "원 선 총방출 미저장 — 장부 선 열은 셸 순율의 부호 분리(§4·§6)",
              "observed_ray_exchange": "관측 광선의 셸별 입사·출사 강도 미저장 — 실제 관측 교환은 구조적 미식별(§2)",
              "X13_C13_ledger": "X13·C13 의 열 장부는 기록되지 않음 — A10·A11 을 복사하지 않음(§6)"}


# ------------------------------------------------------------------ parsing and definition checks
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


def lec_check_manifest(man, tag, ns, nb):
    """common definition of one dump (prereg 3): every registered key present and equal; a missing key fails."""
    why = []
    for k, v in list(GEN[tag].items()) + list(DEF.items()) + [("n_bins", nb), ("n_shells", ns)]:
        if k not in man:
            why.append(f"{tag}:MISSING:{k}")
        elif man[k] != v:
            why.append(f"{tag}:MISMATCH:{k}={man[k]}")
    rif = man.get("radiative_integral_factor")
    if not isinstance(rif, (int, float)) or abs(rif - FOUR_PI) > 1e-12 * FOUR_PI:
        why.append(f"{tag}:RADIATIVE_INTEGRAL_FACTOR")
    return why


def lec_check_columns(header, required, tag, name):
    return [f"{tag}:MISSING_COLUMN:{name}:{c}" for c in required if c not in (header or [])]


# ------------------------------------------------------------------ integrals
def lec_integrate(V, dnu, chi, eta, J, bad):
    """per shell: A = 4 pi V sum chi J dnu, E = 4 pi V sum eta dnu, Q = A - E and the pre-cancellation scale
    S = 4 pi V sum (|chi J| + |eta|) dnu, plus the unweighted sums sum chi J dnu, sum |chi J| dnu, sum eta dnu.
    Signed chi is kept.  A shell with an undefined bin (bad), a negative J or emissivity, a non-positive width or
    volume, a non-finite input, a product of non-zero factors lost to zero, or a non-finite or lost derived value
    is None with its reason (prereg 7); no partial shell sum is returned."""
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
        if reasons:
            out.append({"value": None, "reason": ";".join(reasons)})
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            pa, pe = c * j * w, e * w
            if (((pa == 0) & (c != 0) & (j != 0)) | ((pe == 0) & (e != 0))).any():
                out.append({"value": None, "reason": "PRODUCT_LOST_TO_ZERO"})
                continue
            sa, sabs, se = float(np.sum(pa)), float(np.sum(np.abs(pa))), float(np.sum(pe))
        vals = {"A": FOUR_PI * V[s] * sa, "E": FOUR_PI * V[s] * se, "S": FOUR_PI * V[s] * (sabs + se),
                "sum_chiJdnu": sa, "sum_abs_chiJdnu": sabs, "sum_etadnu": se}
        vals["Q"] = vals["A"] - vals["E"]
        if not all(math.isfinite(v) for v in vals.values()) or (sabs > 0 and vals["S"] == 0.0):
            out.append({"value": None, "reason": "DERIVED_NONFINITE_OR_LOST"})
            continue
        out.append({"value": vals, "reason": None})
    return out


def lec_separated(a, b, scale):
    """prereg 7: |a - b| <= 1e-10 S is unresolved; S = 0 accepts only an exact zero difference.  None if any input
    is not finite (no comparison is made on non-finite values)."""
    if not all(math.isfinite(v) for v in (a, b, scale)):
        return None
    d = a - b
    return bool(d != 0.0) if scale == 0.0 else bool(abs(d) > TOL * scale)


def lec_sign(Q, scale):
    """sign of one state's Q against that state's own pre-cancellation scale."""
    if not (math.isfinite(Q) and math.isfinite(scale)):
        return None
    eps = TOL * scale
    return "순흡수" if Q > eps else "순방출" if Q < -eps else "부호 미분리"


def lec_global(values):
    """sum, positive sum, negative sum and absolute sum over shells; None when any shell is missing or not finite."""
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    pos = sum(v for v in values if v > 0)
    neg = sum(v for v in values if v < 0)
    return {"sum": pos + neg, "positive": pos, "negative": neg, "abs": pos - neg}


# ------------------------------------------------------------------ readers (paths are arguments for the selftest)
def lec_read_dump(pc, it, why, ns=NS, nb=NB):
    """one comparison dump.  Common defects (definition, columns, coverage, grid) go to why as '<tag>:...';
    value-level failures stay per shell (notes) or per bin and component (bad masks)."""
    tag = f"A{it}"
    man = read_json(pc / f"physics_DET_iter{it:04d}.manifest.json")
    why += lec_check_manifest(man, tag, ns, nb)
    with open(pc / f"physics_DET_iter{it:04d}.shell.csv") as f:
        rd = csv.DictReader(f)
        miss = lec_check_columns(rd.fieldnames, SHELL_COLS, tag, "shell")
        shells = list(rd) if not miss else []
    why += miss
    notes = []
    if not miss and [r["shell_id"] for r in shells] != [str(s) for s in range(ns)]:
        why.append(f"{tag}:SHELL_TABLE")
    col = (lambda k: np.array([lec_num(r[k], notes, f"{k}:{r['shell_id']}") for r in shells])) if not miss and \
        len(shells) == ns else (lambda k: np.full(ns, np.nan))
    ri, ro, vi, vo = col("r_inner_cm"), col("r_outer_cm"), col("v_inner_cm_s"), col("v_outer_cm_s")
    ledger = {k: col(k) for k in LEDGER}
    arr = {k: np.full((ns, nb), np.nan) for k in SPEC_COLS[4:]}
    lo, hi = np.full((ns, nb), np.nan), np.full((ns, nb), np.nan)
    seen = np.zeros((ns, nb), dtype=int)
    bad = {k: np.zeros((ns, nb), dtype=bool) for k in ("J", "es", "bb", "bf", "ff")}
    with open(pc / f"physics_DET_iter{it:04d}.spectral.csv") as f:
        rd = csv.DictReader(f)
        miss = lec_check_columns(rd.fieldnames, SPEC_COLS, tag, "spectral")
        why += miss
        for r in (rd if not miss else []):
            try:
                s, b = int(r["shell_id"]), int(r["bin_id"])
            except ValueError:
                why.append(f"{tag}:BIN_ID")
                continue
            if not (0 <= s < ns and 0 <= b < nb):
                why.append(f"{tag}:BIN_ID")
                continue
            seen[s, b] += 1
            we = []
            lo[s, b] = lec_num(r["nu_lo_Hz"], we, "nu_lo")
            hi[s, b] = lec_num(r["nu_hi_Hz"], we, "nu_hi")
            if we:
                why.append(f"{tag}:GRID_VALUE")
            wj = []
            arr["J_nu"][s, b] = lec_num(r["J_nu"], wj, "J_nu")
            bad["J"][s, b] = bool(wj)
            wes = []
            arr["chi_es_cm1"][s, b] = lec_num(r["chi_es_cm1"], wes, "chi_es")
            bad["es"][s, b] = bool(wes)
            for x, (c, e) in COMP.items():
                wx = []
                arr[c][s, b] = lec_num(r[c], wx, c)
                arr[e][s, b] = lec_num(r[e], wx, e)
                bad[x][s, b] = bool(wx)
    if not miss and (seen != 1).any():
        why.append(f"{tag}:BIN_COVERAGE:{int((seen != 1).sum())}")
    edges = np.append(lo[0], hi[0, -1])
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (lo == lo[0]).all() and (hi == hi[0]).all() and
            (hi[0, :-1] == lo[0, 1:]).all() and (np.diff(edges) > 0).all()):
        why.append(f"{tag}:GRID_EDGES")
    hist = f"material {man.get('population_generation')} / radiation {man.get('radiation_generation')} / " \
           f"T {man.get('te_generation')} / opacity {man.get('opacity_generation')} / emission " \
           f"{man.get('emissivity_generation')}"
    return {"tag": tag, "man": man, "hist": hist, "ri": ri, "ro": ro, "vi": vi, "vo": vo, "V": lec_volume(ri, ro),
            "ledger": ledger, "edges": edges, "dnu": np.diff(edges), "arr": arr, "bad": bad, "notes": notes}


def lec_read_capture(cap, cells, bridge, why, ns=NS, nb=NB):
    """the capture identity (meta, geometry) and the diag9 cells.csv bin representation; a per-shell grid
    mismatch is a common grid defect."""
    meta = dict(ln.split("=", 1) for ln in (cap / "meta.txt").read_text().splitlines() if "=" in ln)
    if (meta.get("material_generation"), meta.get("rad_generation"), meta.get("NS"), meta.get("NB")) != \
            ("13", "13", str(ns), str(nb)):
        why.append("C13:CAPTURE_META")
    cman = read_json(cap / "manifest.json")
    gens = cman.get("generations") or {}
    if cman.get("transaction") != "COMMITTED":
        why.append("C13:TRANSACTION")
    if (cman.get("layout") or {}).get("frame") != DEF["frame"]:
        why.append("C13:FRAME")
    if (gens.get("material"), gens.get("radiation")) != (13, 13):
        why.append("C13:MANIFEST_GENERATIONS")
    files = cman.get("files") or {}
    for name in ("meta.txt", "r_inner.f64", "r_outer.f64"):
        if (files.get(name) or {}).get("sha256") != sha(cap / name):
            why.append(f"C13:FILE_NOT_IN_MANIFEST:{name}")
    ri = np.fromfile(cap / "r_inner.f64", dtype="<f8")
    ro = np.fromfile(cap / "r_outer.f64", dtype="<f8")
    if sha(cap / "manifest.json") not in json.dumps(read_json(bridge)):
        why.append("C13:CELLS_NOT_FROM_THIS_CAPTURE")
    try:
        nu_min, nu_max = float.fromhex(meta["nu_min"]), float.fromhex(meta["nu_max"])
    except (KeyError, ValueError):
        why.append("C13:SUPPORT_META")
        nu_min = nu_max = float("nan")
    arr = {k: np.full((ns, nb), np.nan) for k in CELL_COLS[4:]}
    gnu, gdnu = np.full((ns, nb), np.nan), np.full((ns, nb), np.nan)
    seen = np.zeros((ns, nb), dtype=int)
    bad = {k: np.zeros((ns, nb), dtype=bool) for k in ("J", "bb", "cont")}
    with open(cells) as f:
        rd = csv.DictReader(f)
        miss = lec_check_columns(rd.fieldnames, CELL_COLS, "C13", "cells")
        why += miss
        rows = list(rd) if not miss else []
    for r in rows:
        try:
            s, b = int(r["shell"]), int(r["bin"])
        except ValueError:
            why.append("C13:BIN_ID")
            continue
        if not (0 <= s < ns and 0 <= b < nb):
            why.append("C13:BIN_ID")
            continue
        seen[s, b] += 1
        wg = []
        n, d = lec_num(r["nu_Hz"], wg, "nu"), lec_num(r["dnu_Hz"], wg, "dnu")
        if wg:
            why.append("C13:GRID_VALUE")
        gnu[s, b], gdnu[s, b] = n, d
        wj, wb, wc = [], [], []
        arr["J"][s, b] = lec_num(r["J"], wj, "J")
        arr["chi_line_exp"][s, b] = lec_num(r["chi_line_exp"], wb, "chi_line_exp")
        arr["eta_line_used"][s, b] = lec_num(r["eta_line_used"], wb, "eta_line_used")
        for k in ("chi_abs", "eta_bfff", "eta_dep"):
            arr[k][s, b] = lec_num(r[k], wc, k)
        bad["J"][s, b], bad["bb"][s, b], bad["cont"][s, b] = bool(wj), bool(wb), bool(wc)
    if not miss and (seen != 1).any():
        why.append(f"C13:BIN_COVERAGE:{int((seen != 1).sum())}")
    nu, dnu = gnu[0], gdnu[0]
    if not ((gnu == nu).all() and (gdnu == dnu).all()):
        why.append("C13:GRID_PER_SHELL")
    return {"tag": "C13", "meta": meta, "ri": ri, "ro": ro, "V": lec_volume(ri, ro), "nu": nu, "dnu": dnu,
            "arr": arr, "bad": bad, "nu_min": nu_min, "nu_max": nu_max,
            "hist": f"material {meta.get('material_generation')} / radiation {meta.get('rad_generation')} / tau "
                    f"{meta.get('tau_computed_generation')} (capture)"}


# ------------------------------------------------------------------ states
def lec_dump_q(d, J=None, bad_J=None):
    """Q per component on the dump's own coefficients and per-component masks; J (and its mask) replaced for X13,
    emission coefficients kept."""
    JJ = d["arr"]["J_nu"] if J is None else J
    bJ = d["bad"]["J"] if bad_J is None else bad_J
    return {x: lec_integrate(d["V"], d["dnu"], d["arr"][c], d["arr"][e], JJ, d["bad"][x] | bJ)
            for x, (c, e) in COMP.items()}


def lec_capture_q(c):
    a, bJ = c["arr"], c["bad"]["J"]
    zero = np.zeros_like(a["eta_dep"])
    return {"bb": lec_integrate(c["V"], c["dnu"], a["chi_line_exp"], a["eta_line_used"], a["J"], c["bad"]["bb"] | bJ),
            "cont": lec_integrate(c["V"], c["dnu"], a["chi_abs"], a["eta_bfff"] + a["eta_dep"], a["J"],
                                  c["bad"]["cont"] | bJ),
            "dep": lec_integrate(c["V"], c["dnu"], zero, a["eta_dep"], a["J"], c["bad"]["cont"] | bJ)}


def lec_cont(q):
    """Q_cont = Q_bf + Q_ff per shell (None if either is missing); unweighted sums are added as well."""
    out = []
    for a, b in zip(q["bf"], q["ff"]):
        if a["value"] is None or b["value"] is None:
            out.append({"value": None, "reason": a["reason"] or b["reason"]})
        else:
            out.append({"value": {k: a["value"][k] + b["value"][k] for k in a["value"]}, "reason": None})
    return out


# ------------------------------------------------------------------ relations
def lec_ledger_row(d, s):
    """V times every registered ledger term, H, C, R, L and R/(H+C) for one shell (None if any term is missing)."""
    V = d["V"][s]
    t = {k: V * float(d["ledger"][k][s]) for k in LEDGER}
    if not math.isfinite(V) or not all(math.isfinite(v) for v in t.values()):
        return None
    t["L"] = t["line_abs_heat"] - t["line_emit_cool"]
    t["BF_ledger"] = t["photo_heat"] - t["recomb_cool"]
    t["FF_ledger"] = t["ff_abs_heat"] - t["ff_emit_cool"]
    hc = t["sum_heating"] + t["sum_cooling"]
    t["R_over_HC"] = t["residual"] / hc if hc else None
    t["line_scale"] = abs(t["line_abs_heat"]) + abs(t["line_emit_cool"])
    t["bf_scale"] = abs(t["photo_heat"]) + abs(t["recomb_cool"])
    t["ff_scale"] = abs(t["ff_abs_heat"]) + abs(t["ff_emit_cool"])
    return t


def lec_geom(row, d, s):
    row.update(volume_cm3=float(d["V"][s]), r_inner_cm=float(d["ri"][s]), r_outer_cm=float(d["ro"][s]),
               v_inner_cm_s=float(d["vi"][s]), v_outer_cm_s=float(d["vo"][s]))


def lec_d_rows(d, q, rel, support):
    """D10 / D11: ledger against the same dump's transport integrals.  Each component and each comparison is kept
    or marked independently (prereg 7); the row is complete only when the ledger and all three components exist."""
    rows, sep, ident = [], [], []
    for s in range(len(d["V"])):
        t = lec_ledger_row(d, s)
        row = {"relation": rel, "shell": s, "left": f"{d['tag']} ledger", "right": f"{d['tag']} transport",
               "left_history": d["hist"], "right_history": d["hist"], "same_history": "예", "support": support,
               "identity_conditions": "§5 선 대응 조건: 미확인(선 평균장·세대·deck_scale·β·선 집합·S 분할 미저장)",
               "ledger_raw_line_A": None, "ledger_raw_line_E": None, "observed_ray_exchange": None}
        lec_geom(row, d, s)
        reasons = [] if t else ["LEDGER"]
        if t:
            row.update({k: t[k] for k in LEDGER + ("L", "R_over_HC")})
        for x in ("bb", "bf", "ff"):
            v = q[x][s]["value"]
            if v is None:
                reasons.append(f"{x}:{q[x][s]['reason']}")
                continue
            row.update({f"{x}_{k}": v[k] for k in ("A", "E", "Q", "S")})
            row[f"{x}_sign"] = lec_sign(v["Q"], v["S"])
        ls = bs = fs = None
        if t and q["bb"][s]["value"]:
            v = q["bb"][s]["value"]
            sc = t["line_scale"] + v["S"]
            ls = lec_separated(t["L"], v["Q"], sc)
            row.update(line_diff=t["L"] - v["Q"], line_diff_over_delta=(t["L"] - v["Q"]) / DELTA,
                       line_tolerance=TOL * sc, line_separated=ls)
        if t and q["bf"][s]["value"]:
            v = q["bf"][s]["value"]
            sc = t["bf_scale"] + v["S"]
            bs = lec_separated(t["BF_ledger"], v["Q"], sc)
            row.update(bf_identity_diff=t["BF_ledger"] - v["Q"], bf_tolerance=TOL * sc, bf_identity_separated=bs)
        if t and q["ff"][s]["value"]:
            v = q["ff"][s]["value"]
            sc = t["ff_scale"] + v["S"]
            fs = lec_separated(t["FF_ledger"], v["Q"], sc)
            row.update(ff_identity_diff=t["FF_ledger"] - v["Q"], ff_tolerance=TOL * sc, ff_identity_separated=fs)
        if all(q[x][s]["value"] for x in ("bb", "bf", "ff")):
            Qr = sum(q[x][s]["value"]["Q"] for x in ("bb", "bf", "ff"))
            row.update(Q_rad=Qr, Q_rad_sign=lec_sign(Qr, sum(q[x][s]["value"]["S"] for x in ("bb", "bf", "ff"))))
        if None in (ls, bs, fs):
            reasons.append("COMPARISON_NOT_FORMED")
        row["status"] = "산출" if not reasons else "부분"
        row["reason"] = ";".join(reasons)
        if ls is not None:
            row["cls"] = "선 장부–전달 차 분리" if ls else "선 장부–전달 차 미분리"
        rows.append(row)
        sep.append(ls)
        ident.append(None if None in (bs, fs) else (bs or fs))
    return rows, sep, ident


def lec_pair_rows(rel, left, right, lq, rq, support, conditions, extra=None):
    """J13 / O13: bb and continuum of two representations per shell, each side and comparison kept independently."""
    rows = []
    for s in range(len(lq["bb"])):
        row = {"relation": rel, "shell": s, "left": left["name"], "right": right["name"],
               "left_history": left["hist"], "right_history": right["hist"], "same_history": "아니오",
               "support": support, "identity_conditions": conditions, "observed_ray_exchange": None}
        lec_geom(row, left["geom"], s)
        reasons = []
        for x in ("bb", "cont"):
            a, b = lq[x][s]["value"], rq[x][s]["value"]
            for side, v, src in (("left", a, lq), ("right", b, rq)):
                if v is None:
                    reasons.append(f"{x}_{side}:{src[x][s]['reason']}")
                else:
                    row.update({f"{x}_{side}_{k}": v[k] for k in ("A", "E", "Q", "S")})
                    row[f"{x}_{side}_sign"] = lec_sign(v["Q"], v["S"])
            if a is not None and b is not None:
                sc = a["S"] + b["S"]
                sp = lec_separated(a["Q"], b["Q"], sc)
                row.update({f"{x}_diff": a["Q"] - b["Q"], f"{x}_diff_over_delta": (a["Q"] - b["Q"]) / DELTA,
                            f"{x}_tolerance": TOL * sc, f"{x}_separated": sp})
                if sp is None:
                    reasons.append(f"{x}:COMPARISON_NOT_FORMED")
        if extra is not None:
            row.update(extra(s))
        row["status"] = "산출" if not reasons else "부분"
        row["reason"] = ";".join(reasons)
        rows.append(row)
    return rows


def lec_summary(rows, rel, keys):
    """one global row per relation: each listed per-shell key is summed independently; a key with any shell
    missing or not finite is not formed, the other keys are kept."""
    out = {"relation": rel, "shell": "전역"}
    missing = []
    for k in keys:
        g = lec_global([r.get(k) for r in rows])
        if g is None:
            missing.append(k)
            continue
        out.update({f"{k}_{m}": v for m, v in g.items()})
        out[f"{k}_sum_over_delta"] = g["sum"] / DELTA
        out[f"{k}_abs_over_delta"] = g["abs"] / DELTA
    out["status"] = "산출" if not missing else "부분" if len(missing) < len(keys) else "미산출"
    out["reason"] = ("not formed (incomplete or non-finite shells): " + ",".join(missing)) if missing else ""
    return out


def lec_branch(valid_all, separated):
    """0 PROJECTED_GAP / 1 NO_RESOLVED_PROJECTED_GAP need every mandatory calculation valid; otherwise 2."""
    if not valid_all or any(v is None for v in separated):
        return 2
    return 0 if any(separated) else 1


# ------------------------------------------------------------------ pipeline (shared by analyze and selftest)
def lec_pipeline(pc, cap, cells, bridge, ns=NS, nb=NB):
    """inputs -> states -> relations; common defects stop every dependent state before any integral is formed."""
    why10, why11, whyc = [], [], []
    A10 = lec_read_dump(pc, 10, why10, ns, nb)
    A11 = lec_read_dump(pc, 11, why11, ns, nb)
    C = lec_read_capture(cap, cells, bridge, whyc, ns, nb)
    checks = {"A10_A11_geometry_equal": bool(np.array_equal(A10["ri"], A11["ri"]) and
                                             np.array_equal(A10["ro"], A11["ro"])),
              "A10_A11_grid_equal": bool(np.array_equal(A10["edges"], A11["edges"])),
              "capture_geometry_equal": bool(len(C["ri"]) == ns and np.array_equal(C["ri"], A10["ri"]) and
                                             np.array_equal(C["ro"], A10["ro"])),
              "capture_support_ends_equal": [bool(C["nu_min"] == A10["edges"][0]),
                                             bool(C["nu_max"] == A10["edges"][-1])]}
    common = {"A10": list(why10), "A11": list(why11), "C13": list(whyc), "pairs": []}
    if not checks["A10_A11_geometry_equal"]:
        common["pairs"].append("A10_A11_GEOMETRY")
    if not checks["A10_A11_grid_equal"]:
        common["pairs"].append("A10_A11_GRID")
    if not checks["capture_geometry_equal"]:
        common["C13"].append("C13_GEOMETRY")
    if not all(checks["capture_support_ends_equal"]):
        common["C13"].append("C13_SUPPORT_ENDS")
    stop10, stop11 = bool(common["A10"]), bool(common["A11"])
    stopx = stop10 or stop11 or bool(common["pairs"])
    stopc = bool(common["C13"])
    q10 = None if stop10 else lec_dump_q(A10)
    q11 = None if stop11 else lec_dump_q(A11)
    qx = None if stopx else lec_dump_q(A10, J=A11["arr"]["J_nu"], bad_J=A11["bad"]["J"])
    qc = None if stopc else lec_capture_q(C)
    sup = f"[{A10['edges'][0]:.10e}, {A10['edges'][-1]:.10e}] Hz"

    def stopped(rel, why):
        return [{"relation": rel, "shell": s, "status": "미산출", "reason": "COMMON_INPUT_DEFECT:" + ";".join(why[:3])}
                for s in range(ns)]
    rows, sep, ident = [], [], []
    for rel, d, q, w in (("D10", A10, q10, common["A10"]), ("D11", A11, q11, common["A11"])):
        if q is None:
            rows += stopped(rel, w)
            sep += [None] * ns
            ident += [None] * ns
        else:
            rr, ss, ii = lec_d_rows(d, q, rel, sup)
            rows += rr
            sep += ss
            ident += ii
    a10 = {"name": "A10 (J radiation 12)", "hist": A10["hist"], "geom": A10}
    x13 = {"name": "X13 (A10 coefficients, J radiation 13)", "hist": "mixed: A10 coefficients / A11 J (radiation 13)",
           "geom": A10}
    c13 = {"name": "C13 (capture representation)", "hist": C["hist"], "geom": A10}
    if q10 is None or qx is None:
        rows += stopped("J13", common["A10"] + common["A11"] + common["pairs"])
    else:
        rows += lec_pair_rows("J13", a10, x13, {"bb": q10["bb"], "cont": lec_cont(q10)},
                              {"bb": qx["bb"], "cont": lec_cont(qx)}, sup,
                              "계수 동일(A10), J 만 복사12→13 — 혼합 이력의 산술 대입, 완전한 13/13 장부 아님")
    if qx is None or qc is None:
        rows += stopped("O13", common["A10"] + common["A11"] + common["pairs"] + common["C13"])
    else:
        rows += lec_pair_rows("O13", x13, c13, {"bb": qx["bb"], "cont": lec_cont(qx)},
                              {"bb": qc["bb"], "cont": qc["cont"]}, sup,
                              "덤프 구간 경계 폭 대 포획 폭 ν·Δlnν(지원 양 끝 동일 검사), 선 집합·원천·재격자화 표현 "
                              "차 포함 — 동일성 미확인",
                              extra=lambda s: {"C13_E_dep": qc["dep"][s]["value"]["E"] if qc["dep"][s]["value"]
                                               else None})
    dkeys = ["L", "bb_Q", "line_diff", "bf_Q", "ff_Q", "Q_rad"]
    summaries = [lec_summary([r for r in rows if r["relation"] == "D10"], "D10", dkeys),
                 lec_summary([r for r in rows if r["relation"] == "D11"], "D11", dkeys),
                 lec_summary([r for r in rows if r["relation"] == "J13"], "J13",
                             ["bb_left_Q", "bb_right_Q", "bb_diff", "cont_left_Q", "cont_right_Q", "cont_diff"]),
                 lec_summary([r for r in rows if r["relation"] == "O13"], "O13",
                             ["bb_left_Q", "bb_right_Q", "bb_diff", "cont_left_Q", "cont_right_Q", "cont_diff"])]
    states = {"A10": q10, "A11": q11, "X13": qx, "C13": qc}
    per_shell = {}
    for name, q in states.items():
        if q is None:
            per_shell[name] = None
            continue
        per_shell[name] = {x: [r["value"] if r["value"] else {"undefined": r["reason"]} for r in v]
                           for x, v in q.items()}
    undefined = [{"state": n, "component": x, "shell": s, "reason": r["reason"]}
                 for n, q in states.items() if q for x, v in q.items() for s, r in enumerate(v) if r["value"] is None]
    return {"A10": A10, "A11": A11, "C": C, "checks": checks, "common": common, "rows": rows, "sep": sep,
            "ident": ident, "summaries": summaries, "per_shell": per_shell, "undefined": undefined}


# ------------------------------------------------------------------ selftest (prereg 9)
def lec_toy():
    """two shells x three bins with signed bb opacity and a hand-checked integral."""
    V = np.array([2.0, 3.0])
    dnu = np.array([1.0, 2.0, 4.0])
    chi = np.array([[1.0, -0.5, 0.25], [0.0, 2.0, 1.0]])
    eta = np.array([[0.5, 0.0, 1.0], [1.0, 0.0, 0.0]])
    J = np.array([[2.0, 1.0, 4.0], [3.0, 1.0, 0.5]])
    # shell 0: A = 4pi 2 (2 - 1 + 4) = 40 pi, E = 4pi 2 (0.5 + 4) = 36 pi; shell 1: A = 4pi 3 (4 + 2) = 72 pi,
    # E = 4pi 3 (1) = 12 pi
    return V, dnu, chi, eta, J, [(40 * math.pi, 36 * math.pi), (72 * math.pi, 12 * math.pi)]


def lec_write_toy(root, ns=2, nb=3, mutate=None):
    """toy dumps (iter 10 / 11) and a toy capture whose BF / FF ledger satisfy the identities and whose line ledger
    equals the transport net; mutate(kind, obj) may inject a defect into one written object."""
    mutate = mutate or (lambda kind, obj: obj)
    pc, cap = root / "pc", root / "cap"
    pc.mkdir()
    cap.mkdir()
    edges = [1.0e15 * math.exp(0.1 * k) for k in range(nb + 1)]
    ri, ro = [1.0e15 * (s + 1) for s in range(ns)], [1.0e15 * (s + 2) for s in range(ns)]
    V = [lec_volume(a, b) for a, b in zip(ri, ro)]
    for it in (10, 11):
        tag = f"A{it}"
        man = dict(GEN[tag], **DEF, n_bins=nb, n_shells=ns, radiative_integral_factor=FOUR_PI)
        (pc / f"physics_DET_iter{it:04d}.manifest.json").write_text(json.dumps(mutate(("manifest", it), man)))
        spec = []
        for s in range(ns):
            for b in range(nb):
                J = 1e-6 * (1 + s + b) * (1.1 if it == 11 else 1.0)
                spec.append({"shell_id": s, "bin_id": b, "nu_lo_Hz": repr(edges[b]), "nu_hi_Hz": repr(edges[b + 1]),
                             "J_nu": repr(J), "chi_es_cm1": "1e-15", "chi_bb_cm1": repr(2e-16 * (b + 1)),
                             "eta_bb": repr(3e-22 * (s + 1)), "chi_bf_cm1": repr(1e-17), "eta_bf": repr(1e-23),
                             "chi_ff_cm1": repr(5e-18), "eta_ff": repr(2e-24)})
        spec = mutate(("spectral", it), spec)
        shells = []
        for s in range(ns):
            w = [edges[b + 1] - edges[b] for b in range(nb)]
            sel = [r for r in spec if int(r["shell_id"]) == s and int(r["bin_id"]) < nb]

            def fl(t):
                try:
                    return float(t)
                except (TypeError, ValueError):
                    return 0.0

            def acc(c, e):
                A = FOUR_PI * sum(fl(r.get(c)) * fl(r.get("J_nu")) * w[int(r["bin_id"])] for r in sel)
                E = FOUR_PI * sum(fl(r.get(e)) * w[int(r["bin_id"])] for r in sel)
                return A, E
            (Ab, Eb), (Af, Ef), (Aff, Eff) = acc("chi_bb_cm1", "eta_bb"), acc("chi_bf_cm1", "eta_bf"), \
                acc("chi_ff_cm1", "eta_ff")
            Qb = Ab - Eb
            row = {k: "0" for k in LEDGER}
            row.update(shell_id=s, r_inner_cm=repr(ri[s]), r_outer_cm=repr(ro[s]), v_inner_cm_s=repr(ri[s] / EPOCH_S),
                       v_outer_cm_s=repr(ro[s] / EPOCH_S), photo_heat=repr(Af), recomb_cool=repr(Ef),
                       ff_abs_heat=repr(Aff), ff_emit_cool=repr(Eff), line_abs_heat=repr(max(Qb, 0.0)),
                       line_emit_cool=repr(max(-Qb, 0.0)), sum_heating="1", sum_cooling="1", residual="0")
            shells.append(row)
        shells = mutate(("shell", it), shells)
        for name, rows in (("shell", shells), ("spectral", spec)):
            with open(pc / f"physics_DET_iter{it:04d}.{name}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    meta = {"NS": ns, "NB": nb, "material_generation": 13, "rad_generation": 13, "tau_computed_generation": 26,
            "nu_min": edges[0].hex(), "nu_max": edges[-1].hex()}
    meta = mutate(("meta", 0), meta)
    (cap / "meta.txt").write_text("".join(f"{k}={v}\n" for k, v in meta.items()))
    np.array(ri, dtype="<f8").tofile(cap / "r_inner.f64")
    np.array(ro, dtype="<f8").tofile(cap / "r_outer.f64")
    cman = {"transaction": "COMMITTED", "layout": {"frame": "SHELL_COMOVING"},
            "generations": {"material": 13, "radiation": 13},
            "files": {n: {"sha256": sha(cap / n)} for n in ("meta.txt", "r_inner.f64", "r_outer.f64")}}
    (cap / "manifest.json").write_text(json.dumps(mutate(("capture_manifest", 0), cman)))
    (root / "bridge.json").write_text(json.dumps({"capture_manifest_sha256": sha(cap / "manifest.json")}))
    cells = []
    for s in range(ns):
        for b in range(nb):
            nu = math.sqrt(edges[b] * edges[b + 1])
            cells.append({"shell": s, "bin": b, "nu_Hz": repr(nu), "dnu_Hz": repr(nu * 0.1),
                          "J": repr(1e-6 * (1 + s + b) * 1.1), "chi_line_exp": repr(2e-16 * (b + 1)),
                          "chi_abs": repr(1.5e-17), "eta_line_used": repr(3e-22 * (s + 1)), "eta_bfff": repr(1.2e-23),
                          "eta_dep": "0.0"})
    cells = mutate(("cells", 0), cells)
    with open(root / "cells.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cells[0]))
        wr.writeheader()
        wr.writerows(cells)
    return pc, cap, root / "cells.csv", root / "bridge.json"


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
    expect(not close(q[0]["value"]["Q"] / FOUR_PI, want[0][0] - want[0][1]), "inject 4 pi omitted: differs")
    expect(not close(lec_integrate(np.ones(2), dnu, chi, eta, J, None)[1]["value"]["Q"], want[1][0] - want[1][1]),
           "inject volume omitted: differs")
    expect(not close(lec_integrate(V, np.ones(3), chi, eta, J, None)[0]["value"]["Q"], want[0][0] - want[0][1]),
           "inject bin width omitted: differs")
    logw = np.log(np.array([2.0, 3.0, 5.0])) - np.log(np.array([1.0, 2.0, 3.0]))
    expect(not close(lec_integrate(V, logw, chi, eta, J, None)[0]["value"]["Q"], want[0][0] - want[0][1]),
           "inject a log-frequency width: differs")
    expect(lec_sign(q[0]["value"]["Q"], q[0]["value"]["S"]) == "순흡수" and
           lec_sign(-q[0]["value"]["Q"], q[0]["value"]["S"]) == "순방출", "Q sign convention (positive = absorption)")
    expect(not close(lec_integrate(V, dnu, np.abs(chi), eta, J, None)[0]["value"]["Q"], q[0]["value"]["Q"]),
           "inject the valid negative opacity removed (absolute value): differs")
    chi_es = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    expect(not close(lec_integrate(V, dnu, chi + chi_es, eta, J, None)[0]["value"]["Q"], q[0]["value"]["Q"]),
           "inject chi_total J - eta_true_total (scattering source missing): differs")
    J2 = J * np.array([[1.5], [0.5]])
    x13 = lec_integrate(V, dnu, chi, eta, J2, None)
    expect(close(x13[0]["value"]["E"], q[0]["value"]["E"]) and
           not close(lec_integrate(V, dnu, chi, eta * 2, J2, None)[0]["value"]["E"], q[0]["value"]["E"]),
           "X13 keeps the emission integral; inject emission coefficients replaced: detected")
    expect(not close(lec_integrate(V, dnu, chi, eta, J2[::-1], None)[0]["value"]["A"], x13[0]["value"]["A"]) and
           not close(lec_integrate(V, dnu, chi, eta, np.roll(J2, 1, axis=1), None)[0]["value"]["A"],
                     x13[0]["value"]["A"]), "inject J of another shell / another bin: differs")
    tiny = lec_integrate(V, dnu, np.array([[5e-324, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.zeros((2, 3)),
                         np.array([[1e-300, 1.0, 1.0], [1.0, 1.0, 1.0]]), None)
    exact = lec_integrate(V, dnu, np.zeros((2, 3)), np.zeros((2, 3)), J, None)
    expect(tiny[0]["value"] is None and exact[0]["value"]["Q"] == 0.0, "inject a product of non-zero factors lost to "
           "0: undefined; exact zero stays a valid zero")
    huge = lec_integrate(V, dnu, np.full((2, 3), 1e300), np.zeros((2, 3)), np.full((2, 3), 1e300), None)
    expect(all(r["value"] is None and "DERIVED" in r["reason"] for r in huge) and
           lec_separated(float("inf"), float("inf"), 1.0) is None and lec_sign(float("nan"), 1.0) is None,
           "inject finite inputs whose product overflows: undefined, and no comparison on non-finite values")
    w = []
    expect(math.isnan(lec_num("1e-400", w, "x")) and w and lec_num("0.0", [], "x") == 0.0,
           "inject a non-zero stored text read as 0: undefined; a stored zero stays zero")
    expect(lec_sign(1.0, 1e11) == "부호 미분리" and lec_sign(1.0, 1.0) == "순흡수",
           "sign uses the state's own scale (positive control)")
    expect(lec_global([1.0, None]) is None and lec_global([1.0, float("nan")]) is None and
           lec_global([1.0, -2.0]) == {"sum": -1.0, "positive": 1.0, "negative": -2.0, "abs": 3.0},
           "inject a partial or non-finite shell set into a global sum: refused (complete set positive control)")
    expect(lec_branch(True, [False, True]) == 0 and lec_branch(True, [False, False]) == 1 and
           lec_branch(False, [True]) == 2 and lec_branch(True, [None, True]) == 2, "branch fixtures (positive control)")

    # ---- reader-path injections on toy files (job-local temporary directory)
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or None))
    try:
        def run(name, mutate=None):
            root = tmp / name
            root.mkdir()
            return lec_pipeline(*lec_write_toy(root, mutate=mutate), ns=2, nb=3)
        p = run("clean")
        expect(not any(p["common"].values()) and all(r["status"] == "산출" for r in p["rows"]) and
               all(s["status"] == "산출" for s in p["summaries"]) and p["sep"] == [False] * 4 and
               p["ident"] == [False] * 4, "toy files through the reader path: every row and global row produced, "
               "identities and line ledger unresolved (positive control)")
        r0 = [r for r in p["rows"] if r["relation"] == "D10" and r["shell"] == 0][0]
        expect(r0["line_separated"] is False and lec_separated(r0["line_abs_heat"], r0["bb_A"], r0["bb_S"]) and
               lec_separated(r0["line_emit_cool"], r0["bb_E"], r0["bb_S"]),
               "inject the sign-split line heat / cool read as the total absorption A / emission E: separated "
               "(the registered net comparison of the same reader row is unresolved)")

        def drop_key(key):
            return lambda kind, obj: ({k: v for k, v in obj.items() if k != key}
                                      if kind == ("manifest", 10) else obj)
        p = run("nokey", drop_key("frame"))
        expect(any("MISSING:frame" in w for w in p["common"]["A10"]) and
               all(r["status"] == "미산출" for r in p["rows"] if r["relation"] in ("D10", "J13", "O13")) and
               any(r["status"] == "산출" for r in p["rows"] if r["relation"] == "D11"),
               "inject a missing frame key into the A10 manifest: D10, J13, O13 stopped, D11 kept")
        p = run("units", lambda kind, obj: dict(obj, eta_is_per_sr=False) if kind == ("manifest", 11) else obj)
        expect(any("MISMATCH:eta_is_per_sr" in w for w in p["common"]["A11"]) and
               all(r["status"] == "미산출" for r in p["rows"] if r["relation"] in ("D11", "J13", "O13")),
               "inject eta not per sr into the A11 manifest: D11, J13, O13 stopped")
        p = run("gen", lambda kind, obj: dict(obj, opacity_generation=99) if kind == ("manifest", 10) else obj)
        expect(any("MISMATCH:opacity_generation" in w for w in p["common"]["A10"]), "inject another opacity "
               "generation: refused")
        p = run("col", lambda kind, obj: [{k: v for k, v in r.items() if k != "eta_ff"} for r in obj]
                if kind == ("spectral", 10) else obj)
        expect(any("MISSING_COLUMN:spectral:eta_ff" in w for w in p["common"]["A10"]) and
               all(r["status"] == "미산출" for r in p["rows"] if r["relation"] == "D10"),
               "inject a missing spectral column: common defect, no exception, D10 stopped")
        p = run("grid", lambda kind, obj: [dict(r, dnu_Hz=repr(float(r["dnu_Hz"]) * 1.5)) if r["shell"] == 1 else r
                                           for r in obj] if kind == ("cells", 0) else obj)
        expect(any("GRID_PER_SHELL" in w for w in p["common"]["C13"]) and
               all(r["status"] == "미산출" for r in p["rows"] if r["relation"] == "O13") and
               all(r["status"] == "산출" for r in p["rows"] if r["relation"] == "J13"),
               "inject a per-shell grid mismatch into the capture cells: common C13 defect, O13 stopped, J13 kept")
        p = run("bf", lambda kind, obj: [dict(r, eta_bf="bad") if (r["shell_id"], r["bin_id"]) == (1, 2) else r
                                         for r in obj] if kind == ("spectral", 10) else obj)
        d1 = [r for r in p["rows"] if r["relation"] == "D10" and r["shell"] == 1][0]
        d0 = [r for r in p["rows"] if r["relation"] == "D10" and r["shell"] == 0][0]
        g10 = p["summaries"][0]
        expect(not p["common"]["A10"] and "bf_Q" not in d1 and "bb_Q" in d1 and "ff_Q" in d1 and
               d1.get("line_separated") is False and "L" in d1 and d1["status"] == "부분" and d0["status"] == "산출" and
               g10["status"] == "부분" and "L_sum" in g10 and "line_diff_sum" in g10 and "bf_Q_sum" not in g10 and
               "Q_rad_sum" not in g10 and p["ident"][1] is None and p["ident"][0] is False,
               "inject one unparsable BF value: only BF of that shell, its identity, Q_rad and the dependent global "
               "sums undefined; bb, ff, ledger, the line comparison and the other global sums kept")
        p = run("capframe", lambda kind, obj: dict(obj, layout={"frame": "LAB"})
                if kind == ("capture_manifest", 0) else obj)
        expect(any("C13:FRAME" in w for w in p["common"]["C13"]) and
               all(r["status"] == "미산출" for r in p["rows"] if r["relation"] == "O13") and
               all(r["status"] == "산출" for r in p["rows"] if r["relation"] in ("D10", "D11", "J13")),
               "inject another frame into the capture manifest: O13 stopped, D10, D11, J13 kept")
        p = run("capfile", lambda kind, obj: dict(obj, files=dict(obj["files"], **{"r_outer.f64": {"sha256": "0"}}))
                if kind == ("capture_manifest", 0) else obj)
        expect(any("FILE_NOT_IN_MANIFEST:r_outer.f64" in w for w in p["common"]["C13"]),
               "inject capture geometry not registered in the capture manifest: refused")
        p = run("j", lambda kind, obj: [dict(r, J_nu="-1") if (r["shell_id"], r["bin_id"]) == (0, 0) else r
                                        for r in obj] if kind == ("spectral", 11) else obj)
        j0 = [r for r in p["rows"] if r["relation"] == "J13" and r["shell"] == 0][0]
        expect("bb_left_Q" in j0 and "bb_right_Q" not in j0 and j0["status"] == "부분",
               "inject an undefined A11 J value: X13 of that shell undefined, A10 side kept")
        expect(all("line_abs_heat" not in r for r in p["rows"] if r["relation"] in ("J13", "O13")),
               "X13 / C13 rows carry no copied ledger columns")
    finally:
        shutil.rmtree(tmp)
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_LINE_ENERGY_CORR_SELFTEST_V2", "checks": len(results), "failures": fails}))
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
    used = [PC / f"physics_DET_iter{it:04d}.{k}" for it in (10, 11) for k in ("manifest.json", "shell.csv",
                                                                               "spectral.csv")]
    used += [CAP / "manifest.json", CAP / "meta.txt", CAP / "r_inner.f64", CAP / "r_outer.f64", CELLS, BRIDGE]
    before = {str(p): sha(p) for p in used}
    p = lec_pipeline(PC, CAP, CELLS, BRIDGE)
    rows, summaries = p["rows"], p["summaries"]
    for k, w in p["common"].items():
        gates["G-INPUT"] += w[:8]
    ident_fail = sum(1 for v in p["ident"] if v)
    if ident_fail:
        gates["G-CALC"].append(f"BF_FF_IDENTITY:{ident_fail}")
    if any(v is None for v in p["ident"]):
        gates["G-CALC"].append("BF_FF_IDENTITY_NOT_FORMED")
    after = {str(q): sha(q) for q in used}
    if after != before:
        gates["G-INPUT"].append("INPUT_CHANGED_DURING_RUN")
    st = None
    if selftest_log:
        lines = [ln for ln in Path(selftest_log).read_text().splitlines() if ln.startswith("{")] \
            if Path(selftest_log).is_file() else []
        st = json.loads(lines[-1]) if lines else None
        if not st or st.get("failures") != 0:
            gates["G-CALC"].append("SELFTEST_NOT_PASSED")
    produced = all(r.get("status") == "산출" for r in rows) and all(s["status"] == "산출" for s in summaries)
    all_gates = [f for v in gates.values() for f in v]
    code = lec_branch(produced and not all_gates, p["sep"])
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
    cols = {"D": ["relation", "shell", "v_inner_cm_s", "L", "bb_A", "bb_E", "bb_Q", "bb_sign", "line_diff",
                  "line_diff_over_delta", "line_separated", "bf_identity_separated", "ff_identity_separated", "Q_rad",
                  "Q_rad_sign", "R_over_HC", "status", "reason"],
            "P": ["relation", "shell", "v_inner_cm_s", "bb_left_A", "bb_left_E", "bb_left_Q", "bb_right_A",
                  "bb_right_E", "bb_right_Q", "bb_right_sign", "bb_diff", "bb_separated", "cont_left_Q",
                  "cont_right_Q", "cont_right_sign", "cont_diff", "cont_separated", "C13_E_dep", "status", "reason"]}
    with open(out / "shell_energy_correspondence.md", "x") as f:
        f.write(f"# Thermal ledger vs transport energy exchange, O 19.48 d (branch {code} {label})\n\n")
        f.write(f"Context: Delta (diag9 2500-6000 A deficit) = {DELTA:.6e} erg/s. Positive Q = radiation to matter. "
                "D rows compare the sign-split line ledger L = V(line_abs_heat - line_emit_cool) with "
                "Q_bb = 4 pi V sum (chi_bb J - eta_bb) dnu of the same dump; J13 = A10 vs X13 (coefficients fixed, "
                "J radiation 12 -> 13, mixed history); O13 = X13 vs the capture bin representation C13 (its positive "
                "Q is a net absorption of the stored bin representation, not unreemitted energy of observed rays). "
                "Arithmetic decomposition, not causal shares. Every state's A, E, Q, S per shell and component is in "
                "the CSV and in evidence.json.\n\n")
        f.write("Structural nulls: " + "; ".join(f"{k} = {v}" for k, v in STRUCTURAL.items()) + "\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        for rel, key in (("D10", "D"), ("D11", "D"), ("J13", "P"), ("O13", "P")):
            f.write(f"## {rel}\n\n| " + " | ".join(cols[key]) + " |\n|" + "---|" * len(cols[key]) + "\n")
            for r in [r for r in rows if r["relation"] == rel]:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols[key]) + " |\n")
            f.write("\n")
        f.write("## Global rows\n\n")
        for s in summaries:
            f.write("- " + ", ".join(f"{k}={fmt(v)}" for k, v in s.items()) + "\n")
    A10, A11, C = p["A10"], p["A11"], p["C"]
    evidence = {"schema": "DET_LINE_ENERGY_CORR_V2", "prereg": PREREG, "argv": sys.argv,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "branch": code, "label": label, "gates": gates,
                "D_separated_shells": n_sep, "BF_FF_identity_failures": ident_fail,
                "histories": {"A10": A10["hist"], "A11": A11["hist"], "X13": "A10 coefficients with the A11 J",
                              "C13": C["hist"]},
                "support_Hz": [float(A10["edges"][0]), float(A10["edges"][-1])],
                "checks": dict(p["checks"],
                               capture_dnu_over_dump_dnu=[float(np.min(C["dnu"] / A10["dnu"])),
                                                          float(np.max(C["dnu"] / A10["dnu"]))],
                               capture_J_equals_A11_J=bool(np.array_equal(C["arr"]["J"], A11["arr"]["J_nu"]))),
                "common_defects": p["common"], "undefined": p["undefined"],
                "value_notes": {"A10": A10["notes"][:20], "A11": A11["notes"][:20]},
                "structural_nulls": STRUCTURAL,
                "per_shell_integrals": p["per_shell"],
                "tolerance_rule": "1e-10 x pre-cancellation absolute sum; sign of one state against its own scale",
                "selftest": {"log": selftest_log, "summary": st},
                "inputs": {q: {"bytes": Path(q).stat().st_size, "sha256": h} for q, h in before.items()},
                "inputs_unchanged": after == before,
                "code": {str(Path(__file__).resolve()): sha(Path(__file__)),
                         str((D9 / "tools/r1_common.py").resolve()): sha(D9 / "tools/r1_common.py")}}
    write_json(out / "evidence.json", evidence)
    print(f"BRANCH {code} {label} D_separated={json.dumps(n_sep)} identity_failures={ident_fail}")
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
