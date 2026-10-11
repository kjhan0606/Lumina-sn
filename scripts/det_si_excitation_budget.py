#!/usr/bin/env python3
"""DET-SI-EXCITATION-BUDGET (prereg docs/PREREG_DET_SI_EXCITATION_BUDGET_2026-10-11.md).

Fixed-field Si II level budgets at the adopted consumption state of the DET-SI-CONSUMPTION-TRACE run T (material
generation 13, Si shell 10, call 36) and two separate interventions with CMFGEN-footprint channels:
  C1  collisional ionization + 3-body recombination of every Si II level to the Si III ground with the sealed
      Lumina pair rate (ARTIS form; its gate is off in O)
  C2  the tabulated Si II collision strengths interpolated log-log (CMFGEN OMEGA_GEN_V3) instead of linearly
States: adopted SE (FINAL_FL of the adopted call), F0 (steady state of the recorded FL operator), F1 (C1 only),
F2 (C2 only), each against the public CMFGEN Si III/II at 19.480 d.

Subcommands
  negative                    negative controls of the new functions (prereg 9)
  analyze --run T --out DIR   G-INPUT, G-RATE, G-SOLVE, G-TABLE, G-RUN and the main table
Exit codes: 0 pass, 2 refused or gate finding, 3 existing output path.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_si_trace_readout as TR                                                 # noqa: E402

PREREG = "docs/PREREG_DET_SI_EXCITATION_BUDGET_2026-10-11.md"
PREVIOUS = Path("/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/analyze/analysis/evidence.json")
SELECTED = (0, 1, 6, 7, 9, 10, 42, 44, 52, 124)
ADOPTED = {"bundle": 1, "ne_iter": 18, "ce_pass": 1, "call": 36}
SHELL = 10
F0_TOL_DEX, ETA_MAX, APPROACH_DEX, INPUT_RTOL = 0.001, 1e-8, 0.01, 1e-9
H_PLANCK, K_BOLTZMANN, M_ELECTRON = 6.62607015e-27, 1.380649e-16, 9.1093837015e-28
M_PI_VAL, EV_TO_ERG = 3.14159265358979323846, 1.602176634e-12
SIGMA_MAGIC, COL_MAGIC = 0x434D4644, 0x49474331
NFB, NU_MIN, NU_MAX = 1234, 5.8412785919616062e13, 4.0362581455823112e16
G_COL_SI_II = 0.2                 # sealed pair branch: g_col for ion stage 1
NSTAR_CAP, U_MAX = 1e30, 700.0    # sealed pair limits; a level where either binds is undefined for C1
CMFGEN = {"C1": "SUBCOL_MULTI_V6 direct collisional ionization; STEQ_MULTI_V4 upper-ion ground, LTE inverse",
          "C2": "OMEGA_GEN_V3 log-log interpolation of tabulated Omega"}
CHANNEL_ORDER = ("BB radiative upward", "BB radiative downward", "BB collisional (line)", "collisional (tabulated)",
                 "BF ionization", "BF recombination", "DR", "CE", "NT",
                 "C1 collisional ionization", "C1 3-body recombination")


# ------------------------------------------------------------------ inputs and rate statements
def det_si_exc_sigma_edges(path, levels):
    """the pair's sig_edge per sigma row: first positive bin of the CMFGEN row (sealed loader: row = global level);
    None when the level has no CMFGEN row (the sealed pair would fall back to Kramers, outside prereg 3)."""
    with open(path, "rb") as f:
        magic, version, n_lev, n_freq = struct.unpack("<IIii", f.read(16))
        nu_min, nu_max = struct.unpack("<dd", f.read(16))
        if magic != SIGMA_MAGIC or version != 1 or n_freq != NFB or nu_min != NU_MIN or nu_max != NU_MAX:
            raise ValueError("SIGMA_HEADER")
        flags = np.frombuffer(f.read(n_lev), dtype=np.int8)
        pad = (8 - n_lev % 8) % 8
        if f.read(pad) != b"\0" * pad:
            raise ValueError("SIGMA_PAD")
        base = 32 + n_lev + pad
        out = {}
        for gl in levels:
            if not (0 <= gl < n_lev) or flags[gl] != 1:
                out[gl] = None
                continue
            f.seek(base + gl * n_freq * 8)
            row = np.frombuffer(f.read(n_freq * 8), dtype="<f8")
            pos = np.nonzero(row > 0.0)[0]
            out[gl] = float(row[pos[0]]) if len(pos) else None
    return out


def det_si_exc_coltable(path):
    with open(path, "rb") as f:
        magic, version, z, ion0, n_trans, n_temp, n_lev = struct.unpack("<IIiiiii", f.read(28))
        if magic != COL_MAGIC or version != 1 or not (0 < n_temp <= 256) or n_trans <= 0:
            raise ValueError("COL_HEADER")
        tg = np.frombuffer(f.read(8 * n_temp), dtype="<f8").copy()
        recs = []
        for _ in range(n_trans):
            lo, hi = struct.unpack("<ii", f.read(8))
            recs.append((lo, hi, np.frombuffer(f.read(8 * n_temp), dtype="<f8").copy()))
        if f.read(1):
            raise ValueError("COL_TRAILING_BYTES")
    return {"Z": z, "ion0": ion0, "n_lev": n_lev, "tgrid": tg, "records": recs}


def det_si_exc_bracket(tg, T):
    """the sealed col_data bracket: ti and the linear fraction before its clamp."""
    nt, ti = len(tg), 0
    while ti < nt - 2 and T > tg[ti + 1]:
        ti += 1
    denom = tg[ti + 1] - tg[ti]
    return ti, ((T - tg[ti]) / denom if denom > 0.0 else 0.0)


def det_si_exc_omega_lin(tg, om, T):
    """the sealed statement, reproduced only to identify the recorded Omega (its clamp included)."""
    ti, frac = det_si_exc_bracket(tg, T)
    frac = min(max(frac, 0.0), 1.0)
    return om[ti] + frac * (om[ti + 1] - om[ti])


def det_si_exc_omega_log(tg, om, T):
    """OMEGA_GEN_V3 form inside the bracket; None outside the table or for a non-positive bracket value."""
    ti, _ = det_si_exc_bracket(tg, T)
    ta, tb, oa, ob = tg[ti], tg[ti + 1], om[ti], om[ti + 1]
    if T == ta:
        return oa if oa > 0.0 else None
    if T == tb:
        return ob if ob > 0.0 else None
    if not (ta < T < tb) or not (oa > 0.0 and ob > 0.0):
        return None
    return oa * (T / ta) ** (math.log(ob / oa) / math.log(tb / ta))


def det_si_exc_coll_rates(omega, T, n_e, g_lo, g_hi, dE):
    """the sealed col_data rate statement with Omega as the only input changed."""
    inv_sqrt = 1.0 / math.sqrt(T)
    kTe = K_BOLTZMANN * T
    c_up = n_e * 8.629e-6 * inv_sqrt / g_lo * omega * math.exp(-dE / kTe)
    c_down = n_e * 8.629e-6 * inv_sqrt / g_hi * omega
    return c_up, c_down


def det_si_exc_saha(T, n_e, g_i, g_g, nu):
    """the sealed pair's n_star_ratio from first principles: (n_i / n_g+)_LTE = n_e lambda^3 g_i/(2 g_g) e^(h nu/kT)."""
    if g_g < 1 or g_i < 1:
        return float("nan")
    lam3 = (H_PLANCK * H_PLANCK / (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T)) ** 1.5
    return n_e * lam3 * g_i / (2.0 * g_g) * math.exp(H_PLANCK * nu / (K_BOLTZMANN * T))


def det_si_exc_c1(levels, T, n_e, gh_fl, strength=1.0):
    """C1 statements for levels = [(fl, sig_edge, nu_thresh, nstar_recorded)]; returns (extra, undefined).
    C_ion = n_e 1.55e13 / sqrt(T) g_col sig_edge e^-u / u and C_rec = C_ion nstar (the sealed pair statement)."""
    extra, undefined = [], []
    for fl, sig, nu, nstar in levels:
        if sig is None or not (sig > 0.0):
            undefined.append(f"NO_CMFGEN_SIGMA_EDGE:fl{fl}")
            continue
        if not (nstar < NSTAR_CAP):
            undefined.append(f"NSTAR_CAP_BINDS:fl{fl}")
            continue
        u = H_PLANCK * nu / (K_BOLTZMANN * T)
        if not (0.0 < u < U_MAX):
            undefined.append(f"U_OUTSIDE_BRANCH:fl{fl}")
            continue
        c_ion = n_e * 1.55e13 / math.sqrt(T) * G_COL_SI_II * sig * math.exp(-u) / u * strength
        c_rec = c_ion * nstar
        if not (math.isfinite(c_ion) and math.isfinite(c_rec) and c_ion >= 0.0 and c_rec >= 0.0):
            undefined.append(f"C1_NONFINITE:fl{fl}")
            continue
        extra.append((gh_fl, fl, c_ion, "C1 collisional ionization"))
        extra.append((fl, gh_fl, c_rec, "C1 3-body recombination"))
    return extra, undefined


def det_si_exc_detailed_balance(extra, saha_by_fl, rtol=INPUT_RTOL):
    """every C1 pair is one reversible channel with C_rec / C_ion = S_i from first principles."""
    ion = {s: r for d, s, r, n in extra if n == "C1 collisional ionization"}
    rec = {d: r for d, s, r, n in extra if n == "C1 3-body recombination"}
    bad = sorted(set(ion) ^ set(rec))
    for fl in sorted(set(ion) & set(rec)):
        if not math.isclose(rec[fl], ion[fl] * saha_by_fl[fl], rel_tol=rtol, abs_tol=0.0):
            bad.append(fl)
    return bad


def det_si_exc_conservation(A, rtol=1e-12):
    """columns of a rate generator sum to zero (each transition debited on its source diagonal)."""
    scale = np.abs(A).max(axis=0)
    return [int(j) for j in np.nonzero(np.abs(A.sum(axis=0)) > rtol * scale)[0]]


def det_si_exc_closed_classes(A):
    """number of closed communicating classes of the generator (edge j -> i when A[i, j] > 0, i != j)."""
    n = A.shape[0]
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    succ = [np.nonzero(off[:, j] > 0.0)[0].tolist() for j in range(n)]
    pred = [np.nonzero(off[i, :] > 0.0)[0].tolist() for i in range(n)]
    seen, order = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, iter(succ[s]))]
        while stack:
            v, it = stack[-1]
            for w in it:
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, iter(succ[w])))
                    break
            else:
                stack.pop()
                order.append(v)
    comp, nc = [-1] * n, 0
    for s in reversed(order):
        if comp[s] >= 0:
            continue
        comp[s] = nc
        stack = [s]
        while stack:
            v = stack.pop()
            for w in pred[v]:
                if comp[w] < 0:
                    comp[w] = nc
                    stack.append(w)
        nc += 1
    closed = [True] * nc
    for j in range(n):
        for i in succ[j]:
            if comp[i] != comp[j]:
                closed[comp[j]] = False
    return sum(closed)


def det_si_exc_solve(A, total):
    """GTH steady state scaled to the pair total with the registered validity conditions; (x, why, eta_max)."""
    if not np.isfinite(A).all():
        return None, ["NONFINITE_RATE"], float("nan")
    off = A - np.diag(np.diag(A))
    if (off < 0).any():
        return None, ["NEGATIVE_RATE"], float("nan")
    bad = det_si_exc_conservation(A)
    if bad:
        return None, [f"NOT_CONSERVATIVE:{len(bad)}"], float("nan")
    k = det_si_exc_closed_classes(A)
    if k != 1:
        return None, [f"CLOSED_CLASSES:{k}"], float("nan")
    x, w = TR.gth(A)
    if x is None:
        return None, [w], float("nan")
    x = x * total
    eta = TR.backward(A, x)
    eta_max = float(np.nanmax(eta)) if np.isfinite(eta).any() else float("nan")
    why = []
    if not np.isfinite(x).all() or (x < 0).any():
        why.append("NONFINITE_OR_NEGATIVE")
    if not math.isclose(float(x.sum()), total, rel_tol=1e-12):
        why.append("TOTAL_NOT_CONSERVED")
    if not (eta_max <= ETA_MAX):
        why.append(f"BACKWARD_ERROR:{eta_max:.3g}")
    return (None if why else x), why, eta_max


def det_si_exc_budget(channels, n, i, n_ii, f_lte):
    """I, D, O, B, H of level i per channel (prereg 4)."""
    rows = {}
    for name, M in channels.items():
        inflow = float(M[i, :] @ n - M[i, i] * n[i])
        D = float(-M[i, i])
        rows[name] = {"I": inflow, "D": D, "O": D * n[i], "B": inflow - D * n[i], "H": inflow - D * n_ii * f_lte}
    return rows


# ------------------------------------------------------------------ correspondence
def det_si_exc_adopt(ne_rows, calls, cevec, commit, registered):
    """adoption recomputed from the trace (accepted n_e iteration, its last CE pass, commit lineage)."""
    why = []
    last_bundle = max(int(r["bundle"]) for r in ne_rows)
    acc = [r for r in ne_rows if int(r["bundle"]) == last_bundle and r["accepted"] == "1"]
    if len(acc) != 1:
        return None, [f"ACCEPTED_NE_ITERATIONS:{len(acc)}"]
    ne_acc = int(acc[0]["ne_iter"])
    cc = sorted((c for c in calls if int(c["bundle"]) == last_bundle and int(c["ne_iter"]) == ne_acc),
                key=lambda c: int(c["ce_pass"]))
    if not cc:
        return None, ["NO_CALL_FOR_ACCEPTED_NE"]
    passes = [int(c["ce_pass"]) for c in cc]
    if passes != list(range(len(passes))):
        why.append(f"CE_PASS_SEQUENCE:{passes}")
    a = cc[-1]
    keys = {k: int(a[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")}
    if keys != registered:
        why.append(f"ADOPTED_KEYS:{keys}")
    cm = {r["kind"]: r["value"] for r in commit if r["kind"] != "level"}
    commit_fl = [float(r["value"]) for r in sorted((r for r in commit if r["kind"] == "level"),
                                                   key=lambda r: int(r["i"]))]
    after = [float(r["after"]) for r in sorted(
        (r for r in cevec if int(r["bundle"]) == keys["bundle"] and int(r["ne_iter"]) == keys["ne_iter"]
         and int(r["ce_pass"]) == keys["ce_pass"]), key=lambda r: int(r["i_fl"]))]
    if not after or after != commit_fl:
        why.append("COMMIT_LINEAGE")
    if float(cm.get("n_e", "nan")) != float(a["n_e"]) or float(cm.get("T_e", "nan")) != float(a["T_e"]):
        why.append("COMMIT_NE_TE")
    return (None if why else a), why


def det_si_exc_level_map(sm, wanted):
    """level_num -> i_fl of the lower ion; None when a selected level is absent or a level_num repeats."""
    m = {}
    for r in sm:
        if r["upper_ion"] == "0":
            k = int(r["level_num"])
            if k in m:
                return None
            m[k] = int(r["i_fl"])
    return m if all(k in m for k in wanted) else None


def det_si_exc_match_coldata(ops, records, usable):
    """pair the table (production order, production skips) with the recorded lower-ion COLDATA statements."""
    seen, why = set(), []
    for lo, hi, _ in records:
        if (lo, hi) in seen:
            why.append(f"DUPLICATE_TABLE_PAIR:{lo}-{hi}")
        seen.add((lo, hi))
    if why:
        return None, why
    ops = sorted(ops, key=lambda r: int(r["op"]))
    out, k = {}, 0
    for t, rec in enumerate(records):
        if not usable(rec):
            continue
        if k >= len(ops):
            why.append("TABLE_LONGER_THAN_RECORD")
            break
        r = ops[k]
        if (int(r["lev_b"]), int(r["lev_a"])) != (rec[0], rec[1]) or r["op"] in out:
            why.append(f"ORDER_MISMATCH:op{r['op']}:{rec[0]}-{rec[1]}")
            break
        out[r["op"]] = t
        k += 1
    if not why and k != len(ops):
        why.append(f"RECORD_NOT_MATCHED:{len(ops) - k}")
    return (None if why else out), why


# ------------------------------------------------------------------ negative controls (prereg 9)
def det_si_exc_toy():
    """two lower-ion levels in separate SLs and the upper ground: one COLDATA and two BF statements."""
    T, n_e, g = 1.5e3, 1e9, (2, 4, 1)
    E = (0.0, 0.1, 0.0)
    tg, om = np.array([1e3, 2e3, 4e3]), np.array([1.0, 4.0, 3.0])          # log-log != linear inside [1e3, 2e3]
    sm = [{"i_fl": str(i), "sl_local": str(i), "within_sl_frac": "1", "fl_index": str(i), "level_num": str(ln),
           "upper_ion": str(u), "g": str(g[i]), "E_eV": str(E[i])}
          for i, (ln, u) in enumerate(((0, 0), (1, 0), (0, 1)))]
    om_lin = det_si_exc_omega_lin(tg, om, T)
    cu, cd = det_si_exc_coll_rates(om_lin, T, n_e, 2, 4, 0.1 * EV_TO_ERG)

    def op(k, site, dst, src, v1, v2, la, lb, x):
        r = {"op": str(k), "site": str(site), "dst": str(dst), "src": str(src), "v1": repr(float(v1)), "v2": repr(float(v2)),
             "lev_a": str(la), "lev_b": str(lb)}
        r.update({f"x{j + 1}": (repr(float(x[j])) if j < len(x) else "") for j in range(14)})
        return r
    ops = [op(1, 2, 1, 0, cu, cd, 1, 0, (cu, cd, 1.0, 1.0, om_lin)),
           op(2, 5, 2, 0, 3.0, 0.5, 0, 0, (3.0, 1.0, 0.5, 0.0, 1e-3, 1e15)),
           op(3, 5, 2, 1, 7.0, 0.2, 1, 1, (7.0, 1.0, 0.2, 0.0, 2e-3, 9e14))]
    return {"T": T, "n_e": n_e, "tg": tg, "om": om, "om_lin": om_lin, "sm": sm, "ops": ops, "N": 3, "gh": 2}


def cmd_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    t = det_si_exc_toy()
    A, _, und, chs, _ = TR.det_si_exc_build_fl(t["ops"], t["sm"], t["N"], t["gh"])
    x, why, _ = det_si_exc_solve(A, 10.0)
    expect(not und and x is not None and not why, "toy operator: unique valid steady state (positive control)")
    expect(np.array_equal(sum(chs.values()), A), "toy operator: channel matrices sum to the operator")

    # 1. wrong call / level mapping, duplicate C2 target
    ne = [{"bundle": "1", "ne_iter": "18", "accepted": "1"}]
    calls = [{"bundle": "1", "ne_iter": "18", "ce_pass": p, "call": c, "n_e": "1", "T_e": "1"}
             for p, c in (("0", "35"), ("1", "36"))]
    cev = [{"bundle": "1", "ne_iter": "18", "ce_pass": "1", "i_fl": "0", "after": "5"}]
    com = [{"kind": "level", "i": "0", "value": "5"}, {"kind": "n_e", "i": "0", "value": "1"},
           {"kind": "T_e", "i": "0", "value": "1"}]
    expect(det_si_exc_adopt(ne, calls, cev, com, ADOPTED)[0] is not None,
           "adoption: registered keys accepted (positive control)")
    expect(det_si_exc_adopt(ne, calls[:1], cev, com, ADOPTED)[0] is None, "inject wrong adopted call: refused")
    com_bad = [dict(r, value="6") if r["kind"] == "level" else r for r in com]
    expect(det_si_exc_adopt(ne, calls, cev, com_bad, ADOPTED)[0] is None, "inject broken commit lineage: refused")
    expect(det_si_exc_level_map(t["sm"][:1], (0, 1)) is None, "inject missing selected level: refused")
    expect(det_si_exc_level_map(t["sm"][:2] + [dict(t["sm"][1], i_fl="9")], (0, 1)) is None,
           "inject repeated level number: refused")
    recs = [(0, 1, t["om"])]
    ok_m, _ = det_si_exc_match_coldata([t["ops"][0]], recs, lambda r: True)
    expect(ok_m == {"1": 0}, "C2 target matching (positive control)")
    m, w = det_si_exc_match_coldata([t["ops"][0]], recs + [(0, 1, t["om"])], lambda r: True)
    expect(m is None and any("DUPLICATE" in s for s in w), "inject duplicate C2 table target: refused")
    m, _ = det_si_exc_match_coldata([t["ops"][0], dict(t["ops"][0], op="4")], recs, lambda r: True)
    expect(m is None, "inject duplicate recorded C2 statement: refused")

    # 2. negative rate, non-positive Omega, out-of-range temperature, disconnected system
    B = A.copy()
    B[1, 0] -= 2.0 * A[1, 0] + 1.0
    B[0, 0] += 2.0 * A[1, 0] + 1.0
    expect(det_si_exc_solve(B, 10.0)[0] is None, "inject negative rate: no valid steady state")
    expect(det_si_exc_omega_log(t["tg"], np.array([0.0, 1.0, 1.0]), t["T"]) is None,
           "non-positive Omega: C2 undefined")
    expect(det_si_exc_omega_log(t["tg"], t["om"], 5e3) is None, "temperature above the table: C2 undefined")
    expect(det_si_exc_omega_log(t["tg"], t["om"], 5e2) is None, "temperature below the table: C2 undefined")
    expect(det_si_exc_omega_log(t["tg"], t["om"], 2e3) == 4.0,
           "temperature on a table node: node value (positive control)")
    D = np.zeros((4, 4))
    for d, s_, r in ((1, 0, 1.0), (0, 1, 1.0), (3, 2, 1.0), (2, 3, 1.0)):
        D[d, s_] += r
        D[s_, s_] -= r
    expect(det_si_exc_closed_classes(D) == 2 and det_si_exc_solve(D, 10.0)[0] is None,
           "disconnected system: two closed classes, no unique steady state")

    # 3. C1 strength 0 and the original C2 prescription leave the operator unchanged
    lv = [(0, 1e-18, 1e15, 1e-3), (1, 2e-18, 9e14, 2e-3)]
    extra0, _ = det_si_exc_c1(lv, 1e4, t["n_e"], 2, strength=0.0)
    A0 = TR.det_si_exc_build_fl(t["ops"], t["sm"], t["N"], t["gh"], extra=extra0)[0]
    expect(len(extra0) == 4 and np.array_equal(A0, A), "C1 strength 0: operator unchanged")
    same = det_si_exc_coll_rates(t["om_lin"], t["T"], t["n_e"], 2, 4, 0.1 * EV_TO_ERG)
    A2 = TR.det_si_exc_build_fl(t["ops"], t["sm"], t["N"], t["gh"], coldata_override=lambda r, x: same)[0]
    expect(np.array_equal(A2, A), "C2 with the original interpolation: operator unchanged")
    log_r = det_si_exc_coll_rates(det_si_exc_omega_log(t["tg"], t["om"], t["T"]), t["T"], t["n_e"], 2, 4,
                                  0.1 * EV_TO_ERG)
    A2b = TR.det_si_exc_build_fl(t["ops"], t["sm"], t["N"], t["gh"], coldata_override=lambda r, x: log_r)[0]
    expect(not np.array_equal(A2b, A) and not det_si_exc_conservation(A2b),
           "C2 log interpolation: operator changed and conservative (positive control)")

    # 4. missing n_e factor in the reverse rate, transition sign error
    T1 = 1e4
    saha = {0: det_si_exc_saha(T1, t["n_e"], 2, 1, 1e15)}
    extra1, _ = det_si_exc_c1([(0, 1e-18, 1e15, saha[0])], T1, t["n_e"], 2)
    expect(len(extra1) == 2 and not det_si_exc_detailed_balance(extra1, saha),
           "C1 detailed balance (positive control)")
    bad = [(d, s_, r / t["n_e"] if n == "C1 3-body recombination" else r, n) for d, s_, r, n in extra1]
    expect(bool(det_si_exc_detailed_balance(bad, saha)), "inject missing n_e in the reverse rate: refused")
    expect(bool(det_si_exc_detailed_balance(extra1[:1], saha)), "inject one-way C1 statement: refused")
    S = A.copy()
    S[2, 0] += 1.0                                  # transition without its source debit
    expect(bool(det_si_exc_conservation(S)) and det_si_exc_solve(S, 10.0)[0] is None,
           "inject transition without source debit: conservation refuses")
    S = A.copy()
    S[0, 0] += 2.0 * 3.0                            # BF ionization debit with the wrong sign
    expect(bool(det_si_exc_conservation(S)), "inject debit sign error: conservation refuses")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_SI_EXC_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
def det_si_exc_public(vel):
    """the registered reader for the ratio; the same public row for the two fractions."""
    R, v = TR.public_si_ratio(vel)
    spec = importlib.util.spec_from_file_location("diag8_public", TR.DIAG8_PUBLIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = next(dd for vv, dd in mod.public_ions(TR.PUBLIC_SI) if vv == v)
    return R, v, float(d["si1"]), float(d["si2"])


def cmd_analyze(run, out):
    run = Path(run)
    tr = run / "si_trace"
    out = TR.new_dir(out)
    gates = {k: [] for k in ("G-INPUT", "G-RATE", "G-SOLVE", "G-TABLE", "G-RUN")}
    part, cvd = os.environ.get("SLURM_JOB_PARTITION"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if part != "a10" or cvd not in ("", "NoDevFiles"):
        gates["G-RUN"].append(f"NOT_A10_CPU:{part}:{cvd}")
    footer = dict(ln.split("=", 1) for ln in (run / "RUN_FOOTER.txt").read_text().splitlines() if "=" in ln)
    stop_iter = int(footer.get("stop_iteration", "-1"))
    if footer.get("terminal_state") != "TRACE_TARGET_COMPLETE" or stop_iter < 0:
        gates["G-INPUT"].append("TRACE_RUN_NOT_COMPLETE")

    # ---- G-INPUT: adoption, replay, level map, recorded inputs
    calls = TR.read_trace(tr / "si10_calls.csv")
    call, why = det_si_exc_adopt(TR.read_trace(tr / "si10_ne.csv"), calls, TR.read_trace(tr / "si10_cevec.csv"),
                                 TR.read_trace(tr / "si10_commit.csv"), ADOPTED)
    if call is None:
        return TR.refuse(why)
    cid = call["call"]
    N, N_fl, n_lo, gh = int(call["N"]), int(call["N_fl"]), int(call["n_lo_super"]), int(call["ground_hi"])
    T_e, n_e = float(call["T_e"]), float(call["n_e"])
    ops = [r for r in TR.read_trace(tr / "si10_ops.csv") if r["call"] == cid]
    g_a, g_b, _ = TR.read_matrix(tr / f"si10_mat_{int(cid):06d}_gen.bin")
    r_a, r_b = TR.replay(ops, N)
    if not (np.array_equal(r_a, g_a) and np.array_equal(r_b, g_b)):
        gates["G-INPUT"].append("REPLAY_NOT_GENERATOR")
    sm = sorted((r for r in TR.read_trace(tr / "si10_slmap.csv") if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    lvmap = det_si_exc_level_map(sm, SELECTED)
    if lvmap is None or len(sm) != N_fl:
        return TR.refuse(["LEVEL_MAP"])
    fin = np.zeros(N_fl)
    seen_fin = set()
    for r in TR.read_trace(tr / "si10_vec.csv"):
        if r["call"] == cid and r["phase"] == "FINAL_FL":
            fin[int(r["i"])] = float(r["value"])
            seen_fin.add(int(r["i"]))
    if seen_fin != set(range(N_fl)):
        return TR.refuse(["FINAL_FL_INCOMPLETE"])
    total = float(next(r for r in TR.read_trace(tr / "si10_final.csv") if r["call"] == cid)["n_total"])
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    E = np.array([float(r["E_eV"]) for r in sm])
    g = np.array([int(r["g"]) for r in sm])
    lnum = np.array([int(r["level_num"]) for r in sm])
    gl_of = np.array([int(r["global_level"]) for r in sm])
    sl_of = np.array([int(r["sl_local"]) for r in sm])
    lower_fl = [i for i in range(N_fl) if not upper[i]]
    A0, _, und0, ch0, maps = TR.det_si_exc_build_fl(ops, sm, N, gh)
    gh_fl = maps["gh_fl"]
    if und0 or gh_fl is None:
        return TR.refuse(["FL_OPERATOR_UNDEFINED:" + ",".join(sorted(set(und0)))])

    # C1 inputs: recorded BF statements (sigma row, threshold, nstar) and the sigma file
    sites = sorted({int(r["site"]) for r in ops})
    if 6 in sites:
        gates["G-RATE"].append("C1_CHANNEL_ALREADY_RECORDED")
    bf = {}
    for r in ops:
        if r["site"] == "5":
            fl = maps["fl_of_global"][int(r["lev_b"])]
            if fl in bf:
                gates["G-INPUT"].append(f"BF_DUPLICATE:fl{fl}")
            bf[fl] = (int(r["lev_a"]), float(r["x6"]), float(r["x5"]))
    if sorted(bf) != lower_fl:
        return TR.refuse(gates["G-INPUT"] + ["BF_LEVEL_SET"])
    chi = H_PLANCK * bf[lvmap[0]][1]
    nu_bad = [int(lnum[fl]) for fl in bf
              if not math.isclose(bf[fl][1], (chi - E[fl] * EV_TO_ERG) / H_PLANCK, rel_tol=INPUT_RTOL)]
    if nu_bad or E[lvmap[0]] != 0.0:
        gates["G-INPUT"].append(f"THRESHOLD_VS_LEVEL_ENERGY:levels={nu_bad[:10]}")
    saha = {fl: det_si_exc_saha(T_e, n_e, int(g[fl]), int(g[gh_fl]), bf[fl][1]) for fl in bf}
    s_dev = max(abs(saha[fl] / bf[fl][2] - 1.0) for fl in bf)
    if not (s_dev <= INPUT_RTOL):
        gates["G-INPUT"].append(f"NSTAR_RECORD_VS_SAHA:max_rel={s_dev:.3g}")
    sigma_path = run / "input/model/cmfgen_sigma_bf.bin"
    edges = det_si_exc_sigma_edges(sigma_path, sorted({bf[fl][0] for fl in bf}))
    extra1, und1 = det_si_exc_c1([(fl, edges[bf[fl][0]], bf[fl][1], bf[fl][2]) for fl in sorted(bf)],
                                 T_e, n_e, gh_fl)
    if det_si_exc_detailed_balance(extra1, saha):
        gates["G-RATE"].append("C1_DETAILED_BALANCE")
    A1, _, _, ch1, _ = TR.det_si_exc_build_fl(ops, sm, N, gh, extra=extra1)

    # C2 inputs: the Si II table paired with the recorded lower-ion COLDATA statements
    col_path = run / "input/model/ige_col_14_1_cmfgen.bin"
    col = det_si_exc_coltable(col_path)
    if (col["Z"], col["ion0"], col["n_lev"]) != (14, 1, len(lower_fl)):
        gates["G-INPUT"].append(f"COL_TABLE_IDENTITY:{col['Z']},{col['ion0']},{col['n_lev']}")
    lo_ops = [r for r in ops if r["site"] in ("2", "3", "4") and int(r["dst"]) < n_lo and int(r["src"]) < n_lo]
    if any(r["site"] != "2" for r in lo_ops):
        gates["G-INPUT"].append("LOWER_ION_NON_COLDATA_COLLISION")
    lo_ops = [r for r in lo_ops if r["site"] == "2"]
    fl_of_ln = {int(lnum[i]): i for i in lower_fl}

    def usable(rec):
        lo, hi, om = rec
        if lo not in fl_of_ln or hi not in fl_of_ln or sl_of[fl_of_ln[lo]] == sl_of[fl_of_ln[hi]]:
            return False
        if not (det_si_exc_omega_lin(col["tgrid"], om, T_e) > 0.0):
            return False
        return g[fl_of_ln[lo]] > 0 and g[fl_of_ln[hi]] > 0 and (E[fl_of_ln[hi]] - E[fl_of_ln[lo]]) * EV_TO_ERG > 0.0
    match, why_m = det_si_exc_match_coldata(lo_ops, col["records"], usable)
    if match is None:
        gates["G-INPUT"].append("C2_MATCH:" + ";".join(why_m[:5]))
    c2_new, c2_undef, rec_dev, db_bad, om_ratio = {}, [], 0.0, [], []
    for r in (lo_ops if match is not None else []):
        lo, hi, om = col["records"][match[r["op"]]]
        lo_fl, hi_fl = fl_of_ln[lo], fl_of_ln[hi]
        dE = (E[hi_fl] - E[lo_fl]) * EV_TO_ERG
        om_lin = det_si_exc_omega_lin(col["tgrid"], om, T_e)
        c_lin = det_si_exc_coll_rates(om_lin, T_e, n_e, int(g[lo_fl]), int(g[hi_fl]), dE)
        for a, b in ((om_lin, float(r["x5"])), (c_lin[0], float(r["x1"])), (c_lin[1], float(r["x2"]))):
            rec_dev = max(rec_dev, abs(a / b - 1.0) if b != 0.0 else (0.0 if a == 0.0 else math.inf))
        om_log = det_si_exc_omega_log(col["tgrid"], om, T_e)
        if om_log is None:
            c2_undef.append(f"{lo}-{hi}")
            continue
        cu, cd = det_si_exc_coll_rates(om_log, T_e, n_e, int(g[lo_fl]), int(g[hi_fl]), dE)
        if not math.isclose(cu * g[lo_fl], cd * g[hi_fl] * math.exp(-dE / (K_BOLTZMANN * T_e)), rel_tol=1e-12):
            db_bad.append(f"{lo}-{hi}")
        c2_new[r["op"]] = (cu, cd)
        om_ratio.append(om_log / om_lin)
    if not (rec_dev <= INPUT_RTOL):
        gates["G-INPUT"].append(f"COLDATA_RECORD_NOT_REPRODUCED:max_rel={rec_dev:.3g}")
    if db_bad:
        gates["G-RATE"].append(f"C2_DETAILED_BALANCE:{len(db_bad)}")
    A2, _, _, ch2, _ = TR.det_si_exc_build_fl(ops, sm, N, gh, coldata_override=lambda r, x: c2_new.get(r["op"]))

    # ---- G-RATE: channel sums and conservation of every operator
    ops_by_state = {"F0": (A0, ch0), "F1": (A1, ch1), "F2": (A2, ch2)}
    for name, (A, chs) in ops_by_state.items():
        dev = np.abs(sum(chs.values()) - A)
        if float(np.max(dev - 1e-12 * np.abs(A))) > 0.0:
            gates["G-RATE"].append(f"CHANNEL_SUM:{name}")
        if det_si_exc_conservation(A):
            gates["G-RATE"].append(f"CONSERVATION:{name}")

    # ---- G-SOLVE
    sol, eta_max = {}, {}
    for name, (A, _) in ops_by_state.items():
        x, why, eta_max[name] = det_si_exc_solve(A, total)
        sol[name] = x
        if why:
            gates["G-SOLVE"].append(f"{name}:" + ",".join(why))
    eta_se = TR.backward(A0, fin)
    R = {"SE": TR.ratio(fin, upper)}
    for name in ops_by_state:
        R[name] = TR.ratio(sol[name], upper) if sol[name] is not None else float("nan")
    f0_prev = float(json.loads(PREVIOUS.read_text())["ratios"]["R_F"]) if PREVIOUS.is_file() else float("nan")
    f0_dex = TR.dex(R["F0"], f0_prev)
    if not (abs(f0_dex) <= F0_TOL_DEX):
        gates["G-SOLVE"].append(f"F0_REFERENCE:{R['F0']}vs{f0_prev}")

    # ---- G-TABLE: public correspondence, LTE fractions against the ladder
    t_shell = TR.dump_row(run, stop_iter, "shell", SHELL)[0]
    vel = (float(t_shell["v_inner_cm_s"]) + float(t_shell["v_outer_cm_s"])) / 2e5
    pub, pub_vel, pub_ii, pub_iii = det_si_exc_public(vel)
    if not (math.isclose(vel, TR.REGISTERED_VEL["lumina_vel_mid_kms"]) and
            math.isclose(pub_vel, TR.REGISTERED_VEL["public_vel_mid_kms"]) and pub == pub_iii / pub_ii):
        gates["G-TABLE"].append(f"PUBLIC_CORRESPONDENCE:{vel}->{pub_vel}")
    w = np.where(~upper, g * np.exp(-E * EV_TO_ERG / (K_BOLTZMANN * T_e)), 0.0)
    z_sum = float(w.sum())
    f_lte = w / z_sum
    lad = [r for r in TR.read_trace(tr / "si10_ladder.csv")
           if int(r["bundle"]) == ADOPTED["bundle"] and int(r["ne_iter"]) == ADOPTED["ne_iter"]]
    lrow = ([r for r in lad if r["ch_armed"] == "1"] or lad or [None])[-1]
    lad_lv = {int(r["global_level"]): r for r in TR.read_trace(tr / "si10_ladder_levels.csv")
              if lrow is not None and r["lseq"] == lrow["lseq"]}
    z_part = float(next(iter(lad_lv.values()))["Z_part"]) if lad_lv else float("nan")
    lte_bad = [int(lnum[i]) for i in lower_fl if int(gl_of[i]) not in lad_lv or not math.isclose(
        f_lte[i], float(lad_lv[int(gl_of[i])]["f_lev"]), rel_tol=INPUT_RTOL)]
    if lte_bad or len(lad_lv) != len(lower_fl) or lrow is None or float(lrow["T_e"]) != T_e:
        gates["G-TABLE"].append(f"LTE_FRACTION_VS_LADDER:levels={lte_bad[:10]}:Z_part={z_part!r}:sum={z_sum!r}")

    # ---- the table (one memory result; CSV canonical, Markdown display)
    rows = []

    def row(kind, **v):
        rows.append(dict({"kind": kind}, **v))
    row("public", state="CMFGEN toy06 19.480 d", time_d=19.48, vel_kms=pub_vel, si_II=pub_ii, si_III=pub_iii,
        III_II=pub, note="public file holds ion fractions only; no Si II level populations")
    d_f0 = TR.dex(R["F0"], pub)
    cls = {}
    for state in ("SE", "F0", "F1", "F2"):
        d = TR.dex(R[state], pub)
        G = abs(d_f0) - abs(d) if state in ("F1", "F2") and math.isfinite(d) and math.isfinite(d_f0) else None
        c = None
        if state in ("F1", "F2"):
            c = "undefined" if G is None else "toward" if G > APPROACH_DEX else "away" if G < -APPROACH_DEX \
                else "unresolved"
            cls[state] = c
        row("ion sum", state=state, III_II=R[state], d_pub=d, delta_SE=TR.dex(R[state], R["SE"]),
            delta_F0=TR.dex(R[state], R["F0"]), G=G, cls=c,
            note="adopted FINAL_FL, not a FL steady state" if state == "SE" else "GTH steady state")
    states = {"SE": (fin, ch0), "F0": (sol["F0"], ch0), "F1": (sol["F1"], ch1), "F2": (sol["F2"], ch2)}
    for k in SELECTED:
        i = lvmap[k]
        ident = {"level": k, "global_level": int(gl_of[i]), "i_fl": i, "sl": int(sl_of[i]), "E_eV": float(E[i]),
                 "g": int(g[i]), "f_lte": float(f_lte[i])}
        for state, (n, chs) in states.items():
            if n is None:
                row("level total", state=state, **ident, note="state undefined")
                continue
            n_ii = float(n[~upper].sum())
            f_i = float(n[i]) / n_ii
            bud = det_si_exc_budget(chs, n, i, n_ii, float(f_lte[i]))
            tot = {q: sum(b[q] for b in bud.values()) for q in ("I", "O", "B", "D", "H")}
            names = [c for c in CHANNEL_ORDER if c in bud] + sorted(c for c in bud if c not in CHANNEL_ORDER)
            for c in names:
                b = bud[c]
                if b["I"] == 0.0 and b["D"] == 0.0:
                    continue
                row("level channel", state=state, **ident, channel=c, f=f_i, b_rel=f_i / float(f_lte[i]),
                    I=b["I"], O=b["O"], B=b["B"], D=b["D"], H=b["H"],
                    in_share=b["I"] / tot["I"] if tot["I"] > 0 else None,
                    out_share=b["O"] / tot["O"] if tot["O"] > 0 else None)
            top_in = max(bud, key=lambda c: bud[c]["I"])
            top_out = max(bud, key=lambda c: bud[c]["O"])
            row("level total", state=state, **ident, f=f_i, b_rel=f_i / float(f_lte[i]), I=tot["I"], O=tot["O"],
                B=tot["B"], D=tot["D"], H=tot["H"],
                eta_row=float(eta_se[i]) if state == "SE" else float(TR.backward(ops_by_state[state][0], n)[i]),
                steady="no (adopted SE; residual shown)" if state == "SE" else "yes (GTH)",
                note=f"largest inflow {top_in} ({TR.div(bud[top_in]['I'], tot['I']):.3g}); "
                     f"largest outflow {top_out} ({TR.div(bud[top_out]['O'], tot['O']):.3g}); "
                     f"total H {'<0: LTE share not sustainable at fixed supply' if tot['H'] < 0 else '>=0'}")
    row("intervention", state="F1", channel="C1", CMFGEN=CMFGEN["C1"],
        rate="sealed pair branch: C_ion = n_e 1.55e13 T^-1/2 0.2 sigma_edge e^-u/u, C_rec = C_ion n*",
        targets=len(extra1) // 2, undefined=";".join(und1) or "none")
    row("intervention", state="F2", channel="C2", CMFGEN=CMFGEN["C2"],
        rate="sealed col_data statement (8.629e-6, detailed balance) with Omega log-log",
        targets=len(c2_new), undefined=";".join(c2_undef) or "none",
        note=(f"Omega_log/Omega_lin min {min(om_ratio):.6g} median {float(np.median(om_ratio)):.6g} "
              f"max {max(om_ratio):.6g}") if om_ratio else "")
    all_gates = [f for v in gates.values() for f in v]
    valid = not all_gates and all(sol[s] is not None for s in ("F0", "F1", "F2")) and not und1 and not c2_undef \
        and match is not None
    code = 2 if not valid else 0 if "toward" in cls.values() else 1
    label = {0: "TOWARD", 1: "NO_RESOLVED_APPROACH", 2: "INDETERMINATE"}[code]
    row("quality", state="", keys=json.dumps({k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")}),
        T_e=T_e, n_e=n_e, total=total, eta_F0=eta_max["F0"], eta_F1=eta_max["F1"], eta_F2=eta_max["F2"],
        F0_vs_previous_dex=f0_dex, gates=";".join(all_gates) or "all pass", branch=f"{code} {label}",
        note="fixed J, T_e, n_e, CE partners and Si total; evidence.json holds input SHA-256")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "si10_excitation_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    with open(out / "si10_excitation_table.md", "x") as f:
        f.write(f"# Si II excitation budget, material 13, Si shell 10 (branch {code} {label})\n")
        for kind in dict.fromkeys(r["kind"] for r in rows):
            sub = [r for r in rows if r["kind"] == kind]
            cols = [c for c in fields if c != "kind" and any(r.get(c) not in (None, "") for r in sub)]
            f.write(f"\n## {kind}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in sub:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
    used = [tr / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_commit.csv", "si10_cevec.csv", "si10_ops.csv",
                             "si10_slmap.csv", "si10_vec.csv", "si10_final.csv", "si10_ladder.csv",
                             "si10_ladder_levels.csv", f"si10_mat_{int(cid):06d}_gen.bin")]
    used += [run / "RUN_FOOTER.txt", run / f"work/physics_comparison/physics_DET_iter{stop_iter:04d}.shell.csv",
             sigma_path, col_path, TR.PUBLIC_SI, TR.DIAG8_PUBLIC, PREVIOUS]
    evidence = {"schema": "DET_SI_EXC_ANALYSIS_V1", "prereg": PREREG, "argv": sys.argv,
                "adopted": {k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")},
                "T_e": T_e, "n_e": n_e, "total": total, "sites": sites, "ratios": R, "public": pub,
                "public_vel_kms": pub_vel, "public_fractions": {"si_II": pub_ii, "si_III": pub_iii},
                "F0_previous": f0_prev, "F0_vs_previous_dex": f0_dex, "eta_max": eta_max,
                "eta_SE_max": float(np.nanmax(eta_se)), "classification": cls, "branch": code, "label": label,
                "gates": gates, "c1_targets": len(extra1) // 2, "c1_undefined": und1,
                "nstar_vs_saha_max_rel": s_dev, "c2_targets": len(c2_new), "c2_undefined": c2_undef,
                "coldata_record_max_rel": rec_dev, "lte": {"Z_part_ladder": z_part, "boltzmann_sum": z_sum},
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(__file__).resolve()): TR.sha(Path(__file__)),
                         str(Path(TR.__file__).resolve()): TR.sha(Path(TR.__file__))}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str) + "\n")
    print(f"BRANCH {code} {label} R={R} public={pub} classes={cls}")
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
