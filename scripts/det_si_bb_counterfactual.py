#!/usr/bin/env python3
"""DET-SI-BB-COUNTERFACTUAL (prereg docs/PREREG_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md).

Two diagnostic prescriptions on the recorded BB radiative statements of the adopted consumption state of the
DET-SI-CONSUMPTION-TRACE run T (material 13, Si shell 10, call 36), each applied alone to the recorded FL operator:
  D7  statements outside the BB domain (jd_mode 7: Jbar 0, R_spont = A) lose all three radiative rates  -> F7
  D8  net-radiative-bracket statements (jd_mode 8) get escape probability 1 (the three rates / beta)  -> F8
Collisions, BF, DR, CE, NT and the fixed field stay as recorded.  Neither prescription reproduces CMFGEN.

Subcommands
  negative                    negative controls of the new functions (prereg 9)
  analyze --run T --out DIR   G-INPUT, G-RATE, G-SOLVE, G-TABLE, G-RUN and the main table
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

PREREG = "docs/PREREG_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md"
PREVIOUS = Path("/gpfs/kjhan/lumina/cmfgen_map/siexc_20261011/results/run1/evidence.json")
SELECTED, ADOPTED, SHELL = EB.SELECTED, EB.ADOPTED, EB.SHELL
REGISTERED_MODES = {(7, "Si II"): 671, (7, "Si III"): 221, (8, "Si II"): 1169, (8, "Si III"): 1418}
F0_TOL_DEX, APPROACH_DEX, RTOL = 0.001, 0.01, 1e-12
RAD = ("BB radiative upward", "BB radiative downward")
SCOPE = {"D7": "diagnostic: zero radiative coupling of out-of-domain statements (complete-trapping limit of the "
               "net rate); no CMFGEN counterpart, not a III/II bound",
         "D8": "diagnostic: optically thin limit with the recorded incident continuum (EXPONX thin limit, exponx.f); "
               "not SOBJBAR_SIM / MOMJBAR, not a population or III/II bound"}


# ------------------------------------------------------------------ statements and prescriptions
def det_si_bb_exponx(tau):
    """sealed line_net_cmfgen_exponx (CMFGEN EXPONX transcription); None where it is undefined."""
    if not math.isfinite(tau):
        return None
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        value = 1.0 - tau * companion
    elif tau < 40.0:
        value = (1.0 - math.exp(-tau)) / tau
        companion = (1.0 - value) / tau
    else:
        value = 1.0 / tau
        companion = (1.0 - value) / tau
    if not (math.isfinite(value) and value > 0.0 and math.isfinite(companion) and companion > 0.0):
        return None
    return value


def det_si_bb_mode(r):
    return int(float(r["x9"]))


def det_si_bb_ion(r, n_lo):
    """Si II when both SLs belong to the lower ion, Si III when both belong to the upper ion, None when mixed."""
    d, s = int(r["dst"]), int(r["src"])
    if d < n_lo and s < n_lo:
        return "Si II"
    if d >= n_lo and s >= n_lo:
        return "Si III"
    return None


def det_si_bb_check(r, x):
    """registered statement conditions: mode 7 has no pump and beta 1; mode 8 carries the sealed bracket
    (beta = EXPONX(tau_eff), Jbar = c + l, applicable provenance).  x = x1..x14 of the BB record."""
    m = det_si_bb_mode(r)
    if m == 7:
        if x[0] != 0.0 or x[1] != 0.0 or x[9] != 1.0:
            return "MODE7_PUMP_OR_BETA"
        return None if (math.isfinite(x[2]) and x[2] >= 0.0) else "MODE7_SPONT"
    if m == 8:
        beta = x[9]
        if not (math.isfinite(beta) and beta > 0.0):
            return "MODE8_BETA"
        b = det_si_bb_exponx(x[12])
        if b is None or not (abs(b - beta) <= RTOL * beta):
            return "MODE8_BETA_VS_TAU"
        jbar, c, l, prov = x[7], x[10], x[11], x[13]
        if not (math.isfinite(c) and c >= 0.0 and math.isfinite(l) and math.isfinite(jbar)
                and abs(jbar - (c + l)) <= RTOL * jbar):
            return "MODE8_LEDGER"
        if not (math.isfinite(prov) and prov == int(prov) and 0 <= prov <= 3 and not int(prov) & 1):
            return "MODE8_PROVENANCE"
        return None if all(math.isfinite(v) and v >= 0.0 for v in x[0:3]) else "MODE8_RATES"
    return f"UNREGISTERED_MODE:{m}"


def det_si_bb_d7(r, x):
    return (0.0, 0.0, 0.0) if det_si_bb_mode(r) == 7 else None


def det_si_bb_d8(r, x):
    if det_si_bb_mode(r) != 8:
        return None
    b = x[9]
    return (x[0] / b, x[1] / b, x[2] / b)


def det_si_bb_prescription_check(name, fn, stmts):
    """the registered relation between recorded and replaced triples, judged from the outputs only."""
    bad = []
    for r, x in stmts:
        m, out = det_si_bb_mode(r), fn(r, x)
        if name == "D7":
            ok = out == (0.0, 0.0, 0.0) if m == 7 else out is None
        elif m == 8:
            ok = out is not None and all(
                (o == 0.0) if v == 0.0 else math.isclose(o * x[9], v, rel_tol=RTOL, abs_tol=0.0)
                for o, v in zip(out, x[0:3]))
        else:
            ok = out is None
        if not ok:
            bad.append(r["op"])
    return bad


# ------------------------------------------------------------------ operators
def det_si_bb_build(ops, sm, N, gh, override=None):
    """total operator and channels with the BB radiative channels split by recorded mode (subset builds)."""
    A, _, und, ch, maps = TR.det_si_exc_build_fl(ops, sm, N, gh, bb_radiative_override=override)
    out = {k: v for k, v in ch.items() if k not in RAD}
    for m in (7, 8):
        sub = [r for r in ops if r["site"] == "1" and det_si_bb_mode(r) == m]
        res = TR.det_si_exc_build_fl(sub, sm, N, gh, bb_radiative_override=override)
        und = und + res[2]
        for k in RAD:
            if k in res[3]:
                out[f"{k} (mode {m})"] = res[3][k]
    return A, und, out, maps


def det_si_bb_channel_sum_ok(A, chs):
    dev = np.abs(sum(chs.values()) - A)
    return float(np.max(dev - RTOL * np.abs(A))) <= 0.0


def det_si_bb_invariance(ch_ref, ch_new, changed):
    """channels outside the changed set are bit-identical; returns the offending names."""
    return sorted(k for k in set(ch_ref) | set(ch_new) if k not in changed and
                  (k not in ch_ref or k not in ch_new or not np.array_equal(ch_ref[k], ch_new[k])))


def det_si_bb_solve(A, total):
    """the registered solver plus: a backward error that is not finite in every row is not a valid state."""
    x, why, eta_max = EB.det_si_exc_solve(A, total)
    if x is not None and not np.isfinite(TR.backward(A, x)).all():
        return None, why + ["BACKWARD_NONFINITE_ROW"], eta_max
    return x, why, eta_max


# ------------------------------------------------------------------ negative controls (prereg 9)
def det_si_bb_toy():
    """three lower-ion levels in separate SLs and the upper ground; one mode-8 and one mode-7 BB statement
    (with line collisions) and two BF statements."""
    N, gh, n_lo = 4, 3, 3
    sm = [{"i_fl": str(i), "sl_local": str(i), "within_sl_frac": "1", "fl_index": str(i), "level_num": str(ln),
           "upper_ion": str(u)} for i, (ln, u) in enumerate(((0, 0), (1, 0), (2, 0), (0, 1)))]
    tau = 0.5
    beta = det_si_bb_exponx(tau)
    c, l = beta * 1e-5, 2e-6
    ra, rs, rp, cu, cd = 3e8 * c, 1.5e8 * c, 2e8 * beta, 40.0, 30.0
    a7, cu7, cd7 = 5e7, 20.0, 15.0

    def op(k, site, dst, src, v1, v2, la, lb, x):
        r = {"op": str(k), "site": str(site), "dst": str(dst), "src": str(src), "v1": repr(float(v1)),
             "v2": repr(float(v2)), "lev_a": str(la), "lev_b": str(lb)}
        r.update({f"x{j + 1}": (repr(float(x[j])) if j < len(x) else "") for j in range(14)})
        return r
    ops = [op(1, 1, 1, 0, ra + cu, (rs + rp) + cd, 1, 0, (ra, rs, rp, cu, cd, 1.0, 1.0, c + l, 8, beta, c, l, tau, 0)),
           op(2, 1, 2, 1, 0.0 + cu7, (0.0 + a7) + cd7, 2, 1,
              (0.0, 0.0, a7, cu7, cd7, 1.0, 1.0, 0.0, 7, 1.0, 0.0, 0.0, 0.0, 0)),
           op(3, 5, 3, 0, 3.0, 0.5, 0, 0, (3.0, 1.0, 0.5, 0.0, 1e-3, 1e15)),
           op(4, 5, 3, 2, 7.0, 0.2, 2, 2, (7.0, 1.0, 0.2, 0.0, 2e-3, 9e14))]
    return {"N": N, "gh": gh, "n_lo": n_lo, "sm": sm, "ops": ops}


def cmd_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    t = det_si_bb_toy()
    N, gh, sm, ops = t["N"], t["gh"], t["sm"], t["ops"]
    bb = [(r, [TR.num(r[f"x{k}"]) for k in range(1, 15)]) for r in ops if r["site"] == "1"]
    A0, u0, ch0, _ = det_si_bb_build(ops, sm, N, gh)
    A7, u7, ch7, _ = det_si_bb_build(ops, sm, N, gh, det_si_bb_d7)
    A8, u8, ch8, _ = det_si_bb_build(ops, sm, N, gh, det_si_bb_d8)
    expect(not (u0 or u7 or u8) and all(EB.det_si_exc_solve(A, 10.0)[0] is not None for A in (A0, A7, A8)),
           "toy F0/F7/F8: defined operators with unique valid steady states (positive control)")
    expect(all(det_si_bb_check(r, x) is None for r, x in bb), "toy statements meet the registered conditions "
           "(positive control)")
    expect(all(det_si_bb_channel_sum_ok(A, c) for A, c in ((A0, ch0), (A7, ch7), (A8, ch8))),
           "mode-split channels sum to each operator (positive control)")
    expect(not det_si_bb_prescription_check("D7", det_si_bb_d7, bb) and
           not det_si_bb_prescription_check("D8", det_si_bb_d8, bb), "registered prescriptions (positive control)")
    expect(not det_si_bb_invariance(ch0, ch7, {f"{k} (mode 7)" for k in RAD}) and
           not det_si_bb_invariance(ch0, ch8, {f"{k} (mode 8)" for k in RAD}),
           "non-target channels bit-identical under D7 and D8 (positive control)")

    # 1. a no-op callback must leave operator and channels unchanged
    Ai, _, chi, _ = det_si_bb_build(ops, sm, N, gh, lambda r, x: (x[0], x[1], x[2]))
    expect(np.array_equal(Ai, A0) and not det_si_bb_invariance(ch0, chi, set()),
           "identity callback: operator and channels unchanged")
    Ap, _, chp, _ = det_si_bb_build(ops, sm, N, gh, lambda r, x: (x[0] * (1 + 1e-9), x[1], x[2]))
    expect(not np.array_equal(Ap, A0) and bool(det_si_bb_invariance(ch0, chp, set())),
           "inject a perturbing 'no-op' callback: detected")

    # 2. mode misassignment, duplicate application, target omission, ion misassignment
    expect(bool(det_si_bb_prescription_check("D7", lambda r, x: (0.0, 0.0, 0.0), bb)),
           "inject D7 applied to mode-8 statements: refused")
    expect(bool(det_si_bb_prescription_check("D8", lambda r, x: tuple(v / x[9] ** 2 for v in x[0:3])
                                             if det_si_bb_mode(r) == 8 else None, bb)),
           "inject D8 applied twice (beta squared): refused")
    expect(bool(det_si_bb_prescription_check("D7", lambda r, x: None, bb)), "inject D7 target omission: refused")
    mixed = dict(ops[0], dst="3")
    expect(det_si_bb_ion(mixed, t["n_lo"]) is None and det_si_bb_ion(ops[0], t["n_lo"]) == "Si II",
           "inject a statement across the two ions: refused")
    census = {}
    for r, _ in bb:
        census[(det_si_bb_mode(r), det_si_bb_ion(r, t["n_lo"]))] = census.get(
            (det_si_bb_mode(r), det_si_bb_ion(r, t["n_lo"])), 0) + 1
    expect(census == {(8, "Si II"): 1, (7, "Si II"): 1} and census != REGISTERED_MODES,
           "census differs from the registered counts: refused rather than adjusted")

    # 3. statement conditions
    r8, x8 = bb[0]
    r7, x7 = bb[1]
    expect(det_si_bb_check(r7, [x7[0] + 1.0] + x7[1:]) == "MODE7_PUMP_OR_BETA", "inject mode 7 with absorption: "
           "refused")
    expect(det_si_bb_check(r8, x8[:9] + [0.0] + x8[10:]) == "MODE8_BETA", "inject mode 8 with beta 0: refused")
    expect(det_si_bb_check(r8, x8[:9] + [float("nan")] + x8[10:]) == "MODE8_BETA", "inject mode 8 with non-finite "
           "beta: refused")
    expect(det_si_bb_check(r8, x8[:11] + [x8[11] * 1.01] + x8[12:]) == "MODE8_LEDGER",
           "inject mode 8 with a broken Jbar = c + l ledger: refused")
    expect(det_si_bb_check(r8, x8[:12] + [x8[12] * 1.01] + x8[13:]) == "MODE8_BETA_VS_TAU",
           "inject mode 8 whose beta does not follow tau_eff: refused")

    # 4. D8 dividing only the spontaneous rate
    expect(bool(det_si_bb_prescription_check("D8", lambda r, x: (x[0], x[1], x[2] / x[9])
                                             if det_si_bb_mode(r) == 8 else None, bb)),
           "inject D8 dividing only the spontaneous rate: refused")

    # 5. collision change, channel omission, debit sign error
    chc = dict(ch7)
    chc["BB collisional (line)"] = ch7["BB collisional (line)"] * (1 + 1e-9)
    expect(bool(det_si_bb_invariance(ch0, chc, {f"{k} (mode 7)" for k in RAD})),
           "inject a collision change under D7: invariance refuses")
    cho = {k: v for k, v in ch8.items() if k != "BF recombination"}
    expect(not det_si_bb_channel_sum_ok(A8, cho), "inject a missing channel: channel sum refuses")
    S = A7.copy()
    S[0, 0] += 1.0
    expect(bool(EB.det_si_exc_conservation(S)) and det_si_bb_solve(S, 10.0)[0] is None,
           "inject a debit sign error: conservation refuses")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_SI_BB_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
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

    # ---- G-INPUT: adoption, replay, level map, statement census and conditions
    call, why = EB.det_si_exc_adopt(TR.read_trace(tr / "si10_ne.csv"), TR.read_trace(tr / "si10_calls.csv"),
                                    TR.read_trace(tr / "si10_cevec.csv"), TR.read_trace(tr / "si10_commit.csv"),
                                    ADOPTED)
    if call is None:
        return TR.refuse(why)
    cid = call["call"]
    N, N_fl, n_lo, gh = int(call["N"]), int(call["N_fl"]), int(call["n_lo_super"]), int(call["ground_hi"])
    T_e = float(call["T_e"])
    ops = [r for r in TR.read_trace(tr / "si10_ops.csv") if r["call"] == cid]
    g_a, g_b, _ = TR.read_matrix(tr / f"si10_mat_{int(cid):06d}_gen.bin")
    r_a, r_b = TR.replay(ops, N)
    if not (np.array_equal(r_a, g_a) and np.array_equal(r_b, g_b)):
        gates["G-INPUT"].append("REPLAY_NOT_GENERATOR")
    sm = sorted((r for r in TR.read_trace(tr / "si10_slmap.csv") if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    lvmap = EB.det_si_exc_level_map(sm, SELECTED)
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
    gl_of = np.array([int(r["global_level"]) for r in sm])
    sl_of = np.array([int(r["sl_local"]) for r in sm])
    bb = [(r, [TR.num(r[f"x{k}"]) for k in range(1, 15)]) for r in ops if r["site"] == "1"]
    census, tau_sign, bad_stmt = {}, {}, []
    for r, x in bb:
        ion, m = det_si_bb_ion(r, n_lo), det_si_bb_mode(r)
        if ion is None:
            bad_stmt.append(f"MIXED_ION:op{r['op']}")
            continue
        census[(m, ion)] = census.get((m, ion), 0) + 1
        reason = det_si_bb_check(r, x)
        if reason:
            bad_stmt.append(f"{reason}:op{r['op']}")
        if m == 8:
            s = "negative" if x[12] < 0.0 else "zero" if x[12] == 0.0 else "positive"
            tau_sign[(ion, s)] = tau_sign.get((ion, s), 0) + 1
    if census != REGISTERED_MODES:
        gates["G-INPUT"].append("MODE_CENSUS:" + json.dumps({f"{m}:{i}": v for (m, i), v in sorted(census.items())}))
    if bad_stmt:
        gates["G-INPUT"].append(f"STATEMENT_CONDITIONS:{len(bad_stmt)}:" + ";".join(bad_stmt[:5]))

    # ---- operators and G-RATE
    for name, fn in (("D7", det_si_bb_d7), ("D8", det_si_bb_d8)):
        bad = det_si_bb_prescription_check(name, fn, bb)
        if bad:
            gates["G-RATE"].append(f"PRESCRIPTION:{name}:{len(bad)}")
    A0, und0, ch0, maps = det_si_bb_build(ops, sm, N, gh)
    A7, und7, ch7, _ = det_si_bb_build(ops, sm, N, gh, det_si_bb_d7)
    A8, und8, ch8, _ = det_si_bb_build(ops, sm, N, gh, det_si_bb_d8)
    if und0 or und7 or und8 or maps["gh_fl"] is None:
        return TR.refuse(gates["G-INPUT"] + ["FL_OPERATOR_UNDEFINED:" + ",".join(sorted(set(und0 + und7 + und8)))])
    ops_by_state = {"F0": (A0, ch0), "F7": (A7, ch7), "F8": (A8, ch8)}
    for name, (A, chs) in ops_by_state.items():
        if not det_si_bb_channel_sum_ok(A, chs):
            gates["G-RATE"].append(f"CHANNEL_SUM:{name}")
        if EB.det_si_exc_conservation(A):
            gates["G-RATE"].append(f"CONSERVATION:{name}")
    for name, chs, m in (("F7", ch7, 7), ("F8", ch8, 8)):
        inv = det_si_bb_invariance(ch0, chs, {f"{k} (mode {m})" for k in RAD})
        if inv:
            gates["G-RATE"].append(f"INVARIANCE:{name}:" + ",".join(inv))

    # ---- G-SOLVE
    sol, eta_max = {}, {}
    for name, (A, _) in ops_by_state.items():
        x, why, eta_max[name] = det_si_bb_solve(A, total)
        sol[name] = x
        if why:
            gates["G-SOLVE"].append(f"{name}:" + ",".join(why))
    R = {"SE": TR.ratio(fin, upper)}
    for name in ops_by_state:
        R[name] = TR.ratio(sol[name], upper) if sol[name] is not None else float("nan")
    f0_prev = float(json.loads(PREVIOUS.read_text())["ratios"]["F0"]) if PREVIOUS.is_file() else float("nan")
    f0_dex = TR.dex(R["F0"], f0_prev)
    if not (abs(f0_dex) <= F0_TOL_DEX):
        gates["G-SOLVE"].append(f"F0_REFERENCE:{R['F0']}vs{f0_prev}")
    eta_se = TR.backward(A0, fin)

    # ---- G-TABLE: public correspondence
    t_shell = TR.dump_row(run, stop_iter, "shell", SHELL)[0]
    vel = (float(t_shell["v_inner_cm_s"]) + float(t_shell["v_outer_cm_s"])) / 2e5
    pub, pub_vel, pub_ii, pub_iii = EB.det_si_exc_public(vel)
    if not (math.isclose(vel, TR.REGISTERED_VEL["lumina_vel_mid_kms"]) and
            math.isclose(pub_vel, TR.REGISTERED_VEL["public_vel_mid_kms"]) and pub == pub_iii / pub_ii):
        gates["G-TABLE"].append(f"PUBLIC_CORRESPONDENCE:{vel}->{pub_vel}")
    w = np.where(~upper, g * np.exp(-E * EB.EV_TO_ERG / (EB.K_BOLTZMANN * T_e)), 0.0)
    f_lte = w / float(w.sum())

    # ---- the table (one memory result; CSV canonical, Markdown display)
    rows = []

    def row(kind, **v):
        rows.append(dict({"kind": kind}, **v))
    row("public", state="CMFGEN toy06 19.480 d", time_d=19.48, vel_kms=pub_vel, si_II=pub_ii, si_III=pub_iii,
        III_II=pub, note="public file holds ion fractions only; level rows have no public counterpart")
    d_f0 = TR.dex(R["F0"], pub)
    cls = {}
    for state in ("SE", "F0", "F7", "F8"):
        d = TR.dex(R[state], pub)
        G = abs(d_f0) - abs(d) if state in ("F7", "F8") and math.isfinite(d) and math.isfinite(d_f0) else None
        c = None
        if state in ("F7", "F8"):
            c = "undefined" if G is None else "toward" if G > APPROACH_DEX else "away" if G < -APPROACH_DEX \
                else "unresolved"
            cls[state] = c
        row("ion sum", state=state, III_II=R[state], d_pub=d, delta_F0=TR.dex(R[state], R["F0"]), G=G, cls=c,
            valid=(state == "SE") or sol[state] is not None,
            note="adopted FINAL_FL, not a FL steady state" if state == "SE" else "GTH steady state")
    for name, m in (("D7", 7), ("D8", 8)):
        v = {"targets_Si_II": census.get((m, "Si II"), 0), "targets_Si_III": census.get((m, "Si III"), 0)}
        if m == 8:
            v["tau_sign"] = json.dumps({f"{i}:{s}": n for (i, s), n in sorted(tau_sign.items())})
        row("prescription", state="F7" if m == 7 else "F8", channel=f"jd_mode {m}", note=SCOPE[name], **v)
    states = {"SE": (fin, ch0), "F0": (sol["F0"], ch0), "F7": (sol["F7"], ch7), "F8": (sol["F8"], ch8)}
    ref = {}
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
            bud = EB.det_si_exc_budget(chs, n, i, n_ii, float(f_lte[i]))
            tot = {q: sum(b[q] for b in bud.values()) for q in ("I", "O", "B", "D")}
            shares = {c: (TR.div(b["I"], tot["I"]), TR.div(b["O"], tot["O"])) for c, b in bud.items()}
            if state == "F0":
                ref[k] = (bud, shares, f_i)
            for c in sorted(bud):
                b = bud[c]
                rb = ref[k][0].get(c) if state in ("F7", "F8") and k in ref else None
                if b["I"] == 0.0 and b["D"] == 0.0 and (rb is None or (rb["I"] == 0.0 and rb["D"] == 0.0)):
                    continue
                v = {"f": f_i, "b_rel": f_i / float(f_lte[i]), "I": b["I"], "O": b["O"], "B": b["B"], "D": b["D"],
                     "in_share": shares[c][0], "out_share": shares[c][1]}
                if rb is not None:
                    rs_ = ref[k][1][c]
                    v.update(dI=b["I"] - rb["I"], dO=b["O"] - rb["O"], dD=b["D"] - rb["D"],
                             d_in_share=shares[c][0] - rs_[0], d_out_share=shares[c][1] - rs_[1])
                row("level channel", state=state, **ident, channel=c, **v)
            top_in = max(bud, key=lambda c: bud[c]["I"])
            top_out = max(bud, key=lambda c: bud[c]["O"])
            row("level total", state=state, **ident, f=f_i, b_rel=f_i / float(f_lte[i]),
                b_rel_vs_F0=(f_i / ref[k][2]) if state in ("F7", "F8") and k in ref else None,
                I=tot["I"], O=tot["O"], B=tot["B"], D=tot["D"],
                eta_row=float(eta_se[i]) if state == "SE" else float(TR.backward(ops_by_state[state][0], n)[i]),
                steady="no (adopted SE; residual shown)" if state == "SE" else "yes (GTH)",
                note=f"largest inflow {top_in} ({TR.div(bud[top_in]['I'], tot['I']):.3g}); "
                     f"largest outflow {top_out} ({TR.div(bud[top_out]['O'], tot['O']):.3g})")
    all_gates = [f for v in gates.values() for f in v]
    valid = not all_gates and all(sol[s] is not None for s in ("F0", "F7", "F8"))
    code = 2 if not valid else 0 if "toward" in cls.values() else 1
    label = {0: "TOWARD", 1: "NO_RESOLVED_APPROACH", 2: "INDETERMINATE"}[code]
    row("quality", state="", keys=json.dumps({k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")}),
        T_e=T_e, n_e=float(call["n_e"]), total=total, eta_F0=eta_max["F0"], eta_F7=eta_max["F7"],
        eta_F8=eta_max["F8"], F0_vs_previous_dex=f0_dex, gates=";".join(all_gates) or "all pass",
        branch=f"{code} {label}", note="fixed continuum J, T_e, n_e, CE partners and Si total; "
                                        "evidence.json holds input SHA-256")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "si10_bb_counterfactual_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    with open(out / "si10_bb_counterfactual_table.md", "x") as f:
        f.write(f"# Si BB radiative counterfactual, material 13, Si shell 10 (branch {code} {label})\n")
        for kind in dict.fromkeys(r["kind"] for r in rows):
            sub = [r for r in rows if r["kind"] == kind]
            cols = [c for c in fields if c != "kind" and any(r.get(c) not in (None, "") for r in sub)]
            f.write(f"\n## {kind}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in sub:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
    used = [tr / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_commit.csv", "si10_cevec.csv", "si10_ops.csv",
                             "si10_slmap.csv", "si10_vec.csv", "si10_final.csv", f"si10_mat_{int(cid):06d}_gen.bin")]
    used += [run / "RUN_FOOTER.txt", run / f"work/physics_comparison/physics_DET_iter{stop_iter:04d}.shell.csv",
             TR.PUBLIC_SI, TR.DIAG8_PUBLIC, PREVIOUS]
    evidence = {"schema": "DET_SI_BB_ANALYSIS_V1", "prereg": PREREG, "argv": sys.argv,
                "adopted": {k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")},
                "T_e": T_e, "n_e": float(call["n_e"]), "total": total, "ratios": R, "public": pub,
                "public_vel_kms": pub_vel, "public_fractions": {"si_II": pub_ii, "si_III": pub_iii},
                "F0_previous": f0_prev, "F0_vs_previous_dex": f0_dex, "eta_max": eta_max,
                "eta_SE_max": float(np.nanmax(eta_se)), "classification": cls, "branch": code, "label": label,
                "gates": gates, "census": {f"{m}:{i}": v for (m, i), v in sorted(census.items())},
                "tau_sign_mode8": {f"{i}:{s}": n for (i, s), n in sorted(tau_sign.items())},
                "statement_findings": bad_stmt,
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(p).resolve()): TR.sha(Path(p)) for p in (__file__, TR.__file__, EB.__file__)}}
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
