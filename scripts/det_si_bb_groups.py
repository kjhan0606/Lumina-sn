#!/usr/bin/env python3
"""DET-SI-BB-GROUPS (prereg docs/PREREG_DET_SI_BB_GROUPS_2026-10-11.md).

The D8 prescription of DET-SI-BB-COUNTERFACTUAL (mode-8 radiative triple / recorded beta) applied to one registered
line group at a time on the recorded F0 operator of the adopted consumption state (material 13, Si shell 10, call 36):
  G1  Si II mode-8 statements whose lower level has level_num 0 or 1
  G2  Si II mode-8 statements whose lower level has level_num 6, 7, 9 or 10
  G3  the other Si II mode-8 statements
  G4  all Si III mode-8 statements
F0 and F8 (all groups) are reproduced against the previous stage; the table carries the non-additivity residual,
every mode-8 statement's recorded inputs and three representative statements per group.

Subcommands
  negative                                   negative controls of the new code (prereg 9)
  analyze --run T --baseline V --out DIR     G-INPUT, G-RATE, G-SOLVE, G-TABLE, G-RUN and the main table
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

PREREG = "docs/PREREG_DET_SI_BB_GROUPS_2026-10-11.md"
GROUPS = ("G1", "G2", "G3", "G4")
G1_LOWER, G2_LOWER = frozenset({0, 1}), frozenset({6, 7, 9, 10})
RULES = {"G1": "Si II, lower level_num in {0,1}", "G2": "Si II, lower level_num in {6,7,9,10}",
         "G3": "Si II, lower level_num outside G1 and G2", "G4": "Si III, all mode-8 statements"}
REGISTERED = {"Si II": 1169, "Si III": 1418}
SELECTED, ADOPTED, SHELL = EB.SELECTED, EB.ADOPTED, EB.SHELL
REF_TOL_DEX, EFFECT_DEX, STEADY_ETA, TOP = 0.001, 0.01, 1e-8, 3
MODE8 = tuple(f"{k} (mode 8)" for k in CF.RAD)


# ------------------------------------------------------------------ membership
def det_si_bb_groups_fl_info(sm):
    """fl_index -> identity of the FL level (the assembler's lev_a / lev_b key)."""
    return {int(r["fl_index"]): {"i_fl": int(r["i_fl"]), "global_level": int(r["global_level"]),
                                  "level_num": int(r["level_num"]), "upper_ion": r["upper_ion"] == "1",
                                  "E": float(r["E_eV"]), "g": int(r["g"]), "sl": int(r["sl_local"])} for r in sm}


def det_si_bb_groups_assign(stmts, info, n_lo):
    """registered membership of every mode-8 statement keyed by (call, op); returns (assign, findings)."""
    assign, why = {}, []
    for r, _ in stmts:
        if CF.det_si_bb_mode(r) != 8:
            continue
        key = (r["call"], r["op"])
        if key in assign:
            why.append(f"DUPLICATE:op{r['op']}")
            continue
        up, lo, ion = info.get(int(r["lev_a"])), info.get(int(r["lev_b"])), CF.det_si_bb_ion(r, n_lo)
        if up is None or lo is None or ion is None:
            why.append(f"UNMAPPED:op{r['op']}")
            continue
        if any(("Si III" if v["upper_ion"] else "Si II") != ion for v in (up, lo)):
            why.append(f"ION_MISMATCH:op{r['op']}")
            continue
        if int(r["dst"]) != up["sl"] or int(r["src"]) != lo["sl"] or not (up["E"] > lo["E"]):
            why.append(f"ORIENTATION_OR_SL:op{r['op']}")
            continue
        if ion == "Si III":
            assign[key] = "G4"
        else:
            assign[key] = "G1" if lo["level_num"] in G1_LOWER else "G2" if lo["level_num"] in G2_LOWER else "G3"
    return assign, why


def det_si_bb_groups_partition(assign, keys):
    """every mode-8 statement in exactly one registered group; returns findings."""
    why = [f"MISSING:{k}" for k in sorted(set(keys) - set(assign))]
    why += [f"OUTSIDE:{k}" for k in sorted(set(assign) - set(keys))]
    why += [f"UNREGISTERED_GROUP:{k}:{g}" for k, g in sorted(assign.items()) if g not in GROUPS]
    return why


def det_si_bb_groups_callback(assign, group):
    """D8 on the registered statements of one group, nothing elsewhere."""
    def cb(r, x):
        return CF.det_si_bb_d8(r, x) if assign.get((r["call"], r["op"])) == group else None
    return cb


def det_si_bb_groups_subset_check(fn, stmts, assign, group):
    """outputs only: the D8 relation on the group's statements and no change anywhere else."""
    target = [(r, x) for r, x in stmts if assign.get((r["call"], r["op"])) == group]
    bad = CF.det_si_bb_prescription_check("D8", fn, target)
    bad += [r["op"] for r, x in stmts if assign.get((r["call"], r["op"])) != group and fn(r, x) is not None]
    return bad


def det_si_bb_groups_rows_ok(rows, assign):
    return [r["op"] for r in rows if assign.get((r["call"], r["op"])) != r["group"]]


def det_si_bb_groups_state_ok(A, n, levels):
    """a steady state of its own operator: each listed row's backward error finite and within 1e-8."""
    eta = TR.backward(A, n)
    return all(math.isfinite(eta[i]) and eta[i] <= STEADY_ETA for i in levels)


# ------------------------------------------------------------------ negative controls (prereg 9)
def det_si_bb_groups_toy():
    """Si II FL 0..3 (level_num 0, 6, 2, 30; FL 1 has level_num 6) and Si III FL 4..5; four mode-8 statements,
    one per group, plus BF statements to the upper ground."""
    N, gh, n_lo = 6, 4, 4
    lv = ((0, 0, 0.0, 2), (6, 0, 6.8, 6), (2, 0, 5.3, 2), (30, 0, 12.0, 4), (0, 1, 0.0, 1), (1, 1, 6.6, 3))
    sm = [{"i_fl": str(i), "fl_index": str(i), "sl_local": str(i), "within_sl_frac": "1", "global_level": str(i),
           "level_num": str(ln), "upper_ion": str(u), "E_eV": repr(E), "g": str(g)}
          for i, (ln, u, E, g) in enumerate(lv)]

    def bb(k, up, lo, tau, jc, B, A, cu, cd):
        beta = CF.det_si_bb_exponx(tau)
        c, l = beta * jc, 0.3 * jc
        ra, rs, rp = B * c, 0.5 * B * c, A * beta
        x = (ra, rs, rp, cu, cd, 1.0, 1.0, c + l, 8, beta, c, l, tau, 0)
        return mk(k, 1, up, lo, ra + cu, (rs + rp) + cd, up, lo, x)

    def mk(k, site, dst, src, v1, v2, la, lb, x):
        r = {"call": "1", "op": str(k), "site": str(site), "dst": str(dst), "src": str(src), "v1": repr(float(v1)),
             "v2": repr(float(v2)), "lev_a": str(la), "lev_b": str(lb)}
        r.update({f"x{j + 1}": (repr(float(x[j])) if j < len(x) else "") for j in range(14)})
        return r
    ops = [bb(1, 1, 0, 3.0, 1e-5, 4e8, 3e8, 5.0, 4.0), bb(2, 3, 1, 0.7, 2e-5, 2e8, 6e7, 3.0, 2.0),
           bb(3, 3, 2, -0.2, 1e-5, 1e8, 2e7, 1.0, 1.0), bb(4, 5, 4, 2.0, 1e-6, 3e8, 1e8, 2.0, 1.0)]
    for k, lev in ((5, 0), (6, 1), (7, 2), (8, 3)):
        ops.append(mk(k, 5, gh, lev, 2.0 + lev, 0.3, lev, lev, (2.0 + lev, 1.0, 0.3, 0.0, 1e-3, 1e15)))
    return {"N": N, "gh": gh, "n_lo": n_lo, "sm": sm, "ops": ops}


def cmd_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    t = det_si_bb_groups_toy()
    N, gh, n_lo, sm, ops = t["N"], t["gh"], t["n_lo"], t["sm"], t["ops"]
    info = det_si_bb_groups_fl_info(sm)
    stmts = [(r, [TR.num(r[f"x{k}"]) for k in range(1, 15)]) for r in ops if r["site"] == "1"]
    keys = [(r["call"], r["op"]) for r, _ in stmts]
    assign, why = det_si_bb_groups_assign(stmts, info, n_lo)
    expect(not why and assign == {("1", "1"): "G1", ("1", "2"): "G2", ("1", "3"): "G3", ("1", "4"): "G4"}
           and not det_si_bb_groups_partition(assign, keys), "toy membership one per group (positive control)")
    A0, u0, ch0, _ = CF.det_si_bb_build(ops, sm, N, gh)
    builds = {g: CF.det_si_bb_build(ops, sm, N, gh, det_si_bb_groups_callback(assign, g)) for g in GROUPS}
    expect(not u0 and all(not b[1] and CF.det_si_bb_solve(b[0], 10.0)[0] is not None for b in builds.values())
           and all(not det_si_bb_groups_subset_check(det_si_bb_groups_callback(assign, g), stmts, assign, g)
                   for g in GROUPS)
           and all(not CF.det_si_bb_invariance(ch0, builds[g][2], set(MODE8)) for g in GROUPS),
           "toy per-group operators valid, prescriptions and invariance hold (positive control)")

    # 1. orientation, FL number versus level_num, ion
    flip = dict(ops[0], lev_a=ops[0]["lev_b"], lev_b=ops[0]["lev_a"])
    a, w = det_si_bb_groups_assign([(flip, stmts[0][1])], info, n_lo)
    expect(not a and any("ORIENTATION" in s for s in w), "inject upper/lower swapped: refused")
    expect(assign[("1", "2")] == "G2" and info[1]["level_num"] == 6,
           "lower FL 1 carries level_num 6: membership by level_num (G2), not by FL number (G1)")
    wrong_ion = dict(ops[0], lev_a="5")
    a, w = det_si_bb_groups_assign([(wrong_ion, stmts[0][1])], info, n_lo)
    expect(not a and any("ION_MISMATCH" in s for s in w), "inject an upper level of the other ion: refused")

    # 2. duplicate, omission, multiple / unregistered membership
    a, w = det_si_bb_groups_assign(stmts + stmts[:1], info, n_lo)
    expect(any("DUPLICATE" in s for s in w), "inject a duplicate statement identity: refused")
    expect(bool(det_si_bb_groups_partition({k: g for k, g in assign.items() if k != ("1", "3")}, keys)),
           "inject an omitted statement: refused")
    expect(bool(det_si_bb_groups_partition(assign | {("1", "9"): "G1"}, keys)),
           "inject a membership outside the mode-8 set: refused")
    expect(bool(det_si_bb_groups_partition(assign | {("1", "4"): "G5"}, keys)),
           "inject an unregistered group: refused")

    # 3. non-target change, accumulation, spontaneous only, double division
    leak = lambda r, x: CF.det_si_bb_d8(r, x) if assign.get((r["call"], r["op"])) in ("G1", "G2") else None
    expect(bool(det_si_bb_groups_subset_check(leak, stmts, assign, "G2")),
           "inject G2 carrying G1 (accumulated group): refused")
    expect(bool(det_si_bb_groups_subset_check(lambda r, x: CF.det_si_bb_d8(r, x), stmts, assign, "G3")),
           "inject a change on non-target statements: refused")
    spont = lambda r, x: (x[0], x[1], x[2] / x[9]) if assign.get((r["call"], r["op"])) == "G1" else None
    expect(bool(det_si_bb_groups_subset_check(spont, stmts, assign, "G1")), "inject spontaneous-only change: refused")
    twice = lambda r, x: tuple(v / x[9] ** 2 for v in x[0:3]) if assign.get((r["call"], r["op"])) == "G1" else None
    expect(bool(det_si_bb_groups_subset_check(twice, stmts, assign, "G1")), "inject beta divided twice: refused")

    # 4. line rows linked to the wrong group, budgets linked to another state
    rows = [{"call": k[0], "op": k[1], "group": g} for k, g in assign.items()]
    expect(not det_si_bb_groups_rows_ok(rows, assign), "line rows carry their group (positive control)")
    bad_rows = [dict(r, group="G3") if r["op"] == "1" else r for r in rows]
    expect(bool(det_si_bb_groups_rows_ok(bad_rows, assign)), "inject a line row linked to the wrong group: refused")
    n0 = CF.det_si_bb_solve(A0, 10.0)[0]
    n1 = CF.det_si_bb_solve(builds["G1"][0], 10.0)[0]
    expect(det_si_bb_groups_state_ok(builds["G1"][0], n1, range(N)) and
           not det_si_bb_groups_state_ok(builds["G1"][0], n0, range(N)),
           "inject the F0 populations into the G1 budget rows: refused")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_SI_BB_GROUPS_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
def cmd_analyze(run, baseline, out):
    run, baseline = Path(run), Path(baseline)
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

    # ---- G-INPUT: adoption, replay, level map, statement conditions, membership
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
    total = float(next(r for r in TR.read_trace(tr / "si10_final.csv") if r["call"] == cid)["n_total"])
    upper = np.array([r["upper_ion"] == "1" for r in sm])
    E = np.array([float(r["E_eV"]) for r in sm])
    g = np.array([int(r["g"]) for r in sm])
    gl_of = np.array([int(r["global_level"]) for r in sm])
    sl_of = np.array([int(r["sl_local"]) for r in sm])
    info = det_si_bb_groups_fl_info(sm)
    stmts = [(r, [TR.num(r[f"x{k}"]) for k in range(1, 15)]) for r in ops if r["site"] == "1"]
    m8 = [(r, x) for r, x in stmts if CF.det_si_bb_mode(r) == 8]
    bad_stmt = [f"{CF.det_si_bb_check(r, x)}:op{r['op']}" for r, x in m8 if CF.det_si_bb_check(r, x)]
    if bad_stmt:
        gates["G-INPUT"].append(f"STATEMENT_CONDITIONS:{len(bad_stmt)}:" + ";".join(bad_stmt[:5]))
    assign, why_a = det_si_bb_groups_assign(stmts, info, n_lo)
    keys = [(r["call"], r["op"]) for r, _ in m8]
    why_a += det_si_bb_groups_partition(assign, keys)
    if why_a:
        gates["G-INPUT"].append(f"MEMBERSHIP:{len(why_a)}:" + ";".join(why_a[:5]))
    count = {gg: sum(1 for v in assign.values() if v == gg) for gg in GROUPS}
    by_ion = {"Si II": count["G1"] + count["G2"] + count["G3"], "Si III": count["G4"]}
    if by_ion != REGISTERED:
        gates["G-INPUT"].append(f"REGISTERED_COUNTS:{by_ion}")
    tau_sign = {gg: {"negative": 0, "zero": 0, "positive": 0} for gg in GROUPS}
    for r, x in m8:
        gg = assign.get((r["call"], r["op"]))
        if gg:
            tau_sign[gg]["negative" if x[12] < 0 else "zero" if x[12] == 0 else "positive"] += 1

    # ---- operators and G-RATE
    A0, und0, ch0, maps = CF.det_si_bb_build(ops, sm, N, gh)
    A8, und8, ch8, _ = CF.det_si_bb_build(ops, sm, N, gh, CF.det_si_bb_d8)
    builds = {"F0": (A0, ch0), "F8": (A8, ch8)}
    und = und0 + und8
    for gg in GROUPS:
        cb = det_si_bb_groups_callback(assign, gg)
        bad = det_si_bb_groups_subset_check(cb, stmts, assign, gg)
        if bad:
            gates["G-RATE"].append(f"PRESCRIPTION:{gg}:{len(bad)}")
        if count[gg] == 0:
            builds[f"F-{gg}"] = (A0, ch0)
            continue
        A, ug, chg, _ = CF.det_si_bb_build(ops, sm, N, gh, cb)
        und += ug
        builds[f"F-{gg}"] = (A, chg)
    if und or maps["gh_fl"] is None:
        return TR.refuse(gates["G-INPUT"] + ["FL_OPERATOR_UNDEFINED:" + ",".join(sorted(set(und)))])
    for name, (A, chs) in builds.items():
        if not CF.det_si_bb_channel_sum_ok(A, chs):
            gates["G-RATE"].append(f"CHANNEL_SUM:{name}")
        if EB.det_si_exc_conservation(A):
            gates["G-RATE"].append(f"CONSERVATION:{name}")
        if name != "F0":
            inv = CF.det_si_bb_invariance(ch0, chs, set(MODE8))
            if inv:
                gates["G-RATE"].append(f"INVARIANCE:{name}:" + ",".join(inv))

    # ---- G-SOLVE
    sol, eta_max = {}, {}
    for name, (A, _) in builds.items():
        x, why_s, eta_max[name] = CF.det_si_bb_solve(A, total)
        sol[name] = x
        if why_s:
            gates["G-SOLVE"].append(f"{name}:" + ",".join(why_s))
    R = {name: (TR.ratio(sol[name], upper) if sol[name] is not None else float("nan")) for name in builds}
    base = json.loads(baseline.read_text())["ratios"] if baseline.is_file() else {}
    ref_dev = {s: TR.dex(R[s], float(base.get(s, "nan"))) for s in ("F0", "F8")}
    for s, dv in ref_dev.items():
        if not (abs(dv) <= REF_TOL_DEX):
            gates["G-SOLVE"].append(f"{s}_REFERENCE:{R[s]}vs{base.get(s)}")

    # ---- G-TABLE: public correspondence, state linkage
    t_shell = TR.dump_row(run, stop_iter, "shell", SHELL)[0]
    vel = (float(t_shell["v_inner_cm_s"]) + float(t_shell["v_outer_cm_s"])) / 2e5
    pub, pub_vel, pub_ii, pub_iii = EB.det_si_exc_public(vel)
    if not (math.isclose(vel, TR.REGISTERED_VEL["lumina_vel_mid_kms"]) and
            math.isclose(pub_vel, TR.REGISTERED_VEL["public_vel_mid_kms"]) and pub == pub_iii / pub_ii):
        gates["G-TABLE"].append(f"PUBLIC_CORRESPONDENCE:{vel}->{pub_vel}")
    sel_fl = [lvmap[k] for k in SELECTED]
    for name, (A, _) in builds.items():
        if sol[name] is not None and not det_si_bb_groups_state_ok(A, sol[name], sel_fl):
            gates["G-TABLE"].append(f"STATE_LINKAGE:{name}")
    w = np.where(~upper, g * np.exp(-E * EB.EV_TO_ERG / (EB.K_BOLTZMANN * T_e)), 0.0)
    f_lte = w / float(w.sum())

    # ---- the table
    rows = []

    def row(kind, **v):
        rows.append(dict({"kind": kind}, **v))
    row("public", state="CMFGEN toy06 19.480 d", time_d=19.48, vel_kms=pub_vel, si_II=pub_ii, si_III=pub_iii,
        III_II=pub, note="public file holds ion fractions only; level rows have no public counterpart")
    d0 = TR.dex(R["F0"], pub)
    delta, cls = {}, {}
    for name in builds:
        d = TR.dex(R[name], pub)
        delta[name] = TR.dex(R[name], R["F0"])
        G = abs(d0) - abs(d) if name != "F0" and math.isfinite(d) and math.isfinite(d0) else None
        c = None
        if name != "F0":
            eff = "effect resolved" if abs(delta[name]) > EFFECT_DEX else "effect unresolved"
            dist = "undefined" if G is None else "toward" if G > EFFECT_DEX else "away" if G < -EFFECT_DEX \
                else "distance unresolved"
            c = cls[name] = f"{eff}; {dist}"
        row("ion sum", state=name, III_II=R[name], d_pub=d, delta_F0=delta[name], G=G, cls=c,
            valid=sol[name] is not None, reference_dex=ref_dev.get(name))
    eff = sorted((gg for gg in GROUPS if sol[f"F-{gg}"] is not None), key=lambda gg: -abs(delta[f"F-{gg}"]))
    for gg in GROUPS:
        row("group", state=f"F-{gg}", group=gg, rule=RULES[gg], statements=count[gg],
            tau_sign=json.dumps(tau_sign[gg]), rank=(eff.index(gg) + 1) if gg in eff else None,
            delta_F0=delta[f"F-{gg}"], cls=cls[f"F-{gg}"],
            note="largest independent effect" if eff and gg == eff[0] else "")
    S = sum(delta[f"F-{gg}"] for gg in GROUPS)
    eps = delta["F8"] - S
    row("non-additivity", state="F8 vs groups", delta_F8=delta["F8"], sum_groups=S, residual=eps,
        note="non-additivity resolved in this partition" if abs(eps) > EFFECT_DEX else
             "within reporting resolution (not a proof of additivity)")
    budgets, f_ref = {}, {}
    for k in SELECTED:
        i = lvmap[k]
        ident = {"level": k, "global_level": int(gl_of[i]), "i_fl": i, "sl": int(sl_of[i]), "E_eV": float(E[i]),
                 "g": int(g[i]), "f_lte": float(f_lte[i])}
        for name, (A, chs) in builds.items():
            n = sol[name]
            if n is None:
                row("level total", state=name, **ident, note="state undefined")
                continue
            n_ii = float(n[~upper].sum())
            f_i = float(n[i]) / n_ii
            bud = EB.det_si_exc_budget(chs, n, i, n_ii, float(f_lte[i]))
            budgets[(k, name)] = bud
            ref = budgets.get((k, "F0"))
            for c in sorted(bud):
                b = bud[c]
                rb = ref.get(c) if (ref is not None and name != "F0") else None
                if b["I"] == 0.0 and b["D"] == 0.0 and (rb is None or (rb["I"] == 0.0 and rb["D"] == 0.0)):
                    continue
                v = {q: b[q] for q in ("I", "O", "B", "D")}
                if rb is not None:
                    v.update({f"d{q}": b[q] - rb[q] for q in ("I", "O", "B", "D")})
                row("level channel", state=name, **ident, channel=c, f=f_i, b_rel=f_i / float(f_lte[i]), **v)
            tot = {q: sum(b[q] for b in bud.values()) for q in ("I", "O", "B", "D")}
            if name == "F0":
                f_ref[k] = f_i
            row("level total", state=name, **ident, f=f_i, b_rel=f_i / float(f_lte[i]),
                b_rel_vs_F0=(f_i / f_ref[k]) if name != "F0" and k in f_ref else None,
                eta_row=float(TR.backward(A, n)[i]), **tot)
        if all((k, s) in budgets for s in ["F0", "F8"] + [f"F-{gg}" for gg in GROUPS]):
            b0, b8 = budgets[(k, "F0")], budgets[(k, "F8")]
            for c in sorted(set(b0) | set(b8)):
                if all(c in budgets[(k, s)] for s in [f"F-{gg}" for gg in GROUPS]) and c in b0 and c in b8:
                    v = {f"eps_{q}": (b8[c][q] - b0[c][q]) - sum(budgets[(k, f"F-{gg}")][c][q] - b0[c][q]
                                                                for gg in GROUPS) for q in ("I", "O", "B", "D")}
                    if any(v.values()):
                        row("non-additivity level", state="F8 vs groups", **ident, channel=c, **v)
    n0 = sol["F0"]
    fl_of = maps["fl_of_global"]
    line_rows = []
    for r, x in m8:
        key = (r["call"], r["op"])
        gg = assign.get(key)
        up, lo = info.get(int(r["lev_a"])), info.get(int(r["lev_b"]))
        new = CF.det_si_bb_d8(r, x)
        W = None
        if n0 is not None and up is not None and lo is not None:
            W = float(n0[fl_of[int(r["lev_b"])]] * abs(new[0] - x[0]) +
                      n0[fl_of[int(r["lev_a"])]] * (abs(new[1] - x[1]) + abs(new[2] - x[2])))
        line_rows.append({"call": r["call"], "op": r["op"], "aux": int(r["aux"]), "group": gg,
                          "ion": CF.det_si_bb_ion(r, n_lo), "up_fl": int(r["lev_a"]), "lo_fl": int(r["lev_b"]),
                          "up_level_num": up["level_num"] if up else None, "lo_level_num": lo["level_num"] if lo else None,
                          "up_sl": up["sl"] if up else None, "lo_sl": lo["sl"] if lo else None,
                          "up_E_eV": up["E"] if up else None, "lo_E_eV": lo["E"] if lo else None,
                          "up_g": up["g"] if up else None, "lo_g": lo["g"] if lo else None,
                          "tau_eff": x[12], "beta": x[9], "c": x[10], "l": x[11], "Jbar": x[7], "provenance": x[13],
                          "J_c": x[10] / x[9], "r_a": x[0], "r_s": x[1], "r_p": x[2],
                          "r_a_new": new[0], "r_s_new": new[1], "r_p_new": new[2], "W_e": W, "representative": ""})
    if len(line_rows) != len(set((lr["call"], lr["op"]) for lr in line_rows)) or len(line_rows) != len(m8):
        gates["G-TABLE"].append("LINE_ROWS_NOT_ONE_PER_STATEMENT")
    bad_rows = det_si_bb_groups_rows_ok(line_rows, assign)
    if bad_rows:
        gates["G-TABLE"].append(f"LINE_ROW_GROUP:{len(bad_rows)}")
    for gg in GROUPS:
        cand = sorted((lr for lr in line_rows if lr["group"] == gg and lr["W_e"] is not None),
                      key=lambda lr: (-lr["W_e"], lr["aux"], int(lr["op"])))
        for rank, lr in enumerate(cand[:TOP], 1):
            lr["representative"] = f"{gg} #{rank}"
    for lr in line_rows:
        row("line", state="F0 inputs", **lr)
    all_gates = [f for v in gates.values() for f in v]
    valid = not all_gates and all(sol[name] is not None for name in builds)
    code = 2 if not valid else 0 if any(abs(delta[f"F-{gg}"]) > EFFECT_DEX for gg in GROUPS) else 1
    label = {0: "GROUP_EFFECT_RESOLVED", 1: "NO_GROUP_EFFECT_RESOLVED", 2: "INDETERMINATE"}[code]
    row("quality", state="", keys=json.dumps({k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")}),
        T_e=T_e, n_e=float(call["n_e"]), total=total, eta_max=json.dumps(eta_max), reference_dex=json.dumps(ref_dev),
        gates=";".join(all_gates) or "all pass", branch=f"{code} {label}",
        note="fixed continuum J, T_e, n_e, CE partners and Si total; evidence.json holds input SHA-256")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "si10_bb_groups_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    with open(out / "si10_bb_groups_table.md", "x") as f:
        f.write(f"# Si mode-8 sensitivity by line group, material 13, Si shell 10 (branch {code} {label})\n")
        for kind in dict.fromkeys(r["kind"] for r in rows):
            sub = [r for r in rows if r["kind"] == kind]
            cols = [c for c in fields if c != "kind" and any(r.get(c) not in (None, "") for r in sub)]
            f.write(f"\n## {kind}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in sub:
                f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
    used = [tr / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_commit.csv", "si10_cevec.csv", "si10_ops.csv",
                             "si10_slmap.csv", "si10_final.csv", f"si10_mat_{int(cid):06d}_gen.bin")]
    used += [run / "RUN_FOOTER.txt", run / f"work/physics_comparison/physics_DET_iter{stop_iter:04d}.shell.csv",
             TR.PUBLIC_SI, TR.DIAG8_PUBLIC, baseline]
    evidence = {"schema": "DET_SI_BB_GROUPS_ANALYSIS_V1", "prereg": PREREG, "argv": sys.argv,
                "adopted": {k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")},
                "T_e": T_e, "n_e": float(call["n_e"]), "total": total, "ratios": R, "delta_F0": delta,
                "public": pub, "public_vel_kms": pub_vel, "reference_dex": ref_dev, "eta_max": eta_max,
                "counts": count, "tau_sign": tau_sign, "classification": cls, "rank": eff,
                "non_additivity": {"delta_F8": delta["F8"], "sum_groups": S, "residual": eps},
                "representatives": {lr["representative"]: [lr["op"], lr["aux"]] for lr in line_rows
                                    if lr["representative"]},
                "branch": code, "label": label, "gates": gates,
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(p).resolve()): TR.sha(Path(p))
                         for p in (__file__, TR.__file__, EB.__file__, CF.__file__)}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str) + "\n")
    print(f"BRANCH {code} {label} R={R} delta={delta} eps={eps} rank={eff}")
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
    s.add_argument("--baseline", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_negative() if a.cmd == "negative" else cmd_analyze(a.run, a.baseline, a.out)


if __name__ == "__main__":
    sys.exit(main())
