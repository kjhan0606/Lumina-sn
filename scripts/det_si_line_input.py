#!/usr/bin/env python3
"""DET-SI-LINE-INPUT (prereg docs/PREREG_DET_SI_LINE_INPUT_2026-10-11.md).

For the six registered representative lines of DET-SI-BB-GROUPS (G1: Si II 1533.4, 1264.7, 1526.7 A; G2: Si II
6348.9, 3857.1, 6373.1 A) at Si shell 10: the inputs consumed by the adopted call 36 (tau_eff, beta, c, l, Jbar and
the three radiative rates) against
  (a) the sealed production formulas evaluated on the production inputs (material-12 populations = call-1 PRE_FL,
      line_list.csv / levels.csv, t = time_explosion), with J_c taken from the production ledger w5_lines.csv
      because the fine continuum arrays were not persisted (full J_c recomputation unresolved), and
  (b) the production -> consumption transfer (ledger call 12 / radiation 12 / material 12 -> trace call 36),
plus physics comparisons in separate columns (tau with the call-36 FINAL_FL populations, the coarse J_nu bin of the
comparison dump, B_nu(T_e)).

Subcommands
  negative                                   negative controls of the new code (prereg 9)
  analyze --run T --baseline V --out DIR     G-IDENTITY, G-CALC, G-TABLE, G-RUN and the main table
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
import det_si_bb_groups as GR                                                     # noqa: E402

PREREG = "docs/PREREG_DET_SI_LINE_INPUT_2026-10-11.md"
LINES = (("G1", 760321, 1, 7), ("G1", 632403, 1, 10), ("G1", 757019, 0, 7),
         ("G2", 1676767, 7, 12), ("G2", 1353542, 6, 12), ("G2", 1679119, 7, 11))
ADOPTED, SHELL = EB.ADOPTED, EB.SHELL
PRODUCTION_CALL = {"bundle": 1, "ne_iter": 1, "ce_pass": 0}
LEDGER_KEY = ("12", "CMFGEN_NONOVERLAP_SOBOLEV", "12", "12", str(SHELL))
SOBOLEV_COEFF = 2.6540281e-02                       # lumina.h SOBOLEV_COEFF (production constant)
H_PLANCK, C_LIGHT, K_BOLTZMANN = 6.62607015e-27, 2.99792458e10, 1.380649e-16
FOUR_PI = 12.56637061435917295385057353311801153679
INTERNAL_OPACITY_TO_CGS = 1.0e-10
RTOL = 1e-10
CONTEXT = {"public_III_II": 105.69735889036637, "F0": 3.5597568125822305, "d_F0_pub": -1.47264,
           "G1_delta": -0.80359, "G2_delta": 0.10258, "time_d": 19.48, "vel_kms": "11544 <-> 11534.59"}


# ------------------------------------------------------------------ production formulas (sealed transcriptions)
def det_si_line_input_match(a, b):
    """prereg 7: relative difference within 1e-10, exact zeros only equal to exact zeros; None is unresolved."""
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None, None, None
    D, M = abs(a - b), max(abs(a), abs(b))
    return (D / M <= RTOL) if M > 0 else (a == 0.0 and b == 0.0), D, (D / M if M > 0 else 0.0)


def det_si_line_input_tau(f_lu, lam_cm, t, n_l, n_u, g_l, g_u):
    """a208_signed_sobolev_counted: coefficient f lambda t (n_l - (g_l/g_u) n_u), production evaluation order."""
    difference = n_l - (float(g_l) / float(g_u)) * n_u
    return SOBOLEV_COEFF * f_lu * lam_cm * t * difference, abs(SOBOLEV_COEFF * f_lu * lam_cm * t) * (
        abs(n_l) + abs(float(g_l) / float(g_u) * n_u))


def det_si_line_input_exponx(tau):
    """line_net_cmfgen_exponx: (beta, (1-beta)/tau) with the small-tau polynomial; None where undefined."""
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
    return value, companion


def det_si_line_input_material(n_u, A_ul, nu, tau, t):
    """line_net_sobolev_material with the CMFGEN SRCE_CHK policy and one simultaneous line."""
    eta = n_u * A_ul * H_PLANCK * nu / FOUR_PI
    tau_eff, srce = tau, 0
    if tau < -0.5:
        tau_eff, srce = INTERNAL_OPACITY_TO_CGS / 1.0 * C_LIGHT * t / nu, 1
    return eta, tau_eff, srce


def det_si_line_input_radiation(eta, tau_eff, J_c, nu, t):
    """line_net_sobolev_radiation: beta, c = beta J_c, l = eta (c t / nu) (1 - beta)/tau, Jbar = c + l."""
    e = det_si_line_input_exponx(tau_eff)
    if e is None:
        return None
    beta, companion = e
    c = beta * J_c
    l = eta * (C_LIGHT * t / nu) * companion
    return {"beta": beta, "c": c, "l": l, "Jbar": c + l}


def det_si_line_input_gauss(nu, dnu, J, nu_l, vdop, ndoppler=4.0):
    """line_jbar Gaussian profile mean on the fine grid (|x| <= 4, weights e^-x^2 dnu); None without support."""
    num = den = 0.0
    width = nu_l * vdop / C_LIGHT
    for a, d, j in zip(nu, dnu, J):
        x = (a - nu_l) / width
        if abs(x) <= ndoppler:
            w = math.exp(-x * x) * d
            num += w * j
            den += w
    return num / den if den > 0.0 else None


def det_si_line_input_planck(nu, T):
    return 2.0 * H_PLANCK * nu ** 3 / (C_LIGHT ** 2 * math.expm1(H_PLANCK * nu / (K_BOLTZMANN * T)))


def det_si_line_input_jc_status(fine_available, source):
    """the J_c recomputation is complete only from the production fine arrays; a coarse field never counts."""
    if source == "fine" and fine_available:
        return "computed"
    return "미확보"


def det_si_line_input_production_call(calls):
    """call 1 of the generation-13 candidate: bundle 1, n_e iteration 1, CE pass 0 (the production population)."""
    c = [r for r in calls if all(int(r[k]) == v for k, v in PRODUCTION_CALL.items())]
    return c[0] if len(c) == 1 and c[0]["call"] == "1" else None


def det_si_line_input_ledger(rows_iter, line_ids):
    """production ledger rows (call 12, CMFGEN_NONOVERLAP_SOBOLEV, radiation 12, material 12, shell 10)."""
    prefix = ",".join(LEDGER_KEY) + ","
    out = {}
    for ln in rows_iter:
        if not ln.startswith(prefix):
            continue
        t = ln.rstrip("\n").split(",")
        lid = int(t[5])
        if lid in line_ids:
            out.setdefault(lid, []).append(t)
    return out


# ------------------------------------------------------------------ negative controls (prereg 9)
def det_si_line_input_toy():
    """one line with a known production state: populations, atomic data, time, J_c and the derived record."""
    g_l, g_u, f_lu, lam_cm, A, B_lu, B_ul = 2, 4, 0.2, 1.5e-5, 7e8, 3e9, 1.5e9
    t, n_l, n_u, J_c = 1.683072e6, 3.0e2, 4.0e1, 1.2e-7
    nu = C_LIGHT / lam_cm
    tau, s_tau = det_si_line_input_tau(f_lu, lam_cm, t, n_l, n_u, g_l, g_u)
    eta, tau_eff, srce = det_si_line_input_material(n_u, A, nu, tau, t)
    rad = det_si_line_input_radiation(eta, tau_eff, J_c, nu, t)
    return {"g_l": g_l, "g_u": g_u, "f_lu": f_lu, "lam_cm": lam_cm, "A": A, "B_lu": B_lu, "B_ul": B_ul, "t": t,
            "n_l": n_l, "n_u": n_u, "J_c": J_c, "nu": nu, "tau": tau, "eta": eta, "rad": rad,
            "record": {"tau_eff": tau_eff, "beta": rad["beta"], "c": rad["c"], "l": rad["l"], "Jbar": rad["Jbar"],
                       "r_a": B_lu * rad["c"], "r_s": B_ul * rad["c"], "r_p": A * rad["beta"]}}


def cmd_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    t = det_si_line_input_toy()
    rec = t["record"]

    def recompute(n_l=None, n_u=None, lam=None, tt=None, stim=True, beta_twice=False, l_into_c=False, gl=None):
        n_l = t["n_l"] if n_l is None else n_l
        n_u = t["n_u"] if n_u is None else n_u
        lam = t["lam_cm"] if lam is None else lam
        tt = t["t"] if tt is None else tt
        gl = t["g_l"] if gl is None else gl
        tau = det_si_line_input_tau(t["f_lu"], lam, tt, n_l, n_u if stim else 0.0, gl, t["g_u"])[0]
        eta, tau_eff, _ = det_si_line_input_material(n_u, t["A"], C_LIGHT / lam, tau, tt)
        rad = det_si_line_input_radiation(eta, tau_eff, t["J_c"], C_LIGHT / lam, tt)
        c = rad["c"] * (rad["beta"] if beta_twice else 1.0) + (rad["beta"] * rad["l"] if l_into_c else 0.0)
        return {"tau_eff": tau_eff, "c": c, "l": rad["l"], "Jbar": c + rad["l"], "beta": rad["beta"]}

    def agree(r):
        return all(det_si_line_input_match(r[k], rec[k])[0] for k in ("tau_eff", "c", "l", "Jbar", "beta"))
    base = recompute()
    expect(agree(base), "production inputs reproduce the record (positive control)")
    # 1. consumption populations / call numbers
    alt = recompute(n_l=t["n_l"] * 0.7, n_u=t["n_u"] * 1.3)
    expect(t["n_l"] * 0.7 != t["n_l"] and not agree(alt), "inject consumption (generation 13) populations: refused")
    calls = [{"call": "1", "bundle": "1", "ne_iter": "1", "ce_pass": "0"},
             {"call": "36", "bundle": "1", "ne_iter": "18", "ce_pass": "1"}]
    expect(det_si_line_input_production_call(calls)["call"] == "1" and
           det_si_line_input_production_call(calls[1:]) is None, "inject the trace call number as the production call: "
           "refused")
    lines = ["12,CMFGEN_NONOVERLAP_SOBOLEV,12,12,10,5,14,1,x\n", "36,CMFGEN_NONOVERLAP_SOBOLEV,36,13,10,5,14,1,x\n"]
    expect(5 in det_si_line_input_ledger(lines[:1], {5}) and not det_si_line_input_ledger(lines[1:], {5}),
           "inject a ledger row of another call/generation: not selected")
    # 2. upper/lower, row, ion, units
    sw = recompute(n_l=t["n_u"], n_u=t["n_l"])
    expect(t["n_u"] != t["n_l"] and not agree(sw), "inject swapped lower/upper populations: refused")
    expect(not agree(recompute(gl=3)), "inject a wrong statistical weight row: refused")
    expect(not agree(recompute(lam=t["lam_cm"] * 1e8)), "inject wavelength in Angstrom instead of cm: refused")
    expect(not agree(recompute(tt=t["t"] / 86400.0)), "inject time in days instead of seconds: refused")
    info = GR.det_si_bb_groups_fl_info([{"i_fl": "0", "fl_index": "0", "sl_local": "0", "global_level": "0",
                                         "level_num": "1", "upper_ion": "1", "E_eV": "0", "g": "1"}])
    expect(info[0]["upper_ion"] and info[0]["level_num"] == 1, "ion identity read from the level map (positive "
           "control for the ion check)")
    # 3. stimulated term, beta twice, l added to the incident field
    expect(not agree(recompute(stim=False)), "inject a missing stimulated-emission term: refused")
    expect(not agree(recompute(beta_twice=True)), "inject beta applied twice: refused")
    expect(not agree(recompute(l_into_c=True)), "inject l added to the incident continuum: refused")
    # 4. Gaussian weights, support, coarse field as fine
    nu_l, vd = 4.7e14, 1e6
    w = nu_l * vd / C_LIGHT
    grid = [nu_l + k * 0.5 * w for k in range(-12, 13)]
    dnu = [0.5 * w * (1.0 + 0.1 * (k % 3)) for k in range(-12, 13)]
    Jf = [1e-5 * (1.0 + 0.05 * k) if abs(k) <= 8 else 1.0 for k in range(-12, 13)]   # |x| > 4 carries a large field
    g_ok = det_si_line_input_gauss(grid, dnu, Jf, nu_l, vd)
    num = sum(math.exp(-((a - nu_l) / w) ** 2) * j for a, j in zip(grid, Jf) if abs((a - nu_l) / w) <= 4)
    den = sum(math.exp(-((a - nu_l) / w) ** 2) for a in grid if abs((a - nu_l) / w) <= 4)
    expect(g_ok is not None and not det_si_line_input_match(num / den, g_ok)[0], "inject Gaussian weights without "
           "dnu: differs from the registered mean")
    allpts = sum(math.exp(-((a - nu_l) / w) ** 2) * d * j for a, d, j in zip(grid, dnu, Jf)) / \
        sum(math.exp(-((a - nu_l) / w) ** 2) * d for a, d in zip(grid, dnu))
    expect(not det_si_line_input_match(allpts, g_ok)[0], "inject values outside the |x|<=4 support: differs")
    expect(det_si_line_input_jc_status(False, "coarse") == "미확보" and
           det_si_line_input_jc_status(True, "coarse") == "미확보" and det_si_line_input_jc_status(True, "fine") ==
           "computed", "inject the coarse field labelled as the production fine arrays: stays unresolved")
    # 5. missing inputs filled with zero or the record
    expect(det_si_line_input_match(None, rec["tau_eff"])[0] is None, "missing input is unresolved, not a match")
    expect(det_si_line_input_match(0.0, rec["tau_eff"])[0] is False, "inject zero for a missing input: refused")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_SI_LINE_INPUT_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


# ------------------------------------------------------------------ analysis
def cmd_analyze(run, baseline, out):
    run, baseline = Path(run), Path(baseline)
    tr = run / "si_trace"
    pc = run / "work/physics_comparison"
    out = TR.new_dir(out)
    gates = {k: [] for k in ("G-IDENTITY", "G-CALC", "G-TABLE", "G-RUN")}
    part, cvd = os.environ.get("SLURM_JOB_PARTITION"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if part != "a10" or cvd not in ("", "NoDevFiles"):
        gates["G-RUN"].append(f"NOT_A10_CPU:{part}:{cvd}")

    # ---- G-IDENTITY: generations, production population, statements, atomic rows, ledger, time, settings
    calls = TR.read_trace(tr / "si10_calls.csv")
    call, why = EB.det_si_exc_adopt(TR.read_trace(tr / "si10_ne.csv"), calls, TR.read_trace(tr / "si10_cevec.csv"),
                                    TR.read_trace(tr / "si10_commit.csv"), ADOPTED)
    if call is None:
        return TR.refuse(why)
    cid = call["call"]
    pcall = det_si_line_input_production_call(calls)
    if pcall is None:
        return TR.refuse(["PRODUCTION_CALL"])
    man9 = json.loads((pc / "physics_DET_iter0009.manifest.json").read_text())
    man10 = json.loads((pc / "physics_DET_iter0010.manifest.json").read_text())
    if (man9.get("population_generation"), man10.get("population_generation"), man10.get("radiation_generation")) \
            != (12, 13, 12) or int(call["rad_generation"]) != 12 or int(pcall["core_generation"]) != 13:
        gates["G-IDENTITY"].append("GENERATIONS")
    vec = {}
    for r in TR.read_trace(tr / "si10_vec.csv"):
        if r["call"] in ("1", cid) and r["phase"] in ("PRE_FL", "FINAL_FL"):
            vec.setdefault((r["call"], r["phase"]), {})[int(r["i"])] = float(r["value"])
    sm1 = sorted((r for r in TR.read_trace(tr / "si10_slmap.csv") if r["call"] == "1"), key=lambda r: int(r["i_fl"]))
    smA = sorted((r for r in TR.read_trace(tr / "si10_slmap.csv") if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    N_fl = int(call["N_fl"])
    if len(sm1) != N_fl or [(r["fl_index"], r["level_num"], r["upper_ion"], r["g"]) for r in sm1] != \
            [(r["fl_index"], r["level_num"], r["upper_ion"], r["g"]) for r in smA]:
        gates["G-IDENTITY"].append("LEVEL_MAP_CALL1_VS_CALL36")
    pre = np.array([vec.get(("1", "PRE_FL"), {}).get(i, np.nan) for i in range(N_fl)])
    fin = np.array([vec.get((cid, "FINAL_FL"), {}).get(i, np.nan) for i in range(N_fl)])
    if not (np.isfinite(pre).all() and np.isfinite(fin).all()):
        return TR.refuse(["POPULATION_VECTORS_INCOMPLETE"])
    old = {int(r["i_fl"]): float(r["old"]) for r in TR.read_trace(tr / "si10_cevec.csv")
           if (int(r["bundle"]), int(r["ne_iter"]), int(r["ce_pass"])) == (1, 1, 0)}
    if len(old) != N_fl or any(old[i] != pre[i] for i in range(N_fl)):
        gates["G-IDENTITY"].append("PRE_FL_VS_CE_OLD")
    upper = np.array([r["upper_ion"] == "1" for r in sm1])
    ion9 = TR.dump_row(run, 9, "ion", SHELL)[0]
    sums = {"Si II": float(pre[~upper].sum()), "Si III": float(pre[upper].sum())}
    ref9 = {"Si II": float(ion9["Z14_nlte_q1_cm3"]), "Si III": float(ion9["Z14_nlte_q2_cm3"])}
    sum_dev = {k: det_si_line_input_match(sums[k], ref9[k]) for k in sums}
    if not all(v[0] for v in sum_dev.values()):
        gates["G-IDENTITY"].append("PRE_FL_ION_SUMS_VS_MATERIAL12")
    info = GR.det_si_bb_groups_fl_info(sm1)
    fl_of = {int(r["fl_index"]): int(r["i_fl"]) for r in sm1}
    cfg = json.loads((run / "input/model/config.json").read_text())
    t_exp = float(cfg["time_explosion_s"])
    if t_exp != float(man10.get("epoch_s", "nan")):
        gates["G-IDENTITY"].append("TIME_EXPLOSION")
    exports = (run / "input/resolved_lumina.exports").read_text()
    if 'LUMINA_CMF_FINE_VDOP="1000000"' not in exports or 'LUMINA_NLTE_SKIP_Z=""' not in exports:
        gates["G-IDENTITY"].append("SETTINGS")
    line_ids = {lid for _, lid, _, _ in LINES}
    stm = {}
    for r in TR.read_trace(tr / "si10_ops.csv"):
        if r["call"] == cid and r["site"] == "1" and int(r["aux"]) in line_ids:
            stm.setdefault(int(r["aux"]), []).append(r)
    atom = {}
    with open(run / "input/model/line_list.csv") as f:
        head = f.readline().rstrip("\n").split(",")
        for ln in f:
            if not ln.startswith("14,1,"):
                continue
            t = ln.rstrip("\n").split(",")
            if int(t[4]) in line_ids:
                atom.setdefault(int(t[4]), []).append(dict(zip(head, t)))
    lev = {}
    with open(run / "input/model/levels.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["atomic_number"] == "14" and r["ion_number"] == "1":
                lev[int(r["level_number"])] = r
    with open(run / "work/w5_lines.csv") as f:
        ledger = det_si_line_input_ledger(f, line_ids)
    jgrid = [r for r in TR.read_trace(pc / "physics_DET_iter0010.spectral.csv") if int(r["shell_id"]) == SHELL]
    T_e = float(call["T_e"])

    # ---- per line
    rows, findings = [], []
    for grp, lid, lo_ln, up_ln in LINES:
        row = {"group": grp, "aux": lid, "lower_level_num": lo_ln, "upper_level_num": up_ln}
        why_l = []
        st, at, lg = stm.get(lid, []), atom.get(lid, []), ledger.get(lid, [])
        if len(st) != 1 or len(at) != 1 or len(lg) != 1:
            why_l.append(f"ROWS:stmt{len(st)}:atom{len(at)}:ledger{len(lg)}")
            row.update(conclusion="미해결", unresolved=";".join(why_l))
            rows.append(row)
            findings += [f"{lid}:{w}" for w in why_l]
            continue
        r, a, L = st[0], at[0], lg[0]
        x = [TR.num(r[f"x{k}"]) for k in range(1, 15)]
        up, lo = info.get(int(r["lev_a"])), info.get(int(r["lev_b"]))
        if (CF.det_si_bb_mode(r) != 8 or CF.det_si_bb_ion(r, int(call["n_lo_super"])) != "Si II" or up is None or
                lo is None or (lo["level_num"], up["level_num"]) != (lo_ln, up_ln) or up["upper_ion"] or
                lo["upper_ion"] or (int(a["level_number_lower"]), int(a["level_number_upper"])) != (lo_ln, up_ln) or
                L[6:8] != ["14", "1"] or L[17] != "OK" or float(L[8]) != float(a["nu"])):
            why_l.append("STATEMENT_IDENTITY")
        g_l, g_u = int(lev[lo_ln]["g"]), int(lev[up_ln]["g"])
        if (g_l, g_u) != (lo["g"], up["g"]):
            why_l.append("STATISTICAL_WEIGHTS")
        f_lu, lam_cm, nu = float(a["f_lu"]), float(a["wavelength_cm"]), float(a["nu"])
        A_ul, B_lu, B_ul = float(a["A_ul"]), float(a["B_lu"]), float(a["B_ul"])
        n_l, n_u = float(pre[fl_of[int(r["lev_b"])]]), float(pre[fl_of[int(r["lev_a"])]])
        tau_raw, s_tau = det_si_line_input_tau(f_lu, lam_cm, t_exp, n_l, n_u, g_l, g_u)
        eta, tau_eff, srce = det_si_line_input_material(n_u, A_ul, nu, tau_raw, t_exp)
        led = {"J_c": float(L[9]), "beta": float(L[11]), "l": float(L[12]), "Jbar": float(L[13]),
               "tau_eff": float(L[15]), "srce_chk": int(L[16])}
        rad = det_si_line_input_radiation(eta, tau_eff, led["J_c"], nu, t_exp)
        recd = {"tau_eff": x[12], "beta": x[9], "c": x[10], "l": x[11], "Jbar": x[7], "r_a": x[0], "r_s": x[1],
                "r_p": x[2], "provenance": x[13]}
        row.update(call=cid, op=r["op"], ion="Si II", up_fl=int(r["lev_a"]), lo_fl=int(r["lev_b"]),
                   lambda_cm=lam_cm, nu=nu, f_lu=f_lu, A_ul=A_ul, B_lu=B_lu, B_ul=B_ul, g_l=g_l, g_u=g_u,
                   t_s=t_exp, population="call 1 PRE_FL (material 12)", n_l=n_l, n_u=n_u,
                   production="ledger call 12 / radiation 12 / material 12", consumption=f"trace call {cid}",
                   S_tau=s_tau)
        checks = {}
        # production formula: recomputed vs ledger (tau, beta, l, Jbar) and vs record (tau_eff, beta, c, l, Jbar)
        calc = {"tau_eff": tau_eff, "beta": rad["beta"] if rad else None, "c": rad["c"] if rad else None,
                "l": rad["l"] if rad else None, "Jbar": rad["Jbar"] if rad else None}
        for k in ("tau_eff", "beta", "l", "Jbar"):
            checks[f"calc_vs_ledger_{k}"] = det_si_line_input_match(calc[k], led[k])
        for k in ("tau_eff", "beta", "c", "l", "Jbar"):
            checks[f"calc_vs_record_{k}"] = det_si_line_input_match(calc[k], recd[k])
        # transfer: ledger vs record
        for k in ("tau_eff", "beta", "l", "Jbar"):
            checks[f"ledger_vs_record_{k}"] = det_si_line_input_match(led[k], recd[k])
        checks["ledger_vs_record_c"] = det_si_line_input_match(led["beta"] * led["J_c"], recd["c"])
        # radiative rates with the independent atomic coefficients
        checks["rate_r_a"] = det_si_line_input_match(B_lu * recd["c"], recd["r_a"])
        checks["rate_r_s"] = det_si_line_input_match(B_ul * recd["c"], recd["r_s"])
        checks["rate_r_p"] = det_si_line_input_match(A_ul * recd["beta"], recd["r_p"])
        if srce != led["srce_chk"] or recd["provenance"] != 0.0:
            why_l.append("SRCE_CHK_OR_PROVENANCE")
        mism = []
        for k, (ok, D, rel) in checks.items():
            row[f"{k}_rel"] = rel
            if ok is False:
                resolution = k.endswith("tau_eff") and D is not None and D <= RTOL * s_tau
                mism.append(f"{k}{'(resolution)' if resolution else ''}")
        for k in ("tau_eff", "beta", "c", "l", "Jbar"):
            row[f"{k}_record"] = recd[k]
            row[f"{k}_calc"] = calc[k]
        for k in ("tau_eff", "beta", "l", "Jbar", "J_c"):
            row[f"{k}_ledger"] = led[k]
        row["tau_raw_calc"] = tau_raw
        row["J_c_full_recompute"] = det_si_line_input_jc_status(False, "coarse")
        # physics comparisons (separate columns)
        nfl_l, nfl_u = float(fin[fl_of[int(r["lev_b"])]]), float(fin[fl_of[int(r["lev_a"])]])
        tau13 = det_si_line_input_tau(f_lu, lam_cm, t_exp, nfl_l, nfl_u, g_l, g_u)[0]
        binrow = [b for b in jgrid if float(b["nu_lo_Hz"]) <= nu < float(b["nu_hi_Hz"])]
        Jcoarse = float(binrow[0]["J_nu"]) if len(binrow) == 1 else None
        Bnu = det_si_line_input_planck(nu, T_e)
        row.update(tau_FL13=tau13, tau_FL13_over_production=TR.div(tau13, tau_raw), J_coarse_bin=Jcoarse,
                   B_nu_Te=Bnu, J_c_over_B=led["J_c"] / Bnu,
                   J_coarse_over_B=(Jcoarse / Bnu) if Jcoarse is not None else None,
                   dex_J_c_vs_coarse=TR.dex(led["J_c"], Jcoarse) if Jcoarse else None)
        defect = [m for m in mism if "(resolution)" not in m]
        if why_l:
            row.update(conclusion="미해결", unresolved=";".join(why_l))
            findings += [f"{lid}:{w}" for w in why_l]
        elif defect:
            row.update(conclusion="불일치(결함 후보)", mismatch=";".join(mism))
        else:
            row.update(conclusion="생산식·전달 일치; J_c 완전 재계산 미확보",
                       mismatch=";".join(mism) if mism else "")
        rows.append(row)
    if findings:
        gates["G-IDENTITY"].append("LINE_IDENTITY:" + ";".join(findings[:6]))
    if any(r.get("conclusion") == "불일치(결함 후보)" for r in rows) and gates["G-IDENTITY"]:
        gates["G-TABLE"].append("DEFECT_WITHOUT_VERIFIED_IDENTITY")
    all_gates = [f for v in gates.values() for f in v]
    defects = [r for r in rows if r.get("conclusion") == "불일치(결함 후보)"]
    if not all_gates and defects:
        code = 0
    elif not all_gates and all(r.get("J_c_full_recompute") == "computed" for r in rows):
        code = 1
    else:
        code = 2
    label = {0: "INPUT_DEFECT_FOUND", 1: "INPUT_CONSISTENT", 2: "INDETERMINATE"}[code]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "si10_line_input_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.10g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    with open(out / "si10_line_input_table.md", "x") as f:
        f.write(f"# Si shell 10 representative-line inputs (branch {code} {label})\n\n")
        f.write("Context (quoted from DET-SI-BB-GROUPS, not recomputed): " + json.dumps(CONTEXT) + "\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        for r in rows:
            f.write(f"## {r['group']} aux {r['aux']} ({r['lower_level_num']} -> {r['upper_level_num']})\n\n"
                    "| field | value |\n|---|---|\n")
            for k in fields:
                if k in r:
                    f.write(f"| {k} | {fmt(r[k])} |\n")
            f.write("\n")
    used = [tr / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_commit.csv", "si10_cevec.csv", "si10_ops.csv",
                             "si10_slmap.csv", "si10_vec.csv")]
    used += [run / "input/model/line_list.csv", run / "input/model/levels.csv", run / "input/model/config.json",
             run / "input/resolved_lumina.exports", run / "work/w5_lines.csv",
             pc / "physics_DET_iter0009.ion.csv", pc / "physics_DET_iter0009.manifest.json",
             pc / "physics_DET_iter0010.manifest.json", pc / "physics_DET_iter0010.spectral.csv",
             baseline / "evidence.json"]
    evidence = {"schema": "DET_SI_LINE_INPUT_ANALYSIS_V1", "prereg": PREREG, "argv": sys.argv,
                "adopted": {k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call")},
                "production_call": {k: pcall[k] for k in ("call", "bundle", "ne_iter", "ce_pass", "core_generation",
                                                           "rad_generation")},
                "population_source": {"vector": "call 1 PRE_FL", "equals_ce_old": "PRE_FL_VS_CE_OLD" not in
                                      gates["G-IDENTITY"], "ion_sums": sums, "material12_nlte_sums": ref9,
                                      "ion_sum_rel": {k: v[2] for k, v in sum_dev.items()},
                                      "source_path": "det_si_trace.h det_si_trace_call_begin records the pair's FL "
                                      "populations before assembly; lumina_plasma.c population_transaction_begin "
                                      "(about line 22020 of the trace source) copies the published material-12 "
                                      "populations for the generation-13 candidate; Si II/III is NLTE ion 0, the first "
                                      "pair; nlte_update_tau_sobolev and the fine producer read the same published "
                                      "nlte_level_populations"},
                "time_explosion_s": t_exp, "T_e": T_e, "branch": code, "label": label, "gates": gates,
                "conclusions": {str(r["aux"]): r.get("conclusion") for r in rows}, "context": CONTEXT,
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(p).resolve()): TR.sha(Path(p))
                         for p in (__file__, TR.__file__, EB.__file__, CF.__file__, GR.__file__)}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str) + "\n")
    print(f"BRANCH {code} {label} " + json.dumps({str(r["aux"]): r.get("conclusion") for r in rows},
                                                 ensure_ascii=False))
    if all_gates:
        print("GATE FINDINGS: " + "; ".join(all_gates))
        return 2
    return 0


# ------------------------------------------------------------------ DET-JC-DEFINITION (jcdef, jcdef-negative)
JCDEF_PREREG = "docs/PREREG_DET_JC_DEFINITION_2026-10-11.md"
JCDEF_SOURCE = {   # driver's reading of the sealed sources (prereg 3-4); data for the table, not a string-match verdict
    "operator": "production call 12 uses CMFGEN_NONOVERLAP_SOBOLEV (lumina_cmfgen.c about 11770: init_pass -> "
                "INIT_SHARED_GAUSSIAN, else CMFGEN_NONOVERLAP_SOBOLEV); sobolev_operator = line_net_parity && that "
                "operator (about 8535)",
    "direct_line": "excluded: both fine line-deposition loops require !sobolev_operator (lumina_cmfgen.c about 8804 "
                   "and 8883); the unmapped-eps fine folding also requires it (about 8598); fs.chi_line stays zero, "
                   "so chi_tot = chi_es + chi_abs (about 9044-9047) and the line emissivity eta is not deposited",
    "upstream": "mapped lines (the six lines included): no share — fine chi_es/chi_abs are interpolated from the "
                "bin state (about 8385-8399); the bin chi_es carries the line scattering share (chi_ln - chi_ln_th) "
                "(about 2962) with the per-line eps_phys branch (LUMINA_CMFGEN_LINE_EPS_PHYS=1; LINE_EPS and "
                "LINE_EPS_UV unset), and under the active DET-LTEUNMAP-EPS gate mapped lines take eps = 1 (fully "
                "thermal, no scattering share; about 2512-2532 and the helper at about 163); bin chi_abs = bf + ff "
                "only and the fine continuum emissivity w5_cont_eta = bf (Milne) + ff with the line term passed as 0 "
                "(about 2980-2987). Unmapped (other) lines: conditional — their share accumulated in "
                "g_chi_line_sc_unmapped (about 2530) is subtracted only when that side array exists with the matching "
                "size (fine_sub_sc, about 8384-8397); its allocation (about 2463) does not fail the run, and no "
                "record of production call 12 confirms fine_sub_sc = 1",
    "incident": "J_c = Gaussian profile mean of fs.J within +-4 Doppler widths (line_jbar_gaussian_discrete_shells, "
                "lumina_cmfgen.c about 9780) passed as continuum_j to line_net_sobolev_radiation (about 9810): "
                "c = beta J_c, l = eta (c t / nu)(1 - beta)/tau, Jbar = c + l (line_net_rate.c)",
    "cmfgen": "SOBJBAR_SIM (Sobolev path: cmfgen_sub.f SECTION='LINE' about 3504 -> SUB_SOB_LINE_V3 about 3638 -> "
              "sobjbar_sim.f): CHI and SOURCE = ZETA + RJ*THETA come from COMP_OPAC at the line frequency, which "
              "adds bound-free, free-free, two-photon, X-ray, scattering and optional additional opacity but no "
              "line opacity (comp_opac.f), and RJ from COMP_J_BLANK on that opacity; the per-ray AV is weighted "
              "by EXPONX(GAM) and integrated over angle, with the simultaneous lines' CHIL/ETAL entering the "
              "Sobolev optical depth and local term (cmfgen_sub.f about 3535)",
    "same": "확인된 공통점은 대표선 자신의 직접 침적·상류 산란 몫을 제외하고 국소 선 항을 별도로 더한다는 것이다. "
            "Lumina의 비대응 선 산란 몫 차감은 호출12에서 미확정이므로, 다른 선까지 모두 제외된다는 공통점은 "
            "확정하지 않는다.",
    "different": "CMFGEN integrates beta*AV per ray with an angle-dependent Sobolev optical depth; Lumina uses a "
                 "scalar beta times an angle-averaged J_c; Lumina's J_c is a Gaussian mean over +-4 Doppler widths "
                 "of the fine field; CMFGEN sums NUM_SIM_LINES simultaneous lines, Lumina uses one line; the "
                 "continuum transfer methods differ",
    "public_execution": "not established: the inputs do not show that the public toy06 model treated these lines "
                        "with SOBJBAR_SIM (its blanketed comoving-frame lines use a different path)"}


def det_si_jcdef_bin(rows, nu):
    """the comparison bin [nu_lo, nu_hi) holding nu, the last upper edge included; None outside or ambiguous."""
    hit = [r for r in rows if float(r["nu_lo_Hz"]) <= nu < float(r["nu_hi_Hz"])]
    if not hit and rows:
        last = max(rows, key=lambda r: float(r["nu_hi_Hz"]))
        if nu == float(last["nu_hi_Hz"]):
            hit = [last]
    return hit[0] if len(hit) == 1 else None


def det_si_jcdef_components(r):
    """chi_cont = es + bf + ff and q_bb = chi_bb/chi_total, with the component sum checked against chi_total."""
    try:
        es, bb, bf, ff, tot = (float(r[k]) for k in ("chi_es_cm1", "chi_bb_cm1", "chi_bf_cm1", "chi_ff_cm1",
                                                      "chi_total_cm1"))
    except (KeyError, ValueError):
        return None, "COMPONENT_MISSING"
    if not all(math.isfinite(v) for v in (es, bb, bf, ff, tot)):
        return None, "COMPONENT_NONFINITE"
    ok, _, rel = det_si_line_input_match(es + bb + bf + ff, tot)
    if not ok:
        return None, f"COMPONENT_SUM:{rel}"
    if not (tot > 0.0):
        return None, "TOTAL_NOT_POSITIVE"
    return {"chi_es": es, "chi_bb": bb, "chi_bf": bf, "chi_ff": ff, "chi_total": tot, "chi_cont": es + bf + ff,
            "q_bb": bb / tot}, None


def det_si_jcdef_link(base_rows, run_rows):
    """one baseline row per registered line, matched on (aux, call, op); returns (rows, findings)."""
    want = {lid: (grp, lo, up) for grp, lid, lo, up in LINES}
    out, why = {}, []
    for r in base_rows:
        lid = int(r["aux"])
        if lid not in want:
            why.append(f"UNREGISTERED_LINE:{lid}")
            continue
        if lid in out:
            why.append(f"DUPLICATE_LINE:{lid}")
            continue
        if (r["group"], int(r["lower_level_num"]), int(r["upper_level_num"])) != want[lid]:
            why.append(f"IDENTITY:{lid}")
            continue
        out[lid] = r
    why += [f"MISSING_LINE:{lid}" for lid in want if lid not in out]
    for lid, r in out.items():
        if (r.get("call"), r.get("op")) not in run_rows.get(lid, set()):
            why.append(f"STATEMENT_NOT_IN_RUN:{lid}")
    return out, why


def cmd_jcdef_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    base = [{"aux": str(lid), "group": g, "lower_level_num": str(lo), "upper_level_num": str(up), "call": "36",
             "op": str(100 + k)} for k, (g, lid, lo, up) in enumerate(LINES)]
    runs = {int(b["aux"]): {("36", b["op"])} for b in base}
    rows, why = det_si_jcdef_link(base, runs)
    expect(len(rows) == 6 and not why, "six registered rows link (positive control)")
    expect(any("DUPLICATE" in w for w in det_si_jcdef_link(base + base[:1], runs)[1]), "inject a duplicate row: refused")
    expect(any("UNREGISTERED" in w for w in det_si_jcdef_link(base + [dict(base[0], aux="5")], runs)[1]),
           "inject another line: refused")
    expect(any("STATEMENT_NOT_IN_RUN" in w for w in det_si_jcdef_link(base, {k: {("35", "1")} for k in runs})[1]),
           "inject a statement of another call (other shell/state): refused")
    spec = [{"nu_lo_Hz": "1.0e15", "nu_hi_Hz": "1.1e15", "J_nu": "1", "chi_es_cm1": "1", "chi_bb_cm1": "2",
             "chi_bf_cm1": "3", "chi_ff_cm1": "4", "chi_total_cm1": "10"},
            {"nu_lo_Hz": "1.1e15", "nu_hi_Hz": "1.2e15", "J_nu": "2", "chi_es_cm1": "1", "chi_bb_cm1": "1",
             "chi_bf_cm1": "1", "chi_ff_cm1": "1", "chi_total_cm1": "4"}]
    expect(det_si_jcdef_bin(spec, 1.05e15) is spec[0] and det_si_jcdef_bin(spec, 1.1e15) is spec[1] and
           det_si_jcdef_bin(spec, 1.2e15) is spec[1], "bin selection [lo, hi) with the last edge (positive control)")
    expect(det_si_jcdef_bin(spec, 1.05e15 / 1e15) is None, "inject a frequency in PHz instead of Hz: no bin")
    expect(det_si_jcdef_bin(spec, 1.3e15) is None, "inject a frequency outside the grid: no bin")
    c, w = det_si_jcdef_components(spec[0])
    expect(c is not None and c["q_bb"] == 0.2 and c["chi_cont"] == 8.0, "component ratio (positive control)")
    expect(det_si_jcdef_components(dict(spec[0], chi_bb_cm1="nan"))[0] is None, "inject a non-finite component: "
           "refused")
    expect(det_si_jcdef_components(dict(spec[0], chi_total_cm1="11"))[0] is None, "inject a wrong component sum: "
           "refused")
    expect(det_si_jcdef_components({"chi_es_cm1": "0", "chi_bb_cm1": "0", "chi_bf_cm1": "0", "chi_ff_cm1": "0",
                                    "chi_total_cm1": "0"})[0] is None, "inject a zero denominator: refused")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_JC_DEFINITION_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


def cmd_jcdef(run, baseline, out):
    run, baseline = Path(run), Path(baseline)
    out = TR.new_dir(out)
    gates = {k: [] for k in ("G-PATH", "G-MAP", "G-CLAIM", "G-RUN")}
    part, cvd = os.environ.get("SLURM_JOB_PARTITION"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if part != "a10" or cvd not in ("", "NoDevFiles"):
        gates["G-RUN"].append(f"NOT_A10_CPU:{part}:{cvd}")
    # ---- G-PATH: the consumed production branch (settings, operator and gate statistics of call 12)
    exports = (run / "input/resolved_lumina.exports").read_text()
    settings = {k: (f'{k}="{v}"' in exports) for k, v in (("LUMINA_CMF_LINERES_JBAR", "2"),
                                                          ("LUMINA_CMF_FINE_UNMAPPED_EPS", "1"),
                                                          ("LUMINA_CMF_FINE_CONT_NLTE_SOURCE", "1"),
                                                          ("LUMINA_CMF_FINE_VDOP", "1000000"),
                                                          ("LUMINA_CMFGEN_LINE_EPS_PHYS", "1"))}
    for k in ("LUMINA_CMF_FINE_UNMAPPED_EPS_FINE", "LUMINA_CMF_CONTONLY", "LUMINA_CMF_FINE_BF_OPAC",
              "LUMINA_CMFGEN_LINE_EPS=", "LUMINA_CMFGEN_LINE_EPS_UV"):
        settings[f"{k} absent"] = k not in exports
    if not all(settings.values()):
        gates["G-PATH"].append("SETTINGS:" + ",".join(k for k, v in settings.items() if not v))
    op_line = asm_line = None
    with open(run / "stderr.log", errors="replace") as f:
        for ln in f:
            if ln.startswith("[cmf_fine][W5-CONT] call=12 operator="):
                op_line = ln.strip()
            elif ln.startswith(f"[cmfgen_assemble][UNMAPPED-EPS] pass=12 shell={SHELL} "):
                asm_line = ln.strip()
    if not (op_line and "operator=CMFGEN_NONOVERLAP_SOBOLEV rad_generation=12 material_generation=12 "
                        "assemble_pass=12" in op_line):
        gates["G-PATH"].append("PRODUCTION_OPERATOR")
    n_mapped = None
    if asm_line:
        kv = dict(t.split("=", 1) for t in asm_line.split()[1:] if "=" in t)
        n_mapped = int(kv.get("n_mapped", "-1"))
    if not (n_mapped and n_mapped > 0):
        gates["G-PATH"].append("MAPPED_GATE_STATISTICS")
    # ---- G-MAP: six lines, bins, components
    base_rows = TR.read_trace(baseline / "si10_line_input_table.csv")
    run_rows = {}
    for r in TR.read_trace(run / "si_trace/si10_ops.csv"):
        if r["call"] == str(ADOPTED["call"]) and r["site"] == "1" and CF.det_si_bb_mode(r) == 8:
            run_rows.setdefault(int(r["aux"]), set()).add((r["call"], r["op"]))
    linked, why = det_si_jcdef_link(base_rows, run_rows)
    if why:
        gates["G-MAP"].append("LINK:" + ";".join(why[:6]))
    spec = [r for r in TR.read_trace(run / "work/physics_comparison/physics_DET_iter0010.spectral.csv")
            if int(r["shell_id"]) == SHELL]
    rows = []
    for grp, lid, lo, up in LINES:
        b = linked.get(lid)
        row = {"group": grp, "aux": lid, "lower_level_num": lo, "upper_level_num": up}
        if b is None:
            row.update(conclusion="미확정", reason="row not linked")
            rows.append(row)
            continue
        nu = float(b["nu"])
        bins = det_si_jcdef_bin(spec, nu)
        comp, cwhy = det_si_jcdef_components(bins) if bins is not None else (None, "NO_BIN")
        if comp is None:
            gates["G-MAP"].append(f"BIN:{lid}:{cwhy}")
        J_c, J_coarse = float(b["J_c_ledger"]), float(b["J_coarse_bin"])
        if bins is not None and not det_si_line_input_match(float(bins["J_nu"]), J_coarse)[0]:
            gates["G-MAP"].append(f"COARSE_J_NOT_PREVIOUS:{lid}")
        row.update(call=b["call"], op=b["op"], nu=nu, production="call 12 / radiation 12 / material 12",
                   consumption="trace call 36", lumina_operator="CMFGEN_NONOVERLAP_SOBOLEV",
                   self_line_direct="excluded",
                   upstream_residual="self (mapped) line: none (eps = 1, independent of the subtraction); "
                                     "unmapped share: subtraction conditional on fine_sub_sc, unrecorded for call 12",
                   other_lines="mapped lines excluded; unmapped scattering share excluded only if the subtraction "
                               "ran (unconfirmed)", cmfgen_incident="continuum AV (COMP_OPAC + "
                   "COMP_J_BLANK, no line opacity)", cmfgen_same=JCDEF_SOURCE["same"],
                   cmfgen_different=JCDEF_SOURCE["different"], public_execution="not established",
                   J_c=J_c, J_coarse=J_coarse, dex_previous=float(b["dex_J_c_vs_coarse"]),
                   bin_nu_lo=float(bins["nu_lo_Hz"]) if bins else None,
                   bin_nu_hi=float(bins["nu_hi_Hz"]) if bins else None)
        if comp:
            row.update({k: comp[k] for k in ("chi_es", "chi_bb", "chi_bf", "chi_ff", "chi_total", "chi_cont",
                                             "q_bb")})
        row.update(delta_def="not identified (no matched field pair; no new transfer)",
                   magnitude="미식별", direct_mechanism="직접 중복 기전 배제",
                   relation_previous="q_bb is a component ratio of the comparison bin; it does not attribute the "
                                     "previous J_c vs coarse J_nu difference, and that dex difference alone does "
                                     "not show self-line double counting",
                   comparison_context="comparison dump: population 13 / radiation 12 / opacity 23 (manifest) vs J_c "
                                      "production material 12 / radiation 12; J_nu is an integral-preserving "
                                      "piecewise-constant regrid and the opacity components come from the opacity "
                                      "view; not a matched field pair",
                   double_counting="reserved: direct deposition excluded and the mapped self-line upstream share is "
                                   "zero, but the prereg 6 refutation is not completed on recorded evidence "
                                   "(unmapped-share subtraction unconfirmed for call 12)",
                   conclusion="직접 선 침적 배제; 비대응 선 상류 산란 차감 미확정 — 정의 판별 목표 미달")
        rows.append(row)
    for r in rows:
        if r.get("magnitude") and r["magnitude"] not in ("식별", "미식별", "직접 중복 기전 배제"):
            gates["G-CLAIM"].append(f"MAGNITUDE_LABEL:{r['aux']}")
    all_gates = [f for v in gates.values() for f in v]
    complete = (all(r.get("self_line_direct") == "excluded" for r in rows) and len(rows) == 6 and
                not any("unconfirmed" in r.get("other_lines", "") for r in rows))
    code = 2 if (all_gates or not complete) else 1
    label = {0: "DEFINITION_DIFFERENCE", 1: "INCIDENT_SEPARATION_CONFIRMED", 2: "INDETERMINATE"}[code]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "jc_definition_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    with open(out / "jc_definition_table.md", "x") as f:
        f.write(f"# J_c definition, Si shell 10 representative lines (branch {code} {label})\n\n")
        f.write("Context (quoted, not recomputed): " + json.dumps(CONTEXT) + "\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        cols = ["group", "aux", "lower_level_num", "upper_level_num", "call", "op", "nu", "J_c", "J_coarse",
                "dex_previous", "chi_es", "chi_bb", "chi_bf", "chi_ff", "chi_total", "q_bb", "self_line_direct",
                "upstream_residual", "magnitude", "direct_mechanism", "conclusion"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
        f.write("\nComparison context: " + (rows[0].get("comparison_context", "") if rows else "") + "\n")
        f.write("\n## Source determinations (common to the six rows)\n\n")
        for k, v in JCDEF_SOURCE.items():
            f.write(f"- **{k}:** {v}\n")
    S = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src")
    C = Path("/gpfs/kjhan/cmfgen_src/cur_cmf")
    used = [run / "input/resolved_lumina.exports", run / "stderr.log", run / "si_trace/si10_ops.csv",
            run / "work/physics_comparison/physics_DET_iter0010.spectral.csv", baseline / "si10_line_input_table.csv",
            S / "lumina_cmfgen.c", S / "line_jbar.c", S / "line_net_rate.c", C / "new_main/cmfgen_sub.f",
            C / "new_main/mod_subs/comp_opac.f", C / "new_main/mod_subs/comp_j_blank.f",
            C / "new_main/mod_subs/sub_sob_line_v3.f", C / "subs/sobjbar_sim.f"]
    evidence = {"schema": "DET_JC_DEFINITION_V1", "prereg": JCDEF_PREREG, "argv": sys.argv, "branch": code,
                "label": label, "gates": gates, "settings": settings, "production_operator_line": op_line,
                "assemble_gate_line": asm_line, "n_mapped_shell10_pass12": n_mapped, "source": JCDEF_SOURCE,
                "context": CONTEXT,
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(__file__).resolve()): TR.sha(Path(__file__))}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str, ensure_ascii=False) + "\n")
    print(f"BRANCH {code} {label} q_bb=" + json.dumps({str(r["aux"]): r.get("q_bb") for r in rows}))
    if all_gates:
        print("GATE FINDINGS: " + "; ".join(all_gates))
        return 2
    return 0


# ------------------------------------------------------------------ DET-JC-COARSE-CF (jccf, jccf-negative)
JCCF_PREREG = "docs/PREREG_DET_JC_COARSE_CF_2026-10-11.md"
JCCF_STATES = ("F0", "C-ALL", "C-G1", "C-G2", "C-G3", "C-G4", "C-REP6")
JCCF_SCOPE = {"F0": "no substitution (reference steady state)", "C-ALL": "all Si mode-8 statements of the adopted call",
              "C-G1": GR.RULES["G1"], "C-G2": GR.RULES["G2"], "C-G3": GR.RULES["G3"], "C-G4": GR.RULES["G4"],
              "C-REP6": "the six registered representative lines jointly (G1 760321, 632403, 757019; "
                        "G2 1676767, 1353542, 1679119)"}
JCCF_GROUP_COUNTS = {"G1": 62, "G2": 151, "G3": 956, "G4": 1418}
JCCF_R0, JCCF_FINAL_FL = 3.5597568125822305, 3.563683
JCCF_PUBLIC = {"si_II": 0.0092499, "si_III": 0.97769}
JCCF_GENERATIONS = {"radiation_generation": 12, "population_generation": 13, "opacity_generation": 23,
                    "emissivity_generation": 23}
JCCF_GRID_KIND = {"frequency_coordinate": "HZ", "frequency_regrid": "INTEGRAL_PRESERVING_PIECEWISE_CONSTANT",
                  "frame": "SHELL_COMOVING"}
JCCF_RTOL, JCCF_REF_DEX, JCCF_DEX = 1e-12, 0.001, 0.01
JCCF_VDOP, JCCF_NDOPPLER = 1.0e6, 4.0                # LUMINA_CMF_FINE_VDOP (checked in the run settings), +-4 widths
JCCF_ION = {"Si II": "1", "Si III": "2"}
JCCF_EFF = ("이동 분리", "이동 미분리")
JCCF_DIST = ("공개 접근", "이격", "공개 거리 변화 미분리")
JCCF_NOTE = ("the stored coarse bin J_nu (integral-preserving piecewise-constant regrid in Hz; may contain the line "
             "forest; not line-subtracted) replaces J_c in c = beta J_c of the mode-8 bracket with beta, A, tau_eff, l "
             "and every other rate fixed; the coarse field is not the CMFGEN SOBJBAR_SIM incident field, and the "
             "result is a conditional sensitivity of the fixed rate system, not a repair or a production promotion")


def det_si_jccf_close(a, b):
    """prereg 6: relative difference within 1e-12, exact zeros only equal to exact zeros."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    M = max(abs(a), abs(b))
    return abs(a - b) / M <= JCCF_RTOL if M > 0 else (a == 0.0 and b == 0.0)


def det_si_jccf_grid(rows, shell, manifest, rad_generation):
    """the shell's comparison bins with the registered generations and a well-formed Hz grid; (bins, findings)."""
    why = [f"GENERATION:{k}={manifest.get(k)}" for k, v in JCCF_GENERATIONS.items() if manifest.get(k) != v]
    if rad_generation != JCCF_GENERATIONS["radiation_generation"]:
        why.append(f"ADOPTED_RAD_GENERATION:{rad_generation}")
    why += [f"GRID_KIND:{k}" for k, v in JCCF_GRID_KIND.items() if manifest.get(k) != v]
    bins = sorted((r for r in rows if int(r["shell_id"]) == shell), key=lambda r: float(r["nu_lo_Hz"]))
    if len(bins) != manifest.get("n_bins"):
        why.append(f"BIN_COUNT:{len(bins)}")
    why += [f"INVERTED:{r['bin_id']}" for r in bins if not float(r["nu_lo_Hz"]) < float(r["nu_hi_Hz"])]
    why += [f"OVERLAP:{a['bin_id']}" for a, b in zip(bins, bins[1:]) if float(b["nu_lo_Hz"]) < float(a["nu_hi_Hz"])]
    return bins, why


def det_si_jccf_link(r, x, ion, up, lo, atom_rows, ledger_rows, bins):
    """one mode-8 statement -> (substitution record, None), or (None, reason) when the substitution is undefined
    (prereg 5): line / ion / levels / production ledger linkage, the centre bin [nu_lo, nu_hi) of the atomic
    frequency, J_c = c / beta, q = J_coarse / J_c, (q r_a, q r_s, r_p)."""
    if ion not in JCCF_ION or up is None or lo is None:
        return None, "ION_OR_LEVEL"
    if len(atom_rows) != 1:
        return None, f"ATOMIC_ROWS:{len(atom_rows)}"
    a = atom_rows[0]
    if (a["atomic_number"], a["ion_number"], a["line_id"]) != ("14", JCCF_ION[ion], r["aux"]) or \
            (int(a["level_number_lower"]), int(a["level_number_upper"])) != (lo["level_num"], up["level_num"]):
        return None, "LINE_IDENTITY"
    if len(ledger_rows) != 1:
        return None, f"LEDGER_ROWS:{len(ledger_rows)}"
    L, nu = ledger_rows[0], float(a["nu"])
    if L[5] != r["aux"] or L[6:8] != ["14", JCCF_ION[ion]] or L[17] != "OK" or float(L[8]) != nu:
        return None, "LEDGER_IDENTITY"
    beta, c = x[9], x[10]
    if not (math.isfinite(beta) and beta > 0.0):
        return None, "BETA"
    if not (math.isfinite(c) and c >= 0.0):
        return None, "C_NEGATIVE_OR_NONFINITE"
    if c == 0.0:
        return None, "C_ZERO"
    J_c = c / beta
    if not (math.isfinite(J_c) and J_c > 0.0):
        return None, "J_C_REPRESENTATION"
    try:
        led_jc, led_beta = float(L[9]), float(L[11])
    except ValueError:
        return None, "LEDGER_VALUES"
    if not (det_si_jccf_close(led_jc, J_c) and det_si_jccf_close(led_beta, beta)):
        return None, "LEDGER_VS_RECORD"
    b = det_si_jcdef_bin(bins, nu)
    if b is None:
        return None, "NO_BIN"
    Jco = float(b["J_nu"])
    if not (math.isfinite(Jco) and Jco >= 0.0):
        return None, "COARSE_FIELD"
    q = Jco / J_c
    c_new, ra, rs = beta * Jco, q * x[0], q * x[1]
    if not all(math.isfinite(v) for v in (q, c_new, ra, rs)):
        return None, "NONFINITE_SUBSTITUTION"
    # only an exact coarse zero is a valid zero substitution; a positive field lost to zero is a representation failure
    if Jco > 0.0 and (q == 0.0 or c_new == 0.0 or (x[0] != 0.0 and ra == 0.0) or (x[1] != 0.0 and rs == 0.0)):
        return None, "REPRESENTATION"
    w = JCCF_NDOPPLER * JCCF_VDOP / C_LIGHT
    lo_b, hi_b = float(b["nu_lo_Hz"]), float(b["nu_hi_Hz"])
    return {"call": r["call"], "op": r["op"], "aux": int(r["aux"]), "ion": ion, "nu": nu, "bin_id": int(b["bin_id"]),
            "bin_nu_lo": lo_b, "bin_nu_hi": hi_b, "tau_eff": x[12], "beta": beta, "c": c, "J_c": J_c,
            "J_coarse": Jco, "q": q, "c_new": c_new, "r_a": x[0], "r_s": x[1], "r_p": x[2], "r_a_new": ra,
            "r_s_new": rs, "profile_crosses_bin": nu * (1.0 - w) < lo_b or nu * (1.0 + w) > hi_b}, None


def det_si_jccf_statements(m8, n_lo, info, atom, ledger, bins, grid_why):
    """every mode-8 statement -> substitution record or undefined reason: a duplicate identity, the recorded
    statement conditions, a grid finding (all statements), then the per-statement linkage."""
    seen, recs, undefined = {}, {}, {}
    for r, _ in m8:
        seen[(r["call"], r["op"])] = seen.get((r["call"], r["op"]), 0) + 1
    for r, x in m8:
        k = (r["call"], r["op"])
        cond = CF.det_si_bb_check(r, x)
        if seen[k] > 1:
            undefined[k] = "DUPLICATE_STATEMENT"
        elif cond:
            undefined[k] = f"STATEMENT_CONDITIONS:{cond}"
        elif grid_why:
            undefined[k] = "GRID"
        else:
            ion, lid = CF.det_si_bb_ion(r, n_lo), int(r["aux"])
            e, reason = det_si_jccf_link(r, x, ion, info.get(int(r["lev_a"])), info.get(int(r["lev_b"])),
                                         atom.get((JCCF_ION.get(ion), lid), []), ledger.get(lid, []), bins)
            if e is None:
                undefined[k] = reason
            else:
                recs[k] = e
    return recs, undefined


def det_si_jccf_scope(state, m8, assign, rep_keys, rep_why, undefined, counts=JCCF_GROUP_COUNTS):
    """registered targets of one range, their undefined statements, and the range-level reasons that also stop the
    range (membership not resolved, registered counts, the representative six)."""
    if state == "F0":
        return set(), {}, []
    keys = [(r["call"], r["op"]) for r, _ in m8]
    if state == "C-ALL":
        tg = set(keys)
        rng = [] if len(keys) == sum(counts.values()) else [f"REGISTERED_COUNT:{len(keys)}"]
        return tg, {k: undefined.get(k, "MEMBERSHIP") for k in tg if k in undefined or k not in assign}, rng
    if state == "C-REP6":
        tg = set(rep_keys)
        return tg, {k: undefined[k] for k in tg if k in undefined}, [f"REP6:{w}" for w in rep_why]
    g = state[2:]
    tg = {k for k, v in assign.items() if v == g}
    rng = [] if len(tg) == counts[g] else [f"REGISTERED_COUNT:{len(tg)}"]
    missing = len({k for k in keys if k not in assign})
    if missing:
        rng.append(f"MEMBERSHIP_UNRESOLVED:{missing}")
    return tg, {k: undefined[k] for k in tg if k in undefined}, rng


def det_si_jccf_rep6(m8, assign, info):
    """the six registered representative statements by line, group and level numbers; (keys, findings)."""
    keys, why = [], []
    for grp, lid, lo_ln, up_ln in LINES:
        hit = [r for r, _ in m8 if int(r["aux"]) == lid]
        if len(hit) != 1:
            why.append(f"REP6_ROWS:{lid}:{len(hit)}")
            continue
        r = hit[0]
        k = (r["call"], r["op"])
        up, lo = info.get(int(r["lev_a"])), info.get(int(r["lev_b"]))
        if assign.get(k) != grp or up is None or lo is None or (lo["level_num"], up["level_num"]) != (lo_ln, up_ln):
            why.append(f"REP6_IDENTITY:{lid}")
            continue
        keys.append(k)
    return keys, why


def det_si_jccf_callback(recs, targets):
    """the registered substitution on the target statements only; every state starts from F0."""
    def cb(r, x):
        k = (r["call"], r["op"])
        return (recs[k]["r_a_new"], recs[k]["r_s_new"], x[2]) if k in targets else None
    return cb


def det_si_jccf_check(fn, stmts, targets, recs):
    """outputs only: target statements carry (q r_a, q r_s, r_p) with q = J_coarse / (c / beta) taken from the
    record, every other statement is unchanged (None); returns the offending ops."""
    bad = []
    for r, x in stmts:
        k, out = (r["call"], r["op"]), fn(r, x)
        if k not in targets:
            if out is not None:
                bad.append(r["op"])
            continue
        e = recs.get(k)
        if out is None or e is None:
            bad.append(r["op"])
            continue
        q = e["J_coarse"] / (x[10] / x[9])
        if not (out[2] == x[2] and det_si_jccf_close(out[0], q * x[0]) and det_si_jccf_close(out[1], q * x[1])):
            bad.append(r["op"])
    return bad


def det_si_jccf_qstats(qs):
    """equal weight per statement; the median of an even sample is the mean of the two middle values."""
    if not qs:
        return {}
    s, n = sorted(qs), len(qs)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"q_min": s[0], "q_median": med, "q_max": s[-1], "n_q_eq0": sum(1 for v in s if v == 0.0),
            "n_q_0_1": sum(1 for v in s if 0.0 < v < 1.0), "n_q_eq1": sum(1 for v in s if v == 1.0),
            "n_q_gt1": sum(1 for v in s if v > 1.0)}


def det_si_jccf_classify(delta, G):
    """prereg 7: boundary values belong to the unresolved class."""
    eff = JCCF_EFF[0] if abs(delta) > JCCF_DEX else JCCF_EFF[1]
    dist = JCCF_DIST[0] if G > JCCF_DEX else JCCF_DIST[1] if G < -JCCF_DEX else JCCF_DIST[2]
    return f"{eff}; {dist}"


def det_si_jccf_branch(valid_all, G_all):
    """0 RESOLVED_APPROACH / 1 NO_RESOLVED_APPROACH need every registered state valid; otherwise 2 INDETERMINATE."""
    if not valid_all or G_all is None or not math.isfinite(G_all):
        return 2
    return 0 if G_all > JCCF_DEX else 1


def det_si_jccf_claim_check(rows):
    """a state row with undefined statements or a range-level reason carries no ratio; labels stay in the registered
    vocabulary."""
    bad = []
    for r in rows:
        if r.get("kind") != "state":
            continue
        if (r.get("undefined") or r.get("range_reasons")) and r.get("III_II") is not None:
            bad.append(f"PARTIAL_AS_FULL:{r['state']}")
        c = r.get("cls")
        p = c.split("; ") if isinstance(c, str) else []
        if not (c in ("기준", "미산출") or (len(p) == 2 and p[0] in JCCF_EFF and p[1] in JCCF_DIST)):
            bad.append(f"LABEL:{r['state']}")
    return bad


def det_si_jccf_toy():
    """the DET-SI-BB-GROUPS toy (one mode-8 statement per group) with line ids, atomic rows, ledger rows and a
    four-bin coarse grid: statement 2 sits on the interior edge 2e15 (higher bin, J_nu exactly 0) and statement 4
    on the last upper edge 3e15."""
    t = GR.det_si_bb_groups_toy()
    info = GR.det_si_bb_groups_fl_info(t["sm"])
    nus = {"1": 1.5e15, "2": 2.0e15, "3": 2.6e15, "4": 3.0e15}
    ops = [dict(r, aux=str(10 + int(r["op"]))) if r["site"] == "1" else dict(r, aux="0") for r in t["ops"]]
    stmts = [(r, [TR.num(r[f"x{k}"]) for k in range(1, 15)]) for r in ops if r["site"] == "1"]
    atom, ledger = {}, {}
    for r, x in stmts:
        ion = CF.det_si_bb_ion(r, t["n_lo"])
        up, lo = info[int(r["lev_a"])], info[int(r["lev_b"])]
        a = {"atomic_number": "14", "ion_number": JCCF_ION[ion], "level_number_lower": str(lo["level_num"]),
             "level_number_upper": str(up["level_num"]), "line_id": r["aux"], "nu": repr(nus[r["op"]])}
        atom[r["op"]] = [a]
        ledger[r["op"]] = [["12", "CMFGEN_NONOVERLAP_SOBOLEV", "12", "12", "10", r["aux"], "14", JCCF_ION[ion],
                            repr(nus[r["op"]]), repr(x[10] / x[9]), "0", repr(x[9]), "0", "0", "0", repr(x[12]), "0",
                            "OK"]]
    spec = [{"shell_id": "10", "bin_id": str(i), "nu_lo_Hz": repr(a), "nu_hi_Hz": repr(b), "J_nu": repr(j)}
            for i, (a, b, j) in enumerate(((1.0e15, 2.0e15, 3e-5), (2.0e15, 2.4e15, 0.0), (2.4e15, 2.8e15, 2e-6),
                                           (2.8e15, 3.0e15, 5e-6)))]
    manifest = dict(JCCF_GENERATIONS, **JCCF_GRID_KIND, n_bins=4)
    return dict(t, ops=ops, stmts=stmts, info=info, atom=atom, ledger=ledger, spec=spec, manifest=manifest)


def cmd_jccf_negative():
    results = []

    def expect(ok, what):
        results.append((bool(ok), what))
        print(f"[NEG] {'PASS' if ok else 'FAIL'} {what}")
    t = det_si_jccf_toy()
    N, gh, n_lo, sm, ops, stmts, info = t["N"], t["gh"], t["n_lo"], t["sm"], t["ops"], t["stmts"], t["info"]
    bins, why = det_si_jccf_grid(t["spec"], 10, t["manifest"], 12)

    def link(r, x, atom=None, ledger=None, bins_=None, up=None, lo=None):
        return det_si_jccf_link(r, x, CF.det_si_bb_ion(r, n_lo), up or info[int(r["lev_a"])],
                                lo or info[int(r["lev_b"])], t["atom"][r["op"]] if atom is None else atom,
                                t["ledger"][r["op"]] if ledger is None else ledger, bins if bins_ is None else bins_)
    atom_map = {(v[0]["ion_number"], int(v[0]["line_id"])): v for v in t["atom"].values()}
    led_map = {int(v[0][5]): v for v in t["ledger"].values()}
    toy_counts = {"G1": 1, "G2": 1, "G3": 1, "G4": 1}

    def statements(m8, grid_why=()):
        return det_si_jccf_statements(m8, n_lo, info, atom_map, led_map, bins, list(grid_why))
    recs, undef = statements(stmts)
    want = {"1": (0, 3.0), "2": (1, 0.0), "3": (2, 0.2), "4": (3, 5.0)}
    expect(not why and not undef and len(recs) == 4 and all(
        recs[("1", k)]["bin_id"] == b and det_si_jccf_close(recs[("1", k)]["q"], q) for k, (b, q) in want.items()),
           "toy linkage, bins and q (positive control; interior edge -> higher bin, last edge included, exact coarse "
           "zero accepted as q = 0)")
    assign, wa = GR.det_si_bb_groups_assign(stmts, info, n_lo)
    A0, u0, ch0, _ = CF.det_si_bb_build(ops, sm, N, gh)
    ok = not wa and not u0
    for s in ("C-ALL", "C-G1", "C-G2", "C-G3", "C-G4"):
        tg, und_s, rng = det_si_jccf_scope(s, stmts, assign, [], [], undef, toy_counts)
        cb = det_si_jccf_callback(recs, tg)
        A, u, ch, _ = CF.det_si_bb_build(ops, sm, N, gh, cb)
        ok = ok and tg and not und_s and not rng and not u and not det_si_jccf_check(cb, stmts, tg, recs) and \
            not CF.det_si_bb_invariance(ch0, ch, set(GR.MODE8)) and CF.det_si_bb_solve(A, 10.0)[0] is not None
    expect(ok, "toy per-state operators valid, prescription and invariance hold (positive control)")

    # 1. another shell / radiation generation, overlap and inversion of the grid
    expect(any("GENERATION:radiation" in w for w in det_si_jccf_grid(t["spec"], 10, dict(t["manifest"],
           radiation_generation=11), 12)[1]), "inject a comparison dump of another radiation generation: refused")
    expect(any("ADOPTED_RAD" in w for w in det_si_jccf_grid(t["spec"], 10, t["manifest"], 13)[1]),
           "inject an adopted call of another radiation generation: refused")
    other = [dict(b, shell_id="9") for b in t["spec"]]
    expect(any("BIN_COUNT" in w for w in det_si_jccf_grid(other, 10, t["manifest"], 12)[1]),
           "inject another shell's bins: refused")
    ov = [dict(b, nu_hi_Hz="2.1e15") if b["bin_id"] == "0" else b for b in t["spec"]]
    expect(any("OVERLAP" in w for w in det_si_jccf_grid(ov, 10, t["manifest"], 12)[1]), "inject overlapping bins: "
           "refused")
    inv = [dict(b, nu_lo_Hz="2.9e15", nu_hi_Hz="2.8e15") if b["bin_id"] == "3" else b for b in t["spec"]]
    expect(any("INVERTED" in w for w in det_si_jccf_grid(inv, 10, t["manifest"], 12)[1]), "inject an inverted bin: "
           "refused")
    r1, x1 = stmts[0]
    ovb = det_si_jccf_grid(ov, 10, t["manifest"], 12)[0]
    expect(link(stmts[1][0], stmts[1][1], bins_=ovb)[1] == "NO_BIN", "overlapping bins at a line centre: no bin "
           "(ambiguous)")
    # 2. line, level, duplicate and missing linkage
    a1 = t["atom"]["1"][0]
    expect(link(r1, x1, atom=[dict(a1, line_id="99")])[1] == "LINE_IDENTITY", "inject another line's atomic row: "
           "refused")
    expect(link(r1, x1, atom=[dict(a1, level_number_lower=a1["level_number_upper"],
                                   level_number_upper=a1["level_number_lower"])])[1] == "LINE_IDENTITY",
           "inject swapped lower/upper level numbers: refused")
    expect(link(r1, x1, up=info[3])[1] == "LINE_IDENTITY", "inject another upper level: refused")
    expect(link(r1, x1, atom=[a1, a1])[1] == "ATOMIC_ROWS:2", "inject a duplicate atomic row: refused")
    expect(link(r1, x1, ledger=[])[1] == "LEDGER_ROWS:0", "inject a missing ledger row: refused")
    L1 = t["ledger"]["1"][0]
    expect(link(r1, x1, ledger=[L1, L1])[1] == "LEDGER_ROWS:2", "inject a duplicate ledger row: refused")
    expect(link(r1, x1, ledger=[L1[:7] + ["2"] + L1[8:]])[1] == "LEDGER_IDENTITY", "inject a ledger row of the other "
           "ion: refused")
    expect(link(r1, x1, ledger=[L1[:9] + [repr(2.0 * float(L1[9]))] + L1[10:]])[1] == "LEDGER_VS_RECORD",
           "inject a ledger J_c that is not the record's c / beta: refused")
    expect(link(r1, x1, atom=[dict(a1, nu=repr(1.5))])[1] == "LEDGER_IDENTITY", "inject a frequency in PHz instead "
           "of Hz (ledger mismatch): refused")
    out_atom = [dict(a1, nu=repr(4.0e15))]
    out_led = [L1[:8] + [repr(4.0e15)] + L1[9:]]
    expect(link(r1, x1, atom=out_atom, ledger=out_led)[1] == "NO_BIN", "inject a frequency outside the grid: no bin")
    keys6 = []
    m6, inf6, as6 = [], {}, {}
    for i, (grp, lid, lo_ln, up_ln) in enumerate(LINES):
        r = {"call": "36", "op": str(i), "aux": str(lid), "lev_a": str(2 * i + 1), "lev_b": str(2 * i)}
        m6.append((r, None))
        inf6[2 * i], inf6[2 * i + 1] = {"level_num": lo_ln}, {"level_num": up_ln}
        as6[("36", str(i))] = grp
        keys6.append(("36", str(i)))
    k6, w6 = det_si_jccf_rep6(m6, as6, inf6)
    expect(k6 == keys6 and not w6, "six representative statements resolve (positive control)")
    expect(any("REP6_ROWS" in w for w in det_si_jccf_rep6(m6 + [(dict(m6[0][0], op="99"), None)], as6, inf6)[1]),
           "inject a duplicate representative statement: refused")
    expect(any("REP6_ROWS" in w for w in det_si_jccf_rep6(m6[1:], as6, inf6)[1]), "inject a missing representative "
           "statement: refused")
    expect(any("REP6_IDENTITY" in w for w in det_si_jccf_rep6(m6, {**as6, ("36", "0"): "G3"}, inf6)[1]),
           "inject a representative statement in another group: refused")
    # 3. c = 0, beta <= 0, non-finite or negative field
    expect(link(r1, x1[:10] + [0.0] + x1[11:])[1] == "C_ZERO", "inject c = 0: undefined (no ratio substitution)")
    expect(link(r1, x1[:9] + [0.0] + x1[10:])[1] == "BETA", "inject beta = 0: undefined")
    expect(link(r1, x1[:9] + [-0.5] + x1[10:])[1] == "BETA", "inject beta < 0: undefined")
    expect(link(r1, x1, bins_=[dict(b, J_nu="nan") for b in bins])[1] == "COARSE_FIELD", "inject a non-finite coarse "
           "field: undefined")
    expect(link(r1, x1, bins_=[dict(b, J_nu="-1e-6") for b in bins])[1] == "COARSE_FIELD", "inject a negative coarse "
           "field: undefined")
    tiny = [dict(b, J_nu="5e-324") for b in bins]
    x10 = x1[:10] + [x1[9] * 10.0] + x1[11:]
    L10 = L1[:9] + [repr(x10[10] / x10[9])] + L1[10:]
    expect(link(r1, x10, ledger=[L10], bins_=tiny)[1] == "REPRESENTATION", "inject a positive coarse field whose ratio "
           "q underflows to 0: undefined (not a valid zero substitution)")
    expect(link(r1, x1, bins_=tiny)[1] == "REPRESENTATION", "inject a positive coarse field whose c' = beta J_coarse "
           "underflows to 0: undefined")
    expect(link(r1, x1[:9] + [3.0, 5e-324] + x1[11:])[1] == "J_C_REPRESENTATION",
           "inject c / beta lost to 0: undefined before the division (no exception)")
    # 3b. linkage failures stop the registered range (prereg 5), not only the global gate
    frame = det_si_jccf_grid(t["spec"], 10, dict(t["manifest"], frame="OBSERVER"), 12)[1]
    expect(any("GRID_KIND:frame" in w for w in frame), "inject another frame of the comparison grid: refused")
    rg, ug = statements(stmts, ["OVERLAP:0"])
    expect(not rg and set(ug.values()) == {"GRID"} and len(ug) == 4, "inject a grid finding: every statement undefined")
    rd, ud = statements(stmts + stmts[:1])
    tgd, und_d, _ = det_si_jccf_scope("C-ALL", stmts + stmts[:1], assign, [], [], ud, toy_counts)
    expect(ud.get(("1", "1")) == "DUPLICATE_STATEMENT" and ("1", "1") in und_d, "inject a duplicate statement "
           "identity: undefined, C-ALL not produced")
    xb = x1[:7] + [2.0 * x1[7]] + x1[8:]
    rc, uc = statements([(r1, xb)] + stmts[1:])
    expect(str(uc.get(("1", "1"))).startswith("STATEMENT_CONDITIONS"), "inject a statement violating the recorded "
           "bracket (Jbar != c + l): undefined")
    part = {k: v for k, v in assign.items() if k != ("1", "3")}
    tga, und_a, _ = det_si_jccf_scope("C-ALL", stmts, part, [], [], undef, toy_counts)
    _, und_g, rng_g = det_si_jccf_scope("C-G1", stmts, part, [], [], undef, toy_counts)
    expect(und_a.get(("1", "3")) == "MEMBERSHIP" and not und_g and any("MEMBERSHIP_UNRESOLVED" in w for w in rng_g),
           "inject an unassigned statement: C-ALL undefined and every group range stopped")
    expect(bool(det_si_jccf_scope("C-REP6", stmts, assign, [("1", "1")], ["REP6_ROWS:760321:0"], undef)[2]),
           "inject an unresolved representative line: C-REP6 stopped (not computed on the rest)")
    expect(any("REGISTERED_COUNT" in w for w in det_si_jccf_scope("C-G1", stmts, assign, [], [], undef,
                                                                     dict(toy_counts, G1=2))[2]),
           "inject a group count other than the registered one: range stopped")
    # 4. spontaneous change, beta twice, one pump rate, non-target change, accumulated group
    tg1 = det_si_jccf_scope("C-G1", stmts, assign, [], [], undef, toy_counts)[0]
    tg2 = det_si_jccf_scope("C-G2", stmts, assign, [], [], undef, toy_counts)[0]
    good = det_si_jccf_callback(recs, tg1)

    def bad(f, tg=tg1):
        return lambda r, x: f(r, x) if (r["call"], r["op"]) in tg else None
    expect(bool(det_si_jccf_check(bad(lambda r, x: (good(r, x)[0], good(r, x)[1], x[2] / x[9])), stmts, tg1, recs)),
           "inject a changed spontaneous rate: refused")
    expect(bool(det_si_jccf_check(bad(lambda r, x: tuple(v / x[9] for v in good(r, x)[:2]) + (x[2],)), stmts, tg1,
                                  recs)), "inject beta applied twice: refused")
    expect(bool(det_si_jccf_check(bad(lambda r, x: (good(r, x)[0], x[1], x[2])), stmts, tg1, recs)),
           "inject only the absorption rate substituted: refused")
    expect(bool(det_si_jccf_check(lambda r, x: (good(r, x) if (r["call"], r["op"]) in tg1 else (x[0], x[1], x[2])),
                                  stmts, tg1, recs)), "inject a change on non-target statements: refused")
    expect(bool(det_si_jccf_check(det_si_jccf_callback(recs, tg1 | tg2), stmts, tg2, recs)),
           "inject C-G2 carrying C-G1 (accumulated state): refused")
    # 5. distribution, classes, branch, claim
    st = det_si_jccf_qstats([4.0, 1.0, 0.0, 2.0])
    expect(st["q_median"] == 1.5 and st["n_q_eq0"] == 1 and st["n_q_eq1"] == 1 and st["n_q_gt1"] == 2,
           "even-sample median and counts (positive control)")
    expect(det_si_jccf_classify(0.01, 0.01) == "이동 미분리; 공개 거리 변화 미분리" and
           det_si_jccf_classify(-0.02, -0.02) == "이동 분리; 이격", "boundary values unresolved (positive control)")
    expect([det_si_jccf_branch(True, 0.02), det_si_jccf_branch(True, 0.01), det_si_jccf_branch(False, 0.5),
            det_si_jccf_branch(True, float("nan"))] == [0, 1, 2, 2], "branch fixtures (positive control)")
    rows = [{"kind": "state", "state": "C-ALL", "undefined": 1, "III_II": 3.0, "cls": "미산출"},
            {"kind": "state", "state": "C-G1", "undefined": 0, "III_II": 3.0, "cls": "effect resolved"},
            {"kind": "state", "state": "C-REP6", "undefined": 0, "range_reasons": "REP6:x", "III_II": 3.0,
             "cls": "미산출"}]
    expect(det_si_jccf_claim_check(rows) == ["PARTIAL_AS_FULL:C-ALL", "LABEL:C-G1", "PARTIAL_AS_FULL:C-REP6"],
           "inject a partial substitution or a stopped range shown as a full ratio and an unregistered label: refused")
    fails = sum(1 for r in results if not r[0])
    print(json.dumps({"schema": "DET_JC_COARSE_CF_NEGATIVE_V1", "checks": len(results), "failures": fails}))
    return 0 if fails == 0 else 2


def cmd_jccf(run, out):
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

    # ---- G-INPUT: adoption, statements, membership, generations, line / ledger / bin linkage, public row
    call, why = EB.det_si_exc_adopt(TR.read_trace(tr / "si10_ne.csv"), TR.read_trace(tr / "si10_calls.csv"),
                                    TR.read_trace(tr / "si10_cevec.csv"), TR.read_trace(tr / "si10_commit.csv"),
                                    ADOPTED)
    if call is None:
        return TR.refuse(why)
    cid = call["call"]
    N, N_fl, n_lo, gh = int(call["N"]), int(call["N_fl"]), int(call["n_lo_super"]), int(call["ground_hi"])
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
    info = GR.det_si_bb_groups_fl_info(sm)
    stmts = [(r, [TR.num(r[f"x{k}"]) for k in range(1, 15)]) for r in ops if r["site"] == "1"]
    m8 = [(r, x) for r, x in stmts if CF.det_si_bb_mode(r) == 8]
    bad_stmt = [f"{CF.det_si_bb_check(r, x)}:op{r['op']}" for r, x in m8 if CF.det_si_bb_check(r, x)]
    if bad_stmt:
        gates["G-INPUT"].append(f"STATEMENT_CONDITIONS:{len(bad_stmt)}:" + ";".join(bad_stmt[:5]))
    assign, why_a = GR.det_si_bb_groups_assign(stmts, info, n_lo)
    why_a += GR.det_si_bb_groups_partition(assign, [(r["call"], r["op"]) for r, _ in m8])
    if why_a:
        gates["G-INPUT"].append(f"MEMBERSHIP:{len(why_a)}:" + ";".join(why_a[:5]))
    count = {g: sum(1 for v in assign.values() if v == g) for g in GR.GROUPS}
    if count != JCCF_GROUP_COUNTS:
        gates["G-INPUT"].append(f"REGISTERED_COUNTS:{count}")
    rep_keys, why_r = det_si_jccf_rep6(m8, assign, info)
    if why_r:
        gates["G-INPUT"].append("REP6:" + ";".join(why_r))
    manifest = json.loads((pc / "physics_DET_iter0010.manifest.json").read_text())
    bins, why_g = det_si_jccf_grid(TR.read_trace(pc / "physics_DET_iter0010.spectral.csv"), SHELL, manifest,
                                   int(call["rad_generation"]))
    if why_g:
        gates["G-INPUT"].append("GRID:" + ";".join(why_g[:5]))
    exports = (run / "input/resolved_lumina.exports").read_text()
    if f'LUMINA_CMF_FINE_VDOP="{int(JCCF_VDOP)}"' not in exports:
        gates["G-INPUT"].append("SETTINGS_VDOP")
    aux_ids = {int(r["aux"]) for r, _ in m8}
    atom = {}
    with open(run / "input/model/line_list.csv") as f:
        head = f.readline().rstrip("\n").split(",")
        for ln in f:
            if not (ln.startswith("14,1,") or ln.startswith("14,2,")):
                continue
            a = dict(zip(head, ln.rstrip("\n").split(",")))
            if int(a["line_id"]) in aux_ids:
                atom.setdefault((a["ion_number"], int(a["line_id"])), []).append(a)
    with open(run / "work/w5_lines.csv") as f:
        ledger = det_si_line_input_ledger(f, aux_ids)
    recs, undefined = det_si_jccf_statements(m8, n_lo, info, atom, ledger, bins, why_g)
    for k, e in recs.items():
        e["group"] = assign.get(k)
    t_shell = TR.dump_row(run, stop_iter, "shell", SHELL)[0]
    vel = (float(t_shell["v_inner_cm_s"]) + float(t_shell["v_outer_cm_s"])) / 2e5
    pub, pub_vel, pub_ii, pub_iii = EB.det_si_exc_public(vel)
    if not (math.isclose(vel, TR.REGISTERED_VEL["lumina_vel_mid_kms"]) and
            math.isclose(pub_vel, TR.REGISTERED_VEL["public_vel_mid_kms"]) and pub == pub_iii / pub_ii and
            (pub_ii, pub_iii) == (JCCF_PUBLIC["si_II"], JCCF_PUBLIC["si_III"])):
        gates["G-INPUT"].append(f"PUBLIC_CORRESPONDENCE:{vel}->{pub_vel}:{pub_ii}:{pub_iii}")

    # ---- operators and G-RATE (every state built from F0 independently)
    A0, und, ch0, maps = CF.det_si_bb_build(ops, sm, N, gh)
    states = {}
    for s in JCCF_STATES:
        tg, und_s, rng = det_si_jccf_scope(s, m8, assign, rep_keys, why_r, undefined)
        st = {"targets": tg, "undefined": und_s, "range": rng, "A": None, "ch": None, "x": None, "why": []}
        if s == "F0":
            st["A"], st["ch"] = A0, ch0
        elif not tg:
            st["why"] = ["NO_TARGETS"]
        elif und_s or rng:
            st["why"] = (["UNDEFINED_SUBSTITUTION"] if und_s else []) + rng
        else:
            cb = det_si_jccf_callback(recs, tg)
            bad = det_si_jccf_check(cb, stmts, tg, recs)
            if bad:
                gates["G-RATE"].append(f"PRESCRIPTION:{s}:{len(bad)}")
            A, ug, chs, _ = CF.det_si_bb_build(ops, sm, N, gh, cb)
            und = und + ug
            st["A"], st["ch"] = A, chs
        states[s] = st
    if und or maps["gh_fl"] is None:
        return TR.refuse(gates["G-INPUT"] + ["FL_OPERATOR_UNDEFINED:" + ",".join(sorted(set(und)))])
    for s, st in states.items():
        if st["A"] is None:
            continue
        if not CF.det_si_bb_channel_sum_ok(st["A"], st["ch"]):
            gates["G-RATE"].append(f"CHANNEL_SUM:{s}")
        if EB.det_si_exc_conservation(st["A"]):
            gates["G-RATE"].append(f"CONSERVATION:{s}")
        if s != "F0":
            inv = CF.det_si_bb_invariance(ch0, st["ch"], set(GR.MODE8))
            if inv:
                gates["G-RATE"].append(f"INVARIANCE:{s}:" + ",".join(inv))

    # ---- G-SOLVE
    R, eta_max = {}, {}
    for s, st in states.items():
        if st["A"] is None:
            continue
        x, why_s, eta_max[s] = CF.det_si_bb_solve(st["A"], total)
        if x is not None:
            ratio = TR.ratio(x, upper)
            if math.isfinite(ratio) and ratio > 0.0:
                st["x"], R[s] = x, ratio
            else:
                why_s = why_s + ["RATIO_UNDEFINED"]
        st["why"] = why_s
        if why_s:
            gates["G-SOLVE"].append(f"{s}:" + ",".join(why_s))
    ref_dev = TR.dex(R.get("F0", float("nan")), JCCF_R0)
    if not (abs(ref_dev) <= JCCF_REF_DEX):
        gates["G-SOLVE"].append(f"F0_REFERENCE:{R.get('F0')}")

    # ---- the table
    d0 = TR.dex(R["F0"], pub) if "F0" in R else float("nan")
    rows = [{"kind": "public", "state": "CMFGEN toy06 19.480 d", "scope": "public Si II / Si III ion fractions",
             "III_II": pub, "si_II": pub_ii, "si_III": pub_iii, "vel_kms": pub_vel}]
    G = {}
    for s in JCCF_STATES:
        st = states[s]
        tg, und_s = st["targets"], st["undefined"]
        reasons = {}
        for v in und_s.values():
            reasons[v] = reasons.get(v, 0) + 1
        row = {"kind": "state", "state": s, "scope": JCCF_SCOPE[s], "planned": len(tg),
               "definable": len(tg) - len(und_s), "substituted": len(tg) if (st["A"] is not None and s != "F0") else 0,
               "undefined": len(und_s), "undefined_reasons": json.dumps(dict(sorted(reasons.items()))) if und_s else "",
               "range_reasons": ";".join(st["range"]), "eta_max": eta_max.get(s)}
        if s in R:
            d = TR.dex(R[s], pub)
            row.update(III_II=R[s], delta_F0=TR.dex(R[s], R["F0"]) if "F0" in R else None, d_pub=d,
                       position="위" if d > JCCF_DEX else "아래" if d < -JCCF_DEX else "일치(보고 분해능 이내)")
            if s == "F0":
                row.update(cls="기준", reference_dex=ref_dev)
            elif "F0" in R:
                G[s] = abs(d0) - abs(d)
                row.update(G=G[s], cls=det_si_jccf_classify(row["delta_F0"], G[s]),
                           **det_si_jccf_qstats([recs[k]["q"] for k in tg]),
                           profile_crossing=sum(1 for k in tg if recs[k]["profile_crosses_bin"]))
            else:
                row.update(cls="미산출")
        else:
            row.update(cls="미산출", reason=";".join(st["why"]))
        rows.append(row)
    gates["G-CLAIM"] += det_si_jccf_claim_check(rows)
    sub = [r for r in rows if r["kind"] == "state" and r["state"] != "F0"]
    produced = all(r["state"] in R for r in sub)
    small = produced and all(r["cls"].startswith(JCCF_EFF[1]) for r in sub)
    offset = produced and sub[0]["cls"].startswith(JCCF_EFF[1]) and any(r["cls"].startswith(JCCF_EFF[0])
                                                                        for r in sub[1:])
    summary = ("모든 대입 상태가 이동 미분리 — 이 저장장 대입 처방의 영향이 작다(다른 입사장 처방·자기일관 효과의 부재를 "
               "뜻하지 않음)" if small else
               "C-ALL 이동 미분리이나 일부 범위 이동 분리 — 상쇄 가능성; 입사장 축 전체가 작다고 결론내리지 않음" if offset
               else "영향이 작다고 보고하지 않음(이동 분리 또는 미산출 범위 있음)")
    all_gates = [f for v in gates.values() for f in v]
    valid_all = not all_gates and all(s in R for s in JCCF_STATES)
    code = det_si_jccf_branch(valid_all, G.get("C-ALL"))
    label = {0: "RESOLVED_APPROACH", 1: "NO_RESOLVED_APPROACH", 2: "INDETERMINATE"}[code]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "jc_coarse_cf_table.csv", "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}" if math.isfinite(v) else "nan"
        return str(v).replace("|", "/")
    context = {"F0_R0": JCCF_R0, "public_si_II": pub_ii, "public_si_III": pub_iii, "public_III_II": pub,
               "FINAL_FL_III_II_context_only": JCCF_FINAL_FL, "time_d": 19.48,
               "vel_kms": f"Lumina {vel:.0f} <-> public {pub_vel}"}
    with open(out / "jc_coarse_cf_table.md", "x") as f:
        f.write(f"# J_c -> coarse bin J_nu substitution, Si shell 10 mode-8 bracket (branch {code} {label})\n\n")
        f.write("Context (quoted): " + json.dumps(context, ensure_ascii=False) + "\n\n")
        f.write("Scope: " + JCCF_NOTE + "\n\n")
        f.write("Gates: " + (";".join(all_gates) or "all pass") + "\n\n")
        cols = ["state", "scope", "planned", "definable", "substituted", "undefined", "III_II", "delta_F0", "d_pub",
                "position", "G", "q_min", "q_median", "q_max", "n_q_eq0", "n_q_0_1", "n_q_eq1", "n_q_gt1",
                "profile_crossing", "cls", "undefined_reasons", "range_reasons", "reason"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
        f.write(f"\nSummary: {summary}\n\nNotes: delta_F0 = log10(R/R_F0); d_pub = log10(R/R_pub); G = |d_F0| - |d|; "
                "classes at 0.01 dex (reporting resolution, boundary unresolved); position 위/아래 beyond 0.01 dex; q = "
                "J_coarse / J_c per statement with equal weight, n_q_0_1 counts 0 < q < 1; profile_crossing counts "
                "+-4 Doppler widths (v_D 1e6 cm/s) leaving the centre bin (display only, centre-bin value used).\n")
    used = [tr / n for n in ("si10_calls.csv", "si10_ne.csv", "si10_commit.csv", "si10_cevec.csv", "si10_ops.csv",
                             "si10_slmap.csv", "si10_final.csv", f"si10_mat_{int(cid):06d}_gen.bin")]
    used += [run / "RUN_FOOTER.txt", run / "input/model/line_list.csv", run / "input/resolved_lumina.exports",
             run / "work/w5_lines.csv", pc / "physics_DET_iter0010.manifest.json",
             pc / "physics_DET_iter0010.spectral.csv", pc / f"physics_DET_iter{stop_iter:04d}.shell.csv",
             TR.PUBLIC_SI, TR.DIAG8_PUBLIC]
    evidence = {"schema": "DET_JC_COARSE_CF_V1", "prereg": JCCF_PREREG, "argv": sys.argv,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "branch": code, "label": label, "gates": gates,
                "summary": summary, "note": JCCF_NOTE, "context": context,
                "run_identity": {k: footer.get(k) for k in ("job_id", "binary_sha256", "effective_exports_sha256",
                                                            "terminal_state", "stop_iteration")},
                "adopted": {k: int(call[k]) for k in ("bundle", "ne_iter", "ce_pass", "call", "rad_generation")},
                "manifest": {k: manifest.get(k) for k in list(JCCF_GENERATIONS) + list(JCCF_GRID_KIND) + ["n_bins"]},
                "group_counts": count, "rep6": [list(k) for k in rep_keys], "ratios": R, "eta_max": eta_max,
                "reference_dex": ref_dev, "solve": {s: st["why"] for s, st in states.items()},
                "undefined": {f"{k[0]}:{k[1]}": v for k, v in sorted(undefined.items())},
                "statements": [dict(e, key=f"{k[0]}:{k[1]}") for k, e in sorted(recs.items(),
                                                                                  key=lambda kv: int(kv[0][1]))],
                "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": TR.sha(p)} for p in used},
                "code": {str(Path(p).resolve()): TR.sha(Path(p))
                         for p in (__file__, TR.__file__, EB.__file__, CF.__file__, GR.__file__)}}
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
    sp.add_parser("jcdef-negative")
    sp.add_parser("jccf-negative")
    for name in ("analyze", "jcdef"):
        s = sp.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--baseline", required=True)
        s.add_argument("--out", required=True)
    s = sp.add_parser("jccf")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "negative":
        return cmd_negative()
    if a.cmd == "jcdef-negative":
        return cmd_jcdef_negative()
    if a.cmd == "jccf-negative":
        return cmd_jccf_negative()
    if a.cmd == "jcdef":
        return cmd_jcdef(a.run, a.baseline, a.out)
    if a.cmd == "jccf":
        return cmd_jccf(a.run, a.out)
    return cmd_analyze(a.run, a.baseline, a.out)


if __name__ == "__main__":
    sys.exit(main())
