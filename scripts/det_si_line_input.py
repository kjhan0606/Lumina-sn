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
    "upstream": "excluded for mapped lines: fine chi_es/chi_abs are interpolated from the bin state (about "
                "8385-8399); the bin chi_es carries the line scattering share (chi_ln - chi_ln_th) (about 2962) "
                "with the per-line eps_phys branch (LUMINA_CMFGEN_LINE_EPS_PHYS=1; LINE_EPS and LINE_EPS_UV unset), "
                "but under the active DET-LTEUNMAP-EPS gate mapped lines take eps = 1 (fully thermal, no scattering "
                "share; about 2512-2532 and the helper at about 163) and the unmapped share accumulated in "
                "g_chi_line_sc_unmapped (about 2530) is subtracted in the fine interpolation (about 8394); bin "
                "chi_abs = bf + ff only and the fine continuum emissivity w5_cont_eta = bf (Milne) + ff with the "
                "line term passed as 0 (about 2980-2987)",
    "incident": "J_c = Gaussian profile mean of fs.J within +-4 Doppler widths (line_jbar_gaussian_discrete_shells, "
                "lumina_cmfgen.c about 9780) passed as continuum_j to line_net_sobolev_radiation (about 9810): "
                "c = beta J_c, l = eta (c t / nu)(1 - beta)/tau, Jbar = c + l (line_net_rate.c)",
    "cmfgen": "SOBJBAR_SIM (Sobolev path: cmfgen_sub.f SECTION='LINE' about 3504 -> SUB_SOB_LINE_V3 about 3638 -> "
              "sobjbar_sim.f): CHI and SOURCE = ZETA + RJ*THETA come from COMP_OPAC at the line frequency, which "
              "adds bound-free, free-free, two-photon, X-ray, scattering and optional additional opacity but no "
              "line opacity (comp_opac.f), and RJ from COMP_J_BLANK on that opacity; the per-ray AV is weighted "
              "by EXPONX(GAM) and integrated over angle, with the simultaneous lines' CHIL/ETAL entering the "
              "Sobolev optical depth and local term (cmfgen_sub.f about 3535)",
    "same": "the incident field excludes the line itself and other lines; the local line term is added separately",
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
                   self_line_direct="excluded", upstream_residual="none (mapped eps = 1; unmapped share subtracted)",
                   other_lines="excluded from the incident field", cmfgen_incident="continuum AV (COMP_OPAC + "
                   "COMP_J_BLANK, no line opacity)", cmfgen_same=JCDEF_SOURCE["same"],
                   cmfgen_different=JCDEF_SOURCE["different"], public_execution="not established",
                   J_c=J_c, J_coarse=J_coarse, dex_previous=float(b["dex_J_c_vs_coarse"]),
                   bin_nu_lo=float(bins["nu_lo_Hz"]) if bins else None,
                   bin_nu_hi=float(bins["nu_hi_Hz"]) if bins else None)
        if comp:
            row.update({k: comp[k] for k in ("chi_es", "chi_bb", "chi_bf", "chi_ff", "chi_total", "chi_cont",
                                             "q_bb")})
        row.update(delta_def="not identified (no matched field pair; no new transfer)",
                   magnitude="직접 중복 기전 배제",
                   relation_previous="the coarse J_nu is the binned field with line opacity and emission "
                                     "(q_bb shown); J_c is the line-free continuum by construction; the dex "
                                     "difference is not attributed",
                   double_counting="refuted: no self-line inflow into J_c on the consumed branch",
                   conclusion="자기 선 유입 배제; CMFGEN Sobolev 경로와 입사장·국소 선 항 분리 대응(남은 차이 기재)")
        rows.append(row)
    for r in rows:
        if r.get("magnitude") and r["magnitude"] not in ("식별", "미식별", "직접 중복 기전 배제"):
            gates["G-CLAIM"].append(f"MAGNITUDE_LABEL:{r['aux']}")
    all_gates = [f for v in gates.values() for f in v]
    complete = all(r.get("self_line_direct") == "excluded" for r in rows) and len(rows) == 6
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
                "upstream_residual", "magnitude", "conclusion"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
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


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("negative")
    sp.add_parser("jcdef-negative")
    for name in ("analyze", "jcdef"):
        s = sp.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--baseline", required=True)
        s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "negative":
        return cmd_negative()
    if a.cmd == "jcdef-negative":
        return cmd_jcdef_negative()
    if a.cmd == "jcdef":
        return cmd_jcdef(a.run, a.baseline, a.out)
    return cmd_analyze(a.run, a.baseline, a.out)


if __name__ == "__main__":
    sys.exit(main())
