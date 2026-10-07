#!/usr/bin/env python3
"""DET-SIIII-RATES readout (prereg docs/PREREG_DET_SIIII_RATES_2026-10-07.md, rev 3).

Subcommands
  selftest --production-source DIR --out DIR
  analyze  --e-run DIR --r-run DIR --production-source DIR --prior-pop DIR
           --selftest DIR --out DIR

Generation-7 rate budget of the Si III upper levels u=2 (1892.03 A) and u=5
(2542.58 A) in the E' (NRB off) and R' (NRB on) captures.  Populations of u and
of every partner level j joined to u by a Si III line are recovered from the
captured (tau, S_used) of that line (det_siiii_pop_readout.recover).  The only
direct processes on u in the production matrix (prereg 2.3, evidence
production_paths_grep) are Si III bound-bound radiative and bound-bound
collisional terms; every other direct process is STRUCTURAL_ZERO and is only
recorded.  Coefficients follow T2 src/lumina_plasma.c nlte_assemble_rate_matrix_impl:

  radiative, NRB off  : P_lu = B_lu Jbar,   P_ul = A_ul + B_ul Jbar
  radiative, NRB armed: P_lu = B_lu ct,     P_ul = beta A_ul + B_ul ct   (srce_chk == 0)
                        (srce_chk == 1 -> off form, line_net_bracket_rates NOT_APPLICABLE)
  ct = fl(beta J_c), Jbar = fl(ct + L_l)    (src/line_net_rate.c line_net_sobolev_radiation)
  collisions (NLTE_COLL_FIX): ups = max(14.5 f (Ry/dE)^2 0.2, 1)
        C_up = n_e 8.629e-6/(g_l sqrt T) ups exp(-dE/kT), C_down = n_e 8.629e-6/(g_u sqrt T) ups

Budget I_u = sum_j n_j P_ju, D_u = sum_j P_uj, R_u = I_u - n_u D_u; the E'-R' density
difference is split exactly (prereg 6.1-6.3) into NRB form, radiative field,
collision coefficient, partner population and residual terms.  Every number is
an interval [lo, hi] with a middle value.  Commands write only into a new --out.
"""
import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_siiii_pop_readout as POP                                        # noqa: E402
from det_siiii_pop_readout import I, ifsum, printed, recover, prod_tau_S, sha, REC, C, H  # noqa: E402

K_B, EV_TO_ERG = 1.380649e-16, 1.602176634e-12          # T2 src/lumina.h
COLL_K, AXELROD_OMEGA, RYD_EV, GBAR = 8.629e-6, 1.0, 13.6057, 0.2   # T2 lumina_plasma.c NLTE_COLL_FIX branch
NU_MIN, NU_MAX = 1.49896229e14, 2.99792458e16           # T2 src/line_jbar.h bb domain
Z, ION = 14, 2
TARGETS = ({"name": "SiIII_1892", "row": 908565, "lo": 0, "up": 2, "main": [24, 30]},
           {"name": "SiIII_2542", "row": 1097363, "lo": 4, "up": 5, "main": [19]})
READ_SHELLS = list(range(17, 31))
W5_CALL, W5_OP, GEN = 7, "CMFGEN_NONOVERLAP_SOBOLEV", 7
ALLOWED_DIFF = {"LUMINA_BB_NRB", "LUMINA_A210_SPRODUCER_CAPTURE"}
OFF_REQUIRED = ("LUMINA_NLTE_STAGE4", "LUMINA_NLTE_ELEMENT_WIDE", "LUMINA_TOPSTAGE_IV", "LUMINA_TIMEDEP_ION",
                "LUMINA_ARTIS_PARITY", "LUMINA_NLTE_COLL_ION", "LUMINA_OMEGA_CMFGEN", "LUMINA_NLTE_METASTABLE_COLL",
                "LUMINA_NLTE_JEQB", "LUMINA_NLTE_COLL_FLOOR")
STRUCTURAL_ZERO = ("photoionization", "radiative_recombination", "collisional_ionization_three_body",
                   "dielectronic_recombination", "nonthermal_ionization", "charge_exchange")
TERMS = ("NRB_FORM", "RADIATIVE_FIELD", "COLLISION", "DIRECT_IONIZATION", "PARTNER_POPULATION", "RESIDUAL")
TERM_CODE = {"NRB_FORM": 0, "RADIATIVE_FIELD": 1, "COLLISION": 1, "DIRECT_IONIZATION": 1,
             "PARTNER_POPULATION": 2, "RESIDUAL": 3}
CODE_NAME = {0: "NRB_FORM_DOMINANT", 1: "OTHER_RATE_DOMINANT", 2: "PARTNER_POPULATION_DOMINANT",
             3: "RESIDUAL_DOMINANT", 4: "MIXED", 5: "UNRESOLVED", 6: "NOT_IDENTIFIABLE"}
FAULTS = ("drop_beta", "drop_stim", "flip_coll", "dup_line", "drop_residual", "declare_eq")


def neg(a):
    return I(-a.hi, -a.mid, -a.lo)


def sc(a, k):
    """exact-constant scaling (k > 0)."""
    return I(POP.dn(a.lo * k), a.mid * k, POP.up(a.hi * k))


def ivj(a):
    return None if a is None else a.j()


def overlap(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


def det_siiii_rates_exponx(tau):
    """T2 src/line_net_rate.c line_net_cmfgen_exponx, same operation order."""
    if not math.isfinite(tau):
        return None
    if abs(tau) < 1.0e-3:
        comp = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        val = 1.0 - tau * comp
    elif tau < 40.0:
        val = (1.0 - math.exp(-tau)) / tau
        comp = (1.0 - val) / tau
    else:
        val = 1.0 / tau
        comp = (1.0 - val) / tau
    return val if (math.isfinite(val) and val > 0.0 and math.isfinite(comp) and comp > 0.0) else None


# ------------------------------------------------------------------ inputs
def det_siiii_rates_exports(run):
    d = {}
    for ln in open(Path(run) / "input/resolved_lumina.exports"):
        ln = ln.strip()
        if ln.startswith("declare -x ") and "=" in ln:
            k, v = ln[len("declare -x "):].split("=", 1)
            d[k] = v.strip('"')
    return d


def det_siiii_rates_config(eE, eR, e_run, r_run):
    """prereg 1.4 / 2.3: E-R differ only in NRB and the diagnostic capture key (paths
    re-rooted); the structural-zero premises hold in both states."""
    why = []
    keys = set(eE) | set(eR)
    for k in sorted(keys):
        a = eE.get(k)
        b = eR.get(k)
        if a is not None:
            a = a.replace(str(Path(e_run)), str(Path(r_run)))
        if a != b and k not in ALLOWED_DIFF:
            why.append(f"CONFIG_DIFF:{k}")
    for tag, ex in (("E", eE), ("R", eR)):
        for k in OFF_REQUIRED:
            if ex.get(k, "0") not in ("", "0"):
                why.append(f"{tag}:CONFIG_PREMISE:{k}={ex.get(k)}")
        if ex.get("LUMINA_NLTE_COLL_FIX") != "1":
            why.append(f"{tag}:CONFIG_PREMISE:LUMINA_NLTE_COLL_FIX")
        if ex.get("LUMINA_SUPER_LEVELS") != "1" or not ex.get("LUMINA_SUPER_CUTOFF", "").isdigit():
            why.append(f"{tag}:CONFIG_PREMISE:SUPER_LEVELS")
        if str(Z) in ex.get("LUMINA_NLTE_SKIP_Z", "").replace(",", " ").split():
            why.append(f"{tag}:CONFIG_PREMISE:SKIP_Z")
    if eE.get("LUMINA_BB_NRB", "0") not in ("", "0") or eR.get("LUMINA_BB_NRB") != "1":
        why.append("CONFIG_NRB_STATES")
    if eE.get("LUMINA_SUPER_CUTOFF") != eR.get("LUMINA_SUPER_CUTOFF"):
        why.append("CONFIG_DIFF:SUPER_CUTOFF")
    return why


def det_siiii_rates_lines(run, targets):
    """every Si III line joined to a target upper level: data row -> exact floats."""
    ll = Path(run) / "input/model/line_list.csv"
    key = pd.read_csv(ll, usecols=["atomic_number", "ion_number", "level_number_lower", "level_number_upper"])
    ups = {t["up"] for t in targets}
    m = (key.atomic_number == Z) & (key.ion_number == ION) & (
        key.level_number_lower.isin(ups) | key.level_number_upper.isin(ups))
    rows = sorted(int(x) for x in key.index[m])
    want = set(rows)
    d = pd.read_csv(ll, skiprows=lambda i: i > 0 and (i - 1) not in want, float_precision="round_trip")
    d.index = rows
    return d


def det_siiii_rates_w5(run, rows):
    """W5 rows of the given line ids: (call, op, rad, mat, shell, line) -> record, plus
    contradictory-duplicate keys.  Rows are kept for every call; the caller selects."""
    want = set(int(r) for r in rows)
    out, contra = {}, []
    for ch in pd.read_csv(Path(run) / "work/w5_lines.csv", dtype=str, chunksize=1 << 20):
        sub = ch[ch.line_id.astype(int).isin(want)]
        for r in sub.itertuples(index=False):
            k = (int(r.call), r.operator, int(r.rad_generation), int(r.material_generation), int(r.shell),
                 int(r.line_id))
            rec = r._asdict()
            if k in out and out[k] != rec:
                contra.append(k)
            out[k] = rec
    return out, contra


def det_siiii_rates_w5_record(w5, line, s):
    """the call-7 Sobolev row of rad/material generation 7; None if absent; reason."""
    r = w5.get((W5_CALL, W5_OP, GEN, GEN, s, line))
    if r is None:
        other = [k for k in w5 if k[4] == s and k[5] == line and k[0] == W5_CALL]
        return None, ("W5_GENERATION_MISMATCH" if other else "MISSING_LINE_FIELD")
    if r["status"] != "OK":
        return None, f"W5_STATUS_{r['status']}"
    return r, None


def det_siiii_rates_g8(run, shells):
    df = pd.read_csv(Path(run) / "work/nlte_levels_final.csv", dtype={"n_pop": str})
    df = df[(df.Z == Z) & (df.ion == ION) & (df.shell.isin(shells))]
    return {(int(r.shell), int(r.level_idx)): r.n_pop for r in df.itertuples()}


# ------------------------------------------------------------------ coefficients
def det_siiii_rates_coll(n_e, T_e, E_l, E_u, g_l, g_u, f_lu, fault=None):
    """T2 NLTE_COLL_FIX branch, same operation order.  Interval = float value
    (1 +- (32 + 8|x|) u), x = dE/kT: the float carries <= 2u in dE, 4u in x (abs.
    4|x|u in the exponent), <= 2u from libm exp, 1u sqrt and ~10u from the
    ups / prefactor products, i.e. <= (20 + 4|x|) u from the exact formula on
    the same double inputs; the bound doubles that so the exact value is inside."""
    dE = abs(E_u - E_l) * EV_TO_ERG
    if not (T_e > 0.0 and dE > 0.0 and g_l > 0 and g_u > 0):
        return I.x(0.0), I.x(0.0)
    ef = math.exp(-dE / (K_B * T_e))
    dE_eV = dE / EV_TO_ERG
    ry = RYD_EV / (dE_eV if dE_eV > 0.0 else 1e30)
    ups_vr = 14.5 * f_lu * ry * ry * GBAR
    ups = ups_vr if ups_vr > AXELROD_OMEGA else AXELROD_OMEGA
    cu = n_e * COLL_K / (g_l * math.sqrt(T_e)) * ups * ef
    cd = n_e * COLL_K / (g_u * math.sqrt(T_e)) * ups
    if fault == "flip_coll":
        cu, cd = cd, cu
    k = 32.0 + 8.0 * abs(dE / (K_B * T_e))
    return I.rel(cu, k), I.rel(cd, k)


def det_siiii_rates_rad(B_lu, B_ul, A_ul, nu, w, fault=None):
    """(P_lu, P_ul) for the off form and the armed form (srce_chk==1 -> off)."""
    if not (NU_MIN <= nu <= NU_MAX):                 # BB_EXCLUDED_OUTSIDE_DOMAIN: A only
        z = I.x(0.0)
        return {"off": (z, I.x(A_ul)), "armed": (z, I.x(A_ul)), "domain": False}
    Bl, Bu, A = I.x(B_lu), I.x(B_ul), I.x(A_ul)
    jb, je = float(w["Jbar"]), float(w["Jbar_err_upper"])
    jc, jce, beta = float(w["J_c"]), float(w["J_c_err_upper"]), float(w["beta"])
    J = I(POP.dn(jb - je), jb, POP.up(jb + je))
    ct_mid = beta * jc
    ct = I(POP.dn(beta * (jc - jce)), ct_mid, POP.up(beta * (jc + jce)))
    ct = I(POP.dn(ct.lo * (1 - 2 * POP.U)), ct.mid, POP.up(ct.hi * (1 + 2 * POP.U)))
    stim = (lambda x: I.x(0.0)) if fault == "drop_stim" else (lambda x: Bu * x)
    off = (Bl * J, A + stim(J))
    if int(w["srce_chk"]) == 1:
        armed = off
    else:
        bA = A if fault == "drop_beta" else I.rel(A_ul * beta, 1)
        armed = (Bl * ct, bA + stim(ct))
    return {"off": off, "armed": armed, "domain": True}


# ------------------------------------------------------------------ one state
def det_siiii_rates_state(run, tag, targets, lines, lv, K_cut, fault=None):
    cap = Path(run) / "work/det_obs_capture"
    meta = POP.read_meta(cap)
    man = json.load(open(cap / "manifest.json"))
    why = []
    gens = man.get("generations", {})
    if gens.get("nlte_population_committed") != GEN or gens.get("material") != GEN:
        why.append(f"{tag}:LINK_CAPTURE_GENERATION")
    if meta.get("material_generation", str(GEN)) != str(GEN) or meta.get("rad_generation", str(GEN)) != str(GEN):
        why.append(f"{tag}:LINK_CAPTURE_META_GENERATION")
    t_exp = float.fromhex(meta["time_explosion"])
    Te = np.fromfile(cap / "Te.f64", "<f8")
    ne = np.fromfile(cap / "ne.f64", "<f8")
    if POP.deck_sha_of(run, "line_list.csv") != sha(Path(run) / "input/model/line_list.csv"):
        why.append(f"{tag}:LINK_LINE_LIST_NOT_SEALED")
    rows = list(lines.index)
    recs = POP.capture_records(cap, rows, set(READ_SHELLS))
    w5, contra = det_siiii_rates_w5(run, rows)
    for k in contra:
        why.append(f"{tag}:LINK_W5_CONTRADICTORY_DUPLICATE:{k[5]}:{k[4]}")
    for t in targets:
        r = lines.loc[t["row"]] if t["row"] in lines.index else None
        if r is None or (int(r.level_number_lower), int(r.level_number_upper)) != (t["lo"], t["up"]):
            why.append(f"{tag}:LINK_TARGET_ROW:{t['name']}")
    per = {}
    for t in targets:
        u = t["up"]
        con = lines[(lines.level_number_lower == u) | (lines.level_number_upper == u)]
        con_rows = list(con.index) + ([con.index[-1]] if fault == "dup_line" and len(con) else [])
        for s in READ_SHELLS:
            ent = {"lines": [], "missing": [], "n_u_lines": []}
            for row in con_rows:
                r = lines.loc[row]
                lo, hi = int(r.level_number_lower), int(r.level_number_upper)
                j = hi if lo == u else lo
                u_is_upper = (hi == u)
                g_l, g_u = float(lv.loc[lo, "g"]), float(lv.loc[hi, "g"])
                f_lu, lam = float(r.f_lu), float(r.wavelength_cm)
                nu = float(r.nu)
                rec = recs.get((row, s))
                L = {"row": int(row), "j": j, "lo": lo, "up": hi, "u_is_upper": u_is_upper,
                     "lambda_A": float(r.wavelength), "member": j >= K_cut}
                if rec is None or rec["mapped"] != 1:
                    L["status"] = "MISSING_G7_POPULATION"
                    ent["missing"].append(("MISSING_G7_POPULATION", int(row)))
                    ent["lines"].append(L)
                    continue
                rv = recover(rec["tau"], rec["S"], g_l, g_u, f_lu, lam, t_exp)
                if rv is None:
                    L["status"] = "MISSING_G7_POPULATION"
                    ent["missing"].append(("MISSING_G7_POPULATION", int(row)))
                    ent["lines"].append(L)
                    continue
                n_j = rv["n_u"] if u_is_upper is False else rv["n_l"]
                n_u = rv["n_u"] if u_is_upper else rv["n_l"]
                ent["n_u_lines"].append((int(row), n_u))
                L.update({"n_j": n_j, "n_u": n_u, "tau": rec["tau"], "S": rec["S"]})
                cu, cd = det_siiii_rates_coll(float(ne[s]), float(Te[s]), float(lv.loc[lo, "energy_eV"]),
                                              float(lv.loc[hi, "energy_eV"]), g_l, g_u, f_lu, fault)
                w, wwhy = (None, None)
                in_dom = NU_MIN <= nu <= NU_MAX
                if in_dom:
                    w, wwhy = det_siiii_rates_w5_record(w5, int(row), s)
                    if w is not None:
                        if int(w["Z"]) != Z or int(w["ion"]) != ION or float(w["nu_hz"]) != nu:
                            why.append(f"{tag}:LINK_W5_IDENTITY:{row}:{s}")
                        b = det_siiii_rates_exponx(float(w["tau_eff"]))
                        if b is None or b != float(w["beta"]):
                            why.append(f"{tag}:LINK_W5_BETA:{row}:{s}")
                        if float(w["beta"]) * float(w["J_c"]) + float(w["L_l"]) != float(w["Jbar"]):
                            why.append(f"{tag}:LINK_W5_LEDGER:{row}:{s}")
                if in_dom and w is None:
                    L["status"] = wwhy
                    ent["missing"].append((wwhy, int(row)))
                    L.update({"coll": (cu, cd)})
                    ent["lines"].append(L)
                    continue
                rad = det_siiii_rates_rad(float(r.B_lu), float(r.B_ul), float(r.A_ul), nu, w, fault)
                # orient to (in = j -> u, out = u -> j)
                if u_is_upper:
                    pin = {"off": rad["off"][0], "armed": rad["armed"][0], "coll": cu}
                    pout = {"off": rad["off"][1], "armed": rad["armed"][1], "coll": cd}
                else:
                    pin = {"off": rad["off"][1], "armed": rad["armed"][1], "coll": cd}
                    pout = {"off": rad["off"][0], "armed": rad["armed"][0], "coll": cu}
                L.update({"status": "OK", "domain": rad["domain"], "pin": pin, "pout": pout,
                          "srce_chk": None if w is None else int(w["srce_chk"]),
                          "Jbar": None if w is None else float(w["Jbar"]),
                          "beta": None if w is None else float(w["beta"])})
                ent["lines"].append(L)
            # common target population (prereg 3.2): intervals must all intersect
            nu_ = ent["n_u_lines"]
            if nu_:
                lo_ = max(x.lo for _, x in nu_)
                hi_ = min(x.hi for _, x in nu_)
                tgt = [x for rw, x in nu_ if rw == t["row"]]
                mid = tgt[0].mid if tgt else nu_[0][1].mid
                if lo_ <= hi_:
                    ent["n_u"] = I(lo_, min(max(mid, lo_), hi_), hi_)
                else:
                    ent["n_u"] = None
                    ent["missing"].append(("POPULATION_INCONSISTENT", t["up"]))
            else:
                ent["n_u"] = None
                ent["missing"].append(("MISSING_G7_POPULATION", t["up"]))
            ent["Te"], ent["ne"] = float(Te[s]), float(ne[s])
            per[(t["name"], s)] = ent
    sl = det_siiii_rates_sl_check(per, lv, Te, K_cut)
    return {"per": per, "why": why, "t_exp": t_exp, "sl": sl, "Te": Te, "ne": ne}


def det_siiii_rates_sl_check(per, lv, Te, K_cut):
    """members of the lumped super-level K_cut must stand in the within-SL Boltzmann
    ratio at the generation-7 T_e (prereg 2.2 / G-LINK super-level map).  T2
    population_within_superlevel_build: w_j = g_j exp(-(E_j - E_anchor) eV/kT),
    n_j = N_SL fl(w_j / z).  With x_j = (E_j - E_anchor) eV/kT each w_j carries
    <= (4|x_j| + 3) u, the division by the common z and the product with N_SL 2u
    more each, so n_b/n_a = (g_b/g_a) exp(-(x_b - x_a)) (1 +- (4|x_a| + 4|x_b| + 10) u);
    evaluating the Boltzmann ratio here adds <= (4|x_b - x_a| + 6) u.  |x_j| is
    bounded with the lumped level spread (anchor unknown, inside the super-level).
    The check is the overlap of that interval with the recovered ratio interval."""
    out = []
    E_sl = [float(e) for lvl, e in lv["energy_eV"].items() if int(lvl) >= K_cut]
    spread = (max(E_sl) - min(E_sl)) if E_sl else 0.0
    for (name, s), ent in per.items():
        mem = [(L["j"], L["n_j"]) for L in ent["lines"] if L.get("member") and "n_j" in L]
        mem = sorted(dict(mem).items())
        kT = K_B * float(Te[s])
        xmax = spread * EV_TO_ERG / kT
        for a in range(len(mem) - 1):
            (ja, na), (jb, nb) = mem[a], mem[a + 1]
            r = nb.div(na)
            Ea, Eb = float(lv.loc[ja, "energy_eV"]), float(lv.loc[jb, "energy_eV"])
            ga, gb = float(lv.loc[ja, "g"]), float(lv.loc[jb, "g"])
            dx = (Eb - Ea) * EV_TO_ERG / kT
            want = gb / ga * math.exp(-dx)
            k = 8.0 * xmax + 10.0 + 4.0 * abs(dx) + 6.0
            want_iv = I.rel(want, k)
            ok = r is not None and overlap(r, want_iv)
            out.append({"target": name, "shell": s, "j_a": ja, "j_b": jb, "ratio": ivj(r),
                        "boltzmann": want_iv.j(), "consistent": bool(ok)})
            if not ok:
                ent["missing"].append(("SL_PROJECTION_INCONSISTENT", jb))
    return out


# ------------------------------------------------------------------ budgets
def det_siiii_rates_budget(ent, act, fault=None):
    """process sums for one (target, shell, state); act = 'off' or 'armed'."""
    if ent["missing"] or ent["n_u"] is None:
        return None
    nu = ent["n_u"]
    Ir = ifsum(L["n_j"] * L["pin"][act] for L in ent["lines"])
    Dr = ifsum(L["pout"][act] for L in ent["lines"])
    Ic = ifsum(L["n_j"] * L["pin"]["coll"] for L in ent["lines"])
    Dc = ifsum(L["pout"]["coll"] for L in ent["lines"])
    Iu, Du = Ir + Ic, Dr + Dc
    O = nu * Du
    R = Iu - O
    if fault == "declare_eq":
        R = I.x(0.0)
    eps = R.abs().div(Iu + O)
    return {"I_rad": Ir, "D_rad": Dr, "O_rad": nu * Dr, "I_coll": Ic, "D_coll": Dc, "O_coll": nu * Dc,
            "I": Iu, "D": Du, "O": O, "R": R, "eps": eps, "n_u": nu, "n_lines": len(ent["lines"])}


def det_siiii_rates_decompose(eE, eR, bE, bR, fault=None):
    """prereg 6.1-6.3 exact symmetric split of dn_u = n_u(E') - n_u(R')."""
    by_row_R = {L["row"]: L for L in eR["lines"]}
    Db = (bE["D"] + bR["D"]).half()
    nub = (bE["n_u"] + bR["n_u"]).half()
    dn_u = bE["n_u"] - bR["n_u"]
    f_in, f_out, d_in, d_out, c_in, c_out, part = [], [], [], [], [], [], []
    for LE in eE["lines"]:
        LR = by_row_R[LE["row"]]
        njb = (LE["n_j"] + LR["n_j"]).half()
        dnj = LE["n_j"] - LR["n_j"]
        for side, Pb, acc_f, acc_d, wgt in (("pin", None, f_in, d_in, njb), ("pout", None, f_out, d_out, None)):
            offE, armE = LE[side]["off"], LE[side]["armed"]
            offR, armR = LR[side]["off"], LR[side]["armed"]
            form = ((offE - armE) + (offR - armR)).half()
            field = ((offE - offR) + (armE - armR)).half()
            acc_f.append(form * wgt if wgt is not None else form)
            acc_d.append(field * wgt if wgt is not None else field)
        c_in.append(njb * (LE["pin"]["coll"] - LR["pin"]["coll"]))
        c_out.append(LE["pout"]["coll"] - LR["pout"]["coll"])
        Pbar = ((LE["pin"]["off"] + LR["pin"]["armed"]) + (LE["pin"]["coll"] + LR["pin"]["coll"])).half()
        part.append(Pbar * dnj)
    A = {"NRB_FORM": (ifsum(f_in) - nub * ifsum(f_out)).div(Db),
         "RADIATIVE_FIELD": (ifsum(d_in) - nub * ifsum(d_out)).div(Db),
         "COLLISION": (ifsum(c_in) - nub * ifsum(c_out)).div(Db),
         "DIRECT_IONIZATION": I.x(0.0),
         "PARTNER_POPULATION": ifsum(part).div(Db),
         "RESIDUAL": I.x(0.0) if fault == "drop_residual" else neg((bE["R"] - bR["R"]).div(Db) or I.x(math.nan))}
    total = ifsum(A.values())
    closes = overlap(total, dn_u)
    return {"A": A, "dn_u": dn_u, "D_bar": Db, "n_u_bar": nub, "sum": total, "closes": bool(closes)}


def det_siiii_rates_dominance(A, dn_u):
    """prereg 6.4: strict |A_k|lo > sum_{j!=k} |A_j|hi; equality/overlap never dominant."""
    if dn_u is None or A is None or any(A.get(k) is None for k in TERMS):
        return 6, None, False
    if any(not (math.isfinite(A[k].lo) and math.isfinite(A[k].hi)) for k in TERMS):
        return 6, None, False
    sep = {k: (A[k].lo > 0) - (A[k].hi < 0) for k in TERMS}
    cancelling = any(v > 0 for v in sep.values()) and any(v < 0 for v in sep.values())
    if dn_u.lo <= 0.0 <= dn_u.hi:
        return 5, None, cancelling
    from fractions import Fraction
    mag = {k: A[k].abs() for k in TERMS}
    lo = {k: Fraction(mag[k].lo) for k in TERMS}          # exact sums and comparisons
    hi = {k: Fraction(mag[k].hi) for k in TERMS}
    true_k, all_false = [], True
    for k in TERMS:
        oth_hi = sum(hi[j] for j in TERMS if j != k)
        oth_lo = sum(lo[j] for j in TERMS if j != k)
        if lo[k] > oth_hi and sep[k] != 0:
            true_k.append(k)
        if not (hi[k] <= oth_lo):
            all_false = False
    if len(true_k) == 1:
        return TERM_CODE[true_k[0]], true_k[0], cancelling
    if all_false:
        return 4, None, cancelling
    return 5, None, cancelling


# ------------------------------------------------------------------ analyze
def det_siiii_rates_analyze(e_run, r_run, out, targets=TARGETS, fault=None, prior_pop=None, term_override=None):
    """term_override (selftest only): {"<target>:<shell>": {"dn_u": I, "A": {term: I or None}}}
    replaces the split after the arithmetic-closure check and before the branch
    decision, so boundary cases travel the real decision and output path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    eE, eR = det_siiii_rates_exports(e_run), det_siiii_rates_exports(r_run)
    why = det_siiii_rates_config(eE, eR, e_run, r_run)
    K_cut = int(eE.get("LUMINA_SUPER_CUTOFF", "0") or 0)
    lines_E = det_siiii_rates_lines(e_run, targets)
    lines_R = det_siiii_rates_lines(r_run, targets)
    if list(lines_E.index) != list(lines_R.index) or not lines_E.equals(lines_R):
        why.append("LINK_LINE_LIST_E_R_DIFFER")
    levels = pd.read_csv(Path(e_run) / "input/model/levels.csv", float_precision="round_trip")
    lv = levels[(levels.atomic_number == Z) & (levels.ion_number == ION)].set_index("level_number")
    SE = det_siiii_rates_state(e_run, "E", targets, lines_E, lv, K_cut, fault)
    SR = det_siiii_rates_state(r_run, "R", targets, lines_R, lv, K_cut, fault)
    why += SE["why"] + SR["why"]
    if why:                                  # G-LINK: no science from mislinked inputs
        res = {"schema": "DET_SIIII_RATES_RESULT_V1", "validity": "X", "reasons": why}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        return res
    g8E, g8R = det_siiii_rates_g8(e_run, READ_SHELLS), det_siiii_rates_g8(r_run, READ_SHELLS)
    cov, pops, rates, budg, closure, results = [], [], [], [], [], {}
    for t in targets:
        for s in READ_SHELLS:
            key = (t["name"], s)
            role = "MAIN" if s in t["main"] else "AUXILIARY"
            entE, entR = SE["per"][key], SR["per"][key]
            bE = det_siiii_rates_budget(entE, "off", fault)
            bR = det_siiii_rates_budget(entR, "armed", fault)
            miss = sorted(set(f"E:{a}:{b}" for a, b in entE["missing"]) | set(f"R:{a}:{b}" for a, b in entR["missing"]))
            dec = det_siiii_rates_decompose(entE, entR, bE, bR, fault) if (bE and bR) else None
            if dec is None:
                code, dom, canc = 6, None, False
            elif not dec["closes"]:
                code, dom, canc = 6, None, False
                miss.append("ARITHMETIC_CLOSURE_FAILED")
            else:
                ov = (term_override or {}).get(f"{t['name']}:{s}")
                if ov is not None:
                    dec = dict(dec, A=dict(dec["A"], **ov.get("A", {})), dn_u=ov.get("dn_u", dec["dn_u"]))
                code, dom, canc = det_siiii_rates_dominance(dec["A"], dec["dn_u"])
            cov.append({"target": t["name"], "shell": s, "role": role, "n_lines": len(entE["lines"]),
                        "budget_E": bE is not None, "budget_R": bR is not None,
                        "missing": ";".join(miss), "structural_zero": ";".join(STRUCTURAL_ZERO),
                        "code": code})
            for tag, ent, g8 in (("E", entE, g8E), ("R", entR, g8R)):
                for L in ent["lines"]:
                    base = {"state": tag, "target": t["name"], "shell": s, "row": L["row"], "j": L["j"],
                            "lo": L["lo"], "up": L["up"], "lambda_A": L["lambda_A"], "member": L["member"],
                            "status": L["status"]}
                    if "n_j" in L:
                        tok = g8.get((s, L["j"]))
                        p8 = printed(tok) if tok is not None else None
                        pops.append({**base, "n_j_lo": L["n_j"].lo, "n_j_mid": L["n_j"].mid, "n_j_hi": L["n_j"].hi,
                                     "source": "G7_DIRECT", "n_j_g8": None if p8 is None else p8.mid,
                                     "g7_over_g8_mid": None if p8 is None else L["n_j"].mid / p8.mid})
                    if L["status"] != "OK":
                        rates.append(base)
                        continue
                    act = "off" if tag == "E" else "armed"
                    for proc, kin in (("bb_radiative", act), ("bb_collision", "coll")):
                        Iv, Ov = L["n_j"] * L["pin"][kin], ent["n_u"] * L["pout"][kin] if ent["n_u"] else None
                        rates.append({**base, "process": proc,
                                      "branch": ("COLL_FIX" if kin == "coll" else
                                                 "OUT_OF_DOMAIN_A_ONLY" if not L["domain"] else
                                                 "NRB_ARMED" if (kin == "armed" and L["srce_chk"] == 0) else
                                                 "OFF_SRCE_CHK" if kin == "armed" else "OFF"),
                                      "domain": L["domain"], "srce_chk": L["srce_chk"],
                                      "P_in_mid": L["pin"][kin].mid, "P_out_mid": L["pout"][kin].mid,
                                      "P_in_off_mid": L["pin"]["off"].mid, "P_in_armed_mid": L["pin"]["armed"].mid,
                                      "P_out_off_mid": L["pout"]["off"].mid, "P_out_armed_mid": L["pout"]["armed"].mid,
                                      "in_mid": Iv.mid, "in_lo": Iv.lo, "in_hi": Iv.hi,
                                      "out_mid": None if Ov is None else Ov.mid,
                                      "net_mid": None if Ov is None else Iv.mid - Ov.mid,
                                      "Jbar": L["Jbar"], "beta": L["beta"], "authority": "W5 call7 rad7 mat7 + capture g7"})
            row = {"target": t["name"], "shell": s, "role": role, "code": code,
                   "branch": CODE_NAME[code], "dominant_term": dom, "cancelling": canc,
                   "missing": ";".join(miss)}
            for tag, b in (("E", bE), ("R", bR)):
                if b:
                    for k in ("I_rad", "O_rad", "I_coll", "O_coll", "I", "O", "R", "D", "n_u", "eps"):
                        row[f"{tag}_{k}_mid"] = b[k].mid if b[k] is not None else None
                        row[f"{tag}_{k}_lo"] = b[k].lo if b[k] is not None else None
                        row[f"{tag}_{k}_hi"] = b[k].hi if b[k] is not None else None
            if dec:
                row["dn_u_mid"], row["dn_u_lo"], row["dn_u_hi"] = dec["dn_u"].mid, dec["dn_u"].lo, dec["dn_u"].hi
                for k in TERMS:
                    a_ = dec["A"][k]
                    row[f"A_{k}_mid"], row[f"A_{k}_lo"], row[f"A_{k}_hi"] = ((a_.mid, a_.lo, a_.hi) if a_ is not None
                                                                         else (None, None, None))
                row["sum_mid"], row["sum_lo"], row["sum_hi"] = dec["sum"].mid, dec["sum"].lo, dec["sum"].hi
            budg.append(row)
            closure.append({"target": t["name"], "shell": s, "role": role,
                            "arithmetic_closure": None if dec is None else dec["closes"],
                            "eps_E": ivj(bE["eps"]) if bE else None, "eps_R": ivj(bR["eps"]) if bR else None,
                            "R_E": ivj(bE["R"]) if bE else None, "R_R": ivj(bR["R"]) if bR else None,
                            "lines_counted_once": len({L["row"] for L in entE["lines"]}) == len(entE["lines"])})
            if not closure[-1]["lines_counted_once"]:
                cov[-1]["code"] = row["code"] = 6
                row["branch"] = CODE_NAME[6]
            results[f"{t['name']}:{s}"] = {"role": role, "code": row["code"], "branch": row["branch"],
                                           "dominant_term": dom, "cancelling": canc, "missing": miss}
    pd.DataFrame(cov).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(pops).to_csv(out / "populations.csv", index=False)
    pd.DataFrame(rates).to_csv(out / "rates.csv", index=False)
    pd.DataFrame(budg).to_csv(out / "budget.csv", index=False)
    (out / "closure.json").write_text(json.dumps({
        "schema": "DET_SIIII_RATES_CLOSURE_V1", "cells": closure,
        "production_solution": "PRODUCTION_SOLUTION_NOT_TESTABLE",
        "production_solution_reason": "prereg 5.3: generation-7 capture precedes the final R7; candidate n_e "
                                      "closure, CE coefficients and CE damping history are not stored; the "
                                      "final writer records published arrays only",
        "sl_projection": SE["sl"] + SR["sl"]}, indent=1, default=str))
    (out / "checks.json").write_text(json.dumps({
        "schema": "DET_SIIII_RATES_CHECKS_V1",
        "g_link": {"config_premises_and_e_r_diff": "PASS", "capture_generation": "PASS", "line_list_sealed": "PASS",
                   "w5_identity_beta_ledger": "PASS", "w5_contradictory_duplicates": 0,
                   "sl_projection_checks": len(SE["sl"]) + len(SR["sl"]),
                   "sl_projection_inconsistent": sum(1 for x in SE["sl"] + SR["sl"] if not x["consistent"])},
        "g_rate": {"arithmetic_closure_cells": sum(1 for x in closure if x["arithmetic_closure"] is not None),
                   "arithmetic_closure_failed": sum(1 for x in closure if x["arithmetic_closure"] is False),
                   "lines_counted_once": all(x["lines_counted_once"] for x in closure)},
        "selftest": "see the --selftest selftest.json (sha in manifest.json): real analyze path cases"},
        indent=1))
    per_target = {}
    for t in targets:
        codes = [results[f"{t['name']}:{s}"]["code"] for s in t["main"]]
        per_target[t["name"]] = {"main_shells": t["main"], "codes": codes,
                                 "common": codes[0] if len(set(codes)) == 1 else None}
    res = {"schema": "DET_SIIII_RATES_RESULT_V1", "validity": "VALID", "reasons": [],
           "cells": results, "per_target": per_target, "fault": fault,
           "structural_zero_direct": list(STRUCTURAL_ZERO),
           "claim_ceiling": "generation-7 production-definition rates and the algebraic split of the "
                            "E'-R' density difference; no physical correctness, causal share over six "
                            "iterations, NRB promotion or CMFGEN level comparison"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def cmd_analyze(a):
    out = Path(a.out)
    why = []
    stj = json.load(open(Path(a.selftest) / "selftest.json"))
    if stj.get("readout_sha256") != sha(Path(__file__).resolve()) or stj.get("failures") != 0:
        why.append("SELFTEST_NOT_THIS_READOUT_OR_FAILED")
    if why:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": why}, indent=1))
        print(json.dumps({"validity": "X", "reasons": why}))
        return 2
    res = det_siiii_rates_analyze(a.e_run, a.r_run, out, prior_pop=a.prior_pop)
    src = Path(a.production_source)
    T2_FILES = ("src/lumina_plasma.c", "src/line_net_rate.c", "src/lumina_atomic.c", "src/lumina.h",
                "src/line_jbar.h", "src/lumina_cmfgen.c")
    man = {"schema": "DET_SIIII_RATES_MANIFEST_V1",
           "prereg": "docs/PREREG_DET_SIIII_RATES_2026-10-07.md",
           "readout": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
           "dependency": {"path": str(Path(POP.__file__).resolve()), "sha256": sha(Path(POP.__file__).resolve())},
           "selftest": {"path": str(Path(a.selftest) / "selftest.json"),
                        "sha256": sha(Path(a.selftest) / "selftest.json")},
           "production_source": {f: sha(src / f) for f in T2_FILES},
           "argv": sys.argv, "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
           "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "inputs": {}, "outputs": {}}
    for tag, run in (("E", a.e_run), ("R", a.r_run)):
        for rel in ("input/resolved_lumina.exports", "input/deck.sha256", "input/model/line_list.csv",
                    "input/model/levels.csv", "work/det_obs_capture/manifest.json", "work/det_obs_capture/meta.txt",
                    "work/det_obs_capture/Te.f64", "work/det_obs_capture/ne.f64", "work/w5_lines.csv",
                    "work/nlte_levels_final.csv"):
            p = Path(run) / rel
            man["inputs"][f"{tag}:{rel}"] = {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
        lb = Path(run) / "work/det_obs_capture/lines.bin"
        man["inputs"][f"{tag}:work/det_obs_capture/lines.bin"] = {
            "path": str(lb), "bytes": lb.stat().st_size,
            "identity": "sealed by capture manifest.json (sha recorded above); selected records only"}
    for p in sorted(out.iterdir()):
        man["outputs"][p.name] = sha(p)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"validity": res["validity"], "per_target": res.get("per_target"),
                      "reasons": res.get("reasons")}, default=str))
    return 0 if res["validity"] == "VALID" else 2


# ------------------------------------------------------------------ fixture
FX_LEVELS = [  # level, g, E_eV   (cutoff K=7 -> 7,8,9 lumped members)
    (0, 1, 0.0), (1, 1, 6.537028), (2, 3, 6.552972), (3, 5, 6.585422), (4, 3, 10.276357),
    (5, 5, 15.152669), (6, 1, 16.08), (7, 3, 18.40), (8, 5, 18.47), (9, 3, 19.10)]
FX_LINES = [  # (lo, up, f_lu)
    (0, 2, 2.3e-5), (4, 5, 0.045), (2, 6, 0.02), (2, 7, 0.05), (2, 8, 0.11),
    (3, 5, 1.2e-3), (5, 9, 0.08), (1, 2, 1.0e-6), (2, 5, 3.0e-4)]
FX_SHELLS = [17, 19, 24, 30]


def det_siiii_rates_fx_wavelength(lo, up):
    dE = (FX_LEVELS[up][2] - FX_LEVELS[lo][2]) * EV_TO_ERG
    return H * C / dE                       # cm


def det_siiii_rates_make_fixture(root, pops, w5vals, Te, ne, nrb, drop_w5=None, w5_gen=None, w5_contra=None,
                                 w5_wrong_ion=None, drop_cap=None, cap_gen=7, extra_export=None, ion_channels=False):
    """small run with the production layout: Si III levels 0-9, nine lines (two
    targets + partners incl. one outside the bb domain), shells 17/19/24/30.
    pops: {shell: [n_0..n_9]}; w5vals: {(line, shell): (J_c, J_c_err, tau_eff, L_l, srce_chk)}."""
    root = Path(root)
    (root / "input/model").mkdir(parents=True)
    (root / "work/det_obs_capture").mkdir(parents=True)
    with open(root / "input/model/levels.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level,configuration\n")
        for lvl, g, E in FX_LEVELS:
            f.write(f"14,2,{lvl},{E!r},{g},0,0,x\n")
    with open(root / "input/model/line_list.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,"
                "B_lu,B_ul,A_ul,wavelength_cm\n")
        for k, (lo, up, fl) in enumerate(FX_LINES):
            lam = det_siiii_rates_fx_wavelength(lo, up)
            nu = C / lam
            gl, gu = FX_LEVELS[lo][1], FX_LEVELS[up][1]
            A = 6.6702e15 * gl / gu * fl / (lam * 1e8) ** 2 * 1e16 * 1e-16
            Bul = A * C * C / (2 * H * nu ** 3)
            Blu = Bul * gu / gl
            f.write(f"14,2,{lo},{up},{k},{lam * 1e8!r},0,{fl!r},{nu!r},{Blu!r},{Bul!r},{A!r},{lam!r}\n")
    (root / "input/deck.sha256").write_text(f"{sha(root / 'input/model/line_list.csv')}  line_list.csv\n")
    ex = {"LUMINA_NLTE_COLL_FIX": "1", "LUMINA_NLTE_COLL_FLOOR": "0", "LUMINA_SUPER_LEVELS": "1",
          "LUMINA_SUPER_CUTOFF": "7", "LUMINA_NLTE_SKIP_Z": "", "LUMINA_MODEL_DIR": str(root / "input/model")}
    if nrb:
        ex.update({"LUMINA_BB_NRB": "1", "LUMINA_A210_SPRODUCER_CAPTURE": "1"})
    if extra_export:
        ex.update(extra_export)
    (root / "input/resolved_lumina.exports").write_text("".join(f'declare -x {k}="{v}"\n' for k, v in ex.items()))
    NS = 50
    t_exp = 1683072.0
    (root / "work/det_obs_capture/meta.txt").write_text(
        f"NS={NS}\ntime_explosion={t_exp.hex()}\nmaterial_generation={cap_gen}\nrad_generation=7\n")
    (root / "work/det_obs_capture/manifest.json").write_text(json.dumps(
        {"generations": {"material": cap_gen, "nlte_population_committed": cap_gen}}))
    np.full(NS, Te).tofile(root / "work/det_obs_capture/Te.f64")
    np.full(NS, ne).tofile(root / "work/det_obs_capture/ne.f64")
    recs = []
    for k, (lo, up, fl) in enumerate(FX_LINES):
        lam = det_siiii_rates_fx_wavelength(lo, up)
        for s in FX_SHELLS:
            if drop_cap == (k, s):
                continue
            tau, S, *_ = prod_tau_S(pops[s][lo], pops[s][up], FX_LEVELS[lo][1], FX_LEVELS[up][1], fl, lam, t_exp)
            recs.append((k, s, tau, S, 0.0, 1))
    recs.sort(key=lambda r: (r[0], r[1]))
    np.array(recs, dtype=REC).tofile(root / "work/det_obs_capture/lines.bin")
    with open(root / "work/w5_lines.csv", "w") as f:
        f.write("call,operator,rad_generation,material_generation,shell,line_id,Z,ion,nu_hz,J_c,J_c_err_upper,"
                "beta,L_l,Jbar,Jbar_err_upper,tau_eff,srce_chk,status\n")
        for (k, s), (jc, jce, te, Ll, sc_) in sorted(w5vals.items()):
            if drop_w5 == (k, s):
                continue
            lam = det_siiii_rates_fx_wavelength(FX_LINES[k][0], FX_LINES[k][1])
            nu = C / lam
            beta = det_siiii_rates_exponx(te)
            jb = beta * jc + Ll
            gen = w5_gen if (w5_gen is not None and (k, s) == (0, 24)) else 7
            ion = 3 if w5_wrong_ion == (k, s) else 2
            f.write(f"1,INIT_SHARED_GAUSSIAN,1,0,{s},{k},14,{ion},{nu!r},NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,"
                    f"NOT_APPLICABLE,{jb * 3!r},0,NOT_APPLICABLE,NOT_APPLICABLE,OK_INIT_GAUSSIAN_PROFILE_MEAN\n")
            f.write(f"7,{W5_OP},{gen},{gen},{s},{k},14,{ion},{nu!r},{jc!r},{jce!r},{beta!r},{Ll!r},{jb!r},"
                    f"{beta * jce!r},{te!r},{sc_},OK\n")
            if w5_contra == (k, s):
                f.write(f"7,{W5_OP},7,7,{s},{k},14,{ion},{nu!r},{jc * 2!r},{jce!r},{beta!r},{Ll!r},{jb!r},"
                        f"{beta * jce!r},{te!r},{sc_},OK\n")
    with open(root / "work/nlte_levels_final.csv", "w") as f:
        f.write("Z,ion,shell,level_idx,global_idx,E_eV,g,n_pop,T_e,n_ion_total,population_generation\n")
        for s in FX_SHELLS:
            for lvl, g, E in FX_LEVELS:
                f.write(f"14,2,{s},{lvl},{100 + lvl},{E!r},{g},{pops[s][lvl] * 1.5:.6e},{Te},1e9,8\n")
    if ion_channels:
        (root / "work/lumina_ion_channels.csv").write_text("shell_id,Z,stage,gamma_phot,gamma_nt\n24,14,2,1e3,1e3\n")
    return root


class FI:
    """exact rational interval (selftest truth only)."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo, self.hi = lo, (lo if hi is None else hi)

    def __add__(s, o):
        return FI(s.lo + o.lo, s.hi + o.hi)

    def __sub__(s, o):
        return FI(s.lo - o.hi, s.hi - o.lo)

    def __mul__(s, o):
        c = (s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi)
        return FI(min(c), max(c))

    def div(s, o):
        assert o.lo > 0
        c = (s.lo / o.lo, s.lo / o.hi, s.hi / o.lo, s.hi / o.hi)
        return FI(min(c), max(c))

    def half(s):
        return FI(s.lo / 2, s.hi / 2)


def _fr(v):
    from fractions import Fraction
    return Fraction(v)


def _dfun(x, fn):
    """exp or sqrt of an exact rational at 60 significant digits, bracketed by
    the 60-digit rounding bound (relative 1e-58)."""
    from decimal import Decimal, localcontext
    from fractions import Fraction
    with localcontext() as c:
        c.prec = 60
        d = Decimal(x.numerator) / Decimal(x.denominator)
        v = Fraction(d.exp() if fn == "exp" else d.sqrt())
    e = Fraction(1, 10 ** 58)
    return FI(v * (1 - e), v * (1 + e))


def det_siiii_rates_fx_coeffs(k, s, w5vals, Te, ne):
    """exact truth coefficients of line k at shell s: off / armed (P_lu, P_ul) and
    (C_up, C_down) as rational intervals from the fixture's double inputs."""
    lo, up, fl = FX_LINES[k]
    lam = det_siiii_rates_fx_wavelength(lo, up)
    nu = C / lam
    gl, gu = FX_LEVELS[lo][1], FX_LEVELS[up][1]
    A = 6.6702e15 * gl / gu * fl / (lam * 1e8) ** 2 * 1e16 * 1e-16
    Bul = A * C * C / (2 * H * nu ** 3)
    Blu = Bul * gu / gl
    X = lambda v: FI(_fr(v))                                   # noqa: E731
    if NU_MIN <= nu <= NU_MAX:
        jc, jce, te, Ll, sc_ = w5vals[(k, s)]
        beta = det_siiii_rates_exponx(te)
        J = beta * jc + Ll                                    # the W5 Jbar double (same op order as the writer)
        off = (X(Blu) * X(J), X(A) + X(Bul) * X(J))
        if sc_ == 0:
            ct = X(beta) * X(jc)
            armed = (X(Blu) * ct, X(A) * X(beta) + X(Bul) * ct)
        else:
            armed = off
    else:
        off = armed = (X(0.0), X(A))
    dEe = abs(_fr(FX_LEVELS[up][2]) - _fr(FX_LEVELS[lo][2]))
    x = dEe * _fr(EV_TO_ERG) / (_fr(K_B) * _fr(Te))
    ups = max(_fr(14.5) * _fr(fl) * (_fr(RYD_EV) / dEe) ** 2 * _fr(GBAR), _fr(AXELROD_OMEGA))
    pre = X(ne) * X(COLL_K) * FI(ups)
    sq = _dfun(_fr(Te), "sqrt")
    cu = (pre * _dfun(-x, "exp")).div(FI(_fr(gl)) * sq)
    cd = pre.div(FI(_fr(gu)) * sq)
    return {"off": off, "armed": armed, "coll": (cu, cd), "lo": lo, "up": up}


def det_siiii_rates_fx_state(pops, w5vals, Te, ne, u, s, form):
    """exact truth per-line oriented coefficients and the budget of one state."""
    X = lambda v: FI(_fr(v))                                   # noqa: E731
    lines = {}
    for k, (lo, up, fl) in enumerate(FX_LINES):
        if u not in (lo, up):
            continue
        c = det_siiii_rates_fx_coeffs(k, s, w5vals, Te, ne)
        j = up if lo == u else lo
        if up == u:
            pin = {"off": c["off"][0], "armed": c["armed"][0], "coll": c["coll"][0]}
            pout = {"off": c["off"][1], "armed": c["armed"][1], "coll": c["coll"][1]}
        else:
            pin = {"off": c["off"][1], "armed": c["armed"][1], "coll": c["coll"][1]}
            pout = {"off": c["off"][0], "armed": c["armed"][0], "coll": c["coll"][0]}
        lines[k] = {"n_j": X(pops[s][j]), "pin": pin, "pout": pout}
    n_u = X(pops[s][u])
    I_ = FI(_fr(0))
    D_ = FI(_fr(0))
    for L in lines.values():
        I_ = I_ + L["n_j"] * (L["pin"][form] + L["pin"]["coll"])
        D_ = D_ + L["pout"][form] + L["pout"]["coll"]
    return {"lines": lines, "n_u": n_u, "I": I_, "D": D_, "R": I_ - n_u * D_}


def det_siiii_rates_fx_split(tE, tR):
    """exact truth of the prereg 6.1-6.3 terms (E' off, R' armed)."""
    Db = (tE["D"] + tR["D"]).half()
    nub = (tE["n_u"] + tR["n_u"]).half()
    z = FI(_fr(0))
    fi, fo, di, do, ci, co, pa = z, z, z, z, z, z, z
    for k, LE in tE["lines"].items():
        LR = tR["lines"][k]
        njb = (LE["n_j"] + LR["n_j"]).half()
        for side in ("pin", "pout"):
            form = ((LE[side]["off"] - LE[side]["armed"]) + (LR[side]["off"] - LR[side]["armed"])).half()
            field = ((LE[side]["off"] - LR[side]["off"]) + (LE[side]["armed"] - LR[side]["armed"])).half()
            if side == "pin":
                fi, di = fi + njb * form, di + njb * field
            else:
                fo, do = fo + form, do + field
        ci = ci + njb * (LE["pin"]["coll"] - LR["pin"]["coll"])
        co = co + (LE["pout"]["coll"] - LR["pout"]["coll"])
        Pb = ((LE["pin"]["off"] + LR["pin"]["armed"]) + (LE["pin"]["coll"] + LR["pin"]["coll"])).half()
        pa = pa + Pb * (LE["n_j"] - LR["n_j"])
    return {"NRB_FORM": (fi - nub * fo).div(Db), "RADIATIVE_FIELD": (di - nub * do).div(Db),
            "COLLISION": (ci - nub * co).div(Db), "DIRECT_IONIZATION": FI(_fr(0)),
            "PARTNER_POPULATION": pa.div(Db), "RESIDUAL": FI(_fr(0)) - (tE["R"] - tR["R"]).div(Db),
            "dn_u": tE["n_u"] - tR["n_u"]}


def det_siiii_rates_inside(lo, hi, t):
    """analyzer interval [lo, hi] (floats) contains the exact truth interval t."""
    from fractions import Fraction
    return Fraction(lo) <= t.lo and t.hi <= Fraction(hi)


SELFTEST_REQUIRED = (
    "normal: recovered partner and target populations enclose the truth",
    "normal: E' (off) budget encloses the exact truth I, D, R",
    "normal: R' (armed) budget encloses the exact truth I, D, R",
    "normal: equilibrium shell has R enclosing 0, non-equilibrium shell does not",
    "normal: out-of-domain line carries A_ul only",
    "normal: lumped super-level members pass the within-SL Boltzmann interval check",
    "normal: every split term and dn_u enclose their exact truth (two cells)",
    "normal: E=R gives zero difference and UNRESOLVED",
    "normal: E/R data swap flips dn_u exactly and the split still closes",
    "link: missing J-bar row -> NOT_IDENTIFIABLE, no dominance",
    "link: W5 row of a wrong generation -> NOT_IDENTIFIABLE",
    "link: W5 wrong ion -> run rejected",
    "link: W5 contradictory duplicate -> run rejected",
    "link: missing partner capture record -> NOT_IDENTIFIABLE (generation 8 present, not substituted)",
    "link: ion-ledger channels present do not fill a missing line rate",
    "link: capture declared generation 8 -> run rejected",
    "link: structural-zero premise broken (TOPSTAGE_IV on) -> run rejected",
    "link: super-level member off the Boltzmann ratio -> NOT_IDENTIFIABLE",
    "rate: dropping beta on A_ul is detected against the exact truth",
    "rate: dropping stimulated emission is detected against the exact truth",
    "rate: swapped collision direction is detected against the exact truth",
    "rate: a line counted twice is detected",
    "closure: dropping the residual term breaks the identity",
    "closure: declaring equilibrium on a non-equilibrium state is detected",
    "boundary (analysis path): equality is not dominant",
    "boundary (analysis path): overlapping magnitudes are UNRESOLVED",
    "boundary (analysis path): a term without upper bound -> NOT_IDENTIFIABLE",
    "boundary (analysis path): separated single term is dominant and CANCELLING is flagged",
    "boundary (analysis path, natural): wide J-bar errors leave the branch UNRESOLVED",
    "output: computed zero in an active path, STRUCTURAL_ZERO and missing are distinct",
)


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want, inj=None, reason=None, rc=None):
        ok = bool(got == want)
        res.append({"case": name, "pass": ok, "got": str(got)[:300], "want": str(want)[:300],
                    "injection": inj, "reason": reason, "analysis_validity": rc})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:160]} want {str(want)[:160]}")

    fx_targets = ({"name": "SiIII_1892", "row": 0, "lo": 0, "up": 2, "main": [24, 30]},
                  {"name": "SiIII_2542", "row": 1, "lo": 4, "up": 5, "main": [19]})
    Te, neE, neR = 9000.0, 3.0e9, 2.0e9
    base = [1.0e8, 4.0e7, 9.0e6, 1.5e7, 2.0e6, 3.0e4, 5.0e3, 2.0e3, 0.0, 0.0]

    def mkpops(scale_u2, scale_u5):
        p = {}
        for s in FX_SHELLS:
            v = list(base)
            v[2] *= scale_u2 * (1 + 0.01 * s)
            v[5] *= scale_u5
            v[8] = v[7] * (5 / 3) * math.exp(-(18.47 - 18.40) * EV_TO_ERG / (K_B * Te))
            v[9] = v[7] * (3 / 3) * math.exp(-(19.10 - 18.40) * EV_TO_ERG / (K_B * Te))
            p[s] = v
        return p

    def mkw5(jscale, sc_flags=None, rel_err=1e-9, zero=None, ll_zero=False):
        w = {}
        for k, (lo, up, fl) in enumerate(FX_LINES):
            lam = det_siiii_rates_fx_wavelength(lo, up)
            if not (NU_MIN <= C / lam <= NU_MAX):
                continue
            for s in FX_SHELLS:
                jc = 1.0e-6 * jscale * (1 + 0.1 * k)
                Ll = 0.0 if ll_zero else 2.0e-6 * jscale * (1 + 0.05 * s)
                if zero == (k, s):
                    jc, Ll = 0.0, 0.0
                w[(k, s)] = (jc, jc * rel_err, 0.5 + 3.0 * k, Ll, (sc_flags or {}).get(k, 0))
        return w

    popsE, popsR = mkpops(3.0, 5.0), mkpops(1.0, 1.0)
    w5E, w5R = mkw5(1.0, {5: 1}), mkw5(0.5, {5: 1})
    tq = det_siiii_rates_fx_state(popsE, w5E, Te, neE, 2, 30, "off")      # equilibrium target at E' shell 30
    popsE[30][2] = float((tq["I"].lo + tq["I"].hi) / 2 / ((tq["D"].lo + tq["D"].hi) / 2))
    T = Path(out) / "fx"

    def run(name, e_kw=None, r_kw=None, fault=None, pE=None, pR=None, wE=None, wR=None, override=None):
        e = det_siiii_rates_make_fixture(T / name / "E", pE or popsE, wE or w5E, Te, neE, False, **(e_kw or {}))
        r = det_siiii_rates_make_fixture(T / name / "R", pR or popsR, wR or w5R, Te, neR, True, **(r_kw or {}))
        o = T / name / "out"
        rr = det_siiii_rates_analyze(e, r, o, targets=fx_targets, fault=fault, term_override=override)
        return rr, o

    def budget(o):
        return pd.read_csv(o / "budget.csv", float_precision="round_trip").set_index(["target", "shell"])

    def truth(name, u, s, pE=None, pR=None, wE=None, wR=None):
        tE = det_siiii_rates_fx_state(pE or popsE, wE or w5E, Te, neE, u, s, "off")
        tR = det_siiii_rates_fx_state(pR or popsR, wR or w5R, Te, neR, u, s, "armed")
        return tE, tR

    def has(row, col, t):
        return det_siiii_rates_inside(row[f"{col}_lo"], row[f"{col}_hi"], t)

    # ---- normal
    rr, o = run("clean")
    B = budget(o)
    Pp = pd.read_csv(o / "populations.csv", float_precision="round_trip")
    okp = all(q.n_j_lo <= (popsE if q.state == "E" else popsR)[int(q.shell)][int(q.j)] <= q.n_j_hi
              for q in Pp.itertuples())
    case(SELFTEST_REQUIRED[0], (rr["validity"], bool(okp), len(Pp) > 0), ("VALID", True, True), rc=rr["validity"])
    for idx, tag in ((1, "E"), (2, "R")):
        good = True
        for (name, u) in (("SiIII_1892", 2), ("SiIII_2542", 5)):
            for s in (19, 24, 30):
                tE, tR = truth(name, u, s)
                t = tE if tag == "E" else tR
                row = B.loc[(name, s)]
                good &= has(row, f"{tag}_I", t["I"]) and has(row, f"{tag}_D", t["D"]) and has(row, f"{tag}_R", t["R"])
        case(SELFTEST_REQUIRED[idx], bool(good), True)
    rE30, rE24 = B.loc[("SiIII_1892", 30)], B.loc[("SiIII_1892", 24)]
    case(SELFTEST_REQUIRED[3], (bool(rE30.E_R_lo <= 0 <= rE30.E_R_hi), bool(rE24.E_R_lo <= 0 <= rE24.E_R_hi)),
         (True, False))
    Rt = pd.read_csv(o / "rates.csv", float_precision="round_trip")
    odl = Rt[(Rt.row == 7) & (Rt.process == "bb_radiative")]
    case(SELFTEST_REQUIRED[4], (len(odl) > 0, bool((odl.domain == False).all()),           # noqa: E712
                                bool((odl.P_out_mid > 0).all()), bool((odl.P_in_off_mid == 0).all())),
         (True, True, True, True))
    sl = json.load(open(o / "closure.json"))["sl_projection"]
    case(SELFTEST_REQUIRED[5], (len(sl) > 0, all(x["consistent"] for x in sl)), (True, True))
    good = True
    for (name, u, s) in (("SiIII_1892", 2, 24), ("SiIII_2542", 5, 19)):
        sp = det_siiii_rates_fx_split(*truth(name, u, s))
        row = B.loc[(name, s)]
        good &= has(row, "dn_u", sp["dn_u"])
        for k in TERMS:
            good &= has(row, f"A_{k}", sp[k])
    case(SELFTEST_REQUIRED[6], bool(good), True)
    neq = det_siiii_rates_make_fixture(T / "same" / "E", popsE, w5E, Te, neE, False)
    nrq = det_siiii_rates_make_fixture(T / "same" / "R", popsE, w5E, Te, neE, False,
                                       extra_export={"LUMINA_BB_NRB": "1"})
    rsame = det_siiii_rates_analyze(neq, nrq, T / "same" / "out", targets=fx_targets)
    row = budget(T / "same" / "out").loc[("SiIII_1892", 24)]
    case(SELFTEST_REQUIRED[7], (rsame["validity"], float(row.dn_u_mid), int(row.code)), ("VALID", 0.0, 5),
         rc=rsame["validity"])
    e2 = det_siiii_rates_make_fixture(T / "swap" / "E", popsR, w5R, Te, neR, False)
    r2 = det_siiii_rates_make_fixture(T / "swap" / "R", popsE, w5E, Te, neE, True)
    rr3 = det_siiii_rates_analyze(e2, r2, T / "swap" / "out", targets=fx_targets)
    r1, r3 = B.loc[("SiIII_1892", 24)], budget(T / "swap" / "out").loc[("SiIII_1892", 24)]
    clos = r3.sum_lo <= r3.dn_u_hi and r3.dn_u_lo <= r3.sum_hi
    case(SELFTEST_REQUIRED[8], (rr3["validity"], float(r3.dn_u_mid) == -float(r1.dn_u_mid), bool(clos),
                                float(r1.dn_u_mid) != 0.0), ("VALID", True, True, True), rc=rr3["validity"])
    # ---- link / missing
    rr, o = run("noJ", r_kw={"drop_w5": (4, 24)})
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[9], (rr["validity"], c["code"], c["dominant_term"], "MISSING_LINE_FIELD" in ";".join(c["missing"])),
         ("VALID", 6, None, True), inj="drop W5 (line 4, shell 24) in R'", reason=";".join(c["missing"]), rc=rr["validity"])
    rr, o = run("gen6", e_kw={"w5_gen": 6})
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[10], (c["code"], "W5_GENERATION_MISMATCH" in ";".join(c["missing"])), (6, True),
         inj="W5 rad/material generation 6 for (line 0, shell 24)", reason=";".join(c["missing"]), rc=rr["validity"])
    rr, o = run("wion", e_kw={"w5_wrong_ion": (2, 24)})
    case(SELFTEST_REQUIRED[11], (rr["validity"], any("LINK_W5_IDENTITY" in x for x in rr["reasons"])), ("X", True),
         inj="W5 ion 3", reason=";".join(rr["reasons"])[:200], rc=rr["validity"])
    rr, o = run("contra", r_kw={"w5_contra": (3, 30)})
    case(SELFTEST_REQUIRED[12], (rr["validity"], any("CONTRADICTORY" in x for x in rr["reasons"])), ("X", True),
         inj="contradictory duplicate W5 row", reason=";".join(rr["reasons"])[:200], rc=rr["validity"])
    rr, o = run("nocap", r_kw={"drop_cap": (2, 24)})
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[13], (c["code"], "MISSING_G7_POPULATION" in ";".join(c["missing"])), (6, True),
         inj="drop capture (line 2, shell 24) in R'; generation-8 table complete", reason=";".join(c["missing"]),
         rc=rr["validity"])
    rr, o = run("ionled", r_kw={"drop_w5": (3, 24), "ion_channels": True})
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[14], c["code"], 6, inj="drop W5 + ion channels present", reason=";".join(c["missing"]),
         rc=rr["validity"])
    rr, o = run("cap8", e_kw={"cap_gen": 8})
    case(SELFTEST_REQUIRED[15], (rr["validity"], any("GENERATION" in x for x in rr["reasons"])), ("X", True),
         inj="capture generation 8", reason=";".join(rr["reasons"])[:200], rc=rr["validity"])
    rr, o = run("topiv", r_kw={"extra_export": {"LUMINA_TOPSTAGE_IV": "1"}})
    case(SELFTEST_REQUIRED[16], (rr["validity"], any("TOPSTAGE_IV" in x for x in rr["reasons"])), ("X", True),
         inj="LUMINA_TOPSTAGE_IV=1", reason=";".join(rr["reasons"])[:200], rc=rr["validity"])
    badsl = mkpops(1.0, 1.0)
    for s_ in FX_SHELLS:
        badsl[s_][8] *= 1.7
    rr, o = run("slbad", pR=badsl)
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[17], (c["code"], "SL_PROJECTION_INCONSISTENT" in ";".join(c["missing"])), (6, True),
         inj="member 8 off the Boltzmann ratio by 1.7", reason=";".join(c["missing"]), rc=rr["validity"])
    # ---- rate faults against the exact truth (real analysis path)
    for idx, fault, tag in ((18, "drop_beta", "R"), (19, "drop_stim", "E"), (20, "flip_coll", "E")):
        rr, o = run(fault, fault=fault)
        row = budget(o).loc[("SiIII_1892", 24)]
        tE, tR = truth("SiIII_1892", 2, 24)
        t = tR if tag == "R" else tE
        detected = not (has(row, f"{tag}_I", t["I"]) and has(row, f"{tag}_D", t["D"]))
        case(SELFTEST_REQUIRED[idx], bool(detected), True, inj=fault, reason=f"{tag} budget vs exact truth",
             rc=rr["validity"])
    rr, o = run("dup", fault="dup_line")
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[21], c["code"], 6, inj="dup_line", reason=";".join(c["missing"]), rc=rr["validity"])
    rr, o = run("nores", fault="drop_residual")
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[22], (c["code"], "ARITHMETIC_CLOSURE_FAILED" in c["missing"]), (6, True),
         inj="drop_residual", reason=";".join(c["missing"]), rc=rr["validity"])
    rr, o = run("deq", fault="declare_eq")
    row = budget(o).loc[("SiIII_1892", 24)]
    tE, _ = truth("SiIII_1892", 2, 24)
    case(SELFTEST_REQUIRED[23], (not has(row, "E_R", tE["R"]), int(row.code)), (True, 6),
         inj="declare_eq", reason="R forced to 0", rc=rr["validity"])
    # ---- boundaries through the real decision and output path
    X = I.x
    key = "SiIII_1892:24"
    base_A = {"NRB_FORM": X(2.0), "RADIATIVE_FIELD": X(1.0), "COLLISION": X(0.5), "DIRECT_IONIZATION": X(0.0),
              "PARTNER_POPULATION": X(0.5), "RESIDUAL": X(0.0)}
    for idx, nm, A_, dn, want in (
            (24, "eq", base_A, X(4.0), 4),
            (25, "ov", dict(base_A, NRB_FORM=I(1.5, 2.2, 2.6)), I(3.4, 4.2, 4.7), 5),
            (26, "nb", dict(base_A, COLLISION=None), X(4.0), 6),
            (27, "dom", dict(base_A, NRB_FORM=X(10.0), RADIATIVE_FIELD=X(-1.0)), X(10.0), 0)):
        rr, o = run("bd_" + nm, override={key: {"A": A_, "dn_u": dn}})
        c = rr["cells"][key]
        got = (c["code"], c["dominant_term"], c["cancelling"]) if idx == 27 else c["code"]
        exp_ = (0, "NRB_FORM", True) if idx == 27 else want
        bud = budget(o).loc[("SiIII_1892", 24)]
        case(SELFTEST_REQUIRED[idx], (got, int(bud.code)), (exp_, want),
             inj=f"term override after closure ({nm})", reason=c["branch"], rc=rr["validity"])
    wwE, wwR = mkw5(1.0e6, {5: 1}, rel_err=1.0, ll_zero=True), mkw5(0.5e6, {5: 1}, rel_err=1.0, ll_zero=True)
    rr, o = run("wideJ", wE=wwE, wR=wwR)
    c = rr["cells"]["SiIII_1892:24"]
    case(SELFTEST_REQUIRED[28], (rr["validity"], c["code"]), ("VALID", 5),
         inj="pumping-dominated J (scale 1e6), L_l = 0, J_c_err = J_c on every line",
         reason=c["branch"], rc=rr["validity"])
    zE, zR = mkw5(1.0, {5: 1}, zero=(0, 24)), mkw5(0.5, {5: 1}, zero=(0, 24))
    rr, o = run("zeroJ", wE=zE, wR=zR)
    Rz = pd.read_csv(o / "rates.csv", float_precision="round_trip")
    zr = Rz[(Rz.row == 0) & (Rz.shell == 24) & (Rz.target == "SiIII_1892") & (Rz.process == "bb_radiative")]
    covz = pd.read_csv(o / "coverage.csv")
    cz = covz[(covz.target == "SiIII_1892") & (covz.shell == 24)].iloc[0]
    covm = pd.read_csv(T / "noJ" / "out" / "coverage.csv")
    cm = covm[(covm.target == "SiIII_1892") & (covm.shell == 24)].iloc[0]
    case(SELFTEST_REQUIRED[29],
         (len(zr) == 2, bool((zr.status == "OK").all()), bool((zr.in_mid == 0.0).all()), bool((zr.P_in_mid == 0.0).all()),
          "MISSING" in str(cz.missing), "photoionization" in str(cz.structural_zero), int(cz.code) != 6,
          "MISSING_LINE_FIELD" in str(cm.missing)),
         (True, True, True, True, False, True, True, True), inj="J_c = L_l = 0 for (line 0, shell 24)",
         reason="computed zero rate row vs structural_zero column vs missing", rc=rr["validity"])
    names = [r["case"] for r in res]
    missing = [n for n in SELFTEST_REQUIRED if n not in names]
    fails = sum(1 for r in res if not r["pass"]) + len(missing)
    summary = {"schema": "DET_SIIII_RATES_SELFTEST_V2", "readout_sha256": sha(Path(__file__).resolve()),
               "dependency_sha256": sha(Path(POP.__file__).resolve()),
               "production_source": str(a.production_source), "cases": res, "missing_required": missing,
               "n_cases": len(res), "failures": fails,
               "truth": "exact rational arithmetic on the fixture doubles; exp/sqrt at 60 digits bracketed 1e-58; "
                        "comparisons are interval containment without added tolerance"}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1))
    print(f"selftest: {len(res) - fails + len(missing)}/{len(res)} pass, missing {len(missing)}, failures {fails}")
    return 0 if fails == 0 else 1


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--production-source", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("analyze")
    for k in ("--e-run", "--r-run", "--production-source", "--prior-pop", "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit(cmd_selftest(a) if a.cmd == "selftest" else cmd_analyze(a))


if __name__ == "__main__":
    main()
