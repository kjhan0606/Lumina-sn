#!/usr/bin/env python3
"""DET-SIIII-POP readout (prereg docs/PREREG_DET_SIIII_POP_2026-10-07.md).

Subcommands
  selftest --out DIR
  analyze  --e-run DIR --r-run DIR --production-source DIR --prior-paired DIR
           --public-ion TXT --public-phys TXT --selftest DIR --out DIR

Generation 7 (the view the DET-OBS writer consumed): from the captured tau and
S_used of the two Si III transitions the production relations (T2
src/lumina_plasma.c nlte_update_tau_sobolev_with_authority with
src/opacity_publication.c a208_signed_sobolev_counted / a208_line_source_counted)

    d = n_l - (g_l/g_u) n_u,   tau = C_Sob f_lu lam_cm t_exp d,
    S = A (g_l/g_u) n_u / d,   A = 2 h nu^3 / c^2,  nu = c / lam_cm

are inverted for the level ratio and, with tau, the absolute level densities.
Generation 8 (stored final NLTE populations) gives the Si III total, the
internal occupations and the public CMFGEN comparison.  Every number is an
interval [lo, hi] (outward rounding; each production rounding counted as a
relative u) with a round-to-nearest middle value.  Commands write only into a
new --out directory.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

U = 2.0 ** -53
C_SOB, H, C, AMU = 2.6540281e-02, 6.62607015e-27, 2.99792458e10, 1.660539066e-24   # T2 src/lumina.h
TARGETS = ({"name": "SiIII_1892", "row": 908565, "lo": 0, "up": 2, "shells": list(range(22, 31))},
           {"name": "SiIII_2542", "row": 1097363, "lo": 4, "up": 5, "shells": list(range(17, 23))})
READ_SHELLS = list(range(17, 31))
Z, ION = 14, 2
REC = np.dtype([("l", "<u4"), ("s", "<u4"), ("tau", "<f8"), ("S", "<f8"), ("eps", "<f8"), ("m", "u1")])
ABSENT = {"g7_ion_total": "NOT_AVAILABLE_SAME_GENERATION", "g7_ion_fraction": "NOT_AVAILABLE_SAME_GENERATION",
          "g7_lower_density_ionization_split": "NOT_IDENTIFIABLE",
          "cmfgen_level_populations": "NOT_AVAILABLE_PUBLIC_LEVEL_POPULATIONS"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def up(x):
    return math.nextafter(x, math.inf)


def dn(x):
    return math.nextafter(x, -math.inf)


class I:
    """scalar interval with a middle value."""
    __slots__ = ("lo", "mid", "hi")

    def __init__(self, lo, mid, hi):
        self.lo, self.mid, self.hi = lo, mid, hi

    @staticmethod
    def x(v):
        return I(v, v, v)

    @staticmethod
    def rel(v, k):
        """v known to within a relative factor (1 +- k u) (v > 0)."""
        e = k * U * 1.0000001
        return I(dn(v * (1 - e)), v, up(v * (1 + e)))

    def __add__(s, o):
        return I(dn(s.lo + o.lo), s.mid + o.mid, up(s.hi + o.hi))

    def __sub__(s, o):
        return I(dn(s.lo - o.hi), s.mid - o.mid, up(s.hi - o.lo))

    def __mul__(s, o):
        c = (s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi)
        return I(dn(min(c)), s.mid * o.mid, up(max(c)))

    def div(s, o):
        if not o.lo > 0.0:
            return None
        c = (s.lo / o.lo, s.lo / o.hi, s.hi / o.lo, s.hi / o.hi)
        return I(dn(min(c)), s.mid / o.mid, up(max(c)))

    def half(s):
        return I(dn(s.lo * 0.5), s.mid * 0.5, up(s.hi * 0.5))

    def abs(s):
        if s.lo >= 0:
            return s
        if s.hi <= 0:
            return I(-s.hi, -s.mid, -s.lo)
        return I(0.0, abs(s.mid), max(-s.lo, s.hi))

    def j(s):
        return {"lo": s.lo, "mid": s.mid, "hi": s.hi}


def ifsum(items):
    items = list(items)
    return I(dn(math.fsum(i.lo for i in items)), math.fsum(i.mid for i in items), up(math.fsum(i.hi for i in items)))


def printed(tok):
    """interval of a decimal token printed with a fixed number of mantissa digits
    (e.g. %.6e, %.4E): the stored value is within half a unit of the last digit."""
    t = tok.strip()
    v = float(t)
    mant = t.lower().split("e")[0]
    nd = len(mant.split(".")[1]) if "." in mant else 0
    ex = int(t.lower().split("e")[1]) if "e" in t.lower() else 0
    h = 0.5 * 10.0 ** (ex - nd)
    return I(dn(v - h), v, up(v + h))


# ------------------------------------------------------------ production relations
def g8_quantities(n_l, n_u, g_l, g_u, f_lu, lam_cm, t_exp):
    """q, S, d, tau of the stored (printed) generation-8 populations as intervals:
    the exact production relations evaluated on the population intervals with the
    production constants a, K, A (same floats)."""
    a = g_l / g_u
    K = C_SOB * f_lu * lam_cm * t_exp
    nu = C / lam_cm
    A = 2.0 * H * nu * nu * nu / (C * C)
    Y = I.x(a) * n_u
    d = n_l - Y
    return {"q": Y.div(n_l), "d": d, "tau": I.x(K) * d, "S": (I.x(A) * Y).div(d) if d.lo > 0 else None}


def prod_tau_S(n_l, n_u, g_l, g_u, f_lu, lam_cm, t_exp):
    """generation-8 style evaluation in the production operation order (floats)."""
    a = g_l / g_u
    d = n_l - a * n_u
    tau = C_SOB * f_lu * lam_cm * t_exp * d
    nu = C / lam_cm
    A = 2.0 * H * nu * nu * nu / (C * C)
    S = A * a * n_u / d if d != 0.0 else float("nan")
    return tau, S, d, A, a


def recover(tau, S, g_l, g_u, f_lu, lam_cm, t_exp):
    """invert the captured (tau, S_used) for d, a*n_u, n_l, n_u, q and n_u/n_l.
    Target = the stored production populations n_l, n_u (floats).  Production
    (T2 lumina_plasma.c 21428-21450, opacity_publication.c), every operation
    rounded with relative error <= u:
      a = g_l/g_u, K = fl(fl(fl(C f) lam) t), A = fl(...2 h nu^3 / c^2) are
        recomputed here with the identical operation order -> same floats;
      d_f = fl(n_l - fl(a n_u))           => n_l = d_f/(1+e4) + a n_u (1+e5)
      tau = fl(K d_f)                      => d_f = tau/K / (1+e0)
      S = fl(fl(fl(A a) n_u) / d_f)        => a n_u = S d_f / A / ((1+e1)(1+e2)(1+e3))
    so d_f in tau/K (1 +- 2u) (one more u for evaluating tau/K here), a n_u in
    (S/A)(1 +- 2u) d_f (1 +- 4u), n_l in d_f (1 +- u) + a n_u (1 +- u); q = a n_u / n_l
    and n_u = a n_u / a follow by outward interval division."""
    a = g_l / g_u
    K = C_SOB * f_lu * lam_cm * t_exp
    nu = C / lam_cm
    A = 2.0 * H * nu * nu * nu / (C * C)
    if not (tau > 0.0 and S > 0.0 and K > 0.0 and A > 0.0):
        return None
    d = I.rel(tau / K, 3)
    Y = (d * I.rel(S / A, 2)) * I.rel(1.0, 4)          # a n_u
    n_l = (d * I.rel(1.0, 1)) + (Y * I.rel(1.0, 1))
    n_u = Y.div(I.x(a))
    q = Y.div(n_l)
    ratio = n_u.div(n_l)
    return {"d": d, "a_nu": Y, "n_l": n_l, "n_u": n_u, "q": q, "nu_over_nl": ratio, "A": A, "K": K, "a": a}


# ------------------------------------------------------------ inputs
def capture_records(capdir, rows, shells):
    m = np.memmap(Path(capdir) / "lines.bin", dtype=REC, mode="r")
    out = {}
    for row in rows:
        lo = np.searchsorted(m["l"], row, side="left")
        hi = np.searchsorted(m["l"], row, side="right")
        seg = np.array(m[lo:hi])
        for r in seg:
            if int(r["s"]) in shells:
                out[(row, int(r["s"]))] = {"tau": float(r["tau"]), "S": float(r["S"]), "eps": float(r["eps"]),
                                           "mapped": int(r["m"])}
    return out


def read_meta(capdir):
    m = {}
    for ln in open(Path(capdir) / "meta.txt"):
        if "=" in ln:
            k, v = ln.rstrip("\n").split("=", 1)
            m[k] = v
    return m


def line_rows(ll, rows):
    want = set(rows)
    d = pd.read_csv(ll, skiprows=lambda i: i > 0 and (i - 1) not in want, float_precision="round_trip")
    d.index = sorted(want)
    return d


def link_reasons(run, rows_df, levels, deck_line_sha):
    why = []
    for t in TARGETS:
        r = rows_df.loc[t["row"]]
        if (int(r["atomic_number"]), int(r["ion_number"]), int(r["level_number_lower"]),
                int(r["level_number_upper"])) != (Z, ION, t["lo"], t["up"]):
            why.append(f"LINK_LINE_ROW:{t['name']}")
    if sha(Path(run) / "input/model/line_list.csv") != deck_line_sha:
        why.append("LINK_LINE_LIST_NOT_SEALED")
    return why


def deck_sha_of(run, name):
    for ln in open(Path(run) / "input/deck.sha256"):
        p = ln.split()
        if len(p) == 2 and p[1].lstrip("*") == name:
            return p[0]
    return None


def nlte_g8(run, shells):
    """Si III rows of the final NLTE level table (generation 8) for the shells."""
    df = pd.read_csv(Path(run) / "work/nlte_levels_final.csv", dtype={"n_pop": str, "n_ion_total": str})
    df = df[(df.Z == Z) & (df.ion == ION) & (df.shell.isin(shells))]
    return df


def public_block(path, time="19.480"):
    lines = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("#TIME:") and ln.split()[1] == time)
    head = lines[i + 2].lstrip("#").split()
    rows = []
    for ln in lines[i + 3:]:
        if ln.startswith("#") or not ln.strip():
            break
        rows.append(ln.split())
    return head, rows


def interp(vs_tok, xs_tok, v):
    """linear interpolation in velocity between the two bracketing rows.  Both the
    velocity and the quantity tokens are printed values, so the bracket velocities
    are intervals; w = (v - v0)/(v1 - v0) decreases in v0 and in v1, which bounds w.
    None outside support (the bracket is chosen on the printed central values)."""
    vs = np.array([float(x) for x in vs_tok])
    if not (vs[0] <= v <= vs[-1]):
        return None, None
    k = int(np.searchsorted(vs, v))
    if vs[k] == v:
        k = max(k, 1)
    v0, v1 = printed(vs_tok[k - 1]), printed(vs_tok[k])
    if not (v1.lo > v0.hi):
        return None, None
    w = I(dn((v - v0.hi) / (v1.hi - v0.hi)), (v - v0.mid) / (v1.mid - v0.mid), up((v - v0.lo) / (v1.lo - v0.lo)))
    x0, x1 = printed(xs_tok[k - 1]), printed(xs_tok[k])
    return x0 + (x1 - x0) * w, {"v0": vs_tok[k - 1], "v1": vs_tok[k], "x0": xs_tok[k - 1], "x1": xs_tok[k], "w1": w.j()}


# ------------------------------------------------------------ branches
def diff_branch(e, r):
    """generation 7 level-ratio difference: resolved if the intervals are disjoint."""
    if e.lo > r.hi or r.lo > e.hi:
        return "EXCITATION_RATIO_DIFFERENCE"
    return "UNRESOLVED"


def term_branch(t1, t2):
    a1, a2 = t1.abs(), t2.abs()
    tags = []
    if a1.lo > a2.hi:
        tags.append("ION_TOTAL_TERM_LARGER")
    elif a2.lo > a1.hi:
        tags.append("INTERNAL_OCCUPATION_TERM_LARGER")
    else:
        tags.append("UNRESOLVED")
    if (t1.lo > 0 and t2.hi < 0) or (t1.hi < 0 and t2.lo > 0):
        tags.append("CANCELLING")
    return tags


def closer_code(DE, DR, ok):
    if not ok:
        return 3, "NOT_COMPARABLE"
    if DE.hi < DR.lo:
        return 0, "E_CLOSER"
    if DR.hi < DE.lo:
        return 1, "R_CLOSER"
    return 2, "UNRESOLVED"


def symdecomp(N_E, p_E, N_R, p_R):
    """Delta(N p) = dN * mean(p) + mean(N) * dp  (exact identity)."""
    dN, mN = N_E - N_R, (N_E + N_R).half()
    dp, mp = p_E - p_R, (p_E + p_R).half()
    return dN * mp, mN * dp


# ------------------------------------------------------------ analyze
def analyze_state(run, tag):
    cap = Path(run) / "work/det_obs_capture"
    meta = read_meta(cap)
    man = json.load(open(cap / "manifest.json"))
    why = []
    gens = man.get("generations", {})
    if gens.get("nlte_population_committed") != 7 or gens.get("material") != 7:
        why.append(f"{tag}:LINK_CAPTURE_GENERATION")
    t_exp = float.fromhex(meta["time_explosion"])
    rows_df = line_rows(Path(run) / "input/model/line_list.csv", [t["row"] for t in TARGETS])
    levels = pd.read_csv(Path(run) / "input/model/levels.csv")
    lv = levels[(levels.atomic_number == Z) & (levels.ion_number == ION)].set_index("level_number")
    why += [f"{tag}:{x}" for x in link_reasons(run, rows_df, lv, deck_sha_of(run, "line_list.csv"))]
    exports = open(Path(run) / "input/resolved_lumina.exports").read()
    if "LUMINA_FLUOR_ORACLE_X" in exports:
        why.append(f"{tag}:LINK_SOURCE_ORACLE_SET")
    recs = capture_records(cap, [t["row"] for t in TARGETS], set(READ_SHELLS))
    g8 = nlte_g8(run, READ_SHELLS)
    # generation-8 level table vs atomic levels: every shell holds every Si III level
    # exactly once, and index, g and energy agree with levels.csv
    want = sorted(int(x) for x in lv.index)
    for s in READ_SHELLS:
        got = sorted(int(x) for x in g8[g8.shell == s].level_idx)
        if got != want:
            why.append(f"{tag}:LINK_G8_LEVEL_SET:shell{s}")
    for _, r in g8.drop_duplicates("level_idx").iterrows():
        if int(r.level_idx) not in lv.index:
            continue
        a = lv.loc[int(r.level_idx)]
        if int(a.g) != int(r.g) or abs(float(a.energy_eV) - float(r.E_eV)) > 1e-9 * max(1.0, float(r.E_eV)):
            why.append(f"{tag}:LINK_G8_LEVEL_MAP:{int(r.level_idx)}")
    if (g8.population_generation != 8).any():
        why.append(f"{tag}:LINK_G8_GENERATION")
    # Si nuclei: production nlte_pair_total_density = (abund * rho) / (mass_amu * AMU)
    rho = pd.read_csv(Path(run) / "input/model/density.csv").set_index("shell_id")["rho"]
    ab = pd.read_csv(Path(run) / "input/model/abundances.csv").set_index("atomic_number")
    mass = float(pd.read_csv(Path(run) / "input/model/atom_masses.csv").set_index("atomic_number").loc[Z, "mass_amu"])
    stage = pd.read_csv(Path(run) / "work/lumina_ion_pops.csv", dtype={"n_ion": str})
    stage = stage[(stage.Z == Z) & (stage.stage == ION)].set_index("shell_id")
    geo = pd.read_csv(Path(run) / "input/model/geometry.csv").set_index("shell_id")
    Te7 = np.fromfile(cap / "Te.f64", "<f8"); ne7 = np.fromfile(cap / "ne.f64", "<f8")
    ps = pd.read_csv(Path(run) / "work/lumina_plasma_state.csv", dtype={"n_e": str}).set_index("shell_id")
    st = {"g7": {}, "g8": {}, "ions": {}, "geo": {}, "t_exp": t_exp, "why": why}
    if any("LINK_G8_LEVEL_SET" in x for x in why):
        return st, why
    for t in TARGETS:
        r = rows_df.loc[t["row"]]
        g_l, g_u = float(lv.loc[t["lo"], "g"]), float(lv.loc[t["up"], "g"])
        f_lu, lam = float(r["f_lu"]), float(r["wavelength_cm"])
        for s in READ_SHELLS:
            rec = recs.get((t["row"], s))
            if rec is None:
                st["g7"][(t["name"], s)] = {"status": "NOT_CAPTURED"}
            elif rec["mapped"] != 1:
                st["g7"][(t["name"], s)] = {"status": "CAPTURED_UNMAPPED", "tau": rec["tau"], "S": rec["S"]}
            else:
                rv = recover(rec["tau"], rec["S"], g_l, g_u, f_lu, lam, t_exp)
                st["g7"][(t["name"], s)] = ({"status": "RECOVERED", "tau": rec["tau"], "S": rec["S"], **rv}
                                            if rv else {"status": "NOT_RECOVERABLE", "tau": rec["tau"], "S": rec["S"]})
            sub = g8[g8.shell == s].set_index("level_idx")
            n_l8, n_u8 = printed(sub.loc[t["lo"], "n_pop"]), printed(sub.loc[t["up"], "n_pop"])
            q8 = g8_quantities(n_l8, n_u8, g_l, g_u, f_lu, lam, t_exp)
            st["g8"][(t["name"], s)] = {"n_l": n_l8, "n_u": n_u8, **q8, "g_l": g_l, "g_u": g_u}
    for s in READ_SHELLS:
        sub = g8[g8.shell == s]
        Nn = ifsum(printed(x) for x in sub.n_pop)
        Ns = printed(stage.loc[s, "n_ion"])
        NSi = (ab.loc[Z, str(s)] * rho.loc[s]) / (mass * AMU)
        NSi_I = I.rel(NSi, 3)
        st["ions"][s] = {"N_III_NLTE": Nn, "N_III_stage": Ns, "N_Si": NSi_I, "n_levels": int(len(sub)),
                         "x_NLTE": Nn.div(NSi_I), "x_stage": Ns.div(NSi_I),
                         "Te7": float(Te7[s]), "ne7": float(ne7[s]), "Te8": float(ps.loc[s, "T_e"]),
                         "ne8": printed(ps.loc[s, "n_e"])}
        st["geo"][s] = (float(geo.loc[s, "v_inner"]) + float(geo.loc[s, "v_outer"])) / 2.0 / 1.0e5   # km/s
    return st, why


def cmd_analyze(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    why = []
    if getattr(a, "require_selftest", True):
        stj = json.load(open(Path(a.selftest) / "selftest.json"))
        if stj.get("readout_sha256") != sha(Path(__file__).resolve()) or stj.get("failures") != 0:
            why.append("G-PHYS:SELFTEST_NOT_THIS_READOUT_OR_FAILED")
    E, wE = analyze_state(a.e_run, "E")
    R, wR = analyze_state(a.r_run, "R")
    why += wE + wR
    if wE or wR:                         # G-LINK failure: no science output from mislinked inputs
        (out / "result.json").write_text(json.dumps({"schema": "DET_SIIII_POP_RESULT_V1", "validity": "X",
                                                     "reasons": why}, indent=1))
        print(json.dumps({"validity": "X", "reasons": why}))
        return 0
    for f in ("input/model/geometry.csv", "input/model/line_list.csv", "input/model/levels.csv",
              "input/model/density.csv", "input/model/abundances.csv"):
        if sha(Path(a.e_run) / f) != sha(Path(a.r_run) / f):
            why.append(f"G-LINK:E_R_INPUT_DIFFERS:{f}")
    if E["t_exp"] != R["t_exp"]:
        why.append("G-LINK:E_R_EPOCH")
    # generation 7 table
    rows7, res7 = [], {}
    for t in TARGETS:
        res7[t["name"]] = {}
        for s in READ_SHELLS:
            e, r = E["g7"][(t["name"], s)], R["g7"][(t["name"], s)]
            row = {"line": t["name"], "shell": s, "main": s in t["shells"], "generation": 7,
                   "authority": "capture tau,S_used (NLTE_LEVEL via nlte_update_tau_sobolev_with_authority)",
                   "E_status": e["status"], "R_status": r["status"]}
            for k in ("tau", "S"):
                row[f"E_{k}"], row[f"R_{k}"] = e.get(k), r.get(k)
            br = None
            if e["status"] == r["status"] == "RECOVERED":
                for k in ("q", "nu_over_nl", "n_l", "n_u", "d"):
                    row[f"E_{k}"], row[f"R_{k}"] = e[k].j(), r[k].j()
                    dif = e[k] - r[k]
                    row[f"dE_R_{k}"] = dif.j()
                    rat = e[k].div(r[k])
                    row[f"E_over_R_{k}"] = rat.j() if rat else None
                br = diff_branch(e["q"], r["q"])
                row["q_branch"] = br
            if s in t["shells"]:
                res7[t["name"]][s] = br or "NOT_COMPARABLE"
            rows7.append(row)
    # n4/n0 where both lines are recovered in a shell
    for s in READ_SHELLS:
        for X, tag in ((E, "E"), (R, "R")):
            a1, a2 = X["g7"][("SiIII_1892", s)], X["g7"][("SiIII_2542", s)]
            if a1["status"] == a2["status"] == "RECOVERED":
                v = a2["n_l"].div(a1["n_l"])
                rows7.append({"line": "n4_over_n0", "shell": s, "generation": 7, "state": tag, "value": v.j() if v else None})
    # generation 8 tables
    rows8, ions8, dec = [], [], {}
    for t in TARGETS:
        dec[t["name"]] = {}
        for s in READ_SHELLS:
            e8, r8 = E["g8"][(t["name"], s)], R["g8"][(t["name"], s)]
            eI, rI = E["ions"][s]["N_III_NLTE"], R["ions"][s]["N_III_NLTE"]
            pe_l, pr_l = e8["n_l"].div(eI), r8["n_l"].div(rI)
            pe_u, pr_u = e8["n_u"].div(eI), r8["n_u"].div(rI)
            a_ = I.x(e8["g_l"] / e8["g_u"])
            we, wr = pe_l - a_ * pe_u, pr_l - a_ * pr_u
            t1, t2 = symdecomp(eI, pe_l, rI, pr_l)
            u1, u2 = symdecomp(eI, we, rI, wr)
            dnl = e8["n_l"] - r8["n_l"]
            row = {"line": t["name"], "shell": s, "main": s in t["shells"], "generation": 8, "authority": "NLTE_LEVEL",
                   "E_n_l": e8["n_l"].j(), "E_n_u": e8["n_u"].j(), "R_n_l": r8["n_l"].j(), "R_n_u": r8["n_u"].j(),
                   "E_p_l": pe_l.j(), "R_p_l": pr_l.j(), "E_q": e8["q"].j() if e8["q"] else None,
                   "R_q": r8["q"].j() if r8["q"] else None,
                   "dn_l": dnl.j(), "dn_l_ion_total_term": t1.j(), "dn_l_occupation_term": t2.j(),
                   "dn_l_closure_ok": bool((t1 + t2).lo <= dnl.hi and dnl.lo <= (t1 + t2).hi),
                   "dn_l_branch": term_branch(t1, t2), "dd_ion_total_term": u1.j(), "dd_occupation_term": u2.j(),
                   "dd_branch": term_branch(u1, u2)}
            # generation 7 (consumed / recovered) vs generation 8 (stored) for q, S, d, tau
            for X, tag in ((E, "E"), (R, "R")):
                g7 = X["g7"][(t["name"], s)]
                g8q = X["g8"][(t["name"], s)]
                if g7["status"] != "RECOVERED":
                    continue
                g7iv = {"q": g7["q"], "d": g7["d"], "S": I.x(g7["S"]), "tau": I.x(g7["tau"])}
                for k in ("q", "S", "d", "tau"):
                    b8 = g8q.get(k)
                    if b8 is None:
                        row[f"{tag}_generation_{k}"] = "NOT_COMPARABLE"
                        continue
                    row[f"{tag}_g8_{k}"] = b8.j()
                    row[f"{tag}_generation_{k}"] = ("GENERATION_DIFFERENCE_RESOLVED"
                                                    if (g7iv[k].lo > b8.hi or b8.lo > g7iv[k].hi)
                                                    else "GENERATION_DIFFERENCE_UNRESOLVED")
                    rr = b8.div(g7iv[k]) if g7iv[k].lo > 0 else None
                    row[f"{tag}_g8_over_g7_{k}"] = rr.j() if rr else None
            if not row["dn_l_closure_ok"]:
                why.append(f"G-PHYS:G8_DECOMP_CLOSURE:{t['name']}:{s}")
            if s in t["shells"]:
                dec[t["name"]][s] = {"dn_l": row["dn_l_branch"], "dd": row["dd_branch"]}
            rows8.append(row)
    for s in READ_SHELLS:
        for X, tag in ((E, "E"), (R, "R")):
            o = X["ions"][s]
            bad = (o["x_NLTE"] is not None and (o["x_NLTE"].lo > 1.0 or o["x_NLTE"].hi < 0.0))
            ions8.append({"shell": s, "state": tag, "generation": 8, "n_levels": o["n_levels"],
                          "N_III_NLTE": o["N_III_NLTE"].j(), "N_III_stage": o["N_III_stage"].j(), "N_Si": o["N_Si"].j(),
                          "x_III_NLTE": o["x_NLTE"].j() if o["x_NLTE"] else None,
                          "x_III_stage": o["x_stage"].j() if o["x_stage"] else None,
                          "normalization": "POPULATION_NORMALIZATION_INCONSISTENT" if bad else "OK"})
    # public CMFGEN comparison (generation 8 NLTE fraction)
    hI, rI_ = public_block(a.public_ion)
    hP, rP = public_block(a.public_phys)
    vI = [x[0] for x in rI_]; vP = [x[0] for x in rP]
    i_si2 = hI.index("si2"); i_T = hP.index("temp[K]"); i_ne = hP.index("ne[/cm^3]")
    cmp_rows, close = [], {}
    for t in TARGETS:
        eE, eR, ok = [], [], True
        for s in t["shells"]:
            v = E["geo"][s]
            xc, wv = interp(vI, [x[i_si2] for x in rI_], v)
            Tc, wT = interp(vP, [x[i_T] for x in rP], v)
            nc, wn = interp(vP, [x[i_ne] for x in rP], v)
            row = {"set": t["name"], "shell": s, "v_kms": v, "interp": wv, "interp_T": wT, "interp_ne": wn,
                   "x_cmfgen": xc.j() if xc else "OUTSIDE_PUBLIC_VELOCITY_SUPPORT",
                   "T_cmfgen": Tc.j() if Tc else None, "ne_cmfgen": nc.j() if nc else None}
            for X, tag, acc in ((E, "E", eE), (R, "R", eR)):
                o = X["ions"][s]
                row[f"x_{tag}"] = o["x_NLTE"].j() if o["x_NLTE"] else None
                row[f"Te7_{tag}"], row[f"Te8_{tag}"], row[f"ne7_{tag}"] = o["Te7"], o["Te8"], o["ne7"]
                if xc is None or o["x_NLTE"] is None or o["x_NLTE"].hi > 1.0:
                    ok = False
                else:
                    e = (o["x_NLTE"] - xc).abs()
                    acc.append(e)
                    row[f"e_{tag}"] = e.j()
                if Tc:
                    for g_, val in (("7", I.x(o["Te7"])), ("8", I.x(o["Te8"]))):
                        rT = val.div(Tc); row[f"Te{g_}_over_cmfgen_{tag}"] = rT.j() if rT else None
                        row[f"Te{g_}_minus_cmfgen_{tag}"] = (val - Tc).j()
                if nc:
                    for g_, val in (("7", I.x(o["ne7"])), ("8", o["ne8"])):
                        rn = val.div(nc); row[f"ne{g_}_over_cmfgen_{tag}"] = rn.j() if rn else None
                        row[f"ne{g_}_minus_cmfgen_{tag}"] = (val - nc).j()
            cmp_rows.append(row)
        if ok and len(eE) == len(t["shells"]):
            n = float(len(eE))
            DE, DR = ifsum(eE), ifsum(eR)
            DE, DR = I(dn(DE.lo / n), DE.mid / n, up(DE.hi / n)), I(dn(DR.lo / n), DR.mid / n, up(DR.hi / n))
            code, br = closer_code(DE, DR, True)
            close[t["name"]] = {"code": code, "branch": br, "D_E": DE.j(), "D_R": DR.j(),
                                "condition": "generation-8 NLTE level-sum Si III fraction, registered linear velocity interpolation, mean |x - x_CMFGEN|"}
        else:
            close[t["name"]] = {"code": 3, "branch": "NOT_COMPARABLE", "reason": "missing / unsupported / normalization"}
    # outputs
    def wcsv(name, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / name, "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else ("" if v is None else v)) for k, v in r.items()})
    wcsv("levels_g7.csv", rows7)
    wcsv("levels_g8.csv", rows8)
    wcsv("ions_g8.csv", ions8)
    wcsv("cmfgen_compare.csv", cmp_rows)
    checks = {"link_and_physics_reasons": why, "selftest": str(a.selftest)}
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    res = {"schema": "DET_SIIII_POP_RESULT_V1", "readout_sha256": sha(Path(__file__).resolve()),
           "validity": "VALID" if not why else "X", "reasons": why,
           "disposition_cap": "PASS_LIMITED", "absent": ABSENT,
           "g7_q_branch_main_shells": res7, "g8_density_split_main_shells": dec, "cmfgen_closeness": close}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    ins = {}
    for X in (a.e_run, a.r_run):
        for f in ("work/nlte_levels_final.csv", "work/lumina_ion_pops.csv", "work/lumina_plasma_state.csv",
                  "work/det_obs_capture/manifest.json", "input/model/line_list.csv", "input/model/levels.csv",
                  "input/model/density.csv", "input/model/abundances.csv", "input/model/atom_masses.csv",
                  "input/model/geometry.csv"):
            ins[str(Path(X) / f)] = sha(Path(X) / f)
    for f in (a.public_ion, a.public_phys):
        ins[str(f)] = sha(f)
    man = {"schema": "DET_SIIII_POP_MANIFEST_V1", "command": sys.argv, "inputs": ins,
           "production_source": str(a.production_source),
           "production_files": {f: sha(Path(a.production_source) / f) for f in
                                ("src/lumina_plasma.c", "src/opacity_publication.c", "src/lumina.h", "src/cmfgen_final_writer.c")},
           "prior_paired_result": sha(Path(a.prior_paired) / "result.json"),
           "readout_sha256": sha(Path(__file__).resolve()),
           "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"validity": res["validity"], "reasons": why, "g7": res7, "closeness":
                      {k: v["branch"] for k, v in close.items()}}, default=str))
    return 0


# ------------------------------------------------------------ selftest
def make_fixture(root, pops7, pops8, stage_tot, vel_kms, line_rows=None, cap_gen=7, drop_g8=None,
                 dup_g8=None, drop_cap=None, deck_ok=True):
    """small run directory with the production layout: 3 shells, lines 0 (0->2) and
    1 (4->5) of Si III, six Si III levels.  pops7/pops8: {shell: [n_0..n_5]}."""
    root = Path(root)
    (root / "input/model").mkdir(parents=True)
    (root / "work/det_obs_capture").mkdir(parents=True)
    g = [1, 1, 3, 5, 3, 5]
    E = [0.0, 6.5370284255, 6.5529715536, 6.5854219379, 10.2763571337, 15.152669295]
    with open(root / "input/model/levels.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level,configuration\n")
        for i in range(6):
            f.write(f"14,2,{i},{E[i]:.10f},{g[i]},0,0,x\n")
    rows = line_rows or [(14, 2, 0, 2, 2.3e-05, 1.892030164e-05), (14, 2, 4, 5, 0.045, 2.542581244e-05)]
    with open(root / "input/model/line_list.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm\n")
        for k, (z, ion, lo, upv, fl, lam) in enumerate(rows):
            f.write(f"{z},{ion},{lo},{upv},{k},{lam * 1e8!r},0,{fl!r},{C / lam!r},0,0,0,{lam!r}\n")
    dsha = sha(root / "input/model/line_list.csv") if deck_ok else "0" * 64
    (root / "input/deck.sha256").write_text(f"{dsha}  line_list.csv\n")
    (root / "input/resolved_lumina.exports").write_text('declare -x LUMINA_NLTE_SKIP_Z=""\n')
    NS = 3
    (root / "input/model/density.csv").write_text("shell_id,rho\n" + "".join(f"{s_},1e-13\n" for s_ in range(NS)))
    (root / "input/model/abundances.csv").write_text("atomic_number,0,1,2\n14,0.1,0.1,0.1\n")
    (root / "input/model/atom_masses.csv").write_text("atomic_number,mass_amu\n14,28.0855\n")
    with open(root / "input/model/geometry.csv", "w") as f:
        f.write("shell_id,r_inner,r_outer,v_inner,v_outer\n")
        for s_ in range(NS):
            v = vel_kms[s_] * 1e5
            f.write(f"{s_},1e15,2e15,{v - 1e7!r},{v + 1e7!r}\n")
    t_exp = 1683072.0
    (root / "work/det_obs_capture/meta.txt").write_text(f"NS={NS}\ntime_explosion={t_exp.hex()}\n")
    (root / "work/det_obs_capture/manifest.json").write_text(json.dumps(
        {"generations": {"material": cap_gen, "nlte_population_committed": cap_gen}}))
    np.full(NS, 9000.0).tofile(root / "work/det_obs_capture/Te.f64")
    np.full(NS, 1e9).tofile(root / "work/det_obs_capture/ne.f64")
    recs = []
    for k, (z, ion, lo, upv, fl, lam) in enumerate(rows):
        for s_ in range(NS):
            if drop_cap == (k, s_):
                continue
            tau, S, *_ = prod_tau_S(pops7[s_][lo], pops7[s_][upv], g[lo], g[upv], fl, lam, t_exp)
            recs.append((k, s_, tau, S, 0.0, 1))
    np.array(recs, dtype=REC).tofile(root / "work/det_obs_capture/lines.bin")
    with open(root / "work/nlte_levels_final.csv", "w") as f:
        f.write("Z,ion,shell,level_idx,global_idx,E_eV,g,n_pop,T_e,n_ion_total,population_generation\n")
        for s_ in range(NS):
            for i in range(6):
                if drop_g8 == (s_, i):
                    continue
                for _ in range(2 if dup_g8 == (s_, i) else 1):
                    f.write(f"14,2,{s_},{i},{100 + i},{E[i]!r},{g[i]},{pops8[s_][i]:.6e},9000,{stage_tot[s_]:.6e},8\n")
    with open(root / "work/lumina_ion_pops.csv", "w") as f:
        f.write("shell_id,Z,stage,n_ion\n")
        for s_ in range(NS):
            f.write(f"{s_},14,2,{stage_tot[s_]:.6e}\n")
    with open(root / "work/lumina_plasma_state.csv", "w") as f:
        f.write("shell_id,T_e,n_e\n")
        for s_ in range(NS):
            f.write(f"{s_},9000,{1e9:.6e}\n")
    return root


def make_public(root, vels, si2, temp, ne):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    hdr = "#NTIMES:  1\n#NSTAGES:  5\n#TIMES[d]:  19.480\n#\n#TIME:    19.480\n#NVEL: %d\n" % len(vels)
    with open(root / "ion.txt", "w") as f:
        f.write(hdr + "#vel_mid[km/s]           si0           si1           si2           si3           si4\n")
        for v, x in zip(vels, si2):
            f.write(f"    {v:.6E}   1.0000E-099   1.0000E-01   {x:.4E}   1.0000E-02   1.0000E-099\n")
    with open(root / "phys.txt", "w") as f:
        f.write(hdr + "#vel_mid[km/s]       temp[K]        rho[gcc]     ne[/cm^3]   natom[/cm^3]\n")
        for v, T, n in zip(vels, temp, ne):
            f.write(f"    {v:.6E}    {T:.4E}    1.0000E-13    {n:.4E}    1.0000E+09\n")
    return root


REQUIRED = (
    "recovery: known populations are enclosed",
    "recovery: small d (near-inversion) is enclosed",
    "recovery: dropping the stimulated term is detected",
    "recovery: dropping the g ratio is detected",
    "structure: uniform scaling keeps S and scales tau",
    "structure: internal redistribution changes q",
    "decomposition: identity closes, a missing term or a sign flip does not",
    "authority: wrong line row / declared generation 8 / stage total as NLTE sum are rejected",
    "comparison: overlapping distances are unresolved, disjoint ones are not",
    "comparison: zero denominator, outside support, missing main shell",
    "comparison: same state is unresolved and E/R swap swaps the branch",
    "printed tokens: half-unit intervals",
    "analysis path: clean fixture is VALID and encloses the generation-7 population truth",
    "analysis path: NLTE total is the level sum, not the stage array",
    "analysis path: wrong line row is rejected",
    "analysis path: unsealed line list is rejected",
    "analysis path: capture declared generation 8 is rejected",
    "analysis path: missing or duplicated generation-8 level is rejected",
    "analysis path: missing main-shell capture record is NOT_COMPARABLE",
    "analysis path: main shell outside public velocity support is NOT_COMPARABLE",
    "analysis path: closeness branch follows the fixture distances",
)


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want):
        ok = bool(got == want)
        res.append({"case": name, "pass": ok, "got": str(got)[:300], "want": str(want)[:300]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:160]} want {str(want)[:160]}")

    def inside(iv, v):
        return iv is not None and iv.lo <= v <= iv.hi
    f_lu, lam, t_exp, g_l, g_u = 0.045, 2.542581244e-05, 1683072.0, 3.0, 5.0
    n_l, n_u = 241.8166, 0.8020518
    tau, S, d, A, a_ = prod_tau_S(n_l, n_u, g_l, g_u, f_lu, lam, t_exp)
    r = recover(tau, S, g_l, g_u, f_lu, lam, t_exp)
    case("recovery: known populations are enclosed", (inside(r["n_l"], n_l), inside(r["n_u"], n_u)), (True, True))
    n_u2 = n_l * (g_u / g_l) * 0.999999
    tau2, S2, *_ = prod_tau_S(n_l, n_u2, g_l, g_u, f_lu, lam, t_exp)
    r2 = recover(tau2, S2, g_l, g_u, f_lu, lam, t_exp)
    case("recovery: small d (near-inversion) is enclosed", (inside(r2["n_l"], n_l), inside(r2["n_u"], n_u2)), (True, True))
    # faults: a production variant without the stimulated term, or with g ratio 1
    tau_ns = C_SOB * f_lu * lam * t_exp * n_l
    r3 = recover(tau_ns, S, g_l, g_u, f_lu, lam, t_exp)
    tau_g, S_g, *_ = prod_tau_S(n_l, n_u, 1.0, 1.0, f_lu, lam, t_exp)
    r4 = recover(tau_g, S_g, g_l, g_u, f_lu, lam, t_exp)
    case("recovery: dropping the stimulated term is detected", inside(r3["n_l"], n_l) and inside(r3["n_u"], n_u), False)
    case("recovery: dropping the g ratio is detected", inside(r4["n_u"], n_u), False)
    k = 7.0
    tauk, Sk, *_ = prod_tau_S(k * n_l, k * n_u, g_l, g_u, f_lu, lam, t_exp)
    rk = recover(tauk, Sk, g_l, g_u, f_lu, lam, t_exp)
    same_S = rk["q"].lo <= r["q"].hi and r["q"].lo <= rk["q"].hi
    tau_scaled = abs(tauk / tau - k) < 1e-13 * k
    tau_m, S_m, *_ = prod_tau_S(n_l, 2 * n_u, g_l, g_u, f_lu, lam, t_exp)
    rm = recover(tau_m, S_m, g_l, g_u, f_lu, lam, t_exp)
    case("structure: uniform scaling keeps S and scales tau", (same_S, tau_scaled), (True, True))
    case("structure: internal redistribution changes q", diff_branch(rm["q"], r["q"]), "EXCITATION_RATIO_DIFFERENCE")
    NE, NR, pE, pR = I.x(1.2e7), I.x(0.8e7), I.x(2e-5), I.x(3e-5)
    t1, t2 = symdecomp(NE, pE, NR, pR)
    true = NE * pE - NR * pR
    tot = t1 + t2
    miss = t1
    flip = t1 - t2
    case("decomposition: identity closes, a missing term or a sign flip does not",
         (tot.lo <= true.hi and true.lo <= tot.hi, miss.lo <= true.hi and true.lo <= miss.hi,
          flip.lo <= true.hi and true.lo <= flip.hi), (True, False, False))
    # authority negatives on small synthetic tables
    rows_ok = [(14, 2, 0, 2), (14, 2, 4, 5)]
    rows_bad = [(14, 2, 0, 2), (14, 1, 4, 5)]
    def row_check(rows):
        return [t["name"] for t, r in zip(TARGETS, rows) if r != (Z, ION, t["lo"], t["up"])]
    gen_check = (lambda g: g.get("nlte_population_committed") == 7)
    levels_pop = [printed("1.188790e+07"), printed("5.944759e+03"), printed("1.752581e+04")]
    nsum = ifsum(levels_pop)
    stage_total = printed("1.193854e+07")
    stage_as_nlte_rejected = not (stage_total.lo <= nsum.hi and nsum.lo <= stage_total.hi)
    case("authority: wrong line row / declared generation 8 / stage total as NLTE sum are rejected",
         (row_check(rows_ok), row_check(rows_bad), gen_check({"nlte_population_committed": 8}), stage_as_nlte_rejected),
         ([], ["SiIII_2542"], False, True))
    case("comparison: overlapping distances are unresolved, disjoint ones are not",
         (closer_code(I(0.1, 0.12, 0.14), I(0.13, 0.15, 0.17), True)[1],
          closer_code(I(0.1, 0.11, 0.12), I(0.2, 0.21, 0.22), True)[1],
          closer_code(I(0.3, 0.31, 0.32), I(0.2, 0.21, 0.22), True)[1]),
         ("UNRESOLVED", "E_CLOSER", "R_CLOSER"))
    xc, _ = interp(np.array([1000.0, 2000.0]), ["1.0000E-01", "2.0000E-01"], 5000.0)
    case("comparison: zero denominator, outside support, missing main shell",
         (I.x(1.0).div(I(0.0, 0.0, 1.0)), xc, closer_code(None, None, False)[1]), (None, None, "NOT_COMPARABLE"))
    D = I(0.1, 0.11, 0.12)
    case("comparison: same state is unresolved and E/R swap swaps the branch",
         (closer_code(D, D, True)[1], closer_code(I(0.1, 0.11, 0.12), I(0.2, 0.21, 0.22), True)[1],
          closer_code(I(0.2, 0.21, 0.22), I(0.1, 0.11, 0.12), True)[1]), ("UNRESOLVED", "E_CLOSER", "R_CLOSER"))
    p = printed("1.188790e+07")
    case("printed tokens: half-unit intervals", (p.lo < 1.188790e7 < p.hi, abs((p.hi - p.lo) - 10.0) < 1e-6), (True, True))
    # ---- fixture runs through the real analysis path ----
    global TARGETS, READ_SHELLS
    saved = (TARGETS, READ_SHELLS)
    TARGETS = ({"name": "SiIII_1892", "row": 0, "lo": 0, "up": 2, "shells": [1, 2]},
               {"name": "SiIII_2542", "row": 1, "lo": 4, "up": 5, "shells": [0, 1]})
    READ_SHELLS = [0, 1, 2]
    try:
        fx = out / "fx"
        p7E = {s_: [1.0e7, 5e3, 2e4, 3e4, 240.0 + 10 * s_, 0.8 + 0.1 * s_] for s_ in range(3)}
        p7R = {s_: [8.0e6, 5e1, 1e2, 2e2, 2.0 + s_, 1e-4 * (1 + s_)] for s_ in range(3)}
        p8E = {s_: [v * 1.01 for v in p7E[s_]] for s_ in range(3)}
        p8R = {s_: [v * 0.99 for v in p7R[s_]] for s_ in range(3)}
        totE = [math.fsum(p8E[s_]) * 1.5 for s_ in range(3)]          # stage array deliberately != level sum
        totR = [math.fsum(p8R[s_]) * 2.0 for s_ in range(3)]
        vel = [18000.0, 20000.0, 22000.0]
        NSi = (0.1 * 1e-13) / (28.0855 * AMU)
        xE = [math.fsum(p8E[s_]) / NSi for s_ in range(3)]
        xR = [math.fsum(p8R[s_]) / NSi for s_ in range(3)]
        pub = make_public(fx / "pub", [15000.0, 19000.0, 21000.0, 25000.0],
                          [xE[0] * 0.98, xE[1] * 0.97, xE[2] * 0.99, xE[2] * 0.99], [9000.0] * 4, [1e9] * 4)
        prior = fx / "prior"; prior.mkdir(parents=True); (prior / "result.json").write_text("{}")

        def run(tag, e_kw=None, r_kw=None, pub_dir=pub):
            e = make_fixture(fx / tag / "E", p7E, p8E, totE, vel, **(e_kw or {}))
            r = make_fixture(fx / tag / "R", p7R, p8R, totR, vel, **(r_kw or {}))
            ns = argparse.Namespace(e_run=str(e), r_run=str(r), production_source=str(a.production_source),
                                    prior_paired=str(prior), public_ion=str(pub_dir / "ion.txt"),
                                    public_phys=str(pub_dir / "phys.txt"), selftest="", out=str(fx / tag / "out"),
                                    require_selftest=False)
            cmd_analyze(ns)
            return json.load(open(fx / tag / "out/result.json")), fx / tag / "out"
        r0, o0 = run("clean")
        enclosed = True
        for row in csv.DictReader(open(o0 / "levels_g7.csv")):
            if row.get("E_status") != "RECOVERED":
                continue
            t = next(t_ for t_ in TARGETS if t_["name"] == row["line"])
            s_ = int(row["shell"])
            for st_, pp in (("E", p7E), ("R", p7R)):
                for key, lev in (("n_l", t["lo"]), ("n_u", t["up"])):
                    iv = json.loads(row[f"{st_}_{key}"])
                    enclosed &= iv["lo"] <= pp[s_][lev] <= iv["hi"]
        case("analysis path: clean fixture is VALID and encloses the generation-7 population truth",
             (r0.get("validity"), enclosed), ("VALID", True))
        nl_ok = True
        for row in csv.DictReader(open(o0 / "ions_g8.csv")):
            s_ = int(row["shell"]); pp = p8E if row["state"] == "E" else p8R
            iv = json.loads(row["N_III_NLTE"])
            nl_ok &= iv["lo"] <= math.fsum(float(f"{v:.6e}") for v in pp[s_]) <= iv["hi"]
            st_tot = json.loads(row["N_III_stage"])
            nl_ok &= not (st_tot["lo"] <= iv["hi"] and iv["lo"] <= st_tot["hi"])
        case("analysis path: NLTE total is the level sum, not the stage array", nl_ok, True)
        rb, _ = run("badrow", e_kw={"line_rows": [(14, 2, 0, 2, 2.3e-05, 1.892030164e-05), (14, 1, 4, 5, 0.045, 2.542581244e-05)]})
        case("analysis path: wrong line row is rejected", any("LINK_LINE_ROW" in x for x in rb["reasons"]), True)
        rs, _ = run("unsealed", r_kw={"deck_ok": False})
        case("analysis path: unsealed line list is rejected", any("LINK_LINE_LIST_NOT_SEALED" in x for x in rs["reasons"]), True)
        rg, _ = run("gen8", e_kw={"cap_gen": 8})
        case("analysis path: capture declared generation 8 is rejected", any("LINK_CAPTURE_GENERATION" in x for x in rg["reasons"]), True)
        rm_, _ = run("misslev", e_kw={"drop_g8": (1, 3)})
        rd_, _ = run("duplev", r_kw={"dup_g8": (0, 5)})
        case("analysis path: missing or duplicated generation-8 level is rejected",
             (any("LINK_G8_LEVEL_SET" in x for x in rm_["reasons"]), any("LINK_G8_LEVEL_SET" in x for x in rd_["reasons"])),
             (True, True))
        rc_, _ = run("misscap", e_kw={"drop_cap": (0, 1)})
        case("analysis path: missing main-shell capture record is NOT_COMPARABLE",
             rc_.get("g7_q_branch_main_shells", {}).get("SiIII_1892", {}).get("1"), "NOT_COMPARABLE")
        pub_narrow = make_public(fx / "pub_narrow", [15000.0, 19000.0, 21000.0], [0.5] * 3, [9000.0] * 3, [1e9] * 3)
        rv_, _ = run("outside", pub_dir=pub_narrow)
        case("analysis path: main shell outside public velocity support is NOT_COMPARABLE",
             rv_.get("cmfgen_closeness", {}).get("SiIII_1892", {}).get("branch"), "NOT_COMPARABLE")
        case("analysis path: closeness branch follows the fixture distances",
             (r0["cmfgen_closeness"]["SiIII_1892"]["branch"], r0["cmfgen_closeness"]["SiIII_2542"]["branch"]),
             ("E_CLOSER", "E_CLOSER"))
    finally:
        TARGETS, READ_SHELLS = saved
    names = {r_["case"] for r_ in res}
    case("battery covers every required case", sorted(set(REQUIRED) - names), [])
    nf = sum(1 for r_ in res if not r_["pass"])
    (out / "selftest.json").write_text(json.dumps({"readout_sha256": sha(Path(__file__).resolve()), "n_cases": len(res),
                                                   "failures": nf, "cases": res}, indent=1))
    print(f"DET-SIIII-POP selftest: {len(res) - nf}/{len(res)} -> {'PASS' if nf == 0 else 'FAIL'}")
    return 0 if nf == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("selftest"); p.add_argument("--out", required=True); p.add_argument("--production-source", required=True)
    p = sp.add_parser("analyze")
    for k in ("--e-run", "--r-run", "--production-source", "--prior-paired", "--public-ion", "--public-phys",
              "--selftest", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
