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
    Rounding accounting (each production operation relative <= u):
      tau = fl(K d) with K = fl(fl(fl(C f) lam) t)      -> d = tau/K (1 +- 2u)  [K computed here identically]
      S = fl(fl(fl(A a) n_u)/d), A, a computed here identically -> a n_u = S d / A (1 +- 4u)
      d = fl(n_l - fl(a n_u))                            -> n_l = d (1 +- u) + a n_u (1 +- u)"""
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
    q = I.rel(S, 0).div(I.rel(A, 0) + I.rel(S, 0))      # S/(A+S) from the consumed S, A computed here
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


def interp(vs, xs_tok, v):
    """linear interpolation in velocity between the two bracketing rows (printed
    tokens -> intervals); None outside support."""
    if not (vs[0] <= v <= vs[-1]):
        return None, None
    k = int(np.searchsorted(vs, v))
    if vs[k] == v:
        return printed(xs_tok[k]), {"v0": vs[k], "v1": vs[k], "w1": 0.0}
    v0, v1 = vs[k - 1], vs[k]
    w = (v - v0) / (v1 - v0)
    x0, x1 = printed(xs_tok[k - 1]), printed(xs_tok[k])
    wI = I(dn(w), w, up(w))
    return x0 + (x1 - x0) * wI, {"v0": v0, "v1": v1, "w1": w}


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
    # generation-8 level table vs atomic levels (index, g, energy)
    for _, r in g8[g8.level_idx.isin([0, 2, 4, 5])].drop_duplicates("level_idx").iterrows():
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
    st = {"g7": {}, "g8": {}, "ions": {}, "geo": {}, "t_exp": t_exp}
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
            tau8, S8, d8, A8, a8 = prod_tau_S(n_l8.mid, n_u8.mid, g_l, g_u, f_lu, lam, t_exp)
            st["g8"][(t["name"], s)] = {"n_l": n_l8, "n_u": n_u8, "tau_mid": tau8, "S_mid": S8,
                                        "q": I.x(a8) * n_u8.div(n_l8) if n_l8.lo > 0 else None, "g_l": g_l, "g_u": g_u}
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
    stj = json.load(open(Path(a.selftest) / "selftest.json"))
    if stj.get("readout_sha256") != sha(Path(__file__).resolve()) or stj.get("failures") != 0:
        why.append("G-PHYS:SELFTEST_NOT_THIS_READOUT_OR_FAILED")
    E, wE = analyze_state(a.e_run, "E")
    R, wR = analyze_state(a.r_run, "R")
    why += wE + wR
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
                   "R_q": r8["q"].j() if r8["q"] else None, "E_S_mid": e8["S_mid"], "R_S_mid": r8["S_mid"],
                   "E_tau_mid": e8["tau_mid"], "R_tau_mid": r8["tau_mid"],
                   "dn_l": dnl.j(), "dn_l_ion_total_term": t1.j(), "dn_l_occupation_term": t2.j(),
                   "dn_l_closure_ok": bool((t1 + t2).lo <= dnl.hi and dnl.lo <= (t1 + t2).hi),
                   "dn_l_branch": term_branch(t1, t2), "dd_ion_total_term": u1.j(), "dd_occupation_term": u2.j(),
                   "dd_branch": term_branch(u1, u2)}
            # generation 7 vs 8 for the same transition (S, tau middle values vs g7 consumed)
            for X, tag in ((E, "E"), (R, "R")):
                g7 = X["g7"][(t["name"], s)]
                if g7["status"] == "RECOVERED" and X["g8"][(t["name"], s)]["q"] is not None:
                    q8 = X["g8"][(t["name"], s)]["q"]
                    row[f"{tag}_generation_q"] = ("GENERATION_DIFFERENCE_RESOLVED"
                                                  if (g7["q"].lo > q8.hi or q8.lo > g7["q"].hi)
                                                  else "GENERATION_DIFFERENCE_UNRESOLVED")
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
    vI = np.array([float(x[0]) for x in rI_]); vP = np.array([float(x[0]) for x in rP])
    i_si2 = hI.index("si2"); i_T = hP.index("temp[K]"); i_ne = hP.index("ne[/cm^3]")
    cmp_rows, close = [], {}
    for t in TARGETS:
        eE, eR, ok = [], [], True
        for s in t["shells"]:
            v = E["geo"][s]
            xc, wv = interp(vI, [x[i_si2] for x in rI_], v)
            Tc, _ = interp(vP, [x[i_T] for x in rP], v)
            nc, _ = interp(vP, [x[i_ne] for x in rP], v)
            row = {"set": t["name"], "shell": s, "v_kms": v, "interp": wv,
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
                    rT = I.x(o["Te8"]).div(Tc); row[f"Te8_over_cmfgen_{tag}"] = rT.j() if rT else None
                if nc:
                    rn = o["ne8"].div(nc); row[f"ne8_over_cmfgen_{tag}"] = rn.j() if rn else None
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
    p = sp.add_parser("selftest"); p.add_argument("--out", required=True)
    p = sp.add_parser("analyze")
    for k in ("--e-run", "--r-run", "--production-source", "--prior-paired", "--public-ion", "--public-phys",
              "--selftest", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
