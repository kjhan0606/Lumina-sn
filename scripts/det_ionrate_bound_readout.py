#!/usr/bin/env python3
"""DET-IONRATE-BOUND readout (prereg docs/PREREG_DET_IONRATE_BOUND_2026-10-10.md).

Stored O state (coll_presc_20261009/OMEGA/run1, material generation 13) against the public toy06 CMFGEN state at
19.480 d for Si, Fe, Co, Ni II<->III at shells 0-49 (200 pairs x 7 rows = 1400 rows):
  ION_RATIO                     x = f_III / f_II (PUBLIC)
  PHOTOIONIZATION / NONTHERMAL_IONIZATION / COLLISIONAL_IONIZATION
                                O recorded value where the stored channel ledger identifies the observed sweep
                                (O_RECORDED); the CMFGEN denominator is NOT_IDENTIFIABLE (prereg 5)
  RADIATIVE_RECOMBINATION       common-atom Milne coefficient (deck sigma_bf, III ground core) at the O and the public
                                (T_e, n_e): d = log10(n_e^O a(T^O) / n_e^C a(T^C)) (COMMON_ATOM)
  DIELECTRONIC_RECOMBINATION    common-atom DR_TABLE coefficient (first match, multiplier 1) likewise (COMMON_ATOM)
  TOTAL_IONIZATION_REQUIRED     U_req^C* = x^C (R_M^C* + R_D^C*) when RR/DR separation is admissible
                                (BALANCE_CONDITIONAL); d_U only when the three O ionisation channels are identified
Codes: 0 negative dex, 1 positive dex, 4 interval contains 0, 5 undefined; X inadmissible.

Subcommands
  analyze --out DIR      (runs the direct negative controls first; Slurm a10 only)
Exit codes: 0 VALID, 2 X / refused, 3 existing output path.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_blue_ion_map_readout as BIM                                           # noqa: E402
import det_nrb_judge_readout as NRB                                              # noqa: E402
import det_siiii_pop_readout as POP                                              # noqa: E402
from det_siiii_pop_readout import I, up, dn                                      # noqa: E402

P = Path(__file__).resolve().parent.parent
PREREG = "docs/PREREG_DET_IONRATE_BOUND_2026-10-10.md"
O = Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1")
PC = O / "work/physics_comparison"
MAN, ION, SHELL, SPECTRAL = (PC / f"physics_DET_iter0010.{x}" for x in ("manifest.json", "ion.csv", "shell.csv",
                                                                          "spectral.csv"))
CHANNELS = O / "work/lumina_ion_channels.csv"
DECK = O / "input/model"
LEVELS, IONIZ, SIGMA = DECK / "levels.csv", DECK / "ionization_energies.csv", DECK / "cmfgen_sigma_bf.bin"
PROV = DECK / "DECK_PROVENANCE.json"
DR_SRC = P / "src/lumina_plasma.c"
PUB = P / "data/standart_data1/toy06"
PUBLIC_ION = {14: PUB / "ionfrac_si_toy06_cmfgen.txt", 26: PUB / "ionfrac_fe_toy06_cmfgen.txt",
              27: PUB / "ionfrac_co_toy06_cmfgen.txt", 28: PUB / "ionfrac_ni_toy06_cmfgen.txt"}
PUBLIC_PHYS = PUB / "phys_toy06_cmfgen.txt"
SYM = {14: "si", 26: "fe", 27: "co", 28: "ni"}
ZS = (14, 26, 27, 28)
NS = 50
G_POP, G_RAD, ITER = 13, 12, 10
ROWS = ("ION_RATIO", "PHOTOIONIZATION", "NONTHERMAL_IONIZATION", "COLLISIONAL_IONIZATION",
        "RADIATIVE_RECOMBINATION", "DIELECTRONIC_RECOMBINATION", "TOTAL_IONIZATION_REQUIRED")
# PHOT route identity (driver evidence, prereg 4): the deck's CMFGEN links use only route A of each II ion, whose
# final state is the III ground term (excitation 0.0).  Si II A is Nahar unsmoothed (resonances); Fe/Co/Ni II A are
# "very crude" fits.  Recorded in evidence.json with the file paths and header lines.
PHOT_A = {14: ("/gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/phot_data_A", "MILNE_WITH_RESONANCES"),
          26: ("/gpfs/kjhan/cmfgen_21jun23/atomic/FE/II/19apr23/phot_data_A", "MILNE"),
          27: ("/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/atomic_local/COB/II/19apr23/phot_data_A", "MILNE"),
          28: ("/gpfs/kjhan/cmfgen_21jun23/atomic/NICK/II/19apr23/phot_data_A", "MILNE")}
LINKS = Path("/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/atomic_links.txt")
C_LIGHT, H, K_B, M_E, EV = 2.99792458e10, 6.62607015e-27, 1.380649e-16, 9.1093837015e-28, 1.602176634e-12
LABEL = {0: "BELOW_REFERENCE", 1: "ABOVE_REFERENCE", 4: "UNRESOLVED", 5: "NOT_IDENTIFIABLE"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_of(d):
    if d is None:
        return 5
    return 0 if d.hi < 0.0 else 1 if d.lo > 0.0 else 4


# ------------------------------------------------------------------ atomic coefficients (common atom)
def det_ionrate_bound_sigma(path, n_levels_expected):
    with open(path, "rb") as f:
        magic, ver = np.frombuffer(f.read(8), "<u4")
        n_lev, n_freq = np.frombuffer(f.read(8), "<i4")
        nu_min, nu_max = np.frombuffer(f.read(16), "<f8")
        if magic != 0x434D4644 or ver != 1 or n_lev != n_levels_expected:
            raise ValueError(f"SIGMA_HEADER:{magic:x}:{ver}:{n_lev}:{n_levels_expected}")
        has = np.frombuffer(f.read(int(n_lev)), "i1").astype(int)
        f.read((8 - (int(n_lev) % 8)) % 8)
        sig = np.frombuffer(f.read(), "<f8").reshape(int(n_lev), int(n_freq))
    return has, sig, float(nu_min), float(nu_max), int(n_freq)


class det_ionrate_bound_milne:
    """spontaneous Milne RR coefficient (cm^3 s^-1) to the III ground core over the stored sigma grid (prereg 4):
    sigma piecewise constant per bin, nu^2 exp(-(h nu - chi)/kT) integrated analytically above each threshold;
    vectorised over levels x bins and memoised by temperature."""

    def __init__(self, Z, levels, ioniz, has, sig, edges):
        lv = levels.reset_index(drop=True)
        self.info, self.ok = {}, False
        g3 = lv[(lv.atomic_number == Z) & (lv.ion_number == 2) & (lv.level_number == 0)].g
        ie = ioniz[(ioniz.atomic_number == Z) & (ioniz.ion_number == 1)].ionization_energy_eV
        if len(g3) != 1 or len(ie) != 1:
            self.info["why"] = "III_GROUND_OR_IE_ABSENT"
            return
        self.g3, self.ie = float(g3.iloc[0]), float(ie.iloc[0])
        idx = [i for i in lv.index[(lv.atomic_number == Z) & (lv.ion_number == 1)] if has[i]]
        chi = np.array([(self.ie - float(lv.loc[i, "energy_eV"])) * EV for i in idx])
        keep = chi > 0.0
        self.idx = np.array(idx)[keep]
        self.chi = chi[keep]
        self.g = np.array([float(lv.loc[i, "g"]) for i in self.idx])
        self.sig = sig[self.idx, :]
        self.lo = np.array([e[0] for e in edges])
        self.hi = np.array([e[1] for e in edges])
        self.memo = {}
        self.ok = len(self.idx) > 0
        self.info = {"levels_used": int(len(self.idx)), "g_III_ground": self.g3, "ionization_energy_eV": self.ie}

    def __call__(self, T):
        if T in self.memo:
            return self.memo[T]
        a = H / (K_B * T)
        nut = (self.chi / H)[:, None]
        lo = np.maximum(self.lo[None, :], nut)
        valid = (self.hi[None, :] > nut) & (self.sig > 0.0)
        c = (self.chi / (K_B * T))[:, None]

        def F(nu):
            x = a * nu - c
            return -np.exp(-x) * (nu * nu / a + 2.0 * nu / a ** 2 + 2.0 / a ** 3)
        seg = np.where(valid, self.sig * (F(self.hi[None, :]) - F(lo)), 0.0)
        pref = 4.0 * math.pi / C_LIGHT ** 2 * (H * H / (2.0 * math.pi * M_E * K_B * T)) ** 1.5
        val = float(pref * np.sum(self.g / self.g3 * seg.sum(axis=1)))
        self.memo[T] = val
        return val


def det_ionrate_bound_dr_table(path):
    """the DR_TABLE entries of the fixed source: (Z, ion_recomb) -> first match (production dr_lookup order)."""
    s = Path(path).read_text()
    i = s.index("static const DRCoefficient DR_TABLE[] = {")
    j = s.index("#define DR_N_ENTRIES", i)
    body = re.sub(r"/\*.*?\*/", "", s[i:j], flags=re.S)
    first = {}
    for m in re.finditer(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*(DR_SOURCE_\w+)\s*\}",
                         body):
        Z, ion, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        c = [float(x) for x in m.group(4).replace("\n", " ").split(",") if x.strip()]
        E = [float(x) for x in m.group(5).replace("\n", " ").split(",") if x.strip()]
        if len(c) != n or len(E) != n:
            raise ValueError(f"DR_TABLE_PARSE:{Z}:{ion}")
        first.setdefault((Z, ion), {"c": c, "E": E, "source": m.group(6)})
    return first


def det_ionrate_bound_dr(coef, T):
    """dr_alpha_eval with the empirical multiplier fixed at 1 (prereg 4)."""
    return sum(ci * math.exp(-Ei / T) for ci, Ei in zip(coef["c"], coef["E"]) if -Ei / T >= -700.0) * T ** -1.5


def iv_eval(fn, Tiv, n=5):
    """interval of fn over the temperature interval by dense sampling (no monotonicity assumed) widened outward."""
    vals = [fn(Tiv.lo + (Tiv.hi - Tiv.lo) * k / (n - 1)) for k in range(n)] + [fn(Tiv.mid)]
    lo, hi = min(vals), max(vals)
    return I(dn(lo * (1 - 1e-12)), fn(Tiv.mid), up(hi * (1 + 1e-12)))


# ------------------------------------------------------------------ public blocks
def det_ionrate_bound_phys(path, time_label="19.480"):
    cols, rows = BIM.det_blue_ion_map_public(path, time_label)
    return cols, rows


# ------------------------------------------------------------------ O channel ledger: identify the observed sweep
def det_ionrate_bound_channels(ch, ion, shell):
    """prereg 3/5: the ledger 'gen' is a sweep counter, not the material generation.  A sweep is accepted as the
    generation-13 record only when its Si II->III r equals the dump's stage ratio at every shell (relative 1e-12) and
    exactly one distinct sweep state matches; otherwise every O channel is NOT_IDENTIFIABLE."""
    x = ion.set_index("shell_id")
    tgt = {s: float(x.loc[s, "Z14_stage_q2_cm3"]) / float(x.loc[s, "Z14_stage_q1_cm3"])
           for s in range(NS) if float(x.loc[s, "Z14_stage_q1_cm3"]) > 0.0}
    si = ch[(ch.Z == 14) & (ch.stage == 1)]
    hits = []
    for g, grp in si.groupby("gen"):
        if len(grp) != NS:
            continue
        rr = dict(zip(grp.shell.astype(int), grp.r.astype(float)))
        if all(s in rr and abs(rr[s] - v) <= 1e-12 * abs(v) for s, v in tgt.items()):
            hits.append(int(g))
    rec = {"matching_gens": hits, "zs_in_ledger": sorted(int(z) for z in ch.Z.unique()),
           "stages_in_ledger": sorted(int(z) for z in ch.stage.unique()), "max_gen": int(ch.gen.max())}
    if not hits:
        return None, dict(rec, why="NO_SWEEP_MATCHES_DUMP")
    states = {tuple(si[si.gen == g].sort_values("shell")[["gamma_phot", "gamma_nt", "c_ion"]].values.ravel()) for g in hits}
    if len(states) != 1:
        return None, dict(rec, why="AMBIGUOUS_SWEEPS")
    return si[si.gen == hits[-1]].set_index("shell"), dict(rec, why="")


# ------------------------------------------------------------------ table
def det_ionrate_bound_rows(ion, shell, pub, phys, coeff, chan, pair=(1, 2), phys_dv=0.0):
    """1400 rows; pair = (q_low, q_high) of the dump / public columns (II, III = 1, 2)."""
    out = []
    x = ion.set_index("shell_id")
    sh = shell.set_index("shell_id")
    pcols, prows = phys
    for Z in ZS:
        cols, rows = pub[Z]
        for s in range(NS):
            v = (float(sh.loc[s, "v_inner_cm_s"]) + float(sh.loc[s, "v_outer_cm_s"])) / 2.0 / 1e5
            TO, neO = float(sh.loc[s, "T_e_K"]), float(sh.loc[s, "n_e_cm3"])
            base = {"Z": Z, "shell": s, "pair": f"{['I','II','III','IV'][pair[0]]}<->{['I','II','III','IV','V'][pair[1]]}",
                    "v_O_kms": v, "T_O": TO, "ne_O": neO}
            prow = BIM.det_blue_ion_map_nearest(rows, v)
            frow = BIM.det_blue_ion_map_nearest(prows, v + phys_dv)
            why0 = []
            if prow is None or frow is None:
                why0.append("PUBLIC_RANGE")
            elif prow[0] != frow[0] and phys_dv == 0.0:
                why0.append("PUBLIC_ROWS_DIFFER")
            tot = float(x.loc[s, f"Z{Z}_element_total_cm3"])
            q1, q2 = float(x.loc[s, f"Z{Z}_nlte_q{pair[0]}_cm3"]), float(x.loc[s, f"Z{Z}_nlte_q{pair[1]}_cm3"])
            xO = I.x(q2).div(I.x(q1)) if (tot > 0.0 and q1 > 0.0 and q2 >= 0.0) else None
            if tot <= 0.0:
                why0.append("ELEMENT_TOTAL_ZERO")
            xC = TC = neC = None
            if not why0:
                base.update({"v_public_kms": prow[0], "v_phys_kms": frow[0]})
                f1, f2 = POP.printed(prow[1][cols.index(f"{SYM[Z]}{pair[0]}")]), POP.printed(prow[1][cols.index(f"{SYM[Z]}{pair[1]}")])
                xC = f2.div(f1) if f1.lo > 0.0 else None
                TC = POP.printed(frow[1][pcols.index("temp[K]")])
                neC = POP.printed(frow[1][pcols.index("ne[/cm^3]")])
                base.update({"T_C": TC.mid, "ne_C": neC.mid})
            # 1 ION_RATIO
            d = NRB.det_nrb_judge_log10i(xO.div(xC)) if (xO is not None and xC is not None and xO.lo > 0.0) else None
            r = dict(base, row="ION_RATIO", grade="PUBLIC", O_value=None if xO is None else xO.mid,
                     ref_value=None if xC is None else xC.mid, why=";".join(why0) if d is None else "")
            NRBcols(r, d)
            out.append(r)
            # 2-4 ionisation channels: O recorded only (CMFGEN denominator not identifiable)
            for nm, col in (("PHOTOIONIZATION", "gamma_phot"), ("NONTHERMAL_IONIZATION", "gamma_nt"),
                            ("COLLISIONAL_IONIZATION", "c_ion")):
                ov = None
                if chan is not None and Z == 14 and pair == (1, 2) and s in chan.index:
                    ov = float(chan.loc[s, col])
                r = dict(base, row=nm, grade="O_RECORDED" if ov is not None else "NOT_IDENTIFIABLE", O_value=ov,
                         O_unit="s^-1 per II ion", ref_value=None, why="CMFGEN_DENOMINATOR_NOT_IDENTIFIABLE" +
                         ("" if ov is not None else ";O_CHANNEL_NOT_RECORDED"))
                NRBcols(r, None)
                out.append(r)
            # 5-6 recombination: common atom
            R = {}
            for nm, key in (("RADIATIVE_RECOMBINATION", "RR"), ("DIELECTRONIC_RECOMBINATION", "DR")):
                c = coeff.get((Z, key))
                r = dict(base, row=nm, grade="COMMON_ATOM", why="", coefficient_id=None if c is None else c["id"])
                dd = None
                if c is None or c.get("fn") is None:
                    r.update(grade="NOT_IDENTIFIABLE", why="COEFFICIENT_NOT_IDENTIFIABLE")
                elif why0:
                    r.update(why=";".join(why0))
                else:
                    aO = c["fn"](TO)
                    aC = iv_eval(c["fn"], TC)
                    RO, RC = I.x(neO).__mul__(I.x(aO)), neC.__mul__(aC)
                    R[key] = (RO, RC)
                    r.update(O_value=RO.mid, ref_value=RC.mid, O_unit="s^-1 per III ion", alpha_O=aO, alpha_C=aC.mid,
                             flag=c.get("flag"), why="TOTAL_RECOMB:RR_DR_SPLIT_NOT_IDENTIFIABLE" if c.get("flag") == "TOTAL_RECOMB" else "")
                    dd = NRB.det_nrb_judge_log10i(RO.div(RC)) if RC.lo > 0.0 and RO.lo > 0.0 else None
                NRBcols(r, dd)
                out.append(r)
            # 7 required total ionisation
            r = dict(base, row="TOTAL_IONIZATION_REQUIRED", grade="BALANCE_CONDITIONAL", why="")
            rr, dr = coeff.get((Z, "RR")), coeff.get((Z, "DR"))
            sep_ok = rr is not None and dr is not None and rr.get("flag") == "MILNE" and dr.get("pure_dr", False)
            dU = None
            if not sep_ok:
                r.update(grade="NOT_IDENTIFIABLE", why="RR_DR_SEPARATION_NOT_ADMISSIBLE")
            elif xC is None or "RR" not in R or "DR" not in R:
                r.update(grade="NOT_IDENTIFIABLE", why=";".join(why0) or "REQUIRED_INPUT_ABSENT")
            else:
                Ureq = xC.__mul__(R["RR"][1] + R["DR"][1])
                r.update(ref_value=Ureq.mid, ref_unit="s^-1 per II ion (conditional)")
                if chan is not None and Z == 14 and s in chan.index:
                    U = sum(float(chan.loc[s, k]) for k in ("gamma_phot", "gamma_nt", "c_ion"))
                    r["O_value"] = U
                    dU = NRB.det_nrb_judge_log10i(I.x(U).div(Ureq)) if U > 0.0 else None
                else:
                    r["why"] = "O_TOTAL_NOT_IDENTIFIABLE"
            NRBcols(r, dU)
            out.append(r)
    return out


def NRBcols(r, d):
    for k_ in ("lo", "mid", "hi"):
        r[f"d_{k_}"] = None if d is None else getattr(d, k_)
    r["code"] = code_of(d)
    r["label"] = LABEL[r["code"]]


def det_ionrate_bound_validate(rows, ion, shell, phys, coeff):
    """G-TABLE / G-PHYS on the assembled rows (before writing)."""
    why = []
    keys = [(r["Z"], r["shell"], r["row"]) for r in rows]
    if len(keys) != 1400 or len(set(keys)) != 1400:
        why.append(f"G-TABLE:ROWS:{len(keys)}:{len(set(keys))}")
    sh = shell.set_index("shell_id")
    for r in rows:
        if r["pair"] != "II<->III":
            why.append(f"G-PHYS:PAIR:{r['Z']}:{r['shell']}:{r['pair']}")
            break
    for r in rows:
        if r["row"] in ("RADIATIVE_RECOMBINATION", "DIELECTRONIC_RECOMBINATION") and r.get("O_value") is not None:
            c = coeff[(r["Z"], "RR" if r["row"].startswith("RAD") else "DR")]
            want = float(sh.loc[r["shell"], "n_e_cm3"]) * c["fn"](float(sh.loc[r["shell"], "T_e_K"]))
            if abs(r["O_value"] / want - 1.0) > 1e-12:
                why.append(f"G-PHYS:RATE_FORMULA:{r['Z']}:{r['shell']}:{r['row']}")
        if r["row"] == "TOTAL_IONIZATION_REQUIRED" and r["grade"] == "BALANCE_CONDITIONAL":
            rr, dr = coeff[(r["Z"], "RR")], coeff[(r["Z"], "DR")]
            if not (rr.get("flag") == "MILNE" and dr.get("pure_dr", False)) or dr.get("source") == "DR_SOURCE_NORAD":
                why.append(f"G-PHYS:RR_DR_DOUBLE:{r['Z']}:{r['shell']}")
        if r["row"] in ("PHOTOIONIZATION", "NONTHERMAL_IONIZATION", "COLLISIONAL_IONIZATION"):
            if r.get("ref_value") is not None or r["code"] != 5:
                why.append(f"G-PHYS:CMFGEN_CHANNEL_FILLED:{r['Z']}:{r['shell']}:{r['row']}")
            if r["grade"] == "O_RECORDED" and r.get("O_value") is None:
                why.append(f"G-PHYS:O_CHANNEL_MISSING:{r['Z']}:{r['shell']}:{r['row']}")
            if r["grade"] == "NOT_IDENTIFIABLE" and r.get("O_value") is not None:
                why.append(f"G-PHYS:O_CHANNEL_ZERO_FILLED:{r['Z']}:{r['shell']}:{r['row']}")
        if r.get("v_public_kms") is not None:
            pcols, prows = phys
            fr = BIM.det_blue_ion_map_nearest(prows, r["v_O_kms"])
            if fr is None or fr[0] != r["v_public_kms"] or r.get("v_phys_kms") != fr[0]:
                why.append(f"G-TABLE:PHYS_ROW:{r['Z']}:{r['shell']}")
    return why


def det_ionrate_bound_inputs(man, ch_info):
    why = []
    if (man.get("population_generation"), man.get("radiation_generation"), man.get("iteration")) != (G_POP, G_RAD, ITER):
        why.append(f"G-INPUT:DUMP_GENERATION:{man.get('population_generation')}")
    if ch_info.get("used_as_generation"):
        why.append("G-INPUT:LEDGER_GEN_AS_GENERATION")
    return why


# ------------------------------------------------------------------ coefficients bundle
def det_ionrate_bound_coeffs(levels, ioniz, has, sig, edges, drt):
    out, ev = {}, {}
    for Z in ZS:
        mil = det_ionrate_bound_milne(Z, levels, ioniz, has, sig, edges)
        a0 = mil(10000.0) if mil.ok else None
        out[(Z, "RR")] = {"id": f"MILNE:{SYM[Z]}2:route_A", "fn": mil if (mil.ok and a0 and a0 > 0.0) else None,
                          "flag": PHOT_A[Z][1]}
        ev[f"RR:{Z}"] = dict(mil.info, alpha_10kK=a0, phot_file=PHOT_A[Z][0], flag=PHOT_A[Z][1])
        c = drt.get((Z, 2))
        if c is None:
            out[(Z, "DR")] = {"id": None, "fn": None}
        else:
            src = c["source"]
            # the source header (lumina_plasma.c DR_TABLE comment) labels NORAD as "R-matrix unified RR+DR total":
            # a total coefficient, so RR / DR separation is not admissible for it (prereg 4: TOTAL_RECOMB)
            total = src == "DR_SOURCE_NORAD"
            out[(Z, "DR")] = {"id": f"DR_TABLE:{Z}:2:{src}", "fn": (lambda T, c=c: det_ionrate_bound_dr(c, T)),
                              "flag": "TOTAL_RECOMB" if total else src, "source": src,
                              "pure_dr": src in ("DR_SOURCE_BADNELL", "DR_SOURCE_MAZZOTTA", "DR_SOURCE_AUTOSTRUCT")}
            ev[f"DR:{Z}"] = {"source": src, "flag": "TOTAL_RECOMB" if total else "DR", "n_terms": len(c["c"]),
                             "alpha_10kK": det_ionrate_bound_dr(c, 10000.0)}
    return out, ev


# ------------------------------------------------------------------ direct negative controls (prereg 8)
def det_ionrate_bound_selftest(ctx):
    ion, shell, pub, phys, coeff, chan = ctx
    recs = []

    def case(name, got, want):
        recs.append({"case": name, "got": str(got), "want": str(want), "pass": got == want})
    man_ok = {"population_generation": 13, "radiation_generation": 12, "iteration": 10}
    w_ok = det_ionrate_bound_inputs(man_ok, {})
    w14 = det_ionrate_bound_inputs(dict(man_ok, population_generation=14), {})
    wgen = det_ionrate_bound_inputs(man_ok, {"used_as_generation": True})
    case("1 fraction generation 14 / ledger gen used as material generation refused",
         (w_ok, any("DUMP_GENERATION" in w for w in w14), any("LEDGER_GEN" in w for w in wgen)), ([], True, True))
    good = det_ionrate_bound_rows(ion, shell, pub, phys, coeff, chan)
    v_good = det_ionrate_bound_validate(good, ion, shell, phys, coeff)
    bad_pair = det_ionrate_bound_rows(ion, shell, pub, phys, coeff, chan, pair=(2, 3)) \
        if all(f"Z{Z}_nlte_q3_cm3" in ion.columns for Z in ZS) else \
        [dict(r, pair="III<->IV") for r in good]
    bad_phys = det_ionrate_bound_rows(ion, shell, pub, phys, coeff, chan, phys_dv=728.0)
    case("2 III<->IV pair / physics row bound at the neighbour velocity refused",
         (v_good, any("G-PHYS:PAIR" in w for w in det_ionrate_bound_validate(bad_pair, ion, shell, phys, coeff)),
          any("PHYS_ROW" in w for w in det_ionrate_bound_validate(bad_phys, ion, shell, phys, coeff))), ([], True, True))
    # 3 units: coefficient used as a rate (no n_e) / n_e applied twice
    sh = shell.set_index("shell_id")
    bad_unit = [dict(r, O_value=(r["O_value"] / float(sh.loc[r["shell"], "n_e_cm3"]))) if (
        r["row"] == "RADIATIVE_RECOMBINATION" and r.get("O_value") is not None) else r for r in good]
    bad_ne2 = [dict(r, O_value=(r["O_value"] * float(sh.loc[r["shell"], "n_e_cm3"]))) if (
        r["row"] == "DIELECTRONIC_RECOMBINATION" and r.get("O_value") is not None) else r for r in good]
    case("3 cm^3 s^-1 coefficient as s^-1 / n_e applied twice refused by the registered formula",
         (any("RATE_FORMULA" in w for w in det_ionrate_bound_validate(bad_unit, ion, shell, phys, coeff)),
          any("RATE_FORMULA" in w for w in det_ionrate_bound_validate(bad_ne2, ion, shell, phys, coeff))), (True, True))
    # 4 RR+DR total labelled pure DR and summed with a resonant Milne: requires Si (MILNE_WITH_RESONANCES)
    coeff_bad = dict(coeff)
    coeff_bad[(14, "RR")] = dict(coeff[(14, "RR")], flag="MILNE")
    forced = det_ionrate_bound_rows(ion, shell, pub, phys, coeff_bad, chan)
    coeff_tot = dict(coeff)
    coeff_tot[(26, "DR")] = dict(coeff[(26, "DR")], flag="DR_SOURCE_NORAD", pure_dr=True)
    forced_tot = det_ionrate_bound_rows(ion, shell, pub, phys, coeff_tot, chan)
    case("4 RR+DR total (Fe NORAD) entered as pure DR / resonant Milne relabelled pure, summed with Milne, refused",
         (any("RR_DR_DOUBLE:26" in w for w in det_ionrate_bound_validate(forced_tot, ion, shell, phys, coeff)),
          any("RR_DR_DOUBLE:14" in w for w in det_ionrate_bound_validate(forced, ion, shell, phys, coeff))), (True, True))
    # 5 required total placed in the unidentifiable CMFGEN photoionisation cell / missing O channel entered as 0
    filled = [dict(r, ref_value=1.0, code=0) if r["row"] == "PHOTOIONIZATION" else r for r in good]
    zero = [dict(r, O_value=0.0) if (r["row"] == "NONTHERMAL_IONIZATION" and r["grade"] == "NOT_IDENTIFIABLE") else r
            for r in good]
    case("5 required total in the CMFGEN photoionisation cell / missing O channel as 0 refused",
         (any("CMFGEN_CHANNEL_FILLED" in w for w in det_ionrate_bound_validate(filled, ion, shell, phys, coeff)),
          any("ZERO_FILLED" in w for w in det_ionrate_bound_validate(zero, ion, shell, phys, coeff))), (True, True))
    return {"records": recs, "failures": sum(1 for r in recs if not r["pass"])}


# ------------------------------------------------------------------ output
def fmt(v, f):
    return "—" if v is None or (isinstance(v, float) and v != v) else f.format(v)


def det_ionrate_bound_write(out, rows, ch_info, coeff_ev):
    pd.DataFrame(rows).to_csv(out / "cmfgen_comparison.csv", index=False)
    L = ["# DET-IONRATE-BOUND — O(세대 13) 대 공개 CMFGEN(19.48 d), II↔III", "",
         "d = log10(O / 비교값). 코드 0 낮음 · 1 높음 · 4 부호 미분리 · 5 미식별. 등급: PUBLIC · COMMON_ATOM(같은 원자 처방을 "
         "두 열적 상태에서 평가, 실제 CMFGEN 율 아님) · BALANCE_CONDITIONAL(정상 인접단계 평형 조건부) · O_RECORDED · NOT_IDENTIFIABLE.",
         "", f"O 채널 장부: Z {ch_info.get('zs_in_ledger')}, 일치 sweep {ch_info.get('matching_gens')}, 사유 "
         f"`{ch_info.get('why') or '-'}`.", "", "## 1. 셸별 표 (원소가 있는 셸)", ""]
    for Z in ZS:
        L += [f"### {SYM[Z].capitalize()}", "",
              "| 셸 | v | T_O/T_C | n_e,O/n_e,C | d_x (III/II) | d_RR | d_DR | U_req^C* (s⁻¹) | U^O (s⁻¹) | d_U |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in range(NS):
            rr = {r["row"]: r for r in rows if r["Z"] == Z and r["shell"] == s}
            ir = rr["ION_RATIO"]
            if "ELEMENT_TOTAL_ZERO" in (ir.get("why") or ""):
                continue
            tc, nc = ir.get("T_C"), ir.get("ne_C")
            L.append(f"| {s} | {ir['v_O_kms']:.0f} | {fmt(ir['T_O'] / tc if tc else None, '{:.3f}')} | "
                     f"{fmt(ir['ne_O'] / nc if nc else None, '{:.3f}')} | {fmt(ir.get('d_mid'), '{:+.2f}')} | "
                     f"{fmt(rr['RADIATIVE_RECOMBINATION'].get('d_mid'), '{:+.3f}')} | "
                     f"{fmt(rr['DIELECTRONIC_RECOMBINATION'].get('d_mid'), '{:+.3f}')} | "
                     f"{fmt(rr['TOTAL_IONIZATION_REQUIRED'].get('ref_value'), '{:.3e}')} | "
                     f"{fmt(rr['TOTAL_IONIZATION_REQUIRED'].get('O_value'), '{:.3e}')} | "
                     f"{fmt(rr['TOTAL_IONIZATION_REQUIRED'].get('d_mid'), '{:+.2f}')} |")
        L.append("")
    L += ["## 2. 원자 계수 출처", "", "```", json.dumps(coeff_ev, indent=1, default=str), "```"]
    (out / "cmfgen_comparison.md").write_text("\n".join(L) + "\n")


def cmd_analyze(a):
    out = Path(a.out)
    if out.exists():
        print(f"OUT_EXISTS {out}")
        return 3
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "a10":
        print("G-COST:NOT_A10_SLURM")
        return 2
    out.mkdir(parents=True)
    man = json.loads(MAN.read_text())
    ion = pd.read_csv(ION, float_precision="round_trip")
    shell = pd.read_csv(SHELL, float_precision="round_trip")
    levels = pd.read_csv(LEVELS)
    ioniz = pd.read_csv(IONIZ)
    has, sig, nu_min, nu_max, nfreq = det_ionrate_bound_sigma(SIGMA, len(levels))
    sp = pd.read_csv(SPECTRAL, usecols=["shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz"])
    sp0 = sp[sp.shell_id == 0].sort_values("bin_id")
    edges = list(zip(sp0.nu_lo_Hz.astype(float), sp0.nu_hi_Hz.astype(float)))
    drt = det_ionrate_bound_dr_table(DR_SRC)
    coeff, coeff_ev = det_ionrate_bound_coeffs(levels, ioniz, has, sig, edges, drt)
    pub = {Z: BIM.det_blue_ion_map_public(p) for Z, p in PUBLIC_ION.items()}
    phys = det_ionrate_bound_phys(PUBLIC_PHYS)
    ch = pd.read_csv(CHANNELS)
    chan, ch_info = det_ionrate_bound_channels(ch, ion, shell)
    ev = {"schema": "DET_IONRATE_BOUND_EVIDENCE_V1", "job_id": os.environ.get("SLURM_JOB_ID"),
          "partition": os.environ.get("SLURM_JOB_PARTITION"), "readout_sha256": sha(Path(__file__).resolve()),
          "channel_ledger": ch_info, "coefficients": coeff_ev,
          "sigma": {"n_levels": int(len(levels)), "n_freq": nfreq, "nu_min": nu_min, "nu_max": nu_max,
                    "edges_match_grid": bool(len(edges) == nfreq and abs(edges[0][0] / nu_min - 1) < 1e-12 and
                                             abs(edges[-1][1] / nu_max - 1) < 1e-12)},
          "phot_route_evidence": {"links": str(LINKS), "links_sha256": sha(LINKS),
                                  "files": {str(Z): {"path": PHOT_A[Z][0], "sha256": sha(PHOT_A[Z][0]),
                                                     "final_state": [ln.strip() for ln in open(PHOT_A[Z][0], errors="replace")
                                                                     if "Final state in ion" in ln
                                                                     or "Excitation energy of final state" in ln][:2]}
                                            for Z in ZS}}}
    st = det_ionrate_bound_selftest((ion, shell, pub, phys, coeff, chan))
    ev["selftest"] = st
    if st["failures"]:
        ev["validity"] = "X"
        (out / "evidence.json").write_text(json.dumps(ev, indent=1, default=str))
        print(json.dumps({"validity": "X", "selftest_failures": st["failures"]}))
        return 2
    why = det_ionrate_bound_inputs(man, {})
    if not ev["sigma"]["edges_match_grid"]:
        why.append("G-PHYS:SIGMA_GRID")
    rows = det_ionrate_bound_rows(ion, shell, pub, phys, coeff, chan)
    why += det_ionrate_bound_validate(rows, ion, shell, phys, coeff)
    det_ionrate_bound_write(out, rows, ch_info, coeff_ev)
    ev.update({"validity": "X" if why else "VALID", "disposition": "INADMISSIBLE" if why else "PASS_LIMITED",
               "reasons": why, "n_rows": len(rows),
               "inputs": {str(p): sha(p) for p in [MAN, ION, SHELL, SPECTRAL, CHANNELS, LEVELS, IONIZ, SIGMA, PROV, DR_SRC,
                                                   PUBLIC_PHYS, *PUBLIC_ION.values()]},
               "reused": {m.__name__: sha(Path(m.__file__)) for m in (BIM, NRB, POP)},
               "claim_ceiling": "conditional comparison only: COMMON_ATOM / BALANCE_CONDITIONAL are not CMFGEN rates; "
                                "no causal attribution, production promotion, convergence"})
    (out / "evidence.json").write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps({"validity": ev["validity"], "rows": len(rows), "reasons": why[:10],
                      "ledger": {k: ch_info[k] for k in ("matching_gens", "why")}}))
    return 0 if not why else 2


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("analyze")
    s.add_argument("--out", required=True)
    return cmd_analyze(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
