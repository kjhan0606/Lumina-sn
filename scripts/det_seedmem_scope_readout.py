#!/usr/bin/env python3
"""DET-SEEDMEM-SCOPE readout (prereg docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md, revision 1).

Offline only: no production change, no production run, GPU 0.  Fixed sample = the Si III / S III cells of the
diag6 census (W5 shells 10,15,19,24,30; E' call-2 beta < 1e-3), with the Si III 1206.50 A control and the
line 471957 / shell 10 counterexample.  Computed from stored W5 ledgers, generation-labelled physics dumps,
the generation-7 capture and the deck, with the sealed production pure functions (line_net_rate.c and
opacity_publication.c built unmodified into a CPU shared library: NRB bracket, EXPONX, Sobolev material and
radiation, signed Sobolev tau).  Super-level solve units follow the run's effective knobs (deck f_to_s, the
LUMINA_SUPER_CUTOFF override min(level_number, K), LUMINA_SUPER_LEVELS) checked against the run stdout.
Closeout round-1 repair (astra B1-B5): solve-unit authority, forward ledger reconstruction with the production
material / radiation / tau functions, W5 error propagation in P2, per-combination production rates, X kept apart
from science codes, registered validators that reject injected faults, aggregates inside this readout:
  identifiability.csv   per run / key / call or generation: what can be identified, with reasons
  model.csv             conditional two-level source applicability (E', X off form; R' NRB form), calls 3-7
  cross_substitution.csv E'/X fixed-background exchange of the local emission term L (calls 2-6)
  pathways.csv          generation-7 bound-bound budgets of the target and its connected same-ion transitions
  stationarity.csv      P2 measures (ion sums, n_e, J-bar, recoverable level populations), adjacent generations
  next_contract.json    decision table for the next NRB contract (prereg 5.4)
  summary.json          the aggregates the verdict quotes (set x solve-unit class, shell denominators, residual
                        distributions, transition-set counts), computed here from the files above
  result.json / manifest.json / sample.json
Codes per proposition: 0 SURVIVES, 1 REFUTED, 4 UNRESOLVED, 5 NOT_IDENTIFIABLE, X inadmissible.  1e-3 and
0.0043 dex are registered comparison lines (prereg 4.2, 5.3), never numerical tolerances.

Subcommands
  selftest --production-source SRC --out DIR
  analyze  --prereg DOC --census CSV --e-run E --r-run R --x-run X --production-source SRC --selftest DIR --out DIR
"""
import argparse
import ctypes
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_siiii_pop_readout as POP                                              # noqa: E402
import det_siiii_rates_readout as RATES                                          # noqa: E402
import det_siiii_1206mech_readout as MECH                                        # noqa: E402
import det_siiii_repair_readout as REP                                           # noqa: E402
from det_siiii_pop_readout import I, sha, up, dn                                 # noqa: E402

P = Path(__file__).resolve().parent.parent
PREREG = "docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md"
SHELLS = (10, 15, 19, 24, 30)
TARGET = {"Z": (14, 16), "ion": 2, "beta_max": 1e-3, "counts": {14: 86, 16: 424}, "control": 600693,
          "counter": (471957, 10), "shells": SHELLS, "forms": {"E": "OFF", "R": "NRB", "X": "OFF"}}
CRIT = 1e-3
P2_CRIT = 0.0043
CODE = {"SURVIVES": 0, "REFUTED": 1, "UNRESOLVED": 4, "NOT_IDENTIFIABLE": 5}
MODEL_CALLS = (3, 4, 5, 6, 7)
CROSS_CALLS = (2, 3, 4, 5, 6)
POP_GENS = (3, 4, 5, 6, 7, 8)
P2_PAIRS = ((4, 5), (5, 6), (6, 7))
ION_QTY = ((14, 1), (14, 2), (16, 1), (16, 2))
H_PLANCK, C_LIGHT = 6.62607015e-27, 2.99792458e10                                # line_net_rate.c 7-8
FOUR_PI = 12.56637061435917295385057353311801153679                              # line_net_rate.h 18
LINK_REASONS = ("W5_GENERATION_MISMATCH", "W5_INIT_IDENTITY", "W5_BETA_LEDGER", "W5_JBAR_LEDGER", "W5_IDENTITY",
                "FORWARD_LEDGER_MISMATCH", "SOURCE_POPULATION_IDENTITY", "RATE_RECONSTRUCTION", "BRACKET_RC_-3",
                "MATERIAL_GENERATION_MISMATCH", "LEVEL_POPULATION_INCONSISTENT")
U = 2.0 ** -53
REGISTERED_5 = {
    "generation2_collisional_response": "generation-2 n_e has no stored authority (dumps start at population 3)",
    "init_call1_nrb_decomposition": "INIT call 1 carries no Sobolev ledger (beta, J_c, L_l NOT_APPLICABLE)",
    "full_coupled_cross_response_G": "stored inputs are not a complete checkpoint of every consumed state",
    "pathway_causal_contributions": "three runs and local functions cannot apportion seed memory between pathways",
    "nrb_converged_solution_seed_equality": "no converged NRB trajectories for two seeds",
    "public_spectrum_change_cause": "outside the measured quantities of this stage",
}


def det_seedmem_scope_neg(a):
    return I(-a.hi, -a.mid, -a.lo)


def det_seedmem_scope_isum(items):
    items = [i for i in items]
    if not items:
        return I.x(0.0)
    return I(dn(math.fsum(i.lo for i in items)), math.fsum(i.mid for i in items), up(math.fsum(i.hi for i in items)))


def det_seedmem_scope_overlap(a, b):
    return a.lo <= b.hi and b.lo <= a.hi


def det_seedmem_scope_universal(states, fault=None):
    """prereg 6.3: any inadmissible cell -> X (no final set judgement; eligible counterexamples are kept apart by
    the caller); else confirmed counterexample -> REFUTED; else missing -> NOT_IDENTIFIABLE; else boundary
    overlap -> UNRESOLVED; else SURVIVES."""
    s = list(states)
    if not s:
        return "NOT_IDENTIFIABLE"
    order = ("X", "REFUTED", "NOT_IDENTIFIABLE", "UNRESOLVED")
    if fault == "mix_codes":
        order = ("X", "REFUTED", "UNRESOLVED")            # injected: a missing cell read as surviving
    for st in order:
        if st in s:
            return st
    return "SURVIVES"


def det_seedmem_scope_universal_check(states):
    """independent re-derivation of the prereg 6.3 precedence (validator; not used to produce codes)."""
    s = set(states)
    if not s:
        return "NOT_IDENTIFIABLE"
    if "X" in s:
        return "X"
    if "REFUTED" in s:
        return "REFUTED"
    if "NOT_IDENTIFIABLE" in s:
        return "NOT_IDENTIFIABLE"
    if "UNRESOLVED" in s:
        return "UNRESOLVED"
    if s != {"SURVIVES"}:
        return "X"
    return "SURVIVES"


def det_seedmem_scope_set_code(states, fault=None):
    """set-level record: the code (X when any cell is inadmissible) and the eligible counterexamples kept apart."""
    st = det_seedmem_scope_universal(states, fault)
    return {"code": "X" if st == "X" else CODE[st], "state": st, "cells": len(states),
            "eligible_refuted_cells": sum(1 for x in states if x == "REFUTED"),
            "x_cells": sum(1 for x in states if x == "X")}


def det_seedmem_scope_le(iv, crit, fault=None):
    """|d| <= crit decided on the whole interval (UNRESOLVED when the interval straddles the line)."""
    if iv is None:
        return "NOT_IDENTIFIABLE"
    if fault == "drop_intervals":
        return "SURVIVES" if iv.mid <= crit else "REFUTED"
    if fault == "widen_crit":
        crit = crit * 1.0e4                             # injected: the comparison line widened after the fact
    if iv.hi <= crit:
        return "SURVIVES"
    if iv.lo > crit:
        return "REFUTED"
    return "UNRESOLVED"


# ------------------------------------------------------------------ sealed production pure functions (CPU library)
LIB_SOURCES = ("line_net_rate.c", "opacity_publication.c")
LIB_HEADERS = ("line_net_rate.h", "opacity_publication.h")


def det_seedmem_scope_build_lib(src, out):
    """build the sealed line_net_rate.c and opacity_publication.c unmodified into a CPU shared library inside the new
    result path (both are self-contained C: libc / libm only)."""
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    lib = out / "libline_net_rate.so"
    cmd = ["gcc", "-O2", "-std=gnu11", "-fPIC", "-shared", "-I", str(src), "-o", str(lib)] + \
        [str(src / f) for f in LIB_SOURCES] + ["-lm"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"cmd": " ".join(cmd), "rc": p.returncode, "stderr_tail": p.stderr[-2000:], "elapsed_s": time.time() - t0,
           "source_sha256": sha(src / "line_net_rate.c"), "header_sha256": sha(src / "line_net_rate.h"),
           "sources_sha256": {f: sha(src / f) for f in LIB_SOURCES + LIB_HEADERS},
           "lib": str(lib), "lib_sha256": sha(lib) if lib.exists() else None,
           "gcc": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.split("\n")[0]}
    return rec


class det_seedmem_scope_Lib:
    """ctypes binding of the production pure functions this stage consumes: the NRB bracket, EXPONX, the Sobolev
    material / radiation producers of the W5 ledger and the signed Sobolev tau of the opacity publication."""

    def __init__(self, path):
        self.path = str(path)
        L = ctypes.CDLL(self.path)

        class BR(ctypes.Structure):
            _fields_ = [("beta", ctypes.c_double), ("R_absorb", ctypes.c_double), ("R_stim", ctypes.c_double),
                        ("R_spont", ctypes.c_double)]

        class MAT(ctypes.Structure):
            _fields_ = [("raw_integrated_opacity", ctypes.c_double), ("emission_per_sr", ctypes.c_double),
                        ("effective_integrated_opacity", ctypes.c_double), ("effective_tau", ctypes.c_double),
                        ("srce_chk_applied", ctypes.c_int), ("exact_zero_provenance", ctypes.c_int)]

        class RAD(ctypes.Structure):
            _fields_ = [("beta", ctypes.c_double), ("one_minus_beta_over_tau", ctypes.c_double),
                        ("continuum_term", ctypes.c_double), ("local_emission_term", ctypes.c_double),
                        ("jbar", ctypes.c_double), ("jbar_absolute_uncertainty", ctypes.c_double)]

        class TAU(ctypes.Structure):
            _fields_ = [("value", ctypes.c_double), ("validity", ctypes.c_int), ("generation", ctypes.c_uint64)]
        self.BR, self.MAT, self.RAD = BR, MAT, RAD
        f = L.line_net_bracket_rates
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_double] * 4 + [ctypes.c_uint] + [ctypes.c_double] * 3 + [ctypes.POINTER(BR)]
        g = L.line_net_cmfgen_exponx
        g.restype = ctypes.c_int
        g.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        m = L.line_net_sobolev_material
        m.restype = ctypes.c_int
        m.argtypes = [ctypes.c_double] * 5 + [ctypes.c_int, ctypes.c_uint, ctypes.POINTER(MAT)]
        r = L.line_net_sobolev_radiation
        r.restype = ctypes.c_int
        r.argtypes = [ctypes.POINTER(MAT)] + [ctypes.c_double] * 4 + [ctypes.POINTER(RAD)]
        t = L.a208_signed_sobolev
        t.restype = TAU
        t.argtypes = [ctypes.c_double] * 8 + [ctypes.c_uint64]
        self.f, self.g, self.m, self.r, self.t = f, g, m, r, t

    def bracket(self, ct, le, tau, jbar, prov, B_lu, B_ul, A_ul):
        o = self.BR()
        rc = self.f(ct, le, tau, jbar, prov, B_lu, B_ul, A_ul, ctypes.byref(o))
        return rc, o.beta, o.R_absorb, o.R_stim, o.R_spont

    def exponx(self, tau):
        b, c = ctypes.c_double(), ctypes.c_double()
        rc = self.g(tau, ctypes.byref(b), ctypes.byref(c))
        return rc, b.value, c.value

    def material_radiation(self, n_u, A, nu, tau, t_exp, jc, jce):
        """production W5 producer order: material(policy CMFGEN_SRCE_CHK, 1 line) then radiation (lumina_cmfgen.c
        W5 call site); returns (rc, effective_tau, beta, local_emission_term, jbar, jbar_uncertainty)."""
        mat, rad = self.MAT(), self.RAD()
        if self.m(n_u, A, nu, tau, t_exp, 1, 1, ctypes.byref(mat)) != 0:
            return (-1,) + (math.nan,) * 5
        if self.r(ctypes.byref(mat), jc, jce, nu, t_exp, ctypes.byref(rad)) != 0:
            return (-2,) + (math.nan,) * 5
        return 0, mat.effective_tau, rad.beta, rad.local_emission_term, rad.jbar, rad.jbar_absolute_uncertainty

    def tau(self, csob, f_lu, lam, t_exp, n_l, n_u, g_l, g_u):
        o = self.t(csob, f_lu, lam, t_exp, n_l, n_u, float(g_l), float(g_u), 0)
        return o.validity, o.value


def det_seedmem_scope_csob(src):
    m = re.search(r"#define\s+SOBOLEV_COEFF\s+([0-9.eE+-]+)", (Path(src) / "lumina.h").read_text(errors="replace"))
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------ sample, deck, connectivity
def det_seedmem_scope_sample(census, target, fault=None):
    df = pd.read_csv(census, float_precision="round_trip")
    s = df[df.Z.isin(target["Z"]) & (df.ion == target["ion"]) & df.shell.isin(target["shells"]) &
           (df.beta2_E < target["beta_max"])]
    keys = sorted((int(r.Z), int(r.line_id), int(r.shell)) for r in s.itertuples())
    flags = {(int(r.Z), int(r.line_id), int(r.shell)): {"L2": int(r.L2), "D2": int(r.D2), "beta2_E": float(r.beta2_E)}
             for r in s.itertuples()}
    if fault == "drop_counter":
        keys = [k for k in keys if (k[1], k[2]) != target["counter"]]
    if fault == "add_shell":
        k0 = keys[0]
        keys = sorted(keys + [(k0[0], k0[1], 11)])
        flags[(k0[0], k0[1], 11)] = flags[k0]
    why = []
    if len(set(keys)) != len(keys):
        why.append("SAMPLE_DUPLICATE_KEY")
    counts = {z: sum(1 for k in keys if k[0] == z) for z in target["Z"]}
    if counts != target["counts"]:
        why.append(f"SAMPLE_COUNT:{counts}")
    if sum(1 for k in keys if k[1] == target["control"]) != len(target["shells"]):
        why.append("SAMPLE_CONTROL")
    if not any((k[1], k[2]) == target["counter"] for k in keys):
        why.append("SAMPLE_COUNTER_MISSING")
    if any(k[2] not in target["shells"] for k in keys):
        why.append("SAMPLE_SHELL_OUTSIDE_W5")
    sets = {"control_1206": [k for k in keys if k[1] == target["control"]],
            "SiIII_other": [k for k in keys if k[0] == 14 and k[1] != target["control"]],
            "SIII": [k for k in keys if k[0] == 16]}
    return {"keys": keys, "flags": flags, "sets": sets, "counts": counts, "why": why}


def det_seedmem_scope_deck(model_dir, Zs, ion):
    """deck rows of the target ions (line_id = 0-based data row, checked) and their levels / super-levels."""
    model_dir = Path(model_dir)
    cols = ["atomic_number", "ion_number", "level_number_lower", "level_number_upper", "line_id", "f_lu", "nu", "B_lu",
            "B_ul", "A_ul", "wavelength_cm"]
    parts, why, base = [], [], 0
    for ch in pd.read_csv(model_dir / "line_list.csv", usecols=cols, chunksize=1 << 20, float_precision="round_trip"):
        idx = np.arange(base, base + len(ch))
        base += len(ch)
        m = ch.atomic_number.isin(Zs).values & (ch.ion_number.values == ion)
        sub = ch[m].copy()
        sub["row"] = idx[m]
        parts.append(sub)
    lines = pd.concat(parts)
    if not (lines.row.values == lines.line_id.values).all():
        why.append("DECK_LINE_ID_NOT_ROW")
    lv_all = pd.read_csv(model_dir / "levels.csv", float_precision="round_trip")
    lv = lv_all[lv_all.atomic_number.isin(Zs) & (lv_all.ion_number == ion)]
    levels = {(int(r.atomic_number), int(r.level_number)): {"E": float(r.energy_eV), "g": int(r.g),
                                                             "deck_sl": int(r.super_level)} for r in lv.itertuples()}
    at = {}
    for r in lines.itertuples():
        Z, lo, upl = int(r.atomic_number), int(r.level_number_lower), int(r.level_number_upper)
        L_lo, L_up = levels.get((Z, lo)), levels.get((Z, upl))
        if L_lo is None or L_up is None:
            continue
        at[int(r.line_id)] = {"Z": Z, "lo": lo, "up": upl, "f_lu": float(r.f_lu), "nu": float(r.nu),
                              "B_lu": float(r.B_lu), "B_ul": float(r.B_ul), "A_ul": float(r.A_ul),
                              "lam_cm": float(r.wavelength_cm), "g_l": L_lo["g"], "g_u": L_up["g"], "E_l": L_lo["E"],
                              "E_u": L_up["E"]}
    return {"at": at, "levels": levels, "n_levels_total": int(len(lv_all)),
            "level_numbers_all": lv_all.level_number.values.astype(int),
            "ion_level_counts": {int(z): int((lv.atomic_number == z).sum()) for z in Zs}, "why": why}


def det_seedmem_scope_sl(deck, exports, stdout_text, ion, fault=None):
    """the run's NLTE solve units for the target ions (prereg 3.4, super-level row): the deck f_to_s column, overridden
    by LUMINA_SUPER_CUTOFF=K as super = min(level_number, K) (lumina_atomic.c loader), active only with
    LUMINA_SUPER_LEVELS != 0 and the projection's super_mode (stdout 'Super-levels: ACTIVE').  The cutoff count and
    the per-ion level counts are checked against the run stdout; an unconfirmed mode is inadmissible."""
    why = []
    K = det_seedmem_scope_atoi(exports.get("LUMINA_SUPER_CUTOFF"))
    req = det_seedmem_scope_atoi(exports.get("LUMINA_SUPER_LEVELS")) != 0
    m_act = re.search(r"\[NLTE\] Super-levels: (ACTIVE|off \(identity\))", stdout_text)
    if m_act is None:
        why.append("SL_MODE_NOT_IN_STDOUT")
    active = bool(m_act and m_act.group(1) == "ACTIVE")
    if active and not req:
        why.append("SL_ACTIVE_WITHOUT_REQUEST")
    if K > 0:
        m_cut = re.search(r"\[ARTIS super-cutoff\] K=(\d+): (\d+) levels lumped", stdout_text)
        lumped = int((deck["level_numbers_all"] >= K).sum())
        if m_cut is None or int(m_cut.group(1)) != K or int(m_cut.group(2)) != lumped:
            why.append(f"SL_CUTOFF_NOT_CONFIRMED:K={K}:deck_lumped={lumped}")
    for Z, n in deck["ion_level_counts"].items():
        m_n = re.search(rf"Z={Z} ion={ion}: (\d+) levels", stdout_text)
        if m_n is None or int(m_n.group(1)) != n:
            why.append(f"SL_ION_LEVEL_COUNT:{Z}:{n}")
    unit = {}
    for (Z, l), v in deck["levels"].items():
        if not active:
            unit[(Z, l)] = l
        elif K > 0 and fault != "deck_sl_column":
            unit[(Z, l)] = min(l, K)
        else:
            unit[(Z, l)] = v["deck_sl"]
    size = {}
    for (Z, l), u in unit.items():
        size[(Z, u)] = size.get((Z, u), 0) + 1
    registered = {(Z, l): (min(l, K) if (active and K > 0) else (v["deck_sl"] if active else l))
                  for (Z, l), v in deck["levels"].items()}
    if registered != unit:
        why.append("SL_AUTHORITY_NOT_REGISTERED")
    return {"active": active, "requested": req, "cutoff": K, "unit": unit, "size": size, "why": why,
            "explicit_levels": sum(1 for (Z, l), u in unit.items() if size[(Z, u)] == 1),
            "lumped_levels": sum(1 for (Z, l), u in unit.items() if size[(Z, u)] > 1)}


def det_seedmem_scope_sl_of(rd, a):
    """solve-unit sizes (lower, upper) of a deck transition in a run."""
    s = rd["sl"]
    return s["size"][(a["Z"], s["unit"][(a["Z"], a["lo"])])], s["size"][(a["Z"], s["unit"][(a["Z"], a["up"])])]


def det_seedmem_scope_connect(sample_keys, at):
    """same-ion transitions sharing the lower or the upper level of each sample transition (deck connectivity)."""
    by_level = {}
    for l, a in at.items():
        by_level.setdefault((a["Z"], a["lo"]), set()).add(l)
        by_level.setdefault((a["Z"], a["up"]), set()).add(l)
    conn = {}
    for Z, line, _ in sample_keys:
        a = at[line]
        conn[line] = sorted((by_level[(Z, a["lo"])] | by_level[(Z, a["up"])]) - {line})
    return conn


def det_seedmem_scope_r(a, csob):
    """r_l = (h nu B_lu / 4 pi) / (C_SOB f lam nu / c): production opacity normalisation over the B_lu one."""
    v = (H_PLANCK * a["nu"] * a["B_lu"] / FOUR_PI) / (csob * a["f_lu"] * a["lam_cm"] * a["nu"] / C_LIGHT)
    return I.rel(v, 16)


# ------------------------------------------------------------------ run inputs
def det_seedmem_scope_atoi(v):
    m = re.match(r"\s*([+-]?\d+)", str(v) if v is not None else "")
    return int(m.group(1)) if m else 0


def det_seedmem_scope_coll_authority(exports, src_text, fault=None, run=None, cuda_text=""):
    """the collision branch the stored run consumed (production gates read with atoi/atof semantics) and the
    presence of that branch structure in the sealed source; only COLL_FIX per-line with floor 0 and no generic
    col_data pass is implemented by det_siiii_rates_coll."""
    on = lambda k: det_seedmem_scope_atoi(exports.get(k)) != 0                                   # noqa: E731
    omcm, parity, fix = on("LUMINA_OMEGA_CMFGEN"), on("LUMINA_ARTIS_PARITY"), on("LUMINA_NLTE_COLL_FIX")
    ftxt = exports.get("LUMINA_NLTE_COLL_FLOOR")
    try:
        floor = max(0.0, float(ftxt)) if ftxt not in (None, "") else 0.0
    except ValueError:
        floor = 0.0
    generic = parity or omcm
    if fault == "assume_coldata":
        generic = True
    branch = "OMEGA_CMFGEN" if omcm else "ARTIS_PARITY" if parity else "COLL_FIX_PER_LINE" if fix else "LEGACY_PER_LINE"
    snippets = {"omega_first": "if (omcm_on) {", "parity_second": "if (artis_parity_enabled()) {",
                "coll_fix_third": "} else if (nlte_coll_fix_enabled()) {",
                "generic_coldata_gate": "if ((parity_on || omcm_on) && atom->ncol_ions > 0",
                "floor_gate": "coll_floor > 0.0 && T_e > 0.0"}
    present = {k: (v in src_text) for k, v in snippets.items()}
    supported = branch == "COLL_FIX_PER_LINE" and floor == 0.0 and not generic and all(present.values())
    # prereg 3.3: data present / loader called / matrix consumption, kept apart
    files = sorted(p.name for p in (Path(run) / "input/model").iterdir() if "col" in p.name) if run else []
    so = (Path(run) / "stdout.log").read_text(errors="replace") if run and (Path(run) / "stdout.log").exists() else ""
    markers = sum(1 for ln in so.split("\n") if "[OMEGA-CMFGEN]" in ln or "[ARTIS-PARITY" in ln)
    consumes = generic
    if fault == "files_as_consumed":
        consumes = bool(files)                          # injected: deck file presence read as matrix consumption
    why = []
    # prereg 3.3: consumption needs the loader (gated by OMEGA_CMFGEN / ARTIS_PARITY) and the generic gate
    if consumes and not (generic and (parity or omcm)):
        why.append("COLLISION_AUTHORITY_CONTRADICTION:consumption_without_loader_or_gate")
    if (parity or omcm) and markers == 0:
        why.append("COLLISION_AUTHORITY_CONTRADICTION:loader_knob_without_stdout_marker")
    return {"branch": branch, "floor": floor, "generic_coldata": generic, "feiii_only_coldata": True,
            "source_snippets": present, "supported": supported and not why,
            "deck_collision_files": files,
            "loader_gated_by_parity_or_omega": ("if (omega_cmfgen_enabled()) {" in cuda_text and
                                                "load_ion_coldata_manifest(&atom_data, ref_dir)" in cuda_text),
            "loader_called": parity or omcm, "stdout_loader_markers": markers,
            "matrix_consumes_col_data": consumes, "why": why}


def det_seedmem_scope_dumps(run):
    """generation-labelled physics dumps: population generation -> n_e, T_e, NLTE ion sums (by shell_id)."""
    d = Path(run) / "work/physics_comparison"
    out = {"ne": {}, "Te": {}, "ion": {}, "rad": {}, "files": {}, "label": {}, "why": []}
    for m in sorted(d.glob("physics_DET_iter*.manifest.json")):
        js = json.loads(m.read_text())
        g, rg = js.get("population_generation"), js.get("radiation_generation")
        sh, io = m.with_name(m.name.replace(".manifest.json", ".shell.csv")), m.with_name(
            m.name.replace(".manifest.json", ".ion.csv"))
        if isinstance(g, bool) or not isinstance(g, int) or not sh.exists() or not io.exists():
            out["why"].append(f"DUMP_INCOMPLETE:{m.name}")
            continue
        if g in out["ne"]:
            out["why"].append(f"DUMP_DUPLICATE_GENERATION:{g}")
            continue
        s = pd.read_csv(sh, float_precision="round_trip").sort_values("shell_id")
        ns = int(s.shell_id.max()) + 1
        ne, te = np.full(ns, np.nan), np.full(ns, np.nan)
        ne[s.shell_id.values] = s.n_e_cm3.values
        te[s.shell_id.values] = s.T_e_K.values
        x = pd.read_csv(io, float_precision="round_trip").sort_values("shell_id")
        if not (x.population_generation.values == g).all():
            out["why"].append(f"DUMP_GENERATION_MISMATCH:{m.name}")
            continue
        ion = {}
        for Z, q in ION_QTY:
            v = np.full(ns, np.nan)
            v[x.shell_id.values] = x[f"Z{Z}_nlte_q{q}_cm3"].values
            ion[(Z, q)] = v
        out["ne"][g], out["Te"][g], out["ion"][g], out["rad"][g] = ne, te, ion, rg
        out["files"][g] = [str(m), str(sh), str(io)]
        out["label"][g] = g
    for g in out["rad"]:
        if out["rad"][g] != g - 1:
            out["why"].append(f"DUMP_RADIATION_NOT_POPULATION_MINUS_1:{g}")
    return out


def det_seedmem_scope_dumps_check(dumps):
    """every stored generation is served only by the dump whose manifest carries that population generation."""
    return [f"DUMP_GENERATION_FILLED:{g}<-{lab}" for g, lab in dumps["label"].items() if lab != g]


SHA_CACHE = {}


def det_seedmem_scope_sha_cached(p):
    p = Path(p)
    st = p.stat()
    k = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if k not in SHA_CACHE:
        SHA_CACHE[k] = sha(p)
    return SHA_CACHE[k]


def det_seedmem_scope_capture_integrity(capd):
    """the capture file is the one its manifest committed (sha256, record count) and its mapped flags add up to the
    manifest census (prereg 3.1: mapped authority)."""
    why = []
    man = json.loads((capd / "manifest.json").read_text())
    lb = capd / "lines.bin"
    ent = (man.get("files") or {}).get("lines.bin") or {}
    if not ent.get("sha256") or det_seedmem_scope_sha_cached(lb) != ent["sha256"]:
        why.append("CAPTURE_FILE_SHA_MISMATCH")
    m = np.memmap(lb, dtype=POP.REC, mode="r")
    cen = man.get("census") or {}
    n_mapped = int(np.count_nonzero(m["m"] == 1))
    if cen.get("consumed") != int(len(m)) or cen.get("mapped") != n_mapped:
        why.append(f"CAPTURE_CENSUS_MISMATCH:records={len(m)}:mapped={n_mapped}")
    del m
    return why


def det_seedmem_scope_run(run, tag, keys, src_text, deck, fault=None, cuda_text=""):
    run = Path(run)
    meta = MECH.det_1206mech_meta(run)
    why = [f"{tag}:{w}" for w in meta["why"]]
    hist, contra = REP.det_siiii_repair_w5_load(run, keys)
    if contra:
        why.append(f"{tag}:W5_CONTRADICTORY_DUPLICATE:{len(contra)}")
    dumps = det_seedmem_scope_dumps(run)
    if fault == "fill_gen2_ne" and 3 in dumps["ne"]:
        dumps["ne"][2], dumps["Te"][2], dumps["label"][2] = dumps["ne"][3], dumps["Te"][3], 3
    why += [f"{tag}:{w}" for w in dumps["why"] + det_seedmem_scope_dumps_check(dumps)]
    cne = run / "work/det_obs_capture/ne.f64"
    if 7 not in dumps["ne"] or not cne.exists() or dumps["ne"][7].tobytes() != cne.read_bytes():
        why.append(f"{tag}:LINK_NE_DUMP7_NOT_CAPTURE")
    why += [f"{tag}:{w}" for w in det_seedmem_scope_capture_integrity(run / "work/det_obs_capture")]
    so = (run / "stdout.log").read_text(errors="replace") if (run / "stdout.log").exists() else ""
    sl = det_seedmem_scope_sl(deck, meta["exports"], so, TARGET["ion"], fault)
    why += [f"{tag}:{w}" for w in sl["why"]]
    coll = det_seedmem_scope_coll_authority(meta["exports"], src_text, fault, run, cuda_text)
    why += [f"{tag}:{w}" for w in coll["why"]]
    return {"run": str(run), "tag": tag, "meta": meta, "t_exp": meta["t_exp"], "exports": meta["exports"],
            "form": "NRB" if meta["nrb"] else "OFF", "hist": hist, "dumps": dumps, "sl": sl,
            "coll": coll, "why": why}


def det_seedmem_scope_row(rd, line, shell, call):
    return rd["hist"].get((line, shell), {}).get(call)


def det_seedmem_scope_row_check(row, call, a, Z):
    """MECH row admissibility plus the deck identity of the row (frequency, Z, ion)."""
    why = MECH.det_1206mech_check_row(row, call)
    if row is not None and a is not None:
        if float(row["nu_hz"]) != a["nu"] or int(row["Z"]) != Z or int(row["ion"]) != TARGET["ion"]:
            why.append("W5_IDENTITY")
    return why


def det_seedmem_scope_is_link(why):
    return any(w.startswith(LINK_REASONS) for w in why)


# ------------------------------------------------------------------ production relations on stored ledgers
def det_seedmem_scope_meas(row, a, r_iv, fault=None):
    """measured source S~_E = L_l / ((1 - beta) r_l)  (prereg 4.1)."""
    b, L = float(row["beta"]), float(row["L_l"])
    if not (0.0 <= b < 1.0 and L >= 0.0 and math.isfinite(L)):
        return None
    s_tau = I.x(L).div(I.rel(1.0 - b, 1))
    if s_tau is None:
        return None
    return s_tau if fault == "omit_r" else s_tau.div(r_iv)


def det_seedmem_scope_rates(row, a, form, lib, fault=None):
    """radiative (R_abs, R_stim, R_spont) intervals of the stored call in the run's production form.
    OFF: B J-bar, A; NRB: production line_net_bracket_rates on the stored ledger (SRCE_CHK cells keep the base)."""
    B, Bu, A = a["B_lu"], a["B_ul"], a["A_ul"]
    if not (RATES.NU_MIN <= a["nu"] <= RATES.NU_MAX):          # BB_EXCLUDED_OUTSIDE_DOMAIN: J = 0, A only, no NRB
        return (I.x(0.0), I.x(0.0), I.x(A)), "A_ONLY_OUTSIDE_BB_DOMAIN"
    if form == "NRB":
        jc, jce, beta = float(row["J_c"]), float(row["J_c_err_upper"]), float(row["beta"])
        prov = det_seedmem_scope_atoi(float(row["srce_chk"]) if math.isfinite(float(row["srce_chk"])) else 255)
        rc, b2, Ra, Rs, Rp = lib.bracket(beta * jc, float(row["L_l"]), float(row["tau_eff"]), float(row["Jbar"]),
                                         prov, B, Bu, A)
        if rc == 0:
            ct = I.x(beta) * I(dn(jc - jce), jc, up(jc + jce))
            Ra_i, Rs_i, Rp_i = I.x(B) * ct, I.x(Bu) * ct, I.x(A) * I.rel(beta, 1)
            Ra_i.mid, Rs_i.mid, Rp_i.mid = Ra, Rs, Rp
            if fault == "omit_beta":
                Rp_i = I.x(A)
            if fault == "omit_stim":
                Rs_i = I.x(0.0)
            # registered reconstruction check: the consumed rate intervals enclose the production function output
            if not all(iv.lo <= v <= iv.hi for iv, v in ((Ra_i, Ra), (Rs_i, Rs), (Rp_i, Rp))):
                return None, "RATE_RECONSTRUCTION"
            return (Ra_i, Rs_i, Rp_i), "NRB_BRACKET"
        if rc != 1:
            if fault == "ignore_ledger":
                ct = I.x(beta * jc)
                return (I.x(B) * ct, I.x(Bu) * ct, I.rel(A * beta, 1)), "NRB_LEDGER_IGNORED"
            return None, f"BRACKET_RC_{rc}"
        base = "BASE_SRCE_CHK"
    else:
        base = "OFF"
    jb, je = float(row["Jbar"]), float(row["Jbar_err_upper"])
    J = I(dn(jb - je), jb, up(jb + je))
    return (I.x(B) * J, I.x(Bu) * J, I.x(A)), base


def det_seedmem_scope_identity(row, a, meas, t_exp, csob, lib):
    """prereg 4.1 / 3.2: with the recovered populations of the same row, S~_E = L / ((1 - beta) r_l) equals the
    population source A n_u / (B_lu (n_l - a n_u)) algebraically (r_l converts the oscillator-strength opacity of tau
    to the B_lu one).  The only non-identity is production's beta vs the EXPONX companion: (1 - beta) against
    tau q, whose relative difference (plus 4 u) widens the comparison.  MISMATCH is a link failure."""
    if meas is None:
        return "NOT_APPLICABLE"
    inv, why = det_seedmem_scope_invert(row, a, t_exp, csob, lib)
    if inv is None:
        return "MISMATCH" if why == "FORWARD_LEDGER_MISMATCH" else "NOT_APPLICABLE"
    tau, beta = float(row["tau_eff"]), float(row["beta"])
    s_pop = (I.rel(a["A_ul"] / a["B_lu"], 1) * inv["n_u"]).div(inv["d"])      # d = n_l - a n_u = tau / K
    if s_pop is None:
        return "NOT_APPLICABLE"
    rel = abs((1.0 - beta) - tau * inv["q"]) / (1.0 - beta) + 4 * U
    wide = I(dn(s_pop.lo * (1 - rel)), s_pop.mid, up(s_pop.hi * (1 + rel)))
    return "OK" if det_seedmem_scope_overlap(wide, meas) else "MISMATCH"


def det_seedmem_scope_coll(ne, Te, a):
    if not (math.isfinite(ne) and math.isfinite(Te)):
        return None
    return RATES.det_siiii_rates_coll(ne, Te, a["E_l"], a["E_u"], a["g_l"], a["g_u"], a["f_lu"])


def det_seedmem_scope_source2(rad, coll, a):
    """two-level source S(x) = (A/B_lu) x / (1 - a x), x = (R_abs + C_up)/(R_stim + R_spont + C_down)."""
    if rad is None or coll is None:
        return None
    Ra, Rs, Rp = rad
    Cu, Cd = coll
    x = (Ra + Cu).div(Rs + Rp + Cd)
    if x is None:
        return None
    d = I.x(1.0) - I.rel(a["g_l"] / a["g_u"], 1) * x
    if not d.lo > 0.0:
        return None
    return (I.rel(a["A_ul"] / a["B_lu"], 1) * x).div(d)


def det_seedmem_scope_invert(row, a, t_exp, csob, lib, fault=None):
    """populations of the row's material generation from the production ledger (prereg 3.2):
    L_l = eta (c t/nu) q(tau), eta = n_u A h nu / 4 pi  ->  n_u;  tau = C_SOB f lam t (n_l - a n_u)  ->  n_l."""
    if row is None:
        return None, "W5_ROW_ABSENT"
    sc = row["srce_chk"]
    if not math.isfinite(float(sc)) or det_seedmem_scope_atoi(float(sc)) != 0:
        return None, "SRCE_CHK_OR_UNDEFINED"
    tau, L = float(row["tau_eff"]), float(row["L_l"])
    A, f, nu = a["A_ul"], a["f_lu"], a["nu"]
    if fault == "wrong_A":
        A = 2.0 * A
    if not (tau > 0.0 and L > 0.0 and A > 0.0 and f > 0.0):
        return None, "INVERSION_DOMAIN"
    rc, beta, q = lib.exponx(tau)
    if rc != 0 or not q > 0.0:
        return None, "EXPONX"
    scale = C_LIGHT * t_exp / nu
    nu_mid = L / (scale * q) * FOUR_PI / (A * H_PLANCK * nu)
    n_u = I.rel(nu_mid, 24)
    K = csob * f * a["lam_cm"] * t_exp
    g_l, g_u = (a["g_u"], a["g_l"]) if fault == "swap_levels" else (a["g_l"], a["g_u"])
    d = I.rel(tau / K, 3)
    n_l = (d * I.rel(1.0, 1)) + (I.rel(g_l / g_u, 1) * n_u * I.rel(1.0, 1))
    inv = {"n_u": n_u, "n_l": n_l, "q": q, "d": d}
    if not det_seedmem_scope_forward(row, a, inv, t_exp, csob, lib):
        return None, "FORWARD_LEDGER_MISMATCH"
    return inv, None


FWD_CACHE = {}


def det_seedmem_scope_widen(lo, hi, k):
    for _ in range(k):
        lo, hi = dn(lo), up(hi)
    return lo, hi


def det_seedmem_scope_forward(row, a, inv, t_exp, csob, lib):
    """prereg 3.2 (2): the recovered populations, put through the unmodified production producers, reproduce the
    stored W5 ledger: signed Sobolev tau (opacity_publication.c) from (n_l, n_u), then material / radiation
    (line_net_rate.c, W5 producer order) from n_u at the stored effective tau and the stored J_c.  Each producer
    is monotone in its population argument, so the recovered interval corners enclose the image; 8 ulps (tau: of
    the larger subtraction term) cover the forward rounding.  beta and the J-bar bound must be bit-equal."""
    tau, L, jb = float(row["tau_eff"]), float(row["L_l"]), float(row["Jbar"])
    jc, jce, jbe, beta = float(row["J_c"]), float(row["J_c_err_upper"]), float(row["Jbar_err_upper"]), float(row["beta"])
    key = (a["nu"], a["A_ul"], a["g_l"], a["g_u"], tau, L, jb, jc, jce, jbe, beta, inv["n_u"].lo, inv["n_u"].hi,
           inv["n_l"].lo, inv["n_l"].hi)
    if key in FWD_CACHE:
        return FWD_CACHE[key]
    n_u, n_l = inv["n_u"], inv["n_l"]
    taus = []
    for nl_ in (n_l.lo, n_l.hi):
        for nu_ in (n_u.lo, n_u.hi):
            v, t = lib.tau(csob, a["f_lu"], a["lam_cm"], t_exp, nl_, nu_, a["g_l"], a["g_u"])
            if v not in (1, 2):
                FWD_CACHE[key] = False
                return False
            taus.append(t)
    slack = 8 * math.ulp(abs(csob * a["f_lu"] * a["lam_cm"] * t_exp * n_l.hi))
    ok = (min(taus) - slack) <= tau <= (max(taus) + slack)
    outs = [lib.material_radiation(nu_, a["A_ul"], a["nu"], tau, t_exp, jc, jce) for nu_ in (n_u.lo, n_u.hi)]
    if any(o[0] != 0 for o in outs):
        FWD_CACHE[key] = False
        return False
    Llo, Lhi = det_seedmem_scope_widen(min(o[3] for o in outs), max(o[3] for o in outs), 8)
    Jlo, Jhi = det_seedmem_scope_widen(min(o[4] for o in outs), max(o[4] for o in outs), 8)
    ok = ok and Llo <= L <= Lhi and Jlo <= jb <= Jhi and all(o[1] == tau and o[2] == beta and o[5] == jbe
                                                            for o in outs)
    FWD_CACHE[key] = bool(ok)
    return bool(ok)


# ------------------------------------------------------------------ registered computations
def det_seedmem_scope_model(runs, sample, deck, csob, lib, fault=None):
    """prereg 4.2: d = S~_E,meas(k) / S(x(call k-1 radiation, generation-k n_e T_e)) - 1, calls 3-7."""
    rows, ident = [], []
    for tag, rd in runs.items():
        form = rd["form"]
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            sl_lo, sl_up = det_seedmem_scope_sl_of(rd, a)
            for k in MODEL_CALLS:
                rk, rp = det_seedmem_scope_row(rd, line, s, k), det_seedmem_scope_row(rd, line, s, k - 1)
                if fault == "beta_current":
                    rp = rk
                why = det_seedmem_scope_row_check(rk, k, a, Z)
                if fault != "skip_row_check":
                    why += det_seedmem_scope_row_check(rp, k - 1, a, Z)
                gk = k + 1 if fault == "material_plus1" else k
                if gk in rd["dumps"]["label"] and rd["dumps"]["label"][gk] != k:
                    why.append("MATERIAL_GENERATION_MISMATCH")
                ne = rd["dumps"]["ne"].get(gk, [np.nan] * (s + 1))
                te = rd["dumps"]["Te"].get(gk, [np.nan] * (s + 1))
                rec = {"run": tag, "form": form, "Z": Z, "line": line, "shell": s, "call": k, "material_generation": k,
                       "material_generation_used": rd["dumps"]["label"].get(gk), "radiation_call": k - 1,
                       "sl_lo": sl_lo, "sl_up": sl_up, "independent_levels": int(sl_lo == 1 and sl_up == 1),
                       "collision_branch": rd["coll"]["branch"]}
                meas = pred = None
                if not why:
                    meas = det_seedmem_scope_meas(rk, a, deck["r"][line], fault)
                    rad, br = det_seedmem_scope_rates(rp, a, form, lib, fault)
                    rec["rate_branch"] = br
                    coll = det_seedmem_scope_coll(float(ne[s]), float(te[s]), a) if rd["coll"]["supported"] else None
                    pred = det_seedmem_scope_source2(rad, coll, a)
                    if coll is None:
                        why.append("COLLISION_BRANCH_UNSUPPORTED" if not rd["coll"]["supported"] else "NE_TE_ABSENT")
                    elif rad is None:
                        why.append(br)
                    elif pred is None or meas is None:
                        why.append("SOURCE_UNDEFINED")
                    rec["source_population_identity"] = det_seedmem_scope_identity(rk, a, meas, rd["t_exp"], csob, lib)
                    if rec["source_population_identity"] == "MISMATCH":
                        why.append("SOURCE_POPULATION_IDENTITY")
                d = None
                if meas is not None and pred is not None:
                    q = meas.div(pred)
                    d = None if q is None else q - I.x(1.0)
                st = det_seedmem_scope_le(None if d is None else d.abs(), CRIT, fault)
                if det_seedmem_scope_is_link(why):
                    st = "X"
                REP.det_siiii_repair_cols(rec, "S_meas", meas)
                REP.det_siiii_repair_cols(rec, "S_pred", pred)
                REP.det_siiii_repair_cols(rec, "d", d)
                rec["status"], rec["reasons"] = st, ";".join(why)
                rows.append(rec)
                ident.append({"run": tag, "Z": Z, "line": line, "shell": s, "question": "two_level_source",
                              "generation": k, "status": "IDENTIFIABLE" if st in CODE and st != "NOT_IDENTIFIABLE"
                              else ("X" if st == "X" else "NOT_IDENTIFIABLE"), "reason": ";".join(why)})
    return rows, ident


def det_seedmem_scope_cross(runs, sample, deck, lib, fault=None):
    """prereg 5.1-5.2: E'/X fixed-background exchange of L at calls 2-6 with the production NRB function."""
    E, X = runs["E"], runs["X"]
    rows = []
    for Z, line, s in sample["keys"]:
        a = deck["at"][line]
        B, Bu, A = a["B_lu"], a["B_ul"], a["A_ul"]
        for j in CROSS_CALLS:
            rE, rX = det_seedmem_scope_row(E, line, s, j), det_seedmem_scope_row(X, line, s, j)
            why = det_seedmem_scope_row_check(rE, j, a, Z) + det_seedmem_scope_row_check(rX, j, a, Z)
            rec = {"Z": Z, "line": line, "shell": s, "call": j, "material_generation_response": j + 1}
            if why:
                rec.update({"status": "X" if det_seedmem_scope_is_link(why) else "NOT_IDENTIFIABLE",
                            "reasons": ";".join(why)})
                rows.append(rec)
                continue
            z = {}
            for tag, rd, r in (("E", E, rE), ("X", X, rX)):
                ne = rd["dumps"]["ne"].get(j + 1)
                te = rd["dumps"]["Te"].get(j + 1)
                z[tag] = {"c": float(r["beta"]) * float(r["J_c"]), "tau": float(r["tau_eff"]),
                          "prov": det_seedmem_scope_atoi(float(r["srce_chk"])),
                          "coll": None if ne is None else det_seedmem_scope_coll(float(ne[s]), float(te[s]), a)}
            Ls = {"E": float(rE["L_l"]), "X": float(rX["L_l"])}
            H, bad = {}, []
            for lt in ("E", "X"):
                for zt in ("E", "X"):
                    zz = z[zt]
                    c = zz["c"]
                    if fault == "move_c_with_L":
                        c = z[lt]["c"]
                    if c != zz["c"]:
                        bad.append("CROSS_BACKGROUND_NOT_FIXED")
                    jb = c + Ls[lt]
                    rc, b2, Ra, Rs, Rp = lib.bracket(c, Ls[lt], zz["tau"], jb, zz["prov"], B, Bu, A)
                    if rc != 0 or zz["coll"] is None:
                        bad.append(f"BRACKET_RC_{rc}" if rc != 0 else "NE_TE_ABSENT")
                        continue
                    Cu, Cd = zz["coll"]
                    s_nrb = det_seedmem_scope_source2((I.x(Ra), I.x(Rs), I.x(Rp)), (I.x(Cu.mid), I.x(Cd.mid)), a)
                    s_off = det_seedmem_scope_source2((I.x(B * jb), I.x(Bu * jb), I.x(A)), (I.x(Cu.mid), I.x(Cd.mid)), a)
                    H[(lt, zt)] = {"rates": (Ra, Rs, Rp), "off_rates": (B * jb, Bu * jb, A), "jbar": jb, "c": c,
                                   "L": Ls[lt], "tau": zz["tau"], "C_up": Cu.mid, "C_down": Cd.mid,
                                   "nrb": None if s_nrb is None else s_nrb.mid,
                                   "off": None if s_off is None else s_off.mid}
            if bad or len(H) != 4 or any(v["nrb"] is None or v["off"] is None for v in H.values()):
                rec.update({"status": "X" if "CROSS_BACKGROUND_NOT_FIXED" in bad else "NOT_IDENTIFIABLE",
                            "reasons": ";".join(sorted(set(bad)) or ["SOURCE_UNDEFINED"])})
                rows.append(rec)
                continue
            inv = all(H[("E", zt)]["rates"] == H[("X", zt)]["rates"] for zt in ("E", "X"))
            rec["nrb_rates_invariant_under_L"] = int(inv)
            rec["status"] = "SURVIVES" if inv else "REFUTED"
            for form in ("nrb", "off"):
                hEE, hXE, hEX, hXX = (H[("E", "E")][form], H[("X", "E")][form], H[("E", "X")][form],
                                      H[("X", "X")][form])
                DL = 0.5 * ((hXE - hEE) + (hXX - hEX))
                Dz = 0.5 * ((hEX - hEE) + (hXX - hXE))
                rec[f"{form}_H_EE"], rec[f"{form}_H_XE"], rec[f"{form}_H_EX"], rec[f"{form}_H_XX"] = hEE, hXE, hEX, hXX
                rec[f"{form}_D_L"], rec[f"{form}_D_z"] = DL, Dz
                rec[f"{form}_ledger_residual"] = (DL + Dz) - (hXX - hEE)
            for tag in ("E", "X"):
                rec[f"c_{tag}"], rec[f"L_{tag}"], rec[f"tau_{tag}"] = z[tag]["c"], Ls[tag], z[tag]["tau"]
            for (lt, zt), h in H.items():           # combination inputs and production rates (prereg 7)
                cmb = f"L{lt}z{zt}"
                rec[f"{cmb}_c"], rec[f"{cmb}_L"], rec[f"{cmb}_tau"], rec[f"{cmb}_jbar"] = h["c"], h["L"], h["tau"], h["jbar"]
                rec[f"{cmb}_C_up"], rec[f"{cmb}_C_down"] = h["C_up"], h["C_down"]
                for nm, v in zip(("R_absorb", "R_stim", "R_spont"), h["rates"]):
                    rec[f"{cmb}_nrb_{nm}"] = v
                for nm, v in zip(("R_absorb", "R_stim", "R_spont"), h["off_rates"]):
                    rec[f"{cmb}_off_{nm}"] = v
            mE = det_seedmem_scope_row(E, line, s, j + 1)
            mX = det_seedmem_scope_row(X, line, s, j + 1)
            if mE is not None and mX is not None and not (det_seedmem_scope_row_check(mE, j + 1, a, Z) +
                                                          det_seedmem_scope_row_check(mX, j + 1, a, Z)):
                SE = det_seedmem_scope_meas(mE, a, deck["r"][line])
                SX = det_seedmem_scope_meas(mX, a, deck["r"][line])
                if SE is not None and SX is not None:
                    rec["delta_S_meas"] = SX.mid - SE.mid
                    rec["delta_e_model_off"] = rec["delta_S_meas"] - (H[("X", "X")]["off"] - H[("E", "E")]["off"])
            rec["reasons"] = ""
            rows.append(rec)
    return rows


def det_seedmem_scope_budget(runs, sample, deck, conn, csob, lib, fault=None, cap_status=None):
    """prereg 4.4: generation-7 populations (W5 call 7 inversion) and call-6 rates of the target and of every
    connected same-ion transition at the shell; budgets are bound-bound only (bf and line-free channels are not
    stored), so closure is never claimed.  A recovered population that fails the forward production ledger, or
    disagreeing recoveries of one level, make the row inadmissible (X); a target without the independent capture
    authority carries that absence in its reason (prereg 3.2 (3))."""
    rows, ident = [], []
    cap_status = cap_status or {}
    for tag, rd in runs.items():
        form, t_exp = rd["form"], rd["t_exp"]
        ne7, te7 = rd["dumps"]["ne"].get(7), rd["dumps"]["Te"].get(7)
        table = {}                      # (Z, level, shell) -> generation-7 populations recovered from any loaded line
        fwd_bad = set()
        for (l2, s2), h in rd["hist"].items():
            b = deck["at"].get(l2)
            r7 = h.get(7)
            if b is None or r7 is None or det_seedmem_scope_row_check(r7, 7, b, b["Z"]):
                continue
            inv, rsn = det_seedmem_scope_invert(r7, b, t_exp, csob, lib, fault)
            if rsn == "FORWARD_LEDGER_MISMATCH":
                fwd_bad.add((l2, s2))
            if inv is not None:
                table.setdefault((b["Z"], b["lo"], s2), []).append(inv["n_l"])
                table.setdefault((b["Z"], b["up"], s2), []).append(inv["n_u"])
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            lines = [line] + conn[line]
            if fault == "drop_connected" and len(lines) > 1:
                lines = lines[:-1]
            if fault == "double_count" and len(lines) > 1:
                lines = lines + [lines[-1]]
            expected = 1 + len(conn[line])
            flows, pops, missing, pending, aonly, link, aonly_lines = {}, {}, [], [], 0, [], []
            for l2 in lines:
                b = deck["at"][l2]
                if (l2, s) in fwd_bad:
                    link.append(f"FORWARD_LEDGER_MISMATCH:{l2}")
                r7, r6 = det_seedmem_scope_row(rd, l2, s, 7), det_seedmem_scope_row(rd, l2, s, 6)
                why = det_seedmem_scope_row_check(r7, 7, b, Z) + det_seedmem_scope_row_check(r6, 6, b, Z)
                outside = not (RATES.NU_MIN <= b["nu"] <= RATES.NU_MAX)
                if why and outside and "W5_STATUS_OUTSIDE_FINE_SUPPORT" in why and \
                        set(why) <= {"W5_STATUS_OUTSIDE_FINE_SUPPORT", "W5_SOBOLEV_FIELD_MISSING"}:
                    pending.append(l2)          # production A-only rates; populations from other lines
                    continue
                if why:
                    missing.append(f"{l2}:{why[0]}")
                    continue
                inv, rsn = det_seedmem_scope_invert(r7, b, t_exp, csob, lib, fault)
                if inv is None:
                    (link if rsn == "FORWARD_LEDGER_MISMATCH" else missing).append(f"{l2}:{rsn}")
                    continue
                rad, br = det_seedmem_scope_rates(r6, b, form, lib)
                if rad is None and det_seedmem_scope_is_link([br]):
                    link.append(f"{l2}:{br}")
                    continue
                coll = (det_seedmem_scope_coll(float(ne7[s]), float(te7[s]), b)
                        if (ne7 is not None and rd["coll"]["supported"]) else None)
                if rad is None or coll is None:
                    missing.append(f"{l2}:{br if rad is None else 'COLLISION'}")
                    continue
                Ra, Rs, Rp = rad
                Cu, Cd = coll
                F = inv["n_l"] * (Ra + Cu) - inv["n_u"] * (Rs + Rp + Cd)
                flows.setdefault(l2, []).append(F)
                pops.setdefault(b["lo"], []).append(inv["n_l"])
                pops.setdefault(b["up"], []).append(inv["n_u"])
            consistent = all(det_seedmem_scope_overlap(x, y) for v in pops.values() for x in v for y in v)
            for l2 in pending:
                b = deck["at"][l2]
                pl, pu = table.get((Z, b["lo"], s)), table.get((Z, b["up"], s))
                if pl and pu and not all(det_seedmem_scope_overlap(x, y) for v in (pl, pu) for x in v for y in v):
                    pl = pu = None
                coll = (det_seedmem_scope_coll(float(ne7[s]), float(te7[s]), b)
                        if (ne7 is not None and rd["coll"]["supported"]) else None)
                if not pl or not pu or coll is None or not consistent:
                    missing.append(f"{l2}:A_ONLY_LEVEL_POPULATION_ABSENT")
                    continue
                n_l = I(max(x.lo for x in pl), pl[0].mid, min(x.hi for x in pl))
                n_u = I(max(x.lo for x in pu), pu[0].mid, min(x.hi for x in pu))
                Cu, Cd = coll
                flows.setdefault(l2, []).append(n_l * Cu - n_u * (I.x(b["A_ul"]) + Cd))
                aonly += 1
                aonly_lines.append(l2)
            dup = any(len(v) > 1 for v in flows.values())
            count_ok = len(set(lines)) == expected and len(lines) == expected
            complete_bb = (not missing) and count_ok and not dup and not link
            if fault == "partial_as_complete" and missing:
                complete_bb = True                  # injected: a partial sum labelled complete
            if not consistent:
                link.append("LEVEL_POPULATION_INCONSISTENT")
            Ft = flows.get(line, [None])[0]
            inflow_up, inflow_lo = [], []
            for l2, Fs in flows.items():
                if l2 == line:
                    continue
                b = deck["at"][l2]
                for F in Fs:
                    if b["up"] == a["up"]:
                        inflow_up.append(F)
                    if b["lo"] == a["up"]:
                        inflow_up.append(det_seedmem_scope_neg(F))
                    if b["up"] == a["lo"]:
                        inflow_lo.append(F)
                    if b["lo"] == a["lo"]:
                        inflow_lo.append(det_seedmem_scope_neg(F))
            Fup = det_seedmem_scope_isum(inflow_up)
            Flo = det_seedmem_scope_isum(inflow_lo)
            sl_lo, sl_up = det_seedmem_scope_sl_of(rd, a)
            cap = cap_status.get((tag, line, s), "ABSENT")
            rec = {"run": tag, "Z": Z, "line": line, "shell": s, "n_connected": len(conn[line]),
                   "n_lines_used": len(set(flows)), "n_a_only_outside_domain": aonly, "missing": ";".join(missing),
                   "count_ok": int(count_ok),
                   "duplicate": int(dup), "complete_bb": int(complete_bb), "level_populations_consistent": int(consistent),
                   "closed": 0, "closure_reason": "BF_AND_LINE_FREE_CHANNELS_NOT_STORED" +
                   ("" if sl_lo == 1 and sl_up == 1 else ";SUPER_LEVEL_ROW") +
                   ("" if cap == "OK" else f";CAPTURE_AUTHORITY_{cap}"),
                   "sl_lo": sl_lo, "sl_up": sl_up, "capture_authority": cap, "link": ";".join(link),
                   "a_only_lines": ";".join(str(x) for x in aonly_lines)}
            REP.det_siiii_repair_cols(rec, "F_target", Ft)
            REP.det_siiii_repair_cols(rec, "F_other_into_upper", Fup if Ft is not None else None)
            REP.det_siiii_repair_cols(rec, "F_other_into_lower", Flo if Ft is not None else None)
            if Ft is not None:
                rec["F_target_sign"] = ("POSITIVE" if Ft.lo > 0 else "NEGATIVE" if Ft.hi < 0 else "ZERO_NOT_SEPARATED")
                tot = Ft + Fup
                if Ft.lo > 0 or Ft.hi < 0:
                    rec["partial_upper_residual_ratio"] = abs(tot.mid) / abs(Ft.mid)
                    # bound from the stored intervals (not the middle values): max |sum| / min |F_target|
                    rec["partial_upper_residual_ratio_interval_hi"] = up(max(abs(tot.lo), abs(tot.hi)) /
                                                                         min(abs(Ft.lo), abs(Ft.hi)))
                    rec["other_opposes_target"] = int((Ft.mid > 0) != (Fup.mid > 0) and Fup.mid != 0.0)
            rec["status"] = "NOT_IDENTIFIABLE"
            if not count_ok or dup or link:
                rec["status"] = "X"
            # registered completeness validator (independent of the flag computation above)
            if rec["complete_bb"] == 1 and not (rec["missing"] == "" and rec["count_ok"] == 1 and rec["duplicate"] == 0
                                                and rec["n_lines_used"] == rec["n_connected"] + 1):
                rec["status"] = "X"
                rec["link"] = (rec["link"] + ";" if rec["link"] else "") + "BUDGET_COMPLETENESS"
            rows.append(rec)
            ident.append({"run": tag, "Z": Z, "line": line, "shell": s, "question": "generation7_bb_budget",
                          "generation": 7, "status": "NOT_IDENTIFIABLE" if rec["status"] != "X" else "X",
                          "reason": rec["closure_reason"] + ("" if complete_bb else ";BB_INCOMPLETE:" + ";".join(missing))})
    return rows, ident


def det_seedmem_scope_capture_check(runs, sample, deck, csob, lib, fault=None):
    """prereg 3.2 (3): W5 call-7 inversion vs the independent generation-7 capture inversion (same populations).
    Returns the per-key authority status consumed by the generation-7 budget and the generation-7 level measure:
    OK, INCOMPATIBLE (inadmissible), or the absence (MISSING / UNMAPPED / W5 not invertible)."""
    rows, why, status = [], [], {}
    for tag, rd in runs.items():
        capd = Path(rd["run"]) / "work/det_obs_capture"
        recs = POP.capture_records(capd, sorted({k[1] for k in sample["keys"]}), set(TARGET["shells"]) |
                                   {k[2] for k in sample["keys"]})
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            r7 = det_seedmem_scope_row(rd, line, s, 7)
            w = det_seedmem_scope_row_check(r7, 7, a, Z)
            inv, rsn = (None, w[0]) if w else det_seedmem_scope_invert(r7, a, rd["t_exp"], csob, lib, fault)
            c = recs.get((line, s))
            cap = None
            if c is not None and c["mapped"] == 1:
                cap = POP.recover(c["tau"], c["S"], a["g_l"], a["g_u"], a["f_lu"], a["lam_cm"], rd["t_exp"])
            ok = None
            if inv is not None and cap is not None:
                ok = det_seedmem_scope_overlap(inv["n_u"], cap["n_u"]) and det_seedmem_scope_overlap(inv["n_l"], cap["n_l"])
                if not ok:
                    why.append(f"{tag}:POP_AUTHORITY_MISMATCH:{line}:{s}")
            capst = "MISSING" if c is None else ("UNMAPPED" if c["mapped"] != 1 else "OK")
            if rsn == "FORWARD_LEDGER_MISMATCH":
                why.append(f"{tag}:FORWARD_LEDGER_MISMATCH:{line}:{s}")
            status[(tag, line, s)] = ("OK" if ok else "INCOMPATIBLE") if ok is not None else \
                (capst if capst != "OK" else f"W5_{rsn}")
            rows.append({"run": tag, "line": line, "shell": s, "w5_inversion": rsn or "OK", "capture": capst,
                         "compatible": None if ok is None else int(ok), "authority": status[(tag, line, s)]})
    return rows, why, status


def det_seedmem_scope_counter_detail(rd, deck, line, s, lib):
    """prereg 4.3 for the counterexample at call 3 (R' NRB): order of S~_meas and the previous J_c, detailed balance
    of the consumed collision form, the two-level source range [radiative-only, collision-only] against the
    measurement, and the production-coefficient collision source against the Planck approximation."""
    a = deck["at"][line]
    r3, r2 = det_seedmem_scope_row(rd, line, s, 3), det_seedmem_scope_row(rd, line, s, 2)
    if r3 is None or r2 is None or det_seedmem_scope_row_check(r3, 3, a, a["Z"]) or \
            det_seedmem_scope_row_check(r2, 2, a, a["Z"]):
        return {"status": "NOT_IDENTIFIABLE"}
    meas = det_seedmem_scope_meas(r3, a, deck["r"][line])
    ne, te = float(rd["dumps"]["ne"][3][s]), float(rd["dumps"]["Te"][3][s])
    Cu, Cd = det_seedmem_scope_coll(ne, te, a)
    dE = abs(a["E_u"] - a["E_l"]) * RATES.EV_TO_ERG
    boltz = a["g_u"] / a["g_l"] * math.exp(-dE / (RATES.K_B * te))
    rad, br = det_seedmem_scope_rates(r2, a, rd["form"], lib)
    zero = (I.x(0.0), I.x(0.0))
    s_coll = det_seedmem_scope_source2((I.x(0.0), I.x(0.0), I.x(0.0)), (Cu, Cd), a)
    s_rad = det_seedmem_scope_source2(rad, zero, a)
    s_pred = det_seedmem_scope_source2(rad, (Cu, Cd), a)
    planck = 2 * H_PLANCK * a["nu"] ** 3 / C_LIGHT ** 2 / math.expm1(H_PLANCK * a["nu"] / (RATES.K_B * te))
    jc2 = float(r2["J_c"])
    out = {"line": line, "shell": s, "call": 3, "rate_branch": br, "T_e_gen3": te, "n_e_gen3": ne,
           "S_meas": meas.j(), "J_c_call2": jc2,
           "S_meas_vs_Jc_call2": "BELOW" if meas.hi < jc2 else ("ABOVE" if meas.lo > jc2 else "NOT_SEPARATED"),
           "collision_ratio_over_boltzmann_minus_1": (Cu.mid / Cd.mid) / boltz - 1.0,
           "S_coll_production_coefficients": None if s_coll is None else s_coll.j(),
           "S_coll_planck_approximation": planck,
           "S_coll_vs_planck_minus_1": None if s_coll is None else s_coll.mid / planck - 1.0,
           "S_rad_only": None if s_rad is None else s_rad.j(), "S_pred": None if s_pred is None else s_pred.j()}
    if s_coll is not None and s_rad is not None:
        lo, hi = min(s_coll.lo, s_rad.lo), max(s_coll.hi, s_rad.hi)
        out["two_level_range"] = [lo, hi]
        out["S_meas_in_two_level_range"] = ("INSIDE" if (meas.lo >= lo and meas.hi <= hi) else
                                            "OUTSIDE" if (meas.hi < lo or meas.lo > hi) else "NOT_SEPARATED")
    return out


def det_seedmem_scope_m(a, b):
    """|log10(b/a)| as an interval for exact positive a, b (outward); exact-zero handling by the caller."""
    r = I.x(b).div(I.x(a))
    lo, hi = math.log10(r.lo), math.log10(r.hi)
    lo, hi = dn(dn(lo)), up(up(hi))
    if lo >= 0:
        return I(lo, math.log10(r.mid), hi)
    if hi <= 0:
        return I(-hi, -math.log10(r.mid), -lo)
    return I(0.0, abs(math.log10(r.mid)), max(-lo, hi))


def det_seedmem_scope_p2_pairs(per_g, fault=None):
    """pair states over the fixed common pairs and the axis code (prereg 5.3)."""
    pair_st = []
    for g1, g2 in P2_PAIRS:
        sts = []
        for g in (g1, g2):
            v = per_g.get(g)
            if v is None:
                sts.append("NOT_IDENTIFIABLE")
            elif v["M"].lo >= P2_CRIT:
                sts.append("REFUTED")
            elif v["missing"]:
                sts.append("NOT_IDENTIFIABLE")
            elif v["M"].hi < P2_CRIT:
                sts.append("SURVIVES")
            else:
                sts.append("UNRESOLVED")
        if fault == "single_change":
            pair_st.append("SURVIVES" if "SURVIVES" in sts else det_seedmem_scope_universal(sts))
        else:
            pair_st.append(det_seedmem_scope_universal(sts))
    if "SURVIVES" in pair_st:
        code = "SURVIVES"
    elif all(p == "REFUTED" for p in pair_st):
        code = "REFUTED"
    elif "UNRESOLVED" in pair_st:
        code = "UNRESOLVED"
    else:
        code = "NOT_IDENTIFIABLE" if "NOT_IDENTIFIABLE" in pair_st else "REFUTED"
    return pair_st, code


def det_seedmem_scope_p2_validate(rows, codes, expected):
    """registered P2 validator, independent of the code path: (1) every registered quantity is accounted for at
    every generation (measured + missing + structural zero = registered count: no cell deletion); (2) the pair
    states and the axis code re-derived from the stored per-generation rows equal the reported ones."""
    why = []
    df = pd.DataFrame(rows)
    for (tag, axis), sub in df.groupby(["run", "axis"]):
        n_exp = expected[axis]
        for r in sub.itertuples():
            if int(r.n_quantities) + int(r.missing) + int(r.structural_zero) != n_exp:
                why.append(f"P2_QUANTITY_ACCOUNTING:{tag}:{axis}:{r.generation}")
        st = {int(r.generation): r for r in sub.itertuples()}
        pair_st = []
        for g1, g2 in P2_PAIRS:
            one = []
            for g in (g1, g2):
                r = st.get(g)
                if r is None:
                    one.append("NOT_IDENTIFIABLE")
                elif r.M_lo >= P2_CRIT:
                    one.append("REFUTED")
                elif int(r.missing):
                    one.append("NOT_IDENTIFIABLE")
                elif r.M_hi < P2_CRIT:
                    one.append("SURVIVES")
                else:
                    one.append("UNRESOLVED")
            pair_st.append("SURVIVES" if one == ["SURVIVES", "SURVIVES"] else det_seedmem_scope_universal_check(one))
        rep = codes.get(tag, {}).get(axis, {})
        if list(rep.get("pairs", {}).values()) != pair_st:
            why.append(f"P2_PAIR_RECOMPUTE:{tag}:{axis}")
        # final axis state: an existence claim over the fixed pairs (prereg 5.3)
        if "SURVIVES" in pair_st:
            fin = "SURVIVES"
        elif set(pair_st) == {"REFUTED"}:
            fin = "REFUTED"
        elif "UNRESOLVED" in pair_st:
            fin = "UNRESOLVED"
        elif "NOT_IDENTIFIABLE" in pair_st:
            fin = "NOT_IDENTIFIABLE"
        else:
            fin = "X"
        if rep.get("state") != fin or rep.get("code") != CODE.get(fin):
            why.append(f"P2_CODE_RECOMPUTE:{tag}:{axis}")
    return why


def det_seedmem_scope_p2(runs, sample, deck, csob, lib, fault=None, cap_status=None):
    """prereg 5.3: adjacent-generation changes of four registered axes; two consecutive small changes over the fixed
    common pairs (4,5),(5,6),(6,7).  J-bar carries the stored W5 bound; generation-7 level populations need the
    independent capture authority of the line they come from."""
    rows, codes = [], {}
    cap_status = cap_status or {}
    levels = sorted({(Z, deck["at"][line][w], s) for Z, line, s in sample["keys"] for w in ("lo", "up")})
    shells = sorted({k[2] for k in sample["keys"]})
    expected = {"ion": len(ION_QTY) * len(shells), "ne": len(shells), "jbar": len(sample["keys"]), "level": len(levels)}
    for tag, rd in runs.items():
        D = rd["dumps"]
        series = {"ion": {}, "ne": {}, "jbar": {}, "level": {}}
        for s in shells:
            for Z, q in ION_QTY:
                series["ion"][(Z, q, s)] = {g: (float(D["ion"][g][(Z, q)][s]) if g in D["ion"] else None)
                                            for g in POP_GENS}
            series["ne"][s] = {g: (float(D["ne"][g][s]) if g in D["ne"] else None) for g in POP_GENS}
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            ser = {}
            for g in MODEL_CALLS:
                r = det_seedmem_scope_row(rd, line, s, g)
                if r is None or det_seedmem_scope_row_check(r, g, a, Z):
                    ser[g] = None
                    continue
                jb, je = float(r["Jbar"]), float(r["Jbar_err_upper"])
                ser[g] = I(dn(jb - je), jb, up(jb + je))
            series["jbar"][(line, s)] = ser
        rec_pop = {}
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            for g in MODEL_CALLS:
                r = det_seedmem_scope_row(rd, line, s, g)
                if r is None or det_seedmem_scope_row_check(r, g, a, Z):
                    continue
                if g == 7 and cap_status.get((tag, line, s)) != "OK":
                    continue
                inv, _ = det_seedmem_scope_invert(r, a, rd["t_exp"], csob, lib, fault)
                if inv is None:
                    continue
                rec_pop.setdefault((Z, a["lo"], s, g), []).append(inv["n_l"])
                rec_pop.setdefault((Z, a["up"], s, g), []).append(inv["n_u"])
        for Z, lev, s in levels:
            ser = {}
            for g in MODEL_CALLS:
                v = rec_pop.get((Z, lev, s, g))
                if not v or not all(det_seedmem_scope_overlap(x, y) for x in v for y in v):
                    ser[g] = None
                else:
                    ser[g] = I(max(x.lo for x in v), v[0].mid, min(x.hi for x in v))
            series["level"][(Z, lev, s)] = ser
        for axis, qs in series.items():
            gens = POP_GENS if axis in ("ion", "ne") else MODEL_CALLS
            per_g = {}
            for g in gens[1:]:
                ms, miss, zero = [], 0, 0
                for qk, ser in qs.items():
                    va, vb = ser.get(g - 1), ser.get(g)
                    if va is None or vb is None:
                        miss += 1
                        continue
                    A_ = va if isinstance(va, I) else I.x(va)
                    B_ = vb if isinstance(vb, I) else I.x(vb)
                    if A_.hi == 0.0 and B_.hi == 0.0 and A_.lo == 0.0 and B_.lo == 0.0:
                        zero += 1
                        continue
                    if (A_.hi == 0.0) != (B_.hi == 0.0):
                        if fault == "drop_zero_cross":
                            continue
                        ms.append(I(math.inf, math.inf, math.inf))
                        continue
                    if not (A_.lo > 0 and B_.lo > 0):
                        miss += 1
                        continue
                    if isinstance(va, I) or isinstance(vb, I):
                        r = B_.div(A_)
                        lo, hi = dn(dn(math.log10(r.lo))), up(up(math.log10(r.hi)))
                        m = I(max(0.0, lo) if lo > 0 else (0.0 if hi > 0 else -hi), abs(math.log10(r.mid)),
                              max(abs(lo), abs(hi)))
                    else:
                        m = det_seedmem_scope_m(va, vb)
                    if fault == "drop_large" and m.lo >= P2_CRIT:
                        continue
                    ms.append(m)
                M = I(max((x.lo for x in ms), default=0.0), max((x.mid for x in ms), default=0.0),
                      max((x.hi for x in ms), default=0.0))
                per_g[g] = {"M": M, "missing": miss, "structural_zero": zero, "n": len(ms)}
                rows.append({"run": tag, "axis": axis, "generation": g, "pair": f"{g - 1}->{g}", "M_lo": M.lo,
                             "M_mid": M.mid, "M_hi": M.hi, "n_quantities": len(ms), "missing": miss,
                             "structural_zero": zero, "registered_quantities": expected[axis],
                             "small": "SURVIVES" if (M.hi < P2_CRIT and not miss) else
                             ("REFUTED" if M.lo >= P2_CRIT else ("NOT_IDENTIFIABLE" if miss else "UNRESOLVED"))})
            pair_st, code = det_seedmem_scope_p2_pairs(per_g, fault)
            first = next((f"{a_}->{b_}" for (a_, b_), p in zip(P2_PAIRS, pair_st) if p == "SURVIVES"), None)
            codes.setdefault(tag, {})[axis] = {"code": CODE[code], "state": code, "pairs": dict(zip(
                [f"{a_},{b_}" for a_, b_ in P2_PAIRS], pair_st)), "first_two_small": first}
    return rows, codes, det_seedmem_scope_p2_validate(rows, codes, expected)


def det_seedmem_scope_identifiability(runs, sample, deck, cap_status=None):
    rows = []
    cap_status = cap_status or {}
    for tag, rd in runs.items():
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            for c in range(1, 8):
                r = det_seedmem_scope_row(rd, line, s, c)
                w = det_seedmem_scope_row_check(r, c, a, Z)
                st = "IDENTIFIABLE" if not w else ("X" if det_seedmem_scope_is_link(w) else "NOT_IDENTIFIABLE")
                if c == 1 and st == "IDENTIFIABLE":
                    st, w = "NOT_IDENTIFIABLE", ["INIT_NO_SOBOLEV_LEDGER"]
                rows.append({"run": tag, "Z": Z, "line": line, "shell": s, "question": "w5_sobolev_ledger",
                             "generation": c, "status": st, "reason": ";".join(w)})
            for g in (2,):
                rows.append({"run": tag, "Z": Z, "line": line, "shell": s, "question": "collisional_response_n_e",
                             "generation": g, "status": "IDENTIFIABLE" if g in rd["dumps"]["ne"] else "NOT_IDENTIFIABLE",
                             "reason": "" if g in rd["dumps"]["ne"] else "NO_STORED_N_E_AUTHORITY"})
            cs = cap_status.get((tag, line, s), "ABSENT")
            rows.append({"run": tag, "Z": Z, "line": line, "shell": s, "question": "generation7_population_authority",
                         "generation": 7, "status": "IDENTIFIABLE" if cs == "OK" else
                         ("X" if cs == "INCOMPATIBLE" else "NOT_IDENTIFIABLE"),
                         "reason": "" if cs == "OK" else f"CAPTURE_{cs}"})
    for q, why in REGISTERED_5.items():
        rows.append({"run": "ALL", "Z": None, "line": None, "shell": None, "question": q, "generation": None,
                     "status": "NOT_IDENTIFIABLE", "reason": why})
    return rows


UNDECIDED = {4: "UNRESOLVED (interval meets the comparison line)", 5: "NOT_IDENTIFIABLE (stored authority absent)",
             "X": "INADMISSIBLE (link / implementation failure)"}


def det_seedmem_scope_next(sets_model, cross_code, budget_rows, p2codes, coll, sl, mlevel, validity):
    """prereg 5.4 decision table (consumption rules; not a run authorisation).  Codes 1 (refuted), 4 (unresolved),
    5 (not identifiable) and X (inadmissible) lead to different entries; only 1 is read as a refutation, and an X
    scope yields no decision at all (no extension, no pass)."""
    R = sets_model.get("R", {})
    dec = []
    xmsg = "INADMISSIBLE scope (a registered check could not reproduce it): no decision; report before any use"
    ctrl = R.get("control_1206", {}).get("code")
    dec.append({"result": "1206 control (R' NRB, line 600693, W5 shells, calls 3-7)", "code": ctrl,
                "evidence": "model.csv rows run=R line=600693; result.json codes.H-MODEL.R.control_1206",
                "next": xmsg if ctrl == "X" else
                ("keep P1 source-match range (line 600693, W5 shells, calls 3-7) and the 1e-3 comparison line"
                 if ctrl == 0 else f"P1 source-match control not reproduced (code {ctrl}): report before any P1 "
                                   "registration")})
    for st in ("SiIII_other", "SIII"):
        c = R.get(st, {}).get("code")
        if c == "X":
            nxt = xmsg
        elif c == 1:
            nxt = f"do not generalise the source-match universal condition to the fixed {st} set"
        elif c == 0:
            nxt = f"offer the whole fixed {st} set as a conditional extension candidate (no pass-only subset)"
        else:
            nxt = f"{st} set undecided, {UNDECIDED.get(c, c)}: keep it outside P1; not a refutation"
        dec.append({"result": f"{st} two-level NRB source model (R')", "code": c,
                    "evidence": f"model.csv run=R set={st}; eligible refuted cells "
                                f"{R.get(st, {}).get('eligible_refuted_cells')}/{R.get(st, {}).get('cells')}; "
                                f"collision branch {coll['R']['branch']} (deck collision tables not consumed: "
                                f"{not coll['R']['matrix_consumes_col_data']}) — model failure is conditional on that "
                                f"branch",
                    "next": nxt})
    if cross_code == "X":
        dec.append({"result": "NRB rate invariance", "code": "X", "evidence": "cross_substitution.csv", "next": xmsg})
    elif cross_code == 0:
        dec.append({"result": "production NRB rates invariant under exchange of L at fixed c, tau", "code": 0,
                    "evidence": "cross_substitution.csv nrb_rates_invariant_under_L and the per-combination "
                                "production rates",
                    "next": "split P1 into (a) explicit self-term removal of the rates and (b) reduced-model source match"})
        if any(R.get(st, {}).get("code") == 1 for st in ("SiIII_other", "SIII")):
            dec.append({"result": "rates invariant but the reduced source model refuted outside 1206", "code": 1,
                        "evidence": "the two rows above",
                        "next": "P1 refutation wording must distinguish NRB implementation failure from reduced-model "
                                "failure"})
    else:
        dec.append({"result": "NRB rate invariance", "code": cross_code,
                    "evidence": "cross_substitution.csv",
                    "next": f"rate self-term removal not established ({UNDECIDED.get(cross_code, cross_code)}): P1 (a) "
                            "stays a hypothesis"})
    br = pd.DataFrame(budget_rows)
    miss = sorted({m.split(":", 1)[1] for v in br.missing.dropna() for m in str(v).split(";") if ":" in m}) \
        if len(br) else []
    mcode = "X" if any(v.get("code") == "X" for v in mlevel.values()) else 5
    dec.append({"result": "generation-7 bound-bound budgets", "code": mcode,
                "evidence": f"pathways.csv: complete bound-bound sets {int(br.complete_bb.sum()) if len(br) else 0}/"
                            f"{len(br)} (run x key); unrecovered connected transitions by reason {miss}; solve units: "
                            f"cutoff K={sl['R']['cutoff']}, active={sl['R']['active']}",
                "next": xmsg if mcode == "X" else "closure not identifiable; minimal storage needed at the consumed "
                        "generation: per-level bf "
                        "rates, the SE row residual per solve unit, the full-level -> solve-unit map with the "
                        "within-unit fractions, and the populations of connected transitions the W5 ledger cannot "
                        "invert (non-positive tau, SRCE_CHK, outside the fine support)"})
    for tag, ax in p2codes.items():
        ion_ne = [ax.get("ion", {}).get("code"), ax.get("ne", {}).get("code")]
        rad_lv = [ax.get("jbar", {}).get("code"), ax.get("level", {}).get("code")]
        if any(c == "X" for c in ion_ne + rad_lv):
            nxt = xmsg + " (P2 axes " + ", ".join(k for k, v in ax.items() if v.get("code") == "X") + ")"
        elif any(c == 1 for c in ion_ne):
            nxt = "P2 stationarity not reached on ion sums / n_e (refuted); no substitution by other measures"
        elif any(c in (4, 5, "X") for c in ion_ne):
            nxt = "P2 on ion sums / n_e undecided (" + ", ".join(str(UNDECIDED.get(c)) for c in ion_ne if c != 0) + \
                  "); not a refutation"
        elif all(c == 0 for c in rad_lv):
            nxt = "only 'two consecutive small changes of the registered W5 measures' may be claimed"
        elif any(c == 1 for c in rad_lv):
            nxt = "ion sums / n_e small but radiation or level measures refuted: register them separately in the next P2"
        else:
            nxt = "ion sums / n_e small; radiation or level measures undecided: register them separately in the next P2"
        dec.append({"result": f"P2 measures, run {tag}", "codes": ax, "evidence": "stationarity.csv",
                    "next": nxt})
    return {"schema": "DET_SEEDMEM_SCOPE_NEXT_V2", "validity": validity,
            "provisional": validity != "VALID", "decisions": dec,
            "unchanged": ["1e-3 and 0.0043 dex are not re-derived from residual maxima or quantiles",
                          "call 8+, W5-outside shells, field convergence and two-seed converged solutions are outside "
                          "this evidence"],
            "not_authorised": "no NRB run, production change, collision-data change, constant repair or extension "
                              "follows from this table"}


# ------------------------------------------------------------------ analyze
def det_seedmem_scope_status_check(lo, hi):
    """independent re-derivation of the registered |d| <= 1e-3 state from the stored interval (validator)."""
    if not (isinstance(lo, float) and isinstance(hi, float)) or math.isnan(lo) or math.isnan(hi):
        return "NOT_IDENTIFIABLE"
    alo = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
    ahi = max(abs(lo), abs(hi))
    if ahi <= CRIT:
        return "SURVIVES"
    if alo > CRIT:
        return "REFUTED"
    return "UNRESOLVED"


def det_seedmem_scope_validate(md, cr, br, sample, runs, sets_model, set_states):
    """registered row accounting and recomputation (G-READOUT): every (run, key, call) row exists once; every stored
    state follows from the stored d interval; every set code follows from its cell states by the prereg 6.3 order."""
    why = []
    n_keys, tags = len(sample["keys"]), list(runs)
    want = {(t, Z, l, s, k) for t in tags for Z, l, s in sample["keys"] for k in MODEL_CALLS}
    got = list(zip(md.run, md.Z, md.line, md.shell, md.call)) if len(md) else []
    if len(got) != len(want) or set(got) != want:
        why.append(f"MODEL_ROW_ACCOUNTING:{len(got)}/{len(want)}")
    if len(cr) != n_keys * len(CROSS_CALLS):
        why.append(f"CROSS_ROW_ACCOUNTING:{len(cr)}/{n_keys * len(CROSS_CALLS)}")
    if len(br) != n_keys * len(tags):
        why.append(f"BUDGET_ROW_ACCOUNTING:{len(br)}/{n_keys * len(tags)}")
    dlo = pd.to_numeric(md["d_lo"], errors="coerce") if "d_lo" in md.columns else pd.Series([np.nan] * len(md))
    dhi = pd.to_numeric(md["d_hi"], errors="coerce") if "d_hi" in md.columns else pd.Series([np.nan] * len(md))
    for r, lo_, hi_ in zip(md.itertuples(), dlo, dhi):
        if r.status == "X":
            continue
        exp = det_seedmem_scope_status_check(float(lo_), float(hi_))
        if exp != r.status:
            why.append(f"STATUS_RECOMPUTE:{r.run}:{r.line}:{r.shell}:{r.call}:{r.status}!={exp}")
            if len(why) > 50:
                break
    for tag, sets in sets_model.items():
        for st, rec in sets.items():
            if det_seedmem_scope_universal_check(set_states[tag][st]) != rec["state"]:
                why.append(f"SET_CODE_RECOMPUTE:{tag}:{st}")
    return why


def det_seedmem_scope_q(v):
    v = pd.to_numeric(pd.Series(list(v), dtype=object), errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    qs = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(len(v)), "min": qs[0], "p05": qs[1], "p25": qs[2], "p50": qs[3], "p75": qs[4], "p95": qs[5],
            "max": qs[6]}


def det_seedmem_scope_summary(md, cr, br, cap_rows, sample, conn, runs, sets_of):
    """the aggregates the verdict quotes (prereg 7 readout; descriptive, no new criterion).  Small-residual counts
    use the registered 1e-3 comparison line of prereg 4.4 and are bound-bound only (not closure)."""
    out = {"schema": "DET_SEEDMEM_SCOPE_SUMMARY_V1"}
    sl_lines = sorted({k[1] for k in sample["keys"]})
    union = sorted({l2 for v in conn.values() for l2 in v})
    md = md.copy()
    for c_ in ("d_lo", "d_mid", "d_hi"):
        md[c_] = pd.to_numeric(md[c_], errors="coerce") if c_ in md.columns else np.nan
    out["transitions"] = {"distinct_sample_lines": len(sl_lines), "connected_union": len(union),
                          "auxiliary_excluding_sample_lines": len(set(union) - set(sl_lines)),
                          "all_used_lines": len(set(union) | set(sl_lines)),
                          "sample_keys": len(sample["keys"])}
    md = md.copy()
    md["set"] = [sets_of[(z, l, s)] for z, l, s in zip(md.Z, md.line, md.shell)]
    md["unit_class"] = [("lo_" + ("explicit" if a == 1 else "lumped")) + "/" + ("up_" + ("explicit" if b == 1 else "lumped"))
                        for a, b in zip(md.sl_lo, md.sl_up)]
    m = {}
    for (tag, st), sub in md.groupby(["run", "set"]):
        d = m.setdefault(tag, {})
        keys_all = sub.groupby(["line", "shell"]).status.apply(lambda x: bool((x == "SURVIVES").all()))
        d[st] = {"cells_by_status": sub.status.value_counts().to_dict(),
                 "keys_all_calls_survive": int(keys_all.sum()), "keys": int(len(keys_all)),
                 "d_mid_quantiles": det_seedmem_scope_q(sub.d_mid),
                 "max_abs_d_mid": float(sub.d_mid.abs().max()) if sub.d_mid.notna().any() else None,
                 "refuted_negative_d": int(((sub.status == "REFUTED") & (sub.d_mid < 0)).sum()),
                 "refuted_positive_d": int(((sub.status == "REFUTED") & (sub.d_mid > 0)).sum()),
                 "survives_by_unit_class": {k: f"{int((g.status == 'SURVIVES').sum())}/{len(g)}"
                                            for k, g in sub.groupby("unit_class")},
                 "survives_by_shell": {int(k): f"{int((g.status == 'SURVIVES').sum())}/{len(g)}"
                                       for k, g in sub.groupby("shell")}}
    out["model"] = m
    cl, cs = TARGET["counter"]
    out["counter_d_mid"] = {tag: {int(r.call): r.d_mid for r in md[(md.run == tag) & (md.line == cl) &
                                                                     (md.shell == cs)].itertuples()} for tag in runs}
    c = cr[cr.status == "SURVIVES"] if len(cr) else cr
    if len(c):
        share = (c.off_D_L.abs() / (c.off_D_L.abs() + c.off_D_z.abs())).tolist()
        dm = c[c.delta_S_meas.notna()]
        out["cross"] = {"rows_by_status": cr.status.value_counts().to_dict(),
                        "nrb_invariant": int(cr.nrb_rates_invariant_under_L.fillna(0).sum()),
                        "nrb_D_L_max_abs": float(c.nrb_D_L.abs().max()),
                        "off_D_L_share_quantiles": det_seedmem_scope_q(share),
                        "off_D_L_share_median_by_call": {int(k): float(np.median(
                            (g.off_D_L.abs() / (g.off_D_L.abs() + g.off_D_z.abs())).values)) for k, g in c.groupby("call")},
                        "sign_agreement_off_D_L_vs_delta_S_meas": f"{int((np.sign(dm.off_D_L) == np.sign(dm.delta_S_meas)).sum())}/{len(dm)}",
                        "abs_delta_e_model_over_abs_delta_S_meas_quantiles": det_seedmem_scope_q(
                            (dm.delta_e_model_off.abs() / dm.delta_S_meas.abs()).tolist())}
    b = br.copy()
    for c_ in ("partial_upper_residual_ratio", "partial_upper_residual_ratio_interval_hi", "other_opposes_target",
               "F_target_sign", "a_only_lines", "missing"):
        if c_ not in b.columns:
            b[c_] = np.nan
    for c_ in ("partial_upper_residual_ratio", "partial_upper_residual_ratio_interval_hi"):
        b[c_] = pd.to_numeric(b[c_], errors="coerce")
    b["upper_unit"] = np.where(b.sl_up == 1, "explicit", "lumped")
    pw = {}
    for tag, sub in b.groupby("run"):
        cb = sub[(sub.complete_bb == 1) & sub.partial_upper_residual_ratio.notna()]
        miss = [m_.split(":", 1)[1] for v in sub.missing.dropna() for m_ in str(v).split(";") if ":" in m_]
        aol = {x for v in sub.a_only_lines.dropna() for x in str(v).split(";") if x}
        pw[tag] = {"status": sub.status.value_counts().to_dict(), "complete_bb": int(sub.complete_bb.sum()),
                   "rows": int(len(sub)), "other_opposes_target": int(sub.other_opposes_target.fillna(0).sum()),
                   "F_target_sign": sub.F_target_sign.value_counts().to_dict(),
                   "missing_incidences_by_reason": pd.Series(miss).value_counts().to_dict() if miss else {},
                   "a_only_incorporations": int(sub.n_a_only_outside_domain.sum()),
                   "a_only_distinct_lines": len(aol),
                   "upper_bb_residual_ratio_by_upper_unit": {
                       k: {"quantiles_of_middle_values": det_seedmem_scope_q(g.partial_upper_residual_ratio),
                           "middle_value_le_1e-3_bb_only_not_closure":
                               f"{int((g.partial_upper_residual_ratio <= CRIT).sum())}/{len(g)}",
                           "interval_guaranteed_le_1e-3": f"{int((g.partial_upper_residual_ratio_interval_hi <= CRIT).sum())}/{len(g)}",
                           "interval_upper_bound_max": float(g.partial_upper_residual_ratio_interval_hi.max())}
                       for k, g in cb.groupby("upper_unit")}}
    out["pathways"] = pw
    cp = pd.DataFrame(cap_rows)
    out["capture_authority"] = {t: g.authority.value_counts().to_dict() for t, g in cp.groupby("run")} if len(cp) else {}
    out["solve_units"] = {t: {"cutoff": rd["sl"]["cutoff"], "active": rd["sl"]["active"],
                              "target_ion_levels_explicit": rd["sl"]["explicit_levels"],
                              "target_ion_levels_lumped": rd["sl"]["lumped_levels"]} for t, rd in runs.items()}
    k0 = md[(md.run == "R") & (md.call == MODEL_CALLS[0])]
    out["sample_keys_by_unit_class"] = {f"{st}:{uc}": int(len(g)) for (st, uc), g in k0.groupby(["set", "unit_class"])}
    return out


def det_seedmem_scope_block(rec, why):
    """a scope whose code a registered check could not reproduce becomes X; the code it would have had is kept as
    provisional (never consumed by the decision table)."""
    if rec.get("code") != "X":
        rec["provisional_code"] = rec.get("code")
    rec["code"], rec["state"] = "X", "X"
    rec.setdefault("blocked_by", []).append(why)


def det_seedmem_scope_propagate(vwhy, sets_model, counter_rec, cross_rec, mlevel, p2codes, sample, counter_key):
    """prereg 6.3: route every validator finding to the scope it invalidates (model sets, the counterexample, the
    cross-substitution proposition, the generation-7 budgets, a P2 axis); unaffected scopes keep their codes."""
    line_z = {l: z for z, l, _ in sample["keys"]}
    sets_of = {k: st for st, ks in sample["sets"].items() for k in ks}
    for w in vwhy:
        f = w.split(":")
        kind = f[0]
        if kind == "MODEL_ROW_ACCOUNTING":
            for tag in sets_model:
                for st in sets_model[tag]:
                    det_seedmem_scope_block(sets_model[tag][st], w)
            det_seedmem_scope_block(counter_rec, w)
        elif kind == "STATUS_RECOMPUTE" and len(f) >= 5:
            tag, line, shell = f[1], int(f[2]), int(f[3])
            st = sets_of.get((line_z.get(line), line, shell))
            if tag in sets_model and st in sets_model[tag]:
                det_seedmem_scope_block(sets_model[tag][st], w)
            if tag == "R" and (line, shell) == counter_key:
                det_seedmem_scope_block(counter_rec, w)
        elif kind == "SET_CODE_RECOMPUTE" and len(f) >= 3:
            if f[1] in sets_model and f[2] in sets_model[f[1]]:
                det_seedmem_scope_block(sets_model[f[1]][f[2]], w)
        elif kind == "CROSS_ROW_ACCOUNTING":
            det_seedmem_scope_block(cross_rec, w)
        elif kind == "BUDGET_ROW_ACCOUNTING":
            for tag in mlevel:
                det_seedmem_scope_block(mlevel[tag], w)
        elif kind.startswith("P2_") and len(f) >= 3:
            if f[1] in p2codes and f[2] in p2codes[f[1]]:
                det_seedmem_scope_block(p2codes[f[1]][f[2]], w)
    return sets_model, counter_rec, cross_rec, mlevel, p2codes


def det_seedmem_scope_analyze(census, runs_paths, src, out, lib, target=TARGET, fault=None, prereg=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = Path(src)
    src_text = (src / "lumina_plasma.c").read_text(errors="replace")
    csob = det_seedmem_scope_csob(src)
    why = []
    if csob is None or csob != POP.C_SOB:
        why.append("C_SOB_SOURCE_MISMATCH")
    sample = det_seedmem_scope_sample(census, target, fault)
    why += sample["why"]
    deck = det_seedmem_scope_deck(Path(runs_paths["E"]) / "input/model", target["Z"], target["ion"])
    why += deck["why"]
    deck_ids = {}
    for tag, rp in runs_paths.items():
        ent = dict(reversed(ln.split(None, 1)) for ln in (Path(rp) / "input/deck.sha256").read_text().split("\n")
                   if ln.strip())
        deck_ids[tag] = {k.strip(): ent.get(k) for k in ("line_list.csv", "levels.csv")}
    if len({json.dumps(v, sort_keys=True) for v in deck_ids.values()}) != 1:
        why.append("DECK_DIFFERS_BETWEEN_RUNS")
    for k in ("line_list.csv", "levels.csv"):
        if deck_ids["E"].get(k) != sha(Path(runs_paths["E"]) / "input/model" / k):
            why.append(f"DECK_FILE_NOT_REGISTERED:{k}")
    missing_lines = sorted({k[1] for k in sample["keys"]} - set(deck["at"]))
    if missing_lines:
        why.append(f"SAMPLE_LINE_NOT_IN_DECK:{missing_lines[:5]}")
    if why:
        res = {"schema": "DET_SEEDMEM_SCOPE_RESULT_V2", "validity": "X", "reasons": why, "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    conn = det_seedmem_scope_connect(sample["keys"], deck["at"])
    deck["r"] = {l: det_seedmem_scope_r(deck["at"][l], csob) for l in {k[1] for k in sample["keys"]} | {
        l2 for v in conn.values() for l2 in v}}
    keys = {(k[1], k[2]) for k in sample["keys"]} | {(l2, k[2]) for k in sample["keys"] for l2 in conn[k[1]]}
    cuda = src / "lumina_cuda.cu"
    cuda_text = cuda.read_text(errors="replace") if cuda.exists() else ""
    runs = {tag: det_seedmem_scope_run(p, tag, keys, src_text, deck, fault, cuda_text) for tag, p in runs_paths.items()}
    for tag, rd in runs.items():
        why += rd["why"]
        if rd["form"] != target["forms"][tag]:
            why.append(f"{tag}:RATE_FORM_NOT_REGISTERED:{rd['form']}")
    for tag, rd in runs.items():
        for Z, line, s in sample["keys"]:
            for c in range(1, 8):
                if fault == "skip_row_check":
                    break                       # injected: the ledger reaches the production function unchecked
                w = det_seedmem_scope_row_check(det_seedmem_scope_row(rd, line, s, c), c, deck["at"][line], Z)
                if det_seedmem_scope_is_link(w):
                    why.append(f"{tag}:SAMPLE_ROW_LINK:{line}:{s}:{c}:{w[0]}")
    cap_rows, cap_why, cap_status = det_seedmem_scope_capture_check(runs, sample, deck, csob, lib, fault)
    why += cap_why
    timing = {"inputs_s": time.time() - t0}
    if why:
        res = {"schema": "DET_SEEDMEM_SCOPE_RESULT_V2", "validity": "X", "reasons": why[:200], "n_reasons": len(why),
               "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        pd.DataFrame(cap_rows).to_csv(out / "capture_check.csv", index=False)
        return res
    model_rows, ident_m = det_seedmem_scope_model(runs, sample, deck, csob, lib, fault)
    if fault == "drop_failed_cell":
        model_rows = [r for r in model_rows if r["status"] != "REFUTED"]
    cross_rows = det_seedmem_scope_cross(runs, sample, deck, lib, fault)
    budget_rows, ident_b = det_seedmem_scope_budget(runs, sample, deck, conn, csob, lib, fault, cap_status)
    p2_rows, p2codes, p2why = det_seedmem_scope_p2(runs, sample, deck, csob, lib, fault, cap_status)
    ident = det_seedmem_scope_identifiability(runs, sample, deck, cap_status) + ident_m + ident_b
    timing["compute_s"] = time.time() - t0 - timing["inputs_s"]
    md = pd.DataFrame(model_rows)
    sets_of = {k: st for st, ks in sample["sets"].items() for k in ks}
    sets_model, set_states, counter = {}, {}, {}
    for tag in runs:
        sub = md[md.run == tag] if len(md) else md
        set_states[tag] = {st: [] for st in sample["sets"]}
        for r in sub.itertuples():
            set_states[tag][sets_of[(r.Z, r.line, r.shell)]].append(r.status)
        sets_model[tag] = {st: det_seedmem_scope_set_code(v, fault) for st, v in set_states[tag].items()}
        cl, cs = target["counter"]
        sub2 = sub[(sub.line == cl) & (sub.shell == cs)] if len(sub) else sub
        counter[tag] = {int(r.call): r.status for r in sub2.itertuples()}
    cr = pd.DataFrame(cross_rows)
    cst = list(cr.status) if len(cr) else []
    cross_rec = det_seedmem_scope_set_code(cst)
    cross_code = cross_rec["code"]
    br = pd.DataFrame(budget_rows)
    vwhy = det_seedmem_scope_validate(md, cr, br, sample, runs, sets_model, set_states) + p2why
    n_x = {"model": int((md.status == "X").sum()) if len(md) else 0, "cross": int(cst.count("X")),
           "budget": int((br.status == "X").sum()) if len(br) else 0}
    validity, reasons = "VALID", []
    if vwhy or any(n_x.values()):
        validity = "X"
        reasons = vwhy[:200] + [f"X_ROWS:{k}={v}" for k, v in n_x.items() if v]
    counter_rec = det_seedmem_scope_set_code(list(counter.get("R", {}).values()))
    mlevel = {tag: {"code": "X" if ((br.run == tag) & (br.status == "X")).any() else 5,
                    "reason": "closure not identifiable: bf / line-free channels and the SE row residual per solve unit "
                              "are not stored",
                    "complete_bb_targets": int(br[(br.run == tag)].complete_bb.sum()) if len(br) else 0,
                    "targets": int((br.run == tag).sum()) if len(br) else 0} for tag in runs}
    sets_model, counter_rec, cross_rec, mlevel, p2codes = det_seedmem_scope_propagate(
        vwhy, sets_model, counter_rec, cross_rec, mlevel, p2codes, sample, target["counter"])
    cross_code = cross_rec["code"]
    codes = {
        "H-MODEL": {tag: {st: rec["code"] for st, rec in v.items()} for tag, v in sets_model.items()},
        "H-MODEL_sets": sets_model,
        "H-COUNTEREXAMPLE": {"line": target["counter"][0], "shell": target["counter"][1],
                             "R_two_level_nrb_applicability": counter_rec["code"], "R_record": counter_rec,
                             "per_call": counter},
        "H-LOCAL": cross_code,
        "H-LOCAL_record": cross_rec,
        "H-MULTILEVEL": mlevel,
        "H-P2": p2codes,
        "registered_not_identifiable": {q: 5 for q in REGISTERED_5},
    }
    if validity == "X":
        codes["note"] = "validity X: scopes a check could not reproduce are X (provisional_code kept, never consumed); " \
                        "the remaining codes are provisional until validity is restored"
    cl, cs = target["counter"]
    codes["H-COUNTEREXAMPLE"]["call3_detail_R"] = det_seedmem_scope_counter_detail(runs["R"], deck, cl, cs, lib)
    coll = {tag: rd["coll"] for tag, rd in runs.items()}
    sl = {tag: {k: v for k, v in rd["sl"].items() if k not in ("unit", "size")} for tag, rd in runs.items()}
    nxt = det_seedmem_scope_next(sets_model, cross_code, budget_rows, p2codes, coll, sl, mlevel, validity)
    summ = det_seedmem_scope_summary(md, cr, br, cap_rows, sample, conn, runs, sets_of) if len(md) and len(br) else {}
    sample_json = {"schema": "DET_SEEDMEM_SCOPE_SAMPLE_V2", "authority": str(census), "authority_sha256": sha(census),
                   "selection": {"Z": list(target["Z"]), "ion": target["ion"], "shells": list(target["shells"]),
                                 "beta2_E_below": target["beta_max"]},
                   "counts": sample["counts"], "n_keys": len(sample["keys"]), "keys": sample["keys"],
                   "sets": {k: len(v) for k, v in sample["sets"].items()}, "control": target["control"],
                   "counter": list(target["counter"]),
                   "connected_transitions_by_sample_line": {str(l): v for l, v in conn.items()},
                   "level_set": sorted({(Z, deck["at"][line][w]) for Z, line, _ in sample["keys"] for w in ("lo", "up")}),
                   "solve_unit_sizes_of_sample_by_run": {tag: {str(k[1]): list(det_seedmem_scope_sl_of(rd, deck["at"][k[1]]))
                                                               for k in sample["keys"]} for tag, rd in runs.items()},
                   "solve_unit_authority": sl}
    (out / "sample.json").write_text(json.dumps(sample_json, indent=1, default=str))
    pd.DataFrame(ident).to_csv(out / "identifiability.csv", index=False)
    md.to_csv(out / "model.csv", index=False)
    cr.to_csv(out / "cross_substitution.csv", index=False)
    br.to_csv(out / "pathways.csv", index=False)
    pd.DataFrame(p2_rows).to_csv(out / "stationarity.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(out / "capture_check.csv", index=False)
    (out / "next_contract.json").write_text(json.dumps(nxt, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    res = {"schema": "DET_SEEDMEM_SCOPE_RESULT_V2", "validity": validity, "reasons": reasons, "fault": fault,
           "codes": codes, "collision_authority": coll, "solve_unit_authority": sl,
           "rate_forms": {t: rd["form"] for t, rd in runs.items()},
           "counts": {"keys": len(sample["keys"]), "model_rows": len(md), "cross_rows": len(cr),
                      "budget_rows": len(br), "x_rows": n_x},
           "timing_s": timing,
           "claim_ceiling": "offline readout of stored states; no production change, no physical correctness, no "
                            "public agreement, no causal pathway apportionment, no run authorisation"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ fixtures (direct tests, prereg 8.2)
FX_SHELLS = (10, 15)
FX_CUTOFF = 4
FX_NS = 16
FX_TEXP = float.fromhex("0x1.9ae8p+20")
FX_TARGET = {"Z": (14, 16), "ion": 2, "beta_max": 1e-3, "counts": {14: 6, 16: 2}, "control": 0, "counter": (3, 10),
             "shells": FX_SHELLS, "forms": {"E": "OFF", "R": "NRB", "X": "OFF"}}
EV = RATES.EV_TO_ERG


def det_seedmem_scope_fx_deck(model_dir):
    """synthetic deck: Si III levels 0,3,4 independent, 1-2 one super-level; S III levels 0,1,2 independent;
    lines 0 (0-3, control), 1 (0-4), 2 (3-4 connected), 3 (1-4 counter), 4 (S III 0-1), 5 (S III 1-2), 6 decoy."""
    model_dir.mkdir(parents=True)
    lv = [(14, 2, 0, 0.0, 1, 0), (14, 2, 1, 6.5, 3, 1), (14, 2, 2, 6.55, 5, 1), (14, 2, 3, 10.0, 9, 2),
          (14, 2, 4, 12.5, 15, 3), (14, 2, 5, 12.8, 5, 4), (16, 2, 0, 0.0, 1, 0), (16, 2, 1, 8.0, 3, 1),
          (16, 2, 2, 11.0, 5, 2),
          (14, 1, 0, 0.0, 2, 0), (14, 1, 1, 9.0, 4, 1)]
    with open(model_dir / "levels.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level,configuration\n")
        for Z, q, n, E, g, sl in lv:
            f.write(f"{Z},{q},{n},{E!r},{g},0,{sl},c{n}\n")
    E = {(Z, q, n): (En, g) for Z, q, n, En, g, _ in lv}
    lines = [(14, 2, 0, 3, 1.7), (14, 2, 0, 4, 0.3), (14, 2, 3, 4, 0.2), (14, 2, 1, 4, 0.5), (16, 2, 0, 1, 0.9),
             (16, 2, 1, 2, 0.4), (14, 1, 0, 1, 0.6), (14, 2, 0, 5, 0.1), (14, 2, 4, 5, 0.05)]
    with open(model_dir / "line_list.csv", "w") as f:
        f.write("atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,"
                "B_ul,A_ul,wavelength_cm\n")
        for i, (Z, q, lo, upl, fl) in enumerate(lines):
            El, gl = E[(Z, q, lo)]
            Eu, gu = E[(Z, q, upl)]
            nu = (Eu - El) * EV / H_PLANCK
            lam = C_LIGHT / nu
            B_lu = 0.02654 * 1.01 * fl * lam * nu / C_LIGHT * 4 * math.pi / (H_PLANCK * nu)
            B_ul = B_lu * gl / gu
            A = 2 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 * B_ul
            f.write(f"{Z},{q},{lo},{upl},{i},{lam * 1e8!r},{-fl * gl / gu!r},{fl!r},{nu!r},{B_lu!r},{B_ul!r},{A!r},"
                    f"{lam!r}\n")
    (model_dir / "siii_col_cmfgen.csv").write_text("lower,upper,omega\n")      # present, never consumed
    sh = []
    for fn in ("line_list.csv", "levels.csv"):
        sh.append(f"{sha(model_dir / fn)}  {fn}")
    (model_dir.parent / "deck.sha256").write_text("\n".join(sh) + "\n")
    return lv


def det_seedmem_scope_fx_run(root, tag, lib, seed_scale, ne_scale, nrb, ion_series, csob, mutate=None):
    """one fixture run: W5 calls 1-7 generated forward by the production relations from populations that obey
    the run's two-level model for lines 0, 1 (Si III) and 4 (S III); line 3 (counter) and the connected lines take
    populations set directly, so their two-level model does not hold."""
    run = root / tag
    (run / "input").mkdir(parents=True)
    lv = det_seedmem_scope_fx_deck(run / "input/model")
    (run / "work/physics_comparison").mkdir(parents=True)
    (run / "work/det_obs_capture").mkdir(parents=True)
    deck = det_seedmem_scope_deck(run / "input/model", (14, 16), 2)
    at = deck["at"]
    ex = ['declare -x LUMINA_NLTE_COLL_FIX="1"', 'declare -x LUMINA_NLTE_COLL_FLOOR="0"',
          'declare -x LUMINA_SUPER_LEVELS="1"', f'declare -x LUMINA_SUPER_CUTOFF="{FX_CUTOFF}"',
          'declare -x LUMINA_INNER_BB_SCALE="1.0"']
    lumped = sum(1 for t in lv if t[2] >= FX_CUTOFF)
    (run / "stdout.log").write_text(
        f"  Levels: {len(lv)} loaded (super_level column present, super-levels active)\n"
        f"  [ARTIS super-cutoff] K={FX_CUTOFF}: {lumped} levels lumped -> per-ion explicit + 1 super-level\n"
        f"    Z=14 ion=2: {sum(1 for t in lv if t[:2] == (14, 2))} levels\n"
        f"    Z=16 ion=2: {sum(1 for t in lv if t[:2] == (16, 2))} levels\n"
        f"  [NLTE] Super-levels: ACTIVE ({len(lv)} FL -> {len(lv) - 1} SL across ions)\n")
    if nrb:
        ex.append('declare -x LUMINA_BB_NRB="1"')
    (run / "input/resolved_lumina.exports").write_text("\n".join(ex) + "\n")
    Te = np.array([1.0e5 + 100.0 * s for s in range(FX_NS)])
    ne = {g: np.array([5.0e13 * ne_scale * (1.0 + 0.05 * g) * (1.0 + 0.001 * s) for s in range(FX_NS)])
          for g in range(2, 9)}
    pops = {}                       # (Z, level, shell, g) -> population
    rows = {}                       # (line, shell, call) -> dict
    model_lines = (0, 1, 4)
    for s in FX_SHELLS:
        for g in range(2, 8):
            for Z, n0 in ((14, 1.0e3), (16, 5.0e2)):
                pops[(Z, 0, s, g)] = n0 * (1.0 + 0.002 * s)
            pops[(14, 1, s, g)] = 2.0e3 * (1.0 + 0.05 * g)
            pops[(14, 2, s, g)] = 9.0e2 * (1.0 + 0.04 * g)
            pops[(16, 2, s, g)] = 5.0e1 * (1.0 + 0.03 * g)
            pops[(14, 5, s, g)] = 3.0e2 * (1.0 + 0.02 * g)
            for line in model_lines:
                a = at[line]
                if g == 2:
                    J = (0.04 if line != 4 else 0.03) * seed_scale
                    rad = (I.x(a["B_lu"] * J), I.x(a["B_ul"] * J), I.x(a["A_ul"]))
                else:
                    r = rows[(line, s, g - 1)]
                    if nrb:
                        rc, b2, Ra, Rs, Rp = lib.bracket(r["beta"] * r["J_c"], r["L_l"], r["tau_eff"], r["Jbar"], 0,
                                                         a["B_lu"], a["B_ul"], a["A_ul"])
                        rad = (I.x(Ra), I.x(Rs), I.x(Rp))
                    else:
                        rad = (I.x(a["B_lu"] * r["Jbar"]), I.x(a["B_ul"] * r["Jbar"]), I.x(a["A_ul"]))
                Cu, Cd = RATES.det_siiii_rates_coll(float(ne[g][s]), float(Te[s]), a["E_l"], a["E_u"], a["g_l"],
                                                    a["g_u"], a["f_lu"])
                x = (rad[0].mid + Cu.mid) / (rad[1].mid + rad[2].mid + Cd.mid)
                pops[(a["Z"], a["up"], s, g)] = x * pops[(a["Z"], a["lo"], s, g)]
            for line, a in at.items():
                if a["Z"] not in (14, 16) or not (RATES.NU_MIN <= a["nu"] <= RATES.NU_MAX):
                    continue
                n_l, n_u = pops[(a["Z"], a["lo"], s, g)], pops[(a["Z"], a["up"], s, g)]
                tau = csob * a["f_lu"] * a["lam_cm"] * FX_TEXP * (n_l - (a["g_l"] / a["g_u"]) * n_u)
                if not tau > 0.0:
                    raise ValueError(f"FIXTURE_TAU_NOT_POSITIVE line={line} shell={s} g={g} tau={tau}")
                rc, beta, q = lib.exponx(tau)
                eta = n_u * a["A_ul"] * H_PLANCK * a["nu"] / FOUR_PI
                L = eta * (C_LIGHT * FX_TEXP / a["nu"]) * q
                Jc = 0.05 * (1.0 + 0.05 * g) * (1.0 + 0.1 * line) * (1.0 + 0.01 * s)
                rows[(line, s, g)] = {"J_c": Jc, "beta": beta, "L_l": L, "Jbar": beta * Jc + L, "tau_eff": tau}
    with open(run / "work/w5_lines.csv", "w") as f:
        f.write("call,operator,rad_generation,material_generation,shell,line_id,Z,ion,nu_hz,J_c,J_c_err_upper,beta,"
                "L_l,Jbar,Jbar_err_upper,tau_eff,srce_chk,status\n")
        for call in range(1, 8):
            for line, a in sorted(at.items()):
                for s in FX_SHELLS:
                    if not (RATES.NU_MIN <= a["nu"] <= RATES.NU_MAX):
                        op = "INIT_SHARED_GAUSSIAN,1,0" if call == 1 else f"CMFGEN_NONOVERLAP_SOBOLEV,{call},{call}"
                        f.write(f"{call},{op},{s},{line},{a['Z']},2,{a['nu']!r},,,,,,,,,OUTSIDE_FINE_SUPPORT\n")
                        continue
                    if call == 1:
                        J1 = (0.04 if line != 4 else 0.03) * seed_scale
                        f.write(f"1,INIT_SHARED_GAUSSIAN,1,0,{s},{line},{a['Z']},2,{a['nu']!r},NOT_APPLICABLE,"
                                f"NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,{J1!r},0,NOT_APPLICABLE,NOT_APPLICABLE,"
                                f"OK_INIT_GAUSSIAN_PROFILE_MEAN\n")
                        continue
                    r = rows[(line, s, call)]
                    f.write(f"{call},CMFGEN_NONOVERLAP_SOBOLEV,{call},{call},{s},{line},{a['Z']},2,{a['nu']!r},"
                            f"{r['J_c']!r},0,{r['beta']!r},{r['L_l']!r},{r['Jbar']!r},0,{r['tau_eff']!r},0,OK\n")
    for k in range(6):
        g = k + 3
        (run / f"work/physics_comparison/physics_DET_iter{k:04d}.manifest.json").write_text(
            json.dumps({"iteration": k, "population_generation": g, "radiation_generation": g - 1}))
        with open(run / f"work/physics_comparison/physics_DET_iter{k:04d}.shell.csv", "w") as f:
            f.write("shell_id,T_e_K,n_e_cm3\n" + "".join(f"{s},{float(Te[s])!r},{float(ne[g][s])!r}\n"
                                                         for s in range(FX_NS)))
        with open(run / f"work/physics_comparison/physics_DET_iter{k:04d}.ion.csv", "w") as f:
            f.write("population_generation,shell_id," + ",".join(f"Z{Z}_nlte_q{q}_cm3" for Z, q in ION_QTY) + "\n")
            for s in range(FX_NS):
                f.write(f"{g},{s}," + ",".join(f"{float(ion_series[(Z, q)][g])!r}" for Z, q in ION_QTY) + "\n")
    capd = run / "work/det_obs_capture"
    recs = []
    for line, a in sorted(at.items()):
        if not (RATES.NU_MIN <= a["nu"] <= RATES.NU_MAX):
            continue
        for s in FX_SHELLS:
            n_l, n_u = pops[(a["Z"], a["lo"], s, 7)], pops[(a["Z"], a["up"], s, 7)]
            tau, S, *_ = POP.prod_tau_S(n_l, n_u, a["g_l"], a["g_u"], a["f_lu"], a["lam_cm"], FX_TEXP)
            recs.append((line, s, tau, S, 0.0, 1))
    np.array(sorted(recs), dtype=POP.REC).tofile(capd / "lines.bin")
    ne[7].astype("<f8").tofile(capd / "ne.f64")
    (run / "fixture_truth.json").write_text(json.dumps({f"{k[0]}:{k[1]}:{k[2]}:{k[3]}": v for k, v in pops.items()}))
    (capd / "meta.txt").write_text(f"time_explosion={FX_TEXP.hex()}\nT_inner={(10020.0).hex()}\n"
                                   f"inner_bb_scale={(1.0).hex()}\nmaterial_generation=7\nrad_generation=7\n")
    (capd / "manifest.json").write_text(json.dumps({"generations": {"material": 7, "nlte_population_committed": 7,
                                                                   "radiation": 7}, "transaction": "COMMITTED",
                                                   "census": {"mapped_B_fallback": 0, "consumed": len(recs),
                                                              "mapped": len(recs)},
                                                   "files": {"lines.bin": {"sha256": sha(capd / "lines.bin")}}}))
    if mutate:
        mutate(run)
    return run


def det_seedmem_scope_fx(base, lib, csob, mutate=None, census_mutate=None, ion_series=None):
    base = Path(base)
    base.mkdir(parents=True)
    if ion_series is None:
        ion_series = {(Z, q): {g: (1.0e6 * (1 + Z / 100 + q) * (1.0 if g >= 5 else 1.0 + 0.2 * (5 - g)))
                               for g in POP_GENS} for Z, q in ION_QTY}
    runs = {}
    for tag, sc, nesc, nrb in (("E", 1.0, 1.0, False), ("R", 1.0, 1.0, True), ("X", 1.3, 1.0, False)):
        runs[tag] = det_seedmem_scope_fx_run(base, tag, lib, sc, nesc, nrb, ion_series, csob,
                                             (lambda r, t=tag: mutate(r, t)) if mutate else None)
    rows = []
    for line in (0, 1, 3, 4):
        Z = 16 if line == 4 else 14
        for s in FX_SHELLS:
            rows.append({"line_id": line, "shell": s, "Z": Z, "ion": 2, "beta2_E": 1e-6, "L2": 1, "D2": 1})
    rows.append({"line_id": 2, "shell": 10, "Z": 14, "ion": 2, "beta2_E": 5e-3, "L2": 0, "D2": 0})
    rows.append({"line_id": 6, "shell": 10, "Z": 14, "ion": 1, "beta2_E": 1e-6, "L2": 0, "D2": 0})
    cen = pd.DataFrame(rows)
    if census_mutate:
        cen = census_mutate(cen)
    cen.to_csv(base / "census.csv", index=False)
    return {k: str(v) for k, v in runs.items()}, base / "census.csv"


def det_seedmem_scope_hp_tests(lib, csob):
    """prereg 6.2 direct tests against an independent high-precision reference (decimal, 60 digits):
    the production-order inversion encloses the stored double populations in every EXPONX regime and near the
    tau cancellation; the log10 change measure and the d interval enclose the high-precision value; the EXPONX
    double evaluation stays within its rounding bound of the same formula evaluated in high precision."""
    import decimal
    from decimal import Decimal as Dm
    ctx = decimal.Context(prec=60)
    out = []
    t_exp = FX_TEXP
    lam, f, gl, gu = 1.2065e-5, 1.67, 1.0, 3.0
    nu = C_LIGHT / lam
    A = 2.678248e9
    for name, n_l, n_u in (("thick", 4.79e4, 5.3765e-4), ("moderate", 3.1e-2, 4.0e-6), ("polynomial", 2.0e-9, 1.0e-13),
                           ("beyond_40_cancel", 1.0e6, 2.99985e6)):
        v, tau = lib.tau(csob, f, lam, t_exp, n_l, n_u, gl, gu)
        jc = 3.36e-8
        rc, et, beta, L, jb, jbe = lib.material_radiation(n_u, A, nu, tau, t_exp, jc, 0.0)
        row = {"tau_eff": tau, "L_l": L, "Jbar": jb, "J_c": jc, "J_c_err_upper": 0.0, "Jbar_err_upper": jbe,
               "beta": beta, "srce_chk": 0.0}
        a = {"A_ul": A, "f_lu": f, "nu": nu, "lam_cm": lam, "g_l": gl, "g_u": gu, "B_lu": 1.0}
        inv, why = det_seedmem_scope_invert(row, a, t_exp, csob, lib)
        # (i) production tau of the double populations against the 60-digit value; bound: 4 products + the rounded
        # a n_u inside the difference, amplified by a n_u / |n_l - a n_u|
        dhp = ctx.subtract(Dm(n_l), ctx.multiply(ctx.divide(Dm(gl), Dm(gu)), Dm(n_u)))
        K_hp = ctx.multiply(ctx.multiply(ctx.multiply(Dm(csob), Dm(f)), Dm(lam)), Dm(t_exp))
        tau_hp = ctx.multiply(K_hp, dhp)
        tau_rel = float(abs(ctx.divide(ctx.subtract(Dm(tau), tau_hp), tau_hp))) if tau_hp != 0 else None
        amp = float(abs(ctx.divide(ctx.multiply(ctx.divide(Dm(gl), Dm(gu)), Dm(n_u)), dhp))) if dhp != 0 else math.inf
        tau_ok = tau_rel is not None and tau_rel <= (6.0 + 2.0 * amp) * U
        # (ii) independent 60-digit inversion of the stored ledger (tau, L and production's EXPONX companion) with the
        # production double constants, compared with the readout's interval inversion
        ok_hp = False
        nu_hp = nl_hp = None
        if inv is not None:
            rcq, bq, q = lib.exponx(tau)
            scale = ctx.divide(ctx.multiply(Dm(C_LIGHT), Dm(t_exp)), Dm(nu))
            nu_hp = ctx.multiply(ctx.divide(Dm(L), ctx.multiply(scale, Dm(q))),
                                 ctx.divide(Dm(FOUR_PI), ctx.multiply(ctx.multiply(Dm(A), Dm(H_PLANCK)), Dm(nu))))
            nl_hp = ctx.add(ctx.divide(Dm(tau), K_hp), ctx.multiply(ctx.divide(Dm(gl), Dm(gu)), nu_hp))
            ok_hp = (Dm(inv["n_u"].lo) <= nu_hp <= Dm(inv["n_u"].hi)) and (Dm(inv["n_l"].lo) <= nl_hp <= Dm(inv["n_l"].hi))
        # (iii) round trip: the intervals contain the double populations that generated the ledger
        ok_rt = (inv is not None and inv["n_u"].lo <= n_u <= inv["n_u"].hi and inv["n_l"].lo <= n_l <= inv["n_l"].hi)
        out.append({"test": f"inversion_{name}", "tau": tau, "why": why,
                    "pass": bool(inv is not None and tau_ok and ok_hp and ok_rt),
                    "tau_rel_err_vs_hp": tau_rel, "tau_bound": (6.0 + 2.0 * amp) * U,
                    "hp_inverse_in_interval": bool(ok_hp), "round_trip_in_interval": bool(ok_rt),
                    "n_u_hp": None if nu_hp is None else f"{nu_hp:.25E}",
                    "n_l_hp": None if nl_hp is None else f"{nl_hp:.25E}",
                    "n_u_interval": None if inv is None else [inv["n_u"].lo, inv["n_u"].hi],
                    "n_l_interval": None if inv is None else [inv["n_l"].lo, inv["n_l"].hi]})
    for ta in (1e-9, 5e-4, 9.99e-4, 1.5e-3, 0.37, 7.0, 39.9, 40.0, 1.0e4):
        rc, b, c = lib.exponx(ta)
        T = Dm(ta)
        if abs(ta) < 1e-3:
            comp = ctx.subtract(Dm("0.5"), ctx.multiply(ctx.divide(T, Dm(6)), ctx.subtract(Dm(1), ctx.divide(T, Dm(4)))))
            val = ctx.subtract(Dm(1), ctx.multiply(T, comp))
            bound_v, bound_c = 8 * U, 8 * U
        elif ta < 40.0:
            val = ctx.divide(ctx.subtract(Dm(1), ctx.exp(-T)), T)
            comp = ctx.divide(ctx.subtract(Dm(1), val), T)
            bound_v = 8 * U * (1 + 1 / ta)
            bound_c = 8 * U * (1 + 1 / ta) * (1 + 2 / ta)
        else:
            val = ctx.divide(Dm(1), T)
            comp = ctx.divide(ctx.subtract(Dm(1), val), T)
            bound_v, bound_c = 4 * U, 8 * U
        ev = float(abs(ctx.divide(ctx.subtract(Dm(b), val), val)))
        ec = float(abs(ctx.divide(ctx.subtract(Dm(c), comp), comp)))
        out.append({"test": f"exponx_{ta!r}", "pass": rc == 0 and ev <= bound_v and ec <= bound_c,
                    "rel_err_beta": ev, "rel_err_companion": ec, "bound_beta": bound_v, "bound_companion": bound_c})
    ln10 = ctx.ln(Dm(10))
    for a_, b_ in ((1.0, 1.0 + 2 ** -50), (3.3634e-8, 3.1330e-8), (1e-150, 1e150), (2.0, 2.0), (7.5e5, 7.49e5)):
        m = det_seedmem_scope_m(a_, b_)
        hp = abs(ctx.divide(ctx.ln(ctx.divide(Dm(b_), Dm(a_))), ln10))
        out.append({"test": f"log10_{a_!r}_{b_!r}", "pass": Dm(m.lo) <= hp <= Dm(m.hi), "hp": float(hp),
                    "interval": [m.lo, m.hi]})
    for sm, sp in ((3.1329697558645786e-08, 3.368739245333992e-08), (1.0, 1.0 + 1e-3), (2.5e-15, 2.5e-15)):
        dv = (I.rel(sm, 2).div(I.rel(sp, 2))) - I.x(1.0)
        hp = ctx.subtract(ctx.divide(Dm(sm), Dm(sp)), Dm(1))
        out.append({"test": f"d_{sm!r}_{sp!r}", "pass": Dm(dv.lo) <= hp <= Dm(dv.hi), "hp": float(hp),
                    "interval": [dv.lo, dv.hi]})
    return out


def det_seedmem_scope_selftest_cases(base, lib, src):
    csob = det_seedmem_scope_csob(src)
    res = []

    def case(name, bundle, got, want, fault=None, results=()):
        ok = got == want
        rec = {"case": name, "bundle": bundle, "fault": fault, "pass": bool(ok), "got": str(got)[:600],
               "want": str(want)[:600],
               "runs": [{"validity": r.get("validity"),
                         "analyze_rc_equivalent": 0 if r.get("validity") == "VALID" else 2,
                         "reasons": [str(x)[:160] for x in (r.get("reasons") or [])[:6]]} for r in results]}
        res.append(rec)
        print(f"[{'PASS' if ok else 'FAIL'}] ({bundle}) {name}: got {str(got)[:220]} want {str(want)[:220]}", flush=True)

    def run(name, fault=None, mutate=None, census_mutate=None, ion_series=None, target=FX_TARGET):
        runs, census = det_seedmem_scope_fx(Path(base) / name / "in", lib, csob, mutate, census_mutate, ion_series)
        return det_seedmem_scope_analyze(census, runs, src, Path(base) / name / "out", lib, target, fault), \
            Path(base) / name / "out"

    def has(r, token):
        return r.get("validity") == "X" and any(token in str(w) for w in (r.get("reasons") or []))

    def nxt(o):
        return json.loads((o / "next_contract.json").read_text())

    def dec(o, prefix):
        return [d_ for d_ in nxt(o)["decisions"] if d_["result"].startswith(prefix)]

    def xrows(o, fname, token):
        p_ = o / fname
        if not p_.exists():
            return False
        d_ = pd.read_csv(p_)
        col = "reasons" if "reasons" in d_.columns else "link"
        return bool(((d_.status == "X") & d_[col].fillna("").str.contains(token, regex=False)).any())

    def w5_edit(r, pred, edit):
        p_ = r / "work/w5_lines.csv"
        out_ = []
        for ln in p_.read_text().split("\n"):
            f_ = ln.split(",")
            if len(f_) > 13 and pred(f_):
                f_ = edit(f_)
                if f_ is None:
                    continue
            out_.append(",".join(f_))
        p_.write_text("\n".join(out_))

    r0, o0 = run("clean")
    m0 = pd.read_csv(o0 / "model.csv")
    ctrl = m0[(m0.run == "R") & (m0.line == 0)]
    case("clean fixture: valid; R' control survives at calls 3-7; counter refuted; S III model line survives",
         "0 clean", (r0["validity"], r0["codes"]["H-MODEL"]["R"]["control_1206"], r0["codes"]["H-COUNTEREXAMPLE"][
             "R_two_level_nrb_applicability"], r0["codes"]["H-MODEL"]["R"]["SIII"], sorted(ctrl.call.unique().tolist())),
         ("VALID", 0, 1, 0, [3, 4, 5, 6, 7]), results=[r0])
    case("clean fixture: off model survives for E' and X control; NRB rates invariant (H-LOCAL 0); every admissible "
         "model row passes the source-population identity",
         "0 clean", (r0["codes"]["H-MODEL"]["E"]["control_1206"], r0["codes"]["H-MODEL"]["X"]["control_1206"],
                     r0["codes"]["H-LOCAL"], sorted(set(m0.source_population_identity.dropna()))),
         (0, 0, 0, ["OK"]))
    sj = json.loads((o0 / "sample.json").read_text())
    case("solve units: LUMINA_SUPER_CUTOFF overrides the deck column (levels 1,2 explicit, 4,5 one unit) and is "
         "confirmed by stdout", "13 solve units",
         (sj["solve_unit_sizes_of_sample_by_run"]["R"]["3"], sj["solve_unit_sizes_of_sample_by_run"]["R"]["1"],
          sj["solve_unit_authority"]["R"]["cutoff"], sj["solve_unit_authority"]["R"]["active"]),
         ([1, 2], [1, 2], FX_CUTOFF, True))
    rsl1, _ = run("deck_sl_column", fault="deck_sl_column")

    def no_cutoff_line(r, t):
        p_ = r / "stdout.log"
        p_.write_text("\n".join(ln for ln in p_.read_text().split("\n") if "super-cutoff" not in ln))
    rsl2, _ = run("sl_unconfirmed", mutate=no_cutoff_line)
    case("solve units: reading the deck column under a cutoff, or a cutoff the stdout does not confirm, is "
         "inadmissible", "13 solve units",
         (has(rsl1, "SL_AUTHORITY_NOT_REGISTERED"), has(rsl2, "SL_CUTOFF_NOT_CONFIRMED")), (True, True),
         fault="deck_sl_column", results=[rsl1, rsl2])
    # 1 sample fixing
    rs1, _ = run("drop_counter", fault="drop_counter")
    rs2, _ = run("add_shell", fault="add_shell")
    rs3, os3 = run("drop_failed_cell", fault="drop_failed_cell")
    case("sample: counterexample deletion, a non-W5 shell and dropping refuted cells after the fact are "
         "inadmissible (G-LINK); the final decision table then carries no model decision",
         "1 sample",
         (has(rs1, "COUNTER"), has(rs2, "SHELL") or has(rs2, "COUNT"), has(rs3, "MODEL_ROW_ACCOUNTING"),
          sorted({d_["code"] for d_ in dec(os3, "1206 control") + dec(os3, "SiIII_other two") + dec(os3, "SIII two")}),
          nxt(os3)["provisional"]),
         (True, True, True, ["X"], True), fault="drop_counter|add_shell|drop_failed_cell", results=[rs1, rs2, rs3])
    # 2 generation linking
    rg1, og1 = run("beta_current", fault="beta_current")
    rg2, og2 = run("material_plus1", fault="material_plus1")

    def dump_relabel(r, t):
        p_ = r / "work/physics_comparison/physics_DET_iter0004.manifest.json"
        js = json.loads(p_.read_text())
        js["population_generation"] = 8
        p_.write_text(json.dumps(js))
    rg3, _ = run("dump_relabel", mutate=dump_relabel)
    case("generations: current-call radiation and generation k+1 material are inadmissible rows (X, no science "
         "code); a relabelled dump is X", "2 generations",
         (rg1["validity"], xrows(og1, "model.csv", "W5_GENERATION_MISMATCH"), rg2["validity"],
          xrows(og2, "model.csv", "MATERIAL_GENERATION_MISMATCH"),
          rg2["codes"]["H-MODEL"]["R"]["control_1206"], rg3["validity"]),
         ("X", True, "X", True, "X", "X"), fault="beta_current|material_plus1|dump_relabel", results=[rg1, rg2, rg3])
    # 3 missing data
    idn = pd.read_csv(o0 / "identifiability.csv")
    g2 = idn[(idn.question == "collisional_response_n_e") & (idn.generation == 2)]
    rm, om = run("drop_row", mutate=lambda r, t: w5_edit(
        r, lambda f_: f_[0] == "5" and f_[4] == "15" and f_[5] == "1", lambda f_: None))
    mm = pd.read_csv(om / "model.csv")
    miss = mm[(mm.line == 1) & (mm.shell == 15) & (mm.call.isin([5, 6]))]
    rf1, _ = run("fill_gen2_ne", fault="fill_gen2_ne")

    def zero_fill(f_):
        for k_ in (9, 10, 11, 12, 13, 14, 15, 16):
            f_[k_] = "0"
        return f_
    rf2, _ = run("zero_fill", mutate=lambda r, t: w5_edit(
        r, lambda f_: f_[0] == "5" and f_[4] == "15" and f_[5] == "1", zero_fill))
    case("missing: generation-2 n_e is 5; a removed W5 row gives 5 (not 0) and the counterexample stays 1; filling "
         "generation 2 from generation 3 or a zero-filled row is inadmissible", "3 missing",
         (set(g2.status), set(miss.status), rm["validity"], rm["codes"]["H-COUNTEREXAMPLE"]["R_two_level_nrb_applicability"],
          has(rf1, "DUMP_GENERATION_FILLED"), has(rf2, "SAMPLE_ROW_LINK")),
         ({"NOT_IDENTIFIABLE"}, {"NOT_IDENTIFIABLE"}, "VALID", 1, True, True), fault="fill_gen2_ne|zero_fill",
         results=[rm, rf1, rf2])
    # 4 W5 inversion (+ high-precision references)
    cc = pd.read_csv(o0 / "capture_check.csv")
    rw1, _ = run("omit_r", fault="omit_r")
    rw2, _ = run("wrong_A", fault="wrong_A")
    rw3, _ = run("swap_levels", fault="swap_levels")
    case("inversion: W5 inversion encloses the generation-7 populations (capture compatible); omitting r_l breaks the "
         "source-population identity; wrong A or swapped levels fail the forward production ledger",
         "4 inversion", (set(cc.compatible.dropna().astype(int)), has(rw1, "X_ROWS"),
                         has(rw2, "FORWARD_LEDGER_MISMATCH"), has(rw3, "FORWARD_LEDGER_MISMATCH")),
         ({1}, True, True, True), fault="omit_r|wrong_A|swap_levels", results=[rw1, rw2, rw3])
    hp = det_seedmem_scope_hp_tests(lib, csob)
    case("high-precision references (decimal 60 digits): inversion encloses the stored populations in every EXPONX "
         "regime; EXPONX, log10 change and d intervals within / enclosing the reference",
         "4 inversion", [t["test"] for t in hp if not t["pass"]], [])
    res[-1]["hp_tests"] = hp
    # 5 independent authority
    def cap_gen(r, t):
        p_ = r / "work/det_obs_capture/manifest.json"
        js = json.loads(p_.read_text())
        js["generations"]["material"] = 8
        p_.write_text(json.dumps(js))

    def cap_source(r, t):
        p_ = r / "work/det_obs_capture/lines.bin"
        m_ = np.fromfile(p_, dtype=POP.REC)
        m_["S"] = m_["S"] * 1.5
        m_.tofile(p_)

    def cap_mapped(r, t):
        p_ = r / "work/det_obs_capture/lines.bin"
        m_ = np.fromfile(p_, dtype=POP.REC)
        m_["m"][(m_["l"] == 0) & (m_["s"] == 10)] = 0
        m_.tofile(p_)

    def cap_unmapped_recorded(r, t):
        cap_mapped(r, t)
        capd = r / "work/det_obs_capture"
        js = json.loads((capd / "manifest.json").read_text())
        m_ = np.fromfile(capd / "lines.bin", dtype=POP.REC)
        js["census"]["mapped"] = int((m_["m"] == 1).sum())
        js["files"]["lines.bin"]["sha256"] = sha(capd / "lines.bin")
        (capd / "manifest.json").write_text(json.dumps(js))
    ra1, _ = run("cap_gen", mutate=cap_gen)
    ra2, _ = run("cap_source", mutate=cap_source)
    ra3, _ = run("cap_mapped", mutate=cap_mapped)
    ra4, oa4 = run("cap_unmapped_recorded", mutate=cap_unmapped_recorded)
    pa4 = pd.read_csv(oa4 / "pathways.csv")
    ia4 = pd.read_csv(oa4 / "identifiability.csv")
    case("independent authority: a capture generation change, a replaced capture source and a flipped mapped flag are "
         "inadmissible; a recorded unmapped line propagates its absence to the generation-7 budget and identifiability",
         "5 authority", (ra1["validity"], ra2["validity"], has(ra3, "CAPTURE_FILE_SHA_MISMATCH"), ra4["validity"],
                         set(pa4[(pa4.line == 0) & (pa4.shell == 10)].capture_authority),
                         set(ia4[(ia4.question == "generation7_population_authority") & (ia4.line == 0) &
                                 (ia4.shell == 10)].status)),
         ("X", "X", True, "VALID", {"UNMAPPED"}, {"NOT_IDENTIFIABLE"}), fault="capture mutations",
         results=[ra1, ra2, ra3, ra4])
    # 6 production NRB link
    rn1, _ = run("omit_beta", fault="omit_beta")
    rn2, _ = run("omit_stim", fault="omit_stim")

    def ledger_break(r, t):
        if t != "R":
            return
        w5_edit(r, lambda f_: f_[0] == "4" and f_[5] == "0" and f_[4] == "10",
                lambda f_: f_[:13] + [repr(float(f_[13]) * (1 + 1e-9))] + f_[14:])
    rn3, _ = run("ledger_break", mutate=ledger_break)
    rn4, on4 = run("ledger_break_unchecked", mutate=ledger_break, fault="skip_row_check")
    case("NRB link: omitted beta or stimulated term fail the rate reconstruction (X); a ledger mismatch is refused by "
         "the row check and, when the check is bypassed, by the production function itself (rc -3, X)",
         "6 NRB link", (has(rn1, "X_ROWS"), has(rn2, "X_ROWS"), has(rn3, "SAMPLE_ROW_LINK"),
                        xrows(on4, "model.csv", "BRACKET_RC_-3")),
         (True, True, True, True), fault="omit_beta|omit_stim|skip_row_check", results=[rn1, rn2, rn3, rn4])
    # 7 cross substitution
    cx = pd.read_csv(o0 / "cross_substitution.csv")
    rc1, _ = run("move_c", fault="move_c_with_L")
    case("cross substitution: fixed c, tau and L exchange keep the production NRB rates bit-identical (stored per "
         "combination) and the ledger closes; moving c with L labelled as fixed background is inadmissible",
         "7 cross", (set(cx.nrb_rates_invariant_under_L.dropna().astype(int)), float(cx.nrb_D_L.abs().max()) == 0.0,
                     bool((cx.LEzE_nrb_R_spont == cx.LXzE_nrb_R_spont).all()),
                     bool((cx.LEzE_off_R_absorb != cx.LXzE_off_R_absorb).any()),
                     bool((cx.off_ledger_residual.abs() <= 1e-12 * cx.off_H_XX.abs()).all()), rc1["validity"]),
         ({1}, True, True, True, True, "X"), fault="move_c_with_L", results=[rc1])
    # 8 multilevel budget
    pb = pd.read_csv(o0 / "pathways.csv")
    truth = json.loads((Path(base) / "clean/in/E/fixture_truth.json").read_text())
    row = pb[(pb.run == "E") & (pb.line == 0) & (pb.shell == 10)].iloc[0]
    fxdeck = det_seedmem_scope_deck(Path(base) / "clean/in/E/input/model", (14, 16), 2)
    a0 = fxdeck["at"][0]
    w6 = REP.det_siiii_repair_w5_load(Path(base) / "clean/in/E", {(0, 10)})[0][(0, 10)][6]
    ne7 = float(pd.read_csv(Path(base) / "clean/in/E/work/physics_comparison/physics_DET_iter0004.shell.csv").set_index(
        "shell_id").loc[10, "n_e_cm3"])
    te7 = float(pd.read_csv(Path(base) / "clean/in/E/work/physics_comparison/physics_DET_iter0004.shell.csv").set_index(
        "shell_id").loc[10, "T_e_K"])
    Cu_, Cd_ = RATES.det_siiii_rates_coll(ne7, te7, a0["E_l"], a0["E_u"], a0["g_l"], a0["g_u"], a0["f_lu"])
    nl_t, nu_t = truth[f"14:{a0['lo']}:10:7"], truth[f"14:{a0['up']}:10:7"]
    jb6 = float(w6["Jbar"])
    t1, t2 = nl_t * (a0["B_lu"] * jb6 + Cu_.mid), nu_t * (a0["B_ul"] * jb6 + a0["A_ul"] + Cd_.mid)
    F_true = t1 - t2
    slack = 64 * math.ulp(max(abs(t1), abs(t2)))
    flow_ok = []
    for tag_ in ("E", "R"):
        rdir = Path(base) / f"clean/in/{tag_}"
        tr_ = json.loads((rdir / "fixture_truth.json").read_text())
        at_ = fxdeck["at"]
        sh7 = pd.read_csv(rdir / "work/physics_comparison/physics_DET_iter0004.shell.csv").set_index("shell_id")
        hist_ = REP.det_siiii_repair_w5_load(rdir, {(l_, 10) for l_ in at_})[0]
        for tl in (0, 1):
            t_ = at_[tl]
            others = [l_ for l_, b_ in at_.items() if l_ != tl and b_["Z"] == t_["Z"] and
                      {b_["lo"], b_["up"]} & {t_["lo"], t_["up"]}]
            up_true, lo_true, mag = 0.0, 0.0, 0.0
            for l_ in others:
                b_ = at_[l_]
                n_lo, n_up = tr_[f"{b_['Z']}:{b_['lo']}:10:7"], tr_[f"{b_['Z']}:{b_['up']}:10:7"]
                cu_, cd_ = RATES.det_siiii_rates_coll(float(sh7.loc[10, "n_e_cm3"]), float(sh7.loc[10, "T_e_K"]),
                                                      b_["E_l"], b_["E_u"], b_["g_l"], b_["g_u"], b_["f_lu"])
                if not (RATES.NU_MIN <= b_["nu"] <= RATES.NU_MAX):
                    Ra_, Rs_, Rp_ = 0.0, 0.0, b_["A_ul"]
                else:
                    w_ = hist_[(l_, 10)][6]
                    if tag_ == "R":
                        _, _, Ra_, Rs_, Rp_ = lib.bracket(float(w_["beta"]) * float(w_["J_c"]), float(w_["L_l"]),
                                                          float(w_["tau_eff"]), float(w_["Jbar"]), 0,
                                                          b_["B_lu"], b_["B_ul"], b_["A_ul"])
                    else:
                        Ra_, Rs_, Rp_ = b_["B_lu"] * float(w_["Jbar"]), b_["B_ul"] * float(w_["Jbar"]), b_["A_ul"]
                p1, p2_ = n_lo * (Ra_ + cu_.mid), n_up * (Rs_ + Rp_ + cd_.mid)
                F_ = p1 - p2_
                mag += abs(p1) + abs(p2_)
                if b_["up"] == t_["up"]:
                    up_true += F_
                if b_["lo"] == t_["up"]:
                    up_true -= F_
                if b_["up"] == t_["lo"]:
                    lo_true += F_
                if b_["lo"] == t_["lo"]:
                    lo_true -= F_
            rw_ = pb[(pb.run == tag_) & (pb.line == tl) & (pb.shell == 10)].iloc[0]
            sl_ = 64 * math.ulp(mag)
            flow_ok.append((tag_, tl, len(others),
                            bool(rw_.F_other_into_upper_lo - sl_ <= up_true <= rw_.F_other_into_upper_hi + sl_),
                            bool(rw_.F_other_into_lower_lo - sl_ <= lo_true <= rw_.F_other_into_lower_hi + sl_)))
    rb1, ob1 = run("drop_connected", fault="drop_connected")
    rb2, _ = run("double_count", fault="double_count")
    rb3, _ = run("partial_as_complete", fault="partial_as_complete", mutate=lambda r, t: w5_edit(
        r, lambda f_: f_[0] == "7" and f_[4] == "10" and f_[5] == "2", lambda f_: None))
    case("budget: complete bound-bound sets (out-of-domain A-only lines from recovered levels) counted; the target "
         "net flow and the direction-signed inflow sums of the connected transitions into the upper and the lower "
         "level match the fixture truth (off and NRB, with an A-only line); closure stays 5 (bf absent); a dropped, a "
         "double-counted connected transition or a partial sum labelled complete is inadmissible and the budget "
         "decision becomes X", "8 budget",
         (int(pb.complete_bb.sum()) == len(pb), int(pb.n_a_only_outside_domain.sum()) > 0, set(pb.closed),
          r0["codes"]["H-MULTILEVEL"]["R"]["code"], bool(row.F_target_lo - slack <= F_true <= row.F_target_hi + slack),
          [f_[3:] for f_ in flow_ok], min(f_[2] for f_ in flow_ok) >= 2,
          rb1["validity"], rb2["validity"], has(rb3, "X_ROWS"),
          [d_["code"] for d_ in dec(ob1, "generation-7 bound-bound budgets")]),
         (True, True, {0}, 5, True, [(True, True)] * 4, True, "X", "X", True, ["X"]),
         fault="drop_connected|double_count|partial_as_complete", results=[rb1, rb2, rb3])
    res[-1]["flow_truth"] = [list(f_) for f_ in flow_ok]
    # 9 collision consumption
    def omega_branch(r, t):
        p_ = r / "input/resolved_lumina.exports"
        p_.write_text(p_.read_text() + 'declare -x LUMINA_OMEGA_CMFGEN="1"\n')
        q_ = r / "stdout.log"
        q_.write_text(q_.read_text() + "  [OMEGA-CMFGEN] col_data manifest loaded\n")
    rk, _ = run("omega_branch", mutate=omega_branch)
    rk2, _ = run("files_as_consumed", fault="files_as_consumed")
    rk3, _ = run("assume_coldata", fault="assume_coldata")
    ca = r0["collision_authority"]["R"]
    case("collisions: COLL_FIX per line, loader not called, matrix not consuming although a deck table is present; a "
         "consumed col_data branch (OMEGA_CMFGEN, loader marked) makes the model 5; reading file presence or an "
         "unloaded table as consumption is inadmissible", "9 collisions",
         (ca["branch"], ca["supported"], ca["loader_called"], ca["matrix_consumes_col_data"],
          bool(ca["deck_collision_files"]), rk["validity"], rk["codes"]["H-MODEL"]["R"]["control_1206"],
          has(rk2, "COLLISION_AUTHORITY_CONTRADICTION"), has(rk3, "COLLISION_AUTHORITY_CONTRADICTION"),
          r0["codes"]["H-COUNTEREXAMPLE"]["call3_detail_R"].get("S_meas_in_two_level_range") in
          ("INSIDE", "OUTSIDE", "NOT_SEPARATED")),
         ("COLL_FIX_PER_LINE", True, False, False, True, "VALID", 5, True, True, True),
         fault="files_as_consumed|assume_coldata", results=[rk, rk2, rk3])
    # 10 P2
    ion_one = {(Z, q): {g: (1.0e6 * (1.0 if g in (5, 6) else (1.3 if g == 7 else 1.0 + 0.2 * abs(5 - g))))
                        for g in POP_GENS} for Z, q in ION_QTY}
    rp1, _ = run("p2_single", ion_series=ion_one)
    rp1f, op1f = run("p2_single_fault", ion_series=ion_one, fault="single_change")
    ion_zero = {(Z, q): {g: (0.0 if (Z, q) == (14, 1) and g == 5 else 1.0e6) for g in POP_GENS} for Z, q in ION_QTY}
    rp2, _ = run("p2_zero", ion_series=ion_zero)
    rp2f, _ = run("p2_zero_fault", ion_series=ion_zero, fault="drop_zero_cross")
    rp3f, _ = run("p2_drop_large", fault="drop_large")
    ion_struct = {(Z, q): {g: (0.0 if (Z, q) == (14, 1) else 1.0e6) for g in POP_GENS} for Z, q in ION_QTY}
    rp4, op4 = run("p2_structural_zero", ion_series=ion_struct)
    st4 = pd.read_csv(op4 / "stationarity.csv")
    ion_nan = {(Z, q): {g: (math.nan if (Z, q) == (14, 1) and g in (5, 6, 7) else 1.0e6) for g in POP_GENS}
               for Z, q in ION_QTY}
    rp5, _ = run("p2_missing", ion_series=ion_nan)
    ion_nan_ref = {(Z, q): {g: (math.nan if (Z, q) == (14, 1) and g in (5, 6, 7) else
                                (2.0e6 if (Z, q) == (16, 1) and g >= 5 else 1.0e6)) for g in POP_GENS} for Z, q in ION_QTY}
    rp6, _ = run("p2_missing_with_refutation", ion_series=ion_nan_ref)
    case("P2: clean ion sums small from 5->6; one small change alone is not two; a zero crossing refutes; exact zeros "
         "at both ends are structural (counted, not missing); accepting one change, deleting a zero crossing or a "
         "large cell is inadmissible", "10 P2",
         (r0["codes"]["H-P2"]["E"]["ion"]["state"], rp1["codes"]["H-P2"]["E"]["ion"]["state"],
          has(rp1f, "P2_PAIR_RECOMPUTE"), rp2["codes"]["H-P2"]["E"]["ion"]["state"],
          has(rp2f, "P2_QUANTITY_ACCOUNTING"), has(rp3f, "P2_QUANTITY_ACCOUNTING"),
          rp4["validity"], int(st4[(st4.run == "E") & (st4.axis == "ion")].structural_zero.min()),
          rp4["codes"]["H-P2"]["E"]["ion"]["state"]),
         ("SURVIVES", "REFUTED", True, "REFUTED", True, True, "VALID", len(FX_SHELLS), "SURVIVES"),
         fault="single_change|drop_zero_cross|drop_large", results=[rp1f, rp2f, rp3f, rp4])
    case("P2: a required missing value with no counterexample gives 5 (VALID); with a counterexample in two pairs the "
         "refuted pair states are kept while the axis stays 5; the axis a check could not reproduce is X in the "
         "final decision table", "10 P2",
         (rp5["validity"], rp5["codes"]["H-P2"]["E"]["ion"]["code"], rp5["codes"]["H-P2"]["E"]["ion"]["pairs"],
          rp6["validity"], rp6["codes"]["H-P2"]["E"]["ion"]["code"], rp6["codes"]["H-P2"]["E"]["ion"]["pairs"],
          rp1f["codes"]["H-P2"]["E"]["ion"]["code"],
          [d_["next"][:12] for d_ in dec(op1f, "P2 measures, run E")]),
         ("VALID", 5, {"4,5": "NOT_IDENTIFIABLE", "5,6": "NOT_IDENTIFIABLE", "6,7": "NOT_IDENTIFIABLE"},
          "VALID", 5, {"4,5": "REFUTED", "5,6": "REFUTED", "6,7": "NOT_IDENTIFIABLE"}, "X", ["INADMISSIBLE"]),
         results=[rp5, rp6, rp1f])
    # 11 numerics and codes (analyze-level)
    def wide_jc(r, t):
        if t != "E":
            return

        def ed(f_):
            beta, jb = float(f_[11]), float(f_[13])
            jce = 0.2 * jb / beta
            f_[10], f_[14] = repr(jce), repr(math.nextafter(beta * jce, math.inf))
            return f_
        w5_edit(r, lambda f_: f_[0] == "4" and f_[4] == "10" and f_[5] == "4", ed)
    rq0, oq0 = run("boundary", mutate=wide_jc)
    mq = pd.read_csv(oq0 / "model.csv")
    rq1, _ = run("boundary_drop_intervals", mutate=wide_jc, fault="drop_intervals")
    rq2, _ = run("widen_crit", fault="widen_crit")
    rq3, oq3 = run("mix_codes", fault="mix_codes", mutate=lambda r, t: w5_edit(
        r, lambda f_: f_[0] == "5" and f_[4] == "15" and f_[5] == "4", lambda f_: None))
    case("numerics (analyze): a W5 J-bar bound straddling 1e-3 stores UNRESOLVED and gives set code 4; dropping the "
         "interval, widening the line or mixing 5 into 0 are inadmissible by recomputation",
         "11 numerics", (rq0["validity"], set(mq[(mq.run == "E") & (mq.line == 4) & (mq.shell == 10) &
                                                (mq.call == 5)].status), rq0["codes"]["H-MODEL"]["E"]["SIII"],
                         has(rq1, "STATUS_RECOMPUTE"), has(rq2, "STATUS_RECOMPUTE"), has(rq3, "SET_CODE_RECOMPUTE"),
                         [(d_["code"], "extension candidate" in d_["next"]) for d_ in dec(oq3, "SIII two")],
                         rq3["codes"]["H-MODEL_sets"]["R"]["SIII"].get("provisional_code")),
         ("VALID", {"UNRESOLVED"}, 4, True, True, True, [("X", False)], 0),
         fault="drop_intervals|widen_crit|mix_codes", results=[rq0, rq1, rq2, rq3])
    case("numerics (helpers): an interval straddling 1e-3 is UNRESOLVED; X precedes every science state; a refutation "
         "survives a missing cell", "11 numerics",
         (det_seedmem_scope_le(I(0.9e-3, 1.0e-3, 1.1e-3), CRIT),
          det_seedmem_scope_universal(["SURVIVES", "X", "REFUTED"]),
          det_seedmem_scope_universal(["SURVIVES", "NOT_IDENTIFIABLE", "REFUTED"]),
          det_seedmem_scope_set_code(["X", "REFUTED", "SURVIVES"])["code"]),
         ("UNRESOLVED", "X", "REFUTED", "X"))
    # 12 preservation
    try:
        det_seedmem_scope_analyze(Path(base) / "clean/in/census.csv",
                                  {t: str(Path(base) / f"clean/in/{t}") for t in ("E", "R", "X")}, src, o0, lib,
                                  FX_TARGET)
        pres = "WROTE"
    except FileExistsError:
        pres = "REFUSED_BEFORE_WRITING"
    case("preservation: an existing result path is refused before writing", "12 preservation", pres,
         "REFUSED_BEFORE_WRITING")
    return res


# ------------------------------------------------------------------ CLI
def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    librec = det_seedmem_scope_build_lib(a.production_source, out / "lib")
    if librec["rc"] != 0:
        (out / "selftest.json").write_text(json.dumps({"failures": 1, "lib": librec}, indent=1))
        return 1
    lib = det_seedmem_scope_Lib(librec["lib"])
    tests = det_seedmem_scope_selftest_cases(out / "fx", lib, Path(a.production_source))
    fails = sum(1 for t in tests if not t["pass"])
    summary = {"schema": "DET_SEEDMEM_SCOPE_SELFTEST_V2", "readout_sha256": sha(Path(__file__).resolve()),
               "lib": librec, "cases": tests, "failures": fails, "elapsed_s": time.time() - t0,
               "bundles_covered": sorted({t["bundle"] for t in tests}),
               "rc_meaning": "analyze_rc_equivalent = the rc cmd_analyze returns for that validity (0 VALID, 2 X); the "
                             "fixture analyses are function calls, not separate processes",
               "not_demonstrated": ["a production-side W5 ledger that is self-consistent but carries a wrong generation "
                                    "label beyond the dump / capture / row labels checked here",
                                    "errors shared by the forward-ledger and source-population checks and the "
                                    "inversion (same wrong input or coefficient on both sides): these checks prove "
                                    "implementation consistency, not population authority, SE validity or model "
                                    "applicability"],
               "dependencies": {str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (POP, RATES, MECH, REP)}}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"failures": fails, "cases": len(tests), "elapsed_s": summary["elapsed_s"]}))
    return 0 if fails == 0 else 1


def cmd_analyze(a):
    out = Path(a.out)
    sj = json.loads((Path(a.selftest) / "selftest.json").read_text())
    if sj.get("readout_sha256") != sha(Path(__file__).resolve()) or sj.get("failures") != 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": ["SELFTEST_NOT_THIS_READOUT_OR_FAILED"]}))
        return 2
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=False)
    librec = det_seedmem_scope_build_lib(a.production_source, out / "lib")
    lib = det_seedmem_scope_Lib(librec["lib"])
    runs = {"E": a.e_run, "R": a.r_run, "X": a.x_run}
    res = det_seedmem_scope_analyze(a.census, runs, a.production_source, out / "analysis", lib, prereg=a.prereg)
    man = {"schema": "DET_SEEDMEM_SCOPE_MANIFEST_V1",
           "contract": {"path": str(Path(a.prereg).resolve()), "sha256": sha(a.prereg)},
           "readout": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
           "imports": {str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (POP, RATES, MECH, REP)},
           "imported_functions": {"det_siiii_pop_readout": ["I", "sha", "up", "dn", "recover", "capture_records",
                                                            "prod_tau_S", "C_SOB"],
                                  "det_siiii_rates_readout": ["det_siiii_rates_coll", "det_siiii_rates_exports"],
                                  "det_siiii_1206mech_readout": ["det_1206mech_meta", "det_1206mech_check_row"],
                                  "det_siiii_repair_readout": ["det_siiii_repair_w5_load", "det_siiii_repair_cols"]},
           "lib": librec, "selftest": {"path": str(Path(a.selftest) / "selftest.json"),
                                       "sha256": sha(Path(a.selftest) / "selftest.json")},
           "production_source": {f: sha(Path(a.production_source) / f) for f in
                                 ("line_net_rate.c", "line_net_rate.h", "opacity_publication.c", "opacity_publication.h",
                                  "lumina_plasma.c", "lumina_cmfgen.c", "lumina_atomic.c", "lumina_cuda.cu", "lumina.h")},
           "census": {"path": a.census, "sha256": sha(a.census)},
           "argv": sys.argv, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "inputs": {}, "outputs": {}}
    for tag, run in runs.items():
        for rel in ("input/resolved_lumina.exports", "input/deck.sha256", "input/run_build_identity.json",
                    "STAGE_IDENTITY.txt", "stdout.log", "work/w5_lines.csv", "work/det_obs_capture/lines.bin",
                    "work/det_obs_capture/meta.txt", "work/det_obs_capture/manifest.json", "work/det_obs_capture/ne.f64"):
            p = Path(run) / rel
            if p.exists():
                man["inputs"][f"{tag}:{rel}"] = {"sha256": det_seedmem_scope_sha_cached(p), "bytes": p.stat().st_size}
        for p in sorted((Path(run) / "work/physics_comparison").glob("physics_DET_iter*")):
            man["inputs"][f"{tag}:work/physics_comparison/{p.name}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
    for k in ("line_list.csv", "levels.csv"):
        p = Path(runs["E"]) / "input/model" / k
        man["inputs"][f"E:input/model/{k}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
    if (out / "analysis").exists():
        for p in sorted((out / "analysis").iterdir()):
            man["outputs"][p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    man["elapsed_s"] = time.time() - t0
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "reasons": res.get("reasons"), "codes": res.get("codes")},
                     default=str)[:4000])
    return 0 if res["validity"] == "VALID" else 2


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--production-source", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("analyze")
    for k in ("--prereg", "--census", "--e-run", "--r-run", "--x-run", "--production-source", "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit({"selftest": cmd_selftest, "analyze": cmd_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
