#!/usr/bin/env python3
"""DET-SEEDMEM-SCOPE readout (prereg docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md, revision 1).

Offline only: no production change, no production run, GPU 0.  Fixed sample = the Si III / S III cells of the
diag6 census (W5 shells 10,15,19,24,30; E' call-2 beta < 1e-3), with the Si III 1206.50 A control and the
line 471957 / shell 10 counterexample.  Computed from stored W5 ledgers, generation-labelled physics dumps,
the generation-7 capture and the deck, with the sealed production pure functions (line_net_rate.c built
unmodified into a CPU shared library):
  identifiability.csv   per run / key / call or generation: what can be identified, with reasons
  model.csv             conditional two-level source applicability (E', X off form; R' NRB form), calls 3-7
  cross_substitution.csv E'/X fixed-background exchange of the local emission term L (calls 2-6)
  pathways.csv          generation-7 bound-bound budgets of the target and its connected same-ion transitions
  stationarity.csv      P2 measures (ion sums, n_e, J-bar, recoverable level populations), adjacent generations
  next_contract.json    decision table for the next NRB contract (prereg 5.4)
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
LINK_REASONS = ("W5_GENERATION_MISMATCH", "W5_INIT_IDENTITY", "W5_BETA_LEDGER", "W5_JBAR_LEDGER", "W5_IDENTITY")
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


def det_seedmem_scope_universal(states):
    """prereg 6.3: confirmed counterexample -> REFUTED; else missing -> NOT_IDENTIFIABLE; else boundary overlap ->
    UNRESOLVED; else SURVIVES."""
    s = list(states)
    if not s:
        return "NOT_IDENTIFIABLE"
    for st in ("REFUTED", "NOT_IDENTIFIABLE", "UNRESOLVED"):
        if st in s:
            return st
    return "SURVIVES"


def det_seedmem_scope_le(iv, crit, fault=None):
    """|d| <= crit decided on the whole interval (UNRESOLVED when the interval straddles the line)."""
    if iv is None:
        return "NOT_IDENTIFIABLE"
    if fault == "drop_intervals":
        return "SURVIVES" if iv.mid <= crit else "REFUTED"
    if fault == "widen_crit":
        crit = crit * 100.0
    if iv.hi <= crit:
        return "SURVIVES"
    if iv.lo > crit:
        return "REFUTED"
    return "UNRESOLVED"


# ------------------------------------------------------------------ sealed production pure functions (CPU library)
def det_seedmem_scope_build_lib(src, out):
    """build the sealed line_net_rate.c unmodified into a CPU shared library inside the new result path."""
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    lib = out / "libline_net_rate.so"
    cmd = ["gcc", "-O2", "-std=gnu11", "-fPIC", "-shared", "-I", str(src), "-o", str(lib), str(src / "line_net_rate.c"),
           "-lm"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"cmd": " ".join(cmd), "rc": p.returncode, "stderr_tail": p.stderr[-2000:], "elapsed_s": time.time() - t0,
           "source_sha256": sha(src / "line_net_rate.c"), "header_sha256": sha(src / "line_net_rate.h"),
           "lib": str(lib), "lib_sha256": sha(lib) if lib.exists() else None,
           "gcc": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.split("\n")[0]}
    return rec


class det_seedmem_scope_Lib:
    """ctypes binding of the two production functions this stage consumes."""

    def __init__(self, path):
        self.path = str(path)
        L = ctypes.CDLL(self.path)

        class BR(ctypes.Structure):
            _fields_ = [("beta", ctypes.c_double), ("R_absorb", ctypes.c_double), ("R_stim", ctypes.c_double),
                        ("R_spont", ctypes.c_double)]
        self.BR = BR
        f = L.line_net_bracket_rates
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_double] * 4 + [ctypes.c_uint] + [ctypes.c_double] * 3 + [ctypes.POINTER(BR)]
        g = L.line_net_cmfgen_exponx
        g.restype = ctypes.c_int
        g.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.f, self.g = f, g

    def bracket(self, ct, le, tau, jbar, prov, B_lu, B_ul, A_ul):
        o = self.BR()
        rc = self.f(ct, le, tau, jbar, prov, B_lu, B_ul, A_ul, ctypes.byref(o))
        return rc, o.beta, o.R_absorb, o.R_stim, o.R_spont

    def exponx(self, tau):
        b, c = ctypes.c_double(), ctypes.c_double()
        rc = self.g(tau, ctypes.byref(b), ctypes.byref(c))
        return rc, b.value, c.value


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
    lv = pd.read_csv(model_dir / "levels.csv", float_precision="round_trip")
    lv = lv[lv.atomic_number.isin(Zs) & (lv.ion_number == ion)]
    levels = {(int(r.atomic_number), int(r.level_number)): {"E": float(r.energy_eV), "g": int(r.g),
                                                             "sl": int(r.super_level)} for r in lv.itertuples()}
    sl_size = lv.groupby(["atomic_number", "super_level"]).size().to_dict()
    at = {}
    for r in lines.itertuples():
        Z, lo, upl = int(r.atomic_number), int(r.level_number_lower), int(r.level_number_upper)
        L_lo, L_up = levels.get((Z, lo)), levels.get((Z, upl))
        if L_lo is None or L_up is None:
            continue
        at[int(r.line_id)] = {"Z": Z, "lo": lo, "up": upl, "f_lu": float(r.f_lu), "nu": float(r.nu),
                              "B_lu": float(r.B_lu), "B_ul": float(r.B_ul), "A_ul": float(r.A_ul),
                              "lam_cm": float(r.wavelength_cm), "g_l": L_lo["g"], "g_u": L_up["g"], "E_l": L_lo["E"],
                              "E_u": L_up["E"], "sl_lo": int(sl_size[(Z, L_lo["sl"])]),
                              "sl_up": int(sl_size[(Z, L_up["sl"])])}
    return {"at": at, "levels": levels, "sl_size": {f"{k[0]}:{k[1]}": int(v) for k, v in sl_size.items()}, "why": why}


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
    return {"branch": branch, "floor": floor, "generic_coldata": generic, "feiii_only_coldata": True,
            "source_snippets": present, "supported": supported,
            "deck_collision_files": files,
            "loader_gated_by_parity_or_omega": ("if (omega_cmfgen_enabled()) {" in cuda_text and
                                                "load_ion_coldata_manifest(&atom_data, ref_dir)" in cuda_text),
            "loader_called": parity or omcm,
            "stdout_loader_markers": sum(1 for ln in so.split("\n") if "[OMEGA-CMFGEN]" in ln or "[ARTIS-PARITY" in ln),
            "matrix_consumes_col_data": generic}


def det_seedmem_scope_dumps(run):
    """generation-labelled physics dumps: population generation -> n_e, T_e, NLTE ion sums (by shell_id)."""
    d = Path(run) / "work/physics_comparison"
    out = {"ne": {}, "Te": {}, "ion": {}, "rad": {}, "files": {}, "why": []}
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
    for g in out["rad"]:
        if out["rad"][g] != g - 1:
            out["why"].append(f"DUMP_RADIATION_NOT_POPULATION_MINUS_1:{g}")
    return out


def det_seedmem_scope_run(run, tag, keys, src_text, fault=None, cuda_text=""):
    run = Path(run)
    meta = MECH.det_1206mech_meta(run)
    why = [f"{tag}:{w}" for w in meta["why"]]
    hist, contra = REP.det_siiii_repair_w5_load(run, keys)
    if contra:
        why.append(f"{tag}:W5_CONTRADICTORY_DUPLICATE:{len(contra)}")
    dumps = det_seedmem_scope_dumps(run)
    why += [f"{tag}:{w}" for w in dumps["why"]]
    cne = run / "work/det_obs_capture/ne.f64"
    if 7 not in dumps["ne"] or not cne.exists() or dumps["ne"][7].tobytes() != cne.read_bytes():
        why.append(f"{tag}:LINK_NE_DUMP7_NOT_CAPTURE")
    return {"run": str(run), "tag": tag, "meta": meta, "t_exp": meta["t_exp"], "exports": meta["exports"],
            "form": "NRB" if meta["nrb"] else "OFF", "hist": hist, "dumps": dumps,
            "coll": det_seedmem_scope_coll_authority(meta["exports"], src_text, fault, run, cuda_text), "why": why}


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
            Ra_i, Rs_i, Rp_i = I.x(B) * ct, I.x(Bu) * ct, I.rel(Rp, 1)
            Ra_i.mid, Rs_i.mid = Ra, Rs
            if fault == "omit_beta":
                Rp_i = I.x(A)
            if fault == "omit_stim":
                Rs_i = I.x(0.0)
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
    return {"n_u": n_u, "n_l": n_l}, None


# ------------------------------------------------------------------ registered computations
def det_seedmem_scope_model(runs, sample, deck, csob, lib, fault=None):
    """prereg 4.2: d = S~_E,meas(k) / S(x(call k-1 radiation, generation-k n_e T_e)) - 1, calls 3-7."""
    rows, ident = [], []
    for tag, rd in runs.items():
        form = rd["form"]
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            for k in MODEL_CALLS:
                rk, rp = det_seedmem_scope_row(rd, line, s, k), det_seedmem_scope_row(rd, line, s, k - 1)
                if fault == "beta_current":
                    rp = rk
                why = det_seedmem_scope_row_check(rk, k, a, Z) + det_seedmem_scope_row_check(rp, k - 1, a, Z)
                gk = k + 1 if fault == "material_plus1" else k
                ne = rd["dumps"]["ne"].get(gk, [np.nan] * (s + 1))
                te = rd["dumps"]["Te"].get(gk, [np.nan] * (s + 1))
                rec = {"run": tag, "form": form, "Z": Z, "line": line, "shell": s, "call": k, "material_generation": k,
                       "radiation_call": k - 1, "sl_lo": a["sl_lo"], "sl_up": a["sl_up"],
                       "independent_levels": int(a["sl_lo"] == 1 and a["sl_up"] == 1),
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
                    H[(lt, zt)] = {"rates": (Ra, Rs, Rp), "nrb": None if s_nrb is None else s_nrb.mid,
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


def det_seedmem_scope_budget(runs, sample, deck, conn, csob, lib, fault=None):
    """prereg 4.4: generation-7 populations (W5 call 7 inversion) and call-6 rates of the target and of every
    connected same-ion transition at the shell; budgets are bound-bound only (bf and line-free channels are not
    stored), so closure is never claimed."""
    rows, ident = [], []
    for tag, rd in runs.items():
        form, t_exp = rd["form"], rd["t_exp"]
        ne7, te7 = rd["dumps"]["ne"].get(7), rd["dumps"]["Te"].get(7)
        table = {}                      # (Z, level, shell) -> generation-7 populations recovered from any loaded line
        for (l2, s2), h in rd["hist"].items():
            b = deck["at"].get(l2)
            r7 = h.get(7)
            if b is None or r7 is None or det_seedmem_scope_row_check(r7, 7, b, b["Z"]):
                continue
            inv, _ = det_seedmem_scope_invert(r7, b, t_exp, csob, lib, fault)
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
            flows, pops, missing, pending, aonly = {}, {}, [], [], 0
            for l2 in lines:
                b = deck["at"][l2]
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
                    missing.append(f"{l2}:{rsn}")
                    continue
                rad, br = det_seedmem_scope_rates(r6, b, form, lib)
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
            dup = any(len(v) > 1 for v in flows.values())
            count_ok = len(set(lines)) == expected and len(lines) == expected
            complete_bb = (not missing) and count_ok and not dup
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
            rec = {"run": tag, "Z": Z, "line": line, "shell": s, "n_connected": len(conn[line]),
                   "n_lines_used": len(set(flows)), "n_a_only_outside_domain": aonly, "missing": ";".join(missing),
                   "count_ok": int(count_ok),
                   "duplicate": int(dup), "complete_bb": int(complete_bb), "level_populations_consistent": int(consistent),
                   "closed": 0, "closure_reason": "BF_AND_LINE_FREE_CHANNELS_NOT_STORED" +
                   ("" if a["sl_lo"] == 1 and a["sl_up"] == 1 else ";SUPER_LEVEL_ROW"),
                   "sl_lo": a["sl_lo"], "sl_up": a["sl_up"]}
            REP.det_siiii_repair_cols(rec, "F_target", Ft)
            REP.det_siiii_repair_cols(rec, "F_other_into_upper", Fup if Ft is not None else None)
            REP.det_siiii_repair_cols(rec, "F_other_into_lower", Flo if Ft is not None else None)
            if Ft is not None:
                rec["F_target_sign"] = ("POSITIVE" if Ft.lo > 0 else "NEGATIVE" if Ft.hi < 0 else "ZERO_NOT_SEPARATED")
                tot = Ft + Fup
                if Ft.lo > 0 or Ft.hi < 0:
                    rec["partial_upper_residual_ratio"] = abs(tot.mid) / abs(Ft.mid)
                    rec["other_opposes_target"] = int((Ft.mid > 0) != (Fup.mid > 0) and Fup.mid != 0.0)
            rec["status"] = "NOT_IDENTIFIABLE"
            if not count_ok or dup:
                rec["status"] = "X"
            rows.append(rec)
            ident.append({"run": tag, "Z": Z, "line": line, "shell": s, "question": "generation7_bb_budget",
                          "generation": 7, "status": "NOT_IDENTIFIABLE" if rec["status"] != "X" else "X",
                          "reason": rec["closure_reason"] + ("" if complete_bb else ";BB_INCOMPLETE:" + ";".join(missing))})
    return rows, ident


def det_seedmem_scope_capture_check(runs, sample, deck, csob, lib, fault=None):
    """prereg 3.2 (3): W5 call-7 inversion vs the independent generation-7 capture inversion (same populations)."""
    rows, why = [], []
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
            rows.append({"run": tag, "line": line, "shell": s, "w5_inversion": rsn or "OK",
                         "capture": "MISSING" if c is None else ("UNMAPPED" if c["mapped"] != 1 else "OK"),
                         "compatible": None if ok is None else int(ok)})
    return rows, why


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


def det_seedmem_scope_p2(runs, sample, deck, csob, lib, fault=None):
    """prereg 5.3: adjacent-generation changes of four registered axes; two consecutive small changes over the fixed
    common pairs (4,5),(5,6),(6,7)."""
    rows, codes = [], {}
    levels = sorted({(Z, deck["at"][line][w], s) for Z, line, s in sample["keys"] for w in ("lo", "up")})
    for tag, rd in runs.items():
        D = rd["dumps"]
        series = {"ion": {}, "ne": {}, "jbar": {}, "level": {}}
        for s in sorted({k[2] for k in sample["keys"]}):
            for Z, q in ION_QTY:
                series["ion"][(Z, q, s)] = {g: (float(D["ion"][g][(Z, q)][s]) if g in D["ion"] else None)
                                            for g in POP_GENS}
            series["ne"][s] = {g: (float(D["ne"][g][s]) if g in D["ne"] else None) for g in POP_GENS}
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            ser = {}
            for g in MODEL_CALLS:
                r = det_seedmem_scope_row(rd, line, s, g)
                ser[g] = float(r["Jbar"]) if r is not None and not det_seedmem_scope_row_check(r, g, a, Z) else None
            series["jbar"][(line, s)] = ser
        rec_pop = {}
        for Z, line, s in sample["keys"]:
            a = deck["at"][line]
            for g in MODEL_CALLS:
                r = det_seedmem_scope_row(rd, line, s, g)
                if r is None or det_seedmem_scope_row_check(r, g, a, Z):
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
                    ms.append(m)
                M = I(max((x.lo for x in ms), default=0.0), max((x.mid for x in ms), default=0.0),
                      max((x.hi for x in ms), default=0.0))
                per_g[g] = {"M": M, "missing": miss, "structural_zero": zero, "n": len(ms)}
                rows.append({"run": tag, "axis": axis, "generation": g, "pair": f"{g - 1}->{g}", "M_lo": M.lo,
                             "M_mid": M.mid, "M_hi": M.hi, "n_quantities": len(ms), "missing": miss,
                             "structural_zero": zero,
                             "small": "SURVIVES" if (M.hi < P2_CRIT and not miss) else
                             ("REFUTED" if M.lo >= P2_CRIT else ("NOT_IDENTIFIABLE" if miss else "UNRESOLVED"))})
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
            first = next((f"{a_}->{b_}" for (a_, b_), p in zip(P2_PAIRS, pair_st) if p == "SURVIVES"), None)
            codes.setdefault(tag, {})[axis] = {"code": CODE[code], "state": code, "pairs": dict(zip(
                [f"{a_},{b_}" for a_, b_ in P2_PAIRS], pair_st)), "first_two_small": first}
    return rows, codes


def det_seedmem_scope_identifiability(runs, sample, deck):
    rows = []
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
    for q, why in REGISTERED_5.items():
        rows.append({"run": "ALL", "Z": None, "line": None, "shell": None, "question": q, "generation": None,
                     "status": "NOT_IDENTIFIABLE", "reason": why})
    return rows


def det_seedmem_scope_next(codes, sets_model, cross_code, budget_rows, p2codes):
    """prereg 5.4 decision table (consumption rules; not a run authorisation)."""
    R = sets_model.get("R", {})
    dec = []
    ctrl = R.get("control_1206")
    dec.append({"result": "1206 control (R' NRB, 25 pairs)", "code": ctrl,
                "next": "keep P1 source-match range (line 600693, W5 shells, calls 3-7) and the 1e-3 comparison line"
                if ctrl == 0 else "P1 source-match control not reproduced: report before any P1 registration"})
    for st in ("SiIII_other", "SIII"):
        c = R.get(st)
        if c == 1:
            nxt = f"do not generalise the source-match universal condition to the fixed {st} set"
        elif c == 0:
            nxt = f"offer the whole fixed {st} set as a conditional extension candidate (no pass-only subset)"
        else:
            nxt = f"{st} set undecided (code {c}): keep it outside P1"
        dec.append({"result": f"{st} two-level NRB source model (R')", "code": c, "next": nxt})
    if cross_code == 0:
        dec.append({"result": "production NRB rates invariant under exchange of L at fixed c, tau", "code": 0,
                    "next": "split P1 into (a) explicit self-term removal of the rates and (b) reduced-model source match"})
        if any(R.get(st) == 1 for st in ("SiIII_other", "SIII")):
            dec.append({"result": "rates invariant but the reduced source model refuted outside 1206", "code": 1,
                        "next": "P1 refutation wording must distinguish NRB implementation failure from reduced-model "
                                "failure"})
    else:
        dec.append({"result": "NRB rate invariance", "code": cross_code,
                    "next": "rate self-term removal not established: P1 (a) stays a hypothesis"})
    dec.append({"result": "generation-7 bound-bound budgets", "code": 5,
                "next": "closure not identifiable (bf and line-free channels, super-level rows not stored); minimal "
                        "storage needed: per-level bf rates and the SE row residual per solve unit at the consumed "
                        "generation"})
    for tag, ax in p2codes.items():
        ion_ne = [ax.get("ion", {}).get("code"), ax.get("ne", {}).get("code")]
        rad_lv = [ax.get("jbar", {}).get("code"), ax.get("level", {}).get("code")]
        if all(c == 0 for c in ion_ne) and any(c != 0 for c in rad_lv):
            nxt = "register the radiation / level measures separately in the next P2"
        elif all(c == 0 for c in ion_ne + rad_lv):
            nxt = "only 'two consecutive small changes of the registered W5 measures' may be claimed"
        elif any(c != 0 for c in ion_ne):
            nxt = "P2 stationarity not reached on ion sums / n_e; no substitution by other measures"
        else:
            nxt = "P2 undecided"
        dec.append({"result": f"P2 measures, run {tag}", "codes": ax, "next": nxt})
    return {"schema": "DET_SEEDMEM_SCOPE_NEXT_V1", "decisions": dec,
            "unchanged": ["1e-3 and 0.0043 dex are not re-derived from residual maxima or quantiles",
                          "call 8+, W5-outside shells, field convergence and two-seed converged solutions are outside "
                          "this evidence"],
            "not_authorised": "no NRB run, production change, constant repair or extension follows from this table"}


# ------------------------------------------------------------------ analyze
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
        res = {"schema": "DET_SEEDMEM_SCOPE_RESULT_V1", "validity": "X", "reasons": why, "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    conn = det_seedmem_scope_connect(sample["keys"], deck["at"])
    deck["r"] = {l: det_seedmem_scope_r(deck["at"][l], csob) for l in {k[1] for k in sample["keys"]} | {
        l2 for v in conn.values() for l2 in v}}
    keys = {(k[1], k[2]) for k in sample["keys"]} | {(l2, k[2]) for k in sample["keys"] for l2 in conn[k[1]]}
    cuda = src / "lumina_cuda.cu"
    cuda_text = cuda.read_text(errors="replace") if cuda.exists() else ""
    runs = {tag: det_seedmem_scope_run(p, tag, keys, src_text, fault, cuda_text) for tag, p in runs_paths.items()}
    for tag, rd in runs.items():
        why += rd["why"]
        if rd["form"] != target["forms"][tag]:
            why.append(f"{tag}:RATE_FORM_NOT_REGISTERED:{rd['form']}")
    for tag, rd in runs.items():
        for Z, line, s in sample["keys"]:
            for c in range(1, 8):
                w = det_seedmem_scope_row_check(det_seedmem_scope_row(rd, line, s, c), c, deck["at"][line], Z)
                if det_seedmem_scope_is_link(w):
                    why.append(f"{tag}:SAMPLE_ROW_LINK:{line}:{s}:{c}:{w[0]}")
    cap_rows, cap_why = det_seedmem_scope_capture_check(runs, sample, deck, csob, lib, fault)
    why += cap_why
    timing = {"inputs_s": time.time() - t0}
    if why:
        res = {"schema": "DET_SEEDMEM_SCOPE_RESULT_V1", "validity": "X", "reasons": why[:200], "n_reasons": len(why),
               "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        pd.DataFrame(cap_rows).to_csv(out / "capture_check.csv", index=False)
        return res
    model_rows, ident_m = det_seedmem_scope_model(runs, sample, deck, csob, lib, fault)
    cross_rows = det_seedmem_scope_cross(runs, sample, deck, lib, fault)
    budget_rows, ident_b = det_seedmem_scope_budget(runs, sample, deck, conn, csob, lib, fault)
    p2_rows, p2codes = det_seedmem_scope_p2(runs, sample, deck, csob, lib, fault)
    ident = det_seedmem_scope_identifiability(runs, sample, deck) + ident_m + ident_b
    timing["compute_s"] = time.time() - t0 - timing["inputs_s"]
    md = pd.DataFrame(model_rows)
    sets_model, counter = {}, {}
    for tag in runs:
        sets_model[tag] = {}
        for st, ks in sample["sets"].items():
            ks = set(ks)
            sub = md[(md.run == tag) & md.apply(lambda r: (r.Z, r.line, r.shell) in ks, axis=1)]
            states = list(sub.status)
            sets_model[tag][st] = "X" if "X" in states else CODE[det_seedmem_scope_universal(states)]
        cl, cs = target["counter"]
        sub = md[(md.run == tag) & (md.line == cl) & (md.shell == cs)]
        counter[tag] = {int(r.call): r.status for r in sub.itertuples()}
    cr = pd.DataFrame(cross_rows)
    cst = list(cr.status) if len(cr) else []
    cross_code = "X" if "X" in cst else CODE[det_seedmem_scope_universal(cst)]
    br = pd.DataFrame(budget_rows)
    bad_budget = int((br.status == "X").sum()) if len(br) else 0
    validity = "VALID"
    reasons = []
    if any(v == "X" for d in sets_model.values() for v in d.values()) or cross_code == "X" or bad_budget:
        validity, reasons = "X", ["MODEL_OR_CROSS_OR_BUDGET_LINK"]
    codes = {
        "H-MODEL": sets_model,
        "H-COUNTEREXAMPLE": {"line": target["counter"][0], "shell": target["counter"][1],
                             "R_two_level_nrb_applicability":
                             CODE.get(det_seedmem_scope_universal(list(counter.get("R", {}).values())), "X"),
                             "per_call": counter},
        "H-LOCAL": cross_code,
        "H-MULTILEVEL": {tag: {"code": 5, "reason": "closure not identifiable: bf / line-free channels and super-level "
                                                    "SE rows are not stored",
                               "complete_bb_targets": int(br[(br.run == tag)].complete_bb.sum()) if len(br) else 0,
                               "targets": int((br.run == tag).sum()) if len(br) else 0} for tag in runs},
        "H-P2": p2codes,
        "registered_not_identifiable": {q: 5 for q in REGISTERED_5},
    }
    cl, cs = target["counter"]
    codes["H-COUNTEREXAMPLE"]["call3_detail_R"] = det_seedmem_scope_counter_detail(runs["R"], deck, cl, cs, lib)
    nxt = det_seedmem_scope_next(codes, sets_model, cross_code, budget_rows, p2codes)
    sample_json = {"schema": "DET_SEEDMEM_SCOPE_SAMPLE_V1", "authority": str(census), "authority_sha256": sha(census),
                   "selection": {"Z": list(target["Z"]), "ion": target["ion"], "shells": list(target["shells"]),
                                 "beta2_E_below": target["beta_max"]},
                   "counts": sample["counts"], "n_keys": len(sample["keys"]), "keys": sample["keys"],
                   "sets": {k: len(v) for k, v in sample["sets"].items()}, "control": target["control"],
                   "counter": list(target["counter"]),
                   "auxiliary_lines": {str(l): v for l, v in conn.items()},
                   "level_set": sorted({(Z, deck["at"][line][w]) for Z, line, _ in sample["keys"] for w in ("lo", "up")}),
                   "super_level_sizes_of_sample": {str(k[1]): [deck["at"][k[1]]["sl_lo"], deck["at"][k[1]]["sl_up"]]
                                                   for k in sample["keys"]}}
    (out / "sample.json").write_text(json.dumps(sample_json, indent=1, default=str))
    pd.DataFrame(ident).to_csv(out / "identifiability.csv", index=False)
    md.to_csv(out / "model.csv", index=False)
    cr.to_csv(out / "cross_substitution.csv", index=False)
    br.to_csv(out / "pathways.csv", index=False)
    pd.DataFrame(p2_rows).to_csv(out / "stationarity.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(out / "capture_check.csv", index=False)
    (out / "next_contract.json").write_text(json.dumps(nxt, indent=1, default=str))
    coll = {tag: rd["coll"] for tag, rd in runs.items()}
    res = {"schema": "DET_SEEDMEM_SCOPE_RESULT_V1", "validity": validity, "reasons": reasons, "fault": fault,
           "codes": codes, "collision_authority": coll, "rate_forms": {t: rd["form"] for t, rd in runs.items()},
           "counts": {"keys": len(sample["keys"]), "model_rows": len(md), "cross_rows": len(cr),
                      "budget_rows": len(br)},
           "timing_s": timing,
           "claim_ceiling": "offline readout of stored states; no production change, no physical correctness, no "
                            "public agreement, no causal pathway apportionment, no run authorisation"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ fixtures (direct tests, prereg 8.2)
FX_SHELLS = (10, 15)
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
    sh = []
    for fn in ("line_list.csv", "levels.csv"):
        sh.append(f"{sha(model_dir / fn)}  {fn}")
    (model_dir.parent / "deck.sha256").write_text("\n".join(sh) + "\n")


def det_seedmem_scope_fx_run(root, tag, lib, seed_scale, ne_scale, nrb, ion_series, csob, mutate=None):
    """one fixture run: W5 calls 1-7 generated forward by the production relations from populations that obey
    the run's two-level model for lines 0, 1 (Si III) and 4 (S III); line 3 (counter) and the connected lines take
    populations set directly, so their two-level model does not hold."""
    run = root / tag
    (run / "input").mkdir(parents=True)
    det_seedmem_scope_fx_deck(run / "input/model")
    (run / "work/physics_comparison").mkdir(parents=True)
    (run / "work/det_obs_capture").mkdir(parents=True)
    deck = det_seedmem_scope_deck(run / "input/model", (14, 16), 2)
    at = deck["at"]
    ex = ['declare -x LUMINA_NLTE_COLL_FIX="1"', 'declare -x LUMINA_NLTE_COLL_FLOOR="0"',
          'declare -x LUMINA_SUPER_LEVELS="1"', 'declare -x LUMINA_INNER_BB_SCALE="1.0"']
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
    (capd / "meta.txt").write_text(f"time_explosion={FX_TEXP.hex()}\nT_inner={(10020.0).hex()}\n"
                                   f"inner_bb_scale={(1.0).hex()}\nmaterial_generation=7\nrad_generation=7\n")
    (capd / "manifest.json").write_text(json.dumps({"generations": {"material": 7, "nlte_population_committed": 7,
                                                                   "radiation": 7}, "transaction": "COMMITTED",
                                                   "census": {"mapped_B_fallback": 0}}))
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


def det_seedmem_scope_selftest_cases(base, lib, src):
    csob = det_seedmem_scope_csob(src)
    res = []

    def case(name, got, want):
        ok = got == want
        res.append({"case": name, "pass": bool(ok), "got": str(got)[:600], "want": str(want)[:600]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:200]} want {str(want)[:200]}", flush=True)

    def run(name, fault=None, mutate=None, census_mutate=None, ion_series=None, target=FX_TARGET):
        runs, census = det_seedmem_scope_fx(Path(base) / name / "in", lib, csob, mutate, census_mutate, ion_series)
        return det_seedmem_scope_analyze(census, runs, src, Path(base) / name / "out", lib, target, fault), \
            Path(base) / name / "out"

    r0, o0 = run("clean")
    m0 = pd.read_csv(o0 / "model.csv")
    ctrl = m0[(m0.run == "R") & (m0.line == 0)]
    case("clean fixture: valid; control (R' NRB) survives at calls 3-7; counter line refuted; S III model line survives",
         (r0["validity"], r0["codes"]["H-MODEL"]["R"]["control_1206"], r0["codes"]["H-COUNTEREXAMPLE"][
             "R_two_level_nrb_applicability"], r0["codes"]["H-MODEL"]["R"]["SIII"], sorted(ctrl.call.unique().tolist())),
         ("VALID", 0, 1, 0, [3, 4, 5, 6, 7]))
    case("clean fixture: off model survives for E' and X control; NRB rates invariant under L exchange (H-LOCAL 0)",
         (r0["codes"]["H-MODEL"]["E"]["control_1206"], r0["codes"]["H-MODEL"]["X"]["control_1206"],
          r0["codes"]["H-LOCAL"]), (0, 0, 0))
    # 1 sample fixing
    rs1, _ = run("drop_counter", fault="drop_counter")
    rs2, _ = run("add_shell", fault="add_shell")
    case("sample: counterexample deletion and a non-W5 shell are inadmissible (G-LINK)",
         (rs1["validity"], any("COUNTER" in w for w in rs1["reasons"]), rs2["validity"],
          any("SHELL" in w or "COUNT" in w for w in rs2["reasons"])), ("X", True, "X", True))
    # 2 generation linking
    rg1, _ = run("beta_current", fault="beta_current")
    rg2, _ = run("material_plus1", fault="material_plus1")

    def dump_relabel(r, t):
        p = r / "work/physics_comparison/physics_DET_iter0004.manifest.json"
        js = json.loads(p.read_text())
        js["population_generation"] = 8
        p.write_text(json.dumps(js))
    rg3, _ = run("dump_relabel", mutate=dump_relabel)
    case("generations: current-call radiation or generation k+1 n_e break the control match; a relabelled dump is X",
         (rg1["codes"]["H-MODEL"]["R"]["control_1206"] != 0, rg2["codes"]["H-MODEL"]["R"]["control_1206"] != 0,
          rg3["validity"]), (True, True, "X"))
    # 3 missing data
    idn = pd.read_csv(o0 / "identifiability.csv")
    g2 = idn[(idn.question == "collisional_response_n_e") & (idn.generation == 2)]

    def drop_row(r, t):
        p = r / "work/w5_lines.csv"
        lines = p.read_text().split("\n")
        p.write_text("\n".join(ln for ln in lines if not ln.startswith("5,CMFGEN_NONOVERLAP_SOBOLEV,5,5,15,1,")))
    rm, om = run("drop_row", mutate=drop_row)
    mm = pd.read_csv(om / "model.csv")
    miss = mm[(mm.line == 1) & (mm.shell == 15) & (mm.call.isin([5, 6]))]
    case("missing: generation-2 n_e is NOT_IDENTIFIABLE; a removed W5 row gives 5 (not 0), the counterexample stays 1",
         (set(g2.status), set(miss.status), rm["codes"]["H-COUNTEREXAMPLE"]["R_two_level_nrb_applicability"]),
         ({"NOT_IDENTIFIABLE"}, {"NOT_IDENTIFIABLE"}, 1))
    # 4 W5 inversion
    cc = pd.read_csv(o0 / "capture_check.csv")
    rw1, _ = run("omit_r", fault="omit_r")
    rw2, _ = run("wrong_A", fault="wrong_A")
    rw3, _ = run("swap_levels", fault="swap_levels")
    case("inversion: W5 ledger inversion encloses the generation-7 populations (capture compatible); omitting r_l "
         "breaks the control; wrong A or swapped levels fail the population authority",
         (set(cc.compatible.dropna().astype(int)), rw1["codes"]["H-MODEL"]["R"]["control_1206"] != 0,
          rw2["validity"], rw3["validity"]), ({1}, True, "X", "X"))
    # 5 independent authority

    def cap_gen(r, t):
        p = r / "work/det_obs_capture/manifest.json"
        js = json.loads(p.read_text())
        js["generations"]["material"] = 8
        p.write_text(json.dumps(js))

    def cap_unmapped(r, t):
        p = r / "work/det_obs_capture/lines.bin"
        m = np.fromfile(p, dtype=POP.REC)
        m["S"] = m["S"] * 1.5
        m.tofile(p)
    ra1, _ = run("cap_gen", mutate=cap_gen)
    ra2, _ = run("cap_source", mutate=cap_unmapped)
    case("independent authority: a capture generation change and a replaced capture source are inadmissible",
         (ra1["validity"], ra2["validity"]), ("X", "X"))
    # 6 production NRB link
    rn1, _ = run("omit_beta", fault="omit_beta")
    rn2, _ = run("omit_stim", fault="omit_stim")

    def ledger_break(r, t):
        if t != "R":
            return
        p = r / "work/w5_lines.csv"
        out = []
        for ln in p.read_text().split("\n"):
            f = ln.split(",")
            if len(f) > 13 and f[0] == "4" and f[5] == "0" and f[4] == "10":
                f[13] = repr(float(f[13]) * (1 + 1e-9))
            out.append(",".join(f))
        p.write_text("\n".join(out))
    rn3, _ = run("ledger_break", mutate=ledger_break)
    case("NRB link: omitted beta or stimulated term breaks the control; a ledger mismatch is never consumed as OK",
         (rn1["codes"]["H-MODEL"]["R"]["control_1206"] != 0, rn2["codes"]["H-MODEL"]["R"]["control_1206"] != 0,
          rn3["validity"]), (True, True, "X"))
    # 7 cross substitution
    cx = pd.read_csv(o0 / "cross_substitution.csv")
    rc1, _ = run("move_c", fault="move_c_with_L")
    case("cross substitution: fixed c, tau and L exchange keep the NRB rates bit-identical and the ledger closes; moving "
         "c with L labelled as fixed background is inadmissible",
         (set(cx.nrb_rates_invariant_under_L.dropna().astype(int)), float(cx.nrb_D_L.abs().max()) == 0.0,
          bool((cx.off_ledger_residual.abs() <= 1e-12 * cx.off_H_XX.abs()).all()), rc1["validity"]),
         ({1}, True, True, "X"))
    # 8 multilevel budget
    pb = pd.read_csv(o0 / "pathways.csv")
    rb1, _ = run("drop_connected", fault="drop_connected")
    rb2, _ = run("double_count", fault="double_count")
    case("budget: complete bound-bound sets (out-of-domain A-only lines from recovered levels) are counted, closure "
         "stays NOT_IDENTIFIABLE (bf absent); a dropped or a "
         "double-counted connected transition is inadmissible",
         (int(pb.complete_bb.sum()) == len(pb), int(pb.n_a_only_outside_domain.sum()) > 0, set(pb.closed),
          r0["codes"]["H-MULTILEVEL"]["R"]["code"], rb1["validity"], rb2["validity"]), (True, True, {0}, 5, "X", "X"))
    # 9 collision consumption
    rk, _ = run("assume_coldata", fault="assume_coldata")
    case("collisions: the stored knobs select COLL_FIX per line; treating col_data as consumed makes the model "
         "NOT_IDENTIFIABLE instead of silently using it",
         (r0["collision_authority"]["R"]["branch"], r0["collision_authority"]["R"]["supported"],
          r0["collision_authority"]["R"]["loader_called"], r0["collision_authority"]["R"]["matrix_consumes_col_data"],
          rk["codes"]["H-MODEL"]["R"]["control_1206"],
          r0["codes"]["H-COUNTEREXAMPLE"]["call3_detail_R"].get("S_meas_in_two_level_range") in
          ("INSIDE", "OUTSIDE", "NOT_SEPARATED")), ("COLL_FIX_PER_LINE", True, False, False, 5, True))
    # 10 P2
    ion_one = {(Z, q): {g: (1.0e6 * (1.0 if g in (5, 6) else (1.3 if g == 7 else 1.0 + 0.2 * abs(5 - g))))
                        for g in POP_GENS} for Z, q in ION_QTY}
    rp1, _ = run("p2_single", ion_series=ion_one)
    rp1f, _ = run("p2_single_fault", ion_series=ion_one, fault="single_change")
    ion_zero = {(Z, q): {g: (0.0 if (Z, q) == (14, 1) and g == 5 else 1.0e6) for g in POP_GENS} for Z, q in ION_QTY}
    rp2, _ = run("p2_zero", ion_series=ion_zero)
    rp2f, _ = run("p2_zero_fault", ion_series=ion_zero, fault="drop_zero_cross")
    case("P2: clean ion sums small from 5->6 on; one small change alone is not two; a zero crossing refutes and "
         "deleting it is detected",
         (r0["codes"]["H-P2"]["E"]["ion"]["state"], rp1["codes"]["H-P2"]["E"]["ion"]["state"],
          rp1f["codes"]["H-P2"]["E"]["ion"]["state"], rp2["codes"]["H-P2"]["E"]["ion"]["state"],
          rp2f["codes"]["H-P2"]["E"]["ion"]["state"]),
         ("SURVIVES", "REFUTED", "SURVIVES", "REFUTED", "SURVIVES"))
    # 11 numerics and codes
    st = det_seedmem_scope_le(I(0.9e-3, 1.0e-3, 1.1e-3), CRIT)
    case("numerics: an interval straddling 1e-3 is UNRESOLVED; dropping the interval or widening the line flips it; "
         "a refutation survives a missing cell in the universal rule",
         (st, det_seedmem_scope_le(I(0.9e-3, 1.0e-3, 1.1e-3), CRIT, "drop_intervals"),
          det_seedmem_scope_le(I(2e-3, 3e-3, 4e-3), CRIT, "widen_crit"),
          det_seedmem_scope_universal(["SURVIVES", "NOT_IDENTIFIABLE", "REFUTED"])),
         ("UNRESOLVED", "SURVIVES", "SURVIVES", "REFUTED"))
    # 12 preservation
    try:
        det_seedmem_scope_analyze(Path(base) / "clean/in/census.csv",
                                  {t: str(Path(base) / f"clean/in/{t}") for t in ("E", "R", "X")}, src, o0, lib,
                                  FX_TARGET)
        pres = "WROTE"
    except FileExistsError:
        pres = "REFUSED_BEFORE_WRITING"
    case("preservation: an existing result path is refused before writing", pres, "REFUSED_BEFORE_WRITING")
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
    summary = {"schema": "DET_SEEDMEM_SCOPE_SELFTEST_V1", "readout_sha256": sha(Path(__file__).resolve()),
               "lib": librec, "cases": tests, "failures": fails, "elapsed_s": time.time() - t0,
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
                                 ("line_net_rate.c", "line_net_rate.h", "lumina_plasma.c", "lumina_cmfgen.c", "lumina.h")},
           "census": {"path": a.census, "sha256": sha(a.census)},
           "argv": sys.argv, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "inputs": {}, "outputs": {}}
    for tag, run in runs.items():
        for rel in ("input/resolved_lumina.exports", "input/deck.sha256", "input/run_build_identity.json",
                    "STAGE_IDENTITY.txt", "work/w5_lines.csv", "work/det_obs_capture/lines.bin",
                    "work/det_obs_capture/meta.txt", "work/det_obs_capture/manifest.json", "work/det_obs_capture/ne.f64"):
            p = Path(run) / rel
            if p.exists():
                man["inputs"][f"{tag}:{rel}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
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
