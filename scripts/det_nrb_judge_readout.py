#!/usr/bin/env python3
"""DET-NRB-JUDGE readout (prereg docs/PREREG_DET_NRB_JUDGE_2026-10-08.md, revision 1).

One registered production run N: the R' sealed binary, inputs and exports, cold start, LUMINA_PURE_CMFGEN_ITER 6 -> 12
(the only allowed difference; seven path keys re-rooted).  Read-out from the stored W5 ledgers (calls 1..13), the
generation-labelled physics dumps (populations 3..14), the generation-13 observer capture, the deck and the sealed
production pure functions (line_net_rate.c / opacity_publication.c built unmodified into a CPU library):
  P1(a)  NRB applied on every registered cell (510 keys x calls 2..13): branch, flags, ledger admissibility, the
         production bracket rates; H-LOCAL inherited by identity from SEEDMEM-SCOPE r4 plus a 60-cell recall
         (line 600693 x W5 x calls 2..13) with the local term exchanged
  P1(b)  Si III 1206.50 A conditional two-level source, core calls 3-7 and extension calls 8-13, |d| <= 1e-3
  P2     ion sums (20), n_e (5), J-bar (510), level populations (328): terminal window (12,13), M_a < 0.0043 dex
         strict; NOT_CONVERGED trajectory descriptor M_a(4), M_a(13), D_a; generation 14 kept apart
  public S II/III, Si II/III fractions at generation 13 against toy06 19.480 d (|d^f| <= 0.3 dex), and the UV /
         optical spectrum against the registered B-T secondary targets
Codes per proposition: 0 SURVIVES, 1 REFUTED, 4 UNRESOLVED, 5 NOT_IDENTIFIABLE, X inadmissible.  Comparison lines are
registered (1e-3, 0.0043 dex, 0.3 dex, the expectations file), never numerical tolerances.

Subcommands
  selftest  --production-source SRC --runner RUNNER --prior-selftest DIR --out DIR
  preflight --prereg DOC --baseline-run E --reference-run R --prior-analysis A --prior-selftest DIR
            --spectral-expectations JSON --production-source SRC --selftest DIR --runner RUNNER --out DIR
  stage     --reference-run R --preflight DIR --gpu-profile h200x2|h100x2 --runner RUNNER --run N
  analyze   --prereg DOC --run N --baseline-run E --reference-run R --prior-analysis A --prior-selftest DIR
            --spectral-expectations JSON --production-source SRC --public-spectrum F --public-si F --public-s F
            --selftest DIR --out DIR
"""
import argparse
import json
import math
import os
import platform
import re
import shutil
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
import det_seedmem_scope_readout as SEED                                         # noqa: E402
from det_siiii_pop_readout import I, sha, up, dn                                 # noqa: E402

P = Path(__file__).resolve().parent.parent
PREREG = "docs/PREREG_DET_NRB_JUDGE_2026-10-08.md"
OUTER = 12
CRIT, P2_CRIT, ION_CRIT = 1e-3, 0.0043, 0.3
CODE = SEED.CODE
ION_QTY = SEED.ION_QTY
TARGET = {"Z": (14, 16), "ion": 2, "control": 600693, "counts": {14: 86, 16: 424}, "n_keys": 510, "n_levels": 328,
          "shells": (10, 15, 19, 24, 30), "outer": OUTER, "public_time": "19.480",
          "vmid_lumina": {10: 11544.0, 15: 15184.0, 19: 18096.0, 24: 21736.0, 30: 26104.0},
          "vmid_public": {10: 11534.59, 15: 15320.31, 19: 18264.10, 24: 21629.71, 30: 25881.39}}
PATH_KEYS = ("LUMINA_MODEL_DIR", "LUMINA_CMFGEN_SIGMA_BF", "LUMINA_DEPOSITION_FILE", "LUMINA_FIXED_TE_PROFILE",
             "LUMINA_IONIZATION_REFERENCE_FILE", "LUMINA_TOPION_LEVELS_FILE", "LUMINA_PHYSICS_COMPARISON_DIR")
ITER_KEY = "LUMINA_PURE_CMFGEN_ITER"
STAGED_FILES = ("lumina_cuda", "fixed_te_profile.txt", "binary.sha256", "sigma.sha256", "deck.sha256",
                "topion.sha256", "run_build_identity.json")
STAGED_DIRS = ("model", "global_atomic")
GPU_PROFILES = {"h200x2": "h200", "h100x2": "h100"}
TIME_CEILING_S, GPU_HOURS = 75804, 42.113333
REGISTERED_5 = {
    "two_seed_nrb_terminal_equality": "single-seed run",
    "pure_nrb_effect_at_equal_iterations": "no 12-iteration off control",
    "full_multilevel_se_closure": "bf rates, SE residuals and the complete consumed state are not stored",
    "memory_pathway_causal_contributions": "local reconstruction and a single trajectory cannot apportion",
    "init_nrb_sobolev_decomposition": "INIT is a separate operator (call 1, armed=0)",
    "generation2_collisional_response": "no stored material authority for generation 2",
    "full_matrix_cellwise_rate_bytes": "the stored ledgers are not a complete matrix trace",
    "stationarity_outside_registered_sample": "outside the registered sample (W5, 510 keys, 328 levels)",
    "public_ion_column_meaning": "header naming assumption si0..si4 / s0..s4 = I..V, no external definition",
}
RUNNER_CONSTANTS = {"OUTER_ITERATIONS": str(OUTER), "TIME_CEILING_S": str(TIME_CEILING_S)}


def FT(fault, name):
    """selftest-only fault switch; several faults join with '+' (one case can inject a front and a back scope)."""
    return bool(fault) and name in str(fault).split("+")


def det_nrb_judge_gens(outer):
    """registered generation linkage (prereg 4.2) for an outer-iteration count."""
    g_obs, g_last = outer + 1, outer + 2
    return {"outer": outer, "g_obs": g_obs, "g_last": g_last, "calls": list(range(2, g_obs + 1)),
            "pops": list(range(3, g_last + 1)), "p1b_core": [k for k in range(3, 8) if k <= g_obs],
            "p1b_ext": list(range(8, g_obs + 1)), "matter_gens": list(range(3, g_last + 1)),
            "rad_gens": list(range(3, g_obs + 1)), "terminal": (outer, g_obs), "start": 4,
            "iter_obs": outer - 2, "iter_last": outer - 1}


# ------------------------------------------------------------------ small helpers
def det_nrb_judge_state(iv, crit, fault=None):
    """|x| <= crit on the whole interval (SEEDMEM le); None -> NOT_IDENTIFIABLE."""
    return SEED.det_seedmem_scope_le(iv, crit, fault)


def det_nrb_judge_small(M, missing):
    if M is None:
        return "NOT_IDENTIFIABLE"
    if M.lo >= P2_CRIT:
        return "REFUTED"
    if missing:
        return "NOT_IDENTIFIABLE"
    if M.hi < P2_CRIT:
        return "SURVIVES"
    return "UNRESOLVED"


def det_nrb_judge_sub(a, b, side):
    """a - b rounded outward only when the subtraction is inexact (TwoSum error term); an exact difference (e.g. of
    two exact zeros) stays exact so END_EQUAL is reachable (prereg 5.3)."""
    s_ = a - b
    if not math.isfinite(s_):
        return s_
    bb = s_ - a
    err = (a - (s_ - bb)) + (-b - bb)
    if err == 0.0:
        return s_
    return dn(s_) if side < 0 else up(s_)


def det_nrb_judge_end_code(a_lo, a_hi, b_lo, b_hi):
    """END_* from the stored endpoint intervals [a_lo, a_hi] (start) and [b_lo, b_hi] (end)."""
    lo, hi = det_nrb_judge_sub(b_lo, a_hi, -1), det_nrb_judge_sub(b_hi, a_lo, +1)
    if hi < 0.0:
        return "END_LOWER", lo, hi
    if lo > 0.0:
        return "END_HIGHER", lo, hi
    if lo == 0.0 and hi == 0.0:
        return "END_EQUAL", lo, hi
    return "END_UNRESOLVED", lo, hi


def det_nrb_judge_log10i(a):
    """log10 of a positive interval (outward)."""
    if a is None or not a.lo > 0.0:
        return None
    return I(dn(dn(math.log10(a.lo))), math.log10(a.mid), up(up(math.log10(a.hi))))


def det_nrb_judge_exports_lines(path):
    """ordered (key, value, raw line) of a resolved exports file; non-declare lines are kept as ('', '', raw)."""
    out = []
    for ln in Path(path).read_text().split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("declare -x ") and "=" in s:
            k, v = s[len("declare -x "):].split("=", 1)
            out.append((k, v.strip('"'), ln))
        else:
            out.append(("", "", ln))
    return out


def det_nrb_judge_footer(run):
    p = Path(run) / "RUN_FOOTER.txt"
    if not p.exists():
        return None
    d = {}
    for ln in p.read_text().split("\n"):
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k] = v
    return d


def det_nrb_judge_kv(path):
    d = {}
    if Path(path).exists():
        for ln in Path(path).read_text().split("\n"):
            if "=" in ln:
                k, v = ln.split("=", 1)
                d[k.strip()] = v.strip()
    return d


# ------------------------------------------------------------------ run loading (generation-generic)
def det_nrb_judge_markers(stderr_text, n_shells):
    """[BB-NRB] activation flags: INIT (operator 1, armed 0) and one armed line per ordinary iteration."""
    init = re.findall(r"^\[BB-NRB\] iter=0 operator=1 armed=0 captured=\d n_shells=(\d+) skip_z=(\S+)$", stderr_text, re.M)
    armed = re.findall(r"^\[BB-NRB\] iter=(\d+) operator=2 armed=1 captured=1 n_shells=(\d+) skip_z=(\S+)$",
                       stderr_text, re.M)
    return {"init": [(int(a), b) for a, b in init], "armed_iters": [int(a) for a, b, c in armed],
            "armed_n_shells": sorted({int(b) for a, b, c in armed}), "armed_skip_z": sorted({c for a, b, c in armed}),
            "n_shells_expected": n_shells}


def det_nrb_judge_run(run, tag, keys, src_text, deck, outer, fault=None, cuda_text="", reference_binary_sha=None):
    """stored run of `outer` iterations: exports, NRB form, activation markers, capture authority at g_obs, W5
    history of the requested keys, generation-labelled dumps, solve units, collision authority."""
    run = Path(run)
    G = det_nrb_judge_gens(outer)
    why = []
    ex = RATES.det_siiii_rates_exports(run)
    capd = run / "work/det_obs_capture"
    meta = POP.read_meta(capd) if (capd / "meta.txt").exists() else {}
    t_exp = float.fromhex(meta["time_explosion"]) if "time_explosion" in meta else math.nan
    g_obs = 7 if FT(fault, "capture_fixed7") else G["g_obs"]
    if str(meta.get("material_generation")) != str(g_obs) or str(meta.get("rad_generation")) != str(g_obs):
        why.append(f"{tag}:LINK_CAPTURE_META_GENERATION:{meta.get('material_generation')}!={g_obs}")
    man = json.loads((capd / "manifest.json").read_text()) if (capd / "manifest.json").exists() else {}
    gens = man.get("generations", {})
    for fld in ("material", "nlte_population_committed", "radiation"):
        if gens.get(fld) != g_obs:
            why.append(f"{tag}:LINK_CAPTURE_GENERATION:{fld}:{gens.get(fld)}!={g_obs}")
    if gens.get("iter") != G["iter_last"]:
        why.append(f"{tag}:LINK_CAPTURE_ITER:{gens.get('iter')}!={G['iter_last']}")
    # prereg 4.2: the tau / opacity / emissivity counters are checked on their own, never folded into one number
    if gens.get("tau_required") is None or gens.get("tau_required") != gens.get("tau_computed"):
        why.append(f"{tag}:LINK_CAPTURE_TAU_COUNTERS:{gens.get('tau_required')}!={gens.get('tau_computed')}")
    if man.get("transaction") != "COMMITTED":
        why.append(f"{tag}:LINK_CAPTURE_NOT_COMMITTED")
    if (man.get("census") or {}).get("mapped_B_fallback") != 0:
        why.append(f"{tag}:LINK_CAPTURE_SOURCE_FALLBACK")
    if (capd / "lines.bin").exists():
        why += [f"{tag}:{w}" for w in SEED.det_seedmem_scope_capture_integrity(capd)]
    else:
        why.append(f"{tag}:LINK_CAPTURE_FILE_MISSING")
    st = run / "work/det_obs_status.json"
    obs = json.loads(st.read_text()) if st.exists() else {}
    if obs.get("det_obs_state") != "SUCCESS" or obs.get("rad_generation") != g_obs or obs.get("iter") != G["iter_last"]:
        why.append(f"{tag}:LINK_OBS_STATUS:{obs}")
    nrb = ex.get("LUMINA_BB_NRB") == "1" and ex.get("LUMINA_A210_SPRODUCER_CAPTURE") == "1"
    it = SEED.det_seedmem_scope_atoi(ex.get(ITER_KEY))
    if it != outer:
        why.append(f"{tag}:LINK_ITERATIONS:{it}!={outer}")
    hist, contra = REP.det_siiii_repair_w5_load(run, keys) if keys else ({}, [])
    if contra:
        why.append(f"{tag}:W5_CONTRADICTORY_DUPLICATE:{len(contra)}")
    dumps = SEED.det_seedmem_scope_dumps(run)
    if FT(fault, "last_as_observed") and G["g_last"] in dumps["ne"]:
        for k in ("ne", "Te", "ion"):
            dumps[k][G["g_obs"]] = dumps[k][G["g_last"]]
        dumps["label"][G["g_obs"]] = G["g_last"]
    why += [f"{tag}:{w}" for w in dumps["why"] + SEED.det_seedmem_scope_dumps_check(dumps)]
    if sorted(dumps["ne"]) != G["pops"]:
        why.append(f"{tag}:DUMP_GENERATIONS:{sorted(dumps['ne'])}!={G['pops']}")
    cne = capd / "ne.f64"
    if G["g_obs"] not in dumps["ne"] or not cne.exists() or dumps["ne"][G["g_obs"]].tobytes() != cne.read_bytes():
        why.append(f"{tag}:LINK_NE_DUMP_OBS_NOT_CAPTURE")
    so = (run / "stdout.log").read_text(errors="replace") if (run / "stdout.log").exists() else ""
    se = (run / "stderr.log").read_text(errors="replace") if (run / "stderr.log").exists() else ""
    n_shells = len(dumps["ne"][G["g_obs"]]) if G["g_obs"] in dumps["ne"] else -1
    mk = det_nrb_judge_markers(se, n_shells)
    if nrb:
        if mk["armed_iters"] != list(range(outer)) or mk["armed_n_shells"] != [n_shells] or mk["armed_skip_z"] != ["none"]:
            why.append(f"{tag}:NRB_ACTIVATION_MARKERS:{mk['armed_iters']}")
        if not mk["init"]:
            why.append(f"{tag}:NRB_INIT_MARKER_MISSING")
    if "[DET-TRANSACTIONAL] status=COMPLETE" not in so:
        why.append(f"{tag}:DET_TERMINAL_MISSING")
    m_it = re.search(r"pure deterministic radiation driver: (\d+) shells, .*?, (\d+) outer iters", so)
    if m_it is None or int(m_it.group(2)) != outer:
        why.append(f"{tag}:STDOUT_OUTER_ITERS:{None if m_it is None else m_it.group(2)}!={outer}")
    sl = SEED.det_seedmem_scope_sl(deck, ex, so, TARGET["ion"], fault)
    why += [f"{tag}:{w}" for w in sl["why"]]
    coll = SEED.det_seedmem_scope_coll_authority(ex, src_text, fault, run, cuda_text)
    why += [f"{tag}:{w}" for w in coll["why"]]
    foot = det_nrb_judge_footer(run)
    ident = det_nrb_judge_kv(run / "STAGE_IDENTITY.txt")
    bsha = (run / "input/binary.sha256").read_text().split()[0] if (run / "input/binary.sha256").exists() else None
    actual = sha(run / "input/lumina_cuda") if (run / "input/lumina_cuda").exists() else None
    binary_link = {"stage_identity": ident.get("run_binary_sha256"), "binary_sha256_file": bsha, "actual": actual,
                   "footer": None if foot is None else foot.get("binary_sha256"),
                   "reference": reference_binary_sha}
    vals = {v for k, v in binary_link.items() if v is not None}
    if len(vals) != 1:
        why.append(f"{tag}:LINK_BINARY_IDENTITY:{binary_link}")
    if foot is not None:
        if foot.get("terminal_state") != "SUCCESS":
            why.append(f"{tag}:RUN_NOT_SUCCESS:{foot.get('terminal_state')}")
        if str(foot.get("outer_iterations")) != str(outer):
            why.append(f"{tag}:FOOTER_ITERATIONS:{foot.get('outer_iterations')}!={outer}")
    else:
        why.append(f"{tag}:RUN_FOOTER_MISSING")
    counters = {"capture_tau_required": gens.get("tau_required"), "capture_tau_computed": gens.get("tau_computed"),
                "capture_material": gens.get("material"), "capture_radiation": gens.get("radiation")}
    for m in sorted((run / "work/physics_comparison").glob("physics_DET_iter*.manifest.json")):
        js = json.loads(m.read_text())
        if js.get("population_generation") == G["g_obs"]:
            counters["obs_dump_opacity_generation"] = js.get("opacity_generation")
            counters["obs_dump_emissivity_generation"] = js.get("emissivity_generation")
            counters["obs_dump_radiation_generation"] = js.get("radiation_generation")
    return {"run": str(run), "tag": tag, "outer": outer, "G": G, "t_exp": t_exp, "exports": ex, "nrb": nrb,
            "counters": counters,
            "form": "NRB" if nrb else "OFF", "hist": hist, "dumps": dumps, "sl": sl, "coll": coll, "markers": mk,
            "capture_manifest": man, "obs_status": obs, "footer": foot, "binary_link": binary_link,
            "n_shells": n_shells, "why": why}


def det_nrb_judge_input_identity(n_run):
    """prereg 2.2 / 3.2 (3): every staged input of N has the content the preflight recorded for the R' input
    (reference_inputs.json next to the preflight offline.json the stage identity names), the file set is the same,
    and the effective exports are the ones the stage identity recorded."""
    n_run = Path(n_run)
    ident = det_nrb_judge_kv(n_run / "STAGE_IDENTITY.txt")
    out = {"why": [], "checked": 0}
    pre = ident.get("preflight")
    ref_p = Path(pre).parent / "reference_inputs.json" if pre else None
    if ref_p is None or not ref_p.exists():
        out["why"].append("INPUT_IDENTITY_REFERENCE_ABSENT")
        return out
    ref = json.loads(ref_p.read_text())
    out["reference"] = {"path": str(ref_p), "sha256": sha(ref_p)}
    have = {str(q.relative_to(n_run / "input")) for d in STAGED_DIRS for q in (n_run / "input" / d).rglob("*") if q.is_file()}
    have |= {f for f in STAGED_FILES if (n_run / "input" / f).is_file()}
    if have != set(ref):
        out["why"].append(f"INPUT_SET_DIFFERS:{sorted(have ^ set(ref))[:5]}")
    for rel, v in sorted(ref.items()):
        q = n_run / "input" / rel
        out["checked"] += 1
        if v is None or not q.is_file() or sha(q) != v["sha256"]:
            out["why"].append(f"INPUT_IDENTITY:{rel}")
    ex = n_run / "input/resolved_lumina.exports"
    if not ex.exists() or sha(ex) != ident.get("effective_exports_sha256"):
        out["why"].append("INPUT_IDENTITY:resolved_lumina.exports_vs_stage_identity")
    out["stage_runner_sha256"] = ident.get("runner_sha256")
    out["job_slurm_sha256"] = sha(n_run / "job.slurm") if (n_run / "job.slurm").exists() else None
    return out


def det_nrb_judge_row(rd, line, shell, call):
    return rd["hist"].get((line, shell), {}).get(call)


def det_nrb_judge_ion_dump(run, gen, shells):
    """generation-labelled ion dump rows of the requested shells: element totals, stage and NLTE sums (Z 14, 16)."""
    d = Path(run) / "work/physics_comparison"
    for m in sorted(d.glob("physics_DET_iter*.manifest.json")):
        js = json.loads(m.read_text())
        if js.get("population_generation") != gen:
            continue
        io = m.with_name(m.name.replace(".manifest.json", ".ion.csv"))
        x = pd.read_csv(io, float_precision="round_trip").set_index("shell_id")
        out = {}
        for s in shells:
            if s not in x.index:
                continue
            r = x.loc[s]
            if int(r["population_generation"]) != gen:
                return None, f"ION_DUMP_GENERATION_MISMATCH:{m.name}"
            out[s] = {f"Z{Z}_{c}": float(r[f"Z{Z}_{c}"]) for Z in TARGET["Z"]
                      for c in ("element_total_cm3", "stage_q1_cm3", "stage_q2_cm3", "nlte_q1_cm3", "nlte_q2_cm3")}
        return {"manifest": str(m), "file": str(io), "generation": gen, "rows": out}, None
    return None, f"ION_DUMP_GENERATION_ABSENT:{gen}"


# ------------------------------------------------------------------ P1(a): applied NRB and the local recall
def det_nrb_judge_p1a(rd, keys, deck, lib, fault=None):
    """every registered cell (key x ordinary call): row admissibility, deck identity, the production bracket on the
    stored ledger, the branch it returned and the reconstructed rates (prereg 5.1)."""
    rows, states = [], []
    G = rd["G"]
    for Z, line, s in keys:
        a = deck["at"].get(line)
        for k in G["calls"]:
            row = det_nrb_judge_row(rd, line, s, k)
            why = SEED.det_seedmem_scope_row_check(row, k, a, Z) if a is not None else ["DECK_LINE_ABSENT"]
            rec = {"run": rd["tag"], "Z": Z, "line": line, "shell": s, "call": k, "consuming_material_generation": k + 1,
                   "material_generation_of_row": None if row is None else int(row["material_generation"]),
                   "operator": None if row is None else row["operator"]}
            br, rc_fn, rates = None, None, None
            if row is not None and not why:
                rates, br = SEED.det_seedmem_scope_rates(row, a, "NRB", lib, fault)
                jc, beta, prov = float(row["J_c"]), float(row["beta"]), row["srce_chk"]
                prov_i = SEED.det_seedmem_scope_atoi(float(prov)) if math.isfinite(float(prov)) else 255
                rc_fn, b2, Ra, Rs, Rp = lib.bracket(beta * jc, float(row["L_l"]), float(row["tau_eff"]), float(row["Jbar"]),
                                                    prov_i, a["B_lu"], a["B_ul"], a["A_ul"])
                rec.update({"c_continuum": beta * jc, "L_local": float(row["L_l"]), "Jbar": float(row["Jbar"]),
                            "tau_eff": float(row["tau_eff"]), "beta": beta, "provenance": prov_i, "bracket_rc": rc_fn,
                            "R_absorb": Ra if rc_fn == 0 else None, "R_stim": Rs if rc_fn == 0 else None,
                            "R_spont": Rp if rc_fn == 0 else None})
                if rates is not None:
                    for nm, iv in zip(("R_absorb_i", "R_stim_i", "R_spont_i"), rates):
                        REP.det_siiii_repair_cols(rec, nm, iv)
            if why and SEED.det_seedmem_scope_is_link(why):
                st = "X"
            elif row is None or why:
                st = "NOT_IDENTIFIABLE"
            elif br == "NRB_BRACKET":
                st = "SURVIVES"
            elif br == "BASE_SRCE_CHK":
                st = "REFUTED"                       # the production kept the base rates: NRB not applied here
                why.append("NORMAL_EXCEPTION_SRCE_CHK")
            elif br == "A_ONLY_OUTSIDE_BB_DOMAIN":
                st = "NOT_IDENTIFIABLE"
                why.append(br)
            else:
                st = "X"
                why.append(br or "RATE_BRANCH_ABSENT")
            if FT(fault, "status_flip_all") and st != "X":
                st = "REFUTED"                       # injected: every stored state replaced
            rec["branch"], rec["status"], rec["reasons"] = br, st, ";".join(why)
            rows.append(rec)
            states.append(st)
    return rows, states


def det_nrb_judge_recall(rd, deck, lib, control, fault=None):
    """prereg 5.1 fixed sub-sample: line `control` x W5 x calls 2..g_obs, the production function called again with
    c, tau, B, A fixed and only L (and J-bar = c + L) changed to a different finite value; bit identity of the rates."""
    rows = []
    G = rd["G"]
    a = deck["at"].get(control)
    for s in TARGET_SHELLS(rd):
        for k in G["calls"]:
            row = det_nrb_judge_row(rd, control, s, k)
            why = SEED.det_seedmem_scope_row_check(row, k, a, a["Z"]) if a is not None else ["DECK_LINE_ABSENT"]
            rec = {"run": rd["tag"], "line": control, "shell": s, "call": k, "recalled": 0}
            if row is None or why:
                rec.update({"status": "X" if SEED.det_seedmem_scope_is_link(why) else "NOT_IDENTIFIABLE",
                            "reasons": ";".join(why or ["W5_CALL_MISSING"])})
                rows.append(rec)
                continue
            jc, beta, L, tau, jb = (float(row[c]) for c in ("J_c", "beta", "L_l", "tau_eff", "Jbar"))
            prov = SEED.det_seedmem_scope_atoi(float(row["srce_chk"])) if math.isfinite(float(row["srce_chk"])) else 255
            c = beta * jc
            rc1, b1, Ra1, Rs1, Rp1 = lib.bracket(c, L, tau, jb, prov, a["B_lu"], a["B_ul"], a["A_ul"])
            L2 = L * 1.5 if L > 0.0 else 1.0e-20
            c2 = c * 1.5 if FT(fault, "move_c_with_L") else c
            rc2, b2, Ra2, Rs2, Rp2 = lib.bracket(c2, L2, tau, c2 + L2, prov, a["B_lu"], a["B_ul"], a["A_ul"])
            rec.update({"recalled": 1, "c_continuum": c, "c_recalled": c2, "tau_eff": tau, "L_original": L,
                        "L_recalled": L2, "Jbar_original": jb, "Jbar_recalled": c2 + L2, "rc_original": rc1,
                        "rc_recalled": rc2, "R_absorb": Ra1, "R_stim": Rs1, "R_spont": Rp1, "R_absorb_recalled": Ra2,
                        "R_stim_recalled": Rs2, "R_spont_recalled": Rp2,
                        "inputs_differ": int(L2 != L and math.isfinite(L2))})
            if rc1 != 0:
                st = "NOT_IDENTIFIABLE"
                rec["reasons"] = f"BRACKET_RC_{rc1}"
            elif rc2 != 0:
                st, rec["reasons"] = "X", f"RECALL_RC_{rc2}"
            elif not rec["inputs_differ"]:
                st, rec["reasons"] = "X", "RECALL_INPUT_NOT_DIFFERENT"
            elif (Ra1, Rs1, Rp1) == (Ra2, Rs2, Rp2):
                st, rec["reasons"] = "SURVIVES", ""
            else:
                st, rec["reasons"] = "REFUTED", "RATES_DEPEND_ON_LOCAL_TERM"
            rec["status"] = st
            rows.append(rec)
    return rows


def TARGET_SHELLS(rd):
    return rd["shells"]


def det_nrb_judge_inheritance(lib_rec, prior_analysis, prior_selftest):
    """H-LOCAL inheritance (prereg 5.1): the CPU library sources of this readout carry the same SHA-256 as the SEEDMEM
    r4 manifest, A/result.json records H-LOCAL 0 over 2550 cells, and the r4 selftest carries the passing cross bundle."""
    A = Path(prior_analysis)
    man_p = A.parent / "manifest.json"
    out = {"prior_manifest": str(man_p), "prior_result": str(A / "result.json"), "prior_selftest": str(prior_selftest),
           "why": []}
    try:
        man = json.loads(man_p.read_text())
        res = json.loads((A / "result.json").read_text())
        st = json.loads((Path(prior_selftest) / "selftest.json").read_text())
    except (OSError, ValueError) as e:
        out["why"].append(f"INHERITANCE_FILES:{e}")
        return out
    prior = (man.get("lib") or {}).get("sources_sha256") or {}
    mine = lib_rec.get("sources_sha256") or {}
    out["sources_sha256_prior"], out["sources_sha256_mine"] = prior, mine
    for f in SEED.LIB_SOURCES + SEED.LIB_HEADERS:
        if not prior.get(f) or prior.get(f) != mine.get(f):
            out["why"].append(f"INHERITANCE_SOURCE_SHA:{f}")
    rec = (res.get("codes") or {}).get("H-LOCAL_record") or {}
    out["prior_H_LOCAL"] = rec
    if rec.get("code") != 0 or rec.get("cells") != 2550 or rec.get("x_cells") != 0 or res.get("validity") != "VALID":
        out["why"].append("INHERITANCE_PRIOR_H_LOCAL_NOT_0_2550")
    cross = [c for c in st.get("cases", []) if c.get("bundle") == "7 cross"]
    out["prior_cross_cases"] = [{"case": c["case"], "pass": c["pass"]} for c in cross]
    if not cross or not all(c.get("pass") for c in cross) or st.get("failures") != 0:
        out["why"].append("INHERITANCE_PRIOR_SELFTEST_CROSS")
    out["prior_selftest_readout_sha256"] = st.get("readout_sha256")
    out["inherited"] = not out["why"]
    return out


# ------------------------------------------------------------------ P1(b): the 1206 source
def det_nrb_judge_p1b(rd, deck, csob, lib, control, fault=None):
    """prereg 5.2: d_k = S_meas(k) / S_pred(x(call k-1 radiation, generation-k n_e, T_e)) - 1 on the control line,
    core calls 3-7 and extension calls 8..g_obs; |d| <= 1e-3 on the whole interval."""
    rows, states = [], {"core": [], "extension": []}
    G = rd["G"]
    a = deck["at"][control]
    r_iv = deck["r"][control]
    for s in rd["shells"]:
        for k in G["p1b_core"] + G["p1b_ext"]:
            rng = "core" if k in G["p1b_core"] else "extension"
            rk, rp = det_nrb_judge_row(rd, control, s, k), det_nrb_judge_row(rd, control, s, k - 1)
            if FT(fault, "beta_current"):
                rp = rk
            why = SEED.det_seedmem_scope_row_check(rk, k, a, a["Z"]) + SEED.det_seedmem_scope_row_check(rp, k - 1, a, a["Z"])
            gk = k + 1 if FT(fault, "material_plus1") else k
            if gk in rd["dumps"]["label"] and rd["dumps"]["label"][gk] != k:
                why.append("MATERIAL_GENERATION_MISMATCH")
            ne = rd["dumps"]["ne"].get(gk)
            te = rd["dumps"]["Te"].get(gk)
            rec = {"run": rd["tag"], "range": rng, "line": control, "shell": s, "call": k, "material_generation": k,
                   "material_generation_used": rd["dumps"]["label"].get(gk), "radiation_call": k - 1,
                   "collision_branch": rd["coll"]["branch"]}
            meas = pred = None
            if not why:
                meas = SEED.det_seedmem_scope_meas(rk, a, r_iv, fault)
                rad, br = SEED.det_seedmem_scope_rates(rp, a, rd["form"], lib, fault)
                rec["rate_branch"] = br
                coll = None
                if ne is not None and te is not None and rd["coll"]["supported"]:
                    coll = SEED.det_seedmem_scope_coll(float(ne[s]), float(te[s]), a)
                pred = SEED.det_seedmem_scope_source2(rad, coll, a)
                if coll is None:
                    why.append("COLLISION_BRANCH_UNSUPPORTED" if not rd["coll"]["supported"] else "NE_TE_ABSENT")
                elif rad is None:
                    why.append(br)
                elif pred is None or meas is None:
                    why.append("SOURCE_UNDEFINED")
                rec["source_population_identity"] = SEED.det_seedmem_scope_identity(rk, a, meas, rd["t_exp"], csob, lib)
                if rec["source_population_identity"] == "MISMATCH":
                    why.append("SOURCE_POPULATION_IDENTITY")
                rec["n_e"], rec["T_e"] = (None if ne is None else float(ne[s])), (None if te is None else float(te[s]))
            d = None
            if meas is not None and pred is not None:
                q = meas.div(pred)
                d = None if q is None else q - I.x(1.0)
            st = det_nrb_judge_state(None if d is None else d.abs(), CRIT, fault)
            if SEED.det_seedmem_scope_is_link(why):
                st = "X"
            REP.det_siiii_repair_cols(rec, "S_meas", meas)
            REP.det_siiii_repair_cols(rec, "S_pred", pred)
            REP.det_siiii_repair_cols(rec, "d", d)
            rec["status"], rec["reasons"] = st, ";".join(why)
            rows.append(rec)
            states[rng].append(st)
    return rows, states


# ------------------------------------------------------------------ level authority and P2
def det_nrb_judge_capture_check(rd, keys, deck, csob, lib, fault=None):
    """W5 inversion at call g_obs vs the independent generation-g_obs capture inversion (SEEDMEM 3.2 (3) at the
    registered observation generation)."""
    rows, why, status = [], [], {}
    g = rd["G"]["g_obs"]
    capd = Path(rd["run"]) / "work/det_obs_capture"
    recs = POP.capture_records(capd, sorted({k[1] for k in keys}), set(rd["shells"]) | {k[2] for k in keys}) \
        if (capd / "lines.bin").exists() else {}
    for Z, line, s in keys:
        a = deck["at"][line]
        r = det_nrb_judge_row(rd, line, s, g)
        w = SEED.det_seedmem_scope_row_check(r, g, a, Z)
        inv, rsn = (None, w[0]) if w else SEED.det_seedmem_scope_invert(r, a, rd["t_exp"], csob, lib, fault)
        c = recs.get((line, s))
        cap = None
        if c is not None and c["mapped"] == 1:
            cap = POP.recover(c["tau"], c["S"], a["g_l"], a["g_u"], a["f_lu"], a["lam_cm"], rd["t_exp"])
        ok = None
        if inv is not None and cap is not None:
            ok = SEED.det_seedmem_scope_overlap(inv["n_u"], cap["n_u"]) and SEED.det_seedmem_scope_overlap(inv["n_l"], cap["n_l"])
            if not ok:
                why.append(f"{rd['tag']}:POP_AUTHORITY_MISMATCH:{line}:{s}")
        capst = "MISSING" if c is None else ("UNMAPPED" if c["mapped"] != 1 else "OK")
        if rsn == "FORWARD_LEDGER_MISMATCH":
            why.append(f"{rd['tag']}:FORWARD_LEDGER_MISMATCH:{line}:{s}")
        status[(line, s)] = ("OK" if ok else "INCOMPATIBLE") if ok is not None else (capst if capst != "OK" else f"W5_{rsn}")
        rows.append({"run": rd["tag"], "generation": g, "line": line, "shell": s, "w5_inversion": rsn or "OK",
                     "capture": capst, "compatible": None if ok is None else int(ok), "authority": status[(line, s)]})
    return rows, why, status


def det_nrb_judge_series(rd, keys, deck, csob, lib, cap_status, fault=None):
    """per-axis quantity series over the registered generations (ion, n_e: matter generations; J-bar, level: radiation
    calls = material generations of the W5 rows)."""
    G, D = rd["G"], rd["dumps"]
    shells = rd["shells"]
    levels = sorted({(Z, deck["at"][line][w], s) for Z, line, s in keys for w in ("lo", "up")})
    series = {"ion": {}, "ne": {}, "jbar": {}, "level": {}}
    for s in shells:
        for Z, q in ION_QTY:
            series["ion"][(Z, q, s)] = {g: (float(D["ion"][g][(Z, q)][s]) if g in D["ion"] else None) for g in G["matter_gens"]}
        series["ne"][s] = {g: (float(D["ne"][g][s]) if g in D["ne"] else None) for g in G["matter_gens"]}
    rec_pop = {}
    for Z, line, s in keys:
        a = deck["at"][line]
        ser = {}
        for g in G["rad_gens"]:
            r = det_nrb_judge_row(rd, line, s, g)
            if r is None or SEED.det_seedmem_scope_row_check(r, g, a, Z):
                ser[g] = None
                continue
            jb, je = float(r["Jbar"]), float(r["Jbar_err_upper"])
            ser[g] = I(dn(jb - je), jb, up(jb + je))
            if g == G["g_obs"] and cap_status.get((line, s)) != "OK":
                continue
            inv, _ = SEED.det_seedmem_scope_invert(r, a, rd["t_exp"], csob, lib, fault)
            if inv is None:
                continue
            rec_pop.setdefault((Z, a["lo"], s, g), []).append(inv["n_l"])
            rec_pop.setdefault((Z, a["up"], s, g), []).append(inv["n_u"])
        series["jbar"][(line, s)] = ser
    contradictions = []
    for Z, lev, s in levels:
        ser = {}
        for g in G["rad_gens"]:
            v = rec_pop.get((Z, lev, s, g))
            if not v:
                ser[g] = None
            elif not all(SEED.det_seedmem_scope_overlap(x, y) for x in v for y in v):
                # prereg 5.3: admissible recoveries of one level that contradict each other make the range X
                ser[g] = None
                contradictions.append((Z, lev, s, g, len(v)))
            else:
                ser[g] = I(max(x.lo for x in v), v[0].mid, min(x.hi for x in v))
        series["level"][(Z, lev, s)] = ser
    expected = {"ion": len(ION_QTY) * len(shells), "ne": len(shells), "jbar": len(keys), "level": len(levels)}
    return series, expected, levels, contradictions


def det_nrb_judge_change(va, vb, fault=None):
    """|log10(b/a)| interval of one quantity between adjacent generations; ('zero', 'missing', 'cross', m)."""
    if va is None or vb is None:
        return "missing", None
    A_ = va if isinstance(va, I) else I.x(va)
    B_ = vb if isinstance(vb, I) else I.x(vb)
    if A_.hi == 0.0 and B_.hi == 0.0 and A_.lo == 0.0 and B_.lo == 0.0:
        return "zero", None
    if (A_.hi == 0.0) != (B_.hi == 0.0):
        if FT(fault, "drop_zero_cross"):
            return "dropped", None
        return "cross", I(math.inf, math.inf, math.inf)
    if not (A_.lo > 0 and B_.lo > 0):
        return "missing", None
    if isinstance(va, I) or isinstance(vb, I):
        r = B_.div(A_)
        lo, hi = dn(dn(math.log10(r.lo))), up(up(math.log10(r.hi)))
        m = I(max(0.0, lo) if lo > 0 else (0.0 if hi > 0 else -hi), abs(math.log10(r.mid)), max(abs(lo), abs(hi)))
    else:
        m = SEED.det_seedmem_scope_m(va, vb)
    if FT(fault, "drop_large") and m.lo >= P2_CRIT:
        return "dropped", None
    return "ok", m


def det_nrb_judge_p2(rd, keys, deck, csob, lib, cap_status, fault=None):
    """prereg 5.3: adjacent-generation maximum changes of four axes; the terminal window (outer, g_obs) decides;
    the trajectory descriptor M(4), M(g_obs), D = M(g_obs) - M(4) is recorded for every axis."""
    G = rd["G"]
    series, expected, levels, contradictions = det_nrb_judge_series(rd, keys, deck, csob, lib, cap_status, fault)
    rows, codes, per_axis = [], {}, {}
    for axis, qs in series.items():
        gens = G["matter_gens"] if axis in ("ion", "ne") else G["rad_gens"]
        per_g = {}
        for g in gens[1:]:
            ms, miss, zero, worst = [], 0, 0, None
            for qk, ser in qs.items():
                kind, m = det_nrb_judge_change(ser.get(g - 1), ser.get(g), fault)
                if kind == "missing":
                    miss += 1
                elif kind == "zero":
                    zero += 1
                elif kind == "dropped":
                    continue
                else:
                    ms.append(m)
                    if worst is None or m.mid > worst[1]:
                        worst = (qk, m.mid)
            M = I(max((x.lo for x in ms), default=0.0), max((x.mid for x in ms), default=0.0),
                  max((x.hi for x in ms), default=0.0))
            per_g[g] = {"M": M, "missing": miss, "structural_zero": zero, "n": len(ms)}
            rows.append({"run": rd["tag"], "axis": axis, "generation": g, "pair": f"{g - 1}->{g}", "M_lo": M.lo,
                         "M_mid": M.mid, "M_hi": M.hi, "n_quantities": len(ms), "missing": miss, "structural_zero": zero,
                         "registered_quantities": expected[axis], "worst_key": None if worst is None else str(worst[0]),
                         "small": det_nrb_judge_small(M, miss),
                         "terminal_window": int(g in G["terminal"]), "last_material_update": int(g == G["g_last"])})
        per_axis[axis] = per_g
        codes[axis] = det_nrb_judge_axis_code(per_g, G, fault)
        if FT(fault, "p2_code_zero") and axis == "ion":
            codes[axis]["code"], codes[axis]["state"] = 0, "SURVIVES"     # injected: final axis code replaced
    return rows, codes, expected, contradictions


def det_nrb_judge_axis_code(per_g, G, fault=None):
    """terminal-window state (both generations of (outer, g_obs) small), the descriptor and the recorded past windows."""
    t0, t1 = G["terminal"]
    st = {g: (det_nrb_judge_small(per_g[g]["M"], per_g[g]["missing"]) if g in per_g else "NOT_IDENTIFIABLE")
          for g in (t0, t1)}
    if FT(fault, "single_change"):
        term = "SURVIVES" if "SURVIVES" in st.values() else SEED.det_seedmem_scope_universal_check(list(st.values()))
    else:
        term = "SURVIVES" if list(st.values()) == ["SURVIVES", "SURVIVES"] else \
            SEED.det_seedmem_scope_universal_check(list(st.values()))
    past, after = {}, {}
    gens = sorted(per_g)
    for g1, g2 in zip(gens[:-1], gens[1:]):
        if g2 > t1:
            s1 = det_nrb_judge_small(per_g[g1]["M"], per_g[g1]["missing"])
            s2 = det_nrb_judge_small(per_g[g2]["M"], per_g[g2]["missing"])
            after[f"{g1},{g2}"] = [s1, s2]                # touches the last material update: recorded apart
            continue
        s1 = det_nrb_judge_small(per_g[g1]["M"], per_g[g1]["missing"])
        s2 = det_nrb_judge_small(per_g[g2]["M"], per_g[g2]["missing"])
        past[f"{g1},{g2}"] = "SURVIVES" if (s1, s2) == ("SURVIVES", "SURVIVES") else \
            SEED.det_seedmem_scope_universal_check([s1, s2])
    if FT(fault, "past_window") and "SURVIVES" in past.values():
        term = "SURVIVES"
    desc = det_nrb_judge_descriptor(per_g, G)
    last = per_g.get(G["g_last"])
    return {"code": "X" if term == "X" else CODE[term], "state": term, "terminal_window": f"{t0},{t1}",
            "terminal_states": {str(g): v for g, v in st.items()}, "past_windows": past,
            "windows_with_last_material_update": after,
            "any_past_window_small": any(v == "SURVIVES" for k, v in past.items() if k != f"{t0},{t1}"),
            "descriptor": desc,
            "last_material_update": None if last is None else {"generation": G["g_last"], "M": last["M"].j(),
                                                                "state": det_nrb_judge_small(last["M"], last["missing"])}}


def det_nrb_judge_descriptor(per_g, G):
    """prereg 5.3 descriptor: M(start), M(g_obs), D = M(g_obs) - M(start) as an interval -> END_* code."""
    s, e = G["start"], G["g_obs"]
    a, b = per_g.get(s), per_g.get(e)
    out = {"start_generation": s, "end_generation": e}
    for nm, v in (("M_start", a), ("M_end", b)):
        out[nm] = None if v is None else v["M"].j()
    bad = []
    for nm, v in (("start", a), ("end", b)):
        if v is None:
            bad.append(f"{nm}:ABSENT")
        elif v["missing"]:
            bad.append(f"{nm}:MISSING_{v['missing']}")
        elif not all(math.isfinite(x) for x in (v["M"].lo, v["M"].mid, v["M"].hi)):
            bad.append(f"{nm}:NONFINITE")
    if bad:
        out.update({"D": None, "code": "END_UNAVAILABLE", "reason": ";".join(bad)})
        return out
    code_, lo, hi = det_nrb_judge_end_code(a["M"].lo, a["M"].hi, b["M"].lo, b["M"].hi)
    out["D"] = {"lo": lo, "mid": b["M"].mid - a["M"].mid, "hi": hi}
    out["code"] = code_
    return out


def det_nrb_judge_p2_validate(rows, codes, expected, G):
    """registered P2 validator from the stored rows: quantity accounting per generation, the terminal states, the
    axis code and the descriptor re-derived independently of the code path."""
    why = []
    df = pd.DataFrame(rows)
    if not len(df):
        return ["P2_NO_ROWS"]
    for axis, sub in df.groupby("axis"):
        n_exp = expected[axis]
        gens = G["matter_gens"] if axis in ("ion", "ne") else G["rad_gens"]
        if sorted(int(g) for g in sub.generation) != gens[1:]:
            why.append(f"P2_GENERATION_ACCOUNTING:{axis}")
        for r in sub.itertuples():
            if int(r.n_quantities) + int(r.missing) + int(r.structural_zero) != n_exp:
                why.append(f"P2_QUANTITY_ACCOUNTING:{axis}:{r.generation}")
        st = {int(r.generation): r for r in sub.itertuples()}

        def small(r):
            if r is None:
                return "NOT_IDENTIFIABLE"
            if r.M_lo >= P2_CRIT:
                return "REFUTED"
            if int(r.missing):
                return "NOT_IDENTIFIABLE"
            if r.M_hi < P2_CRIT:
                return "SURVIVES"
            return "UNRESOLVED"
        t0, t1 = G["terminal"]
        one = [small(st.get(t0)), small(st.get(t1))]
        fin = "SURVIVES" if one == ["SURVIVES", "SURVIVES"] else SEED.det_seedmem_scope_universal_check(one)
        rep = codes.get(axis, {})
        if rep.get("state") != fin or rep.get("code") != CODE.get(fin, "X") or \
                list((rep.get("terminal_states") or {}).values()) != one:
            why.append(f"P2_CODE_RECOMPUTE:{axis}")
        a, b = st.get(G["start"]), st.get(G["g_obs"])
        if a is None or b is None or int(a.missing) or int(b.missing) or \
                not all(math.isfinite(float(x)) for x in (a.M_lo, a.M_mid, a.M_hi, b.M_lo, b.M_mid, b.M_hi)):
            dcode = "END_UNAVAILABLE"
        else:
            dcode = det_nrb_judge_end_code(float(a.M_lo), float(a.M_hi), float(b.M_lo), float(b.M_hi))[0]
        if (rep.get("descriptor") or {}).get("code") != dcode:
            why.append(f"P2_DESCRIPTOR_RECOMPUTE:{axis}")
    return why


# ------------------------------------------------------------------ public ions and spectrum
def det_nrb_judge_public_block(path, time_label, fault=None):
    """the public ion-fraction block at exactly `time_label` (no nearest-time substitution): rows of
    (vel_mid, tokens[]) with the printed tokens kept for the display-precision interval."""
    rows, t, found = [], None, False
    if FT(fault, "time_shift"):
        time_label = {"19.480": "17.710"}.get(time_label, time_label)
    for ln in open(path):
        if ln.startswith("#TIME:"):
            t = ln.split()[1]
            found = found or (t == time_label)
            continue
        if ln.startswith("#") or not ln.strip():
            continue
        if t == time_label:
            tok = ln.split()
            rows.append((float(tok[0]), tok[1:]))
    return rows if found else None


def det_nrb_judge_ions(rd, ion_dump, public, target, fault=None):
    """prereg 5.4: f_{Z,q} = nlte_q / element_total at the observation generation vs the public fractions at the
    registered nearest vel_mid; |log10 f_N / f_pub| <= 0.3 dex on the interval; III/II ratios alongside."""
    rows, states = [], {Z: [] for Z in target["Z"]}
    pub_cols = {14: ("si1", "si2"), 16: ("s1", "s2")}
    idx = {14: (1, 2), 16: (1, 2)}
    if FT(fault, "q_shift"):
        idx = {14: (0, 1), 16: (0, 1)}
    for Z in target["Z"]:
        blk = public.get(Z)
        for s in rd["shells"]:
            dr = (ion_dump or {}).get("rows", {}).get(s)
            vl = target["vmid_lumina"][s]
            if FT(fault, "space_shift"):
                vl = vl + 150.0                     # injected: the spatial match moved off the registered row
            prow = None
            if blk:
                prow = min(blk, key=lambda r: abs(r[0] - vl))
            pub_v = None if prow is None else prow[0]
            f_pub = {}
            if prow is not None:
                for q, j in zip((1, 2), idx[Z]):
                    f_pub[q] = POP.printed(prow[1][j])
            for q in (1, 2):
                rec = {"run": rd["tag"], "generation": None if ion_dump is None else ion_dump["generation"],
                       "authority": None if ion_dump is None else ion_dump["manifest"], "Z": Z, "ion": q + 1,
                       "shell": s, "v_mid_lumina": vl, "v_mid_public_registered": target["vmid_public"][s],
                       "v_mid_public_matched": pub_v, "public_column": pub_cols[Z][q - 1],
                       "public_token": None if prow is None else prow[1][idx[Z][q - 1]]}
                why = []
                fN = None
                if dr is not None:
                    num = dr[f"Z{Z}_{'stage' if FT(fault, 'saha_column') else 'nlte'}_q{q}_cm3"]
                    den = (dr[f"Z{Z}_nlte_q1_cm3"] + dr[f"Z{Z}_nlte_q2_cm3"]) if FT(fault, "renormalize") else \
                        dr[f"Z{Z}_element_total_cm3"]
                    rec.update({"n_nlte": num, "n_element_total": den})
                    if den > 0.0 and num >= 0.0 and math.isfinite(num) and math.isfinite(den):
                        fN = I.x(num).div(I.x(den))
                    else:
                        why.append("FRACTION_UNDEFINED")
                else:
                    why.append("ION_DUMP_ABSENT")
                if prow is None:
                    why.append("PUBLIC_BLOCK_ABSENT")
                elif pub_v != target["vmid_public"][s]:
                    why.append(f"PUBLIC_VMID_NOT_REGISTERED:{pub_v}")
                d = None
                if fN is not None and f_pub.get(q) is not None and f_pub[q].lo > 0.0:
                    d = det_nrb_judge_log10i(fN.div(f_pub[q]))
                REP.det_siiii_repair_cols(rec, "f_N", fN)
                REP.det_siiii_repair_cols(rec, "f_public", f_pub.get(q))
                REP.det_siiii_repair_cols(rec, "d_f", d)
                st = det_nrb_judge_state(None if d is None else d.abs(), ION_CRIT)
                if any(w.startswith("PUBLIC_VMID_NOT_REGISTERED") for w in why):
                    st = "X"
                rec["status"], rec["reasons"] = st, ";".join(why)
                rows.append(rec)
                states[Z].append(st)
            # III/II ratio (recorded, not a decision)
            if dr is not None and dr[f"Z{Z}_nlte_q1_cm3"] > 0 and dr[f"Z{Z}_nlte_q2_cm3"] > 0:
                ell = math.log10(dr[f"Z{Z}_nlte_q2_cm3"] / dr[f"Z{Z}_nlte_q1_cm3"])
                ellp = None
                if f_pub.get(1) is not None and f_pub.get(2) is not None and f_pub[1].mid > 0 and f_pub[2].mid > 0:
                    ellp = math.log10(f_pub[2].mid / f_pub[1].mid)
                rows.append({"run": rd["tag"], "generation": ion_dump["generation"], "authority": ion_dump["manifest"],
                             "Z": Z, "ion": "III/II", "shell": s, "v_mid_lumina": vl,
                             "v_mid_public_registered": target["vmid_public"][s], "v_mid_public_matched": pub_v,
                             "ell_N": ell, "ell_public": ellp, "delta_ell": None if ellp is None else ell - ellp,
                             "status": "RECORD", "reasons": ""})
    return rows, states


def det_nrb_judge_ions_validate(rows, run_dir, gens, target, public_files, fault=None):
    """independent recomputation of the stored fractions from the registered columns of the dump file the row names,
    of the public tokens at the registered time and vel_mid, and of the state from the stored d interval."""
    why = []
    cache = {}
    for r in rows:
        if r.get("status") == "RECORD" or r.get("run") != "N":
            continue
        g = r.get("generation")
        if g != gens["g_obs"]:
            why.append(f"ION_GENERATION:{r['Z']}:{r['shell']}:{g}")
            continue
        if r.get("authority") not in cache:
            m = Path(r["authority"])
            js = json.loads(m.read_text())
            x = pd.read_csv(m.with_name(m.name.replace(".manifest.json", ".ion.csv")), float_precision="round_trip")
            cache[r["authority"]] = (js, x.set_index("shell_id"))
        js, x = cache[r["authority"]]
        if js.get("population_generation") != gens["g_obs"]:
            why.append(f"ION_AUTHORITY_GENERATION:{r['Z']}:{r['shell']}")
        q = r["ion"] - 1
        row = x.loc[r["shell"]]
        num, den = float(row[f"Z{r['Z']}_nlte_q{q}_cm3"]), float(row[f"Z{r['Z']}_element_total_cm3"])
        if r.get("n_nlte") != num or r.get("n_element_total") != den:
            why.append(f"ION_COLUMN_RECOMPUTE:{r['Z']}:{q}:{r['shell']}")
        if r.get("v_mid_public_matched") != target["vmid_public"][r["shell"]]:
            why.append(f"ION_VMID_RECOMPUTE:{r['Z']}:{r['shell']}")
        blk = det_nrb_judge_public_block(public_files[r["Z"]], target["public_time"])
        tokc = None
        if blk:
            prow = min(blk, key=lambda t: abs(t[0] - target["vmid_public"][r["shell"]]))
            if prow[0] == target["vmid_public"][r["shell"]]:
                tokc = prow[1][q]
        if tokc != r.get("public_token"):
            why.append(f"ION_PUBLIC_TOKEN_RECOMPUTE:{r['Z']}:{q}:{r['shell']}")
        lo, hi = r.get("d_f_lo"), r.get("d_f_hi")
        if r["status"] != "X":
            exp = det_nrb_judge_state(None if lo is None else I(lo, r.get("d_f_mid"), hi).abs(), ION_CRIT)
            if exp != r["status"]:
                why.append(f"ION_STATUS_RECOMPUTE:{r['Z']}:{q}:{r['shell']}")
    return why


def det_nrb_judge_spectrum(specs, targets, fault=None):
    """prereg 5.5: BT-SEED comparison (REP) with the registered interval targets; per-condition and overall states."""
    rows, summ = REP.det_siiii_repair_spectrum_table(specs)
    iv = lambda v: None if v is None else I(v["lo"], v["mid"], v["hi"])                     # noqa: E731
    tv = lambda v: I.x(float(v)) if not isinstance(v, dict) else I(v["lo"], v["mid"], v["hi"])  # noqa: E731
    conds = {}
    sN = summ.get("N", {})
    if sN.get("status") == "OK":
        tN, tU, tR = tv(targets["abs_d_NUV_max"]), tv(targets["abs_d_UV2500_max"]), tv(targets["R8_below"])
        if FT(fault, "retarget"):
            tN, tU, tR = iv(sN["d_NUV"]).abs(), iv(sN["d_UV2500"]).abs(), iv(sN["R8"]) + I.x(1e-9)
        dN, dU, r8 = iv(sN["d_NUV"]), iv(sN["d_UV2500"]), iv(sN["R8"])
        conds = {"NUV": REP.det_siiii_repair_le(None if dN is None else dN.abs(), tN),
                 "UV2500": REP.det_siiii_repair_le(None if dU is None else dU.abs(), tU)}
        for k_, v in sN["d_keep4"].items():
            conds[k_] = REP.det_siiii_repair_le(None if v is None else iv(v).abs(), tv(targets["abs_d_keep4_max"]))
        conds["R8"] = REP.det_siiii_repair_lt(r8, tR)
        use = {k: v for k, v in conds.items() if k in ("NUV", "UV2500")} if FT(fault, "subset_bands") else conds
        overall = SEED.det_seedmem_scope_universal_check(list(use.values()))
    else:
        overall = "NOT_IDENTIFIABLE"
    if FT(fault, "hide_opt") and isinstance(summ.get("N"), dict):
        summ["N"].pop("L_OPT", None)
        summ["N"].pop("L_OPT_over_public", None)
    return rows, summ, conds, overall


def det_nrb_judge_spectrum_validate(summ, conds, overall, targets):
    why = []
    sN = summ.get("N", {})
    if sN.get("status") != "OK":
        # closeout round 5: a not-identifiable spectrum keeps its reason (the summary status) and an empty condition set
        if overall != "NOT_IDENTIFIABLE":
            why.append("SPECTRUM_OVERALL_RECOMPUTE")
        if conds:
            why.append("SPECTRUM_CONDITIONS_WITHOUT_SUMMARY")
        if sN.get("status") is None and summ.get("status") is None:
            why.append("SPECTRUM_NOT_IDENTIFIABLE_REASON_ABSENT")
        return why
    iv = lambda v: None if v is None else I(v["lo"], v["mid"], v["hi"])                     # noqa: E731
    tv = lambda v: I.x(float(v)) if not isinstance(v, dict) else I(v["lo"], v["mid"], v["hi"])  # noqa: E731
    exp = {"NUV": REP.det_siiii_repair_le(iv(sN["d_NUV"]).abs() if sN.get("d_NUV") else None, tv(targets["abs_d_NUV_max"])),
           "UV2500": REP.det_siiii_repair_le(iv(sN["d_UV2500"]).abs() if sN.get("d_UV2500") else None,
                                             tv(targets["abs_d_UV2500_max"]))}
    for k_, v in sN["d_keep4"].items():
        exp[k_] = REP.det_siiii_repair_le(None if v is None else iv(v).abs(), tv(targets["abs_d_keep4_max"]))
    exp["R8"] = REP.det_siiii_repair_lt(iv(sN["R8"]) if sN.get("R8") else None, tv(targets["R8_below"]))
    for k, v in exp.items():
        if conds.get(k) != v:
            why.append(f"SPECTRUM_CONDITION_RECOMPUTE:{k}")
    if SEED.det_seedmem_scope_universal_check(list(exp.values())) != overall:
        why.append("SPECTRUM_OVERALL_RECOMPUTE")
    if "L_OPT" not in sN or sN.get("L_OPT_over_public") is None:
        why.append("SPECTRUM_OPT_CHANGE_NOT_RECORDED")
    return why


def det_nrb_judge_targets(expectations):
    js = json.loads(Path(expectations).read_text())
    t = ((js.get("B-T") or {}).get("secondary_spectral_expectation") or {}).get("targets")
    if not t or any(k not in t for k in ("abs_d_NUV_max", "abs_d_UV2500_max", "abs_d_keep4_max", "R8_below")):
        return None
    return t


# ------------------------------------------------------------------ byte comparison (prereg 3.4)
def det_nrb_judge_bytes(n_run, r_run, outer_ref=6):
    """N vs R' stored material dumps of generations 3..7 (iter0000..0004 shell / ion csv): raw bytes, linked by the
    manifests' population / radiation generations first."""
    out = []
    for k in range(0, 5):
        g = k + 3
        rec = {"iteration": k, "population_generation": g, "radiation_generation": g - 1, "files": {}}
        mans = {}
        for tag, run in (("N", n_run), ("R", r_run)):
            m = Path(run) / f"work/physics_comparison/physics_DET_iter{k:04d}.manifest.json"
            mans[tag] = json.loads(m.read_text()) if m.exists() else None
        link = all(mans[t] and mans[t].get("population_generation") == g and mans[t].get("radiation_generation") == g - 1
                   and mans[t].get("transaction_status") == "COMMITTED" for t in ("N", "R"))
        rec["generation_link"] = "OK" if link else "MISMATCH"
        for ext in ("shell", "ion"):
            f = {}
            for tag, run in (("N", n_run), ("R", r_run)):
                p = Path(run) / f"work/physics_comparison/physics_DET_iter{k:04d}.{ext}.csv"
                f[tag] = {"path": str(p), "sha256": sha(p) if p.exists() else None,
                          "bytes": p.stat().st_size if p.exists() else None}
            f["identical"] = (f["N"]["sha256"] is not None and f["N"]["sha256"] == f["R"]["sha256"] and link)
            f["result"] = "IDENTICAL" if f["identical"] else ("NOT_COMPARABLE" if not link or f["N"]["sha256"] is None
                                                             or f["R"]["sha256"] is None else "DIFFERENT")
            rec["files"][ext] = f
        out.append(rec)
    summary = {"generations_1_2": "NOT_COMPARABLE (no stored material dump)",
               "identical": sum(1 for r in out for f in r["files"].values() if f["result"] == "IDENTICAL"),
               "different": sum(1 for r in out for f in r["files"].values() if f["result"] == "DIFFERENT"),
               "not_comparable": sum(1 for r in out for f in r["files"].values() if f["result"] == "NOT_COMPARABLE"),
               "consumption": "identical -> evidence for the stored-state reproduction and the single-change link; "
                              "different -> no byte-reproduction claim, N read on its own data; neither changes a "
                              "comparison line or a science code"}
    return {"per_iteration": out, "summary": summary}


# ------------------------------------------------------------------ validation and decision
def det_nrb_judge_validate(p1a_rows, recall_rows, p1b_rows, keys, G, outer):
    why = []
    want = {(Z, l, s, k) for Z, l, s in keys for k in G["calls"]}
    got = [(r["Z"], r["line"], r["shell"], r["call"]) for r in p1a_rows]
    if len(got) != len(want) or set(got) != want:
        why.append(f"P1A_ROW_ACCOUNTING:{len(got)}/{len(want)}")
    for r in p1a_rows:
        if r["status"] == "X":
            continue
        exp = ("SURVIVES" if r.get("branch") == "NRB_BRACKET" else "REFUTED" if r.get("branch") == "BASE_SRCE_CHK"
               else "NOT_IDENTIFIABLE")
        if r.get("branch") == "NRB_BRACKET" and r.get("bracket_rc") != 0:
            exp = "X"
        if exp != r["status"]:
            why.append(f"P1A_STATUS_RECOMPUTE:{r['line']}:{r['shell']}:{r['call']}:{r['status']}!={exp}")
    n_rec = len(TARGET_SHELLS_OF(keys)) * len(G["calls"])
    if len(recall_rows) != n_rec:
        why.append(f"P1A_RECALL_ACCOUNTING:{len(recall_rows)}/{n_rec}")
    for r in recall_rows:
        if r.get("recalled") and (r.get("c_recalled") != r.get("c_continuum") or r.get("Jbar_recalled") !=
                                  r.get("c_recalled") + r.get("L_recalled")):
            why.append(f"P1A_RECALL_BACKGROUND_MOVED:{r['shell']}:{r['call']}")
        if r.get("recalled") and r["status"] not in ("X", "NOT_IDENTIFIABLE"):
            same = (r["R_absorb"], r["R_stim"], r["R_spont"]) == (r["R_absorb_recalled"], r["R_stim_recalled"],
                                                                  r["R_spont_recalled"])
            if (same and r["status"] != "SURVIVES") or (not same and r["status"] != "REFUTED"):
                why.append(f"P1A_RECALL_RECOMPUTE:{r['shell']}:{r['call']}")
    n_b = len(TARGET_SHELLS_OF(keys)) * (len(G["p1b_core"]) + len(G["p1b_ext"]))
    if len(p1b_rows) != n_b:
        why.append(f"P1B_ROW_ACCOUNTING:{len(p1b_rows)}/{n_b}")
    for r in p1b_rows:
        if r["status"] == "X":
            continue
        exp = SEED.det_seedmem_scope_status_check(float(r["d_lo"]) if r.get("d_lo") is not None else math.nan,
                                                  float(r["d_hi"]) if r.get("d_hi") is not None else math.nan)
        if exp != r["status"]:
            why.append(f"P1B_STATUS_RECOMPUTE:{r['shell']}:{r['call']}:{r['status']}!={exp}")
        if (r["call"] in G["p1b_core"]) != (r["range"] == "core"):
            why.append(f"P1B_RANGE:{r['shell']}:{r['call']}")
    return why


def TARGET_SHELLS_OF(keys):
    return sorted({k[2] for k in keys})


def det_nrb_judge_propagate(vwhy, scopes):
    """route every finding to the scope it invalidates (X with the provisional code kept, never consumed)."""
    for w in vwhy:
        kind = w.split(":")[0]
        if kind.startswith("P1A_RECALL"):
            SEED.det_seedmem_scope_block(scopes["P1A_LOCAL"], w)
        elif kind.startswith("P1A"):
            SEED.det_seedmem_scope_block(scopes["P1A"], w)
        elif kind.startswith("P1B"):
            f = w.split(":")
            if kind == "P1B_ROW_ACCOUNTING":
                SEED.det_seedmem_scope_block(scopes["P1B_CORE"], w)
                SEED.det_seedmem_scope_block(scopes["P1B_EXT"], w)
            elif kind == "P1B_X_ROWS":
                SEED.det_seedmem_scope_block(scopes["P1B_CORE" if f[1] == "core" else "P1B_EXT"], w)
            else:
                k = int(f[2]) if len(f) > 2 and f[2].isdigit() else None
                tgt = "P1B_CORE" if (k is not None and k <= 7) else "P1B_EXT"
                SEED.det_seedmem_scope_block(scopes[tgt], w)
        elif kind.startswith("P2"):
            f = w.split(":")
            if len(f) > 1 and f"P2:{f[1]}" in scopes:
                SEED.det_seedmem_scope_block(scopes[f"P2:{f[1]}"], w)
            else:
                for k in scopes:
                    if k.startswith("P2:"):
                        SEED.det_seedmem_scope_block(scopes[k], w)
        elif kind.startswith("ION"):
            f = w.split(":")
            z = f[1] if len(f) > 1 else None
            for k in scopes:
                if k.startswith("IONS:") and (z is None or k == f"IONS:{z}"):
                    SEED.det_seedmem_scope_block(scopes[k], w)
        elif kind.startswith("SPECTRUM"):
            SEED.det_seedmem_scope_block(scopes["SPECTRUM"], w)
        else:
            for k in scopes:
                SEED.det_seedmem_scope_block(scopes[k], w)
    return scopes


def det_nrb_judge_decision(scopes, p2_disposition, descriptors, validity, incomplete):
    """prereg 10.3 follow-up table consuming admissible results only; X scopes carry no decision, and while the run
    is X no science follow-up is produced at all (closeout round 4)."""
    code = lambda k: (scopes.get(k) or {"code": "X"})["code"]                          # noqa: E731
    rows = []
    if incomplete:
        rows.append({"condition": "INCOMPLETE", "follow_up": "required run or output missing: no completion"})
    for k, v in scopes.items():
        if v.get("code") == "X":
            rows.append({"condition": f"{k} X", "follow_up": "no pass, extension or physical-cause decision on this scope "
                                                              "(prereg 10.3)", "blocked_by": v.get("blocked_by", [])[:5]})
    if validity == "X":
        rows.append({"condition": "run inadmissible", "follow_up": "no science follow-up while validity is X "
                                                                     "(prereg 6.2, 10.3)"})
        return {"schema": "DET_NRB_JUDGE_DECISION_V1", "validity": validity,
                "p2_disposition": "INADMISSIBLE" if any(code(k) == "X" for k in scopes if k.startswith("P2:"))
                else p2_disposition, "rows": rows,
                "not_authorised": "no collision-table activation, constant repair, extra storage, other seed or extra GPU "
                                  "run follows automatically (prereg 10.3)"}
    p1a, p1b_c, p1b_e = code("P1A"), code("P1B_CORE"), code("P1B_EXT")
    if p1a == "X":
        pass                                          # recorded above with the other X scopes
    elif p1a != 0:
        rows.append({"condition": "P1(a) link failure", "follow_up": "investigate run / ledger / branch linkage offline"})
    elif p1b_c == 1 or p1b_e == 1:
        rows.append({"condition": "P1(a) survives, P1(b) refuted",
                     "follow_up": "distinguish self-term removal from the reduced source-model failure"})
    p2x = any(scopes[k]["code"] == "X" for k in scopes if k.startswith("P2:"))
    pub_codes = [code(k) for k in scopes if k.startswith("IONS:")] + [code("SPECTRUM")]
    pub_refuted = any(c == 1 for c in pub_codes)
    pub_x = any(c == "X" for c in pub_codes)
    if p2x:
        rows.append({"condition": "P2 X", "follow_up": "P2 inadmissible: not reported as a small-change shortfall"})
    elif p1a == "X" or validity == "X":
        pass
    elif p2_disposition == "NOT_CONVERGED":
        rows.append({"condition": "P2 not met", "follow_up": "report the stationarity.csv trajectory and the registered "
                     "endpoint descriptors / axis codes; iteration extension is a user decision",
                     "descriptors": descriptors})
    if pub_x or validity == "X":
        pass
    elif p2_disposition == "P2_SMALL_CHANGE" and pub_refuted:
        rows.append({"condition": "P2 survives, public comparison refuted",
                     "follow_up": "small change of the registered sample does not guarantee public agreement"})
    elif p2_disposition == "NOT_CONVERGED" and not pub_refuted and all(c == 0 for c in pub_codes):
        rows.append({"condition": "public comparison survives, P2 not met",
                     "follow_up": "restricted to a finite-iteration state comparison"})
    elif p2_disposition == "P2_SMALL_CHANGE" and all(c == 0 for c in pub_codes):
        rows.append({"condition": "P2 and public comparison survive",
                     "follow_up": "restricted to single seed, fixed T, registered sample and bands"})
    return {"schema": "DET_NRB_JUDGE_DECISION_V1", "validity": validity,
            "p2_disposition": "INADMISSIBLE" if p2x else p2_disposition, "rows": rows,
            "not_authorised": "no collision-table activation, constant repair, extra storage, other seed or extra GPU "
                              "run follows automatically (prereg 10.3)"}


# ------------------------------------------------------------------ analyze
def det_nrb_judge_analyze(n_run, e_run, r_run, prior_analysis, prior_selftest, expectations, src, public, out, lib_dir,
                          target=TARGET, fault=None, prereg=None, bytes_ref=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = Path(src)
    outer = target["outer"]
    G = det_nrb_judge_gens(outer)
    why = []
    lib_rec = SEED.det_seedmem_scope_build_lib(src, Path(lib_dir))
    if lib_rec["rc"] != 0:
        why.append("LIB_BUILD_FAILED")
    lib = SEED.det_seedmem_scope_Lib(lib_rec["lib"]) if lib_rec["rc"] == 0 else None
    src_text = (src / "lumina_plasma.c").read_text(errors="replace")
    cuda = src / "lumina_cuda.cu"
    cuda_text = cuda.read_text(errors="replace") if cuda.exists() else ""
    csob = SEED.det_seedmem_scope_csob(src)
    if csob is None or csob != POP.C_SOB:
        why.append("C_SOB_SOURCE_MISMATCH")
    for nm in ("line_net_bracket_rates", "bb_nrb_arm"):
        if nm not in src_text:
            why.append(f"PRODUCTION_CALL_PATH_ABSENT:{nm}")
    # sample inherited by reference (prereg 2.4)
    A = Path(prior_analysis)
    sample = json.loads((A / "sample.json").read_text())
    keys = [tuple(k) for k in sample["keys"]]
    if FT(fault, "drop_counter"):
        keys = keys[:-1]
    sample_rec = {"path": str(A / "sample.json"), "sha256": sha(A / "sample.json"), "schema": sample.get("schema"),
                  "n_keys": len(keys), "counts": {z: sum(1 for k in keys if k[0] == z) for z in target["Z"]},
                  "control_shells": sum(1 for k in keys if k[1] == target["control"])}
    if len(keys) != target["n_keys"] or sample_rec["counts"] != target["counts"] or \
            sample_rec["control_shells"] != len(target["shells"]) or any(k[2] not in target["shells"] for k in keys):
        why.append(f"SAMPLE_NOT_REGISTERED:{sample_rec}")
    deck = SEED.det_seedmem_scope_deck(Path(r_run) / "input/model", target["Z"], target["ion"])
    why += deck["why"]
    missing_lines = sorted({k[1] for k in keys} - set(deck["at"]))
    if missing_lines:
        why.append(f"SAMPLE_LINE_NOT_IN_DECK:{missing_lines[:5]}")
    if target["control"] not in deck["at"]:
        why.append("CONTROL_NOT_IN_DECK")
    level_set = sorted({(Z, deck["at"][l][w], s) for Z, l, s in keys if l in deck["at"] for w in ("lo", "up")})
    sample_rec["n_level_entries"] = len(level_set)
    if len(level_set) != target["n_levels"]:
        why.append(f"LEVEL_SET_NOT_REGISTERED:{len(level_set)}!={target['n_levels']}")
    deck_ids = {}
    for tag, rp in (("N", n_run), ("E", e_run), ("R", r_run)):
        p = Path(rp) / "input/deck.sha256"
        ent = dict(reversed(ln.split(None, 1)) for ln in p.read_text().split("\n") if ln.strip()) if p.exists() else {}
        deck_ids[tag] = {k.strip(): ent.get(k) for k in ("line_list.csv", "levels.csv")}
    if len({json.dumps(v, sort_keys=True) for v in deck_ids.values()}) != 1:
        why.append("DECK_DIFFERS_BETWEEN_RUNS")
    for k in ("line_list.csv", "levels.csv"):
        if deck_ids["N"].get(k) != sha(Path(n_run) / "input/model" / k):
            why.append(f"DECK_FILE_NOT_REGISTERED:{k}")
    if why:
        res = {"schema": "DET_NRB_JUDGE_RESULT_V1", "validity": "X", "reasons": why, "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    deck["r"] = {l: SEED.det_seedmem_scope_r(deck["at"][l], csob) for l in {k[1] for k in keys}}
    r_bin = det_nrb_judge_kv(Path(r_run) / "STAGE_IDENTITY.txt").get("run_binary_sha256")
    runs = {}
    for tag, rp, o in (("N", n_run, outer), ("R", r_run, 6), ("E", e_run, 6)):
        rd = det_nrb_judge_run(rp, tag, {(k[1], k[2]) for k in keys} if tag == "N" else set(), src_text, deck, o,
                               fault if tag == "N" else None, cuda_text, r_bin if tag == "N" else None)
        rd["shells"] = list(target["shells"])
        runs[tag] = rd
    N = runs["N"]
    why += N["why"]
    if N["form"] != "NRB":
        why.append("N:RATE_FORM_NOT_NRB")
    if runs["R"]["form"] != "NRB" or runs["E"]["form"] != "OFF":
        why.append("BASELINE_FORMS_NOT_REGISTERED")
    ex_why = det_nrb_judge_exports_check(Path(r_run) / "input/resolved_lumina.exports",
                                         Path(n_run) / "input/resolved_lumina.exports", r_run, n_run, outer)
    why += [f"N:EXPORTS:{w}" for w in ex_why["why"]]
    if Path(n_run).resolve() != Path(r_run).resolve():
        in_id = det_nrb_judge_input_identity(n_run)
    else:                                            # preflight regression: the reference read as N (prereg 3.4)
        in_id = {"why": [], "not_applicable": "reference run read as N (preflight regression); it is the identity source"}
    why += [f"N:{w}" for w in in_id["why"]]
    incomplete = any(w.startswith(("N:RUN_NOT_SUCCESS", "N:RUN_FOOTER_MISSING", "N:LINK_CAPTURE_FILE_MISSING",
                                   "N:DUMP_GENERATIONS")) for w in N["why"])
    timing = {"inputs_s": time.time() - t0}
    if why:
        res = {"schema": "DET_NRB_JUDGE_RESULT_V1", "validity": "X", "reasons": why[:200], "n_reasons": len(why),
               "fault": fault, "incomplete": incomplete, "run_links": {t: runs[t]["binary_link"] for t in runs},
               "exports_check": ex_why}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    # P1(a)
    p1a_rows, p1a_states = det_nrb_judge_p1a(N, keys, deck, lib, fault)
    recall_rows = det_nrb_judge_recall(N, deck, lib, target["control"], fault)
    inherit = det_nrb_judge_inheritance(lib_rec, prior_analysis, prior_selftest)
    # P1(b)
    p1b_rows, p1b_states = det_nrb_judge_p1b(N, deck, csob, lib, target["control"], fault)
    # level authority and P2
    cap_rows, cap_why, cap_status = det_nrb_judge_capture_check(N, keys, deck, csob, lib, fault)
    p2_rows, p2_codes, p2_expected, p2_contra = det_nrb_judge_p2(N, keys, deck, csob, lib, cap_status, fault)
    if FT(fault, "mixed_windows"):
        for ax, c in p2_codes.items():
            if c["any_past_window_small"]:
                c["code"], c["state"] = 0, "SURVIVES"
    p2_why = det_nrb_judge_p2_validate(p2_rows, p2_codes, p2_expected, G) + [w for w in cap_why] + \
        [f"P2_LEVEL_CONTRADICTION:level:{Z}:{lev}:{s}:{g}:{n}" for Z, lev, s, g, n in p2_contra]
    # public ions (N generation g_obs; E', R' generation 7 alongside)
    pub_blocks = {Z: det_nrb_judge_public_block(public[Z], target["public_time"], fault) for Z in target["Z"]}
    ion_rows, ion_states = [], {}
    for tag in ("N", "R", "E"):
        rd = runs[tag]
        dump, derr = det_nrb_judge_ion_dump(rd["run"], rd["G"]["g_obs"], rd["shells"])
        rws, sts = det_nrb_judge_ions(rd, dump, pub_blocks, target, fault if tag == "N" else None)
        ion_rows += rws
        if tag == "N":
            ion_states = sts
            if derr:
                why.append(f"N:{derr}")
    ion_why = det_nrb_judge_ions_validate(ion_rows, n_run, G, target, public)
    # spectrum
    targets = det_nrb_judge_targets(expectations)
    specs = {"public": REP.det_siiii_repair_public(public["spectrum"]),
             "N": REP.det_siiii_repair_spec(Path(n_run) / "work/lumina_spectrum_det_obs.csv")
             if (Path(n_run) / "work/lumina_spectrum_det_obs.csv").exists() else None,
             "EprimeN": REP.det_siiii_repair_spec(Path(e_run) / "work/lumina_spectrum_det_obs.csv"),
             "RprimeN": REP.det_siiii_repair_spec(Path(r_run) / "work/lumina_spectrum_det_obs.csv")}
    spec_rows, spec_summ, spec_conds, spec_overall = det_nrb_judge_spectrum(specs, targets, fault) if targets else \
        ([], {"status": "TARGETS_ABSENT"}, {}, "NOT_IDENTIFIABLE")
    spec_why = det_nrb_judge_spectrum_validate(spec_summ, spec_conds, spec_overall, targets) if targets else \
        ["SPECTRUM_TARGETS_ABSENT"]
    if specs["N"] is None:
        incomplete = True
    # bytes (prereg 3.4)
    byte_cmp = det_nrb_judge_bytes(n_run, r_run) if bytes_ref else None
    timing["compute_s"] = time.time() - t0 - timing["inputs_s"]
    # validators and propagation
    vwhy = det_nrb_judge_validate(p1a_rows, recall_rows, p1b_rows, keys, G, outer) + p2_why + ion_why + spec_why
    if FT(fault, "status_flip_all"):
        pass                                        # the flipped states are caught by P1A_STATUS_RECOMPUTE above
    # closeout B1: the fixed 60-cell recall keeps its denominator (missing and X cells included); a failed
    # inheritance adds NOT_IDENTIFIABLE but never hides an X or a counterexample of the recall
    local_states = [r["status"] for r in recall_rows] or ["NOT_IDENTIFIABLE"]
    if FT(fault, "recall_recalled_only"):
        local_states = [r["status"] for r in recall_rows if r.get("recalled")] or ["NOT_IDENTIFIABLE"]
    if not inherit["inherited"]:
        local_states = local_states + ["NOT_IDENTIFIABLE"]
    scopes = {"P1A": SEED.det_seedmem_scope_set_code(p1a_states),
              "P1A_LOCAL": SEED.det_seedmem_scope_set_code(local_states),
              "P1B_CORE": SEED.det_seedmem_scope_set_code(p1b_states["core"]),
              "P1B_EXT": SEED.det_seedmem_scope_set_code(p1b_states["extension"] or ["NOT_IDENTIFIABLE"]),
              "SPECTRUM": {"code": "X" if spec_overall == "X" else CODE[spec_overall], "state": spec_overall,
                           "conditions": {k: CODE.get(v, "X") for k, v in spec_conds.items()}}}
    for ax, c in p2_codes.items():
        scopes[f"P2:{ax}"] = c
    for Z in target["Z"]:
        scopes[f"IONS:{Z}"] = SEED.det_seedmem_scope_set_code(ion_states.get(Z, []))
    scopes["P1A_LOCAL"]["inheritance"] = "INHERITED" if inherit["inherited"] else "NOT_INHERITED:" + ";".join(inherit["why"])
    n_x = {"p1a": p1a_states.count("X"), "recall": sum(1 for r in recall_rows if r["status"] == "X"),
           "p1b_core": p1b_states["core"].count("X"), "p1b_extension": p1b_states["extension"].count("X"),
           "ions_14": ion_states.get(14, []).count("X"), "ions_16": ion_states.get(16, []).count("X")}
    # closeout B3: an X row makes its scope X and the run inadmissible (prereg 6.2), routed to that scope only
    for k_, nm in (("p1a", "P1A_X_ROWS"), ("recall", "P1A_RECALL_X_ROWS"), ("p1b_core", "P1B_X_ROWS:core"),
                   ("p1b_extension", "P1B_X_ROWS:extension"), ("ions_14", "ION_X_ROWS:14"), ("ions_16", "ION_X_ROWS:16")):
        if n_x[k_]:
            vwhy.append(f"{nm}:{n_x[k_]}")
    exp_local = SEED.det_seedmem_scope_universal_check([r["status"] for r in recall_rows] +
                                                       ([] if inherit["inherited"] else ["NOT_IDENTIFIABLE"]))
    if scopes["P1A_LOCAL"]["state"] != exp_local:
        vwhy.append(f"P1A_RECALL_SET_RECOMPUTE:{scopes['P1A_LOCAL']['state']}!={exp_local}")
    if scopes["P1A"]["state"] != SEED.det_seedmem_scope_universal_check(p1a_states):
        vwhy.append("P1A_SET_RECOMPUTE")
    validity = "VALID" if not vwhy else "X"
    reasons = vwhy[:50]                              # stored messages only; propagation below consumes every finding
    scopes = det_nrb_judge_propagate(vwhy, scopes)
    p2_states = {ax: scopes[f"P2:{ax}"]["state"] for ax in p2_codes}
    if any(s == "X" for s in p2_states.values()):
        p2_disp = "INADMISSIBLE"
    elif all(s == "SURVIVES" for s in p2_states.values()):
        p2_disp = "P2_SMALL_CHANGE"
    else:
        p2_disp = "NOT_CONVERGED"
    if incomplete:
        p2_disp = "INCOMPLETE"
    descriptors = {ax: scopes[f"P2:{ax}"].get("descriptor") for ax in p2_codes}
    decision = det_nrb_judge_decision(scopes, p2_disp, descriptors, validity, incomplete)
    # outputs
    pd.DataFrame(p1a_rows + [{"run": "N", "kind": "RECALL"} | r for r in recall_rows]).to_csv(out / "p1_rates.csv", index=False)
    pd.DataFrame(p1b_rows).to_csv(out / "p1_source_1206.csv", index=False)
    pd.DataFrame(p2_rows).to_csv(out / "stationarity.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(out / "capture_check.csv", index=False)
    pd.DataFrame(ion_rows).to_csv(out / "ion_fraction.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(out / "spectrum.csv", index=False)
    res = {"schema": "DET_NRB_JUDGE_RESULT_V1", "validity": validity, "reasons": reasons, "n_reasons": len(vwhy),
           "fault": fault, "incomplete": incomplete, "outer_iterations": outer, "generations": G,
           "codes": {k: v for k, v in scopes.items()},
           "P2_disposition": p2_disp, "P2_descriptors": descriptors,
           "P2_last_material_update": {ax: scopes[f"P2:{ax}"].get("last_material_update") for ax in p2_codes},
           "P1A_activation": {"markers": N["markers"], "exports": {k: N["exports"].get(k) for k in
                                                                     ("LUMINA_BB_NRB", "LUMINA_A210_SPRODUCER_CAPTURE",
                                                                      ITER_KEY)},
                              "binary_link": N["binary_link"], "footer": N["footer"]},
           "P1A_H_LOCAL_inheritance": inherit,
           "P1A_recall": {"cells": len(recall_rows), "recalled": sum(1 for r in recall_rows if r.get("recalled")),
                          "by_status": pd.Series([r["status"] for r in recall_rows]).value_counts().to_dict()},
           "P1A_by_branch": pd.Series([r.get("branch") for r in p1a_rows]).value_counts(dropna=False).to_dict(),
           "P1B": {rng: {"cells": len(v), "by_status": pd.Series(v).value_counts().to_dict()} for rng, v in p1b_states.items()},
           "capture_authority": pd.Series([r["authority"] for r in cap_rows]).value_counts().to_dict(),
           "ions": {str(Z): {"by_status": pd.Series(v).value_counts().to_dict()} for Z, v in ion_states.items()},
           "ion_column_assumption": "public si0..si4 / s0..s4 read as I..V (header naming; conditional, prereg 5.4)",
           "spectrum": {"summary": spec_summ, "conditions": spec_conds, "overall": spec_overall, "targets": targets,
                        "targets_file": str(expectations)},
           "bytes_vs_reference": byte_cmp, "collision_authority": N["coll"],
           "solve_unit_authority": {k: v for k, v in N["sl"].items() if k not in ("unit", "size")},
           "registered_not_identifiable": {q: 5 for q in REGISTERED_5}, "registered_not_identifiable_reasons": REGISTERED_5,
           "sample": sample_rec, "exports_check": ex_why, "input_identity": in_id, "x_counts": n_x, "timing_s": timing,
           "lib_build": lib_rec, "run_counters": N["counters"],
           "claim_ceiling": "finite-iteration state of the deterministic lane (MC transport bypassed); no production "
                            "promotion, no physical correctness, no seed independence, no cause attribution"}
    det_nrb_judge_write(out, res, decision, fault)
    # stored-output check (closeouts 1-2): the written files are read back; every scope's code / state, the validity,
    # each P2 axis state and descriptor (generations, M, D, code) recomputed from stationarity.csv, the P2 disposition
    # and the whole decision table recomputed from the stored result; a disagreement makes the run X (files rewritten)
    sw = det_nrb_judge_stored_check(out, G)
    if sw:
        res = json.loads((out / "result.json").read_text())
        res["validity"], res["reasons"] = "X", (res.get("reasons") or []) + sw[:20]
        res["n_reasons"] = res.get("n_reasons", 0) + len(sw)
        res["stored_check"] = sw
        det_nrb_judge_stored_block(res, sw)
        decision = det_nrb_judge_decision(res["codes"], res["P2_disposition"], res.get("P2_descriptors"), "X",
                                          res.get("incomplete"))
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        (out / "decision.json").write_text(json.dumps(decision, indent=1, default=str))
    return res


def det_nrb_judge_stored_block(res, sw):
    """closeout round 4: every stored-check finding makes the proposition it concerns X (an absent registered
    proposition is inserted as X); a finding without a proposition (validity, decision table) leaves the run X only."""
    codes = res.setdefault("codes", {})
    for k in REQUIRED_SCOPES:
        if k not in codes:
            codes[k] = {"code": "X", "state": "X", "blocked_by": [f"STORED_SCOPE_ABSENT:{k}"]}
    for w in sw:
        f = w.split(":")
        kind = f[0]
        tgt = []
        if kind in ("STORED_CODE_STATE", "STORED_FINAL_CODE") and len(f) > 1:
            tgt = [f[1] + (":" + f[2] if f[1] in ("P2", "IONS") and len(f) > 2 else "")]
        elif kind.startswith("STORED_SPECTRUM"):
            tgt = ["SPECTRUM"]
        elif kind in ("P2_STORED_STATE", "P2_STORED_TERMINAL") and len(f) > 1:
            tgt = [f"P2:{f[1]}"]
        elif kind in ("P2_STORED_DESCRIPTOR", "P2_STORED_DESCRIPTOR_ABSENT") and len(f) > 2:
            tgt = [f"P2:{f[2]}"]
        elif kind == "STORED_P2_DISPOSITION":
            tgt = [k for k in codes if k.startswith("P2:")]
        for k in tgt:
            if k in codes and codes[k].get("code") != "X":
                SEED.det_seedmem_scope_block(codes[k], w)
    states = {k: v.get("state") for k, v in codes.items() if k.startswith("P2:")}
    if any(s_ == "X" for s_ in states.values()):
        res["P2_disposition"] = "INADMISSIBLE"
    elif states and all(s_ == "SURVIVES" for s_ in states.values()):
        res["P2_disposition"] = "P2_SMALL_CHANGE"
    else:
        res["P2_disposition"] = "NOT_CONVERGED"
    if res.get("incomplete"):
        res["P2_disposition"] = "INCOMPLETE"
    return res


def det_nrb_judge_write(out, res, decision, fault=None):
    """write result.json / decision.json; selftest-only write faults simulate a defective writer after the in-memory
    validation (the stored-output check must reject each)."""
    r_ = json.loads(json.dumps(res, default=str))
    d_ = json.loads(json.dumps(decision, default=str))
    if FT(fault, "decision_descriptor_flip"):
        for row in d_["rows"]:
            for dsc in (row.get("descriptors") or {}).values():
                dsc["code"] = "END_HIGHER" if dsc.get("code") != "END_HIGHER" else "END_LOWER"
    if FT(fault, "stored_code_flip"):
        r_["codes"]["P2:ion"]["code"] = 0                 # code replaced, state kept
    if FT(fault, "stored_D_alter"):
        r_["P2_descriptors"]["ion"]["D"] = {"lo": 999.0, "mid": 999.0, "hi": 999.0}
        r_["codes"]["P2:ion"]["descriptor"]["D"] = {"lo": 999.0, "mid": 999.0, "hi": 999.0}
    if FT(fault, "stored_rows_delete"):
        d_["rows"] = []
    if FT(fault, "stored_disposition_both"):
        r_["P2_disposition"] = d_["p2_disposition"] = "P2_SMALL_CHANGE"
    if FT(fault, "stored_decision_inadmissible"):
        d_["p2_disposition"] = "INADMISSIBLE"
    if FT(fault, "stored_4_as_1"):
        for k, v in r_["codes"].items():
            if k.startswith("P2:") and v.get("state") == "UNRESOLVED":
                v["code"], v["state"] = 1, "REFUTED"
    if FT(fault, "stored_5_as_0"):
        for k, v in r_["codes"].items():
            if k.startswith("P2:") and v.get("state") == "NOT_IDENTIFIABLE":
                v["code"], v["state"] = 0, "SURVIVES"
    if FT(fault, "stored_ions_flip"):
        r_["codes"]["IONS:14"]["code"], r_["codes"]["IONS:14"]["state"] = 0, "SURVIVES"
    if FT(fault, "stored_spectrum_flip"):
        r_["codes"]["SPECTRUM"]["code"], r_["codes"]["SPECTRUM"]["state"] = 0, "SURVIVES"
    if FT(fault, "stored_p1b_flip"):
        r_["codes"]["P1B_CORE"]["code"], r_["codes"]["P1B_CORE"]["state"] = 1, "REFUTED"
    if FT(fault, "stored_scope_delete"):
        r_["codes"].pop("P1A_LOCAL", None)
        r_["codes"].pop("IONS:16", None)
    if FT(fault, "stored_spec_targets_delete"):
        r_["spectrum"].pop("targets", None)
        r_["codes"]["SPECTRUM"]["code"], r_["codes"]["SPECTRUM"]["state"] = 0, "SURVIVES"
    if FT(fault, "stored_spectrum_section_delete"):
        r_.pop("spectrum", None)
    if FT(fault, "stored_spec_conditions_delete"):
        r_["spectrum"].pop("conditions", None)
    if FT(fault, "stored_required_delete"):
        for k in ("P1A", "P1B_CORE", "P1B_EXT", "SPECTRUM"):
            r_["codes"].pop(k, None)
    if FT(fault, "stored_X_as_valid"):
        r_["validity"] = d_["validity"] = "VALID"
    (Path(out) / "result.json").write_text(json.dumps(r_, indent=1, default=str))
    (Path(out) / "decision.json").write_text(json.dumps(d_, indent=1, default=str))


REQUIRED_SCOPES = ("P1A", "P1A_LOCAL", "P1B_CORE", "P1B_EXT", "SPECTRUM", "P2:ion", "P2:ne", "P2:jbar", "P2:level",
                   "IONS:14", "IONS:16")


def det_nrb_judge_stored_final_codes(out, res, codes):
    """closeout round 3: the final codes outside P2 recomputed from the stored rows and conditions (prereg 6.2 universal
    aggregation): P1(a) from p1_rates.csv, the fixed recall with the stored inheritance state, P1(b) core / extension
    from p1_source_1206.csv, the public ions from ion_fraction.csv, the spectrum from the stored summary and targets."""
    out = Path(out)
    why = []
    uc = SEED.det_seedmem_scope_universal_check
    pr = pd.read_csv(out / "p1_rates.csv", float_precision="round_trip")
    kind = pr["kind"] if "kind" in pr.columns else pd.Series([None] * len(pr))
    cells = pr[kind.fillna("") != "RECALL"]
    recall = pr[kind.fillna("") == "RECALL"]
    inherited = ((res.get("P1A_H_LOCAL_inheritance") or {}).get("inherited") is True)
    sb = pd.read_csv(out / "p1_source_1206.csv", float_precision="round_trip")
    io = pd.read_csv(out / "ion_fraction.csv", float_precision="round_trip")
    ion_rows = io[(io.run == "N") & (io.ion.astype(str) != "III/II")]
    exp = {"P1A": uc(list(cells.status)) if len(cells) else "NOT_IDENTIFIABLE",
           "P1A_LOCAL": uc(list(recall.status) + ([] if inherited else ["NOT_IDENTIFIABLE"])) if len(recall) else "NOT_IDENTIFIABLE",
           "P1B_CORE": uc(list(sb[sb.range == "core"].status)) if (sb.range == "core").any() else "NOT_IDENTIFIABLE",
           "P1B_EXT": uc(list(sb[sb.range == "extension"].status)) if (sb.range == "extension").any() else "NOT_IDENTIFIABLE"}
    for Z in (14, 16):
        st_ = list(ion_rows[ion_rows.Z == Z].status)
        exp[f"IONS:{Z}"] = uc(st_) if st_ else "NOT_IDENTIFIABLE"
    spec = res.get("spectrum") or {}
    # a scope already X keeps its X (blocked by an earlier finding); only admissible stored codes are recomputed, and an
    # admissible spectrum code needs its stored evidence (summary, conditions, overall, targets) — closeout round 4
    sp_code = (codes.get("SPECTRUM") or {}).get("code")
    if sp_code is not None and sp_code != "X":
        # closeout round 5: presence is by type; the empty condition set of a not-identifiable spectrum is a stored value
        need = {"summary": dict, "conditions": dict, "overall": str, "targets": dict}
        absent = [k for k, t in need.items() if not isinstance(spec.get(k), t) or (k != "conditions" and not spec.get(k))]
        if absent:
            why.append(f"STORED_SPECTRUM_EVIDENCE_ABSENT:{','.join(absent)}")
    if spec.get("targets") and spec.get("summary") and sp_code is not None and sp_code != "X":
        sw_ = det_nrb_judge_spectrum_validate(spec.get("summary") or {}, spec.get("conditions") or {}, spec.get("overall"),
                                              spec["targets"])
        why += [f"STORED_{w}" for w in sw_]
        exp["SPECTRUM"] = spec.get("overall")
        sc = (codes.get("SPECTRUM") or {}).get("conditions") or {}
        if sc != {k: CODE.get(v, "X") for k, v in (spec.get("conditions") or {}).items()} and \
                (codes.get("SPECTRUM") or {}).get("code") != "X":
            why.append("STORED_SPECTRUM_CONDITION_CODES")
    for k, e in exp.items():
        v = codes.get(k)
        if v is None or v.get("code") == "X":
            continue
        if v.get("state") != e:
            why.append(f"STORED_FINAL_CODE:{k}:{v.get('state')}!={e}")
    return why


def det_nrb_judge_stored_check(out, G):
    """stored-output check (independent of the code path): read result.json, decision.json and stationarity.csv
    (round-trip floats) and recompute (1) every scope's code from its state, (2) validity X whenever a scope is X,
    (3) each P2 axis' terminal states and state, (4) each descriptor's generations, M endpoints, D interval and code,
    (5) the P2 disposition, (6) the whole decision table from the stored codes."""
    out = Path(out)
    why = []
    res = json.loads((out / "result.json").read_text())
    dec = json.loads((out / "decision.json").read_text())
    st = pd.read_csv(out / "stationarity.csv", float_precision="round_trip")
    codes = res.get("codes") or {}
    for k in REQUIRED_SCOPES:                             # closeout round 3: every registered proposition present
        if k not in codes:
            why.append(f"STORED_SCOPE_ABSENT:{k}")
    for k, v in codes.items():
        c_, s_ = v.get("code"), v.get("state")
        if not ((c_ == "X" and s_ == "X") or (s_ in CODE and c_ == CODE[s_])):
            why.append(f"STORED_CODE_STATE:{k}:{c_}:{s_}")
    why += det_nrb_judge_stored_final_codes(out, res, codes)
    if any(v.get("code") == "X" for v in codes.values()) and res.get("validity") != "X":
        why.append("STORED_VALIDITY_WITH_X_SCOPE")
    t0, t1 = G["terminal"]
    states = {}
    for ax in ("ion", "ne", "jbar", "level"):
        sub = {int(r.generation): r for r in st[st.axis == ax].itertuples()}

        def small(r):
            if r is None:
                return "NOT_IDENTIFIABLE"
            if r.M_lo >= P2_CRIT:
                return "REFUTED"
            if int(r.missing):
                return "NOT_IDENTIFIABLE"
            return "SURVIVES" if r.M_hi < P2_CRIT else "UNRESOLVED"
        one = [small(sub.get(t0)), small(sub.get(t1))]
        fin = "SURVIVES" if one == ["SURVIVES", "SURVIVES"] else SEED.det_seedmem_scope_universal_check(one)
        rc = codes.get(f"P2:{ax}", {})
        states[ax] = rc.get("state")
        if rc.get("code") != "X":
            if rc.get("state") != fin:
                why.append(f"P2_STORED_STATE:{ax}:{rc.get('state')}!={fin}")
            if list((rc.get("terminal_states") or {}).values()) != one:
                why.append(f"P2_STORED_TERMINAL:{ax}")
        a, b = sub.get(G["start"]), sub.get(G["g_obs"])
        exp = {"start_generation": G["start"], "end_generation": G["g_obs"],
               "M_start": None if a is None else {"lo": float(a.M_lo), "mid": float(a.M_mid), "hi": float(a.M_hi)},
               "M_end": None if b is None else {"lo": float(b.M_lo), "mid": float(b.M_mid), "hi": float(b.M_hi)}}
        if a is None or b is None or int(a.missing) or int(b.missing) or \
                not all(math.isfinite(float(x)) for x in (a.M_lo, a.M_mid, a.M_hi, b.M_lo, b.M_mid, b.M_hi)):
            exp["code"], exp["D"] = "END_UNAVAILABLE", None
        else:
            code_, lo, hi = det_nrb_judge_end_code(float(a.M_lo), float(a.M_hi), float(b.M_lo), float(b.M_hi))
            exp["code"], exp["D"] = code_, {"lo": lo, "mid": float(b.M_mid) - float(a.M_mid), "hi": hi}
        dec_descs = [(row.get("descriptors") or {}).get(ax) for row in dec.get("rows", []) if row.get("descriptors")]
        for d_dec in dec_descs:                        # the decision's copy of a P2 descriptor belongs to that axis
            if d_dec is not None and any(d_dec.get(f_) != exp[f_] for f_ in ("start_generation", "end_generation",
                                                                            "M_start", "M_end", "code", "D")):
                why.append(f"P2_STORED_DESCRIPTOR:decision:{ax}:fields")
        for where, d_ in (("result", (res.get("P2_descriptors") or {}).get(ax)),
                          ("code", (rc.get("descriptor") if rc.get("code") != "X" or "descriptor" in rc else None))):
            if d_ is None:
                why.append(f"P2_STORED_DESCRIPTOR_ABSENT:{where}:{ax}")
                continue
            for f_ in ("start_generation", "end_generation", "M_start", "M_end", "code", "D"):
                if d_.get(f_) != exp[f_]:
                    why.append(f"P2_STORED_DESCRIPTOR:{where}:{ax}:{f_}")
    if any(s_ == "X" for s_ in states.values()):
        disp = "INADMISSIBLE"
    elif all(s_ == "SURVIVES" for s_ in states.values()):
        disp = "P2_SMALL_CHANGE"
    else:
        disp = "NOT_CONVERGED"
    if res.get("incomplete"):
        disp = "INCOMPLETE"
    if res.get("P2_disposition") != disp:
        why.append(f"STORED_P2_DISPOSITION:{res.get('P2_disposition')}!={disp}")
    codes_x = dict(codes)
    for k in REQUIRED_SCOPES:
        codes_x.setdefault(k, {"code": "X", "state": "X"})
    exp_dec = det_nrb_judge_decision(codes_x, disp, res.get("P2_descriptors"), res.get("validity"), res.get("incomplete"))
    exp_dec = json.loads(json.dumps(exp_dec, default=str))
    if exp_dec != dec:
        diff = [k for k in set(exp_dec) | set(dec) if exp_dec.get(k) != dec.get(k)]
        why.append(f"STORED_DECISION_RECOMPUTE:{','.join(sorted(diff))}")
    return why


# ------------------------------------------------------------------ exports / staging (prereg 2.2, 3.2)
def det_nrb_judge_exports_map(ref_lines, ref_run, new_run, outer, fault=None):
    """the N exports from the R' exports: the seven path keys re-rooted (exact per-key mapping), the iteration key
    set to `outer`, every other line kept byte-identical."""
    ref_run, new_run = str(Path(ref_run)), str(Path(new_run))
    out, mapping = [], {}
    for k, v, raw in ref_lines:
        if k == ITER_KEY:
            nv = str(13 if FT(fault, "iter13") else 6 if FT(fault, "iter6") else outer)
            out.append(f'declare -x {k}="{nv}"')
            mapping[k] = (v, nv)
        elif k in PATH_KEYS:
            if not v.startswith(ref_run + "/"):
                raise ValueError(f"PATH_KEY_NOT_UNDER_REFERENCE:{k}")
            nv = new_run + v[len(ref_run):]
            out.append(f'declare -x {k}="{nv}"')
            mapping[k] = (v, nv)
        elif FT(fault, "nrb_off") and k == "LUMINA_BB_NRB":
            out.append(f'declare -x {k}="0"')
        elif FT(fault, "capture_off") and k == "LUMINA_A210_SPRODUCER_CAPTURE":
            out.append(f'declare -x {k}="0"')
        else:
            out.append(raw)
    if FT(fault, "extra_env"):
        out.append('declare -x LUMINA_NRB_EXTRA="1"')
    return "\n".join(out) + "\n", mapping


def det_nrb_judge_exports_check(ref_path, new_path, ref_run, new_run, outer):
    """prereg 2.2 / 3.2 (4): after the per-key re-rooting the two exports differ only in the iteration key."""
    ref = det_nrb_judge_exports_lines(ref_path)
    new = det_nrb_judge_exports_lines(new_path)
    expect, _ = det_nrb_judge_exports_map(ref, ref_run, new_run, outer)
    got = Path(new_path).read_text()
    why = []
    rk = {k: v for k, v, _ in ref if k}
    nk = {k: v for k, v, _ in new if k}
    for k in set(rk) | set(nk):
        if k == ITER_KEY:
            if rk.get(k) != "6" or nk.get(k) != str(outer):
                why.append(f"ITER:{rk.get(k)}->{nk.get(k)}")
        elif k in PATH_KEYS:
            if k not in rk or k not in nk or nk[k] != str(new_run) + rk[k][len(str(ref_run)):]:
                why.append(f"PATH_KEY:{k}")
        elif rk.get(k) != nk.get(k):
            why.append(f"KEY_DIFFERS:{k}:{rk.get(k)}!={nk.get(k)}")
    if got != expect:
        why.append("EXPORTS_NOT_BYTE_EXPECTED")
    if nk.get("LUMINA_BB_NRB") != "1" or nk.get("LUMINA_A210_SPRODUCER_CAPTURE") != "1":
        why.append("NRB_FORM_NOT_ARMED")
    return {"reference_sha256": sha(ref_path), "new_sha256": sha(new_path), "why": why,
            "delta": {k: (rk.get(k), nk.get(k)) for k in set(rk) | set(nk) if rk.get(k) != nk.get(k)}}


def det_nrb_judge_copy_tree(src, dst, listing):
    """explicit copy (no hard links, no filesystem feature): every file is copied with copyfile and its SHA-256 verified."""
    src, dst = Path(src), Path(dst)
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=False)
            continue
        if p.is_symlink():
            raise ValueError(f"SYMLINK_IN_INPUT:{p}")
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        h1, h2 = sha(p), sha(q)
        if h1 != h2:
            raise ValueError(f"COPY_VERIFY_FAILED:{rel}")
        listing[str(rel)] = {"sha256": h1, "bytes": p.stat().st_size}


def det_nrb_judge_stage(ref_run, run, runner, gpu_profile, preflight_dir, outer=OUTER, prereg=PREREG):
    """prereg 9.4 stage: a new run root with the R' inputs copied (explicit list), the re-rooted exports with the
    registered iteration count, the runner copy with the GPU profile line, STAGE_IDENTITY / OWNER / SUBMIT_COMMAND."""
    ref_run, run, runner = Path(ref_run), Path(run), Path(runner)
    if run.exists():
        raise FileExistsError(f"RUN_EXISTS:{run}")
    if gpu_profile not in GPU_PROFILES:
        raise ValueError(f"GPU_PROFILE_INVALID:{gpu_profile}")
    pre = json.loads((Path(preflight_dir) / "offline.json").read_text())
    if not pre.get("submit_allowed"):
        raise ValueError("PREFLIGHT_NOT_PASSED")
    ref_ident = det_nrb_judge_kv(ref_run / "STAGE_IDENTITY.txt")
    listing = {}
    (run / "input").mkdir(parents=True)
    (run / "work").mkdir()
    for f in STAGED_FILES:
        p = ref_run / "input" / f
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"REFERENCE_INPUT_MISSING:{f}")
        shutil.copyfile(p, run / "input" / f)
        shutil.copymode(p, run / "input" / f)
        if sha(p) != sha(run / "input" / f):
            raise ValueError(f"COPY_VERIFY_FAILED:{f}")
        listing[f] = {"sha256": sha(p), "bytes": p.stat().st_size}
    for d in STAGED_DIRS:
        det_nrb_judge_copy_tree(ref_run / "input" / d, run / "input" / d, listing)
    ref_lines = det_nrb_judge_exports_lines(ref_run / "input/resolved_lumina.exports")
    text, mapping = det_nrb_judge_exports_map(ref_lines, ref_run, run, outer)
    (run / "input/resolved_lumina.exports").write_text(text)
    ex_sha = sha(run / "input/resolved_lumina.exports")
    (run / "input/resolved_lumina.sha256").write_text(ex_sha + "\n")
    chk = det_nrb_judge_exports_check(ref_run / "input/resolved_lumina.exports", run / "input/resolved_lumina.exports",
                                      ref_run, run, outer)
    if chk["why"]:
        raise ValueError(f"EXPORTS_CHECK:{chk['why']}")
    rtext = runner.read_text()
    if len(re.findall(r"^readonly GPU_PROFILE=.*$", rtext, re.M)) != 1:
        raise ValueError("RUNNER_PROFILE_LINE_NOT_UNIQUE")
    rtext = re.sub(r"^readonly GPU_PROFILE=.*$", f"readonly GPU_PROFILE={gpu_profile}", rtext, flags=re.M)
    rtext = re.sub(r"^#SBATCH --partition=.*$", f"#SBATCH --partition={GPU_PROFILES[gpu_profile]}", rtext, flags=re.M)
    kind = "H200" if gpu_profile == "h200x2" else "H100"
    rtext = re.sub(r"^#SBATCH --gres=gpu:.*$", f"#SBATCH --gres=gpu:{kind}:2", rtext, flags=re.M)
    (run / "job.slurm").write_text(rtext)
    os.chmod(run / "job.slurm", 0o644)
    bsha = (run / "input/binary.sha256").read_text().split()[0]
    actual = sha(run / "input/lumina_cuda")
    if bsha != actual or ref_ident.get("run_binary_sha256") != actual:
        raise ValueError("BINARY_IDENTITY")
    ident = {"schema": "NRB_JUDGE_STAGE_V1", "run": str(run), "evidence": str(run), "gpu_profile": gpu_profile,
             "outer_iterations": outer, "run_binary_sha256": actual,
             "run_build_identity_sha256": sha(run / "input/run_build_identity.json"),
             "effective_exports_sha256": ex_sha, "runner_sha256": sha(run / "job.slurm"),
             "registered_runner_sha256": sha(runner), "contract": prereg, "lineage_run": str(ref_run),
             "lineage_exports_sha256": sha(ref_run / "input/resolved_lumina.exports"),
             "runner_edits": f"GPU_PROFILE={gpu_profile};partition;gres (vs registered runner)",
             "exports_delta_vs_lineage": f"{ITER_KEY} 6->{outer}; path keys re-rooted: " + ",".join(PATH_KEYS),
             "preflight": str(Path(preflight_dir) / "offline.json"), "preflight_sha256": sha(Path(preflight_dir) / "offline.json"),
             "time_ceiling_s": TIME_CEILING_S, "gpu_hours_ceiling": GPU_HOURS, "staged_files": len(listing)}
    (run / "STAGE_IDENTITY.txt").write_text("".join(f"{k}={v}\n" for k, v in ident.items()))
    (run / "OWNER.txt").write_text("project=LUMINA-SN\nrepo=Lumina-sn thenmc-macroatom-fluorescence\n"
                                   f"purpose=DET-NRB-JUDGE N run (R' binary/inputs, {ITER_KEY}={outer}, cold start, "
                                   f"{gpu_profile}), user decision 2026-10-08\nowner=Claude driver session\nDO_NOT_CANCEL\n")
    (run / "SUBMIT_COMMAND.txt").write_text(
        f"cd {run} && sbatch --exclude=syn102,syn03 --time=21:03:24 --output={run}/slurm-%j.out "
        f"--error={run}/slurm-%j.err {run}/job.slurm {run}\n")
    (run / "STAGE_FILES.json").write_text(json.dumps({"files": listing, "exports_mapping": mapping}, indent=1))
    return ident


# ------------------------------------------------------------------ preflight (prereg 3.2, 3.4)
def det_nrb_judge_runner_selftest(runner, out):
    t0 = time.time()
    p = subprocess.run(["bash", str(runner), "--selftest"], capture_output=True, text=True)
    (Path(out) / "runner_selftest.log").write_text(p.stdout + p.stderr)
    cases = re.findall(r"^RUNNER_SELFTEST case=(\S+) expected=(\S+) rc=(\d+)(?: reason=(\S+))?", p.stdout, re.M)
    m = re.search(r"NRB_JUDGE_RUNNER_SELFTEST_PASS positive=(\d+) negative=(\d+)", p.stdout)
    consts = {k: re.search(rf"^readonly {k}=(\S+)$", Path(runner).read_text(), re.M) for k in RUNNER_CONSTANTS}
    consts = {k: (None if v is None else v.group(1)) for k, v in consts.items()}
    return {"rc": p.returncode, "elapsed_s": time.time() - t0, "cases": [{"case": c, "expected": e, "rc": int(r),
                                                                          "reason": s or None} for c, e, r, s in cases],
            "positive": None if m is None else int(m.group(1)), "negative": None if m is None else int(m.group(2)),
            "constants": consts, "constants_registered": consts == RUNNER_CONSTANTS, "runner_sha256": sha(runner),
            "profiles_in_runner": sorted(re.findall(r"^\s+(h\d+x2)\) expected_partition", Path(runner).read_text(), re.M)),
            "expected_kinds": sorted(set(re.findall(r"expected_kinds=(\w+)", Path(runner).read_text())))}


def det_nrb_judge_regress(e_run, r_run, prior_analysis, prior_selftest, expectations, src, public, out, lib_dir):
    """prereg 3.4: the new readout on the stored R' (outer 6) against the SEEDMEM r4 / BT-SEED / diag6 records."""
    A = Path(prior_analysis)
    rec = {"why": []}
    tgt = TARGET | {"outer": 6}
    sub = Path(out) / "regress"
    try:
        res = det_nrb_judge_analyze(r_run, e_run, r_run, prior_analysis, prior_selftest, expectations, src, public, sub,
                                    lib_dir, tgt, bytes_ref=False)
    except Exception as e:                           # noqa: BLE001 - recorded, fail-closed
        rec["why"].append(f"REGRESS_ANALYZE_EXCEPTION:{e!r}")
        return rec
    rec["validity"] = res.get("validity")
    rec["reasons"] = res.get("reasons")
    if res.get("validity") != "VALID":
        rec["why"].append("REGRESS_R_NOT_VALID")
        return rec
    # P1(b) core on R' vs model.csv (run R, control line)
    md = pd.read_csv(A / "model.csv")
    md = md[(md.run == "R") & (md.line == TARGET["control"])]
    new = pd.read_csv(sub / "p1_source_1206.csv")
    n_cmp, bad = 0, []
    for r in new.itertuples():
        o = md[(md.shell == r.shell) & (md.call == r.call)]
        if len(o) != 1:
            bad.append(f"ABSENT:{r.shell}:{r.call}")
            continue
        o = o.iloc[0]
        n_cmp += 1
        if o.status != r.status or not (o.d_lo == r.d_lo and o.d_mid == r.d_mid and o.d_hi == r.d_hi):
            bad.append(f"P1B:{r.shell}:{r.call}:{o.status}/{r.status}:{o.d_mid}/{r.d_mid}")
    rec["p1b_pairs_compared"], rec["p1b_mismatch"] = n_cmp, bad
    if n_cmp != 25 or bad:
        rec["why"].append("REGRESS_P1B")
    # P2 rows vs stationarity.csv (run R)
    st = pd.read_csv(A / "stationarity.csv")
    st = st[st.run == "R"]
    ns = pd.read_csv(sub / "stationarity.csv")
    bad2, n2 = [], 0
    for r in ns.itertuples():
        o = st[(st.axis == r.axis) & (st.generation == r.generation)]
        if len(o) != 1:
            bad2.append(f"ABSENT:{r.axis}:{r.generation}")
            continue
        o = o.iloc[0]
        n2 += 1
        if not (o.M_lo == r.M_lo and o.M_mid == r.M_mid and o.M_hi == r.M_hi and o.n_quantities == r.n_quantities and
                o.missing == r.missing and o.structural_zero == r.structural_zero and o.small == r.small):
            bad2.append(f"P2:{r.axis}:{r.generation}:{o.M_mid}/{r.M_mid}:{o.small}/{r.small}")
    rec["p2_rows_compared"], rec["p2_mismatch"] = n2, bad2
    if n2 != 18 or bad2 or len(ns) != 18:
        rec["why"].append("REGRESS_P2")
    # spectrum E'/R' vs the repair-stage record
    F = json.loads(Path(expectations).with_name("result.json").read_text())
    fs = F.get("spectrum_summary", {})
    spec = res["spectrum"]["summary"]
    bad3 = []
    for nm, key in (("EprimeN", "capture_E"), ("RprimeN", "capture_R")):    # repair-stage records of the same files
        a, b = fs.get(key, {}), spec.get(nm, {})
        for q in ("R8", "d_NUV", "d_UV2500"):
            if (a.get(q) or {}).get("mid") != (b.get(q) or {}).get("mid"):
                bad3.append(f"{nm}:{q}:{(a.get(q) or {}).get('mid')}/{(b.get(q) or {}).get('mid')}")
        if a.get("NUV_over_OPT") != b.get("NUV_over_OPT"):
            bad3.append(f"{nm}:NUV_over_OPT")
    rec["spectrum_mismatch"] = bad3
    if bad3:
        rec["why"].append("REGRESS_SPECTRUM")
    # ion III/II (R', generation 7, NLTE) vs diag6 r2_ionization.csv
    d6 = Path("/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/r2_ionization.csv")
    bad4 = []
    if d6.exists():
        r2 = pd.read_csv(d6)
        ion = pd.read_csv(sub / "ion_fraction.csv")
        ion = ion[(ion.run == "N") & (ion.ion == "III/II")]
        for r in ion.itertuples():
            el = {14: "Si", 16: "S"}[int(r.Z)]
            o = r2[(r2.element == el) & (r2.shell == r.shell)]
            if len(o) != 1 or abs(float(o.iloc[0].R_g7_NLTE) - float(r.ell_N)) > 1e-12 or \
                    abs(float(o.iloc[0].public_log10_III_II) - float(r.ell_public)) > 1e-12:
                bad4.append(f"{el}:{r.shell}")
        rec["ion_ratio_rows_compared"] = int(len(ion))
    else:
        bad4.append("DIAG6_FILE_ABSENT")
    rec["ion_mismatch"] = bad4
    if bad4 or rec.get("ion_ratio_rows_compared") != 10:
        rec["why"].append("REGRESS_IONS")
    rec["result"] = str(sub / "result.json")
    return rec


def det_nrb_judge_input_listing(R):
    """explicit input list of a reference run: the staged files and every file under the staged directories."""
    R = Path(R)
    inputs = {}
    for f in STAGED_FILES:
        p = R / "input" / f
        inputs[f] = {"sha256": sha(p), "bytes": p.stat().st_size} if p.is_file() and not p.is_symlink() else None
    for d in STAGED_DIRS:
        for p in sorted((R / "input" / d).rglob("*")):
            if p.is_file():
                inputs[str(p.relative_to(R / "input"))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return inputs


def det_nrb_judge_preflight(a, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    R, E = Path(a.reference_run), Path(a.baseline_run)
    rec = {"schema": "DET_NRB_JUDGE_OFFLINE_V1", "why": [], "checks": {}}
    # 1 R' identity chain
    ident = det_nrb_judge_kv(R / "STAGE_IDENTITY.txt")
    foot = det_nrb_judge_footer(R) or {}
    build = json.loads((R / "input/run_build_identity.json").read_text())
    actual = sha(R / "input/lumina_cuda")
    chain = {"stage_identity": ident.get("run_binary_sha256"), "footer": foot.get("binary_sha256"),
             "build_record": (build.get("binary") or {}).get("sha256"),
             "binary_sha256_file": (R / "input/binary.sha256").read_text().split()[0], "actual": actual,
             "outer_iterations_R": ident.get("outer_iterations"), "footer_iterations_R": foot.get("outer_iterations"),
             "elapsed_seconds_R": foot.get("elapsed_seconds"), "terminal_state_R": foot.get("terminal_state")}
    rec["checks"]["reference_identity"] = chain
    if len({v for k, v in chain.items() if k in ("stage_identity", "footer", "build_record", "binary_sha256_file", "actual")}) != 1:
        rec["why"].append("REFERENCE_BINARY_CHAIN")
    if chain["outer_iterations_R"] != "6" or chain["footer_iterations_R"] != "6" or chain["terminal_state_R"] != "SUCCESS":
        rec["why"].append("REFERENCE_RUN_NOT_6_SUCCESS")
    # 3 explicit input list with hashes
    inputs = det_nrb_judge_input_listing(R)
    rec["checks"]["reference_inputs"] = {"n_files": len(inputs), "missing": [k for k, v in inputs.items() if v is None],
                                         "total_bytes": sum(v["bytes"] for v in inputs.values() if v)}
    (out / "reference_inputs.json").write_text(json.dumps(inputs, indent=1))
    if rec["checks"]["reference_inputs"]["missing"]:
        rec["why"].append("REFERENCE_INPUTS_MISSING")
    deck_ok = subprocess.run(["sha256sum", "-c", "--quiet", str(R / "input/deck.sha256")], cwd=R / "input/model",
                             capture_output=True, text=True)
    top_ok = subprocess.run(["sha256sum", "-c", "--quiet", str(R / "input/topion.sha256")], cwd=R / "input/global_atomic",
                            capture_output=True, text=True)
    rec["checks"]["deck_verify_rc"], rec["checks"]["catalog_verify_rc"] = deck_ok.returncode, top_ok.returncode
    if deck_ok.returncode != 0 or top_ok.returncode != 0:
        rec["why"].append("REFERENCE_DECK_OR_CATALOG_VERIFY")
    # 4 exports: planned mapping (dry) from the R' file
    ref_lines = det_nrb_judge_exports_lines(R / "input/resolved_lumina.exports")
    text, mapping = det_nrb_judge_exports_map(ref_lines, R, Path("/NEW_RUN_ROOT"), OUTER)
    rk = {k: v for k, v, _ in ref_lines if k}
    rec["checks"]["exports"] = {"reference_sha256": sha(R / "input/resolved_lumina.exports"), "n_keys": len(rk),
                                "iteration_key": rk.get(ITER_KEY), "planned_delta_keys": sorted(mapping),
                                "nrb": rk.get("LUMINA_BB_NRB"), "capture": rk.get("LUMINA_A210_SPRODUCER_CAPTURE"),
                                "mgpu": rk.get("LUMINA_CMF_FINE_MGPU_DEVICES"),
                                "path_keys_all_under_reference": all(rk.get(k, "").startswith(str(R) + "/") for k in PATH_KEYS)}
    if rk.get(ITER_KEY) != "6" or rk.get("LUMINA_BB_NRB") != "1" or rk.get("LUMINA_A210_SPRODUCER_CAPTURE") != "1" or \
            rk.get("LUMINA_CMF_FINE_MGPU_DEVICES") != "2" or not rec["checks"]["exports"]["path_keys_all_under_reference"]:
        rec["why"].append("REFERENCE_EXPORTS_NOT_REGISTERED")
    ed = {k: (rkE, rk.get(k)) for k, rkE in RATES.det_siiii_rates_exports(E).items() if rk.get(k) != rkE and k not in PATH_KEYS}
    rec["checks"]["baseline_vs_reference_exports_delta"] = ed
    # 5-7 runner
    rs = det_nrb_judge_runner_selftest(a.runner, out)
    rec["checks"]["runner"] = rs
    if rs["rc"] != 0 or not rs["constants_registered"] or rs["profiles_in_runner"] != ["h100x2", "h200x2"]:
        rec["why"].append("RUNNER_SELFTEST_OR_CONSTANTS")
    # selftest linkage
    sj = json.loads((Path(a.selftest) / "selftest.json").read_text())
    rec["checks"]["selftest"] = {"path": str(Path(a.selftest) / "selftest.json"), "failures": sj.get("failures"),
                                 "readout_sha256": sj.get("readout_sha256"), "this_readout": sha(Path(__file__).resolve())}
    if sj.get("failures") != 0 or sj.get("readout_sha256") != sha(Path(__file__).resolve()):
        rec["why"].append("SELFTEST_NOT_THIS_READOUT_OR_FAILED")
    # 3.4 regression on R'
    public = {"spectrum": a.public_spectrum, 14: a.public_si, 16: a.public_s}
    reg = det_nrb_judge_regress(E, R, a.prior_analysis, a.prior_selftest, a.spectral_expectations, a.production_source,
                                public, out, out / "lib")
    rec["checks"]["regression"] = reg
    rec["why"] += reg["why"]
    # cost ceilings as registered
    rec["ceilings"] = {"time_s": TIME_CEILING_S, "time": "21:03:24", "gpu_hours": GPU_HOURS, "gpus": 2, "cpus": 16,
                       "mem_gib": 64, "approved_expression": "about 40 GPU.h", "runs": 1, "outer_iterations": OUTER}
    rec["submit_allowed"] = not rec["why"]
    rec["elapsed_s"] = time.time() - t0
    (out / "offline.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


# ------------------------------------------------------------------ fixtures (direct tests, prereg 8.2)
FX_SHELLS, FX_NS, FX_OUTER, FX_R = SEED.FX_SHELLS, SEED.FX_NS, OUTER, 0.3
FX_TEXP = SEED.FX_TEXP
FX_TARGET = {"Z": (14, 16), "ion": 2, "control": 0, "counts": {14: 6, 16: 2}, "n_keys": 8, "n_levels": 12,
             "shells": FX_SHELLS, "outer": FX_OUTER, "public_time": "19.480",
             "vmid_lumina": {10: 1000.0, 15: 1500.0}, "vmid_public": {10: 1001.0, 15: 1499.0}}
SANDBOX = "SELFTEST_SANDBOX.json"


def det_nrb_judge_target_json(t):
    return {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else list(v) if isinstance(v, tuple) else v)
            for k, v in t.items()}


def det_nrb_judge_target_from_json(j):
    t = dict(j)
    t["Z"], t["shells"] = tuple(t["Z"]), tuple(t["shells"])
    for k in ("counts", "vmid_lumina", "vmid_public"):
        t[k] = {int(a): b for a, b in t[k].items()}
    return t


def det_nrb_judge_fx_inputs(run, outer, nrb=True, binary_text="binary\n"):
    """fixture run inputs (deck, atomic files, profile, binary, identities) and resolved exports of `outer` iterations
    with the seven path keys under the run root."""
    run = Path(run)
    (run / "input").mkdir(parents=True)
    SEED.det_seedmem_scope_fx_deck(run / "input/model")
    (run / "input/global_atomic").mkdir()
    (run / "input/global_atomic/topion_levels.csv").write_text("x\n")
    (run / "input/global_atomic/ionization_reference.csv").write_text("x\n")
    (run / "input/model/cmfgen_sigma_bf.bin").write_bytes(b"sigma")
    (run / "input/model/deposition_cmfgen.csv").write_text("x\n")
    (run / "input/sigma.sha256").write_text(sha(run / "input/model/cmfgen_sigma_bf.bin") + "\n")
    (run / "input/topion.sha256").write_text(f"{sha(run / 'input/global_atomic/topion_levels.csv')}  topion_levels.csv\n")
    (run / "input/fixed_te_profile.txt").write_text("".join(f"{s} {1.0e5 + 100.0 * s:.6E}\n" for s in range(FX_NS)))
    (run / "input/lumina_cuda").write_text(binary_text)
    os.chmod(run / "input/lumina_cuda", 0o750)
    bsha = sha(run / "input/lumina_cuda")
    (run / "input/binary.sha256").write_text(bsha + "\n")
    (run / "input/run_build_identity.json").write_text(json.dumps({"schema": "CMFGEN_TWIN_RUN_BUILD_V1_SELFTEST",
                                                                   "binary": {"sha256": bsha}}))
    ex = ['declare -x LUMINA_ENV_STRICT="1"', 'declare -x LUMINA_PURE_CMFGEN="1"', 'declare -x LUMINA_DET_TRANSACTIONAL="1"',
          f'declare -x {ITER_KEY}="{outer}"', 'declare -x LUMINA_NLTE_COLL_FIX="1"', 'declare -x LUMINA_NLTE_COLL_FLOOR="0"',
          'declare -x LUMINA_SUPER_LEVELS="1"', f'declare -x LUMINA_SUPER_CUTOFF="{SEED.FX_CUTOFF}"',
          'declare -x LUMINA_INNER_BB_SCALE="1.0"', 'declare -x LUMINA_CMF_FINE_MGPU_DEVICES="2"',
          'declare -x LUMINA_ION_POP_DUMP="1"', 'declare -x LUMINA_ION_POP_DUMP_ITER="1"', 'declare -x LUMINA_NLTE_LEVEL_DUMP="1"',
          f'declare -x LUMINA_MODEL_DIR="{run}/input/model"', f'declare -x LUMINA_CMFGEN_SIGMA_BF="{run}/input/model/cmfgen_sigma_bf.bin"',
          f'declare -x LUMINA_DEPOSITION_FILE="{run}/input/model/deposition_cmfgen.csv"',
          f'declare -x LUMINA_FIXED_TE_PROFILE="{run}/input/fixed_te_profile.txt"',
          f'declare -x LUMINA_IONIZATION_REFERENCE_FILE="{run}/input/global_atomic/ionization_reference.csv"',
          f'declare -x LUMINA_TOPION_LEVELS_FILE="{run}/input/global_atomic/topion_levels.csv"',
          f'declare -x LUMINA_PHYSICS_COMPARISON_DIR="{run}/work/physics_comparison"']
    ex += ['declare -x LUMINA_BB_NRB="1"', 'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"'] if nrb else \
        ['declare -x LUMINA_A210_SPRODUCER_CAPTURE="0"']
    (run / "input/resolved_lumina.exports").write_text("\n".join(sorted(ex)) + "\n")
    (run / "STAGE_IDENTITY.txt").write_text(f"schema=NRB_JUDGE_STAGE_V1\nrun={run}\nouter_iterations={outer}\n"
                                            f"run_binary_sha256={bsha}\n")
    return run


def det_nrb_judge_fx_outputs(run, outer, lib, csob, conv=FX_R, ion_series=None, ne_factor=None, spec_scale=1.0,
                             ne_adjust=None):
    """fixture run outputs as the production would leave them after `outer` iterations: W5 calls 1..outer+1 from
    populations that obey the two-level model for lines 0, 1, 4 (others set directly), dumps for populations
    3..outer+2, the observer capture at generation outer+1, status, publication, spectrum, logs and footer."""
    run = Path(run)
    G = det_nrb_judge_gens(outer)
    ex = RATES.det_siiii_rates_exports(run)
    nrb = ex.get("LUMINA_BB_NRB") == "1"
    (run / "work/physics_comparison").mkdir(parents=True, exist_ok=True)
    (run / "work/det_obs_capture").mkdir(parents=True, exist_ok=True)
    deck = SEED.det_seedmem_scope_deck(run / "input/model", (14, 16), 2)
    at = deck["at"]
    lv = pd.read_csv(run / "input/model/levels.csv")
    lumped = int((lv.level_number >= SEED.FX_CUTOFF).sum())
    (run / "stdout.log").write_text(
        f"  Levels: {len(lv)} loaded (super_level column present, super-levels active)\n"
        f"  [ARTIS super-cutoff] K={SEED.FX_CUTOFF}: {lumped} levels lumped -> per-ion explicit + 1 super-level\n"
        f"    Z=14 ion=2: {int(((lv.atomic_number == 14) & (lv.ion_number == 2)).sum())} levels\n"
        f"    Z=16 ion=2: {int(((lv.atomic_number == 16) & (lv.ion_number == 2)).sum())} levels\n"
        f"  [NLTE] Super-levels: ACTIVE ({len(lv)} FL -> {len(lv) - 1} SL across ions)\n"
        f"[CMFGEN] pure deterministic radiation driver: {FX_NS} shells, 1234 bins, 58 rays, {outer} outer iters, 24 ALI/iter, "
        f"t_exp=1.6831e+06 s\n[DET-TRANSACTIONAL] status=COMPLETE\n")
    se = [f"[BB-NRB] iter=0 operator=1 armed=0 captured=0 n_shells={FX_NS} skip_z=none"]
    if nrb:
        se += [f"[BB-NRB] iter={i} operator=2 armed=1 captured=1 n_shells={FX_NS} skip_z=none" for i in range(outer)]
    (run / "stderr.log").write_text("\n".join(se) + "\n")
    f = lambda g: 1.0 + conv ** g                                                       # noqa: E731
    nf = ne_factor or (lambda g: 5.0e13 * f(g))
    Te = np.array([1.0e5 + 100.0 * s for s in range(FX_NS)])
    ne = {g: np.array([nf(g) * (1.0 + 0.001 * s) for s in range(FX_NS)]) for g in range(2, G["g_last"] + 1)}
    if ne_adjust:
        ne_adjust(ne)
    if ion_series is None:
        ion_series = {(Z, q): {g: 1.0e6 * (1 + Z / 100 + q) * f(g) for g in G["pops"]} for Z, q in ION_QTY}
    pops, rows = {}, {}
    model_lines = (0, 1, 4)
    for s in FX_SHELLS:
        for g in range(2, G["g_obs"] + 1):
            for Z, n0 in ((14, 1.0e3), (16, 5.0e2)):
                pops[(Z, 0, s, g)] = n0 * (1.0 + 0.002 * s)
            pops[(14, 1, s, g)] = 2.0e3 * f(g)
            pops[(14, 2, s, g)] = 9.0e2 * f(g)
            pops[(16, 2, s, g)] = 5.0e1 * f(g)
            pops[(14, 5, s, g)] = 3.0e2 * f(g)
            for line in model_lines:
                a = at[line]
                if g == 2:
                    J = (0.04 if line != 4 else 0.03) * f(1)
                    rad = (I.x(a["B_lu"] * J), I.x(a["B_ul"] * J), I.x(a["A_ul"]))
                else:
                    r = rows[(line, s, g - 1)]
                    if nrb:
                        rc, b2, Ra, Rs, Rp = lib.bracket(r["beta"] * r["J_c"], r["L_l"], r["tau_eff"], r["Jbar"], 0,
                                                         a["B_lu"], a["B_ul"], a["A_ul"])
                        rad = (I.x(Ra), I.x(Rs), I.x(Rp))
                    else:
                        rad = (I.x(a["B_lu"] * r["Jbar"]), I.x(a["B_ul"] * r["Jbar"]), I.x(a["A_ul"]))
                Cu, Cd = RATES.det_siiii_rates_coll(float(ne[g][s]), float(Te[s]), a["E_l"], a["E_u"], a["g_l"], a["g_u"],
                                                    a["f_lu"])
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
                eta = n_u * a["A_ul"] * SEED.H_PLANCK * a["nu"] / SEED.FOUR_PI
                L = eta * (SEED.C_LIGHT * FX_TEXP / a["nu"]) * q
                Jc = 0.05 * f(g) * (1.0 + 0.1 * line) * (1.0 + 0.01 * s)
                rows[(line, s, g)] = {"J_c": Jc, "beta": beta, "L_l": L, "Jbar": beta * Jc + L, "tau_eff": tau}
    with open(run / "work/w5_lines.csv", "w") as fh:
        fh.write("call,operator,rad_generation,material_generation,shell,line_id,Z,ion,nu_hz,J_c,J_c_err_upper,beta,"
                 "L_l,Jbar,Jbar_err_upper,tau_eff,srce_chk,status\n")
        for call in range(1, G["g_obs"] + 1):
            for line, a in sorted(at.items()):
                for s in FX_SHELLS:
                    if not (RATES.NU_MIN <= a["nu"] <= RATES.NU_MAX):
                        op = "INIT_SHARED_GAUSSIAN,1,0" if call == 1 else f"CMFGEN_NONOVERLAP_SOBOLEV,{call},{call}"
                        fh.write(f"{call},{op},{s},{line},{a['Z']},2,{a['nu']!r},,,,,,,,,OUTSIDE_FINE_SUPPORT\n")
                        continue
                    if call == 1:
                        J1 = (0.04 if line != 4 else 0.03) * f(1)
                        fh.write(f"1,INIT_SHARED_GAUSSIAN,1,0,{s},{line},{a['Z']},2,{a['nu']!r},NOT_APPLICABLE,"
                                 f"NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,{J1!r},0,NOT_APPLICABLE,NOT_APPLICABLE,"
                                 f"OK_INIT_GAUSSIAN_PROFILE_MEAN\n")
                        continue
                    r = rows[(line, s, call)]
                    fh.write(f"{call},CMFGEN_NONOVERLAP_SOBOLEV,{call},{call},{s},{line},{a['Z']},2,{a['nu']!r},"
                             f"{r['J_c']!r},0,{r['beta']!r},{r['L_l']!r},{r['Jbar']!r},0,{r['tau_eff']!r},0,OK\n")
    for k in range(outer):
        g = k + 3
        (run / f"work/physics_comparison/physics_DET_iter{k:04d}.manifest.json").write_text(json.dumps(
            {"schema": "LUMINA_PHYSICS_COMPARISON_V1", "transaction_status": "COMMITTED", "te_lane": "FIXED_T",
             "iteration": k, "population_generation": g, "radiation_generation": g - 1, "te_generation": g - 1,
             "opacity_generation": 2 * g - 3, "emissivity_generation": 2 * g - 3,
             "shell_file": f"physics_DET_iter{k:04d}.shell.csv", "ion_file": f"physics_DET_iter{k:04d}.ion.csv"}))
        with open(run / f"work/physics_comparison/physics_DET_iter{k:04d}.shell.csv", "w") as fh:
            fh.write("shell_id,T_e_K,n_e_cm3\n" + "".join(f"{s},{float(Te[s])!r},{float(ne[g][s])!r}\n" for s in range(FX_NS)))
        with open(run / f"work/physics_comparison/physics_DET_iter{k:04d}.ion.csv", "w") as fh:
            fh.write("population_generation,shell_id," + ",".join(
                f"Z{Z}_{c}" for Z in (14, 16) for c in ("element_total_cm3", "stage_q1_cm3", "stage_q2_cm3", "nlte_q1_cm3",
                                                         "nlte_q2_cm3")) + "\n")
            for s in range(FX_NS):
                vals = []
                for Z in (14, 16):
                    q1, q2 = ion_series[(Z, 1)][g], ion_series[(Z, 2)][g]
                    vals += [4.0e6 * (1 + Z / 100), 0.9 * q1, 1.1 * q2, q1, q2]
                fh.write(f"{g},{s}," + ",".join(f"{float(v)!r}" for v in vals) + "\n")
    capd = run / "work/det_obs_capture"
    recs = []
    g_obs = G["g_obs"]
    for line, a in sorted(at.items()):
        if not (RATES.NU_MIN <= a["nu"] <= RATES.NU_MAX):
            continue
        for s in FX_SHELLS:
            n_l, n_u = pops[(a["Z"], a["lo"], s, g_obs)], pops[(a["Z"], a["up"], s, g_obs)]
            tau, S, *_ = POP.prod_tau_S(n_l, n_u, a["g_l"], a["g_u"], a["f_lu"], a["lam_cm"], FX_TEXP)
            recs.append((line, s, tau, S, 0.0, 1))
    np.array(sorted(recs), dtype=POP.REC).tofile(capd / "lines.bin")
    ne[g_obs].astype("<f8").tofile(capd / "ne.f64")
    (capd / "meta.txt").write_text(f"time_explosion={FX_TEXP.hex()}\nT_inner={(10020.0).hex()}\ninner_bb_scale={(1.0).hex()}\n"
                                   f"material_generation={g_obs}\nrad_generation={g_obs}\niter={G['iter_last']}\n")
    (capd / "manifest.json").write_text(json.dumps({"generations": {"material": g_obs, "nlte_population_committed": g_obs,
                                                                   "radiation": g_obs, "iter": G["iter_last"],
                                                                   "tau_required": 2 * g_obs, "tau_computed": 2 * g_obs},
                                                   "transaction": "COMMITTED",
                                                   "census": {"mapped_B_fallback": 0, "consumed": len(recs), "mapped": len(recs)},
                                                   "files": {"lines.bin": {"sha256": sha(capd / "lines.bin")}}}))
    (run / "work/det_obs_status.json").write_text(json.dumps({"schema": "DET_OBS_STATUS_V1", "det_obs_state": "SUCCESS",
                                                               "rad_generation": g_obs, "iter": G["iter_last"]}))
    (run / "work/cmfgen_final_publication.json").write_text(json.dumps({"schema": "CMFGEN_FINAL_PUBLICATION_V1", "complete": True,
                                                                         "population_generation": G["g_last"],
                                                                         "required_population_generation": G["g_last"]}))
    lam = np.linspace(2000.0, 10500.0, 1200)
    flux = spec_scale * 1.0e40 * (lam / 5000.0) ** -2.5 * np.exp(-((lam - 4500.0) / 6000.0) ** 2)
    with open(run / "work/lumina_spectrum_det_obs.csv", "w") as fh:
        fh.write("wavelength_angstrom,flux\n" + "".join(f"{float(x)!r},{float(y)!r}\n" for x, y in zip(lam, flux)))
    bsha = sha(run / "input/lumina_cuda")
    (run / "RUN_FOOTER.txt").write_text(f"schema=NRB_JUDGE_RUN_V1\njob_id=1\nouter_iterations={outer}\nbinary_sha256={bsha}\n"
                                        f"elapsed_seconds=10\nmodel_rc=0\nterminal_state=SUCCESS\n")
    (run / "fixture_truth.json").write_text(json.dumps({f"{k[0]}:{k[1]}:{k[2]}:{k[3]}": v for k, v in pops.items()}))
    return run


def det_nrb_judge_fx_public(base, run_N, target, pub_factor=None, spec_scale=1.0):
    """public fixtures: ion-fraction files whose 19.480 block carries the N generation-g_obs fractions (times a
    per-cell factor) and a 17.710 decoy block, a spectrum file with the N spectrum at 19.48 d, an expectations file."""
    base = Path(base)
    G = det_nrb_judge_gens(target["outer"])
    dump, _ = det_nrb_judge_ion_dump(run_N, G["g_obs"], target["shells"])
    files = {}
    for Z, nm in ((14, "si"), (16, "s")):
        lines = ["#NTIMES:     2", "#NSTAGES:     5", "#TIMES[d]:    17.710   19.480", "#"]
        for t in ("17.710", "19.480"):
            lines += [f"#TIME:    {t}", "#NVEL: 4", f"#vel_mid[km/s]           {nm}0           {nm}1           {nm}2           "
                      f"{nm}3           {nm}4"]
            vrows = [(target["vmid_public"][s], s) for s in target["shells"]] + [(1200.0, None), (1700.0, None)]
            for v, s in sorted(vrows):
                if s is None:
                    fr = [1e-99, 0.3, 0.6, 0.1, 1e-99]
                else:
                    r = dump["rows"][s]
                    f1 = r[f"Z{Z}_nlte_q1_cm3"] / r[f"Z{Z}_element_total_cm3"]
                    f2 = r[f"Z{Z}_nlte_q2_cm3"] / r[f"Z{Z}_element_total_cm3"]
                    k = 3.0 if t != "19.480" else (pub_factor or {}).get((Z, s), 1.0)
                    fr = [1e-99, max(f1 * k, 1e-30), max(f2 * k, 1e-30), 1e-3, 1e-99]
                lines.append(f"    {v:.6E}   " + "   ".join(f"{x:.4E}" for x in fr))
        p = base / f"ionfrac_{nm}_fx.txt"
        p.write_text("\n".join(lines) + "\n")
        files[Z] = str(p)
    spec = REP.det_siiii_repair_spec(Path(run_N) / "work/lumina_spectrum_det_obs.csv")
    lines = ["#NTIMES:  2", f"#NWAVE: {len(spec)}", "#TIMES[d]: 1.77100e+01 1.94800e+01", "#"]
    for lam, fl in spec:
        lines.append(f"{lam:.6f} {fl * 0.5:.6e} {fl * spec_scale:.6e}")
    (base / "spectra_fx.txt").write_text("\n".join(lines) + "\n")
    files["spectrum"] = str(base / "spectra_fx.txt")
    exp = {"B-T": {"secondary_spectral_expectation": {"targets": {
        "abs_d_NUV_max": {"lo": 0.1155, "mid": 0.11551623544341516, "hi": 0.1156},
        "abs_d_UV2500_max": {"lo": 0.3065, "mid": 0.3065384957773474, "hi": 0.3066},
        "abs_d_keep4_max": 0.05, "R8_below": {"lo": 0.2818, "mid": 0.2818388065975096, "hi": 0.2819}}}}}
    (base / "expectations.json").write_text(json.dumps(exp))
    files["expectations"] = str(base / "expectations.json")
    return files


def det_nrb_judge_fx_prior(base, lib_rec, keys, bad_source=False):
    """fixture 'prior analysis' (A) and 'prior selftest' carrying the inheritance identity of this library build."""
    A = Path(base) / "prior/analysis"
    A.mkdir(parents=True)
    (A / "sample.json").write_text(json.dumps({"schema": "DET_SEEDMEM_SCOPE_SAMPLE_V2", "n_keys": len(keys),
                                               "keys": [list(k) for k in keys]}))
    (A / "result.json").write_text(json.dumps({"validity": "VALID", "codes": {"H-LOCAL_record": {
        "code": 0, "state": "SURVIVES", "cells": 2550, "eligible_refuted_cells": 0, "x_cells": 0}}}))
    src_sha = dict(lib_rec["sources_sha256"])
    if bad_source:
        src_sha["line_net_rate.c"] = "0" * 64
    (A.parent / "manifest.json").write_text(json.dumps({"lib": {"sources_sha256": src_sha}}))
    st = Path(base) / "prior/selftest"
    st.mkdir(parents=True)
    (st / "selftest.json").write_text(json.dumps({"readout_sha256": "fixture", "failures": 0, "cases": [
        {"case": "cross substitution fixture", "bundle": "7 cross", "pass": True}]}))
    return str(A), str(st)


def det_nrb_judge_cli(args):
    """the readout as a separate process (the real CLI and its exit status)."""
    p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve())] + [str(x) for x in args],
                       capture_output=True, text=True)
    return p.returncode, p.stdout[-3000:], p.stderr[-3000:]


def det_nrb_judge_fx(base, lib, csob, lib_rec, runner, conv=FX_R, ion_series=None, ne_factor=None, pub_factor=None,
                     spec_scale=1.0, mutate_N=None, mutate_R=None, prior_bad_source=False, profile="h200x2", ne_adjust=None):
    """the full fixture set: R (6 iterations, NRB) and E (6, off) runs, N staged from R by the stage CLI and given the
    outputs of the registered 12 iterations, the stage's preflight record, public files, prior analysis / selftest."""
    base = Path(base)
    base.mkdir(parents=True)
    R = det_nrb_judge_fx_inputs(base / "R", 6, True)
    det_nrb_judge_fx_outputs(R, 6, lib, csob, conv)
    E = det_nrb_judge_fx_inputs(base / "E", 6, False)
    det_nrb_judge_fx_outputs(E, 6, lib, csob, conv)
    pre = base / "pre"
    pre.mkdir()
    (pre / "offline.json").write_text(json.dumps({"schema": "DET_NRB_JUDGE_OFFLINE_V1", "submit_allowed": True,
                                                  "fixture": True}))
    (pre / "reference_inputs.json").write_text(json.dumps(det_nrb_judge_input_listing(R)))
    N = base / "N"
    rc, so, se = det_nrb_judge_cli(["stage", "--reference-run", R, "--preflight", pre, "--gpu-profile", profile,
                                    "--runner", runner, "--run", N])
    if rc != 0:
        raise RuntimeError(f"FIXTURE_STAGE_FAILED rc={rc} {se[-500:]}")
    det_nrb_judge_fx_outputs(N, FX_OUTER, lib, csob, conv, ion_series, ne_factor, spec_scale, ne_adjust)
    keys = sorted((16 if line == 4 else 14, line, s) for line in (0, 1, 3, 4) for s in FX_SHELLS)
    # the public files come from the unedited N; an edit applied afterwards is a difference from the public run
    public = det_nrb_judge_fx_public(base, N, FX_TARGET, pub_factor, spec_scale)
    if mutate_R:
        mutate_R(R)
    if mutate_N:
        mutate_N(N)
    A, st = det_nrb_judge_fx_prior(base, lib_rec, keys, prior_bad_source)
    return {"N": N, "R": R, "E": E, "pre": pre, "public": public, "A": A, "st": st, "keys": keys}


def det_nrb_judge_w5_edit(run, fn):
    """rewrite the fixture W5 ledger row by row; fn(fields) returns the new field list or None to drop the row."""
    p = Path(run) / "work/w5_lines.csv"
    lines = p.read_text().split("\n")
    out = [lines[0]]
    for ln in lines[1:]:
        if not ln:
            continue
        f = fn(ln.split(","))
        if f is not None:
            out.append(",".join(f))
    p.write_text("\n".join(out) + "\n")


def det_nrb_judge_selftest_cases(base, src, runner, prior_selftest):
    res = []
    base = Path(base)
    lib_rec = SEED.det_seedmem_scope_build_lib(src, base / "lib")
    lib = SEED.det_seedmem_scope_Lib(lib_rec["lib"])
    csob = SEED.det_seedmem_scope_csob(src)
    G = det_nrb_judge_gens(FX_OUTER)

    def case(name, bundle, got, want, fault=None, runs=(), kind="NEW"):
        ok = got == want
        res.append({"case": name, "bundle": bundle, "kind": kind, "fault": fault, "pass": bool(ok), "got": str(got)[:900],
                    "want": str(want)[:900],
                    "runs": [{"case_dir": r_.get("dir"), "cli_rc": r_.get("rc"), "validity": r_.get("validity"),
                              "injection": r_.get("injection"), "reasons": [str(x)[:160] for x in (r_.get("reasons") or [])[:6]]}
                             for r_ in runs]})
        print(f"[{'PASS' if ok else 'FAIL'}] ({bundle}) {name}: got {str(got)[:260]} want {str(want)[:260]}", flush=True)

    def analyze(name, fault=None, target=FX_TARGET, injection=None, **fxkw):
        """build the fixtures, write the selftest sandbox (fault and fixture target), run the analyze CLI."""
        d = base / name
        fx = det_nrb_judge_fx(d / "in", lib, csob, lib_rec, runner, **fxkw)
        (d / SANDBOX).write_text(json.dumps({"schema": "DET_NRB_JUDGE_SELFTEST_SANDBOX_V1", "fault": fault,
                                             "target": det_nrb_judge_target_json(target)}))
        pub = fx["public"]
        rc, so, se = det_nrb_judge_cli(
            ["selftest-analyze", "--prereg", P / PREREG, "--run", fx["N"], "--baseline-run", fx["E"], "--reference-run", fx["R"],
             "--prior-analysis", fx["A"], "--prior-selftest", fx["st"], "--spectral-expectations", pub["expectations"],
             "--production-source", src, "--public-spectrum", pub["spectrum"], "--public-si", pub[14], "--public-s", pub[16],
             "--selftest", d / "no_selftest", "--out", d / "out"])
        rp = d / "out/analysis/result.json"
        r = json.loads(rp.read_text()) if rp.exists() else {"validity": "NO_RESULT", "stderr": se[-800:]}
        r.update({"rc": rc, "dir": str(d), "injection": injection or fault, "_out": d / "out/analysis", "_fx": fx})
        return r

    def has(r, token):
        return r.get("validity") == "X" and any(token in str(w) for w in (r.get("reasons") or []))

    def code(r, k):
        return ((r.get("codes") or {}).get(k) or {}).get("code")

    def dec(r):
        p_ = r["_out"] / "decision.json"
        return json.loads(p_.read_text()) if p_.exists() else {}

    def csv(r, nm):
        return pd.read_csv(r["_out"] / nm)

    # ---- 0 clean (registered 12 iterations: observation 13, last update 14)
    r0 = analyze("clean")
    p1a = csv(r0, "p1_rates.csv")
    p1b = csv(r0, "p1_source_1206.csv")
    st0 = csv(r0, "stationarity.csv")
    io0 = csv(r0, "ion_fraction.csv")
    case("clean (12 iterations): CLI rc 0, VALID; P1(a) 0 on 8 keys x calls 2-13 (96 cells NRB_BRACKET); fixed recall 24 "
         "cells invariant and H-LOCAL inherited; P1(b) core 3-7 and extension 8-13 survive; P2 four axes small in the "
         "terminal window (12,13) -> P2_SMALL_CHANGE; public ions and spectrum survive", "0 clean",
         (r0["rc"], r0.get("validity"), code(r0, "P1A"), int((p1a[p1a.kind != "RECALL"].branch == "NRB_BRACKET").sum()),
          code(r0, "P1A_LOCAL"), (r0.get("P1A_recall") or {}).get("cells"), code(r0, "P1B_CORE"), code(r0, "P1B_EXT"),
          r0.get("P2_disposition"), [code(r0, f"P2:{ax}") for ax in ("ion", "ne", "jbar", "level")], code(r0, "IONS:14"),
          code(r0, "IONS:16"), code(r0, "SPECTRUM")),
         (0, "VALID", 0, 96, 0, 24, 0, 0, "P2_SMALL_CHANGE", [0, 0, 0, 0], 0, 0, 0), runs=[r0])
    bts = (r0.get("bytes_vs_reference") or {}).get("summary", {})
    case("generation linkage (clean, 12 iterations): INIT call 1 excluded, ordinary calls 2-13 consuming material 3-14, "
         "P1(b) rows 3-13 with material = call, ion / n_e changes 4-14 and J-bar / level 4-13, terminal window (12,13), "
         "last material update 14 kept apart, ion fractions read at generation 13 from iter0010, the capture counters "
         "(tau 26 = 26) recorded, and the generation 3-7 dumps byte-identical to the reference", "3 generation linkage",
         (sorted(set(p1a[p1a.kind != "RECALL"].call.astype(int))), sorted(set(p1b.call.astype(int))),
          bool((p1b.material_generation == p1b.call).all()), sorted(set(st0[st0.axis == "ion"].generation.astype(int))),
          sorted(set(st0[st0.axis == "jbar"].generation.astype(int))), (r0.get("codes") or {}).get("P2:ion", {}).get("terminal_window"),
          ((r0.get("P2_last_material_update") or {}).get("ion") or {}).get("generation"),
          sorted(set(io0[(io0.run == "N") & (io0.ion != "III/II")].generation.astype(int))),
          (r0.get("run_counters") or {}).get("capture_tau_required"), bts.get("identical")),
         (G["calls"], G["p1b_core"] + G["p1b_ext"], True, list(range(4, 15)), list(range(4, 14)), "12,13", 14, [13], 26, 10),
         runs=[r0])
    # ---- 1 single change and inputs (analyze CLI on staged N with mutated inputs)
    def ex_edit(key, val, add=False):
        def m(N):
            p_ = N / "input/resolved_lumina.exports"
            t = [ln for ln in p_.read_text().split("\n") if ln]
            if add:
                t.append(f'declare -x {key}="{val}"')
            else:
                t = [f'declare -x {key}="{val}"' if ln.startswith(f"declare -x {key}=") else ln for ln in t]
            p_.write_text("\n".join(t) + "\n")
        return m
    r1 = {nm: analyze(f"ex_{nm}", injection=nm, mutate_N=m) for nm, m in (
        ("nrb_off", ex_edit("LUMINA_BB_NRB", "0")), ("capture_off", ex_edit("LUMINA_A210_SPRODUCER_CAPTURE", "0")),
        ("iter6", ex_edit(ITER_KEY, "6")), ("iter13", ex_edit(ITER_KEY, "13")),
        ("extra_env", ex_edit("LUMINA_NRB_EXTRA", "1", add=True)))}
    case("single change (analyze CLI): N exports with NRB off, producer capture off, 6 or 13 iterations, or an extra env "
         "key are refused (G-LINK, rc 2) before any science code", "1 single change",
         {k: (v["rc"], has(v, "N:EXPORTS")) for k, v in r1.items()},
         {k: (2, True) for k in r1}, fault="exports mutations", runs=list(r1.values()))

    def wr(rel, data):
        def m(N):
            (N / rel).write_text(data) if isinstance(data, str) else (N / rel).write_bytes(data)
        return m
    r1b = {"fixed_te_profile": analyze("in_profile", injection="fixed_te_profile.txt edited",
                                       mutate_N=lambda N: (N / "input/fixed_te_profile.txt").write_text(
                                           (N / "input/fixed_te_profile.txt").read_text().replace("1.000000E+05", "1.000001E+05"))),
           "global_atomic": analyze("in_catalog", injection="global_atomic/topion_levels.csv edited",
                                    mutate_N=wr("input/global_atomic/topion_levels.csv", "y\n")),
           "binary": analyze("in_binary", injection="lumina_cuda edited", mutate_N=wr("input/lumina_cuda", "tampered\n"))}
    case("input identity (analyze CLI): an edited FIXED_T profile, an edited global atomic file (non-binary inputs) and an "
         "edited binary differ from the reference contents recorded at preflight and are refused (rc 2)", "1 single change",
         {k: (v["rc"], has(v, "INPUT_IDENTITY")) for k, v in r1b.items()} | {"binary_chain": has(r1b["binary"], "LINK_BINARY_IDENTITY")},
         {k: (2, True) for k in r1b} | {"binary_chain": True}, fault="input edits", runs=list(r1b.values()))
    # stage CLI behaviour
    fx_s = r0["_fx"]
    n2 = base / "stage_cli/NRB/run1"
    rcA, soA, seA = det_nrb_judge_cli(["stage", "--reference-run", fx_s["R"], "--preflight", fx_s["pre"], "--gpu-profile",
                                       "h100x2", "--runner", runner, "--run", n2])
    rcB, soB, seB = det_nrb_judge_cli(["stage", "--reference-run", fx_s["R"], "--preflight", fx_s["pre"], "--gpu-profile",
                                       "h200x2", "--runner", runner, "--run", n2])
    bad_pre = base / "stage_cli/pre_bad"
    bad_pre.mkdir(parents=True)
    (bad_pre / "offline.json").write_text(json.dumps({"submit_allowed": False}))
    rcC, soC, seC = det_nrb_judge_cli(["stage", "--reference-run", fx_s["R"], "--preflight", bad_pre, "--gpu-profile",
                                       "h200x2", "--runner", runner, "--run", base / "stage_cli/NRB/run2"])
    rcD, soD, seD = det_nrb_judge_cli(["stage", "--reference-run", fx_s["R"], "--preflight", fx_s["pre"], "--gpu-profile",
                                       "a100x2", "--runner", runner, "--run", base / "stage_cli/NRB/run3"])
    js = (n2 / "job.slurm").read_text() if (n2 / "job.slurm").exists() else ""
    last = lambda t: (t.strip().split("\n") or [""])[-1][-200:]                     # noqa: E731
    stage_runs = [{"dir": str(n2), "rc": rcA, "injection": "h100x2", "validity": "STAGED", "reasons": [last(soA)]},
                  {"dir": str(n2), "rc": rcB, "injection": "existing run root", "validity": "REFUSED", "reasons": [last(seB)]},
                  {"dir": str(base / "stage_cli/NRB/run2"), "rc": rcC, "injection": "preflight submit_allowed false",
                   "validity": "REFUSED", "reasons": [last(seC)]},
                  {"dir": str(base / "stage_cli/NRB/run3"), "rc": rcD, "injection": "a100x2 profile", "validity": "REFUSED",
                   "reasons": [last(seD)]}]
    case("stage (CLI): copies the reference inputs with sha verification, writes the iteration-12 exports and the identity "
         "files; h100x2 rewrites profile, partition and gres together; an existing run root, a preflight without "
         "submit_allowed and the a100 profile are refused (real rc and the refusal line kept, nothing written for the "
         "refused roots)", "1 single change",
         (rcA, "readonly GPU_PROFILE=h100x2" in js, "#SBATCH --partition=h100" in js, "#SBATCH --gres=gpu:H100:2" in js,
          (n2 / "OWNER.txt").exists() and (n2 / "SUBMIT_COMMAND.txt").exists() and (n2 / "STAGE_FILES.json").exists(),
          (rcB, "RUN_EXISTS" in seB), (rcC, "PREFLIGHT_NOT_PASSED" in seC, (base / "stage_cli/NRB/run2").exists()),
          (rcD, "GPU_PROFILE_INVALID" in seD, (base / "stage_cli/NRB/run3").exists())),
         (0, True, True, True, True, (1, True), (1, True, False), (1, True, False)), runs=stage_runs)
    # ---- 2 run count and resources: the GPU runner's own selftest (bash, real rc)
    rs = det_nrb_judge_runner_selftest(runner, base)
    want_cases = {"rerun_reject", "h200x2_one_gpu", "a100_profile_reject", "time_limit_over_ceiling", "iterations_6",
                  "iterations_13", "nrb_off", "requeued_reject", "h100x2_low_memory", "armed_markers_11",
                  "capture_generation_7", "last_population_as_observed"}
    got_cases = {c_["case"] for c_ in rs["cases"] if c_["expected"] == "fail" and c_["rc"] != 0}
    case("run count / resources (GPU runner bash selftest, real rc): re-entry, GPU count / kind / memory, the a100 profile, "
         "a time limit over the ceiling, requeue, 6 / 13 iterations and NRB off are refused before the model; 11 armed "
         "markers, a capture at generation 7 and the last population reported as observed fail the post-checks; "
         "constants registered (12 iterations, 75804 s); profiles h200x2 / h100x2 only", "2 run count",
         (rs["rc"], sorted(want_cases - got_cases), rs["constants_registered"], rs["profiles_in_runner"], rs["positive"] >= 3),
         (0, [], True, ["h100x2", "h200x2"], True))
    res[-1]["runner_selftest"] = {k: v for k, v in rs.items() if k != "cases"} | {"cases": rs["cases"]}
    # ---- 3 generation linkage faults (implementation faults through the sandbox; data faults through edited files)
    r3 = {fl: analyze(f"gen_{fl}", fault=fl) for fl in ("beta_current", "material_plus1", "capture_fixed7", "last_as_observed")}

    def relabel(N):
        m_ = N / "work/physics_comparison/physics_DET_iter0010.manifest.json"
        j_ = json.loads(m_.read_text())
        j_["population_generation"] = 15
        m_.write_text(json.dumps(j_))

    def meta12(N):
        m_ = N / "work/det_obs_capture/meta.txt"
        m_.write_text(m_.read_text().replace("material_generation=13", "material_generation=12"))
    r3["dump_relabel"] = analyze("gen_dump_relabel", injection="iter0010 manifest relabelled 15", mutate_N=relabel)
    r3["capture_meta_12"] = analyze("gen_capture_meta_12", injection="capture meta material 12", mutate_N=meta12)
    d3 = {k: [row["condition"] for row in dec(v).get("rows", [])] for k, v in r3.items()}
    case("generation linkage faults (analyze CLI, 12 iterations): the current call's radiation for the prediction and "
         "material shifted by one make P1(b) rows X -> scope X, run X (rc 2) and an X row in decision.json; the capture "
         "pinned to generation 7, the last population 14 used as the observed 13, a relabelled observation dump and a "
         "capture meta generation 12 are refused at the link (rc 2)", "3 generation linkage",
         (r3["beta_current"]["rc"], code(r3["beta_current"], "P1B_CORE"), "P1B_CORE X" in d3["beta_current"],
          r3["material_plus1"]["rc"], code(r3["material_plus1"], "P1B_CORE"), "P1B_CORE X" in d3["material_plus1"],
          (r3["capture_fixed7"]["rc"], has(r3["capture_fixed7"], "LINK_CAPTURE")),
          (r3["last_as_observed"]["rc"], has(r3["last_as_observed"], "LINK_NE_DUMP_OBS_NOT_CAPTURE") or
           has(r3["last_as_observed"], "DUMP_GENERATION_FILLED")),
          (r3["dump_relabel"]["rc"], has(r3["dump_relabel"], "DUMP_GENERATIONS") or has(r3["dump_relabel"], "LINK_NE_DUMP")),
          (r3["capture_meta_12"]["rc"], has(r3["capture_meta_12"], "LINK_CAPTURE_META_GENERATION"))),
         (2, "X", True, 2, "X", True, (2, True), (2, True), (2, True), (2, True)),
         fault="beta_current|material_plus1|capture_fixed7|last_as_observed|dump_relabel|capture_meta_12", runs=list(r3.values()))
    # ---- 4 P2 terminal window, descriptors, denominators, final code
    def ser(fn):
        return {(Z, q): {g: 1.0e6 * (1 + Z / 100 + q) * fn(g, Z, q) for g in G["pops"]} for Z, q in ION_QTY}
    past_only = ser(lambda g, Z, q: 1.0 if g in (4, 5, 6) else 1.0 + 0.05 * ((g % 2) + 1))
    one_small = ser(lambda g, Z, q: 1.0 if g <= 12 else 1.2)
    r4 = {"past_only": analyze("p2_past_only", ion_series=past_only),
          "single_change": analyze("p2_single", fault="single_change", ion_series=one_small),
          "past_window": analyze("p2_past", fault="past_window", ion_series=past_only),
          "mixed_windows": analyze("p2_mixed", fault="mixed_windows", ion_series=past_only),
          "p2_code_zero": analyze("p2_code_zero", fault="p2_code_zero", ion_series=past_only),
          "descriptor_flip": analyze("p2_desc_flip", fault="decision_descriptor_flip", ion_series=past_only)}
    c4 = (r4["past_only"].get("codes") or {}).get("P2:ion", {})
    case("P2 terminal window (analyze CLI): the ion axis small only in the past windows is NOT_CONVERGED (rc 0, VALID) with "
         "the past window recorded and the decision row carrying the descriptors; accepting one small terminal change, a "
         "past window, mixed windows or an axis code replaced by 0 is refused by the recomputation, and a descriptor "
         "changed in the stored decision.json is refused by the stored-output check (X, rc 2, P2 INADMISSIBLE)",
         "4 P2 terminal window",
         (r4["past_only"]["rc"], c4.get("state"), c4.get("any_past_window_small"), r4["past_only"].get("P2_disposition"),
          any("descriptors" in row for row in dec(r4["past_only"]).get("rows", [])),
          {k: (v["rc"], has(v, "P2_CODE_RECOMPUTE") or has(v, "P2_STORED") or has(v, "STORED_DECISION"),
               dec(v).get("p2_disposition"))
           for k, v in r4.items() if k != "past_only"}),
         (0, "REFUTED", True, "NOT_CONVERGED", True,
          {k: (2, True, "INADMISSIBLE") for k in r4 if k != "past_only"}),
         fault="single_change|past_window|mixed_windows|p2_code_zero|decision_descriptor_flip", runs=list(r4.values()))
    end_higher = ser(lambda g, Z, q: 1.0 + 0.2 * (g - 3) * (g % 2))
    end_equal = ser(lambda g, Z, q: 0.0 if g in (3, 4, 12, 13) else 1.0 + 0.1 * g)
    r4d = {"END_HIGHER": analyze("p2_end_higher", ion_series=end_higher),
           "END_EQUAL": analyze("p2_end_equal", ion_series=end_equal),
           "END_UNRESOLVED": analyze("p2_end_unresolved", ne_factor=lambda g: 5.0e10 * 2.0 ** g)}
    st_eq = csv(r4d["END_EQUAL"], "stationarity.csv")
    case("P2 descriptors (analyze CLI): clean axes END_LOWER; an ion series rising at the end END_HIGHER; exact structural "
         "zeros at generations 4 and 13 (M = 0 at both ends, 8 structural zeros counted) END_EQUAL; an n_e series with a "
         "constant factor 2 (identical end intervals) END_UNRESOLVED; the descriptors stored in result.json and "
         "decision.json agree with the recomputation from stationarity.csv", "4 P2 terminal window",
         ([((r0.get("P2_descriptors") or {}).get(ax) or {}).get("code") for ax in ("ion", "ne", "jbar", "level")],
          ((r4d["END_HIGHER"].get("P2_descriptors") or {}).get("ion") or {}).get("code"),
          ((r4d["END_EQUAL"].get("P2_descriptors") or {}).get("ion") or {}).get("code"),
          int(st_eq[(st_eq.axis == "ion") & (st_eq.generation == 4)].structural_zero.iloc[0]),
          ((r4d["END_UNRESOLVED"].get("P2_descriptors") or {}).get("ne") or {}).get("code"),
          [v.get("validity") for v in r4d.values()]),
         (["END_LOWER"] * 4, "END_HIGHER", "END_EQUAL", 8, "END_UNRESOLVED", ["VALID"] * 3), runs=list(r4d.values()))

    def drop_term(line, shell):
        def m(N):
            det_nrb_judge_w5_edit(N, lambda f: None if (f[0] == str(G["g_obs"]) and f[4] == str(shell) and f[5] == str(line))
                                  else f)
        return m

    def drop_and_bump(N):
        drop_term(1, 10)(N)

        def bump(f):
            if f[0] == str(G["outer"]) and f[5] == "4" and f[4] == "10":
                L = float(f[12]) * 1.2
                f[12], f[13] = repr(L), repr(float(f[11]) * float(f[9]) + L)
            return f
        det_nrb_judge_w5_edit(N, bump)
    zero_cross = ser(lambda g, Z, q: 0.0 if (Z == 14 and q == 1 and g == 13) else 1.0)
    r4m = {"missing": analyze("p2_missing", injection="terminal W5 row of line 1 / shell 10 removed", mutate_N=drop_term(1, 10)),
           "missing_with_counter": analyze("p2_missing_counter", injection="row removed + line 4 / shell 10 L x1.2 at call 12",
                                           mutate_N=drop_and_bump),
           "zero_cross": analyze("p2_zero_cross", ion_series=zero_cross),
           "drop_large": analyze("p2_drop_large", fault="drop_large", ion_series=past_only),
           "drop_zero_cross": analyze("p2_drop_zero", fault="drop_zero_cross", ion_series=zero_cross)}
    stm = csv(r4m["missing"], "stationarity.csv")
    jm = stm[(stm.axis == "jbar") & (stm.generation == G["g_obs"])].iloc[0]
    case("P2 denominators (analyze CLI): a removed terminal W5 row is a counted missing quantity (J-bar NOT_IDENTIFIABLE at "
         "the terminal window, code 5, descriptor END_UNAVAILABLE, run VALID); with a counterexample in the other "
         "terminal generation the axis is REFUTED (code 1); a zero crossing refutes (infinite M kept, descriptor "
         "END_UNAVAILABLE as a non-finite endpoint, VALID); deleting large cells "
         "or a zero crossing breaks the quantity accounting (X, rc 2)", "4 P2 terminal window",
         (r4m["missing"]["rc"], int(jm.missing), code(r4m["missing"], "P2:jbar"),
          ((r4m["missing"].get("P2_descriptors") or {}).get("jbar") or {}).get("code"),
          code(r4m["missing_with_counter"], "P2:jbar"), code(r4m["zero_cross"], "P2:ion"), r4m["zero_cross"].get("validity"),
          ((r4m["zero_cross"].get("P2_descriptors") or {}).get("ion") or {}).get("code"),
          (r4m["drop_large"]["rc"], has(r4m["drop_large"], "P2_QUANTITY_ACCOUNTING")),
          (r4m["drop_zero_cross"]["rc"], has(r4m["drop_zero_cross"], "P2_QUANTITY_ACCOUNTING"))),
         (0, 1, 5, "END_UNAVAILABLE", 1, 1, "VALID", "END_UNAVAILABLE", (2, True), (2, True)), fault="drop_large|drop_zero_cross",
         runs=list(r4m.values()))

    def contradict(N):
        def m(f):
            if f[0] == "5" and f[5] == "3":
                L = float(f[12]) * 2.0
                f[12], f[13] = repr(L), repr(float(f[11]) * float(f[9]) + L)
            return f
        det_nrb_judge_w5_edit(N, m)
    r4c = analyze("p2_level_contradiction", injection="line 3 L doubled at call 5 (level 4 recovered twice, disjoint)",
                  mutate_N=contradict)
    case("level authority (closeout B2, analyze CLI): two admissible recoveries of Si III level 4 at generation 5 that do "
         "not overlap make the level axis X with the key and generation recorded (not a missing value), the run X "
         "(rc 2) and the P2 decision INADMISSIBLE", "4 P2 terminal window",
         (r4c["rc"], has(r4c, "P2_LEVEL_CONTRADICTION:level:14:4"), code(r4c, "P2:level"), code(r4c, "P2:jbar"),
          dec(r4c).get("p2_disposition")), (2, True, "X", 0, "INADMISSIBLE"), runs=[r4c])
    # ---- 5 public ions
    r5 = {fl: analyze(f"ion_{fl}", fault=fl) for fl in ("saha_column", "q_shift", "renormalize", "time_shift", "space_shift")}
    r5x = analyze("ion_refuted", pub_factor={(14, 10): 3.0})
    i5 = csv(r5x, "ion_fraction.csv")
    i5 = i5[(i5.run == "N") & (i5.Z == 14) & (i5.shell == 10) & (i5.ion != "III/II")]
    case("public ions (analyze CLI): NLTE q1 / q2 over the element total at generation 13, the 19.480 block and the "
         "registered vel_mid; the SAHA stage column, a q0/q1 shift, II+III renormalisation, the 17.710 block and a moved "
         "spatial match are refused (X, rc 2); a public value 3x off refutes that element (code 1, VALID, rc 0)",
         "5 public ions",
         ({k: (v["rc"], v.get("validity")) for k, v in r5.items()}, has(r5["saha_column"], "ION_COLUMN_RECOMPUTE"),
          has(r5["q_shift"], "ION_PUBLIC_TOKEN_RECOMPUTE"), has(r5["renormalize"], "ION_COLUMN_RECOMPUTE"),
          has(r5["time_shift"], "ION_PUBLIC_TOKEN_RECOMPUTE"), code(r5["space_shift"], "IONS:14"),
          r5x["rc"], code(r5x, "IONS:14"), code(r5x, "IONS:16"), set(i5.status)),
         ({k: (2, "X") for k in r5}, True, True, True, True, "X", 0, 1, 0, {"REFUTED"}),
         fault="saha_column|q_shift|renormalize|time_shift|space_shift", runs=list(r5.values()) + [r5x])
    # ---- 6 public spectrum
    def scale_band(lo, hi, k):
        def f_(N):
            p_ = N / "work/lumina_spectrum_det_obs.csv"
            rows_ = p_.read_text().split("\n")
            out_ = [rows_[0]]
            for ln in rows_[1:]:
                if ln:
                    x, y = ln.split(",")
                    out_.append(f"{x},{float(y) * (k if lo <= float(x) < hi else 1.0)!r}")
            p_.write_text("\n".join(out_) + "\n")
        return f_
    r6 = {"retarget": analyze("spec_retarget", fault="retarget"),
          "subset_bands": analyze("spec_subset", fault="subset_bands", mutate_N=scale_band(5000.0, 6000.0, 1.5)),
          "hide_opt": analyze("spec_hide", fault="hide_opt")}
    # the public file is written from the unmutated N spectrum; the N spectrum alone is tilted below 3000 A
    r6t = analyze("spec_tilt", mutate_N=scale_band(0.0, 3000.0, 3.0))
    # closeout round 5: the N spectrum zero over the OPT band (normalisation impossible) is the regular code 5
    r6n = analyze("spec_opt_zero", injection="N spectrum zero over 3900-6100 A", mutate_N=scale_band(3900.0, 6100.0, 0.0))
    r6n_sp = r6n.get("spectrum") or {}
    case("public spectrum (analyze CLI): thresholds regenerated from N, a band subset declared as the whole and the OPT "
         "change hidden are refused (X, rc 2); a 3x excess below 3000 A refutes the registered targets (code 1, VALID, "
         "rc 0) with the absolute OPT recorded; (closeout round 5) an N spectrum with zero OPT integral is the regular "
         "not-identifiable code 5 (VALID, rc 0, the reason kept in the summary, an empty condition set, the stored-output "
         "check clean)", "6 public spectrum",
         ((r6["retarget"]["rc"], has(r6["retarget"], "SPECTRUM_CONDITION_RECOMPUTE")),
          (r6["subset_bands"]["rc"], has(r6["subset_bands"], "SPECTRUM_OVERALL_RECOMPUTE")),
          (r6["hide_opt"]["rc"], has(r6["hide_opt"], "SPECTRUM_OPT_CHANGE_NOT_RECORDED")), r6t["rc"], code(r6t, "SPECTRUM"),
          ((r6t.get("codes") or {}).get("SPECTRUM") or {}).get("conditions", {}).get("UV2500"),
          ((r6t.get("spectrum") or {}).get("summary") or {}).get("N", {}).get("L_OPT") is not None,
          (r6n["rc"], r6n.get("validity"), code(r6n, "SPECTRUM"), ((r6n.get("codes") or {}).get("SPECTRUM") or {}).get("state"),
           ((r6n.get("codes") or {}).get("SPECTRUM") or {}).get("conditions"), r6n_sp.get("conditions"),
           ((r6n_sp.get("summary") or {}).get("N") or {}).get("status"), r6n.get("stored_check"),
           [w for w in (r6n.get("reasons") or []) if "SPECTRUM" in str(w)])),
         ((2, True), (2, True), (2, True), 0, 1, 1, True,
          (0, "VALID", 5, "NOT_IDENTIFIABLE", {}, {}, "NOT_IDENTIFIABLE", None, [])),
         fault="retarget|subset_bands|hide_opt", runs=list(r6.values()) + [r6t, r6n])
    # ---- 7 P1(a) recall and propagation
    def drop_ctrl(N):
        det_nrb_judge_w5_edit(N, lambda f: None if (f[0] == "7" and f[4] == "10" and f[5] == "0") else f)
    r7a = analyze("recall_missing", injection="control line 0 / shell 10 / call 7 row removed", mutate_N=drop_ctrl)
    r7b = analyze("recall_recalled_only", fault="recall_recalled_only", mutate_N=drop_ctrl)
    r7c = analyze("recall_x_inherit_fail", fault="move_c_with_L", prior_bad_source=True)
    r7d = analyze("inherit_fail_only", prior_bad_source=True)
    case("P1(a) local (closeout B1, analyze CLI): a missing cell of the fixed recall keeps the denominator 24 and gives "
         "P1A_LOCAL 5 (not 0); counting recalled cells only is refused by the set recomputation (X, rc 2); with a failed "
         "inheritance an X recall stays X (not 5), and a failed inheritance alone gives 5", "7 propagation",
         (code(r7a, "P1A_LOCAL"), (r7a.get("P1A_recall") or {}).get("cells"), r7a.get("validity"),
          (r7b["rc"], has(r7b, "P1A_RECALL_SET_RECOMPUTE")), code(r7c, "P1A_LOCAL"), code(r7d, "P1A_LOCAL"), r7d.get("validity")),
         (5, 24, "VALID", (2, True), "X", 5, "VALID"), fault="recall_recalled_only|move_c_with_L+inheritance",
         runs=[r7a, r7b, r7c, r7d])
    r7 = analyze("flip_all", fault="status_flip_all+retarget")
    d7 = dec(r7)
    spec_msgs = [w for w in r7.get("reasons", []) if w.startswith("SPECTRUM")]
    r7e = analyze("drop_counter", fault="drop_counter")
    case("codes / decision propagation (mass errors, analyze CLI): every P1(a) state flipped (96 findings, stored list "
         "capped at 50) with regenerated spectral thresholds in the same run: P1(a) and SPECTRUM both X although the "
         "spectrum finding lies beyond the cap, P1(b) keeps its code, decision.json validity X with both X rows, rc 2; a "
         "shortened sample is refused before any science code", "7 propagation",
         (r7["rc"], r7.get("n_reasons", 0) > 50, len(r7.get("reasons", [])), spec_msgs, code(r7, "P1A"), code(r7, "SPECTRUM"),
          code(r7, "P1B_CORE"), d7.get("validity"), sorted(row["condition"] for row in d7.get("rows", []) if row["condition"].endswith(" X")),
          (r7e["rc"], has(r7e, "SAMPLE_NOT_REGISTERED"))),
         (2, True, 50, [], "X", "X", 0, "X", ["P1A X", "SPECTRUM X"], (2, True)),
         fault="status_flip_all+retarget|drop_counter", runs=[r7, r7e])
    # ---- 8 byte comparison and capture authority
    def r_byte(R):
        p_ = R / "work/physics_comparison/physics_DET_iter0002.ion.csv"
        p_.write_text(p_.read_text() + " ")

    def r_gen(R):
        m_ = R / "work/physics_comparison/physics_DET_iter0001.manifest.json"
        j_ = json.loads(m_.read_text())
        j_["population_generation"] = 99
        m_.write_text(json.dumps(j_))
    r8a = analyze("bytes_diff", injection="reference iter0002.ion.csv one byte appended", mutate_R=r_byte)
    r8b = analyze("bytes_gen", injection="reference iter0001 manifest generation 99", mutate_R=r_gen)
    sa, sb = (r8a.get("bytes_vs_reference") or {}).get("summary", {}), (r8b.get("bytes_vs_reference") or {}).get("summary", {})
    keep = lambda r_: {k: code(r_, k) for k in (r_.get("codes") or {})}                 # noqa: E731
    case("byte comparison (prereg 3.4, analyze CLI): clean 10/10 identical; one appended byte in the reference generation-5 "
         "ion dump is DIFFERENT (1) and a reference manifest whose population generation is not 4 makes that iteration "
         "NOT_COMPARABLE (2 files) — both runs stay VALID with every science code equal to the clean run (the comparison "
         "is consumed as an observation only)", "8 bytes",
         (bts.get("identical"), (sa.get("identical"), sa.get("different")), (sb.get("identical"), sb.get("not_comparable")),
          r8a.get("validity"), r8b.get("validity"), keep(r8a) == keep(r0), keep(r8b) == keep(r0)),
         (10, (9, 1), (8, 2), "VALID", "VALID", True, True), runs=[r8a, r8b])

    def cap_edit(N):
        capd = N / "work/det_obs_capture"
        m = np.fromfile(capd / "lines.bin", dtype=POP.REC)
        i_ = int(np.nonzero((m["l"] == 1) & (m["s"] == 10))[0][0])
        m["S"][i_] = m["S"][i_] * 1.5
        m.tofile(capd / "lines.bin")
        j_ = json.loads((capd / "manifest.json").read_text())
        j_["files"]["lines.bin"]["sha256"] = sha(capd / "lines.bin")
        (capd / "manifest.json").write_text(json.dumps(j_))
    r8c = analyze("capture_authority", injection="capture S of line 1 / shell 10 x1.5 (manifest sha updated)", mutate_N=cap_edit)
    case("independent authority (analyze CLI): a generation-13 capture source that disagrees with the W5 inversion of "
         "the same line is refused (POP_AUTHORITY_MISMATCH, X, rc 2)", "5 authority",
         (r8c["rc"], has(r8c, "POP_AUTHORITY_MISMATCH")), (2, True), runs=[r8c])
    # ---- closeout round 2: sandbox boundary (R1) and stored-output check (R2)
    sb = base / "sandbox_boundary"
    sb.mkdir()
    (sb / SANDBOX).write_text(json.dumps({"schema": "DET_NRB_JUDGE_SELFTEST_SANDBOX_V1", "fault": None,
                                          "target": det_nrb_judge_target_json(FX_TARGET)}))
    pubs = fx_s["public"]
    common = ["--prereg", P / PREREG, "--baseline-run", fx_s["E"], "--reference-run", fx_s["R"], "--prior-analysis",
              fx_s["A"], "--prior-selftest", fx_s["st"], "--spectral-expectations", pubs["expectations"],
              "--production-source", src, "--public-spectrum", pubs["spectrum"], "--public-si", pubs[14], "--public-s",
              pubs[16], "--selftest", base / "t_none"]
    rcR1, soR1, seR1 = det_nrb_judge_cli(["analyze", "--run", fx_s["N"], "--out", sb / "out_registered"] + common)
    rcR2, soR2, seR2 = det_nrb_judge_cli(["selftest-analyze", "--run", fx_s["N"], "--out", sb / "out_outside"] + common)
    nb = base / "no_sandbox"
    nb.mkdir()
    rcR3, soR3, seR3 = det_nrb_judge_cli(["selftest-analyze", "--run", fx_s["N"], "--out", nb / "out"] + common)
    case("sandbox boundary (closeout R1, CLI): the registered analyze refuses a SELFTEST_SANDBOX.json next to --out "
         "(rc 2, nothing written); selftest-analyze refuses inputs outside the sandbox directory and a missing sandbox "
         "(rc 2) — a fault or a fixture target cannot reach a registered result", "1 single change",
         ((rcR1, "SANDBOX_REFUSED" in soR1, (sb / "out_registered").exists()),
          (rcR2, "SANDBOX_PATH_OUTSIDE" in soR2, (sb / "out_outside").exists()),
          (rcR3, "SANDBOX_ABSENT" in soR3, (nb / "out").exists())),
         ((2, True, False), (2, True, False), (2, True, False)), fault="sandbox next to registered analyze|outside paths|absent",
         runs=[{"dir": str(sb), "rc": rcR1, "injection": "registered analyze + sandbox", "reasons": [last(soR1)]},
               {"dir": str(sb), "rc": rcR2, "injection": "inputs outside the sandbox", "reasons": [last(soR2)]},
               {"dir": str(nb), "rc": rcR3, "injection": "no sandbox", "reasons": [last(soR3)]}])

    def ne_cross(ne):
        for g in (G["g_obs"], G["g_last"]):
            for s_ in range(FX_NS):
                v = float(ne[G["outer"]][s_]) * 10.0 ** P2_CRIT
                for _ in range(200):
                    m = SEED.det_seedmem_scope_m(float(ne[G["outer"]][s_]), v)
                    if m.lo < P2_CRIT <= m.hi:
                        break
                    v = dn(v) if m.lo >= P2_CRIT else up(v)
                ne[g][s_] = v
    r_u = analyze("p2_ne_unresolved", injection="n_e change 12->13 straddling 0.0043 dex", ne_adjust=ne_cross)
    stored = {"stored_code_flip": analyze("st_code_flip", fault="stored_code_flip", ion_series=past_only),
              "stored_D_alter": analyze("st_D_alter", fault="stored_D_alter", ion_series=past_only),
              "stored_rows_delete": analyze("st_rows_delete", fault="stored_rows_delete", ion_series=past_only),
              "stored_disposition_both": analyze("st_disp_both", fault="stored_disposition_both", ion_series=past_only),
              "stored_decision_inadmissible": analyze("st_dec_inadm", fault="stored_decision_inadmissible", ion_series=past_only),
              "stored_4_as_1": analyze("st_4_as_1", fault="stored_4_as_1", ne_adjust=ne_cross),
              "stored_5_as_0": analyze("st_5_as_0", fault="stored_5_as_0", mutate_N=drop_term(1, 10)),
              "stored_X_as_valid": analyze("st_X_as_valid", fault="beta_current+stored_X_as_valid"),
              "stored_ions_flip": analyze("st_ions_flip", fault="stored_ions_flip", pub_factor={(14, 10): 3.0}),
              "stored_spectrum_flip": analyze("st_spec_flip", fault="stored_spectrum_flip",
                                              mutate_N=scale_band(0.0, 3000.0, 3.0)),
              "stored_p1b_flip": analyze("st_p1b_flip", fault="stored_p1b_flip"),
              "stored_scope_delete": analyze("st_scope_delete", fault="stored_scope_delete"),
              "stored_spec_targets_delete": analyze("st_spec_targets_delete", fault="stored_spec_targets_delete",
                                                    mutate_N=scale_band(0.0, 3000.0, 3.0)),
              "stored_spectrum_section_delete": analyze("st_spec_section_delete", fault="stored_spectrum_section_delete"),
              "stored_spec_conditions_delete": analyze("st_spec_conditions_delete", fault="stored_spec_conditions_delete",
                                                       mutate_N=scale_band(3900.0, 6100.0, 0.0)),
              "stored_required_delete": analyze("st_required_delete", fault="stored_required_delete")}
    scope_of = {"stored_ions_flip": ["IONS:14"], "stored_spectrum_flip": ["SPECTRUM"], "stored_p1b_flip": ["P1B_CORE"],
                "stored_scope_delete": ["P1A_LOCAL", "IONS:16"], "stored_spec_targets_delete": ["SPECTRUM"],
                "stored_spectrum_section_delete": ["SPECTRUM"], "stored_spec_conditions_delete": ["SPECTRUM"],
                "stored_required_delete": ["P1A", "P1B_CORE", "P1B_EXT", "SPECTRUM"]}

    def no_science_rows(v):
        return all(r_["condition"].endswith(" X") or r_["condition"] in ("run inadmissible", "INCOMPLETE")
                   for r_ in dec(v).get("rows", []))
    case("stored-output check (closeout R2, analyze CLI): with a defective writer simulated after validation — an axis code "
         "1 -> 0 with the state kept, descriptor D replaced, decision rows deleted, the P2 disposition changed in both "
         "files, the decision disposition alone set to INADMISSIBLE, an UNRESOLVED (4) axis written as REFUTED (1), a "
         "NOT_IDENTIFIABLE (5) axis written as SURVIVES (0), an X scope written into a VALID run, and (closeout round 3) "
         "a refuted public-ion code written as SURVIVES, a refuted spectrum code written as SURVIVES, a surviving P1(b) "
         "core written as REFUTED and two registered propositions deleted, and (closeout round 4) the stored spectrum "
         "targets deleted with a SURVIVES code, the spectrum section deleted, four registered propositions deleted, and "
         "(closeout round 5) the condition set of a not-identifiable spectrum deleted — "
         "every stored file set is read back, recomputed from its stored rows / conditions and refused (X, rc 2), the "
         "proposition concerned is X in the rewritten result and decision.json carries X rows only (no science "
         "follow-up); the unedited n_e straddle gives code 4 (VALID)",
         "4 P2 terminal window",
         ((r_u["rc"], r_u.get("validity"), code(r_u, "P2:ne")),
          {k: (v["rc"], v.get("validity"), bool(v.get("stored_check")), no_science_rows(v),
               [code(v, sc) for sc in scope_of.get(k, [])]) for k, v in stored.items()}),
         ((0, "VALID", 4), {k: (2, "X", True, True, ["X"] * len(scope_of.get(k, []))) for k in stored}),
         fault="|".join(stored), runs=[r_u] + list(stored.values()))
    # ---- 9 preservation (real CLI)
    rcP, soP, seP = det_nrb_judge_cli(["analyze", "--prereg", P / PREREG, "--run", fx_s["N"], "--baseline-run", fx_s["E"],
                                       "--reference-run", fx_s["R"], "--prior-analysis", fx_s["A"], "--prior-selftest",
                                       fx_s["st"], "--spectral-expectations", fx_s["public"]["expectations"],
                                       "--production-source", src, "--public-spectrum", fx_s["public"]["spectrum"],
                                       "--public-si", fx_s["public"][14], "--public-s", fx_s["public"][16],
                                       "--selftest", base / "clean/no_selftest", "--out", base / "clean/out"])
    case("preservation (analyze CLI): an existing result path is refused before writing (rc 3, OUT_EXISTS)", "9 preservation",
         (rcP, "OUT_EXISTS" in soP + seP), (3, True))
    # ---- inherited pure functions
    inh = det_nrb_judge_inherited_bundles(prior_selftest)
    case("inherited pure functions: the SEEDMEM-SCOPE r4 selftest (readout sha equal to the imported module, failures 0) "
         "carries the cases of the unmodified functions this readout calls (inversion / forward ledger / source identity, "
         "the NRB rate reconstruction, the local-exchange premise, the collision authority, solve units, interval helpers); "
         "the new P2, propagation, decision and stored-output code is tested directly above", "10 inheritance",
         (inh["dependency_sha_matches"], inh["failures"], sorted(inh["bundles"]) == sorted(inh["required"])),
         (True, 0, True), kind="INHERITED")
    res[-1]["inherited"] = inh
    return res


INHERITED_BUNDLES = {"4 inversion": "det_seedmem_scope_invert / _forward / _identity / _meas / _source2 / _coll / _r",
                     "6 NRB link": "det_seedmem_scope_rates / _row_check / _is_link (P1(a) rate reconstruction)",
                     "7 cross": "H-LOCAL premise (local-term exchange of the production bracket)",
                     "9 collisions": "det_seedmem_scope_coll_authority",
                     "11 numerics": "det_seedmem_scope_le / _universal_check / _set_code / _m / _overlap",
                     "13 solve units": "det_seedmem_scope_sl"}


def det_nrb_judge_inherited_bundles(prior_selftest):
    p = Path(prior_selftest) / "selftest.json"
    st = json.loads(p.read_text())
    seed_sha = sha(Path(SEED.__file__).resolve())
    cases = st.get("cases", [])
    bundles = sorted({c.get("bundle") for c in cases if c.get("bundle") in INHERITED_BUNDLES})
    return {"path": str(p), "sha256": sha(p), "prior_readout_sha256": st.get("readout_sha256"), "seed_module_sha256": seed_sha,
            "dependency_sha_matches": st.get("readout_sha256") == seed_sha, "failures": st.get("failures"),
            "bundles": bundles, "required": sorted(INHERITED_BUNDLES), "functions": INHERITED_BUNDLES,
            "cases": [{"bundle": c.get("bundle"), "case": c.get("case")[:160], "pass": c.get("pass"), "fault": c.get("fault")}
                      for c in cases if c.get("bundle") in INHERITED_BUNDLES],
            "not_inherited": "P2 per-generation measures, terminal window, descriptors, ions, spectrum linkage, validators, "
                             "propagation, decision and stored-output check are new code and are tested directly"}


# ------------------------------------------------------------------ CLI
def det_nrb_judge_manifest(a, out, extra):
    man = {"schema": "DET_NRB_JUDGE_MANIFEST_V1", "contract": {"path": str(Path(a.prereg).resolve()), "sha256": sha(a.prereg)}
           if getattr(a, "prereg", None) else None,
           "readout": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
           "imports": {str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (POP, RATES, MECH, REP, SEED)},
           "imported_functions": {
               "det_seedmem_scope_readout": ["det_seedmem_scope_build_lib", "det_seedmem_scope_Lib", "det_seedmem_scope_csob",
                                             "det_seedmem_scope_deck", "det_seedmem_scope_r", "det_seedmem_scope_sl",
                                             "det_seedmem_scope_coll_authority", "det_seedmem_scope_dumps",
                                             "det_seedmem_scope_dumps_check", "det_seedmem_scope_capture_integrity",
                                             "det_seedmem_scope_row_check", "det_seedmem_scope_is_link",
                                             "det_seedmem_scope_rates", "det_seedmem_scope_meas", "det_seedmem_scope_coll",
                                             "det_seedmem_scope_source2", "det_seedmem_scope_identity",
                                             "det_seedmem_scope_invert", "det_seedmem_scope_m", "det_seedmem_scope_le",
                                             "det_seedmem_scope_overlap", "det_seedmem_scope_universal_check",
                                             "det_seedmem_scope_set_code", "det_seedmem_scope_status_check",
                                             "det_seedmem_scope_block", "det_seedmem_scope_fx_deck"],
               "det_siiii_repair_readout": ["det_siiii_repair_w5_load", "det_siiii_repair_public", "det_siiii_repair_spec",
                                            "det_siiii_repair_spectrum_table", "det_siiii_repair_le", "det_siiii_repair_lt",
                                            "det_siiii_repair_cols"],
               "det_siiii_pop_readout": ["I", "sha", "up", "dn", "printed", "read_meta", "capture_records", "recover",
                                         "prod_tau_S", "REC", "C_SOB"],
               "det_siiii_rates_readout": ["det_siiii_rates_coll", "det_siiii_rates_exports", "NU_MIN", "NU_MAX"],
               "det_siiii_1206mech_readout": ["det_1206mech_check_row (via SEED)"]},
           "argv": sys.argv, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
           "numpy": np.__version__, "pandas": pd.__version__, "outputs": {}} | extra
    for p in sorted(Path(out).rglob("*")):
        if p.is_file() and p.suffix in (".json", ".csv", ".so", ".txt", ".log"):
            man["outputs"][str(p.relative_to(out))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return man


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    tests = det_nrb_judge_selftest_cases(out / "fx", Path(a.production_source), Path(a.runner), a.prior_selftest)
    fails = sum(1 for t in tests if not t["pass"])
    summary = {"schema": "DET_NRB_JUDGE_SELFTEST_V1", "readout_sha256": sha(Path(__file__).resolve()), "cases": tests,
               "failures": fails, "elapsed_s": time.time() - t0,
               "bundles_new": sorted({t["bundle"] for t in tests if t.get("kind") == "NEW"}),
               "records": {"new_cli_records": sum(1 for t in tests if t.get("kind") == "NEW"),
                           "inherited_identity_records": sum(1 for t in tests if t.get("kind") == "INHERITED")},
               "bundles_inherited": sorted(INHERITED_BUNDLES),
               "rc_meaning": "cli_rc = the exit status of a separate readout process (analyze / stage CLI: 0 VALID or "
                             "staged, 2 inadmissible, 3 existing result path, 1 refused stage); the runner selftest is a "
                             "separate bash process; selftest faults and the fixture target reach the analyze CLI only "
                             "through a SELFTEST_SANDBOX.json next to --out (absent in every registered analysis)",
               "runner_sha256": sha(Path(a.runner)),
               "dependencies": {str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (POP, RATES, MECH, REP, SEED)}}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"failures": fails, "cases": len(tests), "elapsed_s": summary["elapsed_s"]}))
    return 0 if fails == 0 else 1


def cmd_preflight(a):
    out = Path(a.out)
    rec = det_nrb_judge_preflight(a, out)
    man = det_nrb_judge_manifest(a, out, {"kind": "preflight", "submit_allowed": rec["submit_allowed"]})
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"submit_allowed": rec["submit_allowed"], "why": rec["why"]}))
    return 0 if rec["submit_allowed"] else 2


def cmd_stage(a):
    ident = det_nrb_judge_stage(a.reference_run, a.run, a.runner, a.gpu_profile, a.preflight)
    print(json.dumps({"staged": a.run, "identity": ident}, default=str)[:2000])
    return 0


SANDBOX_PATH_ARGS = ("run", "baseline_run", "reference_run", "prior_analysis", "prior_selftest", "spectral_expectations",
                     "public_spectrum", "public_si", "public_s")


def cmd_analyze(a):
    """registered analysis: the registered target, no fault, the selftest of this readout required; a selftest
    sandbox file next to --out is refused (closeout round 2, R1)."""
    out = Path(a.out)
    if out.exists():
        print(f"OUT_EXISTS {out}")
        return 3
    if (out.parent / SANDBOX).exists():
        print(f"SANDBOX_REFUSED {out.parent / SANDBOX}")
        return 2
    sj_p = Path(a.selftest) / "selftest.json"
    sj = json.loads(sj_p.read_text()) if sj_p.exists() else {}
    if sj.get("readout_sha256") != sha(Path(__file__).resolve()) or sj.get("failures") != 0:
        print("SELFTEST_NOT_THIS_READOUT_OR_FAILED")
        return 2
    return det_nrb_judge_cli_analyze(a, out, TARGET, None, None)


def cmd_selftest_analyze(a):
    """selftest-only analysis (prereg 8.2 direct tests): requires the SELFTEST_SANDBOX.json next to --out and every
    run / prior / public / expectations input and --out inside that sandbox directory; reads the fixture target and the
    fault from it.  The registered `analyze` refuses such a sandbox, so a fault or a fixture target cannot reach a
    registered result."""
    out = Path(a.out)
    if out.exists():
        print(f"OUT_EXISTS {out}")
        return 3
    root = out.parent.resolve()
    sbx_p = root / SANDBOX
    if not sbx_p.exists():
        print("SANDBOX_ABSENT")
        return 2
    sandbox = json.loads(sbx_p.read_text())
    if sandbox.get("schema") != "DET_NRB_JUDGE_SELFTEST_SANDBOX_V1":
        print("SANDBOX_SCHEMA")
        return 2
    outside = [k for k in SANDBOX_PATH_ARGS if root not in Path(getattr(a, k)).resolve().parents]
    if outside:
        print(f"SANDBOX_PATH_OUTSIDE {outside}")
        return 2
    return det_nrb_judge_cli_analyze(a, out, det_nrb_judge_target_from_json(sandbox["target"]), sandbox.get("fault"),
                                     sandbox)


def det_nrb_judge_cli_analyze(a, out, target, fault, sandbox):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=False)
    public = {"spectrum": a.public_spectrum, 14: a.public_si, 16: a.public_s}
    res = det_nrb_judge_analyze(a.run, a.baseline_run, a.reference_run, a.prior_analysis, a.prior_selftest,
                                a.spectral_expectations, a.production_source, public, out / "analysis", out / "lib",
                                target, fault)
    inputs = {}
    for tag, rp in (("N", a.run), ("E", a.baseline_run), ("R", a.reference_run)):
        rels = ["STAGE_IDENTITY.txt", "RUN_FOOTER.txt", "input/run_build_identity.json", "input/resolved_lumina.exports",
                "input/binary.sha256", "input/deck.sha256", "input/fixed_te_profile.txt", "work/det_obs_capture/manifest.json",
                "work/det_obs_capture/meta.txt", "work/det_obs_status.json", "work/cmfgen_final_publication.json",
                "work/lumina_spectrum_det_obs.csv", "stdout.log", "stderr.log"]
        if tag == "N":                              # closeout B4: the ledgers and capture the readout consumed
            rels += ["work/w5_lines.csv", "work/det_obs_capture/lines.bin", "work/det_obs_capture/ne.f64",
                     "STAGE_FILES.json", "job.slurm", "SUBMIT_COMMAND.txt"]
        for rel in rels:
            p = Path(rp) / rel
            if p.exists():
                inputs[f"{tag}:{rel}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted((Path(rp) / "work/physics_comparison").glob("physics_DET_iter*")):
            if p.suffix in (".json", ".csv") and "spectral" not in p.name:
                inputs[f"{tag}:work/physics_comparison/{p.name}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
    for k, p in (("public_spectrum", a.public_spectrum), ("public_si", a.public_si), ("public_s", a.public_s),
                 ("expectations", a.spectral_expectations), ("prior_sample", Path(a.prior_analysis) / "sample.json"),
                 ("prior_result", Path(a.prior_analysis) / "result.json"),
                 ("prior_stationarity", Path(a.prior_analysis) / "stationarity.csv"),
                 ("prior_manifest", Path(a.prior_analysis).parent / "manifest.json"),
                 ("prior_selftest", Path(a.prior_selftest) / "selftest.json")):
        p = Path(p)
        if p.exists():
            inputs[k] = {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
    man = det_nrb_judge_manifest(a, out, {"kind": "selftest-analyze" if sandbox else "analyze", "inputs": inputs,
                                          "sandbox": sandbox,
                                          "lib_build": res.get("lib_build"), "input_identity": res.get("input_identity"),
                                          "selftest": {"path": str(Path(a.selftest) / "selftest.json"),
                                                       "sha256": sha(Path(a.selftest) / "selftest.json")
                                                       if (Path(a.selftest) / "selftest.json").exists() else None},
                                          "sample": res.get("sample"), "H_LOCAL_inheritance": res.get("P1A_H_LOCAL_inheritance"),
                                          "bytes_vs_reference": res.get("bytes_vs_reference"),
                                          "elapsed_s": time.time() - t0})
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "reasons": (res.get("reasons") or [])[:10], "incomplete": res.get("incomplete"),
                      "P2": res.get("P2_disposition"), "codes": {k: (v.get("code") if isinstance(v, dict) else v)
                                                                  for k, v in (res.get("codes") or {}).items()}},
                     default=str)[:3000])
    return 0 if res["validity"] == "VALID" else 2


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    for k in ("--production-source", "--runner", "--prior-selftest", "--out"):
        s.add_argument(k, required=True)
    s = sp.add_parser("preflight")
    for k in ("--prereg", "--baseline-run", "--reference-run", "--prior-analysis", "--prior-selftest",
              "--spectral-expectations", "--production-source", "--selftest", "--runner", "--public-spectrum",
              "--public-si", "--public-s", "--out"):
        s.add_argument(k, required=True)
    s = sp.add_parser("stage")
    for k in ("--reference-run", "--preflight", "--gpu-profile", "--runner", "--run"):
        s.add_argument(k, required=True)
    for nm in ("analyze", "selftest-analyze"):
        s = sp.add_parser(nm)
        for k in ("--prereg", "--run", "--baseline-run", "--reference-run", "--prior-analysis", "--prior-selftest",
                  "--spectral-expectations", "--production-source", "--public-spectrum", "--public-si", "--public-s",
                  "--selftest", "--out"):
            s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit({"selftest": cmd_selftest, "preflight": cmd_preflight, "stage": cmd_stage, "analyze": cmd_analyze,
              "selftest-analyze": cmd_selftest_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
