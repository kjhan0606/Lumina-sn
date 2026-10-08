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
    return {"run": str(run), "tag": tag, "outer": outer, "G": G, "t_exp": t_exp, "exports": ex, "nrb": nrb,
            "form": "NRB" if nrb else "OFF", "hist": hist, "dumps": dumps, "sl": sl, "coll": coll, "markers": mk,
            "capture_manifest": man, "obs_status": obs, "footer": foot, "binary_link": binary_link,
            "n_shells": n_shells, "why": why}


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
    for Z, lev, s in levels:
        ser = {}
        for g in G["rad_gens"]:
            v = rec_pop.get((Z, lev, s, g))
            if not v or not all(SEED.det_seedmem_scope_overlap(x, y) for x in v for y in v):
                ser[g] = None
            else:
                ser[g] = I(max(x.lo for x in v), v[0].mid, min(x.hi for x in v))
        series["level"][(Z, lev, s)] = ser
    expected = {"ion": len(ION_QTY) * len(shells), "ne": len(shells), "jbar": len(keys), "level": len(levels)}
    return series, expected, levels


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
    series, expected, levels = det_nrb_judge_series(rd, keys, deck, csob, lib, cap_status, fault)
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
    return rows, codes, expected


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
    past = {}
    gens = sorted(per_g)
    for g1, g2 in zip(gens[:-1], gens[1:]):
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
    D = I(dn(b["M"].lo - a["M"].hi), b["M"].mid - a["M"].mid, up(b["M"].hi - a["M"].lo))
    out["D"] = D.j()
    if D.hi < 0.0:
        out["code"] = "END_LOWER"
    elif D.lo > 0.0:
        out["code"] = "END_HIGHER"
    elif D.lo == 0.0 and D.hi == 0.0:
        out["code"] = "END_EQUAL"
    else:
        out["code"] = "END_UNRESOLVED"
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
            lo, hi = dn(b.M_lo - a.M_hi), up(b.M_hi - a.M_lo)
            dcode = "END_LOWER" if hi < 0 else "END_HIGHER" if lo > 0 else "END_EQUAL" if (lo == 0 and hi == 0) \
                else "END_UNRESOLVED"
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
        return why if overall == "NOT_IDENTIFIABLE" else ["SPECTRUM_OVERALL_RECOMPUTE"]
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
    """prereg 10.3 follow-up table consuming admissible results only; X scopes carry no decision."""
    code = lambda k: scopes[k]["code"]                                                   # noqa: E731
    rows = []
    if incomplete:
        rows.append({"condition": "INCOMPLETE", "follow_up": "required run or output missing: no completion"})
    p1a, p1b_c, p1b_e = code("P1A"), code("P1B_CORE"), code("P1B_EXT")
    if p1a == "X":
        rows.append({"condition": "P1(a) X", "follow_up": "no decision on this scope"})
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
    elif p2_disposition == "NOT_CONVERGED":
        rows.append({"condition": "P2 not met", "follow_up": "report the stationarity.csv trajectory and the registered "
                     "endpoint descriptors / axis codes; iteration extension is a user decision",
                     "descriptors": descriptors})
    if pub_x:
        rows.append({"condition": "public comparison X", "follow_up": "no decision on that scope"})
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
    p2_rows, p2_codes, p2_expected = det_nrb_judge_p2(N, keys, deck, csob, lib, cap_status, fault)
    if FT(fault, "mixed_windows"):
        for ax, c in p2_codes.items():
            if c["any_past_window_small"]:
                c["code"], c["state"] = 0, "SURVIVES"
    p2_why = det_nrb_judge_p2_validate(p2_rows, p2_codes, p2_expected, G) + [w for w in cap_why]
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
    scopes = {"P1A": SEED.det_seedmem_scope_set_code(p1a_states),
              "P1A_LOCAL": SEED.det_seedmem_scope_set_code([r["status"] for r in recall_rows if r.get("recalled")] or
                                                           ["NOT_IDENTIFIABLE"]),
              "P1B_CORE": SEED.det_seedmem_scope_set_code(p1b_states["core"]),
              "P1B_EXT": SEED.det_seedmem_scope_set_code(p1b_states["extension"] or ["NOT_IDENTIFIABLE"]),
              "SPECTRUM": {"code": "X" if spec_overall == "X" else CODE[spec_overall], "state": spec_overall,
                           "conditions": {k: CODE.get(v, "X") for k, v in spec_conds.items()}}}
    for ax, c in p2_codes.items():
        scopes[f"P2:{ax}"] = c
    for Z in target["Z"]:
        scopes[f"IONS:{Z}"] = SEED.det_seedmem_scope_set_code(ion_states.get(Z, []))
    if not inherit["inherited"]:
        scopes["P1A_LOCAL"] = {"code": 5, "state": "NOT_IDENTIFIABLE", "reason": "H-LOCAL not inherited: " +
                               ";".join(inherit["why"]), "recall": scopes["P1A_LOCAL"]}
    n_x = {"p1a": p1a_states.count("X"), "recall": sum(1 for r in recall_rows if r["status"] == "X"),
           "p1b": sum(1 for v in p1b_states.values() for s in v if s == "X"),
           "ions": sum(1 for v in ion_states.values() for s in v if s == "X")}
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
           "sample": sample_rec, "exports_check": ex_why, "x_counts": n_x, "timing_s": timing,
           "claim_ceiling": "finite-iteration state of the deterministic lane (MC transport bypassed); no production "
                            "promotion, no physical correctness, no seed independence, no cause attribution"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "decision.json").write_text(json.dumps(decision, indent=1, default=str))
    return res


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
    inputs = {}
    for f in STAGED_FILES:
        p = R / "input" / f
        inputs[f] = {"sha256": sha(p), "bytes": p.stat().st_size} if p.is_file() and not p.is_symlink() else None
    for d in STAGED_DIRS:
        for p in sorted((R / "input" / d).rglob("*")):
            if p.is_file():
                inputs[str(p.relative_to(R / "input"))] = {"sha256": sha(p), "bytes": p.stat().st_size}
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
FX_SHELLS, FX_NS, FX_OUTER, FX_R = SEED.FX_SHELLS, SEED.FX_NS, 9, 0.3
FX_TEXP = SEED.FX_TEXP
FX_TARGET = {"Z": (14, 16), "ion": 2, "control": 0, "counts": {14: 6, 16: 2}, "n_keys": 8, "n_levels": 12,
             "shells": FX_SHELLS, "outer": FX_OUTER, "public_time": "19.480",
             "vmid_lumina": {10: 1000.0, 15: 1500.0}, "vmid_public": {10: 1001.0, 15: 1499.0}}


def det_nrb_judge_fx_run(root, tag, outer, lib, csob, conv=FX_R, ion_series=None, mutate=None, nrb=True,
                         pub_factor=None, spec_scale=1.0, binary_text="binary\n"):
    """one fixture run of `outer` iterations: W5 calls 1..outer+1 from populations and J_c approaching a limit as
    (1 + conv^g), dumps for populations 3..outer+2, the observer capture at generation outer+1, the final publication,
    spectrum, footer and identity files."""
    run = Path(root) / tag
    G = det_nrb_judge_gens(outer)
    (run / "input").mkdir(parents=True)
    lv = SEED.det_seedmem_scope_fx_deck(run / "input/model")
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
    (run / "work/physics_comparison").mkdir(parents=True)
    (run / "work/det_obs_capture").mkdir(parents=True)
    deck = SEED.det_seedmem_scope_deck(run / "input/model", (14, 16), 2)
    at = deck["at"]
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
    if nrb:
        ex += ['declare -x LUMINA_BB_NRB="1"', 'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"']
    else:
        ex += ['declare -x LUMINA_A210_SPRODUCER_CAPTURE="0"']
    (run / "input/resolved_lumina.exports").write_text("\n".join(sorted(ex)) + "\n")
    lumped = sum(1 for t in lv if t[2] >= SEED.FX_CUTOFF)
    (run / "stdout.log").write_text(
        f"  Levels: {len(lv)} loaded (super_level column present, super-levels active)\n"
        f"  [ARTIS super-cutoff] K={SEED.FX_CUTOFF}: {lumped} levels lumped -> per-ion explicit + 1 super-level\n"
        f"    Z=14 ion=2: {sum(1 for t in lv if t[:2] == (14, 2))} levels\n"
        f"    Z=16 ion=2: {sum(1 for t in lv if t[:2] == (16, 2))} levels\n"
        f"  [NLTE] Super-levels: ACTIVE ({len(lv)} FL -> {len(lv) - 1} SL across ions)\n"
        f"[CMFGEN] pure deterministic radiation driver: {FX_NS} shells, 1234 bins, 58 rays, {outer} outer iters, 24 ALI/iter, "
        f"t_exp=1.6831e+06 s\n[DET-TRANSACTIONAL] status=COMPLETE\n")
    se = [f"[BB-NRB] iter=0 operator=1 armed=0 captured=0 n_shells={FX_NS} skip_z=none"]
    if nrb:
        se += [f"[BB-NRB] iter={i} operator=2 armed=1 captured=1 n_shells={FX_NS} skip_z=none" for i in range(outer)]
    (run / "stderr.log").write_text("\n".join(se) + "\n")
    f = lambda g: 1.0 + conv ** g                                                       # noqa: E731
    Te = np.array([1.0e5 + 100.0 * s for s in range(FX_NS)])
    ne = {g: np.array([5.0e13 * f(g) * (1.0 + 0.001 * s) for s in range(FX_NS)]) for g in range(2, G["g_last"] + 1)}
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
                    tot = 4.0e6 * (1 + Z / 100)
                    vals += [tot, 0.9 * q1, 1.1 * q2, q1, q2]
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
                                                                   "radiation": g_obs, "iter": G["iter_last"]},
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
    (run / "RUN_FOOTER.txt").write_text(f"schema=NRB_JUDGE_RUN_V1\njob_id=1\nouter_iterations={outer}\nbinary_sha256={bsha}\n"
                                        f"elapsed_seconds=10\nmodel_rc=0\nterminal_state=SUCCESS\n")
    (run / "STAGE_IDENTITY.txt").write_text(f"schema=NRB_JUDGE_STAGE_V1\nrun={run}\nouter_iterations={outer}\n"
                                            f"run_binary_sha256={bsha}\n")
    (run / "fixture_truth.json").write_text(json.dumps({f"{k[0]}:{k[1]}:{k[2]}:{k[3]}": v for k, v in pops.items()}))
    if mutate:
        mutate(run)
    return run, ion_series


def det_nrb_judge_fx_public(base, run_N, target, pub_factor=None, spec_scale=1.0):
    """public fixtures: ion-fraction files whose 19.480 block carries the N generation-g_obs fractions (times a
    per-cell factor), a spectrum file with the N spectrum (times a scale) at 19.48 d, an expectations file."""
    base = Path(base)
    G = det_nrb_judge_gens(target["outer"])
    dump, _ = det_nrb_judge_ion_dump(run_N, G["g_obs"], target["shells"])
    files = {}
    for Z, nm, cols in ((14, "si", "si"), (16, "s", "s")):
        lines = ["#NTIMES:     2", "#NSTAGES:     5", "#TIMES[d]:    17.710   19.480", "#"]
        for t in ("17.710", "19.480"):
            lines += [f"#TIME:    {t}", "#NVEL: 4", f"#vel_mid[km/s]           {cols}0           {cols}1           {cols}2           {cols}3           {cols}4"]
            vrows = [(target["vmid_public"][s], s) for s in target["shells"]] + [(1200.0, None), (1700.0, None)]
            for v, s in sorted(vrows):
                if s is None:
                    fr = [1e-99, 0.3, 0.6, 0.1, 1e-99]
                else:
                    r = dump["rows"][s]
                    f1 = r[f"Z{Z}_nlte_q1_cm3"] / r[f"Z{Z}_element_total_cm3"]
                    f2 = r[f"Z{Z}_nlte_q2_cm3"] / r[f"Z{Z}_element_total_cm3"]
                    k = 1.0 if (pub_factor is None or t != "19.480") else pub_factor.get((Z, s), 1.0)
                    if t != "19.480":
                        k = 3.0
                    fr = [1e-99, f1 * k, f2 * k, 1e-3, 1e-99]
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
    (base / "result.json").write_text(json.dumps({"spectrum_summary": {}}))
    files["expectations"] = str(base / "expectations.json")
    return files


def det_nrb_judge_fx_prior(base, run_R, lib_rec, keys):
    """fixture 'prior analysis' (A) and 'prior selftest' with the inheritance identity of this library build."""
    A = Path(base) / "prior/analysis"
    A.mkdir(parents=True)
    (A / "sample.json").write_text(json.dumps({"schema": "DET_SEEDMEM_SCOPE_SAMPLE_V2", "n_keys": len(keys),
                                               "keys": [list(k) for k in keys]}))
    (A / "result.json").write_text(json.dumps({"validity": "VALID", "codes": {"H-LOCAL_record": {
        "code": 0, "state": "SURVIVES", "cells": 2550, "eligible_refuted_cells": 0, "x_cells": 0}}}))
    (A.parent / "manifest.json").write_text(json.dumps({"lib": {"sources_sha256": lib_rec["sources_sha256"]}}))
    st = Path(base) / "prior/selftest"
    st.mkdir(parents=True)
    (st / "selftest.json").write_text(json.dumps({"readout_sha256": "fixture", "failures": 0, "cases": [
        {"case": "cross substitution fixture", "bundle": "7 cross", "pass": True}]}))
    return str(A), str(st)


def det_nrb_judge_fx(base, lib, lib_rec, csob, outer=FX_OUTER, mutate_N=None, conv=FX_R, ion_series=None, pub_factor=None,
                     spec_scale=1.0, nrb=True, spec_mutate=None):
    """the full fixture set: N (outer), R (6) and E (6, off) runs, public files, prior analysis / selftest."""
    base = Path(base)
    base.mkdir(parents=True)
    runs = {}
    runs["N"], ion_N = det_nrb_judge_fx_run(base, "N", outer, lib, csob, conv, ion_series, mutate_N, nrb, spec_scale=spec_scale)
    runs["R"], _ = det_nrb_judge_fx_run(base, "R", 6, lib, csob, conv, None, None, True, binary_text="binary\n")
    runs["E"], _ = det_nrb_judge_fx_run(base, "E", 6, lib, csob, conv, None, None, False, binary_text="binary\n")
    keys = sorted((16 if line == 4 else 14, line, s) for line in (0, 1, 3, 4) for s in FX_SHELLS)
    public = det_nrb_judge_fx_public(base, runs["N"], FX_TARGET, pub_factor, spec_scale)
    if spec_mutate:
        spec_mutate(runs["N"])                      # after the public file: the N spectrum alone moves
    A, st = det_nrb_judge_fx_prior(base, runs["R"], lib_rec, keys)
    return runs, public, A, st, keys


def det_nrb_judge_selftest_cases(base, src, runner, prior_selftest):
    res = []
    base = Path(base)
    lib_rec = SEED.det_seedmem_scope_build_lib(src, base / "lib")
    lib = SEED.det_seedmem_scope_Lib(lib_rec["lib"])
    csob = SEED.det_seedmem_scope_csob(src)

    def case(name, bundle, got, want, fault=None, results=(), kind="NEW"):
        ok = got == want
        res.append({"case": name, "bundle": bundle, "kind": kind, "fault": fault, "pass": bool(ok), "got": str(got)[:700],
                    "want": str(want)[:700],
                    "runs": [{"validity": r.get("validity"), "analyze_rc_equivalent": 0 if r.get("validity") == "VALID" else 2,
                              "reasons": [str(x)[:160] for x in (r.get("reasons") or [])[:6]]} for r in results]})
        print(f"[{'PASS' if ok else 'FAIL'}] ({bundle}) {name}: got {str(got)[:240]} want {str(want)[:240]}", flush=True)

    def run(name, fault=None, mutate_N=None, target=None, **kw):
        runs, public, A, st, keys = det_nrb_judge_fx(base / name / "in", lib, lib_rec, csob, mutate_N=mutate_N, **kw)
        tgt = target or FX_TARGET
        r = det_nrb_judge_analyze(runs["N"], runs["E"], runs["R"], A, st, public["expectations"], src, public,
                                  base / name / "out", base / name / "lib", tgt, fault)
        return r, base / name / "out", runs, public, A, st

    def has(r, token):
        return r.get("validity") == "X" and any(token in str(w) for w in (r.get("reasons") or []))

    G = det_nrb_judge_gens(FX_OUTER)
    # ---- clean
    r0, o0, runs0, pub0, A0, st0 = run("clean")
    p1a = pd.read_csv(o0 / "p1_rates.csv")
    p1b = pd.read_csv(o0 / "p1_source_1206.csv")
    st_ = pd.read_csv(o0 / "stationarity.csv")
    ions = pd.read_csv(o0 / "ion_fraction.csv")
    c = r0.get("codes", {})
    case("clean (outer 9): VALID; P1(a) 0 over 8 keys x calls 2-10 (every cell NRB_BRACKET); H-LOCAL inherited and the "
         "18-cell recall invariant; P1(b) control survives in core 3-7 and extension 8-10; P2 four axes small in the "
         "terminal window (9,10) -> P2_SMALL_CHANGE; public ions and spectrum survive", "0 clean",
         (r0["validity"], c.get("P1A", {}).get("code"), int((p1a[p1a.kind != "RECALL"].branch == "NRB_BRACKET").sum()),
          c.get("P1A_LOCAL", {}).get("code"), r0.get("P1A_H_LOCAL_inheritance", {}).get("inherited"),
          c.get("P1B_CORE", {}).get("code"), c.get("P1B_EXT", {}).get("code"), r0.get("P2_disposition"),
          [c[f"P2:{ax}"]["code"] for ax in ("ion", "ne", "jbar", "level")], c.get("IONS:14", {}).get("code"),
          c.get("IONS:16", {}).get("code"), c.get("SPECTRUM", {}).get("code")),
         ("VALID", 0, 8 * len(G["calls"]), 0, True, 0, 0, "P2_SMALL_CHANGE", [0, 0, 0, 0], 0, 0, 0), results=[r0])
    case("clean: generation linkage — INIT call 1 excluded, 8 x 9 = 72 ordinary cells consuming material 3-11, P1(b) rows "
         "3-10 with material = call, the observed material 10 (iter0007) and the last update 11 kept apart in P2",
         "3 generation linkage",
         (sorted(set(p1a[p1a.kind != "RECALL"].call.astype(int))), sorted(set(p1b.call.astype(int))),
          bool((p1b.material_generation == p1b.call).all()), G["g_obs"], G["g_last"],
          sorted(set(st_[st_.axis == "ion"].generation.astype(int))), sorted(set(st_[st_.axis == "jbar"].generation.astype(int))),
          r0["P2_last_material_update"]["ion"]["generation"], int(ions[(ions.run == "N") & (ions.ion != "III/II")].generation.iloc[0])),
         (G["calls"], G["p1b_core"] + G["p1b_ext"], True, 10, 11, list(range(4, 12)), list(range(4, 11)), 11, 10),
         results=[r0])
    # ---- 1 single change and runner
    ref = runs0["R"]
    ref_lines = det_nrb_judge_exports_lines(ref / "input/resolved_lumina.exports")
    exp_chk = {}
    for fl in (None, "nrb_off", "capture_off", "iter6", "iter13", "extra_env"):
        d = base / f"exports_{fl}"
        d.mkdir(parents=True)
        text, _ = det_nrb_judge_exports_map(ref_lines, ref, d, FX_OUTER, fl)
        (d / "new.exports").write_text(text)
        exp_chk[fl] = det_nrb_judge_exports_check(ref / "input/resolved_lumina.exports", d / "new.exports", ref, d, FX_OUTER)["why"]
    case("single change: the re-rooted exports differ from R' in the iteration key only; NRB off, capture off, 6 or 13 "
         "iterations and an extra env key are refused (G-LINK)", "1 single change",
         (exp_chk[None], bool(exp_chk["nrb_off"]), bool(exp_chk["capture_off"]), bool(exp_chk["iter6"]),
          bool(exp_chk["iter13"]), bool(exp_chk["extra_env"])), ([], True, True, True, True, True),
         fault="nrb_off|capture_off|iter6|iter13|extra_env")
    pre_dir = base / "pre_fx"
    pre_dir.mkdir()
    (pre_dir / "offline.json").write_text(json.dumps({"submit_allowed": True}))
    n_new = base / "stage_fx/NRB/run1"
    ident = det_nrb_judge_stage(ref, n_new, runner, "h200x2", pre_dir, FX_OUTER)
    chk = det_nrb_judge_exports_check(ref / "input/resolved_lumina.exports", n_new / "input/resolved_lumina.exports", ref,
                                      n_new, FX_OUTER)
    try:
        det_nrb_judge_stage(ref, n_new, runner, "h200x2", pre_dir, FX_OUTER)
        again = "WROTE"
    except FileExistsError:
        again = "REFUSED"
    def tamper(r):
        (r / "input/lumina_cuda").write_text("tampered\n")
    runs_t, pub_t, A_t, st_t, _ = det_nrb_judge_fx(base / "tamper/in", lib, lib_rec, csob, mutate_N=tamper)
    r_t = det_nrb_judge_analyze(runs_t["N"], runs_t["E"], runs_t["R"], A_t, st_t, pub_t["expectations"], src, pub_t,
                                base / "tamper/out", base / "tamper/lib", FX_TARGET)
    case("single change / runner: stage copies the reference inputs (sha verified, no links), writes the iteration-12 "
         "exports, STAGE_IDENTITY with the binary and runner identity, and refuses an existing run root; a modified "
         "binary breaks the identity chain (X)", "1 single change",
         (ident["outer_iterations"], chk["why"], ident["run_binary_sha256"] == sha(ref / "input/lumina_cuda"),
          (n_new / "job.slurm").exists() and "readonly GPU_PROFILE=h200x2" in (n_new / "job.slurm").read_text(),
          (n_new / "OWNER.txt").exists() and (n_new / "SUBMIT_COMMAND.txt").exists(), again,
          sorted(os.listdir(n_new / "input")) == sorted(list(STAGED_FILES) + list(STAGED_DIRS) +
                                                        ["resolved_lumina.exports", "resolved_lumina.sha256"]),
          has(r_t, "LINK_BINARY_IDENTITY")),
         (FX_OUTER, [], True, True, True, "REFUSED", True, True), fault="binary tamper", results=[r_t])
    # ---- 2 run count and resources: the runner's own selftest (bash, real rc)
    rs = det_nrb_judge_runner_selftest(runner, base)
    want_cases = {"rerun_reject", "h200x2_one_gpu", "a100_profile_reject", "time_limit_over_ceiling", "iterations_6",
                  "iterations_13", "nrb_off", "requeued_reject", "h100x2_low_memory"}
    got_cases = {c["case"] for c in rs["cases"] if c["expected"] == "fail" and c["rc"] != 0}
    case("run count / resources: the GPU runner selftest (separate bash process, real rc) rejects re-entry, GPU count / kind / "
         "memory, the a100 profile, a time limit over the ceiling, requeue, 6 / 13 iterations and NRB off before the "
         "model; constants registered (12 iterations, 75804 s); profiles h200x2 / h100x2 only", "2 run count",
         (rs["rc"], want_cases <= got_cases, rs["constants_registered"], rs["profiles_in_runner"], rs["positive"] >= 3),
         (0, True, True, ["h100x2", "h200x2"], True))
    res[-1]["runner_selftest"] = {k: v for k, v in rs.items() if k != "cases"} | {"n_cases": len(rs["cases"])}
    # ---- 3 generation linkage faults
    r3 = {}
    for fl in ("beta_current", "material_plus1", "capture_fixed7", "last_as_observed"):
        r3[fl], *_ = run(f"gen_{fl}", fault=fl)
    case("generation linkage: the current call's radiation for the prediction (X), material shifted by one (X), the "
         "capture pinned to generation 7 (X), the last population 11 used as the observed 10 (X)", "3 generation linkage",
         (has(r3["beta_current"], "P1B") or r3["beta_current"]["codes"]["P1B_CORE"]["code"] == "X",
          has(r3["material_plus1"], "P1B") or r3["material_plus1"]["codes"]["P1B_CORE"]["code"] == "X",
          has(r3["capture_fixed7"], "LINK_CAPTURE"), has(r3["last_as_observed"], "LINK_NE_DUMP_OBS_NOT_CAPTURE") or
          has(r3["last_as_observed"], "DUMP_GENERATION_FILLED")),
         (True, True, True, True), fault="beta_current|material_plus1|capture_fixed7|last_as_observed",
         results=list(r3.values()))
    # ---- 4 P2 terminal window
    def series_past_only():
        # small changes at generations 5-6 only, large again at the end
        out = {}
        for Z, q in ION_QTY:
            b = 1.0e6 * (1 + Z / 100 + q)
            out[(Z, q)] = {g: b * (1.0 + (0.0 if g in (5, 6, 4) else 0.05 * ((g % 2) + 1))) for g in G["pops"]}
        return out
    r4a, o4a, *_ = run("p2_past_only", ion_series=series_past_only())
    c4a = r4a["codes"]["P2:ion"]
    def series_one_small():
        out = {}
        for Z, q in ION_QTY:
            b = 1.0e6 * (1 + Z / 100 + q)
            out[(Z, q)] = {g: b * (1.0 + (0.1 if (g % 2 and g <= 9) or g == 10 else 0.0)) * (1.2 if g == 11 else 1.0)
                           for g in G["pops"]}
        return out
    r4b, o4b, *_ = run("p2_single", fault="single_change", ion_series=series_one_small())
    r4c, o4c, *_ = run("p2_past", fault="past_window", ion_series=series_past_only())
    r4d, o4d, *_ = run("p2_mixed", fault="mixed_windows", ion_series=series_past_only())
    def series_end_higher():
        out = {}
        for Z, q in ION_QTY:
            b = 1.0e6 * (1 + Z / 100 + q)
            out[(Z, q)] = {g: b * (1.0 + 0.2 * (g - 3) * ((g % 2))) for g in G["pops"]}
        return out
    r4e, o4e, *_ = run("p2_end_higher", ion_series=series_end_higher())
    def series_unavailable():
        out = {}
        for Z, q in ION_QTY:
            b = 1.0e6 * (1 + Z / 100 + q)
            out[(Z, q)] = {g: (b if not (Z == 14 and q == 1 and g == 10) else 0.0) for g in G["pops"]}
        return out
    r4f, o4f, *_ = run("p2_zero_cross", ion_series=series_unavailable())
    case("P2 terminal window: the ion axis small only in the past windows (4,5),(5,6) is NOT_CONVERGED with the past "
         "window recorded; accepting one small terminal change, a past window, or mixed windows is refused by the "
         "recomputation (X on the axis, decision INADMISSIBLE); an ion series rising at the end gives END_HIGHER; a zero "
         "crossing at generation 10 refutes the ion axis (infinite M kept) and the run stays VALID", "4 P2 terminal window",
         (r4a["validity"], c4a["state"], c4a["any_past_window_small"], r4a["P2_disposition"],
          has(r4b, "P2_CODE_RECOMPUTE"), r4b["P2_disposition"], has(r4c, "P2_CODE_RECOMPUTE"),
          has(r4d, "P2_CODE_RECOMPUTE"), r4e["codes"]["P2:ion"]["descriptor"]["code"], r4e["codes"]["P2:ion"]["state"],
          r4f["codes"]["P2:ion"]["state"], r4f["validity"]),
         ("VALID", "REFUTED", True, "NOT_CONVERGED", True, "INADMISSIBLE", True, True, "END_HIGHER", "REFUTED",
          "REFUTED", "VALID"), fault="single_change|past_window|mixed_windows", results=[r4a, r4b, r4c, r4d, r4e, r4f])
    case("P2 descriptor: clean axes converge (END_LOWER on every axis); ion/n_e generation 11 (last material update) is "
         "recorded apart from the terminal window; n_quantities + missing + structural zero = registered count on every "
         "row", "4 P2 terminal window",
         ([r0["codes"][f"P2:{ax}"]["descriptor"]["code"] for ax in ("ion", "ne", "jbar", "level")],
          r0["P2_last_material_update"]["ne"]["generation"], r0["codes"]["P2:ion"]["terminal_window"],
          bool(((st_.n_quantities + st_.missing + st_.structural_zero) == st_.registered_quantities).all()),
          sorted(set(st_.registered_quantities))),
         (["END_LOWER"] * 4, 11, "9,10", True, [2, 8, 12]), results=[r0])
    def drop_row(r):
        p = r / "work/w5_lines.csv"
        lines = p.read_text().split("\n")
        keep = [ln for ln in lines if not ln.startswith(f"{G['g_obs']},CMFGEN_NONOVERLAP_SOBOLEV,{G['g_obs']},{G['g_obs']},10,1,")]
        p.write_text("\n".join(keep))
    r4g, o4g, *_ = run("p2_missing", mutate_N=drop_row)
    st4g = pd.read_csv(o4g / "stationarity.csv")
    jb = st4g[(st4g.axis == "jbar") & (st4g.generation == G["g_obs"])].iloc[0]
    r4h, o4h, *_ = run("p2_drop_large", fault="drop_large")
    r4i, o4i, *_ = run("p2_drop_zero", fault="drop_zero_cross", ion_series=series_unavailable())
    case("P2 denominators: a removed terminal W5 row is a missing quantity (counted, axis NOT_IDENTIFIABLE at the terminal "
         "window, descriptor END_UNAVAILABLE, run VALID); deleting large cells or a zero crossing fails the quantity "
         "accounting (X)", "4 P2 terminal window",
         (r4g["validity"], int(jb.missing), r4g["codes"]["P2:jbar"]["state"], r4g["codes"]["P2:jbar"]["descriptor"]["code"],
          has(r4h, "P2_QUANTITY_ACCOUNTING"), has(r4i, "P2_QUANTITY_ACCOUNTING")),
         ("VALID", 1, "NOT_IDENTIFIABLE", "END_UNAVAILABLE", True, True), fault="drop_large|drop_zero_cross",
         results=[r4g, r4h, r4i])
    # ---- 5 public ions
    r5 = {}
    for fl in ("saha_column", "q_shift", "renormalize", "time_shift", "space_shift"):
        r5[fl], *_ = run(f"ion_{fl}", fault=fl)
    r5x, o5x, *_ = run("ion_refuted", pub_factor={(14, 10): 3.0})
    i5 = pd.read_csv(o5x / "ion_fraction.csv")
    i5 = i5[(i5.run == "N") & (i5.Z == 14) & (i5.shell == 10) & (i5.ion != "III/II")]
    case("public ions: NLTE q1 / q2 over the element total at generation 10, the fixed 19.480 block and the registered "
         "vel_mid; the SAHA stage column, a q0/q1 shift, II+III renormalisation, another time block and a moved spatial "
         "match are refused (X); a public value 3x off refutes that element (code 1, VALID)", "5 public ions",
         (has(r5["saha_column"], "ION_COLUMN_RECOMPUTE"), has(r5["q_shift"], "ION_PUBLIC_TOKEN_RECOMPUTE"),
          has(r5["renormalize"], "ION_COLUMN_RECOMPUTE"), has(r5["time_shift"], "ION_PUBLIC_TOKEN_RECOMPUTE"),
          has(r5["space_shift"], "ION_VMID_RECOMPUTE") or r5["space_shift"]["codes"]["IONS:14"]["code"] == "X",
          r5x["validity"], r5x["codes"]["IONS:14"]["code"], r5x["codes"]["IONS:16"]["code"], set(i5.status)),
         (True, True, True, True, True, "VALID", 1, 0, {"REFUTED"}),
         fault="saha_column|q_shift|renormalize|time_shift|space_shift", results=list(r5.values()) + [r5x])
    # ---- spectrum (inherited functions; new target linkage)
    def scale_band(lo, hi, k):
        def f_(r):
            p = r / "work/lumina_spectrum_det_obs.csv"
            rows = p.read_text().split("\n")
            out = [rows[0]]
            for ln in rows[1:]:
                if not ln:
                    continue
                x, y = ln.split(",")
                out.append(f"{x},{float(y) * (k if lo <= float(x) < hi else 1.0)!r}")
            p.write_text("\n".join(out) + "\n")
        return f_
    r6a, o6a, *_ = run("spec_retarget", fault="retarget")
    r6b, o6b, *_ = run("spec_subset", fault="subset_bands", spec_mutate=scale_band(5000.0, 6000.0, 1.5))
    r6c, o6c, *_ = run("spec_hide", fault="hide_opt")
    r6e, o6e, *_ = run("spec_tilt", spec_mutate=scale_band(0.0, 3000.0, 3.0))
    case("public spectrum: thresholds regenerated from N, a band subset declared as the whole, the OPT change hidden -> X; "
         "a 3x UV2500 excess refutes the registered targets (code 1, VALID) with the absolute OPT recorded",
         "6 public spectrum", (has(r6a, "SPECTRUM_CONDITION_RECOMPUTE"), has(r6b, "SPECTRUM_OVERALL_RECOMPUTE"),
                               has(r6c, "SPECTRUM_OPT_CHANGE_NOT_RECORDED"), r6e["validity"], r6e["codes"]["SPECTRUM"]["code"],
                               r6e["codes"]["SPECTRUM"]["conditions"].get("UV2500"), r6e["spectrum"]["summary"]["N"].get("L_OPT") is not None),
         (True, True, True, "VALID", 1, 1, True), fault="retarget|subset_bands|hide_opt|tilt", results=[r6a, r6b, r6c, r6e])
    # ---- codes / decision propagation (mass errors)
    r7, o7, *_ = run("flip_all", fault="status_flip_all+retarget")
    d7 = json.loads((o7 / "decision.json").read_text())
    r7b, o7b, *_ = run("move_c", fault="move_c_with_L")
    r7c, o7c, *_ = run("drop_counter", fault="drop_counter")
    spec_msgs = [w for w in r7.get("reasons", []) if w.startswith("SPECTRUM")]
    case("codes / decision propagation (mass errors): every P1(a) state flipped (72 findings, the stored message list "
         "capped at 50) together with regenerated spectral thresholds in the same run; the P1(a) scope and the later "
         "SPECTRUM scope are both X although the spectrum finding lies beyond the stored cap, unaffected scopes keep "
         "their codes, decision.json carries validity X; moving c with L in the recall is refused; a shortened sample "
         "is refused before any science code", "7 propagation",
         (r7["n_reasons"] > 50, len(r7["reasons"]), spec_msgs, r7["codes"]["P1A"]["code"], r7["codes"]["SPECTRUM"]["code"],
          r7["codes"]["P1B_CORE"]["code"], d7["validity"], has(r7b, "P1A_RECALL_BACKGROUND_MOVED"),
          r7b["codes"]["P1A_LOCAL"]["code"], has(r7c, "SAMPLE_NOT_REGISTERED")),
         (True, 50, [], "X", "X", 0, "X", True, "X", True), fault="status_flip_all+retarget|move_c_with_L|drop_counter",
         results=[r7, r7b, r7c])
    # ---- preservation
    try:
        det_nrb_judge_analyze(runs0["N"], runs0["E"], runs0["R"], A0, st0, pub0["expectations"], src, pub0, o0,
                              base / "clean/lib2", FX_TARGET)
        pres = "WROTE"
    except FileExistsError:
        pres = "REFUSED_BEFORE_WRITING"
    case("preservation: an existing result path is refused before writing", "8 preservation", pres,
         "REFUSED_BEFORE_WRITING", kind="INHERITED_RERUN")
    # ---- inheritance record
    inh = det_nrb_judge_inherited_bundles(prior_selftest)
    case("inherited bundles: the SEEDMEM-SCOPE r4 selftest (readout sha equal to the imported module, failures 0) carries "
         "the P1(a), P1(b), inversion / authority, P2 denominators, P2 final code, spectrum, propagation and preservation "
         "cases this readout reuses by function identity", "9 inheritance",
         (inh["dependency_sha_matches"], inh["failures"], sorted(inh["bundles"]) == sorted(inh["required"])),
         (True, 0, True), kind="INHERITED")
    res[-1]["inherited"] = inh
    return res


INHERITED_BUNDLES = {"4 inversion": "P1(b) / recovery", "5 authority": "independent authority", "6 NRB link": "P1(a)",
                     "7 cross": "P1(a) local invariance", "10 P2": "P2 denominators / final code",
                     "11 numerics": "codes / propagation", "12 preservation": "preservation",
                     "3 missing": "P2 missing / zero"}


def det_nrb_judge_inherited_bundles(prior_selftest):
    p = Path(prior_selftest) / "selftest.json"
    st = json.loads(p.read_text())
    seed_sha = sha(Path(SEED.__file__).resolve())
    cases = st.get("cases", [])
    bundles = sorted({c.get("bundle") for c in cases if c.get("bundle") in INHERITED_BUNDLES})
    return {"path": str(p), "sha256": sha(p), "prior_readout_sha256": st.get("readout_sha256"), "seed_module_sha256": seed_sha,
            "dependency_sha_matches": st.get("readout_sha256") == seed_sha, "failures": st.get("failures"),
            "bundles": bundles, "required": sorted(INHERITED_BUNDLES),
            "cases": [{"bundle": c.get("bundle"), "case": c.get("case")[:160], "pass": c.get("pass"), "fault": c.get("fault")}
                      for c in cases if c.get("bundle") in INHERITED_BUNDLES],
            "spectrum_functions": "REP.det_siiii_repair_spectrum_table / _le / _lt (sha in manifest imports)"}


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
               "bundles_inherited": sorted(INHERITED_BUNDLES),
               "rc_meaning": "analyze_rc_equivalent = the rc cmd_analyze returns for that validity (0 VALID, 2 X); the fixture "
                             "analyses are function calls; the runner selftest is a separate bash process with its real rc",
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


def cmd_analyze(a):
    out = Path(a.out)
    sj = json.loads((Path(a.selftest) / "selftest.json").read_text())
    if sj.get("readout_sha256") != sha(Path(__file__).resolve()) or sj.get("failures") != 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": ["SELFTEST_NOT_THIS_READOUT_OR_FAILED"]}))
        return 2
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=False)
    public = {"spectrum": a.public_spectrum, 14: a.public_si, 16: a.public_s}
    res = det_nrb_judge_analyze(a.run, a.baseline_run, a.reference_run, a.prior_analysis, a.prior_selftest,
                                a.spectral_expectations, a.production_source, public, out / "analysis", out / "lib")
    inputs = {}
    for tag, rp in (("N", a.run), ("E", a.baseline_run), ("R", a.reference_run)):
        for rel in ("STAGE_IDENTITY.txt", "RUN_FOOTER.txt", "input/run_build_identity.json", "input/resolved_lumina.exports",
                    "input/binary.sha256", "input/deck.sha256", "input/fixed_te_profile.txt", "work/det_obs_capture/manifest.json",
                    "work/det_obs_capture/meta.txt", "work/det_obs_status.json", "work/cmfgen_final_publication.json",
                    "work/lumina_spectrum_det_obs.csv", "stdout.log", "stderr.log"):
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
    man = det_nrb_judge_manifest(a, out, {"kind": "analyze", "inputs": inputs,
                                          "selftest": {"path": str(Path(a.selftest) / "selftest.json"),
                                                       "sha256": sha(Path(a.selftest) / "selftest.json")},
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
    s = sp.add_parser("analyze")
    for k in ("--prereg", "--run", "--baseline-run", "--reference-run", "--prior-analysis", "--prior-selftest",
              "--spectral-expectations", "--production-source", "--public-spectrum", "--public-si", "--public-s",
              "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit({"selftest": cmd_selftest, "preflight": cmd_preflight, "stage": cmd_stage, "analyze": cmd_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
