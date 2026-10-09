#!/usr/bin/env python3
"""DET-COLL-PRESC readout (prereg docs/PREREG_DET_COLL_PRESC_2026-10-09.md).

One registered run O: the R' sealed binary and inputs, the N exports (12 outer iterations, DET-NRB-JUDGE) with
LUMINA_OMEGA_CMFGEN=1 added (COLL_FIX kept), cold start.  Primary output: one CMFGEN divergence table with the columns
CMFGEN, N, O at material generation 13 — S II/III and Si II/III fractions at the five registered shells, the eight band colours, R8
and L_OPT/public — and the per-row movement code of prereg section 6.  The public-comparison functions are imported
from the DET-NRB-JUDGE and DET-SIIII-REPAIR readouts unmodified.

Subcommands
  selftest       --runner GPU_RUNNER --out DIR
  prepare        --runner GPU_RUNNER --gpu-profile h200x2|h100x2 --out DIR   (selftest, N table, stage O)
  analyze        --selftest DIR --out DIR                                     (O column, codes, evidence)
  check-exports  REF_EXPORTS NEW_EXPORTS REF_RUN NEW_RUN
  check-omega    O_STDOUT R_STDOUT
  check-table    ROWS_CSV
Exit codes: 0 pass / VALID, 2 refused / X, 3 existing output path.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_nrb_judge_readout as NRB                                              # noqa: E402
import det_siiii_repair_readout as REP                                           # noqa: E402
from det_siiii_pop_readout import I, sha                                         # noqa: E402

P = Path(__file__).resolve().parent.parent
PREREG = "docs/PREREG_DET_COLL_PRESC_2026-10-09.md"
OUTER, G_OBS = 12, 13
BASE = Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009")
RPRIME = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1")      # binary / input lineage
NRUN = Path("/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1")              # baseline N (12 iterations)
ORUN = BASE / "OMEGA/run1"
PUB = P / "data/standart_data1/toy06"
PUBLIC = {"spectrum": PUB / "spectra_toy06_cmfgen.txt", 14: PUB / "ionfrac_si_toy06_cmfgen.txt",
          16: PUB / "ionfrac_s_toy06_cmfgen.txt"}
TARGET = NRB.TARGET
SHELLS = TARGET["shells"]
OMEGA_KEY = "LUMINA_OMEGA_CMFGEN"
OMEGA_LINE = 'declare -x LUMINA_OMEGA_CMFGEN="1"'
TIME_LIMIT, TIME_CEILING_S, GPU_HOURS = "21:00:00", 75600, 42.0
OMEGA_RE = re.compile(r"\[OMEGA-CMFGEN\] ON: (\d+) Omega tables \(feiii\+col_ion\) \| of (\d+) bb lines (\d+) tabulated "
                      r"\(no floor\), (\d+) van-Regemorter gbar-floor, (\d+) OMEGA_SET=(\S+) ")
SPECIES = ((16, 2, "S II"), (16, 3, "S III"), (14, 2, "Si II"), (14, 3, "Si III"))
BANDS = [f"{a}-{b}" for a, b in REP.BANDS8]
LABEL = {0: "TOWARD", 1: "NOT_TOWARD", 4: "UNRESOLVED", 5: "NOT_IDENTIFIABLE"}


# ------------------------------------------------------------------ G-LINK: the single change
def det_coll_presc_add_omega(text):
    """the O exports: the Omega line inserted at its sorted position (the R' exports are a sorted declare -x list)."""
    lines = [ln for ln in text.split("\n") if ln]
    if any(ln.startswith(f"declare -x {OMEGA_KEY}=") for ln in lines):
        raise ValueError("OMEGA_ALREADY_PRESENT")
    keys = [ln.split("=", 1)[0] for ln in lines]
    pos = next((i for i, k in enumerate(keys) if k > f"declare -x {OMEGA_KEY}"), len(lines))
    return "\n".join(lines[:pos] + [OMEGA_LINE] + lines[pos:]) + "\n"


def det_coll_presc_exports_check(ref_path, new_path, ref_run, new_run):
    """prereg 2: against the N exports, after reversing the path-key re-rooting the only difference is
    LUMINA_OMEGA_CMFGEN absent -> "1"."""
    ref_run, new_run = str(Path(ref_run)), str(Path(new_run))
    rk = {k: v for k, v, _ in NRB.det_nrb_judge_exports_lines(ref_path) if k}
    nk = {k: v for k, v, _ in NRB.det_nrb_judge_exports_lines(new_path) if k}
    why = []
    for k in sorted(set(rk) | set(nk)):
        if k == OMEGA_KEY:
            if k in rk or nk.get(k) != "1":
                why.append(f"OMEGA_NOT_ADDED:{rk.get(k)}->{nk.get(k)}")
        elif k not in rk:
            why.append(f"KEY_ADDED:{k}")
        elif k not in nk:
            why.append(f"KEY_REMOVED:{k}")
        elif k in NRB.PATH_KEYS:
            if not rk[k].startswith(ref_run + "/") or nk[k] != new_run + rk[k][len(ref_run):]:
                why.append(f"PATH_KEY:{k}")
        elif rk[k] != nk[k]:
            why.append(f"KEY_DIFFERS:{k}:{rk[k]}!={nk[k]}")
    return why


# ------------------------------------------------------------------ G-OMEGA: the application evidence
def det_coll_presc_omega(o_stdout, r_stdout):
    """prereg 4: exactly one [OMEGA-CMFGEN] ON: line in O (tables > 0, every tier > 0, tiers sum to the bb lines,
    OMEGA_SET = 0.1) and none in the baseline N."""
    why, rec = [], None
    hits = [m for m in OMEGA_RE.finditer(Path(o_stdout).read_text(errors="replace"))]
    if len(hits) != 1:
        why.append(f"OMEGA_ON_LINE_COUNT:{len(hits)}")
    else:
        t, n, t1, t2, t3, oset = hits[0].groups()
        rec = {"tables": int(t), "bb_lines": int(n), "tier1": int(t1), "tier2": int(t2), "tier3": int(t3),
               "omega_set": float(oset)}
        if not (rec["tables"] > 0 and min(rec["tier1"], rec["tier2"], rec["tier3"]) > 0):
            why.append("OMEGA_TIER_EMPTY")
        if rec["tier1"] + rec["tier2"] + rec["tier3"] != rec["bb_lines"]:
            why.append("OMEGA_TIER_SUM")
        if rec["omega_set"] != 0.1:
            why.append(f"OMEGA_SET:{rec['omega_set']}")
    if "[OMEGA-CMFGEN] ON:" in Path(r_stdout).read_text(errors="replace"):
        why.append("OMEGA_ON_IN_REFERENCE")
    return rec, why


# ------------------------------------------------------------------ G-RUN: completion and observation generation
def det_coll_presc_run_check(run):
    run, why = Path(run), []
    ft = NRB.det_nrb_judge_kv(run / "RUN_FOOTER.txt") if (run / "RUN_FOOTER.txt").exists() else {}
    if ft.get("terminal_state") != "SUCCESS" or ft.get("model_rc") != "0":
        why.append(f"RUN_NOT_SUCCESS:{ft.get('terminal_state')}")
    if ft.get("outer_iterations") != str(OUTER):
        why.append(f"OUTER_ITERATIONS:{ft.get('outer_iterations')}")
    man = run / "work/det_obs_capture/manifest.json"
    g = json.loads(man.read_text()).get("generations", {}) if man.exists() else {}
    if g.get("material") != G_OBS or g.get("radiation") != G_OBS:
        why.append(f"CAPTURE_GENERATION:{g.get('material')}:{g.get('radiation')}")
    return ft, why


# ------------------------------------------------------------------ G-TABLE: rows of the registered comparison
def det_coll_presc_ion_rows(runs, target=TARGET):
    """S / Si II, III fractions of each run at material generation 13 (NRB functions, registered vel_mid mapping)."""
    pub_blocks = {Z: NRB.det_nrb_judge_public_block(PUBLIC[Z], target["public_time"]) for Z in target["Z"]}
    rows, why = [], []
    for tag, run in runs.items():
        dump, derr = NRB.det_nrb_judge_ion_dump(run, G_OBS, SHELLS)
        if derr:
            why.append(f"{tag}:{derr}")
        rws, _ = NRB.det_nrb_judge_ions({"tag": tag, "shells": SHELLS}, dump, pub_blocks, target)
        rows += [r for r in rws if r["ion"] != "III/II"]
    return rows, why


def det_coll_presc_table_check(rows, tags):
    """every (run, Z, ion, shell) row present once, at generation 13, matched to the registered public vel_mid."""
    why, seen = [], {}
    for r in rows:
        k = (r["run"], int(r["Z"]), int(r["ion"]), int(r["shell"]))
        seen[k] = seen.get(k, 0) + 1
        if r.get("generation") is None or int(r["generation"]) != G_OBS:
            why.append(f"ROW_GENERATION:{k}")
        if r.get("v_mid_public_matched") != TARGET["vmid_public"][k[3]]:
            why.append(f"SHELL_MAPPING:{k}:{r.get('v_mid_public_matched')}")
    for tag in tags:
        for Z, q, _ in SPECIES:
            for s in SHELLS:
                if seen.get((tag, Z, q, s)) != 1:
                    why.append(f"ROW_COUNT:{tag}:{Z}:{q}:{s}:{seen.get((tag, Z, q, s), 0)}")
    return why


def det_coll_presc_spectra(runs):
    specs = {"public": REP.det_siiii_repair_public(PUBLIC["spectrum"])}
    for tag, run in runs.items():
        p = Path(run) / "work/lumina_spectrum_det_obs.csv"
        specs[tag] = REP.det_siiii_repair_spec(p) if p.exists() else None
    rows, summ = REP.det_siiii_repair_spectrum_table(specs)
    opt = {k: (REP.det_siiii_repair_bands(v) or {}).get(REP.OPT) if v is not None else None for k, v in specs.items()}
    return rows, summ, opt


# ------------------------------------------------------------------ prereg 6: movement codes
def det_coll_presc_move(dR, dO, equal):
    """0 |d_O| below |d_N| (interval-separated), 1 above or exactly equal, 4 not separated, 5 undefined."""
    if dR is None or dO is None:
        return 5
    if equal:
        return 1
    aR, aO = dR.abs(), dO.abs()
    if aO.hi < aR.lo:
        return 0
    if aO.lo > aR.hi:
        return 1
    return 4


def iv(r, key):
    lo, mid, hi = r.get(f"{key}_lo"), r.get(f"{key}_mid"), r.get(f"{key}_hi")
    return None if lo is None or pd.isna(lo) else I(lo, mid, hi)


def det_coll_presc_table(ion_rows, spec_rows, summ, opt, tags):
    """the CMFGEN divergence table: 20 fraction rows, 8 colour rows, R8, L_OPT/public (prereg 5)."""
    out = []
    idx = {(r["run"], r["Z"], r["ion"], r["shell"]): r for r in ion_rows}
    for Z, q, name in SPECIES:
        for s in SHELLS:
            rec = {"kind": "fraction", "item": name, "shell": s, "v_mid_public": TARGET["vmid_public"][s]}
            for tag in tags:
                r = idx.get((tag, Z, q, s))
                rec["CMFGEN"] = None if r is None else r.get("f_public_mid")
                rec[f"{tag}_value"] = None if r is None else r.get("f_N_mid")
                d = None if r is None else iv(r, "d_f")
                for k_ in ("lo", "mid", "hi"):
                    rec[f"{tag}_d_{k_}"] = None if d is None else getattr(d, k_)
            if "O" in tags:
                rR, rO = idx.get(("N", Z, q, s)), idx.get(("O", Z, q, s))
                equal = rR is not None and rO is not None and rR.get("n_nlte") == rO.get("n_nlte") and \
                    rR.get("n_element_total") == rO.get("n_element_total")
                rec["equal"] = equal
                rec["code"] = det_coll_presc_move(None if rR is None else iv(rR, "d_f"), None if rO is None else iv(rO, "d_f"), equal)
            out.append(rec)
    sidx = {(r["spectrum"], r["band"]): r for r in spec_rows}
    same_spec = "O" in tags and all(
        (sidx.get(("N", b)) or {}).get("status") == "OK" and (sidx.get(("O", b)) or {}).get("status") == "OK" and
        sidx[("N", b)]["L_band"] == sidx[("O", b)]["L_band"] and sidx[("N", b)]["L_OPT"] == sidx[("O", b)]["L_OPT"]
        for b in BANDS)
    for b in BANDS:
        rec = {"kind": "colour", "item": b, "shell": None, "v_mid_public": None}
        for tag in tags:
            r = sidx.get((tag, b))
            rec["CMFGEN"] = None if r is None or r.get("status") != "OK" else r.get("colour_public")
            rec[f"{tag}_value"] = None if r is None or r.get("status") != "OK" else r.get("colour")
            d = None if r is None or r.get("status") != "OK" else iv(r, "d_dex")
            for k_ in ("lo", "mid", "hi"):
                rec[f"{tag}_d_{k_}"] = None if d is None else getattr(d, k_)
        if "O" in tags:
            rR, rO = sidx.get(("N", b)), sidx.get(("O", b))
            ok = all(r is not None and r.get("status") == "OK" for r in (rR, rO))
            equal = ok and rR["L_band"] == rO["L_band"] and rR["L_OPT"] == rO["L_OPT"]
            rec["equal"] = equal
            rec["code"] = det_coll_presc_move(iv(rR, "d_dex") if ok else None, iv(rO, "d_dex") if ok else None, equal)
        out.append(rec)
    rec = {"kind": "R8", "item": "R8", "shell": None, "v_mid_public": None, "CMFGEN": 0.0}
    r8 = {}
    for tag in tags:
        sm = summ.get(tag) or {}
        r8[tag] = None if sm.get("status") != "OK" or sm.get("R8") is None else I(sm["R8"]["lo"], sm["R8"]["mid"], sm["R8"]["hi"])
        rec[f"{tag}_value"] = None if r8[tag] is None else r8[tag].mid
        for k_ in ("lo", "mid", "hi"):
            rec[f"{tag}_d_{k_}"] = None if r8[tag] is None else getattr(r8[tag], k_)
    if "O" in tags:
        rec["equal"] = same_spec
        rec["code"] = det_coll_presc_move(r8["N"], r8["O"], same_spec)
    out.append(rec)
    rec = {"kind": "OPT", "item": "L_OPT/public", "shell": None, "v_mid_public": None, "CMFGEN": 1.0,
           "L_OPT_public": None if opt.get("public") is None else opt["public"].mid}
    la = {}
    for tag in tags:
        a = None if opt.get(tag) is None or opt.get("public") is None else opt[tag].div(opt["public"])
        la[tag] = NRB.det_nrb_judge_log10i(a)
        rec[f"{tag}_value"] = None if a is None else a.mid
        rec[f"L_OPT_{tag}"] = None if opt.get(tag) is None else opt[tag].mid
        for k_ in ("lo", "mid", "hi"):
            rec[f"{tag}_d_{k_}"] = None if la[tag] is None else getattr(la[tag], k_)
    if "O" in tags:
        equal = opt.get("N") is not None and opt.get("O") is not None and opt["N"].mid == opt["O"].mid
        rec["equal"] = equal
        rec["code"] = det_coll_presc_move(la["N"], la["O"], equal)
    out.append(rec)
    for rec in out:
        if "code" in rec:
            rec["label"] = "EQUAL" if rec["code"] == 1 and rec["equal"] else LABEL[rec["code"]]
    return out


def det_coll_presc_counts(table):
    """prereg 6/10: per species (denominator 5) and over the eight bands: TOWARD / NOT_TOWARD / EQUAL / UNRESOLVED /
    NOT_IDENTIFIABLE; R8 and OPT rows as single comparisons.  No weighted overall score."""
    cnt = {}
    for rec in table:
        if "label" not in rec:
            continue
        grp = rec["item"] if rec["kind"] == "fraction" else "colour (8 bands)" if rec["kind"] == "colour" else rec["item"]
        c = cnt.setdefault(grp, {k: 0 for k in ("TOWARD", "NOT_TOWARD", "EQUAL", "UNRESOLVED", "NOT_IDENTIFIABLE")})
        c[rec["label"]] += 1
    return cnt


def fmt(v, f="{:.3g}"):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f.format(v)


def det_coll_presc_write(out, table, tags, counts=None):
    pd.DataFrame(table).to_csv(out / "cmfgen_comparison.csv", index=False)
    head = ["항목", "셸", "CMFGEN"] + [f"{t} 값 (d dex)" for t in tags] + (["이동"] if "O" in tags else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table:
        cells = [r["item"], fmt(r["shell"], "{}"), fmt(r.get("CMFGEN"))]
        for t in tags:
            cells.append(f"{fmt(r.get(f'{t}_value'))} ({fmt(r.get(f'{t}_d_mid'), '{:+.3f}')})")
        if "O" in tags:
            cells.append(r.get("label", "—"))
        lines.append("| " + " | ".join(cells) + " |")
    opt = next(r for r in table if r["kind"] == "OPT")
    lines.append("")
    lines.append("L_OPT 절대값 (erg/s): CMFGEN " + fmt(opt.get("L_OPT_public")) + "".join(
        f", {t} {fmt(opt.get(f'L_OPT_{t}'))}" for t in tags))
    lines.append("d = log10(Lumina/CMFGEN): 분율은 f/f_pub, 색은 C_b/C_b^pub, OPT 행은 log10 A_OPT, R8 행은 R8 값 자체.")
    if counts:
        lines += ["", "| 묶음 | TOWARD | NOT_TOWARD | EQUAL | UNRESOLVED | NOT_IDENTIFIABLE |", "|---|---|---|---|---|---|"]
        for g, c in counts.items():
            lines.append(f"| {g} | " + " | ".join(str(c[k]) for k in ("TOWARD", "NOT_TOWARD", "EQUAL", "UNRESOLVED",
                                                                      "NOT_IDENTIFIABLE")) + " |")
    (out / "cmfgen_comparison.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ CLI checks (also the selftest subjects)
def cmd_check_exports(a):
    why = det_coll_presc_exports_check(a.ref_exports, a.new_exports, a.ref_run, a.new_run)
    print(json.dumps({"check": "exports", "why": why}))
    return 2 if why else 0


def cmd_check_omega(a):
    rec, why = det_coll_presc_omega(a.o_stdout, a.r_stdout)
    print(json.dumps({"check": "omega", "record": rec, "why": why}))
    return 2 if why else 0


def cmd_check_table(a):
    rows = pd.read_csv(a.rows_csv, float_precision="round_trip").to_dict("records")
    tags = sorted({r["run"] for r in rows})
    why = det_coll_presc_table_check(rows, tags)
    print(json.dumps({"check": "table", "why": why[:20], "n_why": len(why)}))
    return 2 if why else 0


def cli(args):
    p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve())] + [str(x) for x in args],
                       capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-600:]


def det_coll_presc_selftest(runner, out):
    """prereg 8: the GPU runner selftest and three negative controls through the CLI (real exit codes)."""
    out.mkdir(parents=True)
    fx = out / "fixtures"
    fx.mkdir()
    recs = []

    def case(name, got, want, runs):
        recs.append({"case": name, "got": got, "want": want, "pass": got == want, "runs": runs})
    p = subprocess.run(["bash", str(runner), "--selftest"], capture_output=True, text=True)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    case("GPU runner selftest", (p.returncode, last.startswith("COLL_PRESC_RUNNER_SELFTEST_PASS")), (0, True),
         [{"rc": p.returncode, "last": last}])
    # 1 exports: Omega alone accepted, Omega + collisional ionisation refused
    ref_run, new_run = fx / "ref", fx / "new"
    text = (NRUN / "input/resolved_lumina.exports").read_text().replace(str(NRUN), str(ref_run))
    (fx / "ref.exports").write_text(text)
    good = det_coll_presc_add_omega(text.replace(str(ref_run), str(new_run)))
    (fx / "new_ok.exports").write_text(good)
    (fx / "new_collion.exports").write_text(good + 'declare -x LUMINA_NLTE_COLL_ION="1"\n')
    r1 = cli(["check-exports", fx / "ref.exports", fx / "new_ok.exports", ref_run, new_run])
    r2 = cli(["check-exports", fx / "ref.exports", fx / "new_collion.exports", ref_run, new_run])
    case("exports: Omega alone passes; adding LUMINA_NLTE_COLL_ION=1 is refused", (r1[0], r2[0], "KEY_ADDED:LUMINA_NLTE_COLL_ION" in r2[1]),
         (0, 2, True), [{"rc": r1[0], "tail": r1[1]}, {"rc": r2[0], "tail": r2[1]}])
    # 2 Omega evidence: the ON line present passes, removed is refused
    on = ("  [OMEGA-CMFGEN] ON: 19 Omega tables (feiii+col_ion) | of 100 bb lines 10 tabulated (no floor), "
          "80 van-Regemorter gbar-floor, 10 OMEGA_SET=0.1 (f<=1e-10)\n")
    (fx / "o_ok.log").write_text("x\n" + on + "y\n")
    (fx / "o_none.log").write_text("x\ny\n")
    (fx / "r.log").write_text("x\n")
    r3 = cli(["check-omega", fx / "o_ok.log", fx / "r.log"])
    r4 = cli(["check-omega", fx / "o_none.log", fx / "r.log"])
    case("Omega evidence: one ON line passes; the line removed is refused", (r3[0], r4[0], "OMEGA_ON_LINE_COUNT:0" in r4[1]),
         (0, 2, True), [{"rc": r3[0], "tail": r3[1]}, {"rc": r4[0], "tail": r4[1]}])
    # 3 table: N rows with the registered mapping pass; shell 15 assembled at shell 19's velocity is refused
    rows, _ = det_coll_presc_ion_rows({"N": NRUN})
    pd.DataFrame(rows).to_csv(fx / "rows_ok.csv", index=False)
    vm = dict(TARGET["vmid_lumina"])
    vm[15] = vm[19]
    rows_bad, _ = det_coll_presc_ion_rows({"N": NRUN}, dict(TARGET, vmid_lumina=vm))
    pd.DataFrame(rows_bad).to_csv(fx / "rows_shift.csv", index=False)
    r5 = cli(["check-table", fx / "rows_ok.csv"])
    r6 = cli(["check-table", fx / "rows_shift.csv"])
    case("table: registered shell mapping passes; shell 15 assembled at shell 19's vel_mid is refused",
         (r5[0], r6[0], "SHELL_MAPPING" in r6[1]), (0, 2, True), [{"rc": r5[0], "tail": r5[1]}, {"rc": r6[0], "tail": r6[1]}])
    res = {"schema": "DET_COLL_PRESC_SELFTEST_V1", "readout_sha256": sha(Path(__file__).resolve()),
           "runner_sha256": sha(runner), "records": [dict(r, got=str(r["got"]), want=str(r["want"])) for r in recs],
           "failures": sum(1 for r in recs if not r["pass"])}
    (out / "selftest.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def cmd_selftest(a):
    out = Path(a.out)
    if out.exists():
        print(f"OUT_EXISTS {out}")
        return 3
    res = det_coll_presc_selftest(Path(a.runner), out)
    print(json.dumps({k: res[k] for k in ("failures", "readout_sha256")}))
    return 0 if res["failures"] == 0 else 1


def cmd_prepare(a):
    """prereg 3: selftest, the CMFGEN / N columns, then stage O (NRB stage from the R' lineage at 12 iterations, as
    for N, plus the Omega line; the single-change check is against the N exports)."""
    out = Path(a.out)
    if out.exists():
        print(f"OUT_EXISTS {out}")
        return 3
    st = det_coll_presc_selftest(Path(a.runner), out / "selftest")
    if st["failures"]:
        print("SELFTEST_FAILED")
        return 2
    rows, why = det_coll_presc_ion_rows({"N": NRUN})
    why += det_coll_presc_table_check(rows, ["N"])
    _, rwhy = det_coll_presc_run_check(NRUN)
    why += [f"N:{w}" for w in rwhy]
    srows, summ, opt = det_coll_presc_spectra({"N": NRUN})
    table = det_coll_presc_table(rows, srows, summ, opt, ["N"])
    (out / "baseline").mkdir()
    det_coll_presc_write(out / "baseline", table, ["N"])
    pd.DataFrame(rows).to_csv(out / "baseline/ion_rows.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "baseline/spectrum_rows.csv", index=False)
    off = {"schema": "DET_COLL_PRESC_PREPARE_V1", "submit_allowed": not why, "why": why,
           "selftest_failures": st["failures"], "readout_sha256": sha(Path(__file__).resolve())}
    (out / "offline.json").write_text(json.dumps(off, indent=1))
    if why:
        print(json.dumps(off))
        return 2
    NRB.det_nrb_judge_stage(RPRIME, ORUN, Path(a.runner), a.gpu_profile, out, outer=OUTER, prereg=PREREG)
    ex = ORUN / "input/resolved_lumina.exports"
    ex.write_text(det_coll_presc_add_omega(ex.read_text()))
    (ORUN / "input/resolved_lumina.sha256").write_text(sha(ex) + "\n")
    ewhy = det_coll_presc_exports_check(NRUN / "input/resolved_lumina.exports", ex, NRUN, ORUN)
    if ewhy:
        raise ValueError(f"EXPORTS_CHECK:{ewhy}")
    old = NRB.det_nrb_judge_kv(ORUN / "STAGE_IDENTITY.txt")
    ident = {"schema": "COLL_PRESC_STAGE_V1", "run": str(ORUN), "gpu_profile": a.gpu_profile, "outer_iterations": OUTER,
             "run_binary_sha256": old["run_binary_sha256"], "run_build_identity_sha256": old["run_build_identity_sha256"],
             "effective_exports_sha256": sha(ex), "runner_sha256": sha(ORUN / "job.slurm"),
             "registered_runner_sha256": sha(a.runner), "contract": PREREG, "lineage_run": str(RPRIME),
             "lineage_exports_sha256": sha(RPRIME / "input/resolved_lumina.exports"), "baseline_run": str(NRUN),
             "baseline_exports_sha256": sha(NRUN / "input/resolved_lumina.exports"),
             "exports_delta_vs_baseline": f"{OMEGA_KEY} absent->1; path keys re-rooted: " + ",".join(NRB.PATH_KEYS),
             "prepare": str(out / "offline.json"), "prepare_sha256": sha(out / "offline.json"),
             "time_ceiling_s": TIME_CEILING_S, "gpu_hours_ceiling": GPU_HOURS, "staged_files": old["staged_files"]}
    (ORUN / "STAGE_IDENTITY.txt").write_text("".join(f"{k}={v}\n" for k, v in ident.items()))
    (ORUN / "OWNER.txt").write_text("project=LUMINA-SN\nrepo=Lumina-sn thenmc-macroatom-fluorescence\n"
                                    f"purpose=DET-COLL-PRESC run O (N exports + {OMEGA_KEY}=1, 12 iterations, "
                                    f"{a.gpu_profile}), user decision 2026-10-09\nowner=Claude driver session\nDO_NOT_CANCEL\n")
    (ORUN / "SUBMIT_COMMAND.txt").write_text(
        f"cd {ORUN} && sbatch --exclude=syn102,syn03 --time={TIME_LIMIT} --output={ORUN}/slurm-%j.out "
        f"--error={ORUN}/slurm-%j.err {ORUN}/job.slurm {ORUN}\n")
    sf = json.loads((ORUN / "STAGE_FILES.json").read_text())
    sf["omega_line_added"] = OMEGA_LINE
    (ORUN / "STAGE_FILES.json").write_text(json.dumps(sf, indent=1))
    print(json.dumps({"staged": str(ORUN), "identity": ident}))
    return 0


def cmd_analyze(a):
    """prereg 4-6, 8, 10: gates, the three-column table, movement codes, counts, evidence."""
    out = Path(a.out)
    if out.exists():
        print(f"OUT_EXISTS {out}")
        return 3
    out.mkdir(parents=True)
    st = json.loads((Path(a.selftest) / "selftest.json").read_text())
    gates = {"G-LINK": [], "G-RUN": [], "G-OMEGA": [], "G-TABLE": []}
    if st.get("failures") != 0 or st.get("readout_sha256") != sha(Path(__file__).resolve()):
        gates["G-LINK"].append("SELFTEST_NOT_CURRENT")
    gates["G-LINK"] += det_coll_presc_exports_check(NRUN / "input/resolved_lumina.exports",
                                                    ORUN / "input/resolved_lumina.exports", NRUN, ORUN)
    oid = NRB.det_nrb_judge_kv(ORUN / "STAGE_IDENTITY.txt")
    rid = NRB.det_nrb_judge_kv(RPRIME / "STAGE_IDENTITY.txt")
    if oid.get("run_binary_sha256") != rid.get("run_binary_sha256") or sha(ORUN / "input/lumina_cuda") != rid.get("run_binary_sha256"):
        gates["G-LINK"].append("BINARY_IDENTITY")
    if sha(ORUN / "input/resolved_lumina.exports") != oid.get("effective_exports_sha256"):
        gates["G-LINK"].append("EXPORTS_NOT_STAGED")
    ft = {}
    for tag, run in (("N", NRUN), ("O", ORUN)):
        f_, w_ = det_coll_presc_run_check(run)
        ft[tag] = f_
        gates["G-RUN"] += [f"{tag}:{w}" for w in w_]
    if ft["O"].get("effective_exports_sha256") != oid.get("effective_exports_sha256"):
        gates["G-RUN"].append("O:RUN_EXPORTS_NOT_STAGED")
    om, gates["G-OMEGA"] = det_coll_presc_omega(ORUN / "stdout.log", NRUN / "stdout.log")
    runs = {"N": NRUN, "O": ORUN}
    rows, w_ = det_coll_presc_ion_rows(runs)
    gates["G-TABLE"] += w_ + det_coll_presc_table_check(rows, ["N", "O"])
    srows, summ, opt = det_coll_presc_spectra(runs)
    if any((summ.get(t) or {}).get("status") != "OK" for t in runs):
        gates["G-TABLE"].append("SPECTRUM_NOT_OK")
    table = det_coll_presc_table(rows, srows, summ, opt, ["N", "O"])
    counts = det_coll_presc_counts(table)
    valid = not any(gates.values())
    det_coll_presc_write(out, table, ["N", "O"], counts)
    pd.DataFrame(rows).to_csv(out / "ion_rows.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "spectrum_rows.csv", index=False)
    ev = {"schema": "DET_COLL_PRESC_EVIDENCE_V1", "validity": "VALID" if valid else "X",
          "disposition": "PASS_LIMITED" if valid else "INADMISSIBLE", "gates": gates, "omega": om, "counts": counts,
          "footers": ft, "readout_sha256": sha(Path(__file__).resolve()), "selftest": str(a.selftest),
          "inputs": {"N_exports": sha(NRUN / "input/resolved_lumina.exports"), "O_exports": sha(ORUN / "input/resolved_lumina.exports"),
                     **{f"public_{k}": sha(v) for k, v in PUBLIC.items()}},
          "claim_ceiling": "12-iteration non-converged state of the deterministic lane; no production promotion, no physical "
                           "correctness, no convergence, no seed independence, no causal attribution to one ion or path"}
    (out / "evidence.json").write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps({"validity": ev["validity"], "counts": counts, "gates": gates}, default=str))
    return 0 if valid else 2


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--runner", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("prepare")
    s.add_argument("--runner", required=True)
    s.add_argument("--gpu-profile", required=True, choices=sorted(NRB.GPU_PROFILES))
    s.add_argument("--out", required=True)
    s = sp.add_parser("analyze")
    s.add_argument("--selftest", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("check-exports")
    for k in ("ref_exports", "new_exports", "ref_run", "new_run"):
        s.add_argument(k)
    s = sp.add_parser("check-omega")
    s.add_argument("o_stdout")
    s.add_argument("r_stdout")
    s = sp.add_parser("check-table")
    s.add_argument("rows_csv")
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "prepare": cmd_prepare, "analyze": cmd_analyze, "check-exports": cmd_check_exports,
            "check-omega": cmd_check_omega, "check-table": cmd_check_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
