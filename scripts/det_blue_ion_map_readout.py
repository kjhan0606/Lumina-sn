#!/usr/bin/env python3
"""DET-BLUE-ION-MAP readout (prereg docs/PREREG_DET_BLUE_ION_MAP_2026-10-10.md).

Reads the stored O state (coll_presc_20261009/OMEGA/run1, material generation 13) and writes one CMFGEN comparison
table of 309 rows: Fe / Co / Ni II and III fractions at all 50 shells against the public toy06 CMFGEN fractions at
19.480 d (nearest public vel_mid), and the eight absolute band luminosities plus OPT against the public spectrum at
19.48 d.  Codes: 0 below public, 1 above public, 4 sign not separated, 5 undefined, X inadmissible (prereg 6).

Subcommands
  analyze --out DIR      (runs the direct negative controls first; Slurm a10 only)
Exit codes: 0 VALID, 2 X / refused, 3 existing output path.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_nrb_judge_readout as NRB                                              # noqa: E402
import det_siiii_repair_readout as REP                                           # noqa: E402
import det_siiii_pop_readout as POP                                              # noqa: E402
from det_siiii_pop_readout import I                                              # noqa: E402

P = Path(__file__).resolve().parent.parent
PREREG = "docs/PREREG_DET_BLUE_ION_MAP_2026-10-10.md"
O = Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1")
PC = O / "work/physics_comparison"
MAN, ION, SHELL = PC / "physics_DET_iter0010.manifest.json", PC / "physics_DET_iter0010.ion.csv", \
    PC / "physics_DET_iter0010.shell.csv"
CAPMAN, SPEC = O / "work/det_obs_capture/manifest.json", O / "work/lumina_spectrum_det_obs.csv"
PUB = P / "data/standart_data1/toy06"
PUBLIC_ION = {26: PUB / "ionfrac_fe_toy06_cmfgen.txt", 27: PUB / "ionfrac_co_toy06_cmfgen.txt",
              28: PUB / "ionfrac_ni_toy06_cmfgen.txt"}
PUBLIC_SPEC = PUB / "spectra_toy06_cmfgen.txt"
SYM = {26: "fe", 27: "co", 28: "ni"}
ROMAN = {1: "II", 2: "III"}
TIME = "19.480"
NS = 50
G_POP, G_RAD, ITER = 13, 12, 10
CAP_GEN = {"material": 13, "radiation": 13, "iter": 11}
BANDS = [(2500, 3000), (3000, 3500), (3500, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000),
         (8000, 10000), (4000, 6000)]
LABEL = {0: "BELOW_PUBLIC", 1: "ABOVE_PUBLIC", 4: "UNRESOLVED", 5: "NOT_IDENTIFIABLE"}


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


# ------------------------------------------------------------------ G-INPUT
def det_blue_ion_map_inputs(man, capman, spec_path):
    """prereg 2: the fixed dump generation, the observation capture and the stored spectrum identity."""
    why = []
    if (man.get("population_generation"), man.get("radiation_generation"), man.get("iteration")) != (G_POP, G_RAD, ITER):
        why.append(f"G-INPUT:DUMP_GENERATION:{man.get('population_generation')}:{man.get('radiation_generation')}:"
                   f"{man.get('iteration')}")
    if man.get("transaction_status") != "COMMITTED":
        why.append("G-INPUT:DUMP_NOT_COMMITTED")
    g = capman.get("generations", {})
    if any(g.get(k) != v for k, v in CAP_GEN.items()):
        why.append(f"G-INPUT:CAPTURE_GENERATION:{g.get('material')}:{g.get('radiation')}:{g.get('iter')}")
    rec = (capman.get("outputs") or {}).get(Path(spec_path).name) or {}
    if rec.get("sha256") != sha(spec_path) or rec.get("bytes") != Path(spec_path).stat().st_size:
        why.append("G-INPUT:SPECTRUM_IDENTITY")
    return why


# ------------------------------------------------------------------ public ion block with its column names
def det_blue_ion_map_public(path, time_label=TIME):
    """the exact time block: column names from its header line and rows (vel_mid, tokens)."""
    cols, rows, t = None, [], None
    for ln in open(path):
        if ln.startswith("#TIME:"):
            t = ln.split()[1]
            continue
        if t != time_label:
            continue
        if ln.startswith("#vel_mid"):
            cols = ln[1:].split()[1:]
            continue
        if ln.startswith("#") or not ln.strip():
            continue
        tok = ln.split()
        rows.append((float(tok[0]), tok[1:]))
    return cols, rows


def det_blue_ion_map_nearest(rows, v):
    """prereg 3.3-3.4: nearest public vel_mid, ties to the lower velocity; outside the public range -> None."""
    vs = [r[0] for r in rows]
    if not rows or v < min(vs) or v > max(vs):
        return None
    return min(rows, key=lambda r: (abs(r[0] - v), r[0]))


def det_blue_ion_map_fraction(row, Z, q):
    """registered fraction: NLTE q / element total of the same row (no stage column, no II+III renormalisation)."""
    num, den = float(row[f"Z{Z}_nlte_q{q}_cm3"]), float(row[f"Z{Z}_element_total_cm3"])
    return num, den


def det_blue_ion_map_ion_rows(ion, shell, pub, fraction=det_blue_ion_map_fraction, column_of=None, ns=NS):
    """300 fraction rows (Z 26-28 x q 1-2 x shell 0-49; ns smaller only in the direct controls)."""
    column_of = column_of or (lambda Z, q: f"{SYM[Z]}{q}")
    x = ion.set_index("shell_id")
    sh = shell.set_index("shell_id")
    out = []
    for Z in (26, 27, 28):
        cols, rows = pub[Z]
        for q in (1, 2):
            col = column_of(Z, q)
            for s in range(ns):
                rec = {"kind": "fraction", "item": f"{SYM[Z].capitalize()} {ROMAN[q]}", "Z": Z, "q": q, "shell": s,
                       "public_column": col, "why": ""}
                why = []
                v = (float(sh.loc[s, "v_inner_cm_s"]) + float(sh.loc[s, "v_outer_cm_s"])) / 2.0 / 1e5
                rec["v_lumina_kms"] = v
                if abs(v - (4264.0 + 728.0 * s)) > 1e-6:
                    why.append("SHELL_GRID")
                prow = det_blue_ion_map_nearest(rows, v)
                f_pub = None
                if prow is None:
                    why.append("PUBLIC_RANGE")
                else:
                    rec["v_public_kms"], rec["dv_kms"] = prow[0], prow[0] - v
                    j = cols.index(col) if (cols and col in cols) else None
                    if j is None:
                        why.append("PUBLIC_COLUMN")
                    else:
                        rec["public_token"] = prow[1][j]
                        f_pub = POP.printed(prow[1][j])
                num, den = fraction(x.loc[s], Z, q)
                rec["n_num"], rec["n_den"] = num, den
                fO = I.x(num).div(I.x(den)) if (den > 0.0 and num >= 0.0) else None
                if fO is None:
                    why.append("FRACTION_UNDEFINED")
                d = None
                if fO is not None and f_pub is not None and f_pub.lo > 0.0 and fO.lo > 0.0:
                    d = NRB.det_nrb_judge_log10i(fO.div(f_pub))
                elif fO is not None and f_pub is not None:
                    why.append("LOG_UNDEFINED")
                for k_, iv_ in (("f_O", fO), ("f_public", f_pub), ("d", d)):
                    REP.det_siiii_repair_cols(rec, k_, iv_)
                rec["code"] = code_of(d)
                rec["why"] = ";".join(why)
                out.append(rec)
    return out


def det_blue_ion_map_band_rows(pts_O, pts_pub, ratio=None):
    """9 absolute band rows: d = log10(L_O / L_public) on the stored intervals (prereg 4)."""
    bO, bP = REP.det_siiii_repair_bands(pts_O), REP.det_siiii_repair_bands(pts_pub)
    out = []
    for b in BANDS:
        lo, hi = b
        rec = {"kind": "band", "item": f"{lo}-{hi}" + (" (OPT)" if b == (4000, 6000) else ""), "why": ""}
        LO, LP = bO.get(b), bP.get(b)
        r = (ratio or (lambda LO_, LP_, bO_, bP_: LO_.div(LP_)))(LO, LP, bO, bP) if (LO and LP) else None
        d = NRB.det_nrb_judge_log10i(r) if r is not None else None
        rec.update({"L_O": None if LO is None else LO.mid, "L_public": None if LP is None else LP.mid,
                    "ratio": None if r is None else r.mid})
        REP.det_siiii_repair_cols(rec, "d", d)
        rec["code"] = code_of(d)
        rec["why"] = "" if d is not None else "BAND_UNDEFINED"
        out.append(rec)
    return out


# ------------------------------------------------------------------ G-TABLE validators (on the assembled rows)
def det_blue_ion_map_validate(rows, ion, shell, pub):
    why = []
    x = ion.set_index("shell_id")
    keys = [(r["Z"], r["q"], r["shell"]) for r in rows if r["kind"] == "fraction"]
    if len(keys) != 300 or len(set(keys)) != 300:
        why.append(f"G-TABLE:FRACTION_ROWS:{len(keys)}:{len(set(keys))}")
    if sum(1 for r in rows if r["kind"] == "band") != 9:
        why.append("G-TABLE:BAND_ROWS")
    for r in rows:
        if r["kind"] != "fraction":
            continue
        Z, q, s = r["Z"], r["q"], r["shell"]
        if r["public_column"] != f"{SYM[Z]}{q}":
            why.append(f"G-TABLE:COLUMN:{Z}:{q}:{s}:{r['public_column']}")
        if r["n_num"] != float(x.loc[s, f"Z{Z}_nlte_q{q}_cm3"]) or r["n_den"] != float(x.loc[s, f"Z{Z}_element_total_cm3"]):
            why.append(f"G-TABLE:FRACTION_DEFINITION:{Z}:{q}:{s}")
        v = (float(shell.set_index("shell_id").loc[s, "v_inner_cm_s"]) +
             float(shell.set_index("shell_id").loc[s, "v_outer_cm_s"])) / 2.0 / 1e5
        prow = det_blue_ion_map_nearest(pub[Z][1], v)
        if (prow is None) != ("v_public_kms" not in r) or (prow is not None and r.get("v_public_kms") != prow[0]):
            why.append(f"G-TABLE:SHELL_MAPPING:{Z}:{q}:{s}")
    return why


def det_blue_ion_map_validate_bands(rows):
    why = []
    for r in rows:
        if r["kind"] != "band" or r.get("L_O") is None or r.get("L_public") is None:
            continue
        if r.get("ratio") is None or abs(r["ratio"] / (r["L_O"] / r["L_public"]) - 1.0) > 1e-12:
            why.append(f"G-TABLE:BAND_NOT_ABSOLUTE:{r['item']}")
    return why


# ------------------------------------------------------------------ direct negative controls (prereg 8)
def det_blue_ion_map_selftest():
    recs = []

    def case(name, got, want):
        recs.append({"case": name, "got": str(got), "want": str(want), "pass": got == want})
    man_ok = {"population_generation": 13, "radiation_generation": 12, "iteration": 10, "transaction_status": "COMMITTED"}
    capman = json.loads(CAPMAN.read_text())
    w_ok = det_blue_ion_map_inputs(man_ok, capman, SPEC)
    w_bad = det_blue_ion_map_inputs(dict(man_ok, population_generation=14, radiation_generation=13, iteration=11),
                                    capman, SPEC)
    case("1 generation 14 input refused by G-INPUT", (w_ok, any("DUMP_GENERATION" in w for w in w_bad)), ([], True))
    # small fixture: 4 shells, distinct NLTE / stage / total / II+III values; public block with 3 rows
    n = 4
    ion = pd.DataFrame({"shell_id": range(n)})
    for Z in (26, 27, 28):
        ion[f"Z{Z}_element_total_cm3"] = [10.0] * n
        ion[f"Z{Z}_nlte_q1_cm3"] = [2.0] * n
        ion[f"Z{Z}_nlte_q2_cm3"] = [3.0] * n
        ion[f"Z{Z}_stage_q1_cm3"] = [4.0] * n
        ion[f"Z{Z}_stage_q2_cm3"] = [1.0] * n
    shell = pd.DataFrame({"shell_id": range(n), "v_inner_cm_s": [(3900.0 + 728.0 * s) * 1e5 for s in range(n)],
                          "v_outer_cm_s": [(4628.0 + 728.0 * s) * 1e5 for s in range(n)]})
    pub = {Z: ([f"{SYM[Z]}{k}" for k in range(6)],
               [(4264.0 + 728.0 * s, ["1.0E-03", "2.0000E-01", "3.0000E-01", "4.0E-01", "1.0E-01", "0.0E+00"])
                for s in range(n)]) for Z in (26, 27, 28)}
    ok = det_blue_ion_map_ion_rows(ion, shell, pub, ns=n)
    # mapping: shell 3 bound to the neighbour's velocity; Fe column set to Co
    pub_shift = {Z: (c, [(r[0] + (728.0 if k == 3 else 0.0), r[1]) for k, r in enumerate(rows)]) for Z, (c, rows) in pub.items()}
    bad_map = det_blue_ion_map_ion_rows(ion, shell, pub_shift, ns=n)
    bad_col = det_blue_ion_map_ion_rows(ion, shell, pub, column_of=lambda Z, q: f"co{q}" if Z == 26 else f"{SYM[Z]}{q}",
                                        ns=n)

    def vfix(rows):
        return [w for w in det_blue_ion_map_validate(rows + [{"kind": "band"}] * 9, ion, shell, pub)
                if not w.startswith("G-TABLE:FRACTION_ROWS")]
    case("2 shell 3 bound to the neighbour velocity / Fe column set to Co refused by the mapping check",
         (vfix(ok), any("SHELL_MAPPING" in w for w in vfix(bad_map)), any("COLUMN" in w for w in vfix(bad_col))),
         ([], True, True))
    # wrong numerator (stage) and wrong denominator (II + III)
    bad_num = det_blue_ion_map_ion_rows(ion, shell, pub, fraction=lambda r, Z, q: (float(r[f"Z{Z}_stage_q{q}_cm3"]),
                                                                                   float(r[f"Z{Z}_element_total_cm3"])),
                                        ns=n)
    bad_den = det_blue_ion_map_ion_rows(ion, shell, pub, fraction=lambda r, Z, q: (
        float(r[f"Z{Z}_nlte_q{q}_cm3"]), float(r[f"Z{Z}_nlte_q1_cm3"]) + float(r[f"Z{Z}_nlte_q2_cm3"])), ns=n)
    case("3 stage numerator / II+III denominator refused by the registered fraction check",
         (any("FRACTION_DEFINITION" in w for w in vfix(bad_num)), any("FRACTION_DEFINITION" in w for w in vfix(bad_den))),
         (True, True))
    # colour ratio entered as absolute dex
    lam = [2000.0 + 50.0 * k for k in range(181)]
    ptsO = [(x_, 1.0e38 * (1.0 + x_ / 4000.0)) for x_ in lam]
    ptsP = [(x_, 4.0e38 * (2.0 - x_ / 12000.0)) for x_ in lam]
    good = det_blue_ion_map_band_rows(ptsO, ptsP)
    colour = det_blue_ion_map_band_rows(ptsO, ptsP, ratio=lambda LO, LP, bO, bP: LO.div(bO[(4000, 6000)]).div(
        LP.div(bP[(4000, 6000)])))
    case("4 colour ratio entered as absolute dex refused by the band check",
         (det_blue_ion_map_validate_bands(good), bool(det_blue_ion_map_validate_bands(colour))), ([], True))
    return {"records": recs, "failures": sum(1 for r in recs if not r["pass"])}


# ------------------------------------------------------------------ analyze
def fmt(v, f):
    return "—" if v is None or (isinstance(v, float) and v != v) else f.format(v)


def det_blue_ion_map_write(out, rows):
    pd.DataFrame(rows).to_csv(out / "cmfgen_comparison.csv", index=False)
    L = ["# DET-BLUE-ION-MAP — O(세대 13) 대 공개 CMFGEN(19.48 d)", "",
         "d = log10(O/CMFGEN). 코드 0 공개보다 낮음 · 1 높음 · 4 부호 미분리 · 5 미정의. 공개 이온 열 si0…/fe0… 를 I… 로 "
         "읽는 명명 가정에 조건부.", "", "## 1. 절대 대역 광도 (기존 O 의 재표시)", "",
         "| 대역 (Å) | CMFGEN (erg/s) | O (erg/s) | O/CMFGEN | d (dex) | 코드 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] == "band":
            L.append(f"| {r['item']} | {fmt(r['L_public'], '{:.4e}')} | {fmt(r['L_O'], '{:.4e}')} | "
                     f"{fmt(r['ratio'], '{:.4f}')} | {fmt(r.get('d_mid'), '{:+.3f}')} | {LABEL[r['code']]} |")
    L += ["", "## 2. 이온별 셸 수 (분모 50)", "", "| 이온 | 공개보다 낮음 | 높음 | 부호 미분리 | 미정의 | d 범위 (dex) |",
          "|---|---|---|---|---|---|"]
    fr = [r for r in rows if r["kind"] == "fraction"]
    for Z in (26, 27, 28):
        for q in (1, 2):
            sub = [r for r in fr if r["Z"] == Z and r["q"] == q]
            ds = [r["d_mid"] for r in sub if r.get("d_mid") is not None]
            c = {k: sum(1 for r in sub if r["code"] == k) for k in (0, 1, 4, 5)}
            L.append(f"| {sub[0]['item']} | {c[0]} | {c[1]} | {c[4]} | {c[5]} | "
                     f"{fmt(min(ds) if ds else None, '{:+.2f}')} … {fmt(max(ds) if ds else None, '{:+.2f}')} |")
    L += ["", "## 3. II/III 쌍 패턴 (같은 원소·셸)", "",
          "II 높음·III 낮음 = 「두 단계 사이에서 낮은 전리 쪽으로 치우친 패턴」, 반대 = 「높은 전리 쪽 패턴」. 그 밖은 그대로.", "",
          "| 원소 | 낮은 전리 쪽 | 높은 전리 쪽 | 같은 부호·미분리·미정의 |", "|---|---|---|---|"]
    for Z in (26, 27, 28):
        lo = hi = other = 0
        for s in range(NS):
            c2 = next(r["code"] for r in fr if r["Z"] == Z and r["q"] == 1 and r["shell"] == s)
            c3 = next(r["code"] for r in fr if r["Z"] == Z and r["q"] == 2 and r["shell"] == s)
            if (c2, c3) == (1, 0):
                lo += 1
            elif (c2, c3) == (0, 1):
                hi += 1
            else:
                other += 1
        L.append(f"| {SYM[Z].capitalize()} | {lo} | {hi} | {other} |")
    L += ["", "## 4. 셸별 분율 표", "", "| 이온 | 셸 | v_O (km/s) | v_pub (km/s) | CMFGEN | O | d (dex) | 코드 | 사유 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in fr:
        L.append(f"| {r['item']} | {r['shell']} | {r['v_lumina_kms']:.0f} | {fmt(r.get('v_public_kms'), '{:.1f}')} | "
                 f"{fmt(r.get('f_public_mid'), '{:.3e}')} | {fmt(r.get('f_O_mid'), '{:.3e}')} | {fmt(r.get('d_mid'), '{:+.3f}')} | "
                 f"{LABEL[r['code']]} | {r['why']} |")
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
    st = det_blue_ion_map_selftest()
    ev = {"schema": "DET_BLUE_ION_MAP_EVIDENCE_V1", "selftest": st, "readout_sha256": sha(Path(__file__).resolve()),
          "job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if st["failures"]:
        ev["validity"] = "X"
        (out / "evidence.json").write_text(json.dumps(ev, indent=1))
        print(json.dumps({"validity": "X", "selftest_failures": st["failures"]}))
        return 2
    man, capman = json.loads(MAN.read_text()), json.loads(CAPMAN.read_text())
    why = det_blue_ion_map_inputs(man, capman, SPEC)
    ion = pd.read_csv(ION, float_precision="round_trip")
    shell = pd.read_csv(SHELL, float_precision="round_trip")
    if (ion.population_generation != G_POP).any():
        why.append("G-INPUT:ION_ROW_GENERATION")
    pub = {Z: det_blue_ion_map_public(p) for Z, p in PUBLIC_ION.items()}
    rows = det_blue_ion_map_ion_rows(ion, shell, pub)
    rows += det_blue_ion_map_band_rows(REP.det_siiii_repair_spec(SPEC), REP.det_siiii_repair_public(PUBLIC_SPEC))
    why += det_blue_ion_map_validate(rows, ion, shell, pub) + det_blue_ion_map_validate_bands(rows)
    if len(rows) != 309:
        why.append(f"G-TABLE:ROWS:{len(rows)}")
    det_blue_ion_map_write(out, rows)
    ev.update({"validity": "X" if why else "VALID", "disposition": "INADMISSIBLE" if why else "PASS_LIMITED",
               "reasons": why, "n_rows": len(rows),
               "inputs": {str(p): sha(p) for p in [MAN, ION, SHELL, CAPMAN, SPEC, PUBLIC_SPEC, *PUBLIC_ION.values()]},
               "reused": {m.__name__: sha(Path(m.__file__)) for m in (NRB, REP, POP)},
               "claim_ceiling": "stored O (12 iterations, generation 13); no production promotion, physical correctness, "
                                "causal attribution, convergence or seed independence"})
    (out / "evidence.json").write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps({"validity": ev["validity"], "rows": len(rows), "reasons": why[:10]}))
    return 0 if not why else 2


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("analyze")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_analyze(a)


if __name__ == "__main__":
    sys.exit(main())
