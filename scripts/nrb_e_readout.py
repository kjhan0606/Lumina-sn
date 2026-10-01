#!/usr/bin/env python3
"""DET-NRB-E readout (prereg docs/PREREG_DET_NRB_E_2026-10-02.md §2, §4, §5).

Subcommands
  select   --base E --out FILE           materialise the fixed line set L from E call 7 (before submission)
  gid      --base E --test R --out DIR   identity gate (pre-submit or post-run)
  readout  --base E --test R --lines FILE --selftest JSON --slurm-states JSON --out DIR
  selftest [--base E] --out DIR          §4.3 NC battery

E = baseline (NT-ETA run1), R = NRB run.  All changes are R - E.  Branches R0..R5 and a
separate validity field (VALID / X).  No clamps, floors or substitutions.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w5_cont_readout import log_planck, logsumexp                     # frozen 7008f709
from nt_eta_readout import (sha, fnum, finite_pos, read_exports, footer, bins_reason,   # frozen
                            approved_reasons, content_reasons, footer_reasons, te_reasons,
                            load_ion_iter, cell_outcome, aggregate, band_integral,
                            PATH_KEYS, W5_BIN, W5_BUILD, W5_APPROVED, SRC_ROOT)

C_CGS = 2.99792458e10
SHELLS = (10, 15, 19, 24, 30)
ELEMS = (16, 14, 20)
AUTHS = ("stage", "levels")
GEN_K = {3: 6, 4: 7, 5: 8}
THR = 0.30
MUST_KEEP = {"LUMINA_PURE_CMFGEN_ITER": "6", "LUMINA_DET_TRANSACTIONAL": "1",
             "LUMINA_CMF_FINE_MGPU_DEVICES": "2", "LUMINA_CMF_FINE_CONT_NLTE_SOURCE": "1",
             "LUMINA_ION_NT": "1", "LUMINA_POP_DJDN": "1", "LUMINA_NLTE_SKIP_Z": "",
             "LUMINA_XSHELL_RESPONSE": "0", "LUMINA_NT_FULL": "0"}
DELTA = {"LUMINA_BB_NRB": (None, "1"), "LUMINA_A210_SPRODUCER_CAPTURE": ("0", "1")}
J_SHELLS = {"D": tuple(range(15, 25)), "B": tuple(range(19, 25))}
J_BANDS = {"D": (912.0, 1500.0), "B": (531.0, 760.0)}
# §4.2 (pre-audit K-1): explicit failure markers; [DPROXY][SL-POP-COMPLETE] status=BLOCKED is a normal diagnostic
FORBIDDEN = ("[cmf_fine][BLOCKED]", "[R6][BLOCKED]", "[R6][FATAL]", "[R7][FATAL]", "[RADIATION-FIELD][FATAL]",
             "[cmfgen_assemble][BLOCKED]", "[A2-INIT][FATAL]", "[CMFGEN][FATAL]")


# ------------------------------------------------------------------ categories / branch
def category(x):
    if not (isinstance(x, float) and math.isfinite(x)):
        return "MISSING"
    if x <= -THR:
        return "down"
    if x >= THR:
        return "up"
    return "flat"


def shell_aggregate(cats):
    """①: same category in >= 3 of 5 shells, else mixed; any missing -> MISSING."""
    if len(cats) != 5 or "MISSING" in cats:
        return "MISSING"
    for c in ("down", "up", "flat"):
        if cats.count(c) >= 3:
            return c
    return "mixed"


def sulfur_mode(outcomes):
    if len(outcomes) != 10 or any(o not in "abcd" for o in outcomes):
        return "MISSING"
    a, _, _ = aggregate(outcomes)
    n = {o: outcomes.count(o) for o in "abcd"}
    top = max(n.values())
    return "mixed" if sum(1 for v in n.values() if v == top) > 1 else a


def branch(agg1, agg3, c4a, c4b):
    """Two-stage table (§5.6).  Returns (branch, R5 reasons)."""
    why = []
    if "MISSING" in (agg1, agg3, c4a, c4b):
        why.append("MISSING")
    if agg1 == "mixed":
        why.append("MIXED_1")
    if agg3 == "mixed":
        why.append("MIXED_3")
    if {c4a, c4b} == {"down", "up"}:
        why.append("FUV_OPPOSITE")
    if why:
        return "R5", why
    if agg1 in ("flat", "up"):
        return "R0", []
    if agg3 == "a":
        return ("R1" if c4a == "down" else "R2"), []
    if agg3 in ("c", "d"):
        return "R3", []
    return "R4", []


# ------------------------------------------------------------------ ① lines
def read_ledger(run, keys=None):
    """{(call, shell, line_id): row}; duplicates counted."""
    out, dup = {}, 0
    with open(Path(run) / "work/w5_lines.csv") as fh:
        for r in csv.DictReader(fh):
            k = (int(r["call"]), int(r["shell"]), int(r["line_id"]))
            if keys is not None and (k[1], k[2]) not in keys:
                continue
            if k in out:
                dup += 1
            out[k] = r
    return out, dup


def select_lines(ledger_E):
    """§5.1: E call 7, tau_eff >= 1e3, 912 <= lambda < 1500 A, registered shells (no status filter)."""
    sel = []
    for (c, s, l), r in ledger_E.items():
        if c != 7 or s not in SHELLS:
            continue
        tau, nu = fnum(r["tau_eff"]), fnum(r["nu_hz"])
        if not (math.isfinite(tau) and math.isfinite(nu) and nu > 0.0):
            continue
        lam = C_CGS / nu * 1e8
        if tau >= 1.0e3 and 912.0 <= lam < 1500.0:
            sel.append((s, l))
    return sorted(sel)


def source(r):
    """S = L_l / (1 - beta); defined iff finite, 1-beta != 0 and S > 0 (no beta range gate)."""
    if r is None or r.get("status") != "OK":
        return None
    L, b = fnum(r["L_l"]), fnum(r["beta"])
    if not (math.isfinite(L) and math.isfinite(b)) or b == 1.0:
        return None
    S = L / (1.0 - b)
    return S if finite_pos(S) else None


def ledger_row_ok(r, c):
    """Sobolev call c row: operator and radiation / material generation = c."""
    return r is not None and r.get("operator") == "CMFGEN_NONOVERLAP_SOBOLEV" \
        and r.get("rad_generation") == str(c) and r.get("material_generation") == str(c)


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def delta1(ledE, ledR, lines):
    """Δ₁(s,c) = median over L_s of [log S^R_c - log S^E_7]; any undefined line -> None + reason."""
    res, why, traj = {}, [], []
    if len(set(lines)) != len(lines):
        why.append("LINES_DUPLICATE_KEY")
    for s in SHELLS:
        Ls = sorted({l for (ss, l) in lines if ss == s})
        if not Ls:
            res[s] = {c: None for c in range(3, 8)}
            why.append(f"L_EMPTY:sh{s}")
            continue
        res[s] = {}
        for c in range(3, 8):
            d, bad = [], False
            for l in Ls:
                rE, rR = ledE.get((7, s, l)), ledR.get((c, s, l))
                sE = source(rE) if ledger_row_ok(rE, 7) else None
                sR = source(rR) if ledger_row_ok(rR, c) else None
                row = {"shell": s, "line_id": l, "call": c, "S_E7": sE, "S_R": sR}
                for side, rr in (("E7", rE), ("R", rR)):
                    for f in ("L_l", "beta", "J_c", "Jbar", "tau_eff", "srce_chk", "status",
                              "operator", "rad_generation", "material_generation"):
                        row[f"{f}_{side}"] = (rr or {}).get(f)
                traj.append(row)
                if sE is None or sR is None:
                    why.append(f"S_UNDEFINED:sh{s}/l{l}/c{c}")
                    bad = True
                    continue
                d.append(math.log10(sR) - math.log10(sE))
            res[s][c] = None if bad or not d else median(d)
    return res, why, traj


# ------------------------------------------------------------------ ② / ④
JB_RTOL = 1e-12   # float re-evaluation of contiguous bin edges


def jb_ratio(rows, T, lo_A, hi_A):
    """Full-support, non-overlapping bins required; returns dict with log10 ratio and pieces or None."""
    n_hi, n_lo = C_CGS / (lo_A * 1e-8), C_CGS / (hi_A * 1e-8)
    if not (isinstance(T, float) and math.isfinite(T) and T > 0.0):
        return None
    use = sorted((lo, hi, J) for lo, hi, J in rows if min(hi, n_hi) - max(lo, n_lo) > 0.0)
    if not use or use[0][0] > n_lo * (1 + JB_RTOL) or use[-1][1] < n_hi * (1 - JB_RTOL):
        return None
    for (l0, h0, _), (l1, h1, _) in zip(use[:-1], use[1:]):
        if abs(l1 - h0) > JB_RTOL * h0:
            return None          # gap or overlap / duplicate bin
    num, lden = 0.0, []
    for lo, hi, J in use:
        if not math.isfinite(J):
            return None
        ov = min(hi, n_hi) - max(lo, n_lo)
        num += J * ov
        lden.append(log_planck(math.sqrt(lo * hi), T) + math.log(ov))
    if num <= 0.0:
        return None
    ld = logsumexp(lden)
    return {"log10_ratio": (math.log(num) - ld) / math.log(10.0), "numerator": num,
            "log_denominator": ld, "bins": len(use), "support_Hz": [use[0][0], use[-1][1]]}


def jb_table(run):
    P = Path(run) / "work/physics_comparison/physics_DET_iter0005"
    m = json.load(open(f"{P}.manifest.json"))
    T = {int(r["shell_id"]): fnum(r["T_e_K"]) for r in csv.DictReader(open(f"{P}.shell.csv"))}
    need = set(J_SHELLS["D"]) | set(J_SHELLS["B"])
    rows = {}
    for r in csv.DictReader(open(f"{P}.spectral.csv")):
        s = int(r["shell_id"])
        if s in need:
            rows.setdefault(s, []).append((fnum(r["nu_lo_Hz"]), fnum(r["nu_hi_Hz"]), fnum(r["J_nu"])))
    out = {"radiation_generation": m.get("radiation_generation")}
    for band, shells in J_SHELLS.items():
        for s in shells:
            out[f"{band}/sh{s}"] = jb_ratio(rows.get(s, []), T.get(s), *J_BANDS[band])
    return out


def fuv(run):
    pts = sorted((fnum(r["wavelength_angstrom"]), fnum(r["flux"]))
                 for r in csv.DictReader(open(Path(run) / "work/lumina_spectrum.csv")))
    if any(not (math.isfinite(l) and math.isfinite(f)) for l, f in pts) or \
            len({l for l, _ in pts}) != len(pts):
        return {"status": "NONFINITE_OR_DUPLICATE"}
    out = {"status": "OK"}
    den, sd = band_integral(pts, 4000.0, 6000.0)
    for nm, lo, hi in (("a", 1200.0, 1500.0), ("b", 1500.0, 2000.0)):
        num, sn = band_integral(pts, lo, hi)
        if sn != "OK" or sd != "OK" or not (num > 0.0 and den > 0.0):
            return {"status": f"UNDEFINED:{nm}"}
        out[f"F{nm}"] = num / den
        out[f"L_{nm}"] = num
    out["L_4000_6000"] = den
    return out


# ------------------------------------------------------------------ ③
def y_series(iters, Z, A, s):
    tag = "stage" if A == "stage" else "nlte"
    ys = {}
    for k in (3, 4, 5):
        r = iters[k].get(s)
        if r is None or r.get("population_generation") != str(GEN_K[k]):
            return None
        q1, q2 = fnum(r.get(f"Z{Z}_{tag}_q1_cm3")), fnum(r.get(f"Z{Z}_{tag}_q2_cm3"))
        if not (finite_pos(q1) and finite_pos(q2)):
            return None
        y = math.log10(q2) - math.log10(q1)
        if not math.isfinite(y):
            return None
        ys[k] = y
    return ys


def cell(yE, yR):
    dE, dR = abs(yE[5] - yE[3]) / 2.0, abs(yR[5] - yR[3]) / 2.0
    D = max(dE, dR)
    d3 = (yR[4] + yR[5]) / 2.0 - (yE[4] + yE[5]) / 2.0
    return {"ybar_E": (yE[4] + yE[5]) / 2.0, "ybar_R": (yR[4] + yR[5]) / 2.0, "d_E": dE, "d_R": dR,
            "D": D, "T": max(THR, 3.0 * D), "delta3": d3, "outcome": cell_outcome(d3, D)}


# ------------------------------------------------------------------ gates
def exports_delta2(dE, dR, eroot, rroot):
    why = []
    for k in sorted(set(dE) | set(dR) | set(DELTA) | set(PATH_KEYS)):
        vE, vR = dE.get(k), dR.get(k)
        if k in PATH_KEYS:
            if vE is None or vR is None or not vE.startswith(eroot) or vR != rroot + vE[len(eroot):]:
                why.append(f"EXPORTS_PATH:{k}")
            continue
        if k in DELTA:
            if (vE, vR) != DELTA[k]:
                why.append(f"EXPORTS_DELTA:{k}:{vE}->{vR}")
            continue
        if vE != vR:
            why.append(f"EXPORTS_EXTRA_DIFF:{k}")
    for k, v in MUST_KEEP.items():
        if dE.get(k) != v or dR.get(k) != v:
            why.append(f"EXPORTS_MUST_KEEP:{k}")
    return why


def gid(eroot, rroot, post_run):
    E, R = Path(eroot), Path(rroot)
    why, ev = [], {}
    bins = {"build_record": (Path(W5_BUILD) / "binary.sha256").read_text().split()[0],
            "build_file": sha(Path(W5_BUILD) / "lumina_cuda")}
    for nm, root in (("E", E), ("R", R)):
        bins[f"{nm}_file"] = sha(root / "input/lumina_cuda")
        bins[f"{nm}_record"] = (root / "input/binary.sha256").read_text().strip()
        bins[f"{nm}_rbi"] = json.load(open(root / "input/run_build_identity.json"))["binary"]["sha256"]
        st = dict(ln.rstrip("\n").split("=", 1) for ln in open(root / "STAGE_IDENTITY.txt") if "=" in ln)
        bins[f"{nm}_stage"] = st.get("run_binary_sha256", "").strip()
        if post_run:
            bins[f"{nm}_footer"] = footer(root).get("binary_sha256", "")
    ev["binary"] = bins
    why += bins_reason(bins)
    srcs = {nm: (root / "input/source_sha_at_build.txt").read_text() for nm, root in (("E", E), ("R", R))}
    if srcs["E"] != srcs["R"] or srcs["R"] != (Path(W5_BUILD) / "source_sha_at_build.txt").read_text():
        why.append("SOURCE_RECORD_MISMATCH")
    rec = {}
    for ln in srcs["R"].splitlines():
        h, p = ln.split(None, 1)
        rec[p.lstrip("*")] = h
    bad = [p for p, h in rec.items() if sha(Path(SRC_ROOT) / p) != h]
    if bad:
        why.append(f"SOURCE_TREE_MISMATCH:{len(bad)}")
    why += approved_reasons(rec, Path(W5_APPROVED).read_text())
    for f in ("deck.sha256", "topion.sha256", "sigma.sha256", "fixed_te_profile.txt", "run_build_identity.json"):
        if sha(E / "input" / f) != sha(R / "input" / f):
            why.append(f"INPUT_MISMATCH:{f}")
    why += content_reasons(E, "E") + content_reasons(R, "R")
    dE, dupE = read_exports(E / "input/resolved_lumina.exports")
    dR, dupR = read_exports(R / "input/resolved_lumina.exports")
    if dupE or dupR:
        why.append("EXPORTS_DUPLICATE_KEY")
    why += exports_delta2(dE, dR, str(E), str(R))
    ev["runner_sha_R"] = sha(R / "job.slurm")
    for nm, root in (("E", E), ("R", R)):
        why += linkage_reasons(root, nm, post_run)
    return why, ev


def linkage_reasons(root, nm, post_run):
    """exports / runner hashes consumed = stage record (= footer after the run)."""
    why = []
    st = dict(ln.rstrip("\n").split("=", 1) for ln in open(root / "STAGE_IDENTITY.txt") if "=" in ln)
    ex = sha(root / "input/resolved_lumina.exports")
    if st.get("effective_exports_sha256", "").strip() != ex:
        why.append(f"STAGE_EXPORTS_HASH:{nm}")
    if st.get("runner_sha256", "").strip() != sha(root / "job.slurm"):
        why.append(f"STAGE_RUNNER_HASH:{nm}")
    if post_run:
        ft = footer(root)
        if ft.get("effective_exports_sha256") != ex:
            why.append(f"FOOTER_EXPORTS_HASH:{nm}")
        if ft.get("partition") not in ("h100", "a100") or ft.get("allocated_gpus") != "2" \
                or ft.get("allocated_cpus") != "16":
            why.append(f"FOOTER_RESOURCES:{nm}")
    return why


def nrb_witness(lines_of_log, armed_expected):
    """(iter,operator)-complete [BB-NRB] tags and strictly parsed SPRODUCER-CAPTURE lines."""
    why, tags, cap = [], {}, 0
    for ln in lines_of_log:
        if ln.startswith("[BB-NRB][FATAL]"):
            why.append("BB_NRB_FATAL")
        elif ln.startswith("[BB-NRB] "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            key = (d.get("iter"), d.get("operator"))
            if key in tags:
                why.append(f"BB_NRB_DUPLICATE:{key}")
            tags[key] = d
        elif ln.startswith("[cmf_fine][SPRODUCER-CAPTURE]"):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            if d.get("status") == "CAPTURED" and d.get("n_shells") == "50" \
                    and d.get("operator") == "CMFGEN_NONOVERLAP_SOBOLEV":
                cap += 1
            else:
                why.append("SPRODUCER_CAPTURE_MALFORMED")
    if not armed_expected:
        if any(t.get("armed") == "1" or t.get("captured") == "1" for t in tags.values()) or cap:
            why.append("BASELINE_HAS_NRB")
        return why
    init = tags.get(("0", "1"))
    if not init or init.get("armed") != "0" or init.get("captured") != "0":
        why.append("BB_NRB_INIT_WITNESS")
    for k in range(6):
        t = tags.get((str(k), "2"))
        if not t or t.get("armed") != "1" or t.get("captured") != "1" or t.get("n_shells") != "50" \
                or t.get("skip_z") != "none":
            why.append(f"BB_NRB_ITER_WITNESS:k{k}")
    extra = [k for k in tags if k != ("0", "1") and k not in {(str(i), "2") for i in range(6)}]
    if extra:
        why.append(f"BB_NRB_UNEXPECTED:{extra}")
    if cap != 6:
        why.append(f"SPRODUCER_CAPTURED_COUNT:{cap}")
    return why


def grun(run, slurm_state, armed_expected):
    W = Path(run) / "work"
    why = footer_reasons(footer(run), slurm_state)
    prof = {}
    for ln in open(Path(run) / "input/fixed_te_profile.txt"):
        a = ln.split()
        if len(a) == 2:
            prof[int(a[0])] = float(a[1])
    why += te_reasons(run, prof)[0]
    for k in range(6):
        mp = W / f"physics_comparison/physics_DET_iter{k:04d}.manifest.json"
        if not mp.exists():
            why.append(f"MANIFEST_MISSING:k{k}")
            continue
        m = json.load(open(mp))
        if m.get("transaction_status") != "COMMITTED" or m.get("radiation_generation") != k + 2 \
                or m.get("population_generation") != k + 3 or m.get("te_lane") != "FIXED_T":
            why.append(f"MANIFEST_STATE:k{k}")
    why += publication_reasons(W)
    log = list(open(Path(run) / "stderr.log", errors="replace")) if (Path(run) / "stderr.log").exists() else []
    ok = sum(1 for ln in log if ln.startswith("[cmf_fine][EXACT-MULTIGPU-EPOCH]") and " status=OK " in ln + " ")
    if ok != 7:
        why.append(f"EXACT_SOLVE_OK:{ok}")
    hit = [t for t in FORBIDDEN if any(ln.startswith(t) for ln in log)]
    if hit:
        why.append(f"FORBIDDEN_TAG:{hit}")
    why += nrb_witness(log, armed_expected)
    for p in ["w5_lines.csv", "lumina_spectrum.csv", "physics_comparison/physics_DET_iter0005.spectral.csv",
              "physics_comparison/physics_DET_iter0005.shell.csv"] + \
             [f"physics_comparison/physics_DET_iter{k:04d}.ion.csv" for k in (3, 4, 5)]:
        if not (W / p).exists():
            why.append(f"FILE_MISSING:{p}")
    return why


def publication_reasons(W):
    fp = W / "cmfgen_final_publication.json"
    if not fp.exists():
        return ["FINAL_PUBLICATION_MISSING"]
    f = json.load(open(fp))
    why = []
    if not (f.get("complete") is True and f.get("population_generation") == 8
            and f.get("required_population_generation") == 8):
        why.append("FINAL_PUBLICATION_STATE")
    want = {"plasma": "PUBLIC_NE_TE", "saha": "SAHA_STAGE", "nlte": "NLTE_LEVEL"}
    files = f.get("files", {})
    for k, auth in want.items():
        e = files.get(k, {})
        if e.get("authority") != auth or not (W / str(e.get("path"))).exists():
            why.append(f"FINAL_PUBLICATION_FILE:{k}")
    return why


# ------------------------------------------------------------------ commands
def cmd_select(a):
    out = Path(a.out)
    if out.exists():
        print("OUT_EXISTS")
        return 2
    led, dup = read_ledger(a.base)
    if dup:
        print(f"E_LEDGER_DUPLICATE:{dup}")
        return 1
    sel = select_lines(led)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"# E ledger sha256={sha(Path(a.base) / 'work/w5_lines.csv')} rule=call7,tau_eff>=1e3,912<=lambda<1500,shells{SHELLS}\n")
        fh.write("shell,line_id,tau_eff_E7,lambda_A,S_E7\n")
        for s, l in sel:
            r = led[(7, s, l)]
            fh.write(f"{s},{l},{r['tau_eff']},{C_CGS / fnum(r['nu_hz']) * 1e8!r},{source(r)!r}\n")
    print(json.dumps({"n_lines": len(sel), "per_shell": {s: sum(1 for x in sel if x[0] == s) for s in SHELLS}}))
    return 0


def read_lines_file(p):
    rows = [ln for ln in open(p) if not ln.startswith("#")]
    return [(int(r["shell"]), int(r["line_id"])) for r in csv.DictReader(rows)]


def verify_lines_file(p, base):
    """The sealed set must equal the §5.1 selection recomputed from E call 7 and carry E's ledger hash."""
    why = []
    head = open(p).readline()
    if f"sha256={sha(Path(base) / 'work/w5_lines.csv')}" not in head:
        why.append("LINES_E_LEDGER_HASH")
    lines = read_lines_file(p)
    if len(set(lines)) != len(lines):
        why.append("LINES_DUPLICATE_KEY")
    led, dup = read_ledger(base)
    if dup:
        why.append("E_LEDGER_DUPLICATE")
    if sorted(set(lines)) != select_lines(led):
        why.append("LINES_NE_SELECTION")
    return lines, why


def readout_gate(path):
    """G-READOUT: selftest summary produced by THIS readout file, every case passed."""
    try:
        d = json.load(open(path))
    except Exception:
        return "UNREADABLE"
    if not isinstance(d, dict) or d.get("readout_sha256") != sha(Path(__file__).resolve()):
        return "NOT_THIS_READOUT"
    cases = d.get("cases", [])
    if not cases or d.get("n_cases") != len(cases) or any(not c.get("pass") for c in cases):
        return "NOT_ALL_PASSED"
    return "PASS"


def cmd_gid(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    why, ev = gid(a.base, a.test, post_run=False)
    (out / "gid.json").write_text(json.dumps({"reasons": why, "evidence": ev}, indent=1))
    print(json.dumps({"G-ID": "PASS" if not why else "FAIL", "reasons": why}))
    return 0 if not why else 1


def cmd_readout(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    invalid = []
    why_id, ev = gid(a.base, a.test, post_run=True)
    invalid += [f"G-ID:{w}" for w in why_id]
    st = json.load(open(a.slurm_states))
    for nm in ("E", "R"):
        toks = (st.get(nm) or "").split()
        if not {"TIMELIMIT=1-06:00:00", "REQMEM=64G"} <= set(toks):
            invalid.append(f"SLURM_RESOURCES:{nm}")
    gr = {"E": grun(a.base, st.get("E"), False), "R": grun(a.test, st.get("R"), True)}
    invalid += [f"G-RUN:{k}:{w}" for k, v in gr.items() for w in v]
    g_read = readout_gate(a.selftest)
    if g_read != "PASS":
        invalid.append(f"G-READOUT:{g_read}")
    lines, lw = verify_lines_file(a.lines, a.base)
    invalid += lw
    keys = set(lines)
    ledE, dupE = read_ledger(a.base, keys)
    ledR, dupR = read_ledger(a.test, keys)
    missing = []
    if dupE or dupR:
        missing.append("LEDGER_DUPLICATE")
    d1, w1, traj = delta1(ledE, ledR, lines)
    missing += w1
    cats1 = {s: category(d1[s][7]) for s in SHELLS}
    agg1 = shell_aggregate([cats1[s] for s in SHELLS])
    iters = {}
    for nm, run in (("E", a.base), ("R", a.test)):
        iters[nm] = {}
        for k in (3, 4, 5):
            if not (Path(run) / f"work/physics_comparison/physics_DET_iter{k:04d}.ion.csv").exists():
                missing.append(f"ION_FILE_MISSING:{nm}:k{k}")
                iters[nm][k] = {}
                continue
            rows, dup = load_ion_iter(run, k)
            if dup:
                missing.append(f"ION_DUPLICATE:{nm}:k{k}")
            iters[nm][k] = rows
    cells = []
    for Z in ELEMS:
        for A in AUTHS:
            for s in SHELLS:
                yE, yR = y_series(iters["E"], Z, A, s), y_series(iters["R"], Z, A, s)
                row = {"Z": Z, "authority": A, "shell": s}
                if yE is None or yR is None:
                    row["outcome"] = "MISSING"
                    missing.append(f"CELL:{Z}/{A}/sh{s}")
                else:
                    row.update({f"yE{k}": yE[k] for k in (3, 4, 5)})
                    row.update({f"yR{k}": yR[k] for k in (3, 4, 5)})
                    row.update(cell(yE, yR))
                cells.append(row)
    agg3 = sulfur_mode([c["outcome"] for c in cells if c["Z"] == 16])
    fE, fR = fuv(a.base), fuv(a.test)
    d4 = {}
    for nm in ("a", "b"):
        d4[nm] = (math.log10(fR[f"F{nm}"]) - math.log10(fE[f"F{nm}"])) \
            if fE.get("status") == "OK" and fR.get("status") == "OK" else None
        if d4[nm] is None:
            missing.append(f"FUV_{nm}")
    c4a, c4b = category(d4["a"]), category(d4["b"])
    jE, jR = jb_table(a.base), jb_table(a.test)
    d2 = {k: (jR[k]["log10_ratio"] - jE[k]["log10_ratio"]) if isinstance(jE.get(k), dict) and isinstance(jR.get(k), dict)
          else None for k in jE if k != "radiation_generation"}
    if jE.get("radiation_generation") != 7 or jR.get("radiation_generation") != 7:
        missing.append("JB_GENERATION")
    if any(v is None for v in d2.values()):
        missing.append("JB_UNDEFINED")
    if any(m.startswith(("S_UNDEFINED", "L_EMPTY", "LEDGER")) for m in missing):
        agg1 = "MISSING"
    br, r5 = branch(agg1, agg3, c4a, c4b)
    if missing and br != "R5":
        br, r5 = "R5", r5 + ["MISSING"]
    validity = "X" if (invalid or missing) else "VALID"
    res = {"schema": "NRB_E_READOUT_V1", "base_E": a.base, "test_R": a.test,
           "branch": br if not invalid else None, "statistic_branch": br, "R5_reasons": r5,
           "validity": validity, "invalid_reasons": invalid, "missing": missing,
           "delta1_call7": {f"sh{s}": d1[s][7] for s in SHELLS},
           "category1": cats1, "aggregate1": agg1,
           "sulfur_counts": {o: sum(1 for c in cells if c["Z"] == 16 and c["outcome"] == o) for o in "abcd"},
           "aggregate3": agg3, "distribution30": {o: sum(1 for c in cells if c["outcome"] == o) for o in "abcd"},
           "delta4": d4, "category4": {"a": c4a, "b": c4b}, "G-READOUT": g_read,
           "identity_evidence": ev}
    with open(out / "source_shells.csv", "w") as fh:
        fh.write("shell,n_lines," + ",".join(f"delta1_c{c}" for c in range(3, 8)) + ",category_c7\n")
        for s in SHELLS:
            fh.write(f"{s},{sum(1 for x in lines if x[0] == s)}," +
                     ",".join("" if d1[s][c] is None else repr(d1[s][c]) for c in range(3, 8)) + f",{cats1[s]}\n")
    for nm, tab in (("line_trajectory.csv", traj), ("population23.csv", cells)):
        keys_ = sorted({k for r in tab for k in r})
        with open(out / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys_)
            w.writeheader()
            w.writerows(tab)
    (out / "auxiliary.json").write_text(json.dumps({"JB_log10": {"E": jE, "R": jR, "delta": d2},
                                                    "FUV": {"E": fE, "R": fR, "delta_log10": d4}}, indent=1))
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    me = Path(__file__).resolve()
    cons = []
    for root in (a.base, a.test):
        for rel in ["work/w5_lines.csv", "work/lumina_spectrum.csv", "RUN_FOOTER.txt", "stderr.log"] + \
                   [f"work/physics_comparison/physics_DET_iter{k:04d}.{x}" for k in range(6) for x in ("manifest.json", "ion.csv")] + \
                   ["work/physics_comparison/physics_DET_iter0005.spectral.csv", "work/physics_comparison/physics_DET_iter0005.shell.csv"]:
            q = Path(root) / rel
            cons.append({"path": str(q), "sha256": sha(q) if q.exists() else None})
    (out / "evidence_manifest.json").write_text(json.dumps(
        {"readout": {"path": str(me), "sha256": sha(me)},
         "deps": {p: sha(me.parent / p) for p in ("w5_cont_readout.py", "nt_eta_readout.py")},
         "lines": {"path": a.lines, "sha256": sha(a.lines)}, "consumed": cons,
         "gate_evidence": [{"path": x, "sha256": sha(x)} for x in (a.selftest, a.slurm_states)]}, indent=1))
    print(json.dumps({k: res[k] for k in ("branch", "statistic_branch", "R5_reasons", "validity", "invalid_reasons",
                                          "aggregate1", "aggregate3", "category4", "delta1_call7", "sulfur_counts")},
                     default=str))
    return 0


# ------------------------------------------------------------------ selftest
def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want):
        ok = got == want
        res.append({"case": name, "pass": ok, "got": str(got), "want": str(want)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got} want {want}")

    nxt = math.nextafter
    # categories at exact boundaries
    case("-0.30 -> down", category(-THR), "down")
    case("+0.30 -> up", category(THR), "up")
    case("just inside -0.30 -> flat", category(nxt(-THR, 0.0)), "flat")
    case("just inside +0.30 -> flat", category(nxt(THR, 0.0)), "flat")
    case("NaN -> MISSING", category(float("nan")), "MISSING")
    # ① shell aggregation: all 21 count distributions of 5 shells over 3 categories
    bad = n = 0
    for nd in range(6):
        for nu in range(6 - nd):
            nf = 5 - nd - nu
            n += 1
            cats = ["down"] * nd + ["up"] * nu + ["flat"] * nf
            got = shell_aggregate(cats)
            want = "down" if nd >= 3 else "up" if nu >= 3 else "flat" if nf >= 3 else "mixed"
            bad += got != want
    case("① all 21 distributions", (n, bad), (21, 0))
    case("① (2,2,1) -> mixed", shell_aggregate(["down", "down", "up", "up", "flat"]), "mixed")
    case("① 3/5 -> down", shell_aggregate(["down"] * 3 + ["up", "flat"]), "down")
    case("① missing -> MISSING", shell_aggregate(["down"] * 4 + ["MISSING"]), "MISSING")
    # ③ cell partition reuses NT-ETA cell_outcome; spot checks + sign convention
    case("D=0.15 delta=-0.60 -> a", cell_outcome(-0.60, 0.15), "a")
    D3 = 0.1
    while 3.0 * D3 > THR:
        D3 = nxt(D3, 0.0)
    case("3D==0.30 below threshold -> c", cell_outcome(nxt(-THR, 0.0), D3), "c")
    case("delta=+T -> b", cell_outcome(THR, 0.0), "b")
    c = cell({3: 0.0, 4: 0.0, 5: 0.0}, {3: -1.0, 4: -1.0, 5: -1.0})
    case("sign R - E: R lower -> negative delta3", (c["delta3"], c["outcome"]), (-1.0, "a"))
    c = cell({3: 1.0, 4: 0.0, 5: 0.0}, {3: 0.0, 4: 0.0, 5: 0.0})
    case("E-side drift enters D", (c["D"], c["outcome"]), (0.5, "c" if 3 * 0.5 <= THR else "d"))
    c = cell({3: 0.0, 4: 0.0, 5: 0.0}, {3: 0.0, 4: -2.0, 5: 0.0})
    case("log mean of k4,k5 (-1.0), k3/k5 drift only", (c["delta3"], c["D"]), (-1.0, 0.0))
    # ③ sulfur: all 286 count distributions
    bad = n = 0
    for na in range(11):
        for nb in range(11 - na):
            for nc in range(11 - na - nb):
                nd = 10 - na - nb - nc
                n += 1
                cnt = {"a": na, "b": nb, "c": nc, "d": nd}
                top = max(cnt.values())
                modes = [k for k in "abcd" if cnt[k] == top]
                want = modes[0] if len(modes) == 1 else "mixed"
                bad += sulfur_mode(["a"] * na + ["b"] * nb + ["c"] * nc + ["d"] * nd) != want
    case("③ all 286 sulfur distributions", (n, bad), (286, 0))
    case("③ (4,3,2,1) -> a", sulfur_mode(list("aaaabbbccd")), "a")
    # branch: 36 combinations of stage 2, plus R5 precedence for every one
    cnt = {}
    prec_bad = 0
    for c1 in ("down", "flat", "up"):
        for c3 in "abcd":
            for c4 in ("down", "flat", "up"):
                b, w = branch(c1, c3, c4, "flat")
                cnt[b] = cnt.get(b, 0) + 1
                opp = "up" if c4 == "down" else "down" if c4 == "up" else None
                if opp and branch(c1, c3, c4, opp)[0] != "R5":
                    prec_bad += 1
    def expect(c1, c3, c4):
        if c1 in ("flat", "up"):
            return "R0"
        return {"a": "R1" if c4 == "down" else "R2", "b": "R4", "c": "R3", "d": "R3"}[c3]
    per_bad = sum(branch(c1, c3, c4, "flat")[0] != expect(c1, c3, c4)
                  for c1 in ("down", "flat", "up") for c3 in "abcd" for c4 in ("down", "flat", "up"))
    case("stage 2: each of 36 combinations gives its expected branch", per_bad, 0)
    case("stage 2: 36 combinations R0/R1/R2/R3/R4 = 24/1/2/6/3",
         tuple(cnt.get(k, 0) for k in ("R0", "R1", "R2", "R3", "R4")), (24, 1, 2, 6, 3))
    case("④b opposite to ④a always -> R5 first", prec_bad, 0)
    case("counterexample: all down, a, 4a -0.4, 4b +0.4 -> R5",
         branch("down", "a", category(-0.4), category(0.4))[0], "R5")
    case("same with 4b -0.1 -> R1", branch("down", "a", category(-0.4), category(-0.1))[0], "R1")
    case("several R5 reasons -> one branch",
         branch("mixed", "mixed", "down", "up"), ("R5", ["MIXED_1", "MIXED_3", "FUV_OPPOSITE"]))
    case("missing -> R5", branch("MISSING", "a", "down", "down"), ("R5", ["MISSING"]))
    # ① line-level: median of per-line log differences, not difference of medians
    ledE = {(7, 10, 1): {"status": "OK", "L_l": "1", "beta": "0"}, (7, 10, 2): {"status": "OK", "L_l": "100", "beta": "0"},
            (7, 10, 3): {"status": "OK", "L_l": "10", "beta": "0"}}
    ledR = {}
    for c_ in range(3, 8):
        ledR[(c_, 10, 1)] = {"status": "OK", "L_l": "100", "beta": "0"}
        ledR[(c_, 10, 2)] = {"status": "OK", "L_l": "1", "beta": "0"}
        ledR[(c_, 10, 3)] = {"status": "OK", "L_l": "10", "beta": "0"}
    def _stamp(dd):
        for k_, v_ in dd.items():
            v_.update(operator="CMFGEN_NONOVERLAP_SOBOLEV", rad_generation=str(k_[0]), material_generation=str(k_[0]))
    _stamp(ledE)
    _stamp(ledR)
    d, w, _ = delta1(ledE, ledR, [(10, 1), (10, 2), (10, 3)])
    case("symmetric per-line diffs (+2,-2,0) -> median 0", round(d[10][7], 12), 0.0)
    ledR[(7, 10, 3)] = {"status": "OK", "L_l": "0.1", "beta": "0"}
    _stamp(ledR)
    d, w, _ = delta1(ledE, ledR, [(10, 1), (10, 2), (10, 3)])
    case("asymmetric: per-line diffs (+2,-2,-2) median -2 (difference of medians would be -1)",
         round(d[10][7], 12), -2.0)
    ledR.pop((5, 10, 2))
    d, w, _ = delta1(ledE, ledR, [(10, 1), (10, 2), (10, 3)])
    case("missing fixed key -> undefined + reason", (d[10][5], any(x.startswith("S_UNDEFINED:sh10/l2/c5") for x in w)), (None, True))
    case("beta==1 -> undefined source", source({"status": "OK", "L_l": "1", "beta": "1"}), None)
    case("beta>1 with L<0 -> positive S defined", round(source({"status": "OK", "L_l": "-1", "beta": "1.1"}), 9), 10.0)
    case("negative S -> undefined", source({"status": "OK", "L_l": "-1", "beta": "0.5"}), None)
    okrow = lambda c_, L_: {"status": "OK", "L_l": L_, "beta": "0", "operator": "CMFGEN_NONOVERLAP_SOBOLEV",
                            "rad_generation": str(c_), "material_generation": str(c_)}
    E2 = {(7, 10, l): okrow(7, "1") for l in (1, 2)}
    R2 = {(c_, 10, l): okrow(c_, str(10 ** (2 * (l - 1)))) for c_ in range(3, 8) for l in (1, 2)}
    case("duplicate fixed key reported", "LINES_DUPLICATE_KEY" in delta1(E2, R2, [(10, 1), (10, 2), (10, 2)])[1], True)
    R3 = dict(R2)
    R3[(7, 10, 1)] = dict(R2[(7, 10, 1)], rad_generation="999")
    case("wrong radiation generation -> undefined", delta1(E2, R3, [(10, 1), (10, 2)])[0][10][7], None)
    R4 = dict(R2)
    R4[(7, 10, 1)] = dict(R2[(7, 10, 1)], operator="WRONG")
    case("wrong operator -> undefined", delta1(E2, R4, [(10, 1), (10, 2)])[0][10][7], None)
    its = {k: {10: {"population_generation": str(GEN_K[k]), "Z16_stage_q1_cm3": "1e-300", "Z16_stage_q2_cm3": "1e300"}}
           for k in (3, 4, 5)}
    case("q2/q1 overflow handled in log space", y_series(its, 16, "stage", 10)[5], 600.0)
    its2 = {k: {10: {"population_generation": str(GEN_K[k]), "Z16_stage_q1_cm3": "1e300", "Z16_stage_q2_cm3": "1e-300"}}
            for k in (3, 4, 5)}
    case("q2/q1 underflow handled in log space", y_series(its2, 16, "stage", 10)[5], -600.0)
    case("selection: tau 1e3 inclusive, D edges", select_lines({
        (7, 10, 1): {"status": "OK", "tau_eff": "1000", "nu_hz": repr(C_CGS / 1000e-8)},
        (7, 10, 2): {"status": "OK", "tau_eff": "999.9999", "nu_hz": repr(C_CGS / 1000e-8)},
        (7, 10, 3): {"status": "OK", "tau_eff": "5000", "nu_hz": repr(C_CGS / 1500e-8)},
        (7, 10, 4): {"status": "OK", "tau_eff": "5000", "nu_hz": repr(C_CGS / 912e-8)},
        (6, 10, 5): {"status": "OK", "tau_eff": "5000", "nu_hz": repr(C_CGS / 1000e-8)},
        (7, 11, 6): {"status": "OK", "tau_eff": "5000", "nu_hz": repr(C_CGS / 1000e-8)}}), [(10, 1), (10, 4)])
    # ②/④ integrals
    pts = [(1100.0, 1100.0), (2100.0, 2100.0), (3900.0, 1.0), (6100.0, 1.0)]
    v, s_ = band_integral(pts, 1200.0, 1500.0)
    case("linear L_lambda integral 1200-1500", (round(v, 6), s_), (round((1500.0 ** 2 - 1200.0 ** 2) / 2.0, 6), "OK"))
    case("unsupported FUV band", band_integral([(1300.0, 1.0), (2000.0, 1.0)], 1200.0, 1500.0)[1], "UNSUPPORTED")
    T = 1.0e4
    rows = []
    nu = C_CGS / (400e-8)
    while nu > C_CGS / (2000e-8):
        lo = nu * math.exp(-0.01)
        rows.append((lo, nu, math.exp(log_planck(math.sqrt(lo * nu), T))))
        nu = lo
    case("J=B -> log10 J/B = 0 in D band", round(jb_ratio(rows, T, 912.0, 1500.0)["log10_ratio"], 12), 0.0)
    lo_, hi_ = C_CGS / (1500e-8), C_CGS / (912e-8)
    case("J/B partial support -> undefined", jb_ratio([(lo_, lo_ + (hi_ - lo_) / 4, 1e-6)], T, 912.0, 1500.0), None)
    inside = [i for i, (lo, hi, _) in enumerate(rows) if 912.0 < C_CGS / math.sqrt(lo * hi) * 1e8 < 1500.0]
    mid = inside[len(inside) // 2]
    dup_rows = rows + [rows[mid]]
    case("J/B duplicated bin -> undefined", jb_ratio(dup_rows, T, 912.0, 1500.0), None)
    gap_rows = [r for i, r in enumerate(rows) if i != mid]
    case("J/B gap -> undefined", jb_ratio(gap_rows, T, 912.0, 1500.0), None)
    case("J/B NaN -> undefined", jb_ratio([(1e15, 1.1e15, float("nan"))], T, 912.0, 3000.0), None)
    # identity / execution
    E_, R_ = "/E", "/R"
    base = {k: f"{E_}/input/x" for k in PATH_KEYS}
    base.update(MUST_KEEP)
    base["LUMINA_A210_SPRODUCER_CAPTURE"] = "0"
    dR = {k: (v.replace(E_, R_, 1) if k in PATH_KEYS else v) for k, v in base.items()}
    dR["LUMINA_A210_SPRODUCER_CAPTURE"] = "1"
    dR["LUMINA_BB_NRB"] = "1"
    case("exports: exactly the two registered keys", exports_delta2(base, dR, E_, R_), [])
    case("exports: extra diff", "EXPORTS_EXTRA_DIFF:LUMINA_ION_NT" in exports_delta2(base, dict(dR, LUMINA_ION_NT="0"), E_, R_), True)
    case("exports: NRB missing", any(w.startswith("EXPORTS_DELTA:LUMINA_BB_NRB") for w in exports_delta2(base, {k: v for k, v in dR.items() if k != "LUMINA_BB_NRB"}, E_, R_)), True)
    case("exports: empty string != absent", any(w.startswith("EXPORTS_DELTA:LUMINA_BB_NRB") for w in exports_delta2(dict(base, LUMINA_BB_NRB=""), dR, E_, R_)), True)
    case("exports: MUST_KEEP violated", "EXPORTS_MUST_KEEP:LUMINA_NT_FULL" in exports_delta2(dict(base, LUMINA_NT_FULL="1"), dict(dR, LUMINA_NT_FULL="1"), E_, R_), True)
    case("binary mismatch", bins_reason({"a": W5_BIN, "b": "0" * 64}), ["BINARY_MISMATCH"])
    case("approved mismatch", approved_reasons({"src/x.c": "a"}, "src/x.c=b"), ["APPROVED_MISMATCH:src/x.c"])
    good_log = ["[BB-NRB] iter=0 operator=1 armed=0 captured=0 n_shells=50 skip_z=none\n"] + \
        [f"[BB-NRB] iter={k} operator=2 armed=1 captured=1 n_shells=50 skip_z=none\n" for k in range(6)] + \
        ["[cmf_fine][SPRODUCER-CAPTURE] status=CAPTURED cells=1 n_shells=50 operator=CMFGEN_NONOVERLAP_SOBOLEV\n"] * 6
    case("NRB witness good", nrb_witness(good_log, True), [])
    case("NRB tag deleted", nrb_witness([l for l in good_log if "iter=3 operator=2" not in l], True), ["BB_NRB_ITER_WITNESS:k3"])
    case("NRB tag duplicated", any(w.startswith("BB_NRB_DUPLICATE") for w in nrb_witness(good_log + [good_log[2]], True)), True)
    case("NRB armed=0 in Sobolev", nrb_witness([l.replace("iter=2 operator=2 armed=1", "iter=2 operator=2 armed=0") for l in good_log], True), ["BB_NRB_ITER_WITNESS:k2"])
    case("capture count 5", nrb_witness(good_log[:-1], True), ["SPRODUCER_CAPTURED_COUNT:5"])
    case("NRB FATAL", "BB_NRB_FATAL" in nrb_witness(good_log + ["[BB-NRB][FATAL] reason=X\n"], True), True)
    bogus = good_log[:7] + ["[cmf_fine][SPRODUCER-CAPTURE] status=CAPTURED_BOGUS n_shells=500 operator=WRONG\n"] * 6
    case("malformed CAPTURED lines rejected", "SPRODUCER_CAPTURE_MALFORMED" in nrb_witness(bogus, True), True)
    case("INIT captured=1 rejected", "BB_NRB_INIT_WITNESS" in nrb_witness([good_log[0].replace("captured=0", "captured=1")] + good_log[1:], True), True)
    case("baseline E with captured=1 tag rejected", nrb_witness(["[BB-NRB] iter=0 operator=1 armed=0 captured=1\n"], False), ["BASELINE_HAS_NRB"])
    case("baseline E: OFF accepted", nrb_witness(["[BB-NRB] iter=0 operator=1 armed=0\n"], False), [])
    case("baseline E with armed NRB rejected", nrb_witness(good_log, False), ["BASELINE_HAS_NRB"])
    tmp = out / "nc_tmp"
    (tmp / "work").mkdir(parents=True)
    (tmp / "work/cmfgen_final_publication.json").write_text(json.dumps(
        {"complete": True, "population_generation": 8, "required_population_generation": 7,
         "files": {"plasma": {"path": "p.csv", "authority": "PUBLIC_NE_TE"},
                   "saha": {"path": "s.csv", "authority": "SAHA_STAGE"},
                   "nlte": {"path": "n.csv", "authority": "NLTE_LEVEL"}}}))
    (tmp / "work/p.csv").write_text("x")
    (tmp / "work/s.csv").write_text("x")
    case("publication: required gen != 8 and missing referenced file",
         publication_reasons(tmp / "work"), ["FINAL_PUBLICATION_STATE", "FINAL_PUBLICATION_FILE:nlte"])
    (tmp / "input").mkdir()
    (tmp / "input/resolved_lumina.exports").write_text('declare -x A="1"\n')
    (tmp / "job.slurm").write_text("x")
    (tmp / "STAGE_IDENTITY.txt").write_text("effective_exports_sha256=00\nrunner_sha256=00\n")
    (tmp / "RUN_FOOTER.txt").write_text("effective_exports_sha256=00\npartition=h200\nallocated_gpus=2\nallocated_cpus=16\n")
    case("identity linkage: stage/footer hashes and H200 rejected", linkage_reasons(tmp, "R", True),
         ["STAGE_EXPORTS_HASH:R", "STAGE_RUNNER_HASH:R", "FOOTER_EXPORTS_HASH:R", "FOOTER_RESOURCES:R"])
    (tmp / "summary_one.json").write_text(json.dumps({"readout_sha256": "0" * 64, "n_cases": 1, "cases": [{"pass": True}]}))
    case("G-READOUT rejects a summary not produced by this readout", readout_gate(tmp / "summary_one.json"), "NOT_THIS_READOUT")
    (tmp / "summary_two.json").write_text(json.dumps({"readout_sha256": sha(Path(__file__).resolve()), "n_cases": 2,
                                                     "cases": [{"pass": True}, {"pass": False}]}))
    case("G-READOUT rejects a failed case", readout_gate(tmp / "summary_two.json"), "NOT_ALL_PASSED")
    gf = {"terminal_state": "SUCCESS", "model_rc": "0", "allocated_gpus": "2", "gpu_kind": "A100",
          "outer_iterations": "6", "fixed_te_loader_consumed": "1"}
    case("forbidden tag list excludes the normal DPROXY BLOCKED diagnostic",
         [t for t in FORBIDDEN if "[DPROXY][SL-POP-COMPLETE] status=BLOCKED x".startswith(t)], [])
    case("footer good", footer_reasons(gf, "COMPLETED RESTARTS=0"), [])
    case("not completed", footer_reasons(dict(gf, terminal_state="FAILED"), "TIMEOUT RESTARTS=0"),
         ["FOOTER_NOT_SUCCESS", "SLURM_STATE:TIMEOUT RESTARTS=0"])
    # normal control: E vs E statistics -> deltas 0, branch R0
    if a.base:
        led, _ = read_ledger(a.base)
        sel = select_lines(led)
        ledE7 = {k: v for k, v in led.items() if k[0] == 7 and (k[1], k[2]) in set(sel)}
        ledR_E = {(c_, s, l): dict(led[(7, s, l)], rad_generation=str(c_), material_generation=str(c_))
                  for c_ in range(3, 8) for s, l in sel}
        d, w, _ = delta1(ledE7, ledR_E, sel)
        case("E vs E: Δ₁ = 0 for all shells, no missing", ([d[s][7] for s in SHELLS], w), ([0.0] * 5, []))
        lf = out / "nc_tmp/lines_bad.csv"
        lf.write_text("# E ledger sha256=" + "0" * 64 + "\nshell,line_id\n10,1\n10,1\n")
        case("lines file: wrong E hash, duplicate key, not the selection",
             verify_lines_file(lf, a.base)[1], ["LINES_E_LEDGER_HASH", "LINES_DUPLICATE_KEY", "LINES_NE_SELECTION"])
        cats = shell_aggregate([category(d[s][7]) for s in SHELLS])
        its = {k: load_ion_iter(a.base, k)[0] for k in (3, 4, 5)}
        outs = []
        for A in AUTHS:
            for s in SHELLS:
                y = y_series(its, 16, A, s)
                outs.append(cell(y, y)["outcome"])
        f = fuv(a.base)
        d4a = math.log10(f["Fa"]) - math.log10(f["Fa"])
        case("E vs E: ③ cells only c/d, branch R0",
             (all(o in "cd" for o in outs), branch(cats, sulfur_mode(outs) if sulfur_mode(outs) != "mixed" else "c",
                                                    category(d4a), category(d4a))[0]), (True, "R0"))
    nfail = sum(1 for r in res if not r["pass"])
    (out / "readout_selftest.json").write_text(json.dumps(
        {"readout_sha256": sha(Path(__file__).resolve()), "n_cases": len(res), "failures": nfail, "cases": res}, indent=1))
    print(f"\nNRB-E NC: {len(res) - nfail}/{len(res)} -> {'PASS' if nfail == 0 else 'FAIL'}")
    return 0 if nfail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("gid")
    p.add_argument("--base", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("readout")
    for k in ("--base", "--test", "--lines", "--selftest", "--slurm-states", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("selftest")
    p.add_argument("--base")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"select": cmd_select, "gid": cmd_gid, "readout": cmd_readout, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
