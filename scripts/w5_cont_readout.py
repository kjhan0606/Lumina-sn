#!/usr/bin/env python3
"""DET-W5-CONT-SOURCE readout (prereg docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md §2.5, §4, §5).

Subcommands
  referents --c1 C1 --out DIR
      C1-only fixed referents, produced BEFORE the W5 run: C1 population table,
      public toy06 19.480 d II/III fractions interpolated separately at the shell
      centre velocity (no extrapolation), and the line-selection manifest
      (shells 10/15/19/24/30 x S/Si/Ca II/III x four wavelength bundles, top four
      mapped lines by n_l f_lu from the C1 final level populations; both endpoint
      levels must exist and f_lu>0; ties by line index; empty bundle = EMPTY).
      The manifest is a readout referent only; it is never a runtime input.
  readout --run RUN --c1 C1 --referents DIR --out DIR [--gates GATES.json] [--stats-only]
      §5 tables (population authorities, delta/m/M and the registered branch,
      window J/B, emission, outer trajectory, W5-internal line response) and
      result.json.  Branch codes: X=0, MOVE=1, NONE=2.  G-ID/G-SELF come from the
      driver's gates JSON ({"G-ID": "PASS", "G-SELF": "PASS"}); without it the
      full branch is X (GATE_EVIDENCE_NOT_SUPPLIED); --stats-only additionally
      skips G-RUN (used for the C1-vs-C1 demonstration).
  selftest [--c1 C1] --out DIR
      NC-READOUT battery: missing row / NaN / zero population -> X, threshold
      just below / exactly at 0.30 dex, both directions, minority-shell movement,
      C1 vs C1 -> NONE, line-selection rules (empty bundle, ties, ineligible
      lines, top four), window-integral cases.  rc=0 iff every case is
      classified as registered.

Reused logic (ranges per §3): diag_fable_20260930/ionmap.py (public fraction
interpolation, stage vs level-sum authorities), fixall_20260928/ab_readout.py
(emission band integrals), lamp_20260930/window_by_iter.py (shell/spectral
join).  Not carried over: hard-coded 398939 references, Planck exponent limits
(clip/min(x,700)/B=0 for x>=700), maximum(...,1e-300) denominators.  Ratios that
are not representable are reported in log form or as an explicit undefined state.
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
SHELLS = (10, 15, 19, 24, 30)
ELEMS = (14, 16, 20)
IONS = ((14, 1), (14, 2), (16, 1), (16, 2), (20, 1), (20, 2))
BUNDLES = ((100.0, 531.0, False), (531.0, 912.0, False),
           (912.0, 2000.0, False), (2000.0, 20000.0, True))   # last is closed
WINDOWS = (("blue_end-531", None, 531.0), ("531-760", 531.0, 760.0),
           ("760-912", 760.0, 912.0), ("912-1500", 912.0, 1500.0),
           ("1500-2000", 1500.0, 2000.0))
THRESH = 0.30
BR_CODE = {"W5-X": 0, "W5-MOVE": 1, "W5-NONE": 2}
PUB = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06"
PUB_SYM = {14: "si", 16: "s", 20: "ca"}
N_ITER = 6


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# ----------------------------------------------------------------- populations
def load_pops(run):
    """{k: {shell: row}} from physics_DET_iter*.ion.csv plus shell.csv."""
    pc = Path(run) / "work" / "physics_comparison"
    out, shells = {}, {}
    for k in range(N_ITER):
        f = pc / f"physics_DET_iter{k:04d}.ion.csv"
        g = pc / f"physics_DET_iter{k:04d}.shell.csv"
        if f.exists():
            out[k] = {int(r["shell_id"]): r for r in csv.DictReader(open(f))}
        if g.exists():
            shells[k] = {int(r["shell_id"]): r for r in csv.DictReader(open(g))}
    return out, shells


def xval(pops, k, s, Z, auth):
    """x = n(q2)/n(q1) for the authority; (value, status)."""
    row = pops.get(k, {}).get(s)
    if row is None:
        return float("nan"), "MISSING_ROW"
    tag = "stage" if auth == "stage" else "nlte"
    q1 = fnum(row.get(f"Z{Z}_{tag}_q1_cm3"))
    q2 = fnum(row.get(f"Z{Z}_{tag}_q2_cm3"))
    if not (math.isfinite(q1) and math.isfinite(q2)):
        return float("nan"), "NONFINITE"
    if q1 <= 0.0 or q2 <= 0.0:
        return q2 / q1 if q1 != 0.0 else float("nan"), "NONPOSITIVE"
    return q2 / q1, "OK"


def deltas(pw, pc):
    """delta_{Z,A,s} over k=4,5; status X when any log is undefined."""
    d, bad = {}, []
    for Z in ELEMS:
        for A in ("stage", "levels"):
            for s in SHELLS:
                acc = 0.0
                ok = True
                for k in (4, 5):
                    xw, sw = xval(pw, k, s, Z, A)
                    xc, sc = xval(pc, k, s, Z, A)
                    if sw != "OK" or sc != "OK":
                        ok = False
                        bad.append(f"Z{Z}/{A}/sh{s}/k{k}:W5={sw},C1={sc}")
                        continue
                    acc += math.log10(xw) - math.log10(xc)
                d[(Z, A, s)] = 0.5 * acc if ok else float("nan")
    return d, bad


def branch_from_deltas(d):
    """Registered statistic: m_{Z,A}=median_s|delta|, M=max; MOVE iff M>=0.30."""
    m = {}
    for Z in ELEMS:
        for A in ("stage", "levels"):
            v = [d.get((Z, A, s), float("nan")) for s in SHELLS]
            if len(v) != len(SHELLS) or any(not math.isfinite(x) for x in v):
                return "W5-X", m, float("nan"), ["UNDEFINED_DELTA"]
            a = sorted(abs(x) for x in v)
            m[(Z, A)] = a[len(a) // 2]
    M = max(m.values())
    return ("W5-MOVE" if M >= THRESH else "W5-NONE"), m, M, []


# ----------------------------------------------------------------- public
def read_pub(sym, t="19.480"):
    rows, on = [], False
    with open(f"{PUB}/ionfrac_{sym}_toy06_cmfgen.txt") as f:
        for ln in f:
            if ln.startswith("#TIME:"):
                on = ln.split()[1] == t
                continue
            if on and not ln.startswith("#") and ln.strip():
                rows.append([float(x) for x in ln.split()])
    return rows  # v_mid, s0..s4


def interp(v, xs, ys):
    if not xs or v < xs[0] or v > xs[-1]:
        return float("nan")        # no extrapolation: undefined
    for i in range(len(xs) - 1):
        if xs[i] <= v <= xs[i + 1]:
            t = (v - xs[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0.0
            return ys[i] + t * (ys[i + 1] - ys[i])
    return float("nan")


def public_x(shells_k):
    out = {}
    for Z in ELEMS:
        rows = read_pub(PUB_SYM[Z])
        vs = [r[0] for r in rows]
        for s in SHELLS:
            h = shells_k.get(s)
            if h is None:
                out[(Z, s)] = (float("nan"), float("nan"), float("nan"), "MISSING_SHELL")
                continue
            vmid = 0.5 * (fnum(h["v_inner_cm_s"]) + fnum(h["v_outer_cm_s"])) / 1e5
            f1 = interp(vmid, vs, [r[2] for r in rows])
            f2 = interp(vmid, vs, [r[3] for r in rows])
            st = "OK" if (math.isfinite(f1) and math.isfinite(f2) and f1 > 0.0) else "UNDEFINED"
            out[(Z, s)] = (vmid, f1, f2, st)
    return out


# ----------------------------------------------------------------- line selection
def select_lines(lines, levels, pops, mapped_ions):
    """lines: [(line_id, Z, q, lower, upper, wav_A, f_lu)]; levels: set (Z,q,lev);
    pops: {(Z,q,shell,lev): n}; returns rows and eligibility counts."""
    rows, stats = [], {"ineligible_endpoint": 0, "ineligible_f": 0, "no_population": 0}
    cand = {}
    for (lid, Z, q, lo, up, wav, flu) in lines:
        if (Z, q) not in mapped_ions:
            continue
        b = None
        for i, (a, c, closed) in enumerate(BUNDLES):
            if a <= wav < c or (closed and wav == c):
                b = i
        if b is None:
            continue
        if (Z, q, lo) not in levels or (Z, q, up) not in levels:
            stats["ineligible_endpoint"] += 1
            continue
        if not (flu > 0.0):
            stats["ineligible_f"] += 1
            continue
        cand.setdefault((Z, q, b), []).append((lid, lo, up, wav, flu))
    for s in SHELLS:
        for (Z, q) in IONS:
            if (Z, q) not in mapped_ions:
                continue
            for b in range(len(BUNDLES)):
                scored = []
                for (lid, lo, up, wav, flu) in cand.get((Z, q, b), []):
                    n = pops.get((Z, q, s, lo))
                    if n is None or not math.isfinite(n):
                        stats["no_population"] += 1
                        continue
                    scored.append((-(n * flu), lid, lo, up, wav, flu, n))
                scored.sort(key=lambda t: (t[0], t[1]))
                if not scored:
                    rows.append({"shell": s, "Z": Z, "ion": q, "bundle": b, "rank": 0,
                                 "line_id": "EMPTY"})
                for r, t in enumerate(scored[:4], 1):
                    rows.append({"shell": s, "Z": Z, "ion": q, "bundle": b, "rank": r,
                                 "line_id": t[1], "wavelength_A": t[4], "f_lu": t[5],
                                 "lower_level": t[2], "upper_level": t[3],
                                 "n_lower": t[6], "score_n_flu": -t[0]})
    return rows, stats


def load_c1_line_inputs(c1):
    m = Path(c1) / "input" / "model"
    levels, energy = set(), {}
    for r in csv.DictReader(open(m / "levels.csv")):
        key = (int(r["atomic_number"]), int(r["ion_number"]), int(r["level_number"]))
        levels.add(key)
        energy[key] = float(r["energy_eV"])
    lines, id_mismatch = [], 0
    with open(m / "line_list.csv") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if int(r["line_id"]) != i:
                id_mismatch += 1
            Z, q = int(r["atomic_number"]), int(r["ion_number"])
            if Z not in ELEMS or q not in (1, 2):
                continue
            lines.append((i, Z, q, int(r["level_number_lower"]),
                          int(r["level_number_upper"]), float(r["wavelength"]),
                          float(r["f_lu"])))
    pops, e_mismatch, gens = {}, 0, set()
    with open(Path(c1) / "work" / "nlte_levels_final.csv") as f:
        for r in csv.DictReader(f):
            Z, q, s = int(r["Z"]), int(r["ion"]), int(r["shell"])
            if (Z, q) not in IONS or s not in SHELLS:
                continue
            lev = int(r["level_idx"])
            e = energy.get((Z, q, lev))
            if e is None or abs(e - float(r["E_eV"])) > 1e-6 * max(1.0, abs(e)):
                e_mismatch += 1
            pops[(Z, q, s, lev)] = float(r["n_pop"])
            gens.add(r["population_generation"])
    mapped = {(Z, q) for (Z, q, _, _) in pops}
    return lines, levels, pops, mapped, {"line_id_vs_row_mismatch": id_mismatch,
                                         "level_energy_mismatch": e_mismatch,
                                         "population_generations": sorted(gens)}


def cmd_referents(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pops, shells = load_pops(a.c1)
    with open(out / "c1_population.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "shell", "Z", "authority", "x", "status", "population_generation"])
        for k in range(N_ITER):
            for s in SHELLS:
                for Z in ELEMS:
                    for A in ("stage", "levels"):
                        x, st = xval(pops, k, s, Z, A)
                        w.writerow([k, s, Z, A, repr(x), st,
                                    pops.get(k, {}).get(s, {}).get("population_generation", "")])
    pub = public_x(shells.get(N_ITER - 1, {}))
    with open(out / "public_x.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shell", "Z", "v_mid_km_s", "f_II", "f_III", "x_pub", "status"])
        for (Z, s), (v, f1, f2, st) in sorted(pub.items(), key=lambda t: (t[0][1], t[0][0])):
            w.writerow([s, Z, repr(v), repr(f1), repr(f2),
                        repr(f2 / f1) if st == "OK" else "UNDEFINED", st])
    lines, levels, lpops, mapped, chk = load_c1_line_inputs(a.c1)
    rows, stats = select_lines(lines, levels, lpops, mapped)
    cols = ["shell", "Z", "ion", "bundle", "rank", "line_id", "wavelength_A", "f_lu",
            "lower_level", "upper_level", "n_lower", "score_n_flu"]
    with open(out / "line_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (repr(r[c]) if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})
    man = {"schema": "W5_REFERENTS_V1", "c1": str(a.c1),
           "mapped_ions": sorted(mapped), "selection_stats": stats, "input_checks": chk,
           "selected_lines": sum(1 for r in rows if r["line_id"] != "EMPTY"),
           "empty_bundles": sum(1 for r in rows if r["line_id"] == "EMPTY"),
           "inputs_sha256": {p: sha256(Path(a.c1) / p) for p in (
               "input/model/line_list.csv", "input/model/levels.csv",
               "work/nlte_levels_final.csv")},
           "outputs_sha256": {n: sha256(out / n) for n in (
               "c1_population.csv", "public_x.csv", "line_manifest.csv")},
           "runtime_input": False}
    (out / "referents_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("mapped_ions", "selection_stats", "input_checks",
                                          "selected_lines", "empty_bundles")}))
    ok = (chk["line_id_vs_row_mismatch"] == 0 and chk["level_energy_mismatch"] == 0 and
          len(mapped) == len(IONS))
    return 0 if ok else 1


# ----------------------------------------------------------------- windows / emission
def log_planck(nu, T):
    """log B_nu(T) without exponent limits (Wien form in log space)."""
    x = H_CGS * nu / (K_CGS * T)
    lpre = math.log(2.0 * H_CGS) + 3.0 * math.log(nu) - 2.0 * math.log(C_CGS)
    if x > 50.0:
        return lpre - x - math.log1p(-math.exp(-x))
    return lpre - math.log(math.expm1(x))


def logsumexp(v):
    v = [x for x in v if x != -math.inf]
    if not v:
        return -math.inf
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def window_ratios(spec_rows, T):
    """R_W = sum J dnu_cap / sum B(nu_b,T) dnu_cap; nu_b = sqrt(nu_lo nu_hi)."""
    nu_top = max(r[1] for r in spec_rows)
    res = {}
    for name, l_lo, l_hi in WINDOWS:
        n_hi = nu_top if l_lo is None else C_CGS / (l_lo * 1e-8)
        n_lo = C_CGS / (l_hi * 1e-8)
        num, lden, nb, bad = 0.0, [], 0, 0
        for lo, hi, J in spec_rows:
            ov = min(hi, n_hi) - max(lo, n_lo)
            if ov <= 0.0:
                continue
            nb += 1
            if not math.isfinite(J):
                bad += 1
                continue
            num += J * ov
            lden.append(log_planck(math.sqrt(lo * hi), T) + math.log(ov))
        ld = logsumexp(lden)
        if bad or nb == 0 or ld == -math.inf:
            res[name] = {"R": None, "log10_R": None, "bins": nb,
                         "status": "NONFINITE_J" if bad else "UNDEFINED"}
            continue
        if num == 0.0:
            res[name] = {"R": 0.0, "log10_R": None, "bins": nb, "status": "ZERO_J"}
            continue
        lr = (math.log(num) - ld) / math.log(10.0)
        R = 10.0 ** lr if -300.0 < lr < 300.0 else None
        res[name] = {"R": R, "log10_R": lr, "bins": nb,
                     "status": "OK" if R is not None else "LOG_ONLY"}
    return res


def load_spectral(run, k, shells):
    f = Path(run) / "work" / "physics_comparison" / f"physics_DET_iter{k:04d}.spectral.csv"
    out = {s: [] for s in shells}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            s = int(r["shell_id"])
            if s in out:
                out[s].append((float(r["nu_lo_Hz"]), float(r["nu_hi_Hz"]), fnum(r["J_nu"])))
    return out


def trapz_band(pts, lo, hi):
    """trapezoid on the recorded grid restricted to [lo,hi) with linear edges."""
    tot = 0.0
    for (l0, f0), (l1, f1) in zip(pts[:-1], pts[1:]):
        a, b = max(l0, lo), min(l1, hi)
        if b <= a:
            continue
        fa = f0 + (f1 - f0) * (a - l0) / (l1 - l0)
        fb = f0 + (f1 - f0) * (b - l0) / (l1 - l0)
        tot += 0.5 * (fa + fb) * (b - a)
    return tot


def emission(run):
    f = Path(run) / "work" / "lumina_spectrum.csv"
    pts = [(float(r["wavelength_angstrom"]), fnum(r["flux"])) for r in csv.DictReader(open(f))]
    pts.sort()
    if any(not math.isfinite(v) for _, v in pts):
        return {"status": "NONFINITE"}
    L = trapz_band(pts, pts[0][0], pts[-1][0] + 1e-9)
    return {"status": "OK", "L_grid": L, "grid_A": [pts[0][0], pts[-1][0]],
            "frac_lt2500": trapz_band(pts, pts[0][0], 2500.0) / L if L != 0 else None,
            "frac_912_2000": trapz_band(pts, 912.0, 2000.0) / L if L != 0 else None,
            "negative_flux_points": sum(1 for _, v in pts if v < 0.0)}


# ----------------------------------------------------------------- G-RUN / logs
TAG_BLOCK = re.compile(r"^\[(cmf_fine|R6|CMFGEN|cmfgen_assemble|R7|A2-10|RADIATION-FIELD)\]\[(BLOCKED|FATAL)\]")


def parse_log(run):
    p = Path(run) / "stderr.log"
    info = {"present": p.exists(), "bin_tags": 0, "fine_tags": [], "blocked": [],
            "conv": [], "fine_status": []}
    if not p.exists():
        return info
    for ln in open(p, errors="replace"):
        if ln.startswith("[cmfgen_assemble][W5-CONT]"):
            info["bin_tags"] += 1
        elif ln.startswith("[cmf_fine][W5-CONT]"):
            info["fine_tags"].append(dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv))
        elif ln.startswith("[cmfgen_solve_J][CONV]"):
            info["conv"].append(dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv))
        elif ln.startswith("[cmf_fine][EXACT-MULTIGPU-EPOCH]") or \
                ln.startswith("[cmf_fine][EXACT-EPOCH]"):
            m = re.search(r"status=(\S+).*?iterations=(\d+).*?residual=(\S+)", ln)
            if m:
                info["fine_status"].append({"status": m.group(1), "iterations": int(m.group(2)),
                                            "residual": m.group(3)})
        if TAG_BLOCK.match(ln):
            info["blocked"].append(ln.strip()[:240])
    return info


def g_run(run, log, lines_summary):
    reasons = []
    w = Path(run) / "work"
    need = [w / "nlte_levels_final.csv", w / "lumina_spectrum.csv", w / "w5_lines.csv"]
    for k in range(N_ITER):
        for ext in ("ion.csv", "shell.csv", "spectral.csv"):
            need.append(w / "physics_comparison" / f"physics_DET_iter{k:04d}.{ext}")
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        reasons.append(f"MISSING_FILES:{len(miss)}")
    foot = Path(run) / "RUN_FOOTER.txt"
    if foot.exists():
        if "terminal_state=SUCCESS" not in foot.read_text(errors="replace"):
            reasons.append("FOOTER_NOT_SUCCESS")
    else:
        reasons.append("NO_RUN_FOOTER")
    if not log["present"]:
        reasons.append("NO_STDERR_LOG")
    if log["blocked"]:
        reasons.append(f"BLOCKED_LINES:{len(log['blocked'])}")
    ft = log["fine_tags"]
    ops = [t.get("operator") for t in ft]
    if len(ft) != N_ITER + 1 or ops[:1] != ["INIT_SHARED_GAUSSIAN"] or \
            any(o != "CMFGEN_NONOVERLAP_SOBOLEV" for o in ops[1:]):
        reasons.append(f"FINE_TAGS:{len(ft)}")
    for t in ft:
        if not (t.get("interp_cells") == t.get("source_cells") == t.get("fine_cells")):
            reasons.append("FINE_CONSUMPTION_COUNT")
    if log["bin_tags"] < N_ITER + 1:
        reasons.append(f"BIN_TAGS:{log['bin_tags']}")
    fs = log["fine_status"]
    if len(fs) != N_ITER + 1 or any(s["status"] != "OK" for s in fs):
        reasons.append(f"FINE_SOLVE_STATUS:{len(fs)}")
    if lines_summary.get("status") != "OK":
        reasons.append(f"W5_LINES:{lines_summary.get('status')}")
    return reasons


def evidence_reasons(lines, windows, em):
    """Required §5.3 evidence: every selected line row present and OK for INIT and
    k=0..5, every registered window (both runs, all k, all shells) defined, and
    both emission integrals finite. Missing evidence is a G-RUN failure (-> X)."""
    reasons = []
    inc = 0
    for rec in lines:
        if rec.get("status") == "EMPTY":
            continue
        if rec.get("init_status") != "OK_INIT_GAUSSIAN_PROFILE_MEAN" or \
                any(f"k{k}" not in rec or "Jbar" not in rec[f"k{k}"] for k in range(N_ITER)):
            inc += 1
    if not lines:
        reasons.append("LINES_EMPTY")
    elif inc:
        reasons.append(f"LINES_INCOMPLETE:{inc}")
    wbad = 0
    for nm in ("W5", "C1"):
        for k in range(N_ITER):
            for s in SHELLS:
                wr = windows.get(f"{nm}/k{k}/sh{s}")
                if wr is None:
                    wbad += len(WINDOWS)
                    continue
                for name, _, _ in WINDOWS:
                    st = wr.get(name, {}).get("status")
                    if st not in ("OK", "LOG_ONLY", "ZERO_J"):
                        wbad += 1
    if wbad:
        reasons.append(f"WINDOWS_UNDEFINED:{wbad}")
    for nm in ("W5", "C1"):
        e = em.get(nm, {})
        if e.get("status") != "OK" or not isinstance(e.get("L_grid"), float) or \
                not math.isfinite(e["L_grid"]) or e["L_grid"] == 0.0:
            reasons.append(f"EMISSION_{nm}:{e.get('status')}")
    return reasons


# ----------------------------------------------------------------- lines
def read_w5_lines(run, want):
    """rows for the manifest (line_id, shell) pairs; per-call row counts for G-RUN."""
    f = Path(run) / "work" / "w5_lines.csv"
    if not f.exists():
        return {}, {"status": "MISSING"}
    got, per_call, per_call_out = {}, {}, {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            c = int(r["call"])
            per_call[c] = per_call.get(c, 0) + 1
            if r["status"] == "OUTSIDE_FINE_SUPPORT":
                per_call_out[c] = per_call_out.get(c, 0) + 1
            key = (int(r["line_id"]), int(r["shell"]))
            if key in want:
                got.setdefault(key, []).append(r)
    counts = sorted(set(per_call.values()))
    st = "OK" if set(per_call) == set(range(1, N_ITER + 2)) and len(counts) == 1 \
        and counts[0] > 0 else "INCONSISTENT"
    return got, {"status": st, "calls": sorted(per_call), "rows_per_call": counts,
                 "outside_rows_per_call": sorted(set(per_call_out.values())),
                 "rows_total": sum(per_call.values())}


def line_table(got, manifest):
    out = []
    for m in manifest:
        if m["line_id"] == "EMPTY":
            out.append({**m, "status": "EMPTY"})
            continue
        key = (int(m["line_id"]), int(m["shell"]))
        rows = got.get(key, [])
        byk = {}
        init = None
        for r in rows:
            g = int(r["rad_generation"])
            if r["operator"] == "INIT_SHARED_GAUSSIAN":
                init = r
            else:
                byk[g - 2] = r
        rec = {**m, "C1_line_field": "BASE_LINE_FIELD_UNAVAILABLE",
               "init_Jbar_gaussian": init["Jbar"] if init else None,
               "init_status": init["status"] if init else "ABSENT"}
        for k in range(N_ITER):
            r = byk.get(k)
            if r is None or r["status"] != "OK":
                rec[f"k{k}"] = {"status": r["status"] if r else "ABSENT"}
                continue
            jc, beta, L, jb = (fnum(r[c]) for c in ("J_c", "beta", "L_l", "Jbar"))
            cont = beta * jc
            a_l = cont / jb if jb != 0.0 else None
            rec[f"k{k}"] = {"J_c": jc, "beta": beta, "L_l": L, "Jbar": jb,
                            "a_l": a_l if a_l is not None else "UNDEFINED_JBAR_ZERO",
                            "a_l_in_0_1": (a_l is not None and 0.0 <= a_l <= 1.0),
                            "tau_eff": fnum(r["tau_eff"]), "srce_chk": r["srce_chk"]}
        for (k0, k1) in ((0, 5), (3, 5)):
            a0, a1 = rec.get(f"k{k0}", {}), rec.get(f"k{k1}", {})
            if "Jbar" in a0 and "Jbar" in a1 and a0["Jbar"] > 0 and a1["Jbar"] > 0 and \
                    a0["J_c"] > 0 and a1["J_c"] > 0:
                rec[f"d_k{k0}_k{k1}"] = {
                    "log10_Jbar_ratio": math.log10(a1["Jbar"] / a0["Jbar"]),
                    "log10_Jc_ratio": math.log10(a1["J_c"] / a0["J_c"]),
                    "beta_ratio": a1["beta"] / a0["beta"],
                    "L_l_ratio": (a1["L_l"] / a0["L_l"]) if a0["L_l"] > 0 else None}
            else:
                rec[f"d_k{k0}_k{k1}"] = "UNDEFINED"
        out.append(rec)
    return out


def read_manifest(ref):
    rows = list(csv.DictReader(open(Path(ref) / "line_manifest.csv")))
    return rows


# ----------------------------------------------------------------- readout
def population_tables(pw, pc, sw, sc, pub):
    tab = []
    for Z in ELEMS:
        for s in SHELLS:
            xp = pub.get((Z, s), (None, None, None, "UNDEFINED"))
            xpub = xp[2] / xp[1] if xp[3] == "OK" else None
            for A in ("stage", "levels"):
                for k in range(N_ITER):
                    xw, stw = xval(pw, k, s, Z, A)
                    xc, stc = xval(pc, k, s, Z, A)
                    row = {"k": k, "shell": s, "Z": Z, "authority": A,
                           "x_W5": xw if stw == "OK" else stw,
                           "x_C1": xc if stc == "OK" else stc, "x_pub": xpub}
                    for nm, x, st in (("W5", xw, stw), ("C1", xc, stc)):
                        if st == "OK" and xpub:
                            row[f"x_over_pub_{nm}"] = x / xpub
                            row[f"log10_x_over_pub_{nm}"] = math.log10(x / xpub)
                    if stw == "OK" and stc == "OK":
                        row["log10_W5_minus_C1"] = math.log10(xw) - math.log10(xc)
                    tab.append(row)
    # stage - levels and trajectory changes per run
    extra = []
    for nm, P in (("W5", pw), ("C1", pc)):
        for Z in ELEMS:
            for s in SHELLS:
                for k in range(N_ITER):
                    xs, a = xval(P, k, s, Z, "stage")
                    xl, b = xval(P, k, s, Z, "levels")
                    e = {"run": nm, "Z": Z, "shell": s, "k": k,
                         "log10_stage_minus_levels": (math.log10(xs) - math.log10(xl))
                         if a == b == "OK" else "UNDEFINED"}
                    for A in ("stage", "levels"):
                        if k in (4, 5):
                            x0, s0 = xval(P, k - 1, s, Z, A)
                            x1, s1 = xval(P, k, s, Z, A)
                            e[f"log10_change_k{k-1}_k{k}_{A}"] = (
                                math.log10(x1) - math.log10(x0)) if s0 == s1 == "OK" else "UNDEFINED"
                    extra.append(e)
    return tab, extra


def outer_table(pw, sw, log):
    out = []
    for s in (19, 24, 30):
        for k in range(N_ITER):
            row = {"shell": s, "k": k,
                   "n_e": fnum(sw.get(k, {}).get(s, {}).get("n_e_cm3"))}
            for Z in ELEMS:
                for A in ("stage", "levels"):
                    x, st = xval(pw, k, s, Z, A)
                    row[f"Z{Z}_{A}_x"] = x if st == "OK" else st
            out.append(row)
    return {"shells": out, "bin_conv_last": log["conv"][-1] if log["conv"] else None,
            "bin_conv": log["conv"], "fine_solve": log["fine_status"],
            "note": "fixed iteration count: finite-iteration response, not a converged solution"}


def cmd_readout(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pw, sw = load_pops(a.run)
    pc, sc = load_pops(a.c1)
    pub = public_x(sc.get(N_ITER - 1, {}))
    d, bad = deltas(pw, pc)
    stat_branch, m, M, why = branch_from_deltas(d)
    reasons = list(why) + ([f"UNDEFINED_LOG:{len(bad)}"] if bad else [])
    tab, extra = population_tables(pw, pc, sw, sc, pub)
    log = parse_log(a.run)
    manifest = read_manifest(a.referents)
    want = {(int(r["line_id"]), int(r["shell"])) for r in manifest if r["line_id"] != "EMPTY"}
    got, lsum = read_w5_lines(a.run, want)
    lines = line_table(got, manifest) if lsum.get("status") != "MISSING" else []
    # levels-final identity vs k=5 level sums
    lf = {}
    fpath = Path(a.run) / "work" / "nlte_levels_final.csv"
    if fpath.exists():
        with open(fpath) as fh:
            for r in csv.DictReader(fh):
                Z, q, s = int(r["Z"]), int(r["ion"]), int(r["shell"])
                if Z in ELEMS and q in (1, 2) and s in SHELLS:
                    k_ = (Z, q, s)
                    lf.setdefault(k_, [0.0, set()])
                    lf[k_][0] += float(r["n_pop"])
                    lf[k_][1].add(r["population_generation"])
    lvl_check = []
    for (Z, q, s), (tot, gens) in sorted(lf.items()):
        row5 = pw.get(N_ITER - 1, {}).get(s, {})
        ref = fnum(row5.get(f"Z{Z}_nlte_q{q}_cm3"))
        lvl_check.append({"Z": Z, "ion": q, "shell": s, "levels_final_sum": tot,
                          "ion_csv_k5": ref,
                          "rel_diff": (tot - ref) / ref if ref else None,
                          "generation_levels_final": sorted(gens),
                          "generation_ion_csv_k5": row5.get("population_generation")})
    windows = {}
    for k in range(N_ITER):
        for nm, run, P in (("W5", a.run, sw), ("C1", a.c1, sc)):
            try:
                spec = load_spectral(run, k, SHELLS)
            except FileNotFoundError:
                continue
            for s in SHELLS:
                T = fnum(P.get(k, {}).get(s, {}).get("T_e_K"))
                if spec.get(s) and math.isfinite(T) and T > 0:
                    windows[f"{nm}/k{k}/sh{s}"] = window_ratios(spec[s], T)
    em = {}
    for nm, run in (("W5", a.run), ("C1", a.c1)):
        try:
            em[nm] = emission(run)
        except FileNotFoundError:
            em[nm] = {"status": "MISSING"}
    gates = {}
    if a.gates:
        gates = json.loads(Path(a.gates).read_text())
    grun = [] if a.stats_only else g_run(a.run, log, lsum) + evidence_reasons(lines, windows, em)
    full = stat_branch
    freasons = list(reasons)
    if not a.stats_only:
        for gname in ("G-ID", "G-SELF"):
            if gates.get(gname) != "PASS":
                freasons.append(f"{gname}:{gates.get(gname, 'GATE_EVIDENCE_NOT_SUPPLIED')}")
        if grun:
            freasons.append("G-RUN:" + ";".join(grun))
        if freasons:
            full = "W5-X"
    res = {"schema": "W5_READOUT_V1", "run": str(a.run), "c1": str(a.c1),
           "referents": str(a.referents), "stats_only": a.stats_only,
           "statistic_branch": stat_branch, "statistic_branch_code": BR_CODE[stat_branch],
           "branch": full, "branch_code": BR_CODE[full], "reasons": freasons,
           "M": M if math.isfinite(M) else None, "threshold_dex": THRESH,
           "m": {f"Z{Z}/{A}": v for (Z, A), v in m.items()},
           "delta": {f"Z{Z}/{A}/sh{s}": v for (Z, A, s), v in d.items()},
           "delta_signs": {f"Z{Z}/{A}": "".join("+" if d[(Z, A, s)] > 0 else
                                                ("-" if d[(Z, A, s)] < 0 else "0")
                                                for s in SHELLS if math.isfinite(d[(Z, A, s)]))
                           for Z in ELEMS for A in ("stage", "levels")},
           "undefined_logs": bad, "g_run_reasons": grun, "gates_supplied": gates,
           "w5_lines_summary": lsum, "levels_final_check": lvl_check,
           "emission": em, "line_C1_column": "BASE_LINE_FIELD_UNAVAILABLE",
           "claim_limits": "MOVE is not improvement; lines are W5-internal trajectories"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "population.json").write_text(json.dumps({"table": tab, "per_run": extra},
                                                    indent=1, default=str))
    (out / "windows.json").write_text(json.dumps(
        {"rule": "R=sum J dnu_overlap / sum B(nu_b,T_e) dnu_overlap; nu_b=sqrt(nu_lo nu_hi); "
                 "overlap in Hz with window edges c/lambda", "windows": windows}, indent=1))
    (out / "lines.json").write_text(json.dumps(lines, indent=1, default=str))
    (out / "outer.json").write_text(json.dumps(outer_table(pw, sw, log), indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("statistic_branch", "branch", "branch_code", "M",
                                          "m", "reasons")}, default=str))
    return 0


# ----------------------------------------------------------------- selftest
def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []

    def case(name, got, want):
        ok = got == want
        res.append((name, ok, got, want))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got} registered {want}")

    def mkd(v):
        return {(Z, A, s): v[i] if isinstance(v, list) else v
                for Z in ELEMS for A in ("stage", "levels") for i, s in enumerate(SHELLS)}

    # branch statistic boundaries on exact delta values
    case("exact boundary 0.30 -> MOVE", branch_from_deltas(mkd(0.30))[0], "W5-MOVE")
    case("just below 0.2999999 -> NONE", branch_from_deltas(mkd(0.2999999))[0], "W5-NONE")
    case("negative direction -0.35 -> MOVE", branch_from_deltas(mkd(-0.35))[0], "W5-MOVE")
    d = mkd(0.0)
    for i, s in enumerate(SHELLS):
        d[(16, "levels", s)] = [0.4, -0.5, 0.31, 0.0, 0.1][i]
    case("S levels majority |delta|>=0.3 mixed sign -> MOVE", branch_from_deltas(d)[0], "W5-MOVE")
    d = mkd(0.0)
    for i, s in enumerate(SHELLS):
        d[(20, "stage", s)] = [2.0, 2.0, 0.0, 0.0, 0.0][i]
    case("minority shells only (2/5) -> NONE", branch_from_deltas(d)[0], "W5-NONE")
    d = mkd(0.1)
    d[(14, "stage", 19)] = float("nan")
    case("NaN delta -> X", branch_from_deltas(d)[0], "W5-X")
    d = mkd(0.1)
    del d[(16, "stage", 24)]
    case("missing delta -> X", branch_from_deltas(d)[0], "W5-X")

    # line selection rules on a small synthetic list
    levels = {(16, 1, l) for l in range(6)} | {(14, 2, l) for l in range(3)}
    pops = {(16, 1, s, l): 10.0 ** (-l) for s in SHELLS for l in range(6)}
    pops.update({(14, 2, s, l): 1.0 for s in SHELLS for l in range(3)})
    lines = [(0, 16, 1, 0, 1, 600.0, 0.5), (1, 16, 1, 0, 2, 620.0, 0.5),   # tie -> id order
             (2, 16, 1, 1, 3, 700.0, 5.0), (3, 16, 1, 2, 4, 710.0, 90.0),
             (4, 16, 1, 0, 5, 720.0, 0.0),                                  # f_lu=0
             (5, 16, 1, 0, 9, 730.0, 1.0),                                  # missing upper
             (6, 16, 1, 3, 5, 740.0, 1.0), (7, 16, 1, 0, 3, 750.0, 0.01),
             (8, 14, 2, 0, 1, 1000.0, 0.2), (9, 20, 1, 0, 1, 1000.0, 1.0)]   # unmapped ion
    rows, st = select_lines(lines, levels, pops, {(16, 1), (14, 2)})
    sel = [r["line_id"] for r in rows if r["shell"] == 19 and r["Z"] == 16 and r["ion"] == 1
           and r["bundle"] == 1]
    case("top four by n_l f_lu (0.9 first) with the 0.5 three-way tie in index order", sel, [3, 0, 1, 2])
    case("ineligible counted (f_lu=0, missing endpoint)",
         (st["ineligible_f"], st["ineligible_endpoint"]), (1, 1))
    emp = [r for r in rows if r["shell"] == 19 and r["Z"] == 16 and r["ion"] == 1 and r["bundle"] == 0]
    case("empty bundle -> EMPTY", [r["line_id"] for r in emp], ["EMPTY"])
    case("unmapped ion never selected", any(r["Z"] == 20 for r in rows), False)
    one = [r["line_id"] for r in rows if r["shell"] == 10 and r["Z"] == 14 and r["bundle"] == 2]
    case("bundle [912,2000) holds 1000 A line", one, [8])

    # window integral
    T = 8000.0
    spec = []
    nu = C_CGS / (400e-8)
    while nu > C_CGS / (2100e-8):
        lo = nu * math.exp(-0.01)
        mid = math.sqrt(lo * nu)
        spec.append((lo, nu, math.exp(log_planck(mid, T))))
        nu = lo
    w = window_ratios(spec, T)
    case("J=B -> R=1 in 912-1500", round(w["912-1500"]["R"], 12), 1.0)
    spec0 = [(lo, hi, 0.0) for lo, hi, _ in spec]
    case("J=0 -> exact zero", window_ratios(spec0, T)["531-760"]["status"], "ZERO_J")
    wc = window_ratios([(lo, hi, 1e-30) for lo, hi, _ in spec], 300.0)
    case("cold Wien tail keeps a finite log (no B=0 truncation)",
         wc["blue_end-531"]["status"] in ("OK", "LOG_ONLY") and
         wc["blue_end-531"]["log10_R"] is not None and wc["blue_end-531"]["log10_R"] > 100, True)

    # G-RUN evidence completeness (sol audit Q4): each missing piece must force X
    def full_lines():
        r = {"line_id": "1", "shell": "10", "init_status": "OK_INIT_GAUSSIAN_PROFILE_MEAN"}
        for k in range(N_ITER):
            r[f"k{k}"] = {"J_c": 1.0, "beta": 0.5, "L_l": 0.5, "Jbar": 1.0}
        return [r, {"line_id": "EMPTY", "status": "EMPTY"}]

    def full_windows():
        return {f"{nm}/k{k}/sh{s}": {w[0]: {"status": "OK"} for w in WINDOWS}
                for nm in ("W5", "C1") for k in range(N_ITER) for s in SHELLS}
    full_em = {"W5": {"status": "OK", "L_grid": 1.0e42}, "C1": {"status": "OK", "L_grid": 1.1e42}}
    case("evidence complete -> no G-RUN reason",
         evidence_reasons(full_lines(), full_windows(), full_em), [])
    ln = full_lines()
    del ln[0]["k3"]
    case("selected line missing k=3 row -> LINES_INCOMPLETE",
         evidence_reasons(ln, full_windows(), full_em), ["LINES_INCOMPLETE:1"])
    ln = full_lines()
    ln[0]["init_status"] = "ABSENT"
    case("selected line missing INIT row -> LINES_INCOMPLETE",
         evidence_reasons(ln, full_windows(), full_em), ["LINES_INCOMPLETE:1"])
    case("no line table -> LINES_EMPTY",
         evidence_reasons([], full_windows(), full_em), ["LINES_EMPTY"])
    wd = full_windows()
    del wd["W5/k5/sh24"]
    case("missing window cell -> WINDOWS_UNDEFINED",
         evidence_reasons(full_lines(), wd, full_em), [f"WINDOWS_UNDEFINED:{len(WINDOWS)}"])
    wd = full_windows()
    wd["C1/k4/sh19"]["531-760"]["status"] = "NONFINITE_J"
    case("nonfinite window -> WINDOWS_UNDEFINED",
         evidence_reasons(full_lines(), wd, full_em), ["WINDOWS_UNDEFINED:1"])
    case("nonfinite emission -> EMISSION_W5",
         evidence_reasons(full_lines(), full_windows(),
                          {"W5": {"status": "NONFINITE"}, "C1": full_em["C1"]}),
         ["EMISSION_W5:NONFINITE"])
    nolog = {"present": True, "bin_tags": N_ITER + 1, "blocked": [], "conv": [],
             "fine_tags": [{"operator": "INIT_SHARED_GAUSSIAN", "interp_cells": "1",
                            "source_cells": "1", "fine_cells": "1"}] +
                          [{"operator": "CMFGEN_NONOVERLAP_SOBOLEV", "interp_cells": "1",
                            "source_cells": "1", "fine_cells": "1"}] * N_ITER,
             "fine_status": []}
    case("zero exact-solve tags -> FINE_SOLVE_STATUS:0",
         [r for r in g_run(out, nolog, {"status": "OK"}) if r.startswith("FINE_SOLVE")],
         ["FINE_SOLVE_STATUS:0"])
    nolog["fine_status"] = [{"status": "OK"}] * N_ITER
    case("six of seven exact-solve tags -> FINE_SOLVE_STATUS:6",
         [r for r in g_run(out, nolog, {"status": "OK"}) if r.startswith("FINE_SOLVE")],
         ["FINE_SOLVE_STATUS:6"])

    hdr = ("call,operator,rad_generation,material_generation,shell,line_id,Z,ion,nu_hz,J_c,"
           "J_c_err_upper,beta,L_l,Jbar,Jbar_err_upper,tau_eff,srce_chk,status\n")
    for tag, calls, want in (("seven calls", range(1, N_ITER + 2), "OK"),
                             ("call 7 missing", range(1, N_ITER + 1), "INCONSISTENT"),
                             ("call 1 missing", range(2, N_ITER + 2), "INCONSISTENT")):
        fake = out / f"fake_{tag.replace(' ', '_')}"
        (fake / "work").mkdir(parents=True, exist_ok=True)
        with open(fake / "work" / "w5_lines.csv", "w") as fh:
            fh.write(hdr)
            for c in calls:
                fh.write(f"{c},X,{c + 1},{c + 1},10,1,16,1,1e15,1,0,1,0,1,0,1,0,OK\n")
        case(f"w5_lines ledger {tag} -> {want}", read_w5_lines(fake, set())[1]["status"], want)

    # end-to-end on the sealed C1 populations
    if a.c1:
        pc, _ = load_pops(a.c1)
        d, bad = deltas(pc, pc)
        b, m, M, _ = branch_from_deltas(d)
        case("C1 vs C1 -> NONE (M=0)", (b, M), ("W5-NONE", 0.0))

        def scaled(P, f, Z=16, A="stage"):
            Q = {k: {s: dict(r) for s, r in v.items()} for k, v in P.items()}
            tag = "stage" if A == "stage" else "nlte"
            for k in Q:
                for s in SHELLS:
                    r = Q[k][s]
                    r[f"Z{Z}_{tag}_q2_cm3"] = repr(fnum(r[f"Z{Z}_{tag}_q2_cm3"]) * f)
            return Q
        for f, want in ((10 ** 0.35, "W5-MOVE"), (10 ** 0.25, "W5-NONE"),
                        (10 ** -0.35, "W5-MOVE"), (10 ** -0.25, "W5-NONE")):
            d, _ = deltas(scaled(pc, f), pc)
            case(f"C1 x S stage scaled by 10^{math.log10(f):+.2f} -> {want}",
                 branch_from_deltas(d)[0], want)
        Q = {k: {s: dict(r) for s, r in v.items()} for k, v in pc.items()}
        del Q[5][19]
        case("C1 with a missing k=5 shell-19 row -> X", branch_from_deltas(deltas(Q, pc)[0])[0], "W5-X")
        Q = {k: {s: dict(r) for s, r in v.items()} for k, v in pc.items()}
        Q[4][24]["Z14_nlte_q1_cm3"] = "nan"
        case("C1 with NaN population -> X", branch_from_deltas(deltas(Q, pc)[0])[0], "W5-X")
        Q = {k: {s: dict(r) for s, r in v.items()} for k, v in pc.items()}
        Q[5][10]["Z20_stage_q2_cm3"] = "0"
        case("C1 with an exact-zero population -> X (no small positive)",
             branch_from_deltas(deltas(Q, pc)[0])[0], "W5-X")
    nfail = sum(1 for r in res if not r[1])
    (out / "readout_selftest.json").write_text(json.dumps(
        [{"case": n, "pass": ok, "got": str(g), "registered": str(w)} for n, ok, g, w in res],
        indent=1))
    print(f"\nNC-READOUT: {len(res) - nfail}/{len(res)} classified as registered -> "
          f"{'PASS' if nfail == 0 else 'FAIL'}")
    return 0 if nfail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("referents")
    r.add_argument("--c1", required=True)
    r.add_argument("--out", required=True)
    r = sp.add_parser("readout")
    r.add_argument("--run", required=True)
    r.add_argument("--c1", required=True)
    r.add_argument("--referents", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gates")
    r.add_argument("--stats-only", action="store_true")
    r = sp.add_parser("selftest")
    r.add_argument("--c1")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"referents": cmd_referents, "readout": cmd_readout,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
