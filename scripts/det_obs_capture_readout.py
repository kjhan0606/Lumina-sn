#!/usr/bin/env python3
"""DET-OBS-CAPTURE readout (prereg docs/PREREG_DET_OBS_CAPTURE_2026-10-06.md §4-§9).

Subcommands
  selftest --c-battery DIR --out DIR      reader / bound / branch negative controls
  preflight --baseline-e E0 --baseline-r R0 --source T2 --build B2 --run-e EC --run-r RC
            --selftest JSON --out DIR      G-ID before submission
  replay   --run X --baseline X0 --slurm-state STR --selftest JSON --out DIR
           committed capture -> reader checks, C replay (scalar bytes + component
           closure), band luminosities with rounding bounds
  paired   --e DIR --r DIR --out DIR       §7 shares, E'-R' differences, branches

Every command writes only into a new --out directory.  The capture reader never
fills a missing value from other files and never applies a tolerance to the
byte gates.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_obs_readout import (sha, footer, footer_core_reasons, resource_reasons,   # noqa: E402
                             runner_rev5, submit_expected, read_spectrum)
from nrb_e_readout import band_integral                                             # noqa: E402

P_ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
W = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002")
C = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006")
T1, B1, E0, R0 = W / "source", W / "cuda_build2", W / "Eprime/run1", W / "Rprime/run2"
T2, B2, EC, RC = C / "source", C / "cuda_build1", C / "Eprime/run1", C / "Rprime/run1"
R_RUNNER = Path("/gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/run1/job.slurm")
REPLAY_BIN = C / "stage1_fix2r/bin/replay"
PAIR_CAP = 2_000_000_000
STATE_CAP = PAIR_CAP // 2          # each state <= half: the pair never exceeds the cap
MANIFEST_MAX = 32768
RUN_FILES = ("input/run_build_identity.json", "input/resolved_lumina.exports", "job.slurm", "STAGE_IDENTITY.txt")
CHANGESET = ("src/lumina_cmfgen.c", "src/env_universe.h", "tests/det_obs_capture_selftest.c",
             "tools/det_obs_capture_replay.c")
PATCH, PATCH_SHA = C / "evidence/det_obs_capture.patch", C / "evidence/det_obs_capture.patch.sha256"
CAPTURE_KEY = "LUMINA_DET_OBS_CAPTURE"
JOB = {"E": "LUMINA_detobs_Ec1", "R": "LUMINA_detobs_Rc1"}
EXPECT = {"material": 7, "radiation": 7, "iter": 5, "asm_serial": 7, "mode": 1, "src_nlte": 1,
          "NS": 50, "NB": 1234, "NL": 2588798}
FILES = ("lines.bin", "chi_a.f64", "chi_e.f64", "eta_c.f64", "eta_dep.f64", "J.f64", "Te.f64",
         "ne.f64", "nu.f64", "dnu.f64", "line_nu.f64", "r_inner.f64", "r_outer.f64", "meta.txt")
LANES = ("mapped_line", "unmapped_thermal", "unmapped_scatter", "bf_ff", "electron_scatter",
         "inner_boundary", "deposition")
BANDS = {"FUVb": (1500.0, 2000.0), "UV2500": (2500.0, 3000.0), "OPT": (4000.0, 6000.0)}
N_FILE, L_FILE = "work/lumina_spectrum_det_obs.csv", "work/lumina_spectrum_det_obs_legacy.csv"
WRITER_SAME = ("rad_generation", "material_generation", "iter", "asm_serial", "mode", "src_nlte",
               "shells", "bins", "lines", "capture_fnv1a", "view_fnv1a", "census",
               "reconcile_max_rel", "jumps", "nsub_capped_segments", "chi_e_bound",
               "chi_e_subtraction_max_dev")

U = 2.0 ** -53
AMIN = 2.0 ** -1074


def up(x):
    return math.nextafter(x, math.inf)


def rho(x):
    """outward rounding bound of one binary64 operation whose result is x."""
    return up(up(up(U * abs(x)) / (1.0 - U)) + AMIN)


def add_up(*xs):
    s = 0.0
    for x in xs:
        s = up(s + x)
    return s


def mul_up(a, b):
    return up(a * b)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------ band integrals with bounds
def band_with_bound(pts, lo, hi):
    """band_integral's exact operation order (nt_eta_readout) and the outward
    rounding bound of its evaluation against the linear functional with the
    shared fp wavelength differences.  Returns (value, bound, status)."""
    if not pts or pts[0][0] > lo or pts[-1][0] < hi:
        return None, None, "UNSUPPORTED"
    tot, err = 0.0, 0.0
    for (l0, f0), (l1, f1) in zip(pts[:-1], pts[1:]):
        if l1 <= l0:
            return None, None, "NONMONOTONE"
        a, b = max(l0, lo), min(l1, hi)
        if b <= a:
            continue
        beta = l1 - l0
        res = []
        for edge in (a, b):
            alpha = edge - l0
            d = f1 - f0
            m = d * alpha
            q = m / beta
            fx = f0 + q
            e = add_up(mul_up(rho(d), up(abs(alpha) / beta)), up(rho(m) / beta), rho(q), rho(fx))
            res.append((fx, e))
        (fa, ea), (fb, eb) = res
        gamma = b - a
        s = fa + fb
        h = 0.5 * s
        t = h * gamma
        tot2 = tot + t
        # value check against the shared expression in band_integral
        e_t = add_up(mul_up(0.5 * abs(gamma), add_up(ea, eb, rho(s))), mul_up(abs(gamma), rho(h)), rho(t))
        err = add_up(err, e_t, rho(tot2))
        tot = tot2
    return tot, err, "OK"


def band_of_bound(pts_e, lo, hi):
    """upper bound of the band integral of a non-negative pointwise bound."""
    if not pts_e or pts_e[0][0] > lo or pts_e[-1][0] < hi:
        return None
    tot = 0.0
    for (l0, e0), (l1, e1) in zip(pts_e[:-1], pts_e[1:]):
        a, b = max(l0, lo), min(l1, hi)
        if b <= a:
            continue
        tot = up(tot + mul_up(up(0.5 * up(max(e0, e1) + max(e0, e1))), up(b - a)))
    return tot


def read_components(path):
    rows = list(csv.reader(open(path)))
    head = rows[0]
    want = ["wavelength_angstrom", "scalar", *LANES, "bound", "closure_ok"]
    if head != want:
        raise ValueError("COMPONENTS_HEADER")
    data = [[float(x) for x in r[:-1]] + [int(r[-1])] for r in rows[1:]]
    data.sort(key=lambda r: r[0])
    return data


def component_bands(data, ranges):
    """per band: lane values and bounds, scalar value and bound, closure."""
    lam = [r[0] for r in data]
    out = {}
    for name, (lo, hi) in ranges.items():
        sc_pts = [(l, r[1]) for l, r in zip(lam, data)]
        vs, es, st = band_with_bound(sc_pts, lo, hi)
        if st != "OK":
            out[name] = {"status": st}
            continue
        ref, _ = band_integral(sc_pts, lo, hi)
        lanes = {}
        for j, ln in enumerate(LANES):
            v, e, _ = band_with_bound([(l, r[2 + j]) for l, r in zip(lam, data)], lo, hi)
            lanes[ln] = {"L": v, "err": e}
        s, srnd = 0.0, 0.0
        for ln in LANES:
            s = s + lanes[ln]["L"]
            srnd = add_up(srnd, rho(s))
        pointwise = band_of_bound([(l, r[9]) for l, r in zip(lam, data)], lo, hi)
        lim = add_up(pointwise, es, srnd, *[lanes[ln]["err"] for ln in LANES])
        d = abs(s - vs)
        fin = all(math.isfinite(x) for x in (pointwise, es, srnd, lim, d, s, vs)) and \
            all(math.isfinite(lanes[ln]["L"]) and math.isfinite(lanes[ln]["err"]) for ln in LANES)
        out[name] = {"status": "OK", "scalar": vs, "scalar_err": es, "scalar_equals_band_integral": vs == ref,
                     "lanes": lanes, "sum": s, "sum_err": add_up(srnd, *[lanes[ln]["err"] for ln in LANES]),
                     "closure_diff": d, "closure_bound": lim, "bound_finite": fin,
                     "closure_ok": fin and d <= add_up(lim, rho(d))}
    return out


# ------------------------------------------------------------ capture reader
def read_meta(path):
    meta = {}
    for ln in open(path):
        if "=" in ln:
            k, v = ln.rstrip("\n").split("=", 1)
            meta[k] = v
    return meta


REC = np.dtype([("l", "<u4"), ("s", "<u4"), ("tau", "<f8"), ("S", "<f8"), ("eps", "<f8"), ("m", "u1")])


def records_reasons(lp, mi, meta):
    """sparse records: order, consumed predicate, the writer's value domain, counts."""
    why = []
    size = lp.stat().st_size
    if size % 33 or size // 33 != mi["n_pairs"]:
        return ["CAPTURE_PAIRS_COUNT"]
    r = np.fromfile(lp, dtype=REC)
    key = r["l"].astype(np.int64) * mi["NS"] + r["s"].astype(np.int64)
    if r.size and (np.any(r["l"] >= mi["NL"]) or np.any(r["s"] >= mi["NS"]) or np.any(np.diff(key) <= 0)
                   or np.any(r["m"] > 1)):
        why.append("CAPTURE_PAIRS_ORDER_OR_DOMAIN")
        return why
    tau, S, eps, m = r["tau"], r["S"], r["eps"], r["m"]
    try:
        nu_min, nu_max = float.fromhex(meta["nu_min"]), float.fromhex(meta["nu_max"])
        line_nu = np.fromfile(lp.parent / "line_nu.f64", dtype="<f8")
    except (KeyError, ValueError, OSError):
        return why + ["CAPTURE_META:grid"]
    nl = line_nu[r["l"]] if r.size else line_nu[:0]
    bad = ~(np.isfinite(tau) & (tau > 1e-12)) | ~((nl > nu_min) & (nl < nu_max)) | \
        ~(np.isfinite(S) & (S > 0.0)) | ~np.isfinite(eps) | \
        np.where(m == 1, eps != 0.0, (eps < 0.0) | (eps > 1.0))
    if np.any(bad):
        why.append(f"CAPTURE_RECORD_DOMAIN:n={int(bad.sum())}")
    if int(m.sum()) != mi["n_mapped"]:
        why.append("CAPTURE_PAIRS_MAPPED")
    return why


def capture_reasons(capdir, writer_manifest, run_root=None, expect=EXPECT, b2=None):
    """reader acceptance of a committed capture (prereg §4.1, §4.4-§4.5)."""
    why = []
    capdir = Path(capdir)
    mp = capdir / "manifest.json"
    if not mp.exists():
        return ["CAPTURE_NOT_COMMITTED"]
    if capdir.name != "det_obs_capture":
        return ["CAPTURE_NOT_FINAL_NAME"]   # the manifest is finished before the commit rename
    try:
        m = json.load(open(mp))
        meta = read_meta(capdir / "meta.txt")
    except (OSError, ValueError):
        return ["CAPTURE_MANIFEST_UNREADABLE"]
    if m.get("schema") != "DET_OBS_CAPTURE_V1" or m.get("transaction") != "COMMITTED":
        why.append("CAPTURE_SCHEMA_OR_TRANSACTION")
    files = m.get("files", {})
    for f in FILES:
        rec = files.get(f)
        p = capdir / f
        if rec is None or not p.exists():
            why.append(f"CAPTURE_FILE_MISSING:{f}")
            continue
        if p.stat().st_size != rec.get("bytes") or sha256_file(p) != rec.get("sha256"):
            why.append(f"CAPTURE_FILE_IDENTITY:{f}")
    g = m.get("generations", {})
    for k, key in (("material", "material"), ("radiation", "radiation"), ("iter", "iter"),
                   ("material", "nlte_population_committed"), ("material", "capture_material")):
        if g.get(key) != expect[k]:
            why.append(f"CAPTURE_GENERATION:{key}")
    try:
        mi = {k: int(meta[k]) for k in ("NS", "NB", "NL", "mode", "src_nlte", "n_pairs", "n_mapped",
                                         "n_unmapped", "asm_serial", "material_generation",
                                         "rad_generation", "iter", "tau_required_generation",
                                         "tau_computed_generation")}
    except (KeyError, ValueError):
        return why + ["CAPTURE_META"]
    # manifest identity == expected == meta (two records of the same capture)
    ident = m.get("identity", {})
    for k in ("mode", "src_nlte", "asm_serial", "NS", "NB", "NL"):
        if ident.get(k) != expect[k] or ident.get(k) != mi[k]:
            why.append(f"CAPTURE_IDENTITY:{k}")
    if g.get("iter") != mi["iter"] or g.get("tau_required") != mi["tau_required_generation"] or \
            g.get("tau_computed") != mi["tau_computed_generation"]:
        why.append("CAPTURE_IDENTITY:meta_generations")
    tr = ident.get("transport", {})
    for mk, tk in (("T_inner", "T_inner"), ("inner_bb_scale", "inner_bb_scale"), ("inv_ct", "inv_ct"),
                   ("time_explosion", "time_explosion"), ("inner_bb_scale_env", "inner_bb_scale_env")):
        if meta.get(mk) != tr.get(tk):
            why.append(f"CAPTURE_IDENTITY:transport:{tk}")
    if tr.get("rays") != 256 or tr.get("sub_step_km_s") != 30:
        why.append("CAPTURE_IDENTITY:transport:grid")
    if ident.get("callsite") != "cmfgen_run R6 final pass -> det_obs_final_pass_hook -> det_obs_write":
        why.append("CAPTURE_IDENTITY:callsite")
    tot = sum((files.get(f) or {}).get("bytes", 0) for f in FILES)
    msize = mp.stat().st_size
    if m.get("committed_payload_bytes") != tot or m.get("state_cap_bytes") != STATE_CAP or \
            m.get("pair_cap_bytes") != PAIR_CAP or msize > MANIFEST_MAX or tot + msize > STATE_CAP:
        why.append("CAPTURE_PAYLOAD_BYTES")
    if b2 is not None:
        try:
            if ident.get("binary_sha256") != (Path(b2) / "binary.sha256").read_text().split()[0]:
                why.append("CAPTURE_IDENTITY:binary")
        except OSError:
            why.append("CAPTURE_IDENTITY:binary_record")
    for k in ("NS", "NB", "NL", "mode", "src_nlte", "asm_serial"):
        if mi[k] != expect[k]:
            why.append(f"CAPTURE_META:{k}")
    if mi["material_generation"] != expect["material"] or mi["rad_generation"] != expect["radiation"]:
        why.append("CAPTURE_META:generation")
    cen = m.get("census", {})
    wcen = (writer_manifest or {}).get("census", {})
    if cen.get("consumed") != mi["n_pairs"] or cen.get("mapped") != mi["n_mapped"] or \
            cen.get("unmapped") != mi["n_unmapped"]:
        why.append("CAPTURE_CENSUS_META")
    if writer_manifest is not None and (cen.get("consumed") != wcen.get("consumed") or
                                        cen.get("mapped") != wcen.get("consumed_mapped") or
                                        cen.get("unmapped") != wcen.get("consumed_unmapped") or
                                        cen.get("mapped_B_fallback") != wcen.get("mapped_B_fallback")):
        why.append("CAPTURE_CENSUS_WRITER")
    if cen.get("unmapped_line_source_positive") != 0:
        why.append("CAPTURE_UNMAPPED_SOURCE_POSITIVE")
    ld = m.get("live_digest", {})
    if ld.get("before") != ld.get("after") or ld.get("file_pairs") != (ld.get("before") or {}).get("pairs"):
        why.append("CAPTURE_LIVE_DIGEST")
    # sparse records: size, order, predicate, domain, counts (independent of the writer)
    lp = capdir / "lines.bin"
    if lp.exists():
        why += records_reasons(lp, mi, meta)
    if run_root is not None:
        rf = ident.get("run_files_sha256", {})
        for rel in RUN_FILES:
            p = Path(run_root) / rel
            if not p.exists() or rf.get(rel) != sha256_file(p):
                why.append(f"CAPTURE_IDENTITY:run_file:{rel}")
        wp = Path(run_root) / "work/det_obs_manifest.json"
        if not wp.exists() or ident.get("writer_manifest_sha256") != sha256_file(wp):
            why.append("CAPTURE_IDENTITY:writer_manifest")
        if ident.get("cwd") != str(Path(run_root) / "work"):
            why.append("CAPTURE_IDENTITY:cwd")
        outs = m.get("outputs", {})
        for rel in (N_FILE, L_FILE):
            p = Path(run_root) / rel
            rec = outs.get(Path(rel).name, {})
            if not p.exists() or sha256_file(p) != rec.get("sha256") or p.stat().st_size != rec.get("bytes"):
                why.append(f"CAPTURE_OUTPUT_LINK:{Path(rel).name}")
    return why


# ------------------------------------------------------------ identity (G-ID)
def parse_exports(path):
    d, dup = {}, []
    for ln in open(path):
        mm = re.match(r'declare -x ([A-Z0-9_]+)(?:="(.*)")?\s*$', ln.rstrip("\n"))
        if not mm:
            continue
        k, v = mm.group(1), mm.group(2)
        if k in d:
            dup.append(k)
        d[k] = v
    return d, dup


def exports_capture_reasons(base_exports, run_exports, base_root, run_root):
    why = []
    b, db = parse_exports(base_exports)
    r, dr = parse_exports(run_exports)
    if db or dr:
        why.append("EXPORTS_DUPLICATE_KEY")
    bt = {k: (v.replace(str(base_root), "RUN") if v is not None else None) for k, v in b.items()}
    rt = {k: (v.replace(str(run_root), "RUN") if v is not None else None) for k, v in r.items()}
    if rt.get(CAPTURE_KEY) != "1" or CAPTURE_KEY in bt:
        why.append("EXPORTS_CAPTURE_KEY")
    rt.pop(CAPTURE_KEY, None)
    extra = sorted(k for k in set(bt) | set(rt) if bt.get(k, "<absent>") != rt.get(k, "<absent>"))
    if extra:
        why.append(f"EXPORTS_EXTRA_DIFF:{extra}")
    return why


def lineage_capture_reasons(t1=T1, t2=T2, b1=B1, patch=PATCH, patch_sha=PATCH_SHA):
    """T1 == B1 build sources; T2 = T1 + registered patch, differing only in CHANGESET."""
    why = []
    built = {}
    for ln in open(Path(b1) / "source_sha_at_build.txt"):
        h, p = ln.split(None, 1)
        built[p.strip()] = h
    if [p for p, h in built.items() if sha(Path(t1) / p) != h]:
        why.append("T1_NE_B1_BUILD")
    t1f = {q.relative_to(t1).as_posix() for q in Path(t1).rglob("*") if q.is_file()}
    t2f = {q.relative_to(t2).as_posix() for q in Path(t2).rglob("*") if q.is_file()}
    for p in sorted(t1f | t2f):
        if p in CHANGESET:
            continue
        if p not in t1f or p not in t2f or sha(Path(t1) / p) != sha(Path(t2) / p):
            why.append(f"T2_UNREGISTERED:{p}")
    if not Path(patch).exists() or sha(patch) != Path(patch_sha).read_text().split()[0]:
        why.append("PATCH_DIGEST")
        return why
    why += patch_apply_reasons(t1, t2, patch)
    return why


def patch_apply_reasons(t1, t2, patch, changeset=CHANGESET):
    """the registered patch applied to T1 reproduces every changeset file of T2
    byte for byte and touches no other file."""
    why = []
    touched = sorted({ln[4:].split("\t")[0].strip().split("/", 1)[1]
                      for ln in open(patch) if ln.startswith("+++ ")})
    if touched != sorted(changeset):
        why.append(f"PATCH_SCOPE:{touched}")
    with tempfile.TemporaryDirectory() as td:
        for f in changeset:
            src = Path(t1) / f
            if src.exists():
                (Path(td) / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, Path(td) / f)
        pr = subprocess.run(["patch", "-p1", "-s", "-N", "-d", td, "-i", str(Path(patch).resolve())],
                            capture_output=True, text=True)
        if pr.returncode != 0:
            return why + [f"PATCH_APPLY_RC:{pr.returncode}"]
        for f in changeset:
            a, b = Path(td) / f, Path(t2) / f
            if not a.exists() or not b.exists() or sha(a) != sha(b):
                why.append(f"T2_NE_T1_PLUS_PATCH:{f}")
    return why


def build_reasons(b2=B2, t2=T2):
    why = []
    try:
        rec = (Path(b2) / "binary.sha256").read_text().split()[0]
        if sha(Path(b2) / "lumina_cuda") != rec or (Path(b2) / "build.rc").read_text().strip() != "0":
            why.append("B2_BINARY")
        for ln in open(Path(b2) / "source_sha_at_build.txt"):
            h, p = ln.split(None, 1)
            if sha(Path(t2) / p.strip()) != h:
                why.append(f"B2_SOURCE:{p.strip()}")
    except OSError:
        why.append("B2_UNREADABLE")
    return why


def run_identity_reasons(base, run, job, b2=B2):
    why = []
    base, run = Path(base), Path(run)
    try:
        if sha(run / "input/lumina_cuda") != (Path(b2) / "binary.sha256").read_text().split()[0]:
            why.append("RUN_BINARY")
        for f in ("deck.sha256", "topion.sha256", "sigma.sha256", "fixed_te_profile.txt"):
            if sha(base / "input" / f) != sha(run / "input" / f):
                why.append(f"RUN_INPUT:{f}")
        why += exports_capture_reasons(base / "input/resolved_lumina.exports",
                                       run / "input/resolved_lumina.exports", base, run)
        exp = runner_rev5(R_RUNNER.read_text(), job)
        if exp is None or (run / "job.slurm").read_text() != exp:
            why.append("RUNNER_UNREGISTERED_DIFF")
        if (run / "SUBMIT_COMMAND.txt").read_text() != submit_expected(str(run)):
            why.append("SUBMIT_COMMAND_NOT_REGISTERED")
        st = dict(ln.rstrip("\n").split("=", 1) for ln in open(run / "STAGE_IDENTITY.txt") if "=" in ln)
        if st.get("runner_sha256") != sha(run / "job.slurm") or \
                st.get("effective_exports_sha256") != sha(run / "input/resolved_lumina.exports"):
            why.append("STAGE_IDENTITY")
    except OSError as e:
        why.append(f"RUN_UNREADABLE:{e.filename}")
    return why


def readout_gate(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return "UNREADABLE"
    if d.get("readout_sha256") != sha(Path(__file__).resolve()):
        return "NOT_THIS_READOUT"
    cases = d.get("cases", [])
    if not cases or any(not c.get("pass") for c in cases):
        return "NOT_ALL_PASSED"
    if not set(REQUIRED_CASES) <= {c.get("case") for c in cases}:
        return "MISSING_REQUIRED"
    return "PASS"


# ------------------------------------------------------------ §7 branches
def shares(lanes, total_err):
    """interval shares f_k = L_k / L with L = sum_k L_k; None if L not > err."""
    L = 0.0
    for v in lanes.values():
        L = L + v["L"]
    eL = add_up(total_err, *[v["err"] for v in lanes.values()])
    if not (L - eL > 0.0):
        return None
    out = {}
    for k, v in lanes.items():
        lo = max(0.0, (v["L"] - v["err"])) / up(L + eL)
        hi = up(up(v["L"] + v["err"]) / (L - eL))
        out[k] = {"f": v["L"] / L, "lo": lo, "hi": min(1.0, hi)}
    return out


def classify(sh):
    """§7.1: MAJORITY (lower > 1/2), MIXED_LARGEST (lower > every other upper),
    UNRESOLVED (numeric overlap), X (no shares)."""
    if not sh:
        return "X", None
    for k, v in sh.items():
        if v["lo"] > 0.5:
            return "MAJORITY", k
    best = max(sh, key=lambda k: sh[k]["f"])
    if all(sh[best]["lo"] > sh[j]["hi"] for j in sh if j != best):
        return "MIXED_LARGEST", best
    return "UNRESOLVED", best


def difference_branch(e_lanes, r_lanes):
    """§7.2: D_k = L_E - L_R with bounds; sign of the total; P/M; branch on the
    contributing side; returns a dict."""
    D, P, M, eD = {}, 0.0, 0.0, 0.0
    for k in LANES:
        d = e_lanes[k]["L"] - r_lanes[k]["L"]
        e = add_up(e_lanes[k]["err"], r_lanes[k]["err"], rho(d))
        D[k] = {"D": d, "err": e}
        eD = add_up(eD, e)
    tot = 0.0
    for k in LANES:
        tot = tot + D[k]["D"]
        eD = add_up(eD, rho(tot))
        P = P + max(D[k]["D"], 0.0)
        M = M + max(-D[k]["D"], 0.0)
    if tot - eD > 0.0:
        side = {k: {"L": max(v["D"], 0.0), "err": v["err"]} for k, v in D.items()}
        sign = "POSITIVE"
    elif tot + eD < 0.0:
        side = {k: {"L": max(-v["D"], 0.0), "err": v["err"]} for k, v in D.items()}
        sign = "NEGATIVE"
    else:
        return {"D": D, "D_total": tot, "D_err": eD, "P": P, "M": M, "sign": "UNDETERMINED",
                "branch": "UNDETERMINED_SIGN", "lead": None}
    sh = shares(side, 0.0)
    br, lead = classify(sh)
    return {"D": D, "D_total": tot, "D_err": eD, "P": P, "M": M, "sign": sign, "branch": br,
            "lead": lead, "shares": sh}


def pair_estimate_reasons(writer_manifests):
    """§5.3 before submission: expected committed bytes of each state from the
    baseline writer census (33-byte records + arrays + meta + manifest bound)."""
    why, est = [], []
    for wm in writer_manifests:
        try:
            ns, nb, nl, n = wm["shells"], wm["bins"], wm["lines"], wm["census"]["consumed"]
        except (KeyError, TypeError):
            return ["PAIR_ESTIMATE_INPUT"]
        e = 33 * n + 8 * (5 * ns * nb + 4 * ns + 2 * nb + nl) + 4096 + MANIFEST_MAX
        est.append(e)
        if e > STATE_CAP:
            why.append(f"STATE_CAP_ESTIMATE:{e}")
    if sum(est) > PAIR_CAP:
        why.append(f"PAIR_CAP_ESTIMATE:{sum(est)}")
    return why


def replay_allowed(why):
    """the C replay reads only an accepted committed capture of a valid run."""
    return not any(x.startswith(("G-READOUT", "G-RUN", "G-CAP")) for x in why)


def colour_terms(eb, rb, eo, ro):
    """§7.2 Q_k = L_Ek/L_E,OPT - L_Rk/L_R,OPT with the scalar OPT luminosities
    as denominators, the total colour difference of the scalar band values, and
    the bound on |sum_k Q_k - total|: in exact arithmetic of the computed
    values the difference is (sum_k L_Ek - L_Eb)/L_E,OPT - (same for R'), so it
    is bounded by each band's closure bound over its denominator plus the
    rounding of every operation.  None when an OPT luminosity is not
    numerically separated from zero."""
    dE, dR = eo["scalar"], ro["scalar"]
    if not (dE - eo["scalar_err"] > 0.0 and dR - ro["scalar_err"] > 0.0):
        return None
    Q, qsum, rnd = {}, 0.0, 0.0
    for k in LANES:
        a, b = eb["lanes"][k]["L"] / dE, rb["lanes"][k]["L"] / dR
        Q[k] = a - b
        qsum = qsum + Q[k]
        rnd = add_up(rnd, rho(a), rho(b), rho(Q[k]), rho(qsum))
    tE, tR = eb["scalar"] / dE, rb["scalar"] / dR
    total = tE - tR
    rnd = add_up(rnd, rho(tE), rho(tR), rho(total))
    bound = add_up(up(eb["closure_bound"] / dE), up(rb["closure_bound"] / dR), rnd)
    diff = abs(qsum - total)
    return {"Q": Q, "Q_sum": qsum, "colour_diff_total": total, "Q_closure_diff": diff,
            "Q_closure_bound": bound, "Q_closure_ok": math.isfinite(bound) and diff <= add_up(bound, rho(diff))}


# ------------------------------------------------------------ commands
def cmd_preflight(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    why = []
    g = readout_gate(a.selftest)
    if g != "PASS":
        why.append(f"G-READOUT:{g}")
    why += lineage_capture_reasons(t2=Path(a.source))
    why += build_reasons(Path(a.build), Path(a.source))
    why += [f"E:{x}" for x in run_identity_reasons(a.baseline_e, a.run_e, JOB["E"], Path(a.build))]
    why += [f"R:{x}" for x in run_identity_reasons(a.baseline_r, a.run_r, JOB["R"], Path(a.build))]
    for nm, base in (("E", a.baseline_e), ("R", a.baseline_r)):
        for rel in (N_FILE, L_FILE, "work/det_obs_manifest.json"):
            if not (Path(base) / rel).exists():
                why.append(f"BASELINE_MISSING:{nm}:{rel}")
    try:
        why += pair_estimate_reasons([json.load(open(Path(b) / "work/det_obs_manifest.json"))
                                      for b in (a.baseline_e, a.baseline_r)])
    except (OSError, ValueError):
        why.append("PAIR_ESTIMATE_INPUT")
    (out / "preflight.json").write_text(json.dumps({"reasons": why}, indent=1))
    print(json.dumps({"G-ID": "PASS" if not why else "FAIL", "reasons": why}))
    return 0 if not why else 1


def cmd_replay(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    run, base = Path(a.run), Path(a.baseline)
    why = []
    g = readout_gate(a.selftest)
    if g != "PASS":
        why.append(f"G-READOUT:{g}")
    ft = footer(run)
    why += [f"G-RUN:{x}" for x in footer_core_reasons(ft, a.slurm_state)]
    why += [f"G-RUN:{x}" for x in resource_reasons(run, a.slurm_state)]
    try:
        wm = json.load(open(run / "work/det_obs_manifest.json"))
        st = json.load(open(run / "work/det_obs_status.json"))
        if st.get("det_obs_state") != "SUCCESS":
            why.append("G-RUN:WRITER_STATE")
    except (OSError, ValueError):
        wm = None
        why.append("G-RUN:WRITER_MANIFEST")
    try:
        cst = json.load(open(run / "work/det_obs_capture_status.json"))
        if cst.get("state") != "COMMITTED":
            why.append(f"G-CAP:STATUS:{cst.get('state')}")
    except (OSError, ValueError):
        why.append("G-CAP:STATUS_MISSING")
    why += [f"G-CAP:{x}" for x in capture_reasons(run / "work/det_obs_capture", wm, run, b2=B2)]
    try:
        mpath = run / "work/det_obs_capture/manifest.json"
        cap_bytes = json.load(open(mpath)).get("committed_payload_bytes") + mpath.stat().st_size
    except (OSError, ValueError, TypeError):
        cap_bytes = None
    # baseline byte gate (§5.4) and writer manifest identity
    base_ok = {}
    for rel in (N_FILE, L_FILE):
        same = (run / rel).exists() and (base / rel).exists() and \
            (run / rel).read_bytes() == (base / rel).read_bytes()
        base_ok[Path(rel).name] = same
        if not same:
            why.append(f"G-CAP:BASELINE_MISMATCH:{Path(rel).name}")
    try:
        bm = json.load(open(base / "work/det_obs_manifest.json"))
        diff = [k for k in WRITER_SAME if (wm or {}).get(k) != bm.get(k)]
        if diff:
            why.append(f"G-CAP:WRITER_MANIFEST_DIFF:{diff}")
    except (OSError, ValueError):
        why.append("G-CAP:BASELINE_MANIFEST")
    # replay (C tool) only on an accepted capture (prereg 4.5 step 6, 6.1)
    rdir = out / "replay"
    if not replay_allowed(why):
        res = {"schema": "DET_OBS_CAPTURE_READOUT_V1", "readout_sha256": sha(Path(__file__).resolve()),
               "run": str(run), "baseline": str(base), "validity": "X", "reasons": why + ["REPLAY_NOT_RUN"],
               "baseline_bytes": base_ok, "capture_payload_bytes": cap_bytes}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps({"validity": "X", "reasons": res["reasons"]}))
        return 0
    rr = subprocess.run([str(a.replay_bin), str(run / "work/det_obs_capture"), str(rdir), str(a.threads)],
                        capture_output=True, text=True)
    (out / "replay.stdout").write_text(rr.stdout)
    (out / "replay.stderr").write_text(rr.stderr)
    rep = {}
    if rr.returncode != 0:
        why.append(f"G-CLOSE:REPLAY_RC:{rr.returncode}")
    try:
        rep = json.load(open(rdir / "replay.json"))
        if (rdir / "scalar.csv").read_bytes() != (run / N_FILE).read_bytes():
            why.append("G-CLOSE:SCALAR_BYTES")
        for k in ("scalar_vs_vector_lane_mismatch_bins", "closure_fail_bins", "interp_mismatch", "nonfinite"):
            if rep.get(k) != 0:
                why.append(f"G-CLOSE:{k}")
    except (OSError, ValueError):
        why.append("G-CLOSE:REPLAY_OUTPUT")
    bands = {}
    try:
        data = read_components(rdir / "components.csv")
        lam = [r[0] for r in data]
        ranges = dict(BANDS)
        ranges["TOTAL"] = (lam[0], lam[-1])
        bands = component_bands(data, ranges)
        for nm, b in bands.items():
            if b.get("status") != "OK" or not b.get("closure_ok") or not b.get("scalar_equals_band_integral"):
                why.append(f"G-CLOSE:BAND:{nm}")
        dep = [r[8] for r in data]
        if any(x != 0.0 for x in dep):
            why.append("G-CLOSE:DEPOSITION_NONZERO")
    except (OSError, ValueError) as e:
        why.append(f"G-CLOSE:COMPONENTS:{e}")
    res = {"schema": "DET_OBS_CAPTURE_READOUT_V1", "readout_sha256": sha(Path(__file__).resolve()),
           "run": str(run), "baseline": str(base), "validity": "VALID" if not why else "X",
           "reasons": why, "baseline_bytes": base_ok, "replay": rep, "bands": bands,
           "capture_payload_bytes": cap_bytes}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "reasons": why}))
    return 0


def cmd_paired(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    e = json.load(open(Path(a.e) / "result.json"))
    r = json.load(open(Path(a.r) / "result.json"))
    res = {"schema": "DET_OBS_CAPTURE_PAIRED_V1", "readout_sha256": sha(Path(__file__).resolve()),
           "inputs": {"E": str(a.e), "R": str(a.r)}}
    if e.get("validity") != "VALID" or r.get("validity") != "VALID":
        res.update({"validity": "X", "reason": "INPUT_NOT_VALID"})
        (out / "result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"validity": "X"}))
        return 0
    pair_bytes = (e.get("capture_payload_bytes") or 0) + (r.get("capture_payload_bytes") or 0)
    res["capture_pair_bytes"] = pair_bytes
    why = []
    if not e.get("capture_payload_bytes") or not r.get("capture_payload_bytes") or pair_bytes > PAIR_CAP:
        why.append("PAIR_PAYLOAD_CAP")
    br = {}
    for b in ("FUVb", "UV2500"):
        eb, rb = e["bands"][b], r["bands"][b]
        sh = shares(eb["lanes"], 0.0)
        cls, lead = classify(sh)
        dif = difference_branch(eb["lanes"], rb["lanes"])
        ct = colour_terms(eb, rb, e["bands"]["OPT"], r["bands"]["OPT"])
        if ct is None:
            why.append(f"OPT_DENOMINATOR:{b}")
        elif not ct["Q_closure_ok"]:
            why.append(f"Q_CLOSURE:{b}")
        br[b] = {"E_shares": sh, "E_branch": cls, "E_lead": lead, "difference": dif, "colour": ct}
    res["bands"] = br
    res["validity"] = "VALID" if not why else "X"
    res["reasons"] = why
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "reasons": why}))
    print(json.dumps({b: {"E_branch": v["E_branch"], "E_lead": v["E_lead"],
                          "diff_branch": v["difference"]["branch"], "diff_lead": v["difference"]["lead"]}
                      for b, v in br.items()}))
    return 0


# ------------------------------------------------------------ selftest
REQUIRED_CASES = (
    "reader accepts the committed C-battery capture",
    "reader rejects a bit flip in lines.bin",
    "reader rejects a generation mismatch",
    "reader rejects a missing pair",
    "reader rejects a duplicated pair",
    "reader rejects a non-committed directory",
    "reader rejects a .partial with a finished manifest",
    "pair estimate: baseline-sized states fit the caps",
    "pair estimate: a state above half the pair cap is rejected",
    "band bound: lanes of the C battery close in every band",
    "band bound: a lane offset beyond rounding fails the closure",
    "band value equals band_integral",
    "branch: majority",
    "branch: mixed largest",
    "branch: numeric overlap is unresolved",
    "branch: tie is unresolved",
    "branch: missing shares is X",
    "difference: positive total with cancellation reports P and M",
    "difference: zero total is undetermined sign",
    "exports: capture key and path transplant only",
    "exports: extra difference rejected",
    "reader rejects an identity mismatch",
    "reader rejects a record domain violation",
    "reader rejects a payload byte record mismatch",
    "lineage: registered patch reproduces T2",
    "lineage: T2 differing from T1 + patch is rejected",
    "lineage: patch touching a file outside the changeset is rejected",
    "band bound: an infinite bound fails the closure",
    "colour: OPT denominator not separated from zero",
    "colour: Q sum closes to the colour difference",
    "colour: a lane offset breaks the Q closure",
    "replay is not run on a rejected capture",
)


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want):
        ok = got == want
        res.append({"case": name, "pass": ok, "got": str(got)[:300], "want": str(want)[:300]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:200]} want {str(want)[:200]}")

    cb = Path(a.c_battery)
    cap = cb / "on/det_obs_capture"
    wm = json.load(open(cb / "on/det_obs_manifest.json"))
    fx_expect = dict(EXPECT, material=8, radiation=9, NS=32, NL=2, asm_serial=int(read_meta(cap / "meta.txt")["asm_serial"]))
    fx_expect["NB"] = int(read_meta(cap / "meta.txt")["NB"])
    case("reader accepts the committed C-battery capture", capture_reasons(cap, wm, None, fx_expect), [])
    tmp = out / "fx"

    def clone(name):
        d = tmp / name / "det_obs_capture"
        shutil.copytree(cap, d)
        return d

    d = clone("flip")
    b = bytearray((d / "lines.bin").read_bytes()); b[40] ^= 1; (d / "lines.bin").write_bytes(bytes(b))
    case("reader rejects a bit flip in lines.bin", "CAPTURE_FILE_IDENTITY:lines.bin" in capture_reasons(d, wm, None, fx_expect), True)
    d = clone("gen")
    m = json.load(open(d / "manifest.json")); m["generations"]["material"] = 6
    (d / "manifest.json").write_text(json.dumps(m))
    case("reader rejects a generation mismatch", "CAPTURE_GENERATION:material" in capture_reasons(d, wm, None, fx_expect), True)

    def rewrite_lines(dd, fn):
        raw = (dd / "lines.bin").read_bytes()
        recs = [raw[i:i + 33] for i in range(0, len(raw), 33)]
        new = b"".join(fn(recs))
        (dd / "lines.bin").write_bytes(new)
        mm = json.load(open(dd / "manifest.json"))
        mm["files"]["lines.bin"] = {"bytes": len(new), "sha256": hashlib.sha256(new).hexdigest()}
        (dd / "manifest.json").write_text(json.dumps(mm))
    d = clone("miss")
    rewrite_lines(d, lambda r: r[:-1])
    case("reader rejects a missing pair", "CAPTURE_PAIRS_COUNT" in capture_reasons(d, wm, None, fx_expect), True)
    d = clone("dup")
    rewrite_lines(d, lambda r: r[:-1] + [r[-2]])
    case("reader rejects a duplicated pair", "CAPTURE_PAIRS_ORDER_OR_DOMAIN" in capture_reasons(d, wm, None, fx_expect), True)
    case("reader rejects a non-committed directory", capture_reasons(tmp / "absent", wm, None, fx_expect), ["CAPTURE_NOT_COMMITTED"])
    case("reader rejects a .partial with a finished manifest",
         capture_reasons(cb / "f5/det_obs_capture.partial", wm, None, fx_expect), ["CAPTURE_NOT_FINAL_NAME"])
    base_like = {"shells": 50, "bins": 1234, "lines": 2588798, "census": {"consumed": 20323129}}
    case("pair estimate: baseline-sized states fit the caps", pair_estimate_reasons([base_like, base_like]), [])
    big = {"shells": 50, "bins": 1234, "lines": 2588798, "census": {"consumed": 31000000}}
    case("pair estimate: a state above half the pair cap is rejected",
         any(x.startswith("STATE_CAP_ESTIMATE") for x in pair_estimate_reasons([big, base_like])), True)
    data = read_components(cb / "replay_ok/components.csv")
    lam = [r[0] for r in data]
    rng = {"FIRST": (lam[1], lam[len(lam) // 3]), "TOTAL": (lam[0], lam[-1])}
    bands = component_bands(data, rng)
    case("band bound: lanes of the C battery close in every band",
         all(v.get("closure_ok") for v in bands.values()), True)
    bad = [list(r) for r in data]
    for r in bad:
        r[2 + 4] = r[2 + 4] * (1.0 + 1e-6) + 1e-30
    case("band bound: a lane offset beyond rounding fails the closure",
         component_bands(bad, rng)["TOTAL"]["closure_ok"], False)
    case("band value equals band_integral", all(v.get("scalar_equals_band_integral") for v in bands.values()), True)

    def mk(vals, err=0.0):
        return {k: {"L": v, "err": err} for k, v in zip(LANES, vals)}
    case("branch: majority", classify(shares(mk([6, 1, 1, 1, 0, 1, 0]), 0.0)), ("MAJORITY", "mapped_line"))
    case("branch: mixed largest", classify(shares(mk([4, 3, 1, 1, 0, 1, 0]), 0.0)), ("MIXED_LARGEST", "mapped_line"))
    case("branch: numeric overlap is unresolved",
         classify(shares(mk([4, 3.999, 1, 1, 0, 1, 0], err=0.01), 0.0))[0], "UNRESOLVED")
    case("branch: tie is unresolved", classify(shares(mk([3, 3, 1, 1, 0, 1, 0]), 0.0))[0], "UNRESOLVED")
    case("branch: missing shares is X", classify(shares(mk([0, 0, 0, 0, 0, 0, 0]), 0.0)), ("X", None))
    dpos = difference_branch(mk([5, 1, 1, 1, 0, 1, 0]), mk([1, 2, 1, 1, 0, 1, 0]))
    case("difference: positive total with cancellation reports P and M",
         (dpos["sign"], dpos["P"], dpos["M"], dpos["branch"], dpos["lead"]), ("POSITIVE", 4.0, 1.0, "MAJORITY", "mapped_line"))
    dz = difference_branch(mk([2, 1, 1, 1, 0, 1, 0]), mk([1, 2, 1, 1, 0, 1, 0]))
    case("difference: zero total is undetermined sign", (dz["sign"], dz["branch"]), ("UNDETERMINED", "UNDETERMINED_SIGN"))
    ex = tmp / "ex"
    ex.mkdir()
    (ex / "B").write_text('declare -x A="1"\ndeclare -x LUMINA_MODEL_DIR="/base/input/model"\n')
    (ex / "R").write_text('declare -x A="1"\ndeclare -x LUMINA_MODEL_DIR="/run/input/model"\ndeclare -x LUMINA_DET_OBS_CAPTURE="1"\n')
    case("exports: capture key and path transplant only", exports_capture_reasons(ex / "B", ex / "R", "/base", "/run"), [])
    (ex / "R2").write_text('declare -x A="2"\ndeclare -x LUMINA_MODEL_DIR="/run/input/model"\ndeclare -x LUMINA_DET_OBS_CAPTURE="1"\n')
    case("exports: extra difference rejected",
         any(x.startswith("EXPORTS_EXTRA_DIFF") for x in exports_capture_reasons(ex / "B", ex / "R2", "/base", "/run")), True)
    d = clone("ident")
    m = json.load(open(d / "manifest.json")); m["identity"]["asm_serial"] += 1
    (d / "manifest.json").write_text(json.dumps(m))
    case("reader rejects an identity mismatch", "CAPTURE_IDENTITY:asm_serial" in capture_reasons(d, wm, None, fx_expect), True)

    def bad_eps(recs):
        out_, done = [], False
        for r_ in recs:
            if not done and r_[32] == 0:
                r_ = r_[:24] + struct.pack("<d", 1.5) + r_[32:]
                done = True
            out_.append(r_)
        return out_
    d = clone("dom")
    rewrite_lines(d, bad_eps)
    case("reader rejects a record domain violation",
         any(x.startswith("CAPTURE_RECORD_DOMAIN") for x in capture_reasons(d, wm, None, fx_expect)), True)
    d = clone("bytes")
    m = json.load(open(d / "manifest.json")); m["committed_payload_bytes"] += 1
    (d / "manifest.json").write_text(json.dumps(m))
    case("reader rejects a payload byte record mismatch", "CAPTURE_PAYLOAD_BYTES" in capture_reasons(d, wm, None, fx_expect), True)
    lt = tmp / "lin"
    for t in ("T1", "T2"):
        (lt / t / "src").mkdir(parents=True)
    (lt / "T1/src/a.c").write_text("int a;\nint b;\n")
    (lt / "T2/src/a.c").write_text("int a;\nint b;\nint c;\n")
    (lt / "T1/src/z.c").write_text("z\n"); (lt / "T2/src/z.c").write_text("z\n")
    pt = subprocess.run(["diff", "-u", "--label", "T1/src/a.c", "--label", "T2/src/a.c",
                         str(lt / "T1/src/a.c"), str(lt / "T2/src/a.c")], capture_output=True, text=True).stdout
    (lt / "p.patch").write_text(pt)
    case("lineage: registered patch reproduces T2", patch_apply_reasons(lt / "T1", lt / "T2", lt / "p.patch", ("src/a.c",)), [])
    (lt / "T2/src/a.c").write_text("int a;\nint b;\nint c;\nint d;\n")
    case("lineage: T2 differing from T1 + patch is rejected",
         "T2_NE_T1_PLUS_PATCH:src/a.c" in patch_apply_reasons(lt / "T1", lt / "T2", lt / "p.patch", ("src/a.c",)), True)
    (lt / "p2.patch").write_text(pt + pt.replace("src/a.c", "src/z.c"))
    case("lineage: patch touching a file outside the changeset is rejected",
         any(x.startswith("PATCH_SCOPE") for x in patch_apply_reasons(lt / "T1", lt / "T2", lt / "p2.patch", ("src/a.c",))), True)
    infb = [list(r_) for r_ in data]
    infb[len(infb) // 2][9] = math.inf
    case("band bound: an infinite bound fails the closure", component_bands(infb, rng)["TOTAL"]["closure_ok"], False)
    bE = component_bands(data, rng)
    two = [[r_[0]] + [2.0 * x for x in r_[1:10]] + [r_[10]] for r_ in data]
    bR = component_bands(two, rng)
    zero_opt = dict(bE["TOTAL"], scalar_err=bE["TOTAL"]["scalar"])
    case("colour: OPT denominator not separated from zero", colour_terms(bE["FIRST"], bR["FIRST"], zero_opt, bR["TOTAL"]), None)
    ct = colour_terms(bE["FIRST"], bR["FIRST"], bE["TOTAL"], bR["TOTAL"])
    case("colour: Q sum closes to the colour difference", bool(ct and ct["Q_closure_ok"]), True)
    off = json.loads(json.dumps(bE["FIRST"]))
    big = max(LANES, key=lambda k: off["lanes"][k]["L"])
    off["lanes"][big]["L"] = off["lanes"][big]["L"] * 1.01
    ct2 = colour_terms(off, bR["FIRST"], bE["TOTAL"], bR["TOTAL"])
    case("colour: a lane offset breaks the Q closure", bool(ct2) and ct2["Q_closure_ok"], False)
    case("replay is not run on a rejected capture",
         (replay_allowed(["G-CAP:CAPTURE_FILE_IDENTITY:J.f64"]), replay_allowed(["G-RUN:X"]), replay_allowed([])),
         (False, False, True))
    names = {r["case"] for r in res}
    case("battery covers every required case", sorted(set(REQUIRED_CASES) - names), [])
    nfail = sum(1 for r in res if not r["pass"])
    (out / "readout_selftest.json").write_text(json.dumps(
        {"readout_sha256": sha(Path(__file__).resolve()), "n_cases": len(res), "failures": nfail, "cases": res}, indent=1))
    print(f"DET-OBS-CAPTURE readout selftest: {len(res) - nfail}/{len(res)} -> {'PASS' if nfail == 0 else 'FAIL'}")
    return 0 if nfail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("selftest")
    p.add_argument("--c-battery", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("preflight")
    for k in ("--baseline-e", "--baseline-r", "--source", "--build", "--run-e", "--run-r", "--selftest", "--out"):
        p.add_argument(k, required=True)
    p = sp.add_parser("replay")
    for k in ("--run", "--baseline", "--slurm-state", "--selftest", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--replay-bin", default=str(REPLAY_BIN)); p.add_argument("--threads", default="16")
    p = sp.add_parser("paired")
    for k in ("--e", "--r", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "preflight": cmd_preflight, "replay": cmd_replay,
            "paired": cmd_paired}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
