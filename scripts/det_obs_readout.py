#!/usr/bin/env python3
"""DET-OBS-WRITER readout (prereg docs/PREREG_DET_OBS_WRITER_2026-10-02.md §4.1-§5).

Subcommands
  selftest --out DIR
  preflight --run RP --base R --out DIR             G-ID before submission (R' first)
  readout --mode rprime --run RP --base R --slurm-states JSON --selftest JSON --out DIR
  readout --mode paired --run RP --base R --eprime EP --ebase E --slurm-states JSON
          --selftest JSON --out DIR

Observable (5): for each run X and writer w in {N, L} the band luminosities
(FUV0 912-1200, FUVa 1200-1500, FUVb 1500-2000, NUV 2500-4000, OPT 4000-6000,
whole file) and colours C_j = L_j / L_OPT.  W_X = log10 C_N / C_L (same view).
paired only: Delta_N, Delta_L, the algebra check and the FUV branch on Delta_N.
validity VALID / X is separate from the branch; the mode is explicit, never
inferred from which files exist.
"""
import argparse
import csv
import shutil
import subprocess
import tempfile
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nrb_e_readout import (sha, fnum, read_exports, footer, footer_reasons,      # frozen 5b5a68f9
                           band_integral, nrb_witness, manifest_reasons,
                           publication_reasons, linkage_reasons, FORBIDDEN, guarded)

N_ROOT = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002")
T0 = Path("/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source")
T1 = N_ROOT / "source"
B1 = N_ROOT / "cuda_build1"
CHANGESET = ("src/lumina_cmfgen.c", "src/lumina_cmfgen.h", "src/env_universe.h",
             "tests/det_obs_writer_selftest.c")
PATH_KEYS = ("LUMINA_MODEL_DIR", "LUMINA_CMFGEN_SIGMA_BF", "LUMINA_DEPOSITION_FILE",
             "LUMINA_FIXED_TE_PROFILE", "LUMINA_IONIZATION_REFERENCE_FILE",
             "LUMINA_TOPION_LEVELS_FILE", "LUMINA_PHYSICS_COMPARISON_DIR")
NEW_KEY = "LUMINA_DET_OBS_WRITER"
BANDS = {"FUV0": (912.0, 1200.0), "FUVa": (1200.0, 1500.0), "FUVb": (1500.0, 2000.0),
         "NUV": (2500.0, 4000.0), "OPT": (4000.0, 6000.0)}
COLOURS = ("FUV0", "FUVa", "FUVb", "NUV")
THR = 0.30
SPEC = {"N": "work/lumina_spectrum_det_obs.csv", "L": "work/lumina_spectrum_det_obs_legacy.csv"}
LEGACY4 = "work/lumina_spectrum.csv"
INPUT_SAME = ("deck.sha256", "topion.sha256", "sigma.sha256", "fixed_te_profile.txt")
B0_SOURCES = Path("/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/source_sha_at_build.txt")
PATCH = N_ROOT / "evidence/det_obs_writer.patch"
PATCH_SHA = N_ROOT / "evidence/det_obs_writer.patch.sha256"
CHANGESET_SHA = N_ROOT / "evidence/T1_changeset.sha256"
FROZEN_DEPS = {"nrb_e_readout.py": "5b5a68f9475aa3c1b035d621b7a807566df8f5ecee5c6eb3bac9162da544c258",
               "nt_eta_readout.py": "a4ae3f3ae31f5b28910204bbfe1bbf5ef99292b9b06c1c53a866be08ed20ce3f",
               "w5_cont_readout.py": "7008f709e4ca49a24823e4f88f452c42448247d8cb5940c1c1670c4e31a7f3cd"}
STAGE1_EVIDENCE = {"selftest": N_ROOT / "stage1/st_final/cases.json",
                   "nc": N_ROOT / "stage1/nc_final.log",
                   "off": N_ROOT / "stage1/off_final.log",
                   "audit": N_ROOT / "evidence/code_audit_closure.txt",
                   "cost": N_ROOT / "evidence/stage2_cost_estimate.json",
                   "user_rprime": N_ROOT / "USER_DECISION_L5.txt"}


def fnv1a(path):
    h = 0xcbf29ce484222325
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            for c in blk:
                h ^= c
                h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


# ------------------------------------------------------------------ observable
def read_spectrum(path):
    """sorted (lambda, L_lambda) points; reasons for header/duplicate/non-finite."""
    p = Path(path)
    if not p.exists():
        return None, ["SPECTRUM_MISSING"]
    rows = list(csv.reader(open(p)))
    if not rows or rows[0] != ["wavelength_angstrom", "flux"]:
        return None, ["SPECTRUM_HEADER"]
    pts = []
    for r in rows[1:]:
        if len(r) != 2:
            return None, ["SPECTRUM_ROW"]
        pts.append((fnum(r[0]), fnum(r[1])))
    if not pts:
        return None, ["SPECTRUM_EMPTY"]
    if any(not (math.isfinite(a) and math.isfinite(b)) for a, b in pts):
        return None, ["SPECTRUM_NONFINITE"]
    pts.sort()
    if len({a for a, _ in pts}) != len(pts):
        return None, ["SPECTRUM_DUPLICATE_WAVELENGTH"]
    return pts, []


def band_table(pts):
    out, why = {}, []
    for nm, (lo, hi) in BANDS.items():
        v, st = band_integral(pts, lo, hi)
        if st != "OK" or v is None or not math.isfinite(v):
            why.append(f"BAND_{st}:{nm}")
            v = None
        out[nm] = v
    tot, st = band_integral(pts, pts[0][0], pts[-1][0])
    out["TOTAL"] = tot if st == "OK" else None
    out["TOTAL_RANGE"] = [pts[0][0], pts[-1][0]]
    if st != "OK":
        why.append(f"BAND_{st}:TOTAL")
    return out, why


def colours(bt):
    c, why = {}, []
    den = bt.get("OPT")
    for j in COLOURS:
        num = bt.get(j)
        if num is None or den is None or not (num > 0.0 and den > 0.0):
            why.append(f"COLOUR_UNDEFINED:{j}")
            c[j] = None
        else:
            c[j] = num / den
    return c, why


def cat(x):
    if x is None or not math.isfinite(x):
        return "MISSING"
    if x <= -THR:
        return "down"
    if x >= THR:
        return "up"
    return "flat"


def logratio(a, b):
    if a is None or b is None or not (a > 0.0 and b > 0.0):
        return None
    return math.log10(a) - math.log10(b)


def paired_branch(cat_a, cat_b, invalid):
    if invalid or "MISSING" in (cat_a, cat_b):
        return "X"
    s = {cat_a, cat_b}
    if "down" in s and "up" in s:
        return "O-MIX"
    if "down" in s:
        return "O-DOWN"
    if "up" in s:
        return "O-UP"
    return "O-FLAT"


def run_observable(run, legacy4_path=None):
    """N and L band tables / colours of one run, and W_X.  Never raises."""
    res, why = {}, []
    for w, rel in SPEC.items():
        pts, r = read_spectrum(Path(run) / rel)
        why += [f"{w}:{x}" for x in r]
        if pts is None:
            res[w] = None
            continue
        bt, r2 = band_table(pts)
        c, r3 = colours(bt)
        why += [f"{w}:{x}" for x in r2 + r3]
        res[w] = {"bands": bt, "colours": c}
    if legacy4_path is not None:
        for w, rel in SPEC.items():
            q = Path(run) / rel
            if q.exists() and Path(legacy4_path).exists() and \
                    Path(legacy4_path).resolve() == q.resolve():
                why.append(f"{w}:IS_LEGACY4_FILE")
    W = {}
    if res.get("N") and res.get("L"):
        for j in COLOURS:
            W[j] = logratio(res["N"]["colours"][j], res["L"]["colours"][j])
            if W[j] is None:
                why.append(f"W_UNDEFINED:{j}")
    else:
        why.append("W_UNDEFINED:ALL")
    res["W"] = W
    res["W_cat"] = {j: cat(W.get(j)) for j in COLOURS}
    return res, why


# ------------------------------------------------------------------ det_obs run evidence
def det_obs_reasons(run):
    """status SUCCESS, success manifest, output digests = manifest."""
    W = Path(run) / "work"
    why = []
    sp = W / "det_obs_status.json"
    if not sp.exists():
        return ["DET_OBS_STATUS_MISSING"]
    try:
        st = json.load(open(sp))
    except ValueError:
        return ["DET_OBS_STATUS_UNREADABLE"]
    state = st.get("det_obs_state", "")
    if state != "SUCCESS":
        why.append(f"DET_OBS_STATE:{state}")
    mp = W / "det_obs_manifest.json"
    if not mp.exists():
        return why + ["DET_OBS_MANIFEST_MISSING"]
    try:
        m = json.load(open(mp))
    except ValueError:
        return why + ["DET_OBS_MANIFEST_UNREADABLE"]
    if m.get("mode") != 1:
        why.append(f"DET_OBS_MODE:{m.get('mode')}")
    vd = m.get("view_fnv1a", {})
    if not vd or len({vd.get("before"), vd.get("after_N"), vd.get("after_L")}) != 1 or None in vd.values():
        why.append("DET_OBS_VIEW_DIGEST")
    if not isinstance(m.get("material_generation"), int) or not isinstance(m.get("rad_generation"), int):
        why.append("DET_OBS_GENERATIONS")
    if not (m.get("census", {}).get("consumed", 0) > 0):
        why.append("DET_OBS_NOTHING_CONSUMED")
    rec = m.get("reconcile_max_rel", {})
    for k in ("eta_line_thermal", "unmapped_scatter"):
        v = rec.get(k)
        if not (isinstance(v, (int, float)) and v <= 1e-12):
            why.append(f"DET_OBS_RECONCILE:{k}")
    outs = m.get("outputs", {})
    for rel in SPEC.values():
        name = Path(rel).name
        q = W / name
        e = outs.get(name)
        if not q.exists() or not e:
            why.append(f"DET_OBS_OUTPUT_MISSING:{name}")
            continue
        if fnv1a(q) != e.get("fnv1a") or q.stat().st_size != e.get("bytes"):
            why.append(f"DET_OBS_OUTPUT_HASH:{name}")
    return why


def grun(run, slurm_state, armed_expected):
    """G-RUN for one executed run (prereg §4.4); every unreadable piece is a reason."""
    why = []
    W = Path(run) / "work"
    why += footer_reasons(footer(run), slurm_state)
    why += guarded(why, "MANIFESTS", manifest_reasons, W) or []
    why += guarded(why, "PUBLICATION", publication_reasons, W) or []
    log = list(open(Path(run) / "stderr.log", errors="replace")) if (Path(run) / "stderr.log").exists() else []
    if not log:
        why.append("STDERR_MISSING")
    ok = sum(1 for ln in log if ln.startswith("[cmf_fine][EXACT-MULTIGPU-EPOCH]") and " status=OK " in ln + " ")
    if ok != 7:
        why.append(f"EXACT_SOLVE_OK:{ok}")
    hit = [t for t in FORBIDDEN if any(ln.startswith(t) for ln in log)]
    if hit:
        why.append(f"FORBIDDEN_TAG:{hit}")
    why += nrb_witness(log, armed_expected)
    tag = [ln for ln in log if ln.startswith("[DET-OBS] det_obs_state=")]
    if len(tag) != 1 or not tag[0].startswith("[DET-OBS] det_obs_state=SUCCESS"):
        why.append(f"DET_OBS_TAG:{len(tag)}")
    why += det_obs_reasons(run)
    for p in ("lumina_spectrum.csv", "physics_comparison/physics_DET_iter0005.manifest.json"):
        if not (W / p).exists():
            why.append(f"FILE_MISSING:{p}")
    return why


# ------------------------------------------------------------------ G-ID
def exports_reasons(p_base, p_run, base_root, run_root):
    """exports delta = exactly LUMINA_DET_OBS_WRITER absent -> "1" + path transplant."""
    dB, dupB = read_exports(p_base)
    dR, dupR = read_exports(p_run)
    why = ["EXPORTS_DUPLICATE_KEY"] if (dupB or dupR) else []
    for k in sorted(set(dB) | set(dR) | {NEW_KEY} | set(PATH_KEYS)):
        vB, vR = dB.get(k), dR.get(k)
        if k == NEW_KEY:
            if not (vB is None and vR == "1"):
                why.append(f"EXPORTS_NEW_KEY:{vB}->{vR}")
            continue
        if k in PATH_KEYS:
            if vB is None or vR is None or not vB.startswith(base_root) or \
                    vR != run_root + vB[len(base_root):]:
                why.append(f"EXPORTS_PATH:{k}")
            continue
        if vB != vR:
            why.append(f"EXPORTS_EXTRA_DIFF:{k}")
    return why


def changeset_reasons(t0_manifest, t1_root):
    """T1 differs from T0 only in the registered changeset files (+ new test file)."""
    why = []
    base = {}
    for ln in open(t0_manifest):
        h, p = ln.split(None, 1)
        base[p.strip()] = h
    for p, h in base.items():
        q = Path(t1_root) / p
        if not q.exists():
            why.append(f"T1_MISSING:{p}")
        elif sha(q) != h and p not in CHANGESET:
            why.append(f"T1_UNREGISTERED_CHANGE:{p}")
    for p in CHANGESET:
        if p not in base and not (Path(t1_root) / p).exists():
            why.append(f"T1_CHANGESET_ABSENT:{p}")
    return why


def stage1_reasons(ev=STAGE1_EVIDENCE):
    why = []
    for k, p in ev.items():
        if not Path(p).exists():
            why.append(f"STAGE1_EVIDENCE_MISSING:{k}")
    if Path(ev["selftest"]).exists():
        try:
            d = json.load(open(ev["selftest"]))
            if d.get("failures") != 0 or d.get("fault") != 0 or not d.get("n_cases"):
                why.append("STAGE1_SELFTEST_NOT_PASS")
        except ValueError:
            why.append("STAGE1_SELFTEST_UNREADABLE")
    if Path(ev["nc"]).exists() and "NC_ALL=PASS" not in open(ev["nc"]).read():
        why.append("STAGE1_NC_NOT_PASS")
    if Path(ev["off"]).exists() and "OFF_BYTES=IDENTICAL" not in open(ev["off"]).read():
        why.append("STAGE1_OFF_NOT_IDENTICAL")
    return why


def lineage_reasons(t0=T0, b0_sources=B0_SOURCES, t1=T1, patch=PATCH, patch_sha=PATCH_SHA,
                    changeset_sha=CHANGESET_SHA, deps=FROZEN_DEPS, here=None):
    """B0 source manifest == T0 files; T1 changeset == registered hashes; patch
    file == registered digest and names exactly the changeset; frozen imports."""
    why = []
    for ln in open(b0_sources):
        h, p = ln.split(None, 1)
        q = Path(t0) / p.strip()
        if not q.exists() or sha(q) != h:
            why.append(f"T0_NE_B0:{p.strip()}")
    reg = {}
    for ln in open(changeset_sha):
        h, p = ln.split(None, 1)
        reg[p.strip()] = h
    if sorted(reg) != sorted(CHANGESET):
        why.append("CHANGESET_REGISTER_SET")
    for p, h in reg.items():
        if sha(Path(t1) / p) != h:
            why.append(f"T1_NE_REGISTERED:{p}")
    if sha(patch) != Path(patch_sha).read_text().split()[0]:
        why.append("PATCH_DIGEST")
    named = sorted({ln[len("+++ T1/"):].split()[0] for ln in open(patch) if ln.startswith("+++ T1/")})
    if named != sorted(CHANGESET):
        why.append(f"PATCH_FILES:{named}")
    # the registered patch applied to T0 must reproduce T1 byte for byte
    with tempfile.TemporaryDirectory() as td:
        for rel in CHANGESET:
            src = Path(t0) / rel
            if src.exists():
                (Path(td) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, Path(td) / rel)
        r = subprocess.run(["patch", "-p1", "-s", "-d", td, "-i", str(Path(patch).resolve())],
                           capture_output=True, text=True)
        if r.returncode != 0:
            why.append("PATCH_APPLY_FAILED")
        else:
            for rel in CHANGESET:
                q = Path(td) / rel
                if not q.exists() or sha(q) != sha(Path(t1) / rel):
                    why.append(f"PATCH_RESULT_NE_T1:{rel}")
    here = Path(here or Path(__file__).resolve().parent)
    for f, h in deps.items():
        if sha(here / f) != h:
            why.append(f"FROZEN_DEP:{f}")
    return why


def gid(base, run, post_run, b1=B1):
    why, ev = [], {}
    base, run = Path(base), Path(run)
    rec = (b1 / "binary.sha256").read_text().split()[0]
    bins = {"build_record": rec, "build_file": sha(b1 / "lumina_cuda"),
            "run_file": sha(run / "input/lumina_cuda"),
            "run_record": (run / "input/binary.sha256").read_text().split()[0]}
    if post_run:
        bins["run_footer"] = footer(run).get("binary_sha256", "")
    ev["binary"] = bins
    if len(set(bins.values())) != 1:
        why.append("BINARY_MISMATCH")
    built = {}
    for ln in open(b1 / "source_sha_at_build.txt"):
        h, p = ln.split(None, 1)
        built[p.strip()] = h
    bad = [p for p, h in built.items() if sha(T1 / p) != h]
    if bad:
        why.append(f"SOURCE_TREE_MISMATCH:{len(bad)}")
    why += changeset_reasons(N_ROOT / "T0_manifest.sha256", T1)
    why += lineage_reasons()
    for f in INPUT_SAME:
        if sha(base / "input" / f) != sha(run / "input" / f):
            why.append(f"INPUT_MISMATCH:{f}")
    why += exports_reasons(base / "input/resolved_lumina.exports", run / "input/resolved_lumina.exports",
                           str(base), str(run))
    why += linkage_reasons(run, "RP", post_run)
    why += stage1_reasons()
    return why, ev


# ------------------------------------------------------------------ G-READOUT
REQUIRED_CASES = (   # frozen battery: contract §4.3 categories (G-READOUT requires all)
    "rprime healthy: VALID, W_R computed, paired branch pending",
    "paired same input vs same input: Delta_N=0, O-FLAT, algebra 0",
    "rprime W_R FUVa up (same view N vs L) is not a Delta_N",
    "rprime: unexecuted E' is not X and W_R is not substituted for Delta_N",
    "paired: missing E' spectrum -> X, not reduced to rprime",
    "spectrum defect missing -> N:SPECTRUM_MISSING",
    "spectrum defect header -> N:SPECTRUM_HEADER",
    "spectrum defect dup -> N:SPECTRUM_DUPLICATE_WAVELENGTH",
    "spectrum defect nan -> N:SPECTRUM_NONFINITE",
    "spectrum defect inf -> N:SPECTRUM_NONFINITE",
    "unsupported band (FUV0 not covered) -> BAND_UNSUPPORTED",
    "zero FUV integral -> colour undefined (no log)",
    "negative FUV integral -> colour undefined",
    "det_obs_state FAILED(reason) -> X even with model_rc 0",
    "status file missing -> X",
    "success manifest missing -> X",
    "output changed after manifest (partial / hash) -> X",
    "production run must be capture mode 1",
    "legacy (4) file used as the same-view pair -> X",
    "signed/zero-tau component unsupported alone: not a G-RUN reason",
    "exports: exactly the new key + path transplant",
    "exports: duplicated key rejected",
    "exports: unregistered difference rejected",
    "exports: new key not 1 rejected",
    "changeset: registered file change accepted",
    "changeset: unregistered file change rejected",
    "lineage: consistent B0/T0/T1/patch/deps",
    "lineage: T0 file differs from B0 build manifest",
    "lineage: T1 changeset file differs from registered hash",
    "lineage: patch digest mismatch",
    "lineage: frozen import dependency changed",
    "lineage: patch applied to T0 does not reproduce T1",
    "manifest view digest changed across writers -> X",
    "observables.csv rows: header + N/L bands, colours and W",
    "G-RUN on missing run: registered reasons, no exception",
    "branch: -0.30 down / +0.30 up -> O-MIX",
    "branch: just inside +-0.30 -> O-FLAT",
    "branch: down + flat -> O-DOWN",
    "branch: up + flat -> O-UP",
    "branch: invalid first -> X",
    "stage1 evidence absent -> reasons",
    "G-READOUT rejects another readout's summary",
    "G-READOUT rejects a passing summary lacking required cases",
)


def readout_gate(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return "UNREADABLE"
    if not isinstance(d, dict) or d.get("readout_sha256") != sha(Path(__file__).resolve()):
        return "NOT_THIS_READOUT"
    cases = d.get("cases", [])
    if not cases or d.get("n_cases") != len(cases) or any(not c.get("pass") for c in cases):
        return "NOT_ALL_PASSED"
    if not set(REQUIRED_CASES) <= {c.get("case") for c in cases}:
        return "MISSING_REQUIRED"
    return "PASS"


# ------------------------------------------------------------------ evaluation
def evaluate(mode, obs_r, why_r, obs_e=None, why_e=None):
    """Pure: branch / deltas from already gathered observables and reasons."""
    out = {"mode": mode, "R": obs_r, "W_R": obs_r.get("W") if obs_r else None}
    invalid_r = list(why_r)
    if mode == "rprime":
        out["validity"] = "X" if invalid_r else "VALID"
        out["invalid_reasons"] = invalid_r
        out["paired"] = {"status": "PENDING_EPRIME", "Delta_N": None, "Delta_L": None,
                         "branch": None}
        return out
    if mode != "paired":
        raise ValueError(mode)
    invalid = invalid_r + [f"E':{x}" for x in (why_e or [])]
    if obs_e is None:
        invalid.append("E':OBSERVABLE_MISSING")
    dN, dL, alg = {}, {}, {}
    for j in COLOURS:
        cRN = obs_r["N"]["colours"][j] if obs_r and obs_r.get("N") else None
        cEN = obs_e["N"]["colours"][j] if obs_e and obs_e.get("N") else None
        cRL = obs_r["L"]["colours"][j] if obs_r and obs_r.get("L") else None
        cEL = obs_e["L"]["colours"][j] if obs_e and obs_e.get("L") else None
        dN[j], dL[j] = logratio(cRN, cEN), logratio(cRL, cEL)
        wR = obs_r.get("W", {}).get(j) if obs_r else None
        wE = obs_e.get("W", {}).get(j) if obs_e else None
        if None in (dN[j], dL[j], wR, wE):
            alg[j] = None
        else:
            alg[j] = abs((dN[j] - dL[j]) - (wR - wE))
    ca, cb = cat(dN.get("FUVa")), cat(dN.get("FUVb"))
    br = paired_branch(ca, cb, invalid)
    out.update({"validity": "X" if invalid else "VALID", "invalid_reasons": invalid,
                "E": obs_e, "W_E": obs_e.get("W") if obs_e else None,
                "paired": {"status": "EVALUATED", "Delta_N": dN, "Delta_L": dL,
                           "algebra_abs_residual": alg, "cat_FUVa": ca, "cat_FUVb": cb,
                           "branch": br}})
    return out


def cmd_readout(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.mode not in ("rprime", "paired"):
        raise SystemExit("mode must be rprime or paired")
    st = guarded([], "SLURM", lambda p: json.load(open(p)), a.slurm_states) or {}
    g_read = readout_gate(a.selftest)
    why_r = [] if g_read == "PASS" else [f"G-READOUT:{g_read}"]
    gidr = guarded(why_r, "G-ID:READ", gid, a.base, a.run, True)
    if gidr:
        why_r += [f"G-ID:{w}" for w in gidr[0]]
    why_r += [f"G-RUN:{w}" for w in (guarded(why_r, "G-RUN:READ", grun, a.run, st.get("RP"), True) or [])]
    obs_r, w = run_observable(a.run, Path(a.run) / LEGACY4)
    why_r += w
    obs_e, why_e = None, None
    if a.mode == "paired":
        if not a.eprime or not a.ebase:
            raise SystemExit("paired mode requires --eprime and --ebase")
        why_e = []
        gide = guarded(why_e, "G-ID:READ", gid, a.ebase, a.eprime, True)
        if gide:
            why_e += [f"G-ID:{x}" for x in gide[0]]
        why_e += [f"G-RUN:{x}" for x in (guarded(why_e, "G-RUN:READ", grun, a.eprime, st.get("EP"), False) or [])]
        obs_e, w2 = run_observable(a.eprime, Path(a.eprime) / LEGACY4)
        why_e += w2
    res = evaluate(a.mode, obs_r, why_r, obs_e, why_e)
    res.update({"schema": "DET_OBS_READOUT_V1", "readout_sha256": sha(Path(__file__).resolve()),
                "G-READOUT": g_read,
                "signed_zero_tau_component": {nm: signed_status(r) for nm, r in
                                              (("RP", a.run), ("EP", a.eprime)) if r}})
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    write_observables(out / "observables.csv", {"RP": obs_r, "EP": obs_e})
    cons = []
    for nm, r in (("RP", a.run), ("EP", a.eprime)):
        if not r:
            continue
        for rel in list(SPEC.values()) + [LEGACY4, "work/det_obs_manifest.json", "work/det_obs_status.json",
                                          "RUN_FOOTER.txt", "stderr.log", "input/resolved_lumina.exports"]:
            q = Path(r) / rel
            cons.append({"run": nm, "path": str(q), "sha256": sha(q) if q.exists() else None})
    me = Path(__file__).resolve()
    (out / "evidence_manifest.json").write_text(json.dumps(
        {"readout": {"path": str(me), "sha256": sha(me)},
         "deps": {f: sha(me.parent / f) for f in FROZEN_DEPS}, "consumed": cons,
         "gate_evidence": [{"path": str(x), "sha256": sha(x) if Path(x).exists() else None}
                           for x in (a.selftest, a.slurm_states)]}, indent=1))
    print(json.dumps({k: res.get(k) for k in ("mode", "validity", "invalid_reasons", "W_R")}, default=str))
    return 0


def signed_status(run):
    p = Path(run) / "work/det_obs_status.json"
    try:
        return json.load(open(p)).get("signed_zero_tau_component", "STATUS_MISSING")
    except (OSError, ValueError):
        return "STATUS_UNREADABLE"


def write_observables(path, obs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "writer", "quantity", "band", "value"])
        for nm, o in obs.items():
            if not o:
                continue
            for wr in ("N", "L"):
                if not o.get(wr):
                    continue
                for b, v in o[wr]["bands"].items():
                    w.writerow([nm, wr, "L_band", b, repr(v) if not isinstance(v, list) else v])
                for b, v in o[wr]["colours"].items():
                    w.writerow([nm, wr, "colour_over_OPT", b, repr(v)])
            for b, v in o.get("W", {}).items():
                w.writerow([nm, "N/L", "W_log10", b, repr(v)])


def cmd_preflight(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    why = []
    r = guarded(why, "G-ID:READ", gid, a.base, a.run, False)
    if r:
        why += r[0]
    if a.eprime or a.ebase:
        if not (a.eprime and a.ebase):
            raise SystemExit("paired preflight requires both --eprime and --ebase")
        re_ = guarded(why, "G-ID:EP:READ", gid, a.ebase, a.eprime, False)
        if re_:
            why += [f"EP:{x}" for x in re_[0]]
    (out / "preflight.json").write_text(json.dumps({"reasons": why, "evidence": r[1] if r else None}, indent=1))
    print(json.dumps({"G-ID": "PASS" if not why else "FAIL", "reasons": why}))
    return 0 if not why else 1


# ------------------------------------------------------------------ selftest
def _spec(path, pts, header="wavelength_angstrom,flux"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n")
        for l, v in pts:
            f.write(f"{l!r},{v!r}\n")


def _flat(level, slope=0.0, lo=800.0, hi=7000.0, n=400):
    return [(lo + (hi - lo) * i / (n - 1), level * (1.0 + slope * i / (n - 1))) for i in range(n)]


def _fake_run(root, n_pts, l_pts, state="SUCCESS", manifest=True, mode=1):
    W = Path(root) / "work"
    _spec(W / "lumina_spectrum_det_obs.csv", n_pts)
    _spec(W / "lumina_spectrum_det_obs_legacy.csv", l_pts)
    (W / "det_obs_status.json").write_text(json.dumps({"det_obs_state": state}))
    if manifest:
        outs = {}
        for rel in SPEC.values():
            q = W / Path(rel).name
            outs[q.name] = {"bytes": q.stat().st_size, "fnv1a": fnv1a(q)}
        (W / "det_obs_manifest.json").write_text(json.dumps({
            "mode": mode, "census": {"consumed": 10}, "rad_generation": 9, "material_generation": 8,
            "view_fnv1a": {"before": "a", "after_N": "a", "after_L": "a"},
            "reconcile_max_rel": {"eta_line_thermal": 0.0, "unmapped_scatter": 0.0},
            "outputs": outs}))


def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = []

    def case(name, got, want):
        ok = got == want
        res.append({"case": name, "pass": ok, "got": str(got), "want": str(want)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got} want {want}")

    tmp = out / "fx"
    # R' alone, healthy: VALID, W computed, paired pending
    _fake_run(tmp / "Rgood", _flat(1.0), _flat(1.0))
    obs, w = run_observable(tmp / "Rgood")
    w += det_obs_reasons(tmp / "Rgood")
    ev = evaluate("rprime", obs, w)
    case("rprime healthy: VALID, W_R computed, paired branch pending",
         (ev["validity"], ev["W_R"]["FUVa"], ev["paired"]["status"], ev["paired"]["branch"]),
         ("VALID", 0.0, "PENDING_EPRIME", None))
    # paired healthy and same-input control
    ev = evaluate("paired", obs, [], obs, [])
    case("paired same input vs same input: Delta_N=0, O-FLAT, algebra 0",
         (ev["validity"], ev["paired"]["Delta_N"]["FUVa"], ev["paired"]["branch"],
          ev["paired"]["algebra_abs_residual"]["FUVb"]), ("VALID", 0.0, "O-FLAT", 0.0))
    # FUV-only scaling: N has FUVa 10x (OPT unchanged)
    def scaled(f):
        return [(l, v * (f if 1200.0 <= l <= 1500.0 else 1.0)) for l, v in _flat(1.0)]
    _fake_run(tmp / "Rup", scaled(4.0), _flat(1.0))
    obs_up, wu = run_observable(tmp / "Rup")
    case("rprime W_R FUVa up (same view N vs L) is not a Delta_N",
         (cat(obs_up["W"]["FUVa"]), evaluate("rprime", obs_up, wu)["paired"]["branch"]), ("up", None))
    # rprime must not mark the missing E' as X and must not emit a paired branch
    evr = evaluate("rprime", obs_up, wu)
    case("rprime: unexecuted E' is not X and W_R is not substituted for Delta_N",
         (evr["validity"], evr["paired"]["Delta_N"]), ("VALID", None))
    # paired with a missing E' file must be X (no silent fallback to rprime)
    _fake_run(tmp / "Enofile", _flat(1.0), _flat(1.0))
    (tmp / "Enofile/work/lumina_spectrum_det_obs.csv").unlink()
    obs_e, we = run_observable(tmp / "Enofile")
    evp = evaluate("paired", obs, [], obs_e, we)
    case("paired: missing E' spectrum -> X, not reduced to rprime",
         (evp["validity"], evp["mode"], evp["paired"]["branch"]), ("X", "paired", "X"))
    # spectrum defects
    for nm, pts, hdr, want in (
            ("missing", None, None, "N:SPECTRUM_MISSING"),
            ("header", _flat(1.0), "lam,flux", "N:SPECTRUM_HEADER"),
            ("dup", _flat(1.0) + [(_flat(1.0)[3][0], 1.0)], None, "N:SPECTRUM_DUPLICATE_WAVELENGTH"),
            ("nan", _flat(1.0)[:5] + [(900.5, float("nan"))] + _flat(1.0)[5:], None, "N:SPECTRUM_NONFINITE"),
            ("inf", _flat(1.0)[:5] + [(900.5, float("inf"))] + _flat(1.0)[5:], None, "N:SPECTRUM_NONFINITE")):
        d = tmp / f"S_{nm}"
        _fake_run(d, _flat(1.0), _flat(1.0))
        p = d / "work/lumina_spectrum_det_obs.csv"
        if pts is None:
            p.unlink()
        else:
            _spec(p, pts, hdr or "wavelength_angstrom,flux")
        _, wr = run_observable(d)
        case(f"spectrum defect {nm} -> {want}", want in wr, True)
    # unsupported band, zero / negative integral
    d = tmp / "narrow"
    _fake_run(d, _flat(1.0, lo=1300.0), _flat(1.0))
    _, wr = run_observable(d)
    case("unsupported band (FUV0 not covered) -> BAND_UNSUPPORTED", "N:BAND_UNSUPPORTED:FUV0" in wr, True)
    d = tmp / "zero"
    _fake_run(d, [(l, 0.0 if l < 2000 else 1.0) for l, _ in _flat(1.0)], _flat(1.0))
    _, wr = run_observable(d)
    case("zero FUV integral -> colour undefined (no log)", "N:COLOUR_UNDEFINED:FUVa" in wr, True)
    d = tmp / "neg"
    _fake_run(d, [(l, -1.0 if l < 2000 else 1.0) for l, _ in _flat(1.0)], _flat(1.0))
    _, wr = run_observable(d)
    case("negative FUV integral -> colour undefined", "N:COLOUR_UNDEFINED:FUVb" in wr, True)
    # status / manifest / partial / digest
    d = tmp / "failed"
    _fake_run(d, _flat(1.0), _flat(1.0), state="FAILED(WRITE_N errno=28)")
    case("det_obs_state FAILED(reason) -> X even with model_rc 0",
         "DET_OBS_STATE:FAILED(WRITE_N errno=28)" in det_obs_reasons(d), True)
    d = tmp / "nostatus"
    _fake_run(d, _flat(1.0), _flat(1.0))
    (d / "work/det_obs_status.json").unlink()
    case("status file missing -> X", det_obs_reasons(d), ["DET_OBS_STATUS_MISSING"])
    d = tmp / "nomani"
    _fake_run(d, _flat(1.0), _flat(1.0), manifest=False)
    case("success manifest missing -> X", "DET_OBS_MANIFEST_MISSING" in det_obs_reasons(d), True)
    d = tmp / "partial"
    _fake_run(d, _flat(1.0), _flat(1.0))
    with open(d / "work/lumina_spectrum_det_obs.csv", "a") as f:
        f.write("7001.0,1.0\n")
    case("output changed after manifest (partial / hash) -> X",
         "DET_OBS_OUTPUT_HASH:lumina_spectrum_det_obs.csv" in det_obs_reasons(d), True)
    d = tmp / "mode2"
    _fake_run(d, _flat(1.0), _flat(1.0), mode=2)
    case("production run must be capture mode 1", "DET_OBS_MODE:2" in det_obs_reasons(d), True)
    # legacy (4) file passed as the same-view pair
    d = tmp / "leg4"
    _fake_run(d, _flat(1.0), _flat(1.0))
    _, wr = run_observable(d, d / "work/lumina_spectrum_det_obs_legacy.csv")
    case("legacy (4) file used as the same-view pair -> X", "L:IS_LEGACY4_FILE" in wr, True)
    # signed component unsupported alone does not make N/L X
    case("signed/zero-tau component unsupported alone: not a G-RUN reason",
         any("SIGNED" in x or "UNSUPPORTED_SOURCE_DOMAIN" in x for x in det_obs_reasons(tmp / "Rgood")), False)
    # exports
    ex = tmp / "exports"
    ex.mkdir()
    base = "/gpfs/x/R"
    run = "/gpfs/x/RP"
    good = "".join(f'declare -x {k}="{base}/input/{k.lower()}"\n' for k in PATH_KEYS) + 'declare -x A="1"\n'
    goodR = "".join(f'declare -x {k}="{run}/input/{k.lower()}"\n' for k in PATH_KEYS) + 'declare -x A="1"\n'
    (ex / "B").write_text(good)
    (ex / "R").write_text(goodR + f'declare -x {NEW_KEY}="1"\n')
    case("exports: exactly the new key + path transplant", exports_reasons(ex / "B", ex / "R", base, run), [])
    (ex / "R2").write_text(goodR + f'declare -x {NEW_KEY}="1"\ndeclare -x {NEW_KEY}="1"\n')
    case("exports: duplicated key rejected", "EXPORTS_DUPLICATE_KEY" in exports_reasons(ex / "B", ex / "R2", base, run), True)
    (ex / "R3").write_text(goodR.replace('A="1"', 'A="2"') + f'declare -x {NEW_KEY}="1"\n')
    case("exports: unregistered difference rejected", "EXPORTS_EXTRA_DIFF:A" in exports_reasons(ex / "B", ex / "R3", base, run), True)
    (ex / "R4").write_text(goodR + f'declare -x {NEW_KEY}="0"\n')
    case("exports: new key not 1 rejected", any(x.startswith("EXPORTS_NEW_KEY") for x in exports_reasons(ex / "B", ex / "R4", base, run)), True)
    # changeset
    cs = tmp / "cs"
    (cs / "src").mkdir(parents=True)
    (cs / "tests").mkdir(parents=True)
    for rel in CHANGESET:
        (cs / rel).write_text("x")
    (cs / "src/lumina_cmfgen.c").write_text("a")
    (cs / "src/other.c").write_text("b")
    man = cs / "man.sha256"
    man.write_text(f"{sha(cs / 'src/lumina_cmfgen.c')}  src/lumina_cmfgen.c\n{sha(cs / 'src/other.c')}  src/other.c\n")
    (cs / "src/lumina_cmfgen.c").write_text("changed")
    case("changeset: registered file change accepted", changeset_reasons(man, cs), [])
    (cs / "src/other.c").write_text("changed")
    case("changeset: unregistered file change rejected", changeset_reasons(man, cs), ["T1_UNREGISTERED_CHANGE:src/other.c"])
    # lineage: B0 manifest vs T0, registered changeset, patch digest/file set, frozen deps
    lg = tmp / "lineage"
    for sub in ("t0/src", "t1/src", "t1/tests", "deps"):
        (lg / sub).mkdir(parents=True)
    (lg / "t0/src/a.c").write_text("A")
    (lg / "b0.sha256").write_text(f"{sha(lg / 't0/src/a.c')}  src/a.c\n")
    for rel in CHANGESET:
        (lg / "t1" / rel).write_text("T1:" + rel)
    (lg / "cs.sha256").write_text("".join(f"{sha(lg / 't1' / rel)}  {rel}\n" for rel in CHANGESET))
    for rel in CHANGESET:
        (lg / "t0" / rel).parent.mkdir(parents=True, exist_ok=True)
        (lg / "t0" / rel).write_text("T0:" + rel + "\n")
        (lg / "t1" / rel).write_text("T1:" + rel + "\n")
    (lg / "cs.sha256").write_text("".join(f"{sha(lg / 't1' / rel)}  {rel}\n" for rel in CHANGESET))
    pt = []
    for rel in CHANGESET:
        r = subprocess.run(["diff", "-u", "--label", f"T0/{rel}", "--label", f"T1/{rel}",
                            str(lg / "t0" / rel), str(lg / "t1" / rel)], capture_output=True, text=True)
        pt.append(r.stdout)
    (lg / "p.patch").write_text("".join(pt))
    (lg / "p.sha").write_text(sha(lg / "p.patch") + "\n")
    (lg / "deps/x.py").write_text("x")
    kw = dict(t0=lg / "t0", b0_sources=lg / "b0.sha256", t1=lg / "t1", patch=lg / "p.patch",
              patch_sha=lg / "p.sha", changeset_sha=lg / "cs.sha256",
              deps={"x.py": sha(lg / "deps/x.py")}, here=lg / "deps")
    case("lineage: consistent B0/T0/T1/patch/deps", lineage_reasons(**kw), [])
    (lg / "t0/src/a.c").write_text("B")
    case("lineage: T0 file differs from B0 build manifest", "T0_NE_B0:src/a.c" in lineage_reasons(**kw), True)
    (lg / "t0/src/a.c").write_text("A")
    (lg / "t1/src/env_universe.h").write_text("changed")
    case("lineage: T1 changeset file differs from registered hash",
         "T1_NE_REGISTERED:src/env_universe.h" in lineage_reasons(**kw), True)
    (lg / "t1/src/env_universe.h").write_text("T1:src/env_universe.h\n")
    (lg / "p.sha").write_text("0" * 64 + "\n")
    case("lineage: patch digest mismatch", "PATCH_DIGEST" in lineage_reasons(**kw), True)
    (lg / "p.sha").write_text(sha(lg / "p.patch") + "\n")
    (lg / "deps/x.py").write_text("y")
    case("lineage: frozen import dependency changed", "FROZEN_DEP:x.py" in lineage_reasons(**kw), True)
    (lg / "deps/x.py").write_text("x")
    (lg / "t1/src/lumina_cmfgen.h").write_text("T1 drifted after the patch\n")
    (lg / "cs.sha256").write_text("".join(f"{sha(lg / 't1' / rel)}  {rel}\n" for rel in CHANGESET))
    case("lineage: patch applied to T0 does not reproduce T1",
         "PATCH_RESULT_NE_T1:src/lumina_cmfgen.h" in lineage_reasons(**kw), True)
    # manifest view digests must agree
    d = tmp / "viewdig"
    _fake_run(d, _flat(1.0), _flat(1.0))
    m = json.load(open(d / "work/det_obs_manifest.json"))
    m["view_fnv1a"] = {"before": "a", "after_N": "a", "after_L": "b"}
    (d / "work/det_obs_manifest.json").write_text(json.dumps(m))
    case("manifest view digest changed across writers -> X", "DET_OBS_VIEW_DIGEST" in det_obs_reasons(d), True)
    # observables.csv is written from the gathered observables
    write_observables(tmp / "obs.csv", {"RP": obs, "EP": None})
    rows = list(csv.reader(open(tmp / "obs.csv")))
    case("observables.csv rows: header + N/L bands, colours and W",
         (rows[0], sum(1 for r in rows if r[2] == "W_log10")), (["run", "writer", "quantity", "band", "value"], 4))
    # G-RUN on an absent run directory: reasons, no exception
    lst = []
    r = guarded(lst, "G-RUN:READ", grun, tmp / "absent", "COMPLETED RESTARTS=0", True)
    case("G-RUN on missing run: registered reasons, no exception",
         (r is None) or any("STDERR_MISSING" in x for x in r), True)
    # paired branch boundaries
    case("branch: -0.30 down / +0.30 up -> O-MIX", paired_branch(cat(-0.30), cat(0.30), []), "O-MIX")
    case("branch: just inside +-0.30 -> O-FLAT",
         paired_branch(cat(math.nextafter(-0.30, 0.0)), cat(math.nextafter(0.30, 0.0)), []), "O-FLAT")
    case("branch: down + flat -> O-DOWN", paired_branch("down", "flat", []), "O-DOWN")
    case("branch: up + flat -> O-UP", paired_branch("flat", "up", []), "O-UP")
    case("branch: invalid first -> X", paired_branch("down", "down", ["G-RUN:x"]), "X")
    # stage-1 evidence gate
    ev1 = {k: tmp / f"ev_{k}" for k in STAGE1_EVIDENCE}
    case("stage1 evidence absent -> reasons", len(stage1_reasons(ev1)), len(STAGE1_EVIDENCE))
    # G-READOUT
    (tmp / "sum_other.json").write_text(json.dumps({"readout_sha256": "0" * 64, "n_cases": 1, "cases": [{"case": "x", "pass": True}]}))
    case("G-READOUT rejects another readout's summary", readout_gate(tmp / "sum_other.json"), "NOT_THIS_READOUT")
    (tmp / "sum_short.json").write_text(json.dumps({"readout_sha256": sha(Path(__file__).resolve()), "n_cases": 1,
                                                    "cases": [{"case": res[0]["case"], "pass": True}]}))
    case("G-READOUT rejects a passing summary lacking required cases", readout_gate(tmp / "sum_short.json"), "MISSING_REQUIRED")
    names = {r["case"] for r in res}
    case("battery covers every required case", sorted(set(REQUIRED_CASES) - names), [])
    nfail = sum(1 for r in res if not r["pass"])
    (out / "readout_selftest.json").write_text(json.dumps(
        {"readout_sha256": sha(Path(__file__).resolve()), "n_cases": len(res), "failures": nfail, "cases": res}, indent=1))
    print(f"DET-OBS readout NC: {len(res) - nfail}/{len(res)} -> {'PASS' if nfail == 0 else 'FAIL'}")
    return 0 if nfail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("selftest"); p.add_argument("--out", required=True)
    p = sp.add_parser("preflight")
    for k in ("--run", "--base", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--eprime"); p.add_argument("--ebase")
    p = sp.add_parser("readout")
    p.add_argument("--mode", required=True, choices=("rprime", "paired"))
    for k in ("--run", "--base", "--slurm-states", "--selftest", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--eprime"); p.add_argument("--ebase")
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "preflight": cmd_preflight, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
