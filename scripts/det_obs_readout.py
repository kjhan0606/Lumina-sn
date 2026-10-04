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
import re
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
B1 = N_ROOT / "cuda_build2"   # prereg rev 4 (run1 used cuda_build1)
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
PATCH = N_ROOT / "evidence/rev4/det_obs_writer.patch"
PATCH_SHA = N_ROOT / "evidence/rev4/det_obs_writer.patch.sha256"
CHANGESET_SHA = N_ROOT / "evidence/rev4/T1_changeset.sha256"
FROZEN_DEPS = {"nrb_e_readout.py": "5b5a68f9475aa3c1b035d621b7a807566df8f5ecee5c6eb3bac9162da544c258",
               "nt_eta_readout.py": "a4ae3f3ae31f5b28910204bbfe1bbf5ef99292b9b06c1c53a866be08ed20ce3f",
               "w5_cont_readout.py": "7008f709e4ca49a24823e4f88f452c42448247d8cb5940c1c1670c4e31a7f3cd"}
STAGE1_EVIDENCE = {"selftest": N_ROOT / "stage1/rev4/st_final/cases.json",
                   "nc": N_ROOT / "stage1/rev4/nc_final.log",
                   "off": N_ROOT / "stage1/rev4/off_final.log",
                   "audit": N_ROOT / "evidence/rev5/code_audit_closure.txt",
                   "cost": N_ROOT / "evidence/rev5/stage2_cost_estimate.json",
                   "user_rprime": N_ROOT / "evidence/rev4/USER_DECISION_2026-10-04.txt",
                   "user_h200": N_ROOT / "evidence/rev4/USER_DECISION_H200_2026-10-05.txt"}
AUDITS = (N_ROOT / "evidence/rev4/sol_audit.md", N_ROOT / "evidence/rev4/sol_audit_r2.md",
          N_ROOT / "evidence/rev5/sol_audit.md")
# prereg rev 5 (§3, §3.3, §4.4): run2 executes the registered h200x2 profile
RUNNER_TESTED = N_ROOT / "evidence/rev5/runner_tested.txt"
RUNNER_TESTED_EP = N_ROOT / "evidence/rev5/runner_tested_ep.txt"
READOUT_SELFTEST = N_ROOT / "evidence/rev5/readout_st/readout_selftest.json"
R_RUNNER = Path("/gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/run1/job.slurm")
RP_JOB_NAME = "LUMINA_detobs_Rp2"
EP_JOB_NAME = "LUMINA_detobs_Ep1"
H200X2 = {"partition": "h200", "gpu_kind": "H200", "gpu_profile": "h200x2",
          "allocated_gpus": "2", "allocated_cpus": "16"}
H200_MIN_MIB = 140000
RUNNER_SBATCH = ("#SBATCH --partition=h200", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                 "#SBATCH --gres=gpu:H200:2", "#SBATCH --cpus-per-task=16", "#SBATCH --mem=64G",
                 "#SBATCH --time=30:00:00", "#SBATCH --no-requeue")

COST_LIMIT_H = 30.0
USER_DECISION_DATE = "2026-10-04"


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
    why += footer_core_reasons(footer(run), slurm_state)
    why += resource_reasons(run, slurm_state)
    ft = footer(run)
    ex = Path(run) / "input/resolved_lumina.exports"
    if not ex.exists() or ft.get("effective_exports_sha256") != sha(ex):
        why.append("FOOTER_EXPORTS_HASH")
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


def changeset_reasons(t0_manifest, t1_root, t0_root):
    """T1 differs from T0 only in the registered changeset files (+ new test file);
    every T1 file outside the manifest must be the same file in T0 or registered."""
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
    for q in sorted(Path(t1_root).rglob("*")):
        if not q.is_file():
            continue
        p = q.relative_to(t1_root).as_posix()
        if p in base or p in CHANGESET:
            continue
        o = Path(t0_root) / p
        if not o.is_file() or sha(o) != sha(q):
            why.append(f"T1_UNREGISTERED_ADD:{p}")
    return why


def audit_blocking_count(path):
    """the audit's own verdict line '«폐합 차단» N건' (None if absent/unreadable)."""
    try:
        m = re.search(r"«폐합 차단»\s*(\d+)\s*건", open(path, encoding="utf-8").read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def audit_closure_reasons(path, audits=AUDITS):
    """closure: 'audit=<path> sha256=<hex> blocking=<n> resolved=<n>' per registered
    audit; per blocking item k = 1..n one line 'item=<audit path>#<k> status=resolved
    finding_line=<L> evidence=<path> evidence_sha256=<hex>' where line L of the audit
    carries «폐합 차단» (distinct L and distinct evidence per item, evidence hash
    matches); and 'blocking_open=0'.  n is read from the audit's own verdict line."""
    why, seen, items, opened = [], {}, {}, None
    try:
        for ln in open(path):
            ln = ln.strip()
            if ln.startswith("audit="):
                kv = dict(x.split("=", 1) for x in ln.split())
                seen[kv["audit"]] = kv
            elif ln.startswith("item="):
                kv = dict(x.split("=", 1) for x in ln.split())
                items[kv["item"]] = kv
            elif ln.startswith("blocking_open="):
                opened = ln.split("=", 1)[1]
    except (OSError, ValueError, KeyError):
        return ["AUDIT_CLOSURE_UNREADABLE"]
    for a in audits:
        nm = Path(a).name
        kv = seen.get(str(a))
        if kv is None:
            why.append(f"AUDIT_NOT_CLOSED:{nm}")
            continue
        if not Path(a).exists() or sha(a) != kv.get("sha256"):
            why.append(f"AUDIT_DIGEST:{nm}")
        n = audit_blocking_count(a) if Path(a).exists() else None
        if n is None:
            why.append(f"AUDIT_VERDICT_UNPARSEABLE:{nm}")
            continue
        if kv.get("blocking") != str(n) or kv.get("resolved") != str(n):
            why.append(f"AUDIT_UNRESOLVED:{nm}")
        lines = open(a, encoding="utf-8").read().split("\n")
        used_l, used_e = set(), set()
        for k in range(1, n + 1):
            it = items.get(f"{a}#{k}")
            ok = it is not None and it.get("status") == "resolved"
            if ok:
                L, evp = it.get("finding_line", ""), it.get("evidence", "")
                ok = (L.isdigit() and 1 <= int(L) <= len(lines) and "«폐합 차단»" in lines[int(L) - 1]
                      and L not in used_l and evp not in used_e and Path(evp).is_file()
                      and sha(evp) == it.get("evidence_sha256"))
                used_l.add(L)
                used_e.add(evp)
            if not ok:
                why.append(f"AUDIT_ITEM_UNRESOLVED:{nm}:{k}")
    if opened != "0":
        why.append("AUDIT_BLOCKING_OPEN")
    return why


def cost_reasons(path):
    """prereg §3.3: 2 T_R + T_add <= 30 h and a positive memory margin."""
    try:
        d = json.load(open(path))
        tr = float(d["T_R_hours_conservative"])
        ta = float(d["T_add_hours_with_2x_margin"])
        tr_meas = max(float(d["T_R_hours"]), float(d["run1_measured"]["elapsed_hours"]))
        rss_gb = float(d["run1_measured"]["MaxRSS_K"]) / 1048576.0
        add_gb = float(d["memory_added_rev4"]["bytes"]) / 1073741824.0
        ta_basis = 2.0 * (float(d["N_hours_range"][1]) + float(d["L_hours_upper"]))
        m = float(d["margin_GB_vs_64G_rev5"])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return ["COST_UNREADABLE"]
    why = []
    if not all(math.isfinite(x) for x in (tr, ta, tr_meas, rss_gb, add_gb, ta_basis, m)):
        return ["COST_NONFINITE"]
    if tr < 0.0 or ta < 0.0 or rss_gb <= 0.0 or add_gb < 0.0:
        why.append("COST_NEGATIVE")
    if ta < ta_basis:
        why.append("COST_T_ADD_BELOW_BASIS")
    if tr < tr_meas:
        why.append("COST_T_R_BELOW_MEASURED")
    if not (2.0 * tr + ta <= COST_LIMIT_H):
        why.append("COST_OVER_LIMIT")
    if abs(m - (64.0 - rss_gb - add_gb)) > 0.1 or not (m > 0.0):
        why.append("COST_MEMORY_MARGIN")
    return why


def stage1_reasons(ev=STAGE1_EVIDENCE, audits=AUDITS):
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
    if Path(ev["audit"]).exists():
        why += [f"STAGE1_{x}" for x in audit_closure_reasons(ev["audit"], audits)]
    if Path(ev["cost"]).exists():
        why += [f"STAGE1_{x}" for x in cost_reasons(ev["cost"])]
    if Path(ev["user_rprime"]).exists() and USER_DECISION_DATE not in open(ev["user_rprime"]).read():
        why.append("STAGE1_USER_DECISION_NOT_RECORDED")
    if Path(ev["user_h200"]).exists() and "h200x2" not in open(ev["user_h200"]).read():
        why.append("STAGE1_USER_H200_NOT_RECORDED")
    return why


# ------------------------------------------------------------------ rev 5: runner / resources
RUNNER_CASES = """  run_case h200x1_ok h200 0 6 GPU-a h200 h200x1 1 pass
  run_case h200x2_ok h200 0 6 GPU-a,GPU-b h200 h200x2 2 pass
  run_case h100x2_ok good 0 6 GPU-a,GPU-b h100 h100-a100x2 2 pass
  run_case a100x2_ok a100 0 6 GPU-a,GPU-b a100 h100-a100x2 2 pass
  run_case h200x2_one_gpu h200 0 6 GPU-a h200 h200x2 2 fail VISIBLE_GPU_MASK_INVALID
  run_case h200x2_wrong_kind good 0 6 GPU-a,GPU-b h200 h200x2 2 fail GPU_KIND_PROFILE_MISMATCH
  run_case h200x2_low_memory h200_lowmem 0 6 GPU-a,GPU-b h200 h200x2 2 fail GPU_MEMORY_BELOW_PROFILE_MINIMUM
  run_case h200x2_wrong_partition h200 0 6 GPU-a,GPU-b h100 h200x2 2 fail PARTITION_NOT_QUALIFIED
  run_case h200x2_duplicate_uuid h200dup 0 6 GPU-a,GPU-b h200 h200x2 2 fail DUPLICATE_GPU_UUID
  CASE_CPUS=32 run_case h200x2_wrong_cpus h200 0 6 GPU-a,GPU-b h200 h200x2 2 fail ALLOCATION_SHAPE_MISMATCH
  run_case h200x2_mgpu_one h200 0 6 GPU-a,GPU-b h200 h200x2 1 fail GPU_OWNER_COUNT_MISMATCH
  run_case a100_on_h100_reject a100 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail GPU_PARTITION_MISMATCH
  run_case h100_low_memory lowmem 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail GPU_MEMORY_BELOW_PROFILE_MINIMUM
  run_case h200_low_memory h200_lowmem 0 6 GPU-a h200 h200x1 1 fail GPU_MEMORY_BELOW_PROFILE_MINIMUM
  run_case a100_40gb_low_memory a100_lowmem 0 6 GPU-a,GPU-b a100 h100-a100x2 2 fail GPU_MEMORY_BELOW_PROFILE_MINIMUM
  run_case duplicate_uuid duplicate 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail DUPLICATE_GPU_UUID
  run_case mixed_kind mixed 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail GPU_KIND_PROFILE_MISMATCH
  run_case h200_two_gpus h200 0 6 GPU-a,GPU-b h200 h200x1 1 fail VISIBLE_GPU_MASK_INVALID
  run_case h100_one_gpu good 0 6 GPU-a h100 h100-a100x2 2 fail VISIBLE_GPU_MASK_INVALID
  run_case wrong_partition good 0 6 GPU-a,GPU-b h200 h100-a100x2 2 fail PARTITION_NOT_QUALIFIED
  run_case exports_count_mismatch h200 0 6 GPU-a h200 h200x1 2 fail GPU_OWNER_COUNT_MISMATCH
  run_case model_failure good 9 6 GPU-a,GPU-b h100 h100-a100x2 2 fail MODEL_EXIT_NONZERO
  run_case missing_consumption good 0 5 GPU-a,GPU-b h100 h100-a100x2 2 fail FIXED_TE_CONSUMPTION_EVIDENCE_MISSING
  run_case public_t_mismatch good 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail PUBLIC_TE_PROFILE_MISMATCH
  run_case duplicate_profile_shell good 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail PUBLIC_TE_PROFILE_MISMATCH
  run_case manifest_hash_mismatch good 0 6 GPU-a,GPU-b h100 h100-a100x2 2 fail FIXED_TE_CONSUMPTION_EVIDENCE_MISSING
"""


def runner_rev5(base_text, job_name=RP_JOB_NAME):
    """The registered run2/E' runner = R runner + exactly the prereg §3 rev 5 edits:
    (1) job name (2) SBATCH partition/gres (3) GPU_PROFILE (4) h200x2 case
    (5) runner_selftest repair + cases.  Single source: the staged runner is produced
    by this function and the readout regenerates it to compare whole files.
    Returns None when an anchor is not found exactly once."""
    t = base_text
    jn = re.findall(r"(?m)^#SBATCH --job-name=.*\n", t)
    if len(jn) != 1:
        return None
    sel_a = t.find("  run_case h200x1_ok ")
    sel_b = t.find('  set +e\n  PATH="$tmp/h100x2_ok/bin:$PATH"')
    if sel_a < 0 or sel_b < sel_a:
        return None
    t = t[:sel_a] + RUNNER_CASES + t[sel_b:]
    edits = (
        (jn[0], f"#SBATCH --job-name={job_name}\n"),
        ("#SBATCH --partition=h100,a100\n", "#SBATCH --partition=h200\n"),
        ("#SBATCH --gres=gpu:2\n", "#SBATCH --gres=gpu:H200:2\n"),
        ("readonly GPU_PROFILE=h100-a100x2\n", "readonly GPU_PROFILE=h200x2\n"),
        ("    h200x1) expected_partition=h200; expected_gpu_count=1; expected_min_memory=140000; "
         "expected_kinds=H200 ;;\n",
         "    h200x1) expected_partition=h200; expected_gpu_count=1; expected_min_memory=140000; "
         "expected_kinds=H200 ;;\n    h200x2) expected_partition=h200; expected_gpu_count=2; "
         "expected_min_memory=140000; expected_kinds=H200 ;;\n"),
        ("""'declare -x LUMINA_PURE_CMFGEN_ITER="2"'""", """'declare -x LUMINA_PURE_CMFGEN_ITER="6"'"""),
        ('h200:GPU-b) echo "NVIDIA H200, 143771, GPU-bbbb, 555.1, 9.0";;',
         'h200:GPU-b) echo "NVIDIA H200, 143771, GPU-bbbb, 555.1, 9.0";; '
         'h200dup:*) echo "NVIDIA H200, 143771, GPU-aaaa, 555.1, 9.0";;'),
        ('mgpu_devices="$8" expect="$9" root\n', 'mgpu_devices="$8" expect="$9" want="${10:-}" root\n'),
        ("""    if [[ "$profile" == h100-a100x2 ]]; then
      sed -i 's/^#SBATCH --partition=h200$/#SBATCH --partition=h100,a100/; s/^#SBATCH --gres=gpu:1$/#SBATCH --gres=gpu:2/; s/^readonly GPU_PROFILE=h200x1$/readonly GPU_PROFILE=h100-a100x2/' "$root/test_runner.sh"
    fi
""", """    [[ "$(grep -c '^readonly GPU_PROFILE=' "$root/test_runner.sh")" -eq 1 ]] || die "SELFTEST_PROFILE_LINE_NOT_UNIQUE"
    sed -i "s/^readonly GPU_PROFILE=.*\\$/readonly GPU_PROFILE=$profile/" "$root/test_runner.sh"
    [[ "$(grep -c "^readonly GPU_PROFILE=$profile\\$" "$root/test_runner.sh")" -eq 1 ]] || die "SELFTEST_PROFILE_NOT_SET case=$name"
"""),
        ('SLURM_JOB_PARTITION="$partition" SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_CPUS_PER_TASK=32 SLURM_RESTART_COUNT=0 \\',
         'SLURM_JOB_PARTITION="$partition" SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_CPUS_PER_TASK="${CASE_CPUS:-16}" SLURM_RESTART_COUNT=0 \\'),
        ("""      [[ "$rc" -ne 0 ]] || die "SELFTEST_NEGATIVE_ACCEPTED case=$name"
""", """      [[ "$rc" -ne 0 ]] || die "SELFTEST_NEGATIVE_ACCEPTED case=$name"
      [[ -n "$want" ]] && grep -q "CMFGEN_TWIN_MAP_RUN_FAIL reason=$want" "$root/test.log" || \\
        { sed -n '1,40p' "$root/test.log" >&2; die "SELFTEST_NEGATIVE_WRONG_REASON case=$name want=$want"; }
"""),
        ("""SLURM_JOB_PARTITION=h100 SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_CPUS_PER_TASK=32 SLURM_RESTART_COUNT=0 \\
    bash "$tmp/h100x2_ok/test_runner.sh" "$tmp/h100x2_ok" >/dev/null 2>&1
  rc=$?; set -e; [[ "$rc" -ne 0 ]] || die "SELFTEST_RERUN_ACCEPTED";""",
         """SLURM_JOB_PARTITION=h100 SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_CPUS_PER_TASK=16 SLURM_RESTART_COUNT=0 \\
    bash "$tmp/h100x2_ok/test_runner.sh" "$tmp/h100x2_ok" > "$tmp/rerun.log" 2>&1
  rc=$?; set -e; [[ "$rc" -ne 0 ]] || die "SELFTEST_RERUN_ACCEPTED"
  grep -q "CMFGEN_TWIN_MAP_RUN_FAIL reason=RUN_ALREADY_ATTEMPTED" "$tmp/rerun.log" || die "SELFTEST_RERUN_WRONG_REASON";"""))
    for o, n in edits:
        if t.count(o) != 1:
            return None
        t = t.replace(o, n)
    return t


def submit_expected(run_root):
    return f"cd {run_root} && sbatch --exclude=syn102 job.slurm {run_root}\n"


def runner_reasons(base_runner, run_runner, tested=RUNNER_TESTED, submit=None, job_name=RP_JOB_NAME,
                   run_root=None):
    """whole runner == runner_rev5(R runner); tested identity; SBATCH resources;
    submit command exactly submit_expected(run_root)."""
    why = []
    try:
        bt, rt = Path(base_runner).read_text(), Path(run_runner).read_text()
    except OSError:
        return ["RUNNER_UNREADABLE"]
    exp = runner_rev5(bt, job_name)
    if exp is None or rt != exp:
        why.append("RUNNER_UNREGISTERED_DIFF")
    lines = set(rt.splitlines())
    for ln in RUNNER_SBATCH:
        if ln not in lines:
            why.append(f"RUNNER_RESOURCE:{ln.split()[1]}")
    try:
        kv = dict(x.split("=", 1) for x in Path(tested).read_text().split())
    except (OSError, ValueError):
        kv = {}
    if kv.get("runner_sha256") != sha(run_runner):
        why.append("RUNNER_NOT_TESTED")
    log = Path(tested).parent / "runner_selftest.log"
    if not log.exists() or kv.get("selftest_log_sha256") != sha(log) or \
            "CMFGEN_TWIN_MAP_RUNNER_SELFTEST_PASS" not in log.read_text():
        why.append("RUNNER_SELFTEST_NOT_PASS")
    if submit is not None:
        try:
            cmd = Path(submit).read_text()
        except OSError:
            cmd = None
        if cmd is None:
            why.append("SUBMIT_COMMAND_MISSING")
        elif run_root is None or cmd != submit_expected(run_root):
            why.append("SUBMIT_COMMAND_NOT_REGISTERED")
    return why


def footer_core_reasons(ft, slurm_state):
    """frozen nt_eta footer_reasons minus its H100/A100-only GPU line, which the
    registered h200x2 check (resource_reasons) replaces (prereg §4.3 rev 5)."""
    why = []
    if ft.get("terminal_state") != "SUCCESS":
        why.append("FOOTER_NOT_SUCCESS")
    if ft.get("model_rc") != "0":
        why.append(f"MODEL_RC:{ft.get('model_rc')}")
    if ft.get("outer_iterations") != "6":
        why.append(f"OUTER_ITERATIONS:{ft.get('outer_iterations')}")
    if ft.get("fixed_te_loader_consumed") != "1":
        why.append("FIXED_TE_NOT_CONSUMED")
    if slurm_state is None or "COMPLETED" not in slurm_state.split():
        why.append(f"SLURM_STATE:{slurm_state}")
    if slurm_state is not None and "RESTARTS=0" not in slurm_state.split():
        why.append("SLURM_RESTARTS")
    return why


def resource_reasons(run, slurm_state):
    """footer (partition, gpu_kind, gpu_profile, allocated_gpus, allocated_cpus) ==
    registered h200x2; Slurm allocation and gpu_identity.csv agree (prereg §4.4)."""
    why = []
    ft = footer(run)
    for k, v in H200X2.items():
        if ft.get(k) != v:
            why.append(f"FOOTER_RESOURCE:{k}")
    toks = (slurm_state or "").split()
    if "PARTITION=h200" not in toks:
        why.append("SLURM_PARTITION")
    alloc = next((t[len("ALLOC="):] for t in toks if t.startswith("ALLOC=")), "")
    parts = set(alloc.split(","))
    if "cpu=16" not in parts or "gres/gpu=2" not in parts:
        why.append("SLURM_ALLOCATION")
    try:
        rows = list(csv.DictReader(open(Path(run) / "gpu_identity.csv")))
    except OSError:
        rows = []
    uu = [r.get("uuid", "").strip() for r in rows]
    if len(rows) != 2 or len(set(uu)) != 2 or not all(u.startswith("GPU-") for u in uu):
        why.append("GPU_IDENTITY_COUNT_UUID")
    for r in rows:
        mem = r.get("memory_total_mib", "").strip()
        if "H200" not in r.get("name", "") or not mem.isdigit() or int(mem) < H200_MIN_MIB:
            why.append("GPU_IDENTITY_KIND_MEMORY")
            break
    if rows and ft.get("cuda_visible_devices") != ",".join(r.get("cuda_visible_token", "") for r in rows):
        why.append("GPU_IDENTITY_FOOTER_MASK")
    return why


def module_env_norm(path):
    """module list + ldd resolution with the per-process load addresses removed."""
    return [re.sub(r"\s*\(0x[0-9a-f]+\)", "", ln.rstrip()) for ln in open(path)]


def paired_resource_reasons(rp, ep):
    """E' must match R' run2: footer resources, binary and module environment."""
    a, b = footer(rp), footer(ep)
    why = [f"PAIRED_RESOURCE:{k}" for k in list(H200X2) + ["binary_sha256"] if a.get(k) != b.get(k)]
    ma, mb = Path(rp) / "module_env.txt", Path(ep) / "module_env.txt"
    if not (ma.exists() and mb.exists()) or module_env_norm(ma) != module_env_norm(mb):
        why.append("PAIRED_RESOURCE:module_env")
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


def gid(base, run, post_run, b1=B1, job_name=RP_JOB_NAME):
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
    why += changeset_reasons(N_ROOT / "T0_manifest.sha256", T1, T0)
    why += lineage_reasons()
    for f in INPUT_SAME:
        if sha(base / "input" / f) != sha(run / "input" / f):
            why.append(f"INPUT_MISMATCH:{f}")
    why += exports_reasons(base / "input/resolved_lumina.exports", run / "input/resolved_lumina.exports",
                           str(base), str(run))
    why += linkage_reasons(run, "RP", False)
    why += runner_reasons(base / "job.slurm", run / "job.slurm",
                          tested=RUNNER_TESTED if job_name == RP_JOB_NAME else RUNNER_TESTED_EP,
                          submit=run / "SUBMIT_COMMAND.txt", job_name=job_name, run_root=str(run))
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
    "stage1 audit closure: closed accepted; digest, unresolved, open rejected",
    "stage1 cost: 2T_R+T_add over 30 h rejected",
    "stage1 audit closure: self-declared zero, missing item, absent evidence, empty audit rejected",
    "stage1 cost: negative time, T_R below measured, unfounded memory margin rejected",
    "stage1 user decision without the dated record rejected",
    "changeset: unregistered added file rejected",
    "h200x2: healthy footer, Slurm allocation and gpu identity accepted",
    "h200x2: each footer resource field missing or wrong -> reason",
    "h200x2: Slurm allocation or gpu identity disagreeing -> reason",
    "runner: allowed h200x2 diff accepted, unregistered diff rejected",
    "runner: untested runner hash or failed selftest rejected",
    "runner: submit command overriding resources rejected",
    "paired: E'/R' resource mismatch -> reason",
    "stage1 user h200 decision without h200x2 rejected",
    "stage1 audit closure: item identity (shared evidence, non-blocking line, reused line, evidence hash) rejected",
    "stage1 cost: T_add below the bench basis or margin ignoring added buffers rejected",
    "paired: module environment compared without load addresses",
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
        gide = guarded(why_e, "G-ID:READ", gid, a.ebase, a.eprime, True, B1, EP_JOB_NAME)
        if gide:
            why_e += [f"G-ID:{x}" for x in gide[0]]
        why_e += [f"G-RUN:{x}" for x in (guarded(why_e, "G-RUN:READ", grun, a.eprime, st.get("EP"), False) or [])]
        obs_e, w2 = run_observable(a.eprime, Path(a.eprime) / LEGACY4)
        why_e += w2
        why_e += guarded(why_e, "PAIRED_RESOURCE:READ", paired_resource_reasons, a.run, a.eprime) or []
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
    g_read = readout_gate(a.selftest)
    if g_read != "PASS":
        why.append(f"G-READOUT:{g_read}")
    r = guarded(why, "G-ID:READ", gid, a.base, a.run, False)
    if r:
        why += r[0]
    if a.eprime or a.ebase:
        if not (a.eprime and a.ebase):
            raise SystemExit("paired preflight requires both --eprime and --ebase")
        re_ = guarded(why, "G-ID:EP:READ", gid, a.ebase, a.eprime, False, B1, EP_JOB_NAME)
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
    man = tmp / "cs_man.sha256"
    man.write_text(f"{sha(cs / 'src/lumina_cmfgen.c')}  src/lumina_cmfgen.c\n{sha(cs / 'src/other.c')}  src/other.c\n")
    t0f = tmp / "cs_t0"
    (t0f / "src").mkdir(parents=True)
    (t0f / "src/lumina_cmfgen.c").write_text("a")
    (t0f / "src/other.c").write_text("b")
    (cs / "src/lumina_cmfgen.c").write_text("changed")
    case("changeset: registered file change accepted", changeset_reasons(man, cs, t0f), [])
    (cs / "src/added.c").write_text("new")
    case("changeset: unregistered added file rejected", changeset_reasons(man, cs, t0f), ["T1_UNREGISTERED_ADD:src/added.c"])
    (cs / "src/added.c").unlink()
    (cs / "src/other.c").write_text("changed")
    case("changeset: unregistered file change rejected", changeset_reasons(man, cs, t0f), ["T1_UNREGISTERED_CHANGE:src/other.c"])
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
    au = tmp / "audit_a.md"
    au.write_text("총평: Q0 불가 · «폐합 차단» 2건.\n- «폐합 차단» first finding\n- «폐합 차단» second finding\n"
                  "- «기재로 족함» note\n")
    clo = tmp / "closure.txt"
    e1, e2 = tmp / "fix_evidence_1.txt", tmp / "fix_evidence_2.txt"
    e1.write_text("fix one")
    e2.write_text("fix two")
    it1 = f"item={au}#1 status=resolved finding_line=2 evidence={e1} evidence_sha256={sha(e1)}\n"
    it2 = f"item={au}#2 status=resolved finding_line=3 evidence={e2} evidence_sha256={sha(e2)}\n"
    good = f"audit={au} sha256={sha(au)} blocking=2 resolved=2\n" + it1 + it2 + "blocking_open=0\n"
    res_c = []
    def clo_case(text, audits=(au,)):
        clo.write_text(text)
        return audit_closure_reasons(clo, audits)
    res_c.append(clo_case(good) == [])
    res_c.append(clo_case(good.replace(sha(au), "0" * 64)) == ["AUDIT_DIGEST:audit_a.md"])
    res_c.append(clo_case(good.replace("resolved=2", "resolved=1")) == ["AUDIT_UNRESOLVED:audit_a.md"])
    res_c.append(clo_case(good.replace("blocking_open=0", "blocking_open=1")) == ["AUDIT_BLOCKING_OPEN"])
    res_c.append(clo_case("anything\n") == ["AUDIT_NOT_CLOSED:audit_a.md", "AUDIT_BLOCKING_OPEN"])
    case("stage1 audit closure: closed accepted; digest, unresolved, open rejected", res_c[:5], [True] * 5)
    res_c.append(clo_case(good.replace("blocking=2 resolved=2", "blocking=0 resolved=0")) == ["AUDIT_UNRESOLVED:audit_a.md"])
    res_c.append(clo_case(good.replace(it2, "")) == ["AUDIT_ITEM_UNRESOLVED:audit_a.md:2"])
    res_c.append(clo_case(good.replace(f"evidence={e2}", f"evidence={tmp}/absent")) == ["AUDIT_ITEM_UNRESOLVED:audit_a.md:2"])
    empty = tmp / "audit_empty.md"
    empty.write_text("")
    res_c.append(clo_case(f"audit={empty} sha256={sha(empty)} blocking=0 resolved=0\nblocking_open=0\n", (empty,))
                 == ["AUDIT_VERDICT_UNPARSEABLE:audit_empty.md"])
    case("stage1 audit closure: self-declared zero, missing item, absent evidence, empty audit rejected",
         res_c[5:], [True] * 4)
    idres = [clo_case(good.replace(f"evidence={e2} evidence_sha256={sha(e2)}", f"evidence={e1} evidence_sha256={sha(e1)}"))
             == ["AUDIT_ITEM_UNRESOLVED:audit_a.md:2"],
             clo_case(good.replace("finding_line=3", "finding_line=4")) == ["AUDIT_ITEM_UNRESOLVED:audit_a.md:2"],
             clo_case(good.replace("finding_line=3", "finding_line=2")) == ["AUDIT_ITEM_UNRESOLVED:audit_a.md:2"],
             clo_case(good.replace(f"evidence_sha256={sha(e2)}", "evidence_sha256=" + "0" * 64))
             == ["AUDIT_ITEM_UNRESOLVED:audit_a.md:2"]]
    case("stage1 audit closure: item identity (shared evidence, non-blocking line, reused line, evidence hash) rejected",
         idres, [True] * 4)
    co = tmp / "cost.json"
    rss_k, add_b = 32483132, 1480800
    base_c = {"T_R_hours": 11.61, "T_R_hours_conservative": 12.91, "T_add_hours_with_2x_margin": 2.6,
              "N_hours_range": [0.8, 1.2], "L_hours_upper": 0.073, "memory_added_rev4": {"bytes": add_b},
              "run1_measured": {"elapsed_hours": 12.91, "MaxRSS_K": rss_k},
              "margin_GB_vs_64G_rev5": 64.0 - rss_k / 1048576.0 - add_b / 1073741824.0}
    def cost_case(**kw):
        co.write_text(json.dumps(dict(base_c, **kw)))
        return cost_reasons(co)
    ok_c = cost_case()
    case("stage1 cost: 2T_R+T_add over 30 h rejected", [ok_c, cost_case(T_R_hours_conservative=14.0)],
         [[], ["COST_OVER_LIMIT"]])
    case("stage1 cost: negative time, T_R below measured, unfounded memory margin rejected",
         [cost_case(T_add_hours_with_2x_margin=-5.0), cost_case(T_R_hours_conservative=13.0, T_R_hours=13.5),
          cost_case(margin_GB_vs_64G_rev5=33.0 + 0.5)],
         [["COST_NEGATIVE", "COST_T_ADD_BELOW_BASIS"], ["COST_T_R_BELOW_MEASURED"], ["COST_MEMORY_MARGIN"]])
    case("stage1 cost: T_add below the bench basis or margin ignoring added buffers rejected",
         [cost_case(T_add_hours_with_2x_margin=2.0),
          cost_case(memory_added_rev4={"bytes": 4 * 1073741824})],
         [["COST_T_ADD_BELOW_BASIS"], ["COST_MEMORY_MARGIN"]])
    ev2 = {k: tmp / f"ev2_{k}" for k in STAGE1_EVIDENCE}
    ev2["user_rprime"].write_text("some other decision")
    case("stage1 user decision without the dated record rejected",
         "STAGE1_USER_DECISION_NOT_RECORDED" in stage1_reasons(ev2), True)
    # rev 5: h200x2 resources
    hr = tmp / "h200run"
    hr.mkdir(parents=True)
    (hr / "input").mkdir()
    (hr / "input/resolved_lumina.exports").write_text("declare -x A=\"1\"\n")
    gf = {"terminal_state": "SUCCESS", "model_rc": "0", "outer_iterations": "6", "fixed_te_loader_consumed": "1",
          "partition": "h200", "gpu_kind": "H200", "gpu_profile": "h200x2", "allocated_gpus": "2",
          "allocated_cpus": "16", "cuda_visible_devices": "0,1",
          "effective_exports_sha256": sha(hr / "input/resolved_lumina.exports")}
    def wfoot(d, f):
        (d / "RUN_FOOTER.txt").write_text("".join(f"{k}={v}\n" for k, v in f.items()))
    def wgpu(d, rows):
        (d / "gpu_identity.csv").write_text("visible_ordinal,cuda_visible_token,name,memory_total_mib,uuid,"
                                            "driver_version,compute_capability\n" + "".join(rows))
    good_rows = ["0,0,NVIDIA H200 NVL,143771,GPU-aaaa,555.1,9.0\n", "1,1,NVIDIA H200 NVL,143771,GPU-bbbb,555.1,9.0\n"]
    sl = "COMPLETED RESTARTS=0 PARTITION=h200 ALLOC=billing=40,cpu=16,gres/gpu=2,mem=64G,node=1"
    wfoot(hr, gf); wgpu(hr, good_rows)
    case("h200x2: healthy footer, Slurm allocation and gpu identity accepted",
         (footer_core_reasons(footer(hr), sl), resource_reasons(hr, sl)), ([], []))
    fres = []
    for k in H200X2:
        for bad in (None, "x"):
            f2 = dict(gf)
            if bad is None:
                f2.pop(k)
            else:
                f2[k] = bad
            wfoot(hr, f2)
            fres.append(f"FOOTER_RESOURCE:{k}" in resource_reasons(hr, sl))
    wfoot(hr, gf)
    case("h200x2: each footer resource field missing or wrong -> reason", fres, [True] * 10)
    sres = [resource_reasons(hr, sl.replace("PARTITION=h200", "PARTITION=h100")),
            resource_reasons(hr, sl.replace("cpu=16", "cpu=32"))]
    wgpu(hr, good_rows[:1]); sres.append(resource_reasons(hr, sl))
    wgpu(hr, [good_rows[0], good_rows[0].replace("0,0,", "1,1,")]); sres.append(resource_reasons(hr, sl))
    wgpu(hr, [good_rows[0], good_rows[1].replace("NVIDIA H200 NVL", "NVIDIA H100 80GB HBM3")]); sres.append(resource_reasons(hr, sl))
    wgpu(hr, [good_rows[0], good_rows[1].replace("143771", "139999")]); sres.append(resource_reasons(hr, sl))
    wgpu(hr, good_rows); wfoot(hr, dict(gf, cuda_visible_devices="2,3")); sres.append(resource_reasons(hr, sl))
    wfoot(hr, gf)
    case("h200x2: Slurm allocation or gpu identity disagreeing -> reason", sres,
         [["SLURM_PARTITION"], ["SLURM_ALLOCATION"], ["GPU_IDENTITY_COUNT_UUID", "GPU_IDENTITY_FOOTER_MASK"],
          ["GPU_IDENTITY_COUNT_UUID"], ["GPU_IDENTITY_KIND_MEMORY"], ["GPU_IDENTITY_KIND_MEMORY"],
          ["GPU_IDENTITY_FOOTER_MASK"]])
    # runner: whole file == runner_rev5(R runner); tested identity; submit command
    rb = tmp / "runner"
    rb.mkdir()
    base_rn = R_RUNNER.read_text()
    good_full = runner_rev5(base_rn)
    (rb / "RP.slurm").write_text(good_full)
    (rb / "runner_selftest.log").write_text("CMFGEN_TWIN_MAP_RUNNER_SELFTEST_PASS positive=5 negative=23\n")
    (rb / "tested.txt").write_text(f"runner_sha256={sha(rb / 'RP.slurm')} selftest_log_sha256={sha(rb / 'runner_selftest.log')}\n")
    runroot = "/gpfs/x/RP"
    (rb / "submit.txt").write_text(submit_expected(runroot))
    def rr(name, submit="submit.txt"):
        return runner_reasons(R_RUNNER, rb / name, rb / "tested.txt", rb / submit, run_root=runroot)
    ok_r = rr("RP.slurm")
    (rb / "RPbad.slurm").write_text(good_full.replace("#SBATCH --mem=64G", "#SBATCH --mem=128G"))
    bad_r = rr("RPbad.slurm")
    (rb / "RPin.slurm").write_text(good_full.replace("  run_case h200x2_ok ", "}\nrun_model() { :; }\nx() {\n  run_case h200x2_ok ", 1))
    in_r = rr("RPin.slurm")
    case("runner: allowed h200x2 diff accepted, unregistered diff rejected",
         (ok_r, "RUNNER_UNREGISTERED_DIFF" in bad_r, "RUNNER_RESOURCE:--mem=64G" in bad_r,
          "RUNNER_UNREGISTERED_DIFF" in in_r), ([], True, True, True))
    (rb / "RP2.slurm").write_text(good_full)
    (rb / "tested2.txt").write_text("runner_sha256=" + "0" * 64 + f" selftest_log_sha256={sha(rb / 'runner_selftest.log')}\n")
    un_r = runner_reasons(R_RUNNER, rb / "RP2.slurm", rb / "tested2.txt", rb / "submit.txt", run_root=runroot)
    (rb / "runner_selftest.log").write_text("SELFTEST_POSITIVE_FAILED\n")
    fl_r = rr("RP.slurm")
    case("runner: untested runner hash or failed selftest rejected",
         ("RUNNER_NOT_TESTED" in un_r, "RUNNER_SELFTEST_NOT_PASS" in fl_r), (True, True))
    subs = []
    for i, text in enumerate((submit_expected(runroot).replace("sbatch ", "sbatch --partition=h100 "),
                              submit_expected(runroot).replace("sbatch ", "sbatch --time=48:00:00 "),
                              submit_expected(runroot).replace("job.slurm", "other.slurm"),
                              submit_expected("/gpfs/x/OTHER"),
                              submit_expected(runroot).replace("--exclude=syn102", "--exclude=syn101"))):
        (rb / f"sub{i}.txt").write_text(text)
        subs.append("SUBMIT_COMMAND_NOT_REGISTERED" in runner_reasons(R_RUNNER, rb / "RP.slurm", rb / "tested.txt",
                                                                       rb / f"sub{i}.txt", run_root=runroot))
    case("runner: submit command overriding resources rejected", subs, [True] * 5)
    # paired resources
    ep = tmp / "h200ep"
    ep.mkdir()
    (hr / "module_env.txt").write_text("gnu13/13.2.0\ncuda/13.0.2\n")
    (ep / "module_env.txt").write_text("gnu13/13.2.0\ncuda/13.0.2\n")
    wfoot(ep, gf)
    p_ok = paired_resource_reasons(hr, ep)
    wfoot(ep, dict(gf, partition="h100", gpu_kind="H100", gpu_profile="h100-a100x2"))
    (ep / "module_env.txt").write_text("gnu13/13.2.0\ncuda/12.8.1\n")
    (hr / "module_env.txt").write_text("gnu13/13.2.0\n\tlibm.so.6 => /lib64/libm.so.6 (0x00007f1a2b3c4000)\n")
    (ep / "module_env.txt").write_text("gnu13/13.2.0\n\tlibm.so.6 => /lib64/libm.so.6 (0x00007fffaaaa0000)\n")
    wfoot(ep, gf)
    m_ok = paired_resource_reasons(hr, ep)
    (ep / "module_env.txt").write_text("gnu13/13.2.0\n\tlibm.so.6 => /opt/other/libm.so.6 (0x00007fffaaaa0000)\n")
    case("paired: module environment compared without load addresses", (m_ok, paired_resource_reasons(hr, ep)),
         ([], ["PAIRED_RESOURCE:module_env"]))
    (hr / "module_env.txt").write_text("gnu13/13.2.0\ncuda/13.0.2\n")
    (ep / "module_env.txt").write_text("gnu13/13.2.0\ncuda/13.0.2\n")
    wfoot(ep, gf)
    p_ok = paired_resource_reasons(hr, ep)
    wfoot(ep, dict(gf, partition="h100", gpu_kind="H100", gpu_profile="h100-a100x2"))
    (ep / "module_env.txt").write_text("gnu13/13.2.0\ncuda/12.8.1\n")
    case("paired: E'/R' resource mismatch -> reason", (p_ok, sorted(paired_resource_reasons(hr, ep))),
         ([], sorted(["PAIRED_RESOURCE:partition", "PAIRED_RESOURCE:gpu_kind", "PAIRED_RESOURCE:gpu_profile",
                      "PAIRED_RESOURCE:module_env"])))
    ev3 = {k: tmp / f"ev3_{k}" for k in STAGE1_EVIDENCE}
    ev3["user_h200"].write_text("user chose h100,a100")
    case("stage1 user h200 decision without h200x2 rejected", "STAGE1_USER_H200_NOT_RECORDED" in stage1_reasons(ev3), True)
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
    for k in ("--run", "--base", "--selftest", "--out"):
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
