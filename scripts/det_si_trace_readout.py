#!/usr/bin/env python3
"""DET-SI-CONSUMPTION-TRACE readout (prereg docs/PREREG_DET_SI_CONSUMPTION_TRACE_2026-10-10.md).

One limited trace run T: the O inputs (DET-COLL-PRESC run O, 12 registered outer iterations) with a rebuilt binary
carrying the default-off Si II/III shell-10 consumption trace, armed by two keys, stopping after the material
generation 13 comparison dump.  Primary output: one table (si10_trace_table.csv, Markdown display copy) that
separates the stage/NLTE III/II gap of Si shell 10 into the registered residual kinds, channel budgets and
conditional links, with the public CMFGEN row.

Subcommands
  scope   --out DIR                            G-SCOPE: changed sources insertion-only, registry delta, others identical
  stage   --build DIR --run DIR --scope DIR    stage T from O (copied inputs, new binary, exports + two trace keys)
  analyze --run DIR --out DIR                  G-TRACE, G-EQUIV, G-TABLE, G-RUN and the main table
Exit codes: 0 pass, 2 refused, 3 existing output path.
"""
import argparse
import difflib
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

P = Path(__file__).resolve().parent.parent
PREREG = "docs/PREREG_DET_SI_CONSUMPTION_TRACE_2026-10-10.md"
BASE = Path("/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010")
ORUN = Path("/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1")
SEALED = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source")
SEALED_MANIFEST = Path("/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/cuda_build1/source_sha_at_build.txt")
WORK = BASE / "source"
CHANGED = ("src/lumina_plasma.c", "src/lumina_cmfgen.c", "src/lumina_cuda.cu")
REGISTRY = "src/env_universe.h"
NEW_FILES = ("src/det_si_trace.h",)
TRACE_KEYS = ("LUMINA_SI_TRACE", "LUMINA_SI_TRACE_DIR")
TRACE_TOKEN = re.compile(r"det_si|DET_SI|DET-SI")
TARGET = {"Z": 14, "ion_lo": 1, "ion_hi": 2, "shell": 10, "generation": 13, "dump_radiation": 12}
TIME_CEILING_S, GPU_HOURS_CEILING = 75600, 42.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def refuse(why):
    print("REFUSED: " + "; ".join(why))
    return 2


def new_dir(path):
    path = Path(path)
    if path.exists():
        print(f"EXISTS {path}")
        raise SystemExit(3)
    path.mkdir(parents=True)
    return path


# ------------------------------------------------------------------ G-SCOPE
def det_si_trace_insertion_only(old_lines, new_lines):
    """old must be a subsequence of new (every original line kept, in order); returns the inserted blocks as
    (first_new_line_1based, [lines]) or None when an original line was removed or changed."""
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    blocks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            removed = old_lines[i1:i2]
            added = new_lines[j1:j2]
            # alignment artefact: a removed line that reappears verbatim inside the replacement is not a removal
            pos = 0
            for line in removed:
                try:
                    pos = added.index(line, pos) + 1
                except ValueError:
                    return None
            kept = set()
            pos = 0
            for line in removed:
                k = added.index(line, pos)
                kept.add(k)
                pos = k + 1
            run = []
            for k, line in enumerate(added):
                if k in kept:
                    if run:
                        blocks.append(run)
                        run = []
                else:
                    run.append((j1 + k + 1, line))
            if run:
                blocks.append(run)
            continue
        blocks.append([(j1 + k + 1, line) for k, line in enumerate(new_lines[j1:j2])])
    return [(b[0][0], [line for _, line in b]) for b in blocks]


def det_si_trace_registry_delta(old_text, new_text):
    """env_universe.h: exactly the two trace keys added in sorted position and the count 536 -> 538."""
    why = []
    old_keys = re.findall(r'^    "(LUMINA_[A-Z0-9_]+)",$', old_text, re.M)
    new_keys = re.findall(r'^    "(LUMINA_[A-Z0-9_]+)",$', new_text, re.M)
    if [k for k in new_keys if k not in TRACE_KEYS] != old_keys:
        why.append("REGISTRY_EXISTING_ENTRIES_CHANGED")
    if sorted(set(new_keys) - set(old_keys)) != sorted(TRACE_KEYS) or len(new_keys) != len(old_keys) + 2:
        why.append("REGISTRY_ADDED_KEYS")
    for key in TRACE_KEYS:          # the existing list has its own collation; check the neighbours only
        if key in new_keys:
            i = new_keys.index(key)
            if not (i > 0 and new_keys[i - 1] < key and i + 1 < len(new_keys) and key < new_keys[i + 1]):
                why.append(f"REGISTRY_POSITION:{key}")
    oc = re.findall(r"#define LUMINA_ENV_UNIVERSE_COUNT (\d+)", old_text)
    nc = re.findall(r"#define LUMINA_ENV_UNIVERSE_COUNT (\d+)", new_text)
    if oc != [str(len(old_keys))] or nc != [str(len(new_keys))]:
        why.append(f"REGISTRY_COUNT:{oc}->{nc}")
    strip = lambda t: re.sub(r'#define LUMINA_ENV_UNIVERSE_COUNT \d+|    "LUMINA_SI_TRACE(_DIR)?",\n', "", t)
    if strip(old_text) != strip(new_text):
        why.append("REGISTRY_OTHER_TEXT_CHANGED")
    return why


def cmd_scope(out):
    out = new_dir(out)
    why, report = [], {"changed": {}, "identical": 0, "new_files": {}}
    manifest = [ln.split(None, 1) for ln in SEALED_MANIFEST.read_text().splitlines() if ln.strip()]
    for digest, rel in manifest:
        rel = rel.strip()
        if rel in CHANGED or rel == REGISTRY:
            continue
        if sha(WORK / rel) != digest or sha(SEALED / rel) != digest:
            why.append(f"UNCHANGED_FILE_DIFFERS:{rel}")
        else:
            report["identical"] += 1
    for rel in CHANGED:
        old = (SEALED / rel).read_text(encoding="utf-8", errors="surrogateescape").split("\n")
        new = (WORK / rel).read_text(encoding="utf-8", errors="surrogateescape").split("\n")
        blocks = det_si_trace_insertion_only(old, new)
        if blocks is None:
            why.append(f"NOT_INSERTION_ONLY:{rel}")
            continue
        untagged = [first for first, lines in blocks
                    if not any(TRACE_TOKEN.search(x) for x in lines) and any(x.strip() for x in lines)]
        if untagged:
            why.append(f"INSERTION_WITHOUT_TRACE_TOKEN:{rel}:{untagged}")
        report["changed"][rel] = {"sealed_sha256": sha(SEALED / rel), "work_sha256": sha(WORK / rel),
                                  "inserted_blocks": len(blocks),
                                  "inserted_lines": sum(len(b) for _, b in blocks),
                                  "first_lines": [first for first, _ in blocks]}
        with open(out / f"inserted_{Path(rel).name}.txt", "x", encoding="utf-8") as f:
            for first, lines in blocks:
                f.write(f"@@ new line {first}\n" + "\n".join(lines) + "\n")
    why += det_si_trace_registry_delta((SEALED / REGISTRY).read_text(), (WORK / REGISTRY).read_text())
    report["changed"][REGISTRY] = {"sealed_sha256": sha(SEALED / REGISTRY), "work_sha256": sha(WORK / REGISTRY)}
    for rel in NEW_FILES:
        if (SEALED / rel).exists() or not (WORK / rel).is_file():
            why.append(f"NEW_FILE_STATE:{rel}")
        else:
            report["new_files"][rel] = sha(WORK / rel)
    report["refusals"] = why
    (out / "scope.json").write_text(json.dumps(report, indent=1) + "\n")
    return refuse(why) if why else (print(f"G-SCOPE PASS identical={report['identical']}") or 0)


# ------------------------------------------------------------------ staging of T from O
def exports_lines(path):
    rows = []
    for line in Path(path).read_text().split("\n"):
        if not line:
            continue
        m = re.fullmatch(r'declare -x ([A-Z0-9_]+)="(.*)"', line)
        if not m:
            raise ValueError(f"EXPORTS_LINE_FORMAT:{line[:60]}")
        rows.append((m.group(1), m.group(2)))
    return rows


def det_si_trace_exports(o_text_path, o_run, t_run):
    """O exports re-rooted to T (values beginning with the O run path) plus the two trace keys, sorted."""
    o_run, t_run = str(o_run), str(t_run)
    rows, mapping = [], {}
    for key, value in exports_lines(o_text_path):
        if key in TRACE_KEYS:
            raise ValueError("TRACE_KEY_ALREADY_IN_O")
        if value.startswith(o_run + "/"):
            mapping[key] = [value, t_run + value[len(o_run):]]
            value = t_run + value[len(o_run):]
        rows.append((key, value))
    rows += [("LUMINA_SI_TRACE", "1"), ("LUMINA_SI_TRACE_DIR", t_run + "/si_trace")]
    rows.sort()
    return "".join(f'declare -x {k}="{v}"\n' for k, v in rows), mapping


def det_si_trace_exports_check(o_exports, t_exports, o_run, t_run):
    o = dict(exports_lines(o_exports))
    t = dict(exports_lines(t_exports))
    why = []
    for key in sorted(set(o) | set(t)):
        if key == "LUMINA_SI_TRACE":
            if key in o or t.get(key) != "1":
                why.append("TRACE_KEY")
        elif key == "LUMINA_SI_TRACE_DIR":
            if key in o or t.get(key) != f"{t_run}/si_trace":
                why.append("TRACE_DIR_KEY")
        elif key not in o or key not in t:
            why.append(f"KEY_SET:{key}")
        elif o[key].startswith(f"{o_run}/"):
            if t[key] != f"{t_run}{o[key][len(str(o_run)):]}":
                why.append(f"PATH_KEY:{key}")
        elif o[key] != t[key]:
            why.append(f"KEY_DIFFERS:{key}")
    return why


def cmd_stage(build, run, scope, gpu_profile):
    build, run, scope = Path(build), Path(run), Path(scope)
    if json.loads((scope / "scope.json").read_text()).get("refusals"):
        return refuse(["SCOPE_NOT_PASSED"])
    binary = build / "lumina_cuda"
    if not binary.is_file() or (build / "build.rc").read_text().strip() != "0":
        return refuse(["BUILD_NOT_OK"])
    if run.exists():
        print(f"EXISTS {run}")
        return 3
    oin = ORUN / "input"
    model_names = [ln.split(None, 1)[1].strip() for ln in (oin / "deck.sha256").read_text().splitlines() if ln.strip()]
    atomic_names = [ln.split(None, 1)[1].strip() for ln in (oin / "topion.sha256").read_text().splitlines() if ln.strip()]
    if sorted(p.name for p in (oin / "model").iterdir()) != sorted(model_names):
        return refuse(["O_MODEL_SET_NOT_DECK_LIST"])
    if sorted(p.name for p in (oin / "global_atomic").iterdir()) != sorted(atomic_names):
        return refuse(["O_ATOMIC_SET_NOT_TOPION_LIST"])
    tin = run / "input"
    (tin / "model").mkdir(parents=True)
    (tin / "global_atomic").mkdir()
    (run / "work").mkdir()
    (run / "si_trace").mkdir()
    files = {}
    for name in model_names:
        shutil.copyfile(oin / "model" / name, tin / "model" / name)
    for name in atomic_names:
        shutil.copyfile(oin / "global_atomic" / name, tin / "global_atomic" / name)
    for name in ("fixed_te_profile.txt", "deck.sha256", "topion.sha256", "sigma.sha256"):
        shutil.copyfile(oin / name, tin / name)
    shutil.copyfile(binary, tin / "lumina_cuda")
    (tin / "lumina_cuda").chmod(0o750)
    text, mapping = det_si_trace_exports(oin / "resolved_lumina.exports", ORUN, run)
    with open(tin / "resolved_lumina.exports", "x") as f:
        f.write(text)
    bsha = sha(tin / "lumina_cuda")
    (tin / "binary.sha256").write_text(bsha + "\n")
    (tin / "resolved_lumina.sha256").write_text(sha(tin / "resolved_lumina.exports") + "\n")
    build_identity = json.loads((build / "build_identity.json").read_text())
    if build_identity["binary"]["sha256"] != bsha:
        return refuse(["BUILD_IDENTITY_BINARY"])
    (tin / "run_build_identity.json").write_text(json.dumps(build_identity, indent=1) + "\n")
    why = []
    for name in model_names:
        if sha(tin / "model" / name) != sha(oin / "model" / name):
            why.append(f"MODEL_COPY:{name}")
    for name in atomic_names:
        if sha(tin / "global_atomic" / name) != sha(oin / "global_atomic" / name):
            why.append(f"ATOMIC_COPY:{name}")
    if sha(tin / "fixed_te_profile.txt") != sha(oin / "fixed_te_profile.txt"):
        why.append("TE_PROFILE_COPY")
    why += det_si_trace_exports_check(oin / "resolved_lumina.exports", tin / "resolved_lumina.exports", ORUN, run)
    for p in sorted(tin.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(tin))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    (run / "STAGE_FILES.json").write_text(json.dumps({"files": files, "exports_mapping": mapping,
                                                      "trace_keys_added": list(TRACE_KEYS)}, indent=1) + "\n")
    runner = BASE / "tools/run_det_si_trace_gpu.slurm"
    (run / "STAGE_IDENTITY.txt").write_text(
        "schema=SI_TRACE_STAGE_V1\n"
        f"run={run}\ngpu_profile={gpu_profile}\nouter_iterations=12\n"
        f"run_binary_sha256={bsha}\nrun_build_identity_sha256={sha(tin / 'run_build_identity.json')}\n"
        f"effective_exports_sha256={sha(tin / 'resolved_lumina.exports')}\n"
        f"runner_sha256={sha(runner)}\ncontract={PREREG}\n"
        f"base_run={ORUN}\nbase_exports_sha256={sha(oin / 'resolved_lumina.exports')}\n"
        f"base_binary_sha256={sha(oin / 'lumina_cuda')}\n"
        "exports_delta_vs_base=path keys re-rooted; LUMINA_SI_TRACE absent->1; LUMINA_SI_TRACE_DIR absent->run/si_trace\n"
        f"scope={scope / 'scope.json'}\nscope_sha256={sha(scope / 'scope.json')}\n"
        f"time_ceiling_s={TIME_CEILING_S}\ngpu_hours_ceiling={GPU_HOURS_CEILING}\nstaged_files={len(files)}\n")
    (run / "OWNER.txt").write_text(
        "project=LUMINA-SN\nrepo=Lumina-sn thenmc-macroatom-fluorescence\n"
        "purpose=DET-SI-CONSUMPTION-TRACE run T (O inputs + Si shell-10 consumption trace, stops after the "
        "material-13 dump), user decision 2026-10-10\nowner=Claude driver session\nDO_NOT_CANCEL\n")
    if why:
        return refuse(why)
    print(f"STAGED {run} files={len(files)} binary={bsha}")
    return 0


# ------------------------------------------------------------------ analysis (prereg sections 3-6, 8)
SITE = {1: "BB", 2: "COLDATA", 3: "METACOLL", 4: "FEIII", 5: "BF", 6: "COLLION", 7: "TOPIV", 8: "DR", 9: "CE",
        10: "NT", 11: "TIMEDEP"}
FOUR_PAIR = (1, 2, 3, 4)          # (dst,src)+=v1 (src,dst)+=v2 (src,src)-=v1 (dst,dst)-=v2
BF_LIKE = (5, 6)                  # (dst,src)+=v1 (src,src)-=v1 (src,dst)+=v2 (dst,dst)-=v2
TWO = (8, 9, 10)                  # (dst,src)+=v1 (src,src)-=v1
DIAG_B = (7, 11)                  # (dst,dst)-=v1 ; b[dst]-=v2
RES_NORM, RES_DEX, EQUIV_RTOL = 1e-8, 0.01, 1e-8
PUBLIC_SI = P / "data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt"
DIAG8_PUBLIC = Path("/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/tools/r2_crosscheck.py")
REGISTERED_VEL = {"lumina_vel_mid_kms": 11544.0, "public_vel_mid_kms": 11534.59}


def read_trace(path):
    import csv
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def num(x):
    import math
    return float(x) if x not in ("", None) else math.nan


def read_matrix(path):
    import struct
    import numpy as np
    raw = Path(path).read_bytes()
    if raw[:8] != b"DSTM0001":
        raise ValueError(f"MATRIX_MAGIC:{path}")
    n, has_b = struct.unpack("<ii", raw[8:16])
    total = struct.unpack("<d", raw[16:24])[0]
    a = np.frombuffer(raw, dtype="<f8", count=n * n, offset=24).reshape((n, n)).T.copy()   # ACM[i, j]
    b = np.frombuffer(raw, dtype="<f8", count=n, offset=24 + 8 * n * n).copy() if has_b else None
    return a, b, total


def replay(ops, n):
    """the recorded statements in op order from a zero matrix (prereg 7: every pre-snapshot write is recorded)."""
    import numpy as np
    a, b = np.zeros((n, n)), np.zeros(n)
    for r in sorted(ops, key=lambda r: int(r["op"])):
        s, d, c, v1, v2 = int(r["site"]), int(r["dst"]), int(r["src"]), float(r["v1"]), float(r["v2"])
        if s in FOUR_PAIR:
            a[d, c] += v1; a[c, d] += v2; a[c, c] -= v1; a[d, d] -= v2
        elif s in BF_LIKE:
            a[d, c] += v1; a[c, c] -= v1; a[c, d] += v2; a[d, d] -= v2
        elif s in TWO:
            a[d, c] += v1; a[c, c] -= v1
        elif s in DIAG_B:
            a[d, d] -= v1; b[d] -= v2
        else:
            raise ValueError(f"SITE_UNKNOWN:{s}")
    return a, b


def gth(a):
    """stationary vector of a column generator (a[i, j] = rate j -> i) by Grassmann-Taksar-Heyman; no subtraction."""
    import numpy as np
    q = a.T.copy()
    np.fill_diagonal(q, 0.0)
    if (q < 0).any() or not np.isfinite(q).all():
        return None, "OFFDIAGONAL_NOT_RATE"
    n = q.shape[0]
    for k in range(n - 1, 0, -1):
        s = q[k, :k].sum()
        if not (s > 0.0):
            return None, f"GTH_ZERO_PIVOT:{k}"
        q[:k, k] /= s
        q[:k, :k] += np.outer(q[:k, k], q[k, :k])
    pi = np.zeros(n)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ q[:k, k]
    if not np.isfinite(pi).all() or (pi < 0).any() or pi.sum() <= 0:
        return None, "GTH_NONFINITE"
    return pi / pi.sum(), None


def backward(a, x, b=None):
    """componentwise backward error per row; 0/0 -> 0, x/0 -> nan (undefined)."""
    import numpy as np
    r = a @ x - (b if b is not None else 0.0)
    den = np.abs(a) @ np.abs(x) + (np.abs(b) if b is not None else 0.0)
    out = np.full(len(r), np.nan)
    zero = (den == 0)
    out[zero & (r == 0)] = 0.0
    ok = ~zero
    out[ok] = np.abs(r[ok]) / den[ok]
    return out


def ratio(x, upper):
    import numpy as np
    lo, hi = float(np.sum(x[~upper])), float(np.sum(x[upper]))
    return hi / lo if lo > 0 else float("nan")


def dex(a, b):
    import math
    return math.log10(a / b) if (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)) else float("nan")


def div(a, b):
    return a / b if b else float("nan")


def equiv(a, b):
    if a == 0.0 and b == 0.0:
        return True
    if a == 0.0 or b == 0.0:
        return False
    return abs(a - b) <= EQUIV_RTOL * max(abs(a), abs(b))


def public_si_ratio(vel_kms):
    """diag8 public reader (19.480 d block); nearest public vel_mid, ties to the lower, no extrapolation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("diag8_public", DIAG8_PUBLIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.public_ions(PUBLIC_SI)
    vs = [v for v, _ in rows]
    if not (min(vs) <= vel_kms <= max(vs)):
        return float("nan"), float("nan")
    v, d = min(rows, key=lambda p: (abs(p[0] - vel_kms), p[0]))
    return float(d["si2"]) / float(d["si1"]), v


def dump_row(run_dir, iteration, kind, shell):
    rows = read_trace(Path(run_dir) / f"work/physics_comparison/physics_DET_iter{iteration:04d}.{kind}.csv")
    key = "shell_id"
    return [r for r in rows if int(r[key]) == shell]


def det_si_exc_build_fl(ops, sm, N, gh, coldata_override=None, extra=(), bb_radiative_override=None):
    """FL operator of one recorded pair call, statements in op order (DET-SI-CONSUMPTION-TRACE prereg 4;
    DET-SI-EXCITATION-BUDGET prereg 5).  Returns (A_fl, bb_level, fl_undefined, channels, maps):
    A_fl is the total operator accumulated exactly as before; channels maps a channel name to the FL matrix of
    its own statements (off-diagonal = rate j -> i per particle in j, diagonal = -outflow).
    coldata_override(r, x) -> (C_up, C_down) or None replaces one COLDATA/METACOLL/FEIII statement;
    extra = [(dst_fl, src_fl, rate, channel), ...] statements appended after the recorded ones;
    bb_radiative_override(r, x) -> (R_absorb, R_stim, R_spont) or None replaces the radiative triple of one BB
    statement after its recorded split is checked, collisions kept (DET-SI-BB-COUNTERFACTUAL prereg 5)."""
    import numpy as np
    sm = sorted(sm, key=lambda r: int(r["i_fl"]))
    N_fl = len(sm)
    sl_of = np.array([int(r["sl_local"]) for r in sm])
    frac = np.array([float(r["within_sl_frac"]) for r in sm])
    fl_index = np.array([int(r["fl_index"]) for r in sm])
    fl_of_global = {int(g): i for i, g in enumerate(fl_index)}
    members = {k: np.where(sl_of == k)[0] for k in range(N)}
    lvl = {(int(r["sl_local"]), int(r["level_num"])): int(r["i_fl"]) for r in sm}
    fl_undefined = []
    channels = {}

    def ch(name):
        return channels.setdefault(name, np.zeros((N_fl, N_fl)))

    def put(M, dst, src, rate):
        M[dst, src] += rate
        M[src, src] -= rate

    def single(k):
        m = members[k]
        return int(m[0]) if len(m) == 1 and frac[m[0]] == 1.0 else None
    gh_fl = single(gh)
    A_fl = np.zeros((N_fl, N_fl))
    bb_level = np.zeros((N_fl, N_fl))
    for r in sorted(ops, key=lambda r: int(r["op"])):
        s = int(r["site"])
        x = [num(r[f"x{k}"]) for k in range(1, 15)]
        if s == 1:
            up, lo = fl_of_global[int(r["lev_a"])], fl_of_global[int(r["lev_b"])]
            tu, td = x[0] + x[3], x[1] + x[2] + x[4]
            if tu * x[5] != float(r["v1"]) or td * x[6] != float(r["v2"]):
                fl_undefined.append(f"BB_RATE_SPLIT:op{r['op']}")
            ra, rs, rp = x[0], x[1], x[2]
            if bb_radiative_override is not None:
                rep = bb_radiative_override(r, x)
                if rep is not None:
                    ra, rs, rp = rep
                    tu, td = ra + x[3], rs + rp + x[4]
            for M in (A_fl, bb_level):
                M[up, lo] += tu; M[lo, lo] -= tu; M[lo, up] += td; M[up, up] -= td
            put(ch("BB radiative upward"), up, lo, ra); put(ch("BB radiative downward"), lo, up, rs + rp)
            put(ch("BB collisional (line)"), up, lo, x[3]); put(ch("BB collisional (line)"), lo, up, x[4])
        elif s in (2, 3, 4):
            hi = lvl.get((int(r["dst"]), int(r["lev_a"])))
            lo = lvl.get((int(r["src"]), int(r["lev_b"])))
            if hi is None or lo is None:
                fl_undefined.append(f"COLL_LEVEL_MAP:op{r['op']}")
                continue
            cu, cd = x[0], x[1]
            if coldata_override is not None:
                rep = coldata_override(r, x)
                if rep is not None:
                    cu, cd = rep
            for M in (A_fl, bb_level):
                M[hi, lo] += cu; M[lo, lo] -= cu; M[lo, hi] += cd; M[hi, hi] -= cd
            put(ch("collisional (tabulated)"), hi, lo, cu); put(ch("collisional (tabulated)"), lo, hi, cd)
        elif s in BF_LIKE:
            lev = fl_of_global[int(r["lev_b"])]
            if gh_fl is None:
                fl_undefined.append("UPPER_GROUND_SL_NOT_SINGLE")
                continue
            ion_rate, rec_rate = x[0], x[2]
            A_fl[gh_fl, lev] += ion_rate; A_fl[lev, lev] -= ion_rate
            A_fl[lev, gh_fl] += rec_rate; A_fl[gh_fl, gh_fl] -= rec_rate
            name = "BF" if s == 5 else "collisional ionization (recorded)"
            put(ch(f"{name} ionization"), gh_fl, lev, ion_rate)
            put(ch(f"{name} recombination"), lev, gh_fl, rec_rate)
        elif s in TWO:
            d, c, v = int(r["dst"]), int(r["src"]), float(r["v1"])
            if len(members[c]) == 1 and frac[members[c][0]] == 1.0:
                srcs = [int(members[c][0])]
            elif s == 9:
                srcs = [int(i) for i in members[c]]          # per-particle CE rate applies to every member
            else:
                fl_undefined.append(f"SL_ONLY_TERM:{SITE[s]}")
                continue
            dfl = single(d)
            if dfl is None:
                fl_undefined.append(f"SL_ONLY_TARGET:{SITE[s]}")
                continue
            for c_fl in srcs:
                A_fl[dfl, c_fl] += v; A_fl[c_fl, c_fl] -= v
                put(ch(SITE[s]), dfl, c_fl, v)
        else:
            fl_undefined.append(f"SL_ONLY_TERM:{SITE[s]}")
    for dst, src, rate, name in extra:
        A_fl[dst, src] += rate; A_fl[src, src] -= rate
        put(ch(name), dst, src, rate)
    maps = {"members": members, "fl_of_global": fl_of_global, "sl_of": sl_of, "frac": frac,
            "fl_index": fl_index, "gh_fl": gh_fl, "g0_fl": single(0), "lvl": lvl}
    return A_fl, bb_level, fl_undefined, channels, maps


def cmd_analyze(run, out):
    import csv
    import math
    import numpy as np
    run = Path(run)
    tr = run / "si_trace"
    out = new_dir(out)
    why, notes = [], []
    gates = {}
    S = TARGET["shell"]

    # ---------------- G-RUN
    footer = dict(l.split("=", 1) for l in (run / "RUN_FOOTER.txt").read_text().splitlines() if "=" in l)
    gpu_h = int(footer.get("allocated_gpus", 0)) * int(footer.get("elapsed_seconds", 0)) / 3600.0
    g_run = []
    if footer.get("terminal_state") != "TRACE_TARGET_COMPLETE" or footer.get("model_rc") != "0":
        g_run.append(f"TERMINAL:{footer.get('terminal_state')}")
    if footer.get("model_entries") != "1":
        g_run.append("MODEL_ENTRIES")
    if not (0 < int(footer.get("slurm_time_limit_s", 0)) <= TIME_CEILING_S):
        g_run.append("TIME_LIMIT")
    if gpu_h > GPU_HOURS_CEILING:
        g_run.append(f"GPU_HOURS:{gpu_h:.3f}")
    status = {r["key"]: r["value"] for r in read_trace(tr / "si10_status.csv")}
    if status.get("status") != "TRACE_TARGET_COMPLETE":
        g_run.append("TRACE_STATUS")
    stop_iter = int(footer.get("stop_iteration", "-1"))
    gates["G-RUN"] = g_run

    # ---------------- load trace
    calls = read_trace(tr / "si10_calls.csv")
    solve = {r["call"]: r for r in read_trace(tr / "si10_solve.csv")}
    final = {r["call"]: r for r in read_trace(tr / "si10_final.csv")}
    ops_all = read_trace(tr / "si10_ops.csv")
    vec_all = read_trace(tr / "si10_vec.csv")
    slmap_all = read_trace(tr / "si10_slmap.csv")
    ladder = read_trace(tr / "si10_ladder.csv")
    ladlev = read_trace(tr / "si10_ladder_levels.csv")
    stage_rows = read_trace(tr / "si10_stage.csv")
    ce_rows = read_trace(tr / "si10_ce.csv")
    cevec = read_trace(tr / "si10_cevec.csv")
    ne_rows = read_trace(tr / "si10_ne.csv")
    commit = read_trace(tr / "si10_commit.csv")
    cm = {r["kind"]: r["value"] for r in commit if r["kind"] != "level"}
    commit_fl = np.array([float(r["value"]) for r in sorted((r for r in commit if r["kind"] == "level"),
                                                            key=lambda r: int(r["i"]))])

    # ---------------- G-TRACE: keys, adopted lineage, required records
    g_trace = []
    if int(cm.get("committed_generation", -1)) != TARGET["generation"] or \
       int(cm.get("dump_population_generation", -1)) != TARGET["generation"] or \
       int(cm.get("dump_radiation_generation", -1)) != TARGET["dump_radiation"]:
        g_trace.append("COMMIT_GENERATIONS")
    if int(cm.get("iteration", -2)) != stop_iter:
        g_trace.append("COMMIT_ITERATION_NOT_STOP")
    last_bundle = max(int(r["bundle"]) for r in ne_rows) if ne_rows else -1
    acc = [r for r in ne_rows if int(r["bundle"]) == last_bundle and r["accepted"] == "1"]
    if len(acc) != 1:
        g_trace.append(f"ACCEPTED_NE_ITERATIONS:{len(acc)}")
    ne_acc = int(acc[-1]["ne_iter"]) if acc else -1
    ccalls = [c for c in calls if int(c["bundle"]) == last_bundle and int(c["ne_iter"]) == ne_acc]
    if not ccalls:
        g_trace.append("NO_CALL_FOR_ACCEPTED_NE")
        return refuse(g_trace)
    passes = [int(c["ce_pass"]) for c in ccalls]
    if passes != list(range(len(passes))):
        g_trace.append(f"CE_PASS_SEQUENCE:{passes}")
    for c in calls:
        cid = c["call"]
        if int(c["core_generation"]) != TARGET["generation"]:
            g_trace.append(f"CALL_GENERATION:{cid}")
        if cid not in solve or cid not in final:
            g_trace.append(f"CALL_INCOMPLETE:{cid}")
        for ph in ("solve",) + (("gen",) if c["capture_generator"] == "1" else ()):
            if not (tr / f"si10_mat_{int(cid):06d}_{ph}.bin").is_file():
                g_trace.append(f"MATRIX_MISSING:{cid}:{ph}")
    adopted = ccalls[-1]
    cid = adopted["call"]
    N, N_fl, n_lo = int(adopted["N"]), int(adopted["N_fl"]), int(adopted["n_lo_super"])
    gh = int(adopted["ground_hi"])
    vec = {}
    for r in vec_all:
        if r["call"] == cid:
            vec.setdefault(r["phase"], {})[int(r["i"])] = float(r["value"])
    vec = {k: np.array([v[i] for i in range(len(v))]) for k, v in vec.items()}
    sm = sorted((r for r in slmap_all if r["call"] == cid), key=lambda r: int(r["i_fl"]))
    upper_fl = np.array([r["upper_ion"] == "1" for r in sm])
    sl_of = np.array([int(r["sl_local"]) for r in sm])
    frac = np.array([float(r["within_sl_frac"]) for r in sm])
    fl_index = np.array([int(r["fl_index"]) for r in sm])
    lev_start = int(fl_index.min())
    upper_sl = np.arange(N) >= n_lo
    last_ce = [r for r in ce_rows if int(r["bundle"]) == last_bundle and int(r["ne_iter"]) == ne_acc]
    last_pass = max(int(r["ce_pass"]) for r in last_ce) if last_ce else -1
    after = np.array([float(r["after"]) for r in sorted(
        (r for r in cevec if int(r["bundle"]) == last_bundle and int(r["ne_iter"]) == ne_acc
         and int(r["ce_pass"]) == last_pass), key=lambda r: int(r["i_fl"]))])
    if last_pass != int(adopted["ce_pass"]):
        g_trace.append(f"LAST_CE_PASS:{last_pass}!={adopted['ce_pass']}")
    if len(after) != N_fl or len(commit_fl) != N_fl or not np.array_equal(after, commit_fl):
        g_trace.append("COMMIT_LINEAGE: last CE pass populations != committed populations")
    lad = [r for r in ladder if int(r["bundle"]) == last_bundle and int(r["ne_iter"]) == ne_acc]
    lad_armed = [r for r in lad if r["ch_armed"] == "1"]
    lrow = (lad_armed or lad)[-1] if lad else None
    if lrow is None:
        g_trace.append("NO_LADDER_FOR_ACCEPTED_NE")
    st = [r for r in stage_rows if int(r["bundle"]) == last_bundle and int(r["ne_iter"]) == ne_acc]
    stage_lo, stage_hi = num(cm.get("stage_lo")), num(cm.get("stage_hi"))
    R_s = stage_hi / stage_lo
    if st and not (num(st[-1]["n_stage_lo"]) == stage_lo and num(st[-1]["n_stage_hi"]) == stage_hi):
        g_trace.append("STAGE_LINEAGE: accepted ladder densities != committed stage densities")
    if lrow is not None and not math.isclose(num(lrow["r_final"]), R_s, rel_tol=1e-12):
        g_trace.append(f"LADDER_RATIO_LINEAGE:{lrow['r_final']}!={R_s!r}")
    if solve[cid]["ret"] != "0":
        g_trace.append("ADOPTED_SOLVE_FAILED")
    gates["G-TRACE"] = g_trace

    # ---------------- G-TABLE: replayed channel statements == generator (bitwise) for every call
    g_table, replay_report = [], []
    ops_by_call = {}
    for r in ops_all:
        ops_by_call.setdefault(r["call"], []).append(r)
    for c in calls:
        if c["capture_generator"] != "1":
            continue
        n_c = int(c["N"])
        g_a, g_b, _ = read_matrix(tr / f"si10_mat_{int(c['call']):06d}_gen.bin")
        r_a, r_b = replay(ops_by_call.get(c["call"], []), n_c)
        ok = np.array_equal(r_a, g_a) and np.array_equal(r_b, g_b)
        d = float(np.max(np.abs(r_a - g_a))) if r_a.shape == g_a.shape else float("inf")
        replay_report.append({"call": c["call"], "bitwise": ok, "max_abs_diff": d})
        if not ok:
            g_table.append(f"REPLAY_NOT_GENERATOR:call={c['call']}:max_abs_diff={d}")
    sites_seen = sorted({SITE[int(r["site"])] for r in ops_by_call.get(cid, [])})
    gates["G-TABLE"] = g_table

    # ---------------- G-EQUIV: T dump at the stop iteration vs O iter0010 (prereg 3)
    g_eq, eq_rows = [], []
    t_it = stop_iter if stop_iter >= 0 else 10
    for kind, keys in (("ion", ("Z14_element_total_cm3", "Z14_stage_q1_cm3", "Z14_stage_q2_cm3",
                                 "Z14_nlte_q1_cm3", "Z14_nlte_q2_cm3", "nonpair_charge_cm3")),
                       ("shell", ("T_e_K", "n_e_cm3"))):
        o = dump_row(ORUN, 10, kind, S)[0]
        t = dump_row(run, t_it, kind, S)[0]
        for k in keys:
            eq_rows.append({"field": f"{kind}:{k}", "O": o[k], "T": t[k], "equal": equiv(float(o[k]), float(t[k]))})
    partners = sorted({(int(float(r["x5"])), int(float(r["x6"]))) for r in ops_by_call.get(cid, []) if r["site"] == "9"} |
                      {(int(float(r["x5"])), int(r["lev_b"])) for r in ops_by_call.get(cid, []) if r["site"] == "9"})
    o_ion, t_ion = dump_row(ORUN, 10, "ion", S)[0], dump_row(run, t_it, "ion", S)[0]
    for z, q in partners:
        for auth in ("stage", "nlte"):
            k = f"Z{z}_{auth}_q{q}_cm3"
            if k in o_ion and k in t_ion:
                eq_rows.append({"field": f"ion:{k}", "O": o_ion[k], "T": t_ion[k],
                                "equal": equiv(float(o_ion[k]), float(t_ion[k]))})
    o_sp = [r for r in read_trace(ORUN / "work/physics_comparison/physics_DET_iter0010.spectral.csv")
            if int(r["shell_id"]) == S]
    t_sp = [r for r in read_trace(run / f"work/physics_comparison/physics_DET_iter{t_it:04d}.spectral.csv")
            if int(r["shell_id"]) == S]
    j_bad = sum(1 for a, b in zip(o_sp, t_sp) if not equiv(float(a["J_nu"]), float(b["J_nu"])))
    eq_rows.append({"field": "spectral:J_nu(all bins)", "O": len(o_sp), "T": len(t_sp),
                    "equal": len(o_sp) == len(t_sp) and j_bad == 0})
    o_man = json.loads((ORUN / "work/physics_comparison/physics_DET_iter0010.manifest.json").read_text())
    t_man = json.loads((run / f"work/physics_comparison/physics_DET_iter{t_it:04d}.manifest.json").read_text())
    for k in ("population_generation", "radiation_generation"):
        eq_rows.append({"field": f"manifest:{k}", "O": o_man[k], "T": t_man[k], "equal": o_man[k] == t_man[k]})
    g_eq = [r["field"] for r in eq_rows if not r["equal"]]
    gates["G-EQUIV"] = g_eq

    # ---------------- the adopted state: matrices, solves, FL operator
    g_a, g_b, _ = read_matrix(tr / f"si10_mat_{int(cid):06d}_gen.bin")
    s_a, s_b, _ = read_matrix(tr / f"si10_mat_{int(cid):06d}_solve.bin")
    x_raw = vec["RAW_SL"]
    used_gen = solve[cid]["used_generator"] == "1"
    if used_gen:
        eta_solve = backward(g_a, x_raw)
        total = float(final[cid]["n_total"])
        norm_res = abs(x_raw.sum() - total) / total if total > 0 else float("nan")
    else:
        eta_solve = backward(s_a, x_raw, s_b)
        norm_res = float("nan")
    eta_phys_raw = backward(g_a, x_raw)
    commit_ok = len(commit_fl) == N_fl
    x_commit_sl = (np.array([commit_fl[sl_of == k].sum() for k in range(N)]) if commit_ok
                   else np.full(N, np.nan))
    eta_phys_commit = backward(g_a, x_commit_sl) if commit_ok else np.full(N, np.nan)

    # FL operator from the recorded per-FL rates (prereg 4: built only from production terms)
    A_fl, bb_level, fl_undefined, _, fl_maps = det_si_exc_build_fl(ops_by_call.get(cid, []), sm, N, gh)
    members, fl_of_global = fl_maps["members"], fl_maps["fl_of_global"]

    # fixed SL and FL steady states (prereg 5): generator and FL operator held fixed
    x_L, why_L = gth(g_a)
    n_F, why_F = (None, "FL_OPERATOR_UNDEFINED:" + ",".join(sorted(set(fl_undefined)))) if fl_undefined else gth(A_fl)
    R_L = ratio(x_L, upper_sl) if x_L is not None else float("nan")
    R_F = ratio(n_F, upper_fl) if n_F is not None else float("nan")
    eta_L = float(np.nanmax(backward(g_a, x_L))) if x_L is not None else float("nan")
    eta_F = float(np.nanmax(backward(A_fl, n_F))) if n_F is not None else float("nan")
    fl_final = vec["FINAL_FL"]
    eta_fl_final = backward(A_fl, fl_final) if not fl_undefined else np.full(N_fl, np.nan)
    fl_sum_rows = np.array([(A_fl @ fl_final)[sl_of == k].sum() for k in range(N)]) if not fl_undefined else None

    # ratios along the consumption path
    R_raw = ratio(x_raw, upper_sl)
    R_xfl = ratio(vec["XFL"], upper_fl)
    R_fin = ratio(fl_final, upper_fl)
    R_c = ratio(commit_fl, upper_fl) if commit_ok else float("nan")
    R_0 = num(lrow["num"]) / num(lrow["den"]) if lrow is not None else float("nan")
    R_s_ladder = num(lrow["r_final"]) if lrow is not None else float("nan")

    # channel per-ion rates at the adopted solution (FINAL_FL) and the ladder (prereg 6)
    n_lo_pop, n_hi_pop = float(fl_final[~upper_fl].sum()), float(fl_final[upper_fl].sum())
    x_fin_sl = np.array([fl_final[sl_of == k].sum() for k in range(N)])
    chan = {}

    def add(name, i_flux, r_flux):
        c = chan.setdefault(name, [0.0, 0.0])
        c[0] += i_flux
        c[1] += r_flux
    groups = {}
    for r in ops_by_call.get(cid, []):
        s = int(r["site"])
        x = [num(r[f"x{k}"]) for k in range(1, 15)]
        if s == 5:
            lev = fl_of_global[int(r["lev_b"])]
            flux_i = x[0] * fl_final[lev]
            add("photoionization", flux_i, 0.0)
            add("radiative + dielectronic recombination", 0.0, x[2] * x_fin_sl[gh])
            k = int(sl_of[lev])
            g = "ground" if lev == 0 else ("explicit excited" if len(members[k]) == 1 else f"super-level {k}")
            gg = groups.setdefault(g, [0.0, 0.0])
            gg[0] += flux_i
            gg[1] += fl_final[lev]
        elif s == 6:
            lev = fl_of_global[int(r["lev_b"])]
            add("collisional ionization / 3-body", x[0] * fl_final[lev], x[2] * x_fin_sl[gh])
        elif s == 8:
            add("radiative + dielectronic recombination", 0.0, float(r["v1"]) * x_fin_sl[gh])
            add("of which dielectronic (SE)", 0.0, float(r["v1"]) * x_fin_sl[gh])
        elif s == 10:
            add("non-thermal ionization", float(r["v1"]) * x_fin_sl[0], 0.0)
        elif s == 9:
            kase, v = int(r["lev_a"]), float(r["v1"])
            src = int(r["src"])
            if kase in (1, 4):
                add("charge exchange", v * x_fin_sl[src], 0.0)
            else:
                add("charge exchange", 0.0, v * x_fin_sl[src])
    lad_rows, lad_groups = [], {}
    if lrow is not None:
        ne = num(lrow["n_e"])
        lad_rows = {"photoionization": (num(lrow["gamma_phot"]), None),
                    "radiative + dielectronic recombination": (None, ne * num(lrow["alpha"])),
                    "collisional ionization / 3-body": (num(lrow["c_ion"]), num(lrow["c_rec"])),
                    "non-thermal ionization": (num(lrow["gamma_nt"]), None)}
        lseq = lrow["lseq"]
        gl_to_fl = {int(r["global_level"]): int(r["i_fl"]) for r in sm}
        for r in ladlev:
            if r["lseq"] != lseq:
                continue
            fl = gl_to_fl.get(int(r["global_level"]))
            if fl is None:
                g = "level outside the NLTE pair"
            else:
                k = int(sl_of[fl])
                g = "ground" if fl == 0 else ("explicit excited" if len(members[k]) == 1 else f"super-level {k}")
            gg = lad_groups.setdefault(g, [0.0, 0.0])
            gg[0] += num(r["contribution"])
            gg[1] += num(r["f_lev"])

    # links (prereg 5) and branch (prereg 8)
    links = {"stage correction log10(R_s/R_0)": dex(R_s, R_0),
             "formulation log10(R_0/R_F)": dex(R_0, R_F),
             "super-level log10(R_F/R_L)": dex(R_F, R_L),
             "solve log10(R_L/R_raw)": dex(R_L, R_raw),
             "damping/post log10(R_raw/R_c)": dex(R_raw, R_c)}
    gap = dex(R_s, R_c)
    defined = all(math.isfinite(v) for v in links.values()) and math.isfinite(gap)
    closure = gap - sum(links.values()) if defined else float("nan")

    def branch(th):
        if not defined or g_eq or g_trace or g_table or abs(closure) > RES_DEX:
            return 3
        form = abs(links["formulation log10(R_0/R_F)"]) > th
        alg = any(abs(v) > th for k, v in links.items() if not k.startswith("formulation"))
        return 2 if form and alg else 0 if form else 1 if alg else 3
    codes = {th: branch(th) for th in (RES_DEX / 2, RES_DEX, RES_DEX * 2)}
    code = codes[RES_DEX] if len(set(codes.values())) == 1 else 3
    label = {0: "FORMULATION", 1: "ALGORITHM", 2: "BOTH", 3: "INDETERMINATE"}[code]

    # public row (prereg 6): 19.480 d, shell-10 vel_mid, nearest public row
    t_shell = dump_row(run, t_it, "shell", S)[0]
    vel = (float(t_shell["v_inner_cm_s"]) + float(t_shell["v_outer_cm_s"])) / 2e5
    pub, pub_vel = public_si_ratio(vel)
    if not (math.isclose(vel, REGISTERED_VEL["lumina_vel_mid_kms"]) and
            math.isclose(pub_vel, REGISTERED_VEL["public_vel_mid_kms"])):
        g_table.append(f"PUBLIC_CORRESPONDENCE:{vel}->{pub_vel}")
    o_nlte = float(o_ion["Z14_nlte_q2_cm3"]) / float(o_ion["Z14_nlte_q1_cm3"])
    o_stage = float(o_ion["Z14_stage_q2_cm3"]) / float(o_ion["Z14_stage_q1_cm3"])

    # ---------------- the table
    rows = []

    def row(group, item, se, stage, other, unit, note=""):
        vals = [v for v in (se, stage, other) if v is not None]
        rows.append({"group": group, "item": item, "nlte_se": se, "stage": stage, "public_or_link": other,
                     "unit": unit, "defined": int(all(math.isfinite(v) for v in vals)), "note": note})
    row("III/II material 13", "T committed (this run)", R_c, R_s, pub, "ratio", "public = CMFGEN 19.480 d")
    row("III/II material 13", "O stored dump", o_nlte, o_stage, pub, "ratio")
    row("III/II material 13", "log10 vs public (T)", dex(R_c, pub), dex(R_s, pub), None, "dex")
    row("III/II material 13", "gap log10(stage/NLTE) (T)", None, None, gap, "dex")
    row("actual solve", "path", None, None, None, "-",
        "GTH generator" if used_gen else "dense with conservation row")
    row("actual solve", "max componentwise backward error", float(np.nanmax(eta_solve)), None,
        float(norm_res), "1", "public_or_link = normalization residual")
    row("actual solve", "raw III/II", R_raw, None, None, "ratio")
    row("physical residual", "SL rows at raw solution (max)", float(np.nanmax(eta_phys_raw)), None,
        None, "1")
    row("physical residual", "SL rows at committed populations, fixed rates (max)",
        float(np.nanmax(eta_phys_commit)), None, None, "1", "fixed-rate substitution")
    row("physical residual", "FL rows at FINAL_FL (max)", float(np.nanmax(eta_fl_final)), None,
        None, "1", ";".join(sorted(set(fl_undefined))) or "")
    if fl_sum_rows is not None:
        g_x = g_a @ x_fin_sl
        sc = np.abs(g_a) @ np.abs(x_fin_sl)
        dd = np.where(sc > 0, np.abs(fl_sum_rows - g_x) / sc, 0.0)
        row("physical residual", "SL sum of FL residuals vs SL residual (max rel)", float(dd.max()),
            None, None, "1")
    for name in sorted(set(chan) | set(lad_rows)):
        i_f, r_f = chan.get(name, [0.0, 0.0])
        gs, as_ = lad_rows.get(name, (None, None))
        if i_f > 0 or gs is not None:
            row("channel per ion", f"{name}: ionization per Si II", div(i_f, n_lo_pop), gs,
                dex(gs, div(i_f, n_lo_pop)) if gs is not None else None, "s^-1", "public_or_link = log10(stage/SE)")
        if r_f > 0 or as_ is not None:
            row("channel per ion", f"{name}: recombination per Si III", div(r_f, n_hi_pop), as_,
                dex(as_, div(r_f, n_hi_pop)) if as_ is not None else None, "s^-1", "public_or_link = log10(stage/SE)")
        if not name.startswith("of which"):
            row("channel flux", f"{name}: I, R, F=I-R (SE)", i_f, r_f, i_f - r_f, "cm^-3 s^-1",
                "nlte_se=I, stage=R, public_or_link=F")
    gi = sum(v[0] for k, v in chan.items() if not k.startswith("of which"))
    gr = sum(v[1] for k, v in chan.items() if not k.startswith("of which"))
    gi, gr = div(gi, n_lo_pop), div(gr, n_hi_pop)
    row("channel per ion", "all channels: ionization / recombination ratio (SE)", div(gi, gr), R_0, dex(R_0, div(gi, gr)),
        "ratio", "SE: stationary => equals III/II of FINAL_FL; stage = num/den")
    for g in sorted(set(groups) | set(lad_groups)):
        se_flux, se_pop = groups.get(g, [0.0, 0.0])
        lg, lf = lad_groups.get(g, [float("nan"), float("nan")])
        row("photoionization by level group", f"{g}: rate per Si II", div(se_flux, n_lo_pop), lg,
            dex(lg, div(se_flux, n_lo_pop)), "s^-1", "stage = LTE-weighted contribution")
        row("photoionization by level group", f"{g}: fraction of Si II", div(se_pop, n_lo_pop), lf,
            dex(lf, div(se_pop, n_lo_pop)), "1", "stage = LTE fraction at T_e")
    gate_bits = int(adopted["gates"])
    for bit, name in ((1, "collisional ionization (pair)"), (2, "metastable floor"), (4, "time-dependent term"),
                      (8, "top-ion IV drain"), (16, "OMEGA-CMFGEN collisions"), (32, "Fe III Zhang data")):
        row("gates", name, float(bool(gate_bits & bit)), None, None, "on=1")
    bb_net = bb_level @ fl_final
    row("bb and collisional excitation", "net change of Si II / Si III totals", float(bb_net[~upper_fl].sum()),
        float(bb_net[upper_fl].sum()), None, "cm^-3 s^-1", "intra-ion redistribution only")
    for name, a_, b_ in (("raw SL -> XFL", R_raw, R_xfl), ("XFL -> FINAL_FL", R_xfl, R_fin),
                         ("FINAL_FL -> committed (CE damping)", R_fin, R_c)):
        row("consumption path", name, dex(b_, a_), None, None, "dex")
    row("consumption path", "stage before -> after dJ/dN", None, dex(R_s_ladder, R_0), None, "dex")
    for k, v in links.items():
        row("conditional links", k, None, None, v, "dex")
    row("conditional links", "fixed solves: max backward error (SL, FL)", eta_L, eta_F, None, "1",
        f"SL:{why_L or 'ok'} FL:{why_F or 'ok'}")
    row("conditional links", "closure gap - sum(links)", None, None, closure, "dex")
    row("attribution", f"branch {code} {label}", None, None, float(code), "code",
        f"codes at 0.005/0.01/0.02 dex: {codes}")

    fields = ["group", "item", "nlte_se", "stage", "public_or_link", "unit", "defined", "note"]
    with open(out / "si10_trace_table.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: ("" if v is None else f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else
                     ("—" if isinstance(v, float) else str(v)))
    with open(out / "si10_trace_table.md", "x") as f:
        f.write("| " + " | ".join(fields) + " |\n|" + "---|" * len(fields) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[k]) for k in fields) + " |\n")
    evidence = {"schema": "DET_SI_TRACE_ANALYSIS_V1", "run": str(run), "gates": gates,
                "adopted": {"bundle": last_bundle, "ne_iter": ne_acc, "ce_pass": int(adopted["ce_pass"]),
                            "call": int(cid), "N": N, "N_fl": N_fl, "n_lo_super": n_lo},
                "ratios": {"R_s": R_s, "R_s_ladder": R_s_ladder, "R_0": R_0, "R_F": R_F, "R_L": R_L, "R_raw": R_raw,
                           "R_xfl": R_xfl, "R_final": R_fin, "R_c": R_c, "public": pub, "O_nlte": o_nlte,
                           "O_stage": o_stage},
                "links": links, "gap": gap, "closure": closure, "branch": code, "label": label,
                "branch_codes_by_threshold": {str(k): v for k, v in codes.items()},
                "fixed_solves": {"SL": why_L or "ok", "FL": why_F or "ok"}, "sites_in_adopted_call": sites_seen,
                "replay": replay_report, "equivalence": eq_rows, "fl_undefined": sorted(set(fl_undefined)),
                "cost": {"allocated_gpu_hours": gpu_h, "footer": str(run / "RUN_FOOTER.txt")},
                "inputs": {p.name: sha(p) for p in sorted(tr.glob("si10_*.csv"))}}
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str) + "\n")
    print(f"BRANCH {code} {label} gap={gap:.6g} links={links}")
    bad = g_run + g_trace + g_table + g_eq
    if bad:
        print("GATE FINDINGS: " + "; ".join(bad))
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scope")
    s.add_argument("--out", required=True)
    s = sp.add_parser("stage")
    s.add_argument("--build", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--scope", required=True)
    s.add_argument("--gpu-profile", default="h200x2", choices=("h200x2", "h100x2"))
    s = sp.add_parser("analyze")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "scope":
        return cmd_scope(a.out)
    if a.cmd == "stage":
        return cmd_stage(a.build, a.run, a.scope, a.gpu_profile)
    return cmd_analyze(a.run, a.out)




if __name__ == "__main__":
    sys.exit(main())
