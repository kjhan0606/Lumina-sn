#!/usr/bin/env python3
"""DET-SIIII-BT-SEED readout (prereg docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md, revision 2: partial B-T).

Subcommands
  selftest --production-source T3 --baseline-source T2 --profile E/input/fixed_te_profile.txt
           --seed-profile B/input/bt_seed_profile.txt --expectations A0/expectations.json --out DIR
      generates and seals the dedicated generation-zero profile (shells of the registered T_inner set
      take E's T_inner, every other shell its sealed FIXED_T value; reused only if byte- and
      input-identical), then offline requirement 2 (G-INTERVENTION, G-LINK pre-run) and the readout
      direct tests (G-READOUT): T2+changeset tree check, owner/call-site check, env-universe check,
      the C harness compiled against T2 and T3 (OFF byte identity, uniform-profile ON byte identity,
      sealed-profile ON: bootstrap guard pass, per-shell non-identity where T_seed != T_inner and
      byte identity where T_seed == T_inner, injected faults), the extended seed publication test and
      the unmodified T2 loader test re-linked to T3.
  stage    --baseline-run E --production-source T3 --build B3 --offline DIR --seed-profile FILE
           --expectations FILE --gpu-profile h200x2|h100x2 --run X
      verifies the offline, seal and build records and creates the one run root X (never submits).
  analyze  --run X --baseline-run E --prior-analysis A0 --production-source T3 --public-spectrum FILE
           --selftest DIR --out DIR
      primary seed delivery, registered generation-7 match and reduced-model applicability,
      2542 A actual response against the registered P1/P2 records, public comparison and the
      122-key call-1 summary; codes per proposition (0 SURVIVES, 1 REFUTED, 4 UNRESOLVED,
      5 NOT_IDENTIFIABLE, X inadmissible).
Every number is an interval [lo, hi] with a middle value; 1e-3, 0.05 dex and the registered
targets are science criteria, never numerical tolerances.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import platform
import re
import shlex
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
from det_siiii_pop_readout import I, recover, sha, C                            # noqa: E402

P = Path(__file__).resolve().parent.parent
GPU_RUNNER = Path("/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_gpu.slurm")
CRIT = 1e-3
EXPECTATIONS_SHA256 = "5a26aba7213ee1fecce8f4ff75c21bd2bba1d45692ec394ed56cee85d9e5262a"
TARGET = {"r1206": 600693, "lo": 0, "up": 4, "r2542": 1097363, "u5": 5, "shells": [10, 15, 19, 24, 30],
          "spatial_2542": [17, 18, 19, 20, 21, 22], "main": 19, "expectations_sha256": EXPECTATIONS_SHA256}
Z, ION = 14, 2
CODE = {"SURVIVES": 0, "REFUTED": 1, "UNRESOLVED": 4, "NOT_IDENTIFIABLE": 5}
SRC_CORE = ("src/lumina_atomic.c", "src/lumina_plasma.c", "src/lumina_cmfgen.c", "src/lumina_cuda.cu")
CHANGED = {"src/lumina_atomic.c", "src/lumina_plasma.c", "src/env_universe.h", "tests/seed_te_publish_selftest.c"}
BROOT = GPU_RUNNER.parent.parent
# prereg 2.1: registered source set of the dedicated generation-zero profile (guard_diag1 failing single shells)
SEED_TINNER_SHELLS = (45, 46, 47, 48, 49)
GUARD_DIAG = BROOT / "results/guard_diag1/guard_diag.json"
GUARD_TOOL = BROOT / "tools/diag_guard_shells.py"
GUARD_BINARY = BROOT / "results/t1/selftest/offline_work/bin/harness_T3"
BT_KNOBS = ["LUMINA_BT_SEED", "LUMINA_BT_SEED_PROFILE"]
EMISS_EXACT_ZERO = 2         # src/emissivity_publication.h: EMISS_OK = 1, EMISS_EXACT_ZERO
ADDED = {"tests/det_siiii_btseed_selftest.c", "validation/instrumentation_debt/ENV_UNIVERSE.json"}
LIB_SRCS = ["src/lumina_plasma.c", "src/population_contract.c", "src/opacity_publication.c", "src/emissivity_publication.c",
            "src/radeq_publication.c", "src/line_net_rate.c", "src/cmfgen_adiabatic.c", "src/nlte_population_candidate.c",
            "src/atomic_internal_energy.c", "src/bf_rate_jnu.c", "src/radiation_field.c", "src/seed_capability.c",
            "src/lumina_element_wide.c", "src/lumina_atomic.c"]
LOADER_CASES = ("NL2-loader-missing-shell", "NL2-loader-duplicate-shell", "NL2-loader-excess-shell",
                "NL2-loader-out-of-domain", "NL3-loader-negative", "NL3-loader-zero", "NL3-loader-nan",
                "NL3-loader-inf", "NL3-loader-parse")
FAULTS = ("drop_jbar_err", "recompute_registered", "omit_n0", "fail_outside")

status = MECH.det_1206mech_status
allst = MECH.det_1206mech_all
absdev = MECH.det_1206mech_absdev
ivj = MECH.ivj
up, dn = MECH.up, MECH.dn
cols = REP.det_siiii_repair_cols
log10i = REP.det_siiii_repair_log10


def det_siiii_btseed_code(st):
    return CODE.get(st, None)


def det_siiii_btseed_universal(states):
    """prereg 6.2: one confirmed counterexample -> REFUTED; else any missing -> NOT_IDENTIFIABLE;
    else any boundary overlap -> UNRESOLVED; else SURVIVES (missing never erases a counterexample)."""
    s = list(states)
    if not s:
        return "NOT_IDENTIFIABLE"
    if "REFUTED" in s:
        return "REFUTED"
    if "NOT_IDENTIFIABLE" in s:
        return "NOT_IDENTIFIABLE"
    if "UNRESOLVED" in s:
        return "UNRESOLVED"
    return "SURVIVES"


def det_siiii_btseed_iv(x):
    return None if x is None else I(x[0], x[1], x[2]) if isinstance(x, (list, tuple)) else I(x["lo"], x["mid"], x["hi"])


def det_siiii_btseed_exports(path):
    """declare -x NAME="value" lines -> ordered dict (shell-unquoted values)."""
    out = {}
    for ln in Path(path).read_text().splitlines():
        if not ln.startswith("declare -x "):
            continue
        body = ln[len("declare -x "):]
        name, _, raw = body.partition("=")
        out[name] = shlex.split(raw)[0] if raw else ""
    return out


def det_siiii_btseed_tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def det_siiii_btseed_names(header):
    return set(re.findall(r'^\s+"(LUMINA_[A-Z0-9_]+)",', Path(header).read_text(), re.M))


def det_siiii_btseed_function_body(text, signature):
    i = text.find(signature)
    if i < 0:
        return None
    j = text.find("\n}\n", i)
    return text[i:j + 3] if j > 0 else None


def det_siiii_btseed_owner_check(t3, t2):
    """the generation-zero owner (lumina_atomic.c load_tardis_reference_data) calls the B-T switch and
    profile seed; OFF keeps the T_inner loop; the warm start (lumina_cmfgen.c) is unchanged; the seed
    publication carries the ON path."""
    why = []
    a3 = (Path(t3) / "src/lumina_atomic.c").read_text(errors="replace")
    body = det_siiii_btseed_function_body(a3, "int load_tardis_reference_data(")
    seed = None if body is None else body.find("Generation-zero material-temperature seed")
    if body is None or seed is None or seed < 0:
        why.append("OWNER_BLOCK_MISSING")
    else:
        blk = body[seed:seed + 1500]
        if "det_siiii_btseed_mode()" not in blk or "det_siiii_btseed_profile_seed(geo->n_shells, opacity->t_electrons" not in blk:
            why.append("OWNER_NOT_CALLING_BTSEED")
        if "opacity->t_electrons[i] = config->T_inner;" not in blk:
            why.append("OFF_TINNER_LOOP_MISSING")
    ps = det_siiii_btseed_function_body(a3, "int det_siiii_btseed_profile_seed(")
    if ps is None or 'getenv("LUMINA_BT_SEED_PROFILE")' not in ps or "LUMINA_FIXED_TE_PROFILE" in ps \
            or "source=BT_SEED_PROFILE" not in ps:
        why.append("OWNER_NOT_READING_SEED_PROFILE")
    if sha(Path(t3) / "src/lumina_cmfgen.c") != sha(Path(t2) / "src/lumina_cmfgen.c"):
        why.append("WARM_START_SOURCE_CHANGED")
    p3 = (Path(t3) / "src/lumina_plasma.c").read_text(errors="replace")
    pub = det_siiii_btseed_function_body(p3, "int lumina_publish_seed_te(")
    if pub is None or "det_siiii_btseed_mode()" not in pub or "A210_TE_LANE_FIXED_T" not in pub \
            or "BT_SEED_PROFILE_TE_MISMATCH" not in pub or 'getenv("LUMINA_BT_SEED_PROFILE")' not in pub \
            or 'getenv("LUMINA_FIXED_TE_PROFILE")' in pub or "source=BT_SEED_PROFILE" not in pub:
        why.append("PUBLICATION_ON_PATH_MISSING")
    return why


def det_siiii_btseed_tree_check(t3, t2):
    a, b = det_siiii_btseed_tree(t2), det_siiii_btseed_tree(t3)
    changed = {k for k in a if k in b and a[k] != b[k]}
    added = set(b) - set(a)
    removed = set(a) - set(b)
    why = []
    if changed != CHANGED:
        why.append(f"TREE_CHANGED_SET:{sorted(changed ^ CHANGED)}")
    if added != ADDED:
        why.append(f"TREE_ADDED_SET:{sorted(added ^ ADDED)}")
    if removed:
        why.append(f"TREE_REMOVED:{sorted(removed)}")
    return why, {"changed": sorted(changed), "added": sorted(added), "removed": sorted(removed), "n_t2": len(a), "n_t3": len(b)}


# ------------------------------------------------------------------ offline harness orchestration
def det_siiii_btseed_compile(name, cwd, args, out, log):
    t0 = time.time()
    cmd = ["gcc"] + args
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    return {"name": name, "cmd": " ".join(shlex.quote(c) for c in cmd), "cwd": str(cwd), "rc": rc,
            "elapsed_s": time.time() - t0, "binary": str(out), "binary_sha256": sha(out) if Path(out).exists() else None}


def det_siiii_btseed_run(name, binary, args, env, cwd, timeout=900):
    t0 = time.time()
    Path(cwd).mkdir(parents=True, exist_ok=False)
    with open(Path(cwd) / "stdout.log", "w") as fo, open(Path(cwd) / "stderr.log", "w") as fe:
        try:
            rc = subprocess.run([str(binary)] + [str(a) for a in args], cwd=cwd, env=env, stdout=fo, stderr=fe,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    return {"name": name, "rc": rc, "elapsed_s": time.time() - t0, "cwd": str(cwd), "args": [str(a) for a in args],
            "binary": str(binary)}


def det_siiii_btseed_hres(d):
    """read one harness output directory."""
    d = Path(d)
    out = {"dir": str(d)}
    sj = d / "summary.json"
    if sj.exists():
        txt = sj.read_text()
        rows = re.findall(r'"row_identity": "([^"]*)"', txt)
        txt2 = re.sub(r',\n "row_identity": "[^"]*"', "", txt)
        out["summary"] = json.loads(txt2)
        out["summary"]["row_identity"] = rows
    for nm in ("t_electrons", "T_e_published", "partition", "ion_density", "n_electron", "upper_target",
               "material_target", "tau_target"):
        p = d / f"{nm}.f64"
        if p.exists():
            out[nm] = p.read_bytes()
    p = d / "upper_status.i32"
    if p.exists():
        out["upper_status"] = p.read_bytes()
    p = d / "tau.sha256"
    if p.exists():
        lines = p.read_text().split()
        out["tau_all"] = lines[1]
        out["tau_shell"] = [lines[i + 2] for i in range(2, len(lines), 3)]
    return out


MATERIAL = ("partition", "ion_density", "n_electron", "upper_target", "material_target", "tau_target", "upper_status")


def det_siiii_btseed_identical(a, b, keys=MATERIAL):
    diff = [k for k in keys if a.get(k) is None or a.get(k) != b.get(k)]
    if a.get("tau_all") is None or a.get("tau_all") != b.get("tau_all"):
        diff.append("tau_all")
    return diff


def det_siiii_btseed_row_ip(h):
    """ion-population index of each target row (7th field of the harness row_identity)."""
    return [int(r.split(":")[6]) for r in h.get("summary", {}).get("row_identity", [])]


def det_siiii_btseed_structural_zero(on, off, ns, n_ip, n_rows):
    """(row, shell) cells whose target ion is absent in the shell in both runs (ion density exactly 0) and whose upper
    population is a valid EMISS_EXACT_ZERO with zero line material in both: temperature cannot change them, so they
    are checked separately (prereg 3.2: exact zeros are examined separately) and leave the non-identity rule."""
    f = lambda b: np.frombuffer(b, "<f8")                                        # noqa: E731
    ip_on, ip_off = det_siiii_btseed_row_ip(on), det_siiii_btseed_row_ip(off)
    if len(ip_on) != n_rows or ip_on != ip_off or any(i < 0 or i >= n_ip for i in ip_on):
        return None
    ia, ib = f(on["ion_density"]).reshape(n_ip, ns), f(off["ion_density"]).reshape(n_ip, ns)
    ua, ub = f(on["upper_target"]).reshape(n_rows, ns), f(off["upper_target"]).reshape(n_rows, ns)
    ma, mb = f(on["material_target"]).reshape(n_rows, ns, 4), f(off["material_target"]).reshape(n_rows, ns, 4)
    sa = np.frombuffer(on["upper_status"], "<i4").reshape(n_rows, ns)
    sb = np.frombuffer(off["upper_status"], "<i4").reshape(n_rows, ns)
    return {(k, s) for k in range(n_rows) for s in range(ns)
            if ia[ip_on[k], s] == 0.0 and ib[ip_on[k], s] == 0.0 and sa[k, s] == EMISS_EXACT_ZERO == sb[k, s]
            and ua[k, s] == 0.0 and ub[k, s] == 0.0 and not ma[k, s].any() and not mb[k, s].any()}


def det_siiii_btseed_shell_nonidentity(on, off, ns, n_ip, n_rows, shells, structural=frozenset()):
    """per shell: every component of the material bundle differs from OFF; for the target-row components the rows that
    are structural exact zeros in that shell are excluded (an all-structural component is reported as such)."""
    f = lambda b: np.frombuffer(b, "<f8")                                        # noqa: E731
    res = {}
    pa, pb = f(on["partition"]).reshape(n_ip, ns), f(off["partition"]).reshape(n_ip, ns)
    ia, ib = f(on["ion_density"]).reshape(n_ip, ns), f(off["ion_density"]).reshape(n_ip, ns)
    na, nb = f(on["n_electron"]), f(off["n_electron"])
    ua, ub = f(on["upper_target"]).reshape(n_rows, ns), f(off["upper_target"]).reshape(n_rows, ns)
    ma, mb = f(on["material_target"]).reshape(n_rows, ns, 4), f(off["material_target"]).reshape(n_rows, ns, 4)
    for s in shells:
        rows = [k for k in range(n_rows) if (k, s) not in structural]
        comp = {"partition": pa[:, s].tobytes() != pb[:, s].tobytes(),
                "ion_density": ia[:, s].tobytes() != ib[:, s].tobytes(),
                "n_electron": na[s].tobytes() != nb[s].tobytes(),
                "tau": on["tau_shell"][s] != off["tau_shell"][s],
                "upper_target": (all(ua[k, s].tobytes() != ub[k, s].tobytes() for k in rows) if rows
                                 else "STRUCTURAL_ZERO"),
                "material_target": (all(ma[k, s, :].tobytes() != mb[k, s, :].tobytes() for k in rows) if rows
                                    else "STRUCTURAL_ZERO")}
        res[s] = comp
    return res


def det_siiii_btseed_shell_identity(on, off, ns, n_ip, n_rows, shells):
    """per shell: every component of the material bundle (and the upper-population status) is byte-equal to OFF."""
    f = lambda b: np.frombuffer(b, "<f8")                                        # noqa: E731
    pa, pb = f(on["partition"]).reshape(n_ip, ns), f(off["partition"]).reshape(n_ip, ns)
    ia, ib = f(on["ion_density"]).reshape(n_ip, ns), f(off["ion_density"]).reshape(n_ip, ns)
    na, nb = f(on["n_electron"]), f(off["n_electron"])
    ua, ub = f(on["upper_target"]).reshape(n_rows, ns), f(off["upper_target"]).reshape(n_rows, ns)
    ma, mb = f(on["material_target"]).reshape(n_rows, ns, 4), f(off["material_target"]).reshape(n_rows, ns, 4)
    sa = np.frombuffer(on["upper_status"], "<i4").reshape(n_rows, ns)
    sb = np.frombuffer(off["upper_status"], "<i4").reshape(n_rows, ns)
    return {s: {"partition": pa[:, s].tobytes() == pb[:, s].tobytes(),
                "ion_density": ia[:, s].tobytes() == ib[:, s].tobytes(),
                "n_electron": na[s].tobytes() == nb[s].tobytes(),
                "tau": on["tau_shell"][s] == off["tau_shell"][s],
                "upper_target": ua[:, s].tobytes() == ub[:, s].tobytes(),
                "upper_status": sa[:, s].tobytes() == sb[:, s].tobytes(),
                "material_target": ma[:, s, :].tobytes() == mb[:, s, :].tobytes()} for s in shells}


def det_siiii_btseed_profile_values(path):
    vals = {}
    for ln in Path(path).read_text().splitlines():
        p = ln.split("#")[0].split()
        if len(p) >= 2:
            vals[int(p[0])] = float(p[1])
    return vals


def det_siiii_btseed_seed_rule(fixed_profile, T_in):
    """prereg 2.1: T_seed[s] = T_inner on the registered set, the sealed FIXED_T value elsewhere."""
    pin = det_siiii_btseed_profile_values(fixed_profile)
    n = len(pin)
    if sorted(pin) != list(range(n)) or not set(SEED_TINNER_SHELLS) <= set(pin):
        raise ValueError("FIXED_T_PROFILE_SHELL_IDS")
    return {s: (float(T_in) if s in SEED_TINNER_SHELLS else pin[s]) for s in range(n)}


def det_siiii_btseed_seed_text(vals):
    return "".join(f"{s} {vals[s]:.17g}\n" for s in sorted(vals))


def det_siiii_btseed_seal(seed_path, fixed_profile, e_run):
    """generate the dedicated generation-zero profile from the fixed rule and seal it (prereg 2.2): exclusive
    create, explicit file list, content hashes; an existing file is reused only if byte- and input-identical."""
    seed_path, fixed_profile, e_run = Path(seed_path), Path(fixed_profile), Path(e_run)
    rec = {"path": str(seed_path), "why": []}
    meta = MECH.det_1206mech_meta(e_run)
    T_in = meta["T_inner"]
    ex = det_siiii_btseed_exports(e_run / "input/resolved_lumina.exports")
    if Path(ex.get("LUMINA_FIXED_TE_PROFILE", "")).resolve() != fixed_profile.resolve():
        rec["why"].append("E_EXPORTS_PROFILE_NOT_THIS_FILE")
    gd = json.loads(GUARD_DIAG.read_text())
    if gd.get("failing_single_shells") != list(SEED_TINNER_SHELLS) or gd.get("T_inner") != T_in:
        rec["why"].append("GUARD_DIAG_NOT_REGISTERED_SET")
    vals = det_siiii_btseed_seed_rule(fixed_profile, T_in)
    text = det_siiii_btseed_seed_text(vals)
    if any(float(ln.split()[1]) != vals[int(ln.split()[0])] for ln in text.splitlines()):
        rec["why"].append("SEED_TEXT_NOT_ROUND_TRIP")
    inputs = {"fixed_te_profile": {"path": str(fixed_profile), "sha256": sha(fixed_profile)},
              "E_resolved_exports": {"path": str(e_run / "input/resolved_lumina.exports"),
                                     "sha256": sha(e_run / "input/resolved_lumina.exports")},
              "E_deck_sha256": {"path": str(e_run / "input/deck.sha256"), "sha256": sha(e_run / "input/deck.sha256")},
              "guard_diag_json": {"path": str(GUARD_DIAG), "sha256": sha(GUARD_DIAG)},
              "guard_diag_tool": {"path": str(GUARD_TOOL), "sha256": sha(GUARD_TOOL)},
              "guard_diag_binary": {"path": str(GUARD_BINARY), "sha256": sha(GUARD_BINARY)},
              "guard_diag_logs": {str(p.relative_to(GUARD_DIAG.parent)): sha(p)
                                  for p in sorted(GUARD_DIAG.parent.glob("*_cwd/*.log"))}}
    seal = {"schema": "DET_SIIII_BTSEED_SEED_SEAL_V1", "rule": "shells in SEED_TINNER_SHELLS take E T_inner, every other "
            "shell its sealed FIXED_T value; '%d %.17g' per line, ascending shell id",
            "T_inner": T_in, "T_inner_source": "E run meta (MECH.det_1206mech_meta)", "n_shells": len(vals),
            "shells_T_inner": list(SEED_TINNER_SHELLS), "shells_T_pin": [s for s in vals if s not in SEED_TINNER_SHELLS],
            "inputs": inputs, "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "generator": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())}}
    sp = seed_path.with_name(seed_path.name + ".seal.json")
    rec.update({k: seal[k] for k in ("T_inner", "n_shells", "shells_T_inner", "shells_T_pin", "rule", "inputs")})
    rec["seal_path"] = str(sp)
    if rec["why"]:
        return rec, vals
    if seed_path.exists() or sp.exists():
        if not seed_path.exists() or not sp.exists() or seed_path.read_bytes() != text.encode() or \
                json.loads(sp.read_text()).get("inputs") != inputs:
            rec["why"].append("EXISTING_SEED_PROFILE_DIFFERS")
            return rec, vals
        rec["reused"] = True
    else:
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seed_path, "x") as f:
            f.write(text)
        with open(sp, "x") as f:
            f.write(json.dumps(seal, indent=1))
        rec["reused"] = False
    rec["file_sha256"] = sha(seed_path)
    rec["seal_sha256"] = sha(sp)
    if rec["file_sha256"] != seal["file_sha256"]:
        rec["why"].append("SEED_FILE_HASH")
    return rec, vals


def det_siiii_btseed_strict_adapter(t3, dest, drop):
    """T3 sources whose env-universe header lacks one B-T knob (count decremented): STRICT-registration fault."""
    shutil.copytree(Path(t3) / "src", Path(dest) / "src")
    h = (Path(t3) / "src/env_universe.h").read_text()
    line = f'    "{drop}",\n'
    m = re.search(r"#define LUMINA_ENV_UNIVERSE_COUNT (\d+)\n", h)
    if h.count(line) != 1 or not m:
        raise ValueError(f"STRICT_ADAPTER_HEADER:{drop}")
    h = h.replace(line, "").replace(m.group(0), f"#define LUMINA_ENV_UNIVERSE_COUNT {int(m.group(1)) - 1}\n")
    (Path(dest) / "src/env_universe.h").write_text(h)


def det_siiii_btseed_reservoir(stdout):
    m = re.findall(r"level-less reservoir: max fraction=([0-9.eE+-]+)", stdout)
    return float(m[-1]) if m else None


def det_siiii_btseed_offline(t3, t2, profile, seed, seal, out, workers=4):
    """offline requirement 2: returns the offline record (G-LINK pre-run part, G-INTERVENTION)."""
    out = Path(out)
    rec = {"schema": "DET_SIIII_BTSEED_OFFLINE_V2", "contract_revision": 2, "checks": {}, "compile": [], "runs": [],
           "faults": {}}
    tw, tinfo = det_siiii_btseed_tree_check(t3, t2)
    rec["tree"] = {"why": tw, **tinfo}
    rec["t3_changeset_sha256"] = {f: sha(Path(t3) / f) for f in sorted(CHANGED | ADDED)}
    rec["owner_source_why"] = det_siiii_btseed_owner_check(t3, t2)
    n2, n3 = det_siiii_btseed_names(Path(t2) / "src/env_universe.h"), det_siiii_btseed_names(Path(t3) / "src/env_universe.h")
    rec["env_universe"] = {"added": sorted(n3 - n2), "removed": sorted(n2 - n3), "n_t2": len(n2), "n_t3": len(n3)}
    run_in = Path(profile).parent
    deck, exports = run_in / "model", run_in / "resolved_lumina.exports"
    T_in = seal["T_inner"]
    vals = det_siiii_btseed_seed_rule(profile, T_in)                 # independent construction from E and the rule
    ns = len(vals)
    seed_bytes = np.array([vals[s] for s in range(ns)], "<f8").tobytes()
    t_in_bytes = np.full(ns, T_in, "<f8").tobytes()
    uniform_bytes = np.full(ns, float(repr(T_in)), "<f8").tobytes()
    diff_shells = [s for s in range(ns) if vals[s] != T_in]
    same_shells = [s for s in range(ns) if vals[s] == T_in]
    rec["shells_T_seed_ne_T_inner"], rec["shells_T_seed_eq_T_inner"] = diff_shells, same_shells
    ck = rec["checks"]
    ck["seed_file_matches_rule"] = det_siiii_btseed_profile_values(seed) == vals and \
        Path(seed).read_bytes() == det_siiii_btseed_seed_text(vals).encode()
    ck["registered_shell_sets"] = (diff_shells == [s for s in range(ns) if s not in SEED_TINNER_SHELLS]
                                   and same_shells == list(SEED_TINNER_SHELLS))
    bin_ = out / "bin"
    bin_.mkdir(parents=True)
    ad_seed, ad_prof = out / "adapter_strict_no_bt_seed", out / "adapter_strict_no_bt_seed_profile"
    det_siiii_btseed_strict_adapter(t3, ad_seed, "LUMINA_BT_SEED")
    det_siiii_btseed_strict_adapter(t3, ad_prof, "LUMINA_BT_SEED_PROFILE")
    harness_src = str(Path(t3) / "tests/det_siiii_btseed_selftest.c")
    flags = ["-O2", "-std=gnu11", "-D_GNU_SOURCE", "-Isrc"]
    hjob = lambda n, cwd: (n, cwd, flags + ["-o", str(bin_ / n), harness_src] + LIB_SRCS + ["-lm", "-fopenmp"])  # noqa: E731
    jobs = [hjob("harness_T2", t2), hjob("harness_T3", t3), hjob("harness_T3_no_bt_seed", ad_seed),
            hjob("harness_T3_no_bt_seed_profile", ad_prof),
            ("seedtest_T3", t3, ["-O2", "-std=gnu11", "-D_GNU_SOURCE", "-o", str(bin_ / "seedtest_T3"),
                                 "tests/seed_te_publish_selftest.c"] + LIB_SRCS + ["-lm", "-fopenmp"]),
            ("seedtest_T2", t2, ["-O2", "-std=gnu11", "-D_GNU_SOURCE", "-o", str(bin_ / "seedtest_T2"),
                                 "tests/seed_te_publish_selftest.c"] + LIB_SRCS + ["-lm", "-fopenmp"]),
            ("stage12_T3", t3, ["-O2", "-Wall", "-Wextra", "-std=gnu11", "-D_GNU_SOURCE", "-ffunction-sections",
                                "-fdata-sections", "-Wl,--gc-sections", "-Isrc", "-o", str(bin_ / "stage12_T3"),
                                "tests/det_stage12_selftest.c", "src/lumina_plasma.c", "src/population_contract.c", "-lm",
                                "-fopenmp"])]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(det_siiii_btseed_compile, n, cwd, a, bin_ / n, bin_ / f"{n}.log") for n, cwd, a in jobs]
        rec["compile"] = [f.result() for f in futs]
    if any(c["rc"] != 0 for c in rec["compile"]):
        rec["verdict"] = "COMPILE_FAILED"
        rec["G_INTERVENTION"] = "FAIL"
        return rec
    prof_dir = out / "profiles"
    prof_dir.mkdir()

    def write_copy(name, v):
        p = prof_dir / name
        p.write_text(det_siiii_btseed_seed_text(v))
        return p
    uniform = write_copy("uniform_T_inner.txt", {s: T_in for s in range(ns)})
    swapped = write_copy("seed_shell10_19_swapped.txt", {**vals, 10: vals[19], 19: vals[10]})
    perturbed = write_copy("seed_shell19_plus1K.txt", {**vals, 19: vals[19] + 1.0})
    tinner_changed = write_copy("seed_shell45_T_inner_plus1K.txt", {**vals, 45: T_in + 1.0})
    sources_swapped = write_copy("seed_sources_44_45_swapped.txt", {**vals, 44: T_in, 45: det_siiii_btseed_profile_values(profile)[45]})
    absent = prof_dir / "seed_absent.txt"
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("LUMINA_")}
    base_env.update(det_siiii_btseed_exports(exports))
    base_env["OMP_NUM_THREADS"] = "1"
    for k in BT_KNOBS:
        base_env.pop(k, None)
    rows = f"{TARGET['r1206']},{TARGET['r2542']}"
    R = out / "runs"
    R.mkdir()

    def env(**kw):
        e = dict(base_env)
        for k, v in kw.items():
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return e
    on = lambda p=seed, **kw: env(LUMINA_BT_SEED="1", LUMINA_BT_SEED_PROFILE=str(p), **kw)     # noqa: E731
    plan = [("T2_off", "harness_T2", env(), "none"),
            ("T3_off_unset", "harness_T3", env(), "none"),
            ("T3_off_zero", "harness_T3", env(LUMINA_BT_SEED="0"), "none"),
            ("T3_on_uniform", "harness_T3", on(uniform), "none"),
            ("T3_on_seed", "harness_T3", on(), "none"),
            ("F_stale_material", "harness_T3", on(), "stale_material"),
            ("F_shell_swap", "harness_T3", on(swapped), "none"),
            ("F_tinner_shell_changed", "harness_T3", on(tinner_changed), "none"),
            ("F_owner_tinner", "harness_T3", on(), "owner_tinner"),
            ("F_perturb_te", "harness_T3", on(), "perturb_te"),
            ("F_profile_changed", "harness_T3", on(), f"swap_profile={perturbed}"),
            ("F_knob_2", "harness_T3", env(LUMINA_BT_SEED="2"), "none"),
            ("F_continue_after_reject", "harness_T3", env(LUMINA_BT_SEED="2"), "continue_after_reject"),
            ("F_seed_knob_unset", "harness_T3", env(LUMINA_BT_SEED="1"), "none"),
            ("F_seed_knob_empty", "harness_T3", on(""), "none"),
            ("F_seed_knob_relative", "harness_T3", on("input/bt_seed_profile.txt"), "none"),
            ("F_seed_knob_unreadable", "harness_T3", on(absent), "none"),
            ("F_off_with_seed_knob", "harness_T3", env(LUMINA_BT_SEED_PROFILE=str(seed)), "none"),
            ("F_strict_no_bt_seed", "harness_T3_no_bt_seed", on(), "none"),
            ("F_strict_no_bt_seed_profile", "harness_T3_no_bt_seed_profile", on(), "none")]
    tests = [("seedtest_T3", "seedtest_T3", [str(R / "seedtest_T3_work")], base_env),
             ("seedtest_T2", "seedtest_T2", [str(R / "seedtest_T2_work")], base_env),
             ("stage12_T3", "stage12_T3", [], base_env)]
    heavy = {"T2_off", "T3_off_unset", "T3_off_zero", "T3_on_uniform", "T3_on_seed", "F_stale_material", "F_shell_swap",
             "F_tinner_shell_changed"}
    order = sorted(plan, key=lambda p: p[0] not in heavy)            # long runs first
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(det_siiii_btseed_run, n, bin_ / b, [deck, R / n, rows, fault], e, R / f"{n}_cwd")
                for n, b, e, fault in order]
        for n, b, a, e in tests:
            Path(a[0]).mkdir() if a else None
            futs.append(ex.submit(det_siiii_btseed_run, n, bin_ / b, a, e, R / f"{n}_cwd"))
        rec["runs"] = [f.result() for f in futs]
    H = {n: det_siiii_btseed_hres(R / n) for n, *_ in plan}
    rc = {r["name"]: r["rc"] for r in rec["runs"]}
    log = lambda n, s="stderr": (R / f"{n}_cwd" / f"{s}.log").read_text(errors="replace")      # noqa: E731
    off = H["T2_off"]
    s0 = off.get("summary", {})
    n_ip, n_rows = s0.get("n_ion_pops", 0), 2
    ss, su = H["T3_on_seed"].get("summary", {}), H["T3_on_uniform"].get("summary", {})
    base = ("T2_off", "T3_off_unset", "T3_off_zero", "T3_on_uniform", "T3_on_seed")
    ck["base_runs_ok"] = all(rc[n] == 0 and H[n].get("summary", {}).get("upper_population_status") == "ALL_VALID"
                             and H[n].get("summary", {}).get("rc_upper_fill") == 0 for n in base)
    ck["geometry_shells_equal_seed_shells"] = s0.get("n_shells") == ns and ss.get("n_shells") == ns
    ck["owner_off_T_inner"] = off.get("t_electrons") == t_in_bytes
    ck["owner_on_seed_all_shells"] = H["T3_on_seed"].get("t_electrons") == seed_bytes
    ck["owner_on_uniform"] = H["T3_on_uniform"].get("t_electrons") == uniform_bytes
    ck["off_unset_byte_identical"] = det_siiii_btseed_identical(H["T3_off_unset"], off,
                                                                MATERIAL + ("t_electrons", "T_e_published")) == []
    ck["off_zero_byte_identical"] = det_siiii_btseed_identical(H["T3_off_zero"], off,
                                                               MATERIAL + ("t_electrons", "T_e_published")) == []
    pub_keys = ("rc_publish", "T_e_generation", "committed_te_generation", "te_lane", "re_root_required", "pinned_shells",
                "te_profile_sha256", "te_manifest_sha256")
    ck["off_publication_identical"] = bool(s0) and all(
        H[n].get("summary", {}).get(k) == s0.get(k) for n in ("T3_off_unset", "T3_off_zero") for k in pub_keys)
    ck["uniform_material_byte_identical"] = det_siiii_btseed_identical(H["T3_on_uniform"], off) == []
    ck["on_publication_metadata"] = all(
        x.get("rc_publish") == 0 and x.get("te_lane") not in (None, s0.get("te_lane")) and x.get("re_root_required") == 0
        and x.get("pinned_shells") == ns and len(x.get("te_profile_sha256", "")) == 64
        and x.get("te_profile_sha256") == x.get("te_manifest_sha256") and x.get("T_e_generation") == 1 for x in (su, ss))
    ck["uniform_manifest_equals_off_manifest"] = su.get("te_manifest_sha256") == s0.get("te_manifest_sha256")
    # H_seed (production loader on the sealed file) and H_fixed (E's own FIXED_T lane log)
    h_seed = ss.get("te_profile_sha256")
    so_seed = log("T3_on_seed", "stdout")
    own = re.findall(r"\[BT-SEED\] generation-zero material temperature source=BT_SEED_PROFILE path=(\S+) "
                     r"te_profile_sha256=([0-9a-f]{64}) shells=(\d+) ", so_seed)
    sd = re.findall(r"\[BT-SEED\]\[SEED\] bootstrap LTE/Saha seed published source=BT_SEED_PROFILE path=(\S+) te_lane=FIXED_T "
                    r"re_root_required=0 pinned_shells=(\d+) te_profile_sha256=([0-9a-f]{64}) te_manifest_sha256=([0-9a-f]{64}) "
                    r"generation=1\n", so_seed)
    ck["seed_log_chain"] = (own == [(str(seed), h_seed, str(ns))] and sd == [(str(seed), str(ns), h_seed, h_seed)])
    eerr = (run_in.parent / "stderr.log").read_text(errors="replace")
    lane = re.findall(r"\[A2-10\]\[TE_PUBLICATION\] te_lane=FIXED_T te_profile_sha256=([0-9a-f]{64}) pinned_shells=(\d+)", eerr)
    h_fixed = lane[0][0] if lane and all(v == lane[0] for v in lane) else None
    ck["h_fixed_from_E_lane"] = h_fixed is not None and len(lane) == 6 and lane[0][1] == str(ns)
    ck["h_seed_ne_h_fixed"] = bool(h_seed) and bool(h_fixed) and h_seed != h_fixed
    rec["seed_profile"] = dict(seal, H_seed=h_seed, H_fixed=h_fixed, H_seed_source="T3_on_seed harness: production loader "
                               "(det_siiii_btseed_profile_seed) and lumina_publish_seed_te on the sealed file",
                               H_fixed_source="E stderr [A2-10][TE_PUBLICATION] te_lane=FIXED_T lines (production FIXED_T "
                               "loader on E/input/fixed_te_profile.txt)",
                               fixed_te_profile_file_sha256=sha(profile))
    # bootstrap guard on the whole sealed profile
    rres = det_siiii_btseed_reservoir(so_seed)
    rec["reservoir_max_fraction"] = {n: det_siiii_btseed_reservoir(log(n, "stdout")) for n in base}
    # the production A2-07 guard itself decides (x30 bound vs 1e-2); this records that it ran and did not stop the bootstrap
    ck["seed_bootstrap_guard_pass"] = (rc["T3_on_seed"] == 0 and ss.get("rc_tau") == 0 and rres is not None
                                       and "POP_INVALID_PARTITION" not in log("T3_on_seed") and "[FATAL]" not in log("T3_on_seed"))
    if ck["base_runs_ok"]:
        sz = det_siiii_btseed_structural_zero(H["T3_on_seed"], off, ns, n_ip, n_rows)
        ck["structural_zero_cells_resolved"] = sz is not None
        sz = sz or set()
        rec["structural_zero_cells"] = [{"row": [TARGET["r1206"], TARGET["r2542"]][k], "shell": sh} for k, sh in sorted(sz)]
        nonid = det_siiii_btseed_shell_nonidentity(H["T3_on_seed"], off, ns, n_ip, n_rows, diff_shells, sz)
        ident = det_siiii_btseed_shell_identity(H["T3_on_seed"], off, ns, n_ip, n_rows, same_shells)
        rec["seed_nonidentity"] = {str(s): v for s, v in nonid.items()}
        rec["seed_identity"] = {str(s): v for s, v in ident.items()}
        ck["seed_material_differs_every_diff_shell"] = len(nonid) > 0 and all(all(v.values()) for v in nonid.values())
        ck["seed_material_identical_every_same_shell"] = len(ident) > 0 and all(all(v.values()) for v in ident.values())
    else:
        ck["seed_material_differs_every_diff_shell"] = ck["seed_material_identical_every_same_shell"] = False
    ck["seedtest_T3_pass"] = rc["seedtest_T3"] == 0 and "verdict=PASS" in log("seedtest_T3", "stdout")
    ck["seedtest_T2_baseline_pass"] = rc["seedtest_T2"] == 0 and "verdict=PASS" in log("seedtest_T2", "stdout")
    # the T2 loader test returns 1 at the first failing case and prints one summary line otherwise;
    # the registered case names must be present (unchanged) in the T2 source that was linked to T3
    st12 = log("stage12_T3", "stdout")
    src12 = (Path(t3) / "tests/det_stage12_selftest.c").read_text(errors="replace")
    case_ok = {c: f'"{c}"' in src12 for c in LOADER_CASES}
    rec["loader_cases_T3"] = {"cases_in_source": case_ok,
                              "source_identical_to_T2": sha(Path(t3) / "tests/det_stage12_selftest.c") ==
                              sha(Path(t2) / "tests/det_stage12_selftest.c"),
                              "summary_line": "PASS det_stage12 NL1..NL5 production_loader" in st12}
    ck["t2_loader_test_on_T3_pass"] = (rc["stage12_T3"] == 0 and all(case_ok.values()) and
                                       rec["loader_cases_T3"]["source_identical_to_T2"] and rec["loader_cases_T3"]["summary_line"])
    rec["env_check_ok"] = rec["env_universe"]["added"] == sorted(BT_KNOBS) and rec["env_universe"]["removed"] == []
    # faults: each must be detected by the registered check
    F = rec["faults"]
    if ck["base_runs_ok"] and rc["F_stale_material"] == 0:
        nonid = det_siiii_btseed_shell_nonidentity(H["F_stale_material"], off, ns, n_ip, n_rows, diff_shells,
                                                   det_siiii_btseed_structural_zero(H["F_stale_material"], off, ns, n_ip,
                                                                                    n_rows) or set())
        F["stale_material"] = {"detected": not all(all(v.values()) for v in nonid.values()),
                               "reason": "material bundle equals T2 OFF in shells with T_seed != T_inner"}
    else:
        F["stale_material"] = {"detected": False, "reason": f"run rc={rc['F_stale_material']}"}
    F["shell_swap"] = {"detected": rc["F_shell_swap"] == 0 and H["F_shell_swap"].get("t_electrons") != seed_bytes,
                       "reason": "owner array does not match the sealed profile shell by shell"}
    tc = H["F_tinner_shell_changed"]
    tc_def = tc.get("t_electrons") is not None and tc.get("t_electrons") != seed_bytes
    tc_id = None
    if rc["F_tinner_shell_changed"] == 0 and ck["base_runs_ok"]:
        tc_id = det_siiii_btseed_shell_identity(tc, off, ns, n_ip, n_rows, [45])[45]
    F["tinner_shell_changed"] = {"detected": tc_def and (tc_id is None or not all(tc_id.values())),
                                 "definition_mismatch": tc_def, "rc": rc["F_tinner_shell_changed"],
                                 "shell45_identity": tc_id}
    F["source_sets_swapped"] = {"detected": det_siiii_btseed_profile_values(sources_swapped) != vals,
                                "reason": "file values differ from the rule-built T_seed (section 2.1 definition)"}
    for n, key in (("F_owner_tinner", "owner_returns_T_inner"), ("F_perturb_te", "one_shell_changed_before_publication"),
                   ("F_profile_changed", "profile_changed_before_publication")):
        sm = H[n].get("summary", {})
        F[key] = {"detected": rc[n] == 5 and sm.get("rc_publish", 0) != 0 and sm.get("T_e_generation") == 0
                  and "BT_SEED_PROFILE_TE_MISMATCH" in log(n) and "rc_tau" not in sm, "rc": rc[n]}
    F["owner_returns_T_inner"]["owner_array_check_fails"] = H["F_owner_tinner"].get("t_electrons") != seed_bytes
    load_reject = lambda n, txt: (rc[n] == 3 and H[n].get("summary", {}).get("rc_load_reference", 0) != 0    # noqa: E731
                                  and "rc_publish" not in H[n].get("summary", {}) and "rc_tau" not in H[n].get("summary", {})
                                  and txt in log(n))
    F["knob_value_2"] = {"detected": load_reject("F_knob_2", "LUMINA_BT_SEED='2'"), "rc": rc["F_knob_2"]}
    sm = H["F_continue_after_reject"].get("summary", {})
    F["publication_after_rejection"] = {"detected": sm.get("rc_publish", 0) != 0 and sm.get("T_e_generation") == 0
                                        and "rc_tau" not in sm, "rc": rc["F_continue_after_reject"]}
    for n, txt in (("F_seed_knob_unset", "requires an absolute LUMINA_BT_SEED_PROFILE"),
                   ("F_seed_knob_empty", "requires an absolute LUMINA_BT_SEED_PROFILE"),
                   ("F_seed_knob_relative", "requires an absolute LUMINA_BT_SEED_PROFILE"),
                   ("F_seed_knob_unreadable", "generation-zero profile rejected"),
                   ("F_off_with_seed_knob", "LUMINA_BT_SEED_PROFILE is set")):
        F[n[2:]] = {"detected": load_reject(n, txt), "rc": rc[n]}
    for n, nm in (("F_strict_no_bt_seed", "LUMINA_BT_SEED"), ("F_strict_no_bt_seed_profile", "LUMINA_BT_SEED_PROFILE")):
        unk = re.findall(r"^\[ENV-SURFACE\]   unknown: (\S+)$", log(n, "stdout"), re.M)
        F[f"{nm.lower()}_missing_from_universe"] = {"detected": rc[n] == 3 and unk == [nm] and "BLOCKED_UNKNOWN_ENV" in log(n),
                                                    "unknown": unk, "rc": rc[n]}
    # warm-start-only intervention: an adapter tree whose atomic.c is T2's and whose cmfgen.c changes fs.J
    wa = out / "adapter_warmstart"
    shutil.copytree(Path(t3) / "src", wa / "src")
    shutil.copy2(Path(t2) / "src/lumina_atomic.c", wa / "src/lumina_atomic.c")
    cm = (wa / "src/lumina_cmfgen.c").read_text(errors="replace").replace("fs.J[idx] = Bnu;", "fs.J[idx] = Bnu_pin;", 1)
    (wa / "src/lumina_cmfgen.c").write_text(cm)
    ow = det_siiii_btseed_owner_check(wa, t2)
    F["warm_start_only"] = {"detected": "OWNER_NOT_CALLING_BTSEED" in ow and "WARM_START_SOURCE_CHANGED" in ow, "why": ow}
    rec["fault_all_detected"] = all(v["detected"] for v in F.values())
    rec["upper_population_status"] = {n: H[n].get("summary", {}).get("upper_population_status") for n in base}
    rec["summaries"] = {n: H[n].get("summary") for n, *_ in plan}
    gi = (not tw and not rec["owner_source_why"] and rec["env_check_ok"] and all(ck.values()) and rec["fault_all_detected"])
    rec["G_INTERVENTION"] = "PASS" if gi else "FAIL"
    rec["verdict"] = rec["G_INTERVENTION"]
    return rec


# ------------------------------------------------------------------ analyze
def det_siiii_btseed_log_hashes(run):
    so = (Path(run) / "stdout.log").read_text(errors="replace") if (Path(run) / "stdout.log").exists() else ""
    se = (Path(run) / "stderr.log").read_text(errors="replace") if (Path(run) / "stderr.log").exists() else ""
    owner = re.findall(r"\[BT-SEED\] generation-zero material temperature source=(\S+) path=(\S+) "
                       r"te_profile_sha256=([0-9a-f]{64}) shells=(\d+) ", so)
    seed = re.findall(r"\[BT-SEED\]\[SEED\] bootstrap LTE/Saha seed published source=(\S+) path=(\S+) te_lane=FIXED_T "
                      r"re_root_required=0 pinned_shells=(\d+) te_profile_sha256=([0-9a-f]{64}) "
                      r"te_manifest_sha256=([0-9a-f]{64}) generation=(\d+)", so)
    lane = re.findall(r"\[A2-10\]\[TE_PUBLICATION\] te_lane=FIXED_T te_profile_sha256=([0-9a-f]{64})", se)
    return {"owner": owner, "seed": seed, "fixed_lane": lane, "n_bt_lines": so.count("[BT-SEED]")}


def det_siiii_btseed_hash_chain(hX, hE, h_seed, h_fixed, seed_path, ns):
    """prereg 2.4/4.2: owner and bootstrap seed logs once each with H_seed, the dedicated path, all shells and generation 1;
    the six later FIXED_T publications (X and E) with H_fixed; the two roles never exchanged."""
    why = []
    hexok = lambda h: isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) is not None     # noqa: E731
    if not hexok(h_seed) or not hexok(h_fixed) or h_seed == h_fixed:
        return ["LINK_STAGE_SEED_FIXED_HASHES"]
    if hX["owner"] != [("BT_SEED_PROFILE", seed_path, h_seed, str(ns))]:
        why.append("LINK_BTSEED_OWNER_LOG")
    if hX["seed"] != [("BT_SEED_PROFILE", seed_path, str(ns), h_seed, h_seed, "1")]:
        why.append("LINK_BTSEED_SEED_LOG")
    if hX["n_bt_lines"] != 2:
        why.append("LINK_BTSEED_LOG_LINES")
    if len(hX["fixed_lane"]) != 6 or any(v != h_fixed for v in hX["fixed_lane"]) or \
            len(hE["fixed_lane"]) != 6 or any(v != h_fixed for v in hE["fixed_lane"]):
        why.append("LINK_FIXED_LANE_HASH")
    return why


def det_siiii_btseed_link(x, e, a0, t3, target):
    why = []
    mX, mE = MECH.det_1206mech_meta(x), MECH.det_1206mech_meta(e)
    why += [f"X:{w}" for w in mX["why"]] + [f"E:{w}" for w in mE["why"]]
    for k in ("t_exp", "T_inner", "b_inner", "vdop", "ppd", "lam_hi"):
        if mX[k] != mE[k]:
            why.append(f"LINK_META_X_E_DIFFER:{k}")
    if mX["nrb"] or mE["nrb"]:
        why.append("LINK_NRB_NOT_OFF")
    stage = {}
    sp = Path(x) / "STAGE_IDENTITY.txt"
    if sp.exists():
        stage = dict(ln.split("=", 1) for ln in sp.read_text().splitlines() if "=" in ln)
    else:
        why.append("LINK_STAGE_IDENTITY_MISSING")
    binp = Path(x) / "input/lumina_cuda"
    bsha = sha(binp) if binp.exists() else None
    if not bsha or stage.get("run_binary_sha256") != bsha or \
            (Path(x) / "input/binary.sha256").read_text().strip() != bsha:
        why.append("LINK_BINARY_IDENTITY")
    exp = Path(a0) / "expectations.json"
    if sha(exp) != target["expectations_sha256"] or stage.get("expectations_sha256") != target["expectations_sha256"]:
        why.append("LINK_EXPECTATIONS_IDENTITY")
    px, pe = Path(x) / "input/fixed_te_profile.txt", Path(e) / "input/fixed_te_profile.txt"
    if sha(px) != sha(pe) or stage.get("fixed_te_profile_file_sha256") != sha(px):
        why.append("LINK_PROFILE_IDENTITY")
    psx = Path(x) / "input/bt_seed_profile.txt"
    seed_path = str(Path(x).resolve() / "input/bt_seed_profile.txt")
    if not psx.exists() or stage.get("seed_profile_file_sha256") != sha(psx):
        why.append("LINK_SEED_PROFILE_IDENTITY")
    rb = json.loads((Path(x) / "input/run_build_identity.json").read_text())
    ent = {f["path"]: f["sha256"] for f in rb.get("source_files", [])}
    if rb.get("binary", {}).get("sha256") != bsha:
        why.append("LINK_RUN_BUILD_BINARY")
    for f in SRC_CORE:
        if ent.get(f) != (sha(Path(t3) / f) if (Path(t3) / f).exists() else None):
            why.append(f"LINK_SOURCE_NOT_T3:{f}")
    dX = det_siiii_btseed_exports(Path(x) / "input/resolved_lumina.exports")
    dE = det_siiii_btseed_exports(Path(e) / "input/resolved_lumina.exports")
    xs, es = str(Path(x).resolve()), str(Path(e).resolve())
    added, removed = set(dX) - set(dE), set(dE) - set(dX)
    if added != set(BT_KNOBS) or dX.get("LUMINA_BT_SEED") != "1" or dX.get("LUMINA_BT_SEED_PROFILE") != seed_path:
        why.append(f"LINK_EXPORTS_ADDED:{sorted(added)}")
    if removed:
        why.append(f"LINK_EXPORTS_REMOVED:{sorted(removed)}")
    rewritten = sorted(k for k in dE if k in dX and dX[k] != dE[k])
    for k in rewritten:
        if dX[k] != dE[k].replace(es, xs):
            why.append(f"LINK_EXPORT_CHANGED:{k}")
    hX, hE = det_siiii_btseed_log_hashes(x), det_siiii_btseed_log_hashes(e)
    ns = len(det_siiii_btseed_profile_values(px)) if px.exists() else None
    why += det_siiii_btseed_hash_chain(hX, hE, stage.get("h_seed"), stage.get("h_fixed"), seed_path, ns)
    orp = stage.get("offline_record")
    if not orp or not Path(orp).exists() or sha(orp) != stage.get("offline_record_sha256"):
        why.append("LINK_OFFLINE_RECORD")
    else:
        sr = json.loads(Path(orp).read_text()).get("seed_profile", {})
        if sr.get("H_seed") != stage.get("h_seed") or sr.get("H_fixed") != stage.get("h_fixed") or \
                sr.get("file_sha256") != stage.get("seed_profile_file_sha256"):
            why.append("LINK_SEAL_RECORD")
    fp = Path(x) / "RUN_FOOTER.txt"
    ft = dict(ln.split("=", 1) for ln in fp.read_text().splitlines() if "=" in ln) if fp.exists() else {}
    if ft.get("terminal_state") != "SUCCESS" or ft.get("h_seed") != stage.get("h_seed") or \
            ft.get("h_fixed") != stage.get("h_fixed") or \
            ft.get("seed_profile_file_sha256") != stage.get("seed_profile_file_sha256"):
        why.append("LINK_RUN_FOOTER")
    tX, tE = Path(x) / "work/det_obs_capture/Te.f64", Path(e) / "work/det_obs_capture/Te.f64"
    if not tX.exists() or not tE.exists() or tX.read_bytes() != tE.read_bytes():
        why.append("LINK_FIXED_T_GENERATION_TE_DIFFER")
    return why, mX, mE, stage, {"rewritten_exports": rewritten, "h_seed": stage.get("h_seed"), "h_fixed": stage.get("h_fixed")}


def det_siiii_btseed_analyze(x, e, a0, t3, public, out, target=TARGET, fault=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    why, mX, mE, stage, linkinfo = det_siiii_btseed_link(x, e, a0, t3, target)
    expd = json.loads((Path(a0) / "expectations.json").read_text())
    bt = expd.get("B-T", {})
    reg_seed = bt.get("primary_expectation", {}).get("B_profile_T_pin", {})
    reg_mech = bt.get("mechanism_expectation", {}).get("registered", {})
    reg_2542 = bt.get("2542_boundary_records", {})
    tg = bt.get("secondary_spectral_expectation", {}).get("targets")
    extp = Path(a0) / "extension.csv"
    ext = pd.read_csv(extp, float_precision="round_trip") if extp.exists() else pd.DataFrame(columns=["line", "shell", "Z", "ion", "T_pin"])
    ext_keys = [(int(k.line), int(k.shell)) for k in ext.itertuples()]
    rows = sorted(set([target["r1206"], target["r2542"]] + [k[0] for k in ext_keys]))
    atoms = REP.det_siiii_repair_atom_rows(x, rows)
    if atoms is None or atoms != REP.det_siiii_repair_atom_rows(e, rows):
        why.append("LINK_ATOM_ROWS")
        atoms = atoms or {}
    a1206, a2542 = atoms.get(target["r1206"]), atoms.get(target["r2542"])
    if a1206 is None or (a1206["Z"], a1206["ion"], a1206["lo"], a1206["up"]) != (Z, ION, target["lo"], target["up"]):
        why.append("LINK_TARGET_ROW_1206")
    if a2542 is None or (a2542["Z"], a2542["ion"], a2542["lo"], a2542["up"]) != (Z, ION, target["up"], target["u5"]):
        why.append("LINK_TARGET_ROW_2542")
    keys = set(ext_keys) | {(target["r1206"], s) for s in target["shells"]}
    hX, cX = REP.det_siiii_repair_w5_load(x, keys)
    hE, cE = REP.det_siiii_repair_w5_load(e, keys)
    if not why:
        for s in target["shells"]:
            k = (target["r1206"], s)
            for tag, h, cc, run, m in (("X", hX, {q for q, _ in cX}, x, mX), ("E", hE, {q for q, _ in cE}, e, mE)):
                if k in cc:
                    why.append(f"{tag}:W5_CONTRADICTORY_DUPLICATE:{s}")
                hk = h.get(k, {})
                why += [f"{tag}:LINK_{w}:{s}" for w in REP.det_siiii_repair_key_check(hk, a1206, Z, ION)]
                cap_, _ = MECH.det_1206mech_capture_ratio(run, target["r1206"], s, a1206, m["t_exp"])
                if cap_ is not None and 7 in hk and float(hk[7]["tau_eff"]) != cap_["tau"]:
                    why.append(f"{tag}:LINK_TAU_W5_CALL7_VS_CAPTURE:{s}")
    if why:
        res = {"schema": "DET_SIIII_BT_SEED_RESULT_V1", "validity": "X", "reasons": why, "fault": fault}
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    grid = MECH.det_1206mech_grid(mX["vdop"], mX["ppd"], mX["lam_hi"])
    kappa, dB, _ = MECH.det_1206mech_kappa(a1206)
    B_in, _ = MECH.det_1206mech_profile_mean(grid, a1206["nu"], mE["T_inner"], mE["b_inner"])
    cap = {t: {nm: np.fromfile(Path(r) / f"work/det_obs_capture/{nm}.f64", "<f8") for nm in ("Te", "ne")}
           for t, r in (("X", x), ("E", e))}
    seed_rows, codes = [], {"seed_delivery": {}, "registered_material7_match": {}, "reduced_model_applicability": {}}
    for s in target["shells"]:
        rX, rE = hX.get((target["r1206"], s), {}), hE.get((target["r1206"], s), {})
        row = {"shell": s, "role": "MAIN" if s == target["main"] else "COMPARISON"}
        J1X = MECH.det_1206mech_jiv(rX[1]) if 1 in rX else None
        if J1X is not None and fault == "drop_jbar_err":
            J1X = I.x(float(rX[1]["Jbar"]))
        J1E = MECH.det_1206mech_jiv(rE[1]) if 1 in rE else None
        B = det_siiii_btseed_iv(reg_seed.get(str(s)))
        D = absdev(J1X, B) if (J1X is not None and B is not None) else None
        st = status(D, CRIT)
        codes["seed_delivery"][str(s)] = det_siiii_btseed_code(st)
        cols(row, "J1_X", J1X)
        cols(row, "J1_E", J1E)
        cols(row, "B_registered", B)
        cols(row, "D_seed", D)
        row["seed_status"] = st
        DE = absdev(J1E, B_in) if J1E is not None else None
        cols(row, "D_seed_E_vs_B_T_inner", DE)
        sh = log10i(J1X.div(J1E)) if (J1X is not None and J1E is not None) else None
        cols(row, "log10_J1_X_over_E", sh)
        row["shift_direction"] = REP.det_siiii_repair_direction(sh)
        for c in range(2, 8):
            if c in rX and c in rE:
                row[f"call{c}_beta_X_over_E"] = float(rX[c]["beta"]) / float(rE[c]["beta"])
                row[f"call{c}_Jc_X_over_E"] = float(rX[c]["J_c"]) / float(rE[c]["J_c"])
        J7X = MECH.det_1206mech_jiv(rX[7]) if 7 in rX else None
        cols(row, "J7_X", J7X)
        # registered generation-7 match and the run's own reduced-model applicability
        capX, wX = MECH.det_1206mech_capture_ratio(x, target["r1206"], s, a1206, mX["t_exp"])
        capE, wE = MECH.det_1206mech_capture_ratio(e, target["r1206"], s, a1206, mE["t_exp"])
        r7X = None if capX is None else capX["r"]
        J6X = MECH.det_1206mech_jiv(rX[6]) if 6 in rX else None
        roff6 = MECH.det_1206mech_r_off(J6X, a1206) if J6X is not None else None
        reg = det_siiii_btseed_iv(reg_mech.get(str(s), {}).get("r7_Ebg"))
        if fault == "recompute_registered":
            reg = roff6
        Dr = absdev(r7X, reg) if (r7X is not None and reg is not None) else None
        str_ = status(Dr, CRIT)
        codes["registered_material7_match"][str(s)] = det_siiii_btseed_code(str_)
        cols(row, "r7_X", r7X)
        cols(row, "r7_E", None if capE is None else capE["r"])
        cols(row, "r7_registered", reg)
        cols(row, "D_registered", Dr)
        row["registered_status"] = str_
        dm = absdev(r7X, roff6) if (r7X is not None and roff6 is not None) else None
        Cu, Cd = RATES.det_siiii_rates_coll(float(cap["X"]["ne"][s]), float(cap["X"]["Te"][s]), a1206["E_l"], a1206["E_u"],
                                            a1206["g_l"], a1206["g_u"], a1206["f_lu"])
        rC = REP.det_siiii_repair_rC(J6X, a1206, Cu, Cd) if J6X is not None else None
        dC = absdev(rC, roff6) if (rC is not None and roff6 is not None) else None
        sm = allst(status(dm, CRIT), status(dC, CRIT))
        codes["reduced_model_applicability"][str(s)] = det_siiii_btseed_code(sm)
        cols(row, "D_reduced_model", dm)
        cols(row, "D_collision", dC)
        row["reduced_model_status"] = sm
        row["n_e7_X_over_E"] = float(cap["X"]["ne"][s]) / float(cap["E"]["ne"][s])
        if str_ == "REFUTED" and J6X is not None:
            HX = MECH.det_1206mech_history(hX, target["r1206"], s, "X")
            preds = MECH.det_1206mech_predict(HX, a1206, kappa, dB)
            row["one_step_map_status"] = allst(*[p["status"] for p in preds])
        else:
            row["one_step_map_status"] = "NOT_RUN_REGISTERED_MATCH_NOT_REFUTED"
        seed_rows.append(row)
    for k in ("seed_delivery", "registered_material7_match", "reduced_model_applicability"):
        inv = {v: k_ for k_, v in CODE.items()}
        codes[k]["overall"] = CODE[det_siiii_btseed_universal([inv[c] if c is not None else "NOT_IDENTIFIABLE"
                                                               for c in codes[k].values()])]
    # 2542 A actual response
    resp = []
    for s in sorted(set(target["spatial_2542"] + target["shells"])):
        row = {"shell": s}
        st = {}
        for tag, run, m in (("X", x, mX), ("E", e, mE)):
            c12, _ = MECH.det_1206mech_capture_ratio(run, target["r1206"], s, a1206, m["t_exp"])
            rec = POP.capture_records(Path(run) / "work/det_obs_capture", [target["r2542"]], {s}).get((target["r2542"], s))
            rv = (recover(rec["tau"], rec["S"], a2542["g_l"], a2542["g_u"], a2542["f_lu"], a2542["lam_cm"], m["t_exp"])
                  if (rec is not None and rec["mapped"] == 1) else None)
            if c12 is None or rv is None:
                row[f"{tag}_status"] = "NOT_IDENTIFIABLE"
                continue
            ok = c12["n_u"].lo <= rv["n_l"].hi and rv["n_l"].lo <= c12["n_u"].hi
            row[f"{tag}_status"] = "OK" if ok else "N4_INCONSISTENT"
            for nm, v in (("n0", c12["n_l"]), ("n4", rv["n_l"]), ("n5", rv["n_u"]), ("q45", rv["q"])):
                cols(row, f"{tag}_{nm}", v)
            row[f"{tag}_S45"], row[f"{tag}_tau45"] = rec["S"], rec["tau"]
            row[f"{tag}_Q45"] = rec["S"] * (-math.expm1(-rec["tau"])) if rec["tau"] > 0 else None
            st[tag] = (c12, rv, rec)
        if row.get("X_status") == "OK" and row.get("E_status") == "OK":
            (cx, vx, _), (ce, ve, _) = st["X"], st["E"]
            a0_ = cx["n_l"].div(ce["n_l"])
            a4 = a0_ * cx["r"].div(ce["r"])                           # (n0'/n0) (n4/n0)'/(n4/n0)
            if fault == "omit_n0":
                a4 = cx["r"].div(ce["r"])
            cols(row, "alpha0", a0_)
            cols(row, "alpha4", a4)
            cols(row, "alpha4_direct", vx["n_l"].div(ve["n_l"]))
            cols(row, "alpha5", vx["n_u"].div(ve["n_u"]))
            qr = (row["X_Q45"] / row["E_Q45"]) if (row["X_Q45"] and row["E_Q45"]) else None
            row["Q_ratio_X_over_E"] = qr
            rb = reg_2542.get(str(s), {})
            p1, p2 = rb.get("P1"), rb.get("P2")
            row["P1_registered"], row["P2_registered"] = p1, p2
            if qr is None or p1 is None or p2 is None:
                row["position_vs_P1_P2"] = "NOT_IDENTIFIABLE"
            else:
                lo_, hi_ = min(p1, p2), max(p1, p2)
                row["position_vs_P1_P2"] = "BELOW" if qr < lo_ else "ABOVE" if qr > hi_ else "BETWEEN"
            if fault == "fail_outside" and row["position_vs_P1_P2"] in ("BELOW", "ABOVE"):
                row["code"] = 1
        resp.append(row)
    # public comparison
    specs = {"public": REP.det_siiii_repair_public(public), "EprimeN": REP.det_siiii_repair_spec(Path(e) / "work/lumina_spectrum_det_obs.csv"),
             "X": REP.det_siiii_repair_spec(Path(x) / "work/lumina_spectrum_det_obs.csv")}
    spec_rows, summ = REP.det_siiii_repair_spectrum_table(specs)
    pub = {}
    if summ.get("status") == "PUBLIC_NOT_AVAILABLE" or tg is None or summ.get("X", {}).get("status") != "OK":
        pub = {"overall": "NOT_IDENTIFIABLE"}
    else:
        sx = summ["X"]
        tN, tU, tR = det_siiii_btseed_iv(tg["abs_d_NUV_max"]), det_siiii_btseed_iv(tg["abs_d_UV2500_max"]), det_siiii_btseed_iv(tg["R8_below"])
        iv = lambda v: None if v is None else I(v["lo"], v["mid"], v["hi"])       # noqa: E731
        dN, dU, r8 = iv(sx["d_NUV"]), iv(sx["d_UV2500"]), iv(sx["R8"])
        pub = {"NUV": REP.det_siiii_repair_le(None if dN is None else dN.abs(), tN),
               "UV2500": REP.det_siiii_repair_le(None if dU is None else dU.abs(), tU),
               "R8": REP.det_siiii_repair_lt(r8, tR)}
        for k_, v in sx["d_keep4"].items():
            pub[k_] = REP.det_siiii_repair_le(None if v is None else iv(v).abs(), I.x(float(tg["abs_d_keep4_max"])))
        pub["overall"] = det_siiii_btseed_universal(list(pub.values()))
    codes["public_comparison"] = {k: CODE[v] for k, v in pub.items()}
    # 122-key call-1 summary (records only)
    ext_rows = []
    for k in ext.itertuples():
        line, s = int(k.line), int(k.shell)
        a = atoms.get(line)
        h1, e1 = hX.get((line, s), {}).get(1), hE.get((line, s), {}).get(1)
        J1x = MECH.det_1206mech_jiv(h1) if h1 is not None else None
        J1e = MECH.det_1206mech_jiv(e1) if e1 is not None else None
        Bk = MECH.det_1206mech_profile_mean(grid, a["nu"], float(k.T_pin), 1.0)[0] if a is not None else None
        D = absdev(J1x, Bk) if (J1x is not None and Bk is not None) else None
        sh = log10i(J1x.div(J1e)) if (J1x is not None and J1e is not None) else None
        r = {"line": line, "shell": s, "Z": int(k.Z), "ion": int(k.ion), "T_pin": float(k.T_pin),
             "seed_vs_registered_status": status(D, CRIT)}
        cols(r, "D_seed", D)
        cols(r, "log10_J1_X_over_E", sh)
        ext_rows.append(r)
    extd = pd.DataFrame(ext_rows)
    ext_agg = {}
    if len(extd):
        def agg(g):
            sh = g.log10_J1_X_over_E_mid.dropna()
            return {"n": int(len(g)), "status": {k_: int(v) for k_, v in g.seed_vs_registered_status.value_counts().items()},
                    "shift_median": float(sh.median()) if len(sh) else None}
        ext_agg = {"all": agg(extd), "by_ion": {f"{a_}_{b_}": agg(g) for (a_, b_), g in extd.groupby(["Z", "ion"])},
                   "by_shell": {str(s_): agg(g) for s_, g in extd.groupby("shell")}}
    pd.DataFrame(seed_rows).to_csv(out / "seed_1206.csv", index=False)
    pd.DataFrame(resp).to_csv(out / "response_2542.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(out / "spectrum.csv", index=False)
    extd.to_csv(out / "extension.csv", index=False)
    res = {"schema": "DET_SIIII_BT_SEED_RESULT_V1", "validity": "VALID", "reasons": [], "fault": fault, "codes": codes,
           "spectrum_summary": summ, "public_targets_used": tg, "extension": ext_agg,
           "response_2542_codes": [r.get("code") for r in resp if r.get("code") is not None],
           "link": linkinfo, "stage": stage,
           "claim_ceiling": "one fixed-temperature, fixed-iteration run; no physical correctness, no production promotion, "
                            "no default change, no further run authorised"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ stage (never submits)
def det_siiii_btseed_stage(e, t3, b3, offline, seed_profile, expectations, gpu_profile, x, runner):
    e, t3, b3, offline, x, seed_profile = Path(e), Path(t3), Path(b3), Path(offline), Path(x), Path(seed_profile)
    why = []
    sj = offline / "selftest/selftest.json"
    oj = offline / "selftest/offline.json"
    o_, sr = {}, {}
    if not sj.exists() or not oj.exists():
        why.append("OFFLINE_RECORD_MISSING")
    else:
        s_ = json.loads(sj.read_text())
        o_ = json.loads(oj.read_text())
        sr = o_.get("seed_profile", {})
        if s_.get("failures") != 0 or s_.get("readout_sha256") != sha(Path(__file__).resolve()):
            why.append("OFFLINE_SELFTEST_NOT_THIS_READOUT_OR_FAILED")
        if o_.get("G_INTERVENTION") != "PASS" or o_.get("contract_revision") != 2:
            why.append("OFFLINE_G_INTERVENTION_NOT_PASS_REV2")
        if o_.get("t3_changeset_sha256") != {f: sha(t3 / f) for f in sorted(CHANGED | ADDED)}:
            why.append("OFFLINE_TESTED_SOURCES_NOT_CURRENT_T3")
    # dedicated generation-zero profile: sealed record, rule, bytes, H_seed / H_fixed (prereg 2.2, 4.2)
    hexok = lambda h: isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) is not None     # noqa: E731
    T_in = MECH.det_1206mech_meta(e)["T_inner"]
    rule_text = det_siiii_btseed_seed_text(det_siiii_btseed_seed_rule(e / "input/fixed_te_profile.txt", T_in)).encode()
    sealp = seed_profile.with_name(seed_profile.name + ".seal.json")
    if not seed_profile.is_absolute() or not seed_profile.exists() or seed_profile.read_bytes() != rule_text or \
            sr.get("path") != str(seed_profile) or sr.get("file_sha256") != sha(seed_profile) or not sealp.exists() or \
            json.loads(sealp.read_text()).get("file_sha256") != sha(seed_profile) or sr.get("T_inner") != T_in or \
            sr.get("shells_T_inner") != list(SEED_TINNER_SHELLS):
        why.append("SEED_PROFILE_SEAL")
    if not hexok(sr.get("H_seed")) or not hexok(sr.get("H_fixed")) or sr.get("H_seed") == sr.get("H_fixed"):
        why.append("SEED_FIXED_HASHES")
    if sr.get("fixed_te_profile_file_sha256") != sha(e / "input/fixed_te_profile.txt"):
        why.append("FIXED_PROFILE_NOT_E")
    if (b3 / "build.rc").read_text().strip() != "0" if (b3 / "build.rc").exists() else True:
        why.append("BUILD_RC")
    binp = b3 / "lumina_cuda"
    if not binp.exists() or (b3 / "binary.sha256").read_text().split()[0] != sha(binp):
        why.append("BUILD_BINARY_IDENTITY")
    built = {}
    if (b3 / "source_sha_at_build.txt").exists():
        for ln in (b3 / "source_sha_at_build.txt").read_text().splitlines():
            h, p = ln.split(None, 1)
            built[p.strip()] = h
    if not built or any(sha(t3 / p) != h for p, h in built.items()):
        why.append("BUILD_SOURCE_NOT_CURRENT_T3")
    for f in CHANGED - {"tests/seed_te_publish_selftest.c"}:
        if f.startswith("src/") and f in built and built[f] != sha(t3 / f):
            why.append(f"BUILD_CHANGESET_FILE:{f}")
    if sha(expectations) != EXPECTATIONS_SHA256:
        why.append("EXPECTATIONS_IDENTITY")
    if gpu_profile not in ("h200x2", "h100x2"):
        why.append("GPU_PROFILE_NOT_REGISTERED")
    if x.exists():
        why.append("RUN_ROOT_EXISTS")
    if why:
        return {"staged": False, "why": why}
    (x / "input").mkdir(parents=True)
    (x / "work").mkdir()
    shutil.copy2(binp, x / "input/lumina_cuda")
    os.chmod(x / "input/lumina_cuda", 0o755)
    bsha = sha(x / "input/lumina_cuda")
    (x / "input/binary.sha256").write_text(bsha + "\n")
    shutil.copytree(e / "input/model", x / "input/model")
    shutil.copytree(e / "input/global_atomic", x / "input/global_atomic")
    for f in ("deck.sha256", "topion.sha256", "sigma.sha256", "fixed_te_profile.txt"):
        shutil.copy2(e / "input" / f, x / "input" / f)
    shutil.copyfile(seed_profile, x / "input/bt_seed_profile.txt")
    if sha(x / "input/bt_seed_profile.txt") != sr["file_sha256"]:
        return {"staged": False, "why": ["SEED_PROFILE_COPY"]}
    deck_bad = []
    for ln in (x / "input/deck.sha256").read_text().splitlines():
        h, f = ln.split(None, 1)
        if sha(x / "input/model" / f.strip()) != h:
            deck_bad.append(f.strip())
    es, xs = str(e.resolve()), str(x.resolve())
    etxt = (e / "input/resolved_lumina.exports").read_text()
    xtxt = etxt.replace(es, xs)
    if not xtxt.endswith("\n"):
        xtxt += "\n"
    xtxt += f'declare -x LUMINA_BT_SEED="1"\ndeclare -x LUMINA_BT_SEED_PROFILE="{xs}/input/bt_seed_profile.txt"\n'
    (x / "input/resolved_lumina.exports").write_text(xtxt)
    (x / "input/resolved_lumina.sha256").write_text(sha(x / "input/resolved_lumina.exports") + "\n")
    rb = {"schema": "CMFGEN_TWIN_RUN_BUILD_V1", "binary": {"path": str(binp), "sha256": bsha, "bytes": binp.stat().st_size},
          "build_command": "DET-SIIII-BT-SEED B3: B2 recipe (same nvcc line, compiler, 186-path manifest) on T3",
          "source_root": str(t3), "source_files": [{"path": p, "sha256": h} for p, h in sorted(built.items())]}
    (x / "input/run_build_identity.json").write_text(json.dumps(rb, indent=1, sort_keys=True))
    rtxt = Path(runner).read_text()
    if rtxt.count("readonly GPU_PROFILE=h200x2\n") != 1:
        return {"staged": False, "why": ["RUNNER_PROFILE_LINE"]}
    rtxt = rtxt.replace("readonly GPU_PROFILE=h200x2\n", f"readonly GPU_PROFILE={gpu_profile}\n")
    if gpu_profile == "h100x2":
        rtxt = rtxt.replace("#SBATCH --partition=h200\n", "#SBATCH --partition=h100\n").replace(
            "#SBATCH --gres=gpu:H200:2\n", "#SBATCH --gres=gpu:H100:2\n")
    (x / "job.slurm").write_text(rtxt)
    dX, dE = det_siiii_btseed_exports(x / "input/resolved_lumina.exports"), det_siiii_btseed_exports(e / "input/resolved_lumina.exports")
    rewritten = sorted(k for k in dE if dX.get(k) != dE[k])
    stage = {"schema": "DET_SIIII_BT_SEED_STAGE_V1", "run": xs, "baseline_run": es, "gpu_profile": gpu_profile,
             "outer_iterations": "6", "contract": "docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md",
             "contract_sha256": sha(P / "docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md"),
             "run_binary_sha256": bsha, "run_build_identity_sha256": sha(x / "input/run_build_identity.json"),
             "effective_exports_sha256": sha(x / "input/resolved_lumina.exports"),
             "expectations_sha256": sha(expectations), "fixed_te_profile_file_sha256": sha(x / "input/fixed_te_profile.txt"),
             "h_fixed": sr["H_fixed"], "seed_profile_file_sha256": sha(x / "input/bt_seed_profile.txt"), "h_seed": sr["H_seed"],
             "seed_profile_sealed_path": str(seed_profile), "seed_seal_record": str(sealp), "seed_seal_record_sha256": sha(sealp),
             "seed_T_inner": repr(T_in), "seed_shells_T_inner": ",".join(map(str, sr["shells_T_inner"])),
             "seed_shells_T_pin": ",".join(map(str, sr["shells_T_pin"])),
             "offline_selftest_sha256": sha(sj), "offline_record": str(oj), "offline_record_sha256": sha(oj),
             "runner_sha256": sha(runner),
             "job_slurm_sha256": sha(x / "job.slurm"), "readout_sha256": sha(Path(__file__).resolve()),
             "exports_delta": "+LUMINA_BT_SEED=1, +LUMINA_BT_SEED_PROFILE=X/input/bt_seed_profile.txt; path rewrites "
                              + ",".join(rewritten),
             "deck_identity": "OK" if not deck_bad else "MISMATCH:" + ",".join(deck_bad)}
    (x / "STAGE_IDENTITY.txt").write_text("".join(f"{k}={v}\n" for k, v in stage.items()))
    (x / "OWNER.txt").write_text("project=LUMINA-SN\nrepo=Lumina-sn thenmc-macroatom-fluorescence\n"
                                 "stage=DET-SIIII-BT-SEED judgment run X (partial B-T generation-zero material temperature, "
                                 "one run)\n"
                                 "owner=Claude driver session\nDO_NOT_CANCEL\n")
    (x / "SUBMIT_COMMAND.txt").write_text(f"cd {xs} && sbatch --exclude=syn102 --output={xs}/slurm-%j.out "
                                          f"--error={xs}/slurm-%j.err {xs}/job.slurm {xs}\n")
    return {"staged": not deck_bad, "why": [f"DECK:{d}" for d in deck_bad], "stage": stage}


# ------------------------------------------------------------------ readout direct-test fixtures (G-READOUT)
FXT = {"r1206": 3, "lo": 0, "up": 4, "r2542": 1, "u5": 5, "shells": [10, 15, 19, 24, 30],
       "spatial_2542": [17, 18, 19, 20, 21, 22], "main": 19, "expectations_sha256": None}
FX_HASH = "ab" * 32          # H_fixed of the fixtures
FX_SEED = "cd" * 32          # H_seed of the fixtures (differs from H_fixed, as registered)


def det_siiii_btseed_fx_hist(J1, tau, Jc, atom, Jerr=0.0, drift=None):
    k_, d_, _ = MECH.det_1206mech_kappa(atom)
    h = {1: ("INIT_SHARED_GAUSSIAN", 1, 0, None, None, None, J1, Jerr)}
    beta = RATES.det_siiii_rates_exponx(tau)
    J = J1
    for c in range(2, 8):
        if drift is None:
            J = MECH.det_1206mech_F(I.x(J), I.x(beta), I.x(Jc), atom, k_, d_)[0].mid
        else:
            J = J1 * (1.0 + drift * (c - 1))
        h[c] = ("CMFGEN_NONOVERLAP_SOBOLEV", c, c, Jc, tau, J - beta * Jc, None, 0.0)
    return h


def det_siiii_btseed_fx_run(root, src, w5, pops, q45, bt, spectrum, extra_lines=True, **kw):
    a45 = MECH.FX_LEVELS[4][1] / MECH.FX_LEVELS[5][1]
    all5 = sorted(set(FXT["spatial_2542"] + FXT["shells"]))
    pops5 = {s: ((pops[s][1] if s in pops else pops[19][1]), (pops[s][1] if s in pops else pops[19][1]) * q45 / a45) for s in all5}
    run = REP.det_siiii_repair_make_fixture(root, w5, pops, pops5, False, src, **kw)
    (run / "input/lumina_cuda").write_bytes(b"fixture binary " + str(root).encode())
    bs = sha(run / "input/lumina_cuda")
    (run / "input/binary.sha256").write_text(bs + "\n")
    rb = json.loads((run / "input/run_build_identity.json").read_text())
    rb["binary"] = {"sha256": bs}
    (run / "input/run_build_identity.json").write_text(json.dumps(rb))
    rr = str(Path(run).resolve())
    ex = (run / "input/resolved_lumina.exports").read_text()
    ex += f'declare -x LUMINA_FIXED_TE_PROFILE="{rr}/input/fixed_te_profile.txt"\ndeclare -x LUMINA_MODEL_DIR="{rr}/input/model"\n'
    seedp = f"{rr}/input/bt_seed_profile.txt"
    if bt:
        (run / "input/bt_seed_profile.txt").write_text(
            det_siiii_btseed_seed_text(det_siiii_btseed_seed_rule(run / "input/fixed_te_profile.txt", 10020.0)))
        ex += f'declare -x LUMINA_BT_SEED="1"\ndeclare -x LUMINA_BT_SEED_PROFILE="{seedp}"\n'
    (run / "input/resolved_lumina.exports").write_text(ex)
    with open(run / "stderr.log", "a") as f:
        for _ in range(6):
            f.write(f"[A2-10][TE_PUBLICATION] te_lane=FIXED_T te_profile_sha256={FX_HASH} pinned_shells=50 re_root_required=false\n")
    if bt and extra_lines:
        with open(run / "stdout.log", "a") as f:
            f.write(f"[BT-SEED] generation-zero material temperature source=BT_SEED_PROFILE path={seedp} "
                    f"te_profile_sha256={FX_SEED} shells=50 te_min_K=8000 te_max_K=10200\n")
            f.write(f"[BT-SEED][SEED] bootstrap LTE/Saha seed published source=BT_SEED_PROFILE path={seedp} te_lane=FIXED_T "
                    f"re_root_required=0 pinned_shells=50 te_profile_sha256={FX_SEED} te_manifest_sha256={FX_SEED} generation=1\n")
    if bt:
        (run / "RUN_FOOTER.txt").write_text(f"contract=DET-SIIII-BT-SEED\nh_seed={FX_SEED}\nh_fixed={FX_HASH}\n"
                                            f"seed_profile_file_sha256={sha(run / 'input/bt_seed_profile.txt')}\n"
                                            "terminal_state=SUCCESS\n")
    with open(run / "work/lumina_spectrum_det_obs.csv", "w") as f:
        f.write("wavelength_angstrom,flux\n")
        for lam, fl in spectrum:
            f.write(f"{lam!r},{fl!r}\n")
    return run


def det_siiii_btseed_fx_case(base, name, delta=None, jerr=None, drop_call1=(), drift=None, n0x=None, rx_scale=None,
                             p12=None, xspec=None, ekw=None, xkw=None, target=None, mutate=None, fault=None, x_lines=True):
    """build E and X fixtures, A0 (expectations + extension) and the public spectrum; run analyze."""
    root = Path(base) / name
    at = MECH.det_1206mech_fx_atom(3)
    grid = MECH.det_1206mech_grid(1.0e6, 12.0, 20000.0)
    Tin = 10020.0
    shells = FXT["shells"]
    tau, Jc = 1.4e7, 3.0e-8
    Bin = MECH.det_1206mech_profile_mean(grid, at["nu"], Tin, 1.0)[0].mid
    Bpin = {s: MECH.det_1206mech_profile_mean(grid, at["nu"], 8000.0 + 50.0 * s, 1.0)[0] for s in shells}
    delta, jerr = delta or {}, jerr or {}
    wE = {(3, s): det_siiii_btseed_fx_hist(Bin, tau, Jc, at) for s in shells}
    wX = {}
    for s in shells:
        J1 = Bpin[s].mid * (1.0 + delta.get(s, 0.0))
        h = det_siiii_btseed_fx_hist(J1, tau, Jc, at, Jerr=jerr.get(s, 0.0) * J1, drift=drift)
        if s in drop_call1:
            del h[1]
        wX[(3, s)] = h
    reg_r = {s: MECH.det_1206mech_r_off(I.x(det_siiii_btseed_fx_hist(Bpin[s].mid, tau, Jc, at)[6][5] +
                                            RATES.det_siiii_rates_exponx(tau) * Jc), at) for s in shells}
    J6E = {s: wE[(3, s)][6][5] + RATES.det_siiii_rates_exponx(tau) * Jc for s in shells}
    J6X = {s: wX[(3, s)][6][5] + RATES.det_siiii_rates_exponx(tau) * Jc for s in shells}
    popsE = {s: (1.0e8 * (1 + 0.01 * s), 1.0e8 * (1 + 0.01 * s) * MECH.det_1206mech_r_off(I.x(J6E[s]), at).mid) for s in shells}
    rX = {s: MECH.det_1206mech_r_off(I.x(J6X[s]), at).mid * ((rx_scale or {}).get(s, 1.0)) for s in shells}
    popsX = {s: ((n0x or {}).get(s, popsE[s][0]), (n0x or {}).get(s, popsE[s][0]) * rX[s]) for s in shells}
    lam_grid = [2000.0 + 10.0 * i for i in range(901)]
    base_f = lambda l_: 1e38 * (l_ / 5000.0) ** -1.0                     # noqa: E731
    pub = [(l_, base_f(l_)) for l_ in lam_grid]
    sE = [(l_, base_f(l_) * (1.6 if l_ < 3000 else 1.3 if l_ < 4000 else 1.0)) for l_ in lam_grid]
    sX = xspec if xspec is not None else pub
    src = REP.det_siiii_repair_make_source(root / "T3")
    e = det_siiii_btseed_fx_run(root / "E", src, wE, popsE, 0.02, False, sE, **(ekw or {}))
    x = det_siiii_btseed_fx_run(root / "X", src, wX, popsX, 0.02, True, sX, extra_lines=x_lines, **(xkw or {}))
    a0 = root / "A0"
    a0.mkdir()
    _, summ = REP.det_siiii_repair_spectrum_table({"public": pub, "EprimeN": sE})
    tg = REP.det_siiii_repair_targets(summ)
    expd = {"B-T": {"primary_expectation": {"B_profile_T_pin": {str(s): [Bpin[s].lo, Bpin[s].mid, Bpin[s].hi] for s in shells}},
                    "mechanism_expectation": {"registered": {str(s): {"r7_Ebg": [reg_r[s].lo, reg_r[s].mid, reg_r[s].hi]}
                                                             for s in shells}},
                    "2542_boundary_records": {str(s): {"P1": (p12 or (0.9, 4.0))[0], "P2": (p12 or (0.9, 4.0))[1]}
                                              for s in sorted(set(FXT["spatial_2542"] + shells))},
                    "secondary_spectral_expectation": {"targets": tg}}}
    (a0 / "expectations.json").write_text(json.dumps(expd))
    pd.DataFrame([{"line": 3, "shell": s, "Z": 14, "ion": 2, "T_pin": 8000.0 + 50.0 * s} for s in shells]).to_csv(
        a0 / "extension.csv", index=False)
    with open(root / "public.txt", "w") as f:
        f.write("#NTIMES:  1\n#TIMES[d]: 1.94800e+01\n#wavelength flux\n")
        for l_, f_ in pub:
            f.write(f"{l_!r} {f_!r}\n")
    tgt = dict(target or FXT, expectations_sha256=sha(a0 / "expectations.json"))
    seed_sha = sha(x / "input/bt_seed_profile.txt")
    (root / "offline.json").write_text(json.dumps({"seed_profile": {"H_seed": FX_SEED, "H_fixed": FX_HASH,
                                                                    "file_sha256": seed_sha}}))
    stage = {"run_binary_sha256": sha(x / "input/lumina_cuda"), "expectations_sha256": sha(a0 / "expectations.json"),
             "fixed_te_profile_file_sha256": sha(x / "input/fixed_te_profile.txt"), "h_fixed": FX_HASH,
             "seed_profile_file_sha256": seed_sha, "h_seed": FX_SEED, "offline_record": str(root / "offline.json"),
             "offline_record_sha256": sha(root / "offline.json")}
    (x / "STAGE_IDENTITY.txt").write_text("".join(f"{k}={v}\n" for k, v in stage.items()))
    if mutate:
        mutate(x, e, a0)
    res = det_siiii_btseed_analyze(x, e, a0, src, root / "public.txt", root / "out", target=tgt, fault=fault)
    return res, root / "out", {"x": x, "e": e, "a0": a0, "Bpin": Bpin, "popsX": popsX, "popsE": popsE}


def det_siiii_btseed_readout_tests(base):
    from fractions import Fraction
    res = []

    def case(name, got, want):
        ok = bool(got == want)
        res.append({"case": name, "pass": ok, "got": str(got)[:400], "want": str(want)[:400]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {str(got)[:160]} want {str(want)[:160]}", flush=True)
    base = Path(base)
    sc = lambda r, k="seed_delivery": r.get("codes", {}).get(k, {})              # noqa: E731
    r0, o0, f0 = det_siiii_btseed_fx_case(base, "clean")
    allcodes = [v for k in ("seed_delivery", "registered_material7_match", "reduced_model_applicability", "public_comparison")
                for v in r0.get("codes", {}).get(k, {}).values()]
    case("seed: all five within -> SURVIVES; clean registered match and public comparison; codes only 0/1/4/5",
         (r0["validity"], sc(r0).get("overall"), sc(r0, "registered_material7_match").get("overall"),
          sc(r0, "public_comparison").get("overall"), all(c in (0, 1, 4, 5) for c in allcodes)),
         ("VALID", 0, 0, 0, True))
    r1, *_ = det_siiii_btseed_fx_case(base, "one_out", delta={19: 2e-3})
    r2, *_ = det_siiii_btseed_fx_case(base, "boundary", delta={19: 1e-3 * (1 - 1e-7)}, jerr={19: 1e-6})
    r2f, *_ = det_siiii_btseed_fx_case(base, "boundary_fault", delta={19: 1e-3 * (1 - 1e-7)}, jerr={19: 1e-6},
                                       fault="drop_jbar_err")
    r3, *_ = det_siiii_btseed_fx_case(base, "missing", drop_call1=(24,))
    r4, *_ = det_siiii_btseed_fx_case(base, "out_and_missing", delta={19: 2e-3}, drop_call1=(24,))
    case("seed: one shell outside -> REFUTED; boundary overlap -> UNRESOLVED; dropping Jbar_err_upper survives wrongly; "
         "missing row -> NOT_IDENTIFIABLE; a counterexample survives a missing row",
         (sc(r1)["19"], sc(r1)["overall"], sc(r2)["19"], sc(r2)["overall"], sc(r2f)["19"], sc(r3)["24"], sc(r3)["overall"],
          sc(r4)["overall"]), (1, 1, 4, 4, 0, 5, 5, 1))

    def swap_call(x, e, a0):
        p = x / "work/w5_lines.csv"
        t = p.read_text().split("\n")
        hdr, body = t[0], t[1:]
        out = [hdr]
        for ln in body:
            f = ln.split(",")
            if len(f) > 5 and f[0] == "1" and f[5] == "3" and f[4] == "19":
                f[0] = "2"
            out.append(",".join(f))
        p.write_text("\n".join(out))
    rx1, *_ = det_siiii_btseed_fx_case(base, "x_call_swap", mutate=swap_call)
    rx2, *_ = det_siiii_btseed_fx_case(base, "x_gen8", xkw={"cap_gen": 8})
    rx3, *_ = det_siiii_btseed_fx_case(base, "x_shell_swap", xkw={"swap_shells": (19, 24)})
    rx4, *_ = det_siiii_btseed_fx_case(base, "x_row", target=dict(FXT, r1206=4))
    case("link: call 1 relabelled as call 2, capture generation 8, shell swap and wrong line row are inadmissible (X)",
         (rx1["validity"], rx2["validity"], rx3["validity"], rx4["validity"]), ("X", "X", "X", "X"))

    def other_profile(x, e, a0):
        p = x / "input/fixed_te_profile.txt"
        p.write_text(p.read_text().replace("8950.0", "8951.0"))

    def other_binary(x, e, a0):
        (x / "input/lumina_cuda").write_bytes(b"another binary")

    def other_expect(x, e, a0):
        d = json.loads((a0 / "expectations.json").read_text())
        d["note"] = "edited"
        (a0 / "expectations.json").write_text(json.dumps(d))
    def seed_hash_swap(x, e, a0):
        p = x / "stdout.log"
        p.write_text(p.read_text().replace(FX_SEED, FX_HASH))

    def fixed_lane_missing(x, e, a0):
        p = x / "stderr.log"
        p.write_text("".join(ln for ln in p.read_text().splitlines(True) if "[A2-10][TE_PUBLICATION]" not in ln))

    def seed_content(x, e, a0):
        p = x / "input/bt_seed_profile.txt"
        p.write_text(p.read_text().replace("45 10020\n", "45 10021\n"))

    def seed_path(x, e, a0):
        p = x / "input/resolved_lumina.exports"
        p.write_text(p.read_text().replace("input/bt_seed_profile.txt", "input/fixed_te_profile.txt"))

    def pinned45(x, e, a0):
        p = x / "stdout.log"
        p.write_text(p.read_text().replace("re_root_required=0 pinned_shells=50", "re_root_required=0 pinned_shells=45"))

    def stage_hash_missing(x, e, a0):
        p = x / "STAGE_IDENTITY.txt"
        p.write_text("".join(ln for ln in p.read_text().splitlines(True) if not ln.startswith("h_seed=")))
    hres = {n: det_siiii_btseed_fx_case(base, f"h_{n}", mutate=m)[0] for n, m in (
        ("seed_hash_swap", seed_hash_swap), ("fixed_lane_missing", fixed_lane_missing), ("seed_content", seed_content),
        ("seed_path", seed_path), ("pinned45", pinned45), ("stage_hash_missing", stage_hash_missing))}
    want_reason = {"seed_hash_swap": "LINK_BTSEED_SEED_LOG", "fixed_lane_missing": "LINK_FIXED_LANE_HASH",
                   "seed_content": "LINK_SEED_PROFILE_IDENTITY", "seed_path": "LINK_EXPORTS_ADDED",
                   "pinned45": "LINK_BTSEED_SEED_LOG", "stage_hash_missing": "LINK_STAGE_SEED_FIXED_HASHES"}
    case("two roles: H_seed != H_fixed is the normal case (clean VALID); seed log carrying H_fixed, missing FIXED_T lane "
         "lines, changed sealed seed content, seed knob pointing elsewhere, seed pinned_shells=45 and a missing H_seed "
         "are inadmissible (X) with the role-specific reason",
         (r0["validity"], {n: (r["validity"], any(w.startswith(want_reason[n]) for w in r.get("reasons", [])))
                           for n, r in hres.items()}),
         ("VALID", {n: ("X", True) for n in want_reason}))
    ri1, *_ = det_siiii_btseed_fx_case(base, "id_profile", mutate=other_profile)
    ri2, *_ = det_siiii_btseed_fx_case(base, "id_binary", mutate=other_binary)
    ri3, *_ = det_siiii_btseed_fx_case(base, "id_expect", mutate=other_expect)
    ri4, *_ = det_siiii_btseed_fx_case(base, "id_btlines", x_lines=False)
    case("link: same-name different profile, binary or expectations file and missing B-T log lines are inadmissible (X)",
         (ri1["validity"], any("PROFILE" in w for w in ri1.get("reasons", [])), ri2["validity"],
          any("BINARY" in w for w in ri2.get("reasons", [])), ri3["validity"],
          any("EXPECTATIONS" in w for w in ri3.get("reasons", [])), ri4["validity"]),
         ("X", True, "X", True, "X", True, "X"))
    lam_grid = [2000.0 + 10.0 * i for i in range(901)]
    bad = [(l_, 1e38 * (l_ / 5000.0) ** -1.0 * (4.0 if l_ < 3000 else 1.0)) for l_ in lam_grid]
    rc_, *_ = det_siiii_btseed_fx_case(base, "coexist", xspec=bad)
    case("primary seed delivery survives while the public comparison is refuted",
         (sc(rc_)["overall"], sc(rc_, "public_comparison")["overall"]), (0, 1))
    rd, *_ = det_siiii_btseed_fx_case(base, "drift", drift=2e-3)
    rdf, *_ = det_siiii_btseed_fx_case(base, "drift_fault", drift=2e-3, fault="recompute_registered")
    case("registered generation-7 value differs from the run-background recomputation: registered comparison refutes, "
         "replacing it by the recomputation is detected",
         (sc(rd)["overall"], sc(rd, "registered_material7_match")["19"], sc(rd, "reduced_model_applicability")["19"],
          sc(rdf, "registered_material7_match")["19"]), (0, 1, 0, 0))
    rn, on, fn = det_siiii_btseed_fx_case(base, "n0", n0x={19: 2.0 * 1.0e8 * 1.19})
    rnf, onf, _ = det_siiii_btseed_fx_case(base, "n0_fault", n0x={19: 2.0 * 1.0e8 * 1.19}, fault="omit_n0")
    t4 = Fraction(fn["popsX"][19][1]) / Fraction(fn["popsE"][19][1])
    x19 = pd.read_csv(on / "response_2542.csv", float_precision="round_trip").set_index("shell").loc[19]
    xf19 = pd.read_csv(onf / "response_2542.csv", float_precision="round_trip").set_index("shell").loc[19]
    case("2542: n0 ratio != 1 enters alpha4 (exact n4 ratio enclosed, equal to the direct n4 ratio); omitting n0'/n0 is detected",
         (Fraction(x19.alpha4_lo) <= t4 <= Fraction(x19.alpha4_hi), Fraction(x19.alpha4_direct_lo) <= t4 <= Fraction(x19.alpha4_direct_hi),
          round(float(x19.alpha0_mid), 12), Fraction(xf19.alpha4_lo) <= t4 <= Fraction(xf19.alpha4_hi)), (True, True, 2.0, False))
    rp, op, _ = det_siiii_btseed_fx_case(base, "outside", p12=(0.5, 0.6))
    rpf, *_ = det_siiii_btseed_fx_case(base, "outside_fault", p12=(0.5, 0.6), fault="fail_outside")
    pos = pd.read_csv(op / "response_2542.csv").set_index("shell").loc[19, "position_vs_P1_P2"]
    case("2542: an actual response outside P1/P2 is recorded, not failed; turning it into a failure is detected",
         (pos, rp["response_2542_codes"], len(rpf["response_2542_codes"]) > 0), ("ABOVE", [], True))
    return res


# ------------------------------------------------------------------ CLI
def cmd_selftest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    seed = Path(a.seed_profile)
    if not seed.is_absolute():
        raise SystemExit("SEED_PROFILE_PATH_NOT_ABSOLUTE")
    seal, _ = det_siiii_btseed_seal(seed, a.profile, Path(a.profile).parent.parent)
    if seal["why"]:
        rec = {"schema": "DET_SIIII_BTSEED_OFFLINE_V2", "contract_revision": 2, "seed_profile": seal,
               "G_INTERVENTION": "FAIL", "verdict": "SEED_PROFILE_SEAL_FAILED"}
    else:
        rec = det_siiii_btseed_offline(a.production_source, a.baseline_source, a.profile, seed, seal, out / "offline_work")
    rec["elapsed_s"] = time.time() - t0
    rec["expectations_sha256"] = sha(a.expectations)
    rec["expectations_registered"] = rec["expectations_sha256"] == EXPECTATIONS_SHA256
    (out / "offline.json").write_text(json.dumps(rec, indent=1, default=str))
    t1 = time.time()
    tests = det_siiii_btseed_readout_tests(out / "readout_fx")
    fails = sum(1 for t in tests if not t["pass"])
    gi = rec.get("G_INTERVENTION") == "PASS"
    summary = {"schema": "DET_SIIII_BT_SEED_SELFTEST_V1", "readout_sha256": sha(Path(__file__).resolve()),
               "G_INTERVENTION": rec.get("G_INTERVENTION"), "G_READOUT": "PASS" if fails == 0 else "FAIL",
               "expectations_registered": rec["expectations_registered"], "readout_cases": tests,
               "failures": fails + (0 if gi else 1) + (0 if rec["expectations_registered"] else 1),
               "offline_elapsed_s": rec["elapsed_s"], "readout_tests_elapsed_s": time.time() - t1,
               "dependencies": {str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (POP, RATES, MECH, REP)}}
    (out / "selftest.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("G_INTERVENTION", "G_READOUT", "expectations_registered", "failures")}))
    return 0 if summary["failures"] == 0 else 1


def cmd_stage(a):
    r = det_siiii_btseed_stage(a.baseline_run, a.production_source, a.build, a.offline, a.seed_profile, a.expectations,
                               a.gpu_profile, a.run, GPU_RUNNER)
    print(json.dumps(r, indent=1, default=str))
    return 0 if r.get("staged") else 2


def cmd_analyze(a):
    out = Path(a.out)
    sj = json.loads((Path(a.selftest) / "selftest.json").read_text())
    if sj.get("readout_sha256") != sha(Path(__file__).resolve()) or sj.get("failures") != 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / "result.json").write_text(json.dumps({"validity": "X", "reasons": ["SELFTEST_NOT_THIS_READOUT_OR_FAILED"]}))
        return 2
    res = det_siiii_btseed_analyze(a.run, a.baseline_run, a.prior_analysis, a.production_source, a.public_spectrum, out)
    man = {"schema": "DET_SIIII_BT_SEED_MANIFEST_V1",
           "contract": {"path": str(P / "docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md"),
                        "sha256": sha(P / "docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md")},
           "readout": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
           "imports": {str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (POP, RATES, MECH, REP)},
           "imported_functions": {"det_siiii_repair_readout": ["det_siiii_repair_public", "det_siiii_repair_spec",
                                                               "det_siiii_repair_spectrum_table", "det_siiii_repair_le",
                                                               "det_siiii_repair_lt", "det_siiii_repair_log10",
                                                               "det_siiii_repair_cols", "det_siiii_repair_direction",
                                                               "det_siiii_repair_rC", "det_siiii_repair_atom_rows",
                                                               "det_siiii_repair_w5_load", "det_siiii_repair_key_check"]},
           "inherited_readout_tests": "det_siiii_repair_readout selftest t5 (spectrum OPT-only, partial gain with worse R8, "
                                      "out of range, time mismatch, boundary)",
           "selftest": {"path": str(Path(a.selftest) / "selftest.json"), "sha256": sha(Path(a.selftest) / "selftest.json")},
           "expectations": {"path": str(Path(a.prior_analysis) / "expectations.json"),
                            "sha256": sha(Path(a.prior_analysis) / "expectations.json")},
           "argv": sys.argv, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
           "inputs": {}, "outputs": {}}
    for tag, run in (("X", a.run), ("E", a.baseline_run)):
        for rel in ("input/resolved_lumina.exports", "input/run_build_identity.json", "input/binary.sha256",
                    "input/fixed_te_profile.txt", "work/det_obs_capture/manifest.json", "work/det_obs_capture/lines.bin",
                    "work/det_obs_capture/Te.f64", "work/det_obs_capture/ne.f64", "work/w5_lines.csv",
                    "work/lumina_spectrum_det_obs.csv", "stdout.log", "stderr.log"):
            p = Path(run) / rel
            if p.exists():
                man["inputs"][f"{tag}:{rel}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
    for p in sorted(out.iterdir()):
        man["outputs"][p.name] = sha(p)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"validity": res["validity"], "codes": res.get("codes"), "reasons": res.get("reasons")}, default=str))
    return 0 if res["validity"] == "VALID" else 2


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    for k in ("--production-source", "--baseline-source", "--profile", "--seed-profile", "--expectations", "--out"):
        s.add_argument(k, required=True)
    s = sp.add_parser("stage")
    for k in ("--baseline-run", "--production-source", "--build", "--offline", "--seed-profile", "--expectations",
              "--gpu-profile", "--run"):
        s.add_argument(k, required=True)
    s = sp.add_parser("analyze")
    for k in ("--run", "--baseline-run", "--prior-analysis", "--production-source", "--public-spectrum", "--selftest", "--out"):
        s.add_argument(k, required=True)
    a = p.parse_args()
    sys.exit({"selftest": cmd_selftest, "stage": cmd_stage, "analyze": cmd_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
