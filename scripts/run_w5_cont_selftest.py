#!/usr/bin/env python3
"""DET-W5-CONT-SOURCE CPU selftest runner (prereg docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md §4 G-SELF).

Builds, with the same compiler and options, the CPU fixture harness
tests/w5_cont_source_selftest.c against
  A = the preserved pre-change frozen source (impl/pre + the unchanged rest of src),
  B = the changed source,
  C = the changed source with -DLUMINA_W5_CONT_SOURCE_SELFTEST (hooks + fault injection),
then runs
  * OFF byte comparison: A(unset) vs B(unset), B("0"), C(unset) on grids A and B
    (bin/fine chi arrays, S_fixed, warm-start and solved J, line Jbar and line
    synthesis terms); the OFF runs must leave no file in their cwd and print no
    W5 tag;
  * NC-OFF: C with the OFF-numerator injection must break the byte comparison;
  * the positive selftest (C) and the four injected-defect NCs (C), each of which
    must exit non-zero naming the detecting check;
  * optionally the readout NC battery (w5_cont_readout.py selftest).
Run on grammar-debug (never on a login node). --out must be a new directory.
Exit 0 iff every item passes; summary.json records every rc.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TREE = Path("/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source")
PRE = Path("/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/impl/pre")
CHANGED = ("lumina_cmfgen.c", "lumina_cmfgen.h", "env_universe.h")
CPU_SRCS = ("lumina_transport.c a2_02c_segment_capture.c radiation_field.c bf_rate_jnu.c "
            "line_jbar.c seed_capability.c jnu_seed.c population_contract.c "
            "atomic_internal_energy.c opacity_publication.c emissivity_publication.c "
            "radeq_publication.c line_net_rate.c cmfgen_adiabatic.c cmf_exact_sliding.c "
            "cmf_error_envelope.c nlte_population_candidate.c physics_comparison.c "
            "gpu_physics_contract.c lumina_plasma.c lumina_element_wide.c lumina_atomic.c "
            "lumina_cmfgen.c cmfgen_final_writer.c").split()
CFLAGS = ["-O2", "-std=gnu11", "-D_GNU_SOURCE", "-fopenmp", "-ffunction-sections",
          "-fdata-sections", "-Wall"]
LDFLAGS = ["-Wl,--gc-sections",
           "-Wl,--wrap=cmf_exact_characteristic_solve_with_envelope", "-lm", "-fopenmp"]
W5 = "LUMINA_CMF_FINE_CONT_NLTE_SOURCE"
NC_EXPECT = {"INTERP": ("FINE_INTERP_RULE", "CONT_COMPONENT"),
             "FF_OMIT": ("BIN_COMPONENTS", "CONT_COMPONENT"),
             "FF_DOUBLE": ("BIN_COMPONENTS", "CONT_COMPONENT"),
             "THERMAL": ("CONSUMED_SOURCE",)}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean_env(w5=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LUMINA_")}
    env["OMP_NUM_THREADS"] = "1"
    if w5 is not None:
        env[W5] = w5
    return env


def build(out, name, src_dir, extra):
    exe = out / "bin" / name
    exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = (["gcc"] + CFLAGS + ["-I" + str(src_dir)] + extra +
           ["-o", str(exe), str(TREE / "tests" / "w5_cont_source_selftest.c")] +
           [str(src_dir / s) for s in CPU_SRCS] + LDFLAGS)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / f"build_{name}.log").write_text(r.stdout + r.stderr)
    return {"name": name, "rc": r.returncode, "cmd": " ".join(cmd),
            "seconds": round(time.time() - t0, 1),
            "warnings": sum(1 for ln in (r.stderr or "").splitlines() if "warning:" in ln),
            "exe_sha256": sha256(exe) if r.returncode == 0 else None}


def run(out, tag, exe, args, w5):
    cwd = out / "cwd" / tag
    cwd.mkdir(parents=True)
    t0 = time.time()
    r = subprocess.run([str(exe)] + args, cwd=cwd, env=clean_env(w5),
                       capture_output=True, text=True)
    (out / "logs" / f"{tag}.stdout").write_text(r.stdout)
    (out / "logs" / f"{tag}.stderr").write_text(r.stderr)
    return {"tag": tag, "rc": r.returncode, "seconds": round(time.time() - t0, 1),
            "cwd_files": sorted(p.name for p in cwd.iterdir()),
            "w5_tag_lines": sum(1 for ln in r.stderr.splitlines() if "W5" in ln),
            "stdout_tail": r.stdout.strip().splitlines()[-3:]}


def compare(da, db):
    fa = sorted(p.name for p in da.iterdir())
    fb = sorted(p.name for p in db.iterdir())
    diff = [n for n in fa if n in fb and (da / n).read_bytes() != (db / n).read_bytes()]
    return {"files_a": len(fa), "files_b": len(fb), "same_list": fa == fb,
            "differing": diff, "identical": fa == fb and not diff and len(fa) > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="new, independent output root")
    ap.add_argument("--pre", default=str(PRE))
    ap.add_argument("--readout", default=str(Path(__file__).with_name("w5_cont_readout.py")))
    ap.add_argument("--c1", default="/gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/runC1")
    ap.add_argument("--no-readout", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"--out {out} exists and is not empty (independent output root required)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    S = {"schema": "W5_CONT_SELFTEST_V1", "out": str(out), "started": time.strftime("%FT%T%z"),
         "host": os.uname().nodename, "items": {}}
    ok = True

    # --- source identity: pre copies vs SHA256SUMS; pre tree = src with 3 files swapped
    pre = Path(a.pre)
    sums = {}
    for ln in (pre / "SHA256SUMS").read_text().splitlines():
        h, p = ln.split()
        sums[Path(p).name] = h
    pre_ok = all(sha256(pre / n) == sums.get(n) for n in CHANGED)
    pre_src = out / "src_pre"
    shutil.copytree(TREE / "src", pre_src)
    for n in CHANGED:
        shutil.copy2(pre / n, pre_src / n)
    S["source"] = {
        "pre_sha256sums_ok": pre_ok,
        "pre": {n: sha256(pre_src / n) for n in CHANGED},
        "post": {n: sha256(TREE / "src" / n) for n in CHANGED},
        "test": sha256(TREE / "tests" / "w5_cont_source_selftest.c"),
        "gcc": subprocess.run(["gcc", "--version"], capture_output=True,
                              text=True).stdout.splitlines()[0]}
    ok &= pre_ok

    # --- builds (same compiler/options; C adds the selftest macro only)
    with cf.ThreadPoolExecutor(3) as ex:
        bs = list(ex.map(lambda x: build(out, *x), [
            ("A_pre", pre_src, []), ("B_post", TREE / "src", []),
            ("C_post_selftest", TREE / "src", ["-DLUMINA_W5_CONT_SOURCE_SELFTEST"])]))
    S["builds"] = bs
    if any(b["rc"] for b in bs):
        S["result"] = "FAIL_BUILD"
        (out / "summary.json").write_text(json.dumps(S, indent=1))
        print(json.dumps(bs, indent=1))
        return 1
    exe = {b["name"][0]: out / "bin" / b["name"] for b in bs}

    # --- runs
    jobs = []
    for g in "AB":
        jobs += [(f"dump_A_unset_{g}", exe["A"], ["dump", str(out / "dump" / f"A_unset_{g}"), g], None),
                 (f"dump_B_unset_{g}", exe["B"], ["dump", str(out / "dump" / f"B_unset_{g}"), g], None),
                 (f"dump_B_zero_{g}", exe["B"], ["dump", str(out / "dump" / f"B_zero_{g}"), g], "0"),
                 (f"dump_C_unset_{g}", exe["C"], ["dump", str(out / "dump" / f"C_unset_{g}"), g], None),
                 (f"dump_C_ncoff_{g}", exe["C"], ["dump", str(out / "dump" / f"C_ncoff_{g}"), g, "OFF"], None)]
    jobs.append(("selftest_C", exe["C"], ["selftest", str(out / "st")], None))
    for n in NC_EXPECT:
        jobs.append((f"nc_{n}", exe["C"], ["nc", n, str(out / f"nc_{n}")], None))
    for _, _, args, _ in jobs:
        Path(args[1] if args[0] in ("dump", "selftest") else args[2]).mkdir(parents=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run, out, *j): j[0] for j in jobs}
        R = {futs[f]: f.result() for f in cf.as_completed(futs)}
    S["runs"] = R

    # --- OFF byte comparison
    comp = {}
    for g in "AB":
        ref = out / "dump" / f"A_unset_{g}"
        for other in ("B_unset", "B_zero", "C_unset"):
            c = compare(ref, out / "dump" / f"{other}_{g}")
            key = f"A_unset_{g}__vs__{other}_{g}"
            c["runs_ok"] = (R[f"dump_A_unset_{g}"]["rc"] == 0 and
                            R[f"dump_{other}_{g}"]["rc"] == 0)
            c["off_no_file_no_tag"] = all(R[t]["cwd_files"] == [] and R[t]["w5_tag_lines"] == 0
                                          for t in (f"dump_A_unset_{g}", f"dump_{other}_{g}"))
            c["pass"] = c["identical"] and c["runs_ok"] and c["off_no_file_no_tag"]
            comp[key] = c
            ok &= c["pass"]
        nc = compare(ref, out / "dump" / f"C_ncoff_{g}")
        nc["run_rc"] = R[f"dump_C_ncoff_{g}"]["rc"]
        nc["injection"] = [ln for ln in R[f"dump_C_ncoff_{g}"]["stdout_tail"] if "NC-OFF" in ln]
        nc["pass"] = (not nc["identical"]) and nc["same_list"] and nc["run_rc"] == 0 and bool(nc["differing"])
        comp[f"NC-OFF_{g}"] = nc
        ok &= nc["pass"]
    S["items"]["off_byte"] = comp

    st = R["selftest_C"]
    S["items"]["selftest"] = {"rc": st["rc"], "tail": st["stdout_tail"], "pass": st["rc"] == 0}
    ok &= st["rc"] == 0
    ncs = {}
    for n, want in NC_EXPECT.items():
        r = R[f"nc_{n}"]
        line = next((ln for ln in r["stdout_tail"] if ln.startswith(f"NC-{n}")), "")
        named = [w for w in want if w in line]
        ncs[n] = {"rc": r["rc"], "line": line, "expected_any_of": want, "named": named,
                  "pass": r["rc"] != 0 and bool(named)}
        ok &= ncs[n]["pass"]
    S["items"]["nc"] = ncs

    if not a.no_readout:
        rd = out / "readout_selftest"
        rd.mkdir()
        r = subprocess.run([sys.executable, "-B", a.readout, "selftest", "--c1", a.c1,
                            "--out", str(rd)], capture_output=True, text=True, env=clean_env())
        (out / "logs" / "readout_selftest.stdout").write_text(r.stdout)
        (out / "logs" / "readout_selftest.stderr").write_text(r.stderr)
        S["items"]["readout_selftest"] = {"rc": r.returncode,
                                          "tail": r.stdout.strip().splitlines()[-4:],
                                          "pass": r.returncode == 0}
        ok &= r.returncode == 0

    S["result"] = "PASS" if ok else "FAIL"
    S["finished"] = time.strftime("%FT%T%z")
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(f"W5 selftest runner: {S['result']}  (summary {out/'summary.json'})")
    for k, c in comp.items():
        print(f"  {k}: pass={c['pass']} identical={c['identical']} differing={c['differing'][:6]}")
    print(f"  selftest rc={st['rc']} :: {st['stdout_tail'][-1] if st['stdout_tail'] else ''}")
    for n, c in ncs.items():
        print(f"  NC-{n}: rc={c['rc']} pass={c['pass']} :: {c['line']}")
    if "readout_selftest" in S["items"]:
        print(f"  readout selftest rc={S['items']['readout_selftest']['rc']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
