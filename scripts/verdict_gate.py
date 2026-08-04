#!/usr/bin/env python3
"""V1 machine gate of the VERDICT PROTOCOL (docs/VERDICT_PROTOCOL.md).

    python3 scripts/verdict_gate.py <run_tag> [--baseline <run_tag>] \
            [--intended-diff SPEC] [--strict-env]

fail-closed: any FAIL -> exit 1 (=> the run may NOT be written into the ledger).

Checks
  1. completion      : 'END RUN FOOTER' present in stdout.log
  2. freshness       : verdict-critical outputs newer than .run_start
  3. config diff     : full RESOLVED CONFIG diff vs baseline; must match
                       --intended-diff exactly (excess or missing => FAIL)
  4. binary / model  : LUMINA_BIN and the argv model dir must be identical to
                       the baseline unless declared in --intended-diff

--intended-diff SPEC grammar (comma separated)
    KEY=VAL   value changes to VAL
    +KEY=VAL  newly set to VAL      (+KEY = newly set, value agnostic)
    -KEY      removed
    KEY       changed somehow (value agnostic)
    LUMINA_BIN / MODEL_DIR[=path] also authorise check 4 differences.
  (values containing a comma cannot be expressed)

Soft keys (OMP_NUM_THREADS, host/scheduler-ish vars) are reported but never
cause a FAIL: parity36a demonstrated 0-row reproduction across OMP 32->16 and
a host change. --strict-env promotes them to hard.
"""

import argparse
import fnmatch
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG_BEGIN = "=== RESOLVED CONFIG"
CFG_END = "=== END CONFIG"
FOOTER_MARK = "END RUN FOOTER"

KEY_RE = re.compile(r"^  (LUMINA_[A-Za-z0-9_]*|OMP_NUM[A-Za-z0-9_]*|SUPER_[A-Za-z0-9_]*)=(.*)$")
ARGV_RE = re.compile(r"^  argv:\s*(.*)$")

CORE_OUTPUTS = [
    "lumina_ion_pops.csv",
    "lumina_plasma_state.csv",
    "lumina_spectrum_formal.csv",
]
OPTIONAL_OUTPUTS = [
    "lumina_c1_bins.csv",
    "lumina_levelpop_resolve_raw.csv",
    "cmf_fine_linedump_*.csv",
    "lumina_jbar_dump.csv",
]

SOFT_KEYS = {"OMP_NUM_THREADS", "OMP_NUM_PROCS", "HOSTNAME", "HOST",
             "CUDA_VISIBLE_DEVICES", "LUMINA_HOST", "LUMINA_GPU",
             "LUMINA_GPU_ID", "LUMINA_DEVICE"}


def is_soft(key):
    return key in SOFT_KEYS or key.startswith("OMP_NUM") or key.startswith("SLURM_")


def run_dir(tag):
    if os.sep in tag or os.path.isdir(tag):
        return os.path.abspath(tag)
    return os.path.join(REPO, "logs", "coevolve_consume_" + tag)


def parse_config(stdout_path):
    """Return (cfg dict, list of duplicate keys). cfg includes pseudo-key 'argv'."""
    cfg, dups = {}, []
    inside = False
    with open(stdout_path, "r", errors="replace") as fh:
        for line in fh:
            if not inside:
                if line.startswith(CFG_BEGIN):
                    inside = True
                continue
            if line.startswith(CFG_END):
                break
            line = line.rstrip("\n")
            m = KEY_RE.match(line)
            if m:
                k, v = m.group(1), m.group(2)
            else:
                m = ARGV_RE.match(line)
                if not m:
                    continue
                k, v = "argv", m.group(1).strip()
            if k in cfg and cfg[k] != v:
                dups.append(k)
            cfg[k] = v
    return cfg, dups


def has_footer(stdout_path):
    with open(stdout_path, "r", errors="replace") as fh:
        for line in fh:
            if FOOTER_MARK in line:
                return True
    return False


def model_dir_of(cfg):
    argv = cfg.get("argv", "")
    parts = argv.split()
    return parts[1] if len(parts) > 1 else None


def binary_of(cfg):
    if "LUMINA_BIN" in cfg:
        return cfg["LUMINA_BIN"]
    parts = cfg.get("argv", "").split()
    return os.path.basename(parts[0]) if parts else None


def parse_spec(spec):
    """-> dict key -> (kind, value|None); kind in {'add','del','set','any'}."""
    out = {}
    if not spec:
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("+"):
            k, sep, v = tok[1:].partition("=")
            out[k.strip()] = ("add", v if sep else None)
        elif tok.startswith("-"):
            out[tok[1:].strip()] = ("del", None)
        else:
            k, sep, v = tok.partition("=")
            out[k.strip()] = ("set", v) if sep else ("any", None)
    return out


def diff_configs(base, run):
    added, removed, changed = {}, {}, {}
    for k, v in run.items():
        if k not in base:
            added[k] = v
        elif base[k] != v:
            changed[k] = (base[k], v)
    for k, v in base.items():
        if k not in run:
            removed[k] = v
    return added, removed, changed


def fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def main():
    ap = argparse.ArgumentParser(add_help=True, description="VERDICT PROTOCOL V1 machine gate")
    ap.add_argument("run_tag")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--intended-diff", default=None)
    ap.add_argument("--strict-env", action="store_true",
                    help="promote OMP/host-like diffs from soft to hard")
    ap.add_argument("--no-write", action="store_true",
                    help="print the report but do not write VERDICT_PREFLIGHT.md")
    # a SPEC may legitimately start with '-' (removal); argparse would take it
    # for an option, so glue it to the flag with '=' first.
    argv = []
    skip = False
    for i, tok in enumerate(sys.argv[1:]):
        if skip:
            skip = False
            continue
        if tok == "--intended-diff" and i + 2 <= len(sys.argv[1:]):
            argv.append("--intended-diff=" + sys.argv[i + 2])
            skip = True
        else:
            argv.append(tok)
    args = ap.parse_args(argv)

    rdir = run_dir(args.run_tag)
    bdir = run_dir(args.baseline) if args.baseline else None
    out, fails, warns = [], [], []
    checks = []  # (name, status, note)

    def add(line=""):
        out.append(line)

    cmd = "python3 scripts/verdict_gate.py " + args.run_tag
    if args.baseline:
        cmd += " --baseline " + args.baseline
    if args.intended_diff:
        cmd += ' --intended-diff "%s"' % args.intended_diff
    if args.strict_env:
        cmd += " --strict-env"

    if not os.path.isdir(rdir):
        sys.stderr.write("FATAL: run dir not found: %s\n" % rdir)
        return 1
    if bdir and not os.path.isdir(bdir):
        sys.stderr.write("FATAL: baseline dir not found: %s\n" % bdir)
        return 1

    run_stdout = os.path.join(rdir, "stdout.log")
    base_stdout = os.path.join(bdir, "stdout.log") if bdir else None

    # ---------------- check 1: completion ----------------
    sec1 = []
    if not os.path.isfile(run_stdout):
        fails.append("stdout.log missing in run dir")
        checks.append(("1 completion", "FAIL", "stdout.log missing"))
        sec1.append("- `stdout.log` **missing** -> FAIL")
    elif has_footer(run_stdout):
        checks.append(("1 completion", "PASS", "END RUN FOOTER present"))
        sec1.append("- `END RUN FOOTER` present in `stdout.log` -> PASS")
    else:
        fails.append("no END RUN FOOTER in stdout.log (run incomplete/aborted)")
        checks.append(("1 completion", "FAIL", "no END RUN FOOTER (incomplete)"))
        sec1.append("- `END RUN FOOTER` **absent** -> FAIL (run still in flight or aborted)")
    if os.path.isfile(run_stdout):
        st = os.stat(run_stdout)
        sec1.append("- stdout.log last write: %s (%d bytes)" % (fmt_ts(st.st_mtime), st.st_size))

    # ---------------- check 2: freshness ----------------
    sec2 = []
    rs = os.path.join(rdir, ".run_start")
    if not os.path.isfile(rs):
        fails.append(".run_start missing -> freshness unverifiable (fail-closed)")
        checks.append(("2 freshness", "FAIL", ".run_start missing"))
        sec2.append("- `.run_start` **missing** -> FAIL (freshness unverifiable)")
    else:
        t0 = os.stat(rs).st_mtime
        sec2.append("`.run_start` = %s" % fmt_ts(t0))
        sec2.append("")
        sec2.append("| file | mtime | dt vs .run_start | status |")
        sec2.append("|---|---|---|---|")
        stale, missing_core, missing_opt = [], [], []
        for pat in CORE_OUTPUTS + OPTIONAL_OUTPUTS:
            core = pat in CORE_OUTPUTS
            if "*" in pat:
                hits = sorted(fnmatch.filter(os.listdir(rdir), pat))
            else:
                hits = [pat] if os.path.isfile(os.path.join(rdir, pat)) else []
            if not hits:
                (missing_core if core else missing_opt).append(pat)
                sec2.append("| `%s` | - | - | %s |" % (pat, "**FAIL (core missing)**" if core else "WARN (absent)"))
                continue
            for h in hits:
                mt = os.stat(os.path.join(rdir, h)).st_mtime
                dt = mt - t0
                ok = dt > 0
                if not ok:
                    stale.append(h)
                sec2.append("| `%s` | %s | %+.0f s | %s |"
                            % (h, fmt_ts(mt), dt, "fresh" if ok else "**FAIL (stale/fossil)**"))
        if missing_core:
            fails.append("core output(s) missing: " + ", ".join(missing_core))
        if stale:
            fails.append("stale output(s) older than .run_start: " + ", ".join(stale))
        if missing_opt:
            warns.append("optional output(s) absent (not produced by this run): " + ", ".join(missing_opt))
        st_fail = bool(missing_core or stale)
        checks.append(("2 freshness", "FAIL" if st_fail else "PASS",
                       ("%d stale, %d core missing" % (len(stale), len(missing_core))) if st_fail
                       else "all present outputs newer than .run_start"))

    # ---------------- parse configs ----------------
    run_cfg, run_dups = ({}, [])
    base_cfg, base_dups = ({}, [])
    if os.path.isfile(run_stdout):
        run_cfg, run_dups = parse_config(run_stdout)
    if base_stdout and os.path.isfile(base_stdout):
        base_cfg, base_dups = parse_config(base_stdout)
    if run_dups:
        warns.append("duplicate keys with conflicting values in run config block: " + ", ".join(run_dups))
    if base_dups:
        warns.append("duplicate keys with conflicting values in baseline config block: " + ", ".join(base_dups))
    if os.path.isfile(run_stdout) and not run_cfg:
        fails.append("no RESOLVED CONFIG block parsed from run stdout.log")

    # ---------------- check 3: config diff ----------------
    sec3 = []
    spec = parse_spec(args.intended_diff)
    unintended, missing_intended, soft_notes = [], [], []
    if not bdir:
        checks.append(("3 config diff", "SKIP", "no --baseline given"))
        sec3.append("_no baseline given -> diff not evaluated (single-variable claims are forbidden without it)_")
        sec3.append("")
        sec3.append("run config: %d keys parsed" % len(run_cfg))
    elif not base_cfg:
        fails.append("no RESOLVED CONFIG block parsed from baseline stdout.log")
        checks.append(("3 config diff", "FAIL", "baseline config unparseable"))
        sec3.append("- baseline `stdout.log` has no parseable RESOLVED CONFIG block -> FAIL")
    else:
        added, removed, changed = diff_configs(base_cfg, run_cfg)
        n = len(added) + len(removed) + len(changed)
        sec3.append("baseline `%s` (%d keys) -> run `%s` (%d keys); **%d diff rows**"
                    % (args.baseline, len(base_cfg), args.run_tag, len(run_cfg), n))
        sec3.append("")
        sec3.append("| kind | key | baseline | run | class |")
        sec3.append("|---|---|---|---|---|")

        def klass(k):
            return "soft" if (is_soft(k) and not args.strict_env) else "hard"

        for k in sorted(added):
            sec3.append("| ADDED | `%s` | (unset) | `%s` | %s |" % (k, added[k], klass(k)))
        for k in sorted(removed):
            sec3.append("| REMOVED | `%s` | `%s` | (unset) | %s |" % (k, removed[k], klass(k)))
        for k in sorted(changed):
            sec3.append("| CHANGED | `%s` | `%s` | `%s` | %s |" % (k, changed[k][0], changed[k][1], klass(k)))
        if n == 0:
            sec3.append("| - | _(none)_ | | | |")

        matched = set()
        if args.intended_diff is not None:
            for k, v in sorted(added.items()):
                s = spec.get(k)
                if s and (s[0] == "any" or (s[0] == "add" and (s[1] is None or s[1] == v))):
                    matched.add(k)
                elif s and s[0] == "add":
                    unintended.append("ADDED %s=%s (intended value was %s)" % (k, v, s[1]))
                    matched.add(k)
                elif s:
                    unintended.append("ADDED %s=%s (spec expected %s)" % (k, v, s[0]))
                    matched.add(k)
                elif klass(k) == "soft":
                    soft_notes.append("ADDED %s=%s" % (k, v))
                else:
                    unintended.append("ADDED %s=%s" % (k, v))
            for k, v in sorted(removed.items()):
                s = spec.get(k)
                if s and s[0] in ("del", "any"):
                    matched.add(k)
                elif s:
                    unintended.append("REMOVED %s (was %s; spec expected %s)" % (k, v, s[0]))
                    matched.add(k)
                elif klass(k) == "soft":
                    soft_notes.append("REMOVED %s (was %s)" % (k, v))
                else:
                    unintended.append("REMOVED %s (was %s)" % (k, v))
            for k, (o, v) in sorted(changed.items()):
                s = spec.get(k)
                if s and (s[0] == "any" or (s[0] == "set" and s[1] == v)):
                    matched.add(k)
                elif s and s[0] == "set":
                    unintended.append("CHANGED %s: %s -> %s (intended -> %s)" % (k, o, v, s[1]))
                    matched.add(k)
                elif s:
                    unintended.append("CHANGED %s: %s -> %s (spec expected %s)" % (k, o, v, s[0]))
                    matched.add(k)
                elif klass(k) == "soft":
                    soft_notes.append("CHANGED %s: %s -> %s" % (k, o, v))
                else:
                    unintended.append("CHANGED %s: %s -> %s" % (k, o, v))
            for k in spec:
                if k in ("LUMINA_BIN", "MODEL_DIR", "argv"):
                    continue
                if k not in matched:
                    missing_intended.append("%s (%s) declared but not observed in the diff" % (k, spec[k][0]))

            sec3.append("")
            sec3.append("**intended-diff**: `%s`" % args.intended_diff)
            sec3.append("")
            sec3.append("- unintended (hard) diffs: **%d**" % len(unintended))
            for u in unintended:
                sec3.append("  - %s" % u)
            sec3.append("- declared-but-absent: **%d**" % len(missing_intended))
            for m in missing_intended:
                sec3.append("  - %s" % m)
            sec3.append("- soft/informational diffs: **%d** (%s)"
                        % (len(soft_notes),
                           "--strict-env: OMP/host-like diffs counted as hard above"
                           if args.strict_env else "not a FAIL reason"))
            for s in soft_notes:
                sec3.append("  - %s" % s)
            if unintended:
                fails.append("%d unintended config diff(s) vs intended-diff" % len(unintended))
            if missing_intended:
                fails.append("%d intended diff(s) not present in the actual config" % len(missing_intended))
            bad = bool(unintended or missing_intended)
            checks.append(("3 config diff", "FAIL" if bad else "PASS",
                           "%d diff rows; %d unintended, %d missing"
                           % (n, len(unintended), len(missing_intended))))
        else:
            sec3.append("")
            sec3.append("_no --intended-diff given -> the diff above is reported but not adjudicated;"
                        " single-variable attribution is forbidden on this basis (V1.2)._")
            checks.append(("3 config diff", "WARN" if n else "PASS",
                           "%d diff rows, no --intended-diff given (not evaluated)" % n))
            if n:
                warns.append("%d config diff rows vs baseline but no --intended-diff to check them against" % n)

    # ---------------- check 4: binary / model ----------------
    sec4 = []
    rbin, rmod = binary_of(run_cfg), model_dir_of(run_cfg)
    sec4.append("- run      : LUMINA_BIN=`%s`  model=`%s`" % (rbin, rmod))
    if bdir and base_cfg:
        bbin, bmod = binary_of(base_cfg), model_dir_of(base_cfg)
        sec4.append("- baseline : LUMINA_BIN=`%s`  model=`%s`" % (bbin, bmod))
        c4 = []
        if rbin != bbin:
            if "LUMINA_BIN" in spec:
                sec4.append("- binary differs but is declared in --intended-diff -> allowed")
            else:
                c4.append("binary differs (%s vs %s) and is not declared in --intended-diff" % (bbin, rbin))
        if rmod != bmod:
            if "MODEL_DIR" in spec or "argv" in spec:
                sec4.append("- model dir differs but is declared in --intended-diff -> allowed")
            else:
                c4.append("model dir differs (%s vs %s) and is not declared in --intended-diff" % (bmod, rmod))
        for c in c4:
            fails.append(c)
            sec4.append("- **FAIL**: %s" % c)
        if not c4:
            sec4.append("- binary and model dir identical (or declared) -> PASS")
        note4 = []
        if rbin != bbin:
            note4.append("binary differs (declared)")
        if rmod != bmod:
            note4.append("model dir differs (declared)")
        checks.append(("4 binary/model", "FAIL" if c4 else "PASS",
                       "; ".join(c4) if c4 else
                       ("; ".join(note4) if note4 else "identical to baseline")))
    else:
        checks.append(("4 binary/model", "SKIP", "no baseline"))

    # ---------------- report ----------------
    verdict = "FAIL" if fails else "PASS"
    add("# VERDICT PREFLIGHT (V1 machine gate) — %s" % args.run_tag)
    add("")
    add("- generated : %s" % fmt_ts(time.time()))
    add("- command   : `%s`" % cmd)
    add("- run dir   : `%s`" % rdir)
    add("- baseline  : `%s`" % (bdir if bdir else "(none)"))
    add("- protocol  : docs/VERDICT_PROTOCOL.md V1")
    add("")
    add("## VERDICT: **%s**" % verdict)
    add("")
    add("| check | status | note |")
    add("|---|---|---|")
    for name, st, note in checks:
        add("| %s | **%s** | %s |" % (name, st, note))
    add("")
    if fails:
        add("**FAIL reasons (%d)**" % len(fails))
        for f in fails:
            add("- %s" % f)
        add("")
    if warns:
        add("**WARN (%d)**" % len(warns))
        for w in warns:
            add("- %s" % w)
        add("")
    add("## 1. Completion")
    add("")
    out.extend(sec1)
    add("")
    add("## 2. Freshness of verdict-critical outputs")
    add("")
    out.extend(sec2)
    add("")
    add("## 3. RESOLVED CONFIG diff (binary-reported environ)")
    add("")
    out.extend(sec3)
    add("")
    add("## 4. Binary / model directory")
    add("")
    out.extend(sec4)
    add("")
    if verdict == "FAIL":
        add("> fail-closed: this run must NOT be recorded in the ledger until every FAIL is resolved.")
    else:
        add("> V1 passed. Attach this block to the V2 verdict draft.")
    add("")

    text = "\n".join(out)
    sys.stdout.write(text + "\n")
    if not args.no_write:
        dest = os.path.join(rdir, "VERDICT_PREFLIGHT.md")
        with open(dest, "w") as fh:
            fh.write(text + "\n")
        sys.stderr.write("[verdict_gate] wrote %s\n" % dest)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
