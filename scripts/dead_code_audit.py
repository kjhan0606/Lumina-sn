#!/usr/bin/env python3
"""Tier 3.3b — symbolic dead-code audit.

Two passes, both fast (text-based, no clang):

  Pass 1 — Unreferenced C functions
    Parse src/*.{c,cu,h}. For every non-static function definition
    (and not CUDA __global__/__device__ entry kernels at top level),
    grep the whole tree for any other mention of the symbol.
    Flag if zero call sites exist outside the defining line.

    Skipped:
      * static functions (intentionally TU-local)
      * names starting with d_  (CUDA device helpers — called via launchers)
      * names matching ^k_      (CUDA kernels — called via <<<>>>)
      * main, ScalarKernel-style typedefs

  Pass 2 — Orphan LUMINA_* env-vars
    Collect names from:
      C reads     :  getenv("LUMINA_*")  in src/*.{c,cu,h}
      script sets :  LUMINA_*= or export LUMINA_*= in
                       scripts/ slurm/ *.sh *.sbatch run_*.sh
    Report:
      * READ_ONLY  : C reads var, no script ever sets it  (always default value;
                     either dead feature or relies on user-set inline)
      * SET_ONLY   : script sets var, no C reads it       (silent no-op knob;
                     #298-class — name renamed or feature removed)

Exit codes:
    0  clean (no findings)
    1  at least one finding
    2  invocation error
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from collections import defaultdict


C_EXT = (".c", ".cu", ".h")
SCRIPT_EXT = (".sh", ".sbatch", ".py")

# Function-definition regex.  Body line opens with `{` or `\n{`.
# Captures: 1=return type chunk, 2=symbol.
FN_DEF = re.compile(
    r"^(?P<ret>(?:static\s+|inline\s+|__device__\s+|__host__\s+|__global__\s+|extern\s+\"C\"\s+)*"
    r"(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\**)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*?\)\s*(?:\{|$)",
    re.MULTILINE,
)

SKIP_SYMBOLS = {
    "main",
    "if", "for", "while", "switch", "return", "sizeof",
    # CUDA call-side noise
    "cudaMalloc", "cudaFree", "cudaMemcpy",
}


def list_files(root: str, exts: tuple[str, ...]) -> list[str]:
    out = []
    for dp, _, names in os.walk(root):
        # avoid output dirs
        if any(seg in dp for seg in ("/.git/", "/build/", "/output", "/lumina_cuda_", "/runs/")):
            continue
        for n in names:
            if n.endswith(exts):
                out.append(os.path.join(dp, n))
    return out


def collect_fn_defs(src_dir: str) -> list[tuple[str, str, int, bool]]:
    """Return [(file, name, line, is_static)]."""
    defs: list[tuple[str, str, int, bool]] = []
    for f in list_files(src_dir, C_EXT):
        with open(f, errors="replace") as fh:
            text = fh.read()
        for m in FN_DEF.finditer(text):
            name = m.group("name")
            ret = m.group("ret")
            if name in SKIP_SYMBOLS:
                continue
            line = text.count("\n", 0, m.start()) + 1
            is_static = "static" in ret
            defs.append((f, name, line, is_static))
    return defs


def count_refs(name: str, tree: list[str], defining_file: str, defining_line: int) -> int:
    """Number of textual occurrences of `name` outside its own def line."""
    pat = re.compile(rf"\b{re.escape(name)}\b")
    hits = 0
    for f in tree:
        with open(f, errors="replace") as fh:
            for ln_no, ln in enumerate(fh, 1):
                if f == defining_file and ln_no == defining_line:
                    continue
                if pat.search(ln):
                    hits += 1
                    if hits >= 2:  # short-circuit: any second reference is enough
                        return hits
    return hits


def pass_unref_fns(src_dir: str, repo_root: str) -> list[tuple[str, str, int]]:
    """Return [(file, name, line)] of non-static fns w/ zero outside refs."""
    defs = collect_fn_defs(src_dir)
    # search corpus = ALL C/CU/H + scripts (calls can come from Python via ctypes, unlikely but safe)
    corpus = list_files(repo_root, C_EXT) + list_files(os.path.join(repo_root, "scripts"), SCRIPT_EXT)
    findings: list[tuple[str, str, int]] = []
    for f, name, line, is_static in defs:
        if is_static:
            continue
        if name.startswith("d_") or name.startswith("k_"):
            continue  # CUDA device/launcher convention
        n = count_refs(name, corpus, f, line)
        if n == 0:
            findings.append((f, name, line))
    return findings


# --- Pass 2: LUMINA_ env-var orphans ---

GETENV_RE = re.compile(r'getenv\s*\(\s*"(LUMINA_[A-Z0-9_]+)"')
# Dynamic snprintf("LUMINA_*%*X*", ...) — turns into a glob-like pattern.
SNPRINTF_RE = re.compile(r'"(LUMINA_[A-Z0-9_]*%[a-zA-Z][A-Za-z0-9_%]*)"')
SET_RE = re.compile(r'(?:^|\s|export\s+)(LUMINA_[A-Z0-9_]+)\s*=')

# Skip these script files when scanning for SETs (they're tooling, not run scripts).
SCRIPT_SKIP_BASENAMES = {
    "dead_code_audit.py",       # self
}


def collect_env_reads(src_dir: str) -> tuple[dict[str, list[str]], list[re.Pattern[str]]]:
    """Return (literal_reads, dynamic_name_patterns).

    `literal_reads` maps "LUMINA_FOO" -> [file:line] from getenv("LUMINA_FOO").
    `dynamic_name_patterns` is a list of compiled regexes that match LUMINA_*
    names built at runtime via `snprintf(buf,..., "LUMINA_*%s_*", ...)`.
    Any SET-only var matching one of these patterns is NOT a dead knob.
    """
    reads: dict[str, list[str]] = defaultdict(list)
    dyn_patterns: list[re.Pattern[str]] = []
    for f in list_files(src_dir, C_EXT):
        with open(f, errors="replace") as fh:
            text = fh.read()
        for m in GETENV_RE.finditer(text):
            ln_no = text.count("\n", 0, m.start()) + 1
            reads[m.group(1)].append(f"{f}:{ln_no}")
        for m in SNPRINTF_RE.finditer(text):
            fmt = m.group(1)
            # Convert printf format -> regex: %d/%s/%x -> [A-Za-z0-9_]* (zero-or-more;
            # the format arg can substitute the empty string, e.g. suffixes[0]="").
            rgx = re.sub(r"%[a-zA-Z]", "[A-Za-z0-9_]*", fmt)
            dyn_patterns.append(re.compile(f"^{rgx}$"))
    return reads, dyn_patterns


def collect_env_sets(repo_root: str) -> dict[str, list[str]]:
    """Scan scripts/ + slurm/ + top-level *.sh for LUMINA_X= assignments."""
    sets: dict[str, list[str]] = defaultdict(list)
    targets = []
    for sub in ("scripts", "slurm"):
        d = os.path.join(repo_root, sub)
        if os.path.isdir(d):
            targets += list_files(d, (".sh", ".sbatch", ".py"))
    # top-level *.sh
    for n in os.listdir(repo_root):
        full = os.path.join(repo_root, n)
        if os.path.isfile(full) and n.endswith((".sh", ".sbatch")):
            targets.append(full)
    for f in targets:
        if os.path.basename(f) in SCRIPT_SKIP_BASENAMES:
            continue
        try:
            with open(f, errors="replace") as fh:
                for ln_no, ln in enumerate(fh, 1):
                    for m in SET_RE.finditer(ln):
                        sets[m.group(1)].append(f"{f}:{ln_no}")
        except OSError:
            continue
    return sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repo root")
    ap.add_argument("--src", default="src", help="C source subdir (relative to repo)")
    ap.add_argument("--quiet-passing", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    src = os.path.join(repo, args.src)
    if not os.path.isdir(src):
        print(f"[FAIL] no src dir: {src}", file=sys.stderr)
        return 2

    n_findings = 0

    # --- Pass 1 ---
    print("=== Pass 1: unreferenced C functions (non-static, non-d_/k_) ===")
    unref = pass_unref_fns(src, repo)
    if unref:
        for f, name, line in unref:
            rel = os.path.relpath(f, repo)
            print(f"  [DEAD] {rel}:{line}  {name}")
        print(f"  total: {len(unref)} unreferenced function(s)")
        n_findings += len(unref)
    elif not args.quiet_passing:
        print("  clean")

    # --- Pass 2 ---
    print("\n=== Pass 2: orphan LUMINA_* env-vars ===")
    reads, dyn_patterns = collect_env_reads(src)
    sets = collect_env_sets(repo)

    def is_dyn_read(name: str) -> bool:
        return any(p.match(name) for p in dyn_patterns)

    read_only = sorted(v for v in set(reads) - set(sets) if not is_dyn_read(v))
    set_only = sorted(v for v in set(sets) - set(reads) if not is_dyn_read(v))

    # READ_ONLY is INFORMATIONAL (vars can legitimately be set inline at sbatch
    # invocation; we can't distinguish "always-default" from "user-set-at-run").
    # Listed for awareness, NOT counted toward exit code.
    if read_only:
        print(f"  READ_ONLY ({len(read_only)}): C getenv() but no script sets — default-only knobs")
        print(f"    (informational — may be set inline at run time, not necessarily dead)")
        for v in read_only:
            sites = reads[v]
            print(f"    {v}  ({len(sites)} read site(s), first: {sites[0]})")
    elif not args.quiet_passing:
        print("  READ_ONLY: clean")

    if set_only:
        print(f"\n  SET_ONLY ({len(set_only)}): script sets but no C reads — silent no-op knob")
        for v in set_only:
            sites = sets[v]
            print(f"    {v}  ({len(sites)} set site(s), first: {sites[0]})")
        n_findings += len(set_only)
    elif not args.quiet_passing:
        print("  SET_ONLY: clean")

    print()
    if n_findings:
        print(f"=== {n_findings} finding(s) ===")
        return 1
    print("=== clean ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
