#!/usr/bin/env python3
"""LUMINA stdout banner smoking-gun scanner.

Scans a run's stdout.log for the printed startup banner + iteration trailer
and flags patterns that historically signaled misconfig or silent no-op:

  WARN  "NOT DESCENDING"        line_list_nu sort verdict (binary-search assumption mismatch)
  WARN  "WARNING:"              any C-level warning printf
  WARN  "reinitializing"        shape-mismatch fallback (catches #298-class file rebuild gaps)
  WARN  "0/<n> bins"            macro-atom or NLTE activation reports zero coverage
  WARN  T_inner err > 5%        convergence not pinned
  INFO  Line interaction mode   (helps spot mode-mismatch retro)
  INFO  NLTE map fraction       (Lines mapped X/Y < 80% flags low coverage)
  INFO  per-iter GPU kernel ms  (sudden drop in late iters can be silent no-op)

Usage:
    scan_lumina_banner.py <run_dir_or_stdout.log>

Exit codes:
    0 = clean
    1 = at least one WARN
    2 = file unreadable
"""
from __future__ import annotations
import argparse
import os
import re
import sys


WARN_PATTERNS = [
    (r"NOT DESCENDING",
     "line_list_nu marked NOT DESCENDING — binary-search assumption mismatch"),
    (r"\bWARNING:",
     "explicit WARNING printed"),
    (r"reinitializing",
     "shape-mismatch fallback — file regeneration likely incomplete"),
    (r"\bERROR\b",
     "ERROR keyword in log"),
    (r"\bNaN\b|\binf\b",
     "NaN/Inf surface"),
]


def scan(path: str) -> int:
    """Return number of WARN-level findings."""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[FAIL] cannot read {path}: {e}", file=sys.stderr)
        return -1

    print(f"=== scan_lumina_banner: {path} ({len(lines)} lines) ===")

    warns = 0

    # --- pattern scan ---
    for ln_no, ln in enumerate(lines, 1):
        for pat, desc in WARN_PATTERNS:
            if re.search(pat, ln):
                warns += 1
                print(f"  [WARN] L{ln_no}: {desc}")
                print(f"         '{ln.rstrip()}'")

    # --- metadata extraction (informational) ---
    print("\n  --- run metadata ---")
    meta_pats = [
        (r"Line interaction:\s*(\w+)", "line interaction"),
        (r"Spectrum mode:\s*([^\n]+)", "spectrum mode"),
        (r"NLTE:\s*(\w+)", "NLTE"),
        (r"Self-consistent T_e:\s*([^\n]+)", "SCE T_e"),
        (r"BF\+FF opacity:\s*(\w+)", "BF+FF opacity"),
        (r"Transition probs:\s*(\w+)", "transition probs"),
        (r"Fe scatter:\s*([^\n]+)", "Fe scatter"),
        (r"n_packets=(\d+).*n_iter=(\d+)", "n_packets/n_iter"),
        (r"T_inner=([0-9.]+) K.*L=([0-9eE.+-]+) erg/s", "T_inner/L"),
        (r"\[NLTE\] Total NLTE levels:\s*(\d+)", "NLTE total levels"),
    ]
    for pat, label in meta_pats:
        for ln in lines:
            m = re.search(pat, ln)
            if m:
                groups = "  ".join(m.groups())
                print(f"  [info ] {label:20s} = {groups}")
                break

    # --- NLTE coverage ---
    for ln in lines:
        m = re.search(r"Lines mapped to NLTE ions:\s*(\d+)\s*/\s*(\d+)", ln)
        if m:
            mapped, total = int(m.group(1)), int(m.group(2))
            frac = mapped / total if total else 0.0
            tag = "[ok   ]" if frac >= 0.80 else "[WARN]"
            print(f"  {tag} NLTE line coverage   = {mapped}/{total} ({100*frac:.1f}%)")
            if frac < 0.80:
                warns += 1

    # --- macro-atom activation (silent no-op fingerprint) ---
    # 0/N at init is legit (NLTE pops not yet computed); flag ONLY if 0/N is the
    # max over all reports (no recovery in any iteration).
    ma_reports = []
    for ln in lines:
        m = re.search(r"Macro-atom activation:\s*(\d+)\s*/\s*(\d+)\s*bins", ln)
        if m:
            ma_reports.append((int(m.group(1)), int(m.group(2))))
    if ma_reports:
        max_on = max(on for on, _ in ma_reports)
        total = ma_reports[-1][1]
        if max_on == 0 and total > 0:
            warns += 1
            print(f"  [WARN] BF macro-atom activation = 0/{total} bins across ALL iterations — silent no-op fingerprint")
        else:
            print(f"  [info ] BF macro-atom activation peak = {max_on}/{total} "
                  f"({100*max_on/max(total,1):.1f}%) across {len(ma_reports)} reports")

    # --- convergence (T_inner err) ---
    for ln in lines:
        m = re.search(r"T_inner final:.*err:\s*([0-9.]+)%", ln)
        if m:
            err = float(m.group(1))
            tag = "[ok   ]" if err < 5 else "[WARN]"
            print(f"  {tag} T_inner err          = {err:.2f}%")
            if err >= 5:
                warns += 1

    # --- per-iter timing collapse (helps spot mode silent-noop) ---
    iter_ms: list[tuple[int, float]] = []
    cur_iter = None
    for ln in lines:
        m = re.search(r"Iteration\s+(\d+)/\d+", ln)
        if m:
            cur_iter = int(m.group(1))
        m = re.search(r"GPU kernel:\s*([0-9.]+)\s*ms", ln)
        if m and cur_iter is not None:
            iter_ms.append((cur_iter, float(m.group(1))))
            cur_iter = None
    if iter_ms:
        first_3 = sum(t for _, t in iter_ms[:3])
        last_3 = sum(t for _, t in iter_ms[-3:])
        max_t = max(t for _, t in iter_ms)
        min_t = min(t for _, t in iter_ms)
        spread = max_t / max(min_t, 1e-9)
        print(f"  [info ] per-iter GPU kernel  = {len(iter_ms)} iters, "
              f"first3={first_3:.0f} ms, last3={last_3:.0f} ms, "
              f"min={min_t:.0f}, max={max_t:.0f}, spread={spread:.1f}×")
        # Flag ONLY if all iters land within 5% of each other (constant timing)
        # AND mode is downbranch/macroatom — that's the silent-noop fingerprint
        # (scatter-mode is naturally flat).
        if spread < 1.05:
            mode = next((re.search(r"Line interaction:\s*(\w+)", ln).group(1)
                         for ln in lines if re.search(r"Line interaction:\s*(\w+)", ln)),
                        "UNKNOWN")
            if mode.upper() in ("DOWNBRANCH", "MACROATOM"):
                warns += 1
                print(f"  [WARN] flat per-iter timing (spread {spread:.2f}×) in {mode} mode — possible silent no-op")

    print()
    if warns:
        print(f"=== {warns} WARN finding(s) ===")
        return 1
    print(f"=== clean ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run directory (looks for stdout.log) or stdout.log file")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        log = os.path.join(args.path, "stdout.log")
    else:
        log = args.path
    if not os.path.isfile(log):
        print(f"[FAIL] log not found: {log}", file=sys.stderr)
        return 2

    rc = scan(log)
    return 2 if rc < 0 else rc


if __name__ == "__main__":
    sys.exit(main())
