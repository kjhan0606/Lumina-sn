#!/usr/bin/env python3
"""Per-iter convergence telemetry across line-interaction modes.

Reads stdout.log files from N runs (typically scatter/downbranch/macroatom for
the same input setup) and prints a side-by-side per-iter table of:
    GPU kernel ms, escaped %, reabsorbed %, T_inner, T_rad@shell 0, W@shell 0,
    L_em (when printed).

Use to diagnose:
  * Mode silent no-op (timings identical across modes)
  * Convergence regression (T_inner err blows up)
  * Plasma instability (L_em/L_req oscillates)
  * Cold/hot run divergence (T_rad inner runs away)

Usage:
    mode_convergence_telemetry.py <run_dir_or_log> [<run_dir_or_log> ...]
                                  [--csv out.csv]   # also dump joined table
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from collections import defaultdict


def find_log(p: str) -> str | None:
    if os.path.isfile(p):
        return p
    cand = os.path.join(p, "stdout.log")
    return cand if os.path.isfile(cand) else None


def parse_log(path: str) -> dict[int, dict[str, float]]:
    """Return {iter: {metric: value}} from a LUMINA stdout.log."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    cur = None
    with open(path, errors="replace") as f:
        for ln in f:
            m = re.search(r"Iteration\s+(\d+)/\d+", ln)
            if m:
                cur = int(m.group(1))
                continue
            if cur is None:
                continue
            m = re.search(r"GPU kernel:\s*([0-9.]+)\s*ms", ln)
            if m:
                out[cur]["gpu_ms"] = float(m.group(1))
                continue
            m = re.search(r"Escaped:\s*\d+\s*\(([0-9.]+)%\),\s*Reabsorbed:\s*\d+\s*\(([0-9.]+)%\)", ln)
            if m:
                out[cur]["esc_pct"] = float(m.group(1))
                out[cur]["reabs_pct"] = float(m.group(2))
                continue
            # shell 0 plasma line:  "    0    W   T_rad   T_e   ..."
            m = re.search(r"^\s*0\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*K\s+([0-9.eE+-]+)\s*K\s+([0-9.]+)", ln)
            if m and "W0" not in out[cur]:
                out[cur]["W0"] = float(m.group(1))
                out[cur]["T_rad0"] = float(m.group(2))
                out[cur]["T_e0"] = float(m.group(3))
                continue
            m = re.search(r"T_inner\s*=\s*([0-9.]+)\s*K", ln)
            if m:
                out[cur]["T_inner"] = float(m.group(1))
                continue
            m = re.search(r"L_em\s*=\s*([0-9.eE+-]+).*L_req\s*=\s*([0-9.eE+-]+)", ln)
            if m:
                out[cur]["L_em"] = float(m.group(1))
                out[cur]["L_req"] = float(m.group(2))
                continue
    return dict(out)


def label_of(path: str, fallback: str) -> str:
    """Pull line-interaction mode from log; fall back to dir basename."""
    log = find_log(path) or path
    try:
        with open(log, errors="replace") as f:
            for ln in f:
                m = re.search(r"Line interaction:\s*(\w+)", ln)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", help="write joined table here too")
    args = ap.parse_args()

    parsed: list[tuple[str, str, dict[int, dict[str, float]]]] = []
    for r in args.runs:
        log = find_log(r)
        if log is None:
            print(f"[FAIL] no stdout.log under {r}", file=sys.stderr)
            return 2
        d = parse_log(log)
        lbl = label_of(log, os.path.basename(r.rstrip("/")))
        parsed.append((r, lbl, d))
        print(f"  parsed {log}: {len(d)} iters, mode={lbl}")

    # union of iters
    all_iters = sorted({i for _, _, d in parsed for i in d})
    metrics = ["gpu_ms", "esc_pct", "T_inner", "T_rad0", "T_e0", "W0", "L_em"]

    print()
    header = f"{'iter':>4s}"
    for r, lbl, _ in parsed:
        header += f"  | {lbl:>10s}_{'mode':<8s}"
    # build column headers more carefully
    cols = ["iter"]
    for _, lbl, _ in parsed:
        for m in metrics:
            cols.append(f"{lbl[:5]}.{m}")
    print("  " + "  ".join(f"{c:>12s}" for c in cols))

    rows = []
    for it in all_iters:
        row = [str(it)]
        for _, _, d in parsed:
            di = d.get(it, {})
            for m in metrics:
                v = di.get(m)
                row.append(f"{v:>12.4g}" if isinstance(v, (int, float)) else f"{'-':>12s}")
        rows.append(row)
        print("  " + "  ".join(f"{c:>12s}" for c in row))

    # Mode-equiv quick test on timings
    if len(parsed) >= 2:
        print()
        for i in range(len(parsed) - 1):
            for j in range(i + 1, len(parsed)):
                ra, la, da = parsed[i]
                rb, lb, db = parsed[j]
                shared = sorted(set(da) & set(db))
                gm_a = [da[it].get("gpu_ms", 0) for it in shared]
                gm_b = [db[it].get("gpu_ms", 0) for it in shared]
                if not (any(gm_a) and any(gm_b)):
                    continue
                # ratio of medians
                gm_a_s = sorted(gm_a); gm_b_s = sorted(gm_b)
                med_a = gm_a_s[len(gm_a_s) // 2]
                med_b = gm_b_s[len(gm_b_s) // 2]
                ratio = med_a / max(med_b, 1e-9)
                tag = "(distinct)" if 0.5 < ratio < 2.0 else ""
                print(f"  median GPU kernel: {la}={med_a:.1f}ms  vs  {lb}={med_b:.1f}ms  (ratio={ratio:.2f}) {tag}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(s.strip() for s in r) + "\n")
        print(f"\n  wrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
