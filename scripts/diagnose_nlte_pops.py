#!/usr/bin/env python3
"""NLTE solver health diagnostic — catches #219e singularity class.

Reads `nlte_levels_iter*.csv` dumped via LUMINA_NLTE_LEVEL_DUMP=1 and flags
per (Z, ion, shell) solutions that look pathological. No access to the rate
matrix itself is required — pathologies are inferred from the population
distribution.

Bug class detected:

  #219e singularity:
    Cr/Fe/Co III had exactly one bb-isolated TOP level → rank-deficient
    rate matrix → garbage x[N-1] ≈ n_ion (entire ion mass dumped to the
    top level). Fingerprint: argmax_pop = top level AND pop[top]/Σpop > 0.99.

  Boltzmann inversion:
    Excited level pop > ground level pop (population inversion) when
    plasma is NOT lasing. Fingerprint: pop[any l>0] > pop[ground].

  Population non-conservation:
    Σpop ≠ n_ion_total (the global solver and the per-level solver disagree).

  Negative populations:
    Always unphysical.

Usage:
    diagnose_nlte_pops.py <csv-or-dir> [--top-frac 0.5] [--inv-ratio 1.0]
                                       [--cons-tol 0.01] [--quiet]

Exit codes:
    0 = no anomalies above thresholds
    1 = anomalies found (review)
    2 = file unreadable / invalid format
"""
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import numpy as np


def analyze_one(df: pd.DataFrame,
                top_frac: float,
                inv_ratio: float,
                cons_tol: float,
                src: str) -> dict:
    """Return summary counts; print per-anomaly lines."""
    flags = {"top_dom": 0, "inv": 0, "neg": 0, "noncons": 0, "groups": 0}
    if not {"Z", "ion", "shell", "level_idx", "n_pop", "n_ion_total"}.issubset(df.columns):
        print(f"  [ERROR] {src}: missing required columns")
        return flags
    # negatives anywhere
    n_neg = int((df.n_pop < 0).sum())
    if n_neg > 0:
        flags["neg"] = n_neg
        print(f"  [ERROR] {src}: {n_neg} rows with n_pop < 0")
    for (Z, ion, shell), sub in df.groupby(["Z", "ion", "shell"], sort=False):
        flags["groups"] += 1
        sub = sub.sort_values("level_idx")
        pops = sub.n_pop.to_numpy()
        levs = sub.level_idx.to_numpy()
        if pops.size < 2:
            continue
        # n_ion_total is per-row constant within a (Z, ion, shell); pick first
        n_ion = float(sub.n_ion_total.iloc[0])
        s = float(pops.sum())
        # 1) TOP-level dominance (#219e fingerprint)
        argmax_l = int(levs[int(np.argmax(pops))])
        top_l = int(levs[-1])
        frac_top = float(pops[-1]) / s if s > 0 else 0.0
        if argmax_l == top_l and frac_top > top_frac:
            flags["top_dom"] += 1
            print(f"  [WARN] Z={Z:2d} ion={ion} shell={shell:2d}: "
                  f"TOP-LEVEL DOMINANT (#219e fingerprint) "
                  f"pop[top={top_l}]/Σpop={frac_top:.4f} (>{top_frac})")
        # 2) Boltzmann inversion: any excited level > ground
        if pops[0] > 0:
            max_ratio = float(pops[1:].max() / pops[0])
            if max_ratio > inv_ratio:
                # find which level
                bad_l = int(levs[1 + int(np.argmax(pops[1:]))])
                flags["inv"] += 1
                print(f"  [WARN] Z={Z:2d} ion={ion} shell={shell:2d}: "
                      f"INVERSION level {bad_l} / ground = {max_ratio:.3f}")
        # 3) Population conservation Σpop ≈ n_ion_total
        if n_ion > 0:
            rel = abs(s - n_ion) / n_ion
            if rel > cons_tol:
                flags["noncons"] += 1
                # only print first 5 of these (can be noisy)
                if flags["noncons"] <= 5:
                    print(f"  [WARN] Z={Z:2d} ion={ion} shell={shell:2d}: "
                          f"Σpop={s:.3e} vs n_ion_total={n_ion:.3e} "
                          f"(rel_err={rel:.3f})")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="nlte_levels_iter*.csv file or directory of them")
    ap.add_argument("--top-frac", type=float, default=0.50,
                    help="flag if pop[TOP] / Σpop > top-frac AND argmax=TOP (default 0.50)")
    ap.add_argument("--inv-ratio", type=float, default=1.0,
                    help="flag if pop[any excited] / pop[ground] > inv-ratio (default 1.0)")
    ap.add_argument("--cons-tol", type=float, default=0.05,
                    help="flag if |Σpop − n_ion_total| / n_ion_total > cons-tol (default 0.05)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if f.startswith("nlte_levels_iter") and f.endswith(".csv"))
        if not files:
            print(f"[FAIL] no nlte_levels_iter*.csv in {args.path}", file=sys.stderr)
            return 2
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        print(f"[FAIL] not a file or dir: {args.path}", file=sys.stderr)
        return 2

    total = {"top_dom": 0, "inv": 0, "neg": 0, "noncons": 0, "groups": 0, "files": 0}
    for f in files:
        if not args.quiet:
            print(f"=== {os.path.basename(f)} ===")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [ERROR] cannot read: {e}", file=sys.stderr)
            return 2
        flags = analyze_one(df, args.top_frac, args.inv_ratio, args.cons_tol, os.path.basename(f))
        for k in flags:
            total[k] += flags[k]
        total["files"] += 1
        if not args.quiet:
            print(f"  groups={flags['groups']}  top_dom={flags['top_dom']}  "
                  f"inv={flags['inv']}  noncons={flags['noncons']}  neg={flags['neg']}")

    print()
    print(f"=== summary ({total['files']} files, {total['groups']} (Z, ion, shell) groups) ===")
    print(f"  TOP-level dominant (#219e):   {total['top_dom']}")
    print(f"  Boltzmann inversions:         {total['inv']}")
    print(f"  Population non-conservation:  {total['noncons']}")
    print(f"  Negative populations:         {total['neg']}")

    if total["top_dom"] or total["neg"]:
        return 1
    if total["inv"] or total["noncons"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
