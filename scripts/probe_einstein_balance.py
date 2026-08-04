#!/usr/bin/env python3
"""Pre-flight probe for B2 (Einstein detailed balance) rebuild.

Loads <ref_dir>/{line_list.csv, levels.csv}, computes the canonical CGS
J_nu-convention values of B_ul, B_lu, f_lu from A_ul + g + nu, and reports
the largest fractional drift vs the stored values per (Z, ion). Use this
BEFORE running finalize_cmfgen_ref_npy.py B2 on a fresh ref dir so the
expected magnitude of the column changes is known up front.

Usage:
  python3 scripts/probe_einstein_balance.py <ref_dir>
  python3 scripts/probe_einstein_balance.py <ref_dir> --ion 26,1   # Fe II only
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

C_CGS  = 2.99792458e10
H_CGS  = 6.62607015e-27
ME_CGS = 9.1093837015e-28
E_CGS  = 4.80320425e-10


def main(ref_dir: Path, ion_filter: tuple[int, int] | None) -> int:
    ll = pd.read_csv(ref_dir / "line_list.csv")
    lev = pd.read_csv(ref_dir / "levels.csv")
    g_lookup = {(int(r.atomic_number), int(r.ion_number), int(r.level_number)):
                float(r.g) for r in lev.itertuples(index=False)}

    if ion_filter is not None:
        z, ion = ion_filter
        mask = (ll["atomic_number"] == z) & (ll["ion_number"] == ion)
        ll = ll[mask].reset_index(drop=True)
        print(f"[probe] filtered to Z={z} ion={ion}: {len(ll):,} lines")

    g_l = np.array([g_lookup.get(
        (int(r.atomic_number), int(r.ion_number), int(r.level_number_lower)), np.nan)
        for r in ll.itertuples(index=False)], dtype=np.float64)
    g_u = np.array([g_lookup.get(
        (int(r.atomic_number), int(r.ion_number), int(r.level_number_upper)), np.nan)
        for r in ll.itertuples(index=False)], dtype=np.float64)
    n_missing = int(np.sum(np.isnan(g_l) | np.isnan(g_u)))
    if n_missing > 0:
        print(f"[probe] WARN: {n_missing} lines have missing g_l or g_u")

    nu  = ll["nu"].to_numpy(dtype=np.float64)
    lam = ll["wavelength_cm"].to_numpy(dtype=np.float64)
    A_ul = ll["A_ul"].to_numpy(dtype=np.float64)

    B_ul_new = (C_CGS**2 / (2.0 * H_CGS * nu**3)) * A_ul
    B_lu_new = (g_u / g_l) * B_ul_new
    f_lu_new = (ME_CGS * C_CGS / (8.0 * np.pi**2 * E_CGS**2)) * \
               (lam**2) * (g_u / g_l) * A_ul

    # Detailed balance check (should be exact on recomputed values)
    db = np.abs(B_lu_new * g_l - B_ul_new * g_u) / np.maximum(np.abs(B_lu_new * g_l), 1e-300)
    print(f"[probe] detailed-balance residual (recomputed): max = {np.nanmax(db):.3e} "
          f"(target < 1e-12)")

    # Detailed balance violation on STORED values (the bug we're fixing)
    db_stored = np.abs(ll["B_lu"].to_numpy() * g_l - ll["B_ul"].to_numpy() * g_u) \
              / np.maximum(np.abs(ll["B_lu"].to_numpy() * g_l), 1e-300)
    n_violations = int(np.sum(db_stored > 1e-6))
    print(f"[probe] detailed-balance violations on STORED values "
          f"(|ΔB/B|>1e-6): {n_violations:,} / {len(ll):,} "
          f"({100*n_violations/max(len(ll),1):.2f}%)")
    if n_violations:
        print(f"[probe]   max stored residual = {np.nanmax(db_stored):.3e}")

    # Per-(Z,ion) drift
    def frac(new, old):
        new = np.asarray(new, dtype=np.float64)
        old = np.asarray(old, dtype=np.float64)
        ok = np.isfinite(old) & (np.abs(old) > 0) & np.isfinite(new)
        if not ok.any():
            return float("nan"), float("nan")
        d = np.abs(new[ok] - old[ok]) / np.abs(old[ok])
        return float(np.median(d)), float(np.max(d))

    print()
    print("                Z ion |   N_lines | median ΔB_ul/B_ul |   max ΔB_ul/B_ul "
          "| median Δf_lu/f_lu |   max Δf_lu/f_lu")
    print("-" * 124)
    keys = ll[["atomic_number", "ion_number"]].apply(tuple, axis=1)
    summary = []
    for (z, ion), idx in ll.groupby(["atomic_number", "ion_number"]).groups.items():
        i = np.array(idx, dtype=np.int64)
        m_bul, x_bul = frac(B_ul_new[i], ll["B_ul"].to_numpy()[i])
        m_flu, x_flu = frac(f_lu_new[i], ll["f_lu"].to_numpy()[i])
        summary.append((int(z), int(ion), len(i), m_bul, x_bul, m_flu, x_flu))
    # Sort by max ΔB_ul descending
    summary.sort(key=lambda r: -r[4] if np.isfinite(r[4]) else 0.0)
    for z, ion, n, m_bul, x_bul, m_flu, x_flu in summary[:20]:
        print(f"  Z={z:2d} ion={ion:2d}  | {n:9,d} | "
              f"        {m_bul:.3e} |       {x_bul:.3e} | "
              f"        {m_flu:.3e} |       {x_flu:.3e}")
    if len(summary) > 20:
        print(f"  ... and {len(summary)-20} more (Z,ion) pairs")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("--ion", default=None,
                    help="filter to Z,ion (e.g. 26,1 for Fe II)")
    args = ap.parse_args()
    ion_filter = None
    if args.ion is not None:
        z_s, ion_s = args.ion.split(",")
        ion_filter = (int(z_s), int(ion_s))
    raise SystemExit(main(args.ref_dir, ion_filter))
