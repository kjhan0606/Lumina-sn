#!/usr/bin/env python3
"""Validate a LUMINA tardis_reference_* directory before production use.

Code-INDEPENDENT data integrity checks. Catches the historical bug classes:

  * L2M schema (transition-index vs level-index)  — bug found 2026-05-28 (#305-3)
  * V4 transition_probabilities all-zero          — bug found 2026-05-22
  * negative-lambda lines silently dropped        — bug found 2026-05-25 (#298)
  * abundance non-conservation                     — sanity check
  * levels/macro_atom_references row mismatch      — relational integrity
  * NaN/Inf in physics arrays                      — solver corruption probe

Levels:
  ERROR (exit 2) — hard physics violation, halts production
  WARN  (exit 1) — suspicious pattern, review required
  INFO  (exit 0) — observation, not actionable

Usage: validate_reference.py <ref_dir> [--strict] [--quiet]
       --strict: WARN also exits with code 2
       --quiet : suppress INFO

Returns 0 if all checks pass at the configured strictness.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from kshape_contract import check_contract


class Report:
    def __init__(self, quiet: bool = False) -> None:
        self.errors: list[str] = []
        self.warns:  list[str] = []
        self.infos:  list[str] = []
        self.quiet = quiet

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  [ERROR] {msg}", file=sys.stderr)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"  [WARN ] {msg}", file=sys.stderr)

    def info(self, msg: str) -> None:
        self.infos.append(msg)
        if not self.quiet:
            print(f"  [INFO ] {msg}")

    def ok(self, msg: str) -> None:
        if not self.quiet:
            print(f"  [ok   ] {msg}")


def require_file(ref: str, name: str, rep: Report) -> str | None:
    path = os.path.join(ref, name)
    if not os.path.exists(path):
        rep.err(f"missing required file: {name}")
        return None
    return path


def check_config(ref: str, rep: Report) -> dict:
    """Note: LUMINA runtime parses only time_explosion_s/T_inner_K/luminosity_inner_erg_s/
    n_packets/n_iterations/seed/T_e_T_rad_ratio. n_lines/n_shells/v_* are advisory only
    (runtime uses CSV row counts). Mismatch is stale-metadata, not a runtime bug."""
    path = require_file(ref, "config.json", rep)
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    rep.ok(f"config.json: n_shells={cfg.get('n_shells')} n_lines={cfg.get('n_lines')} (advisory)")
    return cfg


def check_levels_and_macro(ref: str, rep: Report) -> tuple[int | None, pd.DataFrame | None]:
    lp = require_file(ref, "levels.csv", rep)
    mp = require_file(ref, "macro_atom_references.csv", rep)
    if lp is None or mp is None:
        return None, None
    levels = pd.read_csv(lp)
    mar = pd.read_csv(mp)
    if len(levels) != len(mar):
        rep.err(f"levels.csv ({len(levels)}) vs macro_atom_references.csv ({len(mar)}) row mismatch")
        return None, None
    rep.ok(f"levels↔macro_atom_references: {len(mar)} rows (1:1)")
    # check no duplicate (Z, ion, source_level_number) in mar
    key = mar.set_index(["atomic_number", "ion_number", "source_level_number"])
    if key.index.has_duplicates:
        n_dup = key.index.duplicated().sum()
        rep.err(f"macro_atom_references has {n_dup} duplicate (Z, ion, source_level) keys")
    # check ground level (source_level_number=0) exists for every (Z, ion)
    ions = mar[["atomic_number", "ion_number"]].drop_duplicates()
    missing_ground = 0
    for _, row in ions.iterrows():
        sub = mar[(mar.atomic_number == row.atomic_number) & (mar.ion_number == row.ion_number)]
        if 0 not in sub.source_level_number.values:
            missing_ground += 1
    if missing_ground:
        rep.warn(f"{missing_ground} (Z, ion) pairs missing ground level (source_level_number=0)")
    return len(mar), mar


def check_l2m(ref: str, n_macro_levels: int | None, n_lines_cfg: int | None,
              rep: Report) -> None:
    path = require_file(ref, "line2macro_level_upper.npy", rep)
    if path is None or n_macro_levels is None:
        return
    l2m = np.load(path)
    if l2m.dtype.kind not in ("i", "u"):
        rep.err(f"L2M dtype is {l2m.dtype}, expected integer")
        return
    n = l2m.size
    if n_lines_cfg is not None and n != n_lines_cfg:
        rep.warn(f"L2M size {n} != config.n_lines {n_lines_cfg} (config stale; runtime ignores)")
    lo, hi = int(l2m.min()), int(l2m.max())
    out_hi = int((l2m >= n_macro_levels).sum())
    out_lo = int((l2m < -1).sum())
    n_neg1 = int((l2m == -1).sum())   # -1 = legitimate "unmapped"
    in_range = int(((l2m >= 0) & (l2m < n_macro_levels)).sum())
    pct_in = 100.0 * in_range / n
    pct_neg1 = 100.0 * n_neg1 / n
    pct_out = 100.0 * out_hi / n
    msg = f"L2M range: min={lo} max={hi}, in [0,{n_macro_levels})={pct_in:.2f}% (-1={pct_neg1:.2f}%, >={n_macro_levels}={pct_out:.2f}%)"
    if pct_out > 0.0:
        # ANY out-of-range hi value = SMOKING GUN for transition-index bug (#305-3)
        rep.err(msg + " — TRANSITION-INDEX SCHEMA BUG (#305-3 class)")
    elif out_lo > 0:
        rep.err(f"L2M has {out_lo} values < -1 (only -1 = unmapped is legitimate)")
    elif pct_in < 50.0:
        rep.warn(msg + " — high unmapped fraction, downbranch will mostly no-op")
    else:
        rep.ok(msg)


def check_transition_probabilities(ref: str, n_macro_levels: int | None,
                                   rep: Report) -> None:
    path = require_file(ref, "transition_probabilities.npy", rep)
    if path is None:
        return
    tp = np.load(path, mmap_mode="r")
    if tp.ndim != 2:
        rep.err(f"transition_probabilities ndim={tp.ndim}, expected 2")
        return
    n_tr, n_shells = tp.shape
    # NaN/Inf
    chunk = np.asarray(tp[:min(n_tr, 200000)])
    n_nan = int(np.isnan(chunk).sum())
    n_inf = int(np.isinf(chunk).sum())
    if n_nan or n_inf:
        rep.err(f"transition_probabilities has NaN={n_nan} Inf={n_inf} (sampled first 200k rows)")
    # all-zero rows (V4 bug fingerprint = 100%)
    row_abs_sum = np.abs(np.asarray(tp)).sum(axis=1)
    zero_rows = int((row_abs_sum == 0).sum())
    pct_zero = 100.0 * zero_rows / n_tr
    if pct_zero > 95.0:
        rep.err(f"transition_probabilities: {pct_zero:.2f}% all-zero rows — V4-class bug (finalize_cmfgen_ref_npy)")
    elif pct_zero > 50.0:
        rep.warn(f"transition_probabilities: {pct_zero:.2f}% all-zero rows (high sparsity)")
    else:
        rep.ok(f"transition_probabilities: shape=({n_tr}, {n_shells}), {pct_zero:.2f}% all-zero rows")
    # negative values are unphysical for probabilities
    neg = int((chunk < 0).sum())
    if neg:
        rep.err(f"transition_probabilities has {neg} negative values (sampled first 200k rows)")


def check_tau_sobolev(ref: str, n_lines_cfg: int | None, rep: Report) -> None:
    path = require_file(ref, "tau_sobolev.npy", rep)
    if path is None:
        return
    tau = np.load(path, mmap_mode="r")
    if tau.ndim != 2:
        rep.err(f"tau_sobolev ndim={tau.ndim}, expected 2")
        return
    n_lines, n_shells = tau.shape
    if n_lines_cfg is not None and n_lines != n_lines_cfg:
        rep.warn(f"tau_sobolev rows {n_lines} != config.n_lines {n_lines_cfg} (config stale)")
    chunk = np.asarray(tau[:min(n_lines, 200000)])
    n_nan = int(np.isnan(chunk).sum())
    n_inf = int(np.isinf(chunk).sum())
    if n_nan or n_inf:
        rep.err(f"tau_sobolev has NaN={n_nan} Inf={n_inf} (sampled first 200k rows)")
    neg = int((chunk < 0).sum())
    if neg:
        rep.err(f"tau_sobolev has {neg} negative values (sampled)")
    # A zero seed is representable, but K-FRESH (not this static check) must prove
    # that the solver overwrites it before the first physics consumer.
    zero_rows = int((np.abs(np.asarray(tau)).sum(axis=1) == 0).sum())
    pct = 100.0 * zero_rows / n_lines
    if pct == 100.0:
        rep.info("tau_sobolev: 100% zero disk seed (not freshness evidence)")
    else:
        rep.ok(f"tau_sobolev: shape=({n_lines}, {n_shells}), {pct:.2f}% zero-rows")


def check_line_list(ref: str, n_lines_cfg: int | None, n_macro_levels: int | None,
                    mar: pd.DataFrame | None, rep: Report) -> None:
    path = require_file(ref, "line_list.csv", rep)
    if path is None:
        return
    ll = pd.read_csv(path)
    n_lines = len(ll)
    if n_lines_cfg is not None and n_lines != n_lines_cfg:
        rep.warn(f"line_list.csv rows {n_lines} != config.n_lines {n_lines_cfg} (config stale)")
    # negative or zero wavelength: #298 bug (Kurucz theoretical neg-lam dropped if lam>0 filter)
    n_zero_lam = int((ll.wavelength == 0).sum())
    n_neg_lam = int((ll.wavelength < 0).sum())
    if n_zero_lam:
        rep.err(f"line_list has {n_zero_lam} lines with wavelength=0 (#298-class data corruption)")
    if n_neg_lam:
        # negative lam is physically OK (Kurucz theoretical), but bbfix should have abs()'d them
        rep.warn(f"line_list has {n_neg_lam} lines with wavelength<0 — bbfix abs() missed?")
    # f_lu > 1 sanity (oscillator strength)
    n_strong = int((ll.f_lu > 10).sum())
    if n_strong:
        rep.warn(f"line_list has {n_strong} lines with f_lu>10 (autoionizing/garbage data)")
    # nu strictly descending (binary searches in d_calculate_distance_line assume this)
    if "nu" in ll.columns:
        nu = ll.nu.to_numpy()
        n_ascending_pairs = int((np.diff(nu) > 0).sum())
        if n_ascending_pairs > 0:
            first_bad = int(np.argmax(np.diff(nu) > 0)) + 1
            rep.err(f"line_list.csv 'nu' NOT strictly descending: "
                    f"{n_ascending_pairs} ascending pairs, first at i={first_bad} "
                    f"(nu[{first_bad-1}]={nu[first_bad-1]:.4e} < nu[{first_bad}]={nu[first_bad]:.4e}) "
                    f"— binary-search assumption in d_calculate_distance_line violated")
        else:
            rep.ok(f"line_list 'nu' strictly descending across {len(nu)} rows")
    # check that level_number_upper indices reference valid macro levels via (Z, ion, lvl)
    if mar is not None:
        valid_keys = set(zip(mar.atomic_number, mar.ion_number, mar.source_level_number))
        # spot-check 100k random lines
        idx = np.random.default_rng(0).choice(n_lines, size=min(100_000, n_lines), replace=False)
        sub = ll.iloc[idx]
        keys = list(zip(sub.atomic_number, sub.ion_number, sub.level_number_upper))
        miss = sum(1 for k in keys if k not in valid_keys)
        pct = 100.0 * miss / len(keys)
        if pct > 10.0:
            rep.err(f"line_list (Z, ion, lvl_upper) keys: {pct:.2f}% unmapped in mar (sampled {len(keys)})")
        elif pct > 0.1:
            rep.warn(f"line_list (Z, ion, lvl_upper) keys: {pct:.2f}% unmapped (sampled)")
        else:
            rep.ok(f"line_list keys: {100-pct:.2f}% mapped (sampled {len(keys)})")
    rep.ok(f"line_list.csv: {n_lines} rows")


def check_abundances(ref: str, n_shells_cfg: int | None, rep: Report) -> None:
    path = require_file(ref, "abundances.csv", rep)
    if path is None:
        return
    df = pd.read_csv(path)
    shell_cols = [c for c in df.columns if c.isdigit()]
    n_shells = len(shell_cols)
    if n_shells_cfg is not None and n_shells != n_shells_cfg:
        rep.warn(f"abundances.csv has {n_shells} shell columns, config has n_shells={n_shells_cfg}")
    arr = df[shell_cols].to_numpy()
    sums = arr.sum(axis=0)
    bad = int(((sums < 0.999) | (sums > 1.001)).sum())
    if bad:
        rep.err(f"abundances.csv: {bad}/{n_shells} shells with ΣX ∉ [0.999, 1.001] "
                f"(min={sums.min():.6f} max={sums.max():.6f})")
    else:
        rep.ok(f"abundances.csv: ΣX=1 across {n_shells} shells (max dev={abs(sums-1).max():.2e})")
    neg = int((arr < 0).sum())
    if neg:
        rep.err(f"abundances.csv has {neg} negative mass fractions")


def check_geometry(ref: str, rep: Report) -> None:
    path = require_file(ref, "geometry.csv", rep)
    if path is None:
        return
    g = pd.read_csv(path)
    bad = int(((g.r_outer <= g.r_inner) | (g.v_outer <= g.v_inner)).sum())
    if bad:
        rep.err(f"geometry.csv: {bad} shells with r_outer<=r_inner or v_outer<=v_inner")
    monot = int(((g.r_inner.diff().fillna(1) <= 0).sum()))
    if monot:
        rep.err(f"geometry.csv: {monot} non-monotone r_inner steps")
    else:
        rep.ok(f"geometry.csv: {len(g)} shells, monotone radii")


def check_density(ref: str, rep: Report) -> None:
    path = require_file(ref, "density.csv", rep)
    if path is None:
        return
    d = pd.read_csv(path)
    neg_or_zero = int((d.rho <= 0).sum())
    if neg_or_zero:
        rep.err(f"density.csv: {neg_or_zero} shells with rho<=0")
    nan = int(d.rho.isna().sum())
    if nan:
        rep.err(f"density.csv: {nan} NaN rho values")
    if not (neg_or_zero or nan):
        rep.ok(f"density.csv: {len(d)} shells, all rho>0 finite")


def check_size_consistency(ref: str, rep: Report) -> None:
    """line_list.csv rows MUST == tau_sobolev rows MUST == L2M size.
    Runtime CUDA kernels iterate over n_lines = line_list.csv row count and
    index into tau_sobolev[i*n_shells:(i+1)*n_shells] and L2M[i]. A mismatch
    here is a hard runtime bug (OOB read or silently wrong indexing)."""
    sizes: dict[str, int] = {}
    try:
        ll_n = sum(1 for _ in open(os.path.join(ref, "line_list.csv"))) - 1
        sizes["line_list.csv"] = ll_n
    except FileNotFoundError:
        pass
    try:
        tau = np.load(os.path.join(ref, "tau_sobolev.npy"), mmap_mode="r")
        sizes["tau_sobolev.npy"] = tau.shape[0]
    except FileNotFoundError:
        pass
    try:
        l2m = np.load(os.path.join(ref, "line2macro_level_upper.npy"))
        sizes["line2macro_level_upper.npy"] = l2m.size
    except FileNotFoundError:
        pass
    if len(set(sizes.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sizes.items())
        rep.err(f"line-axis size MISMATCH across files: {detail} — runtime OOB or wrong indexing")
    elif sizes:
        v = next(iter(sizes.values()))
        rep.ok(f"line-axis cross-file size consistent: n_lines={v} (across {len(sizes)} files)")


def check_kshape(ref: str, rep: Report) -> None:
    try:
        values = check_contract(Path(ref))
    except (OSError, ValueError) as exc:
        rep.err(f"K-SHAPE contract failed: {exc}")
        return
    rep.ok(
        "K-SHAPE line epoch + NPY hashes: "
        f"{values['line_list_sha256'][:16]}..."
    )


def check_sigma_bf(ref: str, rep: Report) -> None:
    path = os.path.join(ref, "cmfgen_sigma_bf.bin")
    if not os.path.exists(path):
        rep.warn("cmfgen_sigma_bf.bin missing — BF cross sections fall back to Kramers")
        return
    sz = os.path.getsize(path)
    if sz < 1024:
        rep.err(f"cmfgen_sigma_bf.bin too small ({sz} bytes)")
    else:
        rep.ok(f"cmfgen_sigma_bf.bin: {sz/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("--strict", action="store_true", help="WARN exits non-zero too")
    ap.add_argument("--quiet", action="store_true", help="suppress INFO+ok lines")
    args = ap.parse_args()

    if not os.path.isdir(args.ref_dir):
        print(f"[FAIL] not a directory: {args.ref_dir}", file=sys.stderr)
        return 2

    print(f"=== validate_reference: {args.ref_dir} ===")
    rep = Report(quiet=args.quiet)

    cfg = check_config(args.ref_dir, rep)
    n_shells_cfg = cfg.get("n_shells")
    n_lines_cfg = cfg.get("n_lines")

    check_geometry(args.ref_dir, rep)
    check_density(args.ref_dir, rep)
    check_abundances(args.ref_dir, n_shells_cfg, rep)

    n_macro_levels, mar = check_levels_and_macro(args.ref_dir, rep)
    check_l2m(args.ref_dir, n_macro_levels, n_lines_cfg, rep)
    check_transition_probabilities(args.ref_dir, n_macro_levels, rep)
    check_tau_sobolev(args.ref_dir, n_lines_cfg, rep)
    check_line_list(args.ref_dir, n_lines_cfg, n_macro_levels, mar, rep)
    check_size_consistency(args.ref_dir, rep)
    check_kshape(args.ref_dir, rep)
    check_sigma_bf(args.ref_dir, rep)

    print()
    print(f"=== summary: ERROR={len(rep.errors)}  WARN={len(rep.warns)}  INFO={len(rep.infos)} ===")
    if rep.errors:
        return 2
    if rep.warns and args.strict:
        return 2
    if rep.warns:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
