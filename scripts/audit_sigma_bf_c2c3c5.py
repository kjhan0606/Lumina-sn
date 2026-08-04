#!/usr/bin/env python3
"""C2/C3/C5 audit: inspect the pre-baked CMFGEN sigma_bf grid.

Binary format (per src/lumina_atomic.c:load_cmfgen_sigma_bf):
  uint32 magic        = 0xC0FE5BF0
  uint32 version      = 1
  int32  n_levels
  int32  n_freq_bins
  float64 nu_min, nu_max
  int8[n_levels] has_cmfgen  (+ pad to 8-byte alignment)
  float64[n_levels * n_freq_bins] sigma_bf  (row-major, level-major)

Reports:
  C2 — count of (level, freq) entries with sigma_bf > 1e-15 cm^2
  C3 — count of levels whose sigma_bf is non-monotone in nu
  C5 — per-(Z, ion) coverage of CMFGEN sigma_bf, with Fe II spotlight
"""
import argparse
import struct
from pathlib import Path
import numpy as np
import pandas as pd


def read_bin(path: Path):
    with open(path, "rb") as fp:
        magic, version, n_lev, n_freq = struct.unpack("<IIii", fp.read(16))
        if magic != 0x434D4644:  # "CMFD" little-endian (per src/lumina_atomic.c:799)
            raise ValueError(f"bad magic 0x{magic:08X}")
        nu_min, nu_max = struct.unpack("<dd", fp.read(16))
        has = np.frombuffer(fp.read(n_lev), dtype=np.int8).copy()
        pad = (8 - (n_lev % 8)) % 8
        fp.seek(pad, 1)
        sigma = np.frombuffer(
            fp.read(n_lev * n_freq * 8), dtype=np.float64).reshape(n_lev, n_freq).copy()
    return n_lev, n_freq, nu_min, nu_max, has, sigma


def main(ref_dir: Path) -> int:
    bin_path = ref_dir / "cmfgen_sigma_bf.bin"
    lev_path = ref_dir / "levels.csv"
    n_lev, n_freq, nu_min, nu_max, has, sigma = read_bin(bin_path)
    lev = pd.read_csv(lev_path)
    print(f"[audit] {bin_path.name}: n_levels={n_lev:,} n_freq_bins={n_freq} "
          f"nu in [{nu_min:.3e}, {nu_max:.3e}] Hz")
    print(f"[audit] coverage: {int(has.sum()):,} / {n_lev:,} levels "
          f"({100*has.sum()/n_lev:.2f}%)")
    if n_lev != len(lev):
        print(f"[audit] WARN: bin n_levels={n_lev} != levels.csv rows={len(lev)}")
        return 1

    # C2
    big = sigma > 1e-15
    n_big = int(np.sum(big))
    lev_big = int(np.sum(big.any(axis=1)))
    print()
    print(f"[C2] entries with sigma_bf > 1e-15 cm^2: {n_big:,} / {sigma.size:,} "
          f"on {lev_big:,} distinct levels")
    if lev_big > 0:
        lvl_max = sigma.max(axis=1)
        worst = np.argsort(-lvl_max)[:10]
        print(f"[C2] top-10 sigma_bf-max levels:")
        for i in worst:
            r = lev.iloc[i]
            print(f"     level {i:6d}  Z={int(r.atomic_number):2d}  "
                  f"ion={int(r.ion_number):2d}  ln={int(r.level_number):4d}  "
                  f"E={r.energy_eV:8.3f} eV  g={r.g:5.1f}  "
                  f"max_sigma={lvl_max[i]:.3e}")

    # C3 — non-monotone (any positive forward diff with sigma > 1e-25)
    nonmono = np.zeros(n_lev, dtype=bool)
    for i in range(n_lev):
        if has[i] == 0:
            continue
        diff = np.diff(sigma[i])
        nonmono[i] = bool(np.any((diff > 0) & (sigma[i, :-1] > 1e-25)))
    n_nonmono = int(np.sum(nonmono))
    print()
    print(f"[C3] non-monotone levels (positive forward diff): "
          f"{n_nonmono:,} / {int(has.sum()):,} with CMFGEN data")
    is_ground = (lev["level_number"] == 0).to_numpy()
    is_excited1 = (lev["level_number"] == 1).to_numpy()
    print(f"[C3]   ground-level non-monotone:        {int(np.sum(nonmono & is_ground)):,}")
    print(f"[C3]   first-excited-level non-monotone: {int(np.sum(nonmono & is_excited1)):,}")

    # C5 — per-(Z, ion) coverage
    print()
    print("[C5] worst 25 (Z, ion) by CMFGEN sigma_bf coverage:")
    print(f"     {'Z':>3s} {'ion':>4s} | {'n_lev':>7s} | {'n_cmfgen':>9s} | {'%cov':>7s}")
    rows = []
    for (z, ion), grp in lev.groupby(["atomic_number", "ion_number"]):
        idx = grp.index.to_numpy()
        n_l = len(idx)
        n_c = int(has[idx].sum())
        rows.append((int(z), int(ion), n_l, n_c, 100.0 * n_c / max(n_l, 1)))
    rows.sort(key=lambda r: r[4])
    for z, ion, n_l, n_c, pct in rows[:25]:
        print(f"     {z:3d} {ion:4d} | {n_l:7,d} | {n_c:9,d} | {pct:6.2f}%")
    print(f"     ... ({max(0, len(rows)-25)} more (Z, ion) pairs)")

    fe2 = [r for r in rows if r[0] == 26 and r[1] == 1]
    if fe2:
        z, ion, n_l, n_c, pct = fe2[0]
        print()
        print(f"[C5] Fe II spotlight: {n_c:,}/{n_l:,} ({pct:.2f}%) levels with "
              f"CMFGEN sigma_bf — gap = {n_l-n_c:,} levels using Kramers fallback")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=Path)
    args = ap.parse_args()
    raise SystemExit(main(args.ref_dir))
