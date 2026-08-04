#!/usr/bin/env python3
"""#280 Bake CMFGEN sigma_bf binary against a *ref-merged* levels.csv.

The CMFGEN-only expander writes a bin with n_levels = #CMFGEN levels (21460
for the capraise tree).  But at runtime the C loader compares against the
ref-merged level count (33829 for v3_femerge_capraise).  Mismatch → loader
rejects the bin (`src/lumina_atomic.c:744`) and falls back to per-level
Kramers for all 33829 levels.  Net: UV blanketing under-modeled 7-38× vs HST.

This script reads <ref_dir>/levels.csv, builds a (Z, ion+1, level+1) →
row_idx lookup matching the ref's ordering, then calls the same
`bake_sigma_bf_grid` used by the CMFGEN expander — but with `n_levels`
matching the ref.  Levels not in the CMFGEN ion_data (V3-sourced ions,
beyond-cap iron-peak III levels, etc.) get has_cmfgen=0 → C loader uses
Kramers per-level as before.  Iron-peak III levels in CMFGEN ion_data get
real tabulated σ_bf curves.

Usage:
    python3 scripts/bake_sigma_bf_for_ref.py <ref_dir>

Writes <ref_dir>/cmfgen_sigma_bf.bin.  The slurm runner picks this up via
LUMINA_CMFGEN_SIGMA_BF (ref-local path takes precedence over the global
data/atomic/cmfgen_sigma_bf.bin).
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

# Import the heavy lifting from the expander.
import expand_atomic_data_cmfgen as ex


def load_ref_levels(ref_dir: Path):
    """Read ref-merged levels.csv → (levels_rows, level_lookup).

    levels.csv columns: atomic_number,ion_number,level_number,energy_eV,g,metastable
    Row order = level order in the ref tree (same ordering used by macro-atom
    references, line_list lo/up indices, etc.).
    """
    rows = []
    lookup = {}
    src = ref_dir / 'levels.csv'
    with open(src) as fp:
        reader = csv.DictReader(fp)
        for i, r in enumerate(reader):
            Z      = int(r['atomic_number'])
            ion_csv = int(r['ion_number'])
            lvl_csv = int(r['level_number'])
            E_eV    = float(r['energy_eV'])
            g_val   = int(round(float(r['g'])))
            meta    = int(r['metastable'])
            rows.append((Z, ion_csv, lvl_csv, E_eV, g_val, meta))
            # The bake function looks up via (Z, stage, cmfgen_id_1based)
            # where stage = ion_csv+1 and cmfgen_id_1based = lvl_csv+1.
            lookup[(Z, ion_csv + 1, lvl_csv + 1)] = i
    return rows, lookup


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)

    ref_dir = Path(sys.argv[1]).resolve()
    if not (ref_dir / 'levels.csv').exists():
        sys.exit(f"ERROR: {ref_dir}/levels.csv not found")

    print(f"=== bake sigma_bf for ref: {ref_dir} ===")

    # Read ref level table
    levels_rows, level_lookup = load_ref_levels(ref_dir)
    n_levels_ref = len(levels_rows)
    print(f"ref levels: {n_levels_ref}")

    # Per-(Z, ion) row count for coverage report later
    per_ion_ref = Counter((Z, ion_csv) for (Z, ion_csv, *_) in levels_rows)

    # Parse CMFGEN ions (same data the expander uses)
    ion_data = ex.parse_all_ions()
    print(f"CMFGEN ions: {len(ion_data)}")

    # Bake into a binary sized to the ref level count
    out_bin = ref_dir / 'cmfgen_sigma_bf.bin'
    n_baked, n_covered = ex.bake_sigma_bf_grid(
        ion_data, ion_data, levels_rows, level_lookup, out_bin
    )
    print(f"\nbaked {n_baked} curves -> {n_covered}/{n_levels_ref} ref levels "
          f"covered ({100 * n_covered / n_levels_ref:.1f}%)")
    print(f"wrote {out_bin} "
          f"({n_levels_ref} x {ex.BF_N_FREQ_BIN} doubles = "
          f"{n_levels_ref * ex.BF_N_FREQ_BIN * 8 / (1024 * 1024):.0f} MB)")

    # Per-(Z, ion) coverage report
    print("\nPer-ion coverage (ref-merged levels that got CMFGEN σ_bf):")
    print(f"  {'Z':>3}.{'ion':<3} {'cmfgen':>8} {'ref':>8} {'covered':>8}")
    cmfgen_keys = sorted(ion_data.keys())
    for (Z, stage) in cmfgen_keys:
        ion_csv = stage - 1
        n_cmfgen = ion_data[(Z, stage)]['n_kept']
        n_ref    = per_ion_ref.get((Z, ion_csv), 0)
        # Count how many of this ion's ref-level rows actually mapped
        # successfully via the CMFGEN lookup.
        # (n_baked is a global count of phot entries; per-ion covered is
        # the number of (Z, ion_csv, *) ref rows whose (Z, stage, lvl+1)
        # key exists in the CMFGEN-keyed lookup AND ended up with sigma>0.
        # The bake function already restricts itself to ion_data ion keys,
        # so per-ion coverage is bounded by min(n_cmfgen, n_ref).)
        cov_upper = min(n_cmfgen, n_ref)
        sym = ex.SYM[Z] if Z < len(ex.SYM) else f"Z{Z}"
        rom = ex.ROMAN[stage] if stage < len(ex.ROMAN) else f"st{stage}"
        print(f"  {sym:>2} {rom:<4}  cmfgen={n_cmfgen:5d}  ref={n_ref:5d}  "
              f"≤covered={cov_upper:5d}")


if __name__ == '__main__':
    main()
