#!/usr/bin/env python3
"""Patch line2macro_level_upper.npy: remap from TARDIS transition-index
to LUMINA macro-level-index expected by d_macro_atom_interaction.

Source bug: export_tardis_reference.py wrote `lines_upper2macro_reference_idx`
which is a transition-index in [0, n_transitions). The CUDA kernel expects
the macro-level row index in [0, n_macro_levels). 99.46% of lines fail
bounds check and silently disable downbranch/macroatom in the line channel.

Fix: rebuild from line_list.csv + macro_atom_references.csv using the
(atomic_number, ion_number, level_number_upper) → row_index mapping.

Usage: python3 patch_line2macro.py <ref_dir>
"""
import sys
import os
import shutil
import time
import numpy as np
import pandas as pd


def patch(ref_dir: str) -> None:
    l2m_path = os.path.join(ref_dir, "line2macro_level_upper.npy")
    ll_path = os.path.join(ref_dir, "line_list.csv")
    mar_path = os.path.join(ref_dir, "macro_atom_references.csv")

    for p in (l2m_path, ll_path, mar_path):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    ll = pd.read_csv(ll_path)
    mar = pd.read_csv(mar_path)
    n_lines = len(ll)
    n_macro_levels = len(mar)
    print(f"[INFO] n_lines={n_lines}  n_macro_levels={n_macro_levels}")

    key_cols = ["atomic_number", "ion_number", "source_level_number"]
    mar_lookup = {tuple(int(v) for v in row): i
                  for i, row in enumerate(mar[key_cols].itertuples(index=False))}

    line_keys = ll[["atomic_number", "ion_number", "level_number_upper"]].to_numpy(dtype=np.int64)
    new_l2m = np.full(n_lines, -1, dtype=np.int64)
    n_mapped = 0
    for i, (z, ion, lu) in enumerate(line_keys):
        idx = mar_lookup.get((int(z), int(ion), int(lu)))
        if idx is not None:
            new_l2m[i] = idx
            n_mapped += 1

    print(f"[INFO] mapped {n_mapped}/{n_lines} ({100*n_mapped/n_lines:.2f}%)")
    print(f"[INFO] new range: min={new_l2m.min()}  max={new_l2m.max()}")
    in_range = ((new_l2m >= 0) & (new_l2m < n_macro_levels)).sum()
    print(f"[INFO] in [0, {n_macro_levels}): {in_range}  out: {n_lines - in_range}")

    # Backup original
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{l2m_path}.bak_{ts}"
    shutil.copy2(l2m_path, bak)
    old = np.load(bak)
    print(f"[INFO] backup -> {bak}")
    print(f"[INFO] OLD range: min={old.min()}  max={old.max()}  "
          f"out_of_range_frac={100*(old>=n_macro_levels).mean():.2f}%")

    np.save(l2m_path, new_l2m)
    print(f"[DONE] wrote {l2m_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: patch_line2macro.py <ref_dir>")
    patch(sys.argv[1])
