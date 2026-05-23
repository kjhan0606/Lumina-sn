#!/usr/bin/env python3
"""Emit the .npy companion files LUMINA expects in a CMFGEN-built ref dir.

Inputs read from REF_DIR:
  levels.csv, line_list.csv, macro_atom_references.csv, macro_atom_data.csv

Outputs written into REF_DIR:
  line2macro_level_upper.npy   [n_lines]            int64 — global level idx
                                                          of each line's upper
  tau_sobolev.npy              [n_lines, n_shells]  float64 zeros (loader
                                                          recomputes per iter)
  transition_probabilities.npy [n_trans, n_shells]  float64 zeros (loader
                                                          recomputes per iter)

Default n_shells=30 matches the production driver scripts. LUMINA's loader
reinitializes either array to zeros if its column count != actual n_shells,
so the value here is only a heuristic.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def main(ref_dir: Path, n_shells: int = 30) -> None:
    print(f"[finalize] ref_dir = {ref_dir}")
    print(f"[finalize] n_shells (initial guess) = {n_shells}")

    mr = pd.read_csv(ref_dir / "macro_atom_references.csv")
    lookup: dict[tuple[int, int, int], int] = {
        (int(r.atomic_number), int(r.ion_number), int(r.source_level_number)):
            int(r.references_idx)
        for r in mr.itertuples(index=False)
    }
    print(f"[finalize] level lookup: {len(lookup):,} entries")

    ll = pd.read_csv(ref_dir / "line_list.csv")
    n_lines = len(ll)
    print(f"[finalize] line_list:    {n_lines:,} rows")

    line2macro_level_upper = np.empty(n_lines, dtype=np.int64)
    missing = 0
    for i, r in enumerate(ll.itertuples(index=False)):
        key = (int(r.atomic_number), int(r.ion_number), int(r.level_number_upper))
        if key in lookup:
            line2macro_level_upper[i] = lookup[key]
        else:
            line2macro_level_upper[i] = -1
            missing += 1
    if missing:
        print(f"[finalize] WARN: {missing} lines with missing upper-level lookup")
    np.save(ref_dir / "line2macro_level_upper.npy", line2macro_level_upper)
    print(f"[finalize] wrote line2macro_level_upper.npy")

    ma_lines = sum(1 for _ in open(ref_dir / "macro_atom_data.csv")) - 1
    print(f"[finalize] macro_atom_data: {ma_lines:,} rows")

    tau_sob = np.zeros((n_lines, n_shells), dtype=np.float64)
    np.save(ref_dir / "tau_sobolev.npy", tau_sob)
    print(f"[finalize] wrote tau_sobolev.npy  [{n_lines}, {n_shells}]")

    tp = np.zeros((ma_lines, n_shells), dtype=np.float64)
    np.save(ref_dir / "transition_probabilities.npy", tp)
    print(f"[finalize] wrote transition_probabilities.npy  [{ma_lines}, {n_shells}]")

    print("[finalize] done.")


if __name__ == "__main__":
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tardis_reference_cmfgen")
    n_sh = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(ref.resolve(), n_sh)
