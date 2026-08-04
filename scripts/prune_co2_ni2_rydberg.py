#!/usr/bin/env python3
"""
Task #48: Prune Co II / Ni II Rydberg tail (E > 11 eV) from carsus runtime ref.

Step-10 audit found carsus inflates the >11 eV manifold by 8.8x (Co II)
and 4.2x (Ni II) vs CMFGEN, and that this Rydberg pool dilutes UV-upper
decay branching (mid-quartet 21-24% in carsus vs 43-44% in CMFGEN).

This script removes:
  Co II (Z=27, ion=1) levels with E > 11 eV  : ~2536 levels
  Ni II (Z=28, ion=1) levels with E > 11 eV  : ~918 levels
plus all line_list and macro_atom_data rows that reference them.
Surviving levels are renumbered 0..N-1 per (Z, ion); lines are
re-sorted descending in nu; lines_idx + macro references rebuilt.

Files updated (under data/tardis_reference/, full directory backed up first):
  levels.csv
  line_list.csv
  macro_atom_data.csv
  macro_atom_references.csv
  line2macro_level_upper.npy
"""

import sys
import shutil
import time
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

from kshape_contract import write_contract

ROOT    = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
# Allow overriding REF_DIR via argv[1] (e.g. ../tardis_reference_strat6_highL_pruned)
REF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "tardis_reference"

PRUNE = [(27, 1), (28, 1)]   # (Z, ion) pairs: Co II, Ni II — Cr II removed (caused iter-3 macro-atom hang in 151135)
E_CUT = 11.0                  # eV — drop levels above this


def log(msg):
    print(f"[prune_rydberg] {msg}", flush=True)


def backup():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = ROOT / "data" / f"tardis_reference.bak_pre_rydberg_{stamp}"
    log(f"Backup: {REF_DIR} -> {dst}")
    shutil.copytree(REF_DIR, dst)
    return dst


def main():
    t0 = time.time()
    bak_dir = backup()

    log("Loading…")
    levels = pd.read_csv(REF_DIR / "levels.csv")
    lines  = pd.read_csv(REF_DIR / "line_list.csv")
    macro  = pd.read_csv(REF_DIR / "macro_atom_data.csv")
    refs   = pd.read_csv(REF_DIR / "macro_atom_references.csv")
    l2m    = np.load(REF_DIR / "line2macro_level_upper.npy")
    log(f"  levels={len(levels)} lines={len(lines)} macro={len(macro)} refs={len(refs)}")

    # Drop the unnamed index column from macro if present
    if "Unnamed: 0" in macro.columns:
        macro = macro.drop(columns=["Unnamed: 0"])

    if len(l2m) != len(lines):
        raise RuntimeError(f"l2m {len(l2m)} != lines {len(lines)}")
    if int(refs["count_total"].sum()) != len(macro):
        raise RuntimeError("refs count_total != macro rows (pre)")

    # ---- 1. identify prune set ----
    prune_mask = pd.Series(False, index=levels.index)
    for Z, ion in PRUNE:
        m = (levels.atomic_number == Z) & (levels.ion_number == ion) & (levels.energy_eV > E_CUT)
        prune_mask |= m
    prune_levels = levels[prune_mask]
    keep_levels  = levels[~prune_mask].copy()
    prune_keys = set(zip(prune_levels.atomic_number.astype(int),
                         prune_levels.ion_number.astype(int),
                         prune_levels.level_number.astype(int)))
    log(f"Prune {len(prune_levels)} levels  (kept {len(keep_levels)})")

    # ---- 2. build per-(Z,ion) renumbering map: old_level_number -> new_level_number ----
    # surviving levels keep their original ordering by level_number; new numbers 0..N-1.
    keep_levels = keep_levels.sort_values(["atomic_number", "ion_number", "level_number"]).reset_index(drop=True)
    keep_levels["new_level_number"] = (
        keep_levels.groupby(["atomic_number", "ion_number"]).cumcount()
    )
    rename_map = {(int(r.atomic_number), int(r.ion_number), int(r.level_number)):
                  int(r.new_level_number) for r in keep_levels.itertuples()}
    log(f"Rename map: {len(rename_map)} surviving (Z,ion,old_level) entries")

    # write new levels.csv  (preserve column order, but drop new_level_number after using)
    new_levels = keep_levels.copy()
    new_levels["level_number"] = new_levels["new_level_number"]
    new_levels = new_levels.drop(columns=["new_level_number"])
    new_levels = new_levels[["atomic_number", "ion_number", "level_number",
                              "energy_eV", "g", "metastable"]]

    # ---- 3. filter and renumber line_list.csv ----
    log("Filtering line_list…")
    lln_lower = list(zip(lines.atomic_number.astype(int),
                         lines.ion_number.astype(int),
                         lines.level_number_lower.astype(int)))
    lln_upper = list(zip(lines.atomic_number.astype(int),
                         lines.ion_number.astype(int),
                         lines.level_number_upper.astype(int)))
    keep_line_mask = np.array([(l not in prune_keys) and (u not in prune_keys)
                                for l, u in zip(lln_lower, lln_upper)])
    log(f"  lines kept {keep_line_mask.sum()} / {len(lines)} "
        f"(dropped {(~keep_line_mask).sum()})")

    new_lines = lines[keep_line_mask].copy().reset_index(drop=True)
    # remap level numbers
    new_lines["level_number_lower"] = [rename_map[(int(z), int(i), int(L))]
        for z, i, L in zip(new_lines.atomic_number, new_lines.ion_number, new_lines.level_number_lower)]
    new_lines["level_number_upper"] = [rename_map[(int(z), int(i), int(L))]
        for z, i, L in zip(new_lines.atomic_number, new_lines.ion_number, new_lines.level_number_upper)]

    # ---- 4. sort lines descending by nu (required for binary search in C code) ----
    log("Sorting lines descending nu…")
    sort_idx = np.argsort(-new_lines["nu"].to_numpy(), kind="stable")
    new_lines_sorted = new_lines.iloc[sort_idx].reset_index(drop=True)
    # mapping: old line array index in 'lines' (post-filter) -> new array index
    inv_perm = np.empty(len(sort_idx), dtype=np.int64)
    inv_perm[sort_idx] = np.arange(len(sort_idx))
    # mapping line_id -> new array index (used to rebuild macro lines_idx)
    lid_to_new_arr_idx = {int(lid): int(i) for i, lid in enumerate(new_lines_sorted.line_id)}

    # ---- 5. filter and renumber macro_atom_data.csv ----
    log("Filtering macro_atom_data…")
    src_keys = list(zip(macro.atomic_number.astype(int),
                        macro.ion_number.astype(int),
                        macro.source_level_number.astype(int)))
    dst_keys = list(zip(macro.atomic_number.astype(int),
                        macro.ion_number.astype(int),
                        macro.destination_level_number.astype(int)))
    keep_macro_mask = np.array([(s not in prune_keys) and (d not in prune_keys)
                                for s, d in zip(src_keys, dst_keys)])
    log(f"  macro kept {keep_macro_mask.sum()} / {len(macro)} "
        f"(dropped {(~keep_macro_mask).sum()})")

    new_macro = macro[keep_macro_mask].copy().reset_index(drop=True)
    new_macro["source_level_number"] = [rename_map[(int(z), int(i), int(L))]
        for z, i, L in zip(new_macro.atomic_number, new_macro.ion_number, new_macro.source_level_number)]
    new_macro["destination_level_number"] = [rename_map[(int(z), int(i), int(L))]
        for z, i, L in zip(new_macro.atomic_number, new_macro.ion_number, new_macro.destination_level_number)]
    # remap lines_idx -> new array position via line_id (transition_line_id)
    new_macro["lines_idx"] = [lid_to_new_arr_idx[int(lid)]
                              for lid in new_macro.transition_line_id]

    # ---- 6. filter macro_atom_references.csv ----
    log("Filtering macro_atom_references…")
    rkeys = list(zip(refs.atomic_number.astype(int),
                     refs.ion_number.astype(int),
                     refs.source_level_number.astype(int)))
    keep_refs_mask = np.array([k not in prune_keys for k in rkeys])
    new_refs = refs[keep_refs_mask].copy().reset_index(drop=True)
    new_refs["source_level_number"] = [rename_map[(int(z), int(i), int(L))]
        for z, i, L in zip(new_refs.atomic_number, new_refs.ion_number, new_refs.source_level_number)]
    log(f"  refs kept {len(new_refs)} / {len(refs)}")

    # ---- 7. re-sort macro_atom_data into (atomic_number, ion_number, source_level_number, transition_type)
    #     where transition_type ordering matches original carsus convention:
    #     down section first (interleaved -1, 0 pairs), then up (1).
    #     Original carsus block order: rows are already (-1, 0, -1, 0, ..., 1, 1, ...) per source level.
    #     We sort by (Z, ion, source_level) and use original within-source ordering captured via stable sort.
    # ----
    log("Re-sorting macro_atom_data…")
    new_macro["_orig_idx"] = np.arange(len(new_macro))
    new_macro_sorted = new_macro.sort_values(
        ["atomic_number", "ion_number", "source_level_number", "_orig_idx"],
        kind="stable").reset_index(drop=True)
    macro_sort_perm = new_macro_sorted["_orig_idx"].to_numpy(dtype=np.int64)
    new_macro_sorted = new_macro_sorted.drop(columns=["_orig_idx"])

    # ---- 8. build per-source-level new_refs row index map (Z,ion,L_new) -> ref_idx
    new_refs = new_refs.sort_values(["atomic_number", "ion_number", "source_level_number"]).reset_index(drop=True)
    refidx_map = {(int(r.atomic_number), int(r.ion_number), int(r.source_level_number)): i
                  for i, r in enumerate(new_refs.itertuples())}

    # ---- 9. recompute counts & block_references in new_refs from new_macro_sorted ----
    log("Recomputing counts & block_references…")
    grp = new_macro_sorted.groupby(["atomic_number", "ion_number", "source_level_number"])
    counts = grp["transition_type"].agg(
        count_down=lambda s: int((s == -1).sum()),
        count_up=lambda s: int((s == 1).sum()),
    ).reset_index()

    # Merge into new_refs (preserve order)
    new_refs = new_refs.merge(counts, on=["atomic_number", "ion_number", "source_level_number"],
                              how="left", suffixes=("_old", ""))
    new_refs["count_down"] = new_refs["count_down"].fillna(0).astype(int)
    new_refs["count_up"]   = new_refs["count_up"].fillna(0).astype(int)
    new_refs["count_total"] = 2 * new_refs["count_down"] + new_refs["count_up"]
    if "count_down_old" in new_refs.columns:
        new_refs = new_refs.drop(columns=["count_down_old"])
    if "count_up_old" in new_refs.columns:
        new_refs = new_refs.drop(columns=["count_up_old"])
    if "count_total_old" in new_refs.columns:
        new_refs = new_refs.drop(columns=["count_total_old"])

    # block_references: cumulative sum of count_total
    new_refs["block_references"] = np.concatenate([[0], np.cumsum(new_refs["count_total"].to_numpy())[:-1]]).astype(int)
    new_refs["references_idx"]   = np.arange(len(new_refs), dtype=int)

    # Sanity: total
    if int(new_refs["count_total"].sum()) != len(new_macro_sorted):
        raise RuntimeError(f"refs sum_count_total {int(new_refs['count_total'].sum())} "
                           f"!= macro rows {len(new_macro_sorted)} — block layout broken")

    # ---- 10. fill source_level_idx + destination_level_idx in macro_atom_data ----
    log("Filling level_idx columns in macro_atom_data…")
    src_idx = []
    dst_idx = []
    for r in new_macro_sorted.itertuples():
        s = refidx_map.get((int(r.atomic_number), int(r.ion_number), int(r.source_level_number)), -1)
        d = refidx_map.get((int(r.atomic_number), int(r.ion_number), int(r.destination_level_number)), -1)
        src_idx.append(s); dst_idx.append(d)
    new_macro_sorted["source_level_idx"] = src_idx
    new_macro_sorted["destination_level_idx"] = dst_idx
    if (new_macro_sorted["source_level_idx"] < 0).any():
        raise RuntimeError("Some source_level_idx not found in refs after prune")
    if (new_macro_sorted["destination_level_idx"] < 0).any():
        raise RuntimeError("Some destination_level_idx not found in refs after prune")

    # ---- 11. rebuild line2macro_level_upper for new line ordering ----
    log("Rebuilding line2macro_level_upper…")
    new_l2m = np.empty(len(new_lines_sorted), dtype=np.int64)
    for i, ln in enumerate(new_lines_sorted.itertuples()):
        new_l2m[i] = refidx_map[(int(ln.atomic_number), int(ln.ion_number), int(ln.level_number_upper))]

    # ---- 12. reorder columns + write ----
    log("Writing files…")
    macro_cols = ["atomic_number", "ion_number",
                  "source_level_number", "destination_level_number",
                  "transition_type", "transition_probability",
                  "transition_line_id", "lines_idx",
                  "destination_level_idx", "source_level_idx"]
    new_macro_sorted = new_macro_sorted[macro_cols]
    refs_cols = ["atomic_number", "ion_number", "source_level_number",
                 "count_down", "count_up", "count_total",
                 "block_references", "references_idx"]
    new_refs = new_refs[refs_cols]

    new_levels.to_csv(REF_DIR / "levels.csv", index=False)
    new_lines_sorted.to_csv(REF_DIR / "line_list.csv", index=False)
    new_macro_sorted.to_csv(REF_DIR / "macro_atom_data.csv", index=True)  # carsus emits leading idx col
    new_refs.to_csv(REF_DIR / "macro_atom_references.csv", index=False)
    np.save(REF_DIR / "line2macro_level_upper.npy", new_l2m)

    # Determine n_shells from existing per-shell npy files
    abun = np.load(REF_DIR / "abundances.npy")
    n_shells = abun.shape[-1] if abun.ndim > 1 else int(np.load(REF_DIR / "ion_number_density.npy").shape[-1])
    log(f"  n_shells = {n_shells}")

    # tau_sobolev / j_blues: regenerated each iter (compute_tau_sobolev / MC J_nu).
    # transition_probabilities: NOT regenerated unless LUMINA_DYNAMIC_TRANSPROB=1 —
    # we MUST preserve + remap the original carsus-supplied values, otherwise the
    # macro-atom CDF is all zeros and the kernel branches degenerately.
    n_lines = len(new_lines_sorted)
    n_macro = len(new_macro_sorted)
    np.save(REF_DIR / "tau_sobolev.npy",
            np.zeros((n_lines, n_shells), dtype=np.float64))
    np.save(REF_DIR / "j_blues.npy",
            np.zeros((n_lines, n_shells), dtype=np.float64))
    # If the original transition_probabilities.npy is consistent with macro_atom_data,
    # preserve and remap the carsus-supplied values; otherwise (size mismatch from
    # earlier M1/E2 merges) just write zeros — the runtime regenerates tp every iter
    # via compute_transition_probabilities(), so the on-disk tp is just an initial state.
    orig_tp = np.load(bak_dir / "transition_probabilities.npy")
    if orig_tp.shape[0] == len(macro):
        filtered_tp = orig_tp[keep_macro_mask]
        new_tp = filtered_tp[macro_sort_perm]
        if new_tp.shape != (n_macro, n_shells):
            raise RuntimeError(
                f"new transition_probabilities shape {new_tp.shape} != ({n_macro}, {n_shells})")
        np.save(REF_DIR / "transition_probabilities.npy", new_tp)
        log(f"  Preserved+remapped transition_probabilities ({n_macro}×{n_shells})")
    else:
        log(f"  WARN: orig_tp rows {orig_tp.shape[0]} != macro rows {len(macro)} — "
            f"writing zeros; runtime must regenerate (LUMINA_DYNAMIC_TRANSPROB=1 or default).")
        new_tp = np.zeros((n_macro, n_shells), dtype=np.float64)
        np.save(REF_DIR / "transition_probabilities.npy", new_tp)
    log(f"  Wrote zero-filled tau_sobolev.npy ({n_lines}×{n_shells}), "
        f"transition_probabilities.npy ({n_macro}×{n_shells}), "
        f"j_blues.npy ({n_lines}×{n_shells})")
    contract = write_contract(REF_DIR)
    log(f"  Wrote {contract.name} (line epoch + both NPY hashes)")

    log(f"DONE in {time.time()-t0:.1f}s")
    log(f"  levels: {len(levels)} -> {len(new_levels)}")
    log(f"  lines:  {len(lines)} -> {len(new_lines_sorted)}")
    log(f"  macro:  {len(macro)} -> {len(new_macro_sorted)}")
    log(f"  refs:   {len(refs)} -> {len(new_refs)}")
    log(f"  l2m:    {len(l2m)} -> {len(new_l2m)}")


if __name__ == "__main__":
    main()
