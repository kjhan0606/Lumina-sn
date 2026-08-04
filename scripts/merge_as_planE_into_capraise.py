#!/usr/bin/env python3
"""#282 cross-walk merge: append AS planE high-E levels + new UV bb lines
into the existing v3_femerge_capraise reference.

Strategy: for each AS ion (Cr/Mn/Ni III):
  1. (g,E)-match every AS level against capraise levels (|ΔE|<0.5 eV, same g).
     - Match found → AS uses capraise's level_number.
     - No match AND E_AS > E_max_capraise → assign new level_number = 1000+offset.
     - No match AND E_AS ≤ E_max_capraise → drop (likely duplicate config CMFGEN already has).
  2. Keep AS lines only if UPPER endpoint is an extension level (avoids
     double-counting CMFGEN bb data that is already in capraise).
  3. Append new levels to levels.csv, new lines to line_list.csv. Re-sort
     and re-issue global line_id.
  4. Extend macro_atom_references.csv + macro_atom_data.csv to include the
     new extension levels and the kept AS lines. Each new line contributes:
       +1 internal_up row in the lower level's block,
       +1 internal_down + +1 emission row in the upper level's block.
     Without this extension, finalize_cmfgen_ref_npy.py writes
     line2macro_level_upper = -1 for ~26% of lines (Cr/Mn/Ni III orphans) and
     LUMINA falls back to forced resonance scatter (lumina_cuda.cu:1484),
     killing the UV→optical cascade and washing out P-Cygni troughs.

Inputs:
  data/tardis_reference_v3_femerge_capraise/{levels,line_list,
       macro_atom_data,macro_atom_references}.csv (+ all others)
  data/as_planE_iron3/{cr3,mn3,ni3}_{levels,lines}.csv
Output:
  data/tardis_reference_v3_femerge_capraise_asE/   (mirror of source + merged lev/lines/macro)
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

_SUF = os.environ.get("CMFGEN_OUT_SUFFIX", "")
ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
SRC = ROOT / f"data/tardis_reference_v3_femerge{_SUF}"
AS  = ROOT / "data/as_planE_iron3"
OUT = ROOT / f"data/tardis_reference_v3_femerge{_SUF}_asE" if not _SUF.endswith("_asE") \
      else ROOT / f"data/tardis_reference_v3_femerge{_SUF}"

# AS ion → (Z, ion, tag)
AS_IONS = [(24, 2, "cr3"), (25, 2, "mn3"), (28, 2, "ni3")]

E_MATCH_TOL_EV = 0.5  # cross-walk (g,E) match tolerance


def crosswalk_one_ion(cm_lev: pd.DataFrame, as_lev: pd.DataFrame,
                       N_cap: int) -> tuple[dict[int, int], list[dict]]:
    """Return (as_idx -> capraise level_number) map and list of extension level rows.

    cm_lev: capraise levels.csv subset for this (Z,ion).
    as_lev: AS levels.csv for this ion.
    N_cap : total levels in cm_lev (= max capraise level_number + 1).
    """
    E_max_cap = float(cm_lev.energy_eV.max())
    cm_by_g: dict[int, pd.DataFrame] = {g: g_df.sort_values("energy_eV")
                                         for g, g_df in cm_lev.groupby("g")}
    crosswalk: dict[int, int] = {}
    ext_rows: list[dict] = []
    next_ln = N_cap
    n_match = 0
    n_drop = 0
    for as_idx, row in as_lev.iterrows():
        g_as = int(row.g)
        e_as = float(row.energy_eV)
        match_idx = None
        if g_as in cm_by_g:
            cands = cm_by_g[g_as]
            dE = (cands.energy_eV - e_as).abs()
            best_pos = dE.values.argmin()
            if dE.values[best_pos] <= E_MATCH_TOL_EV:
                match_idx = int(cands.iloc[best_pos].level_number)
        if match_idx is not None:
            crosswalk[int(row.level_number)] = match_idx
            n_match += 1
        else:
            if e_as > E_max_cap:
                crosswalk[int(row.level_number)] = next_ln
                ext_rows.append(dict(
                    atomic_number=int(row.atomic_number),
                    ion_number=int(row.ion_number),
                    level_number=next_ln,
                    energy_eV=e_as,
                    g=g_as,
                    metastable=0,
                ))
                next_ln += 1
            else:
                n_drop += 1
    return crosswalk, ext_rows, n_match, n_drop


MACRO_COLS = [
    "atomic_number", "ion_number",
    "source_level_number", "destination_level_number",
    "transition_type", "transition_probability",
    "transition_line_id", "lines_idx",
    "destination_level_idx", "source_level_idx",
]


def expand_macro_atom_layer(macro_df_old, refs_df_old,
                             ext_level_keys, kept_line_keys, all_ll_sorted):
    """Extend macro_atom layer for AS planE extension levels.

    Each unique (Z, ion, lo, hi) kept AS line contributes 3 macro rows:
        +1 internal_up   (transition_type=1)  in lower-level block
        +1 internal_down (transition_type=-1) in upper-level block
        +1 emission      (transition_type=0)  in upper-level block

    Each extension level gets a new macro_atom_references entry.
    block_references and references_idx are re-issued for the unified refs
    after sorting by (atomic_number, ion_number, source_level_number).

    Returns (macro_new, refs_new). Sanity-asserts internally:
        sum(count_total) == len(macro_new)
        block_references monotonic non-decreasing
        count_total == 2*count_down + count_up
    """
    if "Unnamed: 0" in macro_df_old.columns:
        macro_clean = macro_df_old.drop(columns=["Unnamed: 0"])[MACRO_COLS]
    else:
        macro_clean = macro_df_old[MACRO_COLS].copy()

    # ---- 1. Unified refs (existing + extension), sorted on (Z, ion, level) ----
    ext_rows_df = pd.DataFrame({
        "atomic_number":       [k[0] for k in ext_level_keys],
        "ion_number":          [k[1] for k in ext_level_keys],
        "source_level_number": [k[2] for k in ext_level_keys],
        "count_down": 0, "count_up": 0, "count_total": 0,
        "block_references": 0, "references_idx": 0,
    })
    refs_all = pd.concat([refs_df_old, ext_rows_df], ignore_index=True)
    refs_all = refs_all.sort_values(
        ["atomic_number", "ion_number", "source_level_number"]).reset_index(drop=True)

    key_to_refidx = {
        (int(r.atomic_number), int(r.ion_number), int(r.source_level_number)): i
        for i, r in refs_all.iterrows()
    }
    old_block = {
        (int(r.atomic_number), int(r.ion_number), int(r.source_level_number)):
            (int(r.block_references), int(r.count_down), int(r.count_total),
             int(r.count_up))
        for r in refs_df_old.itertuples(index=False)
    }

    # ---- 2. Final array index in sorted line_list for each kept line key ----
    ll_idx = {}
    for i, r in enumerate(all_ll_sorted.itertuples(index=False)):
        k = (int(r.atomic_number), int(r.ion_number),
             int(r.level_number_lower), int(r.level_number_upper))
        if k not in ll_idx:
            ll_idx[k] = (i, int(r.line_id))

    # ---- 3. Build per-ref add lists ----
    add_down = {i: [] for i in range(len(refs_all))}
    add_up   = {i: [] for i in range(len(refs_all))}
    n_skipped = 0
    for k in kept_line_keys:
        if k not in ll_idx:
            n_skipped += 1
            continue
        Z, ion, lo, hi = k
        lidx, lid = ll_idx[k]
        ref_lo = key_to_refidx[(Z, ion, lo)]
        ref_hi = key_to_refidx[(Z, ion, hi)]

        add_up[ref_lo].append(dict(
            atomic_number=Z, ion_number=ion,
            source_level_number=lo, destination_level_number=hi,
            transition_type=1, transition_probability=0.0,
            transition_line_id=lid, lines_idx=lidx,
            destination_level_idx=ref_hi, source_level_idx=ref_lo,
        ))
        add_down[ref_hi].append(dict(
            atomic_number=Z, ion_number=ion,
            source_level_number=hi, destination_level_number=lo,
            transition_type=-1, transition_probability=0.0,
            transition_line_id=lid, lines_idx=lidx,
            destination_level_idx=ref_lo, source_level_idx=ref_hi,
        ))
        add_down[ref_hi].append(dict(
            atomic_number=Z, ion_number=ion,
            source_level_number=hi, destination_level_number=lo,
            transition_type=0, transition_probability=0.0,
            transition_line_id=lid, lines_idx=lidx,
            destination_level_idx=ref_lo, source_level_idx=ref_hi,
        ))
    if n_skipped:
        print(f"    WARN: {n_skipped} kept lines not found in sorted line_list")

    # ---- 4. Reconstruct macro_atom_data block by block ----
    parts = []
    new_block = np.empty(len(refs_all), dtype=np.int64)
    cum = 0
    for i in range(len(refs_all)):
        r = refs_all.iloc[i]
        key = (int(r.atomic_number), int(r.ion_number), int(r.source_level_number))
        new_block[i] = cum

        if key in old_block:
            bstart, ndown_old, ntot_old, _cu = old_block[key]
            ndown_rows = 2 * ndown_old
            ds = macro_clean.iloc[bstart:bstart + ndown_rows]
            us = macro_clean.iloc[bstart + ndown_rows:bstart + ntot_old]
        else:
            ds = pd.DataFrame(columns=MACRO_COLS)
            us = pd.DataFrame(columns=MACRO_COLS)

        parts.append(ds)
        cum += len(ds)
        if add_down[i]:
            parts.append(pd.DataFrame(add_down[i])[MACRO_COLS])
            cum += len(add_down[i])
        parts.append(us)
        cum += len(us)
        if add_up[i]:
            parts.append(pd.DataFrame(add_up[i])[MACRO_COLS])
            cum += len(add_up[i])

    macro_new = pd.concat(parts, ignore_index=True)

    # ---- 5. Update refs counts + block_references ----
    add_up_n = np.array([len(add_up[i])         for i in range(len(refs_all))], dtype=np.int64)
    add_dn_n = np.array([len(add_down[i]) // 2  for i in range(len(refs_all))], dtype=np.int64)
    old_cu = np.zeros(len(refs_all), dtype=np.int64)
    old_cd = np.zeros(len(refs_all), dtype=np.int64)
    for i in range(len(refs_all)):
        r = refs_all.iloc[i]
        key = (int(r.atomic_number), int(r.ion_number), int(r.source_level_number))
        if key in old_block:
            _, cd_o, _, cu_o = old_block[key]
            old_cu[i] = cu_o
            old_cd[i] = cd_o
    refs_all["count_up"]    = old_cu + add_up_n
    refs_all["count_down"]  = old_cd + add_dn_n
    refs_all["count_total"] = 2 * refs_all["count_down"] + refs_all["count_up"]
    refs_all["block_references"] = new_block
    refs_all["references_idx"]   = new_block.copy()

    # ---- 6. Sanity checks ----
    sct = int(refs_all["count_total"].sum())
    assert sct == len(macro_new), \
        f"sum(count_total)={sct:,} != len(macro_new)={len(macro_new):,}"
    br = refs_all["block_references"].to_numpy()
    assert (np.diff(br) >= 0).all(), "block_references must be non-decreasing"
    inv = int((refs_all["count_total"]
               != 2 * refs_all["count_down"] + refs_all["count_up"]).sum())
    assert inv == 0, f"count_total invariant violated for {inv} rows"

    return macro_new, refs_all


def main():
    OUT.mkdir(exist_ok=True)

    # mirror everything from SRC first
    for f in SRC.iterdir():
        if f.is_dir():
            continue
        dst = OUT / f.name
        if dst.exists():
            dst.unlink()
        if f.is_symlink() or f.suffix in (".bin",):
            # symlink for blobs (sigma_bf, density, etc.) to save space
            dst.symlink_to(f.resolve())
        else:
            shutil.copy2(f, dst)

    cm_lev_all = pd.read_csv(SRC / "levels.csv")
    cm_ll_all = pd.read_csv(SRC / "line_list.csv")
    new_levels: list[pd.DataFrame] = []
    new_lines: list[pd.DataFrame] = []
    ext_level_keys: set[tuple[int, int, int]] = set()
    kept_line_keys: set[tuple[int, int, int, int]] = set()

    for (Z, ion, tag) in AS_IONS:
        cm_lev = cm_lev_all[(cm_lev_all.atomic_number == Z) &
                             (cm_lev_all.ion_number == ion)].copy()
        N_cap = int(cm_lev.level_number.max()) + 1
        as_lev = pd.read_csv(AS / f"{tag}_levels.csv")
        as_ll = pd.read_csv(AS / f"{tag}_lines.csv")

        crosswalk, ext_rows, n_match, n_drop = crosswalk_one_ion(cm_lev, as_lev, N_cap)
        n_ext = len(ext_rows)
        print(f"  Z={Z} ion={ion} ({tag}): AS levels={len(as_lev)}  match={n_match}  ext={n_ext}  drop={n_drop}")

        # Append extension levels
        if ext_rows:
            new_levels.append(pd.DataFrame(ext_rows))
            for er in ext_rows:
                ext_level_keys.add((Z, ion, int(er["level_number"])))

        # Filter AS lines: upper must be in extension space (level_number >= N_cap)
        if not as_ll.empty and n_ext > 0:
            up_map = as_ll.level_number_upper.map(crosswalk)
            lo_map = as_ll.level_number_lower.map(crosswalk)
            keep = (~up_map.isna()) & (~lo_map.isna()) & (up_map.astype("Int64") >= N_cap)
            kept = as_ll[keep].copy()
            kept["level_number_lower"] = lo_map[keep].astype(int).values
            kept["level_number_upper"] = up_map[keep].astype(int).values
            # capraise stores f_ul as positive — AS uses negative emission convention
            kept["f_ul"] = kept["f_ul"].abs()
            # line_id will be re-issued globally below
            kept["line_id"] = -1
            uv = ((kept.wavelength >= 1500) & (kept.wavelength <= 3000)).sum()
            print(f"      new lines: {len(kept):,}  UV[1500-3000]={uv:,}")
            for lo, hi in zip(kept.level_number_lower.astype(int).values,
                              kept.level_number_upper.astype(int).values):
                kept_line_keys.add((Z, ion, int(lo), int(hi)))
            new_lines.append(kept[cm_ll_all.columns.tolist()])

    # ------ write merged levels.csv ------
    if new_levels:
        all_lev = pd.concat([cm_lev_all] + new_levels, ignore_index=True)
    else:
        all_lev = cm_lev_all
    all_lev = all_lev.sort_values(["atomic_number", "ion_number", "level_number"]).reset_index(drop=True)
    all_lev.to_csv(OUT / "levels.csv", index=False)
    print(f"  total levels: {len(all_lev):,} (was {len(cm_lev_all):,}, +{len(all_lev)-len(cm_lev_all):,})")

    # ------ write merged line_list.csv ------
    if new_lines:
        all_ll = pd.concat([cm_ll_all] + new_lines, ignore_index=True)
    else:
        all_ll = cm_ll_all
    # Bug #4: AS planE cross-walk maps multiple (J,L,S) fine-structure
    # components onto the same capraise (g,E) level, producing duplicate
    # (Z,ion,lo,up) rows.  Same physics as V3 Mg I: two decay channels
    # collapsed onto one macro-transition.  Sum rate-like columns at fixed
    # (ν, g_l, g_u); Einstein relation preserved.  NOT an approximation.
    key_cols_ll = ["atomic_number", "ion_number",
                   "level_number_lower", "level_number_upper"]
    n_as_dups = int(all_ll.duplicated(subset=key_cols_ll, keep=False).sum())
    if n_as_dups > 0:
        sum_cols = ["A_ul", "f_lu", "f_ul", "B_lu", "B_ul"]
        first_cols = [c for c in all_ll.columns if c not in key_cols_ll + sum_cols]
        agg = {c: "first" for c in first_cols}
        for c in sum_cols:
            agg[c] = "sum"
        col_order = list(all_ll.columns)
        n_before_ll = len(all_ll)
        dup_per_z = (all_ll[all_ll.duplicated(subset=key_cols_ll, keep=False)]
                     .groupby(["atomic_number","ion_number"]).size().to_dict())
        all_ll = all_ll.groupby(key_cols_ll, as_index=False, sort=False).agg(agg)
        all_ll = all_ll[col_order]
        print(f"  asE line_list dedup: {n_before_ll:,} → {len(all_ll):,} "
              f"(collapsed {n_as_dups} dup rows; per-(Z,ion) {dup_per_z})")
    all_ll = all_ll.sort_values(["atomic_number", "ion_number", "wavelength_cm"]).reset_index(drop=True)
    all_ll["line_id"] = np.arange(len(all_ll), dtype=np.int64)
    all_ll.to_csv(OUT / "line_list.csv", index=False)
    print(f"  total lines:  {len(all_ll):,} (was {len(cm_ll_all):,}, +{len(all_ll)-len(cm_ll_all):,})")
    uv_old = ((cm_ll_all.wavelength >= 1500) & (cm_ll_all.wavelength <= 3000)).sum()
    uv_new = ((all_ll.wavelength >= 1500) & (all_ll.wavelength <= 3000)).sum()
    print(f"  UV[1500-3000] lines: {uv_new:,} (was {uv_old:,}, ×{uv_new/max(uv_old,1):.1f})")

    # ------ extend macro_atom layer for extension levels + kept AS lines ------
    if kept_line_keys:
        print(f"\n  expanding macro_atom layer:")
        print(f"    extension levels: {len(ext_level_keys):,}")
        print(f"    kept AS line keys (deduped on Z,ion,lo,hi): {len(kept_line_keys):,}")
        macro_df_old = pd.read_csv(SRC / "macro_atom_data.csv")
        refs_df_old  = pd.read_csv(SRC / "macro_atom_references.csv")
        print(f"    SRC macro_atom_data rows:        {len(macro_df_old):,}")
        print(f"    SRC macro_atom_references rows:  {len(refs_df_old):,}")

        macro_new, refs_new = expand_macro_atom_layer(
            macro_df_old, refs_df_old, ext_level_keys, kept_line_keys, all_ll)

        print(f"    NEW macro_atom_data rows:        {len(macro_new):,} "
              f"(+{len(macro_new)-len(macro_df_old):,})")
        print(f"    NEW macro_atom_references rows:  {len(refs_new):,} "
              f"(+{len(refs_new)-len(refs_df_old):,})")

        # Match original macro_atom_data.csv layout (unnamed leading index col)
        macro_new.to_csv(OUT / "macro_atom_data.csv", index=True, index_label="")
        refs_new.to_csv(OUT / "macro_atom_references.csv", index=False)
        print(f"  wrote expanded macro_atom_data.csv + macro_atom_references.csv")

    print(f"  output: {OUT}")


if __name__ == "__main__":
    main()
