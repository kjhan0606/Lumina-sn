#!/usr/bin/env python3
"""
#219f surgical merge: replace V3 iron-peak III (Z,ion) atomic data with CMFGEN.

Target ions: (22,2) Ti III, (24,2) Cr III, (25,2) Mn III, (26,2) Fe III,
             (27,2) Co III, (28,2) Ni III
(Sc III and V III stay as V3 — no CMFGEN data available.)

Rationale (#219e): V3 has many bb lines for iron-peak III but the highest
energy level is bb-isolated (no transitions in/out), causing rate-matrix
singularity at conservation row N-1. CMFGEN's truncated level table
(max 800-1500 levels) avoids that singularity if its top level happens
to have bb edges.

Inputs:
  V3      = data/tardis_reference_zeta_clean
  CMFGEN  = data/tardis_reference_cmfgen
Output:
  data/tardis_reference_v3_femerge

For 4 atomic tables (levels, line_list, macro_atom_data, macro_atom_references):
  - drop rows where (Z,ion) is in IRON_PEAK_III from V3
  - append rows from CMFGEN for the same (Z,ion)
  - re-sort by (Z, ion, [level/wavelength])
  - re-issue global indices (line_id, lines_idx, source/destination_level_idx)
  - macro_atom_data.lines_idx remapped via old_lid -> new_lid

After: run scripts/finalize_cmfgen_ref_npy.py to rebuild .npy files.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import os
_SUF = os.environ.get('CMFGEN_OUT_SUFFIX', '')
ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
V3 = ROOT / "data/tardis_reference_zeta_clean"
CMFGEN = ROOT / f"data/tardis_reference_cmfgen{_SUF}"
OUT = ROOT / f"data/tardis_reference_v3_femerge{_SUF}"

IRON_PEAK_III = {(22, 2), (24, 2), (25, 2), (26, 2), (27, 2), (28, 2)}


def merge_filter(v3, cm, ions):
    """Drop rows from v3 where (Z,ion) in ions, append cm rows for same ions."""
    mask_v3_drop = list(zip(v3["atomic_number"], v3["ion_number"]))
    mask_v3_drop = [t in ions for t in mask_v3_drop]
    mask_cm_keep = list(zip(cm["atomic_number"], cm["ion_number"]))
    mask_cm_keep = [t in ions for t in mask_cm_keep]
    kept_v3 = v3[~np.array(mask_v3_drop)].copy()
    added_cm = cm[np.array(mask_cm_keep)].copy()
    return pd.concat([kept_v3, added_cm], ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    print(f"output dir: {OUT}")

    # ---------- 1. levels.csv ----------
    v3_lev = pd.read_csv(V3 / "levels.csv")
    cm_lev = pd.read_csv(CMFGEN / "levels.csv")
    merged_lev = merge_filter(v3_lev, cm_lev, IRON_PEAK_III)
    merged_lev = merged_lev.sort_values(
        ["atomic_number", "ion_number", "level_number"]
    ).reset_index(drop=True)
    merged_lev.to_csv(OUT / "levels.csv", index=False)
    print(f"  levels: V3 {len(v3_lev)} + CMFGEN sub {sum(mask in IRON_PEAK_III for mask in zip(cm_lev['atomic_number'], cm_lev['ion_number']))} = {len(merged_lev)}")

    # ---------- 2. line_list.csv ----------
    v3_ll = pd.read_csv(V3 / "line_list.csv")
    # Bug #4 upstream fix: V3 carsus build collapsed fine-structure components
    # of Mg I (244 rows), Sc IV (6), Al I (2) onto the same (Z,ion,lo,up)
    # without summing rates.  Two rows share λ, ν, g_l, g_u but list different
    # A_ul / f_lu values.  Physically two decay channels through the same
    # macro-transition compete for the same upper-level population:
    #   A_total = ΣA_i, f_lu_total = Σf_i, B_lu_total = ΣB_lu_i.
    # The Einstein relation A = (8πhν³/c²)(g_l/g_u)B_lu is linear in A/f/B,
    # so summing all rate-like columns at fixed (ν, g_l, g_u) preserves the
    # relation exactly — NOT an approximation.  Keep first line_id as the
    # canonical id; remap all orphan line_ids → canonical in v3_ma below.
    key_cols_ll = ["atomic_number", "ion_number",
                   "level_number_lower", "level_number_upper"]
    v3_dup_remap: dict[int, int] = {}
    n_v3_dups = int(v3_ll.duplicated(subset=key_cols_ll, keep=False).sum())
    if n_v3_dups > 0:
        # Stable order so 'first' is deterministic.
        v3_ll_sorted = v3_ll.sort_values(key_cols_ll + ["line_id"]).reset_index(drop=True)
        for _, grp in v3_ll_sorted.groupby(key_cols_ll, sort=False):
            if len(grp) > 1:
                canonical = int(grp.iloc[0]["line_id"])
                for orphan in grp.iloc[1:]["line_id"].astype(int):
                    v3_dup_remap[int(orphan)] = canonical
        sum_cols = ["A_ul", "f_lu", "f_ul", "B_lu", "B_ul"]
        first_cols = [c for c in v3_ll.columns if c not in key_cols_ll + sum_cols]
        agg = {c: "first" for c in first_cols}
        for c in sum_cols:
            agg[c] = "sum"
        col_order = list(v3_ll.columns)
        n_before = len(v3_ll)
        v3_ll = v3_ll_sorted.groupby(key_cols_ll, as_index=False, sort=False).agg(agg)
        v3_ll = v3_ll[col_order]
        zion_counts = (v3_ll_sorted[v3_ll_sorted.duplicated(subset=key_cols_ll, keep=False)]
                       .groupby(["atomic_number", "ion_number"]).size().to_dict())
        print(f"  V3 dedup: {n_before:,} → {len(v3_ll):,} rows "
              f"(collapsed {n_v3_dups} dup rows; remap {len(v3_dup_remap)} "
              f"orphan→canonical; per-(Z,ion) {zion_counts})")
    cm_ll = pd.read_csv(CMFGEN / "line_list.csv")
    merged_ll = merge_filter(v3_ll, cm_ll, IRON_PEAK_III)
    # Stable global ordering: by (Z, ion, wavelength_cm)
    merged_ll = merged_ll.sort_values(
        ["atomic_number", "ion_number", "wavelength_cm"]
    ).reset_index(drop=True)

    # Build old_lid -> new_lid map BEFORE re-issuing line_id
    # We track origin: v3-kept rows keep their old line_id, cm-added rows keep theirs
    # but they sit in disjoint id spaces. After sort, we re-issue.
    # For macro_atom_data we need: (Z, ion, old_line_id_in_source) -> new_line_id
    # Build the lookup from merged_ll row->new_lid, keyed by (Z, ion, old_lid, source_tag)

    # To do this cleanly, do a second pass:
    v3_ll_kept = v3_ll[~v3_ll[["atomic_number", "ion_number"]]
                       .apply(tuple, axis=1).isin(IRON_PEAK_III)].copy()
    v3_ll_kept["src_tag"] = "v3"
    cm_ll_added = cm_ll[cm_ll[["atomic_number", "ion_number"]]
                        .apply(tuple, axis=1).isin(IRON_PEAK_III)].copy()
    cm_ll_added["src_tag"] = "cm"
    merged_ll2 = pd.concat([v3_ll_kept, cm_ll_added], ignore_index=True)
    merged_ll2 = merged_ll2.sort_values(
        ["atomic_number", "ion_number", "wavelength_cm"]
    ).reset_index(drop=True)
    # Save old (src, line_id) for lookup
    old_lid_to_new = {}
    for new_lid, row in enumerate(merged_ll2.itertuples(index=False)):
        old_lid_to_new[(row.src_tag, row.line_id)] = new_lid
    merged_ll2["line_id"] = np.arange(len(merged_ll2), dtype=np.int64)
    merged_ll2.drop(columns=["src_tag"]).to_csv(OUT / "line_list.csv", index=False)
    print(f"  line_list: V3 {len(v3_ll)} → kept {len(v3_ll_kept)}; "
          f"CMFGEN added {len(cm_ll_added)}; total {len(merged_ll2)}")

    # ---------- 3. macro_atom_data.csv ----------
    # First col is unnamed index — read with index_col=0
    v3_ma = pd.read_csv(V3 / "macro_atom_data.csv", index_col=0)
    cm_ma = pd.read_csv(CMFGEN / "macro_atom_data.csv", index_col=0)
    # Bug #4 propagation: orphan v3 line_ids that were collapsed in v3_ll dedup
    # must remap to their canonical line_id here, then sum transition_probability
    # across the now-duplicate (src,dst,type,line_id) macro rows.  Without this
    # step the orphan macro rows hold line_ids that no longer exist in
    # merged_ll, old_lid_to_new returns -1, and emission events lose their
    # photon hand-off.  Summing transition_probability is the exact macro-atom
    # response when two decay channels collapse onto the same line.
    if v3_dup_remap:
        n_remap_lid = int(v3_ma["transition_line_id"].isin(v3_dup_remap).sum())
        n_remap_lix = int(v3_ma["lines_idx"].isin(v3_dup_remap).sum())
        v3_ma["transition_line_id"] = v3_ma["transition_line_id"].map(
            lambda x: v3_dup_remap.get(int(x), int(x)) if pd.notna(x) and x >= 0 else x)
        v3_ma["lines_idx"] = v3_ma["lines_idx"].map(
            lambda x: v3_dup_remap.get(int(x), int(x)) if pd.notna(x) and x >= 0 else x)
        ma_key = ["atomic_number", "ion_number",
                  "source_level_number", "destination_level_number",
                  "transition_type", "transition_line_id"]
        n_ma_dups = int(v3_ma.duplicated(subset=ma_key, keep=False).sum())
        if n_ma_dups > 0:
            agg_ma = {c: "first" for c in v3_ma.columns if c not in ma_key + ["transition_probability"]}
            agg_ma["transition_probability"] = "sum"
            n_before_ma = len(v3_ma)
            v3_ma = v3_ma.groupby(ma_key, as_index=False, sort=False).agg(agg_ma)
            print(f"  V3 macro_atom dedup: line_id remap {n_remap_lid}, "
                  f"lines_idx remap {n_remap_lix}; collapsed {n_ma_dups} dup rows "
                  f"({n_before_ma:,} → {len(v3_ma):,}; Σtransition_probability)")
    v3_ma_kept = v3_ma[~v3_ma[["atomic_number", "ion_number"]]
                       .apply(tuple, axis=1).isin(IRON_PEAK_III)].copy()
    cm_ma_added = cm_ma[cm_ma[["atomic_number", "ion_number"]]
                        .apply(tuple, axis=1).isin(IRON_PEAK_III)].copy()
    # Remap transition_line_id and lines_idx using old_lid_to_new
    # transition_line_id == lines_idx == old line_list line_id (in source)
    def remap_lid(row, src_tag):
        old = row
        if pd.isna(old) or old < 0:
            return old
        return old_lid_to_new.get((src_tag, int(old)), -1)

    v3_ma_kept["transition_line_id"] = v3_ma_kept["transition_line_id"].apply(
        lambda x: remap_lid(x, "v3"))
    v3_ma_kept["lines_idx"] = v3_ma_kept["lines_idx"].apply(
        lambda x: remap_lid(x, "v3"))
    cm_ma_added["transition_line_id"] = cm_ma_added["transition_line_id"].apply(
        lambda x: remap_lid(x, "cm"))
    cm_ma_added["lines_idx"] = cm_ma_added["lines_idx"].apply(
        lambda x: remap_lid(x, "cm"))

    merged_ma = pd.concat([v3_ma_kept, cm_ma_added], ignore_index=True)

    # Recompute source_level_idx, destination_level_idx from merged_lev
    # Build (Z, ion, level_number) -> global level index in merged_lev
    lev_idx = {}
    for gi, row in enumerate(merged_lev.itertuples(index=False)):
        lev_idx[(row.atomic_number, row.ion_number, row.level_number)] = gi

    def lookup_lev(z, ion, lvl):
        return lev_idx.get((int(z), int(ion), int(lvl)), -1)

    merged_ma["source_level_idx"] = [
        lookup_lev(z, i, l) for z, i, l in zip(
            merged_ma["atomic_number"], merged_ma["ion_number"],
            merged_ma["source_level_number"])
    ]
    merged_ma["destination_level_idx"] = [
        lookup_lev(z, i, l) for z, i, l in zip(
            merged_ma["atomic_number"], merged_ma["ion_number"],
            merged_ma["destination_level_number"])
    ]
    # Sort by (Z, ion, source_level, dest_level, type) for stable order
    merged_ma = merged_ma.sort_values(
        ["atomic_number", "ion_number", "source_level_number",
         "destination_level_number", "transition_type"]
    ).reset_index(drop=True)
    merged_ma.to_csv(OUT / "macro_atom_data.csv")
    print(f"  macro_atom_data: V3 {len(v3_ma)} → kept {len(v3_ma_kept)}; "
          f"CMFGEN added {len(cm_ma_added)}; total {len(merged_ma)}")

    # ---------- 4. macro_atom_references.csv ----------
    v3_mr = pd.read_csv(V3 / "macro_atom_references.csv")
    cm_mr = pd.read_csv(CMFGEN / "macro_atom_references.csv")
    merged_mr = merge_filter(v3_mr, cm_mr, IRON_PEAK_III)
    merged_mr = merged_mr.sort_values(
        ["atomic_number", "ion_number", "source_level_number"]
    ).reset_index(drop=True)
    # Bug #4 propagation: V3 dedup collapsed macro_atom rows, so V3-side
    # count_down / count_up / count_total in merged_mr are stale.  Recount
    # from merged_ma so the invariant count_total == 2*count_down + count_up
    # holds and block_references stay aligned with the (smaller) merged_ma.
    if v3_dup_remap:
        # transition_type semantics (per expand_atomic_data_cmfgen.write_macro_atom):
        #   -1 = emission (line-level downward), 0 = internal-down,
        #    1 = internal-up.  Each line contributes 2 down rows + 1 up row.
        #   count_down (line-level) = (#emission rows) = (#internal-down rows).
        #   count_up   (line-level) = (#internal-up rows).
        ma_grp = merged_ma.groupby(
            ["atomic_number", "ion_number", "source_level_number"], sort=False)
        cd_per = ma_grp.apply(lambda g: int((g.transition_type == -1).sum()),
                              include_groups=False)
        cu_per = ma_grp.apply(lambda g: int((g.transition_type ==  1).sum()),
                              include_groups=False)
        ct_per = ma_grp.size()
        cnt_df = pd.DataFrame({"count_down": cd_per, "count_up": cu_per,
                               "count_total": ct_per}).reset_index()
        merged_mr = merged_mr.drop(columns=["count_down","count_up","count_total"]) \
            .merge(cnt_df, on=["atomic_number","ion_number","source_level_number"],
                   how="left").fillna({"count_down":0,"count_up":0,"count_total":0})
        for c in ("count_down", "count_up", "count_total"):
            merged_mr[c] = merged_mr[c].astype(np.int64)
        # Sanity: 2*cd + cu == ct (the asE merger relies on this).
        bad = int(((2*merged_mr.count_down + merged_mr.count_up) != merged_mr.count_total).sum())
        assert bad == 0, f"recount: {bad} mr rows fail 2*cd+cu==ct invariant"
        # And the cumulative count_total must equal len(merged_ma) so block_refs is exact.
        assert int(merged_mr.count_total.sum()) == len(merged_ma), \
            f"Σcount_total {int(merged_mr.count_total.sum())} != len(merged_ma) {len(merged_ma)}"
        print(f"  V3 mr recount: Σcount_total={int(merged_mr.count_total.sum())} "
              f"(matches len(merged_ma)={len(merged_ma)})")
    # block_references and references_idx need recomputing from cumulative counts
    cum = 0
    refs_idx = []
    block_refs = []
    for cnt in merged_mr["count_total"].values:
        block_refs.append(cum)
        refs_idx.append(cum)
        cum += int(cnt)
    merged_mr["block_references"] = block_refs
    merged_mr["references_idx"] = refs_idx
    merged_mr.to_csv(OUT / "macro_atom_references.csv", index=False)
    print(f"  macro_atom_references: V3 {len(v3_mr)} + CMFGEN sub = {len(merged_mr)}")

    # ---------- 5. Copy / symlink other reference files ----------
    # zeta_data.npy + zeta_patch_diag.csv: local to V3 (clean ζ)
    for fname in ["zeta_data.npy", "zeta_patch_diag.csv"]:
        if (V3 / fname).exists():
            shutil.copy2(V3 / fname, OUT / fname)
            print(f"  copied {fname}")

    # Symlink the other lookup tables from V3 (atom_masses, ionization_energies,
    # zeta_ions, zeta_temps). The SLURM script overrides geometry / density /
    # abundances / plasma_state / config / electron_densities.
    for fname in ["atom_masses.csv", "ionization_energies.csv",
                  "zeta_ions.csv", "zeta_temps.csv"]:
        src = (V3 / fname).resolve()
        dst = OUT / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        print(f"  symlink {fname} -> {src}")

    # Also symlink the per-cell defaults (the SLURM script removes & rewrites these
    # anyway, but having them present means the dir is self-sufficient for ad-hoc use)
    for fname in ["abundances.csv", "config.json", "density.csv",
                  "electron_densities.csv", "geometry.csv",
                  "plasma_state.csv", "mc_estimators.csv",
                  "tau_sobolev_stats.csv", "abundances.npy",
                  "ion_number_density.npy", "j_blues.npy"]:
        src_p = V3 / fname
        if not src_p.exists():
            continue
        src = src_p.resolve()
        dst = OUT / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    print()
    print("=== merge complete ===")
    print(f"Next: regenerate npy with scripts/finalize_cmfgen_ref_npy.py")
    print(f"  python3 scripts/finalize_cmfgen_ref_npy.py {OUT}")


if __name__ == "__main__":
    main()
