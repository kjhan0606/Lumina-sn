#!/usr/bin/env python3
"""
Task #47: Merge missing Fe II M1/E2 forbidden lines (Sigut & Pradhan 1998
Lyα-pumped UV-fluorescence channel) from CMFGEN-only reference into the
carsus runtime reference.

Selection:
    Z=26, ion_number=1 (Fe II)
    lower-level energy < 1.0 eV (a^6D ground multiplet)
    upper-level energy in [5.0, 9.0] eV
    A_ul < 1e3 s^-1 (M1/E2 forbidden)
    (lower, upper) pair NOT already in carsus line_list

Outputs (overwrites in /data/tardis_reference/):
    levels.csv (unchanged unless new levels needed; backed up)
    line_list.csv             (+N rows)
    line2macro_level_upper.npy (+N entries)
    macro_atom_data.csv        (+3*N rows, inserted into level blocks)
    macro_atom_references.csv  (count_up/count_down/count_total/block_references updated)

Backup: data/tardis_reference.bak_pre_m1e2_<timestamp>/  (full directory copy)
"""

import os
import sys
import shutil
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants & paths
# ----------------------------------------------------------------------
ROOT       = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
REF_DIR    = ROOT / "data" / "tardis_reference"
CMFGEN_DIR = ROOT / "data" / "tardis_reference_cmfgen"

C_CGS  = 2.99792458e10        # cm/s
H_CGS  = 6.62607015e-27       # erg s

Z_FE   = 26
ION_FE2 = 1

E_GROUND_MAX = 1.0   # eV
E_UPPER_LO   = 5.0
E_UPPER_HI   = 9.0
A_UL_MAX     = 1.0e3

ENERGY_TOL   = 1e-3   # eV  (joining levels by energy)


def log(msg):
    print(f"[merge_fe2_m1e2] {msg}", flush=True)


# ----------------------------------------------------------------------
# Step A: backup
# ----------------------------------------------------------------------
def backup_ref_dir():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = ROOT / "data" / f"tardis_reference.bak_pre_m1e2_{stamp}"
    log(f"Backup: copying {REF_DIR} -> {dst}")
    shutil.copytree(REF_DIR, dst)
    return dst


# ----------------------------------------------------------------------
# Step B: level alignment for Fe II (carsus vs CMFGEN)
# ----------------------------------------------------------------------
def build_fe2_level_map(carsus_levels, cmfgen_levels):
    """Return dict: cmfgen_lev_num -> carsus_lev_num, for Fe II only.

    Match by (energy_eV rounded to 3 dp, g). Print side-by-side for
    energies < ~9 eV.
    """
    fe2_c = carsus_levels[(carsus_levels.atomic_number == Z_FE) &
                          (carsus_levels.ion_number == ION_FE2)].copy()
    fe2_m = cmfgen_levels[(cmfgen_levels.atomic_number == Z_FE) &
                          (cmfgen_levels.ion_number == ION_FE2)].copy()
    log(f"Fe II levels: carsus={len(fe2_c)}, cmfgen={len(fe2_m)}")

    # Round-key match
    fe2_c["key"] = list(zip(fe2_c.energy_eV.round(3), fe2_c.g.astype(int)))
    fe2_m["key"] = list(zip(fe2_m.energy_eV.round(3), fe2_m.g.astype(int)))

    # If keys collide (degeneracy at same E,g) we map cmfgen->lowest carsus level_number.
    c_lookup = (fe2_c.sort_values("level_number")
                     .drop_duplicates("key", keep="first")
                     .set_index("key")["level_number"]
                     .to_dict())

    mapping = {}
    unmatched = []
    for _, row in fe2_m.iterrows():
        if row["key"] in c_lookup:
            mapping[int(row.level_number)] = int(c_lookup[row["key"]])
        else:
            unmatched.append(int(row.level_number))

    log(f"Level map: matched={len(mapping)}/{len(fe2_m)}  unmatched={len(unmatched)}")

    # Print side-by-side for first ~25 levels
    log("Side-by-side Fe II low-E levels (cmfgen_lev, carsus_lev, E_eV, g):")
    head = fe2_m.sort_values("level_number").head(25)
    for _, r in head.iterrows():
        cl = mapping.get(int(r.level_number), "MISS")
        log(f"  cmfgen lev={int(r.level_number):4d} -> carsus lev={cl}"
            f"   E={r.energy_eV:.4f} eV  g={int(r.g)}")
    return mapping, fe2_c, fe2_m


# ----------------------------------------------------------------------
# Step C: filter missing forbidden lines from CMFGEN
# ----------------------------------------------------------------------
def filter_missing_lines(cmfgen_lines, cmfgen_levels, carsus_lines, lev_map):
    fe_lines_m = cmfgen_lines[(cmfgen_lines.atomic_number == Z_FE) &
                              (cmfgen_lines.ion_number == ION_FE2)].copy()
    log(f"CMFGEN Fe II lines total: {len(fe_lines_m)}")

    # Energy lookup for cmfgen Fe II levels
    fe2_m = cmfgen_levels[(cmfgen_levels.atomic_number == Z_FE) &
                          (cmfgen_levels.ion_number == ION_FE2)]
    e_by_lev = fe2_m.set_index("level_number")["energy_eV"].to_dict()

    fe_lines_m["E_low"]  = fe_lines_m["level_number_lower"].map(e_by_lev)
    fe_lines_m["E_high"] = fe_lines_m["level_number_upper"].map(e_by_lev)

    cand = fe_lines_m[
        (fe_lines_m.E_low  < E_GROUND_MAX) &
        (fe_lines_m.E_high >= E_UPPER_LO) & (fe_lines_m.E_high <= E_UPPER_HI) &
        (fe_lines_m.A_ul   < A_UL_MAX)
    ].copy()
    log(f"After E/A_ul filter (ground<{E_GROUND_MAX}, upper [{E_UPPER_LO},"
        f"{E_UPPER_HI}], A_ul<{A_UL_MAX}): {len(cand)}")

    # Translate level numbers
    cand["lo_carsus"] = cand["level_number_lower"].map(lev_map)
    cand["hi_carsus"] = cand["level_number_upper"].map(lev_map)
    n_drop = cand[cand.lo_carsus.isna() | cand.hi_carsus.isna()].shape[0]
    if n_drop:
        log(f"  dropping {n_drop} candidates with unmapped level_number")
    cand = cand.dropna(subset=["lo_carsus", "hi_carsus"]).copy()
    cand["lo_carsus"] = cand["lo_carsus"].astype(int)
    cand["hi_carsus"] = cand["hi_carsus"].astype(int)

    # Drop duplicates already in carsus
    fe_c = carsus_lines[(carsus_lines.atomic_number == Z_FE) &
                        (carsus_lines.ion_number == ION_FE2)]
    existing = set(zip(fe_c.level_number_lower.astype(int),
                       fe_c.level_number_upper.astype(int)))
    log(f"  existing carsus Fe II (lo,hi) pairs: {len(existing)}")
    cand["pair"] = list(zip(cand.lo_carsus, cand.hi_carsus))
    before = len(cand)
    cand = cand[~cand["pair"].isin(existing)].copy()
    log(f"  after dedup: {len(cand)} (removed {before - len(cand)} dup)")

    return cand


# ----------------------------------------------------------------------
# Step D: build new line_list rows with consistent B_lu/B_ul/nu
# ----------------------------------------------------------------------
def build_new_lines(cand, carsus_levels, max_line_id):
    fe2_c = carsus_levels[(carsus_levels.atomic_number == Z_FE) &
                          (carsus_levels.ion_number == ION_FE2)]
    g_by_lev = fe2_c.set_index("level_number")["g"].astype(int).to_dict()

    out = []
    next_id = max_line_id + 1
    for _, r in cand.iterrows():
        wl_aa = float(r.wavelength)
        wl_cm = wl_aa * 1e-8
        nu    = C_CGS / wl_cm
        f_lu  = float(r.f_lu)
        f_ul  = float(r.f_ul)
        A_ul  = float(r.A_ul)
        # B_ul = c^2 / (2 h nu^3) * A_ul        (cgs Einstein)
        B_ul  = (C_CGS * C_CGS) / (2.0 * H_CGS * nu**3) * A_ul
        g_l   = g_by_lev[int(r.lo_carsus)]
        g_u   = g_by_lev[int(r.hi_carsus)]
        B_lu  = (g_u / g_l) * B_ul

        out.append({
            "atomic_number":       Z_FE,
            "ion_number":          ION_FE2,
            "level_number_lower":  int(r.lo_carsus),
            "level_number_upper":  int(r.hi_carsus),
            "line_id":             next_id,
            "wavelength":          wl_aa,
            "f_ul":                f_ul,
            "f_lu":                f_lu,
            "nu":                  nu,
            "B_lu":                B_lu,
            "B_ul":                B_ul,
            "A_ul":                A_ul,
            "wavelength_cm":       wl_cm,
        })
        next_id += 1
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Step F: insert new transitions into macro_atom_data preserving block
# contiguity, and update macro_atom_references.
# ----------------------------------------------------------------------
def insert_macro_rows(macro_df, refs_df, new_lines_df, line_array_idx_offset):
    """Carsus convention (verified):
        - count_down = #distinct lines decaying from this level
        - Each such line contributes 2 rows: internal_down (-1) + emission (0)
        - count_up   = #internal_up (1) rows
        - count_total = 2*count_down + count_up
        - Block layout: [down/emission pairs] then [internal_up rows]

    For each new line we add:
        +1 internal_up row inside lower-level block (appended at end of up section)
        +1 internal_down + +1 emission inside upper-level block
            (appended at end of down section, before up section)

    BUG FIX (2026-04-30): lines_idx is the ARRAY POSITION in the final
    line_list.csv (0..N-1), NOT the scalar line_id value. New lines occupy
    positions [line_array_idx_offset, line_array_idx_offset + len(new_lines)).
    transition_line_id is the scalar line_id field — those are different.
    """
    refs = refs_df.copy().reset_index(drop=True)

    # global macro-level index = row index in refs
    refs["_key"] = list(zip(refs.atomic_number.astype(int),
                            refs.ion_number.astype(int),
                            refs.source_level_number.astype(int)))
    key_to_refidx = {k: i for i, k in enumerate(refs["_key"].tolist())}

    # collect additions per ref-idx, by section
    add_down = {i: [] for i in range(len(refs))}   # pairs (internal_down + emission)
    add_up   = {i: [] for i in range(len(refs))}   # single internal_up rows

    line_to_upper_macro = []
    Z, ion = Z_FE, ION_FE2

    for k, (_, ln) in enumerate(new_lines_df.iterrows()):
        lo  = int(ln.level_number_lower)
        hi  = int(ln.level_number_upper)
        lid = int(ln.line_id)                 # scalar line_id field
        lidx = line_array_idx_offset + k      # array position in final line_list
        ref_lo = key_to_refidx[(Z, ion, lo)]
        ref_hi = key_to_refidx[(Z, ion, hi)]
        line_to_upper_macro.append(ref_hi)

        add_up[ref_lo].append({
            "atomic_number": Z, "ion_number": ion,
            "source_level_number": lo, "destination_level_number": hi,
            "transition_type": 1, "transition_probability": 0.0,
            "transition_line_id": lid, "lines_idx": lidx,
            "destination_level_idx": ref_hi, "source_level_idx": ref_lo,
        })
        add_down[ref_hi].append({
            "atomic_number": Z, "ion_number": ion,
            "source_level_number": hi, "destination_level_number": lo,
            "transition_type": -1, "transition_probability": 0.0,
            "transition_line_id": lid, "lines_idx": lidx,
            "destination_level_idx": ref_lo, "source_level_idx": ref_hi,
        })
        add_down[ref_hi].append({
            "atomic_number": Z, "ion_number": ion,
            "source_level_number": hi, "destination_level_number": lo,
            "transition_type": 0, "transition_probability": 0.0,
            "transition_line_id": lid, "lines_idx": lidx,
            "destination_level_idx": ref_lo, "source_level_idx": ref_hi,
        })

    # Verify consistency convention pre-merge (count_total = 2*count_down + count_up)
    bad = (refs["count_total"]
           != (2 * refs["count_down"] + refs["count_up"])).sum()
    if bad:
        raise RuntimeError(
            f"refs count_total != 2*count_down + count_up for {bad} rows")

    # macro_atom_data.csv has an unnamed leading index column
    if "Unnamed: 0" in macro_df.columns:
        macro_no_idx = macro_df.drop(columns=["Unnamed: 0"])
    else:
        macro_no_idx = macro_df.copy()
    macro_cols = ["atomic_number", "ion_number",
                  "source_level_number", "destination_level_number",
                  "transition_type", "transition_probability",
                  "transition_line_id", "lines_idx",
                  "destination_level_idx", "source_level_idx"]
    macro_no_idx = macro_no_idx[macro_cols]

    block_refs_orig  = refs["block_references"].astype(int).to_numpy()
    count_total_orig = refs["count_total"].astype(int).to_numpy()
    count_down_orig  = refs["count_down"].astype(int).to_numpy()

    parts = []
    new_block_refs = np.empty(len(refs), dtype=np.int64)
    cum = 0

    for i in range(len(refs)):
        bstart = block_refs_orig[i]
        n      = count_total_orig[i]
        ndown_rows = 2 * count_down_orig[i]   # rows in down section
        # original block split
        down_section = macro_no_idx.iloc[bstart:bstart + ndown_rows]
        up_section   = macro_no_idx.iloc[bstart + ndown_rows:bstart + n]

        new_block_refs[i] = cum
        parts.append(down_section)
        cum += len(down_section)
        if add_down[i]:
            df_d = pd.DataFrame(add_down[i])[macro_cols]
            parts.append(df_d)
            cum += len(df_d)
        parts.append(up_section)
        cum += len(up_section)
        if add_up[i]:
            df_u = pd.DataFrame(add_up[i])[macro_cols]
            parts.append(df_u)
            cum += len(df_u)

    new_macro = pd.concat(parts, ignore_index=True)

    # Update counts
    add_up_n   = np.array([len(add_up[i])           for i in range(len(refs))])
    add_dn_n   = np.array([len(add_down[i]) // 2    for i in range(len(refs))])  # #lines, not #rows
    refs["count_up"]    = refs["count_up"].astype(int)   + add_up_n
    refs["count_down"]  = refs["count_down"].astype(int) + add_dn_n
    refs["count_total"] = 2 * refs["count_down"] + refs["count_up"]
    refs["block_references"] = new_block_refs
    refs = refs.drop(columns=["_key"])
    return new_macro, refs, line_to_upper_macro


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    t0 = time.time()

    # Step A
    backup_ref_dir()

    # Load files
    log("Loading carsus levels…")
    carsus_levels = pd.read_csv(REF_DIR / "levels.csv")
    log(f"  {len(carsus_levels)} levels")
    log("Loading carsus line_list…")
    carsus_lines  = pd.read_csv(REF_DIR / "line_list.csv")
    log(f"  {len(carsus_lines)} lines")
    log("Loading carsus macro_atom_data…")
    macro_df      = pd.read_csv(REF_DIR / "macro_atom_data.csv")
    log(f"  {len(macro_df)} macro rows")
    log("Loading carsus macro_atom_references…")
    refs_df       = pd.read_csv(REF_DIR / "macro_atom_references.csv")
    log(f"  {len(refs_df)} ref rows")
    log("Loading line2macro_level_upper.npy…")
    l2m           = np.load(REF_DIR / "line2macro_level_upper.npy")
    log(f"  shape={l2m.shape} dtype={l2m.dtype}")
    log("Loading CMFGEN levels & line_list…")
    cmfgen_levels = pd.read_csv(CMFGEN_DIR / "levels.csv")
    cmfgen_lines  = pd.read_csv(CMFGEN_DIR / "line_list.csv")
    log(f"  cmfgen: {len(cmfgen_levels)} levels, {len(cmfgen_lines)} lines")

    # Sanity: carsus l2m length == lines length
    if len(l2m) != len(carsus_lines):
        raise RuntimeError(
            f"line2macro_level_upper length {len(l2m)} != lines {len(carsus_lines)}")

    # Step B
    lev_map, fe2_c, fe2_m = build_fe2_level_map(carsus_levels, cmfgen_levels)

    # Step C
    cand = filter_missing_lines(cmfgen_lines, cmfgen_levels, carsus_lines, lev_map)
    if cand.empty:
        log("No new lines to add. Exiting.")
        return

    # Step D — new line_list rows
    max_line_id = int(carsus_lines["line_id"].max())
    new_lines = build_new_lines(cand, carsus_levels, max_line_id)
    log(f"Built {len(new_lines)} new line rows; line_id "
        f"{new_lines.line_id.min()}..{new_lines.line_id.max()}")

    # Step F — splice macro_atom_data and update refs.
    # New lines are appended after existing carsus lines, so the array index
    # of the first new line in the final line_list is len(carsus_lines).
    new_macro, new_refs, upper_macro_idx = insert_macro_rows(
        macro_df, refs_df, new_lines, len(carsus_lines))
    log(f"macro_atom_data: {len(macro_df)} -> {len(new_macro)}  "
        f"(+{len(new_macro) - len(macro_df)})")

    # Step E — extend line2macro_level_upper
    new_l2m = np.concatenate([l2m, np.array(upper_macro_idx, dtype=l2m.dtype)])
    log(f"line2macro_level_upper: {len(l2m)} -> {len(new_l2m)}")

    # Step G — sanity checks
    log("=" * 60)
    log("SANITY CHECKS")
    log("=" * 60)
    final_lines = pd.concat([carsus_lines, new_lines], ignore_index=True)
    checks = []

    # 1
    ok = (len(final_lines) == len(carsus_lines) + len(new_lines))
    checks.append(("line_list row count", ok,
                   f"{len(final_lines)} == {len(carsus_lines)}+{len(new_lines)}"))

    # 2
    ok = (len(new_l2m) == len(final_lines))
    checks.append(("line2macro length matches lines", ok,
                   f"{len(new_l2m)} vs {len(final_lines)}"))

    # 3
    br = new_refs["block_references"].to_numpy()
    ok = bool(np.all(np.diff(br) >= 0))
    checks.append(("block_references monotonic non-decreasing", ok,
                   f"min diff = {int(np.diff(br).min()) if len(br)>1 else 'n/a'}"))

    # 4
    ok = (int(new_refs["count_total"].sum()) == len(new_macro))
    checks.append(("sum(count_total) == macro rows", ok,
                   f"{int(new_refs['count_total'].sum())} vs {len(new_macro)}"))

    # 5 — count_up/down deltas (per new line: +1 up at lower, +1 down at upper)
    delta_up = (new_refs["count_up"].astype(int).to_numpy()
                - refs_df["count_up"].astype(int).to_numpy())
    delta_dn = (new_refs["count_down"].astype(int).to_numpy()
                - refs_df["count_down"].astype(int).to_numpy())
    expected_up = int(len(new_lines))
    expected_dn = int(len(new_lines))
    ok = (delta_up.sum() == expected_up) and (delta_dn.sum() == expected_dn)
    checks.append(("count_up/count_down deltas", ok,
                   f"sum dup={delta_up.sum()} (expect {expected_up}); "
                   f"sum dn={delta_dn.sum()} (expect {expected_dn})"))

    # 6 — line_id uniqueness
    ok = (final_lines["line_id"].is_unique)
    checks.append(("line_id is unique", ok,
                   f"n_unique={final_lines['line_id'].nunique()} "
                   f"vs len={len(final_lines)}"))

    # 6b — count_total invariant preserved
    inv_bad = int((new_refs["count_total"]
                   != 2 * new_refs["count_down"] + new_refs["count_up"]).sum())
    ok = (inv_bad == 0)
    checks.append(("count_total == 2*count_down + count_up", ok,
                   f"violations={inv_bad}"))

    # 7 — block contiguity: the macro rows for each level should all share source_level_number
    # spot check first 5 affected ref rows
    affected = np.where((delta_up + delta_dn) > 0)[0]
    spot_fail = 0
    for i in affected[:8]:
        bstart = int(new_refs.loc[i, "block_references"])
        n      = int(new_refs.loc[i, "count_total"])
        block  = new_macro.iloc[bstart:bstart + n]
        srcs   = block["source_level_number"].unique()
        if len(srcs) != 1 or int(srcs[0]) != int(new_refs.loc[i, "source_level_number"]):
            spot_fail += 1
    ok = (spot_fail == 0)
    checks.append(("block contiguity spot check (first 8 affected)", ok,
                   f"{spot_fail} failures"))

    all_ok = True
    for name, ok, detail in checks:
        log(f"  [{ 'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            all_ok = False

    if not all_ok:
        log("ABORT: sanity check failed; NOT writing output.")
        sys.exit(2)

    # ------------------------------------------------------------------
    # Step G2 — Sort merged line_list by descending nu and remap indices.
    # ------------------------------------------------------------------
    # The C code (lumina_atomic.c, lumina_plasma.c, MC kernel) ASSUMES the
    # line_list is sorted by descending nu (= ascending wavelength). The new
    # 132 lines were appended at the tail and break that ordering. Without
    # a re-sort, the binary search in the MC kernel cannot reach the new
    # lines, and the formal-integral z-walk produces inconsistent positions.
    # ------------------------------------------------------------------
    log("=" * 60)
    log("SORT: descending nu + permutation remap")
    log("=" * 60)
    nu_arr = final_lines["nu"].to_numpy()
    # Stable sort to keep original order among equal-nu duplicates
    sort_idx = np.argsort(-nu_arr, kind="stable")
    inv_perm = np.empty(len(sort_idx), dtype=np.int64)
    inv_perm[sort_idx] = np.arange(len(sort_idx))

    # Reorder line_list itself
    final_lines_sorted = final_lines.iloc[sort_idx].reset_index(drop=True)
    nu_sorted = final_lines_sorted["nu"].to_numpy()
    desc_violations = int(np.sum(np.diff(nu_sorted) > 0))
    log(f"  sort: {len(final_lines)} rows; descending violations={desc_violations}")

    # Reorder line2macro_level_upper (per-line array)
    new_l2m_sorted = new_l2m[sort_idx]

    # Remap lines_idx in macro_atom_data: each row's lines_idx points to a
    # row in line_list. After sorting, that same physical line is now at
    # inv_perm[old_idx].
    log("  remapping macro_atom_data.lines_idx through inv_perm…")
    li_old = new_macro["lines_idx"].to_numpy()
    li_new = inv_perm[li_old]
    new_macro = new_macro.copy()
    new_macro["lines_idx"] = li_new

    # Sanity: verify the remap preserves the macro->line correspondence by
    # checking that Z, ion, and source_level_number (== line.level_number_lower
    # for internal_up rows) match. The destination level numbering follows a
    # carsus +1 offset convention which we don't verify here.
    sample_n = min(200, len(new_macro))
    sample_idx = np.random.RandomState(0).choice(len(new_macro), sample_n, replace=False)
    bad_ref = 0
    for i in sample_idx:
        row = new_macro.iloc[int(i)]
        if int(row["transition_type"]) != 1:
            continue
        lidx = int(row["lines_idx"])
        if lidx < 0 or lidx >= len(final_lines_sorted):
            bad_ref += 1
            continue
        ln = final_lines_sorted.iloc[lidx]
        zok  = int(ln.atomic_number)        == int(row["atomic_number"])
        ionok = int(ln.ion_number)          == int(row["ion_number"])
        lok  = int(ln.level_number_lower)   == int(row["source_level_number"])
        if not (zok and ionok and lok):
            bad_ref += 1
    log(f"  random spot check ({sample_n} rows): bad refs = {bad_ref}")
    if bad_ref > 0:
        log("ABORT: lines_idx remap broken; NOT writing.")
        sys.exit(3)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    log("Writing outputs…")
    final_lines_sorted.to_csv(REF_DIR / "line_list.csv", index=False)
    np.save(REF_DIR / "line2macro_level_upper.npy", new_l2m_sorted)
    # macro_atom_data.csv has an unnamed leading index column
    new_macro.to_csv(REF_DIR / "macro_atom_data.csv", index=True, index_label="")
    new_refs.to_csv(REF_DIR / "macro_atom_references.csv", index=False)
    log("Wrote line_list.csv (sorted), line2macro_level_upper.npy (sorted), "
        "macro_atom_data.csv (lines_idx remapped), macro_atom_references.csv")

    # ------------------------------------------------------------------
    # Step H — summary
    # ------------------------------------------------------------------
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Lines added:               {len(new_lines)}")
    affected_levels = set(new_lines.level_number_lower.tolist()) | \
                      set(new_lines.level_number_upper.tolist())
    log(f"Distinct Fe II levels affected: {len(affected_levels)}  "
        f"(ground={len(set(new_lines.level_number_lower))}, "
        f"intermediate={len(set(new_lines.level_number_upper))})")
    log(f"Wavelength range:          "
        f"{new_lines.wavelength.min():.2f} – {new_lines.wavelength.max():.2f} Å")
    log("A_ul histogram:")
    bins = [0, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    h, _ = np.histogram(new_lines["A_ul"].to_numpy(), bins=bins)
    for lo, hi, n in zip(bins[:-1], bins[1:], h):
        log(f"  [{lo:.0e}, {hi:.0e})  {int(n)}")
    log(f"max line_id:               {int(final_lines.line_id.max())}")
    log(f"macro_atom_data rows:      {len(new_macro)}  "
        f"(+{len(new_macro) - len(macro_df)})")
    log(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
