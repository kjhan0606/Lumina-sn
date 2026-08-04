#!/usr/bin/env python3
"""Emit the .npy companion files LUMINA expects in a CMFGEN-built ref dir.

Inputs read from REF_DIR:
  levels.csv, line_list.csv, macro_atom_references.csv, macro_atom_data.csv

Outputs written into REF_DIR:
  line2macro_level_upper.npy   [n_lines]            int64 — global level idx
                                                          of each line's upper
  tau_sobolev.npy              [n_lines, n_shells]  float64 zeros (loader
                                                          recomputes per iter)
  transition_probabilities.npy [n_trans, n_shells]  float64 1/N broadcast from
                                                          macro_atom_data.csv
                                                          (config.hold_iterations
                                                          delays recompute, so
                                                          iter<3 packets need
                                                          valid loaded values)

The shell count is a strict runtime contract.  The generated K-SHAPE sidecar
binds both arrays to this line-list epoch; the C loader rejects any mismatch.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from kshape_contract import write_contract


def main(ref_dir: Path, n_shells: int = 30) -> None:
    print(f"[finalize] ref_dir = {ref_dir}")
    print(f"[finalize] n_shells (initial guess) = {n_shells}")

    mr = pd.read_csv(ref_dir / "macro_atom_references.csv")
    # #305-3 fix: L2M must hold the *level-global index* (row position in
    # macro_atom_references.csv == row position in levels.csv), NOT
    # `references_idx` which is the cumulative transition-index. Using the
    # latter put L2M into the range [0, n_transitions) ≈ 6.7M while the
    # runtime bounds-checks against n_macro_levels ≈ 37513 → 99.46% silent
    # bypass of macro-atom interaction (#305-3 audit).
    lookup: dict[tuple[int, int, int], int] = {
        (int(r.atomic_number), int(r.ion_number), int(r.source_level_number)):
            row_idx
        for row_idx, r in enumerate(mr.itertuples(index=False))
    }
    print(f"[finalize] level lookup: {len(lookup):,} entries")

    ll = pd.read_csv(ref_dir / "line_list.csv")
    n_lines = len(ll)
    print(f"[finalize] line_list:    {n_lines:,} rows")

    # B2 fix: re-derive B_lu, B_ul, f_lu from A_ul + statistical weights so
    # Einstein detailed balance (B_lu · g_l == B_ul · g_u) holds exactly.
    # Audit (Tier physical-sanity, 2026-05-29) found 554,497 / 2,234,702
    # (24.8%) violations, concentrated in iron-peak II (Fe II 124K, Co II 118K,
    # Ti II 36K, Sc II 31K, Cr II 28K, Ni II 28K). Mixed conventions from v3
    # carsus + planE add-on are the likely cause. Recompute uniformly from
    # (A_ul, nu, g_l, g_u) in J_nu CGS convention:
    #   B_ul = (c^2 / 2 h nu^3) · A_ul       [cm^3 erg^-1 s^-2]
    #   B_lu = (g_u / g_l) · B_ul
    #   f_lu = (m_e c / 8 pi^2 e^2) · lambda^2 · (g_u / g_l) · A_ul
    lev = pd.read_csv(ref_dir / "levels.csv")
    g_lookup = {(int(r.atomic_number), int(r.ion_number), int(r.level_number)):
                float(r.g) for r in lev.itertuples(index=False)}
    g_l = np.array([g_lookup.get(
        (int(r.atomic_number), int(r.ion_number), int(r.level_number_lower)), np.nan)
        for r in ll.itertuples(index=False)], dtype=np.float64)
    g_u = np.array([g_lookup.get(
        (int(r.atomic_number), int(r.ion_number), int(r.level_number_upper)), np.nan)
        for r in ll.itertuples(index=False)], dtype=np.float64)
    n_missing_g = int(np.sum(np.isnan(g_l) | np.isnan(g_u)))
    if n_missing_g > 0:
        raise AssertionError(
            f"[finalize] B2: {n_missing_g} lines have missing g_l or g_u in levels.csv"
        )
    # CGS constants
    C_CGS  = 2.99792458e10      # cm/s
    H_CGS  = 6.62607015e-27     # erg·s
    ME_CGS = 9.1093837015e-28   # g
    E_CGS  = 4.80320425e-10     # esu (statcoulomb)
    nu_arr  = ll["nu"].to_numpy(dtype=np.float64)
    lam_arr = ll["wavelength_cm"].to_numpy(dtype=np.float64)
    A_ul_arr = ll["A_ul"].to_numpy(dtype=np.float64)

    B_ul_new = (C_CGS**2 / (2.0 * H_CGS * nu_arr**3)) * A_ul_arr
    B_lu_new = (g_u / g_l) * B_ul_new
    f_lu_new = (ME_CGS * C_CGS / (8.0 * np.pi**2 * E_CGS**2)) * \
               (lam_arr**2) * (g_u / g_l) * A_ul_arr

    # Drift check vs stored values (informational): largest fractional change
    def _frac_max(new, old):
        old = np.asarray(old, dtype=np.float64)
        ok = np.isfinite(old) & (np.abs(old) > 0)
        if not ok.any():
            return float("nan")
        return float(np.max(np.abs(new[ok] - old[ok]) / np.abs(old[ok])))
    drift_Bul = _frac_max(B_ul_new, ll["B_ul"])
    drift_Blu = _frac_max(B_lu_new, ll["B_lu"])
    drift_flu = _frac_max(f_lu_new, ll["f_lu"])
    ll["B_ul"] = B_ul_new
    ll["B_lu"] = B_lu_new
    ll["f_lu"] = f_lu_new
    # f_ul = -(g_l/g_u) * f_lu  (negative emission oscillator strength convention)
    ll["f_ul"] = -(g_l / g_u) * f_lu_new

    # Exact detailed balance check on the new values
    db_lhs = B_lu_new * g_l
    db_rhs = B_ul_new * g_u
    db_max_rel = float(np.max(np.abs(db_lhs - db_rhs) / np.maximum(np.abs(db_lhs), 1e-300)))
    assert db_max_rel < 1e-12, (
        f"[finalize] B2: detailed balance violated after recompute: max|ΔB/B| = {db_max_rel:.3e}"
    )
    print(f"[finalize] B2: re-derived B_ul / B_lu / f_lu / f_ul from A_ul + g")
    print(f"[finalize] B2:   max frac drift vs stored — B_ul {drift_Bul:.3e}, "
          f"B_lu {drift_Blu:.3e}, f_lu {drift_flu:.3e}")
    print(f"[finalize] B2:   detailed balance B_lu·g_l vs B_ul·g_u max rel = {db_max_rel:.3e}")

    # Bug #4 fix: detect duplicate (Z,ion,lo,up) tuples. Duplicates
    # double-count Sobolev opacity in transport and mis-index line→upper
    # lookups. A clean pipeline must yield 0 duplicates; otherwise raise.
    key_cols = ["atomic_number", "ion_number",
                "level_number_lower", "level_number_upper"]
    n_unique = ll[key_cols].drop_duplicates().shape[0]
    n_dups = n_lines - n_unique
    if n_dups > 0:
        sample = ll[ll.duplicated(subset=key_cols, keep=False)].head(10)
        raise AssertionError(
            f"[finalize] {n_dups} duplicate (Z,ion,lo,up) rows in line_list.csv. "
            f"Sample:\n{sample.to_string(index=False)}\n"
            f"Fix in the merge step that produced this ref dir; do NOT silently dedup here."
        )
    print(f"[finalize] line_list dedup check: 0 duplicates")

    # B1 fix: enforce strictly-descending nu globally. LUMINA's
    # d_calculate_distance_line binary-searches line_list_nu assuming
    # nu[i] >= nu[i+1]; a single ascending pair silently mis-routes packets.
    # Upstream merges (merge_iron_peak_III_bb.py, merge_as_planE_into_capraise.py)
    # sort by (Z, ion, wavelength_cm) — correct within ion, NOT globally.
    # Do the global sort here once, capture permutation, remap line_id and
    # every macro_atom_data reference (transition_line_id + lines_idx).
    # NOTE: applied unconditionally — even when already descending, B2 above
    # mutated B/f columns and the file needs rewriting.
    nu = ll["nu"].to_numpy(dtype=np.float64)
    asc_pairs_before = int(np.sum(np.diff(nu) > 0))
    new_order = np.argsort(-nu, kind="mergesort")
    old_lid = ll["line_id"].to_numpy(dtype=np.int64).copy()
    ll = ll.iloc[new_order].reset_index(drop=True)
    # Re-issue line_id := row index in new order (loader expectation)
    old_to_new = np.full(int(old_lid.max()) + 1, -1, dtype=np.int64)
    old_to_new[old_lid[new_order]] = np.arange(len(ll), dtype=np.int64)
    ll["line_id"] = np.arange(len(ll), dtype=np.int64)
    ll.to_csv(ref_dir / "line_list.csv", index=False)
    nu_new = ll["nu"].to_numpy(dtype=np.float64)
    asc_pairs_after = int(np.sum(np.diff(nu_new) > 0))
    assert asc_pairs_after == 0, (
        f"[finalize] post-sort still {asc_pairs_after} ascending pairs"
    )
    if asc_pairs_before > 0:
        print(f"[finalize] B1: globally sorted line_list by nu desc "
              f"({asc_pairs_before} ascending pairs → 0)")
    else:
        print(f"[finalize] B1: line_list already strictly descending "
              f"(rewrote with B2 column updates)")

    # Remap macro_atom_data references through old_to_new
    ma_path = ref_dir / "macro_atom_data.csv"
    ma_full = pd.read_csv(ma_path, index_col=0)
    for col in ("transition_line_id", "lines_idx"):
        v = ma_full[col].to_numpy(dtype=np.int64)
        keep = v >= 0
        v_out = v.copy()
        # bounds guard: any old_lid outside table range stays -1
        in_range = keep & (v < old_to_new.shape[0])
        v_out[in_range] = old_to_new[v[in_range]]
        n_unmapped = int(np.sum(keep & (v_out == -1)))
        if n_unmapped > 0:
            print(f"[finalize] B1 WARN: {n_unmapped} {col} entries unmapped "
                  f"(old_lid not in new table)")
        ma_full[col] = v_out
    ma_full.to_csv(ma_path)
    print(f"[finalize] B1: remapped macro_atom_data line refs through permutation")

    line2macro_level_upper = np.empty(n_lines, dtype=np.int64)
    missing = 0
    missing_keys: list[tuple[int, int, int]] = []
    for i, r in enumerate(ll.itertuples(index=False)):
        key = (int(r.atomic_number), int(r.ion_number), int(r.level_number_upper))
        if key in lookup:
            line2macro_level_upper[i] = lookup[key]
        else:
            line2macro_level_upper[i] = -1
            missing += 1
            if len(missing_keys) < 20:
                missing_keys.append(key)
    # Bug #3 fix: hard-assert no orphan lines. Silent -1 fallback meant
    # extension levels (planE) had line→upper mappings written as -1, which
    # the transport code treats as "no NLTE target" → forced scatter / no
    # cascade → P-Cygni washout. Pipeline must re-run with Bug #1 patch
    # (merge_as_planE macro_atom layer fix) to produce 0-orphan output.
    if missing:
        sample = ", ".join(f"(Z={k[0]},ion={k[1]},lvl={k[2]})" for k in missing_keys[:10])
        raise AssertionError(
            f"[finalize] {missing} lines have no upper-level in macro_atom_references. "
            f"Sample: {sample}. Re-run upstream pipeline so every line.upper has a refs row."
        )
    np.save(ref_dir / "line2macro_level_upper.npy", line2macro_level_upper)
    print(f"[finalize] wrote line2macro_level_upper.npy (0 orphans)")

    ma = pd.read_csv(ref_dir / "macro_atom_data.csv",
                     usecols=["atomic_number", "ion_number",
                              "source_level_number", "transition_probability"])
    ma_lines = len(ma)
    print(f"[finalize] macro_atom_data: {ma_lines:,} rows")

    tau_sob = np.zeros((n_lines, n_shells), dtype=np.float64)
    np.save(ref_dir / "tau_sobolev.npy", tau_sob)
    print(f"[finalize] wrote tau_sobolev.npy  [{n_lines}, {n_shells}]")

    probs = ma["transition_probability"].to_numpy(dtype=np.float64)
    # Bug #6 fix: pre-normalize transition_probability per (Z, ion, source_level)
    # block so each source-level's branching ratios sum to 1.0. Macro-atom
    # transport assumes Σ_target p = 1; CMFGEN raw probabilities are RELATIVE
    # weights, not pre-normalized. Without this step the sampling is biased.
    keys = (ma["atomic_number"].to_numpy().astype(np.int64) * 100_000_000
            + ma["ion_number"].to_numpy().astype(np.int64) * 1_000_000
            + ma["source_level_number"].to_numpy().astype(np.int64))
    block_sum = np.zeros_like(probs)
    np.add.at(block_sum, np.searchsorted(np.unique(keys), keys), 0.0)  # placeholder
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sums, inv, probs)
    safe_sums = np.where(sums > 0.0, sums, 1.0)
    probs_norm = probs / safe_sums[inv]
    n_empty_blocks = int(np.sum(sums == 0.0))
    if n_empty_blocks > 0:
        print(f"[finalize] WARN: {n_empty_blocks} source-level blocks have Σp=0 "
              f"(will stay zero — orphan levels with no outgoing transitions)")
    # sanity: every non-empty block sums to ~1
    sums_after = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sums_after, inv, probs_norm)
    nonempty = sums > 0.0
    max_err = float(np.max(np.abs(sums_after[nonempty] - 1.0))) if nonempty.any() else 0.0
    assert max_err < 1e-9, f"[finalize] per-block normalization failed: max |Σp − 1| = {max_err:.3e}"
    tp = np.broadcast_to(probs_norm[:, None], (ma_lines, n_shells)).copy()
    np.save(ref_dir / "transition_probabilities.npy", tp)
    nz = np.count_nonzero(probs_norm)
    print(f"[finalize] wrote transition_probabilities.npy  [{ma_lines}, {n_shells}] "
          f"(nz={nz}/{ma_lines}, blocks={len(uniq)}, "
          f"mean={probs_norm.mean():.4g}, max_block_err={max_err:.2e})")

    contract = write_contract(ref_dir)
    print(f"[finalize] wrote {contract.name} (line epoch + both NPY hashes)")

    print("[finalize] done.")


if __name__ == "__main__":
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tardis_reference_cmfgen")
    n_sh = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(ref.resolve(), n_sh)
