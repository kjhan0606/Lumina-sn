#!/usr/bin/env python3
"""Cross-walk AS planE Cr/Mn/Ni III levels and bb lines into the capraise
reference (levels.csv + line_list.csv).

Strategy:
  - For each AS level, find the closest CMFGEN level with matching g and
    |ΔE| < tolerance.  Matched AS levels share the CMFGEN level index;
    unmatched AS levels with E > E_max_cmfgen append as level_number 1000+.
  - For each AS bb transition (K=upper, KP=lower):
      lower_new = matched(KP) if exists else extension_idx(KP) if available
      upper_new = matched(K)  if exists else extension_idx(K)  if available
      INCLUDE only if upper_new is an EXTENSION level (≥ original_N_cmf),
      so we never duplicate a CMFGEN bb line.

Output: data/tardis_reference_v3_femerge_capraise_asbb/
  levels.csv (CMFGEN + AS extensions for Cr/Mn/Ni III)
  line_list.csv (CMFGEN + AS extension transitions for Cr/Mn/Ni III)
  All other CSVs hardlink/copy from the source capraise reference.
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn')
SRC = ROOT / 'data' / 'tardis_reference_v3_femerge_capraise'
DST = ROOT / 'data' / 'tardis_reference_v3_femerge_capraise_asbb'
AS_DIR = ROOT / 'data' / 'as_planE_iron3'

# (Z, ion=2 always for iron-peak III) → (AS levels.csv path, AS lines.csv path, dE_tol_eV)
IONS = [
    (24, AS_DIR / 'cr3_levels.csv', AS_DIR / 'cr3_lines.csv'),
    (25, AS_DIR / 'mn3_levels.csv', AS_DIR / 'mn3_lines.csv'),
    (28, AS_DIR / 'ni3_levels.csv', AS_DIR / 'ni3_lines.csv'),
]
ION_NUMBER = 2  # all three are doubly ionized (III)
DE_TOL_EV = 0.5   # AS vs CMFGEN energy match tolerance

C_CGS  = 2.99792458e10
H_CGS  = 6.62607015e-27
M_E_CGS = 9.1093837015e-28
E_CGS  = 4.80320451e-10
PI     = np.pi
HC_EVA = 12398.41984


def cross_walk(as_lv: pd.DataFrame, cmf_lv: pd.DataFrame, dE_tol: float):
    """Return as_lv with added 'matched_cmf_idx' (level_number in CMFGEN
    space) and 'is_extension' columns.  Unmatched levels above
    cmf_lv.energy_eV.max() are flagged for extension."""
    cmf_E = cmf_lv['energy_eV'].values
    cmf_g = cmf_lv['g'].values
    cmf_idx = cmf_lv['level_number'].values
    E_max_cmf = float(cmf_E.max())

    matched = np.full(len(as_lv), -1, dtype=int)
    cmf_taken = np.zeros(len(cmf_lv), dtype=bool)

    # Greedy 1-to-1: sort all AS levels by energy (ascending); for each, find
    # the closest CMFGEN level (with matching g, not yet taken) within tol.
    as_E = as_lv['energy_eV'].values
    as_g = as_lv['g'].values
    order = np.argsort(as_E)
    for i in order:
        g = as_g[i]
        cand = np.where((cmf_g == g) & (~cmf_taken))[0]
        if len(cand) == 0:
            continue
        dE = np.abs(cmf_E[cand] - as_E[i])
        j = cand[np.argmin(dE)]
        if dE[np.argmin(dE)] <= dE_tol:
            matched[i] = cmf_idx[j]
            cmf_taken[j] = True

    is_ext = (matched < 0) & (as_E > E_max_cmf)
    return matched, is_ext, E_max_cmf


def build_lines_from_as(
    as_ll: pd.DataFrame,
    as_lv: pd.DataFrame,
    matched: np.ndarray,
    is_ext: np.ndarray,
    ext_new_idx: np.ndarray,
    next_line_id: int,
    Z: int,
):
    """For each AS bb transition, map endpoints; include only if the upper
    endpoint is an extension level."""
    # AS index 'K_upper' and 'KP_lower' in the parser are 1-based; here we
    # use the AS level_number = K-1 stored in as_lv.level_number.  The
    # parser already converted to level_number_lower / level_number_upper
    # using K-1 / KP-1 in cr3_lines.csv.
    nA = len(as_lv)
    # Per-AS-index mapping: matched cmf idx, or extension new idx, or -1
    mapping = np.full(nA, -1, dtype=int)
    mapping[matched >= 0] = matched[matched >= 0]
    mapping[is_ext] = ext_new_idx[is_ext]

    low_as = as_ll['level_number_lower'].values  # AS-space, 0..nA-1
    up_as  = as_ll['level_number_upper'].values
    low_new = mapping[low_as]
    up_new  = mapping[up_as]

    # Determine extension threshold: original CMFGEN count
    n_cmf = (matched >= 0).sum() + (is_ext == False).sum() - (matched < 0).sum() + 0  # not used
    # Simpler: define ext set by membership
    is_ext_idx = np.zeros(nA, dtype=bool)
    is_ext_idx[is_ext] = True

    # Include line if both endpoints valid AND upper is extension
    valid = (low_new >= 0) & (up_new >= 0) & is_ext_idx[up_as]
    print(f"  Z={Z}: AS-bb rows={len(as_ll)} valid (upper=ext, lower mapped)={valid.sum()}")

    ll_new = as_ll[valid].copy()
    ll_new['atomic_number'] = Z
    ll_new['ion_number'] = ION_NUMBER
    ll_new['level_number_lower'] = low_new[valid]
    ll_new['level_number_upper'] = up_new[valid]
    ll_new['line_id'] = next_line_id + np.arange(valid.sum())
    return ll_new


def main():
    print(f"[merge] source = {SRC}")
    print(f"[merge] dest   = {DST}")
    DST.mkdir(parents=True, exist_ok=True)

    cmf_lv_all = pd.read_csv(SRC / 'levels.csv')
    cmf_ll_all = pd.read_csv(SRC / 'line_list.csv')

    new_levels_pieces = [cmf_lv_all.copy()]
    new_lines_pieces  = [cmf_ll_all.copy()]
    next_line_id = int(cmf_ll_all['line_id'].max()) + 1

    summary = []
    for (Z, as_lv_path, as_ll_path) in IONS:
        print(f"\n=== Z={Z} ion={ION_NUMBER} ===")
        as_lv = pd.read_csv(as_lv_path)
        as_ll = pd.read_csv(as_ll_path)
        cmf_lv = cmf_lv_all[(cmf_lv_all.atomic_number == Z)
                            & (cmf_lv_all.ion_number == ION_NUMBER)].copy()
        cmf_lv = cmf_lv.reset_index(drop=True)

        n_cmf = len(cmf_lv)
        n_as  = len(as_lv)
        E_max_cmf = float(cmf_lv['energy_eV'].max())
        print(f"  CMFGEN levels: {n_cmf}  (E_max {E_max_cmf:.3f} eV)")
        print(f"  AS     levels: {n_as}  (E_max {as_lv.energy_eV.max():.3f} eV)")

        matched, is_ext, _ = cross_walk(as_lv, cmf_lv, DE_TOL_EV)
        n_matched = int((matched >= 0).sum())
        n_ext = int(is_ext.sum())
        n_skip = n_as - n_matched - n_ext
        print(f"  matched to CMFGEN : {n_matched}")
        print(f"  extension (new)   : {n_ext}")
        print(f"  skipped (no match, E<=E_cmf) : {n_skip}")

        # Assign new level_number for extension AS levels
        ext_new_idx = np.full(n_as, -1, dtype=int)
        ext_lv_indices = np.where(is_ext)[0]
        ext_new_idx[ext_lv_indices] = n_cmf + np.arange(n_ext)

        # New extension level rows
        ext_rows = as_lv.iloc[ext_lv_indices].copy()
        ext_rows['atomic_number'] = Z
        ext_rows['ion_number'] = ION_NUMBER
        ext_rows['level_number'] = ext_new_idx[ext_lv_indices]
        ext_rows['metastable'] = 0
        ext_rows = ext_rows[['atomic_number', 'ion_number',
                             'level_number', 'energy_eV', 'g', 'metastable']]
        new_levels_pieces.append(ext_rows)

        # Build the lines.  Note as_ll's level_number_{lower,upper} are
        # in AS index space (K-1), so we map through matched/ext_new_idx.
        new_ll = build_lines_from_as(
            as_ll, as_lv, matched, is_ext, ext_new_idx, next_line_id, Z)
        # The new_ll already has the capraise schema columns from the AS
        # parser; just align column order with cmf_ll_all.
        new_ll = new_ll[cmf_ll_all.columns.tolist()]
        new_lines_pieces.append(new_ll)
        next_line_id += len(new_ll)

        summary.append((Z, n_cmf, n_as, n_matched, n_ext, len(new_ll)))

    new_levels = pd.concat(new_levels_pieces, ignore_index=True)
    new_lines  = pd.concat(new_lines_pieces,  ignore_index=True)

    new_levels.to_csv(DST / 'levels.csv', index=False)
    new_lines.to_csv(DST / 'line_list.csv', index=False)

    # Copy / symlink the other capraise artifacts unchanged.
    for fn in ('abundances.csv', 'abundances.npy', 'atom_masses.csv',
               'cmfgen_sigma_bf.bin', 'config.json', 'density.csv',
               'electron_densities.csv', 'geometry.csv',
               'ion_number_density.npy', 'ionization_energies.csv',
               'j_blues.npy', 'line2macro_level_upper.npy',
               'macro_atom_data.csv', 'macro_atom_references.csv',
               'mc_estimators.csv', 'plasma_state.csv',
               'tau_sobolev.npy', 'tau_sobolev_stats.csv'):
        src = SRC / fn
        if not src.exists():
            continue
        dst = DST / fn
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    print("\n=== summary ===")
    print(f"{'Z':>3} {'CMFGEN':>7} {'AS':>6} {'matched':>8} {'ext':>6} {'new bb':>10}")
    for Z, n_cmf, n_as, n_matched, n_ext, n_new_ll in summary:
        print(f"{Z:>3} {n_cmf:>7} {n_as:>6} {n_matched:>8} {n_ext:>6} {n_new_ll:>10}")
    print(f"\nWrote {len(new_levels)} levels, {len(new_lines)} lines to {DST}")


if __name__ == '__main__':
    main()
