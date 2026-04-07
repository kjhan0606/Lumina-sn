#!/usr/bin/env python3
"""Expand atomic data from HDF5 for 11 elements (C,O,Mg,Si,S,Ca,Ti,Cr,Fe,Co,Ni).

Reads data/atomic/kurucz_cd23_chianti_H_He.h5 and regenerates ALL files in
data/tardis_reference/ including line_list.csv, macro_atom_data.csv, and NPY files.

HDF5 indexing conventions:
  - levels_data, lines_data, macro_atom_references, ionization_data, zeta_data:
    Z is 0-indexed (Z_idx = Z_real - 1)
  - macro_atom_data block0: Z is real (1-indexed)
  - atom_data: Z is real (1-indexed)

Output CSVs use real atomic numbers (6=C, 8=O, 12=Mg, 14=Si, etc.)
"""

import h5py
import numpy as np
import os
import sys

HDF5_PATH = "data/atomic/kurucz_cd23_chianti_H_He.h5"
OUT_DIR = "data/tardis_reference"

# 11 elements: real Z values
ELEMENTS = [6, 8, 12, 14, 16, 20, 22, 24, 26, 27, 28]
ELEMENT_NAMES = {6:'C',8:'O',12:'Mg',14:'Si',16:'S',20:'Ca',
                 22:'Ti',24:'Cr',26:'Fe',27:'Co',28:'Ni'}
# HDF5 0-indexed: Z_real - 1
ELEMENTS_IDX = [z - 1 for z in ELEMENTS]

# Default abundances per element (uniform across shells)
DEFAULT_ABUNDANCES = {
    6:  0.02,    # C
    8:  0.0417,  # O (reduced from 0.05 by 0.0083 for Mg+Ti+Cr)
    12: 0.005,   # Mg (NEW)
    14: 0.1,     # Si
    16: 0.05,    # S
    20: 0.05,    # Ca
    22: 0.0003,  # Ti (NEW)
    24: 0.003,   # Cr (NEW)
    26: 0.5,     # Fe
    27: 0.05,    # Co
    28: 0.13,    # Ni
}

# Number of shells (read from existing geometry)
N_SHELLS = 30


def load_plasma_state():
    """Load W and T_rad from existing plasma_state.csv for j_blues computation."""
    path = os.path.join(OUT_DIR, "plasma_state.csv")
    W = []
    T_rad = []
    with open(path) as fp:
        header = fp.readline()
        for line in fp:
            parts = line.strip().split(',')
            W.append(float(parts[1]))
            T_rad.append(float(parts[2]))
    return np.array(W), np.array(T_rad)


def main():
    if not os.path.exists(HDF5_PATH):
        print(f"ERROR: {HDF5_PATH} not found")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    f = h5py.File(HDF5_PATH, "r")

    # ========================================================
    # 1. LEVELS DATA
    # ========================================================
    print("=== Extracting levels data ===")
    lev_Z_idx = f["levels_data/axis1_label0"][:]
    lev_ion = f["levels_data/axis1_label1"][:]
    lev_num = f["levels_data/axis1_label2"][:]
    lev_energy = f["levels_data/block0_values"][:, 0]
    lev_g = f["levels_data/block1_values"][:, 0]
    lev_metastable = f["levels_data/block2_values"][:, 0]

    lev_mask = np.isin(lev_Z_idx, ELEMENTS_IDX)
    lev_indices = np.where(lev_mask)[0]
    n_levels = len(lev_indices)
    print(f"  Levels: {n_levels} / {len(lev_Z_idx)} selected")

    # Build level lookup: (Z_idx, ion, level_num) -> new level index
    level_lookup = {}
    with open(os.path.join(OUT_DIR, "levels.csv"), "w") as fp:
        fp.write("atomic_number,ion_number,level_number,energy_eV,g,metastable\n")
        for new_idx, old_idx in enumerate(lev_indices):
            z_real = int(lev_Z_idx[old_idx]) + 1
            ion = int(lev_ion[old_idx])
            lvl = int(lev_num[old_idx])
            fp.write(f"{z_real},{ion},{lvl},{lev_energy[old_idx]:.10f},"
                     f"{int(lev_g[old_idx])},{int(lev_metastable[old_idx])}\n")
            level_lookup[(int(lev_Z_idx[old_idx]), ion, lvl)] = new_idx

    for z in ELEMENTS:
        z_idx = z - 1
        n = sum(1 for i in lev_indices if lev_Z_idx[i] == z_idx)
        print(f"    Z={z:2d} ({ELEMENT_NAMES[z]:2s}): {n} levels")

    # ========================================================
    # 2. LINES DATA (sorted by descending nu)
    # ========================================================
    print("\n=== Extracting lines data ===")
    line_Z_idx = f["lines_data/axis1_label0"][:]
    line_ion = f["lines_data/axis1_label1"][:]
    line_lvl_lower = f["lines_data/axis1_label2"][:]
    line_lvl_upper = f["lines_data/axis1_label3"][:]
    line_id_all = f["lines_data/block0_values"][:, 0]  # global HDF5 line IDs
    line_block1 = f["lines_data/block1_values"][:]
    # block1 columns: wavelength, f_ul, f_lu, nu, B_lu, B_ul, A_ul

    line_mask = np.isin(line_Z_idx, ELEMENTS_IDX)
    line_indices = np.where(line_mask)[0]
    n_lines_raw = len(line_indices)
    print(f"  Lines: {n_lines_raw} / {len(line_Z_idx)} selected")

    # Extract filtered data
    filt_Z_idx = line_Z_idx[line_indices]
    filt_ion = line_ion[line_indices]
    filt_lvl_lower = line_lvl_lower[line_indices]
    filt_lvl_upper = line_lvl_upper[line_indices]
    filt_line_id = line_id_all[line_indices]
    filt_block1 = line_block1[line_indices]

    nu_col = filt_block1[:, 3]  # nu

    # Sort by descending nu (required for transport binary search)
    sort_order = np.argsort(-nu_col)

    # Build mapping: HDF5 line_id -> new sorted index
    line_id_to_sorted_idx = {}
    for new_sorted_idx, raw_idx in enumerate(sort_order):
        hdf5_line_id = int(filt_line_id[raw_idx])
        line_id_to_sorted_idx[hdf5_line_id] = new_sorted_idx

    n_lines = len(sort_order)
    print(f"  Sorted {n_lines} lines by descending nu")

    # Write line_list.csv
    with open(os.path.join(OUT_DIR, "line_list.csv"), "w") as fp:
        fp.write("atomic_number,ion_number,level_number_lower,level_number_upper,"
                 "line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm\n")
        for new_idx, raw_idx in enumerate(sort_order):
            z_real = int(filt_Z_idx[raw_idx]) + 1
            ion = int(filt_ion[raw_idx])
            ll = int(filt_lvl_lower[raw_idx])
            lu = int(filt_lvl_upper[raw_idx])
            lid = int(filt_line_id[raw_idx])
            b1 = filt_block1[raw_idx]
            wl = b1[0]       # wavelength (Angstrom)
            f_ul = b1[1]
            f_lu = b1[2]
            nu = b1[3]
            B_lu = b1[4]
            B_ul = b1[5]
            A_ul = b1[6]
            wl_cm = wl * 1e-8
            fp.write(f"{z_real},{ion},{ll},{lu},{lid},{wl},"
                     f"{f_ul},{f_lu},{nu},{B_lu},{B_ul},{A_ul},{wl_cm}\n")

    for z in ELEMENTS:
        z_idx = z - 1
        n = np.sum(filt_Z_idx == z_idx)
        print(f"    Z={z:2d} ({ELEMENT_NAMES[z]:2s}): {n} lines")

    # ========================================================
    # 3. MACRO-ATOM DATA (with index remapping)
    # ========================================================
    print("\n=== Extracting macro-atom data ===")
    ma_b0 = f["macro_atom_data/block0_values"][:]  # int64: [Z_real, ion, src_lvl, dst_lvl, type, line_id]
    ma_b1 = f["macro_atom_data/block1_values"][:]  # float64: [transition_probability]

    # macro_atom_data uses REAL Z in block0[:,0]
    ma_mask = np.isin(ma_b0[:, 0], ELEMENTS)
    ma_indices = np.where(ma_mask)[0]
    n_transitions = len(ma_indices)
    print(f"  Macro transitions: {n_transitions} / {len(ma_b0)} selected")

    # We need to order transitions by level (matching macro_atom_references order)
    # Transitions are grouped by source level: (Z, ion, source_level_number)
    # The order should match the level order in levels.csv

    # Build list of (source_level_key, original_index) for sorting
    ma_filt = ma_b0[ma_indices]
    ma_prob = ma_b1[ma_indices, 0]

    # Sort transitions to match level ordering in levels.csv
    # Group by (Z_real, ion, source_level) and order groups by level_lookup key
    # First, assign each transition its source level's new index
    src_level_new_idx = np.zeros(n_transitions, dtype=np.int64)
    for i in range(n_transitions):
        z_real = int(ma_filt[i, 0])
        ion = int(ma_filt[i, 1])
        src_lvl = int(ma_filt[i, 2])
        z_idx = z_real - 1
        key = (z_idx, ion, src_lvl)
        if key in level_lookup:
            src_level_new_idx[i] = level_lookup[key]
        else:
            src_level_new_idx[i] = -1  # orphan transition

    # Remove transitions referencing missing levels
    valid = src_level_new_idx >= 0
    n_orphan = np.sum(~valid)
    if n_orphan > 0:
        print(f"  WARNING: {n_orphan} transitions reference missing levels, skipping")
    ma_filt = ma_filt[valid]
    ma_prob = ma_prob[valid]
    src_level_new_idx = src_level_new_idx[valid]
    n_transitions = len(ma_filt)

    # Sort by source level new index (preserving relative order within each level)
    sort_by_src = np.argsort(src_level_new_idx, kind='stable')
    ma_filt = ma_filt[sort_by_src]
    ma_prob = ma_prob[sort_by_src]
    src_level_new_idx = src_level_new_idx[sort_by_src]

    # Write macro_atom_data.csv with remapped indices
    print(f"  Writing {n_transitions} transitions...")
    with open(os.path.join(OUT_DIR, "macro_atom_data.csv"), "w") as fp:
        fp.write(",atomic_number,ion_number,source_level_number,"
                 "destination_level_number,transition_type,"
                 "transition_probability,transition_line_id,"
                 "lines_idx,destination_level_idx,source_level_idx\n")
        for i in range(n_transitions):
            z_real = int(ma_filt[i, 0])
            ion = int(ma_filt[i, 1])
            src_lvl = int(ma_filt[i, 2])
            dst_lvl = int(ma_filt[i, 3])
            trans_type = int(ma_filt[i, 4])
            hdf5_line_id = int(ma_filt[i, 5])
            prob = ma_prob[i]

            # Remap line index
            lines_idx = line_id_to_sorted_idx.get(hdf5_line_id, -1)

            # Remap level indices
            z_idx = z_real - 1
            src_key = (z_idx, ion, src_lvl)
            dst_key = (z_idx, ion, dst_lvl)
            src_level_idx = level_lookup.get(src_key, -1)
            dst_level_idx = level_lookup.get(dst_key, -1)

            fp.write(f"{i},{z_real},{ion},{src_lvl},{dst_lvl},{trans_type},"
                     f"{prob},{hdf5_line_id},{lines_idx},"
                     f"{dst_level_idx},{src_level_idx}\n")

    for z in ELEMENTS:
        n = np.sum(ma_filt[:, 0] == z)
        print(f"    Z={z:2d} ({ELEMENT_NAMES[z]:2s}): {n} transitions")

    # ========================================================
    # 4. MACRO-ATOM REFERENCES (block offsets per level)
    # ========================================================
    print("\n=== Building macro-atom references ===")
    ref_Z_idx = f["macro_atom_references/axis1_label0"][:]
    ref_ion = f["macro_atom_references/axis1_label1"][:]
    ref_lvl = f["macro_atom_references/axis1_label2"][:]
    ref_b0 = f["macro_atom_references/block0_values"][:]  # [count_down, count_up, count_total]

    # Build references in same order as levels.csv
    # Count transitions per level from the sorted macro_atom_data
    level_trans_count = np.zeros(n_levels, dtype=np.int64)
    for i in range(n_transitions):
        level_trans_count[src_level_new_idx[i]] += 1

    # Also get count_down and count_up from HDF5 references
    # But we need them in our new level order
    block_refs = np.zeros(n_levels, dtype=np.int64)
    cumulative = 0
    count_down_arr = np.zeros(n_levels, dtype=np.int64)
    count_up_arr = np.zeros(n_levels, dtype=np.int64)
    count_total_arr = np.zeros(n_levels, dtype=np.int64)

    for new_idx, old_idx in enumerate(lev_indices):
        z_idx_val = int(lev_Z_idx[old_idx])
        ion_val = int(lev_ion[old_idx])
        lvl_val = int(lev_num[old_idx])

        # Find this level in HDF5 macro_atom_references
        # references have same axis labels as levels
        ref_match = np.where((ref_Z_idx == z_idx_val) &
                             (ref_ion == ion_val) &
                             (ref_lvl == lvl_val))[0]
        if len(ref_match) > 0:
            ridx = ref_match[0]
            count_down_arr[new_idx] = int(ref_b0[ridx, 0])
            count_up_arr[new_idx] = int(ref_b0[ridx, 1])
            count_total_arr[new_idx] = int(ref_b0[ridx, 2])

        block_refs[new_idx] = cumulative
        cumulative += level_trans_count[new_idx]

    # Verify
    print(f"  Total transitions from block_refs: {cumulative} (expected {n_transitions})")

    with open(os.path.join(OUT_DIR, "macro_atom_references.csv"), "w") as fp:
        fp.write("atomic_number,ion_number,source_level_number,"
                 "count_down,count_up,count_total,"
                 "block_references,references_idx\n")
        for new_idx, old_idx in enumerate(lev_indices):
            z_real = int(lev_Z_idx[old_idx]) + 1
            ion_val = int(lev_ion[old_idx])
            lvl_val = int(lev_num[old_idx])
            fp.write(f"{z_real},{ion_val},{lvl_val},"
                     f"{count_down_arr[new_idx]},{count_up_arr[new_idx]},"
                     f"{count_total_arr[new_idx]},"
                     f"{block_refs[new_idx]},{new_idx}\n")

    print(f"  Macro references: {n_levels} levels")

    # ========================================================
    # 5. IONIZATION ENERGIES
    # ========================================================
    print("\n=== Extracting ionization energies ===")
    iz = f["ionization_data/index_label0"][:]  # 0-indexed Z
    ii = f["ionization_data/index_label1"][:]
    iv = f["ionization_data/values"][:]

    mask_ion = np.isin(iz, ELEMENTS_IDX)
    ion_indices = np.where(mask_ion)[0]
    print(f"  Ionization: {len(ion_indices)} / {len(iz)} selected")

    with open(os.path.join(OUT_DIR, "ionization_energies.csv"), "w") as fp:
        fp.write("atomic_number,ion_number,ionization_energy_eV\n")
        for i in ion_indices:
            z_real = int(iz[i]) + 1
            fp.write(f"{z_real},{int(ii[i])},{iv[i]:.10f}\n")

    # ========================================================
    # 6. ZETA DATA
    # ========================================================
    print("\n=== Extracting zeta data ===")
    zeta_temps = f["zeta_data/axis0"][:]
    zz = f["zeta_data/axis1_label0"][:]  # 0-indexed Z
    zi = f["zeta_data/axis1_label1"][:]
    zeta_vals = f["zeta_data/block0_values"][:]

    mask_zeta = np.isin(zz, ELEMENTS_IDX)
    zeta_indices = np.where(mask_zeta)[0]
    print(f"  Zeta: {len(zeta_indices)} / {len(zz)} selected")

    zeta_selected = zeta_vals[zeta_indices]
    np.save(os.path.join(OUT_DIR, "zeta_data.npy"), zeta_selected)

    with open(os.path.join(OUT_DIR, "zeta_ions.csv"), "w") as fp:
        fp.write("atomic_number,ion_number\n")
        for i in zeta_indices:
            z_real = int(zz[i]) + 1
            fp.write(f"{z_real},{int(zi[i])}\n")

    with open(os.path.join(OUT_DIR, "zeta_temps.csv"), "w") as fp:
        fp.write("temperature\n")
        for t in zeta_temps:
            fp.write(f"{t:.1f}\n")

    # ========================================================
    # 7. ATOM MASSES
    # ========================================================
    print("\n=== Extracting atom masses ===")
    atom_z = f["atom_data/axis1"][:]  # real Z (1-indexed)
    atom_mass = f["atom_data/block0_values"][:, 0]

    with open(os.path.join(OUT_DIR, "atom_masses.csv"), "w") as fp:
        fp.write("atomic_number,mass_amu\n")
        for z_real in ELEMENTS:
            idx = np.where(atom_z == z_real)[0][0]
            fp.write(f"{z_real},{atom_mass[idx]:.10f}\n")

    # ========================================================
    # 8. ABUNDANCES
    # ========================================================
    print("\n=== Writing abundances ===")
    with open(os.path.join(OUT_DIR, "abundances.csv"), "w") as fp:
        header = "atomic_number," + ",".join(str(i) for i in range(N_SHELLS))
        fp.write(header + "\n")
        for z_real in ELEMENTS:
            x = DEFAULT_ABUNDANCES[z_real]
            vals = ",".join(f"{x}" for _ in range(N_SHELLS))
            fp.write(f"{z_real},{vals}\n")

    # ========================================================
    # 9. NPY FILES
    # ========================================================
    print("\n=== Generating NPY files ===")

    # tau_sobolev [n_lines x n_shells] — zeros, recomputed at iter 0
    tau = np.zeros((n_lines, N_SHELLS), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "tau_sobolev.npy"), tau)
    print(f"  tau_sobolev: [{n_lines} x {N_SHELLS}]")

    # transition_probabilities [n_transitions x n_shells]
    # Initialize with equal branching: 1/n per level block
    trans_prob = np.zeros((n_transitions, N_SHELLS), dtype=np.float64)
    for lev_idx in range(n_levels):
        start = int(block_refs[lev_idx])
        count = int(level_trans_count[lev_idx])
        if count > 0:
            trans_prob[start:start + count, :] = 1.0 / count
    np.save(os.path.join(OUT_DIR, "transition_probabilities.npy"), trans_prob)
    print(f"  transition_probabilities: [{n_transitions} x {N_SHELLS}]")

    # j_blues [n_lines x n_shells] — W * B_nu(T_rad)
    try:
        W, T_rad = load_plasma_state()
        nu_sorted = np.zeros(n_lines)
        for new_idx, raw_idx in enumerate(sort_order):
            nu_sorted[new_idx] = filt_block1[raw_idx, 3]
        # B_nu(T_rad) = 2*h*nu^3/c^2 / (exp(h*nu/k*T)-1)
        h = 6.62607015e-27
        k = 1.380649e-16
        c = 2.99792458e10
        j_blues = np.zeros((n_lines, N_SHELLS))
        for s in range(N_SHELLS):
            for l in range(n_lines):
                x = h * nu_sorted[l] / (k * T_rad[s])
                if x < 500:
                    B_nu = 2 * h * nu_sorted[l]**3 / c**2 / (np.exp(x) - 1)
                else:
                    B_nu = 0.0
                j_blues[l, s] = W[s] * B_nu
        np.save(os.path.join(OUT_DIR, "j_blues.npy"), j_blues)
        print(f"  j_blues: [{n_lines} x {N_SHELLS}]")
    except Exception as e:
        print(f"  j_blues: SKIPPED ({e}), will use W*B_nu fallback")
        j_blues = np.zeros((n_lines, N_SHELLS))
        np.save(os.path.join(OUT_DIR, "j_blues.npy"), j_blues)

    # line2macro_level_upper [n_lines] — maps each line to its upper level macro index
    line2macro = np.full(n_lines, -1, dtype=np.int64)
    for new_idx, raw_idx in enumerate(sort_order):
        z_idx = int(filt_Z_idx[raw_idx])
        ion = int(filt_ion[raw_idx])
        lvl_upper = int(filt_lvl_upper[raw_idx])
        key = (z_idx, ion, lvl_upper)
        if key in level_lookup:
            line2macro[new_idx] = level_lookup[key]
    np.save(os.path.join(OUT_DIR, "line2macro_level_upper.npy"), line2macro)
    n_mapped = np.sum(line2macro >= 0)
    print(f"  line2macro_level_upper: [{n_lines}], {n_mapped} mapped")

    # line_interaction_id [n_lines] — sequential
    np.save(os.path.join(OUT_DIR, "line_interaction_id.npy"),
            np.arange(n_lines, dtype=np.int64))
    print(f"  line_interaction_id: [{n_lines}]")

    f.close()

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(ELEMENTS)} elements, {n_levels} levels, "
          f"{n_lines} lines, {n_transitions} transitions")
    print(f"Files written to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
