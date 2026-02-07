#!/usr/bin/env python3
"""Extract atomic data from HDF5 for LUMINA plasma solver.

Reads data/atomic/kurucz_cd23_chianti_H_He.h5 and exports CSV/NPY files
to data/tardis_reference/ for the 8 elements used in the simulation:
C=6, O=8, Si=14, S=16, Ca=20, Fe=26, Co=27, Ni=28.

HDF5 uses 0-indexed Z (0=H, 1=He, ..., 5=C, 7=O, 13=Si, etc.)
Output CSVs use real atomic numbers (6=C, 8=O, 14=Si, etc.)
"""

import h5py
import numpy as np
import os

HDF5_PATH = "data/atomic/kurucz_cd23_chianti_H_He.h5"
OUT_DIR = "data/tardis_reference"

# Our 8 elements: real Z values
ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]
# HDF5 0-indexed: Z_real - 1
ELEMENTS_IDX = [z - 1 for z in ELEMENTS]

def main():
    f = h5py.File(HDF5_PATH, "r")

    # --- 1. Levels data ---
    Z_idx = f["levels_data/axis1_label0"][:]
    ion = f["levels_data/axis1_label1"][:]
    level_num = f["levels_data/axis1_label2"][:]
    energy = f["levels_data/block0_values"][:, 0]
    g = f["levels_data/block1_values"][:, 0]
    metastable = f["levels_data/block2_values"][:, 0]

    mask = np.isin(Z_idx, ELEMENTS_IDX)
    print(f"Levels: {mask.sum()} / {len(Z_idx)} selected")

    with open(os.path.join(OUT_DIR, "levels.csv"), "w") as fp:
        fp.write("atomic_number,ion_number,level_number,energy_eV,g,metastable\n")
        for i in np.where(mask)[0]:
            z_real = int(Z_idx[i]) + 1
            fp.write(f"{z_real},{int(ion[i])},{int(level_num[i])},{energy[i]:.10f},{int(g[i])},{int(metastable[i])}\n")

    # --- 2. Ionization energies ---
    iz = f["ionization_data/index_label0"][:]
    ii = f["ionization_data/index_label1"][:]
    iv = f["ionization_data/values"][:]

    mask_ion = np.isin(iz, ELEMENTS_IDX)
    print(f"Ionization: {mask_ion.sum()} / {len(iz)} selected")

    with open(os.path.join(OUT_DIR, "ionization_energies.csv"), "w") as fp:
        fp.write("atomic_number,ion_number,ionization_energy_eV\n")
        for i in np.where(mask_ion)[0]:
            z_real = int(iz[i]) + 1
            fp.write(f"{z_real},{int(ii[i])},{iv[i]:.10f}\n")

    # --- 3. Zeta data ---
    zeta_temps = f["zeta_data/axis0"][:]
    zz = f["zeta_data/axis1_label0"][:]
    zi = f["zeta_data/axis1_label1"][:]
    zeta_vals = f["zeta_data/block0_values"][:]

    mask_zeta = np.isin(zz, ELEMENTS_IDX)
    print(f"Zeta: {mask_zeta.sum()} / {len(zz)} selected")

    zeta_selected = zeta_vals[mask_zeta]
    np.save(os.path.join(OUT_DIR, "zeta_data.npy"), zeta_selected)

    # Zeta ion mapping
    with open(os.path.join(OUT_DIR, "zeta_ions.csv"), "w") as fp:
        fp.write("atomic_number,ion_number\n")
        for i in np.where(mask_zeta)[0]:
            z_real = int(zz[i]) + 1
            fp.write(f"{z_real},{int(zi[i])}\n")

    # Zeta temps
    with open(os.path.join(OUT_DIR, "zeta_temps.csv"), "w") as fp:
        fp.write("temperature\n")
        for t in zeta_temps:
            fp.write(f"{t:.1f}\n")

    # --- 4. Atom masses ---
    atom_z = f["atom_data/axis1"][:]  # 1-indexed (1=H, 2=He, ..., 30=Zn)
    atom_mass = f["atom_data/block0_values"][:, 0]

    with open(os.path.join(OUT_DIR, "atom_masses.csv"), "w") as fp:
        fp.write("atomic_number,mass_amu\n")
        for z_real in ELEMENTS:
            idx = np.where(atom_z == z_real)[0][0]
            fp.write(f"{z_real},{atom_mass[idx]:.10f}\n")

    f.close()
    print("Done. Files written to", OUT_DIR)

if __name__ == "__main__":
    main()
