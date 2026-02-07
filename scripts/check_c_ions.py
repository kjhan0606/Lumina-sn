#!/usr/bin/env python3
"""Quick check: what does the C code compute for Fe I?"""
import numpy as np

# Load LUMINA ion density output
# lumina_ion_density.csv has: ion_idx,Z,ion_stage,n_ion_s0,partition_s0

data = np.genfromtxt("lumina_ion_density.csv", delimiter=',', names=True, dtype=None, encoding='utf-8')
print("LUMINA ion densities (shell 0):")
print(f"{'idx':>4} {'Z':>3} {'ion':>4} {'n_ion':>14} {'Z_part':>14}")
for row in data:
    z = int(row['Z'])
    ion = int(row['ion_stage'])
    n = float(row['n_ion_s0'])
    p = float(row['partition_s0'])
    if n > 1e-10:
        print(f"{int(row['ion_idx']):4d} {z:3d} {ion:4d} {n:14.6e} {p:14.6e}")

# Compare with TARDIS
ion_npy = np.load("data/tardis_reference/ion_number_density.npy")
ioniz = np.genfromtxt("data/tardis_reference/ionization_energies.csv", delimiter=',', names=True)
ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]
NAMES = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}
ion_map = []
for z in ELEMENTS:
    mask = ioniz['atomic_number'].astype(int) == z
    n_ioniz = np.sum(mask)
    for stage in range(n_ioniz + 1):
        ion_map.append((z, stage))

print(f"\nKey comparison:")
print(f"{'Ion':>10} {'LUMINA':>14} {'TARDIS':>14} {'Ratio':>8}")
print("-" * 50)

# Match by (Z, ion_stage)
for row in data:
    z = int(row['Z'])
    ion = int(row['ion_stage'])
    n_lumina = float(row['n_ion_s0'])

    idx = -1
    for i, (zz, st) in enumerate(ion_map):
        if zz == z and st == ion:
            idx = i
            break
    if idx >= 0 and n_lumina > 1e-10:
        n_tardis = ion_npy[idx, 0]
        if n_tardis > 1e-10:
            ratio = n_lumina / n_tardis
            if abs(ratio - 1.0) > 0.01 or z == 26:
                print(f"{NAMES[z]:>2} {ion} ({NAMES[z]} {'I'*(ion+1):>5}): "
                      f"{n_lumina:14.6e} {n_tardis:14.6e} {ratio:8.4f}")
