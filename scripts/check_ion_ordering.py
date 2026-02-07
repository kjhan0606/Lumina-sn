#!/usr/bin/env python3
"""Check ion ordering in TARDIS ion_number_density.npy"""
import numpy as np

REF = "data/tardis_reference"
ion_npy = np.load(f"{REF}/ion_number_density.npy")
ioniz = np.genfromtxt(f"{REF}/ionization_energies.csv", delimiter=',', names=True)

print(f"ion_number_density shape: {ion_npy.shape}")
print(f"N rows = {ion_npy.shape[0]}, N shells = {ion_npy.shape[1]}")

# Build our ordering
ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]
ELEM_NAMES = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}

ion_map = []
for z in ELEMENTS:
    mask = ioniz['atomic_number'].astype(int) == z
    n_ioniz = np.sum(mask)
    for stage in range(n_ioniz + 1):
        ion_map.append((z, stage))

print(f"\nOur ion ordering ({len(ion_map)} total):")
print(f"{'idx':>5} {'Z':>3} {'ion':>4} {'n_ion(s0)':>14}")
print("-" * 30)
for i, (z, st) in enumerate(ion_map):
    val = ion_npy[i, 0]
    name = f"{ELEM_NAMES[z]} {st}"
    if val > 1.0:
        print(f"{i:5d} {z:3d} {st:4d} {val:14.6e}  {name}")

# Also print the DOMINANT ions
print("\nDominant ions (n > 1e4):")
for i, (z, st) in enumerate(ion_map):
    val = ion_npy[i, 0]
    if val > 1e4:
        print(f"  {ELEM_NAMES[z]} {['I','II','III','IV','V'][st]:>4}: {val:.6e}")

# Check Fe specifically
print("\nFe ions (all):")
for i, (z, st) in enumerate(ion_map):
    if z == 26:
        val = ion_npy[i, 0]
        print(f"  idx={i:3d}: Fe {['I','II','III','IV','V','VI','VII'][st]:>4} = {val:.6e}")

# Check what lumina_ion_density.csv says
print("\nFrom lumina_ion_density.csv (LUMINA's output):")
import csv
with open("lumina_ion_density.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        z = int(row['Z'])
        ion_st = int(row['ion_stage'])
        n = float(row['n_ion_s0'])
        if z == 26 and n > 1e-10:
            print(f"  Fe {['I','II','III','IV','V','VI','VII'][ion_st]:>4} = {n:.6e}")
