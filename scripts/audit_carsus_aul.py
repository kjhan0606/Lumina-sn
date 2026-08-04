#!/usr/bin/env python3
"""#131 Phase B.1: A_ul audit for Si II 6347/Ca II HK/Fe II 5169 vs NIST.
Pulls lines from kurucz_cd23_cmfgen_lumina.h5 (the file LUMINA actually loads),
finds candidates near the rest wavelength, prints A_ul + comparison to NIST."""
import h5py, numpy as np

FN = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/kurucz_cd23_cmfgen_lumina.h5'

# (label, Z, ion(0=I), rest_A, window_A, NIST A_ul s^-1, NIST notes)
TARGETS = [
    ("Si II 6347 (4P°-4D 1/2)",  14, 1, 6347.11, 0.5, 6.62e7, "NIST log gf +0.297"),
    ("Si II 6371 (4P°-4D 3/2)",  14, 1, 6371.37, 0.5, 5.81e7, "NIST log gf -0.003"),
    ("Ca II K  3933",            20, 1, 3933.66, 0.5, 1.47e8, "NIST log gf +0.105"),
    ("Ca II H  3968",            20, 1, 3968.47, 0.5, 1.40e8, "NIST log gf -0.180"),
    ("Fe II 5169 (a4D-z4F)",     26, 1, 5169.03, 0.5, 1.73e7, "NIST log gf -1.18 (m42)"),
]

with h5py.File(FN, 'r') as f:
    ld = f['lines_data']
    cols    = [s.decode() for s in ld['block0_items'][:]]
    Z_map   = ld['axis1_level0'][:]   # idx -> Z
    ion_map = ld['axis1_level1'][:]   # idx -> ion
    Z_idx   = ld['axis1_label0'][:]   # per-line Z slot
    ion_idx = ld['axis1_label1'][:]   # per-line ion slot
    vals    = ld['block0_values'][:]  # (N, 7): wave, f_ul, f_lu, nu, B_lu, B_ul, A_ul
    wave = vals[:, cols.index('wavelength')]
    A    = vals[:, cols.index('A_ul')]
    f_lu = vals[:, cols.index('f_lu')]

print(f"\n=== Carsus A_ul audit vs NIST (file: kurucz_cd23_cmfgen_lumina.h5) ===\n")
print(f"  {'line':<26}  {'λ(Å)':<9}  {'carsus A_ul':<13}  {'NIST A_ul':<11}  {'ratio':<9}  {'log gf':<9}  {'notes'}")
print("-"*120)

for label, Z, ion, lam0, dl, A_NIST, note in TARGETS:
    Z_target_idx   = np.where(Z_map == Z)[0]
    ion_target_idx = np.where(ion_map == ion)[0]
    if len(Z_target_idx) == 0 or len(ion_target_idx) == 0:
        print(f"  {label:<26}  no Z/ion match in carsus"); continue
    m = (Z_idx == Z_target_idx[0]) & (ion_idx == ion_target_idx[0]) & \
        (wave > lam0 - dl) & (wave < lam0 + dl)
    if m.sum() == 0:
        print(f"  {label:<26}  {lam0:7.2f}  no line in ±{dl}Å"); continue
    # rank by A_ul descending (strongest member of multiplet)
    idx = np.where(m)[0]
    idx = idx[np.argsort(-A[idx])]
    for k, j in enumerate(idx[:3]):
        ratio = A[j] / A_NIST
        # g_l f_lu = g_u f_ul; log gf from f_lu (assuming g_l=2 doublet base)
        log_gf = np.log10(max(2 * f_lu[j], 1e-30))
        tag = label if k == 0 else " └─"
        print(f"  {tag:<26}  {wave[j]:7.2f}  {A[j]:.3e}    {A_NIST:.2e}    {ratio:7.3f}   {log_gf:+6.3f}    {note if k==0 else ''}")
print()
