#!/usr/bin/env python3
"""Check impact of neutral-species tau errors on transport.
The key question: do neutrals matter for the spectrum?"""
import numpy as np

REF = "data/tardis_reference"

# Load TARDIS reference tau
tau_ref = np.load(f"{REF}/tau_sobolev.npy")
print(f"tau_sobolev shape: {tau_ref.shape}")

# Load LUMINA computed tau
val = np.genfromtxt("lumina_tau_validation.csv", delimiter=',', skip_header=2)
tau_lumina_s0 = val[:, 1]  # shell 0

# Load line data
line_list = np.genfromtxt(f"{REF}/line_list.csv", delimiter=',', names=True)
Z_line = line_list['atomic_number'].astype(int)
ion_line = line_list['ion_number'].astype(int)

ELEMENTS = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}

# Compare tau statistics by ion stage
print(f"\n{'Ion Stage':>12} {'N_lines':>8} {'sum(tau_T)':>12} {'sum(tau_L)':>12} {'frac_tau%':>10} {'med_ratio':>10}")
print("-" * 70)

tau_T_s0 = tau_ref[:, 0]
mask_valid = tau_T_s0 > 1e-10

total_tau_T = np.sum(tau_T_s0[mask_valid])

for stage in range(6):
    smask = mask_valid & (ion_line == stage)
    n = np.sum(smask)
    if n == 0:
        continue
    sum_T = np.sum(tau_T_s0[smask])
    sum_L = np.sum(tau_lumina_s0[smask])
    frac = sum_T / total_tau_T * 100
    med_ratio = np.median(tau_lumina_s0[smask] / tau_T_s0[smask])
    print(f"  ion={stage:>2d}     {n:8d} {sum_T:12.4e} {sum_L:12.4e} {frac:10.2f} {med_ratio:10.4f}")

# By element, only ion=0 (neutrals)
print(f"\nNeutral species breakdown:")
print(f"{'Element':>8} {'N_lines':>8} {'sum(tau_T)':>12} {'frac%':>8} {'Dominant?':>10}")
print("-" * 50)
for z in sorted(ELEMENTS.keys()):
    smask = mask_valid & (Z_line == z) & (ion_line == 0)
    n = np.sum(smask)
    if n == 0:
        continue
    sum_T = np.sum(tau_T_s0[smask])
    frac = sum_T / total_tau_T * 100
    dominant = "YES" if frac > 1.0 else "no"
    print(f"{ELEMENTS[z]:>8} {n:8d} {sum_T:12.4e} {frac:8.4f} {dominant:>10}")

# Key question: tau-weighted error
print(f"\n--- TAU-WEIGHTED ERROR (shell 0) ---")
tau_weights = tau_T_s0[mask_valid]
ratios = tau_lumina_s0[mask_valid] / tau_T_s0[mask_valid]
weighted_mean_ratio = np.average(ratios, weights=tau_weights)
print(f"  Unweighted median ratio: {np.median(ratios):.4f}")
print(f"  Tau-weighted mean ratio: {weighted_mean_ratio:.4f}")
print(f"  Tau-weighted mean error: {(weighted_mean_ratio - 1)*100:+.2f}%")

# Lines with tau > 1 (these actually matter for transport)
big_mask = mask_valid & (tau_T_s0 > 1.0)
big_ratios = tau_lumina_s0[big_mask] / tau_T_s0[big_mask]
print(f"\n  Lines with tau > 1: {np.sum(big_mask)}")
print(f"  Median ratio (tau>1): {np.median(big_ratios):.4f}")
print(f"  P10-P90 (tau>1): [{np.percentile(big_ratios, 10):.4f}, {np.percentile(big_ratios, 90):.4f}]")

# Lines with tau > 100 (strongly saturated, exact value less important)
huge_mask = mask_valid & (tau_T_s0 > 100.0)
huge_ratios = tau_lumina_s0[huge_mask] / tau_T_s0[huge_mask]
print(f"\n  Lines with tau > 100: {np.sum(huge_mask)}")
if np.sum(huge_mask) > 0:
    print(f"  Median ratio (tau>100): {np.median(huge_ratios):.4f}")

# Check across all shells
print(f"\n--- ACROSS ALL SHELLS (lines with tau > 1) ---")
print(f"{'Shell':>5} {'N_big':>6} {'med_ratio':>10} {'P10':>8} {'P90':>8}")
print("-" * 45)
for s in [0, 5, 10, 15, 20, 25, 29]:
    tau_T = tau_ref[:, s]
    tau_L_data = np.genfromtxt("lumina_tau_validation.csv", delimiter=',', skip_header=2, usecols=(s+1,))
    bmask = tau_T > 1.0
    if np.sum(bmask) == 0:
        continue
    brat = tau_L_data[bmask] / tau_T[bmask]
    print(f"{s:5d} {np.sum(bmask):6d} {np.median(brat):10.4f} {np.percentile(brat, 10):8.4f} {np.percentile(brat, 90):8.4f}")
