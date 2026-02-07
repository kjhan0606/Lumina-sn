#!/usr/bin/env python3
"""Compare C code tau output vs TARDIS reference and Python reference."""
import numpy as np

REF = "data/tardis_reference"
tau_ref = np.load(f"{REF}/tau_sobolev.npy")
line_list = np.genfromtxt(f"{REF}/line_list.csv", delimiter=',', names=True)

# Load C code output
c_tau = []
with open("lumina_tau_validation.csv") as f:
    for line in f:
        if line.startswith('#') or line.startswith('line'):
            continue
        parts = line.strip().split(',')
        if len(parts) >= 2:
            c_tau.append(float(parts[1]))
c_tau = np.array(c_tau)

n_lines = min(len(c_tau), tau_ref.shape[0])
print(f"Lines in C output: {len(c_tau)}")
print(f"Lines in TARDIS ref: {tau_ref.shape[0]}")

# Compare shell 0
tau_c_s0 = c_tau[:n_lines]
tau_t_s0 = tau_ref[:n_lines, 0]

# Significant lines
mask = tau_t_s0 > 1e-10
ratios = tau_c_s0[mask] / tau_t_s0[mask]

print(f"\nOverall tau comparison (shell 0, {np.sum(mask)} lines with tau>1e-10):")
print(f"  Median ratio: {np.median(ratios):.6f}")
print(f"  Mean ratio: {np.mean(ratios):.6f}")
print(f"  P10-P90: [{np.percentile(ratios, 10):.6f}, {np.percentile(ratios, 90):.6f}]")

# By element
ELEM = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}
Z_line = line_list['atomic_number'][:n_lines].astype(int)
ion_line = line_list['ion_number'][:n_lines].astype(int)

print(f"\nBy element:")
for z in sorted(ELEM.keys()):
    elem_mask = mask & (Z_line == z)
    if np.sum(elem_mask) > 0:
        elem_ratios = tau_c_s0[elem_mask] / tau_t_s0[elem_mask]
        print(f"  {ELEM[z]:>2} (Z={z:2d}): N={np.sum(elem_mask):6d}, "
              f"median={np.median(elem_ratios):.6f}, "
              f"mean={np.mean(elem_ratios):.6f}, "
              f"P10={np.percentile(elem_ratios, 10):.6f}, "
              f"P90={np.percentile(elem_ratios, 90):.6f}")

# By (element, ion)
print(f"\nBy (element, ion):")
for z in sorted(ELEM.keys()):
    for ion in range(6):
        elem_ion_mask = mask & (Z_line == z) & (ion_line == ion)
        n_matched = np.sum(elem_ion_mask)
        if n_matched > 0:
            elem_ion_ratios = tau_c_s0[elem_ion_mask] / tau_t_s0[elem_ion_mask]
            print(f"  {ELEM[z]:>2} {ion} ({ELEM[z]} {'I'*(ion+1):>5}): "
                  f"N={n_matched:6d}, median={np.median(elem_ion_ratios):.6f}")

# n_e comparison
print(f"\nn_e LUMINA (from output): 1.8098e+09")
print(f"n_e TARDIS reference:    1.760117e+09")
print(f"n_e error: {(1.8098e9 - 1.760117e9) / 1.760117e9 * 100:.2f}%")

# Check specific lines
print(f"\nSi II 6355 line (idx 86362):")
print(f"  C code tau: {c_tau[86362]:.6e}")
print(f"  TARDIS tau: {tau_ref[86362, 0]:.6e}")
print(f"  Ratio: {c_tau[86362]/tau_ref[86362, 0]:.6f}")

# Check S I worst line (from earlier: idx 95312)
print(f"\nS I 9215A line (idx 95312):")
print(f"  C code tau: {c_tau[95312]:.6e}")
print(f"  TARDIS tau: {tau_ref[95312, 0]:.6e}")
print(f"  Ratio: {c_tau[95312]/tau_ref[95312, 0]:.6f}")
