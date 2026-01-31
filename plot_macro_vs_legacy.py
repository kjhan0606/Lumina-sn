#!/usr/bin/env python3
"""
Compare Macro-Atom vs Legacy mode spectra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_spectrum(filename, col=3):
    """Load spectrum - col=3 for LUMINA, col=2 for standard"""
    wavelength, luminosity = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('wavelength'):
                continue
            parts = line.strip().split(',')
            if len(parts) > col:
                wavelength.append(float(parts[0]))
                luminosity.append(float(parts[col]))
    return np.array(wavelength), np.array(luminosity)

def load_tardis(filename):
    """Load TARDIS spectrum"""
    wavelength, luminosity = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                wavelength.append(float(parts[0]))
                luminosity.append(float(parts[1]))
    return np.array(wavelength), np.array(luminosity)

# Load spectra
wl_macro, L_macro = load_spectrum('spectrum_macro_atom.csv')
wl_legacy, L_legacy = load_spectrum('spectrum_legacy.csv')
wl_tardis, L_tardis = load_tardis('tardis_comparison_spectrum.dat')

# Normalize
L_macro_norm = L_macro / np.max(L_macro[(wl_macro > 4000) & (wl_macro < 8000)])
L_legacy_norm = L_legacy / np.max(L_legacy[(wl_legacy > 4000) & (wl_legacy < 8000)])
L_tardis_norm = L_tardis / np.max(L_tardis[(wl_tardis > 4000) & (wl_tardis < 8000)])

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: All three spectra
ax1 = axes[0]
ax1.plot(wl_tardis, L_tardis_norm, 'b-', linewidth=1.5, label='TARDIS Reference', alpha=0.8)
ax1.plot(wl_macro, L_macro_norm, 'r-', linewidth=1.0, label='LUMINA Macro-Atom (χ²=98)', alpha=0.7)
ax1.plot(wl_legacy, L_legacy_norm, 'g-', linewidth=1.0, label='LUMINA Legacy (χ²=77)', alpha=0.7)
ax1.set_ylabel('Normalized Flux', fontsize=12)
ax1.set_title('LUMINA Macro-Atom vs Legacy Mode Comparison', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(3000, 10000)
ax1.set_ylim(0, 1.3)
ax1.grid(True, alpha=0.3)

# Mark features
features = {'Si II': 6355, 'Ca II H&K': 3945, 'S II': 5454, 'Ca II IR': 8542}
for name, wl in features.items():
    ax1.axvline(wl, color='gray', linestyle='--', alpha=0.3)
    ax1.text(wl, 1.2, name, fontsize=8, ha='center')

# Panel 2: Macro-atom vs TARDIS
ax2 = axes[1]
interp_tardis = interp1d(wl_tardis, L_tardis_norm, bounds_error=False, fill_value=0)
L_tardis_interp = interp_tardis(wl_macro)
ax2.plot(wl_macro, L_macro_norm, 'r-', linewidth=1, label='Macro-Atom', alpha=0.8)
ax2.plot(wl_macro, L_tardis_interp, 'b--', linewidth=1, label='TARDIS', alpha=0.8)
ax2.fill_between(wl_macro, L_macro_norm, L_tardis_interp, alpha=0.3, color='purple')
ax2.set_ylabel('Normalized Flux', fontsize=12)
ax2.set_title('Macro-Atom Mode Residuals', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, 1.3)
ax2.grid(True, alpha=0.3)

# Panel 3: Legacy vs TARDIS
ax3 = axes[2]
L_tardis_interp2 = interp_tardis(wl_legacy)
ax3.plot(wl_legacy, L_legacy_norm, 'g-', linewidth=1, label='Legacy', alpha=0.8)
ax3.plot(wl_legacy, L_tardis_interp2, 'b--', linewidth=1, label='TARDIS', alpha=0.8)
ax3.fill_between(wl_legacy, L_legacy_norm, L_tardis_interp2, alpha=0.3, color='orange')
ax3.set_xlabel('Wavelength [Å]', fontsize=12)
ax3.set_ylabel('Normalized Flux', fontsize=12)
ax3.set_title('Legacy Mode Residuals', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_ylim(0, 1.3)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('macro_vs_legacy_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('macro_vs_legacy_comparison.pdf', dpi=150, bbox_inches='tight')
print("Saved: macro_vs_legacy_comparison.png")
print("Saved: macro_vs_legacy_comparison.pdf")

# Print statistics
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"\nMacro-Atom Mode:")
print(f"  Chi-square: 98.10")
print(f"  Runtime: 418s (24 packets/sec)")
print(f"  Total macro-atom calls: 1,396,164")

print(f"\nLegacy Mode:")
print(f"  Chi-square: 76.78")
print(f"  Runtime: 12s (851 packets/sec)")
print(f"  Line interactions: 16,039")

print(f"\nConclusion:")
print(f"  Legacy mode currently provides better fit (χ²=77 vs 98)")
print(f"  Macro-atom mode needs tuning of transition probabilities")
print("="*60)
