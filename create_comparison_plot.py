#!/usr/bin/env python3
"""
Create TARDIS vs LUMINA spectrum comparison plot.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_tardis_spectrum(filepath):
    """Load TARDIS spectrum from file"""
    data = np.loadtxt(filepath, comments='#')
    wl = data[:, 0]
    flux = data[:, 1]
    # Sort by wavelength
    idx = np.argsort(wl)
    return wl[idx], flux[idx]

def load_lumina_spectrum():
    """Load LUMINA spectrum from file"""
    data = []
    with open('lumina_comparison_spectrum.dat', 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('wavelength'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                wl = float(parts[0])
                L_lumina = float(parts[3])  # LUMINA rotated column
                data.append([wl, L_lumina])
    data = np.array(data)
    idx = np.argsort(data[:, 0])
    return data[idx, 0], data[idx, 1]

# Load spectra
print("Loading spectra...")
wl_tardis, flux_tardis = load_tardis_spectrum('/home/kjhan/TARDIS-SN/tardis_spectrum.dat')
wl_lumina, flux_lumina = load_lumina_spectrum()

# Filter to common range
mask_t = (wl_tardis >= 3000) & (wl_tardis <= 10000)
mask_l = (wl_lumina >= 3000) & (wl_lumina <= 10000)

wl_tardis = wl_tardis[mask_t]
flux_tardis = flux_tardis[mask_t]
wl_lumina = wl_lumina[mask_l]
flux_lumina = flux_lumina[mask_l]

print(f"TARDIS: {len(wl_tardis)} points, {wl_tardis.min():.0f}-{wl_tardis.max():.0f} Å")
print(f"LUMINA: {len(wl_lumina)} points, {wl_lumina.min():.0f}-{wl_lumina.max():.0f} Å")

# Normalize
flux_tardis_norm = flux_tardis / np.max(flux_tardis)
flux_lumina_norm = flux_lumina / np.max(flux_lumina)

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: TARDIS spectrum
ax1 = axes[0]
ax1.plot(wl_tardis, flux_tardis, 'b-', lw=0.8, label='TARDIS')
ax1.set_ylabel('L$_λ$ (erg/s/Å)', fontsize=11)
ax1.set_yscale('log')
ax1.set_ylim(1e37, 1e42)
ax1.legend(loc='upper right', fontsize=11)
ax1.set_title('TARDIS vs LUMINA Spectrum Comparison (Type Ia SN)', fontsize=14)
ax1.axvline(6355, color='green', ls='--', alpha=0.7, lw=1, label='Si II 6355')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(3000, 10000)

# Panel 2: LUMINA spectrum
ax2 = axes[1]
ax2.plot(wl_lumina, flux_lumina, 'r-', lw=0.8, label='LUMINA')
ax2.set_ylabel('L$_λ$ (erg/s/Å)', fontsize=11)
ax2.set_yscale('log')
ax2.set_ylim(1e37, 1e42)
ax2.legend(loc='upper right', fontsize=11)
ax2.axvline(6355, color='green', ls='--', alpha=0.7, lw=1)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(3000, 10000)

# Panel 3: Normalized comparison overlay
ax3 = axes[2]
ax3.plot(wl_tardis, flux_tardis_norm, 'b-', lw=1, alpha=0.8, label='TARDIS (normalized)')
ax3.plot(wl_lumina, flux_lumina_norm, 'r-', lw=1, alpha=0.8, label='LUMINA (normalized)')
ax3.set_xlabel('Wavelength (Å)', fontsize=12)
ax3.set_ylabel('Normalized Flux', fontsize=11)
ax3.legend(loc='upper right', fontsize=11)
ax3.axvline(6355, color='green', ls='--', alpha=0.7, lw=1)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(3000, 10000)
ax3.set_ylim(0, 1.1)

# Mark spectral features
features = [
    (3934, 'Ca II'),
    (4100, 'Hδ'),
    (4300, 'Fe II'),
    (4861, 'Hβ'),
    (5169, 'Fe II'),
    (5890, 'Na I'),
    (6355, 'Si II'),
    (6563, 'Hα'),
    (8542, 'Ca II IR'),
]

for wl_feat, name in features:
    ax3.axvline(wl_feat, color='gray', ls=':', alpha=0.4, lw=0.5)
    if wl_feat in [6355, 3934, 5890]:
        ax3.text(wl_feat+50, 0.95, name, fontsize=8, rotation=0, va='top')

plt.tight_layout()
plt.savefig('tardis_lumina_spectrum_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: tardis_lumina_spectrum_comparison.png")

# Print statistics
print("\n" + "="*60)
print("SPECTRUM STATISTICS")
print("="*60)

print(f"\nTARDIS:")
print(f"  Peak: {flux_tardis.max():.3e} erg/s/Å at {wl_tardis[np.argmax(flux_tardis)]:.0f} Å")
print(f"  Mean: {flux_tardis.mean():.3e} erg/s/Å")

print(f"\nLUMINA:")
print(f"  Peak: {flux_lumina.max():.3e} erg/s/Å at {wl_lumina[np.argmax(flux_lumina)]:.0f} Å")
print(f"  Mean: {flux_lumina.mean():.3e} erg/s/Å")

# Check Si II region
si_mask_t = (wl_tardis > 5900) & (wl_tardis < 6500)
si_mask_l = (wl_lumina > 5900) & (wl_lumina < 6500)

if np.any(si_mask_t) and np.any(si_mask_l):
    print(f"\nSi II region (5900-6500 Å):")
    print(f"  TARDIS min: {flux_tardis[si_mask_t].min():.3e} at {wl_tardis[si_mask_t][np.argmin(flux_tardis[si_mask_t])]:.0f} Å")
    print(f"  LUMINA min: {flux_lumina[si_mask_l].min():.3e} at {wl_lumina[si_mask_l][np.argmin(flux_lumina[si_mask_l])]:.0f} Å")
