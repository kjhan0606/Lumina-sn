#!/usr/bin/env python3
"""
Compare LUMINA virtual packet spectrum with TARDIS spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_lumina_vpacket_spectrum(filepath):
    """Load LUMINA virtual packet spectrum"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('wavelength'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                wl = float(parts[0])
                L_nu = float(parts[2])
                data.append([wl, L_nu])
    data = np.array(data)
    # Sort by wavelength
    idx = np.argsort(data[:, 0])
    return data[idx, 0], data[idx, 1]

def load_tardis_spectrum(filepath):
    """Load TARDIS spectrum from file"""
    data = np.loadtxt(filepath, comments='#')
    wl = data[:, 0]  # Wavelength in Angstrom
    flux = data[:, 1]  # L_nu or L_lambda
    # Sort by wavelength
    idx = np.argsort(wl)
    return wl[idx], flux[idx]

def load_lumina_rotation_spectrum(filepath):
    """Load LUMINA rotation (standard) spectrum"""
    data = []
    with open(filepath, 'r') as f:
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

tardis_path = '/home/kjhan/TARDIS-SN/tardis_spectrum.dat'
lumina_vpacket_path = 'lumina_vpacket_spectrum.dat'
lumina_rotation_path = 'spectrum_sn2011fe.csv'

wl_tardis, flux_tardis = load_tardis_spectrum(tardis_path)
wl_vpacket, flux_vpacket = load_lumina_vpacket_spectrum(lumina_vpacket_path)

# Try to load rotation spectrum for comparison
try:
    wl_rotation, flux_rotation = load_lumina_rotation_spectrum(lumina_rotation_path)
    has_rotation = True
except:
    has_rotation = False

# Filter to common range
wl_min, wl_max = 3000, 10000
mask_t = (wl_tardis >= wl_min) & (wl_tardis <= wl_max)
mask_v = (wl_vpacket >= wl_min) & (wl_vpacket <= wl_max)

wl_tardis = wl_tardis[mask_t]
flux_tardis = flux_tardis[mask_t]
wl_vpacket = wl_vpacket[mask_v]
flux_vpacket = flux_vpacket[mask_v]

if has_rotation:
    mask_r = (wl_rotation >= wl_min) & (wl_rotation <= wl_max)
    wl_rotation = wl_rotation[mask_r]
    flux_rotation = flux_rotation[mask_r]

print(f"TARDIS:        {len(wl_tardis)} points, {wl_tardis.min():.0f}-{wl_tardis.max():.0f} A")
print(f"LUMINA vpacket: {len(wl_vpacket)} points, {wl_vpacket.min():.0f}-{wl_vpacket.max():.0f} A")
if has_rotation:
    print(f"LUMINA rotation: {len(wl_rotation)} points")

# Normalize for comparison
flux_tardis_norm = flux_tardis / np.max(flux_tardis)
flux_vpacket_norm = flux_vpacket / np.max(flux_vpacket)
if has_rotation:
    flux_rotation_norm = flux_rotation / np.max(flux_rotation)

# Create comparison plot
if has_rotation:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
else:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: TARDIS spectrum
ax1 = axes[0]
ax1.plot(wl_tardis, flux_tardis, 'b-', lw=0.8, label='TARDIS')
ax1.set_ylabel(r'L$_\nu$ (erg/s/Hz)', fontsize=11)
ax1.set_yscale('log')
ax1.legend(loc='upper right', fontsize=11)
ax1.set_title('TARDIS vs LUMINA Virtual Packet Spectrum Comparison', fontsize=14)
ax1.axvline(6355, color='green', ls='--', alpha=0.7, lw=1)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(wl_min, wl_max)

# Panel 2: LUMINA virtual packet spectrum
ax2 = axes[1]
ax2.plot(wl_vpacket, flux_vpacket, 'r-', lw=0.8, label='LUMINA Virtual Packet')
ax2.set_ylabel(r'L$_\nu$ (erg/s/Hz)', fontsize=11)
ax2.set_yscale('log')
ax2.legend(loc='upper right', fontsize=11)
ax2.axvline(6355, color='green', ls='--', alpha=0.7, lw=1)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(wl_min, wl_max)

# Panel 3: Normalized comparison overlay
ax3 = axes[2]
ax3.plot(wl_tardis, flux_tardis_norm, 'b-', lw=1, alpha=0.8, label='TARDIS')
ax3.plot(wl_vpacket, flux_vpacket_norm, 'r-', lw=1, alpha=0.8, label='LUMINA vpacket')
if has_rotation:
    ax3.plot(wl_rotation, flux_rotation_norm, 'g--', lw=0.8, alpha=0.6, label='LUMINA rotation')
ax3.set_xlabel('Wavelength (A)', fontsize=12)
ax3.set_ylabel('Normalized Flux', fontsize=11)
ax3.legend(loc='upper right', fontsize=11)
ax3.axvline(6355, color='green', ls='--', alpha=0.7, lw=1)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(wl_min, wl_max)
ax3.set_ylim(0, 1.1)

# Panel 4 (if rotation): All three methods
if has_rotation:
    ax4 = axes[3]
    ax4.plot(wl_tardis, flux_tardis_norm, 'b-', lw=1.2, alpha=0.9, label='TARDIS')
    ax4.plot(wl_vpacket, flux_vpacket_norm, 'r-', lw=1, alpha=0.8, label='LUMINA vpacket')
    ax4.plot(wl_rotation, flux_rotation_norm, 'g--', lw=0.8, alpha=0.6, label='LUMINA rotation')
    ax4.set_xlabel('Wavelength (A)', fontsize=12)
    ax4.set_ylabel('Normalized Flux', fontsize=11)
    ax4.legend(loc='upper right', fontsize=11)
    ax4.set_title('Comparison: TARDIS vs LUMINA vpacket vs rotation', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(wl_min, wl_max)
    ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('vpacket_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: vpacket_comparison.png")

# Print statistics
print("\n" + "="*60)
print("VIRTUAL PACKET SPECTRUM STATISTICS")
print("="*60)

print(f"\nTARDIS:")
print(f"  Peak: {flux_tardis.max():.3e} at {wl_tardis[np.argmax(flux_tardis)]:.0f} A")
print(f"  Mean: {flux_tardis.mean():.3e}")

print(f"\nLUMINA Virtual Packet:")
print(f"  Peak: {flux_vpacket.max():.3e} at {wl_vpacket[np.argmax(flux_vpacket)]:.0f} A")
print(f"  Mean: {flux_vpacket.mean():.3e}")

# Calculate chi-square in overlapping region
from scipy.interpolate import interp1d

# Interpolate LUMINA to TARDIS wavelength grid
if len(wl_vpacket) > 10 and len(wl_tardis) > 10:
    f_vpacket = interp1d(wl_vpacket, flux_vpacket_norm, bounds_error=False, fill_value=0)
    vpacket_interp = f_vpacket(wl_tardis)

    # Mask valid points
    valid = (vpacket_interp > 0) & (flux_tardis_norm > 0)

    if np.sum(valid) > 10:
        residual = flux_tardis_norm[valid] - vpacket_interp[valid]
        chi_sq = np.sum(residual**2) / np.sum(valid)
        rms = np.sqrt(chi_sq)

        print(f"\nComparison with TARDIS:")
        print(f"  RMS difference: {rms:.4f}")
        print(f"  Chi-squared:    {chi_sq:.4f}")
        print(f"  N valid points: {np.sum(valid)}")

print("="*60)
