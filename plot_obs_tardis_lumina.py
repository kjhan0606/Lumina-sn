#!/usr/bin/env python3
"""
Plot comparison of Observation, TARDIS, and LUMINA spectra.

Using TARDIS-matching parameters for LUMINA:
- MACRO_GAUNT_SCALE=1.0
- MACRO_COLLISIONAL_BOOST=1.0
- MACRO_EPSILON=0.0
- MACRO_IR_THERM=0.0

Author: Claude Code Analysis
Date: 2026-02-01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# File paths
OBS_FILE = "../Lumina/data/sn2011fe/spectra/sn2011fe_p1d2d_SN2011fe-20110911-a00p7.dat"
TARDIS_FILE = "../tardis-sn/sn2011fe_stratified_spectrum.dat"
LUMINA_FILE = "spectrum_tardis_match.csv"

def load_observation(filename):
    """Load observation spectrum."""
    wavelength = []
    flux = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    w = float(parts[0])
                    f_val = float(parts[1])
                    wavelength.append(w)
                    flux.append(f_val)
                except ValueError:
                    continue
    return np.array(wavelength), np.array(flux)

def load_tardis(filename):
    """Load TARDIS spectrum."""
    wavelength = []
    flux = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    w = float(parts[0])
                    f_val = float(parts[1])
                    wavelength.append(w)
                    flux.append(f_val)
                except ValueError:
                    continue
    return np.array(wavelength), np.array(flux)

def load_lumina(filename):
    """Load LUMINA spectrum."""
    wavelength = []
    flux = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('wavelength'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    w = float(parts[0])
                    f_val = float(parts[1])
                    wavelength.append(w)
                    flux.append(f_val)
                except ValueError:
                    continue
    return np.array(wavelength), np.array(flux)

def normalize_spectrum(wavelength, flux, ref_range=(5500, 5700)):
    """Normalize spectrum to reference wavelength range."""
    mask = (wavelength >= ref_range[0]) & (wavelength <= ref_range[1])
    if np.sum(mask) > 0:
        norm = np.median(flux[mask])
        if norm > 0:
            return flux / norm
    return flux / np.max(flux)

def calculate_chi_square(obs_w, obs_f, model_w, model_f, wl_range):
    """Calculate chi-square in wavelength range."""
    mask = (obs_w >= wl_range[0]) & (obs_w <= wl_range[1])
    obs_subset = obs_f[mask]
    obs_w_subset = obs_w[mask]

    # Interpolate model to observation wavelengths
    model_interp = np.interp(obs_w_subset, model_w, model_f)

    # Chi-square (assuming unit weights)
    chi2 = np.sum((obs_subset - model_interp)**2) / len(obs_subset)
    return chi2

def main():
    print("Loading spectra...")

    # Load spectra
    obs_w, obs_f = load_observation(OBS_FILE)
    tardis_w, tardis_f = load_tardis(TARDIS_FILE)
    lumina_w, lumina_f = load_lumina(LUMINA_FILE)

    print(f"  Observation: {len(obs_w)} points, {obs_w.min():.0f}-{obs_w.max():.0f} Å")
    print(f"  TARDIS: {len(tardis_w)} points, {tardis_w.min():.0f}-{tardis_w.max():.0f} Å")
    print(f"  LUMINA: {len(lumina_w)} points, {lumina_w.min():.0f}-{lumina_w.max():.0f} Å")

    # Normalize spectra
    obs_f_norm = normalize_spectrum(obs_w, obs_f)
    tardis_f_norm = normalize_spectrum(tardis_w, tardis_f)
    lumina_f_norm = normalize_spectrum(lumina_w, lumina_f)

    # Smooth LUMINA spectrum
    lumina_f_smooth = gaussian_filter1d(lumina_f_norm, sigma=3)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Common wavelength range
    wl_min, wl_max = 3000, 9000

    # Panel 1: Full comparison
    ax1 = axes[0]
    ax1.plot(obs_w, obs_f_norm, 'k-', lw=1, alpha=0.7, label='SN 2011fe (+1.2d)')
    ax1.plot(tardis_w, tardis_f_norm, 'b-', lw=1.5, alpha=0.8, label='TARDIS')
    ax1.plot(lumina_w, lumina_f_smooth, 'r-', lw=1.5, alpha=0.8, label='LUMINA (TARDIS params)')
    ax1.set_xlim(wl_min, wl_max)
    ax1.set_ylim(0, 2.5)
    ax1.set_ylabel('Normalized Flux', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('SN 2011fe: Observation vs TARDIS vs LUMINA\n(LUMINA with TARDIS-matching parameters)', fontsize=14)

    # Mark key features
    features = [
        (3934, 'Ca II H&K'),
        (4130, 'Si II 4130'),
        (5051, 'S II W'),
        (5454, 'S II 5454'),
        (5640, 'S II 5640'),
        (6150, 'Si II 6150'),
        (6355, 'Si II 6355'),
        (8579, 'Ca II IR'),
    ]
    for wl, name in features:
        ax1.axvline(wl, color='gray', ls='--', alpha=0.3, lw=0.5)

    # Panel 2: Blue region zoom
    ax2 = axes[1]
    ax2.plot(obs_w, obs_f_norm, 'k-', lw=1, alpha=0.7, label='Observation')
    ax2.plot(tardis_w, tardis_f_norm, 'b-', lw=1.5, alpha=0.8, label='TARDIS')
    ax2.plot(lumina_w, lumina_f_smooth, 'r-', lw=1.5, alpha=0.8, label='LUMINA')
    ax2.set_xlim(3500, 5500)
    ax2.set_ylim(0, 2.0)
    ax2.set_ylabel('Normalized Flux', fontsize=12)
    ax2.set_title('Blue Region (Ca II H&K, Si II, S II)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)

    # Panel 3: Red region zoom (Si II 6355)
    ax3 = axes[2]
    ax3.plot(obs_w, obs_f_norm, 'k-', lw=1, alpha=0.7, label='Observation')
    ax3.plot(tardis_w, tardis_f_norm, 'b-', lw=1.5, alpha=0.8, label='TARDIS')
    ax3.plot(lumina_w, lumina_f_smooth, 'r-', lw=1.5, alpha=0.8, label='LUMINA')
    ax3.set_xlim(5500, 7500)
    ax3.set_ylim(0, 2.0)
    ax3.set_xlabel('Wavelength (Å)', fontsize=12)
    ax3.set_ylabel('Normalized Flux', fontsize=12)
    ax3.set_title('Red Region (Si II 6355 - key diagnostic)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)

    # Mark Si II 6355
    ax3.axvline(6355, color='green', ls='--', alpha=0.5, lw=1.5, label='Si II 6355 rest')

    plt.tight_layout()
    plt.savefig('obs_tardis_lumina_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('obs_tardis_lumina_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: obs_tardis_lumina_comparison.pdf/.png")

    # Calculate chi-square
    print("\n" + "="*60)
    print("Chi-Square Comparison (TARDIS params)")
    print("="*60)

    regions = [
        ('Blue (3500-4500 Å)', (3500, 4500)),
        ('Green (4500-5500 Å)', (4500, 5500)),
        ('Red (5500-6500 Å)', (5500, 6500)),
        ('Si II 6355 (6000-6500 Å)', (6000, 6500)),
        ('Far-red (6500-7500 Å)', (6500, 7500)),
        ('NIR (7500-9000 Å)', (7500, 9000)),
    ]

    print(f"\n{'Region':<25} {'TARDIS χ²':>12} {'LUMINA χ²':>12} {'Better':>10}")
    print("-"*60)

    for name, wl_range in regions:
        chi2_tardis = calculate_chi_square(obs_w, obs_f_norm, tardis_w, tardis_f_norm, wl_range)
        chi2_lumina = calculate_chi_square(obs_w, obs_f_norm, lumina_w, lumina_f_smooth, wl_range)
        better = "TARDIS" if chi2_tardis < chi2_lumina else "LUMINA"
        print(f"{name:<25} {chi2_tardis:>12.2f} {chi2_lumina:>12.2f} {better:>10}")

    print("\n" + "="*60)
    print("LUMINA Parameters Used:")
    print("  MACRO_GAUNT_SCALE=1.0")
    print("  MACRO_COLLISIONAL_BOOST=1.0")
    print("  MACRO_EPSILON=0.0")
    print("  MACRO_IR_THERM=0.0")
    print("="*60)

if __name__ == '__main__':
    main()
