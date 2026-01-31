#!/usr/bin/env python3
"""
LUMINA-SN Chi-Square Calculation and Spectrum Comparison
Compare LUMINA output to TARDIS reference spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

def load_lumina_spectrum(filename):
    """Load LUMINA spectrum CSV file"""
    wavelength = []
    L_lumina = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('wavelength'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                wavelength.append(float(parts[0]))
                L_lumina.append(float(parts[3]))  # LUMINA column

    return np.array(wavelength), np.array(L_lumina)

def load_tardis_spectrum(filename):
    """Load TARDIS reference spectrum"""
    wavelength = []
    luminosity = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                wavelength.append(float(parts[0]))
                luminosity.append(float(parts[1]))

    return np.array(wavelength), np.array(luminosity)

def calculate_chi_square(wl_data, L_data, wl_ref, L_ref, wl_min=3500, wl_max=9000):
    """
    Calculate chi-square between data and reference spectra.

    Uses normalized spectra and RMS-based chi-square for stability.
    """
    # Interpolate reference to data wavelength grid
    mask = (wl_data >= wl_min) & (wl_data <= wl_max)
    wl_masked = wl_data[mask]
    L_masked = L_data[mask]

    # Create interpolation function for reference
    ref_mask = (wl_ref >= wl_min) & (wl_ref <= wl_max)
    wl_ref_masked = wl_ref[ref_mask]
    L_ref_masked = L_ref[ref_mask]

    if len(wl_ref_masked) < 10:
        print("Warning: Not enough reference points in wavelength range")
        return np.inf, 0, 0

    interp_ref = interp1d(wl_ref_masked, L_ref_masked, kind='linear',
                          bounds_error=False, fill_value=0)

    L_ref_interp = interp_ref(wl_masked)

    # Normalize both spectra to peak = 1
    norm_data = np.max(L_masked) if np.max(L_masked) > 0 else 1
    norm_ref = np.max(L_ref_interp) if np.max(L_ref_interp) > 0 else 1

    L_data_norm = L_masked / norm_data
    L_ref_norm = L_ref_interp / norm_ref

    # Calculate chi-square using relative difference
    # Only consider points where both spectra have significant flux
    threshold = 0.05  # 5% of peak
    valid = (L_data_norm > threshold) | (L_ref_norm > threshold)

    if np.sum(valid) < 10:
        print(f"Warning: Only {np.sum(valid)} valid points")
        return np.inf, 0, 0

    # Use weighted RMS difference
    diff = L_data_norm[valid] - L_ref_norm[valid]
    weights = (L_data_norm[valid] + L_ref_norm[valid]) / 2  # Average flux as weight

    chi2 = np.sum((diff**2) * weights) / np.sum(weights)
    chi2_scaled = chi2 * 1000  # Scale for readability

    # Also compute simple RMS for reference
    rms = np.sqrt(np.mean(diff**2)) * 100  # As percentage

    return chi2_scaled, np.sum(valid), rms

def plot_comparison(wl_data, L_data, wl_ref, L_ref, chi2, output_file='spectrum_comparison_chisq.pdf'):
    """Create comparison plot with chi-square annotation"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Normalize for plotting
    wl_range = (wl_data >= 3500) & (wl_data <= 9000)
    wl_range_ref = (wl_ref >= 3500) & (wl_ref <= 9000)

    norm_data = np.max(L_data[wl_range]) if np.any(wl_range) else 1
    norm_ref = np.max(L_ref[wl_range_ref]) if np.any(wl_range_ref) else 1

    # Top panel: Spectrum comparison
    ax1 = axes[0]
    ax1.plot(wl_ref, L_ref / norm_ref, 'b-', linewidth=1.5, label='TARDIS Reference', alpha=0.8)
    ax1.plot(wl_data, L_data / norm_data, 'r-', linewidth=1.0, label='LUMINA (Macro-Atom)', alpha=0.8)

    ax1.set_xlabel('Wavelength [Å]', fontsize=12)
    ax1.set_ylabel('Normalized Flux', fontsize=12)
    ax1.set_title(f'LUMINA vs TARDIS Spectrum Comparison\n$\\chi^2$ = {chi2:.2f}', fontsize=14)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(3000, 10000)
    ax1.set_ylim(0, 1.3)
    ax1.grid(True, alpha=0.3)

    # Mark key spectral features
    features = {
        'Ca II H&K': 3945,
        'Si II 4130': 4130,
        'Mg II': 4481,
        'Fe II': 4924,
        'S II W': 5454,
        'Si II 5972': 5972,
        'Si II 6355': 6355,
        'Ca II IR': 8542,
    }

    for name, wl in features.items():
        ax1.axvline(wl, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax1.text(wl, 1.25, name, fontsize=8, ha='center', rotation=45)

    # Bottom panel: Residuals
    ax2 = axes[1]

    # Interpolate reference to data grid for residuals
    interp_ref = interp1d(wl_ref, L_ref / norm_ref, kind='linear',
                          bounds_error=False, fill_value=0)
    L_ref_interp = interp_ref(wl_data)

    residual = (L_data / norm_data) - L_ref_interp

    ax2.fill_between(wl_data, residual, 0, alpha=0.5, color='purple')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Wavelength [Å]', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_xlim(3000, 10000)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Also save PNG version
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_file}")

    plt.close()

def main():
    # File paths
    lumina_file = 'spectrum_sn2011fe.csv'
    tardis_file = 'tardis_comparison_spectrum.dat'

    if len(sys.argv) > 1:
        lumina_file = sys.argv[1]
    if len(sys.argv) > 2:
        tardis_file = sys.argv[2]

    print("=" * 60)
    print("LUMINA-SN Chi-Square Analysis")
    print("=" * 60)

    # Load spectra
    print(f"\nLoading LUMINA spectrum: {lumina_file}")
    try:
        wl_lumina, L_lumina = load_lumina_spectrum(lumina_file)
        print(f"  Loaded {len(wl_lumina)} wavelength bins")
        print(f"  Wavelength range: {wl_lumina.min():.1f} - {wl_lumina.max():.1f} Å")
    except Exception as e:
        print(f"Error loading LUMINA spectrum: {e}")
        return 1

    print(f"\nLoading TARDIS reference: {tardis_file}")
    try:
        wl_tardis, L_tardis = load_tardis_spectrum(tardis_file)
        print(f"  Loaded {len(wl_tardis)} wavelength bins")
        print(f"  Wavelength range: {wl_tardis.min():.1f} - {wl_tardis.max():.1f} Å")
    except Exception as e:
        print(f"Error loading TARDIS spectrum: {e}")
        return 1

    # Calculate chi-square
    print("\n" + "-" * 60)
    print("Chi-Square Calculation (3500-9000 Å)")
    print("-" * 60)

    chi2, n_valid, rms = calculate_chi_square(wl_lumina, L_lumina, wl_tardis, L_tardis)
    print(f"\n  Chi-square (weighted): {chi2:.2f}")
    print(f"  RMS difference: {rms:.1f}%")
    print(f"  Valid points: {n_valid}")

    if chi2 < 50:
        quality = "EXCELLENT"
    elif chi2 < 100:
        quality = "GOOD"
    elif chi2 < 200:
        quality = "FAIR"
    else:
        quality = "POOR"

    print(f"  Fit quality: {quality}")

    # Generate comparison plot
    print("\n" + "-" * 60)
    print("Generating comparison plot...")
    print("-" * 60)

    plot_comparison(wl_lumina, L_lumina, wl_tardis, L_tardis, chi2)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    main()
