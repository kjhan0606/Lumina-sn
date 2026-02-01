#!/usr/bin/env python3
"""
Analyze LUMINA spectrum after tau_sobolev fix and compare with TARDIS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Physical constants
C_LIGHT = 2.99792458e10  # cm/s

def load_spectrum(filename, format='lumina'):
    """Load spectrum from CSV file."""
    # Count header lines (starting with # or non-numeric)
    with open(filename, 'r') as f:
        skip = 0
        for line in f:
            if line.startswith('#') or not line[0].isdigit():
                skip += 1
            else:
                break

    data = np.loadtxt(filename, delimiter=',', skiprows=skip)
    if format == 'lumina':
        # LUMINA format: wavelength[A], frequency, L_nu_standard, L_nu_lumina, counts...
        wl = data[:, 0]
        flux = data[:, 3]  # LUMINA flux (L_nu_lumina column)
    else:
        wl = data[:, 0]
        flux = data[:, 1]
    return wl, flux

def load_tardis_spectrum(filename):
    """Load TARDIS spectrum."""
    data = np.loadtxt(filename)
    wl = data[:, 0]  # Wavelength in Angstrom
    flux = data[:, 1]  # Flux
    return wl, flux

def calculate_chi_square(wl1, flux1, wl2, flux2, wl_min=3500, wl_max=7500):
    """Calculate chi-square between two spectra."""
    # Interpolate to common grid
    mask1 = (wl1 >= wl_min) & (wl1 <= wl_max)
    mask2 = (wl2 >= wl_min) & (wl2 <= wl_max)

    if not np.any(mask1) or not np.any(mask2):
        return np.inf

    wl_common = np.linspace(wl_min, wl_max, 500)

    f1_interp = interp1d(wl1[mask1], flux1[mask1], bounds_error=False, fill_value=0)
    f2_interp = interp1d(wl2[mask2], flux2[mask2], bounds_error=False, fill_value=0)

    flux1_common = f1_interp(wl_common)
    flux2_common = f2_interp(wl_common)

    # Normalize
    norm1 = np.mean(flux1_common[flux1_common > 0]) if np.any(flux1_common > 0) else 1
    norm2 = np.mean(flux2_common[flux2_common > 0]) if np.any(flux2_common > 0) else 1

    flux1_norm = flux1_common / norm1
    flux2_norm = flux2_common / norm2

    # Chi-square
    valid = (flux1_norm > 0) & (flux2_norm > 0)
    if not np.any(valid):
        return np.inf

    chi2 = np.sum((flux1_norm[valid] - flux2_norm[valid])**2) / np.sum(valid)
    return chi2 * 1000  # Scale for readability

def find_si_ii_velocity(wl, flux, rest_wavelength=6355.0):
    """Find Si II 6355 absorption velocity from spectrum."""
    # Look for minimum in the 5800-6300 A range (blueshifted Si II)
    mask = (wl >= 5800) & (wl <= 6300)
    if not np.any(mask):
        return None, None

    wl_region = wl[mask]
    flux_region = flux[mask]

    # Smooth to find minimum
    flux_smooth = gaussian_filter1d(flux_region, sigma=3)

    # Find minimum
    min_idx = np.argmin(flux_smooth)
    wl_min = wl_region[min_idx]

    # Calculate velocity
    v_kms = C_LIGHT * (rest_wavelength - wl_min) / rest_wavelength / 1e5

    return wl_min, v_kms

def main():
    print("=" * 70)
    print("LUMINA tau_sobolev Fix Analysis")
    print("=" * 70)

    # Load LUMINA spectrum
    try:
        wl_lumina, flux_lumina = load_spectrum('/tmp/lumina_tardis_match.csv')
        print(f"\nLUMINA spectrum loaded: {len(wl_lumina)} points")
        print(f"  Wavelength range: {wl_lumina.min():.1f} - {wl_lumina.max():.1f} A")
    except Exception as e:
        print(f"Error loading LUMINA spectrum: {e}")
        return

    # Load TARDIS comparison spectrum
    try:
        wl_tardis, flux_tardis = load_tardis_spectrum('tardis_comparison_spectrum.dat')
        print(f"\nTARDIS spectrum loaded: {len(wl_tardis)} points")
        print(f"  Wavelength range: {wl_tardis.min():.1f} - {wl_tardis.max():.1f} A")
        has_tardis = True
    except Exception as e:
        print(f"Warning: Could not load TARDIS spectrum: {e}")
        has_tardis = False

    # Si II 6355 velocity measurement
    print("\n" + "-" * 70)
    print("Si II 6355 Velocity Analysis")
    print("-" * 70)

    wl_min_lumina, v_lumina = find_si_ii_velocity(wl_lumina, flux_lumina)
    if v_lumina is not None:
        print(f"\nLUMINA Si II 6355:")
        print(f"  Absorption minimum: {wl_min_lumina:.1f} A")
        print(f"  Velocity: {v_lumina:.0f} km/s")

    if has_tardis:
        wl_min_tardis, v_tardis = find_si_ii_velocity(wl_tardis, flux_tardis)
        if v_tardis is not None:
            print(f"\nTARDIS Si II 6355:")
            print(f"  Absorption minimum: {wl_min_tardis:.1f} A")
            print(f"  Velocity: {v_tardis:.0f} km/s")

        if v_lumina is not None and v_tardis is not None:
            print(f"\nVelocity difference: {v_lumina - v_tardis:.0f} km/s")

    # Chi-square calculation
    if has_tardis:
        print("\n" + "-" * 70)
        print("Chi-Square Analysis")
        print("-" * 70)

        chi2_full = calculate_chi_square(wl_lumina, flux_lumina, wl_tardis, flux_tardis, 3500, 7500)
        chi2_blue = calculate_chi_square(wl_lumina, flux_lumina, wl_tardis, flux_tardis, 3500, 5000)
        chi2_red = calculate_chi_square(wl_lumina, flux_lumina, wl_tardis, flux_tardis, 5000, 7000)
        chi2_si = calculate_chi_square(wl_lumina, flux_lumina, wl_tardis, flux_tardis, 5800, 6500)

        print(f"\nChi-square vs TARDIS:")
        print(f"  Full (3500-7500 A): {chi2_full:.2f}")
        print(f"  Blue (3500-5000 A): {chi2_blue:.2f}")
        print(f"  Red  (5000-7000 A): {chi2_red:.2f}")
        print(f"  Si II region (5800-6500 A): {chi2_si:.2f}")

    # Create comparison plot
    print("\n" + "-" * 70)
    print("Generating comparison plot...")
    print("-" * 70)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Full spectrum comparison
    ax1 = axes[0]
    ax1.plot(wl_lumina, flux_lumina / np.max(flux_lumina[(wl_lumina > 4000) & (wl_lumina < 7000)]),
             'b-', label='LUMINA (tau fix)', alpha=0.8, linewidth=1.5)
    if has_tardis:
        ax1.plot(wl_tardis, flux_tardis / np.max(flux_tardis[(wl_tardis > 4000) & (wl_tardis < 7000)]),
                 'r--', label='TARDIS', alpha=0.8, linewidth=1.5)

    ax1.set_xlim(3000, 8000)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel('Wavelength [A]')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Full Spectrum Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark Si II 6355
    ax1.axvline(6355, color='green', linestyle=':', alpha=0.5, label='Si II rest')
    if wl_min_lumina:
        ax1.axvline(wl_min_lumina, color='blue', linestyle='--', alpha=0.5)

    # Si II region zoom
    ax2 = axes[1]
    mask_lumina = (wl_lumina >= 5500) & (wl_lumina <= 6800)
    ax2.plot(wl_lumina[mask_lumina],
             flux_lumina[mask_lumina] / np.max(flux_lumina[mask_lumina]),
             'b-', label='LUMINA (tau fix)', alpha=0.8, linewidth=2)

    if has_tardis:
        mask_tardis = (wl_tardis >= 5500) & (wl_tardis <= 6800)
        ax2.plot(wl_tardis[mask_tardis],
                 flux_tardis[mask_tardis] / np.max(flux_tardis[mask_tardis]),
                 'r--', label='TARDIS', alpha=0.8, linewidth=2)

    ax2.axvline(6355, color='green', linestyle=':', alpha=0.7, label='Si II 6355 rest')
    if wl_min_lumina:
        ax2.axvline(wl_min_lumina, color='blue', linestyle='--', alpha=0.7,
                   label=f'LUMINA min ({v_lumina:.0f} km/s)')

    ax2.set_xlim(5500, 6800)
    ax2.set_xlabel('Wavelength [A]')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title('Si II 6355 Region (Key Diagnostic)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tau_fix_comparison.pdf', dpi=150)
    print("Saved: tau_fix_comparison.pdf")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nSi II 6355 velocity: {v_lumina:.0f} km/s" if v_lumina else "Si II not detected")
    if has_tardis:
        print(f"Chi-square vs TARDIS: {chi2_full:.2f}")
        print(f"TARDIS Si II velocity: {v_tardis:.0f} km/s" if v_tardis else "")

    # Check if results improved
    print("\n" + "-" * 70)
    print("Assessment:")
    print("-" * 70)
    if v_lumina and v_lumina > 8000 and v_lumina < 15000:
        print("  Si II velocity: REASONABLE (8000-15000 km/s expected)")
    elif v_lumina:
        print(f"  Si II velocity: OUT OF RANGE (got {v_lumina:.0f} km/s)")

    if has_tardis and chi2_full < 50:
        print("  Chi-square: GOOD (< 50)")
    elif has_tardis and chi2_full < 100:
        print("  Chi-square: MODERATE (50-100)")
    elif has_tardis:
        print(f"  Chi-square: NEEDS IMPROVEMENT (> 100)")

if __name__ == '__main__':
    main()
