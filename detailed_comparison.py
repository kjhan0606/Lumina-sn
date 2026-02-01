#!/usr/bin/env python3
"""
Detailed LUMINA vs TARDIS Spectrum Comparison

Compares specific spectral features to identify remaining differences.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

C_LIGHT = 2.99792458e10  # cm/s

def load_spectrum(filename):
    """Load spectrum, handling different formats."""
    with open(filename, 'r') as f:
        skip = 0
        for line in f:
            if line.startswith('#') or not line[0].isdigit():
                skip += 1
            else:
                break

    data = np.loadtxt(filename, delimiter=',', skiprows=skip)
    if data.shape[1] >= 4:
        # LUMINA format
        wl = data[:, 0]
        flux = data[:, 3]  # L_nu_lumina
    else:
        wl = data[:, 0]
        flux = data[:, 1]
    return wl, flux

def load_tardis(filename):
    """Load TARDIS spectrum."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]

def find_feature_velocity(wl, flux, rest_wl, search_range=500):
    """Find absorption velocity for a feature."""
    mask = (wl >= rest_wl - search_range) & (wl <= rest_wl)
    if not np.any(mask):
        return None, None

    wl_region = wl[mask]
    flux_region = flux[mask]
    flux_smooth = gaussian_filter1d(flux_region, sigma=3)

    min_idx = np.argmin(flux_smooth)
    wl_min = wl_region[min_idx]
    v_kms = C_LIGHT * (rest_wl - wl_min) / rest_wl / 1e5

    return wl_min, v_kms

def main():
    print("=" * 70)
    print("Detailed LUMINA vs TARDIS Spectrum Comparison")
    print("=" * 70)

    # Load spectra
    wl_l, flux_l = load_spectrum('/tmp/lumina_tardis_match.csv')
    wl_t, flux_t = load_tardis('tardis_comparison_spectrum.dat')

    # Normalize in optical region
    mask_l = (wl_l > 4500) & (wl_l < 6500)
    mask_t = (wl_t > 4500) & (wl_t < 6500)
    norm_l = np.max(flux_l[mask_l])
    norm_t = np.max(flux_t[mask_t])
    flux_l_norm = flux_l / norm_l
    flux_t_norm = flux_t / norm_t

    # Key features to check
    features = {
        'Si II 6355': 6355.0,
        'Si II 5972': 5972.0,
        'S II 5640': 5640.0,
        'Ca II H&K': 3945.0,
        'Ca II IR': 8542.0,
    }

    print("\n" + "-" * 70)
    print("Feature Velocities Comparison")
    print("-" * 70)
    print(f"{'Feature':<15} {'Rest [A]':>10} {'LUMINA v':>12} {'TARDIS v':>12} {'Diff':>10}")
    print("-" * 70)

    for name, rest_wl in features.items():
        wl_l_min, v_l = find_feature_velocity(wl_l, flux_l_norm, rest_wl)
        wl_t_min, v_t = find_feature_velocity(wl_t, flux_t_norm, rest_wl)

        if v_l is not None and v_t is not None:
            diff = v_l - v_t
            print(f"{name:<15} {rest_wl:>10.1f} {v_l:>10.0f} km/s {v_t:>10.0f} km/s {diff:>+8.0f}")
        elif v_l is not None:
            print(f"{name:<15} {rest_wl:>10.1f} {v_l:>10.0f} km/s {'N/A':>12} {'N/A':>10}")
        else:
            print(f"{name:<15} {rest_wl:>10.1f} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    # Create detailed comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Full spectrum
    ax = axes[0, 0]
    ax.plot(wl_l, flux_l_norm, 'b-', label='LUMINA', alpha=0.8, lw=1.5)
    ax.plot(wl_t, flux_t_norm, 'r--', label='TARDIS', alpha=0.8, lw=1.5)
    ax.set_xlim(3000, 9000)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Full Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Blue region (Ca H&K)
    ax = axes[0, 1]
    ax.plot(wl_l, flux_l_norm, 'b-', label='LUMINA', alpha=0.8, lw=2)
    ax.plot(wl_t, flux_t_norm, 'r--', label='TARDIS', alpha=0.8, lw=2)
    ax.axvline(3934, color='g', ls=':', alpha=0.5, label='Ca II K rest')
    ax.axvline(3969, color='g', ls=':', alpha=0.5, label='Ca II H rest')
    ax.set_xlim(3500, 4500)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Ca II H&K Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Si II 5972 region
    ax = axes[1, 0]
    ax.plot(wl_l, flux_l_norm, 'b-', label='LUMINA', alpha=0.8, lw=2)
    ax.plot(wl_t, flux_t_norm, 'r--', label='TARDIS', alpha=0.8, lw=2)
    ax.axvline(5972, color='g', ls=':', alpha=0.5, label='Si II 5972 rest')
    ax.set_xlim(5400, 6100)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Si II 5972 Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Si II 6355 region (key diagnostic)
    ax = axes[1, 1]
    ax.plot(wl_l, flux_l_norm, 'b-', label='LUMINA', alpha=0.8, lw=2)
    ax.plot(wl_t, flux_t_norm, 'r--', label='TARDIS', alpha=0.8, lw=2)
    ax.axvline(6355, color='g', ls=':', alpha=0.5, label='Si II 6355 rest')
    wl_l_min, v_l = find_feature_velocity(wl_l, flux_l_norm, 6355)
    wl_t_min, v_t = find_feature_velocity(wl_t, flux_t_norm, 6355)
    if wl_l_min:
        ax.axvline(wl_l_min, color='b', ls='--', alpha=0.5, label=f'LUMINA min ({v_l:.0f} km/s)')
    if wl_t_min:
        ax.axvline(wl_t_min, color='r', ls='--', alpha=0.5, label=f'TARDIS min ({v_t:.0f} km/s)')
    ax.set_xlim(5800, 6800)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Si II 6355 Region (Key Diagnostic)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S II region
    ax = axes[2, 0]
    ax.plot(wl_l, flux_l_norm, 'b-', label='LUMINA', alpha=0.8, lw=2)
    ax.plot(wl_t, flux_t_norm, 'r--', label='TARDIS', alpha=0.8, lw=2)
    ax.axvline(5640, color='g', ls=':', alpha=0.5, label='S II 5640 rest')
    ax.set_xlim(5200, 5800)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('S II Region')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ca II IR triplet
    ax = axes[2, 1]
    ax.plot(wl_l, flux_l_norm, 'b-', label='LUMINA', alpha=0.8, lw=2)
    ax.plot(wl_t, flux_t_norm, 'r--', label='TARDIS', alpha=0.8, lw=2)
    ax.axvline(8498, color='g', ls=':', alpha=0.3)
    ax.axvline(8542, color='g', ls=':', alpha=0.3, label='Ca II IR triplet')
    ax.axvline(8662, color='g', ls=':', alpha=0.3)
    ax.set_xlim(7800, 9000)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Ca II IR Triplet')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('detailed_spectrum_comparison.pdf', dpi=150)
    print("\nSaved: detailed_spectrum_comparison.pdf")

    # Calculate chi-square in different regions
    print("\n" + "-" * 70)
    print("Chi-Square by Spectral Region")
    print("-" * 70)

    regions = [
        ('UV (3000-3500)', 3000, 3500),
        ('Blue (3500-4500)', 3500, 4500),
        ('Green (4500-5500)', 4500, 5500),
        ('Red (5500-6500)', 5500, 6500),
        ('Far-red (6500-7500)', 6500, 7500),
        ('NIR (7500-9000)', 7500, 9000),
    ]

    for name, wl_min, wl_max in regions:
        mask_l = (wl_l >= wl_min) & (wl_l <= wl_max)
        mask_t = (wl_t >= wl_min) & (wl_t <= wl_max)

        if np.sum(mask_l) < 10 or np.sum(mask_t) < 10:
            print(f"{name:<25} N/A")
            continue

        wl_common = np.linspace(wl_min, wl_max, 100)
        f_l = interp1d(wl_l[mask_l], flux_l_norm[mask_l], bounds_error=False, fill_value=0)
        f_t = interp1d(wl_t[mask_t], flux_t_norm[mask_t], bounds_error=False, fill_value=0)

        fl = f_l(wl_common)
        ft = f_t(wl_common)

        valid = (fl > 0.01) & (ft > 0.01)
        if np.sum(valid) < 10:
            print(f"{name:<25} N/A (too few valid points)")
            continue

        chi2 = np.sum((fl[valid] - ft[valid])**2) / np.sum(valid) * 1000
        print(f"{name:<25} {chi2:>8.1f}")

    print("\n" + "=" * 70)
    print("Analysis complete")
    print("=" * 70)

if __name__ == '__main__':
    main()
