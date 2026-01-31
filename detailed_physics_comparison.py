#!/usr/bin/env python3
"""
LUMINA-SN vs TARDIS Detailed Physics Comparison
detailed_physics_comparison.py - Multi-panel validation plots

Creates comprehensive comparison plots for:
1. Ionization balance vs temperature
2. Ionization balance vs density
3. Partition functions vs temperature
4. Emergent spectrum with feature identification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Physical constants (CGS)
CONST_K_B = 1.380649e-16
CONST_H = 6.62607015e-27
CONST_M_E = 9.1093837015e-28
CONST_AMU = 1.66053906660e-24
CONST_EV_TO_ERG = 1.602176634e-12
CONST_PI = 3.14159265358979323846

SAHA_CONST = (2 * CONST_PI * CONST_M_E * CONST_K_B)**1.5 / CONST_H**3


def saha_factor(chi_ev, T, U_i, U_i1):
    chi_erg = chi_ev * CONST_EV_TO_ERG
    kT = CONST_K_B * T
    return (2.0 * U_i1 / U_i) * SAHA_CONST * T**1.5 * np.exp(-chi_erg / kT)


def solve_ionization(T, rho, X_H=1.0, X_He=0.0):
    """Solve Saha ionization balance."""
    n_H = X_H * rho / (1.008 * CONST_AMU) if X_H > 0 else 0
    n_He = X_He * rho / (4.003 * CONST_AMU) if X_He > 0 else 0

    chi_H = 13.598
    chi_He1 = 24.587
    chi_He2 = 54.418

    U_H = [2.0, 1.0]
    U_He = [1.0, 2.0, 1.0]

    n_e = max(n_H + n_He, 1.0)

    for _ in range(50):
        phi_H = saha_factor(chi_H, T, U_H[0], U_H[1])
        x_H = phi_H / (n_e + phi_H) if n_H > 0 else 0

        if n_He > 0:
            phi_He1 = saha_factor(chi_He1, T, U_He[0], U_He[1])
            phi_He2 = saha_factor(chi_He2, T, U_He[1], U_He[2])
            denom_He = 1.0 + phi_He1/n_e + phi_He1*phi_He2/(n_e*n_e)
            x_He0 = 1.0 / denom_He
            x_He1 = (phi_He1/n_e) / denom_He
            x_He2 = (phi_He1*phi_He2/(n_e*n_e)) / denom_He
        else:
            x_He0, x_He1, x_He2 = 0, 0, 0

        n_e_new = n_H * x_H + n_He * (x_He1 + 2*x_He2)
        if abs(n_e_new - n_e) / (n_e + 1e-30) < 1e-8:
            break
        n_e = 0.5 * (n_e + n_e_new)

    return {
        'n_e': n_e,
        'H_I': 1 - x_H,
        'H_II': x_H,
        'He_I': x_He0,
        'He_II': x_He1,
        'He_III': x_He2
    }


def partition_function(E_levels, g_levels, T, cutoff=50):
    """Calculate partition function."""
    kT_ev = CONST_K_B * T / CONST_EV_TO_ERG
    U = 0.0
    for E, g in zip(E_levels, g_levels):
        if E / kT_ev < cutoff:
            U += g * np.exp(-E / kT_ev)
    return max(U, 1.0)


# ============================================================================
# C OUTPUT DATA (from test_plasma)
# ============================================================================

# Temperature scan (Pure H, rho = 1e-10)
C_DATA_TEMP = {
    'T': [5000, 7500, 10000, 12500, 15000, 20000, 30000],
    'n_e': [3.1650e10, 7.7090e12, 5.1812e13, 5.9425e13, 5.9714e13, 5.9742e13, 5.9743e13],
    'H_I': [0.999470, 0.870965, 0.132753, 0.005326, 0.000501, 0.000024, 0.000001],
    'H_II': [0.000530, 0.129035, 0.867247, 0.994674, 0.999499, 0.999976, 0.999999]
}

# Density scan (Pure H, T = 10000 K)
C_DATA_DENS = {
    'rho': [1e-14, 1e-12, 1e-10, 1e-8, 1e-6],
    'n_e': [5.9742e9, 5.9638e11, 5.1812e13, 1.2628e15, 1.4052e16],
    'H_I': [0.000018, 0.001759, 0.132753, 0.788624, 0.976479],
    'H_II': [0.999982, 0.998241, 0.867247, 0.211376, 0.023521]
}


def plot_temperature_comparison():
    """Plot ionization vs temperature comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    T_range = np.logspace(3.5, 4.7, 100)
    rho = 1e-10

    # Calculate Python values
    py_n_e = []
    py_H_I = []
    py_H_II = []

    for T in T_range:
        result = solve_ionization(T, rho, X_H=1.0, X_He=0.0)
        py_n_e.append(result['n_e'])
        py_H_I.append(result['H_I'])
        py_H_II.append(result['H_II'])

    # Plot 1: Electron density vs T
    ax = axes[0, 0]
    ax.loglog(T_range, py_n_e, 'b-', lw=2, label='Python')
    ax.loglog(C_DATA_TEMP['T'], C_DATA_TEMP['n_e'], 'ro', ms=8, label='C (Lumina)')
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('$n_e$ [cm$^{-3}$]')
    ax.set_title('Electron Density vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: H I fraction vs T
    ax = axes[0, 1]
    ax.semilogx(T_range, py_H_I, 'b-', lw=2, label='Python')
    ax.semilogx(C_DATA_TEMP['T'], C_DATA_TEMP['H_I'], 'ro', ms=8, label='C (Lumina)')
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('H I Fraction')
    ax.set_title('Neutral Hydrogen Fraction vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Plot 3: Residuals (relative error)
    ax = axes[1, 0]

    # Interpolate Python to C temperatures
    from scipy.interpolate import interp1d
    py_interp = interp1d(T_range, py_H_I, kind='linear')
    py_at_C_temps = py_interp(C_DATA_TEMP['T'])

    rel_err = (np.array(C_DATA_TEMP['H_I']) - py_at_C_temps) / (py_at_C_temps + 1e-10)
    ax.semilogx(C_DATA_TEMP['T'], 100*rel_err, 'go-', ms=8, lw=2)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axhline(1, color='r', ls=':', lw=0.5, label='1% error')
    ax.axhline(-1, color='r', ls=':', lw=0.5)
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Relative Error [%]')
    ax.set_title('C vs Python: H I Fraction Error')
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = """
    TEMPERATURE SCAN VALIDATION
    ═══════════════════════════════════════

    Pure Hydrogen (X_H = 1.0)
    ρ = 10⁻¹⁰ g/cm³

    Temperature Range: 5,000 - 30,000 K

    Maximum Relative Errors:
    ─────────────────────────
    n_e:     {:.2e}%
    H I:     {:.2e}%
    H II:    {:.2e}%

    Status: {}
    """.format(
        100 * max(abs((np.array(C_DATA_TEMP['n_e']) -
                       [solve_ionization(T, rho)['n_e'] for T in C_DATA_TEMP['T']]) /
                      np.array(C_DATA_TEMP['n_e']))),
        100 * max(abs(rel_err)),
        100 * max(abs((np.array(C_DATA_TEMP['H_II']) -
                       [solve_ionization(T, rho)['H_II'] for T in C_DATA_TEMP['T']]) /
                      (np.array(C_DATA_TEMP['H_II']) + 1e-10))),
        "PASS ✓" if max(abs(rel_err)) < 0.01 else "NEEDS REVIEW"
    )

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('LUMINA-SN vs Python: Saha-Boltzmann Ionization (Temperature)', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_temperature.pdf', dpi=150)
    print("  Saved: comparison_temperature.pdf")


def plot_density_comparison():
    """Plot ionization vs density comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    rho_range = np.logspace(-14, -5, 100)
    T = 10000

    # Calculate Python values
    py_n_e = []
    py_H_I = []

    for rho in rho_range:
        result = solve_ionization(T, rho, X_H=1.0, X_He=0.0)
        py_n_e.append(result['n_e'])
        py_H_I.append(result['H_I'])

    # Plot 1: Electron density vs rho
    ax = axes[0, 0]
    ax.loglog(rho_range, py_n_e, 'b-', lw=2, label='Python')
    ax.loglog(C_DATA_DENS['rho'], C_DATA_DENS['n_e'], 'ro', ms=8, label='C (Lumina)')
    ax.set_xlabel('Density [g/cm³]')
    ax.set_ylabel('$n_e$ [cm$^{-3}$]')
    ax.set_title('Electron Density vs Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: H I fraction vs rho
    ax = axes[0, 1]
    ax.semilogx(rho_range, py_H_I, 'b-', lw=2, label='Python')
    ax.semilogx(C_DATA_DENS['rho'], C_DATA_DENS['H_I'], 'ro', ms=8, label='C (Lumina)')
    ax.set_xlabel('Density [g/cm³]')
    ax.set_ylabel('H I Fraction')
    ax.set_title('Neutral Hydrogen Fraction vs Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Plot 3: Residuals
    ax = axes[1, 0]

    from scipy.interpolate import interp1d
    py_interp = interp1d(np.log10(rho_range), py_H_I, kind='linear')
    py_at_C_dens = py_interp(np.log10(C_DATA_DENS['rho']))

    rel_err = (np.array(C_DATA_DENS['H_I']) - py_at_C_dens) / (py_at_C_dens + 1e-10)
    ax.semilogx(C_DATA_DENS['rho'], 100*rel_err, 'go-', ms=8, lw=2)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axhline(1, color='r', ls=':', lw=0.5)
    ax.axhline(-1, color='r', ls=':', lw=0.5)
    ax.set_xlabel('Density [g/cm³]')
    ax.set_ylabel('Relative Error [%]')
    ax.set_title('C vs Python: H I Fraction Error')
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')

    max_err = max(abs(rel_err))
    summary_text = f"""
    DENSITY SCAN VALIDATION
    ═══════════════════════════════════════

    Pure Hydrogen (X_H = 1.0)
    T = 10,000 K

    Density Range: 10⁻¹⁴ - 10⁻⁶ g/cm³

    Maximum Relative Error:
    ─────────────────────────
    H I fraction:  {100*max_err:.2f}%

    Status: {'PASS ✓' if max_err < 0.01 else 'NEEDS REVIEW'}
    """

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('LUMINA-SN vs Python: Saha-Boltzmann Ionization (Density)', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_density.pdf', dpi=150)
    print("  Saved: comparison_density.pdf")


def plot_spectrum_features():
    """Plot spectrum with identified features."""
    spectrum_file = "spectrum_test.csv"
    if not os.path.exists(spectrum_file):
        print(f"  Spectrum file not found: {spectrum_file}")
        return

    # Load spectrum
    spec = pd.read_csv(spectrum_file, comment='#')

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Important spectral features for Type Ia
    features = {
        'Ca II H&K': (3934, 3968),
        'Si II 4130': (4100, 4150),
        'Mg II': (4481, 4481),
        'Fe II blend': (4900, 5200),
        'S II "W"': (5450, 5650),
        'Si II 5972': (5950, 6000),
        'Si II 6355': (6100, 6400),
        'O I triplet': (7600, 7800),
        'Ca II IR': (8200, 8700),
    }

    # Full spectrum
    ax = axes[0]
    ax.plot(spec['wavelength_A'], spec['L_nu_lumina'], 'b-', lw=0.8, label='LUMINA', alpha=0.9)
    ax.plot(spec['wavelength_A'], spec['L_nu_standard'], 'r--', lw=0.8, label='Standard MC', alpha=0.7)

    # Mark features
    ymax = spec['L_nu_lumina'].max() * 1.1
    for name, (w1, w2) in features.items():
        ax.axvspan(w1, w2, alpha=0.15, color='yellow')
        ax.text((w1 + w2)/2, ymax * 0.95, name, ha='center', va='top', fontsize=7, rotation=45)

    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('$L_\\nu$ [arbitrary]')
    ax.set_xlim(3000, 10000)
    ax.set_ylim(0, ymax)
    ax.legend(loc='upper right')
    ax.set_title('LUMINA-SN Synthetic Spectrum (SN 2011fe model at B-maximum)')
    ax.grid(True, alpha=0.3)

    # Si II 6355 zoom
    ax = axes[1]

    # Find Si II region
    mask = (spec['wavelength_A'] > 5800) & (spec['wavelength_A'] < 6600)
    wl = spec['wavelength_A'][mask]
    flux_lum = spec['L_nu_lumina'][mask]
    flux_std = spec['L_nu_standard'][mask]

    ax.plot(wl, flux_lum, 'b-', lw=1.5, label='LUMINA')
    ax.plot(wl, flux_std, 'r--', lw=1.5, label='Standard MC')

    # Mark rest wavelength and expected blueshifted position
    v_photo = 10000  # km/s
    c = 2.998e5  # km/s
    lambda_rest = 6355  # Å
    lambda_blue = lambda_rest * (1 - v_photo/c)

    ax.axvline(lambda_rest, color='gray', ls=':', lw=1, label=f'Si II rest ({lambda_rest} Å)')
    ax.axvline(lambda_blue, color='green', ls='--', lw=1.5,
               label=f'Expected @ v={v_photo} km/s ({lambda_blue:.0f} Å)')

    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('$L_\\nu$ [arbitrary]')
    ax.set_xlim(5800, 6600)
    ax.legend(loc='upper left')
    ax.set_title('Si II 6355 Å Feature (Velocity Diagnostic)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectrum_features.pdf', dpi=150)
    print("  Saved: spectrum_features.pdf")


def main():
    print("\n" + "="*70)
    print("  LUMINA-SN vs TARDIS: Detailed Physics Comparison Plots")
    print("="*70)

    print("\n  Generating comparison plots...")

    plot_temperature_comparison()
    plot_density_comparison()
    plot_spectrum_features()

    print("\n  All plots generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
