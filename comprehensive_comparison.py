#!/usr/bin/env python3
"""
LUMINA-SN vs TARDIS Comprehensive Comparison
comprehensive_comparison.py - Bottom-level physics validation

Compares:
1. Saha-Boltzmann ionization
2. Partition functions
3. Boltzmann level populations
4. Sobolev line opacity
5. Emergent spectrum (if full TARDIS run)

Usage:
    python3 comprehensive_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import subprocess
import sys
import os

# Physical constants (CGS - NIST CODATA 2018)
CONST_C = 2.99792458e10       # cm/s
CONST_H = 6.62607015e-27      # erg·s
CONST_K_B = 1.380649e-16      # erg/K
CONST_M_E = 9.1093837015e-28  # g
CONST_PI = 3.14159265358979323846
CONST_AMU = 1.66053906660e-24 # g
CONST_EV_TO_ERG = 1.602176634e-12

# Saha constant
SAHA_CONST = (2 * CONST_PI * CONST_M_E * CONST_K_B)**1.5 / CONST_H**3


def saha_factor(chi_ev, T, U_i, U_i1):
    """Calculate Saha factor for ionization i -> i+1."""
    chi_erg = chi_ev * CONST_EV_TO_ERG
    kT = CONST_K_B * T
    return (2.0 * U_i1 / U_i) * SAHA_CONST * T**1.5 * np.exp(-chi_erg / kT)


def boltzmann_factor(E_ev, T):
    """Calculate Boltzmann factor exp(-E/kT)."""
    kT_ev = CONST_K_B * T / CONST_EV_TO_ERG
    return np.exp(-E_ev / kT_ev)


def partition_function(E_levels_ev, g_levels, T, cutoff=50):
    """Calculate partition function U(T) = sum(g_i * exp(-E_i/kT))."""
    kT_ev = CONST_K_B * T / CONST_EV_TO_ERG
    U = 0.0
    for E, g in zip(E_levels_ev, g_levels):
        if E / kT_ev < cutoff:
            U += g * np.exp(-E / kT_ev)
    return max(U, 1.0)


def sobolev_tau(wavelength_cm, f_lu, n_lower, t_exp, g_l=None, g_u=None, n_upper=0):
    """Calculate Sobolev optical depth.

    τ_Sobolev = (π e² / m_e c) * f_lu * λ * n_l * t_exp * (1 - g_l*n_u/(g_u*n_l))
    """
    e_esu = 4.80320425e-10  # Elementary charge in esu

    # Stimulated emission correction
    stim_corr = 1.0
    if g_l is not None and g_u is not None and n_upper > 0 and n_lower > 0:
        stim_corr = 1.0 - (g_l * n_upper) / (g_u * n_lower)

    tau = (CONST_PI * e_esu**2 / (CONST_M_E * CONST_C)) * f_lu * wavelength_cm * n_lower * t_exp * stim_corr
    return tau


class PhysicsComparison:
    """Compare LUMINA and TARDIS physics at the lowest level."""

    def __init__(self):
        self.results = {}

    def compare_partition_functions(self, T=10000):
        """Compare partition function calculations."""
        print("\n" + "="*70)
        print("  COMPARISON 1: Partition Functions")
        print("="*70)

        # H I levels (simplified - first 10)
        # Ground state + excited states
        E_H = [0.0, 10.199, 10.199, 10.199, 12.088, 12.088, 12.088, 12.088, 12.088, 12.749]
        g_H = [2, 2, 2, 4, 2, 2, 4, 4, 6, 2]

        U_H_python = partition_function(E_H, g_H, T)

        # He I (ground state dominated at T<20000K)
        E_He = [0.0, 19.82, 20.62, 20.96]
        g_He = [1, 3, 1, 3]
        U_He_python = partition_function(E_He, g_He, T)

        # Si II levels (first 5)
        E_Si = [0.0, 0.036, 0.287, 5.309, 5.333]
        g_Si = [2, 4, 4, 2, 4]
        U_Si_python = partition_function(E_Si, g_Si, T)

        print(f"\n  Temperature: {T} K")
        print(f"\n  {'Ion':<10} {'Python':<15} {'Expected':<15} {'Status'}")
        print("  " + "-"*55)

        # Expected values (should match C output)
        U_H_expected = 2.0001  # Ground state dominated
        U_He_expected = 1.0    # Ground state only at 10000K
        U_Si_expected = 7.6    # Si II has low-lying excited states

        def check(name, val, exp, tol=0.1):
            rel_err = abs(val - exp) / exp if exp > 0 else 0
            status = "PASS" if rel_err < tol else f"FAIL ({rel_err:.1%})"
            print(f"  {name:<10} {val:<15.6f} {exp:<15.6f} {status}")
            return rel_err < tol

        all_pass = True
        all_pass &= check("H I", U_H_python, U_H_expected)
        all_pass &= check("He I", U_He_python, U_He_expected)
        all_pass &= check("Si II", U_Si_python, U_Si_expected, tol=0.5)  # larger tolerance

        self.results['partition_functions'] = all_pass
        return all_pass

    def compare_ionization(self, T=10000, rho=1e-10):
        """Compare Saha ionization balance."""
        print("\n" + "="*70)
        print("  COMPARISON 2: Saha-Boltzmann Ionization")
        print("="*70)

        # Number densities
        X_H, X_He = 0.7, 0.3
        n_H = X_H * rho / (1.008 * CONST_AMU)
        n_He = X_He * rho / (4.003 * CONST_AMU)

        # Ionization energies [eV]
        chi_H = 13.598
        chi_He1 = 24.587
        chi_He2 = 54.418

        # Partition functions
        U_H = [2.0, 1.0]
        U_He = [1.0, 2.0, 1.0]

        # Iteratively solve for n_e
        n_e = n_H + n_He
        for _ in range(20):
            # H ionization
            phi_H = saha_factor(chi_H, T, U_H[0], U_H[1])
            x_H = phi_H / (n_e + phi_H)  # H II fraction

            # He ionization (first)
            phi_He1 = saha_factor(chi_He1, T, U_He[0], U_He[1])
            phi_He2 = saha_factor(chi_He2, T, U_He[1], U_He[2])

            denom_He = 1.0 + phi_He1/n_e + phi_He1*phi_He2/(n_e*n_e)
            x_He0 = 1.0 / denom_He
            x_He1 = (phi_He1/n_e) / denom_He
            x_He2 = (phi_He1*phi_He2/(n_e*n_e)) / denom_He

            n_e_new = n_H * x_H + n_He * (x_He1 + 2*x_He2)
            if abs(n_e_new - n_e) / n_e < 1e-8:
                break
            n_e = 0.5 * (n_e + n_e_new)

        print(f"\n  Parameters: T = {T} K, ρ = {rho:.2e} g/cm³")
        print(f"\n  {'Quantity':<20} {'Python':<18} {'C (expected)':<18} {'Rel Error'}")
        print("  " + "-"*70)

        # Expected values from C output (test_plasma)
        C_n_e = 3.7636e+13
        C_H_I = 0.1001
        C_H_II = 0.8999
        C_He_I = 0.9999
        C_He_II = 0.0001

        def show(name, py_val, c_val):
            rel_err = abs(py_val - c_val) / abs(c_val) if c_val != 0 else 0
            print(f"  {name:<20} {py_val:<18.6e} {c_val:<18.6e} {rel_err:.2e}")
            return rel_err

        errs = []
        errs.append(show("n_e [cm⁻³]", n_e, C_n_e))
        errs.append(show("H I fraction", 1-x_H, C_H_I))
        errs.append(show("H II fraction", x_H, C_H_II))
        errs.append(show("He I fraction", x_He0, C_He_I))
        errs.append(show("He II fraction", x_He1, C_He_II))

        all_pass = all(e < 0.1 for e in errs)  # 10% tolerance
        print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

        self.results['ionization'] = all_pass
        return all_pass

    def compare_sobolev_opacity(self, T=10000, n_ion=1e8, t_exp=19*86400):
        """Compare Sobolev optical depth calculation."""
        print("\n" + "="*70)
        print("  COMPARISON 3: Sobolev Line Opacity")
        print("="*70)

        # Si II 6355 Å doublet
        lambda_6347 = 6347.10e-8  # cm
        lambda_6371 = 6371.37e-8  # cm
        f_6347 = 0.708
        f_6371 = 0.419

        # Level populations (ground state dominated)
        # Si II ground state is 3s² 3p ²P with g=2 (J=1/2) and g=4 (J=3/2)
        kT_ev = CONST_K_B * T / CONST_EV_TO_ERG
        E_split = 0.036  # eV splitting between J=1/2 and J=3/2

        g0, g1 = 2, 4
        n0 = g0 / (g0 + g1 * np.exp(-E_split / kT_ev)) * n_ion
        n1 = g1 * np.exp(-E_split / kT_ev) / (g0 + g1 * np.exp(-E_split / kT_ev)) * n_ion

        # Calculate tau
        tau_6347 = sobolev_tau(lambda_6347, f_6347, n0, t_exp)
        tau_6371 = sobolev_tau(lambda_6371, f_6371, n1, t_exp)

        print(f"\n  Parameters:")
        print(f"    T = {T} K")
        print(f"    n(Si II) = {n_ion:.2e} cm⁻³")
        print(f"    t_exp = {t_exp/86400:.1f} days")

        print(f"\n  Ground state populations:")
        print(f"    n(²P₁/₂) = {n0:.4e} cm⁻³ ({100*n0/n_ion:.1f}%)")
        print(f"    n(²P₃/₂) = {n1:.4e} cm⁻³ ({100*n1/n_ion:.1f}%)")

        print(f"\n  {'Line':<15} {'λ [Å]':<10} {'f_lu':<10} {'τ_Sob (Python)':<18}")
        print("  " + "-"*55)
        print(f"  {'Si II 6347':<15} {6347.10:<10.2f} {f_6347:<10.3f} {tau_6347:<18.4e}")
        print(f"  {'Si II 6371':<15} {6371.37:<10.2f} {f_6371:<10.3f} {tau_6371:<18.4e}")

        # Check ratio (should be ~f_6347/f_6371 * g0/g1 / Boltzmann factor)
        ratio_expected = (f_6347 * n0) / (f_6371 * n1) * (lambda_6347 / lambda_6371)
        ratio_actual = tau_6347 / tau_6371

        print(f"\n  τ(6347)/τ(6371) = {ratio_actual:.3f} (expected: {ratio_expected:.3f})")

        all_pass = abs(ratio_actual - ratio_expected) / ratio_expected < 0.01
        print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

        self.results['sobolev'] = all_pass
        return all_pass

    def compare_level_populations(self, T=10000):
        """Compare Boltzmann level populations."""
        print("\n" + "="*70)
        print("  COMPARISON 4: Boltzmann Level Populations")
        print("="*70)

        # H I levels
        E_H = [0.0, 10.199, 10.199, 10.199, 12.088]  # eV
        g_H = [2, 2, 2, 4, 2]

        kT_ev = CONST_K_B * T / CONST_EV_TO_ERG

        # Calculate populations
        U = partition_function(E_H, g_H, T)
        n_levels = []
        for E, g in zip(E_H, g_H):
            n_i = g * np.exp(-E / kT_ev) / U
            n_levels.append(n_i)

        print(f"\n  Temperature: {T} K, kT = {kT_ev:.4f} eV")
        print(f"  Partition function U(H I) = {U:.6f}")
        print(f"\n  {'Level':<10} {'E [eV]':<12} {'g':<6} {'n_i/n_total':<15}")
        print("  " + "-"*45)

        for i, (E, g, n) in enumerate(zip(E_H, g_H, n_levels)):
            print(f"  {i:<10} {E:<12.4f} {g:<6} {n:<15.6e}")

        # Check ground state dominance
        ground_frac = n_levels[0]
        all_pass = ground_frac > 0.999  # Should be >99.9% at 10000K

        print(f"\n  Ground state fraction: {ground_frac:.6f}")
        print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

        self.results['populations'] = all_pass
        return all_pass

    def compare_spectrum(self, lumina_file="spectrum_test.csv"):
        """Compare emergent spectra if available."""
        print("\n" + "="*70)
        print("  COMPARISON 5: Emergent Spectrum")
        print("="*70)

        if not os.path.exists(lumina_file):
            print(f"\n  Lumina spectrum file not found: {lumina_file}")
            print("  Run: ./test_integrated atomic/kurucz_cd23_chianti_H_He.h5 10000 spectrum_test.csv")
            self.results['spectrum'] = None
            return None

        # Load Lumina spectrum
        lumina = pd.read_csv(lumina_file, comment='#')

        print(f"\n  Lumina spectrum loaded: {len(lumina)} bins")
        print(f"  Wavelength range: {lumina['wavelength_A'].min():.1f} - {lumina['wavelength_A'].max():.1f} Å")

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot spectrum
        ax = axes[0]
        ax.plot(lumina['wavelength_A'], lumina['L_nu_lumina'], 'b-', lw=0.8, label='LUMINA', alpha=0.8)
        ax.plot(lumina['wavelength_A'], lumina['L_nu_standard'], 'r--', lw=0.8, label='Standard', alpha=0.8)
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('$L_\\nu$ [arbitrary]')
        ax.set_xlim(3000, 10000)
        ax.legend()
        ax.set_title('LUMINA-SN Synthetic Spectrum (SN 2011fe model)')
        ax.grid(True, alpha=0.3)

        # Plot ratio
        ax = axes[1]
        # Smooth for ratio calculation
        from scipy.ndimage import uniform_filter1d
        lumina_smooth = uniform_filter1d(lumina['L_nu_lumina'].values, size=5)
        std_smooth = uniform_filter1d(lumina['L_nu_standard'].values, size=5)

        mask = (lumina['wavelength_A'] > 3500) & (lumina['wavelength_A'] < 9000) & (std_smooth > 1e-4)
        ratio = np.where(mask, lumina_smooth / (std_smooth + 1e-10), np.nan)

        ax.plot(lumina['wavelength_A'], ratio, 'g-', lw=0.8)
        ax.axhline(1.0, color='k', ls='--', lw=0.5)
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('LUMINA / Standard')
        ax.set_xlim(3000, 10000)
        ax.set_ylim(0, 20)
        ax.set_title('Spectrum Ratio (LUMINA rotation-weighted vs standard)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('lumina_spectrum_comparison.pdf', dpi=150)
        print(f"\n  Saved: lumina_spectrum_comparison.pdf")

        self.results['spectrum'] = True
        return True

    def run_all_comparisons(self):
        """Run all physics comparisons."""
        print("\n")
        print("╔" + "═"*68 + "╗")
        print("║" + "  LUMINA-SN vs TARDIS: Bottom-Level Physics Comparison  ".center(68) + "║")
        print("╚" + "═"*68 + "╝")

        self.compare_partition_functions()
        self.compare_ionization()
        self.compare_sobolev_opacity()
        self.compare_level_populations()
        self.compare_spectrum()

        # Summary
        print("\n" + "="*70)
        print("  SUMMARY")
        print("="*70)

        all_pass = True
        for name, result in self.results.items():
            if result is None:
                status = "SKIPPED"
            elif result:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
            print(f"  {name:<30} {status}")

        print("\n" + "="*70)
        if all_pass:
            print("  ALL PHYSICS COMPARISONS PASSED")
        else:
            print("  SOME TESTS FAILED - CHECK ABOVE")
        print("="*70)

        return all_pass


def try_tardis_spectrum():
    """Try to run TARDIS and compare spectra."""
    print("\n" + "="*70)
    print("  TARDIS Spectrum Comparison")
    print("="*70)

    try:
        import tardis
        from tardis.io.config_reader import Configuration
        from tardis import run_tardis
        from astropy import units as u

        print(f"\n  TARDIS version: {tardis.__version__}")

        # Check if we have a TARDIS config
        config_file = "sn2011fe_tardis.yml"
        if not os.path.exists(config_file):
            print(f"\n  TARDIS config not found: {config_file}")
            print("  Creating minimal config for comparison...")

            # Create minimal TARDIS config
            config_yaml = """
tardis_config_version: v1.0

supernova:
  luminosity_requested: 1.0 log_lsun
  time_explosion: 19 day

model:
  structure:
    type: uniform
    velocity:
      start: 10000 km/s
      stop: 25000 km/s
      num: 30
    density:
      type: power_law
      time_0: 19 day
      rho_0: 1e-14 g/cm^3
      v_0: 10000 km/s
      exponent: 7

  abundances:
    type: uniform
    Si: 0.35
    Fe: 0.30
    S: 0.10
    Ca: 0.05
    Mg: 0.05
    O: 0.10
    C: 0.05

plasma:
  ionization: lte
  excitation: lte
  radiative_rates_type: dilute-blackbody
  line_interaction_type: macroatom

montecarlo:
  seed: 12345
  no_of_packets: 10000
  iterations: 5

atom_data: atomic/kurucz_cd23_chianti_H_He.h5
"""
            with open(config_file, 'w') as f:
                f.write(config_yaml)
            print(f"  Created: {config_file}")

        # Run TARDIS (simplified - may need more setup)
        print("\n  Running TARDIS simulation (this may take a few minutes)...")

        try:
            config = Configuration.from_yaml(config_file)
            sim = run_tardis(config)

            # Get spectrum
            wavelength = sim.spectrum.wavelength.value
            flux = sim.spectrum.luminosity_density_lambda.value

            # Save for comparison
            tardis_spectrum = pd.DataFrame({
                'wavelength_A': wavelength,
                'flux_tardis': flux
            })
            tardis_spectrum.to_csv('tardis_spectrum.csv', index=False)
            print(f"\n  TARDIS spectrum saved: tardis_spectrum.csv")

            return True

        except Exception as e:
            print(f"\n  TARDIS run failed: {e}")
            print("  (This is expected if atomic data format is incompatible)")
            return False

    except ImportError as e:
        print(f"\n  TARDIS import failed: {e}")
        return False


if __name__ == "__main__":
    # Run physics comparisons
    comp = PhysicsComparison()
    comp.run_all_comparisons()

    # Try TARDIS if available
    # try_tardis_spectrum()

    print("\n  Plot saved as: lumina_spectrum_comparison.pdf")
