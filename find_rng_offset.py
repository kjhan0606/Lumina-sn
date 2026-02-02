#!/usr/bin/env python3
"""
Find the RNG offset between TARDIS initialization and transport loop.
Traces the first random number used for tau_event calculation.
"""

import numpy as np
from astropy import units as u

def find_first_transport_rng():
    """
    Generate RNG stream and trace TARDIS to find first transport RNG usage.
    """

    # Generate the same RNG stream we exported
    seed = 23111963
    np.random.seed(seed)

    # Pre-generate many random numbers
    n_rng = 1000
    rng_stream = np.random.random(n_rng)

    print("=" * 70)
    print("RNG OFFSET FINDER")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Pre-generated {n_rng} RNG values")
    print()

    # Show first 20 values with indices
    print("First 20 RNG values:")
    print("-" * 40)
    for i in range(20):
        print(f"  [{i:3d}] {rng_stream[i]:.16e}")
    print()

    # Now run TARDIS and try to capture the first transport RNG
    print("=" * 70)
    print("Running TARDIS to trace RNG usage...")
    print("=" * 70)

    from tardis.io.configuration.config_reader import Configuration
    from tardis.simulation import Simulation

    config_dict = {
        'tardis_config_version': 'v1.0',
        'supernova': {
            'luminosity_requested': 1e43 * u.erg / u.s,
            'time_explosion': 13 * u.day,
        },
        'atom_data': 'atomic/kurucz_cd23_chianti_H_He.h5',
        'model': {
            'structure': {
                'type': 'specific',
                'velocity': {
                    'start': 11000 * u.km / u.s,
                    'stop': 20000 * u.km / u.s,
                    'num': 20,
                },
                'density': {
                    'type': 'power_law',
                    'time_0': 13 * u.day,
                    'rho_0': 1e-13 * u.g / u.cm**3,
                    'v_0': 11000 * u.km / u.s,
                    'exponent': -7,
                },
            },
            'abundances': {
                'type': 'uniform',
                'O': 0.19,
                'Mg': 0.03,
                'Si': 0.52,
                'S': 0.19,
                'Ar': 0.04,
                'Ca': 0.03,
            },
        },
        'plasma': {
            'ionization': 'lte',
            'excitation': 'lte',
            'radiative_rates_type': 'dilute-blackbody',
            'line_interaction_type': 'scatter',
        },
        'montecarlo': {
            'seed': seed,
            'no_of_packets': 1,  # Single packet for tracing
            'iterations': 1,
            'last_no_of_packets': 1,
            'no_of_virtual_packets': 0,
        },
        'spectrum': {
            'start': 3000 * u.angstrom,
            'stop': 10000 * u.angstrom,
            'num': 1000,
        },
    }

    # Reset RNG to same seed before TARDIS
    np.random.seed(seed)

    config = Configuration.from_config_dict(config_dict)
    sim = Simulation.from_config(config)

    # Check RNG state before transport
    # NumPy's MT state is complex, but we can check by drawing a number
    print("\nChecking RNG state before/after initialization...")

    # Reset again and count how many draws happen during init
    np.random.seed(seed)
    pre_init_count = 0

    # We can't easily hook into TARDIS, but we know:
    # - Packet initialization uses RNG for: r, mu, nu, energy
    # - Each packet needs several draws for blackbody sampling

    # Let's analyze what we know from LUMINA's trace:
    print("\n" + "=" * 70)
    print("Analysis from LUMINA trace:")
    print("=" * 70)

    # LUMINA's first tau_event draw was index 0: xi = 0.6146...
    # Let's find this in the stream
    lumina_first_xi = 0.6146026678705826

    print(f"\nLUMINA's first xi (Shell 0): {lumina_first_xi:.16e}")

    # Search in stream
    for i, val in enumerate(rng_stream):
        if abs(val - lumina_first_xi) < 1e-10:
            print(f"Found at index {i}: {val:.16e}")
            break

    # If TARDIS uses a different starting point, we need to find it
    # Let's check what values would give us small tau (high xi)
    print("\n" + "=" * 70)
    print("RNG values that would cause ESCATTERING (xi > 0.9):")
    print("=" * 70)

    for i, val in enumerate(rng_stream[:50]):
        if val > 0.9:
            tau = -np.log(val)
            print(f"  [{i:3d}] xi = {val:.16e}, tau = {tau:.6f}")

    # The key insight: if TARDIS gets ESCATTERING, it drew a high xi
    # For Shell 0 with n_e=3.27e9, d_boundary=5.8e13:
    # tau_threshold = sigma_T * n_e * d_boundary = 0.126
    # For ESCATTERING: tau < 0.126 => xi > exp(-0.126) = 0.882

    print("\n" + "=" * 70)
    print("Searching for ESCATTERING-causing xi values:")
    print("=" * 70)

    sigma_T = 6.6524587e-25
    n_e_shell0 = 3.27e9
    d_boundary_shell0 = 5.8e13
    tau_threshold = sigma_T * n_e_shell0 * d_boundary_shell0
    xi_threshold = np.exp(-tau_threshold)

    print(f"tau_threshold = {tau_threshold:.6f}")
    print(f"xi_threshold = {xi_threshold:.6f} (need xi > this for ESCATTERING)")
    print()

    for i, val in enumerate(rng_stream[:100]):
        if val > xi_threshold:
            tau = -np.log(val)
            print(f"  [{i:3d}] xi = {val:.16e}, tau = {tau:.6f} -> ESCATTERING")

    return rng_stream

if __name__ == '__main__':
    find_first_transport_rng()
