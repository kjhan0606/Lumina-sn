#!/usr/bin/env python3
"""
Export TARDIS electron densities for LUMINA validation.
Creates tardis_electron_densities.txt with one n_e value per shell.
"""

import numpy as np
from astropy import units as u

# Try to use existing TARDIS simulation or create one
def export_electron_densities():
    """Export electron densities from TARDIS simulation."""

    from tardis.io.configuration.config_reader import Configuration
    from tardis.simulation import Simulation

    # Configuration matching LUMINA's test_transport.c setup
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
            'line_interaction_type': 'macroatom',
        },
        'montecarlo': {
            'seed': 23111963,
            'no_of_packets': 10000,
            'iterations': 3,
            'last_no_of_packets': 10000,
            'no_of_virtual_packets': 0,
        },
        'spectrum': {
            'start': 3000 * u.angstrom,
            'stop': 10000 * u.angstrom,
            'num': 1000,
        },
    }

    print("=" * 70)
    print("TARDIS Electron Density Exporter")
    print("=" * 70)

    # Create and run simulation
    print("\nInitializing TARDIS simulation...")
    config = Configuration.from_config_dict(config_dict)
    sim = Simulation.from_config(config)

    print("Running TARDIS iteration to converge plasma state...")
    sim.iterate(no_of_packets=10000)

    # Extract electron densities
    # TARDIS stores electron_densities in the plasma object (after iteration)
    # Try different access patterns for different TARDIS versions
    try:
        electron_densities = sim.plasma.electron_densities.values
    except AttributeError:
        try:
            electron_densities = sim.simulation_state.plasma.electron_densities.values
        except AttributeError:
            # Newer TARDIS version - plasma is separate
            electron_densities = sim.transport.transport_state.electron_densities

    # Get shell radii for reference
    try:
        r_inner = sim.simulation_state.geometry.r_inner.to(u.cm).value
        r_outer = sim.simulation_state.geometry.r_outer.to(u.cm).value
    except AttributeError:
        r_inner = sim.model.r_inner.to(u.cm).value
        r_outer = sim.model.r_outer.to(u.cm).value

    n_shells = len(electron_densities)

    print(f"\nExtracted {n_shells} shells:")
    print(f"  r_inner[0] = {r_inner[0]:.6e} cm")
    print(f"  r_outer[-1] = {r_outer[-1]:.6e} cm")

    # Save electron densities
    output_file = 'tardis_electron_densities.txt'
    with open(output_file, 'w') as f:
        f.write(f"# TARDIS Electron Densities\n")
        f.write(f"# Seed: 23111963\n")
        f.write(f"# n_shells: {n_shells}\n")
        f.write(f"# t_explosion: 13 days = 1.1232e6 s\n")
        f.write(f"# v_inner: 11000 km/s, v_outer: 20000 km/s\n")
        f.write(f"# Format: shell_id r_inner[cm] r_outer[cm] n_e[cm^-3]\n")
        f.write(f"#\n")

        for i in range(n_shells):
            f.write(f"{i} {r_inner[i]:.16e} {r_outer[i]:.16e} {electron_densities[i]:.16e}\n")

    print(f"\nExported to {output_file}")

    # Also save just n_e values for simple loading
    simple_file = 'tardis_ne_only.txt'
    np.savetxt(simple_file, electron_densities, fmt='%.16e',
               header=f'TARDIS electron densities [cm^-3], n_shells={n_shells}')
    print(f"Exported simple format to {simple_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Electron Density Profile:")
    print("=" * 70)
    print(f"{'Shell':>5} {'r_mid [cm]':>16} {'n_e [cm^-3]':>16}")
    print("-" * 40)
    for i in range(min(n_shells, 10)):  # Show first 10 shells
        r_mid = 0.5 * (r_inner[i] + r_outer[i])
        print(f"{i:5d} {r_mid:16.6e} {electron_densities[i]:16.6e}")
    if n_shells > 10:
        print(f"  ... ({n_shells - 10} more shells)")

    print("-" * 40)
    print(f"Min n_e: {electron_densities.min():.6e} cm^-3")
    print(f"Max n_e: {electron_densities.max():.6e} cm^-3")
    print(f"Mean n_e: {electron_densities.mean():.6e} cm^-3")

    return electron_densities, r_inner, r_outer

if __name__ == '__main__':
    export_electron_densities()
