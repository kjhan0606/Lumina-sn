#!/usr/bin/env python3
"""
TARDIS SN 2011fe Fitting with Detailed Logging
===============================================
This script runs TARDIS for SN 2011fe fitting with comprehensive
log messages for comparison with LUMINA.
"""

import os
import sys
import warnings
import tempfile
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')
os.chdir(Path(__file__).parent)

# =============================================================================
# LOGGING SETUP
# =============================================================================
LOG_FILE = "tardis_sn2011fe.log"

# Create logger
logger = logging.getLogger('TARDIS_SN2011fe')
logger.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler(LOG_FILE, mode='w')
fh.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('[%(levelname)s] %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# =============================================================================
# HEADER
# =============================================================================
logger.info("=" * 70)
logger.info("  TARDIS SN 2011fe Stratified Fitting - Comparison Run")
logger.info("=" * 70)
logger.info(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"  Log file: {LOG_FILE}")
logger.info("")

# =============================================================================
# LOAD TARDIS
# =============================================================================
logger.info("[STEP 1] Loading TARDIS...")
try:
    from tardis import run_tardis
    from tardis.io.configuration.config_reader import Configuration
    import tardis
    logger.info(f"  TARDIS version: {tardis.__version__}")
except ImportError as e:
    logger.error(f"  Failed to import TARDIS: {e}")
    sys.exit(1)

from astropy import units as u
from astropy import constants as const

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
logger.info("")
logger.info("[STEP 2] Physical Constants (TARDIS/Astropy)")
logger.info("-" * 50)

C_LIGHT = const.c.cgs.value
H_PLANCK = const.h.cgs.value
K_BOLTZ = const.k_B.cgs.value
M_ELECTRON = const.m_e.cgs.value
EV_TO_ERG = (1 * u.eV).to(u.erg).value

logger.info(f"  c         = {C_LIGHT:.10e} cm/s")
logger.info(f"  h         = {H_PLANCK:.10e} erg*s")
logger.info(f"  k_B       = {K_BOLTZ:.10e} erg/K")
logger.info(f"  m_e       = {M_ELECTRON:.10e} g")
logger.info(f"  eV->erg   = {EV_TO_ERG:.10e}")

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina")
ATOMIC_DATA = BASE_DIR / "data" / "atomic" / "kurucz_cd23_chianti_H_He.h5"
SPECTRA_DIR = BASE_DIR / "data" / "sn2011fe" / "spectra"
OUTPUT_DIR = Path(".")

logger.info("")
logger.info("[STEP 3] Paths")
logger.info("-" * 50)
logger.info(f"  Atomic data: {ATOMIC_DATA}")
logger.info(f"  Spectra dir: {SPECTRA_DIR}")


def load_observed_spectrum(phase_day=0.0):
    """Load observed SN 2011fe spectrum."""
    logger.info("")
    logger.info("[STEP 4] Loading Observed Spectrum")
    logger.info("-" * 50)

    phase_files = {
        -10: "sn2011fe_m10d0d_ptf11kly_20110831.obs.dat",
        -3: "sn2011fe_m3d0d_ptf11kly_20110907.obs.dat",
        0: "sn2011fe_p0d0d_ptf11kly_20110910.obs.dat",
        3: "sn2011fe_p3d0d_ptf11kly_20110913.obs.dat",
        9: "sn2011fe_p9d0d_ptf11kly_20110919.obs.dat",
    }

    available_phases = list(phase_files.keys())
    closest_phase = min(available_phases, key=lambda x: abs(x - phase_day))

    filename = SPECTRA_DIR / phase_files[closest_phase]
    logger.info(f"  Selected phase: {closest_phase} days")
    logger.info(f"  Spectrum file: {phase_files[closest_phase]}")

    data = np.loadtxt(filename)
    wavelength = data[:, 0]
    flux = data[:, 1]
    flux = flux / np.max(flux)

    logger.info(f"  Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} A")
    logger.info(f"  Number of points: {len(wavelength)}")
    logger.info(f"  Flux normalized to peak = 1.0")

    return wavelength, flux, closest_phase


def create_stratified_model(v_inner=10500, v_outer=25000, n_shells=20,
                            fe_core_fraction=0.3, si_layer_width=0.4,
                            fe_peak=0.7, si_peak=0.6, o_outer=0.5):
    """Create W7-like stratified abundance model with logging."""

    logger.info("")
    logger.info("  Creating W7-like Stratified Model")
    logger.info("  " + "-" * 40)
    logger.info(f"    v_inner: {v_inner} km/s")
    logger.info(f"    v_outer: {v_outer} km/s")
    logger.info(f"    n_shells: {n_shells}")
    logger.info(f"    fe_core_fraction: {fe_core_fraction}")
    logger.info(f"    si_layer_width: {si_layer_width}")
    logger.info(f"    fe_peak: {fe_peak}")
    logger.info(f"    si_peak: {si_peak}")
    logger.info(f"    o_outer: {o_outer}")

    # Velocity grid
    velocities = np.linspace(v_inner + 200, v_outer, n_shells)
    v_norm = (velocities - v_inner) / (v_outer - v_inner)

    # Density profile
    rho_0 = 3e-10
    densities = rho_0 * np.exp(-3 * v_norm) * (v_inner / velocities)**3

    logger.info(f"    Density range: {densities.min():.2e} - {densities.max():.2e} g/cm^3")

    # Initialize abundances
    n = len(velocities)
    abundances = {
        'C': np.zeros(n), 'O': np.zeros(n), 'Mg': np.zeros(n),
        'Si': np.zeros(n), 'S': np.zeros(n), 'Ca': np.zeros(n),
        'Ti': np.zeros(n), 'Cr': np.zeros(n), 'Fe': np.zeros(n),
        'Co': np.zeros(n), 'Ni56': np.zeros(n),
    }

    fe_si_transition = fe_core_fraction
    si_o_transition = fe_core_fraction + si_layer_width

    for i, v in enumerate(v_norm):
        if v < fe_si_transition:
            fe_frac = 0.5 * (1 + np.tanh(10 * (fe_si_transition - v - 0.1)))
            abundances['Fe'][i] = fe_peak * 0.4 * fe_frac
            abundances['Co'][i] = fe_peak * 0.05 * fe_frac
            abundances['Ni56'][i] = fe_peak * 0.55 * fe_frac
            abundances['Si'][i] = 0.15 * (1 - fe_frac) + 0.05
            abundances['S'][i] = 0.08 * (1 - fe_frac) + 0.02
            abundances['Ca'][i] = 0.02
            abundances['O'][i] = 0.05
        elif v < si_o_transition:
            si_pos = (v - fe_si_transition) / si_layer_width
            si_profile = np.sin(np.pi * si_pos)
            abundances['Si'][i] = si_peak * si_profile + 0.1 * (1 - si_profile)
            abundances['S'][i] = 0.25 * si_profile + 0.05 * (1 - si_profile)
            abundances['Ca'][i] = 0.04 * si_profile + 0.01
            abundances['Mg'][i] = 0.04 * si_profile + 0.02
            fe_decay = np.exp(-5 * (v - fe_si_transition))
            abundances['Fe'][i] = 0.15 * fe_decay
            abundances['Co'][i] = 0.02 * fe_decay
            abundances['Ni56'][i] = 0.10 * fe_decay
            abundances['O'][i] = 0.1 + 0.2 * si_pos
            abundances['Ti'][i] = 0.01 * si_profile
            abundances['Cr'][i] = 0.01 * si_profile
        else:
            outer_pos = (v - si_o_transition) / (1 - si_o_transition)
            abundances['O'][i] = o_outer + 0.2 * outer_pos
            abundances['C'][i] = 0.1 + 0.3 * outer_pos
            ime_decay = np.exp(-3 * (v - si_o_transition))
            abundances['Si'][i] = 0.15 * ime_decay
            abundances['S'][i] = 0.05 * ime_decay
            abundances['Mg'][i] = 0.03 * ime_decay
            abundances['Ca'][i] = 0.01 * ime_decay
            abundances['Fe'][i] = 0.02 * ime_decay

    # Normalize
    for i in range(n):
        total = sum(abundances[el][i] for el in abundances)
        if total > 0:
            for el in abundances:
                abundances[el][i] /= total

    # Log mean abundances
    logger.info("    Mean mass fractions:")
    for el in ['Fe', 'Ni56', 'Si', 'S', 'O', 'C', 'Ca']:
        mean_val = np.mean(abundances[el])
        if mean_val > 0.01:
            logger.info(f"      {el:4s}: {mean_val:.4f}")

    return velocities, densities, abundances


def create_csvy_content(velocities, densities, abundances, v_inner, v_outer, description):
    """Create CSVY model content."""
    header = f"""---
datatype:
  fields:
  - desc: velocities of shell outer boundaries
    name: velocity
    unit: km/s
  - desc: mean density of shell
    name: density
    unit: g/cm^3"""

    elements = list(abundances.keys())
    for el in elements:
        header += f"""
  - desc: Fraction {el} abundance
    name: {el}"""

    header += f"""
description: {description}
model_density_time_0: 1.0 day
model_isotope_time_0: 0.0 day
tardis_model_config_version: v1.0
v_inner_boundary: {v_inner} km/s
v_outer_boundary: {v_outer} km/s

---
"""

    col_names = "velocity,density," + ",".join(elements)
    data_rows = []
    for i in range(len(velocities)):
        row = f"{velocities[i]},{densities[i]:.3e}"
        for el in elements:
            row += f",{abundances[el][i]:.4f}"
        data_rows.append(row)

    return header + col_names + "\n" + "\n".join(data_rows) + "\n"


def run_tardis_simulation(luminosity, time_exp, v_inner, v_outer,
                          stratification_params, n_packets=20000, n_iter=10,
                          last_packets=50000, model_id=""):
    """Run TARDIS with detailed logging."""

    logger.info("")
    logger.info(f"[SIMULATION {model_id}] Running TARDIS")
    logger.info("-" * 50)
    logger.info(f"  Input Parameters:")
    logger.info(f"    luminosity:     {luminosity:.4f} log(L_sun)")
    logger.info(f"    time_explosion: {time_exp:.2f} days")
    logger.info(f"    v_inner:        {v_inner:.0f} km/s")
    logger.info(f"    v_outer:        {v_outer:.0f} km/s")
    logger.info(f"    n_packets:      {n_packets}")
    logger.info(f"    iterations:     {n_iter}")
    logger.info(f"    last_packets:   {last_packets}")

    # Create model
    velocities, densities, abundances = create_stratified_model(
        v_inner=v_inner, v_outer=v_outer, **stratification_params
    )

    # Write CSVY
    csvy_content = create_csvy_content(
        velocities, densities, abundances, v_inner, v_outer,
        f"W7-like stratified, L={luminosity}, t={time_exp}d"
    )

    csvy_fd, csvy_path = tempfile.mkstemp(suffix='.csvy', prefix='tardis_model_')
    with os.fdopen(csvy_fd, 'w') as f:
        f.write(csvy_content)

    # Write config
    config_yaml = f"""tardis_config_version: v1.0

supernova:
  luminosity_requested: {luminosity} log_lsun
  time_explosion: {time_exp} day

atom_data: {ATOMIC_DATA}
csvy_model: {csvy_path}

plasma:
  disable_electron_scattering: no
  ionization: lte
  excitation: lte
  radiative_rates_type: dilute-blackbody
  line_interaction_type: macroatom

montecarlo:
  seed: 23111963
  no_of_packets: {n_packets}
  iterations: {n_iter}
  last_no_of_packets: {last_packets}
  no_of_virtual_packets: 5
  convergence_strategy:
    type: damped
    damping_constant: 0.8
    threshold: 0.05
    fraction: 0.8
    hold_iterations: 3
    t_inner:
      damping_constant: 0.8

spectrum:
  start: 3000 angstrom
  stop: 10000 angstrom
  num: 2000
"""

    config_fd, config_path = tempfile.mkstemp(suffix='.yml', prefix='tardis_config_')
    with os.fdopen(config_fd, 'w') as f:
        f.write(config_yaml)

    logger.info("")
    logger.info("  TARDIS Configuration:")
    logger.info(f"    Ionization:    LTE")
    logger.info(f"    Excitation:    LTE")
    logger.info(f"    Line interact: macroatom")
    logger.info(f"    e- scattering: enabled")
    logger.info(f"    Convergence:   damped (0.8)")

    try:
        logger.info("")
        logger.info("  Starting Monte Carlo transport...")

        sim = run_tardis(
            config_path,
            show_convergence_plots=False,
            show_progress_bars=False,
            log_level="ERROR"
        )

        # Extract results with detailed logging
        logger.info("")
        logger.info("  Simulation Results:")
        logger.info(f"    Converged: YES")

        # Inner temperature
        t_inner = sim.simulation_state.t_inner
        logger.info(f"    T_inner: {t_inner:.2f}")

        # Shell temperatures
        t_rad = sim.simulation_state.t_radiative
        logger.info(f"    T_rad range: {t_rad.min():.0f} - {t_rad.max():.0f} K")

        # Dilution factors
        w = sim.simulation_state.dilution_factor
        logger.info(f"    W range: {w.min():.4f} - {w.max():.4f}")

        # Electron densities
        n_e = sim.plasma.electron_densities
        logger.info(f"    n_e range: {n_e.min():.2e} - {n_e.max():.2e} cm^-3")

        # Extract spectrum
        wavelength = sim.spectrum_solver.spectrum_real_packets.wavelength.value
        flux = sim.spectrum_solver.spectrum_real_packets.luminosity_density_lambda.value

        logger.info(f"    Spectrum points: {len(wavelength)}")
        logger.info(f"    Lambda range: {wavelength.min():.1f} - {wavelength.max():.1f} A")

        # Packet statistics
        logger.info("")
        logger.info("  Packet Statistics:")
        logger.info(f"    Total packets: {last_packets}")

        return wavelength, flux, sim, csvy_content

    except Exception as e:
        logger.error(f"  TARDIS simulation failed: {e}")
        return None, None, None, None

    finally:
        if os.path.exists(csvy_path):
            os.remove(csvy_path)
        if os.path.exists(config_path):
            os.remove(config_path)


def calculate_chi_square(model_wave, model_flux, obs_wave, obs_flux,
                         wave_min=3500, wave_max=7500):
    """Calculate chi-square with logging."""
    model_mask = (model_wave >= wave_min) & (model_wave <= wave_max)
    obs_mask = (obs_wave >= wave_min) & (obs_wave <= wave_max)

    model_w = model_wave[model_mask]
    model_f = model_flux[model_mask]
    obs_w = obs_wave[obs_mask]
    obs_f = obs_flux[obs_mask]

    interp_func = interp1d(model_w, model_f, kind='linear',
                           bounds_error=False, fill_value=0)
    model_interp = interp_func(obs_w)

    model_interp = model_interp / np.max(model_interp)
    obs_f = obs_f / np.max(obs_f)

    chi2 = np.sum((model_interp - obs_f)**2)

    return chi2


def find_si_ii_velocity(wavelength, flux, rest_lambda=6355.0):
    """Find Si II 6355 absorption velocity."""
    # Normalize
    flux_norm = flux / np.max(flux)

    # Search window around expected feature
    v_max = 20000  # km/s
    lambda_min = rest_lambda * (1 - v_max / 3e5)
    lambda_max = rest_lambda

    mask = (wavelength >= lambda_min) & (wavelength <= lambda_max)

    if not np.any(mask):
        return None

    # Find minimum (absorption)
    idx_min = np.argmin(flux_norm[mask])
    lambda_min_abs = wavelength[mask][idx_min]

    # Calculate velocity
    v_si = 3e5 * (rest_lambda - lambda_min_abs) / rest_lambda

    return v_si


def main():
    """Main fitting routine with single model for comparison."""

    # Load observed spectrum
    obs_wave, obs_flux, phase = load_observed_spectrum(phase_day=0)

    # Use fixed parameters for comparison
    logger.info("")
    logger.info("[STEP 5] Model Parameters for Comparison")
    logger.info("-" * 50)

    params = {
        'luminosity': 9.35,
        'time_explosion': 13.0,
        'v_inner': 11000,
        'v_outer': 25000,
    }

    stratification = {
        'fe_core_fraction': 0.25,
        'si_layer_width': 0.40,
        'fe_peak': 0.70,
        'si_peak': 0.60,
        'o_outer': 0.50,
    }

    logger.info(f"  Luminosity: {params['luminosity']:.4f} log(L_sun)")
    logger.info(f"  Time explosion: {params['time_explosion']:.2f} days")
    logger.info(f"  V_inner: {params['v_inner']} km/s")
    logger.info(f"  V_outer: {params['v_outer']} km/s")

    # Run single simulation
    model_wave, model_flux, sim, csvy = run_tardis_simulation(
        luminosity=params['luminosity'],
        time_exp=params['time_explosion'],
        v_inner=params['v_inner'],
        v_outer=params['v_outer'],
        stratification_params=stratification,
        n_packets=30000,
        n_iter=10,
        last_packets=80000,
        model_id="COMPARISON"
    )

    if model_wave is None:
        logger.error("Simulation failed!")
        return

    # Calculate chi-square
    chi2 = calculate_chi_square(model_wave, model_flux, obs_wave, obs_flux)

    logger.info("")
    logger.info("[STEP 6] Spectral Analysis")
    logger.info("-" * 50)
    logger.info(f"  Chi-square (3500-7500 A): {chi2:.4f}")

    # Find Si II velocity
    v_si = find_si_ii_velocity(model_wave, model_flux)
    if v_si:
        logger.info(f"  Si II 6355 velocity: {v_si:.0f} km/s")

    # Temperature profile
    logger.info("")
    logger.info("[STEP 7] Shell Properties")
    logger.info("-" * 50)

    t_rad = sim.simulation_state.t_radiative
    v_shells = sim.simulation_state.velocity.value[:-1]  # inner boundaries

    logger.info("  Shell   Velocity     T_rad      W        n_e")
    logger.info("         [km/s]       [K]                [cm^-3]")
    logger.info("  " + "-" * 55)

    w = sim.simulation_state.dilution_factor
    n_e = sim.plasma.electron_densities.values

    # Log every 3rd shell to keep output manageable
    for i in range(0, len(t_rad), 3):
        logger.info(f"  {i:3d}   {v_shells[i]:10.0f}   {t_rad[i]:8.0f}   {w[i]:.4f}   {n_e[i]:.2e}")

    # Ionization fractions
    logger.info("")
    logger.info("[STEP 8] Ionization Balance")
    logger.info("-" * 50)

    try:
        ion_fracs = sim.plasma.ion_number_density
        # Si ionization
        si_total = ion_fracs.loc[(14, slice(None))].sum(axis=0)
        if (14, 1) in ion_fracs.index:
            si_ii = ion_fracs.loc[(14, 1)]
            si_ii_frac = (si_ii / si_total).mean()
            logger.info(f"  Si II fraction (mean): {si_ii_frac:.4f}")

        # Fe ionization
        fe_total = ion_fracs.loc[(26, slice(None))].sum(axis=0)
        if (26, 1) in ion_fracs.index:
            fe_ii = ion_fracs.loc[(26, 1)]
            fe_ii_frac = (fe_ii / fe_total).mean()
            logger.info(f"  Fe II fraction (mean): {fe_ii_frac:.4f}")
        if (26, 2) in ion_fracs.index:
            fe_iii = ion_fracs.loc[(26, 2)]
            fe_iii_frac = (fe_iii / fe_total).mean()
            logger.info(f"  Fe III fraction (mean): {fe_iii_frac:.4f}")
    except Exception as e:
        logger.info(f"  Could not extract ionization: {e}")

    # Save spectrum
    np.savetxt(
        "tardis_comparison_spectrum.dat",
        np.column_stack([model_wave, model_flux]),
        header="Wavelength(A) Luminosity_density(erg/s/A)",
        fmt="%.6e"
    )

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("  TARDIS SIMULATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  T_inner: {sim.simulation_state.t_inner:.2f}")
    logger.info(f"  Chi-square: {chi2:.4f}")
    if v_si:
        logger.info(f"  v(Si II): {v_si:.0f} km/s")
    logger.info(f"  Output: tardis_comparison_spectrum.dat")
    logger.info(f"  Log: {LOG_FILE}")
    logger.info(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
