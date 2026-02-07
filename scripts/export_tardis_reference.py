#!/usr/bin/env python3
"""Phase 1: Run TARDIS with SN2011fe config and export full converged state.

Exports: W[], T_rad[], tau_sobolev[line,shell], j_blue[line,shell],
         spectrum, line_list_nu, electron_densities, geometry, etc.
All saved to data/tardis_reference/ directory as CSV and HDF5.
"""
import os
import numpy as np
import pandas as pd

# Set up TARDIS
from tardis.io.configuration.config_reader import Configuration
from tardis.simulation import Simulation

# Paths
CONFIG_FILE = "data/sn2011fe/sn2011fe.yml"
OUTPUT_DIR = "data/tardis_reference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 1: Running TARDIS reference simulation")
print("=" * 60)

# Load config and run simulation
config = Configuration.from_yaml(CONFIG_FILE)
sim = Simulation.from_config(config)
sim.run_convergence()
sim.run_final()

print("\nSimulation complete. Exporting state...")

# ===== GEOMETRY =====
model = sim.simulation_state
n_shells = model.no_of_shells
print(f"Number of shells: {n_shells}")

try:
    r_inner = model.r_inner.cgs.value  # cm
except AttributeError:
    r_inner = np.array(model.r_inner)
try:
    r_outer = model.r_outer.cgs.value  # cm
except AttributeError:
    r_outer = np.array(model.r_outer)
try:
    v_inner = model.v_inner.cgs.value  # cm/s
except AttributeError:
    v_inner = np.array(model.v_inner)
try:
    v_outer = model.v_outer.cgs.value  # cm/s
except AttributeError:
    v_outer = np.array(model.v_outer)
try:
    time_explosion = model.time_explosion.cgs.value  # seconds
except AttributeError:
    time_explosion = float(model.time_explosion)

geo_df = pd.DataFrame({
    'shell_id': range(n_shells),
    'r_inner': r_inner,
    'r_outer': r_outer,
    'v_inner': v_inner,
    'v_outer': v_outer,
})
geo_df.to_csv(os.path.join(OUTPUT_DIR, "geometry.csv"), index=False)
print(f"  Geometry: {n_shells} shells, t_exp = {time_explosion:.6e} s")

# ===== PLASMA STATE =====
plasma = sim.plasma

# Dilution factor W and radiation temperature T_rad
W = plasma.w  # array [n_shells]
T_rad = np.array(plasma.t_rad)  # K, array [n_shells]
try:
    T_inner = model.t_inner.cgs.value  # K, scalar
except AttributeError:
    T_inner = float(model.t_inner)  # K, scalar

plasma_df = pd.DataFrame({
    'shell_id': range(n_shells),
    'W': W,
    'T_rad': T_rad,
})
plasma_df.to_csv(os.path.join(OUTPUT_DIR, "plasma_state.csv"), index=False)
print(f"  T_inner = {T_inner:.2f} K")
print(f"  W range: [{W.min():.6f}, {W.max():.6f}]")
print(f"  T_rad range: [{T_rad.min():.2f}, {T_rad.max():.2f}] K")

# ===== ELECTRON DENSITIES =====
try:
    n_e = plasma.electron_densities.values  # cm^-3, array [n_shells]
except AttributeError:
    n_e = np.array(plasma.electron_densities)
ne_df = pd.DataFrame({
    'shell_id': range(n_shells),
    'n_e': n_e,
})
ne_df.to_csv(os.path.join(OUTPUT_DIR, "electron_densities.csv"), index=False)
print(f"  n_e range: [{n_e.min():.6e}, {n_e.max():.6e}] cm^-3")

# ===== DENSITY =====
try:
    rho = model.density.cgs.value  # g/cm^3, array [n_shells]
except AttributeError:
    rho = np.array(model.density)
rho_df = pd.DataFrame({
    'shell_id': range(n_shells),
    'rho': rho,
})
rho_df.to_csv(os.path.join(OUTPUT_DIR, "density.csv"), index=False)

# ===== LINE LIST (sorted descending by frequency, as TARDIS uses) =====
try:
    line_list_nu = plasma.atomic_data.lines.nu.values  # Hz, sorted descending
except AttributeError:
    line_list_nu = np.array(plasma.atomic_data.lines.nu)
n_lines = len(line_list_nu)
print(f"  Lines: {n_lines} total")

# Save line info
lines = plasma.atomic_data.lines
print(f"  Line columns: {list(lines.columns)}")
print(f"  Line index names: {lines.index.names}")
# Save all line data as-is
lines_export = lines.copy()
lines_export['nu'] = line_list_nu
lines_export.to_csv(os.path.join(OUTPUT_DIR, "line_list.csv"))

# ===== TAU SOBOLEV [n_lines, n_shells] =====
try:
    tau_sobolev = plasma.tau_sobolevs.values  # [n_lines, n_shells]
except AttributeError:
    tau_sobolev = np.array(plasma.tau_sobolevs)
print(f"  tau_sobolev shape: {tau_sobolev.shape}")
np.save(os.path.join(OUTPUT_DIR, "tau_sobolev.npy"), tau_sobolev)

# Save summary stats
tau_stats = pd.DataFrame({
    'shell_id': range(n_shells),
    'tau_min': tau_sobolev.min(axis=0),
    'tau_max': tau_sobolev.max(axis=0),
    'tau_mean': tau_sobolev.mean(axis=0),
    'n_active': (tau_sobolev > 1e-10).sum(axis=0),
})
tau_stats.to_csv(os.path.join(OUTPUT_DIR, "tau_sobolev_stats.csv"), index=False)

# ===== J_BLUE [n_lines, n_shells] =====
try:
    j_blues_raw = plasma.j_blues
    j_blues = j_blues_raw.values if hasattr(j_blues_raw, 'values') else np.array(j_blues_raw)
    print(f"  j_blues shape: {j_blues.shape}")
    np.save(os.path.join(OUTPUT_DIR, "j_blues.npy"), j_blues)
except Exception as e:
    print(f"  j_blues not available: {e}")

# ===== TRANSITION PROBABILITIES (macro-atom) =====
try:
    tp_raw = plasma.transition_probabilities
    transition_probabilities = tp_raw.values if hasattr(tp_raw, 'values') else np.array(tp_raw)
    print(f"  transition_probabilities shape: {transition_probabilities.shape}")
    np.save(os.path.join(OUTPUT_DIR, "transition_probabilities.npy"),
            transition_probabilities)
except Exception as e:
    print(f"  transition_probabilities: {e}")

# ===== MACRO ATOM DATA =====
macro_atom_data = plasma.atomic_data.macro_atom_data
if macro_atom_data is not None:
    ma_df = macro_atom_data.copy()
    ma_df.to_csv(os.path.join(OUTPUT_DIR, "macro_atom_data.csv"))
    print(f"  macro_atom_data: {len(ma_df)} transitions")

    # Block references
    macro_atom_references = plasma.atomic_data.macro_atom_references
    ma_ref_df = macro_atom_references.copy()
    ma_ref_df.to_csv(os.path.join(OUTPUT_DIR, "macro_atom_references.csv"))
    print(f"  macro_atom_references: {len(ma_ref_df)} levels")

# ===== LINE2MACRO_LEVEL_UPPER =====
try:
    line2macro = plasma.atomic_data.lines_upper2macro_reference_idx
    np.save(os.path.join(OUTPUT_DIR, "line2macro_level_upper.npy"),
            np.array(line2macro))
    print(f"  line2macro_level_upper: {len(line2macro)} entries")
except Exception as e:
    print(f"  line2macro_level_upper: {e}")

# ===== LINE INTERACTION TYPE ID =====
try:
    if 'line_id' in lines.columns:
        line_interaction_id = np.array(lines.line_id)
    elif hasattr(lines.index, 'get_level_values'):
        line_interaction_id = np.arange(n_lines)
    else:
        line_interaction_id = np.arange(n_lines)
    np.save(os.path.join(OUTPUT_DIR, "line_interaction_id.npy"), line_interaction_id)
    print(f"  line_interaction_id: {len(line_interaction_id)} entries")
except Exception as e:
    print(f"  line_interaction_id: {e}")

# ===== SPECTRUM =====
try:
    spectrum = sim.spectrum_solver.spectrum_real_packets
except AttributeError:
    spectrum = sim.runner.spectrum
try:
    wave = spectrum.wavelength.value  # Angstrom
except AttributeError:
    wave = np.array(spectrum.wavelength)
try:
    flux = spectrum.luminosity_density_lambda.cgs.value
except AttributeError:
    flux = np.array(spectrum.luminosity_density_lambda)
spec_df = pd.DataFrame({
    'wavelength_angstrom': wave,
    'flux': flux,
})
spec_df.to_csv(os.path.join(OUTPUT_DIR, "spectrum_real.csv"), index=False)
print(f"  Real spectrum: {len(wave)} bins")

# Virtual packets spectrum
try:
    try:
        vspec = sim.spectrum_solver.spectrum_virtual_packets
    except AttributeError:
        vspec = sim.runner.spectrum_virtual
    vwave = vspec.wavelength.value if hasattr(vspec.wavelength, 'value') else np.array(vspec.wavelength)
    vflux = vspec.luminosity_density_lambda.cgs.value if hasattr(vspec.luminosity_density_lambda, 'cgs') else np.array(vspec.luminosity_density_lambda)
    vspec_df = pd.DataFrame({
        'wavelength_angstrom': vwave,
        'flux': vflux,
    })
    vspec_df.to_csv(os.path.join(OUTPUT_DIR, "spectrum_virtual.csv"), index=False)
    print(f"  Virtual spectrum: {len(vwave)} bins")
except Exception as e:
    print(f"  Virtual spectrum: {e}")

# ===== MC ESTIMATORS =====
try:
    try:
        estimators = sim.transport.transport_state.radfield_mc_estimators
    except AttributeError:
        estimators = sim.runner.estimators if hasattr(sim.runner, 'estimators') else None
    j_est = np.array(estimators.j_estimator)
    nu_bar_est = np.array(estimators.nu_bar_estimator)
    est_df = pd.DataFrame({
        'shell_id': range(n_shells),
        'j_estimator': j_est,
        'nu_bar_estimator': nu_bar_est,
    })
    est_df.to_csv(os.path.join(OUTPUT_DIR, "mc_estimators.csv"), index=False)
    print(f"  MC estimators saved")
except Exception as e:
    print(f"  MC estimators: {e}")

# ===== ABUNDANCES =====
try:
    abd_raw = model.abundance
    if hasattr(abd_raw, 'values'):
        abundances = abd_raw.values  # [n_elements, n_shells]
    else:
        abundances = np.array(abd_raw)
    np.save(os.path.join(OUTPUT_DIR, "abundances.npy"), abundances)
    if hasattr(abd_raw, 'to_csv'):
        abd_raw.to_csv(os.path.join(OUTPUT_DIR, "abundances.csv"))
    else:
        abd_df = pd.DataFrame(abundances.T,
                               columns=[f'Z{i+1}' for i in range(abundances.shape[0])])
        abd_df.insert(0, 'shell_id', range(n_shells))
        abd_df.to_csv(os.path.join(OUTPUT_DIR, "abundances.csv"), index=False)
    print(f"  Abundances: {abundances.shape}")
except Exception as e:
    print(f"  Abundances: {e}")

# ===== IONIZATION FRACTIONS =====
try:
    ind_raw = plasma.ion_number_density
    ion_number_density = ind_raw.values if hasattr(ind_raw, 'values') else np.array(ind_raw)
    np.save(os.path.join(OUTPUT_DIR, "ion_number_density.npy"), ion_number_density)
    print(f"  ion_number_density shape: {ion_number_density.shape}")
except Exception as e:
    print(f"  ion_number_density: {e}")

# ===== TRANSPORT CONFIG =====
config_data = {
    'time_explosion_s': time_explosion,
    'T_inner_K': T_inner,
    'luminosity_inner_erg_s': float(getattr(model, 'luminosity_inner', getattr(model, '_luminosity_inner', 9.44e42))),
    'n_shells': n_shells,
    'n_lines': n_lines,
    'n_packets': config.montecarlo.no_of_packets,
    'n_iterations': config.montecarlo.iterations,
    'seed': config.montecarlo.seed,
    'v_inner_min_cm_s': v_inner[0],
    'v_outer_max_cm_s': v_outer[-1],
}
import json
with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
    json.dump(config_data, f, indent=2)

# ===== ESCAPE/REABSORB COUNTS =====
try:
    try:
        ts = sim.transport.transport_state
        output_nus = np.array(ts.output_nus)
        output_energies = np.array(ts.output_energies)
    except AttributeError:
        ts = sim.runner
        output_nus = np.array(ts.output_nu)
        output_energies = np.array(ts.output_energy)
    n_emitted = (output_nus > 0).sum()
    n_reabsorbed = (output_nus <= 0).sum()
    n_total = len(output_nus)
    escape_frac = n_emitted / n_total
    print(f"  Packets: {n_total} total, {n_emitted} escaped ({escape_frac:.4f})")

    # Save escaped packet data
    esc_mask = output_nus > 0
    escaped_df = pd.DataFrame({
        'nu': output_nus[esc_mask],
        'energy': output_energies[esc_mask],
    })
    escaped_df.to_csv(os.path.join(OUTPUT_DIR, "escaped_packets.csv"), index=False)
except Exception as e:
    print(f"  Escape data: {e}")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE. All data exported to data/tardis_reference/")
print("=" * 60)
