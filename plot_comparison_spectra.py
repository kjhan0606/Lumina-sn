#!/usr/bin/env python3
"""
Plot comparison spectra: LUMINA vs TARDIS vs Observation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# Paths
BASE_DIR = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina")
SPECTRA_DIR = BASE_DIR / "data" / "sn2011fe" / "spectra"
LUMINA_DIR = Path(".")

# Load observed spectrum
obs_file = SPECTRA_DIR / "sn2011fe_p0d0d_ptf11kly_20110910.obs.dat"
obs_data = np.loadtxt(obs_file)
obs_wave = obs_data[:, 0]
obs_flux = obs_data[:, 1]
obs_flux = obs_flux / np.max(obs_flux[(obs_wave > 4000) & (obs_wave < 7000)])

# Load TARDIS spectrum
tardis_file = LUMINA_DIR / "tardis_comparison_spectrum.dat"
if tardis_file.exists():
    tardis_data = np.loadtxt(tardis_file)
    tardis_wave = tardis_data[:, 0]
    tardis_flux = tardis_data[:, 1]
    tardis_flux = tardis_flux / np.max(tardis_flux[(tardis_wave > 4000) & (tardis_wave < 7000)])
else:
    print("TARDIS spectrum not found!")
    tardis_wave = None
    tardis_flux = None

# Load LUMINA spectrum (with T-iteration)
lumina_file = LUMINA_DIR / "lumina_comparison_spectrum.dat"
if lumina_file.exists():
    lumina_data = np.loadtxt(lumina_file)
    lumina_wave = lumina_data[:, 0]
    lumina_flux = lumina_data[:, 1]
    # Normalize
    mask = (lumina_wave > 4000) & (lumina_wave < 7000)
    if np.any(mask) and np.max(lumina_flux[mask]) > 0:
        lumina_flux = lumina_flux / np.max(lumina_flux[mask])
else:
    print("LUMINA spectrum not found!")
    lumina_wave = None
    lumina_flux = None

# Load LUMINA fixed spectrum (no T-iteration)
lumina_fixed_file = LUMINA_DIR / "lumina_fixed_spectrum.csv"
if lumina_fixed_file.exists():
    # LUMINA CSV format: wavelength_A,frequency_Hz,L_nu_standard,L_nu_lumina,...
    try:
        lumina_fixed_data = np.loadtxt(lumina_fixed_file, delimiter=',', skiprows=4)
        lumina_fixed_wave = lumina_fixed_data[:, 0]
        lumina_fixed_flux = lumina_fixed_data[:, 3]  # L_nu_lumina column
        # Sort by wavelength
        sort_idx = np.argsort(lumina_fixed_wave)
        lumina_fixed_wave = lumina_fixed_wave[sort_idx]
        lumina_fixed_flux = lumina_fixed_flux[sort_idx]
        # Normalize
        mask = (lumina_fixed_wave > 4000) & (lumina_fixed_wave < 7000)
        if np.any(mask) and np.max(lumina_fixed_flux[mask]) > 0:
            lumina_fixed_flux = lumina_fixed_flux / np.max(lumina_fixed_flux[mask])
    except:
        lumina_fixed_wave = None
        lumina_fixed_flux = None
else:
    lumina_fixed_wave = None
    lumina_fixed_flux = None

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

# Main spectrum comparison
ax1 = axes[0]

# Plot observed
ax1.plot(obs_wave, obs_flux, 'k-', linewidth=1.2, label='SN 2011fe (Observed)', alpha=0.9)

# Plot TARDIS
if tardis_wave is not None:
    ax1.plot(tardis_wave, tardis_flux, 'b-', linewidth=1.0, label='TARDIS', alpha=0.8)

# Plot LUMINA (with iteration)
if lumina_wave is not None:
    ax1.plot(lumina_wave, lumina_flux, 'r-', linewidth=1.0, label='LUMINA (T-iter)', alpha=0.7)

# Plot LUMINA fixed (no iteration)
if lumina_fixed_wave is not None:
    ax1.plot(lumina_fixed_wave, lumina_fixed_flux, 'g-', linewidth=1.0, label='LUMINA (fixed T)', alpha=0.8)

# Mark key features
features = {
    'Si II 6355': 6139,  # Blueshifted
    'S II W': 5300,
    'Ca II H&K': 3750,
    'Fe II blend': 4800,
}
ymax = ax1.get_ylim()[1]
for name, wave in features.items():
    ax1.axvline(wave, color='gray', linestyle=':', alpha=0.5)
    ax1.text(wave, ymax * 0.95, name, rotation=90, va='top', ha='right', fontsize=9, color='gray')

ax1.set_xlabel('Wavelength (Å)', fontsize=12)
ax1.set_ylabel('Normalized Flux', fontsize=12)
ax1.set_title('SN 2011fe at B-maximum: LUMINA vs TARDIS vs Observation', fontsize=14)
ax1.set_xlim(3000, 9000)
ax1.set_ylim(0, 1.3)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Residuals panel
ax2 = axes[1]

# Common wavelength grid for residuals
wave_common = np.linspace(3500, 8500, 1000)

# Interpolate observed
obs_interp = interp1d(obs_wave, obs_flux, bounds_error=False, fill_value=0)(wave_common)

if tardis_wave is not None:
    tardis_interp = interp1d(tardis_wave, tardis_flux, bounds_error=False, fill_value=0)(wave_common)
    residual_tardis = obs_interp - tardis_interp
    ax2.plot(wave_common, residual_tardis, 'b-', linewidth=0.8, label='Obs - TARDIS', alpha=0.8)

if lumina_wave is not None:
    lumina_interp = interp1d(lumina_wave, lumina_flux, bounds_error=False, fill_value=0)(wave_common)
    residual_lumina = obs_interp - lumina_interp
    ax2.plot(wave_common, residual_lumina, 'r-', linewidth=0.8, label='Obs - LUMINA', alpha=0.8)

ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax2.fill_between(wave_common, -0.1, 0.1, alpha=0.2, color='gray')
ax2.set_xlabel('Wavelength (Å)', fontsize=12)
ax2.set_ylabel('Residual', fontsize=12)
ax2.set_xlim(3000, 9000)
ax2.set_ylim(-0.5, 0.5)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrum_comparison_tardis_lumina_obs.pdf', dpi=150, bbox_inches='tight')
plt.savefig('spectrum_comparison_tardis_lumina_obs.png', dpi=150, bbox_inches='tight')
print("Saved: spectrum_comparison_tardis_lumina_obs.pdf/png")

# Calculate chi-square values
def calc_chi2(model_wave, model_flux, obs_wave, obs_flux, wave_min=3500, wave_max=7500):
    model_mask = (model_wave >= wave_min) & (model_wave <= wave_max)
    obs_mask = (obs_wave >= wave_min) & (obs_wave <= wave_max)

    model_w = model_wave[model_mask]
    model_f = model_flux[model_mask]
    obs_w = obs_wave[obs_mask]
    obs_f = obs_flux[obs_mask]

    interp_func = interp1d(model_w, model_f, kind='linear', bounds_error=False, fill_value=0)
    model_interp = interp_func(obs_w)

    # Renormalize
    model_interp = model_interp / np.max(model_interp)
    obs_f = obs_f / np.max(obs_f)

    chi2 = np.sum((model_interp - obs_f)**2)
    return chi2

print("\n" + "="*60)
print("  SPECTRAL COMPARISON SUMMARY")
print("="*60)

if tardis_wave is not None:
    chi2_tardis = calc_chi2(tardis_wave, tardis_flux, obs_wave, obs_flux)
    print(f"  TARDIS chi-square (3500-7500 Å): {chi2_tardis:.2f}")

if lumina_wave is not None:
    chi2_lumina = calc_chi2(lumina_wave, lumina_flux, obs_wave, obs_flux)
    print(f"  LUMINA (T-iter) chi-square (3500-7500 Å): {chi2_lumina:.2f}")

if lumina_fixed_wave is not None:
    chi2_lumina_fixed = calc_chi2(lumina_fixed_wave, lumina_fixed_flux, obs_wave, obs_flux)
    print(f"  LUMINA (fixed T) chi-square (3500-7500 Å): {chi2_lumina_fixed:.2f}")

print("="*60)
