#!/usr/bin/env python3
"""
Task Order #32 - Phase 1 Verification: Abundance vs. Velocity Plot

Generates the Si abundance tapering profile to verify the linear decrease
from X_Si = 35% at v < 11,000 km/s to X_Si = 2% at v = 25,000 km/s.
"""

import numpy as np
import matplotlib.pyplot as plt

# Velocity grid (km/s)
v = np.linspace(10000, 25000, 100)

# Phase 1 Si-Tapering Parameters
v_taper_start = 11000.0  # km/s
v_outer_max = 25000.0    # km/s
Si_inner = 0.35          # 35% in photosphere
Si_outer = 0.02          # 2% at outer boundary

# Calculate Si abundance profile
X_Si = np.where(
    v < v_taper_start,
    Si_inner,
    Si_inner - (Si_inner - Si_outer) * (v - v_taper_start) / (v_outer_max - v_taper_start)
)
X_Si = np.clip(X_Si, Si_outer, Si_inner)

# Also calculate Fe profile for reference
Fe_inner = 0.30
Fe_outer = 0.04
X_Fe = np.where(
    v < v_taper_start,
    Fe_inner,
    Fe_inner - (Fe_inner - Fe_outer) * (v - v_taper_start) / (v_outer_max - v_taper_start)
)
X_Fe = np.clip(X_Fe, Fe_outer, Fe_inner)

# C/O profile (increases outward)
C_inner = 0.05
C_outer = 0.20
X_C = np.where(
    v < v_taper_start,
    C_inner,
    C_inner + (C_outer - C_inner) * (v - v_taper_start) / (v_outer_max - v_taper_start)
)

O_inner = 0.10
O_outer = 0.30
X_O = np.where(
    v < v_taper_start,
    O_inner,
    O_inner + (O_outer - O_inner) * (v - v_taper_start) / (v_outer_max - v_taper_start)
)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot abundance profiles
ax.plot(v, X_Si * 100, 'b-', lw=2.5, label='Silicon (Si)')
ax.plot(v, X_Fe * 100, 'r-', lw=2, label='Iron (Fe)')
ax.plot(v, X_C * 100, 'g--', lw=1.5, label='Carbon (C)')
ax.plot(v, X_O * 100, 'm--', lw=1.5, label='Oxygen (O)')

# Mark key velocities
ax.axvline(v_taper_start, color='gray', ls=':', lw=1.5, label=f'Taper start ({v_taper_start:.0f} km/s)')
ax.axvline(10000, color='orange', ls='--', lw=1.5, alpha=0.7, label='v_inner (10,000 km/s)')

# Shade regions
ax.axvspan(10000, v_taper_start, alpha=0.15, color='blue', label='Photospheric zone')
ax.axvspan(v_taper_start, v_outer_max, alpha=0.1, color='gray')

# Annotations
ax.annotate('Si II 6355\nforms here', xy=(10500, 32), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.annotate('Si "curtain"\nremoved', xy=(18000, 15), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Velocity (km/s)', fontsize=12)
ax.set_ylabel('Mass Fraction (%)', fontsize=12)
ax.set_title('Task Order #32 Phase 1: Spatial Abundance Tapering\n'
             'Linear Si decrease to prevent high-velocity "Photospheric Wall"', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(10000, 25000)
ax.set_ylim(0, 40)
ax.grid(True, alpha=0.3)

# Add text box with parameters
textstr = '\n'.join([
    'Phase 1 Parameters:',
    f'  v < {v_taper_start:.0f} km/s: X_Si = {Si_inner*100:.0f}%',
    f'  v ≥ {v_taper_start:.0f} km/s: Linear taper',
    f'  v = {v_outer_max:.0f} km/s: X_Si = {Si_outer*100:.0f}%',
])
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('task32_phase1_abundance_profile.png', dpi=150)
plt.savefig('task32_phase1_abundance_profile.pdf')
print("Phase 1 verification plot saved: task32_phase1_abundance_profile.png")
