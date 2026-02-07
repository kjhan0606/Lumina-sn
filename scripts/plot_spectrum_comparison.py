#!/usr/bin/env python3
"""Plot TARDIS vs LUMINA spectrum comparison: real, virtual, and rotation packets."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

REF = "data/tardis_reference"

# Load TARDIS spectra
t_real = np.genfromtxt(f"{REF}/spectrum_real.csv", delimiter=',', names=True)
t_virt = np.genfromtxt(f"{REF}/spectrum_virtual.csv", delimiter=',', names=True)

# Load LUMINA real spectrum
l_real = np.genfromtxt("lumina_spectrum.csv", delimiter=',', names=True)

# Optionally load LUMINA virtual spectrum
has_virtual = os.path.exists("lumina_spectrum_virtual.csv")
if has_virtual:
    l_virt = np.genfromtxt("lumina_spectrum_virtual.csv", delimiter=',', names=True)

# Optionally load rotation spectrum
has_rotation = os.path.exists("lumina_spectrum_rotation.csv")
if has_rotation:
    l_rot = np.genfromtxt("lumina_spectrum_rotation.csv", delimiter=',', names=True)

# Reverse TARDIS (descending → ascending)
tw = t_real['wavelength_angstrom'][::-1]
tf = t_real['flux'][::-1]
tvw = t_virt['wavelength_angstrom'][::-1]
tvf = t_virt['flux'][::-1]
lrw = l_real['wavelength_angstrom']
lrf = l_real['flux']

# Common grid for interpolation
grid = np.arange(3000, 10001, 5.0)
tf_i = np.interp(grid, tw, tf)
tvf_i = np.interp(grid, tvw, tvf)
lrf_i = np.interp(grid, lrw, lrf)

if has_virtual:
    lvw = l_virt['wavelength_angstrom']
    lvf = l_virt['flux']
    lvf_i = np.interp(grid, lvw, lvf)

if has_rotation:
    lrotw = l_rot['wavelength_angstrom']
    lrotf = l_rot['flux']
    lrotf_i = np.interp(grid, lrotw, lrotf)

# Normalize to peak in 4000-7000 A
opt = (grid >= 4000) & (grid <= 7000)
tf_n = tf_i / tf_i[opt].max()
tvf_n = tvf_i / tvf_i[opt].max()
lrf_n = lrf_i / lrf_i[opt].max()
if has_virtual:
    lvf_n = lvf_i / lvf_i[opt].max()
if has_rotation:
    lrotf_n = lrotf_i / lrotf_i[opt].max()

# ===== FIGURE 1: 4-panel comparison =====
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
modes = []
if has_virtual: modes.append('virtual')
if has_rotation: modes.append('rotation')
mode_str = ' + '.join(['real'] + modes) if modes else 'real'
fig.suptitle(f'LUMINA vs TARDIS: {mode_str} Packet Spectra',
             fontsize=14, fontweight='bold')

# Panel 1: TARDIS real vs LUMINA real (normalized)
ax = axes[0, 0]
ax.plot(grid, tf_n, 'b-', alpha=0.7, linewidth=0.8, label='TARDIS real')
ax.plot(grid, lrf_n, 'r-', alpha=0.7, linewidth=0.8, label='LUMINA real')
if has_rotation:
    ax.plot(grid, lrotf_n, 'g-', alpha=0.7, linewidth=0.8, label='LUMINA rotation')
ax.set_xlim(3000, 10000)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Normalized Flux')
ax.set_title('Real Packets (normalized)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: TARDIS virtual vs LUMINA virtual (normalized)
ax = axes[0, 1]
ax.plot(grid, tvf_n, 'b-', alpha=0.7, linewidth=0.8, label='TARDIS virtual')
if has_virtual:
    ax.plot(grid, lvf_n, 'r-', alpha=0.7, linewidth=0.8, label='LUMINA virtual')
if has_rotation:
    ax.plot(grid, lrotf_n, 'g-', alpha=0.7, linewidth=0.8, label='LUMINA rotation')
ax.set_xlim(3000, 10000)
ax.set_ylim(0, 1.15)
ax.set_title('Virtual / Rotation Packets (normalized)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: All spectra overlaid
ax = axes[1, 0]
ax.plot(grid, tvf_n, 'b-', alpha=0.6, linewidth=0.6, label='TARDIS virt')
ax.plot(grid, tf_n, 'b--', alpha=0.4, linewidth=0.5, label='TARDIS real')
if has_virtual:
    ax.plot(grid, lvf_n, 'r-', alpha=0.6, linewidth=0.6, label='LUMINA virt')
ax.plot(grid, lrf_n, 'r--', alpha=0.4, linewidth=0.5, label='LUMINA real')
if has_rotation:
    ax.plot(grid, lrotf_n, 'g-', alpha=0.7, linewidth=0.8, label='LUMINA rotation')
ax.set_xlim(3000, 10000)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Normalized Flux')
ax.set_title('All Spectra Overlaid')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 4: Residuals (smoothed)
ax = axes[1, 1]
res_real = uniform_filter1d(lrf_n - tf_n, 10)
ax.plot(grid, res_real, 'r-', alpha=0.6, linewidth=0.8, label='Real: LUMINA−TARDIS')
if has_virtual:
    res_virt = uniform_filter1d(lvf_n - tvf_n, 10)
    ax.plot(grid, res_virt, 'b-', alpha=0.6, linewidth=0.8, label='Virtual: LUMINA−TARDIS')
if has_rotation:
    res_rot = uniform_filter1d(lrotf_n - tvf_n, 10)
    ax.plot(grid, res_rot, 'g-', alpha=0.6, linewidth=0.8, label='Rotation: LUMINA−TARDIS(virt)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlim(3000, 10000)
ax.set_ylim(-0.15, 0.15)
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Residual')
ax.set_title('Residuals (smoothed)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Stats
mask_opt = (grid >= 3500) & (grid <= 9000) & (tf_n > 0.05)
rms_r = np.sqrt(np.mean((lrf_n[mask_opt] - tf_n[mask_opt])**2))
stats = f"Real: RMS={rms_r:.3f}"
if has_virtual:
    rms_v = np.sqrt(np.mean((lvf_n[mask_opt] - tvf_n[mask_opt])**2))
    stats += f"\nVirtual: RMS={rms_v:.3f}"
if has_rotation:
    rms_rot = np.sqrt(np.mean((lrotf_n[mask_opt] - tvf_n[mask_opt])**2))
    stats += f"\nRotation: RMS={rms_rot:.3f}"
axes[0, 0].text(0.02, 0.95, stats, transform=axes[0, 0].transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('spectrum_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: spectrum_comparison.png")

# ===== FIGURE 2: Real vs Virtual/Rotation noise comparison =====
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig2.suptitle('Spectrum Noise Comparison: Real vs Virtual/Rotation',
              fontsize=14, fontweight='bold')

# LUMINA real vs virtual/rotation
ax1.plot(grid, lrf_n, 'r-', alpha=0.4, linewidth=0.5, label='LUMINA real')
if has_virtual:
    ax1.plot(grid, lvf_n, 'darkred', alpha=0.8, linewidth=0.8, label='LUMINA virtual')
if has_rotation:
    ax1.plot(grid, lrotf_n, 'green', alpha=0.8, linewidth=0.8, label='LUMINA rotation')
ax1.set_xlim(3000, 10000)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel('Normalized Flux')
ax1.set_title('LUMINA: Real vs Virtual/Rotation Packets')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# TARDIS real vs virtual
ax2.plot(grid, tf_n, 'b-', alpha=0.4, linewidth=0.5, label='TARDIS real')
ax2.plot(grid, tvf_n, 'darkblue', alpha=0.8, linewidth=0.8, label='TARDIS virtual')
ax2.set_xlim(3000, 10000)
ax2.set_ylim(0, 1.15)
ax2.set_xlabel('Wavelength (Å)')
ax2.set_ylabel('Normalized Flux')
ax2.set_title('TARDIS: Real vs Virtual Packets')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrum_real_vs_virtual.png', dpi=150, bbox_inches='tight')
print("Saved: spectrum_real_vs_virtual.png")

# ===== FIGURE 3: Si II detail =====
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Si II 6355 Å Region: TARDIS vs LUMINA',
              fontsize=14, fontweight='bold')

mask_si = (grid >= 5500) & (grid <= 7000)

# Panel 1: Real packets
ax = axes3[0]
ax.plot(grid[mask_si], tf_n[mask_si], 'b-', linewidth=1.0, label='TARDIS real', alpha=0.8)
ax.plot(grid[mask_si], lrf_n[mask_si], 'r-', linewidth=1.0, label='LUMINA real', alpha=0.8)
if has_rotation:
    ax.plot(grid[mask_si], lrotf_n[mask_si], 'g-', linewidth=1.0, label='LUMINA rotation', alpha=0.8)
ax.axvline(6355, color='green', linestyle='--', alpha=0.4, label='Si II rest')
for v in [10000, 15000, 20000]:
    w = 6355 * (1 - v / 3e5)
    ax.axvline(w, color='orange', linestyle=':', alpha=0.3)
    ax.text(w, 0.18, f'{v//1000}k', fontsize=7, color='orange', ha='center')
ax.set_xlim(5500, 7000)
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Real + Rotation Packets')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Virtual packets
ax = axes3[1]
ax.plot(grid[mask_si], tvf_n[mask_si], 'b-', linewidth=1.0, label='TARDIS virtual', alpha=0.8)
if has_virtual:
    ax.plot(grid[mask_si], lvf_n[mask_si], 'r-', linewidth=1.0, label='LUMINA virtual', alpha=0.8)
if has_rotation:
    ax.plot(grid[mask_si], lrotf_n[mask_si], 'g-', linewidth=1.0, label='LUMINA rotation', alpha=0.8)
ax.axvline(6355, color='green', linestyle='--', alpha=0.4, label='Si II rest')
for v in [10000, 15000, 20000]:
    w = 6355 * (1 - v / 3e5)
    ax.axvline(w, color='orange', linestyle=':', alpha=0.3)
    ax.text(w, 0.18, f'{v//1000}k', fontsize=7, color='orange', ha='center')
ax.set_xlim(5500, 7000)
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Virtual + Rotation Packets')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrum_comparison_siII.png', dpi=150, bbox_inches='tight')
print("Saved: spectrum_comparison_siII.png")
