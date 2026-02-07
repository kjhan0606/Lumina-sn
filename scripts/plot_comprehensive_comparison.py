#!/usr/bin/env python3
"""Comprehensive TARDIS vs LUMINA comparison: spectra + plasma state."""
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

# Load LUMINA spectra
l_real = np.genfromtxt("lumina_spectrum.csv", delimiter=',', names=True)
has_virtual = os.path.exists("lumina_spectrum_virtual.csv")
if has_virtual:
    l_virt = np.genfromtxt("lumina_spectrum_virtual.csv", delimiter=',', names=True)

# Load plasma states
import json
with open(f"{REF}/config.json") as f:
    cfg = json.load(f)
t_plasma = np.genfromtxt(f"{REF}/plasma_state.csv", delimiter=',', names=True)
l_plasma = np.genfromtxt("lumina_plasma_state.csv", delimiter=',', names=True)

# Reverse TARDIS (descending -> ascending)
tw = t_real['wavelength_angstrom'][::-1]
tf = t_real['flux'][::-1]
tvw = t_virt['wavelength_angstrom'][::-1]
tvf = t_virt['flux'][::-1]
lrw = l_real['wavelength_angstrom']
lrf = l_real['flux']

# Interpolate onto common grid
grid = np.arange(3000, 10001, 5.0)
tf_i = np.interp(grid, tw, tf)
tvf_i = np.interp(grid, tvw, tvf)
lrf_i = np.interp(grid, lrw, lrf)
if has_virtual:
    lvw = l_virt['wavelength_angstrom']
    lvf = l_virt['flux']
    lvf_i = np.interp(grid, lvw, lvf)

# Normalize to peak in 4000-7000
opt = (grid >= 4000) & (grid <= 7000)
tf_n = tf_i / tf_i[opt].max()
tvf_n = tvf_i / tvf_i[opt].max()
lrf_n = lrf_i / lrf_i[opt].max()
if has_virtual:
    lvf_n = lvf_i / lvf_i[opt].max()

# Compute RMS
mask_opt = (grid >= 3500) & (grid <= 9000) & (tf_n > 0.05)
rms_real = np.sqrt(np.mean((lrf_n[mask_opt] - tf_n[mask_opt])**2))
if has_virtual:
    rms_virt = np.sqrt(np.mean((lvf_n[mask_opt] - tvf_n[mask_opt])**2))
    rms_tardis_vr = np.sqrt(np.mean((tvf_n[mask_opt] - tf_n[mask_opt])**2))

# ==================== FIGURE: 6-panel comprehensive comparison ====================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

fig.suptitle('LUMINA-SN vs TARDIS: Comprehensive Comparison',
             fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Real packet spectra
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(grid, tf_n, 'b-', alpha=0.7, linewidth=0.8, label='TARDIS real')
ax1.plot(grid, lrf_n, 'r-', alpha=0.7, linewidth=0.8, label='LUMINA real')
ax1.set_xlim(3000, 10000)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel('Normalized Flux')
ax1.set_title(f'Real Packet Spectra (RMS = {rms_real:.4f})')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Virtual packet spectra
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(grid, tvf_n, 'b-', alpha=0.7, linewidth=0.8, label='TARDIS virtual')
if has_virtual:
    ax2.plot(grid, lvf_n, 'r-', alpha=0.7, linewidth=0.8, label='LUMINA virtual')
    ax2.set_title(f'Virtual Packet Spectra (RMS = {rms_virt:.4f})')
else:
    ax2.set_title('Virtual Packet Spectra (LUMINA not available)')
ax2.set_xlim(3000, 10000)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel('Normalized Flux')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Si II 6355 region detail
ax3 = fig.add_subplot(gs[1, 0])
mask_si = (grid >= 5500) & (grid <= 7000)
ax3.plot(grid[mask_si], tf_n[mask_si], 'b-', linewidth=1.2, label='TARDIS real', alpha=0.8)
ax3.plot(grid[mask_si], lrf_n[mask_si], 'r-', linewidth=1.2, label='LUMINA real', alpha=0.8)
if has_virtual:
    ax3.plot(grid[mask_si], lvf_n[mask_si], 'r--', linewidth=1.0, label='LUMINA virtual', alpha=0.6)
ax3.axvline(6355, color='green', linestyle='--', alpha=0.4, label='Si II rest')
for v in [10000, 15000, 20000]:
    w = 6355 * (1 - v / 3e5)
    ax3.axvline(w, color='orange', linestyle=':', alpha=0.3)
    ax3.text(w, ax3.get_ylim()[0] + 0.02, f'{v//1000}k', fontsize=7, color='orange', ha='center')
ax3.set_xlim(5500, 7000)
ax3.set_xlabel('Wavelength (A)')
ax3.set_ylabel('Normalized Flux')
ax3.set_title('Si II 6355 A Region')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Residuals
ax4 = fig.add_subplot(gs[1, 1])
res_real = uniform_filter1d(lrf_n - tf_n, 10)
ax4.plot(grid, res_real, 'r-', alpha=0.7, linewidth=0.8, label=f'Real (RMS={rms_real:.4f})')
if has_virtual:
    res_virt = uniform_filter1d(lvf_n - tvf_n, 10)
    ax4.plot(grid, res_virt, 'b-', alpha=0.7, linewidth=0.8, label=f'Virtual (RMS={rms_virt:.4f})')
ax4.axhline(0, color='gray', linewidth=0.5)
ax4.set_xlim(3000, 10000)
ax4.set_ylim(-0.15, 0.15)
ax4.set_xlabel('Wavelength (A)')
ax4.set_ylabel('Residual (LUMINA - TARDIS)')
ax4.set_title('Spectral Residuals (smoothed)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Panel 5: Dilution factor W
ax5 = fig.add_subplot(gs[2, 0])
shells = t_plasma['shell_id'] if 'shell_id' in t_plasma.dtype.names else np.arange(len(t_plasma['W']))
ax5.plot(shells, t_plasma['W'], 'bo-', markersize=4, linewidth=1.0, label='TARDIS')
ax5.plot(l_plasma['shell_id'], l_plasma['W'], 'rs-', markersize=4, linewidth=1.0, label='LUMINA')
ax5.set_xlabel('Shell ID')
ax5.set_ylabel('Dilution Factor W')
ax5.set_title('Dilution Factor W (LUMINA vs TARDIS)')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Inset: W relative error
ax5_inset = ax5.inset_axes([0.55, 0.55, 0.4, 0.4])
w_err = (l_plasma['W'] - t_plasma['W']) / t_plasma['W'] * 100.0
ax5_inset.bar(l_plasma['shell_id'], w_err, color='steelblue', alpha=0.7)
ax5_inset.axhline(0, color='gray', linewidth=0.5)
ax5_inset.set_ylabel('W error (%)', fontsize=7)
ax5_inset.set_xlabel('Shell', fontsize=7)
ax5_inset.tick_params(labelsize=6)
ax5_inset.set_ylim(-3, 3)
ax5_inset.set_title(f'Mean |err| = {np.mean(np.abs(w_err)):.2f}%', fontsize=7)

# Panel 6: Radiation temperature T_rad
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(shells, t_plasma['T_rad'], 'bo-', markersize=4, linewidth=1.0, label='TARDIS')
ax6.plot(l_plasma['shell_id'], l_plasma['T_rad'], 'rs-', markersize=4, linewidth=1.0, label='LUMINA')
ax6.set_xlabel('Shell ID')
ax6.set_ylabel('T_rad (K)')
ax6.set_title('Radiation Temperature T_rad (LUMINA vs TARDIS)')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

# Inset: T_rad relative error
ax6_inset = ax6.inset_axes([0.55, 0.55, 0.4, 0.4])
t_err = (l_plasma['T_rad'] - t_plasma['T_rad']) / t_plasma['T_rad'] * 100.0
ax6_inset.bar(l_plasma['shell_id'], t_err, color='indianred', alpha=0.7)
ax6_inset.axhline(0, color='gray', linewidth=0.5)
ax6_inset.set_ylabel('T_rad error (%)', fontsize=7)
ax6_inset.set_xlabel('Shell', fontsize=7)
ax6_inset.tick_params(labelsize=6)
ax6_inset.set_ylim(-2, 2)
ax6_inset.set_title(f'Mean |err| = {np.mean(np.abs(t_err)):.2f}%', fontsize=7)

plt.savefig('comprehensive_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: comprehensive_comparison.png")

# ==================== FIGURE 2: RMS summary table ====================
print("\n" + "=" * 60)
print("COMPARISON METRICS SUMMARY")
print("=" * 60)
print(f"  Real packets (LUMINA vs TARDIS):    RMS = {rms_real:.4f}")
if has_virtual:
    print(f"  Virtual packets (LUMINA vs TARDIS): RMS = {rms_virt:.4f}")
    print(f"  TARDIS internal (virt vs real):     RMS = {rms_tardis_vr:.4f}")
print(f"  Dilution factor W:   mean |error| = {np.mean(np.abs(w_err)):.2f}%")
print(f"  Radiation temp T_rad: mean |error| = {np.mean(np.abs(t_err)):.2f}%")
print(f"  T_inner: {cfg.get('T_inner', 'N/A')} K (TARDIS reference)")
print("=" * 60)
