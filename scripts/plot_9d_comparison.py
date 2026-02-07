#!/usr/bin/env python3
"""
Compare 7D uniform vs 9D stratified fitting results for SN 2011fe.
Generates a multi-panel comparison figure.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load observed spectrum
obs_file = PROJECT_ROOT / "data" / "sn2011fe" / "sn2011fe_observed_Bmax.csv"
obs = np.genfromtxt(str(obs_file), delimiter=',', names=True)
obs_wave = obs['wavelength_angstrom']
obs_flux = obs['flux_erg_s_cm2_angstrom']

# Load TARDIS reference
tardis_file = PROJECT_ROOT / "data" / "sn2011fe" / "tardis_spectrum.csv"
ts = np.genfromtxt(str(tardis_file), delimiter=',', names=True)
tardis_wave = ts['wavelength_A']
tardis_flux = ts['flux_erg_s_A']

# Common grid
grid = np.arange(3500, 9001, 5.0)
opt = (grid >= 4000) & (grid <= 7000)

# Normalize observed
obs_i = np.interp(grid, obs_wave, obs_flux)
obs_n = obs_i / obs_i[opt].max()

# Normalize TARDIS
tardis_i = np.interp(grid, tardis_wave, tardis_flux)
tardis_n = tardis_i / tardis_i[opt].max()

# Load 9D results
res_9d = np.genfromtxt(str(PROJECT_ROOT / "fit_results_final.csv"), delimiter=',', names=True)

# Load 7D results (from phase1 csv columns: X_Si, X_Fe, X_O format)
res_7d_file = PROJECT_ROOT / "fit_results_phase1.csv"
# Actually get 7D final from git
import subprocess
p = subprocess.run(['git', 'show', 'HEAD:fit_results_final.csv'],
                   capture_output=True, text=True, cwd=str(PROJECT_ROOT))
lines_7d = p.stdout.strip().split('\n')
header_7d = lines_7d[0].split(',')
best_7d = lines_7d[1].split(',')  # best model (sorted by RMS)
rms_7d = float(best_7d[header_7d.index('rms')])
si_depth_7d = float(best_7d[header_7d.index('si_depth')])
si_vel_7d = float(best_7d[header_7d.index('si_velocity')])

# 9D best
rms_9d = float(res_9d['rms'][0])
si_depth_9d = float(res_9d['si_depth'][0])
si_vel_9d = float(res_9d['si_velocity'][0])

# ===== Create figure =====
fig = plt.figure(figsize=(18, 14))
fig.suptitle('SN 2011fe Parameter Fitting: 7D Uniform vs 9D Stratified',
             fontsize=16, fontweight='bold', y=0.98)

# Layout: 2x2 main panels + bottom comparison table
gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.5], hspace=0.35, wspace=0.3)

# --- Panel 1: Full spectra comparison ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(grid, obs_n, 'k-', linewidth=1.5, alpha=0.9, label='Observed (Pereira+2013)')
ax1.plot(grid, tardis_n, 'b-', linewidth=0.8, alpha=0.5, label='TARDIS reference')

# 9D best fit (current fit_results are 9D)
# We don't have the spectra saved, so we just show the RMS comparison
ax1.set_xlim(3500, 9000)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel('Normalized Flux', fontsize=11)
ax1.set_title('Full Optical Spectrum (Observation + TARDIS)', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Si II 6355 detail ---
ax2 = fig.add_subplot(gs[0, 1])
si_mask = (grid >= 5500) & (grid <= 7000)
ax2.plot(grid[si_mask], obs_n[si_mask], 'k-', linewidth=1.5, alpha=0.9, label='Observed')
ax2.plot(grid[si_mask], tardis_n[si_mask], 'b-', linewidth=0.8, alpha=0.5, label='TARDIS')
ax2.axvline(6355, color='green', linestyle='--', alpha=0.4, label='Si II 6355 rest')
for v in [10000, 12000, 15000]:
    w = 6355 * (1 - v / 3e5)
    ax2.axvline(w, color='orange', linestyle=':', alpha=0.3)
    ax2.text(w, 0.12, f'{v // 1000}k', fontsize=7, color='orange', ha='center')
ax2.set_xlim(5500, 7000)
ax2.set_ylabel('Normalized Flux', fontsize=11)
ax2.set_title('Si II 6355 Region', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Panel 3: RMS comparison bar chart ---
ax3 = fig.add_subplot(gs[1, 0])

# Get all 9D Phase 2 results for distribution
p2_file = PROJECT_ROOT / "fit_results_phase2.csv"
if p2_file.exists():
    p2 = np.genfromtxt(str(p2_file), delimiter=',', names=True)
    p2_rms = p2['rms']
else:
    p2_rms = np.array([rms_9d])

# Bar chart
categories = ['7D Uniform\n(Best)', '9D Stratified\n(Best)', '9D Stratified\n(#2)', '9D Stratified\n(#3)']
rms_values = [rms_7d, float(res_9d['rms'][0]), float(res_9d['rms'][1]), float(res_9d['rms'][2])]
colors = ['#4477AA', '#EE6677', '#CCBB44', '#66CCEE']
bars = ax3.bar(categories, rms_values, color=colors, edgecolor='k', linewidth=0.5, width=0.6)
for bar, val in zip(bars, rms_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax3.set_ylabel('RMS vs Observed', fontsize=11)
ax3.set_title('RMS Comparison', fontsize=12)
ax3.set_ylim(0, max(rms_values) * 1.2)
ax3.grid(True, alpha=0.3, axis='y')

# --- Panel 4: Si II depth comparison ---
ax4 = fig.add_subplot(gs[1, 1])
si_categories = ['7D Uniform\n(Best)', '9D #1\n(Best RMS)', '9D #2', '9D #3']
si_values = [si_depth_7d * 100, float(res_9d['si_depth'][0]) * 100,
             float(res_9d['si_depth'][1]) * 100, float(res_9d['si_depth'][2]) * 100]
bars2 = ax4.bar(si_categories, si_values, color=colors, edgecolor='k', linewidth=0.5, width=0.6)
for bar, val in zip(bars2, si_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
# Add TARDIS reference line
ax4.axhline(93, color='blue', linestyle='--', alpha=0.5, label='TARDIS (93%)')
ax4.set_ylabel('Si II 6355 Depth (%)', fontsize=11)
ax4.set_title('Si II Feature Depth', fontsize=12)
ax4.set_ylim(0, 100)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# --- Panel 5: Summary table ---
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Build comparison table
table_data = [
    ['Metric', '7D Uniform (Best)', '9D Stratified (Best)', 'Change'],
    ['RMS vs Observed', f'{rms_7d:.4f}', f'{rms_9d:.4f}',
     f'{(rms_9d - rms_7d)/rms_7d*100:+.1f}%'],
    ['Si II Depth', f'{si_depth_7d:.1%}', f'{si_depth_9d:.1%}',
     f'{(si_depth_9d - si_depth_7d)*100:+.1f}pp'],
    ['Si II Velocity', f'{si_vel_7d:.0f} km/s', f'{si_vel_9d:.0f} km/s',
     f'{si_vel_9d - si_vel_7d:+.0f} km/s'],
    ['Composition', 'Uniform (7 params)',
     '3-zone stratified (9 params)', ''],
    ['Search Space', '7D (150 LHS)', '9D (200 LHS)', '+2 dimensions'],
]

# 9D best-fit zone info
v_core = float(res_9d['v_core'][0])
v_wall = float(res_9d['v_wall'][0])
X_Fe_core = float(res_9d['X_Fe_core'][0])
X_Si_wall = float(res_9d['X_Si_wall'][0])

table_data.append(['9D Best Zones',
    f'Core(<{v_core:.0f}): Fe={X_Fe_core:.2f}',
    f'Wall(<{v_wall:.0f}): Si={X_Si_wall:.2f}',
    f'Outer: Si=0.02 Fe=0.01'])

table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                   colWidths=[0.18, 0.28, 0.28, 0.26])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

# Style header row
for j in range(4):
    table[0, j].set_facecolor('#4477AA')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Alternate row colors
for i in range(1, len(table_data)):
    color = '#f0f0f0' if i % 2 == 0 else 'white'
    for j in range(4):
        table[i, j].set_facecolor(color)

plt.savefig(str(PROJECT_ROOT / "fit_7d_vs_9d_comparison.png"),
            dpi=150, bbox_inches='tight')
print(f"Saved: {PROJECT_ROOT / 'fit_7d_vs_9d_comparison.png'}")
plt.close()

# Print summary
print(f"\n{'='*60}")
print("7D vs 9D Fitting Comparison Summary")
print(f"{'='*60}")
print(f"7D Uniform:     RMS={rms_7d:.4f}  Si_depth={si_depth_7d:.1%}  v_Si={si_vel_7d:.0f}")
print(f"9D Stratified:  RMS={rms_9d:.4f}  Si_depth={si_depth_9d:.1%}  v_Si={si_vel_9d:.0f}")
print(f"RMS improvement: {(rms_7d - rms_9d)/rms_7d*100:.1f}%")
print(f"\n9D Best-fit zones:")
print(f"  Core (<{v_core:.0f} km/s): Fe={X_Fe_core:.3f}")
print(f"  Wall (<{v_wall:.0f} km/s): Si={X_Si_wall:.3f}")
print(f"  Outer (>{v_wall:.0f} km/s): Si=0.02, Fe=0.01")
