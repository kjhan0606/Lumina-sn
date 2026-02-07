#!/usr/bin/env python3
"""
Compare LUMINA spectrum vs TARDIS SN 2011fe model spectrum.
Measures goodness-of-fit via RMS, chi-squared, and key spectral features.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ===== Load spectra =====
REF = "data/tardis_reference"

# TARDIS SN 2011fe model spectrum (virtual packets, erg/s/A)
t_sn = np.genfromtxt("data/sn2011fe/tardis_spectrum.csv", delimiter=',', names=True)
t_wave = t_sn['wavelength_A']
t_flux = t_sn['flux_erg_s_A']  # erg/s/A

# TARDIS reference spectra (from our reference run)
t_ref_real = np.genfromtxt(f"{REF}/spectrum_real.csv", delimiter=',', names=True)
t_ref_virt = np.genfromtxt(f"{REF}/spectrum_virtual.csv", delimiter=',', names=True)
trw = t_ref_real['wavelength_angstrom'][::-1]
trf = t_ref_real['flux'][::-1] / 1e8  # erg/s/cm -> erg/s/A
trvw = t_ref_virt['wavelength_angstrom'][::-1]
trvf = t_ref_virt['flux'][::-1] / 1e8

# LUMINA real packet spectrum
if not os.path.exists("lumina_spectrum.csv"):
    print("ERROR: lumina_spectrum.csv not found. Run simulation first.")
    sys.exit(1)
l_real = np.genfromtxt("lumina_spectrum.csv", delimiter=',', names=True)
lrw = l_real['wavelength_angstrom']
lrf = l_real['flux']  # erg/s/cm -> erg/s/A
lrf_A = lrf / 1e8

# LUMINA virtual packet spectrum (if available)
has_virtual = os.path.exists("lumina_spectrum_virtual.csv")
if has_virtual:
    l_virt = np.genfromtxt("lumina_spectrum_virtual.csv", delimiter=',', names=True)
    lvw = l_virt['wavelength_angstrom']
    lvf = l_virt['flux']
    lvf_A = lvf / 1e8

# Observed SN 2011fe spectrum (Pereira+2013, phase -0.3d from B-max)
obs_file = "data/sn2011fe/sn2011fe_observed_Bmax.csv"
has_obs = os.path.exists(obs_file)
if has_obs:
    obs = np.genfromtxt(obs_file, delimiter=',', names=True)
    obs_wave = obs['wavelength_angstrom']
    obs_flux = obs['flux_erg_s_cm2_angstrom']  # erg/s/cm^2/A (observed at Earth)

# ===== Interpolate to common grid =====
grid = np.arange(3500, 9001, 5.0)  # optical window
tf_i = np.interp(grid, t_wave, t_flux)
trf_i = np.interp(grid, trw, trf)
trvf_i = np.interp(grid, trvw, trvf)
lrf_i = np.interp(grid, lrw, lrf_A)
if has_virtual:
    lvf_i = np.interp(grid, lvw, lvf_A)
if has_obs:
    obs_i = np.interp(grid, obs_wave, obs_flux)

# ===== Normalize by peak in 4000-7000 A =====
opt = (grid >= 4000) & (grid <= 7000)
tf_n = tf_i / tf_i[opt].max()
trf_n = trf_i / trf_i[opt].max()
trvf_n = trvf_i / trvf_i[opt].max()
lrf_n = lrf_i / lrf_i[opt].max()
if has_virtual:
    lvf_n = lvf_i / lvf_i[opt].max()
if has_obs:
    obs_n = obs_i / obs_i[opt].max()

# ===== Spectral feature measurement =====
def measure_feature(wave, flux, line_rest, blue_range, red_range, cont_range):
    """Measure P-Cygni absorption feature."""
    mask_cont = (wave >= cont_range[0]) & (wave <= cont_range[1]) & (flux > 0)
    if mask_cont.sum() == 0:
        return None
    F_cont = np.median(flux[mask_cont])

    mask_blue = (wave >= blue_range[0]) & (wave <= blue_range[1])
    if mask_blue.sum() == 0:
        return None
    idx_min = np.argmin(flux[mask_blue])
    F_min = flux[mask_blue][idx_min]
    wave_min = wave[mask_blue][idx_min]

    mask_red = (wave >= red_range[0]) & (wave <= red_range[1])
    if mask_red.sum() == 0:
        return None
    idx_peak = np.argmax(flux[mask_red])
    F_peak = flux[mask_red][idx_peak]
    wave_peak = wave[mask_red][idx_peak]

    depth_cont = 1.0 - F_min / F_cont if F_cont > 0 else 0
    depth_peak = 1.0 - F_min / F_peak if F_peak > 0 else 0
    v_abs = 3e5 * (line_rest - wave_min) / line_rest  # km/s

    return {
        'F_cont': F_cont, 'F_min': F_min, 'wave_min': wave_min,
        'F_peak': F_peak, 'wave_peak': wave_peak,
        'depth_cont': depth_cont, 'depth_peak': depth_peak,
        'v_abs': v_abs
    }

features = [
    ("Ca II H&K",       3945, (3600, 3900), (3900, 4100), (4200, 4600)),
    ("Si II 6355",      6355, (5700, 6250), (6250, 6600), (7000, 7500)),
    ("Si II 5972",      5972, (5600, 5900), (5900, 6100), (6100, 6300)),
    ("S II W-shape",    5640, (5200, 5500), (5500, 5800), (5900, 6100)),
    ("Ca II IR triplet", 8579, (7800, 8400), (8400, 8800), (9000, 9500)),
]

# ===== Compute metrics =====
mask_fit = (grid >= 3500) & (grid <= 9000) & (tf_n > 0.05)

# LUMINA vs TARDIS SN 2011fe
rms_real = np.sqrt(np.mean((lrf_n[mask_fit] - tf_n[mask_fit])**2))
mae_real = np.mean(np.abs(lrf_n[mask_fit] - tf_n[mask_fit]))
noise_est = 0.02
chi2_real = np.sum(((lrf_n[mask_fit] - tf_n[mask_fit]) / noise_est)**2) / mask_fit.sum()

if has_virtual:
    rms_virt = np.sqrt(np.mean((lvf_n[mask_fit] - tf_n[mask_fit])**2))
    mae_virt = np.mean(np.abs(lvf_n[mask_fit] - tf_n[mask_fit]))
    chi2_virt = np.sum(((lvf_n[mask_fit] - tf_n[mask_fit]) / noise_est)**2) / mask_fit.sum()

# LUMINA vs TARDIS reference (our ground truth)
mask_ref = (grid >= 3500) & (grid <= 9000) & (trvf_n > 0.05)
rms_lr_vs_ref = np.sqrt(np.mean((lrf_n[mask_ref] - trf_n[mask_ref])**2))
rms_lv_vs_ref = np.sqrt(np.mean((lvf_n[mask_ref] - trvf_n[mask_ref])**2)) if has_virtual else 0

# TARDIS reference vs TARDIS SN 2011fe (cross-run consistency)
rms_ref_vs_sn = np.sqrt(np.mean((trvf_n[mask_fit] - tf_n[mask_fit])**2))

# vs Observed SN 2011fe
if has_obs:
    rms_obs_tardis = np.sqrt(np.mean((tf_n[mask_fit] - obs_n[mask_fit])**2))
    rms_obs_lumina = np.sqrt(np.mean((lrf_n[mask_fit] - obs_n[mask_fit])**2))
    rms_obs_lv = np.sqrt(np.mean((lvf_n[mask_fit] - obs_n[mask_fit])**2)) if has_virtual else 0

# ===== Print results =====
print("=" * 70)
print("SN 2011fe MODEL FITTING: LUMINA vs TARDIS")
print("=" * 70)

print(f"\nAbsolute Flux (optical 4000-7000 A):")
print(f"  TARDIS SN 2011fe:  {tf_i[opt].mean():.4e} erg/s/A")
print(f"  TARDIS ref (virt): {trvf_i[opt].mean():.4e} erg/s/A  (ratio: {trvf_i[opt].mean()/tf_i[opt].mean():.4f})")
print(f"  LUMINA real:       {lrf_i[opt].mean():.4e} erg/s/A  (ratio: {lrf_i[opt].mean()/tf_i[opt].mean():.4f})")
if has_virtual:
    print(f"  LUMINA virtual:    {lvf_i[opt].mean():.4e} erg/s/A  (ratio: {lvf_i[opt].mean()/tf_i[opt].mean():.4f})")

print(f"\nNormalized RMS (3500-9000 A):")
print(f"  {'Comparison':>45} {'RMS':>8}")
print(f"  {'-'*45} {'-'*8}")
print(f"  {'TARDIS ref vs TARDIS SN 2011fe':>45} {rms_ref_vs_sn:>8.4f}  (cross-run baseline)")
print(f"  {'LUMINA real vs TARDIS ref (real)':>45} {rms_lr_vs_ref:>8.4f}  (ground truth)")
if has_virtual:
    print(f"  {'LUMINA virt vs TARDIS ref (virt)':>45} {rms_lv_vs_ref:>8.4f}  (ground truth)")
print(f"  {'LUMINA real vs TARDIS SN 2011fe':>45} {rms_real:>8.4f}")
if has_virtual:
    print(f"  {'LUMINA virt vs TARDIS SN 2011fe':>45} {rms_virt:>8.4f}")
if has_obs:
    print(f"  {'-'*45} {'-'*8}")
    print(f"  {'TARDIS SN 2011fe vs Observed':>45} {rms_obs_tardis:>8.4f}  (model accuracy)")
    print(f"  {'LUMINA real vs Observed':>45} {rms_obs_lumina:>8.4f}")
    if has_virtual:
        print(f"  {'LUMINA virt vs Observed':>45} {rms_obs_lv:>8.4f}")

print(f"\n{'='*70}")
print("SPECTRAL FEATURE COMPARISON")
print(f"{'='*70}")
print(f"\n{'Feature':>20} {'':>5} {'TARDIS':>12} {'LUMINA(real)':>14}", end="")
if has_virtual:
    print(f" {'LUMINA(virt)':>14}", end="")
print()
print("-" * (55 + (16 if has_virtual else 0)))

spectra_list = [("TARDIS", grid, tf_n), ("LUMINA(real)", grid, lrf_n)]
if has_virtual:
    spectra_list.append(("LUMINA(virt)", grid, lvf_n))

for name, line_rest, blue_r, red_r, cont_r in features:
    results = []
    for label, w, f in spectra_list:
        results.append(measure_feature(w, f, line_rest, blue_r, red_r, cont_r))

    if results[0] is not None:
        r_t = results[0]
        r_l = results[1]
        print(f"  {name:>18} {'depth%':>5} {r_t['depth_peak']*100:>11.1f}% {r_l['depth_peak']*100:>13.1f}%", end="")
        if has_virtual and results[2]:
            print(f" {results[2]['depth_peak']*100:>13.1f}%", end="")
        print()
        print(f"  {'':>18} {'v km/s':>5} {r_t['v_abs']:>12.0f} {r_l['v_abs']:>14.0f}", end="")
        if has_virtual and results[2]:
            print(f" {results[2]['v_abs']:>14.0f}", end="")
        print()
        print(f"  {'':>18} {'wave':>5} {r_t['wave_min']:>12.1f} {r_l['wave_min']:>14.1f}", end="")
        if has_virtual and results[2]:
            print(f" {results[2]['wave_min']:>14.1f}", end="")
        print()

# ===== FIGURE: 3-panel comparison =====
fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                          gridspec_kw={'height_ratios': [3, 3, 1.5]})
fig.suptitle('LUMINA-SN vs TARDIS SN 2011fe Model',
             fontsize=16, fontweight='bold')

# Panel 1: Full spectrum (normalized, including observation)
ax = axes[0]
if has_obs:
    ax.plot(grid, obs_n, 'k-', alpha=0.8, linewidth=1.2, label='Observed (Pereira+2013, phase -0.3d)')
ax.plot(grid, tf_n, 'b-', alpha=0.6, linewidth=0.8, label='TARDIS SN 2011fe')
ax.plot(grid, lrf_n, 'r-', alpha=0.6, linewidth=0.8, label='LUMINA real')
if has_virtual:
    ax.plot(grid, lvf_n, 'darkred', alpha=0.5, linewidth=0.8, label='LUMINA virtual', linestyle='--')
ax.set_xlim(3500, 9000)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Normalized Flux')
ax.set_title('Model vs Observation (normalized to peak)')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Annotate features
feature_labels = [
    ("Ca II\nH&K", 3800, 0.85),
    ("S II", 5450, 0.60),
    ("Si II\n5972", 5850, 0.45),
    ("Si II\n6355", 6100, 0.35),
    ("Ca II\nIR", 8200, 0.50),
]
ymax = max(t_flux[(t_wave > 3500) & (t_wave < 9000)])
for label, x, yfrac in feature_labels:
    ax.annotate(label, xy=(x, ymax * yfrac), fontsize=7, color='green',
                ha='center', va='center', alpha=0.7)

# Panel 2: Zoomed Si II + S II region with observation
ax = axes[1]
mask_zoom = (grid >= 4500) & (grid <= 7500)
if has_obs:
    ax.plot(grid[mask_zoom], obs_n[mask_zoom], 'k-', alpha=0.8, linewidth=1.2, label='Observed')
ax.plot(grid[mask_zoom], tf_n[mask_zoom], 'b-', alpha=0.6, linewidth=0.8, label='TARDIS')
ax.plot(grid[mask_zoom], lrf_n[mask_zoom], 'r-', alpha=0.6, linewidth=0.8, label='LUMINA real')
if has_virtual:
    ax.plot(grid[mask_zoom], lvf_n[mask_zoom], 'darkred', alpha=0.5, linewidth=0.8, label='LUMINA virtual', linestyle='--')
ax.axvline(6355, color='green', linestyle='--', alpha=0.3, label='Si II 6355')
ax.axvline(5640, color='purple', linestyle='--', alpha=0.3, label='S II 5640')
for v in [10000, 12000, 15000]:
    w = 6355 * (1 - v / 3e5)
    ax.axvline(w, color='orange', linestyle=':', alpha=0.2)
    ax.text(w, 0.12, f'{v//1000}k', fontsize=6, color='orange', ha='center')
ax.set_xlim(4500, 7500)
ax.set_ylabel('Normalized Flux')
ax.set_title('Si II + S II Region Detail')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Add velocity scale for Si II
ax_top = ax.twiny()
si_rest = 6355.0
v_ticks = [5000, 10000, 15000, 20000, 25000]
w_ticks = [si_rest * (1 - v / 3e5) for v in v_ticks]
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(w_ticks)
ax_top.set_xticklabels([f'{v//1000}k' for v in v_ticks], fontsize=7)
ax_top.set_xlabel('Si II velocity (km/s)', fontsize=8)

# Add stats box
stats_text = f"vs Observed:"
if has_obs:
    stats_text += f"\n  TARDIS RMS: {rms_obs_tardis:.4f}"
    stats_text += f"\n  LUMINA RMS: {rms_obs_lumina:.4f}"
stats_text += f"\nvs TARDIS ref:"
stats_text += f"\n  LUMINA RMS: {rms_lr_vs_ref:.4f}"
axes[0].text(0.02, 0.97, stats_text, transform=axes[0].transAxes,
        fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 3: Residuals vs Observed
ax = axes[2]
if has_obs:
    res_tardis_obs = uniform_filter1d(tf_n - obs_n, 10)
    ax.plot(grid, res_tardis_obs, 'b-', alpha=0.7, linewidth=0.8,
            label=f'TARDIS - Obs (RMS={rms_obs_tardis:.4f})')
    res_lumina_obs = uniform_filter1d(lrf_n - obs_n, 10)
    ax.plot(grid, res_lumina_obs, 'r-', alpha=0.7, linewidth=0.8,
            label=f'LUMINA - Obs (RMS={rms_obs_lumina:.4f})')
    if has_virtual:
        res_lv_obs = uniform_filter1d(lvf_n - obs_n, 10)
        ax.plot(grid, res_lv_obs, 'darkred', alpha=0.5, linewidth=0.8,
                label=f'LUMINA virt - Obs (RMS={rms_obs_lv:.4f})', linestyle='--')
else:
    res_real_s = uniform_filter1d(lrf_n - tf_n, 10)
    ax.plot(grid, res_real_s, 'r-', alpha=0.7, linewidth=0.8,
            label=f'LUMINA - TARDIS (RMS={rms_real:.4f})')
ax.axhline(0, color='gray', linewidth=0.5)
ax.fill_between(grid, -0.05, 0.05, color='green', alpha=0.1, label='5% band')
ax.set_xlim(3500, 9000)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel('Wavelength (A)')
ax.set_ylabel('Residual')
ax.set_title('Residuals (LUMINA - TARDIS, smoothed)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sn2011fe_fit_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: sn2011fe_fit_comparison.png")

# ===== FIGURE 2: Si II detail =====
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Si II 6355 A: LUMINA vs TARDIS SN 2011fe', fontsize=14, fontweight='bold')

mask_si = (grid >= 5500) & (grid <= 7000)

if has_obs:
    ax1.plot(grid[mask_si], obs_n[mask_si], 'k-', linewidth=1.5, label='Observed', alpha=0.9)
ax1.plot(grid[mask_si], tf_n[mask_si], 'b-', linewidth=1.5, label='TARDIS', alpha=0.7)
ax1.plot(grid[mask_si], lrf_n[mask_si], 'r-', linewidth=1.5, label='LUMINA real', alpha=0.7)
if has_virtual:
    ax1.plot(grid[mask_si], lvf_n[mask_si], 'r--', linewidth=1.2, label='LUMINA virtual', alpha=0.5)
ax1.axvline(6355, color='green', linestyle='--', alpha=0.4, label='Si II rest')
for v in [10000, 12000, 15000, 20000]:
    w = 6355 * (1 - v / 3e5)
    ax1.axvline(w, color='orange', linestyle=':', alpha=0.3)
    ax1.text(w, 0.15, f'{v//1000}k', fontsize=7, color='orange', ha='center')
ax1.set_xlabel('Wavelength (A)')
ax1.set_ylabel('Normalized Flux')
ax1.set_title('Normalized Spectrum')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# P-Cygni profile analysis
ax2.plot(grid[mask_si], tf_n[mask_si] - lrf_n[mask_si], 'r-', linewidth=1.0,
         label='TARDIS - LUMINA(real)', alpha=0.8)
if has_virtual:
    ax2.plot(grid[mask_si], tf_n[mask_si] - lvf_n[mask_si], 'b-', linewidth=1.0,
             label='TARDIS - LUMINA(virt)', alpha=0.8)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(6355, color='green', linestyle='--', alpha=0.4)
ax2.set_xlabel('Wavelength (A)')
ax2.set_ylabel('Residual (TARDIS - LUMINA)')
ax2.set_title('Si II Region Residual')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sn2011fe_siII_detail.png', dpi=150, bbox_inches='tight')
print(f"Saved: sn2011fe_siII_detail.png")
