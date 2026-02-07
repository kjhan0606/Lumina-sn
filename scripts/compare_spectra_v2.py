#!/usr/bin/env python3
"""Compare TARDIS vs LUMINA spectra — detailed shape comparison."""
import numpy as np
import json

REF = "data/tardis_reference"

# Load spectra
t_real = np.genfromtxt(f"{REF}/spectrum_real.csv", delimiter=',', names=True)
t_virt = np.genfromtxt(f"{REF}/spectrum_virtual.csv", delimiter=',', names=True)
l = np.genfromtxt("lumina_spectrum.csv", delimiter=',', names=True)

# Reverse TARDIS (descending → ascending)
t_wave = t_real['wavelength_angstrom'][::-1]
t_flux = t_real['flux'][::-1]
tv_wave = t_virt['wavelength_angstrom'][::-1]
tv_flux = t_virt['flux'][::-1]

l_wave = l['wavelength_angstrom']
l_flux = l['flux']

with open(f"{REF}/config.json") as f:
    cfg = json.load(f)
t_exp = cfg['time_explosion_s']

# Normalize LUMINA to same units as TARDIS
l_flux_norm = l_flux * t_exp * 1e8

# Interpolate all onto common grid (500-20000 A, 10A steps)
grid = np.arange(2000, 12001, 10.0)
t_interp = np.interp(grid, t_wave, t_flux)
tv_interp = np.interp(grid, tv_wave, tv_flux)
l_interp = np.interp(grid, l_wave, l_flux_norm)

# Normalize to peak in optical window (4000-7000)
opt = (grid >= 4000) & (grid <= 7000)
t_peak = t_interp[opt].max()
tv_peak = tv_interp[opt].max()
l_peak = l_interp[opt].max()

t_n = t_interp / t_peak
tv_n = tv_interp / tv_peak
l_n = l_interp / l_peak

print("=" * 80)
print("NORMALIZED SPECTRAL COMPARISON (peak in 4000-7000 = 1.0)")
print("=" * 80)

# ASCII visualization of spectra
print(f"\n{'Wave(A)':>8} {'TARDIS':>8} {'LUMINA':>8} {'Diff':>7} | Spectrum (T=+, L=o)")
print("-" * 80)

bar_width = 35
for w_center in range(3000, 10001, 100):
    mask = (grid >= w_center - 50) & (grid < w_center + 50)
    if mask.sum() == 0:
        continue
    tv = t_n[mask].mean()
    lv = l_n[mask].mean()
    diff = lv - tv

    # ASCII bar
    t_pos = int(tv * bar_width)
    l_pos = int(lv * bar_width)
    t_pos = min(max(t_pos, 0), bar_width - 1)
    l_pos = min(max(l_pos, 0), bar_width - 1)

    bar = [' '] * bar_width
    if abs(t_pos - l_pos) <= 0:
        bar[t_pos] = '*'  # overlap
    else:
        bar[min(t_pos, bar_width-1)] = '+'
        bar[min(l_pos, bar_width-1)] = 'o'

    # Mark important features
    marker = ""
    if 6050 <= w_center <= 6200:
        marker = " ← Si II trough"
    elif 3800 <= w_center <= 3900:
        marker = " ← Ca II H&K"
    elif 5400 <= w_center <= 5700:
        marker = " ← S II"
    elif 8100 <= w_center <= 8500:
        marker = " ← Ca II IR"

    print(f"{w_center:>8} {tv:8.4f} {lv:8.4f} {diff:+7.4f} |{''.join(bar)}{marker}")

# === Si II analysis with proper continuum ===
print(f"\n{'='*80}")
print("Si II 6355 ANALYSIS (pseudo-continuum from smoothed spectrum)")
print(f"{'='*80}")

def smooth_spectrum(flux, window=30):
    """Simple boxcar smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(flux, kernel, mode='same')

# Smooth for continuum estimation
t_smooth = smooth_spectrum(t_interp, 50)
l_smooth = smooth_spectrum(l_interp, 50)

# Fit pseudo-continuum using peaks
# Si II region: use peaks at ~4500 and ~6500 to define local continuum
def find_local_peak(wave, flux, center, width=300):
    mask = (wave >= center - width) & (wave <= center + width)
    if mask.sum() == 0:
        return center, 0
    idx = np.argmax(flux[mask])
    return wave[mask][idx], flux[mask][idx]

# TARDIS
tw1, tf1 = find_local_peak(grid, t_interp, 4400)
tw2, tf2 = find_local_peak(grid, t_interp, 6500)
tw3, tf3 = find_local_peak(grid, t_interp, 8800)

# Linear continuum between peaks
t_cont_si = tf1 + (tf2 - tf1) * (grid - tw1) / (tw2 - tw1)

# Find Si II trough (minimum between 5700-6300)
mask_si = (grid >= 5700) & (grid <= 6300)
idx_t = np.argmin(t_interp[mask_si])
t_si_wave = grid[mask_si][idx_t]
t_si_flux = t_interp[mask_si][idx_t]
t_si_cont = t_cont_si[mask_si][idx_t]
t_si_depth = 1.0 - t_si_flux / t_si_cont
t_si_vel = 3e5 * (6355 - t_si_wave) / 6355

# LUMINA
lw1, lf1 = find_local_peak(grid, l_interp, 4400)
lw2, lf2 = find_local_peak(grid, l_interp, 6500)
l_cont_si = lf1 + (lf2 - lf1) * (grid - lw1) / (lw2 - lw1)
idx_l = np.argmin(l_interp[mask_si])
l_si_wave = grid[mask_si][idx_l]
l_si_flux = l_interp[mask_si][idx_l]
l_si_cont = l_cont_si[mask_si][idx_l]
l_si_depth = 1.0 - l_si_flux / l_si_cont
l_si_vel = 3e5 * (6355 - l_si_wave) / 6355

print(f"\n{'':>25} {'TARDIS':>14} {'LUMINA':>14}")
print("-" * 55)
print(f"{'Trough wavelength (A)':>25} {t_si_wave:>14.1f} {l_si_wave:>14.1f}")
print(f"{'Absorption velocity (km/s)':>25} {t_si_vel:>14.0f} {l_si_vel:>14.0f}")
print(f"{'Trough depth (%)':>25} {t_si_depth*100:>13.1f}% {l_si_depth*100:>13.1f}%")
print(f"{'Velocity difference':>25} {'':>14} {l_si_vel-t_si_vel:>+14.0f}")

# === GLOBAL METRICS ===
print(f"\n{'='*80}")
print("GLOBAL SPECTRAL METRICS")
print(f"{'='*80}")

# Chi-squared in optical
opt_mask = (grid >= 3500) & (grid <= 9000) & (t_n > 0.05)
residuals = l_n[opt_mask] - t_n[opt_mask]
chi2 = np.mean(residuals**2) / np.mean(t_n[opt_mask]**2) * 100  # % variance

print(f"Normalized RMS (3500-9000A): {np.sqrt(np.mean(residuals**2)):.4f}")
print(f"Mean absolute deviation: {np.mean(np.abs(residuals)):.4f}")
print(f"Relative variance: {chi2:.2f}%")

# Color comparison (flux ratios at different wavelengths)
print(f"\nColor comparison (normalized flux at key wavelengths):")
for w in [3800, 4300, 5000, 5500, 6000, 6500, 7000, 7500, 8500]:
    mask = (grid >= w - 50) & (grid < w + 50)
    if mask.sum() > 0:
        tr = t_n[mask].mean()
        lr = l_n[mask].mean()
        print(f"  {w}A: TARDIS={tr:.4f}  LUMINA={lr:.4f}  ratio={lr/tr:.4f}")

# Integrated luminosity comparison
print(f"\nIntegrated luminosity (3500-9000A):")
mask_int = (grid >= 3500) & (grid <= 9000)
dw = 10.0  # grid spacing
L_tardis = np.sum(t_interp[mask_int]) * dw
L_lumina = np.sum(l_interp[mask_int]) * dw
print(f"  TARDIS: {L_tardis:.4e} erg/s")
print(f"  LUMINA: {L_lumina:.4e} erg/s")
print(f"  Ratio (LUMINA/TARDIS): {L_lumina/L_tardis:.4f}")
