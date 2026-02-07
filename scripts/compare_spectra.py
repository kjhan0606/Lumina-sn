#!/usr/bin/env python3
"""Compare TARDIS vs LUMINA spectra: shape, Si II 6355 trough, normalization."""
import numpy as np
import json

REF = "data/tardis_reference"

# Load TARDIS spectrum (real packets)
t_real = np.genfromtxt(f"{REF}/spectrum_real.csv", delimiter=',', names=True)
t_virt = np.genfromtxt(f"{REF}/spectrum_virtual.csv", delimiter=',', names=True)

# TARDIS wavelength is DESCENDING — reverse to ascending
t_wave = t_real['wavelength_angstrom'][::-1]
t_flux = t_real['flux'][::-1]
tv_wave = t_virt['wavelength_angstrom'][::-1]
tv_flux = t_virt['flux'][::-1]

# Load LUMINA spectrum
l = np.genfromtxt("lumina_spectrum.csv", delimiter=',', names=True)
l_wave = l['wavelength_angstrom']
l_flux = l['flux']

# Load config for normalization analysis
with open(f"{REF}/config.json") as f:
    cfg = json.load(f)
t_exp = cfg['time_explosion_s']

print("=" * 70)
print("SPECTRUM COMPARISON: TARDIS vs LUMINA")
print("=" * 70)

# GPU code already produces correct units (erg/s/cm)
l_flux_norm = l_flux

print(f"\nNo normalization needed (GPU outputs erg/s/cm directly)")

# === OPTICAL WINDOW (3500-8500 A) ===
opt = (3500, 8500)
mask_t = (t_wave >= opt[0]) & (t_wave <= opt[1]) & (t_flux > 0)
mask_l = (l_wave >= opt[0]) & (l_wave <= opt[1]) & (l_flux > 0)

print(f"\nOptical window ({opt[0]}-{opt[1]} A):")
print(f"  TARDIS mean flux: {t_flux[mask_t].mean():.4e} erg/s/cm")
print(f"  LUMINA mean flux (corrected): {l_flux_norm[mask_l].mean():.4e} erg/s/cm")
print(f"  Flux ratio (LUMINA/TARDIS): {l_flux_norm[mask_l].mean()/t_flux[mask_t].mean():.4f}")

# === Si II 6355 ANALYSIS ===
print(f"\n{'='*70}")
print("Si II 6355 Å TROUGH ANALYSIS")
print(f"{'='*70}")

def measure_pcygni(wave, flux, line_rest=6355.0, blue_range=(5700, 6250),
                    red_range=(6250, 6600), cont_range=(7000, 7500)):
    """Measure P-Cygni profile properties."""
    # Continuum level (red side)
    mask_cont = (wave >= cont_range[0]) & (wave <= cont_range[1]) & (flux > 0)
    if mask_cont.sum() == 0:
        return None
    F_cont = np.median(flux[mask_cont])

    # Blue trough (absorption minimum)
    mask_blue = (wave >= blue_range[0]) & (wave <= blue_range[1])
    if mask_blue.sum() == 0:
        return None
    blue_fluxes = flux[mask_blue]
    blue_waves = wave[mask_blue]
    idx_min = np.argmin(blue_fluxes)
    F_min = blue_fluxes[idx_min]
    wave_min = blue_waves[idx_min]

    # Red emission peak
    mask_red = (wave >= red_range[0]) & (wave <= red_range[1])
    red_fluxes = flux[mask_red]
    red_waves = wave[mask_red]
    idx_peak = np.argmax(red_fluxes)
    F_peak = red_fluxes[idx_peak]
    wave_peak = red_waves[idx_peak]

    # Trough depth (relative to continuum)
    depth_cont = 1.0 - F_min / F_cont
    # Trough depth (relative to red peak)
    depth_peak = 1.0 - F_min / F_peak

    # Velocity at absorption minimum
    v_abs = 3e5 * (line_rest - wave_min) / line_rest  # km/s

    return {
        'F_cont': F_cont,
        'F_min': F_min,
        'wave_min': wave_min,
        'F_peak': F_peak,
        'wave_peak': wave_peak,
        'depth_cont': depth_cont,
        'depth_peak': depth_peak,
        'v_abs_km_s': v_abs,
    }

# Measure for TARDIS real, virtual, and LUMINA
for label, w, f in [("TARDIS (real)", t_wave, t_flux),
                     ("TARDIS (virtual)", tv_wave, tv_flux),
                     ("LUMINA", l_wave, l_flux_norm)]:
    result = measure_pcygni(w, f)
    if result is None:
        print(f"\n{label}: Could not measure Si II feature")
        continue
    print(f"\n{label}:")
    print(f"  Continuum (7000-7500A): {result['F_cont']:.4e}")
    print(f"  Trough minimum: {result['F_min']:.4e} at {result['wave_min']:.1f} A")
    print(f"  Red peak: {result['F_peak']:.4e} at {result['wave_peak']:.1f} A")
    print(f"  Trough depth (vs continuum): {result['depth_cont']*100:.1f}%")
    print(f"  Trough depth (vs red peak): {result['depth_peak']*100:.1f}%")
    print(f"  Absorption velocity: {result['v_abs_km_s']:.0f} km/s")

# === FEATURE COMPARISON TABLE ===
print(f"\n{'='*70}")
print("COMPARISON SUMMARY")
print(f"{'='*70}")

r_t = measure_pcygni(t_wave, t_flux)
r_tv = measure_pcygni(tv_wave, tv_flux)
r_l = measure_pcygni(l_wave, l_flux_norm)

if r_t and r_l:
    print(f"{'':>25} {'TARDIS(real)':>14} {'TARDIS(virt)':>14} {'LUMINA':>14}")
    print("-" * 70)
    print(f"{'Si II depth (cont)':>25} {r_t['depth_cont']*100:>13.1f}% {r_tv['depth_cont']*100:>13.1f}% {r_l['depth_cont']*100:>13.1f}%")
    print(f"{'Si II depth (peak)':>25} {r_t['depth_peak']*100:>13.1f}% {r_tv['depth_peak']*100:>13.1f}% {r_l['depth_peak']*100:>13.1f}%")
    print(f"{'Abs velocity (km/s)':>25} {r_t['v_abs_km_s']:>14.0f} {r_tv['v_abs_km_s']:>14.0f} {r_l['v_abs_km_s']:>14.0f}")
    print(f"{'Trough wave (A)':>25} {r_t['wave_min']:>14.1f} {r_tv['wave_min']:>14.1f} {r_l['wave_min']:>14.1f}")
    print(f"{'Red peak wave (A)':>25} {r_t['wave_peak']:>14.1f} {r_tv['wave_peak']:>14.1f} {r_l['wave_peak']:>14.1f}")

# === OTHER FEATURES ===
print(f"\n{'='*70}")
print("OTHER SPECTRAL FEATURES")
print(f"{'='*70}")

features = [
    ("Ca II H&K", 3945.0, (3600, 3900), (3900, 4100), (4200, 4600)),
    ("S II W-shape", 5640.0, (5200, 5500), (5500, 5800), (5900, 6100)),
    ("Si II 5972", 5972.0, (5600, 5900), (5900, 6100), (6100, 6300)),
    ("Ca II IR triplet", 8579.0, (7800, 8400), (8400, 8800), (9000, 9500)),
]

for name, line_rest, blue_r, red_r, cont_r in features:
    print(f"\n{name} ({line_rest:.0f} A):")
    r_t2 = measure_pcygni(t_wave, t_flux, line_rest, blue_r, red_r, cont_r)
    r_tv2 = measure_pcygni(tv_wave, tv_flux, line_rest, blue_r, red_r, cont_r)
    r_l2 = measure_pcygni(l_wave, l_flux_norm, line_rest, blue_r, red_r, cont_r)
    if r_t2 and r_l2:
        print(f"  {'':>15} {'TARDIS(real)':>14} {'TARDIS(virt)':>14} {'LUMINA':>14}")
        print(f"  {'Depth(cont)':>15} {r_t2['depth_cont']*100:>13.1f}% {r_tv2['depth_cont']*100:>13.1f}% {r_l2['depth_cont']*100:>13.1f}%")
        print(f"  {'v_abs (km/s)':>15} {r_t2['v_abs_km_s']:>14.0f} {r_tv2['v_abs_km_s']:>14.0f} {r_l2['v_abs_km_s']:>14.0f}")
    elif r_t2:
        print(f"  TARDIS: depth={r_t2['depth_cont']*100:.1f}%, v={r_t2['v_abs_km_s']:.0f} km/s")
        print(f"  LUMINA: feature not measurable")

# === OVERALL SHAPE COMPARISON ===
print(f"\n{'='*70}")
print("NORMALIZED SHAPE COMPARISON (optical)")
print(f"{'='*70}")

# Interpolate LUMINA onto TARDIS wavelength grid
from numpy import interp
l_interp = interp(t_wave, l_wave, l_flux_norm)

# Compare in optical window (4000-8000)
mask = (t_wave >= 4000) & (t_wave <= 8000) & (t_flux > 0) & (l_interp > 0)
if mask.sum() > 0:
    # Normalize both to their mean in this window
    t_norm = t_flux[mask] / t_flux[mask].mean()
    l_norm = l_interp[mask] / l_interp[mask].mean()

    residual = (l_norm - t_norm)
    rms = np.sqrt(np.mean(residual**2))
    print(f"RMS residual (normalized): {rms:.4f}")
    print(f"Mean absolute residual: {np.mean(np.abs(residual)):.4f}")

    # Per-100A band comparison
    print(f"\n{'Wave band':>12} {'TARDIS_norm':>12} {'LUMINA_norm':>12} {'Residual':>10}")
    print("-" * 50)
    for wc in range(4000, 8001, 500):
        band = (t_wave[mask] >= wc) & (t_wave[mask] < wc + 500)
        if band.sum() > 0:
            t_mean = t_norm[band].mean()
            l_mean = l_norm[band].mean()
            print(f"  {wc}-{wc+500}A {t_mean:12.4f} {l_mean:12.4f} {l_mean-t_mean:+10.4f}")

print(f"\n{'='*70}")
print("NOTE: LUMINA GPU spectrum outputs erg/s/cm directly (no normalization needed)")
print(f"{'='*70}")
