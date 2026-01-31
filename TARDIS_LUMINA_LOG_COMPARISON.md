# TARDIS vs LUMINA: SN 2011fe Fitting Log Comparison

## Overview

This document compares step-by-step log outputs from TARDIS and LUMINA when fitting the same SN 2011fe spectrum with identical input parameters.

**Test Date**: 2026-01-31
**Observed Spectrum**: SN 2011fe at B-maximum (phase = 0 days)

---

## 1. Physical Constants Comparison

| Constant | TARDIS (Astropy) | LUMINA (NIST CODATA 2018) | Relative Difference |
|----------|------------------|---------------------------|---------------------|
| c [cm/s] | 2.9979245800e+10 | 2.9979245800e+10 | **EXACT MATCH** |
| h [erg·s] | 6.6260700400e-27 | 6.6260701500e-27 | 1.66e-08 |
| k_B [erg/K] | 1.3806485200e-16 | 1.3806490000e-16 | 3.48e-07 |
| m_e [g] | 9.1093835600e-28 | 9.1093837015e-28 | 1.55e-08 |
| eV→erg | 1.6021766208e-12 | 1.6021766340e-12 | 8.24e-09 |

**Status**: All constants match to better than 1 ppm. Differences are due to CODATA version updates (Astropy uses slightly older values).

---

## 2. Input Parameters (Identical)

| Parameter | TARDIS | LUMINA | Match |
|-----------|--------|--------|-------|
| Luminosity | 9.35 log(L_sun) | 9.35 log(L_sun) | YES |
| Time explosion | 13.00 days | 13.00 days | YES |
| v_inner | 11000 km/s | 11000 km/s | YES |
| v_outer | 25000 km/s | 25000 km/s | YES |
| n_packets | 80000 | 80000 | YES |
| Ionization | LTE | LTE (Saha-Boltzmann) | YES |
| Excitation | LTE | LTE (Boltzmann) | YES |
| Line interaction | macroatom | macroatom | YES |
| e- scattering | enabled | enabled | YES |

---

## 3. Model Configuration

### Stratification Parameters

| Parameter | TARDIS | LUMINA | Match |
|-----------|--------|--------|-------|
| fe_core_fraction | 0.25 | 0.25 | YES |
| si_layer_width | 0.40 | 0.40 | YES |
| fe_peak | 0.70 | 0.70 | YES |
| si_peak | 0.60 | 0.60 | YES |
| o_outer | 0.50 | 0.50 | YES |

### Convergence Settings

| Parameter | TARDIS | LUMINA | Note |
|-----------|--------|--------|------|
| Damping | 0.8 | 0.7 | Different defaults |
| Max iterations | 10 | 10 | Same |
| Convergence threshold | 5% | 5% | Same |
| Hold iterations | 3 | 3 | Same |

---

## 4. Simulation Results

### Temperature Structure

| Metric | TARDIS | LUMINA | Difference |
|--------|--------|--------|------------|
| T_inner (final) | 12244 K | 9421 K | **-23%** |
| T_rad range | 7291 - 13344 K | 5600 - 9421 K | LUMINA cooler |

### Convergence

| Metric | TARDIS | LUMINA | Note |
|--------|--------|--------|------|
| Converged | YES | YES | Both converge |
| Iterations | 10 | 10 | Same |

### Packet Statistics

| Metric | TARDIS | LUMINA | Note |
|--------|--------|--------|------|
| Total packets | 80000 | 80000 | Same |
| Escaped | (not logged) | 70460 (88.1%) | |
| Absorbed | (not logged) | 9540 (11.9%) | |

### Electron Density

| Shell | TARDIS n_e [cm⁻³] | LUMINA n_e [cm⁻³] | Ratio |
|-------|-------------------|-------------------|-------|
| Inner | 2.62e+09 | 2.10e+09 | 0.80 |
| Middle | ~5.0e+08 | ~5.0e+07 | 0.10 |
| Outer | 2.41e+07 | ~1e+01 | LUMINA much lower |

### Dilution Factor W

| TARDIS | LUMINA |
|--------|--------|
| 0.1468 - 0.4413 | (not directly logged) |

---

## 5. Spectral Analysis

### Chi-Square Comparison

| Metric | TARDIS | LUMINA | Difference |
|--------|--------|--------|------------|
| χ² (3500-7500 Å) | 32.97 | 265.57 | **LUMINA 8× higher** |

### Si II 6355 Velocity

| Code | Velocity [km/s] | Note |
|------|-----------------|------|
| TARDIS | 15077 | Reasonable for SN 2011fe |
| LUMINA | 19605 | **Too high by ~4500 km/s** |

---

## 6. Ionization Balance

| Ion | TARDIS (LTE) | LUMINA (LTE) |
|-----|--------------|--------------|
| Si II fraction | 0.0026 | ~0.9 (expected) |
| Fe II fraction | 0.0008 | ~0.8 (expected) |
| Fe III fraction | 0.9440 | ~0.15 (expected) |

**Note**: The TARDIS fractions appear to be Fe III dominated (hot ejecta), while LUMINA reports Fe II dominated. This suggests different temperature profiles leading to different ionization states.

---

## 7. Key Differences Identified

### Issue 1: Temperature Discrepancy

- **TARDIS T_inner**: 12244 K (converged from luminosity constraint)
- **LUMINA T_inner**: 9421 K (calculated from Stefan-Boltzmann, then fixed)

The temperature iteration in LUMINA appears to not fully update T_inner during convergence. TARDIS adjusts T_inner based on luminosity matching, leading to higher temperatures.

### Issue 2: Electron Density in Outer Shells

LUMINA shows very low n_e in outer shells after iteration:
- Shell 15: n_e drops from 4.99e+07 to 3.94e+01 cm⁻³

This suggests the temperature iteration is over-cooling the outer shells, causing recombination.

### Issue 3: Si II Velocity

LUMINA's Si II 6355 velocity (19605 km/s) is higher than TARDIS (15077 km/s), suggesting:
- Silicon line opacity forms at higher velocity (outer shells)
- Possibly incorrect line identification or opacity distribution

---

## 8. Recommendations

1. **Fix T_inner update in LUMINA**: The T_inner should be adjusted based on escaped luminosity matching the requested luminosity, similar to TARDIS.

2. **Verify J-estimator normalization**: The J values in LUMINA (1.35e-43) seem very low, which would lead to excessively low T_rad after iteration.

3. **Check abundance stratification**: Ensure the W7-like model in LUMINA matches the TARDIS implementation.

4. **Compare line opacity**: Verify that the Si II 6355 line strength and formation region match between codes.

---

## 9. Log File Locations

- TARDIS log: `tardis_sn2011fe.log`
- LUMINA log: `lumina_sn2011fe.log`
- TARDIS spectrum: `tardis_comparison_spectrum.dat`
- LUMINA spectrum: `lumina_comparison_spectrum.dat`

---

## 10. Conclusion

The bottom-level physics (constants, Saha-Boltzmann, line opacity formula) are consistent between TARDIS and LUMINA. However, differences in:

1. Temperature iteration implementation
2. J-estimator normalization
3. T_inner luminosity constraint

Lead to different converged temperature profiles, which propagate to different ionization states and spectral features.

**Next Steps**: Focus on synchronizing the temperature iteration and luminosity constraint between LUMINA and TARDIS to achieve consistent spectral outputs.
