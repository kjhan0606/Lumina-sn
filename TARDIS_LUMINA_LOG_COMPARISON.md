# TARDIS vs LUMINA: Bottom-Level Algorithm Comparison

## Overview

This document provides a function-by-function comparison of the macro-atom algorithms between TARDIS and LUMINA, identifying exact matches and tunable parameter differences.

**Test Date**: 2026-02-01
**Status**: Core algorithms verified identical; tunable parameters identified

---

## 1. Bottom-Level Function Comparison

### All Core Functions MATCH Exactly

| Function | Formula | TARDIS | LUMINA | Status |
|----------|---------|--------|--------|--------|
| Einstein A | A_ul = (8π²e²ν²)/(m_e c³) × f_lu × (g_l/g_u) | 6.8848e+07 | 6.8848e+07 | ✓ MATCH |
| Einstein B_ul | B_ul = c³/(8πhν³) × A_ul | 1.0693e+20 | 1.0693e+20 | ✓ MATCH |
| Einstein B_lu | B_lu = (g_u/g_l) × B_ul | 1.0693e+20 | 1.0693e+20 | ✓ MATCH |
| Sobolev β | β = (1 - exp(-τ))/τ | 0.01667 | 0.01667 | ✓ MATCH |
| Stim. factor | stim = 1 - exp(-hν/kT) | 0.9072 | 0.9072 | ✓ MATCH |
| Mean intensity | J_ν = W × B_ν(T) | 1.5717e-05 | 1.5717e-05 | ✓ MATCH |
| Radiative rate | Rate = A_ul × β + B_ul × J × β | 2.8011e+13 | 2.8011e+13 | ✓ MATCH |
| Upward rate | Rate = B_lu × J × β × stim + C_lu | 2.5411e+13 | 2.5411e+13 | ✓ MATCH |

Test case: Si II 6371 Å line at T=9500K, n_e=1e8 cm⁻³, W=0.1, τ=60

---

## 2. Key Parameter Differences

### Collision Rates (TUNABLE)

| Parameter | TARDIS Default | LUMINA Default | Ratio |
|-----------|---------------|----------------|-------|
| Gaunt factor scale | 1.0 | 5.0 | 5× |
| Collision boost | 1.0 | 10.0 | 10× |
| **Combined effect** | 1× | **50×** | 50× |

**To match TARDIS:**
```bash
export MACRO_GAUNT_SCALE=1.0
export MACRO_COLLISIONAL_BOOST=1.0
```

### Thermalization Layer (LUMINA only)

| Parameter | LUMINA Default | To Match TARDIS |
|-----------|---------------|-----------------|
| thermalization_epsilon | 0.35 | 0.0 |
| ir_thermalization_boost | 0.80 | 0.0 |
| uv_scatter_boost | 0.5 | N/A |

**To disable:**
```bash
export MACRO_EPSILON=0.0
export MACRO_IR_THERM=0.0
```

---

## 3. Simulation Results with TARDIS-Matching Parameters

### Command Used
```bash
MACRO_GAUNT_SCALE=1.0 MACRO_COLLISIONAL_BOOST=1.0 MACRO_EPSILON=0.0 MACRO_IR_THERM=0.0 \
./test_integrated atomic/kurucz_cd23_chianti_H_He.h5 5000 /tmp/lumina_tardis_match.csv --type-ia
```

### Results

| Metric | LUMINA Default | LUMINA (TARDIS params) | Improvement |
|--------|----------------|------------------------|-------------|
| T_inner | 5696 K | **13184 K** | Realistic! |
| Escaped | 48% | **68%** | +20% |
| Macro-atom emit | 36% | **100%** | No thermalization |
| Convergence | 12 iter (not converged) | **4 iter (converged)** | Much better |

### Feature Velocities

| Feature | LUMINA | TARDIS | Difference |
|---------|--------|--------|------------|
| Si II 6355 | 15667 km/s | 13657 km/s | +2011 km/s |
| Ca II H&K | 14997 km/s | 15121 km/s | -124 km/s (excellent!) |

### Chi-Square by Region

| Region | Chi-Square | Assessment |
|--------|------------|------------|
| Red (5500-6500 Å) | 31.4 | Good |
| Green (4500-5500 Å) | 74.2 | OK |
| Far-red (6500-7500 Å) | 69.5 | OK |
| NIR (7500-9000 Å) | 70.0 | OK |
| Blue (3500-4500 Å) | 238.1 | Needs work |
| UV (3000-3500 Å) | 271.7 | Needs work |

---

## 4. Algorithm Verification

### Function-by-Function Test Results

```
================================================================================
  FUNCTION 1: Einstein A coefficient
================================================================================
Formula: A_ul = (8π²e²ν²)/(m_e c³) × f_lu × (g_l/g_u)

  A_ul:
    TARDIS: 6.8847693814e+07 s^-1
    LUMINA: 6.8847693814e+07 s^-1
    Rel.Diff: 0.00e+00
    Status: ✓ MATCH

================================================================================
  FUNCTION 3: Sobolev escape probability β
================================================================================
Formula: β = (1 - exp(-τ))/τ

  β(τ=60.0):
    TARDIS: 1.6666666667e-02
    LUMINA: 1.6666666667e-02
    Rel.Diff: 0.00e+00
    Status: ✓ MATCH

================================================================================
  FUNCTION 7: Radiative de-excitation rate (type=-1)
================================================================================
Formula: Rate = A_ul × β + B_ul × J_ν × β

  Rate_radiative:
    TARDIS: 2.8010613555e+13 s^-1
    LUMINA: 2.8010613555e+13 s^-1
    Rel.Diff: 0.00e+00
    Status: ✓ MATCH
```

---

## 5. Conclusion

### Core Algorithm Status: ✓ IDENTICAL

The macro-atom rate calculation algorithms in TARDIS and LUMINA are **mathematically identical**:
- Einstein coefficients (A, B_ul, B_lu)
- Sobolev escape probability β
- Stimulated emission factor
- Mean intensity from Planck function
- Radiative and upward transition rates

### Differences are TUNABLE PARAMETERS

| Category | LUMINA Enhancement | Purpose |
|----------|-------------------|---------|
| Collision boost | 50× higher | More thermalization |
| Epsilon layer | 35% thermalization | Simulate collisional de-excitation |
| IR boost | 80% for λ>7000Å | Stronger IR absorption |

### Recommended Settings

**For TARDIS matching:**
```bash
export MACRO_GAUNT_SCALE=1.0
export MACRO_COLLISIONAL_BOOST=1.0
export MACRO_EPSILON=0.0
export MACRO_IR_THERM=0.0
```

**For better SN Ia fit (tuned):**
```bash
export MACRO_GAUNT_SCALE=2.0
export MACRO_COLLISIONAL_BOOST=5.0
export MACRO_EPSILON=0.2
export MACRO_IR_THERM=0.5
```

---

## 6. Files Created

| File | Description |
|------|-------------|
| `macro_atom_function_comparison.py` | Function-by-function comparison script |
| `detailed_comparison.py` | Spectral feature analysis |
| `tau_fix_comparison.pdf` | Spectrum comparison plot |
| `detailed_spectrum_comparison.pdf` | Feature-by-feature plots |

---

**Last Updated**: 2026-02-01
**Verified**: Core algorithms identical, parameters tunable
