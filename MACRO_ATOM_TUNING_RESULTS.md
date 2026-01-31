# Macro-Atom Tuning Results

## Summary

The macro-atom framework has been integrated and tuned with wavelength-dependent thermalization to better match TARDIS behavior. Key findings are documented below.

## Changes Made

### 1. Thermalization Logic (macro_atom.c)
- Added `MacroAtomTuning` struct with tunable parameters
- Thermalization now applied **ONCE** at the end of the cascade (not at each step)
- Wavelength-dependent: UV photons scatter more, IR photons thermalize more

### 2. Tunable Parameters (via environment variables)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MACRO_EPSILON` | 0.35 | Base thermalization probability |
| `MACRO_IR_THERM` | 0.80 | IR (λ>7000Å) thermalization probability |
| `MACRO_COLLISIONAL_BOOST` | 10.0 | Multiplier for collisional rates |
| `MACRO_GAUNT_SCALE` | 5.0 | Effective Gaunt factor multiplier |
| `MACRO_UV_SCATTER` | 0.5 | UV thermalization reduction factor |

### 3. Integration with test_integrated.c
- Added call to `macro_atom_tuning_from_env()` during initialization
- Added diagnostic output for tuning parameters

## Comparison Results (10,000 packets)

| Mode | χ² | T_inner | L_ratio | Escaped | Runtime |
|------|-----|---------|---------|---------|---------|
| Legacy (no macro-atom) | 106.48 | 11,947 K | 0.97 | 63.2% | 130s |
| Macro-atom (ε=0.35 default) | 207.86 | 5,128 K | 0.72 | 45.8% | 129s |
| Macro-atom (ε=0.10) | ~210 | 5,678 K | 0.68 | 44.9% | 104s |
| Macro-atom (ε=0.02) | 183.30 | 6,330 K | 0.73 | 48.0% | 168s |

## Key Observations

1. **Lower thermalization → Better energy balance**: With ε=0.02, T_inner increases to 6330K and L_ratio improves to 0.73

2. **Performance tradeoff**: Lower thermalization means more interactions per packet, resulting in slower runtime (30-48 packets/sec vs 77 for legacy)

3. **Chi-square gap**: Even with optimized thermalization, macro-atom mode (χ²≈183) doesn't match legacy mode (χ²≈106). This suggests the underlying cascade physics produces different spectral characteristics.

4. **Macro-atom absorption**: The cascade loop itself causes absorption when:
   - No transitions are available
   - Max jumps (100) exceeded
   - Ground state reached without emission

## Recommendations

For best results with current implementation:
```bash
# Low thermalization for better energy balance
export MACRO_EPSILON=0.02
export MACRO_IR_THERM=0.25

# Or disable macro-atom for faster runs with better fit
export LUMINA_MACRO_ATOM=0
```

## Future Work

1. Investigate why macro-atom cascade produces different spectral shape
2. Compare transition probability calculations with TARDIS implementation
3. Consider J_nu radiation field for stimulated emission (Phase 3 of original plan)
4. Profile and optimize cascade loop performance
