# TARDIS vs LUMINA Macro-Atom Implementation Comparison

## Implementation Status

### ✅ Completed

1. **TARDIS Python reference script** (`tardis_macro_atom_debug.py`)
   - Calculates expected transition probabilities
   - Uses TARDIS formulas: Rate = A_ul × β + B_ul × J_ν × β
   - Outputs debug format matching LUMINA

2. **Macro-atom reference table building** (`atomic_loader.c`)
   - Builds transitions from line data: 815,229 transitions
   - 22,356 unique levels with transitions
   - Fixed hash collision bug in level key calculation

3. **TARDIS-style rate calculation** (`macro_atom.c`)
   - Radiative down (type=-1): Rate = A_ul × β + B_ul × J_ν × β
   - Collisional down (type=0): Rate = C_ul (van Regemorter)
   - Internal up (type=1): Rate = B_lu × J_ν × β × stim_factor + C_lu

4. **Dilution factor integration**
   - W passed to macro_atom_init and calculate_probabilities
   - J_ν calculated from diluted Planck: J_ν = W × B_ν(T)

5. **Debug output matching TARDIS format**
   - Activation info with τ, β, W
   - Transition probability table
   - Selection trace with cumulative probabilities

6. **Sobolev escape probability β** (NEWLY COMPLETED)
   - Formula implemented: β = (1 - exp(-τ)) / τ
   - tau_sobolev array now built from ShellState.active_lines
   - Passed to macro_atom_process_line_interaction via `build_tau_sobolev_array()`
   - Per-shell caching for efficiency

### 🔲 TODO

1. **Validate against TARDIS output**
   - Run same configuration in both codes
   - Compare transition probabilities numerically

2. **Fine-tune chi-square match**
   - Current χ² vs TARDIS: compare after tau fix
   - Si II 6355 velocity should match ~10,500 km/s

## Key Formulas Comparison

### TARDIS (Python)
```python
# transition_probabilities.py
transition_probabilities[i, j] = transition_probability_coef[i] * beta_sobolev[line_idx, j]

# For upward transitions (type=1):
if transition_type[i] == 1:
    transition_probabilities[i, j] *= stimulated_emission_factor[line_idx, j] * j_blues[line_idx, j]
```

### LUMINA (C)
```c
// macro_atom.c calculate_probabilities()

// Radiative down (type=-1):
rate = line->A_ul * beta;
rate += B_ul * J_line * beta;

// Internal up (type=1):
rate = C_ul * g_ratio * exp(-delta_E / kT);  // collisional
rate += B_lu * J_line * beta * stim_factor;  // radiative absorption
```

## Debug Output Comparison

### TARDIS Reference (Si II 6355 Å)
```
╔══════════════════════════════════════════════════════════════════╗
║ TARDIS MACRO-ATOM INTERACTION #1
╠══════════════════════════════════════════════════════════════════╣
║ ACTIVATION:
║   Absorbed line_id=271742, λ=6371.4 Å, ν=4.7053e+14 Hz
║   Line: Z=14, ion=1, lower=1 → upper=5
║   Activation level: Z=14, ion=1, level=5
║   T=9500.0 K, n_e=1.000e+08 cm^-3
║   τ_Sobolev=6.0000e+01, β=0.016667
╠══════════════════════════════════════════════════════════════════╣
[TARDIS DEBUG] calculate_probabilities for Z=14 ion=1 level=5
  T=9500.0 K, n_e=1.000e+08 cm^-3, W=0.1000
  total_rate=4.217600e+13, n_transitions=4
  p_emission=1.0000 (100.0% radiative)
  Transition probabilities:
    idx  dest_lvl   type   rate         prob
    0    0          -1     1.4165e+13   0.335864
    1    1          -1     2.8011e+13   0.664136
    2    0          0      5.5428e-02   0.000000
    3    1          0      3.3229e-02   0.000000
```

### LUMINA Current Output (WITH tau_sobolev FIX)
```
╔══════════════════════════════════════════════════════════════════╗
║ LUMINA MACRO-ATOM INTERACTION #3
╠══════════════════════════════════════════════════════════════════╣
║ ACTIVATION:
║   Absorbed line_id=271742, λ=6371.4 Å, ν=4.7053e+14 Hz
║   Line: Z=14, ion=1, lower=1 → upper=5
║   Activation level: Z=14, ion=1, level=5
║   T=8878.7 K, n_e=6.968e+07 cm^-3, W=0.5000
║   τ_Sobolev=1.0000e+03, β=0.001000   ← FIXED: tau now passed!
╠══════════════════════════════════════════════════════════════════╣
║ TRANSITION LOOP:
╠══════════════════════════════════════════════════════════════════╣
║ RESULT: EMITTED
║   Emission line_id=39241, λ=6373.1 Å, ν=4.7040e+14 Hz
║   n_jumps=1, wavelength shift: 6371.4 → 6373.1 Å
╚══════════════════════════════════════════════════════════════════╝
```

### Other Examples (showing β factor working correctly)
```
# Moderate optical depth:
║   τ_Sobolev=1.2305e+00, β=0.575259   (Mg II 4482Å)

# Optically thick:
║   τ_Sobolev=1.7931e+02, β=0.005577   (Ca II 3738Å)

# Very optically thick (capped):
║   τ_Sobolev=1.0000e+03, β=0.001000   (Si II 6371Å)
```

## Key Parameters Status

| Parameter | TARDIS | LUMINA Status |
|-----------|--------|---------------|
| β (Sobolev) | Uses τ from transport | ✅ Now uses τ from active_lines |
| W (dilution) | From plasma state | ✅ Calculated correctly |
| J_ν | From Monte Carlo estimator | ✅ Diluted Planck |
| Collisions | Detailed rates | ✅ van Regemorter |

## Files Modified

1. `macro_atom.h`: Updated function signatures with W and tau_sobolev
2. `macro_atom.c`: TARDIS-style rate calculation with β and J_ν
3. `atomic_loader.c`: Build macro_atom_references from line data
4. `test_integrated.c`:
   - Added `build_tau_sobolev_array()` helper function
   - Passes tau array to macro_atom_process_line_interaction
   - Added cleanup with `free_tau_sobolev_cache()`
5. `tardis_macro_atom_debug.py`: Reference implementation for comparison

## Running the Comparison

### TARDIS Reference
```bash
python3 tardis_macro_atom_debug.py
```

### LUMINA Debug
```bash
MACRO_ATOM_DEBUG=1 MACRO_ATOM_DEBUG_MAX=5 ./test_integrated \
    atomic/kurucz_cd23_chianti_H_He.h5 100 /tmp/test.csv --type-ia
```

## Implementation Details

### tau_sobolev Array Building (`test_integrated.c`)

The `build_tau_sobolev_array()` function creates a lookup array from the shell's active_lines:

```c
static double *build_tau_sobolev_array(
    const ShellState *shell,
    int64_t n_lines,
    int shell_idx)
{
    // Allocate array for all lines (indexed by line_id)
    // Fill in tau values from shell->active_lines[]
    // Lines not in active_lines default to tau=0 (β=1)
    // Per-shell caching for efficiency
}
```

### Sobolev Escape Probability (`macro_atom.c`)

```c
static double calculate_beta_sobolev(double tau) {
    if (tau < 1e-6) return 1.0;              // Optically thin limit
    else if (tau > 500.0) return 1.0 / tau;  // Asymptotic limit
    return (1.0 - exp(-tau)) / tau;          // Standard formula
}
```

## Next Steps

1. Run full comparison with TARDIS output
2. Compare Si II 6355 Å transition probabilities numerically
3. Tune parameters to match spectrum fit
4. Verify chi-square improvement after tau fix
