# TARDIS vs LUMINA Macro-Atom Implementation Comparison

## Key Finding: Critical Differences Identified

### TARDIS Transition Probability Formulas (Lucy 2002, 2003)

**1. Emission Down (radiative de-excitation):**
```
p_emission_down = (2ν²/c²) × (g_lower/g_i) × f_lower→i × β_Sobolev × (ε_i - ε_lower)
```

**2. Internal Down (non-radiative transition to lower level):**
```
p_internal_down = (2ν²/c²) × (g_lower/g_i) × f_lower→i × β_Sobolev × ε_lower
```

**3. Internal Up (excitation to higher level):**
```
p_internal_up = (1/hν) × f_i→upper × β_Sobolev × J_ν^b × (1 - g_i/g_upper × n_upper/n_i) × ε_i
```

Where:
- **β_Sobolev** = (1 - exp(-τ_Sobolev)) / τ_Sobolev
- **J_ν^b** = Blue radiation field intensity at line frequency
- **ε** = Level energy
- **f** = Oscillator strength

---

### LUMINA Current Implementation (macro_atom.c)

**1. Emission Down:**
```c
rate = A_ul;  // + B_ul × J_ν if J_nu available (currently NULL)
```

**2. Internal Down:**
```c
rate = C_ul;  // Pure collisional de-excitation
```

**3. Internal Up:**
```c
rate = C_lu × (g_u/g_l) × exp(-ΔE/kT);  // + B_lu × J_ν if available
```

---

## Critical Missing Components

### 1. β_Sobolev Factor ❌ MISSING
TARDIS multiplies ALL transition probabilities by β_Sobolev:
```
β_Sobolev = (1 - exp(-τ_Sobolev)) / τ_Sobolev
```

This factor:
- Accounts for the optical depth of each line transition
- Ranges from ~1 (optically thin) to ~0 (optically thick)
- Suppresses transitions in optically thick lines
- **We completely ignore this factor**

### 2. J_ν^b Radiation Field ❌ NOT PASSED
We pass `NULL` for J_nu, disabling:
- Stimulated emission enhancement
- Radiative pumping (upward transitions driven by radiation)

TARDIS calculates J_blues from Monte Carlo estimators each iteration.

### 3. Level Energy Scaling ❌ MISSING
TARDIS uses level energies (ε) directly:
- Emission down: proportional to (ε_i - ε_lower)
- Internal down: proportional to ε_lower
- Internal up: proportional to ε_i

We use energy differences for Boltzmann factors but not in the probability scaling.

### 4. Stimulated Emission Correction ❌ MISSING
TARDIS has: `(1 - g_i/g_upper × n_upper/n_i)`

This corrects for population inversion effects.

---

## Impact Analysis

| Missing Component | Expected Effect |
|-------------------|-----------------|
| β_Sobolev | Wrong relative rates - optically thick lines have same weight as thin |
| J_ν^b | No radiative pumping - upward transitions severely underestimated |
| Level energies | Wrong branching ratios between emission channels |
| Stimulated emission | Minor effect in SN conditions |

---

## Downbranch Mode (TARDIS Intermediate)

TARDIS's **downbranch** mode is simpler:
- Only allows **downward** transitions (type = -1)
- No upward internal jumps (no need for J_ν^b)
- Still uses β_Sobolev for escape probability

This could be a good intermediate step to validate our implementation.

---

## Recommendations

### Priority 1: Add β_Sobolev to Transition Probabilities
```c
// In macro_atom_calculate_probabilities()
double tau_sobolev = get_line_tau(line_id, shell);
double beta_sobolev = (1.0 - exp(-tau_sobolev)) / tau_sobolev;
rate *= beta_sobolev;
```

### Priority 2: Implement Downbranch Mode First
- Filter transitions to only type = -1
- Test without needing J_nu radiation field
- Should match TARDIS downbranch results

### Priority 3: Add J_ν^b Calculation
- Compute from MC estimators during temperature iteration
- Pass to macro_atom_calculate_probabilities

### Priority 4: Add Level Energy Scaling
- Multiply emission rates by (ε_upper - ε_lower)
- Multiply internal down by ε_lower

---

## Test Results Summary

### Configuration Comparison (10k packets)

| Mode | χ² | T_inner | L_ratio | Escaped | Runtime |
|------|-----|---------|---------|---------|---------|
| Legacy (no macro-atom) | **106.48** | 11,947 K | 0.97 | 63.2% | 130s |
| Full macro-atom (ε=0.35) | 207.86 | 5,128 K | 0.72 | 45.8% | 129s |
| Full macro-atom (ε=0.02) | 183.30 | 6,330 K | 0.73 | 48.0% | 168s |
| Downbranch (ε=0.10) | 180.94 | 5,649 K | 0.73 | 46.9% | 191s |
| Downbranch (ε=0.02) | 194.99 | 6,560 K | 0.75 | 49.8% | 354s |

### Key Observations

1. **Temperature Collapse**: Macro-atom modes produce T_inner ≈ 5,000-6,500 K vs legacy's 11,947 K
   - This is the root cause of the chi-square gap
   - The temperature iteration doesn't converge

2. **Energy Balance**: More packets are absorbed in macro-atom mode (50-55%) vs legacy (37%)
   - This causes the luminosity ratio to stay below target

3. **Downbranch ≈ Full Macro-atom**: Little difference between modes suggests the cascade isn't the issue
   - Both produce similar χ² (180-195) and T_inner (5,600-6,600 K)

4. **Thermalization Paradox**: Lower ε gives higher T_inner but worse χ²
   - Some thermalization is needed for spectral shape
   - The energy balance and spectral shape have competing requirements

### Root Cause Analysis

The fundamental difference is likely **β_Sobolev factor missing**:
- TARDIS multiplies ALL transition rates by β = (1-e^(-τ))/τ
- This suppresses transitions in optically thick lines
- Without it, we over-scatter in strong lines, changing the energy distribution

### Next Steps to Match TARDIS

1. **Implement β_Sobolev weighting** (requires passing tau arrays)
2. **Calculate J_nu from MC estimators** (for stimulated emission)
3. **Debug why temperature iteration fails** to converge in macro-atom mode
4. **Compare spectral features** not just chi-square

---

## Conclusion

**TARDIS uses macroatom mode** as their default and recommended line interaction treatment. Our macro-atom implementation is structurally correct but missing critical physics:

1. **β_Sobolev factor** - The most important missing piece
2. **J_ν radiation field** - Needed for stimulated processes
3. **Level energy scaling** - Affects branching ratios

**Current Recommendation**: Use legacy mode (`LUMINA_MACRO_ATOM=0`) for production runs until β_Sobolev is implemented. Legacy mode achieves χ²=106 vs macro-atom's χ²≈180-200.

### References

- Lucy, L. B. 2002, A&A, 384, 725 - "Monte Carlo transition probabilities"
- Lucy, L. B. 2003, A&A, 403, 261 - "Monte Carlo transition probabilities II"
- Kerzendorf & Sim 2014, MNRAS, 440, 387 - TARDIS paper
- TARDIS Documentation: https://tardis-sn.github.io/tardis/
