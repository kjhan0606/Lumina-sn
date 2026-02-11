# LUMINA-SN: Future Directions

Potential physics upgrades beyond the current TARDIS-equivalent implementation.
Prioritized by scientific impact and feasibility.

---

## 1. Comoving-Frame Direct Integration (vs Sobolev)

### Current approach

LUMINA uses the Sobolev approximation with a full Doppler sweep across each
shell (Task #067).  The cumulative-tau method sums optical depths from all lines
a photon encounters as it sweeps through the comoving-frame frequency range,
which correctly handles **line overlap**.

### Assessment: Not needed for photospheric-phase SN Ia

The Sobolev approximation is valid when the expansion velocity gradient
dominates thermal broadening:

```
v_expansion ~ 10,000 -- 30,000 km/s
v_thermal   ~     10 km/s       (T ~ 10,000 K, Fe-group)
v / v_th    ~ 1,000
```

The approximation error is O(v_th / v) ~ 0.1%, far below Monte Carlo noise at
any practical packet count.

Direct comoving-frame integration would require resolving individual line
profiles at ~0.1 A resolution, yielding ~70,000 frequency bins across
3000--10000 A.  This increases per-packet cost by roughly **1000x** for
negligible accuracy improvement in photospheric-phase SN Ia.

### When it would matter

| Regime | v / v_th | Sobolev valid? |
|--------|----------|---------------|
| SN Ia photospheric (t ~ 20d) | ~1000 | Yes |
| SN Ia nebular (t > 100d) | ~100 | Marginal |
| Dense stellar winds | ~10--100 | No |
| Static atmospheres | ~1 | No |

**Conclusion**: If LUMINA is extended to the **nebular phase** (t > 100 days),
a hybrid approach — Sobolev in outer shells, direct integration in inner
low-velocity shells — would be appropriate.  For B-max fitting, the current
implementation is optimal.

---

## 2. 2D/3D Geometry and Clumping

### Current approach

LUMINA assumes 1D spherical symmetry with homologous expansion (v = r/t).
This is the same assumption used by TARDIS and most fast SN Ia codes.

### Scientific motivation

- **Asymmetric explosions**: Delayed detonation and violent merger models
  predict large-scale asymmetries in ejecta composition.
- **High-velocity features (HVFs)**: Detached Si II components at
  v > 20,000 km/s may arise from off-center ignition or viewing-angle effects.
- **Polarization**: Spectropolarimetry of SN Ia shows 0.2--0.7% continuum
  polarization, requiring departure from spherical symmetry.
- **Clumping**: Small-scale density inhomogeneities enhance recombination rates
  and modify ionization balance.

### Implementation path

1. **2D axisymmetric**: Add polar angle dependence; shells become annuli.
   Packet propagation gains one angle coordinate.  Cost ~ 10x over 1D.
2. **Clumping factor**: Volume filling factor f_V < 1 modifies effective
   density: rho_eff = rho / f_V within clumps.  Minimal code change,
   significant physics impact on ionization balance.
3. **Full 3D**: Cartesian or adaptive mesh.  Cost ~ 100--1000x over 1D.
   Required for comparison with 3D explosion models (e.g., Seitenzahl et al.).

### Priority

Medium-term.  Clumping factor is low-cost and could be added as a fitting
parameter.  Full 2D/3D is a major architectural change, better suited as a
separate long-term project.

---

## 3. NLTE Rate Equations (Restricted)

### Current approach

LUMINA uses the same ionization/excitation scheme as TARDIS:

- **Ionization**: Nebular approximation (modified Saha with dilution factor W
  and radiation temperature T_rad)
- **Excitation**: Boltzmann distribution at T_rad for all levels
- **Source function**: Macro-atom formalism (an implicit form of NLTE for the
  radiation field, handling fluorescence and downbranching)

This produces W error < 1% and T_rad error < 0.3% vs TARDIS at 200K packets.

### What full NLTE means

Statistical equilibrium for every level of every ion:

```
For each level i:

    sum_j [ n_j * R(j->i) ] = n_i * sum_j [ R(i->j) ]

where R(i->j) includes:
    - Spontaneous emission:     A_ij
    - Stimulated emission/abs:  B_ij * J_ij
    - Collisional transitions:  C_ij(n_e, T_e)
    - Photoionization:          integral[ 4*pi*J_nu * sigma_bf(nu) / (h*nu) d_nu ]
    - Recombination:            alpha_i(T_e) * n_e
```

This yields a linear system **A x = b** where x is the vector of level
populations, and A is the rate matrix.

### Restricted NLTE: target ions only

Full NLTE for all 10,000+ levels is unnecessary.  A **restricted** approach
targeting key diagnostic ions captures the dominant departures from LTE:

| Ion | Levels | Why |
|-----|--------|-----|
| Si II / III / IV | ~80 | Classification feature; ionization balance sets 6355 A depth |
| Ca II / III | ~60 | Strong H&K (3934, 3968 A) and IR triplet (8498, 8542, 8662 A) |
| Fe II / III | ~300 | Dominant line blanketing; UV flux redistribution |
| S II / III | ~50 | "W" feature diagnostic (5454, 5640 A) |
| **Total** | **~500** | Tractable matrix size |

Remaining species (C, O, Mg, Ti, Cr, Co, Ni) stay on the nebular
approximation — their NLTE corrections are small in the photospheric phase.

### Expected accuracy gains

| Spectral feature | Current accuracy | With restricted NLTE |
|-----------------|------------------|---------------------|
| Si II 6355 A depth | Good (macro-atom helps) | Improved ionization balance |
| Ca II H&K | Good | Better emission-to-absorption ratio |
| UV flux (< 3500 A) | Poor | **Significantly improved** |
| Fe-group blanketing | Good (fluorescence works) | More accurate redistribution |
| S II "W" shape | Moderate | Improved |

### GPU acceleration with Tensor Cores

The rate matrix solve is a dense linear algebra problem — exactly what GPU
Tensor Cores are designed for.

**Matrix dimensions and cost:**

```
Restricted NLTE:  ~500 x 500 matrix per shell
30 shells:        batched solve of 30 matrices simultaneously

LU decomposition: O(N^3) = O(125M) FLOPs per shell
Total:            30 x 125M = 3.75 GFLOP

RTX 5000 Ada (sm_89):
    FP32 Tensor:  ~50 TFLOP/s
    FP64:         ~1.6 TFLOP/s  (sufficient for this problem)
    Batched LU:   cuSOLVER cublasDgetrfBatched()

Estimated time:   < 1 ms for all 30 shells (batched)
```

For comparison, the Monte Carlo transport kernel takes ~700 ms for 200K packets.
The NLTE matrix solve would add **< 0.2%** overhead per iteration.

**Implementation with CUDA:**

```
Per MC iteration:
    1. Transport kernel (existing):  ~700 ms
       -> produces J_nu estimators per shell
    2. Rate matrix assembly:         ~1 ms
       -> compute R(i->j) from J_nu, n_e, T_e for each shell
    3. Batched LU solve:             < 1 ms
       -> cuSOLVER: 30 x (500 x 500) simultaneous solves on GPU
       -> yields new level populations n_i per shell
    4. Update tau_sobolev:           ~1 ms
       -> recompute line opacities from new populations
    5. Repeat until convergence
```

**Key advantage over CPU NLTE codes**: CMFGEN and PHOENIX solve NLTE on CPU,
taking hours per model.  With GPU batched solves, LUMINA could do restricted
NLTE at essentially the same speed as the current nebular approximation.

### Data requirements

All required atomic data is already in `kurucz_cd23_chianti_H_He.h5`:

| Data | Status |
|------|--------|
| Energy levels (E_i) | Available (10,770 levels loaded) |
| Radiative rates (A_ij) | Available (transition_probabilities) |
| Line wavelengths | Available (271,741 lines) |
| Photoionization cross-sections | Available (phot_data in HDF5) |
| Collisional rates | Need to compute: van Regemorter (permitted), Axelrod (forbidden) |

Collisional rate approximations:

- **Permitted transitions**: van Regemorter (1962) formula using oscillator
  strengths (already in atomic data)
- **Forbidden transitions**: Axelrod (1980) approximation, C_ij ~ 8.6e-6 *
  Omega / (g_i * T_e^0.5), with effective collision strength Omega ~ 1
- These approximations are standard in SN Ia codes and sufficient for
  photospheric conditions

### Implementation phases

**Phase A**: Infrastructure (2-3 weeks)
- Define NLTE level subset for Si, Ca, Fe, S
- Build rate matrix assembly on GPU (using existing atomic data)
- Implement cuSOLVER batched LU
- Validate: reproduce nebular approximation as limiting case (W -> 0.5, T_rad -> T_bb)

**Phase B**: J_nu coupling (1-2 weeks)
- Extract frequency-binned J_nu from MC estimators (j_estimator / nu_bar)
- Compute photoionization and stimulated emission rates from J_nu
- Close the loop: MC transport <-> NLTE populations <-> tau_sobolev

**Phase C**: Validation and tuning (2-3 weeks)
- Compare Si II/III ionization fractions vs TARDIS (nebular approx.)
- Compare vs published CMFGEN/PHOENIX results for SN Ia models
- Quantify UV spectrum improvement
- Add NLTE corrections to ML training pipeline

### Priority

**High scientific value, feasible with GPU.**  This is the most impactful
physics upgrade LUMINA can make.  The GPU Tensor Core approach makes it
uniquely fast compared to existing NLTE codes.

Recommended timeline: after the ML pipeline (Stages 1-3) is validated and
producing results.

---

## Summary

| Direction | Impact | Feasibility | Priority |
|-----------|--------|-------------|----------|
| Comoving-frame integration | Low (Sobolev is excellent) | Medium | Low — only for nebular phase |
| 2D/3D geometry | High (asymmetry, polarization) | Hard (major rewrite) | Long-term |
| Clumping factor | Medium (ionization balance) | Easy (one parameter) | Medium-term |
| **Restricted NLTE on GPU** | **High (UV, ionization)** | **Feasible (Tensor Cores)** | **Next major upgrade** |
