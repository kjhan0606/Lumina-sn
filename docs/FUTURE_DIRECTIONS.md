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

## 3. NLTE Rate Equations — IMPLEMENTED

### Status: Complete (February 2026)

The restricted NLTE solver has been fully implemented in LUMINA, targeting
Si II/III, Ca II/III, Fe II/III, and S II/III (2,017 levels total across
8 ion stages, 36,616 NLTE lines).

### Implementation summary

- **Rate equations**: Statistical equilibrium with radiative BB (Einstein A/B),
  collisional BB (van Regemorter + Axelrod), photoionization (Kramers), and
  recombination (Milne detailed balance)
- **J_nu histogram**: 1000 log-spaced frequency bins (1.5e14–3.0e16 Hz),
  accumulated via atomicAdd on GPU or per-thread accumulation on CPU
- **GPU solver**: cuBLAS batched LU factorization (`cublasDgetrfBatched` +
  `cublasDgetrsBatched`) for all 30 shells simultaneously
- **CPU solver**: Column-oriented Gaussian elimination with partial pivoting
  (cache-friendly for column-major layout), OpenMP-parallel across shells

### Matrix sizes (confirmed from atomic data)

| Ion pair | Levels | Matrix size |
|----------|--------|-------------|
| Fe II+III | 796 + 566 = 1,362 | 1362 x 1362 |
| Si II+III | 100 + 169 = 269 | 269 x 269 |
| Ca II+III | 93 + 150 = 243 | 243 x 243 |
| S II+III | 85 + 58 = 143 | 143 x 143 |

### Measured performance (200K packets, 3 iterations)

| Configuration | Total time | NLTE overhead |
|---------------|-----------|--------------|
| GPU transport, no NLTE | 6.1 s | — |
| GPU transport + cuBLAS NLTE | 9.1 s | +3.0 s |
| CPU+OMP transport + Gauss NLTE | 22.8 s | +15.4 s |

GPU memory for cuBLAS: 425 MB (Fe 1362x1362 x 30 shells, pre-allocated).
At 2M production packets, NLTE adds negligible overhead to GPU transport.

### Usage

```bash
./lumina_cuda data/tardis_reference 200000 20 real nlte   # GPU
./lumina data/tardis_reference 200000 20 real nlte        # CPU
LUMINA_NLTE=1 ./lumina_cuda data/tardis_reference 200000 20  # env var
```

### Files

- `src/lumina_plasma.c`: `nlte_assemble_rate_matrix()`, `nlte_solve_all()`,
  column-oriented `gauss_solve()`
- `src/lumina_cuda.cu`: `CudaNLTESolver`, `cuda_nlte_batched_solve()`,
  `nlte_solve_all_gpu()`, J_nu kernel accumulation
- `src/lumina.h`: `NLTEConfig` struct, function declarations
- `Makefile`: `-lcublas` for CUDA build

---

## Summary

| Direction | Impact | Feasibility | Status |
|-----------|--------|-------------|--------|
| Comoving-frame integration | Low (Sobolev is excellent) | Medium | Not needed |
| 2D/3D geometry | High (asymmetry, polarization) | Hard (major rewrite) | Future |
| Clumping factor | Medium (ionization balance) | Easy (one parameter) | Future |
| **Restricted NLTE on GPU** | **High (UV, ionization)** | **Feasible (Tensor Cores)** | **DONE** |
