# Task Order #072: Complete Redesign — TARDIS-Faithful Reimplementation

## Date: 2026-02-07

## Requirements
After 70+ tasks of incremental fixes, LUMINA still had significant discrepancies vs TARDIS:
- W dilution factor: +72% error (0.65 vs TARDIS 0.38)
- Si II absorption depth: 74% vs TARDIS 93%
- bf/ff continuum opacity proved negligible (Task #071)
- Root causes: frozen macro-atom probabilities, missing MC j_blue estimators, composition differences

User chose **Option D**: Complete redesign from scratch with line-by-line TARDIS matching.

## Implementation Plan
1. **Step 0**: Clean slate — delete all old source, keep git/atomic/model/sn2011fe_data
2. **Phase 1**: Run TARDIS reference, export full converged state (W, T_rad, tau_sobolev, j_blue, spectrum)
3. **Phase 2**: C structures + atomic data loader matching TARDIS exactly
4. **Phase 3**: CPU transport kernel — 1:1 port of TARDIS Python
5. **Phase 4**: Plasma solver + convergence loop
6. **Phase 5**: Main driver + verification
7. **Phase 6**: CUDA GPU port
8. **Phase 7**: Production validation

## Key Design Decisions
- Load TARDIS converged plasma state directly (tau_sobolev, transition_probabilities)
- Every C line has `/* Phase X - Step Y */` traceability comments
- Use xoshiro256** RNG (fast, high quality)
- Unit packet energy: E = 1/n_packets (TARDIS convention)
- time_of_simulation = 1/L_inner (TARDIS normalization)

## TARDIS Reference Values (Ground Truth)
- T_inner: 10521.52 K
- W range: [0.090, 0.380] (decreasing outward)
- T_rad range: [6861, 12291] K
- 137,252 lines, 411,756 macro-atom transitions, 10,791 levels
- Escape fraction: ~40%

## Status
- Phase 1 ✅: TARDIS reference exported to tardis_reference/
- Phase 2 ✅: lumina.h, lumina_atomic.c (NPY/CSV readers)
- Phase 3 ✅: lumina_transport.c (all TARDIS transport functions)
- Phase 4 ✅: lumina_plasma.c (radiation field solver)
- Phase 5 🔧: lumina_main.c (driver, builds and runs, estimator normalization fixed)
- First results: T_rad within 20% (scatter mode), W within 2x
