# LUMINA-SN Development History

This document records the complete development history of LUMINA-SN, a C/CUDA Monte Carlo radiative transfer code for Type Ia supernovae, designed to reproduce TARDIS results with GPU acceleration.

---

## 1. Complete Redesign (Task #072)

### Task Order #072: Complete Redesign -- TARDIS-Faithful Reimplementation

#### Date: 2026-02-07

#### Requirements
After 70+ tasks of incremental fixes, LUMINA still had significant discrepancies vs TARDIS:
- W dilution factor: +72% error (0.65 vs TARDIS 0.38)
- Si II absorption depth: 74% vs TARDIS 93%
- bf/ff continuum opacity proved negligible (Task #071)
- Root causes: frozen macro-atom probabilities, missing MC j_blue estimators, composition differences

User chose **Option D**: Complete redesign from scratch with line-by-line TARDIS matching.

#### Implementation Plan
1. **Step 0**: Clean slate -- delete all old source, keep git/atomic/model/sn2011fe_data
2. **Phase 1**: Run TARDIS reference, export full converged state (W, T_rad, tau_sobolev, j_blue, spectrum)
3. **Phase 2**: C structures + atomic data loader matching TARDIS exactly
4. **Phase 3**: CPU transport kernel -- 1:1 port of TARDIS Python
5. **Phase 4**: Plasma solver + convergence loop
6. **Phase 5**: Main driver + verification
7. **Phase 6**: CUDA GPU port
8. **Phase 7**: Production validation

#### Key Design Decisions
- Load TARDIS converged plasma state directly (tau_sobolev, transition_probabilities)
- Every C line has `/* Phase X - Step Y */` traceability comments
- Use xoshiro256** RNG (fast, high quality)
- Unit packet energy: E = 1/n_packets (TARDIS convention)
- time_of_simulation = 1/L_inner (TARDIS normalization)

#### TARDIS Reference Values (Ground Truth)
- T_inner: 10521.52 K
- W range: [0.090, 0.380] (decreasing outward)
- T_rad range: [6861, 12291] K
- 137,252 lines, 411,756 macro-atom transitions, 10,791 levels
- Escape fraction: ~40%

#### Status
- Phase 1: TARDIS reference exported to tardis_reference/
- Phase 2: lumina.h, lumina_atomic.c (NPY/CSV readers)
- Phase 3: lumina_transport.c (all TARDIS transport functions)
- Phase 4: lumina_plasma.c (radiation field solver)
- Phase 5: lumina_main.c (driver, builds and runs, estimator normalization fixed)
- Phase 6: lumina_cuda.cu (CUDA GPU transport kernel, 10x speedup)
- First results: T_rad within 20% (scatter mode), W within 2x
- After convergence fixes: W error 0.67%, T_rad error 0.28%, T_inner error 0.11%

---

## 2. Key Bug Fix History

The following is a chronological summary of the major bugs discovered and fixed during LUMINA-SN development. Each bug had a substantial impact on simulation accuracy and required careful diagnosis.

### a. Task #013: CUDA Transport Bugs (4 Critical Bugs)

**Bug 1: CUDA_MAX_BLOCKS = 1024**
The block count was hard-coded to 1024. With 256 threads per block, only 262,144 packets could run. For a 2M-packet simulation, only 13.1% of packets actually executed. For 20M packets, only 1.3%. This caused the j_estimator to be 76x too low, producing W values around 0.005 instead of the expected 0.5. The fix was to increase CUDA_MAX_BLOCKS to 131,072 (the GPU supports up to 2^31 blocks).

**Bug 2: Shared Memory ShellCache Race Condition**
A `__shared__ ShellCache` structure was shared by all 256 threads in a block. Thread 0's shell data was used by all threads, even those processing packets in different shells. Additionally, divergent `__syncthreads()` calls inside conditional branches produced undefined CUDA behavior. The fix was to remove shared memory entirely and let each thread read from global memory (which is L1 cached).

**Bug 3: Counter Reset Missing Between Iterations**
The device counters `d_n_escaped`, `d_n_absorbed`, and `d_n_scattered` accumulated across all iterations without being reset. The escape fraction was therefore calculated as `cumulative_escapes / n_packets_iter`, which grew wildly wrong over iterations. The T_inner update used this incorrect escape fraction, causing divergent T_inner. The fix was to reset all counters in `cuda_reset_estimators()`.

**Bug 4: Boundary Sticking**
`find_shell_idx()` was called every step. Floating-point arithmetic could land packets at exact shell boundaries, causing ambiguous shell assignment. The fix was to track shells explicitly with `shell_id += delta_shell` on boundary crossing, following the TARDIS approach.

**Results after all fixes:**
- W(shell 0): 0.005 --> 0.500 (100x improvement)
- Escape fraction: 1.2% --> 76.6%
- T_inner: oscillating 7000-14600 K --> converged to 8949 K

### b. Task #024: Shell-Tracking d_min Bug

**The Bug:** A sanity check `d_min <= 0.0` was intended to catch invalid distances, but it also triggered on legitimate cases where d_line = 0 (a line at the exact entry frequency of the Sobolev sweep). When triggered, it replaced d_min with `(r_outer - r) * 0.1` and forced a boundary crossing interaction. This moved the packet partway through a shell but assigned it to the next shell.

**The Cascade:** Every subsequent step used wrong shell properties, wrong boundary distances, and cascading errors. Packets ended up at 44x beyond the outer boundary with beta > 1 (superluminal velocities) and negative energy/frequency.

**The Fix:** Changed `d_min <= 0.0` to `d_min < 0.0` (allowing d_min = 0 as a valid line interaction). Added a boundary nudge of `shell_width * 1e-10` after each boundary crossing to prevent floating-point ambiguity.

**Diagnostic method:** Added an `[OVERSHOOT]` check after the interaction switch that flagged packets outside their current shell's [r_inner, r_outer] range. This immediately showed all 10K packets overshooting at step 0.

**Results after fix:**
- W collapse eliminated: W(shell 29) went from 0.0002 --> 0.500
- Escape fraction: 72% --> 43.7% (TARDIS: 24%)
- nu_bar/j estimators clean: monotonically decreasing, no rogue values

### c. Task #063: Iron Line Blanketing Discovery

**The Problem:** When iron-group elements (Fe, Ni, S) were used as abundance fillers, they created massive line blanketing with 1.08 million active lines compared to only 120,000 with oxygen as the filler. Iron in the core (61% mass fraction) created a pseudo-photosphere at roughly 11,000 km/s that masked high-velocity Si II features.

**The Breakthrough:** Oxygen is the correct filler element. Oxygen has very few optical lines and is effectively transparent in the wavelength range of interest. Switching from iron to oxygen fillers reduced the velocity gap between TARDIS and LUMINA from 15,240 km/s to just 371 km/s. This was the single largest improvement in the entire development history.

### d. Task #067: Sobolev Sweep Repair

**The Problem:** The old `find_next_line` function used a fixed +/-1% frequency window to search for line interactions. In homologous expansion, the comoving-frame frequency nu_cmf(s) = nu_lab * (1 - (r0*mu0 + s)/(c*t_exp)) is linear in propagation distance s. The frequency sweep across thick shells far exceeds +/-1%, so most lines were invisible to photons.

**The Fix:** Implemented a full Doppler sweep from the entry frequency to the exit frequency of each shell, with cumulative optical depth accumulation along the path.

**Results after fix:**
- Line interactions: 0.6% --> massive (3.68M scatters for 20M packets)
- Escaped fraction: 97% --> 24.6%
- Si II absorption depth: 20% --> 45% (TARDIS = 93%)
- Velocity gap: +4,012 km/s (LUMINA 30.3k vs TARDIS 26.2k km/s)
- Runtime: 33s for 20M packets (faster due to fewer shells crossed per packet)

**Remaining gap (45% vs 93%)** was confirmed as a source function issue, not an opacity issue (Tasks #068-069).

### e. Task #070: GPU Downbranch/Fluorescence

**The Problem:** The CUDA kernel had the full downbranch table uploaded to GPU memory (7.4M entries) but never actually used it. All line interactions were pure resonant scattering -- photons were re-emitted at the same frequency they were absorbed, preventing the formation of deep P-Cygni profiles.

**The Fix:** Added an `emission_nu` field to `CudaDownbranchData` containing the rest-frame frequency for each emission candidate. Implemented binary search on cumulative branching probabilities to sample the emission line. Applied wavelength-dependent thermalization: epsilon = 0.80 (IR), 0.35 (optical), 0.175 (UV).

**Results after fix:**
- Si II absorption depth: 45% --> 74.4% (blue-side), 82.8% (vs red peak)
- Escaped: 23.1% (was 24.6%)
- Absorbed: 3.2% (was 1.6%)
- Scattered: 3.14M interactions (was 3.68M)
- Trough minimum at 6473 A (fluorescence notch at rest wavelength)

### f. Task #072: T_inner Convergence Fix (4 Bugs)

**Bug 1: Missing L_requested.** The T_inner update formula used `(1/escape_fraction)^0.25` instead of the correct `(L_requested/L_emitted)^0.25`. Without the target luminosity in the formula, T_inner could not converge to the physically correct value.

**Bug 2: Wrong exponent.** TARDIS uses `(L_emitted/L_requested)^(-0.5)`, not `^(0.25)`. The -0.5 exponent accounts for non-linear radiative transfer effects and was confirmed from the TARDIS source code (`base.py:estimate_t_inner`).

**Bug 3: No W/T_rad damping.** TARDIS damps all convergence quantities: `new = old + 0.5 * (estimated - old)`. LUMINA had direct replacement, which caused oscillations and prevented convergence.

**Bug 4: Wrong L_emitted approximation.** L_emitted was approximated as `L_inner * escape_fraction`. TARDIS sums the actual energies of escaped packets. The approximation introduced systematic errors especially at low escape fractions.

**Results after all four fixes (20 iterations, 200K packets):**
- W error: 0.67%
- T_rad error: 0.28%
- T_inner error: 0.11%

---

## 3. Timeline: Redesign Phases

The complete redesign (Task #072) proceeded through the following phases:

**Phase 1: TARDIS Reference Export**
Ran a fully converged TARDIS simulation of SN 2011fe and exported the complete plasma state to `tardis_reference/`. This included W (dilution factor), T_rad (radiation temperature), tau_sobolev (Sobolev optical depths), j_blue (mean intensity at line frequencies), transition probabilities, and the converged spectrum. These served as ground-truth values for all subsequent validation.

**Phase 2: C Structures + Atomic Data Loader**
Implemented `lumina.h` (header with all data structures) and `lumina_atomic.c` (NPY/CSV readers for atomic data). The data structures were designed to match TARDIS memory layout exactly: 137,252 lines, 411,756 macro-atom transitions, and 10,791 levels. Handled a critical CSV parsing bug where `strtok()` skipped leading delimiters in macro_atom_data.csv, shifting all column indices by 1.

**Phase 3: CPU Transport Kernel**
Implemented `lumina_transport.c` as a line-by-line port of TARDIS Python transport. Key components: full Sobolev sweep (entry-to-exit frequency), cumulative optical depth, macro-atom cascade with downbranching, and xoshiro256** RNG. This was a 1:1 faithful reproduction of TARDIS `montecarlo_main_loop`.

**Phase 4: Plasma Solver**
Implemented `lumina_plasma.c` with the radiation field update equations. Includes W and T_rad estimation from MC estimators (j and nu_bar), partition function calculation (using T_rad for all levels, matching TARDIS), Saha ionization balance, damped n_e iteration (50% damping, 5% convergence threshold), and tau_sobolev recalculation. Fixed a critical bug where T_e was used for metastable levels instead of T_rad.

**Phase 5: Main Driver + Verification**
Implemented `lumina_main.c` tying all components together. Multi-iteration convergence loop with T_inner update using the corrected TARDIS formula. Verified against TARDIS reference: W error under 1%, T_rad error under 0.5%, T_inner error under 0.2% at 200K packets over 20 iterations. Build: `make` (serial) or `make OMP=1` (OpenMP parallel).

**Phase 6: CUDA GPU Port (10x Speedup)**
Implemented `lumina_cuda.cu` (~850 lines) containing all device functions, the transport kernel, and the host driver in a single file. Key design: xoshiro256** RNG on GPU (4 uint64 state per thread), atomicAdd for j/nu_bar estimators, j_blue/Edotlu estimators handled by CPU plasma solver. Performance: 200K packets in 728ms on GPU (3.6 us/packet) vs 7.2s on CPU (36 us/packet) -- approximately 10x speedup. Target GPU: NVIDIA RTX 5000 Ada (sm_89), CUDA 13.0. W error 1.06%, T_rad error 0.58% -- matching CPU within statistical noise. Build: `make cuda`.

**Phase 6 Production Validation: 2M Packets x 20 Iterations (CUDA)**
Full production run with 2M packets per iteration, 20 iterations, virtual packet spectrum. Results:
- Mean |W error|: 0.44% across all 30 shells
- Mean |T_rad error|: 0.36% across all 30 shells
- T_inner final: 10564.38 K (TARDIS: 10521.52 K, error: 0.41%)
- L_emitted converged to L_requested within 0.04%
- GPU kernel: ~240s/iteration (120 us/packet for 2M packets)
- Escape fraction: 40.76% (stable across iterations 8-20)

---

## 4. SN 2011fe Parameter Fitting

### Date: 2026-02-08

### Motivation

With the LUMINA transport engine validated to sub-1% accuracy against TARDIS (W error 0.44%, T_rad error 0.36%), the next step was to demonstrate the code's utility for actual science: fitting observed supernova spectra. SN 2011fe (the best-observed Type Ia supernova) was chosen as the test case, using the Pereira+2013 spectrum at phase -0.3 days from B-maximum.

### Implementation

Created `scripts/fit_parameter_search.py` (~430 lines of Python) implementing a 5-dimensional coarse-to-fine parameter search:

**Parameters**: log_L (luminosity), v_inner (photosphere velocity), log_rho_0 (reference density), X_Si (silicon fraction), X_Fe (iron fraction). Oxygen fills the remaining mass fraction. Fixed: S=0.05, Ca=0.03, Co=0.05, Ni=0.10, C=0.02. Density power law exponent: -7.

**Method**: For each parameter set, the script:
1. Creates a temporary reference directory with symlinks to unchanged atomic data
2. Generates modified config.json, geometry.csv, density.csv, abundances.csv, electron_densities.csv, and plasma_state.csv
3. Runs the LUMINA CPU binary with the "rotation" spectrum mode
4. Computes normalized RMS vs the observed SN 2011fe spectrum (3500-9000 A, 5 A grid)
5. Measures the Si II 6355 trough depth and absorption velocity

**Sampling**: Latin Hypercube Sampling for uniform 5D coverage.

### Results

Total runtime: 30.2 minutes on CPU for all three phases (100 + 20 + 3 = 123 LUMINA runs).

**Phase 1 (Coarse, 20K packets x 5 iters):**
- 100 LHS samples evaluated in 9.8 minutes
- Best coarse RMS: 0.129 (model #28)
- Clear clustering: low v_inner (8000-9000 km/s) and low density (log_rho < -13.2) preferred

**Phase 2 (Refined, 100K packets x 10 iters):**
- Top-20 re-evaluated in 9.7 minutes
- Best refined RMS: 0.092 (improved from 0.129 with more packets)
- Ranking stable: top-5 from Phase 1 remained in top-7 after refinement

**Phase 3 (Production, 500K packets x 20 iters):**
- Top-3 production runs in 10.2 minutes
- Final best RMS: 0.0894

**Best-Fit Parameters:**

| Rank | RMS | log L | v_inner (km/s) | log rho_0 | X_Si | X_Fe | X_O |
|------|-----|-------|----------------|-----------|------|------|-----|
| #1 | 0.0894 | 42.94 | 8,155 | -13.49 | 0.078 | 0.430 | 0.243 |
| #2 | 0.0896 | 43.05 | 8,769 | -13.22 | 0.222 | 0.210 | 0.318 |
| #3 | 0.0993 | 42.98 | 8,662 | -13.46 | 0.194 | 0.335 | 0.221 |

### Physical Interpretation

1. **Photosphere velocity**: The optimal v_inner ~ 8000-9000 km/s is lower than the TARDIS reference (10,000 km/s). This places the photosphere deeper in the ejecta, consistent with the epoch near B-maximum when the photosphere has receded.

2. **Density**: Low log_rho_0 < -13.2 reduces line blanketing in the UV/blue, allowing more flux through the blue continuum to match the observed spectral shape. The -7 power law produces rapidly declining density with velocity.

3. **Composition degeneracy**: Models #1 and #2 achieve nearly identical RMS (0.0894 vs 0.0896) with very different compositions: #1 is Fe-rich (43% Fe, 8% Si, 24% O) while #2 is Si/O-rich (21% Fe, 22% Si, 32% O). This degeneracy reflects the fact that overall opacity (not specific element ratios) dominates the normalized spectral shape at the current spectral resolution.

4. **Si II depth limitation**: The best-fit Si II 6355 depth of 20-49% remains below the observed ~80% and TARDIS 93%. This is the known macro-atom source function limitation: the current implementation over-produces resonant scattering relative to downbranching, keeping too many photons near the Si II line wavelength.

### Output Files

- `fit_results_phase1.csv` -- 100 coarse scan results (all parameters + metrics)
- `fit_results_phase2.csv` -- 20 refined results
- `fit_results_final.csv` -- 3 production results
- `fit_sensitivity.png` -- 5-panel scatter plot (RMS vs each parameter)
- `fit_best_spectrum.png` -- Best-fit spectrum vs observed + TARDIS (3-panel comparison)
- `fit_best_siII.png` -- Si II 6355 region detail for top-3 models
