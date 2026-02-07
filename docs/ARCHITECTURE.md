# LUMINA-SN Architecture

## Overview

LUMINA-SN is a 1D Monte Carlo radiative transfer code for Type Ia supernovae, written in C with a CUDA GPU port. It is designed as a faithful reimplementation of [TARDIS](https://tardis-sn.github.io/tardis/) (the Python/Cython radiative transfer code) with the goal of achieving equivalent physical accuracy at significantly higher performance. The code models photon propagation through homologously expanding supernova ejecta using the Sobolev approximation for line interactions, macro-atom fluorescence for non-LTE source functions, and iterative convergence of the radiation field and plasma state.

## File Structure

| File | Lines | Role |
|------|-------|------|
| `src/lumina.h` | 377 | Core structs, constants, function prototypes |
| `src/lumina_main.c` | 463 | CPU driver, iteration loop, OpenMP parallelization |
| `src/lumina_transport.c` | 515 | CPU transport physics kernels |
| `src/lumina_plasma.c` | 524 | Plasma solver, convergence, spectrum binning |
| `src/lumina_atomic.c` | 694 | Data I/O (NPY/CSV), memory management, RNG |
| `src/lumina_cuda.cu` | 1334 | GPU transport kernel, device functions |

Total: ~3,900 lines of C/CUDA.

## Key Data Structures

All core structures are defined in `src/lumina.h`.

- **`Geometry`** -- 1D radial shell grid. Fields: `r_inner`, `r_outer`, `v_inner`, `v_outer`, `time_explosion`, `n_shells`. Shells are defined by velocity boundaries in homologous expansion (r = v * t_exp).

- **`OpacityState`** -- Line opacity data. Contains the sorted line list (`nu_line`, `tau_sobolev`), macro-atom transition data (levels, transition types, destination level indices, transition probabilities), and downbranch/fluorescence tables.

- **`PlasmaState`** -- Per-shell radiation field and thermodynamic state. Fields: dilution factor `W`, radiation temperature `T_rad`, matter density, electron density `n_e`, and ion number densities.

- **`MCConfig`** -- Simulation control parameters. Includes `n_packets`, `n_iterations`, `T_inner`, `L_requested` (target bolometric luminosity), `damping_constant` (for iterative convergence), and hold iterations.

- **`RPacket`** -- State of a single Monte Carlo radiation packet. Fields: radial position `r`, propagation direction cosine `mu`, comoving-frame frequency `nu`, packet energy, current `shell_id`, current `line_id` (position in sorted line list), and packet status (in-process, escaped, absorbed).

- **`Estimators`** -- Monte Carlo estimators accumulated during transport. Per-shell: mean intensity `j`, mean frequency `nu_bar`. Per-line: `j_blue` (blue-wing mean intensity) and `Edotlu` (energy deposition rate), used to update Sobolev optical depths.

- **`AtomicData`** -- Complete atomic physics dataset for the plasma solver. Includes energy levels, ionization energies, elemental abundances, partition function tables, and zeta factors for nebular ionization.

- **`RNG`** -- xoshiro256** pseudo-random number generator state, consisting of 4 x `uint64_t`. One instance per thread (CPU) or per GPU thread.

## Data Flow

```
Load TARDIS reference data (CSV/NPY)
         |
         v
  Initialize packets at photosphere (T_inner blackbody)
         |
         v
  +----------------------------------------------+
  | Transport loop (per packet, parallelized)     |
  |  +-- Trace packet (find next interaction)     |
  |  |   +-- Distance to shell boundary           |
  |  |   +-- Distance to next line (Sobolev sweep)|
  |  |   +-- Distance to e-scattering event       |
  |  +-- Move packet to nearest interaction       |
  |  +-- Handle interaction:                      |
  |  |   +-- Boundary -> change shell             |
  |  |   +-- Line -> scatter/downbranch/macro-atom|
  |  |   +-- Thomson -> isotropic scatter         |
  |  +-- Update estimators (j, nu_bar)            |
  +----------------------------------------------+
         |
         v
  Solve radiation field (W, T_rad from estimators)
         |
         v
  Update plasma state (partition functions, n_e, ions, tau_sobolev)
         |
         v
  Update T_inner (luminosity convergence)
         |
         v
  Repeat for N iterations
         |
         v
  Output: spectrum CSV, plasma state CSV
```

### Iteration Convergence

Each iteration proceeds as: (1) emit packets from the inner boundary as a blackbody at T_inner, (2) transport all packets through the ejecta, (3) compute new W and T_rad from MC estimators with 50% damping, (4) update the plasma state (Saha ionization, partition functions, electron density via Newton iteration, Sobolev optical depths), (5) update T_inner using `T_new = T_old * (L_requested / L_emitted)^(-0.5)`. This matches the TARDIS convergence strategy.

## CPU vs CUDA Differences

The CPU and GPU code paths implement identical physics but differ in parallelization and memory strategy:

| Aspect | CPU (`lumina_main.c` + `lumina_transport.c`) | GPU (`lumina_cuda.cu`) |
|--------|----------------------------------------------|------------------------|
| Parallelism | OpenMP parallel for over packets | One CUDA thread per packet, 256 threads/block |
| Estimators | Per-thread local arrays, reduced after transport | `atomicAdd` for j/nu_bar (no j_blue/Edotlu due to memory) |
| RNG | Per-thread xoshiro256** state | Per-thread xoshiro256** state (seeded from host) |
| Spectrum | Post-transport binning on CPU | On-device virtual packet tracing with `atomicAdd` into spectrum bins |
| Plasma solver | Runs on CPU | Runs on CPU (GPU only handles transport) |
| Performance | 7.2s for 200K packets | 728ms for 200K packets (~10x speedup) |

The GPU kernel handles only the Monte Carlo transport phase. All pre- and post-processing (packet initialization, plasma solving, convergence checks, spectrum output) runs on the CPU. Data is transferred to/from the GPU each iteration via `cudaMemcpy`.

## Build System

Makefile-based build with three targets:

```
make            # CPU binary (lumina), serial
make OMP=1      # CPU binary with OpenMP parallelization
make cuda       # GPU binary (lumina_cuda), requires NVCC + CUDA toolkit
```

Dependencies: C99 compiler (gcc), math library (`-lm`). CUDA target additionally requires NVCC and is compiled with `-arch=sm_89 -std=c++14` (targeting NVIDIA Ada Lovelace GPUs).

Input data is loaded from TARDIS reference files in NPY and CSV format (no HDF5 dependency at runtime for the main simulation). The atomic line data is read from HDF5 during a separate preprocessing step.

## Parameter Fitting Pipeline

The fitting pipeline (`scripts/fit_parameter_search.py`, ~430 lines of Python) automates multi-dimensional parameter searches by orchestrating many LUMINA runs:

```
Latin Hypercube Sampling (5D)
         |
         v
  For each parameter set:
    +-- Create temp reference dir (symlinks + modified CSVs)
    +-- Generate: config.json, geometry.csv, density.csv,
    |             abundances.csv, electron_densities.csv,
    |             plasma_state.csv
    +-- Run: ./lumina <temp_dir> <n_packets> <n_iters> rotation
    +-- Read: lumina_spectrum_rotation.csv
    +-- Compute: normalized RMS vs observed SN 2011fe
    +-- Measure: Si II 6355 trough depth + velocity
    +-- Cleanup temp dir
         |
         v
  Rank by RMS -> select top-N -> repeat with more packets
         |
         v
  Output: CSV results, sensitivity plots, spectrum comparison
```

The pipeline uses symlinks to avoid copying the large atomic data files (transition_probabilities.npy at 95MB, tau_sobolev.npy at 32MB) for each model. Only the 6 configuration files that depend on physical parameters are regenerated. Each LUMINA run operates in an isolated working directory to prevent output file conflicts.
