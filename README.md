# LUMINA-SN

**1D Monte Carlo Radiative Transfer Code** — Type Ia Supernova Spectrum Simulation

LUMINA-SN is a high-performance radiative transfer code that reimplements the core physics of [TARDIS](https://tardis-sn.github.io/tardis/) in C/CUDA. It traces photon packets through supernova ejecta in homologous expansion using the Monte Carlo method to compute observable spectra.

---

## Directory Structure

```
lumina-sn/
├── README.md                 ← This document
├── Makefile                  ← Build system
├── .gitignore
│
├── src/                      ← C/CUDA source code (6 files, ~3900 lines)
│   ├── lumina.h              ← Structs, constants, function prototypes
│   ├── lumina_main.c         ← CPU main driver + iteration loop
│   ├── lumina_transport.c    ← CPU transport physics kernel
│   ├── lumina_plasma.c       ← Plasma solver + convergence logic
│   ├── lumina_atomic.c       ← Data loader (NPY/CSV) + memory management
│   └── lumina_cuda.cu        ← GPU transport kernel (CUDA)
│
├── scripts/                  ← Python analysis/visualization scripts (15)
│   ├── fit_parameter_search.py      ← 5D parameter fitting for SN 2011fe (primary)
│   ├── plot_spectrum_comparison.py   ← Spectrum comparison plot
│   ├── compare_spectra.py           ← TARDIS vs LUMINA spectrum comparison
│   ├── compare_spectra_v2.py        ← Detailed shape comparison
│   ├── fit_sn2011fe.py              ← SN 2011fe model vs observation comparison
│   ├── diagnose_w.py                ← Dilution factor W diagnostics
│   ├── validate_partition.py        ← Partition function validation
│   ├── validate_plasma.py           ← Plasma state validation
│   ├── validate_tau_detail.py       ← tau_sobolev detailed analysis
│   ├── validate_tau_impact.py       ← Neutral species tau impact analysis
│   ├── debug_neutral_tau.py         ← Neutral atom tau debugging
│   ├── compare_tau_c_vs_python.py   ← C vs Python tau comparison
│   ├── check_c_ions.py              ← Ion density verification
│   ├── check_ion_ordering.py        ← Ion ordering validation
│   ├── extract_atomic_data.py       ← Extract atomic data from HDF5
│   └── export_tardis_reference.py   ← Export TARDIS reference data
│
├── data/                     ← Input data
│   ├── atomic/               ← Atomic data (HDF5)
│   ├── tardis_reference/     ← TARDIS converged state (CSV/NPY)
│   ├── model/                ← Model parameters
│   └── sn2011fe/             ← SN 2011fe observational data + configuration
│
└── docs/                     ← Technical documentation
    ├── ARCHITECTURE.md       ← Code architecture
    ├── PHYSICS.md            ← Physics model description
    └── HISTORY.md            ← Development history + major bug fixes
```

---

## Dependencies

### Required
| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| **GCC** | 8.0+ | C11 compiler |
| **GNU Make** | 3.81+ | Build system |

### GPU Build (Optional)
| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| **CUDA Toolkit** | 11.0+ | GPU kernel compilation (nvcc) |
| **NVIDIA GPU** | Compute Capability 7.0+ | GPU execution |

### Python Scripts (Optional)
| Package | Purpose |
|---------|---------|
| **Python 3** | Script execution |
| **numpy** | Numerical computation, NPY file reading |
| **matplotlib** | Spectrum plot generation |
| **scipy** | Spectrum smoothing |
| **h5py** | HDF5 atomic data reading (extract_atomic_data.py only) |

### Installation Example (RHEL/CentOS/Rocky)
```bash
# C compiler + Make
sudo dnf install gcc make

# Python packages
pip install numpy matplotlib scipy h5py

# CUDA (for GPU usage) — see https://developer.nvidia.com/cuda-downloads
```

---

## Build

Run from the project root.

### CPU Build (Default)
```bash
make
```
On success, the `lumina` binary is created in the project root.

### CPU + OpenMP Parallel Build
```bash
make OMP=1
```
Parallelizes packet transport across multiple CPU cores.

### GPU (CUDA) Build
```bash
make cuda
```
On success, the `lumina_cuda` binary is created in the project root.

> **Note**: The GPU build may require modifying `-arch=sm_89` in the Makefile to match your GPU.
> - RTX 3090/A100: `-arch=sm_80`
> - RTX 4090: `-arch=sm_89`
> - V100: `-arch=sm_70`

### Clean Build
```bash
make clean   # Delete binaries
make         # Rebuild
```

---

## Running

### Basic Usage

```bash
# CPU (default settings: 2M packets x 19 + 20M final, 20 iterations)
./lumina

# GPU (same settings)
./lumina_cuda
```

### Command-Line Arguments

```
./lumina [ref_dir] [n_packets] [n_iterations]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `ref_dir` | `data/tardis_reference` | TARDIS reference data directory |
| `n_packets` | 2000000 | Number of Monte Carlo packets |
| `n_iterations` | 20 | Number of convergence iterations |

### Examples

```bash
# Quick test (10K packets, 5 iterations — a few seconds)
./lumina data/tardis_reference 10000 5

# Medium test (200K packets, 10 iterations)
./lumina data/tardis_reference 200000 10

# Production (2M packets, 20 iterations — CPU: minutes, GPU: tens of seconds)
./lumina data/tardis_reference 2000000 20

# Make shortcuts
make run    # Run with default settings
make test   # Quick test (10K packets, 5 iterations)
```

### Expected Output (Terminal)

```
=== LUMINA-SN Monte Carlo Transport ===
Reference data: data/tardis_reference
  n_shells = 30, n_lines = 137252
  T_inner = 10521.52 K, L_inner = 8.952e+42 erg/s
  ...

--- Iteration 1/20: 2000000 packets ---
  Escaped: 820143 (41.0%), Reabsorbed: 1179857 (59.0%)
  W error: 2.31%, T_rad error: 1.05%
  ...

--- Final Results ---
  T_inner: 10509.3 K (TARDIS: 10521.5 K, error: 0.12%)
  Max W error: 0.67%
  Max T_rad error: 0.28%
```

---

## Output Files

The simulation produces the following files in the project root:

| File | Format | Description |
|------|--------|-------------|
| `lumina_spectrum.csv` | CSV | Final spectrum (wavelength_angstrom, L_lambda_cgs) |
| `lumina_spectrum_virtual.csv` | CSV | Virtual packet spectrum (GPU only) |
| `lumina_spectrum_rotation.csv` | CSV | Rotation packet spectrum |
| `lumina_plasma_state.csv` | CSV | Final plasma state (shell, W, T_rad, n_e) |

### Spectrum CSV Format
```csv
wavelength_angstrom,L_lambda_cgs
500.0,1.234e+38
519.5,2.345e+38
...
```
- `wavelength_angstrom`: Wavelength (Angstroms, 500 -- 20000)
- `L_lambda_cgs`: Luminosity density (erg/s/cm, luminosity per wavelength)

### Plasma State CSV Format
```csv
shell,W,T_rad,n_e
0,0.380,12291,2.34e+09
1,0.312,11543,1.98e+09
...
```
- `W`: Radiation dilution factor
- `T_rad`: Radiation temperature (K)
- `n_e`: Electron density (cm^-3)

---

## SN 2011fe Parameter Fitting

LUMINA-SN includes an automated parameter fitting pipeline that searches a 5-dimensional
physical parameter space to find the best-fit model for the observed SN 2011fe spectrum
(Pereira+2013, phase -0.3d from B-max).

### Fitting Parameters

| # | Parameter | Range | Description |
|---|-----------|-------|-------------|
| 1 | `log_L` (erg/s) | [42.8, 43.15] | Luminosity (controls ionization via T_inner) |
| 2 | `v_inner` (km/s) | [8000, 13000] | Photosphere velocity |
| 3 | `log_rho_0` (g/cm^3) | [-13.5, -12.7] | Reference density at v_inner |
| 4 | `X_Si` | [0.03, 0.25] | Silicon mass fraction |
| 5 | `X_Fe` | [0.15, 0.75] | Iron mass fraction |

Fixed abundances: S=0.05, Ca=0.03, Co=0.05, Ni=0.10, C=0.02. Oxygen fills the remainder
(X_O = 0.75 - X_Si - X_Fe). Density follows a power law: rho(v) = rho_0 * (v/v_inner)^(-7).

### Running the Fit

```bash
# Validate pipeline (single model, ~5 seconds)
python3 scripts/fit_parameter_search.py --test

# Phase 1 only: coarse scan (100 LHS samples, 20K packets x 5 iters, ~10 min)
python3 scripts/fit_parameter_search.py --phase1

# Full 3-phase search (~30 min on CPU)
python3 scripts/fit_parameter_search.py
```

### Coarse-to-Fine Strategy

| Phase | Samples | Packets | Iters | Purpose |
|-------|---------|---------|-------|---------|
| 1 | 100 (Latin Hypercube) | 20K | 5 | Broad parameter scan |
| 2 | Top-20 from Phase 1 | 100K | 10 | Refinement |
| 3 | Top-3 from Phase 2 | 500K | 20 | Production spectra |

### Results (SN 2011fe)

Best-fit RMS = **0.089** (normalized spectrum vs observed Pereira+2013 data):

| Rank | RMS | log L | v_inner | log rho_0 | X_Si | X_Fe | X_O |
|------|-----|-------|---------|-----------|------|------|-----|
| #1 | 0.0894 | 42.94 | 8,155 km/s | -13.49 | 0.078 | 0.430 | 0.243 |
| #2 | 0.0896 | 43.05 | 8,769 km/s | -13.22 | 0.222 | 0.210 | 0.318 |
| #3 | 0.0993 | 42.98 | 8,662 km/s | -13.46 | 0.194 | 0.335 | 0.221 |

Key trends:
- **v_inner ~ 8000-9000 km/s** is optimal (lower than default 10,000 km/s)
- **Low density** (log rho_0 < -13.2) preferred -- reduces UV line blanketing
- Si/Fe abundance degeneracy: two distinct compositions give similar RMS
- Si II 6355 depth: 20-49% (known limitation vs TARDIS 93% -- macro-atom source function)

### Fitting Output Files

| File | Description |
|------|-------------|
| `fit_results_phase1.csv` | 100 coarse scan results |
| `fit_results_phase2.csv` | 20 refined results |
| `fit_results_final.csv` | Top-3 production results |
| `fit_sensitivity.png` | 5-panel: RMS vs each parameter |
| `fit_best_spectrum.png` | Best-fit vs observed + TARDIS (3-panel) |
| `fit_best_siII.png` | Si II 6355 region detail for top-3 models |

---

## Visualization

All Python scripts are run from the project root.

### Spectrum Comparison Plot (Primary)
```bash
python3 scripts/plot_spectrum_comparison.py
```
Generates PNG files comparing the TARDIS reference spectrum with the LUMINA spectrum:
- `spectrum_comparison.png` — Full wavelength range comparison
- `spectrum_comparison_siII.png` — Si II 6355 A absorption trough zoom

### Other Analysis Scripts
```bash
python3 scripts/compare_spectra.py       # Spectrum shape + Si II analysis
python3 scripts/fit_sn2011fe.py          # SN 2011fe model vs observation
python3 scripts/diagnose_w.py            # W discrepancy diagnostics
python3 scripts/validate_plasma.py       # Full plasma state validation
```

---

## FAQ / Troubleshooting

### Q: `nvcc: command not found` error during `make cuda`
The CUDA Toolkit is not installed or not in PATH.
```bash
# Check CUDA path
which nvcc
# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Q: `Failed to load reference data` error
The `data/tardis_reference/` directory is missing reference data. You need to generate reference data from a TARDIS installation first:
```bash
# In a Python environment with TARDIS installed
python3 scripts/export_tardis_reference.py
```

### Q: `unsupported gpu architecture 'compute_89'` error during GPU build
Modify `-arch=sm_89` in the Makefile to match your GPU:
```bash
# Check GPU compute capability
nvidia-smi
# or
nvcc --list-gpu-arch
```

### Q: Results differ from TARDIS
- Too few packets cause high statistical noise. Use at least 200K packets.
- 20 iterations are needed for sufficient W/T_rad convergence.
- With 2M packets x 20 iterations, expect W error < 1%, T_rad error < 0.5%.

### Q: HDF5 warning messages
```
HDF5-DIAG: Error detected in HDF5 ...
```
These are non-fatal warnings during atomic data loading. They can be safely ignored.

### Q: `make clean` deletes output files
`make clean` only deletes binaries. Output CSV/PNG files are listed in `.gitignore` but not deleted. Back up important results separately.

---

## Comparison with Other Codes

LUMINA-SN occupies a unique niche: **TARDIS physics + GPU acceleration + NLTE + ML fitting pipeline**.

| Feature | **LUMINA** | **TARDIS** | **SYN++** | **SEDONA** | **ARTIS** | **CMFGEN** | **PHOENIX** |
|---|---|---|---|---|---|---|---|
| RT method | MC | MC | Parametric | MC | MC | CMF-ALI | ALI |
| Geometry | 1D | 1D | 1D | 3D | 3D | 1D | 1D |
| NLTE | Full* | Dilute-LTE | LTE | LTE/NLTE | Full | Full | Full |
| Macro-atom | Yes | Yes | No | Partial | Yes | --- | --- |
| GPU accel. | **CUDA** | No | No | No | No | No | No |
| Time-dep. | No | No | No | Yes | Yes | No | No |
| Speed (1 model) | **~14s** (GPU) | ~60-120s | ~1s | ~hours | ~hours | ~hours | ~hours |

*\*Full statistical equilibrium for Si, Ca, Fe, S (2017 levels, 36K NLTE lines). Same physics as CMFGEN/PHOENIX (radiative BB, collisional BB, photoionization, recombination). Other species use nebular approximation.*

**Key advantages:**
- **vs TARDIS**: 10x faster (GPU), full NLTE for key ions, no Python runtime dependency
- **vs SYN++**: Self-consistent radiation field, macro-atom fluorescence, quantitative abundances
- **vs SEDONA/ARTIS**: GPU acceleration enables large parameter surveys (5000 models/day)
- **vs CMFGEN/PHOENIX**: Monte Carlo handles line overlap naturally; 100x faster per model

**Trade-offs:** 1D only, single epoch, SN Ia focused, NLTE limited to 4 elements (vs all species in CMFGEN/PHOENIX/ARTIS).

For a detailed comparison, see Chapter 15 of the [Technical Manual](docs/manual/).

---

## Technical Documentation

For detailed technical information, see the `docs/` directory:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Code structure, data flow, CPU vs GPU differences
- **[docs/PHYSICS.md](docs/PHYSICS.md)** — Monte Carlo radiative transfer, Sobolev approximation, macro-atom model
- **[docs/HISTORY.md](docs/HISTORY.md)** — Development history, major bug fix records

---

## References

- Kerzendorf & Sim (2014), *TARDIS: A Monte Carlo radiative-transfer spectral synthesis code*, MNRAS 440, 387
- Lucy (2002), *Monte Carlo transition probabilities*, A&A 384, 725
- Lucy (2003), *Monte Carlo transition probabilities. II.*, A&A 403, 261
- Mazzali & Lucy (1993), *The application of Monte Carlo methods to the synthesis of early-time supernovae spectra*, A&A 279, 447
