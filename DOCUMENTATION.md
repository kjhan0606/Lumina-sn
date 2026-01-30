# LUMINA-SN Documentation

**LUMINA-SN**: A High-Performance Monte Carlo Radiative Transfer Code for Type Ia Supernova Spectral Synthesis

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [File Reference](#file-reference)
   - [simulation_state.h / simulation_state.c](#simulation_stateh--simulation_statec)
   - [plasma_physics.h / plasma_physics.c](#plasma_physicsh--plasma_physicsc)
   - [rpacket.h / rpacket.c](#rpacketh--rpacketc)
   - [atomic_loader.c](#atomic_loaderc)
   - [atomic_data.h](#atomic_datah)
   - [lumina_rotation.h / lumina_rotation.c](#lumina_rotationh--lumina_rotationc)
   - [validation.h / validation.c](#validationh--validationc)
   - [physics_kernels.h](#physics_kernelsh)
   - [test_integrated.c](#test_integratedc)
   - [test_transport.c](#test_transportc)
4. [Build Instructions](#build-instructions)
5. [Usage Examples](#usage-examples)
6. [Physical Constants](#physical-constants)
7. [Task Order History](#task-order-history)

---

## Overview

LUMINA-SN is a Monte Carlo radiative transfer code designed for spectral synthesis of Type Ia supernovae. It implements:

- **Saha-Boltzmann** ionization equilibrium with NLTE dilution corrections
- **Sobolev approximation** for line opacity in homologously expanding ejecta
- **Continuum opacity** (bound-free and free-free) for physical photospheric conditions
- **LUMINA rotation** for post-processing multi-angle spectrum synthesis from a single MC run

The code is designed for numerical accuracy (10⁻¹⁰ tolerance with TARDIS-SN) and HPC performance with OpenMP parallelization and thread-safe RNG.

---

## Architecture

### Data Flow

```
┌─────────────────┐
│   Atomic Data   │  HDF5 (TARDIS format)
│   (Kurucz/CD23) │
└────────┬────────┘
         │ atomic_loader.c
         ▼
┌─────────────────┐
│   AtomicData    │  Elements, ions, levels, lines
│   Structure     │
└────────┬────────┘
         │ plasma_physics.c
         ▼
┌─────────────────┐
│  Saha-Boltzmann │  Ionization fractions
│  Ionization     │  Level populations
└────────┬────────┘
         │ simulation_state.c
         ▼
┌─────────────────┐
│  Sobolev τ      │  Line opacities per shell
│  Calculation    │  Continuum opacities
└────────┬────────┘
         │ rpacket.c + physics_kernels.h
         ▼
┌─────────────────┐
│  Monte Carlo    │  Packet trajectories
│  Transport      │  Line/electron interactions
└────────┬────────┘
         │ lumina_rotation.c
         ▼
┌─────────────────┐
│  Observer-Frame │  Multi-angle spectrum
│  Spectrum       │  CSV output
└─────────────────┘
```

### Key Innovations

| Feature | Description |
|---------|-------------|
| **LUMINA Rotation** | Post-processing multi-angle spectrum from single MC run |
| **Continuum Opacity** | Physical bf/ff replaces artificial 60,000 K hack |
| **Dilution Factor** | NLTE correction via W^0.25 temperature reduction |
| **Frequency-Binned Index** | O(1) line lookup in 270k+ line lists |
| **Si II Injection** | Critical 6347/6371 Å doublet for SN Ia diagnostics |
| **HPC-Hardened** | Thread-safe per-packet RNG, no global state |

---

## File Reference

### simulation_state.h / simulation_state.c

**Purpose:** Integrated plasma-transport state management. Pre-computes plasma properties and line opacities for all shells.

#### Data Structures

```c
typedef struct {
    double t_boundary;              // Planck weighting temperature [K]
    double ir_thermalization_frac;  // IR thermalization probability
    double base_thermalization_frac;
    double blue_opacity_scalar;     // Fe-group blue line reduction
    bool enable_continuum_opacity;  // Enable bf/ff opacity
    double bf_opacity_scale;        // Bound-free scaling
    double ff_opacity_scale;        // Free-free scaling
    double R_photosphere;           // Photospheric radius [cm]
    bool enable_dilution_factor;    // Enable NLTE dilution
    // ... wavelength fluorescence parameters
} PhysicsOverrides;

typedef struct {
    int shell_id;
    double r_inner, r_outer;        // Shell geometry [cm]
    double v_inner, v_outer;        // Velocities [cm/s]
    Abundances abundances;          // Per-shell composition
    PlasmaState plasma;             // Ionization state
    int64_t n_active_lines;
    ActiveLine *active_lines;       // Sobolev τ values
    FrequencyBinnedIndex line_index;
    double tau_electron;            // Thomson scattering τ
} ShellState;

typedef struct {
    int n_shells;
    double t_explosion;             // Expansion time [s]
    ShellState *shells;
    const AtomicData *atomic_data;
    Abundances abundances;
    int64_t total_active_lines;
    bool initialized, opacities_computed;
} SimulationState;
```

#### Functions

| Function | Description |
|----------|-------------|
| `simulation_state_init()` | Initialize state with n_shells |
| `simulation_set_shell_geometry()` | Set shell radii/velocities |
| `simulation_set_shell_density()` | Set density per shell |
| `simulation_set_shell_temperature()` | Set temperature per shell |
| `simulation_set_abundances()` | Set element abundances uniformly |
| `simulation_set_stratified_abundances()` | Velocity-dependent composition (Task Order #32) |
| `simulation_compute_plasma()` | Saha-Boltzmann ionization for all shells |
| `simulation_compute_opacities()` | Calculate Sobolev τ for all lines |
| `calculate_bf_opacity()` | Bound-free opacity (Kramers formula) |
| `calculate_ff_opacity()` | Free-free opacity (bremsstrahlung) |
| `calculate_continuum_opacity()` | Total bf + ff |
| `calculate_dilution_factor()` | W = 0.5×[1 - sqrt(1-(R_ph/r)²)] |
| `calculate_tau_sobolev()` | Sobolev optical depth (oscillator strength form) |
| `find_lines_in_window()` | Binary search for lines in frequency window |

#### Important Constants

```c
#define TAU_MAX_CAP 1000.0      // Maximum τ (numerical stability)
#define OPACITY_SCALE 0.05     // Global opacity scaling (Task Order #32 Phase 2)
#define MAX_SHELLS 100
#define SOBOLEV_CONST 2.6540281e-2  // π e² / m_e c [CGS]
#define SIGMA_THOMSON 6.6524587158e-25  // [cm²]
```

---

### plasma_physics.h / plasma_physics.c

**Purpose:** Saha-Boltzmann solver for ionization balance and level populations.

#### Data Structures

```c
typedef struct {
    double T;                       // Temperature [K]
    double rho;                     // Density [g/cm³]
    double n_e;                     // Electron density [cm⁻³]
    double ion_fractions[MAX_ATOMIC_NUMBER+1][MAX_ION_STAGES];
    double partition_functions[MAX_ATOMIC_NUMBER+1][MAX_ION_STAGES];
} PlasmaState;

typedef struct {
    int n_elements;
    int elements[MAX_ATOMIC_NUMBER+1];
    double mass_fraction[MAX_ATOMIC_NUMBER+1];
} Abundances;
```

#### Functions

| Function | Description |
|----------|-------------|
| `calculate_partition_function()` | U(T) = Σ g_i × exp(-E_i/kT) |
| `calculate_saha_factor()` | Φ_{i,i+1}(T) for ionization equilibrium |
| `calculate_ion_fractions()` | Saha chain for element Z |
| `solve_ionization_balance()` | Newton-Raphson for charge neutrality |
| `solve_ionization_balance_diluted()` | With dilution factor W for NLTE |
| `calculate_level_population_fraction()` | Boltzmann: n_level / n_ion |
| `abundances_set_solar()` | Anders & Grevesse 1989 |
| `abundances_set_type_ia_w7()` | W7 SN Ia composition |
| `abundances_set_type_ia_stratified()` | Velocity-dependent stratification |

#### Important Constants

```c
#define SAHA_CONST 2.4146868e15     // cm⁻³ K^(-3/2)
#define PARTITION_ENERGY_CUTOFF 50.0  // Skip levels with E > 50kT
```

---

### rpacket.h / rpacket.c

**Purpose:** Core Monte Carlo transport engine with thread-safe RNG and optimized hot-path calculations.

#### Data Structures

```c
typedef struct {
    double r;              // Radius [cm]
    double mu;             // Direction cosine
    double nu;             // Frequency [Hz]
    double energy;         // Packet energy [erg]
    uint64_t rng_state;    // Per-packet RNG state
    int64_t packet_index;
    int current_shell_id;
    PacketStatus status;
    int last_interaction_type;
    int64_t last_line_interaction_id;
} RPacket;

typedef enum {
    PACKET_IN_PROCESS = 0,
    PACKET_EMITTED = 1,
    PACKET_REABSORBED = 2
} PacketStatus;

typedef enum {
    INTERACTION_BOUNDARY = 0,
    INTERACTION_LINE = 1,
    INTERACTION_ESCATTERING = 2
} InteractionType;

typedef struct {
    bool enable_full_relativity;
    LineInteractionType line_interaction_type;
    double packet_energy;
} MonteCarloConfig;
```

#### Functions

| Function | Description |
|----------|-------------|
| `rpacket_init()` | Initialize packet with unique RNG seed |
| `rpacket_initialize_line_id()` | Binary search for starting line position |
| `move_r_packet()` | Geometric update: r_new² = r² + d² + 2rd×μ |
| `move_packet_across_shell_boundary()` | Handle shell transitions |
| `thomson_scatter()` | Isotropic electron scattering |
| `line_scatter()` | Resonant scattering with optional fluorescence |
| `trace_packet()` | **CORE**: Find next interaction (boundary/line/electron) |
| `single_packet_loop()` | Main driver: trace → move → interact |
| `rng_xorshift64star()` | Thread-safe Xorshift64* generator |
| `rng_uniform()` | Generate [0,1) uniform deviate |

#### Important Constants

```c
#define C_SPEED_OF_LIGHT 2.99792458e10  // cm/s
#define CLOSE_LINE_THRESHOLD 1e-7
#define MISS_DISTANCE 1e99
```

---

### atomic_loader.c

**Purpose:** Load TARDIS atomic database from HDF5 format.

#### Functions

| Function | Description |
|----------|-------------|
| `atomic_data_load_hdf5()` | Main loader: open HDF5 and load all datasets |
| `load_atom_data()` | Elements (symbol, name, mass) |
| `load_ionization_data()` | Ionization energies for all ions |
| `load_levels_data()` | Energy levels with g-values |
| `load_lines_data()` | Spectral lines with oscillator strengths |
| `inject_si_ii_6355_lines()` | Add missing Si II 6347/6371 Å doublet |
| `atomic_get_element()` | Get element by atomic number |
| `atomic_get_ion()` | Get ion by (Z, ion_number) |
| `atomic_get_ionization_energy()` | Ionization energy [erg] |
| `atomic_find_lines_in_range()` | Find lines in frequency range |

---

### atomic_data.h

**Purpose:** C data structures for TARDIS v2.0 HDF5 atomic database.

#### Data Structures

```c
typedef struct {
    int atomic_number;
    char symbol[4];
    char name[16];
    double mass;  // atomic mass [amu]
} Element;

typedef struct {
    int Z, ion_number;
    double ionization_energy;  // [erg]
    int64_t first_level_idx, n_levels;
    int64_t first_line_idx, n_lines;
} Ion;

typedef struct {
    int Z, ion_number, level_number;
    double energy;  // [erg]
    double g;       // statistical weight
    int metastable;
} Level;

typedef struct {
    int Z, ion_number;
    int level_number_lower, level_number_upper;
    double wavelength;  // [Å]
    double nu;          // [Hz]
    double f_ul, f_lu;  // oscillator strengths
    double A_ul;        // Einstein A [s⁻¹]
    double B_ul, B_lu;  // Einstein B coefficients
} Line;

typedef struct {
    int n_elements, n_ions, n_levels, n_lines;
    Element *elements;
    Ion *ions;
    Level *levels;
    Line *lines;
    // Index tables for fast lookup
    int64_t ion_index[MAX_ATOMIC_NUMBER+1][MAX_ION_STAGES];
    int64_t level_index[MAX_ATOMIC_NUMBER+1][MAX_ION_STAGES];
    int64_t line_index[MAX_ATOMIC_NUMBER+1][MAX_ION_STAGES];
} AtomicData;
```

#### Important Constants

```c
#define MAX_ATOMIC_NUMBER 30      // H through Zn
#define MAX_ION_STAGES 31
#define CONST_EV_TO_ERG 1.602176634e-12
#define CONST_ANGSTROM 1.0e-8
```

---

### lumina_rotation.h / lumina_rotation.c

**Purpose:** Post-processing rotation & weighting for multi-angle spectrum synthesis.

#### Data Structures

```c
typedef struct {
    double mu_observer;         // Observer direction cosine
    double t_explosion;         // Expansion time [s]
    double r_outer;             // Outer radius [cm]
    double wavelength_min;      // Spectrum range [Å]
    double wavelength_max;
    int n_wavelength_bins;
} ObserverConfig;

typedef struct {
    double energy_weighted;     // Energy × weight
    double nu_observer;         // Observer-frame frequency [Hz]
    double wavelength;          // [Å]
    double t_observed;          // Observation time [s]
    double weight;              // Solid angle weight
} RotatedPacket;

typedef struct {
    double *flux;               // Flux per bin
    double *wavelength_centers; // Bin centers [Å]
    double total_luminosity;
    int64_t n_packets_used;
    int n_bins;
} Spectrum;
```

#### Functions

| Function | Description |
|----------|-------------|
| `lumina_rotate_packet()` | Apply rotation and Doppler weighting |
| `lumina_apply_rotation_weighting()` | Process full trace for spectrum |
| `spectrum_create()` | Allocate spectrum with n_bins |
| `spectrum_add_packet()` | Add rotated packet to spectrum bin |
| `spectrum_normalize()` | Normalize by total packets |
| `spectrum_write_csv()` | Output to file |

#### Physics

- **Time delay**: t_obs = t_exp - (r×μ_packet/c)
- **Frequency**: Uses lab frame directly (no extra Doppler transform)
- **Weight**: w = (D_observer/D_packet)² accounts for Doppler beaming

---

### validation.h / validation.c

**Purpose:** Validation framework for comparing C vs Python outputs.

#### Data Structures

```c
typedef struct {
    int64_t step_number;
    double r, mu, nu, energy;
    int shell_id;
    PacketStatus status;
    InteractionType interaction_type;
    double distance;
} PacketSnapshot;

typedef struct {
    int64_t packet_index;
    int64_t n_snapshots;
    int64_t capacity;
    PacketSnapshot *snapshots;
} ValidationTrace;
```

#### Functions

| Function | Description |
|----------|-------------|
| `validation_trace_create()` | Allocate trace with capacity |
| `validation_trace_record()` | Record single packet snapshot |
| `validation_trace_write_binary()` | Write to binary file |
| `validation_trace_write_csv()` | Write human-readable CSV |
| `validation_compare_traces()` | Compare C trace vs Python reference |
| `single_packet_loop_traced()` | Transport with validation recording |

---

### physics_kernels.h

**Purpose:** Exact numerical implementations matching TARDIS-SN physics (10⁻¹⁰ tolerance).

#### Functions (static inline)

| Function | Description |
|----------|-------------|
| `get_doppler_factor()` | D = 1 - βμ |
| `get_inverse_doppler_factor()` | D_inv = 1/(1-βμ) |
| `angle_aberration_CMF_to_LF()` | μ_lab = (μ_cmf + β)/(1 + β×μ_cmf) |
| `angle_aberration_LF_to_CMF()` | Inverse direction transformation |
| `calculate_distance_boundary()` | Distance to shell boundary |
| `calculate_distance_electron()` | Distance to electron scattering event |
| `calculate_distance_line()` | Distance to line resonance (Sobolev) |
| `calculate_tau_electron()` | Thomson scattering optical depth |

---

### test_integrated.c

**Purpose:** Full integrated simulation pipeline.

#### Pipeline Steps

1. Load atomic data from HDF5
2. Configure shell geometry and physics overrides
3. Set stratified abundances (Task Order #32)
4. Compute Saha-Boltzmann ionization
5. Calculate Sobolev opacities
6. Run Monte Carlo transport (OpenMP parallel)
7. Apply LUMINA rotation weighting
8. Output spectrum to CSV

#### Command Line

```bash
./test_integrated [options] [atomic_file] [n_packets] [output_file]

Options:
  --v-inner <km/s>   Set inner velocity (default: 10000)
  --v-outer <km/s>   Set outer velocity (default: 25000)
  --T <K>            Set photospheric temperature (default: 13500)
  --stratified       Use velocity-stratified abundances (default)

Environment Variables:
  LUMINA_T_BOUNDARY    Planck weighting temperature (default: 13000 K)
  LUMINA_OPACITY_SCALE Opacity scaling factor
  LUMINA_DILUTION      Enable NLTE dilution (default: 1)
```

---

### test_transport.c

**Purpose:** Monte Carlo transport validation and simulation driver.

#### Modes

- **Validation**: `--validate trace.bin` - Compare against Python reference
- **Simulation**: `--simulate N` - Run N packets with parallelization
- **Spectrum**: `--spectrum out.csv` - Generate spectrum with LUMINA rotation

#### Default Parameters

```c
#define DEFAULT_N_PACKETS 100000
#define DEFAULT_N_SHELLS 20
#define DEFAULT_N_LINES 1000
#define DEFAULT_T_EXPLOSION 86400  // 1 day [s]
#define DEFAULT_WAVELENGTH_MIN 3000  // [Å]
#define DEFAULT_WAVELENGTH_MAX 10000
```

---

## Build Instructions

### Prerequisites

- GCC with C11 support
- HDF5 library (for atomic data loading)
- OpenMP (optional, for parallelization)

### Compilation

```bash
make clean
make
```

### Output Executables

| Executable | Purpose |
|------------|---------|
| `test_integrated` | Full simulation pipeline |
| `test_transport` | Transport validation |
| `test_atomic` | Atomic data loading test |
| `test_plasma` | Plasma physics test |

---

## Usage Examples

### Run SN 2011fe Simulation

```bash
./test_integrated atomic/kurucz_cd23_chianti_H_He.h5 100000 spectrum.csv
```

### Run with Custom Parameters

```bash
./test_integrated --v-inner 10500 --T 13500 \
    atomic/kurucz_cd23_chianti_H_He.h5 500000 golden_spectrum.csv
```

### Validate Against Python

```bash
./test_transport --validate python_trace.bin
```

---

## Physical Constants

| Constant | Value | Units |
|----------|-------|-------|
| Speed of light | 2.99792458×10¹⁰ | cm/s |
| Thomson cross-section | 6.6524587158×10⁻²⁵ | cm² |
| Sobolev constant | 2.6540281×10⁻² | cm² s⁻¹ |
| Boltzmann constant | 1.380649×10⁻¹⁶ | erg/K |
| Planck constant | 6.62607015×10⁻²⁷ | erg s |
| Electron mass | 9.1093837015×10⁻²⁸ | g |
| eV to erg | 1.602176634×10⁻¹² | erg/eV |

---

## Task Order History

### Task Order #30: Physics Engine Overhaul
- Implemented continuum opacity (bound-free, free-free)
- Added NLTE dilution factor correction
- Replaced 60,000 K hack with physical T_boundary = 13,000 K
- Added wavelength-dependent fluorescence

### Task Order #32: Si II 6355 Velocity Calibration
- **Phase 1**: Spatial abundance tapering (X_Si: 35% → 2%)
- **Phase 2**: Opacity softening (OPACITY_SCALE = 0.05, TAU_MAX_CAP = 1000)
- **Phase 3**: Golden validation (500k packets)
- **Result**: Δv = +12 km/s (within ±500 km/s target)

---

## License

This code is developed for scientific research in supernova spectral modeling.

## Authors

- Original development for LUMINA-SN project
- Task Order implementations with Claude Opus 4.5

---

*Documentation generated: 2026-01-30*
