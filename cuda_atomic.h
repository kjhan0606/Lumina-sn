/**
 * LUMINA-SN CUDA Atomic Data Structures
 * cuda_atomic.h - GPU-optimized atomic data with flattened memory layout
 *
 * Design principles:
 *   1. Flattened structures for coalesced memory access
 *   2. Constant memory for read-only physics constants
 *   3. Texture memory potential for partition function cache
 *   4. SOA (Structure of Arrays) layout for parallel access
 */

#ifndef CUDA_ATOMIC_H
#define CUDA_ATOMIC_H

#include <cuda_runtime.h>
#include <stdint.h>

/* ============================================================================
 * CUDA CONFIGURATION
 * ============================================================================ */

#define CUDA_MAX_ELEMENTS       30
#define CUDA_MAX_IONS           500
#define CUDA_MAX_LEVELS         30000
#define CUDA_MAX_LINES          300000
#define CUDA_MAX_SHELLS         100
#define CUDA_MAX_ACTIVE_LINES   100000  /* Per shell limit */

/* Partition function cache */
#define CUDA_PARTITION_N_TEMPS  50
#define CUDA_PARTITION_T_MIN    2000.0
#define CUDA_PARTITION_T_MAX    50000.0

/* Spectrum bins */
#define CUDA_SPECTRUM_N_BINS    1000

/* Thread configuration */
#define CUDA_THREADS_PER_BLOCK  256
#define CUDA_MAX_BLOCKS         1024

/* ============================================================================
 * PHYSICAL CONSTANTS (Constant Memory)
 * ============================================================================ */

/**
 * Physical constants stored in constant memory for fast broadcast access.
 * All threads in a warp access these simultaneously.
 */
typedef struct {
    double c;              /* Speed of light [cm/s] */
    double h;              /* Planck constant [erg s] */
    double k_B;            /* Boltzmann constant [erg/K] */
    double m_e;            /* Electron mass [g] */
    double e;              /* Electron charge [esu] */
    double amu;            /* Atomic mass unit [g] */
    double sigma_thomson;  /* Thomson cross-section [cm²] */
    double sobolev_const;  /* π e² / (m_e c) [cm² s⁻¹] */
    double angstrom;       /* 1 Angstrom in cm */
    double saha_const;     /* (2π m_e k / h²)^(3/2) */
} CudaPhysicsConstants;

/* ============================================================================
 * FLATTENED ATOMIC DATA (Global Memory - SOA Layout)
 * ============================================================================ */

/**
 * Flattened element data - Structure of Arrays for coalesced access
 */
typedef struct {
    int32_t  n_elements;

    /* SOA arrays - each element property in contiguous memory */
    int8_t   atomic_number[CUDA_MAX_ELEMENTS];  /* Z = 1..30 */
    double   mass[CUDA_MAX_ELEMENTS];           /* Atomic mass [amu] */
    char     symbol[CUDA_MAX_ELEMENTS][3];      /* Element symbol */

    /* Index into ions array: ions[ion_start[Z]:ion_end[Z]] */
    int32_t  ion_start[CUDA_MAX_ELEMENTS + 1];
    int32_t  ion_end[CUDA_MAX_ELEMENTS + 1];
} CudaElementData;

/**
 * Flattened ion data - SOA layout
 */
typedef struct {
    int32_t  n_ions;

    /* SOA arrays for each ion */
    int8_t   atomic_number[CUDA_MAX_IONS];
    int8_t   ion_number[CUDA_MAX_IONS];        /* Ionization stage (0=neutral) */
    double   ionization_energy[CUDA_MAX_IONS]; /* χ [erg] */
    int32_t  n_levels[CUDA_MAX_IONS];          /* Number of levels */

    /* Index into levels array */
    int32_t  level_start[CUDA_MAX_IONS];
    int32_t  level_end[CUDA_MAX_IONS];

    /* Index into lines array (lines with this ion as lower level) */
    int32_t  line_start[CUDA_MAX_IONS];
    int32_t  line_end[CUDA_MAX_IONS];
} CudaIonData;

/**
 * Flattened level data - SOA layout
 * Optimized for Boltzmann population calculations
 */
typedef struct {
    int32_t  n_levels;

    /* SOA arrays - contiguous for vectorized access */
    int8_t   atomic_number[CUDA_MAX_LEVELS];
    int8_t   ion_number[CUDA_MAX_LEVELS];
    int16_t  level_number[CUDA_MAX_LEVELS];
    double   energy[CUDA_MAX_LEVELS];          /* E [erg] from ground state */
    double   g[CUDA_MAX_LEVELS];               /* Statistical weight */
    int32_t  metastable[CUDA_MAX_LEVELS];      /* Is metastable? */
} CudaLevelData;

/**
 * Flattened line data - SOA layout
 * Critical for Sobolev optical depth calculation
 * SORTED BY FREQUENCY for efficient lookup
 */
typedef struct {
    int64_t  n_lines;

    /* Frequency-sorted SOA arrays */
    double   nu[CUDA_MAX_LINES];               /* Line frequency [Hz] */
    double   wavelength[CUDA_MAX_LINES];       /* Wavelength [cm] */
    double   f_lu[CUDA_MAX_LINES];             /* Oscillator strength */
    double   A_ul[CUDA_MAX_LINES];             /* Einstein A [s⁻¹] */

    /* Ion/level indices for population lookup */
    int8_t   atomic_number[CUDA_MAX_LINES];
    int8_t   ion_number[CUDA_MAX_LINES];
    int16_t  level_lower[CUDA_MAX_LINES];
    int16_t  level_upper[CUDA_MAX_LINES];

    /* Precomputed Sobolev constant factor: SOBOLEV_CONST * f_lu * λ */
    double   sobolev_factor[CUDA_MAX_LINES];
} CudaLineData;

/* ============================================================================
 * PARTITION FUNCTION CACHE (Constant/Texture Memory)
 * ============================================================================ */

/**
 * Pre-computed partition functions on logarithmic T grid.
 * Stored in constant memory for fast broadcast access.
 *
 * Access pattern: U(Z, ion, T) via log-space interpolation
 */
typedef struct {
    double T_grid[CUDA_PARTITION_N_TEMPS];
    double log_T_min;
    double log_T_max;
    double d_log_T;

    /* Partition functions: U[Z][ion][T_idx] */
    /* Flattened for coalesced access: index = Z * (Z+1)/2 * N_T + ion * N_T + t */
    double U[CUDA_MAX_ELEMENTS + 1][CUDA_MAX_ELEMENTS + 2][CUDA_PARTITION_N_TEMPS];
} CudaPartitionCache;

/* ============================================================================
 * SHELL STATE (Global Memory with Shared Memory Caching)
 * ============================================================================ */

/**
 * Shell properties - will be cached in shared memory during transport
 */
typedef struct {
    /* Geometry */
    double r_inner;
    double r_outer;
    double v_inner;
    double v_outer;

    /* Plasma state */
    double T;              /* Temperature [K] */
    double rho;            /* Density [g/cm³] */
    double n_e;            /* Electron density [cm⁻³] */

    /* Opacities */
    double sigma_thomson_ne;  /* n_e * σ_T [cm⁻¹] */
    double tau_electron;      /* Electron scattering optical depth */

    /* Ion fractions (for major species only) */
    double ion_fraction[CUDA_MAX_ELEMENTS + 1][6];  /* [Z][ion 0-5] */
    double n_ion[CUDA_MAX_ELEMENTS + 1][6];         /* Ion densities */

    /* Active lines in this shell */
    int64_t n_active_lines;
    int64_t active_line_start;  /* Index into global active lines array */
} CudaShellState;

/**
 * Active line entry - tau > threshold
 */
typedef struct {
    int64_t line_idx;      /* Index into global line data */
    double  nu;            /* Line frequency [Hz] */
    double  tau_sobolev;   /* Pre-computed Sobolev optical depth */
} CudaActiveLine;

/* ============================================================================
 * SIMULATION CONFIGURATION
 * ============================================================================ */

typedef struct {
    /* Model parameters */
    int32_t  n_shells;
    double   t_explosion;
    double   nu_min;
    double   nu_max;

    /* Spectrum binning */
    int32_t  n_bins;
    double   d_nu;
    double   d_log_nu;

    /* Packet configuration */
    int64_t  n_packets;
    double   packet_energy;

    /* Transport limits */
    int32_t  max_interactions;
    int32_t  max_steps;
    double   tau_min_active;

    /* LUMINA rotation */
    double   mu_observer;
    int      enable_lumina;

    /* Task Order #038-Revised: Macro-Atom Physics */
    int32_t  line_interaction_type;   /* 0=SCATTER, 1=DOWNBRANCH, 2=MACROATOM */
    int32_t  enable_full_relativity;  /* Full relativistic corrections */
} CudaSimConfig;

/* ============================================================================
 * MACRO-ATOM DATA STRUCTURES (Task Order #038-Revised)
 * ============================================================================ */

#define CUDA_MAX_DOWNBRANCH_ENTRIES 10000000  /* 10M max emission entries */

/**
 * Downbranch table for macro-atom fluorescence
 * Stores cumulative probabilities for emission line selection
 */
typedef struct {
    int64_t  n_lines;              /* Number of lines with downbranch data */
    int64_t  total_emission_entries;

    /* Per-line offsets into emission arrays */
    int64_t  *emission_start;      /* [n_lines] Start index */
    int32_t  *emission_count;      /* [n_lines] Number of emission options */

    /* Emission line data (flattened) */
    int64_t  *emission_line_id;    /* Target line index */
    double   *branching_prob;      /* Cumulative probability */
} CudaDownbranchData;

/* ============================================================================
 * DEVICE MEMORY POINTERS (Host-side management)
 * ============================================================================ */

/**
 * Container for all device memory pointers
 * Used by host code to manage GPU memory lifecycle
 */
typedef struct {
    /* Atomic data (read-only) */
    CudaElementData  *d_elements;
    CudaIonData      *d_ions;
    CudaLevelData    *d_levels;
    CudaLineData     *d_lines;

    /* Partition cache (constant memory symbol, not pointer) */
    /* Use cudaMemcpyToSymbol for d_partition_cache */

    /* Shell state */
    CudaShellState   *d_shells;

    /* Active lines (per-shell, flattened) */
    CudaActiveLine   *d_active_lines;
    int64_t          total_active_lines;

    /* Task Order #038-Revised: Macro-atom downbranch data */
    CudaDownbranchData *d_downbranch;
    int64_t          total_emission_entries;

    /* Simulation config (constant memory) */
    /* Use cudaMemcpyToSymbol for d_sim_config */

    /* RNG states (per-thread) */
    uint64_t         *d_rng_states;

    /* Output spectrum (atomicAdd target) */
    double           *d_spectrum;        /* L_nu bins */
    double           *d_spectrum_lumina; /* LUMINA-rotated L_nu */
    int64_t          *d_counts;          /* Packet counts per bin */

    /* Statistics (atomic counters) */
    int64_t          *d_n_escaped;
    int64_t          *d_n_absorbed;
    int64_t          *d_n_scattered;

    /* Memory sizes for cleanup */
    size_t           total_memory;
} CudaDeviceMemory;

/* ============================================================================
 * HOST-SIDE API
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize CUDA device and query properties
 */
int cuda_init_device(int device_id);

/**
 * Allocate and populate device memory from CPU atomic data
 */
int cuda_allocate_atomic_data(CudaDeviceMemory *mem,
                               const void *cpu_atomic_data,
                               const void *cpu_simulation_state);

/**
 * Copy partition function cache to constant memory
 */
int cuda_upload_partition_cache(const void *cpu_cache);

/**
 * Task Order #038-Revised: Upload macro-atom downbranch table
 */
int cuda_upload_downbranch_data(CudaDeviceMemory *mem, const void *cpu_atomic);

/**
 * Copy simulation configuration to constant memory
 */
int cuda_upload_sim_config(const CudaSimConfig *config);

/**
 * Allocate output arrays (spectrum, statistics)
 */
int cuda_allocate_output(CudaDeviceMemory *mem, int n_bins, int64_t n_packets);

/**
 * Initialize RNG states for all threads
 */
int cuda_init_rng(CudaDeviceMemory *mem, int64_t n_packets, uint64_t base_seed);

/**
 * Launch transport kernel
 */
int cuda_launch_transport(CudaDeviceMemory *mem, int64_t n_packets);

/**
 * Copy results back to host
 */
int cuda_download_results(const CudaDeviceMemory *mem,
                           double *h_spectrum,
                           double *h_spectrum_lumina,
                           int64_t *h_counts,
                           int64_t *stats);

/**
 * Free all device memory
 */
void cuda_free_memory(CudaDeviceMemory *mem);

/**
 * Get CUDA error string for last error
 */
const char* cuda_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ATOMIC_H */
