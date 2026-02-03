/**
 * LUMINA-SN CUDA Shared Definitions
 * cuda_shared.h - GPU-friendly structures usable by both C and CUDA
 *
 * Task Order #020: CUDA Data Structures & Kernel Porting
 *
 * Design Principles:
 *   1. NO POINTERS in structs (simple cudaMemcpy compatibility)
 *   2. Large arrays passed as separate kernel arguments
 *   3. Structs contain only scalar parameters
 *   4. Identical layout for C and CUDA compilation
 *
 * Usage:
 *   - C code: #include "cuda_shared.h"
 *   - CUDA code: #include "cuda_shared.h"
 *   - Both see the same struct definitions
 */

#ifndef CUDA_SHARED_H
#define CUDA_SHARED_H

#include <stdint.h>

/* ============================================================================
 * CUDA COMPATIBILITY MACROS
 * ============================================================================
 * These macros allow the same code to compile in both C and CUDA.
 */

#ifdef __CUDACC__
    /* Compiling with nvcc */
    #define CUDA_CALLABLE __host__ __device__
    #define CUDA_DEVICE   __device__
    #define CUDA_HOST     __host__
    #define CUDA_GLOBAL   __global__
    #define CUDA_CONSTANT __constant__
#else
    /* Compiling with gcc/clang */
    #define CUDA_CALLABLE
    #define CUDA_DEVICE
    #define CUDA_HOST
    #define CUDA_GLOBAL
    #define CUDA_CONSTANT
#endif

/* ============================================================================
 * PHYSICAL CONSTANTS (Matching TARDIS exactly)
 * ============================================================================ */

#define GPU_C_SPEED_OF_LIGHT   2.99792458e10    /* Speed of light [cm/s] */
#define GPU_SIGMA_THOMSON      6.6524587158e-25 /* Thomson cross-section [cm²] */
#define GPU_MISS_DISTANCE      1e99             /* No interaction possible */
#define GPU_CLOSE_LINE_THRESHOLD 1e-7           /* Numerical stability threshold */

/* ============================================================================
 * PACKET STATUS ENUMS
 * ============================================================================ */

typedef enum {
    GPU_PACKET_IN_PROCESS = 0,
    GPU_PACKET_EMITTED    = 1,
    GPU_PACKET_REABSORBED = 2
} GPUPacketStatus;

typedef enum {
    GPU_INTERACTION_BOUNDARY    = 1,
    GPU_INTERACTION_LINE        = 2,
    GPU_INTERACTION_ESCATTERING = 3
} GPUInteractionType;

typedef enum {
    GPU_LINE_SCATTER    = 0,
    GPU_LINE_DOWNBRANCH = 1,
    GPU_LINE_MACROATOM  = 2
} GPULineInteractionType;

/* ============================================================================
 * RPacket_GPU: GPU-friendly packet structure
 * ============================================================================
 *
 * This struct is designed for efficient GPU memory access:
 *   - All scalar values (no pointers)
 *   - Aligned for coalesced memory access
 *   - RNG state embedded (thread-safe)
 */

typedef struct {
    /* === Primary Phase-Space Coordinates === */
    double r;           /* Radial position [cm] */
    double mu;          /* Direction cosine [-1, +1] */
    double nu;          /* Photon frequency [Hz] */
    double energy;      /* Monte Carlo weight [erg] */

    /* === Discrete State Variables === */
    int64_t next_line_id;       /* Index into line_list_nu array */
    int64_t current_shell_id;   /* Radial zone index [0, n_shells-1] */
    int64_t status;             /* GPUPacketStatus */

    /* === RNG State (Xorshift64*) === */
    uint64_t rng_state;         /* Thread-local RNG state */

    /* === Packet Metadata === */
    int64_t index;              /* Packet ID in ensemble */

    /* === Interaction History === */
    int64_t last_interaction_type;
    double  last_interaction_in_nu;
    int64_t last_line_interaction_in_id;
    int64_t last_line_interaction_out_id;

} RPacket_GPU;

/* Note: RPacket_GPU is approximately 120 bytes (15 x 8-byte fields)
 * No forced size constraint for flexibility */

/* ============================================================================
 * Model_GPU: Simulation geometry (scalars only)
 * ============================================================================
 *
 * Arrays (r_inner, r_outer) are passed separately to the kernel.
 * This struct contains only scalar parameters.
 */

typedef struct {
    double time_explosion;  /* Time since explosion [s] */
    int64_t n_shells;       /* Number of radial zones */

    /* Derived quantities (precomputed on host) */
    double inv_time_explosion;  /* 1.0 / time_explosion */
    double ct;                  /* c * time_explosion */

    /* Boundary radii (for quick checks) */
    double r_inner_boundary;    /* Inner boundary of first shell */
    double r_outer_boundary;    /* Outer boundary of last shell */

    /* Relativity mode */
    int enable_full_relativity;
    int _padding;

} Model_GPU;

/* ============================================================================
 * Plasma_GPU: Plasma parameters (scalars only)
 * ============================================================================
 *
 * Arrays passed separately:
 *   - line_list_nu: Sorted line frequencies [Hz], size: n_lines
 *   - tau_sobolev: Optical depths, size: [n_lines x n_shells] (row-major)
 *   - electron_density: Free electron density per shell, size: n_shells
 */

typedef struct {
    int64_t n_lines;    /* Total number of spectral lines */
    int64_t n_shells;   /* Number of radial zones */

    /* Line interaction mode */
    int line_interaction_type;  /* GPULineInteractionType */
    int disable_line_scattering;
    int disable_scattered_peeling;  /* Task #036 v3: Diagnostic mode */

    /* Precomputed values */
    double nu_min;      /* Minimum line frequency */
    double nu_max;      /* Maximum line frequency */

} Plasma_GPU;

/* ============================================================================
 * MonteCarloConfig_GPU: Runtime configuration
 * ============================================================================ */

typedef struct {
    int enable_full_relativity;
    int disable_line_scattering;
    int line_interaction_type;
    int max_iterations;         /* Safety limit on transport loop */

    /* Spectrum binning */
    int n_wavelength_bins;
    double wavelength_min;
    double wavelength_max;
    double d_wavelength;

    /* Observer angle (for LUMINA rotation) */
    double mu_observer;

} MonteCarloConfig_GPU;

/* ============================================================================
 * GPU KERNEL STATISTICS
 * ============================================================================ */

typedef struct {
    int64_t n_emitted;          /* Packets that escaped */
    int64_t n_reabsorbed;       /* Packets absorbed at inner boundary */
    int64_t n_line_interactions;
    int64_t n_electron_scatters;
    int64_t n_boundary_crossings;
    int64_t n_iterations_total; /* Total loop iterations */
} GPUStats;

/* ============================================================================
 * SPECTRUM_GPU: Peeling-off (Virtual Packet) Spectrum Accumulator
 * ============================================================================
 *
 * Task Order #024: GPU Peeling-off Implementation
 *
 * This struct accumulates flux contributions using atomicAdd.
 * Each interaction point contributes a "virtual packet" toward the observer.
 */

/* Task Order #026: Widen spectrum range to catch all optical + edge cases */
#define GPU_SPECTRUM_NBINS 2000     /* Number of wavelength bins (higher resolution) */
#define GPU_SPECTRUM_WL_MIN 1000.0  /* Minimum wavelength [Angstrom] - catches UV */
#define GPU_SPECTRUM_WL_MAX 25000.0 /* Maximum wavelength [Angstrom] - catches near-IR */

typedef struct {
    /* Flux bins (use atomicAdd in kernel) */
    double flux[GPU_SPECTRUM_NBINS];

    /* Statistics */
    int64_t n_contributions;        /* Total peeling contributions */
    int64_t n_line_contributions;   /* From line interactions */
    int64_t n_escat_contributions;  /* From electron scattering */

    /* Task Order #026: Debug counters for rejection analysis */
    int64_t n_line_peeling_calls;      /* Total line peeling attempts */
    int64_t n_line_wl_rejected;        /* Rejected: wavelength out of range */
    int64_t n_line_escape_rejected;    /* Rejected: escape probability too low */
    int64_t n_escat_peeling_calls;     /* Total e-scat peeling attempts */
    int64_t n_escat_wl_rejected;       /* Rejected: wavelength out of range */

    /* Task Order #038-Revised: Line opacity debug counters */
    int64_t n_line_tau_calls;          /* Calls to calculate_line_tau_on_ray */
    int64_t n_line_tau_found;          /* Times tau > threshold was found */
    double sum_tau_max;                /* Sum of tau_max values */

    /* Binning parameters (precomputed) */
    double wl_min;
    double wl_max;
    double d_wl;                    /* Bin width in Angstrom */
    double inv_d_wl;                /* 1/d_wl for fast binning */
} Spectrum_GPU;

/* ============================================================================
 * MACRO-ATOM GPU DATA STRUCTURES
 * ============================================================================
 *
 * Task Order #024: GPU Macro-Atom Implementation
 *
 * These structures enable fluorescence and thermalization physics on GPU.
 * Ported from macro_atom.h with GPU-friendly fixed-size arrays.
 */

/* Maximum transitions from any single level (GPU constraint) */
#define GPU_MACRO_ATOM_MAX_TRANS 128

/* Maximum jumps in transition loop (safety limit) */
#define GPU_MACRO_ATOM_MAX_JUMPS 50

/**
 * MacroAtomTransition_GPU: Single transition in macro-atom cascade
 */
typedef struct {
    int16_t source_level;           /* Source level number */
    int16_t dest_level;             /* Destination level number */
    int8_t  transition_type;        /* -1=radiative, 0=down internal, +1=up internal */
    int8_t  atomic_number;
    int8_t  ion_number;
    int8_t  _pad;
    int64_t line_id;                /* Line index (if radiative, else -1) */
    double  A_ul;                   /* Spontaneous emission rate [s^-1] */
    double  nu;                     /* Transition frequency [Hz] */
} MacroAtomTransition_GPU;

/**
 * MacroAtomReference_GPU: Index into transition array for a level
 */
typedef struct {
    int8_t  atomic_number;
    int8_t  ion_number;
    int16_t level_number;
    int32_t n_transitions;          /* Total transitions from this level */
    int32_t trans_start_idx;        /* Start index in transitions array */
} MacroAtomReference_GPU;

/**
 * MacroAtomData_GPU: All macro-atom data on device
 */
typedef struct {
    int32_t n_transitions;          /* Total transition count */
    int32_t n_references;           /* Number of level references */
    /* Actual arrays stored separately on device */
} MacroAtomData_GPU;

/**
 * Line_GPU: Simplified line data for macro-atom
 */
typedef struct {
    double nu;                      /* Line frequency [Hz] */
    double A_ul;                    /* Einstein A coefficient [s^-1] */
    double f_lu;                    /* Oscillator strength */
    int8_t atomic_number;
    int8_t ion_number;
    int16_t level_upper;
    int16_t level_lower;
    int16_t _pad;
} Line_GPU;

/**
 * Level_GPU: Level data for macro-atom cascade
 */
typedef struct {
    int8_t atomic_number;
    int8_t ion_number;
    int16_t level_number;
    int16_t g;                      /* Statistical weight */
    int16_t _pad;
    double energy;                  /* Level energy [erg] */
} Level_GPU;

/**
 * MacroAtomTuning_GPU: Tunable parameters for macro-atom physics
 */
typedef struct {
    double thermalization_epsilon;   /* Base thermalization probability (0.35) */
    double ir_thermalization_boost;  /* IR photons thermalization (0.80) */
    double ir_wavelength_threshold;  /* IR threshold in Angstrom (7000) */
    double uv_scatter_boost;         /* UV scatter multiplier (0.5) */
    double uv_wavelength_threshold;  /* UV threshold in Angstrom (3500) */
    double collisional_boost;        /* Collision rate boost (10.0) */
    double gaunt_factor_scale;       /* Gaunt factor multiplier (5.0) */
    int    downbranch_only;          /* Skip cascade, direct emission (0) */
} MacroAtomTuning_GPU;

/* ============================================================================
 * XORSHIFT64* RNG (CUDA-compatible)
 * ============================================================================
 *
 * Identical algorithm to CPU version for reproducibility.
 */

CUDA_CALLABLE static inline uint64_t gpu_rng_xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

CUDA_CALLABLE static inline double gpu_rng_uniform(uint64_t *state) {
    return (gpu_rng_xorshift64star(state) >> 11) * (1.0 / 9007199254740992.0);
}

CUDA_CALLABLE static inline void gpu_rng_init(uint64_t *state, uint64_t seed) {
    *state = seed ? seed : 0x853c49e6748fea9bULL;
    gpu_rng_xorshift64star(state);
    gpu_rng_xorshift64star(state);
}

/* ============================================================================
 * FRAME TRANSFORMATIONS (CUDA-compatible)
 * ============================================================================ */

/**
 * get_doppler_factor_gpu: Transform frequency from Lab → Comoving frame
 */
CUDA_CALLABLE static inline double get_doppler_factor_gpu(
    double r, double mu, double inv_t, int full_relativity)
{
    double beta = r * inv_t / GPU_C_SPEED_OF_LIGHT;

    if (!full_relativity) {
        return 1.0 - mu * beta;
    } else {
        return (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
    }
}

/**
 * get_inverse_doppler_factor_gpu: Transform frequency from Comoving → Lab frame
 */
CUDA_CALLABLE static inline double get_inverse_doppler_factor_gpu(
    double r, double mu, double inv_t, int full_relativity)
{
    double beta = r * inv_t / GPU_C_SPEED_OF_LIGHT;

    if (!full_relativity) {
        return 1.0 / (1.0 - mu * beta);
    } else {
        return (1.0 + mu * beta) / sqrt(1.0 - beta * beta);
    }
}

/**
 * angle_aberration_CMF_to_LF_gpu: Convert direction from Comoving → Lab frame
 */
CUDA_CALLABLE static inline double angle_aberration_CMF_to_LF_gpu(
    double r, double mu_cmf, double ct)
{
    double beta = r / ct;
    return (mu_cmf + beta) / (1.0 + beta * mu_cmf);
}

/**
 * angle_aberration_LF_to_CMF_gpu: Convert direction from Lab → Comoving frame
 */
CUDA_CALLABLE static inline double angle_aberration_LF_to_CMF_gpu(
    double r, double mu_lab, double ct)
{
    double beta = r / ct;
    return (mu_lab - beta) / (1.0 - beta * mu_lab);
}

/* ============================================================================
 * DISTANCE CALCULATIONS (CUDA-compatible)
 * ============================================================================ */

/**
 * calculate_distance_boundary_gpu: Distance to shell boundary [cm]
 */
CUDA_CALLABLE static inline double calculate_distance_boundary_gpu(
    double r, double mu, double r_inner, double r_outer, int *delta_shell)
{
    double distance;
    double mu_sq_minus_1 = mu * mu - 1.0;

    if (mu > 0.0) {
        double discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r;
        distance = sqrt(discriminant) - r * mu;
        *delta_shell = 1;
    } else {
        double check = r_inner * r_inner + r * r * mu_sq_minus_1;

        if (check >= 0.0) {
            distance = -r * mu - sqrt(check);
            *delta_shell = -1;
        } else {
            double discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r;
            distance = sqrt(discriminant) - r * mu;
            *delta_shell = 1;
        }
    }

    return distance;
}

/**
 * calculate_distance_electron_gpu: Distance to Thomson scattering event [cm]
 */
CUDA_CALLABLE static inline double calculate_distance_electron_gpu(
    double electron_density, double tau_event)
{
    return tau_event / (electron_density * GPU_SIGMA_THOMSON);
}

/**
 * calculate_distance_line_gpu: Distance to line resonance [cm]
 */
CUDA_CALLABLE static inline double calculate_distance_line_gpu(
    double nu, double comov_nu, int is_last_line, double nu_line,
    double ct, double r, double mu, int full_relativity)
{
    if (is_last_line) {
        return GPU_MISS_DISTANCE;
    }

    double nu_diff = comov_nu - nu_line;

    if (fabs(nu_diff / nu) < GPU_CLOSE_LINE_THRESHOLD) {
        nu_diff = 0.0;
    }

    if (nu_diff < 0.0) {
        return GPU_MISS_DISTANCE;
    }

    if (!full_relativity) {
        return (nu_diff / nu) * ct;
    } else {
        double nu_r = nu_line / nu;
        double nu_r_sq = nu_r * nu_r;
        double sin_sq = 1.0 - mu * mu;
        double discriminant = ct * ct - (1.0 + r * r * sin_sq * (1.0 + 1.0 / nu_r_sq));

        if (discriminant < 0.0) {
            return GPU_MISS_DISTANCE;
        }

        return -mu * r + (ct - nu_r_sq * sqrt(discriminant)) / (1.0 + nu_r_sq);
    }
}

/**
 * calculate_tau_electron_gpu: Thomson optical depth for given path
 */
CUDA_CALLABLE static inline double calculate_tau_electron_gpu(
    double electron_density, double distance)
{
    return electron_density * GPU_SIGMA_THOMSON * distance;
}

/* ============================================================================
 * PACKET MOVEMENT (CUDA-compatible)
 * ============================================================================ */

/**
 * move_packet_gpu: Update packet position after traveling distance d
 *
 * Geometric update:
 *   r_new = sqrt(r² + d² + 2rd·μ)
 *   μ_new = (μr + d) / r_new
 */
CUDA_CALLABLE static inline void move_packet_gpu(
    double *r, double *mu, double distance)
{
    double r_old = *r;
    double mu_old = *mu;

    double r_new = sqrt(r_old * r_old + distance * distance +
                        2.0 * r_old * distance * mu_old);
    double mu_new = (mu_old * r_old + distance) / r_new;

    *r = r_new;
    *mu = mu_new;
}

/* ============================================================================
 * HOST-SIDE HELPER FUNCTIONS
 * ============================================================================ */

#ifndef __CUDACC__

#include <stdlib.h>
#include <string.h>

/**
 * Convert CPU RPacket to GPU RPacket
 */
static inline void rpacket_to_gpu(RPacket_GPU *gpu_pkt, const void *cpu_pkt_void) {
    /* Assuming cpu_pkt is from rpacket.h */
    const double *cpu = (const double *)cpu_pkt_void;

    /* Manual copy - RPacket layout may differ */
    gpu_pkt->r = cpu[0];
    gpu_pkt->mu = cpu[1];
    gpu_pkt->nu = cpu[2];
    gpu_pkt->energy = cpu[3];
    /* ... other fields require proper struct access ... */
}

/**
 * Convert CPU Model to GPU Model
 */
static inline void model_to_gpu(Model_GPU *gpu_model,
                                 double time_explosion,
                                 int64_t n_shells,
                                 double r_inner_boundary,
                                 double r_outer_boundary,
                                 int enable_full_relativity)
{
    gpu_model->time_explosion = time_explosion;
    gpu_model->n_shells = n_shells;
    gpu_model->inv_time_explosion = 1.0 / time_explosion;
    gpu_model->ct = GPU_C_SPEED_OF_LIGHT * time_explosion;
    gpu_model->r_inner_boundary = r_inner_boundary;
    gpu_model->r_outer_boundary = r_outer_boundary;
    gpu_model->enable_full_relativity = enable_full_relativity;
}

/**
 * Convert CPU Plasma to GPU Plasma
 */
static inline void plasma_to_gpu(Plasma_GPU *gpu_plasma,
                                  int64_t n_lines,
                                  int64_t n_shells,
                                  int line_interaction_type,
                                  int disable_line_scattering,
                                  double nu_min,
                                  double nu_max)
{
    gpu_plasma->n_lines = n_lines;
    gpu_plasma->n_shells = n_shells;
    gpu_plasma->line_interaction_type = line_interaction_type;
    gpu_plasma->disable_line_scattering = disable_line_scattering;
    gpu_plasma->disable_scattered_peeling = 0;  /* Task #036 v3: Default off */
    gpu_plasma->nu_min = nu_min;
    gpu_plasma->nu_max = nu_max;
}

#endif /* !__CUDACC__ */

#endif /* CUDA_SHARED_H */
