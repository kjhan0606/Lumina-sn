/**
 * LUMINA-SN CUDA Transport Kernel
 * cuda_transport.cu - GPU-accelerated Monte Carlo radiative transfer
 *
 * Architecture:
 *   - One thread per packet (persistent thread model)
 *   - Shared memory for shell property caching
 *   - Xorshift64* RNG for thread-local random numbers
 *   - atomicAdd for spectrum accumulation
 *   - LUMINA rotation integrated in kernel
 *
 * Memory hierarchy:
 *   - Constant memory: Physics constants, simulation config
 *   - Global memory: Atomic data, shell states, active lines
 *   - Shared memory: Current shell properties (T, ρ, n_e)
 *   - Registers: Packet state (r, μ, ν, energy)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include "cuda_atomic.h"

/* ============================================================================
 * CONSTANT MEMORY DECLARATIONS
 * ============================================================================ */

/**
 * Physics constants in constant memory - broadcast to all threads
 */
__constant__ CudaPhysicsConstants d_physics = {
    .c             = 2.99792458e10,     /* Speed of light [cm/s] */
    .h             = 6.62607015e-27,    /* Planck constant [erg s] */
    .k_B           = 1.380649e-16,      /* Boltzmann constant [erg/K] */
    .m_e           = 9.1093837015e-28,  /* Electron mass [g] */
    .e             = 4.80320425e-10,    /* Electron charge [esu] */
    .amu           = 1.66053906660e-24, /* Atomic mass unit [g] */
    .sigma_thomson = 6.6524587158e-25,  /* Thomson cross-section [cm²] */
    .sobolev_const = 2.6540281e-2,      /* π e² / (m_e c) [cm² s⁻¹] */
    .angstrom      = 1e-8,              /* Angstrom [cm] */
    .saha_const    = 2.4146853e15       /* (2π m_e k / h²)^(3/2) */
};

/**
 * Simulation configuration in constant memory
 */
__constant__ CudaSimConfig d_config;

/**
 * Partition function cache in constant memory
 * Note: May need texture memory for larger caches
 */
__constant__ CudaPartitionCache d_partition;

/* ============================================================================
 * XORSHIFT64* RANDOM NUMBER GENERATOR
 * ============================================================================
 *
 * Fast, high-quality PRNG suitable for Monte Carlo simulations.
 * Each thread maintains its own 64-bit state.
 *
 * Reference: Vigna, S. (2016). "An experimental exploration of Marsaglia's
 *            xorshift generators, scrambled"
 */

/**
 * Xorshift64* state advancement and random generation
 * Returns uniform random in [0, 1)
 */
__device__ __forceinline__
double rng_uniform(uint64_t *state)
{
    uint64_t x = *state;

    /* Xorshift64* algorithm */
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;

    /* Multiply by golden ratio and convert to double */
    uint64_t result = x * 0x2545F4914F6CDD1DULL;

    /* Convert to [0, 1) with full 53-bit precision */
    return (result >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * Initialize RNG state from packet index
 * Uses splitmix64 to generate uncorrelated starting states
 */
__device__ __forceinline__
uint64_t rng_init_seed(uint64_t base_seed, int64_t packet_idx)
{
    uint64_t z = base_seed + packet_idx * 0x9E3779B97F4A7C15ULL;

    /* Splitmix64 for seed initialization */
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);

    /* Ensure non-zero state */
    return z | 1ULL;
}

/* ============================================================================
 * SHARED MEMORY STRUCTURES
 * ============================================================================ */

/**
 * Shell cache for shared memory - reduces global memory traffic
 * One instance per block, shared by all threads
 */
struct ShellCache {
    double r_inner;
    double r_outer;
    double T;
    double rho;
    double n_e;
    double sigma_thomson_ne;
    int64_t n_active_lines;
    int64_t active_line_start;
};

/* ============================================================================
 * DEVICE HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Convert frequency to wavelength in Angstroms
 */
__device__ __forceinline__
double nu_to_angstrom(double nu)
{
    return d_physics.c / nu / d_physics.angstrom;
}

/**
 * Find shell index for given radius
 * Binary search on shell boundaries
 */
__device__
int find_shell_idx(const CudaShellState *shells, int n_shells, double r)
{
    /* Binary search for shell containing r */
    int lo = 0, hi = n_shells;

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (r < shells[mid].r_inner) {
            hi = mid;
        } else if (r >= shells[mid].r_outer) {
            lo = mid + 1;
        } else {
            return mid;  /* Found */
        }
    }

    return -1;  /* Outside grid */
}

/**
 * Partition function lookup with log-space interpolation
 */
__device__
double partition_function_lookup(int Z, int ion, double T)
{
    if (Z <= 0 || Z > CUDA_MAX_ELEMENTS || ion < 0 || ion > Z) {
        return 1.0;
    }

    /* Clamp to grid bounds */
    if (T <= d_partition.T_grid[0]) {
        return d_partition.U[Z][ion][0];
    }
    if (T >= d_partition.T_grid[CUDA_PARTITION_N_TEMPS - 1]) {
        return d_partition.U[Z][ion][CUDA_PARTITION_N_TEMPS - 1];
    }

    /* Find grid indices */
    double log_T = log(T);
    int idx = (int)((log_T - d_partition.log_T_min) / d_partition.d_log_T);
    if (idx < 0) idx = 0;
    if (idx >= CUDA_PARTITION_N_TEMPS - 1) idx = CUDA_PARTITION_N_TEMPS - 2;

    /* Linear interpolation in log-space */
    double T_lo = d_partition.T_grid[idx];
    double T_hi = d_partition.T_grid[idx + 1];
    double U_lo = d_partition.U[Z][ion][idx];
    double U_hi = d_partition.U[Z][ion][idx + 1];

    double w = (log_T - log(T_lo)) / (log(T_hi) - log(T_lo));

    /* Interpolate in log(U) for smoothness */
    if (U_lo > 0.0 && U_hi > 0.0) {
        return exp(log(U_lo) * (1.0 - w) + log(U_hi) * w);
    }
    return U_lo * (1.0 - w) + U_hi * w;
}

/**
 * Boltzmann level population fraction
 */
__device__
double boltzmann_fraction(double E_level, double T, double U)
{
    double kT = d_physics.k_B * T;
    double boltzmann = exp(-E_level / kT);
    return boltzmann / U;
}

/**
 * Distance to shell boundary
 * Solves quadratic for ray-sphere intersection
 */
__device__
double distance_to_boundary(double r, double mu, double r_inner, double r_outer,
                            int *delta_shell)
{
    /* Ray: r(s) = sqrt(r² + s² + 2rs·μ) */
    /* Intersection with sphere R: r² + s² + 2rs·μ = R² */
    /* Quadratic: s² + 2rs·μ + (r² - R²) = 0 */

    double d_inner = 1e99, d_outer = 1e99;

    /* Distance to inner boundary (if moving inward) */
    if (mu < 0.0) {
        double discriminant = r * r * (mu * mu - 1.0) + r_inner * r_inner;
        if (discriminant >= 0.0) {
            double sqrt_disc = sqrt(discriminant);
            double s = -r * mu - sqrt_disc;
            if (s > 0.0) d_inner = s;
        }
    }

    /* Distance to outer boundary */
    {
        double discriminant = r * r * (mu * mu - 1.0) + r_outer * r_outer;
        if (discriminant >= 0.0) {
            double sqrt_disc = sqrt(discriminant);
            double s = -r * mu + sqrt_disc;
            if (s > 0.0) d_outer = s;
        }
    }

    /* Return minimum distance and direction */
    if (d_inner < d_outer) {
        *delta_shell = -1;
        return d_inner;
    } else {
        *delta_shell = +1;
        return d_outer;
    }
}

/**
 * Find next line interaction using binary search
 * Returns distance to resonance and sets line_idx, tau_line
 */
__device__
double find_next_line(const CudaActiveLine *active_lines,
                      int64_t n_active, int64_t line_start,
                      double nu_cmf, double r, double t_exp,
                      int64_t *line_idx, double *tau_line)
{
    *line_idx = -1;
    *tau_line = 0.0;

    if (n_active == 0) return 1e99;

    /* Search window: ±1% in frequency */
    double nu_min = nu_cmf * 0.99;
    double nu_max = nu_cmf * 1.01;

    /* Binary search for first line with nu >= nu_min */
    int64_t lo = 0, hi = n_active;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (active_lines[line_start + mid].nu < nu_min) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int64_t first = lo;

    /* Find lines in range and select strongest */
    double max_tau = 0.0;
    int64_t best_idx = -1;

    for (int64_t i = first; i < n_active; i++) {
        double nu = active_lines[line_start + i].nu;
        if (nu > nu_max) break;

        double tau = active_lines[line_start + i].tau_sobolev;
        if (tau > max_tau) {
            max_tau = tau;
            best_idx = i;
        }
    }

    if (best_idx >= 0 && max_tau > 0.1) {
        *line_idx = active_lines[line_start + best_idx].line_idx;
        *tau_line = max_tau;

        /* Distance to resonance (simplified Sobolev) */
        double nu_line = active_lines[line_start + best_idx].nu;
        double delta_nu = fabs(nu_line - nu_cmf);

        if (delta_nu < 1e-10 * nu_cmf) {
            return 0.0;  /* At resonance */
        }

        double v = r / t_exp;
        return d_physics.c * delta_nu / (nu_cmf * v / r);
    }

    return 1e99;
}

/**
 * LUMINA rotation - transform packet to observer frame
 */
__device__
void lumina_rotate(double r, double mu, double nu, double energy,
                   double t_exp, double mu_obs,
                   double *nu_observer, double *weight)
{
    /* Velocity at emission point */
    double beta = r / (t_exp * d_physics.c);

    /* Doppler factor to observer */
    double doppler_obs = 1.0 - beta * mu_obs;

    /* Observer frame frequency */
    *nu_observer = nu * doppler_obs;

    /* Weight accounts for angular redistribution */
    /* For isotropic emission: weight = 1 */
    /* For limb darkening: weight = (1 + μ)/2 */
    *weight = 1.0;

    /* Relativistic beaming correction (first order) */
    if (beta > 0.01) {
        *weight *= (1.0 - beta * mu);
    }
}

/* ============================================================================
 * MAIN TRANSPORT KERNEL
 * ============================================================================ */

/**
 * Monte Carlo packet transport kernel
 *
 * Each thread processes one packet from emission to escape/absorption.
 * The full transport loop runs within a single thread.
 *
 * @param shells      Array of shell states
 * @param active_lines Flattened array of active lines (all shells)
 * @param lines       Global line data for frequency lookup
 * @param rng_states  Per-thread RNG states
 * @param spectrum    Output spectrum array (atomicAdd target)
 * @param spectrum_lumina LUMINA-rotated spectrum
 * @param counts      Packet counts per bin
 * @param n_escaped   Atomic counter for escaped packets
 * @param n_absorbed  Atomic counter for absorbed packets
 * @param n_scattered Atomic counter for scattering events
 * @param n_packets   Total number of packets to process
 */
__global__
void transport_packets_kernel(
    const CudaShellState  *shells,
    const CudaActiveLine  *active_lines,
    const CudaLineData    *lines,
    uint64_t              *rng_states,
    double                *spectrum,
    double                *spectrum_lumina,
    int64_t               *counts,
    int64_t               *n_escaped,
    int64_t               *n_absorbed,
    int64_t               *n_scattered,
    int64_t                n_packets)
{
    /* ========== Thread/Block identification ========== */
    int64_t packet_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (packet_idx >= n_packets) return;

    /* ========== Shared memory for shell caching ========== */
    __shared__ ShellCache shell_cache;
    __shared__ int current_shell_id;

    /* Initialize shell cache (first thread in block) */
    if (threadIdx.x == 0) {
        current_shell_id = -1;
    }
    __syncthreads();

    /* ========== Initialize RNG state ========== */
    uint64_t rng_state = rng_states[packet_idx];

    /* ========== Initialize packet state ========== */
    /* Emit from inner boundary with Planckian frequency distribution */
    double r_inner = shells[0].r_inner;
    double r = r_inner * 1.001;  /* Slightly inside first shell */

    /* Initial direction: outward (limb darkening) */
    double mu = sqrt(rng_uniform(&rng_state));
    if (mu < 0.01) mu = 0.01;  /* Ensure outward */

    /* Random frequency (uniform in log space) */
    double log_nu_min = log(d_config.nu_min);
    double log_nu_max = log(d_config.nu_max);
    double nu = exp(log_nu_min + rng_uniform(&rng_state) * (log_nu_max - log_nu_min));

    /* Packet energy (normalized) */
    double energy = d_config.packet_energy;

    /* Packet status: 0=in_process, 1=escaped, 2=absorbed */
    int status = 0;
    int n_interactions = 0;

    /* ========== Main transport loop ========== */
    for (int step = 0; step < d_config.max_steps && status == 0; step++) {

        /* Find current shell */
        int shell_id = find_shell_idx(shells, d_config.n_shells, r);

        if (shell_id < 0) {
            /* Outside grid */
            if (r >= shells[d_config.n_shells - 1].r_outer) {
                /* Escaped! */
                status = 1;

                /* Standard escape: only if nearly radial */
                if (mu > 0.99) {
                    int bin = (int)((nu - d_config.nu_min) / d_config.d_nu);
                    if (bin >= 0 && bin < d_config.n_bins) {
                        atomicAdd(&spectrum[bin], energy);
                        atomicAdd((unsigned long long *)&counts[bin], 1ULL);
                    }
                }

                /* LUMINA rotation: all packets contribute */
                if (d_config.enable_lumina) {
                    double nu_obs, weight;
                    lumina_rotate(r, mu, nu, energy, d_config.t_explosion,
                                  d_config.mu_observer, &nu_obs, &weight);

                    int bin = (int)((nu_obs - d_config.nu_min) / d_config.d_nu);
                    if (bin >= 0 && bin < d_config.n_bins) {
                        atomicAdd(&spectrum_lumina[bin], energy * weight);
                    }
                }
            } else {
                /* Fell below inner boundary - absorbed */
                status = 2;
            }
            break;
        }

        /* ========== Cache shell properties in shared memory ========== */
        /* Only update cache when shell changes (reduces divergence) */
        if (shell_id != current_shell_id) {
            __syncthreads();
            if (threadIdx.x == 0) {
                shell_cache.r_inner = shells[shell_id].r_inner;
                shell_cache.r_outer = shells[shell_id].r_outer;
                shell_cache.T = shells[shell_id].T;
                shell_cache.rho = shells[shell_id].rho;
                shell_cache.n_e = shells[shell_id].n_e;
                shell_cache.sigma_thomson_ne = shells[shell_id].sigma_thomson_ne;
                shell_cache.n_active_lines = shells[shell_id].n_active_lines;
                shell_cache.active_line_start = shells[shell_id].active_line_start;
                current_shell_id = shell_id;
            }
            __syncthreads();
        }

        /* ========== Calculate interaction distances ========== */

        /* Distance to shell boundary */
        int delta_shell;
        double d_boundary = distance_to_boundary(r, mu,
                                                  shell_cache.r_inner,
                                                  shell_cache.r_outer,
                                                  &delta_shell);

        /* Distance to electron scattering */
        double tau_e = -log(rng_uniform(&rng_state) + 1e-30);
        double d_electron = tau_e / (shell_cache.sigma_thomson_ne + 1e-30);

        /* Comoving frame frequency */
        double beta = r / (d_config.t_explosion * d_physics.c);
        double doppler = 1.0 - beta * mu;
        double nu_cmf = nu * doppler;

        /* Distance to line interaction */
        int64_t line_idx;
        double tau_line;
        double d_line = find_next_line(active_lines,
                                        shell_cache.n_active_lines,
                                        shell_cache.active_line_start,
                                        nu_cmf, r, d_config.t_explosion,
                                        &line_idx, &tau_line);

        /* ========== Determine interaction type ========== */
        double d_min = d_boundary;
        int interaction_type = 0;  /* 0=boundary, 1=electron, 2=line */

        if (d_electron < d_min) {
            d_min = d_electron;
            interaction_type = 1;
        }

        if (d_line < d_min && tau_line > 0.1) {
            d_min = d_line;
            interaction_type = 2;
        }

        /* Sanity check */
        if (d_min <= 0.0 || d_min > 1e50 || !isfinite(d_min)) {
            d_min = (shell_cache.r_outer - r) * 0.1;
            if (d_min <= 0.0) d_min = shell_cache.r_outer * 0.001;
            interaction_type = 0;
        }

        /* ========== Move packet ========== */
        double r_new = sqrt(r * r + d_min * d_min + 2.0 * r * d_min * mu);
        double mu_new = (r * mu + d_min) / r_new;

        r = r_new;
        mu = mu_new;

        /* ========== Process interaction ========== */
        switch (interaction_type) {
            case 0:  /* Boundary crossing - no action needed */
                break;

            case 1:  /* Electron scattering */
                /* Isotropic scattering in comoving frame */
                mu = 2.0 * rng_uniform(&rng_state) - 1.0;
                atomicAdd((unsigned long long *)n_scattered, 1ULL);
                n_interactions++;
                break;

            case 2:  /* Line interaction */
                if (line_idx >= 0 && tau_line > 0.1) {
                    /* Resonant scattering: isotropic re-emission */
                    mu = 2.0 * rng_uniform(&rng_state) - 1.0;

                    /* Update frequency to line frequency (lab frame) */
                    nu = lines->nu[line_idx] / doppler;

                    n_interactions++;
                }
                break;
        }

        /* Check interaction limit */
        if (n_interactions >= d_config.max_interactions) {
            status = 2;  /* Absorbed (too many interactions) */
        }
    }

    /* ========== Update statistics ========== */
    if (status == 1) {
        atomicAdd((unsigned long long *)n_escaped, 1ULL);
    } else {
        atomicAdd((unsigned long long *)n_absorbed, 1ULL);
    }

    /* Save RNG state for potential continuation */
    rng_states[packet_idx] = rng_state;
}

/* ============================================================================
 * RNG INITIALIZATION KERNEL
 * ============================================================================ */

__global__
void init_rng_kernel(uint64_t *rng_states, int64_t n_packets, uint64_t base_seed)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_packets) return;

    rng_states[idx] = rng_init_seed(base_seed, idx);
}

/* ============================================================================
 * HOST-SIDE KERNEL LAUNCH WRAPPERS
 * ============================================================================ */

extern "C" {

/**
 * Initialize CUDA device
 */
int cuda_init_device(int device_id)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to set device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║               LUMINA-SN CUDA INITIALIZATION                   ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Device:          %-42s ║\n", prop.name);
    printf("║  Compute:         %d.%d                                         ║\n",
           prop.major, prop.minor);
    printf("║  SMs:             %-42d ║\n", prop.multiProcessorCount);
    printf("║  Global Memory:   %-39.1f GB ║\n",
           prop.totalGlobalMem / 1e9);
    printf("║  Shared/Block:    %-39zu KB ║\n",
           prop.sharedMemPerBlock / 1024);
    printf("║  Warp Size:       %-42d ║\n", prop.warpSize);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return 0;
}

/**
 * Upload simulation configuration to constant memory
 */
int cuda_upload_sim_config(const CudaSimConfig *config)
{
    cudaError_t err = cudaMemcpyToSymbol(d_config, config, sizeof(CudaSimConfig));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to upload config: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Allocate output arrays
 */
int cuda_allocate_output(CudaDeviceMemory *mem, int n_bins, int64_t n_packets)
{
    cudaError_t err;

    /* Spectrum arrays */
    err = cudaMalloc(&mem->d_spectrum, n_bins * sizeof(double));
    if (err != cudaSuccess) return -1;
    cudaMemset(mem->d_spectrum, 0, n_bins * sizeof(double));

    err = cudaMalloc(&mem->d_spectrum_lumina, n_bins * sizeof(double));
    if (err != cudaSuccess) return -1;
    cudaMemset(mem->d_spectrum_lumina, 0, n_bins * sizeof(double));

    err = cudaMalloc(&mem->d_counts, n_bins * sizeof(int64_t));
    if (err != cudaSuccess) return -1;
    cudaMemset(mem->d_counts, 0, n_bins * sizeof(int64_t));

    /* Statistics counters */
    err = cudaMalloc(&mem->d_n_escaped, sizeof(int64_t));
    if (err != cudaSuccess) return -1;
    cudaMemset(mem->d_n_escaped, 0, sizeof(int64_t));

    err = cudaMalloc(&mem->d_n_absorbed, sizeof(int64_t));
    if (err != cudaSuccess) return -1;
    cudaMemset(mem->d_n_absorbed, 0, sizeof(int64_t));

    err = cudaMalloc(&mem->d_n_scattered, sizeof(int64_t));
    if (err != cudaSuccess) return -1;
    cudaMemset(mem->d_n_scattered, 0, sizeof(int64_t));

    /* RNG states */
    err = cudaMalloc(&mem->d_rng_states, n_packets * sizeof(uint64_t));
    if (err != cudaSuccess) return -1;

    mem->total_memory += n_bins * sizeof(double) * 2;
    mem->total_memory += n_bins * sizeof(int64_t);
    mem->total_memory += 3 * sizeof(int64_t);
    mem->total_memory += n_packets * sizeof(uint64_t);

    return 0;
}

/**
 * Initialize RNG states
 */
int cuda_init_rng(CudaDeviceMemory *mem, int64_t n_packets, uint64_t base_seed)
{
    int blocks = (n_packets + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    init_rng_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        mem->d_rng_states, n_packets, base_seed
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: RNG init failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Launch transport kernel
 */
int cuda_launch_transport(CudaDeviceMemory *mem, int64_t n_packets)
{
    int blocks = (n_packets + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
    if (blocks > CUDA_MAX_BLOCKS) blocks = CUDA_MAX_BLOCKS;

    printf("[CUDA] Launching transport: %d blocks × %d threads = %d concurrent\n",
           blocks, CUDA_THREADS_PER_BLOCK, blocks * CUDA_THREADS_PER_BLOCK);

    /* Create events for timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    /* Launch kernel */
    transport_packets_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        mem->d_shells,
        mem->d_active_lines,
        mem->d_lines,
        mem->d_rng_states,
        mem->d_spectrum,
        mem->d_spectrum_lumina,
        mem->d_counts,
        mem->d_n_escaped,
        mem->d_n_absorbed,
        mem->d_n_scattered,
        n_packets
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("[CUDA] Transport complete: %.2f ms (%.0f packets/sec)\n",
           elapsed_ms, n_packets / (elapsed_ms / 1000.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Transport kernel failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Download results to host
 */
int cuda_download_results(const CudaDeviceMemory *mem,
                           double *h_spectrum,
                           double *h_spectrum_lumina,
                           int64_t *h_counts,
                           int64_t *stats)
{
    cudaError_t err;

    /* Download spectrum */
    err = cudaMemcpy(h_spectrum, mem->d_spectrum,
                     CUDA_SPECTRUM_N_BINS * sizeof(double),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    err = cudaMemcpy(h_spectrum_lumina, mem->d_spectrum_lumina,
                     CUDA_SPECTRUM_N_BINS * sizeof(double),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    err = cudaMemcpy(h_counts, mem->d_counts,
                     CUDA_SPECTRUM_N_BINS * sizeof(int64_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    /* Download statistics */
    err = cudaMemcpy(&stats[0], mem->d_n_escaped, sizeof(int64_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    err = cudaMemcpy(&stats[1], mem->d_n_absorbed, sizeof(int64_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    err = cudaMemcpy(&stats[2], mem->d_n_scattered, sizeof(int64_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -1;

    return 0;
}

/**
 * Free all device memory
 */
void cuda_free_memory(CudaDeviceMemory *mem)
{
    if (mem->d_elements) cudaFree(mem->d_elements);
    if (mem->d_ions) cudaFree(mem->d_ions);
    if (mem->d_levels) cudaFree(mem->d_levels);
    if (mem->d_lines) cudaFree(mem->d_lines);
    if (mem->d_shells) cudaFree(mem->d_shells);
    if (mem->d_active_lines) cudaFree(mem->d_active_lines);
    if (mem->d_rng_states) cudaFree(mem->d_rng_states);
    if (mem->d_spectrum) cudaFree(mem->d_spectrum);
    if (mem->d_spectrum_lumina) cudaFree(mem->d_spectrum_lumina);
    if (mem->d_counts) cudaFree(mem->d_counts);
    if (mem->d_n_escaped) cudaFree(mem->d_n_escaped);
    if (mem->d_n_absorbed) cudaFree(mem->d_n_absorbed);
    if (mem->d_n_scattered) cudaFree(mem->d_n_scattered);

    memset(mem, 0, sizeof(CudaDeviceMemory));
}

/**
 * Get last CUDA error string
 */
const char* cuda_get_last_error(void)
{
    return cudaGetErrorString(cudaGetLastError());
}

} /* extern "C" */
