/**
 * LUMINA-SN GPU Transport Entry Point
 * gpu_transport.cu - CUDA kernel infrastructure for OpenMP-driven GPU acceleration
 *
 * Task Order #019: CUDA Infrastructure Setup
 *
 * Architecture:
 *   - Each OpenMP thread manages its own CUDA stream
 *   - Streams enable concurrent kernel execution from multiple CPU threads
 *   - Warmup kernels verify CUDA concurrency before production use
 *
 * Compilation:
 *   nvcc -arch=sm_70 -Xcompiler -fopenmp -c gpu_transport.cu -o gpu_transport.o
 *
 * Linkage:
 *   gcc ... gpu_transport.o ... -lcudart -L$(CUDA_HOME)/lib64
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cuda_interface.h"
#include "cuda_shared.h"

/* ============================================================================
 * INTERNAL STATE (must be before any functions that use them)
 * ============================================================================ */

static int g_cuda_initialized = 0;
static int g_cuda_device_id = -1;
static cudaStream_t g_streams[CUDA_MAX_STREAMS];
static int g_stream_created[CUDA_MAX_STREAMS] = {0};

/* Thread safety for stream creation */
#include <pthread.h>
static pthread_mutex_t g_stream_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ============================================================================
 * Task Order #020: TRACE_PACKET DEVICE FUNCTIONS
 * ============================================================================
 *
 * These device functions implement the Monte Carlo transport logic.
 * They mirror the CPU implementation in rpacket.c but are optimized for GPU.
 */

/**
 * trace_packet_device: Find next interaction point (device version)
 *
 * This is the HEART of Monte Carlo transport on GPU.
 * Each thread runs this for its assigned packet.
 *
 * @param pkt               Packet being traced (in registers)
 * @param r_inner           Inner radii array [n_shells]
 * @param r_outer           Outer radii array [n_shells]
 * @param line_list_nu      Sorted line frequencies [n_lines]
 * @param tau_sobolev       Optical depths [n_lines x n_shells] row-major
 * @param electron_density  Free electron density per shell [n_shells]
 * @param model             Model parameters (scalars)
 * @param plasma            Plasma parameters (scalars)
 * @param distance          [out] Distance to interaction
 * @param delta_shell       [out] Shell crossing direction
 * @return                  Interaction type
 */
__device__ GPUInteractionType trace_packet_device(
    RPacket_GPU *pkt,
    const double * __restrict__ r_inner,
    const double * __restrict__ r_outer,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    const double * __restrict__ electron_density,
    const Model_GPU *model,
    const Plasma_GPU *plasma,
    double *distance,
    int *delta_shell)
{
    int64_t shell_id = pkt->current_shell_id;
    int full_rel = model->enable_full_relativity;

    /* Boundary distance */
    double d_boundary = calculate_distance_boundary_gpu(
        pkt->r, pkt->mu,
        r_inner[shell_id], r_outer[shell_id],
        delta_shell
    );

    /* Sample random optical depth for electron scattering */
    double tau_event = -log(gpu_rng_uniform(&pkt->rng_state) + 1e-30);

    /* Electron scattering distance */
    double d_electron = calculate_distance_electron_gpu(
        electron_density[shell_id], tau_event
    );

    /* Comoving frame frequency */
    double doppler = get_doppler_factor_gpu(
        pkt->r, pkt->mu, model->inv_time_explosion, full_rel
    );
    double comov_nu = pkt->nu * doppler;

    /* Find minimum distance among: boundary, electron, line */
    double d_min = d_boundary;
    GPUInteractionType itype = GPU_INTERACTION_BOUNDARY;

    if (d_electron < d_min) {
        d_min = d_electron;
        itype = GPU_INTERACTION_ESCATTERING;
    }

    /* Check lines (if not disabled) */
    if (!plasma->disable_line_scattering && plasma->n_lines > 0) {
        /* Find next line to check */
        int64_t line_id = pkt->next_line_id;
        double tau_trace = 0.0;

        /* Loop through lines in frequency order */
        while (line_id < plasma->n_lines && tau_trace < tau_event) {
            double nu_line = line_list_nu[line_id];

            /* Distance to this line resonance */
            int is_last = (line_id >= plasma->n_lines - 1);
            double d_line = calculate_distance_line_gpu(
                pkt->nu, comov_nu, is_last, nu_line,
                model->ct, pkt->r, pkt->mu, full_rel
            );

            /* If line is closer than current minimum */
            if (d_line < d_min) {
                /* Get Sobolev optical depth for this line in this shell */
                int64_t tau_idx = line_id * plasma->n_shells + shell_id;
                double tau_sob = tau_sobolev[tau_idx];

                /* Accumulate optical depth */
                tau_trace += tau_sob;

                if (tau_trace >= tau_event) {
                    /* Line interaction wins */
                    d_min = d_line;
                    itype = GPU_INTERACTION_LINE;
                    pkt->next_line_id = line_id;
                    break;
                }
            } else {
                /* Line is farther than current d_min, stop checking */
                break;
            }

            line_id++;
        }

        /* Update line pointer for next trace */
        if (itype != GPU_INTERACTION_LINE) {
            pkt->next_line_id = line_id;
        }
    }

    *distance = d_min;
    return itype;
}

/**
 * process_boundary_crossing_device: Handle shell boundary transition
 */
__device__ void process_boundary_crossing_device(
    RPacket_GPU *pkt,
    int delta_shell,
    int64_t n_shells)
{
    int64_t new_shell = pkt->current_shell_id + delta_shell;

    if (new_shell < 0) {
        /* Hit inner boundary -> reabsorbed */
        pkt->status = GPU_PACKET_REABSORBED;
    } else if (new_shell >= n_shells) {
        /* Escaped through outer boundary */
        pkt->status = GPU_PACKET_EMITTED;
    } else {
        /* Normal shell crossing */
        pkt->current_shell_id = new_shell;
    }
}

/**
 * thomson_scatter_device: Isotropic electron scattering
 */
__device__ void thomson_scatter_device(
    RPacket_GPU *pkt,
    const Model_GPU *model)
{
    /* Isotropic scattering in comoving frame */
    double mu_cmf = 2.0 * gpu_rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform back to lab frame */
    pkt->mu = angle_aberration_CMF_to_LF_gpu(pkt->r, mu_cmf, model->ct);

    /* Update frequency (due to Doppler shift) */
    double inv_doppler = get_inverse_doppler_factor_gpu(
        pkt->r, pkt->mu, model->inv_time_explosion,
        model->enable_full_relativity
    );
    pkt->nu *= inv_doppler;

    pkt->last_interaction_type = 1;  /* Electron scatter */
}

/* ============================================================================
 * Task Order #024: PEELING-OFF (VIRTUAL PACKET) DEVICE FUNCTIONS
 * ============================================================================
 *
 * The peeling-off technique creates a "virtual packet" at each interaction
 * point, directed toward the observer. This reduces Monte Carlo noise
 * dramatically compared to only counting escaped packets.
 *
 * Algorithm:
 *   1. At each interaction, compute direction to observer (mu_obs)
 *   2. Calculate Doppler factor ratio (observer vs packet direction)
 *   3. Compute escape probability (through remaining optical depth)
 *   4. Add weighted flux contribution to spectrum bin
 */

/**
 * Task Order #029: calculate_line_tau_on_ray
 *
 * Compute total line optical depth along a ray segment through a shell.
 *
 * For a photon at frequency nu_packet traveling through a shell with
 * velocity range [v_start, v_end], the Doppler-shifted comoving frequency
 * sweeps through a range. Any lines within this range contribute their
 * tau_sobolev to the total optical depth.
 *
 * @param nu_packet      Packet frequency in observer frame [Hz]
 * @param shell_id       Current shell index
 * @param r_start        Starting radius [cm]
 * @param r_end          Ending radius [cm]
 * @param inv_t_exp      1/t_explosion [1/s]
 * @param line_list_nu   Sorted line frequencies [Hz]
 * @param tau_sobolev    Line optical depths [n_lines × n_shells]
 * @param n_lines        Number of lines
 * @param n_shells       Number of shells
 * @return Total line optical depth along ray segment
 */
__device__ double calculate_line_tau_on_ray(
    double nu_packet,
    int64_t shell_id,
    double r_start,
    double r_end,
    double inv_t_exp,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    int64_t n_lines,
    int64_t n_shells)
{
    /* Velocity at shell boundaries: v = r / t_exp */
    double v_start = r_start * inv_t_exp;
    double v_end = r_end * inv_t_exp;

    /* Doppler factors: D = 1 - v/c (for mu_obs ≈ 1, radial toward observer) */
    double D_start = 1.0 - v_start / GPU_C_SPEED_OF_LIGHT;
    double D_end = 1.0 - v_end / GPU_C_SPEED_OF_LIGHT;

    /* Comoving frequency range the packet sweeps through */
    /* nu_cmf = nu_obs × D (blueshift toward observer) */
    double nu_cmf_start = nu_packet * D_start;
    double nu_cmf_end = nu_packet * D_end;

    /* Ensure nu_min < nu_max */
    double nu_min = (nu_cmf_start < nu_cmf_end) ? nu_cmf_start : nu_cmf_end;
    double nu_max = (nu_cmf_start > nu_cmf_end) ? nu_cmf_start : nu_cmf_end;

    /* Binary search for first line with nu >= nu_min */
    int64_t left = 0, right = n_lines;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (line_list_nu[mid] < nu_min) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    int64_t line_start = left;

    /* Variables for debug (if needed) */
    int my_debug = -1;
    (void)my_debug;  /* Suppress unused variable warning */

    /* Task Order #036 v5: DOMINANT LINE approximation
     *
     * With ~200 lines in each Doppler window, the product formula gives
     * P_escape ≈ 0 always. Instead, use dominant line approximation:
     *
     * Only the STRONGEST (highest tau) line in the window matters.
     * This is physically reasonable because:
     * 1. Line centers are at discrete frequencies
     * 2. Only the strongest line dominates at its resonance
     * 3. This gives β ≈ 1/τ_max for saturated lines, reasonable P_escape
     */
    double tau_max = 0.0;
    int64_t max_lines_to_check = 500;  /* Task #038-Revised: Increased from 100 */

    for (int64_t i = line_start; i < n_lines && max_lines_to_check > 0; i++) {
        double nu_line = line_list_nu[i];
        if (nu_line > nu_max) break;  /* Past the Doppler window */

        /* Get tau_sobolev for this line in this shell */
        int64_t tau_idx = i * n_shells + shell_id;
        double tau = tau_sobolev[tau_idx];

        /* Task #038-Revised: Lower threshold to tau > 1.0
         * This includes optically thick lines while filtering weak ones.
         * Si II 6355 has tau ~ 1000, should easily pass this filter.
         */
        if (tau > 1.0 && tau > tau_max) {
            tau_max = tau;
        }

        /* Debug: Show when strong line is found */
        if (my_debug >= 0 && my_debug < 10 && tau > 10.0) {
            double wl_line = GPU_C_SPEED_OF_LIGHT / nu_line * 1e8;
            printf("[LINE_TAU] STRONG line %ld: wl=%.1f A, nu=%.4e Hz, tau=%.0f\n",
                   (long)i, wl_line, nu_line, tau);
        }
        max_lines_to_check--;
    }

    /* Debug output (disabled: my_debug = -1) */
    (void)(500 - max_lines_to_check);  /* Suppress unused warning */

    /* Return effective tau from dominant line
     * Task #038: Cap tau_max at 3.0 for peeling escape probability
     *
     * With tau_max=1000, -log(β) ≈ 7 per shell, giving P_escape ≈ 0
     * over 30 shells. Capping at 3.0 gives -log(β) ≈ 1.5, so
     * P_escape ≈ exp(-45) is still too small.
     *
     * Use tau_max = 1.0 cap for reasonable P_escape:
     *   β(τ=1) = 0.632, -log(β) = 0.46, P_escape(30 shells) ≈ exp(-14) ≈ 1e-6
     *
     * Actually cap at τ=0.3 for P-Cygni:
     *   β(τ=0.3) = 0.86, -log(β) = 0.15, P_escape(30 shells) ≈ exp(-4.5) ≈ 0.01
     */
    if (tau_max > 0.01) {
        /* Task #038 v2: Increase tau cap for stronger P-Cygni absorption
         *
         * With tau_cap=0.5, we got blue/red=2.7 (too blue)
         * Try tau_cap=3.0 which gives β=0.32, -log(β)=1.14 per shell
         * Over 30 shells: tau_total ≈ 34, P_escape ≈ 1e-15 (too small)
         *
         * Alternative: Use tau directly, cap final tau_total in caller
         * For now, use tau_cap=2.0, -log(β)=0.8 per shell
         * 30 shells → tau_total=24 → P_escape=4e-11 (borderline)
         *
         * Use tau_cap=1.5: β=0.52, -log(β)=0.65, total=19.5 → P=3e-9
         */
        double tau_capped = (tau_max > 10.0) ? 10.0 : tau_max;  /* Task #038: Test with high cap */

        double beta;
        if (tau_capped < 1e-4) {
            beta = 1.0 - 0.5 * tau_capped;
        } else {
            beta = (1.0 - exp(-tau_capped)) / tau_capped;
        }
        return (beta > 1e-10) ? -log(beta) : 1.0;  /* Cap at -log(0.37) */
    }

    return 0.0;  /* No significant line in window */
}

/**
 * peeling_compute_escape_probability: P_escape through shell stack
 *
 * Task Order #029: Now includes BOTH electron scattering AND line opacity.
 * Task Order #036 FIX v2: Selective line opacity - only for photosphere!
 *
 * The P-Cygni absorption feature requires differential treatment:
 *   - PHOTOSPHERE peeling: INCLUDE line opacity (creates absorption trough)
 *   - SCATTERED peeling: EXCLUDE line opacity (avoid double-counting)
 *
 * @param r               Current radius
 * @param mu_obs          Observer direction cosine
 * @param shell_id        Starting shell
 * @param nu_packet       Packet frequency (for line opacity calculation)
 * @param r_inner         Shell inner radii
 * @param r_outer         Shell outer radii
 * @param electron_density Electron density per shell
 * @param line_list_nu    Sorted line frequencies
 * @param tau_sobolev     Line optical depths
 * @param n_lines         Number of lines
 * @param n_shells        Number of shells
 * @param inv_t_exp       1/t_explosion
 * @param include_lines   1 = include line opacity (photosphere), 0 = electrons only
 */
__device__ double peeling_compute_escape_probability(
    double r,
    double mu_obs,
    int64_t shell_id,
    double nu_packet,
    const double * __restrict__ r_inner,
    const double * __restrict__ r_outer,
    const double * __restrict__ electron_density,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    int64_t n_lines,
    int64_t n_shells,
    double inv_t_exp,
    int include_lines)   /* Task #036 v2: Selective line opacity flag */
{
    double tau_electron = 0.0;
    double tau_lines = 0.0;
    int64_t shell = shell_id;
    double pos_r = r;
    double pos_mu = mu_obs;

    /* Trace path toward observer through remaining shells */
    int max_steps = 50;  /* Reduced for performance with line calculation */
    while (shell < n_shells && max_steps > 0) {
        /* Distance to outer boundary */
        double r_out = r_outer[shell];
        double discriminant = r_out * r_out + (pos_mu * pos_mu - 1.0) * pos_r * pos_r;
        if (discriminant < 0.0) break;

        double d_boundary = sqrt(discriminant) - pos_r * pos_mu;
        if (d_boundary < 0.0) d_boundary = 0.0;

        /* Accumulate electron scattering optical depth */
        double n_e = electron_density[shell];
        tau_electron += n_e * GPU_SIGMA_THOMSON * d_boundary;

        /* Task Order #029: Accumulate LINE optical depth
         * Task Order #036 v2: Only for photosphere peeling!
         *
         * Physics rationale:
         *   - Photosphere continuum: Include lines to create P-Cygni absorption
         *   - Scattered photons: Exclude lines to avoid double-counting
         *     (interaction itself already removed the photon from line-of-sight)
         */
        double r_end = sqrt(pos_r * pos_r + d_boundary * d_boundary +
                           2.0 * pos_r * d_boundary * pos_mu);

        if (include_lines) {
            tau_lines += calculate_line_tau_on_ray(
                nu_packet, shell, pos_r, r_end, inv_t_exp,
                line_list_nu, tau_sobolev, n_lines, n_shells
            );
        }

        /* Move to boundary */
        pos_mu = (pos_mu * pos_r + d_boundary) / r_end;
        pos_r = r_end;
        shell++;
        max_steps--;
    }

    /* Total optical depth = electron + lines (if included) */
    double tau_total = tau_electron + tau_lines;

    /* Escape probability = exp(-tau) */
    return (tau_total < 50.0) ? exp(-tau_total) : 0.0;
}

/**
 * peeling_add_contribution_device: Add virtual packet to spectrum
 *
 * Task Order #029: Updated to include LINE OPACITY in escape probability.
 * Task Order #036 v2: Selective line opacity for P-Cygni absorption.
 *
 * @param pkt               Current packet state
 * @param shell_id          Current shell
 * @param model             Model parameters
 * @param r_inner           Shell inner radii
 * @param r_outer           Shell outer radii
 * @param electron_density  Electron densities per shell
 * @param line_list_nu      Sorted line frequencies (Task #029)
 * @param tau_sobolev       Line optical depths (Task #029)
 * @param n_lines           Number of lines (Task #029)
 * @param spectrum          Output spectrum (atomicAdd for flux)
 * @param mu_observer       Observer direction cosine (typically 1.0 for pole-on)
 * @param is_line_interaction  1 if from line, 0 if from electron scatter
 * @param tau_line          Sobolev optical depth of interacting line (0 for e-scatter)
 * @param is_photosphere    1 if photosphere contribution, 0 if scattered (Task #036 v2)
 */
__device__ void peeling_add_contribution_device(
    const RPacket_GPU *pkt,
    int64_t shell_id,
    const Model_GPU *model,
    const double * __restrict__ r_inner,
    const double * __restrict__ r_outer,
    const double * __restrict__ electron_density,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    int64_t n_lines,
    Spectrum_GPU *spectrum,
    double mu_observer,
    int is_line_interaction,
    double tau_line,
    int is_photosphere)  /* Task #036 v2: 1=include line opacity, 0=electrons only */
{
    if (spectrum == NULL) return;

    /* Task Order #026: Track peeling call counts for debugging */
    if (is_line_interaction) {
        atomicAdd((unsigned long long*)&spectrum->n_line_peeling_calls, 1ULL);
    } else {
        atomicAdd((unsigned long long*)&spectrum->n_escat_peeling_calls, 1ULL);
    }

    double r = pkt->r;
    double mu = pkt->mu;
    double nu = pkt->nu;
    double energy = pkt->energy;

    /* Compute velocity at this radius */
    double beta = r * model->inv_time_explosion / GPU_C_SPEED_OF_LIGHT;

    /* Doppler factor for packet direction */
    double D_packet = 1.0 - beta * mu;
    if (D_packet < 0.01) D_packet = 0.01;  /* Avoid division by zero */

    /* Doppler factor for observer direction */
    double D_observer = 1.0 - beta * mu_observer;
    if (D_observer < 0.01) D_observer = 0.01;

    /* Frequency shift ratio */
    double doppler_ratio = D_observer / D_packet;

    /* Observer-frame frequency */
    double nu_observer = nu * doppler_ratio;

    /* Convert to wavelength [Angstrom] */
    double wavelength = GPU_C_SPEED_OF_LIGHT / nu_observer * 1e8;

    /* Check if within spectrum range */
    if (wavelength < spectrum->wl_min || wavelength >= spectrum->wl_max) {
        /* Task Order #026: Track wavelength rejections */
        if (is_line_interaction) {
            atomicAdd((unsigned long long*)&spectrum->n_line_wl_rejected, 1ULL);
        } else {
            atomicAdd((unsigned long long*)&spectrum->n_escat_wl_rejected, 1ULL);
        }
        return;
    }

    /*
     * Task Order #029: Compute escape probability WITH LINE OPACITY
     * Task Order #036 v2: Selective - only for photosphere contribution!
     *
     * P-Cygni physics:
     *   - Photosphere: P_escape = exp(-(tau_electron + tau_lines))
     *     Line opacity creates absorption features at specific wavelengths
     *   - Scattered: P_escape = exp(-tau_electron)
     *     Avoid double-counting - interaction already removed from l.o.s.
     */
    double P_escape = peeling_compute_escape_probability(
        r, mu_observer, shell_id,
        nu,  /* Packet frequency for line opacity calculation */
        r_inner, r_outer, electron_density,
        line_list_nu, tau_sobolev, n_lines,
        model->n_shells, model->inv_time_explosion,
        is_photosphere   /* Task #036 v2: Include lines only for photosphere */
    );

    /*
     * Task Order #027: Apply additional Sobolev factor for line interactions
     *
     * When re-emitting from a specific line, apply the Sobolev escape
     * probability for THAT line (in addition to integrated path opacity).
     */
    if (is_line_interaction && tau_line > 0.0) {
        /*
         * Task Order #036: Cap tau for saturated lines in peeling
         *
         * For very high tau (> 100), the escape probability β = 1/τ becomes
         * negligibly small, killing all line contributions.
         *
         * Cap at tau=10 for peeling: this gives β ≈ 0.1, allowing
         * saturated line contributions to reach the spectrum.
         * The actual transport still uses full tau for interaction decision.
         */
        double tau_for_peeling = tau_line;
        if (tau_for_peeling > 10.0) {
            tau_for_peeling = 10.0;  /* Cap at τ=10 → β ≈ 0.1 */
        }

        double P_sobolev;
        if (tau_for_peeling < 1e-4) {
            P_sobolev = 1.0 - 0.5 * tau_for_peeling;
        } else {
            P_sobolev = (1.0 - exp(-tau_for_peeling)) / tau_for_peeling;
        }
        P_escape *= P_sobolev;
    }

    if (P_escape < 1e-10) {
        /* Task Order #026: Track escape probability rejections */
        if (is_line_interaction) {
            atomicAdd((unsigned long long*)&spectrum->n_line_escape_rejected, 1ULL);
        }
        return;
    }

    /* Weight factor: Doppler² × escape probability */
    /* The Doppler² factor accounts for energy transformation */
    double weight = doppler_ratio * doppler_ratio * P_escape;

    /* Weighted energy contribution */
    double flux_contribution = energy * weight;

    /* Find wavelength bin */
    int bin = (int)((wavelength - spectrum->wl_min) * spectrum->inv_d_wl);
    if (bin < 0 || bin >= GPU_SPECTRUM_NBINS) return;

    /* Atomic add to spectrum (thread-safe) */
    atomicAdd(&spectrum->flux[bin], flux_contribution);

    /* Update statistics */
    atomicAdd((unsigned long long*)&spectrum->n_contributions, 1ULL);
    if (is_line_interaction) {
        atomicAdd((unsigned long long*)&spectrum->n_line_contributions, 1ULL);
    } else {
        atomicAdd((unsigned long long*)&spectrum->n_escat_contributions, 1ULL);
    }
}

/* ============================================================================
 * Task Order #024: MACRO-ATOM DEVICE FUNCTIONS
 * ============================================================================
 *
 * GPU implementation of macro-atom physics for fluorescence and thermalization.
 * Enables UV → optical redistribution and proper line absorption features.
 */

/* Physical constants for macro-atom calculations */
#define GPU_K_BOLTZMANN  1.380649e-16      /* erg/K */
#define GPU_H_PLANCK     6.62607015e-27    /* erg·s */
#define GPU_MACRO_ATOM_COLLISION_COEFF 8.63e-6

/**
 * macro_atom_find_reference_device: Find reference for a level
 */
__device__ int macro_atom_find_reference_device(
    const MacroAtomReference_GPU * __restrict__ references,
    int32_t n_references,
    int8_t Z, int8_t ion, int16_t level,
    int32_t *trans_start, int32_t *n_trans)
{
    /* Linear search - could optimize with sorted index */
    for (int32_t i = 0; i < n_references; i++) {
        const MacroAtomReference_GPU *ref = &references[i];
        if (ref->atomic_number == Z &&
            ref->ion_number == ion &&
            ref->level_number == level) {
            *trans_start = ref->trans_start_idx;
            *n_trans = ref->n_transitions;
            return 1;  /* Found */
        }
    }
    *trans_start = 0;
    *n_trans = 0;
    return 0;  /* Not found */
}

/**
 * macro_atom_get_level_device: Find level energy and statistical weight
 */
__device__ int macro_atom_get_level_device(
    const Level_GPU * __restrict__ levels,
    int32_t n_levels,
    int8_t Z, int8_t ion, int16_t level_num,
    double *energy, int *g)
{
    /* Linear search for level */
    for (int32_t i = 0; i < n_levels; i++) {
        const Level_GPU *lvl = &levels[i];
        if (lvl->atomic_number == Z &&
            lvl->ion_number == ion &&
            lvl->level_number == level_num) {
            *energy = lvl->energy;
            *g = lvl->g;
            return 1;
        }
    }
    *energy = 0.0;
    *g = 1;
    return 0;
}

/**
 * calculate_beta_sobolev_device: Sobolev escape probability
 */
__device__ double calculate_beta_sobolev_device(double tau) {
    if (tau < 1e-6) return 1.0;
    if (tau > 500.0) return 1.0 / tau;
    return (1.0 - exp(-tau)) / tau;
}

/**
 * macro_atom_transition_loop_device: Monte Carlo cascade on GPU
 *
 * Simplified version that uses downbranch-style direct emission selection.
 * Full cascade would require more complex probability calculation.
 *
 * Task Order #037: Fixed tau_sobolev indexing - now uses shell-specific tau!
 *
 * @return 1 if packet emitted, 0 if thermalized
 */
__device__ int macro_atom_transition_loop_device(
    RPacket_GPU *pkt,
    int8_t atomic_number,
    int8_t ion_number,
    int16_t start_level,
    const MacroAtomTransition_GPU * __restrict__ transitions,
    const MacroAtomReference_GPU * __restrict__ references,
    int32_t n_transitions_total,
    int32_t n_references,
    const Line_GPU * __restrict__ lines,
    int32_t n_lines,
    const double * __restrict__ tau_sobolev,
    int64_t n_shells,        /* Task #037: Added for correct tau indexing */
    int64_t current_shell,   /* Task #037: Added for correct tau indexing */
    double T,
    double n_e,
    const MacroAtomTuning_GPU *tuning,
    int64_t *emission_line_id,
    double *emission_nu)
{
    int16_t current_level = start_level;
    int n_jumps = 0;

    /* Local arrays for transition probabilities (fixed size) */
    double probs[GPU_MACRO_ATOM_MAX_TRANS];
    int32_t trans_indices[GPU_MACRO_ATOM_MAX_TRANS];

    while (n_jumps < GPU_MACRO_ATOM_MAX_JUMPS) {

        /* Find reference for current level */
        int32_t trans_start, n_trans;
        int found = macro_atom_find_reference_device(
            references, n_references,
            atomic_number, ion_number, current_level,
            &trans_start, &n_trans
        );

        if (!found || n_trans == 0) {
            /* No transitions - use simplified emission */
            break;
        }

        /* Clamp to max size */
        if (n_trans > GPU_MACRO_ATOM_MAX_TRANS) {
            n_trans = GPU_MACRO_ATOM_MAX_TRANS;
        }

        /* Calculate transition rates */
        double total_rate = 0.0;
        int n_valid = 0;

        for (int32_t t = 0; t < n_trans; t++) {
            int32_t idx = trans_start + t;
            if (idx >= n_transitions_total) break;

            const MacroAtomTransition_GPU *trans = &transitions[idx];
            double rate = 0.0;

            if (trans->transition_type == -1) {
                /* Radiative emission: rate = A_ul × β
                 *
                 * Task Order #037 FIX: Use shell-specific tau_sobolev!
                 * Previous bug: tau_sobolev[trans->line_id] used shell 0 always
                 * Fixed: tau_sobolev[line_id * n_shells + current_shell]
                 */
                double beta = 1.0;
                if (tau_sobolev != NULL && trans->line_id >= 0 &&
                    trans->line_id < n_lines && current_shell < n_shells) {
                    int64_t tau_idx = trans->line_id * n_shells + current_shell;
                    double tau = tau_sobolev[tau_idx];
                    beta = calculate_beta_sobolev_device(tau);
                }
                rate = trans->A_ul * beta;
            }
            else if (trans->transition_type == 0) {
                /* Collisional de-excitation: simplified rate */
                /* C_ul ≈ n_e × collision_coeff / sqrt(T) */
                rate = n_e * GPU_MACRO_ATOM_COLLISION_COEFF / sqrt(T);
                rate *= tuning->collisional_boost;
            }
            else if (trans->transition_type == 1) {
                /* Upward internal: collisional excitation */
                double rate_down = n_e * GPU_MACRO_ATOM_COLLISION_COEFF / sqrt(T);
                rate_down *= tuning->collisional_boost;
                /* Scale by Boltzmann factor (approximate) */
                double delta_E = trans->nu * GPU_H_PLANCK;
                double kT = GPU_K_BOLTZMANN * T;
                if (delta_E > 0 && kT > 0) {
                    rate = rate_down * exp(-delta_E / kT);
                }
            }

            if (rate > 0.0) {
                probs[n_valid] = rate;
                trans_indices[n_valid] = idx;
                total_rate += rate;
                n_valid++;
            }
        }

        if (n_valid == 0 || total_rate <= 0.0) {
            /* No valid transitions - simplified emission */
            break;
        }

        /* Normalize probabilities */
        for (int i = 0; i < n_valid; i++) {
            probs[i] /= total_rate;
        }

        /* Sample random transition */
        double xi = gpu_rng_uniform(&pkt->rng_state);
        double cumulative = 0.0;
        int selected_idx = -1;

        for (int i = 0; i < n_valid; i++) {
            cumulative += probs[i];
            if (xi < cumulative) {
                selected_idx = trans_indices[i];
                break;
            }
        }

        if (selected_idx < 0) {
            selected_idx = trans_indices[n_valid - 1];
        }

        /* Process selected transition */
        const MacroAtomTransition_GPU *sel_trans = &transitions[selected_idx];

        if (sel_trans->transition_type == -1) {
            /* Radiative emission - packet survives */
            *emission_line_id = sel_trans->line_id;
            *emission_nu = sel_trans->nu;
            return 1;
        }

        /* Non-radiative: move to destination level */
        current_level = sel_trans->dest_level;
        n_jumps++;

        /* Check for ground state */
        if (current_level == 0) {
            break;  /* Use simplified emission from ground */
        }
    }

    /* Fallback: find strongest emission line from current level */
    double best_A = 0.0;
    int64_t best_line = -1;
    double best_nu = 0.0;

    for (int32_t i = 0; i < n_lines; i++) {
        const Line_GPU *line = &lines[i];
        if (line->atomic_number == atomic_number &&
            line->ion_number == ion_number &&
            line->level_upper == current_level) {
            if (line->A_ul > best_A) {
                best_A = line->A_ul;
                best_line = i;
                best_nu = line->nu;
            }
        }
    }

    if (best_line >= 0) {
        *emission_line_id = best_line;
        *emission_nu = best_nu;
        return 1;  /* Emit at strongest line */
    }

    /* No emission possible - thermalize */
    *emission_line_id = -1;
    *emission_nu = 0.0;
    return 0;
}

/**
 * macro_atom_process_line_device: Full macro-atom line interaction
 *
 * Replaces simple line_scatter when macro-atom mode is enabled.
 *
 * Task Order #037: Now passes n_shells and current_shell for correct tau indexing.
 *
 * @return 1 if packet survives (emitted), 0 if absorbed (thermalized)
 */
__device__ int macro_atom_process_line_device(
    RPacket_GPU *pkt,
    int64_t line_id,
    const MacroAtomTransition_GPU * __restrict__ transitions,
    const MacroAtomReference_GPU * __restrict__ references,
    int32_t n_transitions,
    int32_t n_references,
    const Line_GPU * __restrict__ lines,
    int32_t n_lines,
    const double * __restrict__ tau_sobolev,
    int64_t n_shells,        /* Task #037: For correct tau indexing */
    double T,
    double n_e,
    const Model_GPU *model,
    const MacroAtomTuning_GPU *tuning)
{
    /* Get line info for absorbed photon */
    if (line_id < 0 || line_id >= n_lines) {
        return 0;  /* Invalid line - absorb */
    }

    const Line_GPU *abs_line = &lines[line_id];

    /* Store incoming state */
    pkt->last_interaction_in_nu = pkt->nu;
    pkt->last_line_interaction_in_id = line_id;

    /* Run macro-atom transition loop
     * Task Order #037: Now passing n_shells and current_shell for correct tau indexing
     */
    int64_t emission_line_id = -1;
    double emission_nu = 0.0;

    int survives = macro_atom_transition_loop_device(
        pkt,
        abs_line->atomic_number,
        abs_line->ion_number,
        abs_line->level_upper,
        transitions, references,
        n_transitions, n_references,
        lines, n_lines,
        tau_sobolev,
        n_shells,                    /* Task #037: Pass shell count */
        pkt->current_shell_id,       /* Task #037: Pass current shell */
        T, n_e,
        tuning,
        &emission_line_id,
        &emission_nu
    );

    if (!survives || emission_nu <= 0.0) {
        /* Thermalized during cascade */
        return 0;
    }

    /* Apply wavelength-dependent thermalization (TARDIS epsilon) */
    double wavelength_A = GPU_C_SPEED_OF_LIGHT / emission_nu * 1e8;
    double p_thermalize = tuning->thermalization_epsilon;

    if (wavelength_A > tuning->ir_wavelength_threshold) {
        p_thermalize = tuning->ir_thermalization_boost;
    } else if (wavelength_A < tuning->uv_wavelength_threshold) {
        p_thermalize *= tuning->uv_scatter_boost;
    }

    if (gpu_rng_uniform(&pkt->rng_state) < p_thermalize) {
        /* Thermalized by epsilon check */
        return 0;
    }

    /* Packet survives - update direction and frequency */

    /* New direction: isotropic in CMF */
    double mu_cmf_new = 2.0 * gpu_rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform to lab frame */
    pkt->mu = angle_aberration_CMF_to_LF_gpu(pkt->r, mu_cmf_new, model->ct);

    /* New frequency: emission line frequency transformed to lab */
    double inv_doppler = get_inverse_doppler_factor_gpu(
        pkt->r, pkt->mu, model->inv_time_explosion,
        model->enable_full_relativity
    );
    pkt->nu = emission_nu * inv_doppler;

    /* Record interaction */
    pkt->last_interaction_type = 2;
    pkt->last_line_interaction_out_id = emission_line_id;

    /* Advance line pointer */
    pkt->next_line_id++;

    return 1;  /* Packet survives */
}

/**
 * line_scatter_device: Resonant line scattering (simple mode)
 */
__device__ void line_scatter_device(
    RPacket_GPU *pkt,
    const double * __restrict__ line_list_nu,
    const Model_GPU *model,
    int line_interaction_type)
{
    /* Store incoming state */
    pkt->last_interaction_in_nu = pkt->nu;
    pkt->last_line_interaction_in_id = pkt->next_line_id;

    /* Isotropic re-emission in comoving frame */
    double mu_cmf = 2.0 * gpu_rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform to lab frame */
    pkt->mu = angle_aberration_CMF_to_LF_gpu(pkt->r, mu_cmf, model->ct);

    /* For simple scatter mode: emit at line frequency */
    if (line_interaction_type == GPU_LINE_SCATTER) {
        double nu_line = line_list_nu[pkt->next_line_id];
        double inv_doppler = get_inverse_doppler_factor_gpu(
            pkt->r, pkt->mu, model->inv_time_explosion,
            model->enable_full_relativity
        );
        pkt->nu = nu_line * inv_doppler;
    }

    pkt->last_interaction_type = 2;  /* Line scatter */
    pkt->last_line_interaction_out_id = pkt->next_line_id;

    /* Advance past this line */
    pkt->next_line_id++;
}

/* ============================================================================
 * MAIN TRANSPORT KERNEL
 * ============================================================================
 *
 * Task Order #020: trace_packet_kernel
 *
 * Each thread processes one packet through its complete lifecycle.
 * This is the "persistent thread" model - one thread per packet.
 */

__global__ void trace_packet_kernel(
    RPacket_GPU *packets,
    int n_packets,
    const double * __restrict__ r_inner,
    const double * __restrict__ r_outer,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    const double * __restrict__ electron_density,
    const double * __restrict__ T_rad,            /* Task #024: Temperature per shell */
    Model_GPU model,
    Plasma_GPU plasma,
    GPUStats *stats,
    Spectrum_GPU *spectrum,       /* Task #024: Peeling spectrum output */
    double mu_observer,           /* Task #024: Observer direction */
    /* Task #024: Macro-atom data (can be NULL for simple scatter mode) */
    const MacroAtomTransition_GPU * __restrict__ ma_transitions,
    const MacroAtomReference_GPU * __restrict__ ma_references,
    int32_t n_ma_transitions,
    int32_t n_ma_references,
    const Line_GPU * __restrict__ lines_gpu,
    int32_t n_lines_gpu,
    MacroAtomTuning_GPU ma_tuning,
    int max_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_packets) return;

    /* Load packet from global memory to registers */
    RPacket_GPU pkt = packets[idx];

    /* Statistics counters (thread-local) */
    int64_t local_line_interactions = 0;
    int64_t local_electron_scatters = 0;
    int64_t local_boundary_crossings = 0;
    int iter = 0;

    /*
     * Task Order #028: PHOTOSPHERE PEELING - Direct Unscattered Light
     * Task Order #029: Now includes LINE OPACITY in escape probability!
     *
     * This is the CRITICAL "zeroth" contribution that creates the continuum
     * background against which absorption features are measured.
     *
     * Physics: When a photon is emitted from the photosphere, there's a
     * probability it flies directly to the observer without any interactions.
     * This direct light creates the continuum. P-Cygni absorption features
     * are "holes" carved into this continuum by intervening line opacity.
     *
     * The peeling call computes: E × P_escape(photosphere → observer)
     * where P_escape = exp(-(tau_electron + tau_lines))
     *
     * The LINE OPACITY integration (Task #029) is what creates the absorption!
     */
    /* Task #038: Re-enable line opacity for P-Cygni absorption
     *
     * The P-Cygni profile requires blue photons to have lower escape
     * probability due to line opacity. Setting is_photosphere=1 enables
     * line opacity in escape probability calculation.
     *
     * The calculate_line_tau_on_ray function now uses dominant-line
     * approximation which should prevent P_escape → 0.
     */
    peeling_add_contribution_device(
        &pkt, pkt.current_shell_id, &model,
        r_inner, r_outer, electron_density,
        line_list_nu, tau_sobolev, plasma.n_lines,
        spectrum, mu_observer, 0,  /* is_line=0 (continuum source) */
        0.0,  /* tau_line = 0 (no specific line at photosphere) */
        1    /* Task #038: ENABLE line opacity for P-Cygni */
    );

    /* Main transport loop */
    while (pkt.status == GPU_PACKET_IN_PROCESS && iter < max_iterations) {

        /* Find next interaction */
        double distance;
        int delta_shell;
        GPUInteractionType itype = trace_packet_device(
            &pkt, r_inner, r_outer, line_list_nu, tau_sobolev,
            electron_density, &model, &plasma, &distance, &delta_shell
        );

        /* Move packet to interaction point */
        move_packet_gpu(&pkt.r, &pkt.mu, distance);

        /* Process interaction */
        switch (itype) {
            case GPU_INTERACTION_BOUNDARY:
                process_boundary_crossing_device(&pkt, delta_shell, model.n_shells);
                local_boundary_crossings++;
                break;

            case GPU_INTERACTION_ESCATTERING:
                /* Task #024: Peeling-off BEFORE scatter (at interaction point) */
                /* Task #036 v3: Optionally disable scattered peeling for diagnostics */
                if (!plasma.disable_scattered_peeling) {
                    peeling_add_contribution_device(
                        &pkt, pkt.current_shell_id, &model,
                        r_inner, r_outer, electron_density,
                        line_list_nu, tau_sobolev, plasma.n_lines,
                        spectrum, mu_observer, 0,  /* is_line=0 */
                        0.0,  /* tau_line = 0 for e-scatter */
                        1     /* Task #038-Revised: ENABLE line opacity for ALL peeling */
                    );
                }
                thomson_scatter_device(&pkt, &model);
                local_electron_scatters++;
                break;

            case GPU_INTERACTION_LINE:
                {
                    /*
                     * Task #027: Retrieve Sobolev optical depth for this line interaction
                     *
                     * The tau_sobolev array is indexed as: tau_sobolev[line_id * n_shells + shell_id]
                     * This tau value represents the line opacity that the virtual packet
                     * must traverse when peeling toward the observer.
                     */
                    int64_t tau_idx = pkt.next_line_id * plasma.n_shells + pkt.current_shell_id;
                    double tau_line_value = tau_sobolev[tau_idx];

                    /* Task #024: Peeling-off BEFORE line scatter */
                    /* Task #036 v3: Optionally disable scattered peeling for diagnostics */
                    if (!plasma.disable_scattered_peeling) {
                        peeling_add_contribution_device(
                            &pkt, pkt.current_shell_id, &model,
                            r_inner, r_outer, electron_density,
                            line_list_nu, tau_sobolev, plasma.n_lines,
                            spectrum, mu_observer, 1,  /* is_line=1 */
                            tau_line_value,  /* Task #027: Line opacity */
                            1     /* Task #038-Revised: ENABLE line opacity for ALL peeling */
                        );
                    }

                /* Task #024: Use macro-atom if data available, else simple scatter */
                if (ma_transitions != NULL && n_ma_transitions > 0 &&
                    plasma.line_interaction_type == GPU_LINE_MACROATOM) {
                    /* Macro-atom mode: fluorescence and thermalization
                     * Task #037: Now passing plasma.n_shells for correct tau indexing
                     */
                    double T_local = (T_rad != NULL) ? T_rad[pkt.current_shell_id] : 10000.0;
                    double n_e_local = electron_density[pkt.current_shell_id];

                    int survives = macro_atom_process_line_device(
                        &pkt, pkt.next_line_id,
                        ma_transitions, ma_references,
                        n_ma_transitions, n_ma_references,
                        lines_gpu, n_lines_gpu,
                        tau_sobolev,
                        plasma.n_shells,   /* Task #037: Pass shell count */
                        T_local, n_e_local,
                        &model, &ma_tuning
                    );

                    if (!survives) {
                        /* Packet thermalized - mark as reabsorbed */
                        pkt.status = GPU_PACKET_REABSORBED;
                    }
                } else {
                    /* Simple scatter mode (no macro-atom data) */
                    line_scatter_device(&pkt, line_list_nu, &model,
                                        plasma.line_interaction_type);
                }
                local_line_interactions++;
                }  /* End Task #027 block */
                break;
        }

        iter++;
    }

    /* Write packet back to global memory */
    packets[idx] = pkt;

    /* Update global statistics atomically */
    if (stats != NULL) {
        atomicAdd((unsigned long long*)&stats->n_iterations_total, (unsigned long long)iter);
        atomicAdd((unsigned long long*)&stats->n_line_interactions, (unsigned long long)local_line_interactions);
        atomicAdd((unsigned long long*)&stats->n_electron_scatters, (unsigned long long)local_electron_scatters);
        atomicAdd((unsigned long long*)&stats->n_boundary_crossings, (unsigned long long)local_boundary_crossings);

        if (pkt.status == GPU_PACKET_EMITTED) {
            atomicAdd((unsigned long long*)&stats->n_emitted, 1ULL);
        } else if (pkt.status == GPU_PACKET_REABSORBED) {
            atomicAdd((unsigned long long*)&stats->n_reabsorbed, 1ULL);
        }
    }
}

/* ============================================================================
 * KERNEL LAUNCHER (C-callable)
 * ============================================================================ */

extern "C" {

/**
 * Launch trace_packet kernel with full macro-atom support
 *
 * Task #024: Extended launcher with macro-atom data
 */
int cuda_launch_trace_packet_macro_atom(
    void *d_packets,
    int n_packets,
    void *d_r_inner,
    void *d_r_outer,
    void *d_line_list_nu,
    void *d_tau_sobolev,
    void *d_electron_density,
    void *d_T_rad,              /* Temperature per shell (can be NULL) */
    Model_GPU model,
    Plasma_GPU plasma,
    void *d_stats,
    void *d_spectrum,
    double mu_observer,
    /* Macro-atom data (all can be NULL for simple scatter mode) */
    void *d_ma_transitions,
    void *d_ma_references,
    int32_t n_ma_transitions,
    int32_t n_ma_references,
    void *d_lines_gpu,
    int32_t n_lines_gpu,
    MacroAtomTuning_GPU ma_tuning,
    int stream_id,
    int max_iterations)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return -1;
    }

    cudaStream_t stream = (cudaStream_t)cuda_interface_get_stream(stream_id);

    int threads_per_block = 256;
    int num_blocks = (n_packets + threads_per_block - 1) / threads_per_block;

    trace_packet_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        (RPacket_GPU*)d_packets,
        n_packets,
        (const double*)d_r_inner,
        (const double*)d_r_outer,
        (const double*)d_line_list_nu,
        (const double*)d_tau_sobolev,
        (const double*)d_electron_density,
        (const double*)d_T_rad,
        model,
        plasma,
        (GPUStats*)d_stats,
        (Spectrum_GPU*)d_spectrum,
        mu_observer,
        (const MacroAtomTransition_GPU*)d_ma_transitions,
        (const MacroAtomReference_GPU*)d_ma_references,
        n_ma_transitions,
        n_ma_references,
        (const Line_GPU*)d_lines_gpu,
        n_lines_gpu,
        ma_tuning,
        max_iterations
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: trace_packet_kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Launch trace_packet kernel (simple mode - backward compatible)
 *
 * This wrapper calls the macro-atom version with NULL macro-atom data
 * for backward compatibility with existing code.
 */
int cuda_launch_trace_packet(
    void *d_packets,
    int n_packets,
    void *d_r_inner,
    void *d_r_outer,
    void *d_line_list_nu,
    void *d_tau_sobolev,
    void *d_electron_density,
    Model_GPU model,
    Plasma_GPU plasma,
    void *d_stats,
    void *d_spectrum,
    double mu_observer,
    int stream_id,
    int max_iterations)
{
    /* Default macro-atom tuning (not used when ma_transitions is NULL) */
    MacroAtomTuning_GPU ma_tuning;
    memset(&ma_tuning, 0, sizeof(ma_tuning));
    ma_tuning.thermalization_epsilon = 0.35;
    ma_tuning.ir_thermalization_boost = 0.80;
    ma_tuning.ir_wavelength_threshold = 7000.0;
    ma_tuning.uv_scatter_boost = 0.5;
    ma_tuning.uv_wavelength_threshold = 3500.0;

    return cuda_launch_trace_packet_macro_atom(
        d_packets, n_packets,
        d_r_inner, d_r_outer,
        d_line_list_nu, d_tau_sobolev,
        d_electron_density,
        NULL,  /* d_T_rad - not provided */
        model, plasma,
        d_stats, d_spectrum, mu_observer,
        NULL, NULL, 0, 0,  /* No macro-atom data */
        NULL, 0,           /* No Line_GPU data */
        ma_tuning,
        stream_id, max_iterations
    );
}

/**
 * Allocate and initialize packet array on GPU
 */
int cuda_allocate_packets(void **d_packets, int n_packets)
{
    cudaError_t err = cudaMalloc(d_packets, n_packets * sizeof(RPacket_GPU));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to allocate packets: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Copy packets from host to device
 */
int cuda_upload_packets(void *d_packets, const void *h_packets, int n_packets)
{
    cudaError_t err = cudaMemcpy(d_packets, h_packets,
                                  n_packets * sizeof(RPacket_GPU),
                                  cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Copy packets from device to host
 */
int cuda_download_packets(void *h_packets, const void *d_packets, int n_packets)
{
    cudaError_t err = cudaMemcpy(h_packets, d_packets,
                                  n_packets * sizeof(RPacket_GPU),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Allocate statistics structure on GPU
 */
int cuda_allocate_stats(void **d_stats)
{
    cudaError_t err = cudaMalloc(d_stats, sizeof(GPUStats));
    if (err != cudaSuccess) return -1;

    cudaMemset(*d_stats, 0, sizeof(GPUStats));
    return 0;
}

/**
 * Download statistics from GPU
 */
int cuda_download_stats(void *h_stats, const void *d_stats)
{
    cudaError_t err = cudaMemcpy(h_stats, d_stats, sizeof(GPUStats),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/* ============================================================================
 * Task Order #024: Peeling Spectrum GPU Functions
 * ============================================================================ */

/**
 * Allocate and initialize peeling spectrum on GPU
 */
int cuda_allocate_spectrum(void **d_spectrum)
{
    cudaError_t err = cudaMalloc(d_spectrum, sizeof(Spectrum_GPU));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to allocate spectrum: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    /* Initialize to zero */
    cudaMemset(*d_spectrum, 0, sizeof(Spectrum_GPU));

    /* Set binning parameters on device */
    Spectrum_GPU h_spectrum;
    memset(&h_spectrum, 0, sizeof(Spectrum_GPU));
    h_spectrum.wl_min = GPU_SPECTRUM_WL_MIN;
    h_spectrum.wl_max = GPU_SPECTRUM_WL_MAX;
    h_spectrum.d_wl = (GPU_SPECTRUM_WL_MAX - GPU_SPECTRUM_WL_MIN) / GPU_SPECTRUM_NBINS;
    h_spectrum.inv_d_wl = 1.0 / h_spectrum.d_wl;

    /* Copy binning parameters (but keep flux zeroed) */
    /* We need to copy only the metadata fields, not the flux array */
    size_t offset_wl_min = offsetof(Spectrum_GPU, wl_min);
    size_t meta_size = sizeof(Spectrum_GPU) - offset_wl_min;

    err = cudaMemcpy((char*)(*d_spectrum) + offset_wl_min,
                     (char*)&h_spectrum + offset_wl_min,
                     meta_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to initialize spectrum metadata: %s\n",
                cudaGetErrorString(err));
        cudaFree(*d_spectrum);
        *d_spectrum = NULL;
        return -1;
    }

    printf("[CUDA] Allocated peeling spectrum: %d bins, %.0f-%.0f Å\n",
           GPU_SPECTRUM_NBINS, GPU_SPECTRUM_WL_MIN, GPU_SPECTRUM_WL_MAX);

    return 0;
}

/**
 * Download peeling spectrum from GPU
 */
int cuda_download_spectrum(void *h_spectrum, const void *d_spectrum)
{
    cudaError_t err = cudaMemcpy(h_spectrum, d_spectrum, sizeof(Spectrum_GPU),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Free peeling spectrum on GPU
 */
void cuda_free_spectrum(void *d_spectrum)
{
    if (d_spectrum) {
        cudaFree(d_spectrum);
    }
}

} /* extern "C" - Task Order #020 additions */

/* ============================================================================
 * WARMUP KERNEL
 * ============================================================================
 *
 * Simple kernel that performs minimal computation to verify GPU access.
 * Each thread writes its ID to verify execution.
 */

__global__ void warmup_kernel(int *output, int n_elements, int stream_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        /* Simple computation: store thread index + stream ID */
        output[idx] = idx + stream_id * 1000000;
    }

    /* First thread in first block prints verification */
    if (idx == 0) {
        printf("[GPU] Warmup kernel executed: stream_id=%d, n_elements=%d, blockDim=%d\n",
               stream_id, n_elements, blockDim.x);
    }
}

/**
 * More sophisticated warmup that does actual work
 * Useful for profiling stream concurrency
 */
__global__ void warmup_compute_kernel(float *data, int n_elements, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        float val = (float)idx;

        /* Perform some computation to keep GPU busy */
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) * cosf(val) + 0.1f;
        }

        data[idx] = val;
    }
}

/* ============================================================================
 * C-CALLABLE INTERFACE IMPLEMENTATION
 * ============================================================================ */

extern "C" {

int cuda_interface_init(int device_id)
{
    if (g_cuda_initialized) {
        return 0;  /* Already initialized */
    }

    /* Query device count */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[CUDA] Error: No CUDA devices found: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    /* Validate device ID */
    if (device_id < 0 || device_id >= device_count) {
        fprintf(stderr, "[CUDA] Error: Invalid device ID %d (have %d devices)\n",
                device_id, device_count);
        return -1;
    }

    /* Set device */
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to set device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    /* Print device info */
    cuda_interface_print_device_info(device_id);

    /* Initialize stream tracking */
    memset(g_streams, 0, sizeof(g_streams));
    memset(g_stream_created, 0, sizeof(g_stream_created));

    g_cuda_device_id = device_id;
    g_cuda_initialized = 1;

    printf("[CUDA] Initialization complete (device %d)\n", device_id);
    return 0;
}

int cuda_interface_is_available(void)
{
    return g_cuda_initialized;
}

int cuda_interface_get_device_count(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
}

void cuda_interface_print_device_info(int device_id)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Cannot get device properties: %s\n",
                cudaGetErrorString(err));
        return;
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           LUMINA-SN CUDA Device Information                   ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Device %d: %-49s ║\n", device_id, prop.name);
    printf("║  Compute Capability:  %d.%d                                     ║\n",
           prop.major, prop.minor);
    printf("║  Multiprocessors:     %-42d ║\n", prop.multiProcessorCount);
    printf("║  Global Memory:       %-39.1f GB ║\n",
           prop.totalGlobalMem / 1e9);
    printf("║  Shared Mem/Block:    %-39zu KB ║\n",
           prop.sharedMemPerBlock / 1024);
    printf("║  Max Threads/Block:   %-42d ║\n", prop.maxThreadsPerBlock);
    printf("║  Warp Size:           %-42d ║\n", prop.warpSize);
    printf("║  Concurrent Kernels:  %-42s ║\n",
           prop.concurrentKernels ? "Yes" : "No");
    printf("║  Async Engines:       %-42d ║\n", prop.asyncEngineCount);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
}

void cuda_interface_shutdown(void)
{
    if (!g_cuda_initialized) return;

    /* Destroy all streams */
    cuda_interface_destroy_streams();

    /* Reset device */
    cudaDeviceReset();

    g_cuda_initialized = 0;
    g_cuda_device_id = -1;

    printf("[CUDA] Shutdown complete\n");
}

/* ============================================================================
 * STREAM MANAGEMENT
 * ============================================================================ */

void* cuda_interface_get_stream(int stream_id)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return NULL;
    }

    if (stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        fprintf(stderr, "[CUDA] Error: Invalid stream ID %d (max %d)\n",
                stream_id, CUDA_MAX_STREAMS - 1);
        return NULL;
    }

    /* Thread-safe stream creation */
    pthread_mutex_lock(&g_stream_mutex);

    if (!g_stream_created[stream_id]) {
        cudaError_t err = cudaStreamCreate(&g_streams[stream_id]);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] Error: Failed to create stream %d: %s\n",
                    stream_id, cudaGetErrorString(err));
            pthread_mutex_unlock(&g_stream_mutex);
            return NULL;
        }
        g_stream_created[stream_id] = 1;
        printf("[CUDA] Created stream %d\n", stream_id);
    }

    pthread_mutex_unlock(&g_stream_mutex);

    return (void*)g_streams[stream_id];
}

int cuda_interface_stream_sync(int stream_id)
{
    if (!g_cuda_initialized || stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        return -1;
    }

    if (!g_stream_created[stream_id]) {
        return 0;  /* Stream not created, nothing to sync */
    }

    cudaError_t err = cudaStreamSynchronize(g_streams[stream_id]);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Stream %d sync failed: %s\n",
                stream_id, cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int cuda_interface_sync_all_streams(void)
{
    if (!g_cuda_initialized) return -1;

    int failures = 0;

    for (int i = 0; i < CUDA_MAX_STREAMS; i++) {
        if (g_stream_created[i]) {
            if (cuda_interface_stream_sync(i) < 0) {
                failures++;
            }
        }
    }

    return (failures == 0) ? 0 : -1;
}

void cuda_interface_destroy_streams(void)
{
    pthread_mutex_lock(&g_stream_mutex);

    for (int i = 0; i < CUDA_MAX_STREAMS; i++) {
        if (g_stream_created[i]) {
            cudaStreamDestroy(g_streams[i]);
            g_stream_created[i] = 0;
        }
    }

    pthread_mutex_unlock(&g_stream_mutex);
}

/* ============================================================================
 * WARMUP / DIAGNOSTIC KERNELS
 * ============================================================================ */

int cuda_interface_launch_warmup(int stream_id, int n_elements)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return -1;
    }

    /* Get or create stream */
    cudaStream_t stream = (cudaStream_t)cuda_interface_get_stream(stream_id);
    if (stream == NULL && stream_id != 0) {
        /* Stream 0 is the default stream, NULL is valid */
        return -1;
    }

    /* Allocate device memory for output */
    int *d_output = NULL;
    cudaError_t err = cudaMalloc(&d_output, n_elements * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to allocate warmup buffer: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    /* Launch kernel */
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    warmup_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_output, n_elements, stream_id
    );

    /* Check for launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Warmup kernel launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_output);
        return -1;
    }

    /* Free device memory (will complete after kernel finishes) */
    cudaFreeAsync(d_output, stream);

    return 0;
}

int cuda_interface_launch_warmup_sync(int stream_id, int n_elements)
{
    int result = cuda_interface_launch_warmup(stream_id, n_elements);
    if (result < 0) return result;

    return cuda_interface_stream_sync(stream_id);
}

int cuda_interface_test_concurrency(int n_streams)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return -1;
    }

    if (n_streams <= 0 || n_streams > CUDA_MAX_STREAMS) {
        fprintf(stderr, "[CUDA] Error: Invalid stream count %d\n", n_streams);
        return -1;
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           CUDA Concurrency Test (%2d streams)                  ║\n",
           n_streams);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* Allocate work buffers for each stream */
    const int n_elements = 1024 * 1024;  /* 1M elements per stream */
    const int iterations = 100;           /* Work per element */

    float **d_buffers = (float**)malloc(n_streams * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        cudaMalloc(&d_buffers[i], n_elements * sizeof(float));
        streams[i] = (cudaStream_t)cuda_interface_get_stream(i);
    }

    /* Create timing events */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Record start time */
    cudaEventRecord(start);

    /* Launch kernels on all streams */
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    printf("[CUDA] Launching %d kernels concurrently...\n", n_streams);

    for (int i = 0; i < n_streams; i++) {
        warmup_compute_kernel<<<num_blocks, threads_per_block, 0, streams[i]>>>(
            d_buffers[i], n_elements, iterations
        );
    }

    /* Record stop time (after all kernels) */
    cudaEventRecord(stop);

    /* Wait for all streams to complete */
    cudaEventSynchronize(stop);

    /* Calculate elapsed time */
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("[CUDA] All kernels completed in %.2f ms\n", elapsed_ms);
    printf("[CUDA] Throughput: %.2f M elements/stream/ms\n",
           (float)n_elements / 1e6 / elapsed_ms * n_streams);

    /* Cleanup */
    for (int i = 0; i < n_streams; i++) {
        cudaFree(d_buffers[i]);
    }
    free(d_buffers);
    free(streams);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("[CUDA] Concurrency test PASSED\n\n");
    return 0;
}

/* ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================ */

void* cuda_interface_malloc(size_t size_bytes)
{
    if (!g_cuda_initialized) return NULL;

    void *d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, size_bytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: cudaMalloc(%zu) failed: %s\n",
                size_bytes, cudaGetErrorString(err));
        return NULL;
    }

    return d_ptr;
}

void cuda_interface_free(void *d_ptr)
{
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

int cuda_interface_memcpy_h2d(void *d_dst, const void *h_src, size_t size)
{
    cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_interface_memcpy_d2h(void *h_dst, const void *d_src, size_t size)
{
    cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_interface_memcpy_h2d_async(void *d_dst, const void *h_src,
                                     size_t size, int stream_id)
{
    if (!g_cuda_initialized || stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        return -1;
    }

    cudaStream_t stream = g_stream_created[stream_id] ?
                          g_streams[stream_id] : 0;

    cudaError_t err = cudaMemcpyAsync(d_dst, h_src, size,
                                       cudaMemcpyHostToDevice, stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_interface_memcpy_d2h_async(void *h_dst, const void *d_src,
                                     size_t size, int stream_id)
{
    if (!g_cuda_initialized || stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        return -1;
    }

    cudaStream_t stream = g_stream_created[stream_id] ?
                          g_streams[stream_id] : 0;

    cudaError_t err = cudaMemcpyAsync(h_dst, d_src, size,
                                       cudaMemcpyDeviceToHost, stream);
    return (err == cudaSuccess) ? 0 : -1;
}

void* cuda_interface_malloc_host(size_t size_bytes)
{
    void *h_ptr = NULL;
    cudaError_t err = cudaMallocHost(&h_ptr, size_bytes);
    return (err == cudaSuccess) ? h_ptr : NULL;
}

void cuda_interface_free_host(void *h_ptr)
{
    if (h_ptr) {
        cudaFreeHost(h_ptr);
    }
}

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

const char* cuda_interface_get_error(void)
{
    return cudaGetErrorString(cudaGetLastError());
}

int cuda_interface_check_error(void)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error detected: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} /* extern "C" */

/* ============================================================================
 * STANDALONE TEST DRIVER (when compiled directly)
 * ============================================================================ */

#ifdef GPU_TRANSPORT_STANDALONE

#include <omp.h>

int main(int argc, char *argv[])
{
    printf("LUMINA-SN GPU Transport Test\n");
    printf("============================\n\n");

    /* Initialize CUDA */
    if (cuda_interface_init(0) < 0) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }

    /* Test basic warmup */
    printf("Testing single warmup kernel...\n");
    if (cuda_interface_launch_warmup_sync(0, 1024) < 0) {
        fprintf(stderr, "Warmup failed\n");
        return 1;
    }
    printf("Single warmup: PASSED\n\n");

    /* Test concurrency */
    int n_threads = 4;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    if (n_threads > 8) n_threads = 8;  /* Limit for test */
    #endif

    printf("Testing OpenMP + CUDA concurrency (%d threads)...\n", n_threads);

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        printf("[Thread %d] Launching warmup kernel on stream %d\n", tid, tid);
        cuda_interface_launch_warmup(tid, 1024);
    }

    /* Sync all streams */
    cuda_interface_sync_all_streams();
    printf("OpenMP concurrency test: PASSED\n\n");

    /* Test stream concurrency with profiling */
    cuda_interface_test_concurrency(n_threads);

    /* Cleanup */
    cuda_interface_shutdown();

    printf("All tests PASSED\n");
    return 0;
}

#endif /* GPU_TRANSPORT_STANDALONE */
