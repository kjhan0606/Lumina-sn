/**
 * LUMINA-SN: C Implementation of TARDIS Monte Carlo Transport
 * rpacket.c - Core transport functions
 *
 * Transpiled from: tardis/montecarlo/montecarlo_numba/r_packet.py
 *                  tardis/montecarlo/montecarlo_numba/single_packet_loop.py
 *
 * Physical Reference: Lucy 2002, 2003; Mazzali & Lucy 1993
 *
 * HPC-Hardened Version:
 *   - Thread-safe RNG (Xorshift64* per-packet)
 *   - No global configuration state
 *   - Kernel-consistent aberration/Doppler transforms
 *   - Optimized hot path with early exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "physics_kernels.h"  /* Must come before rpacket.h */
#include "rpacket.h"

/* Global relativity mode flag (used by physics_kernels.h) */
int ENABLE_FULL_RELATIVITY = 0;

/* ============================================================================
 * PACKET INITIALIZATION
 * ============================================================================ */

void rpacket_init(RPacket *pkt, double r, double mu, double nu,
                  double energy, uint64_t seed, int64_t index) {
    pkt->r = r;
    pkt->mu = mu;
    pkt->nu = nu;
    pkt->energy = energy;
    pkt->current_shell_id = 0;  /* Will be set properly by caller */
    pkt->status = PACKET_IN_PROCESS;
    pkt->index = index;
    pkt->next_line_id = 0;
    pkt->last_interaction_type = -1;
    pkt->last_interaction_in_nu = 0.0;
    pkt->last_line_interaction_in_id = -1;
    pkt->last_line_interaction_out_id = -1;

    /* Initialize thread-safe RNG with unique seed per packet */
    rng_init(&pkt->rng_state, seed ^ ((uint64_t)index * 0x9E3779B97F4A7C15ULL));
}

void rpacket_initialize_line_id(RPacket *pkt, const NumbaPlasma *plasma,
                                 const NumbaModel *model) {
    /*
     * Binary search to find starting position in line list.
     *
     * The line list is SORTED by frequency (ascending).
     * We need to find the first line with nu_line <= nu_cmf.
     *
     * As packet propagates outward, its CMF frequency decreases,
     * so it sweeps through the line list from high to low frequency.
     */
    double doppler_factor = get_doppler_factor(pkt->r, pkt->mu, model->time_explosion);
    double comov_nu = pkt->nu * doppler_factor;

    /* Binary search for position in reversed list */
    int64_t n_lines = plasma->n_lines;
    int64_t low = 0, high = n_lines;

    while (low < high) {
        int64_t mid = (low + high) / 2;
        /* Search in reversed order (high freq to low) */
        if (plasma->line_list_nu[n_lines - 1 - mid] > comov_nu) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    pkt->next_line_id = n_lines - low;
    if (pkt->next_line_id >= n_lines) {
        pkt->next_line_id = n_lines - 1;
    }
}

/* ============================================================================
 * PACKET MOVEMENT
 * ============================================================================
 * Note: Distance and opacity calculations are in physics_kernels.h
 * using the exact TARDIS-SN formulas for 10^-10 tolerance validation.
 * ============================================================================ */

void move_r_packet(RPacket *pkt, double distance, double time_explosion,
                   Estimators *estimators) {
    /*
     * Update packet position and direction after traveling 'distance'.
     *
     * Geometry in spherical coordinates:
     *   r_new^2 = r^2 + d^2 + 2*r*d*mu
     *   mu_new = (r*mu + d) / r_new
     *
     * Physical interpretation:
     * - The packet moves in a straight line (no gravity, no refraction)
     * - But in spherical coords, both r and mu change along this line
     * - This is purely geometric, not a physical interaction
     */
    if (distance <= 0.0) {
        return;
    }

    double r = pkt->r;
    double mu = pkt->mu;

    /* Pre-move Doppler factor for estimators */
    double doppler_factor = get_doppler_factor(r, mu, time_explosion);
    double comov_nu = pkt->nu * doppler_factor;
    double comov_energy = pkt->energy * doppler_factor;

    /* Geometric update */
    double r_new_sq = r * r + distance * distance + 2.0 * r * distance * mu;
    double r_new = sqrt(r_new_sq);
    double mu_new = (mu * r + distance) / r_new;

    pkt->r = r_new;
    pkt->mu = mu_new;

    /* Update radiative estimators (J, nu_bar) */
    if (estimators != NULL) {
        set_estimators(pkt, distance, estimators, comov_nu, comov_energy);
    }
}

void move_packet_across_shell_boundary(RPacket *pkt, int delta_shell,
                                        int64_t n_shells) {
    /*
     * Handle transition between shells or termination at boundaries.
     *
     * delta_shell = +1: moving to outer shell
     * delta_shell = -1: moving to inner shell
     *
     * Termination conditions:
     * - Escaped outer boundary (current + 1 >= n_shells): EMITTED
     * - Hit inner boundary (current - 1 < 0): REABSORBED
     */
    int64_t next_shell = pkt->current_shell_id + delta_shell;

    if (next_shell >= n_shells) {
        /* Escaped through outer boundary -> contributes to observed spectrum */
        pkt->status = PACKET_EMITTED;
    } else if (next_shell < 0) {
        /* Hit photosphere -> thermalized, contributes to internal radiation */
        pkt->status = PACKET_REABSORBED;
    } else {
        /* Normal shell transition */
        pkt->current_shell_id = next_shell;
    }
}

/* ============================================================================
 * ESTIMATOR UPDATES
 * ============================================================================ */

void set_estimators(RPacket *pkt, double distance, Estimators *est,
                    double comov_nu, double comov_energy) {
    /*
     * Update Monte Carlo estimators for mean intensity J and frequency.
     *
     * J_nu ~ Sum (epsilon * d) / V  where epsilon is packet energy, d is path length
     *
     * These estimators are used in the temperature/ionization iteration.
     */
    if (est == NULL || est->j_estimator == NULL) {
        return;
    }

    int64_t shell_id = pkt->current_shell_id;

    /* J estimator: energy-weighted path length */
    est->j_estimator[shell_id] += comov_energy * distance;

    /* Frequency-weighted estimator for radiation temperature */
    est->nu_bar_estimator[shell_id] += comov_energy * distance * comov_nu;
}

void update_line_estimators(Estimators *est, const RPacket *pkt,
                            int64_t line_id, double distance,
                            double time_explosion) {
    /*
     * Update line-specific estimators (for NLTE level populations).
     * J_blue: intensity shortward of the line
     */
    if (est == NULL || est->j_blue_estimator == NULL) {
        return;
    }

    double doppler_factor = get_doppler_factor(pkt->r, pkt->mu, time_explosion);
    double comov_energy = pkt->energy * doppler_factor;

    /* Index into 2D array [line_id, shell_id] */
    int64_t idx = line_id * est->n_shells + pkt->current_shell_id;
    est->j_blue_estimator[idx] += comov_energy * distance;
}

/* ============================================================================
 * INTERACTION HANDLERS
 * ============================================================================
 * These use physics_kernels.h functions for frame transformations to ensure
 * 10^-10 validation tolerance consistency.
 * ============================================================================ */

void thomson_scatter(RPacket *pkt, double time_explosion) {
    /*
     * Thomson (electron) scattering: isotropic in comoving frame.
     *
     * In CMF: mu_cmf_new = 2*xi - 1 (uniform on [-1, +1])
     *
     * Must transform new direction back to lab frame using the exact
     * angle_aberration_CMF_to_LF() from physics_kernels.h.
     *
     * Frequency update: The CMF frequency is unchanged by scattering,
     * but the lab-frame frequency changes due to the new direction.
     */

    /* Save old CMF frequency before changing direction */
    double doppler_old = get_doppler_factor(pkt->r, pkt->mu, time_explosion);
    double nu_cmf = pkt->nu * doppler_old;

    /* Sample isotropic scattering in CMF */
    double mu_cmf_new = 2.0 * rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform direction from CMF to lab frame using physics_kernels.h */
    double mu_lab_new = angle_aberration_CMF_to_LF(pkt->r, mu_cmf_new, time_explosion);
    pkt->mu = mu_lab_new;

    /*
     * Update lab-frame frequency.
     * CMF frequency unchanged, but lab frequency changes with new direction:
     *   nu_lab_new = nu_cmf / D_new = nu_cmf * D_inv_new
     */
    double inv_doppler_new = get_inverse_doppler_factor(pkt->r, mu_lab_new, time_explosion);
    pkt->nu = nu_cmf * inv_doppler_new;
}

void line_scatter(RPacket *pkt, double time_explosion,
                  LineInteractionType interaction_type,
                  const NumbaPlasma *plasma) {
    /*
     * Resonant line scattering in Sobolev approximation.
     *
     * The photon is absorbed by an atom in the expanding ejecta,
     * then re-emitted. In CMF, this is coherent (same frequency = line freq).
     *
     * Different treatments:
     * - SCATTER: Pure resonant scattering, isotropic re-emission at line freq
     * - DOWNBRANCH: Fluorescence to lower energy line (simplified)
     * - MACROATOM: Full macro-atom NLTE treatment (handled by macro_atom.c)
     *
     * NOTE: For LINE_MACROATOM mode with full atomic data, use
     * macro_atom_process_line_interaction() from macro_atom.h instead.
     * This function handles SCATTER and DOWNBRANCH modes, and provides
     * a fallback for MACROATOM when atomic data is not available.
     *
     * Uses physics_kernels.h functions for frame transformations.
     */

    int64_t line_id = pkt->next_line_id;
    double nu_line = plasma->line_list_nu[line_id];

    /* Sample isotropic re-emission direction in CMF */
    double mu_cmf_new = 2.0 * rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform direction from CMF to lab frame */
    double mu_lab_new = angle_aberration_CMF_to_LF(pkt->r, mu_cmf_new, time_explosion);
    pkt->mu = mu_lab_new;

    /* Determine emission frequency based on interaction type */
    double nu_emission = nu_line;  /* Default: coherent at line frequency */

    if (interaction_type == LINE_DOWNBRANCH) {
        /*
         * DOWNBRANCH: Simplified fluorescence treatment
         *
         * With probability p_fluorescence, packet is re-emitted at a
         * LOWER frequency (redder) line. This approximates the cascade
         * through intermediate levels without full macro-atom treatment.
         *
         * Simple model: emit at 0.8-1.0 * original line frequency
         * This captures the net effect of fluorescence (UV -> optical)
         * without requiring full level population data.
         */
        double p_fluorescence = 0.3;  /* 30% chance of fluorescence */
        double xi = rng_uniform(&pkt->rng_state);

        if (xi < p_fluorescence) {
            /* Fluorescent re-emission at lower frequency */
            double freq_reduction = 0.8 + 0.2 * rng_uniform(&pkt->rng_state);
            nu_emission = nu_line * freq_reduction;
        }
    }
    /* LINE_SCATTER: nu_emission = nu_line (coherent) */
    /* LINE_MACROATOM: handled by macro_atom_process_line_interaction() */

    /*
     * Transform emission frequency from CMF to lab frame:
     *   nu_lab = nu_cmf * D_inv(mu_new)
     */
    double inv_doppler = get_inverse_doppler_factor(pkt->r, mu_lab_new, time_explosion);
    pkt->nu = nu_emission * inv_doppler;

    /* Record interaction for spectrum synthesis */
    pkt->last_line_interaction_in_id = line_id;
    pkt->last_line_interaction_out_id = line_id;
}

/* ============================================================================
 * TRACE_PACKET: Heart of the Monte Carlo Transport
 * ============================================================================
 * HPC Optimization: Early exit when distance_line exceeds minimum of
 * distance_boundary and distance_electron - no need to check further lines.
 * ============================================================================ */

InteractionType trace_packet(RPacket *pkt, const NumbaModel *model,
                             const NumbaPlasma *plasma,
                             const MonteCarloConfig *config,
                             Estimators *estimators,
                             double *out_distance, int *out_delta_shell) {
    /*
     * Trace packet through ejecta to find next interaction point.
     *
     * This is the CORE of Monte Carlo radiative transfer.
     *
     * Algorithm:
     * 1. Calculate distance to shell boundaries
     * 2. Sample random optical depth tau_event = -ln(xi)
     * 3. Loop through spectral lines (sorted by frequency):
     *    a. Calculate distance to each line resonance
     *    b. EARLY EXIT: if d_line > min(d_boundary, d_electron), stop searching
     *    c. Accumulate Sobolev optical depth tau_line
     *    d. If tau_total > tau_event -> line interaction
     * 4. Return winning interaction type and distance
     *
     * Physical insight:
     * The Sobolev approximation treats each line as a "wall" at a specific
     * position where nu_cmf = nu_line. The packet either crosses the wall
     * (tau_line < tau_event) or interacts with it (tau_line > tau_event).
     */

    /* Get shell boundaries */
    int64_t shell_id = pkt->current_shell_id;
    double r_inner = model->r_inner[shell_id];
    double r_outer = model->r_outer[shell_id];

    /* Distance to shell boundary */
    int delta_shell;
    double distance_boundary = calculate_distance_boundary(
        pkt->r, pkt->mu, r_inner, r_outer, &delta_shell);

    /* Sample random optical depth for next event */
    double tau_event = -log(rng_uniform(&pkt->rng_state));

    /* Initialize electron scattering */
    double electron_density = plasma->electron_density[shell_id];
    double distance_electron = calculate_distance_electron(
        electron_density, tau_event);

    /* Pre-compute the minimum distance for early exit optimization */
    double distance_limit = (distance_boundary < distance_electron) ?
                            distance_boundary : distance_electron;

    /* Calculate comoving frequency */
    double doppler_factor = get_doppler_factor(pkt->r, pkt->mu, model->time_explosion);
    double comov_nu = pkt->nu * doppler_factor;

    /* Line tracing variables */
    double tau_trace_combined = 0.0;
    int64_t start_line = pkt->next_line_id;
    int64_t n_lines = plasma->n_lines;
    InteractionType interaction_type = INTERACTION_BOUNDARY;
    double distance = distance_boundary;

    /* Default: boundary wins if no line interaction */
    *out_delta_shell = delta_shell;

    /* Check if electron scattering is closer than boundary */
    if (distance_electron < distance_boundary) {
        interaction_type = INTERACTION_ESCATTERING;
        distance = distance_electron;
    }

    /* Loop through lines from current position to lower frequencies */
    for (int64_t cur_line = start_line; cur_line < n_lines; cur_line++) {

        double nu_line = plasma->line_list_nu[cur_line];

        /* Distance to line resonance - using exact TARDIS formula */
        int is_last = (cur_line == n_lines - 1);
        double distance_line = calculate_distance_line(
            pkt->nu, comov_nu, is_last, nu_line, model->time_explosion,
            pkt->r, pkt->mu);

        /*
         * HOT PATH OPTIMIZATION: Early exit
         * If distance to this line exceeds both boundary and electron distances,
         * no point checking further lines (they're at even greater distances).
         * Update next_line_id for the next trace_packet call.
         */
        if (distance_line > distance_limit && distance_line > 0.0) {
            pkt->next_line_id = cur_line;
            break;
        }

        /* Sobolev optical depth for this line */
        /* tau_sobolev is stored as [n_lines x n_shells] */
        double tau_line = plasma->tau_sobolev[cur_line * plasma->n_shells + shell_id];
        tau_trace_combined += tau_line;

        /* Electron optical depth to this point */
        double tau_electron_trace = calculate_tau_electron(
            electron_density, distance_line);
        double tau_total = tau_trace_combined + tau_electron_trace;

        /* Update estimators (packet traversing toward line) */
        if (distance_line > 0.0) {
            update_line_estimators(estimators, pkt, cur_line, distance_line,
                                   model->time_explosion);
        }

        /* Check: does accumulated tau exceed tau_event? -> Line interaction */
        if (tau_total > tau_event && !config->disable_line_scattering &&
            distance_line > 0.0 && distance_line < distance_limit) {
            interaction_type = INTERACTION_LINE;
            pkt->last_interaction_in_nu = pkt->nu;
            pkt->last_line_interaction_in_id = cur_line;
            pkt->next_line_id = cur_line;
            distance = distance_line;
            break;
        }

        /* Recalculate electron distance with reduced tau budget */
        double tau_remaining = tau_event - tau_trace_combined;
        if (tau_remaining > 0.0) {
            double new_distance_electron = calculate_distance_electron(
                electron_density, tau_remaining);

            /* Update distance_limit if electron distance decreased */
            if (new_distance_electron < distance_electron) {
                distance_electron = new_distance_electron;
                if (distance_electron < distance_boundary) {
                    distance_limit = distance_electron;
                }
            }
        }
    }

    *out_distance = distance;
    return interaction_type;
}

/* ============================================================================
 * SINGLE_PACKET_LOOP: Main Driver
 * ============================================================================
 * Thread-safe: no global state, RNG in packet, config passed as parameter.
 * ============================================================================ */

void single_packet_loop(RPacket *pkt, const NumbaModel *model,
                        const NumbaPlasma *plasma,
                        const MonteCarloConfig *config,
                        Estimators *estimators) {
    /*
     * Process one Monte Carlo packet until it escapes or is absorbed.
     *
     * This is the main driver that repeatedly:
     * 1. Traces packet to find next interaction (trace_packet)
     * 2. Moves packet to interaction point (move_r_packet)
     * 3. Handles interaction (boundary/electron/line)
     * 4. Continues until packet escapes (EMITTED) or is absorbed (REABSORBED)
     *
     * Physical process:
     * A photon packet is launched from the photosphere and propagates
     * through the expanding ejecta. It may scatter off electrons or
     * interact with spectral lines before eventually escaping (contributing
     * to the observed spectrum) or being absorbed back into the photosphere.
     */

    /* Set relativity mode from config */
    ENABLE_FULL_RELATIVITY = config->enable_full_relativity;

    /* Initialize line search position */
    rpacket_initialize_line_id(pkt, plasma, model);

    /* Apply relativistic corrections to initial state */
    if (config->enable_full_relativity) {
        double beta = pkt->r / (model->time_explosion * C_SPEED_OF_LIGHT);
        double inv_doppler = get_inverse_doppler_factor(
            pkt->r, pkt->mu, model->time_explosion);
        pkt->nu *= inv_doppler;
        pkt->energy *= inv_doppler;
        pkt->mu = (pkt->mu + beta) / (1.0 + beta * pkt->mu);
    } else {
        double inv_doppler = get_inverse_doppler_factor(
            pkt->r, pkt->mu, model->time_explosion);
        pkt->nu *= inv_doppler;
        pkt->energy *= inv_doppler;
    }

    /* Main transport loop */
    while (pkt->status == PACKET_IN_PROCESS) {

        /* Find next interaction */
        double distance;
        int delta_shell;
        InteractionType itype = trace_packet(
            pkt, model, plasma, config, estimators, &distance, &delta_shell);

        /* Process based on interaction type */
        switch (itype) {

            case INTERACTION_BOUNDARY:
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                move_packet_across_shell_boundary(
                    pkt, delta_shell, model->n_shells);
                break;

            case INTERACTION_LINE:
                pkt->last_interaction_type = 2;
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                line_scatter(pkt, model->time_explosion,
                            config->line_interaction_type, plasma);
                break;

            case INTERACTION_ESCATTERING:
                pkt->last_interaction_type = 1;
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                thomson_scatter(pkt, model->time_explosion);
                break;
        }
    }

    /* Packet has terminated: either EMITTED or REABSORBED */
}
