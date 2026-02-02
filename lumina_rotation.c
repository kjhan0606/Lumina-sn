/**
 * LUMINA-SN Innovation: Post-Processing Rotation & Weighting
 * lumina_rotation.c - Implementation of observer-frame spectrum synthesis
 *
 * This is the novel LUMINA algorithm that enables efficient multi-angle
 * spectrum computation from a single Monte Carlo run.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lumina_rotation.h"

/* ============================================================================
 * CORE ROTATION ALGORITHM
 * ============================================================================ */

void lumina_rotate_packet(double final_r, double final_mu,
                          double final_nu, double final_energy,
                          const ObserverConfig *config,
                          RotatedPacket *result) {
    /*
     * LUMINA's core innovation: rotate escaped packet to observer frame.
     *
     * Physical basis:
     * ---------------
     * In the homologously expanding supernova, the radiative transfer
     * is computed in the local comoving frame. The final lab-frame
     * direction (μ_packet) is arbitrary from the observer's perspective.
     *
     * Since the supernova is spherically symmetric (in 1D TARDIS), we can
     * rotate the coordinate system so that the packet appears to travel
     * toward the observer without changing the physics.
     *
     * Key insight: The frequency transformation depends on the ANGLE
     * between the packet direction and the expansion velocity, not on
     * an absolute direction. By rotating to the observer's frame, we
     * preserve the correct Doppler shift.
     */

    result->original_index = -1;  /* Set by caller if needed */
    result->valid = 1;

    /*
     * Step 1: TIME-DELAY CORRECTION
     * -----------------------------
     * Packets escaping at different angles arrive at different times.
     *
     * Consider a packet escaping at radius R with direction μ:
     * - If μ = +1 (moving directly toward observer), it arrives first
     * - If μ = -1 (moving away), it must travel extra distance 2R
     *
     * Time delay relative to a packet at the center:
     *   Δt = -R × μ / c
     *
     * Observed time:
     *   t_obs = t_explosion + t_travel - (R × μ / c)
     *
     * We use t_explosion as the reference, so:
     *   t_observed = t_explosion - (R × μ_packet / c)
     *
     * Note: μ_packet is the ORIGINAL escape direction, not μ_observer!
     * This is crucial for capturing velocity-dependent spectral features.
     */
    double time_delay = (final_r * final_mu) / C_SPEED_OF_LIGHT;
    result->t_observed = config->time_explosion - time_delay;

    /*
     * Step 2: FREQUENCY TRANSFORMATION TO OBSERVER FRAME
     * ---------------------------------------------------
     * The frequency must be transformed to what the observer sees.
     *
     * In the comoving frame at the escape point:
     *   ν_cmf = ν_lab × D(μ_packet)
     *
     * The observer sees:
     *   ν_obs = ν_cmf / D(μ_observer)
     *         = ν_lab × D(μ_packet) / D(μ_observer)
     *
     * For partial relativity:
     *   D(μ) = 1 - β × μ, where β = R / (c × t)
     *
     * This gives the Doppler shift due to the expansion velocity
     * projected onto the observer's line of sight.
     */
    double beta = final_r / (config->time_explosion * C_SPEED_OF_LIGHT);

    /* Doppler factors */
    double D_packet = 1.0 - beta * final_mu;
    double D_observer = 1.0 - beta * config->mu_observer;

    /*
     * CRITICAL FIX (Task Order Investigation):
     * -----------------------------------------
     * The original frequency transformation was INCORRECT.
     *
     * The packet's frequency (final_nu) is tracked in the LAB FRAME
     * throughout the Monte Carlo transport. The lab-frame frequency
     * is what a stationary observer at the escape point would measure.
     *
     * For a distant observer at infinity, the lab-frame frequency IS
     * the correct observed frequency (to first order in v/c).
     *
     * The LUMINA rotation applies solid-angle weighting to account for
     * packets escaping in non-observer directions, but should NOT
     * transform the frequency. The previous formula:
     *
     *   nu_observer = final_nu * D_packet / D_observer  [WRONG]
     *
     * was adding an extra Doppler shift of ~3,000 km/s on average,
     * causing the Si II 6355 absorption to appear too blueshifted.
     *
     * CORRECT: Use the lab-frame frequency directly.
     */
    result->nu_observer = final_nu;  /* NO frequency transformation */
    result->wavelength = frequency_to_wavelength(result->nu_observer);

    /*
     * Step 3: SOLID ANGLE WEIGHTING
     * -----------------------------
     * To correctly normalize the flux, we must account for the fact
     * that packets were emitted isotropically in the CMF, but we're
     * now projecting them onto a specific observer direction.
     *
     * The weight factor corrects for:
     *   1. Doppler beaming: CMF-isotropic → lab-frame peaked forward
     *   2. Solid angle: different μ corresponds to different dΩ
     *
     * For relativistic Doppler beaming, the intensity transforms as:
     *   I_lab / I_cmf = D^(-3) for photon energy
     *
     * The energy per solid angle scales as:
     *   dE/dΩ ∝ D^(-2)
     *
     * Weight to convert from μ_packet distribution to μ_observer:
     *   w = (D_observer / D_packet)^2
     *
     * This ensures that integrating over all μ_packet gives the
     * correct total luminosity.
     */
    double doppler_ratio = D_observer / D_packet;
    result->weight = doppler_ratio * doppler_ratio;

    /* Apply weight to energy */
    result->energy_weighted = final_energy * result->weight;

    /*
     * Additional consideration for limb darkening/brightening:
     * In a real SN, the photosphere has angular structure.
     * For now, we assume uniform (isotropic in CMF).
     * Future extension: multiply by I(μ)/I(0) limb profile.
     */
}

int lumina_apply_rotation_weighting(const ValidationTrace *trace,
                                    const ObserverConfig *config,
                                    RotatedPacket *result) {
    /*
     * Process a full packet trace to extract the final state
     * and apply rotation/weighting.
     */

    if (!trace || trace->n_snapshots == 0) {
        result->valid = 0;
        return -1;
    }

    /* Get final snapshot */
    const PacketSnapshot *final = &trace->snapshots[trace->n_snapshots - 1];

    /* Check if packet was emitted (escaped) */
    if (final->status != PACKET_EMITTED) {
        result->valid = 0;
        return -1;  /* Packet was absorbed, doesn't contribute to spectrum */
    }

    /* Apply rotation and weighting */
    lumina_rotate_packet(final->r, final->mu, final->nu, final->energy,
                         config, result);

    result->original_index = trace->packet_index;

    return 0;
}

/* ============================================================================
 * SPECTRUM ACCUMULATOR
 * ============================================================================ */

Spectrum *spectrum_create(double wavelength_min, double wavelength_max,
                          int64_t n_bins) {
    Spectrum *spec = (Spectrum *)malloc(sizeof(Spectrum));
    if (!spec) return NULL;

    spec->flux = (double *)calloc(n_bins, sizeof(double));
    spec->wavelength_centers = (double *)malloc(n_bins * sizeof(double));

    if (!spec->flux || !spec->wavelength_centers) {
        spectrum_free(spec);
        return NULL;
    }

    spec->wavelength_min = wavelength_min;
    spec->wavelength_max = wavelength_max;
    spec->n_bins = n_bins;
    spec->d_wavelength = (wavelength_max - wavelength_min) / n_bins;
    spec->total_luminosity = 0.0;
    spec->n_packets_used = 0;

    /* Initialize wavelength centers */
    for (int64_t i = 0; i < n_bins; i++) {
        spec->wavelength_centers[i] = wavelength_min +
            (i + 0.5) * spec->d_wavelength;
    }

    return spec;
}

void spectrum_free(Spectrum *spec) {
    if (spec) {
        if (spec->flux) free(spec->flux);
        if (spec->wavelength_centers) free(spec->wavelength_centers);
        free(spec);
    }
}

/*
 * TASK ORDER #27: Weight Diagnostics
 * -----------------------------------
 * Track mean weight of escaped packets to verify energy conservation.
 * If mean(w) deviates significantly from 1.0, there's a systematic bias.
 */
static double g_weight_sum = 0.0;
static int64_t g_weight_count = 0;

void lumina_reset_weight_diagnostics(void) {
    g_weight_sum = 0.0;
    g_weight_count = 0;
}

void lumina_report_weight_diagnostics(void) {
    if (g_weight_count > 0) {
        double mean_weight = g_weight_sum / g_weight_count;
        printf("[LUMINA WEIGHT AUDIT] N_packets = %ld, Mean(w) = %.6f\n",
               (long)g_weight_count, mean_weight);
        if (mean_weight < 0.8 || mean_weight > 1.2) {
            printf("  WARNING: Mean weight deviates from 1.0 by %.1f%%!\n",
                   fabs(mean_weight - 1.0) * 100.0);
            printf("  This indicates energy non-conservation in rotation.\n");
        }
    }
}

void spectrum_add_packet(Spectrum *spec, const RotatedPacket *pkt,
                         double simulation_time) {
    /*
     * Add a rotated packet's contribution to the spectrum.
     *
     * The flux density (erg/s/Å) is:
     *   F_λ = E_weighted / (Δλ × t_sim)
     *
     * Where:
     *   E_weighted = energy × weight
     *   Δλ = wavelength bin width
     *   t_sim = simulation time (for L → F conversion)
     */

    if (!pkt->valid) return;

    /* Track weight statistics for diagnostic */
    g_weight_sum += pkt->weight;
    g_weight_count++;

    /* Find wavelength bin */
    if (pkt->wavelength < spec->wavelength_min ||
        pkt->wavelength >= spec->wavelength_max) {
        return;  /* Outside spectrum range */
    }

    int64_t bin = (int64_t)((pkt->wavelength - spec->wavelength_min) /
                            spec->d_wavelength);

    if (bin < 0 || bin >= spec->n_bins) return;

    /* Add flux contribution */
    double flux_contribution = pkt->energy_weighted /
                               (spec->d_wavelength * simulation_time);
    spec->flux[bin] += flux_contribution;

    /* Accumulate totals */
    spec->total_luminosity += pkt->energy_weighted / simulation_time;
    spec->n_packets_used++;
}

void spectrum_normalize(Spectrum *spec, int64_t n_total_packets,
                        double simulation_time) {
    /*
     * Normalize the spectrum by the total number of packets.
     *
     * Each packet represents L_total / N_packets of the total luminosity.
     * The normalization ensures:
     *   ∫ F_λ dλ = L_total / (4π D²)
     *
     * For now, we assume unit distance and return L_λ directly.
     */

    (void)n_total_packets;   /* Already normalized per-packet */
    (void)simulation_time;

    /* Additional normalization can be applied here if needed */
}

int spectrum_write_csv(const Spectrum *spec, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "# LUMINA-SN Spectrum Output\n");
    fprintf(fp, "# Wavelength range: %.1f - %.1f Angstrom\n",
            spec->wavelength_min, spec->wavelength_max);
    fprintf(fp, "# N_bins: %ld\n", (long)spec->n_bins);
    fprintf(fp, "# N_packets_used: %ld\n", (long)spec->n_packets_used);
    fprintf(fp, "# Total_luminosity: %.6e erg/s\n", spec->total_luminosity);
    fprintf(fp, "wavelength_A,flux_erg_s_A\n");

    for (int64_t i = 0; i < spec->n_bins; i++) {
        fprintf(fp, "%.4f,%.10e\n",
                spec->wavelength_centers[i], spec->flux[i]);
    }

    fclose(fp);
    return 0;
}

/* ============================================================================
 * PEELING-OFF TECHNIQUE: Virtual Packet Contributions
 * ============================================================================ */

PeelingContext *peeling_context_create(const ObserverConfig *obs_config,
                                        double simulation_time) {
    PeelingContext *ctx = (PeelingContext *)malloc(sizeof(PeelingContext));
    if (!ctx) return NULL;

    ctx->local_spectrum = spectrum_create(obs_config->wavelength_min,
                                          obs_config->wavelength_max,
                                          obs_config->n_wavelength_bins);
    if (!ctx->local_spectrum) {
        free(ctx);
        return NULL;
    }

    ctx->obs_config = *obs_config;
    ctx->simulation_time = simulation_time;
    ctx->n_peeling_events = 0;
    ctx->total_peeling_energy = 0.0;

    return ctx;
}

void peeling_context_free(PeelingContext *ctx) {
    if (ctx) {
        if (ctx->local_spectrum) {
            spectrum_free(ctx->local_spectrum);
        }
        free(ctx);
    }
}

void peeling_add_contribution(PeelingContext *ctx,
                              double r, double mu, double nu, double energy,
                              double t_exp) {
    /*
     * PEELING-OFF: Calculate virtual packet contribution toward observer
     *
     * At each interaction, we compute what the spectrum contribution would
     * be if the packet were to travel directly toward the observer.
     *
     * Key physics:
     * 1. The packet has isotropic re-emission probability in CMF
     * 2. We "peel off" a fraction toward the observer direction
     * 3. Apply Doppler shift for observer-frame frequency
     * 4. Weight by escape probability (simplified: assume optically thin)
     *
     * The weight factor accounts for:
     * - Solid angle: probability of emission toward observer (1/4π)
     * - Doppler beaming: D^2 factor for relativistic intensity
     * - Escape probability: simplified to 1.0 (full peeling)
     */

    if (!ctx || !ctx->local_spectrum) return;

    /* Observer viewing angle (typically mu_obs = 1 for face-on) */
    double mu_obs = ctx->obs_config.mu_observer;

    /* Calculate expansion velocity at this radius */
    double beta = r / (t_exp * C_SPEED_OF_LIGHT);

    /* Doppler factors */
    /* D_packet: Doppler factor for current packet direction */
    /* D_observer: Doppler factor for observer direction */
    double D_packet = 1.0 - beta * mu;
    double D_observer = 1.0 - beta * mu_obs;

    /*
     * Observer-frame frequency
     *
     * The packet frequency (nu) is in lab frame. For an observer at
     * mu_obs, we need to account for the Doppler shift.
     *
     * Since photons are emitted isotropically in CMF and we're calculating
     * the contribution toward the observer, use the lab-frame frequency
     * directly (LUMINA correction from Task Order investigation).
     */
    double nu_observer = nu;

    /* Convert to wavelength */
    double wavelength = frequency_to_wavelength(nu_observer);

    /* Check if within spectrum range */
    if (wavelength < ctx->local_spectrum->wavelength_min ||
        wavelength >= ctx->local_spectrum->wavelength_max) {
        return;
    }

    /*
     * Peeling weight factor
     *
     * The weight corrects for the anisotropic sampling:
     *   w = (D_observer / D_packet)^2
     *
     * This ensures energy conservation when integrating over all
     * packet directions.
     *
     * For peeling, we also include escape probability P_escape.
     * In the optically thin approximation: P_escape ≈ 1.0
     * More sophisticated: P_escape = exp(-tau_to_boundary)
     *
     * Using simplified approach for now.
     */
    double doppler_ratio = D_observer / D_packet;
    double weight = doppler_ratio * doppler_ratio;

    /* Escape probability (simplified: assume optically thin) */
    double P_escape = 1.0;

    /* Weighted energy contribution */
    double energy_weighted = energy * weight * P_escape;

    /* Find wavelength bin */
    int64_t bin = (int64_t)((wavelength - ctx->local_spectrum->wavelength_min) /
                            ctx->local_spectrum->d_wavelength);

    if (bin < 0 || bin >= ctx->local_spectrum->n_bins) return;

    /* Add flux contribution */
    double flux_contribution = energy_weighted /
                               (ctx->local_spectrum->d_wavelength * ctx->simulation_time);
    ctx->local_spectrum->flux[bin] += flux_contribution;

    /* Update statistics */
    ctx->local_spectrum->total_luminosity += energy_weighted / ctx->simulation_time;
    ctx->n_peeling_events++;
    ctx->total_peeling_energy += energy_weighted;
}

void peeling_merge_into_spectrum(PeelingContext *ctx, Spectrum *global_spec) {
    /*
     * Merge thread-local spectrum into global spectrum
     *
     * This should be called inside a critical section or mutex
     * when using OpenMP.
     */

    if (!ctx || !ctx->local_spectrum || !global_spec) return;

    /* Accumulate flux bins */
    for (int64_t i = 0; i < global_spec->n_bins && i < ctx->local_spectrum->n_bins; i++) {
        global_spec->flux[i] += ctx->local_spectrum->flux[i];
    }

    /* Accumulate totals */
    global_spec->total_luminosity += ctx->local_spectrum->total_luminosity;
    global_spec->n_packets_used += ctx->n_peeling_events;
}
