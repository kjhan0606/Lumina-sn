/**
 * LUMINA-SN Virtual Packet Implementation (TARDIS-compatible)
 * virtual_packet.c - Implementation of virtual packet spectrum synthesis
 *
 * This implements the TARDIS virtual packet technique exactly as described in:
 * Kerzendorf & Sim (2014), MNRAS 440, 387
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "virtual_packet.h"
#include "simulation_state.h"

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define CONST_C 2.99792458e10        /* Speed of light [cm/s] */
#define SIGMA_THOMSON 6.6524587158e-25  /* Thomson cross-section [cm^2] */

/* ============================================================================
 * VIRTUAL PACKET COLLECTION
 * ============================================================================ */

void vpacket_collection_init(VPacketCollection *coll, int64_t capacity,
                             double nu_min, double nu_max, int64_t n_bins) {
    coll->capacity = capacity;
    coll->n_packets = 0;
    coll->nus = (double *)calloc(capacity, sizeof(double));
    coll->energies = (double *)calloc(capacity, sizeof(double));

    coll->nu_min = nu_min;
    coll->nu_max = nu_max;
    coll->n_bins = n_bins;
    coll->spectrum = (double *)calloc(n_bins, sizeof(double));
    coll->counts = (int64_t *)calloc(n_bins, sizeof(int64_t));
}

void vpacket_collection_free(VPacketCollection *coll) {
    if (coll->nus) free(coll->nus);
    if (coll->energies) free(coll->energies);
    if (coll->spectrum) free(coll->spectrum);
    if (coll->counts) free(coll->counts);
    memset(coll, 0, sizeof(VPacketCollection));
}

void vpacket_collection_reset(VPacketCollection *coll) {
    coll->n_packets = 0;
    memset(coll->spectrum, 0, coll->n_bins * sizeof(double));
    memset(coll->counts, 0, coll->n_bins * sizeof(int64_t));
}

void vpacket_collection_add(VPacketCollection *coll, double nu, double energy) {
    /* Add to raw list if space available */
    if (coll->n_packets < coll->capacity) {
        coll->nus[coll->n_packets] = nu;
        coll->energies[coll->n_packets] = energy;
        coll->n_packets++;
    }

    /* Add to binned spectrum */
    if (nu >= coll->nu_min && nu < coll->nu_max && energy > 0) {
        double d_nu = (coll->nu_max - coll->nu_min) / coll->n_bins;
        int64_t bin = (int64_t)((nu - coll->nu_min) / d_nu);
        if (bin >= 0 && bin < coll->n_bins) {
            coll->spectrum[bin] += energy;
            coll->counts[bin]++;
        }
    }
}

/* ============================================================================
 * CORE VIRTUAL PACKET FUNCTIONS
 * ============================================================================ */

/**
 * Calculate distance to shell boundary (TARDIS calculate_distance_boundary)
 * Local version to avoid conflicts with physics_kernels.h
 */
static double vpacket_calc_distance_boundary(double r, double mu,
                                              double r_inner, double r_outer,
                                              int *delta_shell) {
    double mu_sq_minus_1 = mu * mu - 1.0;

    if (mu > 0.0) {
        /* Moving outward */
        double discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r;
        *delta_shell = 1;
        return sqrt(discriminant) - r * mu;
    } else {
        /* Moving inward - check if hits inner boundary */
        double check = r_inner * r_inner + r * r * mu_sq_minus_1;
        if (check >= 0.0) {
            *delta_shell = -1;
            return -r * mu - sqrt(check);
        } else {
            /* Misses inner boundary, goes to outer */
            double discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r;
            *delta_shell = 1;
            return sqrt(discriminant) - r * mu;
        }
    }
}

/**
 * Calculate distance to line interaction (TARDIS calculate_distance_line)
 * Local version to avoid conflicts with physics_kernels.h
 */
static double vpacket_calc_distance_line(double nu_lab, double nu_cmf,
                                          double nu_line, double t_exp,
                                          int is_last_line) {
    if (is_last_line) return 1e99;

    double nu_diff = nu_cmf - nu_line;

    /* Close line threshold */
    if (fabs(nu_diff / nu_lab) < 1e-7) {
        nu_diff = 0.0;
    }

    if (nu_diff >= 0) {
        return (nu_diff / nu_lab) * CONST_C * t_exp;
    } else {
        return 1e99;  /* Line is above current frequency */
    }
}

/**
 * Trace virtual packet within a single shell
 * Returns optical depth accumulated in this shell
 */
double trace_vpacket_in_shell(VirtualPacket *vpkt,
                              const ShellState *shell,
                              const SimulationState *state,
                              double *d_boundary, int *delta_shell) {
    /* Distance to boundary */
    *d_boundary = vpacket_calc_distance_boundary(vpkt->r, vpkt->mu,
                                                  shell->r_inner, shell->r_outer,
                                                  delta_shell);

    /* Electron scattering opacity */
    double chi_e = shell->plasma.n_e * SIGMA_THOMSON;

    /* Doppler factor */
    double beta = vpkt->r / (state->t_explosion * CONST_C);
    double D = 1.0 - beta * vpkt->mu;
    double nu_cmf = vpkt->nu * D;

    /* Continuum optical depth */
    double tau_continuum = chi_e * (*d_boundary);
    double tau_total = tau_continuum;

    /* Line optical depth - loop through lines */
    const ActiveLine *lines = shell->active_lines;
    int64_t n_lines = shell->n_active_lines;

    for (int64_t i = vpkt->next_line_id; i < n_lines; i++) {
        double nu_line = lines[i].nu;
        double tau_line = lines[i].tau_sobolev;

        int is_last = (i == n_lines - 1);
        double d_line = vpacket_calc_distance_line(vpkt->nu, nu_cmf, nu_line,
                                                    state->t_explosion, is_last);

        /* If line is beyond boundary, stop */
        if (*d_boundary <= d_line) {
            vpkt->next_line_id = i;
            break;
        }

        /* Accumulate line optical depth */
        tau_total += tau_line;
    }

    return tau_total;
}

/**
 * Trace a single virtual packet to escape
 * Returns total optical depth
 */
double trace_vpacket(VirtualPacket *vpkt, const SimulationState *state) {
    double tau_total = 0.0;
    int step = 0;

    while (vpkt->status == 0 && step < VPACKET_MAX_SHELLS * 10) {
        step++;

        /* Find current shell */
        int shell_id = -1;
        for (int i = 0; i < state->n_shells; i++) {
            if (vpkt->r >= state->shells[i].r_inner &&
                vpkt->r < state->shells[i].r_outer) {
                shell_id = i;
                break;
            }
        }

        /* Check boundaries */
        if (shell_id < 0) {
            if (vpkt->r >= state->shells[state->n_shells - 1].r_outer) {
                vpkt->status = 1;  /* Escaped */
            } else {
                vpkt->status = 2;  /* Absorbed (hit inner boundary) */
            }
            break;
        }

        const ShellState *shell = &state->shells[shell_id];
        vpkt->current_shell = shell_id;

        /* Trace within this shell */
        double d_boundary;
        int delta_shell;
        double tau_shell = trace_vpacket_in_shell(vpkt, shell, state,
                                                   &d_boundary, &delta_shell);
        tau_total += tau_shell;

        /* Russian roulette for high optical depth */
        if (tau_total > VPACKET_TAU_RUSSIAN) {
            if (drand48() > VPACKET_SURVIVAL_PROB) {
                vpkt->energy = 0.0;
                vpkt->status = 1;  /* Effectively absorbed */
                break;
            } else {
                vpkt->energy *= exp(-tau_total) / VPACKET_SURVIVAL_PROB;
                tau_total = 0.0;
            }
        }

        /* Move packet to boundary */
        double r_new = sqrt(vpkt->r * vpkt->r + d_boundary * d_boundary +
                           2.0 * vpkt->r * d_boundary * vpkt->mu);
        vpkt->mu = (vpkt->mu * vpkt->r + d_boundary) / r_new;
        vpkt->r = r_new;

        /* Check for escape */
        if (vpkt->r >= state->shells[state->n_shells - 1].r_outer) {
            vpkt->status = 1;
            break;
        }
        if (vpkt->r <= state->shells[0].r_inner) {
            vpkt->status = 2;
            break;
        }
    }

    return tau_total;
}

/**
 * Spawn virtual packets from r-packet position
 * This is the TARDIS trace_vpacket_volley function
 */
void spawn_vpacket_volley(double r_pkt_r, double r_pkt_mu,
                          double r_pkt_nu, double r_pkt_energy,
                          int shell_id, int64_t next_line_id,
                          const SimulationState *state,
                          VPacketCollection *coll,
                          int n_vpackets) {
    if (n_vpackets <= 0) return;

    double r_inner = state->shells[0].r_inner;
    double t_exp = state->t_explosion;

    /* Calculate mu_min (minimum angle toward observer) */
    double mu_min;
    int on_inner_boundary;

    if (r_pkt_r > r_inner) {
        /* Not on inner boundary */
        double r_inner_over_r = r_inner / r_pkt_r;
        mu_min = -sqrt(1.0 - r_inner_over_r * r_inner_over_r);
        on_inner_boundary = 0;
    } else {
        /* On inner boundary */
        mu_min = 0.0;
        on_inner_boundary = 1;
    }

    /* R-packet Doppler factor */
    double beta = r_pkt_r / (t_exp * CONST_C);
    double D_rpacket = 1.0 - beta * r_pkt_mu;

    /* Mu bin width */
    double mu_bin = (1.0 - mu_min) / n_vpackets;

    /* Spawn each virtual packet */
    for (int i = 0; i < n_vpackets; i++) {
        /* Random mu within bin (CMF direction toward observer) */
        double v_mu = mu_min + i * mu_bin + drand48() * mu_bin;

        /* Weight calculation (K&S 2014) */
        double weight;
        if (on_inner_boundary) {
            /* On inner boundary: weight = 2 * mu / n_vpackets */
            weight = 2.0 * v_mu / n_vpackets;
        } else {
            /* Inside ejecta: weight = (1 - mu_min) / (2 * n_vpackets) */
            weight = (1.0 - mu_min) / (2.0 * n_vpackets);
        }

        /* V-packet Doppler factor */
        double D_vpacket = 1.0 - beta * v_mu;

        /* Doppler factor ratio for frequency/energy transformation */
        double D_ratio = D_rpacket / D_vpacket;

        /* V-packet properties */
        double v_nu = r_pkt_nu * D_ratio;
        double v_energy = r_pkt_energy * weight * D_ratio;

        /* Create virtual packet */
        VirtualPacket vpkt;
        vpkt.r = r_pkt_r;
        vpkt.mu = v_mu;
        vpkt.nu = v_nu;
        vpkt.energy = v_energy;
        vpkt.current_shell = shell_id;
        vpkt.next_line_id = next_line_id;
        vpkt.status = 0;

        /* Trace to escape */
        double tau = trace_vpacket(&vpkt, state);

        /* Attenuate by optical depth */
        vpkt.energy *= exp(-tau);

        /* Add to collection if escaped with energy */
        if (vpkt.status == 1 && vpkt.energy > 0) {
            vpacket_collection_add(coll, vpkt.nu, vpkt.energy);
        }
    }
}

/* ============================================================================
 * SPECTRUM OUTPUT
 * ============================================================================ */

void vpacket_collection_get_spectrum(const VPacketCollection *coll,
                                     double *wavelength, double *flux,
                                     int64_t *n_points) {
    double d_nu = (coll->nu_max - coll->nu_min) / coll->n_bins;

    for (int64_t i = 0; i < coll->n_bins; i++) {
        double nu = coll->nu_min + (i + 0.5) * d_nu;
        wavelength[i] = CONST_C / nu * 1e8;  /* Angstrom */
        flux[i] = coll->spectrum[i] / d_nu;  /* erg/s/Hz */
    }
    *n_points = coll->n_bins;
}

int vpacket_spectrum_write_csv(const VPacketCollection *coll,
                               const char *filename, double t_exp) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "# LUMINA Virtual Packet Spectrum (TARDIS-compatible)\n");
    fprintf(fp, "# t_exp = %.2f days\n", t_exp / 86400.0);
    fprintf(fp, "# n_vpackets_collected = %ld\n", (long)coll->n_packets);
    fprintf(fp, "wavelength_A,frequency_Hz,L_nu_erg_s_Hz,counts\n");

    double d_nu = (coll->nu_max - coll->nu_min) / coll->n_bins;

    for (int64_t i = 0; i < coll->n_bins; i++) {
        double nu = coll->nu_min + (i + 0.5) * d_nu;
        double wl = CONST_C / nu * 1e8;
        double L_nu = coll->spectrum[i] / d_nu;

        fprintf(fp, "%.4f,%.6e,%.6e,%ld\n",
                wl, nu, L_nu, (long)coll->counts[i]);
    }

    fclose(fp);
    return 0;
}
