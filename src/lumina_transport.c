/* lumina_transport.c — Phase 3: CPU Transport Kernel
 * Line-by-line faithful port of TARDIS Python transport.
 * Source files: calculate_distances.py, frame_transformations.py,
 *               r_packet_transport.py, interaction_events.py,
 *               interaction_event_callers.py, macro_atom.py,
 *               radfield_estimator_calcs.py, single_packet_loop.py */

#include "lumina.h" /* Phase 3 - Step 1 */

/* ============================================================ */
/* Phase 3 - Step 2: Frame transformations                      */
/* (frame_transformations.py — partial relativity only)         */
/* ============================================================ */

/* Phase 3 - Step 2: Doppler factor (lab → comoving) */
/* TARDIS: get_doppler_factor_partial_relativity */
double get_doppler_factor(double r, double mu, double time_explosion) {
    double beta = r / (C_SPEED_OF_LIGHT * time_explosion); /* Phase 3 - Step 2 */
    return 1.0 - mu * beta; /* Phase 3 - Step 2 */
}

/* Phase 3 - Step 2: Inverse Doppler factor (comoving → lab) */
/* TARDIS: get_inverse_doppler_factor_partial_relativity */
double get_inverse_doppler_factor(double r, double mu, double time_explosion) {
    double beta = r / (C_SPEED_OF_LIGHT * time_explosion); /* Phase 3 - Step 2 */
    return 1.0 / (1.0 - mu * beta); /* Phase 3 - Step 2 */
}

/* Phase 3 - Step 2: Packet energy at distance along path */
/* TARDIS: calc_packet_energy (frame_transformations.py) */
static double calc_packet_energy(RPacket *pkt, double distance_trace,
                                  double time_explosion) {
    double doppler = 1.0 - (distance_trace + pkt->mu * pkt->r) / /* Phase 3 - Step 2 */
                     (time_explosion * C_SPEED_OF_LIGHT); /* Phase 3 - Step 2 */
    return pkt->energy * doppler; /* Phase 3 - Step 2 */
}

/* ============================================================ */
/* Phase 3 - Step 3: Distance calculations                      */
/* (calculate_distances.py)                                     */
/* ============================================================ */

/* Phase 3 - Step 3: Distance to shell boundary */
/* TARDIS: calculate_distance_boundary */
void calculate_distance_boundary(double r, double mu, double r_inner,
                                  double r_outer, double *out_distance,
                                  int *out_delta_shell) {
    if (mu > 0.0) { /* Phase 3 - Step 3: outward-moving packet */
        *out_distance = sqrt(r_outer * r_outer + (mu * mu - 1.0) * r * r) /* Phase 3 - Step 3 */
                        - r * mu; /* Phase 3 - Step 3 */
        *out_delta_shell = 1; /* Phase 3 - Step 3 */
    } else { /* Phase 3 - Step 3: inward-moving packet */
        double check = r_inner * r_inner + r * r * (mu * mu - 1.0); /* Phase 3 - Step 3 */
        if (check >= 0.0) { /* Phase 3 - Step 3: hits inner boundary */
            *out_distance = -r * mu - sqrt(check); /* Phase 3 - Step 3 */
            *out_delta_shell = -1; /* Phase 3 - Step 3 */
        } else { /* Phase 3 - Step 3: misses inner, bounces to outer */
            *out_distance = sqrt(r_outer * r_outer + /* Phase 3 - Step 3 */
                                 (mu * mu - 1.0) * r * r) - r * mu; /* Phase 3 - Step 3 */
            *out_delta_shell = 1; /* Phase 3 - Step 3 */
        }
    }
}

/* Phase 3 - Step 3: Distance to line resonance */
/* TARDIS: calculate_distance_line (calculate_distances.py) */
/* TARDIS divides by r_packet.nu (lab frame), NOT comov_nu */
double calculate_distance_line(double comov_nu, double nu_lab,
                                int is_last_line, double nu_line,
                                double time_explosion) {
    if (is_last_line) { /* Phase 3 - Step 3 */
        return MISS_DISTANCE; /* Phase 3 - Step 3 */
    }
    double nu_diff = comov_nu - nu_line; /* Phase 3 - Step 3 */
    if (fabs(nu_diff / nu_lab) < CLOSE_LINE_THRESHOLD) { /* Phase 3 - Step 3 */
        nu_diff = 0.0; /* Phase 3 - Step 3 */
    }
    if (nu_diff >= 0.0) { /* Phase 3 - Step 3 */
        return (nu_diff / nu_lab) * C_SPEED_OF_LIGHT * time_explosion; /* Phase 3 - Step 3 */
    }
    /* Phase 3 - Step 3: nu_diff < 0 should not happen in TARDIS */
    /* This means the packet frequency is below the line; return MISS */
    return MISS_DISTANCE; /* Phase 3 - Step 3 */
}

/* Phase 3 - Step 3: Distance to electron scattering */
/* TARDIS: calculate_distance_electron */
double calculate_distance_electron(double electron_density, double tau_event) {
    return tau_event / (electron_density * SIGMA_THOMSON); /* Phase 3 - Step 3 */
}

/* ============================================================ */
/* Phase 3 - Step 4: Estimator updates                          */
/* (radfield_estimator_calcs.py)                                */
/* ============================================================ */

/* Phase 3 - Step 4: Base J and nu_bar estimators */
/* TARDIS: update_base_estimators */
void update_base_estimators(RPacket *pkt, double distance, Estimators *est,
                             double comov_nu, double comov_energy) {
    int shell = pkt->current_shell_id; /* Phase 3 - Step 4 */
    est->j_estimator[shell] += comov_energy * distance; /* Phase 3 - Step 4 */
    est->nu_bar_estimator[shell] += comov_energy * distance * comov_nu; /* Phase 3 - Step 4 */
}

/* Phase 3 - Step 4: Line-specific j_blue and Edotlu */
/* TARDIS: update_line_estimators */
void update_line_estimators(Estimators *est, RPacket *pkt, int cur_line_id,
                             double distance_trace, double time_explosion) {
    if (est->n_lines == 0) return; /* Phase 3 - Step 4: skip if lightweight estimator */
    double energy = calc_packet_energy(pkt, distance_trace, time_explosion); /* Phase 3 - Step 4 */
    int shell = pkt->current_shell_id; /* Phase 3 - Step 4 */
    int n_shells = est->n_shells; /* Phase 3 - Step 4 */
    est->j_blue_estimator[cur_line_id * n_shells + shell] += /* Phase 3 - Step 4 */
        energy / pkt->nu; /* Phase 3 - Step 4 */
    est->Edotlu_estimator[cur_line_id * n_shells + shell] += energy; /* Phase 3 - Step 4 */
}

/* ============================================================ */
/* Phase 3 - Step 5: trace_packet                               */
/* (r_packet_transport.py: trace_packet)                        */
/* This is the core loop that finds the next interaction.       */
/* ============================================================ */

void trace_packet(RPacket *pkt, Geometry *geo, OpacityState *opacity,
                   Estimators *est, double chi_continuum,
                   bool disable_line_scattering, RNG *rng,
                   double *out_distance, InteractionType *out_type,
                   int *out_delta_shell) {

    int shell = pkt->current_shell_id; /* Phase 3 - Step 5 */
    double r_inner = geo->r_inner[shell]; /* Phase 3 - Step 5 */
    double r_outer = geo->r_outer[shell]; /* Phase 3 - Step 5 */

    /* Phase 3 - Step 5: Distance to shell boundary */
    double distance_boundary; /* Phase 3 - Step 5 */
    int delta_shell; /* Phase 3 - Step 5 */
    calculate_distance_boundary(pkt->r, pkt->mu, r_inner, r_outer, /* Phase 3 - Step 5 */
                                 &distance_boundary, &delta_shell); /* Phase 3 - Step 5 */

    /* Phase 3 - Step 5: Sample optical depth for next interaction */
    double tau_event = -log(rng_uniform(rng)); /* Phase 3 - Step 5 */
    double tau_trace_line_combined = 0.0; /* Phase 3 - Step 5 */

    /* Phase 3 - Step 5: Doppler factor at current position */
    double doppler_factor = get_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 5 */
                                                geo->time_explosion); /* Phase 3 - Step 5 */
    double comov_nu = pkt->nu * doppler_factor; /* Phase 3 - Step 5 */

    /* Phase 3 - Step 5: Continuum distance */
    double distance_continuum = tau_event / chi_continuum; /* Phase 3 - Step 5 */

    int start_line_id = pkt->next_line_id; /* Phase 3 - Step 5 */
    int n_lines = opacity->n_lines; /* Phase 3 - Step 5 */
    int n_shells = opacity->n_shells; /* Phase 3 - Step 5 */
    int last_line_id = n_lines - 1; /* Phase 3 - Step 5 */
    int cur_line_id = start_line_id; /* Phase 3 - Step 5: for Numba compat */

    /* Phase 3 - Step 5: Main line-tracing loop (TARDIS for...else pattern) */
    bool broke_out = false; /* Phase 3 - Step 5 */

    for (cur_line_id = start_line_id; cur_line_id < n_lines; cur_line_id++) { /* Phase 3 - Step 5 */
        double nu_line = opacity->line_list_nu[cur_line_id]; /* Phase 3 - Step 5 */
        double tau_sobolev = opacity->tau_sobolev[ /* Phase 3 - Step 5 */
            cur_line_id * n_shells + shell]; /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: Accumulate line tau */
        tau_trace_line_combined += tau_sobolev; /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: Distance to this line resonance */
        int is_last_line = (cur_line_id == last_line_id); /* Phase 3 - Step 5 */
        double distance_trace = calculate_distance_line( /* Phase 3 - Step 5 */
            comov_nu, pkt->nu, is_last_line, nu_line, /* Phase 3 - Step 5 */
            geo->time_explosion); /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: Continuum tau at this trace distance */
        double tau_trace_continuum = chi_continuum * distance_trace; /* Phase 3 - Step 5 */
        double tau_trace_combined = tau_trace_line_combined + /* Phase 3 - Step 5 */
                                     tau_trace_continuum; /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: Find minimum distance */
        double distance = distance_trace; /* Phase 3 - Step 5 */
        if (distance_boundary < distance) distance = distance_boundary; /* Phase 3 - Step 5 */
        if (distance_continuum < distance) distance = distance_continuum; /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: TARDIS: if distance_trace != 0 */
        if (distance_trace != 0.0) { /* Phase 3 - Step 5 */
            if (distance == distance_boundary) { /* Phase 3 - Step 5 */
                /* Phase 3 - Step 5: Boundary is closest */
                *out_type = INTERACTION_BOUNDARY; /* Phase 3 - Step 5 */
                *out_distance = distance_boundary; /* Phase 3 - Step 5 */
                *out_delta_shell = delta_shell; /* Phase 3 - Step 5 */
                pkt->next_line_id = cur_line_id; /* Phase 3 - Step 5 */
                broke_out = true; /* Phase 3 - Step 5 */
                break; /* Phase 3 - Step 5 */
            } else if (distance == distance_continuum) { /* Phase 3 - Step 5 */
                /* Phase 3 - Step 5: Electron scattering is closest */
                *out_type = INTERACTION_ESCATTERING; /* Phase 3 - Step 5 */
                *out_distance = distance_continuum; /* Phase 3 - Step 5 */
                *out_delta_shell = delta_shell; /* Phase 3 - Step 5 */
                pkt->next_line_id = cur_line_id; /* Phase 3 - Step 5 */
                broke_out = true; /* Phase 3 - Step 5 */
                break; /* Phase 3 - Step 5 */
            }
        }

        /* Phase 3 - Step 5: Still on line path — update j_blue estimator */
        update_line_estimators(est, pkt, cur_line_id, /* Phase 3 - Step 5 */
                                distance_trace, geo->time_explosion); /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: Check if combined tau exceeds tau_event */
        if (tau_trace_combined > tau_event && !disable_line_scattering) { /* Phase 3 - Step 5 */
            /* Phase 3 - Step 5: Line interaction */
            *out_type = INTERACTION_LINE; /* Phase 3 - Step 5 */
            *out_distance = distance_trace; /* Phase 3 - Step 5 */
            *out_delta_shell = delta_shell; /* Phase 3 - Step 5 */
            pkt->next_line_id = cur_line_id; /* Phase 3 - Step 5 */
            broke_out = true; /* Phase 3 - Step 5 */
            break; /* Phase 3 - Step 5 */
        }

        /* Phase 3 - Step 5: Recalculate distance_continuum */
        distance_continuum = (tau_event - tau_trace_line_combined) / /* Phase 3 - Step 5 */
                              chi_continuum; /* Phase 3 - Step 5 */
    }

    /* Phase 3 - Step 5: for...else clause (no break occurred) */
    if (!broke_out) { /* Phase 3 - Step 5 */
        /* Phase 3 - Step 5: In Python for...else, cur_line_id = last value from range */
        /* In C for loop, cur_line_id = n_lines after exhaustion */
        /* TARDIS: if cur_line_id == last_line_id: cur_line_id += 1 */
        /* Since C already incremented past, cur_line_id == n_lines, which is correct */
        pkt->next_line_id = cur_line_id; /* Phase 3 - Step 5 */

        /* Phase 3 - Step 5: TARDIS: distance_continuum < distance_boundary */
        if (distance_continuum < distance_boundary) { /* Phase 3 - Step 5 */
            *out_type = INTERACTION_ESCATTERING; /* Phase 3 - Step 5 */
            *out_distance = distance_continuum; /* Phase 3 - Step 5 */
            *out_delta_shell = delta_shell; /* Phase 3 - Step 5 */
        } else { /* Phase 3 - Step 5 */
            *out_type = INTERACTION_BOUNDARY; /* Phase 3 - Step 5 */
            *out_distance = distance_boundary; /* Phase 3 - Step 5 */
            *out_delta_shell = delta_shell; /* Phase 3 - Step 5 */
        }
    }
}

/* ============================================================ */
/* Phase 3 - Step 6: move_r_packet                              */
/* (r_packet_transport.py: move_r_packet)                       */
/* ============================================================ */

void move_r_packet(RPacket *pkt, double distance, double time_explosion,
                    Estimators *est) {
    /* Phase 3 - Step 6: Doppler factor at OLD position */
    double doppler_factor = get_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 6 */
                                                time_explosion); /* Phase 3 - Step 6 */

    /* Phase 3 - Step 6: Move packet to new position */
    /* TARDIS: estimator update only happens when distance > 0 */
    if (distance > 0.0) { /* Phase 3 - Step 6 */
        double r = pkt->r; /* Phase 3 - Step 6 */
        double new_r = sqrt(r * r + distance * distance + /* Phase 3 - Step 6 */
                            2.0 * r * distance * pkt->mu); /* Phase 3 - Step 6 */
        pkt->mu = (pkt->mu * r + distance) / new_r; /* Phase 3 - Step 6 */
        pkt->r = new_r; /* Phase 3 - Step 6 */

        double comov_nu = pkt->nu * doppler_factor; /* Phase 3 - Step 6 */
        double comov_energy = pkt->energy * doppler_factor; /* Phase 3 - Step 6 */

        /* Phase 3 - Step 6: Update base estimators */
        update_base_estimators(pkt, distance, est, comov_nu, comov_energy); /* Phase 3 - Step 6 */
    }
}

/* ============================================================ */
/* Phase 3 - Step 7: move_packet_across_shell_boundary          */
/* (r_packet_transport.py: move_packet_across_shell_boundary)   */
/* ============================================================ */

void move_packet_across_shell_boundary(RPacket *pkt, int delta_shell,
                                        int n_shells) {
    int next_shell = pkt->current_shell_id + delta_shell; /* Phase 3 - Step 7 */
    if (next_shell >= n_shells) { /* Phase 3 - Step 7: escaped */
        pkt->status = PACKET_EMITTED; /* Phase 3 - Step 7 */
    } else if (next_shell < 0) { /* Phase 3 - Step 7: reabsorbed at inner boundary */
        pkt->status = PACKET_REABSORBED; /* Phase 3 - Step 7 */
    } else { /* Phase 3 - Step 7: move to adjacent shell */
        pkt->current_shell_id = next_shell; /* Phase 3 - Step 7 */
    }
}

/* ============================================================ */
/* Phase 3 - Step 8: Interaction events                         */
/* (interaction_events.py, interaction_event_callers.py)        */
/* ============================================================ */

/* Phase 3 - Step 8: Thomson scattering */
/* TARDIS: thomson_scatter (interaction_events.py) */
void thomson_scatter(RPacket *pkt, double time_explosion, RNG *rng) {
    /* Phase 3 - Step 8: Get comoving frame quantities at OLD angle */
    double old_doppler = get_doppler_factor(pkt->r, pkt->mu, time_explosion); /* Phase 3 - Step 8 */
    double comov_nu = pkt->nu * old_doppler; /* Phase 3 - Step 8 */
    double comov_energy = pkt->energy * old_doppler; /* Phase 3 - Step 8 */

    /* Phase 3 - Step 8: Sample new isotropic direction */
    pkt->mu = rng_mu(rng); /* Phase 3 - Step 8 */

    /* Phase 3 - Step 8: Transform back to lab with NEW angle */
    double inv_new_doppler = get_inverse_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 8 */
                                                         time_explosion); /* Phase 3 - Step 8 */
    pkt->nu = comov_nu * inv_new_doppler; /* Phase 3 - Step 8 */
    pkt->energy = comov_energy * inv_new_doppler; /* Phase 3 - Step 8 */
}

/* Phase 3 - Step 8: Line emission */
/* TARDIS: line_emission (interaction_events.py) */
void line_emission(RPacket *pkt, int emission_line_id,
                    double time_explosion, OpacityState *opacity) {
    double inv_doppler = get_inverse_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 8 */
                                                     time_explosion); /* Phase 3 - Step 8 */
    pkt->nu = opacity->line_list_nu[emission_line_id] * inv_doppler; /* Phase 3 - Step 8 */
    pkt->next_line_id = emission_line_id + 1; /* Phase 3 - Step 8 */
}

/* ============================================================ */
/* Phase 3 - Step 9: Macro-atom interaction                     */
/* (macro_atom.py: macro_atom_interaction)                      */
/* ============================================================ */

void macro_atom_interaction(int activation_level_id, int current_shell_id,
                             OpacityState *opacity, RNG *rng,
                             int *out_transition_id, int *out_transition_type) {
    int current_type = 0; /* Phase 3 - Step 9: start as internal */
    int n_shells = opacity->n_shells; /* Phase 3 - Step 9 */
    int ma_iter = 0; /* Phase 3 - Step 9: safety counter */

    /* Phase 3 - Step 9: Loop while internal transitions (type >= 0) */
    while (current_type >= 0 && ma_iter < 500) { /* Phase 3 - Step 9 */
        ma_iter++; /* Phase 3 - Step 9 */

        /* Phase 3 - Step 9: Bounds check on activation_level_id */
        if (activation_level_id < 0 || activation_level_id >= opacity->n_macro_levels) { /* Phase 3 - Step 9 */
            fprintf(stderr, "[MACRO BUG] activation_level_id=%d out of range [0,%d) at iter %d\n", /* Phase 3 - Step 9 */
                    activation_level_id, opacity->n_macro_levels, ma_iter); /* Phase 3 - Step 9 */
            current_type = MA_BB_EMISSION; /* Phase 3 - Step 9 */
            *out_transition_type = current_type; /* Phase 3 - Step 9 */
            break; /* Phase 3 - Step 9 */
        }

        double probability = 0.0; /* Phase 3 - Step 9 */
        double probability_event = rng_uniform(rng); /* Phase 3 - Step 9 */

        /* Phase 3 - Step 9: Get transition block for this level */
        int block_start = opacity->macro_block_references[activation_level_id]; /* Phase 3 - Step 9 */
        int block_end = opacity->macro_block_references[activation_level_id + 1]; /* Phase 3 - Step 9 */

        bool found = false; /* Phase 3 - Step 9 */
        for (int tid = block_start; tid < block_end; tid++) { /* Phase 3 - Step 9 */
            double tp = opacity->transition_probabilities[ /* Phase 3 - Step 9 */
                tid * n_shells + current_shell_id]; /* Phase 3 - Step 9 */
            probability += tp; /* Phase 3 - Step 9 */

            if (probability > probability_event) { /* Phase 3 - Step 9 */
                activation_level_id = opacity->destination_level_id[tid]; /* Phase 3 - Step 9 */
                current_type = opacity->transition_type[tid]; /* Phase 3 - Step 9 */
                *out_transition_id = tid; /* Phase 3 - Step 9 */
                *out_transition_type = current_type; /* Phase 3 - Step 9 */
                found = true; /* Phase 3 - Step 9 */
                break; /* Phase 3 - Step 9 */
            }
        }

        if (!found) { /* Phase 3 - Step 9 */
            if (block_start >= block_end) { /* Phase 3 - Step 9: empty block */
                /* Phase 3 - Step 9: No transitions available — force BB emission */
                current_type = MA_BB_EMISSION; /* Phase 3 - Step 9 */
                *out_transition_type = current_type; /* Phase 3 - Step 9 */
                break; /* Phase 3 - Step 9 */
            }
            /* Phase 3 - Step 9: Probabilities didn't sum to 1 — pick last */
            int tid = block_end - 1; /* Phase 3 - Step 9 */
            activation_level_id = opacity->destination_level_id[tid]; /* Phase 3 - Step 9 */
            current_type = opacity->transition_type[tid]; /* Phase 3 - Step 9 */
            *out_transition_id = tid; /* Phase 3 - Step 9 */
            *out_transition_type = current_type; /* Phase 3 - Step 9 */
        }
    }

    /* Phase 3 - Step 9: Convert transition_id to line_id for emission */
    *out_transition_id = opacity->transition_line_id[*out_transition_id]; /* Phase 3 - Step 9 */
}

/* Phase 3 - Step 9b: Macro-atom event handler */
/* TARDIS: macro_atom_event (interaction_event_callers.py) */
void macro_atom_event(int dest_level_idx, RPacket *pkt,
                       double time_explosion, OpacityState *opacity,
                       RNG *rng) {
    int transition_id; /* Phase 3 - Step 9b */
    int transition_type; /* Phase 3 - Step 9b */

    macro_atom_interaction(dest_level_idx, pkt->current_shell_id, /* Phase 3 - Step 9b */
                            opacity, rng, &transition_id, &transition_type); /* Phase 3 - Step 9b */

    if (transition_type == MA_BB_EMISSION) { /* Phase 3 - Step 9b */
        line_emission(pkt, transition_id, time_explosion, opacity); /* Phase 3 - Step 9b */
    }
    /* Phase 3 - Step 9b: No continuum processes in this implementation */
    /* BF/FF emission would go here if enabled */
}

/* ============================================================ */
/* Phase 3 - Step 10: Line scatter event                        */
/* (interaction_event_callers.py: line_scatter_event)           */
/* ============================================================ */

void line_scatter_event(RPacket *pkt, double time_explosion,
                         int line_interaction_type, OpacityState *opacity,
                         RNG *rng) {
    /* Phase 3 - Step 10: Get comoving frame at OLD angle */
    double old_doppler = get_doppler_factor(pkt->r, pkt->mu, time_explosion); /* Phase 3 - Step 10 */

    /* Phase 3 - Step 10: Sample new isotropic direction */
    pkt->mu = rng_mu(rng); /* Phase 3 - Step 10 */

    /* Phase 3 - Step 10: Transform energy to lab with NEW angle */
    double inv_new_doppler = get_inverse_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 10 */
                                                         time_explosion); /* Phase 3 - Step 10 */
    double comov_energy = pkt->energy * old_doppler; /* Phase 3 - Step 10 */
    pkt->energy = comov_energy * inv_new_doppler; /* Phase 3 - Step 10 */

    if (line_interaction_type == LINE_SCATTER) { /* Phase 3 - Step 10 */
        /* Phase 3 - Step 10: Pure resonant scatter — emit same line */
        line_emission(pkt, pkt->next_line_id, time_explosion, opacity); /* Phase 3 - Step 10 */
    } else { /* Phase 3 - Step 10: macro-atom or downbranch */
        /* Phase 3 - Step 10: Transform frequency for macro-atom */
        double comov_nu = pkt->nu * old_doppler; /* Phase 3 - Step 10 */
        pkt->nu = comov_nu * inv_new_doppler; /* Phase 3 - Step 10 */

        /* Phase 3 - Step 10: Activate macro-atom at upper level */
        int activation_level = opacity->line2macro_level_upper[pkt->next_line_id]; /* Phase 3 - Step 10 */
        macro_atom_event(activation_level, pkt, time_explosion, opacity, rng); /* Phase 3 - Step 10 */
    }
}

/* ============================================================ */
/* Phase 3 - Step 11: single_packet_loop                        */
/* (single_packet_loop.py: single_packet_loop)                  */
/* Complete packet transport from injection to escape/reabsorb  */
/* ============================================================ */

void single_packet_loop(RPacket *pkt, Geometry *geo, OpacityState *opacity,
                          Estimators *est, MCConfig *config, RNG *rng) {
    /* Phase 3 - Step 11: Set initial packet properties (partial relativity) */
    /* TARDIS: set_packet_props_partial_relativity */
    double inv_doppler = get_inverse_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 11 */
                                                     geo->time_explosion); /* Phase 3 - Step 11 */
    pkt->nu *= inv_doppler; /* Phase 3 - Step 11 */
    pkt->energy *= inv_doppler; /* Phase 3 - Step 11 */

    /* Phase 3 - Step 11: Initialize line ID via binary search */
    /* TARDIS: r_packet.initialize_line_id — find first line with nu < comov_nu */
    double comov_nu_init = pkt->nu * get_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 11 */
                                                         geo->time_explosion); /* Phase 3 - Step 11 */
    /* Phase 3 - Step 11: Binary search in descending line list */
    int lo = 0, hi = opacity->n_lines; /* Phase 3 - Step 11 */
    while (lo < hi) { /* Phase 3 - Step 11 */
        int mid = (lo + hi) / 2; /* Phase 3 - Step 11 */
        if (opacity->line_list_nu[mid] > comov_nu_init) { /* Phase 3 - Step 11 */
            lo = mid + 1; /* Phase 3 - Step 11 */
        } else { /* Phase 3 - Step 11 */
            hi = mid; /* Phase 3 - Step 11 */
        }
    }
    /* Phase 3 - Step 11: TARDIS clamps to last line if past end */
    if (lo == opacity->n_lines) lo = opacity->n_lines - 1; /* Phase 3 - Step 11 */
    pkt->next_line_id = lo; /* Phase 3 - Step 11 */

    /* Phase 3 - Step 11: Main transport loop */
    int loop_count = 0; /* Phase 3 - Step 11: safety */
    while (pkt->status == PACKET_IN_PROCESS && loop_count < 100000) { /* Phase 3 - Step 11 */
        loop_count++; /* Phase 3 - Step 11 */
        /* Phase 3 - Step 11: Calculate electron scattering opacity */
        double doppler_factor = get_doppler_factor(pkt->r, pkt->mu, /* Phase 3 - Step 11 */
                                                    geo->time_explosion); /* Phase 3 - Step 11 */
        double comov_nu = pkt->nu * doppler_factor; /* Phase 3 - Step 11 */
        (void)comov_nu; /* Phase 3 - Step 11: used by TARDIS for BF/FF, not here */

        int shell = pkt->current_shell_id; /* Phase 3 - Step 11 */
        double chi_e = opacity->electron_density[shell] * SIGMA_THOMSON; /* Phase 3 - Step 11 */
        double chi_continuum = chi_e; /* Phase 3 - Step 11: only e-scattering */

        /* Phase 3 - Step 11: Trace packet to find next interaction */
        double distance; /* Phase 3 - Step 11 */
        InteractionType interaction_type; /* Phase 3 - Step 11 */
        int delta_shell; /* Phase 3 - Step 11 */
        trace_packet(pkt, geo, opacity, est, chi_continuum, /* Phase 3 - Step 11 */
                      config->disable_line_scattering, rng, /* Phase 3 - Step 11 */
                      &distance, &interaction_type, &delta_shell); /* Phase 3 - Step 11 */

        /* Phase 3 - Step 11: Handle interaction */
        if (interaction_type == INTERACTION_BOUNDARY) { /* Phase 3 - Step 11 */
            move_r_packet(pkt, distance, geo->time_explosion, est); /* Phase 3 - Step 11 */
            move_packet_across_shell_boundary(pkt, delta_shell, /* Phase 3 - Step 11 */
                                               geo->n_shells); /* Phase 3 - Step 11 */
        } else if (interaction_type == INTERACTION_LINE) { /* Phase 3 - Step 11 */
            move_r_packet(pkt, distance, geo->time_explosion, est); /* Phase 3 - Step 11 */
            line_scatter_event(pkt, geo->time_explosion, /* Phase 3 - Step 11 */
                                config->line_interaction_type, opacity, rng); /* Phase 3 - Step 11 */
        } else if (interaction_type == INTERACTION_ESCATTERING) { /* Phase 3 - Step 11 */
            move_r_packet(pkt, distance, geo->time_explosion, est); /* Phase 3 - Step 11 */
            thomson_scatter(pkt, geo->time_explosion, rng); /* Phase 3 - Step 11 */
        }
    }
}
