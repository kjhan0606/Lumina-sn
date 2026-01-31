/**
 * LUMINA-SN Macro-Atom Implementation
 * macro_atom.c - Non-coherent scattering with internal atomic transitions
 *
 * Reference: Lucy 2002, 2003 - "Monte Carlo transition probabilities"
 *
 * This implementation follows the TARDIS-SN macro-atom formalism for
 * handling complex atomic transitions and fluorescence effects.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "macro_atom.h"
#include "rpacket.h"

/* ============================================================================
 * GLOBAL TUNING PARAMETERS
 * ============================================================================ */

MacroAtomTuning g_macro_atom_tuning = {
    .gaunt_factor_scale = 5.0,        /* Boost Gaunt factor from 0.2 to ~1.0 effective */
    .thermalization_epsilon = 0.35,   /* 35% thermalization probability (TARDIS-like) */
    .ir_thermalization_boost = 0.80,  /* 80% of IR photons thermalize */
    .ir_wavelength_threshold = 7000.0,
    .collisional_boost = 10.0,        /* 10x boost to collisional rates */
    .uv_scatter_boost = 0.5,          /* UV photons less likely to thermalize */
    .uv_wavelength_threshold = 3500.0,
    /* TARDIS compatibility - enabled by default */
    .downbranch_only = 0,             /* Full macro-atom with up/down transitions */
    .use_beta_sobolev = 1,            /* Use Sobolev escape probability */
    .use_level_energies = 1           /* Scale by level energies */
};

void macro_atom_tuning_init(void)
{
    g_macro_atom_tuning.gaunt_factor_scale = 5.0;
    g_macro_atom_tuning.thermalization_epsilon = 0.35;
    g_macro_atom_tuning.ir_thermalization_boost = 0.80;
    g_macro_atom_tuning.ir_wavelength_threshold = 7000.0;
    g_macro_atom_tuning.collisional_boost = 10.0;
    g_macro_atom_tuning.uv_scatter_boost = 0.5;
    g_macro_atom_tuning.uv_wavelength_threshold = 3500.0;
    /* TARDIS compatibility */
    g_macro_atom_tuning.downbranch_only = 0;
    g_macro_atom_tuning.use_beta_sobolev = 1;
    g_macro_atom_tuning.use_level_energies = 1;
}

void macro_atom_tuning_from_env(void)
{
    const char *env;

    env = getenv("MACRO_GAUNT_SCALE");
    if (env) g_macro_atom_tuning.gaunt_factor_scale = atof(env);

    env = getenv("MACRO_EPSILON");
    if (env) g_macro_atom_tuning.thermalization_epsilon = atof(env);

    env = getenv("MACRO_IR_THERM");
    if (env) g_macro_atom_tuning.ir_thermalization_boost = atof(env);

    env = getenv("MACRO_COLLISIONAL_BOOST");
    if (env) g_macro_atom_tuning.collisional_boost = atof(env);

    env = getenv("MACRO_UV_SCATTER");
    if (env) g_macro_atom_tuning.uv_scatter_boost = atof(env);

    /* TARDIS compatibility options */
    env = getenv("MACRO_DOWNBRANCH");
    if (env) g_macro_atom_tuning.downbranch_only = atoi(env);

    env = getenv("MACRO_BETA_SOBOLEV");
    if (env) g_macro_atom_tuning.use_beta_sobolev = atoi(env);

    env = getenv("MACRO_LEVEL_ENERGIES");
    if (env) g_macro_atom_tuning.use_level_energies = atoi(env);
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

const MacroAtomReference *macro_atom_find_reference(
    const AtomicData *atomic,
    int Z, int ion, int level)
{
    if (atomic->macro_atom_references == NULL ||
        atomic->n_macro_atom_references == 0) {
        return NULL;
    }

    /* Linear search for now - could optimize with index table */
    for (int32_t i = 0; i < atomic->n_macro_atom_references; i++) {
        const MacroAtomReference *ref = &atomic->macro_atom_references[i];
        if (ref->atomic_number == Z &&
            ref->ion_number == ion &&
            ref->source_level_number == level) {
            return ref;
        }
    }

    return NULL;
}

int macro_atom_has_data(const AtomicData *atomic, int Z, int ion)
{
    if (atomic->macro_atom_references == NULL ||
        atomic->n_macro_atom_references == 0) {
        return 0;
    }

    /* Check if any reference exists for this ion */
    for (int32_t i = 0; i < atomic->n_macro_atom_references; i++) {
        if (atomic->macro_atom_references[i].atomic_number == Z &&
            atomic->macro_atom_references[i].ion_number == ion) {
            return 1;
        }
    }

    return 0;
}

double macro_atom_get_collision_rate(
    const AtomicData *atomic,
    int Z, int ion,
    int level_u, int level_l,
    double T, double n_e)
{
    /*
     * Collision de-excitation rate from upper to lower level.
     *
     * For electron-impact de-excitation:
     *   C_ul = n_e * (8.63e-6 / sqrt(T)) * Ω(T) / g_u
     *
     * where Ω(T) is the thermally-averaged collision strength.
     *
     * If collision data is not available, use the Van Regemorter approximation:
     *   Ω ≈ 0.276 * f_lu * (E_H / ΔE) * exp(-ΔE/kT) * g_bar(T)
     *
     * where g_bar is the effective Gaunt factor (~0.2-0.3).
     */

    /* Get level energies and statistical weights */
    const Level *upper = atomic_get_level(atomic, Z, ion, level_u);
    const Level *lower = atomic_get_level(atomic, Z, ion, level_l);

    if (upper == NULL || lower == NULL) {
        return 0.0;
    }

    double delta_E = upper->energy - lower->energy;
    if (delta_E <= 0.0) {
        return 0.0;  /* Not a de-excitation */
    }

    double g_u = (double)upper->g;
    double kT = CONST_K_B * T;

    /* Van Regemorter approximation for collision strength
     *
     * TUNING NOTE: The original g_bar = 0.2 was too low, resulting in
     * almost no collisional de-excitation. TARDIS uses more realistic
     * collision strengths. We apply a tunable multiplier.
     */
    double g_bar = 0.2 * g_macro_atom_tuning.gaunt_factor_scale;

    /* Find oscillator strength from line data */
    double f_lu = 0.0;
    for (int64_t i = 0; i < atomic->n_lines; i++) {
        const Line *line = &atomic->lines[i];
        if (line->atomic_number == Z &&
            line->ion_number == ion &&
            line->level_number_lower == level_l &&
            line->level_number_upper == level_u) {
            f_lu = line->f_lu;
            break;
        }
    }

    if (f_lu <= 0.0) {
        /* No line data - use approximate value for allowed transitions */
        f_lu = 0.1;  /* Boosted from 0.01 for better thermalization */
    }

    /* Rydberg energy: 13.6 eV = 2.18e-11 erg */
    double E_H = 2.18e-11;

    /* Collision strength with tunable boost */
    double omega = 0.276 * f_lu * (E_H / delta_E) * exp(-delta_E / kT) * g_bar;

    /* Collision de-excitation rate with additional boost factor */
    double C_ul = n_e * MACRO_ATOM_COLLISION_COEFF / (g_u * sqrt(T)) * omega;
    C_ul *= g_macro_atom_tuning.collisional_boost;

    return C_ul;
}

/* ============================================================================
 * PROBABILITY CALCULATION
 * ============================================================================ */

int macro_atom_calculate_probabilities(
    const AtomicData *atomic,
    int Z, int ion, int level,
    double T, double n_e,
    const double *J_nu,
    MacroAtomProbabilities *probs)
{
    /*
     * Calculate transition probabilities from a given macro-atom level.
     *
     * For each possible transition from this level:
     *   - Radiative (type=-1): Rate = A_ul (spontaneous) + B_ul × J_ν (stimulated)
     *   - Internal down (type=0): Rate = C_ul × n_e
     *   - Internal up (type=1): Rate = B_lu × J_ν (absorption) + C_lu × n_e
     *
     * Probabilities are normalized so they sum to 1.
     *
     * Stimulated Emission/Absorption (now implemented):
     * -------------------------------------------------
     * The Einstein B coefficients relate to A_ul via:
     *   B_ul = (c³ / 8π h ν³) × A_ul
     *   B_lu = (g_u / g_l) × B_ul
     *
     * Stimulated emission rate: B_ul × J_ν
     * Stimulated absorption rate: B_lu × J_ν
     *
     * Reference: TARDIS macro_atom.py, Lucy 2002, 2003
     */

    memset(probs, 0, sizeof(MacroAtomProbabilities));

    /* Find macro-atom reference for this level */
    const MacroAtomReference *ref = macro_atom_find_reference(atomic, Z, ion, level);

    if (ref == NULL || ref->count_total == 0) {
        return -1;  /* No transitions from this level */
    }

    probs->n_transitions = ref->count_total;

    /* Allocate arrays */
    probs->probabilities = (double *)calloc(probs->n_transitions, sizeof(double));
    probs->transition_indices = (int32_t *)calloc(probs->n_transitions, sizeof(int32_t));

    if (probs->probabilities == NULL || probs->transition_indices == NULL) {
        macro_atom_probabilities_free(probs);
        return -1;
    }

    /* Physical constants for B coefficient calculation */
    const double c = 2.99792458e10;      /* Speed of light [cm/s] */
    const double h = 6.62607015e-27;     /* Planck constant [erg·s] */
    const double pi = 3.14159265358979;

    /* Calculate rates for each transition */
    double total_rate = 0.0;
    int32_t idx = 0;

    for (int32_t t = 0; t < ref->count_total; t++) {
        int32_t trans_idx = ref->transition_start_idx + t;

        if (trans_idx >= atomic->n_macro_atom_transitions) {
            break;
        }

        const MacroAtomTransition *trans = &atomic->macro_atom_transitions[trans_idx];

        double rate = 0.0;

        switch (trans->transition_type) {
            case -1:  /* Radiative de-excitation (emission) */
                /*
                 * Rate = A_ul (spontaneous) + B_ul × J_ν (stimulated emission)
                 *
                 * B_ul = (c³ / 8π h ν³) × A_ul
                 *
                 * Stimulated emission enhances the de-excitation rate when
                 * the radiation field J_ν is significant.
                 */
                if (trans->transition_line_id >= 0 &&
                    trans->transition_line_id < atomic->n_lines) {
                    const Line *line = &atomic->lines[trans->transition_line_id];

                    /* Spontaneous emission rate */
                    rate = line->A_ul;

                    /* Add stimulated emission if J_nu is available */
                    if (J_nu != NULL && trans->transition_line_id < atomic->n_lines) {
                        double J_line = J_nu[trans->transition_line_id];

                        if (J_line > 0.0 && line->nu > 0.0) {
                            /* Calculate B_ul from A_ul */
                            double nu = line->nu;
                            double nu3 = nu * nu * nu;
                            double B_ul = (c * c * c) / (8.0 * pi * h * nu3) * line->A_ul;

                            /* Stimulated emission contribution */
                            rate += B_ul * J_line;
                        }
                    }
                }

                if (rate > 0.0) {
                    probs->n_down_radiative++;
                }
                break;

            case 0:   /* Downward internal (collisional de-excitation) */
                rate = macro_atom_get_collision_rate(
                    atomic, Z, ion,
                    trans->source_level_number,
                    trans->destination_level_number,
                    T, n_e);
                probs->n_down_internal++;
                break;

            case 1:   /* Upward internal (collisional excitation + radiative absorption) */
                /*
                 * Rate = C_lu × n_e (collisional) + B_lu × J_ν (radiative absorption)
                 *
                 * B_lu = (g_u / g_l) × B_ul
                 *
                 * Radiative absorption allows photons to pump the atom to higher
                 * excitation states, which is important for non-LTE line formation.
                 */
                {
                    /* Collisional excitation rate from detailed balance */
                    double C_ul = macro_atom_get_collision_rate(
                        atomic, Z, ion,
                        trans->destination_level_number,  /* upper = destination for up */
                        trans->source_level_number,       /* lower = source for up */
                        T, n_e);

                    const Level *upper = atomic_get_level(atomic, Z, ion,
                                                          trans->destination_level_number);
                    const Level *lower = atomic_get_level(atomic, Z, ion,
                                                          trans->source_level_number);

                    if (upper != NULL && lower != NULL) {
                        double delta_E = upper->energy - lower->energy;
                        double kT = CONST_K_B * T;
                        double g_u = (double)upper->g;
                        double g_l = (double)lower->g;
                        double g_ratio = g_u / g_l;

                        /* Collisional excitation */
                        if (C_ul > 0.0) {
                            rate = C_ul * g_ratio * exp(-delta_E / kT);
                        }

                        /* Add radiative absorption if J_nu is available */
                        if (J_nu != NULL && trans->transition_line_id >= 0 &&
                            trans->transition_line_id < atomic->n_lines) {
                            const Line *line = &atomic->lines[trans->transition_line_id];
                            double J_line = J_nu[trans->transition_line_id];

                            if (J_line > 0.0 && line->nu > 0.0 && line->A_ul > 0.0) {
                                /* Calculate B_lu from A_ul */
                                double nu = line->nu;
                                double nu3 = nu * nu * nu;
                                double B_ul = (c * c * c) / (8.0 * pi * h * nu3) * line->A_ul;
                                double B_lu = g_ratio * B_ul;

                                /* Radiative absorption contribution */
                                rate += B_lu * J_line;
                            }
                        }
                    }
                }
                probs->n_up_internal++;
                break;
        }

        probs->probabilities[idx] = rate;
        probs->transition_indices[idx] = trans_idx;
        total_rate += rate;
        idx++;
    }

    probs->n_transitions = idx;
    probs->total_rate = total_rate;

    /* Normalize probabilities */
    if (total_rate > 0.0) {
        for (int32_t i = 0; i < probs->n_transitions; i++) {
            probs->probabilities[i] /= total_rate;
        }
    }

    /* Calculate total emission probability */
    probs->p_emission = 0.0;
    for (int32_t i = 0; i < probs->n_transitions; i++) {
        int32_t trans_idx = probs->transition_indices[i];
        if (trans_idx < atomic->n_macro_atom_transitions &&
            atomic->macro_atom_transitions[trans_idx].transition_type == -1) {
            probs->p_emission += probs->probabilities[i];
        }
    }

    return 0;
}

void macro_atom_probabilities_free(MacroAtomProbabilities *probs)
{
    if (probs->probabilities != NULL) {
        free(probs->probabilities);
        probs->probabilities = NULL;
    }
    if (probs->transition_indices != NULL) {
        free(probs->transition_indices);
        probs->transition_indices = NULL;
    }
    probs->n_transitions = 0;
}

/* ============================================================================
 * MACRO-ATOM STATE INITIALIZATION
 * ============================================================================ */

void macro_atom_init(
    MacroAtomState *state,
    const AtomicData *atomic,
    int64_t line_id,
    double T,
    double n_e,
    RNGState *rng)
{
    memset(state, 0, sizeof(MacroAtomState));

    if (line_id < 0 || line_id >= atomic->n_lines) {
        state->is_activated = 0;
        return;
    }

    const Line *line = &atomic->lines[line_id];

    /* Start in the upper level of the absorbed line */
    state->atomic_number = line->atomic_number;
    state->ion_number = line->ion_number;
    state->level_number = line->level_number_upper;

    /* Get level energy */
    const Level *level = atomic_get_level(atomic,
                                           line->atomic_number,
                                           line->ion_number,
                                           line->level_number_upper);
    if (level != NULL) {
        state->energy = level->energy;
    }

    state->temperature = T;
    state->electron_density = n_e;
    state->rng_state = rng;

    state->n_jumps = 0;
    state->max_jumps = MACRO_ATOM_MAX_JUMPS;

    state->is_activated = 0;
    state->emission_line_id = -1;
    state->emission_nu = 0.0;
}

/* ============================================================================
 * MACRO-ATOM MONTE CARLO LOOP
 * ============================================================================ */

int macro_atom_do_transition_loop(
    MacroAtomState *state,
    const AtomicData *atomic)
{
    /*
     * The core macro-atom Monte Carlo loop.
     *
     * Algorithm:
     * 1. Compute transition probabilities from current level
     * 2. Sample random number xi
     * 3. Select transition based on cumulative probabilities
     * 4. If radiative (type=-1): EXIT - record emission line
     * 5. Otherwise: move to new level, repeat
     *
     * DOWNBRANCH MODE (TARDIS intermediate):
     * If downbranch_only is enabled, skip the cascade entirely and
     * immediately select an emission line from the current level.
     * This is equivalent to only allowing type=-1 transitions.
     */

    /* DOWNBRANCH MODE: Skip cascade, directly emit from current level */
    if (g_macro_atom_tuning.downbranch_only) {
        return macro_atom_simplified_transition(state, atomic);
    }

    while (state->n_jumps < state->max_jumps) {

        /* Calculate probabilities from current level */
        MacroAtomProbabilities probs;
        int status = macro_atom_calculate_probabilities(
            atomic,
            state->atomic_number,
            state->ion_number,
            state->level_number,
            state->temperature,
            state->electron_density,
            NULL,  /* J_nu - not using radiation field for now */
            &probs
        );

        if (status != 0 || probs.n_transitions == 0) {
            /* No transitions available - use simplified treatment */
            macro_atom_probabilities_free(&probs);
            return macro_atom_simplified_transition(state, atomic);
        }

        /* Sample random number */
        double xi = rng_uniform(state->rng_state);

        /* Find selected transition by cumulative probability */
        double cumulative = 0.0;
        int32_t selected_trans = -1;

        for (int32_t i = 0; i < probs.n_transitions; i++) {
            cumulative += probs.probabilities[i];
            if (xi < cumulative) {
                selected_trans = probs.transition_indices[i];
                break;
            }
        }

        /* Fallback to last transition if numerical issues */
        if (selected_trans < 0 && probs.n_transitions > 0) {
            selected_trans = probs.transition_indices[probs.n_transitions - 1];
        }

        macro_atom_probabilities_free(&probs);

        if (selected_trans < 0) {
            /* No valid transition - thermalize */
            state->is_activated = 0;
            return 0;
        }

        /* Get the selected transition */
        const MacroAtomTransition *trans = &atomic->macro_atom_transitions[selected_trans];

        /* Process based on transition type */
        if (trans->transition_type == -1) {
            /*
             * RADIATIVE DE-ACTIVATION
             *
             * Note: Thermalization is applied ONLY ONCE at the final emission,
             * not at each step of the cascade. This is handled in the
             * calling function (macro_atom_process_line_interaction) or
             * in macro_atom_simplified_transition for fallback cases.
             *
             * The macro-atom loop itself just handles the cascade physics -
             * selecting which transition to take based on calculated rates.
             */
            state->emission_line_id = trans->transition_line_id;

            if (trans->transition_line_id >= 0 &&
                trans->transition_line_id < atomic->n_lines) {
                state->emission_nu = atomic->lines[trans->transition_line_id].nu;
            }

            /* Packet emits at the selected line frequency */
            state->is_activated = 1;
            return 1;
        }

        /* Non-radiative transition: move to destination level */
        state->level_number = trans->destination_level_number;

        /* Update energy */
        const Level *new_level = atomic_get_level(atomic,
                                                   state->atomic_number,
                                                   state->ion_number,
                                                   state->level_number);
        if (new_level != NULL) {
            state->energy = new_level->energy;
        }

        state->n_jumps++;

        /* Check for ground state (level 0) - automatic de-activation */
        if (state->level_number == 0) {
            /* Reached ground state via internal jumps */
            /* Find strongest line from ground state for emission */
            return macro_atom_simplified_transition(state, atomic);
        }
    }

    /* Max jumps exceeded - thermalize (absorb) */
    state->is_activated = 0;
    return 0;
}

int macro_atom_simplified_transition(
    MacroAtomState *state,
    const AtomicData *atomic)
{
    /*
     * Simplified treatment when full macro-atom data is not available.
     *
     * Strategy:
     * 1. Apply wavelength-dependent thermalization probability (TARDIS-style epsilon)
     * 2. Find all lines from this level (as upper level)
     * 3. Select emission line with probability proportional to A_ul
     * 4. If no lines available: thermalize
     *
     * TARDIS thermalization behavior:
     * - epsilon parameter controls probability of LTE thermalization
     * - IR photons (λ > 7000 Å) have higher thermalization due to stronger
     *   coupling to thermal bath
     * - UV photons (λ < 3500 Å) scatter more (lower thermalization) to enable
     *   fluorescence redistribution to optical
     */

    /* Find all lines with this level as upper */
    double total_A = 0.0;
    int64_t n_emission_lines = 0;
    int64_t emission_candidates[100];  /* Max 100 candidates */
    double emission_rates[100];

    for (int64_t i = 0; i < atomic->n_lines && n_emission_lines < 100; i++) {
        const Line *line = &atomic->lines[i];

        if (line->atomic_number == state->atomic_number &&
            line->ion_number == state->ion_number &&
            line->level_number_upper == state->level_number) {

            emission_candidates[n_emission_lines] = i;
            emission_rates[n_emission_lines] = line->A_ul;
            total_A += line->A_ul;
            n_emission_lines++;
        }
    }

    if (n_emission_lines == 0 || total_A <= 0.0) {
        /* No emission lines available - thermalize */
        state->is_activated = 0;
        return 0;
    }

    /* Select emission line proportional to A_ul */
    double xi = rng_uniform(state->rng_state) * total_A;
    double cumulative = 0.0;
    int64_t selected_line = emission_candidates[0];

    for (int64_t i = 0; i < n_emission_lines; i++) {
        cumulative += emission_rates[i];
        if (xi < cumulative) {
            selected_line = emission_candidates[i];
            break;
        }
    }

    /*
     * Note: Thermalization is now handled ONCE in macro_atom_process_line_interaction()
     * after the entire cascade completes. This avoids double-counting when this
     * simplified fallback is used within the main transition loop.
     */

    /* Set emission properties - packet re-emitted */
    state->is_activated = 1;
    state->emission_line_id = selected_line;
    state->emission_nu = atomic->lines[selected_line].nu;

    return 1;  /* Packet survives */
}

/* ============================================================================
 * INTEGRATION WITH PACKET TRANSPORT
 * ============================================================================ */

int macro_atom_process_line_interaction(
    RPacket *pkt,
    const AtomicData *atomic,
    int64_t line_id,
    double T,
    double n_e,
    double time_explosion)
{
    /*
     * Process a line interaction using the macro-atom formalism.
     *
     * This replaces simple line_scatter when LINE_MACROATOM mode is active.
     *
     * Steps:
     * 1. Initialize macro-atom state from absorbed line
     * 2. Run the transition loop
     * 3. If packet survives: set new direction and frequency
     * 4. Return packet status
     */

    /* Initialize macro-atom state */
    MacroAtomState ma_state;
    macro_atom_init(&ma_state, atomic, line_id, T, n_e, &pkt->rng_state);

    /* Run transition loop */
    int survives = macro_atom_do_transition_loop(&ma_state, atomic);

    if (!survives) {
        /* Packet absorbed (thermalized) during cascade */
        pkt->status = PACKET_REABSORBED;
        return 0;
    }

    /*
     * TARDIS-style epsilon thermalization:
     * Apply thermalization probability ONCE at the end of the macro-atom cascade.
     * This represents collisional de-excitation competing with radiative emission.
     *
     * The probability is wavelength-dependent:
     * - IR (λ > 7000 Å): High thermalization (strong thermal coupling)
     * - Optical: Base epsilon probability
     * - UV (λ < 3500 Å): Low thermalization (allow fluorescence to redistribute energy)
     */
    double nu_emission = ma_state.emission_nu;
    if (nu_emission > 0.0) {
        double wavelength_A = (2.99792458e10 / nu_emission) * 1e8;  /* c/ν in Å */

        double p_thermalize = g_macro_atom_tuning.thermalization_epsilon;

        if (wavelength_A > g_macro_atom_tuning.ir_wavelength_threshold) {
            /* IR photons: enhanced thermalization */
            p_thermalize = g_macro_atom_tuning.ir_thermalization_boost;
        } else if (wavelength_A < g_macro_atom_tuning.uv_wavelength_threshold) {
            /* UV photons: reduced thermalization for fluorescence */
            p_thermalize *= g_macro_atom_tuning.uv_scatter_boost;
        }

        /* Apply thermalization probability */
        double xi_therm = rng_uniform(&pkt->rng_state);
        if (xi_therm < p_thermalize) {
            /* Thermalize: packet energy absorbed into thermal pool */
            pkt->status = PACKET_REABSORBED;
            return 0;
        }
    }

    /* Packet survives - update direction and frequency */

    /* New direction: isotropic in CMF */
    double mu_cmf_new = 2.0 * rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform direction from CMF to lab frame */
    double mu_lab_new = angle_aberration_CMF_to_LF(pkt->r, mu_cmf_new, time_explosion);
    pkt->mu = mu_lab_new;

    /*
     * New frequency: emission line frequency in CMF, transformed to lab
     *
     * This is the KEY difference from simple scattering:
     * The packet can emerge at a DIFFERENT frequency than it was absorbed!
     * This enables fluorescence (UV → optical).
     */
    double nu_line_emission = ma_state.emission_nu;
    double inv_doppler = get_inverse_doppler_factor(pkt->r, mu_lab_new, time_explosion);
    pkt->nu = nu_line_emission * inv_doppler;

    /* Record interaction */
    pkt->last_line_interaction_in_id = line_id;
    pkt->last_line_interaction_out_id = ma_state.emission_line_id;
    pkt->last_interaction_type = 2;  /* Line scatter */

    return 1;  /* Packet survives */
}
