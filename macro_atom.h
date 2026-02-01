/**
 * LUMINA-SN Macro-Atom Formalism
 * macro_atom.h - Non-coherent scattering with internal atomic transitions
 *
 * Reference: Lucy 2002, 2003 - "Monte Carlo transition probabilities"
 *
 * Physical Description:
 * ---------------------
 * In the Macro-Atom formalism, when a photon is absorbed by an atom, it
 * doesn't immediately scatter. Instead, the atom enters an excited state
 * and undergoes a series of internal transitions:
 *
 *   1. INTERNAL JUMPS: The excitation energy can move between levels via:
 *      - Radiative de-excitation (emission of photon)
 *      - Collisional processes (energy exchange with electrons)
 *      - Spontaneous decay to lower levels
 *
 *   2. DE-ACTIVATION: The atom eventually de-activates by:
 *      - Emitting a photon (radiative) - produces an r-packet
 *      - Collisional de-excitation (energy goes to thermal pool)
 *      - Ionization (if energy is sufficient)
 *
 * This allows for FLUORESCENCE: UV absorption can lead to optical emission
 * as the atom cascades through intermediate levels. This is crucial for
 * accurately modeling Type Ia and Type II SN spectra.
 *
 * Transition Types:
 * -----------------
 *   transition_type = -1: Radiative de-excitation (emission) - ACTIVATES OUTPUT
 *   transition_type =  0: Downward internal (non-radiative)
 *   transition_type = +1: Upward internal (collisional excitation)
 */

#ifndef MACRO_ATOM_H
#define MACRO_ATOM_H

#include <stdint.h>
#include "atomic_data.h"
#include "physics_kernels.h"
#include "rpacket.h"

/* ============================================================================
 * MACRO-ATOM STATE
 * ============================================================================
 * Tracks the current state of an atom during the macro-atom Monte Carlo loop.
 */

typedef struct {
    /* Current level information */
    int8_t  atomic_number;
    int8_t  ion_number;
    int16_t level_number;

    /* Energy tracking */
    double  energy;               /* Current excitation energy [erg] */

    /* Temperature context (for collisional rates) */
    double  temperature;          /* Local electron temperature [K] */
    double  electron_density;     /* Local n_e [cm^-3] */
    double  dilution_factor;      /* W: dilution factor for J_ν (0-1) */

    /* Sobolev optical depth array (needed for β factor) */
    const double *tau_sobolev;    /* Array indexed by line_id */

    /* RNG state (inherited from packet for thread safety) */
    RNGState *rng_state;

    /* Statistics */
    int32_t n_jumps;              /* Number of internal jumps performed */
    int32_t max_jumps;            /* Maximum allowed jumps (prevent infinite loops) */

    /* Outcome */
    int     is_activated;         /* True if de-activation occurred */
    int64_t emission_line_id;     /* Line ID of emitted photon (if radiative) */
    double  emission_nu;          /* Frequency of emitted photon [Hz] */

} MacroAtomState;

/* ============================================================================
 * TRANSITION PROBABILITY CALCULATION
 * ============================================================================ */

/**
 * MacroAtomProbabilities: Pre-computed probabilities for all transitions
 * from a given level.
 */
typedef struct {
    int32_t n_transitions;        /* Total number of transitions from this level */
    int32_t n_down_radiative;     /* Count of downward radiative (emission) */
    int32_t n_down_internal;      /* Count of downward internal */
    int32_t n_up_internal;        /* Count of upward internal */

    double  *probabilities;       /* Normalized probabilities [n_transitions] */
    int32_t *transition_indices;  /* Indices into macro_atom_transitions array */

    double  total_rate;           /* Sum of all transition rates [s^-1] */
    double  p_emission;           /* Total probability of radiative emission */

} MacroAtomProbabilities;

/**
 * Calculate transition probabilities from a given level.
 *
 * TARDIS-style calculation:
 *   Radiative down (type=-1): Rate = A_ul × β + B_ul × J_ν × β
 *   Collisional down (type=0): Rate = C_ul
 *   Internal up (type=1): Rate = B_lu × J_ν × β × stim + C_lu
 *
 * where β = (1 - exp(-τ)) / τ is the Sobolev escape probability.
 *
 * @param atomic      Atomic data with macro-atom transitions
 * @param Z           Atomic number
 * @param ion         Ion stage (0 = neutral)
 * @param level       Level number
 * @param T           Temperature [K]
 * @param n_e         Electron density [cm^-3]
 * @param W           Dilution factor (0-1)
 * @param J_nu        Mean intensity at level transition frequencies [optional, can be NULL]
 * @param tau_sobolev Sobolev optical depths for each line [optional, can be NULL]
 * @param probs       Output: pre-allocated MacroAtomProbabilities structure
 * @return            0 on success, -1 if level has no transitions
 */
int macro_atom_calculate_probabilities(
    const AtomicData *atomic,
    int Z, int ion, int level,
    double T, double n_e, double W,
    const double *J_nu,
    const double *tau_sobolev,
    MacroAtomProbabilities *probs
);

/**
 * Free probability arrays allocated by macro_atom_calculate_probabilities
 */
void macro_atom_probabilities_free(MacroAtomProbabilities *probs);

/* ============================================================================
 * MACRO-ATOM MONTE CARLO LOOP
 * ============================================================================ */

/**
 * Initialize macro-atom state from absorbed packet.
 *
 * @param state       Macro-atom state to initialize
 * @param atomic      Atomic data
 * @param line_id     Line that absorbed the packet
 * @param T           Local temperature [K]
 * @param n_e         Local electron density [cm^-3]
 * @param W           Dilution factor (0-1)
 * @param tau_sobolev Array of Sobolev optical depths (indexed by line_id)
 * @param rng         Pointer to RNG state (from packet)
 */
void macro_atom_init(
    MacroAtomState *state,
    const AtomicData *atomic,
    int64_t line_id,
    double T,
    double n_e,
    double W,
    const double *tau_sobolev,
    RNGState *rng
);

/**
 * Perform the macro-atom Monte Carlo loop.
 *
 * Starting from the initially activated level, this function:
 *   1. Computes transition probabilities
 *   2. Samples a random transition
 *   3. If radiative emission (type=-1): de-activates, records emission line
 *   4. Otherwise: moves to new level, repeats
 *
 * The loop continues until:
 *   - Radiative de-activation (packet re-emitted at emission line frequency)
 *   - Collisional de-activation (energy thermalized, packet absorbed)
 *   - Maximum jumps exceeded (safety limit)
 *
 * @param state      Macro-atom state (modified in place)
 * @param atomic     Atomic data
 * @return           1 if packet is re-emitted, 0 if absorbed/thermalized
 */
int macro_atom_do_transition_loop(
    MacroAtomState *state,
    const AtomicData *atomic
);

/**
 * Simplified macro-atom transition for lines without full macro-atom data.
 *
 * When macro-atom transition data is not available for a line, this function
 * performs a simplified treatment:
 *   - Pure resonant scattering (coherent in CMF) with some probability
 *   - Or fluorescence to a lower level transition
 *
 * @param state      Macro-atom state
 * @param atomic     Atomic data
 * @return           1 if packet is re-emitted, 0 if absorbed
 */
int macro_atom_simplified_transition(
    MacroAtomState *state,
    const AtomicData *atomic
);

/* ============================================================================
 * INTEGRATION WITH PACKET TRANSPORT
 * ============================================================================ */

/**
 * Process line interaction using macro-atom formalism.
 *
 * This replaces simple line_scatter when LINE_MACROATOM mode is active.
 *
 * @param pkt            Packet that hit a line
 * @param atomic         Atomic data
 * @param line_id        Index of the interacting line
 * @param T              Local temperature [K]
 * @param n_e            Local electron density [cm^-3]
 * @param W              Dilution factor (0-1)
 * @param tau_sobolev    Array of Sobolev optical depths (indexed by line_id)
 * @param time_explosion Time since explosion [s]
 * @return               1 if packet survives (re-emitted), 0 if absorbed
 */
int macro_atom_process_line_interaction(
    RPacket *pkt,
    const AtomicData *atomic,
    int64_t line_id,
    double T,
    double n_e,
    double W,
    const double *tau_sobolev,
    double time_explosion
);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Find macro-atom reference for a given level.
 *
 * @param atomic    Atomic data
 * @param Z         Atomic number
 * @param ion       Ion stage
 * @param level     Level number
 * @return          Pointer to MacroAtomReference, or NULL if not found
 */
const MacroAtomReference *macro_atom_find_reference(
    const AtomicData *atomic,
    int Z, int ion, int level
);

/**
 * Get collision rate for transition between two levels.
 *
 * Uses the formula:
 *   C_ul = n_e * <σv>_ul
 *
 * where <σv> is the Maxwellian-averaged collision cross-section.
 *
 * @param atomic    Atomic data
 * @param Z         Atomic number
 * @param ion       Ion stage
 * @param level_u   Upper level
 * @param level_l   Lower level
 * @param T         Temperature [K]
 * @param n_e       Electron density [cm^-3]
 * @return          Collision rate [s^-1], or 0 if no data available
 */
double macro_atom_get_collision_rate(
    const AtomicData *atomic,
    int Z, int ion,
    int level_u, int level_l,
    double T, double n_e
);

/**
 * Check if macro-atom data is available for a given ion.
 */
int macro_atom_has_data(const AtomicData *atomic, int Z, int ion);

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/* Maximum number of jumps in macro-atom loop (safety limit) */
#define MACRO_ATOM_MAX_JUMPS 100

/* Minimum probability for considering a transition */
#define MACRO_ATOM_MIN_PROB 1e-30

/* Collisional de-excitation coefficient approximation */
/* C_ul ≈ 8.63e-6 / (g_u * sqrt(T)) * Ω(T) * n_e */
#define MACRO_ATOM_COLLISION_COEFF 8.63e-6

/* ============================================================================
 * TUNABLE PARAMETERS (for matching TARDIS behavior)
 * ============================================================================
 * These parameters control the macro-atom physics to better match TARDIS:
 *
 *   GAUNT_FACTOR_SCALE: Multiplier for effective Gaunt factor in collision rates
 *                       TARDIS uses ~1.0, original was 0.2 (too low)
 *
 *   THERMALIZATION_EPSILON: Base probability of thermalization vs re-emission
 *                           TARDIS epsilon ranges 0.3-0.5 depending on conditions
 *
 *   IR_THERMALIZATION_BOOST: Extra thermalization for IR photons (λ > 7000 Å)
 *                            IR photons couple more strongly to thermal bath
 *
 *   COLLISIONAL_BOOST: Overall boost to collisional rates
 *                      Compensates for missing collision data
 */

typedef struct {
    double gaunt_factor_scale;       /* Effective Gaunt factor multiplier (default: 5.0) */
    double thermalization_epsilon;   /* Base thermalization probability (default: 0.35) */
    double ir_thermalization_boost;  /* Extra thermalization for IR (default: 0.80) */
    double ir_wavelength_threshold;  /* IR threshold in Angstrom (default: 7000) */
    double collisional_boost;        /* Collisional rate multiplier (default: 10.0) */
    double uv_scatter_boost;         /* Reduce thermalization for UV (default: 0.5) */
    double uv_wavelength_threshold;  /* UV threshold in Angstrom (default: 3500) */

    /* TARDIS compatibility options */
    int    downbranch_only;          /* Only allow downward transitions (default: 0) */
    int    use_beta_sobolev;         /* Multiply rates by β_Sobolev factor (default: 1) */
    int    use_level_energies;       /* Scale by level energies like TARDIS (default: 1) */
} MacroAtomTuning;

/* Global tuning parameters */
extern MacroAtomTuning g_macro_atom_tuning;

/* Initialize tuning to defaults */
void macro_atom_tuning_init(void);

/* Set tuning from environment variables */
void macro_atom_tuning_from_env(void);

#endif /* MACRO_ATOM_H */
