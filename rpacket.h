/**
 * LUMINA-SN: C Implementation of TARDIS Monte Carlo Transport
 * rpacket.h - RPacket structure and core transport function declarations
 *
 * Transpiled from: tardis/montecarlo/montecarlo_numba/r_packet.py
 *                  tardis/montecarlo/montecarlo_numba/single_packet_loop.py
 *
 * HPC-Hardened Version:
 *   - Thread-safe RNG (per-packet seed)
 *   - No global configuration state
 *   - Optimized hot path for 270k+ lines
 */

#ifndef RPACKET_H
#define RPACKET_H

#include <stdint.h>

/* ============================================================================
 * PHYSICAL CONSTANTS
 * ============================================================================ */
#define C_SPEED_OF_LIGHT 2.99792458e10  /* cm/s */
#define CLOSE_LINE_THRESHOLD 1e-7       /* Relative frequency threshold */

/* ============================================================================
 * ENUMERATIONS - State Machine
 * ============================================================================ */

/**
 * PacketStatus: Tracks the Monte Carlo packet's lifecycle
 *
 * IN_PROCESS (0): Packet is actively propagating through ejecta
 * EMITTED (1):    Packet escaped through outer boundary -> contributes to spectrum
 * REABSORBED (2): Packet hit inner boundary (photosphere) -> thermalized
 */
typedef enum {
    PACKET_IN_PROCESS = 0,
    PACKET_EMITTED    = 1,
    PACKET_REABSORBED = 2
} PacketStatus;

/**
 * InteractionType: What stopped the packet's free flight
 *
 * BOUNDARY (1):    Hit shell boundary (inner or outer)
 * LINE (2):        Resonant line interaction (Sobolev)
 * ESCATTERING (3): Thomson electron scattering
 */
typedef enum {
    INTERACTION_BOUNDARY    = 1,
    INTERACTION_LINE        = 2,
    INTERACTION_ESCATTERING = 3
} InteractionType;

/**
 * LineInteractionType: How line scattering is treated
 *
 * SCATTER:     Pure resonant scattering (coherent in CMF)
 * DOWNBRANCH:  Fluorescence with downward branching
 * MACROATOM:   Full macro-atom treatment
 */
typedef enum {
    LINE_SCATTER    = 0,
    LINE_DOWNBRANCH = 1,
    LINE_MACROATOM  = 2
} LineInteractionType;

/* ============================================================================
 * THREAD-SAFE RNG: Xorshift64* Generator
 * ============================================================================
 *
 * Why not rand_r()?
 *   - rand_r() has poor statistical quality (short period, correlations)
 *   - Xorshift64* has 2^64-1 period, passes BigCrush test suite
 *   - Same speed as rand_r(), better quality
 *
 * Usage:
 *   uint64_t seed = initial_seed;
 *   double xi = rng_uniform(&seed);  // seed modified in place
 */
typedef uint64_t RNGState;

/**
 * rng_xorshift64star: Core generator (internal use)
 * Period: 2^64 - 1
 * Quality: Passes BigCrush statistical test suite
 */
static inline uint64_t rng_xorshift64star(RNGState *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/**
 * rng_uniform: Generate uniform random number in [0, 1)
 *
 * Thread-safe: each thread uses its own RNGState
 *
 * @param state  Pointer to RNG state (modified in place)
 * @return       Uniform deviate in [0, 1)
 */
static inline double rng_uniform(RNGState *state) {
    /* Convert 64-bit integer to double in [0, 1) */
    /* Use top 53 bits for full double precision mantissa */
    return (rng_xorshift64star(state) >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * rng_init: Initialize RNG state from seed
 *
 * @param state  Pointer to RNG state to initialize
 * @param seed   Initial seed value (should be non-zero)
 */
static inline void rng_init(RNGState *state, uint64_t seed) {
    /* Ensure non-zero state (Xorshift fails with zero) */
    *state = seed ? seed : 0x853c49e6748fea9bULL;
    /* Warm up the generator */
    rng_xorshift64star(state);
    rng_xorshift64star(state);
}

/* ============================================================================
 * CORE DATA STRUCTURE: RPacket
 * ============================================================================
 *
 * Physical Description:
 * ---------------------
 * An RPacket represents an indivisible "energy packet" carrying a fraction
 * of the supernova's luminosity. In Monte Carlo radiative transfer:
 *
 *   L_total = Sum (packet_energy / simulation_time)
 *
 * Each packet has a position (r), direction (mu), frequency (nu), and energy
 * weight. The simulation traces packets through the expanding ejecta until
 * they either escape (spectrum) or are absorbed (thermalization).
 *
 * Coordinate System:
 * ------------------
 * - Spherical symmetry assumed (1D)
 * - r: distance from center [cm]
 * - mu = cos(theta): angle between velocity and radial direction
 *   - mu = +1: moving outward (radially away from center)
 *   - mu = -1: moving inward (toward center)
 *   - mu = 0:  moving tangentially
 *
 * Frame Transformations:
 * ----------------------
 * Due to homologous expansion (v = r/t), we must transform between:
 * - Lab frame: where r, mu are defined
 * - Comoving frame (CMF): where opacities are computed
 *
 * Doppler factor: D = 1 - (r*mu)/(c*t) = 1 - beta*mu
 * nu_cmf = nu_lab * D
 */
typedef struct {
    /* === Primary Phase-Space Coordinates === */

    /**
     * r: Radial position [cm]
     * Physical: Distance from supernova center
     * Range: [r_inner_boundary, r_outer_boundary]
     * Updated by: move_r_packet() after each propagation step
     */
    double r;

    /**
     * mu: Direction cosine [-1, +1]
     * Physical: cos(theta) where theta is angle from radial direction
     * mu = +1: outward radial propagation
     * mu = 0:  tangential propagation
     * mu = -1: inward radial propagation
     * Updated by: move_r_packet() (geometry) and scattering events
     */
    double mu;

    /**
     * nu: Photon frequency [Hz]
     * Physical: Determines which spectral lines can interact
     * Frame: Lab frame (transformed to CMF for opacity calculation)
     * Updated by: Line scattering (frequency redistribution)
     */
    double nu;

    /**
     * energy: Monte Carlo weight [erg]
     * Physical: Fraction of total luminosity this packet represents
     * Note: Can change during interactions (energy conservation)
     * L_packet = energy / t_simulation
     */
    double energy;

    /* === Discrete State Variables === */

    /**
     * next_line_id: Index into line_list_nu array
     * Physical: Next spectral line (sorted by frequency) to check
     * Optimization: Avoids re-searching entire line list each step
     * Range: [0, n_lines - 1]
     */
    int64_t next_line_id;

    /**
     * current_shell_id: Radial zone index
     * Physical: Which shell the packet currently occupies
     * Each shell has uniform density, temperature, composition
     * Range: [0, n_shells - 1]
     */
    int64_t current_shell_id;

    /**
     * status: Packet lifecycle state
     * Values: PACKET_IN_PROCESS, PACKET_EMITTED, PACKET_REABSORBED
     * Termination: Loop exits when status != IN_PROCESS
     */
    int64_t status;

    /**
     * rng_state: Thread-safe random number generator state
     * Purpose: HPC-safe RNG for OpenMP parallelization
     * Each packet has independent RNG stream
     */
    RNGState rng_state;

    /**
     * index: Packet ID in the ensemble
     * Purpose: Tracking and debugging
     */
    int64_t index;

    /* === Interaction History (for spectrum synthesis) === */

    /**
     * last_interaction_type: Type of most recent interaction
     * Values: 1 = electron scatter, 2 = line scatter
     * Used for: Identifying line-forming regions in spectrum
     */
    int64_t last_interaction_type;

    /**
     * last_interaction_in_nu: Frequency before last line scatter [Hz]
     * Used for: Tracking frequency evolution through fluorescence
     */
    double last_interaction_in_nu;

    /**
     * last_line_interaction_in_id: Line that absorbed the packet
     * last_line_interaction_out_id: Line that re-emitted the packet
     * Used for: Macro-atom and branching ratio calculations
     */
    int64_t last_line_interaction_in_id;
    int64_t last_line_interaction_out_id;

} RPacket;

/* ============================================================================
 * MODEL STRUCTURE: NumbaModel (Simulation geometry)
 * ============================================================================ */
typedef struct {
    double *r_inner;        /* Inner radius of each shell [cm], size: n_shells */
    double *r_outer;        /* Outer radius of each shell [cm], size: n_shells */
    double time_explosion;  /* Time since explosion [s] - sets velocity field */
    int64_t n_shells;       /* Number of radial zones */
} NumbaModel;

/* ============================================================================
 * PLASMA STRUCTURE: NumbaPlasma (Physical conditions per shell)
 * ============================================================================ */
typedef struct {
    double *line_list_nu;   /* Sorted line frequencies [Hz], size: n_lines */
    double *tau_sobolev;    /* Sobolev optical depths [n_lines x n_shells] */
    double *electron_density; /* Free electron density per shell [cm^-3] */
    int64_t n_lines;        /* Total number of spectral lines */
    int64_t n_shells;       /* Number of radial zones */
} NumbaPlasma;

/* ============================================================================
 * ESTIMATOR STRUCTURE: For radiative equilibrium iteration
 * ============================================================================ */
typedef struct {
    double *j_estimator;    /* Mean intensity estimator per shell */
    double *nu_bar_estimator; /* Frequency-weighted intensity */
    double *j_blue_estimator; /* Line-specific estimators [n_lines x n_shells] */
    int64_t n_shells;
    int64_t n_lines;
} Estimators;

/* ============================================================================
 * CONFIGURATION STRUCTURE (No longer global - passed as parameter)
 * ============================================================================ */
typedef struct {
    int enable_full_relativity;     /* Include O(v^2/c^2) terms */
    int disable_line_scattering;    /* Turn off line interactions */
    LineInteractionType line_interaction_type;
} MonteCarloConfig;

/**
 * mc_config_default: Return default configuration
 * Thread-safe: returns value, no global state
 */
static inline MonteCarloConfig mc_config_default(void) {
    MonteCarloConfig cfg = {
        .enable_full_relativity = 0,
        .disable_line_scattering = 0,
        .line_interaction_type = LINE_SCATTER
    };
    return cfg;
}

/* ============================================================================
 * FUNCTION DECLARATIONS (HPC-Hardened signatures)
 * ============================================================================ */

/* --- Initialization --- */

/**
 * rpacket_init: Initialize a packet with given parameters
 *
 * @param pkt     Pointer to RPacket to initialize
 * @param r       Initial radial position [cm]
 * @param mu      Initial direction cosine
 * @param nu      Initial frequency [Hz]
 * @param energy  Initial energy weight [erg]
 * @param seed    RNG seed for this packet
 * @param index   Packet ID
 */
void rpacket_init(RPacket *pkt, double r, double mu, double nu,
                  double energy, uint64_t seed, int64_t index);

/**
 * rpacket_initialize_line_id: Find starting position in line list
 *
 * Performs binary search to find the first line with nu_line <= nu_cmf
 * This is the first line the packet could potentially interact with.
 */
void rpacket_initialize_line_id(RPacket *pkt, const NumbaPlasma *plasma,
                                 const NumbaModel *model);

/* --- Core Transport Functions --- */

/**
 * trace_packet: Determine next interaction point
 *
 * The HEART of Monte Carlo transport. This function:
 * 1. Calculates distance to shell boundaries
 * 2. Samples random optical depth for next event
 * 3. Loops through lines, accumulating Sobolev tau
 * 4. Compares distances: boundary vs electron vs line
 * 5. Returns winning interaction type and distance
 *
 * HPC Optimization: Early exit when distance_line > min(d_boundary, d_electron)
 *
 * @param pkt        Packet being traced
 * @param model      Geometry information
 * @param plasma     Opacity/density data
 * @param config     Monte Carlo configuration (thread-safe, not global)
 * @param estimators Updated during trace for J, nu_bar
 * @param distance   [out] Distance to interaction [cm]
 * @param delta_shell [out] +1 if hitting outer, -1 if hitting inner boundary
 *
 * @return InteractionType that stopped the packet
 */
InteractionType trace_packet(RPacket *pkt, const NumbaModel *model,
                             const NumbaPlasma *plasma,
                             const MonteCarloConfig *config,
                             Estimators *estimators,
                             double *distance, int *delta_shell);

/**
 * move_r_packet: Propagate packet by given distance
 *
 * Geometric update using spherical coordinates:
 *   r_new = sqrt(r^2 + d^2 + 2*r*d*mu)
 *   mu_new = (mu*r + d) / r_new
 *
 * Also updates J-estimators for radiative equilibrium.
 *
 * @param pkt            Packet to move (modified in place)
 * @param distance       Distance to travel [cm]
 * @param time_explosion Time since explosion [s]
 * @param estimators     J-estimators to update
 */
void move_r_packet(RPacket *pkt, double distance, double time_explosion,
                   Estimators *estimators);

/**
 * move_packet_across_shell_boundary: Handle shell transition
 *
 * Updates current_shell_id or sets termination status.
 *
 * @param pkt         Packet at boundary
 * @param delta_shell +1 (outward) or -1 (inward)
 * @param n_shells    Total number of shells
 */
void move_packet_across_shell_boundary(RPacket *pkt, int delta_shell,
                                        int64_t n_shells);

/* --- Main Loop --- */

/**
 * single_packet_loop: Process one packet until termination
 *
 * Main driver that calls trace_packet -> move -> interact repeatedly
 * until packet escapes (EMITTED) or is absorbed (REABSORBED).
 *
 * Thread-safe: no global state, RNG state in packet
 *
 * @param pkt        Packet to process
 * @param model      Geometry
 * @param plasma     Plasma conditions
 * @param config     Monte Carlo configuration
 * @param estimators Radiative estimators
 */
void single_packet_loop(RPacket *pkt, const NumbaModel *model,
                        const NumbaPlasma *plasma,
                        const MonteCarloConfig *config,
                        Estimators *estimators);

/* --- Physics Kernels (Distance, Transform, Opacity) --- */
/*
 * The following functions are provided as static inline in physics_kernels.h:
 *   - calculate_distance_boundary()
 *   - calculate_distance_electron()
 *   - calculate_distance_line()
 *   - get_doppler_factor()
 *   - get_inverse_doppler_factor()
 *   - calculate_tau_electron()
 *   - angle_aberration_CMF_to_LF()
 *   - angle_aberration_LF_to_CMF()
 *
 * Include physics_kernels.h BEFORE this header to access these functions.
 */

/* --- Interaction Handlers --- */

/**
 * thomson_scatter: Isotropic electron scattering
 *
 * In CMF: mu_new = 2*xi - 1 (isotropic)
 * Transform back to lab frame using angle_aberration_CMF_to_LF().
 * Update frequency using get_doppler_factor/get_inverse_doppler_factor.
 *
 * @param pkt            Packet to scatter (RNG state used internally)
 * @param time_explosion Time since explosion [s]
 */
void thomson_scatter(RPacket *pkt, double time_explosion);

/**
 * line_scatter: Resonant line interaction
 *
 * Depending on line_interaction_type:
 * - SCATTER: Coherent re-emission, isotropic in CMF
 * - DOWNBRANCH: Fluorescent de-excitation to lower state
 * - MACROATOM: Full macro-atom treatment
 *
 * Uses angle_aberration_CMF_to_LF() for direction transformation.
 * Uses get_inverse_doppler_factor() for frequency update.
 *
 * @param pkt              Packet to scatter
 * @param time_explosion   Time since explosion [s]
 * @param interaction_type Treatment mode
 * @param plasma           Plasma data (for line frequencies)
 */
void line_scatter(RPacket *pkt, double time_explosion,
                  LineInteractionType interaction_type,
                  const NumbaPlasma *plasma);

/* --- Estimator Updates --- */
void set_estimators(RPacket *pkt, double distance, Estimators *est,
                    double comov_nu, double comov_energy);

void update_line_estimators(Estimators *est, const RPacket *pkt,
                            int64_t line_id, double distance,
                            double time_explosion);

#endif /* RPACKET_H */
