/**
 * LUMINA-SN Simulation State
 * simulation_state.h - Integrated plasma-transport state management
 *
 * Holds pre-computed plasma properties and line opacities for all shells,
 * enabling physically consistent Monte Carlo transport with Saha-Boltzmann
 * ionization and Sobolev line opacities.
 *
 * Memory Layout Optimized for:
 *   1. Cache-efficient shell-by-shell access during transport
 *   2. Fast binary search for active lines in frequency windows
 *   3. Minimal memory footprint for large line lists
 */

#ifndef SIMULATION_STATE_H
#define SIMULATION_STATE_H

#include <stdbool.h>
#include "atomic_data.h"
#include "plasma_physics.h"
#include "rpacket.h"

/* ============================================================================
 * CONFIGURATION CONSTANTS
 * ============================================================================ */

/* Maximum number of shells in simulation */
#define MAX_SHELLS 100

/* Maximum number of "active" lines to track per shell */
#define MAX_ACTIVE_LINES 50000

/* Minimum optical depth to consider a line "active" */
#define TAU_MIN_ACTIVE 1e-10

/* Maximum optical depth cap (prevents numerical divergence)
 * Physical basis: For τ > 100, the escape probability is essentially 0,
 * and the line is completely optically thick. Capping prevents numerical
 * spikes from corrupting spectra while preserving physical behavior.
 * Reference: TARDIS uses similar capping for stability.
 */
/*
 * TASK ORDER #32 - PHASE 2: OPACITY SOFTENING & DYNAMIC RANGE
 * ============================================================
 *
 * Problem: "Flat-bottomed" saturated line profiles (τ >> 1) cause the
 *          absorption minimum to form at the high-velocity edge, creating
 *          an artificial blue-shift in the Si II feature.
 *
 * Solution:
 *   - TAU_MAX_CAP = 1000.0: Allow deeper line cores without truncation
 *   - OPACITY_SCALE = 0.05: Soften τ profile to reveal true Doppler centroid
 *
 * Physics: With τ_final = min(τ_raw × OPACITY_SCALE, TAU_MAX_CAP), the line
 *          wings are not artificially truncated, allowing the absorption
 *          profile to reflect the true velocity distribution of absorbers.
 *
 * Reference: TARDIS uses similar τ capping for numerical stability while
 *            preserving physically meaningful line profile shapes.
 */
#define TAU_MAX_CAP 1000.0   /* Phase 2: High cap for deep line cores */

/* Global opacity scaler (Task Order #32 Phase 2)
 * ----------------------------------------------
 * τ_final = min(τ_raw × OPACITY_SCALE, TAU_MAX_CAP)
 *
 * OPACITY_SCALE = 0.05 provides:
 *   - Sufficient optical depth for strong absorption (τ ~ 10-100)
 *   - Gradual wings that reveal the true velocity centroid
 *   - Prevention of "flat-bottomed" saturated profiles
 */
#define OPACITY_SCALE 0.05   /* Phase 2: Moderate softening for centroid accuracy */

/* ============================================================================
 * PHYSICS OVERRIDES CONFIGURATION (Task Order #30)
 * ============================================================================
 * Centralized struct for all physics parameters.
 * Replaces scattered hard-coded values with a portable configuration.
 *
 * Usage:
 *   PhysicsOverrides overrides = physics_overrides_default();
 *   overrides.t_boundary = 13000.0;  // Physical value (no hack)
 *   physics_overrides_set(&overrides);
 */

typedef struct {
    /* Boundary temperature [K]
     * With continuum opacity enabled, this should be PHYSICAL (10000-15000 K).
     * The 60000 K "hack" is only needed when continuum opacity is disabled.
     */
    double t_boundary;

    /* Thermalization probability for line interactions
     * With continuum opacity, this can be physical (0.3-0.5)
     */
    double ir_thermalization_frac;
    double base_thermalization_frac;

    /* Blue Opacity Reduction: Scale factor for Fe-group (Z=21-28) lines
     * in the 3500-4500 Å range. Lower values reduce line blanketing.
     * Range: 0.0-1.0, Default: 0.50
     */
    double blue_opacity_scalar;

    /* Wavelength boundaries [Angstrom] */
    double ir_wavelength_min;     /* λ > this triggers IR thermalization (7000 Å) */
    double blue_wavelength_min;   /* Fe-group reduction range start (3500 Å) */
    double blue_wavelength_max;   /* Fe-group reduction range end (4500 Å) */

    /* =========== NEW: Continuum Opacity Controls (Task Order #30 v2) =========== */

    /* Enable/disable continuum opacity calculation */
    bool enable_continuum_opacity;

    /* Bound-free (photoionization) opacity scaling
     * 1.0 = standard Kramers formula, adjust to calibrate
     */
    double bf_opacity_scale;

    /* Free-free (bremsstrahlung) opacity scaling */
    double ff_opacity_scale;

    /* Photospheric radius for dilution factor [cm]
     * W = 0.5 × [1 - sqrt(1 - (R_ph/r)^2)]
     * Set to 0 to disable NLTE dilution correction
     */
    double R_photosphere;

    /* Dilution factor application flag */
    bool enable_dilution_factor;

    /* =========== NEW: Wavelength-Dependent Fluorescence (Task Order #30 v2.1) =========== */

    /* UV Fluorescence: When UV photons (λ < uv_cutoff) are absorbed by metal lines,
     * they excite atoms to high energy states. The cascade down through intermediate
     * states preferentially produces OPTICAL photons, not thermal emission.
     *
     * This is the key physics that allows T_boundary ~ 13,000 K to work:
     * UV photons don't thermalize at T_boundary; they fluoresce to blue/optical.
     */

    bool   enable_wavelength_fluorescence;   /* Enable wavelength-dependent model */
    double uv_cutoff_angstrom;               /* λ below this → UV fluorescence (3000 Å) */
    double uv_to_blue_probability;           /* Fraction of UV → blue fluorescence (0.8) */
    double blue_fluor_min_angstrom;          /* Blue fluorescence emission range min (3500 Å) */
    double blue_fluor_max_angstrom;          /* Blue fluorescence emission range max (5500 Å) */
    double blue_scatter_probability;         /* Blue photons: scatter vs thermalize (0.7) */

} PhysicsOverrides;

/* Global physics overrides instance */
extern PhysicsOverrides g_physics_overrides;

/* Initialize with default values - PHYSICAL MODE with continuum opacity */
PhysicsOverrides physics_overrides_default(void);

/* Initialize with legacy "hack" values (60,000 K boundary, no continuum) */
PhysicsOverrides physics_overrides_legacy_hack(void);

/* Apply overrides globally */
void physics_overrides_set(const PhysicsOverrides *overrides);

/* Legacy compatibility - will be deprecated */
extern double g_fe_blue_scale;

/* Frequency window factor for line pre-selection */
#define FREQ_WINDOW_FACTOR 1.1

/* ============================================================================
 * LINE OPACITY DATA (per shell)
 * ============================================================================
 * Pre-computed Sobolev optical depths for efficient transport.
 */

typedef struct {
    int64_t line_idx;      /* Index into AtomicData.lines[] */
    double  nu;            /* Line frequency [Hz] */
    double  tau_sobolev;   /* Sobolev optical depth */
} ActiveLine;

/* ============================================================================
 * FREQUENCY-BINNED LINE INDEX
 * ============================================================================
 * Enables O(1) lookup of line search start position in trace_packet.
 * Instead of binary search over 270k+ lines, bin lookup + local scan.
 */

#define LINE_INDEX_N_BINS     2000    /* Number of frequency bins */
#define LINE_INDEX_NU_MIN     1e13    /* 30 μm (IR) */
#define LINE_INDEX_NU_MAX     3e16    /* 100 Å (UV) */

typedef struct {
    int64_t bin_start[LINE_INDEX_N_BINS + 1];  /* Start index for each bin */
    double  nu_min;                             /* Minimum frequency */
    double  nu_max;                             /* Maximum frequency */
    double  d_log_nu;                           /* Log frequency step per bin */
    int64_t n_lines;                            /* Total lines in index */
    bool    initialized;
} FrequencyBinnedIndex;

typedef struct {
    /* Shell properties */
    int     shell_id;
    double  r_inner;       /* Inner radius [cm] */
    double  r_outer;       /* Outer radius [cm] */
    double  v_inner;       /* Inner velocity [cm/s] */
    double  v_outer;       /* Outer velocity [cm/s] */
    double  t_exp;         /* Expansion time [s] */

    /* Per-shell abundances (for stratified composition) */
    Abundances abundances;

    /* Plasma state */
    PlasmaState plasma;

    /* Line opacities: sorted by frequency for binary search */
    int64_t     n_active_lines;
    ActiveLine *active_lines;      /* Array sorted by nu (ascending) */

    /* Frequency-binned index for O(1) line lookup */
    FrequencyBinnedIndex line_index;

    /* Pre-computed totals */
    double  tau_electron;          /* Thomson electron scattering τ */
    double  sigma_thomson_ne;      /* n_e × σ_Thomson */

} ShellState;

/* ============================================================================
 * SIMULATION STATE (global)
 * ============================================================================
 * Master structure holding all pre-computed data for MC transport.
 */

typedef struct {
    /* Simulation metadata */
    char    name[64];
    double  t_explosion;           /* Time since explosion [s] */
    int     n_shells;

    /* Shell data */
    ShellState *shells;

    /* Atomic data reference (not owned) */
    const AtomicData *atomic_data;

    /* Element abundances */
    Abundances abundances;

    /* Global statistics */
    int64_t total_active_lines;    /* Sum over all shells */
    double  memory_usage_mb;

    /* Frequency-sorted global line index for fast lookups */
    int64_t  n_lines_total;
    int64_t *line_freq_order;      /* Global line indices sorted by ν */
    double  *line_freq_sorted;     /* Corresponding frequencies */

    /* Simulation flags */
    bool    initialized;
    bool    opacities_computed;

} SimulationState;

/* ============================================================================
 * INITIALIZATION AND SETUP
 * ============================================================================ */

/**
 * Initialize simulation state with given model parameters
 *
 * @param state      SimulationState to initialize
 * @param atomic     Atomic data (must remain valid during simulation)
 * @param n_shells   Number of radial shells
 * @param t_exp      Expansion time [s]
 * @return 0 on success, -1 on error
 */
int simulation_state_init(SimulationState *state,
                          const AtomicData *atomic,
                          int n_shells,
                          double t_exp);

/**
 * Set shell radii and velocities (homologous expansion assumed)
 *
 * @param state      SimulationState
 * @param shell_id   Shell index (0 to n_shells-1)
 * @param r_inner    Inner radius [cm]
 * @param r_outer    Outer radius [cm]
 */
void simulation_set_shell_geometry(SimulationState *state,
                                    int shell_id,
                                    double r_inner,
                                    double r_outer);

/**
 * Set shell density
 *
 * @param state      SimulationState
 * @param shell_id   Shell index
 * @param rho        Mass density [g/cm³]
 */
void simulation_set_shell_density(SimulationState *state,
                                   int shell_id,
                                   double rho);

/**
 * Set shell temperature
 *
 * @param state      SimulationState
 * @param shell_id   Shell index
 * @param T          Temperature [K]
 */
void simulation_set_shell_temperature(SimulationState *state,
                                       int shell_id,
                                       double T);

/**
 * Set element abundances (applies to all shells uniformly)
 */
void simulation_set_abundances(SimulationState *state,
                                const Abundances *ab);

/**
 * Set stratified abundances (velocity-dependent, per shell)
 *
 * Uses abundances_set_type_ia_stratified() to set composition
 * based on shell velocity, creating realistic SN Ia stratification.
 */
void simulation_set_stratified_abundances(SimulationState *state);

/**
 * Set stratified abundances with scaling factors
 *
 * Same as simulation_set_stratified_abundances() but applies
 * abundance scaling factors for optimizer-driven parameter sweeps.
 *
 * @param Si_scale  Silicon abundance multiplier
 * @param Fe_scale  Iron abundance multiplier
 * @param Ca_scale  Calcium abundance multiplier
 * @param S_scale   Sulfur abundance multiplier
 */
void simulation_set_scaled_abundances(SimulationState *state,
                                       double Si_scale, double Fe_scale,
                                       double Ca_scale, double S_scale);

/**
 * Free all allocated memory
 */
void simulation_state_free(SimulationState *state);

/* ============================================================================
 * PLASMA & OPACITY CALCULATION
 * ============================================================================ */

/**
 * Compute plasma state (ionization, level populations) for all shells
 *
 * Must be called after setting T and ρ for all shells.
 *
 * @return 0 on success, number of shells that failed to converge on error
 */
int simulation_compute_plasma(SimulationState *state);

/**
 * Compute Sobolev optical depths for all lines in all shells
 *
 * Must be called after simulation_compute_plasma().
 *
 * @return Total number of active lines across all shells
 */
int64_t simulation_compute_opacities(SimulationState *state);

/**
 * Build frequency-sorted line index for fast binary search
 *
 * Called automatically by simulation_compute_opacities().
 */
void simulation_build_line_index(SimulationState *state);

/**
 * Build frequency-binned index for O(1) line lookup
 *
 * @param index        FrequencyBinnedIndex to initialize
 * @param lines        Array of active lines (sorted by frequency)
 * @param n_lines      Number of lines
 */
void build_frequency_binned_index(FrequencyBinnedIndex *index,
                                   const ActiveLine *lines,
                                   int64_t n_lines);

/**
 * Find starting index for line search in given frequency range
 *
 * @param index   Frequency-binned index
 * @param nu      Target frequency [Hz]
 * @return        Index to start searching from (or -1 if out of range)
 */
int64_t frequency_index_find_start(const FrequencyBinnedIndex *index, double nu);

/* ============================================================================
 * CONTINUUM OPACITY CALCULATION (Task Order #30 v2)
 * ============================================================================
 * Physical continuum opacity to replace the "60,000 K hack".
 *
 * Key insight: The UV-Leak protocol (T_boundary = 60,000 K) was compensating
 * for missing continuum opacity. By implementing bound-free and free-free
 * opacity, we can use physical T_boundary values (10,000-15,000 K).
 */

/**
 * Calculate bound-free (photoionization) opacity
 *
 * Uses Kramers' cross-section:
 *   σ_bf = σ_0 × (Z^4 / n^5) × (ν_n / ν)^3 × g_bf
 *
 * where:
 *   σ_0 = 7.906e-18 cm² (hydrogen ground state)
 *   Z = effective nuclear charge
 *   n = principal quantum number
 *   ν_n = ionization threshold frequency
 *   g_bf = Gaunt factor ≈ 1 for simplicity
 *
 * @param nu      Photon frequency [Hz]
 * @param T       Temperature [K]
 * @param n_e     Electron density [cm⁻³]
 * @param plasma  PlasmaState with ion populations
 * @return Bound-free opacity coefficient κ_bf [cm⁻¹]
 */
double calculate_bf_opacity(double nu, double T, double n_e,
                            const PlasmaState *plasma);

/**
 * Calculate free-free (bremsstrahlung) opacity
 *
 * Uses standard formula:
 *   κ_ff = 3.7e8 × (g_ff / T^0.5) × (Z^2 × n_e × n_ion) / ν³
 *        × (1 - exp(-hν/kT))
 *
 * where g_ff ≈ 1 (Gaunt factor)
 *
 * @param nu      Photon frequency [Hz]
 * @param T       Temperature [K]
 * @param n_e     Electron density [cm⁻³]
 * @param plasma  PlasmaState with ion populations
 * @return Free-free opacity coefficient κ_ff [cm⁻¹]
 */
double calculate_ff_opacity(double nu, double T, double n_e,
                            const PlasmaState *plasma);

/**
 * Calculate total continuum opacity (bf + ff)
 *
 * @param nu      Photon frequency [Hz]
 * @param shell   ShellState containing plasma properties
 * @return Total continuum opacity κ_cont [cm⁻¹]
 */
double calculate_continuum_opacity(double nu, const ShellState *shell);

/**
 * Calculate optical depth to continuum absorption across shell
 *
 * τ_cont = κ_cont × Δr
 *
 * @param nu      Photon frequency [Hz]
 * @param shell   ShellState
 * @return Continuum optical depth τ_cont
 */
double calculate_tau_continuum(double nu, const ShellState *shell);

/* ============================================================================
 * DILUTION FACTOR CALCULATION (NLTE Correction)
 * ============================================================================
 * The dilution factor W accounts for the geometric dilution of the
 * radiation field in the outer ejecta, crucial for NLTE effects.
 *
 * W = 0.5 × [1 - sqrt(1 - (R_ph/r)^2)]
 *
 * At the photosphere (r = R_ph): W = 0.5
 * Far from photosphere (r >> R_ph): W → 0
 */

/**
 * Calculate dilution factor W at radius r
 *
 * @param r       Current radius [cm]
 * @param R_ph    Photospheric radius [cm]
 * @return Dilution factor W (0 < W < 0.5)
 */
double calculate_dilution_factor(double r, double R_ph);

/**
 * Apply dilution correction to Saha equation
 *
 * In a diluted radiation field, the effective radiation temperature
 * is reduced: T_rad_eff = W^0.25 × T_rad
 *
 * This modifies the ionization balance toward lower ionization states
 * in the outer shells.
 *
 * @param T       Local electron temperature [K]
 * @param W       Dilution factor
 * @return Effective radiation temperature [K]
 */
double apply_dilution_to_temperature(double T, double W);

/* ============================================================================
 * SOBOLEV OPTICAL DEPTH CALCULATION
 * ============================================================================ */

/**
 * Calculate Sobolev optical depth for a single line (oscillator strength form)
 *
 * τ_Sobolev = (π e² / m_e c) × f_lu × λ × n_lower × t_exp
 *
 * Note: Result is capped at TAU_MAX_CAP (100.0) for numerical stability.
 *
 * @param line       Line data
 * @param n_lower    Lower level population [cm⁻³]
 * @param t_exp      Expansion time [s]
 * @return Sobolev optical depth (capped at TAU_MAX_CAP)
 */
double calculate_tau_sobolev(const Line *line, double n_lower, double t_exp);

/**
 * Calculate Sobolev optical depth using Einstein A coefficient form
 *
 * τ_Sobolev = (λ³ / 8π) × A_ul × (g_u / g_l) × n_lower × t_exp × (1 - exp(-hν/kT))^-1
 *
 * This alternative formulation uses the spontaneous emission coefficient A_ul
 * and includes the stimulated emission correction factor.
 *
 * Note: Result is capped at TAU_MAX_CAP (100.0) for numerical stability.
 *
 * @param line       Line data (must have valid A_ul)
 * @param n_lower    Lower level population [cm⁻³]
 * @param t_exp      Expansion time [s]
 * @param T          Temperature [K] (for stimulated emission correction)
 * @return Sobolev optical depth (capped at TAU_MAX_CAP)
 */
double calculate_tau_sobolev_A(const Line *line, double n_lower, double t_exp, double T);

/**
 * Calculate Sobolev optical depths for all lines in a shell
 *
 * @param state      SimulationState
 * @param shell_id   Shell index
 * @return Number of active lines (τ > TAU_MIN_ACTIVE)
 */
int64_t calculate_shell_opacities(SimulationState *state, int shell_id);

/* ============================================================================
 * LINE LOOKUP (Hot Path Optimization)
 * ============================================================================ */

/**
 * Find lines within frequency window [nu_min, nu_max] for a shell
 *
 * Uses binary search on frequency-sorted active_lines array.
 * This is the "hot path" called during packet transport.
 *
 * @param shell      ShellState
 * @param nu_min     Minimum frequency [Hz]
 * @param nu_max     Maximum frequency [Hz]
 * @param first_idx  Output: index of first matching line
 * @param last_idx   Output: index of last matching line (exclusive)
 * @return Number of lines in range
 */
int64_t find_lines_in_window(const ShellState *shell,
                              double nu_min, double nu_max,
                              int64_t *first_idx, int64_t *last_idx);

/**
 * Get the next line interaction for a packet
 *
 * Given packet frequency ν and direction μ, find the closest line
 * and calculate distance to interaction.
 *
 * @param shell      ShellState
 * @param nu_cmf     Comoving frame frequency [Hz]
 * @param r          Current radius [cm]
 * @param mu         Direction cosine
 * @param t_exp      Expansion time [s]
 * @param line_idx   Output: index of interacting line (or -1 if none)
 * @param tau_line   Output: optical depth of line
 * @return Distance to line interaction [cm], or MISS_DISTANCE if none
 */
double get_next_line_interaction(const ShellState *shell,
                                  double nu_cmf, double r, double mu,
                                  double t_exp,
                                  int64_t *line_idx, double *tau_line);

/* ============================================================================
 * TRANSPORT INTEGRATION
 * ============================================================================ */

/**
 * Convert NumbaModel to SimulationState
 *
 * Transfers geometry and plasma properties from TARDIS-style model.
 */
int simulation_from_numba_model(SimulationState *state,
                                 const NumbaModel *model,
                                 const AtomicData *atomic);

/**
 * Update NumbaPlasma from SimulationState
 *
 * Copies computed opacities to TARDIS-style plasma structure.
 */
void simulation_to_numba_plasma(const SimulationState *state,
                                 NumbaPlasma *plasma);

/* ============================================================================
 * DIAGNOSTICS
 * ============================================================================ */

/**
 * Print simulation state summary
 */
void simulation_print_summary(const SimulationState *state);

/**
 * Print detailed shell information
 */
void simulation_print_shell(const SimulationState *state, int shell_id);

/**
 * Validate simulation state consistency
 *
 * @return Number of errors found
 */
int simulation_validate(const SimulationState *state);

#endif /* SIMULATION_STATE_H */
