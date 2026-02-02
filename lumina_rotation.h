/**
 * LUMINA-SN Innovation: Post-Processing Rotation & Weighting
 * lumina_rotation.h - Observer-frame spectrum synthesis
 *
 * ============================================================================
 * PHYSICAL CONCEPT
 * ============================================================================
 *
 * In traditional Monte Carlo radiative transfer (TARDIS), packets escape
 * in all directions. To compute a spectrum as seen by an observer, one must
 * either:
 *   1. Run separate simulations for each viewing angle (expensive)
 *   2. Use "virtual packets" at each interaction (memory intensive)
 *
 * LUMINA's innovation: POST-PROCESSING ROTATION
 * ---------------------------------------------
 * After the Monte Carlo transport is complete, we rotate each escaped
 * packet's final trajectory to align with the observer's line of sight.
 *
 * This is physically valid because:
 *   - The radiative transfer is computed in the comoving frame
 *   - The final lab-frame direction can be rotated to any observer angle
 *   - Time delays and solid angle corrections preserve causality and flux
 *
 * ============================================================================
 * KEY FORMULAS
 * ============================================================================
 *
 * 1. ROTATION TO OBSERVER AXIS
 *    --------------------------
 *    For observer at angle θ_obs from the z-axis:
 *      μ_obs = cos(θ_obs)  (typically μ_obs = 1 for face-on)
 *
 *    The packet's original escape direction (μ_packet) is rotated.
 *    In spherical symmetry, the azimuthal angle is irrelevant, so:
 *      μ_rotated = μ_obs (project onto observer axis)
 *
 * 2. TIME-DELAY CORRECTION
 *    ----------------------
 *    Packets escaping at different angles arrive at the observer at
 *    different times due to light travel time across the ejecta.
 *
 *    t_observed = t_escape - (r_escape × μ_packet) / c
 *
 *    Where:
 *      t_escape: time when packet crossed outer boundary
 *      r_escape: radius at escape (= R_outer)
 *      μ_packet: direction cosine at escape
 *      c: speed of light
 *
 *    This creates "photospheric velocity" features in spectra.
 *
 * 3. SOLID ANGLE WEIGHTING
 *    ----------------------
 *    To correctly normalize the flux, we weight by the solid angle
 *    subtended by the packet's original direction.
 *
 *    w = 1 / (4π) × dΩ / dΩ_obs
 *
 *    For isotropic emission in CMF, the lab-frame distribution is:
 *      dN/dμ ∝ 1 / D²   (Doppler beaming)
 *
 *    Weight factor for observer at μ_obs:
 *      w = D(μ_packet)² / D(μ_obs)²
 *
 * ============================================================================
 */

#ifndef LUMINA_ROTATION_H
#define LUMINA_ROTATION_H

#include <math.h>
#include <stdint.h>
#include "physics_kernels.h"
#include "validation.h"

/* ============================================================================
 * OBSERVER CONFIGURATION
 * ============================================================================ */

typedef struct {
    double mu_observer;        /* cos(θ) of observer viewing angle, 1.0 = face-on */
    double time_explosion;     /* Reference time for delays [s] */
    double r_outer;            /* Outer boundary radius [cm] */
    double wavelength_min;     /* Spectrum range: min wavelength [Å] */
    double wavelength_max;     /* Spectrum range: max wavelength [Å] */
    int64_t n_wavelength_bins; /* Number of wavelength bins */
} ObserverConfig;

/* ============================================================================
 * ROTATED PACKET: Result after post-processing
 * ============================================================================ */

typedef struct {
    double energy_weighted;    /* Energy × weight factor [erg] */
    double nu_observer;        /* Frequency in observer frame [Hz] */
    double wavelength;         /* Wavelength in observer frame [Å] */
    double t_observed;         /* Arrival time at observer [s] */
    double weight;             /* Solid angle weight factor */
    int64_t original_index;    /* Index into original packet array */
    int valid;                 /* 1 if packet contributes, 0 if not */
} RotatedPacket;

/* ============================================================================
 * SPECTRUM ACCUMULATOR
 * ============================================================================ */

typedef struct {
    double *flux;              /* Flux per wavelength bin [erg/s/Å] */
    double *wavelength_centers;/* Center wavelength of each bin [Å] */
    double wavelength_min;
    double wavelength_max;
    double d_wavelength;       /* Bin width [Å] */
    int64_t n_bins;
    double total_luminosity;   /* Integrated L [erg/s] */
    int64_t n_packets_used;    /* Number of contributing packets */
} Spectrum;

/* ============================================================================
 * FUNCTION DECLARATIONS
 * ============================================================================ */

/**
 * lumina_rotate_packet: Apply rotation and weighting to single escaped packet
 *
 * This is the core LUMINA algorithm that transforms a Monte Carlo packet
 * into an observer-frame contribution.
 *
 * @param final_r      Radius at escape [cm]
 * @param final_mu     Direction cosine at escape [-1, 1]
 * @param final_nu     Frequency at escape [Hz]
 * @param final_energy Energy at escape [erg]
 * @param config       Observer configuration
 * @param result       Output: rotated packet properties
 */
void lumina_rotate_packet(double final_r, double final_mu,
                          double final_nu, double final_energy,
                          const ObserverConfig *config,
                          RotatedPacket *result);

/**
 * lumina_apply_rotation_weighting: Process entire trace for spectrum
 *
 * Takes a validation trace of an EMITTED packet and applies the full
 * rotation + weighting algorithm to compute its contribution to the
 * observed spectrum.
 *
 * @param trace        Packet history (must end with status=EMITTED)
 * @param config       Observer configuration
 * @param result       Output: rotated packet
 * @return 0 on success, -1 if packet was not emitted
 */
int lumina_apply_rotation_weighting(const ValidationTrace *trace,
                                    const ObserverConfig *config,
                                    RotatedPacket *result);

/**
 * spectrum_create: Allocate a new spectrum accumulator
 */
Spectrum *spectrum_create(double wavelength_min, double wavelength_max,
                          int64_t n_bins);

/**
 * spectrum_free: Deallocate spectrum
 */
void spectrum_free(Spectrum *spec);

/**
 * spectrum_add_packet: Add a rotated packet's contribution to spectrum
 */
void spectrum_add_packet(Spectrum *spec, const RotatedPacket *pkt,
                         double simulation_time);

/**
 * spectrum_normalize: Normalize by number of packets and solid angle
 */
void spectrum_normalize(Spectrum *spec, int64_t n_total_packets,
                        double simulation_time);

/**
 * spectrum_write_csv: Write spectrum to file
 */
int spectrum_write_csv(const Spectrum *spec, const char *filename);

/**
 * TASK ORDER #27: Weight Diagnostics
 * Reset and report mean weight of escaped packets.
 * If mean(w) deviates from 1.0, energy is not conserved.
 */
void lumina_reset_weight_diagnostics(void);
void lumina_report_weight_diagnostics(void);

/* ============================================================================
 * PEELING-OFF TECHNIQUE: Per-Interaction Virtual Packets
 * ============================================================================
 *
 * The peeling-off (a.k.a. "next event estimation") technique improves
 * signal-to-noise by calculating a virtual packet contribution toward
 * the observer at EVERY interaction point, not just escape.
 *
 * At each interaction (line scatter, electron scatter):
 *   1. Calculate probability of packet reaching observer (P_escape)
 *   2. Rotate packet toward observer direction
 *   3. Apply Doppler shift for observer frame
 *   4. Add weighted energy contribution to spectrum
 *
 * Benefits:
 *   - Captures contribution from packets that eventually get absorbed
 *   - Smoother spectra with same number of packets
 *   - Better sampling of absorption features
 * ============================================================================ */

/**
 * PeelingContext: Thread-local accumulator for peeling contributions
 *
 * Each thread maintains its own PeelingContext with a local spectrum
 * to avoid race conditions during OpenMP parallel transport.
 */
typedef struct {
    Spectrum *local_spectrum;     /* Thread-local spectrum array */
    ObserverConfig obs_config;    /* Observer parameters */
    double simulation_time;       /* For flux normalization */
    int64_t n_peeling_events;     /* Count of peeling contributions */
    double total_peeling_energy;  /* Sum of weighted energies */
} PeelingContext;

/**
 * peeling_context_create: Allocate a new peeling context
 *
 * @param obs_config     Observer configuration
 * @param simulation_time Time for flux normalization [s]
 * @return New context or NULL on failure
 */
PeelingContext *peeling_context_create(const ObserverConfig *obs_config,
                                        double simulation_time);

/**
 * peeling_context_free: Deallocate peeling context
 */
void peeling_context_free(PeelingContext *ctx);

/**
 * peeling_add_contribution: Add peeling contribution at interaction point
 *
 * This is called at each LINE and ESCATTERING interaction to calculate
 * the virtual packet contribution toward the observer.
 *
 * Physical model:
 *   P_escape = exp(-tau_to_observer) approximated by geometric attenuation
 *   weight = solid_angle_correction * P_escape
 *   E_contrib = packet_energy * weight
 *
 * @param ctx      Peeling context (thread-local)
 * @param r        Radial position at interaction [cm]
 * @param mu       Direction cosine at interaction
 * @param nu       Frequency at interaction [Hz]
 * @param energy   Packet energy at interaction [erg]
 * @param t_exp    Time since explosion [s]
 */
void peeling_add_contribution(PeelingContext *ctx,
                              double r, double mu, double nu, double energy,
                              double t_exp);

/**
 * peeling_merge_into_spectrum: Merge thread-local spectrum into global
 *
 * Call this after parallel transport completes to aggregate all
 * thread-local spectra.
 *
 * @param ctx          Thread-local context to merge
 * @param global_spec  Target global spectrum (must be protected by mutex)
 */
void peeling_merge_into_spectrum(PeelingContext *ctx, Spectrum *global_spec);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * frequency_to_wavelength: Convert frequency [Hz] to wavelength [Å]
 * λ = c / ν × 10^8 (cm → Å)
 */
static inline double frequency_to_wavelength(double nu) {
    return C_SPEED_OF_LIGHT / nu * 1e8;  /* cm → Angstrom */
}

/**
 * wavelength_to_frequency: Convert wavelength [Å] to frequency [Hz]
 */
static inline double wavelength_to_frequency(double wavelength_angstrom) {
    return C_SPEED_OF_LIGHT / (wavelength_angstrom * 1e-8);
}

#endif /* LUMINA_ROTATION_H */
