/**
 * LUMINA-SN Physics Kernels
 * physics_kernels.h - Exact numerical implementations matching TARDIS-SN
 *
 * Transpiled from:
 *   - tardis/montecarlo/montecarlo_numba/calculate_distances.py
 *   - tardis/montecarlo/montecarlo_numba/frame_transformations.py
 *   - tardis/montecarlo/montecarlo_numba/opacities.py
 *
 * CRITICAL: These constants and formulas must match TARDIS-SN exactly
 * to achieve 10^-10 relative tolerance in validation tests.
 */

#ifndef PHYSICS_KERNELS_H
#define PHYSICS_KERNELS_H

#include <math.h>
#include <stdint.h>

/* ============================================================================
 * PHYSICAL CONSTANTS - Must match TARDIS numba_config.py exactly
 * ============================================================================ */

/**
 * Speed of light [cm/s]
 * Source: NIST CODATA 2018
 * TARDIS uses: from astropy import constants
 */
#define C_SPEED_OF_LIGHT 2.99792458e10

/**
 * Thomson scattering cross-section [cm²]
 * σ_T = (8π/3) × (e²/m_e c²)² = 6.6524587158e-25 cm²
 * TARDIS value from scipy.constants
 */
#define SIGMA_THOMSON 6.6524587158e-25

/**
 * "Miss" distance - returned when no interaction possible
 * Effectively infinite distance in simulation units
 */
#define MISS_DISTANCE 1e99

/**
 * Close line threshold - for numerical stability
 * If |Δν/ν| < threshold, treat as zero to avoid floating-point issues
 * TARDIS default: 1e-7 (from numba_config.py)
 */
#define CLOSE_LINE_THRESHOLD 1e-7

/* ============================================================================
 * RELATIVITY MODE - Compile-time or runtime flag
 * ============================================================================
 *
 * ENABLE_FULL_RELATIVITY = 0: First-order Doppler (v/c terms only)
 *   D = 1 - βμ
 *   Sufficient for v << c (typical SN ejecta: v ~ 0.01-0.1 c)
 *
 * ENABLE_FULL_RELATIVITY = 1: Special relativistic
 *   D = (1 - βμ) / √(1 - β²)
 *   Includes time dilation factor γ = 1/√(1-β²)
 */
extern int ENABLE_FULL_RELATIVITY;

/* ============================================================================
 * FORWARD DECLARATIONS (for RPacket dependency)
 * ============================================================================ */
struct RPacket;  /* Forward declaration - full definition in rpacket.h */

/* ============================================================================
 * FRAME TRANSFORMATIONS
 * ============================================================================
 *
 * In homologous expansion: v(r) = r/t (velocity proportional to radius)
 * β = v/c = r/(ct)
 *
 * Doppler effect:
 * - Photon moving through expanding medium sees frequency shift
 * - Lab frame: observer at infinity
 * - Comoving frame (CMF): local rest frame of plasma at position r
 */

/**
 * get_doppler_factor: Transform frequency from Lab → Comoving frame
 *
 * ν_cmf = ν_lab × D
 *
 * Physical meaning:
 * - D < 1 for outward-moving photon (μ > 0): redshift in CMF
 * - D > 1 for inward-moving photon (μ < 0): blueshift in CMF
 *
 * @param r              Radial position [cm]
 * @param mu             Direction cosine (cos θ from radial)
 * @param time_explosion Time since explosion [s]
 * @return Doppler factor D
 */
static inline double get_doppler_factor(double r, double mu, double time_explosion) {
    double inv_c = 1.0 / C_SPEED_OF_LIGHT;
    double inv_t = 1.0 / time_explosion;
    double beta = r * inv_t * inv_c;

    if (!ENABLE_FULL_RELATIVITY) {
        /* Partial relativity: D = 1 - βμ */
        return 1.0 - mu * beta;
    } else {
        /* Full relativity: D = (1 - βμ) / γ where γ = 1/√(1-β²) */
        return (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
    }
}

/**
 * get_inverse_doppler_factor: Transform frequency from Comoving → Lab frame
 *
 * ν_lab = ν_cmf × D_inv
 *
 * IMPORTANT: The full-relativity formula is NOT simply 1/D!
 * This is because the transformation depends on the direction in each frame.
 *
 * Partial: D_inv = 1/(1 - βμ)
 * Full:    D_inv = (1 + βμ)/√(1 - β²)   [Note: + sign, not -]
 *
 * @param r              Radial position [cm]
 * @param mu             Direction cosine in lab frame
 * @param time_explosion Time since explosion [s]
 * @return Inverse Doppler factor
 */
static inline double get_inverse_doppler_factor(double r, double mu, double time_explosion) {
    double inv_c = 1.0 / C_SPEED_OF_LIGHT;
    double inv_t = 1.0 / time_explosion;
    double beta = r * inv_t * inv_c;

    if (!ENABLE_FULL_RELATIVITY) {
        /* Partial relativity: D_inv = 1/(1 - βμ) */
        return 1.0 / (1.0 - mu * beta);
    } else {
        /* Full relativity: D_inv = (1 + βμ)/√(1 - β²) */
        return (1.0 + mu * beta) / sqrt(1.0 - beta * beta);
    }
}

/**
 * angle_aberration_CMF_to_LF: Convert direction from Comoving → Lab frame
 *
 * μ_lab = (μ_cmf + β) / (1 + β × μ_cmf)
 *
 * This is the relativistic aberration formula.
 */
static inline double angle_aberration_CMF_to_LF(double r, double mu_cmf, double time_explosion) {
    double beta = r / (C_SPEED_OF_LIGHT * time_explosion);
    return (mu_cmf + beta) / (1.0 + beta * mu_cmf);
}

/**
 * angle_aberration_LF_to_CMF: Convert direction from Lab → Comoving frame
 *
 * μ_cmf = (μ_lab - β) / (1 - β × μ_lab)
 */
static inline double angle_aberration_LF_to_CMF(double r, double mu_lab, double time_explosion) {
    double beta = r / (C_SPEED_OF_LIGHT * time_explosion);
    return (mu_lab - beta) / (1.0 - beta * mu_lab);
}

/* ============================================================================
 * DISTANCE CALCULATIONS
 * ============================================================================ */

/**
 * calculate_distance_boundary: Distance to shell boundary [cm]
 *
 * Geometry: Photon at (r, μ) traveling in straight line.
 * Find distance d where it intersects sphere of radius R.
 *
 * |r + d×μ̂|² = R²
 * r² + d² + 2rdμ = R²
 * d² + 2rdμ + (r² - R²) = 0
 *
 * Using (μ² - 1) = -sin²θ:
 * d = -rμ ± √(R² - r²sin²θ)
 * d = -rμ ± √(R² + r²(μ² - 1))
 *
 * Sign selection:
 * - Outward (μ > 0): d = √(...) - rμ (positive root)
 * - Inward (μ < 0): depends on whether ray hits inner boundary
 *
 * @param r           Current radial position [cm]
 * @param mu          Direction cosine
 * @param r_inner     Inner radius of shell [cm]
 * @param r_outer     Outer radius of shell [cm]
 * @param delta_shell [out] +1 if hitting outer, -1 if hitting inner
 * @return Distance to boundary [cm]
 */
static inline double calculate_distance_boundary(double r, double mu,
                                                   double r_inner, double r_outer,
                                                   int *delta_shell) {
    double distance;
    double mu_sq_minus_1 = mu * mu - 1.0;  /* = -sin²θ, always ≤ 0 */

    if (mu > 0.0) {
        /*
         * Moving outward: will definitely hit outer boundary
         * d = √(r_outer² + (μ²-1)r²) - rμ
         */
        double discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r;
        distance = sqrt(discriminant) - r * mu;
        *delta_shell = 1;
    } else {
        /*
         * Moving inward: check if ray intersects inner boundary
         * Discriminant: r_inner² + r²(μ² - 1) = r_inner² - r²sin²θ
         * If ≥ 0: ray hits inner boundary
         * If < 0: ray misses inner (closest approach > r_inner), curves to outer
         */
        double check = r_inner * r_inner + r * r * mu_sq_minus_1;

        if (check >= 0.0) {
            /* Hits inner boundary */
            distance = -r * mu - sqrt(check);
            *delta_shell = -1;
        } else {
            /* Misses inner boundary, will hit outer */
            double discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r;
            distance = sqrt(discriminant) - r * mu;
            *delta_shell = 1;
        }
    }

    return distance;
}

/**
 * calculate_distance_line: Distance to line resonance [cm]
 *
 * In Sobolev approximation, a photon interacts with a line when its
 * comoving frequency equals the line rest frequency:
 *   ν_cmf(r + d) = ν_line
 *
 * Due to expansion, ν_cmf redshifts as packet moves outward.
 * For partial relativity:
 *   ν_cmf = ν_lab × (1 - βμ)
 *
 * At distance d, the Doppler factor changes. Solving for d:
 *   d = (ν_cmf - ν_line) / ν_lab × c × t
 *     = (Δν/ν) × c × t
 *
 * @param nu          Lab-frame frequency [Hz]
 * @param comov_nu    Current comoving frequency [Hz]
 * @param is_last_line  If true, return MISS_DISTANCE
 * @param nu_line     Line rest frequency [Hz]
 * @param time_explosion Time since explosion [s]
 * @param r           Current position [cm] (for full relativity)
 * @param mu          Direction cosine (for full relativity)
 * @return Distance to resonance [cm]
 */
static inline double calculate_distance_line(double nu, double comov_nu,
                                              int is_last_line, double nu_line,
                                              double time_explosion,
                                              double r, double mu) {
    if (is_last_line) {
        return MISS_DISTANCE;
    }

    double nu_diff = comov_nu - nu_line;

    /*
     * Numerical stability: if line is extremely close, treat as d=0
     * This prevents issues with floating-point precision
     */
    if (fabs(nu_diff / nu) < CLOSE_LINE_THRESHOLD) {
        nu_diff = 0.0;
    }

    if (nu_diff < 0.0) {
        /*
         * Line is bluer than current CMF frequency
         * This shouldn't happen in normal operation - would mean
         * the packet has passed the line without noticing
         */
        return MISS_DISTANCE;  /* Or could raise error like Python version */
    }

    if (!ENABLE_FULL_RELATIVITY) {
        /* Partial relativity formula */
        return (nu_diff / nu) * C_SPEED_OF_LIGHT * time_explosion;
    } else {
        /* Full relativity formula from TARDIS */
        double nu_r = nu_line / nu;
        double ct = C_SPEED_OF_LIGHT * time_explosion;

        /*
         * Full relativistic distance formula:
         * d = -μr + (ct - ν_r² × √(ct² - (1 + r²(1-μ²)(1 + 1/ν_r²)))) / (1 + ν_r²)
         */
        double nu_r_sq = nu_r * nu_r;
        double sin_sq = 1.0 - mu * mu;
        double discriminant = ct * ct - (1.0 + r * r * sin_sq * (1.0 + 1.0 / nu_r_sq));

        if (discriminant < 0.0) {
            return MISS_DISTANCE;
        }

        double distance = -mu * r + (ct - nu_r_sq * sqrt(discriminant)) / (1.0 + nu_r_sq);
        return distance;
    }
}

/**
 * calculate_distance_electron: Distance to Thomson scattering event [cm]
 *
 * Given a sampled optical depth τ_event, find distance where
 * τ_electron = τ_event
 *
 * τ = n_e × σ_T × d
 * d = τ / (n_e × σ_T)
 *
 * @param electron_density  Free electron density n_e [cm⁻³]
 * @param tau_event         Sampled optical depth (dimensionless)
 * @return Distance [cm]
 */
static inline double calculate_distance_electron(double electron_density, double tau_event) {
    return tau_event / (electron_density * SIGMA_THOMSON);
}

/* ============================================================================
 * OPACITY CALCULATIONS
 * ============================================================================ */

/**
 * calculate_tau_electron: Thomson optical depth for given path [dimensionless]
 *
 * τ = n_e × σ_T × d
 *
 * @param electron_density  Free electron density [cm⁻³]
 * @param distance          Path length [cm]
 * @return Optical depth (dimensionless)
 */
static inline double calculate_tau_electron(double electron_density, double distance) {
    return electron_density * SIGMA_THOMSON * distance;
}

/**
 * calc_packet_energy: Energy at distance along path (for estimators)
 *
 * The packet energy in the comoving frame changes along the path
 * due to the changing Doppler factor.
 *
 * @param energy      Packet energy at current position [erg]
 * @param r           Current position [cm]
 * @param mu          Direction cosine
 * @param distance    Distance along path [cm]
 * @param time_explosion Time [s]
 * @return Comoving energy at (r + distance)
 */
static inline double calc_packet_energy(double energy, double r, double mu,
                                         double distance, double time_explosion) {
    /*
     * Doppler factor at position (r + d):
     * For small d, linearize: D(r+d) ≈ D(r) - d×∂D/∂r
     *
     * TARDIS uses: D = 1 - (d + μr)/(ct)
     */
    double doppler_factor = 1.0 - (distance + mu * r) / (time_explosion * C_SPEED_OF_LIGHT);
    return energy * doppler_factor;
}

#endif /* PHYSICS_KERNELS_H */
