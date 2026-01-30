/**
 * LUMINA-SN Plasma Physics Module
 * plasma_physics.h - Saha-Boltzmann solver for ionization & level populations
 *
 * Physics Reference:
 *   Mihalas (1978) "Stellar Atmospheres", Chapter 5
 *   TARDIS: tardis/plasma/properties/ion_population.py
 *
 * Key Equations:
 * --------------
 * Saha Equation (ionization balance):
 *   n_{i+1} * n_e     2 U_{i+1}(T)   (2π m_e k T)^(3/2)
 *   ------------- = --------------- * ----------------- * exp(-χ_i / kT)
 *       n_i              U_i(T)            h³
 *
 * Boltzmann Equation (level populations):
 *   n_level     g_level
 *   ------- = --------- * exp(-E_level / kT)
 *    n_ion      U(T)
 *
 * Partition Function:
 *   U(T) = Σ g_i * exp(-E_i / kT)
 *
 * Charge Neutrality:
 *   n_e = Σ_{Z,i} i * n_{Z,i}
 */

#ifndef PLASMA_PHYSICS_H
#define PLASMA_PHYSICS_H

#include "atomic_data.h"
#include <stdbool.h>

/* ============================================================================
 * PHYSICAL CONSTANTS (CGS)
 * ============================================================================ */

#ifndef CONST_C
#define CONST_C         2.99792458e10     /* Speed of light [cm/s] */
#endif
#ifndef CONST_H
#define CONST_H         6.62607015e-27    /* Planck constant [erg·s] */
#endif
#ifndef CONST_K_B
#define CONST_K_B       1.380649e-16      /* Boltzmann constant [erg/K] */
#endif
#ifndef CONST_M_E
#define CONST_M_E       9.1093837015e-28  /* Electron mass [g] */
#endif
#ifndef CONST_M_P
#define CONST_M_P       1.67262192369e-24 /* Proton mass [g] */
#endif
#ifndef CONST_AMU
#define CONST_AMU       1.66053906660e-24 /* Atomic mass unit [g] */
#endif
#ifndef CONST_PI
#define CONST_PI        3.14159265358979323846
#endif

/* Saha constant: (2π m_e k)^(3/2) / h³ = 2.4146868e15 [cm⁻³ K^(-3/2)] */
#define SAHA_CONST      2.4146868042051944e15

/* Energy cutoff for partition function (levels with E > cutoff*kT are skipped) */
#define PARTITION_ENERGY_CUTOFF  50.0

/* ============================================================================
 * PLASMA STATE STRUCTURE
 * ============================================================================
 * Holds the computed plasma properties for a single cell/shell.
 */

typedef struct {
    /* Input conditions */
    double T;                   /* Temperature [K] */
    double rho;                 /* Mass density [g/cm³] */

    /* Derived quantities */
    double n_e;                 /* Electron number density [cm⁻³] */
    double n_ion_total;         /* Total ion number density [cm⁻³] */

    /* Per-element number densities */
    double n_element[MAX_ATOMIC_NUMBER + 1];  /* n_Z [cm⁻³] */

    /* Ion fractions: ion_fraction[Z][i] = n_{Z,i} / n_Z */
    double ion_fraction[MAX_ATOMIC_NUMBER + 1][MAX_ATOMIC_NUMBER + 2];

    /* Ion number densities: n_ion[Z][i] = n_{Z,i} [cm⁻³] */
    double n_ion[MAX_ATOMIC_NUMBER + 1][MAX_ATOMIC_NUMBER + 2];

    /* Partition functions: U[Z][i] = partition function for (Z, ion_stage) */
    double partition_function[MAX_ATOMIC_NUMBER + 1][MAX_ATOMIC_NUMBER + 2];

    /* Convergence info */
    int iterations;
    double convergence_error;
    bool converged;

} PlasmaState;

/* ============================================================================
 * ABUNDANCE STRUCTURE
 * ============================================================================
 * Mass fractions X_Z for each element (Σ X_Z = 1).
 */

typedef struct {
    double mass_fraction[MAX_ATOMIC_NUMBER + 1];  /* X_Z = ρ_Z / ρ */
    int n_elements;                                /* Number of elements with X > 0 */
    int elements[MAX_ATOMIC_NUMBER];               /* List of active elements */
} Abundances;

/* ============================================================================
 * PARTITION FUNCTION CALCULATION
 * ============================================================================ */

/**
 * Calculate partition function U(T) for a given ion
 *
 * U(T) = Σ g_i * exp(-E_i / kT)
 *
 * @param data      Atomic data
 * @param Z         Atomic number (1-30)
 * @param ion_stage Ion stage (0 = neutral, 1 = +1, ...)
 * @param T         Temperature [K]
 * @return Partition function (dimensionless)
 */
double calculate_partition_function(const AtomicData *data, int Z, int ion_stage, double T);

/**
 * Calculate partition functions for all ions at temperature T
 *
 * @param data   Atomic data
 * @param T      Temperature [K]
 * @param plasma PlasmaState to store results
 */
void calculate_all_partition_functions(const AtomicData *data, double T, PlasmaState *plasma);

/* ============================================================================
 * SAHA IONIZATION SOLVER
 * ============================================================================ */

/**
 * Calculate Saha factor Φ_{i,i+1}(T) for ionization from stage i to i+1
 *
 * Φ = (2 U_{i+1} / U_i) * (2π m_e k T / h²)^(3/2) * exp(-χ_i / kT)
 *   = (2 U_{i+1} / U_i) * SAHA_CONST * T^(3/2) * exp(-χ_i / kT)
 *
 * Saha equation: n_{i+1} * n_e / n_i = Φ
 *
 * @param data      Atomic data
 * @param Z         Atomic number
 * @param ion_stage Ion stage being ionized FROM (0 = neutral)
 * @param T         Temperature [K]
 * @param U_i       Partition function for stage i
 * @param U_i1      Partition function for stage i+1
 * @return Saha factor Φ [cm⁻³]
 */
double calculate_saha_factor(const AtomicData *data, int Z, int ion_stage,
                              double T, double U_i, double U_i1);

/**
 * Calculate ion fractions for element Z given n_e
 *
 * Uses Saha equation iteratively:
 *   n_1/n_0 = Φ_01 / n_e
 *   n_2/n_1 = Φ_12 / n_e
 *   ...
 *   Σ n_i = n_Z (total element density)
 *
 * @param data      Atomic data
 * @param Z         Atomic number
 * @param T         Temperature [K]
 * @param n_e       Electron density [cm⁻³]
 * @param n_element Total element number density [cm⁻³]
 * @param plasma    PlasmaState to store ion fractions
 */
void calculate_ion_fractions(const AtomicData *data, int Z, double T,
                              double n_e, double n_element, PlasmaState *plasma);

/**
 * Solve for self-consistent electron density using Newton-Raphson
 *
 * Iterates to find n_e satisfying charge neutrality:
 *   n_e = Σ_{Z,i} i * n_{Z,i}(n_e)
 *
 * @param data       Atomic data
 * @param abundances Element abundances (mass fractions)
 * @param T          Temperature [K]
 * @param rho        Mass density [g/cm³]
 * @param plasma     PlasmaState to store results
 * @return 0 on success, -1 on convergence failure
 */
int solve_ionization_balance(const AtomicData *data, const Abundances *abundances,
                              double T, double rho, PlasmaState *plasma);

/**
 * Solve ionization balance with NLTE dilution correction
 *
 * Same as solve_ionization_balance, but applies the dilution factor W
 * to the effective radiation temperature driving ionization.
 *
 * In a diluted radiation field (outer ejecta), ionization is reduced
 * because fewer photons are available to ionize atoms.
 *
 * @param data       Atomic data
 * @param abundances Element abundances (mass fractions)
 * @param T          Local electron temperature [K]
 * @param rho        Mass density [g/cm³]
 * @param W          Dilution factor (0 < W <= 0.5)
 * @param plasma     PlasmaState to store results
 * @return 0 on success, -1 on convergence failure
 */
int solve_ionization_balance_diluted(const AtomicData *data, const Abundances *abundances,
                                      double T, double rho, double W, PlasmaState *plasma);

/* ============================================================================
 * BOLTZMANN LEVEL POPULATIONS
 * ============================================================================ */

/**
 * Calculate level population fraction for a specific level
 *
 * n_level / n_ion = g_level * exp(-E_level / kT) / U(T)
 *
 * @param data      Atomic data
 * @param Z         Atomic number
 * @param ion_stage Ion stage
 * @param level     Level number (0 = ground state)
 * @param T         Temperature [K]
 * @param U         Partition function (pre-calculated)
 * @return Population fraction n_level / n_ion
 */
double calculate_level_population_fraction(const AtomicData *data, int Z,
                                            int ion_stage, int level,
                                            double T, double U);

/**
 * Calculate all level populations for a given ion
 *
 * @param data       Atomic data
 * @param Z          Atomic number
 * @param ion_stage  Ion stage
 * @param T          Temperature [K]
 * @param n_ion      Ion number density [cm⁻³]
 * @param n_levels   Output: array of level number densities [cm⁻³]
 * @param max_levels Maximum number of levels to calculate
 * @return Number of levels calculated
 */
int calculate_level_populations(const AtomicData *data, int Z, int ion_stage,
                                 double T, double n_ion,
                                 double *n_levels, int max_levels);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Initialize abundances with solar composition (Anders & Grevesse 1989)
 */
void abundances_set_solar(Abundances *ab);

/**
 * Initialize abundances with pure hydrogen
 */
void abundances_set_pure_hydrogen(Abundances *ab);

/**
 * Initialize abundances with H/He mixture
 *
 * @param ab        Abundances struct
 * @param X_H       Hydrogen mass fraction
 * @param X_He      Helium mass fraction
 */
void abundances_set_h_he(Abundances *ab, double X_H, double X_He);

/**
 * Initialize abundances for Type Ia supernova (W7-like composition)
 *
 * W7 deflagration model composition (Nomoto et al. 1984):
 *   - Dominated by Ni-56 decay products (Fe, Co)
 *   - Intermediate mass elements: Si, S, Ca
 *   - Some unburned C/O at high velocities
 *
 * @param ab        Abundances struct
 */
void abundances_set_type_ia_w7(Abundances *ab);

/**
 * Initialize stratified abundances for Type Ia (velocity-dependent)
 *
 * Physical motivation (Nomoto et al. 1984, Iwamoto et al. 1999):
 *   - Outer layers (v > 15,000 km/s): Unburned C/O fuel
 *   - Intermediate layers (10,000-15,000 km/s): Si, S, Ca from explosive burning
 *   - Inner layers (v < 10,000 km/s): Fe-group from complete Si-burning (Ni-56)
 *
 * This stratification is crucial for matching observed line profiles where
 * Si II forms at the photosphere (~10,000 km/s) and Fe features appear deeper.
 *
 * @param ab        Abundances struct to fill
 * @param velocity  Shell velocity [cm/s]
 */
void abundances_set_type_ia_stratified(Abundances *ab, double velocity);

/**
 * Set single element abundance
 */
void abundances_set_element(Abundances *ab, int Z, double mass_fraction);

/* ============================================================================
 * PARTITION FUNCTION CACHE
 * ============================================================================
 * Pre-computed partition functions U(T) on a temperature grid.
 * Interpolation is used for intermediate temperatures.
 */

#define PARTITION_CACHE_N_TEMPS   50    /* Number of temperature grid points */
#define PARTITION_CACHE_T_MIN   2000.0  /* Minimum temperature [K] */
#define PARTITION_CACHE_T_MAX  50000.0  /* Maximum temperature [K] */

typedef struct {
    double T_grid[PARTITION_CACHE_N_TEMPS];  /* Temperature grid */
    double U[MAX_ATOMIC_NUMBER + 1][MAX_ATOMIC_NUMBER + 2][PARTITION_CACHE_N_TEMPS];
    bool initialized;
} PartitionFunctionCache;

/**
 * Initialize partition function cache with pre-computed values
 *
 * @param cache  Cache structure to initialize
 * @param data   Atomic data
 */
void partition_cache_init(PartitionFunctionCache *cache, const AtomicData *data);

/**
 * Get partition function from cache (with linear interpolation)
 *
 * @param cache  Initialized partition cache
 * @param Z      Atomic number
 * @param ion    Ion stage
 * @param T      Temperature [K]
 * @return       Interpolated partition function
 */
double partition_cache_get(const PartitionFunctionCache *cache, int Z, int ion, double T);

/**
 * Calculate all partition functions using cache (faster than direct calculation)
 *
 * @param cache  Initialized partition cache
 * @param T      Temperature [K]
 * @param plasma PlasmaState to store results
 */
void calculate_all_partition_functions_cached(const PartitionFunctionCache *cache,
                                               double T, PlasmaState *plasma);

/**
 * Initialize PlasmaState structure
 */
void plasma_state_init(PlasmaState *plasma);

/**
 * Print plasma state summary
 */
void plasma_state_print(const PlasmaState *plasma, const AtomicData *data);

/**
 * Calculate mean molecular weight μ
 *
 * μ = ρ / (n_ion + n_e) / m_u
 */
double calculate_mean_molecular_weight(const PlasmaState *plasma);

/* ============================================================================
 * VALIDATION / TESTING
 * ============================================================================ */

/**
 * Run validation tests comparing with known analytical results
 *
 * @return Number of failed tests
 */
int plasma_physics_validation_tests(const AtomicData *data);

#endif /* PLASMA_PHYSICS_H */
