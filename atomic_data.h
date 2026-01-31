/**
 * LUMINA-SN Atomic Data Structures
 * atomic_data.h - C representation of TARDIS HDF5 atomic database
 *
 * Data Format: TARDIS v2.0 atomic data (Kurucz CD23 + CHIANTI)
 * Reference: https://tardis-sn.github.io/tardis/io/configuration/components/atomic/atomic_data_description.html
 *
 * Design Principles:
 * ------------------
 * 1. FLAT ARRAYS with index offsets for fast O(1) lookups
 * 2. Separate storage for hot (frequently accessed) vs cold data
 * 3. Memory layout optimized for cache efficiency in MC transport
 * 4. All units in CGS (matching TARDIS internal convention)
 *
 * CGS Units:
 * ----------
 *   Energy:     erg (1 eV = 1.602e-12 erg)
 *   Wavelength: cm (1 Å = 1e-8 cm)
 *   Frequency:  Hz (ν = c / λ)
 *   Cross-section: cm²
 *   Rate:       s⁻¹
 */

#ifndef ATOMIC_DATA_H
#define ATOMIC_DATA_H

#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * PHYSICAL CONSTANTS (CGS)
 * ============================================================================ */

#define CONST_C         2.99792458e10     /* Speed of light [cm/s] */
#define CONST_H         6.62607015e-27    /* Planck constant [erg·s] */
#define CONST_K_B       1.380649e-16      /* Boltzmann constant [erg/K] */
#define CONST_M_E       9.1093837015e-28  /* Electron mass [g] */
#define CONST_E         4.80320425e-10    /* Elementary charge [esu] */
#define CONST_EV_TO_ERG 1.602176634e-12   /* eV to erg conversion */
#define CONST_ANGSTROM  1.0e-8            /* Angstrom to cm */

/* Maximum supported atomic number (H=1 through Zn=30) */
#define MAX_ATOMIC_NUMBER 30

/* ============================================================================
 * ELEMENT DATA (Basic Atomic Properties)
 * ============================================================================
 * From: atom_data table (30 elements)
 */

typedef struct {
    int8_t  atomic_number;    /* Z: 1 (H) to 30 (Zn) */
    char    symbol[4];        /* Element symbol: "H", "He", "Li", ... */
    char    name[16];         /* Full name: "Hydrogen", "Helium", ... */
    double  mass;             /* Atomic mass [amu] */
    double  mass_cgs;         /* Atomic mass [g] = mass × 1.66054e-24 */
} Element;

/* ============================================================================
 * IONIZATION DATA
 * ============================================================================
 * From: ionization_data table
 *
 * Indexed by (atomic_number, ion_number) where:
 *   ion_number = 0 → neutral atom
 *   ion_number = 1 → singly ionized
 *   ion_number = Z → fully ionized (bare nucleus)
 *
 * Example: H I (neutral H) has ion_number=0, ionization produces H II (ion_number=1)
 */

typedef struct {
    int8_t  atomic_number;
    int8_t  ion_number;       /* 0 = neutral, 1 = +1, ..., Z = fully ionized */
    double  ionization_energy; /* χ_ion [erg] - energy to remove electron */
    int32_t n_levels;         /* Number of energy levels for this ion */
    int32_t level_start_idx;  /* Index into levels array where this ion's levels begin */
    int32_t n_lines;          /* Number of lines for this ion */
    int32_t line_start_idx;   /* Index into lines array */
} Ion;

/* ============================================================================
 * ENERGY LEVEL DATA
 * ============================================================================
 * From: levels_data table (24,806 levels)
 *
 * Each level is uniquely identified by (Z, ion_number, level_number).
 * level_number = 0 is the ground state; energy = 0 for ground state.
 */

typedef struct {
    int8_t  atomic_number;
    int8_t  ion_number;
    int16_t level_number;     /* 0 = ground state, ascending in energy */
    double  energy;           /* Excitation energy [erg] relative to ground */
    int32_t g;                /* Statistical weight (degeneracy) = 2J + 1 */
    bool    metastable;       /* True if level is metastable */
} Level;

/* ============================================================================
 * SPECTRAL LINE DATA
 * ============================================================================
 * From: lines_data table (271,741 lines)
 *
 * Each line is identified by (Z, ion_number, level_lower, level_upper).
 * Contains all Einstein coefficients and oscillator strengths.
 */

typedef struct {
    int64_t line_id;          /* Unique line identifier (global) */

    /* Quantum numbers */
    int8_t  atomic_number;
    int8_t  ion_number;
    int16_t level_number_lower;
    int16_t level_number_upper;

    /* Spectroscopic properties */
    double  wavelength;       /* λ [cm] (note: cm, not Angstrom!) */
    double  nu;               /* ν [Hz] = c / λ */

    /* Oscillator strengths */
    double  f_ul;             /* f (upper → lower) [dimensionless] */
    double  f_lu;             /* f (lower → upper) [dimensionless] */

    /* Einstein coefficients */
    double  A_ul;             /* Spontaneous emission [s⁻¹] */
    double  B_ul;             /* Stimulated emission [cm² erg⁻¹ s⁻¹ Hz] */
    double  B_lu;             /* Absorption [cm² erg⁻¹ s⁻¹ Hz] */
} Line;

/* ============================================================================
 * MACRO-ATOM TRANSITION DATA (for Non-LTE)
 * ============================================================================
 * From: macro_atom_data table (815,223 transitions)
 *
 * Macro-atom formalism: levels undergo internal transitions that can be
 * radiative (photon emission/absorption) or collisional.
 *
 * transition_type:
 *   -1 = downward radiative (emission)
 *    0 = downward internal (non-radiative de-excitation)
 *   +1 = upward internal (collisional excitation)
 */

typedef struct {
    int8_t  atomic_number;
    int8_t  ion_number;
    int16_t source_level_number;
    int16_t destination_level_number;
    int8_t  transition_type;  /* -1, 0, or +1 */
    double  transition_probability;
    int64_t transition_line_id;  /* Link to Line (if radiative) */
} MacroAtomTransition;

/* ============================================================================
 * MACRO-ATOM LEVEL REFERENCES (Bookkeeping)
 * ============================================================================
 * From: macro_atom_references table
 *
 * For each level, counts of up/down transitions for fast probability calc.
 */

typedef struct {
    int8_t  atomic_number;
    int8_t  ion_number;
    int16_t source_level_number;
    int32_t count_down;       /* Number of downward transitions */
    int32_t count_up;         /* Number of upward transitions */
    int32_t count_total;      /* Total transitions from this level */
    int32_t transition_start_idx; /* Index into macro_atom_transitions array */
} MacroAtomReference;

/* ============================================================================
 * COLLISION DATA (Electron Impact Excitation)
 * ============================================================================
 * From: collisions_data table (CHIANTI format)
 *
 * Collision strengths Υ(T) for electron-impact excitation.
 * Used for detailed balance and non-LTE level populations.
 */

typedef struct {
    int64_t e_col_id;
    int8_t  atomic_number;
    int8_t  ion_number;
    int8_t  level_number_lower;
    int8_t  level_number_upper;
    double  energy_lower;
    double  energy_upper;
    double  g_lower;
    double  g_upper;
    double  delta_e;          /* ΔE = E_upper - E_lower [erg] */
    double  gf;               /* gf-value */
    int8_t  ttype;            /* Temperature dependence type */
    double  cups;             /* Collision strength at high T */
    double  btemp;            /* Characteristic temperature */
    double  bscups;           /* Scaled collision strength parameter */
} CollisionData;

/* ============================================================================
 * ZETA DATA (Recombination Ground State Fraction)
 * ============================================================================
 * From: zeta_data table
 *
 * ζ(T) = fraction of recombinations that go directly to ground state.
 * Tabulated at 20 temperature points from 2000K to 40000K.
 */

#define ZETA_N_TEMPERATURES 20

typedef struct {
    int8_t  atomic_number;
    int8_t  ion_charge;       /* Charge of recombining ion */
    double  zeta[ZETA_N_TEMPERATURES];  /* ζ(T) values */
} ZetaData;

/* Temperature grid for zeta data [K] */
extern const double ZETA_TEMPERATURES[ZETA_N_TEMPERATURES];

/* ============================================================================
 * MASTER ATOMIC DATA STRUCTURE
 * ============================================================================
 * Top-level container holding all atomic data with fast index lookups.
 */

typedef struct AtomicData {
    /* === Metadata === */
    char    format_version[8];  /* "2.0" */
    char    source_file[256];   /* HDF5 filename */

    /* === Element Data === */
    int32_t n_elements;
    Element *elements;          /* Array[MAX_ATOMIC_NUMBER] */

    /* === Ion Data === */
    int32_t n_ions;
    Ion     *ions;              /* Flattened array */

    /* Ion lookup: ion_index[Z][ion_number] → index in ions array */
    /* Use: ions[ion_index[Z][ion_num]] */
    int32_t ion_index[MAX_ATOMIC_NUMBER + 1][MAX_ATOMIC_NUMBER + 1];

    /* === Level Data === */
    int32_t n_levels;
    Level   *levels;            /* Flattened array, sorted by (Z, ion, level) */

    /* Level lookup: levels[ion->level_start_idx + level_number] */

    /* === Line Data === */
    int64_t n_lines;
    Line    *lines;             /* Flattened array */

    /* Line lookup by frequency: sorted_line_indices[i] gives index into lines
       sorted by ascending frequency for fast binary search */
    int64_t *sorted_line_indices;
    double  *sorted_line_nu;    /* Sorted frequencies for binary search */

    /* === Macro-Atom Data (optional, for non-LTE) === */
    int64_t n_macro_atom_transitions;
    MacroAtomTransition *macro_atom_transitions;

    int32_t n_macro_atom_references;
    MacroAtomReference *macro_atom_references;

    /* === Collision Data (optional) === */
    int32_t n_collisions;
    CollisionData *collisions;

    /* === Zeta Data (optional) === */
    int32_t n_zeta;
    ZetaData *zeta_data;

    /* === Downbranch Table (for line fluorescence) === */
    /* Pre-computed branching ratios for line downbranch (fluorescence cascade) */
    struct {
        int64_t *emission_line_start;   /* Index into emission_lines for each absorbing line */
        int64_t *emission_line_count;   /* Number of emission candidates for each line */
        int64_t *emission_lines;        /* Emission line IDs (flattened array) */
        double  *branching_probs;       /* Cumulative branching probabilities */
        int64_t total_emission_entries; /* Total entries in emission_lines/branching_probs */
        bool    initialized;
    } downbranch;

    /* === Memory management flags === */
    bool    owns_memory;        /* True if this struct allocated the arrays */

} AtomicData;

/* ============================================================================
 * FUNCTION DECLARATIONS: Loading & Initialization
 * ============================================================================ */

/**
 * atomic_data_load_hdf5: Load complete atomic database from HDF5 file
 *
 * @param filename  Path to HDF5 file (e.g., "kurucz_cd23_chianti_H_He.h5")
 * @param data      Pointer to AtomicData struct to populate
 * @return 0 on success, -1 on error
 */
int atomic_data_load_hdf5(const char *filename, AtomicData *data);

/**
 * atomic_data_free: Free all allocated memory
 */
void atomic_data_free(AtomicData *data);

/**
 * atomic_data_print_summary: Print summary statistics (for validation)
 */
void atomic_data_print_summary(const AtomicData *data);

/**
 * atomic_data_sanity_check: Verify loaded data against known values
 *
 * Checks:
 *   - H I ionization energy: 13.598 eV
 *   - He II ionization energy: 54.418 eV
 *   - Total number of lines
 *
 * @return 0 if all checks pass, number of failures otherwise
 */
int atomic_data_sanity_check(const AtomicData *data);

/* ============================================================================
 * FUNCTION DECLARATIONS: Fast Lookups
 * ============================================================================ */

/**
 * Get element by atomic number
 */
const Element *atomic_get_element(const AtomicData *data, int atomic_number);

/**
 * Get ion by (atomic_number, ion_number)
 */
const Ion *atomic_get_ion(const AtomicData *data, int atomic_number, int ion_number);

/**
 * Get level by (atomic_number, ion_number, level_number)
 */
const Level *atomic_get_level(const AtomicData *data, int atomic_number,
                               int ion_number, int level_number);

/**
 * Find lines within frequency range [nu_min, nu_max]
 *
 * @param data      Atomic data
 * @param nu_min    Minimum frequency [Hz]
 * @param nu_max    Maximum frequency [Hz]
 * @param indices   Output array of line indices (caller allocates)
 * @param max_lines Maximum lines to return
 * @return Number of lines found
 */
int64_t atomic_find_lines_in_range(const AtomicData *data,
                                    double nu_min, double nu_max,
                                    int64_t *indices, int64_t max_lines);

/**
 * Get ionization energy [erg]
 */
double atomic_get_ionization_energy(const AtomicData *data,
                                     int atomic_number, int ion_number);

/**
 * Get statistical weight for a level
 */
int atomic_get_g(const AtomicData *data, int atomic_number,
                 int ion_number, int level_number);

/* ============================================================================
 * DOWNBRANCH (LINE FLUORESCENCE) FUNCTIONS
 * ============================================================================
 * Pre-compute and use branching ratios for line fluorescence (downbranch).
 *
 * When a photon is absorbed by a line transition, the atom can de-excite through
 * various paths. The downbranch table pre-computes branching ratios:
 *   p_k = A_ul(k) / Σ_j A_ul(j)
 *
 * where the sum is over all emission lines from the upper level.
 */

/**
 * Build downbranch table for all lines
 *
 * For each line (absorbing transition), finds all emission candidates from
 * the upper level and computes cumulative branching probabilities.
 *
 * @param data  AtomicData structure (downbranch will be populated)
 * @return 0 on success, -1 on error
 */
int atomic_build_downbranch_table(AtomicData *data);

/**
 * Free downbranch table memory
 */
void atomic_free_downbranch_table(AtomicData *data);

/**
 * Sample emission line for downbranch (fluorescence)
 *
 * Given an absorbing line and a random number, selects the emission line
 * according to pre-computed branching probabilities.
 *
 * @param data      AtomicData with initialized downbranch table
 * @param line_id   Index of the absorbing line
 * @param xi        Random number in [0, 1)
 * @return Index of emission line, or line_id if no candidates (resonant scatter)
 */
int64_t atomic_sample_downbranch(const AtomicData *data, int64_t line_id, double xi);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Convert wavelength in Angstroms to frequency in Hz
 */
static inline double wavelength_angstrom_to_nu(double wavelength_A) {
    return CONST_C / (wavelength_A * CONST_ANGSTROM);
}

/**
 * Convert frequency to wavelength in Angstroms
 */
static inline double nu_to_wavelength_angstrom(double nu) {
    return CONST_C / nu / CONST_ANGSTROM;
}

/**
 * Convert energy in eV to erg
 */
static inline double ev_to_erg(double energy_eV) {
    return energy_eV * CONST_EV_TO_ERG;
}

/**
 * Convert energy in erg to eV
 */
static inline double erg_to_ev(double energy_erg) {
    return energy_erg / CONST_EV_TO_ERG;
}

#endif /* ATOMIC_DATA_H */
