/* lumina.h — Phase 2 - Step 1: Master header for LUMINA-SN
 * All structures match TARDIS Python exactly.
 * Every struct member has a TARDIS equivalent documented. */
#ifndef LUMINA_H
#define LUMINA_H

#include <stdio.h>    /* Phase 2 - Step 1 */
#include <stdlib.h>   /* Phase 2 - Step 1 */
#include <string.h>   /* Phase 2 - Step 1 */
#include <math.h>     /* Phase 2 - Step 1 */
#include <stdbool.h>  /* Phase 2 - Step 1 */
#include <stdint.h>   /* Phase 2 - Step 1 */
#include <float.h>    /* Phase 2 - Step 1 */

/* ============================================================ */
/* Phase 2 - Step 2: Physical constants (CGS, matching TARDIS)  */
/* ============================================================ */
#define C_SPEED_OF_LIGHT  2.99792458e10    /* Phase 2 - Step 2: cm/s */
#define SIGMA_THOMSON     6.6524616e-25    /* Phase 2 - Step 2: cm^2 */
#define H_PLANCK          6.62607015e-27   /* Phase 2 - Step 2: erg*s */
#define K_BOLTZMANN       1.380649e-16     /* Phase 2 - Step 2: erg/K */
#define SIGMA_SB          5.670374419e-5   /* Phase 2 - Step 2: erg/cm^2/s/K^4 */
#define M_PI_VAL          3.14159265358979323846 /* Phase 2 - Step 2 */
#define MISS_DISTANCE     1.0e99           /* Phase 2 - Step 2: line past end */
#define CLOSE_LINE_THRESHOLD 1.0e-14       /* Phase 2 - Step 2: relative freq tol (TARDIS) */

/* Task #072: Constants for plasma solver */
#define SOBOLEV_COEFF     2.6540281e-02    /* pi * e^2 / (m_e * c) in CGS */
#define EV_TO_ERG         1.602176634e-12  /* eV to erg conversion */
#define AMU               1.660539066e-24  /* atomic mass unit in g */
#define M_ELECTRON        9.1093837015e-28 /* electron mass in g */

/* Phase 2 - Step 2: TARDIS estimator constants (CGS) */
/* T_RADIATIVE = (pi^4 / (15 * 24 * zeta(5))) * (h/k_B) */
/* zeta(5) = 1.0369277551433699 */
/* h = 6.62607015e-27 erg*s, k_B = 1.380649e-16 erg/K */
/* h/k_B = 4.7992e-11 s*K */
/* pi^4/(15*24*zeta(5)) = 0.26087... */
/* T_RAD_CONST = 0.26087 * 4.7992e-11 = 1.2523e-11 K*s */
#define T_RADIATIVE_CONSTANT  1.2523374827e-11 /* Phase 2 - Step 2 */

/* ============================================================ */
/* Phase 2 - Step 3: Enums matching TARDIS                      */
/* ============================================================ */

/* Phase 2 - Step 3: Packet status (r_packet.py) */
typedef enum {
    PACKET_IN_PROCESS = 0,  /* Phase 2 - Step 3 */
    PACKET_EMITTED    = 1,  /* Phase 2 - Step 3 */
    PACKET_REABSORBED = 2   /* Phase 2 - Step 3 */
} PacketStatus;             /* Phase 2 - Step 3 */

/* Phase 2 - Step 3: Interaction types (r_packet_transport.py) */
typedef enum {
    INTERACTION_BOUNDARY    = 0, /* Phase 2 - Step 3 */
    INTERACTION_LINE        = 1, /* Phase 2 - Step 3 */
    INTERACTION_ESCATTERING = 2, /* Phase 2 - Step 3 */
    INTERACTION_CONTINUUM   = 3  /* Phase 2 - Step 3 */
} InteractionType;               /* Phase 2 - Step 3 */

/* Phase 2 - Step 3: Line interaction types (interaction_events.py) */
typedef enum {
    LINE_SCATTER    = 0, /* Phase 2 - Step 3: resonant scatter */
    LINE_DOWNBRANCH = 1, /* Phase 2 - Step 3: downbranch */
    LINE_MACROATOM  = 2  /* Phase 2 - Step 3: macro-atom */
} LineInteractionType;   /* Phase 2 - Step 3 */

/* Phase 2 - Step 3: Macro-atom transition types (macro_atom.py) */
typedef enum {
    MA_BB_EMISSION     = -1, /* Phase 2 - Step 3: bound-bound emission */
    MA_BF_EMISSION     = -2, /* Phase 2 - Step 3: bound-free emission */
    MA_FF_EMISSION     = -3, /* Phase 2 - Step 3: free-free emission */
    MA_ADIABATIC_COOL  = -4, /* Phase 2 - Step 3: adiabatic cooling */
    MA_BF_COOLING      = -5, /* Phase 2 - Step 3: bf cooling */
    MA_INTERNAL_DOWN   =  0, /* Phase 2 - Step 3: internal down */
    MA_INTERNAL_UP     =  1  /* Phase 2 - Step 3: internal up */
} MacroAtomTransitionType;   /* Phase 2 - Step 3 */

/* ============================================================ */
/* Phase 2 - Step 4: Data structures                            */
/* ============================================================ */

/* Phase 2 - Step 4: RPacket — matches TARDIS r_packet.py exactly */
typedef struct {
    double r;                 /* Phase 2 - Step 4: radial position [cm] */
    double mu;                /* Phase 2 - Step 4: cos(theta) direction */
    double nu;                /* Phase 2 - Step 4: frequency [Hz] (lab frame) */
    double energy;            /* Phase 2 - Step 4: packet energy [erg] */
    int    current_shell_id;  /* Phase 2 - Step 4: current shell index */
    int    next_line_id;      /* Phase 2 - Step 4: next line to interact with */
    PacketStatus status;      /* Phase 2 - Step 4: IN_PROCESS/EMITTED/REABSORBED */
    int    index;             /* Phase 2 - Step 4: packet index for RNG */
} RPacket;                    /* Phase 2 - Step 4 */

/* Phase 2 - Step 4: Radial 1D geometry — TARDIS NumbaRadial1DGeometry */
typedef struct {
    int     n_shells;         /* Phase 2 - Step 4: number of shells */
    double *r_inner;          /* Phase 2 - Step 4: [n_shells] inner radii [cm] */
    double *r_outer;          /* Phase 2 - Step 4: [n_shells] outer radii [cm] */
    double *v_inner;          /* Phase 2 - Step 4: [n_shells] inner velocities [cm/s] */
    double *v_outer;          /* Phase 2 - Step 4: [n_shells] outer velocities [cm/s] */
    double  time_explosion;   /* Phase 2 - Step 4: time since explosion [s] */
} Geometry;                   /* Phase 2 - Step 4 */

/* Phase 2 - Step 4: Opacity state — TARDIS OpacityState */
typedef struct {
    int     n_lines;          /* Phase 2 - Step 4: total number of lines */
    int     n_shells;         /* Phase 2 - Step 4: number of shells */
    double *line_list_nu;     /* Phase 2 - Step 4: [n_lines] sorted DESCENDING */
    double *tau_sobolev;      /* Phase 2 - Step 4: [n_lines * n_shells] row-major */
    double *electron_density; /* Phase 2 - Step 4: [n_shells] n_e [cm^-3] */
    double *t_electrons;      /* Phase 2 - Step 4: [n_shells] T_e [K] */

    /* Phase 2 - Step 4: Macro-atom data */
    int     n_macro_levels;              /* Phase 2 - Step 4: number of levels */
    int     n_macro_transitions;         /* Phase 2 - Step 4: total transitions */
    int    *macro_block_references;      /* Phase 2 - Step 4: [n_levels+1] */
    int    *transition_type;             /* Phase 2 - Step 4: [n_transitions] */
    int    *destination_level_id;        /* Phase 2 - Step 4: [n_transitions] */
    int    *transition_line_id;          /* Phase 2 - Step 4: [n_transitions] */
    double *transition_probabilities;    /* Phase 2 - Step 4: [n_transitions * n_shells] */
    int    *line2macro_level_upper;      /* Phase 2 - Step 4: [n_lines] */
} OpacityState;                          /* Phase 2 - Step 4 */

/* Phase 2 - Step 4: MC Estimators — TARDIS RadfieldMCEstimators */
typedef struct {
    int     n_shells;         /* Phase 2 - Step 4 */
    int     n_lines;          /* Phase 2 - Step 4 */
    double *j_estimator;      /* Phase 2 - Step 4: [n_shells] mean intensity */
    double *nu_bar_estimator; /* Phase 2 - Step 4: [n_shells] freq-weighted J */
    double *j_blue_estimator; /* Phase 2 - Step 4: [n_lines * n_shells] */
    double *Edotlu_estimator; /* Phase 2 - Step 4: [n_lines * n_shells] */

    /* NLTE: J_nu frequency histogram (CPU accumulation) */
    double *j_nu_estimator;   /* [n_shells * n_freq_bins] or NULL */
    int     nlte_n_freq_bins; /* 0 if NLTE disabled */
    double  nlte_nu_min;
    double  nlte_d_log_nu;
} Estimators;                 /* Phase 2 - Step 4 */

/* Phase 2 - Step 4: Monte Carlo configuration */
typedef struct {
    bool    enable_full_relativity;     /* Phase 2 - Step 4 */
    bool    disable_line_scattering;    /* Phase 2 - Step 4 */
    int     line_interaction_type;      /* Phase 2 - Step 4: 0=scatter,1=down,2=macro */
    int     n_packets;                  /* Phase 2 - Step 4 */
    int     n_iterations;               /* Phase 2 - Step 4 */
    int     hold_iterations;            /* Phase 2 - Step 4 */
    double  damping_constant;           /* Phase 2 - Step 4 */
    uint64_t seed;                      /* Phase 2 - Step 4 */
    double  T_inner;                    /* Phase 2 - Step 4: inner boundary temp [K] */
    double  luminosity_requested;       /* Phase 2 - Step 4: [erg/s] */
    bool    enable_nlte;                /* NLTE: enable restricted NLTE solver */
} MCConfig;                             /* Phase 2 - Step 4 */

/* Phase 2 - Step 4: Plasma state for convergence */
typedef struct {
    int     n_shells;         /* Phase 2 - Step 4 */
    double *W;                /* Phase 2 - Step 4: [n_shells] dilution factor */
    double *T_rad;            /* Phase 2 - Step 4: [n_shells] radiation temp [K] */
    double *rho;              /* Phase 2 - Step 4: [n_shells] density [g/cm^3] */
    double *n_electron;       /* Task #072: [n_shells] self-consistent n_e */
    double  T_e_T_rad_ratio;  /* T_e/T_rad ratio for Saha equation (default 0.9) */
} PlasmaState;                /* Phase 2 - Step 4 */

/* Task #072: Atomic data for plasma solver */
typedef struct {
    /* Per-line data (from line_list.csv) */
    int    *line_atomic_number;       /* [n_lines] Z (real, e.g. 14=Si) */
    int    *line_ion_number;          /* [n_lines] ion stage (0=neutral) */
    int    *line_level_lower;         /* [n_lines] lower level index */
    int    *line_level_upper;         /* [n_lines] upper level index */
    double *line_f_lu;                /* [n_lines] oscillator strength */
    double *line_wavelength_cm;       /* [n_lines] wavelength in cm */

    /* NLTE: Einstein coefficients and line frequencies */
    double *line_A_ul;                /* [n_lines] spontaneous emission rate [s^-1] */
    double *line_B_lu;                /* [n_lines] stimulated absorption [cm^2 Hz / erg] */
    double *line_B_ul;                /* [n_lines] stimulated emission [cm^2 Hz / erg] */
    double *line_nu;                  /* [n_lines] line frequency [Hz] */
    int     n_lines;                  /* number of lines (from line_list.csv) */

    /* Level data (from levels.csv) */
    int     n_levels;
    int    *level_Z;                  /* [n_levels] atomic number (real) */
    int    *level_ion;                /* [n_levels] ion number */
    int    *level_num;                /* [n_levels] level number */
    double *level_energy_eV;          /* [n_levels] energy in eV */
    int    *level_g;                  /* [n_levels] statistical weight */
    int    *level_metastable;         /* [n_levels] metastable flag */

    /* Ionization data (from ionization_energies.csv) */
    int     n_ionization;             /* total ionization entries */
    int    *ioniz_Z;                  /* [n_ionization] atomic number */
    int    *ioniz_ion;                /* [n_ionization] ion number */
    double *ioniz_energy_eV;          /* [n_ionization] chi in eV */

    /* Zeta factors (from zeta_data.npy + zeta_ions.csv + zeta_temps.csv) */
    int     n_zeta_ions;
    int    *zeta_Z;                   /* [n_zeta_ions] */
    int    *zeta_ion;                 /* [n_zeta_ions] */
    double *zeta_data;                /* [n_zeta_ions * n_zeta_temps] */
    double *zeta_temps;               /* [n_zeta_temps] */
    int     n_zeta_temps;

    /* Element data (from atom_masses.csv + abundances.csv) */
    int     n_elements;               /* 8 */
    int    *element_Z;                /* [n_elements] */
    double *element_mass_amu;         /* [n_elements] */
    double *abundances;               /* [n_elements * n_shells] mass fractions */

    /* Lookup: ion_offset[elem_idx] = first ion index for element elem_idx */
    /* n_ion_pops_per_elem[elem_idx] = number of ion populations */
    int     n_ion_pops;               /* total ion populations (153) */
    int    *ion_pop_Z;                /* [n_ion_pops] atomic number */
    int    *ion_pop_stage;            /* [n_ion_pops] ion stage (0..Z) */
    int    *elem_ion_offset;          /* [n_elements+1] offset into ion_pop arrays */

    /* Level lookup: level_offset[ion_pop_idx] = first level index for that ion */
    int    *level_offset;             /* [n_ion_pops+1] */

    /* Per-shell computed quantities */
    double *ion_number_density;       /* [n_ion_pops * n_shells] */
    double *partition_functions;      /* [n_ion_pops * n_shells] */
} AtomicData;

/* ============================================================ */
/* NLTE: Configuration and data structures                      */
/* ============================================================ */

#define NLTE_N_FREQ_BINS  1000
#define NLTE_NU_MIN       1.5e14    /* c / 20000 A */
#define NLTE_NU_MAX       3.0e16    /* c / 100 A */
#define NLTE_MAX_IONS     12        /* Si,Ca,Fe,S,Co,Ni  II/III (6 pairs) */

typedef struct {
    int    enabled;
    int    n_freq_bins;
    double nu_min, nu_max, d_log_nu;

    /* Target ions: (Z, ion_stage) pairs */
    int    n_nlte_ions;                        /* 8 */
    int    nlte_Z[NLTE_MAX_IONS];              /* atomic numbers */
    int    nlte_ion[NLTE_MAX_IONS];            /* ion stages */

    /* Level index maps */
    int    n_nlte_levels_total;                /* ~2017 */
    int    nlte_ion_level_offset[NLTE_MAX_IONS + 1]; /* cumulative offset */
    int   *nlte_to_global_level;               /* [n_nlte_levels_total] -> global level idx */
    int   *global_to_nlte_level;               /* [n_levels] -> NLTE level idx or -1 */
    int   *nlte_line_map;                      /* [n_lines] -> NLTE ion idx or -1 */

    /* Results */
    double *nlte_level_populations;            /* [n_nlte_levels_total * n_shells] */
    double *j_nu_estimator;                    /* [n_shells * n_freq_bins] raw MC */
    double *J_nu;                              /* [n_shells * n_freq_bins] normalized */
} NLTEConfig;

/* ============================================================ */
/* Step 1.5: Charge Exchange Coupling                           */
/* ============================================================ */

#define CE_MAX_REACTIONS  4
#define CE_N_REACTIONS    4

typedef struct {
    int    Z_A, ion_A;       /* reactant A: A^(ion_A) */
    int    Z_B, ion_B;       /* reactant B: B^(ion_B) */
    double rate_coeff;       /* <σv> at T=10⁴K [cm³/s] */
    double alpha;            /* temp exponent: k(T) = rate_coeff * (T/1e4)^alpha */
    double delta_E_eV;       /* energy defect [eV], negative = exothermic forward */
} ChargeExchangeReaction;

/* Phase 2 - Step 4: Spectrum output */
typedef struct {
    int     n_bins;           /* Phase 2 - Step 4 */
    double  lambda_min;       /* Phase 2 - Step 4: [Angstrom] */
    double  lambda_max;       /* Phase 2 - Step 4: [Angstrom] */
    double *flux;             /* Phase 2 - Step 4: [n_bins] luminosity density */
    double *wavelength;       /* Phase 2 - Step 4: [n_bins] bin centers [Angstrom] */
} Spectrum;                   /* Phase 2 - Step 4 */

/* ============================================================ */
/* Phase 2 - Step 5: RNG (xoshiro256** for speed + quality)     */
/* ============================================================ */

typedef struct {
    uint64_t s[4]; /* Phase 2 - Step 5: xoshiro256** state */
} RNG;             /* Phase 2 - Step 5 */

#ifdef __cplusplus   /* Phase 6 - Step 9: extern C guard for NVCC */
extern "C" {         /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

/* Phase 2 - Step 5: Initialize RNG from seed */
void rng_init(RNG *rng, uint64_t seed); /* Phase 2 - Step 5 */
/* Phase 2 - Step 5: Return uniform [0, 1) */
double rng_uniform(RNG *rng);           /* Phase 2 - Step 5 */
/* Phase 2 - Step 5: Return uniform [-1, 1] for mu */
double rng_mu(RNG *rng);               /* Phase 2 - Step 5 */

/* ============================================================ */
/* Phase 2 - Step 6: Function declarations                      */
/* ============================================================ */

/* Phase 2 - Step 6: Atomic data loading */
int load_tardis_reference_data(
    const char *ref_dir,   /* Phase 2 - Step 6 */
    Geometry   *geo,       /* Phase 2 - Step 6 */
    OpacityState *opacity, /* Phase 2 - Step 6 */
    PlasmaState  *plasma,  /* Phase 2 - Step 6 */
    MCConfig     *config   /* Phase 2 - Step 6 */
);

/* Phase 2 - Step 6: Memory management */
void free_geometry(Geometry *geo);          /* Phase 2 - Step 6 */
void free_opacity_state(OpacityState *op);  /* Phase 2 - Step 6 */
void free_estimators(Estimators *est);      /* Phase 2 - Step 6 */
void free_plasma_state(PlasmaState *ps);    /* Phase 2 - Step 6 */
void free_spectrum(Spectrum *spec);         /* Phase 2 - Step 6 */

/* Phase 2 - Step 6: Estimator management */
Estimators *create_estimators(int n_shells, int n_lines); /* Phase 2 - Step 6 */
void reset_estimators(Estimators *est);                   /* Phase 2 - Step 6 */

/* Phase 2 - Step 6: Spectrum management */
Spectrum *create_spectrum(double lambda_min, double lambda_max, int n_bins); /* Phase 2 - Step 6 */
void reset_spectrum(Spectrum *spec);                                        /* Phase 2 - Step 6 */

/* Phase 3 - Step 1: Transport functions */
void calculate_distance_boundary(
    double r, double mu, double r_inner, double r_outer,  /* Phase 3 - Step 1 */
    double *out_distance, int *out_delta_shell             /* Phase 3 - Step 1 */
);

double calculate_distance_line(
    double comov_nu, double nu_lab, int is_last_line,    /* Phase 3 - Step 1 */
    double nu_line, double time_explosion                /* Phase 3 - Step 1 */
);

double calculate_distance_electron(
    double electron_density, double tau_event /* Phase 3 - Step 1 */
);

double get_doppler_factor(
    double r, double mu, double time_explosion /* Phase 3 - Step 1 */
);

double get_inverse_doppler_factor(
    double r, double mu, double time_explosion /* Phase 3 - Step 1 */
);

void trace_packet(
    RPacket *pkt, Geometry *geo, OpacityState *opacity,  /* Phase 3 - Step 1 */
    Estimators *est, double chi_continuum,               /* Phase 3 - Step 1 */
    bool disable_line_scattering, RNG *rng,              /* Phase 3 - Step 1 */
    double *out_distance, InteractionType *out_type,     /* Phase 3 - Step 1 */
    int *out_delta_shell                                 /* Phase 3 - Step 1 */
);

void move_r_packet(
    RPacket *pkt, double distance, double time_explosion, /* Phase 3 - Step 1 */
    Estimators *est                                       /* Phase 3 - Step 1 */
);

void move_packet_across_shell_boundary(
    RPacket *pkt, int delta_shell, int n_shells /* Phase 3 - Step 1 */
);

void thomson_scatter(
    RPacket *pkt, double time_explosion, RNG *rng /* Phase 3 - Step 1 */
);

void line_scatter_event(
    RPacket *pkt, double time_explosion,              /* Phase 3 - Step 1 */
    int line_interaction_type, OpacityState *opacity,  /* Phase 3 - Step 1 */
    RNG *rng                                           /* Phase 3 - Step 1 */
);

void line_emission(
    RPacket *pkt, int emission_line_id,  /* Phase 3 - Step 1 */
    double time_explosion,               /* Phase 3 - Step 1 */
    OpacityState *opacity                /* Phase 3 - Step 1 */
);

void macro_atom_event(
    int dest_level_idx, RPacket *pkt,     /* Phase 3 - Step 1 */
    double time_explosion,                /* Phase 3 - Step 1 */
    OpacityState *opacity, RNG *rng       /* Phase 3 - Step 1 */
);

void macro_atom_interaction(
    int activation_level_id, int current_shell_id, /* Phase 3 - Step 1 */
    OpacityState *opacity, RNG *rng,               /* Phase 3 - Step 1 */
    int *out_transition_id,                        /* Phase 3 - Step 1 */
    int *out_transition_type                       /* Phase 3 - Step 1 */
);

void update_base_estimators(
    RPacket *pkt, double distance, Estimators *est, /* Phase 3 - Step 1 */
    double comov_nu, double comov_energy            /* Phase 3 - Step 1 */
);

void update_line_estimators(
    Estimators *est, RPacket *pkt, int cur_line_id, /* Phase 3 - Step 1 */
    double distance_trace, double time_explosion    /* Phase 3 - Step 1 */
);

void single_packet_loop(
    RPacket *pkt, Geometry *geo, OpacityState *opacity, /* Phase 3 - Step 1 */
    Estimators *est, MCConfig *config, RNG *rng         /* Phase 3 - Step 1 */
);

/* Phase 4 - Step 1: Plasma solver */
void solve_radiation_field(
    Estimators *est, double time_explosion,     /* Phase 4 - Step 1 */
    double time_simulation, double *volume,     /* Phase 4 - Step 1 */
    OpacityState *opacity, PlasmaState *plasma, /* Phase 4 - Step 1 */
    double damping_constant                     /* Task #072: TARDIS W/T_rad damping */
);

void update_t_inner(
    MCConfig *config, double L_emitted /* Task #072: TARDIS-style L_emitted */
);

/* Spectrum building: bins escaped packet luminosity into L_lambda [erg/s/cm] */
void bin_escaped_packet(Spectrum *spec, double nu, double energy);

/* Task #072: Atomic data loading and plasma solver */
int load_atomic_data(AtomicData *atom, const char *ref_dir, int n_shells);
void free_atomic_data(AtomicData *atom);
void compute_plasma_state(AtomicData *atom, PlasmaState *plasma,
                          OpacityState *opacity, double time_explosion);

/* NLTE: Restricted NLTE rate equation solver */
int  nlte_init(NLTEConfig *nlte, AtomicData *atom, OpacityState *opacity,
               int n_shells);
void nlte_free(NLTEConfig *nlte);
void nlte_normalize_j_nu(NLTEConfig *nlte, double time_simulation,
                          double *volume, int n_shells);
void nlte_solve_all(NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
                     OpacityState *opacity, double time_explosion,
                     int n_shells);

/* NLTE: Assemble rate matrix (column-major A[N*N] + RHS b[N]) for GPU/CPU solve */
void nlte_assemble_rate_matrix(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                int ion_idx_lo, int ion_idx_hi,
                                int shell, double time_explosion,
                                double *A_cm, double *b, int N);

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

#endif /* LUMINA_H */
