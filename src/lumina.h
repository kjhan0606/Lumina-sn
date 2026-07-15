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
#include <locale.h>   /* A6: setlocale(LC_NUMERIC,"C") for ko_KR-safe sscanf */

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
#define C_FF_OPACITY      3.6926e8         /* free-free opacity coefficient (CGS) */
#define KPKT_FB_NEDGE     16               /* [FB-MULTI] max recomb continua per shell for per-edge k-packet fb sampling */

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

/* [EVENT-LOG] archival packet event record (little-endian). NOTE: the fields
 * below sum to 20 bytes (4*4 scalar words + 4 uint8), not the 16 quoted in the
 * design note — 4 four-byte members cannot coexist with 4 one-byte members in
 * 16 bytes. record_size is therefore sizeof(EventRec)=20 and read_events.py
 * mirrors this exact layout. etype codes:
 *   1=line absorption (macro-atom activation)   5=k-packet free-bound  (type -3)
 *   2=line emission (macro-atom bb exit / MA-cap)6=escape
 *   3=bf continuum absorption                    7=electron scatter (opt-in)
 *   4=k-packet free-free (type -2)               8=bf-absorption re-emission (legacy) */
typedef struct {
    unsigned int  pkt_id;    /* packet index */
    int           line_id;   /* line index, or -1 for continuum / escape */
    float         nu_comov;  /* comoving frequency of the event [Hz] */
    float         energy;    /* packet comoving energy weight */
    unsigned char etype;     /* event type code (see above) */
    unsigned char shell;     /* current shell id (0-255) */
    unsigned char iter;      /* co-evolve iteration */
    unsigned char pad;       /* reserved (0) */
} EventRec;

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
    double *line_source_S;    /* CMF: NLTE two-level line source fn [n_lines*n_shells]; <=0 => use fallback */
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
    /* k-packet thermal pool (collisional macro-atom, LUMINA_KPACKET). Built in
     * compute_transition_probabilities; NULL when disabled. */
    double *p_kpacket;                   /* [n_macro_levels * n_shells] P(coll. deactivation→k-packet) */
    double *kpacket_cdf;                 /* [n_shells * n_macro_levels] per-shell cumulative re-excitation dist */
    double *p_kpacket_ff;                /* [n_shells] P(free-free continuum) once a k-packet forms (Path A); else coll-exc re-excite */
    double *p_kpacket_fb;                /* [n_shells] P(free-bound continuum | k-packet) — UV recombination-edge escape */
    double *kpacket_fb_nu;              /* [n_shells] representative recombination edge frequency [Hz] for fb emission */
    /* [FB-MULTI] per-continuum k-packet free-bound edge tables (LUMINA_KPKT_FB_MULTI).
     * Built in compute_transition_probabilities; NULL unless the gate is on. Each
     * shell owns a length-KPKT_FB_NEDGE slice; edge j is the recombination continuum
     * of recombining ion (Z,stage) with edge freq kpacket_fb_edge_nu, selectable via
     * the normalized cumulative fb-energy weight kpacket_fb_edge_cdf. */
    double *kpacket_fb_edge_nu;         /* [n_shells * KPKT_FB_NEDGE] per-continuum recomb edge freq [Hz] */
    double *kpacket_fb_edge_cdf;        /* [n_shells * KPKT_FB_NEDGE] normalized cumulative fb-energy weight */
    int    *kpacket_fb_edge_zstage;     /* [n_shells * KPKT_FB_NEDGE] Z*100+stage of the recombining ion (diagnostic) */
    int    *kpacket_fb_edge_count;      /* [n_shells] number of valid edges (<=KPKT_FB_NEDGE) */
    /* MC-estimator macro-atom (THEN_MC): per-line Sobolev j_blue estimator of
     * J_bar at each line, accumulated from real MC packet crossings, replacing
     * the frozen binned J in the internal-up rate B_lu*J_bar (faithful Lucy-2002
     * /TARDIS macro-atom estimator). NULL unless cmfgen_then_mc. */
    double *jbar_line;                   /* [n_lines * n_shells] normalized J_bar at each line */
    int    *jbar_count;                  /* [n_lines * n_shells] resonance-crossing count (undersample guard) */
    int     use_jbar_line;               /* 1 = use jbar_line for internal-up (iter>=1); 0 = binned-J seed */
    /* [IUP-JBLUE] ARTIS blue-wing J_blue: same per-line crossing tally + normalization
     * as jbar_line, consumed by the (B_lu - B_ul n_u/n_l)*beta*J_blue up-rate.
     * NULL unless LUMINA_IUP_JBLUE=1 (co-evolve transport). */
    double *jblue_line;                  /* [n_lines * n_shells] normalized blue-wing J_blue or NULL */
    /* P7 Stage-II: DETERMINISTIC line-resolved J_bar_l = Int phi_l J_nu dnu from the
     * fine-grid CMF solve (NOT the MC estimator above; the validated cure for the
     * binned-J contrast-collapse, ladder gates 4c/5b). NULL until the producer fills
     * it; consumed by the bb up-rate only when LUMINA_CMF_LINERES_JBAR=1. */
    double *jbar_line_det;               /* [n_lines * n_shells] or NULL */
    /* Fine-ν LOCAL continuum mean intensity from the cmfgen_fine_jbar producer,
     * retained (instead of freed) so bound-free PHOTOIONIZATION rates can be
     * integrated on the fine grid — the binned J collapses at the UV bf edges in
     * the thin outer (Jth_wt→0) and under-ionizes it. NULL until the producer runs
     * with LUMINA_CMF_FINE_PHOTOION; consumed by coupled_photoion_rate_jnu. */
    double *jnu_fine;                    /* [n_shells * n_fine] erg/s/cm^2/Hz/sr */
    double *nu_fine;                     /* [n_fine] Hz (log-uniform) */
    int     n_fine;                      /* fine-grid bin count (0 = absent) */
    double  nu_lo_fine, dlognu_fine;     /* log-grid params for σ_bf interpolation */
    /* bf recombination-cascade channel (macro-atom, LUMINA_MACROATOM_BF).
     * Parallel recomb topology (CSR keyed by the SOURCE upper-ion global level),
     * built once. Increment 1 = INTERNALDOWNLOWER (cross-ion internal jump, no
     * photon). recomb_nu_edge/recomb_is_emit reserved for increment 2 (RADRECOMB
     * continuum emit, opcode -4). All NULL / n_recomb=0 when the gate is off =>
     * byte-identical baseline. */
    int    *recomb_block_refs;           /* [n_macro_levels+1] CSR offsets */
    int    *recomb_dest_level;           /* [n_recomb] global lower-ion target level j */
    double *recomb_nu_edge;              /* [n_recomb] (chi_ion-E_j)/h (increment 2) */
    int    *recomb_is_emit;              /* [n_recomb] 0=INTERNALDOWNLOWER (all 0 inc1) */
    double *recomb_prob;                 /* [n_recomb * n_shells] normalized CDF weights */
    int     n_recomb;                    /* total recomb entries */
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
    int     fe_scatter_mode;            /* 0=off, 1=Fe II two-level, 2=all Fe two-level */
    int    *line_atomic_number;         /* [n_lines] Z, borrowed pointer from AtomicData */
    int    *line_ion_number;            /* [n_lines] ion stage, borrowed from AtomicData */
} MCConfig;                             /* Phase 2 - Step 4 */

/* Phase 2 - Step 4: Plasma state for convergence */
typedef struct {
    int     n_shells;         /* Phase 2 - Step 4 */
    double *W;                /* Phase 2 - Step 4: [n_shells] dilution factor */
    double *T_rad;            /* Phase 2 - Step 4: [n_shells] radiation temp [K] */
    double *rho;              /* Phase 2 - Step 4: [n_shells] density [g/cm^3] */
    double *n_electron;       /* Task #072: [n_shells] self-consistent n_e */
    double  T_e_T_rad_ratio;  /* T_e/T_rad ratio for Saha equation (default 0.9) */
    double *T_e;              /* P6: [n_shells] per-shell electron temperature [K] */
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
    int    *level_super;              /* [n_levels] CMFGEN super-level idx (per-ion, 0-based); = level_num if no f_to_s */
    signed char *level_mult;          /* [n_levels] spin multiplicity 2S+1 (0=unknown). NULL unless
                                       * LUMINA_ALPHA_SPINGATE=1 (loaded from level_multiplicity.csv);
                                       * OFF-path stays NULL so heap layout is unchanged. */

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

    /* CMFGEN-baked photoionization cross-sections.
     * Pre-baked onto LUMINA's fixed bf opacity grid (NLTE_N_FREQ_BINS bins,
     * NLTE_NU_MIN..NLTE_NU_MAX log-spaced). Loaded from cmfgen_sigma_bf.bin.
     * Layout: cmfgen_sigma_bf[level_idx * n_freq_bins + bin] in cm^2.
     * cmfgen_has_sigma[level_idx]==1 → use baked curve; ==0 → Kramers fallback. */
    int      cmfgen_loaded;           /* 1 if grid loaded; 0 → all Kramers */
    int      cmfgen_n_freq_bins;      /* must equal NLTE_N_FREQ_BINS */
    double   cmfgen_nu_min;
    double   cmfgen_nu_max;
    int     *cmfgen_has_sigma;        /* [n_levels] */
    double  *cmfgen_sigma_bf;         /* [n_levels * n_freq_bins] */
} AtomicData;

/* ============================================================ */
/* NLTE: Configuration and data structures                      */
/* ============================================================ */

#define NLTE_N_FREQ_BINS  1000
#define NLTE_NU_MIN       1.5e14    /* c / 20000 A */
#define NLTE_NU_MAX       3.0e16    /* c / 100 A */
#define NLTE_MAX_IONS     31        /* 14 II/III pairs + O I/II/III overlap (31 slots, 16 pairs; slot 29=O II shared) */
#define NLTE_PAIR_COUNT   16        /* #281: pair 15 = (slot 29 O II, slot 30 O III) overlaps pair 14 upper for CMFGEN triplet fidelity */

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
    int    *j_nu_count;                        /* [n_shells * n_freq_bins] MC per-bin packet tally */
    double *J_nu;                              /* [n_shells * n_freq_bins] normalized */

    /* Current iteration index (set by host before each nlte_solve_all call). */
    int    current_iter;

    /* CMFGEN super-level collapse (gated by LUMINA_SUPER_LEVELS).
     * The SE solve runs on super-levels (SL); full-level (FL) populations are
     * redistributed within each SL by Boltzmann at local T_e. When super_mode==0
     * every map is identity and the FL solve path is byte-identical to baseline. */
    int    super_mode;                              /* 1 => collapse to super-levels */
    int    n_super_total;                           /* total SL across all NLTE ions */
    int    nlte_ion_super_offset[NLTE_MAX_IONS + 1];/* cumulative SL offset per ion */
    int   *fl_to_super;                             /* [n_nlte_levels_total] FL nlte idx -> global SL solve idx */
    int   *super_anchor_global;                     /* [n_super_total] lowest-E FL global idx per SL */
    double *within_sl_frac;                         /* [n_nlte_levels_total * n_shells] Boltzmann fraction f_i of FL within its SL */

    /* ARTIS-style grey/LTE criterion: inward electron-scattering optical depth
     * per shell, recomputed each outer iteration. Optically-thick cells are
     * routed to LTE@T_e only during the first GREY_ITERS iterations; full NLTE
     * everywhere afterward (replaces the permanent density LTE_NCRIT zone). */
    double *shell_tau;                              /* [n_shells] inward tau_es */
} NLTEConfig;

/* ============================================================ */
/* Step 1.5: Charge Exchange Coupling                           */
/* ============================================================ */

#define CE_MAX_REACTIONS  20
#define CE_N_REACTIONS    17

typedef struct {
    int    Z_A, ion_A;       /* reactant A: A^(ion_A) */
    int    Z_B, ion_B;       /* reactant B: B^(ion_B) */
    double rate_coeff;       /* <σv> at T=10⁴K [cm³/s] */
    double alpha;            /* temp exponent: k(T) = rate_coeff * (T/1e4)^alpha */
    double delta_E_eV;       /* energy defect [eV], negative = exothermic forward */
} ChargeExchangeReaction;

/* ============================================================ */
/* Heavy.2 / Task #139: Dielectronic Recombination (DR)         */
/* Burgess-form fit: α_DR(T) = T^(-3/2) * Σ c_i * exp(-E_i / T)  */
/* T in K, c_i in cm³ s⁻¹ K^(3/2), E_i in K.                     */
/* Convention: ion_recomb = recombining (upper) ion stage,       */
/* i.e. for Fe III → Fe II we set Z=26, ion_recomb=2.            */
/* DR is added as ground(upper) → ground(lower) channel in the   */
/* NLTE rate matrix; cascades within the lower ion redistribute  */
/* via the bound-bound and Milne network already present.        */
/* ============================================================ */

#define DR_MAX_TERMS  10

typedef enum {
    DR_SOURCE_NONE       = 0,
    DR_SOURCE_BADNELL    = 1,   /* Strathclyde clist_K (AUTOSTRUCTURE) */
    DR_SOURCE_NORAD      = 2,   /* Nahar OSU R-matrix unified RR+DR    */
    DR_SOURCE_MAZZOTTA   = 3,   /* Mazzotta+1998 LS-coupling            */
    DR_SOURCE_AUTOSTRUCT = 4,   /* our AUTOSTRUCTURE self-compute       */
    DR_SOURCE_EST_ISOEL  = 5,   /* isoelectronic interpolation estimate */
    DR_SOURCE_CMFGEN     = 6    /* CMFGEN LTDR file (DIE*), summed+fit  */
} DRSource;

typedef struct {
    int       Z;                    /* atomic number              */
    int       ion_recomb;           /* recombining ion stage      */
    int       n_terms;              /* number of (c_i, E_i) pairs */
    double    c_i[DR_MAX_TERMS];    /* cm³/s · K^(3/2)            */
    double    E_i[DR_MAX_TERMS];    /* K                          */
    DRSource  source;
} DRCoefficient;

double dr_alpha_eval(const DRCoefficient *coef, double T_e);
const DRCoefficient* dr_lookup(int Z, int ion_recomb);

/* ============================================================ */
/* Gamma-ray energy deposition from 56Ni/56Co decay             */
/* ============================================================ */

typedef struct {
    int     n_shells;
    double *heating_rate;           /* [n_shells] erg/s/cm³ */
    double *nonthermal_ioniz_rate;  /* [n_shells] ionizations/s/cm³ */
} GammaDeposition;

/* ============================================================ */
/* Bound-free (photoionization) opacity                        */
/* ============================================================ */

typedef struct {
    int     enabled;
    int     n_freq_bins;    /* NLTE_N_FREQ_BINS (1000) */
    int     n_shells;
    double  nu_min;         /* NLTE_NU_MIN (1.5e14 Hz = c/20000A) */
    double  nu_max;         /* NLTE_NU_MAX (3.0e16 Hz = c/100A) */
    double  d_log_nu;       /* log(nu_max/nu_min) / n_freq_bins */
    double *chi_bf;         /* [n_shells * n_freq_bins] cm^-1 */
    double *eta_bf;         /* [n_shells * n_freq_bins] erg/s/cm^3/Hz/sr — bf(+ff)
                             * emissivity; = chi*B(T_e) thermal, or NLTE Milne
                             * source-function form under LUMINA_CMF_BF_MILNE */
    int    *activation_level; /* [n_shells * n_freq_bins] macro-atom level or -1 */
} BFOpacity;

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
void rescale_epoch(Geometry *geo, PlasmaState *plasma, double t_new);

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
    const int *line_atomic_number, const int *line_ion_number,
    int fe_scatter_mode,
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
    Estimators *est, MCConfig *config,                  /* Phase 3 - Step 1 */
    BFOpacity *bf, PlasmaState *plasma, RNG *rng        /* BF opacity support */
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
int load_cmfgen_sigma_bf(AtomicData *atom, const char *path);
void inject_topstage_continuum_levels(AtomicData *atom, OpacityState *opacity);
void free_atomic_data(AtomicData *atom);
void compute_plasma_state(AtomicData *atom, PlasmaState *plasma,
                          OpacityState *opacity, double time_explosion);
void compute_transition_probabilities(AtomicData *atom, PlasmaState *plasma,
                                       OpacityState *opacity,
                                       NLTEConfig *nlte,
                                       double damping_constant, int apply_damping);

void diag_macro_branch(AtomicData *atom, PlasmaState *plasma,
                       OpacityState *opacity, int diag_shell);

/* NLTE: Interpolate J_nu at a given frequency from the histogram */
double nlte_get_J_at_nu(NLTEConfig *nlte, int shell, double nu);

/* NLTE: Restricted NLTE rate equation solver */
/* Returns 1 if Mihalas-Lucy ion-lock should be active at current_iter.
 * Reads LUMINA_NLTE_ION_LOCK and LUMINA_NLTE_LOCK_START_ITER once.
 * Note: ion-lock has two effects coupled — matrix-row replacement (in solver)
 * and freeze-plasma-transport-only (in iter driver). To enable only the
 * post-solve per-ion rescale path without either, set LUMINA_NLTE_PER_ION_RESCALE=1. */
int  nlte_ion_lock_active(int current_iter);
int  nlte_per_ion_rescale_active(void);
int  nlte_skip_dead_pairs(void);
double nlte_inv_ceiling(void);

int  nlte_init(NLTEConfig *nlte, AtomicData *atom, OpacityState *opacity,
               int n_shells);
void nlte_free(NLTEConfig *nlte);
void nlte_normalize_j_nu(NLTEConfig *nlte, double time_simulation,
                          double *volume, int n_shells);
void nlte_apply_uv_jnu_cap(NLTEConfig *nlte, PlasmaState *plasma, int n_shells);
void nlte_solve_all(NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
                     OpacityState *opacity, double time_explosion,
                     int n_shells, GammaDeposition *gamma_dep);
/* Refresh within-super-level Boltzmann fractions at current T_e (call before
 * every solve on both CPU and GPU paths; see definition in lumina_plasma.c). */
void nlte_precompute_within_sl_frac(NLTEConfig *nlte, AtomicData *atom,
                                    PlasmaState *plasma, int n_shells);

/* Refresh per-line Sobolev optical depths + NLTE line source from the current
 * NLTE level populations (writes opacity->tau_sobolev, opacity->line_source_S).
 * Exposed so the pure-CMFGEN -> THEN_MC hand-off can push the CONVERGED line
 * opacity to the GPU (the GPU NLTE solve does not call this internally). */
void nlte_update_tau_sobolev(NLTEConfig *nlte, AtomicData *atom,
                              OpacityState *opacity,
                              double time_explosion, int n_shells);

/* Task #40 (A)+(B): photoionization rate lookup, populated by the GPU GEMM.
 * R_bf_table is col-major [L_phot_total × n_shells]; for a given (pair_idx,
 * lev_within_pair, shell):
 *   R_bf = R_bf_table[shell * L_phot_total + (phot_offset[pair_idx] + lev)]
 * Pass NULL to nlte_assemble_rate_matrix to use the inline CPU computation. */
typedef struct {
    const double *R_bf_table;
    const int    *phot_offset;   /* [n_pairs+1] */
    int           L_phot_total;  /* = phot_offset[n_pairs], stride of R_bf */
} NLTERateLookup;

/* NLTE: Assemble rate matrix (column-major A[N*N] + RHS b[N]) for GPU/CPU solve.
 * pair_idx is the index in the GPU pair list (0..n_pairs-1) when lookup != NULL;
 * ignored otherwise. */
void nlte_assemble_rate_matrix(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                int ion_idx_lo, int ion_idx_hi,
                                int shell, double time_explosion,
                                double *A_cm, double *b, int N,
                                GammaDeposition *gamma_dep,
                                const NLTERateLookup *lookup,
                                int pair_idx);

/* P7 / Task #138: tabulated ground-state photoionization σ_0 (Verner-CMFGEN).
 * Defined in lumina_plasma.c. Returns 0 → caller falls back to Kramers 7.91e-18/Z_eff^2. */
double get_bf_sigma0(int Z, int stage);

/* (2) 2026-05-14: NLTE pair conservation total. Returns Σ_{i=lo..hi} n_ion[i],
 * unless LUMINA_NLTE_NO_ML_LOCK=1, in which case returns element mass density
 * (n_element = abund·rho/m_amu) — drops the Mihalas-Lucy phi_neb soft lock. */
double nlte_pair_total_density(NLTEConfig *nlte, AtomicData *atom,
                               PlasmaState *plasma,
                               int Z_nl, int ion_idx_lo, int ion_idx_hi,
                               int shell);

/* Task #29 (Probe-B fix): write NLTE-solved ion split back into
 * atom->ion_number_density (per-pair, pair-total preserved) and rebuild bulk
 * tau_sobolev, so non-NLTE-tracked iron-peak line opacity uses the rate-solved
 * ionization instead of nebular phi_neb. Gated by LUMINA_NLTE_OPACITY_IONSTAGE=1.
 * Shared by CPU (lumina_plasma.c) and GPU (lumina_cuda.cu) NLTE solvers. */
void nlte_writeback_ion_stage(NLTEConfig *nlte, AtomicData *atom,
                              PlasmaState *plasma, OpacityState *opacity,
                              double time_explosion, int n_shells,
                              int pairs[][2], int n_pairs);

/* Task #40 (A)+(B): GPU NLTE photoionization rates via TF32 GEMM.
 * Pre-bakes K[ν, lev] in init; per call computes R_bf = K^T · J_nu. */
int  nlte_rates_gpu_init(NLTEConfig *nlte, AtomicData *atom, int n_shells);
int  nlte_rates_gpu_compute(NLTEConfig *nlte, NLTERateLookup *out_lookup);
void nlte_rates_gpu_free(void);
/* Register the producer's fine-ν field so nlte_rates_gpu_compute corrects R_bf over
 * the fine window (frequency-resolved photoionization). Pass jnu=NULL to disable. */
void nlte_rates_gpu_set_fine(const double *jnu, const double *nu, int n_fine,
                             double nu_lo, double dlognu, int n_shells, AtomicData *atom);

/* GPU port of the dominant per-line bound-bound radiative + collisional
 * assembly loop in nlte_assemble_rate_matrix (the 99.6% bottleneck). The CPU
 * assembles everything EXCEPT the bb loop (set via nlte_assemble_set_skip_bb);
 * the GPU kernel atomicAdds the bb+collisional contributions on top.
 * Gated by LUMINA_NLTE_ASSEMBLE_GPU (driver-side). See lumina_nlte_assemble.cu. */
int  nlte_assemble_gpu_init(NLTEConfig *nlte, AtomicData *atom,
                            OpacityState *opacity, int n_shells);
/* Returns 1 if the currently-active environment gates are within the GPU bb
 * path's supported domain (default binned J + dilute field + van Regemorter/
 * Axelrod collisions + coll-floor). Returns 0 if a sealed/experimental bb mode
 * is on (JBAR_POPS, MALI, JEQB, LINERES consumer, ...) -> caller stays on CPU. */
int  nlte_assemble_gpu_supported(void);
/* Re-upload per-iteration varying data (within_sl_frac, per-shell T_e/n_e/
 * T_rad/W, J_nu). Call once per CE iteration before the pair loop. */
void nlte_assemble_gpu_refresh(NLTEConfig *nlte, PlasmaState *plasma);
/* Add the bb+collisional contributions for one ion pair to the per-shell
 * column-major matrices held in h_matrices[n_shells*N*N] (which already hold
 * the CPU-assembled remainder). active[s]!=0 selects shells to fill (dead-pair
 * skip mirror). */
void nlte_assemble_bb_gpu_pair(double *h_matrices, int N, int n_shells,
                               int pair_lo, int pair_hi, int super_start,
                               int n_lo_super, const int *active);
void nlte_assemble_set_skip_bb(int v);
void nlte_assemble_gpu_free(void);

/* Gamma-ray deposition: 56Ni/56Co decay energy deposition */
void gamma_deposition_init(GammaDeposition *gd, int n_shells);
void compute_gamma_deposition(GammaDeposition *gd, AtomicData *atom,
                               PlasmaState *plasma, Geometry *geo);
/* Derive nonthermal_ioniz_rate from heating_rate + register for the freeze guard.
 * Call after loading an external deposition file (which sets only heating_rate). */
void gamma_deposition_compute_nonthermal(GammaDeposition *gd);
/* Register the fine-ν local field (from cmfgen_fine_jbar) so bf photoion rates
 * integrate on the fine grid (LUMINA_CMF_FINE_PHOTOION). */
void coupled_set_fine_jnu(const double *jnu, const double *nu, int n_fine,
                          double nu_lo, double dlognu, int n_shells);
void gamma_deposition_free(GammaDeposition *gd);

/* Line overlap correction: reduce tau_sobolev for overlapping UV lines */
void apply_overlap_corrections(AtomicData *atom, OpacityState *opacity,
                                PlasmaState *plasma);

/* Bound-free opacity: Kramers photoionization cross-section grid */
void bf_opacity_init(BFOpacity *bf, int n_shells);
void bf_opacity_free(BFOpacity *bf);
void compute_bf_opacity(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
                         int n_shells);
/* [BF-NLTE-POPS] Fix A: register the live NLTEConfig so compute_bf_opacity can
 * source chi_bf level populations from the NLTE solve (gate LUMINA_BF_NLTE_POPS).
 * NULL => dilute-Boltzmann fallback everywhere. */
void bf_set_nlte_pops(NLTEConfig *nlte);
double bf_get_chi(BFOpacity *bf, int shell, double nu);
double bf_get_eta(BFOpacity *bf, int shell, double nu);
/* Fine-ν bf opacity (sharp bf edges on the fine grid) for the CMFGEN-method producer.
 * chi_bf_fine_out is [n_shells * n_fine] row-major. Returns 0 ok / −1 fallback. */
int bf_gemm_compute_fine(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
        int n_shells, const double *nu_fine, int n_fine,
        double nu_min_bin, double dlognu_bin, double *chi_bf_fine_out);
/* Register bf + atom so cmfgen_fine_jbar can build the fine bf continuum opacity
 * (LUMINA_CMF_FINE_BF_OPAC). Pass NULL to disable. */
void cmfgen_fine_set_bf_atom(BFOpacity *bf, AtomicData *atom);
int    bf_get_activation_level(BFOpacity *bf, int shell, double nu);

/* Task #39: GPU bf opacity via cuBLAS GEMM (TF32 tensor cores).
 * Reformulates per-level loop as chi_bf[s,f] = n_level[s,l] @ sigma_bf[l,f].
 * Defined in lumina_cuda.cu; CPU build links against weak stubs.
 * Returns 0 on success, -1 on fallback. */
int  bf_gemm_init(AtomicData *atom, int n_shells);
int  bf_gemm_compute(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
                     int n_shells);
void bf_gemm_free(void);
void bf_absorption_event(RPacket *pkt, double time_explosion,
                          PlasmaState *plasma, OpacityState *opacity, RNG *rng);
double sample_planck_frequency(double T, RNG *rng);

/* P6: Self-consistent per-shell electron temperature */
void compute_electron_temperature(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                   double time_explosion, int n_shells,
                                   int self_consistent);

/* Task #20: real radiative-equilibrium T_e (heating = cooling), gated by
 * LUMINA_RADEQ_TE=1. Uses lagged NLTE pops + J_nu (operator-split). */
void compute_radiative_equilibrium_te(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                      NLTEConfig *nlte, AtomicData *atom,
                                      OpacityState *opacity,
                                      double time_explosion, int n_shells);

/* PATH-A / A2: per-shell COUPLED-NEWTON solve of {n_e, T_e} (simultaneous
 * linearization of radiative equilibrium + charge conservation), replacing the
 * operator-split RADEQ→ionization fixed point on non-frozen inner shells.
 * Gated by LUMINA_COUPLED_NEWTON=1; call after compute_radiative_equilibrium_te
 * + compute_plasma_state to overwrite their (T_e, n_e) with the coupled solution. */
void coupled_newton_solve_all(PlasmaState *plasma, GammaDeposition *gamma_dep,
                              NLTEConfig *nlte, AtomicData *atom,
                              OpacityState *opacity, Geometry *geo,
                              double time_explosion, int n_shells);

/* Option-2 integral radiative equilibrium: register the per-(shell,bin) CMFGEN
 * line opacity/source so the RADEQ/Newton T_e solve can add the radiative line
 * term 4π∫χ_line(J−S_l)dν (T_e-responsive) in place of the collisional bound-
 * bound cooling. Arrays are [n_shells*n_bins]; nu/dnu are [n_bins]. Call once
 * per CMFGEN outer iter after cmfgen_assemble + cmfgen_solve_J, before the
 * RADEQ/Newton solve. Pass chi_line=NULL to disable (default). Gated at use
 * time by LUMINA_RADEQ_LINE_RE=1. */
/* Physical two-level destruction probability eps=C_ul/(C_ul+A_ul*beta_esc)
 * for bb line `line` at (n_e,T_e,tau); -1 if the RADEQ table isn't built yet
 * (caller treats the line as fully thermal). Used by cmfgen_assemble. */
double radeq_line_eps_phys(int line, double n_e, double T_e, double tau);

/* A4 Stage-4': register the per-shell continuum-window color temperature of
 * the deterministic J (cmfgen_window_color) — frozen-tail T_e anchor, gated
 * by LUMINA_A4_TAIL_COLOR=1. */
void radeq_set_tail_color(const double *t_color, int n_shells);

/* A4 Stage-2.5: register the persisted tridiagonal Lambda response
 * (cs->tri_lo/tri_up/tri_r) for the global Newton's delta-J resolvent. */
void radeq_set_tri_response(const double *lo, const double *up,
                            const double *r, int n_shells, int n_bins);

void radeq_set_line_re_source(const double *chi_line, const double *chi_abs,
                              const double *chi_tot, const double *S_fixed,
                              const double *J, const double *nu,
                              const double *dnu, const double *lambda_star,
                              const double *T_e_assemble,
                              const double *chi_line_full,
                              const double *chi_line_cls,
                              int n_shells, int n_bins);

void plasma_set_photoion_mc_field(const double *J, double alpha, int nshells, int nfb,
                                  const int *counts);

/* [IUP-JBLUE] last-solve counters: internal-up lines that used the MC blue-wing
 * J_blue estimator vs those that fell back to the full-line J_line. */
void plasma_get_iup_jblue_counts(long *used, long *fallback);

/* [JBLUE-ANCHOR] internal normalization anchor: per-bucket count and log-mean
 * of log10(J_blue/J_line) for THIN (Sobolev beta>0.5) and THICK (beta<0.01)
 * lines, plus counts of ratios clamped outside [1e-3,1e3]. Thin log-mean ~0
 * = jblue estimator normalization anchored; systematic offset = bug in dex. */
void plasma_get_jblue_anchor(long *thin_n, double *thin_logmean,
                             long *thick_n, double *thick_logmean,
                             long *thin_clamp, long *thick_clamp);

/* P5: Formal integral spectrum (noise-free, p-z formalism) */
void compute_formal_integral_spectrum(
    Geometry *geo, PlasmaState *plasma, OpacityState *opacity,
    AtomicData *atom, NLTEConfig *nlte, double T_inner,
    Spectrum *spec_formal, int n_impact);

/* CMF (comoving-frame) formal solver — paper-method (Blondin+2013/CMFGEN)
 * line transfer with finite Gaussian Doppler profiles deposited on a fine
 * z-grid, then a formal solution of the RTE along impact-parameter tangent
 * rays using converged NLTE source functions. Single formal pass (no ALI).
 * Reduces to the Sobolev formal integral in the thin single-line limit;
 * the only new physics is line overlap (the suspected UV-forest divergence).
 *   n_zstep    : z-cells per ray (env LUMINA_CMF_NZ, default 2000)
 *   v_turb_cms : microturbulent velocity [cm/s] added to thermal width */
void compute_cmf_formal_spectrum(
    Geometry *geo, PlasmaState *plasma, OpacityState *opacity,
    AtomicData *atom, NLTEConfig *nlte, BFOpacity *bf, double T_inner,
    Spectrum *spec, int n_impact, int n_zstep, double v_turb_cms);

/* [MA-FATE] Macro-atom packet fate histogram (UV→? cascade diagnostic).
 * Bands: 0=UV (1700-3000 Å), 1=blue (3000-4500 Å),
 *        2=opt (4500-7000 Å), 3=other (NIR/far-UV).
 * Counts (entry_band, exit_band) pairs across all macro-atom interactions
 * (line-activated and BF-activated). Used to detect closed-loop UV trapping
 * vs Mazzali-Lucy fluorescence redistribution.
 *
 * macro_atom_fate_record() is thread-safe via OpenMP atomic; called from
 * macro_atom_event() in CPU transport. GPU transport uses a device-side
 * histogram (cuda_ma_fate_*) and aggregates into the same counters. */
#define MA_FATE_NBANDS 8
int  macro_atom_fate_band_from_nu(double nu_comov);
void macro_atom_fate_record(double entry_nu_comov, double exit_nu_comov);
void macro_atom_fate_reset(void);
void macro_atom_fate_print(const char *label);
void macro_atom_fate_add_counts(const unsigned long long add[MA_FATE_NBANDS * MA_FATE_NBANDS]);

/* [H3] Per-(Z, ion, entry_band, exit_band) attribution histogram.
 * Z-index map: 0=C(6) 1=O(8) 2=Mg(12) 3=Al(13) 4=Si(14) 5=S(16)
 *              6=Ca(20) 7=Sc(21) 8=Ti(22) 9=V(23) 10=Cr(24) 11=Mn(25)
 *              12=Fe(26) 13=Co(27) 14=Ni(28)
 * Ion: 0=I 1=II 2=III 3=IV (clamped). */
#define MA_FATE_NZ 15
#define MA_FATE_NION 4
#define MA_FATE_ZI_LEN (MA_FATE_NZ * MA_FATE_NION * MA_FATE_NBANDS * MA_FATE_NBANDS)
extern const int MA_FATE_Z_LIST[MA_FATE_NZ];
void macro_atom_fate_zi_add_counts(const unsigned long long add[MA_FATE_ZI_LEN]);
void macro_atom_fate_zi_reset(void);
void macro_atom_fate_zi_dump_csv(const char *path, const char *label);

/* GPU side: defined in lumina_cuda.cu, weak-stubbed in CPU build. */
void cuda_ma_fate_reset(void);
void cuda_ma_fate_download_and_aggregate(void);
void cuda_set_ma_fate_zi_enabled(int v);
void cuda_ma_fate_zi_reset(void);
void cuda_ma_fate_zi_download_and_aggregate(void);

/* [MA-CYCLE] Per-packet macro-atom internal cycle count histogram.
 * Each macro-atom interaction loops until it picks an emission transition;
 * ma_iter records how many hops a packet performs before emitting. Used to
 * detect closed-loop UV trapping (high cycle counts) vs prompt emission. */
#define MA_CYCLE_BINS 5001  /* 0..5000 inclusive (cap at 5000 in transport) */
void macro_atom_cycle_record(int n_cycles);
void macro_atom_cycle_reset(void);
void macro_atom_cycle_print(const char *label);
void macro_atom_cycle_add_counts(const unsigned long long add[MA_CYCLE_BINS]);

void cuda_ma_cycle_reset(void);
void cuda_ma_cycle_download_and_aggregate(void);

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

#endif /* LUMINA_H */
