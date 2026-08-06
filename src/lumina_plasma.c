/* lumina_plasma.c — Phase 4: Plasma Solver and Convergence
 * Implements TARDIS mc_rad_field_solver.py for T_rad, W updates.
 * Implements T_inner convergence from escape fraction. */

#include "lumina.h" /* Phase 4 - Step 1 */
#include "bf_rate_jnu.h" /* A2-05 canonical-view bf photoionization rate */
#include "lumina_radeq_col_pairs.h" /* withParityO: CMFGEN-faithful all-pair COL cooling */
#include <assert.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus   /* Phase 6 - Step 9: extern C guard for NVCC */
extern "C" {         /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

/* Wave-3.2 D7 runtime counter hooks.  Implemented by the explicitly linked EW
 * translation unit; the hooks increment only while that lane is armed. */
extern void nlte_ew_note_save_restore_call(void);
extern void nlte_ew_note_per_ion_pin_call(void);
extern void nlte_ew_note_topstage_IV_call(void);
extern int nlte_ew_publish_runtime_counts(const NLTEConfig *nlte);

/* Production counter: unlike the frozen oracle observer this survives normal
 * runs and records every legacy GPU table rejected because the selected bf
 * radiation field is estimator/JEQB rather than the table's J field. */
static unsigned long g_nlte_bf_gpu_field_bypass_levels = 0;

unsigned long nlte_bf_gpu_field_bypass_count(void) {
    return g_nlte_bf_gpu_field_bypass_levels;
}

/* ascending double comparator for qsort (b_k partial-LTE rate-pin median) */
static int cmf_dcmp_pl(const void *a, const void *b) {
    double d = *(const double*)a - *(const double*)b;
    return (d > 0) - (d < 0);
}

#ifdef LUMINA_FROZEN_ORACLE
/* Compile-time-only Gate-B observer.  Normal production targets preprocess this
 * entire block (and every ORACLE_* call below) away. */
static PopulationStatus compute_partition_functions(AtomicData *atom,
                                                     PlasmaState *plasma,
                                                     int n_shells);
#define ORACLE_NION 8
#define ORACLE_NWAVE 2
typedef struct {
    int seen;
    int line;
    int lo_level, up_level;
    double lambda_A, score, j_line, jbar_raw, beta;
    long jbar_count;
    double r_up, r_stim, r_spont, c_lu, c_ul;
} OracleTopLine;
typedef struct {
    FILE *fp;
    int shell_label;
    int bf_rate_seen[ORACLE_NION];
    double gamma_num[ORACLE_NION], gamma_den[ORACLE_NION];
    double alpha_total[ORACLE_NION], alpha_spont[ORACLE_NION],
           alpha_stim[ORACLE_NION];
    double chi[ORACLE_NION][ORACLE_NWAVE];
    double eta[ORACLE_NION][ORACLE_NWAVE];
    double ff_chi[ORACLE_NWAVE], ff_eta[ORACLE_NWAVE];
    double ff_cooling_grid;
    int bf_bins_loaded, bf_bins_expected;
    long bf_estimator_consumptions, bf_fallback_consumptions;
    long bf_gpu_lookup_level_consumptions, bf_gpu_field_bypass_levels;
    int jbar_ion_complete[ORACLE_NION];
    int thermal_seen;
    double thermal_photoion, thermal_deposition, thermal_ff, thermal_bf;
    double thermal_bb_collisional, thermal_adiabatic, thermal_net;
    int ma_line_destruct_seen;
    double ma_line_destruct_heating;
    OracleTopLine top[ORACLE_NION];
} OracleTrace;
static OracleTrace g_oracle;
static const int g_oracle_Z[ORACLE_NION] = {14,14,16,16,26,26,26,27};
static const int g_oracle_stage[ORACLE_NION] = {1,2,1,2,1,2,3,2};
static const double g_oracle_wave_A[ORACLE_NWAVE] = {1000.0, 5000.0};

static int oracle_ion_slot(int Z, int stage) {
    for (int i = 0; i < ORACLE_NION; i++)
        if (g_oracle_Z[i] == Z && g_oracle_stage[i] == stage) return i;
    return -1;
}

static int oracle_wave_slot(const BFOpacity *bf, int b) {
    double nu = bf->nu_min * exp(((double)b + 0.5) * bf->d_log_nu);
    double lam = C_SPEED_OF_LIGHT / nu * 1.0e8;
    int best = -1;
    double best_d = 1.0e300;
    for (int k = 0; k < ORACLE_NWAVE; k++) {
        double d = fabs(log(lam / g_oracle_wave_A[k]));
        if (d < best_d) { best_d = d; best = k; }
    }
    /* Half a log-grid cell: exactly one production bin per requested lambda. */
    return (best >= 0 && best_d <= 0.51 * bf->d_log_nu) ? best : -1;
}

static void oracle_csv_value(FILE *fp, int shell, const char *category,
                             const char *quantity, int Z, int stage,
                             const char *transition, double nu, double value,
                             const char *unit, const char *fn) {
    fprintf(fp, "%d,%s,%s,%d,%d,%s,%.17e,%.17e,%s,%s,available,\n",
            shell, category, quantity, Z, stage,
            transition ? transition : "", nu, value, unit, fn);
}

static void oracle_csv_unavailable(FILE *fp, int shell, const char *category,
                                   const char *quantity, int Z, int stage,
                                   const char *reason) {
    fprintf(fp, "%d,%s,%s,%d,%d,,0,,,%s,unavailable,%s\n",
            shell, category, quantity, Z, stage,
            "not_computed_by_frozen_production_call", reason);
}

void lumina_oracle_trace_begin(FILE *fp, int shell_label) {
    memset(&g_oracle, 0, sizeof(g_oracle));
    g_oracle.fp = fp;
    g_oracle.shell_label = shell_label;
    for (int i = 0; i < ORACLE_NION; i++) g_oracle.top[i].score = -1.0;
}

void lumina_oracle_set_frozen_input_coverage(int bf_bins_loaded,
                                              int bf_bins_expected,
                                              const int *jbar_ion_complete,
                                              int n_jbar_ions) {
    if (!g_oracle.fp) return;
    g_oracle.bf_bins_loaded = bf_bins_loaded;
    g_oracle.bf_bins_expected = bf_bins_expected;
    for (int i = 0; i < ORACLE_NION; i++)
        g_oracle.jbar_ion_complete[i] =
            (jbar_ion_complete && i < n_jbar_ions) ? jbar_ion_complete[i] : 0;
}

void lumina_oracle_set_transport_thermal(double ma_line_destruct_heating,
                                         int available) {
    if (!g_oracle.fp) return;
    g_oracle.ma_line_destruct_seen = available ? 1 : 0;
    g_oracle.ma_line_destruct_heating = ma_line_destruct_heating;
}

void lumina_oracle_trace_end(void) {
    FILE *fp = g_oracle.fp;
    if (!fp) return;
    oracle_csv_value(fp, g_oracle.shell_label, "input",
                     "bf_rate_estimator_bins_loaded", 0, -1, "", 0.0,
                     (double)g_oracle.bf_bins_loaded, "count",
                     "input:lumina_c2_bfr_dump.csv");
    oracle_csv_value(fp, g_oracle.shell_label, "input",
                     "bf_rate_estimator_bins_expected", 0, -1, "", 0.0,
                     (double)g_oracle.bf_bins_expected, "count",
                     "NLTEConfig.n_freq_bins");
    oracle_csv_value(fp, g_oracle.shell_label, "input",
                     "bf_rate_estimator_positive_consumptions", 0, -1, "", 0.0,
                     (double)g_oracle.bf_estimator_consumptions, "count",
                     "nlte_assemble_rate_matrix");
    oracle_csv_value(fp, g_oracle.shell_label, "input",
                     "bf_rate_estimator_fallback_consumptions", 0, -1, "", 0.0,
                     (double)g_oracle.bf_fallback_consumptions, "count",
                     "nlte_assemble_rate_matrix");
    oracle_csv_value(fp, g_oracle.shell_label, "input",
                     "bf_gpu_lookup_level_consumptions", 0, -1, "", 0.0,
                     (double)g_oracle.bf_gpu_lookup_level_consumptions, "count",
                     "nlte_bf_field_source");
    oracle_csv_value(fp, g_oracle.shell_label, "input",
                     "bf_gpu_field_bypass_levels", 0, -1, "", 0.0,
                     (double)g_oracle.bf_gpu_field_bypass_levels, "count",
                     "nlte_bf_field_source");
    for (int i = 0; i < ORACLE_NION; i++)
        oracle_csv_value(fp, g_oracle.shell_label, "input",
                         "raw_jbar_ion_recorded", g_oracle_Z[i],
                         g_oracle_stage[i], "", 0.0,
                         (double)g_oracle.jbar_ion_complete[i],
                         "dimensionless", "input:lumina_jbar_dump.csv");
    for (int i = 0; i < ORACLE_NION; i++) {
        int Z = g_oracle_Z[i], st = g_oracle_stage[i];
        if (g_oracle.bf_rate_seen[i]) {
            double gamma = g_oracle.gamma_den[i] > 0.0
                         ? g_oracle.gamma_num[i] / g_oracle.gamma_den[i] : 0.0;
            oracle_csv_value(fp, g_oracle.shell_label, "bf", "Gamma_photoion_total",
                             Z, st, "", 0.0, gamma, "s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "bf", "alpha_recomb_total",
                             Z, st, "", 0.0, g_oracle.alpha_total[i], "cm^3_s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "bf", "alpha_recomb_spont",
                             Z, st, "", 0.0, g_oracle.alpha_spont[i], "cm^3_s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "bf", "alpha_recomb_stim",
                             Z, st, "", 0.0, g_oracle.alpha_stim[i], "cm^3_s^-1",
                             "nlte_assemble_rate_matrix");
        } else {
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bf",
                                   "Gamma_photoion_total", Z, st,
                                   "ion is not a lower member of an assembled NLTE pair");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bf",
                                   "alpha_recomb_total", Z, st,
                                   "ion is not a lower member of an assembled NLTE pair");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bf",
                                   "alpha_recomb_spont", Z, st,
                                   "ion is not a lower member of an assembled NLTE pair");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bf",
                                   "alpha_recomb_stim", Z, st,
                                   "ion is not a lower member of an assembled NLTE pair");
        }
        for (int w = 0; w < ORACLE_NWAVE; w++) {
            char q[64];
            double nu = C_SPEED_OF_LIGHT / (g_oracle_wave_A[w] * 1.0e-8);
            snprintf(q, sizeof(q), "chi_bf_at_%.0fA", g_oracle_wave_A[w]);
            oracle_csv_value(fp, g_oracle.shell_label, "bf", q, Z, st, "",
                             nu, g_oracle.chi[i][w], "cm^-1",
                             "compute_bf_opacity");
            snprintf(q, sizeof(q), "eta_bf_at_%.0fA", g_oracle_wave_A[w]);
            oracle_csv_value(fp, g_oracle.shell_label, "bf", q, Z, st, "",
                             nu, g_oracle.eta[i][w],
                             "erg_s^-1_cm^-3_Hz^-1_sr^-1",
                             "compute_bf_opacity");
        }
        OracleTopLine *t = &g_oracle.top[i];
        if (t->seen) {
            char tr[96];
            double nu = C_SPEED_OF_LIGHT / (t->lambda_A * 1.0e-8);
            snprintf(tr, sizeof(tr), "line%d_l%d_u%d_%.4fA",
                     t->line, t->lo_level, t->up_level, t->lambda_A);
            oracle_csv_value(fp, g_oracle.shell_label, "bb", "jbar_representative",
                             Z, st, tr, nu, t->j_line,
                             "erg_s^-1_cm^-2_Hz^-1_sr^-1",
                             g_oracle.jbar_ion_complete[i]
                                 ? "nlte_assemble_rate_matrix:frozen_raw_jbar"
                                 : "nlte_assemble_rate_matrix:C1_fallback_replay");
            if (g_oracle.jbar_ion_complete[i] && t->jbar_raw >= 0.0)
                oracle_csv_value(fp, g_oracle.shell_label, "bb", "jbar_input_raw",
                                 Z, st, tr, nu, t->jbar_raw,
                                 "erg_s^-1_cm^-2_Hz^-1_sr^-1",
                                 "nlte_assemble_rate_matrix");
            else
                oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                       "jbar_input_raw", Z, st,
                                       g_oracle.jbar_ion_complete[i]
                                           ? "representative line absent from the selected frozen raw-Jbar block"
                                           : "original raw-Jbar/tau was filtered before archival; representative J and rates are separately quantified by the production C1-fallback replay, but are not relabelled as the missing raw measurement");
            oracle_csv_value(fp, g_oracle.shell_label, "bb", "sobolev_beta",
                             Z, st, tr, nu, t->beta, "dimensionless",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "bb", "R_lu_radiative",
                             Z, st, tr, nu, t->r_up, "s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "bb", "R_ul_stimulated",
                             Z, st, tr, nu, t->r_stim, "s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "bb", "R_ul_spontaneous",
                             Z, st, tr, nu, t->r_spont, "s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "collisional", "C_lu",
                             Z, st, tr, nu, t->c_lu, "s^-1",
                             "nlte_assemble_rate_matrix");
            oracle_csv_value(fp, g_oracle.shell_label, "collisional", "C_ul",
                             Z, st, tr, nu, t->c_ul, "s^-1",
                             "nlte_assemble_rate_matrix");
        } else {
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                   "jbar_representative", Z, st,
                                   "no_positive_lower-level population flow selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                   "jbar_input_raw", Z, st,
                                   "no representative transition was selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                   "sobolev_beta", Z, st,
                                   "no representative transition was selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                   "R_lu_radiative", Z, st,
                                   "no_positive lower-level population flow selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                   "R_ul_stimulated", Z, st,
                                   "no_positive lower-level population flow selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "bb",
                                   "R_ul_spontaneous", Z, st,
                                   "no_positive lower-level population flow selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "collisional",
                                   "C_lu", Z, st,
                                   "no_positive lower-level population flow selected");
            oracle_csv_unavailable(fp, g_oracle.shell_label, "collisional",
                                   "C_ul", Z, st,
                                   "no_positive lower-level population flow selected");
        }
    }
    for (int w = 0; w < ORACLE_NWAVE; w++) {
        char q[64];
        double nu = C_SPEED_OF_LIGHT / (g_oracle_wave_A[w] * 1.0e-8);
        snprintf(q, sizeof(q), "chi_ff_at_%.0fA", g_oracle_wave_A[w]);
        oracle_csv_value(fp, g_oracle.shell_label, "ff", q, 0, -1, "", nu,
                         g_oracle.ff_chi[w], "cm^-1", "compute_bf_opacity");
        snprintf(q, sizeof(q), "eta_ff_at_%.0fA", g_oracle_wave_A[w]);
        oracle_csv_value(fp, g_oracle.shell_label, "ff", q, 0, -1, "", nu,
                         g_oracle.ff_eta[w], "erg_s^-1_cm^-3_Hz^-1_sr^-1",
                         "compute_bf_opacity");
    }
    oracle_csv_value(fp, g_oracle.shell_label, "ff", "cooling_ff_grid",
                     0, -1, "", 0.0, g_oracle.ff_cooling_grid,
                     "erg_s^-1_cm^-3", "compute_bf_opacity");
    if (g_oracle.thermal_seen) {
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "heating_photoion",
                         0, -1, "", 0.0, g_oracle.thermal_photoion,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "heating_deposition",
                         0, -1, "", 0.0, g_oracle.thermal_deposition,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        if (g_oracle.ma_line_destruct_seen)
            oracle_csv_value(fp, g_oracle.shell_label, "thermal",
                             "heating_MA_LINE_DESTRUCT", 0, -1, "", 0.0,
                             g_oracle.ma_line_destruct_heating,
                             "erg_s^-1_cm^-3",
                             "input:lumina_ma_line_destruct.csv");
        else
            oracle_csv_unavailable(fp, g_oracle.shell_label, "thermal",
                                   "heating_MA_LINE_DESTRUCT", 0, -1,
                                   "selected archived run has global terminal/destroyed event counts but no lumina_ma_line_destruct.csv; shell ownership and packet-energy/volume normalization are unavailable from this archive");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "cooling_ff",
                         0, -1, "", 0.0, g_oracle.thermal_ff,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "cooling_bf",
                         0, -1, "", 0.0, g_oracle.thermal_bf,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "cooling_bf_net",
                         0, -1, "", 0.0,
                         g_oracle.thermal_bf - g_oracle.thermal_photoion,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "cooling_bb_collisional",
                         0, -1, "", 0.0, g_oracle.thermal_bb_collisional,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "cooling_adiabatic",
                         0, -1, "", 0.0, g_oracle.thermal_adiabatic,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
        oracle_csv_value(fp, g_oracle.shell_label, "thermal", "thermal_net",
                         0, -1, "", 0.0, g_oracle.thermal_net,
                         "erg_s^-1_cm^-3", "compute_radiative_equilibrium_te/radeq_simul_all/simul_r1");
    } else {
        static const char *tq[] = {
            "heating_photoion", "heating_deposition", "heating_MA_LINE_DESTRUCT",
            "cooling_ff", "cooling_bf", "cooling_bf_net",
            "cooling_bb_collisional",
            "cooling_adiabatic", "thermal_net"
        };
        for (size_t i = 0; i < sizeof(tq) / sizeof(tq[0]); i++)
            oracle_csv_unavailable(fp, g_oracle.shell_label, "thermal", tq[i],
                                   0, -1,
                                   "production simultaneous thermal residual was not reached");
    }
    fflush(fp);
    g_oracle.fp = NULL;
}

void lumina_oracle_prepare_partitions(AtomicData *atom, PlasmaState *plasma,
                                      int n_shells) {
    (void)compute_partition_functions(atom, plasma, n_shells);
}
#endif

static inline double planck_bnu(double T, double nu);
/* Binned-J estimator: fit dilute Planck (T_rad,W) to the frequency-resolved
 * J_nu histogram instead of the nu_bar/j Wien moments. Returns 1 on success
 * (writes *T_out,*W_out), 0 if the histogram is unavailable/empty. */
static int fit_dilute_planck_binned_j(const Estimators *est, int shell,
                                      double volume, double time_simulation,
                                      double *T_out, double *W_out);

/* ============================================================ */
/* ARTIS-PARITY collisional rate network (Group A: A1-A6).      */
/* Master gate LUMINA_ARTIS_PARITY (default OFF => byte-identical). */
/* Faithful port of ARTIS macroatom.cc col_excitation_ratecoeff  */
/* / col_deexcitation_ratecoeff (constants from constants.h).    */
/* ============================================================ */
#define ARTIS_C_0            5.465e-11      /* constants.h C_0 */
#define ARTIS_VR_PREF        14.51039491    /* ARTIS van-Regemorter numeric prefactor */
#define ARTIS_H_IONPOT_ERG   (13.5979996 * EV_TO_ERG)  /* H ionization potential [erg] */
#define ARTIS_EULERGAMMA     0.5772156649015329
#define ARTIS_GBAR           0.2            /* effective Gaunt floor (permitted E1) */
#define ARTIS_COL_CONST      8.629e-6       /* effective collision-strength constant */
#define ARTIS_FORB_UPS       0.01           /* Axelrod g-scaled forbidden floor (A5) */
#define ARTIS_GAUNT_UCROSS   0.33421        /* u where 0.276 e^u(-gamma-ln u) == g_bar */

/* Master gate: LUMINA_ARTIS_PARITY=1 turns on the full ARTIS-consistent
 * collisional network. Default OFF => every parity branch is skipped and
 * behaviour is byte-identical to the champion. Env-cached (thread-safe: first
 * call wins; the driver reads it once at startup before the parallel region). */
int artis_parity_enabled(void) {
    static int init = 0, enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_ARTIS_PARITY");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Wave-3.2 R3: one source-of-truth for the bound-free radiation field AND the
 * permission to consume a precomputed GPU lookup.  Return values are private
 * to plasma/element_wide: 0=pref*J, 1=sigma*Gamma estimator (with per-bin
 * pref*J fallback), 2=pref*B_nu(T_e) JEQB falsifier.  A GPU lookup represents
 * the legacy J-field integral, so it is legal only for source 0.  When another
 * source is selected the helper explicitly reports the field mismatch and
 * forces the caller through the CPU integration; no caller may reconstruct
 * C2/JEQB/parity conditions independently. */
int nlte_bf_field_source(const NLTEConfig *nlte, double T_e, double nu,
                         double J_default, int gpu_lookup_available,
                         int *use_gpu_lookup, int *gpu_field_bypassed,
                         double *J_selected) {
    static int initialized = 0, c2_matrix_bf = 0, bf_jeqb = 0;
    if (!initialized) {
        const char *c2 = getenv("LUMINA_C2_MATRIX_BF");
        const char *jq = getenv("LUMINA_NLTE_BF_JEQB");
        c2_matrix_bf = (c2 && atoi(c2)) ? 1 : 0;
        bf_jeqb = (jq && atoi(jq)) ? 1 : 0;
        initialized = 1;
    }
    int source = bf_jeqb ? 2 :
        (((artis_parity_enabled() || c2_matrix_bf) && nlte &&
          nlte->bf_rate_estimator) ? 1 : 0);
    if (use_gpu_lookup)
        *use_gpu_lookup = gpu_lookup_available && source == 0;
    int bypassed = gpu_lookup_available && source != 0;
    if (gpu_field_bypassed) *gpu_field_bypassed = bypassed;
    if (bypassed) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        g_nlte_bf_gpu_field_bypass_levels++;
    }
    if (J_selected)
        *J_selected = (source == 2) ? planck_bnu(T_e, nu) : J_default;
    return source;
}

/* D3 coordinate 4: collisional bf has one policy in both owners. */
int nlte_bf_collisional_enabled(void) {
    return artis_parity_enabled();
}

/* A2-05 (SPEC_A2_05_V2): single choke point for the CPU bound-free
 * photoionization rate.  Legacy field sources 0 (pref*J) and 1 (sigma*Gamma
 * estimator) both collapse to a conservative integral of the canonical
 * RadiationField view; source 2 (JEQB falsifier) and the GPU lookup keep
 * their devices (R4 allowlist).  Only the J ownership moves: sigma stays the
 * per-level legacy-grid row (or the Kramers evaluation at legacy bin centers)
 * the site already owns, re-encoded as a bin-constant step tabulation so the
 * integrator reproduces the current sigma reading exactly.
 * Always fills *out; rc != 0 only on argument errors.  A non-VIEW_OK owner
 * reports STALE — never a substituted rate. */
static void nlte_bb_counter_inc(uint64_t *counter);
int nlte_bf_gamma_canonical(NLTEConfig *nlte, int shell,
                            const double *sigma_row, double sigma_0,
                            double nu_thresh, BfRateResult *out)
{
    static __thread double node_nu[2 * NLTE_N_FREQ_BINS];
    static __thread double node_sigma[2 * NLTE_N_FREQ_BINS];
    if (!out) return -1;
    out->gamma = 0.0;
    out->state = BF_RATE_STALE;
    out->w_miss = 0.0;
    out->sample_count = 0;
    if (!nlte || shell < 0 || !(nu_thresh > 0.0)) return -1;
    /* The NULL-view test also protects zero-initialized configs, where a
     * memset status would alias VIEW_OK (=0) without any published field. */
    if (nlte->radfield_view_status != RADIATION_FIELD_VIEW_OK ||
        !nlte->radfield_view.J_nu) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        nlte->bf_view_blocked_stale++;
        if (nlte->population_required_generation >
            nlte->population_committed_generation) {
#ifdef _OPENMP
#pragma omp critical(a2_07_population_error)
#endif
            {
                if (nlte->population_error_count == 0)
                    nlte->population_first_error = POP_BF_STALE;
                nlte->population_error_count++;
                population_counter_note(&nlte->population_counters,
                                        POP_BF_STALE);
            }
        }
        return 0;
    }
    int rc = bf_rate_gamma_legacy_grid(&nlte->radfield_view, (size_t)shell,
                                       nlte->n_freq_bins, nlte->nu_min,
                                       nlte->d_log_nu, sigma_row, sigma_0,
                                       nu_thresh, node_nu, node_sigma, out);
    if (rc != 0) return rc;
    if (out->state == BF_RATE_VALID || out->state == BF_RATE_EXACT_ZERO) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        nlte->bf_view_rate_terms++;
        nlte_bb_counter_inc(&nlte->population_counters.pop_bf_terms);
        if (out->state == BF_RATE_EXACT_ZERO)
            nlte_bb_counter_inc(
                &nlte->population_counters.pop_exact_zero_terms);
    } else if (out->state == BF_RATE_STALE) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        nlte->bf_view_blocked_stale++;
    } else if (out->state == BF_RATE_UNSAMPLED) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        nlte->bf_view_blocked_unsampled++;
    } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
        nlte->bf_view_blocked_out_of_grid++;
    }
    if (out->state != BF_RATE_VALID && out->state != BF_RATE_EXACT_ZERO &&
        nlte->population_required_generation >
        nlte->population_committed_generation) {
        PopulationStatus ps = out->state == BF_RATE_UNSAMPLED
                            ? POP_BF_UNSAMPLED
                            : out->state == BF_RATE_OUT_OF_GRID
                            ? POP_BF_OOG : POP_BF_STALE;
#ifdef _OPENMP
#pragma omp critical(a2_07_population_error)
#endif
        {
            if (nlte->population_error_count == 0)
                nlte->population_first_error = ps;
            nlte->population_error_count++;
            population_counter_note(&nlte->population_counters, ps);
        }
    }
    return 0;
}

/* A2-06: the only production CPU bound-bound radiation-field read.  A bad
 * checked view, cache miss, or unusable entry blocks only the radiative term;
 * it never supplies zero as a value and never falls back to a coarse/legacy
 * field.  Callers retain spontaneous A_ul and any separately computed
 * collisional term. */
static void nlte_bb_counter_inc(uint64_t *counter)
{
#ifdef _OPENMP
#pragma omp atomic update
#endif
    (*counter)++;
}

static int nlte_bb_jbar_canonical(NLTEConfig *nlte, int shell, int line,
                                  double *jbar)
{
    LineJbarValue value;
    if (!jbar || !nlte || shell < 0 || line < 0) return 0;
    *jbar = 0.0;

    if (nlte->line_view_status != LINE_JBAR_VIEW_OK ||
        !nlte->line_view.jbar) {
        PopulationStatus ps = POP_BB_STALE;
        switch (nlte->line_view_status) {
        case LINE_JBAR_VIEW_DISABLED:
            nlte_bb_counter_inc(&nlte->bb_view_blocked_disabled);
            break;
        case LINE_JBAR_VIEW_PROFILE:
            nlte_bb_counter_inc(&nlte->bb_view_blocked_profile);
            ps = POP_PROFILE_MISMATCH;
            break;
        case LINE_JBAR_VIEW_QHASH:
            nlte_bb_counter_inc(&nlte->bb_view_blocked_qhash);
            ps = POP_QUERY_HASH_MISMATCH;
            break;
        default:
            nlte_bb_counter_inc(&nlte->bb_view_blocked_stale);
            break;
        }
        if (nlte->population_required_generation >
            nlte->population_committed_generation) {
#ifdef _OPENMP
#pragma omp critical(a2_07_population_error)
#endif
            {
                if (nlte->population_error_count == 0)
                    nlte->population_first_error = ps;
                nlte->population_error_count++;
                population_counter_note(&nlte->population_counters, ps);
            }
        }
        return 0;
    }

    int rc = line_jbar_lookup(&nlte->line_view, (size_t)shell,
                              (uint64_t)line, &value);
    if (rc != 0) {
        nlte_bb_counter_inc(rc == -2 ? &nlte->bb_view_blocked_miss
                                     : &nlte->bb_view_blocked_stale);
        if (nlte->population_required_generation >
            nlte->population_committed_generation) {
            PopulationStatus ps = rc == -2 ? POP_BB_MISS : POP_BB_STALE;
#ifdef _OPENMP
#pragma omp critical(a2_07_population_error)
#endif
            {
                if (nlte->population_error_count == 0)
                    nlte->population_first_error = ps;
                nlte->population_error_count++;
                population_counter_note(&nlte->population_counters, ps);
            }
        }
        return 0;
    }
    if (value.validity == LINE_JBAR_VALID ||
        value.validity == LINE_JBAR_EXACT_ZERO) {
        *jbar = value.jbar;
        nlte_bb_counter_inc(&nlte->bb_view_rate_terms);
        nlte_bb_counter_inc(&nlte->population_counters.pop_bb_terms);
        if (value.validity == LINE_JBAR_EXACT_ZERO)
            nlte_bb_counter_inc(&nlte->population_counters.pop_exact_zero_terms);
        return 1;
    }
    if (value.validity == LINE_JBAR_UNSAMPLED)
        nlte_bb_counter_inc(&nlte->bb_view_blocked_unsampled);
    else if (value.validity == LINE_JBAR_OUT_OF_BB_DOMAIN)
        nlte_bb_counter_inc(&nlte->bb_view_blocked_oog);
    else
        nlte_bb_counter_inc(&nlte->bb_view_blocked_stale);
    if (nlte->population_required_generation >
        nlte->population_committed_generation) {
        PopulationStatus ps = value.validity == LINE_JBAR_UNSAMPLED
                            ? POP_BB_UNSAMPLED
                            : value.validity == LINE_JBAR_OUT_OF_BB_DOMAIN
                            ? POP_BB_OOG : POP_BB_STALE;
#ifdef _OPENMP
#pragma omp critical(a2_07_population_error)
#endif
        {
            if (nlte->population_error_count == 0)
                nlte->population_first_error = ps;
            nlte->population_error_count++;
            population_counter_note(&nlte->population_counters, ps);
        }
    }
    return 0;
}

/* Ownership status transferred from the current EW solve to the later
 * tau/source writeback.  Process-global because the public writeback signature
 * predates the EW lane; nlte_free releases the final retained pass. */
static int *g_ew_tau_authority = NULL;
static int g_ew_tau_authority_nshells = 0;

/* ============================================================================
 * [RATES-FIX] Master gate LUMINA_RATES_FIX (default OFF => every repair below
 * falls through to the pre-existing code path => byte-identical output).
 * Four by-construction defects of the IONIZATION rate machinery, localized by
 * the Saha known-answer harness (scripts/run_ioniz_saha_selftest.sh):
 *   F1  Boltzmann-cut asymmetry: the radeq Gph all-level loop dropped levels
 *       with E_l/kT >= 50, while the Milne alpha (frozenin_alpha_rr) keeps
 *       them. Gamma's per-level weight g e^{-E/kT} I and alpha's
 *       g e^{(chi-E)/kT} I differ only by the (level-independent) Saha factor,
 *       so cutting one side alone makes Gamma systematically LOW.
 *   F2  U_ion fallback read a stray g: when the upper ion carries no levels
 *       (n1==n0) the code read level_g[level_offset[ip_next]], which is the
 *       NEXT element's ground level (out of bounds for the last ion pop).
 *   F3  0 x inf -> NaN: Rbf was zeroed by the x>700 bin skip and then
 *       multiplied by exp(chi_l/kT)=inf. Fixed by FUSING the exponent into
 *       the integrand, exp((chi_l-h nu)/kT) <= 1 above threshold (the correct
 *       form of the Milne integral; also restores the finite low-T alpha the
 *       bin skip was throwing away).
 *   F5  Kramers sigma_0 fallback used Z_eff = Z-stage (the BOUND-electron
 *       count) instead of stage+1, the charge the ejected electron sees
 *       (documented convention; simul_ladder already uses stage+1).
 * F4 (missing DR autoionization partner) is deliberately NOT touched:
 * LUMINA_FROZENIN_DR=OFF is the consistent configuration.
 * ==========================================================================*/
static long g_rates_fix_n_emptyU = 0;   /* F2 firings; counted gate-INDEPENDENTLY */
static void rates_fix_report(void) {
    printf("[RATES-FIX] F2 empty-upper-ion U_ion fallbacks: %ld\n",
           g_rates_fix_n_emptyU);
    fflush(stdout);
}
static int rates_fix_enabled(void) {
    static int init = 0, on = 0;
    if (!init) {
        const char *e = getenv("LUMINA_RATES_FIX");
        on = (e && atoi(e) != 0) ? 1 : 0;
        init = 1;
        if (on) {
            printf("[RATES-FIX] F1 F2 F3 F5 active\n");
            fflush(stdout);
            atexit(rates_fix_report);
        }
    }
    return on;
}

/* THE shared ARTIS collision-coefficient helper (A6: ONE implementation for the
 * NLTE matrix, RADEQ cooling, and k-packet). Faithful port of ARTIS
 * macroatom.cc col_excitation_ratecoeff / col_deexcitation_ratecoeff.
 *   Inputs : T_e[K], n_e[cm^-3], dE=epsilon_trans[erg]>0, g_lo/g_up stat weights,
 *            f_lu oscillator strength, coll_str (effective collision strength;
 *            <0 => derive from f_lu/forbidden), forbidden flag (used iff coll_str<0).
 *   Outputs: *C_up   (multiply by LOWER-level pop -> up-rate   [s^-1])
 *            *C_down (multiply by UPPER-level pop -> down-rate  [s^-1]).
 * Detailed balance is exact per channel: C_up/C_down = (g_up/g_lo)*exp(-dE/kTe). */
static inline void artis_col_rates(double T_e, double n_e, double dE,
                                   double g_lo, double g_up, double f_lu,
                                   double coll_str, int forbidden,
                                   double *C_up, double *C_down) {
    *C_up = 0.0; *C_down = 0.0;
    if (!(T_e > 0.0) || !(n_e > 0.0) || !(dE > 0.0) || g_lo <= 0.0 || g_up <= 0.0)
        return;
    const double sqrtTe = sqrt(T_e);
    const double u = dE / (K_BOLTZMANN * T_e);   /* eoverkt */
    if (coll_str < 0.0) {
        if (!forbidden) {
            /* permitted E1: van Regemorter + energy-dependent Gaunt + Bethe
             * (H_ionpot/dE)^2 factor (macroatom.cc:757 / :718). gauntfac ==
             * max(g_bar, 0.276 e^u(-gamma-ln u)); the crossover is at u=0.33421. */
            const double gaunt = (u > ARTIS_GAUNT_UCROSS) ? ARTIS_GBAR
                              : 0.276 * exp(u) * (-ARTIS_EULERGAMMA - log(u));
            const double ry = ARTIS_H_IONPOT_ERG / dE;
            const double base = ARTIS_C_0 * ARTIS_VR_PREF * n_e * sqrtTe *
                                f_lu * ry * ry;
            *C_up   = base * u * exp(-u) * gaunt;
            *C_down = base * u * (g_lo / g_up) * gaunt;
        } else {
            /* forbidden M1/E2: Axelrod g-scaled floor, effective Upsilon =
             * 0.01*g_lo*g_up (macroatom.cc:723,765 -> A5). */
            *C_down = n_e * ARTIS_COL_CONST * ARTIS_FORB_UPS * g_lo / sqrtTe;
            *C_up   = n_e * ARTIS_COL_CONST * ARTIS_FORB_UPS * g_up * exp(-u) / sqrtTe;
        }
    } else {
        /* real effective collision strength (Osterbrock & Ferland p51; A3). */
        *C_down = n_e * ARTIS_COL_CONST * coll_str / g_up / sqrtTe;
        *C_up   = n_e * ARTIS_COL_CONST * coll_str * exp(-u) / g_lo / sqrtTe;
    }
}

/* Does this (Z, 0-based ion stage) have a REAL collision-strength table loaded
 * (Fe III Zhang OR a generic imported ion)? Used to suppress the per-line
 * vR/Axelrod proxy + the metastable floor so real data is not double-counted. */
static int ion_has_realcoldata(const AtomicData *atom, int Z, int ion0) {
    if (atom->feiii_col_loaded && Z == atom->feiii_col_Z && ion0 == atom->feiii_col_ion)
        return 1;
    for (int c = 0; c < atom->ncol_ions; c++)
        if (atom->col_ion_Z[c] == Z && atom->col_ion_stage[c] == ion0)
            return 1;
    return 0;
}

/* [MA-REAL-UPSILON] (Fix-P1) source registry + eval-time interpolator for wiring
 * REAL close-coupling Upsilon into the MACRO-ATOM transport collision rates. The
 * NLTE population matrix (plasma.c:12482/12570, nlte_assemble.cu) already consumes
 * the real Omega tables, but the MA transport rates (eps drain / k-packet CDF /
 * kp_deact / INTERNALUPSAME) fall back to the van-Regemorter/Axelrod proxy inside
 * artis_col_rates -> a self-consistency gap (pops solved with real Omega,
 * transport re-emitted with vR). ma_ru_upsilon() reproduces the NLTE-matrix lookup
 * EXACTLY: linear-in-T interpolation of Omega over the tabulated grid, clamped to
 * the ends. The eval-time C then equals the shared ARTIS real branch (artis_col_rates
 * coll_str>=0): C_down = n_e*8.629e-6*Ups/(g_up*sqrt(Te)); C_up with exp(-dE/kTe)/g_lo
 * — identical to plasma.c:12496-12497 and nlte_assemble.cu:195-196. Gated by
 * LUMINA_MA_REAL_UPSILON (default OFF => the sites keep the vR sentinel => byte-
 * identical). NO Omega floor is applied on the MA side (real table only). */
typedef struct { int ntemp; const double *tgrid; const double *omega; } MaRuSrc;

static inline double ma_ru_upsilon(const MaRuSrc *S, int t, double T_e) {
    if (!S || t < 0 || S->ntemp < 2 || !S->tgrid || !S->omega || !(T_e > 0.0))
        return -1.0;
    int nt = S->ntemp;
    const double *tg = S->tgrid;
    int ti = 0;
    while (ti < nt - 2 && T_e > tg[ti + 1]) ti++;
    double frac = 0.0, den = tg[ti + 1] - tg[ti];
    if (den > 0.0) frac = (T_e - tg[ti]) / den;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    const double *om = &S->omega[(size_t)t * (size_t)nt];
    double ups = om[ti] + frac * (om[ti + 1] - om[ti]);
    return (ups > 0.0) ? ups : -1.0;
}

/* ==========================================================================
 * [OMEGA-CMFGEN]  CMFGEN's 3-tier collision strength   (LUMINA_OMEGA_CMFGEN)
 *
 * WHY: LUMINA_RADEQ_OMEGA_FLOOR=1 is an invented clamp (Upsilon>=1). Measured
 * on the parity line census it floors 88.16% of ALL bb lines (median coeff
 * amplification 253x) and, because it is applied AFTER the real close-coupling
 * substitution, it OVERWRITES 85.9% of the tabulated Upsilon values (those
 * with Upsilon<1) by a median factor 18.7. This gate replaces it with the
 * prescription CMFGEN itself uses, a faithful port of
 *   /gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/omega_gen_v3.f  L159-197
 * (cmfgen_sub.f:224,1660,1718 call OMEGA_GEN_V3 — v3, not v2):
 *
 *   (i)   OMEGA(I,J) tabulated in col_data  ->  used AS IS.  NO FLOOR.
 *   (ii)  f_lu > 1e-5   (v3 L162 "Now set Omega=0.1 when f < 1.0E-05")
 *           OMEGA = 47.972 * OMEGA_SCALE * GBAR * f_lu * g_lo / FL
 *         with FL = nu in 1e15 Hz, OMEGA_SCALE = 1.0 (verified: all 40 ion
 *         col_data files declare "1.0 !Scaling factor for OMEGA"), and
 *           ZION>=2 (ions)    : GBAR = max(0.2, 0.276 e^X E1(X))
 *           ZION==1 (neutrals): GBAR = 0.276 e^X E1(X)      (X<=14)
 *                               GBAR = 0.066(1+1.5/X)/sqrt(X)  (X>14)
 *         X = h nu / k T_e; ZION = ion0 + 1 (charge seen by the outer e-).
 *   (iii) f_lu <= 1e-5  ->  OMEGA = OMEGA_SET = 0.1.
 *
 * NB the CMFGEN array EIN_A(I,J) for I<J holds the OSCILLATOR STRENGTH, not
 * the Einstein A — proved by the v3 change note ("Omega=0.1 when f < 1.0E-05"
 * guarding EIN_A(I,J).LE.1.0E-05) and by the prefactor identity
 *   47.972 = (8 pi / sqrt 3) * nu_Ry[1e15 Hz] = 14.5104 * 3.28984 = 47.746,
 * i.e. tier (ii) IS standard van Regemorter Omega = 14.5 gbar g_lo f (Ry/dE).
 *
 * RESIDUALS (deliberate, reported):
 *  - SAME_N GBAR=0.7 (omega_gen_v3 G1=0.7 when the two levels share the
 *    principal quantum number) is NOT applied: Lumina's levels.csv carries no
 *    configuration label, so SAME_N cannot be evaluated. Impact is bounded:
 *    same-n pairs have small dE => small X => 0.276 e^X E1(X) normally already
 *    exceeds 0.7 and the floor does not bind.
 *  - Tier (i) interpolation is linear-in-T (ma_ru_upsilon), the convention
 *    every other Lumina consumer of these tables uses; CMFGEN interpolates
 *    log-log. Changing it here alone would desync this path from the dedicated
 *    col_data NLTE pass, so it is left as a separate item.
 *  - OMEGA_SET is taken as the 0.1 constant. 38 of the 40 imported ions declare
 *    0.1; O II and O III declare 0.01 (per-ion OMEGA_SET is not carried by the
 *    .bin format).  Override with LUMINA_OMEGA_CMFGEN_SET.
 * Default OFF => omega_cmfgen_enabled()==0 => every call site falls through.
 * ======================================================================== */
static int find_ion_pop_idx(AtomicData *atom, int Z, int ion_stage);

#define OMCM_VR_PREF    47.972      /* omega_gen_v3.f L191                    */
#define OMCM_GBAR_ION   0.2         /* G1, ZION>=2, different-n               */
#define OMCM_FMIN       1.0e-5      /* omega_gen_v3.f L162                    */
#define OMCM_SET_DEF    0.1         /* OMEGA_SET (col_data header)            */
#define OMCM_HZ15       1.0e15

/* EXP(X)*E1(X), exact port of CMFGEN subs/ex_e1x_fun.f (Abramowitz & Stegun
 * p231; |err| < 2e-7 for X<=1, < 2e-8 for X>1). */
static double omcm_ex_e1x(double x) {
    static const double W[6] = {-0.57721566,  0.99999193, -0.24991055,
                                 0.05519968, -0.00976004,  0.00107857};
    static const double A[4] = {0.2677737343, 8.6347608925,
                                18.0590169730, 8.5733287401};
    static const double B[4] = {3.9584969228, 21.0996530827,
                                25.6329561486, 9.5733223454};
    if (!(x > 0.0)) return 0.0;
    if (x <= 1.0) {
        double p = W[5];
        for (int i = 4; i >= 0; i--) p = p * x + W[i];
        return exp(x) * (p - log(x));
    }
    double num = 1.0, den = 1.0;
    for (int i = 3; i >= 0; i--) { num = num * x + A[i]; den = den * x + B[i]; }
    return num / den / x;
}

/* GBAR of omega_gen_v3.f L169-186. zion = ionic charge = ion0+1. */
static double omcm_gbar(double x, int zion) {
    if (zion <= 1) {                       /* Auer & Mihalas 1973 ApJ 184,151 */
        if (x <= 14.0) return 0.276 * omcm_ex_e1x(x);
        return 0.066 * (1.0 + 1.5 / x) / sqrt(x);
    }
    double g2 = 0.276 * omcm_ex_e1x(x);    /* Mihalas 2nd ed p133             */
    return (OMCM_GBAR_ION > g2) ? OMCM_GBAR_ION : g2;
}

int omega_cmfgen_enabled(void) {
    static int init = 0, on = 0;
    if (!init) {
        const char *e = getenv("LUMINA_OMEGA_CMFGEN");
        on = (e && atoi(e) != 0) ? 1 : 0;
        init = 1;
    }
    return on;
}

static double omcm_oset(void) {
    static int init = 0; static double v = OMCM_SET_DEF;
    if (!init) { const char *e = getenv("LUMINA_OMEGA_CMFGEN_SET");
                 if (e) { double t = atof(e); if (t > 0.0) v = t; } init = 1; }
    return v;
}

/* Per-line map into the loaded tables. src: -1 = feiii_col, >=0 = col_ion slot,
 * OMCM_NOSRC = this line has no tabulated entry. Armed once, single-threaded,
 * then read-only (safe inside the OMP assembler / radeq bake). */
#define OMCM_NOSRC (-2)
static int   g_omcm_armed    = 0;
static int  *g_omcm_line_src = NULL;
static int  *g_omcm_line_t   = NULL;
static double *g_omcm_line_dE  = NULL;  /* [n_lines] |E_up-E_lo| erg (0 = bad) */
static double *g_omcm_line_glo = NULL;  /* [n_lines] g of the LOWER level      */
static int   g_omcm_nlines   = 0;
static long  g_omcm_n_tab = 0, g_omcm_n_vr = 0, g_omcm_n_set = 0;
static int   g_omcm_nsrc = 0;

/* Tier (ii)/(iii): no tabulated entry for this pair. */
static inline double omcm_fallback(double f_lu, double g_lo, double dE_erg,
                                   double T_e, int ion0, int *tier) {
    double fl = dE_erg / (H_PLANCK * OMCM_HZ15);          /* nu in 1e15 Hz */
    if (!(f_lu > OMCM_FMIN) || !(fl > 0.0) || !(T_e > 0.0) || !(g_lo > 0.0)) {
        if (tier) *tier = 3;
        return omcm_oset();
    }
    if (tier) *tier = 2;
    return OMCM_VR_PREF * omcm_gbar(dE_erg / (K_BOLTZMANN * T_e), ion0 + 1) *
           f_lu * g_lo / fl;
}

double omega_cmfgen_line(const AtomicData *atom, int line, double T_e, int *tier) {
    if (g_omcm_armed && line >= 0 && line < g_omcm_nlines) {
        int s = g_omcm_line_src[line], t = g_omcm_line_t[line];
        if (s != OMCM_NOSRC && t >= 0) {
            MaRuSrc S;
            if (s < 0) { S.ntemp = atom->feiii_col_n_temp;
                         S.tgrid = atom->feiii_col_tgrid;
                         S.omega = atom->feiii_col_omega; }
            else       { S.ntemp = atom->col_ion_n_temp[s];
                         S.tgrid = atom->col_ion_tgrid[s];
                         S.omega = atom->col_ion_omega[s]; }
            double ups = ma_ru_upsilon(&S, t, T_e);
            if (ups > 0.0) { if (tier) *tier = 1; return ups; }
        }
    }
    /* tier (ii)/(iii): dE and g_lo were resolved once at arm time (the level
     * walk is O(nlev) and this is called per (line,shell) in the assembler). */
    double dE = 0.0, g_lo = 0.0;
    if (g_omcm_armed && line >= 0 && line < g_omcm_nlines) {
        dE   = g_omcm_line_dE[line];
        g_lo = g_omcm_line_glo[line];
    }
    return omcm_fallback(atom->line_f_lu ? atom->line_f_lu[line] : 0.0,
                         g_lo, dE, T_e, atom->line_ion_number[line], tier);
}

/* Build the per-line map + print the 1-line census banner. Idempotent. */
void omega_cmfgen_arm(AtomicData *atom) {
    if (!omega_cmfgen_enabled() || g_omcm_armed || !atom || atom->n_lines <= 0)
        return;
    if (!artis_parity_enabled()) {
        fprintf(stderr, "[OMEGA-CMFGEN][WARN] LUMINA_ARTIS_PARITY=0 -> no col "
                        "tables are loaded and the parity Omega sites are not "
                        "reached; gate has NO effect. Set LUMINA_ARTIS_PARITY=1.\n");
        return;
    }
    int n = atom->n_lines;
    g_omcm_line_src = (int *)malloc((size_t)n * sizeof(int));
    g_omcm_line_t   = (int *)malloc((size_t)n * sizeof(int));
    g_omcm_line_dE  = (double *)malloc((size_t)n * sizeof(double));
    g_omcm_line_glo = (double *)malloc((size_t)n * sizeof(double));
    if (!g_omcm_line_src || !g_omcm_line_t || !g_omcm_line_dE || !g_omcm_line_glo) {
        free(g_omcm_line_src); free(g_omcm_line_t);
        free(g_omcm_line_dE);  free(g_omcm_line_glo);
        g_omcm_line_src = NULL; g_omcm_line_t = NULL;
        g_omcm_line_dE = NULL;  g_omcm_line_glo = NULL;
        fprintf(stderr, "[OMEGA-CMFGEN][ERROR] per-line map alloc failed\n");
        return;
    }
    for (int l = 0; l < n; l++) { g_omcm_line_src[l] = OMCM_NOSRC;
                                  g_omcm_line_t[l] = -1;
                                  g_omcm_line_dE[l] = 0.0;
                                  g_omcm_line_glo[l] = 0.0; }
    /* dE + g_lo per line (level_number -> global slot within the ion block) */
    {
        long n_nolev = 0;
        for (int l = 0; l < n; l++) {
            int ip = find_ion_pop_idx(atom, atom->line_atomic_number[l],
                                           atom->line_ion_number[l]);
            if (ip < 0) { n_nolev++; continue; }
            int lb = atom->level_offset[ip], lt = atom->level_offset[ip + 1];
            int lo_g = -1, up_g = -1;
            for (int g = lb; g < lt; g++) {
                if (atom->level_num[g] == atom->line_level_lower[l]) lo_g = g;
                if (atom->level_num[g] == atom->line_level_upper[l]) up_g = g;
                if (lo_g >= 0 && up_g >= 0) break;
            }
            if (lo_g < 0 || up_g < 0) { n_nolev++; continue; }
            g_omcm_line_dE[l]  = fabs(atom->level_energy_eV[up_g] -
                                      atom->level_energy_eV[lo_g]) * EV_TO_ERG;
            g_omcm_line_glo[l] = (double)atom->level_g[lo_g];
        }
        if (n_nolev)
            fprintf(stderr, "[OMEGA-CMFGEN][WARN] %ld of %d bb lines have no "
                            "resolvable level pair -> forced to OMEGA_SET\n",
                            n_nolev, n);
    }
    g_omcm_nlines = n;
    /* dense (a<b on level_number) -> transition index maps, one per source */
    struct { int Z, ion0, nlev; int *dm; } bs[1 + LUMINA_MAX_COL_IONS];
    int nb = 0;
    for (int src = -1; src < atom->ncol_ions; src++) {
        int Zs, ion0s, ntr;
        const int *tlo, *thi;
        if (src < 0) {
            if (!atom->feiii_col_loaded) continue;
            Zs = atom->feiii_col_Z; ion0s = atom->feiii_col_ion;
            ntr = atom->feiii_col_n_trans;
            tlo = atom->feiii_col_lo; thi = atom->feiii_col_hi;
        } else {
            Zs = atom->col_ion_Z[src]; ion0s = atom->col_ion_stage[src];
            ntr = atom->col_ion_n_trans[src];
            tlo = atom->col_ion_lo[src]; thi = atom->col_ion_hi[src];
        }
        if (ntr <= 0 || !tlo || !thi) continue;
        int maxlev = 0;
        for (int t = 0; t < ntr; t++) {
            if (tlo[t] > maxlev) maxlev = tlo[t];
            if (thi[t] > maxlev) maxlev = thi[t];
        }
        int nlev = maxlev + 1;
        int *dm = (int *)malloc((size_t)nlev * (size_t)nlev * sizeof(int));
        if (!dm) continue;
        for (size_t i = 0; i < (size_t)nlev * (size_t)nlev; i++) dm[i] = -1;
        for (int t = 0; t < ntr; t++) {
            int a = tlo[t] < thi[t] ? tlo[t] : thi[t];
            int b = tlo[t] < thi[t] ? thi[t] : tlo[t];
            if (a < 0 || b >= nlev) continue;
            dm[(size_t)a * (size_t)nlev + (size_t)b] = t;
        }
        bs[nb].Z = Zs; bs[nb].ion0 = ion0s; bs[nb].nlev = nlev; bs[nb].dm = dm;
        for (int l = 0; l < n; l++) {
            if (atom->line_atomic_number[l] != Zs ||
                atom->line_ion_number[l]    != ion0s) continue;
            int a = atom->line_level_lower[l], b = atom->line_level_upper[l];
            if (a > b) { int tmp = a; a = b; b = tmp; }
            if (a < 0 || b >= nlev) continue;
            int t = dm[(size_t)a * (size_t)nlev + (size_t)b];
            if (t >= 0) { g_omcm_line_src[l] = src; g_omcm_line_t[l] = t; }
        }
        nb++;
    }
    for (int b = 0; b < nb; b++) free(bs[b].dm);
    g_omcm_nsrc = nb;
    g_omcm_armed = 1;
    /* census (tier of every bb line at the radeq bake temperature) */
    g_omcm_n_tab = g_omcm_n_vr = g_omcm_n_set = 0;
    for (int l = 0; l < n; l++) {
        int tier = 3;
        (void)omega_cmfgen_line(atom, l, 10400.0, &tier);
        if      (tier == 1) g_omcm_n_tab++;
        else if (tier == 2) g_omcm_n_vr++;
        else                g_omcm_n_set++;
    }
    printf("  [OMEGA-CMFGEN] ON: %d Omega tables (feiii+col_ion) | of %d bb lines "
           "%ld tabulated (no floor), %ld van-Regemorter gbar-floor, %ld OMEGA_SET=%g "
           "(f<=%g)\n", g_omcm_nsrc, n, g_omcm_n_tab, g_omcm_n_vr, g_omcm_n_set,
           omcm_oset(), OMCM_FMIN);
    fflush(stdout);
}

/* ============================================================ */
/* Phase 4 - Step 2: Radiation field solver                     */
/* (mc_rad_field_solver.py: estimate_dilute_planck_radiation_field) */
/* ============================================================ */

void solve_radiation_field(Estimators *est, double time_explosion,
                            double time_simulation, double *volume,
                            OpacityState *opacity, PlasmaState *plasma,
                            double damping_constant) {
    /* A2-17: the canonical RadiationField commit owns the estimator payload.
     * No fitted scalar state is produced or retained here. */
    (void)est;
    (void)time_explosion;
    (void)time_simulation;
    (void)volume;
    (void)opacity;
    (void)plasma;
    (void)damping_constant;
}

/* Fit a dilute Planck W*B_nu(T) to the per-shell J_nu histogram.
 *
 * J_nu[b] = raw[b] / (4*pi * V * t_sim * dnu_b)   [erg/s/cm^2/Hz/sr]
 * Total mean intensity J = sum_b J_nu[b]*dnu_b.
 * For a dilute Planck, J = W * sigma_SB*T^4/pi, so at any trial T the amplitude
 * is fixed analytically by W(T) = pi*J/(sigma_SB*T^4); only the SHAPE (T) is
 * searched. We minimize an energy-weighted log-space residual so the fit tracks
 * the SED peak rather than the redshift-sensitive first moment.
 *
 * Returns 1 and writes (*T_out,*W_out) on success, 0 if histogram unavailable. */
static int fit_dilute_planck_binned_j(const Estimators *est, int shell,
                                      double volume, double time_simulation,
                                      double *T_out, double *W_out) {
    if (est->j_nu_estimator == NULL || est->nlte_n_freq_bins <= 0) return 0;
    if (volume <= 0.0 || time_simulation <= 0.0) return 0;

    int    nb    = est->nlte_n_freq_bins;
    double nu_lo0 = est->nlte_nu_min;
    double dlog  = est->nlte_d_log_nu;
    const double *raw = &est->j_nu_estimator[(size_t)shell * nb];
    double norm = 1.0 / (4.0 * M_PI_VAL * volume * time_simulation);

    /* Build J_nu, bin centers/widths, total J, and energy weights. */
    double *Jnu = (double *)malloc((size_t)nb * sizeof(double));
    double *nu_c = (double *)malloc((size_t)nb * sizeof(double));
    double *wgt = (double *)malloc((size_t)nb * sizeof(double));
    if (!Jnu || !nu_c || !wgt) { free(Jnu); free(nu_c); free(wgt); return 0; }

    double J_tot = 0.0;
    int n_pos = 0;
    for (int b = 0; b < nb; b++) {
        double nu_a = nu_lo0 * exp((double)b * dlog);
        double nu_b = nu_lo0 * exp((double)(b + 1) * dlog);
        double dnu  = nu_b - nu_a;
        nu_c[b] = 0.5 * (nu_a + nu_b);
        double j = (raw[b] > 0.0 && dnu > 0.0) ? raw[b] * norm / dnu : 0.0;
        Jnu[b]  = j;
        double ener = j * dnu;          /* energy content of this bin */
        wgt[b]  = ener;
        J_tot  += ener;
        if (j > 0.0) n_pos++;
    }
    if (J_tot <= 0.0 || n_pos < 4) { free(Jnu); free(nu_c); free(wgt); return 0; }

    /* Two-pass T search: coarse log grid, then local linear refine. */
    const double T_LO = 1500.0, T_HI = 50000.0;
    double best_T = 0.0, best_res = -1.0;
    for (int pass = 0; pass < 2; pass++) {
        double t_lo, t_hi; int nstep;
        if (pass == 0) { t_lo = T_LO; t_hi = T_HI; nstep = 80; }
        else {
            /* refine within +-1.5 coarse steps around best_T (log spacing) */
            double f = exp(1.5 * log(T_HI / T_LO) / 80.0);
            t_lo = best_T / f; t_hi = best_T * f; nstep = 60;
            if (t_lo < T_LO) t_lo = T_LO;
            if (t_hi > T_HI) t_hi = T_HI;
        }
        for (int k = 0; k <= nstep; k++) {
            double T = (pass == 0)
                ? t_lo * pow(t_hi / t_lo, (double)k / nstep)
                : t_lo + (t_hi - t_lo) * (double)k / nstep;
            double W_T = M_PI_VAL * J_tot / (SIGMA_SB * pow(T, 4));
            if (W_T <= 0.0) continue;
            double res = 0.0, wsum = 0.0;
            for (int b = 0; b < nb; b++) {
                if (Jnu[b] <= 0.0) continue;
                double model = W_T * planck_bnu(T, nu_c[b]);
                if (model <= 0.0) continue;
                double d = log(Jnu[b]) - log(model);
                res  += wgt[b] * d * d;
                wsum += wgt[b];
            }
            if (wsum <= 0.0) continue;
            res /= wsum;
            if (best_res < 0.0 || res < best_res) { best_res = res; best_T = T; }
        }
    }

    free(Jnu); free(nu_c); free(wgt);
    if (best_T <= 0.0) return 0;
    *T_out = best_T;
    *W_out = M_PI_VAL * J_tot / (SIGMA_SB * pow(best_T, 4));
    return 1;
}

/* ============================================================ */
/* [ARTIS-PARITY C1] per-bin (W,T_R) dilute-BB radiation field.  */
/* Faithful port of ARTIS radfield.cc fit_parameters (735-804),  */
/* find_bin_T_R (326-358), calculate_planck_integral (277-301)   */
/* and radfield() evaluation (717-732). ARTIS fits 24 WIDE bins   */
/* (nu_bar carries the SED slope); Lumina's 1000 narrow log bins  */
/* are aggregated into ARTIS_RADFIELD_NC coarse bins for the fit, */
/* then radfield(nu)=W·planck(nu,T_R) is evaluated back onto the  */
/* fine grid so the downstream rate integrals read a smooth,      */
/* per-bin-slope-aware field. Gated (master LUMINA_ARTIS_PARITY). */
/* ============================================================ */
#define ARTIS_RADFIELD_NC       24        /* radfield.cc RADFIELDBINCOUNT */
#define ARTIS_RADFIELD_TR_MIN   500.0     /* radfield.cc bins_T_R_min */
#define ARTIS_RADFIELD_TR_MAX   250000.0  /* radfield.cc bins_T_R_max */
/* [withParityQ item1] ARTIS T_e-superbin edge. artisoptions.h:70
 *   constexpr double RADFIELDBINS_NU_MAX = (CLIGHT / 1085e-8);
 * i.e. the fitted multi-bin radiation-field model STOPS at 1085 A; everything
 * shortward lives in ARTIS's single "superbin" whose solution is NOT fitted but
 * pinned:  radfield.cc:766-767  `if (binindex == RADFIELDBINCOUNT-1) T_R_bin =
 * grid::get_Te(nonemptymgi);`  followed by the ordinary
 * `W_bin = J_bin / calculate_planck_integral(T_R_bin, nu_lower, nu_upper)`
 * (radfield.cc:776,778).  radfield.cc:62 fixes the superbin's LOWER edge at
 * exactly CLIGHT/1085e-8:  delta_nu = (NU_MAX-NU_MIN)/(RADFIELDBINCOUNT-1)
 * ("- 1 for the top super bin"), so bins 0..22 tile [NU_MIN, NU_MAX] and bin 23
 * runs [NU_MAX, RADFIELDBINS_T_E_SUPERBIN_NU_MAX=CLIGHT/10e-8] (artisoptions.h:72).
 * ARTIS's pinned region is therefore exactly lambda in [10, 1085] A. */
#define ARTIS_RADFIELD_SUPERBIN_LAM_A 1085.0

/* radfield.cc:227 — series form of ∫_x^∞ for the Planck integral (x=hν/kT). */
static double artis_planck_x_to_inf(double x, double epsrel) {
    if (x > 700.0) return 0.0;
    double sum = 0.0, x2 = x * x, x3 = x2 * x;
    for (int n = 1; n < 1000; n++) {
        double dn = (double)n, n2 = dn * dn, n3 = n2 * dn, n4 = n3 * dn;
        double term = exp(-dn * x) * (x3 / dn + 3.0 * x2 / n2 + 6.0 * x / n3 + 6.0 / n4);
        sum += term;
        if (term < sum * epsrel) break;
    }
    return sum;
}
/* radfield.cc:250 — same, with an extra ν in the integrand. */
static double artis_nu_planck_x_to_inf(double x, double epsrel) {
    if (x > 700.0) return 0.0;
    double sum = 0.0, x2 = x * x, x3 = x2 * x, x4 = x3 * x;
    for (int n = 1; n < 1000; n++) {
        double dn = (double)n, n2 = dn * dn, n3 = n2 * dn, n4 = n3 * dn, n5 = n4 * dn;
        double term = exp(-dn * x) *
            (x4 / dn + 4.0 * x3 / n2 + 12.0 * x2 / n3 + 24.0 * x / n4 + 24.0 / n5);
        sum += term;
        if (term < sum * epsrel) break;
    }
    return sum;
}
/* radfield.cc:277 — ∫_{nu_low}^{nu_high} B_ν dν (times_nu=1: ∫ ν B_ν dν). */
static double artis_planck_integral(double T, double nu_low, double nu_high, int times_nu) {
    if (T <= 0.0) return 0.0;
    const double epsrel = 1e-15;
    double kb = K_BOLTZMANN, h = H_PLANCK, c2 = C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT;
    double x_low  = h * nu_low  / (kb * T);
    double x_high = h * nu_high / (kb * T);
    if (times_nu) {
        double kb5 = kb * kb * kb * kb * kb, T5 = T * T * T * T * T, h4 = h * h * h * h;
        double cf = (2.0 * kb5 * T5) / (h4 * c2);
        return cf * (artis_nu_planck_x_to_inf(x_low, epsrel) -
                     artis_nu_planck_x_to_inf(x_high, epsrel));
    }
    double kb4 = kb * kb * kb * kb, T4 = T * T * T * T, h3 = h * h * h;
    double cf = (2.0 * kb4 * T4) / (h3 * c2);
    return cf * (artis_planck_x_to_inf(x_low, epsrel) -
                 artis_planck_x_to_inf(x_high, epsrel));
}
/* radfield.cc:305 — nu_bar_planck(T_R) - nu_bar_estimator over a bin. */
static double artis_deltanubar(double T_R, double nu_lo, double nu_hi, double nu_bar_est) {
    double nuP = artis_planck_integral(T_R, nu_lo, nu_hi, 1);
    double P   = artis_planck_integral(T_R, nu_lo, nu_hi, 0);
    if (!(P > 0.0)) return -nu_bar_est;
    return nuP / P - nu_bar_est;
}
/* radfield.cc:326 — solve nu_bar_planck(T_R)=nu_bar_est in [TR_min,TR_max]
 * (bisection in place of Boost TOMS748; ARTIS clamp semantics preserved). */
static double artis_find_bin_T_R(double nu_bar_est, double nu_lo, double nu_hi) {
    double f_min = artis_deltanubar(ARTIS_RADFIELD_TR_MIN, nu_lo, nu_hi, nu_bar_est);
    double f_max = artis_deltanubar(ARTIS_RADFIELD_TR_MAX, nu_lo, nu_hi, nu_bar_est);
    int invalid = (!isfinite(f_min) || !isfinite(f_max));
    if (!invalid && f_min * f_max < 0.0) {
        double a = ARTIS_RADFIELD_TR_MIN, b = ARTIS_RADFIELD_TR_MAX, fa = f_min;
        for (int it = 0; it < 100; it++) {
            double m = 0.5 * (a + b);
            double fm = artis_deltanubar(m, nu_lo, nu_hi, nu_bar_est);
            if (fm == 0.0 || (b - a) < 1e-4 * m) return m;
            if (fa * fm < 0.0) b = m; else { a = m; fa = fm; }
        }
        return 0.5 * (a + b);
    }
    if (invalid || f_max < 0.0) return ARTIS_RADFIELD_TR_MAX;
    return ARTIS_RADFIELD_TR_MIN;
}

void nlte_build_perbin_dilute_field(NLTEConfig *nlte, double time_simulation,
                                    double *volume, double *T_e, int n_shells) {
    if (!nlte || !nlte->J_nu || !nlte->j_nu_estimator) return;
    const int nb = nlte->n_freq_bins;
    if (nb <= 0) return;
    for (int s = 0; s < n_shells; ++s) {
        const double V = volume[s];
        if (!(V > 0.0) || !(time_simulation > 0.0)) continue;
        const double norm = 1.0 / (4.0 * M_PI_VAL * V * time_simulation);
        for (int f = 0; f < nb; ++f) {
            const double nu_lo = nlte->nu_min * exp((double)f * nlte->d_log_nu);
            const double nu_hi = nlte->nu_min * exp((double)(f + 1) * nlte->d_log_nu);
            const double dnu = nu_hi - nu_lo;
            if (!(dnu > 0.0)) return;
            nlte->J_nu[(size_t)s * nb + f] =
                nlte->j_nu_estimator[(size_t)s * nb + f] * norm / dnu;
        }
    }
    (void)T_e;
    return;
#if 0 /* A2-17: retired production dilute-Planck scalar compression. */
    const double nu_min = nlte->nu_min, dlog = nlte->d_log_nu;
    const int NC = ARTIS_RADFIELD_NC;
    double raw_J[ARTIS_RADFIELD_NC], raw_nuJ[ARTIS_RADFIELD_NC];
    double W_c[ARTIS_RADFIELD_NC], TR_c[ARTIS_RADFIELD_NC];
    int    f_first[ARTIS_RADFIELD_NC], f_last[ARTIS_RADFIELD_NC];

    /* [C1-DEGEN-FALLBACK] opt-in field-representation fallback (default OFF ->
     * byte-identical). In railed+starved deep-EUV coarse bins the (W,T_R) fit
     * degenerates (T_R rails to 250 kK, W~1e-10, RJ plateau) and inflates
     * ground-channel photoionization 9-52x; publish the honest raw per-Hz field
     * there instead. Gate read + banner accounting only here; the actual field
     * change is confined to the fallback_on branches below. */
    int fallback_on = 0;
    { const char *e = getenv("LUMINA_C1_DEGEN_FALLBACK"); fallback_on = (e && atoi(e)); }
    static int fb_call = 0, fb_banner_done = 0;
    int fb_M_total = 0, fb_nsh = 0;
    char fb_shbuf[256]; fb_shbuf[0] = '\0';
    if (fallback_on) fb_call++;

    /* ------------------------------------------------------------------ */
    /* [withParityQ item1] GATE LUMINA_C1_SUPERBIN_TEPIN (default OFF).     */
    /* ARTIS radfield.cc:757-800 (fit_parameters) final-bin semantics:      */
    /*                                                                     */
    /*   if (J_bin > 0) {                                                  */
    /*     if (binindex == RADFIELDBINCOUNT - 1)                           */
    /*       T_R_bin = grid::get_Te(nonemptymgi);        // <-- the PIN    */
    /*     else { T_R_bin = find_bin_T_R(...); ... }                       */
    /*     planck_integral_result =                                        */
    /*         calculate_planck_integral(T_R_bin, nu_lower, nu_upper, false); */
    /*     W_bin = J_bin / planck_integral_result;                         */
    /*     if (W_bin > 1e4 || !isfinite(W_bin)) { retry at bins_T_R_max;   */
    /*       if still > 1e4 { T_R_bin = -99.; W_bin = 0.; } }              */
    /*   } else { T_R_bin = 0.; W_bin = 0.; }                              */
    /*                                                                     */
    /* ARTIS's fitted bin ladder ENDS at RADFIELDBINS_NU_MAX = CLIGHT/1085e-8 */
    /* (artisoptions.h:64-72); the pinned last bin is the deep-EUV superbin.  */
    /* Lumina's 24 coarse bins span 1.5e14..3.0e16 Hz, so the region ARTIS   */
    /* covers with ONE pinned superbin is here split over several coarse     */
    /* bins.  Semantics ported = "shortward of 1085 A the colour temperature */
    /* is NOT fitted, it IS T_e"; each such Lumina coarse bin is pinned      */
    /* individually (its own W from its own Planck integral), which is the   */
    /* faithful generalisation and strictly finer than ARTIS's single bin.   */
    /* Boundary rule: a coarse bin that STRADDLES 1085 A keeps the ordinary  */
    /* fit (ARTIS has no such bin — its ladder edge is exactly 1085 A — so   */
    /* there is no ARTIS behaviour to copy; leaving the straddler fitted is  */
    /* the conservative choice and is recorded as mode "fit" in the dump).   */
    /* On the shipped 1.5e14..3.0e16 Hz / 24-coarse-bin grid the straddler is */
    /* coarse bin 13 (1131.3..905.6 A); its 1085..905.6 A part belongs to     */
    /* ARTIS's superbin but stays FITTED here. Bins 14..23 (lam_hi<=905.6 A)  */
    /* are the pinned set. This is the one documented boundary difference.    */
    int tepin_on = 0;
    { const char *e = getenv("LUMINA_C1_SUPERBIN_TEPIN"); tepin_on = (e && atoi(e)); }
    static int tp_call = 0;
    int tp_M_total = 0, tp_nsh = 0;
    if (tepin_on) tp_call++;
    /* nu at 1085 A: bins with nu_lo >= this (i.e. lam_hi <= 1085 A) are pinned. */
    const double nu_superbin =
        C_SPEED_OF_LIGHT / (ARTIS_RADFIELD_SUPERBIN_LAM_A * 1.0e-8);

    /* ------------------------------------------------------------------ */
    /* [withParityQ item2] INSTRUMENT LUMINA_C1_BIN_DUMP (default OFF).     */
    /* Appends the C1 per-(shell,coarse-bin) fit result once per call (=    */
    /* once per co-evolve outer iteration; nlte_build_perbin_dilute_field   */
    /* has exactly one call site, lumina_cuda.cu, inside the `it` loop).    */
    /* Read-only: it prints W_c/TR_c AS BUILT, it does not recompute them.  */
    static int   bd_on = -1;            /* -1 unparsed; 0 off; 1 on (getenv once) */
    static FILE *bd_fp = NULL;
    static int   bd_iter = -1;          /* invocation index == outer iteration */
    if (bd_on < 0) {
        const char *e = getenv("LUMINA_C1_BIN_DUMP");
        bd_on = (e && atoi(e)) ? 1 : 0;
        if (bd_on) {
            bd_fp = fopen("lumina_c1_bins.csv", "w");
            if (bd_fp) {
                fprintf(bd_fp,
                    "iter,shell,bin,lam_lo_A,lam_hi_A,J_bin,W,T_R,mode\n");
                fflush(bd_fp);
                fprintf(stderr, "[withParityQ LUMINA_C1_BIN_DUMP] ARMED -> "
                        "lumina_c1_bins.csv (all shells x %d coarse bins, "
                        "appended every C1 build; mode=fit|pin|degen|empty; "
                        "lam_lo=c/nu_hi, lam_hi=c/nu_lo; J_bin=INTEGRATED "
                        "int J_nu dnu over the bin [erg/cm2/s/sr])\n", NC);
            } else {
                fprintf(stderr, "[withParityQ LUMINA_C1_BIN_DUMP] *** FAILED to "
                        "open lumina_c1_bins.csv - DUMP DISABLED ***\n");
                bd_on = 0;              /* fail-loud, fail-closed */
            }
        }
    }
    if (bd_on == 1) bd_iter++;

    /* ------------------------------------------------------------------ */
    /* [withParityY Y2] INSTRUMENT LUMINA_C2_BFR_DUMP (default OFF).       */
    /* Same shape as LUMINA_C1_BIN_DUMP above: one static getenv, one file, */
    /* one invocation counter (== the co-evolve outer iteration, since this */
    /* function has exactly one call site inside the `it` loop).            */
    static int   c2d_on = -1;           /* -1 unparsed; 0 off; 1 on */
    static FILE *c2d_fp = NULL;
    static int   c2d_iter = -1;
    if (c2d_on < 0) {
        const char *e = getenv("LUMINA_C2_BFR_DUMP");
        c2d_on = (e && atoi(e)) ? 1 : 0;
        if (c2d_on) {
            c2d_fp = fopen("lumina_c2_bfr_dump.csv", "w");
            if (c2d_fp) {
                fprintf(c2d_fp, "iter,shell,bin,nu_mid,J_raw,bfr,j_nu_count\n");
                fflush(c2d_fp);
                fprintf(stderr, "[withParityY LUMINA_C2_BFR_DUMP] ARMED -> "
                        "lumina_c2_bfr_dump.csv (all shells x %d fine bins, "
                        "appended every C1/C2 build; J_raw = RAW MC per-Hz field "
                        "rebuilt from j_nu_estimator because nlte->J_nu already "
                        "holds the C1 dilute-BB refit at the write point; "
                        "bfr = normalized Gamma_bf density; j_nu_count = MC "
                        "packet tally, -1 if the array is absent)\n", nb);
            } else {
                fprintf(stderr, "[withParityY LUMINA_C2_BFR_DUMP] *** FAILED to "
                        "open lumina_c2_bfr_dump.csv - DUMP DISABLED ***\n");
                c2d_on = 0;             /* fail-loud, fail-closed */
            }
        }
    }
    if (c2d_on == 1) c2d_iter++;

    for (int s = 0; s < n_shells; s++) {
        double V = volume[s];
        if (!(V > 0.0) || !(time_simulation > 0.0)) continue;
        double norm = 1.0 / (4.0 * M_PI_VAL * V * time_simulation); /* raw -> ∫J_ν dν */
        for (int c = 0; c < NC; c++) {
            raw_J[c] = 0.0; raw_nuJ[c] = 0.0; f_first[c] = -1; f_last[c] = -1;
        }
        const double *jr  = &nlte->j_nu_estimator[(size_t)s * nb];
        const double *nur = &nlte->nu_bar_nu_estimator[(size_t)s * nb];
        for (int f = 0; f < nb; f++) {
            int c = (int)((long)f * NC / nb);
            if (c < 0) c = 0; if (c >= NC) c = NC - 1;
            raw_J[c]   += jr[f];
            raw_nuJ[c] += nur[f];
            if (f_first[c] < 0) f_first[c] = f;
            f_last[c] = f;
        }
        /* [withParityQ item1] per-shell record of which coarse bins took the
         * T_e pin.  Read below (a) to let the pin BEAT C1_DEGEN_FALLBACK and
         * (b) by the item2 dump for the `mode` column.  Zero unless the gate
         * is armed => OFF path never changes a decision. */
        int pin_c[ARTIS_RADFIELD_NC];
        for (int c = 0; c < NC; c++) pin_c[c] = 0;
        for (int c = 0; c < NC; c++) {
            if (f_first[c] < 0) { W_c[c] = 0.0; TR_c[c] = 0.0; continue; }
            double J_c = raw_J[c] * norm;         /* ∫J_ν dν over the coarse bin */
            /* ARTIS floor (fit_parameters:796): J_bin<=0 -> W=0 -> zero field. */
            if (!(J_c > 0.0) || !(raw_J[c] > 0.0)) { W_c[c] = 0.0; TR_c[c] = 0.0; continue; }
            double nu_lo = nu_min * exp((double)f_first[c] * dlog);
            double nu_hi = nu_min * exp((double)(f_last[c] + 1) * dlog);
            /* [withParityQ item1] LUMINA_C1_SUPERBIN_TEPIN: deep-EUV bins take
             * the ARTIS superbin solution instead of a colour-temperature fit.
             *   T_R := T_e(shell)   (radfield.cc:767 grid::get_Te)
             *   W    := J_bin / int_bin B_nu(T_e) dnu   (radfield.cc:776,778,
             *           the SAME calculate_planck_integral used by the fit)
             * plus ARTIS's W>1e4 / non-finite rail retry (radfield.cc:780-795),
             * which in ARTIS is applied to the superbin as well (it sits after
             * the if/else that chose T_R_bin, not inside the fitted branch).
             * Entered only for bins entirely shortward of 1085 A; the J_bin<=0
             * exit above already ran, so the ARTIS `else {T_R=0; W=0;}` branch
             * is the shared code path (both give a zero field, see below). */
            if (tepin_on && nu_lo >= nu_superbin) {
                double Te_s = T_e ? T_e[s] : 0.0;
                if (Te_s > 0.0) {
                    double Pp = artis_planck_integral(Te_s, nu_lo, nu_hi, 0);
                    double Wp = (Pp > 0.0) ? J_c / Pp : 0.0;
                    double TRp = Te_s;
                    if (Wp > 1e4 || !isfinite(Wp)) {   /* radfield.cc:779 rail */
                        Pp = artis_planck_integral(ARTIS_RADFIELD_TR_MAX,
                                                   nu_lo, nu_hi, 0);
                        Wp = (Pp > 0.0) ? J_c / Pp : 0.0;
                        if (Wp > 1e4 || !isfinite(Wp)) { TRp = 0.0; Wp = 0.0; }
                        else TRp = ARTIS_RADFIELD_TR_MAX;
                    }
                    W_c[c] = Wp; TR_c[c] = TRp;
                    pin_c[c] = 1; tp_M_total++;
                    continue;                      /* pin wins: skip the fit */
                }
                /* T_e unavailable (NULL/<=0): fall through to the ordinary fit
                 * rather than publish a field pinned to a bogus temperature. */
            }
            double nu_bar = raw_nuJ[c] / raw_J[c];
            double TR = artis_find_bin_T_R(nu_bar, nu_lo, nu_hi);
            double P  = artis_planck_integral(TR, nu_lo, nu_hi, 0);
            double W  = (P > 0.0) ? J_c / P : 0.0;
            if (W > 1e4 || !isfinite(W)) {       /* fit_parameters:780 rail handling */
                P = artis_planck_integral(ARTIS_RADFIELD_TR_MAX, nu_lo, nu_hi, 0);
                W = (P > 0.0) ? J_c / P : 0.0;
                if (W > 1e4 || !isfinite(W)) { TR = 0.0; W = 0.0; }
                else TR = ARTIS_RADFIELD_TR_MAX;
            }
            W_c[c] = W; TR_c[c] = TR;
        }
        /* [withParityQ item1] shell tally for the armed-gate banner. */
        if (tepin_on) {
            int pin_here = 0;
            for (int c = 0; c < NC; c++) if (pin_c[c]) pin_here = 1;
            if (pin_here) tp_nsh++;
        }
        /* [C1-DEGEN-FALLBACK] criterion D(s,c) (dig_S3-validated: 0 FP / 74
         * healthy bins): T_R >= 0.95*TR_CEILING AND raw_frac < 1e-3, where
         * raw_frac = raw_J[c] / sum_c raw_J[c]. The per-shell norm cancels in the
         * ratio, so the un-normalized raw_J sums reproduce dig_S3_degeneracy's
         * raw_jint fraction exactly. degen_c is READ below only when fallback_on. */
        int degen_c[ARTIS_RADFIELD_NC] = {0};
        int fb_M_shell = 0;
        if (fallback_on) {
            double shell_total_raw = 0.0;
            for (int c = 0; c < NC; c++) shell_total_raw += raw_J[c];
            const double TR_rail = 0.95 * ARTIS_RADFIELD_TR_MAX;
            for (int c = 0; c < NC; c++) {
                if (f_first[c] < 0) continue;
                /* [withParityQ item1] the T_e pin BEATS the degeneracy fallback:
                 * a pinned bin is no longer a railed 250 kK fit, so the D(s,c)
                 * criterion cannot fire on it by construction — this guard makes
                 * that explicit and order-independent. pin_c is all-zero unless
                 * LUMINA_C1_SUPERBIN_TEPIN is armed => OFF path unchanged. */
                if (pin_c[c]) continue;
                double raw_frac = (shell_total_raw > 0.0)
                                  ? raw_J[c] / shell_total_raw : 0.0;
                if (TR_c[c] >= TR_rail && raw_frac < 1e-3) {
                    degen_c[c] = 1; fb_M_shell++;
                }
            }
            if (fb_M_shell > 0) {
                fb_M_total += fb_M_shell; fb_nsh++;
                if (strlen(fb_shbuf) < sizeof(fb_shbuf) - 8) {
                    char tmp[16];
                    snprintf(tmp, sizeof(tmp), "%s%d", fb_shbuf[0] ? "," : "", s);
                    strncat(fb_shbuf, tmp, sizeof(fb_shbuf) - strlen(fb_shbuf) - 1);
                }
            }
        }
        /* [withParityQ item2] LUMINA_C1_BIN_DUMP: append THIS shell's C1 result.
         * Placed after the fit + pin + degeneracy classification and before the
         * field is written back, so W/T_R are exactly the values the evaluation
         * below (and therefore next iteration's rate solve) will use.  Pure
         * observation: no array here is written. */
        if (bd_on == 1 && bd_fp) {
            for (int c = 0; c < NC; c++) {
                double J_c_d  = raw_J[c] * norm;
                double lam_lo = 0.0, lam_hi = 0.0;
                if (f_first[c] >= 0) {
                    double nlo = nu_min * exp((double)f_first[c] * dlog);
                    double nhi = nu_min * exp((double)(f_last[c] + 1) * dlog);
                    lam_lo = (C_SPEED_OF_LIGHT / nhi) * 1.0e8;   /* short edge */
                    lam_hi = (C_SPEED_OF_LIGHT / nlo) * 1.0e8;   /* long edge  */
                }
                const char *mode;
                if (f_first[c] < 0 || !(raw_J[c] > 0.0) || !(J_c_d > 0.0))
                    mode = "empty";                 /* ARTIS J_bin<=0: W=T_R=0 */
                else if (pin_c[c])            mode = "pin";
                else if (fallback_on && degen_c[c]) mode = "degen";
                else                          mode = "fit";
                fprintf(bd_fp, "%d,%d,%d,%.2f,%.2f,%.6e,%.6e,%.2f,%s\n",
                        bd_iter, s, c, lam_lo, lam_hi,
                        J_c_d, W_c[c], TR_c[c], mode);
            }
        }
        double *Jrow = &nlte->J_nu[(size_t)s * nb];
        for (int f = 0; f < nb; f++) {          /* radfield.cc:717 radfield(nu) */
            int c = (int)((long)f * NC / nb);
            if (c < 0) c = 0; if (c >= NC) c = NC - 1;
            if (fallback_on && degen_c[c]) {
                /* honest raw per-Hz field: raw_jint(f)/dnu(f) = jr[f]*norm/dnu_f.
                 * Zero raw -> zero (honest starvation), never the railed hot-fit
                 * plateau. A fine-resolution raw estimator (jr[f], per fine bin)
                 * exists here, so the per-fine-bin form is used (not coarse spread). */
                double nu_lo_f = nu_min * exp((double)f * dlog);
                double nu_hi_f = nu_min * exp((double)(f + 1) * dlog);
                double dnu_f = nu_hi_f - nu_lo_f;
                Jrow[f] = (dnu_f > 0.0) ? jr[f] * norm / dnu_f : 0.0;
            } else if (W_c[c] > 0.0 && TR_c[c] > 0.0) {
                double nu_ctr = nu_min * exp(((double)f + 0.5) * dlog);
                Jrow[f] = W_c[c] * planck_bnu(TR_c[c], nu_ctr);
            } else {
                Jrow[f] = 0.0;
            }
        }
        /* [ARTIS-PARITY C2] normalize the raw bf-rate estimator to the ARTIS Γ_bf
         * density Σ(comov_e·dist/ν)/(V·t·H) (radfield.cc:850). The photoion R_bf
         * loop then reads R_bf += Σ σ[bin]·bf_rate_estimator[bin]. */
        if (nlte->bf_rate_estimator) {
            double *bfr = &nlte->bf_rate_estimator[(size_t)s * nb];
            double bnorm = 1.0 / (V * time_simulation * H_PLANCK);
            for (int f = 0; f < nb; f++) bfr[f] *= bnorm;
        }
        /* [withParityY Y2] INSTRUMENT LUMINA_C2_BFR_DUMP (default OFF).
         * The C2 Gamma_bf density is dumped NOWHERE else.  This writer sits
         * immediately AFTER the normalization above, so `bfr` is byte-for-byte
         * the array the consumers (MA iup_prob, parity_gamma_phot, the CPU
         * matrix bfr branch) will read this iteration.
         * COLUMN SEMANTICS -- read this before using the file:
         *   J_raw  = jr[f]*norm/dnu_f  = the RAW per-Hz MC field.  nlte->J_nu was
         *            ALREADY OVERWRITTEN three lines above by the C1 dilute-BB
         *            refit (Jrow[f] = W_c*B_nu(TR_c)), so the pre-refit raw J is
         *            NOT in J_nu at this point.  It is recovered here from the
         *            untouched raw accumulator nlte->j_nu_estimator by exactly
         *            the expression the C1-degeneracy branch uses (:1061), i.e.
         *            this IS the raw MC estimate, not a re-fit.  The refit field
         *            (W,T_R per coarse bin) is dumped separately by
         *            LUMINA_C1_BIN_DUMP -> lumina_c1_bins.csv.
         *   bfr    = normalized ARTIS Gamma_bf density, i.e. the quantity the
         *            consumers use as R_bf += sigma[bin] * bfr[bin]
         *   j_nu_count = MC packet tally for the bin (-1 if the array is absent)
         * Pure observation: nothing here writes any physics array. */
        if (c2d_on == 1 && c2d_fp) {
            const double *bfr_r = nlte->bf_rate_estimator
                ? &nlte->bf_rate_estimator[(size_t)s * nb] : NULL;
            const int *cnt_r = nlte->j_nu_count
                ? &nlte->j_nu_count[(size_t)s * nb] : NULL;
            for (int f = 0; f < nb; f++) {
                double nu_lo_f = nu_min * exp((double)f * dlog);
                double nu_hi_f = nu_min * exp((double)(f + 1) * dlog);
                double dnu_f   = nu_hi_f - nu_lo_f;
                double nu_mid  = nu_min * exp(((double)f + 0.5) * dlog);
                double J_raw   = (dnu_f > 0.0) ? jr[f] * norm / dnu_f : 0.0;
                fprintf(c2d_fp, "%d,%d,%d,%.6e,%.6e,%.6e,%d\n",
                        c2d_iter, s, f, nu_mid, J_raw,
                        bfr_r ? bfr_r[f] : 0.0, cnt_r ? cnt_r[f] : -1);
            }
        }
    }
    if (fallback_on && fb_M_total > 0) {
        if (!fb_banner_done) {
            printf("[C1-DEGEN-FALLBACK] it %d: %d coarse bins railed->raw (shells %s)\n",
                   fb_call, fb_M_total, fb_shbuf);
            fb_banner_done = 1;
        } else {
            printf("[C1-DEGEN-FALLBACK] it %d: %d coarse bins -> raw (%d shells)\n",
                   fb_call, fb_M_total, fb_nsh);
        }
        fflush(stdout);
    }
    /* [withParityQ item1] fail-loud: one banner line per build while armed, so a
     * mis-set gate (or a bin ladder that never reaches 1085 A) is impossible to
     * overlook.  M=0 while armed is itself the alarm. */
    if (tepin_on) {
        printf("[C1-SUPERBIN-TEPIN] it %d: %d coarse bins pinned to T_R=T_e "
               "(lam_hi<=%.0fA, nu_lo>=%.4eHz) in %d shells "
               "[ARTIS radfield.cc:760-800 + artisoptions.h:64-72]\n",
               tp_call, tp_M_total, ARTIS_RADFIELD_SUPERBIN_LAM_A,
               nu_superbin, tp_nsh);
        fflush(stdout);
    }
    /* [withParityQ item2] flush the per-iteration block so a killed run still
     * leaves every completed iteration on disk. */
    if (bd_on == 1 && bd_fp) {
        fflush(bd_fp);
        printf("[C1-BIN-DUMP] it %d: %d shells x %d coarse bins appended -> "
               "lumina_c1_bins.csv\n", bd_iter, n_shells, NC);
        fflush(stdout);
    }
    /* [withParityY Y2] same per-iteration flush + banner for the C2 dump. */
    if (c2d_on == 1 && c2d_fp) {
        fflush(c2d_fp);
        printf("[C2-BFR-DUMP] it %d: %d shells x %d fine bins appended -> "
               "lumina_c2_bfr_dump.csv\n", c2d_iter, n_shells, nb);
        fflush(stdout);
    }
    (void)T_e;   /* radfield.cc:767 sets the last (superbin) T_R=T_e; Lumina's top
                  * log bin is not a wide T_e superbin, so all bins use the general
                  * fit. T_e retained in the signature for a future faithful superbin. */
    /* [withParityQ item1] SUPERSEDED WHEN ARMED: the "future faithful superbin"
     * above is now LUMINA_C1_SUPERBIN_TEPIN, which reads T_e[s] for every coarse
     * bin shortward of 1085 A.  The (void)T_e cast is retained (harmless) so the
     * gate-OFF translation unit is unchanged. */
#endif
}

/* [DIAG-T2] Per-(shell,coarse-bin) dilute-BB field census. Recomputes the SAME
 * ARTIS C1 (W_bin,T_R_bin) fit as nlte_build_perbin_dilute_field WITHOUT touching
 * nlte->J_nu, and writes it alongside the coarse-bin-integrated MC field J_bin and
 * the local Planck integral B_bin=∫B_ν(T_e)dν so a deviating radiation field is
 * localizable to a (shell,bin). Read-only; no-op if the C1 estimators are absent. */
void nlte_dump_perbin_field_csv(NLTEConfig *nlte, double time_simulation,
                                double *volume, double *T_e, int n_shells) {
    if (!nlte || !nlte->J_nu || !nlte->j_nu_estimator ||
        !nlte->nu_bar_nu_estimator) return;
    const int nb = nlte->n_freq_bins;
    if (nb <= 0) return;
    const double nu_min = nlte->nu_min, dlog = nlte->d_log_nu;
    const int NC = ARTIS_RADFIELD_NC;
    FILE *pf = fopen("lumina_census_perbin_field.csv", "w");
    if (!pf) return;
    fprintf(pf, "shell,coarse_bin,nu_lo_Hz,nu_hi_Hz,lam_hi_A,lam_lo_A,"
                "W_bin,T_R_bin_K,T_e_K,J_bin,B_bin_Te,J_over_B\n");
    double raw_J[ARTIS_RADFIELD_NC], raw_nuJ[ARTIS_RADFIELD_NC];
    int    f_first[ARTIS_RADFIELD_NC], f_last[ARTIS_RADFIELD_NC];
    for (int s = 0; s < n_shells; s++) {
        double V = volume[s], Te = T_e[s];
        if (!(V > 0.0) || !(time_simulation > 0.0)) continue;
        double norm = 1.0 / (4.0 * M_PI_VAL * V * time_simulation);
        for (int c = 0; c < NC; c++) {
            raw_J[c] = 0.0; raw_nuJ[c] = 0.0; f_first[c] = -1; f_last[c] = -1;
        }
        const double *jr  = &nlte->j_nu_estimator[(size_t)s * nb];
        const double *nur = &nlte->nu_bar_nu_estimator[(size_t)s * nb];
        for (int f = 0; f < nb; f++) {
            int c = (int)((long)f * NC / nb);
            if (c < 0) c = 0; if (c >= NC) c = NC - 1;
            raw_J[c]   += jr[f];
            raw_nuJ[c] += nur[f];
            if (f_first[c] < 0) f_first[c] = f;
            f_last[c] = f;
        }
        for (int c = 0; c < NC; c++) {
            if (f_first[c] < 0) continue;
            double nu_lo = nu_min * exp((double)f_first[c] * dlog);
            double nu_hi = nu_min * exp((double)(f_last[c] + 1) * dlog);
            double J_c = raw_J[c] * norm, W = 0.0, TR = 0.0;
            if (J_c > 0.0 && raw_J[c] > 0.0) {
                double nu_bar = raw_nuJ[c] / raw_J[c];
                TR = artis_find_bin_T_R(nu_bar, nu_lo, nu_hi);
                double P = artis_planck_integral(TR, nu_lo, nu_hi, 0);
                W = (P > 0.0) ? J_c / P : 0.0;
                if (W > 1e4 || !isfinite(W)) {
                    P = artis_planck_integral(ARTIS_RADFIELD_TR_MAX, nu_lo, nu_hi, 0);
                    W = (P > 0.0) ? J_c / P : 0.0;
                    if (W > 1e4 || !isfinite(W)) { TR = 0.0; W = 0.0; }
                    else TR = ARTIS_RADFIELD_TR_MAX;
                }
            }
            double B_c = (Te > 0.0) ? artis_planck_integral(Te, nu_lo, nu_hi, 0) : 0.0;
            fprintf(pf, "%d,%d,%.6e,%.6e,%.2f,%.2f,%.6e,%.2f,%.2f,%.6e,%.6e,%.4e\n",
                    s, c, nu_lo, nu_hi,
                    (nu_lo > 0.0) ? (C_SPEED_OF_LIGHT / nu_lo) * 1e8 : 0.0,
                    (nu_hi > 0.0) ? (C_SPEED_OF_LIGHT / nu_hi) * 1e8 : 0.0,
                    W, TR, Te, J_c, B_c, (B_c > 0.0) ? J_c / B_c : -1.0);
        }
    }
    fclose(pf);
    printf("[DIAG-T2] per-bin (W,T_R,mc_J,B) field census -> lumina_census_perbin_field.csv\n");
    /* [DIAG-T2 FINE] LUMINA_JNU_FINE_DUMP=1: per-FINE-bin raw MC estimator
     * (bin-INTEGRATED, erg/cm2/s/sr) + the published nlte->J_nu (per-Hz) —
     * provenance instrument for the spurious deep-EUV coarse-bin energy and
     * the C1 within-bin shape. Sparse rows (both ~0) skipped. Default OFF. */
    if (getenv("LUMINA_JNU_FINE_DUMP") && atoi(getenv("LUMINA_JNU_FINE_DUMP"))) {
        FILE *ffp = fopen("lumina_census_jnu_fine.csv", "w");
        if (ffp) {
            fprintf(ffp, "shell,fine_bin,nu_lo_Hz,nu_hi_Hz,lam_mid_A,"
                         "raw_jint,J_pub_perHz\n");
            for (int s = 0; s < n_shells; s++) {
                double V = volume[s];
                if (!(V > 0.0) || !(time_simulation > 0.0)) continue;
                double norm = 1.0 / (4.0 * M_PI_VAL * V * time_simulation);
                const double *jr = &nlte->j_nu_estimator[(size_t)s * nb];
                for (int f = 0; f < nb; f++) {
                    double nlo = nu_min * exp((double)f * dlog);
                    double nhi = nu_min * exp((double)(f + 1) * dlog);
                    double raw = jr[f] * norm;
                    double Jp  = nlte->J_nu ? nlte->J_nu[(size_t)s * nb + f] : 0.0;
                    if (raw == 0.0 && Jp <= 1e-30) continue;
                    fprintf(ffp, "%d,%d,%.6e,%.6e,%.2f,%.6e,%.6e\n", s, f, nlo, nhi,
                            C_SPEED_OF_LIGHT / (0.5 * (nlo + nhi)) * 1e8, raw, Jp);
                }
            }
            fclose(ffp);
            printf("[DIAG-T2] fine-grid j_nu (raw_jint + J_pub) -> "
                   "lumina_census_jnu_fine.csv\n");
        }
    }
}

/* ============================================================ */
/* Phase 4 - Step 3: T_inner update from escape fraction        */
/* ============================================================ */

void update_t_inner(MCConfig *config, double L_emitted) {
    /* Phase 4 - Step 3: TARDIS convergence formula (base.py: estimate_t_inner)
     * T_inner_est = T_inner * (L_emitted / L_requested)^(t_inner_update_exponent)
     * t_inner_update_exponent = -0.5 (TARDIS default)
     * Then damping: T_inner_new = T_inner + d * (T_inner_est - T_inner) */
    if (L_emitted > 0.0) {
        double luminosity_ratio = L_emitted / config->luminosity_requested;
        double T_inner_estimated = config->T_inner * pow(luminosity_ratio, -0.5);
        /* TARDIS damping: T_inner_new = T_inner + d * (T_inner_est - T_inner) */
        config->T_inner += config->damping_constant *
            (T_inner_estimated - config->T_inner);
    }
}

/* ============================================================ */
/* Task #072: Plasma solver — tau_sobolev recomputation          */
/* ============================================================ */

/* Task #072: Helper — find ion population index for (Z, ion_stage) */
static int find_ion_pop_idx(AtomicData *atom, int Z, int ion_stage) {
    /* ion_stage is the ABSOLUTE carsus ion number (neutral=0). The ion-pop
       ladder may start above neutral (CMFGEN omits Ti I / Mn I), so slot index
       != absolute ion number in general. Match against the absolute stage
       stored in ion_pop_stage rather than assuming a relative offset. */
    for (int e = 0; e < atom->n_elements; e++) {
        if (atom->element_Z[e] != Z) continue;
        int start = atom->elem_ion_offset[e];
        int end   = atom->elem_ion_offset[e + 1];
        for (int ip = start; ip < end; ip++) {
            if (atom->ion_pop_stage[ip] == ion_stage) return ip;
        }
        return -1;
    }
    return -1;
}

/* Task #072: Helper — find ionization energy for (Z, ion_stage) -> (Z, ion_stage+1) */
static double find_ioniz_energy(AtomicData *atom, int Z, int ion_stage) {
    for (int i = 0; i < atom->n_ionization; i++) {
        if (atom->ioniz_Z[i] == Z && atom->ioniz_ion[i] == ion_stage)
            return atom->ioniz_energy_eV[i];
    }
    return 1e10; /* impossibly high — prevents ionization */
}

/* P4 (2026-05-14): ζ override env knobs for sensitivity probing.
 *   LUMINA_ZETA_OVERRIDE_ZMASK    comma-list of Z (e.g. "27,28")
 *   LUMINA_ZETA_OVERRIDE_IONMASK  comma-list of ion_number values matching csv (e.g. "2")
 *   LUMINA_ZETA_OVERRIDE_VAL      replacement ζ (default 0.5)
 * Probes M-L hybrid response to ζ for known Carsus placeholder rows
 * (Z=27 Co III ≡ Z=28 Ni III ≡ Z=22 Ti III; Z=27 Co II ≡ Z=21 Sc II). */
static unsigned int zeta_override_z_mask = 0;
static unsigned int zeta_override_ion_mask = 0;
static double zeta_override_val = 0.5;
static int zeta_override_initialized = 0;

static void init_zeta_override(void) {
    if (zeta_override_initialized) return;
    zeta_override_initialized = 1;

    const char *zm = getenv("LUMINA_ZETA_OVERRIDE_ZMASK");
    if (zm && *zm) {
        const char *p = zm;
        while (*p) {
            int z = atoi(p);
            if (z > 0 && z < 32) zeta_override_z_mask |= (1u << z);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *im = getenv("LUMINA_ZETA_OVERRIDE_IONMASK");
    if (im && *im) {
        const char *p = im;
        while (*p) {
            int ii = atoi(p);
            if (ii >= 0 && ii < 8) zeta_override_ion_mask |= (1u << ii);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *vv = getenv("LUMINA_ZETA_OVERRIDE_VAL");
    if (vv) zeta_override_val = atof(vv);

    if (zeta_override_z_mask != 0 && zeta_override_ion_mask != 0) {
        printf("  [ζ-override] val=%.3f zmask=0x%x ionmask=0x%x\n",
               zeta_override_val, zeta_override_z_mask, zeta_override_ion_mask);
    }
}

/* P3 (2026-05-14): Two-component ion-lock — W-threshold conditional LTE-at-T_e.
 *   LUMINA_LOCK_W_THRESH   shell W threshold (>= activates LTE-at-T_e lock; default 0 = off)
 *   LUMINA_LOCK_ZMASK      comma-list of Z (default iron-peak 21..28 when active)
 *   LUMINA_LOCK_IONMASK    comma-list of k array indices (default k=1,2 = II→III + III→IV)
 * Effect: inner shells (W>thresh) get phi = phi_LTE_at_Te (pure Saha at T_e, more
 * recombined → less Fe III → less UV opacity). Outer shells keep M-L hybrid. */
static double lock_w_thresh = 0.0;
static unsigned int lock_z_mask = 0;
static unsigned int lock_ion_mask = 0;
static int lock_initialized = 0;

static void init_twocomp_lock(void) {
    if (lock_initialized) return;
    lock_initialized = 1;

    const char *wt = getenv("LUMINA_LOCK_W_THRESH");
    if (wt) lock_w_thresh = atof(wt);

    const char *zm = getenv("LUMINA_LOCK_ZMASK");
    if (zm && *zm) {
        const char *p = zm;
        while (*p) {
            int z = atoi(p);
            if (z > 0 && z < 32) lock_z_mask |= (1u << z);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *im = getenv("LUMINA_LOCK_IONMASK");
    if (im && *im) {
        const char *p = im;
        while (*p) {
            int ii = atoi(p);
            if (ii >= 0 && ii < 8) lock_ion_mask |= (1u << ii);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }

    if (lock_w_thresh > 0.0) {
        printf("  [2-comp-lock] W_thresh=%.3f zmask=0x%x ionmask=0x%x\n",
               lock_w_thresh, lock_z_mask, lock_ion_mask);
    }
}

static inline double apply_twocomp_lock(double phi_neb, double phi_LTE_at_Te,
                                         int Z, int k, double W) {
    if (lock_w_thresh <= 0.0 || W < lock_w_thresh) return phi_neb;
    if (lock_z_mask != 0 &&
        (Z < 0 || Z >= 32 || !(lock_z_mask & (1u << Z))))
        return phi_neb;
    if (lock_ion_mask != 0 &&
        (k < 0 || k >= 8 || !(lock_ion_mask & (1u << k))))
        return phi_neb;
    return phi_LTE_at_Te;
}

/* (2) NLTE no-ML-lock (2026-05-14): replace phi_neb-derived n_total in the
 * NLTE pair conservation row with the element's total mass-conserved density.
 * Removes the Mihalas-Lucy "soft lock" on the (II+III) pair total, letting
 * the rate matrix redistribute mass across all ion stages it tracks.
 * Probe to test whether the structural cooling-feedback divergence stems
 * from the phi_neb input itself rather than from level/ion redistribution. */
static int nlte_no_ml_lock_init = 0;
static int nlte_no_ml_lock_enabled = 0;
static void init_nlte_no_ml_lock(void) {
    if (nlte_no_ml_lock_init) return;
    nlte_no_ml_lock_init = 1;
    const char *e = getenv("LUMINA_NLTE_NO_ML_LOCK");
    if (e && atoi(e) != 0) {
        nlte_no_ml_lock_enabled = 1;
        printf("  [NLTE no-ML-lock] pair n_total -> n_element (mass conservation)\n");
    }
}

double nlte_pair_total_density(NLTEConfig *nlte, AtomicData *atom,
                               PlasmaState *plasma,
                               int Z_nl, int ion_idx_lo, int ion_idx_hi,
                               int shell) {
    init_nlte_no_ml_lock();
    int n_shells = plasma->n_shells;
    if (nlte_no_ml_lock_enabled) {
        for (int e = 0; e < atom->n_elements; e++) {
            if (atom->element_Z[e] != Z_nl) continue;
            double mass_amu = atom->element_mass_amu[e];
            double rho = plasma->rho[shell];
            double abund = atom->abundances[e * n_shells + shell];
            return (abund * rho) / (mass_amu * AMU);
        }
        return 0.0;
    }
    double n_total = 0.0;
    for (int i = ion_idx_lo; i <= ion_idx_hi; i++) {
        int ip = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[i]);
        if (ip >= 0)
            n_total += atom->ion_number_density[ip * n_shells + shell];
    }
    return n_total;
}

int lumina_zinert_element_inactive(const AtomicData *atom, int element,
                                   int n_shells) {
    if (!atom || !atom->abundances || element < 0 ||
        element >= atom->n_elements || n_shells <= 0)
        return 0;
    for (int s = 0; s < n_shells; s++)
        if (atom->abundances[(size_t)element * n_shells + s] != 0.0)
            return 0;
    return 1;
}

/* L0-CLOSE-R2 section 3.7, Z-INERT.
 *
 * Topology is deliberately retained for zero-abundance elements.  This audit
 * therefore distinguishes represented cells from physically live cells and
 * checks the last element-attributable quantities before continuum/CMF/rate
 * consumers combine them into element-agnostic totals.  A candidate count is
 * the exact production admission predicate:
 *   continuum        n_ion >= 1e-30 (compute_bf_opacity)
 *   emissivity       tau > 1e-12    (cmfgen_assemble)
 *   transport        tau > 0        (nonzero contribution to tau accumulation)
 *   heating/cooling  any nonzero ion or NLTE level population
 * Non-finite values are violations as well as nonzero values. */
int lumina_zinert_validate(const AtomicData *atom, const NLTEConfig *nlte,
                           const OpacityState *opacity, int n_shells,
                           const char *stage) {
    typedef struct {
        unsigned long long zero_shells;
        unsigned long long ion_cells, ion_nonzero;
        unsigned long long population_cells, population_nonzero;
        unsigned long long line_cells, line_opacity_nonzero;
        unsigned long long line_source_nonzero;
        unsigned long long continuum_candidates;
        unsigned long long emissivity_candidates;
        unsigned long long transport_candidates;
        unsigned long long heating_cooling_candidates;
    } ZCount;
    ZCount count[100];
    unsigned char *zero_shell = NULL;
    int failed = 0;

    memset(count, 0, sizeof(count));
    if (!atom || !atom->abundances || !atom->element_Z || n_shells <= 0 ||
        (size_t)n_shells > SIZE_MAX / 100) {
        fprintf(stderr,
                "[Z-INERT][FATAL] stage=%s invalid audit state n_shells=%d\n",
                stage ? stage : "unknown", n_shells);
        return 1;
    }
    zero_shell = (unsigned char *)calloc((size_t)100 * n_shells, 1);
    if (!zero_shell) {
        fprintf(stderr, "[Z-INERT][FATAL] stage=%s audit allocation failed\n",
                stage ? stage : "unknown");
        return 1;
    }

    for (int e = 0; e < atom->n_elements; e++) {
        int Z = atom->element_Z[e];
        if (Z <= 0 || Z >= 100) continue;
        if (!lumina_zinert_element_inactive(atom, e, n_shells)) continue;
        for (int s = 0; s < n_shells; s++) {
            zero_shell[(size_t)Z * n_shells + s] = 1;
            count[Z].zero_shells++;
        }
    }

    if (atom->ion_number_density && atom->elem_ion_offset) {
        for (int e = 0; e < atom->n_elements; e++) {
            int Z = atom->element_Z[e];
            if (Z <= 0 || Z >= 100) continue;
            for (int ip = atom->elem_ion_offset[e];
                 ip < atom->elem_ion_offset[e + 1]; ip++) {
                for (int s = 0; s < n_shells; s++) {
                    if (!zero_shell[(size_t)Z * n_shells + s]) continue;
                    double value = atom->ion_number_density[(size_t)ip * n_shells + s];
                    count[Z].ion_cells++;
                    if (value != 0.0 || !isfinite(value)) {
                        count[Z].ion_nonzero++;
                        count[Z].heating_cooling_candidates++;
                        failed = 1;
                    }
                    if (isfinite(value) && value >= 1e-30)
                        count[Z].continuum_candidates++;
                }
            }
        }
    }

    if (nlte && nlte->nlte_level_populations) {
        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int Z = nlte->nlte_Z[ii];
            if (Z <= 0 || Z >= 100) continue;
            for (int level = nlte->nlte_ion_level_offset[ii];
                 level < nlte->nlte_ion_level_offset[ii + 1]; level++) {
                for (int s = 0; s < n_shells; s++) {
                    if (!zero_shell[(size_t)Z * n_shells + s]) continue;
                    double value = nlte->nlte_level_populations[
                        (size_t)level * n_shells + s];
                    count[Z].population_cells++;
                    if (value != 0.0 || !isfinite(value)) {
                        count[Z].population_nonzero++;
                        count[Z].heating_cooling_candidates++;
                        failed = 1;
                    }
                }
            }
        }
    }

    if (opacity && opacity->tau_sobolev && atom->line_atomic_number) {
        int n_lines = opacity->n_lines;
        for (int line = 0; line < n_lines; line++) {
            int Z = atom->line_atomic_number[line];
            if (Z <= 0 || Z >= 100) continue;
            for (int s = 0; s < n_shells; s++) {
                if (!zero_shell[(size_t)Z * n_shells + s]) continue;
                size_t at = (size_t)line * n_shells + s;
                double tau = opacity->tau_sobolev[at];
                count[Z].line_cells++;
                if (tau != 0.0 || !isfinite(tau)) {
                    count[Z].line_opacity_nonzero++;
                    failed = 1;
                }
                if (isfinite(tau) && tau > 0.0)
                    count[Z].transport_candidates++;
                if (isfinite(tau) && tau > 1e-12)
                    count[Z].emissivity_candidates++;
                if (opacity->line_source_S) {
                    double source = opacity->line_source_S[at];
                    if (source != 0.0 || !isfinite(source)) {
                        count[Z].line_source_nonzero++;
                        failed = 1;
                    }
                }
            }
        }
    }

    for (int Z = 1; Z < 100; Z++) {
        ZCount *c = &count[Z];
        if (c->zero_shells == 0) continue;
        int z_failed = c->ion_nonzero != 0 || c->population_nonzero != 0 ||
                       c->line_opacity_nonzero != 0 ||
                       c->line_source_nonzero != 0 ||
                       c->continuum_candidates != 0 ||
                       c->emissivity_candidates != 0 ||
                       c->heating_cooling_candidates != 0 ||
                       c->transport_candidates != 0;
        printf("[Z-INERT] stage=%s Z=%d zero_shells=%llu "
               "ions=%llu ion_nonzero=%llu populations=%llu "
               "population_nonzero=%llu lines=%llu line_opacity_nonzero=%llu "
               "line_source_nonzero=%llu continuum_candidates=%llu "
               "emissivity_candidates=%llu heating_cooling_candidates=%llu "
               "transport_candidates=%llu verdict=%s\n",
               stage ? stage : "unknown", Z, c->zero_shells,
               c->ion_cells, c->ion_nonzero,
               c->population_cells, c->population_nonzero,
               c->line_cells, c->line_opacity_nonzero,
               c->line_source_nonzero, c->continuum_candidates,
               c->emissivity_candidates, c->heating_cooling_candidates,
               c->transport_candidates, z_failed ? "FAIL" : "PASS");
    }
    fflush(stdout);
    if (failed)
        fprintf(stderr,
                "[Z-INERT][FATAL] stage=%s exact-zero downstream violation\n",
                stage ? stage : "unknown");
    free(zero_shell);
    return failed;
}

static int zinert_audit_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("LUMINA_ZINERT_AUDIT");
        enabled = (value && atoi(value) != 0) ? 1 : 0;
    }
    return enabled;
}

/* Task #072: Helper — interpolate zeta factor for (Z, ion_stage) at temperature T */
static double interpolate_zeta(AtomicData *atom, int Z, int ion_stage, double T) {
    /* P4 override: replace ζ with constant value for masked (Z, ion) pairs */
    if (zeta_override_z_mask != 0 && zeta_override_ion_mask != 0 &&
        Z > 0 && Z < 32 && ion_stage >= 0 && ion_stage < 8 &&
        (zeta_override_z_mask & (1u << Z)) &&
        (zeta_override_ion_mask & (1u << ion_stage))) {
        return zeta_override_val;
    }

    /* Find zeta entry for this (Z, ion) */
    int zidx = -1;
    for (int i = 0; i < atom->n_zeta_ions; i++) {
        if (atom->zeta_Z[i] == Z && atom->zeta_ion[i] == ion_stage) {
            zidx = i;
            break;
        }
    }
    if (zidx < 0) return 1.0; /* no zeta data -> LTE (zeta=1) */

    int nt = atom->n_zeta_temps;
    double *temps = atom->zeta_temps;
    double *vals = atom->zeta_data + zidx * nt;

    /* Clamp to grid bounds */
    if (T <= temps[0]) return vals[0];
    if (T >= temps[nt - 1]) return vals[nt - 1];

    /* Linear interpolation */
    for (int i = 0; i < nt - 1; i++) {
        if (T >= temps[i] && T <= temps[i + 1]) {
            double frac = (T - temps[i]) / (temps[i + 1] - temps[i]);
            return vals[i] + frac * (vals[i + 1] - vals[i]);
        }
    }
    return vals[nt - 1];
}

/* A2-07: the sole production partition owner.  The helper accepts only the
 * atomic membership and T_e; no PlasmaState radiation quantities can enter. */
static PopulationAtomicView population_atomic_view(const AtomicData *atom) {
    PopulationAtomicView view;
    memset(&view, 0, sizeof(view));
    view.n_ions = atom ? (size_t)atom->n_ion_pops : 0;
    view.n_levels = atom ? (size_t)atom->n_levels : 0;
    view.level_offset = atom ? atom->level_offset : NULL;
    view.energy_eV = atom ? atom->level_energy_eV : NULL;
    view.g = atom ? atom->level_g : NULL;
    view.level_Z = atom ? atom->level_Z : NULL;
    view.level_ion = atom ? atom->level_ion : NULL;
    return view;
}

static PopulationStatus compute_partition_functions(AtomicData *atom,
                                                     PlasmaState *plasma,
                                                     int n_shells) {
    if (!atom || !plasma || !plasma->T_e || n_shells <= 0 ||
        plasma->n_shells != n_shells || plasma->T_e_generation == 0)
        return POP_INVALID_TE;
    PopulationAtomicView view = population_atomic_view(atom);
    uint64_t generation = atom->population_committed_generation + 1;
    if (generation == 0) return POP_STALE_DERIVED_TEMPERATURE;
    PopulationStatus status = population_partition_build(
        &view, plasma->T_e, (size_t)n_shells, generation,
        plasma->T_e_generation, atom->partition_functions,
        &atom->partition_stamp);
    if (status == POP_OK)
        printf("  [A2-07] partition Z(T_e) committed generation=%llu te_generation=%llu\n",
               (unsigned long long)generation,
               (unsigned long long)plasma->T_e_generation);
    else {
        /* 실패 경로 전용 진단.  상위는 상태 코드만 받으므로 **어느 이온이** 걸렸는지
         * 알 수 없다 — 침묵으로 되돌아가지 않도록 여기서 특정한다. */
        fprintf(stderr, "[A2-07][FATAL] partition build failed: %s "
                "(n_ions=%zu n_levels=%zu pub=%p te_gen=%llu)\n",
                population_status_name(status), view.n_ions, view.n_levels,
                (void *)atom->partition_functions,
                (unsigned long long)plasma->T_e_generation);
        if (!view.level_offset || !view.energy_eV || !view.g)
            fprintf(stderr, "  view missing: level_offset=%p energy_eV=%p g=%p\n",
                    (void *)view.level_offset, (void *)view.energy_eV,
                    (void *)view.g);
        else {
            int n_empty = 0;
            for (size_t i = 0; i < view.n_ions; i++) {
                int lo = view.level_offset[i], hi = view.level_offset[i + 1];
                if (hi > lo) continue;
                n_empty++;
                int z = atom->ion_pop_Z ? atom->ion_pop_Z[i] : -1;
                int st = atom->ion_pop_stage ? atom->ion_pop_stage[i] : -1;
                /* 그 원소의 최상단 population 인가?  최상단이면 준위가 없는 것이
                 * 결손이 아니라 **정상**이다(전리 에너지 n 개 -> population n+1 개). */
                int top = 0;
                if (atom->ion_pop_Z && i + 1 < view.n_ions)
                    top = (atom->ion_pop_Z[i + 1] != z);
                else if (i + 1 == view.n_ions)
                    top = 1;
                fprintf(stderr, "  empty ion_pop %zu: Z=%d stage=%d (lo=%d hi=%d)%s\n",
                        i, z, st, lo, hi, top ? "  <- 원소 최상단" : "");
            }
            fprintf(stderr, "  empty ion_pops total=%d / %zu\n", n_empty, view.n_ions);
        }
    }
    return status;
}

/* [FB-MILNE] Exact per-level radiative-recombination bf-cooling coefficient,
 * mirroring ARTIS ratecoeff.cc (bfcooling_integrand :82-89, modified_sahafact
 * :145, assembly :173-182). Milne integral of the CMFGEN per-level sigma_bf
 * over the shared NLTE freq grid, charging the electron pool the photoelectron
 * KINETIC energy h(nu-nu0) only (not the binding energy hnu0):
 *   Lambda = 4pi * SAHACONST*(g_lo/g_up)*Te^-1.5
 *            * SUM_{f: nu_f>nu0} sigma[f]*(nu-nu0)*(2h/c^2)*nu^2*exp(-h(nu-nu0)/kTe)*dnu_f
 * so the cooling-rate density = n_e * n_upperion * Lambda [erg/s/cm^3], the same
 * units/role as C_ff. sigma_row = atom->cmfgen_sigma_bf + (size_t)l*NLTE_N_FREQ_BINS.
 * This is the detailed-balance partner of the transport bf opacity chi_bf (which
 * uses the SAME sigma_bf and NLTE pops) => S_nu=j/chi satisfies Kirchhoff by
 * construction (sub-Planckian where the recombined level is sub-thermal). */
static double fb_milne_cooling_coeff(const double *sigma_row, double nu0,
                                     double g_lo, double g_up, double Te) {
    if (!(Te > 0.0) || !(nu0 > 0.0) || !(g_up > 0.0) || !(g_lo > 0.0)) return 0.0;
    const double SAHACONST = 2.0706659e-16; /* (h^2/(2 pi m_e k_B))^1.5 in cgs */
    const double CL = 2.99792458e10;        /* c [cm/s] */
    double kTe = K_BOLTZMANN * Te;
    double dln = log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS;
    double acc = 0.0;
    for (int f = 0; f < NLTE_N_FREQ_BINS; f++) {
        double nu = NLTE_NU_MIN * exp(((double)f + 0.5) * dln);
        if (nu <= nu0) continue;
        double sig = sigma_row[f];
        if (sig <= 0.0) continue;
        double x = H_PLANCK * (nu - nu0) / kTe;
        if (x > 500.0) break;                /* Wien tail: grid is nu-increasing */
        double dnu = nu * dln;               /* log-grid bin width dnu = nu*dln */
        acc += sig * (nu - nu0) * (2.0 * H_PLANCK / (CL * CL))
               * nu * nu * exp(-x) * dnu;
    }
    double sahafact = SAHACONST * (g_lo / g_up) * pow(Te, -1.5);
    return 4.0 * M_PI_VAL * sahafact * acc;
}


/* Mihalas-Lucy phi_neb correction (Path d, 2026-05-13):
 *   phi_neb' = phi_neb * boost * (T_e/T_rad)^t_ratio_exp
 * Default boost=1.0, t_ratio_exp=0 (no change). Use Z/ion bitmasks
 * to apply selectively. Targets over-ionization at SN photosphere
 * where interpolate_zeta() falls back to 1.0 (no ζ data). */
static double ml_phi_neb_boost = 1.0;
static double ml_phi_neb_t_ratio_exp = 0.0;
static unsigned int ml_phi_neb_z_mask = 0;
static unsigned int ml_phi_neb_ion_mask = 0;
static int ml_phi_neb_initialized = 0;

static void init_ml_phi_neb_correction(void) {
    if (ml_phi_neb_initialized) return;
    ml_phi_neb_initialized = 1;

    const char *eb = getenv("LUMINA_ML_PHI_NEB_BOOST");
    if (eb) ml_phi_neb_boost = atof(eb);
    const char *ee = getenv("LUMINA_ML_PHI_NEB_T_RATIO_EXP");
    if (ee) ml_phi_neb_t_ratio_exp = atof(ee);

    const char *zm = getenv("LUMINA_ML_PHI_NEB_ZMASK");
    if (zm && *zm) {
        const char *p = zm;
        while (*p) {
            int z = atoi(p);
            if (z > 0 && z < 32) ml_phi_neb_z_mask |= (1u << z);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }
    const char *im = getenv("LUMINA_ML_PHI_NEB_IONMASK");
    if (im && *im) {
        const char *p = im;
        while (*p) {
            int ii = atoi(p);
            if (ii >= 0 && ii < 8) ml_phi_neb_ion_mask |= (1u << ii);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    }

    if (ml_phi_neb_boost != 1.0 || ml_phi_neb_t_ratio_exp != 0.0) {
        printf("  [M-L phi_neb] boost=%g t_ratio_exp=%g zmask=0x%x ionmask=0x%x\n",
               ml_phi_neb_boost, ml_phi_neb_t_ratio_exp,
               ml_phi_neb_z_mask, ml_phi_neb_ion_mask);
    }
}

static inline double apply_ml_phi_neb_correction(double phi_neb, int Z, int k,
                                                  double T_e, double T_rad) {
    if (ml_phi_neb_z_mask != 0 &&
        (Z < 0 || Z >= 32 || !(ml_phi_neb_z_mask & (1u << Z))))
        return phi_neb;
    if (ml_phi_neb_ion_mask != 0 &&
        (k < 0 || k >= 8 || !(ml_phi_neb_ion_mask & (1u << k))))
        return phi_neb;

    double factor = ml_phi_neb_boost;
    if (ml_phi_neb_t_ratio_exp != 0.0 && T_rad > 0.0) {
        double t_ratio = T_e / T_rad;
        if (t_ratio > 0.0) factor *= pow(t_ratio, ml_phi_neb_t_ratio_exp);
    }
    return phi_neb * factor;
}

/* Task #072 Step 4b: Compute ion number densities (Saha + nebular)
 * Uses TARDIS formula (Mazzali & Lucy 1993, eq. 14):
 *   phi_nebular = phi_lte * W * (zeta*delta + W*(1-zeta)) * sqrt(T_e/T_rad)
 *   ratio = phi_nebular / n_e
 *
 * phi_lte = (Z_{i+1}/Z_i) * 2 * g_electron * exp(-chi * beta_rad)
 * g_electron = (2*pi*m_e*kB*T_rad/h^2)^1.5
 * delta = (T_e/T_rad) * exp(chi * (beta_rad - beta_electron))  for chi >= chi_0
 * beta_rad = 1/(kB*T_rad), beta_electron = 1/(kB*T_e)
 */
/* Forward decls for the LUMINA_ION_NT channel (defined with the frozen-in
 * machinery below): per-pair RR(+DR) coefficient and the registered
 * non-thermal ionization rate [ionizations/s/cm^3]. */
static double frozenin_alpha_rr(AtomicData *atom, int ip, int ip_next, double T);
static const double *g_nt_ioniz_rate;
static int g_nt_ioniz_n;
static int g_simul_on;   /* tentative; = -1 definition at the SIMUL module */

/* ============================================================================
 * [ARTIS-PARITY R1] true rate-statistical-equilibrium ionization closure.
 *
 * Replaces the interim B2 pin (LTE Saha at T_e, W=1) with the ARTIS rate-SE
 * adjacent-ion balance (nltepop.cc nltepop_matrix_add_ionisation:563-619):
 *   n(X+1)*n_e / n(X) = [Gamma_phot + n_e*C_ion] / [alpha_rad + n_e*C_3body]
 *   => r = n(X+1)/n(X) = (Gamma + n_e*C_ion) / (n_e*alpha + n_e^2*C_3body)
 * Active only under LUMINA_ARTIS_PARITY (sub-gate LUMINA_ARTIS_PARITY_R1,
 * default ON under parity), and only where the Group-C MC field is built.
 * Fail-closed to the B2 LTE-Saha pin otherwise (mirrors ARTIS's LTE start).
 *
 * Reused machinery (NO new integrator, NO fabricated rate):
 *   Gamma_phot : the SAME per-bin R_bf estimator as the NLTE matrix
 *                (compute nlte pair matrix, plasma.c:11794-11820) -- prefers the
 *                C2 transport Gamma_bf (nlte->bf_rate_estimator) per bin, falls
 *                back to the C1 dilute-BB field integral 4pi*sigma*J/(h*nu)*dnu
 *                over nlte->J_nu. Summed pop-weighted over the lower ion levels.
 *   alpha_rad  : frozenin_alpha_rr (the Milne recombination coeff w/ spin-gate),
 *                exactly the alpha the simul_ladder rate-SE closure uses (5840).
 *   C_ion/C_3b : the A4 Seaton collisional ionization + exact-DB 3-body inverse,
 *                verbatim from simul_ladder (plasma.c:5844-5865).
 * ==========================================================================*/
static NLTEConfig *g_bf_nlte_pops;   /* fwd; defined w/ bf_set_nlte_pops (~4241) */

/* R1 sub-gate: default ON under parity (the ARTIS closure), settable OFF (=0)
 * for A/B against the B2 interim pin. */
static int r1_rate_se_enabled(void) {
    return 1; /* A2-07 production owner; legacy selector is shadow-only. */
}

/* Shell has a usable transported MC field? (fail-closed gate; 1e-30 is the
 * J_nu floor set at nlte build, so >1e-25 means an actually-sampled bin).
 * [A2-05] The bf_rate_estimator scan is replaced by the canonical view's
 * validity row: a shell counts as built when any bin is VALID. */
static int parity_field_built(NLTEConfig *nlte, int s) {
    return nlte && nlte->enabled && s >= 0 &&
           nlte->radfield_view_status == RADIATION_FIELD_VIEW_OK &&
           nlte->radfield_view.J_nu && nlte->radfield_view.validity &&
           (size_t)s < nlte->radfield_view.n_shells;
}

/* Pop-weighted per-ion photoionization rate Gamma_phot(ip) [s^-1] from the
 * transported field.  [A2-05] The per-level rate is the canonical-view
 * integral (nlte_bf_gamma_canonical); the old C2 bf_rate_estimator / C1
 * dilute-BB per-bin mix is retired.  Level population fraction = Boltzmann at
 * T_e (the B3 partition functions are built at T_e,W=1 under parity, so this
 * is self-consistent).  Blocked (non-VALID) levels contribute nothing and are
 * counted on the R6 counters. */
static PopulationStatus parity_gamma_phot_checked(
        AtomicData *atom, NLTEConfig *nlte, int s, int n_shells, int ip,
        double T_e, double chi_erg, double *gamma_out) {
    if (!gamma_out || !nlte) return POP_ATOMIC_MISSING;
    *gamma_out = 0.0;
    if (T_e <= 0.0 || !isfinite(T_e)) return POP_INVALID_TE;
    if (!parity_field_built(nlte, s)) return POP_BF_STALE;
    int nfb = nlte->n_freq_bins;
    const int use_cmfgen = atom->cmfgen_loaded &&
                           atom->cmfgen_n_freq_bins == nfb;
    int Z = atom->ion_pop_Z[ip];
    int stage = atom->ion_pop_stage[ip];
    double sigma_0 = get_bf_sigma0(Z, stage);
    if (sigma_0 <= 0.0) {
        /* [RATES-FIX F5] Z-stage is the BOUND-electron count; the Kramers
         * sigma_0 = 7.91e-18/Z_eff^2 wants the charge the ejected electron
         * sees = stage+1 (simul_ladder's convention). */
        int Zeff = rates_fix_enabled() ? (stage + 1) : (Z - stage);
        if (Zeff < 1) Zeff = 1;
        sigma_0 = 7.91e-18 / ((double)Zeff * (double)Zeff);
    }
    double Z_part = atom->partition_functions[(size_t)ip * n_shells + s];
    if (!(Z_part > 0.0) || !isfinite(Z_part)) return POP_INVALID_PARTITION;
    PopulationAtomicView av = population_atomic_view(atom);
    int lev_start = atom->level_offset[ip];
    int lev_end   = atom->level_offset[ip + 1];
    double Gamma = 0.0;
    for (int l = lev_start; l < lev_end; l++) {
        double E_lev_erg = atom->level_energy_eV[l] * EV_TO_ERG;
        double nu_thresh = (chi_erg - E_lev_erg) / H_PLANCK;
        if (nu_thresh <= 0.0) continue;
        double f_lev = 0.0;
        PopulationStatus pop_status = population_lte_level_fraction(
            &av, (size_t)ip, (size_t)l, T_e, Z_part, &f_lev);
        if (pop_status == POP_EXACT_ZERO) continue;
        if (pop_status != POP_OK) return pop_status;
        const double *sigma_row = (use_cmfgen && atom->cmfgen_has_sigma &&
                                   atom->cmfgen_has_sigma[l])
            ? &atom->cmfgen_sigma_bf[(size_t)l * (size_t)nfb] : NULL;
        BfRateResult br;
        if (nlte_bf_gamma_canonical(nlte, s, sigma_row, sigma_0,
                                    nu_thresh, &br) != 0)
            return POP_BF_MISS;
        if (br.state == BF_RATE_UNSAMPLED) return POP_BF_UNSAMPLED;
        if (br.state == BF_RATE_OUT_OF_GRID) return POP_BF_OOG;
        if (br.state == BF_RATE_STALE) return POP_BF_STALE;
        if (br.state != BF_RATE_VALID && br.state != BF_RATE_EXACT_ZERO)
            return POP_BF_MISS;
        /* nlte_bf_gamma_canonical owns the one-and-only term accounting. */
        Gamma += f_lev * br.gamma;
    }
    if (!isfinite(Gamma) || Gamma < 0.0) return POP_NONFINITE;
    *gamma_out = Gamma;
    return Gamma == 0.0 ? POP_EXACT_ZERO : POP_OK;
}

/* Rate-SE adjacent-ion ratio r = n(ip_next)/n(ip_cur). Reuses frozenin_alpha_rr
 * (Milne alpha) and the A4 Seaton C_ion/3-body (verbatim from simul_ladder).
 * Degenerate guard: Gamma=0 & recomb=0 -> keep phi_LTE_ratio (previous split);
 * recomb=0 with Gamma>0 -> cap at the 1e30 runaway bound (as the existing pin). */
static PopulationStatus parity_rate_se_ratio_checked(
        AtomicData *atom, NLTEConfig *nlte, int s, int n_shells,
        int ip_cur, int ip_next, double T_e, double n_e, double chi_erg,
        double gamma_nt_atom, double *ratio_out) {
    if (!ratio_out) return POP_ATOMIC_MISSING;
    if (!isfinite(n_e) || n_e <= 0.0) return POP_NE_NOT_CONVERGED;
    double Gamma = 0.0;
    PopulationStatus gamma_status = parity_gamma_phot_checked(
        atom, nlte, s, n_shells, ip_cur, T_e, chi_erg, &Gamma);
    if (gamma_status != POP_OK && gamma_status != POP_EXACT_ZERO)
        return gamma_status;
    if (gamma_nt_atom > 0.0) Gamma += gamma_nt_atom;   /* ARTIS NT channel (additive) */
    double alpha = frozenin_alpha_rr(atom, ip_cur, ip_next, T_e);
    double num = Gamma;                    /* + n_e*C_ion below            */
    double den = n_e * alpha;              /* + n_e^2*C_3body below (=R_rec)*/
    if (T_e > 0.0 && n_e > 0.0) {
        int has_upper = (ip_next >= 0 &&
            atom->level_offset[ip_next + 1] > atom->level_offset[ip_next]);
        double u = (chi_erg > 0.0) ? chi_erg / (K_BOLTZMANN * T_e) : 0.0;
        if (u > 0.0 && u < 700.0 && has_upper) {
            int st = atom->ion_pop_stage[ip_cur];
            int zeff = st + 1; if (zeff < 1) zeff = 1;
            double sig = 7.91e-18 / ((double)zeff * (double)zeff);
            double g_col = (st <= 0) ? 0.1 : (st == 1) ? 0.2 : 0.3;
            int glo = atom->level_g[atom->level_offset[ip_cur]];
            int gup = atom->level_g[atom->level_offset[ip_next]];
            if (glo > 0 && gup > 0) {
                const double SIM_SAHACONST = 2.0706659e-16;
                double C_ion = n_e * 1.55e13 / sqrt(T_e) * g_col * sig *
                               exp(-u) / u;
                double C_rec = n_e * n_e * SIM_SAHACONST *
                               ((double)glo / (double)gup) * 1.55e13 *
                               g_col * sig * K_BOLTZMANN / (T_e * chi_erg);
                if (isfinite(C_ion) && C_ion > 0.0) num += C_ion;
                if (isfinite(C_rec) && C_rec > 0.0) den += C_rec;
            }
        }
    }
    if (!isfinite(num) || !isfinite(den) || num < 0.0 || den < 0.0)
        return POP_NONFINITE;
    if (num == 0.0 && den == 0.0) return POP_RANK_INCOMPLETE;
    if (den == 0.0) return POP_RANK_INCOMPLETE;
    double r = num / den;
    if (!isfinite(r) || r < 0.0) return POP_NONFINITE;
    *ratio_out = r;
    return r == 0.0 ? POP_EXACT_ZERO : POP_OK;
}

/* steady-state nebular-Saha ion partition for ONE shell (all elements). Extracted
 * so the coupled Newton can reconcile only the shells it does NOT own. */
static PopulationStatus compute_ion_populations_shell(
        AtomicData *atom, PlasmaState *plasma, int s, int n_shells) {
    if (!atom || !plasma || !plasma->T_e || !plasma->n_electron ||
        s < 0 || s >= n_shells)
        return POP_ATOMIC_MISSING;
    if (!isfinite(plasma->T_e[s]) || plasma->T_e[s] <= 0.0)
        return POP_INVALID_TE;
    if (!isfinite(plasma->n_electron[s]) || plasma->n_electron[s] <= 0.0)
        return POP_NE_NOT_CONVERGED;
    /* LUMINA_ION_NT=1: ARTIS non-thermal ionization channel in the ION-STAGE
     * balance (this function OWNS the stages; the NLTE matrix at 8245 only
     * moves levels). Lucy-Mazzali nebular Saha is the photo/recomb equilibrium
     * ratio Gamma_photo/(alpha n_e), so the Spencer-Fano channel enters
     * ADDITIVELY: ratio += Gamma_nt/(alpha(T_e) n_e), Gamma_nt = eta*dep/
     * (n_atom*W_ion) per atom (gamma_deposition_compute_nonthermal), alpha =
     * RR(+DR when LUMINA_FROZENIN_DR=1). Dominant only in the dilute outer
     * (inner suppressed by n_e and the strong phi_neb term) — keeps the hot
     * partially-ionized Fe IV/V state CMFGEN reaches at the thin edge
     * (reference_artis_nonthermal_outer). */
    static int ion_nt = -1;
    if (ion_nt < 0) { const char *e = getenv("LUMINA_ION_NT");
                      ion_nt = (e && atoi(e)) ? 1 : 0; }
    double gamma_nt_atom = 0.0;
    if (ion_nt && g_nt_ioniz_rate && s < g_nt_ioniz_n) {
        double natom = 0.0;
        for (int e = 0; e < atom->n_elements; e++)
            natom += atom->abundances[e * n_shells + s] * plasma->rho[s] /
                     (atom->element_mass_amu[e] * AMU);
        if (natom > 0.0) gamma_nt_atom = g_nt_ioniz_rate[s] / natom;
    }
    /* [ARTIS-PARITY R1] rate-SE closure decision for this shell (once): active
     * under parity+R1 when the transported MC field is built for this shell;
     * else fail-closed to the B2 LTE-Saha pin below. */
    int r1_on   = g_bf_nlte_pops && r1_rate_se_enabled();
    int r1_use  = r1_on && parity_field_built(g_bf_nlte_pops, s);
    for (int e = 0; e < atom->n_elements; e++) {
        int Z_elem = atom->element_Z[e];
        double mass_amu = atom->element_mass_amu[e];
        int ip_start = atom->elem_ion_offset[e];
        int ip_end   = atom->elem_ion_offset[e + 1];
        int n_pops   = ip_end - ip_start;
        if (n_pops <= 0) continue;

        /* Z-INERT owns this decision before any partition/rate lookup.  A
         * topology-only element has no physical rate equation to solve, and
         * must not be rejected merely because its ghost ladder has no BF
         * field or ionization data. */
        int inactive = lumina_zinert_element_inactive(atom, e, n_shells);
        if (inactive) {
            for (int ip = ip_start; ip < ip_end; ip++)
                atom->ion_number_density[(size_t)ip * n_shells + s] = 0.0;
            continue;
        }

        {
            double T_e   = plasma->T_e[s];
            /* Diagnostic-only legacy formula. The checked BF view below is
             * the sole production supplier and fails closed when unavailable. */
            double T_rad = T_e;
            double W = 1.0;
            double n_e   = plasma->n_electron[s];
            double rho   = plasma->rho[s];
            double abund = atom->abundances[e * n_shells + s];

            double n_element = (abund * rho) / (mass_amu * AMU);

            /* g_electron = (2*pi*m_e*kB*T_rad/h^2)^1.5 */
            double g_electron = pow(2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_rad
                                     / (H_PLANCK * H_PLANCK), 1.5);

            double beta_rad = 1.0 / (K_BOLTZMANN * T_rad);
            double beta_electron = 1.0 / (K_BOLTZMANN * T_e);

            if (!isfinite(n_element) || n_element < 0.0)
                return POP_NONFINITE;
            double *ratios = (double *)calloc((size_t)n_pops, sizeof(double));
            if (!ratios) return POP_SOLVE_FAILED;

            for (int k = 0; k < n_pops - 1; k++) {
                int ip_cur  = ip_start + k;
                int ip_next = ip_start + k + 1;
                /* Absolute ion stage of ip_cur. Equals k only when the element's ion
                 * ladder starts at neutral. Ti/Mn dropped their neutral stage from the
                 * ref, so they start at ion 1; keying ionizE/zeta on k would query the
                 * absent ion-0 energy -> find_ioniz_energy returns the 1e10 sentinel ->
                 * the whole element collapses into a level-less stage. Use the real stage. */
                int stage = atom->ion_pop_stage[ip_cur];

                /* Dilute partition functions (W-weighted, consistent with level pops) */
                double Z_cur  = atom->partition_functions[ip_cur  * n_shells + s];
                double Z_next = atom->partition_functions[ip_next * n_shells + s];
                if (!isfinite(Z_cur) || Z_cur <= 0.0 ||
                    !isfinite(Z_next) || Z_next <= 0.0) {
                    free(ratios);
                    return POP_INVALID_PARTITION;
                }

                double chi_eV  = find_ioniz_energy(atom, Z_elem, stage);
                double chi_erg = chi_eV * EV_TO_ERG;

                /* Decomposed phi_neb to avoid 0*inf cancellation:
                 *   Original:  phi_neb = phi_lte * W * (zeta*delta + W*(1-zeta)) * sqrt(Te/Trad)
                 *              where phi_lte=exp(-chi*beta_rad) and delta has exp(+chi*(beta_rad-beta_e))
                 *              -> at low T_rad+high chi, phi_lte->0 AND delta->inf simultaneously => NaN.
                 *   Identity:  phi_lte * delta = (Te/Trad) * phi_LTE_at_Te
                 *              (the exp(-chi*beta_rad) and exp(+chi*beta_rad) cancel analytically)
                 *   Each phi_LTE_* is a single exp(negative): underflows safely to 0, never inf. */
                double prefactor = (Z_next / Z_cur) * 2.0 * g_electron;
                double phi_LTE_at_Trad = prefactor * exp(-chi_erg * beta_rad);
                double phi_LTE_at_Te   = prefactor * exp(-chi_erg * beta_electron);

                double phi_neb;
                if (artis_parity_enabled() && T_e > 0.0) {
                    /* [ARTIS-PARITY B2/B3] ionization closure at T_e, undiluted.
                     * ARTIS runs FORCE_SAHA_ION_BALANCE=false (ion fractions are
                     * the rate-SE solved output) with the LTE Saha REFERENCE at
                     * T_e (ltepop.cc). Lumina's absolute-ionization pin (the value
                     * the NLTE conservation row normalizes to) is the nebular
                     * Lucy-Mazzali Saha at (T_rad, W, zeta). Under parity, evaluate
                     * that pin as pure LTE Saha at T_e with W=1 (g_electron at T_e,
                     * matching the T_e partition functions from B3) — the ion SPLIT
                     * inside each NLTE pair stays the matrix solution; only the
                     * number-conserving total is set here. */
                    double g_e_Te = pow(2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN *
                                        T_e / (H_PLANCK * H_PLANCK), 1.5);
                    phi_neb = (Z_next / Z_cur) * 2.0 * g_e_Te *
                              exp(-chi_erg * beta_electron);
                } else {
                    double zeta = interpolate_zeta(atom, Z_elem, stage, T_rad);
                    double sqrt_te_tr = sqrt(T_e / T_rad);
                    phi_neb = W * sqrt_te_tr *
                        (zeta * (T_e / T_rad) * phi_LTE_at_Te +
                         W * (1.0 - zeta) * phi_LTE_at_Trad);
                    phi_neb = apply_ml_phi_neb_correction(phi_neb, Z_elem, stage, T_e, T_rad);
                    phi_neb = apply_twocomp_lock(phi_neb, phi_LTE_at_Te, Z_elem, stage, W);
                }

                /* PROBE (LUMINA_OUTER_ION_BOOST=<factor>,<smin>): force the outer
                 * (s>=smin) more ionised by boosting phi_neb, to TEST whether higher
                 * ionisation drops the cooling → T_e runs to the hot branch (is the
                 * outer ionisation the lever?). Diagnostic, not a fix. */
                { static double ib_f = -1.0; static int ib_s = 1000000;
                  if (ib_f < 0.0) { const char *e = getenv("LUMINA_OUTER_ION_BOOST");
                    if (e) { double f=0; int sm=1000000; sscanf(e,"%lf,%d",&f,&sm);
                             ib_f = (f>0)?f:1.0; ib_s = sm; } else ib_f = 1.0; }
                  if (ib_f != 1.0 && s >= ib_s) phi_neb *= ib_f; }

                /* ratio n_{i+1}/n_i */
                /* The legacy nebular/Saha value above is a shadow diagnostic.
                 * The only physical supplier is the checked canonical BF view. */
                (void)phi_neb;
                if (!r1_use) {
                    free(ratios);
                    return POP_BF_STALE;
                }
                double ratio = 0.0;
                PopulationStatus ratio_status = parity_rate_se_ratio_checked(
                    atom, g_bf_nlte_pops, s, n_shells, ip_cur, ip_next,
                    T_e, n_e, chi_erg, gamma_nt_atom, &ratio);
                if (ratio_status != POP_OK && ratio_status != POP_EXACT_ZERO) {
                    if (g_bf_nlte_pops->population_error_count == 0)
                        g_bf_nlte_pops->population_first_error = ratio_status;
                    g_bf_nlte_pops->population_error_count++;
                    population_counter_note(
                        &g_bf_nlte_pops->population_counters, ratio_status);
                    free(ratios);
                    return ratio_status;
                }
                ratios[k] = ratio;
            }

            /* Same adjacent-stage conservation equation, evaluated as a
             * log-sum-exp ladder so overflow is not repaired by a 1e30 cap. */
            double log_product = 0.0;
            double max_log_weight = 0.0;
            for (int k = 0; k < n_pops - 1; k++) {
                if (ratios[k] == 0.0 || !isfinite(log_product))
                    log_product = -INFINITY;
                else
                    log_product += log(ratios[k]);
                ratios[k] = log_product;
                if (log_product > max_log_weight)
                    max_log_weight = log_product;
            }
            long double sum_scaled = expl(-max_log_weight);
            for (int k = 0; k < n_pops - 1; k++) {
                if (isfinite(ratios[k]))
                    sum_scaled += expl(ratios[k] - max_log_weight);
            }
            if (!isfinite((double)sum_scaled) || sum_scaled <= 0.0L) {
                free(ratios);
                return POP_NONFINITE;
            }
            double n_0 = n_element *
                exp(-max_log_weight) / (double)sum_scaled;
            if (!isfinite(n_0) || n_0 < 0.0) {
                free(ratios);
                return POP_NONFINITE;
            }
            atom->ion_number_density[ip_start * n_shells + s] = n_0;
            for (int k = 0; k < n_pops - 1; k++) {
                double n_ion = !isfinite(ratios[k]) ? 0.0 :
                    n_element * exp(ratios[k] - max_log_weight) /
                    (double)sum_scaled;
                if (!isfinite(n_ion) || n_ion < 0.0) {
                    free(ratios);
                    return POP_NONFINITE;
                }
                atom->ion_number_density[(ip_start + k + 1) * n_shells + s] =
                    n_ion;
            }

            free(ratios);
        }
    }
    return POP_OK;
}

static PopulationStatus compute_ion_populations(
        AtomicData *atom, PlasmaState *plasma, int n_shells) {
    /* LUMINA_RADEQ_SIMUL owns the ion partition; skip the nebular rewrite. */
    if (g_simul_on == 1) return POP_OK;
    if (artis_parity_enabled()) {
        static int b2_banner = 0;
        if (!b2_banner) {
            printf("  [A2-07] ionization closure = canonical BF rates at T_e "
                   "(number-conserving adjacent-stage solve)\n");
            b2_banner = 1;
        }
    }
    init_ml_phi_neb_correction();
    init_zeta_override();
    init_twocomp_lock();
    /* [ARTIS-PARITY R1] tally rate-driven vs fail-closed-LTE shells for this pass. */
    int r1_on = g_bf_nlte_pops && r1_rate_se_enabled();
    long r1_rate_n = 0, r1_lte_n = 0;
    for (int s = 0; s < n_shells; s++) {
        if (r1_on) {
            if (parity_field_built(g_bf_nlte_pops, s)) r1_rate_n++;
            else                                       r1_lte_n++;
        }
        PopulationStatus status =
            compute_ion_populations_shell(atom, plasma, s, n_shells);
        if (status != POP_OK) return status;
    }
    if (r1_on)
        printf("  [ARTIS-PARITY R1] rate-SE closure: %ld shells rate-driven, "
               "%ld blocked-no-view\n", r1_rate_n, r1_lte_n);
    return POP_OK;
}

/* Task #072 Step 4c: Compute electron density (iterative)
 * Uses the correct TARDIS nebular Saha formula with TARDIS-style damped iteration:
 *   n_e_new_damped = 0.5 * (n_e_computed + n_e_old)
 *   convergence threshold: 5% (TARDIS default)
 *   max iterations: 100 (TARDIS default) */
static int compute_electron_density(AtomicData *atom, PlasmaState *plasma,
                                    int n_shells) {
    init_ml_phi_neb_correction();
    init_zeta_override();
    init_twocomp_lock();
    /* ★2026-08-07: 이 함수는 네 곳에서 침묵하며 -1 을 냈다.  상위는 POP_NE_NOT_CONVERGED
     * 하나만 보므로 "수렴 실패" 와 "입력이 애초에 비유한" 이 구분되지 않았다.
     * 실패 경로에서만 찍는다. */
#define NE_FAIL(fmt, ...) do {                                                  \
        fprintf(stderr, "[A2-07][n_e][FATAL] shell %d: " fmt "\n",              \
                s, ##__VA_ARGS__);                                              \
        return -1;                                                              \
    } while (0)
    for (int s = 0; s < n_shells; s++) {
        double n_e = plasma->n_electron[s];
        if (!isfinite(n_e) || n_e <= 0.0)
            NE_FAIL("seed n_e invalid: %.17g", n_e);

        int shell_converged = 0;
        double last_new = 0.0, last_old = 0.0;
        for (int iteration = 0; iteration < 100; iteration++) {
            double n_e_old = n_e;

            /* The same checked ion-rate helper owns both the n_e iteration and
             * the final ion population solve; no independent selector/fallback. */
            PopulationStatus ion_status =
                compute_ion_populations_shell(atom, plasma, s, n_shells);
            if (ion_status != POP_OK)
                NE_FAIL("ion populations failed at iter %d: %s (n_e=%.6e)",
                        iteration, population_status_name(ion_status), n_e);

            /* Sum electron density: n_e_new = sum(ion_stage * n_ion) */
            double n_e_new = 0.0;
            for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                int charge = atom->ion_pop_stage[ip];
                double n_ion_contrib = atom->ion_number_density[ip * n_shells + s];
                if (isfinite(n_ion_contrib) && n_ion_contrib > 0.0)
                    n_e_new += charge * n_ion_contrib;
            }
            if (!isfinite(n_e_new) || n_e_new <= 0.0)
                NE_FAIL("sum(charge*n_ion) invalid at iter %d: %.17g", iteration, n_e_new);
            last_new = n_e_new; last_old = n_e_old;

            /* TARDIS-style damped update: n_e = 0.5 * (n_e_new + n_e_old) */
            n_e = 0.5 * (n_e_new + n_e_old);
            plasma->n_electron[s] = n_e;

            /* TARDIS convergence: 5% relative threshold */
            if (n_e_old > 0.0 && fabs(n_e_new - n_e_old) / n_e_old < 0.05) {
                shell_converged = 1;
                break;
            }
        }
        if (!shell_converged)
            NE_FAIL("no convergence in 100 iters: last n_e_new=%.6e n_e_old=%.6e "
                    "rel=%.3e (threshold 0.05)", last_new, last_old,
                    last_old > 0.0 ? fabs(last_new - last_old) / last_old : -1.0);
    }
    return 0;
#undef NE_FAIL
}

/* Task #072 Step 4d: Compute tau_sobolev from ion populations */
/* Per-Z line-opacity zero-out via LUMINA_OPACITY_SKIP_Z (comma list, e.g. "8,6"). */
static int opacity_skip_z[100];
static int opacity_skip_z_init = 0;
static void opacity_skip_z_load(void) {
    if (opacity_skip_z_init) return;
    opacity_skip_z_init = 1;
    const char *e = getenv("LUMINA_OPACITY_SKIP_Z");
    if (!e || !*e) return;
    char buf[256]; strncpy(buf, e, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    char *tok = strtok(buf, ", \t");
    while (tok) {
        int z = atoi(tok);
        if (z > 0 && z < 100) opacity_skip_z[z] = 1;
        tok = strtok(NULL, ", \t");
    }
    printf("  [OPACITY] LUMINA_OPACITY_SKIP_Z active (line tau zeroed): ");
    for (int i = 1; i < 100; i++) if (opacity_skip_z[i]) printf("Z=%d ", i);
    printf("\n");
}
/* A2: was non-static but unreferenced (dead-code audit Tier 3.3b finding).
 * Kept as a static helper; redundant with the inline lookup at
 * compute_tau_sobolev:line 636 below. */
__attribute__((unused))
static int opacity_skip_z_is_masked(int Z) {
    opacity_skip_z_load();
    return (Z > 0 && Z < 100 && opacity_skip_z[Z]);
}

static void compute_tau_sobolev(AtomicData *atom, PlasmaState *plasma,
                                 OpacityState *opacity, double time_explosion) {
    int n_lines = opacity->n_lines;
    int n_shells = opacity->n_shells;
    opacity_skip_z_load();

    for (int line = 0; line < n_lines; line++) {
        int Z         = atom->line_atomic_number[line];
        int ion_stage = atom->line_ion_number[line];
        int lev_lower = atom->line_level_lower[line];
        int lev_upper = atom->line_level_upper[line];
        double f_lu   = atom->line_f_lu[line];
        double lam_cm = atom->line_wavelength_cm[line];

        /* Diagnostic: zero-out lines for masked Z */
        if (Z > 0 && Z < 100 && opacity_skip_z[Z]) {
            for (int s = 0; s < n_shells; s++) {
                opacity->tau_sobolev[line * n_shells + s] = 0.0;
                if (opacity->tau_validity)
                    opacity->tau_validity[line * n_shells + s] = A208_EXACT_ZERO;
            }
            continue;
        }

        /* Find ion population index */
        int ip = find_ion_pop_idx(atom, Z, ion_stage);
        if (ip < 0) {
            for (int s = 0; s < n_shells; s++) {
                opacity->tau_sobolev[line * n_shells + s] = 0.0;
                if (opacity->tau_validity)
                    opacity->tau_validity[line * n_shells + s] = A208_MISS;
            }
            continue;
        }

        /* Find level data for lower and upper */
        int lev_start = atom->level_offset[ip];
        int lev_end   = atom->level_offset[ip + 1];

        /* Search for lower and upper levels */
        int lower_idx = -1, upper_idx = -1;
        for (int l = lev_start; l < lev_end; l++) {
            if (atom->level_num[l] == lev_lower) lower_idx = l;
            if (atom->level_num[l] == lev_upper) upper_idx = l;
            if (lower_idx >= 0 && upper_idx >= 0) break;
        }

        if (lower_idx < 0 || upper_idx < 0) {
            for (int s = 0; s < n_shells; s++) {
                opacity->tau_sobolev[line * n_shells + s] = 0.0;
                if (opacity->tau_validity)
                    opacity->tau_validity[line * n_shells + s] = A208_MISS;
            }
            continue;
        }

        int g_lower    = atom->level_g[lower_idx];
        int g_upper    = atom->level_g[upper_idx];

        int element_inactive = 0;
        for (int e = 0; e < atom->n_elements; e++) {
            if (atom->element_Z[e] == Z) {
                element_inactive = lumina_zinert_element_inactive(atom, e, n_shells);
                break;
            }
        }
        for (int s = 0; s < n_shells; s++) {
            double n_ion = atom->ion_number_density[ip * n_shells + s];
            if (element_inactive) {
                opacity->tau_sobolev[(size_t)line * n_shells + s] = 0.0;
                if (opacity->tau_validity)
                    opacity->tau_validity[(size_t)line * n_shells + s] = A208_EXACT_ZERO;
                continue;
            }
            if (!plasma || !plasma->T_e || s >= plasma->n_shells ||
                !isfinite(plasma->T_e[s]) || plasma->T_e[s] <= 0.0) {
                /* A2-07 section 3.1: an active LTE population has no
                 * radiation-temperature fallback.  Preserve the exact-zero
                 * Z-inert short circuit above, but fail closed for active
                 * lines instead of dereferencing a missing T_e view. */
                opacity->tau_sobolev[(size_t)line * n_shells + s] = NAN;
                if (opacity->tau_validity)
                    opacity->tau_validity[(size_t)line * n_shells + s] = A208_INVALID_TE;
                continue;
            }
            double T_e = plasma->T_e[s];
            double Z_part = atom->partition_functions[ip * n_shells + s];

            PopulationAtomicView av = population_atomic_view(atom);
            double f_lower = 0.0, f_upper = 0.0;
            PopulationStatus ps_lo = population_lte_level_fraction(
                &av, (size_t)ip, (size_t)lower_idx, T_e, Z_part, &f_lower);
            PopulationStatus ps_up = population_lte_level_fraction(
                &av, (size_t)ip, (size_t)upper_idx, T_e, Z_part, &f_upper);
            if ((ps_lo != POP_OK && ps_lo != POP_EXACT_ZERO) ||
                (ps_up != POP_OK && ps_up != POP_EXACT_ZERO)) {
                opacity->tau_sobolev[(size_t)line * n_shells + s] = NAN;
                if (opacity->tau_validity)
                    opacity->tau_validity[(size_t)line * n_shells + s] = A208_INVALID_POPULATION;
                continue;
            }
            double n_lower = n_ion * f_lower;
            double n_upper = n_ion * f_upper;

            A208ValueView tau = a208_signed_sobolev(
                SOBOLEV_COEFF, f_lu, lam_cm, time_explosion,
                n_lower, n_upper, g_lower, g_upper,
                opacity->tau_required_generation);
            opacity->tau_sobolev[line * n_shells + s] = tau.value;
            if (opacity->tau_validity)
                opacity->tau_validity[line * n_shells + s] = tau.validity;
        }
    }
}

#ifdef LUMINA_FROZEN_ORACLE
void lumina_oracle_compute_tau_sobolev(AtomicData *atom, PlasmaState *plasma,
                                       OpacityState *opacity,
                                       double time_explosion) {
    compute_tau_sobolev(atom, plasma, opacity, time_explosion);
}
#endif

/* Probe-B fix (task #29): write the NLTE-solved ion stage back into
 * atom->ion_number_density so the BULK (non-NLTE-tracked) line opacity built by
 * compute_tau_sobolev uses the rate-solved ionization instead of the
 * over-ionized nebular phi_neb. scripts/probe_b_three_way_ionization.py showed
 * the NLTE solve yields the correct II-dominant iron curtain at line-forming
 * shells (Fe fIII 0.06-0.31) while phi_neb stays III-dominant (0.98-0.998), and
 * opacity was discarding the NLTE split. Each pair is rescaled to its EXISTING
 * nebular pair-total (which the NLTE conservation already preserves), so only the
 * II/III split changes — abundances and untracked stages are untouched. The O
 * triplet overlap pairs (indices 14,15: slot 29 shared) are skipped. After the
 * write-back the bulk tau is rebuilt; callers then re-apply the per-line NLTE tau
 * override on top. Gated by LUMINA_NLTE_OPACITY_IONSTAGE=1 (default off = no-op).
 * Shared by the CPU and GPU NLTE solvers. */
void nlte_writeback_ion_stage(NLTEConfig *nlte, AtomicData *atom,
                              PlasmaState *plasma, OpacityState *opacity,
                              double time_explosion, int n_shells,
                              int pairs[][2], int n_pairs) {
    const char *e = getenv("LUMINA_NLTE_OPACITY_IONSTAGE");
    if (!e || e[0] != '1') return;

    int n_writeback = 0;
    for (int p = 0; p < n_pairs; p++) {
        /* Skip any triplet-overlap pair (shares a slot with another pair):
         * Si II/III/IV, Fe II/III/IV, O I/II/III. Generic, not index-hardcoded. */
        int shares = 0;
        for (int q = 0; q < n_pairs; q++) {
            if (q == p) continue;
            if (pairs[q][0] == pairs[p][0] || pairs[q][1] == pairs[p][0] ||
                pairs[q][0] == pairs[p][1] || pairs[q][1] == pairs[p][1]) { shares = 1; break; }
        }
        if (shares) continue;
        int lo = pairs[p][0], hi = pairs[p][1];
        int ip_lo = find_ion_pop_idx(atom, nlte->nlte_Z[lo], nlte->nlte_ion[lo]);
        int ip_hi = find_ion_pop_idx(atom, nlte->nlte_Z[hi], nlte->nlte_ion[hi]);
        if (ip_lo < 0 || ip_hi < 0) continue;
        int los = nlte->nlte_ion_level_offset[lo];
        int loe = nlte->nlte_ion_level_offset[lo + 1];
        int his = nlte->nlte_ion_level_offset[hi];
        int hie = nlte->nlte_ion_level_offset[hi + 1];
        for (int s = 0; s < n_shells; s++) {
            double sum_lo = 0.0, sum_hi = 0.0;
            for (int l = los; l < loe; l++)
                sum_lo += nlte->nlte_level_populations[(size_t)l * n_shells + s];
            for (int l = his; l < hie; l++)
                sum_hi += nlte->nlte_level_populations[(size_t)l * n_shells + s];
            double tot = sum_lo + sum_hi;
            double T_neb = atom->ion_number_density[(size_t)ip_lo * n_shells + s]
                         + atom->ion_number_density[(size_t)ip_hi * n_shells + s];
            if (tot > 0.0 && T_neb > 0.0 && isfinite(tot)) {
                double new_lo = T_neb * sum_lo / tot;
                double new_hi = T_neb * sum_hi / tot;
                if (new_lo < 1e-300) new_lo = 1e-300;
                if (new_hi < 1e-300) new_hi = 1e-300;
                atom->ion_number_density[(size_t)ip_lo * n_shells + s] = new_lo;
                atom->ion_number_density[(size_t)ip_hi * n_shells + s] = new_hi;
            }
        }
        n_writeback++;
    }
    printf("  [NLTE] LUMINA_NLTE_OPACITY_IONSTAGE: wrote NLTE ion split back to "
           "ion_number_density for %d pairs; rebuilding bulk tau\n", n_writeback);
    compute_tau_sobolev(atom, plasma, opacity, time_explosion);
}

/* ============================================================ */
/* Dynamic macro-atom transition probability recomputation      */
/* ============================================================ */

static inline double beta_sobolev(double tau) {
    if (tau < 1e-6) return 1.0 - 0.5 * tau;   /* Taylor expansion */
    if (tau > 500.0) return 1.0 / tau;          /* asymptotic */
    return (1.0 - exp(-tau)) / tau;
}

static inline double planck_bnu(double T, double nu) {
    double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    if (x > 500.0) return 0.0;
    return (2.0 * H_PLANCK * nu * nu * nu / (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT))
           / (exp(x) - 1.0);
}

/* A2-17 ABI tombstone.  Callers retain this symbol while downstream plugins
 * migrate, but it deliberately has no radiation-to-material scalar path. */
void compute_electron_temperature(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                   double time_explosion, int n_shells,
                                   int self_consistent) {
    (void)gamma_dep;
    (void)time_explosion;
    (void)n_shells;
    (void)plasma;
    (void)self_consistent;
}

/* ==========================================================================
 * [ALPHA-SPINGATE] / [withParityY Y4]  SHARED SPIN-SELECTION HELPERS
 * --------------------------------------------------------------------------
 * These three helpers are the ONE definition of the spin rule that the owner
 * path frozenin_alpha_rr already implemented inline.  They were promoted here
 * (ahead of recomb_alpha_per_level, the first new user) so the three OTHER
 * recombination sites can apply the IDENTICAL predicate instead of restating
 * it.  frozenin_alpha_rr now calls them; its arithmetic is unchanged.
 *
 * THE RULE.  Radiative recombination X^{+1}(ground core, spin S_core) + e^-(1/2)
 * can only form daughter terms whose multiplicity M = 2S+1 lies in
 * {M_core - 1, M_core + 1}.  A daughter level of any other KNOWN multiplicity is
 * spin-forbidden from the ground core.  Unknown multiplicity (level_mult == 0,
 * or the ion has no M_core at all) is NEVER skipped -- conservative by design.
 *
 * DETAILED-BALANCE CAVEAT (declared, not hidden).  Gating RECOMBINATION alone
 * breaks exact Saha recovery at J = B for precisely the skipped levels: the
 * photoionization half of the pair is left intact (photoionization of a
 * spin-forbidden level to an EXCITED upper-ion core is physical and does
 * happen), so for those levels R_bf and R_rec no longer satisfy the
 * LTE detailed-balance identity and the J=B fixed point is no longer the exact
 * Saha population.  The true DB partner of that photoionization is recombination
 * from the EXCITED upper-ion cores, which Lumina does not track (every ion is
 * represented by its ground core only).  Removing the spin-forbidden
 * recombination therefore removes a channel that the code was crediting to the
 * GROUND core, at the cost of a known, bounded DB defect on those same levels.
 * That is the trade this gate makes, and it is the reason it is a GATE.
 * ========================================================================== */

/* Ground-term spin multiplicity (2S+1) of the RECOMBINING ion (Z, charge) from
 * NIST ground terms.  Used as the M_core fallback when the CMFGEN companion
 * table lacks the recombining ion's ground level -- notably Fe/Co/Ni IV are NOT
 * in cmfgen_config_lumina.yml, so their M_core (6, 5, 4) can only come from
 * here.  Returns 0 if not tabulated => that ion is not gated. */
static int spingate_core_mult(int Z, int charge) {
    switch (Z) {
    case 14: /* Si */ switch (charge) {
        case 0: return 3;  /* Si I   3p2 3P */   case 1: return 2;  /* Si II  3p 2P */
        case 2: return 1;  /* Si III 3s2 1S */   case 3: return 2;  /* Si IV  3s 2S */
        default: return 0; }
    case 16: /* S */  switch (charge) {
        case 0: return 3;  /* S I    3p4 3P */   case 1: return 4;  /* S II   3p3 4S */
        case 2: return 3;  /* S III  3p2 3P */   case 3: return 2;  /* S IV   3p 2P */
        default: return 0; }
    case 20: /* Ca */ switch (charge) {
        case 0: return 1;  /* Ca I   4s2 1S */   case 1: return 2;  /* Ca II  4s 2S */
        case 2: return 1;  /* Ca III 3p6 1S */   case 3: return 2;  /* Ca IV  3p5 2P */
        default: return 0; }
    case 26: /* Fe */ switch (charge) {
        case 0: return 5;  /* Fe I   3d6 4s2 5D */ case 1: return 6;  /* Fe II  3d6 4s a6D */
        case 2: return 5;  /* Fe III 3d6 5D */     case 3: return 6;  /* Fe IV  3d5 6S */
        case 4: return 5;  /* Fe V   3d4 5D */     case 5: return 4;  /* Fe VI  3d3 4F */
        default: return 0; }
    case 27: /* Co */ switch (charge) {
        case 0: return 4;  /* Co I   3d7 4s2 a4F */ case 1: return 3;  /* Co II  3d8 a3F */
        case 2: return 4;  /* Co III 3d7 a4F */     case 3: return 5;  /* Co IV  3d6 5D */
        case 4: return 6;  /* Co V   3d5 6S */
        default: return 0; }
    case 28: /* Ni */ switch (charge) {
        case 0: return 3;  /* Ni I   3d8 4s2 3F */ case 1: return 2;  /* Ni II  3d9 2D */
        case 2: return 3;  /* Ni III 3d8 3F */     case 3: return 4;  /* Ni IV  3d7 4F */
        case 4: return 5;  /* Ni V   3d6 5D */
        default: return 0; }
    default: return 0;
    }
}

/* M_core resolution, promoted VERBATIM from frozenin_alpha_rr (the owner path):
 * prefer the CMFGEN companion table entry for the recombining ion's GROUND level
 * (level_num == 0 => level_offset[ip_next]), fall back to the NIST table above.
 * `ip_next` < 0 or level_mult == NULL simply falls through to the table.
 * `src_out` (optional) receives "data"/"table"/"none" for banners. */
static int spingate_resolve_core_mult(AtomicData *atom, int ip_next,
                                      int Z, int core_charge,
                                      const char **src_out) {
    int M_core = 0;
    const char *src = "none";
    if (atom->level_mult && ip_next >= 0) {
        int gnd = atom->level_offset[ip_next];      /* level_num==0 = ground */
        if (gnd < atom->level_offset[ip_next + 1] && atom->level_mult[gnd] > 0) {
            M_core = atom->level_mult[gnd];
            src = "data";
        }
    }
    if (M_core == 0) {
        M_core = spingate_core_mult(Z, core_charge);
        if (M_core > 0) src = "table";
    }
    if (src_out) *src_out = src;
    return M_core;
}

/* The skip predicate, promoted VERBATIM from frozenin_alpha_rr:
 * multiplicity KNOWN (>0) and outside {M_core-1, M_core+1} => spin-forbidden.
 * Returns 0 whenever M_core is unknown or the companion table is absent. */
static int spingate_level_forbidden(AtomicData *atom, int gl, int M_core) {
    if (M_core <= 0 || !atom->level_mult) return 0;
    int Ml = atom->level_mult[gl];
    return (Ml > 0 && Ml != M_core - 1 && Ml != M_core + 1) ? 1 : 0;
}

/* [withParityY Y4] gate LUMINA_REC_SPINGATE (default OFF) for the three
 * per-level Milne RECOMBINATION distribution sites (S3 recomb_alpha_per_level,
 * S1 matrix I_rec, S2 TOPSTAGE_IV).  Independent of LUMINA_ALPHA_SPINGATE,
 * which owns the per-ion alpha_rr total.
 * FAIL-LOUD DEPENDENCY: atom->level_mult is loaded by lumina_atomic.c ONLY when
 * LUMINA_ALPHA_SPINGATE=1.  With REC_SPINGATE=1 alone, level_mult is NULL, the
 * per-level predicate can never fire and this gate is INERT -- so say so, once,
 * instead of letting a silent no-op be read as a null result. */
static int rec_spingate_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LUMINA_REC_SPINGATE");
        on = (e && atoi(e)) ? 1 : 0;
        if (on)
            printf("[REC-SPINGATE] per-level Milne RECOMBINATION restricted to "
                   "spin-allowed daughter terms (M_core +/- 1) at S3 "
                   "recomb_alpha_per_level / S1 matrix I_rec / S2 TOPSTAGE_IV; "
                   "photoionization (R_bf) NOT gated -> detailed balance at J=B "
                   "is broken for the skipped levels BY DESIGN (true DB partner "
                   "= recombination from excited, untracked upper-ion cores)\n");
    }
    return on;
}

/* One-shot inertness alarm; separate from the arming banner so it can fire from
 * whichever site first sees a live atom pointer. */
static void rec_spingate_check_data(AtomicData *atom) {
    static int warned = 0;
    if (warned || !atom) return;
    warned = 1;
    if (!atom->level_mult)
        fprintf(stderr, "[REC-SPINGATE][WARN] atom->level_mult is NULL "
                "(level_multiplicity.csv is loaded only when "
                "LUMINA_ALPHA_SPINGATE=1) -> LUMINA_REC_SPINGATE is INERT, "
                "ZERO levels will be skipped\n");
}

/* bf recombination-cascade channel (LUMINA_MACROATOM_BF). A macro-atom that was
 * bf-activated on an upper-ion ground level (ionized_ground) recombines into a
 * lower-ion level and continues cascading down that ion's lines. */

/* Per-target spontaneous radiative-recombination summand alpha_sp(i->gl) [cm^3/s]
 * for the cascade. Identical Milne integral as frozenin_alpha_rr but with the
 * recombining UPPER-ion statistical weight taken from the specific activation
 * level i (g_i = g(ionized_ground)) instead of the full partition function, and
 * NO dielectronic term (radiative recomb only). ip_lower = ion-pop index of the
 * recombined (lower) ion that owns target level gl. */
static double recomb_alpha_per_level(AtomicData *atom, int ip_lower, int gl,
                                     double g_i, double T) {
    if (!atom->cmfgen_loaded || T <= 0.0 || g_i <= 0.0) return 0.0;
    if (!atom->cmfgen_has_sigma || !atom->cmfgen_has_sigma[gl]) return 0.0;
    int Z = atom->ion_pop_Z[ip_lower];
    int stage = atom->ion_pop_stage[ip_lower];
    /* [withParityY Y4 (a) S3] spin-selection on the RECOMBINATION target level.
     * The recombining (source) ion is stage+1, so its ground multiplicity is
     * M_core.  Spin-forbidden target => this level receives no recombination.
     * See the DB caveat in the shared-helper block above. */
    if (rec_spingate_enabled()) {
        rec_spingate_check_data(atom);
        int ip_next = find_ion_pop_idx(atom, Z, stage + 1);
        int M_core = spingate_resolve_core_mult(atom, ip_next, Z, stage + 1, NULL);
        if (spingate_level_forbidden(atom, gl, M_core)) return 0.0;
    }
    double chi_ion_eV = find_ioniz_energy(atom, Z, stage);
    if (chi_ion_eV <= 0.0 || chi_ion_eV >= 1e9) return 0.0;
    double chi_ion_erg = chi_ion_eV * EV_TO_ERG;
    double chi_l = chi_ion_erg - atom->level_energy_eV[gl] * EV_TO_ERG;
    if (chi_l <= 0.0) return 0.0;
    double nu_th = chi_l / H_PLANCK;
    double lam3 = pow(H_PLANCK * H_PLANCK /
                      (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T), 1.5);
    int nfreq = atom->cmfgen_n_freq_bins;
    double log_numin = log(atom->cmfgen_nu_min);
    double d_log_nu = (log(atom->cmfgen_nu_max) - log_numin) / nfreq;
    const double *sigma_row = &atom->cmfgen_sigma_bf[(size_t)gl * (size_t)nfreq];
    const int rfix_alpha = rates_fix_enabled();   /* [RATES-FIX F3] */
    double Rbf = 0.0;
    for (int bb = 0; bb < nfreq; bb++) {
        double log_nu_lo = log_numin + bb * d_log_nu;
        double nu_c = exp(log_nu_lo + 0.5 * d_log_nu);
        if (nu_c < nu_th) continue;
        double sig = sigma_row[bb];
        if (sig <= 0.0) continue;
        double x = H_PLANCK * nu_c / (K_BOLTZMANN * T);
        double B;
        if (rfix_alpha) {
            /* [RATES-FIX F3] same fused Milne exponent as frozenin_alpha_rr:
             * the x>700 skip zeroed Rbf and exp(chi_l/kT) overflowed => NaN. */
            double y = (H_PLANCK * nu_c - chi_l) / (K_BOLTZMANN * T);
            B = (2.0 * H_PLANCK * nu_c * nu_c * nu_c /
                 (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) *
                (exp(-y) / (-expm1(-x)));
        } else {
            if (x > 700.0) continue;
            B = (2.0 * H_PLANCK * nu_c * nu_c * nu_c /
                 (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) / expm1(x);
        }
        double dnu = exp(log_nu_lo + d_log_nu) - exp(log_nu_lo);
        Rbf += 4.0 * M_PI_VAL * B * sig / (H_PLANCK * nu_c) * dnu;
    }
    if (rfix_alpha)
        return Rbf * lam3 * (double)atom->level_g[gl] / (2.0 * g_i);
    return Rbf * lam3 * (double)atom->level_g[gl] / (2.0 * g_i)
           * exp(chi_l / (K_BOLTZMANN * T));
}

/* Build the recomb-cascade topology once (CSR keyed by SOURCE upper-ion global
 * level). Source i = ground level of (Z, stage+1); targets j = the lower ion's
 * (Z, stage) levels carrying a bf cross-section. Plasma-independent, so it is
 * built once and reused across iterations. Gated by LUMINA_MACROATOM_BF. */
static void build_recomb_topology(AtomicData *atom, OpacityState *opacity,
                                  int n_shells) {
    static int bf_mode = -1;
    if (bf_mode < 0) {
        const char *e = getenv("LUMINA_MACROATOM_BF");
        bf_mode = (e && atoi(e) > 0) ? atoi(e) : 0;
        /* [ARTIS-PARITY D1/D3] promote the bf-activation macro-atom: under parity a
         * bf absorption activates the macro-atom on the upper-ion level and cascades
         * down the recomb topology (ion-changing INTERNALDOWNLOWER subset). Env still
         * wins when set; radiative-recomb CONTINUUM photon emission (is_emit) is the
         * deferred residual (see report). Gate OFF => recomb_block_refs stays NULL. */
        if (bf_mode <= 0 && artis_parity_enabled()) bf_mode = 1;
    }
    if (bf_mode <= 0) return;                  /* gate off => baseline */
    if (opacity->recomb_block_refs) return;    /* built once */
    if (!atom->cmfgen_loaded || !atom->cmfgen_has_sigma) return;

    /* [MA-RADRECOMB inc2] When LUMINA_MA_RADRECOMB=1 and the upper-ion TARGET map is
     * loaded, flag recomb entries whose lower-ion target level j photoionizes TO the
     * recomb source (consuming-verification: ma_rr_target[j] == source). Flagged
     * entries emit a bf continuum photon on the device (transition_type -5) instead of
     * INTERNALDOWNLOWER. Gate OFF or map absent => is_emit stays all 0 (byte-identical
     * to the parity baseline). */
    static int rr_mode = -1;
    if (rr_mode < 0) {
        const char *er = getenv("LUMINA_MA_RADRECOMB");
        rr_mode = (er && atoi(er) != 0) ? 1 : 0;
    }
    const int *rr_tgt = (rr_mode && atom->ma_rr_loaded) ? atom->ma_rr_target : NULL;

    int n_levels = opacity->n_macro_levels;

    /* ground level of the NEXT-higher ion for each ion pop = recomb activation
     * (source) level (same construction as compute_bf_opacity). Also record the
     * upper ion's pop index for the stage-IV source-coverage extension below. */
    int *ionized_ground = (int *)malloc(atom->n_ion_pops * sizeof(int));
    int *ionized_up_ip  = (int *)malloc(atom->n_ion_pops * sizeof(int));
    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        ionized_ground[ip] = -1;
        ionized_up_ip[ip]  = -1;
        int Z = atom->ion_pop_Z[ip];
        int next = atom->ion_pop_stage[ip] + 1;
        for (int jp = 0; jp < atom->n_ion_pops; jp++) {
            if (atom->ion_pop_Z[jp] == Z && atom->ion_pop_stage[jp] == next) {
                ionized_up_ip[ip] = jp;
                int ls = atom->level_offset[jp], le = atom->level_offset[jp + 1];
                for (int l = ls; l < le; l++)
                    if (atom->level_num[l] == 0) { ionized_ground[ip] = l; break; }
                break;
            }
        }
    }

    /* ADDENDUM (Fork A): the recomb thermal EXIT is only reachable from a level
     * that carries recomb entries; the baseline source set is ONLY each upper-ion
     * GROUND (level_num==0 above). The deep Co IV funnel traps packets in EXCITED
     * Co IV levels (e.g. lev144 / the 1490-1650A complex), which are NOT sources
     * -> the exit door is unreachable from where packets sit. Under
     * LUMINA_NLTE_STAGE4 (Co/Fe/Ni/... IV now SE-solved = the continuum drain),
     * extend the source set to EVERY level of a promoted stage-IV upper ion so a
     * packet on any IV level can recombine into the (III) manifold. Gate OFF =>
     * ground-only sources => byte-identical baseline. `use_all_src(ip)` is the
     * single predicate shared by the count and fill passes. */
    int stage4_bf = nlte_stage4_enabled();
    #define STAGE4_IS_PROMOTED_IV(Z) ((Z)==14||(Z)==26||(Z)==27||(Z)==28|| \
                                      (Z)==22||(Z)==24||(Z)==13)

    /* count entries per source level (upper-ion ground, or all IV levels) */
    int *cnt = (int *)calloc((size_t)n_levels + 1, sizeof(int));
    int n_recomb = 0;
    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        int up_ip = ionized_up_ip[ip];
        int use_all = stage4_bf && up_ip >= 0 &&
                      atom->ion_pop_stage[up_ip] == 3 &&
                      STAGE4_IS_PROMOTED_IV(atom->ion_pop_Z[up_ip]);
        int g0 = atom->level_offset[ip], g1 = atom->level_offset[ip + 1];
        int ndest = 0;
        for (int j = g0; j < g1; j++)
            if (atom->cmfgen_has_sigma[j]) ndest++;
        if (ndest == 0) continue;
        if (use_all) {
            int sl0 = atom->level_offset[up_ip], sl1 = atom->level_offset[up_ip + 1];
            for (int sl = sl0; sl < sl1; sl++)
                if (sl >= 0 && sl < n_levels) { cnt[sl] += ndest; n_recomb += ndest; }
        } else {
            int src = ionized_ground[ip];
            if (src >= 0 && src < n_levels) { cnt[src] += ndest; n_recomb += ndest; }
        }
    }

    int *refs = (int *)malloc(((size_t)n_levels + 1) * sizeof(int));
    int acc = 0, n_src = 0, n_src_stage4 = 0;
    for (int l = 0; l < n_levels; l++) {
        refs[l] = acc; acc += cnt[l];
        if (cnt[l] > 0) n_src++;
    }
    refs[n_levels] = acc;   /* == n_recomb */

    size_t na = (size_t)(n_recomb > 0 ? n_recomb : 1);
    int    *dest    = (int *)malloc(na * sizeof(int));
    double *nu_edge = (double *)malloc(na * sizeof(double));
    int    *is_emit = (int *)calloc(na, sizeof(int));
    int    *fill    = (int *)malloc((size_t)n_levels * sizeof(int));
    for (int l = 0; l < n_levels; l++) fill[l] = refs[l];

    /* [MA-RADRECOMB tau-gate] per-entry edge cross-section sigma_bf(dest, nu_edge),
     * baked once (shell-independent) on the fixed bf grid. Only the is_emit
     * candidates matter (others never emit). Allocated only under the rr gate. */
    double *sig_edge = rr_tgt ? (double *)calloc(na, sizeof(double)) : NULL;
    int    rr_nfb   = atom->cmfgen_n_freq_bins;
    double rr_numin = atom->cmfgen_nu_min;
    double rr_dln   = (rr_numin > 0.0 && atom->cmfgen_nu_max > rr_numin && rr_nfb > 0)
                      ? log(atom->cmfgen_nu_max / rr_numin) / rr_nfb : 0.0;

    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        int Z = atom->ion_pop_Z[ip];
        int stage = atom->ion_pop_stage[ip];
        int up_ip = ionized_up_ip[ip];
        int use_all = stage4_bf && up_ip >= 0 &&
                      atom->ion_pop_stage[up_ip] == 3 &&
                      STAGE4_IS_PROMOTED_IV(atom->ion_pop_Z[up_ip]);
        double chi_eV = find_ioniz_energy(atom, Z, stage);
        double chi_erg = (chi_eV > 0.0 && chi_eV < 1e9) ? chi_eV * EV_TO_ERG : 0.0;
        int g0 = atom->level_offset[ip], g1 = atom->level_offset[ip + 1];
        if (use_all) {
            int sl0 = atom->level_offset[up_ip], sl1 = atom->level_offset[up_ip + 1];
            for (int sl = sl0; sl < sl1; sl++) {
                if (sl < 0 || sl >= n_levels) continue;
                if (cnt[sl] > 0) n_src_stage4++;
                for (int j = g0; j < g1; j++) {
                    if (!atom->cmfgen_has_sigma[j]) continue;
                    int k = fill[sl]++;
                    dest[k] = j;
                    double chi_l = chi_erg - atom->level_energy_eV[j] * EV_TO_ERG;
                    nu_edge[k] = (chi_l > 0.0) ? chi_l / H_PLANCK : 0.0;
                    /* [MA-RADRECOMB inc2] emit when j's CMFGEN photoion target == source sl. */
                    is_emit[k] = (rr_tgt && nu_edge[k] > 0.0 && rr_tgt[j] == sl) ? 1 : 0;
                    if (sig_edge && is_emit[k] && rr_dln > 0.0 && atom->cmfgen_sigma_bf) {
                        int b0 = (int)floor(log(nu_edge[k] / rr_numin) / rr_dln);
                        if (b0 < 0) b0 = 0;
                        const double *srow = &atom->cmfgen_sigma_bf[(size_t)j * rr_nfb];
                        double se = 0.0;
                        for (int bb = b0; bb < rr_nfb && bb <= b0 + 5; bb++)
                            if (srow[bb] > se) se = srow[bb];
                        sig_edge[k] = se;   /* [MA-RADRECOMB tau-gate] edge sigma_bf */
                    }
                }
            }
        } else {
            int src = ionized_ground[ip];
            if (src < 0 || src >= n_levels) continue;
            for (int j = g0; j < g1; j++) {
                if (!atom->cmfgen_has_sigma[j]) continue;
                int k = fill[src]++;
                dest[k] = j;
                double chi_l = chi_erg - atom->level_energy_eV[j] * EV_TO_ERG;
                nu_edge[k] = (chi_l > 0.0) ? chi_l / H_PLANCK : 0.0;  /* increment 2 */
                /* [MA-RADRECOMB inc2] emit when j's CMFGEN photoion target == source. */
                is_emit[k] = (rr_tgt && nu_edge[k] > 0.0 && rr_tgt[j] == src) ? 1 : 0;
                if (sig_edge && is_emit[k] && rr_dln > 0.0 && atom->cmfgen_sigma_bf) {
                    int b0 = (int)floor(log(nu_edge[k] / rr_numin) / rr_dln);
                    if (b0 < 0) b0 = 0;
                    const double *srow = &atom->cmfgen_sigma_bf[(size_t)j * rr_nfb];
                    double se = 0.0;
                    for (int bb = b0; bb < rr_nfb && bb <= b0 + 5; bb++)
                        if (srow[bb] > se) se = srow[bb];
                    sig_edge[k] = se;   /* [MA-RADRECOMB tau-gate] edge sigma_bf */
                }
            }
        }
    }
    #undef STAGE4_IS_PROMOTED_IV

    opacity->recomb_block_refs = refs;
    opacity->recomb_dest_level = dest;
    opacity->recomb_nu_edge    = nu_edge;
    opacity->recomb_is_emit    = is_emit;
    opacity->n_recomb          = n_recomb;
    opacity->recomb_prob = (double *)calloc(na * (size_t)n_shells, sizeof(double));
    /* [MA-RADRECOMB tau-gate] per-shell emit decision (filled each call by the
     * per-shell loop from live populations). Allocated only under the rr gate. */
    opacity->recomb_sigma_edge = sig_edge;
    opacity->recomb_emit_shell = rr_tgt
        ? (int *)calloc(na * (size_t)n_shells, sizeof(int)) : NULL;

    /* [MA-RADRECOMB iup] INTERNALUPHIGHER topology (ARTIS macroatom.cc:165-185).
     * Mirror of the recomb (DOWN) block in the UP direction: every lower-ion macro
     * level `lev` that (a) carries a bf cross-section and (b) has a mapped photoion
     * TARGET (ma_rr_target[lev]) which is itself a recomb SOURCE gets a single
     * up-jump to that target. On the device a fair-draw hit on this entry activates
     * the macro-atom on the upper-ion ground (Co IV/Fe IV/Ni IV), from which the
     * RADRECOMB continuum valve fires — so a bb-line-activated Co III/Fe III/Ni III
     * packet is reprocessed into recombination continuum instead of re-emitting the
     * trapped III line (the residual deep-EUV lamp). rr_tgt NULL (gate off) => both
     * arrays stay NULL => byte-identical. Co IV/Ni IV/S III etc. are fail-closed in
     * the target map (upper ion absent) so they get no up-jump => no runaway. */
    if (rr_tgt) {
        int *idst = (int *)malloc((size_t)n_levels * sizeof(int));
        int n_iup = 0;
        for (int lev = 0; lev < n_levels; lev++) {
            idst[lev] = -1;
            if (!atom->cmfgen_has_sigma || !atom->cmfgen_has_sigma[lev]) continue;
            int tgt = rr_tgt[lev];
            if (tgt < 0 || tgt >= n_levels) continue;
            /* dead-end guard: the target must carry recomb entries so the climbed
             * packet has a radiative-recombination exit back down. */
            if (refs[tgt + 1] > refs[tgt]) { idst[lev] = tgt; n_iup++; }
        }
        opacity->iup_dest_level = idst;
        opacity->iup_prob = (double *)calloc((size_t)n_levels * (size_t)n_shells,
                                             sizeof(double));
        printf("  [MA-RADRECOMB] inc2 internal-up-higher: %d source levels routed to "
               "mapped upper-ion grounds (photoionization up-jump ON)\n", n_iup);
    }

    free(ionized_ground); free(ionized_up_ip); free(cnt); free(fill);
    printf("  [MACROATOM_BF] recomb cascade topology: %d entries over %d source "
           "levels (mode=%d)\n", n_recomb, n_src, bf_mode);
    if (rr_tgt) {
        int n_emit = 0;
        for (int k = 0; k < n_recomb; k++) if (is_emit[k]) n_emit++;
        printf("  [MA-RADRECOMB] inc2 continuum emitters: %d/%d recomb entries flagged "
               "is_emit (target-map verified) => RADRECOMB continuum ON\n",
               n_emit, n_recomb);
    }
    if (stage4_bf)
        printf("  [STAGE4-BF] recomb source-coverage extension ON: %d of the %d "
               "sources are promoted stage-IV excited levels (Co/Fe/Ni/... IV "
               "manifolds now reachable recomb exits)\n", n_src_stage4, n_src);
}

static int g_ctp_idown_beta = -1;    /* LUMINA_MACROATOM_IDOWN_BETA (hoisted) */
static double g_ma_coll_limit_ev = -1.0; /* LUMINA_MA_COLLISION_LIMIT_EV */
static double *g_ctp_lev_gap = NULL;     /* [n_levels] chi_ion - E_lev (eV) */
static int g_ctp_idown_coll = -1;    /* LUMINA_MACROATOM_IDOWN_COLL (hoisted) */
static int g_ctp_lineres_jbar = -1;  /* LUMINA_CMF_LINERES_JBAR (hoisted) */
/* [withParityY Y6] MA internal-up minimum MC crossing count.  Historically
 * HARDCODED 10 at the internal-up J_line read while the matrix assembly reads
 * LUMINA_JBAR_MIN (production sets 3) -- an undeclared N2 threshold SPLIT: the
 * same line could be "well sampled" for the rate matrix and "unsampled" for the
 * macro-atom in the same iteration.  Default stays 10 => byte-identical. */
static int g_ctp_jbar_min_ma = 10;
static int g_ctp_iup_trad = -1;      /* LUMINA_MACROATOM_IUP_TRAD (ARTIS/TARDIS
                                      * dilute-blackbody internal-up pump) */
static int g_ctp_iup_beta = -1;      /* [Div-3] LUMINA_MACROATOM_IUP_BETA: apply the Sobolev
                                      * escape beta to the internal-up branching rate
                                      * (B_lu*beta*J), matching ARTIS rad_excitation_ratecoeff.
                                      * Fixes the p_iup/p_idn over-pump (up had no beta, down did). */
static int g_ctp_iup_jblue = -1;     /* [IUP-JBLUE] LUMINA_IUP_JBLUE: ARTIS-exact internal-up
                                      * rate (B_lu - B_ul n_u/n_l)*beta*J_blue with the MC
                                      * blue-wing estimator (opacity->jblue_line). Takes
                                      * precedence over LUMINA_MACROATOM_IUP_BETA. */
static long g_iup_jblue_used = 0;    /* [IUP-JBLUE] last-solve counters */
static long g_iup_jblue_fb   = 0;
/* [IUP-BINFIELD] LUMINA_IUP_BINFIELD (default OFF): ARTIS-classic field source for
 * the macro-atom INTERNAL-UP rate.  ARTIS ships
 *     artisoptions.h:74  constexpr bool DETAILED_LINE_ESTIMATORS_ON = false;
 * so rad_excitation_ratecoeff (macroatom.cc:571-599) never enters its per-line
 * estimator branch
 *     :588  if (DETAILED_LINE_ESTIMATORS_ON && !globals::lte_iteration) {
 *     :591    return R_over_J_nu * radfield::get_Jb_lu(nonemptymgi, jblueindex);
 * and always returns
 *     :596  const double R = R_over_J_nu * radfield::radfield(nu_trans, nonemptymgi);
 * i.e. the up-rate consumes the (W_bin,T_R_bin) MODEL EVALUATION of the binned
 * radiation field at the line's CMF frequency -- not a per-line MC estimator.
 * Armed, this gate reproduces that: the internal-up J is read straight out of
 * nlte->J_nu's fine bin at nu_line (nlte_get_J_at_nu), the C1 per-bin dilute field
 * already built by nlte_build_perbin_dilute_field as W_c*B_nu(TR_c,nu_ctr) -- which
 * therefore automatically carries the LUMINA_C1_SUPERBIN_TEPIN EUV-superbin pin and
 * the LUMINA_C1_DEGEN_FALLBACK raw publication when those are armed.  No field is
 * recomputed here.
 * SCOPE: the macro-atom transition-rate assembly in this function ONLY.  The NLTE
 * rate-matrix jbar consumption (LUMINA_NLTE_JBAR_POPS mode 3) is deliberately NOT
 * touched.  OFF => byte-identical to the per-line J_blue path. */
static int  g_ctp_iup_binfield = -1;
static long g_iup_binfield_used   = 0;  /* up-rate lines switched to the bin field */
static long g_iup_binfield_bypass = 0;  /* armed, but the taken branch is uncovered */
/* [JBLUE-ANCHOR] internal normalization self-check: per-bucket log-mean of
 * log10(J_blue/J_line) over lines where BOTH jblue>0 and J_line>0, bucketed
 * by the Sobolev escape beta. THIN (beta>0.5, tau~0) => no in-line (1-beta)S
 * saturation => ratio~1 (log~0) when the estimator is anchored; a systematic
 * offset is the jblue normalization bug magnitude in dex. THICK (beta<0.01)
 * => J_line carries the (1-beta)S saturation => ratio>1 (log>0). clamp = ratio
 * fell outside [1e-3,1e3] (log clamped to +/-3 so it cannot pollute the mean). */
static long   g_jba_thin_n = 0,     g_jba_thick_n = 0;
static double g_jba_thin_sum = 0.0, g_jba_thick_sum = 0.0;
static long   g_jba_thin_clamp = 0, g_jba_thick_clamp = 0;

/* [JBLUE-ANCHOR2] yardstick repair (2026-07-26). The single-number log-mean
 * above is unreadable for two reasons:
 *  (a) the +/-3 dex clamp is counted WITHOUT SIGN, so a bucket saturated at
 *      -3 (J_blue starved) and one saturated at +3 (J_blue blown up) report the
 *      same "clamp=N" and the mean sits in between -- the classic average-
 *      saturation misread. Split lo (lr<-3) / hi (lr>+3).
 *  (b) thin/thick alone mixes lines INSIDE the deterministic CMF fine window
 *      (where J_line can be the fine J_bar_l) with lines OUTSIDE it (where
 *      J_line necessarily falls back to the C1 binned read). Those are two
 *      different comparators, so their means must never be summed. Split by
 *      window membership.
 * Bucket index = (thick?2:0) + (out_of_window?1:0):
 *      0 thin-in   1 thin-out   2 thick-in   3 thick-out
 * g_jba_all_n counts EVERY (line,shell) pair that reached the check with both
 * J>0, so the reader can see what fraction the four buckets actually cover
 * (mid-beta lines, 0.01<=beta<=0.5, fall in NO bucket -- pre-existing, and now
 * visible instead of silent). The legacy accumulators above are untouched so
 * the original [JBLUE-ANCHOR] line stays byte-comparable across versions. */
static long   g_jba4_n[4]    = {0, 0, 0, 0};
static double g_jba4_sum[4]  = {0.0, 0.0, 0.0, 0.0};
static long   g_jba4_clo[4]  = {0, 0, 0, 0};   /* clamped at -3 dex (J_blue starved) */
static long   g_jba4_chi[4]  = {0, 0, 0, 0};   /* clamped at +3 dex (J_blue blown up) */
static long   g_jba_all_n    = 0;
/* Window bounds: the SAME env pair + defaults the deterministic CMF fine solver
 * uses (lumina_cmfgen.c cmfgen_fine_jbar: `double lam_lo = 1000.0, lam_hi =
 * 4000.0;` then LUMINA_CMF_FINE_LAMLO / LUMINA_CMF_FINE_LAMHI). Read here so a
 * shifted window shifts BOTH the producer and this yardstick together. */
static double g_jba_win_lo = -1.0, g_jba_win_hi = -1.0;
static void jba_window_init(void) {
    if (g_jba_win_lo > 0.0) return;
    double lo = 1000.0, hi = 4000.0;
    { const char *e = getenv("LUMINA_CMF_FINE_LAMLO"); if (e) lo = atof(e); }
    { const char *e = getenv("LUMINA_CMF_FINE_LAMHI"); if (e) hi = atof(e); }
    if (!(lo > 0.0) || !(hi > lo)) { lo = 1000.0; hi = 4000.0; }
    g_jba_win_lo = lo; g_jba_win_hi = hi;
}

/* [FB-COOL-KT] LUMINA_FB_COOL_KT: free-bound thermal cooling energy weight.
 * ARTIS ratecoeff.cc bfcooling_integrand charges the electron heat bath only
 * the photoelectron kinetic energy h(nu-nu_edge) (<> ~ kT_e); the binding
 * energy chi is the ionization ledger's, not the thermal pool's. Default OFF
 * keeps the legacy (chi + kT_e) weight (byte-identical). Idempotent init +
 * one-shot banner; pre-initialized serially by each caller before its OMP
 * region so parallel reads never race and never re-print. */
static int g_fb_cool_kt = -1;
static int fb_cool_kt_on(void) {
    if (g_fb_cool_kt < 0) {
        const char *e = getenv("LUMINA_FB_COOL_KT");
        g_fb_cool_kt = (e && atoi(e)) ? 1 : 0;
        if (g_fb_cool_kt)
            printf("[FB-COOL-KT] fb thermal cooling energy weight = kTe "
                   "(ARTIS bfcooling ledger; chi -> ionization ledger)\n");
    }
    return g_fb_cool_kt;
}

void plasma_get_iup_jblue_counts(long *used, long *fallback) {
    if (used) *used = g_iup_jblue_used;
    if (fallback) *fallback = g_iup_jblue_fb;
}

/* [FB-EDGE-METER] wiring-audit N9: k-packet fb "dominant recombining edge" lookup.
 * kpacket_fb_nu[s] is set from find_ioniz_energy(Z_dom, stage_dom-1); when that ion
 * is absent from the ionization table the helper returns its 1e10 sentinel and the
 * edge silently becomes 0 -> every fb exit in that shell degenerates (resonant line
 * re-emission at the line-activated site, no-op at the bf-activated one). Run-
 * cumulative shell-update count + the last offending (Z, recombining stage), so the
 * failure has a number instead of being invisible. Counting only. */
static long g_fb_dom_edge_fail = 0;   /* (shell,update) pairs with dom_n>0 but edge=0 */
static int  g_fb_dom_edge_fail_z = 0, g_fb_dom_edge_fail_stage = 0;  /* last offender */
void plasma_get_fb_dom_edge_fail(long *n_fail, int *last_Z, int *last_stage) {
    if (n_fail)    *n_fail    = g_fb_dom_edge_fail;
    if (last_Z)    *last_Z    = g_fb_dom_edge_fail_z;
    if (last_stage) *last_stage = g_fb_dom_edge_fail_stage;
}

/* [IUP-BINFIELD] armed flag + last-solve coverage counters. `armed` stays 0 until
 * compute_transition_probabilities has parsed the gate at least once. */
void plasma_get_iup_binfield_counts(int *armed, long *used, long *bypass) {
    if (armed)  *armed  = (g_ctp_iup_binfield > 0);
    if (used)   *used   = g_iup_binfield_used;
    if (bypass) *bypass = g_iup_binfield_bypass;
}

void plasma_get_jblue_anchor(long *thin_n, double *thin_logmean,
                             long *thick_n, double *thick_logmean,
                             long *thin_clamp, long *thick_clamp) {
    if (thin_n)  *thin_n  = g_jba_thin_n;
    if (thick_n) *thick_n = g_jba_thick_n;
    if (thin_logmean)  *thin_logmean  = g_jba_thin_n  ? g_jba_thin_sum  / (double)g_jba_thin_n  : 0.0;
    if (thick_logmean) *thick_logmean = g_jba_thick_n ? g_jba_thick_sum / (double)g_jba_thick_n : 0.0;
    if (thin_clamp)  *thin_clamp  = g_jba_thin_clamp;
    if (thick_clamp) *thick_clamp = g_jba_thick_clamp;
}

/* [JBLUE-ANCHOR2] 4-bucket read-out. Arrays are [4]:
 * 0 thin-in, 1 thin-out, 2 thick-in, 3 thick-out (see the accumulator block).
 * `logmean` is the per-bucket mean of the CLAMPED log10(J_blue/J_line); read it
 * together with clamp_lo/clamp_hi -- a bucket whose clamp counts are a large
 * fraction of n has a mean pinned by saturation, not by the population. */
void plasma_get_jblue_anchor4(long n[4], double logmean[4],
                              long clamp_lo[4], long clamp_hi[4],
                              long *all_n, double *win_lo, double *win_hi) {
    for (int i = 0; i < 4; i++) {
        if (n)        n[i]        = g_jba4_n[i];
        if (logmean)  logmean[i]  = g_jba4_n[i] ? g_jba4_sum[i] / (double)g_jba4_n[i] : 0.0;
        if (clamp_lo) clamp_lo[i] = g_jba4_clo[i];
        if (clamp_hi) clamp_hi[i] = g_jba4_chi[i];
    }
    if (all_n)  *all_n  = g_jba_all_n;
    if (win_lo) *win_lo = g_jba_win_lo;
    if (win_hi) *win_hi = g_jba_win_hi;
}

void compute_transition_probabilities(AtomicData *atom, PlasmaState *plasma,
                                       OpacityState *opacity,
                                       NLTEConfig *nlte,
                                       double damping_constant, int apply_damping,
                                       Geometry *geom) {
    (void)damping_constant;
    (void)apply_damping;
    int n_shells = opacity->n_shells;
    if (opacity->tau_sobolev) {
        size_t n=(size_t)opacity->n_lines*n_shells;
        for(size_t k=0;k<n;k++) if(opacity->tau_sobolev[k]<0.0) {
            a208_counters()->blocked_negative_transition++;
            fprintf(stderr,"[A2-08][BLOCKED] consumer=P06 reason="
                    "BLOCKED_NEGATIVE_OPACITY_SEMANTICS identity=%zu rc=3\n",k);
            return;
        }
    }

    /* [MA-RADRECOMB tau-gate] optically-thin threshold for the RADRECOMB continuum
     * emit decision (dig_E2 double-count repair). Edges with tau_bf > thresh are
     * treated on-the-spot (no continuum photon); thin edges emit. Diagnostic
     * override via LUMINA_RR_TAU_THRESH (default 1.0). */
    static double rr_tau_thresh = -1.0;
    if (rr_tau_thresh < 0.0) {
        const char *e = getenv("LUMINA_RR_TAU_THRESH");
        rr_tau_thresh = (e && atof(e) > 0.0) ? atof(e) : 1.0;
    }
    int n_levels = opacity->n_macro_levels;
    int n_trans  = opacity->n_macro_transitions;

    /* Use J_nu histogram for internal_up if NLTE is active and J_nu populated */
    int use_j_nu = (nlte != NULL && nlte->enabled && nlte->J_nu != NULL);

    /* P5: optional cap on B_lu·J_ν internal-up rate. When the carsus 3.4M-line
     * macro-atom is iron-forest-trapped, J_ν(UV) inflates 10-100× above LTE,
     * driving runaway UV pumping that locks the cascade. Capping J_line at
     * (cap_factor × W·B_ν(T_rad)) re-imposes a Mazzali-Lucy expectation
     * without disabling NLTE entirely. Set with LUMINA_J_CAP_FACTOR (>0 to
     * enable; typical 2-10). 0/unset = no cap. */
    static double j_cap_factor = -1.0;
    if (j_cap_factor < 0.0) {
        const char *e = getenv("LUMINA_J_CAP_FACTOR");
        j_cap_factor = (e ? atof(e) : 0.0);
        if (j_cap_factor < 0.0) j_cap_factor = 0.0;
    }

    /* Detailed-rate shot-noise regularization (the documented `detailed` mode
     * requirement, SIROCCO/Python macro-atom docs). At low T_rad in outer
     * shells a single MC photon in a bin can spike J_ν → one internal-up rate
     * dominates the source-level block → shot noise → convergence runaway.
     * Floor the binned-J up-rate at j_floor_factor·W·B_ν(T_rad): under-sampled
     * (near-zero) bins are pulled toward the smooth dilute-Planck anchor while
     * well-sampled bins keep their MC value. Symmetric to j_cap_factor (which
     * caps from above). Set with LUMINA_J_FLOOR_FACTOR (>0 enable; typical
     * 0.05-0.5). 0/unset = no floor. */
    static double j_floor_factor = -1.0;
    if (j_floor_factor < 0.0) {
        const char *e = getenv("LUMINA_J_FLOOR_FACTOR");
        j_floor_factor = (e ? atof(e) : 0.0);
        if (j_floor_factor < 0.0) j_floor_factor = 0.0;
    }
    /* [Wave-2 non-bf B / C28] A macro-atom radiative internal-up rate consumes
     * the represented radiation field. J<=factor*W*B and J>=factor*W*B are
     * falsifier priors, not transfer identities: non-LTE fields may be either
     * super- or sub-Planckian. The repair disables both diagnostic clamps at
     * their MA consumption point without changing their legacy gate semantics. */
    static int fix_ma_j_unclamp = -1;
    if (fix_ma_j_unclamp < 0) {
        const char *e = getenv("LUMINA_FIX_MA_J_UNCLAMP");
        fix_ma_j_unclamp = (e && atoi(e)) ? 1 : 0;
        if (fix_ma_j_unclamp)
            printf("[FIX-MA-J-UNCLAMP] internal-up consumes represented J; "
                   "LUMINA_J_CAP_FACTOR/J_FLOOR_FACTOR are ignored\n");
    }
    double j_cap_effective = fix_ma_j_unclamp ? 0.0 : j_cap_factor;
    double j_floor_effective = fix_ma_j_unclamp ? 0.0 : j_floor_factor;

    /* A2-07: k-packet population selection is no longer environment-gated.
     * A committed solved level wins; an untracked level uses LTE@T_e. */
    int kpemiss_se_pops = nlte &&
        nlte->population_committed_generation > 0;

    /* Emit-only boost on Fe II / Ni II UV→opt transitions, isolating the
     * spontaneous-emission lever from the radiative-pumping lever. Only
     * affects ttype == -1 (BB emit) for lines on Z={26,28}, ion=1 within
     * [LAM_MIN, LAM_MAX]. Three independent windows so UVtg ([3200,3800]),
     * fluo ([3800,4500]), and CaK ([2600,3200]) can be tuned separately.
     * Defaults: factor=1.0. */
    static double uvopt_emit_boost  = -1.0;
    static double uvopt_lam_min     = -1.0;
    static double uvopt_lam_max     = -1.0;
    static double uvopt_emit_boost2 = -1.0;
    static double uvopt_lam_min2    = -1.0;
    static double uvopt_lam_max2    = -1.0;
    static double uvopt_emit_boost3 = -1.0;
    static double uvopt_lam_min3    = -1.0;
    static double uvopt_lam_max3    = -1.0;
    /* W4 narrow window with its own Z/ion mask so we can target iron-peak III
     * for [3000,3100] bump suppress without altering W1/W2/W3 (Fe II+Ni II). */
    static double uvopt_emit_boost4 = -1.0;
    static double uvopt_lam_min4    = -1.0;
    static double uvopt_lam_max4    = -1.0;
    static unsigned int uvopt_z_mask4   = 0;  /* falls back to global if unset */
    static unsigned int uvopt_ion_mask4 = 0;  /* falls back to global if unset */
    /* P1 2026-05-13: UV internal-down suppress — break UV→UV cascade in
     * iron-peak macro-atom by scaling ttype=0 rate when destination line is UV.
     * Defaults: factor=1.0 (no-op), thresh=4000 Å, Z∈{21..28}, ion∈{1,2}. */
    static double macro_uv_idown_factor = -1.0;
    static double macro_uv_idown_thresh = -1.0;
    static unsigned int macro_uv_idown_z_mask = 0;
    static unsigned int macro_uv_idown_ion_mask = 0;
    /* Global Z mask for boost (bitmap over Z 1..30). Default = Fe(26) | Ni(28). */
    static unsigned int uvopt_z_mask = 0;
    /* Global ion-stage mask for boost (bitmap over ion 0..7). Default = ion=1 (II species only). */
    static unsigned int uvopt_ion_mask = 0;
    if (uvopt_emit_boost < 0.0) {
        const char *e   = getenv("LUMINA_UVOPT_EMIT_BOOST");
        const char *emn = getenv("LUMINA_UVOPT_EMIT_LAM_MIN");
        const char *emx = getenv("LUMINA_UVOPT_EMIT_LAM_MAX");
        const char *e2   = getenv("LUMINA_UVOPT_EMIT_BOOST2");
        const char *emn2 = getenv("LUMINA_UVOPT_EMIT_LAM_MIN2");
        const char *emx2 = getenv("LUMINA_UVOPT_EMIT_LAM_MAX2");
        const char *e3   = getenv("LUMINA_UVOPT_EMIT_BOOST3");
        const char *emn3 = getenv("LUMINA_UVOPT_EMIT_LAM_MIN3");
        const char *emx3 = getenv("LUMINA_UVOPT_EMIT_LAM_MAX3");
        const char *e4   = getenv("LUMINA_UVOPT_EMIT_BOOST4");
        const char *emn4 = getenv("LUMINA_UVOPT_EMIT_LAM_MIN4");
        const char *emx4 = getenv("LUMINA_UVOPT_EMIT_LAM_MAX4");
        uvopt_emit_boost  = (e    ? atof(e)    : 1.0);
        uvopt_lam_min     = (emn  ? atof(emn)  : 4500.0);
        uvopt_lam_max     = (emx  ? atof(emx)  : 7000.0);
        uvopt_emit_boost2 = (e2   ? atof(e2)   : 1.0);
        uvopt_lam_min2    = (emn2 ? atof(emn2) : 3800.0);
        uvopt_lam_max2    = (emx2 ? atof(emx2) : 4500.0);
        uvopt_emit_boost3 = (e3   ? atof(e3)   : 1.0);
        uvopt_lam_min3    = (emn3 ? atof(emn3) : 2600.0);
        uvopt_lam_max3    = (emx3 ? atof(emx3) : 3200.0);
        uvopt_emit_boost4 = (e4   ? atof(e4)   : 1.0);
        uvopt_lam_min4    = (emn4 ? atof(emn4) : 3000.0);
        uvopt_lam_max4    = (emx4 ? atof(emx4) : 3100.0);
        /* Allow factors in (0, +inf): >1 boosts emit, <1 suppresses. Branching
         * ratios re-normalize to 1.0 in the rates loop below, so a sub-unity
         * factor redistributes probability to the unaffected branches. */
        if (uvopt_emit_boost  <= 0.0) uvopt_emit_boost  = 1.0;
        if (uvopt_emit_boost2 <= 0.0) uvopt_emit_boost2 = 1.0;
        if (uvopt_emit_boost3 <= 0.0) uvopt_emit_boost3 = 1.0;
        if (uvopt_emit_boost4 <= 0.0) uvopt_emit_boost4 = 1.0;
        const char *zmask = getenv("LUMINA_UVOPT_EMIT_ZMASK");
        if (zmask && *zmask) {
            const char *p = zmask;
            while (*p) {
                int z = atoi(p);
                if (z > 0 && z < 32) uvopt_z_mask |= (1u << z);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_z_mask = (1u << 26) | (1u << 28);  /* Fe II + Ni II default */
        }
        const char *imask = getenv("LUMINA_UVOPT_EMIT_IONMASK");
        if (imask && *imask) {
            const char *p = imask;
            while (*p) {
                int ii = atoi(p);
                if (ii >= 0 && ii < 8) uvopt_ion_mask |= (1u << ii);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_ion_mask = (1u << 1);  /* default: ion=1 (II species) — backward compat */
        }
        /* Per-window masks for W4 (fall back to global if unset). */
        const char *zmask4 = getenv("LUMINA_UVOPT_EMIT_ZMASK4");
        if (zmask4 && *zmask4) {
            const char *p = zmask4;
            while (*p) {
                int z = atoi(p);
                if (z > 0 && z < 32) uvopt_z_mask4 |= (1u << z);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_z_mask4 = uvopt_z_mask;
        }
        const char *imask4 = getenv("LUMINA_UVOPT_EMIT_IONMASK4");
        if (imask4 && *imask4) {
            const char *p = imask4;
            while (*p) {
                int ii = atoi(p);
                if (ii >= 0 && ii < 8) uvopt_ion_mask4 |= (1u << ii);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            uvopt_ion_mask4 = uvopt_ion_mask;
        }

        /* P1 UV internal-down suppress (env once) */
        const char *euid    = getenv("LUMINA_MACRO_UV_IDOWN_FACTOR");
        const char *euidth  = getenv("LUMINA_MACRO_UV_IDOWN_THRESH");
        const char *euidz   = getenv("LUMINA_MACRO_UV_IDOWN_ZMASK");
        const char *euidim  = getenv("LUMINA_MACRO_UV_IDOWN_IONMASK");
        macro_uv_idown_factor = (euid   ? atof(euid)   : 1.0);
        macro_uv_idown_thresh = (euidth ? atof(euidth) : 4000.0);
        if (macro_uv_idown_factor <= 0.0) macro_uv_idown_factor = 1.0;
        if (euidz && *euidz) {
            const char *p = euidz;
            while (*p) {
                int z = atoi(p);
                if (z > 0 && z < 32) macro_uv_idown_z_mask |= (1u << z);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            /* default: Sc..Ni (21..28) */
            for (int z = 21; z <= 28; z++) macro_uv_idown_z_mask |= (1u << z);
        }
        if (euidim && *euidim) {
            const char *p = euidim;
            while (*p) {
                int ii = atoi(p);
                if (ii >= 0 && ii < 8) macro_uv_idown_ion_mask |= (1u << ii);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else {
            macro_uv_idown_ion_mask = (1u << 1) | (1u << 2);  /* default II + III */
        }
    }

    /* Find max block size for temp buffer */
    int max_block = 0;
    for (int lev = 0; lev < n_levels; lev++) {
        int bs = opacity->macro_block_references[lev + 1] -
                 opacity->macro_block_references[lev];
        if (bs > max_block) max_block = bs;
    }
    /* OMP note: rates_buf/kp_emiss are per-THREAD scratch, allocated inside
     * the parallel region below. Loop-body lazy-init statics were hoisted
     * here (g_ctp_*) — lazy init inside the parallel loop is a data race. */
    if (g_ctp_idown_beta < 0) { const char *e =
            getenv("LUMINA_MACROATOM_IDOWN_BETA");
        /* [ARTIS-PARITY D2] internal-down = A_ul*beta (Lucy/ARTIS escape-only form,
         * macroatom.cc:96-103) default-ON under parity; the legacy A_ul*(1-beta)
         * trapping double-count is retained only when the gate is OFF. */
        g_ctp_idown_beta = ((e && atoi(e)) || artis_parity_enabled()) ? 1 : 0; }
    if (g_ctp_idown_coll < 0) { const char *e =
            getenv("LUMINA_MACROATOM_IDOWN_COLL");
        /* [ARTIS-PARITY D2] internal-down carries the collisional (R+C)*eps_target
         * (macroatom.cc:103) default-ON under parity. */
        g_ctp_idown_coll = ((e && atoi(e)) || artis_parity_enabled()) ? 1 : 0; }
    if (g_ma_coll_limit_ev < 0.0) {
        const char *e = getenv("LUMINA_MA_COLLISION_LIMIT_EV");
        /* [ARTIS-PARITY E3] the forced p_kpkt=1 collision-limit shortcut is a Lumina-
         * only Rydberg-trap patch with no ARTIS analog; disable under parity. */
        g_ma_coll_limit_ev = (e && !artis_parity_enabled()) ? atof(e) : 0.0;   /* 0 = off */
        if (g_ma_coll_limit_ev > 0.0) {
            g_ctp_lev_gap = (double *)malloc(atom->n_levels * sizeof(double));
            long ncut = 0;
            for (int l2 = 0; l2 < atom->n_levels; l2++) {
                double chi = find_ioniz_energy(atom, atom->level_Z[l2],
                                               atom->level_ion[l2]);
                g_ctp_lev_gap[l2] = (chi < 1e9)
                    ? chi - atom->level_energy_eV[l2] : 1e9;
                if (g_ctp_lev_gap[l2] < g_ma_coll_limit_ev) ncut++;
            }
            printf("[COLL-LIMIT] p_kpkt=1 for %ld/%d levels within %.2f eV "
                   "of their continuum\n", ncut, atom->n_levels,
                   g_ma_coll_limit_ev);
        }
    }
    if (g_ctp_lineres_jbar < 0) { const char *e = getenv("LUMINA_CMF_LINERES_JBAR");
        g_ctp_lineres_jbar = (e && atoi(e)) ? 1 : 0; }
    /* [withParityY Y6] gate LUMINA_JBAR_UNIFY (default OFF): make the MA
     * internal-up site read the SAME env the matrix assembly reads
     * (LUMINA_JBAR_MIN), instead of its hardcoded 10.
     * NOTE, stated rather than hidden: when LUMINA_JBAR_MIN is UNSET the two
     * sites still differ -- assembly falls back to 50 (jbar_pops_mode==2) or 10,
     * this gate falls back to 3 (the production value, run_coevolve_s01.sh:127).
     * With the env set (the production configuration) they are identical. */
    {
        static int jbu_init = 0;
        if (!jbu_init) {
            jbu_init = 1;
            const char *g = getenv("LUMINA_JBAR_UNIFY");
            if (g && atoi(g)) {
                const char *e = getenv("LUMINA_JBAR_MIN");
                int v = (e && atoi(e) > 0) ? atoi(e) : 3;
                g_ctp_jbar_min_ma = v;
                printf("[JBAR-UNIFY] MA internal-up min MC crossings %d -> %d "
                       "(LUMINA_JBAR_MIN=%s; matrix assembly threshold), N2 "
                       "threshold split closed\n",
                       10, g_ctp_jbar_min_ma, e ? e : "(unset->3)");
            }
        }
    }
    if (g_ctp_iup_trad < 0) { const char *e = getenv("LUMINA_MACROATOM_IUP_TRAD");
        g_ctp_iup_trad = (e && atoi(e)) ? 1 : 0;
        if (g_ctp_iup_trad) printf("[IUP-TRAD] macro-atom internal-up pump = "
            "B_lu*W*B(nu,T_rad) (ARTIS/TARDIS dilute-blackbody, decoupled T_rad; "
            "replaces binned-J that thermalizes to B(T_e))\n"); }
    if (g_ctp_iup_beta < 0) { const char *e = getenv("LUMINA_MACROATOM_IUP_BETA");
        g_ctp_iup_beta = (e && atoi(e)) ? 1 : 0;
        if (g_ctp_iup_beta) printf("[IUP-BETA] macro-atom internal-up branching rate scaled by "
            "Sobolev beta (B_lu*beta*J, ARTIS-consistent; fixes p_iup/p_idn over-pump)\n"); }
    if (g_ctp_iup_jblue < 0) { const char *e = getenv("LUMINA_IUP_JBLUE");
        int env_on = (e && atoi(e));
        /* [ARTIS-PARITY C4] enable the per-line MC J_blue up-rate by default under
         * the master gate (ARTIS makes the detailed line estimator the up-rate
         * driver with a low-count fallback to the dilute-BB / per-bin field). */
        g_ctp_iup_jblue = (env_on || artis_parity_enabled()) ? 1 : 0;
        if (!env_on && artis_parity_enabled())
            printf("  [ARTIS-PARITY C4] per-line MC J_blue up-rate default-ON "
                   "((B_lu-B_ul n_u/n_l)·beta·J_blue; >=10-count fallback -> C1 per-bin field)\n");
        if (g_ctp_iup_jblue) {
            printf("[IUP-JBLUE] macro-atom internal-up rate = (B_lu - B_ul*n_u/n_l)*beta"
                   "*J_blue (ARTIS rad_excitation, MC blue-wing estimator; maser clamp "
                   "rate>=0; fallback=J_line when jblue NULL/unsampled)\n");
            if (g_ctp_iup_beta)
                printf("[IUP-JBLUE] notice: LUMINA_MACROATOM_IUP_BETA also set -> "
                       "IUP_JBLUE takes precedence (beta applied exactly once)\n");
        } }
    if (g_ctp_iup_jblue) { g_iup_jblue_used = 0; g_iup_jblue_fb = 0;
        /* [JBLUE-ANCHOR] per-solve reset */
        g_jba_thin_n = g_jba_thick_n = 0;
        g_jba_thin_sum = g_jba_thick_sum = 0.0;
        g_jba_thin_clamp = g_jba_thick_clamp = 0;
        /* [JBLUE-ANCHOR2] per-solve reset + one-shot window parse */
        jba_window_init();
        for (int _b = 0; _b < 4; _b++) {
            g_jba4_n[_b] = 0; g_jba4_sum[_b] = 0.0;
            g_jba4_clo[_b] = 0; g_jba4_chi[_b] = 0;
        }
        g_jba_all_n = 0; } /* per-solve reset */
    /* [IUP-BINFIELD] LUMINA_IUP_BINFIELD: switch the macro-atom internal-up field
     * source from the per-line MC estimators to the C1 binned-field evaluation.
     * Parsed AFTER g_ctp_iup_trad so the precedence warning below can be honest. */
    if (g_ctp_iup_binfield < 0) { const char *e = getenv("LUMINA_IUP_BINFIELD");
        g_ctp_iup_binfield = (e && atoi(e)) ? 1 : 0;
        if (g_ctp_iup_binfield) {
            printf("[IUP-BINFIELD] macro-atom internal-up J = C1 binned-field "
                   "evaluation at nu_line (nlte->J_nu fine bin, W_c*B_nu(TR_c) as "
                   "built by nlte_build_perbin_dilute_field); per-line MC J_blue, "
                   "MC J_bar and deterministic J_bar_l are BYPASSED for the up-rate "
                   "[ARTIS macroatom.cc:596 radfield(nu_trans) with "
                   "artisoptions.h:74 DETAILED_LINE_ESTIMATORS_ON=false]\n");
            if (g_ctp_iup_trad)
                printf("[IUP-BINFIELD] *** WARNING: LUMINA_MACROATOM_IUP_TRAD is also "
                       "armed and is evaluated FIRST -> every line takes the "
                       "W*B(nu,T_rad) pump and the bin-field switch is INERT ***\n");
            if (!use_j_nu)
                printf("[IUP-BINFIELD] *** WARNING: NLTE J_nu unavailable (use_j_nu=0) "
                       "-> there is no binned field to read; the switch is INERT ***\n");
        } }
    if (g_ctp_iup_binfield) { g_iup_binfield_used = 0;   /* per-solve reset */
                              g_iup_binfield_bypass = 0; }

    /* ---- k-packet thermal pool (collisional macro-atom) ----
     * A purely-radiative macro-atom cascade has no thermalization channel and
     * systematically redshifts (the down-branch A_ul·β is radiation-field-
     * independent). Real CMFGEN/TARDIS couples line energy to the electron pool
     * via COLLISIONS: collisional de-excitation transfers hν to a free electron
     * (a "k-packet"), which re-excites a level sampled from the thermal
     * collisional emissivity — pinning re-emission near the local Planck peak.
     * Here we build two per-shell tables consumed by the GPU selector:
     *   p_kpacket[lev][s]   = P(macro-atom at lev deactivates collisionally)
     *                       = ΣC_down / (ΣC_down + Σradiative_rates)
     *   kpacket_cdf[s][lev] = cumulative dist over which level a k-packet
     *                         re-excites, weight = n_lower·C_up·dE.
     * Collisional rates reuse the NLTE van Regemorter/Axelrod forms. Gated by
     * LUMINA_KPACKET (default off). Collisional internal-up (staying a macro-
     * atom without thermalizing) is folded into the k-packet re-excitation
     * (second-order; the dominant down-coll→k-packet→thermal-up loop is kept). */
    static int kpacket_init = 0;
    static int kpacket_mode = 0;
    if (!kpacket_init) {
        const char *e = getenv("LUMINA_KPACKET");
        if (e && atoi(e) != 0) kpacket_mode = 1;
        kpacket_init = 1;
    }
    opacity->kpacket_enabled = kpacket_mode;
    /* [Wave-1 C59 / FB-MULTI] LUMINA_FIX_BF_MULTI_EDGE=1: sample the
     * k-packet free-bound emissivity from the existing per-shell continuum CDF,
     * rather than replacing the Milne sum by one dominant representative edge.
     * Physical basis: j_fb is a sum over recombining continua; the weights below
     * are the same per-ion/per-level Milne cooling weights that form C_fb.
     * LUMINA_KPKT_FB_MULTI remains an exact legacy alias so certified launchers
     * retain their behavior. Both unset => the old single-edge path verbatim. */
    static int fb_multi_init = 0;
    static int kpkt_fb_multi = 0;
    if (!fb_multi_init) {
        kpkt_fb_multi = lumina_fix_bf_multi_edge_enabled();
        fb_multi_init = 1;
    }
    fb_cool_kt_on();   /* [FB-COOL-KT] serial pre-init before the OMP region */
    /* LUMINA_MACROATOM_EWEIGHT=1: weight each macro-atom transition rate by the
     * line photon energy hν (∝ line_nu) before block normalization. The bare-rate
     * recompute (A_ul·β / B_lu·J) is a NUMBER-flow (downbranch) weighting; the
     * Lucy-2002 macro-atom transports an indivisible ENERGY packet, so the
     * emission probability on line i→j must ∝ A_ij·β·hν_ij to reproduce the line
     * emissivity η_ij ∝ n_i A_ij hν_ij. Constant h cancels in the per-block
     * normalization, so multiplying by line_nu is sufficient. A/B falsifier vs
     * the bare-rate (downbranch) recompute on the cmfgen_then_mc spectrum. */
    static int eweight_init = 0, eweight_on = 0, eweight_neutral = 1;
    static double accum_ip_eV[32 * 8];   /* [Z*8+ion] = Σ IP(Z, <ion) (neutral-ground ref) */
    if (!eweight_init) {
        const char *e = getenv("LUMINA_MACROATOM_EWEIGHT");
        /* [ARTIS-PARITY D2] Lucy-2002 energy-flow weighting default-ON under parity
         * (the macro-atom transports an indivisible ENERGY packet). */
        if ((e && atoi(e) != 0) || artis_parity_enabled()) eweight_on = 1;
        /* Lucy/ARTIS macro-atom internal-transition weight uses the lower level's
         * energy measured from the NEUTRAL ground (excitation + accumulated
         * ionization potential), NOT each ion's own ground. The ion-ground form
         * (legacy) zeroes the dominant absolute-energy term so the macro-atom
         * de-activates near the absorbed frequency instead of cascading -> under-
         * produces UV->optical fluorescence -> too-red. LUMINA_MACROATOM_NEUTRAL_E=0
         * reverts to the legacy ion-ground reference for A/B falsification. */
        const char *en = getenv("LUMINA_MACROATOM_NEUTRAL_E");
        if (en) eweight_neutral = atoi(en);
        /* [ARTIS-PARITY D2] neutral-ground energy reference (excitation + accumulated
         * IP) is the ARTIS/Lucy internal-transition weight; force it on under parity. */
        if (artis_parity_enabled()) eweight_neutral = 1;
        if (artis_parity_enabled())
            printf("  [ARTIS-PARITY D2] macro-atom emission = internal-up (B_lu-B_ul n_u/n_l)"
                   "*beta*J_blue | internal-down (A_ul+C)*beta*eps | Lucy neutral-ground "
                   "energy weighting (thermal line-source overrides disabled -> D4)\n");
        for (int Z = 0; Z < 32; Z++) {
            double s = 0.0;
            for (int ion = 0; ion < 8; ion++) {
                accum_ip_eV[Z * 8 + ion] = s;       /* Σ IP of stages below `ion` */
                double chi = find_ioniz_energy(atom, Z, ion);
                if (chi < 1e9) s += chi;            /* skip the sentinel high value */
            }
        }
        eweight_init = 1;
    }
    /* Per-line cached lookups (static; line→global-level map is iteration- and
     * shell-invariant). glo/gup = global level idx of lower/upper; ip = ion-pop
     * slot. -1 = unresolved (skip). */
    static int   kp_n_lines_cached = -1;
    static int  *kp_glo = NULL, *kp_gup = NULL, *kp_ip = NULL;
    const double VAN_REG_COEFF = 2.16e-6, AX_OMEGA = 1.0; /* match NLTE solver */
    /* The lower/upper global-level map (kp_glo/kp_gup) is needed by BOTH the
     * k-packet collisional channels AND the Lucy-2002 energy weighting (which
     * weights internal transitions by the lower-level energy). Build it when
     * either is active. ([IUP-JBLUE] also needs it for the B_ul*n_u/n_l
     * stimulated term of the blue-wing up-rate.) */
    if (kpacket_mode || eweight_on || g_ctp_iup_jblue) {
        if (kp_n_lines_cached != atom->n_lines) {
            free(kp_glo); free(kp_gup); free(kp_ip);
            kp_glo = (int *)malloc(atom->n_lines * sizeof(int));
            kp_gup = (int *)malloc(atom->n_lines * sizeof(int));
            kp_ip  = (int *)malloc(atom->n_lines * sizeof(int));
            for (int line = 0; line < atom->n_lines; line++) {
                int ip = find_ion_pop_idx(atom, atom->line_atomic_number[line],
                                          atom->line_ion_number[line]);
                kp_ip[line] = ip; kp_glo[line] = -1; kp_gup[line] = -1;
                if (ip < 0) continue;
                int lev_base = atom->level_offset[ip];
                int lev_top  = atom->level_offset[ip + 1];
                for (int l = lev_base; l < lev_top; l++) {
                    if (atom->level_num[l] == atom->line_level_lower[line]) kp_glo[line] = l;
                    if (atom->level_num[l] == atom->line_level_upper[line]) kp_gup[line] = l;
                    if (kp_glo[line] >= 0 && kp_gup[line] >= 0) break;
                }
            }
            kp_n_lines_cached = atom->n_lines;
        }
    }
    /* [MA-REAL-UPSILON] (Fix-P1) gate + per-line real close-coupling Upsilon map.
     * When LUMINA_MA_REAL_UPSILON=1, build a T-invariant per-line map (line ->
     * source-slot + transition index) into the ALREADY-LOADED Omega tables
     * (feiii_col_* Zhang + generic col_ion_*), so the five MA-transport collision
     * sites below can swap the vR/Axelrod sentinel for the real Omega at eval time.
     * Level pairing is min/max on level_number (== energy rank, matching the radeq
     * bake plasma.c:6392 and the NLTE matrix). Default OFF => map never built, the
     * sites keep their vR sentinel => byte-identical. */
    static int   ma_real_ups_init = 0, ma_real_ups = 0;
    if (!ma_real_ups_init) {
        const char *e = getenv("LUMINA_MA_REAL_UPSILON");
        ma_real_ups = (e && atoi(e)) ? 1 : 0;
        ma_real_ups_init = 1;
    }
    static MaRuSrc ma_ru_src_reg[1 + LUMINA_MAX_COL_IONS];
    static int     ma_ru_nsrc = 0;
    static int    *ma_ru_line_src = NULL;   /* [n_lines] source slot, -1 = no real hit */
    static int    *ma_ru_line_t   = NULL;   /* [n_lines] transition idx within source  */
    static long    ma_ru_nhit = 0, ma_ru_nfall = 0;
    static int     ma_ru_map_lines = -1;
    if (ma_real_ups && ma_ru_map_lines != atom->n_lines) {
        free(ma_ru_line_src); free(ma_ru_line_t);
        ma_ru_line_src = (int *)malloc((size_t)atom->n_lines * sizeof(int));
        ma_ru_line_t   = (int *)malloc((size_t)atom->n_lines * sizeof(int));
        for (int l = 0; l < atom->n_lines; l++) { ma_ru_line_src[l] = -1; ma_ru_line_t[l] = -1; }
        ma_ru_nsrc = 0; ma_ru_nhit = 0; ma_ru_nfall = 0;
        /* temp per-source dense (a<b level_number) -> transition-index maps */
        struct { int Z, ion0, nlev; int *dm; } bsrc[1 + LUMINA_MAX_COL_IONS];
        int nb = 0;
        for (int src = -1; src < atom->ncol_ions; src++) {
            int Zs, ion0s, ntr, ntemp;
            const double *tg, *tom; const int *tlo, *thi;
            if (src < 0) {
                if (!atom->feiii_col_loaded) continue;
                Zs = atom->feiii_col_Z; ion0s = atom->feiii_col_ion;
                ntr = atom->feiii_col_n_trans; ntemp = atom->feiii_col_n_temp;
                tg = atom->feiii_col_tgrid; tlo = atom->feiii_col_lo;
                thi = atom->feiii_col_hi; tom = atom->feiii_col_omega;
            } else {
                Zs = atom->col_ion_Z[src]; ion0s = atom->col_ion_stage[src];
                ntr = atom->col_ion_n_trans[src]; ntemp = atom->col_ion_n_temp[src];
                tg = atom->col_ion_tgrid[src]; tlo = atom->col_ion_lo[src];
                thi = atom->col_ion_hi[src]; tom = atom->col_ion_omega[src];
            }
            if (ntr <= 0 || ntemp <= 0 || !tg || !tlo || !thi || !tom) continue;
            int maxlev = 0;
            for (int t = 0; t < ntr; t++) {
                if (tlo[t] > maxlev) maxlev = tlo[t];
                if (thi[t] > maxlev) maxlev = thi[t];
            }
            int nlev = maxlev + 1;
            if (nlev <= 0) continue;
            int *dm = (int *)malloc((size_t)nlev * (size_t)nlev * sizeof(int));
            if (!dm) continue;
            for (size_t i = 0; i < (size_t)nlev * (size_t)nlev; i++) dm[i] = -1;
            for (int t = 0; t < ntr; t++) {
                int lo = tlo[t], hi = thi[t];
                if (lo < 0 || hi < 0 || lo >= nlev || hi >= nlev) continue;
                int a = lo < hi ? lo : hi, b = lo < hi ? hi : lo;
                dm[(size_t)a * (size_t)nlev + (size_t)b] = t;
            }
            int slot = ma_ru_nsrc;
            ma_ru_src_reg[slot].ntemp = ntemp;
            ma_ru_src_reg[slot].tgrid = tg;
            ma_ru_src_reg[slot].omega = tom;
            bsrc[nb].Z = Zs; bsrc[nb].ion0 = ion0s; bsrc[nb].nlev = nlev; bsrc[nb].dm = dm;
            nb++; ma_ru_nsrc++;
        }
        for (int l = 0; l < atom->n_lines; l++) {
            int Zl = atom->line_atomic_number[l];
            int ionl = atom->line_ion_number[l];
            for (int b = 0; b < nb; b++) {
                if (bsrc[b].Z != Zl || bsrc[b].ion0 != ionl) continue;
                int a = atom->line_level_lower[l], c = atom->line_level_upper[l];
                if (a > c) { int tmp = a; a = c; c = tmp; }
                if (a >= 0 && c < bsrc[b].nlev) {
                    int t = bsrc[b].dm[(size_t)a * (size_t)bsrc[b].nlev + (size_t)c];
                    if (t >= 0) { ma_ru_line_src[l] = b; ma_ru_line_t[l] = t; ma_ru_nhit++; }
                    else ma_ru_nfall++;
                } else ma_ru_nfall++;
                break;
            }
        }
        for (int b = 0; b < nb; b++) free(bsrc[b].dm);
        ma_ru_map_lines = atom->n_lines;
        printf("  [MA-REAL-UPSILON] armed: %d ion tables -> real close-coupling Upsilon "
               "in MA transport collisions (eps drain / k-packet CDF / kp_deact / "
               "INTERNALUPSAME); %ld lines hit real table, %ld covered-ion fallback (vR)\n",
               ma_ru_nsrc, ma_ru_nhit, ma_ru_nfall);
        if (ma_ru_nhit == 0)
            fprintf(stderr, "  [MA-REAL-UPSILON][WARN] 0 real-table hits — real Upsilon NOT "
                            "wired (need LUMINA_ARTIS_PARITY=1 + col tables loaded); "
                            "MA collisions stay vR/Axelrod everywhere\n");
    }
    if (kpacket_mode || eweight_on) {
        if (!opacity->p_kpacket)
            opacity->p_kpacket = (double *)calloc((size_t)n_levels * n_shells, sizeof(double));
        if (!opacity->kpacket_cdf)
            opacity->kpacket_cdf = (double *)calloc((size_t)n_shells * n_levels, sizeof(double));
        if (!opacity->p_kpacket_ff)
            opacity->p_kpacket_ff = (double *)calloc((size_t)n_shells, sizeof(double));
        if (!opacity->p_kpacket_fb)
            opacity->p_kpacket_fb = (double *)calloc((size_t)n_shells, sizeof(double));
        if (!opacity->kpacket_fb_nu)
            opacity->kpacket_fb_nu = (double *)calloc((size_t)n_shells, sizeof(double));
        /* [FB-MULTI] per-continuum edge tables (only when the gate is on). */
        if (kpkt_fb_multi) {
            if (!opacity->kpacket_fb_edge_nu)
                opacity->kpacket_fb_edge_nu = (double *)calloc(
                    (size_t)n_shells * KPKT_FB_NEDGE, sizeof(double));
            if (!opacity->kpacket_fb_edge_cdf)
                opacity->kpacket_fb_edge_cdf = (double *)calloc(
                    (size_t)n_shells * KPKT_FB_NEDGE, sizeof(double));
            if (!opacity->kpacket_fb_edge_zstage)
                opacity->kpacket_fb_edge_zstage = (int *)calloc(
                    (size_t)n_shells * KPKT_FB_NEDGE, sizeof(int));
            if (!opacity->kpacket_fb_edge_lev)   /* [FB-MILNE C2] recombined-level idx per edge */
                opacity->kpacket_fb_edge_lev = (int *)calloc(
                    (size_t)n_shells * KPKT_FB_NEDGE, sizeof(int));
            if (!opacity->kpacket_fb_edge_count)
                opacity->kpacket_fb_edge_count = (int *)calloc(
                    (size_t)n_shells, sizeof(int));
        }
    }
    /* [MA-LINE-DESTRUCT] LUMINA_MA_LINE_DESTRUCT=1: allocate the per-(transition,
     * shell) two-level photon-destruction table eps=C_ul/(C_ul+A_ul*beta). Requires
     * the k-packet thermal pool (the destroyed photon's energy sink); warn + disable
     * otherwise. OFF (default) => ma_line_eps stays NULL => nothing written, uploaded,
     * or consumed => byte-identical baseline. */
    static int ma_line_destruct_gate = -1;
    if (ma_line_destruct_gate < 0) {
        const char *e = getenv("LUMINA_MA_LINE_DESTRUCT");
        ma_line_destruct_gate = (e && atoi(e)) ? 1 : 0;
        if (ma_line_destruct_gate && !kpacket_mode) {
            fprintf(stderr, "[MA-LINE-DESTRUCT][WARN] requires LUMINA_KPACKET=1 "
                            "(thermal sink) — DISABLED (no destruction wired)\n");
            ma_line_destruct_gate = 0;
        }
    }
    if (ma_line_destruct_gate && !opacity->ma_line_eps)
        opacity->ma_line_eps = (double *)calloc((size_t)n_trans * n_shells,
                                                sizeof(double));

    /* bf recomb-cascade topology (LUMINA_MACROATOM_BF). No-op (no alloc) when the
     * gate is off => recomb_block_refs stays NULL => byte-identical baseline. */
    build_recomb_topology(atom, opacity, n_shells);

    /* [ARTIS-PARITY D5] per-shell free-free HEATING coefficient (independent of the
     * k-packet pool; filled in the per-shell loop below). Allocated once under
     * parity; NULL otherwise => byte-identical baseline (never uploaded/consumed). */
    if (artis_parity_enabled() && !opacity->chi_ff_nnionpart)
        opacity->chi_ff_nnionpart =
            (double *)calloc((size_t)n_shells, sizeof(double));

    /* [ARTIS-PARITY M1] hoist the master gate once for the per-(level,shell)
     * fair-draw weight corrections below (COLDEEXC epsilon_trans weighting +
     * INTERNALUPSAME collisional up-term). Read-only int => safe to share across
     * the OMP region. OFF => every M1 branch is skipped (byte-identical). */
    const int parity_ma = artis_parity_enabled();

    /* Shells are independent (all writes are s-indexed); per-thread scratch
     * rates_buf (max transition block) + kp_emiss (level slots). Results are
     * bitwise order-independent. Was the dominant serial chunk (~1.28e8
     * (line,shell) entries) once the fluorescence pipeline made ctp hot. */
    #pragma omp parallel
    {
    double *rates_buf = (double *)malloc(max_block * sizeof(double));
    double *kp_emiss = kpacket_mode ?
        (double *)malloc(n_levels * sizeof(double)) : NULL;
    double *kpd_fe_arr = kpacket_mode ?    /* [KPD-FE2] per-level f_emit scratch */
        (double *)malloc(n_levels * sizeof(double)) : NULL;
    long jblue_used_loc = 0, jblue_fb_loc = 0;   /* [IUP-JBLUE] thread-local */
    long binf_used_loc = 0, binf_bypass_loc = 0; /* [IUP-BINFIELD] thread-local */
    /* [JBLUE-ANCHOR] thin/thick bucket accumulators (thread-local) */
    long   jba_thin_n_loc = 0,     jba_thick_n_loc = 0;
    double jba_thin_sum_loc = 0.0, jba_thick_sum_loc = 0.0;
    long   jba_thin_clamp_loc = 0, jba_thick_clamp_loc = 0;
    /* [JBLUE-ANCHOR2] 4-bucket thread-locals (thin/thick x in/out-of-window) */
    long   jba4_n_loc[4]   = {0, 0, 0, 0};
    double jba4_sum_loc[4] = {0.0, 0.0, 0.0, 0.0};
    long   jba4_clo_loc[4] = {0, 0, 0, 0};
    long   jba4_chi_loc[4] = {0, 0, 0, 0};
    long   jba_all_n_loc   = 0;

    #pragma omp for schedule(dynamic)
    for (int s = 0; s < n_shells; s++) {
        double T_e   = plasma->T_e[s];
        /* Only the diagnostic shadow below observes these locals. */
        double W = 1.0, T_rad = T_e;
        double n_e   = plasma->n_electron ? plasma->n_electron[s] :
                       (opacity->electron_density ? opacity->electron_density[s] : 0.0);
        double inv_sqrt_Te = (T_e > 0.0) ? 1.0 / sqrt(T_e) : 0.0;
        if (kpacket_mode) for (int j = 0; j < n_levels; j++) kp_emiss[j] = 0.0;
        if (kpd_fe_arr) for (int j = 0; j < n_levels; j++) kpd_fe_arr[j] = -1.0;
        /* [KPD] collexc-collapse discriminator: was tot killed by guards (add=0)
         * or by tiny inputs (add>0, maxw small)? Parity diagnosis only. */
        long kpd_seen = 0, kpd_nlow0 = 0, kpd_add = 0;
        double kpd_maxw = 0.0, kpd_maxnlow = 0.0;
        /* [KPD-FE] confirm the f_emit collapse: per-level emission share of the
         * normalized block + pk, bucketed per shell. */
        long fe_b[5] = {0,0,0,0,0};   /* f_emit <1e-8,<1e-6,<1e-4,<1e-2,>= */
        long pk_hi = 0, fe_n = 0;
        double fe_aggE = 0.0, fe_aggR = 0.0, pk_sum = 0.0;

        /* [ARTIS-PARITY D5] chi_ff_nnionpart[s] = 3.69255e8/sqrt(T_e) *
         * Sum_ions ioncharge^2 * n_ion (ARTIS rpkt.cc:785-799, g_ff=1). ion_pop_stage
         * is the net charge (0=neutral); neutrals do not free-free. */
        if (opacity->chi_ff_nnionpart) {
            double s_z2n = 0.0;
            for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                int q = atom->ion_pop_stage[ip];
                if (q <= 0) continue;
                double n_ion = atom->ion_number_density[(size_t)ip * n_shells + s];
                if (n_ion > 0.0) s_z2n += (double)q * (double)q * n_ion;
            }
            opacity->chi_ff_nnionpart[s] =
                (T_e > 0.0) ? s_z2n * 3.69255e8 * inv_sqrt_Te : 0.0;
        }

        for (int lev = 0; lev < n_levels; lev++) {
            int block_start = opacity->macro_block_references[lev];
            int block_end   = opacity->macro_block_references[lev + 1];
            if (block_start >= block_end) continue;

            /* Phase 1: Compute raw rates into temp buffer */
            double sum_rates = 0.0;
            double kp_deact  = 0.0;  /* collisional deactivation rate out of lev */
            double sum_emit  = 0.0;  /* [KPD-FE] ttype==-1 share of sum_rates */

            for (int tid = block_start; tid < block_end; tid++) {
                int ttype   = opacity->transition_type[tid];
                int line_id = opacity->transition_line_id[tid];
                double rate = 0.0;

                if (line_id >= 0 && line_id < atom->n_lines) {
                    double tau = opacity->tau_sobolev[line_id * n_shells + s];
                    double beta = beta_sobolev(tau);

                    if (ttype == -1) {
                        /* BB emission: A_ul * beta_sobolev */
                        rate = atom->line_A_ul[line_id] * beta;
                        /* Emit-only λ-window scale (env-controlled, four windows).
                         * W1/W2/W3 use the global Z/ion mask (defaults Fe+Ni II).
                         * W4 has its own Z/ion mask so it can target iron-peak III
                         * for [3000,3100] bump suppress without altering W1/W2/W3. */
                        int Z   = atom->line_atomic_number[line_id];
                        int ion = atom->line_ion_number[line_id];
                        double lambda_A = 2.99792458e18 / atom->line_nu[line_id];
                        if ((uvopt_emit_boost != 1.0 || uvopt_emit_boost2 != 1.0 ||
                             uvopt_emit_boost3 != 1.0) &&
                            Z > 0 && Z < 32 && ion >= 0 && ion < 8 &&
                            (uvopt_ion_mask & (1u << ion)) && (uvopt_z_mask & (1u << Z))) {
                            if (uvopt_emit_boost != 1.0 &&
                                lambda_A >= uvopt_lam_min && lambda_A < uvopt_lam_max)
                                rate *= uvopt_emit_boost;
                            if (uvopt_emit_boost2 != 1.0 &&
                                lambda_A >= uvopt_lam_min2 && lambda_A < uvopt_lam_max2)
                                rate *= uvopt_emit_boost2;
                            if (uvopt_emit_boost3 != 1.0 &&
                                lambda_A >= uvopt_lam_min3 && lambda_A < uvopt_lam_max3)
                                rate *= uvopt_emit_boost3;
                        }
                        if (uvopt_emit_boost4 != 1.0 &&
                            Z > 0 && Z < 32 && ion >= 0 && ion < 8 &&
                            (uvopt_ion_mask4 & (1u << ion)) && (uvopt_z_mask4 & (1u << Z)) &&
                            lambda_A >= uvopt_lam_min4 && lambda_A < uvopt_lam_max4) {
                            rate *= uvopt_emit_boost4;
                        }
                    } else if (ttype == 0) {
                        /* Internal down. Lucy-2002/ARTIS (macroatom.cc:96-103):
                         * the SAME escape-only R = A_ul*beta feeds both the
                         * emission and internal-down channels (the epsilon
                         * weighting alone splits them). The legacy A_ul*(1-beta)
                         * double-counts trapping (a trapped photon re-excites
                         * the same transition, net zero) and makes THICK
                         * transitions descend silently, emitting only at thin
                         * NIR lines -> the measured UV->NIR 52-72% cascade dump
                         * + UV<->IR ping-pong (MA-FATE, epay13/17).
                         * LUMINA_MACROATOM_IDOWN_BETA=1 selects the Lucy form;
                         * default keeps legacy for A/B. */
                        rate = g_ctp_idown_beta
                             ? atom->line_A_ul[line_id] * beta
                             : atom->line_A_ul[line_id] * (1.0 - beta);
                        /* ARTIS macroatom.cc:103: internal-down carries the
                         * COLLISIONAL de-excitation too, (R + C)*eps_target —
                         * without C the cascade under-descends in dense shells
                         * and UV fluorescence lands short of the optical.
                         * LUMINA_MACROATOM_IDOWN_COLL=1 adds the same vR/Omega
                         * C_down used by the kpkt deactivation channel. */
                        if (g_ctp_idown_coll && kp_gup && kp_glo &&
                            kp_gup[line_id] >= 0 && kp_glo[line_id] >= 0) {
                            double g_up2 = (double)atom->level_g[kp_gup[line_id]];
                            double g_lo2 = (double)atom->level_g[kp_glo[line_id]];
                            double f_lu2 = atom->line_f_lu[line_id];
                            if (g_up2 > 0.0 && n_e > 0.0) {
                                double C_down;
                                if (artis_parity_enabled()) {
                                    /* [ARTIS-PARITY E2/A6] the ONE shared ARTIS helper
                                     * (vR+Bethe+Gaunt permitted / g-scaled Axelrod
                                     * forbidden) drives the internal-down C too. */
                                    double cu, cd; int forb = (f_lu2 <= 1e-10);
                                    double dE2 = H_PLANCK * atom->line_nu[line_id];
                                    /* [MA-REAL-UPSILON] real close-coupling Omega for
                                     * covered transitions; else vR/Axelrod sentinel. */
                                    double cs2 = forb ? -2.0 : -1.0;
                                    if (ma_real_ups && ma_ru_line_src[line_id] >= 0) {
                                        double ups = ma_ru_upsilon(
                                            &ma_ru_src_reg[ma_ru_line_src[line_id]],
                                            ma_ru_line_t[line_id], T_e);
                                        if (ups > 0.0) cs2 = ups;
                                    }
                                    artis_col_rates(T_e, n_e, dE2, g_lo2, g_up2, f_lu2,
                                                    cs2, forb, &cu, &cd);
                                    C_down = cd;
                                } else {
                                    C_down = (f_lu2 > 1e-10)
                                        ? VAN_REG_COEFF * n_e * f_lu2 * 0.2 *
                                          inv_sqrt_Te / g_up2
                                        : 8.63e-6 * n_e * AX_OMEGA *
                                          inv_sqrt_Te / g_up2;
                                }
                                rate += C_down;
                            }
                        }
                        /* P1: UV internal-down suppress (break UV→UV cascade) */
                        if (macro_uv_idown_factor != 1.0) {
                            int Z   = atom->line_atomic_number[line_id];
                            int ion = atom->line_ion_number[line_id];
                            double lambda_A = 2.99792458e18 / atom->line_nu[line_id];
                            if (lambda_A < macro_uv_idown_thresh &&
                                Z > 0 && Z < 32 && ion >= 0 && ion < 8 &&
                                (macro_uv_idown_z_mask & (1u << Z)) &&
                                (macro_uv_idown_ion_mask & (1u << ion))) {
                                rate *= macro_uv_idown_factor;
                            }
                        }
                    } else if (ttype == 1) {
                        /* A2_06_DIAGNOSTIC_SHADOW_BEGIN: retain the historical
                         * source-selection/falsifier observables, but none of
                         * the values in this block owns the production rate. */
                        double legacy_rate_shadow = 0.0;
                        /* Internal up: B_lu * J_nu. THEN_MC MC-estimator macro-atom:
                         * use the per-line Sobolev j_blue J_bar accumulated from real
                         * MC packet crossings (faithful Lucy-2002/TARDIS) when it is
                         * enabled and the line is adequately sampled; otherwise fall
                         * back to the coarse binned J_nu histogram. The binned field
                         * is frequency-averaged (no line-resolved UV contrast), which
                         * is exactly what thermalizes the fluorescence -> the MC line
                         * estimator restores the realized UV pump at the line. */
                        double nu_line = atom->line_nu[line_id];
                        if (g_ctp_iup_trad) {
                            /* ARTIS/TARDIS dilute-blackbody internal-up pump:
                             * B_lu·W·B(ν,T_rad). T_rad is the radiation COLOR
                             * temperature, decoupled from T_e; in the outer
                             * line-forming region T_rad > T_e (dilute-hot UV
                             * from the deeper photosphere) => a super-thermal
                             * pump that drives the fluorescence cascade UP,
                             * which the binned-J path kills by thermalizing to
                             * B(T_e) (b_k=1). Cross-code: this is the TARDIS
                             * rate mode; the binned-MC estimator done right is
                             * ARTIS 'detailed' (the next rung). */
                            legacy_rate_shadow = atom->line_B_lu[line_id] *
                                                 W * planck_bnu(T_rad, nu_line);
                            /* [IUP-BINFIELD] NOT covered by the gate: IUP_TRAD is a
                             * dilute-BB model pump (no per-line MC estimator is read)
                             * and it is evaluated first, so it wins. Counted so the
                             * banner reports the uncovered residue instead of
                             * silently claiming a switch that never happened. */
                            if (g_ctp_iup_binfield) binf_bypass_loc++;
                        } else if (use_j_nu) {
                            double J_line;
                            /* P7 Stage-II (LUMINA_CMF_LINERES_JBAR=1): the DETERMINISTIC
                             * fine-grid line-resolved J_bar_l — the validated cure for the
                             * binned-J contrast collapse (ladder 4c/5b). Preferred over the
                             * MC estimator (sparse in UV) and the binned read. NULL/off =>
                             * fall through to the legacy paths (byte-identical baseline). */
                            /* [IUP-BINFIELD] armed => skip BOTH per-line estimators
                             * (the deterministic CMF J_bar_l and the MC per-line
                             * J_bar) so that J_line IS the C1 binned-field read in
                             * the final else -- ARTIS radfield(nu_trans) semantics
                             * (macroatom.cc:596). The mandate's "J_line fallback
                             * included" clause: the fallback must not be a per-line
                             * quantity either. OFF => the two guards are the
                             * pre-existing conditions verbatim. */
                            if (!g_ctp_iup_binfield &&
                                g_ctp_lineres_jbar && opacity->jbar_line_det &&
                                opacity->jbar_line_det[line_id * n_shells + s] >= 0.0) {
                                /* fine-grid line-resolved J_bar (P7 producer); -1
                                 * sentinel = line outside the fine window => fall back */
                                J_line = opacity->jbar_line_det[line_id * n_shells + s];
                            } else if (!g_ctp_iup_binfield &&
                                opacity->use_jbar_line && opacity->jbar_line &&
                                opacity->jbar_count[line_id * n_shells + s]
                                    >= g_ctp_jbar_min_ma) {
                                J_line = opacity->jbar_line[line_id * n_shells + s];
                            } else {
                                J_line = nlte_get_J_at_nu(nlte, s, nu_line);
                            }
                            /* [IUP-BINFIELD] coverage tally: this (line,shell) up-rate
                             * now takes its J from the C1 bin field. */
                            if (g_ctp_iup_binfield) binf_used_loc++;
                            if (j_cap_effective > 0.0 || j_floor_effective > 0.0) {
                                double J_lte = W * planck_bnu(T_rad, nu_line);
                                if (j_cap_effective > 0.0) {
                                    double J_max = j_cap_effective * J_lte;
                                    if (J_line > J_max) J_line = J_max;
                                }
                                if (j_floor_effective > 0.0) {
                                    double J_min = j_floor_effective * J_lte;
                                    if (J_line < J_min) J_line = J_min;
                                }
                            }
                            if (g_ctp_iup_jblue) {
                                /* [IUP-JBLUE] ARTIS rad_excitation_ratecoeff:
                                 *   rate = (B_lu - B_ul*n_u/n_l) * beta * J_blue
                                 * J_blue = MC blue-wing estimator (the CMF field
                                 * JUST BEFORE the packet redshifts into resonance,
                                 * no in-line (1-beta)*S saturation => no beta^2
                                 * double-suppression). Fallback to J_line when the
                                 * line is unsampled (entry 0) or the array is NULL.
                                 * n_u/n_l via the dilute-Boltzmann approximation
                                 * used by the kpkt branch below (NLTE pops are not
                                 * line-indexed at this point). Takes precedence
                                 * over IUP_BETA (beta applied exactly once). */
                                double J_blue = opacity->jblue_line
                                    ? opacity->jblue_line[line_id * n_shells + s] : 0.0;
                                if (J_blue > 0.0) {
                                    jblue_used_loc++;
                                    /* [JBLUE-ANCHOR] normalization self-check (pure
                                     * diagnostic; uses the RAW jblue before any
                                     * fallback and does NOT alter J_blue or rate). */
                                    if (J_line > 0.0) {
                                        double lr = log10(J_blue / J_line);
                                        int clamped = 0;
                                        int clamp_hi_dir = 0;   /* [ANCHOR2] +3 vs -3 */
                                        if (lr < -3.0)     { lr = -3.0; clamped = 1; }
                                        else if (lr > 3.0) { lr =  3.0; clamped = 1; clamp_hi_dir = 1; }
                                        if (beta > 0.5) {
                                            jba_thin_n_loc++;  jba_thin_sum_loc  += lr;
                                            if (clamped) jba_thin_clamp_loc++;
                                        } else if (beta < 0.01) {
                                            jba_thick_n_loc++; jba_thick_sum_loc += lr;
                                            if (clamped) jba_thick_clamp_loc++;
                                        }
                                        /* [JBLUE-ANCHOR2] signed clamp + 4-way bucket.
                                         * lambda is the line's REST (comoving) wavelength
                                         * -- the same quantity cmfgen_fine_jbar tests
                                         * against [lam_lo, lam_hi] when it decides whether
                                         * the line gets a deterministic fine J_bar_l or the
                                         * -1 out-of-window sentinel. */
                                        jba_all_n_loc++;
                                        {
                                            double lam_A_jba = (nu_line > 0.0)
                                                ? (C_SPEED_OF_LIGHT / nu_line) * 1.0e8 : -1.0;
                                            int out_w = !(lam_A_jba >= g_jba_win_lo &&
                                                          lam_A_jba <= g_jba_win_hi);
                                            int bidx = -1;
                                            if (beta > 0.5)        bidx = out_w ? 1 : 0;
                                            else if (beta < 0.01)  bidx = out_w ? 3 : 2;
                                            if (bidx >= 0) {
                                                jba4_n_loc[bidx]++;
                                                jba4_sum_loc[bidx] += lr;
                                                if (clamped) {
                                                    if (clamp_hi_dir) jba4_chi_loc[bidx]++;
                                                    else              jba4_clo_loc[bidx]++;
                                                }
                                            }
                                        }
                                    }
                                } else { J_blue = J_line; jblue_fb_loc++; }
                                /* [IUP-BINFIELD] THE FIELD-SOURCE SWITCH. The rate
                                 * form (B_lu - B_ul n_u/n_l)*beta*J is ARTIS's
                                 * R_over_J_nu (macroatom.cc:583-585) and is kept
                                 * exactly; only the J it multiplies changes, from the
                                 * per-line MC blue-wing estimator (ARTIS's
                                 * DETAILED_LINE_ESTIMATORS_ON branch, :588-593) to
                                 * radfield(nu_trans) (:596) = our C1 bin field, which
                                 * J_line already holds under this gate. The
                                 * [IUP-JBLUE]/[JBLUE-ANCHOR] counters above are left
                                 * intact on purpose: armed, they report what the OLD
                                 * path WOULD have consumed (the A/B comparator). */
                                if (g_ctp_iup_binfield) J_blue = J_line;
                                double coeff = atom->line_B_lu[line_id];
                                if (kp_glo && kp_gup && kp_glo[line_id] >= 0 &&
                                    kp_gup[line_id] >= 0 && T_rad > 0.0) {
                                    int glo = kp_glo[line_id], gup = kp_gup[line_id];
                                    double g_lo2 = (double)atom->level_g[glo];
                                    double g_up2 = (double)atom->level_g[gup];
                                    double w_lo2 = atom->level_metastable[glo] ? 1.0 : W;
                                    double w_up2 = atom->level_metastable[gup] ? 1.0 : W;
                                    double dboltz = (atom->level_energy_eV[gup] -
                                                     atom->level_energy_eV[glo]) *
                                                    EV_TO_ERG / (K_BOLTZMANN * T_rad);
                                    if (g_lo2 > 0.0 && w_lo2 > 0.0 &&
                                        dboltz > -500.0 && dboltz < 500.0) {
                                        double nu_nl = (w_up2 * g_up2) / (w_lo2 * g_lo2) *
                                                       exp(-dboltz);
                                        coeff -= atom->line_B_ul[line_id] * nu_nl;
                                    }
                                }
                                if (coeff < 0.0) coeff = 0.0;  /* maser clamp */
                                legacy_rate_shadow = coeff * beta * J_blue;
                            } else {
                                legacy_rate_shadow = atom->line_B_lu[line_id] * J_line;
                                if (g_ctp_iup_beta) legacy_rate_shadow *= beta;  /* [Div-3] ARTIS Sobolev escape */
                            }
                        } else {
                            legacy_rate_shadow = atom->line_B_lu[line_id] * W *
                                                 planck_bnu(T_rad, nu_line);
                            /* [IUP-BINFIELD] NOT covered: no NLTE J_nu exists
                             * (use_j_nu=0), so there is no binned field to read.
                             * Counted; the arm-time banner already warned. */
                            if (g_ctp_iup_binfield) binf_bypass_loc++;
                        }
                        (void)legacy_rate_shadow;
                        /* A2_06_DIAGNOSTIC_SHADOW_END */
                        {
                            double Jbar_view = 0.0;
                            if (nlte_bb_jbar_canonical(nlte, s, line_id,
                                                       &Jbar_view))
                                rate = atom->line_B_lu[line_id] * Jbar_view;
                            else
                                rate = 0.0; /* blocked radiative contribution */
                        }
                        /* [ARTIS-PARITY M1] INTERNALUPSAME collisional up-term
                         * (macroatom.cc:128-132: sum_internal_up_same += (R + C + NT)
                         * * epsilon_current). Fold C_up into the internal-up RATE (still
                         * bare here; the eweight block below multiplies the whole rate by
                         * e_low = the source/current-level neutral-ground energy =
                         * epsilon_current, giving (B_lu*J + C_up)*epsilon_current exactly
                         * as ARTIS). This is the MA-internal collisional up-jump -- a
                         * distinct process from, and NOT a double-count of, the k-packet
                         * COLLEXC re-excitation (kp_emiss CDF below), which starts from a
                         * k-packet, not an already-activated macro-atom. OFF adds nothing
                         * => byte-identical. */
                        if (parity_ma && n_e > 0.0 && T_e > 0.0 &&
                            kp_glo && kp_gup &&
                            kp_glo[line_id] >= 0 && kp_gup[line_id] >= 0) {
                            double g_lo_u = (double)atom->level_g[kp_glo[line_id]];
                            double g_up_u = (double)atom->level_g[kp_gup[line_id]];
                            double f_lu_u = atom->line_f_lu[line_id];
                            double dE_u   = H_PLANCK * atom->line_nu[line_id];
                            int    forb_u = (f_lu_u <= 1e-10);
                            double cu_u, cd_u;
                            /* [MA-REAL-UPSILON] real Omega for the internal-up C too. */
                            double cs_u = forb_u ? -2.0 : -1.0;
                            if (ma_real_ups && ma_ru_line_src[line_id] >= 0) {
                                double ups = ma_ru_upsilon(
                                    &ma_ru_src_reg[ma_ru_line_src[line_id]],
                                    ma_ru_line_t[line_id], T_e);
                                if (ups > 0.0) cs_u = ups;
                            }
                            artis_col_rates(T_e, n_e, dE_u, g_lo_u, g_up_u, f_lu_u,
                                            cs_u, forb_u, &cu_u, &cd_u);
                            if (cu_u > 0.0) rate += cu_u; /* eweighted below by eps_current */
                        }
                    }

                    /* Lucy-2002 macro-atom energy-flow weighting (the macro-atom
                     * transports an indivisible ENERGY packet, so branching must
                     * be energy-flow-weighted, NOT bare number-flow). The factor
                     * is per-transition-type (Lucy 2002 eqs; TARDIS macro_atom):
                     *   emission  i->j : x (e_i - e_j) = h*nu_ij  (PHOTON energy)
                     *   internal  i->j : x e_lower               (lower-level energy)
                     * Both internal jumps (down: dest=lower; up: source=lower) use
                     * the LOWER level's energy = the line's lower level (kp_glo).
                     * A uniform x h*nu (the earlier form) misweights internal jumps
                     * -- large h*nu but small e_lower -- driving the cascade into
                     * low-lying levels and re-emitting in the NIR instead of
                     * fluorescing in the optical. e measured from the ion ground
                     * (level_energy_eV), so internal jumps to ground vanish: a
                     * UV-excited atom must EMIT rather than silently sink to ground. */
                    if (eweight_on) {
                        if (ttype == -1) {
                            rate *= H_PLANCK * atom->line_nu[line_id];     /* h*nu_ij (reference-invariant difference) */
                        } else if (kp_glo[line_id] >= 0) {
                            /* internal jump weight ∝ lower-level energy. Neutral-ground
                             * reference (Lucy/ARTIS): excitation + accumulated IP. */
                            double e_low = atom->level_energy_eV[kp_glo[line_id]];
                            if (eweight_neutral) {
                                int Zl = atom->line_atomic_number[line_id];
                                int ionl = atom->line_ion_number[line_id];
                                if (Zl >= 0 && Zl < 32 && ionl >= 0 && ionl < 8)
                                    e_low += accum_ip_eV[Zl * 8 + ionl];
                            }
                            rate *= e_low * EV_TO_ERG;
                        } else {
                            rate = 0.0;  /* unresolved lower level: cannot weight */
                        }
                    }

                    /* k-packet collisional channels (van Regemorter / Axelrod;
                     * exp(±dE/kTe) cancels in C_down so it needs only g_up). */
                    if (kpacket_mode && n_e > 0.0 && T_e > 0.0 &&
                        kp_glo[line_id] >= 0 && kp_gup[line_id] >= 0) {
                        int    glo = kp_glo[line_id], gup = kp_gup[line_id];
                        double g_lo = (double)atom->level_g[glo];
                        double g_up = (double)atom->level_g[gup];
                        double f_lu = atom->line_f_lu[line_id];
                        double dE   = H_PLANCK * atom->line_nu[line_id]; /* erg */
                        if (ttype == -1 && g_up > 0.0) {
                            /* collisional de-excitation rate (lev is upper level).
                             * A6: under parity, the ONE shared ARTIS helper (vR+
                             * Bethe+Gaunt permitted / g-scaled Axelrod forbidden)
                             * replaces the k-packet's local vR/Axelrod copy. */
                            double C_down;
                            if (artis_parity_enabled()) {
                                double cu, cd; int forb = (f_lu <= 1e-10);
                                /* [MA-REAL-UPSILON] real Omega -> the k-packet C_down
                                 * that drives BOTH the ma_line_eps two-level drain
                                 * eps=C/(C+A*beta) and the COLDEEXC kp_deact term. */
                                double cs_d = forb ? -2.0 : -1.0;
                                if (ma_real_ups && ma_ru_line_src[line_id] >= 0) {
                                    double ups = ma_ru_upsilon(
                                        &ma_ru_src_reg[ma_ru_line_src[line_id]],
                                        ma_ru_line_t[line_id], T_e);
                                    if (ups > 0.0) cs_d = ups;
                                }
                                artis_col_rates(T_e, n_e, dE, g_lo, g_up, f_lu,
                                                cs_d, forb, &cu, &cd);
                                C_down = cd;
                            } else {
                                C_down = (f_lu > 1e-10)
                                    ? VAN_REG_COEFF * n_e * f_lu * 0.2 * inv_sqrt_Te / g_up
                                    : 8.63e-6 * n_e * AX_OMEGA * inv_sqrt_Te / g_up;
                            }
                            /* [MA-LINE-DESTRUCT] two-level photon-destruction
                             * probability for THIS terminal line, self-consistent with
                             * the A_ul*beta emission rate the MA lottery uses
                             * (plasma.c:2787): eps = C_ul/(C_ul + A_ul*beta_sobolev),
                             * with C_ul = C_down (just computed) and A_ul*beta the BARE
                             * emission rate (recomputed here because `rate` was already
                             * energy-flow weighted above). Numerically identical to
                             * radeq_line_eps_phys (same (1-e^-tau)/tau beta). Stored per
                             * (transition,shell); consumed on-device at the terminal MA
                             * deactivation. ma_line_eps NULL unless the gate is armed
                             * => this store is skipped => byte-identical baseline. */
                            if (opacity->ma_line_eps) {
                                double rad_bare = atom->line_A_ul[line_id] * beta;
                                double denom_e  = C_down + rad_bare;
                                opacity->ma_line_eps[(size_t)tid * n_shells + s] =
                                    (denom_e > 0.0) ? (C_down / denom_e) : 0.0;
                            }
                            /* [ARTIS-PARITY M1] COLDEEXC = Sum C_down * epsilon_trans
                             * (macroatom.cc:102 `sum_coldeexc += C * epsilon_trans`, :109).
                             * The k-packet pre-roll pk = kp_deact/(sum_rates + kp_deact)
                             * IS the ARTIS single fair draw between COLDEEXC and the
                             * same-ion radiative block (RADDEEXC + INTERNALDOWNSAME +
                             * INTERNALUPSAME): P(k)=pk and P(rad_i)=(1-pk)*p_i =
                             * rate_i/(kp_deact+sum_rates). But under parity the radiative
                             * block is ENERGY-flow (Lucy-2002/D2) weighted -- emit *= h*nu,
                             * internal *= eps_target -- so sum_rates carries erg while
                             * kp_deact (bare s^-1) does not. That ~1e11x dimensional
                             * mismatch forces COLDEEXC certainty (pk->1) and IS the
                             * fluorescence funnel (parity4: inner shells line-emit=0, all
                             * MA activations drain to k-pool ff/fb). Weighting C_down by
                             * dE=h*nu=epsilon_trans makes COLDEEXC a dimensionally
                             * consistent peer of RADDEEXC (=A_ul*beta*h*nu): since
                             * epsilon_trans (photon) << eps_target (neutral-ground level
                             * energy carried by INTERNALDOWNSAME), collisions now
                             * predominantly drive the INTERNAL down-jump (the ladder-walk
                             * escape valve) instead of certain thermalization -- exactly
                             * ARTIS. OFF path keeps the bare rate => byte-identical. */
                            kp_deact += parity_ma ? (C_down * dE) : C_down;
                        } else if (ttype == 1 && g_lo > 0.0) {
                            /* k-packet re-excitation weight n_lower·C_up·dE,
                             * deposited at the upper (destination) level. The
                             * exp(-dE/kTe) inside C_up provides the thermal
                             * (red-peaked) weighting that pins re-emission to
                             * the local Planck peak. */
                            int    ip = kp_ip[line_id];
                            double n_lower = 0.0;
                            /* Committed solved population for tracked levels;
                             * otherwise the sole LTE@T_e reference below. */
                            int gnl = (kpemiss_se_pops && nlte &&
                                       nlte->global_to_nlte_level)
                                    ? nlte->global_to_nlte_level[glo] : -1;
                            if (gnl >= 0 && nlte->nlte_level_populations) {
                                n_lower = nlte->nlte_level_populations[
                                            (size_t)gnl * n_shells + s];
                            } else if (ip >= 0) {
                                double n_ion = atom->ion_number_density[(size_t)ip * n_shells + s];
                                /* A2-07: one LTE@T_e reference, irrespective of
                                 * the legacy kpemiss selector (now shadow-only). */
                                double Zp = atom->partition_functions[(size_t)ip * n_shells + s];
                                PopulationAtomicView av = population_atomic_view(atom);
                                double frac = 0.0;
                                PopulationStatus ps = population_lte_level_fraction(
                                    &av, (size_t)ip, (size_t)glo, T_e, Zp, &frac);
                                if (ps == POP_OK || ps == POP_EXACT_ZERO)
                                    n_lower = n_ion * frac;
                            }
                            kpd_seen++;
                            if (n_lower <= 0.0) kpd_nlow0++;
                            else if (n_lower > kpd_maxnlow) kpd_maxnlow = n_lower;
                            if (n_lower > 0.0) {
                                double C_up;
                                if (artis_parity_enabled()) {
                                    /* A6: shared ARTIS helper (exp(-u) is internal). */
                                    double cu, cd; int forb = (f_lu <= 1e-10);
                                    /* [MA-REAL-UPSILON] real Omega -> k-packet re-excite
                                     * weight n_lower*C_up*dE (kp_emiss CDF). */
                                    double cs_e = forb ? -2.0 : -1.0;
                                    if (ma_real_ups && ma_ru_line_src[line_id] >= 0) {
                                        double ups = ma_ru_upsilon(
                                            &ma_ru_src_reg[ma_ru_line_src[line_id]],
                                            ma_ru_line_t[line_id], T_e);
                                        if (ups > 0.0) cs_e = ups;
                                    }
                                    artis_col_rates(T_e, n_e, dE, g_lo, g_up, f_lu,
                                                    cs_e, forb, &cu, &cd);
                                    C_up = cu;
                                } else {
                                    double exp_up = exp(-dE / (K_BOLTZMANN * T_e));
                                    C_up = (f_lu > 1e-10)
                                        ? VAN_REG_COEFF * n_e * f_lu * exp_up * 0.2 * inv_sqrt_Te / g_lo
                                        : 8.63e-6 * n_e * AX_OMEGA * exp_up * inv_sqrt_Te / g_lo;
                                }
                                double w = n_lower * C_up * dE;
                                int dst = opacity->destination_level_id[tid];
                                if (w > 0.0 && dst >= 0 && dst < n_levels) {
                                    kp_emiss[dst] += w;
                                    kpd_add++;
                                    if (w > kpd_maxw) kpd_maxw = w;
                                }
                            }
                        }
                    }
                }
                if (rate < 0.0) rate = 0.0;
                rates_buf[tid - block_start] = rate;
                sum_rates += rate;
                if (ttype == -1) sum_emit += rate;   /* [KPD-FE] */
            }

            /* bf recomb cascade (LUMINA_MACROATOM_BF): for a source level (the
             * upper-ion ground that bf activation lands on) add the cross-ion
             * INTERNALDOWNLOWER weights into the SAME sum_rates BEFORE the block
             * is normalized. w_down = n_e*alpha_sp(i->j)*eps_j; eps_j is the
             * lower-ion target's neutral-ground energy (excitation + accumulated
             * IP). Unnormalized w_down is parked in recomb_prob, divided through
             * by sum_rates in Phase 2 below. recomb_block_refs!=NULL only when
             * the gate is on => no cost on the baseline. */
            int rec_s = 0, rec_e = 0;
            if (opacity->recomb_block_refs) {
                rec_s = opacity->recomb_block_refs[lev];
                rec_e = opacity->recomb_block_refs[lev + 1];
            }
            if (rec_e > rec_s && n_e > 0.0 && T_e > 0.0) {
                double g_i = (double)atom->level_g[lev];
                int j0 = opacity->recomb_dest_level[rec_s];
                int ip_lo = find_ion_pop_idx(atom, atom->level_Z[j0],
                                             atom->level_ion[j0]);
                if (ip_lo >= 0) {
                    int Z_j = atom->ion_pop_Z[ip_lo];
                    int stage_j = atom->ion_pop_stage[ip_lo];
                    for (int k = rec_s; k < rec_e; k++) {
                        int j = opacity->recomb_dest_level[k];
                        double R = n_e * recomb_alpha_per_level(atom, ip_lo, j, g_i, T_e);
                        double eps_j = atom->level_energy_eV[j];
                        if (Z_j >= 0 && Z_j < 32 && stage_j >= 0 && stage_j < 8)
                            eps_j += accum_ip_eV[Z_j * 8 + stage_j];
                        double w_down = R * eps_j * EV_TO_ERG;
                        /* [RATES-FIX F3] a NaN from the old 0 x inf Milne form
                         * passed this test (NaN < 0 is false) and poisoned the
                         * cascade probability normalization. */
                        if (rates_fix_enabled() ? (!isfinite(w_down) || w_down < 0.0)
                                                : (w_down < 0.0)) w_down = 0.0;
                        opacity->recomb_prob[(size_t)k * n_shells + s] = w_down;
                        sum_rates += w_down;

                        /* [MA-RADRECOMB tau-gate] per-shell emit decision: emit the
                         * recombination continuum ONLY where dest level j's edge is
                         * optically THIN (tau_bf <= thresh). tau_bf = sigma_edge(k) *
                         * n_j(s) * dr(s). Thick edges (ground/low levels, whose
                         * photons are reabsorbed on-the-spot => dig_E2 double count)
                         * stay INTERNALDOWNLOWER (no photon). Writes every live (k,s)
                         * each call (same coverage as recomb_prob). NULL array (gate
                         * off) => untouched => byte-identical baseline. */
                        if (opacity->recomb_emit_shell) {
                            int emit = 0;
                            if (opacity->recomb_is_emit && opacity->recomb_is_emit[k] &&
                                opacity->recomb_sigma_edge && geom) {
                                double dr = geom->r_outer[s] - geom->r_inner[s];
                                /* n_dest = population of the recombined-onto lower-ion
                                 * level j (the level that reabsorbs an edge photon via
                                 * photoionization). Prefer the live NLTE SE population;
                                 * fall back to the committed LTE-at-T_e Boltzmann pop
                                 * LTE@T_e whenever j is outside the committed solved
                                 * subset. A committed exact zero remains zero. */
                                double n_j = 0.0;
                                int n_j_from_solve = 0;
                                if (nlte && nlte->nlte_level_populations &&
                                    nlte->global_to_nlte_level &&
                                    nlte->population_committed_generation > 0) {
                                    int gnl = nlte->global_to_nlte_level[j];
                                    if (gnl >= 0) {
                                        n_j = nlte->nlte_level_populations[
                                                  (size_t)gnl * n_shells + s];
                                        n_j_from_solve = 1;
                                    }
                                }
                                if (!n_j_from_solve && atom->ion_number_density &&
                                    atom->partition_functions && T_e > 0.0) {
                                    double Z_part = atom->partition_functions[
                                                        (size_t)ip_lo * n_shells + s];
                                    double n_ion  = atom->ion_number_density[
                                                        (size_t)ip_lo * n_shells + s];
                                    if (Z_part > 0.0 && n_ion > 0.0) {
                                        PopulationAtomicView av = population_atomic_view(atom);
                                        double frac = 0.0;
                                        PopulationStatus ps = population_lte_level_fraction(
                                            &av, (size_t)ip_lo, (size_t)j,
                                            T_e, Z_part, &frac);
                                        if (ps == POP_OK || ps == POP_EXACT_ZERO)
                                            n_j = n_ion * frac;
                                    }
                                }
                                double tau_bf = opacity->recomb_sigma_edge[k] * n_j * dr;
                                emit = (tau_bf <= rr_tau_thresh) ? 1 : 0;
                            }
                            opacity->recomb_emit_shell[(size_t)k * n_shells + s] = emit;
                        }
                    }
                }
            }

            /* [MA-RADRECOMB iup] INTERNALUPHIGHER weight: the ion-changing
             * photoionization + collisional-ionization up-jump out of THIS
             * (lower-ion) source level, added to the SAME sum_rates before the
             * block is normalized. ARTIS macroatom.cc:165-185:
             *   sum_up_higher += (R_photoion + C_collion) * epsilon_current
             * R_photoion is the field-weighted per-level bf rate — computed here
             * with the SAME per-bin estimator the ion balance uses
             * (parity_gamma_phot: C2 bf_rate_estimator under parity, else the C1
             * dilute-BB integral over J_nu). C_collion is the Seaton edge rate
             * (ARTIS col_ionisation_ratecoeff). epsilon_current = the source level's
             * neutral-ground energy (Lucy weighting, matching the recomb/INTERNALUP
             * channels). Unnormalized weight parked in iup_prob, divided by sum_rates
             * below. iup_prob NULL (gate off) => nothing added => byte-identical. */
            double iup_w = 0.0;
            if (opacity->iup_prob && opacity->iup_dest_level &&
                opacity->iup_dest_level[lev] >= 0 && n_e > 0.0 && T_e > 0.0 &&
                use_j_nu && atom->cmfgen_has_sigma && atom->cmfgen_has_sigma[lev] &&
                atom->cmfgen_n_freq_bins == nlte->n_freq_bins) {
                int Zl = atom->level_Z[lev], stl = atom->level_ion[lev];
                double chi_eV = find_ioniz_energy(atom, Zl, stl);
                if (chi_eV > 0.0 && chi_eV < 1e9) {
                    double chi_erg_l = chi_eV * EV_TO_ERG;
                    double E_lev_erg = atom->level_energy_eV[lev] * EV_TO_ERG;
                    double eps_trans = chi_erg_l - E_lev_erg;     /* threshold energy */
                    double nu_thresh = eps_trans / H_PLANCK;
                    if (nu_thresh > 0.0) {
                        int nfb = nlte->n_freq_bins;
                        const double *sigma_row =
                            &atom->cmfgen_sigma_bf[(size_t)lev * (size_t)nfb];
                        double log_numin = log(nlte->nu_min), dln = nlte->d_log_nu;
                        /* [A2-05] R_ph = canonical-view integral (blocked =>
                         * no field term, counted).  The reduced loop keeps only
                         * the Seaton edge sigma extraction. */
                        double R_ph = 0.0, sigma_edge = 0.0;
                        BfRateResult br_iup;
                        if (nlte_bf_gamma_canonical(nlte, s, sigma_row, 0.0,
                                                    nu_thresh, &br_iup) == 0 &&
                            (br_iup.state == BF_RATE_VALID ||
                             br_iup.state == BF_RATE_EXACT_ZERO))
                            R_ph = br_iup.gamma;
                        for (int bb = 0; bb < nfb; bb++) {
                            double nu_bin = exp(log_numin + (bb + 0.5) * dln);
                            if (nu_bin < nu_thresh) continue;
                            if (sigma_row[bb] > 0.0) {
                                sigma_edge = sigma_row[bb]; /* first bin>=edge */
                                break;
                            }
                        }
                        /* Seaton collisional ionization at the edge (ARTIS
                         * col_ionisation_ratecoeff, gaunt by ionstage). */
                        double C_ion = 0.0;
                        double fac1 = eps_trans / (K_BOLTZMANN * T_e);
                        if (sigma_edge > 0.0 && fac1 > 0.0 && fac1 < 700.0) {
                            double g_col = (stl <= 0) ? 0.1 : (stl == 1) ? 0.2 : 0.3;
                            C_ion = n_e * 1.55e13 * inv_sqrt_Te * g_col * sigma_edge *
                                    exp(-fac1) / fac1;
                        }
                        double eps_cur = atom->level_energy_eV[lev];
                        if (Zl >= 0 && Zl < 32 && stl >= 0 && stl < 8)
                            eps_cur += accum_ip_eV[Zl * 8 + stl];
                        iup_w = (R_ph + C_ion) * eps_cur * EV_TO_ERG;
                        if (!(iup_w > 0.0)) iup_w = 0.0;
                        sum_rates += iup_w;
                    }
                }
            }
            if (opacity->iup_prob)
                opacity->iup_prob[(size_t)lev * n_shells + s] = iup_w; /* normalized below */

            /* Phase 2: Normalize and apply (with optional damping) */
            if (sum_rates > 0.0) {
                for (int tid = block_start; tid < block_end; tid++) {
                    double p_new = rates_buf[tid - block_start] / sum_rates;
                    opacity->transition_probabilities[tid * n_shells + s] = p_new;
                }
            } else {
                /* A2-09: an empty block is invalid.  Never retain an older
                 * generation or manufacture a final BB channel. */
                a209_counters()->transition_empty++;
                for (int tid=block_start;tid<block_end;tid++)
                    opacity->transition_probabilities[tid*n_shells+s]=0.0;
            }

            /* Normalize the recomb-cascade weights by the recomb-inclusive
             * sum_rates (no damping on this channel). */
            if (rec_e > rec_s) {
                for (int k = rec_s; k < rec_e; k++) {
                    size_t idx = (size_t)k * n_shells + s;
                    opacity->recomb_prob[idx] = (sum_rates > 0.0)
                        ? opacity->recomb_prob[idx] / sum_rates : 0.0;
                }
            }

            /* [MA-RADRECOMB iup] Normalize the internal-up-higher weight by the same
             * recomb-inclusive sum_rates (no damping, mirroring the recomb channel).
             * block_probs + recomb_probs + iup_prob now partition [0,1]. */
            if (opacity->iup_prob) {
                size_t idx = (size_t)lev * n_shells + s;
                opacity->iup_prob[idx] = (sum_rates > 0.0)
                    ? opacity->iup_prob[idx] / sum_rates : 0.0;
            }

            /* k-packet deactivation probability for this level: collisional
             * deactivation competes with all radiative channels (sum_rates).
             * [COLLISION-LIMIT] levels within LUMINA_MA_COLLISION_LIMIT_EV of
             * their ionization continuum are collision-DOMINATED (Rydberg
             * Delta-n collision rates scale ~n^4, far beyond van Regemorter):
             * force p_kpkt=1 there. Kills the unphysical far-IR Rydberg
             * radiative cycle (UV -> cascade -> ~100-300um Rydberg photon ->
             * reabsorbed tau~1e4 -> ladder climb -> UV; the cascade-walk
             * falsifier showed the Rydberg forest carries 99.94% of the
             * NIR-entry flux). CMFGEN's super-levels do the same job. */
            if (kpacket_mode) {
                double denom = sum_rates + kp_deact;
                double pkv = (denom > 0.0) ? (kp_deact / denom) : 0.0;
                if (g_ma_coll_limit_ev > 0.0 && g_ctp_lev_gap &&
                    g_ctp_lev_gap[lev] < g_ma_coll_limit_ev)
                    pkv = 1.0;
                opacity->p_kpacket[(size_t)lev * n_shells + s] = pkv;
                /* [KPD-FE] bucket this level's emission share + pk */
                if (sum_rates > 0.0) {
                    double fe = sum_emit / sum_rates;
                    fe_n++;
                    fe_b[fe < 1e-8 ? 0 : fe < 1e-6 ? 1 : fe < 1e-4 ? 2 :
                         fe < 1e-2 ? 3 : 4]++;
                    fe_aggE += sum_emit; fe_aggR += sum_rates;
                    pk_sum += pkv; if (pkv > 0.9) pk_hi++;
                    if (kpd_fe_arr) kpd_fe_arr[lev] = fe;
                } else if (kpd_fe_arr) kpd_fe_arr[lev] = -1.0;
            }
        }

        /* Build the per-shell k-packet re-excitation CDF (cumulative over
         * levels, contiguous per shell for GPU binary search). Normalized to
         * end at 1.0; a flat fallback if the shell has no collisional weight. */
        if (kpacket_mode) {
            double *cdf = opacity->kpacket_cdf + (size_t)s * n_levels;
            double tot = 0.0;
            for (int j = 0; j < n_levels; j++) tot += kp_emiss[j];
            if (artis_parity_enabled()) {
                fprintf(stderr, "[KPD] s%d tot=%.3e seen=%ld nlow0=%ld add=%ld "
                        "maxw=%.3e maxnlow=%.3e\n",
                        s, tot, kpd_seen, kpd_nlow0, kpd_add, kpd_maxw, kpd_maxnlow);
                fprintf(stderr, "[KPD-FE] s%d n=%ld fe_buckets(<1e-8,<1e-6,<1e-4,"
                        "<1e-2,>=)=%ld/%ld/%ld/%ld/%ld agg_emit_share=%.3e "
                        "pk_mean=%.3f pk>0.9=%ld\n",
                        s, fe_n, fe_b[0], fe_b[1], fe_b[2], fe_b[3], fe_b[4],
                        (fe_aggR > 0.0) ? fe_aggE / fe_aggR : 0.0,
                        (fe_n > 0) ? pk_sum / (double)fe_n : 0.0, pk_hi);
                /* [KPD-FE2] VISITED-weighted view: the collexc CDF's top landing
                 * levels — where re-excited MAs actually go. */
                if (kpd_fe_arr && tot > 0.0) {
                    for (int r = 0; r < 8; r++) {
                        int jmax = -1; double wmax = 0.0;
                        for (int j = 0; j < n_levels; j++)
                            if (kp_emiss[j] > wmax) { wmax = kp_emiss[j]; jmax = j; }
                        if (jmax < 0) break;
                        fprintf(stderr, "[KPD-FE2] s%d lev%d share=%.3f fe=%.3e "
                                "pk=%.4f\n", s, jmax, wmax / tot,
                                kpd_fe_arr[jmax],
                                opacity->p_kpacket[(size_t)jmax * n_shells + s]);
                        kp_emiss[jmax] = -kp_emiss[jmax];  /* mark visited */
                    }
                    for (int j = 0; j < n_levels; j++)
                        if (kp_emiss[j] < 0.0) kp_emiss[j] = -kp_emiss[j];
                }
            }

            /* Path A free-free channel (ARTIS kpkt.cc:65): once a k-packet forms,
             * it converts to a continuum r-packet by free-free emission with prob
             * C_ff/(C_ff + C_collexc). C_ff = 1.426e-27·√T_e·n_e·Σ_ions(charge²·n_ion);
             * C_collexc = Σ kp_emiss (the collisional re-excitation weight = `tot`).
             * This is the thermalization SINK that lets UV energy leave the line
             * cascade as a thermal continuum photon (breaks the redward walk). */
            if (opacity->p_kpacket_ff) {
                double Te_s = plasma->T_e[s];
                double ne_s = (plasma->n_electron ? plasma->n_electron[s]
                                                  : opacity->electron_density[s]);
                double C_ff = 0.0, C_fb = 0.0, dom_edge_nu = 0.0, dom_n = 0.0;
                int dom_Z = 0, dom_stage = 0;   /* [FB-EDGE-METER] N9: who the edge came from */
                if (Te_s > 0.0 && ne_s > 0.0) {
                    double sum_z2n = 0.0;
                    double te4 = pow(Te_s / 1e4, -0.75);   /* Kramers recomb T-scaling */
                    double kTe = K_BOLTZMANN * Te_s;
                    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                        int stage = atom->ion_pop_stage[ip]; /* 0=neutral; free-free/recomb charge */
                        if (stage <= 0) continue;
                        double n_ion = atom->ion_number_density[(size_t)ip * n_shells + s];
                        sum_z2n += (double)stage * stage * n_ion;            /* free-free */
                        /* free-bound (recombination) cooling, ion stage -> stage-1.
                         * alpha ~ Kramers, charge^2 scaled; emitted energy ~ kT_e above edge. */
                        double alpha = 2.6e-13 * (double)stage * stage * te4; /* cm^3/s */
                        C_fb += alpha * n_ion * ne_s * kTe;                   /* erg/s/cm^3 */
                        /* representative recombination edge = the dominant recombining
                         * ion's edge (ionization energy of stage-1 -> stage). Fe-group
                         * edges are in the UV -> fb re-emits blue (toward ARTIS). */
                        if (n_ion > dom_n) {
                            dom_n = n_ion;
                            dom_Z = atom->ion_pop_Z[ip]; dom_stage = stage;
                            double chi = find_ioniz_energy(atom, atom->ion_pop_Z[ip], stage - 1);
                            dom_edge_nu = (chi > 0.0 && chi < 1e9)
                                ? chi * EV_TO_ERG / H_PLANCK : 0.0;
                        }
                    }
                    C_ff = 1.426e-27 * sqrt(Te_s) * ne_s * sum_z2n;
                }
                /* [FB-EDGE-METER] N9: a dominant recombining ion exists but its edge
                 * lookup failed (find_ioniz_energy sentinel) -> kpacket_fb_nu[s]=0 and
                 * every fb exit in this shell degenerates on the device. Count only. */
                if (dom_n > 0.0 && dom_edge_nu <= 0.0) {
                    #pragma omp critical (fb_edge_meter)
                    {
                        g_fb_dom_edge_fail++;
                        g_fb_dom_edge_fail_z     = dom_Z;
                        g_fb_dom_edge_fail_stage = dom_stage;
                    }
                }
                double denom = C_ff + C_fb + tot;            /* tot = C_collexc */
                opacity->p_kpacket_ff[s] = (denom > 0.0) ? (C_ff / denom) : 0.0;
                opacity->p_kpacket_fb[s] = (denom > 0.0) ? (C_fb / denom) : 0.0;
                opacity->kpacket_fb_nu[s] = dom_edge_nu;

                /* [FB-MULTI] per-continuum fb edge table for THIS shell. Enumerate
                 * every recombining continuum (Z, stage k>=1 -> k-1) with n_ion>0,
                 * weight by the fb energy-emission proxy
                 *   w = n_e * n_ion * alpha_tot * (h*nu0 + kB*T_e),
                 * keep the top KPKT_FB_NEDGE by weight, store normalized cumulative
                 * weights. alpha_tot = the SAME Milne radiative-recombination
                 * (CMFGEN sigma_bf) + Badnell DR (when LUMINA_FROZENIN_DR) the
                 * ionization balance uses (frozenin_alpha_rr, exactly as simul_ladder
                 * calls it). Edge freq nu0 = ionization threshold of the product stage
                 * k-1 (same convention as the single dom_edge above). */
                if (kpkt_fb_multi && opacity->kpacket_fb_edge_nu &&
                    Te_s > 0.0 && ne_s > 0.0) {
                    double kTe2 = K_BOLTZMANN * Te_s;
                    double e_nu[KPKT_FB_NEDGE], e_w[KPKT_FB_NEDGE];
                    int    e_zs[KPKT_FB_NEDGE];
                    int    e_lev[KPKT_FB_NEDGE];      /* [FB-MILNE C2] recombined-level idx per edge (-1 = per-ion) */
                    memset(e_lev, 0xFF, sizeof(e_lev)); /* init -1 */
                    int    nedge = 0;
                    /* [FB-MULTI] Fix C: physical free-bound sum over ALL recombining
                     * continua (not just the top-KPKT_FB_NEDGE kept for the CDF),
                     * reusing the same per-ion frozenin_alpha_rr weights. Replaces
                     * the Kramers C_fb in p_fb below. */
                    /* [FB-MILNE] LUMINA_FB_MILNE_EXACT: replace the per-ion Kramers/
                     * alpha_RR fb cooling (w = n_e n_ion alpha (h nu0 + kTe)) with the
                     * EXACT per-LEVEL radiative-recombination Milne cooling integral
                     * over the actual CMFGEN sigma_bf (ARTIS ratecoeff.cc parity), so
                     * the fb emissivity is the detailed-balance partner of chi_bf ->
                     * sub-Planckian EUV by construction, general (no per-case tuning).
                     * Requires cmfgen_sigma_bf loaded; OFF => byte-identical per-ion. */
                    static int fb_milne_exact = -1;
                    if (fb_milne_exact < 0) {
                        const char *e = getenv("LUMINA_FB_MILNE_EXACT");
                        fb_milne_exact = (e && atoi(e) && atom->cmfgen_sigma_bf &&
                                          atom->cmfgen_has_sigma) ? 1 : 0;
                        if (fb_milne_exact)
                            fprintf(stderr, "[FB-MILNE] exact per-level radiative-recombination "
                                    "bf cooling ON (Milne integral over CMFGEN sigma_bf, "
                                    "KE-charged; per-ion Kramers/alpha_RR fb path retired)\n");
                    }
                    double C_fb_real = 0.0;
                    if (!fb_milne_exact) {
                      for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                        int stage = atom->ion_pop_stage[ip]; /* recombining charge */
                        if (stage <= 0) continue;
                        double n_ion = atom->ion_number_density[(size_t)ip * n_shells + s];
                        if (n_ion <= 0.0) continue;
                        int Z = atom->ion_pop_Z[ip];
                        /* recombination product (stage-1) must be a tracked ion pop
                         * (skips Ti/Mn stage 1 whose neutral is absent). */
                        int ip_prod = find_ion_pop_idx(atom, Z, stage - 1);
                        if (ip_prod < 0) continue;
                        double chi = find_ioniz_energy(atom, Z, stage - 1); /* product edge */
                        if (!(chi > 0.0 && chi < 1e9)) continue;
                        double nu0 = chi * EV_TO_ERG / H_PLANCK;
                        double alpha = frozenin_alpha_rr(atom, ip_prod, ip, Te_s);
                        if (!(alpha > 0.0)) continue;
                        /* [FB-COOL-KT] fb thermal cooling weight: ARTIS charges
                         * the electron pool only the photoelectron kinetic energy
                         * (~kTe2), not the binding energy h*nu0 (ionization ledger).
                         * OFF => legacy (h*nu0 + kTe2), byte-identical. This single w
                         * feeds both C_fb_real (p_fb numerator) and the edge CDF. */
                        double w = ne_s * n_ion * alpha *
                                   (g_fb_cool_kt ? kTe2 : (H_PLANCK * nu0 + kTe2));
                        if (!(w > 0.0)) continue;
                        C_fb_real += w;   /* [FB-MULTI] Fix C: physical C_fb */
                        int zs = Z * 100 + stage;
                        if (nedge < KPKT_FB_NEDGE) {
                            e_nu[nedge] = nu0; e_w[nedge] = w; e_zs[nedge] = zs;
                            nedge++;
                        } else {
                            int jmin = 0;      /* evict the lightest if this is heavier */
                            for (int q = 1; q < KPKT_FB_NEDGE; q++)
                                if (e_w[q] < e_w[jmin]) jmin = q;
                            if (w > e_w[jmin]) { e_nu[jmin] = nu0; e_w[jmin] = w; e_zs[jmin] = zs; }
                        }
                      }
                    } else {
                      /* [FB-MILNE] exact per-LEVEL recombination: for each recombining
                       * ion (Z,stage) with pop n_upper, sum recombination TO every level
                       * l of the product ion (Z,stage-1) that carries a CMFGEN sigma_bf.
                       * Edge nu0_l = first non-zero sigma bin (CMFGEN threshold); weight
                       * w_l = n_e n_upper Lambda_l (Milne KE-cooling, ground-parent Saha
                       * g_lo/g_up). p_fb sums ALL levels; the CDF keeps top-NEDGE. */
                      double dln = log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS;
                      for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                        int stage = atom->ion_pop_stage[ip]; /* recombining charge */
                        if (stage <= 0) continue;
                        double n_upper = atom->ion_number_density[(size_t)ip * n_shells + s];
                        if (n_upper <= 0.0) continue;
                        int Z = atom->ion_pop_Z[ip];
                        int ip_prod = find_ion_pop_idx(atom, Z, stage - 1);
                        if (ip_prod < 0) continue;
                        double g_up = (double)atom->level_g[atom->level_offset[ip]]; /* upper-ion ground */
                        if (!(g_up > 0.0)) continue;
                        int l0 = atom->level_offset[ip_prod], l1 = atom->level_offset[ip_prod + 1];
                        for (int l = l0; l < l1; l++) {
                            if (!atom->cmfgen_has_sigma[l]) continue;
                            const double *sig = atom->cmfgen_sigma_bf +
                                                (size_t)l * NLTE_N_FREQ_BINS;
                            double nu0 = 0.0;
                            for (int f = 0; f < NLTE_N_FREQ_BINS; f++)
                                if (sig[f] > 0.0) { nu0 = NLTE_NU_MIN * exp(((double)f + 0.5) * dln); break; }
                            if (!(nu0 > 0.0)) continue;
                            double g_lo = (double)atom->level_g[l];
                            double Lam = fb_milne_cooling_coeff(sig, nu0, g_lo, g_up, Te_s);
                            if (!(Lam > 0.0)) continue;
                            double w = ne_s * n_upper * Lam;  /* erg/s/cm^3 cooling density */
                            if (!(w > 0.0)) continue;
                            C_fb_real += w;
                            int zs = Z * 100 + stage;
                            if (nedge < KPKT_FB_NEDGE) {
                                e_nu[nedge] = nu0; e_w[nedge] = w; e_zs[nedge] = zs;
                                e_lev[nedge] = l;   /* [FB-MILNE C2] recombined level for sigma_bf draw */
                                nedge++;
                            } else {
                                int jmin = 0;
                                for (int q = 1; q < KPKT_FB_NEDGE; q++)
                                    if (e_w[q] < e_w[jmin]) jmin = q;
                                if (w > e_w[jmin]) { e_nu[jmin] = nu0; e_w[jmin] = w; e_zs[jmin] = zs; e_lev[jmin] = l; }
                            }
                        }
                      }
                    }
                    double wtot = 0.0;
                    for (int q = 0; q < nedge; q++) wtot += e_w[q];
                    double *o_nu  = opacity->kpacket_fb_edge_nu  + (size_t)s * KPKT_FB_NEDGE;
                    double *o_cdf = opacity->kpacket_fb_edge_cdf + (size_t)s * KPKT_FB_NEDGE;
                    int    *o_zs  = opacity->kpacket_fb_edge_zstage + (size_t)s * KPKT_FB_NEDGE;
                    int    *o_lev = opacity->kpacket_fb_edge_lev ?
                                    opacity->kpacket_fb_edge_lev + (size_t)s * KPKT_FB_NEDGE : NULL;
                    if (wtot > 0.0) {
                        double acc = 0.0;
                        for (int q = 0; q < nedge; q++) {
                            acc += e_w[q];
                            o_nu[q]  = e_nu[q];
                            o_cdf[q] = acc / wtot;
                            o_zs[q]  = e_zs[q];
                            if (o_lev) o_lev[q] = e_lev[q];
                        }
                        o_cdf[nedge - 1] = 1.0;
                        opacity->kpacket_fb_edge_count[s] = nedge;
                    } else {
                        /* no valid continua -> GPU falls back to the single edge */
                        opacity->kpacket_fb_edge_count[s] = 0;
                    }

                    /* [FB-MULTI] Fix C: swap the Kramers C_fb for the physical
                     * C_fb_real in the k-packet exit probability. C_ff and
                     * C_collexc (tot) are unchanged; the p_fb/p_ff normalization
                     * denominator picks up C_fb_real. No second frozenin_alpha_rr:
                     * C_fb_real reused the loop weights above. */
                    double kramers_pfb = opacity->p_kpacket_fb[s];
                    double denom_real = C_ff + C_fb_real + tot;
                    if (denom_real > 0.0) {
                        opacity->p_kpacket_ff[s] = C_ff / denom_real;
                        opacity->p_kpacket_fb[s] = C_fb_real / denom_real;
                    }
                    if (s == 0 || s == n_shells / 3 || s == n_shells - 1)
                        printf("  [FB-MULTI] p_fb s%d: %.3e (code-Kramers %.3e)\n",
                               s, opacity->p_kpacket_fb[s], kramers_pfb);
                }
            }

            if (tot > 0.0) {
                double acc = 0.0;
                for (int j = 0; j < n_levels; j++) {
                    acc += kp_emiss[j];
                    cdf[j] = acc / tot;
                }
                cdf[n_levels - 1] = 1.0;
            } else {
                /* no collisional weight: degenerate uniform (rarely hit; the
                 * GPU side only samples when a k-packet actually forms). */
                for (int j = 0; j < n_levels; j++)
                    cdf[j] = (double)(j + 1) / (double)n_levels;
            }
        }
    }

    free(rates_buf);
    free(kp_emiss);
    free(kpd_fe_arr);
    if (g_ctp_iup_binfield) {   /* [IUP-BINFIELD] fold thread-local counters */
        #pragma omp atomic
        g_iup_binfield_used   += binf_used_loc;
        #pragma omp atomic
        g_iup_binfield_bypass += binf_bypass_loc;
    }
    if (g_ctp_iup_jblue) {   /* [IUP-JBLUE] fold thread-local counters */
        #pragma omp atomic
        g_iup_jblue_used += jblue_used_loc;
        #pragma omp atomic
        g_iup_jblue_fb += jblue_fb_loc;
        /* [JBLUE-ANCHOR] fold thin/thick bucket accumulators */
        #pragma omp atomic
        g_jba_thin_n     += jba_thin_n_loc;
        #pragma omp atomic
        g_jba_thick_n    += jba_thick_n_loc;
        #pragma omp atomic
        g_jba_thin_sum   += jba_thin_sum_loc;
        #pragma omp atomic
        g_jba_thick_sum  += jba_thick_sum_loc;
        #pragma omp atomic
        g_jba_thin_clamp += jba_thin_clamp_loc;
        #pragma omp atomic
        g_jba_thick_clamp += jba_thick_clamp_loc;
        /* [JBLUE-ANCHOR2] fold the 4 buckets */
        for (int _b = 0; _b < 4; _b++) {
            #pragma omp atomic
            g_jba4_n[_b]   += jba4_n_loc[_b];
            #pragma omp atomic
            g_jba4_sum[_b] += jba4_sum_loc[_b];
            #pragma omp atomic
            g_jba4_clo[_b] += jba4_clo_loc[_b];
            #pragma omp atomic
            g_jba4_chi[_b] += jba4_chi_loc[_b];
        }
        #pragma omp atomic
        g_jba_all_n += jba_all_n_loc;
    }
    }   /* end omp parallel */

    /* [MA-RADRECOMB tau-gate] banner: over all (shell, is_emit-eligible entry)
     * pairs, how many emit the MC continuum (edge optically THIN) vs stay
     * on-the-spot (thick, INTERNALDOWNLOWER). Serial scan after the parallel
     * region (recomb_emit_shell is s-indexed, fully written). Only under the
     * rr gate (recomb_emit_shell != NULL). */
    if (opacity->recomb_emit_shell && opacity->recomb_is_emit) {
        long n_elig = 0, pairs_emit = 0, pairs_spot = 0;
        for (int k = 0; k < opacity->n_recomb; k++) {
            if (!opacity->recomb_is_emit[k]) continue;
            n_elig++;
            for (int s = 0; s < n_shells; s++) {
                if (opacity->recomb_emit_shell[(size_t)k * n_shells + s])
                    pairs_emit++;
                else
                    pairs_spot++;
            }
        }
        printf("  [MA-RADRECOMB tau-gate] thresh=%.3g: %ld eligible entries x %d "
               "shells => %ld (shell,entry) EMIT (edge thin) / %ld on-the-spot "
               "(edge thick, INTERNALDOWNLOWER)\n",
               rr_tau_thresh, n_elig, n_shells, pairs_emit, pairs_spot);
    }

    /* [MA-RADRECOMB tau-gate] one-time verification banner: for the ground edge of
     * a few reference destination ions (S II, Fe III, Ni III) print the exact
     * tau_bf ingredients (sigma_edge, n_dest, dr, tau) at shell 8, reproducing the
     * per-shell classifier so the numbers can be checked against the CMFGEN edge
     * opacity (S II ground ~thick). Serial, first call only. */
    if (opacity->recomb_emit_shell && opacity->recomb_is_emit &&
        opacity->recomb_sigma_edge) {
        static int rr_tau_dbg_done = 0;
        if (!rr_tau_dbg_done) {
            rr_tau_dbg_done = 1;
            int sdbg = (n_shells > 8) ? 8 : (n_shells - 1);
            double dr8 = geom ? (geom->r_outer[sdbg] - geom->r_inner[sdbg]) : 0.0;
            double Te8 = plasma ? plasma->T_e[sdbg] : 0.0;
            struct { int Z, ion; const char *name; } tgt[3] = {
                {16, 1, "S II  ground"}, {26, 2, "Fe III ground"},
                {28, 2, "Ni III ground"} };
            for (int t = 0; t < 3; t++) {
                int kbest = -1; double ebest = 0.0;
                for (int k = 0; k < opacity->n_recomb; k++) {
                    if (!opacity->recomb_is_emit[k]) continue;
                    int jj = opacity->recomb_dest_level[k];
                    if (atom->level_Z[jj] != tgt[t].Z ||
                        atom->level_ion[jj] != tgt[t].ion) continue;
                    if (kbest < 0 || atom->level_energy_eV[jj] < ebest) {
                        kbest = k; ebest = atom->level_energy_eV[jj];
                    }
                }
                if (kbest < 0) {
                    printf("  [MA-RADRECOMB tau-gate] sample s=%d %s: no is_emit "
                           "recomb entry\n", sdbg, tgt[t].name);
                    continue;
                }
                int j = opacity->recomb_dest_level[kbest];
                int ip_lo = find_ion_pop_idx(atom, atom->level_Z[j],
                                             atom->level_ion[j]);
                double se = opacity->recomb_sigma_edge[kbest];
                double n_j = 0.0; const char *nsrc = "NLTE";
                int n_j_from_solve = 0;
                if (nlte && nlte->nlte_level_populations &&
                    nlte->global_to_nlte_level &&
                    nlte->population_committed_generation > 0) {
                    int gnl = nlte->global_to_nlte_level[j];
                    if (gnl >= 0) {
                        n_j = nlte->nlte_level_populations[
                                  (size_t)gnl * n_shells + sdbg];
                        n_j_from_solve = 1;
                    }
                }
                if (!n_j_from_solve && ip_lo >= 0 && atom->ion_number_density &&
                    atom->partition_functions && Te8 > 0.0) {
                    double Zp = atom->partition_functions[
                                    (size_t)ip_lo * n_shells + sdbg];
                    double ni = atom->ion_number_density[
                                    (size_t)ip_lo * n_shells + sdbg];
                    if (Zp > 0.0 && ni > 0.0) {
                        PopulationAtomicView av = population_atomic_view(atom);
                        double frac = 0.0;
                        PopulationStatus ps = population_lte_level_fraction(
                            &av, (size_t)ip_lo, (size_t)j, Te8, Zp, &frac);
                        if (ps == POP_OK || ps == POP_EXACT_ZERO)
                            n_j = ni * frac;
                    }
                    nsrc = "LTE@Te";
                }
                double tau = se * n_j * dr8;
                printf("  [MA-RADRECOMB tau-gate] sample s=%d %s (glev %d, E=%.3f eV): "
                       "sigma_edge=%.3e cm2  n_dest=%.3e cm-3 (%s)  dr=%.3e cm  "
                       "tau=%.3e -> %s\n",
                       sdbg, tgt[t].name, j, atom->level_energy_eV[j], se, n_j, nsrc,
                       dr8, tau, (tau <= rr_tau_thresh) ? "EMIT" : "on-the-spot");
            }
        }
    }

    if (kpacket_mode) {
        /* Diagnostic: mean k-packet deactivation prob at inner/outer shell. */
        double pk0 = 0.0, pkN = 0.0; long n0 = 0, nN = 0;
        int sN = n_shells - 1;
        for (int lev = 0; lev < n_levels; lev++) {
            double a = opacity->p_kpacket[(size_t)lev * n_shells + 0];
            double b = opacity->p_kpacket[(size_t)lev * n_shells + sN];
            if (a > 0.0) { pk0 += a; n0++; }
            if (b > 0.0) { pkN += b; nN++; }
        }
        printf("  [KPACKET] mean p_kpacket: shell0=%.3e (%ld lev>0)  shell%d=%.3e (%ld lev>0)\n",
               n0 ? pk0 / n0 : 0.0, n0, sN, nN ? pkN / nN : 0.0, nN);
    }

    /* [MA-LINE-DESTRUCT] verification banner (serial, first call only): for the
     * deepest representative Co III / Fe III / Ni III UV resonance line at shell 8,
     * print the exact eps = C_ul/(C_ul + A_ul*beta) ingredients (tau, beta, A_ul,
     * C_ul, eps) plus the per-ion mean eps over all that ion's UV lines at that shell
     * (the rate-level, ion-resolved destruction fraction — DIAGNOSTIC ONLY, the
     * shipped physics is uniform over all lines/shells/ions). Prints only when the
     * gate is armed (ma_line_eps != NULL). Confirms deep tau>>1 lines are
     * destruction-dominated (eps -> 1). */
    if (opacity->ma_line_eps) {
        static int mald_dbg_done = 0;
        if (!mald_dbg_done) {
            mald_dbg_done = 1;
            int sdbg = (n_shells > 8) ? 8 : (n_shells - 1);
            double Te8 = plasma ? plasma->T_e[sdbg] : 0.0;
            double ne8 = (plasma && plasma->n_electron) ? plasma->n_electron[sdbg]
                         : (opacity->electron_density ? opacity->electron_density[sdbg] : 0.0);
            double is_te8 = (Te8 > 0.0) ? 1.0 / sqrt(Te8) : 0.0;
            struct { int Z, ion; const char *name; } tgt[3] = {
                {27, 2, "Co III"}, {26, 2, "Fe III"}, {28, 2, "Ni III"} };
            printf("  [MA-LINE-DESTRUCT] ARMED eps=C_ul/(C_ul+A_ul*beta) destroyed "
                   "photon -> k-packet thermal pool (uniform all lines/shells/ions). "
                   "Sample s=%d T_e=%.0fK n_e=%.3e:\n", sdbg, Te8, ne8);
            for (int t = 0; t < 3; t++) {
                double eps_sum = 0.0, b_tau = -1.0, b_eps = 0.0, b_beta = 0.0;
                double b_A = 0.0, b_C = 0.0, b_lam = 0.0;
                long eps_n = 0;
                for (int l = 0; l < atom->n_lines; l++) {
                    if (atom->line_atomic_number[l] != tgt[t].Z ||
                        atom->line_ion_number[l] != tgt[t].ion) continue;
                    double lam = 2.99792458e18 / atom->line_nu[l];
                    if (lam < 1000.0 || lam > 3200.0) continue;   /* UV window */
                    if (kp_glo[l] < 0 || kp_gup[l] < 0) continue;
                    double tau  = opacity->tau_sobolev[(size_t)l * n_shells + sdbg];
                    double beta = beta_sobolev(tau);
                    double g_up = (double)atom->level_g[kp_gup[l]];
                    double g_lo = (double)atom->level_g[kp_glo[l]];
                    double f_lu = atom->line_f_lu[l];
                    double dE   = H_PLANCK * atom->line_nu[l];
                    double C_down;
                    if (artis_parity_enabled()) {
                        double cu, cd; int forb = (f_lu <= 1e-10);
                        /* [MA-REAL-UPSILON] mirror the shipped substitution so this
                         * diagnostic eps reflects the eps the lottery actually uses. */
                        double cs_m = forb ? -2.0 : -1.0;
                        if (ma_real_ups && ma_ru_line_src[l] >= 0) {
                            double ups = ma_ru_upsilon(&ma_ru_src_reg[ma_ru_line_src[l]],
                                                       ma_ru_line_t[l], Te8);
                            if (ups > 0.0) cs_m = ups;
                        }
                        artis_col_rates(Te8, ne8, dE, g_lo, g_up, f_lu,
                                        cs_m, forb, &cu, &cd);
                        C_down = cd;
                    } else {
                        C_down = (f_lu > 1e-10)
                            ? VAN_REG_COEFF * ne8 * f_lu * 0.2 * is_te8 / g_up
                            : 8.63e-6 * ne8 * AX_OMEGA * is_te8 / g_up;
                    }
                    double rad = atom->line_A_ul[l] * beta;
                    double eps = (C_down + rad > 0.0) ? C_down / (C_down + rad) : 0.0;
                    eps_sum += eps; eps_n++;
                    if (tau > b_tau) {
                        b_tau = tau; b_eps = eps; b_beta = beta;
                        b_A = atom->line_A_ul[l]; b_C = C_down; b_lam = lam;
                    }
                }
                if (eps_n == 0) {
                    printf("    %s: no collisionally-mapped UV line at s=%d\n",
                           tgt[t].name, sdbg);
                    continue;
                }
                printf("    %s deepest-UV %.1fA: tau=%.3e beta=%.3e A_ul=%.3e "
                       "C_ul=%.3e -> eps=%.4f | ion-mean eps(UV)=%.4f over %ld lines\n",
                       tgt[t].name, b_lam, b_tau, b_beta, b_A, b_C, b_eps,
                       eps_sum / eps_n, eps_n);
            }
        }
    }
    /* [MA-REAL-UPSILON] (Fix-P1) T3 verification banner (serial, first call): the
     * global real-table hit/fallback census (fail-loud) + for three representative
     * covered transitions (Fe III thick EUV forbidden, Co III thick, Fe III allowed)
     * the C_ul(vR) vs C_ul(realUpsilon) and eps_before/after at a sample shell, so
     * the drain change is directly auditable. Prints only when the gate is armed. */
    if (ma_real_ups) {
        static int maru_dbg_done = 0;
        if (!maru_dbg_done) {
            maru_dbg_done = 1;
            int sdbg = (n_shells > 8) ? 8 : (n_shells - 1);
            double Te = plasma ? plasma->T_e[sdbg] : 0.0;
            double ne = (plasma && plasma->n_electron) ? plasma->n_electron[sdbg]
                        : (opacity->electron_density ? opacity->electron_density[sdbg] : 0.0);
            printf("  [MA-REAL-UPSILON] census: %ld lines hit real Omega, %ld covered-ion "
                   "fallback (vR); %d ion tables. Sample s=%d T_e=%.0fK n_e=%.3e:\n",
                   ma_ru_nhit, ma_ru_nfall, ma_ru_nsrc, sdbg, Te, ne);
            if (ma_ru_nhit == 0)
                fprintf(stderr, "  [MA-REAL-UPSILON][WARN] 0 real-table hits at banner — "
                                "verify LUMINA_ARTIS_PARITY=1 and loaded col tables\n");
            struct { int Z, ion, mode; const char *name; } cat[3] = {
                {26, 2, 0, "Fe III forbid"},   /* mode 0: f_lu<=1e-10 (thick EUV) */
                {27, 2, 1, "Co III thick"},    /* mode 1: any covered, deepest tau */
                {26, 2, 2, "Fe III allowed"} };/* mode 2: f_lu>0.01 */
            for (int c = 0; c < 3; c++) {
                int bl = -1; double btau = -1.0;
                for (int l = 0; l < atom->n_lines; l++) {
                    if (atom->line_atomic_number[l] != cat[c].Z ||
                        atom->line_ion_number[l] != cat[c].ion) continue;
                    if (ma_ru_line_src[l] < 0) continue;   /* real-covered only */
                    if (!kp_glo || !kp_gup || kp_glo[l] < 0 || kp_gup[l] < 0) continue;
                    double f = atom->line_f_lu[l];
                    if (cat[c].mode == 0 && !(f <= 1e-10)) continue;
                    if (cat[c].mode == 2 && !(f > 0.01)) continue;
                    double tau = opacity->tau_sobolev[(size_t)l * n_shells + sdbg];
                    if (tau > btau) { btau = tau; bl = l; }
                }
                if (bl < 0) {
                    printf("    %-14s: no real-covered line matched\n", cat[c].name);
                    continue;
                }
                double f = atom->line_f_lu[bl];
                int forb = (f <= 1e-10);
                double g_up = (double)atom->level_g[kp_gup[bl]];
                double g_lo = (double)atom->level_g[kp_glo[bl]];
                double dE  = H_PLANCK * atom->line_nu[bl];
                double lam = 2.99792458e18 / atom->line_nu[bl];
                double beta = beta_sobolev(btau);
                double A = atom->line_A_ul[bl];
                double cu_v, cd_v, cu_r, cd_r;
                artis_col_rates(Te, ne, dE, g_lo, g_up, f, forb ? -2.0 : -1.0, forb,
                                &cu_v, &cd_v);
                double ups = ma_ru_upsilon(&ma_ru_src_reg[ma_ru_line_src[bl]],
                                           ma_ru_line_t[bl], Te);
                artis_col_rates(Te, ne, dE, g_lo, g_up, f, ups, forb, &cu_r, &cd_r);
                double rad = A * beta;
                double eps_b = (cd_v + rad > 0.0) ? cd_v / (cd_v + rad) : 0.0;
                double eps_a = (cd_r + rad > 0.0) ? cd_r / (cd_r + rad) : 0.0;
                printf("    %-14s %7.1fA f=%.2e tau=%.2e beta=%.2e Ups=%.3g | "
                       "C_ul vR=%.3e real=%.3e (x%.1f) | eps %.4f -> %.4f\n",
                       cat[c].name, lam, f, btau, beta, ups, cd_v, cd_r,
                       (cd_v > 0.0 ? cd_r / cd_v : 0.0), eps_b, eps_a);
            }
        }
    }
    /* [FB-MULTI] diagnostic record: per shell, the index j of the Si II edge
     * (Z=14, stage 2->1 => zstage code 1402) in the per-continuum table, and its
     * cumulative-weight share. Reported for a few representative shells. */
    if (kpkt_fb_multi && opacity->kpacket_fb_edge_zstage) {
        int probes[3] = { 0, n_shells / 3, n_shells - 1 };
        for (int pk = 0; pk < 3; pk++) {
            int s = probes[pk];
            if (s < 0 || s >= n_shells) continue;
            int cnt = opacity->kpacket_fb_edge_count[s];
            const int    *zs  = opacity->kpacket_fb_edge_zstage + (size_t)s * KPKT_FB_NEDGE;
            const double *cdf = opacity->kpacket_fb_edge_cdf    + (size_t)s * KPKT_FB_NEDGE;
            int si2 = -1;
            for (int q = 0; q < cnt; q++) if (zs[q] == 1402) { si2 = q; break; }
            double share = 0.0;
            if (si2 >= 0)
                share = cdf[si2] - (si2 > 0 ? cdf[si2 - 1] : 0.0);
            printf("  [FB-MULTI] shell%2d: %d continua, SiII-edge idx=%d share=%.1f%%\n",
                   s, cnt, si2, 100.0 * share);
        }
    }
    printf("  [TransProb] Recomputed %d transitions x %d shells (damping=%s, J_src=%s, j_cap=%.2g, j_floor=%.2g, W1=%.2g W2=%.2g W3=%.2g W4=%.2g[%g-%g] uv_idown=%.2g[<%.0fÅ])\n",
           n_trans, n_shells, apply_damping ? "on" : "off",
           use_j_nu ? "MC_histogram" : "W*Bnu",
           j_cap_effective, j_floor_effective,
           uvopt_emit_boost, uvopt_emit_boost2, uvopt_emit_boost3,
           uvopt_emit_boost4, uvopt_lam_min4, uvopt_lam_max4,
           macro_uv_idown_factor, macro_uv_idown_thresh);

    /* Diagnose at multiple shells to expose inner Fe II vs outer C/S/Si physics. */
    diag_macro_branch(atom, plasma, opacity, 0);
    if (n_shells > 3)  diag_macro_branch(atom, plasma, opacity, 3);
    if (n_shells > 10) diag_macro_branch(atom, plasma, opacity, n_shells / 3);
}

/* [MA-BRANCH] Effective branching diagnostic.
 * For each macro-atom level, find its strongest Sobolev line at `diag_shell`,
 * bin the level by that strong-line's band, and report aggregated branching
 * probabilities (p_emit / p_iup / p_idn) across ALL its transitions.
 *
 * UV-strong (band 0) levels get an extra breakdown:
 *   - destination band of BB-emit (where does UV resonance scatter land?)
 *   - per-(Z, ion) split (which species own UV-strong levels?)
 *   - first 2 Fe II UV-strong levels printed transition-by-transition.
 *
 * Reads `transition_probabilities[]` and `tau_sobolev[]` as stored — does NOT
 * recompute. Safe to call any time after compute_plasma_state(). */
void diag_macro_branch(AtomicData *atom, PlasmaState *plasma,
                       OpacityState *opacity, int diag_shell) {
    int n_shells = opacity->n_shells;
    int n_levels = opacity->n_macro_levels;
    (void)plasma;
    {
        const double tau_strong = 1.0;
        int n_per_band[MA_FATE_NBANDS] = {0};
        double sum_emit[MA_FATE_NBANDS] = {0};
        double sum_iup [MA_FATE_NBANDS] = {0};
        double sum_idn [MA_FATE_NBANDS] = {0};
        /* Band-resolved BB-emission destination, per strong-line band group.
         * sum_emit_dest[group][dest_band] = ∑ p_emit landing in dest_band */
        double sum_emit_dest[MA_FATE_NBANDS][MA_FATE_NBANDS] = {{0}};
        /* Per-(Z, ion) breakdown for UV-strong-line levels (band 0).
         * Tracks: n_levels, sum p_iup, sum p_idn, sum p_emit per dest band. */
        #define DIAG_ZMAX 31
        #define DIAG_IMAX 5
        int    ion_n   [DIAG_ZMAX][DIAG_IMAX] = {{0}};
        double ion_iup [DIAG_ZMAX][DIAG_IMAX] = {{0}};
        double ion_idn [DIAG_ZMAX][DIAG_IMAX] = {{0}};
        double ion_emit_dest[DIAG_ZMAX][DIAG_IMAX][MA_FATE_NBANDS] = {{{0}}};
        /* [3000,3100]Å bump sub-band BB-emit per-(Z, ion) tracker.
         * Independent of max_band — picks up ANY upper level that has at least
         * one BB-emit channel landing in the bump window, regardless of which
         * band its strongest line lives in. */
        double bump_emit_per_ion[DIAG_ZMAX][DIAG_IMAX] = {{0}};
        int    bump_lev_per_ion[DIAG_ZMAX][DIAG_IMAX] = {{0}};
        int n_idle = 0, n_weak = 0;
        int n_dumped_fe2_uv = 0;
        const int N_DUMP_FE2_UV = 2;
        const double C_LIGHT_AA = 2.99792458e18;  /* Hz·Å */

        for (int lev = 0; lev < n_levels; lev++) {
            int block_start = opacity->macro_block_references[lev];
            int block_end   = opacity->macro_block_references[lev + 1];
            if (block_start >= block_end) { n_idle++; continue; }

            /* Find the strongest line touching this level (any ttype). */
            double tau_max  = 0.0;
            int    max_band = -1;
            for (int tid = block_start; tid < block_end; tid++) {
                int line_id = opacity->transition_line_id[tid];
                if (line_id < 0 || line_id >= atom->n_lines) continue;
                double tau = opacity->tau_sobolev[line_id * n_shells + diag_shell];
                if (tau > tau_max) {
                    tau_max  = tau;
                    max_band = macro_atom_fate_band_from_nu(atom->line_nu[line_id]);
                }
            }
            if (tau_max < tau_strong || max_band < 0) { n_weak++; continue; }

            double p_emit_band[MA_FATE_NBANDS] = {0};
            double p_iup = 0.0, p_idn = 0.0;
            double level_bump_p_emit = 0.0;  /* BB-emit weight in [3000,3100]Å */
            /* Element of this level: take from strongest line. */
            int strong_line_id = -1;
            for (int tid = block_start; tid < block_end; tid++) {
                int line_id = opacity->transition_line_id[tid];
                if (line_id < 0 || line_id >= atom->n_lines) continue;
                double tau = opacity->tau_sobolev[line_id * n_shells + diag_shell];
                if (tau == tau_max) { strong_line_id = line_id; break; }
            }
            int level_Z   = strong_line_id >= 0 ? atom->line_atomic_number[strong_line_id] : -1;
            int level_ion = strong_line_id >= 0 ? atom->line_ion_number[strong_line_id]    : -1;

            for (int tid = block_start; tid < block_end; tid++) {
                double p    = opacity->transition_probabilities[tid * n_shells + diag_shell];
                int    ttype  = opacity->transition_type[tid];
                int    line_id = opacity->transition_line_id[tid];
                if (ttype == -1) {
                    if (line_id >= 0 && line_id < atom->n_lines) {
                        int b = macro_atom_fate_band_from_nu(atom->line_nu[line_id]);
                        p_emit_band[b] += p;
                        double lam_em = C_LIGHT_AA / atom->line_nu[line_id];
                        if (lam_em >= 3000.0 && lam_em <= 3100.0)
                            level_bump_p_emit += p;
                    }
                } else if (ttype == 0) {
                    p_idn += p;
                } else if (ttype == 1) {
                    p_iup += p;
                }
            }

            double p_emit_tot = 0.0;
            for (int b = 0; b < MA_FATE_NBANDS; b++) p_emit_tot += p_emit_band[b];
            n_per_band[max_band]++;
            sum_emit[max_band] += p_emit_tot;
            sum_iup [max_band] += p_iup;
            sum_idn [max_band] += p_idn;
            for (int b = 0; b < MA_FATE_NBANDS; b++)
                sum_emit_dest[max_band][b] += p_emit_band[b];

            /* UV-strong levels: track per-(Z, ion) breakdown */
            if (max_band == 0 && level_Z >= 0 && level_Z < DIAG_ZMAX
                && level_ion >= 0 && level_ion < DIAG_IMAX) {
                ion_n  [level_Z][level_ion]++;
                ion_iup[level_Z][level_ion] += p_iup;
                ion_idn[level_Z][level_ion] += p_idn;
                for (int b = 0; b < MA_FATE_NBANDS; b++)
                    ion_emit_dest[level_Z][level_ion][b] += p_emit_band[b];
            }
            /* [3000,3100]Å bump per-(Z, ion): any level with non-zero bump emit */
            if (level_bump_p_emit > 0.0
                && level_Z >= 0 && level_Z < DIAG_ZMAX
                && level_ion >= 0 && level_ion < DIAG_IMAX) {
                bump_emit_per_ion[level_Z][level_ion] += level_bump_p_emit;
                bump_lev_per_ion[level_Z][level_ion]++;
            }

            /* Dump first N_DUMP Fe II UV-strong levels that ALSO have BB-emission
             * exits (skip ground-multiplet members with only I-UP). These are
             * the cascade-relevant upper levels of UV resonance lines. */
            int has_emit = (p_emit_band[0] + p_emit_band[1] + p_emit_band[2]
                            + p_emit_band[3]) > 0.0;
            if (max_band == 0 && level_Z == 26 && level_ion == 1 && has_emit &&
                n_dumped_fe2_uv < N_DUMP_FE2_UV) {
                printf("    [MA-DUMP] Fe II UV-strong level lev=%d (Z=%d ion=%d) "
                       "tau_max=%.2e at shell %d, block size=%d\n",
                       lev, level_Z, level_ion, tau_max, diag_shell,
                       block_end - block_start);
                printf("      tid  ttype  line_id    lambda(A)     A_ul[s^-1]    "
                       "tau_sob       p\n");
                for (int tid = block_start; tid < block_end; tid++) {
                    double p     = opacity->transition_probabilities[tid * n_shells + diag_shell];
                    int    ttype = opacity->transition_type[tid];
                    int    lid   = opacity->transition_line_id[tid];
                    double lam = 0.0, A_ul = 0.0, tau = 0.0;
                    if (lid >= 0 && lid < atom->n_lines) {
                        lam   = C_LIGHT_AA / atom->line_nu[lid];
                        A_ul  = atom->line_A_ul[lid];
                        tau   = opacity->tau_sobolev[lid * n_shells + diag_shell];
                    }
                    const char *ts =
                        ttype == -1 ? "EMIT  " :
                        ttype ==  0 ? "I-DN  " :
                        ttype ==  1 ? "I-UP  " : "?     ";
                    printf("      %4d %s %7d  %10.2f   %10.3e   %10.3e  %10.3e\n",
                           tid, ts, lid, lam, A_ul, tau, p);
                }
                n_dumped_fe2_uv++;
            }
        }

        static const char *band_name[MA_FATE_NBANDS] = {
            "UVblnk", "CaIIKb", "UVtgt ", "fluor ",
            "green ", "red   ", "NIR1  ", "NIR2  "};
        printf("  [MA-BRANCH] Strong-line (tau > %.1f) branching at shell %d\n",
               tau_strong, diag_shell);
        printf("    bin by strongest-line band  |    n  | <p_emit> | <p_iup> | <p_idn>\n");
        for (int b = 0; b < MA_FATE_NBANDS; b++) {
            if (n_per_band[b] == 0) continue;
            double inv = 1.0 / (double)n_per_band[b];
            printf("    strong-line in %s        | %5d |  %6.4f  | %6.4f  | %6.4f\n",
                   band_name[b], n_per_band[b],
                   sum_emit[b] * inv, sum_iup[b] * inv, sum_idn[b] * inv);
        }
        printf("    weak-line (tau<=%.1f, skipped) | %5d |\n", tau_strong, n_weak);
        if (n_idle) printf("    orphan/empty block          | %5d |\n", n_idle);

        /* BB-emission destination band, normalized by total UV-strong p_emit. */
        if (n_per_band[0] > 0) {
            double s = sum_emit[0]; if (s <= 0) s = 1.0;
            printf("    UV-strong levels' BB-emit destination band (%% of p_emit):\n     ");
            for (int b = 0; b < MA_FATE_NBANDS; b++)
                printf(" %s=%5.1f%%", band_name[b], 100.0 * sum_emit_dest[0][b] / s);
            printf("\n");
            double up = sum_iup[0] / (double)n_per_band[0];
            double dn = sum_idn[0] / (double)n_per_band[0];
            printf("    UV-strong-group: <p_iup>/<p_idn>=%.2f"
                   "  (Mazzali-Lucy expects <p_idn> >> <p_iup>;"
                   " ratio>=1 => J_nu(UV) pumping the cascade up).\n",
                   dn > 0 ? up / dn : -1.0);
        }

        /* Per-(Z, ion) breakdown for UV-strong-line levels.
         * Rank ions by n_levels; print top 6 with their fluor/UV/iup/idn mass. */
        {
            int idx_Z[DIAG_ZMAX * DIAG_IMAX], idx_I[DIAG_ZMAX * DIAG_IMAX];
            int n_ions = 0;
            for (int Z = 1; Z < DIAG_ZMAX; Z++)
                for (int I = 0; I < DIAG_IMAX; I++)
                    if (ion_n[Z][I] > 0) {
                        idx_Z[n_ions] = Z; idx_I[n_ions] = I; n_ions++;
                    }
            for (int i = 0; i < n_ions; i++)
                for (int j = i + 1; j < n_ions; j++)
                    if (ion_n[idx_Z[j]][idx_I[j]] > ion_n[idx_Z[i]][idx_I[i]]) {
                        int tz = idx_Z[i]; idx_Z[i] = idx_Z[j]; idx_Z[j] = tz;
                        int ti = idx_I[i]; idx_I[i] = idx_I[j]; idx_I[j] = ti;
                    }
            int top = n_ions < 6 ? n_ions : 6;
            if (top > 0) {
                printf("    UV-strong levels by ion (top %d by count):\n", top);
                printf("       ion       n  | <p_iup> <p_idn> | UV%%  CaK%%  Utg%%  flu%%  grn%%  red%% NIR1%% NIR2%%\n");
                for (int k = 0; k < top; k++) {
                    int Z = idx_Z[k], I = idx_I[k];
                    int n = ion_n[Z][I]; if (n <= 0) continue;
                    double inv = 1.0 / (double)n;
                    double esum = 0.0;
                    for (int b = 0; b < MA_FATE_NBANDS; b++) esum += ion_emit_dest[Z][I][b];
                    if (esum <= 0.0) esum = 1.0;
                    /* Roman numeral for ion stage. */
                    static const char *rom[5] = {"I  ", "II ", "III", "IV ", "V  "};
                    printf("      Z=%2d %s %5d | %6.4f  %6.4f | "
                           "%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f\n",
                           Z, rom[I], n,
                           ion_iup[Z][I] * inv, ion_idn[Z][I] * inv,
                           100.0 * ion_emit_dest[Z][I][0] / esum,
                           100.0 * ion_emit_dest[Z][I][1] / esum,
                           100.0 * ion_emit_dest[Z][I][2] / esum,
                           100.0 * ion_emit_dest[Z][I][3] / esum,
                           100.0 * ion_emit_dest[Z][I][4] / esum,
                           100.0 * ion_emit_dest[Z][I][5] / esum,
                           100.0 * ion_emit_dest[Z][I][6] / esum,
                           100.0 * ion_emit_dest[Z][I][7] / esum);
                }
            }
        }

        /* [3000,3100]Å bump BB-emit per-(Z, ion) — identifies which iron-peak
         * species generates the residual bump in champion 152761. */
        {
            double bump_total = 0.0;
            for (int Z = 1; Z < DIAG_ZMAX; Z++)
                for (int I = 0; I < DIAG_IMAX; I++)
                    bump_total += bump_emit_per_ion[Z][I];
            if (bump_total > 0.0) {
                int idx_Z[DIAG_ZMAX * DIAG_IMAX], idx_I[DIAG_ZMAX * DIAG_IMAX];
                int n_bump = 0;
                for (int Z = 1; Z < DIAG_ZMAX; Z++)
                    for (int I = 0; I < DIAG_IMAX; I++)
                        if (bump_emit_per_ion[Z][I] > 0.0) {
                            idx_Z[n_bump] = Z; idx_I[n_bump] = I; n_bump++;
                        }
                for (int i = 0; i < n_bump; i++)
                    for (int j = i + 1; j < n_bump; j++)
                        if (bump_emit_per_ion[idx_Z[j]][idx_I[j]] >
                            bump_emit_per_ion[idx_Z[i]][idx_I[i]]) {
                            int tz = idx_Z[i]; idx_Z[i] = idx_Z[j]; idx_Z[j] = tz;
                            int ti = idx_I[i]; idx_I[i] = idx_I[j]; idx_I[j] = ti;
                        }
                int top = n_bump < 10 ? n_bump : 10;
                static const char *rom[5] = {"I  ", "II ", "III", "IV ", "V  "};
                printf("    [3000,3100]Å BUMP BB-emit by ion (top %d, total p=%.4f):\n",
                       top, bump_total);
                printf("       ion    n_lev  | sum_p_emit |  %%total\n");
                for (int k = 0; k < top; k++) {
                    int Z = idx_Z[k], I = idx_I[k];
                    printf("      Z=%2d %s   %5d  | %10.4f | %5.1f%%\n",
                           Z, rom[I], bump_lev_per_ion[Z][I],
                           bump_emit_per_ion[Z][I],
                           100.0 * bump_emit_per_ion[Z][I] / bump_total);
                }
            } else {
                printf("    [3000,3100]Å BUMP BB-emit: no levels with bump emit channels\n");
            }
        }
        #undef DIAG_ZMAX
        #undef DIAG_IMAX
    }
}

/* ---- Task #7: Frozen-in recombination freeze-out (Chugai/Potashov single-epoch) ----
 * The outer ejecta of a young SN Ia carries a fossil ionization plateau (CMFGEN
 * DDC15 0.976d: outer <Z>~0.53 at ~2500 K) that NO steady-state ionization can
 * produce: recombination has frozen out because tau_rec/t_exp grows as t^2
 * (Dessart & Hillier 2008) and reaches >300x in the outermost shells. The cure
 * is the missing TIME-DEPENDENT physics, not a steady-state lever.
 *
 * Per shell we (i) build the faithful per-ion radiative recombination coefficient
 * alpha_rr(Z,i,T_e) as the Milne integral of the CMFGEN photoionization cross
 * sections (atom->cmfgen_sigma_bf) against the Planck function B(T_e) -- the same
 * integral the NLTE assembly does against J_nu, but against B (LTE recomb). NOT a
 * hydrogenic approximation. (ii) Find the parameter-free freeze-out epoch
 *   t_0 = sqrt(alpha_bar * n_e(t_exp) * t_exp^3)   [criterion tau_rec(t_0)=t_0].
 * If t_0 >= t_exp the shell never decouples -> leave the steady-state solution
 * (NLTE already matches the inner region). Otherwise (iii) integrate the
 * recombination cascade in ion-fraction space
 *   dy_{e,k}/dt = a_{e,k} n_e(t) y_{e,k+1} - a_{e,k-1} n_e(t) y_{e,k}
 * over the homologous density history n_e(t) ~ t^-3 from t_0 (seeded at the
 * equilibrium ~singly-ionized partition) to t_exp, with a self-consistent n_e.
 *
 * Validated in Python before this port (scripts/frozen_in_milne_prototype.py):
 * reproduces the outer <Z>~0.53 plateau (0.600) with ZERO fitting and resolves
 * the per-element split (O stays II; Si/Fe recombine toward neutral).
 *
 * Gated LUMINA_FROZENIN=1 (default off -> byte-identical). Needs cmfgen sigma_bf.
 * Pairs with LUMINA_NLTE_ION_LOCK so the downstream NLTE solve preserves the
 * frozen per-ion totals (otherwise NLTE re-solves ion balance and undoes it). */
#define FROZENIN_MAXSTAGE 4

static int frozenin_active(void) {
    const char *e = getenv("LUMINA_FROZENIN");
    return (e && e[0] == '1');
}

/* [ALPHA-SPINGATE] spingate_core_mult() was MOVED UP (see the shared-helper
 * block above recomb_alpha_per_level) so the S3/S1/S2 recombination sites can
 * reuse it; the body is unchanged.  [withParityY Y4] */

/* Milne radiative-recombination coefficient [cm^3/s] producing ion-pop `ip`
 * (charge stage k) by recombination from stage k+1. `ip_next` (charge k+1)
 * supplies the recombining-ion ground statistical weight. T = T_e. */
static double frozenin_alpha_rr(AtomicData *atom, int ip, int ip_next, double T) {
    if (!atom->cmfgen_loaded || T <= 0.0) return 0.0;
    int Z = atom->ion_pop_Z[ip];
    int stage = atom->ion_pop_stage[ip];
    double chi_ion_eV = find_ioniz_energy(atom, Z, stage);
    if (chi_ion_eV <= 0.0) return 0.0;
    double chi_ion_erg = chi_ion_eV * EV_TO_ERG;

    /* Saha/Milne pairing needs the upper ion's PARTITION FUNCTION U_II(T),
     * not its ground-LEVEL g: Si II/C II carry a ^2P fine-structure partner
     * within ~0.04 eV (fully populated), so g_ground=2 vs U_II~5.7 biased
     * their recombination x2.8 high (=> x2.8 under-ionization at the J=B
     * fixed point; S/Mg/Al/O unaffected, U_II~g there). */
    double U_ion = 1.0;
    if (ip_next >= 0) {
        int n0 = atom->level_offset[ip_next];
        int n1 = atom->level_offset[ip_next + 1];
        double kT = K_BOLTZMANN * T;
        double u = 0.0;
        for (int l = n0; l < n1; l++) {
            double x = atom->level_energy_eV[l] * EV_TO_ERG / kT;
            if (x < 50.0) u += (double)atom->level_g[l] * exp(-x);
        }
        if (u >= 1.0) U_ion = u;
        else if (n1 > n0) {
            int gg = atom->level_g[n0];
            U_ion = (gg >= 1) ? (double)gg : 1.0;
        } else {
            /* [RATES-FIX F2] upper ion carries NO levels (n1==n0): the old
             * fallback read level_g[n0], which is the ground g of the NEXT
             * element (and is out of bounds for the last ion pop) -- a stray
             * g that scaled alpha by 1/g. U_ion=1.0 is the recipe the
             * reference partition function (ioniz_selftest_U) already uses.
             * Counted unconditionally; reported at exit when the gate is on. */
#ifdef _OPENMP
            #pragma omp atomic
#endif
            g_rates_fix_n_emptyU++;
            if (rates_fix_enabled()) U_ion = 1.0;
            else {                      /* pre-fix path: the stray read */
                int gg = atom->level_g[n0];
                U_ion = (gg >= 1) ? (double)gg : 1.0;
            }
        }
    }
    double lam3 = pow(H_PLANCK * H_PLANCK /
                      (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T), 1.5);
    int nfreq = atom->cmfgen_n_freq_bins;
    double log_numin = log(atom->cmfgen_nu_min);
    double d_log_nu = (log(atom->cmfgen_nu_max) - log_numin) / nfreq;

    /* [ALPHA-SPINGATE] Restrict the Milne recombination sum to SPIN-ALLOWED
     * daughter terms. Recombination X+1(ground core, S_core) + e(1/2) can only
     * form daughter terms of multiplicity M in {M_core-1, M_core+1}; CMFGEN's
     * sigma_bf carries large cross-sections for spin-forbidden terms that inflate
     * alpha ~4.5-6.7x vs NORAD (see scripts/fe3_alpha_inflation_audit.py). When
     * gated on, levels whose multiplicity is KNOWN and outside the allowed set
     * are skipped. UNKNOWN multiplicity (0) is kept (conservative); an unknown
     * M_core leaves that ion ungated. Gate off => M_core stays 0 and the loop is
     * byte-identical to the original. */
    static int alpha_spingate = -1;
    if (alpha_spingate < 0) {
        const char *e = getenv("LUMINA_ALPHA_SPINGATE");
        alpha_spingate = (e && atoi(e)) ? 1 : 0;
        if (alpha_spingate)
            printf("[ALPHA-SPINGATE] Milne recombination restricted to "
                   "spin-allowed daughter terms (M_core +/- 1)\n");
    }
    int M_core = 0;                     /* 0 => this ion is not gated */
    const char *mcore_src = "none";
    /* [withParityY Y4] the resolution below was PROMOTED to
     * spingate_resolve_core_mult(); the logic is line-for-line the same
     * (data from level_mult of the recombining ion's ground level, else the
     * NIST table), so this call is behaviour-preserving. */
    if (alpha_spingate)
        M_core = spingate_resolve_core_mult(atom, ip_next, Z, stage + 1,
                                            &mcore_src);
    /* one-time detailed diagnostic for Fe III (Z=26 stage=2) */
    static int spingate_diag_done = 0;
    int want_diag = (alpha_spingate && !spingate_diag_done &&
                     Z == 26 && stage == 2 && M_core > 0);
    double a_full_diag = 0.0; int n_lev_diag = 0, n_skip_diag = 0;

    const int rfix_alpha = rates_fix_enabled();   /* [RATES-FIX F3] */
    int g0 = atom->level_offset[ip];
    int g1 = atom->level_offset[ip + 1];
    double a_tot = 0.0;
    for (int gl = g0; gl < g1; gl++) {
        if (!atom->cmfgen_has_sigma[gl]) continue;
        /* spin-selection skip: multiplicity known and not M_core +/- 1
         * [withParityY Y4] promoted to spingate_level_forbidden(); identical
         * predicate (the M_core>0 / level_mult!=NULL guards live inside it). */
        int spin_skip = alpha_spingate
                        ? spingate_level_forbidden(atom, gl, M_core) : 0;
        if (spin_skip && !want_diag) continue;   /* fast path: skip the integral */
        double E_l_erg = atom->level_energy_eV[gl] * EV_TO_ERG;
        double chi_l = chi_ion_erg - E_l_erg;       /* binding energy of level l */
        if (chi_l <= 0.0) continue;
        double nu_th = chi_l / H_PLANCK;
        const double *sigma_row = &atom->cmfgen_sigma_bf[(size_t)gl * (size_t)nfreq];
        double Rbf = 0.0;
        for (int bb = 0; bb < nfreq; bb++) {
            double log_nu_lo = log_numin + bb * d_log_nu;
            double nu_c = exp(log_nu_lo + 0.5 * d_log_nu);
            if (nu_c < nu_th) continue;
            double sig = sigma_row[bb];
            if (sig <= 0.0) continue;
            double x = H_PLANCK * nu_c / (K_BOLTZMANN * T);
            double B;
            if (rfix_alpha) {
                /* [RATES-FIX F3] FUSE the exp(+chi_l/kT) Milne factor into the
                 * integrand: exp(chi_l/kT)/expm1(x) == exp(-(h nu - chi_l)/kT) /
                 * (-expm1(-x)), which is O(1) above threshold. The old form
                 * skipped x>700 (=> Rbf==0) and then multiplied by
                 * exp(chi_l/kT)==inf: 0 x inf = NaN below T ~ chi_l/(700 k),
                 * and even where finite the skip discarded the dominant
                 * near-threshold low-T contribution. */
                double y = (H_PLANCK * nu_c - chi_l) / (K_BOLTZMANN * T);
                B = (2.0 * H_PLANCK * nu_c * nu_c * nu_c /
                     (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) *
                    (exp(-y) / (-expm1(-x)));
            } else {
                if (x > 700.0) continue;
                B = (2.0 * H_PLANCK * nu_c * nu_c * nu_c /
                     (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT)) / expm1(x);
            }
            double dnu = exp(log_nu_lo + d_log_nu) - exp(log_nu_lo);
            Rbf += 4.0 * M_PI_VAL * B * sig / (H_PLANCK * nu_c) * dnu;
        }
        double a_l = rfix_alpha
                 ? Rbf * lam3 * (double)atom->level_g[gl] / (2.0 * U_ion)
                 : Rbf * lam3 * (double)atom->level_g[gl] / (2.0 * U_ion)
                       * exp(chi_l / (K_BOLTZMANN * T));
        if (want_diag) { a_full_diag += a_l; n_lev_diag++; if (spin_skip) n_skip_diag++; }
        if (!spin_skip) a_tot += a_l;
    }
    if (want_diag) {
        double ratio = (a_full_diag > 0.0) ? a_tot / a_full_diag : 0.0;
        /* [withParityY Y5] TRUTH FIX (unconditional, banner text only).  The
         * ratio is a ONE-SHOT diagnostic: it is evaluated at whatever T the
         * FIRST call into this routine happened to carry, and that caller is
         * NOT guaranteed to be a shell.  In production the T_e bisection's cold
         * bracket probe (Tlo = 3500 K, the simul_r1 bracket) gets here first, so
         * the historical "= 0.09" was the 3500 K ratio and not the 0.108-0.113
         * that real shell temperatures give.  The printed line stated no
         * temperature at all, so the two were indistinguishable.  T is now
         * printed and the line is self-verifying; nothing else changes. */
        printf("[ALPHA-SPINGATE] Fe III: alpha_gated/alpha_full = %.2f "
               "at T=%.0fK (skipped %d of %d levels, M_core=%d [%s])\n",
               ratio, T, n_skip_diag, n_lev_diag, M_core, mcore_src);
        spingate_diag_done = 1;
    }
    /* + dielectronic recombination (LUMINA_FROZENIN_DR=1): the U_II-corrected
     * Milne RR exposed that the old x2.8-high alpha was accidentally standing
     * in for the missing DR channel in the frozen-in/tdep cascade (outer dex
     * 0.053->0.138 on the U_II fix alone). ADAS DR_TABLE reused from the
     * NLTE-matrix path; recombining ion = stage+1. */
    static int fz_dr = -1;
    if (fz_dr < 0) {
        const char *e = getenv("LUMINA_FROZENIN_DR");
        fz_dr = (e && atoi(e)) ? 1 : 0;
    }
    if (fz_dr) {
        const DRCoefficient *coef = dr_lookup(Z, stage + 1);
        if (coef) a_tot += dr_alpha_eval(coef, T);
    }
    return a_tot;
}

/* [RATES-FIX F3] direct probe of the PRODUCTION Milne integral for the
 * ionization self-test harness (low-T NaN test). Same (ip, ip+1) pairing the
 * simul ladder uses. Returns the raw value -- NaN/Inf are NOT sanitized, that
 * is the point of the test. -1.0 = (Z,stage) absent from the ion-pop table. */
double lumina_rates_alpha_probe(AtomicData *atom, int Z, int stage, double T) {
    int ip = -1;
    for (int i = 0; i < atom->n_ion_pops; i++)
        if (atom->ion_pop_Z[i] == Z && atom->ion_pop_stage[i] == stage) {
            ip = i; break;
        }
    if (ip < 0) return -1.0;
    int ip_next = (ip + 1 < atom->n_ion_pops) ? ip + 1 : -1;
    return frozenin_alpha_rr(atom, ip, ip_next, T);
}

/* dy/dt for the homologous recombination cascade (fraction space). State y is
 * [nelem * FROZENIN_MAXSTAGE], indexed [e*MS + k] with k = absolute charge.
 * n_e(t) = (t_exp/t)^3 * sum_e n_elem0[e] * <Z>_e(t) (self-consistent). */
static void frozenin_deriv(int nelem, const double *alpha, const double *n_elem0,
                           double t_exp, double t, const double *y, double *dydt) {
    const int MS = FROZENIN_MAXSTAGE;
    double scale = (t_exp / t) * (t_exp / t) * (t_exp / t);
    double ne = 0.0;
    for (int e = 0; e < nelem; e++) {
        double zbar = 0.0;
        for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
        ne += n_elem0[e] * zbar;
    }
    ne *= scale;
    for (int e = 0; e < nelem; e++) {
        for (int k = 0; k < MS; k++) {
            double inflow  = (k + 1 < MS) ? alpha[e * MS + k] * ne * y[e * MS + k + 1] : 0.0;
            double outflow = (k - 1 >= 0) ? alpha[e * MS + k - 1] * ne * y[e * MS + k] : 0.0;
            dydt[e * MS + k] = inflow - outflow;
        }
    }
}

/* Integrate the cascade from t0 to t_exp with RK4 on a log-spaced time grid
 * (density ~ t^-3 -> log nodes are natural; coupling is order-unity at t0 by the
 * freeze-out criterion, so the system is non-stiff over [t0, t_exp]). */
static void frozenin_integrate(int nelem, const double *alpha, const double *n_elem0,
                               double t0, double t_exp, double *y) {
    const int MS = FROZENIN_MAXSTAGE;
    int n = nelem * MS;
    const int NT = 400;
    double k1[FROZENIN_MAXSTAGE * 64], k2[FROZENIN_MAXSTAGE * 64];
    double k3[FROZENIN_MAXSTAGE * 64], k4[FROZENIN_MAXSTAGE * 64];
    double ytmp[FROZENIN_MAXSTAGE * 64];
    double ratio = pow(t_exp / t0, 1.0 / NT);
    double t = t0;
    for (int step = 0; step < NT; step++) {
        double tn = (step == NT - 1) ? t_exp : t * ratio;
        double H = tn - t;
        /* The self-consistent n_e ~ na*(t_exp/t)^3 is enormous near t0 (t0<<t_exp),
         * so the recomb rate alpha*n_e can give alpha*n_e*H >> 1 in the first
         * macro-steps. Fixed explicit RK4 blows up there. Substep so the local
         * stiffness (max recomb rate * substep) stays small -> stable. */
        double ne_now = 0.0;
        for (int e = 0; e < nelem; e++) {
            double zbar = 0.0;
            for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
            ne_now += n_elem0[e] * zbar;
        }
        ne_now *= (t_exp / t) * (t_exp / t) * (t_exp / t);
        double amax = 0.0;
        for (int i = 0; i < n; i++) if (alpha[i] > amax) amax = alpha[i];
        double rate = amax * ne_now;
        /* AUDIT FIX (2026-06-12): the old int cast (int)(rate*H/0.05)+1
         * OVERFLOWED for deep-freeze states (rate*H ~ 1e10+ -> UB -> nsub=1
         * -> explicit RK4 instant blow-up -> NaN through the commit floors).
         * Stiff macro-steps now take the BACKWARD-EULER chain (the cascade is
         * linear in y at lagged n_e and strictly lower-triangular: solve
         * top stage down, unconditionally stable). */
        double nsub_d = rate * H / 0.05 + 1.0;
        int stiff = (!isfinite(nsub_d) || nsub_d > 100000.0);
        int nsub = stiff ? 256 : (int)nsub_d;
        if (nsub < 1) nsub = 1;
        double h = H / nsub;
        for (int sub = 0; sub < nsub; sub++) {
            double ts = t + sub * h;
            if (stiff) {
                /* lagged n_e at substep start */
                double scale = (t_exp / ts) * (t_exp / ts) * (t_exp / ts);
                double ne_s = 0.0;
                for (int e = 0; e < nelem; e++) {
                    double zbar = 0.0;
                    for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
                    ne_s += n_elem0[e] * zbar;
                }
                ne_s *= scale;
                for (int e = 0; e < nelem; e++) {
                    for (int k = MS - 1; k >= 0; k--) {
                        double in_  = (k + 1 < MS)
                                    ? alpha[e * MS + k] * ne_s * y[e * MS + k + 1]
                                    : 0.0;   /* y[k+1] already implicit-updated */
                        double dout = (k - 1 >= 0)
                                    ? alpha[e * MS + k - 1] * ne_s : 0.0;
                        y[e * MS + k] = (y[e * MS + k] + h * in_) /
                                        (1.0 + h * dout);
                    }
                }
            } else {
                frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts, y, k1);
                for (int i = 0; i < n; i++) ytmp[i] = y[i] + 0.5 * h * k1[i];
                frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts + 0.5 * h, ytmp, k2);
                for (int i = 0; i < n; i++) ytmp[i] = y[i] + 0.5 * h * k2[i];
                frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts + 0.5 * h, ytmp, k3);
                for (int i = 0; i < n; i++) ytmp[i] = y[i] + h * k3[i];
                frozenin_deriv(nelem, alpha, n_elem0, t_exp, ts + h, ytmp, k4);
                for (int i = 0; i < n; i++)
                    y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
        }
        t = tn;
    }
    /* clip + renormalize per element (number conservation); NaN-catching
     * (audit fix: NaN<0 is false so the old clip passed NaN through) */
    for (int e = 0; e < nelem; e++) {
        double sum = 0.0;
        for (int k = 0; k < MS; k++) {
            if (!(y[e * MS + k] > 0.0)) y[e * MS + k] = 0.0;
            sum += y[e * MS + k];
        }
        if (sum > 0.0)
            for (int k = 0; k < MS; k++) y[e * MS + k] /= sum;
    }
}

/* Replace the steady-state ion partition with the frozen-in freeze-out cascade
 * for shells whose t_0 < t_exp. Overwrites atom->ion_number_density and
 * plasma->n_electron for those shells only; inner/unfrozen shells untouched. */
/* shells frozen by apply_frozenin_freezeout (set here, read by the coupled Newton
 * so the time-dependent block and the frozen-in cascade partition the grid
 * cleanly with no gap and no double-write). */
static unsigned char *frozenin_is_frozen = NULL;
static int frozenin_is_frozen_n = 0;

/* Non-thermal (deposition) ionization rate density [ionizations/s/cm^3] per shell,
 * registered from compute_gamma_deposition so the freeze-out decision can keep a
 * shell OUT of the frozen cascade where the deposition actively re-ionizes it
 * within t_exp (ARTIS: Gamma_nt sourced by decay keeps the thin outer ionized even
 * when the radiation field is dilute — reference_artis_nonthermal_outer). */
static const double *g_nt_ioniz_rate = NULL;
static int g_nt_ioniz_n = 0;
void frozenin_set_nt_rate(const double *rate, int n_shells) {
    g_nt_ioniz_rate = rate; g_nt_ioniz_n = n_shells;
}

static void apply_frozenin_freezeout(AtomicData *atom, PlasmaState *plasma,
                                     int n_shells, double time_explosion) {
    if (!frozenin_active()) return;
    if (!atom->cmfgen_loaded) {
        printf("  [FROZENIN][WARN] cmfgen sigma_bf not loaded; skipping (Milne alpha needs it)\n");
        return;
    }
    {
        /* The frozen per-ion totals survive the downstream NLTE solve only if a
         * per-ion lock pins each ion's level sum to ion_number_density. Either
         * gate does that (plasma.c:4438). PER_ION_RESCALE is preferred: it locks
         * the totals WITHOUT the transport-only plasma freeze that ION_LOCK
         * triggers (cuda.cu:3188), which would skip this very function. */
        const char *lk = getenv("LUMINA_NLTE_ION_LOCK");
        const char *rs = getenv("LUMINA_NLTE_PER_ION_RESCALE");
        int locked = (lk && lk[0] == '1') || (rs && rs[0] == '1');
        if (!locked)
            printf("  [FROZENIN][WARN] neither LUMINA_NLTE_PER_ION_RESCALE nor "
                   "LUMINA_NLTE_ION_LOCK set; downstream NLTE may re-solve ion "
                   "balance and undo the frozen partition\n");
    }

    const int MS = FROZENIN_MAXSTAGE;
    double t_exp = time_explosion;
    int nelem = atom->n_elements;
    if (nelem > 64) {
        printf("  [FROZENIN][WARN] n_elements=%d exceeds scratch bound 64; skipping\n", nelem);
        return;
    }

    double *n_elem0  = (double *)malloc((size_t)nelem * sizeof(double));
    double *f_frac   = (double *)malloc((size_t)nelem * sizeof(double));
    double *a_rep    = (double *)malloc((size_t)nelem * sizeof(double));
    double *alpha    = (double *)malloc((size_t)nelem * MS * sizeof(double));
    double *y        = (double *)malloc((size_t)nelem * MS * sizeof(double));
    int    *topstage = (int *)malloc((size_t)nelem * sizeof(int));

    /* (re)allocate the per-shell frozen flag for the coupled-Newton partition */
    if (frozenin_is_frozen_n != n_shells) {
        free(frozenin_is_frozen);
        frozenin_is_frozen = (unsigned char *)calloc((size_t)n_shells, 1);
        frozenin_is_frozen_n = frozenin_is_frozen ? n_shells : 0;
    }
    if (frozenin_is_frozen) memset(frozenin_is_frozen, 0, (size_t)n_shells);

    int n_frozen = 0;
    for (int s = 0; s < n_shells; s++) {
        double T_e = plasma->T_e[s];
        double rho = plasma->rho[s];
        if (!(T_e > 0.0) || !(rho > 0.0)) continue;

        /* element number densities at t_exp + per-(element,stage) Milne alpha */
        double n_atom_tot = 0.0;
        for (int e = 0; e < nelem; e++) {
            double abund = atom->abundances[e * n_shells + s];
            double mass_amu = atom->element_mass_amu[e];
            n_elem0[e] = (abund * rho) / (mass_amu * AMU);
            n_atom_tot += n_elem0[e];

            int ip0 = atom->elem_ion_offset[e];
            int ip1 = atom->elem_ion_offset[e + 1];
            int tops = 0;
            for (int k = 0; k < MS; k++) alpha[e * MS + k] = 0.0;
            for (int ip = ip0; ip < ip1; ip++) {
                int stage = atom->ion_pop_stage[ip];
                if (stage >= 0 && stage < MS && stage > tops) tops = stage;
                if (stage < 0 || stage >= MS - 1) continue; /* need stage+1 to recomb in */
                int ip_next = (ip + 1 < ip1 &&
                               atom->ion_pop_stage[ip + 1] == stage + 1) ? ip + 1 : -1;
                if (ip_next < 0) continue;
                alpha[e * MS + stage] = frozenin_alpha_rr(atom, ip, ip_next, T_e);
            }
            topstage[e] = (tops < MS) ? tops : MS - 1;
            /* representative alpha for t_0 = recomb producing the lowest stage */
            a_rep[e] = (alpha[e * MS + 0] > 0.0) ? alpha[e * MS + 0]
                       : 2.6e-13 * pow(T_e / 1e4, -0.8);
        }
        if (n_atom_tot <= 0.0) continue;
        for (int e = 0; e < nelem; e++) f_frac[e] = n_elem0[e] / n_atom_tot;

        /* self-consistent freeze: t_0(n_e) and cascade n_e converge together */
        double ne_sc = plasma->n_electron[s];
        if (!(ne_sc > 0.0)) ne_sc = 1.0;
        int froze = 0;
        double new_ne = ne_sc;
        for (int it = 0; it < 16; it++) {
            double a_bar = 0.0;
            for (int e = 0; e < nelem; e++) a_bar += f_frac[e] * a_rep[e];
            double t0 = sqrt(a_bar * ne_sc * t_exp * t_exp * t_exp);
            /* Non-thermal-ionization freeze guard (LUMINA_FROZENIN_NT): a shell that
             * decoupled by recombination (t0<t_exp) is NOT actually frozen if the
             * deposition re-ionizes it faster than t_exp. Compare the per-atom
             * non-thermal ionization timescale t_nt=1/Gamma_nt with t_exp; if
             * Gamma_nt*t_exp >= thr the gas is actively ionized → fall to the
             * steady-state NLTE balance (which carries Gamma_nt) instead of seeding
             * the cascade at ~singly ionized. This is the ARTIS mechanism that keeps
             * the thin outer ionized (→ fewer line coolants → hot T_e). */
            static int fz_nt = -1; static double fz_nt_thr = 1.0;
            if (fz_nt < 0) { const char *e = getenv("LUMINA_FROZENIN_NT");
                fz_nt = (e && atoi(e)) ? 1 : 0;
                const char *t = getenv("LUMINA_FROZENIN_NT_THR");
                if (t) fz_nt_thr = atof(t); }
            if (fz_nt && g_nt_ioniz_rate && s < g_nt_ioniz_n && n_atom_tot > 0.0) {
                double gamma_nt = g_nt_ioniz_rate[s] / n_atom_tot;  /* [s^-1]/atom */
                if (gamma_nt * t_exp >= fz_nt_thr) { froze = 0; break; }
            }
            if (!(t0 < t_exp)) { froze = 0; break; }   /* never decouples -> steady-state */
            froze = 1;

            for (int e = 0; e < nelem; e++) {
                for (int k = 0; k < MS; k++) y[e * MS + k] = 0.0;
                int ks = (topstage[e] < 1) ? topstage[e] : 1;  /* seed ~singly ionized */
                y[e * MS + ks] = 1.0;
            }
            frozenin_integrate(nelem, alpha, n_elem0, t0, t_exp, y);

            new_ne = 0.0;
            for (int e = 0; e < nelem; e++) {
                double zbar = 0.0;
                for (int k = 0; k < MS; k++) zbar += k * y[e * MS + k];
                new_ne += n_elem0[e] * zbar;
            }
            if (!(new_ne > 0.0)) new_ne = 1.0;
            double drel = fabs(new_ne - ne_sc) / (ne_sc + 1.0);
            ne_sc = 0.5 * ne_sc + 0.5 * new_ne;
            if (drel < 1e-3) break;
        }
        if (!froze) continue;

        /* write frozen ion partition + electron density for this shell */
        for (int e = 0; e < nelem; e++) {
            int ip0 = atom->elem_ion_offset[e];
            int ip1 = atom->elem_ion_offset[e + 1];
            for (int ip = ip0; ip < ip1; ip++) {
                int stage = atom->ion_pop_stage[ip];
                double n_ion = (stage >= 0 && stage < MS)
                               ? y[e * MS + stage] * n_elem0[e] : 0.0;
                if (lumina_zinert_element_inactive(atom, e, n_shells)) n_ion = 0.0;
                else if (!(n_ion > 1e-300)) n_ion = 1e-300;  /* NaN-catching */
                atom->ion_number_density[ip * n_shells + s] = n_ion;
            }
        }
        plasma->n_electron[s] = new_ne;
        if (frozenin_is_frozen) frozenin_is_frozen[s] = 1;
        n_frozen++;
    }

    printf("  [FROZENIN] freeze-out applied: %d/%d shells frozen (t_0<t_exp), "
           "rest steady-state\n", n_frozen, n_shells);

    free(n_elem0); free(f_frac); free(a_rep);
    free(alpha); free(y); free(topstage);
}

/* Task #072 Step 4e: Master plasma state update */
void tau_sobolev_require_refresh(OpacityState *opacity, const char *reason) {
    if (!opacity) return;
    if (opacity->tau_required_generation == UINT64_MAX) {
        fprintf(stderr, "[K-FRESH][FATAL] tau generation overflow before %s\n",
                reason ? reason : "refresh");
        abort();
    }
    opacity->tau_required_generation++;
}

void tau_sobolev_mark_computed(OpacityState *opacity, const char *producer) {
    if (!opacity || !opacity->tau_sobolev ||
        opacity->tau_required_generation == 0) {
        fprintf(stderr, "[K-FRESH][FATAL] invalid tau producer state at %s\n",
                producer ? producer : "unknown producer");
        abort();
    }
    opacity->tau_computed_generation = opacity->tau_required_generation;
}

int tau_sobolev_assert_fresh(OpacityState *opacity, const char *consumer) {
    if (!opacity || !opacity->tau_sobolev ||
        opacity->tau_computed_generation == 0 ||
        opacity->tau_computed_generation < opacity->tau_required_generation) {
        fprintf(stderr,
                "[K-FRESH][FATAL] stale tau blocked before %s "
                "(computed_generation=%llu required_generation=%llu)\n",
                consumer ? consumer : "unknown consumer",
                (unsigned long long)(opacity ? opacity->tau_computed_generation : 0),
                (unsigned long long)(opacity ? opacity->tau_required_generation : 0));
        return -1;
    }
    if (opacity->tau_first_consumer_generation == 0) {
        opacity->tau_first_consumer_generation = opacity->tau_computed_generation;
        printf("[K-FRESH] first consumer=%s computed_generation=%llu "
               "required_generation=%llu owner=solver\n",
               consumer ? consumer : "unknown",
               (unsigned long long)opacity->tau_computed_generation,
               (unsigned long long)opacity->tau_required_generation);
    }
    return 0;
}

/* A2-07/A2-10 부트스트랩: 덱의 seed T_e 를 **1세대로 발행**한다.
 *
 * 왜 필요한가.  반복 안 순서는 수송 → T_e → 플라즈마/tau 다.  따라서 반복 0 의 수송에
 * 솔버가 만든 tau 를 주려면 그 앞에 플라즈마 풀이가 있어야 하고, 그러려면 **발행된 T_e**
 * 가 있어야 한다.  그런데 복사장이 아직 없으므로 radeq 는 돌 수 없다:
 *
 *     tau ← population ← 발행된 T_e ← 복사장 ← 수송 ← tau
 *
 * 이 고리를 끊는 정직한 지점은 "첫 상태는 seed 온도의 LTE" 하나뿐이며 CMFGEN·ARTIS 가
 * 실제로 하는 방식이다.  A2-07 은 seed 를 `generation-zero material seed` 로 두어
 * **소비 금지**했고, 이 함수는 그것을 명시적 1세대 발행으로 승격한다 — 계약 개정이다
 * (user 판정 2026-08-07, 안 B; docs/RUNG_SEED_TE_PUBLICATION.md).
 *
 * radeq 발행이 **아니다**.  구분이 대장에 남도록:
 *   · A2-10 계수기의 seed_generation_attempts 를 올린다(그 자리는 비어 있었다)
 *   · [A2-10][SEED] 배너를 찍는다 — radeq 발행과 로그에서 절대 섞이지 않게
 *
 * 불변식은 radeq 와 동일하게 지킨다: gen = T_e_generation + 1, committed = gen,
 * manifest = sha256(seed T_e).  그래야 반복 0 의 A2-10 스탬프 대조가 성립한다.
 *
 * seed 가 비유한·비양수면 **고치지 않고 거부**한다(fail-closed).  잘못된 seed 를
 * 눌러 담는 순간 그것이 클램프다.
 */
int lumina_publish_seed_te(PlasmaState *plasma, const char *reason) {
    if (!plasma || !plasma->T_e || plasma->n_shells <= 0) {
        fprintf(stderr, "[A2-10][SEED][FATAL] invalid plasma for seed publication\n");
        return -1;
    }
    if (plasma->T_e_generation != 0) {
        /* 부트스트랩은 1회다.  두 번째 호출은 radeq 발행을 덮어쓸 수 있으므로 버그다. */
        fprintf(stderr,
                "[A2-10][SEED][FATAL] T_e already published (generation=%llu) — "
                "seed bootstrap is once per run\n",
                (unsigned long long)plasma->T_e_generation);
        return -1;
    }
    for (int s = 0; s < plasma->n_shells; s++) {
        if (!isfinite(plasma->T_e[s]) || plasma->T_e[s] <= 0.0) {
            fprintf(stderr,
                    "[A2-10][SEED][FATAL] deck seed T_e invalid at shell %d: %.17g "
                    "(fail-closed: seed 는 고치지 않는다)\n", s, plasma->T_e[s]);
            return -1;
        }
    }
    char te_hash[65];
    if (population_te_manifest_sha256(plasma->T_e, (size_t)plasma->n_shells,
                                      te_hash) != POP_OK) {
        fprintf(stderr, "[A2-10][SEED][FATAL] te manifest hash failed\n");
        return -1;
    }
    uint64_t gen = plasma->T_e_generation + 1;   /* == 1 */
    plasma->te_publication.required_te_generation = gen;
    plasma->te_publication.committed_te_generation = gen;
    memcpy(plasma->te_publication.te_manifest_sha256, te_hash, sizeof(te_hash));
    plasma->T_e_generation = gen;
    a210_counters()->seed_generation_attempts++;
    printf("[A2-10][SEED] bootstrap T_e published generation=%llu n_shells=%d "
           "manifest=%.12s reason=%s\n",
           (unsigned long long)gen, plasma->n_shells, te_hash,
           reason ? reason : "(none)");
    return 0;
}

int lumina_prepare_solver_owned_tau(AtomicData *atom, PlasmaState *plasma,
                                    OpacityState *opacity,
                                    double time_explosion,
                                    const char *first_consumer) {
    if (!atom || !plasma || !opacity || !opacity->tau_sobolev) {
        fprintf(stderr, "[K-FRESH][FATAL] cannot prepare solver-owned tau\n");
        return -1;
    }
    if (compute_plasma_state(atom, plasma, opacity, time_explosion) != 0) {
        /* ★침묵 금지.  T3 를 여섯 번 죽인 것이 이 조용한 return 이다(2026-08-07).
         * compute_plasma_state 는 실패 이유를 population_last_error 에 적지만
         * 그것을 읽는 곳이 하나도 없었다 — 런은 `exit 1`, 메시지 없음만 남기고
         * 사라졌고, 나는 그 위에 가설을 다섯 번 쌓았다.
         * 진단은 이유를 아는 곳에서 나와야 한다. */
        fprintf(stderr,
                "[K-FRESH][FATAL] compute_plasma_state failed: %s "
                "(consumer=%s n_shells=%d T_e_gen=%llu err_count=%llu)\n",
                population_status_name(plasma->population_last_error),
                first_consumer ? first_consumer : "(null)",
                plasma->n_shells,
                (unsigned long long)plasma->T_e_generation,
                (unsigned long long)plasma->population_error_count);
        return -1;
    }
    return tau_sobolev_assert_fresh(opacity, first_consumer);
}

int compute_plasma_state(AtomicData *atom, PlasmaState *plasma,
                         OpacityState *opacity, double time_explosion) {
    if (!atom || !plasma || !opacity || !plasma->T_e ||
        !plasma->n_electron || plasma->n_shells <= 0) {
        if (plasma) {
            plasma->population_last_error = POP_INVALID_TE;
            plasma->population_error_count++;
        }
        return -1;
    }
    plasma->population_last_error = POP_OK;
    const char *frozenin = getenv("LUMINA_FROZENIN");
    if (frozenin && atoi(frozenin) != 0) {
        plasma->population_last_error = POP_FORBIDDEN_FALLBACK;
        plasma->population_error_count++;
        fprintf(stderr, "[A2-07] forbidden population fallback configuration: LUMINA_FROZENIN\n");
        return -1;
    }
    int n_shells = plasma->n_shells;
    char te_hash[65];
    if (population_te_manifest_sha256(plasma->T_e, (size_t)n_shells,
                                      te_hash) != POP_OK ||
        plasma->T_e_generation == 0) {
        plasma->population_last_error = POP_INVALID_TE;
        plasma->population_error_count++;
        return -1;
    }

    uint64_t required_generation = atom->population_committed_generation + 1;
    PopulationTransaction pop_tx;
    double *published_ion = atom->ion_number_density;
    double *published_ne = plasma->n_electron;
    double *published_partition = atom->partition_functions;
    PopulationDerivedStamp prior_stamp = atom->partition_stamp;
    if (population_transaction_begin(
            &pop_tx, published_ion, (size_t)atom->n_ion_pops * n_shells,
            NULL, 0, published_ne, (size_t)n_shells,
            published_partition, (size_t)atom->n_ion_pops * n_shells,
            required_generation, &atom->population_committed_generation) != 0) {
        plasma->population_last_error = POP_SOLVE_FAILED;
        plasma->population_error_count++;
        return -1;
    }
    atom->ion_number_density = pop_tx.work_ion;
    plasma->n_electron = pop_tx.work_ne;
    atom->partition_functions = pop_tx.work_partition;
#define A2_07_PLASMA_ABORT(status_) do {                                  \
        atom->ion_number_density = published_ion;                          \
        plasma->n_electron = published_ne;                                 \
        atom->partition_functions = published_partition;                   \
        atom->partition_stamp = prior_stamp;                               \
        population_transaction_abort(&pop_tx, (status_));                  \
        plasma->population_last_error = (status_);                         \
        plasma->population_error_count++;                                  \
        return -1;                                                         \
    } while (0)

    tau_sobolev_require_refresh(opacity, "compute_plasma_state");

    printf("  [Plasma] Computing partition functions...\n");
    PopulationStatus partition_status =
        compute_partition_functions(atom, plasma, n_shells);
    if (partition_status != POP_OK)
        A2_07_PLASMA_ABORT(partition_status);

    /* TOY DIAGNOSTIC (LUMINA_FIXED_NE_PROFILE=<file>): impose per-shell n_e and SKIP the
     * iterative electron-density solve, so toy models fix the thermodynamic state (T_e+n_e)
     * as INPUT and isolate the line/NLTE/fluorescence physics. File rows = "shell_id n_e". */
    static int fne_init = 0, fne_on = 0, fne_n = 0; static double *fne = NULL;
    if (!fne_init) { fne_init = 1;
        const char *fp = getenv("LUMINA_FIXED_NE_PROFILE");
        if (fp && *fp) { FILE *f = fopen(fp, "r");
            if (f) { fne = (double*)calloc(n_shells, sizeof(double)); char ln[256];
                while (fgets(ln, sizeof(ln), f)) { if (ln[0]=='#') continue; int s; double v;
                    if (sscanf(ln, "%d %lf", &s, &v)==2 && s>=0 && s<n_shells) { fne[s]=v; fne_n++; } }
                fclose(f); fne_on = (fne_n == n_shells);
                printf("  [fixed-ne] %s: %d/%d shells -> %s\n", fp, fne_n, n_shells,
                       fne_on ? "ACTIVE (n_e solve skipped)" : "INCOMPLETE, ignored"); }
            else printf("  [fixed-ne] could not open %s\n", fp); } }
    if (fne_on) {
        for (int s = 0; s < n_shells; s++) plasma->n_electron[s] = fne[s];
        printf("  [Plasma] electron density IMPOSED (fixed-ne): n_e[0]=%.4e\n", plasma->n_electron[0]);
    } else {
        printf("  [Plasma] Computing electron density (iterative)...\n");
        /* LUMINA_RADEQ_SIMUL owns n_e AND the ion partition: this second
         * Saha path was OVERWRITING the SIMUL commit every iteration and, at
         * the poisoned-T_rad shell s=40, exploding n_e to 9.565e7 (48x charge
         * conservation) via the audit-A5 defect (1e30 ladder overflow breaks
         * sum_norm while the n_ion product keeps multiplying -> unnormalized
         * top stage). The lamp that erased the outer root. */
        if (g_simul_on != 1 &&
            compute_electron_density(atom, plasma, n_shells) != 0)
            A2_07_PLASMA_ABORT(POP_NE_NOT_CONVERGED);
        printf("    n_e[0]=%.4e, n_e[%d]=%.4e\n",
               plasma->n_electron[0], n_shells - 1, plasma->n_electron[n_shells - 1]);
    }

    printf("  [Plasma] Computing ion populations...\n");
    PopulationStatus ion_status =
        compute_ion_populations(atom, plasma, n_shells);
    if (ion_status != POP_OK)
        A2_07_PLASMA_ABORT(ion_status);

    /* ★T0(2026-08-07) 최상단 reservoir 근사의 **유효 영역 검사**.
     *
     * 로더는 전리에너지 n 개에 대해 population n+1 개를 만들므로 원소마다 최상단
     * population 에 속박준위가 없다(실측 15/74, 전부 최상단, 덱 3종 동일).
     * 기준 배선 ARTIS 는 그 이온에 준위 1 개(바닥)를 주어 Z=g_ground 를 쓰는데
     * (SINGLE_LEVEL_TOP_ION), 우리 덱에는 그 g 가 없어 지금은 Z=1 을 대입한다.
     *
     * Z=1 은 g=1 에서만 정확하다.  그러므로 **미지의 g 를 가정하지 않고 상한으로 감싼다**:
     * 참 분율은 Z=1 로 얻은 분율의 최대 G_BOUND 배다(Saha 비가 Z_top 에 선형).
     * 그 상한이 MAX_TRUE 를 넘으면 근사가 유효하지 않다 ⟹ **값을 누르지 않고 거부**한다.
     * 누르면 그것이 클램프다. */
#define A2_07_RESERVOIR_G_BOUND    30.0   /* 이 단계들 바닥 g 의 보수적 상한(가정, 기재) */
#define A2_07_RESERVOIR_MAX_TRUE   1.0e-2 /* 참 분율 상한이 원소의 1% 를 넘으면 거부 */
    {
        double worst = 0.0;
        int worst_ip = -1, worst_s = -1;
        for (int e = 0; e < atom->n_elements; e++) {
            int b = atom->elem_ion_offset[e], t = atom->elem_ion_offset[e + 1];
            for (int s = 0; s < n_shells; s++) {
                double tot = 0.0;
                for (int ip = b; ip < t; ip++)
                    tot += atom->ion_number_density[(size_t)ip * n_shells + s];
                if (!(tot > 0.0)) continue;
                for (int ip = b; ip < t; ip++) {
                    if (atom->level_offset[ip + 1] > atom->level_offset[ip]) continue;
                    double f = atom->ion_number_density[(size_t)ip * n_shells + s] / tot;
                    if (f > worst) { worst = f; worst_ip = ip; worst_s = s; }
                }
            }
        }
        printf("  [A2-07] level-less reservoir: max fraction=%.3e "
               "(x%.0f bound = %.3e, limit %.0e)\n",
               worst, A2_07_RESERVOIR_G_BOUND,
               worst * A2_07_RESERVOIR_G_BOUND, A2_07_RESERVOIR_MAX_TRUE);
        if (worst * A2_07_RESERVOIR_G_BOUND > A2_07_RESERVOIR_MAX_TRUE) {
            fprintf(stderr,
                    "[A2-07][FATAL] level-less top stage is populated: ion_pop %d "
                    "(Z=%d stage=%d) shell %d fraction=%.3e -> bounded true %.3e > %.0e.\n"
                    "  Z=1 대입이 유효하지 않다. 그 이온의 바닥 g 를 도입해 준위 1 개를 주어야 한다\n"
                    "  (ARTIS SINGLE_LEVEL_TOP_ION 과 같은 배선).\n",
                    worst_ip,
                    atom->ion_pop_Z ? atom->ion_pop_Z[worst_ip] : -1,
                    atom->ion_pop_stage ? atom->ion_pop_stage[worst_ip] : -1,
                    worst_s, worst, worst * A2_07_RESERVOIR_G_BOUND,
                    A2_07_RESERVOIR_MAX_TRUE);
            A2_07_PLASMA_ABORT(POP_INVALID_PARTITION);
        }
    }

    /* Task #7: frozen-in recombination freeze-out (gated LUMINA_FROZENIN).
     * Overrides outer-shell ion populations + n_e with the time-dependent
     * cascade; no-op (byte-identical) when off. */
    apply_frozenin_freezeout(atom, plasma, n_shells, time_explosion);

    atom->ion_number_density = published_ion;
    plasma->n_electron = published_ne;
    atom->partition_functions = published_partition;
    PopulationStatus publish_status = population_transaction_commit(&pop_tx);
    if (publish_status != POP_OK) {
        atom->partition_stamp = prior_stamp;
        plasma->population_last_error = publish_status;
        plasma->population_error_count++;
        return -1;
    }

    /* Copy self-consistent n_e back to opacity for transport */
    for (int s = 0; s < n_shells; s++)
        opacity->electron_density[s] = plasma->n_electron[s];

    printf("  [Plasma] Computing tau_sobolev...\n");
    compute_tau_sobolev(atom, plasma, opacity, time_explosion);

    /* Line overlap correction (enabled by LUMINA_OVERLAP_CORR=1) */
    if (getenv("LUMINA_OVERLAP_CORR") && atoi(getenv("LUMINA_OVERLAP_CORR")) > 0) {
        printf("  [Plasma] Applying line overlap corrections...\n");
        apply_overlap_corrections(atom, opacity, plasma);
    }

    if (zinert_audit_enabled())
        (void)lumina_zinert_validate(atom, NULL, opacity, n_shells,
                                     "pre-transport");

    tau_sobolev_mark_computed(opacity, "compute_plasma_state");
    plasma->population_last_error = POP_OK;

    /* Print tau stats for key lines */
    int n_lines = opacity->n_lines;
    double tau_min = 1e99, tau_max = 0.0;
    int n_significant = 0;
    /* [TAU-DIAG] band-resolved counts (shell 0). lambda_A = c / nu in Å. */
    int n_uv = 0, n_uv_sig = 0;          /* 1700 - 3000 Å */
    int n_blue = 0, n_blue_sig = 0;      /* 3000 - 4500 Å */
    int n_opt = 0, n_opt_sig = 0;        /* 4500 - 7000 Å */
    double tau_sum_uv = 0.0, tau_sum_blue = 0.0, tau_sum_opt = 0.0;
    for (int l = 0; l < n_lines; l++) {
        double t = opacity->tau_sobolev[l * n_shells + 0]; /* shell 0 */
        if (t > tau_max) tau_max = t;
        if (t < tau_min && t > 1e-100) tau_min = t;
        if (t > 1.0) n_significant++;
        double nu = atom->line_nu[l];
        double lam_A = (nu > 0.0) ? (C_SPEED_OF_LIGHT / nu * 1e8) : 0.0;
        if (lam_A >= 1700.0 && lam_A < 3000.0)      { n_uv++;   if (t > 1.0) n_uv_sig++;   tau_sum_uv += t; }
        else if (lam_A >= 3000.0 && lam_A < 4500.0) { n_blue++; if (t > 1.0) n_blue_sig++; tau_sum_blue += t; }
        else if (lam_A >= 4500.0 && lam_A < 7000.0) { n_opt++;  if (t > 1.0) n_opt_sig++;  tau_sum_opt += t; }
    }
    printf("    Shell 0: tau_min=%.2e, tau_max=%.2e, lines with tau>1: %d/%d\n",
           tau_min, tau_max, n_significant, n_lines);
    printf("    [TAU-DIAG] tau>1 by band: UV(1700-3000)=%d/%d (sum=%.1f) | "
           "blue(3000-4500)=%d/%d (sum=%.1f) | opt(4500-7000)=%d/%d (sum=%.1f)\n",
           n_uv_sig, n_uv, tau_sum_uv,
           n_blue_sig, n_blue, tau_sum_blue,
           n_opt_sig, n_opt, tau_sum_opt);

    /* [TAU-BY-ION] per-(Z,ion) shell-0 line-opacity decomposition.
       Identifies which iron-peak ions build the "iron curtain" that drives
       reabsorption / T_inner overshoot. Gated: LUMINA_TAU_BY_ION=1.
       LUMINA_TAU_BY_ION_SHELL=k selects shell k (default 0, the inner edge). */
    if (getenv("LUMINA_TAU_BY_ION") && atoi(getenv("LUMINA_TAU_BY_ION")) > 0) {
        const int ZMAX = 31, IMAX = 16;
        int sh = 0;
        if (getenv("LUMINA_TAU_BY_ION_SHELL"))
            sh = atoi(getenv("LUMINA_TAU_BY_ION_SHELL"));
        if (sh < 0) sh = 0;
        if (sh >= n_shells) sh = n_shells - 1;
        /* Wavelength window (rest-frame Å); default broad-band [1700,10000].
           Narrow it (e.g. 4050/4150) to attribute a specific feature. */
        double lam_lo = 1700.0, lam_hi = 10000.0;
        if (getenv("LUMINA_TAU_BY_ION_LAM_LO"))
            lam_lo = atof(getenv("LUMINA_TAU_BY_ION_LAM_LO"));
        if (getenv("LUMINA_TAU_BY_ION_LAM_HI"))
            lam_hi = atof(getenv("LUMINA_TAU_BY_ION_LAM_HI"));
        static double tau_zi[31 * 16];   /* sum of tau over window */
        static int    cnt_zi[31 * 16];   /* count of tau>1 lines */
        memset(tau_zi, 0, sizeof(tau_zi));
        memset(cnt_zi, 0, sizeof(cnt_zi));
        double tau_tot = 0.0;
        for (int l = 0; l < n_lines; l++) {
            double nu = atom->line_nu[l];
            double lam_A = (nu > 0.0) ? (C_SPEED_OF_LIGHT / nu * 1e8) : 0.0;
            if (lam_A < lam_lo || lam_A >= lam_hi) continue;
            double t = opacity->tau_sobolev[(size_t)l * n_shells + sh];
            int Z = atom->line_atomic_number[l];
            int ion = atom->line_ion_number[l];
            if (Z < 0 || Z >= ZMAX || ion < 0 || ion >= IMAX) continue;
            tau_zi[Z * IMAX + ion] += t;
            if (t > 1.0) cnt_zi[Z * IMAX + ion]++;
            tau_tot += t;
        }
        /* print top contributors by tau-sum */
        printf("    [TAU-BY-ION] shell %d (tau_tot[%.0f-%.0fA]=%.3e) top ions:\n",
               sh, lam_lo, lam_hi, tau_tot);
        for (int rank = 0; rank < 15; rank++) {
            int best = -1; double bestv = 0.0;
            for (int k = 0; k < ZMAX * IMAX; k++)
                if (tau_zi[k] > bestv) { bestv = tau_zi[k]; best = k; }
            if (best < 0 || bestv <= 0.0) break;
            int Z = best / IMAX, ion = best % IMAX;
            printf("      Z=%2d ion=%2d  tau_sum=%.3e (%.1f%%)  N(tau>1)=%d\n",
                   Z, ion, bestv, 100.0 * bestv / (tau_tot > 0 ? tau_tot : 1.0),
                   cnt_zi[best]);
            tau_zi[best] = 0.0;  /* consume */
        }
    }

    /* [ION-FRAC] per-shell ionization fractions for selected elements.
       Tests whether a carrier ion (e.g. Co II) is over-populated due to
       under-ionization. Gated: LUMINA_ION_FRAC_DUMP=1.
       LUMINA_ION_FRAC_Z="27,26,28,20" picks elements (default Co,Fe,Ni,Ca). */
    if (getenv("LUMINA_ION_FRAC_DUMP") && atoi(getenv("LUMINA_ION_FRAC_DUMP")) > 0) {
        int z_list[8] = {27, 26, 28, 20, 0, 0, 0, 0};
        int n_z = 4;
        const char *zenv = getenv("LUMINA_ION_FRAC_Z");
        if (zenv) {
            n_z = 0;
            char buf[128]; strncpy(buf, zenv, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
            char *tok = strtok(buf, ",");
            while (tok && n_z < 8) { z_list[n_z++] = atoi(tok); tok = strtok(NULL, ","); }
        }
        for (int zi = 0; zi < n_z; zi++) {
            int Z = z_list[zi];
            printf("    [ION-FRAC] Z=%d  (stage 0=I,1=II,2=III,...) by shell:\n", Z);
            for (int s = 0; s < n_shells; s++) {
                double tot = 0.0, frac[6] = {0,0,0,0,0,0};
                for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                    if (atom->ion_pop_Z[ip] != Z) continue;
                    tot += atom->ion_number_density[(size_t)ip * n_shells + s];
                }
                if (tot <= 0.0) continue;
                for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                    if (atom->ion_pop_Z[ip] != Z) continue;
                    int st = atom->ion_pop_stage[ip];
                    if (st >= 0 && st < 6)
                        frac[st] = atom->ion_number_density[(size_t)ip * n_shells + s] / tot;
                }
                printf("      shell %2d: n_tot=%.3e  I=%.3f II=%.3f III=%.3f IV=%.3f V=%.3f\n",
                       s, tot, frac[0], frac[1], frac[2], frac[3], frac[4]);
            }
        }
    }
#undef A2_07_PLASMA_ABORT
    return 0;
}

/* ============================================================ */
/* Bound-free (photoionization) opacity                        */
/* Kramers cross-section grid: chi_bf[shell][freq_bin]         */
/* ============================================================ */

/* Wave-1 bf repair gates. These accessors are deliberately side-effect-free:
 * an unset/zero LUMINA_FIX_* variable leaves every legacy branch and banner
 * untouched. CUDA bf helpers call the shared accessors so a producer cannot
 * silently use a different neutral/multi-edge policy from the CPU path. */
int lumina_fix_bf_stim_recomb_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LUMINA_FIX_BF_STIM_RECOMB");
        on = (e && atoi(e)) ? 1 : 0;
    }
    return on;
}

int lumina_fix_bf_neutral_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LUMINA_FIX_BF_NEUTRAL");
        on = (e && atoi(e)) ? 1 : 0;
    }
    return on;
}

int lumina_fix_bf_multi_edge_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e_fix = getenv("LUMINA_FIX_BF_MULTI_EDGE");
        const char *e_old = getenv("LUMINA_KPKT_FB_MULTI");
        /* An explicit setting of the canonical gate is authoritative, including
         * explicit OFF. The legacy spelling is consulted only when the canonical
         * variable is absent, so an alias can never override user intent. */
        on = e_fix ? (atoi(e_fix) != 0)
                   : (e_old && atoi(e_old) != 0);
    }
    return on;
}

int lumina_fix_bf_continuum_event_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LUMINA_FIX_BF_CONTINUUM_EVENT");
        on = (e && atoi(e)) ? 1 : 0;
    }
    return on;
}

static int lumina_fix_bf_eta_spingate_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LUMINA_FIX_BF_ETA_SPINGATE");
        on = (e && atoi(e)) ? 1 : 0;
    }
    return on;
}

void bf_opacity_init(BFOpacity *bf, int n_shells) {
    memset(bf, 0, sizeof(*bf));
    bf->enabled = 1;
    bf->n_freq_bins = NLTE_N_FREQ_BINS;
    bf->n_shells = n_shells;
    bf->nu_min = NLTE_NU_MIN;
    bf->nu_max = NLTE_NU_MAX;
    bf->d_log_nu = log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS;
    bf->chi_bf = (double *)calloc((size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    bf->eta_bf = (double *)calloc((size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    bf->activation_level = (int *)malloc((size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(int));
    memset(bf->activation_level, -1, (size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(int));
}

void bf_opacity_free(BFOpacity *bf) {
    free(bf->chi_bf);
    free(bf->eta_bf);
    free(bf->activation_level);
    free(bf->event_level_offset);
    free(bf->event_element);
    free(bf->event_ion);
    free(bf->event_level);
    free(bf->event_target);
    free(bf->event_target_fallback);
    free(bf->event_has_sigma);
    free(bf->event_nu_edge);
    free(bf->event_sigma0);
    free(bf->event_weight);
    free(bf->event_stim_ratio);
    free(bf->event_chi_bf);
    free(bf->event_Te);
    memset(bf, 0, sizeof(*bf));
}

/* Look up macro-atom activation level for BF absorption at given frequency.
 * Returns macro-atom level index (global level idx) or -1 for thermal fallback. */
int bf_get_activation_level(BFOpacity *bf, int shell, double nu) {
    if (!bf->enabled || !bf->activation_level || nu < bf->nu_min || nu >= bf->nu_max)
        return -1;
    int bin = (int)(log(nu / bf->nu_min) / bf->d_log_nu);
    if (bin < 0 || bin >= bf->n_freq_bins) return -1;
    return bf->activation_level[shell * bf->n_freq_bins + bin];
}

/* Evaluate one continuum route at the packet's actual event frequency. CMFGEN
 * cross sections are stored as per-bin averages represented at log-grid bin
 * centres, so interpolate those values in log(nu); analytic Kramers routes are
 * evaluated directly at nu. */
static double bf_event_route_contribution(const BFOpacity *bf, int shell,
                                          int route, double nu) {
    double w = bf->event_weight[(size_t)shell * bf->event_n_routes + route];
    double edge = bf->event_nu_edge[route];
    if (!(w > 0.0) || !(edge > 0.0) || nu < edge) return 0.0;

    int l = bf->event_level[route];
    double sigma = 0.0;
    if (bf->event_sigma_bf && bf->event_has_sigma[route] && l >= 0) {
        const double *row =
            bf->event_sigma_bf + (size_t)l * bf->n_freq_bins;
        double pos = log(nu / bf->nu_min) / bf->d_log_nu - 0.5;
        if (pos <= 0.0) {
            sigma = row[0];
        } else if (pos >= (double)(bf->n_freq_bins - 1)) {
            sigma = row[bf->n_freq_bins - 1];
        } else {
            int lo = (int)floor(pos);
            double frac = pos - (double)lo;
            sigma = row[lo] + frac * (row[lo + 1] - row[lo]);
        }
    } else {
        double x = edge / nu;
        sigma = bf->event_sigma0[route] * x * x * x;
    }
    if (!(sigma > 0.0)) return 0.0;

    /* Packet selection consumes the independently nonnegative gross measure.
     * Stimulated recombination belongs only to the signed net coefficient. */
    return w * sigma;
}

/* CPU mirror of the D-1 GPU route draw. Returns the selected route index and
 * resolves its upper target/edge. Both the endpoint and cumulative sum are
 * constructed at the packet's actual event nu_cmf (ARTIS rpkt.cc:371). */
int bf_sample_continuum_event(BFOpacity *bf, int shell, double nu,
                              RNG *rng, int *target, double *nu_edge) {
    /* Once a bf event reaches this selector ARTIS consumes the continuum-CDF
     * draw even if inconsistent input leaves no represented route. */
    double u_cdf = rng_uniform(rng);
    if (!bf || !bf->event_enabled || !bf->event_chi_bf ||
        !bf->event_weight || bf->event_n_routes <= 0 ||
        shell < 0 || shell >= bf->n_shells ||
        nu < bf->nu_min || nu >= bf->nu_max)
        return -1;

    double total = 0.0;
    for (int r = 0; r < bf->event_n_routes; r++)
        total += bf_event_route_contribution(bf, shell, r, nu);
    if (!(total > 0.0)) return -1;
    double threshold = u_cdf * total;
    double cumulative = 0.0;
    int selected = -1, last_valid = -1;
    for (int r = 0; r < bf->event_n_routes; r++) {
        double contrib = bf_event_route_contribution(bf, shell, r, nu);
        if (!(contrib > 0.0)) continue;
        cumulative += contrib;
        last_valid = r;
        if (cumulative > threshold) { selected = r; break; }
    }
    if (selected < 0) selected = last_valid;
    if (selected >= 0) {
        *target = bf->event_target[selected];
        *nu_edge = bf->event_nu_edge[selected];
    }
    return selected;
}

/* Interpolate chi_bf at arbitrary frequency (linear in log-nu grid) */
double bf_get_chi(BFOpacity *bf, int shell, double nu) {
    if (!bf->enabled || nu < bf->nu_min || nu >= bf->nu_max) return 0.0;
    double log_ratio = log(nu / bf->nu_min);
    int bin = (int)(log_ratio / bf->d_log_nu);
    if (bin < 0) return 0.0;
    if (bin >= bf->n_freq_bins - 1) return bf->chi_bf[shell * bf->n_freq_bins + bf->n_freq_bins - 1];
    /* Linear interpolation between bins */
    double frac = log_ratio / bf->d_log_nu - (double)bin;
    double chi0 = bf->chi_bf[shell * bf->n_freq_bins + bin];
    double chi1 = bf->chi_bf[shell * bf->n_freq_bins + bin + 1];
    return chi0 + frac * (chi1 - chi0);
}

/* D-1's bound-free-only endpoint. Legacy chi_bf also contains free-free;
 * selecting this grid keeps ARTIS' bf and ff absorption channels distinct. */
double bf_get_event_measure(BFOpacity *bf, int shell, double nu) {
    if (!bf->event_enabled || !bf->event_chi_bf ||
        nu < bf->nu_min || nu >= bf->nu_max) return 0.0;
    double x = log(nu / bf->nu_min) / bf->d_log_nu;
    int bin = (int)x;
    if (bin < 0) return 0.0;
    if (bin >= bf->n_freq_bins - 1)
        return bf->event_chi_bf[(size_t)shell * bf->n_freq_bins +
                                bf->n_freq_bins - 1];
    double frac = x - (double)bin;
    double c0 = bf->event_chi_bf[(size_t)shell * bf->n_freq_bins + bin];
    double c1 = bf->event_chi_bf[(size_t)shell * bf->n_freq_bins + bin + 1];
    return c0 + frac * (c1 - c0);
}

/* bf emissivity lookup (same grid/interpolation as bf_get_chi). */
double bf_get_eta(BFOpacity *bf, int shell, double nu) {
    if (!bf->enabled || !bf->eta_bf || nu < bf->nu_min || nu >= bf->nu_max)
        return 0.0;
    double log_ratio = log(nu / bf->nu_min);
    int bin = (int)(log_ratio / bf->d_log_nu);
    if (bin < 0) return 0.0;
    if (bin >= bf->n_freq_bins - 1) return bf->eta_bf[shell * bf->n_freq_bins + bf->n_freq_bins - 1];
    double frac = log_ratio / bf->d_log_nu - (double)bin;
    double e0 = bf->eta_bf[shell * bf->n_freq_bins + bin];
    double e1 = bf->eta_bf[shell * bf->n_freq_bins + bin + 1];
    return e0 + frac * (e1 - e0);
}

/* Compute chi_bf grid for all shells and frequency bins.
 * Uses Kramers hydrogenic cross-section: sigma(nu) = sigma_0 * (nu_edge/nu)^3
 * where sigma_0 = 7.91e-18 / Z_eff^2 cm^2.
 * Sums over all ions and their levels weighted by level population. */
/* P7: Tabulated ground-state photoionization cross-sections from CMFGEN data.
 * Returns σ₀ in cm² (1 Mb = 1e-18 cm²). Ions not in table return 0 → Kramers fallback.
 * Sources: CMFGEN phot_data files (C,Mg,Ca,Cr,Fe,Co,Ni); estimated for Si,S,Ti,Al,Sc,V,Mn.
 * Non-static so the CUDA NLTE GEMM (lumina_nlte_gemm.cu) can share the same table. */
double get_bf_sigma0(int Z, int stage) {
    switch (Z) {
    case 6:  return (stage == 1) ? 3.75e-18 : (stage == 2) ? 1.27e-18 : 0;  /* C  */
    case 12: return (stage == 1) ? 0.23e-18 : (stage == 2) ? 5.42e-18 : 0;  /* Mg */
    case 13: return (stage == 1) ? 0.30e-18 : (stage == 2) ? 1.00e-18 : (stage == 3) ? 0.80e-18 : 0;  /* Al (est, UV multiplets) */
    case 14: return (stage == 1) ? 1.00e-18 : (stage == 2) ? 3.00e-18 : 0;  /* Si (est) */
    case 16: return (stage == 1) ? 2.00e-18 : (stage == 2) ? 3.00e-18 : 0;  /* S  (est) */
    case 20: return (stage == 1) ? 0.31e-18 : (stage == 2) ? 1.92e-18 : 0;  /* Ca */
    case 21: return (stage == 1) ? 3.00e-18 : (stage == 2) ? 2.50e-18 : (stage == 3) ? 2.00e-18 : 0;  /* Sc (est, d-shell) */
    case 22: return (stage == 1) ? 3.00e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Ti (est) */
    case 23: return (stage == 1) ? 3.50e-18 : (stage == 2) ? 3.00e-18 : (stage == 3) ? 2.50e-18 : 0;  /* V (est, d-shell) */
    case 24: return (stage == 1) ? 2.00e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Cr (est) */
    case 25: return (stage == 1) ? 2.50e-18 : (stage == 2) ? 3.00e-18 : (stage == 3) ? 2.50e-18 : 0;  /* Mn (est, d-shell) */
    case 26: return (stage == 1) ? 5.26e-18 : (stage == 2) ? 8.82e-18 : 0;  /* Fe */
    case 27: return (stage == 1) ?10.10e-18 : (stage == 2) ? 2.00e-18 : 0;  /* Co */
    case 28: return (stage == 1) ? 7.27e-18 : (stage == 2) ? 3.00e-18 : 0;  /* Ni */
    default: return 0;
    }
}

/* Wave-3.2 R5/D3: exact Kramers edge prescription used by the legacy pair
 * assembler.  Exported locally (element_wide.c declares it extern) so the EW
 * fallback cannot drift from the authority lane again. */
double nlte_bf_kramers_sigma0(int Z, int stage) {
    double sigma_0 = get_bf_sigma0(Z, stage);
    if (sigma_0 <= 0.0) {
        int Z_eff = rates_fix_enabled() ? (stage + 1) : (Z - stage);
        if (Z_eff < 1) Z_eff = 1;
        sigma_0 = 7.91e-18 / ((double)Z_eff * (double)Z_eff);
    }
    return sigma_0;
}

/* [BF-NLTE-POPS] Fix A: active NLTEConfig registered for chi_bf level pops.
 * compute_bf_opacity takes no nlte arg and two of its callers (lumina_main.c,
 * lumina_cmfgen.c) are outside the editable set, so the CUDA driver registers
 * the live NLTEConfig here right after nlte_init. The struct address and its
 * nlte_level_populations buffer are stable across outer iterations, so one
 * registration covers every compute_bf_opacity call in the executable.
 * NULL (e.g. CPU build, or NLTE disabled) => LUMINA_BF_NLTE_POPS falls back to
 * dilute-Boltzmann everywhere (graceful, counted). */
static NLTEConfig *g_bf_nlte_pops = NULL;
void bf_set_nlte_pops(NLTEConfig *nlte) { g_bf_nlte_pops = nlte; }

int bf_fill_committed_level_populations(const AtomicData *atom,
                                        const PlasmaState *plasma,
                                        int n_shells, float *out) {
    if (!atom || !plasma || !out || n_shells <= 0 ||
        atom->population_committed_generation == 0 ||
        !atom->ion_number_density || !atom->partition_functions ||
        !plasma->T_e) return -1;
    PopulationAtomicView av = population_atomic_view(atom);
    int use_nlte = g_bf_nlte_pops &&
        g_bf_nlte_pops->population_committed_generation ==
            atom->population_committed_generation &&
        g_bf_nlte_pops->nlte_level_populations &&
        g_bf_nlte_pops->global_to_nlte_level;
    for (int s = 0; s < n_shells; ++s) {
        for (int ip = 0; ip < atom->n_ion_pops; ++ip) {
            double nion = atom->ion_number_density[(size_t)ip*n_shells+s];
            double z = atom->partition_functions[(size_t)ip*n_shells+s];
            for (int l = atom->level_offset[ip]; l < atom->level_offset[ip+1]; ++l) {
                double value = 0.0;
                int ni = use_nlte ? g_bf_nlte_pops->global_to_nlte_level[l] : -1;
                if (ni >= 0) {
                    value = g_bf_nlte_pops->nlte_level_populations[
                        (size_t)ni*n_shells+s];
                } else {
                    double fraction = 0.0;
                    PopulationStatus st = population_lte_level_fraction(
                        &av, (size_t)ip, (size_t)l, plasma->T_e[s], z,
                        &fraction);
                    if (st != POP_OK && st != POP_EXACT_ZERO) return -1;
                    value = nion * fraction;
                }
                if (!isfinite(value) || value < 0.0 || value > FLT_MAX)
                    return -1;
                out[(size_t)s*atom->n_levels+l] = (float)value;
            }
        }
    }
    return 0;
}

/* Build the static ARTIS allcont-style route identity once. The per-shell
 * weights and stimulated-recombination ratios are refreshed by
 * compute_bf_opacity on every call. Unmapped levels retain a route whose target
 * is the represented upper-ion ground; if that is also unavailable target=-1
 * makes the event fail closed to the thermal pool rather than inventing an MA
 * level. */
static void bf_event_build_routes(BFOpacity *bf, AtomicData *atom,
                                  const int *ionized_ground, int n_shells) {
    if (!bf->event_enabled || bf->event_level_offset) return;

    int nlev = atom->n_levels;
    int nroutes = 0;
    bf->event_level_offset = (int *)malloc((size_t)(nlev + 1) * sizeof(int));
    for (int l = 0; l < nlev; l++) {
        bf->event_level_offset[l] = nroutes;
        int nr = 0;
        if (atom->ma_rr_loaded && atom->ma_rr_target_offset)
            nr = atom->ma_rr_target_offset[l + 1] -
                 atom->ma_rr_target_offset[l];
        nroutes += (nr > 0) ? nr : 1;
    }
    bf->event_level_offset[nlev] = nroutes;
    bf->event_n_levels = nlev;
    bf->event_n_routes = nroutes;

    bf->event_element = (int *)malloc((size_t)nroutes * sizeof(int));
    bf->event_ion = (int *)malloc((size_t)nroutes * sizeof(int));
    bf->event_level = (int *)malloc((size_t)nroutes * sizeof(int));
    bf->event_target = (int *)malloc((size_t)nroutes * sizeof(int));
    bf->event_target_fallback =
        (int *)calloc((size_t)nroutes, sizeof(int));
    bf->event_has_sigma = (int *)calloc((size_t)nroutes, sizeof(int));
    bf->event_nu_edge = (double *)malloc((size_t)nroutes * sizeof(double));
    bf->event_sigma0 = (double *)calloc((size_t)nroutes, sizeof(double));
    bf->event_weight = (double *)calloc((size_t)n_shells * nroutes,
                                        sizeof(double));
    bf->event_stim_ratio = (double *)malloc((size_t)n_shells * nroutes *
                                             sizeof(double));
    bf->event_chi_bf = (double *)calloc((size_t)n_shells * bf->n_freq_bins,
                                        sizeof(double));
    bf->event_Te = (double *)malloc((size_t)n_shells * sizeof(double));
    bf->event_sigma_bf = (atom->cmfgen_loaded &&
                           atom->cmfgen_n_freq_bins == bf->n_freq_bins)
                        ? atom->cmfgen_sigma_bf : NULL;
    for (size_t i = 0; i < (size_t)n_shells * nroutes; i++)
        bf->event_stim_ratio[i] = -1.0;

    int ip = 0;
    int fallback_routes = 0;
    for (int l = 0; l < nlev; l++) {
        while (ip + 1 < atom->n_ion_pops && l >= atom->level_offset[ip + 1])
            ip++;
        int rb = bf->event_level_offset[l];
        int re = bf->event_level_offset[l + 1];
        int csr_begin = 0, csr_count = 0;
        if (atom->ma_rr_loaded && atom->ma_rr_target_offset) {
            csr_begin = atom->ma_rr_target_offset[l];
            csr_count = atom->ma_rr_target_offset[l + 1] - csr_begin;
        }
        int has_target_map = csr_count > 0 && atom->ma_rr_targets;
        for (int r = rb; r < re; r++) {
            int q = r - rb;
            bf->event_element[r] = atom->level_Z[l];
            bf->event_ion[r] = atom->level_ion[l];
            bf->event_level[r] = l;
            bf->event_target[r] = has_target_map
                ? atom->ma_rr_targets[csr_begin + q]
                : ionized_ground[ip];
            bf->event_target_fallback[r] = has_target_map ? 0 : 1;
            fallback_routes += bf->event_target_fallback[r];
            bf->event_nu_edge[r] = -1.0;
        }
    }

    printf("[FIX-BF-CONTINUUM-EVENT] route table: %d lower levels, "
           "%d (element,ion,level,target) continua (%d phixs-mapped, "
           "%d upper-ground fallback); event CDF + nu_edge/nu split armed\n",
           nlev, nroutes, nroutes - fallback_routes, fallback_routes);
}

void compute_bf_opacity(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
                         int n_shells) {
    if (!bf->enabled) return;

    /* Zero the grid and activation table */
    size_t grid_size = (size_t)n_shells * bf->n_freq_bins;
    memset(bf->chi_bf, 0, grid_size * sizeof(double));
    if (bf->eta_bf) memset(bf->eta_bf, 0, grid_size * sizeof(double));
    memset(bf->activation_level, -1, grid_size * sizeof(int));
    bf->event_enabled = lumina_fix_bf_continuum_event_enabled();
    if (bf->event_chi_bf)
        memset(bf->event_chi_bf, 0, grid_size * sizeof(double));
    if (bf->event_weight)
        memset(bf->event_weight, 0,
               (size_t)n_shells * bf->event_n_routes * sizeof(double));
    if (bf->event_stim_ratio)
        for (size_t i = 0;
             i < (size_t)n_shells * bf->event_n_routes; i++)
            bf->event_stim_ratio[i] = -1.0;
    if (bf->event_Te && plasma->T_e)
        memcpy(bf->event_Te, plasma->T_e,
               (size_t)n_shells * sizeof(double));

    /* LUMINA_CMF_BF_MILNE=1: build the bf EMISSIVITY eta_bf alongside chi_bf.
     * Metastable levels (weight=1, trustworthy pops) get the NLTE bf source-
     * function form  S_bf = (2 h nu^3/c^2) / (b_l e^{h nu/kTe} - 1)  with the
     * Menzel departure b_l = n_l/n_l*(Saha-Boltzmann @T_e, actual n_e/n_next),
     * rewritten overflow-safe as  b_l e^{h nu/kTe} = Cinv_l e^{h(nu-nu_edge)/kTe},
     * Cinv_l = 2 U_next n_l / (n_next n_e g_l saha(T_e)).  Dilute (non-meta)
     * levels stay thermal S=B(T_e) — applying the departure there injects the
     * spurious optical super-thermal 1e5-1e7 (probe-confirmed, 2026-07-05).
     * LTE limit: b_l -> 1 gives S_bf -> B(T_e) exactly. Gate off => eta_bf
     * stays zero and the assemble uses the legacy chi*B path (byte-identical). */
    static int bf_milne = -1;
    static int bf_ots = 0;   /* LUMINA_CMF_OTS=1: case-B / on-the-spot —
        * ground-edge recombination photons are locally reabsorbed (net zero),
        * so they are EXCLUDED from the emissivity eta_bf; the EPAY budget
        * then exits through excited edges/lines/ff. Standard nebular
        * approximation; kills the far-edge recombination-photon recycling
        * loop (Gph(S III)=1.9e-6 >> n_e*alpha=1.4e-7 from paid edge photons,
        * where CMFGEN's intact S III column self-shields, tau_bf~2.6). */
    if (bf_milne < 0) { const char *e = getenv("LUMINA_CMF_BF_MILNE");
                        bf_milne = e ? atoi(e) : 0;
                        const char *o = getenv("LUMINA_CMF_OTS");
                        bf_ots = (o && atoi(o)) ? 1 : 0;
                        if (bf_ots) printf("[BF-OTS] case-B: ground-edge "
                                           "recombination emission excluded\n");
                        if (bf_milne) printf("[BF-MILNE] eta_bf source-function "
                                             "build ON (%s)\n",
                                             bf_milne >= 2 ? "ALL levels"
                                                           : "meta-only departure"); }

    const int bf_stim_recomb = lumina_fix_bf_stim_recomb_enabled();
    const int bf_neutral = lumina_fix_bf_neutral_enabled();
    const int bf_eta_spingate_fix = lumina_fix_bf_eta_spingate_enabled();
    const int bf_eta_spingate = bf_eta_spingate_fix && rec_spingate_enabled();
    {
        static int stim_banner = 0, neutral_banner = 0, eta_spin_banner = 0;
        if (bf_stim_recomb && !stim_banner) {
            printf("[FIX-BF-STIM-RECOMB] chi_bf uses ARTIS rpkt.cc:733-765 "
                   "net coefficient max(0,1-r*exp[-h(nu-nu_edge)/kT_e])\n");
            stim_banner = 1;
        }
        if (bf_neutral && !neutral_banner) {
            printf("[FIX-BF-NEUTRAL] neutral photoionization continua included "
                   "(stage=0; e.g. O I -> O II)\n");
            neutral_banner = 1;
        }
        if (bf_eta_spingate_fix && !eta_spin_banner) {
            printf("[FIX-BF-ETA-SPINGATE] eta_bf Milne level emissivity %s "
                   "LUMINA_REC_SPINGATE spin predicate\n",
                   bf_eta_spingate ? "uses" : "requested but INERT without");
            eta_spin_banner = 1;
        }
    }

    /* A2-07: a committed solved population is always preferred. Untracked
     * levels (and the pre-solve generation) use the one LTE@T_e reference;
     * the legacy selector is diagnostic shadow and cannot change physics. */
    const int use_nlte_pops = g_bf_nlte_pops &&
                              g_bf_nlte_pops->population_committed_generation > 0 &&
                              g_bf_nlte_pops->nlte_level_populations &&
                              g_bf_nlte_pops->global_to_nlte_level;
    long bf_nlte_used = 0, bf_nlte_fb = 0;   /* [BF-NLTE-POPS] per-call tally */

    /* Precompute bin center frequencies (used by both CPU and free-free paths) */
    double *nu_bin = (double *)malloc(bf->n_freq_bins * sizeof(double));
    for (int b = 0; b < bf->n_freq_bins; b++) {
        nu_bin[b] = bf->nu_min * exp((b + 0.5) * bf->d_log_nu);
    }

#ifdef LUMINA_HAS_CUDA_BF_GEMM
    /* Task #39: GPU GEMM path (TF32 tensor cores) when CMFGEN sigma_bf is
     * loaded and LUMINA_BF_GEMM=1. Fills chi_bf[s,f] = sum_l n_level[s,l] *
     * sigma_bf[l,f] in a single batched GEMM, then jumps straight to free-free.
     * D-3 corrfactor is level+frequency dependent and cannot be represented by
     * this unmodified GEMM, so its repair gate deliberately selects the exact
     * CPU summation. Gate OFF preserves the original GEMM selection byte-for-byte. */
    if (atom->cmfgen_loaded && !bf_milne && !bf_stim_recomb &&
        !bf->event_enabled &&
        getenv("LUMINA_BF_GEMM") &&
        atom->population_committed_generation > 0) {
        if (bf_gemm_compute(bf, atom, plasma, n_shells) == 0) {
            goto compute_ff;
        }
        fprintf(stderr, "[A2-14][FATAL] committed GPU opacity publication failed; "
                        "CPU fallback forbidden\n");
        exit(EXIT_FAILURE);
    }
#endif

    /* Legacy per-bin dominant absorber tracking. D-1 replaces this structure
     * rather than building an unused argmax table on its ON arm. */
    double *best_chi = bf->event_enabled
                     ? NULL : (double *)calloc(grid_size, sizeof(double));
    int *best_ip = bf->event_enabled
                 ? NULL : (int *)malloc(grid_size * sizeof(int));
    if (best_ip) memset(best_ip, -1, grid_size * sizeof(int));

    /* [BF-DIAG] one-shot tally of CMFGEN-vs-Kramers per-level σ_bf usage.
     * Only the first call with non-empty active levels prints; subsequent iters
     * reuse the same atomic data. Set LUMINA_BF_DIAG=0 to suppress. */
    static int bf_diag_emitted = 0;
    long bf_diag_cmfgen_levels = 0;
    long bf_diag_kramers_levels = 0;

    /* Precompute ground-state level index of the NEXT-HIGHER ion for each ion pop.
     * When ion ip (Z, stage) absorbs BF, the atom becomes (Z, stage+1).
     * We activate macro-atom at ground state of (Z, stage+1). */
    int *ionized_ground = (int *)malloc(atom->n_ion_pops * sizeof(int));
    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        ionized_ground[ip] = -1;
        int Z_ion = atom->ion_pop_Z[ip];
        int next_stage = atom->ion_pop_stage[ip] + 1;
        /* Find ion pop for (Z, next_stage) */
        for (int jp = 0; jp < atom->n_ion_pops; jp++) {
            if (atom->ion_pop_Z[jp] == Z_ion && atom->ion_pop_stage[jp] == next_stage) {
                /* Find ground level (level_num=0) of that ion */
                int ls = atom->level_offset[jp];
                int le = atom->level_offset[jp + 1];
                for (int l = ls; l < le; l++) {
                    if (atom->level_num[l] == 0) {
                        ionized_ground[ip] = l;
                        break;
                    }
                }
                break;
            }
        }
    }

    bf_event_build_routes(bf, atom, ionized_ground, n_shells);
    if (bf->event_Te && plasma->T_e)
        memcpy(bf->event_Te, plasma->T_e,
               (size_t)n_shells * sizeof(double));

    /* [MA-RADRECOMB B4] Data-driven bf-activation target: when LUMINA_MA_RADRECOMB=1
     * and the CMFGEN photoionization TARGET map is loaded, per ion pop resolve the
     * upper-ion level each bf absorption should activate from the map (the first level
     * of the ion with a mapped target — all levels of a mapped ion route to the same
     * upper-ion ground). Falls back to ionized_ground where unmapped (fail-closed). For
     * the mapped iron-peak/S/Si/Ca ions the CMFGEN final state is the upper-ion GROUND
     * (excitation energy 0), so rr_act == ionized_ground: the "not ground-only" routing
     * is a validated identity here, now sourced from data. Gate OFF => rr_act NULL =>
     * ionized_ground used verbatim (byte-identical). */
    int *rr_act = NULL;
    {
        static int rr_mode_bf = -1;
        if (rr_mode_bf < 0) {
            const char *er = getenv("LUMINA_MA_RADRECOMB");
            rr_mode_bf = (er && atoi(er) != 0) ? 1 : 0;
        }
        if (rr_mode_bf && atom->ma_rr_loaded && atom->ma_rr_target) {
            rr_act = (int *)malloc(atom->n_ion_pops * sizeof(int));
            for (int ip = 0; ip < atom->n_ion_pops; ip++) {
                rr_act[ip] = ionized_ground[ip];       /* default = ground */
                int ls = atom->level_offset[ip], le = atom->level_offset[ip + 1];
                for (int l = ls; l < le; l++)
                    if (atom->ma_rr_target[l] >= 0) { rr_act[ip] = atom->ma_rr_target[l]; break; }
            }
        }
    }

    /* Scratch for ARTIS' per-target continuum sum. A lower level may have more
     * than one upper target; corrfactor is target-dependent through n_upper/g_upper,
     * so probabilities cannot be collapsed before applying max(0,1-stimfactor). */
    int stim_max_routes = 1;
    if (bf_stim_recomb && atom->ma_rr_target_offset) {
        for (int l = 0; l < atom->n_levels; l++) {
            int nr = atom->ma_rr_target_offset[l + 1] -
                     atom->ma_rr_target_offset[l];
            if (nr > stim_max_routes) stim_max_routes = nr;
        }
    }
    double *stim_route_ratio = bf_stim_recomb
        ? (double *)calloc((size_t)stim_max_routes, sizeof(double)) : NULL;
    double *stim_route_prob = bf_stim_recomb
        ? (double *)calloc((size_t)stim_max_routes, sizeof(double)) : NULL;
    unsigned char *stim_route_valid = bf_stim_recomb
        ? (unsigned char *)calloc((size_t)stim_max_routes, 1) : NULL;

    for (int ip = 0; ip < atom->n_ion_pops; ip++) {
        int Z_ion = atom->ion_pop_Z[ip];
        int stage = atom->ion_pop_stage[ip];
        /* [Wave-1 neutral-bf] Neutral atoms have ordinary bound-free edges
         * X I + hnu -> X II + e (O I is the audit witness). The historical
         * stage<1 skip confused zero ionic charge (relevant to free-free) with
         * absence of photoionization. Retain that exact skip unless the
         * default-OFF repair gate is armed; missing upper stages still fail
         * closed through the ionization-energy lookup below. */
        if (stage < 1 && !bf_neutral) continue;

        /* Find ionization energy for this ion */
        double chi_eV = -1.0;
        for (int k = 0; k < atom->n_ionization; k++) {
            if (atom->ioniz_Z[k] == Z_ion && atom->ioniz_ion[k] == stage) {
                chi_eV = atom->ioniz_energy_eV[k];
                break;
            }
        }
        if (chi_eV <= 0.0) continue;
        double chi_erg = chi_eV * EV_TO_ERG;

        /* P7: Tabulated cross-section (CMFGEN) or Kramers fallback */
        double sigma_0_kramers = get_bf_sigma0(Z_ion, stage);
        if (sigma_0_kramers <= 0.0) {
            /* [RATES-FIX F5] see parity_gamma_phot: Z_eff = stage+1, not Z-stage */
            int Z_eff = (rates_fix_enabled() || (bf_neutral && stage == 0))
                      ? (stage + 1) : (Z_ion - stage);
            if (Z_eff < 1) Z_eff = 1;
            sigma_0_kramers = 7.91e-18 / ((double)Z_eff * (double)Z_eff);
        }

        int lev_start = atom->level_offset[ip];
        int lev_end   = atom->level_offset[ip + 1];

        /* Upper ion used by both the Milne emissivity and the ARTIS
         * stimulated-recombination departure ratio. */
        int ip_next = -1;
        if (bf_milne || bf_stim_recomb || bf_eta_spingate) {
            for (int jp = 0; jp < atom->n_ion_pops; jp++)
                if (atom->ion_pop_Z[jp] == Z_ion &&
                    atom->ion_pop_stage[jp] == stage + 1) { ip_next = jp; break; }
        }
        int eta_M_core = 0;
        if (bf_milne && bf_eta_spingate) {
            rec_spingate_check_data(atom);
            eta_M_core = spingate_resolve_core_mult(atom, ip_next, Z_ion,
                                                     stage + 1, NULL);
        }

        /* Task #38: Per-level CMFGEN ν-dependent σ_bf when available.
         * Baked grid layout matches bf->n_freq_bins exactly, so we can index
         * directly without interpolation. Falls back to Kramers per-level. */
        const int  use_cmfgen = atom->cmfgen_loaded &&
                                atom->cmfgen_n_freq_bins == bf->n_freq_bins;
        const double *sigma_grid = use_cmfgen ? atom->cmfgen_sigma_bf : NULL;
        const int    *has_sigma  = use_cmfgen ? atom->cmfgen_has_sigma : NULL;

        for (int s = 0; s < n_shells; s++) {
            double n_ion = atom->ion_number_density[ip * n_shells + s];
            double Z_part = atom->partition_functions[ip * n_shells + s];
            if (n_ion < 1e-30 || Z_part < 1e-300) continue;

            /* BF-MILNE per-shell pieces: kTe, saha(T_e), next-ion density and
             * partition (U floor 1 — tiny-positive U exploded the v1 Milne). */
            double kTe_m = 0.0, saha_m = 0.0, n_next_m = 0.0, U_next_m = 0.0;
            int milne_ok = 0;
            if ((bf_milne || bf_stim_recomb) && ip_next >= 0 &&
                plasma->n_electron) {
                double Te_m = plasma->T_e[s];
                double ne_m = plasma->n_electron[s];
                n_next_m = atom->ion_number_density[ip_next * n_shells + s];
                U_next_m = atom->partition_functions[ip_next * n_shells + s];
                if (Te_m > 0.0 && ne_m > 0.0 && n_next_m > 1e-30 &&
                    U_next_m > 0.0 && isfinite(U_next_m)) {
                    kTe_m = K_BOLTZMANN * Te_m;
                    saha_m = pow(H_PLANCK * H_PLANCK /
                                 (2.0 * M_PI_VAL * M_ELECTRON * kTe_m), 1.5);
                    /* fold the level-independent part: 2 U_next/(n_next n_e saha) */
                    saha_m = 2.0 * U_next_m / (n_next_m * ne_m * saha_m);
                    milne_ok = 1;
                }
            }

            for (int l = lev_start; l < lev_end; l++) {
                double E_eV = atom->level_energy_eV[l];
                int g = atom->level_g[l];
                int is_meta = atom->level_metastable[l];

                /* Population consumption is exact: a tracked level consumes its
                 * committed NLTE value, including an exact physical zero; only an
                 * untracked level uses the sole LTE@T_e reference. */
                PopulationAtomicView av = population_atomic_view(atom);
                double level_fraction = 0.0;
                double n_level = 0.0;
                int nlte_idx = (use_nlte_pops &&
                                g_bf_nlte_pops->global_to_nlte_level)
                             ? g_bf_nlte_pops->global_to_nlte_level[l] : -1;
                if (nlte_idx >= 0) {
                    n_level = g_bf_nlte_pops->nlte_level_populations[
                        (size_t)nlte_idx * n_shells + s];
                    if (!isfinite(n_level) || n_level < 0.0) continue;
                    bf_nlte_used++;
                } else {
                    PopulationStatus level_status = population_lte_level_fraction(
                        &av, (size_t)ip, (size_t)l, plasma->T_e[s], Z_part,
                        &level_fraction);
                    if (level_status != POP_OK && level_status != POP_EXACT_ZERO)
                        continue;
                    n_level = n_ion * level_fraction;
                    if (use_nlte_pops) bf_nlte_fb++; /* untracked LTE@T_e */
                }
                if (n_level < 1e-30) continue;

                /* Ionization edge for this level: nu_edge = (chi_ion - E_level) / h.
                 * The repair arm uses the ARTIS constants verbatim, including the
                 * historical H/KB values in constants.h, for numeric as well as
                 * algebraic parity. Gate OFF retains the Lumina/NIST edge exactly. */
                double E_level_erg = E_eV * EV_TO_ERG;
                const double ARTIS_H = 6.6260755e-27;
                const double ARTIS_KB = 1.38064852e-16;
                double nu_edge = (chi_erg - E_level_erg) /
                                 (bf_stim_recomb ? ARTIS_H : H_PLANCK);
                if (nu_edge <= bf->nu_min) continue;  /* edge below our grid */

                /* Find starting bin for this edge */
                int bin_start = 0;
                if (nu_edge > bf->nu_min) {
                    bin_start = (int)(log(nu_edge / bf->nu_min) / bf->d_log_nu);
                    if (bin_start < 0) bin_start = 0;
                }

                int level_has_cmfgen = use_cmfgen && has_sigma[l];
                const double *sigma_row = level_has_cmfgen
                    ? &sigma_grid[(size_t)l * (size_t)bf->n_freq_bins]
                    : NULL;
                if (bf->event_enabled && bf->event_level_offset) {
                    int erb = bf->event_level_offset[l];
                    int ere = bf->event_level_offset[l + 1];
                    for (int er = erb; er < ere; er++) {
                        bf->event_nu_edge[er] = nu_edge;
                        bf->event_sigma0[er] = sigma_0_kramers;
                        bf->event_has_sigma[er] = level_has_cmfgen;
                    }
                }
                if (!bf_diag_emitted && s == 0) {
                    if (level_has_cmfgen) bf_diag_cmfgen_levels++;
                    else                  bf_diag_kramers_levels++;
                }

                /* BF-MILNE: level departure prefactor Cinv_l (see head note);
                 * meta-only. Non-meta/unavailable -> thermal B(T_e). */
                double Cinv_l = 0.0;
                int use_milne_l = 0;
                double Te_s = plasma->T_e[s];
                if (bf_milne && milne_ok && (is_meta || bf_milne >= 2) && g > 0) {
                    Cinv_l = saha_m * n_level / (double)g;
                    if (Cinv_l > 0.0 && isfinite(Cinv_l)) use_milne_l = 1;
                }

                /* [Wave-1 D-3] ARTIS rpkt.cc:733-765, verbatim physics:
                 *   r_mod = n_upper/n_l * n_e * SAHACONST
                 *           * (g_l/g_upper) * T_e^(-3/2)
                 *   corr = max(0, 1 - r_mod exp[-HOVERKB(nu-nu_edge)/T_e])
                 *   chi_l = n_l sigma_bf target_probability corr.
                 * The repair gate independently loads ma_rr_target and its
                 * probability; established v1 CMFGEN maps are single-route p=1.
                 * Unmapped/Kramers continua retain their represented upper-ion
                 * ground and p=1 fallback.
                 *
                 * SAHACONST=2.0706659e-16 is ARTIS constants.h and equals one
                 * half of the electron thermal de-Broglie volume coefficient.
                 * At LTE r_mod=exp(-h nu_edge/kT), hence corr recovers the
                 * Kirchhoff factor 1-exp(-h nu/kT). */
                int stim_route_begin = 0;
                int stim_route_count = 0;
                int stim_has_csr = 0;
                if (bf_stim_recomb && atom->ma_rr_loaded &&
                    atom->ma_rr_target_offset && atom->ma_rr_targets &&
                    atom->ma_rr_probability) {
                    stim_route_begin = atom->ma_rr_target_offset[l];
                    stim_route_count = atom->ma_rr_target_offset[l + 1] -
                                       stim_route_begin;
                    stim_has_csr = stim_route_count > 0;
                }
                if (bf_stim_recomb && !stim_has_csr) stim_route_count = 1;
                for (int q = 0; q < stim_route_count; q++) {
                    stim_route_ratio[q] = 0.0;
                    stim_route_valid[q] = 0;
                    stim_route_prob[q] = stim_has_csr
                        ? atom->ma_rr_probability[stim_route_begin + q] : 1.0;
                }
                if (bf_stim_recomb && milne_ok && g > 0) {
                  for (int q = 0; q < stim_route_count; q++) {
                    int upper_l = stim_has_csr
                        ? atom->ma_rr_targets[stim_route_begin + q]
                        : ionized_ground[ip];
                    if (stim_route_prob[q] > 0.0 &&
                        upper_l >= 0 && upper_l < atom->n_levels) {
                        int g_upper = atom->level_g[upper_l];
                        double n_upper = 0.0;
                        if (g_upper > 0) {
                            int ni_upper = (use_nlte_pops &&
                                            g_bf_nlte_pops->global_to_nlte_level)
                                         ? g_bf_nlte_pops->global_to_nlte_level[upper_l]
                                         : -1;
                            if (ni_upper >= 0) {
                                n_upper = g_bf_nlte_pops->nlte_level_populations[
                                    (size_t)ni_upper * n_shells + s];
                            } else {
                                double upper_fraction = 0.0;
                                PopulationStatus upper_status =
                                    population_lte_level_fraction(
                                        &av, (size_t)ip_next, (size_t)upper_l,
                                        plasma->T_e[s], U_next_m,
                                        &upper_fraction);
                                if (upper_status == POP_OK ||
                                    upper_status == POP_EXACT_ZERO)
                                    n_upper = n_next_m * upper_fraction;
                                else
                                    n_upper = NAN;
                            }
                            if (n_upper >= 0.0 && isfinite(n_upper)) {
                                const double ARTIS_SAHACONST = 2.0706659e-16;
                                double Te_saha = plasma->T_e[s];
                                double clump = (plasma->clump_factor &&
                                                plasma->clump_factor[s] > 0.0)
                                             ? plasma->clump_factor[s] : 1.0;
                                double clumped_ne = plasma->n_electron[s] * clump;
                                double modified_departure_ratio =
                                    n_upper / n_level * clumped_ne *
                                    ARTIS_SAHACONST *
                                    ((double)g / (double)g_upper) *
                                    pow(Te_saha, -1.5);
                                if (modified_departure_ratio >= 0.0 &&
                                    isfinite(modified_departure_ratio)) {
                                    stim_route_ratio[q] = modified_departure_ratio;
                                    stim_route_valid[q] = 1;
                                }
                            }
                        }
                    }
                  }
                }

                /* D-1 route coefficients. Target probabilities are part of
                 * continuum identity even when the independent D-3
                 * stimulated-recombination gate is off. */
                if (bf->event_enabled && bf->event_level_offset) {
                    int erb = bf->event_level_offset[l];
                    int ere = bf->event_level_offset[l + 1];
                    int csr_begin = 0, csr_count = 0;
                    if (atom->ma_rr_loaded && atom->ma_rr_target_offset) {
                        csr_begin = atom->ma_rr_target_offset[l];
                        csr_count = atom->ma_rr_target_offset[l + 1] -
                                    csr_begin;
                    }
                    for (int er = erb; er < ere; er++) {
                        int q = er - erb;
                        double p_route = (csr_count > 0 &&
                                          atom->ma_rr_probability)
                                       ? atom->ma_rr_probability[csr_begin + q]
                                       : 1.0;
                        size_t ei = (size_t)s * bf->event_n_routes + er;
                        bf->event_weight[ei] = n_level * p_route;
                        if (bf_stim_recomb && q < stim_route_count &&
                            stim_route_valid[q])
                            bf->event_stim_ratio[ei] = stim_route_ratio[q];
                    }
                }

                /* Add contribution to all bins above the edge */
                for (int b = bin_start; b < bf->n_freq_bins; b++) {
                    double nu = nu_bin[b];
                    if (nu < nu_edge) continue;
                    double sigma;
                    if (sigma_row) {
                        sigma = sigma_row[b];
                        if (sigma <= 0.0) continue;
                    } else {
                        double ratio = nu_edge / nu;
                        sigma = sigma_0_kramers * ratio * ratio * ratio;
                    }
                    double chi_raw = n_level * sigma;
                    double chi_contrib = chi_raw;
                    double chi_spont_base = chi_contrib;
                    if (bf_stim_recomb) {
                        double weighted_corr = 0.0;
                        double probability_sum = 0.0;
                        double expfactor = exp(-(ARTIS_H / ARTIS_KB) *
                                               (nu - nu_edge) / plasma->T_e[s]);
                        for (int q = 0; q < stim_route_count; q++) {
                            double p = stim_route_prob[q];
                            double corrfactor = 1.0;
                            if (stim_route_valid[q]) {
                                double stimfactor = stim_route_ratio[q] * expfactor;
                                corrfactor = 1.0 - stimfactor;
                            }
                            probability_sum += p;
                            weighted_corr += p * corrfactor;
                        }
                        chi_spont_base = chi_raw * probability_sum;
                        chi_contrib = chi_raw * weighted_corr;
                    }
                    int idx = s * bf->n_freq_bins + b;
                    bf->chi_bf[idx] += chi_contrib;
                    if (bf->event_enabled && bf->event_chi_bf)
                        bf->event_chi_bf[idx] += chi_spont_base;
#ifdef LUMINA_FROZEN_ORACLE
                    double oracle_eta_contrib = 0.0;
#endif
                    /* [Wave-1 B18-a] When both gates are armed, eta_bf uses the
                     * identical REC_SPINGATE predicate as S3/S1: radiative
                     * capture from a core of multiplicity M_core reaches only
                     * daughter multiplicities M_core+/-1. Unknown multiplicity
                     * remains allowed, matching spingate_level_forbidden().
                     * LUMINA_FIX_BF_ETA_SPINGATE=0 leaves this test unreachable. */
                    if (bf_milne && bf->eta_bf &&
                        !(bf_ots && atom->level_num[l] == 0) &&
                        !(bf_eta_spingate &&
                          spingate_level_forbidden(atom, l, eta_M_core))) {
                        double S_l;
                        double eta_opacity;
                        if (use_milne_l) {
                            /* SPONTANEOUS Milne recombination only:
                             * eta = chi_raw * (2 h nu^3/c^2) e^{-x} / Cinv
                             *     = sigma (2 h nu^3/c^2) (n_e n_+ g_l saha /
                             *       2U_+) e^{-h(nu-nu_edge)/kTe}
                             * — no denominator, always bounded/positive.
                             * (The b e^x - 1 source-function form lases when
                             * b e^x <= 1: den-floor 1e-10 turned population-
                             * inversion bins into 1e10x maser spikes -> the
                             * epay4 J=1e42 runaway.) The D-3 corrfactor belongs
                             * only to net absorption; spontaneous eta therefore
                             * retains chi_raw here, exactly as in ARTIS. */
                            double x = H_PLANCK * (nu - nu_edge) / kTe_m;
                            S_l = (x > 600.0) ? 0.0
                                : 2.0 * H_PLANCK * nu * nu * nu /
                                  (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT) *
                                  exp(-x) / Cinv_l;
                            eta_opacity = chi_spont_base;
                        } else {
                            /* Thermal fallback is expressed as eta=chi_net*B;
                             * this recovers Kirchhoff when D-3 is enabled. */
                            S_l = planck_bnu(Te_s, nu);
                            eta_opacity = chi_contrib;
                        }
                        bf->eta_bf[idx] += eta_opacity * S_l;
#ifdef LUMINA_FROZEN_ORACLE
                        oracle_eta_contrib = eta_opacity * S_l;
#endif
                    }

#ifdef LUMINA_FROZEN_ORACLE
                    if (g_oracle.fp && s == 0) {
                        int os = oracle_ion_slot(Z_ion, stage);
                        int ow = oracle_wave_slot(bf, b);
                        if (os >= 0 && ow >= 0) {
                            g_oracle.chi[os][ow] += chi_contrib;
                            g_oracle.eta[os][ow] += oracle_eta_contrib;
                        }
                    }
#endif

                    /* Track dominant absorber for macro-atom activation */
                    if (!bf->event_enabled && chi_contrib > best_chi[idx]) {
                        best_chi[idx] = chi_contrib;
                        best_ip[idx] = ip;
                    }
                }
            }
        }
    }

    /* Build activation level table from dominant absorber. [MA-RADRECOMB B4] route
     * through the data-driven target rr_act when loaded (== ionized_ground for the
     * mapped ions), else ground-only. */
    int n_activated = 0;
    if (!bf->event_enabled) {
        for (size_t idx = 0; idx < grid_size; idx++) {
            int ip = best_ip[idx];
            if (ip < 0) continue;
            int act = rr_act ? rr_act[ip] : ionized_ground[ip];
            if (act >= 0) {
                bf->activation_level[idx] = act;
                n_activated++;
            }
        }
    }

    free(best_chi);
    free(best_ip);
    free(ionized_ground);
    free(rr_act);
    free(stim_route_ratio);
    free(stim_route_prob);
    free(stim_route_valid);

    {
        long total = bf_diag_cmfgen_levels + bf_diag_kramers_levels;
        if (!bf_diag_emitted && total > 0) {
            double pct = 100.0 * bf_diag_cmfgen_levels / (double)total;
            printf("[BF-DIAG] σ_bf source over %ld active levels (shell 0): "
                   "CMFGEN=%ld (%.1f%%), Kramers=%ld (%.1f%%)\n",
                   total, bf_diag_cmfgen_levels, pct,
                   bf_diag_kramers_levels, 100.0 - pct);
            bf_diag_emitted = 1;
        }
    }

    if (use_nlte_pops) {
        printf("  [A2-07][BF-POPS] chi_bf: solved=%ld  LTE@T_e-reference=%ld\n",
               bf_nlte_used, bf_nlte_fb);
    }

#ifdef LUMINA_HAS_CUDA_BF_GEMM
compute_ff:
#endif
    /* --- Free-free (bremsstrahlung) opacity --- */
    for (int s = 0; s < n_shells; s++) {
        double T_e = plasma->T_e[s];
        double n_e = plasma->n_electron[s];
        if (T_e <= 0.0 || n_e <= 0.0) continue;

        double sqrt_Te_inv = 1.0 / sqrt(T_e);
        double kT_e = K_BOLTZMANN * T_e;

        /* Sum Z_eff^2 * n_ion over all ions */
        double Z2_n_sum = 0.0;
        for (int ip = 0; ip < atom->n_ion_pops; ip++) {
            int ion_stage = atom->ion_pop_stage[ip];  /* 0=neutral, 1=II, 2=III */
            if (ion_stage < 1) continue;              /* neutrals don't contribute */
            double Z_eff = (double)ion_stage;
            double n_ion = atom->ion_number_density[ip * n_shells + s];
            Z2_n_sum += Z_eff * Z_eff * n_ion;
        }

        double coeff = C_FF_OPACITY * sqrt_Te_inv * n_e * Z2_n_sum;

        for (int b = 0; b < bf->n_freq_bins; b++) {
            double nu = nu_bin[b];
            double nu3 = nu * nu * nu;
            double stim = 1.0 - exp(-H_PLANCK * nu / kT_e);
            double chi_ff_c = coeff / nu3 * stim;
            bf->chi_bf[s * bf->n_freq_bins + b] += chi_ff_c;
            if (bf->eta_bf && bf_milne)
                bf->eta_bf[s * bf->n_freq_bins + b] += chi_ff_c * planck_bnu(T_e, nu);
#ifdef LUMINA_FROZEN_ORACLE
            if (g_oracle.fp && s == 0) {
                double nu_lo = bf->nu_min * exp((double)b * bf->d_log_nu);
                double nu_hi = nu_lo * exp(bf->d_log_nu);
                double eta_ff_c = (bf->eta_bf && bf_milne)
                                ? chi_ff_c * planck_bnu(T_e, nu) : 0.0;
                g_oracle.ff_cooling_grid +=
                    4.0 * M_PI_VAL * eta_ff_c * (nu_hi - nu_lo);
                int ow = oracle_wave_slot(bf, b);
                if (ow >= 0) {
                    g_oracle.ff_chi[ow] += chi_ff_c;
                    g_oracle.ff_eta[ow] += eta_ff_c;
                }
            }
#endif
        }
    }

    free(nu_bin);

    /* Print diagnostics: BF and FF contributions separately for shell 0 */
    double chi_bf_max_opt = 0.0, chi_bf_max_uv = 0.0;
    double chi_ff_max_opt = 0.0, chi_ff_max_uv = 0.0;
    {
        /* Recompute FF-only for shell 0 for diagnostics */
        double T_e0 = plasma->T_e[0];
        double n_e0 = plasma->n_electron[0];
        double sqrt_Te_inv0 = (T_e0 > 0.0) ? 1.0 / sqrt(T_e0) : 0.0;
        double kT_e0 = K_BOLTZMANN * T_e0;
        double Z2_n_sum0 = 0.0;
        for (int ip = 0; ip < atom->n_ion_pops; ip++) {
            int ion_stage = atom->ion_pop_stage[ip];
            if (ion_stage < 1) continue;
            double Z_eff = (double)ion_stage;
            double n_ion = atom->ion_number_density[ip * n_shells + 0];
            Z2_n_sum0 += Z_eff * Z_eff * n_ion;
        }
        double coeff0 = C_FF_OPACITY * sqrt_Te_inv0 * n_e0 * Z2_n_sum0;

        for (int b = 0; b < bf->n_freq_bins; b++) {
            double nu = bf->nu_min * exp((b + 0.5) * bf->d_log_nu);
            double lam_A = C_SPEED_OF_LIGHT / nu * 1e8;
            double chi_total = bf->chi_bf[0 * bf->n_freq_bins + b];

            /* FF contribution at this freq */
            double nu3 = nu * nu * nu;
            double stim = (kT_e0 > 0.0) ? 1.0 - exp(-H_PLANCK * nu / kT_e0) : 0.0;
            double chi_ff = (coeff0 > 0.0) ? coeff0 / nu3 * stim : 0.0;
            double chi_bf = chi_total - chi_ff;

            if (lam_A >= 3500.0 && lam_A <= 9000.0) {
                if (chi_bf > chi_bf_max_opt) chi_bf_max_opt = chi_bf;
                if (chi_ff > chi_ff_max_opt) chi_ff_max_opt = chi_ff;
            }
            if (lam_A >= 1000.0 && lam_A < 3500.0) {
                if (chi_bf > chi_bf_max_uv) chi_bf_max_uv = chi_bf;
                if (chi_ff > chi_ff_max_uv) chi_ff_max_uv = chi_ff;
            }
        }
    }
    double chi_e0 = plasma->n_electron[0] * SIGMA_THOMSON;
    printf("  [BF+FF] Shell 0 (optical): chi_bf=%.2e  chi_ff=%.2e  chi_e=%.2e  (bf/e=%.2e  ff/e=%.2e)\n",
           chi_bf_max_opt, chi_ff_max_opt, chi_e0, chi_bf_max_opt/chi_e0, chi_ff_max_opt/chi_e0);
    printf("  [BF+FF] Shell 0 (UV):      chi_bf=%.2e  chi_ff=%.2e  chi_e=%.2e  (bf/e=%.2e  ff/e=%.2e)\n",
           chi_bf_max_uv, chi_ff_max_uv, chi_e0, chi_bf_max_uv/chi_e0, chi_ff_max_uv/chi_e0);
    printf("  [BF] Macro-atom activation: %d/%d bins have valid levels\n",
           n_activated, (int)grid_size);
}

int a208_publish_cpu_opacity(OpacityState *opacity, const BFOpacity *bf,
                             const AtomicData *atom, const PlasmaState *plasma,
                             const NLTEConfig *nlte,
                             double epoch) {
    if (!opacity || !atom || !plasma || !plasma->n_electron || !plasma->T_e ||
        opacity->n_shells<=0 || opacity->n_lines<0) return 5;
    size_t ns=(size_t)opacity->n_shells;
    size_t nb=bf?(size_t)bf->n_freq_bins:(size_t)NLTE_N_FREQ_BINS;
    CpuOpacityPublication candidate={0};
    /* The line publication is the generation-bound OpacityState value/status
     * pair; avoid a second multi-gigabyte copy of the 125M-cell line slab. */
    /* A2-14 extends the A2-08 publication to CUDA without changing the CPU
     * arithmetic: one aggregate BF route carries the nonnegative packet-event
     * measure independently of the signed BF coefficient. */
    if(a208_publication_init(&candidate,ns,nb,0,1)) return 2;
    candidate.generation_required=opacity->cpu_opacity.generation_required+1;
    candidate.epoch=epoch;
    candidate.population_generation=atom->population_committed_generation;
    candidate.partition_generation=atom->population_committed_generation;
    candidate.te_generation=plasma->T_e_generation;
    candidate.ne_generation=atom->population_committed_generation;
    candidate.tau_generation=opacity->tau_computed_generation;
    candidate.radiation_generation=nlte?nlte->radfield_view.generation:0;
    candidate.line_jbar_generation=nlte?nlte->line_view.generation:0;
    double nu_min=bf?bf->nu_min:NLTE_NU_MIN;
    double dlog=bf?bf->d_log_nu:log(NLTE_NU_MAX/NLTE_NU_MIN)/(double)nb;
    for(size_t b=0;b<=nb;b++) candidate.frequency_edges[b]=nu_min*exp(dlog*(double)b);
    A208Counters *ctr=a208_counters();
    ctr->generation_required=candidate.generation_required;
    ctr->shells_attempted+=ns;ctr->cells_attempted+=ns*nb;
    for(size_t s=0;s<ns;s++) {
        double Te=plasma->T_e[s],ne=plasma->n_electron[s],z2ni=0.0;
        if(!isfinite(Te)||Te<=0.0||!isfinite(ne)||ne<0.0){a208_publication_free(&candidate);return 5;}
        for(int ip=0;ip<atom->n_ion_pops;ip++){
            double ni=atom->ion_number_density[(size_t)ip*ns+s];
            double z=(double)atom->ion_pop_stage[ip];
            if(isfinite(ni)&&ni>=0.0)z2ni+=z*z*ni;
        }
        for(size_t b=0;b<nb;b++) {
            size_t k=s*nb+b;double nu=sqrt(candidate.frequency_edges[b]*candidate.frequency_edges[b+1]);
            double es=ne*SIGMA_THOMSON;
            double x=H_PLANCK*nu/(K_BOLTZMANN*Te);
            double ff=C_FF_OPACITY/sqrt(Te)*ne*z2ni/(nu*nu*nu)*(-expm1(-x));
            double legacy=bf?bf->chi_bf[k]:ff;
            double bfnet=legacy-ff;
            double event_bf = (bf && bf->event_enabled && bf->event_chi_bf)
                            ? bf->event_chi_bf[k] : bfnet;
            if (!isfinite(event_bf) || event_bf < 0.0) {
                ctr->event_measure_unavailable++;
                a208_publication_free(&candidate); return 5;
            }
            candidate.chi_es[k]=es;candidate.chi_bf[k]=bfnet;candidate.chi_ff[k]=ff;
            candidate.bf_net_route[k]=bfnet;
            candidate.bf_event_measure[k]=event_bf;
            candidate.bf_route_validity[k]=event_bf==0.0?A208_EXACT_ZERO:A208_VALID;
            candidate.chi_validity[k]=(es==0.0)?A208_EXACT_ZERO:A208_VALID;
            candidate.chi_validity[2*ns*nb+k]=(bfnet==0.0)?A208_EXACT_ZERO:A208_VALID;
            candidate.chi_validity[3*ns*nb+k]=(ff==0.0)?A208_EXACT_ZERO:A208_VALID;
            ctr->es_terms++;ctr->bf_terms++;ctr->ff_terms++;
        }
    }
    for(int l=0;l<opacity->n_lines;l++) {
        double nu=opacity->line_list_nu?opacity->line_list_nu[l]:0.0;
        if(!(nu>candidate.frequency_edges[0]&&nu<candidate.frequency_edges[nb]))continue;
        size_t b=(size_t)(log(nu/nu_min)/dlog);if(b>=nb)b=nb-1;
        double dnu=candidate.frequency_edges[b+1]-candidate.frequency_edges[b];
        for(size_t s=0;s<ns;s++) {
            size_t lk=(size_t)l*ns+s,ck=s*nb+b;
            double tau=opacity->tau_sobolev[lk];
            A208Validity tv=opacity->tau_validity?opacity->tau_validity[lk]:
                (isfinite(tau)?(tau==0.0?A208_EXACT_ZERO:A208_VALID):A208_NONFINITE);
            if(tv==A208_VALID) {
                double frac=-expm1(-tau);
                double term=nu*frac/(C_SPEED_OF_LIGHT*epoch*dnu);
                if(!isfinite(term)){candidate.chi_validity[ns*nb+ck]=A208_NONFINITE;ctr->nonfinite_failures++;}
                else {candidate.chi_bb[ck]+=term;ctr->bb_terms++;if(term<0.0)ctr->negative_bb_line_shells++;}
            }
        }
    }
    for(size_t k=0;k<ns*nb;k++) {
        if(candidate.chi_validity[ns*nb+k]==0)
            candidate.chi_validity[ns*nb+k]=candidate.chi_bb[k]==0.0?A208_EXACT_ZERO:A208_VALID;
        candidate.chi_total[k]=((candidate.chi_es[k]+candidate.chi_bb[k])+candidate.chi_bf[k])+candidate.chi_ff[k];
        if(candidate.chi_bf[k]<0.0)ctr->negative_bf_shell_bins++;
        if(candidate.chi_total[k]<0.0)ctr->negative_total_shell_bins++;
    }
    if(a208_publication_commit(&opacity->cpu_opacity,&candidate)!=0){ctr->partial_publish_attempts++;a208_publication_free(&candidate);return 5;}
    ctr->shells_published+=ns;ctr->cells_published+=ns*nb;
    return 0;
}

int a209_publish_cpu_emissivity(OpacityState *opacity,const BFOpacity *bf,
 const AtomicData *atom,const PlasmaState *plasma,const NLTEConfig *nlte,double epoch){
 if(!opacity||!bf||!atom||!plasma||!nlte||!bf->eta_bf||
    opacity->cpu_opacity.generation_committed==0){a209_counters()->blocked_stale_opacity++;return 3;}
 size_t ns=(size_t)opacity->n_shells,nb=(size_t)bf->n_freq_bins,n=ns*nb;
 CpuEmissivityPublication c={0};if(a209_publication_init(&c,ns,nb))return 2;
 c.required_emissivity_generation=opacity->cpu_opacity.generation_committed;
 c.radfield_generation=nlte->radfield_view.generation;
 c.line_view_generation=nlte->line_view.generation;
 c.population_generation=atom->population_committed_generation;
 c.opacity_generation=opacity->cpu_opacity.generation_committed;
 c.te_generation=plasma->T_e_generation;
 A209Counters*ctr=a209_counters();ctr->generation_required=c.required_emissivity_generation;
 ctr->shells_attempted+=ns;ctr->cells_attempted+=n;
 if(!c.radfield_generation){ctr->blocked_stale_rf++;a209_publication_free(&c);return 3;}
 if(!c.line_view_generation){ctr->blocked_stale_line++;a209_publication_free(&c);return 3;}
 if(!c.population_generation){ctr->blocked_stale_pop++;a209_publication_free(&c);return 3;}
 memcpy(c.nu_edge,opacity->cpu_opacity.frequency_edges,(nb+1)*sizeof(double));
 for(size_t s=0;s<ns;s++){
  double Te=plasma->T_e[s];if(!isfinite(Te)||Te<=0){a209_publication_free(&c);return 5;}
  for(size_t b=0;b<nb;b++){size_t i=s*nb+b;double nu=sqrt(c.nu_edge[b]*c.nu_edge[b+1]);
   double B=planck_bnu(Te,nu),ff=opacity->cpu_opacity.chi_ff[i]*B;
   double bfeta=bf->eta_bf[i]-ff;
   if(!isfinite(ff)||ff<0||!isfinite(bfeta)||bfeta<0){ctr->nonfinite_failures++;a209_publication_free(&c);return 5;}
   c.eta_ff[i]=ff;c.eta_bf[i]=bfeta;c.component_status[2*n+i]=ff==0?EMISS_EXACT_ZERO:EMISS_OK;c.component_status[n+i]=bfeta==0?EMISS_EXACT_ZERO:EMISS_OK;ctr->ff_terms++;ctr->bf_terms++;
  }
 }
 /* Deposit each status-bearing line source on the same conservative bin used
  * by A2-08.  Signed chi and signed S multiply; no abs/clip/source fallback. */
 double dlog=log(c.nu_edge[nb]/c.nu_edge[0])/(double)nb;
 for(int l=0;l<opacity->n_lines;l++){double nu=opacity->line_list_nu[l];if(!(nu>c.nu_edge[0]&&nu<c.nu_edge[nb]))continue;size_t b=(size_t)(log(nu/c.nu_edge[0])/dlog);if(b>=nb)b=nb-1;double dnu=c.nu_edge[b+1]-c.nu_edge[b];for(size_t s=0;s<ns;s++){size_t lk=(size_t)l*ns+s,i=s*nb+b;A208Validity sv=opacity->line_source_validity?opacity->line_source_validity[lk]:A208_SOURCE_CANCELLATION_SINGULAR;if(sv!=A208_VALID&&sv!=A208_EXACT_ZERO){ctr->blocked_source++;continue;}double tau=opacity->tau_sobolev[lk];double chi=nu*(-expm1(-tau))/(C_SPEED_OF_LIGHT*epoch*dnu);double eta=chi*opacity->line_source_S[lk];if(!isfinite(eta)||eta<0){ctr->blocked_source++;continue;}c.eta_bb[i]+=eta;ctr->bb_terms++;}}
 for(size_t i=0;i<n;i++){c.component_status[i]=c.eta_bb[i]==0?EMISS_EXACT_ZERO:EMISS_OK;c.component_status[3*n+i]=EMISS_EXACT_ZERO;c.component_status[4*n+i]=EMISS_EXACT_ZERO;c.eta_true_total[i]=(c.eta_bb[i]+c.eta_bf[i])+c.eta_ff[i];c.eta_total_for_declared_semantics[i]=c.eta_true_total[i];c.cell_status[i]=c.eta_true_total[i]==0?EMISS_EXACT_ZERO:EMISS_OK;if(c.cell_status[i]==EMISS_EXACT_ZERO)ctr->exact_zero_terms++;}
 if(a209_build_reemit_cdf(&c,0x7)||a209_publication_commit(&opacity->cpu_emissivity,&c)){a209_publication_free(&c);return 5;}
 ctr->shells_published+=ns;ctr->cells_published+=n;return 0;
}

/* Sample Planck frequency using Bjorkman-Wood method (4-random) */
double sample_planck_frequency(double T, RNG *rng) {
    double kT_h = K_BOLTZMANN * T / H_PLANCK;
    double xi0 = rng_uniform(rng);
    double l_coef = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 90.0;
    double target = xi0 * l_coef;
    double cumsum = 0.0;
    double l_min = 1.0;
    for (int l = 1; l <= 1000; l++) {
        double ld = (double)l;
        double l_inv4 = 1.0 / (ld * ld * ld * ld);
        cumsum += l_inv4;
        if (cumsum >= target) {
            l_min = ld;
            break;
        }
    }
    double r1 = rng_uniform(rng);
    double r2 = rng_uniform(rng);
    double r3 = rng_uniform(rng);
    double r4 = rng_uniform(rng);
    if (r1 < 1e-300) r1 = 1e-300;
    if (r2 < 1e-300) r2 = 1e-300;
    if (r3 < 1e-300) r3 = 1e-300;
    if (r4 < 1e-300) r4 = 1e-300;
    double x = -log(r1 * r2 * r3 * r4) / l_min;
    return x * kT_h;
}

/* BF absorption event: A2-09 eta_reemit is the only frequency law. */
void bf_absorption_event(RPacket *pkt, double time_explosion,
                          PlasmaState *plasma, OpacityState *opacity,
                          RNG *rng) {
    /* 1. New isotropic direction */
    pkt->mu = rng_mu(rng);

    /* 2. One fixed-stream draw selects the piecewise-constant eta CDF. */
    (void)plasma;
    double comov_nu=0.0;
    if(a209_sample_reemit_frequency(&opacity->cpu_emissivity,
       (size_t)pkt->current_shell_id,
       opacity->cpu_emissivity.committed_emissivity_generation,
       rng_uniform(rng),&comov_nu)!=0){pkt->status=PACKET_REABSORBED;return;}

    /* 3. Transform to lab frame (inline Doppler to avoid lumina_transport.c dependency) */
    double beta = pkt->r / (C_SPEED_OF_LIGHT * time_explosion);
    double doppler = 1.0 - beta * pkt->mu;
    pkt->nu = comov_nu / doppler;  /* inv_doppler = 1/doppler */

    /* 4. Reinitialize next_line_id for new frequency (binary search) */
    double comov_check = pkt->nu * doppler;
    int lo = 0, hi = opacity->n_lines;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (opacity->line_list_nu[mid] > comov_check)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == opacity->n_lines) lo = opacity->n_lines - 1;
    pkt->next_line_id = lo;
}

/* ============================================================ */
/* NLTE: Full NLTE Rate Equation Solver                         */
/* Targets: Si,Ca,Fe,S,Co,Ni,C,Mg,Ti,Cr II/III (10 pairs)     */
/* ============================================================ */

/* NLTE target ion definitions: 16 element pairs over 31 slots
 * Original 10: Si,Ca,Fe,S,Co,Ni,C,Mg,Ti,Cr II/III
 * Added 4 (#207-era): Al,Sc,V,Mn II/III for 3000-3500Å UV line blanketing
 * Added 1 (#273): O II/III for O I 7773 triplet line-formation completeness
 * #281 refactor (#279 item 2): full O triplet (I,II,III) via overlap.
 *   Slot 28 = O I, Slot 29 = O II, Slot 30 = O III.
 *   Pair 14 = (slot 28, slot 29) = (O I, O II)  — populates O I 3s⁵S° → 3p⁵P (7774)
 *   Pair 15 = (slot 29, slot 30) = (O II, O III) — keeps O III NLTE (matches CMFGEN)
 *   Slot 29 is shared (single global→nlte map entry); sequential solve in outer iter
 *   loop converges {O I, O II, O III} as one block. */
static const int NLTE_TARGET_Z[]   = { 14, 14, 20, 20, 26, 26, 16, 16, 27, 27, 28, 28,
                                         6,  6, 12, 12, 22, 22, 24, 24,
                                        13, 13, 21, 21, 23, 23, 25, 25,
                                         8,  8,  8 };
static const int NLTE_TARGET_ION[] = {  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,
                                         1,  2,  1,  2,  1,  2,  1,  2,
                                         1,  2,  1,  2,  1,  2,  1,  2,
                                         0,  1,  2 };

/* Fork-A stage-IV promotion layout (LUMINA_NLTE_STAGE4=1): 38 slots. IV is
 * inserted IMMEDIATELY after each element's III so every (III,IV) pair is
 * contiguous (hi=lo+1) — the O I/II/III triplet's precedent — because the
 * overlap solve computes the pair size as super_offset[hi+1]-super_offset[lo]
 * (a contiguous span), so lo/hi MUST be adjacent slots. Only the 7 elements with
 * stage-IV levels+lines+IP(III->IV) get a IV slot (inventory:
 * Si,Fe,Co,Ni,Ti,Cr,Al). Gate OFF => NLTE_TARGET_Z/ION above are used verbatim. */
static const int NLTE_TARGET_Z4[]   = { 14, 14, 14, 20, 20, 26, 26, 26, 16, 16,
                                        27, 27, 27, 28, 28, 28,  6,  6, 12, 12,
                                        22, 22, 22, 24, 24, 24, 13, 13, 13,
                                        21, 21, 23, 23, 25, 25,  8,  8,  8 };
static const int NLTE_TARGET_ION4[] = {  1,  2,  3,  1,  2,  1,  2,  3,  1,  2,
                                         1,  2,  3,  1,  2,  3,  1,  2,  1,  2,
                                         1,  2,  3,  1,  2,  3,  1,  2,  3,
                                         1,  2,  1,  2,  1,  2,  0,  1,  2 };

/* Wave-3 CPU Stage-2A layout.  It is selected only after the strict EW gate
 * parser accepts an explicit Z list and shell.  Relative to the byte-stable
 * base layout it inserts Fe IV and S IV adjacent to their III stages; no other
 * legacy stage-IV promotion is pulled into the pilot. */
static const int NLTE_TARGET_Z_EW[NLTE_EW_IONS] = {
    14,14, 20,20, 26,26,26, 16,16,16, 27,27, 28,28, 6,6, 12,12,
    22,22, 24,24, 13,13, 21,21, 23,23, 25,25, 8,8,8
};
static const int NLTE_TARGET_ION_EW[NLTE_EW_IONS] = {
     1, 2,  1, 2,  1, 2, 3,  1, 2, 3,  1, 2,  1, 2, 1,2,  1, 2,
     1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2, 0,1,2
};

/* Heavy.2 / Task #139: Dielectronic recombination Burgess-form table.
 * Sources (provenance per entry):
 *   BADNELL: clist_K master fit (AUTOSTRUCTURE, Strathclyde, 2023-05-12).
 *   NORAD:   R-matrix unified RR+DR by Nahar et al. (Fe I-VI, Ni II).
 *   MAZZOTTA: Mazzotta+1998 LS-coupling fits (low-T-inaccurate floor).
 *   EST_ISOEL: isoelectronic interpolation placeholder (flagged).
 * Convention: ion_recomb = the recombining (upper) ion stage.
 * E.g. Fe III → Fe II is stored as {Z=26, ion_recomb=2}. */
static const DRCoefficient DR_TABLE[] = {
    /* --- Badnell (parsed from clist_K) ------------------------------ */
    /* Si III → Si II  (Z=14, N=12, Mg-like) */
    {14, 2, 5,
     {2.930e-06, 2.803e-06, 9.023e-05, 6.909e-03, 2.582e-05},
     {1.162e+02, 5.721e+03, 3.477e+04, 1.176e+05, 3.505e+06},
     DR_SOURCE_BADNELL},
    /* Si IV → Si III  (Z=14, N=11, Na-like) — outer-shell coolant restorer:
     * alpha_DR(2.5e4K)=4.6e-11 ~ 60x RR (offline_cell_balance validation) */
    {14, 3, 5,
     {3.819e-06, 2.421e-05, 2.283e-04, 8.604e-03, 2.617e-03},
     {3.802e+03, 1.280e+04, 5.953e+04, 1.026e+05, 1.154e+06},
     DR_SOURCE_BADNELL},
    /* Si V → Si IV    (Z=14, N=10, Ne-like; closed shell -> negligible <1e5K) */
    {14, 4, 3,
     {1.422e-04, 9.474e-03, 1.650e-03},
     {7.685e+05, 1.208e+06, 1.839e+06},
     DR_SOURCE_BADNELL},
    /* S IV → S III    (Z=16, N=13, Al-like) */
    {16, 3, 6,
     {5.817e-07, 1.391e-06, 1.123e-05, 1.521e-04, 1.875e-03, 2.097e-02},
     {3.628e+02, 1.058e+03, 7.160e+03, 3.260e+04, 1.235e+05, 2.070e+05},
     DR_SOURCE_BADNELL},
    /* S V → S IV      (Z=16, N=12, Mg-like) */
    {16, 4, 5,
     {9.571e-06, 6.268e-05, 3.807e-04, 1.874e-02, 5.526e-03},
     {1.180e+03, 6.443e+03, 2.264e+04, 1.530e+05, 3.564e+05},
     DR_SOURCE_BADNELL},
    /* Si II → Si I    (Z=14, N=13, Al-like) */
    {14, 1, 6,
     {3.408e-08, 1.913e-07, 1.679e-07, 7.523e-07, 8.386e-05, 4.083e-03},
     {2.431e+01, 1.293e+02, 4.272e+02, 3.729e+03, 5.514e+04, 1.295e+05},
     DR_SOURCE_BADNELL},
    /* S III → S II    (Z=16, N=14, Si-like) */
    {16, 2, 7,
     {3.040e-07, 4.393e-07, 1.609e-06, 4.980e-06, 3.457e-05, 8.617e-03, 9.284e-04},
     {5.016e+01, 3.266e+02, 3.102e+03, 1.210e+04, 4.969e+04, 2.010e+05, 2.575e+05},
     DR_SOURCE_BADNELL},
    /* S II → S I      (Z=16, N=15, P-like) */
    {16, 1, 7,
     {7.300e-08, 2.577e-07, 4.961e-08, 9.520e-07, 9.586e-07, 6.849e-04, 6.539e-04},
     {5.077e+02, 6.007e+02, 2.342e+03, 7.269e+03, 2.190e+04, 1.483e+05, 1.906e+05},
     DR_SOURCE_BADNELL},
    /* C III → C II    (Z=6,  N=4,  Be-like) */
    {6, 2, 6,
     {3.489e-06, 2.222e-07, 1.954e-05, 4.212e-03, 2.037e-04, 2.936e-04},
     {2.660e+03, 3.756e+03, 2.566e+04, 1.400e+05, 1.801e+06, 4.307e+06},
     DR_SOURCE_BADNELL},
    /* C II → C I      (Z=6,  N=5,  B-like) */
    {6, 1, 5,
     {6.346e-09, 9.793e-09, 1.634e-06, 8.369e-04, 3.355e-04},
     {1.217e+01, 7.380e+01, 1.523e+04, 1.207e+05, 2.144e+05},
     DR_SOURCE_BADNELL},
    /* Mg III → Mg II  (Z=12, N=10, Ne-like) */
    {12, 2, 3,
     {6.269e-06, 9.181e-04, 3.082e-04},
     {4.104e+05, 5.766e+05, 7.310e+05},
     DR_SOURCE_BADNELL},
    /* Mg II → Mg I    (Z=12, N=11, Na-like) */
    {12, 1, 4,
     {3.871e-08, 4.732e-07, 1.599e-03, 2.628e-05},
     {8.415e+03, 1.682e+04, 5.000e+04, 2.759e+05},
     DR_SOURCE_BADNELL},
    /* Ca III → Ca II  (Z=20, N=18, Ar-like) */
    {20, 2, 3,
     {3.843e-04, 8.040e-03, 8.670e-03},
     {2.282e+05, 3.682e+05, 4.479e+05},
     DR_SOURCE_BADNELL},

    /* --- NORAD (Nahar OSU R-matrix unified RR+DR total) -------------
     * Source files: data/atomic/dr_norad/{Fe1..Fe5,Si1,Si2}.csv (81-pt grid);
     * the Ni file (shipped as Ni1.csv, regenerated as Ni2.csv after the 2026-07-30
     * label fix) feeds the shadowed {28,2} NORAD entry further below.
     * Burgess fits at data/atomic/dr_norad/burgess_fits.txt (max err <1.1%
     * over 100 ≤ T ≤ 1e5 K). NOTE: these are TOTAL recomb (RR+DR), not
     * DR-only — slight double-counting with Milne RR (~10-20% at SN T_e),
     * acceptable floor for Phase 1 stabilization. Can be refined to
     * DR-only later by subtracting low-n RR column. */
    /* Fe II → Fe I  (Nahar Bautista Pradhan 1997 ApJ 479 497) */
    {26, 1, 5,
     {3.389049e-08, 1.406860e-07, 4.492892e-07, 1.494212e-06, 4.035439e-04},
     {1.080998e+02, 7.995730e+02, 2.676773e+03, 1.335320e+04, 5.680572e+04},
     DR_SOURCE_NORAD},
    /* Fe III → Fe II (Nahar 1997 PRA 55 1980) */
    {26, 2, 6,
     {7.553554e-08, 2.179776e-07, 9.813920e-07, 4.370443e-04, 7.175326e-06, 2.206982e-05},
     {9.418447e+01, 7.125228e+02, 3.095721e+03, 1.415009e+05, 1.190955e+04, 3.291757e+04},
     DR_SOURCE_NORAD},
    /* Fe IV → Fe III (Nahar 1996 PRA 53 2417) */
    {26, 3, 6,
     {2.762952e-07, 6.854108e-07, 1.112431e-05, 3.071916e-05, 2.100067e-06, 9.690480e-04},
     {9.502922e+01, 7.045133e+02, 1.554180e+04, 4.609481e+04, 3.392230e+03, 2.483458e+05},
     DR_SOURCE_NORAD},
    /* Fe V → Fe IV (Nahar Bautista Pradhan 1998 PRA 58 4593) */
    {26, 4, 6,
     {3.918497e-07, 1.262495e-06, 4.004534e-06, 2.069640e-05, 5.013299e-05, 1.753367e-03},
     {1.016419e+02, 7.656969e+02, 3.443625e+03, 1.488907e+04, 6.201225e+04, 2.933028e+05},
     DR_SOURCE_NORAD},
    /* Fe VI → Fe V (Nahar Bautista 1999 ApJS 120 327) */
    {26, 5, 6,
     {1.054419e-06, 2.771906e-06, 7.897993e-06, 3.215862e-05, 1.542221e-03, 9.613848e-05},
     {8.824877e+01, 7.638482e+02, 3.536893e+03, 1.466235e+04, 2.688904e+05, 6.446002e+04},
     DR_SOURCE_NORAD},
    /* (the NORAD Ni row was mislabeled "Ni II → Ni I" {28,1} here; the raw file is
     *  Ni III → Ni II — moved to the {28,2} block below, see there.) */
    /* Si II → Si I  (Z=14, Nahar 2000 ApJS 126 537 — Si-sequence R-matrix unified RR+DR)
     * Burgess 6-term fit, max rel err 0.96%, median 0.42% over T=100..1e5 K.
     * α(8000 K) ≈ 1.65e-12 cm³/s. */
    {14, 1, 6,
     {3.692246e-08, 1.246370e-07, 4.742018e-07, 2.341124e-06, 1.499678e-05, 7.501304e-04},
     {1.008423e+02, 7.150889e+02, 3.024003e+03, 1.108178e+04, 3.860763e+04, 1.118590e+05},
     DR_SOURCE_NORAD},
    /* Si III → Si II (Z=14, Nahar Pradhan Zhang 2000 ApJS 131 375 — R-matrix unified RR+DR)
     * Burgess 5-term fit, max rel err 1.64%, median 0.85% over T=100..1e5 K.
     * α(8000 K) ≈ 2.84e-12 cm³/s.  Closes Si II→III over-ionization at SN photosphere
     * (n_e·α ≈ 7e-3 s⁻¹ > R_bf_ground ≈ 5.9e-3 from logs/diag_ratebal_154431). */
    {14, 2, 5,
     {9.651958e-08, 3.283584e-07, 1.361371e-06, 6.897597e-03, 2.329208e-05},
     {1.094415e+02, 8.894031e+02, 4.719150e+03, 1.160976e+05, 2.613932e+04},
     DR_SOURCE_NORAD},

    /* --- Mazzotta+1998 LS-coupling floor for K-like+ iron-peak -------
     * Source files: data/atomic/dr_mazzotta/Z{Z}_ion{n}.txt
     * 17 from CHIANTI v11.0.2 drparams (Mazzotta1998); Ca from original CDS.
     * Burgess form: alpha_DR(T) = T^-1.5 * sum(c_i * exp(-E_i / T)).
     * Convention: file's "ion" is 1-indexed roman of the recombining ion;
     *   here ion_recomb = roman - 1 (0-indexed stage; II=1, III=2).
     * KNOWN LIMITATION: LS-coupling misses near-threshold resonances —
     *   underestimates low-T (T_e < 5e4 K) DR by 10×–10³× for Fe-peak.
     *   This is the "floor" — should be replaced by AUTOSTRUCT (Task #141)
     *   or Open-ADAS submissions when available.
     * Mazzotta entries duplicating Badnell (Ca III) or NORAD (Fe II–IV,
     *   Ni II) are skipped — keep the higher-quality data. */

    /* O II → O I  (Z=8, recombining = O II) — Mazzotta1998 single-term fit
     * (CDS table1.dat row "O II", converted eV→K: c_K = c_eV * 11604.519^1.5,
     *  E_K = E_eV * 11604.519). Added in #273 with O II/III NLTE pair for
     *  O I 7773 triplet line-formation completeness. Mazzotta LS-coupling
     *  floor; near-threshold resonances missing — refine later. */
    {8, 1, 1, {1.211336e-03}, {1.810305e+05}, DR_SOURCE_MAZZOTTA},
    /* O III → O II (Z=8, recombining = O III) — Mazzotta1998 single-term fit
     * (CDS table1.dat row "O III"). */
    {8, 2, 1, {4.775338e-03}, {2.115504e+05}, DR_SOURCE_MAZZOTTA},
    /* Ca II → Ca I  (Z=20, recombining = Ca II = stage 1) */
    {20, 1, 1, {1.987015e-04}, {3.585796e+04}, DR_SOURCE_MAZZOTTA},
    /* Sc II → Sc I  (Z=21, stage 1) */
    {21, 1, 1, {2.426200e-03}, {5.883500e+04}, DR_SOURCE_MAZZOTTA},
    /* Sc III → Sc II (Z=21, K-like recomb) — AUTOSTRUCTURE FULL-PHYSICS deck
     * MXCONF=5 (3d+4s+4p+4d+4f), MXCCF=15 captured Sc II configs, COREX='3-4'.
     * adf09 Burgess 6-term fit, gate max |rel err| 0.07%, transient 1.0% peak
     * (T~2000 K, where DR is small). α(8000 K)=1.14e-12 cm³/s ≈ 110× Mazzotta
     * peak-fit floor (and ~11× the MXCONF=2 pilot, confirming 4l channels dominate
     * at SN photospheric T). adf09: /home/kjhan/local/autostructure/runs/sc3_dr_full/ */
    {21, 2, 6,
     {2.3542e-07, 3.6318e-07, 4.7949e-11, 6.8761e-07, 1.9013e-05, 3.3273e-04},
     {8.4060e+02, 9.5042e+02, 1.2952e+03, 7.5144e+03, 5.8950e+04, 1.1377e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Sc III → Sc II (Z=21, stage 2) — Mazzotta1998 fallback (kept below AS entry,
     * never matched by dr_lookup since AS line above hits first). */
    {21, 2, 1, {1.309500e-02}, {1.151200e+05}, DR_SOURCE_MAZZOTTA},
    /* Ti II → Ti I  (Z=22, stage 1) */
    {22, 1, 1, {4.242400e-03}, {1.122200e+05}, DR_SOURCE_MAZZOTTA},
    /* Ti III → Ti II (Z=22, Sc-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d²+3d4s+3d4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit (rel err <0.5% at 4-100 kK). α(8000K)=1.81e-12 cm³/s.
     * adf09: /home/kjhan/local/autostructure/runs/ti3_dr_planC/ */
    {22, 2, 5,
     {4.8753e-08, 1.2221e-06, 1.8819e-06, 2.1847e-05, 1.1183e-03},
     {6.1884e+02, 3.2501e+03, 1.1777e+04, 6.8805e+04, 1.3899e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Ti III → Ti II (Z=22, stage 2) — Mazzotta1998 fallback (never matched). */
    {22, 2, 1, {2.351900e-02}, {1.972800e+05}, DR_SOURCE_MAZZOTTA},
    /* V II → V I    (Z=23, stage 1) */
    {23, 1, 1, {3.524100e-03}, {1.357700e+05}, DR_SOURCE_MAZZOTTA},
    /* V III → V II (Z=23, Ti-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d³+3d²4s+3d²4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit (rel err <0.6% at 4-100 kK). α(8000K)=3.96e-12 cm³/s.
     * adf09: /home/kjhan/local/autostructure/runs/v3_dr_planC/ */
    {23, 2, 5,
     {3.5549e-07, 1.7437e-06, 3.8422e-06, 1.6154e-05, 1.1303e-03},
     {1.3153e+02, 1.7990e+03, 1.0450e+04, 4.6119e+04, 1.8102e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* V III → V II (Z=23, stage 2) — Mazzotta1998 fallback (never matched). */
    {23, 2, 1, {1.436500e-02}, {1.659400e+05}, DR_SOURCE_MAZZOTTA},
    /* Cr II → Cr I  (Z=24, stage 1) */
    {24, 1, 1, {2.678200e-03}, {2.021500e+05}, DR_SOURCE_MAZZOTTA},
    /* Cr III → Cr II (Z=24, V-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁴+3d³4s+3d³4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit, gate max |rel err| 0.22% (T∈[4000,100000] K).
     * α(8000 K)=5.47e-12 cm³/s ≈ 10⁴× Mazzotta1998 (V-like 3d⁴ huge resonance density).
     * adf09: /home/kjhan/local/autostructure/runs/cr3_dr_planC/ */
    {24, 2, 5,
     {3.1657e-07, 1.4880e-06, 1.0498e-05, 2.7076e-05, 1.0159e-03},
     {2.5698e+03, 4.4490e+03, 1.0659e+04, 4.9445e+04, 2.2249e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Cr III → Cr II (Z=24, stage 2) — Mazzotta1998 fallback (never matched). */
    {24, 2, 1, {9.758900e-03}, {1.360000e+05}, DR_SOURCE_MAZZOTTA},
    /* Mn II → Mn I  (Z=25, stage 1) */
    {25, 1, 1, {1.121200e-03}, {1.329900e+05}, DR_SOURCE_MAZZOTTA},
    /* Mn III → Mn II (Z=25, Cr-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁵+3d⁴4s+3d⁴4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit (rel err <0.4% at 4-100 kK). α(8000K)=1.41e-12 cm³/s.
     * adf09: /home/kjhan/local/autostructure/runs/mn3_dr_planC/ */
    {25, 2, 5,
     {2.8136e-09, 6.0328e-08, 5.4823e-06, 1.5525e-05, 1.1715e-03},
     {9.3288e+02, 2.2496e+03, 1.4007e+04, 6.2004e+04, 2.6043e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Mn III → Mn II (Z=25, stage 2) — Mazzotta1998 fallback (never matched). */
    {25, 2, 1, {7.785800e-03}, {2.004100e+05}, DR_SOURCE_MAZZOTTA},
    /* Co II → Co I  (Z=27, stage 1) */
    {27, 1, 2,
     {2.855200e-04, 2.838900e-04},
     {5.779000e+04, 2.658600e+05},
     DR_SOURCE_MAZZOTTA},
    /* Co III → Co II (Z=27, Mn-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁷+3d⁶4s+3d⁶4p), MXCCF=6 (3d⁸+3d⁷4s+3d⁷4p+3d⁶4s²+3d⁶4s4p+3d⁶4p²),
     * NMAX=10, LMAX=5, COREX='3-4'. adf09 5-term fit, gate max |rel err| 0.25%
     * (T∈[4000,100000] K). α(8000 K)=6.42e-13 cm³/s ≈ 64× Mazzotta1998.
     * Co III pilot (MXCONF=2) gave only ~1× Mazzotta — Mn-like 7-valence 3d⁷
     * requires 4p-capture channel for SN photospheric T_e=8000 K.
     * adf09: /home/kjhan/local/autostructure/runs/co3_dr_planB/ */
    {27, 2, 5,
     {2.4215e-11, 2.9793e-07, 7.7205e-07, 1.2043e-05, 4.2601e-04},
     {1.5709e+02, 1.7237e+03, 1.0538e+04, 5.5318e+04, 2.6117e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Co III → Co II (Z=27, stage 2) — Mazzotta1998 fallback (kept below AS line,
     * never matched by dr_lookup since AS entry above hits first). */
    {27, 2, 1, {2.858100e-03}, {1.444800e+05}, DR_SOURCE_MAZZOTTA},
    /* Ni III → Ni II (Z=28, Co-like recomb) — AUTOSTRUCTURE plan-C deck
     * MXCONF=3 (3d⁸+3d⁷4s+3d⁷4p), MXCCF=6, NMAX=10, LMAX=5, COREX='3-4'.
     * adf09 5-term fit, gate max |rel err| 0.01% (T∈[4000,100000] K).
     * α(8000 K)=6.7e-13 cm³/s ≈ 10¹⁰× Mazzotta1998 floor (Mazzotta has E=2.6e5 K barrier).
     * adf09: /home/kjhan/local/autostructure/runs/ni3_dr_planC/ */
    {28, 2, 5,
     {1.6796e-07, 4.3130e-07, 3.3611e-06, 1.9387e-05, 4.0494e-04},
     {1.4518e+03, 6.9076e+03, 2.4518e+04, 8.3081e+04, 2.5458e+05},
     DR_SOURCE_AUTOSTRUCT},
    /* Ni III → Ni II (Z=28, stage 2) — Mazzotta1998 fallback (never matched). */
    {28, 2, 1, {9.215000e-03}, {2.604100e+05}, DR_SOURCE_MAZZOTTA},
    /* Ni III → Ni II (Z=28, stage 2) — NORAD unified RR+DR total, Nahar & Bautista
     * 2001 ApJS 137 201 (raw file data/atomic/dr_norad/raw_ni2.rrc.txt, header
     * "Process: Ni III + e -> Ni II"; α(1e4 K)=4.28e-12 cm³/s).
     * PROVENANCE / RELABEL (2026-07-30): these are the SAME six c_i/E_i that stood
     * above in the NORAD block as {28,1} "Ni II → Ni I". That label was wrong —
     * data/atomic/dr_norad/parse_norad.py had ion_recombining=1 for a file whose
     * process is Ni III + e → Ni II (generator fixed in the same batch; NORAD names
     * rrc files after the PRODUCT ion, so raw_ni2 ⇒ recombining stage 2). Only the
     * (Z,ion) label moved; the coefficients are untouched.
     * SHADOWED ON PURPOSE: kept BELOW the AUTOSTRUCT {28,2} entry, so dr_lookup
     * (first match wins) still returns AUTOSTRUCT — same convention as the Sc/Ti/Co
     * Mazzotta fallbacks above. Which of AUTOSTRUCT / NORAD / Mazzotta should own
     * Ni III→II is an adoption question, NOT settled here; this entry exists so the
     * NORAD datum is registered under its true label instead of a wrong one. */
    {28, 2, 6,
     {1.591064e-07, 6.798302e-07, 1.811202e-06, 4.724703e-04, 7.704725e-05, 5.987436e-06},
     {1.116045e+02, 7.047342e+02, 2.845402e+03, 1.435708e+05, 5.704062e+04, 1.181851e+04},
     DR_SOURCE_NORAD},

    /* --- CMFGEN LTDR (low-T dielectronic) from DIE* files ------------
     * Co IV → Co III  (Z=27, N=24, Cr-like recomb).  Derived from CMFGEN's
     * LTDR file  atomic/COB/III/19apr23/DIECoIII_14840  (14840 autoionizing→
     * bound stabilizing transitions, print-cutoff ALPHA(DIE)<1e-4·1e-12).
     * Per-transition α reconstructed with the CMFGEN reader formula
     * (rdgendie_v4.f:209-212):
     *   α_i(T)=2.07e-10·Gu·A/GION·exp(-HDKT·v(exc)/T4)/T4^1.5   [1e-12 cm³/s]
     *   HDKT=4.7994145,  GION=25 (Co IV 3d⁶ ⁵D ground term; self-calibrated
     *   from the a-columns to 25.000±5e-3 and confirmed by the phot-file
     *   header "Statistical weight of ion=25.0"),  T4=T/1e4.
     *   E_i[K]=47994.145·v(exc);  summed over all 14840 transitions, then
     *   compressed to this 3-term fit.  Builder: scripts/build_dr_cob3.py.
     * Reproduces CMFGEN's own summary total ("LTDR for all listed states"
     *   =1.359e-11 @1e4 K) to ratio 1.0000.  α(1e4)=1.30e-11, α(2e4)=6.40e-12.
     *   Fit max |rel err| 0.02% over 5e3-1e5 K (0.04% over 1e3-1.5e5 K).
     *   NB: file's cutoff-converged "Total for all states"=1.512e-11 @1e4 K is
     *   ~11% higher (below-print-cutoff tail, not reconstructable) — this entry
     *   is faithful to the LISTED transitions only, no ad-hoc scaling.
     * Cross file: DIECoIII_2590 (higher cutoff 1e-3) gives ~72% of 14840 at all
     *   T (14840/2590≈1.37-1.40); 14840 preferred.
     * VALIDATION: identical pipeline on C III (dieciii_ic.dat, GION=2 self-
     *   calibrated) vs Lumina's Badnell C III→C II (6,2): DIE/Badnell = 5.7×
     *   @1e4 K, 2.75× @2e4 K, 0.21× @5e4 K — LTDR-only exceeds near threshold
     *   and undershoots the Badnell total above ~2e4 K (expected: Badnell adds
     *   high-n/high-T DR the LTDR list omits).
     * DOUBLE-COUNT: Co III photoionization σ (phot_nosm & phot_data_A) is 100%
     *   smooth analytic (types 1/2/8/9: Seaton/hydrogenic/Verner-Yakovlev; ZERO
     *   type-20/21 Opacity-Project resonances) → Milne-inversion yields radiative
     *   recomb only, so this LTDR is fully COMPLEMENTARY (no double-count).
     *   CMFGEN's own guard (rd_phot_die_v1.f:131-147) warns of overlap only for
     *   OP type-20/21 σ, which Co III never uses.
     * STATUS: consulted only when LUMINA_FROZENIN_DR=1 (default OFF) — dormant
     *   bookkeeping/CMFGEN-comparison entry; changes no default behavior. */
    {27, 3, 3,
     {2.7336e-06, 9.9735e-06, 1.2109e-05},
     {6.1307e+02, 4.3869e+03, 9.7077e+03},
     DR_SOURCE_CMFGEN},

    /* --- AUTOSTRUCT self-compute: pending Task #141 ----------------- */
};
#define DR_N_ENTRIES ((int)(sizeof(DR_TABLE) / sizeof(DR_TABLE[0])))

/* Diagnostic: empirical multiplier on α_DR via env var LUMINA_DR_BOOST.
 * Used to test the LS-coupling-underestimate hypothesis (Mazzotta+1998
 * misses near-threshold autoionizing resonances; Fe-peak underestimated
 * by 10–10³× at SN T_e ~ 5e3–1e4 K). Default 1.0 (no scaling). Env var
 * accepts decimal floats, e.g. "10", "100", "31.6". Per-source masks
 * ("LUMINA_DR_BOOST_MAZZOTTA", "LUMINA_DR_BOOST_BADNELL" etc.) stack
 * multiplicatively if present. */
static double dr_boost_factor(DRSource src) {
    static int initialized = 0;
    static double boost_all = 1.0;
    static double boost_per_src[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    if (!initialized) {
        const char *all = getenv("LUMINA_DR_BOOST");
        if (all) boost_all = atof(all);
        const char *names[7] = {"NONE","BADNELL","NORAD","MAZZOTTA","AUTOSTRUCT",
                                "EST_ISOEL","CMFGEN"};
        for (int i = 1; i <= 6; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "LUMINA_DR_BOOST_%s", names[i]);
            const char *v = getenv(buf);
            if (v) boost_per_src[i] = atof(v);
        }
        initialized = 1;
    }
    double f = boost_all;
    if ((int)src >= 0 && (int)src < 7) f *= boost_per_src[(int)src];
    return f;
}

double dr_alpha_eval(const DRCoefficient *coef, double T_e) {
    if (!coef || coef->n_terms <= 0) return 0.0;
    if (T_e < 1.0) T_e = 1.0;
    double sum = 0.0;
    for (int i = 0; i < coef->n_terms; i++) {
        double arg = -coef->E_i[i] / T_e;
        if (arg < -700.0) continue;
        sum += coef->c_i[i] * exp(arg);
    }
    return sum * pow(T_e, -1.5) * dr_boost_factor(coef->source);
}

const DRCoefficient* dr_lookup(int Z, int ion_recomb) {
    for (int i = 0; i < DR_N_ENTRIES; i++) {
        if (DR_TABLE[i].Z == Z && DR_TABLE[i].ion_recomb == ion_recomb)
            return &DR_TABLE[i];
    }
    return NULL;
}

/* Step 1.5: Charge Exchange reaction table
 * Forward: A^(ion_A) + B^(ion_B) → A^(ion_A+1) + B^(ion_B-1)
 * Convention: A,B chosen so forward is exothermic (IP(A:1→2) < IP(B:1→2)).
 * Reverse via detailed balance: k_rev = k_fwd * exp(|ΔE|/kT)
 * Rate coefficients from Kingdon & Ferland (1996), generic 1e-9 cm³/s.
 *
 * Task #140 (Heavy.3) expansion (rows 7-17, 11 new) targets Gate B-4
 * Ni II 36× over-recombination by adding Ni-X (X=Cr,Mn,Ti,V) cross-coupling
 * channels — reverse direction (Ni+ + X2+ → Ni2+ + X+) acts as a Ni II
 * depletion pathway when Ni+ is over-saturated. Also adds Co-X (X=Cr,Mn,
 * Ti,V) and Fe-X (X=Mn,V,Sc) to round out the iron-peak network.
 * ΔE_eV computed from NIST II→III ionization potentials. */
static const ChargeExchangeReaction CE_REACTIONS[CE_N_REACTIONS] = {
  /* Z_A ion_A  Z_B ion_B  rate       alpha  ΔE_eV  */
  {  26,   1,    27,   2,   1.0e-9,    0.0,  -0.89 },  /* Fe+ + Co2+ → Fe2+ + Co+ */
  {  26,   1,    28,   2,   1.0e-9,    0.0,  -1.98 },  /* Fe+ + Ni2+ → Fe2+ + Ni+ */
  {  27,   1,    28,   2,   1.0e-9,    0.0,  -1.09 },  /* Co+ + Ni2+ → Co2+ + Ni+ */
  {  20,   1,    14,   2,   1.0e-9,    0.0,  -4.48 },  /* Ca+ + Si2+ → Ca2+ + Si+ */
  {  26,   1,    24,   2,   1.0e-9,    0.0,  -1.47 },  /* Fe+ + Cr2+ → Fe2+ + Cr+ */
  {  26,   1,    22,   2,   1.0e-9,    0.0,  -0.77 },  /* Fe+ + Ti2+ → Fe2+ + Ti+ */
  /* --- Heavy.3 expansion (Ni-coupling for Gate B-4 fix) --- */
  {  25,   1,    26,   2,   1.0e-9,    0.0,  -0.56 },  /* Mn+ + Fe2+ → Mn2+ + Fe+ */
  {  23,   1,    26,   2,   1.0e-9,    0.0,  -1.58 },  /* V+  + Fe2+ → V2+  + Fe+ */
  {  21,   1,    26,   2,   1.0e-9,    0.0,  -3.40 },  /* Sc+ + Fe2+ → Sc2+ + Fe+ */
  {  24,   1,    27,   2,   1.0e-9,    0.0,  -0.59 },  /* Cr+ + Co2+ → Cr2+ + Co+ */
  {  25,   1,    27,   2,   1.0e-9,    0.0,  -1.44 },  /* Mn+ + Co2+ → Mn2+ + Co+ */
  {  22,   1,    27,   2,   1.0e-9,    0.0,  -3.50 },  /* Ti+ + Co2+ → Ti2+ + Co+ */
  {  23,   1,    27,   2,   1.0e-9,    0.0,  -2.46 },  /* V+  + Co2+ → V2+  + Co+ */
  {  24,   1,    28,   2,   1.0e-9,    0.0,  -1.68 },  /* Cr+ + Ni2+ → Cr2+ + Ni+ */
  {  25,   1,    28,   2,   1.0e-9,    0.0,  -2.53 },  /* Mn+ + Ni2+ → Mn2+ + Ni+ */
  {  22,   1,    28,   2,   1.0e-9,    0.0,  -4.59 },  /* Ti+ + Ni2+ → Ti2+ + Ni+ */
  {  23,   1,    28,   2,   1.0e-9,    0.0,  -3.55 },  /* V+  + Ni2+ → V2+  + Ni+ */
};

/* Step 1.5: Get total ion number density for (Z, ion_stage, shell).
 * Uses NLTE level populations if available, otherwise nebular density. */
static double nlte_get_ion_density(NLTEConfig *nlte, AtomicData *atom,
                                    int Z, int ion_stage, int shell,
                                    int n_shells) {
    /* Check if this (Z, ion_stage) is an NLTE ion → sum level populations */
    if (nlte != NULL) {
        for (int i = 0; i < nlte->n_nlte_ions; i++) {
            if (nlte->nlte_Z[i] == Z && nlte->nlte_ion[i] == ion_stage) {
                int lev_s = nlte->nlte_ion_level_offset[i];
                int lev_e = nlte->nlte_ion_level_offset[i + 1];
                double sum = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    sum += nlte->nlte_level_populations[l * n_shells + shell];
                if (sum > 0.0) return sum;
                break;  /* found ion but no populations yet, fall through */
            }
        }
    }
    /* Fall back to nebular ion_number_density */
    int ip = find_ion_pop_idx(atom, Z, ion_stage);
    if (ip >= 0)
        return atom->ion_number_density[ip * n_shells + shell];
    return 0.0;
}

/* van Regemorter collision rate constant:
 * C_ij = 14.5 * a_0^2 * sqrt(2*pi*k_B/(m_e)) * n_e * f_ij / sqrt(T_e) * exp(-dE/kT)
 * Numerically: coeff = 14.5 * (5.29e-9)^2 * sqrt(2*pi*1.38e-16/9.11e-28)
 *            = 14.5 * 2.8e-17 * sqrt(9.53e11) = 14.5 * 2.8e-17 * 9.76e5 = 3.96e-10
 * We use the standard form: C_12 = 2.16e-6 * n_e * f_lu * exp(-dE/kT) / (g_1*sqrt(T_e)) * g_bar
 * where g_bar ~ 0.2 for allowed (van Regemorter), 1.0 for forbidden (Axelrod)
 */
#define VAN_REGEMORTER_COEFF  2.16e-6  /* effective Gaunt factor included */
#define AXELROD_OMEGA         1.0      /* collision strength for forbidden trans */

/* Mihalas-Lucy ion-lock gate (env-cached, iter-aware). */
int nlte_ion_lock_active(int current_iter) {
    static int init = 0;
    static int enabled = 0;
    static int start_iter = 0;
    if (!init) {
        const char *e1 = getenv("LUMINA_NLTE_ION_LOCK");
        const char *e2 = getenv("LUMINA_NLTE_LOCK_START_ITER");
        if (e1 && atoi(e1) != 0) enabled = 1;
        if (e2) start_iter = atoi(e2);
        init = 1;
    }
    return enabled && current_iter >= start_iter;
}

/* Per-ion rescale gate (env-cached). Decouples the post-solve per-ion rescale
 * path from LUMINA_NLTE_ION_LOCK, which also triggers freeze-plasma-transport-only
 * in the iter driver (cuda.cu / main.c). Set LUMINA_NLTE_PER_ION_RESCALE=1 to enable
 * only the rescale path at plasma.c:2941/2989 (and cuda.cu:458/520) — useful for
 * isolating the #223 combined-Σ collapse fix without freezing plasma. */
int nlte_per_ion_rescale_active(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_PER_ION_RESCALE");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Skip rate-matrix assembly for ion pairs whose total density has collapsed to
 * essentially zero (trace species in shells where the element abundance -> 0).
 * Assembly is 96.6% of NLTE wall time; a dead pair contributes ~0 populations
 * whether solved or Boltzmann-filled, so the costly per-level assembly is pure
 * waste. With the gate on, the assembly loop leaves the (already-zeroed) matrix
 * untouched -> the GPU getrf flags it singular -> the existing inline
 * Boltzmann@T_rad fallback fills the ~0 pops. Env-gated (default off) so the
 * gate-off path stays byte-identical for A/B verification. */
int nlte_skip_dead_pairs(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_SKIP_DEAD");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Fork-A stage-IV promotion gate (LUMINA_NLTE_STAGE4). Default OFF => the
 * baseline 31-slot/16-pair NLTE set is used verbatim => byte-identical baseline.
 * ON => append the (III,IV) pair for the 7 elements with stage-IV level+line+IP
 * data (Si,Fe,Co,Ni,Ti,Cr,Al), each IV inserted ADJACENT to its III so every
 * pair stays contiguous (hi=lo+1), matching the O I/II/III triplet's slot
 * adjacency that the overlap solve machinery requires. */
int nlte_stage4_enabled(void) {
    static int init = 0, enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_STAGE4");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Centralized (lo,hi) + names table for the NLTE ion pairs. Single source of
 * truth replacing the 4 hardcoded literals (GPU solve, CPU solve, max_N
 * precompute, R_bf GEMM). Fills the caller's arrays (>= NLTE_PAIR_COUNT) and
 * returns n_pairs. Gate OFF => the original 16-pair base layout (byte-identical);
 * gate ON => the 23-pair stage-IV adjacent layout. Every pair has hi=lo+1. */
int nlte_get_pairs(int pairs[][2], const char *names[]) {
    if (nlte_element_wide_layout_enabled()) {
        /* The two IV slots exist solely for the candidate assembler.  The
         * authority/fallback lane remains the original 16 physical pair calls
         * in the original order, translated through the EW slot permutation. */
        static const int P[NLTE_BASE_PAIRS][2] = {
            {0,1},{2,3},{4,5},{7,8},{10,11},{12,13},{14,15},{16,17},
            {18,19},{20,21},{22,23},{24,25},{26,27},{28,29},
            {30,31},{31,32}
        };
        static const char *N[NLTE_BASE_PAIRS] = {
            "Si","Ca","Fe","S","Co","Ni","C","Mg","Ti","Cr",
            "Al","Sc","V","Mn","O(I-II)","O(II-III)"
        };
        for (int p = 0; p < NLTE_BASE_PAIRS; p++) {
            pairs[p][0]=P[p][0]; pairs[p][1]=P[p][1]; names[p]=N[p];
        }
        return NLTE_BASE_PAIRS;
    } else if (nlte_stage4_enabled()) {
        /* slots: 0SiII 1SiIII 2SiIV 3CaII 4CaIII 5FeII 6FeIII 7FeIV 8SII 9SIII
         *   10CoII 11CoIII 12CoIV 13NiII 14NiIII 15NiIV 16CII 17CIII 18MgII
         *   19MgIII 20TiII 21TiIII 22TiIV 23CrII 24CrIII 25CrIV 26AlII 27AlIII
         *   28AlIV 29ScII 30ScIII 31VII 32VIII 33MnII 34MnIII 35OI 36OII 37OIII */
        static const int P[NLTE_STAGE4_PAIRS][2] = {
            {0,1},{1,2}, {3,4}, {5,6},{6,7}, {8,9}, {10,11},{11,12},
            {13,14},{14,15}, {16,17}, {18,19}, {20,21},{21,22},
            {23,24},{24,25}, {26,27},{27,28}, {29,30}, {31,32}, {33,34},
            {35,36},{36,37} };
        static const char *N[NLTE_STAGE4_PAIRS] = {
            "Si(II-III)","Si(III-IV)","Ca","Fe(II-III)","Fe(III-IV)","S",
            "Co(II-III)","Co(III-IV)","Ni(II-III)","Ni(III-IV)","C","Mg",
            "Ti(II-III)","Ti(III-IV)","Cr(II-III)","Cr(III-IV)",
            "Al(II-III)","Al(III-IV)","Sc","V","Mn","O(I-II)","O(II-III)" };
        for (int p = 0; p < NLTE_STAGE4_PAIRS; p++) {
            pairs[p][0] = P[p][0]; pairs[p][1] = P[p][1]; names[p] = N[p];
        }
        return NLTE_STAGE4_PAIRS;
    } else {
        static const int P[NLTE_BASE_PAIRS][2] = {
            {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},
            {16,17},{18,19},{20,21},{22,23},{24,25},{26,27},{28,29},{29,30} };
        static const char *N[NLTE_BASE_PAIRS] = {
            "Si","Ca","Fe","S","Co","Ni","C","Mg","Ti","Cr","Al","Sc","V","Mn",
            "O(I-II)","O(II-III)" };
        for (int p = 0; p < NLTE_BASE_PAIRS; p++) {
            pairs[p][0] = P[p][0]; pairs[p][1] = P[p][1]; names[p] = N[p];
        }
        return NLTE_BASE_PAIRS;
    }
}

/* Time-dependent ionization gate (env-cached). When set, nlte_assemble_rate_matrix
 * adds a backward-Euler dn_i/dt term to the rate rows using Dt = time_explosion
 * and a fully-ionized t=0 initial condition. This is the single-epoch (option A)
 * frozen-in approximation: ion stages whose net recombination rate << 1/Dt cannot
 * relax in the SN's age and stay over-ionized (the tau_rec >> t_exp outer layers),
 * while fast-recombining inner shells relax to the steady-state solution
 * (Dt->infinity recovers the steady-state matrix exactly). Default off. */
int nlte_timedep_active(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_TIMEDEP_ION");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* van Regemorter "trap" fix gate (root cause of the super-thermal S_l saga,
 * 2026-06-18). The legacy collisional bb branch (plasma.c) dispatches by the
 * raw radiative oscillator strength f_lu (threshold 1e-10), which mis-routes
 * forbidden M1/E2 lines carrying a tiny nonzero f_lu (~1e-9) into the van
 * Regemorter formula (C ∝ f_lu → ~0), starving the metastable<->ground
 * collisional coupling and leaving an O III-metastable rate-matrix null-space.
 * When set, dispatch by collision STRENGTH Upsilon = max(van Regemorter with the
 * proper Bethe (Ry/dE)^2 scaling, Omega=1 forbidden floor). Parameter-free. */
int nlte_coll_fix_enabled(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_COLL_FIX");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* COLD-CASE-P fix gate (LUMINA_NLTE_METASTABLE_COLL, default OFF => byte-identical).
 * A level flagged metastable=1 in levels.csv that ALSO has zero downward radiative
 * lines (drainless_metastable[global]==1, precomputed in nlte_init) has NO drain in
 * the baseline network: the per-line collision assembly loops over LINES only, so a
 * level with no line gets no collisional channel either. It fills by cascade and its
 * b_k piles up to the g_stage4_bk_cap ceiling, driving the photospheric IGE
 * over-ionization via its EUV photoionization. When ON, the assembler adds an
 * Axelrod-floor (Omega=1) collisional de-excitation channel from each such level to
 * its ion's GROUND level, restoring the missing drain (detailed balance exact). */
int nlte_metacoll_enabled(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_METASTABLE_COLL");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* Metastable-collision COUPLING MODE (LUMINA_NLTE_METACOLL_MODE).
 *   1 (default) = current behaviour: couple each drainless-metastable to its ion's
 *                 GROUND only, with the Axelrod forbidden floor Omega=1. Byte-identical
 *                 to kpr9 (same arithmetic, same off-diagonal placement).
 *   2           = couple each drainless-metastable to ALL lower levels of the same ion
 *                 (E_l < E_m), each with Omega = LUMINA_NLTE_METACOLL_OMEGA (default 0.1,
 *                 CMFGEN's f=0 forbidden floor). This matches CMFGEN's FeIII_COL_DATA
 *                 (Zhang 1996) topology, which drains every metastable to every lower
 *                 level rather than to ground alone. Any value != 2 falls back to 1. */
int nlte_metacoll_mode(void) {
    static int init = 0;
    static int mode = 1;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_METACOLL_MODE");
        if (e && atoi(e) == 2) mode = 2;
        init = 1;
    }
    return mode;
}

/* Per-channel collision strength Omega used by METACOLL_MODE=2 (LUMINA_NLTE_METACOLL_OMEGA).
 * Default 0.1 = CMFGEN's documented forbidden-transition floor ("Value for OMEGA if f=0:
 * 0.1", FeIII_COL_DATA header). Used per drainless-metastable -> lower-level channel as a
 * parameter-free approximation to the true Zhang96 per-transition Omega (imported later as
 * the fidelity endpoint). MODE=1 ignores this and uses AXELROD_OMEGA(=1) as before. */
double nlte_metacoll_omega(void) {
    static int init = 0;
    static double omega = 0.1;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_METACOLL_OMEGA");
        if (e) { double v = atof(e); if (v > 0.0) omega = v; }
        init = 1;
    }
    return omega;
}

/* Real Fe III collisional data gate (LUMINA_FEIII_COLDATA, default OFF =>
 * byte-identical). When ON (and atom->feiii_col_loaded), the NLTE assembler
 * drives ALL Fe III (Z=26, ion=2) collisional bound-bound rates from the
 * imported Zhang 1996 close-coupling Omega table (CMFGEN FeIII_COL_DATA)
 * instead of the van-Regemorter-from-oscillator-strength proxy + Axelrod floor
 * + METACOLL floor. This is the exact-physics fix for the over-populated Fe III
 * levels (25, 17, 18, 28, 31, 32): their radiative down-lines are forbidden
 * (f_lu ~ 1e-8) so van Regemorter gives ~0 collisional drain, while Zhang gives
 * real Omega ~ 1-9 to every lower level. To avoid double counting, when this
 * gate is on the per-line Fe III collision term is zeroed and the METACOLL
 * pass skips Fe III — Zhang is the sole Fe III collision source (CMFGEN
 * parity: collisions come from col_data, radiation from osc_data). */
int nlte_feiii_coldata_enabled(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *e = getenv("LUMINA_FEIII_COLDATA");
        if (e && atoi(e) != 0) enabled = 1;
        init = 1;
    }
    return enabled;
}

/* When the GPU bound-bound assembly path is active (LUMINA_NLTE_ASSEMBLE_GPU),
 * the driver sets this flag so the CPU assembler SKIPS the dominant per-line bb
 * radiative+collisional loop (the GPU kernel adds those contributions instead).
 * Everything else (bf/rec, CE, DR, conservation, top-stage, time-dep) stays on
 * the CPU. Default 0 -> the CPU does the full assembly (byte-identical). */
static int g_nlte_skip_bb = 0;
void nlte_assemble_set_skip_bb(int v) { g_nlte_skip_bb = v; }
static int nlte_assemble_skip_bb(void) { return g_nlte_skip_bb; }

/* Boltzmann-ceiling margin for the NLTE finite-garbage sanity gate.
 * A near-singular (but not exactly singular) rate matrix can yield a FINITE
 * solution whose excited-level pops sit 1e9-1e11x above the ion ground state —
 * far past the LTE ceiling x_i/x_0 <= g_i/g_0. Such solutions pass the
 * isfinite/info checks; the conservation rescale fixes only the sum, not the
 * inverted shape. The gate rejects a solve when any level exceeds
 * (g_i/g_ground) * margin, routing it to the Boltzmann@T_rad fallback.
 * Margin tunable via LUMINA_NLTE_INV_CEIL (default 1e4); <=0 disables the gate. */
double nlte_inv_ceiling(void) {
    static int init = 0;
    static double margin = 1e4;
    if (!init) {
        const char *e = getenv("LUMINA_NLTE_INV_CEIL");
        if (e) margin = atof(e);
        init = 1;
    }
    return margin;
}

/* ============================================================
 * Task #20: real radiative-equilibrium electron temperature.
 *
 * Replaces the parametric Compton + fudge-boost / adiabatic-only balance
 * (compute_electron_temperature self-consistent branch) with a genuine
 * per-shell heating = cooling solve, gated by LUMINA_RADEQ_TE=1.
 *
 *   Heating  H = photoionization + Compton + gamma-ray deposition
 *   Cooling  C = radiative recombination + free-free + collisional
 *                bound-bound (line) + adiabatic expansion
 *
 * Net(T_e) = H(T_e) - C(T_e) is monotone-decreasing in T_e, solved by
 * bisection on [0.1, 2.0]*T_rad. Operator-split: NLTE level populations
 * and J_nu are lagged (held at the previous iteration's values during the
 * T_e solve), matching CMFGEN's linearization. Method-faithfulness
 * (Phase-1), NOT a tuning knob — no free parameters.
 * ============================================================ */
typedef struct {
    int nlte_lo;      /* NLTE level index of lower level, or -1 if not NLTE-tracked */
    int nlte_up;      /* NLTE level index of upper level, or -1 if not NLTE-tracked */
    int lo_g, up_g;   /* GLOBAL level indices (for nebular pop / E / g / meta lookup) */
    int ip;           /* ion-population index (for n_ion / partition fn lookup) */
    int line;         /* line index (for tau_sobolev[line*n_shells+s] lookup) */
    double dE;        /* transition energy [erg] */
    double beta;      /* dE / k_B  [K]  (precomputed) */
    double coeff;     /* collisional rate coefficient: K*f_lu (permitted) or
                       * 8.63e-6*omega (forbidden); EXCLUDES n_e and 1/(g*sqrtTe) */
    double A_ul;      /* spontaneous emission rate [s^-1] (radiative-escape cooling) */
    int g_lo, g_up;   /* statistical weights */
} RadEqLine;

static RadEqLine *radeq_lines = NULL;
static long radeq_n_lines = -1;

/* Physical two-level destruction probability for one bb line at (n_e,T_e,tau):
 * eps = C_ul/(C_ul + A_ul*beta_esc). Exported for cmfgen_assemble's per-line
 * thermal-channel accumulation (LUMINA_CMFGEN_LINE_EPS_PHYS). Returns -1 when
 * the RADEQ line table is not built yet or the line is unknown (caller falls
 * back to the legacy fully-thermal treatment). */
double radeq_line_eps_phys(int line, double n_e, double T_e, double tau);

static double radeq_beta_esc(double tau);

double radeq_line_eps_phys(int line, double n_e, double T_e, double tau) {
    static int *line2k = NULL;
    static int  line2k_n = 0;
    if (!radeq_lines || radeq_n_lines <= 0 || line < 0 || T_e <= 0.0)
        return -1.0;
    if (!line2k) {
        int maxline = 0;
        for (long k = 0; k < radeq_n_lines; k++)
            if (radeq_lines[k].line > maxline) maxline = radeq_lines[k].line;
        int *m = (int *)malloc((size_t)(maxline + 1) * sizeof(int));
        if (!m) return -1.0;
        for (int i = 0; i <= maxline; i++) m[i] = -1;
        for (long k = 0; k < radeq_n_lines; k++) m[radeq_lines[k].line] = (int)k;
        line2k_n = maxline + 1;
        line2k = m;
    }
    if (line >= line2k_n) return -1.0;
    int k = line2k[line];
    if (k < 0) return -1.0;
    const RadEqLine *rl = &radeq_lines[k];
    double C_ul = n_e * rl->coeff / ((double)rl->g_up * sqrt(T_e));
    double denom = C_ul + rl->A_ul * radeq_beta_esc(tau);
    return (denom > 0.0) ? C_ul / denom : 1.0;
}

int radeq_line_local_response(int line, double n_e, double T_e, double tau,
                              double *beta_out, double *eps0_out) {
    static int *line2k = NULL;
    static int line2k_n = 0;
    if (!beta_out || !eps0_out || !radeq_lines || radeq_n_lines <= 0 ||
        line < 0 || !(n_e >= 0.0) || !(T_e > 0.0) || !(tau > 0.0) ||
        !isfinite(n_e) || !isfinite(T_e) || !isfinite(tau))
        return -1;
    if (!line2k) {
        int maxline = -1;
        for (long k = 0; k < radeq_n_lines; k++)
            if (radeq_lines[k].line > maxline) maxline = radeq_lines[k].line;
        if (maxline < 0) return -1;
        int *m = (int *)malloc((size_t)(maxline + 1) * sizeof(int));
        if (!m) return -1;
        for (int i = 0; i <= maxline; i++) m[i] = -1;
        for (long k = 0; k < radeq_n_lines; k++) {
            int l = radeq_lines[k].line;
            if (l < 0 || l > maxline || m[l] != -1) { free(m); return -1; }
            m[l] = (int)k;
        }
        line2k = m;
        line2k_n = maxline + 1;
    }
    if (line >= line2k_n || line2k[line] < 0) return -1;
    const RadEqLine *rl = &radeq_lines[line2k[line]];
    if (!(rl->g_up > 0) || !(rl->coeff >= 0.0) || !(rl->A_ul >= 0.0) ||
        !isfinite(rl->coeff) || !isfinite(rl->A_ul)) return -1;
    double beta = -expm1(-tau) / tau;
    if (!(beta > 0.0) || !(beta <= 1.0) || !isfinite(beta)) return -1;

    /* Evaluate C/(C+A) without ever forming C or C+A.  Both can overflow
     * although the ratio is defined.  The log-domain logistic is exact at
     * the representable endpoints and introduces no substitute value. */
    double eps0;
    if (n_e == 0.0 || rl->coeff == 0.0) {
        if (rl->A_ul == 0.0) return -1;       /* 0/(0+0) is undefined */
        eps0 = 0.0;
    } else if (rl->A_ul == 0.0) {
        eps0 = 1.0;
    } else {
        double logC = log(n_e) + log(rl->coeff) - log((double)rl->g_up)
                    - 0.5 * log(T_e);
        double x = log(rl->A_ul) - logC;
        if (!isfinite(logC) || !isfinite(x)) return -1;
        if (x >= 0.0) {
            double r = exp(-x);
            eps0 = r / (1.0 + r);
        } else {
            double r = exp(x);
            eps0 = 1.0 / (1.0 + r);
        }
    }
    if (!(eps0 >= 0.0) || !(eps0 <= 1.0) || !isfinite(eps0)) return -1;
    *beta_out = beta;
    *eps0_out = eps0;
    return 0;
}

static void build_radeq_line_table(NLTEConfig *nlte, AtomicData *atom,
                                   OpacityState *opacity) {
    if (radeq_n_lines >= 0) return;  /* already built */
    int n_lines = opacity->n_lines;
    long count = 0;
    /* van-Regemorter Gaunt factor for PERMITTED lines. The NLTE rate matrix uses
     * 0.2 for permitted bb collisions; the cooling table historically used 1.0.
     * Default 1.0 = byte-identical; LUMINA_RADEQ_VR_GAUNT=0.2 reconciles the
     * cooling-rate magnitude (and ETLA SE branching) with the population solver. */
    double vr_gaunt = 1.0;
    { const char *vg = getenv("LUMINA_RADEQ_VR_GAUNT"); if (vg) vr_gaunt = atof(vg); }
    /* Build the cooling table from ALL valid bb lines (not just NLTE-tracked),
     * since collisionally-excited line cooling is the dominant outer-ejecta
     * coolant and is carried by the full line census, not the few NLTE levels.
     * Per-level populations are resolved at solve time: NLTE pop where the level
     * is tracked, dilute-Boltzmann (nebular) otherwise. */
    /* ------------------------------------------------------------------ *
     * dig_C5-A/B (ARTIS parity ONLY): real close-coupling Upsilon + Omega
     * floor for the radeq ETLA collisional line-cooling coeff. Both are
     * confined to the artis_parity_enabled() branch below; when the master
     * gate is OFF none of this runs and the table is byte-identical.
     *
     * A (real-Upsilon, ON by default under parity — exact physics, the A6
     * completion): for every (Z,ion,lo,hi) covered by an ALREADY-LOADED close-
     * coupling table (feiii_col_* Zhang / col_ion_* Fe II,Co III,Ni III) bake
     * coeff = ARTIS_COL_CONST*Upsilon(T_ref). The eval C_ul =
     * n_e*coeff/(g_up*sqrt(Te)) then reproduces artis_col_rates' real branch
     * C_down = n_e*ARTIS_COL_CONST*Upsilon/(g_up*sqrt(Te)) and the NLTE-matrix
     * col_data passes EXACTLY (A6 unification). T_ref=10400K matches how the
     * whole table is baked T-independent (consumed in the per-trial-Te root
     * solve). Upsilon is interpolated linearly in T over the tabulated grid,
     * clamped to the ends — identical to the NLTE-matrix passes (Fe III / col_ion).
     *
     * B (Omega floor, OPT-IN under parity): coeff = max(coeff,
     * ARTIS_COL_CONST*om_floor) when LUMINA_RADEQ_OMEGA_FLOOR is explicitly
     * set >0. Mirrors the EXACT non-parity vr_std floor semantics below (a flat
     * coeff floor, NO degeneracy factor). Unset/0 => byte-identical parity
     * (preserves parity10-15 rerun comparability). Matches dig_C5's reference
     * coeff_variant() 'realfloor': A then max(., ARTIS_COL_CONST*om_floor). */
    const double rc_Tref = 10400.0;
    long realsub_count = 0;
    double p_om_floor = -1.0;   /* -1 => floor disabled (byte-identical) */
    struct { int Z, ion0, nlev; double *cf; } rcs[1 + LUMINA_MAX_COL_IONS];
    int n_rcs = 0;
    /* [OMEGA-CMFGEN] the orthodox replacement of dig_C5-A/B: per-transition
     * CMFGEN 3-tier (table as-is | vR gbar-floor | OMEGA_SET) instead of
     * "real-Upsilon then clamp everything to Upsilon>=om_floor". Mutually
     * exclusive with the floor; the floor loses. */
    const int omcm_on = omega_cmfgen_enabled() && artis_parity_enabled();
    long omcm_cnt[4] = {0, 0, 0, 0};
    if (artis_parity_enabled()) {
        const char *of = getenv("LUMINA_RADEQ_OMEGA_FLOOR");
        if (of) { double v = atof(of); if (v > 0.0) p_om_floor = v; }
        if (omcm_on && p_om_floor > 0.0) {
            fprintf(stderr, "[OMEGA-CMFGEN][WARN] LUMINA_RADEQ_OMEGA_FLOOR=%g is "
                            "mutually exclusive with LUMINA_OMEGA_CMFGEN=1 -> the "
                            "floor is IGNORED (3-tier wins).\n", p_om_floor);
            p_om_floor = -1.0;
        }
    }
    if (!omcm_on && artis_parity_enabled()) {
        /* Build per-source dense coeff(=ARTIS_COL_CONST*Upsilon@Tref) maps. A
         * source is (Fe III Zhang, src==-1) plus each generic col_ion_* ion. The
         * dense [lo*nlev+hi] layout (lo<hi by level_number == energy rank) gives
         * O(1) per-line lookup; -1 marks an uncovered pair. */
        for (int src = -1; src < atom->ncol_ions; src++) {
            int Zs, ion0s, ntr, ntemp;
            const double *tg; const int *tlo; const int *thi; const double *tom;
            if (src < 0) {
                if (!atom->feiii_col_loaded) continue;
                Zs = atom->feiii_col_Z; ion0s = atom->feiii_col_ion;
                ntr = atom->feiii_col_n_trans; ntemp = atom->feiii_col_n_temp;
                tg = atom->feiii_col_tgrid; tlo = atom->feiii_col_lo;
                thi = atom->feiii_col_hi; tom = atom->feiii_col_omega;
            } else {
                Zs = atom->col_ion_Z[src]; ion0s = atom->col_ion_stage[src];
                ntr = atom->col_ion_n_trans[src]; ntemp = atom->col_ion_n_temp[src];
                tg = atom->col_ion_tgrid[src]; tlo = atom->col_ion_lo[src];
                thi = atom->col_ion_hi[src]; tom = atom->col_ion_omega[src];
            }
            if (ntr <= 0 || ntemp <= 0 || !tg || !tlo || !thi || !tom) continue;
            /* interpolation weights at T_ref (same clamp as the NLTE passes) */
            int ti = 0; while (ti < ntemp - 2 && rc_Tref > tg[ti + 1]) ti++;
            double frac_t = 0.0, denom = tg[ti + 1] - tg[ti];
            if (denom > 0.0) frac_t = (rc_Tref - tg[ti]) / denom;
            if (frac_t < 0.0) frac_t = 0.0;
            if (frac_t > 1.0) frac_t = 1.0;
            int maxlev = 0;
            for (int t = 0; t < ntr; t++) {
                if (tlo[t] > maxlev) maxlev = tlo[t];
                if (thi[t] > maxlev) maxlev = thi[t];
            }
            int nlev = maxlev + 1;
            if (nlev <= 0) continue;
            double *cf = (double *)malloc((size_t)nlev * (size_t)nlev * sizeof(double));
            if (!cf) continue;
            for (size_t i = 0; i < (size_t)nlev * (size_t)nlev; i++) cf[i] = -1.0;
            for (int t = 0; t < ntr; t++) {
                int lo = tlo[t], hi = thi[t];
                if (lo < 0 || hi < 0 || lo >= nlev || hi >= nlev) continue;
                const double *om = &tom[(size_t)t * (size_t)ntemp];
                double ups = om[ti] + frac_t * (om[ti + 1] - om[ti]);
                if (!(ups > 0.0)) continue;
                int a = lo < hi ? lo : hi, b = lo < hi ? hi : lo;
                cf[(size_t)a * (size_t)nlev + (size_t)b] = ARTIS_COL_CONST * ups;
            }
            rcs[n_rcs].Z = Zs; rcs[n_rcs].ion0 = ion0s;
            rcs[n_rcs].nlev = nlev; rcs[n_rcs].cf = cf;
            n_rcs++;
        }
    }
    for (int pass = 0; pass < 2; pass++) {
        long k = 0;
        for (int line = 0; line < n_lines; line++) {
            int ion_s = atom->line_ion_number[line];
            int ip = find_ion_pop_idx(atom, atom->line_atomic_number[line], ion_s);
            if (ip < 0) continue;
            int lev_base = atom->level_offset[ip];
            int lev_top  = atom->level_offset[ip + 1];
            int lo_g = -1, up_g = -1;
            for (int l = lev_base; l < lev_top; l++) {
                if (atom->level_num[l] == atom->line_level_lower[line]) lo_g = l;
                if (atom->level_num[l] == atom->line_level_upper[line]) up_g = l;
                if (lo_g >= 0 && up_g >= 0) break;
            }
            if (lo_g < 0 || up_g < 0) continue;
            double dE = fabs(atom->level_energy_eV[up_g] -
                             atom->level_energy_eV[lo_g]) * EV_TO_ERG;
            if (dE <= 0.0) continue;
            if (pass == 1) {
                double f_lu = atom->line_f_lu[line];
                radeq_lines[k].nlte_lo = nlte->global_to_nlte_level[lo_g];
                radeq_lines[k].nlte_up = nlte->global_to_nlte_level[up_g];
                radeq_lines[k].lo_g = lo_g;
                radeq_lines[k].up_g = up_g;
                radeq_lines[k].ip   = ip;
                radeq_lines[k].line = line;
                radeq_lines[k].dE   = dE;
                radeq_lines[k].beta = dE / K_BOLTZMANN;
                /* LUMINA_RADEQ_VR_STD=1: STANDARD van Regemorter normalization
                 * Omega = 14.5*gbar*f*g_lo*(Ry/dE) (Allen/vR 1962; the physics-
                 * agent closure Lambda(25690K)=3.1e-11=H_dep closed with THIS
                 * form). The legacy 2.16e-6*f is 15x (UV) to 80x (optical) LOW
                 * — the 'Bethe (Ry/dE) missing' note from the vR-trap fix was
                 * never completed. gbar = LUMINA_RADEQ_VR_GAUNT (default 0.2). */
                static int vr_std = -1;
                if (vr_std < 0) { const char *vs = getenv("LUMINA_RADEQ_VR_STD");
                                  vr_std = (vs && atoi(vs)) ? 1 : 0; }
                if (omcm_on) {
                    /* [OMEGA-CMFGEN] one per-transition CMFGEN Omega, baked at
                     * the same T_ref the whole table uses. coeff = 8.629e-6*Omega
                     * keeps the eval-time C_ul = n_e*coeff/(g_up*sqrt(Te)) in the
                     * Osterbrock convention, identical to the tier-1 (real table)
                     * and NLTE col_data passes. NO floor is applied anywhere. */
                    int tier = 3;
                    double ups = omega_cmfgen_line(atom, line, rc_Tref, &tier);
                    radeq_lines[k].coeff = ARTIS_COL_CONST * ups;
                    omcm_cnt[tier]++;
                    if (tier == 1) realsub_count++;
                } else
                if (artis_parity_enabled()) {
                    /* A6: ONE collision form. Bake the effective coeff so the
                     * eval-time C_ul = n_e*coeff/(g_up*sqrt(Te)) equals the shared
                     * ARTIS C_down (deexcitation): permitted = vR + Bethe
                     * (H_ionpot/dE)^2; forbidden = g-scaled Axelrod (0.01*g_lo*g_up).
                     * The energy-dependent Gaunt is frozen at its g_bar=0.2 floor
                     * because this table is T-independent (consumed in the per-trial
                     * -Te root solve); the NLTE matrix + k-packet use the exact
                     * eval-time Gaunt via artis_col_rates(). */
                    int g_lo_r = atom->level_g[lo_g];
                    int g_up_r = atom->level_g[up_g];
                    if (f_lu > 1e-10) {
                        double ry = ARTIS_H_IONPOT_ERG / dE;
                        radeq_lines[k].coeff = ARTIS_C_0 * ARTIS_VR_PREF * f_lu *
                            ry * ry * (double)g_lo_r * ARTIS_GBAR * (dE / K_BOLTZMANN);
                    } else {
                        radeq_lines[k].coeff = ARTIS_COL_CONST * ARTIS_FORB_UPS *
                            (double)g_lo_r * (double)g_up_r;
                    }
                    /* dig_C5-A: real close-coupling Upsilon override (exact,
                     * default ON). Covered (Z,ion,lo,hi) replace the vR/Axelrod
                     * proxy with coeff = ARTIS_COL_CONST*Upsilon(T_ref). Level
                     * pair normalised min/max on level_number (== energy rank),
                     * matching dig_C5's reference lookup. */
                    if (n_rcs > 0) {
                        int Zl = atom->line_atomic_number[line];
                        for (int rs = 0; rs < n_rcs; rs++) {
                            if (rcs[rs].Z != Zl || rcs[rs].ion0 != ion_s) continue;
                            int a = atom->level_num[lo_g];
                            int b = atom->level_num[up_g];
                            if (a > b) { int tmp = a; a = b; b = tmp; }
                            if (a >= 0 && b < rcs[rs].nlev) {
                                double cf = rcs[rs].cf[(size_t)a * (size_t)rcs[rs].nlev + (size_t)b];
                                if (cf >= 0.0) {
                                    radeq_lines[k].coeff = cf;
                                    realsub_count++;
                                }
                            }
                            break;
                        }
                    }
                    /* dig_C5-B: Omega floor (opt-in). Flat coeff floor mirroring
                     * the non-parity vr_std semantics (no g factor); applied AFTER
                     * A so real Upsilon<om_floor is raised too (reference 'realfloor'). */
                    if (p_om_floor > 0.0) {
                        double c_min = ARTIS_COL_CONST * p_om_floor;
                        if (radeq_lines[k].coeff < c_min) radeq_lines[k].coeff = c_min;
                    }
                } else
                if (vr_std && f_lu > 1e-10) {
                    double gbar = (vr_gaunt != 1.0) ? vr_gaunt : 0.2;
                    double ry_de = 13.605693 / (radeq_lines[k].dE / EV_TO_ERG);
                    if (ry_de > 136.0) ry_de = 136.0;   /* dE>=0.1 eV validity cap */
                    double c_vr = 8.63e-6 * 14.5 * gbar * f_lu *
                                  (double)atom->level_g[lo_g] * ry_de;
                    /* Upsilon = max(vR, Omega_floor): the June vR-trap design's
                     * second half, never implemented. Semi-forbidden/forbidden
                     * lines carry f~1e-9..1e-4 -> f-scaled vR Omega is 1e2-1e6
                     * LOW (true Omega([S III]9069, [Ca II]7291..) ~ O(1)).
                     * These low-dE lines are THE classic nebular valley coolants
                     * (calculator: valley +35-40% hot without them) — and the
                     * valley overheat opens the 35eV bf window that feeds the
                     * regional strip attractor. Floor is harmless for permitted
                     * lines (their Omega >> 1) and for high-dE lines (exp cut).
                     * Gate LUMINA_RADEQ_OMEGA_FLOOR (default 1.0; 0 disables). */
                    static double om_floor = -1.0;
                    if (om_floor < 0.0) { const char *of = getenv("LUMINA_RADEQ_OMEGA_FLOOR");
                                          om_floor = of ? atof(of) : 1.0; }
                    double c_min = 8.63e-6 * om_floor;
                    radeq_lines[k].coeff = (c_vr > c_min) ? c_vr : c_min;
                } else
                radeq_lines[k].coeff = (f_lu > 1e-10) ?
                    VAN_REGEMORTER_COEFF * f_lu * vr_gaunt : 8.63e-6 * AXELROD_OMEGA;
                radeq_lines[k].A_ul = atom->line_A_ul ? atom->line_A_ul[line] : 0.0;
                radeq_lines[k].g_lo = atom->level_g[lo_g];
                radeq_lines[k].g_up = atom->level_g[up_g];
            }
            k++;
        }
        if (pass == 0) {
            count = k;
            radeq_lines = (RadEqLine *)malloc((size_t)(count > 0 ? count : 1) *
                                              sizeof(RadEqLine));
        }
    }
    radeq_n_lines = count;
    printf("  [RADEQ] collisional line-cooling table: %ld bb transitions (all ions)\n",
           radeq_n_lines);
    if (omcm_on) {
        printf("  [OMEGA-CMFGEN] radeq ETLA coeff = 8.629e-6*Omega_CMFGEN(T=%.0fK), "
               "NO floor: %ld tabulated | %ld vR(gbar>=0.2) | %ld OMEGA_SET=%g  "
               "(of %ld lines)\n", rc_Tref, omcm_cnt[1], omcm_cnt[2], omcm_cnt[3],
               omcm_oset(), radeq_n_lines);
    } else
    if (artis_parity_enabled()) {
        printf("  [ARTIS-PARITY REAL-UPSILON] radeq ETLA: real close-coupling Upsilon"
               " baked (T=%.0fK) for %ld lines (%d ion tables)\n",
               rc_Tref, realsub_count, n_rcs);
        if (p_om_floor > 0.0)
            printf("  [ARTIS-PARITY OMEGA-FLOOR] radeq ETLA coeff floored at Upsilon>=%g"
                   " (+real-Upsilon for %ld lines)\n", p_om_floor, realsub_count);
    }
    for (int rs = 0; rs < n_rcs; rs++) free(rcs[rs].cf);
    {   /* DIAG: how many lines have BOTH levels NLTE-tracked (the COOL_NLTE_ONLY=1
         * cooling set). If ~0, the faithful coolant is empty because
         * global_to_nlte_level returns -1 for (super-level-folded) levels. */
        long n_lo = 0, n_up = 0, n_both = 0;
        for (long kk = 0; kk < radeq_n_lines; kk++) {
            if (radeq_lines[kk].nlte_lo >= 0) n_lo++;
            if (radeq_lines[kk].nlte_up >= 0) n_up++;
            if (radeq_lines[kk].nlte_lo >= 0 && radeq_lines[kk].nlte_up >= 0) n_both++;
        }
        printf("  [RADEQ-NLTETRACK] lo>=0:%ld up>=0:%ld BOTH:%ld of %ld (%.3f%% both = faithful coolant)\n",
               n_lo, n_up, n_both, radeq_n_lines,
               100.0 * (double)n_both / (double)(radeq_n_lines > 0 ? radeq_n_lines : 1));
    }
}

#ifdef LUMINA_FROZEN_ORACLE
void lumina_oracle_prepare_line_eps(NLTEConfig *nlte, AtomicData *atom,
                                    OpacityState *opacity) {
    build_radeq_line_table(nlte, atom, opacity);
}
#endif

/* Net energy rate H(T_e) - C(T_e) [erg/s/cm^3] for one shell. The lagged,
 * T_e-independent heating (H_photo + H_gamma) and the per-line cooling
 * coefficients (a/b/beta over the active line set) are precomputed by the
 * caller; everything T_e-dependent is evaluated here so the bisection
 * re-evaluates cheaply. */
/* Bound-free radiative cooling, the EMISSION half of the detailed-balance pair.
 * Built (per shell) from the SAME σ_bf, f_above weight, AND the SAME lagged level
 * population n_lev used by the photoheating integral, but with the radiation field
 * replaced by the local Wien Planck function B_ν(T_e)=(2hν³/c²)e^{−hν/kT_e}:
 *   C_emit(T_e) = Σ_ν [ Σ_l n_lev · 4π (2hν³/c²) σ_bf f_above dν ] · e^{−hν/kT_e}
 *               = Σ_ν emit_nu[ν] · e^{−hν/kT_e}
 * The bracketed sum is T_e-independent and accumulated into emit_nu[] alongside
 * H_photo. Because heating uses n_lev·J_ν and cooling uses n_lev·B_ν over the
 * identical ν-grid, the bound-free net = n_lev ∫4π(J_ν−B_ν)σ_bf f_above dν cancels
 * BIN-BY-BIN at LTE (J_ν=B_ν) and, crucially, the cooling carries n_lev too — so a
 * spiking departure coefficient b_l=n_lev/n*_l grows BOTH terms together, bounding
 * the net by (J_ν−B_ν) and restoring the outer-shell thermostat that the old
 * n*_l-weighted (Saha) cooling lacked. */
/* ---- LUMINA_RADEQ_FB_RATE=1: rate-based free-bound cooling ----
 * The emit_nu-based radeq_recomb_cool integrates NLTE LEVEL populations; at
 * the thin outer the ill-conditioned NLTE pops overpopulate high levels and
 * C_rec(T_trial) explodes ~1e7-1e12 above deposition (fix2 RTRUTH s=49:
 * 2.6e-5 @25kK vs H=3.1e-11), killing every hot root. Replace with the
 * ARTIS-kpkt / offline-calculator form, consistent with the ionization
 * channel (same alpha = Milne RR + Badnell DR):
 *   C_fb = sum_pairs n_e * n_{j+1} * alpha_j(T_e) * (chi_j + k T_e)
 * Ion densities (not NLTE levels) supply the populations. Registered once
 * per solver entry; thread-local current-shell index keeps OMP-safe. */
static double frozenin_alpha_rr(AtomicData *atom, int ip, int ip_next, double T);
#define FBR_MAXP 96
static struct { int np; int ipa[FBR_MAXP]; double nnext[FBR_MAXP], chi[FBR_MAXP], ne; } *g_fbr = NULL;
static int g_fbr_ns = 0, g_fbr_on = -1;
static AtomicData *g_fbr_atom = NULL;
static __thread int g_fbr_s = -1;
static int g_bfrp;   /* tentative; = -1 definition below (BF_RATE_POPS) */
static void radeq_fb_rate_register(AtomicData *atom, PlasmaState *plasma, int n_shells) {
    if (g_bfrp < 0) { const char *e = getenv("LUMINA_BF_RATE_POPS");
                      g_bfrp = (e && atoi(e)) ? 1 : 0; }
    if (g_fbr_on < 0) { const char *e = getenv("LUMINA_RADEQ_FB_RATE");
                        g_fbr_on = (e && atoi(e)) ? 1 : 0; }
    if (!g_fbr_on) return;
    if (!g_fbr || g_fbr_ns != n_shells) {
        free(g_fbr);
        g_fbr = calloc((size_t)n_shells, sizeof(*g_fbr));
        g_fbr_ns = n_shells;
    }
    g_fbr_atom = atom;
    for (int s = 0; s < n_shells; s++) {
        int np = 0;
        for (int e = 0; e < atom->n_elements; e++) {
            int ip0 = atom->elem_ion_offset[e], ip1 = atom->elem_ion_offset[e + 1];
            for (int ip = ip0; ip < ip1 - 1 && np < FBR_MAXP; ip++) {
                double nx = atom->ion_number_density[(size_t)(ip + 1) * n_shells + s];
                if (nx <= 0.0) continue;
                double chi_eV = find_ioniz_energy(atom, atom->ion_pop_Z[ip],
                                                  atom->ion_pop_stage[ip]);
                if (chi_eV <= 0.0 || chi_eV > 1e9) continue;
                g_fbr[s].ipa[np] = ip;
                g_fbr[s].nnext[np] = nx;
                g_fbr[s].chi[np] = chi_eV * EV_TO_ERG;
                np++;
            }
        }
        g_fbr[s].np = np;
        g_fbr[s].ne = plasma->n_electron[s];
    }
}
/* LUMINA_BF_RATE_POPS=1: bf-HEATING populations from ion densities +
 * dilute-Boltzmann levels instead of raw NLTE level pops. 3rd member of the
 * NLTE-pop poisoning family (emit_nu C_rec -> FB_RATE; H_photo -> this):
 * thin/cold-shell NLTE garbage inflated the valley H_photo to 2e-4..1e0
 * erg/cm3/s vs the CMFGEN edep budget ~5e-8 (fix2/fix4 RADEQ-DIAG s=25),
 * swinging the balance target by 7 decades between iterations = the valley
 * see-saw. Recipe mirrors the cooling-table untracked-level branch. */
static int g_bfrp = -1;
static double bf_rate_pop(AtomicData *atom, int Z, int ion_stage, int gidx,
                          int s, int n_shells, double T_e) {
    int ip = find_ion_pop_idx(atom, Z, ion_stage);
    if (ip < 0) return 0.0;
    double n_ion = atom->ion_number_density[(size_t)ip * n_shells + s];
    if (n_ion <= 0.0) return 0.0;
    double U = atom->partition_functions[(size_t)ip * n_shells + s];
    PopulationAtomicView av = population_atomic_view(atom);
    double fraction = 0.0;
    PopulationStatus status = population_lte_level_fraction(
        &av, (size_t)ip, (size_t)gidx, T_e, U, &fraction);
    return (status == POP_OK || status == POP_EXACT_ZERO)
         ? n_ion * fraction : NAN;
}

static double radeq_fb_rate_eval(double T_e) {
    if (g_fbr_s < 0 || g_fbr_s >= g_fbr_ns || !g_fbr || !g_fbr_atom) return 0.0;
    const int s = g_fbr_s;
    double C = 0.0;
    for (int p = 0; p < g_fbr[s].np; p++) {
        int ip = g_fbr[s].ipa[p];
        double al = frozenin_alpha_rr(g_fbr_atom, ip, ip + 1, T_e);
        if (al > 0.0)
            C += g_fbr[s].ne * g_fbr[s].nnext[p] * al *
                 (g_fb_cool_kt > 0 ? (K_BOLTZMANN * T_e)
                                   : (g_fbr[s].chi[p] + K_BOLTZMANN * T_e));
    }
    return C;
}

/* LUMINA_TE_STEP_CLAMP=1: ARTIS-mirror per-iteration T_e change limit
 * (thermalbalance.cc:290-298, T_e in [0.5,2]xT_old). A convergence damper
 * that cannot move a fixed point; kills the pops<->T_e see-saw the honest
 * (non-stiff) energy terms exposed (fix4 valley 3.4kK cold-trap swing). */
static double radeq_te_step_clamp(double T_new, double T_old) {
    static int sc = -1;
    if (sc < 0) { const char *e = getenv("LUMINA_TE_STEP_CLAMP");
                  sc = (e && atoi(e)) ? 1 : 0; }
    if (!sc || T_old <= 100.0) return T_new;
    if (T_new > 2.0 * T_old) return 2.0 * T_old;
    if (T_new < 0.5 * T_old) return 0.5 * T_old;
    return T_new;
}


/* ============================================================
 * LUMINA_RADEQ_SIMUL=1: ARTIS-mirror SIMULTANEOUS T_e/ionization solve.
 * Design: docs/OUTER_THIN_LINECOOLING_DESIGN.md Part 2. Validated prototype:
 * scripts/offline_cell_balance.py (reproduces the CMFGEN outer turn-up).
 * At EVERY trial T_e the shell's ionization ladder — lagged-J photoionization
 * + non-thermal vs Milne RR + Badnell DR — is re-solved (ARTIS
 * thermalbalance.cc:141-150), level populations follow Boltzmann at the trial
 * state, and r(T) = H_dep + H_photo(T) − C_ff − C_ad − C_fb(T) − Λ_ETLA(T)
 * is bisected on the wide physical bracket [3500,140000] K (artisoptions.h).
 * After convergence the ladder runs once more at T* and T_e, n_e AND the ion
 * partition are committed (thermalbalance.cc:304 mirror); the downstream
 * nebular-Saha rewrite is skipped (single ownership; fixes the operator-split
 * limit cycle fix10/12 measured: valley 2445<->9195 K).
 * ============================================================ */
static int g_simul_on = -1;
#define SIM_MAXP 96
#define SIM_UT 24
static double *g_sim_ulut = NULL;   /* [n_ip][SIM_UT] partition U on log-T grid */
static double g_sim_logT0, g_sim_dlogT;
static void simul_build_ulut(AtomicData *atom) {
    int n_ip = atom->n_ion_pops;
    if (!g_sim_ulut) g_sim_ulut = (double *)malloc((size_t)n_ip * SIM_UT * sizeof(double));
    g_sim_logT0 = log(3500.0);
    g_sim_dlogT = (log(140000.0) - g_sim_logT0) / (SIM_UT - 1);
    for (int ip = 0; ip < n_ip; ip++) {
        int l0 = atom->level_offset[ip], l1 = atom->level_offset[ip + 1];
        for (int k = 0; k < SIM_UT; k++) {
            double T = exp(g_sim_logT0 + k * g_sim_dlogT);
            double u = 0.0;
            for (int l = l0; l < l1; l++) {
                double x = atom->level_energy_eV[l] * EV_TO_ERG / (K_BOLTZMANN * T);
                if (x < 300.0) u += (double)atom->level_g[l] * exp(-x);
            }
            g_sim_ulut[(size_t)ip * SIM_UT + k] = (u > 0.0) ? u : 1.0;
        }
    }
}
static double simul_U(int ip, double T) {
    double x = (log(T) - g_sim_logT0) / g_sim_dlogT;
    if (x < 0) x = 0; if (x > SIM_UT - 1) x = SIM_UT - 1;
    int k = (int)x; if (k >= SIM_UT - 1) k = SIM_UT - 2;
    double f = x - k;
    return (1.0 - f) * g_sim_ulut[(size_t)ip * SIM_UT + k]
         + f * g_sim_ulut[(size_t)ip * SIM_UT + k + 1];
}

/* per-shell scratch for the simultaneous evaluation */
typedef struct {
    /* pair (ionization) data */
    int np; int ipa[SIM_MAXP]; double chi[SIM_MAXP], Gph[SIM_MAXP], Hex[SIM_MAXP];
    int ne_first[SIM_MAXP];      /* 1 = first pair of its element */
    double gnt_p[SIM_MAXP];      /* per-stage NT rate (Lotz chi-suppression) */
    double nelem[SIM_MAXP];      /* element number density (on first pair) */
    int npops[SIM_MAXP];         /* element ladder length (on first pair) */
    double gnt;                  /* per-atom NT rate [s^-1] */
    double H_dep, ff_pref, Gamma_ad, natom;
    /* culled ETLA line table */
    long nl;
    int *l_ip; double *l_dE, *l_beta, *l_coeff, *l_glo, *l_gup, *l_Elo;
    double *l_BluJ, *l_ABulJ, *l_ftau;   /* B_lu*Jb, A_ul+B_ul*Jb, Sobolev tau/n_lo */
    double *nion;                /* [n_ip] trial ion densities */
    /* [DBFB] LUMINA_RADEQ_DB_FB detailed-balance bf-emission spectrum. Built in
     * the Gph loop on the SAME sigma_bf grid / f_above weight / lagged pops as
     * sh.Hex, so the bf net n*(Hex - C_fb) cancels bin-by-bin when J = B_nu^Wien(T).
     * emit_bf[p*nfb+bb] = Sum_l pop_l * 4*pi*sigma f_above dnu * (2 h nu^3/c^2)
     * (T-independent). Cooling(T) = Sum_p nion[ipa[p]] * Sum_bb emit_bf[p][bb]
     * * exp(-emit_bx[bb]/T).  Allocated/used only when g_radeq_db_fb==1. */
    double *emit_bf;             /* [SIM_MAXP*nfb] per-pair per-bin emission weight */
    double *emit_bx;             /* [nfb] h*nu_mid/k_B [K] for the Wien factor */
    int     nfb;                 /* == nlte->n_freq_bins */
    /* withParityO (LUMINA_RADEQ_COL_PAIRS): CMFGEN-faithful all-level-pair COL
     * cooling prefactors for the covered col-table ions, built per shell from
     * the LIVE NLTE populations. Replaces those ions' 2-level simul_line_term
     * (skipped from the lam sum) -> no double count. a/b/beta consumed by
     * radeq_line_cool at trial T (signed; the IGE super-elastic heaters keep
     * their sign). NULL/0 when the gate is off => byte-identical. */
    double *cp_a, *cp_b, *cp_beta;   /* [cp_cap] pair prefactors */
    long    cp_n;                    /* active pair count this shell */
} SimShell;

/* [DBFB] LUMINA_RADEQ_DB_FB: replace simul_r1's analytic frozenin_alpha_rr C_fb
 * with the emit_nu/Wien detailed-balance PARTNER of the photoheating integral
 * (construction documented at the emit_nu block ~4798-4810). Parsed once in the
 * serial prologue of radeq_simul_all (alongside fb_cool_kt_on); -1 = unparsed,
 * 0 = off => analytic C_fb, emit_bf never allocated => byte-identical. simul_r1
 * only READS this static (never -1 there). */
static int g_radeq_db_fb = -1;

/* withParityO forward decls (simul_r1, below, consumes both). */
static double radeq_line_cool(double T_e, double n_e, const double *a,
                              const double *b, const double *beta,
                              long n_active, int nonneg);
static int g_cp_on = -1;   /* LUMINA_RADEQ_COL_PAIRS gate; -1 unparsed, 0 off, 1 on */

static void simul_ladder(AtomicData *atom, SimShell *sh, double T, double *n_e_io) {
    /* equilibrium ladder with n_e fixed point (calculator solve_ion mirror) */
    double n_e = *n_e_io;
    if (!(n_e > 0.0)) n_e = 0.5 * sh->natom;
    int n_ip = atom->n_ion_pops;
    /* [STAGE4-R2 A2] top-ion Saha closure. A level-less destination rung
     * (e.g. Ni V / Co V: no levels, no sigma_bf, no NLTE in this dataset) has no
     * recombination floor, so the product-chain ladder y[j+1]=y[j]*r runs away UP
     * into it (round-1 Ni f(V) ~ 1). Mirror the NLTE hi_is_topstage detection
     * (plasma.c:9627-9632): when stage j+1 is a level-less rung, truncate the
     * ladder there (r=0 => y[j+1]=0). Gated on stage4 + LUMINA_SIMUL_CAP_TOPION
     * (default ON under stage4, OFF otherwise => byte-identical baseline). */
    static int g_simul_cap_topion = -1;
    if (g_simul_cap_topion < 0) {
        const char *e = getenv("LUMINA_SIMUL_CAP_TOPION");
        if (e) g_simul_cap_topion = atoi(e) ? 1 : 0;
        else   g_simul_cap_topion = nlte_stage4_enabled() ? 1 : 0;
    }
    /* A4 (ARTIS parity): thermal collisional ionization + 3-body recombination in
     * the ion-balance ladder. Adds C_ion to the up-rate and the EXACT detailed-
     * balance 3-body partner C_rec to the recomb rate, so the LTE Saha fixed point
     * is preserved (both channels vanish net at LTE). Only under the master gate
     * => byte-identical when off. Inert unless LUMINA_RADEQ_SIMUL is also on. */
    const int    parity_ladder = artis_parity_enabled();
    const double SIM_SAHACONST = 2.0706659e-16;  /* (h^2/(2 pi m_e k))^1.5 [cgs] */
    for (int it = 0; it < 20; it++) {
        double ne_new = 0.0;
        for (int p = 0; p < sh->np; ) {
            int npop = sh->npops[p];
            double nel = sh->nelem[p];
            double y[SIM_MAXP]; y[0] = 1.0; double ysum = 1.0;
            for (int j = 0; j < npop - 1; j++) {
                double al = frozenin_alpha_rr(atom, sh->ipa[p + j],
                                              sh->ipa[p + j] + 1, T);
                double G = sh->Gph[p + j] + sh->gnt_p[p + j];
                double rec_rate = n_e * al;
                if (parity_ladder && T > 0.0 && n_e > 0.0) {
                    int ipl = sh->ipa[p + j], ipu = ipl + 1;
                    double chi = sh->chi[p + j];
                    double u = (chi > 0.0) ? chi / (K_BOLTZMANN * T) : 0.0;
                    int has_upper = (ipu < n_ip &&
                        atom->level_offset[ipu + 1] > atom->level_offset[ipu]);
                    if (u > 0.0 && u < 700.0 && has_upper) {
                        int st = atom->ion_pop_stage[ipl];
                        int zeff = st + 1; if (zeff < 1) zeff = 1;
                        double sig = 7.91e-18 / ((double)zeff * (double)zeff);
                        double g_col = (st <= 0) ? 0.1 : (st == 1) ? 0.2 : 0.3;
                        int glo = atom->level_g[atom->level_offset[ipl]];
                        int gup = atom->level_g[atom->level_offset[ipu]];
                        if (glo > 0 && gup > 0) {
                            double C_ion = n_e * 1.55e13 / sqrt(T) * g_col * sig *
                                           exp(-u) / u;
                            double C_rec = n_e * n_e * SIM_SAHACONST *
                                           ((double)glo / (double)gup) * 1.55e13 *
                                           g_col * sig * K_BOLTZMANN / (T * chi);
                            if (isfinite(C_ion) && C_ion > 0.0) G += C_ion;
                            if (isfinite(C_rec) && C_rec > 0.0) rec_rate += C_rec;
                        }
                    }
                }
                double r = (rec_rate > 0.0) ? G / rec_rate : 0.0;
                if (!isfinite(r) || r < 0.0) r = 0.0;
                if (r > 1e28) r = 1e28;
                /* [STAGE4-R2 A2] clamp the step into a level-less top rung. */
                if (g_simul_cap_topion) {
                    int ipn = sh->ipa[p + j] + 1;
                    if (ipn < n_ip &&
                        atom->level_offset[ipn + 1] == atom->level_offset[ipn])
                        r = 0.0;
                }
                y[j + 1] = y[j] * r;
                if (y[j + 1] > 1e280) { for (int q = 0; q <= j + 1; q++) y[q] /= 1e280; }
            }
            ysum = 0.0; for (int j = 0; j < npop; j++) ysum += y[j];
            double zbar = 0.0;
            for (int j = 0; j < npop; j++) {
                double fr = (ysum > 0.0) ? y[j] / ysum : (j == 0 ? 1.0 : 0.0);
                sh->nion[sh->ipa[p] + j] = nel * fr;
                zbar += (double)atom->ion_pop_stage[sh->ipa[p] + j] * fr;
            }
            ne_new += nel * zbar;
            p += (npop - 1);
        }
        double ne_next = 0.5 * (n_e + (ne_new > 1e-6 * sh->natom ? ne_new
                                                                 : 1e-6 * sh->natom));
        if (fabs(ne_next - n_e) < 1e-3 * n_e) { n_e = ne_next; break; }
        n_e = ne_next;
    }
    *n_e_io = n_e;
}

static int g_simul_nested = 0;        /* LUMINA_SIMUL_NESTED inner threads */
static long g_simul_nested_nl = 300000; /* LUMINA_SIMUL_NESTED_NL threshold */

static inline double simul_line_term(const SimShell *sh, long m, double T,
                                     double n_e, double invsq) {
    double nion = sh->nion[sh->l_ip[m]];
    if (nion <= 0.0) return 0.0;
    double x = sh->l_Elo[m] / (K_BOLTZMANN * T);
    if (x > 300.0) return 0.0;
    double U = simul_U(sh->l_ip[m], T);
    double nlo = nion * sh->l_glo[m] * exp(-x) / U;
    if (nlo <= 0.0) return 0.0;
    double exb = exp(-fmin(sh->l_beta[m] / T, 300.0));
    double qlu = sh->l_coeff[m] / sh->l_glo[m] * invsq * exb;
    double qul = sh->l_coeff[m] / sh->l_gup[m] * invsq;
    double Clu = n_e * qlu, Cul = n_e * qul;
    double tau = sh->l_ftau[m] * nlo;
    double be = radeq_beta_esc(tau);
    double Rul = sh->l_ABulJ[m] * be, Rlu = sh->l_BluJ[m] * be;
    double den = Cul + Rul;
    if (den <= 0.0) return 0.0;
    double nup = nlo * (Clu + Rlu) / den;
    return sh->l_dE[m] * (nlo * qlu * n_e - nup * qul * n_e);
}

/* [SIMUL-CT] per-term cooling shadows of the LAST simul_r1 call on this thread
 * (diagnostic read-out for the RADEQ_DIAG scan; the C accumulation stream is
 * untouched — shadows are separate adds/stores, champion FP byte-identical). */
static __thread double g_r1d_H, g_r1d_ff, g_r1d_ad, g_r1d_fb, g_r1d_lam,
                       g_r1d_colpairs;
static double simul_r1(AtomicData *atom, SimShell *sh, double T, double *n_e_out) {
    double n_e = *n_e_out;
    simul_ladder(atom, sh, T, &n_e);
    *n_e_out = n_e;
    /* heating: deposition (full) + bf excess re-weighted by trial ions */
    double H = sh->H_dep;
    for (int p = 0; p < sh->np; p++)
        H += sh->nion[sh->ipa[p]] * sh->Hex[p];
    /* continuum cooling */
    double c_ff_d = sh->ff_pref * n_e * n_e * sqrt(T);
    double C = c_ff_d;
    double c_ad_d = 1.5 * n_e * K_BOLTZMANN * T * sh->Gamma_ad;
    C += c_ad_d;
    g_r1d_H = H; g_r1d_ff = c_ff_d; g_r1d_ad = c_ad_d; g_r1d_fb = 0.0;
    if (g_radeq_db_fb == 1) {
        /* [DBFB] detailed-balance bf cooling = the emit_nu/Wien EMISSION partner of
         * H_photo (heating H += nion[ip]*Hex[p]).  C_fb(T) = Sum_p nion[ip] *
         * Sum_bb emit_bf[p][bb] * exp(-h nu_bb/kT).  emit_bf was built in the Gph
         * loop from the SAME sigma_bf, f_above weight and lagged level pops as Hex,
         * so the bf net nion[ip]*(Hex[p] - C_fb[p](T)) cancels bin-by-bin whenever
         * the field J_bb = B_nu^Wien(T) = (2 h nu^3/c^2) exp(-h nu/kT). Replaces the
         * analytic frozenin_alpha_rr term below (that term was FIELD-decoupled, so a
         * brightening ionizing field made H_photo unbounded with no cooling partner
         * -> cold root vanished -> pin_hi ratchet; TRACE_LEDGER.txt root cause). */
        int nfb = sh->nfb;
        double invT = 1.0 / T;
        double wien[NLTE_N_FREQ_BINS];
        for (int bb = 0; bb < nfb; bb++)
            wien[bb] = exp(-sh->emit_bx[bb] * invT);
        for (int p = 0; p < sh->np; p++) {
            double np_ = sh->nion[sh->ipa[p]];
            if (np_ <= 0.0) continue;
            const double *ep = sh->emit_bf + (size_t)p * nfb;
            double acc = 0.0;
            for (int bb = 0; bb < nfb; bb++) acc += ep[bb] * wien[bb];
            double fbterm = np_ * acc;
            C += fbterm;
            g_r1d_fb += fbterm;
        }
    } else
    for (int p = 0; p < sh->np; p++) {                     /* fb: rate-based */
        double al = frozenin_alpha_rr(atom, sh->ipa[p], sh->ipa[p] + 1, T);
        if (al > 0.0) {
            /* [FB-COOL-KT] fb thermal cooling weight: ARTIS charges only the
             * photoelectron kinetic energy (~kTe); chi is the ionization ledger.
             * OFF => legacy (chi + kTe), byte-identical (g_fb_cool_kt pre-inited
             * serially in radeq_simul_all, so it is 0/1 here, never -1). */
            double fbterm = n_e * sh->nion[sh->ipa[p] + 1] * al *
                 (g_fb_cool_kt ? (K_BOLTZMANN * T) : (sh->chi[p] + K_BOLTZMANN * T));
            C += fbterm;
            g_r1d_fb += fbterm;
        }
    }
    /* ETLA two-level SE line exchange with trial pops (signed; heating allowed).
     * [SIMUL-LB] The line table is the SIMUL load imbalance: s0 carries 2.14M
     * lines vs 71.7k outside (30x) — the outer shell-parallel loop's wall
     * clock is bounded by s0's serial sum (x ~30-50 T-trials). For heavy
     * shells (nl >= LUMINA_SIMUL_NESTED_NL, default 3e5) the sum runs as a
     * NESTED omp reduction with LUMINA_SIMUL_NESTED threads (default 0 = off,
     * opt-in). Pure reduction over read-only tables + LUT partition function
     * => thread-safe; FP sum order changes => verify by physics equivalence
     * (T_e |ratio-1| ~ 1e-12-1e-6), not bitwise. */
    double invsq = 1.0 / sqrt(T), lam = 0.0;
    if (g_simul_nested > 1 && sh->nl >= g_simul_nested_nl) {
        #pragma omp parallel for num_threads(g_simul_nested) \
                reduction(+:lam) schedule(static)
        for (long m = 0; m < sh->nl; m++)
            lam += simul_line_term(sh, m, T, n_e, invsq);
    } else {
        for (long m = 0; m < sh->nl; m++)
            lam += simul_line_term(sh, m, T, n_e, invsq);
    }
    C += lam;
    g_r1d_lam = lam;
    /* withParityO: CMFGEN-faithful all-pair COL cooling for the covered ions
     * (their 2-level lines were skipped from lam above => no double count).
     * Signed (nonneg=0): the IGE super-elastic collisional HEATERS keep their
     * sign. cp_n==0 when the gate is off => byte-identical. */
    g_r1d_colpairs = 0.0;
    if (g_cp_on == 1 && sh->cp_n > 0) {
        g_r1d_colpairs = radeq_line_cool(
            T, n_e, sh->cp_a, sh->cp_b, sh->cp_beta, sh->cp_n, 0);
        C += g_r1d_colpairs;
    }
    return H - C;
}

/* P1 co-evolve: lagged MC shadow field for the photoionization rate integral
 * (registered from the CUDA co-evolve consume block; NULL => bare deterministic J).
 * Declared here (above radeq_simul_all) because the LIVE photoion rate — the one
 * that actually drives S/Si ionization in the RADEQ_SIMUL champion consume config —
 * lives inside radeq_simul_all's Gph loop. */
static const double *g_photoion_mc_J = NULL;
static const int    *g_photoion_mc_count = NULL;  /* [nshells*nfb] MC per-bin packet tally */
static double g_photoion_mc_alpha = 0.0;
static int g_photoion_mc_nshells = 0, g_photoion_mc_nfb = 0;
/* Occupancy guard (LUMINA_COEVOLVE_PHOTOION_OCC): where the MC transport tallied
 * ZERO packets in a bin, keep the deterministic J instead of the shot-noise-starved
 * blend. Zero-count is a statistical fact, not a tunable threshold. Read once. */
static int g_photoion_mc_occ = -1;
/* LUMINA_GPH_ALLLEVEL: detailed-balance photoionization. The simul recombination
 * side (frozenin_alpha_rr) integrates the Milne coefficient over ALL levels of the
 * ion, but the Gph photoionization side integrates only the GROUND level -> net
 * over-recombination -> IGE (Fe/Co) trapped in III (benchmark: IV). When on (and
 * only effective with LUMINA_GPH_SIGMA_CMFGEN + cmfgen_loaded for per-level
 * sigma_bf), Gph is summed over all levels with Boltzmann population weights,
 * restoring symmetry with the Milne alpha. Parsed once in radeq_simul_all's
 * single-threaded prologue; -1 = unparsed, 0 = off (Gph loop byte-identical). */
static int g_gph_alllevel = -1;
/* LUMINA_GPH_ALLLEVEL_NLTE: within the all-level block above, use the ACTUAL NLTE
 * level populations (departure b_k) as photoion weights instead of Boltzmann@T_e.
 * Boltzmann (b_k=1) over-ionizes the IME (Si/S); the true IGE excited levels are
 * over-populated (b_k>>1) and IME depressed (b_k<<1). Only effective when
 * g_gph_alllevel is also on; -1 = unparsed, 0 = off (all-level loop unchanged). */
static int g_gph_alllevel_nlte = -1;
/* [IONIZ-SELFTEST] known-answer (constrained) test of the IONIZATION path, the
 * counterpart of the four transport self-tests (cmf_plasma/nlte/fsolve/obs).
 *   Identity: with J_nu = B_nu(T_e) and detailed balance intact, the photoion /
 *   recombination equilibrium MUST reproduce Saha exactly:
 *       q  = Gamma_phot / alpha_rec                        [cm^-3]
 *       q_saha = 2 (U_{i+1}/U_i) (2 pi m_e k T/h^2)^{3/2} exp(-chi/kT)
 *   (n(i+1)/n(i) = q/n_e, so q is the n_e-free form of the r in the brief.)
 * Modes: 1 = force every photoion field read to B_nu(T_e[s]) AND print the
 *            per-(shell,ion) identity table  <- the known-answer test
 *        2 = print only, leave the live field alone (audit of a real run)
 * Default (env unset) => 0 => no site fires, no print => byte-identical. */
static int g_ioniz_selftest = -1;
static int ioniz_selftest_mode(void) {
    if (g_ioniz_selftest < 0) {
        const char *e = getenv("LUMINA_IONIZ_SELFTEST");
        g_ioniz_selftest = (e && *e) ? atoi(e) : 0;
    }
    return g_ioniz_selftest;
}
/* forward decl: the NLTE-route photoion rate, cross-checked in the same table */
static double coupled_photoion_rate_jnu(AtomicData *atom, NLTEConfig *nlte,
                                        int ip, int s, double T_e, int n_shells,
                                        const double *jblend_lstar,
                                        const double *jblend_b, double jblend_W,
                                        double wbfloor_T, double *Krow);
/* Partition function on the SAME level list + x<50 cut the production integrals
 * use (frozenin_alpha_rr's U_ion recipe), so the reference is not a second
 * convention. floor_g: fall back to the ground g when the sum underflows. */
static double ioniz_selftest_U(AtomicData *atom, int ip, double T, int floor_g) {
    if (ip < 0) return 1.0;
    int n0 = atom->level_offset[ip], n1 = atom->level_offset[ip + 1];
    double kT = K_BOLTZMANN * T, u = 0.0;
    for (int l = n0; l < n1; l++) {
        double x = atom->level_energy_eV[l] * EV_TO_ERG / kT;
        if (x < 50.0) u += (double)atom->level_g[l] * exp(-x);
    }
    if (u >= 1.0) return u;
    if (floor_g && n1 > n0) { int gg = atom->level_g[n0]; return (gg >= 1) ? (double)gg : 1.0; }
    return (u > 0.0) ? u : 1.0;
}
/* [TEHOLD] LINE_THERM addendum: per-shell radeq root status, captured in the
 * radeq_simul_all shell loop and printed after the parallel region (host-side, in
 * shell order). Distinguishes "T_e failed to climb because thermalization is
 * innocent" from "T_e frozen because the solver HELD (pin_lo/pin_hi, no root in
 * bracket)". Status: 0=not-solved-this-iter, 1=root-found, 2=pin_lo(HOLD prev),
 * 3=pin_hi(HOLD prev). Written only when the LINE_THERM gate is on. */
#define TEHOLD_MAXSH 512
static int    g_tehold_status[TEHOLD_MAXSH];
static double g_tehold_te[TEHOLD_MAXSH];
static double g_tehold_told[TEHOLD_MAXSH];
static const char *tehold_root_name(int st) {
    return (st == 1) ? "root-found" :
           (st == 2) ? "pin_lo(HOLD-no-root)" :
           (st == 3) ? "pin_hi(HOLD-no-root)" : "not-solved-this-iter";
}
void plasma_set_photoion_mc_field(const double *J, double alpha, int nshells, int nfb,
                                  const int *counts) {
    g_photoion_mc_J = J; g_photoion_mc_alpha = alpha;
    g_photoion_mc_nshells = nshells; g_photoion_mc_nfb = nfb;
    g_photoion_mc_count = counts;
    if (g_photoion_mc_occ < 0)
        g_photoion_mc_occ = (getenv("LUMINA_COEVOLVE_PHOTOION_OCC") &&
                             atoi(getenv("LUMINA_COEVOLVE_PHOTOION_OCC"))) ? 1 : 0;
    /* Liveness: once per plasma pass (setter armed once per co-evolve iter),
     * report MC occupancy for the two documented hot-root shells. */
    if (g_photoion_mc_occ && counts && nfb > 0) {
        double dlognu = log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)nfb;
        int probe[2] = {40, 47};
        for (int pi = 0; pi < 2; pi++) {
            int s = probe[pi];
            if (s >= nshells) continue;
            int zero = 0, zero_uv = 0;
            for (int bb = 0; bb < nfb; bb++) {
                if (counts[(size_t)s * nfb + bb] != 0) continue;
                zero++;
                double nu = NLTE_NU_MIN * exp((bb + 0.5) * dlognu);
                if (2.99792458e18 / nu < 3000.0) zero_uv++;  /* lambda < 3000 A */
            }
            printf("[OCC] s%d: %d/%d bins MC-sampled, %d unsampled kept "
                   "deterministic (uv<3000A: %d)\n",
                   s, nfb - zero, nfb, zero, zero_uv);
        }
        fflush(stdout);
    }
}

/* [PUMPF] LUMINA_RADEQ_PUMP_FIELD: unify the radeq line-pump field. The ETLA
 * line term (simul_line_term) baked its per-line Jb from the HARD-WIRED
 * deterministic cs_J (nlte_get_J_at_nu, plasma.c line-build), while the Gph
 * photoion + bf-heating in the SAME balance consume the alpha-blended MC field
 * (J = alpha*mc_J + (1-alpha)*cs_J). te_bias_budget C2: the super-thermal cs_J
 * pump under-cools/pump-heats (bias -664/-1563/-3863 K at s0/s4/s8). When ON,
 * the line-build Jb is read from that SAME blended field. ONLY the field SOURCE
 * changes -- the bin mapping is byte-identical to nlte_get_J_at_nu and the
 * cooling/stimulated structure of simul_line_term is untouched. -1 unparsed,
 * 0 off (byte-identical), 1 on. */
static int g_radeq_pump_field = -1;

/* FIX-2 [PUMPF fallback]: LUMINA_RADEQ_PUMP_FALLBACK. When the alpha-blended line
 * pump falls back on a ZERO-COUNT mc bin (mc field armed but this bin was never
 * sampled), the legacy path returns the deterministic super-thermal cs_J (int
 * J_cs ~100x mc at depth -> pump-heating, kpr4 VERDICT §b: -56..-1683 K). =1
 * routes those zero-count bins to the local thermal B_nu(T_e) instead (consistent
 * with the DBFB ledger), removing a warm-biased term the current field cannot
 * self-correct. Default 0 => cs_J (byte-identical). The mc-not-armed (pre-
 * transport) and out-of-grid guards are untouched. */
static int g_radeq_pump_fallback = -1;

/* Return the per-line pump Jb from the SAME alpha-blended field the Gph/Hex loop
 * consumes.  The cs_J branch is byte-identical to nlte_get_J_at_nu(nlte,s,nu_l)
 * (same log-grid bin, same out-of-range 1e-30) so ONLY the field source changes.
 * The blend + guard MIRROR the Gph sites EXACTLY (plasma.c ground ~5873,
 * NLTE/all-level ~5750/5806): if the MC field is NULL (pre first transport pass),
 * shape-mismatched, or a zero-count bin, keep cs_J -- the SAME startup fallback
 * the Gph loop uses (guard condition false => J stays cs_J).  Per-line source
 * tallied in *n_blend / *n_cs (host-side; the reduction copies are thread-private
 * where this is called). Reuses g_photoion_mc_* (one field, no new parameter). */
static inline double radeq_pump_line_Jb(NLTEConfig *nlte, int s, int nfb,
                                        double nu_l, double Te_s,
                                        long *n_blend, long *n_cs, long *n_bnu) {
    if (nu_l <= nlte->nu_min || nu_l >= nlte->nu_max) { (*n_cs)++; return 1e-30; }
    int bin = (int)(log(nu_l / nlte->nu_min) / nlte->d_log_nu);
    if (bin < 0) bin = 0;
    if (bin >= nlte->n_freq_bins) bin = nlte->n_freq_bins - 1;
    double cs = nlte->J_nu[(size_t)s * nlte->n_freq_bins + bin];  /* == nlte_get_J_at_nu */
    int mc_armed = (g_photoion_mc_J && s < g_photoion_mc_nshells &&
                    g_photoion_mc_nfb == nfb);
    int bin_zero = (g_photoion_mc_occ && g_photoion_mc_count &&
                    g_photoion_mc_count[(size_t)s * nfb + bin] == 0);
    if (mc_armed && !bin_zero) {
        (*n_blend)++;
        return g_photoion_mc_alpha * g_photoion_mc_J[(size_t)s * nfb + bin]
               + (1.0 - g_photoion_mc_alpha) * cs;
    }
    (*n_cs)++;
    /* FIX-2: genuine zero-count mc bin (field armed) -> local thermal B_nu(Te)
     * instead of the super-thermal cs_J. The pre-transport case (mc field NULL /
     * shape-mismatched) keeps cs_J (the untouched startup fallback). */
    if (g_radeq_pump_fallback == 1 && mc_armed && bin_zero && Te_s > 0.0) {
        (*n_bnu)++;
        return planck_bnu(Te_s, nu_l);
    }
    return cs;
}

/* =====================================================================
 * withParityO — LUMINA_RADEQ_COL_PAIRS registry + per-shell build.
 *
 * Replaces the covered col-table ions' 2-level simul_line_term collisional
 * cooling with the CMFGEN-faithful all-level-pair COL sum (shared core in
 * lumina_radeq_col_pairs.h), fed the LIVE NLTE level populations and NO
 * beta-escape. The covered ions are dropped from the lam line sum (skip flag)
 * so there is no double count. Gate default OFF => registry never armed, skip
 * flag all-zero, cp_n==0 => the SIMUL solve is byte-identical.
 *
 * Certified offline: src/lumina_radeq_col_pairs_bench.c reproduces the dig_F11
 * numbers (Si/S/Fe/Co/Ni III) to 1.0000 through this exact core.
 * ===================================================================== */
typedef struct {
    int Z, ion0, ip, src_slot;      /* src_slot: -1 = feiii_col, else col_ion slot */
    int ntemp, n_trans;
    const double *tgrid;            /* [ntemp] Kelvin */
    const double *om;               /* [n_trans*ntemp] split-J Omega(T) */
    const int    *clo, *chi;        /* [n_trans] level_number (clo = lower) */
    int      nline;                 /* pruned line f-list (both level_num < maxlev) */
    int     *llo, *lhi; double *lf;
    long     nlev_max;              /* max level_number seen (for lvmap sizing) */
} CpIon;
static int    g_cp_maxlev = 512;        /* per-ion level cap (fail-loud on drop) */
static CpIon  g_cp_ions[1 + LUMINA_MAX_COL_IONS];
static int    g_cp_nions = 0;
static char  *g_cp_line_covered = NULL; /* [radeq_n_lines] 1 = owned by pair loop */
static const double g_cp_oset = 0.1;    /* CMFGEN forbidden Omega default */
/* fail-loud shell-8 spot-check + one-shot census, written at s==8, host-printed
 * after the parallel region. Indexed by a small (Z,ion0) probe map. */
#define CP_NPROBE 5
static const int g_cp_probe_Z[CP_NPROBE]   = {14, 16, 26, 27, 28};
static const int g_cp_probe_ion[CP_NPROBE] = { 2,  2,  1,  2,  2};
static const char *g_cp_probe_name[CP_NPROBE] = {"SiIII","SIII","FeIII","CoIII","NiIII"};
static double g_cp_s8_cool[CP_NPROBE];
static RcpCensus g_cp_s8_cen[CP_NPROBE];
static long g_cp_s8_dropped = 0;
static int  g_cp_census_done = 0;

/* Arm the registry once per process (idempotent). Reads the gate, enumerates the
 * covered col-table ions (feiii_col + col_ion_*), prunes their low-level line
 * f-lists for the van-Regemorter fill, and flags the RadEqLine rows those ions
 * own so the lam sum can skip them. */
static void radeq_colpairs_register(AtomicData *atom, OpacityState *opacity) {
    if (g_cp_on >= 0) return;                       /* already armed */
    const char *e = getenv("LUMINA_RADEQ_COL_PAIRS");
    g_cp_on = (e && atoi(e)) ? 1 : 0;
    const char *ml = getenv("LUMINA_RADEQ_COLPAIRS_MAXLEV");
    if (ml) { int v = atoi(ml); if (v >= 2) g_cp_maxlev = v; }
    if (!g_cp_on) return;
    g_cp_nions = 0;
    for (int src = -1; src < atom->ncol_ions; src++) {
        CpIon *c = &g_cp_ions[g_cp_nions];
        if (src < 0) {
            if (!atom->feiii_col_loaded) continue;
            c->Z = atom->feiii_col_Z; c->ion0 = atom->feiii_col_ion;
            c->ntemp = atom->feiii_col_n_temp; c->n_trans = atom->feiii_col_n_trans;
            c->tgrid = atom->feiii_col_tgrid; c->om = atom->feiii_col_omega;
            c->clo = atom->feiii_col_lo; c->chi = atom->feiii_col_hi; c->src_slot = -1;
        } else {
            c->Z = atom->col_ion_Z[src]; c->ion0 = atom->col_ion_stage[src];
            c->ntemp = atom->col_ion_n_temp[src]; c->n_trans = atom->col_ion_n_trans[src];
            c->tgrid = atom->col_ion_tgrid[src]; c->om = atom->col_ion_omega[src];
            c->clo = atom->col_ion_lo[src]; c->chi = atom->col_ion_hi[src]; c->src_slot = src;
        }
        c->ip = find_ion_pop_idx(atom, c->Z, c->ion0);
        if (c->ip < 0 || c->n_trans <= 0 || c->ntemp <= 0) continue;
        c->nlev_max = 0;
        for (int t = 0; t < c->n_trans; t++) {
            if (c->clo[t] > c->nlev_max) c->nlev_max = c->clo[t];
            if (c->chi[t] > c->nlev_max) c->nlev_max = c->chi[t];
        }
        /* pruned line f-list: only pairs both below the level cap (higher levels
         * are dropped from the pair sum anyway) — bounds Fe III's huge line list. */
        long cnt = 0;
        for (int l = 0; l < atom->n_lines; l++)
            if (atom->line_atomic_number[l] == c->Z &&
                atom->line_ion_number[l] == c->ion0 &&
                atom->line_level_lower[l] < g_cp_maxlev &&
                atom->line_level_upper[l] < g_cp_maxlev) cnt++;
        c->nline = 0;
        c->llo = (int *)malloc((size_t)(cnt > 0 ? cnt : 1) * sizeof(int));
        c->lhi = (int *)malloc((size_t)(cnt > 0 ? cnt : 1) * sizeof(int));
        c->lf  = (double *)malloc((size_t)(cnt > 0 ? cnt : 1) * sizeof(double));
        for (int l = 0; l < atom->n_lines; l++) {
            if (atom->line_atomic_number[l] != c->Z ||
                atom->line_ion_number[l] != c->ion0) continue;
            int lo = atom->line_level_lower[l], hi = atom->line_level_upper[l];
            if (lo >= g_cp_maxlev || hi >= g_cp_maxlev) continue;
            c->llo[c->nline] = lo; c->lhi[c->nline] = hi;
            c->lf[c->nline] = atom->line_f_lu ? atom->line_f_lu[l] : 0.0;
            c->nline++;
        }
        g_cp_nions++;
    }
    /* flag the RadEqLine rows owned by a covered ion (skip them from lam) */
    if (radeq_n_lines > 0) {
        g_cp_line_covered = (char *)calloc((size_t)radeq_n_lines, 1);
        for (long k = 0; k < radeq_n_lines; k++) {
            int Zk = atom->ion_pop_Z[radeq_lines[k].ip];
            int ik = atom->ion_pop_stage[radeq_lines[k].ip];
            for (int c = 0; c < g_cp_nions; c++)
                if (g_cp_ions[c].Z == Zk && g_cp_ions[c].ion0 == ik) {
                    g_cp_line_covered[k] = 1; break;
                }
        }
    }
    printf("  [COL-PAIRS] LUMINA_RADEQ_COL_PAIRS=1 ARMED: %d covered ions, "
           "maxlev=%d, no-beta CMFGEN COL sum with LIVE NLTE pops replaces their "
           "2-level line cooling\n", g_cp_nions, g_cp_maxlev);
    for (int c = 0; c < g_cp_nions; c++)
        printf("    covered: Z=%d ion0=%d (%d col pairs, %d low lines, nlev_max=%ld)\n",
               g_cp_ions[c].Z, g_cp_ions[c].ion0, g_cp_ions[c].n_trans,
               g_cp_ions[c].nline, g_cp_ions[c].nlev_max);
    fflush(stdout);
}

/* Build the pair prefactors (a,b,beta) for ONE covered ion at shell s from the
 * live NLTE populations, appending to a[]/b[]/beta[] at *n. Returns the census
 * (per-source pair counts + cooling at T_ref) and the count of levels dropped by
 * the cap. Caller guarantees capacity >= *n + maxlev*(maxlev-1)/2. */
static void radeq_colpairs_ion_shell(const CpIon *c, AtomicData *atom,
        NLTEConfig *nlte, int n_shells, int s, double ne, double T_ref,
        int maxlev, double *a, double *b, double *beta, long *n,
        RcpCensus *cen, long *n_dropped) {
    if (cen) { cen->n_tab = cen->n_vr = cen->n_set = 0;
               cen->c_tab = cen->c_vr = cen->c_set = 0.0; cen->n_pairs = 0; }
    int ip = c->ip;
    int l0 = atom->level_offset[ip], l1 = atom->level_offset[ip + 1];
    int cnt = l1 - l0;
    if (cnt < 2) return;
    /* gather + select the lowest-energy maxlev levels */
    int    *ord = (int *)malloc((size_t)cnt * sizeof(int));
    for (int i = 0; i < cnt; i++) ord[i] = l0 + i;
    /* partial selection by energy (simple insertion of K smallest) */
    int K = cnt < maxlev ? cnt : maxlev;
    for (int i = 0; i < K; i++) {
        int mn = i;
        for (int j = i + 1; j < cnt; j++)
            if (atom->level_energy_eV[ord[j]] < atom->level_energy_eV[ord[mn]]) mn = j;
        int tmp = ord[i]; ord[i] = ord[mn]; ord[mn] = tmp;
    }
    if (n_dropped) *n_dropped += (cnt - K);
    double *edge = (double *)malloc((size_t)K * sizeof(double));
    double *npop = (double *)malloc((size_t)K * sizeof(double));
    int    *gg   = (int *)malloc((size_t)K * sizeof(int));
    int    *pqn  = (int *)malloc((size_t)K * sizeof(int));
    int    *lvmap = (int *)malloc((size_t)(c->nlev_max + 1) * sizeof(int));
    for (long i = 0; i <= c->nlev_max; i++) lvmap[i] = -1;
    for (int k = 0; k < K; k++) {
        int l = ord[k];
        double E = atom->level_energy_eV[l];
        edge[k] = (1000.0 - E) * EV_TO_ERG / RCP_HPL15;   /* higher edge = lower E */
        gg[k]   = atom->level_g[l];
        pqn[k]  = -1;                                      /* config-n unknown -> 0.2 */
        int nl  = nlte->global_to_nlte_level ? nlte->global_to_nlte_level[l] : -1;
        npop[k] = (nl >= 0 && nlte->nlte_level_populations)
                ? nlte->nlte_level_populations[(size_t)nl * n_shells + s] : 0.0;
        int lnum = atom->level_num[l];
        if (lnum >= 0 && lnum <= c->nlev_max) lvmap[lnum] = k;
    }
    /* tabulated split-J Omega records, remapped + log-log interp at T_ref */
    int cap_tab = c->n_trans;
    int *tlo = (int *)malloc((size_t)(cap_tab > 0 ? cap_tab : 1) * sizeof(int));
    int *thi = (int *)malloc((size_t)(cap_tab > 0 ? cap_tab : 1) * sizeof(int));
    double *tom = (double *)malloc((size_t)(cap_tab > 0 ? cap_tab : 1) * sizeof(double));
    int ntab = 0;
    for (int t = 0; t < c->n_trans; t++) {
        int lo = c->clo[t], hi = c->chi[t];
        if (lo < 0 || hi < 0 || lo > c->nlev_max || hi > c->nlev_max) continue;
        int cl = lvmap[lo], ch = lvmap[hi];
        if (cl < 0 || ch < 0) continue;                   /* level dropped by cap */
        tlo[ntab] = cl; thi[ntab] = ch;                   /* cl = lower level_num = lower E */
        tom[ntab] = rcp_loglog(c->tgrid, &c->om[(size_t)t * c->ntemp], c->ntemp, T_ref);
        ntab++;
    }
    /* oscillator-strength records for the van-Regemorter fill */
    int *flo = (int *)malloc((size_t)(c->nline > 0 ? c->nline : 1) * sizeof(int));
    int *fhi = (int *)malloc((size_t)(c->nline > 0 ? c->nline : 1) * sizeof(int));
    double *fv = (double *)malloc((size_t)(c->nline > 0 ? c->nline : 1) * sizeof(double));
    int nf = 0;
    for (int t = 0; t < c->nline; t++) {
        int lo = c->llo[t], hi = c->lhi[t];
        if (lo > c->nlev_max || hi > c->nlev_max) continue;
        int cl = lvmap[lo], ch = lvmap[hi];
        if (cl < 0 || ch < 0) continue;
        flo[nf] = cl; fhi[nf] = ch; fv[nf] = c->lf[t]; nf++;
    }
    radeq_col_pairs_build(K, edge, gg, npop, pqn, ne, T_ref / 1.0e4, g_cp_oset, 1.0,
                          ntab, tlo, thi, tom, nf, flo, fhi, fv,
                          a, b, beta, n, cen);
    free(ord); free(edge); free(npop); free(gg); free(pqn); free(lvmap);
    free(tlo); free(thi); free(tom); free(flo); free(fhi); free(fv);
}

static int g_simul_iter_no = 0;   /* outer-iteration tag for SIMUL diag prints */
static void radeq_simul_all(PlasmaState *plasma, GammaDeposition *gamma_dep,
                            NLTEConfig *nlte, AtomicData *atom,
                            OpacityState *opacity, double time_explosion,
                            int n_shells) {
    g_simul_iter_no++;            /* serial prologue; 1-based call count */
    build_radeq_line_table(nlte, atom, opacity);
    if (!g_sim_ulut) simul_build_ulut(atom);
    radeq_colpairs_register(atom, opacity);   /* withParityO: arm once, gated */
    double radeq_damp = 0.5;
    { const char *d = getenv("LUMINA_RADEQ_DAMP"); if (d) radeq_damp = atof(d); }
    /* [TEHOLD] LINE_THERM addendum diagnostic (default OFF => no print, no state
     * change). When on, record each gated shell's radeq root branch + committed
     * T_e so the probe can tell whether s0's root is actually being re-solved. */
    int tehold_on = (getenv("LUMINA_LINE_THERM") && atoi(getenv("LUMINA_LINE_THERM"))) ? 1 : 0;
    int tehold_smax = 2;
    if (getenv("LUMINA_LINE_THERM_SMAX")) tehold_smax = atoi(getenv("LUMINA_LINE_THERM_SMAX"));
    if (tehold_smax < 0) tehold_smax = 0;
    if (tehold_on)
        for (int s = 0; s < n_shells && s < TEHOLD_MAXSH; s++) g_tehold_status[s] = 0;
    int n_ip = atom->n_ion_pops;
    int nfb = nlte->n_freq_bins;
    long n_pin_hi = 0, n_pin_lo = 0;
    long n_jtable_evals = 0;    /* #33: Gph field evals that used the CMFGEN J-table */
    long n_te_table_pins = 0;   /* F3-T: shells whose T_e was pinned to the CMFGEN table */
    long n_pumpf_bl = 0, n_pumpf_fb = 0;  /* [PUMPF] line-Jb: blended vs cs-fallback count */
    long n_pumpf_bnu = 0;                  /* [PUMPF] FIX-2: zero-count bins routed to B_nu(Te) */
    /* ---- diagnostic / detailed-balance gates (parse once, print once) ----
     * Both default OFF: env unset => g_*<0 branch sets the gate to 0 with no
     * print, and every gated site below is skipped => byte-identical behavior.
     * Parsed in this single-threaded prologue (before the omp parallel region);
     * the per-shell loop only READS these statics. */
    static int    g_te_pin_on = -1;                 /* LUMINA_DIAG_TE_PIN */
    static int    g_te_pin_smin = 0, g_te_pin_smax = 0;
    static double g_te_pin_T0 = 0.0, g_te_pin_T1 = 0.0;
    static int    g_gph_sigma_cmfgen = -1;          /* LUMINA_GPH_SIGMA_CMFGEN */
    static double g_cmf_log_numin = 0.0, g_cmf_inv_dlognu = 0.0;
    static int    g_cmf_nfreq = 0;
    /* #33 GRADIENT-TRANSPLANT diagnostic (see loader block below). */
    static int    g_gph_jtable_on = -1;             /* LUMINA_GPH_JTABLE parse latch */
    static double *g_gph_jtable = NULL;             /* [n_shells*nfb] CMFGEN J or NULL */
    /* F3-T TEMPERATURE-TABLE probe (see loader block below). */
    static int    g_te_table_on = -1;               /* LUMINA_TE_TABLE parse latch */
    static double *g_te_table = NULL;               /* [n_shells] CMFGEN T_e(v) or NULL */
    if (g_te_pin_on < 0) {
        g_te_pin_on = 0;
        const char *e = getenv("LUMINA_DIAG_TE_PIN");
        if (e && *e) {
            int a = 0, b = 0; double t0 = 0.0, t1 = 0.0;
            if (sscanf(e, "%d:%d:%lf:%lf", &a, &b, &t0, &t1) == 4 && b >= a) {
                g_te_pin_on = 1;
                g_te_pin_smin = a; g_te_pin_smax = b;
                g_te_pin_T0 = t0;  g_te_pin_T1 = t1;
                printf("[TE-PIN] shells %d..%d T_e %g..%g K (diagnostic)\n",
                       a, b, t0, t1);
                fflush(stdout);
            }
        }
    }
    if (g_gph_sigma_cmfgen < 0) {
        g_gph_sigma_cmfgen = 0;
        const char *e = getenv("LUMINA_GPH_SIGMA_CMFGEN");
        if (e && atoi(e)) {
            g_gph_sigma_cmfgen = 1;
            g_cmf_nfreq = atom->cmfgen_n_freq_bins;
            if (atom->cmfgen_loaded && g_cmf_nfreq > 0 &&
                atom->cmfgen_nu_min > 0.0 &&
                atom->cmfgen_nu_max > atom->cmfgen_nu_min) {
                g_cmf_log_numin = log(atom->cmfgen_nu_min);
                g_cmf_inv_dlognu = (double)g_cmf_nfreq /
                                   (log(atom->cmfgen_nu_max) - g_cmf_log_numin);
            } else {
                g_cmf_inv_dlognu = 0.0;   /* no grid => Kramers fallback for all */
            }
            /* count photoionizing ions lacking a ground-state CMFGEN record
             * (they degrade to Kramers) — mirrors the Gph-loop pairing exactly */
            int nfb_fallback = 0;
            for (int e2 = 0; e2 < atom->n_elements; e2++) {
                int ip0c = atom->elem_ion_offset[e2];
                int ip1c = atom->elem_ion_offset[e2 + 1];
                int npopc = ip1c - ip0c;
                if (npopc < 2) continue;
                for (int jc = 0; jc < npopc - 1; jc++) {
                    int gl0 = atom->level_offset[ip0c + jc];
                    if (!(atom->cmfgen_loaded && atom->cmfgen_has_sigma &&
                          atom->cmfgen_has_sigma[gl0]))
                        nfb_fallback++;
                }
            }
            printf("[GPH-SIGMA] CMFGEN ground-state sigma_bf active "
                   "(%d ions fallback)\n", nfb_fallback);
            fflush(stdout);
        }
    }
    /* LUMINA_GPH_ALLLEVEL detailed-balance photoionization (see file-scope decl).
     * Parsed AFTER g_gph_sigma_cmfgen above so its resolved value + the CMFGEN
     * grid statics (g_cmf_nfreq/log_numin/inv_dlognu) are available. Only effective
     * with the per-level CMFGEN sigma_bf table; otherwise announced INACTIVE so a
     * requested-but-null run is never silent (env-chain verification). */
    /* [RATES-FIX] arm the master gate here, in the single-threaded prologue, so
     * the env read + banner happen once before the shell-parallel region. */
    (void)rates_fix_enabled();
    if (g_gph_alllevel < 0) {
        g_gph_alllevel = 0;
        const char *e = getenv("LUMINA_GPH_ALLLEVEL");
        if (e && atoi(e)) {
            if (g_gph_sigma_cmfgen && atom->cmfgen_loaded && g_cmf_nfreq > 0 &&
                g_cmf_inv_dlognu > 0.0) {
                g_gph_alllevel = 1;
                printf("[GPH-ALLLEVEL] simul photoionization = all-level "
                       "population-weighted (detailed-balance with Milne alpha)\n");
            } else {
                printf("[GPH-ALLLEVEL] requested but INACTIVE (needs "
                       "LUMINA_GPH_SIGMA_CMFGEN=1 + CMFGEN per-level sigma)\n");
            }
            fflush(stdout);
        }
    }
    /* LUMINA_GPH_ALLLEVEL_NLTE (see file-scope decl). Parsed AFTER g_gph_alllevel
     * so its resolved value is known: only effective when g_gph_alllevel is on;
     * requested-but-null announced INACTIVE (env-chain verification). */
    if (g_gph_alllevel_nlte < 0) {
        g_gph_alllevel_nlte = 0;
        const char *e = getenv("LUMINA_GPH_ALLLEVEL_NLTE");
        if (e && atoi(e)) {
            if (g_gph_alllevel) {
                g_gph_alllevel_nlte = 1;
                printf("[GPH-ALLLEVEL-NLTE] photoionization weights = actual NLTE "
                       "level populations (b_k) instead of Boltzmann@T_e\n");
            } else {
                printf("[GPH-ALLLEVEL-NLTE] requested but INACTIVE (needs "
                       "LUMINA_GPH_ALLLEVEL=1)\n");
            }
            fflush(stdout);
        }
    }
    /* [IONIZ-SELFTEST] parse once in this serial prologue (default OFF). */
    if (ioniz_selftest_mode()) {
        printf("[IONIZ-SELFTEST] mode=%d  (%s)  route: alllevel=%d alllevel_nlte=%d "
               "sigma_cmfgen=%d spingate=%s frozenin_dr=%s\n",
               g_ioniz_selftest,
               g_ioniz_selftest == 1 ? "J forced to B_nu(T_e): Gamma/alpha MUST equal Saha"
                                     : "live field, print only",
               g_gph_alllevel, g_gph_alllevel_nlte, g_gph_sigma_cmfgen,
               getenv("LUMINA_ALPHA_SPINGATE") ? getenv("LUMINA_ALPHA_SPINGATE") : "0",
               getenv("LUMINA_FROZENIN_DR") ? getenv("LUMINA_FROZENIN_DR") : "0");
        printf("[IONIZ-SELFTEST] %-4s %-9s %-4s %-5s %-11s %-11s %-11s %-11s %-9s %-11s %-9s\n",
               "s", "T_e", "Z", "stage", "Gamma_radeq", "alpha_Milne", "q=G/alpha",
               "q_saha", "ratio", "Gamma_cpl", "ratio_cpl");
        fflush(stdout);
    }
    /* #33 GRADIENT-TRANSPLANT loader (LUMINA_GPH_JTABLE=<path>, default OFF).
     * Reads the offline CMFGEN J_nu(v) table (scripts/build_cmfgen_jtable.py) built
     * onto THIS run's Gph frequency grid (n_shells x nfb; NLTE_NU_MIN..NLTE_NU_MAX,
     * bin center exp((bb+0.5)*d_log_nu)). When present, the three Gph field sites
     * below FULLY replace the mc-blend/nlte J with the table value for every
     * (shell,bin) whose table entry > 0 (a value of 0 = outside CMFGEN coverage =>
     * existing field kept). Env absent => g_gph_jtable stays NULL => every gated
     * site is skipped => byte-identical. Parsed in this single-threaded prologue;
     * the per-shell loop only READS the buffer (race-free). This is a surgical
     * ionization-causality probe: it does NOT touch thermal balance, line transfer,
     * or the MC/deterministic estimators -- ONLY the photoionization rate integral. */
    if (g_gph_jtable_on < 0) {
        g_gph_jtable_on = 0;
        const char *e = getenv("LUMINA_GPH_JTABLE");
        if (e && *e) {
            FILE *fp = fopen(e, "rb");
            if (!fp) {
                printf("[JTABLE] ERROR: cannot open '%s' -- gate INACTIVE\n", e);
            } else {
                int hdr[4] = {0, 0, 0, 0};   /* magic, version, nshells, nfb */
                size_t nr = fread(hdr, sizeof(int), 4, fp);
                if (nr != 4 || hdr[0] != 0x4A544142 /* 'JTAB' */) {
                    printf("[JTABLE] ERROR: bad header in '%s' (magic=0x%x) -- "
                           "INACTIVE\n", e, nr == 4 ? (unsigned)hdr[0] : 0u);
                    fclose(fp);
                } else if (hdr[2] != n_shells || hdr[3] != nfb) {
                    printf("[JTABLE] ERROR: grid mismatch '%s' shells=%d nfb=%d "
                           "(run has shells=%d nfb=%d) -- INACTIVE\n",
                           e, hdr[2], hdr[3], n_shells, nfb);
                    fclose(fp);
                } else {
                    size_t nt = (size_t)n_shells * nfb;
                    double *buf = (double *)malloc(nt * sizeof(double));
                    size_t got = buf ? fread(buf, sizeof(double), nt, fp) : 0;
                    fclose(fp);
                    if (!buf || got != nt) {
                        printf("[JTABLE] ERROR: short read '%s' (%zu/%zu) -- "
                               "INACTIVE\n", e, got, nt);
                        free(buf);
                    } else {
                        g_gph_jtable = buf;
                        /* liveness banner: freq-bin coverage + FUV(918-1290A) band
                         * geometric-mean at the two documented span shells s0,s8. */
                        double dln = log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)nfb;
                        long nz_bins = 0, nz_cells = 0;
                        double bs0 = 0.0, bs8 = 0.0; int cs0 = 0, cs8 = 0;
                        for (int bb = 0; bb < nfb; bb++) {
                            double nu = NLTE_NU_MIN * exp((bb + 0.5) * dln);
                            double lamA = 2.99792458e18 / nu;
                            int any = 0;
                            for (int s2 = 0; s2 < n_shells; s2++)
                                if (buf[(size_t)s2 * nfb + bb] > 0.0) { nz_cells++; any = 1; }
                            if (any) nz_bins++;
                            if (lamA >= 918.0 && lamA <= 1290.0) {
                                double v0 = buf[(size_t)0 * nfb + bb];
                                double v8 = (n_shells > 8) ? buf[(size_t)8 * nfb + bb] : 0.0;
                                if (v0 > 0.0) { bs0 += log(v0); cs0++; }
                                if (v8 > 0.0) { bs8 += log(v8); cs8++; }
                            }
                        }
                        printf("[JTABLE] loaded %s shells=%d nfb=%d nonzero_bins=%ld "
                               "(cells=%ld) J[s0,FUVband]=%.3e J[s8,FUVband]=%.3e\n",
                               e, n_shells, nfb, nz_bins, nz_cells,
                               cs0 ? exp(bs0 / cs0) : 0.0, cs8 ? exp(bs8 / cs8) : 0.0);
                    }
                }
            }
            fflush(stdout);
        }
    }
    /* F3-T TEMPERATURE-TABLE loader (LUMINA_TE_TABLE=<path>, default OFF).
     * Reads the offline CMFGEN T_e(v) CSV (scripts/build_cmfgen_te_table.py):
     * rows "shell_id,vel_mid_kms,T_e_K". When present, the commit site below
     * REPLACES the radeq-solved T_e with the table value for every shell whose
     * table entry > 0 -- a WHOLE-STATE pin: the pin is applied BEFORE simul_ladder
     * so the ion ladder + n_e solve at the table temperature, and the committed
     * plasma->T_e[s] (uploaded to the GPU for k-packet ff/fb, the coevolve
     * birth-Planck SED and collisional rates) carries it => every T_e consumer
     * inherits the pin. Env absent => g_te_table stays NULL => every gated site is
     * skipped => byte-identical. Parsed in this single-threaded prologue; the
     * per-shell loop only READS the buffer (race-free). */
    if (g_te_table_on < 0) {
        g_te_table_on = 0;
        const char *e = getenv("LUMINA_TE_TABLE");
        if (e && *e) {
            FILE *fp = fopen(e, "r");
            if (!fp) {
                printf("[TETAB] ERROR: cannot open '%s' -- gate INACTIVE\n", e);
            } else {
                double *buf = (double *)calloc((size_t)n_shells, sizeof(double));
                int nfilled = 0; char ln[256];
                while (buf && fgets(ln, sizeof(ln), fp)) {
                    if (ln[0] == '#' || ln[0] == '\n' || ln[0] == '\0') continue;
                    int s2; double v2, T2;
                    if (sscanf(ln, "%d,%lf,%lf", &s2, &v2, &T2) == 3 &&
                        s2 >= 0 && s2 < n_shells && T2 > 0.0) {
                        if (buf[s2] <= 0.0) nfilled++;
                        buf[s2] = T2;
                    }
                }
                fclose(fp);
                if (!buf || nfilled != n_shells) {
                    printf("[TETAB] ERROR: '%s' filled %d/%d shells -- INACTIVE\n",
                           e, nfilled, n_shells);
                    free(buf);
                } else {
                    g_te_table = buf;
                    printf("[TETAB] loaded %s shells=%d T[s0]=%.0f T[s8]=%.0f "
                           "T[s%d]=%.0f (WHOLE-STATE T_e pin ACTIVE)\n",
                           e, n_shells, buf[0], (n_shells > 8) ? buf[8] : 0.0,
                           n_shells - 1, buf[n_shells - 1]);
                }
            }
            fflush(stdout);
        }
    }
    fb_cool_kt_on();   /* [FB-COOL-KT] serial pre-init; simul_r1 only READS the static */
    /* [DBFB] LUMINA_RADEQ_DB_FB serial pre-init + detailed-balance self-check.
     * The bf net is  n*Sum_bb 4*pi*sigma f_above dnu * (J_bb - B_nu^Wien(T));  the
     * emit_bf partner is built from the SAME geom weight as Hex, and the Wien
     * partner B_nu^Wien(T)=(2 h nu^3/c^2) exp(-h nu/kT) is the exact Milne emission
     * SED in the Wien limit, so at J=B the heating integrand w*(h nu-chi) and the
     * cooling integrand emit_bf*exp(-h nu/kT) are the SAME product bin-by-bin =>
     * net=0 by construction.  Self-check reproduces that identity on the REAL nu
     * grid for a synthetic single-edge bf at 3 temperatures; any residual > 1e-6
     * signals a wiring defect (mismatched geom / bnu_pref / exponent) and aborts. */
    if (g_radeq_db_fb < 0) {
        const char *e = getenv("LUMINA_RADEQ_DB_FB");
        g_radeq_db_fb = (e && atoi(e)) ? 1 : 0;
        if (g_radeq_db_fb == 1) {
            double numin = nlte->nu_min, dln = nlte->d_log_nu;
            double nu0t  = numin * exp(0.35 * (double)nfb * dln);  /* mid-grid edge */
            double chit  = H_PLANCK * nu0t;
            double Ttest[3] = { 10000.0, 18000.0, 30000.0 };
            double worst = 0.0;
            for (int q = 0; q < 3; q++) {
                double T = Ttest[q], invkT = 1.0 / (K_BOLTZMANN * T);
                double Hh = 0.0, Cc = 0.0;
                for (int bb = 0; bb < nfb; bb++) {
                    double lo = log(numin) + bb * dln;
                    double nu = exp(lo + 0.5 * dln);
                    if (nu < nu0t) continue;
                    double dnu = exp(lo + dln) - exp(lo);
                    double sig = 7.91e-18 * (nu0t / nu) * (nu0t / nu) * (nu0t / nu);
                    double hnu = H_PLANCK * nu;
                    double bnu_pref = 2.0 * H_PLANCK * nu * nu * nu /
                                      (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                    double geom = 4.0 * M_PI_VAL * sig * dnu / hnu * (hnu - chit);
                    double J = bnu_pref * exp(-hnu * invkT);   /* J := B_nu^Wien(T) */
                    double w = 4.0 * M_PI_VAL * sig * J / hnu * dnu; /* heating rate */
                    Hh += w * (hnu - chit);                    /* heating integrand */
                    Cc += (geom * bnu_pref) * exp(-hnu * invkT);/* emit_bf * wien   */
                }
                double res = (Hh > 0.0) ? fabs(Hh - Cc) / Hh : 0.0;
                if (res > worst) worst = res;
                printf("[DBFB] selfcheck s0: T=%.0f  net(J=B)/H = %.2e\n", T, res);
            }
            printf("[DBFB] selfcheck s0: net(J=B)/H = %.1e (worst over T=10/18/30kK)\n",
                   worst);
            if (worst > 1e-6) {
                fprintf(stderr, "[DBFB][FATAL] detailed-balance self-check FAILED: "
                        "worst net(J=B)/H = %.3e > 1e-6 -- bf heating and cooling "
                        "are NOT a detailed-balance pair; aborting.\n", worst);
                fflush(stderr);
                exit(1);
            }
            printf("[DBFB] LUMINA_RADEQ_DB_FB=1: simul_r1 bf cooling = emit_nu/Wien "
                   "detailed-balance partner of H_photo (replaces analytic C_fb)\n");
            fflush(stdout);
        }
    }
    /* [PUMPF] LUMINA_RADEQ_PUMP_FIELD serial pre-init (alongside DBFB). Default
     * OFF => the line-build Jb stays the hard-wired cs_J => byte-identical. When
     * ON, simul_line_term's per-line Jb reads the SAME alpha-blended field the
     * Gph/Hex photoion loop consumes -- unifying the split-field radeq pump
     * (te_bias_budget C2). alpha is the live g_photoion_mc_alpha (set by the
     * co-evolve consume setter from LUMINA_COEVOLVE_PHOTOION_ALPHA); before the
     * first transport pass the MC field is NULL and every line falls back to cs_J
     * (the SAME startup behaviour the Gph guard has). */
    if (g_radeq_pump_field < 0) {
        const char *e = getenv("LUMINA_RADEQ_PUMP_FIELD");
        g_radeq_pump_field = (e && atoi(e)) ? 1 : 0;
        if (g_radeq_pump_field == 1) {
            double a_disp = g_photoion_mc_alpha;   /* live blend weight if already armed */
            if (!g_photoion_mc_J) {                /* not yet armed: show configured alpha */
                const char *ae = getenv("LUMINA_COEVOLVE_PHOTOION_ALPHA");
                a_disp = ae ? atof(ae) : 0.5;
            }
            printf("[PUMPF] LUMINA_RADEQ_PUMP_FIELD=1: simul_line_term Jb = "
                   "alpha-blend (alpha=%.2f) -- split-field pump residue unified\n",
                   a_disp);
            if (!g_photoion_mc_J)
                printf("[PUMPF] mc_J not yet armed at parse (pre-transport) -> Jb "
                       "falls back to cs_J until first co-evolve consume pass\n");
            fflush(stdout);
        }
    }
    /* FIX-2 [PUMPF fallback] pre-init (alongside PUMP_FIELD). Only takes effect
     * when PUMP_FIELD=1 (the sole path through radeq_pump_line_Jb). */
    if (g_radeq_pump_fallback < 0) {
        const char *e = getenv("LUMINA_RADEQ_PUMP_FALLBACK");
        g_radeq_pump_fallback = (e && atoi(e)) ? 1 : 0;
        if (g_radeq_pump_fallback == 1)
            printf("[PUMPF] LUMINA_RADEQ_PUMP_FALLBACK=1: zero-count mc bins -> "
                   "B_nu(T_e) (thermal) instead of super-thermal cs_J\n");
        fflush(stdout);
    }
#ifdef _OPENMP
    if (g_simul_nested == 0) {
        const char *e = getenv("LUMINA_SIMUL_NESTED");
        g_simul_nested = e ? atoi(e) : -1;          /* -1 = parsed, off */
        const char *en = getenv("LUMINA_SIMUL_NESTED_NL");
        if (en) g_simul_nested_nl = atol(en);
        if (g_simul_nested > 1) {
            omp_set_max_active_levels(2);
            printf("[SIMUL-LB] nested line-sum: %d inner threads for shells "
                   "with nl>=%ld\n", g_simul_nested, g_simul_nested_nl);
        }
    }
    #pragma omp parallel reduction(+:n_pin_hi,n_pin_lo,n_jtable_evals,n_te_table_pins,n_pumpf_bl,n_pumpf_fb)
#endif
    {
    SimShell sh;
    memset(&sh, 0, sizeof(sh));
    sh.nion   = (double *)calloc((size_t)n_ip, sizeof(double));
    long cap  = radeq_n_lines > 0 ? radeq_n_lines : 1;
    sh.l_ip   = (int *)malloc(cap * sizeof(int));
    sh.l_dE   = (double *)malloc(cap * sizeof(double));
    sh.l_beta = (double *)malloc(cap * sizeof(double));
    sh.l_coeff= (double *)malloc(cap * sizeof(double));
    sh.l_glo  = (double *)malloc(cap * sizeof(double));
    sh.l_gup  = (double *)malloc(cap * sizeof(double));
    sh.l_Elo  = (double *)malloc(cap * sizeof(double));
    sh.l_BluJ = (double *)malloc(cap * sizeof(double));
    sh.l_ABulJ= (double *)malloc(cap * sizeof(double));
    sh.l_ftau = (double *)malloc(cap * sizeof(double));
    /* [DBFB] per-pair per-bin bf-emission spectrum + Wien exponent grid (only when
     * the gate is on; else NULL, never touched => byte-identical). */
    sh.nfb = nfb;
    /* withParityO: per-shell CMFGEN COL pair prefactors (thread-private, reused
     * across shells). Allocated only when the gate is armed => byte-identical off. */
    sh.cp_a = sh.cp_b = sh.cp_beta = NULL; sh.cp_n = 0;
    if (g_cp_on == 1 && g_cp_nions > 0) {
        long per = (long)g_cp_maxlev * (g_cp_maxlev - 1) / 2;
        size_t cp_cap = (size_t)g_cp_nions * (size_t)(per > 0 ? per : 1);
        sh.cp_a    = (double *)malloc(cp_cap * sizeof(double));
        sh.cp_b    = (double *)malloc(cp_cap * sizeof(double));
        sh.cp_beta = (double *)malloc(cp_cap * sizeof(double));
    }
    sh.emit_bf = NULL; sh.emit_bx = NULL;
    if (g_radeq_db_fb == 1) {
        sh.emit_bf = (double *)malloc((size_t)SIM_MAXP * (size_t)nfb * sizeof(double));
        sh.emit_bx = (double *)malloc((size_t)nfb * sizeof(double));
        for (int bb = 0; bb < nfb; bb++) {
            double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
            double nu_mid = exp(lo + 0.5 * nlte->d_log_nu);
            sh.emit_bx[bb] = H_PLANCK * nu_mid / K_BOLTZMANN;   /* [K] */
        }
    }
#ifdef _OPENMP
    #pragma omp for schedule(dynamic, 1)
#endif
    for (int s = 0; s < n_shells; s++) {
        /* sh is REUSED across shells within a thread: stale other-element ion
         * densities from a previously-processed (inner, Fe-rich) shell leaked
         * into the commit of Fe-free outer shells -> n_e 60x charge-conservation
         * violation at s=37-42 -> the hard-UV lamp that erased the outer root. */
        memset(sh.nion, 0, (size_t)n_ip * sizeof(double));
        if (g_radeq_db_fb == 1)   /* [DBFB] clear the per-pair emission spectrum */
            memset(sh.emit_bf, 0, (size_t)SIM_MAXP * (size_t)nfb * sizeof(double));
        double T_rad = plasma->T_e[s];
        /* ---- shell constants ---- */
        sh.H_dep = (gamma_dep && gamma_dep->heating_rate) ?
                   gamma_dep->heating_rate[s] : 0.0;
        sh.ff_pref = 1.426e-27 * 1.2;
        sh.Gamma_ad = 3.0 / time_explosion;
        /* ---- pairs: photoion integrals from the lagged binned J ---- */
        sh.np = 0; sh.natom = 0.0;
        for (int e = 0; e < atom->n_elements; e++) {
            int ip0 = atom->elem_ion_offset[e], ip1 = atom->elem_ion_offset[e + 1];
            int npop = ip1 - ip0;
            double nel = atom->abundances[e * n_shells + s] * plasma->rho[s] /
                         (atom->element_mass_amu[e] * AMU);
            if (nel <= 0.0) continue;
            sh.natom += nel;
            if (npop < 2) { sh.nion[ip0] = nel; continue; }
            if (sh.np + npop - 1 > SIM_MAXP) continue;
            for (int j = 0; j < npop - 1; j++) {
                int p = sh.np + j;
                sh.ipa[p] = ip0 + j;
                double chi_eV = find_ioniz_energy(atom, atom->ion_pop_Z[ip0 + j],
                                                  atom->ion_pop_stage[ip0 + j]);
                sh.chi[p] = chi_eV * EV_TO_ERG;
                double nu0 = sh.chi[p] / H_PLANCK;
                int zeff = atom->ion_pop_stage[ip0 + j] + 1;
                /* LUMINA_GPH_SIGMA_CMFGEN: detailed-balance-consistent bf cross
                 * section. The recombination side (frozenin_alpha_rr) integrates
                 * the REAL per-level CMFGEN sigma_bf; the Kramers sigma below
                 * breaks detailed balance. When gated on, swap in this ion's
                 * GROUND-LEVEL sigma_bf row (same table, same freq grid the Milne
                 * integral reads). Gate off => gnd_sig NULL => Kramers unchanged. */
                const double *gnd_sig = NULL;
                if (g_gph_sigma_cmfgen && atom->cmfgen_loaded) {
                    int gl0 = atom->level_offset[ip0 + j];
                    if (atom->cmfgen_has_sigma && atom->cmfgen_has_sigma[gl0])
                        gnd_sig = &atom->cmfgen_sigma_bf[
                                     (size_t)gl0 * (size_t)g_cmf_nfreq];
                }
                double G = 0.0, Hx = 0.0;
                if (g_gph_alllevel) {
                    /* Detailed-balance: sum photoionization over ALL levels of
                     * ion (ip0+j) with Boltzmann population weights, mirroring the
                     * all-level Milne alpha in frozenin_alpha_rr. T_e is this
                     * shell's lagged value (Gph is built once, outside the T
                     * bisection) -- intended approximation. */
                    int gl0 = atom->level_offset[ip0 + j];
                    int gl1 = atom->level_offset[ip0 + j + 1];
                    double kT = K_BOLTZMANN * plasma->T_e[s];
                    /* [RATES-FIX F1] the two x_l>=50 level skips below are the
                     * Gamma-side half of a cut that alpha (frozenin_alpha_rr)
                     * does NOT apply: per level, Gamma weights g e^{-E/kT} I and
                     * alpha weights g e^{(chi-E)/kT} I -- the SAME relative
                     * share. Cutting Gamma alone makes it systematically low
                     * (Si III @5000K: 8.3x). exp() underflows to 0 past x~745
                     * on its own, so no cut is needed at all. */
                    const int rfix_gph = rates_fix_enabled();
                    double U_ion = 0.0;                 /* lower-ion partition fn */
                    for (int l = gl0; l < gl1; l++) {
                        double x = atom->level_energy_eV[l] * EV_TO_ERG / kT;
                        if (x < 50.0) U_ion += (double)atom->level_g[l] * exp(-x);
                    }
                    if (!(U_ion >= 1.0)) U_ion = 1.0;
                    /* LUMINA_GPH_ALLLEVEL_NLTE: resolve this lower ion's NLTE ion
                     * index. If present with a positive summed population, weight
                     * each level by its ACTUAL population fraction (n_level/n_ion)
                     * over the ion's NLTE levels; else fall through to the Boltzmann
                     * path below unchanged (non-NLTE ions stay LTE-weighted). */
                    int use_nlte = 0, off_i = 0, off_i1 = 0;
                    double n_ion_nlte = 0.0;
                    /* ADDENDUM (Fork A): the III combs of promoted pairs were
                     * LTE-weighted (b_k=1) here, making Gph(III) ~22x low because
                     * their real departures are b_k=800-5000. Naively NLTE-
                     * weighting ALL top-ion combs over-corrects (super-thermal
                     * inflation for lack of a continuum drain); but stage-IV
                     * promotion GIVES the III comb an SE drain (IV in the NLTE
                     * set). So under LUMINA_NLTE_STAGE4, additionally NLTE-weight
                     * a III comb (stage==2) iff its IV (stage 3) is now an NLTE
                     * ion = "newly drained". Scoped to exactly the promoted III
                     * combs; non-drained top ions stay LTE-weighted. Gate OFF or
                     * g_gph_alllevel_nlte off => byte-identical (want_nlte_w=0). */
                    /* [STAGE4-R2 A1] depth-gate + b_k cap for the promoted III-comb
                     * NLTE weighting. WTHR (LUMINA_STAGE4_GPH_WTHR, default 0.13):
                     * NLTE-weight ONLY where W(s)>WTHR (deep/continuum-thick, CMFGEN
                     * f(IV)>LTE); photospheric shells fall through to the Boltzmann
                     * path below (already CMFGEN-correct — round-1 blew them up).
                     * BK_CAP (LUMINA_STAGE4_BK_CAP, default 1000): per-level
                     * departure clamp applied in the use_nlte loop below. Read once. */
                    static double g_stage4_gph_wthr = -1.0;
                    static double g_stage4_bk_cap   = -1.0;
                    if (g_stage4_gph_wthr < 0.0) {
                        const char *e = getenv("LUMINA_STAGE4_GPH_WTHR");
                        g_stage4_gph_wthr = e ? atof(e) : 0.13;
                        if (g_stage4_gph_wthr < 0.0) g_stage4_gph_wthr = 0.0;
                    }
                    if (g_stage4_bk_cap < 0.0) {
                        const char *e = getenv("LUMINA_STAGE4_BK_CAP");
                        g_stage4_bk_cap = e ? atof(e) : 1000.0;
                        if (g_stage4_bk_cap < 0.0) g_stage4_bk_cap = 0.0;
                    }
                    int want_nlte_w = g_gph_alllevel_nlte;
                    if (!want_nlte_w && nlte_stage4_enabled() &&
                        atom->ion_pop_stage[ip0 + j] == 2 &&
                        nlte_get_J_at_nu(nlte, s, nlte->nu_min) >
                            g_stage4_gph_wthr) {   /* diagnostic depth gate */
                        int Zc = atom->ion_pop_Z[ip0 + j];
                        for (int i = 0; i < nlte->n_nlte_ions; i++)
                            if (nlte->nlte_Z[i] == Zc && nlte->nlte_ion[i] == 3) {
                                want_nlte_w = 1; break;
                            }
                    }
                    if (want_nlte_w) {
                        int Zc = atom->ion_pop_Z[ip0 + j];
                        int stc = atom->ion_pop_stage[ip0 + j];
                        for (int i = 0; i < nlte->n_nlte_ions; i++) {
                            if (nlte->nlte_Z[i] == Zc && nlte->nlte_ion[i] == stc) {
                                off_i  = nlte->nlte_ion_level_offset[i];
                                off_i1 = nlte->nlte_ion_level_offset[i + 1];
                                for (int ln = off_i; ln < off_i1; ln++)
                                    n_ion_nlte += nlte->nlte_level_populations[
                                                    (size_t)ln * n_shells + s];
                                if (n_ion_nlte > 0.0) use_nlte = 1;
                                break;
                            }
                        }
                    }
                    /* one-time detailed-balance ratio for a representative IGE ion
                     * (Fe III = Z26 stage2). Only the single s==0 thread writes the
                     * flag, so the static is race-free for correctness. */
                    static int diag_done = 0;
                    int want_diag = (!diag_done && s == 0 &&
                                     atom->ion_pop_Z[ip0 + j] == 26 &&
                                     atom->ion_pop_stage[ip0 + j] == 2);
                    double G_gnd_diag = 0.0;
                    double G_boltz_diag = 0.0;      /* NLTE diag: G w/ Boltzmann wts */
                    if (use_nlte) {
                        /* NLTE-weighted all-level path: iterate the ion's NLTE
                         * levels, map to the global level index, weight by the
                         * actual population fraction. Inner freq loop identical. */
                        for (int ln = off_i; ln < off_i1; ln++) {
                            int l = nlte->nlte_to_global_level[ln];
                            if (!(atom->cmfgen_has_sigma && atom->cmfgen_has_sigma[l]))
                                continue;
                            double E_l = atom->level_energy_eV[l] * EV_TO_ERG;
                            double chi_l = sh.chi[p] - E_l;   /* binding energy of l */
                            if (chi_l <= 0.0) continue;
                            double nu_l = chi_l / H_PLANCK;    /* level threshold */
                            double x_l = E_l / kT;
                            /* [RATES-FIX F1] (here the weight is the ACTUAL NLTE
                             * population, so a Boltzmann-energy cut is doubly wrong) */
                            if (!rfix_gph && x_l >= 50.0) continue;
                            double pop_l = nlte->nlte_level_populations[
                                            (size_t)ln * n_shells + s] / n_ion_nlte;
                            if (pop_l <= 0.0) continue;
                            /* Boltzmann weight for the SAME level (diagnostic only) */
                            double pop_l_boltz = (double)atom->level_g[l] *
                                                 exp(-x_l) / U_ion;
                            /* [STAGE4-R2 A1] cap the departure b_l = pop_l/pop_l_boltz
                             * at BK_CAP: pop_l -> min(pop_l, BK_CAP*pop_l^LTE). Bounds
                             * the pathological super-thermal comb (esp. Ni III, s8
                             * b_k~2e9) without touching the deep physical drain (s0
                             * comb << cap). stage4-only; cap<=0 => no-op. */
                            if (g_stage4_bk_cap > 0.0 && nlte_stage4_enabled() &&
                                !nlte_element_wide_matches(
                                    atom->ion_pop_Z[ip0 + j], s) &&
                                pop_l_boltz > 0.0) {
                                double cap_pop = g_stage4_bk_cap * pop_l_boltz;
                                if (pop_l > cap_pop) pop_l = cap_pop;
                            }
                            const double *sig_row_l = &atom->cmfgen_sigma_bf[
                                            (size_t)l * (size_t)g_cmf_nfreq];
                            for (int bb = 0; bb < nfb; bb++) {
                                double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                                double nu = exp(lo + 0.5 * nlte->d_log_nu);
                                if (nu < nu_l) continue;
                                int bc = (int)((log(nu) - g_cmf_log_numin) *
                                               g_cmf_inv_dlognu);
                                if (bc < 0 || bc >= g_cmf_nfreq) continue;
                                double sig = sig_row_l[bc];
                                if (sig <= 0.0) continue;
                                double dnu = exp(lo + nlte->d_log_nu) - exp(lo);
                                if (g_radeq_db_fb == 1) {   /* [DBFB] emission partner: build BEFORE
                                     * the J<=0 skip so ALL above-threshold sig>0 bins are covered
                                     * (exact bin-by-bin cancellation at J=B_nu^Wien(T)). */
                                    double hnu = H_PLANCK * nu;
                                    double bnu_pref = 2.0 * H_PLANCK * nu * nu * nu /
                                                      (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                                    sh.emit_bf[(size_t)p * nfb + bb] += pop_l *
                                        (4.0 * M_PI_VAL * sig * dnu / hnu * (hnu - chi_l)) * bnu_pref;
                                }
                                double J = nlte->J_nu[(size_t)s * nfb + bb];
                                /* same P1 MC-shadow blend as the ground path below */
                                if (g_photoion_mc_J && s < g_photoion_mc_nshells &&
                                    g_photoion_mc_nfb == nfb &&
                                    !(g_photoion_mc_occ && g_photoion_mc_count &&
                                      g_photoion_mc_count[(size_t)s * nfb + bb] == 0))
                                    J = g_photoion_mc_alpha *
                                            g_photoion_mc_J[(size_t)s * nfb + bb]
                                        + (1.0 - g_photoion_mc_alpha) * J;
                                /* #33 GRADIENT-TRANSPLANT: full override with the
                                 * external CMFGEN J-table (0 => keep field above). */
                                if (g_gph_jtable) {
                                    double Jtab = g_gph_jtable[(size_t)s * nfb + bb];
                                    if (Jtab > 0.0) { J = Jtab; n_jtable_evals++; }
                                }
                                /* [IONIZ-SELFTEST] known-answer field: J -> B_nu(T_e). */
                                if (g_ioniz_selftest == 1) J = planck_bnu(plasma->T_e[s], nu);
                                if (J <= 0.0) continue;
                                double w = 4.0 * M_PI_VAL * sig * J /
                                           (H_PLANCK * nu) * dnu;
                                G  += pop_l * w;
                                /* excess energy above THIS level's threshold chi_l */
                                Hx += pop_l * w * (H_PLANCK * nu - chi_l);
                                if (want_diag && l == gl0) G_gnd_diag += w;
                                if (want_diag) G_boltz_diag += pop_l_boltz * w;
                            }
                        }
                    } else
                    for (int l = gl0; l < gl1; l++) {
                        if (!(atom->cmfgen_has_sigma && atom->cmfgen_has_sigma[l]))
                            continue;
                        double E_l = atom->level_energy_eV[l] * EV_TO_ERG;
                        double chi_l = sh.chi[p] - E_l;   /* binding energy of l */
                        if (chi_l <= 0.0) continue;
                        double nu_l = chi_l / H_PLANCK;    /* level threshold */
                        double x_l = E_l / kT;
                        if (!rfix_gph && x_l >= 50.0) continue;   /* [RATES-FIX F1] */
                        double pop_l = (double)atom->level_g[l] * exp(-x_l) / U_ion;
                        if (pop_l <= 0.0) continue;
                        const double *sig_row_l = &atom->cmfgen_sigma_bf[
                                        (size_t)l * (size_t)g_cmf_nfreq];
                        for (int bb = 0; bb < nfb; bb++) {
                            double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                            double nu = exp(lo + 0.5 * nlte->d_log_nu);
                            if (nu < nu_l) continue;
                            int bc = (int)((log(nu) - g_cmf_log_numin) *
                                           g_cmf_inv_dlognu);
                            if (bc < 0 || bc >= g_cmf_nfreq) continue;
                            double sig = sig_row_l[bc];
                            if (sig <= 0.0) continue;
                            double dnu = exp(lo + nlte->d_log_nu) - exp(lo);
                            if (g_radeq_db_fb == 1) {   /* [DBFB] emission partner (see NLTE path) */
                                double hnu = H_PLANCK * nu;
                                double bnu_pref = 2.0 * H_PLANCK * nu * nu * nu /
                                                  (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                                sh.emit_bf[(size_t)p * nfb + bb] += pop_l *
                                    (4.0 * M_PI_VAL * sig * dnu / hnu * (hnu - chi_l)) * bnu_pref;
                            }
                            double J = nlte->J_nu[(size_t)s * nfb + bb];
                            /* same P1 MC-shadow blend as the ground path below */
                            if (g_photoion_mc_J && s < g_photoion_mc_nshells &&
                                g_photoion_mc_nfb == nfb &&
                                !(g_photoion_mc_occ && g_photoion_mc_count &&
                                  g_photoion_mc_count[(size_t)s * nfb + bb] == 0))
                                J = g_photoion_mc_alpha *
                                        g_photoion_mc_J[(size_t)s * nfb + bb]
                                    + (1.0 - g_photoion_mc_alpha) * J;
                            /* #33 GRADIENT-TRANSPLANT: full override with the
                             * external CMFGEN J-table (0 => keep field above). */
                            if (g_gph_jtable) {
                                double Jtab = g_gph_jtable[(size_t)s * nfb + bb];
                                if (Jtab > 0.0) { J = Jtab; n_jtable_evals++; }
                            }
                            /* [IONIZ-SELFTEST] known-answer field: J -> B_nu(T_e). */
                            if (g_ioniz_selftest == 1) J = planck_bnu(plasma->T_e[s], nu);
                            if (J <= 0.0) continue;
                            double w = 4.0 * M_PI_VAL * sig * J /
                                       (H_PLANCK * nu) * dnu;
                            G  += pop_l * w;
                            /* excess energy above THIS level's threshold chi_l */
                            Hx += pop_l * w * (H_PLANCK * nu - chi_l);
                            if (want_diag && l == gl0) G_gnd_diag += w;
                        }
                    }
                    if (want_diag) {
                        printf("[GPH-ALLLEVEL] s0 Fe III: G_all/G_gnd = %.3g "
                               "(G_all=%.3e G_gnd=%.3e U=%.2f nlev=%d)\n",
                               G_gnd_diag > 0.0 ? G / G_gnd_diag : -1.0,
                               G, G_gnd_diag, U_ion, gl1 - gl0);
                        if (g_gph_alllevel_nlte) {
                            printf("[GPH-ALLLEVEL-NLTE] s0 Fe III: G_nlte/G_boltz "
                                   "= %.3g (n_ion_nlte=%.3e)\n",
                                   G_boltz_diag > 0.0 ? G / G_boltz_diag : -1.0,
                                   n_ion_nlte);
                        }
                        fflush(stdout);
                        diag_done = 1;
                    }
                }
                else
                for (int bb = 0; bb < nfb; bb++) {
                    double lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                    double nu = exp(lo + 0.5 * nlte->d_log_nu);
                    if (nu < nu0) continue;
                    double dnu = exp(lo + nlte->d_log_nu) - exp(lo);
                    double sig = 7.91e-18 / ((double)zeff * zeff) *
                                 (nu0 / nu) * (nu0 / nu) * (nu0 / nu);
                    if (gnd_sig) {
                        /* map this nlte bin's nu onto the CMFGEN sigma grid;
                         * out-of-coverage or zero record => keep Kramers here */
                        int bc = (int)((log(nu) - g_cmf_log_numin) *
                                       g_cmf_inv_dlognu);
                        if (bc >= 0 && bc < g_cmf_nfreq && gnd_sig[bc] > 0.0)
                            sig = gnd_sig[bc];
                    }
                    if (g_radeq_db_fb == 1) {   /* [DBFB] emission partner (ground, pop=1;
                         * chi = sh.chi[p]; built BEFORE the J<=0 skip). */
                        double hnu = H_PLANCK * nu;
                        double bnu_pref = 2.0 * H_PLANCK * nu * nu * nu /
                                          (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                        sh.emit_bf[(size_t)p * nfb + bb] +=
                            (4.0 * M_PI_VAL * sig * dnu / hnu * (hnu - sh.chi[p])) * bnu_pref;
                    }
                    double J = nlte->J_nu[(size_t)s * nfb + bb];
                    /* P1 (LIVE site): soften the too-blue deterministic J with the
                     * lagged, transported MC shadow continuum before it drives the
                     * photoion rate — the actual S/Si over-ionization lever in the
                     * RADEQ_SIMUL config. Gate off (setter never called) => NULL =>
                     * byte-identical. Symmetric to the bb jbar_line rewiring. */
                    if (g_photoion_mc_J && s < g_photoion_mc_nshells &&
                        g_photoion_mc_nfb == nfb &&
                        /* unsampled-bin fallback, zero-count = statistical fact (NO-OVERFITTING) */
                        !(g_photoion_mc_occ && g_photoion_mc_count &&
                          g_photoion_mc_count[(size_t)s * nfb + bb] == 0))
                        J = g_photoion_mc_alpha * g_photoion_mc_J[(size_t)s * nfb + bb]
                            + (1.0 - g_photoion_mc_alpha) * J;
                    /* #33 GRADIENT-TRANSPLANT: full override with the external
                     * CMFGEN J-table (0 => keep field above). This ground/Kramers
                     * site is the ELSE of g_gph_alllevel; in the a10_kx (all-level)
                     * config only the two all-level sites above fire. */
                    if (g_gph_jtable) {
                        double Jtab = g_gph_jtable[(size_t)s * nfb + bb];
                        if (Jtab > 0.0) { J = Jtab; n_jtable_evals++; }
                    }
                    /* [IONIZ-SELFTEST] known-answer field: J -> B_nu(T_e). */
                    if (g_ioniz_selftest == 1) J = planck_bnu(plasma->T_e[s], nu);
                    if (J <= 0.0) continue;
                    double w = 4.0 * M_PI_VAL * sig * J / (H_PLANCK * nu) * dnu;
                    G += w;
                    Hx += w * (H_PLANCK * nu - sh.chi[p]);
                }
                sh.Gph[p] = G; sh.Hex[p] = Hx;
                /* [IONIZ-SELFTEST] detailed-balance identity for this pair. Both
                 * halves are the PRODUCTION ones: G above (radeq Gph route) and
                 * frozenin_alpha_rr (the alpha simul_ladder divides by).  Also
                 * evaluates the NLTE-route Gamma (coupled_photoion_rate_jnu) on
                 * the same field for a side-by-side. Gate off => never runs. */
                if (g_ioniz_selftest) {
                    int ipl = ip0 + j;
                    double Ts = plasma->T_e[s];
                    double alpha = frozenin_alpha_rr(atom, ipl, ipl + 1, Ts);
                    double Ul = ioniz_selftest_U(atom, ipl, Ts, 0);
                    double Uu = ioniz_selftest_U(atom, ipl + 1, Ts, 1);
                    /* reciprocal of the EXACT lam3 expression frozenin_alpha_rr uses */
                    double inv_lam3 = 1.0 / pow(H_PLANCK * H_PLANCK /
                        (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * Ts), 1.5);
                    double xchi = sh.chi[p] / (K_BOLTZMANN * Ts);
                    double q_saha = (xchi < 700.0)
                        ? 2.0 * (Uu / Ul) * inv_lam3 * exp(-xchi) : 0.0;
                    double q = (alpha > 0.0) ? G / alpha : -1.0;
                    double Gc = coupled_photoion_rate_jnu(atom, nlte, ipl, s, Ts,
                                                          n_shells, NULL, NULL, 0.0,
                                                          0.0, NULL);
                    double qc = (alpha > 0.0 && Gc >= 0.0) ? Gc / alpha : -1.0;
#ifdef _OPENMP
                    #pragma omp critical
#endif
                    {
                    printf("[IONIZ-SELFTEST] %-4d %-9.0f %-4d %-5d %-11.4e %-11.4e "
                           "%-11.4e %-11.4e %-9.4f %-11.4e %-9.4f\n",
                           s, Ts, atom->ion_pop_Z[ipl], atom->ion_pop_stage[ipl],
                           G, alpha, q, q_saha,
                           (q_saha > 0.0 && q >= 0.0) ? q / q_saha : -1.0,
                           Gc, (q_saha > 0.0 && qc >= 0.0) ? qc / q_saha : -1.0);
                    /* [RATES-FIX] the %.4f ratio column above only resolves
                     * 5e-5; the acceptance criterion is 1e-6, so emit the
                     * signed deviations at full precision. Gate off => not
                     * printed => the pre-fix log is byte-identical. */
                    if (rates_fix_enabled())
                        printf("[RATES-FIX-DEV] s=%d T=%.0f Z=%d stage=%d "
                               "dev=%.6e dev_cpl=%.6e\n",
                               s, Ts, atom->ion_pop_Z[ipl],
                               atom->ion_pop_stage[ipl],
                               (q_saha > 0.0 && q >= 0.0) ? q / q_saha - 1.0 : -1.0,
                               (q_saha > 0.0 && qc >= 0.0) ? qc / q_saha - 1.0 : -1.0);
                    fflush(stdout);
                    }
                }
                sh.nelem[p] = nel; sh.npops[p] = npop;
            }
            sh.np += npop - 1;
        }
        sh.gnt = 0.0;
        if (g_nt_ioniz_rate && s < g_nt_ioniz_n && sh.natom > 0.0)
            sh.gnt = g_nt_ioniz_rate[s] / sh.natom;
        /* Per-stage NT suppression (Lotz sigma ~ 1/chi^2 + secondary-spectrum
         * softening): Gamma_nt,j = gnt*(35eV/chi_j)^p, p = LUMINA_SIMUL_NTP
         * (default 2; calculator-validated). Load-bearing for STABILITY: with
         * the constant per-atom rate, any J-collapsed iteration strips every
         * stage at every trial T (uniform Gamma_nt >> n_e*alpha) -> no coolant
         * ions -> the ~40kK root vanishes -> 140kK ratchet (fix13 flight 3). */
        { static double sim_ntp = -1.0;
          if (sim_ntp < 0.0) { const char *e = getenv("LUMINA_SIMUL_NTP");
                               sim_ntp = e ? atof(e) : 2.0; }
          for (int p2 = 0; p2 < sh.np; p2++) {
              double chi_eV = sh.chi[p2] / EV_TO_ERG;
              double fac = pow(35.0 / (chi_eV > 1.0 ? chi_eV : 1.0), sim_ntp);
              if (fac > 1.0) fac = 1.0;   /* suppress high-chi only */
              sh.gnt_p[p2] = sh.gnt * fac;
          } }
        /* [IONIZ-SELFTEST] end-to-end ladder closure: run the PRODUCTION
         * simul_ladder on the Gph just built, at the shell's T_e, and compare the
         * ion fractions it returns with the analytic Saha ladder at the SAME T and
         * the SAME converged n_e. sh.nion is scratch that the real solve rewrites
         * below, so this probe is state-free. Gate off => never runs. */
        if (g_ioniz_selftest) {
            double Ts = plasma->T_e[s];
            double ne_p = (plasma->n_electron && plasma->n_electron[s] > 0.0)
                          ? plasma->n_electron[s] : 0.5 * sh.natom;
            simul_ladder(atom, &sh, Ts, &ne_p);
            double worst = 0.0; int worst_ip = -1;
            double ne_chk = 0.0;
            for (int p2 = 0; p2 < sh.np; ) {
                int npop2 = sh.npops[p2], ip0b = sh.ipa[p2];
                double nel2 = sh.nelem[p2];
                double ys[SIM_MAXP + 1]; ys[0] = 1.0; double ysum2 = 1.0;
                for (int jj = 0; jj < npop2 - 1; jj++) {
                    int ipl = ip0b + jj;
                    double Ul = ioniz_selftest_U(atom, ipl, Ts, 0);
                    double Uu = ioniz_selftest_U(atom, ipl + 1, Ts, 1);
                    double inv_lam3 = 1.0 / pow(H_PLANCK * H_PLANCK /
                        (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * Ts), 1.5);
                    double xchi = sh.chi[p2 + jj] / (K_BOLTZMANN * Ts);
                    double qs = (xchi < 700.0)
                        ? 2.0 * (Uu / Ul) * inv_lam3 * exp(-xchi) : 0.0;
                    ys[jj + 1] = ys[jj] * qs / ne_p;
                    if (!isfinite(ys[jj + 1])) ys[jj + 1] = 0.0;
                }
                ysum2 = 0.0; for (int jj = 0; jj < npop2; jj++) ysum2 += ys[jj];
                for (int jj = 0; jj < npop2; jj++) {
                    double f_saha = (ysum2 > 0.0) ? ys[jj] / ysum2 : (jj == 0 ? 1.0 : 0.0);
                    double f_code = (nel2 > 0.0) ? sh.nion[ip0b + jj] / nel2 : 0.0;
                    ne_chk += nel2 * (double)atom->ion_pop_stage[ip0b + jj] * f_code;
                    double d = fabs(f_code - f_saha);
                    if (d > worst) { worst = d; worst_ip = ip0b + jj; }
                }
                p2 += (npop2 - 1);
            }
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
            printf("[IONIZ-LADDER]  s=%d T=%.0f n_e=%.6e  max|f_code-f_saha|=%.3e"
                   " (Z=%d stage=%d)  charge-conservation n_e_recomputed/n_e=%.9f\n",
                   s, Ts, ne_p, worst,
                   worst_ip >= 0 ? atom->ion_pop_Z[worst_ip] : -1,
                   worst_ip >= 0 ? atom->ion_pop_stage[worst_ip] : -1,
                   (ne_p > 0.0) ? ne_chk / ne_p : -1.0);
            fflush(stdout);
            }
        }
        /* ---- culled ETLA line table (lagged tau/Jbar; T-independent parts) ---- */
        sh.nl = 0;
        double cull = 0.01;
        { const char *c = getenv("LUMINA_RADEQ_LINE_CULL"); if (c) cull = atof(c); }
        double thr = cull * (sh.H_dep > 0.0 ? sh.H_dep : 1e-30) /
                     (double)(radeq_n_lines > 0 ? radeq_n_lines : 1);
        for (long k = 0; k < radeq_n_lines; k++) {
            const RadEqLine *rl = &radeq_lines[k];
            /* withParityO: covered ions' bb collisional cooling is owned by the
             * pair loop (built below) — drop their 2-level lines from lam so the
             * two do not double count. Gate off => g_cp_line_covered NULL => kept. */
            if (g_cp_on == 1 && g_cp_line_covered && g_cp_line_covered[k]) continue;
            /* cull bound must cover ALL trial ionization states (the lagged
             * state culled away the mid-T coolant lines -> no root): use the
             * ELEMENT density as the n_ion upper bound, n_e <= 6*natom. */
            int Ze = atom->ion_pop_Z[rl->ip];
            double nel_k = 0.0;
            for (int e2 = 0; e2 < atom->n_elements; e2++)
                if (atom->element_Z[e2] == Ze) {
                    nel_k = atom->abundances[e2 * n_shells + s] * plasma->rho[s] /
                            (atom->element_mass_amu[e2] * AMU);
                    break;
                }
            if (nel_k <= 0.0) continue;
            double cmax = rl->dE * rl->coeff * nel_k / rl->g_lo *
                          (6.0 * sh.natom) / sqrt(2000.0);
            if (cmax < thr) continue;
            double nu_l = rl->dE / H_PLANCK;
            /* [PUMPF] route the line-pump Jb through the SAME alpha-blended field
             * the Gph loop consumes (field source only; binning + line-term
             * structure unchanged). Gate off => byte-identical hard-wired cs_J. */
            double Te_pump = plasma->T_e[s];
            double legacy_Jb_shadow = (g_radeq_pump_field == 1)
                ? radeq_pump_line_Jb(nlte, s, nfb, nu_l, Te_pump,
                                     &n_pumpf_bl, &n_pumpf_fb, &n_pumpf_bnu)
                : nlte_get_J_at_nu(nlte, s, nu_l);
            (void)legacy_Jb_shadow; /* A2-06 diagnostic/falsifier shadow only */
            double Jb = 0.0;
            (void)nlte_bb_jbar_canonical(nlte, s, rl->line, &Jb);
            double B_ul = rl->A_ul * C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT /
                          (2.0 * H_PLANCK * nu_l * nu_l * nu_l);
            double B_lu = B_ul * (double)rl->g_up / (double)rl->g_lo;
            long m = sh.nl++;
            sh.l_ip[m] = rl->ip;
            sh.l_dE[m] = rl->dE;   sh.l_beta[m] = rl->beta;
            sh.l_coeff[m] = rl->coeff;
            sh.l_glo[m] = (double)rl->g_lo; sh.l_gup[m] = (double)rl->g_up;
            sh.l_Elo[m] = atom->level_energy_eV[rl->lo_g] * EV_TO_ERG;
            sh.l_BluJ[m]  = B_lu * Jb;
            sh.l_ABulJ[m] = rl->A_ul + B_ul * Jb;
            /* SELF-CONSISTENT Sobolev tau: tau(T) = ftau * n_lo(T). Using the
             * LAGGED tau_sobolev here created hysteresis: a committed stripped
             * state zeroed tau -> beta=1 -> cooling loss -> the ~40kK root
             * vanished and the ladder ratcheted into the 140kK strip attractor
             * (fix13 first flight). The offline calculator uses trial-tau and
             * finds the root robustly.  f_lu = 1.4992e-16*A_ul*(g_up/g_lo)*lam_A^2. */
            {
                double lam_A = 1e8 * C_SPEED_OF_LIGHT / nu_l;
                double f_lu = 1.4992e-16 * rl->A_ul *
                              ((double)rl->g_up / (double)rl->g_lo) * lam_A * lam_A;
                double lam_cm = C_SPEED_OF_LIGHT / nu_l;
                sh.l_ftau[m] = 0.02654 * f_lu * lam_cm * time_explosion;
            }
        }
        /* ---- withParityO: CMFGEN all-level-pair COL cooling for the covered
         * ions at this shell, built from the LIVE NLTE populations (replaces the
         * covered lines skipped from lam above). Omega + vR GBAR frozen at the
         * lagged T_e; the trial-T exp(-beta/T)/sqrt(T) is applied in simul_r1's
         * radeq_line_cool. cp_n stays 0 when the gate is off => byte-identical. */
        sh.cp_n = 0;
        if (g_cp_on == 1 && g_cp_nions > 0 && sh.cp_a) {
            double cp_ne = plasma->n_electron[s];
            double cp_Tref = plasma->T_e[s];
            long dropped = 0;
            for (int c = 0; c < g_cp_nions; c++) {
                RcpCensus cen;
                radeq_colpairs_ion_shell(&g_cp_ions[c], atom, nlte, n_shells, s,
                                         cp_ne, cp_Tref, g_cp_maxlev,
                                         sh.cp_a, sh.cp_b, sh.cp_beta, &sh.cp_n,
                                         &cen, &dropped);
                if (s == 8) {
                    for (int pr = 0; pr < CP_NPROBE; pr++)
                        if (g_cp_probe_Z[pr] == g_cp_ions[c].Z &&
                            g_cp_probe_ion[pr] == g_cp_ions[c].ion0) {
                            g_cp_s8_cen[pr] = cen;
                            g_cp_s8_cool[pr] = cen.c_tab + cen.c_vr + cen.c_set;
                        }
                }
            }
            if (s == 8) g_cp_s8_dropped = dropped;
        }
#ifdef LUMINA_FROZEN_ORACLE
        /* Dedicated observer build: evaluate the actual production residual once
         * at the recorded cell temperature and stop before root solving/commit.
         * simul_r1 owns the thermal arithmetic; these are read-only shadows. */
        if (g_oracle.fp) {
            double ne_probe = plasma->n_electron[s];
            double net = simul_r1(atom, &sh, plasma->T_e[s], &ne_probe);
            g_oracle.thermal_seen = 1;
            g_oracle.thermal_deposition = sh.H_dep;
            g_oracle.thermal_photoion = g_r1d_H - sh.H_dep;
            g_oracle.thermal_ff = g_r1d_ff;
            g_oracle.thermal_bf = g_r1d_fb;
            g_oracle.thermal_bb_collisional = g_r1d_lam + g_r1d_colpairs;
            g_oracle.thermal_adiabatic = g_r1d_ad;
            g_oracle.thermal_net = net;
            continue;
        }
#endif
        /* INPUT-DIFF dump (flip forensics): per-pair Gph + UV J bins.
         * The T* flip 43k->140k at iter-3 survived J/T/ion damping — diff
         * these inputs across iterations to name the switching channel. */
        if (getenv("LUMINA_RADEQ_DIAG") && s == n_shells - 1) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                printf("  [SIMUL-IN s=%d] gnt=%.3e Gph:", s, sh.gnt);
                for (int p2 = 0; p2 < sh.np; p2++) printf(" %.2e", sh.Gph[p2]);
                printf("\n  [SIMUL-IN s=%d] Jbin:", s);
                for (int bb = 700; bb < 1000; bb += 60)
                    printf(" %.2e", nlte->J_nu[(size_t)s * nfb + bb]);
                printf("\n");
            }
        }
        /* diag: r1(T) scan at the outer trace shell (calculator cross-check) */
        static int trace_sh = -2;
        if (trace_sh == -2) { const char *ts = getenv("LUMINA_SIMUL_TRACE");
                              trace_sh = ts ? atoi(ts) : -1; }
        if (getenv("LUMINA_RADEQ_DIAG") &&
            (s == n_shells - 1 || s == n_shells / 2 || s == trace_sh)) {
            double Tg[12] = {5000, 8000, 11000, 14000, 18000, 24000, 32000,
                             45000, 60000, 80000, 110000, 140000};
            for (int q = 0; q < 12; q++) {
                double nq = plasma->n_electron[s];
                double rq = simul_r1(atom, &sh, Tg[q], &nq);
                /* recompute components for the print */
                double Hp = 0.0;
                for (int p2 = 0; p2 < sh.np; p2++) Hp += sh.nion[sh.ipa[p2]] * sh.Hex[p2];
#ifdef _OPENMP
                #pragma omp critical
#endif
                {
                printf("  [SIMUL-SCAN it=%d s=%d] T=%6.0f r1=%+.3e H_photo=%.3e n_e=%.3e\n",
                       g_simul_iter_no, s, Tg[q], rq, Hp, nq);
                /* [SIMUL-CT] per-term cooling of the SAME simul_r1 call (thread-local
                 * shadows; Dig-C calibration closer: lam vs C_fb split of C_total). */
                printf("  [SIMUL-CT   it=%d s=%d] T=%6.0f H=%.3e Cff=%.3e Cad=%.3e "
                       "Cfb=%.3e lam=%.3e\n",
                       g_simul_iter_no, s, Tg[q], g_r1d_H, g_r1d_ff, g_r1d_ad,
                       g_r1d_fb, g_r1d_lam);
                }
            }
        }
        /* ---- bisection on the wide physical bracket ---- */
        double Tlo = 3500.0, Thi = 140000.0;
        double ne_l = plasma->n_electron[s], ne_h = plasma->n_electron[s];
        double f_lo = simul_r1(atom, &sh, Tlo, &ne_l);
        double f_hi = simul_r1(atom, &sh, Thi, &ne_h);
        double T_e, ne_fin;
        int held = 0;
        /* NO-ROOT => NOT a solution: committing a bracket endpoint was audit
         * defect B1, and inside SIMUL it FUELED the strip attractor — the
         * pinned-hi commit ratcheted T_e[49] to 50-95kK whose OWN B(T_e)
         * hard-UV emission (J bin +12 dex in one iter, 1e21 x dilute-
         * photospheric = unphysical self-illumination) photo-stripped every
         * trial state. HOLD the previous T_e instead and let the transport
         * relax (forensics: fix14 [SIMUL-IN] iter2->3). */
        if (f_lo <= 0.0)      { T_e = plasma->T_e[s] > 100.0 ? plasma->T_e[s] : Tlo;
                                ne_fin = ne_l; n_pin_lo++; held = 1;
                                if (tehold_on && s < TEHOLD_MAXSH) g_tehold_status[s] = 2; }
        else if (f_hi >= 0.0) { T_e = plasma->T_e[s] > 100.0 ? plasma->T_e[s] : Thi;
                                ne_fin = ne_h; n_pin_hi++; held = 1;
                                if (tehold_on && s < TEHOLD_MAXSH) g_tehold_status[s] = 3; }
        else {
            if (tehold_on && s < TEHOLD_MAXSH) g_tehold_status[s] = 1; /* [TEHOLD] root-found */
            /* LOWEST-ROOT selection (cold-branch continuation): the balance is
             * multi-rooted in the bistable window (e.g. s=40: cold ~13kK root
             * AND a self-illuminated strip root ~100kK). Plain bisection on
             * [Tlo,Thi] lands on an arbitrary crossing; CMFGEN/ARTIS follow
             * the branch continuously from the (cold) previous state. Coarse
             * upward march to the FIRST sign change, then bisect inside it. */
            double ne_m = plasma->n_electron[s];
            double Ta = Tlo, fa = f_lo;
            double fac = pow(Thi / Tlo, 1.0 / 24.0);
            for (int q = 1; q <= 24; q++) {
                double Tb = Tlo * pow(fac, (double)q);
                if (q == 24) Tb = Thi;
                double nb = plasma->n_electron[s];
                double fb = simul_r1(atom, &sh, Tb, &nb);
                if (fa > 0.0 && fb <= 0.0) { Tlo = Ta; Thi = Tb; break; }
                Ta = Tb; fa = fb;
            }
            for (int it = 0; it < 45 && (Thi - Tlo) > 2.0; it++) {
                double Tm = 0.5 * (Tlo + Thi);
                double fm = simul_r1(atom, &sh, Tm, &ne_m);
                if (fm > 0.0) Tlo = Tm; else Thi = Tm;
            }
            T_e = 0.5 * (Tlo + Thi); ne_fin = ne_m;
        }
        /* ---- commit (ARTIS thermalbalance.cc:304 mirror): final ladder at T*,
         *      write T_e (damped+clamped), n_e and the ion partition ---- */
        double T_old = plasma->T_e[s];
        double T_new = T_e;
        (void)held;
        if (radeq_damp < 1.0 && T_old > 100.0)
            T_new = radeq_damp * T_e + (1.0 - radeq_damp) * T_old;
        T_new = radeq_te_step_clamp(T_new, T_old);
        /* LUMINA_DIAG_TE_PIN: diagnostic HARD override of T_e over a shell range.
         * Pinned as a CONSTRAINT of the solve, not a cosmetic post-write: the ion
         * ladder / n_e below (simul_ladder at T_new) run at the pinned T_e, and the
         * committed plasma->T_e[s] — downloaded by the CUDA transport next iter for
         * k-packet ff/fb + collisional rates — carries the pinned value. Gate off
         * (g_te_pin_on==0) => branch skipped => damp/clamp arithmetic byte-identical. */
        if (g_te_pin_on && s >= g_te_pin_smin && s <= g_te_pin_smax) {
            double frac = (g_te_pin_smax > g_te_pin_smin)
                        ? (double)(s - g_te_pin_smin) /
                          (double)(g_te_pin_smax - g_te_pin_smin)
                        : 0.0;
            T_new = g_te_pin_T0 + (g_te_pin_T1 - g_te_pin_T0) * frac;
        }
        /* F3-T TEMPERATURE-TABLE probe: REPLACE the radeq-solved T_e with CMFGEN's
         * published T_e(v) for this shell. Placed here -- BEFORE simul_ladder and
         * the plasma->T_e[s] commit -- so the ion ladder / n_e / committed T_e (and
         * therefore next iter's GPU emissivity/k-packet/collisional consumers) all
         * inherit the table temperature = WHOLE-STATE pin (see loader block above).
         * Gate off (g_te_table==NULL) => branch skipped => byte-identical. */
        if (g_te_table && g_te_table[s] > 0.0) {
            T_new = g_te_table[s];
            n_te_table_pins++;
        }
        /* SELF-CONSISTENT commit: ladder at the COMMITTED temperature, not at
         * T* — committing T*(~40k)-stripped ions with a damped T_e(12k) killed
         * the outer blanketing ahead of the temperature, re-opening the hard-UV
         * window and locking the strip attractor (fix14 iter2). */
        simul_ladder(atom, &sh, T_new, &ne_fin);
        plasma->T_e[s] = T_new;
        if (tehold_on && s < TEHOLD_MAXSH) {   /* [TEHOLD] record committed T_e + prev */
            g_tehold_te[s] = T_new; g_tehold_told[s] = T_old;
        }
        /* ION-side under-relaxation (CMFGEN D/Dt population-inertia miniature,
         * steq_co_mov_deriv RELAX mirror; fixed-point unchanged): with J damped
         * and T_e clamped, the instantaneously-committed equilibrium ions were
         * the last un-damped state variable — one iteration could still flip
         * the outer blanketing/Gph channels and lock the strip attractor. */
        static double ion_damp = -1.0;
        if (ion_damp < 0.0) { const char *e = getenv("LUMINA_SIMUL_ION_DAMP");
                              ion_damp = e ? atof(e) : 0.5; }
        double ne_mix = 0.0;
        for (int e2 = 0; e2 < atom->n_elements; e2++) {
            int ip0 = atom->elem_ion_offset[e2], ip1 = atom->elem_ion_offset[e2 + 1];
            double nel2 = atom->abundances[e2 * n_shells + s] * plasma->rho[s] /
                          (atom->element_mass_amu[e2] * AMU);
            for (int ip = ip0; ip < ip1; ip++) {
                double prev = atom->ion_number_density[(size_t)ip * n_shells + s];
                /* PHYSICAL BOUND on the blend memory: the ion-damp blend was
                 * PRESERVING super-physical seed garbage (s=37-42 n_e 60x over
                 * charge conservation, decaying only 0.5/iter) whose n_e^2 ff
                 * emission was the hard-UV lamp that photo-stripped the outer
                 * (fix15 forensics: corrupt shells identical with clean sh.nion
                 * => the memory term was the carrier). n_ion <= n_element. */
                if (prev > nel2) prev = nel2;
                double mixed = ion_damp * sh.nion[ip] + (1.0 - ion_damp) * prev;
                if (mixed > nel2) mixed = nel2;
                atom->ion_number_density[(size_t)ip * n_shells + s] = mixed;
                ne_mix += (double)atom->ion_pop_stage[ip] * mixed;
            }
        }
        plasma->n_electron[s] = (ne_mix > 0.0) ? ne_mix : ne_fin;
        if (getenv("LUMINA_RADEQ_DIAG") &&
            (s == 0 || s == n_shells / 2 || s == n_shells - 1))
#ifdef _OPENMP
            #pragma omp critical
#endif
            printf("  [SIMUL s=%d] T*=%.0f -> committed %.0f  n_e=%.3e  "
                   "H_dep=%.3e r1(T*)=~0  nl=%ld np=%d\n",
                   s, T_e, T_new, ne_fin, sh.H_dep, sh.nl, sh.np);
        (void)T_rad;
    }
    free(sh.nion); free(sh.l_ip); free(sh.l_dE); free(sh.l_beta);
    free(sh.l_coeff); free(sh.l_glo); free(sh.l_gup); free(sh.l_Elo);
    free(sh.l_BluJ); free(sh.l_ABulJ); free(sh.l_ftau);
    free(sh.emit_bf); free(sh.emit_bx);   /* [DBFB] (NULL-safe when gate off) */
    free(sh.cp_a); free(sh.cp_b); free(sh.cp_beta);   /* withParityO (NULL-safe) */
    }   /* omp parallel */
    printf("  [SIMUL it=%d] done: pins hi=%ld lo=%ld of %d shells\n",
           g_simul_iter_no,
           n_pin_hi, n_pin_lo, n_shells);
    /* withParityO fail-loud: per-iter s8 census (pair-count split tab/vR/0.1) +
     * per-ion cooling spot-check, so a masked drop / iter-0 no-pop state is never
     * silent. Cooling >0 = cooling, <0 = super-elastic collisional HEATING. */
    if (g_cp_on == 1 && g_cp_nions > 0) {
        int pops_absent = (nlte == NULL || nlte->nlte_level_populations == NULL);
        if (pops_absent)
            printf("  [COL-PAIRS] it=%d: NLTE level populations absent this iter "
                   "(pre-NLTE) — pair cooling forced to 0, engages once pops exist\n",
                   g_simul_iter_no);
        if (!g_cp_census_done && !pops_absent) {
            printf("  [COL-PAIRS s8 census] per-ion pair split (tab/vR/0.1), "
                   "maxlev=%d, levels_dropped=%ld:\n", g_cp_maxlev, g_cp_s8_dropped);
            for (int pr = 0; pr < CP_NPROBE; pr++) {
                RcpCensus *c = &g_cp_s8_cen[pr];
                if (c->n_pairs == 0) {
                    printf("    %-6s: NO pairs at s8 (ion not covered, or NLTE pops "
                           "absent this iter — cooling contributes 0)\n",
                           g_cp_probe_name[pr]);
                    continue;
                }
                printf("    %-6s: pairs tab=%ld vR=%ld 0.1=%ld\n", g_cp_probe_name[pr],
                       c->n_tab, c->n_vr, c->n_set);
            }
            g_cp_census_done = 1;
        }
        printf("  [COL-PAIRS s8 cool] it=%d (erg/cm^3/s, +=cool -=heat): "
               "SiIII=%+.3e SIII=%+.3e FeIII=%+.3e CoIII=%+.3e NiIII=%+.3e\n",
               g_simul_iter_no, g_cp_s8_cool[0], g_cp_s8_cool[1], g_cp_s8_cool[2],
               g_cp_s8_cool[3], g_cp_s8_cool[4]);
        fflush(stdout);
    }
    /* [TEHOLD] LINE_THERM addendum: per-iteration radeq root status for the gated
     * shells 0..SMAX plus s8 (control). Reveals whether s0's root is actually being
     * re-solved (root-found) or frozen (pin_lo/pin_hi HOLD) — the amended-gate(iii)
     * discriminator. Ordered, host-side; printed only when LINE_THERM is on. */
    if (tehold_on) {
        for (int s = 0; s <= tehold_smax && s < n_shells && s < TEHOLD_MAXSH; s++)
            printf("  [TEHOLD] s%d: T_e=%.0fK (prev=%.0fK) radeq_root=%s\n",
                   s, g_tehold_te[s], g_tehold_told[s],
                   tehold_root_name(g_tehold_status[s]));
        if (8 > tehold_smax && 8 < n_shells && 8 < TEHOLD_MAXSH)
            printf("  [TEHOLD] s8(ctrl): T_e=%.0fK (prev=%.0fK) radeq_root=%s\n",
                   g_tehold_te[8], g_tehold_told[8],
                   tehold_root_name(g_tehold_status[8]));
        fflush(stdout);
    }
    /* #33: per-iteration effect counter -- zero N with the gate ON = wiring bug. */
    if (g_gph_jtable)
        printf("  [JTABLE] gph_evals_using_table=%ld\n", n_jtable_evals);
    /* F3-T: per-iteration pin counter -- shells_pinned<n_shells with the gate ON
     * = silent no-op (a pin that never fired must be loudly visible). */
    if (g_te_table)
        printf("  [TETAB] shells_pinned=%ld\n", n_te_table_pins);
    /* [PUMPF] per-iteration line-Jb source counter -- zero blended with the gate
     * ON *after* the first transport pass = wiring bug; all-fallback (blended=0)
     * on iter 0 is the expected pre-transport (mc_J NULL) startup, mirroring the
     * Gph guard's silent cs_J keep. */
    if (g_radeq_pump_field == 1)
        printf("  [PUMPF] line_Jb: blended=%ld cs_fallback=%ld fallback_mode=%d "
               "bnu_routed=%ld\n",
               n_pumpf_bl, n_pumpf_fb, (g_radeq_pump_fallback == 1 ? 1 : 0),
               n_pumpf_bnu);
}

static double radeq_recomb_cool(double T_e, const double *emit_nu,
                                const double *nu_mid, int nbins) {
    /* LUMINA_RADEQ_FB_RATE=1: rate-based replacement (see block above) */
    if (g_fbr_on == 1) return radeq_fb_rate_eval(T_e);
    double inv_kT = 1.0 / (K_BOLTZMANN * T_e);
    double sum = 0.0;
    for (int bb = 0; bb < nbins; bb++) {
        if (emit_nu[bb] == 0.0) continue;
        sum += emit_nu[bb] * exp(-H_PLANCK * nu_mid[bb] * inv_kT);
    }
    return sum;
}

/* Collisional bound-bound (line) cooling, evaluated from per-shell precomputed
 * coefficients. For each active line the net (excitation - deexcitation) cooling
 * reduces analytically (the deexcitation exp(dE/kT) cancels the excitation
 * exp(-dE/kT)) to  (n_e/sqrtTe) * [a*exp(-beta/Te) - b],  with
 *   a = dE*coeff*n_lo/g_lo  (excitation),  b = dE*coeff*n_up/g_up (deexcitation).
 * Both a,b are T_e-independent (lagged pops), so only one exp per line per eval.
 * When nonneg!=0, each per-line term is floored at 0 so spurious NLTE level
 * inversions cannot turn this coolant into a (numerical) heating term. */
static double radeq_line_cool(double T_e, double n_e,
                              const double *a, const double *b,
                              const double *beta, long n_active, int nonneg) {
    double sqrtTe = sqrt(T_e);
    double sum = 0.0;
    for (long m = 0; m < n_active; m++) {
        double br = a[m] * exp(-beta[m] / T_e) - b[m];
        if (nonneg && br < 0.0) br = 0.0;
        sum += br;
    }
    return n_e / sqrtTe * sum;
}

/* Sobolev escape probability beta_esc(tau) = (1-e^-tau)/tau (Castor 1970),
 * with the optically-thin limit beta_esc(0)=1 handled smoothly. */
static double radeq_beta_esc(double tau) {
    if (tau <= 1e-6) return 1.0;
    if (tau > 700.0) return 1.0 / tau;          /* avoid exp underflow; beta->1/tau */
    return (1.0 - exp(-tau)) / tau;
}

/* A3 incr-1 / Phase-1: diagonal-Λ* radiation response (heating side). The lagged
 * H_photo used the frozen binned J_ν; this lets the bf-absorbed field follow the
 * trial T_e via the ALI linearization ΔJ_b = Λ*_b·ε_b·(B_ν(T_e)−blag_b), where
 * Λ*_b=∂J/∂S is the formal-solve diagonal operator and ε_b=χ_abs/χ_tot the thermal
 * absorption fraction. The caller folds Λ*·ε into lstar[bb], so W_lstar=1 in the
 * faithful pure-CMFGEN path (the τ-proxy fallback passes its own scale). gbin[bb]=
 * Σ n_lev·(4πσ f_above dν) is the SAME per-bin photo-weight that built H_photo.
 * FAITHFUL ANCHOR: the caller sets blag_b = J*_b (the frozen per-bin mean intensity
 * that built H_photo) in the pure-CMFGEN path, so bf-net + H_resp collapses to
 * Σ_bb gbin·(1−Λ*ε)·(J*−B_ν(T_e)) — the streaming fraction keeps a non-zero
 * restoring slope dr/dT_e<0 → thick limit thermalizes T_e to the J* color temp.
 * (Anchoring at B_ν(Te_lag), as the τ-proxy fallback still does, makes the pair
 * cancel identically and pins T_e at the seed.) gbin==NULL → 0 (byte-identical). */
static double radeq_Hresp(double T_e, const double *gbin, const double *lstar,
                          const double *blag, double W_lstar,
                          const double *nu_mid, int nbins) {
    if (!gbin) return 0.0;
    /* LUMINA_HRESP_CLAMP=<fac>: trust region on the ALI linearization.
     * dB = B_nu(T_trial) - blag is a LOCAL response; at trial T far above
     * T_lag the Wien factor makes dB explode (fix4 RTRUTH s=25:
     * r1(140kK)=+7.7e4 vs physics ~5e-8 — 12 orders of meaningless
     * extrapolation that destroys every hot root once the lre term is
     * retired). Clamp |dB| <= fac*max(blag, |B|/10) per bin. 0 = off. */
    static double hclamp = -1.0;
    if (hclamp < 0.0) { const char *e = getenv("LUMINA_HRESP_CLAMP");
                        hclamp = e ? atof(e) : 0.0; if (hclamp < 0) hclamp = 0.0; }
    double H_resp = 0.0;
    for (int bb = 0; bb < nbins; bb++) {
        if (gbin[bb] == 0.0 || lstar[bb] == 0.0) continue;
        double B = planck_bnu(T_e, nu_mid[bb]);
        double dB = B - blag[bb];
        if (hclamp > 0.0) {
            double lim = hclamp * (blag[bb] > 0.0 ? blag[bb] : fabs(B) * 0.1);
            if (dB >  lim) dB =  lim;
            if (dB < -lim) dB = -lim;
        }
        H_resp += gbin[bb] * lstar[bb] * W_lstar * dB;
    }
    return H_resp;
}

/* A3 (A): ETLA T_e-responsive bound-bound cooling. The signed/floored/escape forms
 * all feed a LAGGED upper population n_up into the cooling sum, so inside the Newton
 * the cooling does NOT track the trial T_e — a lagged non-SE inversion flips it to
 * spurious heating (signed runaway) or the escape form omits the B_lu·J̄ absorption
 * source (catastrophic over-cool). Here n_up is recomputed in LOCAL statistical
 * equilibrium at the trial (T_e,n_e) of the equivalent two-level atom:
 *   n_up = n_lo·(C_lu + R_lu)/(C_ul + R_ul),
 * with C_lu=n_e·q_lu, C_ul=n_e·q_ul the (same-coeff) collisional rates and
 * R_lu=B_lu·J̄·β_esc, R_ul=(A_ul+B_ul·J̄)·β_esc the Sobolev-escape radiative rates.
 * Net collisional cooling C = n_e·Σ dE·(n_lo·q_lu − n_up·q_ul). NOTE this is
 * SIGNED: it cools iff J̄≤B_ν(T_e) and would flip to absorption-pumped line
 * HEATING for a super-thermal J̄ (q_lu·R_ul < q_ul·R_lu). The lagged binned-MC J̄
 * is known over-hard/noisy at the iron-curtain UV edge in exactly the sh11-24
 * transition shells, which could re-arm the T_e runaway through radiative pumping.
 * Faithful guard: cap n_up at its local Boltzmann ceiling n_lo·(g_up/g_lo)·e^{−β/Te}
 * — the exact no-line-heating constraint. It preserves all genuine sub-thermal
 * cooling and the LTE limit (capped net = 0 there), and only floors the spurious
 * pumping-heating branch. A[m]=coeff/g_lo, B[m]=coeff/g_up so A/B=g_up/g_lo
 * (q_lu=A·e^{−β/Te}/√Te, q_ul=B/√Te).
 *
 * MODE (hybrid, line_respond==2): the 3-arm batch (164668/9/70) showed pure SE
 * recompute REMOVES the photosphere coolant — where the lagged binned-MC J̄ is
 * super-thermal the SE n_up→Boltzmann zeros the net cooling, so sh0-8 T_e runs up
 * to ~8900K even though signed's lagged near-thermal n_up cooled it correctly to
 * 4402K. Pure ETLA only WINS in the transition (sh14/18 8-11kK→~5kK). The faithful,
 * non-spatial discriminator is the SIGN of the lagged term itself: use the lagged
 * n_up cooling where it is a genuine coolant (≥0, as at the photosphere), and fall
 * back to the capped-SE n_up only where the lagged inversion produces unphysical
 * self-heating (signed<0, as in the transition). nup_lag carries the lagged pop. */
static double radeq_line_cool_etla(double T_e, double n_e,
                                   const double *A, const double *B,
                                   const double *beta, const double *dE,
                                   const double *nlo, const double *nup_lag,
                                   const double *Rlu, const double *Rul,
                                   long n_active, int hybrid) {
    double invsqrt = 1.0 / sqrt(T_e);
    double sum = 0.0;
    for (long m = 0; m < n_active; m++) {
        if (nlo[m] <= 0.0 || B[m] <= 0.0) continue;
        double exb = exp(-beta[m] / T_e);
        double qlu = A[m] * invsqrt * exb;
        double qul = B[m] * invsqrt;
        if (hybrid) {
            double net_lag = dE[m] * (nlo[m] * qlu - nup_lag[m] * qul);
            if (net_lag > 0.0) { sum += net_lag; continue; }  /* genuine coolant */
        }
        double Clu = n_e * qlu, Cul = n_e * qul;
        double denom = Cul + Rul[m];
        if (denom <= 0.0) continue;
        double nup = nlo[m] * (Clu + Rlu[m]) / denom;
        /* Boltzmann ceiling (no line-pumped heating): guard for NOISY MC J̄.
         * With deterministic binned J it BLOCKS GENUINE line heating — at a
         * cold thick shell J̄(color~T_rad)>B(T_e) is real absorption heating
         * (fix8: inner collapsed to 1000K floor with ceiling on; audit D3).
         * LUMINA_ETLA_ALLOW_HEAT=1 lifts it. */
        static int allow_heat = -1;
        if (allow_heat < 0) { const char *e = getenv("LUMINA_ETLA_ALLOW_HEAT");
                              allow_heat = (e && atoi(e)) ? 1 : 0; }
        if (!allow_heat) {
            double nup_lte = nlo[m] * (A[m] / B[m]) * exb;
            if (nup > nup_lte) nup = nup_lte;
        }
        sum += dE[m] * (nlo[m] * qlu - nup * qul);
    }
    return n_e * sum;
}

/* ---- Option-2 integral radiative equilibrium: CMFGEN line opacity/source ----
 * Registered once per CMFGEN outer iteration. All arrays are lagged at the
 * assemble-time T_e (Te_lag); chi_line/chi_abs/chi_tot/S_fixed/J are
 * [n_shells*n_bins], nu/dnu are [n_bins]. NULL chi_line disables the term. */
static const double *g_lre_chi_line = NULL, *g_lre_chi_abs = NULL,
                    *g_lre_chi_tot = NULL, *g_lre_S_fixed = NULL,
                    *g_lre_J = NULL, *g_lre_nu = NULL, *g_lre_dnu = NULL,
                    *g_lre_lambda_star = NULL,
                    *g_lre_chi_line_full = NULL,  /* unweighted chi_line (A4 boot) */
                    *g_lre_chi_line_cls = NULL;   /* eps*beta/(eps+beta-eps*beta)
                                                     two-level+Sobolev gas-coupling
                                                     chi (A4 SRC_BLEND closure) */
static int g_lre_nshells = 0, g_lre_nbins = 0;
/* assemble-time T_e SNAPSHOT (copied, not aliased): the pre-Newton bisection
 * rewrites plasma->T_e between registration and the coupled Newton, so reading
 * plasma->T_e there yields the WRONG lag for the eta_lag subtraction in
 * radeq_line_re wherever the bisection moved T_e. */
static double *g_lre_te_lag = NULL;
/* A4 Stage-1 bootstrap flags: per shell, 1 = the bf pair is strong enough to
 * anchor T_e (switch to the SYMMETRIC eps-weighted line term = converged
 * physics), 0 = bf still empty (early NLTE iters) so keep the legacy
 * all-thermal line closure as the transient stabilizer. Measured basis:
 * the eps-weighted line slope at s=0 is ~4e-8 erg/cm3/s/K (same order as
 * ff) — physically too weak to anchor; the converged anchor is bf (Wien). */
static unsigned char *g_a4_boot = NULL;
static int g_a4_boot_n = 0;
/* A4 Stage-4': continuum-window color temperature of the deterministic J
 * (cmfgen_window_color). The faithful frozen-tail T_e anchor: the outer
 * energy equation is a self-referential line-trough thermostat (treadmill,
 * asymptote -10%), while the window color of the SAME field carries gold's
 * outer T_e (= the field's optical color temp, measured 2505K at sh48). */
static const double *g_tail_color = NULL;
static int g_tail_color_n = 0;

/* A4 Stage-2.5: per-(shell,bin) tridiagonal Lambda response coefficients
 * persisted from cmfgen_solve_J (last ALI pass): off-diagonals L[s,s-1]/
 * L[s,s+1] and the scattering albedo r. Consumed by the global Newton's
 * delta-J resolvent (Stage 3/4). */
static const double *g_tri_lo = NULL, *g_tri_up = NULL, *g_tri_r = NULL;
static int g_tri_ns = 0, g_tri_nb = 0;

void radeq_set_tri_response(const double *lo, const double *up,
                            const double *r, int n_shells, int n_bins) {
    g_tri_lo = lo; g_tri_up = up; g_tri_r = r;
    g_tri_ns = n_shells; g_tri_nb = n_bins;
}

void radeq_set_tail_color(const double *t_color, int n_shells) {
    g_tail_color = t_color;
    g_tail_color_n = n_shells;
}
static int g_lre_te_lag_n = 0;


void radeq_set_line_re_source(const double *chi_line, const double *chi_abs,
                              const double *chi_tot, const double *S_fixed,
                              const double *J, const double *nu,
                              const double *dnu, const double *lambda_star,
                              const double *T_e_assemble,
                              const double *chi_line_full,
                              const double *chi_line_cls,
                              int n_shells, int n_bins) {
    g_lre_chi_line_full = chi_line_full;
    g_lre_chi_line_cls = chi_line_cls;
    if (n_shells > 0 && g_a4_boot_n != n_shells) {
        free(g_a4_boot);
        g_a4_boot = (unsigned char *)calloc((size_t)n_shells, 1);
        g_a4_boot_n = g_a4_boot ? n_shells : 0;
    }
    g_lre_chi_line = chi_line; g_lre_chi_abs = chi_abs;
    g_lre_chi_tot = chi_tot;   g_lre_S_fixed = S_fixed;
    g_lre_J = J; g_lre_nu = nu; g_lre_dnu = dnu;
    g_lre_lambda_star = lambda_star;
    g_lre_nshells = n_shells;  g_lre_nbins = n_bins;
    if (T_e_assemble && n_shells > 0) {
        if (g_lre_te_lag_n != n_shells) {
            free(g_lre_te_lag);
            g_lre_te_lag = (double *)malloc((size_t)n_shells * sizeof(double));
            g_lre_te_lag_n = g_lre_te_lag ? n_shells : 0;
        }
        if (g_lre_te_lag)
            memcpy(g_lre_te_lag, T_e_assemble,
                   (size_t)n_shells * sizeof(double));
    } else {
        free(g_lre_te_lag); g_lre_te_lag = NULL; g_lre_te_lag_n = 0;
    }
}

/* Option-2 radiative line term for one shell, H_line(T_e) = 4π∫χ_line(J−S_eff)dν
 * [erg/s/cm³], added to the HEATING side of radeq_net (J>S ⇒ net heating).
 * T_e-RESPONSIVE (codex form (b)): the lagged NLTE line emissivity is carried,
 * and only the thermal Planck piece is re-evaluated at the trial T_e:
 *   η_lag = S_fixed·χ_tot − χ_abs·B(Te_lag)   (= χ_line·S_line_lag)
 *   η_pre = η_lag + χ_line·(B(Te) − B(Te_lag))
 * so dH_line/dT_e = −4π∫χ_line·dB/dT_e dν ≤ 0 (restoring → root exists).
 * NO-PUMPING CLAMP (floor at B(Te)): η_eff = max(η_pre, χ_line·B(Te)) caps the
 * spurious heating from sub-Planck lagged sources in the thin-UV J≫S bins (the
 * A2 over-heat hazard); LTE (J=B, S_lag=B) gives exactly 0, genuine cooling
 * (S_pre>J) is preserved. */
static double radeq_line_re(double T_e, double Te_lag, int s) {
    if (!g_lre_chi_line || s < 0 || s >= g_lre_nshells) return 0.0;
    int nb = g_lre_nbins;
    const double *cl = &g_lre_chi_line[(size_t)s * nb];
    const double *ca = &g_lre_chi_abs[(size_t)s * nb];
    const double *ct = &g_lre_chi_tot[(size_t)s * nb];
    const double *sf = &g_lre_S_fixed[(size_t)s * nb];
    const double *Jb = &g_lre_J[(size_t)s * nb];
    double H = 0.0;
    /* TRANSFER-ONLY eps_uv mode: cooling-only closure on the FULL chi_line,
     * H = 4pi*Int chi_line*(min(J,B(T_e)) - B(T_e)) dnu. Vanishes at J=B,
     * supplies the restoring slope when J<B (the operator-split T_e anchor),
     * contributes ZERO when transfer-eps makes the FUV J superthermal (no
     * spurious line heating; codex ruling 2026-06-11). No eta_lag here:
     * S_fixed carries the transfer thermal share, the wrong owner for the
     * full-chi closure. */
    static int uv_mode = -1, a4_sym = -1, a4_blend = -1;
    if (uv_mode < 0) {
        const char *u = getenv("LUMINA_CMFGEN_LINE_EPS_UV");
        uv_mode = (u && atof(u) > 0.0) ? 1 : 0;
        const char *a4 = getenv("LUMINA_A4_STAGE1");
        a4_sym = (a4 && atoi(a4)) ? 1 : 0;
        const char *ab = getenv("LUMINA_A4_SRC_BLEND");
        a4_blend = (ab && atoi(ab)) ? 1 : 0;
        if (a4_blend) { a4_sym = 0; uv_mode = 0; } /* blend owns the closure */
    }
    /* A4 SRC_BLEND closure: symmetric two-level form on the exact
     * eps*beta/(eps+beta-eps*beta) gas-coupling chi (Newton-owned shells) —
     * saturated lines drop out (their trapped field is in detailed balance
     * with the gas), thin lines keep the eps-weighted transported anchor.
     * FROZEN shells keep the FULL-chi thermostat: their gas is kinetically
     * frozen at the radiation color temperature, and extracting that color
     * from the transported J is exactly what full-chi does (A0b: T_e[48]
     * +1.0%). The frozen partition is a physical boundary (the cascade's),
     * not a latch. No clamps anywhere. */
    if (a4_blend) {
        const double *cl_use = NULL;
        int frozen = (frozenin_is_frozen && s < frozenin_is_frozen_n &&
                      frozenin_is_frozen[s]);
        if (!frozen && g_lre_chi_line_cls)
            cl_use = &g_lre_chi_line_cls[(size_t)s * nb];
        else if (g_lre_chi_line_full)
            cl_use = &g_lre_chi_line_full[(size_t)s * nb];
        if (cl_use) {
            for (int b = 0; b < nb; b++) {
                if (cl_use[b] <= 0.0) continue;
                double B_te = planck_bnu(T_e, g_lre_nu[b]);
                H += cl_use[b] * (Jb[b] - B_te) * g_lre_dnu[b];
            }
            return 4.0 * M_PI_VAL * H;
        }
    }
    /* A4 Stage-1: SYMMETRIC two-level closure on the eps-weighted thermal
     * channel, H = 4pi*Int chi_line_th*(J - B(T_e)) dnu, NO clamps. The
     * cooling-only clamp (uv_mode below) amputated the J>B heating branch and
     * caused the eps crashes (design review 2026-06-11): with J solved
     * self-consistently against the same eps source, J>B(T_e_local) is
     * physical scattered light and its eps-thermalized share is genuine line
     * heating. cl[] must be the eps-weighted chi_line_th (registered by the
     * LINE_EPS_PHYS assemble path); slope 4pi*Int cl*dB/dT dnu is the T_e
     * anchor replacing the retired all-thermal thermostat. */
    if (a4_sym && g_a4_boot && s < g_a4_boot_n && g_a4_boot[s]) {
        for (int b = 0; b < nb; b++) {
            if (cl[b] <= 0.0) continue;
            double B_te = planck_bnu(T_e, g_lre_nu[b]);
            H += cl[b] * (Jb[b] - B_te) * g_lre_dnu[b];
        }
        return 4.0 * M_PI_VAL * H;
    }
    /* (a4_sym but bf not yet booted on this shell: fall through to the legacy
     * closure below — transient stabilizer only, retired per shell once
     * H_photo > LUMINA_A4_BOOT_FAC * C_ff. The stabilizer needs the FULL
     * chi_line (the eps-weighted channel is the ~4e-8-slope term that cannot
     * anchor); swap in the full array registered alongside. */
    const double *cl_leg = cl;
    if (a4_sym && g_lre_chi_line_full)
        cl_leg = &g_lre_chi_line_full[(size_t)s * nb];
    if (uv_mode) {
        for (int b = 0; b < nb; b++) {
            if (cl[b] <= 0.0) continue;
            double B_te = planck_bnu(T_e, g_lre_nu[b]);
            double Jeff = (Jb[b] < B_te) ? Jb[b] : B_te;
            H += cl[b] * (Jeff - B_te) * g_lre_dnu[b];
        }
        return 4.0 * M_PI_VAL * H;
    }
    for (int b = 0; b < nb; b++) {
        if (cl_leg[b] <= 0.0) continue;
        double nu = g_lre_nu[b];
        double B_lag = planck_bnu(Te_lag, nu);
        double B_te  = planck_bnu(T_e,   nu);
        double eta_lag = sf[b] * ct[b] - ca[b] * B_lag;     /* = χ_line·S_line_lag */
        if (eta_lag < 0.0) eta_lag = 0.0;
        double eta_pre = eta_lag + cl_leg[b] * (B_te - B_lag);
        double eta_flr = cl_leg[b] * B_te;                   /* no-pumping floor */
        double eta_eff = (eta_pre > eta_flr) ? eta_pre : eta_flr;
        H += (cl_leg[b] * Jb[b] - eta_eff) * g_lre_dnu[b];
    }
    return 4.0 * M_PI_VAL * H;
}

/* ============== A4 Stage 2.5: analytic energy-row Jacobian ==============
 * Closed-form dr1/dT_e and dr1/dln n_e for the coupled-Newton 2x2 block,
 * mirroring each term of r1 = radeq_net + Hresp + line_re EXACTLY (same
 * arrays, same clamp branches). Replaces the FD column (LUMINA_A4_ANALYTIC_
 * JAC=1): exact at clamp corners where FD smears, and global-ready — these
 * forms become the diagonal energy blocks of the Stage-4 block-tridiagonal
 * solve (the charge row stays shell-local FD; its dalpha/dT_e is carried by
 * the FD difference automatically). */

/* dB_nu/dT = B*(x/T)/(1-e^{-x}), x=h nu/kT; overflow-safe via planck_bnu
 * (e^{-x} underflow -> Wien dB/dT = B*x/T). */
static double planck_bnu_dT(double T, double nu) {
    double x = H_PLANCK * nu / (K_BOLTZMANN * T);
    double B = planck_bnu(T, nu);
    if (B <= 0.0 || !(x > 0.0)) return 0.0;
    double om = -expm1(-x);
    if (om < 1e-300) return 0.0;
    return B * (x / T) / om;
}

/* d/dT of radeq_recomb_cool = Sum emit*e^{-x}*x/T (x=h nu/kT). */
static double radeq_recomb_cool_dT(double T_e, const double *emit_nu,
                                   const double *nu_mid, int nbins) {
    double inv_kT = 1.0 / (K_BOLTZMANN * T_e);
    double sum = 0.0;
    for (int bb = 0; bb < nbins; bb++) {
        if (emit_nu[bb] == 0.0) continue;
        double x = H_PLANCK * nu_mid[bb] * inv_kT;
        sum += emit_nu[bb] * exp(-x) * x / T_e;
    }
    return sum;
}

/* d/dT of radeq_line_cool: per active (nonneg-surviving) line,
 * d[(n_e/sqrtT)(a e^{-b/T}-b)]/dT = (n_e/sqrtT)[a e^{-beta/T} beta/T^2]
 *                                  - (1/2T)(n_e/sqrtT)(a e^{-beta/T}-b). */
static double radeq_line_cool_dT(double T_e, double n_e,
                                 const double *a, const double *b,
                                 const double *beta, long n_active, int nonneg) {
    double sqrtTe = sqrt(T_e);
    double s_exp = 0.0, s_br = 0.0;
    for (long m = 0; m < n_active; m++) {
        double ae = a[m] * exp(-beta[m] / T_e);
        double br = ae - b[m];
        if (nonneg && br < 0.0) continue;
        s_exp += ae * beta[m];
        s_br  += br;
    }
    return n_e / sqrtTe * (s_exp / (T_e * T_e) - 0.5 * s_br / T_e);
}

/* d/dT of radeq_Hresp = Sum gbin*lstar*dB/dT (blag is the frozen J* — no
 * T_e dependence). */
static double radeq_Hresp_dT(double T_e, const double *gbin, const double *lstar,
                             const double *nu_mid, int nbins) {
    if (!gbin) return 0.0;
    double d = 0.0;
    for (int bb = 0; bb < nbins; bb++) {
        if (gbin[bb] == 0.0 || lstar[bb] == 0.0) continue;
        d += gbin[bb] * lstar[bb] * planck_bnu_dT(T_e, nu_mid[bb]);
    }
    return d;
}

/* d/dT of radeq_line_re, mirroring its branch ladder. Legacy + a4-booted:
 * -4pi*Int cl*dB/dT dnu (the no-pumping max() has the SAME slope cl*dB/dT on
 * both branches, so the derivative is smooth through the clamp). uv_mode
 * cooling-only min(J,B)-B: slope -cl*dB/dT only where J < B(T_e). */
static double radeq_line_re_dT(double T_e, int s) {
    if (!g_lre_chi_line || s < 0 || s >= g_lre_nshells) return 0.0;
    int nb = g_lre_nbins;
    const double *cl = &g_lre_chi_line[(size_t)s * nb];
    const double *Jb = &g_lre_J[(size_t)s * nb];
    static int uv_mode = -1, a4_sym = -1, a4_blend = -1;
    if (uv_mode < 0) {
        const char *u = getenv("LUMINA_CMFGEN_LINE_EPS_UV");
        uv_mode = (u && atof(u) > 0.0) ? 1 : 0;
        const char *a4 = getenv("LUMINA_A4_STAGE1");
        a4_sym = (a4 && atoi(a4)) ? 1 : 0;
        const char *ab = getenv("LUMINA_A4_SRC_BLEND");
        a4_blend = (ab && atoi(ab)) ? 1 : 0;
        if (a4_blend) { a4_sym = 0; uv_mode = 0; }
    }
    double d = 0.0;
    if (a4_blend) {
        const double *cl_use = NULL;
        int frozen = (frozenin_is_frozen && s < frozenin_is_frozen_n &&
                      frozenin_is_frozen[s]);
        if (!frozen && g_lre_chi_line_cls)
            cl_use = &g_lre_chi_line_cls[(size_t)s * nb];
        else if (g_lre_chi_line_full)
            cl_use = &g_lre_chi_line_full[(size_t)s * nb];
        if (cl_use) {
            for (int b = 0; b < nb; b++) {
                if (cl_use[b] <= 0.0) continue;
                d += cl_use[b] * planck_bnu_dT(T_e, g_lre_nu[b]) * g_lre_dnu[b];
            }
            return -4.0 * M_PI_VAL * d;
        }
    }
    if (a4_sym && g_a4_boot && s < g_a4_boot_n && g_a4_boot[s]) {
        for (int b = 0; b < nb; b++) {
            if (cl[b] <= 0.0) continue;
            d += cl[b] * planck_bnu_dT(T_e, g_lre_nu[b]) * g_lre_dnu[b];
        }
        return -4.0 * M_PI_VAL * d;
    }
    const double *cl_leg = cl;
    if (a4_sym && g_lre_chi_line_full)
        cl_leg = &g_lre_chi_line_full[(size_t)s * nb];
    if (uv_mode) {
        for (int b = 0; b < nb; b++) {
            if (cl[b] <= 0.0) continue;
            if (Jb[b] >= planck_bnu(T_e, g_lre_nu[b])) continue;
            d += cl[b] * planck_bnu_dT(T_e, g_lre_nu[b]) * g_lre_dnu[b];
        }
        return -4.0 * M_PI_VAL * d;
    }
    for (int b = 0; b < nb; b++) {
        if (cl_leg[b] <= 0.0) continue;
        d += cl_leg[b] * planck_bnu_dT(T_e, g_lre_nu[b]) * g_lre_dnu[b];
    }
    return -4.0 * M_PI_VAL * d;
}

static double radeq_net(double T_e, double T_rad, double n_e,
                        double H_photo, double H_gamma,
                        double compton_heat_coef, double ff_coef,
                        double Gamma_ad,
                        const double *a, const double *b, const double *beta,
                        long n_active, int nonneg, double C_line_const,
                        const double *emit_nu, const double *nu_mid, int nbins) {
    double H = H_photo + H_gamma + compton_heat_coef * (T_rad - T_e);

    double C = ff_coef * sqrt(T_e);                     /* free-free emission */
    C += 1.5 * n_e * K_BOLTZMANN * T_e * Gamma_ad;      /* adiabatic expansion */

    /* Bound-free radiative cooling (detailed-balance emission half; cancels
     * H_photo bin-by-bin at LTE since both carry n_lev over the same ν-grid) */
    C += radeq_recomb_cool(T_e, emit_nu, nu_mid, nbins);

    /* Bound-bound (line) cooling: T_e-independent radiative-escape constant when
     * the escape form is active (C_line_const), else the T_e-dependent
     * collisional-difference form over the active line set. */
    C += C_line_const;
    C += radeq_line_cool(T_e, n_e, a, b, beta, n_active, nonneg);

    return H - C;
}

typedef struct {
    const CpuOpacityPublication *op;
    const CpuEmissivityPublication *em;
    const double *J;
    const double *te_ref,*ne;
    const GammaDeposition *gamma;
    size_t ns,nb;
    double time_explosion;
} A210ProdContext;

static RadeqStatus a210_production_residual(size_t s,double te,
                                            A210TermLedger*l,void*opaque){
    A210ProdContext*c=(A210ProdContext*)opaque;
    if(!c||!l||s>=c->ns||!isfinite(te)||te<=0)return RADEQ_INVALID_TE_TRIAL;
    memset(l,0,sizeof(*l));
    for(int k=0;k<A210_NHEAT;k++)l->heating_status[k]=A210_EXACT_ZERO;
    for(int k=0;k<A210_NCOOL;k++)l->cooling_status[k]=A210_EXACT_ZERO;
    double photo_abs=0,line_abs=0,ff_abs=0,recomb=0,line_emit=0,ff_emit=0;
    double j_int=0,jnu_int=0;
    for(size_t b=0;b<c->nb;b++){
        size_t i=s*c->nb+b;double dnu=c->em->nu_edge[b+1]-c->em->nu_edge[b];
        double J=c->J[i];
        if(!isfinite(J)||J<0||!(dnu>0))return RADEQ_TERM_SCHEMA;
        photo_abs+=c->op->chi_bf[i]*J*dnu;
        line_abs+=c->op->chi_bb[i]*J*dnu;
        ff_abs+=c->op->chi_ff[i]*J*dnu;
        recomb+=c->em->eta_bf[i]*sqrt(c->te_ref[s]/te)*dnu;
        line_emit+=c->em->eta_bb[i]*dnu;
        ff_emit+=c->em->eta_ff[i]*sqrt(te/c->te_ref[s])*dnu;
        double nu=sqrt(c->em->nu_edge[b]*c->em->nu_edge[b+1]);
        j_int+=J*dnu;jnu_int+=J*nu*dnu;
    }
    const double fourpi=4.0*M_PI_VAL;
    photo_abs*=fourpi;line_abs*=fourpi;ff_abs*=fourpi;
    recomb*=fourpi;line_emit*=fourpi;ff_emit*=fourpi;
    if(photo_abs>=0){l->heating[A210_PHOTO]=photo_abs;l->heating_status[A210_PHOTO]=photo_abs?A210_INCLUDED:A210_EXACT_ZERO;}
    else{recomb-=photo_abs;l->heating_status[A210_PHOTO]=A210_INCLUDED;}
    l->A_line=line_abs;l->E_line=line_emit;l->m_line=1;
    l->radiative_line_included=1;l->collisional_or_escape_included=0;
    if(line_abs>=0)l->heating[A210_LINE_ABS]=line_abs;
    else l->cooling[A210_LINE_EMIT]-=line_abs;
    l->heating_status[A210_LINE_ABS]=line_abs?A210_INCLUDED:A210_EXACT_ZERO;
    l->cooling[A210_LINE_EMIT]+=line_emit;
    l->cooling_status[A210_LINE_EMIT]=line_emit?A210_INCLUDED:A210_EXACT_ZERO;
    if(ff_abs>=0)l->heating[A210_FF_ABS]=ff_abs;
    else ff_emit-=ff_abs;
    l->heating_status[A210_FF_ABS]=ff_abs?A210_INCLUDED:A210_EXACT_ZERO;
    l->cooling[A210_RECOMB]=recomb;l->cooling_status[A210_RECOMB]=recomb?A210_INCLUDED:A210_EXACT_ZERO;
    l->cooling[A210_FF_EMIT]=ff_emit;l->cooling_status[A210_FF_EMIT]=ff_emit?A210_INCLUDED:A210_EXACT_ZERO;
    /* Frequency-moment Compton exchange; never a T_rad proxy. */
    if(j_int>0){double trad=H_PLANCK*(jnu_int/j_int)/(4.0*K_BOLTZMANN);double q=4.0*K_BOLTZMANN*SIGMA_THOMSON*c->ne[s]/(9.1093837015e-28*C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)*fourpi*j_int*(trad-te);if(q>=0){l->heating[A210_COMPTON_H]=q;l->heating_status[A210_COMPTON_H]=q?A210_INCLUDED:A210_EXACT_ZERO;}else{l->cooling[A210_COMPTON_C]=-q;l->cooling_status[A210_COMPTON_C]=A210_INCLUDED;}}
    double qg=(c->gamma&&c->gamma->heating_rate)?c->gamma->heating_rate[s]:0;
    if(!isfinite(qg)||qg<0)return RADEQ_SIGN_MISMATCH;
    l->heating[A210_GAMMA]=qg;l->heating_status[A210_GAMMA]=qg?A210_INCLUDED:A210_EXACT_ZERO;
    l->heating_status[A210_NONTHERMAL]=A210_EXACT_ZERO;
    l->cooling[A210_ADIABATIC]=1.5*c->ne[s]*K_BOLTZMANN*te*(2.0/c->time_explosion);
    l->cooling_status[A210_ADIABATIC]=l->cooling[A210_ADIABATIC]?A210_INCLUDED:A210_EXACT_ZERO;
    l->cooling_status[A210_COLL_LINE]=A210_REPLACED_NOT_APPLICABLE;
    return a210_line_owner_finalize(l);
}

static int a210_rebin_checked_J(const RadiationFieldView*rf,
                                const CpuEmissivityPublication*em,double*out){
    if(!rf||!em||!out||!rf->frequency_bin_edges||!rf->J_nu)return-1;
    for(size_t s=0;s<em->n_shells;s++)for(size_t b=0;b<em->n_bins;b++){
        double lo=em->nu_edge[b],hi=em->nu_edge[b+1],integ=0,covered=0;
        for(size_t q=0;q<rf->n_bins;q++){double a=fmax(lo,rf->frequency_bin_edges[q]),z=fmin(hi,rf->frequency_bin_edges[q+1]);if(z<=a)continue;size_t qi=s*rf->n_bins+q;if(rf->validity[qi]!=RADIATION_FIELD_VALID&&rf->validity[qi]!=RADIATION_FIELD_EXACT_ZERO)return-1;integ+=rf->J_nu[qi]*(z-a);covered+=z-a;}
        if(fabs(covered-(hi-lo))>1e-10*(hi-lo))return-1;
        out[s*em->n_bins+b]=integ/(hi-lo);
    }return 0;
}

static int a210_production_solve(PlasmaState*plasma,GammaDeposition*gamma,
 NLTEConfig*nlte,AtomicData*atom,OpacityState*opacity,double epoch,int ns){
    A210Counters*ct=a210_counters();
    if(getenv("LUMINA_FIXED_TE_PROFILE")){ct->fixed_te_attempts++;return 0;}
    if(!plasma||!nlte||!atom||!opacity||ns<=0||
       nlte->radfield_view_status!=RADIATION_FIELD_VIEW_OK){ct->blocked_stale++;return 0;}
    const CpuOpacityPublication*op=&opacity->cpu_opacity;
    const CpuEmissivityPublication*em=&opacity->cpu_emissivity;
    if(!op->generation_committed||!em->committed_emissivity_generation||
       op->generation_committed!=em->opacity_generation||
       em->radfield_generation!=nlte->radfield_view.generation||
       em->population_generation!=atom->population_committed_generation||
       em->n_shells!=(size_t)ns||op->n_shells!=(size_t)ns||
       em->n_bins!=op->n_bins){ct->blocked_stale++;return 0;}
    size_t n=(size_t)ns,nb=em->n_bins;double*J=malloc(n*nb*sizeof(double));double*lo=malloc(n*sizeof(double));double*hi=malloc(n*sizeof(double));
    if(!J||!lo||!hi){free(J);free(lo);free(hi);return 0;}
    if(a210_rebin_checked_J(&nlte->radfield_view,em,J)){ct->blocked_missing_term++;free(J);free(lo);free(hi);return 0;}
    for(size_t s=0;s<n;s++){lo[s]=10.0;hi[s]=1.0e7;}
    char geo[65];RadiationField*f=&nlte->radiation_field.field;
    if(a210_geometry_sha256(f->shell_boundaries.values,f->shell_boundaries.count,geo)!=RADEQ_OK){ct->te_context_mismatch++;free(J);free(lo);free(hi);return 0;}
    A210ProdContext c={op,em,J,plasma->T_e,plasma->n_electron,gamma,n,nb,epoch};
    uint64_t gen=plasma->T_e_generation+1;if(gen==0)gen=1;
    int rc=a210_solve_transaction(&plasma->te_publication,lo,hi,plasma->n_electron,n,(uint64_t)llround(epoch),gen,geo,a210_production_residual,&c,plasma->T_e,plasma->n_electron);
    free(J);free(lo);free(hi);if(rc)return 0;
    plasma->te_publication.radfield_generation=nlte->radfield_view.generation;
    plasma->te_publication.bf_rate_generation=nlte->radfield_view.generation;
    plasma->te_publication.line_view_generation=nlte->line_view.generation;
    plasma->te_publication.population_generation=atom->population_committed_generation;
    plasma->te_publication.opacity_generation=op->generation_committed;
    plasma->te_publication.emissivity_generation=em->committed_emissivity_generation;
    return 1;
}

int compute_radiative_equilibrium_te(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                     NLTEConfig *nlte, AtomicData *atom,
                                     OpacityState *opacity,
                                     double time_explosion, int n_shells) {
    /* A2-10 is the only production Te path.  A failed transaction preserves
     * the previously committed material temperature; there is no scalar-
     * radiation fallback. */
    return a210_production_solve(plasma,gamma_dep,nlte,atom,opacity,
                                 time_explosion,n_shells);
}


void coupled_set_fine_jnu(const double *jnu, const double *nu, int n_fine,
                          double nu_lo, double dlognu, int n_shells) {
    (void)jnu;
    (void)nu;
    (void)n_fine;
    (void)nu_lo;
    (void)dlognu;
    (void)n_shells;
}

void coupled_newton_solve_all(PlasmaState *plasma, GammaDeposition *gamma_dep,
                              NLTEConfig *nlte, AtomicData *atom,
                              OpacityState *opacity, Geometry *geo,
                              double time_explosion, int n_shells) {
    (void)plasma;
    (void)gamma_dep;
    (void)nlte;
    (void)atom;
    (void)opacity;
    (void)geo;
    (void)time_explosion;
    (void)n_shells;
}

/* Compact "FeIII" / "CoIII" style label for the [METACOLL] banner. ion is the
 * 0-based stage (0=I). Falls back to "Z%d/ion%d" for elements outside the table. */
static void metacoll_ion_label(int Z, int ion, char *buf, size_t n) {
    static const struct { int Z; const char *sym; } S[] = {
        {6,"C"},{8,"O"},{11,"Na"},{12,"Mg"},{13,"Al"},{14,"Si"},{16,"S"},
        {20,"Ca"},{21,"Sc"},{22,"Ti"},{23,"V"},{24,"Cr"},{25,"Mn"},
        {26,"Fe"},{27,"Co"},{28,"Ni"} };
    static const char *R[] = {"I","II","III","IV","V","VI","VII","VIII"};
    const char *sym = NULL;
    for (size_t i = 0; i < sizeof(S)/sizeof(S[0]); i++)
        if (S[i].Z == Z) { sym = S[i].sym; break; }
    const char *rom = (ion >= 0 && ion < (int)(sizeof(R)/sizeof(R[0]))) ? R[ion] : NULL;
    if (sym && rom) snprintf(buf, n, "%s%s", sym, rom);
    else            snprintf(buf, n, "Z%d/ion%d", Z, ion);
}

/* Build every layout-derived NLTE projection in one place.  Both the normal
 * nlte_init() lane and the Wave-3.2 private shadow lane call this routine, so
 * offsets, bidirectional level maps, line ownership, super-level anchors, and
 * within-SL storage cannot drift independently. */
int nlte_build_projection(NLTEConfig *nlte, AtomicData *atom,
                          OpacityState *opacity, int n_shells,
                          const int *target_Z, const int *target_ion,
                          int n_targets, int super_requested,
                          int *n_lines_mapped) {
    int rc = -1;
    int *cursor = NULL;
    if (n_lines_mapped) *n_lines_mapped = 0;
    if (!nlte || !atom || !opacity || !target_Z || !target_ion ||
        n_targets <= 0 || n_targets > NLTE_MAX_IONS || n_shells <= 0)
        return -1;

    /* AtomicData owns line identities while OpacityState owns line-sized
     * arrays.  A split cardinality would make either projection out-of-bounds;
     * never guess which side is authoritative. */
    if (atom->n_lines != opacity->n_lines) {
        fprintf(stderr,
                "[NLTE][PROJECTION-FAIL] atom n_lines=%d != opacity n_lines=%d\n",
                atom->n_lines, opacity->n_lines);
        assert(atom->n_lines == opacity->n_lines);
        return -1;
    }

    memset(nlte->nlte_Z, 0, sizeof(nlte->nlte_Z));
    memset(nlte->nlte_ion, 0, sizeof(nlte->nlte_ion));
    memset(nlte->nlte_ion_level_offset, 0,
           sizeof(nlte->nlte_ion_level_offset));
    memset(nlte->nlte_ion_super_offset, 0,
           sizeof(nlte->nlte_ion_super_offset));
    nlte->n_nlte_ions = n_targets;
    for (int i = 0; i < n_targets; i++) {
        nlte->nlte_Z[i] = target_Z[i];
        nlte->nlte_ion[i] = target_ion[i];
        int count = 0, max_super = -1;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_Z[l] != target_Z[i] ||
                atom->level_ion[l] != target_ion[i]) continue;
            count++;
            if (atom->level_super[l] > max_super)
                max_super = atom->level_super[l];
        }
        nlte->nlte_ion_level_offset[i + 1] =
            nlte->nlte_ion_level_offset[i] + count;
        nlte->nlte_ion_super_offset[i + 1] =
            nlte->nlte_ion_super_offset[i] +
            (max_super >= 0 ? max_super + 1 : 0);
    }
    nlte->n_nlte_levels_total = nlte->nlte_ion_level_offset[n_targets];
    nlte->n_super_total = nlte->nlte_ion_super_offset[n_targets];
    nlte->super_mode = super_requested &&
                       nlte->n_super_total < nlte->n_nlte_levels_total;

    size_t nlevels = (size_t)(nlte->n_nlte_levels_total > 0 ?
                              nlte->n_nlte_levels_total : 1);
    size_t nsuper = (size_t)(nlte->n_super_total > 0 ?
                             nlte->n_super_total : 1);
    size_t nglobal = (size_t)(atom->n_levels > 0 ? atom->n_levels : 1);
    size_t nlines = (size_t)(atom->n_lines > 0 ? atom->n_lines : 1);
    nlte->nlte_to_global_level = (int *)malloc(nlevels * sizeof(int));
    nlte->global_to_nlte_level = (int *)malloc(nglobal * sizeof(int));
    nlte->nlte_line_map = (int *)malloc(nlines * sizeof(int));
    nlte->fl_to_super = (int *)malloc(nlevels * sizeof(int));
    nlte->super_anchor_global = (int *)malloc(nsuper * sizeof(int));
    nlte->within_sl_frac = (double *)malloc(
        nlevels * (size_t)n_shells * sizeof(double));
    nlte->nlte_level_populations = (double *)calloc(
        nlevels * (size_t)n_shells, sizeof(double));
    cursor = (int *)calloc((size_t)NLTE_MAX_IONS, sizeof(int));
    if (!nlte->nlte_to_global_level || !nlte->global_to_nlte_level ||
        !nlte->nlte_line_map || !nlte->fl_to_super ||
        !nlte->super_anchor_global || !nlte->within_sl_frac ||
        !nlte->nlte_level_populations || !cursor)
        goto cleanup;

    for (int l = 0; l < atom->n_levels; l++)
        nlte->global_to_nlte_level[l] = -1;
    for (int s = 0; s < nlte->n_super_total; s++)
        nlte->super_anchor_global[s] = -1;
    for (int l = 0; l < atom->n_levels; l++) {
        for (int i = 0; i < n_targets; i++) {
            if (atom->level_Z[l] != target_Z[i] ||
                atom->level_ion[l] != target_ion[i]) continue;
            int g = nlte->nlte_ion_level_offset[i] + cursor[i]++;
            int sl = nlte->nlte_ion_super_offset[i] + atom->level_super[l];
            nlte->nlte_to_global_level[g] = l;
            nlte->global_to_nlte_level[l] = g;
            nlte->fl_to_super[g] = sl;
            int prior = nlte->super_anchor_global[sl];
            if (prior < 0 || atom->level_energy_eV[l] <
                             atom->level_energy_eV[prior])
                nlte->super_anchor_global[sl] = l;
            break;
        }
    }
    for (int line = 0; line < atom->n_lines; line++) {
        nlte->nlte_line_map[line] = -1;
        for (int i = 0; i < n_targets; i++) {
            if (atom->line_atomic_number[line] != target_Z[i] ||
                atom->line_ion_number[line] != target_ion[i]) continue;
            nlte->nlte_line_map[line] = i;
            if (n_lines_mapped) (*n_lines_mapped)++;
            break;
        }
    }
    for (size_t k = 0; k < nlevels * (size_t)n_shells; k++)
        nlte->within_sl_frac[k] = 1.0;
    rc = 0;

cleanup:
    free(cursor);
    if (rc != 0) {
        free(nlte->nlte_to_global_level);
        free(nlte->global_to_nlte_level);
        free(nlte->nlte_line_map);
        free(nlte->fl_to_super);
        free(nlte->super_anchor_global);
        free(nlte->within_sl_frac);
        free(nlte->nlte_level_populations);
        nlte->nlte_to_global_level = NULL;
        nlte->global_to_nlte_level = NULL;
        nlte->nlte_line_map = NULL;
        nlte->fl_to_super = NULL;
        nlte->super_anchor_global = NULL;
        nlte->within_sl_frac = NULL;
        nlte->nlte_level_populations = NULL;
        fprintf(stderr, "[NLTE][PROJECTION-FAIL] allocation failed\n");
    }
    return rc;
}

int nlte_init(NLTEConfig *nlte, AtomicData *atom, OpacityState *opacity,
              int n_shells) {
    memset(nlte, 0, sizeof(NLTEConfig));
    nlte->enabled = 1;
    /* A2-05: memset zero would alias VIEW_OK; no field is published yet. */
    nlte->radfield_view_status = RADIATION_FIELD_VIEW_DISABLED;
    nlte->line_view_status = LINE_JBAR_VIEW_DISABLED; /* A2-06 same aliasing trap */
    nlte->n_freq_bins = NLTE_N_FREQ_BINS;
    nlte->nu_min = NLTE_NU_MIN;
    nlte->nu_max = NLTE_NU_MAX;
    nlte->d_log_nu = log(NLTE_NU_MAX / NLTE_NU_MIN) / NLTE_N_FREQ_BINS;

    /* Set up target ions. LUMINA_NLTE_STAGE4 promotes stage-IV ions (adjacent
     * layout, 38 slots); gate OFF => the original 31-slot table verbatim =>
     * byte-identical baseline (n_nlte_ions=31, loops below bounded by it, the 7
     * appended fixed-array slots untouched). */
    int element_wide = nlte_element_wide_layout_enabled();
    int stage4 = !element_wide && nlte_stage4_enabled();
    const int *TGT_Z = element_wide ? NLTE_TARGET_Z_EW :
                       (stage4 ? NLTE_TARGET_Z4 : NLTE_TARGET_Z);
    const int *TGT_ION = element_wide ? NLTE_TARGET_ION_EW :
                         (stage4 ? NLTE_TARGET_ION4 : NLTE_TARGET_ION);
    int n_targets = element_wide ? NLTE_EW_IONS :
                    (stage4 ? NLTE_STAGE4_IONS : NLTE_BASE_IONS);
    const char *super_env = getenv("LUMINA_SUPER_LEVELS");
    int super_requested = (super_env && atoi(super_env) != 0) || element_wide;
    int n_nlte_lines = 0;
    if (nlte_build_projection(nlte, atom, opacity, n_shells, TGT_Z, TGT_ION,
                              n_targets, super_requested,
                              &n_nlte_lines) != 0)
        return -1;

    printf("  [NLTE] Total NLTE levels: %d\n", nlte->n_nlte_levels_total);
    for (int i = 0; i < nlte->n_nlte_ions; i++) {
        int n = nlte->nlte_ion_level_offset[i + 1] - nlte->nlte_ion_level_offset[i];
        printf("    Z=%d ion=%d: %d levels\n", nlte->nlte_Z[i], nlte->nlte_ion[i], n);
    }
    if (stage4) {
        printf("  [STAGE4] LUMINA_NLTE_STAGE4=1: %d NLTE slots (base %d + stage-IV),"
               " promoted (III,IV) pairs:\n", nlte->n_nlte_ions, NLTE_BASE_IONS);
        for (int i = 0; i < nlte->n_nlte_ions; i++) {
            if (nlte->nlte_ion[i] != 3) continue;   /* ion_number 3 = spectroscopic IV */
            int n = nlte->nlte_ion_level_offset[i + 1] - nlte->nlte_ion_level_offset[i];
            printf("  [STAGE4]   Z=%d IV (slot %d): %d NLTE levels\n",
                   nlte->nlte_Z[i], i, n);
        }
    }
    if (element_wide)
        printf("  [EW] CPU pilot layout: %d slots; Fe/S II-IV are contiguous\n",
               nlte->n_nlte_ions);

    printf("  [NLTE] Super-levels: %s (%d FL -> %d SL across ions)\n",
           nlte->super_mode ? "ACTIVE" : "off (identity)",
           nlte->n_nlte_levels_total, nlte->n_super_total);

    /* Line ownership was built by the shared projection builder above. */
    int n_lines = opacity->n_lines;
    printf("  [NLTE] Lines mapped to NLTE ions: %d / %d\n",
           n_nlte_lines, n_lines);

    /* ---- COLD-CASE-P precompute: drainless-metastable flags (LUMINA_NLTE_METASTABLE_COLL).
     * A level is "drainless-metastable" iff levels.csv flags it metastable=1 AND no line
     * has it as its UPPER level (zero downward radiative transitions). Such levels cannot
     * de-excite in the baseline network (the per-line collision assembly loops over LINES,
     * so no-line => no collision channel either) -> b_k pileup. The flag is precomputed ONCE
     * here (global-level indexed); the assembler consults it only when the gate is on. */
    nlte->drainless_metastable = (int *)calloc(atom->n_levels, sizeof(int));
    {
        /* has_downward[global] = level l appears as the UPPER level of >=1 line. Mirror the
         * assembly-loop lookup (per-ion-pop level range + level_num match). */
        int *has_downward = (int *)calloc(atom->n_levels, sizeof(int));
        for (int line = 0; line < n_lines; line++) {
            int Z   = atom->line_atomic_number[line];
            int ion = atom->line_ion_number[line];
            int ip  = find_ion_pop_idx(atom, Z, ion);
            if (ip < 0) continue;
            int lb = atom->level_offset[ip], lt = atom->level_offset[ip + 1];
            int up_num = atom->line_level_upper[line];
            for (int l = lb; l < lt; l++)
                if (atom->level_num[l] == up_num) { has_downward[l] = 1; break; }
        }
        for (int l = 0; l < atom->n_levels; l++)
            nlte->drainless_metastable[l] =
                (atom->level_metastable[l] == 1 && !has_downward[l]) ? 1 : 0;
        free(has_downward);

        /* Banner: per-NLTE-ion count of drainless-metastable levels that will receive a
         * ground drain (EXCLUDING each ion's own ground level_num==0, which is the target
         * and cannot drain to itself). Printed only when the fix is armed. */
        if (nlte_metacoll_enabled()) {
            int    mmode  = nlte_metacoll_mode();
            double momega = (mmode == 2) ? nlte_metacoll_omega() : AXELROD_OMEGA;
            const char *coupling = (mmode == 2) ? "all-lower" : "ground";
            char line_buf[1024];
            int off = snprintf(line_buf, sizeof(line_buf),
                "  [METACOLL] mode=%d Omega=%.2f: drainless-metastable -> %s coupling;",
                mmode, momega, coupling);
            long total = 0;
            for (int i = 0; i < nlte->n_nlte_ions; i++) {
                int cnt = 0;
                int l0 = nlte->nlte_ion_level_offset[i];
                int l1 = nlte->nlte_ion_level_offset[i + 1];
                for (int nl = l0; nl < l1; nl++) {
                    int g = nlte->nlte_to_global_level[nl];
                    if (nlte->drainless_metastable[g] && atom->level_num[g] != 0) cnt++;
                }
                if (cnt > 0 && off < (int)sizeof(line_buf) - 24) {
                    char lab[16];
                    metacoll_ion_label(nlte->nlte_Z[i], nlte->nlte_ion[i], lab, sizeof(lab));
                    off += snprintf(line_buf + off, sizeof(line_buf) - off,
                                    " %s=%d", lab, cnt);
                    total += cnt;
                }
            }
            printf("%s  (total=%ld)\n", line_buf, total);
        }
    }

    /* Allocate results arrays */
    nlte->j_nu_estimator = (double *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    nlte->j_nu_count = (int *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(int));
    nlte->J_nu = (double *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    /* [ARTIS-PARITY C1/C2] per-bin MC field moment accumulators (always allocated;
     * only accumulated/consumed under LUMINA_ARTIS_PARITY => OFF path unaffected). */
    nlte->nu_bar_nu_estimator = (double *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));
    nlte->bf_rate_estimator = (double *)calloc(
        (size_t)n_shells * NLTE_N_FREQ_BINS, sizeof(double));

    printf("  [NLTE] Initialization complete. Memory: %.1f MB\n",
           ((double)nlte->n_nlte_levels_total * n_shells * 8 +
            (double)n_shells * NLTE_N_FREQ_BINS * 16) / 1048576.0);
    return 0;
}

void nlte_free(NLTEConfig *nlte) {
    a209_counters_print(stdout);
    a210_counters_print(stdout);
    population_counters_print(stdout, &nlte->population_counters);
    fflush(stdout);
    /* A2-05 R4/R6 observability: the zero-consumer gate and the blocked-term
     * audit read these totals from the run log (one line, always printed when
     * the view path was ever exercised). */
    if (nlte->bf_view_rate_terms || nlte->bf_view_blocked_stale ||
        nlte->bf_view_blocked_unsampled || nlte->bf_view_blocked_out_of_grid) {
        printf("[A2-05][BF-VIEW] rate_terms=%llu blocked_stale=%llu "
               "blocked_unsampled=%llu blocked_out_of_grid=%llu\n",
               (unsigned long long)nlte->bf_view_rate_terms,
               (unsigned long long)nlte->bf_view_blocked_stale,
               (unsigned long long)nlte->bf_view_blocked_unsampled,
               (unsigned long long)nlte->bf_view_blocked_out_of_grid);
        fflush(stdout);
    }
    if (nlte->bb_view_rate_terms || nlte->bb_view_blocked_stale ||
        nlte->bb_view_blocked_unsampled || nlte->bb_view_blocked_oog ||
        nlte->bb_view_blocked_miss || nlte->bb_view_blocked_profile ||
        nlte->bb_view_blocked_qhash || nlte->bb_view_blocked_disabled) {
        /* A2-06 observability */
        printf("[A2-06][BB-VIEW] rate_terms=%llu blocked_stale=%llu "
               "blocked_unsampled=%llu blocked_oog=%llu miss=%llu "
               "blocked_profile=%llu blocked_qhash=%llu blocked_disabled=%llu\n",
               (unsigned long long)nlte->bb_view_rate_terms,
               (unsigned long long)nlte->bb_view_blocked_stale,
               (unsigned long long)nlte->bb_view_blocked_unsampled,
               (unsigned long long)nlte->bb_view_blocked_oog,
               (unsigned long long)nlte->bb_view_blocked_miss,
               (unsigned long long)nlte->bb_view_blocked_profile,
               (unsigned long long)nlte->bb_view_blocked_qhash,
               (unsigned long long)nlte->bb_view_blocked_disabled);
        fflush(stdout);
    }
    free(nlte->nlte_to_global_level);
    free(nlte->global_to_nlte_level);
    free(nlte->nlte_line_map);
    free(nlte->drainless_metastable);
    free(nlte->nlte_level_populations);
    free(nlte->j_nu_estimator);
    free(nlte->j_nu_count);
    free(nlte->J_nu);
    free(nlte->nu_bar_nu_estimator);   /* [ARTIS-PARITY C1] */
    free(nlte->bf_rate_estimator);     /* [ARTIS-PARITY C2] */
    free(nlte->fl_to_super);
    free(nlte->super_anchor_global);
    free(nlte->within_sl_frac);
    free(g_ew_tau_authority);
    g_ew_tau_authority = NULL;
    g_ew_tau_authority_nshells = 0;
}

/* Normalize raw j_nu estimator to physical J_nu [erg/s/cm^2/Hz/sr] */
void nlte_normalize_j_nu(NLTEConfig *nlte, double time_simulation,
                          double *volume, int n_shells) {
    for (int s = 0; s < n_shells; s++) {
        for (int b = 0; b < nlte->n_freq_bins; b++) {
            int idx = s * nlte->n_freq_bins + b;
            double raw = nlte->j_nu_estimator[idx];

            /* Compute bin width in Hz */
            double log_nu_lo = log(nlte->nu_min) + b * nlte->d_log_nu;
            double log_nu_hi = log_nu_lo + nlte->d_log_nu;
            double delta_nu = exp(log_nu_hi) - exp(log_nu_lo);

            /* J_nu = j_raw / (4*pi * V * t_sim * delta_nu) */
            if (raw > 0.0 && volume[s] > 0.0 && delta_nu > 0.0) {
                nlte->J_nu[idx] = raw /
                    (4.0 * M_PI_VAL * volume[s] * time_simulation * delta_nu);
            } else {
                nlte->J_nu[idx] = 1e-30; /* floor */
            }
        }
    }
    /* Gate M0 (LUMINA_MC_JDUMP=1): dump the MC binned J_nu per (shell,bin) so it
     * can be cross-checked against the deterministic pure-CMFGEN J (LUMINA_CMFGEN_
     * JDUMP -> lumina_cmfgen_jnu.csv). Same grid/units. Agreement in continuum bins
     * validates that the MC reproduces pure-CMFGEN (the independent-method anchor);
     * divergence in line bins measures the binned-Sobolev resolution error. */
    {
        const char *md = getenv("LUMINA_MC_JDUMP");
        if (md && atoi(md)) {
            FILE *mf = fopen("lumina_mc_jnu.csv", "w");
            if (mf) {
                fprintf(mf, "shell,bin,nu,J_mc\n");
                for (int s = 0; s < n_shells; ++s)
                    for (int b = 0; b < nlte->n_freq_bins; ++b) {
                        double nu = exp(log(nlte->nu_min) + (b + 0.5) * nlte->d_log_nu);
                        fprintf(mf, "%d,%d,%.6e,%.6e\n", s, b, nu,
                                nlte->J_nu[(size_t)s * nlte->n_freq_bins + b]);
                    }
                fclose(mf);
                printf("[MC-JDUMP] wrote lumina_mc_jnu.csv (%d shells x %d bins)\n",
                       n_shells, nlte->n_freq_bins);
            }
        }
    }
}

/* Cap super-Planckian J_nu in UV (λ < lambda_max) at W_cap * B_nu(T_rad).
 * UV reprocessing in inner shells drives W to 1.7–2.7 (see
 * project_pathB_prime_jnu_diagnosis.md), pumping bound-free rates 1.7–2.7×
 * above what an undiluted Planck would give. Clipping J_nu directly attacks
 * the σ_bf saturation gap without touching ion-lock or atomic data.
 * Controlled by:
 *   LUMINA_J_NU_UV_CAP            (0/1, default 0)
 *   LUMINA_J_NU_UV_CAP_LAMBDA_MAX (Å,   default 3500)
 *   LUMINA_J_NU_UV_W_CAP          (-,   default 1.0 = Planckian) */
void nlte_apply_uv_jnu_cap(NLTEConfig *nlte, PlasmaState *plasma, int n_shells) {
    static int initialized = 0;
    static int enabled = 0;
    static double lambda_max_aa = 3500.0;
    static double lambda_min_aa = 2000.0;
    static double W_cap = 1.0;
    if (!initialized) {
        const char *env_en   = getenv("LUMINA_J_NU_UV_CAP");
        enabled = (env_en != NULL && atoi(env_en) != 0);
        const char *env_lmax = getenv("LUMINA_J_NU_UV_CAP_LAMBDA_MAX");
        if (env_lmax) lambda_max_aa = atof(env_lmax);
        const char *env_lmin = getenv("LUMINA_J_NU_UV_CAP_LAMBDA_MIN");
        if (env_lmin) lambda_min_aa = atof(env_lmin);
        const char *env_w    = getenv("LUMINA_J_NU_UV_W_CAP");
        if (env_w)   W_cap = atof(env_w);
        if (enabled) {
            printf("  [J_nu UV cap] enabled: %.0f A <= lambda <= %.0f A, W <= %.2f\n",
                   lambda_min_aa, lambda_max_aa, W_cap);
        }
        initialized = 1;
    }
    if (!enabled) return;

    /* λ_min < λ < λ_max  ⇔  nu_lo < nu < nu_hi  (note swap) */
    double nu_lo = C_SPEED_OF_LIGHT / (lambda_max_aa * 1e-8); /* Hz */
    double nu_hi = C_SPEED_OF_LIGHT / (lambda_min_aa * 1e-8); /* Hz */
    long long n_capped = 0;
    double sum_ratio = 0.0;
    double max_ratio = 0.0;

    for (int s = 0; s < n_shells; s++) {
        double T = plasma->T_e[s];
        if (T <= 0.0) continue;
        for (int b = 0; b < nlte->n_freq_bins; b++) {
            double nu_mid = nlte->nu_min * exp((b + 0.5) * nlte->d_log_nu);
            if (nu_mid < nu_lo || nu_mid > nu_hi) continue; /* out of band */
            int idx = s * nlte->n_freq_bins + b;
            double J = nlte->J_nu[idx];
            double Bnu = planck_bnu(T, nu_mid);
            double J_max = W_cap * Bnu;
            if (J_max > 0.0 && J > J_max) {
                double ratio = J / J_max;
                sum_ratio += ratio;
                if (ratio > max_ratio) max_ratio = ratio;
                nlte->J_nu[idx] = J_max;
                n_capped++;
            }
        }
    }
    if (n_capped > 0) {
        printf("  [J_nu UV cap] iter %d: capped %lld bins, "
               "mean J/J_max = %.2f, max = %.2f\n",
               nlte->current_iter, n_capped,
               sum_ratio / (double)n_capped, max_ratio);
    }
}

/* Interpolate J_nu at a given frequency from the histogram */
double nlte_get_J_at_nu(NLTEConfig *nlte, int shell, double nu) {
    if (nu <= nlte->nu_min || nu >= nlte->nu_max)
        return 1e-30;
    double log_ratio = log(nu / nlte->nu_min);
    int bin = (int)(log_ratio / nlte->d_log_nu);
    if (bin < 0) bin = 0;
    if (bin >= nlte->n_freq_bins) bin = nlte->n_freq_bins - 1;
    return nlte->J_nu[shell * nlte->n_freq_bins + bin];
}

/* Column-oriented Gaussian elimination with partial pivoting for Ax=b.
 * A is N x N column-major matrix, b is N x 1 RHS vector.
 * Inner loop iterates rows within a column = stride-1 = cache-friendly.
 * Solution returned in b. Returns 0 on success, -1 on singular matrix. */
static int gauss_solve(double *A, double *b, int N) {
    /* Column-major: A(i,j) = A[j*N + i] */
    for (int k = 0; k < N; k++) {
        /* Partial pivoting: find max in column k, rows k..N-1 (contiguous) */
        int max_row = k;
        double max_val = fabs(A[k * N + k]);
        for (int i = k + 1; i < N; i++) {
            double v = fabs(A[k * N + i]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val < 1e-300) return -1;

        /* Swap rows k and max_row across all columns + b */
        if (max_row != k) {
            for (int j = 0; j < N; j++) {
                double tmp = A[j * N + k];
                A[j * N + k] = A[j * N + max_row];
                A[j * N + max_row] = tmp;
            }
            double tmp = b[k]; b[k] = b[max_row]; b[max_row] = tmp;
        }

        /* Compute multipliers in column k (contiguous write) */
        double pivot_inv = 1.0 / A[k * N + k];
        for (int i = k + 1; i < N; i++)
            A[k * N + i] *= pivot_inv;

        /* Update trailing submatrix column-by-column (inner loop contiguous!) */
        for (int j = k + 1; j < N; j++) {
            double A_kj = A[j * N + k]; /* pivot row element in column j */
            for (int i = k + 1; i < N; i++)
                A[j * N + i] -= A[k * N + i] * A_kj;
        }

        /* Update RHS using multipliers */
        double b_k = b[k];
        for (int i = k + 1; i < N; i++)
            b[i] -= A[k * N + i] * b_k;

        /* Zero multipliers (restore matrix for back-substitution) */
        for (int i = k + 1; i < N; i++)
            A[k * N + i] = 0.0;
    }

    /* Back substitution */
    for (int k = N - 1; k >= 0; k--) {
        double sum = b[k];
        for (int j = k + 1; j < N; j++)
            sum -= A[j * N + k] * b[j];
        b[k] = sum / A[k * N + k];
    }
    return 0;
}

/* ============================================================ */
/* [DIAG-T3] NLTE per-level rate-channel decomposition for Fe     */
/* II/III/IV at LUMINA_DIAG_SHELL (default 8). For lo-ion X the    */
/* pair (X,X+1) assembly holds the COMPLETE rate picture for X:    */
/* all its intra-X bb rates + its photoion/recomb + collion/3-body.*/
/* We accumulate per (solve) level and, after that pair's assembly,*/
/* rewrite the whole file so lumina_rates_decomp.csv always holds  */
/* the latest solve sweep. A deviating b_k is then attributable to */
/* ONE rate channel. CPU assembly only (LUMINA_NLTE_ASSEMBLE_GPU=0)*/
/* and only under the master parity gate (=> OFF-path untouched).  */
/* ============================================================ */
#define DDC_NSTAGE 3    /* Fe II, III, IV */
#define DDC_MAXLEV 64
typedef struct {
    int    seen [DDC_NSTAGE][DDC_MAXLEV];
    int    levnum[DDC_NSTAGE][DDC_MAXLEV];
    double E_eV [DDC_NSTAGE][DDC_MAXLEV];
    int    g    [DDC_NSTAGE][DDC_MAXLEV];
    double pop  [DDC_NSTAGE][DDC_MAXLEV];
    double radup[DDC_NSTAGE][DDC_MAXLEV];   /* Σ R_absorb  (J-driven up)      */
    double raddn[DDC_NSTAGE][DDC_MAXLEV];   /* Σ R_stim+R_spont (A·β down)    */
    double colup[DDC_NSTAGE][DDC_MAXLEV];   /* Σ C_up   (coll excite up)      */
    double coldn[DDC_NSTAGE][DDC_MAXLEV];   /* Σ C_down (coll de-excite down) */
    double pion [DDC_NSTAGE][DDC_MAXLEV];   /* R_bf  photoionization out      */
    double rec  [DDC_NSTAGE][DDC_MAXLEV];   /* R_rec radiative recomb in      */
    double cion [DDC_NSTAGE][DDC_MAXLEV];   /* C_ion collisional ioniz out    */
    double c3b  [DDC_NSTAGE][DDC_MAXLEV];   /* C_rec 3-body recomb in         */
} DiagDecomp;
static DiagDecomp g_ddc;
static int g_ddc_shell = -2;   /* -2 unread, else the diag shell (-1 disables) */
static int nlte_diag_decomp_shell(void) {
    if (g_ddc_shell == -2) {
        const char *e = getenv("LUMINA_DIAG_SHELL");
        g_ddc_shell = e ? atoi(e) : 8;   /* default photosphere s8 */
    }
    return g_ddc_shell;
}

/* Assemble NLTE rate matrix for one ion pair in one shell.
 * Outputs column-major A_cm[N*N] and RHS b[N] (both must be pre-zeroed).
 * Called by both CPU (gauss_solve) and GPU (cuBLAS batched) paths. */
/* ============================================================ */
/* withParityP GATE ②: LUMINA_JBAR_DUMP — per-line consumed-jbar observer.     */
/* Pure observation: reads opacity->jbar_line / jbar_count at the exact         */
/* consumption point in nlte_assemble_rate_matrix (below) and appends a CSV     */
/* row. Never feeds back to any field/population. OFF => byte-identical.        */
/* ============================================================ */
static int   g_jbdump_on    = -1;   /* -1 unparsed; 0 off; 1 on (getenv once) */
static FILE *g_jbdump_fp    = NULL;
static int   g_jbdump_nions = 0;
static int   g_jbdump_Z[64];
static int   g_jbdump_ion[64];
static int   g_jbdump_arm   = 0;    /* armed by caller around the authoritative solve */
static int   g_jbdump_iter  = -1;   /* outer iteration recorded at arm time */
static int   g_jbdump_pass  = -1;   /* current CE inner pass (only 0 dumps) */

static void nlte_jbar_dump_init(void) {
    if (g_jbdump_on >= 0) return;                 /* env parsed exactly once */
    const char *e = getenv("LUMINA_JBAR_DUMP");
    g_jbdump_on = (e && atoi(e) != 0) ? 1 : 0;
    if (!g_jbdump_on) return;
    const char *f = getenv("LUMINA_JBAR_DUMP_IONS");
    /* Gate-B Phase 1.6 capture preset.  This only widens the existing observer
     * filter; it does not alter jbar, rates, populations, or transport. */
    const char *gateb = getenv("LUMINA_GATEB_ORACLE_CAPTURE");
    if (gateb && atoi(gateb) != 0)
        f = "14:1,14:2,16:1,16:2,26:1,26:2,26:3,27:2";
    if (!f || !*f) f = "14:2";                    /* default Si III (Z=14, nlte_ion=2) */
    const char *p = f;
    while (*p && g_jbdump_nions < 64) {
        int Z = 0, ion = -1;
        if (sscanf(p, "%d:%d", &Z, &ion) == 2 && Z > 0 && ion >= 0) {
            g_jbdump_Z[g_jbdump_nions]   = Z;
            g_jbdump_ion[g_jbdump_nions] = ion;
            g_jbdump_nions++;
        }
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    g_jbdump_fp = fopen("lumina_jbar_dump.csv", "w");
    if (g_jbdump_fp) {
        fprintf(g_jbdump_fp,
            "iter,shell,Z,ion,line_idx,lambda_A,jbar_line,jbar_count,beta,mode,B_planck_Te\n");
        fflush(g_jbdump_fp);
        fprintf(stderr, "[withParityP GATE2 LUMINA_JBAR_DUMP] ARMED -> "
                        "lumina_jbar_dump.csv; ions=");
        for (int i = 0; i < g_jbdump_nions; i++)
            fprintf(stderr, "%s%d:%d", i ? "," : "", g_jbdump_Z[i], g_jbdump_ion[i]);
        fprintf(stderr, "  (jbar_line=raw opacity->jbar_line array [the EMA-consumed "
                        "field]; mode 0=binned/nlte_get_J,1=m1_jbar_beta1,2=m2_diff,"
                        "3=m3_beta_jinc,4=dilute)\n");
    } else {
        fprintf(stderr, "[withParityP GATE2 LUMINA_JBAR_DUMP] *** FAILED to open "
                        "lumina_jbar_dump.csv — DUMP DISABLED ***\n");
        g_jbdump_on = 0;                          /* fail-loud, fail-closed */
    }
}

void nlte_jbar_dump_arm(int outer_iter) {
    nlte_jbar_dump_init();
    if (g_jbdump_on != 1) return;
    g_jbdump_arm  = 1;
    g_jbdump_iter = outer_iter;
    g_jbdump_pass = -1;
}
void nlte_jbar_dump_set_pass(int ce_pass) {
    if (g_jbdump_on != 1) return;
    g_jbdump_pass = ce_pass;
}
void nlte_jbar_dump_disarm(void) {
    if (g_jbdump_on != 1) return;
    g_jbdump_arm  = 0;
    g_jbdump_pass = -1;
    if (g_jbdump_fp) fflush(g_jbdump_fp);
}
/* True iff this (Z,ion) line should be dumped now: armed + CE pass 0 (one block
 * per outer iteration, since jbar_line is held fixed across CE passes). */
static int nlte_jbar_dump_want(int Z, int ion) {
    if (g_jbdump_on != 1 || !g_jbdump_arm || g_jbdump_pass != 0 || !g_jbdump_fp)
        return 0;
    for (int i = 0; i < g_jbdump_nions; i++)
        if (g_jbdump_Z[i] == Z && g_jbdump_ion[i] == ion) return 1;
    return 0;
}

void nlte_assemble_rate_matrix(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                int ion_idx_lo, int ion_idx_hi,
                                int shell, double time_explosion,
                                double *A_cm, double *b, int N,
                                GammaDeposition *gamma_dep,
                                const NLTERateLookup *lookup,
                                int pair_idx) {
    const int ew_capture = nlte_ew_capture_active();
    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int n_shells = plasma->n_shells;
    double T_rad = plasma->T_e[shell];
    double T_e   = plasma->T_e[shell];
    double n_e   = plasma->n_electron[shell];
#ifdef LUMINA_FROZEN_ORACLE
    if (g_oracle.fp) {
        int os = oracle_ion_slot(nlte->nlte_Z[ion_idx_lo],
                                 nlte->nlte_ion[ion_idx_lo]);
        if (os >= 0) g_oracle.bf_rate_seen[os] = 1;
    }
#endif
    /* Real Fe III collisions from Zhang col_data active for this call? (gate ON
     * AND table loaded). Used to suppress the per-line proxy + METACOLL for Fe
     * III and to run the dedicated Zhang pass below. */
    int feiii_coldata_on = (nlte_feiii_coldata_enabled() || artis_parity_enabled())
                           && atom->feiii_col_loaded;
    int parity_on = artis_parity_enabled();
    /* [OMEGA-CMFGEN] per-transition 3-tier Omega for the bb collision block. */
    const int omcm_on = omega_cmfgen_enabled() && parity_on;

    /* Column-major access: ACM(i,j) = A_cm[j*N + i] */
    #define ACM(i,j) A_cm[(j) * N + (i)]

    /* Super-level solve indexing. In identity mode fl_to_super[g]==g and
     * super_start==lev_start, so SOLVE_OF reduces to the FL nlte index and
     * FRAC_OF==1 — the matrix is byte-identical to the FL solve. In super mode
     * the matrix dim N is the super-level count and FL contributions aggregate
     * into their SL row/col, weighted by the within-SL Boltzmann fraction. */
    int super_start = nlte->nlte_ion_super_offset[ion_idx_lo];
    int n_lo_super  = nlte->nlte_ion_super_offset[ion_idx_lo + 1] - super_start;
    #define SOLVE_OF(fl_g) (nlte->fl_to_super[(fl_g)] - super_start)
    #define FRAC_OF(fl_g)  (nlte->within_sl_frac[(size_t)(fl_g) * n_shells + shell])

    /* [DIAG-T3] arm the per-level rate decomposition for this pair iff it is the
     * lo-ion of a Fe II/III/IV stage at the diag shell (=> complete picture). We
     * key the accumulator by the LO-ion solve index (== within-ion level in
     * identity mode). Reset this stage's slot on entry; the pair recomputes it
     * fully. Off unless the master parity gate is on. */
    int ddc_stage = -1;
    if (artis_parity_enabled()) {
        int ds = nlte_diag_decomp_shell();
        if (ds >= 0 && shell == ds && nlte->nlte_Z[ion_idx_lo] == 26) {
            int io = nlte->nlte_ion[ion_idx_lo];
            /* Lumina nlte_ion is 0-based: Fe II=1, III=2, IV=3 (levelpop.csv
             * convention) — not the ARTIS 1-based ionstage. */
            if (io >= 1 && io <= 3) {
                ddc_stage = io - 1;
                for (int L = 0; L < DDC_MAXLEV; L++) {
                    g_ddc.seen[ddc_stage][L] = 0;
                    g_ddc.pop[ddc_stage][L]  = 0.0;
                    g_ddc.radup[ddc_stage][L] = g_ddc.raddn[ddc_stage][L] = 0.0;
                    g_ddc.colup[ddc_stage][L] = g_ddc.coldn[ddc_stage][L] = 0.0;
                    g_ddc.pion[ddc_stage][L]  = g_ddc.rec[ddc_stage][L]   = 0.0;
                    g_ddc.cion[ddc_stage][L]  = g_ddc.c3b[ddc_stage][L]   = 0.0;
                }
            }
        }
    }
    #define DDC_CAP(i) ((i) >= 0 && (i) < n_lo_super && (i) < DDC_MAXLEV)
    #define DDC_RAD(ilo, iup, rup, rdn) do { if (ddc_stage >= 0) { \
        if (DDC_CAP(ilo)) g_ddc.radup[ddc_stage][ilo] += (rup); \
        if (DDC_CAP(iup)) g_ddc.raddn[ddc_stage][iup] += (rdn); } } while (0)
    #define DDC_COLL(ilo, iup, cu, cd) do { if (ddc_stage >= 0) { \
        if (DDC_CAP(ilo)) g_ddc.colup[ddc_stage][ilo] += (cu); \
        if (DDC_CAP(iup)) g_ddc.coldn[ddc_stage][iup] += (cd); } } while (0)

    /* α #286 floor-pop regularization: track bb-connectivity per level so we
     * can detect bb-isolated upper-ion levels (e.g. Cr/Fe/Co III top) that
     * cause conservation-row collapse (#219e). Allocated only when knob is on. */
    static int floor_reg_init = 0;
    static int floor_reg_mode = 0;
    if (!floor_reg_init) {
        const char *e = getenv("LUMINA_NLTE_FLOOR_REG");
        if (e && atoi(e) != 0) floor_reg_mode = 1;
        floor_reg_init = 1;
    }
    int *bb_connected = (floor_reg_mode && !ew_capture) ?
        (int *)calloc(N, sizeof(int)) : NULL;
    /* TOPSTAGE_THERMALIZE: per-level max Sobolev tau of connecting bb lines, so the
     * post-solve anchor forces Boltzmann@T_e on the TOP NLTE stage's bb-connected
     * EXCITED (upper-ion) levels — the super-thermal optical carriers O/C/S/Al III.
     * ROOT (2026-06-16, instrumented): these carriers are excited-EXCITED lines,
     * THIN in the thermal limit (lower level Boltzmann-suppressed, tau~1e-50) —
     * they look thick only BECAUSE they are super-thermal. So a tau-gate is the
     * wrong criterion; the right target is the over-populated EXCITED LEVEL itself.
     * The rate solve can't reach Boltzmann here (capped bf, sub-critical collisions,
     * no continuum partner for the top stage) — FORCE_LTE proved Boltzmann@T_e is
     * the correct target (-> gold-like MC features). Top-stage detected generically
     * (no NLTE pair has ion_hi+1). Optional departure gate LUMINA_TOPSTAGE_DEPARTURE
     * (>0: only anchor levels whose lagged pop exceeds Boltzmann by that factor;
     * 0 default: anchor all top-stage excited levels). Gated default off. */
    static int tsth_init = 0, tsth_mode = 0; static double tsth_dep = 0.0;
    if (!tsth_init) {
        const char *e = getenv("LUMINA_TOPSTAGE_THERMALIZE");
        if (e && atoi(e) != 0) tsth_mode = 1;
        const char *t = getenv("LUMINA_TOPSTAGE_DEPARTURE"); if (t) tsth_dep = atof(t);
        tsth_init = 1;
    }
    int tsth_on = tsth_mode && floor_reg_mode;   /* needs bb_connected tracking */
    /* hi is the TOP NLTE stage iff no NLTE ion is (same Z, ion_hi+1). */
    int hi_is_topstage = 0;
    if (tsth_on) {
        hi_is_topstage = 1;
        int Zh = nlte->nlte_Z[ion_idx_hi], ih = nlte->nlte_ion[ion_idx_hi];
        for (int i = 0; i < nlte->n_nlte_ions; i++)
            if (nlte->nlte_Z[i] == Zh && nlte->nlte_ion[i] == ih + 1) { hi_is_topstage = 0; break; }
    }

    /* Rate-budget diagnostic setup (LUMINA_NLTE_BUDGET_DUMP). */
    static int budget_init = 0, budget_on = 0, budget_Z = 8, budget_stage = 2, budget_shell = 8;
    static int budget_lines_hdr = 0, budget_rec_hdr = 0;
    if (!budget_init) {
        const char *e = getenv("LUMINA_NLTE_BUDGET_DUMP");
        budget_on = (e && atoi(e) != 0);
        const char *z  = getenv("LUMINA_BUDGET_Z");     if (z)  budget_Z = atoi(z);
        const char *st = getenv("LUMINA_BUDGET_STAGE"); if (st) budget_stage = atoi(st);
        const char *sh = getenv("LUMINA_BUDGET_SHELL"); if (sh) budget_shell = atoi(sh);
        budget_init = 1;
    }

    /* Collisional floor (LUMINA_NLTE_COLL_FLOOR=ε, default 0=off). At low n_e the
     * van Regemorter/Axelrod collision rate C∝n_e→0, the bf edge sits above the
     * (cold) radiation cutoff so R_bf=0 for low levels, and the bb radiative net
     * is near-conservative at J≈B — the rate matrix loses rank and the LU solve
     * returns a FLAT null-space vector (every level ≈ n_total/N). Flat pops give
     * b_k=1e68 at the near-continuum levels → super-thermal S_l → too-blue spectrum.
     * Floor C_up at ε·A_ul so a minimal thermalizing collision always couples the
     * level pair; C_down is derived from C_up by the exact detailed-balance ratio
     * below, so the floor drives n_u/n_l → Boltzmann@T_e (the correct thermalization
     * limit), NOT the flat garbage. ε≪1 is negligible wherever real rates exist. */
    static double coll_floor = -1.0;
    if (coll_floor < 0.0) {
        const char *e = getenv("LUMINA_NLTE_COLL_FLOOR");
        coll_floor = e ? atof(e) : 0.0;
        if (coll_floor < 0.0) coll_floor = 0.0;
    }
    /* Fire on the PAIR that contains the target stage as EITHER ion: O III is
     * only ever the upper ion (no O IV target), so its bb lines live in the
     * (O II, O III) pair as ion_idx_hi. */
    /* MALI gate (Sobolev escape on bb radiative rates). */
    static int mali_init = 0, mali_on = 0;
    if (!mali_init) {
        const char *e = getenv("LUMINA_MALI");
        mali_on = (e && atoi(e) != 0);
        mali_init = 1;
    }
    /* Stage A (LUMINA_NLTE_JBAR_POPS=1): drive the bb absorption rate from the
     * per-line REALIZED MC field J_bar_l (opacity->jbar_line) instead of the
     * frequency-averaged binned ambient J. This is the untouched lever: the
     * binned J washes out the line-resolved UV contrast before the rate matrix
     * sees it -> thermal populations -> no fluorescence. J_bar_l carries the
     * non-thermal UV pump, so the POPULATIONS become fluorescent. Explosion-safe
     * because J_bar_l is the realized EXTERNAL field held FIXED during the solve
     * (lagged Lambda-iteration, ARTIS-style), NOT the intra-solve self-coupling
     * that exploded in 165510. J_bar_l already includes the trapped (1-beta)S_l
     * via the packet crossings, so the rates use NO extra beta (codex form
     * R_lu=B_lu*Jbar, R_ul=A_ul+B_ul*Jbar). Undersampled lines fall back to the
     * binned J. */
    static int jbar_pops_init = 0, jbar_pops_mode = 0;
    if (!jbar_pops_init) {
        const char *e = getenv("LUMINA_NLTE_JBAR_POPS");
        jbar_pops_mode = e ? atoi(e) : 0;  /* 0 off; 1 naive(SEALED); 2 differenced(SEALED); 3 β·J_inc faithful Sobolev */
        jbar_pops_init = 1;
    }

    /* Dilute photospheric radiation field for the bb NLTE excitation rate
     * (LUMINA_NLTE_DILUTE_FIELD=1, default 0=off -> byte-identical baseline).
     * ARTIS/TARDIS-faithful fix for the LTE-like excitation (b_k~=1): the bb
     * radiative rate is driven by the LOCAL thermalized binned J (measured
     * J_bar/B(T_e)~=0.97 even where W~=0.43) -> detailed balance forces b_k=1
     * -> too-red LTE spectra. Instead drive the rate with the DILUTE PHOTOSPHERIC
     * blackbody J_bar_line = W(s)*B(nu_line,T_R), where T_R is a radiation color
     * temperature DECOUPLED from the local T_e (only ~T_e deep in the thermalized
     * core; hotter = photospheric color in the line-forming layers) and W(s)<1 the
     * geometric dilution. With J_bar = W*B(nu,T_R) != B(nu,T_e), detailed balance no
     * longer pins b_k=1 -> genuine NLTE excitation (artis-ref/radfield.cc:717-732,
     * macroatom.cc:571-604). T_R source: plasma->T_rad is the binned-J dilute-Planck
     * COLOR fit, but it has thermalized to the local T_e in the line-forming layers
     * (measured T_rad~=T_e) so it is NOT the hot photospheric color. We instead take
     * the INNERMOST shell's radiation temperature (shell 0 = smallest r_inner) as the
     * photospheric T_inner proxy -- deepest/hottest, W~=1, fully thermalized to the
     * inner-boundary BB -- and carry it outward with the per-shell dilution W(s).
     * T_inner itself lives in MCConfig (not passed to this routine); the innermost-
     * shell T_rad is the faithful, wiring-free proxy for it. Override the color with
     * LUMINA_NLTE_DILUTE_TR_K (float, K) for testing; default = innermost T_rad. */
    static int dilute_init = 0, dilute_on = 0; static double dilute_tr_override = 0.0;
    if (!dilute_init) {
        const char *e = getenv("LUMINA_NLTE_DILUTE_FIELD");
        dilute_on = (e && atoi(e) != 0);
        const char *t = getenv("LUMINA_NLTE_DILUTE_TR_K");
        if (t) dilute_tr_override = atof(t);
        dilute_init = 1;
    }
    double dilute_W = 0.0; /* scalar pump removed; checked J_nu owns rates */
    double dilute_TR = dilute_on
        ? ((dilute_tr_override > 0.0) ? dilute_tr_override : plasma->T_e[0])
        : 0.0;

    int budget_hit = budget_on && (nlte->nlte_Z[ion_idx_lo] == budget_Z) &&
                     (nlte->nlte_ion[ion_idx_lo] == budget_stage ||
                      nlte->nlte_ion[ion_idx_hi] == budget_stage) &&
                     (shell == budget_shell);

    /* ---- Radiative bound-bound rates from line data ---- */
    int n_lines = opacity->n_lines;
    if (!nlte_assemble_skip_bb())
    for (int line = 0; line < n_lines; line++) {
        int map = nlte->nlte_line_map[line];
        if (map < ion_idx_lo || map > ion_idx_hi) continue;

        int ion_s = atom->line_ion_number[line];
        int ip = find_ion_pop_idx(atom, atom->line_atomic_number[line], ion_s);
        if (ip < 0) continue;
        int lev_base = atom->level_offset[ip];
        int lev_top  = atom->level_offset[ip + 1];

        int lower_global = -1, upper_global = -1;
        for (int l = lev_base; l < lev_top; l++) {
            if (atom->level_num[l] == atom->line_level_lower[line]) lower_global = l;
            if (atom->level_num[l] == atom->line_level_upper[line]) upper_global = l;
            if (lower_global >= 0 && upper_global >= 0) break;
        }
        if (lower_global < 0 || upper_global < 0) continue;

        int fl_lo_g = nlte->global_to_nlte_level[lower_global];
        int fl_up_g = nlte->global_to_nlte_level[upper_global];
        int i_lo = SOLVE_OF(fl_lo_g);
        int i_up = SOLVE_OF(fl_up_g);
        if (i_lo < 0 || i_lo >= N || i_up < 0 || i_up >= N) continue;

        double nu_line = atom->line_nu[line];
        /* Mode 2 forms a differenced estimator bJext=J_bar-(1-beta)S_l (two large
         * near-equal numbers for thick lines), so it needs more crossings than the
         * raw mode-1/branching consumer to keep MC noise from biasing the clamped
         * pump positive (codex 9th-strike flag). */
        /* LUMINA_JBAR_MIN overrides the min MC crossings to trust jbar_line: lower it so
         * SPARSE-line ions (S II/Ca II/Si II — the optical carriers, few UV lines) still pump
         * off the transported field instead of falling back to the thermal binned J (which
         * pins b_k=1). Fe II (dense forest) already clears the default 10. Default -> 10/50. */
        static int jbar_min_env = -2;
        if (jbar_min_env == -2) { const char *e = getenv("LUMINA_JBAR_MIN");
            jbar_min_env = (e && atoi(e) > 0) ? atoi(e) : -1; }
        int jbar_min = jbar_min_env > 0 ? jbar_min_env : ((jbar_pops_mode == 2) ? 50 : 10);
        double J_jbar = (jbar_pops_mode && opacity->jbar_line && opacity->jbar_count &&
                         opacity->jbar_count[(size_t)line * n_shells + shell] >= jbar_min)
                        ? opacity->jbar_line[(size_t)line * n_shells + shell] : -1.0;
        /* FALSIFIER (LUMINA_JBAR_SRC_BINNED=1, codex 2026-06-22): the sealed mode-3
         * explosion (S_l/B=3.34e71, 167752) is hypothesized to be the CONTAMINATED MC
         * jbar_line (residence/trapped-packet over-count ~1e85 in thick cells), NOT the
         * rate form. Swap the MC source for the BOUNDED binned continuum J at the line:
         * if the explosion vanishes, the mode-3 algebra is correct and the MC input was
         * the sole failure => gate 5 = keep mode-3, feed deterministic Sobolev J_inc. */
        static int jbar_src_binned = -1;
        if (jbar_src_binned < 0) { const char *e = getenv("LUMINA_JBAR_SRC_BINNED");
            jbar_src_binned = (e && atoi(e)) ? 1 : 0; }
        if (jbar_src_binned && jbar_pops_mode)
            J_jbar = nlte_get_J_at_nu(nlte, shell, nu_line);   /* bounded, no MC over-count */

        /* P7 CONSUMER (LUMINA_CMF_LINERES_JBAR=1, gate II-3): the fine-grid
         * deterministic full J_bar_l (cmfgen_fine_jbar) OVERRIDES the MC estimator.
         * It is the FULL line field (1-beta)S_l + beta*J_ext with NO MC noise, so the
         * mode-2 differenced external pump bJext = J_bar_l - (1-beta)S_lag = beta*J_ext
         * is exact (no crossing-count guard) and captures cross-line forest overlap
         * that a per-line Sobolev beta*J_inc misses. In-window lines route through
         * mode-2 (forced below). Out-of-window lines (sentinel -1) are left UNTOUCHED
         * (original MC source + configured mode) so an A/B isolates the in-window
         * deterministic effect. */
        static int lineres_jbar_pops = -1;
        if (lineres_jbar_pops < 0) { const char *e = getenv("LUMINA_CMF_LINERES_CONSUME");
            lineres_jbar_pops = (e && atoi(e)) ? 1 : 0; }
        int det_jbar = 0;
        if (lineres_jbar_pops && opacity->jbar_line_det) {
            /* self-contained: LINERES_JBAR alone activates the deterministic mode-2
             * pump on in-window lines, independent of the (sealed) JBAR_POPS modes.
             * Out-of-window lines keep whatever the configured path is (binned by
             * default), so an A/B isolates the in-window deterministic effect. */
            double vdet = opacity->jbar_line_det[(size_t)line * n_shells + shell];
            if (vdet >= 0.0 && isfinite(vdet)) {
                J_jbar = vdet; det_jbar = 1;
                static int det_announced = 0;
                if (!det_announced) { det_announced = 1; fprintf(stderr,
                    "[cmf_consume] deterministic line-resolved J_bar mode-2 ACTIVE "
                    "(in-window UV-pump lines)\n"); }
            }
        }
        /* DETAILED-BALANCE FALSIFIER (LUMINA_NLTE_JEQB=1): force the bb radiation field
         * to the LOCAL Planck B(nu,Te). By Einstein relations + Planck, J=B(Te) makes
         * R_lu/R_ul = Boltzmann in EVERY mode/beta, so b_k MUST -> 1 (S_l/B -> 1) for a
         * DB-respecting bb network. If S_l/B stays >1 here, a bb rate VIOLATES detailed
         * balance = a definitive bug (this is a theorem, not an interpretive metric). */
        static int jeqb = -1;
        if (jeqb < 0) { const char *e = getenv("LUMINA_NLTE_JEQB"); jeqb = (e && atoi(e)) ? 1 : 0; }
        if (jeqb) { J_jbar = planck_bnu(T_e, nu_line); det_jbar = 0; }
        int use_jbar = (J_jbar > 0.0 && isfinite(J_jbar));  /* guard MC outliers/NaN */

        /* MALI (LUMINA_MALI=1): multiply the bound-bound radiative rates by the
         * Sobolev escape probability β_esc(τ). For a thick line (β→0) the
         * radiative coupling vanishes and the detailed-balanced collisional pair
         * (C_up/C_down) sets n_u/n_l → Boltzmann → S_l→B(T_e) (thermalized); thin
         * lines (β→1) are unchanged. β cancels in the Einstein ratios so detailed
         * balance is preserved. */
        double mali_beta = 1.0;
        if (mali_on) {
            double tau_l = opacity->tau_sobolev
                ? opacity->tau_sobolev[(size_t)line * n_shells + shell] : 0.0;
            mali_beta = radeq_beta_esc(tau_l);
        }

        /* A2_06_DIAGNOSTIC_SHADOW_BEGIN: legacy source/mode arithmetic remains
         * observable, but is overwritten before matrix/rate consumers. */
        /* effective line field (for the budget diagnostic + the mode-1/binned path) */
        double J_line = use_jbar ? J_jbar : nlte_get_J_at_nu(nlte, shell, nu_line);
        double R_absorb, R_stim, R_spont;
        /* [withParityP GATE2] which assembly branch + escape factor is applied to
         * this line (set inside each branch below; dumped after the chain). */
        int    jd_mode = -1; double jd_beta = 1.0;
        if (dilute_on) {
            /* Dilute photospheric field (LUMINA_NLTE_DILUTE_FIELD): drive the bb
             * radiative rates with J_bar_line = W(s)*B(nu_line,T_R), T_R the hot
             * photospheric color decoupled from the local T_e. The SAME J_bar feeds
             * absorption (B_lu*J_bar) and stimulated emission (B_ul*J_bar);
             * spontaneous A_ul is untouched. Because J_bar != B(nu,T_e) the
             * Einstein-relation detailed balance no longer forces n_u/n_l to
             * Boltzmann@T_e, so b_k departs from 1 (the ARTIS NLTE mechanism). */
            double Jbar = dilute_W * planck_bnu(dilute_TR, nu_line);
            R_absorb = atom->line_B_lu[line] * Jbar;
            R_stim   = atom->line_B_ul[line] * Jbar;
            R_spont  = atom->line_A_ul[line];
            J_line   = Jbar;   /* keep the budget diagnostic consistent */
            jd_mode = 4; jd_beta = 1.0;   /* [withParityP GATE2] dilute W*B(T_R) */
        } else if (use_jbar && jbar_pops_mode == 3 && !det_jbar) {
            /* Stage A v3 — faithful Sobolev/MALI with the INCIDENT field.
             * KEY (verified, 2026-06-21): the Lucy j_blue estimator jbar_line is
             * the incident mean intensity J_inc at the line frequency, NOT the
             * full trapped J_bar. Because the comoving frequency redshifts
             * monotonically (homologous expansion), a packet crosses each line at
             * most once and a packet emitted BY this line can never re-cross it
             * (in any shell) — so the estimator STRUCTURALLY excludes this line's
             * self-emission and carries only continuum + bluer-line (cross-line /
             * forest-overlap) photons = the genuine external UV pump. The exact
             * Sobolev two-level net rate is
             *     net = β[n_l B_lu J_inc − n_u(A_ul + B_ul J_inc)],
             * the trapped (1−β)S_l self-term cancelling analytically (the MALI
             * identity). So we apply the escape factor β DIRECTLY to the
             * incident-field pump. This self-limits at thick lines (β→0 ⇒
             * pump→0), fixing the sealed mode-1/2 runaway — whose real bug was
             * the MISSING β: mode 2's bJext=J_jbar−(1−β)S_lag ≈ J_jbar at thick
             * lines (the subtraction is negligible when J_inc≫S_lag) and it was
             * fed with NO escape factor → R_absorb=B_lu·J_inc → 167719/167720
             * 4.5e70. Here β·J_inc keeps the UV pump at the τ~0.5−3 feature
             * layers (β~0.6, F-ρ: 84% of 4475 Fe lines pump-survive) so the
             * populations FLUORESCE, while thick lines thermalize (β→0). */
            double tau_l = opacity->tau_sobolev
                ? opacity->tau_sobolev[(size_t)line * n_shells + shell] : 0.0;
            double beta = radeq_beta_esc(tau_l);
            R_absorb = atom->line_B_lu[line] * beta * J_jbar;
            R_stim   = atom->line_B_ul[line] * beta * J_jbar;
            R_spont  = atom->line_A_ul[line] * beta;
            jd_mode = 3; jd_beta = beta;  /* [withParityP GATE2] m3 beta*J_inc */
        } else if (use_jbar && (jbar_pops_mode == 2 || det_jbar)) {
            /* Stage A v2 — Λ*-preconditioned / faithful Sobolev-MALI with the
             * REALIZED external field. The MC estimator carries the FULL line
             * field J_bar = (1−β)S_l + β·J_ext. The sealed mode 1 fed it whole
             * (no escape factor) so the trapped (1−β)S_l self-term pumped n_up
             * with no damping → the 167719 runaway (S_l/B=4.5e70). Remove the
             * self-trap to recover the external UV pump  β·J_ext = J_bar −
             * (1−β)S_l_lagged, and keep the escape factor on spontaneous decay.
             * Net rate = β[n_l B_lu J_ext − n_u(A_ul + B_ul J_ext)] = the exact
             * Sobolev/MALI form, but J_ext is the line-resolved realized UV (not
             * the thermal binned J), so the populations FLUORESCE. Stable because
             * the β·A_ul down-channel self-limits the pump (F-ρ: β·A_ul>C_down at
             * the τ~0.5−3 feature-forming layers, 84% of 4475 Fe lines). */
            double tau_l = opacity->tau_sobolev
                ? opacity->tau_sobolev[(size_t)line * n_shells + shell] : 0.0;
            double beta = radeq_beta_esc(tau_l);
            double S_lag = opacity->line_source_S
                ? opacity->line_source_S[(size_t)line * n_shells + shell] : 0.0;
            /* For the deterministic producer, J_bar_l was computed with the SAME
             * line source convention (line_source_S>0 else B(nu,Te) fallback); use
             * the matching S_lag so bJext = J_bar_l - (1-beta)S_l = beta*J_ext is the
             * consistent external pump (S_lag=0 would leave bJext=full J_bar = the
             * sealed mode-1 over-pump). */
            if (det_jbar && !(S_lag > 0.0)) S_lag = planck_bnu(T_e, nu_line);
            double bJext = J_jbar - (1.0 - beta) * S_lag;   /* = β·J_ext */
            if (!(bJext > 0.0)) bJext = 0.0;                /* clamp noise/oversub */
            R_absorb = atom->line_B_lu[line] * bJext;
            R_stim   = atom->line_B_ul[line] * bJext;
            R_spont  = atom->line_A_ul[line] * beta;
            jd_mode = 2; jd_beta = beta;  /* [withParityP GATE2] m2 differenced */
        } else {
            /* mode 1 (naive, SEALED → explodes; kept only for A/B) uses the full
             * J_bar with no escape factor; the binned-J path keeps MALI β. */
            double J_line = use_jbar ? J_jbar
                                     : nlte_get_J_at_nu(nlte, shell, nu_line);
            double beta_use = use_jbar ? 1.0 : mali_beta;
            R_absorb = atom->line_B_lu[line] * J_line * beta_use;
            R_stim   = atom->line_B_ul[line] * J_line * beta_use;
            R_spont  = atom->line_A_ul[line] * beta_use;
            /* [withParityP GATE2] 1=m1 jbar naive (beta=1); 0=binned nlte_get_J */
            jd_mode = use_jbar ? 1 : 0; jd_beta = beta_use;
        }
        /* A2_06_DIAGNOSTIC_SHADOW_END */

        /* Production split.  JEQB remains the registered detailed-balance
         * falsifier; every ordinary path consumes only the checked line view. */
        {
            double Jbar_view = 0.0;
            if (jeqb) {
                Jbar_view = planck_bnu(T_e, nu_line);
                jd_mode = 5;  /* diagnostic code: JEQB */
            } else {
                (void)nlte_bb_jbar_canonical(nlte, shell, line, &Jbar_view);
                jd_mode = 6;  /* diagnostic code: A2-06 view */
            }
            J_line = Jbar_view;
            jd_beta = 1.0;
            R_absorb = atom->line_B_lu[line] * Jbar_view;
            R_stim   = atom->line_B_ul[line] * Jbar_view;
            R_spont  = atom->line_A_ul[line];
        }

        /* [withParityP GATE2] Observe the per-line jbar the matrix just consumed,
         * read straight from opacity->jbar_line / jbar_count at the consumption
         * point (no recomputation). Thread-safe append (shells run OMP-parallel). */
        if (nlte_jbar_dump_want(nlte->nlte_Z[map], nlte->nlte_ion[map])) {
            double jl = opacity->jbar_line
                        ? opacity->jbar_line[(size_t)line * n_shells + shell] : -1.0;
            long   jc = opacity->jbar_count
                        ? (long)opacity->jbar_count[(size_t)line * n_shells + shell] : -1L;
            double lamA = C_SPEED_OF_LIGHT / nu_line * 1.0e8;
            double Bpl  = planck_bnu(T_e, nu_line);
            #ifdef _OPENMP
            #pragma omp critical (lumina_jbar_dump)
            #endif
            {
                fprintf(g_jbdump_fp,
                        "%d,%d,%d,%d,%d,%.4f,%.6e,%ld,%.6e,%d,%.6e\n",
                        g_jbdump_iter, shell, nlte->nlte_Z[map], nlte->nlte_ion[map],
                        line, lamA, jl, jc, jd_beta, jd_mode, Bpl);
            }
        }

        double dE = fabs(atom->level_energy_eV[upper_global] -
                         atom->level_energy_eV[lower_global]) * EV_TO_ERG;
        int g_lo = atom->level_g[lower_global];
        int g_up = atom->level_g[upper_global];
        double f_lu = atom->line_f_lu[line];

        double C_up = 0.0;
        double C_down = 0.0;
        if (T_e > 0.0 && dE > 0.0 && g_lo > 0 && g_up > 0) {
            double exp_factor = exp(-dE / (K_BOLTZMANN * T_e));
            if (omcm_on) {
                /* [OMEGA-CMFGEN] per-TRANSITION dispatch (CMFGEN's own rule),
                 * replacing the per-ION suppression below: a covered ion's pairs
                 * that are NOT in col_data must still get the vR / OMEGA_SET
                 * fallback, not zero. Tabulated pairs are left to the dedicated
                 * col_data pass (tier 1 -> C=0 here) so nothing double-counts. */
                int tier = 3;
                double ups = omega_cmfgen_line(atom, line, T_e, &tier);
                if (tier != 1)
                    artis_col_rates(T_e, n_e, dE, (double)g_lo, (double)g_up,
                                    f_lu, ups, 0, &C_up, &C_down);
            } else
            if (artis_parity_enabled()) {
                /* A2 (severe, default under master gate) + A5: ARTIS van
                 * Regemorter with the Bethe (H_ionpot/dE)^2 factor + the
                 * energy-dependent Gaunt Γ=max(0.2,0.276 e^u(-γ-ln u)) for
                 * permitted E1 lines; the g-scaled Axelrod floor (eff. Upsilon
                 * = 0.01·g_lo·g_up) for forbidden. Dispatched on the f_lu
                 * permitted/forbidden proxy (ARTIS reads a per-transition
                 * forbidden flag Lumina's line table does not carry; f_lu<=1e-10
                 * is the E1-vs-M1/E2 signal). Lines of ions that have REAL
                 * close-coupling Omega are zeroed below and driven by the
                 * dedicated col_data pass instead (no double-count). */
                int forb = (f_lu <= 1e-10);
                artis_col_rates(T_e, n_e, dE, (double)g_lo, (double)g_up, f_lu,
                                forb ? -2.0 : -1.0, forb, &C_up, &C_down);
            } else if (nlte_coll_fix_enabled()) {
                /* Dispatch by collision STRENGTH, not raw f_lu (codex 019ed80e).
                 * Upsilon = max(van Regemorter w/ the Bethe (Ry/dE)^2 scaling the
                 * legacy branch dropped, Omega=1 forbidden floor). Symmetric
                 * collision-strength form keeps detailed balance exactly:
                 *   C_up/C_down = (g_up/g_lo) * exp(-dE/kTe). */
                double dE_eV = dE / EV_TO_ERG;
                double ry_over_dE = 13.6057 / (dE_eV > 0.0 ? dE_eV : 1e30);
                double gbar = 0.2;                 /* eff. Gaunt factor, ions */
                double ups_vr = 14.5 * f_lu * ry_over_dE * ry_over_dE * gbar;
                double ups = (ups_vr > AXELROD_OMEGA) ? ups_vr : AXELROD_OMEGA;
                C_up   = n_e * 8.629e-6 / (g_lo * sqrt(T_e)) * ups * exp_factor;
                C_down = n_e * 8.629e-6 / (g_up * sqrt(T_e)) * ups;
            } else {
                if (f_lu > 1e-10) {
                    C_up = VAN_REGEMORTER_COEFF * n_e * f_lu *
                           exp_factor / (g_lo * sqrt(T_e)) * 0.2;
                } else {
                    C_up = 8.63e-6 * n_e * AXELROD_OMEGA *
                           exp_factor / (g_lo * sqrt(T_e));
                }
                C_down = C_up * ((double)g_lo / (double)g_up) *
                         exp(dE / (K_BOLTZMANN * T_e));
            }
        }
        /* Collisional floor: the thermalizing rate is the de-excitation C_down;
         * floor it at ε·A_ul (minimum thermalization parameter ε=C_down/(C_down+A))
         * and re-derive C_up from the floored C_down by the inverse detailed-balance
         * ratio. Keeps the matrix non-singular at low n_e and drives n_u/n_l →
         * Boltzmann@T_e (DB-preserved; the equilibrium is independent of ε so the
         * floor only sets the approach, not a new fixed point). Floor on C_down (not
         * C_up) avoids the exp(+dE/kTe) blow-up for UV lines. */
        if (!ew_capture && !artis_parity_enabled() &&
            coll_floor > 0.0 && T_e > 0.0 && g_lo > 0 && g_up > 0) {
            double cd_min = coll_floor * atom->line_A_ul[line];
            if (C_down < cd_min) {
                C_down = cd_min;
                C_up = cd_min * ((double)g_up / (double)g_lo) *
                       exp(-dE / (K_BOLTZMANN * T_e));
            }
        }

        /* FEIII_COLDATA: suppress the per-line (van Regemorter / Axelrod floor)
         * collision for Fe III lines — the real Zhang col_data collisions are
         * added in the dedicated pass below, and would double count otherwise.
         * Radiative rates (R_absorb/R_stim/R_spont) are untouched. */
        if (!omcm_on && feiii_coldata_on &&
            atom->line_atomic_number[line] == 26 && atom->line_ion_number[line] == 2) {
            C_up = 0.0; C_down = 0.0;
        }
        /* A3: under parity, suppress the per-line proxy for EVERY ion that has a
         * real close-coupling Omega table (Fe II, Co III, Ni III, ...); those
         * rates are added by the generic col_data pass below. */
        if (!omcm_on && artis_parity_enabled() &&
            ion_has_realcoldata(atom, atom->line_atomic_number[line],
                                atom->line_ion_number[line])) {
            C_up = 0.0; C_down = 0.0;
        }

        double total_up   = R_absorb + C_up;
        double total_down = R_stim + R_spont + C_down;

#ifdef LUMINA_FROZEN_ORACLE
        /* Representative = largest actual upward population flow n_l R_lu for
         * each requested ion.  Rates and J are the locals consumed below. */
        if (g_oracle.fp) {
            int os = oracle_ion_slot(atom->line_atomic_number[line],
                                     atom->line_ion_number[line]);
            if (os >= 0) {
                int nl_lo_o = nlte->global_to_nlte_level[lower_global];
                double np_lo = (nl_lo_o >= 0)
                    ? nlte->nlte_level_populations[(size_t)nl_lo_o*n_shells+shell]
                    : 0.0;
                double score = (np_lo > 0.0) ? fabs(np_lo * R_absorb) : -1.0;
                if (score > g_oracle.top[os].score) {
                    OracleTopLine *ot = &g_oracle.top[os];
                    ot->seen = 1; ot->score = score; ot->line = line;
                    ot->lo_level = atom->level_num[lower_global];
                    ot->up_level = atom->level_num[upper_global];
                    ot->lambda_A = C_SPEED_OF_LIGHT / nu_line * 1.0e8;
                    ot->j_line = J_line;
                    ot->jbar_raw = opacity->jbar_line
                        ? opacity->jbar_line[(size_t)line*n_shells+shell] : -1.0;
                    ot->jbar_count = opacity->jbar_count
                        ? opacity->jbar_count[(size_t)line*n_shells+shell] : -1;
                    ot->beta = jd_beta;
                    ot->r_up = R_absorb; ot->r_stim = R_stim;
                    ot->r_spont = R_spont; ot->c_lu = C_up; ot->c_ul = C_down;
                }
            }
        }
#endif

        /* Within-SL Boltzmann fractions: only the fraction of the lower SL
         * residing in this FL absorbs, only the fraction of the upper SL in
         * this FL emits. Intra-SL transitions (i_lo==i_up) cancel — correct,
         * since within an SL the populations are pinned to Boltzmann. */
        double f_lo = FRAC_OF(fl_lo_g);
        double f_up = FRAC_OF(fl_up_g);
        ACM(i_up, i_lo) += total_up   * f_lo;
        ACM(i_lo, i_up) += total_down * f_up;
        ACM(i_lo, i_lo) -= total_up   * f_lo;
        ACM(i_up, i_up) -= total_down * f_up;
        nlte_ew_capture_transition(NLTE_EW_RAD_BB, i_up, i_lo,
                                   R_absorb * f_lo);
        nlte_ew_capture_transition(NLTE_EW_RAD_BB, i_lo, i_up,
                                   (R_stim + R_spont) * f_up);
        nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_up, i_lo,
                                   C_up * f_lo);
        nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_lo, i_up,
                                   C_down * f_up);
        DDC_RAD(i_lo, i_up, R_absorb, R_stim + R_spont);   /* [DIAG-T3] */
        DDC_COLL(i_lo, i_up, C_up, C_down);                /* [DIAG-T3] */

        if (bb_connected && (total_up + total_down) > 1e-30) {
            bb_connected[i_lo] = 1;
            bb_connected[i_up] = 1;
        }

        /* Rate-budget diagnostic (LUMINA_NLTE_BUDGET_DUMP): per bb-line of a
         * target (Z,stage,shell), dump the rate coefficients + prev-iter pops so
         * the actual feed of n_upper (recomb-cascade vs radiative pump vs coll)
         * can be reconstructed offline — decides the O III/S III super-thermal
         * mechanism. */
        if (budget_hit) {
            #ifdef _OPENMP
            #pragma omp critical(nlte_budget_dump)
            #endif
            {
                FILE *bf = fopen("nlte_budget_lines.csv", budget_lines_hdr ? "a" : "w");
                if (bf) {
                    if (!budget_lines_hdr) {
                        fprintf(bf, "pairZ,pair_lo,line_Z,line_ion,shell,line,lambda_A,"
                                    "i_lo,i_up,n_lo,n_up,R_absorb,R_stim,R_spont,"
                                    "C_up,C_down,J_line\n");
                        budget_lines_hdr = 1;
                    }
                    int nl_lo = nlte->global_to_nlte_level[lower_global];
                    int nl_up = nlte->global_to_nlte_level[upper_global];
                    double n_lo_p = nlte->nlte_level_populations[(size_t)nl_lo*n_shells+shell];
                    double n_up_p = nlte->nlte_level_populations[(size_t)nl_up*n_shells+shell];
                    fprintf(bf, "%d,%d,%d,%d,%d,%d,%.2f,%d,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                            nlte->nlte_Z[ion_idx_lo], nlte->nlte_ion[ion_idx_lo],
                            atom->line_atomic_number[line], atom->line_ion_number[line],
                            shell, line, C_SPEED_OF_LIGHT/nu_line*1e8, i_lo, i_up,
                            n_lo_p, n_up_p, R_absorb, R_stim, R_spont, C_up, C_down, J_line);
                    fclose(bf);
                }
            }
        }
    }

    /* ---- COLD-CASE-P fix: collisional drain for drainless-metastable levels ----
     * (LUMINA_NLTE_METASTABLE_COLL, default OFF => this block is skipped entirely, so
     * the assembled matrix is byte-identical to the baseline.)
     *
     * The per-line collision loop above only builds channels for level pairs that
     * EXIST as a radiative line. A level flagged metastable=1 with zero downward lines
     * (nlte->drainless_metastable[], precomputed in nlte_init) therefore gets NO
     * de-excitation channel at all — neither radiative (no line) nor collisional (the
     * assembly is line-driven). It fills by cascade and its b_k pins at the
     * g_stage4_bk_cap ceiling, driving the photospheric IGE over-ionization through its
     * EUV photoionization edge (Fe III level 17, 3.73 eV: 371 upward pumps, 0 downward
     * lines -> holds ~51% of Fe III at s8).
     *
     * Restore the missing drain with a forbidden-collision floor. Two topologies,
     * selected by LUMINA_NLTE_METACOLL_MODE (nlte_metacoll_mode()):
     *
     *   MODE 1 (default, byte-identical to kpr9): couple each drainless-metastable
     *     (as "upper") to its ion's GROUND (as "lower") with the Axelrod Omega=1
     *     floor. GROUND is the single partner; one Omega=1 channel per level.
     *
     *   MODE 2 (this session): couple each drainless-metastable to ALL lower levels
     *     of the same ion (E_l < E_m), each with the CMFGEN f=0 forbidden floor
     *     Omega = nlte_metacoll_omega() (default 0.1). This matches the topology of
     *     CMFGEN's FeIII_COL_DATA (Zhang 1996), which drains every metastable to
     *     every lower level, and its documented forbidden floor ("Value for OMEGA if
     *     f=0: 0.1"). MODE 1's ground-only Omega=1 UNDER-drains (Fe III lvl 17 stays
     *     b_k~40 at the photosphere), forcing the b_k cap; MODE 2 completes the drain
     *     as a PARAMETER-FREE APPROXIMATION to CMFGEN's true per-transition Zhang96
     *     Omega (imported later as the fidelity endpoint).
     *
     * Both modes use the exact symmetric collision-strength form, so detailed balance
     * is exact per channel: C_up/C_down = (g_meta/g_lower) * exp(-dE/kTe). Off-diagonal
     * placement is identical to the line-based collisions above (meta="upper",
     * lower="lower"; total_up=C_up, total_down=C_down). */
    if (!ew_capture && (nlte_metacoll_enabled() || parity_on) && T_e > 0.0 && n_e > 0.0) {
      /* A1: under the master gate, the metastable full-connect (mode 2: every
       * drainless metastable -> all lower levels) is the parity default, with
       * the ARTIS g-scaled Axelrod forbidden floor (A5) per channel and real
       * close-coupling Omega superseding it where loaded (skipped below). */
      int metacoll_mode = parity_on ? 2 : nlte_metacoll_mode();
      if (metacoll_mode == 1) {
        /* ===== MODE 1 (default): ground-only Omega=1, byte-identical to kpr9 ===== */
        for (int ion = ion_idx_lo; ion <= ion_idx_hi; ion++) {
            /* Fe III drained by the real Zhang table below -> skip its floor. */
            if (feiii_coldata_on && nlte->nlte_Z[ion] == 26 && nlte->nlte_ion[ion] == 2)
                continue;
            int l0 = nlte->nlte_ion_level_offset[ion];
            int l1 = nlte->nlte_ion_level_offset[ion + 1];
            /* locate this ion's ground level (level_num == 0) among its NLTE levels */
            int ground_global = -1;
            for (int nl = l0; nl < l1; nl++) {
                int gl = nlte->nlte_to_global_level[nl];
                if (atom->level_num[gl] == 0) { ground_global = gl; break; }
            }
            if (ground_global < 0) continue;
            int fl_g = nlte->global_to_nlte_level[ground_global];
            int i_g  = SOLVE_OF(fl_g);
            if (i_g < 0 || i_g >= N) continue;
            int    g_ground = atom->level_g[ground_global];
            double E_ground = atom->level_energy_eV[ground_global];
            double f_g      = FRAC_OF(fl_g);
            if (g_ground <= 0) continue;
            for (int nl = l0; nl < l1; nl++) {
                int m = nlte->nlte_to_global_level[nl];
                if (!nlte->drainless_metastable[m]) continue;
                if (m == ground_global) continue;          /* ground can't drain to itself */
                int    g_meta = atom->level_g[m];
                double dE = (atom->level_energy_eV[m] - E_ground) * EV_TO_ERG;
                if (g_meta <= 0 || !(dE > 0.0)) continue;   /* need a real up-transition */
                int fl_m = nlte->global_to_nlte_level[m];
                int i_m  = SOLVE_OF(fl_m);
                if (i_m < 0 || i_m >= N || i_m == i_g) continue;
                /* Axelrod Omega=1 forbidden-collision floor (numerical constant
                 * 8.629e-6, same as the coll_fix / Axelrod branches). */
                double C_down = n_e * 8.629e-6 / (g_meta   * sqrt(T_e)) * AXELROD_OMEGA;
                double C_up   = n_e * 8.629e-6 / (g_ground * sqrt(T_e)) * AXELROD_OMEGA
                                * exp(-dE / (K_BOLTZMANN * T_e));
                double f_m = FRAC_OF(fl_m);
                ACM(i_m, i_g) += C_up   * f_g;   /* rate into meta   from ground */
                ACM(i_g, i_m) += C_down * f_m;   /* rate into ground from meta   */
                ACM(i_g, i_g) -= C_up   * f_g;
                ACM(i_m, i_m) -= C_down * f_m;
                nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_m, i_g, C_up*f_g);
                nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_g, i_m, C_down*f_m);
                DDC_COLL(i_g, i_m, C_up, C_down);   /* [DIAG-T3] metacoll m1 */
            }
        }
      } else {
        /* ===== MODE 2: all-lower-levels, CMFGEN forbidden floor Omega (default 0.1) ==== */
        double omega = nlte_metacoll_omega();
        for (int ion = ion_idx_lo; ion <= ion_idx_hi; ion++) {
            /* Fe III drained by the real Zhang table below -> skip its floor. */
            if (feiii_coldata_on && nlte->nlte_Z[ion] == 26 && nlte->nlte_ion[ion] == 2)
                continue;
            /* A3: under parity, any ion with a real close-coupling Omega table is
             * drained by the generic col_data pass below -> skip its floor too
             * (avoid double-count; ion_has_realcoldata also covers Fe III). */
            if (parity_on &&
                ion_has_realcoldata(atom, nlte->nlte_Z[ion], nlte->nlte_ion[ion]))
                continue;
            /* Same per-ion level range the assembler uses everywhere to enumerate an
             * ion's levels: [nlte_ion_level_offset[ion], nlte_ion_level_offset[ion+1]). */
            int l0 = nlte->nlte_ion_level_offset[ion];
            int l1 = nlte->nlte_ion_level_offset[ion + 1];
            for (int nl_m = l0; nl_m < l1; nl_m++) {
                int m = nlte->nlte_to_global_level[nl_m];
                if (!nlte->drainless_metastable[m]) continue;
                int    g_meta = atom->level_g[m];
                if (g_meta <= 0) continue;
                double E_m = atom->level_energy_eV[m];
                int fl_m = nlte->global_to_nlte_level[m];
                int i_m  = SOLVE_OF(fl_m);
                if (i_m < 0 || i_m >= N) continue;
                double f_m = FRAC_OF(fl_m);
                /* Couple m (upper) to EVERY lower level l (E_l < E_m) of the SAME ion,
                 * each with the CMFGEN f=0 forbidden floor Omega. APPROXIMATION: the
                 * true per-transition value is Zhang96 (imported later); here the
                 * topology (all-lower, not ground-only) and the 0.1 floor come straight
                 * from FeIII_COL_DATA. Detailed balance is exact PER CHANNEL. */
                for (int nl_l = l0; nl_l < l1; nl_l++) {
                    if (nl_l == nl_m) continue;
                    int l = nlte->nlte_to_global_level[nl_l];
                    int    g_l = atom->level_g[l];
                    if (g_l <= 0) continue;
                    double dE = (E_m - atom->level_energy_eV[l]) * EV_TO_ERG;
                    if (!(dE > 0.0)) continue;   /* l must be strictly lower than m */
                    int fl_l = nlte->global_to_nlte_level[l];
                    int i_l  = SOLVE_OF(fl_l);
                    if (i_l < 0 || i_l >= N || i_l == i_m) continue;
                    double f_l = FRAC_OF(fl_l);
                    double C_down = n_e * 8.629e-6 / (g_meta * sqrt(T_e)) * omega;
                    double C_up   = n_e * 8.629e-6 / (g_l    * sqrt(T_e)) * omega
                                    * exp(-dE / (K_BOLTZMANN * T_e));
                    ACM(i_m, i_l) += C_up   * f_l;   /* rate into meta  from lower */
                    ACM(i_l, i_m) += C_down * f_m;   /* rate into lower from meta  */
                    ACM(i_l, i_l) -= C_up   * f_l;
                    ACM(i_m, i_m) -= C_down * f_m;
                    nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_m, i_l, C_up*f_l);
                    nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_l, i_m, C_down*f_m);
                    DDC_COLL(i_l, i_m, C_up, C_down);   /* [DIAG-T3] metacoll m2 */
                }
            }
        }
      }
    }

    /* ---- FEIII_COLDATA: real Fe III collisional bound-bound rates (Zhang 1996) ----
     * (LUMINA_FEIII_COLDATA, default OFF => this block is skipped, byte-identical.)
     *
     * Replaces Lumina's per-line collision proxy for Fe III with CMFGEN's imported
     * FeIII_COL_DATA (Zhang H.L., A&A Sup. 119, 523). The per-line loop derived the
     * collision strength from the radiative oscillator strength via van Regemorter
     * (C ~ f_lu), but the over-populated Fe III levels (25, 17, 18, 28, 31, 32) drain
     * only through FORBIDDEN lines (f_lu ~ 1e-8), so the proxy gives ~0 collisional
     * drain and their b_k pile to 5-110. CMFGEN uses real close-coupling Omega ~ 1-9
     * to EVERY lower level (level 25 -> level 17 Omega=8.76, -> ground 1.61; level 17
     * -> ground 1.36). Here we add those exact rates. The per-line Fe III collision
     * and the METACOLL Fe III floor were suppressed above, so Zhang is the sole Fe III
     * collision source (CMFGEN parity: collisions from col_data, radiation from
     * osc_data). Rate form is CMFGEN's exact
     *   C(i,k) = 8.63e-8 Omega exp(-U0) / g_i / sqrt(T_4)
     * which is algebraically identical to Lumina's 8.629e-6/sqrt(T_e) convention.
     * Detailed balance is exact per channel: C_up/C_down = (g_hi/g_lo) exp(-dE/kTe). */
    if (feiii_coldata_on && T_e > 0.0 && n_e > 0.0) {
        for (int ion = ion_idx_lo; ion <= ion_idx_hi; ion++) {
            if (nlte->nlte_Z[ion] != 26 || nlte->nlte_ion[ion] != 2) continue;
            int l0 = nlte->nlte_ion_level_offset[ion];
            int l1 = nlte->nlte_ion_level_offset[ion + 1];
            int nlev_ion = l1 - l0;   /* FeIII level count; level_num is 0..nlev_ion-1 */
            if (nlev_ion <= 0) continue;
            /* Build level_num -> (solve idx, within-SL frac, g, E) maps for this ion.
             * Thread-safe local heap (assembler runs under omp parallel-for/shell). */
            int    *ln_i = (int *)   malloc((size_t)nlev_ion * sizeof(int));
            double *ln_f = (double *)malloc((size_t)nlev_ion * sizeof(double));
            int    *ln_g = (int *)   malloc((size_t)nlev_ion * sizeof(int));
            double *ln_E = (double *)malloc((size_t)nlev_ion * sizeof(double));
            if (!ln_i || !ln_f || !ln_g || !ln_E) {
                free(ln_i); free(ln_f); free(ln_g); free(ln_E); continue;
            }
            for (int k = 0; k < nlev_ion; k++) ln_i[k] = -1;
            int any = 0;
            for (int nl = l0; nl < l1; nl++) {
                int gl = nlte->nlte_to_global_level[nl];
                int ln = atom->level_num[gl];
                if (ln < 0 || ln >= nlev_ion) continue;
                int fl = nlte->global_to_nlte_level[gl];
                int i  = SOLVE_OF(fl);
                if (i < 0 || i >= N) continue;   /* only when FeIII is the pair's lower ion */
                ln_i[ln] = i;
                ln_f[ln] = FRAC_OF(fl);
                ln_g[ln] = atom->level_g[gl];
                ln_E[ln] = atom->level_energy_eV[gl];
                any = 1;
            }
            if (!any) { free(ln_i); free(ln_f); free(ln_g); free(ln_E); continue; }

            /* Interpolate Omega(T_e) linearly in T (clamped to the tabulated ends). */
            int nt = atom->feiii_col_n_temp;
            const double *tg = atom->feiii_col_tgrid;
            int ti = 0;
            while (ti < nt - 2 && T_e > tg[ti + 1]) ti++;
            double frac_t = 0.0, denom = tg[ti + 1] - tg[ti];
            if (denom > 0.0) frac_t = (T_e - tg[ti]) / denom;
            if (frac_t < 0.0) frac_t = 0.0;
            if (frac_t > 1.0) frac_t = 1.0;

            double inv_sqrtTe = 1.0 / sqrt(T_e);
            double kTe = K_BOLTZMANN * T_e;
            int n_trans = atom->feiii_col_n_trans;
            for (int t = 0; t < n_trans; t++) {
                int lo_ln = atom->feiii_col_lo[t];
                int hi_ln = atom->feiii_col_hi[t];
                if (lo_ln >= nlev_ion || hi_ln >= nlev_ion) continue;
                int i_l = ln_i[lo_ln], i_h = ln_i[hi_ln];
                if (i_l < 0 || i_h < 0 || i_l == i_h) continue;
                const double *om = &atom->feiii_col_omega[(size_t)t * nt];
                double omega = om[ti] + frac_t * (om[ti + 1] - om[ti]);
                if (!(omega > 0.0)) continue;
                int g_lo = ln_g[lo_ln], g_hi = ln_g[hi_ln];
                if (g_lo <= 0 || g_hi <= 0) continue;
                double dE = (ln_E[hi_ln] - ln_E[lo_ln]) * EV_TO_ERG;
                if (!(dE > 0.0)) continue;
                double f_l = ln_f[lo_ln], f_h = ln_f[hi_ln];
                double C_up   = n_e * 8.629e-6 * inv_sqrtTe / g_lo * omega * exp(-dE / kTe);
                double C_down = n_e * 8.629e-6 * inv_sqrtTe / g_hi * omega;
                ACM(i_h, i_l) += C_up   * f_l;   /* into higher from lower */
                ACM(i_l, i_h) += C_down * f_h;   /* into lower  from higher */
                ACM(i_l, i_l) -= C_up   * f_l;
                ACM(i_h, i_h) -= C_down * f_h;
                nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_h, i_l, C_up*f_l);
                nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_l, i_h, C_down*f_h);
                DDC_COLL(i_l, i_h, C_up, C_down);   /* [DIAG-T3] Fe III Zhang */
            }
            free(ln_i); free(ln_f); free(ln_g); free(ln_E);
        }
    }

    /* ---- A3 (ARTIS parity): real close-coupling Omega for the generic imported
     * IGE coolant ions (Fe II, Co III, Ni III) beyond Fe III. Mirrors the Fe III
     * Zhang pass above, dispatching on the loaded col_ion_* table for each (Z,ion0)
     * present in this pair. The per-line vR/Axelrod proxy AND the metastable floor
     * for these ions were suppressed above (ion_has_realcoldata), so this pass is
     * their SOLE collisional source (no double-count). Table lo/hi indices are
     * level_number == CMFGEN osc energy rank == atom->level_num (loader-verified).
     * Rate form is CMFGEN's C(i,k)=8.63e-8*Omega*exp(-U0)/g_i/sqrt(T_4) (identical
     * to Lumina's 8.629e-6/sqrt(T_e) convention); detailed balance exact per
     * channel: C_up/C_down = (g_hi/g_lo)*exp(-dE/kTe). Only under the master gate. */
    if (parity_on && atom->ncol_ions > 0 && T_e > 0.0 && n_e > 0.0) {
        for (int ion = ion_idx_lo; ion <= ion_idx_hi; ion++) {
            int Zi = nlte->nlte_Z[ion], ioni = nlte->nlte_ion[ion];
            int c = -1;
            for (int cc = 0; cc < atom->ncol_ions; cc++)
                if (atom->col_ion_Z[cc] == Zi && atom->col_ion_stage[cc] == ioni) {
                    c = cc; break;
                }
            if (c < 0) continue;
            int l0 = nlte->nlte_ion_level_offset[ion];
            int l1 = nlte->nlte_ion_level_offset[ion + 1];
            int nlev_ion = l1 - l0;
            if (nlev_ion <= 0) continue;
            int    *ln_i = (int *)   malloc((size_t)nlev_ion * sizeof(int));
            double *ln_f = (double *)malloc((size_t)nlev_ion * sizeof(double));
            int    *ln_g = (int *)   malloc((size_t)nlev_ion * sizeof(int));
            double *ln_E = (double *)malloc((size_t)nlev_ion * sizeof(double));
            if (!ln_i || !ln_f || !ln_g || !ln_E) {
                free(ln_i); free(ln_f); free(ln_g); free(ln_E); continue;
            }
            for (int k = 0; k < nlev_ion; k++) ln_i[k] = -1;
            int any = 0;
            for (int nl = l0; nl < l1; nl++) {
                int gl = nlte->nlte_to_global_level[nl];
                int ln = atom->level_num[gl];
                if (ln < 0 || ln >= nlev_ion) continue;
                int fl = nlte->global_to_nlte_level[gl];
                int i  = SOLVE_OF(fl);
                if (i < 0 || i >= N) continue;   /* only when this ion is the pair's lower ion */
                ln_i[ln] = i;
                ln_f[ln] = FRAC_OF(fl);
                ln_g[ln] = atom->level_g[gl];
                ln_E[ln] = atom->level_energy_eV[gl];
                any = 1;
            }
            if (!any) { free(ln_i); free(ln_f); free(ln_g); free(ln_E); continue; }

            int nt = atom->col_ion_n_temp[c];
            const double *tg = atom->col_ion_tgrid[c];
            int ti = 0;
            while (ti < nt - 2 && T_e > tg[ti + 1]) ti++;
            double frac_t = 0.0, denom = tg[ti + 1] - tg[ti];
            if (denom > 0.0) frac_t = (T_e - tg[ti]) / denom;
            if (frac_t < 0.0) frac_t = 0.0;
            if (frac_t > 1.0) frac_t = 1.0;

            double inv_sqrtTe = 1.0 / sqrt(T_e);
            double kTe = K_BOLTZMANN * T_e;
            int n_trans = atom->col_ion_n_trans[c];
            const int    *tlo = atom->col_ion_lo[c];
            const int    *thi = atom->col_ion_hi[c];
            const double *tom = atom->col_ion_omega[c];
            for (int t = 0; t < n_trans; t++) {
                int lo_ln = tlo[t], hi_ln = thi[t];
                if (lo_ln >= nlev_ion || hi_ln >= nlev_ion) continue;
                int i_l = ln_i[lo_ln], i_h = ln_i[hi_ln];
                if (i_l < 0 || i_h < 0 || i_l == i_h) continue;
                const double *om = &tom[(size_t)t * nt];
                double omega = om[ti] + frac_t * (om[ti + 1] - om[ti]);
                if (!(omega > 0.0)) continue;
                int g_lo = ln_g[lo_ln], g_hi = ln_g[hi_ln];
                if (g_lo <= 0 || g_hi <= 0) continue;
                double dE = (ln_E[hi_ln] - ln_E[lo_ln]) * EV_TO_ERG;
                if (!(dE > 0.0)) continue;
                double f_l = ln_f[lo_ln], f_h = ln_f[hi_ln];
                double C_up   = n_e * 8.629e-6 * inv_sqrtTe / g_lo * omega * exp(-dE / kTe);
                double C_down = n_e * 8.629e-6 * inv_sqrtTe / g_hi * omega;
                ACM(i_h, i_l) += C_up   * f_l;   /* into higher from lower */
                ACM(i_l, i_h) += C_down * f_h;   /* into lower  from higher */
                ACM(i_l, i_l) -= C_up   * f_l;
                ACM(i_h, i_h) -= C_down * f_h;
                nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_h, i_l, C_up*f_l);
                nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_l, i_h, C_down*f_h);
                DDC_COLL(i_l, i_h, C_up, C_down);   /* [DIAG-T3] A3 col_data */
            }
            free(ln_i); free(ln_f); free(ln_g); free(ln_E);
        }
    }

    /* ---- Photoionization / Recombination ---- */
    int Z_elem = nlte->nlte_Z[ion_idx_lo];
    double chi_eV = find_ioniz_energy(atom, Z_elem, nlte->nlte_ion[ion_idx_lo]);
    double chi_erg = chi_eV * EV_TO_ERG;
    double nu_edge = chi_erg / H_PLANCK;

    int n_lo_levels = nlte->nlte_ion_level_offset[ion_idx_lo + 1] -
                      nlte->nlte_ion_level_offset[ion_idx_lo];
    /* ground_hi is the matrix row/col of the upper-ion ground: it is the SL
     * count of the lower ion (== n_lo_levels in identity mode). The upper ion
     * is not collapsed here, so its FL == SL.
     * [ARTIS-PARITY B4 residual] ARTIS photoionises into the SPECIFIC phixs
     * target level and recombines from it (level-resolved bf cascade). Lumina
     * routes every level's photoion -> ground_hi and recomb <- ground_hi. The
     * per-level phixs TARGET level is NOT stored in Lumina's atomic data
     * (cmfgen_sigma_bf holds only per-LOWER-level sigma_bf curves, no upper
     * target index), so under parity we keep ground routing and REPORT B4
     * partial rather than fabricate a target. Adding a cmfgen phixs-target
     * channel is the remaining B4 work. */
    int ground_hi = n_lo_super;
    int hi_ground_global_nlte = nlte->nlte_ion_level_offset[ion_idx_hi];

    /* Diagnostic accumulators for LUMINA_NLTE_RATE_DUMP */
    double sum_R_bf = 0.0, sum_R_rec = 0.0;
    double sum_R_bf_ground = 0.0, sum_R_rec_ground = 0.0;
    double n_star_ground = 0.0;
    int sum_R_bf_levels = 0;

    if (ground_hi < N && nu_edge > 0.0 && nu_edge < nlte->nu_max) {
        /* Task #138 Heavy.1: per-level CMFGEN σ_bf where available, Kramers fallback.
         * Same grid as NLTE J_ν (NLTE_N_FREQ_BINS, log-spaced) → direct index. */
        int Z_ion = nlte->nlte_Z[ion_idx_lo];
        int ion_lo_stage = nlte->nlte_ion[ion_idx_lo];
        double sigma_0 = nlte_bf_kramers_sigma0(Z_ion, ion_lo_stage);
        const int use_cmfgen = atom->cmfgen_loaded &&
                               atom->cmfgen_n_freq_bins == nlte->n_freq_bins;

        /* Task #40 (A)+(B): GPU lookup path. R_bf_table is col-major
         * [L_phot_total × n_shells]; pair_idx selects the row offset. */
        int gpu_R_bf_available =
            (lookup != NULL && lookup->R_bf_table != NULL &&
             lookup->phot_offset != NULL && pair_idx >= 0);
        int phot_base = gpu_R_bf_available ? lookup->phot_offset[pair_idx] : 0;
        int L_phot_total = gpu_R_bf_available ? lookup->L_phot_total : 0;

        /* [withParityY Y4 (b) S1] spin-selection M_core for the RECOMBINING ion
         * of this pair, hoisted out of the per-level loop (one resolution per
         * (pair, shell) call).  Recombining core = the upper ion of the pair. */
        int rec_sg = rec_spingate_enabled();
        int M_core_pair = 0;
        if (rec_sg) {
            rec_spingate_check_data(atom);
            int ip_next_pair = find_ion_pop_idx(atom, Z_ion, ion_lo_stage + 1);
            M_core_pair = spingate_resolve_core_mult(atom, ip_next_pair, Z_ion,
                                                     ion_lo_stage + 1, NULL);
        }

        for (int lev = 0; lev < n_lo_levels; lev++) {
            int global_lev = nlte->nlte_to_global_level[lev_start + lev];
            double E_lev = atom->level_energy_eV[global_lev] * EV_TO_ERG;
            double nu_thresh = (chi_erg - E_lev) / H_PLANCK;
            if (nu_thresh <= 0.0) continue;

            /* The EW assembler owns its bound-free arithmetic because a route's
             * upper identity changes epsilon_trans, the inverse statistical
             * weight and therefore all four forward/inverse rates.  The legacy
             * pair path below intentionally retains its historical single-ground
             * producer and is not used as a target-independent EW shortcut. */
            if (ew_capture) {
                int sl_ew = SOLVE_OF(lev_start + lev);
                double f_ew = FRAC_OF(lev_start + lev);
                nlte_ew_capture_bf_target_rates(global_lev, sl_ew, f_ew,
                                                 T_e, n_e);
                continue;
            }

            int level_has_cmfgen = use_cmfgen && atom->cmfgen_has_sigma[global_lev];
            const double *sigma_row = level_has_cmfgen ?
                &atom->cmfgen_sigma_bf[(size_t)global_lev *
                                       (size_t)atom->cmfgen_n_freq_bins] : NULL;

            double R_bf = 0.0;
            double sig_edge = 0.0;   /* A4: threshold sigma_bf (first non-zero bin) */
            /* Milne recombination integral I_rec = ∫(4πσ/hν)(2hν³/c²+J_ν)e^{-hν/kTe}dν.
             * FAITHFUL FIX (2026-06-14, codex+agent verified): the old
             * R_rec = R_bf*n_star_ratio tied recombination to the photoionization
             * integral (J only), so where the ionizing J_ν collapses (cold outer
             * shells, Wien cutoff) R_rec->0 — SPONTANEOUS radiative recombination
             * (the 2hν³/c² term, J-independent) vanished and the lower-ion levels
             * drained from the continuum (super-thermal S_l, optical 3260x;
             * O+Si-pin falsifier collapsed it 2960->77). The full Milne form keeps
             * recombination finite as J->0 and, by the Planck identity
             * (2hν³/c²+B)e^{-x}=B, reduces to ∫4πBσ/hν dν at J=B (LTE Saha fixed
             * point preserved => no-op in the hot inner region). The e^{-hν/kTe}
             * weight is T_e-dependent so the static-K GPU R_bf table cannot hold
             * it; I_rec is always integrated here, and the same loop also computes
             * R_bf in the CPU fallback path. */
            double I_rec = 0.0;
#ifdef LUMINA_FROZEN_ORACLE
            double I_rec_spont = 0.0, I_rec_stim = 0.0;
#endif
            {
                const double kTe = K_BOLTZMANN * T_e;
                const double c2  = C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT;
                /* FALSIFIER (LUMINA_NLTE_BF_JEQB=1): force the bf continuum field to
                 * B(Te) in BOTH R_bf and I_rec → full-LTE bf. With bb-JEQB this makes
                 * the whole rate net DB-clean (Saha). If cold-shell garbage vanishes,
                 * the culprit is the binned-J continuum feeding bf (not a rate bug). */
                int use_gpu_R_bf = 0, gpu_field_bypassed = 0;
                int bf_field_source = nlte_bf_field_source(
                    nlte, T_e, 0.0, 0.0, gpu_R_bf_available,
                    &use_gpu_R_bf, &gpu_field_bypassed, NULL);
#ifdef LUMINA_FROZEN_ORACLE
                if (g_oracle.fp) {
                    if (use_gpu_R_bf) g_oracle.bf_gpu_lookup_level_consumptions++;
                    if (gpu_field_bypassed) g_oracle.bf_gpu_field_bypass_levels++;
                }
#endif
                if (use_gpu_R_bf) {
                    int idx = phot_base + lev;
                    R_bf = lookup->R_bf_table[(size_t)shell * L_phot_total + idx];
                } else if (bf_field_source != 2) {
                    /* [A2-05] CPU rate for field sources 0/1 = canonical-view
                     * integral; the per-bin C2 estimator / C1 fallback mix is
                     * retired.  Blocked (non-VALID) => no field term, counted
                     * on the R6 counters.  Source 2 (JEQB falsifier) keeps its
                     * per-bin B(T_e) device below. */
                    BfRateResult br_bf;
                    if (nlte_bf_gamma_canonical(nlte, shell, sigma_row, sigma_0,
                                                nu_thresh, &br_bf) == 0 &&
                        (br_bf.state == BF_RATE_VALID ||
                         br_bf.state == BF_RATE_EXACT_ZERO))
                        R_bf = br_bf.gamma;
                }
                for (int bb = 0; bb < nlte->n_freq_bins; bb++) {
                    double log_nu_lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                    double nu_bin = exp(log_nu_lo + 0.5 * nlte->d_log_nu);
                    if (nu_bin < nu_thresh) continue;
                    double delta_nu = exp(log_nu_lo + nlte->d_log_nu) - exp(log_nu_lo);
                    double J_bin = 0.0;
                    (void)nlte_bf_field_source(
                        nlte, T_e, nu_bin,
                        nlte->J_nu[shell * nlte->n_freq_bins + bb],
                        0, NULL, NULL, &J_bin);
                    double sigma;
                    if (sigma_row) {
                        sigma = sigma_row[bb];
                        if (sigma <= 0.0) continue;
                    } else {
                        sigma = sigma_0 * pow(nu_thresh / nu_bin, 3.0);
                    }
                    if (sig_edge <= 0.0) sig_edge = sigma;   /* A4: sigma_bf at edge */
                    double pref = 4.0 * M_PI_VAL * sigma / (H_PLANCK * nu_bin) * delta_nu;
                    if (!use_gpu_R_bf && bf_field_source == 2) {
                        R_bf += pref * J_bin;   /* JEQB: J_bin == B_nu(T_e) */
#ifdef LUMINA_FROZEN_ORACLE
                        if (g_oracle.fp) g_oracle.bf_fallback_consumptions++;
#endif
                    }
                    if (kTe > 0.0) {
                        double x = H_PLANCK * nu_bin / kTe;
                        if (x < 700.0) {
                            double spont = 2.0 * H_PLANCK * nu_bin * nu_bin * nu_bin / c2;
                            I_rec += pref * (spont + J_bin) * exp(-x);
#ifdef LUMINA_FROZEN_ORACLE
                            I_rec_spont += pref * spont * exp(-x);
                            I_rec_stim  += pref * J_bin * exp(-x);
#endif
                        }
                    }
                }
            }

            double n_star_ratio = 1.0;
            if (T_e > 0.0 && n_e > 0.0) {
                int g_lev = atom->level_g[global_lev];
                double thermal_deBroglie = pow(H_PLANCK * H_PLANCK /
                    (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_e), 1.5);
                double chi_lev_erg = chi_erg - E_lev;
                if (chi_lev_erg > 0.0) {
                    double exp_factor = exp(chi_lev_erg / (K_BOLTZMANN * T_e));
                    int g_ion = 1;
                    if (ground_hi < N) {
                        int global_ghi = nlte->nlte_to_global_level[hi_ground_global_nlte];
                        g_ion = atom->level_g[global_ghi];
                        if (g_ion < 1) g_ion = 1;
                    }
                    n_star_ratio = n_e * thermal_deBroglie *
                        (double)g_lev / (2.0 * (double)g_ion) * exp_factor;
                    if (n_star_ratio > 1e30) n_star_ratio = 1e30;
                }
            }
            /* [withParityY Y4 (b) S1] spin-forbidden target => NO recombination
             * into this level.  I_rec is zeroed here, BEFORE R_rec is formed, so
             * every downstream consumer (R_rec, the DIAG-T3 ledger, sum_R_rec)
             * sees one consistent number.  R_bf is deliberately NOT touched:
             * photoionization of a spin-forbidden level to an EXCITED upper-ion
             * core is physical.  That asymmetry is the declared DB caveat -- see
             * the shared-helper block near recomb_alpha_per_level. */
            if (rec_sg && spingate_level_forbidden(atom, global_lev, M_core_pair)) {
                I_rec = 0.0;
#ifdef LUMINA_FROZEN_ORACLE
                I_rec_spont = 0.0;
                I_rec_stim = 0.0;
#endif
            }
            double R_rec = n_star_ratio * I_rec;

#ifdef LUMINA_FROZEN_ORACLE
            if (g_oracle.fp) {
                int os = oracle_ion_slot(Z_ion, ion_lo_stage);
                if (os >= 0) {
                    double np = nlte->nlte_level_populations[
                        (size_t)(lev_start + lev) * n_shells + shell];
                    g_oracle.gamma_num[os] += np * R_bf;
                    g_oracle.gamma_den[os] += np;
                    if (n_e > 0.0) {
                        g_oracle.alpha_total[os] += R_rec / n_e;
                        g_oracle.alpha_spont[os] += n_star_ratio * I_rec_spont / n_e;
                        g_oracle.alpha_stim[os] += n_star_ratio * I_rec_stim / n_e;
                    }
                }
            }
#endif

            /* Collapse this FL to its SL solve index. Ionization out is weighted
             * by the FL's within-SL fraction (only that fraction of the SL pop
             * sits in this FL and ionizes); recombination in is unweighted and
             * sums over the SL's FL (total recomb captured by the SL).
             * Guard now fires on R_rec>0 too: spontaneous recombination must
             * couple the level to the continuum even where R_bf==0 (J->0). */
            int sl = SOLVE_OF(lev_start + lev);
            double f_lev = FRAC_OF(lev_start + lev);
            if ((R_bf > 0.0 || R_rec > 0.0) && sl >= 0 && sl < N && ground_hi < N) {
                ACM(ground_hi, sl) += R_bf * f_lev;
                ACM(sl, sl)        -= R_bf * f_lev;
                ACM(sl, ground_hi) += R_rec;
                ACM(ground_hi, ground_hi) -= R_rec;
                if (ddc_stage >= 0 && DDC_CAP(sl)) {   /* [DIAG-T3] */
                    g_ddc.seen[ddc_stage][sl]   = 1;
                    g_ddc.levnum[ddc_stage][sl] = atom->level_num[global_lev];
                    g_ddc.E_eV[ddc_stage][sl]   = atom->level_energy_eV[global_lev];
                    g_ddc.g[ddc_stage][sl]      = atom->level_g[global_lev];
                    g_ddc.pop[ddc_stage][sl]    = nlte->nlte_level_populations[
                                                   (size_t)(lev_start + lev) * n_shells + shell];
                    g_ddc.pion[ddc_stage][sl]  += R_bf;
                    g_ddc.rec[ddc_stage][sl]   += R_rec;
                }

                sum_R_bf  += R_bf;
                sum_R_rec += R_rec;
                sum_R_bf_levels++;
                if (lev == 0) {
                    sum_R_bf_ground  = R_bf;
                    sum_R_rec_ground = R_rec;
                    n_star_ground    = n_star_ratio;
                }
            }

            /* ---- A4 (ARTIS parity): thermal collisional ionization + 3-body
             * collisional recombination (ARTIS macroatom.cc col_ionisation_
             * ratecoeff:662-682 / col_recombination_ratecoeff:630-658). Routed
             * per level -> upper-ion ground, mirroring the radiative bf channel
             * above. C_ion is FIELD-INDEPENDENT (fires even where R_bf==0, e.g.
             * cold shells below the ionizing cutoff). C_rec is the EXACT
             * detailed-balance inverse: C_rec = C_ion * n_star_ratio, where
             * n_star_ratio is the SAME LTE Saha population ratio (n_lower_star
             * over n_upper_star) the radiative Milne pair uses. Algebra:
             * C_ion*n_lower_star == C_rec*n_upper_star at LTE by construction, and
             * C_ion*n_star_ratio reduces exactly to ARTIS SAHACONST 3-body form
             * n_e^2 * SAHACONST * (g_lo/g_up) * 1.55e13 * g * sigma * k_B/(T_e*chi).
             * u = h*nu_thresh/(k*T_e) = chi_level/(k*T_e); g = ARTIS coll-ioniz
             * Gaunt(ionstage) (0.1/0.2/0.3, macroatom.cc:309). sigma_bf(edge) = the
             * first non-zero CMFGEN cross-section bin above threshold (== Kramers
             * sigma_0 fallback). Only under the master gate => byte-identical off. */
            if (nlte_bf_collisional_enabled() && T_e > 0.0 && n_e > 0.0 &&
                sig_edge > 0.0 &&
                sl >= 0 && sl < N && ground_hi < N) {
                double u_ion = H_PLANCK * nu_thresh / (K_BOLTZMANN * T_e);
                if (u_ion > 0.0 && u_ion < 700.0) {
                    double g_col = (ion_lo_stage <= 0) ? 0.1
                                 : (ion_lo_stage == 1) ? 0.2 : 0.3;
                    double C_ion = n_e * 1.55e13 / sqrt(T_e) * g_col * sig_edge *
                                   exp(-u_ion) / u_ion;
                    double C_rec = C_ion * n_star_ratio;   /* exact DB inverse */
                    if (C_ion > 0.0 && isfinite(C_ion) && isfinite(C_rec)) {
                        ACM(ground_hi, sl) += C_ion * f_lev;   /* coll. ioniz out  */
                        ACM(sl, sl)        -= C_ion * f_lev;
                        ACM(sl, ground_hi) += C_rec;           /* 3-body recomb in */
                        ACM(ground_hi, ground_hi) -= C_rec;
                        if (ddc_stage >= 0 && DDC_CAP(sl)) {   /* [DIAG-T3] */
                            g_ddc.cion[ddc_stage][sl] += C_ion;
                            g_ddc.c3b[ddc_stage][sl]  += C_rec;
                        }
                    }
                }
            }

            if (budget_hit) {
                double n_p = nlte->nlte_level_populations[
                    (size_t)(lev_start + lev) * n_shells + shell];
                double E_eV = atom->level_energy_eV[global_lev];
                #ifdef _OPENMP
                #pragma omp critical(nlte_budget_dump)
                #endif
                {
                    FILE *rf = fopen("nlte_budget_rec.csv", budget_rec_hdr ? "a" : "w");
                    if (rf) {
                        if (!budget_rec_hdr) {
                            fprintf(rf, "Z,stage,shell,lev,sl,E_eV,g,n_pop,R_bf,R_rec,"
                                        "I_rec,n_star_ratio\n");
                            budget_rec_hdr = 1;
                        }
                        fprintf(rf, "%d,%d,%d,%d,%d,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                                budget_Z, budget_stage, shell, lev, sl, E_eV,
                                atom->level_g[global_lev], n_p, R_bf, R_rec, I_rec,
                                n_star_ratio);
                        fclose(rf);
                    }
                }
            }
        }
    }

    /* ---- TOP-ION CONTINUUM DRAIN (fixed Saha-IV reservoir) ----
     * The hi ion (III) is the top NLTE stage; with no (III,IV) pair its excited
     * levels get no photoion/recomb edge -> rank-deficient block -> the LU solve
     * returns a UNIFORM (flat) population -> super-thermal S_l (triple-verified
     * 2026-06-16, docs/TOPSTAGE_IV_CONTINUUM_NODE_DESIGN.md). Restore detailed
     * balance by adding, to each hi-ion level, a photoionization sink (III->IV)
     * and a Milne recombination SOURCE from the Saha IV reservoir (already in
     * n_e via the upstream ionization solve; codex+phys-agent: R_rec is
     * independent of the IV abs population, only the III SHAPE matters, conserv.
     * sets the total). Source = n_lte_hl * I_rec_hl with n_lte_hl = n_IV *
     * n_star_ratio computed in LOG space (no 1e30 cap; the n_IV*exp(chi/kTe)
     * product is the level's LTE pop = well-scaled). Gauge-free: the IV ground
     * g cancels in the within-III ratios. Env-gated; optional single-Z filter. */
    {
        static int tic_mode = -1, tic_zonly = 0;
        if (tic_mode < 0) {
            const char *e = getenv("LUMINA_TOPSTAGE_IV");
            tic_mode = (e && atoi(e)) ? 1 : 0;
            const char *zf = getenv("LUMINA_TOPSTAGE_IV_ZONLY");
            tic_zonly = zf ? atoi(zf) : 0;   /* 0 = all top ions */
        }
        int Zh = nlte->nlte_Z[ion_idx_hi];
        int ion_hi_stage = nlte->nlte_ion[ion_idx_hi];
        /* Only the TOP NLTE ion needs this: if (Zh, ion_hi+1) is itself an NLTE
         * ion, the hi ion already has a real continuum pair (e.g. O II in the
         * (O I,O II) pair has O III above it) -> skip to avoid double-coupling. */
        int hi_is_top = 1;
        for (int q = 0; q < nlte->n_nlte_ions; q++)
            if (nlte->nlte_Z[q] == Zh && nlte->nlte_ion[q] == ion_hi_stage + 1) {
                hi_is_top = 0; break;
            }
        if (!ew_capture && tic_mode && hi_is_top && (tic_zonly == 0 || tic_zonly == Zh) &&
            T_e > 0.0 && n_e > 0.0) {
            nlte_ew_note_topstage_IV_call();
            double chi_hi_eV = find_ioniz_energy(atom, Zh, ion_hi_stage); /* III->IV */
            int ip_iv = find_ion_pop_idx(atom, Zh, ion_hi_stage + 1);     /* IV stage */
            int ip_hi = find_ion_pop_idx(atom, Zh, ion_hi_stage);         /* III stage */
            double n_iv = (ip_iv >= 0) ?
                atom->ion_number_density[(size_t)ip_iv * n_shells + shell] : 0.0;
            double n_hi_total = (ip_hi >= 0) ?
                atom->ion_number_density[(size_t)ip_hi * n_shells + shell] : 0.0;
            if (getenv("LUMINA_TOPSTAGE_IV_DIAG") && shell == 6) {
                static int tic_diag = 0;
                if (tic_diag < 40) {
                    fprintf(stderr, "[TOPIV] Z=%d ion_hi=%d top=%d chi=%.2f "
                            "n_hi=%.3e n_iv=%.3e ip_hi=%d ip_iv=%d FIRE=%d\n",
                            Zh, ion_hi_stage, hi_is_top, chi_hi_eV, n_hi_total,
                            n_iv, ip_hi, ip_iv,
                            (chi_hi_eV>0.0 && chi_hi_eV<1e9 && n_hi_total>0.0));
                    tic_diag++;
                }
            }
            if (chi_hi_eV > 0.0 && chi_hi_eV < 1e9 && n_hi_total > 0.0) {
                double chi_hi_erg = chi_hi_eV * EV_TO_ERG;
                double nu_edge_hi  = chi_hi_erg / H_PLANCK;
                const double kTe = K_BOLTZMANN * T_e;
                const double c2  = C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT;
                double deBroglie = pow(H_PLANCK * H_PLANCK /
                    (2.0 * M_PI_VAL * M_ELECTRON * kTe), 1.5);
                /* IV ground g (common factor -> cancels in III ratios); default 1 */
                int g_iv = 1;
                for (int l = 0; l < atom->n_levels; l++)
                    if (atom->level_Z[l] == Zh &&
                        atom->level_ion[l] == ion_hi_stage + 1 &&
                        atom->level_num[l] == 0) { g_iv = atom->level_g[l]; break; }
                if (g_iv < 1) g_iv = 1;
                double sigma0_hi = get_bf_sigma0(Zh, ion_hi_stage);
                if (sigma0_hi <= 0.0) {
                    int Ze = Zh - ion_hi_stage; if (Ze < 1) Ze = 1;
                    sigma0_hi = 7.91e-18 / ((double)Ze * (double)Ze);
                }
                const int use_cmf = atom->cmfgen_loaded &&
                                    atom->cmfgen_n_freq_bins == nlte->n_freq_bins;
                int h0 = nlte->nlte_ion_level_offset[ion_idx_hi];
                int h1 = nlte->nlte_ion_level_offset[ion_idx_hi + 1];
                /* n_IV reservoir: use the real Saha IV density where meaningful;
                 * floor to the Saha self-consistent value n_hi_total/n_star_ground
                 * where it underflowed to ~0 in cold outer shells (codex: never let
                 * the continuum normalization be 0 -> III block stays rank-deficient
                 * -> flat). n_lte is bounded by n_hi_total below for conditioning. */
                int gl0 = nlte->nlte_to_global_level[h0];
                int g0  = atom->level_g[gl0]; if (g0 < 1) g0 = 1;
                double ln_nstar_g = log(n_e) + log(deBroglie)
                                  + log((double)g0 / (2.0 * (double)g_iv))
                                  + chi_hi_erg / kTe;
                double n_iv_saha = exp(log(n_hi_total) - ln_nstar_g);
                double n_iv_eff  = (n_iv > n_iv_saha) ? n_iv : n_iv_saha;
                double ln_pref = log(n_iv_eff) + log(n_e) + log(deBroglie)
                                 - log(2.0 * (double)g_iv);
                /* [withParityY Y4 (c) S2] spin-selection M_core for the stage-IV
                 * recombining core, hoisted out of the level loop. */
                int rec_sg_iv = rec_spingate_enabled();
                int M_core_iv = 0;
                if (rec_sg_iv) {
                    rec_spingate_check_data(atom);
                    int ip_iv_pop = find_ion_pop_idx(atom, Zh, ion_hi_stage + 1);
                    M_core_iv = spingate_resolve_core_mult(atom, ip_iv_pop, Zh,
                                                           ion_hi_stage + 1, NULL);
                }
                int bf_field_source_iv = nlte_bf_field_source(
                    nlte, T_e, 0.0, 0.0, 0, NULL, NULL, NULL);
                for (int hl = h0; hl < h1; hl++) {
                    int gl = nlte->nlte_to_global_level[hl];
                    double E_hl = atom->level_energy_eV[gl] * EV_TO_ERG;
                    double chi_lev = chi_hi_erg - E_hl;
                    if (chi_lev <= 0.0) continue;
                    /* Each excited level photoionizes across ITS OWN threshold
                     * nu_edge_lev = chi_lev/h (< nu_edge_hi for E_hl>0), NOT the
                     * ground edge. Using nu_edge_hi for all levels skipped the
                     * [nu_edge_lev, nu_edge_hi] band where the high levels' bf
                     * cross-section peaks -> high excited levels were never
                     * drained -> stayed super-thermal (the FUV/UV emitters that
                     * keep the formal spectrum too blue). */
                    double nu_edge_lev = chi_lev / H_PLANCK;
                    const double *srow = (use_cmf && atom->cmfgen_has_sigma[gl]) ?
                        &atom->cmfgen_sigma_bf[(size_t)gl * atom->cmfgen_n_freq_bins]
                        : NULL;
                    double R_bf_hl = 0.0, I_rec_hl = 0.0;
                    /* [A2-05 ADDENDUM site 7] stage-IV excited-level rate:
                     * sources 0/1 = canonical-view integral (blocked => no
                     * field term, counted); source 2 keeps the per-bin JEQB
                     * device below, which also still owns the Milne term. */
                    if (bf_field_source_iv != 2) {
                        BfRateResult br_hl;
                        if (nlte_bf_gamma_canonical(nlte, shell, srow, sigma0_hi,
                                                    nu_edge_lev, &br_hl) == 0 &&
                            (br_hl.state == BF_RATE_VALID ||
                             br_hl.state == BF_RATE_EXACT_ZERO))
                            R_bf_hl = br_hl.gamma;
                    }
                    for (int bb = 0; bb < nlte->n_freq_bins; bb++) {
                        double log_lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
                        double nu_bin = exp(log_lo + 0.5 * nlte->d_log_nu);
                        if (nu_bin < nu_edge_lev) continue;
                        double dnu = exp(log_lo + nlte->d_log_nu) - exp(log_lo);
                        double sigma = srow ? srow[bb]
                            : sigma0_hi * pow(nu_edge_lev / nu_bin, 3.0);
                        if (sigma <= 0.0) continue;
                        double J_bin = 0.0;
                        (void)nlte_bf_field_source(
                            nlte, T_e, nu_bin,
                            nlte->J_nu[(size_t)shell * nlte->n_freq_bins + bb],
                            0, NULL, NULL, &J_bin);
                        double pref = 4.0 * M_PI_VAL * sigma / (H_PLANCK * nu_bin) * dnu;
                        if (bf_field_source_iv == 2)
                            R_bf_hl += pref * J_bin;  /* J_bin == B_nu(T_e) */
                        double x = H_PLANCK * nu_bin / kTe;
                        if (x < 700.0) {
                            double spont = 2.0 * H_PLANCK * nu_bin * nu_bin * nu_bin / c2;
                            I_rec_hl += pref * (spont + J_bin) * exp(-x);
                        }
                    }
                    if (R_bf_hl <= 0.0 && I_rec_hl <= 0.0) continue;
                    /* n_lte_hl = n_IV_eff * n_star_ratio (log space, no cap);
                     * a single level cannot exceed the ion total (Saha-consistent
                     * bound -> well-conditioned at all shells). */
                    double ln_nlte = ln_pref + log((double)atom->level_g[gl]) +
                                     chi_lev / kTe;
                    double n_lte_hl = (ln_nlte < 700.0) ? exp(ln_nlte) : exp(700.0);
                    if (n_lte_hl > n_hi_total) n_lte_hl = n_hi_total;
                    int sl = SOLVE_OF(hl);
                    if (sl < 0 || sl >= N) continue;
                    double f = FRAC_OF(hl);
                    ACM(sl, sl) -= R_bf_hl * f;   /* photoionization sink III->IV */
                    /* [withParityY Y4 (c) S2] same predicate as S1/S3: a
                     * spin-forbidden level gets no Milne recombination source
                     * from the IV ground core.  The photoionization sink on the
                     * line above is deliberately left in place (declared DB
                     * caveat, shared-helper block). */
                    if (rec_sg_iv &&
                        spingate_level_forbidden(atom, gl, M_core_iv))
                        I_rec_hl = 0.0;
                    b[sl]       -= n_lte_hl * I_rec_hl; /* Milne recomb source from IV */
                }
            }
        }
    }

    /* Diagnostic dump: per-(Z, ion_pair, shell) rate balance summary.
     * Writes to nlte_rate_balance.csv (append mode, header on first call). */
    {
        const char *env = getenv("LUMINA_NLTE_RATE_DUMP");
        if (env && env[0] == '1') {
            static int header_written = 0;
            #ifdef _OPENMP
            #pragma omp critical(nlte_rate_dump)
            #endif
            {
                FILE *fp = fopen("nlte_rate_balance.csv", header_written ? "a" : "w");
                if (fp) {
                    if (!header_written) {
                        fprintf(fp, "Z,ion_lo,shell,N_levels,T_e,T_rad,W,n_e,"
                                    "n_ion_lo_neb,n_ion_hi_neb,"
                                    "sum_R_bf,sum_R_rec,sum_R_bf_lev_count,"
                                    "R_bf_ground,R_rec_ground,n_star_ground,"
                                    "ratio_RrecRbf,ratio_NLTE_predicted_lo_to_hi\n");
                        header_written = 1;
                    }
                    int Z_d = Z_elem;
                    int ion_lo_d = nlte->nlte_ion[ion_idx_lo];
                    int ip_lo = find_ion_pop_idx(atom, Z_d, ion_lo_d);
                    int ip_hi = find_ion_pop_idx(atom, Z_d, nlte->nlte_ion[ion_idx_hi]);
                    double n_neb_lo = (ip_lo >= 0) ?
                        atom->ion_number_density[ip_lo * n_shells + shell] : 0.0;
                    double n_neb_hi = (ip_hi >= 0) ?
                        atom->ion_number_density[ip_hi * n_shells + shell] : 0.0;
                    double ratio_RR = (sum_R_bf > 0.0) ? sum_R_rec / sum_R_bf : 0.0;
                    fprintf(fp, "%d,%d,%d,%d,%.2f,%.2f,%.6e,%.6e,"
                                "%.6e,%.6e,"
                                "%.6e,%.6e,%d,"
                                "%.6e,%.6e,%.6e,"
                                "%.6e,%.6e\n",
                            Z_d, ion_lo_d, shell, N, T_e, T_rad,
                            1.0, n_e,
                            n_neb_lo, n_neb_hi,
                            sum_R_bf, sum_R_rec, sum_R_bf_levels,
                            sum_R_bf_ground, sum_R_rec_ground, n_star_ground,
                            ratio_RR, n_star_ground);
                    fclose(fp);
                }
            }
        }
    }

    /* ---- Step 1.5: Charge Exchange rates ---- */
    int Z_pair = nlte->nlte_Z[ion_idx_lo]; /* element Z for this ion pair */
    int ion_lo_stage = nlte->nlte_ion[ion_idx_lo];   /* e.g. 1 for II */
    int ion_hi_stage = nlte->nlte_ion[ion_idx_hi];   /* e.g. 2 for III */

    /* ---- Heavy.2 / Task #139: Dielectronic recombination ----
     * α_DR(T_e) is added as a non-Milne recombination channel from
     * upper-ion ground (ground_hi) into lower-ion ground (lev=0). This
     * lifts R_rec where Milne-from-CMFGEN-σ_bf alone gives Saha-LTE-at-T_e
     * over-ionization. Cascades through the lower-ion bb network repopulate
     * excited levels self-consistently. Requires the ion_recomb=ion_hi_stage
     * entry in DR_TABLE; if missing the rate is zero (legacy behavior).
     *
     * LUMINA_DR_FLOOR_CMS env var (default 0): empirical phenomenological
     * α_DR floor in cm³/s, applied uniformly to *every* NLTE (II,III) pair.
     * Probes "would any extra recombination close the gap?" independent of
     * T-dependence. Mazzotta LS-coupling misses low-T near-threshold
     * resonances; this lets us decouple "is more recomb needed" from
     * "is the T-shape right". Use ~1e-12 to 1e-10 cm³/s as test range. */
    {
        const DRCoefficient *coef = dr_lookup(Z_pair, ion_hi_stage);
        double alpha_dr = (coef && n_e > 0.0) ? dr_alpha_eval(coef, T_e) : 0.0;

        static int floor_init = 0;
        static double alpha_dr_floor = 0.0;
        if (!floor_init) {
            const char *fenv = getenv("LUMINA_DR_FLOOR_CMS");
            if (fenv) alpha_dr_floor = atof(fenv);
            floor_init = 1;
        }
        if (!ew_capture && alpha_dr_floor > 0.0 && alpha_dr < alpha_dr_floor)
            alpha_dr = alpha_dr_floor;

        double R_dr = alpha_dr * n_e;   /* [s⁻¹] per upper-ion ion */
        if (R_dr > 0.0 && ground_hi < N) {
            ACM(0, ground_hi)         += R_dr;
            ACM(ground_hi, ground_hi) -= R_dr;
            nlte_ew_capture_transition(NLTE_EW_AUTOION_DR, 0, ground_hi, R_dr);
        }
    }

    for (int r = 0; !ew_capture && r < CE_N_REACTIONS; r++) {
        const ChargeExchangeReaction *ce = &CE_REACTIONS[r];
        double k_fwd = ce->rate_coeff * pow(T_e / 1e4, ce->alpha);
        double k_rev = k_fwd * exp(ce->delta_E_eV * EV_TO_ERG /
                                    (K_BOLTZMANN * T_e));

        /* Case 1: This pair is element A (forward: A^ion_A → A^(ion_A+1))
         * Requires: Z_pair == Z_A, ion_A == lower ion, ion_A+1 == upper ion */
        if (ce->Z_A == Z_pair && ce->ion_A == ion_lo_stage &&
            ce->ion_A + 1 == ion_hi_stage) {
            double n_partner = nlte_get_ion_density(nlte, atom,
                ce->Z_B, ce->ion_B, shell, n_shells);
            double n_partner_lower = nlte_get_ion_density(nlte, atom,
                ce->Z_B, ce->ion_B - 1, shell, n_shells);

            double R_fwd = k_fwd * n_partner;    /* [s⁻¹] per A^ion_A ion */
            double R_rev = k_rev * n_partner_lower; /* [s⁻¹] per A^(ion_A+1) ion */

            /* Forward: all A^ion_A levels → A^(ion_A+1) ground.
             * Level-independent rate: applied once per super-level. */
            for (int sl = 0; sl < n_lo_super; sl++) {
                ACM(ground_hi, sl) += R_fwd;
                ACM(sl, sl)        -= R_fwd;
            }
            /* Reverse: A^(ion_A+1) ground → A^ion_A ground */
            ACM(0, ground_hi)          += R_rev;
            ACM(ground_hi, ground_hi)  -= R_rev;
        }

        /* Case 2: This pair is element B (forward: B^ion_B → B^(ion_B-1))
         * Requires: Z_pair == Z_B, ion_B == upper ion, ion_B-1 == lower ion */
        if (ce->Z_B == Z_pair && ce->ion_B == ion_hi_stage &&
            ce->ion_B - 1 == ion_lo_stage) {
            double n_partner = nlte_get_ion_density(nlte, atom,
                ce->Z_A, ce->ion_A, shell, n_shells);
            double n_partner_upper = nlte_get_ion_density(nlte, atom,
                ce->Z_A, ce->ion_A + 1, shell, n_shells);

            double R_fwd = k_fwd * n_partner;        /* B^ion_B → B^(ion_B-1) */
            double R_rev = k_rev * n_partner_upper;   /* B^(ion_B-1) → B^ion_B */

            /* Forward: B^(ion_B) ground → B^(ion_B-1) ground
             * (ion_B is the upper ion in this pair, ground_hi is its first level) */
            ACM(0, ground_hi)          += R_fwd;
            ACM(ground_hi, ground_hi)  -= R_fwd;
            /* Reverse: all B^(ion_B-1) levels → B^ion_B ground.
             * Level-independent rate: applied once per super-level. */
            for (int sl = 0; sl < n_lo_super; sl++) {
                ACM(ground_hi, sl) += R_rev;
                ACM(sl, sl)        -= R_rev;
            }
        }
    }

    /* ---- Non-thermal gamma-ray ionization ---- */
    if (gamma_dep != NULL && gamma_dep->nonthermal_ioniz_rate[shell] > 0.0) {
        double R_nt_total = gamma_dep->nonthermal_ioniz_rate[shell]; /* ionizations/s/cm³ */

        /* Compute total atom number density in shell from all elements */
        double n_total_atoms = 0.0;
        for (int e = 0; e < atom->n_elements; e++) {
            double X_e = atom->abundances[e * n_shells + shell];
            double A_e = atom->element_mass_amu[e];
            n_total_atoms += X_e * plasma->rho[shell] / (A_e * AMU);
        }

        /* Per-particle ionization rate (distributed equally per atom) */
        if (n_total_atoms > 0.0 && ground_hi < N) {
            double R_nt_per_particle = R_nt_total / n_total_atoms; /* [s⁻¹] */

            if (ew_capture) {
                /* ARTIS traverses every lower level and its continuum target.
                 * Projection fractions make the rate apply once per source SL. */
                nlte_ew_capture_nt_routes(R_nt_per_particle);
            } else {
                /* Historical pair-wise baseline: ground-to-ground. */
                ACM(ground_hi, 0) += R_nt_per_particle;
                ACM(0, 0)         -= R_nt_per_particle;
            }
        }
    }

    /* Element-wide Stage-2A owns closure and solve.  Return immediately after
     * the seven physical process planes have been captured: this guarantees
     * zero TOPSTAGE_IV, time-dependent, pin, floor, anchor and repair calls. */
    if (ew_capture) {
        free(bb_connected);
        return;
    }

    /* ---- Time-dependent ionization: backward-Euler dn_i/dt term (option A) ----
     *
     * Steady-state assumes dn_i/dt = 0, i.e. infinite time to equilibrate, so at
     * the cold outer ejecta (T~2500K) it recombines to near-neutral and n_e
     * collapses. CMFGEN keeps the ions frozen because recombination there is far
     * slower than the expansion age (tau_rec/t_exp ~ 300). We restore that by
     * solving the implicit (backward-Euler) step
     *     (n^new - n^old)/Dt = A n^new   =>   (A - I/Dt) n^new = -n^old/Dt,
     * with Dt = time_explosion (single step from t=0) and n^old the fully-ionized
     * initial condition (all pair mass in the hi-ion ground). The 1/Dt diagonal
     * (= 1.19e-5 s^-1 at 0.976d) sits exactly at the tau_rec = t_exp crossover:
     * rows whose net rate >> 1/Dt relax to steady state, rows whose rate << 1/Dt
     * stay at n^old. The conservation row below still pins Sum n to the mass-based
     * nebular total, so only the ion PARTITION becomes time-dependent. The
     * conservation/anchor rows are overwritten afterwards, dropping the time term
     * from exactly those rows (correct: they are closures, not rate equations). */
    if (nlte_timedep_active() && time_explosion > 0.0) {
        double inv_dt = 1.0 / time_explosion;
        double n_pair_total = nlte_pair_total_density(nlte, atom, plasma,
                                  nlte->nlte_Z[ion_idx_lo],
                                  ion_idx_lo, ion_idx_hi, shell);
        for (int i = 0; i < N; i++) {
            double n_old_i = (i == ground_hi) ? n_pair_total : 0.0;
            ACM(i, i) -= inv_dt;
            b[i]      -= n_old_i * inv_dt;
        }
    }

    /* ---- Conservation equation(s) ----
     *
     * Default: single combined row (sum over ALL levels of (lo,hi) = nebular total
     * ion-pair density). This lets the bf+rec+CE+DR matrix entries determine
     * the ion partitioning self-consistently. Works only if those rates are
     * physically reasonable.
     *
     * LUMINA_NLTE_ION_LOCK=1: Mihalas-Lucy hybrid. Replace TWO rows
     * (last row of lower-ion block, last row of upper-ion block) with
     * per-ion conservation. n_lo_total and n_hi_total are pinned to nebular
     * W·ζ-Saha values. NLTE then only redistributes excited levels within
     * each ion. Avoids the Milne-T_e-vs-T_rad over-ionization trap that DR
     * (any magnitude) cannot fix. Standard SN-code approach (TARDIS exact).
     */
    int Z_nl = nlte->nlte_Z[ion_idx_lo];
    int ion_lock_mode = nlte_ion_lock_active(nlte->current_iter);

    /* A2-07: a conservation row may move to a connected row, but an isolated
     * physical row is never replaced by a Boltzmann population anchor.  A truly
     * deficient matrix therefore fails in the checked solve below. */
    int alt_row_hi = N - 1;
    int alt_row_lo = n_lo_super - 1;
    if (bb_connected) {
        /* Pick the last bb-connected upper-ion row as conservation destination.
         * Falls back to N-1 if every row is bb-isolated (shouldn't happen). */
        for (int k = N - 1; k >= n_lo_super; k--) {
            if (bb_connected[k]) { alt_row_hi = k; break; }
        }
        /* Same for lower-ion block (ion-lock mode only). */
        for (int k = n_lo_super - 1; k >= 0; k--) {
            if (bb_connected[k]) { alt_row_lo = k; break; }
        }

        if (ground_hi < N) {
            int ref_global = nlte->super_anchor_global[super_start + ground_hi];
            double E_ref = atom->level_energy_eV[ref_global] * EV_TO_ERG;
            int g_ref = atom->level_g[ref_global];
            if (g_ref < 1) g_ref = 1;
            /* TOPSTAGE_THERMALIZE: force Boltzmann@T_e on the TOP NLTE stage's
             * bb-CONNECTED EXCITED levels (the over-populated super-thermal carriers
             * O/C/S/Al III). The carriers are excited-EXCITED lines that are THIN in
             * the thermal limit, so a tau-gate misses them — the right target is the
             * over-populated LEVEL. n_k/n_ground = (g_k/g_ref)exp(−ΔE/kT_e), ref =
             * own-ion ground (ground_hi), T_e, NO dilution W. Default anchors all
             * top-stage excited levels (FORCE_LTE for the top stage, proven gold-like);
             * LUMINA_TOPSTAGE_DEPARTURE>0 restricts to levels whose lagged pop exceeds
             * Boltzmann by that factor (preserves near-thermal levels). */
            if (hi_is_topstage) {
                static int tsth_diag = 0; int tsth_nanch = 0;
                double n_ground = 0.0;
                if (tsth_dep > 0.0) {
                    int gnl = nlte->global_to_nlte_level[
                        nlte->super_anchor_global[super_start + ground_hi]];
                    if (gnl >= 0) n_ground =
                        nlte->nlte_level_populations[(size_t)gnl * n_shells + shell];
                }
                for (int k = n_lo_super; k < N; k++) {
                    if (!bb_connected[k]) continue;        /* isolated handled above */
                    if (k == alt_row_hi) continue;         /* conservation row */
                    if (k == ground_hi) continue;          /* anchor reference */
                    int gk_global = nlte->super_anchor_global[super_start + k];
                    double E_k = atom->level_energy_eV[gk_global] * EV_TO_ERG;
                    int g_k = atom->level_g[gk_global];
                    if (g_k < 1) g_k = 1;
                    double dE = E_k - E_ref;
                    if (dE < 0.0) dE = 0.0;
                    double br = (double)g_k / (double)g_ref *
                                exp(-dE / (K_BOLTZMANN * T_e));   /* T_e, no W */
                    if (!isfinite(br)) br = 0.0;
                    if (tsth_dep > 0.0) {   /* departure gate: skip near-thermal levels */
                        int knl = nlte->global_to_nlte_level[gk_global];
                        double n_k = (knl >= 0) ?
                            nlte->nlte_level_populations[(size_t)knl * n_shells + shell] : 0.0;
                        double n_boltz = br * n_ground;
                        if (!(n_boltz > 0.0) || n_k <= tsth_dep * n_boltz) continue;
                    }
                    tsth_nanch++;
                    for (int j = 0; j < N; j++) ACM(k, j) = 0.0;
                    ACM(k, k) = 1.0;
                    ACM(k, ground_hi) = -br;
                    b[k] = 0.0;
                }
                if (tsth_nanch > 0 && tsth_diag < 20) {
                    fprintf(stderr, "[TSTH] top-stage Z=%d ion=%d shell=%d: anchored "
                            "%d/%d excited levels -> Boltzmann@T_e=%.0fK (dep=%.0f)\n",
                            nlte->nlte_Z[ion_idx_hi], nlte->nlte_ion[ion_idx_hi], shell,
                            tsth_nanch, N - n_lo_super, T_e, tsth_dep);
                    tsth_diag++;
                }
            }
        }
    }

    /* ===== DIAGNOSTIC (LUMINA_NLTE_NSTAR_DUMP=1): write the TRUE cross-ion
     * Saha-Boltzmann LTE reference n*_i for the target pair/shell, so the
     * orthodox conditioning recipe (departure-coeff transform M=D^-1 A D with
     * D=diag(n*), zero-out of decoupled levels) can be validated OFFLINE on the
     * raw matrix (dumped by cuda.cu MATDUMP) before wiring it into the solve.
     * n* within an ion = g_i exp(-E_i/kTe); across ions = x Saha factor
     * S = (2/n_e) (2pi m_e k Te/h^2)^{3/2} exp(-chi_lo/kTe). Off by default. */
    if (getenv("LUMINA_NLTE_NSTAR_DUMP") &&
        Z_nl == (getenv("LUMINA_POP_Z") ? atoi(getenv("LUMINA_POP_Z")) : 8) &&
        nlte->nlte_ion[ion_idx_lo] == (getenv("LUMINA_POP_ION") ? atoi(getenv("LUMINA_POP_ION")) : 1) &&
        shell == (getenv("LUMINA_POP_SHELL") ? atoi(getenv("LUMINA_POP_SHELL")) : 24)) {
        double kTe = K_BOLTZMANN * T_e;
        double chi_lo_eV = find_ioniz_energy(atom, Z_nl, nlte->nlte_ion[ion_idx_lo]);
        double lam3 = pow(H_PLANCK * H_PLANCK /
                          (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_e), 1.5);
        double Saha = (n_e > 0.0)
            ? (2.0 / n_e) * (1.0 / lam3) * exp(-chi_lo_eV * EV_TO_ERG / kTe)
            : 0.0;
        int   *ionf  = (int   *)malloc((size_t)N * sizeof(int));
        double *Ev   = (double *)malloc((size_t)N * sizeof(double));
        double *gv   = (double *)malloc((size_t)N * sizeof(double));
        double *nst  = (double *)malloc((size_t)N * sizeof(double));
        for (int i = 0; i < N; i++) {
            int gl = nlte->nlte_to_global_level[lev_start + i];
            double E_eV = atom->level_energy_eV[gl];
            double g    = (double)atom->level_g[gl];
            int is_hi   = (i >= n_lo_super) ? 1 : 0;
            double boltz = g * exp(-E_eV * EV_TO_ERG / kTe);
            ionf[i] = is_hi;
            Ev[i]   = E_eV;
            gv[i]   = g;
            nst[i]  = is_hi ? boltz * Saha : boltz;
        }
        const char *npath = getenv("LUMINA_NLTE_NSTAR_PATH");
        if (!npath) npath = "bk2_nstar.bin";
        FILE *nf = fopen(npath, "wb");
        if (nf) {
            int hdr[5] = { N, n_lo_super, Z_nl, nlte->nlte_ion[ion_idx_lo], shell };
            double scal[3] = { T_e, n_e, chi_lo_eV };
            fwrite(hdr, sizeof(int), 5, nf);
            fwrite(scal, sizeof(double), 3, nf);
            fwrite(ionf, sizeof(int), (size_t)N, nf);
            fwrite(Ev,  sizeof(double), (size_t)N, nf);
            fwrite(gv,  sizeof(double), (size_t)N, nf);
            fwrite(nst, sizeof(double), (size_t)N, nf);
            fclose(nf);
            fprintf(stderr, "[NSTAR_DUMP] wrote %s N=%d n_lo=%d Z=%d ion=%d s=%d "
                    "Te=%.0fK n_e=%.3e Saha=%.3e\n", npath, N, n_lo_super, Z_nl,
                    nlte->nlte_ion[ion_idx_lo], shell, T_e, n_e, Saha);
        }
        free(ionf); free(Ev); free(gv); free(nst);
    }

    /* ===== b_k-SPACE PARTIAL-LTE conditioning fix (LUMINA_NLTE_BK_PARTIAL=1) =====
     * ROOT (verified): at cold Te the raw-population rate matrix spans ~77 orders
     * (Boltzmann e^{-E/kTe}, E to 35 eV, kTe~0.2 eV) >> double precision -> cond~1e15 ->
     * getrf garbage -> Boltzmann@T_rad fallback -> super-thermal S_l. FIX: solve in
     * departure-coefficient space n=n* b (n*=LTE Boltzmann). Similarity transform
     * M_ij = A_ij n*_j/n*_i scales the Boltzmann factor OUT (rates are DB-correct so
     * M=O(rates)); pin negligible levels (n_star/n_star_ground < thr) to b_k=1.
     * Offline-verified on the real O II matrix: cond 1.6e15 -> 4e3, all b_k=1 at J=B.
     * The conservation rows below are written in b_k form (sum n*_j b_j = n_total).
     * cuda.cu back-converts n_i = b_i * n*_i after the solve. Gated/off => byte-identical. */
    static int bk_partial = -1;
    if (bk_partial < 0) { const char *e = getenv("LUMINA_NLTE_BK_PARTIAL");
        bk_partial = (e && atoi(e)) ? 1 : 0; }
    double *bk_nstar = NULL;
    if (bk_partial && N > 0) {
        bk_nstar = (double*)malloc((size_t)N * sizeof(double));
        /* REFERENCE = PREVIOUS iterate populations (NOT per-ion LTE Boltzmann). The prior
         * pops are Saha+Boltzmann consistent across BOTH ions, so n_old_j/n_old_i carries
         * the correct CROSS-ION LTE ratio for free. The per-ion-ground LTE form had NO Saha
         * factor -> cross-ion entries ~1e64 -> cond 1e81 -> garbage (verified). By detailed
         * balance M_ij = A_ij n_old_j/n_old_i = the reverse rate = O(rates); fluorescent
         * departures come out in the b_k solution (matrix stays conditioned). Identity-mode
         * indexing (solve idx i <-> fine nlte level lev_start+i). */
        double pmax = 1e-300;
        for (int i = 0; i < N; i++) {
            double p = nlte->nlte_level_populations[(size_t)(lev_start + i) * n_shells + shell];
            if (p > pmax) pmax = p;
        }
        double pfloor = pmax * 1e-30;
        for (int i = 0; i < N; i++) {
            double p = nlte->nlte_level_populations[(size_t)(lev_start + i) * n_shells + shell];
            bk_nstar[i] = (p > pfloor) ? p : pfloor;
        }
        /* similarity transform of the assembled RATE rows (conservation set fresh below) */
        for (int i = 0; i < N; i++) {
            double inv_ni = 1.0 / bk_nstar[i];
            for (int j = 0; j < N; j++) ACM(i, j) *= bk_nstar[j] * inv_ni;
            b[i] *= inv_ni;
        }
        /* PIN only DEAD levels (b_k-row off-diagonal rate < thr * median) to b_k=1 (LTE).
         * NOT an LTE-population/energy criterion: that wrongly freezes radiatively-pumped
         * UV levels (verified: 333/337 energy-pinned levels HAVE significant rates ->
         * pinning them kills the fluorescence, making pops J-insensitive). Rate-based
         * pinning freezes only the genuinely unconnected (singular) rows, preserving the
         * pumped manifold. Offline: cond 9e14 -> 3.6e11 (6 pinned, 334 active). */
        double pin_thr = 1e-4;
        { const char *e = getenv("LUMINA_NLTE_BK_PIN_THR"); if (e) pin_thr = atof(e); }
        double *rmax = (double*)malloc((size_t)N * sizeof(double));
        double *rsort = (double*)malloc((size_t)N * sizeof(double));
        for (int i = 0; i < N; i++) {
            double mx = 0.0;
            for (int j = 0; j < N; j++) if (j != i) { double a = fabs(ACM(i, j)); if (a > mx) mx = a; }
            rmax[i] = mx; rsort[i] = mx;
        }
        qsort(rsort, N, sizeof(double), cmf_dcmp_pl);
        double rmed = rsort[N / 2];
        if (rmed <= 0.0) rmed = 1.0;
        for (int i = 0; i < N; i++) {
            if (rmax[i] < pin_thr * rmed) {
                for (int j = 0; j < N; j++) ACM(i, j) = 0.0;
                ACM(i, i) = 1.0; b[i] = 1.0;
            }
        }
        free(rmax); free(rsort);
    }
    #define CONS_W(j) (bk_partial ? bk_nstar[(j)] : 1.0)

    if (ion_lock_mode && n_lo_super > 0 && n_lo_super < N) {
        nlte_ew_note_per_ion_pin_call();
        double n_lo_total = 0.0;
        double n_hi_total = 0.0;
        int ip_lo = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_lo]);
        int ip_hi = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_hi]);
        if (ip_lo >= 0)
            n_lo_total = atom->ion_number_density[ip_lo * n_shells + shell];
        if (ip_hi >= 0)
            n_hi_total = atom->ion_number_density[ip_hi * n_shells + shell];

        int row_lo = alt_row_lo;
        int row_hi = alt_row_hi;
        for (int j = 0; j < N; j++) {
            ACM(row_lo, j) = (j < n_lo_super) ? CONS_W(j) : 0.0;
            ACM(row_hi, j) = (j >= n_lo_super) ? CONS_W(j) : 0.0;
        }
        b[row_lo] = n_lo_total;
        b[row_hi] = n_hi_total;
    } else {
        double n_total = nlte_pair_total_density(nlte, atom, plasma, Z_nl,
                                                  ion_idx_lo, ion_idx_hi, shell);
        int row = bb_connected ? alt_row_hi : (N - 1);
        for (int j = 0; j < N; j++)
            ACM(row, j) = CONS_W(j);
        b[row] = n_total;
    }
    #undef CONS_W
    if (bk_nstar) free(bk_nstar);

    if (bb_connected) free(bb_connected);

    /* [DIAG-T3] this pair completed the lo-ion stage's decomposition; rewrite the
     * whole file so it holds every Fe stage seen in the latest solve sweep. Rate
     * coefficients are per-atom [s^-1] (photoion/recomb/collion/3-body are the
     * per-level channel totals; multiply by n_pop for the volumetric rate). */
    if (ddc_stage >= 0) {
        #ifdef _OPENMP
        #pragma omp critical(nlte_diag_decomp)
        #endif
        {
            FILE *rf = fopen("lumina_rates_decomp.csv", "w");
            if (rf) {
                fprintf(rf, "shell,Z,ion,solve_level,level_num,E_eV,g,n_pop,"
                            "rad_up,rad_down,coll_up,coll_down,"
                            "photoion,recomb,collion,three_body\n");
                for (int st = 0; st < DDC_NSTAGE; st++) {
                    for (int L = 0; L < DDC_MAXLEV; L++) {
                        if (!g_ddc.seen[st][L]) continue;
                        fprintf(rf, "%d,26,%d,%d,%d,%.4f,%d,%.6e,"
                                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                                nlte_diag_decomp_shell(), st + 1, L,
                                g_ddc.levnum[st][L], g_ddc.E_eV[st][L], g_ddc.g[st][L],
                                g_ddc.pop[st][L], g_ddc.radup[st][L], g_ddc.raddn[st][L],
                                g_ddc.colup[st][L], g_ddc.coldn[st][L], g_ddc.pion[st][L],
                                g_ddc.rec[st][L], g_ddc.cion[st][L], g_ddc.c3b[st][L]);
                    }
                }
                fclose(rf);
            }
        }
    }
    #undef DDC_CAP
    #undef DDC_RAD
    #undef DDC_COLL

    #undef ACM
    #undef SOLVE_OF
    #undef FRAC_OF
}

/* CPU NLTE solver: assemble + Gauss elimination for one ion pair in one shell */
static int nlte_solve_ion_shell(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                int ion_idx_lo, int ion_idx_hi,
                                int shell, double time_explosion,
                                GammaDeposition *gamma_dep,
                                int pair_shares_slot) {
    int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];
    int super_start = nlte->nlte_ion_super_offset[ion_idx_lo];
    /* Matrix is assembled and solved on super-levels (N); full-level pops are
     * then redistributed within each SL by the within-SL Boltzmann fraction.
     * In identity mode N == N_fl and the SL->FL expansion is a no-op. */
    int N    = nlte->nlte_ion_super_offset[ion_idx_hi + 1] - super_start;
    int N_fl = nlte->nlte_ion_level_offset[ion_idx_hi + 1] - lev_start;
    if (N <= 0 || N_fl <= 0) return -1;
    int n_shells = plasma->n_shells;
    int n_lo_super = nlte->nlte_ion_super_offset[ion_idx_lo + 1] - super_start;

    double *A_cm = (double *)calloc((size_t)N * N, sizeof(double));
    double *b = (double *)calloc((size_t)N, sizeof(double));
    if (!A_cm || !b) { free(A_cm); free(b); return -1; }

    /* Dead-pair skip (mirror of the CUDA assembly skip): no atoms of this
     * element here -> ~0 pops either way; route straight to the Boltzmann
     * fallback below, skipping the costly assemble+solve. Env-gated. */
    int ret = 0;
    int has_nonfinite = 0;
    if (nlte_skip_dead_pairs()) {
        int Z_dead = nlte->nlte_Z[ion_idx_lo];
        double n_tot = nlte_pair_total_density(nlte, atom, plasma, Z_dead,
                                               ion_idx_lo, ion_idx_hi, shell);
        if (n_tot < 1e-10) has_nonfinite = 1;
    }

    if (!has_nonfinite) {
        uint64_t population_errors_before = nlte->population_error_count;
        nlte_assemble_rate_matrix(nlte, atom, plasma, opacity,
                                   ion_idx_lo, ion_idx_hi, shell, time_explosion,
                                   A_cm, b, N, gamma_dep,
                                   NULL, -1);
        if (nlte->population_error_count != population_errors_before)
            ret = -1;
        else {
            PopulationStatus rank_status = population_dense_rank_check(
                A_cm, (size_t)N, 1.0e-14);
            if (rank_status != POP_OK) {
#ifdef _OPENMP
#pragma omp critical(a2_07_population_error)
#endif
                {
                    if (nlte->population_error_count == 0)
                        nlte->population_first_error = rank_status;
                    nlte->population_error_count++;
                    population_counter_note(&nlte->population_counters,
                                            rank_status);
                }
                ret = -1;
            } else
            ret = gauss_solve(A_cm, b, N);
        }
    }

    /* Detect non-finite output: gauss_solve may succeed but produce NaN/Inf
     * when the rate matrix is ill-conditioned at high T_e. */
    if (!has_nonfinite && ret == 0) {
        for (int i = 0; i < N; i++) {
            if (!isfinite(b[i])) { has_nonfinite = 1; break; }
        }
        /* Boltzmann-ceiling sanity gate: a near-singular matrix can yield a
         * FINITE but inverted solution (excited levels 1e9-1e11x ground).
         * Reject when any level exceeds its ion ground pop by more than
         * (g_i/g_ground)*margin, routing into the Boltzmann fallback below. */
        double inv_ceil = nlte_inv_ceiling();
        if (!has_nonfinite && inv_ceil > 0.0) {
            int n_lo = n_lo_super;   /* SL-space ground/excited split */
            int g0_lo = atom->level_g[nlte->super_anchor_global[super_start]];
            int g0_hi = (n_lo < N) ?
                atom->level_g[nlte->super_anchor_global[super_start + n_lo]] : 1;
            double b0_lo = b[0];
            double b0_hi = (n_lo < N) ? b[n_lo] : 1.0;
            for (int i = 0; i < N; i++) {
                double bg = (i < n_lo) ? b0_lo : b0_hi;
                int gg = (i < n_lo) ? g0_lo : g0_hi;
                if (bg <= 0.0) {
                    /* Empty/negative ground with a populated excited level is
                     * itself an inversion — the garbage solve drained ground. */
                    if (b[i] > 0.0) { has_nonfinite = 1; break; }
                    continue;
                }
                int gi = atom->level_g[nlte->super_anchor_global[super_start + i]];
                double ceil_ratio = ((double)gi / (double)(gg > 0 ? gg : 1)) * inv_ceil;
                if (b[i] / bg > ceil_ratio) { has_nonfinite = 1; break; }
            }
        }
    }

    if (ret != 0 || has_nonfinite) {
        free(A_cm); free(b);
        return -1;
    }
    {
        /* Clamp negatives + rescale to enforce conservation.
         * Default: combined Σ x_i = n_pair_total.
         * LUMINA_NLTE_ION_LOCK=1: per-ion rescale to (n_lo_total, n_hi_total)
         * to preserve the ion-lock from the matrix. */
        int Z_nl = nlte->nlte_Z[ion_idx_lo];
        /* pair_shares_slot: overlapping O pairs ({28,29} & {29,30}) must
         * pin each ion to its own nebular total. Combined renorm scales O III
         * by the O II-dominated pair sum → O III 1e13x / O I 500x blowup. */
        int lock = nlte_ion_lock_active(nlte->current_iter) ||
                   nlte_per_ion_rescale_active() || pair_shares_slot;

        int n_lo_levels = nlte->nlte_ion_level_offset[ion_idx_lo + 1] -
                          nlte->nlte_ion_level_offset[ion_idx_lo];

        /* A2-07: negative/non-finite solutions are terminal. Legacy LTE repair,
         * flat floors and b-space caps are diagnostic shadow only and cannot
         * alter a production population candidate. */
        for (int i = 0; i < N; i++) {
            if (!isfinite(b[i]) || b[i] < 0.0) {
                free(A_cm); free(b);
                return -1;
            }
        }

        /* Redistribute super-level solution to full levels:
         *   n_FL = x_SL[SL(FL)] * f_FL,   f_FL = within-SL Boltzmann fraction.
         * Identity mode: SL(FL)==FL nlte idx and f_FL==1, so xfl == b. */
        double *xfl = (double *)malloc((size_t)N_fl * sizeof(double));
        if (!xfl) { free(A_cm); free(b); return -1; }
        for (int i = 0; i < N_fl; i++) {
            int sl = nlte->fl_to_super[lev_start + i] - super_start;
            double f = nlte->within_sl_frac[(size_t)(lev_start + i) * n_shells + shell];
            xfl[i] = b[sl] * f;
        }

        if (lock && n_lo_levels > 0 && n_lo_levels < N_fl) {
            double n_lo_total = 0.0, n_hi_total = 0.0;
            int ip_lo = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_lo]);
            int ip_hi = find_ion_pop_idx(atom, Z_nl, nlte->nlte_ion[ion_idx_hi]);
            if (ip_lo >= 0)
                n_lo_total = atom->ion_number_density[ip_lo * n_shells + shell];
            if (ip_hi >= 0)
                n_hi_total = atom->ion_number_density[ip_hi * n_shells + shell];

            double sum_lo = 0.0, sum_hi = 0.0;
            for (int i = 0; i < n_lo_levels; i++) sum_lo += xfl[i];
            for (int i = n_lo_levels; i < N_fl; i++) sum_hi += xfl[i];
            double scale_lo = (sum_lo > 0.0 && n_lo_total > 0.0)
                            ? n_lo_total / sum_lo : (n_lo_total == 0.0 ? 0.0 : 1.0);
            double scale_hi = (sum_hi > 0.0 && n_hi_total > 0.0)
                            ? n_hi_total / sum_hi : (n_hi_total == 0.0 ? 0.0 : 1.0);
            for (int i = 0; i < n_lo_levels; i++) {
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] =
                    xfl[i] * scale_lo;
            }
            for (int i = n_lo_levels; i < N_fl; i++) {
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] =
                    xfl[i] * scale_hi;
            }
        } else {
            double n_total = nlte_pair_total_density(nlte, atom, plasma, Z_nl,
                                                      ion_idx_lo, ion_idx_hi, shell);
            double sum = 0.0;
            for (int i = 0; i < N_fl; i++) sum += xfl[i];
            double scale = (sum > 0.0 && n_total > 0.0)
                         ? n_total / sum : (n_total == 0.0 ? 0.0 : 1.0);
            for (int i = 0; i < N_fl; i++) {
                nlte->nlte_level_populations[(lev_start + i) * n_shells + shell] =
                    xfl[i] * scale;
            }
        }
        free(xfl);
    }

    free(A_cm);
    free(b);
    return 0;
}

/* Update tau_sobolev for NLTE lines using NLTE level populations.
 * Floor the result at the nebular tau already in opacity->tau_sobolev:
 * NLTE rate matrices can collapse populations of dominant ions (e.g. Si II)
 * in inner shells where photoion pumping is unbalanced, producing tau values
 * many orders of magnitude below the Saha-Boltzmann nebular estimate.
 * Using max(nlte, nebular) preserves NLTE refinements for ions/levels where
 * NLTE actually increases opacity (e.g. Fe II in outer UV-forming shells)
 * while preventing pathological under-population in the inner photosphere. */
/* Per-Z skip mask, parsed once from LUMINA_NLTE_SKIP_Z (comma list, e.g. "14,16"). */
/* Gate: keep the SKIP_Z tau skip but still write line_source_S from the NLTE
 * populations (see the call site for why the two were entangled). */
int nlte_sl_write_on_skipz(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("LUMINA_SL_WRITE_SKIPZ");
                 v = (e && atoi(e)) ? 1 : 0;
                 if (v) printf("  [SL-WRITE] LUMINA_SL_WRITE_SKIPZ=1: SKIP_Z elements keep "
                               "nebular tau but DO get an NLTE line_source_S\n"); }
    return v;
}

static int nlte_skip_z[100];
static int nlte_skip_z_init = 0;
static void nlte_skip_z_load(void) {
    if (nlte_skip_z_init) return;
    nlte_skip_z_init = 1;
    const char *e = getenv("LUMINA_NLTE_SKIP_Z");
    if (!e || !*e) return;
    char buf[256]; strncpy(buf, e, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    char *tok = strtok(buf, ", \t");
    while (tok) {
        int z = atoi(tok);
        if (z > 0 && z < 100) nlte_skip_z[z] = 1;
        tok = strtok(NULL, ", \t");
    }
    printf("  [NLTE] LUMINA_NLTE_SKIP_Z active: ");
    for (int i = 1; i < 100; i++) if (nlte_skip_z[i]) printf("%d ", i);
    printf("(these elements keep nebular tau)\n");
}

void nlte_update_tau_sobolev(NLTEConfig *nlte, AtomicData *atom,
                              OpacityState *opacity,
                              double time_explosion, int n_shells) {
    tau_sobolev_require_refresh(opacity, "nlte_update_tau_sobolev");
    int n_lines = opacity->n_lines;
    nlte_skip_z_load();
    unsigned char pair_owned[NLTE_MAX_IONS] = {0};
    {
        int pairs[NLTE_PAIR_COUNT][2];
        const char *names[NLTE_PAIR_COUNT];
        int n_pairs = nlte_get_pairs(pairs, names);
        for (int p = 0; p < n_pairs; p++) {
            pair_owned[pairs[p][0]] = 1;
            pair_owned[pairs[p][1]] = 1;
        }
    }

    /* F0 fluorescence falsifier (DIAGNOSTIC ONLY, never a production config):
     * impose a controlled super-thermal departure S_l = X*B(T_e) on the Fe lines
     * in the 4475A window, on the FROZEN plasma, to test whether a non-thermal
     * 4475 source would actually emerge through transport. Since the converged
     * S_l is ~B(T_e) (audit S_l/B=1.0000), multiplying S_l by X imposes X*B.
     * PASS (modest X -> 4475 appears) => the populations are the binding layer
     * -> build the line-specific-Jbar rate-eq fix. FAIL (even X=10 nothing) =>
     * the blocker is upstream (super-level smearing / opacity / UV reservoir). */
    static int    fluor_init = 0;
    static double fluor_oracle_x = 1.0;
    static double fluor_lam_lo_cm = 4400e-8, fluor_lam_hi_cm = 4550e-8;
    if (!fluor_init) {
        const char *e = getenv("LUMINA_FLUOR_ORACLE_X");
        if (e) fluor_oracle_x = atof(e);
        const char *lo = getenv("LUMINA_FLUOR_ORACLE_LAM_LO");
        const char *hi = getenv("LUMINA_FLUOR_ORACLE_LAM_HI");
        if (lo) fluor_lam_lo_cm = atof(lo) * 1e-8;
        if (hi) fluor_lam_hi_cm = atof(hi) * 1e-8;
        if (fluor_oracle_x > 1.0)
            printf("  [FLUOR-ORACLE] S_l *= %.2f on Z=26 lines in [%.0f,%.0f]A "
                   "(DIAGNOSTIC; frozen-plasma 4475 falsifier)\n",
                   fluor_oracle_x, fluor_lam_lo_cm * 1e8, fluor_lam_hi_cm * 1e8);
        fluor_init = 1;
    }
    long fluor_hits = 0;

    for (int line = 0; line < n_lines; line++) {
        int ion_idx = nlte->nlte_line_map[line];
        if (ion_idx < 0) continue; /* not an NLTE line */

        /* Wave-3.2 R1: the 33-slot layout is an indexer, not an authority
         * grant.  Slots absent from every pair may write tau/source only in a
         * successfully committed EW target cell.  Shadow and off-target cells
         * retain their pre-existing nebular values byte-for-byte. */
        int candidate_only_slot = !pair_owned[ion_idx];

        int Z     = atom->line_atomic_number[line];
        /* SKIP_Z means "keep nebular tau" for this element. It used to `continue`
         * here, which ALSO skipped the line_source_S write below — an unintended
         * second effect: line_source_S then stayed 0 for the whole element and
         * every consumer silently substituted B(T_e), i.e. treated a strongly
         * NLTE ion's line source as LTE. Measured 2026-07-27 on parity33: Si II
         * (790 in-window lines, 157 NLTE levels) and Si III (669 lines, 147
         * levels, b_k up to 22.8) were the only ions with NLTE populations but a
         * thermal source; consumers are cmfgen.c:227 (binned line emissivity),
         * cmfgen.c:2359 (fine-grid solve) and plasma.c:13116 (mode-2 S_lag).
         * LUMINA_SL_WRITE_SKIPZ=1 keeps the tau skip but restores the source
         * write. Default 0 = previous behaviour, bit-identical. */
        int skip_tau = (Z > 0 && Z < 100 && nlte_skip_z[Z]);
        if (skip_tau && !nlte_sl_write_on_skipz()) continue;
        int ion_s = atom->line_ion_number[line];
        double f_lu   = atom->line_f_lu[line];
        double lam_cm = atom->line_wavelength_cm[line];

        /* Find the NLTE level indices for lower and upper */
        int ip = find_ion_pop_idx(atom, Z, ion_s);
        if (ip < 0) continue;
        int lev_base = atom->level_offset[ip];
        int lev_top  = atom->level_offset[ip + 1];

        int lower_global = -1, upper_global = -1;
        for (int l = lev_base; l < lev_top; l++) {
            if (atom->level_num[l] == atom->line_level_lower[line]) lower_global = l;
            if (atom->level_num[l] == atom->line_level_upper[line]) upper_global = l;
            if (lower_global >= 0 && upper_global >= 0) break;
        }
        if (lower_global < 0 || upper_global < 0) continue;

        int nlte_lo = nlte->global_to_nlte_level[lower_global];
        int nlte_up = nlte->global_to_nlte_level[upper_global];
        if (nlte_lo < 0 || nlte_up < 0) continue;

        int g_lo = atom->level_g[lower_global];
        int g_up = atom->level_g[upper_global];
        int element_index = -1;
        for (int e = 0; e < atom->n_elements; e++) {
            if (atom->element_Z[e] == Z) { element_index = e; break; }
        }
        int element_inactive = element_index >= 0 &&
            lumina_zinert_element_inactive(atom, element_index, n_shells);

        double nu_l = C_SPEED_OF_LIGHT / lam_cm;
        double src_prefac = 2.0 * H_PLANCK * nu_l * nu_l * nu_l
                            / (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
        for (int s = 0; s < n_shells; s++) {
            if (element_inactive) {
                opacity->tau_sobolev[(size_t)line * n_shells + s] = 0.0;
                if (opacity->line_source_S)
                    opacity->line_source_S[(size_t)line * n_shells + s] = 0.0;
                if (opacity->tau_validity)
                    opacity->tau_validity[(size_t)line * n_shells + s] = A208_EXACT_ZERO;
                if (opacity->line_source_validity)
                    opacity->line_source_validity[(size_t)line * n_shells + s] = A208_EXACT_ZERO;
                continue;
            }
            if (candidate_only_slot) {
                int zi = (Z == 16) ? 0 : (Z == 26 ? 1 : -1);
                int committed = nlte_element_wide_commit_enabled() && zi >= 0 &&
                    nlte_element_wide_matches(Z, s) && g_ew_tau_authority &&
                    g_ew_tau_authority_nshells == n_shells &&
                    g_ew_tau_authority[(size_t)zi * n_shells + s] == 1;
                if (!committed) continue;
            }
            double n_lower = nlte->nlte_level_populations[nlte_lo * n_shells + s];
            double n_upper = nlte->nlte_level_populations[nlte_up * n_shells + s];

            A208ValueView tau_view = a208_signed_sobolev(
                SOBOLEV_COEFF, f_lu, lam_cm, time_explosion,
                n_lower, n_upper, g_lo, g_up,
                opacity->tau_required_generation);
            if (!skip_tau)   /* SKIP_Z elements keep their nebular tau */
                opacity->tau_sobolev[line * n_shells + s] = tau_view.value;
            if (!skip_tau && opacity->tau_validity)
                opacity->tau_validity[line * n_shells + s] = tau_view.validity;

            /* CMF NLTE line source function (paper-method, fluorescence-bearing):
             * S_l = (2hv^3/c^2) / (g_u n_l / (g_l n_u) - 1), from the NLTE level
             * pops. Stored for the CMF formal solver; <=0 left for fallback. */
            A208ValueView source_view = a208_line_source(
                src_prefac, n_lower, n_upper, g_lo, g_up,
                opacity->tau_required_generation);
            double S_l = source_view.value;
            /* F0 fluorescence falsifier: impose S_l = X*B on Fe 4475-window lines */
            if (fluor_oracle_x > 1.0 && Z == 26 &&
                lam_cm >= fluor_lam_lo_cm && lam_cm <= fluor_lam_hi_cm) {
                if (source_view.validity == A208_VALID) S_l *= fluor_oracle_x;
                fluor_hits++;
            }
            opacity->line_source_S[line * n_shells + s] = S_l;
            if (opacity->line_source_validity)
                opacity->line_source_validity[line * n_shells + s] = source_view.validity;
        }
    }
    if (fluor_oracle_x > 1.0)
        printf("  [FLUOR-ORACLE] matched %ld (line,shell) cells (Z=26 in window)\n",
               fluor_hits);
    tau_sobolev_mark_computed(opacity, "nlte_update_tau_sobolev");
}

/* Master NLTE solver: solve all ions in all shells, update tau.
 * Step 1.5: Iterative CE convergence wrapper — because CE couples
 * different elements, we iterate until ion densities converge. */
/* Precompute the within-super-level Boltzmann fractions f_i from the current
 * T_e (energies measured relative to each SL's lowest-E anchor to avoid
 * overflow). f_i is the fraction of a super-level's population that sits in full
 * level i — used both to weight bb/bf rates during assembly AND to redistribute
 * the SL solution back to full levels. Identity SLs (1 FL) trivially get f=1.
 *
 * MUST be called before every solve (CPU and GPU paths) using the SAME current
 * T_e fed to the rate assembly. If skipped, within_sl_frac stays at its init
 * value 1.0, the reconstruction n_FL = n_SL*f collapses to a flat distribution,
 * and the departure coefficients blow up as b_k ~ exp(dE/kT_e) (the GPU-path
 * super-thermal bug). The f-weighting, when current, preserves detailed balance
 * exactly for both bb and bf. */
int nlte_precompute_within_sl_frac_checked(NLTEConfig *nlte, AtomicData *atom,
                                           PlasmaState *plasma, int n_shells) {
    PopulationAtomicView av = population_atomic_view(atom);
    PopulationStatus stamp_status = population_partition_view_check(
        &atom->partition_stamp, &av, plasma ? plasma->T_e : NULL,
        (size_t)n_shells, atom->partition_stamp.required_population_generation,
        plasma ? plasma->T_e_generation : 0);
    if (stamp_status != POP_OK) {
        fprintf(stderr, "[A2-07] within-super-level blocked: %s\n",
                population_status_name(stamp_status));
        return -1;
    }
    if (!nlte->super_mode) {
        nlte->within_sl_stamp = atom->partition_stamp;
        nlte->within_sl_stamp.n_items =
            (size_t)nlte->n_nlte_levels_total;
        return 0;
    }
    double *Zsl = (double *)malloc(
        (nlte->n_super_total > 0 ? nlte->n_super_total : 1) * sizeof(double));
    size_t frac_count = (size_t)nlte->n_nlte_levels_total * n_shells;
    double *work = (double *)malloc((frac_count ? frac_count : 1) * sizeof(double));
    if (!Zsl || !work) {
        fprintf(stderr,
                "[NLTE][OOM] within-super-level partition allocation failed\n");
        free(Zsl);
        free(work);
        return -1;
    }
    for (int s = 0; s < n_shells; s++) {
        double T_e = plasma->T_e[s];
        if (!isfinite(T_e) || T_e <= 0.0) {
            free(Zsl); free(work);
            return -1;
        }
        double kT = K_BOLTZMANN * T_e;
        for (int sl = 0; sl < nlte->n_super_total; sl++) Zsl[sl] = 0.0;
        for (int g = 0; g < nlte->n_nlte_levels_total; g++) {
            int gl = nlte->nlte_to_global_level[g];
            int sl = nlte->fl_to_super[g];
            int anchor = nlte->super_anchor_global[sl];
            double E_rel = (atom->level_energy_eV[gl] -
                            atom->level_energy_eV[anchor]) * EV_TO_ERG;
            if (!isfinite(E_rel) || E_rel < 0.0 || atom->level_g[gl] <= 0) {
                free(Zsl); free(work);
                return -1;
            }
            double w = (double)atom->level_g[gl] * exp(-E_rel / kT);
            if (!isfinite(w) || w < 0.0) {
                free(Zsl); free(work);
                return -1;
            }
            work[(size_t)g * n_shells + s] = w;
            Zsl[sl] += w;
        }
        for (int g = 0; g < nlte->n_nlte_levels_total; g++) {
            int sl = nlte->fl_to_super[g];
            double Z = Zsl[sl];
            size_t idx = (size_t)g * n_shells + s;
            if (!isfinite(Z) || Z <= 0.0) {
                free(Zsl); free(work);
                return -1;
            }
            work[idx] /= Z;
        }
    }
    memcpy(nlte->within_sl_frac, work, frac_count * sizeof(double));
    nlte->within_sl_stamp = atom->partition_stamp;
    nlte->within_sl_stamp.n_items =
        (size_t)nlte->n_nlte_levels_total;
    free(Zsl);
    free(work);
    return 0;
}

/* Legacy name retained, but the error is no longer explicitly discarded. */
int nlte_precompute_within_sl_frac(NLTEConfig *nlte, AtomicData *atom,
                                   PlasmaState *plasma, int n_shells) {
    return nlte_precompute_within_sl_frac_checked(
        nlte, atom, plasma, n_shells);
}

int nlte_solve_all(NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
                     OpacityState *opacity, double time_explosion,
                     int n_shells, GammaDeposition *gamma_dep) {
    printf("  [NLTE] Solving rate equations (with CE coupling)...\n");

    const char *forbidden_population_knobs[] = {
        "LUMINA_NLTE_FORCE_LTE_LEVELS",
        "LUMINA_NLTE_LTE_REPAIR",
        "LUMINA_NLTE_FLOOR_MODE",
        "LUMINA_NLTE_FLOOR_REG",
        "LUMINA_TOPSTAGE_THERMALIZE",
        "LUMINA_NLTE_BK_PARTIAL",
        "LUMINA_NLTE_BF_JEQB",
        "LUMINA_C2_MATRIX_BF",
        "LUMINA_NLTE_JEQB",
        "LUMINA_FROZENIN"
    };
    int forbidden_population_config = 0;
    for (size_t i = 0;
         i < sizeof(forbidden_population_knobs) /
             sizeof(forbidden_population_knobs[0]); i++) {
        const char *value = getenv(forbidden_population_knobs[i]);
        if (value && atoi(value) != 0) {
            forbidden_population_config = 1;
            break;
        }
    }
    if (forbidden_population_config) {
        nlte->population_first_error = POP_FORBIDDEN_FALLBACK;
        nlte->population_error_count++;
        population_counter_note(&nlte->population_counters,
                                POP_FORBIDDEN_FALLBACK);
        fprintf(stderr, "[A2-07] forbidden population fallback configuration\n");
        return -1;
    }
    if (nlte->radfield_view_status != RADIATION_FIELD_VIEW_OK ||
        !nlte->radfield_view.J_nu) {
        nlte->population_first_error = POP_BF_STALE;
        nlte->population_error_count++;
        population_counter_note(&nlte->population_counters, POP_BF_STALE);
        return -1;
    }
    if (nlte->line_view_status != LINE_JBAR_VIEW_OK ||
        !nlte->line_view.jbar) {
        PopulationStatus ps = nlte->line_view_status == LINE_JBAR_VIEW_PROFILE
                            ? POP_PROFILE_MISMATCH
                            : nlte->line_view_status == LINE_JBAR_VIEW_QHASH
                            ? POP_QUERY_HASH_MISMATCH : POP_BB_STALE;
        nlte->population_first_error = ps;
        nlte->population_error_count++;
        population_counter_note(&nlte->population_counters, ps);
        return -1;
    }
    PopulationStatus rate_pair_status = population_rate_views_check(
        POP_OK, nlte->radfield_view.generation,
        POP_OK, nlte->line_view.generation,
        nlte->radfield_view.generation);
    if (rate_pair_status != POP_OK) {
        nlte->population_first_error = rate_pair_status;
        nlte->population_error_count++;
        nlte->population_counters.pop_generation_mismatch++;
        return -1;
    }

    uint64_t next_generation = atom->population_committed_generation + 1;
    if (next_generation == 0) return -1;
    nlte->population_required_generation = next_generation;
    nlte->population_counters.pop_generation_required = next_generation;
    nlte->population_counters.pop_shells_attempted += (uint64_t)n_shells;
    PopulationTransaction pop_tx;
    double *published_level_populations = nlte->nlte_level_populations;
    double *published_ion_populations = atom->ion_number_density;
    double *published_ne = plasma->n_electron;
    double *published_partition = atom->partition_functions;
    PopulationDerivedStamp published_partition_stamp = atom->partition_stamp;
    PopulationDerivedStamp published_within_sl_stamp = nlte->within_sl_stamp;
    if (population_transaction_begin(
            &pop_tx, atom->ion_number_density,
            (size_t)atom->n_ion_pops * n_shells,
            nlte->nlte_level_populations,
            (size_t)nlte->n_nlte_levels_total * n_shells,
            plasma->n_electron, (size_t)n_shells,
            atom->partition_functions,
            (size_t)atom->n_ion_pops * n_shells, next_generation,
            &atom->population_committed_generation) != 0) {
        population_counter_note(&nlte->population_counters, POP_SOLVE_FAILED);
        return -1;
    }
    atom->ion_number_density = pop_tx.work_ion;
    nlte->nlte_level_populations = pop_tx.work_level;
    plasma->n_electron = pop_tx.work_ne;
    atom->partition_functions = pop_tx.work_partition;
#define A2_07_POP_ABORT(status_) do {                                      \
        atom->ion_number_density = published_ion_populations;              \
        nlte->nlte_level_populations = published_level_populations;        \
        plasma->n_electron = published_ne;                                 \
        atom->partition_functions = published_partition;                   \
        atom->partition_stamp = published_partition_stamp;                 \
        nlte->within_sl_stamp = published_within_sl_stamp;                 \
        population_transaction_abort(&pop_tx, (status_));                  \
        if (nlte->population_error_count == 0)                             \
            nlte->population_first_error = (status_);                      \
        nlte->population_error_count++;                                    \
        population_counter_note(&nlte->population_counters, (status_));    \
        return -1;                                                         \
    } while (0)

    PopulationAtomicView population_view = population_atomic_view(atom);
    PopulationStatus partition_status = population_partition_build(
        &population_view, plasma->T_e, (size_t)n_shells, next_generation,
        plasma->T_e_generation, atom->partition_functions,
        &atom->partition_stamp);
    if (partition_status != POP_OK)
        A2_07_POP_ABORT(partition_status);
    if (nlte_precompute_within_sl_frac_checked(
            nlte, atom, plasma, n_shells) != 0) {
        fprintf(stderr, "[NLTE] solve aborted: within-SL projection unavailable\n");
        A2_07_POP_ABORT(POP_STALE_DERIVED_TEMPERATURE);
    }

    /* Pair layout from the centralized builder (16 base pairs, or 23 with the O
     * triplet + stage-IV (III,IV) pairs under LUMINA_NLTE_STAGE4). Last-overlap
     * pairs share a prior slot; the solve below detects that generically. */
    int pairs[NLTE_PAIR_COUNT][2];
    const char *names[NLTE_PAIR_COUNT];
    int n_pairs = nlte_get_pairs(pairs, names);

    int ce_max_iter = 5;
    double ce_threshold = 1e-2;  /* 1% relative convergence on ion totals */
    double ce_damping = 0.5;     /* 50% damping */
    /* [ARTIS-PARITY B1 partial] tighten the outer coupling (CPU mirror of the GPU
     * path). Drop the 50% damping and iterate to convergence; the full
     * element-wide single SE matrix remains the residual. Gate OFF =>
     * byte-identical (5 iters / 0.5 damping). */
    if (artis_parity_enabled()) {
        ce_max_iter = 20;
        ce_damping  = 1.0;
        printf("  [ARTIS-PARITY B1] outer CE coupling tightened: damping=1.0, "
               "max_iter=%d (element-wide single matrix = residual)\n", ce_max_iter);
    }

    /* Save old ion totals for convergence check (n_nlte_ions * n_shells) */
    int n_ion_totals = nlte->n_nlte_ions * n_shells;
    double *old_ion_totals = (double *)calloc(n_ion_totals, sizeof(double));
    size_t pop_size = (size_t)nlte->n_nlte_levels_total * n_shells;
    double *old_pops = (double *)malloc(pop_size * sizeof(double));
    if (!old_ion_totals || !old_pops) {
        fprintf(stderr, "[NLTE][OOM] convergence state allocation failed\n");
        free(old_ion_totals);
        free(old_pops);
        A2_07_POP_ABORT(POP_SOLVE_FAILED);
    }
    /* Allocated only in the explicitly armed Wave-3 lane.  status is per
     * (S/Fe,shell): 1 means the self-tested candidate may replace legacy for
     * this CE pass; -1 means fail-closed and legacy must run. */
    if (nlte_element_wide_config_status() != 0) {
        fprintf(stderr, "[EW] invalid gate configuration; solve aborted\n");
        free(old_ion_totals);
        free(old_pops);
        A2_07_POP_ABORT(POP_FORBIDDEN_FALLBACK);
    }
    int ew_on = nlte_element_wide_enabled();
    int *ew_status = ew_on ? (int *)calloc((size_t)2 * n_shells, sizeof(int)) : NULL;
    if (ew_on && !ew_status) {
        fprintf(stderr, "[EW][OOM] status allocation failed; solve aborted\n");
        free(old_ion_totals);
        free(old_pops);
        A2_07_POP_ABORT(POP_SOLVE_FAILED);
    }
    if (ew_on) {
        for (int s = 0; s < n_shells; s++) {
            if (nlte_element_wide_matches(16, s)) {
                int verdict_pass = 0;
                int rc = nlte_element_wide_run_status(
                    nlte, atom, plasma, opacity, 16, s, time_explosion,
                    gamma_dep, nlte_element_wide_commit_enabled(),
                    &verdict_pass);
                ew_status[s] = rc < 0 ? -1 : verdict_pass;
                if (rc < 0) {
                    fprintf(stderr, "[EW] operational failure Z=16 s=%d\n", s);
                    free(ew_status);
                    free(old_ion_totals);
                    free(old_pops);
                    A2_07_POP_ABORT(POP_SOLVE_FAILED);
                }
            }
            if (nlte_element_wide_matches(26, s)) {
                int verdict_pass = 0;
                int rc = nlte_element_wide_run_status(
                    nlte, atom, plasma, opacity, 26, s, time_explosion,
                    gamma_dep, nlte_element_wide_commit_enabled(),
                    &verdict_pass);
                ew_status[n_shells + s] = rc < 0 ? -1 : verdict_pass;
                if (rc < 0) {
                    fprintf(stderr, "[EW] operational failure Z=26 s=%d\n", s);
                    free(ew_status);
                    free(old_ion_totals);
                    free(old_pops);
                    A2_07_POP_ABORT(POP_SOLVE_FAILED);
                }
            }
        }
    }

    int ce_converged = 0;
    for (int ce_iter = 0; ce_iter < ce_max_iter; ce_iter++) {
        nlte_jbar_dump_set_pass(ce_iter);   /* [withParityP GATE2] CE pass marker */
        /* Save current populations + compute old ion totals */
        memcpy(old_pops, nlte->nlte_level_populations, pop_size * sizeof(double));
        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int lev_s = nlte->nlte_ion_level_offset[ii];
            int lev_e = nlte->nlte_ion_level_offset[ii + 1];
            for (int s = 0; s < n_shells; s++) {
                double sum = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    sum += nlte->nlte_level_populations[l * n_shells + s];
                old_ion_totals[ii * n_shells + s] = sum;
            }
        }

        /* Solve all ion pairs */
        for (int p = 0; p < n_pairs; p++) {
            int lo = pairs[p][0], hi = pairs[p][1];

            /* C1: Preserve overlapping lo-ion populations across pairs that
             * share a slot (e.g. pair 14={28,29}=O(I-II) and pair 15={29,30}
             * =O(II-III) both touch slot 29 = O II). Without this, pair 15
             * silently overwrites the O II populations pair 14 just set,
             * with a Saha-derived ratio that ignores the O I↔O II coupling.
             * Save before, restore after — pair p's solve sees consistent
             * O II as the lower-boundary population, and the prior pair's
             * answer survives. */
            int lo_overlaps_prior = 0;
            int pair_shares_slot = 0;
            for (int pp = 0; pp < n_pairs; pp++) {
                if (pp == p) continue;
                if (pairs[pp][0] == lo || pairs[pp][1] == lo) {
                    pair_shares_slot = 1;
                    if (pp < p) lo_overlaps_prior = 1;
                }
                if (pairs[pp][0] == hi || pairs[pp][1] == hi)
                    pair_shares_slot = 1;
            }
            double *saved_lo = NULL;
            int saved_lev_s = 0, saved_lev_e = 0;
            if (lo_overlaps_prior) {
                saved_lev_s = nlte->nlte_ion_level_offset[lo];
                saved_lev_e = nlte->nlte_ion_level_offset[lo + 1];
                int n_save = (saved_lev_e - saved_lev_s) * n_shells;
                saved_lo = (double *)malloc((size_t)n_save * sizeof(double));
                if (!saved_lo) {
                    fprintf(stderr, "[NLTE][OOM] overlap save allocation failed\n");
                    free(ew_status);
                    free(old_ion_totals);
                    free(old_pops);
                    A2_07_POP_ABORT(POP_SOLVE_FAILED);
                }
                if (!ew_on) {
                    memcpy(saved_lo,
                           &nlte->nlte_level_populations[(size_t)saved_lev_s * n_shells],
                           (size_t)n_save * sizeof(double));
                } else {
                    for (int l = saved_lev_s; l < saved_lev_e; l++)
                        for (int s = 0; s < n_shells; s++)
                            saved_lo[(size_t)(l-saved_lev_s)*n_shells+s] =
                                nlte->nlte_level_populations[(size_t)l*n_shells+s];
                }
            }

            int pair_solve_failed = 0;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int s = 0; s < n_shells; s++) {
                int Zp = nlte->nlte_Z[lo];
                int zi = (Zp == 16) ? 0 : (Zp == 26 ? 1 : -1);
                if (ew_on && nlte_element_wide_commit_enabled() && zi >= 0 &&
                    nlte_element_wide_matches(Zp, s) &&
                    ew_status[(size_t)zi*n_shells+s] == 1)
                    continue; /* no pair/pin/topstage call for committed (Z,s) */
                if (nlte_solve_ion_shell(nlte, atom, plasma, opacity,
                                         lo, hi, s, time_explosion, gamma_dep,
                                         pair_shares_slot) != 0) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                    pair_solve_failed = 1;
                }
            }

            if (pair_solve_failed) {
                free(saved_lo);
                free(ew_status);
                free(old_ion_totals);
                free(old_pops);
                A2_07_POP_ABORT(POP_SOLVE_FAILED);
            }

            if (saved_lo) {
                nlte_ew_note_save_restore_call();
                int n_save = (saved_lev_e - saved_lev_s) * n_shells;
                if (!ew_on) {
                    memcpy(&nlte->nlte_level_populations[(size_t)saved_lev_s * n_shells],
                           saved_lo, (size_t)n_save * sizeof(double));
                } else {
                    int Zp = nlte->nlte_Z[lo];
                    int zi = (Zp == 16) ? 0 : (Zp == 26 ? 1 : -1);
                    for (int l = saved_lev_s; l < saved_lev_e; l++)
                        for (int s = 0; s < n_shells; s++) {
                            int committed = nlte_element_wide_commit_enabled() &&
                                zi >= 0 && nlte_element_wide_matches(Zp,s) &&
                                ew_status[(size_t)zi*n_shells+s] == 1;
                            if (!committed)
                                nlte->nlte_level_populations[(size_t)l*n_shells+s] =
                                  saved_lo[(size_t)(l-saved_lev_s)*n_shells+s];
                        }
                }
                free(saved_lo);
            }
        }

        /* Apply damping for iter >= 1 */
        if (ce_iter > 0) {
            if (!ew_on) {
                for (size_t i = 0; i < pop_size; i++) {
                    double n_new = nlte->nlte_level_populations[i];
                    double n_old = old_pops[i];
                    nlte->nlte_level_populations[i] = n_old +
                        ce_damping * (n_new - n_old);
                }
            } else {
                for (int g = 0; g < nlte->n_nlte_levels_total; g++) {
                    int gl = nlte->nlte_to_global_level[g], Zg = atom->level_Z[gl];
                    int zi = (Zg == 16) ? 0 : (Zg == 26 ? 1 : -1);
                    for (int s = 0; s < n_shells; s++) {
                        size_t i = (size_t)g*n_shells+s;
                        int committed = nlte_element_wide_commit_enabled() && zi >= 0 &&
                            nlte_element_wide_matches(Zg,s) &&
                            ew_status[(size_t)zi*n_shells+s] == 1;
                        if (!committed)
                            nlte->nlte_level_populations[i] = old_pops[i] +
                                ce_damping*(nlte->nlte_level_populations[i]-old_pops[i]);
                    }
                }
            }
        }

        /* Convergence: max relative change of ion totals */
        double max_rel_change = 0.0;
        if (ce_iter == 0) {
            /* Check if any old ion totals were nonzero */
            int has_prior = 0;
            for (int k = 0; k < n_ion_totals; k++) {
                if (old_ion_totals[k] > 1.0) { has_prior = 1; break; }
            }
            if (!has_prior) {
                printf("    CE iter %d: first solve (no prior populations)\n",
                       ce_iter + 1);
                continue;
            }
        }

        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int lev_s = nlte->nlte_ion_level_offset[ii];
            int lev_e = nlte->nlte_ion_level_offset[ii + 1];
            for (int s = 0; s < n_shells; s++) {
                double new_total = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    new_total += nlte->nlte_level_populations[l * n_shells + s];
                double old_total = old_ion_totals[ii * n_shells + s];
                if (old_total > 1.0) {
                    double rel = fabs(new_total - old_total) / old_total;
                    if (rel > max_rel_change) max_rel_change = rel;
                }
            }
        }

        printf("    CE iter %d: max_ion_rel_change = %.2e\n",
               ce_iter + 1, max_rel_change);

        if (max_rel_change < ce_threshold) {
            printf("    CE converged in %d iterations\n", ce_iter + 1);
            ce_converged = 1;
            break;
        }
    }
    free(old_pops);
    free(old_ion_totals);
    if (!ce_converged) {
        free(ew_status);
        A2_07_POP_ABORT(POP_SOLVE_FAILED);
    }

    /* Candidate assembly has ended; these are the authoritative counts from
     * the actual save/restore, per-ion pin and top-stage owner branches. */
    if (nlte_ew_publish_runtime_counts(nlte) != 0) {
        fprintf(stderr, "[EW] runtime-manifest I/O failure; solve aborted\n");
        free(ew_status);
        A2_07_POP_ABORT(POP_SOLVE_FAILED);
    }

    /* Print ion pair level counts */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0], hi = pairs[p][1];
        int n_levels = nlte->nlte_ion_level_offset[hi + 1] -
                       nlte->nlte_ion_level_offset[lo];
        printf("    %s (%d levels): done\n", names[p], n_levels);
    }

    /* Probe-B fix (task #29): feed the NLTE-solved ion stage back into opacity. */
    nlte_writeback_ion_stage(nlte, atom, plasma, opacity, time_explosion,
                             n_shells, pairs, n_pairs);

    /* Update tau_sobolev for NLTE lines */
    printf("  [NLTE] Updating tau_sobolev from NLTE populations...\n");
    free(g_ew_tau_authority);
    g_ew_tau_authority = ew_status;
    g_ew_tau_authority_nshells = ew_status ? n_shells : 0;
    nlte_update_tau_sobolev(nlte, atom, opacity, time_explosion, n_shells);

    if (zinert_audit_enabled() &&
        lumina_zinert_validate(atom, nlte, opacity, n_shells,
                               "post-nlte-cpu") != 0) {
        free(g_ew_tau_authority);
        g_ew_tau_authority = NULL;
        g_ew_tau_authority_nshells = 0;
        A2_07_POP_ABORT(POP_NONFINITE);
    }

    /* Print diagnostics: compare total NLTE vs nebular ion densities */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0];
        int lev_s = nlte->nlte_ion_level_offset[lo];
        int lev_e = nlte->nlte_ion_level_offset[lo + 1];
        double sum_nlte = 0.0;
        for (int l = lev_s; l < lev_e; l++)
            sum_nlte += nlte->nlte_level_populations[l * n_shells + 0];
        int ip = find_ion_pop_idx(atom, nlte->nlte_Z[lo], nlte->nlte_ion[lo]);
        double n_neb = (ip >= 0) ? atom->ion_number_density[ip * n_shells + 0] : 0.0;
        printf("    %s II shell 0: NLTE n_total=%.3e, nebular n_ion=%.3e\n",
               names[p], sum_nlte, n_neb);
    }
    /* [NLTE-DUMP] Per-shell, per-level population dump for UV diagnostic.
     * Activated by env LUMINA_NLTE_LEVEL_DUMP=1. Counter increments per
     * call so successive iterations land in distinct files. */
    {
        const char *env = getenv("LUMINA_NLTE_LEVEL_DUMP");
        if (env && env[0] == '1') {
            static int dump_counter = 0;
            char path[256];
            snprintf(path, sizeof(path),
                     "nlte_levels_iter%03d.csv", dump_counter++);
            FILE *fp = fopen(path, "w");
            if (!fp) {
                fprintf(stderr, "[NLTE-DUMP] failed to open %s\n", path);
            } else {
                fprintf(fp, "Z,ion,shell,level_idx,global_idx,E_eV,g,n_pop,T_e,n_ion_total,population_generation\n");
                for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
                    int Zv  = nlte->nlte_Z[ii];
                    int ion = nlte->nlte_ion[ii];
                    int lev_s = nlte->nlte_ion_level_offset[ii];
                    int lev_e = nlte->nlte_ion_level_offset[ii + 1];
                    int ip = find_ion_pop_idx(atom, Zv, ion);
                    for (int l = lev_s; l < lev_e; l++) {
                        int gi = nlte->nlte_to_global_level[l];
                        double E_eV = atom->level_energy_eV[gi];
                        int gw = atom->level_g[gi];
                        int local_l = l - lev_s;
                        for (int s = 0; s < n_shells; s++) {
                            double n_pop = nlte->nlte_level_populations[
                                (size_t)l * n_shells + s];
                            double T_e   = plasma->T_e[s];
                            double n_ion = (ip >= 0) ?
                                atom->ion_number_density[ip * n_shells + s] : 0.0;
                            fprintf(fp, "%d,%d,%d,%d,%d,%.6f,%d,%.6e,%.2f,%.6e,%llu\n",
                                    Zv, ion, s, local_l, gi, E_eV, gw, n_pop,
                                    T_e, n_ion,
                                    (unsigned long long)nlte->population_required_generation);
                        }
                    }
                }
                fclose(fp);
                printf("  [NLTE-DUMP] wrote %s\n", path);
            }
        }
    }
    atom->ion_number_density = published_ion_populations;
    nlte->nlte_level_populations = published_level_populations;
    plasma->n_electron = published_ne;
    atom->partition_functions = published_partition;
    PopulationStatus publish_status = population_transaction_commit(&pop_tx);
    if (publish_status != POP_OK) {
        atom->partition_stamp = published_partition_stamp;
        nlte->within_sl_stamp = published_within_sl_stamp;
        nlte->population_first_error = publish_status;
        nlte->population_error_count++;
        population_counter_note(&nlte->population_counters, publish_status);
        return -1;
    }
    nlte->population_committed_generation = next_generation;
    nlte->population_counters.pop_generation_committed = next_generation;
    nlte->population_counters.pop_shells_published += (uint64_t)n_shells;
#undef A2_07_POP_ABORT
    return 0;
}

/* ============================================================ */
/* Gamma-ray energy deposition from 56Ni/56Co decay             */
/* ============================================================ */

/* Physical constants for 56Ni/56Co decay */
#define LAMBDA_NI56   1.318e-6    /* 56Ni decay constant [s⁻¹], t½=6.077d */
#define LAMBDA_CO56   1.038e-7    /* 56Co decay constant [s⁻¹], t½=77.27d */
#define Q_NI56        2.803e-6    /* 56Ni decay energy [erg/decay] (1.75 MeV) */
#define Q_CO56        5.976e-6    /* 56Co decay energy [erg/decay] (3.73 MeV) */
#define KAPPA_GAMMA   0.025       /* Gray gamma-ray opacity [cm²/g] (Swartz+ 1995) */
#define ETA_NONTHERMAL 0.05       /* Fraction of deposition → ionization (Kozma & Fransson 1992) */
#define W_ION_EV      35.0        /* Mean energy per ion pair [eV] */

/* Derive the non-thermal ionization rate density from heating_rate and register it
 * for the freeze-out guard. Shared by compute_gamma_deposition AND the external
 * deposition-file path (cuda.cu), which only loads heating_rate — without this the
 * non-thermal ionization is silently ZERO in ARTIS-comparison runs. */
void gamma_deposition_compute_nonthermal(GammaDeposition *gd) {
    if (!gd || !gd->nonthermal_ioniz_rate || !gd->heating_rate) return;
    /* Non-thermal ionizations = deposition / W, where W = mean energy per ion pair
     * (Kozma&Fransson 1992; ~33-35 eV) ALREADY includes the heating/excitation
     * losses per ionization. The legacy formula multiplied by an EXTRA
     * ETA_NONTHERMAL=0.05 — a double-count that under-ionized by ~20× and starved
     * the thin outer. Physical rate (no free parameter): heating/W. (LUMINA_NT_LEGACY
     * restores the old 0.05·heating/W for the A/B record.) */
    /* DEFAULT = the constant ETA_NONTHERMAL (encodes the high-x_e effective ionpot,
     * ~700 eV; correct in the ionized interior). Removing it (LUMINA_NT_FULL=1, the
     * heating/W max rate with eta_ion=1) over-ionizes the inner ~20× — the physical
     * eff_ionpot is x_e-dependent (Xu&McCray 1991: eta_ion(x_e=0.5)=0.033→0 at high
     * x_e), so the non-thermal is PHYSICALLY WEAK at the outer (x_e~0.5) and is NOT
     * the lever for the hot ionized outer (that is photoionization / hard UV). */
    static int full = -1;
    if (full < 0) { const char *e = getenv("LUMINA_NT_FULL");
                    full = (e && atoi(e)) ? 1 : 0; }
    double pref = full ? 1.0 : ETA_NONTHERMAL;
    for (int s = 0; s < gd->n_shells; s++)
        gd->nonthermal_ioniz_rate[s] = pref * gd->heating_rate[s]
                                        / (W_ION_EV * EV_TO_ERG);
    frozenin_set_nt_rate(gd->nonthermal_ioniz_rate, gd->n_shells);
}

void gamma_deposition_init(GammaDeposition *gd, int n_shells) {
    gd->n_shells = n_shells;
    gd->heating_rate = (double *)calloc(n_shells, sizeof(double));
    gd->nonthermal_ioniz_rate = (double *)calloc(n_shells, sizeof(double));
}

void gamma_deposition_free(GammaDeposition *gd) {
    free(gd->heating_rate);
    free(gd->nonthermal_ioniz_rate);
    gd->heating_rate = NULL;
    gd->nonthermal_ioniz_rate = NULL;
}

void compute_gamma_deposition(GammaDeposition *gd, AtomicData *atom,
                               PlasmaState *plasma, Geometry *geo) {
    int n_shells = gd->n_shells;
    double t_exp = geo->time_explosion;

    /* Find element indices for Ni(Z=28) and Co(Z=27) */
    int elem_ni = -1, elem_co = -1;
    for (int e = 0; e < atom->n_elements; e++) {
        if (atom->element_Z[e] == 28) elem_ni = e;
        if (atom->element_Z[e] == 27) elem_co = e;
    }

    /* Bateman equation for 56Ni → 56Co → 56Fe:
     * N_Ni(t) = N_Ni(0) × exp(-λ_Ni × t)
     * N_Co(t) = N_Ni(0) × [λ_Ni/(λ_Co-λ_Ni)] × [exp(-λ_Ni×t) - exp(-λ_Co×t)]
     *           + N_Co(0) × exp(-λ_Co × t)
     * Note: At t=0, all Ni is 56Ni; Co abundance is initial 56Co (if any). */
    double exp_ni = exp(-LAMBDA_NI56 * t_exp);
    double exp_co = exp(-LAMBDA_CO56 * t_exp);
    double bateman_factor = LAMBDA_NI56 / (LAMBDA_CO56 - LAMBDA_NI56);

    /* Compute per-shell energy generation and outward column density */
    double *epsilon_gamma = (double *)calloc(n_shells, sizeof(double)); /* erg/s/cm³ */
    double *column_density = (double *)calloc(n_shells, sizeof(double)); /* g/cm² */

    for (int s = 0; s < n_shells; s++) {
        double rho = plasma->rho[s];

        /* Number density of 56Ni and 56Co from mass fractions.
         * We use the current Ni/Co abundances as initial mass fractions,
         * then apply time evolution. Ni mass = 56 amu. */
        double X_ni = (elem_ni >= 0) ? atom->abundances[elem_ni * n_shells + s] : 0.0;
        double X_co = (elem_co >= 0) ? atom->abundances[elem_co * n_shells + s] : 0.0;

        /* Initial number densities at t=0 */
        double n_ni_0 = X_ni * rho / (56.0 * AMU); /* cm⁻³ */
        double n_co_0 = X_co * rho / (56.0 * AMU);

        /* Current number densities from decay */
        double n_ni = n_ni_0 * exp_ni;
        double n_co = n_ni_0 * bateman_factor * (exp_ni - exp_co) + n_co_0 * exp_co;
        if (n_co < 0.0) n_co = 0.0;

        /* Local gamma-ray energy generation rate [erg/s/cm³] */
        epsilon_gamma[s] = LAMBDA_NI56 * n_ni * Q_NI56 + LAMBDA_CO56 * n_co * Q_CO56;
    }

    /* Outward column density: Σ(s) = Σ_{s'=s}^{N-1} ρ(s') × Δr(s') */
    column_density[n_shells - 1] = plasma->rho[n_shells - 1] *
        (geo->r_outer[n_shells - 1] - geo->r_inner[n_shells - 1]);
    for (int s = n_shells - 2; s >= 0; s--) {
        double dr = geo->r_outer[s] - geo->r_inner[s];
        column_density[s] = column_density[s + 1] + plasma->rho[s] * dr;
    }

    /* Deposition fraction and rates */
    for (int s = 0; s < n_shells; s++) {
        double tau_gamma = KAPPA_GAMMA * column_density[s];
        double f_dep = 1.0 - exp(-tau_gamma);

        gd->heating_rate[s] = epsilon_gamma[s] * f_dep;
    }
    gamma_deposition_compute_nonthermal(gd);   /* nonthermal rate + register */

    free(epsilon_gamma);
    free(column_density);
}

/* ============================================================ */
/* Sobolev line overlap correction                               */
/* ============================================================ */

void apply_overlap_corrections(AtomicData *atom, OpacityState *opacity,
                                PlasmaState *plasma) {
    tau_sobolev_require_refresh(opacity, "apply_overlap_corrections");
    int n_lines = opacity->n_lines;
    int n_shells = opacity->n_shells;

    /* Work on a copy of the original tau values */
    size_t tau_size = (size_t)n_lines * n_shells;
    double *tau_orig = (double *)malloc(tau_size * sizeof(double));
    memcpy(tau_orig, opacity->tau_sobolev, tau_size * sizeof(double));

    /* C7: build a Z -> mass_amu lookup once instead of approximating A ≈ 2Z.
     * 2Z understates heavy elements (Fe Z=26 → A=52 vs true ~56, Ni Z=28 →
     * 56 vs 58.7) and overstates light ones (He Z=2 → 4 vs 4.0 ≈ ok, but C
     * Z=6 → 12 vs 12.01 ok; lopsided for Si Z=14 → 28 vs 28.09 ok; only
     * heavy/odd-Z and H differ — H Z=1 → 2 vs 1.008 = 2× error). Doppler
     * width v_th ∝ 1/√A, so a 2× mass error gives √2 ≈ 1.4× wrong v_th. */
    double Z_to_amu[100];
    for (int z = 0; z < 100; z++) Z_to_amu[z] = (double)(2 * z); /* fallback */
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        if (z > 0 && z < 100) Z_to_amu[z] = atom->element_mass_amu[e];
    }

    for (int s = 0; s < n_shells; s++) {
        double T_rad = plasma->T_e[s];

        for (int i = 0; i < n_lines; i++) {
            double tau_i = tau_orig[i * n_shells + s];
            if (tau_i < 1e-10) continue; /* skip negligible lines */

            double nu_i = opacity->line_list_nu[i];
            int Z_i = atom->line_atomic_number[i];
            double mass_amu = (Z_i > 0 && Z_i < 100) ? Z_to_amu[Z_i] : (2.0 * Z_i);
            double v_th = sqrt(2.0 * K_BOLTZMANN * T_rad / (mass_amu * AMU));
            double delta_nu_th = nu_i * v_th / C_SPEED_OF_LIGHT;

            if (delta_nu_th <= 0.0) continue;

            /* Scan forward neighbors (lower frequency, j > i in descending array) */
            double tau_overlap = 0.0;
            for (int j = i + 1; j < n_lines && j <= i + 10; j++) {
                double dnu = nu_i - opacity->line_list_nu[j];
                if (dnu > 3.0 * delta_nu_th) break;
                double tau_j = tau_orig[j * n_shells + s];
                tau_overlap += tau_j * exp(-(dnu / delta_nu_th) * (dnu / delta_nu_th));
            }

            /* Scan backward neighbors (higher frequency) */
            for (int j = i - 1; j >= 0 && j >= i - 10; j--) {
                double dnu = opacity->line_list_nu[j] - nu_i;
                if (dnu > 3.0 * delta_nu_th) break;
                double tau_j = tau_orig[j * n_shells + s];
                tau_overlap += tau_j * exp(-(dnu / delta_nu_th) * (dnu / delta_nu_th));
            }

            /* Apply correction: tau_eff = tau_i² / (tau_i + tau_overlap) */
            if (tau_overlap > 0.01 * tau_i) {
                double correction = tau_i / (tau_i + tau_overlap);
                opacity->tau_sobolev[i * n_shells + s] = tau_i * correction;
            }
        }
    }

    free(tau_orig);
    tau_sobolev_mark_computed(opacity, "apply_overlap_corrections");
}

/* Rescale geometry and density for a new epoch (homologous expansion).
   v = r/t is invariant; r(t_new) = v * t_new, rho ~ t^-3. */
void rescale_epoch(Geometry *geo, PlasmaState *plasma, double t_new) {
    double t_ref = geo->time_explosion;
    double ratio = t_new / t_ref;
    double rho_scale = 1.0 / (ratio * ratio * ratio);
    for (int i = 0; i < geo->n_shells; i++) {
        geo->r_inner[i] = geo->v_inner[i] * t_new;
        geo->r_outer[i] = geo->v_outer[i] * t_new;
        plasma->rho[i] *= rho_scale;
    }
    geo->time_explosion = t_new;
}

/* ============================================================ */
/* P5: Formal integral spectrum (noise-free, p-z formalism)     */
/*                                                              */
/* For each observed frequency nu_obs, integrate along rays     */
/* with impact parameters p from 0 to r_outer:                 */
/*   I_nu(p) = I_core * exp(-tau_tot) + sum_lines S_l*(1-e^-tau_l)*e^-tau_above */
/*   L_nu = 4*pi * integral( I_nu(p) * 2*pi*p dp )            */
/* ============================================================ */

/* Binary search in descending-sorted array: find first index with val <= target */
static int bsearch_descending_le(const double *arr, int n, double target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] > target) lo = mid + 1;
        else hi = mid;
    }
    return lo; /* first index where arr[lo] <= target */
}

/* [FORMAL-CONS-WINDOW] ==================================================
 * int_x^inf t^3/(e^t - 1) dt  in closed form (Widger & Woodall 1976):
 *     sum_{n>=1} e^{-nx} ( x^3/n + 3x^2/n^2 + 6x/n^3 + 6/n^4 )
 * exact (not a quadrature rule): expand 1/(e^t-1) = sum_{n>=1} e^{-nt} and
 * integrate term by term.  At x=0 it returns sum 6/n^4 = 6*pi^4/90 = pi^4/15,
 * the Stefan-Boltzmann normalization, so the two limits are consistent by
 * construction.  Summed until the term is below 1e-18 of the running total
 * (double round-off), hard cap 20000 terms; for the windows this routine uses
 * (x >= 0.7) the cap is never approached -- the tail decays like e^{-nx}.
 * NOT used by any physics path: read only by the FORMAL-CONS report. */
static double planck_tail_integral(double x)
{
    const double pi4_15 = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 15.0;
    if (!(x > 0.0)) return pi4_15;
    double s = 0.0;
    for (int n = 1; n <= 20000; n++) {
        double dn = (double)n, nx = dn * x;
        if (nx > 700.0) break;              /* e^-700 underflows to 0 */
        double term = exp(-nx) * (x * x * x / dn + 3.0 * x * x / (dn * dn)
                                  + 6.0 * x / (dn * dn * dn)
                                  + 6.0 / (dn * dn * dn * dn));
        s += term;
        if (term < 1.0e-18 * s) break;
    }
    return s;
}

/* Fraction of a Planck surface's total emergent flux that lands inside the
 * wavelength window [lam_lo_A, lam_hi_A]:
 *     f_win = pi * int_win B_lambda(T) dlambda / (sigma T^4)
 * With x = h c / (lambda k T) this is
 *     [ G(x(lam_hi)) - G(x(lam_lo)) ] / (pi^4/15),   G = planck_tail_integral,
 * built from the SAME header constants (H_PLANCK, C_SPEED_OF_LIGHT,
 * K_BOLTZMANN, SIGMA_SB) the rest of this file uses. */
static double planck_band_fraction(double T, double lam_lo_A, double lam_hi_A)
{
    if (!(T > 0.0) || !(lam_lo_A > 0.0) || !(lam_hi_A > lam_lo_A)) return 0.0;
    const double hck = H_PLANCK * C_SPEED_OF_LIGHT / K_BOLTZMANN; /* cm*K */
    double x_hi = hck / (lam_lo_A * 1.0e-8 * T);  /* short lambda -> large x */
    double x_lo = hck / (lam_hi_A * 1.0e-8 * T);
    /* deliberately unclamped: 0 <= f <= 1 holds by construction, so a value
     * outside it would be a defect to see, not to hide. */
    return (planck_tail_integral(x_lo) - planck_tail_integral(x_hi))
         / (M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 15.0);
}

/* Binary search in descending-sorted array: find last index with val >= target */
static int bsearch_descending_ge(const double *arr, int n, double target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] >= target) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1; /* last index where arr[lo-1] >= target */
}

void compute_formal_integral_spectrum(
    Geometry *geo, PlasmaState *plasma, OpacityState *opacity,
    AtomicData *atom, NLTEConfig *nlte, double T_inner,
    Spectrum *spec_formal, int n_impact, double L_dep)
{
    int n_shells = geo->n_shells;
    int n_lines  = opacity->n_lines;
    double t_exp = geo->time_explosion;
    double ct    = C_SPEED_OF_LIGHT * t_exp;
    double r_phot  = geo->r_inner[0];
    double r_outer = geo->r_outer[n_shells - 1];
    double beta_max = r_outer / (C_SPEED_OF_LIGHT * t_exp); /* v_outer / c */
    double dp = r_outer / n_impact;

    /* === Formal-integral ablation knobs (Task #227) === */
    const char *_env_cut  = getenv("LUMINA_FI_TAU_CUTOFF");
    const char *_env_cont = getenv("LUMINA_FI_CONT_OPACITY");
    const char *_env_idil = getenv("LUMINA_FI_INNER_DILUTE");
    /* === [FORMAL-FIX] reduced-scope repair gate (2026-07-29) ===================
     * LUMINA_FORMAL_FIX=1 turns on the three items the adversarial re-verification
     * of the clamp census left CONFIRMED for this routine
     * (validation/cmfgen_toy06_19p48d/analysis/clamp_census/ADVERSARIAL.md):
     *   R1  zero-point: the impact-parameter quadrature charges a whole annulus
     *       to the core, so the backlight leg is +6.8% high (D-5) -> exact
     *       two-zone ray placement (lumina_fi_impact_ray, lumina.h).
     *   R2  no continuum sink: fi_use_cont defaulted to 0, so nothing on the ray
     *       is attenuated by electron scattering (radial tau_es = 1.486) ->
     *       default it ON under the gate, reusing the existing segment machinery
     *       (attenuates the line emission AND the p<r_phot backlight alike).
     *   R3  tau/S pair hygiene: S comes from the NLTE writer while tau may come
     *       from the nebular writer; where the two are not the same population,
     *       the emission is replaced by the fused closed form that is exactly
     *       consistent with the tau actually used (see the consumption site).
     * DEFAULT 0 => every arithmetic path below is the withParityU path, bit for
     * bit.  THIS GATE DOES NOT CLAIM TO FIX THE 24.99x OVER-LUMINOSITY: the same
     * audit bounded the W*B(T_rad) fallback leg at 0.07% of the 2000-4000A
     * participating lines (D-4) and the missing continuum sink at <= e^-1.486
     * (~4x); the residual carrier is the super-thermal S_l itself (D-7), which is
     * out of scope here. */
    int fi_fix = 0;
    { const char *_e = getenv("LUMINA_FORMAL_FIX"); if (_e && atoi(_e)) fi_fix = 1; }
    double fi_tau_cutoff = _env_cut  ? atof(_env_cut)  : 1.0e-5;
    int    fi_use_cont   = _env_cont ? atoi(_env_cont) : (fi_fix ? 1 : 0);
    double fi_W_inner    = _env_idil ? atof(_env_idil) : 1.0;
    /* Decisive test (2026-06-17): clamp thick-line S_l -> B(T_e) for tau>thr to
     * test whether super-thermal feature-line source functions are the
     * formal-spectrum killer. Value = tau threshold (e.g. 1.0); 0 = off. */
    const char *_env_clamp = getenv("LUMINA_FI_CLAMP_SL");
    double fi_clamp_sl = _env_clamp ? atof(_env_clamp) : 0.0;
    if (fi_clamp_sl > 0.0)
        printf("  [FI] S_l -> B(T_e) clamp for tau>%.2f (LUMINA_FI_CLAMP_SL)\n", fi_clamp_sl);
    /* Session-closing falsifier (codex 2026-06-19): test whether the too-RED
     * continuum is driven by blue-selective IGE forest blanketing.
     *  FOREST_NOBLANK: IGE forest lines (Z>=21) contribute NOTHING to tau_acc
     *    (remove their wavelength-selective blanketing) -> peak should jump
     *    blueward 8500->~6541A (the bare B(T_inner) backlight color) if the
     *    forest blanketing is the reddening cause.
     *  LEDGER: per output-bin, the BACKLIGHT luminosity absorbed by IGE forest
     *    vs other lines = B(T_inner)*exp(-tau_acc)*(1-e^-tau_l) on core rays,
     *    split by ion group -> lumina_fi_abs_ledger.csv. Confirms blue IGE
     *    absorption dominates the deleted-flux budget. */
    int fi_noblank = 0, fi_ledger = 0;
    { const char *nb = getenv("LUMINA_FI_FOREST_NOBLANK");
      if (nb && atoi(nb)) fi_noblank = 1;
      const char *lg = getenv("LUMINA_FI_LEDGER");
      if (lg && atoi(lg)) fi_ledger = 1; }
    double *absL_ige = NULL, *absL_oth = NULL;
    if (fi_ledger) {
        absL_ige = (double*)calloc(spec_formal->n_bins, sizeof(double));
        absL_oth = (double*)calloc(spec_formal->n_bins, sizeof(double));
    }
    if (fi_noblank) printf("  [FI] IGE forest (Z>=21) blanketing REMOVED from tau_acc\n");
    if (fi_ledger)  printf("  [FI] absorbed-energy ledger ON -> lumina_fi_abs_ledger.csv\n");

    printf("\n=== Formal Integral Spectrum ===\n");
    printf("  Impact parameters: %d, beta_max=%.4f\n", n_impact, beta_max);
    if (fi_tau_cutoff != 1.0e-5)
        printf("  [FI] tau_sob cutoff = %.2e (LUMINA_FI_TAU_CUTOFF)\n", fi_tau_cutoff);
    if (fi_use_cont)
        printf("  [FI] continuum e-scatter opacity ON (LUMINA_FI_CONT_OPACITY=1)\n");
    if (fi_W_inner != 1.0)
        printf("  [FI] inner Planck dilution W=%.3f (LUMINA_FI_INNER_DILUTE)\n", fi_W_inner);

    /* [FORMAL-FIX R3] per-line tau/S pair provenance, built once.
     *   FI_PAIR_UNKNOWN(0) the nebular writer could not resolve the (ion,levels)
     *                      of this line, so it holds tau=1e-100 (compute_tau_
     *                      sobolev) and there is no population to emit with.
     *   1/2/3              nebular-owned: tau came from compute_tau_sobolev with
     *                      dilute-LTE level pops n_k = w_k*(g_k/Z)*n_ion*
     *                      exp(-E_k/kT_rad), w_k = 1 (metastable) else W. The
     *                      code records w_upper/w_lower = 1, W, or 1/W.
     *   FI_PAIR_NLTE(4)    nlte_update_tau_sobolev wrote BOTH tau and
     *                      line_source_S from the same NLTE populations in the
     *                      same loop -> the pair is sound, leave it alone.
     * The nebular-owned branch is the one the legacy code served with
     * W*B(T_rad), a value that belongs to no population in the run. */
    enum { FI_PAIR_UNKNOWN = 0, FI_PAIR_NEB_SAME = 1, FI_PAIR_NEB_UDIL = 2,
           FI_PAIR_NEB_LDIL = 3, FI_PAIR_NLTE = 4 };
    unsigned char *fi_pair = NULL;
    if (fi_fix && atom && atom->line_atomic_number && atom->line_ion_number) {
        nlte_skip_z_load();
        fi_pair = (unsigned char *)calloc((size_t)n_lines, 1);
    }
    if (fi_pair) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int l = 0; l < n_lines; l++) {
            int Z = atom->line_atomic_number[l];
            int skip_tau = (Z > 0 && Z < 100 && nlte_skip_z[Z]);
            if (nlte && nlte->nlte_line_map && nlte->nlte_line_map[l] >= 0 &&
                opacity->line_source_S != NULL && !skip_tau) {
                fi_pair[l] = FI_PAIR_NLTE;
                continue;
            }
            /* mirror compute_tau_sobolev's lookup exactly: same ion index, same
             * level search, same metastable flags that set the dilution weights */
            int ipop = find_ion_pop_idx(atom, Z, atom->line_ion_number[l]);
            if (ipop < 0) continue;
            int lo = atom->level_offset[ipop], hi = atom->level_offset[ipop + 1];
            int li = -1, ui = -1;
            for (int k = lo; k < hi; k++) {
                if (atom->level_num[k] == atom->line_level_lower[l]) li = k;
                if (atom->level_num[k] == atom->line_level_upper[l]) ui = k;
                if (li >= 0 && ui >= 0) break;
            }
            if (li < 0 || ui < 0) continue;
            int ml = atom->level_metastable[li] ? 1 : 0;
            int mu = atom->level_metastable[ui] ? 1 : 0;
            fi_pair[l] = (ml == mu) ? FI_PAIR_NEB_SAME
                                    : (mu ? FI_PAIR_NEB_LDIL : FI_PAIR_NEB_UDIL);
        }
    }
    long fi_n_fused = 0, fi_n_orphan = 0;

    if (fi_fix) {
        /* R1 zero-point, measured on this very geometry: core quadrature weight
         * before/after, normalized by the exact core weight r_phot^2/2. */
        double wc_old = 0.0, wc_new = 0.0, wc_exact = 0.5 * r_phot * r_phot;
        for (int ip = 0; ip < n_impact; ip++) {
            double p_o = dp * (ip + 0.5), p_n, d_n;
            if (p_o < r_phot && p_o < r_outer) wc_old += p_o * dp;
            lumina_fi_impact_ray(1, ip, n_impact, r_phot, r_outer, &p_n, &d_n);
            if (p_n < r_phot) wc_new += p_n * d_n;
        }
        printf("  [FORMAL-FIX] LUMINA_FORMAL_FIX=1: R1 core-ray quadrature zero-point "
               "%.6f -> %.6f (exact=1); R2 continuum sink %s; R3 tau/S pair hygiene ON "
               "(nebular-owned lines -> fused n_u closed form, NOT a 25x claim)\n",
               (wc_exact > 0.0) ? wc_old / wc_exact : 0.0,
               (wc_exact > 0.0) ? wc_new / wc_exact : 0.0,
               fi_use_cont ? "ON" : "OFF(env override)");
    }

    /* Zero output spectrum */
    for (int b = 0; b < spec_formal->n_bins; b++)
        spec_formal->flux[b] = 0.0;

    /* For each wavelength bin (parallelized) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10) \
                             reduction(+:fi_n_fused, fi_n_orphan)
    #endif
    for (int bin = 0; bin < spec_formal->n_bins; bin++) {
        double lambda_cm = spec_formal->wavelength[bin] * 1.0e-8;
        double nu_obs = C_SPEED_OF_LIGHT / lambda_cm;

        /* Frequency range of lines that could resonate within the ejecta */
        double nu_line_min = nu_obs * (1.0 - beta_max);
        double nu_line_max = nu_obs * (1.0 + beta_max);

        /* Binary search in line_list_nu (sorted DESCENDING by frequency) */
        int l_first = bsearch_descending_le(opacity->line_list_nu, n_lines, nu_line_max);
        int l_last  = bsearch_descending_ge(opacity->line_list_nu, n_lines, nu_line_min);

        double L_nu_integral = 0.0;
        double ledg_ige = 0.0, ledg_oth = 0.0;   /* absorbed-backlight ledger */

        for (int ip = 0; ip < n_impact; ip++) {
            /* [FORMAL-FIX R1] fi_fix=0 reproduces p = dp*(ip+0.5), dp_ray = dp
             * exactly (same values, same association) => bit-identical. */
            double p, dp_ray;
            lumina_fi_impact_ray(fi_fix, ip, n_impact, r_phot, r_outer, &p, &dp_ray);
            double p2 = p * p;
            if (p >= r_outer) continue;

            double z_max_ejecta = sqrt(r_outer * r_outer - p2);
            double z_phot = (p < r_phot) ? sqrt(r_phot * r_phot - p2) : 0.0;

            double I_nu = 0.0;
            double tau_acc = 0.0;
            double z_prev = z_max_ejecta; /* observer-side end of current segment */

            /* Walk from observer side inward (z decreasing).
             * Lines sorted nu descending → as index increases, nu decreases, z increases.
             * So iterate from l_last (lowest nu, largest z) backward to l_first. */
            for (int l = l_last; l >= l_first; l--) {
                double nu_l = opacity->line_list_nu[l];
                double z = ct * (1.0 - nu_l / nu_obs);

                /* Check z within valid ejecta range for this impact parameter */
                if (z > z_max_ejecta || z < -z_max_ejecta) continue;
                if (p < r_phot && z < z_phot) continue; /* behind photosphere */

                double r = sqrt(p2 + z * z);
                if (r < r_phot || r > r_outer) continue;

                /* Find shell */
                int shell = -1;
                for (int s = 0; s < n_shells; s++) {
                    if (r >= geo->r_inner[s] && r < geo->r_outer[s]) {
                        shell = s;
                        break;
                    }
                }
                if (shell < 0) continue;

                double tau_sob = opacity->tau_sobolev[l * n_shells + shell];
                if (tau_sob < fi_tau_cutoff) continue; /* env-gated cutoff */

                /* Continuum segment from z_prev to z (env-gated).
                 * Adds e-scatter attenuation + thermal continuum emission. */
                if (fi_use_cont && z_prev > z) {
                    double dz = z_prev - z;
                    double z_mid = 0.5 * (z_prev + z);
                    double r_mid = sqrt(p2 + z_mid * z_mid);
                    int shell_mid = -1;
                    for (int s = 0; s < n_shells; s++) {
                        if (r_mid >= geo->r_inner[s] && r_mid < geo->r_outer[s]) {
                            shell_mid = s; break;
                        }
                    }
                    if (shell_mid >= 0) {
                        double chi_cont = opacity->electron_density[shell_mid] * SIGMA_THOMSON;
                        double dtau_c = chi_cont * dz;
                        double S_cont = nlte_get_J_at_nu(nlte, shell_mid, nu_obs);
                        double oz = (dtau_c > 500.0) ? 1.0 : (1.0 - exp(-dtau_c));
                        I_nu   += S_cont * oz * exp(-tau_acc);
                        tau_acc += dtau_c;
                    }
                }

                /* LINE source function: the NLTE two-level source S_l (carries
                 * fluorescence/thermalization), NOT the binned mean intensity J.
                 * Using S=J made the line re-emit what it absorbs at the same
                 * observer frequency -> P-Cygni troughs refilled -> featureless
                 * (coherent-scatter degeneracy). This is the exact fix the
                 * sibling CMF path already carries (line_source_S, ~8252);
                 * back-ported here per 2-reviewer verdict 2026-06-13. Fallback
                 * dilute-LTE W*B(T_rad) for lines outside the NLTE network. */
                double S = (opacity->line_source_S != NULL)
                         ? opacity->line_source_S[l * n_shells + shell] : 0.0;
                if (!fi_fix) {
                    if (S <= 0.0)
                        S = nlte_get_J_at_nu(nlte, shell, nu_l);
                } else {
                    /* [FORMAL-FIX R3] tau/S pair hygiene.
                     * Sobolev identity (exact, no new physics):
                     *   tau  = C_sob f_lu lam t n_l (1 - x),  x = g_l n_u/(g_u n_l)
                     *   S    = (2h nu^3/c^2) x/(1-x)
                     *   => S*tau = (2h nu^3/c^2) C_sob f_lu lam t (g_l/g_u) n_u
                     *   => S*(1-e^-tau) = (2h nu^3/c^2) C_sob f_lu lam t (g_l/g_u)
                     *                     * n_u * beta(tau),  beta=(1-e^-tau)/tau
                     * i.e. the emission is FIXED by n_u and the tau actually used
                     * for the attenuation. For a nebular-owned line, tau was built
                     * from dilute-LTE pops, so its own x is known in closed form,
                     * x = (w_u/w_l) exp(-h nu/k T_rad), and the emission below is
                     * exactly that fused expression -- proportional to n_u,
                     * bounded by it, and no invented value. The legacy
                     * W*B(T_rad) is NOT this number (for two equally diluted
                     * levels the pair-consistent source is B(T_rad), undiluted).
                     * NLTE-owned lines already have a matched pair and are left
                     * untouched; an NLTE-owned line with S<=0 has no positive
                     * source to emit with (and, per the adversarial identity
                     * S_l==0 <=> tau==1e-100, cannot pass fi_tau_cutoff at all) --
                     * it is counted as unresolved and emits nothing rather than
                     * being handed an invented one. */
                    unsigned char pc = fi_pair ? fi_pair[l] : FI_PAIR_UNKNOWN;
                    if (pc == FI_PAIR_NLTE) {
                        if (!(S > 0.0)) { S = 0.0; fi_n_orphan++; }
                    } else {
                        S = nlte_get_J_at_nu(nlte, shell, nu_l);
                        if (S > 0.0) fi_n_fused++; else fi_n_orphan++;
                    }
                }
                /* Decisive clamp test: thermalize thick-line source to B(T_e). */
                if (fi_clamp_sl > 0.0 && tau_sob > fi_clamp_sl)
                    S = planck_bnu(plasma->T_e[shell], nu_l);

                /* Line contribution: S * (1 - exp(-tau)) * exp(-tau_accumulated) */
                double one_minus_exp = (tau_sob > 500.0) ? 1.0 : (1.0 - exp(-tau_sob));
                int is_ige = (atom && atom->line_atomic_number &&
                              atom->line_atomic_number[l] >= 21);
                /* ledger: backlight luminosity this line removes (core rays) */
                if ((fi_ledger || fi_noblank) && p < r_phot) {
                    double Babs = fi_W_inner * planck_bnu(T_inner, nu_obs)
                                * exp(-tau_acc) * one_minus_exp;
                    double dL = Babs * p * dp_ray;
                    if (is_ige) ledg_ige += dL; else ledg_oth += dL;
                }
                I_nu += S * one_minus_exp * exp(-tau_acc);
                /* FOREST_NOBLANK: IGE forest does not blanket (skip tau_acc add) */
                if (!(fi_noblank && is_ige)) tau_acc += tau_sob;
                z_prev = z;
            }

            /* Final continuum segment from z_prev to inner boundary (env-gated) */
            if (fi_use_cont) {
                double z_inner_b = (p < r_phot) ? z_phot : -z_max_ejecta;
                if (z_prev > z_inner_b) {
                    double dz = z_prev - z_inner_b;
                    double z_mid = 0.5 * (z_prev + z_inner_b);
                    double r_mid = sqrt(p2 + z_mid * z_mid);
                    int shell_mid = -1;
                    for (int s = 0; s < n_shells; s++) {
                        if (r_mid >= geo->r_inner[s] && r_mid < geo->r_outer[s]) {
                            shell_mid = s; break;
                        }
                    }
                    if (shell_mid >= 0) {
                        double chi_cont = opacity->electron_density[shell_mid] * SIGMA_THOMSON;
                        double dtau_c = chi_cont * dz;
                        double S_cont = nlte_get_J_at_nu(nlte, shell_mid, nu_obs);
                        double oz = (dtau_c > 500.0) ? 1.0 : (1.0 - exp(-dtau_c));
                        I_nu   += S_cont * oz * exp(-tau_acc);
                        tau_acc += dtau_c;
                    }
                }
            }

            /* Inner boundary: dilute Planck at T_inner (env-gated W_inner) */
            if (p < r_phot) {
                I_nu += fi_W_inner * planck_bnu(T_inner, nu_obs) * exp(-tau_acc);
            }

            /* Integrate: L_nu += I_nu * 2*pi*p * dp */
            L_nu_integral += I_nu * p * dp_ray;
        }

        /* L_nu = 4*pi * integral(I_nu * 2*pi*p dp) = 8*pi^2 * sum */
        double L_nu = 8.0 * M_PI_VAL * M_PI_VAL * L_nu_integral;

        /* Convert L_nu [erg/s/Hz] to L_lambda [erg/s/cm]: L_lambda = L_nu * c / lambda^2 */
        spec_formal->flux[bin] = L_nu * C_SPEED_OF_LIGHT / (lambda_cm * lambda_cm);
        if (absL_ige) {
            double k = 8.0 * M_PI_VAL * M_PI_VAL;
            absL_ige[bin] = k * ledg_ige;
            absL_oth[bin] = k * ledg_oth;
        }
    }

    if (fi_ledger && absL_ige) {
        FILE *lf = fopen("lumina_fi_abs_ledger.csv", "w");
        if (lf) {
            fprintf(lf, "wavelength_angstrom,absL_IGE,absL_other\n");
            for (int b = 0; b < spec_formal->n_bins; b++)
                fprintf(lf, "%.4f,%.6e,%.6e\n",
                        spec_formal->wavelength[b], absL_ige[b], absL_oth[b]);
            fclose(lf);
            printf("  [FI] absorbed-energy ledger -> lumina_fi_abs_ledger.csv\n");
        }
    }
    free(absL_ige); free(absL_oth);

    /* [FORMAL-CONS] SCALAR ENERGY GATE on the yardstick itself (always on, no env).
     * The formal spectrum has never carried a conservation check, so a spectrum
     * whose band-integrated luminosity is orders of magnitude above what the model
     * injects has been read as physics. Emit the number every time the yardstick
     * is produced.
     *   integral  = trapezoid of L_lambda over the OUTPUT WAVELENGTH GRID.
     *               spec_formal->flux[b] is L_lambda in erg/s/cm (set above:
     *               L_nu * c / lambda_cm^2) and ->wavelength[b] is in Angstrom,
     *               so d(lambda) must be converted A -> cm (1e-8).
     *   L_inj     = the inner-boundary luminosity this very routine injects:
     *               the only source term in the integral is the p<r_phot
     *               backlight  I_nu += fi_W_inner * B_nu(T_inner) * exp(-tau),
     *               i.e. a (dilute) Planck surface of radius geo->r_inner[0] at
     *               T_inner, whose emergent luminosity is W * 4*pi*r^2*sigma*T^4.
     *               Identical expression to the transport driver's L_inner
     *               (lumina_cuda.cu "Phase 6 - Step 8" / main.c "Phase 5 - Step 4":
     *               4*pi*r_inner[0]^2*SIGMA_SB*T_inner^4), recomputed here from the
     *               same variables because it is not in scope at either call site.
     * R > 1 can only come from (a) scattering/line source terms that are not
     * energy-limited by the backlight, or (b) a normalization error; either way
     * the yardstick is not conserving and R is the magnitude. Bins outside the
     * output window are NOT counted -- the window is printed so the caller can
     * tell truncation from over-emission. */
    {
        double Lint = 0.0;
        for (int b = 0; b + 1 < spec_formal->n_bins; b++) {
            double dl_cm = (spec_formal->wavelength[b + 1] -
                            spec_formal->wavelength[b]) * 1.0e-8;
            Lint += 0.5 * (spec_formal->flux[b] + spec_formal->flux[b + 1]) * dl_cm;
        }
        double L_inj = fi_W_inner * 4.0 * M_PI_VAL * r_phot * r_phot *
                       SIGMA_SB * pow(T_inner, 4.0);
        /* [FORMAL-FIX] R3 census rides on the existing line (empty when the gate
         * is off => byte-identical output). */
        char fixsfx[128]; fixsfx[0] = '\0';
        if (fi_fix)
            snprintf(fixsfx, sizeof(fixsfx),
                     " [FFIX r3_fused=%ld r3_unresolved=%ld]",
                     fi_n_fused, fi_n_orphan);
        /* [FORMAL-CONS-WINDOW] gate LUMINA_FORMAL_CONS_WINDOW=1, default OFF.
         * SECOND ZERO POINT of this yardstick (B-register item 3): the numerator
         * Lint is windowed (it is the trapezoid over THIS grid only) while L_inj
         * above is the FULL Planck sigma*T^4, so the printed ratio has always
         * carried a fossil factor B = (in-window Planck fraction) that judgments
         * had to divide out by hand (A*B = 1.0524 at parity42; B = 0.985622 at
         * T_inner = 10020 K over 504.875-19995.125 A, certified in
         * validation/cmfgen_toy06_19p48d/analysis/metering_batch1 M3).
         * Under the gate the SAME injected luminosity is restricted to the SAME
         * window the numerator integrates over -- the window edges are read from
         * spec_formal->wavelength[] (the trapezoid runs from wavelength[0] to
         * wavelength[n_bins-1]), never hardcoded -- and BOTH ratios are printed:
         * the legacy one is untouched for continuity.
         *   L_inj_win = W*4pi*r_in^2 * pi*int_win B_lambda(T_inner) dlambda
         *             = L_inj * f_win,   f_win = planck_band_fraction(...)
         * (identity pi*int_0^inf B_lambda dlambda = sigma T^4; using L_inj*f_win
         * rather than rebuilding sigma from h,k,c keeps BOTH denominators on the
         * one constant, so the printed ratio of the two is exactly f_win.  The
         * two routes differ by 3.25e-11 relative -- that is SIGMA_SB vs
         * 2 pi^5 k^4/(15 c^2 h^3) built from this header's own h,k,c, measured
         * in impl_withParityX/selftest_fwin_precision.out.)
         * NO PHYSICS PATH READS THIS: spec_formal->flux[] is already final and
         * is not touched here.  Gate OFF => winsfx stays empty => byte-identical
         * line. */
        char winsfx[192]; winsfx[0] = '\0';
        { const char *_ew = getenv("LUMINA_FORMAL_CONS_WINDOW");
          if (_ew && atoi(_ew)) {
            double lam_a = spec_formal->wavelength[0];
            double lam_b = spec_formal->wavelength[spec_formal->n_bins - 1];
            double lam_lo = (lam_a < lam_b) ? lam_a : lam_b;
            double lam_hi = (lam_a < lam_b) ? lam_b : lam_a;
            double f_win = planck_band_fraction(T_inner, lam_lo, lam_hi);
            double L_inj_win = L_inj * f_win;
            snprintf(winsfx, sizeof(winsfx),
                     " [CONSWIN L/L_inj_win=%.6g L_inj_win=%.6e erg/s "
                     "f_win=%.6f win=%.4f-%.4f A]",
                     (L_inj_win > 0.0) ? Lint / L_inj_win : 0.0,
                     L_inj_win, f_win, lam_lo, lam_hi);
          } }
        printf("[FORMAL-CONS] integral L=%.6e erg/s = %.4g x L_inj "
               "(L_inj=%.6e erg/s = W%.3f*4pi*r_in^2*sigma*T_inner^4, "
               "r_in=%.4e cm, T_inner=%.2f K) window=%.1f-%.1f A nbins=%d%s%s\n",
               Lint, (L_inj > 0.0) ? Lint / L_inj : 0.0, L_inj, fi_W_inner,
               r_phot, T_inner,
               spec_formal->wavelength[0],
               spec_formal->wavelength[spec_formal->n_bins - 1],
               spec_formal->n_bins, fixsfx, winsfx);
        double L_total_in = L_inj + L_dep;
        printf("[FORMAL-CONS] L_total_in=%.6e erg/s ratio_total=%.6g "
               "(L_inj=%.6e + L_dep=%.6e erg/s)\n",
               L_total_in,
               (L_total_in > 0.0) ? Lint / L_total_in : 0.0,
               L_inj, L_dep);
        fflush(stdout);
    }
    free(fi_pair);

    printf("  Formal integral spectrum computed.\n");
}

/* ============================================================ */
/* CMF (comoving-frame) formal solver — paper-method line transfer */
/*                                                                  */
/* Local line absorption coefficient in homologous flow:            */
/*   chi_l(z) = tau_S / (sqrt(pi)*sigma_z) * exp(-((z-z_res)/sigma_z)^2) */
/* with sigma_z = v_Dopp * t_exp and z_res = c*t*(1 - nu_l/nu_obs).  */
/* Exact identity int chi_l dz = tau_S, so a single thin line        */
/* reproduces the Sobolev result; the new physics is line OVERLAP.   */
/* Per cell we deposit the integral-preserving fraction              */
/*   dtau_k = tau_S * 0.5*[erf((z_hi-z_res)/sigma) - erf((z_lo-z_res)/sigma)] */
/* so sub-cell-narrow lines drop their full tau into the nearest cell.*/
/* ============================================================ */
void compute_cmf_formal_spectrum(
    Geometry *geo, PlasmaState *plasma, OpacityState *opacity,
    AtomicData *atom, NLTEConfig *nlte, BFOpacity *bf, double T_inner,
    Spectrum *spec, int n_impact, int n_zstep, double v_turb_cms)
{
    int n_shells = geo->n_shells;
    int n_lines  = opacity->n_lines;
    double t_exp = geo->time_explosion;
    double ct    = C_SPEED_OF_LIGHT * t_exp;
    double r_phot  = geo->r_inner[0];
    double r_outer = geo->r_outer[n_shells - 1];
    double beta_max = r_outer / ct;
    double dp = r_outer / n_impact;

    /* Line element atomic numbers, indexed to match line_list_nu / tau_sobolev
     * (atom->line_* share the global nu-descending ordering of line_list_nu). */
    const int *line_Z = atom->line_atomic_number;

    /* Z -> amu lookup from the element table (fallback 2*Z) */
    double amu_by_Z[120];
    for (int z = 0; z < 120; z++) amu_by_Z[z] = 2.0 * (double)z;
    amu_by_Z[1] = 1.008;
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        if (z >= 0 && z < 120) amu_by_Z[z] = atom->element_mass_amu[e];
    }

    /* Ablation knobs (mirror the Sobolev formal integral plus CMF-specific) */
    const char *_env_cut  = getenv("LUMINA_CMF_TAU_CUTOFF");
    const char *_env_nsig = getenv("LUMINA_CMF_NSIGMA");
    const char *_env_idil = getenv("LUMINA_CMF_INNER_DILUTE");
    double cmf_tau_cutoff = _env_cut  ? atof(_env_cut)  : 1.0e-4;
    double cmf_nsigma     = _env_nsig ? atof(_env_nsig) : 4.0;
    double cmf_W_inner    = _env_idil ? atof(_env_idil) : 1.0;

    printf("\n=== CMF Formal Solver (comoving-frame line transfer) ===\n");
    printf("  Impact params: %d  z-cells/ray: %d  beta_max=%.4f\n",
           n_impact, n_zstep, beta_max);
    printf("  v_turb=%.2f km/s  tau cutoff=%.1e  profile half-width=%.1f sigma\n",
           v_turb_cms * 1.0e-5, cmf_tau_cutoff, cmf_nsigma);

    for (int b = 0; b < spec->n_bins; b++) spec->flux[b] = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        /* Thread-private per-ray accumulators */
        double *dtau  = (double *)malloc((size_t)n_zstep * sizeof(double));
        double *sdtau = (double *)malloc((size_t)n_zstep * sizeof(double));

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 8)
        #endif
        for (int bin = 0; bin < spec->n_bins; bin++) {
            double lambda_cm = spec->wavelength[bin] * 1.0e-8;
            double nu_obs = C_SPEED_OF_LIGHT / lambda_cm;

            double nu_line_min = nu_obs * (1.0 - beta_max);
            double nu_line_max = nu_obs * (1.0 + beta_max);
            int l_first = bsearch_descending_le(opacity->line_list_nu, n_lines, nu_line_max);
            int l_last  = bsearch_descending_ge(opacity->line_list_nu, n_lines, nu_line_min);

            double L_nu_integral = 0.0;

            for (int ip = 0; ip < n_impact; ip++) {
                double p = dp * (ip + 0.5);
                if (p >= r_outer) continue;
                double p2 = p * p;

                double z_max = sqrt(r_outer * r_outer - p2);
                double z_in  = (p < r_phot) ? sqrt(r_phot * r_phot - p2) : -z_max;
                double span  = z_max - z_in;
                if (span <= 0.0) continue;
                double dz = span / n_zstep;

                for (int k = 0; k < n_zstep; k++) { dtau[k] = 0.0; sdtau[k] = 0.0; }

                /* --- Continuum (e-scatter + bound-free) per cell --- */
                for (int k = 0; k < n_zstep; k++) {
                    double z_mid = z_in + (k + 0.5) * dz;
                    double r = sqrt(p2 + z_mid * z_mid);
                    if (r < r_phot || r > r_outer) continue;
                    int shell = -1;
                    for (int s = 0; s < n_shells; s++)
                        if (r >= geo->r_inner[s] && r < geo->r_outer[s]) { shell = s; break; }
                    if (shell < 0) continue;

                    double nu_cmf = nu_obs * (1.0 - z_mid / ct); /* comoving freq here */

                    double chi_es = opacity->electron_density[shell] * SIGMA_THOMSON;
                    double dtau_es = chi_es * dz;
                    double S_es;
                    S_es = (nlte != NULL && nlte->enabled)
                         ? nlte_get_J_at_nu(nlte, shell, nu_cmf) : 0.0;
                    dtau[k]  += dtau_es;
                    sdtau[k] += S_es * dtau_es;

                    if (bf != NULL && bf->enabled) {
                        double chi_bf = bf_get_chi(bf, shell, nu_cmf);
                        if (chi_bf > 0.0) {
                            double dtau_bf = chi_bf * dz;
                            double S_bf = planck_bnu(plasma->T_e[shell], nu_cmf); /* thermal */
                            dtau[k]  += dtau_bf;
                            sdtau[k] += S_bf * dtau_bf;
                        }
                    }
                }

                /* --- Lines: deposit Gaussian Doppler profiles onto the grid --- */
                for (int l = l_last; l >= l_first; l--) {
                    double nu_l = opacity->line_list_nu[l];
                    double z_res = ct * (1.0 - nu_l / nu_obs);

                    /* Resonance shell sets T_e and the line's Doppler width */
                    double r_res = sqrt(p2 + z_res * z_res);
                    int shell = -1;
                    for (int s = 0; s < n_shells; s++)
                        if (r_res >= geo->r_inner[s] && r_res < geo->r_outer[s]) { shell = s; break; }
                    if (shell < 0) continue;

                    double tau_S = opacity->tau_sobolev[l * n_shells + shell];
                    if (tau_S < cmf_tau_cutoff) continue;

                    int Z = (line_Z != NULL) ? line_Z[l] : 0;
                    double A = (Z > 0 && Z < 120) ? amu_by_Z[Z] : 2.0 * (Z > 0 ? Z : 28);
                    if (A <= 0.0) A = 56.0;
                    double v_th2 = 2.0 * K_BOLTZMANN * plasma->T_e[shell] / (A * AMU);
                    double v_dopp = sqrt(v_th2 + v_turb_cms * v_turb_cms);
                    double sigma_z = v_dopp * t_exp;
                    if (sigma_z <= 0.0) continue;
                    double inv_sigma = 1.0 / sigma_z;

                    /* Cells covered by +/- cmf_nsigma sigma around z_res */
                    double z_lo_w = z_res - cmf_nsigma * sigma_z;
                    double z_hi_w = z_res + cmf_nsigma * sigma_z;
                    if (z_hi_w <= z_in || z_lo_w >= z_max) continue;
                    int k0 = (int)floor((z_lo_w - z_in) / dz);
                    int k1 = (int)ceil ((z_hi_w - z_in) / dz);
                    if (k0 < 0) k0 = 0;
                    if (k1 > n_zstep) k1 = n_zstep;

                    /* Paper-method line source function: the NLTE two-level
                     * source S_l = (2hv^3/c^2)/(g_u n_l/(g_l n_u) - 1) computed
                     * from the NLTE level populations (carries fluorescence /
                     * thermalization). This replaces the old S_l = J (coherent
                     * scatter) which made CMF degenerate with Sobolev/scatter.
                     * Lines outside the NLTE network (S<=0) fall back to the
                     * dilute-LTE source W*B(T_rad) -- still not coherent scatter. */
                    double S_l = (opacity->line_source_S != NULL)
                               ? opacity->line_source_S[l * n_shells + shell] : 0.0;
                    if (S_l <= 0.0 && nlte != NULL && nlte->enabled)
                        S_l = nlte_get_J_at_nu(nlte, shell, nu_l);

                    double e_lo = erf((z_in + k0 * dz - z_res) * inv_sigma);
                    for (int k = k0; k < k1; k++) {
                        double e_hi = erf((z_in + (k + 1) * dz - z_res) * inv_sigma);
                        double frac = 0.5 * (e_hi - e_lo); /* fraction of profile in cell */
                        e_lo = e_hi;
                        if (frac <= 0.0) continue;
                        double dtau_l = tau_S * frac;
                        dtau[k]  += dtau_l;
                        sdtau[k] += S_l * dtau_l;
                    }
                }

                /* --- Formal solution, inner boundary -> observer (z increasing) --- */
                double I_nu = (p < r_phot)
                            ? cmf_W_inner * planck_bnu(T_inner, nu_obs) : 0.0;
                for (int k = 0; k < n_zstep; k++) {
                    double dt = dtau[k];
                    if (dt <= 0.0) continue;
                    double S_eff = sdtau[k] / dt;
                    double one_minus_exp = (dt > 500.0) ? 1.0 : (1.0 - exp(-dt));
                    I_nu = I_nu * (1.0 - one_minus_exp) + S_eff * one_minus_exp;
                }

                L_nu_integral += I_nu * p * dp;
            }

            double L_nu = 8.0 * M_PI_VAL * M_PI_VAL * L_nu_integral;
            spec->flux[bin] = L_nu * C_SPEED_OF_LIGHT / (lambda_cm * lambda_cm);
        }

        free(dtau);
        free(sdtau);
    }

    printf("  CMF formal spectrum computed.\n");
}

/* Spectrum binning: energy is luminosity in erg/s, output L_lambda in erg/s/cm */
void bin_escaped_packet(Spectrum *spec, double nu, double energy) {
    double lambda_A = C_SPEED_OF_LIGHT / nu * 1.0e8; /* frequency → wavelength (Å) */

    if (lambda_A < spec->lambda_min || lambda_A >= spec->lambda_max) {
        return;
    }

    double dlambda_A = (spec->lambda_max - spec->lambda_min) / spec->n_bins;
    int bin = (int)((lambda_A - spec->lambda_min) / dlambda_A);
    if (bin >= 0 && bin < spec->n_bins) {
        /* L_lambda [erg/s/cm] = luminosity [erg/s] / dlambda [cm] */
        double dlambda_cm = dlambda_A * 1.0e-8;
        spec->flux[bin] += energy / dlambda_cm;
    }
}

/* ============================================================ */
/* [MA-FATE] Macro-atom packet fate histogram                    */
/* Counts (entry_band, exit_band) for each macro-atom cascade.   */
/* ============================================================ */
static unsigned long long g_ma_fate_hist[MA_FATE_NBANDS * MA_FATE_NBANDS] = {0};

int macro_atom_fate_band_from_nu(double nu_comov) {
    if (nu_comov <= 0.0) return 7;
    double lam_A = (C_SPEED_OF_LIGHT / nu_comov) * 1.0e8;
    if (lam_A >= 1700.0  && lam_A <  3000.0) return 0;  /* UV-blanket  */
    if (lam_A >= 3000.0  && lam_A <  3300.0) return 1;  /* CaIIK-blue  */
    if (lam_A >= 3300.0  && lam_A <  3700.0) return 2;  /* UV-target   */
    if (lam_A >= 3700.0  && lam_A <  4400.0) return 3;  /* blue+fluor  */
    if (lam_A >= 4400.0  && lam_A <  5500.0) return 4;  /* green       */
    if (lam_A >= 5500.0  && lam_A <  7000.0) return 5;  /* red         */
    if (lam_A >= 7000.0  && lam_A < 10000.0) return 6;  /* NIR1        */
    return 7;                                            /* NIR2/far    */
}

void macro_atom_fate_record(double entry_nu_comov, double exit_nu_comov) {
    int eb = macro_atom_fate_band_from_nu(entry_nu_comov);
    int xb = macro_atom_fate_band_from_nu(exit_nu_comov);
    int idx = eb * MA_FATE_NBANDS + xb;
#pragma omp atomic
    g_ma_fate_hist[idx]++;
}

void macro_atom_fate_reset(void) {
    for (int i = 0; i < MA_FATE_NBANDS * MA_FATE_NBANDS; i++) g_ma_fate_hist[i] = 0;
}

void macro_atom_fate_add_counts(const unsigned long long add[MA_FATE_NBANDS * MA_FATE_NBANDS]) {
    for (int i = 0; i < MA_FATE_NBANDS * MA_FATE_NBANDS; i++) g_ma_fate_hist[i] += add[i];
}

void macro_atom_fate_print(const char *label) {
    static const char *band_name[MA_FATE_NBANDS] = {
        "UVblnk", "CaIIKb", "UVtgt ", "fluor ",
        "green ", "red   ", "NIR1  ", "NIR2  "
    };
    unsigned long long row_tot[MA_FATE_NBANDS] = {0};
    unsigned long long col_tot[MA_FATE_NBANDS] = {0};
    unsigned long long grand = 0;
    for (int e = 0; e < MA_FATE_NBANDS; e++) {
        for (int x = 0; x < MA_FATE_NBANDS; x++) {
            unsigned long long n = g_ma_fate_hist[e * MA_FATE_NBANDS + x];
            row_tot[e] += n;
            col_tot[x] += n;
            grand     += n;
        }
    }
    printf("\n[MA-FATE] Macro-atom packet fate histogram (%s)\n",
           label ? label : "");
    printf("  Bands: UVblnk=1700-3000 CaIIKb=3000-3300 UVtgt=3300-3700 fluor=3700-4400\n"
           "         green =4400-5500 red   =5500-7000 NIR1 =7000-10000 NIR2 =>10000\n");
    if (grand == 0) {
        printf("  (no macro-atom interactions recorded)\n");
        return;
    }
    printf("              | exit->");
    for (int x = 0; x < MA_FATE_NBANDS; x++) printf("    %s ", band_name[x]);
    printf("|    row tot\n");
    for (int e = 0; e < MA_FATE_NBANDS; e++) {
        printf("  entry %s |", band_name[e]);
        for (int x = 0; x < MA_FATE_NBANDS; x++) {
            unsigned long long n = g_ma_fate_hist[e * MA_FATE_NBANDS + x];
            double pct = row_tot[e] > 0 ? 100.0 * (double)n / (double)row_tot[e] : 0.0;
            printf(" %8llu(%4.1f%%)", n, pct);
        }
        printf(" | %10llu\n", row_tot[e]);
    }
    printf("  col tot     |       ");
    for (int x = 0; x < MA_FATE_NBANDS; x++) printf(" %14llu", col_tot[x]);
    printf(" | %10llu\n", grand);
    /* Diagnostic: UV-blanket-entry redistribution. The key physics question:
     * does Fe II UV-line absorbed energy emerge at UVtgt+fluor (3300-4400, the
     * Mazzali-Lucy fluorescence peaks observed in HST 2011fe), or does it
     * preferentially leak to NIR via cascade through low-lying levels? */
    unsigned long long uv_row = row_tot[0];
    if (uv_row > 0) {
        double pct_self = 100.0 * (double)g_ma_fate_hist[0 * MA_FATE_NBANDS + 0] / (double)uv_row;
        double pct_uvtgt = 100.0 * (double)g_ma_fate_hist[0 * MA_FATE_NBANDS + 2] / (double)uv_row;
        double pct_fluor = 100.0 * (double)g_ma_fate_hist[0 * MA_FATE_NBANDS + 3] / (double)uv_row;
        double pct_opt   = 100.0 * (double)(g_ma_fate_hist[0 * MA_FATE_NBANDS + 4]
                                          + g_ma_fate_hist[0 * MA_FATE_NBANDS + 5]) / (double)uv_row;
        double pct_nir   = 100.0 * (double)(g_ma_fate_hist[0 * MA_FATE_NBANDS + 6]
                                          + g_ma_fate_hist[0 * MA_FATE_NBANDS + 7]) / (double)uv_row;
        printf("  UVblnk-entry fate: -> UVblnk %.1f%% | UVtgt %.1f%% | fluor %.1f%% |"
               " opt(grn+red) %.1f%% | NIR %.1f%%\n",
               pct_self, pct_uvtgt, pct_fluor, pct_opt, pct_nir);
        printf("  (HST 2011fe physics: UVtgt+fluor should dominate; large NIR fraction\n"
               "   means cascade terminates at low-lying levels instead of emitting at 3300-4400 A.)\n");
    }
}

/* ============================================================ */
/* [H3] Per-(Z, ion, entry_band, exit_band) attribution           */
/* ============================================================ */
const int MA_FATE_Z_LIST[MA_FATE_NZ] = {
    6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28
};
static const char *MA_FATE_Z_NAME[MA_FATE_NZ] = {
    "C","O","Mg","Al","Si","S","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni"
};
static unsigned long long g_ma_fate_zihist[MA_FATE_ZI_LEN] = {0};

void macro_atom_fate_zi_reset(void) {
    for (int i = 0; i < MA_FATE_ZI_LEN; i++) g_ma_fate_zihist[i] = 0;
}
void macro_atom_fate_zi_add_counts(const unsigned long long add[MA_FATE_ZI_LEN]) {
    for (int i = 0; i < MA_FATE_ZI_LEN; i++) g_ma_fate_zihist[i] += add[i];
}
void macro_atom_fate_zi_dump_csv(const char *path, const char *label) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[MA-FATE-ZI] could not open %s for write\n", path);
        return;
    }
    fprintf(f, "# label=%s\n", label ? label : "");
    fprintf(f, "# bands: 0=UVblnk[1700,3000) 1=CaIIKb[3000,3300) "
               "2=UVtgt[3300,3700) 3=fluor[3700,4400) 4=green[4400,5500) "
               "5=red[5500,7000) 6=NIR1[7000,10000) 7=NIR2[>=10000)\n");
    fprintf(f, "Z,Z_name,ion,entry_band,exit_band,count\n");
    for (int zi = 0; zi < MA_FATE_NZ; zi++) {
        for (int io = 0; io < MA_FATE_NION; io++) {
            for (int eb = 0; eb < MA_FATE_NBANDS; eb++) {
                for (int xb = 0; xb < MA_FATE_NBANDS; xb++) {
                    int idx = ((zi*MA_FATE_NION + io)*MA_FATE_NBANDS + eb)
                              *MA_FATE_NBANDS + xb;
                    unsigned long long n = g_ma_fate_zihist[idx];
                    if (n == 0) continue;
                    fprintf(f, "%d,%s,%d,%d,%d,%llu\n",
                            MA_FATE_Z_LIST[zi], MA_FATE_Z_NAME[zi],
                            io, eb, xb, n);
                }
            }
        }
    }
    fclose(f);
    printf("[MA-FATE-ZI] CSV written to %s\n", path);
}

/* ============================================================ */
/* [MA-CYCLE] Macro-atom internal cycle count histogram          */
/* ma_iter is the number of branching iterations per packet      */
/* before it picks an emission (BB or BF) transition.            */
/* ============================================================ */
static unsigned long long g_ma_cycle_hist[MA_CYCLE_BINS] = {0};

void macro_atom_cycle_record(int n_cycles) {
    int b = n_cycles;
    if (b < 0) b = 0;
    if (b >= MA_CYCLE_BINS) b = MA_CYCLE_BINS - 1;
#pragma omp atomic
    g_ma_cycle_hist[b]++;
}

void macro_atom_cycle_reset(void) {
    for (int i = 0; i < MA_CYCLE_BINS; i++) g_ma_cycle_hist[i] = 0;
}

void macro_atom_cycle_add_counts(const unsigned long long add[MA_CYCLE_BINS]) {
    for (int i = 0; i < MA_CYCLE_BINS; i++) g_ma_cycle_hist[i] += add[i];
}

void macro_atom_cycle_print(const char *label) {
    unsigned long long total = 0;
    for (int i = 0; i < MA_CYCLE_BINS; i++) total += g_ma_cycle_hist[i];
    if (total == 0) {
        printf("\n[MA-CYCLE] %s: no data\n", label);
        return;
    }
    double mean = 0.0;
    unsigned long long cumul = 0;
    int p50 = -1, p90 = -1, p99 = -1;
    for (int i = 0; i < MA_CYCLE_BINS; i++) {
        mean  += (double)i * (double)g_ma_cycle_hist[i];
        cumul += g_ma_cycle_hist[i];
        if (p50 < 0 && cumul * 2 >= total) p50 = i;
        if (p90 < 0 && cumul * 10 >= total * 9) p90 = i;
        if (p99 < 0 && cumul * 100 >= total * 99) p99 = i;
    }
    mean /= (double)total;
    unsigned long long cap_hits = g_ma_cycle_hist[MA_CYCLE_BINS - 1];

    printf("\n[MA-CYCLE] %s\n", label);
    printf("  total=%llu  mean=%.2f  p50=%d  p90=%d  p99=%d  cap@%d=%llu (%.4f%%)\n",
           total, mean, p50, p90, p99, MA_CYCLE_BINS - 1,
           cap_hits, 100.0 * (double)cap_hits / (double)total);
    int bins[] = {0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
    int n_bins = (int)(sizeof(bins) / sizeof(bins[0]));
    printf("  cycle range  count           fraction\n");
    for (int b = 0; b < n_bins - 1; b++) {
        unsigned long long sub = 0;
        for (int i = bins[b]; i < bins[b+1]; i++) sub += g_ma_cycle_hist[i];
        printf("  [%4d,%4d)  %14llu  %6.2f%%\n",
               bins[b], bins[b+1], sub,
               100.0 * (double)sub / (double)total);
    }
    printf("  [%4d    ]  %14llu  %6.2f%%\n",
           MA_CYCLE_BINS - 1, cap_hits,
           100.0 * (double)cap_hits / (double)total);
}

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */
