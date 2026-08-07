/* Wave-3 Stage-2A element-wide NLTE reference path.
 *
 * This file deliberately contains only the gate, matrix orchestration,
 * conditioning/solve diagnostics, dumps, and commit.  Physical rate arithmetic
 * remains in nlte_assemble_rate_matrix(); small capture hooks in that producer
 * place the already-computed rates into the seven element-wide channel planes.
 * The default/explicit-OFF path reaches none of this file's allocation, dump,
 * banner, counter, or random/permutation code.
 */
#include "lumina.h"
#include "bf_rate_jnu.h" /* A2-05 canonical-view bf rate result type */

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EW(A,n,i,j) ((A)[(size_t)(j) * (size_t)(n) + (size_t)(i)])

/* Shared implementations live in lumina_plasma.c; keeping the declarations
 * here avoids widening the public header for Wave-3.2-only plumbing. */
extern int nlte_bf_field_source(const NLTEConfig *nlte, double T_e, double nu,
                                double J_default, int gpu_lookup_available,
                                int *use_gpu_lookup, int *gpu_field_bypassed,
                                double *J_selected);
extern int nlte_bf_collisional_enabled(void);
extern double nlte_bf_kramers_sigma0(int Z, int stage);
extern unsigned long nlte_bf_gpu_field_bypass_count(void);
/* A2-05: canonical-view bf photoionization rate (owner in lumina_plasma.c). */
extern int nlte_bf_gamma_canonical(NLTEConfig *nlte, int shell,
                                   const double *sigma_row, double sigma_0,
                                   double nu_thresh, BfRateResult *out);
extern int nlte_precompute_within_sl_frac_checked(
    NLTEConfig *, AtomicData *, PlasmaState *, int);
extern int nlte_build_projection(NLTEConfig *nlte, AtomicData *atom,
                                 OpacityState *opacity, int n_shells,
                                 const int *target_Z, const int *target_ion,
                                 int n_targets, int super_requested,
                                 int *n_lines_mapped);

static const char *const ew_channel_name[NLTE_EW_N_CHANNELS] = {
    "rad_bb", "coll_bb", "nt_bb", "rad_bf", "coll_bf", "nt_bf",
    "autoion_DR"
};

typedef struct {
    int parsed, requested, enabled, shell, commit, dump;
    unsigned char zmask[100];
    char dump_dir[512];
    char error[192];
} EWGate;

static EWGate ew_gate;
static int ew_dump_failed;

/* Complete a stdio artifact transaction.  A buffered fprintf() can appear to
 * succeed and report ENOSPC only at flush/close, so every EW artifact uses this
 * common completion check before an adoption commit is allowed. */
static int ew_finish_file(FILE *fp, const char *path) {
    int saved_errno = 0;
    if (fflush(fp) != 0) saved_errno = errno ? errno : EIO;
    if (ferror(fp) && !saved_errno) saved_errno = errno ? errno : EIO;
    if (fclose(fp) != 0 && !saved_errno) saved_errno = errno ? errno : EIO;
    if (saved_errno) {
        fprintf(stderr, "[EW][DUMP-FAIL] write/close %s: %s\n",
                path, strerror(saved_errno));
        return -1;
    }
    return 0;
}

static void ew_close_dump(FILE *fp, const char *path) {
    if (ew_finish_file(fp, path) != 0) ew_dump_failed = 1;
}

/* CPU-only fixture entry; it exercises the exact production completion check
 * with /dev/full as the buffered-write negative control. */
int nlte_ew_test_dump_io(const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    (void)fprintf(fp, "wave32-ew-artifact\n");
    return ew_finish_file(fp, path);
}

static int ew_env_true(const char *name) {
    const char *v = getenv(name);
    return v && atoi(v) != 0;
}

static void ew_parse_gate(void) {
    if (ew_gate.parsed) return;
    memset(&ew_gate, 0, sizeof(ew_gate));
    ew_gate.parsed = 1;
    ew_gate.shell = -1;
    strcpy(ew_gate.dump_dir, ".");

    const char *master = getenv("LUMINA_NLTE_ELEMENT_WIDE");
    if (!master || atoi(master) == 0) return; /* unset and explicit 0: same branch */
    ew_gate.requested = 1;
    if (atoi(master) != 1) {
        snprintf(ew_gate.error, sizeof(ew_gate.error), "ELEMENT_WIDE must be 0 or 1");
        goto invalid;
    }

    const char *zs = getenv("LUMINA_NLTE_ELEMENT_WIDE_Z");
    const char *ss = getenv("LUMINA_NLTE_ELEMENT_WIDE_SHELL");
    if (!zs || !*zs || !ss || !*ss) {
        snprintf(ew_gate.error, sizeof(ew_gate.error),
                 "explicit Z list and shell are required");
        goto invalid;
    }
    {
        const char *p = zs;
        int count = 0;
        while (*p) {
            char *end = NULL;
            long z = strtol(p, &end, 10);
            if (end == p || (z != 16 && z != 26)) {
                snprintf(ew_gate.error, sizeof(ew_gate.error),
                         "Z allow-list is restricted to 16,26");
                goto invalid;
            }
            ew_gate.zmask[z] = 1;
            count++;
            p = end;
            while (*p == ' ' || *p == '\t') p++;
            if (!*p) break;
            if (*p++ != ',') {
                snprintf(ew_gate.error, sizeof(ew_gate.error), "malformed Z allow-list");
                goto invalid;
            }
            while (*p == ' ' || *p == '\t') p++;
        }
        if (!count) {
            snprintf(ew_gate.error, sizeof(ew_gate.error), "empty Z allow-list");
            goto invalid;
        }
    }
    {
        char *end = NULL;
        long shell = strtol(ss, &end, 10);
        if (end == ss || *end || (shell != 0 && shell != 8 && shell != 20)) {
            snprintf(ew_gate.error, sizeof(ew_gate.error),
                     "Stage-2A shell must be one of 0,8,20");
            goto invalid;
        }
        ew_gate.shell = (int)shell;
    }
    ew_gate.commit = ew_env_true("LUMINA_NLTE_ELEMENT_WIDE_COMMIT");
    ew_gate.dump = ew_env_true("LUMINA_NLTE_ELEMENT_WIDE_DUMP");
    if (ew_env_true("LUMINA_NLTE_STAGE4")) {
        snprintf(ew_gate.error, sizeof(ew_gate.error),
                 "C65 conflict: LUMINA_NLTE_STAGE4 must be off in the EW lane");
        goto invalid;
    }
    if (ew_gate.commit && ew_gate.shell != 8) {
        snprintf(ew_gate.error, sizeof(ew_gate.error),
                 "commit is restricted to the s8 pilot; s0/s20 are shadow-only");
        goto invalid;
    }
    {
        const char *d = getenv("LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR");
        if (d && *d) snprintf(ew_gate.dump_dir, sizeof(ew_gate.dump_dir), "%s", d);
    }
    ew_gate.enabled = 1;
    fprintf(stderr,
            "[EW] Stage-2A CPU path armed: Z=%s%s shell=%d commit=%d dump=%d\n",
            ew_gate.zmask[16] ? "16" : "",
            ew_gate.zmask[26] ? (ew_gate.zmask[16] ? ",26" : "26") : "",
            ew_gate.shell, ew_gate.commit, ew_gate.dump);
    return;

invalid:
    fprintf(stderr, "[EW][CONFIG-FAIL] %s; element-wide path disabled\n",
            ew_gate.error);
}

int nlte_element_wide_enabled(void) {
    ew_parse_gate();
    return ew_gate.enabled;
}

int nlte_element_wide_config_status(void) {
    ew_parse_gate();
    return ew_gate.requested && !ew_gate.enabled ? -1 : 0;
}

int nlte_element_wide_layout_enabled(void) {
    ew_parse_gate();
    if (!ew_gate.enabled) return 0;
#ifdef LUMINA_FROZEN_ORACLE
    /* The frozen pass-through is resolved before nlte_init().  It requests a
     * real commit-capable layout even when the production commit knob is left
     * at zero, while ordinary frozen shadow runs keep the base layout. */
    {
        const char *v = getenv("LUMINA_EW_FROZEN_COMMIT");
        if (v && strcmp(v, "1") == 0) return 1;
    }
#endif
    return ew_gate.commit;
}

int nlte_element_wide_matches(int Z, int shell) {
    ew_parse_gate();
    if (!ew_gate.enabled || shell != ew_gate.shell || Z < 0 || Z >= 100 ||
        !ew_gate.zmask[Z]) return 0;
    if (shell == 0 && Z != 26) return 0;
    if (shell == 20 && Z != 16) return 0;
    return 1;
}

int nlte_element_wide_commit_enabled(void) {
    ew_parse_gate();
    return ew_gate.enabled && ew_gate.commit;
}

typedef struct {
    int active, N_pair, N_wide, wide_base, pair_super_start, n_lo_super;
    int shell, n_shells, target_expected, target_mapped, target_fail, bad_rate;
    int channel_events[NLTE_EW_N_CHANNELS];
    int kramers_fallback, continuum_deleted, target_fallback;
    int nstar_cap, nonfinite_guard;
    int bf_estimator_bins, bf_pref_j_bins, bf_jeqb_bins;
    int skip_lo_bb;
    NLTEConfig *nlte;
    AtomicData *atom;
    double *plane[NLTE_EW_N_CHANNELS];
    double *expected_outflow[NLTE_EW_N_CHANNELS];
} EWCapture;

static EWCapture ew_cap;

/* Every counter reachable from the shell-parallel assembler enters through
 * this primitive.  Matrix/ledger arithmetic has separate ownership; this is
 * deliberately limited to integer telemetry and topology counters. */
static inline void ew_atomic_inc_int(int *counter) {
#ifdef _OPENMP
#pragma omp atomic update
#endif
    (*counter)++;
}

static int ew_find_slot(const NLTEConfig *n, int Z, int stage);
static double ew_find_chi(const AtomicData *a, int Z, int stage);

#ifdef WAVE32_COUNTER_SELFTEST
enum { EW_TEST_CAPTURE_COUNTERS = 19 };

void nlte_ew_test_capture_counters_reset(void) {
    memset(&ew_cap, 0, sizeof(ew_cap));
}

void nlte_ew_test_capture_counters_bump(void) {
    ew_atomic_inc_int(&ew_cap.target_expected);
    ew_atomic_inc_int(&ew_cap.target_mapped);
    ew_atomic_inc_int(&ew_cap.target_fail);
    ew_atomic_inc_int(&ew_cap.bad_rate);
    for (int c = 0; c < NLTE_EW_N_CHANNELS; c++)
        ew_atomic_inc_int(&ew_cap.channel_events[c]);
    ew_atomic_inc_int(&ew_cap.kramers_fallback);
    ew_atomic_inc_int(&ew_cap.continuum_deleted);
    ew_atomic_inc_int(&ew_cap.target_fallback);
    ew_atomic_inc_int(&ew_cap.nstar_cap);
    ew_atomic_inc_int(&ew_cap.nonfinite_guard);
    ew_atomic_inc_int(&ew_cap.bf_estimator_bins);
    ew_atomic_inc_int(&ew_cap.bf_pref_j_bins);
    ew_atomic_inc_int(&ew_cap.bf_jeqb_bins);
}

void nlte_ew_test_capture_counters_snapshot(int out[EW_TEST_CAPTURE_COUNTERS]) {
    int k = 0;
    out[k++] = ew_cap.target_expected;
    out[k++] = ew_cap.target_mapped;
    out[k++] = ew_cap.target_fail;
    out[k++] = ew_cap.bad_rate;
    for (int c = 0; c < NLTE_EW_N_CHANNELS; c++)
        out[k++] = ew_cap.channel_events[c];
    out[k++] = ew_cap.kramers_fallback;
    out[k++] = ew_cap.continuum_deleted;
    out[k++] = ew_cap.target_fallback;
    out[k++] = ew_cap.nstar_cap;
    out[k++] = ew_cap.nonfinite_guard;
    out[k++] = ew_cap.bf_estimator_bins;
    out[k++] = ew_cap.bf_pref_j_bins;
    out[k++] = ew_cap.bf_jeqb_bins;
}
#endif

int nlte_ew_capture_active(void) {
    return ew_cap.active;
}

static void ew_capture_begin(NLTEConfig *nlte, AtomicData *atom, int shell,
                             int n_shells,
                             int ion_lo, int N_pair, int N_wide, int wide_base,
                             int skip_lo_bb,
                             double *plane[NLTE_EW_N_CHANNELS],
                             double *expected_outflow[NLTE_EW_N_CHANNELS]) {
    memset(&ew_cap, 0, sizeof(ew_cap));
    ew_cap.active = 1;
    ew_cap.N_pair = N_pair;
    ew_cap.N_wide = N_wide;
    ew_cap.wide_base = wide_base;
    ew_cap.pair_super_start = nlte->nlte_ion_super_offset[ion_lo];
    ew_cap.n_lo_super = nlte->nlte_ion_super_offset[ion_lo + 1] -
                        ew_cap.pair_super_start;
    ew_cap.shell = shell;
    ew_cap.n_shells = n_shells;
    ew_cap.skip_lo_bb = skip_lo_bb;
    ew_cap.nlte = nlte;
    ew_cap.atom = atom;
    for (int c = 0; c < NLTE_EW_N_CHANNELS; c++) {
        ew_cap.plane[c] = plane[c];
        ew_cap.expected_outflow[c] = expected_outflow[c];
    }
}

typedef struct {
    int channel_events[NLTE_EW_N_CHANNELS];
    int kramers_fallback, continuum_deleted, target_fallback;
    int nstar_cap, nonfinite_guard;
    int bf_estimator_bins, bf_pref_j_bins, bf_jeqb_bins;
    int candidate_assembler_calls, candidate_pair_owner_calls;
    int save_restore_calls;
    int per_ion_pin_calls, topstage_IV_calls;
} EWFireCounts;

typedef struct {
    unsigned long save_restore_calls;
    unsigned long per_ion_pin_calls;
    unsigned long topstage_IV_calls;
} EWRuntimeCounts;
static EWRuntimeCounts ew_runtime_counts;

void nlte_ew_note_save_restore_call(void) {
    if (nlte_element_wide_enabled()) {
#ifdef _OPENMP
#pragma omp atomic update
#endif
        ew_runtime_counts.save_restore_calls++;
    }
}
void nlte_ew_note_per_ion_pin_call(void) {
    if (nlte_element_wide_enabled()) {
#ifdef _OPENMP
#pragma omp atomic update
#endif
        ew_runtime_counts.per_ion_pin_calls++;
    }
}
void nlte_ew_note_topstage_IV_call(void) {
    if (nlte_element_wide_enabled()) {
#ifdef _OPENMP
#pragma omp atomic update
#endif
        ew_runtime_counts.topstage_IV_calls++;
    }
}
void nlte_ew_runtime_counts_snapshot(unsigned long out[3]) {
    if (!out) return;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    out[0] = ew_runtime_counts.save_restore_calls;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    out[1] = ew_runtime_counts.per_ion_pin_calls;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    out[2] = ew_runtime_counts.topstage_IV_calls;
}

/* Publish after nlte_solve_all has traversed the real pair-owner paths.  The
 * earlier candidate-window values are intentionally not promoted: save/restore
 * lives outside that window and topstage is excluded while capture is active. */
int nlte_ew_publish_runtime_counts(const NLTEConfig *nlte) {
    ew_parse_gate();
    if (!nlte || !ew_gate.enabled || !ew_gate.dump) return 0;
    int failed = 0;
    unsigned long counts[3];
    nlte_ew_runtime_counts_snapshot(counts);
    const int zlist[2] = {16, 26};
    for (int zi = 0; zi < 2; zi++) {
        int Z = zlist[zi];
        if (!nlte_element_wide_matches(Z, ew_gate.shell)) continue;
        char path[768];
        snprintf(path, sizeof(path),
                 "%s/lumina_ew_iter%04d_z%02d_s%03d_manifest.csv",
                 ew_gate.dump_dir, nlte->current_iter, Z, ew_gate.shell);
        FILE *fp = fopen(path, "a");
        if (!fp) {
            fprintf(stderr, "[EW][DUMP-FAIL] runtime append %s: %s\n",
                    path, strerror(errno));
            failed = 1;
            continue;
        }
        fprintf(fp,
                "runtime_final_save_restore_calls,%lu\n"
                "runtime_final_per_ion_pin_calls,%lu\n"
                "runtime_final_topstage_IV_calls,%lu\n",
                counts[0], counts[1], counts[2]);
        if (ew_finish_file(fp, path) != 0) failed = 1;
    }
    return failed ? -1 : 0;
}

static void ew_capture_end(int *expected, int *mapped, int *fail, int *bad_rate,
                           EWFireCounts *fire) {
    if (expected) *expected += ew_cap.target_expected;
    if (mapped) *mapped += ew_cap.target_mapped;
    if (fail) *fail += ew_cap.target_fail;
    if (bad_rate) *bad_rate += ew_cap.bad_rate;
    if (fire) {
        for (int c = 0; c < NLTE_EW_N_CHANNELS; c++)
            fire->channel_events[c] += ew_cap.channel_events[c];
        fire->kramers_fallback += ew_cap.kramers_fallback;
        fire->continuum_deleted += ew_cap.continuum_deleted;
        fire->target_fallback += ew_cap.target_fallback;
        fire->nstar_cap += ew_cap.nstar_cap;
        fire->nonfinite_guard += ew_cap.nonfinite_guard;
        fire->bf_estimator_bins += ew_cap.bf_estimator_bins;
        fire->bf_pref_j_bins += ew_cap.bf_pref_j_bins;
        fire->bf_jeqb_bins += ew_cap.bf_jeqb_bins;
        fire->candidate_assembler_calls++;
    }
    memset(&ew_cap, 0, sizeof(ew_cap));
}

void nlte_ew_capture_transition(int channel, int target, int source,
                                double rate) {
    if (!ew_cap.active) return;
    if (channel < 0 || channel >= NLTE_EW_N_CHANNELS ||
        target < 0 || target >= ew_cap.N_pair ||
        source < 0 || source >= ew_cap.N_pair ||
        !(rate >= 0.0) || !isfinite(rate)) {
        ew_atomic_inc_int(&ew_cap.bad_rate);
        return;
    }
    if (rate == 0.0 || target == source) return;
    if (ew_cap.skip_lo_bb &&
        (channel == NLTE_EW_RAD_BB || channel == NLTE_EW_COLL_BB) &&
        target < ew_cap.n_lo_super && source < ew_cap.n_lo_super) return;
    int i = ew_cap.wide_base + target;
    int j = ew_cap.wide_base + source;
    if (i < 0 || j < 0 || i >= ew_cap.N_wide || j >= ew_cap.N_wide) {
        ew_atomic_inc_int(&ew_cap.bad_rate);
        return;
    }
    /* One synchronization domain owns the matrix placement and its independent
     * debit ledger update.  The capture is normally entered by one pair-owner,
     * but the critical region makes that ownership executable if callers ever
     * reach this file-static capture from multiple OpenMP workers. */
#ifdef _OPENMP
#pragma omp critical(lumina_ew_capture_accumulate)
#endif
    {
        EW(ew_cap.plane[channel], ew_cap.N_wide, i, j) += rate;
        EW(ew_cap.plane[channel], ew_cap.N_wide, j, j) -= rate;
        /* D6 reads this ledger only after capture ends and checks both matrix
         * inflow and diagonal debit independently against it. */
        ew_cap.expected_outflow[channel][j] += rate;
    }
    ew_atomic_inc_int(&ew_cap.channel_events[channel]);
}

#ifdef WAVE32_COUNTER_SELFTEST
static double ew_test_plane[NLTE_EW_N_CHANNELS][4];
static double ew_test_expected_outflow[NLTE_EW_N_CHANNELS][2];

void nlte_ew_test_expected_outflow_reset(void) {
    memset(&ew_cap, 0, sizeof(ew_cap));
    memset(ew_test_plane, 0, sizeof(ew_test_plane));
    memset(ew_test_expected_outflow, 0, sizeof(ew_test_expected_outflow));
    ew_cap.active = 1;
    ew_cap.N_pair = 2;
    ew_cap.N_wide = 2;
    for (int c = 0; c < NLTE_EW_N_CHANNELS; c++) {
        ew_cap.plane[c] = ew_test_plane[c];
        ew_cap.expected_outflow[c] = ew_test_expected_outflow[c];
    }
}

void nlte_ew_test_expected_outflow_bump(void) {
    nlte_ew_capture_transition(NLTE_EW_RAD_BB, 1, 0, 1.0);
}

void nlte_ew_test_expected_outflow_seed_invalid(void) {
    nlte_ew_capture_transition(NLTE_EW_RAD_BB, 1, 0, NAN);
}

void nlte_ew_test_expected_outflow_snapshot(double out[3], int *bad_rate) {
    out[0] = ew_test_expected_outflow[NLTE_EW_RAD_BB][0];
    out[1] = EW(ew_test_plane[NLTE_EW_RAD_BB], 2, 1, 0);
    out[2] = EW(ew_test_plane[NLTE_EW_RAD_BB], 2, 0, 0);
    if (bad_rate) *bad_rate = ew_cap.bad_rate;
}
#endif

/* ARTIS nltepop.cc:581-615 equivalent topology: rates are evaluated inside
 * the target loop.  In particular, the target excitation energy and g enter
 * the threshold and inverse rate; a lower-level rate is never computed once
 * and then merely split among upper targets. */
void nlte_ew_capture_bf_target_rates(int lower_global, int lower_sl,
                                      double lower_fraction,
                                      double T_e, double n_e) {
    if (!ew_cap.active) return;
    AtomicData *a = ew_cap.atom;
    NLTEConfig *n = ew_cap.nlte;
    if (lower_global < 0 || lower_global >= a->n_levels) {
        ew_atomic_inc_int(&ew_cap.bad_rate);
        return;
    }
    int has_sigma = a->cmfgen_loaded &&
                    a->cmfgen_n_freq_bins == n->n_freq_bins &&
                    a->cmfgen_has_sigma &&
                    a->cmfgen_sigma_bf &&
                    a->cmfgen_has_sigma[lower_global];
    int kramers = !has_sigma;
    if (kramers) {
        ew_atomic_inc_int(&ew_cap.kramers_fallback);
    }
    if (!kramers &&
        (!a->ma_rr_loaded || !a->ma_rr_target_offset || !a->ma_rr_targets ||
         !a->ma_rr_probability)) {
        ew_atomic_inc_int(&ew_cap.target_expected);
        ew_atomic_inc_int(&ew_cap.target_fail);
        return;
    }
    int begin = kramers ? 0 : a->ma_rr_target_offset[lower_global];
    int end = kramers ? 1 : a->ma_rr_target_offset[lower_global + 1];
    if (!kramers &&
        (begin < 0 || end < begin || end > a->ma_rr_n_routes || begin == end)) {
        ew_atomic_inc_int(&ew_cap.target_expected);
        ew_atomic_inc_int(&ew_cap.target_fail);
        return;
    }
    double chi_eV = ew_find_chi(a, a->level_Z[lower_global],
                                a->level_ion[lower_global]);
    const double *sigma_row = has_sigma ? &a->cmfgen_sigma_bf[
        (size_t)lower_global * (size_t)a->cmfgen_n_freq_bins] : NULL;
    double sigma_0 = kramers ? nlte_bf_kramers_sigma0(
        a->level_Z[lower_global], a->level_ion[lower_global]) : 0.0;
    int field_source = nlte_bf_field_source(
        n, T_e, 0.0, 0.0, 0, NULL, NULL, NULL);
    double psum = 0.0;
    for (int r = begin; r < end; r++) {
        ew_atomic_inc_int(&ew_cap.target_expected);
        int upper_global = kramers ? n->super_anchor_global[
            ew_cap.pair_super_start + ew_cap.n_lo_super] : a->ma_rr_targets[r];
        double p = kramers ? 1.0 : a->ma_rr_probability[r];
        if (upper_global < 0 || upper_global >= a->n_levels ||
            !(p > 0.0) || !isfinite(p) ||
            a->level_Z[upper_global] != a->level_Z[lower_global] ||
            a->level_ion[upper_global] != a->level_ion[lower_global] + 1) {
            ew_atomic_inc_int(&ew_cap.target_fail);
            if (kramers) {
                ew_atomic_inc_int(&ew_cap.continuum_deleted);
            }
            continue;
        }
        int ng = n->global_to_nlte_level[upper_global];
        if (ng < 0) {
            ew_atomic_inc_int(&ew_cap.target_fail);
            continue;
        }
        int upper_sl = n->fl_to_super[ng] - ew_cap.pair_super_start;
        if (upper_sl < 0 || upper_sl >= ew_cap.N_pair) {
            ew_atomic_inc_int(&ew_cap.target_fail);
            continue;
        }
        double f_upper = n->within_sl_frac[(size_t)ng * ew_cap.n_shells +
                                            ew_cap.shell];
        psum += p;
        ew_atomic_inc_int(&ew_cap.target_mapped);
        double threshold_eV = chi_eV - a->level_energy_eV[lower_global] +
                              a->level_energy_eV[upper_global];
        double nu_threshold = threshold_eV * EV_TO_ERG / H_PLANCK;
        if (!(threshold_eV > 0.0) || !(nu_threshold < n->nu_max) ||
            !(T_e > 0.0) || !(n_e > 0.0)) {
            ew_atomic_inc_int(&ew_cap.bad_rate);
            continue;
        }
        double rad_ion = 0.0, milne = 0.0, sigma_edge = 0.0;
        const double kTe = K_BOLTZMANN * T_e;
        /* [A2-05] rad_ion for field sources 0/1 = canonical-view integral
         * (blocked => no field term, counted on the NLTEConfig R6 counters).
         * Source 2 (JEQB falsifier) keeps its per-bin B(T_e) device in the
         * loop below, which also still owns sigma_edge and the Milne term. */
        if (field_source != 2) {
            BfRateResult br_ew;
            if (nlte_bf_gamma_canonical(n, ew_cap.shell, sigma_row, sigma_0,
                                        nu_threshold, &br_ew) == 0 &&
                (br_ew.state == BF_RATE_VALID ||
                 br_ew.state == BF_RATE_EXACT_ZERO))
                rad_ion = br_ew.gamma;
        }
        for (int bb = 0; bb < n->n_freq_bins; bb++) {
            double loglo = log(n->nu_min) + bb * n->d_log_nu;
            double nu = exp(loglo + 0.5 * n->d_log_nu);
            if (nu < nu_threshold) continue;
            double sigma = sigma_row ? sigma_row[bb] :
                sigma_0 * pow(nu_threshold / nu, 3.0);
            if (!(sigma > 0.0) || !isfinite(sigma)) continue;
            if (sigma_edge == 0.0) sigma_edge = sigma;
            double dnu = exp(loglo + n->d_log_nu) - exp(loglo);
            double pref = 4.0 * M_PI_VAL * sigma /
                          (H_PLANCK * nu) * dnu;
            double J = 0.0;
            (void)nlte_bf_field_source(
                n, T_e, nu,
                n->J_nu[(size_t)ew_cap.shell * n->n_freq_bins + bb],
                0, NULL, NULL, &J);
            if (field_source == 2) {
                rad_ion += pref * J;   /* J == B_nu(T_e) under JEQB */
                ew_atomic_inc_int(&ew_cap.bf_pref_j_bins);
                ew_atomic_inc_int(&ew_cap.bf_jeqb_bins);
            }
            double x = H_PLANCK * nu / kTe;
            if (x < 700.0) {
                double spont = 2.0 * H_PLANCK * nu * nu * nu /
                               (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
                milne += pref * (spont + J) * exp(-x);
            }
        }
        int g_lower = a->level_g[lower_global] > 0 ? a->level_g[lower_global] : 1;
        int g_upper = a->level_g[upper_global] > 0 ? a->level_g[upper_global] : 1;
        double log_nstar = log(n_e) + 1.5 * log(H_PLANCK * H_PLANCK /
            (2.0 * M_PI_VAL * M_ELECTRON * K_BOLTZMANN * T_e)) +
            log((double)g_lower / (2.0 * (double)g_upper)) +
            threshold_eV * EV_TO_ERG / kTe;
        if (!(log_nstar < log(DBL_MAX)) || !isfinite(log_nstar)) {
            ew_atomic_inc_int(&ew_cap.nstar_cap); /* measured rejection; never capped */
            continue;
        }
        double nstar = exp(log_nstar);
        double rad_rec = nstar * milne;
        double coll_ion = 0.0, coll_rec = 0.0;
        if (nlte_bf_collisional_enabled() && sigma_edge > 0.0) {
            double u = threshold_eV * EV_TO_ERG / kTe;
            if (u > 0.0 && u < 700.0) {
                int stage = a->level_ion[lower_global];
                double gaunt = stage <= 0 ? 0.1 : (stage == 1 ? 0.2 : 0.3);
                coll_ion = n_e * 1.55e13 / sqrt(T_e) * gaunt * sigma_edge *
                           exp(-u) / u;
                coll_rec = coll_ion * nstar;
            }
        }
        if (!isfinite(rad_ion) || !isfinite(rad_rec) ||
            !isfinite(coll_ion) || !isfinite(coll_rec)) {
            ew_atomic_inc_int(&ew_cap.nonfinite_guard);
            continue;
        }
        nlte_ew_capture_transition(NLTE_EW_RAD_BF, upper_sl, lower_sl,
                                   p * rad_ion * lower_fraction);
        nlte_ew_capture_transition(NLTE_EW_RAD_BF, lower_sl, upper_sl,
                                   p * rad_rec * f_upper);
        nlte_ew_capture_transition(NLTE_EW_COLL_BF, upper_sl, lower_sl,
                                   p * coll_ion * lower_fraction);
        nlte_ew_capture_transition(NLTE_EW_COLL_BF, lower_sl, upper_sl,
                                   p * coll_rec * f_upper);
    }
    if (fabs(psum - 1.0) > 1e-12)
        ew_atomic_inc_int(&ew_cap.target_fail);
}

void nlte_ew_capture_nt_routes(double rate_per_particle) {
    if (!ew_cap.active || !(rate_per_particle > 0.0) ||
        !isfinite(rate_per_particle)) return;
    AtomicData *a = ew_cap.atom;
    NLTEConfig *n = ew_cap.nlte;
    int lo_slot = -1;
    for (int i = 0; i < n->n_nlte_ions; i++)
        if (n->nlte_ion_super_offset[i] == ew_cap.pair_super_start) {
            lo_slot = i; break;
        }
    if (lo_slot < 0) { ew_atomic_inc_int(&ew_cap.bad_rate); return; }
    int l0 = n->nlte_ion_level_offset[lo_slot];
    int l1 = n->nlte_ion_level_offset[lo_slot + 1];
    for (int g = l0; g < l1; g++) {
        int lower = n->nlte_to_global_level[g];
        if (!a->ma_rr_loaded || !a->ma_rr_target_offset) continue;
        int b = a->ma_rr_target_offset[lower];
        int e = a->ma_rr_target_offset[lower + 1];
        if (b < 0 || e <= b || e > a->ma_rr_n_routes) continue;
        int source = n->fl_to_super[g] - ew_cap.pair_super_start;
        double f = n->within_sl_frac[(size_t)g * ew_cap.n_shells + ew_cap.shell];
        for (int r = b; r < e; r++) {
            int upper = a->ma_rr_targets[r];
            double p = a->ma_rr_probability[r];
            int ng = (upper >= 0 && upper < a->n_levels) ?
                     n->global_to_nlte_level[upper] : -1;
            if (ng < 0 || !(p > 0.0) || !isfinite(p)) {
                ew_atomic_inc_int(&ew_cap.target_fallback);
                continue;
            }
            int target = n->fl_to_super[ng] - ew_cap.pair_super_start;
            nlte_ew_capture_transition(NLTE_EW_NT_BF, target, source,
                                       rate_per_particle * p * f);
        }
    }
}

static int ew_find_slot(const NLTEConfig *n, int Z, int stage) {
    for (int i = 0; i < n->n_nlte_ions; i++)
        if (n->nlte_Z[i] == Z && n->nlte_ion[i] == stage) return i;
    return -1;
}

static int ew_find_ip(const AtomicData *a, int Z, int stage) {
    for (int i = 0; i < a->n_ion_pops; i++)
        if (a->ion_pop_Z[i] == Z && a->ion_pop_stage[i] == stage) return i;
    return -1;
}

typedef struct {
    int ions_expected, ions_mapped, sl_expected, sl_mapped;
    int full_expected, full_mapped, lines_expected, lines_mapped;
    int continua_expected, continua_mapped, targets_expected, targets_mapped;
    int continua_kramers;
    int checksum_mismatch;
    int first_bad_lower, first_bad_target;
} EWCoverage;

static double ew_find_chi(const AtomicData *a, int Z, int stage) {
    for (int i = 0; i < a->n_ionization; i++)
        if (a->ioniz_Z[i] == Z && a->ioniz_ion[i] == stage)
            return a->ioniz_energy_eV[i];
    return -1.0;
}

/* R1 source invariant: shadow assembly owns this expanded index view.  It is a
 * private shallow clone for radiation/estimator inputs plus private copies of
 * every layout-derived array and population buffer.  Consequently arming
 * COMMIT=0 cannot change NLTEConfig.n_nlte_ions, super_mode, offsets, maps,
 * within_sl_frac, or production populations. */
static const int ew_view_Z[NLTE_EW_IONS] = {
    14,14, 20,20, 26,26,26, 16,16,16, 27,27, 28,28, 6,6, 12,12,
    22,22, 24,24, 13,13, 21,21, 23,23, 25,25, 8,8,8
};
static const int ew_view_stage[NLTE_EW_IONS] = {
     1, 2,  1, 2,  1, 2, 3,  1, 2, 3,  1, 2,  1, 2, 1,2,  1, 2,
     1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2, 0,1,2
};

typedef struct {
    NLTEConfig cfg;
    int active;
} EWPrivateView;

static void ew_private_view_free(EWPrivateView *v) {
    if (!v || !v->active) return;
    free(v->cfg.nlte_to_global_level);
    free(v->cfg.global_to_nlte_level);
    free(v->cfg.nlte_line_map);
    free(v->cfg.nlte_level_populations);
    free(v->cfg.fl_to_super);
    free(v->cfg.super_anchor_global);
    free(v->cfg.within_sl_frac);
    memset(v, 0, sizeof(*v));
}

static int ew_private_view_init(EWPrivateView *v, const NLTEConfig *src,
                                AtomicData *a, PlasmaState *p,
                                OpacityState *opacity) {
    memset(v, 0, sizeof(*v));
    v->cfg = *src;
    NLTEConfig *n = &v->cfg;
    n->nlte_to_global_level = NULL;
    n->global_to_nlte_level = NULL;
    n->nlte_line_map = NULL;
    n->nlte_level_populations = NULL;
    n->fl_to_super = NULL;
    n->super_anchor_global = NULL;
    n->within_sl_frac = NULL;
    if (nlte_build_projection(n, a, opacity, p->n_shells,
                              ew_view_Z, ew_view_stage, NLTE_EW_IONS, 1,
                              NULL) != 0)
        return -1;
    v->active = 1;
    /* Projection topology is shared; only the shadow population seed is
     * private-lane work. */
    for (int gl = 0; gl < a->n_levels; gl++) {
        int g = n->global_to_nlte_level[gl];
        int src_g = src->global_to_nlte_level ?
                    src->global_to_nlte_level[gl] : -1;
        if (g >= 0 && src_g >= 0)
            for (int s = 0; s < p->n_shells; s++)
                n->nlte_level_populations[(size_t)g * p->n_shells + s] =
                    src->nlte_level_populations[(size_t)src_g * p->n_shells + s];
    }
    if (nlte_precompute_within_sl_frac_checked(
            n, a, p, p->n_shells) != 0) {
        ew_private_view_free(v);
        return -1;
    }
    return 0;
}

static void ew_projection_coverage(const NLTEConfig *n, const AtomicData *a,
                                   int Z, int slots[3], EWCoverage *c) {
    memset(c, 0, sizeof(*c));
    c->first_bad_lower = c->first_bad_target = -1;
    c->ions_expected = 3;
    for (int q = 0; q < 3; q++) {
        int slot = slots[q];
        if (slot < 0) continue;
        c->ions_mapped++;
        int s0 = n->nlte_ion_super_offset[slot], s1 = n->nlte_ion_super_offset[slot+1];
        c->sl_expected += s1-s0; c->sl_mapped += s1-s0;
        int l0 = n->nlte_ion_level_offset[slot], l1 = n->nlte_ion_level_offset[slot+1];
        c->full_expected += l1-l0;
        for (int g = l0; g < l1; g++) {
            int gl = n->nlte_to_global_level[g];
            if (gl >= 0 && gl < a->n_levels && n->global_to_nlte_level[gl] == g &&
                n->fl_to_super[g] >= s0 && n->fl_to_super[g] < s1 &&
                n->super_anchor_global[n->fl_to_super[g]] >= 0)
                c->full_mapped++;
        }
    }
    for (int line = 0; line < a->n_lines; line++) {
        int st = a->line_ion_number[line];
        if (a->line_atomic_number[line] != Z || st < 1 || st > 3) continue;
        c->lines_expected++;
        int slot = slots[st-1];
        if (slot >= 0 && n->nlte_line_map[line] == slot) {
            int ip = ew_find_ip(a,Z,st), lo=-1, up=-1;
            if (ip >= 0) for (int gl=a->level_offset[ip]; gl<a->level_offset[ip+1]; gl++) {
                if (a->level_num[gl] == a->line_level_lower[line]) lo=gl;
                if (a->level_num[gl] == a->line_level_upper[line]) up=gl;
            }
            if (lo >= 0 && up >= 0 && n->global_to_nlte_level[lo] >= 0 &&
                n->global_to_nlte_level[up] >= 0) c->lines_mapped++;
        }
    }
    for (int q = 0; q < 2; q++) {
        int slot=slots[q], l0=n->nlte_ion_level_offset[slot], l1=n->nlte_ion_level_offset[slot+1];
        double chi=ew_find_chi(a,Z,q+1);
        for (int g=l0; g<l1; g++) {
            int gl=n->nlte_to_global_level[g];
            if (!(chi > a->level_energy_eV[gl])) continue; /* no positive continuum */
            c->continua_expected++;
            /* D3: exactly the authority lane's missing-row contract: Kramers
             * sigma and a single upper-ion-ground target. */
            if (!a->cmfgen_loaded ||
                a->cmfgen_n_freq_bins != n->n_freq_bins ||
                !a->cmfgen_has_sigma ||
                !a->cmfgen_sigma_bf || !a->cmfgen_has_sigma[gl]) {
                c->continua_kramers++;
                c->targets_expected++;
                int upper_sl = n->nlte_ion_super_offset[slots[q+1]];
                int upper = n->super_anchor_global[upper_sl];
                if (upper >= 0 && upper < a->n_levels &&
                    a->level_Z[upper] == Z && a->level_ion[upper] == q+2) {
                    c->continua_mapped++;
                    c->targets_mapped++;
                } else if (c->first_bad_lower < 0) {
                    c->first_bad_lower=gl; c->first_bad_target=upper;
                }
                continue;
            }
            int ok=1, nr=0; double psum=0.0;
            if (!a->ma_rr_loaded || !a->ma_rr_target_offset || !a->ma_rr_targets ||
                !a->ma_rr_probability) ok=0;
            int b=ok?a->ma_rr_target_offset[gl]:0;
            int e=ok?a->ma_rr_target_offset[gl+1]:0;
            if (b<0 || e<=b || e>a->ma_rr_n_routes) ok=0;
            if (!ok) {
                c->targets_expected++;
                if (c->first_bad_lower < 0) c->first_bad_lower=gl;
                continue;
            }
            for (int r=b;r<e;r++) {
                int t=a->ma_rr_targets[r]; double p=a->ma_rr_probability[r];
                c->targets_expected++; nr++; psum+=p;
                if (t>=0 && t<a->n_levels && p>0.0 && isfinite(p) &&
                    a->level_Z[t]==Z && a->level_ion[t]==q+2 &&
                    n->global_to_nlte_level[t]>=0) c->targets_mapped++;
                else {
                    ok=0;
                    if(c->first_bad_lower<0){c->first_bad_lower=gl;c->first_bad_target=t;}
                }
            }
            if (nr<=0 || fabs(psum-1.0)>1e-12) {
                ok=0; if(c->first_bad_lower<0)c->first_bad_lower=gl;
            }
            if (ok) c->continua_mapped++;
        }
    }
}

static double ew_element_density(const AtomicData *a, const PlasmaState *p,
                                 int Z, int shell) {
    for (int e = 0; e < a->n_elements; e++) {
        if (a->element_Z[e] != Z || !(a->element_mass_amu[e] > 0.0)) continue;
        double X = a->abundances[(size_t)e * p->n_shells + shell];
        double v = X * p->rho[shell] / (a->element_mass_amu[e] * AMU);
        if (v > 0.0 && isfinite(v)) return v;
    }
    double sum = 0.0;
    for (int i = 0; i < a->n_ion_pops; i++)
        if (a->ion_pop_Z[i] == Z)
            sum += a->ion_number_density[(size_t)i * p->n_shells + shell];
    return sum;
}

static uint64_t ew_fnv_bytes(uint64_t h, const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

typedef struct {
    int active;
    int level_count, sigma_count, route_source_count;
    int route_count, valid_route_count, bad_count;
    int data_pass, projection_pass, assembly_pass;
    int m_index;
    double *q_by_global;
    double q_sum, q_min, q_max;
    uint64_t q_checksum;
    double n_total, m_outside, m_before, m_after;
    double sum_stage[3];
    double conservation_residual, row_residual, flux_residual;
    double phi_forward[2], phi_reverse[2];
    double *forward_rate[2];
    double *reverse_rate[2];
} EWBoundaryMass;

static void ew_boundary_mass_free(EWBoundaryMass *bm) {
    if (!bm) return;
    free(bm->q_by_global);
    free(bm->forward_rate[0]);
    free(bm->forward_rate[1]);
    free(bm->reverse_rate[0]);
    free(bm->reverse_rate[1]);
    memset(bm, 0, sizeof(*bm));
}

static int ew_q_projection_valid(const double *q, int n, double *sum_out,
                                 double *min_out, double *max_out) {
    if (!q || n <= 0) return 0;
    double sum = 0.0, qmin = INFINITY, qmax = 0.0;
    for (int i = 0; i < n; i++) {
        if (!(q[i] >= 0.0) || !isfinite(q[i])) return 0;
        sum += q[i];
        if (!isfinite(sum)) return 0;
        if (q[i] < qmin) qmin = q[i];
        if (q[i] > qmax) qmax = q[i];
    }
    if (sum_out) *sum_out = sum;
    if (min_out) *min_out = qmin;
    if (max_out) *max_out = qmax;
    return fabs(sum - 1.0) <= 1e-12;
}

/* Seeded-defect fixture entry.  Production and the fixture deliberately use
 * this same strict validator; it records no repaired projection. */
int nlte_ew_test_q_projection(const double *q, int n, double *sum_out) {
    return ew_q_projection_valid(q, n, sum_out, NULL, NULL);
}

/* Fe V is represented by one matrix unknown, but its 200 target identities
 * remain explicit in this normalized Boltzmann projection.  No value is
 * clipped or repaired: malformed weights close the boundary stage. */
static int ew_boundary_mass_prepare(EWBoundaryMass *bm, const AtomicData *a,
                                    const PlasmaState *p, int Z, int shell,
                                    int N, double n_total) {
    memset(bm, 0, sizeof(*bm));
    if (Z != 26) return 0;
    bm->active = 1;
    bm->m_index = N - 1;
    bm->n_total = n_total;
    bm->q_min = INFINITY;
    bm->q_by_global = (double *)calloc(
        (size_t)(a->n_levels > 0 ? a->n_levels : 1), sizeof(double));
    bm->forward_rate[0] = (double *)calloc((size_t)N, sizeof(double));
    bm->forward_rate[1] = (double *)calloc((size_t)N, sizeof(double));
    bm->reverse_rate[0] = (double *)calloc((size_t)N, sizeof(double));
    bm->reverse_rate[1] = (double *)calloc((size_t)N, sizeof(double));
    if (!bm->q_by_global || !bm->forward_rate[0] || !bm->forward_rate[1] ||
        !bm->reverse_rate[0] || !bm->reverse_rate[1])
        return -1;
    double T_e = p->T_e[shell];
    if (!(T_e > 0.0) || !isfinite(T_e)) return 0;
    double E0 = INFINITY;
    for (int gl = 0; gl < a->n_levels; gl++)
        if (a->level_Z[gl] == 26 && a->level_ion[gl] == 4) {
            bm->level_count++;
            if (a->level_energy_eV[gl] < E0) E0 = a->level_energy_eV[gl];
        }
    if (bm->level_count != 200 || !isfinite(E0)) return 0;
    double kT = K_BOLTZMANN * T_e;
    double Zq = 0.0;
    for (int gl = 0; gl < a->n_levels; gl++) {
        if (a->level_Z[gl] != 26 || a->level_ion[gl] != 4) continue;
        double Erel = (a->level_energy_eV[gl] - E0) * EV_TO_ERG;
        double w = (double)a->level_g[gl] * exp(-Erel / kT);
        if (Erel < 0.0 || !(w >= 0.0) || !isfinite(w)) {
            bm->bad_count++;
            continue;
        }
        bm->q_by_global[gl] = w;
        Zq += w;
    }
    if (!(Zq > 0.0) || !isfinite(Zq) || bm->bad_count) return 0;
    bm->q_sum = 0.0;
    bm->q_checksum = UINT64_C(1469598103934665603);
    double q_target[200];
    int q_count = 0;
    for (int gl = 0; gl < a->n_levels; gl++) {
        if (a->level_Z[gl] != 26 || a->level_ion[gl] != 4) continue;
        double q = bm->q_by_global[gl] / Zq;
        if (!(q >= 0.0) || !isfinite(q)) { bm->bad_count++; continue; }
        bm->q_by_global[gl] = q;
        if (q_count < 200) q_target[q_count++] = q;
        else bm->bad_count++;
        bm->q_sum += q;
        if (q < bm->q_min) bm->q_min = q;
        if (q > bm->q_max) bm->q_max = q;
        bm->q_checksum = ew_fnv_bytes(bm->q_checksum, &gl, sizeof(gl));
        bm->q_checksum = ew_fnv_bytes(bm->q_checksum, &q, sizeof(q));
    }
    double recorded_sum = 0.0, recorded_min = INFINITY, recorded_max = 0.0;
    bm->projection_pass = !bm->bad_count && q_count == 200 &&
        ew_q_projection_valid(q_target, q_count, &recorded_sum, &recorded_min,
        &recorded_max);
    bm->q_sum = recorded_sum;
    bm->q_min = recorded_min;
    bm->q_max = recorded_max;
    for (int ip = 0; ip < a->n_ion_pops; ip++) {
        if (a->ion_pop_Z[ip] != 26) continue;
        double mass = a->ion_number_density[(size_t)ip*p->n_shells + shell];
        if (!(mass >= 0.0) || !isfinite(mass)) { bm->bad_count++; continue; }
        if (a->ion_pop_stage[ip] == 4) bm->m_before += mass;
        if (a->ion_pop_stage[ip] == 0 || a->ion_pop_stage[ip] >= 5)
            bm->m_outside += mass;
    }
    bm->data_pass = bm->projection_pass && bm->level_count == 200 &&
                    bm->bad_count == 0 && isfinite(ew_find_chi(a,26,4));
    return 0;
}

static void ew_boundary_transition(EWBoundaryMass *bm, double *plane,
                                   double *ledger, int N, int channel_slot,
                                   int target, int source, double rate) {
    if (!(rate >= 0.0) || !isfinite(rate) || target < 0 || source < 0 ||
        target >= N || source >= N) { bm->bad_count++; return; }
    if (rate == 0.0 || target == source) return;
    EW(plane,N,target,source) += rate;
    EW(plane,N,source,source) -= rate;
    ledger[source] += rate;
    if (target == bm->m_index) bm->forward_rate[channel_slot][source] += rate;
    if (source == bm->m_index) bm->reverse_rate[channel_slot][target] += rate;
}

/* Assemble the available Fe IV 200/200 continuum routes into the scalar M_V
 * row/column.  Fe V->VI, Fe V bb and DR/autoionization are deliberately absent. */
static void ew_boundary_mass_assemble(EWBoundaryMass *bm, NLTEConfig *n,
                                      AtomicData *a, PlasmaState *p,
                                      int iv_slot, int iv_start, int shell,
                                      double *plane[NLTE_EW_N_CHANNELS],
                                      double *ledger[NLTE_EW_N_CHANNELS],
                                      EWFireCounts *fire, int N) {
    if (!bm->active || !bm->data_pass) return;
    double T_e = p->T_e[shell], n_e = p->n_electron[shell];
    if (!(n_e > 0.0) || !isfinite(n_e) || !a->cmfgen_loaded ||
        a->cmfgen_n_freq_bins != n->n_freq_bins || !a->cmfgen_has_sigma ||
        !a->cmfgen_sigma_bf || !a->ma_rr_loaded ||
        !a->ma_rr_target_offset || !a->ma_rr_targets ||
        !a->ma_rr_probability) { bm->bad_count++; return; }
    int field_source = nlte_bf_field_source(
        n,T_e,0.0,0.0,0,NULL,NULL,NULL);
    int g0=n->nlte_ion_level_offset[iv_slot];
    int g1=n->nlte_ion_level_offset[iv_slot+1];
    int sl0=n->nlte_ion_super_offset[iv_slot];
    double chi_eV=ew_find_chi(a,26,3);
    for(int g=g0;g<g1;g++){
        int lower=n->nlte_to_global_level[g];
        if(lower<0 || !a->cmfgen_has_sigma[lower]){bm->bad_count++;continue;}
        bm->sigma_count++;
        int begin=a->ma_rr_target_offset[lower];
        int end=a->ma_rr_target_offset[lower+1];
        if(begin<0||end<=begin||end>a->ma_rr_n_routes){bm->bad_count++;continue;}
        double psum=0.0;
        int source=iv_start+n->fl_to_super[g]-sl0;
        double f_lower=n->within_sl_frac[(size_t)g*p->n_shells+shell];
        const double *sigma_row=&a->cmfgen_sigma_bf[
            (size_t)lower*a->cmfgen_n_freq_bins];
        for(int r=begin;r<end;r++){
            bm->route_count++;
            int upper=a->ma_rr_targets[r];
            double prob=a->ma_rr_probability[r];
            if(upper<0||upper>=a->n_levels||a->level_Z[upper]!=26||
               a->level_ion[upper]!=4||!(prob>0.0)||!isfinite(prob)||
               !(bm->q_by_global[upper]>=0.0)||
               !isfinite(bm->q_by_global[upper])){bm->bad_count++;continue;}
            psum+=prob; bm->valid_route_count++;
            double threshold=chi_eV-a->level_energy_eV[lower]+
                             a->level_energy_eV[upper];
            double nu0=threshold*EV_TO_ERG/H_PLANCK;
            if(!(threshold>0.0)||!(nu0<n->nu_max)){bm->bad_count++;continue;}
            double rad_ion=0.0,milne=0.0,sigma_edge=0.0;
            double kT=K_BOLTZMANN*T_e;
            /* [A2-05] sources 0/1: canonical-view integral (blocked => no
             * field term, counted).  Source 2 keeps its per-bin JEQB device. */
            if(field_source!=2){
                BfRateResult br_bm;
                if(nlte_bf_gamma_canonical(n,shell,sigma_row,0.0,nu0,&br_bm)==0&&
                   (br_bm.state==BF_RATE_VALID||
                    br_bm.state==BF_RATE_EXACT_ZERO))
                    rad_ion=br_bm.gamma;
            }
            for(int bb=0;bb<n->n_freq_bins;bb++){
                double loglo=log(n->nu_min)+bb*n->d_log_nu;
                double nu=exp(loglo+0.5*n->d_log_nu);
                if(nu<nu0)continue;
                double sigma=sigma_row[bb];
                if(!(sigma>0.0)||!isfinite(sigma))continue;
                if(sigma_edge==0.0)sigma_edge=sigma;
                double dnu=exp(loglo+n->d_log_nu)-exp(loglo);
                double pref=4.0*M_PI_VAL*sigma/(H_PLANCK*nu)*dnu;
                double J=0.0;
                (void)nlte_bf_field_source(n,T_e,nu,
                    n->J_nu[(size_t)shell*n->n_freq_bins+bb],0,NULL,NULL,&J);
                if(field_source==2){
                    rad_ion+=pref*J;   /* J == B_nu(T_e) under JEQB */
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    fire->bf_pref_j_bins++;
#ifdef _OPENMP
#pragma omp atomic update
#endif
                    fire->bf_jeqb_bins++;
                }
                double x=H_PLANCK*nu/kT;
                double spont=2.0*H_PLANCK*nu*nu*nu/
                             (C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT);
                double term=pref*(spont+J)*exp(-x);
                if(!isfinite(term)){bm->bad_count++;continue;}
                milne+=term;
            }
            int gl=a->level_g[lower],gu=a->level_g[upper];
            if(gl<=0||gu<=0){bm->bad_count++;continue;}
            double log_nstar=log(n_e)+1.5*log(H_PLANCK*H_PLANCK/
                (2.0*M_PI_VAL*M_ELECTRON*K_BOLTZMANN*T_e))+
                log((double)gl/(2.0*(double)gu))+threshold*EV_TO_ERG/kT;
            if(!isfinite(log_nstar)||!(log_nstar<log(DBL_MAX))){bm->bad_count++;continue;}
            double nstar=exp(log_nstar);
            double coll_ion=0.0,coll_rec=0.0;
            if(nlte_bf_collisional_enabled()&&sigma_edge>0.0){
                double u=threshold*EV_TO_ERG/kT;
                if(u>0.0&&isfinite(u)){
                    coll_ion=n_e*1.55e13/sqrt(T_e)*0.3*sigma_edge*exp(-u)/u;
                    coll_rec=coll_ion*nstar;
                }
            }
            double q=bm->q_by_global[upper];
            double rates[4]={prob*rad_ion*f_lower,prob*nstar*milne*q,
                             prob*coll_ion*f_lower,prob*coll_rec*q};
            for(int k=0;k<4;k++)if(!isfinite(rates[k])||rates[k]<0.0)bm->bad_count++;
            ew_boundary_transition(bm,plane[NLTE_EW_RAD_BF],ledger[NLTE_EW_RAD_BF],
                N,0,bm->m_index,source,rates[0]);
            ew_boundary_transition(bm,plane[NLTE_EW_RAD_BF],ledger[NLTE_EW_RAD_BF],
                N,0,source,bm->m_index,rates[1]);
            ew_boundary_transition(bm,plane[NLTE_EW_COLL_BF],ledger[NLTE_EW_COLL_BF],
                N,1,bm->m_index,source,rates[2]);
            ew_boundary_transition(bm,plane[NLTE_EW_COLL_BF],ledger[NLTE_EW_COLL_BF],
                N,1,source,bm->m_index,rates[3]);
            fire->channel_events[NLTE_EW_RAD_BF]+=2;
            if(rates[2]>0.0||rates[3]>0.0)fire->channel_events[NLTE_EW_COLL_BF]+=2;
        }
        if(fabs(psum-1.0)<=1e-12)bm->route_source_count++;
        else bm->bad_count++;
    }
    bm->assembly_pass=bm->bad_count==0&&bm->level_count==200&&
        bm->sigma_count==200&&bm->route_source_count==200&&
        bm->route_count==bm->valid_route_count;
}

static uint64_t ew_atomic_checksum(const NLTEConfig *n, const AtomicData *a,
                                   int slots[3]) {
    uint64_t h = UINT64_C(1469598103934665603);
    h = ew_fnv_bytes(h, &a->n_levels, sizeof(a->n_levels));
    h = ew_fnv_bytes(h, &a->n_lines, sizeof(a->n_lines));
    h = ew_fnv_bytes(h, &a->cmfgen_n_freq_bins,
                     sizeof(a->cmfgen_n_freq_bins));
    for (int q = 0; q < 3; q++) {
        int Z = n->nlte_Z[slots[q]], stage = n->nlte_ion[slots[q]];
        h = ew_fnv_bytes(h, &Z, sizeof(Z));
        h = ew_fnv_bytes(h, &stage, sizeof(stage));
        double chi = ew_find_chi(a, Z, stage);
        h = ew_fnv_bytes(h, &chi, sizeof(chi));
        int l0 = n->nlte_ion_level_offset[slots[q]];
        int l1 = n->nlte_ion_level_offset[slots[q] + 1];
        for (int g = l0; g < l1; g++) {
            int gl = n->nlte_to_global_level[g];
            h = ew_fnv_bytes(h, &a->level_Z[gl], sizeof(int));
            h = ew_fnv_bytes(h, &a->level_ion[gl], sizeof(int));
            h = ew_fnv_bytes(h, &a->level_num[gl], sizeof(int));
            h = ew_fnv_bytes(h, &a->level_super[gl], sizeof(int));
            h = ew_fnv_bytes(h, &a->level_energy_eV[gl], sizeof(double));
            h = ew_fnv_bytes(h, &a->level_g[gl], sizeof(int));
            h = ew_fnv_bytes(h, &a->level_metastable[gl], sizeof(int));
            if (a->cmfgen_loaded && a->cmfgen_has_sigma) {
                h = ew_fnv_bytes(h, &a->cmfgen_has_sigma[gl], sizeof(int));
                if (a->cmfgen_has_sigma[gl] && a->cmfgen_sigma_bf)
                    h = ew_fnv_bytes(h, &a->cmfgen_sigma_bf[
                        (size_t)gl * a->cmfgen_n_freq_bins],
                        (size_t)a->cmfgen_n_freq_bins * sizeof(double));
            }
            if (a->ma_rr_loaded && a->ma_rr_target_offset) {
                int b = a->ma_rr_target_offset[gl], e = a->ma_rr_target_offset[gl + 1];
                h = ew_fnv_bytes(h, &b, sizeof(b));
                h = ew_fnv_bytes(h, &e, sizeof(e));
                for (int r = b; r < e; r++) {
                    h = ew_fnv_bytes(h, &a->ma_rr_targets[r], sizeof(int));
                    h = ew_fnv_bytes(h, &a->ma_rr_probability[r], sizeof(double));
                }
            }
        }
    }
    for (int line = 0; line < a->n_lines; line++) {
        int Z = a->line_atomic_number[line];
        int stage = a->line_ion_number[line];
        int selected = 0;
        for (int q = 0; q < 3; q++)
            if (Z == n->nlte_Z[slots[q]] && stage == n->nlte_ion[slots[q]])
                selected = 1;
        if (!selected) continue;
        h = ew_fnv_bytes(h, &line, sizeof(line));
        h = ew_fnv_bytes(h, &a->line_level_lower[line], sizeof(int));
        h = ew_fnv_bytes(h, &a->line_level_upper[line], sizeof(int));
        h = ew_fnv_bytes(h, &a->line_f_lu[line], sizeof(double));
        h = ew_fnv_bytes(h, &a->line_A_ul[line], sizeof(double));
        h = ew_fnv_bytes(h, &a->line_B_lu[line], sizeof(double));
        h = ew_fnv_bytes(h, &a->line_B_ul[line], sizeof(double));
        h = ew_fnv_bytes(h, &a->line_nu[line], sizeof(double));
    }
    if (a->feiii_col_loaded && n->nlte_Z[slots[0]] == 26) {
        h = ew_fnv_bytes(h, &a->feiii_col_n_trans, sizeof(int));
        h = ew_fnv_bytes(h, &a->feiii_col_n_temp, sizeof(int));
        h = ew_fnv_bytes(h, a->feiii_col_tgrid,
            (size_t)a->feiii_col_n_temp * sizeof(double));
        h = ew_fnv_bytes(h, a->feiii_col_lo,
            (size_t)a->feiii_col_n_trans * sizeof(int));
        h = ew_fnv_bytes(h, a->feiii_col_hi,
            (size_t)a->feiii_col_n_trans * sizeof(int));
        h = ew_fnv_bytes(h, a->feiii_col_omega,
            (size_t)a->feiii_col_n_trans * a->feiii_col_n_temp * sizeof(double));
    }
    return h;
}

static double ew_maxabs(const double *A, int N) {
    double m = 0.0;
    for (size_t k = 0; k < (size_t)N * N; k++)
        if (fabs(A[k]) > m) m = fabs(A[k]);
    return m;
}

static int ew_lu_factor(double *A, int N, int *piv, int *rank,
                        double *pivot_growth) {
    double amax = ew_maxabs(A, N), umax = 0.0;
    double tol = DBL_EPSILON * (double)(N > 1 ? N : 1) * (amax > 1.0 ? amax : 1.0);
    int rnk = 0;
    for (int k = 0; k < N; k++) {
        int p = k;
        double best = fabs(EW(A, N, k, k));
        for (int i = k + 1; i < N; i++) {
            double v = fabs(EW(A, N, i, k));
            if (v > best) { best = v; p = i; }
        }
        piv[k] = p;
        if (!(best > tol) || !isfinite(best)) continue;
        rnk++;
        if (p != k) for (int j = 0; j < N; j++) {
            double t = EW(A, N, k, j);
            EW(A, N, k, j) = EW(A, N, p, j);
            EW(A, N, p, j) = t;
        }
        double akk = EW(A, N, k, k);
        for (int i = k + 1; i < N; i++) {
            EW(A, N, i, k) /= akk;
            double lik = EW(A, N, i, k);
            for (int j = k + 1; j < N; j++)
                EW(A, N, i, j) -= lik * EW(A, N, k, j);
        }
    }
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            if (fabs(EW(A, N, i, j)) > umax) umax = fabs(EW(A, N, i, j));
    *rank = rnk;
    *pivot_growth = (amax > 0.0) ? umax / amax : INFINITY;
    return rnk == N ? 0 : -1;
}

static int ew_lu_solve(const double *LU, int N, const int *piv,
                       const double *rhs, double *x) {
    memcpy(x, rhs, (size_t)N * sizeof(double));
    for (int k = 0; k < N; k++) if (piv[k] != k) {
        double t = x[k]; x[k] = x[piv[k]]; x[piv[k]] = t;
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < i; j++) x[i] -= EW(LU, N, i, j) * x[j];
    for (int i = N - 1; i >= 0; i--) {
        for (int j = i + 1; j < N; j++) x[i] -= EW(LU, N, i, j) * x[j];
        double d = EW(LU, N, i, i);
        if (!(fabs(d) > 0.0) || !isfinite(d)) return -1;
        x[i] /= d;
    }
    return 0;
}

static void ew_matvec(const double *A, int N, const double *x, double *y,
                      int transpose) {
    for (int i = 0; i < N; i++) y[i] = 0.0;
    if (!transpose) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) y[i] += EW(A, N, i, j) * x[j];
    } else {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) y[j] += EW(A, N, i, j) * x[i];
    }
}

static double ew_norm2(const double *x, int N) {
    double scale = 0.0, ssq = 1.0;
    for (int i = 0; i < N; i++) if (x[i] != 0.0) {
        double ax = fabs(x[i]);
        if (scale < ax) { double q = scale / ax; ssq = 1.0 + ssq * q * q; scale = ax; }
        else { double q = ax / scale; ssq += q * q; }
    }
    return scale == 0.0 ? 0.0 : scale * sqrt(ssq);
}

static double ew_pythag(double a, double b) { return hypot(a,b); }
static double ew_sign(double a, double b) { return b >= 0.0 ? fabs(a) : -fabs(a); }

/* Golub-Reinsch SVD (Householder bidiagonalization plus implicit QR).  This is
 * a full singular-value algorithm: no normal equations, inverse iteration or
 * fixed-count extremal estimate is used. */
static int ew_singular_extrema(const double *A, int N, double *smax,
                               double *smin) {
    double *a=(double*)malloc((size_t)N*N*sizeof(double));
    double *v=(double*)calloc((size_t)N*N,sizeof(double));
    double *w=(double*)calloc((size_t)N,sizeof(double));
    double *rv1=(double*)calloc((size_t)N,sizeof(double));
    if(!a||!v||!w||!rv1) goto fail;
    for(int i=0;i<N;i++)for(int j=0;j<N;j++)a[(size_t)i*N+j]=EW(A,N,i,j);
    double g=0.0,scale=0.0,anorm=0.0;
    for(int i=0;i<N;i++){
        int l=i+1; rv1[i]=scale*g; g=scale=0.0; double s=0.0;
        for(int k=i;k<N;k++)scale+=fabs(a[(size_t)k*N+i]);
        if(scale){
            for(int k=i;k<N;k++){a[(size_t)k*N+i]/=scale;s+=a[(size_t)k*N+i]*a[(size_t)k*N+i];}
            double f=a[(size_t)i*N+i];g=-ew_sign(sqrt(s),f);double h=f*g-s;a[(size_t)i*N+i]=f-g;
            for(int j=l;j<N;j++){s=0.0;for(int k=i;k<N;k++)s+=a[(size_t)k*N+i]*a[(size_t)k*N+j];f=s/h;for(int k=i;k<N;k++)a[(size_t)k*N+j]+=f*a[(size_t)k*N+i];}
            for(int k=i;k<N;k++)a[(size_t)k*N+i]*=scale;
        }
        w[i]=scale*g; g=scale=0.0;s=0.0;
        if(i<N-1){
            for(int k=l;k<N;k++)scale+=fabs(a[(size_t)i*N+k]);
            if(scale){
                for(int k=l;k<N;k++){a[(size_t)i*N+k]/=scale;s+=a[(size_t)i*N+k]*a[(size_t)i*N+k];}
                double f=a[(size_t)i*N+l];g=-ew_sign(sqrt(s),f);double h=f*g-s;a[(size_t)i*N+l]=f-g;
                for(int k=l;k<N;k++)rv1[k]=a[(size_t)i*N+k]/h;
                for(int j=l;j<N;j++){s=0.0;for(int k=l;k<N;k++)s+=a[(size_t)j*N+k]*a[(size_t)i*N+k];for(int k=l;k<N;k++)a[(size_t)j*N+k]+=s*rv1[k];}
                for(int k=l;k<N;k++)a[(size_t)i*N+k]*=scale;
            }
        }
        double z=fabs(w[i])+fabs(rv1[i]);if(z>anorm)anorm=z;
    }
    for(int i=N-1;i>=0;i--){
        int l=i+1;
        if(i<N-1){
            if(g){for(int j=l;j<N;j++)v[(size_t)j*N+i]=(a[(size_t)i*N+j]/a[(size_t)i*N+l])/g;for(int j=l;j<N;j++){double s=0.0;for(int k=l;k<N;k++)s+=a[(size_t)i*N+k]*v[(size_t)k*N+j];for(int k=l;k<N;k++)v[(size_t)k*N+j]+=s*v[(size_t)k*N+i];}}
            for(int j=l;j<N;j++)v[(size_t)i*N+j]=v[(size_t)j*N+i]=0.0;
        }
        v[(size_t)i*N+i]=1.0;g=rv1[i];
    }
    for(int i=N-1;i>=0;i--){
        int l=i+1;g=w[i];for(int j=l;j<N;j++)a[(size_t)i*N+j]=0.0;
        if(g){g=1.0/g;for(int j=l;j<N;j++){double s=0.0;for(int k=l;k<N;k++)s+=a[(size_t)k*N+i]*a[(size_t)k*N+j];double f=(s/a[(size_t)i*N+i])*g;for(int k=i;k<N;k++)a[(size_t)k*N+j]+=f*a[(size_t)k*N+i];}for(int j=i;j<N;j++)a[(size_t)j*N+i]*=g;}
        else for(int j=i;j<N;j++)a[(size_t)j*N+i]=0.0;
        a[(size_t)i*N+i]+=1.0;
    }
    for(int k=N-1;k>=0;k--){
        int converged=0;
        for(int its=0;its<100;its++){
            int l,nm=0,flag=1;
            for(l=k;l>=0;l--){nm=l-1;if(fabs(rv1[l])+anorm==anorm){flag=0;break;}if(nm<0||fabs(w[nm])+anorm==anorm)break;}
            if(flag&&nm>=0){double c=0.0,s=1.0;for(int i=l;i<=k;i++){double f=s*rv1[i];rv1[i]=c*rv1[i];if(fabs(f)+anorm==anorm)break;g=w[i];double h=ew_pythag(f,g);w[i]=h;h=1.0/h;c=g*h;s=-f*h;for(int j=0;j<N;j++){double y=a[(size_t)j*N+nm],z=a[(size_t)j*N+i];a[(size_t)j*N+nm]=y*c+z*s;a[(size_t)j*N+i]=z*c-y*s;}}}
            double z=w[k];if(l==k){if(z<0.0){w[k]=-z;for(int j=0;j<N;j++)v[(size_t)j*N+k]=-v[(size_t)j*N+k];}converged=1;break;}
            if(its==99)goto fail;
            double x=w[l],y=w[k-1];g=rv1[k-1];double h=rv1[k];double f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);g=ew_pythag(f,1.0);f=((x-z)*(x+z)+h*((y/(f+ew_sign(g,f)))-h))/x;
            double c=1.0,s=1.0;
            for(int j=l;j<k;j++){int i=j+1;g=rv1[i];y=w[i];h=s*g;g=c*g;z=ew_pythag(f,h);rv1[j]=z;c=f/z;s=h/z;f=x*c+g*s;g=g*c-x*s;h=y*s;y*=c;for(int jj=0;jj<N;jj++){x=v[(size_t)jj*N+j];z=v[(size_t)jj*N+i];v[(size_t)jj*N+j]=x*c+z*s;v[(size_t)jj*N+i]=z*c-x*s;}z=ew_pythag(f,h);w[j]=z;if(z){z=1.0/z;c=f*z;s=h*z;}f=c*g+s*y;x=c*y-s*g;for(int jj=0;jj<N;jj++){y=a[(size_t)jj*N+j];z=a[(size_t)jj*N+i];a[(size_t)jj*N+j]=y*c+z*s;a[(size_t)jj*N+i]=z*c-y*s;}}
            rv1[l]=0.0;rv1[k]=f;w[k]=x;
        }
        if(!converged)goto fail;
    }
    *smax=0.0;*smin=INFINITY;
    for(int i=0;i<N;i++){double z=fabs(w[i]);if(z>*smax)*smax=z;if(z<*smin)*smin=z;}
    free(a);free(v);free(w);free(rv1);
    return (*smax > 0.0 && *smin > 0.0 && isfinite(*smax) && isfinite(*smin)) ? 0 : -1;
fail:
    free(a);free(v);free(w);free(rv1);
    *smax = INFINITY; *smin = 0.0;
    return -1;
}

static double ew_lu_rcond_1(const double *A, const double *LU, int N,
                            const int *piv) {
    double anorm = 0.0;
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int i = 0; i < N; i++) s += fabs(EW(A,N,i,j));
        if (s > anorm) anorm = s;
    }
    double *rhs = (double *)calloc((size_t)N, sizeof(double));
    double *x = (double *)malloc((size_t)N * sizeof(double));
    if (!rhs || !x || !(anorm > 0.0)) { free(rhs); free(x); return 0.0; }
    double ainvnorm = 0.0;
    for (int j = 0; j < N; j++) {
        memset(rhs, 0, (size_t)N * sizeof(double)); rhs[j] = 1.0;
        if (ew_lu_solve(LU,N,piv,rhs,x)) { ainvnorm = INFINITY; break; }
        double s = 0.0;
        for (int i = 0; i < N; i++) s += fabs(x[i]);
        if (s > ainvnorm) ainvnorm = s;
    }
    free(rhs); free(x);
    return (ainvnorm > 0.0 && isfinite(ainvnorm)) ? 1.0/(anorm*ainvnorm) : 0.0;
}

static int ew_rank_only(const double *A, int N) {
    const int M=N-1; /* row 0 is the designated conservation replacement row */
    double *LU = (double *)malloc((size_t)M * N * sizeof(double));
    if (!LU) return -1;
    double scale=0.0;
    for(int j=0;j<N;j++) for(int i=0;i<M;i++){
        double v=EW(A,N,i+1,j); LU[(size_t)j*M+i]=v;
        if(fabs(v)>scale)scale=fabs(v);
    }
    if(!(scale>0.0)||!isfinite(scale)){free(LU);return 0;}
    for(size_t k=0;k<(size_t)M*N;k++) LU[k]/=scale;
    /* Rank the N-1 SE rows that remain after row 0 is replaced by conservation.
     * Channel column sums separately prove that the omitted generator row is
     * their dependent closure.  The tolerance matches the kappa_2 <= 1e12 gate. */
    const double tol=1e-12;
    int rank=0;
    for(int col=0;col<N && rank<M;col++){
        int p=rank; double best=fabs(LU[(size_t)col*M+rank]);
        for(int i=rank+1;i<M;i++) if(fabs(LU[(size_t)col*M+i])>best){best=fabs(LU[(size_t)col*M+i]);p=i;}
        if(!(best>tol)||!isfinite(best)) continue;
        if(p!=rank) for(int j=0;j<N;j++){double t=LU[(size_t)j*M+rank];LU[(size_t)j*M+rank]=LU[(size_t)j*M+p];LU[(size_t)j*M+p]=t;}
        double d=LU[(size_t)col*M+rank];
        for(int i=rank+1;i<M;i++){double f=LU[(size_t)col*M+i]/d;for(int j=col+1;j<N;j++)LU[(size_t)j*M+i]-=f*LU[(size_t)j*M+rank];}
        rank++;
    }
    free(LU);
    return rank;
}

static int ew_components(const double *A, int N, int *zero_rows, int *zero_cols) {
    unsigned char *seen = (unsigned char *)calloc((size_t)N, 1);
    int *stack = (int *)malloc((size_t)N * sizeof(int));
    if (!seen || !stack) { free(seen); free(stack); return -1; }
    *zero_rows = *zero_cols = 0;
    for (int i = 0; i < N; i++) {
        int hr = 0, hc = 0;
        for (int j = 0; j < N; j++) {
            if (EW(A, N, i, j) != 0.0) hr = 1;
            if (EW(A, N, j, i) != 0.0) hc = 1;
        }
        if (!hr) (*zero_rows)++;
        if (!hc) (*zero_cols)++;
    }
    int comp = 0;
    for (int root = 0; root < N; root++) if (!seen[root]) {
        comp++; int top = 0; stack[top++] = root; seen[root] = 1;
        while (top) {
            int i = stack[--top];
            for (int j = 0; j < N; j++) if (!seen[j] && i != j &&
                (EW(A, N, i, j) != 0.0 || EW(A, N, j, i) != 0.0)) {
                seen[j] = 1; stack[top++] = j;
            }
        }
    }
    free(seen); free(stack); return comp;
}

static int ew_equilibrate(double *A, double *b, int N, double *row_scale,
                          double *col_scale) {
    for (int i = 0; i < N; i++) row_scale[i] = col_scale[i] = 1.0;
    int used = 0;
    for (int it = 0; it < 10; it++) {
        double maxdev = 0.0;
        for (int i = 0; i < N; i++) {
            double scale = 0.0, ssq = 1.0;
            for (int j = 0; j < N; j++) {
                double av=fabs(EW(A,N,i,j)); if(av==0.0)continue;
                if(scale<av){double q=scale/av;ssq=1.0+ssq*q*q;scale=av;}
                else{double q=av/scale;ssq+=q*q;}
            }
            double rn=scale==0.0?0.0:scale*sqrt(ssq);
            if (!(rn > 0.0)) continue;
            double f=1.0/sqrt(rn);
            if (!isfinite(f)) return -1;
            for(int j=0;j<N;j++) EW(A,N,i,j)*=f;
            b[i]*=f; row_scale[i]*=f;
            if(fabs(f-1.0)>maxdev)maxdev=fabs(f-1.0);
        }
        for (int i = 0; i < N; i++) {
            double scale = 0.0, ssq = 1.0;
            for (int j = 0; j < N; j++) {
                double av=fabs(EW(A,N,j,i)); if(av==0.0)continue;
                if(scale<av){double q=scale/av;ssq=1.0+ssq*q*q;scale=av;}
                else{double q=av/scale;ssq+=q*q;}
            }
            double cn=scale==0.0?0.0:scale*sqrt(ssq);
            if (!(cn > 0.0)) continue;
            double f=1.0/sqrt(cn);
            if (!isfinite(f)) return -1;
            for(int j=0;j<N;j++) EW(A,N,j,i)*=f;
            col_scale[i]*=f;
            if(fabs(f-1.0)>maxdev)maxdev=fabs(f-1.0);
        }
        used = it + 1;
        if (maxdev <= 1e-3) break;
    }
    return used;
}

typedef struct {
    int rank, equil_iters, refinement_iters, solve_fail;
    double pivot_growth, smax, smin, kappa, rcond;
    double refinement[11];
} EWSolveDiag;

static int ew_basic_solve(const double *A, const double *b, int N,
                          double *x, double *Aeq_out, double *beq_out,
                          double *row_scale_out, double *col_scale_out,
                          int *pivot_out, EWSolveDiag *d, int condition_diag) {
    double *Aeq = (double *)malloc((size_t)N * N * sizeof(double));
    double *beq = (double *)malloc((size_t)N * sizeof(double));
    double *Aref = (double *)malloc((size_t)N * N * sizeof(double));
    double *bref = (double *)malloc((size_t)N * sizeof(double));
    double *LU = (double *)malloc((size_t)N * N * sizeof(double));
    double *r = (double *)malloc((size_t)N * sizeof(double));
    double *dx = (double *)malloc((size_t)N * sizeof(double));
    double *rs = (double *)malloc((size_t)N * sizeof(double));
    double *cs = (double *)malloc((size_t)N * sizeof(double));
    int *piv = (int *)malloc((size_t)N * sizeof(int));
    if (!Aeq || !beq || !Aref || !bref || !LU || !r || !dx || !rs || !cs || !piv)
        goto fail;
    memset(d, 0, sizeof(*d));
    memcpy(Aeq, A, (size_t)N * N * sizeof(double)); memcpy(beq, b, (size_t)N*sizeof(double));
    d->equil_iters = ew_equilibrate(Aeq, beq, N, rs, cs);
    if (d->equil_iters < 0) goto fail;
    memcpy(Aref, Aeq, (size_t)N*N*sizeof(double)); memcpy(bref, beq, (size_t)N*sizeof(double));
    memcpy(LU, Aeq, (size_t)N*N*sizeof(double));
    if (ew_lu_factor(LU, N, piv, &d->rank, &d->pivot_growth)) goto fail;
    d->rcond = condition_diag ? ew_lu_rcond_1(Aref, LU, N, piv) : 0.0;
    if (pivot_out) memcpy(pivot_out,piv,(size_t)N*sizeof(int));
    if (ew_lu_solve(LU, N, piv, beq, x)) goto fail;
    for (int it = 0; it <= 10; it++) {
        long double nr2 = 0.0L;
        double nb = ew_norm2(bref, N);
        for (int i = 0; i < N; i++) {
            long double ax=0.0L;
            for(int j=0;j<N;j++)ax+=(long double)EW(Aref,N,i,j)*x[j];
            long double ri=(long double)bref[i]-ax;
            r[i]=(double)ri;nr2+=ri*ri;
        }
        double nr = sqrt((double)nr2) / (nb > 0.0 ? nb : 1.0);
        d->refinement[it] = nr;
        d->refinement_iters = it;
        if ((nr <= 1e-15 && it >= 2) || it == 10) break;
        if (ew_lu_solve(LU, N, piv, r, dx)) goto fail;
        for (int i = 0; i < N; i++) x[i] += dx[i];
    }
    for (int i = 0; i < N; i++) x[i] *= cs[i];
    if (Aeq_out) memcpy(Aeq_out, Aref, (size_t)N*N*sizeof(double));
    if (beq_out) memcpy(beq_out, bref, (size_t)N*sizeof(double));
    if (row_scale_out) memcpy(row_scale_out, rs, (size_t)N*sizeof(double));
    if (col_scale_out) memcpy(col_scale_out, cs, (size_t)N*sizeof(double));
    if (condition_diag) {
        ew_singular_extrema(Aref, N, &d->smax, &d->smin);
        d->kappa = (d->smin > 0.0) ? d->smax / d->smin : INFINITY;
    } else {
        d->smax=d->smin=d->kappa=0.0;
    }
    free(Aeq); free(beq); free(Aref); free(bref); free(LU); free(r); free(dx);
    free(rs); free(cs); free(piv); return 0;
fail:
    if (d) d->solve_fail = 1;
    free(Aeq); free(beq); free(Aref); free(bref); free(LU); free(r); free(dx);
    free(rs); free(cs); free(piv); return -1;
}

static double ew_channel_colsum(const double *A, int N) {
    if (!A || N <= 0) return INFINITY;
    double maxa = 0.0, maxsum = 0.0;
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int i = 0; i < N; i++) {
            double value = EW(A, N, i, j);
            if (!isfinite(value)) return INFINITY;
            if (fabs(value) > maxa) maxa = fabs(value);
            s += value;
            if (!isfinite(s)) return INFINITY;
        }
        if (fabs(s) > maxsum) maxsum = fabs(s);
    }
    return maxa > 0.0 ? maxsum / maxa : 0.0;
}

static double ew_channel_assembly_residual(const double *A,
                                           const double *expected_outflow,
                                           int N) {
    if (!A || !expected_outflow || N <= 0) return INFINITY;
    double scale = 0.0, worst = 0.0;
    for (int j = 0; j < N; j++) {
        double offdiag = 0.0;
        for (int i = 0; i < N; i++) {
            double value = EW(A, N, i, j);
            if (!isfinite(value)) return INFINITY;
            if (i != j) {
                offdiag += value;
                if (!isfinite(offdiag)) return INFINITY;
            }
        }
        double expected = expected_outflow[j];
        if (!isfinite(expected)) return INFINITY;
        double e_in = fabs(offdiag - expected);
        double e_debit = fabs(EW(A, N, j, j) + expected);
        if (!isfinite(e_in) || !isfinite(e_debit)) return INFINITY;
        if (fabs(expected) > scale) scale = fabs(expected);
        if (e_in > worst) worst = e_in;
        if (e_debit > worst) worst = e_debit;
    }
    double residual = scale > 0.0 ? worst / scale : worst;
    return isfinite(residual) ? residual : INFINITY;
}

/* Seeded-defect fixture entry: it invokes the same D6 ledger comparison used
 * by the production gate, without exposing any mutation hook in production. */
double nlte_ew_test_assembly_residual(const double *A,
                                      const double *expected_outflow, int N) {
    return ew_channel_assembly_residual(A, expected_outflow, N);
}

/* Backward-error allowance for sign classification.  This is an audit bound,
 * not a population repair: kappa is estimated as 1/rcond and the raw solution
 * remains untouched and separately reported. */
static int ew_negative_audit(const double *x, int N, double rcond,
                             double *raw_min_out, double *bound_out) {
    double raw_min = INFINITY, xnorm = 0.0;
    for (int i = 0; i < N; i++) {
        if (!isfinite(x[i])) continue;
        if (x[i] < raw_min) raw_min = x[i];
        if (fabs(x[i]) > xnorm) xnorm = fabs(x[i]);
    }
    double neps = (double)N * DBL_EPSILON;
    double bound = 0.0;
    if (rcond > 0.0 && isfinite(rcond) && neps < 1.0 && isfinite(xnorm))
        bound = (1.0 / rcond) * (neps / (1.0 - neps)) * xnorm;
    if (!isfinite(bound)) bound = 0.0; /* unbounded is not permission to pass */
    int negative = 0;
    for (int i = 0; i < N; i++)
        if (isfinite(x[i]) && x[i] < -bound) negative++;
    if (raw_min_out) *raw_min_out = raw_min;
    if (bound_out) *bound_out = bound;
    return negative;
}

int nlte_ew_test_negative_audit(const double *x, int N, double rcond,
                                double *raw_min_out, double *bound_out) {
    return ew_negative_audit(x, N, rcond, raw_min_out, bound_out);
}

/* Re-read the final normalized conservation row and RHS.  Coefficient/RHS
 * contract damage and equation residual are independent components. */
static double ew_boundary_row_audit(const double *Anorm, const double *b,
                                    const double *x, int N,
                                    double active_mass) {
    if (!Anorm || !b || !x || N <= 0 || !(active_mass > 0.0) ||
        !isfinite(active_mass) || !isfinite(b[0])) return INFINITY;
    double row_lhs = 0.0, coeff_worst = 0.0;
    for (int j = 0; j < N; j++) {
        double coeff = EW(Anorm, N, 0, j);
        if (!isfinite(coeff) || !isfinite(x[j])) return INFINITY;
        row_lhs += coeff * x[j];
        if (!isfinite(row_lhs)) return INFINITY;
        double delta = fabs(coeff - 1.0);
        if (delta > coeff_worst) coeff_worst = delta;
    }
    double rhs_residual = fabs(b[0] - active_mass) / active_mass;
    double equation_residual = fabs(row_lhs - b[0]) / active_mass;
    if (!isfinite(rhs_residual) || !isfinite(equation_residual))
        return INFINITY;
    double worst = coeff_worst;
    if (rhs_residual > worst) worst = rhs_residual;
    if (equation_residual > worst) worst = equation_residual;
    return worst;
}

double nlte_ew_test_boundary_row_audit(const double *Anorm, const double *b,
                                       const double *x, int N,
                                       double active_mass) {
    return ew_boundary_row_audit(Anorm, b, x, N, active_mass);
}

/* Compare every final Araw IV<->M coefficient with the independent route/q
 * ledger, then compare the population-weighted flux.  Per-target reverse
 * entries make a q redistribution detectable even when total reverse rate is
 * accidentally preserved. */
static double ew_boundary_flux_audit(
        const double *Araw, const double *x, int N, int m_index,
        int iv_begin, int iv_end, const double *forward0,
        const double *forward1, const double *reverse0,
        const double *reverse1) {
    if (!Araw || !x || !forward0 || !forward1 || !reverse0 || !reverse1 ||
        N <= 0 || m_index < 0 || m_index >= N || iv_begin < 0 ||
        iv_end > N || iv_begin >= iv_end) return INFINITY;
    double rate_scale = 0.0, rate_worst = 0.0;
    double flux_scale = 0.0, flux_worst = 0.0;
    for (int j = iv_begin; j < iv_end; j++) {
        double expected_forward = forward0[j] + forward1[j];
        double expected_reverse = reverse0[j] + reverse1[j];
        double matrix_forward = EW(Araw, N, m_index, j);
        double matrix_reverse = EW(Araw, N, j, m_index);
        if (!isfinite(expected_forward) || !isfinite(expected_reverse) ||
            !isfinite(matrix_forward) || !isfinite(matrix_reverse) ||
            !isfinite(x[j]) || !isfinite(x[m_index])) return INFINITY;
        double ef = fabs(matrix_forward - expected_forward);
        double er = fabs(matrix_reverse - expected_reverse);
        if (fabs(expected_forward) > rate_scale) rate_scale = fabs(expected_forward);
        if (fabs(expected_reverse) > rate_scale) rate_scale = fabs(expected_reverse);
        if (ef > rate_worst) rate_worst = ef;
        if (er > rate_worst) rate_worst = er;
        double ff = fabs((matrix_forward - expected_forward) * x[j]);
        double fr = fabs((matrix_reverse - expected_reverse) * x[m_index]);
        double sf = fabs(expected_forward * x[j]);
        double sr = fabs(expected_reverse * x[m_index]);
        if (!isfinite(ef) || !isfinite(er) || !isfinite(ff) ||
            !isfinite(fr) || !isfinite(sf) || !isfinite(sr)) return INFINITY;
        if (sf > flux_scale) flux_scale = sf;
        if (sr > flux_scale) flux_scale = sr;
        if (ff > flux_worst) flux_worst = ff;
        if (fr > flux_worst) flux_worst = fr;
    }
    double rate_residual = rate_scale > 0.0 ? rate_worst / rate_scale : rate_worst;
    double flux_residual = flux_scale > 0.0 ? flux_worst / flux_scale : flux_worst;
    if (!isfinite(rate_residual) || !isfinite(flux_residual)) return INFINITY;
    return rate_residual > flux_residual ? rate_residual : flux_residual;
}

double nlte_ew_test_boundary_flux_audit(
        const double *Araw, const double *x, int N, int m_index,
        int iv_begin, int iv_end, const double *forward0,
        const double *forward1, const double *reverse0,
        const double *reverse1) {
    return ew_boundary_flux_audit(Araw, x, N, m_index, iv_begin, iv_end,
                                  forward0, forward1, reverse0, reverse1);
}

static double ew_boundary_population_fraction(int has_population,
                                              double density,
                                              double n_elem) {
    if (!isfinite(n_elem)) return INFINITY;
    if (!has_population || !(n_elem > 0.0)) return 0.0;
    double fraction = density / n_elem;
    return isfinite(fraction) ? fraction : INFINITY;
}

static int ew_boundary_tau_add(double tau, int is_boundary,
                               double *tau_all, double *tau_boundary) {
    if (!tau_all || !tau_boundary || !isfinite(tau) ||
        !isfinite(*tau_all) || !isfinite(*tau_boundary)) goto nonfinite;
    A208ValueView signed_tau = {tau, tau == 0.0 ? A208_EXACT_ZERO : A208_VALID, 0};
    A208TauInteractionMeasure measure = a208_tau_interaction_measure(signed_tau);
    *tau_all += measure.value;
    if (!isfinite(*tau_all)) goto nonfinite;
    if (is_boundary) {
        *tau_boundary += tau;
        if (!isfinite(*tau_boundary)) goto nonfinite;
    }
    return 0;

nonfinite:
    if (tau_all) *tau_all = INFINITY;
    if (tau_boundary) *tau_boundary = INFINITY;
    return -1;
}

double nlte_ew_test_boundary_population_fraction(int has_population,
                                                 double density,
                                                 double n_elem) {
    return ew_boundary_population_fraction(has_population, density, n_elem);
}

int nlte_ew_test_boundary_tau_add(double tau, int is_boundary,
                                  double *tau_all, double *tau_boundary) {
    return ew_boundary_tau_add(tau, is_boundary, tau_all, tau_boundary);
}

static int ew_guard_config_count(void) {
    static const char *const guard[] = {
        "LUMINA_NLTE_FLOOR_REG", "LUMINA_NLTE_COLL_FLOOR",
        "LUMINA_NLTE_METASTABLE_COLL", "LUMINA_NLTE_BK_PARTIAL",
        "LUMINA_NLTE_FORCE_LTE_LEVELS", "LUMINA_NLTE_LTE_REPAIR",
        "LUMINA_NLTE_FLOOR_MODE", "LUMINA_NLTE_LTE_FLOOR",
        "LUMINA_NLTE_BK_CEIL", "LUMINA_NLTE_PER_ION_RESCALE",
        "LUMINA_NLTE_ION_LOCK", "LUMINA_DR_FLOOR_CMS", "LUMINA_TIMEDEP_ION",
        "LUMINA_SUPER_CUTOFF", "LUMINA_NLTE_STAGE4", "LUMINA_STAGE4_BK_CAP"
    };
    int n = 0;
    for (size_t i = 0; i < sizeof(guard)/sizeof(guard[0]); i++)
        if (ew_env_true(guard[i])) n++;
    return n;
}

static FILE *ew_open_dump(char *path, size_t n, int iter, int Z, int shell,
                          const char *role) {
    snprintf(path, n, "%s/lumina_ew_iter%04d_z%02d_s%03d_%s.csv",
             ew_gate.dump_dir, iter, Z, shell, role);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[EW][DUMP-FAIL] %s: %s\n", path, strerror(errno));
        ew_dump_failed = 1;
    }
    return fp;
}

static void ew_dump_identity(const NLTEConfig *n, const AtomicData *a,
                             int slots[3], int N, int iter, int Z, int shell,
                             uint64_t checksum) {
    char path[768]; FILE *fp = ew_open_dump(path, sizeof(path), iter, Z, shell, "identity");
    if (!fp) return;
    fprintf(fp, "matrix_index,Z,spectroscopic_stage,internal_stage,sl_id,anchor_global_level,member_full_level_ids,energy_eV,g_or_SL_partition,source_atomic_checksum\n");
    int base = 0;
    for (int q = 0; q < 3; q++) {
        int slot = slots[q], s0 = n->nlte_ion_super_offset[slot];
        int s1 = n->nlte_ion_super_offset[slot + 1];
        for (int sl = s0; sl < s1; sl++) {
            int anchor = n->super_anchor_global[sl];
            fprintf(fp, "%d,%d,%d,%d,%d,%d,", base + sl-s0, Z, q+2, q+1,
                    sl-s0, anchor);
            int first = 1;
            int l0 = n->nlte_ion_level_offset[slot], l1 = n->nlte_ion_level_offset[slot+1];
            double part = 0.0;
            for (int g = l0; g < l1; g++) if (n->fl_to_super[g] == sl) {
                int gl = n->nlte_to_global_level[g];
                fprintf(fp, "%s%d", first ? "" : ";", gl); first = 0;
                part += a->level_g[gl];
            }
            fprintf(fp, ",%.17g,%.17g,%016llx\n",
                    anchor >= 0 ? a->level_energy_eV[anchor] : NAN, part,
                    (unsigned long long)checksum);
        }
        base += s1-s0;
    }
    (void)N; ew_close_dump(fp, path);
}

static void ew_dump_sparse_matrix(const char *role, const double *A, const double *b,
                                  const double *row_scale,const double *col_scale,
                                  int N, int iter, int Z, int shell) {
    char path[768]; FILE *fp = ew_open_dump(path, sizeof(path), iter, Z, shell, role);
    if (!fp) return;
    fprintf(fp, "row,column,value,rhs,row_scale,column_scale\n");
    for (int j = 0; j < N; j++) for (int i = 0; i < N; i++)
        if (EW(A,N,i,j) != 0.0)
            fprintf(fp, "%d,%d,%.17g,%.17g,%.17g,%.17g\n", i,j,EW(A,N,i,j),
                    b ? b[i] : 0.0,row_scale?row_scale[i]:1.0,
                    col_scale?col_scale[j]:1.0);
    ew_close_dump(fp, path);
}

static void ew_dump_raw(double *plane[NLTE_EW_N_CHANNELS], int N,
                        int iter, int Z, int shell) {
    char path[768]; FILE *fp = ew_open_dump(path, sizeof(path), iter, Z, shell, "matrix_raw");
    if (!fp) return;
    fprintf(fp, "channel,row,column,value,rhs\n");
    for (int c = 0; c < NLTE_EW_N_CHANNELS; c++)
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++)
            if (EW(plane[c],N,i,j) != 0.0)
                fprintf(fp, "%s,%d,%d,%.17g,0\n", ew_channel_name[c],i,j,
                        EW(plane[c],N,i,j));
    ew_close_dump(fp, path);
}

static double ew_scaled_residual(const double *Araw, const double *x, int N,
                                 double n_elem, double t_ref) {
    double worst = 0.0;
    for (int i = 1; i < N; i++) {
        double in = 0.0, out = 0.0, r = 0.0;
        for (int j = 0; j < N; j++) {
            double f = EW(Araw,N,i,j) * x[j]; r += f;
            if (f >= 0.0) in += f; else out -= f;
        }
        double state_rate = (t_ref > 0.0 && isfinite(t_ref)) ?
                            fabs(x[i]) / t_ref : 0.0;
        double den = fmax(fmax(in,out), state_rate);
        double v = fabs(r) / (den > n_elem*DBL_MIN ? den : 1.0);
        if (v > worst) worst = v;
    }
    return worst;
}

static double ew_permutation_check(const double *A, const double *b, int N,
                                   const double *xref, int stage_start[4],
                                   uint64_t seed) {
    int *perm = (int *)malloc((size_t)N*sizeof(int));
    double *Ap = (double *)malloc((size_t)N*N*sizeof(double));
    double *bp = (double *)malloc((size_t)N*sizeof(double));
    double *xp = (double *)malloc((size_t)N*sizeof(double));
    double *xu = (double *)malloc((size_t)N*sizeof(double));
    if (!perm || !Ap || !bp || !xp || !xu) { free(perm);free(Ap);free(bp);free(xp);free(xu); return INFINITY; }
    double worst = 0.0;
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 0) {
            int k = 0;
            for (int q = 2; q >= 0; q--)
                for (int i = stage_start[q]; i < stage_start[q+1]; i++) perm[k++] = i;
            /* Optional scalar boundary-mass unknown follows the three stages. */
            for (int i = stage_start[3]; i < N; i++) perm[k++] = i;
        } else {
            for (int i = 0; i < N; i++) perm[i] = i;
            uint64_t x = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
            for (int i = N-1; i > 0; i--) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                int j = (int)(x % (uint64_t)(i+1)); int t=perm[i];perm[i]=perm[j];perm[j]=t;
            }
        }
        for (int i=0;i<N;i++) { bp[i]=b[perm[i]]; for(int j=0;j<N;j++) EW(Ap,N,i,j)=EW(A,N,perm[i],perm[j]); }
        EWSolveDiag d;
        if (ew_basic_solve(Ap,bp,N,xp,NULL,NULL,NULL,NULL,NULL,&d,0)) { worst=INFINITY; break; }
        for (int i=0;i<N;i++) xu[perm[i]]=xp[i];
        for (int q=0;q<3;q++) {
            double a=0.0,c=0.0;
            for(int i=stage_start[q];i<stage_start[q+1];i++){a+=xref[i];c+=xu[i];}
            double dv=fabs(a-c); if(dv>worst)worst=dv;
        }
        for(int i=stage_start[3];i<N;i++){
            double dv=fabs(xref[i]-xu[i]); if(dv>worst)worst=dv;
        }
    }
    free(perm);free(Ap);free(bp);free(xp);free(xu); return worst;
}

/* D7 hot/cold audit: rebuild channel planes from the current lagged
 * populations without solving or touching the production capture buffers. */
static int ew_rebuild_matrix(NLTEConfig *nlte, AtomicData *atom,
                             PlasmaState *plasma, OpacityState *opacity,
                             int slots[3], int stage_start[4], int N, int shell,
                             double time_explosion, GammaDeposition *gamma_dep,
                             double *out) {
    double *plane[NLTE_EW_N_CHANNELS]={0};
    double *ledger[NLTE_EW_N_CHANNELS]={0};
    int failed=0;
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++){
        plane[c]=(double*)calloc((size_t)N*N,sizeof(double));
        ledger[c]=(double*)calloc((size_t)N,sizeof(double));
        if(!plane[c]||!ledger[c])failed=1;
    }
    if(failed) goto done;
    for(int pair=0;pair<2;pair++){
        int lo=slots[pair],hi=slots[pair+1],base=stage_start[pair];
        int Np=nlte->nlte_ion_super_offset[hi+1]-
               nlte->nlte_ion_super_offset[lo];
        double *tmpA=(double*)calloc((size_t)Np*Np,sizeof(double));
        double *tmpb=(double*)calloc((size_t)Np,sizeof(double));
        if(!tmpA||!tmpb){free(tmpA);free(tmpb);failed=1;break;}
        EWFireCounts ignored; memset(&ignored,0,sizeof(ignored));
        ew_capture_begin(nlte,atom,shell,plasma->n_shells,lo,Np,N,base,
                         pair==1,plane,ledger);
        if(nlte_assemble_rate_matrix(nlte,atom,plasma,opacity,lo,hi,shell,
                                    time_explosion,tmpA,tmpb,Np,gamma_dep,NULL,-1)!=0)
            failed=1;
        ew_capture_end(NULL,NULL,NULL,NULL,&ignored);
        free(tmpA);free(tmpb);
    }
    if(!failed)
        for(int c=0;c<NLTE_EW_N_CHANNELS;c++)
            for(size_t k=0;k<(size_t)N*N;k++)out[k]+=plane[c][k];
done:
    if(ew_cap.active) memset(&ew_cap,0,sizeof(ew_cap));
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++){free(plane[c]);free(ledger[c]);}
    return failed ? -1 : 0;
}

static int ew_run_impl(NLTEConfig *nlte, AtomicData *atom,
                       PlasmaState *plasma, OpacityState *opacity,
                       int Z, int shell, int shell_label, double time_explosion,
                       GammaDeposition *gamma_dep, int commit_requested,
                       int *verdict_pass_out) {
    if (verdict_pass_out) *verdict_pass_out = 0;
    if (nlte_element_wide_config_status() != 0) return -1;
    if (!nlte_element_wide_matches(Z, shell_label)) return 0;
    GammaDepositionPublicationStatus gamma_status=
        gamma_deposition_require(gamma_dep,time_explosion);
    if(gamma_status!=GAMMA_PUBLICATION_OK){
        fprintf(stderr,"[GAMMA][BLOCKED] consumer=ELEMENT_WIDE_NONTHERMAL "
                "reason=%s expected_epoch=%.17g generation=%llu "
                "published_epoch=%.17g action=TERMINATE\n",
                gamma_deposition_publication_status_name(gamma_status),
                time_explosion,
                (unsigned long long)(gamma_dep?gamma_dep->generation:0),
                gamma_dep?gamma_dep->epoch:0.0);
        return -1;
    }
    ew_dump_failed = 0;
    EWPrivateView private_view;
    memset(&private_view, 0, sizeof(private_view));
    if (!commit_requested) {
        if (ew_private_view_init(&private_view, nlte, atom, plasma, opacity)) {
            fprintf(stderr,
                    "[EW][OOM] Z=%d s=%d private shadow index view failed\n",
                    Z, shell_label);
            return -1;
        }
        nlte = &private_view.cfg;
    }
    int slots[3] = { ew_find_slot(nlte,Z,1), ew_find_slot(nlte,Z,2), ew_find_slot(nlte,Z,3) };
    int iter = nlte->current_iter;
    int topology_fail = 0, guard_config_count = ew_guard_config_count();
    for (int q=0;q<3;q++) if (slots[q] < 0) topology_fail++;
    if (topology_fail) {
        fprintf(stderr,"[EW][TOPOLOGY-FAIL] Z=%d s=%d missing II/III/IV slot\n",Z,shell_label);
        ew_private_view_free(&private_view);
        return -1;
    }
    int stage_start[4] = {0,0,0,0};
    for(int q=0;q<3;q++) stage_start[q+1]=stage_start[q]+
        nlte->nlte_ion_super_offset[slots[q]+1]-nlte->nlte_ion_super_offset[slots[q]];
    int boundary_mass_active = (Z == 26);
    int N=stage_start[3] + (boundary_mass_active ? 1 : 0);
    if(N<=0) { ew_private_view_free(&private_view); return -1; }
    double n_elem=ew_element_density(atom,plasma,Z,shell);
    EWBoundaryMass boundary;
    if(ew_boundary_mass_prepare(&boundary,atom,plasma,Z,shell,N,n_elem)!=0){
        fprintf(stderr,"[EW][OOM] Z=%d s=%d boundary-mass projection\n",Z,shell_label);
        ew_boundary_mass_free(&boundary);
        ew_private_view_free(&private_view);
        return -1;
    }
    int c48_projected_levels=0;
    {
        const char *cut=getenv("LUMINA_SUPER_CUTOFF");
        int K=cut?atoi(cut):0;
        if(K>0)for(int q=0;q<3;q++)
            for(int g=nlte->nlte_ion_level_offset[slots[q]];
                g<nlte->nlte_ion_level_offset[slots[q]+1];g++){
                int gl=nlte->nlte_to_global_level[g];
                if(atom->level_num[gl]>=K&&atom->level_super[gl]==K)
                    c48_projected_levels++;
            }
    }
    EWCoverage coverage;
    ew_projection_coverage(nlte,atom,Z,slots,&coverage);
    uint64_t checksum_before=ew_atomic_checksum(nlte,atom,slots);
    double *plane[NLTE_EW_N_CHANNELS]={0};
    double *expected_outflow[NLTE_EW_N_CHANNELS]={0};
    double *Araw=(double*)calloc((size_t)N*N,sizeof(double));
    double *Anorm=(double*)calloc((size_t)N*N,sizeof(double));
    double *Aeq=(double*)calloc((size_t)N*N,sizeof(double));
    double *b=(double*)calloc((size_t)N,sizeof(double));
    double *beq=(double*)calloc((size_t)N,sizeof(double));
    double *x=(double*)calloc((size_t)N,sizeof(double));
    double *rs=(double*)calloc((size_t)N,sizeof(double));
    double *cs=(double*)calloc((size_t)N,sizeof(double));
    int *solve_piv=(int*)malloc((size_t)N*sizeof(int));
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++) {
        plane[c]=(double*)calloc((size_t)N*N,sizeof(double));
        expected_outflow[c]=(double*)calloc((size_t)N,sizeof(double));
    }
    int oom=!Araw||!Anorm||!Aeq||!b||!beq||!x||!rs||!cs||!solve_piv;
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++)
        if(!plane[c] || !expected_outflow[c]) oom=1;
    if(oom) { fprintf(stderr,"[EW][OOM] Z=%d s=%d N=%d\n",Z,shell_label,N); goto cleanup_fail; }

    int target_expected=0,target_mapped=0,target_fail=0,bad_rate=0;
    EWFireCounts fire; memset(&fire,0,sizeof(fire));
    for(int pair=0;pair<2;pair++) {
        int lo=slots[pair],hi=slots[pair+1];
        int base=stage_start[pair];
        int Np=nlte->nlte_ion_super_offset[hi+1]-nlte->nlte_ion_super_offset[lo];
        double *tmpA=(double*)calloc((size_t)Np*Np,sizeof(double));
        double *tmpb=(double*)calloc((size_t)Np,sizeof(double));
        if(!tmpA||!tmpb){free(tmpA);free(tmpb);goto cleanup_fail;}
        ew_capture_begin(nlte,atom,shell,plasma->n_shells,lo,Np,N,base,pair==1,
                         plane,expected_outflow);
        unsigned long calls_before[3], calls_after[3];
        nlte_ew_runtime_counts_snapshot(calls_before);
        if(nlte_assemble_rate_matrix(nlte,atom,plasma,opacity,lo,hi,shell,
                                    time_explosion,tmpA,tmpb,Np,gamma_dep,NULL,-1)!=0){
            free(tmpA);free(tmpb);goto cleanup_fail;
        }
        nlte_ew_runtime_counts_snapshot(calls_after);
        ew_capture_end(&target_expected,&target_mapped,&target_fail,&bad_rate,&fire);
        fire.save_restore_calls += (int)(calls_after[0] - calls_before[0]);
        fire.per_ion_pin_calls += (int)(calls_after[1] - calls_before[1]);
        fire.topstage_IV_calls += (int)(calls_after[2] - calls_before[2]);
        free(tmpA);free(tmpb);
    }
    ew_boundary_mass_assemble(&boundary,nlte,atom,plasma,slots[2],
                              stage_start[2],shell,plane,expected_outflow,
                              &fire,N);
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++)
        for(size_t k=0;k<(size_t)N*N;k++) Araw[k]+=plane[c][k];
    memcpy(Anorm,Araw,(size_t)N*N*sizeof(double));
    int hot_cold_required=0,hot_cold_rebuilds=0,hot_cold_identical=0;
    double hot_cold_max_abs=INFINITY;
    {
        const char *jp=getenv("LUMINA_NLTE_JBAR_POPS");
        const char *ma=getenv("LUMINA_MALI");
        int jm=jp?atoi(jp):0;
        hot_cold_required=(jm==2||jm==3||(ma&&atoi(ma)))?1:0;
    }
    if(hot_cold_required){
        int g0=nlte->nlte_ion_level_offset[slots[0]];
        int g1=nlte->nlte_ion_level_offset[slots[2]+1];
        int ng=g1-g0;
        double *saved=(double*)malloc((size_t)ng*sizeof(double));
        double *cold=(double*)calloc((size_t)N*N,sizeof(double));
        double *hot=(double*)calloc((size_t)N*N,sizeof(double));
        if(saved&&cold&&hot){
            for(int g=g0;g<g1;g++){
                saved[g-g0]=nlte->nlte_level_populations[
                    (size_t)g*plasma->n_shells+shell];
                nlte->nlte_level_populations[
                    (size_t)g*plasma->n_shells+shell]=0.0;
            }
            nlte->nlte_level_populations[
                (size_t)g0*plasma->n_shells+shell]=n_elem;
            if(!ew_rebuild_matrix(nlte,atom,plasma,opacity,slots,stage_start,N,
                                  shell,time_explosion,gamma_dep,cold))
                hot_cold_rebuilds++;
            for(int g=g0;g<g1;g++)
                nlte->nlte_level_populations[
                    (size_t)g*plasma->n_shells+shell]=0.0;
            int gh=nlte->nlte_ion_level_offset[slots[2]];
            nlte->nlte_level_populations[
                (size_t)gh*plasma->n_shells+shell]=n_elem;
            if(!ew_rebuild_matrix(nlte,atom,plasma,opacity,slots,stage_start,N,
                                  shell,time_explosion,gamma_dep,hot))
                hot_cold_rebuilds++;
            hot_cold_max_abs=0.0;
            for(size_t k=0;k<(size_t)N*N;k++){
                double d=fabs(cold[k]-hot[k]);
                if(d>hot_cold_max_abs)hot_cold_max_abs=d;
            }
            hot_cold_identical=(hot_cold_rebuilds==2 &&
                memcmp(cold,hot,(size_t)N*N*sizeof(double))==0);
            for(int g=g0;g<g1;g++)
                nlte->nlte_level_populations[
                    (size_t)g*plasma->n_shells+shell]=saved[g-g0];
        }
        free(saved);free(cold);free(hot);
    }
    double active_mass=n_elem-boundary.m_outside;
    if(!boundary.active)active_mass=n_elem;
    if(!(active_mass>=0.0)||!isfinite(active_mass))boundary.bad_count++;
    for(int j=0;j<N;j++) EW(Anorm,N,0,j)=1.0;
    b[0]=active_mass;

    uint64_t checksum_after=ew_atomic_checksum(nlte,atom,slots);
    coverage.checksum_mismatch = checksum_after != checksum_before;
    int raw_rank=ew_rank_only(Araw,N),zero_rows=0,zero_cols=0;
    int components=ew_components(Araw,N,&zero_rows,&zero_cols);
    int projection_ok = coverage.ions_mapped==coverage.ions_expected &&
        coverage.sl_mapped==coverage.sl_expected &&
        coverage.full_mapped==coverage.full_expected &&
        coverage.lines_mapped==coverage.lines_expected &&
        coverage.continua_mapped==coverage.continua_expected &&
        coverage.targets_expected>0 &&
        coverage.targets_mapped==coverage.targets_expected &&
        coverage.checksum_mismatch==0 && target_fail==0 && bad_rate==0 &&
        (!boundary.active || (boundary.data_pass && boundary.projection_pass &&
                              boundary.assembly_pass));
    EWSolveDiag sd; memset(&sd,0,sizeof(sd));
    sd.smax=INFINITY; sd.smin=0.0; sd.kappa=INFINITY;
    sd.rcond=0.0; sd.pivot_growth=INFINITY;
    int solve_attempted = projection_ok ? 1 : 0;
    int solve_fail = projection_ok ?
        ew_basic_solve(Anorm,b,N,x,Aeq,beq,rs,cs,solve_piv,&sd,1) : 1;
    double raw_smax=0.0,raw_smin=0.0;
    int raw_svd_fail=projection_ok ?
        ew_singular_extrema(Anorm,N,&raw_smax,&raw_smin) : 1;
    double raw_kappa=(!raw_svd_fail&&raw_smin>0.0)?raw_smax/raw_smin:INFINITY;
    double colsum[NLTE_EW_N_CHANNELS];
    double assembly_residual[NLTE_EW_N_CHANNELS],max_assembly_residual=0.0;
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++)
        colsum[c]=ew_channel_colsum(plane[c],N);
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++){
        assembly_residual[c]=ew_channel_assembly_residual(
            plane[c],expected_outflow[c],N);
        if(assembly_residual[c]>max_assembly_residual)
            max_assembly_residual=assembly_residual[c];
    }
    int nonfinite=0;
    double sumx=0.0;
    for(int i=0;i<N;i++){
        if(!isfinite(x[i]))nonfinite++;
        else sumx+=x[i];
    }
    double raw_solution_min=INFINITY,negative_error_bound=0.0;
    int negative=ew_negative_audit(x,N,sd.rcond,&raw_solution_min,
                                   &negative_error_bound);
    double conservation=(n_elem>0.0)?fabs(sumx-active_mass)/n_elem:INFINITY;
    double se_res=solve_fail?INFINITY:ew_scaled_residual(Araw,x,N,n_elem,
                                                         time_explosion);
    uint64_t checksum=checksum_after;
    double perm_abs=solve_fail?INFINITY:ew_permutation_check(Anorm,b,N,x,stage_start,checksum);
    double perm_frac=(n_elem>0.0)?perm_abs/n_elem:INFINITY;

    boundary.m_after=boundary.active&&!solve_fail?x[boundary.m_index]:0.0;
    if(boundary.active){
        for(int q=0;q<3;q++)
            for(int i=stage_start[q];i<stage_start[q+1];i++)
                boundary.sum_stage[q]+=x[i];
        boundary.conservation_residual=conservation;
        boundary.row_residual=ew_boundary_row_audit(
            Anorm,b,x,N,active_mass);
        for(int k=0;k<2;k++){
            double ledger_forward=0.0;
            double ledger_reverse=0.0;
            for(int j=stage_start[2];j<stage_start[3];j++) {
                ledger_forward+=boundary.forward_rate[k][j]*x[j];
                ledger_reverse+=boundary.reverse_rate[k][j]*x[boundary.m_index];
            }
            boundary.phi_forward[k]=ledger_forward;
            boundary.phi_reverse[k]=ledger_reverse;
        }
        boundary.flux_residual=ew_boundary_flux_audit(
            Araw,x,N,boundary.m_index,stage_start[2],stage_start[3],
            boundary.forward_rate[0],boundary.forward_rate[1],
            boundary.reverse_rate[0],boundary.reverse_rate[1]);
    }

    double boundary_max=0.0;
    for(int st_i=0;st_i<2;st_i++){
        int st=st_i?(boundary.active?5:4):0,ip=ew_find_ip(atom,Z,st);
        double density=ip>=0 ?
            atom->ion_number_density[(size_t)ip*plasma->n_shells+shell] : 0.0;
        double f=ew_boundary_population_fraction(ip>=0,density,n_elem);
        if(!isfinite(f))boundary_max=INFINITY;
        else if(f>boundary_max)boundary_max=f;
    }
    double tau_all=0.0,tau_boundary=0.0;
    if(opacity && opacity->tau_sobolev){
        for(int line=0;line<atom->n_lines;line++) if(atom->line_atomic_number[line]==Z){
            int st=atom->line_ion_number[line];
            int is_boundary=st==0||(!boundary.active&&st==4)||
                            (boundary.active&&st>=5);
            size_t tau_index=(size_t)line*plasma->n_shells+shell;
            double tau=opacity->tau_sobolev[tau_index];
            if(opacity->tau_validity &&
               opacity->tau_validity[tau_index]!=A208_VALID &&
               opacity->tau_validity[tau_index]!=A208_EXACT_ZERO){
                tau_all=INFINITY;tau_boundary=INFINITY;break;
            }
            if(ew_boundary_tau_add(tau,is_boundary,
                                   &tau_all,&tau_boundary)!=0) break;
        }
    }
    double boundary_opacity=(tau_all>0.0)?tau_boundary/tau_all:0.0;
    if(!isfinite(boundary_opacity))boundary_opacity=INFINITY;
    /* No boundary-stage rate/heating producer is present in the Stage-2A
     * assembler.  A non-zero adjacent-stage reservoir is therefore not
     * measurable from this lane and must fail closed rather than be inferred
     * negligible from opacity alone. */
    if(boundary.active)
        boundary_max=(n_elem>0.0)?boundary.m_outside/n_elem:INFINITY;
    int boundary_process_coverage=boundary.active ? boundary.assembly_pass :
        (boundary_max==0.0 && tau_boundary==0.0);
    /* Kramers is the authority lane's declared continuum, not a clamp/repair
     * guard firing.  A deletion remains fail-closed through target_fail. */
    int guard_firing_count=fire.target_fallback+fire.nstar_cap+
        fire.nonfinite_guard;
    int channel_inventory_fail =
        fire.channel_events[NLTE_EW_RAD_BB] == 0 ||
        fire.channel_events[NLTE_EW_COLL_BB] == 0 ||
        fire.channel_events[NLTE_EW_RAD_BF] == 0 ||
        fire.channel_events[NLTE_EW_COLL_BF] == 0 ||
        (gamma_dep && gamma_dep->nonthermal_ioniz_rate[shell] > 0.0 &&
         fire.channel_events[NLTE_EW_NT_BF] == 0);
    int p_elem_valid=solve_attempted && !solve_fail && nonfinite==0 &&
                     negative==0 && conservation<=1e-12 && se_res<=1e-10;
    int hot_cold_fail=hot_cold_required && !hot_cold_identical;
    int topology_gate_pass=!topology_fail && !hot_cold_fail &&
        guard_firing_count==0 &&
        target_fail==0 && bad_rate==0 &&
        !channel_inventory_fail &&
        coverage.ions_mapped==coverage.ions_expected &&
        coverage.sl_mapped==coverage.sl_expected &&
        coverage.full_mapped==coverage.full_expected &&
        coverage.lines_mapped==coverage.lines_expected &&
        coverage.continua_mapped==coverage.continua_expected &&
        coverage.targets_expected>0 && coverage.targets_mapped==coverage.targets_expected &&
        coverage.checksum_mismatch==0 && raw_rank==N-1 &&
        components==1 && zero_rows==0 && zero_cols==0 &&
        max_assembly_residual<=1e-12 &&
        (!boundary.active || (boundary.row_residual<=1e-12 &&
                              boundary.flux_residual<=1e-12));
    int numerical_gate_pass=!solve_fail && sd.rank==N &&
        max_assembly_residual<=1e-12 && sd.kappa<=1e12 && sd.pivot_growth<=1e8 &&
        se_res<=1e-10 && conservation<=1e-12 && nonfinite==0 && negative==0 &&
        perm_frac<=1e-10 &&
        (!boundary.active || (boundary.row_residual<=1e-12 &&
                              boundary.flux_residual<=1e-12));
    int boundary_gate_pass=boundary_max<=1e-8 && boundary_opacity<=1e-4 &&
                           boundary_process_coverage;
    int pass=topology_gate_pass&&numerical_gate_pass&&boundary_gate_pass;
    if (verdict_pass_out) *verdict_pass_out = pass;
    int commit_performed = pass && commit_requested;
    const char *commit_blocked_by = !commit_requested ? "not_requested" :
        (!topology_gate_pass ? "topology_gate" :
         (!numerical_gate_pass ? "numerical_gate" :
          (!boundary_gate_pass ? "boundary_gate" : "none")));
    const char *verdict=pass?"EW_PASS":
        (p_elem_valid&&topology_gate_pass&&numerical_gate_pass?
         "EW_VALID_P_ELEM_SCOPE_FAIL":
         (commit_requested?"EW_FAIL_FALLBACK_BASELINE":"EW_FAIL_SHADOW"));

    if(ew_gate.dump){
        ew_dump_identity(nlte,atom,slots,N,iter,Z,shell_label,checksum);
        ew_dump_raw(plane,N,iter,Z,shell_label);
        ew_dump_sparse_matrix("matrix_normalized",Anorm,b,NULL,NULL,N,iter,Z,shell_label);
        ew_dump_sparse_matrix("matrix_equilibrated",Aeq,beq,rs,cs,N,iter,Z,shell_label);
        char path[768]; FILE *fp=ew_open_dump(path,sizeof(path),iter,Z,shell_label,"solution");
        if(fp){fprintf(fp,"record_type,matrix_index,internal_stage,sl_id,global_level,raw_solution,restored_population,ion_total\n");for(int q=0;q<3;q++){double tot=0;for(int i=stage_start[q];i<stage_start[q+1];i++)tot+=x[i];for(int i=stage_start[q];i<stage_start[q+1];i++)fprintf(fp,"SL,%d,%d,%d,-1,%.17g,%.17g,%.17g\n",i,q+1,i-stage_start[q],x[i]/(cs[i]!=0.0?cs[i]:1.0),x[i],tot);int slot=slots[q],ss0=nlte->nlte_ion_super_offset[slot];for(int g=nlte->nlte_ion_level_offset[slot];g<nlte->nlte_ion_level_offset[slot+1];g++){int gl=nlte->nlte_to_global_level[g],local=stage_start[q]+nlte->fl_to_super[g]-ss0;double frac=nlte->within_sl_frac[(size_t)g*plasma->n_shells+shell];fprintf(fp,"FL,%d,%d,%d,%d,%.17g,%.17g,%.17g\n",local,q+1,nlte->fl_to_super[g]-ss0,gl,x[local]/(cs[local]!=0.0?cs[local]:1.0),x[local]*frac,tot);}}if(boundary.active)fprintf(fp,"BOUNDARY_MASS,%d,4,0,-1,%.17g,%.17g,%.17g\n",boundary.m_index,x[boundary.m_index]/(cs[boundary.m_index]!=0.0?cs[boundary.m_index]:1.0),x[boundary.m_index],x[boundary.m_index]);ew_close_dump(fp,path);}
        fp=ew_open_dump(path,sizeof(path),iter,Z,shell_label,"diagnostics");
        if(fp){fprintf(fp,"key,value\nN,%d\nion_coverage,%d/%d\nsl_coverage,%d/%d\nfull_level_coverage,%d/%d\nline_coverage,%d/%d\ncontinuum_coverage,%d/%d\nkramers_continuum_count,%d\ntarget_coverage,%d/%d\nfirst_bad_lower_global,%d\nfirst_bad_target_global,%d\nchecksum_mismatch,%d\nraw_rank,%d\nsolve_attempted,%d\np_elem_valid,%d\nnormalized_rank,%d\ngraph_components,%d\nzero_rows,%d\nzero_columns,%d\nassembled_target_expected,%d\nassembled_target_mapped,%d\nassembled_target_fail,%d\nbad_rate,%d\nchannel_inventory_fail,%d\nlegacy_guard_configured_count,%d\nlegacy_guard_firing_count,%d\nC48_projected_identity_level_count,%d\nkramers_fallback_firing_count,%d\ncontinuum_deletion_firing_count,%d\ntarget_fallback_firing_count,%d\nnstar_cap_firing_count,%d\nnonfinite_guard_firing_count,%d\nbf_estimator_bin_consumptions,%d\nbf_pref_J_bin_consumptions,%d\nbf_JEQB_bin_consumptions,%d\nindependent_assembly_residual_max,%.17g\nraw_singular_value_max,%.17g\nraw_singular_value_min,%.17g\nraw_kappa_2,%.17g\nequilibrated_singular_value_max,%.17g\nequilibrated_singular_value_min,%.17g\nkappa_2,%.17g\nrcond_1_solver_estimate,%.17g\npivot_growth,%.17g\nequilibration_iterations,%d\nrefinement_iterations,%d\nscaled_se_residual,%.17g\nscaled_se_t_ref_seconds,%.17g\nconservation_residual,%.17g\nraw_solution_min,%.17g\nnegative_error_bound,%.17g\nnegative_count,%d\nnonfinite_count,%d\npermutation_ion_fraction_max_abs,%.17g\nboundary_fraction_max,%.17g\nboundary_opacity_fraction,%.17g\nboundary_process_coverage,%d\nclamp_floor_freeze_anchor_repair_fallback_count,%d\nverdict,%s\n",N,coverage.ions_mapped,coverage.ions_expected,coverage.sl_mapped,coverage.sl_expected,coverage.full_mapped,coverage.full_expected,coverage.lines_mapped,coverage.lines_expected,coverage.continua_mapped,coverage.continua_expected,coverage.continua_kramers,coverage.targets_mapped,coverage.targets_expected,coverage.first_bad_lower,coverage.first_bad_target,coverage.checksum_mismatch,raw_rank,solve_attempted,p_elem_valid,sd.rank,components,zero_rows,zero_cols,target_expected,target_mapped,target_fail,bad_rate,channel_inventory_fail,guard_config_count,guard_firing_count,c48_projected_levels,fire.kramers_fallback,fire.continuum_deleted,fire.target_fallback,fire.nstar_cap,fire.nonfinite_guard,fire.bf_estimator_bins,fire.bf_pref_j_bins,fire.bf_jeqb_bins,max_assembly_residual,raw_smax,raw_smin,raw_kappa,sd.smax,sd.smin,sd.kappa,sd.rcond,sd.pivot_growth,sd.equil_iters,sd.refinement_iters,se_res,time_explosion,conservation,raw_solution_min,negative_error_bound,negative,nonfinite,perm_frac,boundary_max,boundary_opacity,boundary_process_coverage,guard_firing_count,verdict);fprintf(fp,"candidate_assembler_calls,%d\ncandidate_pair_owner_calls,%d\nsave_restore_calls,%d\nper_ion_pin_calls,%d\ntopstage_IV_calls,%d\nhot_cold_required,%d\nhot_cold_rebuilds,%d\nhot_cold_matrix_byte_identical,%d\nhot_cold_matrix_max_abs,%.17g\ncommit_requested,%d\ncommit_performed,%d\ncommit_blocked_by,%s\n",fire.candidate_assembler_calls,fire.candidate_pair_owner_calls,fire.save_restore_calls,fire.per_ion_pin_calls,fire.topstage_IV_calls,hot_cold_required,hot_cold_rebuilds,hot_cold_identical,hot_cold_max_abs,commit_requested,commit_performed,commit_blocked_by);fprintf(fp,"topology_gate_pass,%d\nnumerical_gate_pass,%d\nboundary_gate_pass,%d\n",topology_gate_pass,numerical_gate_pass,boundary_gate_pass);for(int c=0;c<NLTE_EW_N_CHANNELS;c++){fprintf(fp,"channel_colsum_%s,%.17g\n",ew_channel_name[c],colsum[c]);fprintf(fp,"channel_independent_assembly_residual_%s,%.17g\n",ew_channel_name[c],assembly_residual[c]);fprintf(fp,"channel_event_count_%s,%d\n",ew_channel_name[c],fire.channel_events[c]);}if(!solve_attempted)fprintf(fp,"pivot_status,not_attempted\nrefinement_status,not_attempted\n");for(int i=0;solve_attempted&&i<N;i++)fprintf(fp,"pivot_%d,%d\n",i,solve_piv[i]);for(int i=0;solve_attempted&&i<=sd.refinement_iters;i++)fprintf(fp,"refinement_residual_%d,%.17g\n",i,sd.refinement[i]);ew_close_dump(fp,path);}
        fp=ew_open_dump(path,sizeof(path),iter,Z,shell_label,"provenance");
        if(fp){fprintf(fp,"channel,row,column,source_identity,target_identity,aggregated_rate,units,producer,field_generation,target_route,probability_applied,full_to_sl_weight,aggregation\n");for(int c=0;c<NLTE_EW_N_CHANNELS;c++)for(int j=0;j<N;j++)for(int i=0;i<N;i++)if(i!=j&&EW(plane[c],N,i,j)>0.0)fprintf(fp,"%s,%d,%d,SL:%d,SL:%d,%.17g,s^-1,nlte_assemble_rate_matrix,frozen_cell_field,%s,%s,applied_at_capture,sum_of_transition_routes\n",ew_channel_name[c],i,j,j,i,EW(plane[c],N,i,j),(c==NLTE_EW_RAD_BF||c==NLTE_EW_COLL_BF)?"ma_rr_CSR":"direct_level_pair",(c==NLTE_EW_RAD_BF||c==NLTE_EW_COLL_BF)?"outside_once":"not_applicable");ew_close_dump(fp,path);}
        fp=ew_open_dump(path,sizeof(path),iter,Z,shell_label,"manifest");
        if(fp){fprintf(fp,"key,value\nschema,A2_17_MATERIAL_STATE_V1\nrun_id,iter%04d\nZ,%d\nshell,%d\nassembler,lumina_element_wide_cpu\nN,%d\nT_e,%.17g\nn_e,%.17g\nn_element,%.17g\nverdict,%s\n",iter,Z,shell_label,N,plasma->T_e[shell],plasma->n_electron[shell],n_elem,verdict);ew_close_dump(fp,path);}
        /* path still names the manifest created immediately above.  Append the
         * process-wide production meter rather than the frozen-only observer. */
        fp = fopen(path, "a");
        if (fp) {
            fprintf(fp, "production_bf_gpu_field_bypass_levels,%lu\n",
                    nlte_bf_gpu_field_bypass_count());
            if (boundary.active) {
                fprintf(fp,
                        "boundary_mass_stage,FeV_boundary_mass\n"
                        "boundary_mass_unknown_index,%d\n"
                        "boundary_mass_FeIV_sigma_coverage,%d/200\n"
                        "boundary_mass_route_source_coverage,%d/200\n"
                        "boundary_mass_FeV_to_FeVI,absent_declared_truncation\n"
                        "boundary_mass_DR_autoion,not_replicated_no_producer\n"
                        "boundary_mass_projection_pass,%d\n"
                        "boundary_mass_assembly_pass,%d\n",
                        boundary.m_index,boundary.sigma_count,
                        boundary.route_source_count,boundary.projection_pass,
                        boundary.assembly_pass);
            }
            ew_close_dump(fp, path);
        } else {
            fprintf(stderr, "[EW][DUMP-FAIL] append %s: %s\n",
                    path, strerror(errno));
            ew_dump_failed = 1;
        }
        if(boundary.active){
            fp=ew_open_dump(path,sizeof(path),iter,Z,shell_label,"boundary_mass");
            if(fp){
                double phi_forward=boundary.phi_forward[0]+boundary.phi_forward[1];
                double phi_reverse=boundary.phi_reverse[0]+boundary.phi_reverse[1];
                fprintf(fp,"run,iter,shell,Z,stage,atomic_checksum,route_count,valid_route_count,n_Fe_total,M_outside,M_V_before,M_V_after,sum_II,sum_III,sum_IV,conservation_residual,raw_solution_min,negative_error_bound,q_count,sum_q,q_min,q_max,q_checksum,Phi_IV_to_V_rad,Phi_V_to_IV_rad,Phi_IV_to_V_coll,Phi_V_to_IV_coll,Phi_IV_to_V_nonthermal,Phi_V_to_IV_nonthermal,Phi_DR_autoion,Phi_forward,Phi_reverse,Phi_net,matrix_event_flux_residual,boundary_row_residual,FeI_fraction,FeVIplus_fraction,right_boundary_producer_coverage,V_to_VI_contract,DR_contract,verdict\n");
                double fe1=0.0,fe6=0.0;
                int ip1=ew_find_ip(atom,26,0);
                if(ip1>=0&&n_elem>0.0)fe1=atom->ion_number_density[(size_t)ip1*plasma->n_shells+shell]/n_elem;
                for(int ip=0;ip<atom->n_ion_pops;ip++)if(atom->ion_pop_Z[ip]==26&&atom->ion_pop_stage[ip]>=5&&n_elem>0.0)fe6+=atom->ion_number_density[(size_t)ip*plasma->n_shells+shell]/n_elem;
                fprintf(fp,"iter%04d,%d,%d,26,FeV_boundary_mass,%016llx,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%016llx,%.17g,%.17g,%.17g,%.17g,0,0,0,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,0,absent_declared_truncation,not_replicated_no_producer,%s\n",
                    iter,iter,shell_label,(unsigned long long)checksum,
                    boundary.route_count,boundary.valid_route_count,n_elem,
                    boundary.m_outside,boundary.m_before,boundary.m_after,
                    boundary.sum_stage[0],boundary.sum_stage[1],boundary.sum_stage[2],
                    boundary.conservation_residual,raw_solution_min,
                    negative_error_bound,boundary.level_count,
                    boundary.q_sum,boundary.q_min,boundary.q_max,
                    (unsigned long long)boundary.q_checksum,
                    boundary.phi_forward[0],boundary.phi_reverse[0],
                    boundary.phi_forward[1],boundary.phi_reverse[1],
                    phi_forward,phi_reverse,phi_forward-phi_reverse,
                    boundary.flux_residual,boundary.row_residual,fe1,fe6,
                    pass?"PASS":"FAIL");
                ew_close_dump(fp, path);
            }
        }
    }

    if (ew_dump_failed) {
        fprintf(stderr, "[EW][I/O-FAIL] Z=%d s=%d artifacts incomplete; commit blocked\n",
                Z, shell_label);
        if (verdict_pass_out) *verdict_pass_out = 0;
        goto cleanup_fail;
    }

    fprintf(stderr,"[EW] Z=%d s=%d N=%d verdict=%s rank=%d/%d kappa=%.3e residual=%.3e targets=%d/%d\n",Z,shell_label,N,verdict,sd.rank,N,sd.kappa,se_res,coverage.targets_mapped,coverage.targets_expected);
    if(commit_performed){
        for(int q=0;q<3;q++){
            int slot=slots[q],ss0=nlte->nlte_ion_super_offset[slot];
            int l0=nlte->nlte_ion_level_offset[slot],l1=nlte->nlte_ion_level_offset[slot+1];
            for(int g=l0;g<l1;g++){
                int local=stage_start[q]+nlte->fl_to_super[g]-ss0;
                double frac=nlte->within_sl_frac[(size_t)g*plasma->n_shells+shell];
                nlte->nlte_level_populations[(size_t)g*plasma->n_shells+shell]=x[local]*frac;
            }
        }
        if(boundary.active){
            int ip=ew_find_ip(atom,26,4);
            if(ip>=0)
                atom->ion_number_density[(size_t)ip*plasma->n_shells+shell]=
                    x[boundary.m_index];
        }
    }
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++){
        free(plane[c]); free(expected_outflow[c]);
    }
    free(Araw);free(Anorm);free(Aeq);free(b);free(beq);free(x);free(rs);free(cs);free(solve_piv);
    ew_boundary_mass_free(&boundary);
    ew_private_view_free(&private_view);
    /* A completed scientific judgment is not a process error.  EW_PASS,
     * SCOPE_FAIL, and either gate FAIL are recorded above and all return zero;
     * negative RC is reserved for configuration, allocation, and I/O errors. */
    return 0;

cleanup_fail:
    for(int c=0;c<NLTE_EW_N_CHANNELS;c++){
        free(plane[c]); free(expected_outflow[c]);
    }
    free(Araw);free(Anorm);free(Aeq);free(b);free(beq);free(x);free(rs);free(cs);free(solve_piv);
    ew_boundary_mass_free(&boundary);
    ew_private_view_free(&private_view);
    return -1;
}

int nlte_element_wide_run(NLTEConfig *nlte, AtomicData *atom,
                           PlasmaState *plasma, OpacityState *opacity,
                           int Z, int shell, double time_explosion,
                           GammaDeposition *gamma_dep, int commit_requested) {
    return nlte_element_wide_run_status(
        nlte, atom, plasma, opacity, Z, shell, time_explosion,
        gamma_dep, commit_requested, NULL);
}

int nlte_element_wide_run_status(NLTEConfig *nlte, AtomicData *atom,
                           PlasmaState *plasma, OpacityState *opacity,
                           int Z, int shell, double time_explosion,
                           GammaDeposition *gamma_dep, int commit_requested,
                           int *verdict_pass_out) {
    return ew_run_impl(nlte,atom,plasma,opacity,Z,shell,shell,time_explosion,
                       gamma_dep,commit_requested,verdict_pass_out);
}

#ifdef LUMINA_FROZEN_ORACLE
int nlte_element_wide_run_labeled(NLTEConfig *nlte, AtomicData *atom,
                           PlasmaState *plasma, OpacityState *opacity,
                           int Z, int shell_index, int shell_label,
                           double time_explosion, GammaDeposition *gamma_dep) {
    int frozen_commit = 0;
    const char *v = getenv("LUMINA_EW_FROZEN_COMMIT");
    if (v && *v) {
        if (strcmp(v, "1") != 0) {
            fprintf(stderr,
                    "[EW][CONFIG-FAIL] LUMINA_EW_FROZEN_COMMIT must be exactly 1\n");
            return -1;
        }
        frozen_commit = 1;
    }
    return ew_run_impl(nlte,atom,plasma,opacity,Z,shell_index,shell_label,
                       time_explosion,gamma_dep,frozen_commit,NULL);
}
#endif
