#include "radiation_field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures;
#define CHECK(c, l) do { if (!(c)) { \
    fprintf(stderr, "A2_06_DUAL_COMMIT_FAIL %s line=%d\n", l, __LINE__); \
    failures++; } } while (0)

#define QH "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#define EH "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
#define PH "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

int main(void)
{
    const double EPOCH = 1683072.0;
    size_t NS = 2, NQ = 3;
    size_t cells = NS * (size_t)LUMINA_RADFIELD_N_BINS;
    size_t lcells = NQ * NS;

    RadiationFieldOwner owner;
    CHECK(radiation_field_owner_init(&owner, NS) == 0, "init");
    double v_inner[2] = {1.0e8, 1.5e8}, v_outer[2] = {1.5e8, 2.0e8};
    double volume[2] = {3.0e40, 5.0e40};

    /* MC accumulation: one contribution in a mid bin per shell */
    CHECK(radiation_field_begin_mc(&owner, v_inner, v_outer, NS, EPOCH, 1) == 0,
          "begin");
    double nu_mid = sqrt(owner.field.frequency_bin_edges.values[2000] *
                         owner.field.frequency_bin_edges.values[2001]);
    for (size_t s = 0; s < NS; s++)
        CHECK(radiation_field_accumulator_add(&owner.accumulator, s, nu_mid,
                                              5.0) == 0, "acc");

    uint64_t line_id[3] = {11, 42, 900};
    double lsum[6] = {2.0, 0.0, 0.0, 1.0, 4.0, 0.0};
    double lsq[6] = {1.5, 0.0, 0.0, 0.7, 9.0, 0.0};
    uint64_t lct[6] = {3, 0, 2, 1, 5, 0};
    uint64_t NPK = 100;

    RadiationFieldCommitRequest req;
    memset(&req, 0, sizeof(req));
    req.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    req.producer = "A2_06_SELFTEST_MC";
    req.generation = 1;
    req.epoch = EPOCH;
    req.n_shells = NS;
    req.v_inner = v_inner;
    req.v_outer = v_outer;
    req.source_n_bins = LUMINA_RADFIELD_N_BINS;
    req.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
    req.source_count = owner.accumulator.contribution_count;
    req.raw_path_length = owner.accumulator.raw_path_length;
    req.volume = volume;
    req.time_simulation = 2.0;
    req.line_n = NQ;
    req.line_id = line_id;
    req.line_q_set_hash = QH;
    req.line_profile_id = 1;
    req.line_profile_hash = PH;
    req.line_sum = lsum;
    req.line_sumsq = lsq;
    req.line_count = lct;
    req.line_n_packets = NPK;
    CHECK(radiation_field_commit(&owner, &req) == 0, "commit1");

    /* view + lookup values against direct formulas */
    LineJbarView view;
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 1, QH, 1, PH, &view) ==
          LINE_JBAR_VIEW_OK, "view-ok");
    LineJbarValue lv;
    double norm0 = 1.0 / (4.0 * M_PI * volume[0] * 2.0);
    CHECK(line_jbar_lookup(&view, 0, 11, &lv) == 0, "lk1");
    CHECK(line_jbar_lookup_index(&view, 0, 0, 11, &lv) == 0 &&
          fabs(lv.jbar - norm0 * 2.0) < 1e-18 + 1e-12 * lv.jbar,
          "lk1-index");
    CHECK(line_jbar_lookup_index(&view, 0, 0, 42, &lv) == -2,
          "lk-index-id-mismatch");
    CHECK(line_jbar_lookup_index(&view, 0, 3, 11, &lv) == -2,
          "lk-index-range");
    CHECK(line_jbar_lookup_index(&view, 0, 0, 11, &lv) == 0,
          "lk-index-restore");
    CHECK(fabs(lv.jbar - norm0 * 2.0) < 1e-18 + 1e-12 * lv.jbar, "lk1-val");
    CHECK(lv.validity == LINE_JBAR_VALID && lv.count == 3, "lk1-meta");
    {
        double N = (double)NPK;
        double s2 = (1.5 - 2.0 * 2.0 / N) / (N - 1.0);
        double want_se = norm0 * sqrt(N * s2);
        CHECK(fabs(lv.se - want_se) < 1e-25 + 1e-12 * want_se, "lk1-se");
    }
    CHECK(line_jbar_lookup(&view, 1, 11, &lv) == 0 &&
          lv.validity == LINE_JBAR_UNSAMPLED && lv.jbar == 0.0, "lk-unsampled");
    CHECK(line_jbar_lookup(&view, 0, 42, &lv) == 0 &&
          lv.validity == LINE_JBAR_EXACT_ZERO && lv.count == 2, "lk-zero");
    CHECK(line_jbar_lookup(&view, 0, 777, &lv) == -2, "lk-miss");

    /* view error codes */
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 2, QH, 1, PH, &view) ==
          LINE_JBAR_VIEW_STALE_GENERATION, "v-stale");
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 1, PH, 1, PH, &view) ==
          LINE_JBAR_VIEW_QHASH, "v-qhash");
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 1, QH, 7, PH, &view) ==
          LINE_JBAR_VIEW_PROFILE, "v-profile");
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH + 1, NS, 1, QH, 1, PH, &view)
          == LINE_JBAR_VIEW_EPOCH_SHELLS, "v-epoch");
    CHECK(radiation_field_line_jbar_view(
              NULL, EPOCH, NS, 1, QH, 1, PH, &view) ==
          LINE_JBAR_VIEW_DISABLED, "v-null");

    /* ---- partial-commit injections: NOTHING public may change ---- */
    double *snap_j = malloc(cells * sizeof(double));
    double *snap_l = malloc(lcells * sizeof(double));
    memcpy(snap_j, owner.field.J_nu.values, cells * sizeof(double));
    memcpy(snap_l, owner.line_jbar_cache.jbar_value, lcells * sizeof(double));
    uint64_t snap_gen = owner.field.generation.computed_generation;
    uint64_t snap_lgen = owner.line_jbar_cache.generation.computed_generation;

    CHECK(radiation_field_begin_mc(&owner, v_inner, v_outer, NS, EPOCH, 2) == 0,
          "begin2");
    for (size_t s = 0; s < NS; s++)
        CHECK(radiation_field_accumulator_add(&owner.accumulator, s, nu_mid,
                                              7.0) == 0, "acc2");
    req.generation = 2;

    /* (a) J OK + line bad: negative sum */
    double bad_sum[6]; memcpy(bad_sum, lsum, sizeof(bad_sum));
    bad_sum[0] = -1.0;
    req.line_sum = bad_sum;
    CHECK(radiation_field_commit(&owner, &req) != 0, "inj-line-bad");
    CHECK(memcmp(owner.field.J_nu.values, snap_j, cells * sizeof(double)) == 0,
          "inj-a-j-frozen");
    CHECK(memcmp(owner.line_jbar_cache.jbar_value, snap_l,
                 lcells * sizeof(double)) == 0, "inj-a-l-frozen");
    CHECK(owner.field.generation.computed_generation == snap_gen &&
          owner.line_jbar_cache.generation.computed_generation == snap_lgen,
          "inj-a-gen-frozen");

    /* (a2) nonnegative moments with an impossible negative variance are not
     * clamped to zero; the whole field+line candidate remains unpublished. */
    double bad_sq[6]; memcpy(bad_sq, lsq, sizeof(bad_sq));
    bad_sq[0] = 0.0;
    req.line_sum = lsum;
    req.line_sumsq = bad_sq;
    CHECK(radiation_field_commit(&owner, &req) != 0,
          "inj-negative-variance-no-clamp");
    CHECK(owner.field.generation.computed_generation == snap_gen &&
          owner.line_jbar_cache.generation.computed_generation == snap_lgen,
          "inj-negative-variance-atomic");
    req.line_sumsq = lsq;

    /* (b) line OK + J bad: wrong generation */
    req.line_sum = lsum;
    req.generation = 5;
    CHECK(radiation_field_commit(&owner, &req) != 0, "inj-j-bad");
    CHECK(owner.field.generation.computed_generation == snap_gen &&
          owner.line_jbar_cache.generation.computed_generation == snap_lgen,
          "inj-b-gen-frozen");

    /* (c) accumulation latch refuses the commit */
    req.generation = 2;
    req.line_error_latch = 1;
    CHECK(radiation_field_commit(&owner, &req) != 0, "inj-latch");
    req.line_error_latch = 0;

    /* (d) clean retry then succeeds and views advance together */
    CHECK(radiation_field_commit(&owner, &req) == 0, "commit2");
    CHECK(owner.field.generation.computed_generation == 2 &&
          owner.line_jbar_cache.generation.computed_generation == 2,
          "dual-advance");
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 2, QH, 1, PH, &view) ==
          LINE_JBAR_VIEW_OK, "view2");
    /* previous generation is now refused (negative control 3.4-1) */
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 1, QH, 1, PH, &view) ==
          LINE_JBAR_VIEW_STALE_GENERATION, "old-gen-refused");

    /* Deterministic replay carries a certified absolute error upper in the
     * canonical `se` slot.  A negative proof bound rejects continuum+line
     * atomically before generation 3 is visible. */
    double det_jbar[6] = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    double det_error[6] = {1e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12};
    int32_t det_validity[6] = {
        LINE_JBAR_VALID, LINE_JBAR_VALID, LINE_JBAR_VALID,
        LINE_JBAR_VALID, LINE_JBAR_VALID, LINE_JBAR_VALID
    };
    RadiationFieldCommitRequest dreq;
    memset(&dreq, 0, sizeof(dreq));
    dreq.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    dreq.producer = "A2_06_SELFTEST_DETERMINISTIC";
    dreq.generation = 3;
    dreq.epoch = EPOCH;
    dreq.n_shells = NS;
    dreq.v_inner = v_inner;
    dreq.v_outer = v_outer;
    dreq.source_n_bins = LUMINA_RADFIELD_N_BINS;
    dreq.source_frequency_bin_edges = owner.field.frequency_bin_edges.values;
    dreq.source_J_nu = owner.field.J_nu.values;
    dreq.source_validity = owner.field.validity.values;
    dreq.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    dreq.line_n = NQ;
    dreq.line_id = line_id;
    dreq.line_q_set_hash = QH;
    dreq.line_profile_id = 1;
    dreq.line_profile_hash = PH;
    dreq.line_provenance_kind =
        RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL;
    dreq.line_producer = LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER;
    dreq.line_jbar = det_jbar;
    dreq.line_error_upper = det_error;
    dreq.line_validity = det_validity;
    double saved_error = det_error[2];
    det_error[2] = -1.0;
    CHECK(radiation_field_commit(&owner, &dreq) != 0,
          "det-negative-error-rejected");
    CHECK(owner.field.generation.computed_generation == 2 &&
          owner.line_jbar_cache.generation.computed_generation == 2,
          "det-negative-error-atomic");
    det_error[2] = saved_error;
    CHECK(radiation_field_commit(&owner, &dreq) == 0,
          "det-error-bound-commit");
    CHECK(radiation_field_line_jbar_view(
              &owner, EPOCH, NS, 3, QH, 1, PH, &view) ==
              LINE_JBAR_VIEW_OK,
          "det-error-bound-view");
    CHECK(line_jbar_lookup(&view, 0, 42, &lv) == 0 &&
          lv.jbar == det_jbar[2] && lv.se == det_error[2] &&
          lv.count == 0,
          "det-error-bound-canonical-se");

    free(snap_j); free(snap_l);
    radiation_field_owner_free(&owner);

    /* ---- Q_E single-cache schema with sparse Q_g rate view ---- */
    RadiationFieldOwner eowner;
    CHECK(radiation_field_owner_init(&eowner, NS) == 0, "energy-init");
    CHECK(radiation_field_begin_mc(&eowner, v_inner, v_outer, NS,
                                   EPOCH, 1) == 0, "energy-begin");
    for (size_t s = 0; s < NS; ++s)
        CHECK(radiation_field_accumulator_add(&eowner.accumulator, s,
                                              nu_mid, 5.0) == 0,
              "energy-continuum-acc");
    uint64_t energy_id[4] = {11, 42, 77, 900};
    uint64_t rate_id[2] = {42, 900};
    double esum[8] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0};
    double esq[8] = {1.0, 0.0, 4.0, 0.0, 9.0, 0.0, 16.0, 0.0};
    uint64_t ect[8] = {1, 0, 1, 0, 1, 0, 1, 0};
    RadiationFieldCommitRequest ereq;
    memset(&ereq, 0, sizeof(ereq));
    ereq.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    ereq.producer = "A2_06_SELFTEST_QE_MC";
    ereq.generation = 1;
    ereq.epoch = EPOCH;
    ereq.n_shells = NS;
    ereq.v_inner = v_inner;
    ereq.v_outer = v_outer;
    ereq.source_n_bins = LUMINA_RADFIELD_N_BINS;
    ereq.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
    ereq.source_count = eowner.accumulator.contribution_count;
    ereq.raw_path_length = eowner.accumulator.raw_path_length;
    ereq.volume = volume;
    ereq.time_simulation = 2.0;
    ereq.line_n = 4;
    ereq.line_id = energy_id;
    ereq.line_q_set_hash = EH;
    ereq.line_set_kind = LINE_JBAR_SET_ENERGY_DOMAIN;
    ereq.line_rate_graph_n = 2;
    ereq.line_rate_graph_id = rate_id;
    ereq.line_rate_graph_hash = QH;
    ereq.line_profile_id = 1;
    ereq.line_profile_hash = PH;
    ereq.line_sum = esum;
    ereq.line_sumsq = esq;
    ereq.line_count = ect;
    ereq.line_n_packets = NPK;
    CHECK(radiation_field_commit(&eowner, &ereq) == 0,
          "energy-atomic-commit");
    CHECK(eowner.line_jbar_cache.set_kind ==
              LINE_JBAR_SET_ENERGY_DOMAIN &&
          eowner.line_n_compact == 4 &&
          eowner.line_rate_graph_n_compact == 2 &&
          eowner.line_rate_graph_cache_index[0] == 1 &&
          eowner.line_rate_graph_cache_index[1] == 3,
          "energy-owner-schema");

    LineJbarView energy_view, rate_view;
    CHECK(radiation_field_line_jbar_energy_view(
              &eowner, EPOCH, NS, 1, EH, 1, PH, &energy_view) ==
          LINE_JBAR_VIEW_OK && energy_view.n_lines == 4 &&
          energy_view.cache_n_lines == 4 &&
          energy_view.cache_index == NULL,
          "energy-checked-view");
    CHECK(radiation_field_line_jbar_rate_view(
              &eowner, EPOCH, NS, 1, QH, 1, PH, &rate_view) ==
          LINE_JBAR_VIEW_OK && rate_view.n_lines == 2 &&
          rate_view.cache_n_lines == 4 && rate_view.cache_index &&
          rate_view.cache_index[0] == 1 && rate_view.cache_index[1] == 3,
          "rate-subset-view");
    CHECK(line_jbar_lookup(&rate_view, 0, 42, &lv) == 0 &&
          fabs(lv.jbar - 2.0 * norm0) < 1e-18 + 1e-12 * lv.jbar,
          "rate-subset-map-value");
    CHECK(line_jbar_lookup(&rate_view, 0, 11, &lv) == -2,
          "rate-view-hides-energy-only-line");
    CHECK(line_jbar_lookup(&energy_view, 0, 11, &lv) == 0,
          "energy-view-sees-energy-line");
    CHECK(radiation_field_line_jbar_view(
              &eowner, EPOCH, NS, 1, QH, 1, PH, &rate_view) ==
          LINE_JBAR_VIEW_OK && rate_view.n_lines == 2,
          "compatibility-view-is-rate-view");
    CHECK(radiation_field_line_jbar_energy_view(
              &eowner, EPOCH, NS, 1, QH, 1, PH, &energy_view) ==
          LINE_JBAR_VIEW_QHASH,
          "energy-hash-mismatch");
    CHECK(radiation_field_line_jbar_rate_view(
              &eowner, EPOCH, NS, 1, EH, 1, PH, &rate_view) ==
          LINE_JBAR_VIEW_QHASH,
          "rate-hash-mismatch");

    /* A seeded Q_g line absent from Q_E rejects the whole next commit. */
    CHECK(radiation_field_begin_mc(&eowner, v_inner, v_outer, NS,
                                   EPOCH, 2) == 0, "energy-begin2");
    for (size_t s = 0; s < NS; ++s)
        CHECK(radiation_field_accumulator_add(&eowner.accumulator, s,
                                              nu_mid, 7.0) == 0,
              "energy-continuum-acc2");
    uint64_t missing_rate_id[2] = {42, 901};
    ereq.generation = 2;
    ereq.line_rate_graph_id = missing_rate_id;
    CHECK(radiation_field_commit(&eowner, &ereq) != 0,
          "energy-missing-rate-line-rejected");
    CHECK(eowner.field.generation.computed_generation == 1 &&
          eowner.line_jbar_cache.generation.computed_generation == 1 &&
          eowner.line_rate_graph_ids_compact[1] == 900,
          "energy-subset-failure-atomic");
    radiation_field_owner_free(&eowner);
    if (failures) {
        fprintf(stderr, "A2_06_DUAL_COMMIT_SELFTEST FAIL failures=%d\n", failures);
        return 1;
    }
    printf("A2_06_DUAL_COMMIT_SELFTEST PASS\n");
    return 0;
}
