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
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH, NS, 1, QH, 1, &view) ==
          LINE_JBAR_VIEW_OK, "view-ok");
    LineJbarValue lv;
    CHECK(line_jbar_lookup(&view, 0, 11, &lv) == 0, "lk1");
    double norm0 = 1.0 / (4.0 * M_PI * volume[0] * 2.0);
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
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH, NS, 2, QH, 1, &view) ==
          LINE_JBAR_VIEW_STALE_GENERATION, "v-stale");
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH, NS, 1, PH, 1, &view) ==
          LINE_JBAR_VIEW_QHASH, "v-qhash");
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH, NS, 1, QH, 7, &view) ==
          LINE_JBAR_VIEW_PROFILE, "v-profile");
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH + 1, NS, 1, QH, 1, &view)
          == LINE_JBAR_VIEW_EPOCH_SHELLS, "v-epoch");
    CHECK(radiation_field_line_jbar_view(NULL, EPOCH, NS, 1, QH, 1, &view) ==
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
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH, NS, 2, QH, 1, &view) ==
          LINE_JBAR_VIEW_OK, "view2");
    /* previous generation is now refused (negative control 3.4-1) */
    CHECK(radiation_field_line_jbar_view(&owner, EPOCH, NS, 1, QH, 1, &view) ==
          LINE_JBAR_VIEW_STALE_GENERATION, "old-gen-refused");

    free(snap_j); free(snap_l);
    radiation_field_owner_free(&owner);
    if (failures) {
        fprintf(stderr, "A2_06_DUAL_COMMIT_SELFTEST FAIL failures=%d\n", failures);
        return 1;
    }
    printf("A2_06_DUAL_COMMIT_SELFTEST PASS\n");
    return 0;
}
