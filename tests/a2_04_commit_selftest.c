#include "radiation_field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition, label) do { \
    if (!(condition)) { \
        fprintf(stderr, "A2_04_COMMIT_FAIL %s line=%d\n", label, __LINE__); \
        failures++; \
    } \
} while (0)

static int same_public_state(const RadiationField *field, const double *values,
                             const RadiationFieldValidityState *validity,
                             const uint64_t *count,
                             RadiationFieldGeneration generation)
{
    size_t cells = field->J_nu.n_shells * field->J_nu.n_bins;
    return memcmp(field->J_nu.values, values, cells * sizeof(double)) == 0 &&
           memcmp(field->validity.values, validity,
                  cells * sizeof(*validity)) == 0 &&
           memcmp(field->estimator_count_or_variance.count, count,
                  cells * sizeof(uint64_t)) == 0 &&
           memcmp(&field->generation, &generation, sizeof(generation)) == 0;
}

int main(void)
{
    RadiationFieldShadow owner;
    CHECK(radiation_field_owner_init(&owner, 1) == 0, "init");
    const double v_inner[1] = {1.0e8};
    const double v_outer[1] = {2.0e8};
    const double volume[1] = {4.0};
    CHECK(radiation_field_begin_mc(&owner, v_inner, v_outer, 1,
                                          86400.0, 1) == 0, "mc-work-begin");
    size_t mc_bin = 321;
    double mc_nu = sqrt(owner.field.frequency_bin_edges.values[mc_bin] *
                        owner.field.frequency_bin_edges.values[mc_bin + 1]);
    CHECK(radiation_field_accumulator_add(&owner.accumulator, 0, mc_nu, 8.0) == 0,
          "mc-add");

    RadiationFieldCommitRequest mc;
    memset(&mc, 0, sizeof(mc));
    mc.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    mc.producer = "A2_04_SELFTEST_MC";
    mc.generation = 1;
    mc.epoch = 86400.0;
    mc.n_shells = 1;
    mc.v_inner = v_inner;
    mc.v_outer = v_outer;
    mc.source_n_bins = LUMINA_RADFIELD_N_BINS;
    mc.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
    mc.source_count = owner.accumulator.contribution_count;
    mc.raw_path_length = owner.accumulator.raw_path_length;
    mc.volume = volume;
    mc.time_simulation = 2.0;
    CHECK(radiation_field_commit(&owner, &mc) == 0, "mc-common-commit");
    CHECK(owner.field.validity.values[mc_bin] == RADIATION_FIELD_VALID &&
          owner.field.estimator_count_or_variance.count[mc_bin] == 1,
          "mc-valid-count");
    CHECK(owner.field.validity.values[mc_bin + 1] == RADIATION_FIELD_UNSAMPLED &&
          owner.field.J_nu.values[mc_bin + 1] == 0.0,
          "mc-unsampled-no-floor");

    size_t cells = LUMINA_RADFIELD_N_BINS;
    double *saved_values = (double *)malloc(cells * sizeof(double));
    RadiationFieldValidityState *saved_validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*saved_validity));
    uint64_t *saved_count = (uint64_t *)malloc(cells * sizeof(uint64_t));
    CHECK(saved_values && saved_validity && saved_count, "snapshot-allocation");
    memcpy(saved_values, owner.field.J_nu.values, cells * sizeof(double));
    memcpy(saved_validity, owner.field.validity.values,
           cells * sizeof(*saved_validity));
    memcpy(saved_count, owner.field.estimator_count_or_variance.count,
           cells * sizeof(uint64_t));
    RadiationFieldGeneration generation = owner.field.generation;

    RadiationFieldCommitRequest wrong_generation = mc;
    wrong_generation.generation = 3;
    CHECK(radiation_field_commit(&owner, &wrong_generation) != 0,
          "negative-4-generation-gap-rejected");
    CHECK(same_public_state(&owner.field, saved_values, saved_validity,
                            saved_count, generation),
          "failed-commit-is-atomic");

    RadiationFieldCommitRequest double_normalized = mc;
    double_normalized.generation = 2;
    double_normalized.source_J_nu = saved_values;
    CHECK(radiation_field_commit(&owner, &double_normalized) != 0,
          "negative-7-double-normalization-form-rejected");
    CHECK(same_public_state(&owner.field, saved_values, saved_validity,
                            saved_count, generation),
          "double-normalization-no-publish");

    size_t first = 100, last = 3900, source_bins = last - first;
    double *source_edges = (double *)malloc((source_bins + 1) * sizeof(double));
    double *source_values = (double *)malloc(source_bins * sizeof(double));
    RadiationFieldValidityState *source_state =
        (RadiationFieldValidityState *)malloc(source_bins * sizeof(*source_state));
    CHECK(source_edges && source_values && source_state, "cmf-work-allocation");
    memcpy(source_edges, &owner.field.frequency_bin_edges.values[first],
           (source_bins + 1) * sizeof(double));
    for (size_t b = 0; b < source_bins; ++b) {
        source_values[b] = 2.0;
        source_state[b] = RADIATION_FIELD_VALID;
    }
    source_values[17] = 0.0;
    source_state[17] = RADIATION_FIELD_EXACT_ZERO;

    RadiationFieldCommitRequest cmf;
    memset(&cmf, 0, sizeof(cmf));
    cmf.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    cmf.producer = "A2_04_SELFTEST_PURE_CMFGEN";
    cmf.generation = 2;
    cmf.epoch = 86400.0;
    cmf.n_shells = 1;
    cmf.v_inner = v_inner;
    cmf.v_outer = v_outer;
    cmf.source_n_bins = source_bins;
    cmf.source_frequency_bin_edges = source_edges;
    cmf.source_J_nu = source_values;
    cmf.source_validity = source_state;
    cmf.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    CHECK(radiation_field_commit(&owner, &cmf) == 0,
          "pure-cmfgen-common-commit");
    CHECK(owner.field.generation.computed_generation == 2 &&
          owner.field.generation.required_generation == 2,
          "generation-only-commit-advances");
    CHECK(owner.field.validity.values[first - 1] == RADIATION_FIELD_OUT_OF_GRID &&
          owner.field.J_nu.values[first - 1] == 0.0,
          "outside-grid-explicit");
    CHECK(owner.field.validity.values[first] == RADIATION_FIELD_VALID &&
          owner.field.J_nu.values[first] == 2.0,
          "conservative-bin-average");
    CHECK(owner.field.validity.values[first + 17] == RADIATION_FIELD_EXACT_ZERO &&
          owner.field.J_nu.values[first + 17] == 0.0,
          "deterministic-exact-zero");

    memcpy(saved_values, owner.field.J_nu.values, cells * sizeof(double));
    memcpy(saved_validity, owner.field.validity.values,
           cells * sizeof(*saved_validity));
    memcpy(saved_count, owner.field.estimator_count_or_variance.count,
           cells * sizeof(uint64_t));
    generation = owner.field.generation;
    RadiationFieldCommitRequest planck = cmf;
    planck.generation = 3;
    planck.provenance_kind =
        RADIATION_FIELD_PROVENANCE_DILUTE_PLANCK_LEGACY_APPROXIMATION;
    CHECK(radiation_field_commit(&owner, &planck) != 0,
          "negative-1-planck-overwrite-rejected");
    CHECK(same_public_state(&owner.field, saved_values, saved_validity,
                            saved_count, generation),
          "planck-rejection-preserves-owner");

    free(source_edges); free(source_values); free(source_state);
    free(saved_values); free(saved_validity); free(saved_count);
    radiation_field_owner_free(&owner);
    if (failures) return 1;
    printf("A2_04_COMMIT_SELFTEST PASS common_callers=MC,CMFGEN "
           "negative_1=PASS negative_4=PASS negative_7=PASS "
           "unsampled_floor=0 out_of_grid=EXPLICIT generation_atomic=PASS\n");
    return 0;
}
