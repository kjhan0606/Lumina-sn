#define _POSIX_C_SOURCE 200809L
#include "radiation_field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition, label) do { \
    if (!(condition)) { \
        fprintf(stderr, "A2_03_SELFTEST_FAIL %s line=%d\n", label, __LINE__); \
        failures++; \
    } \
} while (0)

static int close_relative(double actual, double expected, double tolerance)
{
    if (expected == 0.0) return actual == 0.0;
    return fabs(actual - expected) <= tolerance * fabs(expected);
}

int main(void)
{
    RadiationFieldShadow off_shadow;
    unsetenv("LUMINA_RADFIELD_SHADOW");
    CHECK(radiation_field_owner_init(&off_shadow, 2) == 0, "off-init-rc");
    CHECK(off_shadow.enabled == 1, "a2-04-canonical-default-enabled");
    CHECK(off_shadow.field.J_nu.values != NULL, "canonical-allocation");
    radiation_field_owner_free(&off_shadow);

    setenv("LUMINA_RADFIELD_SHADOW", "1", 1);
    RadiationFieldShadow shadow;
    CHECK(radiation_field_owner_init(&shadow, 2) == 0, "on-init");
    CHECK(shadow.enabled == 1, "on-enabled");
    CHECK(shadow.field.frequency_bin_edges.count ==
          (size_t)LUMINA_RADFIELD_N_BINS + 1,
          "selected-canonical-bin-grid");
    CHECK(strcmp(shadow.field.provenance.frequency_union_sha256,
                 LUMINA_RADFIELD_UNION_SHA256) == 0, "amended-union-binding");
    CHECK(strcmp(shadow.field.provenance.frequency_edge_sha256,
                 LUMINA_RADFIELD_EDGE_SHA256) == 0, "edge-hash-binding");

    const double v_inner[2] = {1.0e8, 2.0e8};
    const double v_outer[2] = {2.0e8, 3.0e8};
    CHECK(radiation_field_begin_mc(&shadow, v_inner, v_outer, 2,
                                          86400.0, 1) == 0, "generation-begin");
    CHECK(shadow.field.generation.required_generation == 0 &&
          shadow.field.generation.computed_generation == 0,
          "generation-private-before-commit");
    CHECK(shadow.field.validity.values[0] == RADIATION_FIELD_STALE,
          "precommit-stale");

    RadiationFieldAccumulator *local = radiation_field_accumulator_create(2);
    CHECK(local != NULL, "local-accumulator");
    const size_t bin = 137;
    double nu_lo = shadow.field.frequency_bin_edges.values[bin];
    double nu_hi = shadow.field.frequency_bin_edges.values[bin + 1];
    double nu_inside = sqrt(nu_lo * nu_hi);
    CHECK(radiation_field_accumulator_add(local, 0, nu_inside, 2.0) == 0,
          "measured-contribution-1");
    CHECK(radiation_field_accumulator_add(local, 0, nu_inside, 3.0) == 0,
          "measured-contribution-2");
    double exact_zero_nu = sqrt(
        shadow.field.frequency_bin_edges.values[bin + 2] *
        shadow.field.frequency_bin_edges.values[bin + 3]);
    CHECK(radiation_field_accumulator_add(local, 0,
          exact_zero_nu, 0.0) == 0,
          "exact-zero-contribution");
    CHECK(radiation_field_accumulator_add(local, 0,
          LUMINA_RADFIELD_NU_MIN_HZ * 0.5, 1.0) == 1,
          "out-of-grid-state-path");
    CHECK(radiation_field_accumulator_reduce(&shadow.accumulator, local) == 0,
          "thread-reduction");
    radiation_field_accumulator_free(local);

    const double volume[2] = {2.0, 4.0};
    const double time_simulation = 5.0;
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    request.producer = "A2_03_REGRESSION_MC";
    request.generation = 1;
    request.epoch = 86400.0;
    request.n_shells = 2;
    request.v_inner = v_inner;
    request.v_outer = v_outer;
    request.source_n_bins = LUMINA_RADFIELD_N_BINS;
    request.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
    request.source_count = shadow.accumulator.contribution_count;
    request.raw_path_length = shadow.accumulator.raw_path_length;
    request.volume = volume;
    request.time_simulation = time_simulation;
    request.out_of_grid_contribution_count =
        shadow.accumulator.out_of_grid_contribution_count;
    CHECK(radiation_field_commit(&shadow, &request) == 0, "commit");
    size_t measured = bin;
    size_t exact_zero = bin + 2;
    size_t unsampled = bin + 3;
    double expected = 5.0 /
        (4.0 * 3.14159265358979323846 * volume[0] * time_simulation *
         (nu_hi - nu_lo));
    CHECK(close_relative(shadow.field.J_nu.values[measured], expected, 1e-14),
          "bin-average-not-center-sample");
    CHECK(shadow.field.validity.values[measured] == RADIATION_FIELD_VALID,
          "measured-valid");
    CHECK(shadow.field.estimator_count_or_variance.count[measured] == 2,
          "actual-contribution-count");
    CHECK(shadow.field.validity.values[exact_zero] == RADIATION_FIELD_EXACT_ZERO &&
          shadow.field.J_nu.values[exact_zero] == 0.0 &&
          shadow.field.estimator_count_or_variance.count[exact_zero] == 1,
          "exact-zero-distinct-from-missing");
    CHECK(shadow.field.validity.values[unsampled] == RADIATION_FIELD_UNSAMPLED &&
          shadow.field.J_nu.values[unsampled] == 0.0 &&
          shadow.field.estimator_count_or_variance.count[unsampled] == 0,
          "unsampled-state-no-floor");
    CHECK(shadow.field.provenance.out_of_grid_contribution_count == 1,
          "out-of-grid-counted");
    CHECK(shadow.field.generation.required_generation == 1 &&
          shadow.field.generation.computed_generation == 1,
          "k-fresh-commit");
    CHECK(radiation_field_validate_owner(&shadow) == 0, "owner-valid");

    /* ORDER section 13 path 9: observer-frame injection must be rejected. */
    shadow.field.frame = RADIATION_FIELD_FRAME_OBSERVER;
    CHECK(radiation_field_validate_owner(&shadow) != 0,
          "negative-9-observer-frame-rejected");
    shadow.field.frame = RADIATION_FIELD_FRAME_SHELL_COMOVING;

    /* Path 10: J=0 and missing are not interchangeable. */
    RadiationFieldValidityState saved_validity = shadow.field.validity.values[measured];
    shadow.field.validity.values[measured] = RADIATION_FIELD_UNSAMPLED;
    CHECK(radiation_field_validate_owner(&shadow) != 0,
          "negative-10-valid-zero-missing-swap-rejected");
    shadow.field.validity.values[measured] = saved_validity;

    /* Path 11: an unsampled cell may not be filled with the historical floor. */
    shadow.field.J_nu.values[unsampled] = 1e-30;
    CHECK(radiation_field_validate_owner(&shadow) != 0,
          "negative-11-floor-fill-rejected");
    shadow.field.J_nu.values[unsampled] = 0.0;

    shadow.field.generation.required_generation = 2;
    CHECK(radiation_field_validate_owner(&shadow) != 0,
          "negative-stale-generation-rejected");
    shadow.field.generation.required_generation = 1;
    CHECK(radiation_field_validate_owner(&shadow) == 0,
          "owner-restored");

    radiation_field_owner_free(&shadow);
    unsetenv("LUMINA_RADFIELD_SHADOW");
    if (failures != 0) return 1;
    printf("A2_03_RADIATION_FIELD_SELFTEST PASS negative_9=PASS "
           "negative_10=PASS negative_11=PASS fields=10 bins=%d\n",
           LUMINA_RADFIELD_N_BINS);
    return 0;
}
