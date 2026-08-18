#include "cmf_exact_sliding.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(condition, label) do { if (!(condition)) { \
    fprintf(stderr, "CMF_EXACT_SLIDING_FAIL %s line=%d\n", label, __LINE__); \
    ++failures; \
} } while (0)

static double relative(double a, double b)
{
    return fabs(a - b) / (fabs(b) + 1.0e-30);
}

int main(void)
{
    enum { NS = 3, NB = 96 };
    const double dlog = 1.0e-3;
    const double texp = 1.0e6;
    double r_inner[NS] = {3.0e14, 4.0e14, 5.0e14};
    double r_outer[NS] = {4.0e14, 5.0e14, 6.0e14};
    double nu[NB], chi_tot[NS * NB], chi_es[NS * NB];
    double fixed[NS * NB], j_sliding[NS * NB], j_positive[NS * NB];
    double j_direct[NS * NB], j_envelope[NS * NB];
    double error_envelope[NS * NB];
    for (int b = 0; b < NB; ++b) nu[b] = 1.0e15 * exp((b + 0.5) * dlog);
    for (int s = 0; s < NS; ++s) {
        for (int b = 0; b < NB; ++b) {
            size_t i = (size_t)s * NB + (size_t)b;
            double ripple = 1.0 + 0.35 * sin(0.17 * b + 0.4 * s);
            chi_tot[i] = (2.0 + s) * 1.0e-15 * ripple;
            chi_es[i] = 0.27 * chi_tot[i];
            fixed[i] = (1.0 + 0.1 * s) * 1.0e-7 *
                       (1.0 + 0.2 * cos(0.11 * b));
            j_sliding[i] = j_positive[i] = j_direct[i] =
                j_envelope[i] = 0.8e-7;
        }
    }

    CMFExactReport sliding, positive, direct;
    CMFExactStatus rs = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_sliding, 80, 1.0e-11,
        CMF_EXACT_MODE_SLIDING, &sliding);
    CMFExactStatus rd = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_direct, 120, 1.0e-14,
        CMF_EXACT_MODE_DIRECT_REFERENCE, &direct);
    CMFExactStatus rp = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_positive, 80, 1.0e-11,
        CMF_EXACT_MODE_POSITIVE_SLIDING, &positive);
    CHECK(rs == CMF_EXACT_OK, "sliding-converged");
    CHECK(rd == CMF_EXACT_OK, "direct-converged");
    CHECK(rp == CMF_EXACT_OK, "positive-sliding-converged");
    CHECK(sliding.final_max_relative_change < sliding.tolerance,
          "sliding-residual");
    CHECK(sliding.final_max_absolute_change >= 0.0 &&
          isfinite(sliding.final_max_absolute_change),
          "sliding-absolute-change");
    CHECK(sliding.max_scattering_ratio > 0.0 &&
          sliding.max_scattering_ratio < 1.0,
          "sliding-contraction-bound");
    CHECK(sliding.fixed_point_absolute_error_bound >= 0.0 &&
          isfinite(sliding.fixed_point_absolute_error_bound),
          "sliding-finite-error-bound");
    CHECK(sliding.negative_recurrence_count == 0,
          "sliding-no-negative-recurrence");
    CHECK(sliding.max_characteristic_drift_bins > 0.5,
          "nontrivial-drift");
    double max_relative = 0.0, max_positive_relative = 0.0;
    for (size_t i = 0; i < (size_t)NS * NB; ++i) {
        double e = relative(j_sliding[i], j_direct[i]);
        if (e > max_relative) max_relative = e;
        e = relative(j_positive[i], j_direct[i]);
        if (e > max_positive_relative) max_positive_relative = e;
    }
    CHECK(max_relative < 2.0e-10, "sliding-direct-agreement");
    CHECK(max_positive_relative < 2.0e-10,
          "positive-sliding-direct-agreement");
    CHECK(positive.negative_recurrence_count == 0,
          "positive-sliding-no-negative-recurrence");

    double apply_lower[NS * NB], apply_nearest[NS * NB];
    double apply_upper[NS * NB];
    CHECK(cmf_exact_characteristic_apply_positive_bounds(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, j_positive,
              apply_lower, apply_nearest, apply_upper) == CMF_EXACT_OK,
          "positive-apply-outward-status");
    size_t strict_intervals = 0;
    double max_apply_residual = 0.0;
    for (size_t i = 0; i < (size_t)NS * NB; ++i) {
        CHECK(apply_lower[i] <= apply_nearest[i] &&
              apply_nearest[i] <= apply_upper[i],
              "positive-apply-outward-order");
        if (apply_lower[i] < apply_upper[i]) ++strict_intervals;
        double residual = fabs(apply_nearest[i] - j_positive[i]);
        if (residual > max_apply_residual) max_apply_residual = residual;
    }
    CHECK(strict_intervals > 0, "outward-interval-has-width");

    /* The certified componentwise supersolution is an error bound, not a
     * floor/cap on J.  An independently accumulated, tighter-tolerance direct
     * solve is a regression oracle (the supersolution check is the proof). */
    CMFExactReport envelope_report;
    CMFExactStatus re = cmf_exact_characteristic_solve_with_envelope(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_envelope, error_envelope, 5U,
        120, 1.0e-13, CMF_EXACT_MODE_POSITIVE_SLIDING,
        &envelope_report);
    CHECK(re == CMF_EXACT_OK, "componentwise-envelope-status");
    CHECK(envelope_report.componentwise_error_envelope_verified == 1,
          "componentwise-envelope-verified");
    CHECK(envelope_report.componentwise_error_seed_attempts >= 1U,
          "componentwise-envelope-seed-attempts");
    CHECK(envelope_report.componentwise_error_refinement_iterations == 5U,
          "componentwise-envelope-refined");
    CHECK(envelope_report.componentwise_residual_upper_max >= 0.0 &&
          isfinite(envelope_report.componentwise_residual_upper_max),
          "componentwise-residual-finite");
    size_t locally_tighter = 0;
    double max_envelope_ratio = 0.0;
    for (size_t i = 0; i < (size_t)NS * NB; ++i) {
        CHECK(error_envelope[i] >= 0.0 && isfinite(error_envelope[i]),
              "componentwise-envelope-finite");
        double observed = fabs(j_envelope[i] - j_direct[i]);
        CHECK(observed <= error_envelope[i],
              "componentwise-envelope-covers-direct-oracle");
        if (error_envelope[i] < envelope_report.componentwise_error_upper_max)
            ++locally_tighter;
        if (error_envelope[i] > 0.0) {
            double ratio = observed / error_envelope[i];
            if (ratio > max_envelope_ratio) max_envelope_ratio = ratio;
        }
    }
    CHECK(locally_tighter > 0, "componentwise-envelope-is-local");

    /* One-sided feasibility oracle: positivity of F(J)=b+KJ gives the exact
     * fixed point J*=b+KJ* >= b.  A lower-directed application at J=0 is
     * therefore a proof field, not a floor applied to the physical iterate. */
    double zero_j[NS * NB] = {0.0};
    double source_lower[NS * NB], source_nearest[NS * NB];
    double source_upper[NS * NB];
    CHECK(cmf_exact_characteristic_apply_positive_bounds(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, zero_j,
              source_lower, source_nearest, source_upper) == CMF_EXACT_OK,
          "source-only-lower-status");
    size_t strict_source_lower = 0;
    for (size_t i = 0; i < (size_t)NS * NB; ++i) {
        CHECK(source_lower[i] >= 0.0 &&
              source_lower[i] <= source_nearest[i] &&
              source_nearest[i] <= source_upper[i],
              "source-only-directed-order");
        CHECK(source_lower[i] <= j_direct[i],
              "source-only-lower-below-tight-fixed-point");
        if (source_lower[i] > 0.0 && source_lower[i] < j_direct[i])
            ++strict_source_lower;
    }
    CHECK(strict_source_lower > 0,
          "source-only-lower-is-finite-positive-proof");

    /* The certified operator is subtraction-free positive sliding only.
     * Reject other modes before mutating either the iterate or the bounds. */
    double rejected_j[NS * NB], rejected_bound[NS * NB];
    double rejected_j_before[NS * NB], rejected_bound_before[NS * NB];
    memcpy(rejected_j, j_positive, sizeof(rejected_j));
    for (size_t i = 0; i < (size_t)NS * NB; ++i)
        rejected_bound[i] = -17.0 - (double)i;
    memcpy(rejected_j_before, rejected_j, sizeof(rejected_j));
    memcpy(rejected_bound_before, rejected_bound, sizeof(rejected_bound));
    CHECK(cmf_exact_characteristic_solve_with_envelope(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, rejected_j, rejected_bound, 1U,
              20, 1.0e-8, CMF_EXACT_MODE_SLIDING, NULL) ==
              CMF_EXACT_INVALID_INPUT,
          "componentwise-envelope-rejects-subtractive-mode");
    CHECK(memcmp(rejected_j, rejected_j_before, sizeof(rejected_j)) == 0,
          "componentwise-envelope-reject-preserves-j");
    CHECK(memcmp(rejected_bound, rejected_bound_before,
                 sizeof(rejected_bound)) == 0,
          "componentwise-envelope-reject-preserves-bound");

    /* A mild signed line contribution may make the nonnegative electron-
     * scattering coefficient slightly larger than the still-positive total
     * extinction.  This is a gain iteration, not invalid input; convergence
     * of the global formal operator remains the qualification. */
    double chi_gain[NS * NB], es_gain[NS * NB], j_gain[NS * NB];
    memcpy(chi_gain, chi_tot, sizeof(chi_gain));
    memcpy(es_gain, chi_es, sizeof(es_gain));
    memcpy(j_gain, j_sliding, sizeof(j_gain));
    for (size_t i = 0; i < (size_t)NS * NB; ++i)
        es_gain[i] = 1.01 * chi_gain[i];
    CMFExactReport gain_report;
    CMFExactStatus gain_status = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_gain, es_gain, fixed, j_gain, 120, 1.0e-10,
        CMF_EXACT_MODE_SLIDING, &gain_report);
    CHECK(gain_status == CMF_EXACT_OK, "mild-signed-line-converged");
    CHECK(gain_report.max_scattering_ratio > 1.0,
          "mild-signed-line-gain-bound");
    CHECK(isinf(gain_report.fixed_point_absolute_error_bound),
          "mild-signed-line-unqualified-error-bound");
    CHECK(gain_report.negative_recurrence_count == 0,
          "mild-signed-line-no-negative-recurrence");
    for (size_t i = 0; i < (size_t)NS * NB; ++i)
        CHECK(isfinite(j_gain[i]) && j_gain[i] >= 0.0,
              "mild-signed-line-finite-nonnegative");

    /* Iteration exhaustion is a status, never an implicit success. */
    double j_short[NS * NB];
    memcpy(j_short, j_sliding, sizeof(j_short));
    CMFExactReport short_report;
    CMFExactStatus short_status = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_short, 2, 1.0e-30,
        CMF_EXACT_MODE_SLIDING, &short_report);
    CHECK(short_status == CMF_EXACT_NOT_CONVERGED, "cap-fail-closed");
    CHECK(short_report.iterations_used == 2, "cap-count");

    /* Invalid signed extinction is rejected before any formal sweep. */
    double bad_chi[NS * NB], j_bad[NS * NB];
    memcpy(bad_chi, chi_tot, sizeof(bad_chi));
    memcpy(j_bad, j_sliding, sizeof(j_bad));
    bad_chi[7] = -1.0;
    CMFExactReport bad_report;
    CHECK(cmf_exact_characteristic_solve(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              bad_chi, chi_es, fixed, j_bad, 20, 1.0e-8,
              CMF_EXACT_MODE_SLIDING, &bad_report) == CMF_EXACT_NONFINITE,
          "negative-opacity-rejected");

    if (failures) return 1;
    printf("CMF_EXACT_SLIDING_SELFTEST PASS max_rel=%.3e "
           "positive_max_rel=%.3e iterations=%d/%d/%d "
           "residual=%.3e apply_residual=%.3e envelope_ratio=%.3e "
           "source_lower_positive=%zu "
           "drift_bins=%.3f\n",
           max_relative, max_positive_relative, sliding.iterations_used,
           positive.iterations_used, direct.iterations_used,
           sliding.final_max_relative_change,
           max_apply_residual,
           max_envelope_ratio,
           strict_source_lower,
           sliding.max_characteristic_drift_bins);
    return 0;
}
