#ifndef LUMINA_CMF_EXACT_SLIDING_H
#define LUMINA_CMF_EXACT_SLIDING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CMF_EXACT_MODE_SLIDING = 1,
    CMF_EXACT_MODE_DIRECT_REFERENCE = 2,
    /* Subtraction-free O(1)-amortized window composition.  This mode is kept
     * separate until direct-oracle and full production gates are sealed. */
    CMF_EXACT_MODE_POSITIVE_SLIDING = 3
} CMFExactMode;

typedef enum {
    CMF_EXACT_OK = 0,
    CMF_EXACT_INVALID_INPUT = 1,
    CMF_EXACT_ALLOCATION_FAILED = 2,
    CMF_EXACT_NONFINITE = 3,
    CMF_EXACT_NEGATIVE_RECURRENCE = 4,
    CMF_EXACT_NOT_CONVERGED = 5,
    CMF_EXACT_ERROR_ENVELOPE_FAILED = 6
} CMFExactStatus;

typedef struct {
    CMFExactStatus status;
    int mode;
    int iterations_used;
    int iteration_cap;
    double tolerance;
    double final_max_relative_change;
    double final_max_absolute_change;
    /* Conservative infinity-norm qualification.  The formal lambda operator
     * has row sum <=1, so max(chi_es/chi_tot) is an upper bound on the fixed-
     * point iteration norm.  A finite error bound is published only below 1. */
    double max_scattering_ratio;
    double fixed_point_absolute_error_bound;
    double max_characteristic_drift_bins;
    uint64_t negative_recurrence_count;
    double first_negative_recurrence;
    size_t n_rays;
    size_t segment_slots;
    int componentwise_error_envelope_verified;
    size_t componentwise_error_seed_attempts;
    size_t componentwise_error_refinement_iterations;
    double componentwise_residual_upper_max;
    double componentwise_error_upper_min;
    double componentwise_error_upper_max;
} CMFExactReport;

/* Exact homologous drifting-characteristic formal solution.
 *
 * Frequency bins are ascending in nu (red -> blue), with constant dlognu.
 * The production mode is CMF_EXACT_MODE_SLIDING, O(1) work per output bin.
 * CMF_EXACT_MODE_DIRECT_REFERENCE is an independent O(beta) oracle intended
 * only for small-grid selftests.  J is both the ALI initial state and output.
 */
CMFExactStatus cmf_exact_characteristic_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report);

CMFExactStatus cmf_exact_characteristic_solve_with_envelope(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *componentwise_error_upper,
    size_t envelope_refinement_iterations,
    int iteration_cap, double tolerance, CMFExactMode mode,
    CMFExactReport *report);

/* One affine F(J)=b+KJ application using the positive sliding owner.  lower
 * and upper are outward binary64 enclosures of the same operation returned in
 * nearest.  This diagnostic API is the small-grid oracle for the production
 * residual/envelope integration; it does not mutate input_J. */
CMFExactStatus cmf_exact_characteristic_apply_positive_bounds(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper);

const char *cmf_exact_status_name(CMFExactStatus status);

#ifdef __cplusplus
}
#endif

#endif
