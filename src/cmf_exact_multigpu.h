#ifndef LUMINA_CMF_EXACT_MULTIGPU_H
#define LUMINA_CMF_EXACT_MULTIGPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ray-sharded CUDA exact owner.  The epoch envelope entry point is wired to
 * the fine-grid production caller only through its explicit device-count
 * gate; unset/zero retains the serial owner.  Both publication buffers are
 * transactional and the directed componentwise certificate is mandatory. */
typedef enum {
    CMF_MGPU_OK = 0,
    CMF_MGPU_INVALID_INPUT = 1,
    CMF_MGPU_CUDA_UNAVAILABLE = 2,
    CMF_MGPU_INSUFFICIENT_DEVICES = 3,
    CMF_MGPU_ALLOCATION_FAILED = 4,
    CMF_MGPU_CUDA_FAILURE = 5,
    CMF_MGPU_NONFINITE = 6,
    CMF_MGPU_NOT_CONVERGED = 7,
    CMF_MGPU_NEGATIVE_RECURRENCE = 8,
    CMF_MGPU_ERROR_ENVELOPE_FAILED = 9
} CMFMultiGPUStatus;

typedef enum {
    CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS = 0,
    CMF_MGPU_PARTITION_EQUAL_RAYS = 1
} CMFMultiGPUPartitionMode;

typedef struct {
    int block_size;
    int epoch_batch_cardinality;
    int direct_replay_max_window;
} CMFMultiGPUEpochSchedule;

/* The production comparisons currently use at most four devices.  Keep a
 * bounded diagnostic surface large enough for ordinary single-node CUDA
 * systems without making report publication depend on dynamic allocation. */
#define CMF_MGPU_REPORT_MAX_DEVICES 32

typedef struct {
    CMFMultiGPUStatus status;
    int visible_devices;
    int devices_used;
    int iterations_used;
    int iteration_cap;
    double tolerance;
    double final_max_relative_change;
    double final_max_absolute_change;
    /* Same infinity-norm qualification published by the serial exact owner.
     * These are diagnostics/certificates, never numerical repair controls. */
    double max_scattering_ratio;
    double fixed_point_absolute_error_bound;
    double max_characteristic_drift_bins;
    size_t n_rays;
    size_t owned_rays;
    size_t computed_rays_with_halos;
    size_t max_device_allocated_bytes;
    size_t total_device_allocated_bytes;
    int deterministic_host_reduction;
    int positive_sliding;
    int epoch_frequency_parallel;
    int epoch_block_size;
    int epoch_batch_cardinality;
    int epoch_direct_replay_max_window;
    size_t epoch_workspace_bytes_per_device_max;
    size_t max_positive_window_bins;
    int componentwise_error_envelope_verified;
    size_t componentwise_error_seed_attempts;
    size_t componentwise_error_refinement_iterations;
    double componentwise_residual_upper_max;
    double componentwise_error_upper_min;
    double componentwise_error_upper_max;
    size_t persistent_context_initializations;
    size_t persistent_bound_applications;
    size_t persistent_upper_operator_applications;
    int weighted_contiguous_ray_partition;
    size_t min_owned_ray_segment_work;
    size_t max_owned_ray_segment_work;
    size_t min_computed_ray_segment_work;
    size_t max_computed_ray_segment_work;
    int device_partition_count;
    int device_ray_begin[CMF_MGPU_REPORT_MAX_DEVICES];
    int device_ray_end[CMF_MGPU_REPORT_MAX_DEVICES];
    size_t device_owned_segment_work[CMF_MGPU_REPORT_MAX_DEVICES];
    size_t device_computed_segment_work[CMF_MGPU_REPORT_MAX_DEVICES];
    size_t device_allocated_bytes[CMF_MGPU_REPORT_MAX_DEVICES];
    /* Monotonic wall-clock diagnostics.  They observe the existing execution
     * and never participate in a physical value, convergence decision, or
     * directed bound. */
    double initialization_seconds;
    double source_assembly_seconds;
    double host_to_device_seconds;
    double device_sweep_seconds;
    double device_to_host_seconds;
    double host_reduction_seconds;
    double convergence_check_seconds;
    double fixed_point_solve_seconds;
    double envelope_context_setup_seconds;
    double bounds_seconds;
    double envelope_residual_seconds;
    double envelope_verify_seconds;
    double envelope_refine_seconds;
    double publication_seconds;
    double cleanup_seconds;
    double total_seconds;
    int failure_phase;
    int failure_iteration;
    size_t failure_cell_index;
    int failure_device_index;
    int failure_ray_begin;
    int failure_ray_end;
    int failure_sweep_stage;
    int failure_segment_index;
    int failure_local_ray_index;
    int failure_bin_index;
    int failure_outward;
    size_t failure_active_ray_count;
    size_t failure_positive_intensity_count;
    int failure_global_ray_index;
    double failure_ray_in;
    double failure_ray_out;
    double failure_host_recomputed_partial;
    double failure_device_geometry_partial;
    double failure_device_rmid;
    double failure_max_impact_absolute_difference;
    double failure_lower;
    double failure_nearest;
    double failure_upper;
} CMFMultiGPUReport;

/* Solve the same discrete fixed-point problem as
 * CMF_EXACT_MODE_DIRECT_REFERENCE, with contiguous ray ranges distributed
 * over the first requested visible CUDA devices.  Range boundaries balance
 * the number of active ray/segment pairs while preserving global ray order.
 * One halo ray is recomputed at each range boundary, so no peer access or
 * Unified Memory is used.
 *
 * The host reduction order is fixed from largest impact-parameter ray block
 * to smallest.  The caller's J buffer is transactional: it is updated only
 * after convergence and remains byte-unchanged on every non-OK return.
 */
CMFMultiGPUStatus cmf_exact_multigpu_direct_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report);

/* Apply the same ray ownership and fixed-point iteration using the
 * subtraction-free positive two-stack affine monoid.  Work is O(n_bins) per
 * segment and does not grow with the drift window beta.  This nearest-rounding
 * prototype is still non-production: it intentionally has no directed
 * lower/upper publication or componentwise supersolution certificate. */
CMFMultiGPUStatus cmf_exact_multigpu_positive_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report);

/* Explicit experimental frequency-parallel path.  The ordinary positive
 * API above remains the serial two-stack reference.  The schedule controls
 * only CUDA execution grouping; changing it must not change any result bit. */
CMFMultiGPUStatus cmf_exact_multigpu_positive_solve_epoch(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    const CMFMultiGPUEpochSchedule *schedule,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report);

/* One affine F(J)=b+KJ application evaluated three ways with the ray-sharded
 * positive monoid.  lower/nearest/upper are transactional and are published
 * together only after the componentwise ordering is verified.  This proves
 * an enclosure of the implemented discrete application; it is not yet the
 * fixed-point componentwise error envelope. */
CMFMultiGPUStatus cmf_exact_multigpu_apply_positive_bounds(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper,
    int requested_devices, CMFMultiGPUReport *report);

CMFMultiGPUStatus cmf_exact_multigpu_apply_positive_bounds_epoch(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper,
    const CMFMultiGPUEpochSchedule *schedule,
    int requested_devices, CMFMultiGPUReport *report);

/* Transactional positive fixed-point solve plus a componentwise a-posteriori
 * error certificate.  The same ray-sharded directed upper operator evaluates
 * K*u with zero fixed source and zero inner boundary.  J and error_upper are
 * both byte-unchanged unless convergence and the final supersolution check
 * succeed. */
CMFMultiGPUStatus cmf_exact_multigpu_positive_solve_envelope(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report);

/* Controlled experimental entry point for partition-only A/B measurements.
 * The ordinary envelope API above remains fixed to weighted segment work.
 * Both modes retain contiguous ray ownership and the same one-ray halo; only
 * the positions of inter-device boundaries differ. */
CMFMultiGPUStatus cmf_exact_multigpu_positive_solve_envelope_partitioned(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations, CMFMultiGPUPartitionMode partition_mode,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report);

CMFMultiGPUStatus
cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations, CMFMultiGPUPartitionMode partition_mode,
    const CMFMultiGPUEpochSchedule *schedule,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report);

const char *cmf_multigpu_status_name(CMFMultiGPUStatus status);

#ifdef __cplusplus
}
#endif

#endif
