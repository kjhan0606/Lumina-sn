#include "cmf_exact_multigpu.h"
#include "cmf_exact_sliding.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures;
#define CHECK(condition, label) do { if (!(condition)) { \
    std::fprintf(stderr, "CMF_EXACT_MULTIGPU_FAIL %s line=%d\n", \
                 label, __LINE__); \
    ++failures; \
} } while (0)

static double relative(double a, double b)
{
    return std::fabs(a - b) / (std::fabs(b) + 1.0e-30);
}

static bool bitwise_equal_or_report(
    const char *label, const double *actual, const double *reference,
    size_t count)
{
    size_t first = count;
    size_t different = 0U;
    double max_relative = 0.0;
    for (size_t index = 0; index < count; ++index) {
        if (std::memcmp(actual + index, reference + index,
                        sizeof(double)) != 0) {
            if (first == count) first = index;
            ++different;
            max_relative = std::fmax(
                max_relative, relative(actual[index], reference[index]));
        }
    }
    if (first == count) return true;
    uint64_t actual_bits = 0U, reference_bits = 0U;
    std::memcpy(&actual_bits, actual + first, sizeof(actual_bits));
    std::memcpy(&reference_bits, reference + first, sizeof(reference_bits));
    std::fprintf(
        stderr,
        "CMF_EXACT_EPOCH_DIFF label=%s first=%zu actual=%.17g "
        "reference=%.17g actual_bits=%016llx reference_bits=%016llx "
        "different=%zu/%zu max_relative=%.17g\n",
        label, first, actual[first], reference[first],
        (unsigned long long)actual_bits,
        (unsigned long long)reference_bits, different, count, max_relative);
    return false;
}

int main(void)
{
    int visible = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess || visible < 2) {
        std::fprintf(stderr,
                     "CMF_EXACT_MULTIGPU_BLOCKED visible_devices=%d required=2\n",
                     visible);
        return 77;
    }
    int requested = visible < 4 ? visible : 4;
    const char *requested_env = std::getenv("CMF_MGPU_TEST_DEVICES");
    if (requested_env && *requested_env) requested = std::atoi(requested_env);
    if (requested < 2 || requested > visible) {
        std::fprintf(stderr,
                     "CMF_EXACT_MULTIGPU_BAD_REQUEST requested=%d visible=%d\n",
                     requested, visible);
        return 78;
    }

    enum { NS = 3, NB = 96 };
    constexpr size_t cells = (size_t)NS * NB;
    const double dlog = 1.0e-3;
    const double texp = 1.0e6;
    double r_inner[NS] = {3.0e14, 4.0e14, 5.0e14};
    double r_outer[NS] = {4.0e14, 5.0e14, 6.0e14};
    double nu[NB], chi_tot[cells], chi_es[cells], fixed[cells];
    double j_cpu[cells], j_cpu_positive[cells];
    double j_gpu[cells], j_one[cells];
    double j_gpu_positive[cells], j_one_positive[cells], initial[cells];
    double j_gpu_positive_epoch[cells];
    double cpu_bound_lower[cells], cpu_bound_nearest[cells];
    double cpu_bound_upper[cells], gpu_bound_lower[cells];
    double gpu_bound_nearest[cells], gpu_bound_upper[cells];
    double one_bound_lower[cells], one_bound_nearest[cells];
    double one_bound_upper[cells];
    double epoch_bound_lower[cells], epoch_bound_nearest[cells];
    double epoch_bound_upper[cells];
    double j_gpu_envelope[cells], gpu_error_envelope[cells];
    double j_epoch_envelope[cells], epoch_error_envelope[cells];
    for (int b = 0; b < NB; ++b)
        nu[b] = 1.0e15 * std::exp((b + 0.5) * dlog);
    for (int s = 0; s < NS; ++s) {
        for (int b = 0; b < NB; ++b) {
            size_t i = (size_t)s * NB + (size_t)b;
            double ripple = 1.0 + 0.35 * std::sin(0.17 * b + 0.4 * s);
            chi_tot[i] = (2.0 + s) * 1.0e-15 * ripple;
            chi_es[i] = 0.27 * chi_tot[i];
            fixed[i] = (1.0 + 0.1 * s) * 1.0e-7 *
                       (1.0 + 0.2 * std::cos(0.11 * b));
            initial[i] = j_cpu[i] = j_cpu_positive[i] =
                j_gpu[i] = j_one[i] =
                j_gpu_positive[i] = j_one_positive[i] =
                j_gpu_positive_epoch[i] = j_gpu_envelope[i] =
                j_epoch_envelope[i] = 0.8e-7;
        }
    }

    CMFExactReport cpu_report;
    CMFExactStatus cpu_status = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_cpu, 120, 1.0e-13,
        CMF_EXACT_MODE_DIRECT_REFERENCE, &cpu_report);
    CHECK(cpu_status == CMF_EXACT_OK, "cpu-direct-oracle");
    CMFExactReport cpu_positive_report;
    CMFExactStatus cpu_positive_status = cmf_exact_characteristic_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_cpu_positive, 120, 1.0e-13,
        CMF_EXACT_MODE_POSITIVE_SLIDING, &cpu_positive_report);
    CHECK(cpu_positive_status == CMF_EXACT_OK, "cpu-positive-oracle");

    CMFMultiGPUReport gpu_report, one_report;
    CMFMultiGPUStatus gpu_status = cmf_exact_multigpu_direct_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_gpu, requested, 120, 1.0e-13,
        &gpu_report);
    CMFMultiGPUStatus one_status = cmf_exact_multigpu_direct_solve(
        NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
        chi_tot, chi_es, fixed, j_one, 1, 120, 1.0e-13,
        &one_report);
    CHECK(gpu_status == CMF_MGPU_OK, "multi-converged");
    CHECK(one_status == CMF_MGPU_OK, "one-device-converged");

    CMFMultiGPUReport gpu_positive_report, one_positive_report;
    CMFMultiGPUStatus gpu_positive_status =
        cmf_exact_multigpu_positive_solve(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_gpu_positive, requested, 120, 1.0e-13,
            &gpu_positive_report);
    CMFMultiGPUStatus one_positive_status =
        cmf_exact_multigpu_positive_solve(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_one_positive, 1, 120, 1.0e-13,
            &one_positive_report);
    CHECK(gpu_positive_status == CMF_MGPU_OK, "multi-positive-converged");
    CHECK(one_positive_status == CMF_MGPU_OK,
          "one-device-positive-converged");
    CHECK(gpu_positive_report.positive_sliding == 1,
          "positive-mode-reported");
    CHECK(gpu_report.positive_sliding == 0, "direct-mode-reported");
    CHECK(gpu_positive_report.max_positive_window_bins > 0,
          "positive-window-reported");
    CHECK(gpu_positive_report.max_device_allocated_bytes >
              gpu_report.max_device_allocated_bytes,
          "positive-workspace-accounted");

    const CMFMultiGPUEpochSchedule epoch_schedule = {64, 2, 1};
    CMFMultiGPUReport epoch_positive_report;
    CMFMultiGPUStatus epoch_positive_status =
        cmf_exact_multigpu_positive_solve_epoch(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_gpu_positive_epoch,
            &epoch_schedule, requested, 120, 1.0e-13,
            &epoch_positive_report);
    CHECK(epoch_positive_status == CMF_MGPU_OK,
          "epoch-positive-converged");
    CHECK(epoch_positive_report.epoch_frequency_parallel == 1 &&
              epoch_positive_report.epoch_block_size == 64 &&
              epoch_positive_report.epoch_batch_cardinality == 2 &&
              epoch_positive_report.epoch_direct_replay_max_window == 1,
          "epoch-positive-schedule-reported");
    CHECK(epoch_positive_report.epoch_workspace_bytes_per_device_max > 0U,
          "epoch-positive-workspace-reported");
    CHECK(bitwise_equal_or_report(
              "positive-solve", j_gpu_positive_epoch, j_gpu_positive, cells),
          "epoch-positive-bitwise-serial");

    CMFExactStatus cpu_bound_status =
        cmf_exact_characteristic_apply_positive_bounds(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_cpu_positive,
            cpu_bound_lower, cpu_bound_nearest, cpu_bound_upper);
    CHECK(cpu_bound_status == CMF_EXACT_OK, "cpu-positive-bounds");
    CMFMultiGPUReport gpu_bound_report, one_bound_report;
    CMFMultiGPUStatus gpu_bound_status =
        cmf_exact_multigpu_apply_positive_bounds(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_cpu_positive,
            gpu_bound_lower, gpu_bound_nearest, gpu_bound_upper,
            requested, &gpu_bound_report);
    CMFMultiGPUStatus one_bound_status =
        cmf_exact_multigpu_apply_positive_bounds(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_cpu_positive,
            one_bound_lower, one_bound_nearest, one_bound_upper,
            1, &one_bound_report);
    CHECK(gpu_bound_status == CMF_MGPU_OK, "multi-positive-bounds");
    CHECK(one_bound_status == CMF_MGPU_OK, "one-device-positive-bounds");
    CHECK(gpu_bound_report.positive_sliding == 1,
          "bound-positive-mode-reported");
    CMFMultiGPUReport epoch_bound_report;
    CMFMultiGPUStatus epoch_bound_status =
        cmf_exact_multigpu_apply_positive_bounds_epoch(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_cpu_positive,
            epoch_bound_lower, epoch_bound_nearest, epoch_bound_upper,
            &epoch_schedule, requested, &epoch_bound_report);
    CHECK(epoch_bound_status == CMF_MGPU_OK, "epoch-positive-bounds");
    CHECK(bitwise_equal_or_report(
              "bounds-lower", epoch_bound_lower, gpu_bound_lower, cells) &&
              bitwise_equal_or_report(
                  "bounds-nearest", epoch_bound_nearest,
                  gpu_bound_nearest, cells) &&
              bitwise_equal_or_report(
                  "bounds-upper", epoch_bound_upper, gpu_bound_upper, cells),
          "epoch-bounds-bitwise-serial");
    const CMFMultiGPUEpochSchedule schedule_matrix[] = {
        {32, 1, 1}, {128, 7, 4}, {256, 1000, 64}
    };
    for (const CMFMultiGPUEpochSchedule &schedule : schedule_matrix) {
        CMFMultiGPUReport schedule_report;
        CMFMultiGPUStatus schedule_status =
            cmf_exact_multigpu_apply_positive_bounds_epoch(
                NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
                chi_tot, chi_es, fixed, j_cpu_positive,
                epoch_bound_lower, epoch_bound_nearest, epoch_bound_upper,
                &schedule, requested, &schedule_report);
        CHECK(schedule_status == CMF_MGPU_OK,
              "epoch-schedule-matrix-bounds");
        bool lower_equal = bitwise_equal_or_report(
            "schedule-lower", epoch_bound_lower, gpu_bound_lower, cells);
        bool nearest_equal = bitwise_equal_or_report(
            "schedule-nearest", epoch_bound_nearest,
            gpu_bound_nearest, cells);
        bool upper_equal = bitwise_equal_or_report(
            "schedule-upper", epoch_bound_upper, gpu_bound_upper, cells);
        CHECK(lower_equal && nearest_equal && upper_equal,
              "epoch-schedule-matrix-bitwise-serial");
        CHECK(schedule_report.epoch_block_size == schedule.block_size &&
                  schedule_report.epoch_batch_cardinality ==
                      schedule.epoch_batch_cardinality &&
                  schedule_report.epoch_direct_replay_max_window ==
                      schedule.direct_replay_max_window,
              "epoch-schedule-matrix-reported");
    }

    CMFMultiGPUReport gpu_envelope_report;
    CMFMultiGPUStatus gpu_envelope_status =
        cmf_exact_multigpu_positive_solve_envelope(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_gpu_envelope, gpu_error_envelope,
            5U, requested, 120, 1.0e-13, &gpu_envelope_report);
    CHECK(gpu_envelope_status == CMF_MGPU_OK,
          "multi-componentwise-envelope");
    CHECK(gpu_envelope_report.componentwise_error_envelope_verified == 1,
          "multi-componentwise-envelope-verified");
    CHECK(gpu_envelope_report.componentwise_error_seed_attempts >= 1U,
          "multi-componentwise-envelope-seed");
    CHECK(gpu_envelope_report.componentwise_error_refinement_iterations == 5U,
          "multi-componentwise-envelope-refined");
    CHECK(gpu_envelope_report.max_scattering_ratio ==
              cpu_positive_report.max_scattering_ratio &&
              gpu_envelope_report.max_scattering_ratio >= 0.0 &&
              gpu_envelope_report.max_scattering_ratio < 1.0,
          "production-scattering-ratio-certificate");
    double expected_absolute_bound =
        gpu_envelope_report.max_scattering_ratio == 0.0 ? 0.0 :
        (gpu_envelope_report.max_scattering_ratio /
         (1.0 - gpu_envelope_report.max_scattering_ratio)) *
            gpu_envelope_report.final_max_absolute_change;
    CHECK(gpu_envelope_report.fixed_point_absolute_error_bound ==
              expected_absolute_bound &&
              std::isfinite(
                  gpu_envelope_report.fixed_point_absolute_error_bound),
          "production-absolute-error-certificate");
    CHECK(gpu_envelope_report.persistent_context_initializations == 1U,
          "persistent-context-initialized-once");
    CHECK(gpu_envelope_report.persistent_upper_operator_applications ==
              gpu_envelope_report.componentwise_error_seed_attempts + 1U +
              2U * gpu_envelope_report.
                       componentwise_error_refinement_iterations,
          "persistent-upper-application-count");
    CHECK(gpu_envelope_report.persistent_bound_applications ==
              3U + gpu_envelope_report.
                       persistent_upper_operator_applications,
          "persistent-bound-application-count");
    CMFMultiGPUReport epoch_envelope_report;
    CMFMultiGPUStatus epoch_envelope_status =
        cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, j_epoch_envelope,
            epoch_error_envelope, 5U,
            CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, &epoch_schedule,
            requested, 120, 1.0e-13, &epoch_envelope_report);
    CHECK(epoch_envelope_status == CMF_MGPU_OK,
          "epoch-componentwise-envelope");
    CHECK(bitwise_equal_or_report(
              "envelope-J", j_epoch_envelope, j_gpu_envelope, cells) &&
              bitwise_equal_or_report(
                  "envelope-error", epoch_error_envelope,
                  gpu_error_envelope, cells),
          "epoch-envelope-bitwise-serial");
    CHECK(gpu_report.devices_used == requested, "device-count");
    CHECK(gpu_report.visible_devices == visible, "visible-count");
    CHECK(gpu_report.weighted_contiguous_ray_partition == 1 &&
              gpu_positive_report.weighted_contiguous_ray_partition == 1 &&
              gpu_bound_report.weighted_contiguous_ray_partition == 1 &&
              gpu_envelope_report.weighted_contiguous_ray_partition == 1,
          "weighted-partition-all-multigpu-paths");
    CHECK(one_report.weighted_contiguous_ray_partition == 1 &&
              one_positive_report.weighted_contiguous_ray_partition == 1 &&
              one_bound_report.weighted_contiguous_ray_partition == 1,
          "weighted-partition-one-device-paths");
    CHECK(one_report.min_owned_ray_segment_work == 51U &&
              one_report.max_owned_ray_segment_work == 51U &&
              one_report.min_computed_ray_segment_work == 51U &&
              one_report.max_computed_ray_segment_work == 51U,
          "one-device-ray-segment-work");
    CHECK(gpu_report.min_owned_ray_segment_work > 0U &&
              gpu_report.min_owned_ray_segment_work <=
                  gpu_report.max_owned_ray_segment_work &&
              gpu_report.min_computed_ray_segment_work >=
                  gpu_report.min_owned_ray_segment_work &&
              gpu_report.max_computed_ray_segment_work >=
                  gpu_report.max_owned_ray_segment_work,
          "weighted-partition-work-order");
    CHECK(gpu_positive_report.min_owned_ray_segment_work ==
              gpu_report.min_owned_ray_segment_work &&
              gpu_positive_report.max_owned_ray_segment_work ==
                  gpu_report.max_owned_ray_segment_work &&
              gpu_bound_report.min_computed_ray_segment_work ==
                  gpu_report.min_computed_ray_segment_work &&
              gpu_envelope_report.max_computed_ray_segment_work ==
                  gpu_report.max_computed_ray_segment_work,
          "weighted-partition-shared-across-paths");
    CHECK(gpu_report.owned_rays == gpu_report.n_rays,
          "all-rays-owned-once");
    CHECK(gpu_report.computed_rays_with_halos ==
              gpu_report.n_rays + (size_t)requested - 1U,
          "one-halo-per-boundary");
    CHECK(gpu_report.deterministic_host_reduction == 1,
          "deterministic-reduction");
    CHECK(gpu_report.max_device_allocated_bytes > 0 &&
          gpu_report.total_device_allocated_bytes >=
              gpu_report.max_device_allocated_bytes,
          "device-memory-accounting");
    CHECK(gpu_report.max_device_allocated_bytes <
              one_report.max_device_allocated_bytes,
          "per-device-memory-reduced");
    CHECK(gpu_report.final_max_relative_change < gpu_report.tolerance,
          "multi-residual");

    double max_cpu_multi = 0.0, max_one_multi = 0.0;
    double max_cpu_positive_multi = 0.0;
    double max_one_positive_multi = 0.0;
    double max_positive_direct = 0.0;
    double max_cpu_bound_nearest = 0.0;
    double max_one_bound_nearest = 0.0;
    double max_bound_relative_width = 0.0;
    double max_envelope_observed_ratio = 0.0;
    size_t envelope_covers = 0;
    for (size_t i = 0; i < cells; ++i) {
        double value = relative(j_gpu[i], j_cpu[i]);
        if (value > max_cpu_multi) max_cpu_multi = value;
        value = relative(j_gpu[i], j_one[i]);
        if (value > max_one_multi) max_one_multi = value;
        value = relative(j_gpu_positive[i], j_cpu_positive[i]);
        if (value > max_cpu_positive_multi)
            max_cpu_positive_multi = value;
        value = relative(j_gpu_positive[i], j_one_positive[i]);
        if (value > max_one_positive_multi)
            max_one_positive_multi = value;
        value = relative(j_gpu_positive[i], j_gpu[i]);
        if (value > max_positive_direct) max_positive_direct = value;
        value = relative(gpu_bound_nearest[i], cpu_bound_nearest[i]);
        if (value > max_cpu_bound_nearest) max_cpu_bound_nearest = value;
        value = relative(gpu_bound_nearest[i], one_bound_nearest[i]);
        if (value > max_one_bound_nearest) max_one_bound_nearest = value;
        double width = (gpu_bound_upper[i] - gpu_bound_lower[i]) /
                       (std::fabs(gpu_bound_nearest[i]) + 1.0e-30);
        if (width > max_bound_relative_width)
            max_bound_relative_width = width;
        CHECK(std::isfinite(j_gpu[i]) && j_gpu[i] >= 0.0,
              "finite-nonnegative-result");
        CHECK(std::isfinite(j_gpu_positive[i]) && j_gpu_positive[i] >= 0.0,
              "finite-nonnegative-positive-result");
        CHECK(gpu_bound_lower[i] <= gpu_bound_nearest[i] &&
                  gpu_bound_nearest[i] <= gpu_bound_upper[i] &&
                  std::isfinite(gpu_bound_upper[i]),
              "gpu-directed-bound-order");
        CHECK(one_bound_lower[i] <= one_bound_nearest[i] &&
                  one_bound_nearest[i] <= one_bound_upper[i] &&
                  std::isfinite(one_bound_upper[i]),
              "one-directed-bound-order");
        double envelope_observed = std::fabs(j_gpu_envelope[i] - j_cpu[i]);
        CHECK(gpu_error_envelope[i] >= 0.0 &&
                  std::isfinite(gpu_error_envelope[i]),
              "multi-componentwise-envelope-finite");
        CHECK(envelope_observed <= gpu_error_envelope[i],
              "multi-componentwise-envelope-covers-direct");
        if (envelope_observed <= gpu_error_envelope[i]) ++envelope_covers;
        if (gpu_error_envelope[i] > 0.0) {
            double envelope_ratio =
                envelope_observed / gpu_error_envelope[i];
            if (envelope_ratio > max_envelope_observed_ratio)
                max_envelope_observed_ratio = envelope_ratio;
        }
    }
    CHECK(max_cpu_multi < 2.0e-10, "cpu-multigpu-agreement");
    CHECK(max_one_multi < 2.0e-10, "one-multigpu-agreement");
    CHECK(max_cpu_positive_multi < 2.0e-10,
          "cpu-positive-multigpu-agreement");
    CHECK(max_one_positive_multi < 2.0e-10,
          "one-positive-multigpu-agreement");
    CHECK(max_positive_direct < 2.0e-10,
          "positive-direct-multigpu-agreement");
    CHECK(max_cpu_bound_nearest < 2.0e-10,
          "cpu-gpu-bound-nearest-agreement");
    CHECK(max_one_bound_nearest < 2.0e-10,
          "one-multi-bound-nearest-agreement");
    CHECK(max_bound_relative_width > 0.0 &&
              std::isfinite(max_bound_relative_width),
          "directed-bound-has-finite-width");
    CHECK(envelope_covers == cells, "componentwise-envelope-covers-all");
    if (!(max_cpu_multi < 2.0e-10) || !(max_one_multi < 2.0e-10))
        std::fprintf(stderr,
                     "CMF_EXACT_MULTIGPU_DIFF cpu=%.17g one=%.17g "
                     "cpu_positive=%.17g one_positive=%.17g "
                     "positive_direct=%.17g\n",
                     max_cpu_multi, max_one_multi,
                     max_cpu_positive_multi, max_one_positive_multi,
                     max_positive_direct);

    double max_partition_relative = 0.0;
    double max_positive_partition_relative = 0.0;
    double max_bound_partition_relative = 0.0;
    for (int devices = 2; devices < requested; ++devices) {
        std::vector<double> partition(initial, initial + cells);
        CMFMultiGPUReport partition_report;
        CMFMultiGPUStatus partition_status = cmf_exact_multigpu_direct_solve(
            NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
            chi_tot, chi_es, fixed, partition.data(), devices, 120, 1.0e-13,
            &partition_report);
        CHECK(partition_status == CMF_MGPU_OK,
              "intermediate-partition-converged");
        CHECK(partition_report.devices_used == devices,
              "intermediate-partition-count");
        CHECK(partition_report.weighted_contiguous_ray_partition == 1 &&
                  partition_report.min_owned_ray_segment_work > 0U,
              "intermediate-weighted-partition");
        CHECK(partition_report.computed_rays_with_halos ==
                  partition_report.n_rays + (size_t)devices - 1U,
                  "intermediate-partition-halos");
        std::vector<double> positive_partition(initial, initial + cells);
        CMFMultiGPUReport positive_partition_report;
        CMFMultiGPUStatus positive_partition_status =
            cmf_exact_multigpu_positive_solve(
                NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
                chi_tot, chi_es, fixed, positive_partition.data(), devices,
                120, 1.0e-13, &positive_partition_report);
        CHECK(positive_partition_status == CMF_MGPU_OK,
              "intermediate-positive-partition-converged");
        CHECK(positive_partition_report.devices_used == devices,
              "intermediate-positive-partition-count");
        CHECK(positive_partition_report.min_owned_ray_segment_work ==
                  partition_report.min_owned_ray_segment_work &&
                  positive_partition_report.max_computed_ray_segment_work ==
                      partition_report.max_computed_ray_segment_work,
              "intermediate-weighted-partition-shared");
        std::vector<double> partition_lower(cells, -1.0);
        std::vector<double> partition_nearest(cells, -1.0);
        std::vector<double> partition_upper(cells, -1.0);
        CMFMultiGPUReport partition_bound_report;
        CMFMultiGPUStatus partition_bound_status =
            cmf_exact_multigpu_apply_positive_bounds(
                NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
                chi_tot, chi_es, fixed, j_cpu_positive,
                partition_lower.data(), partition_nearest.data(),
                partition_upper.data(), devices, &partition_bound_report);
        CHECK(partition_bound_status == CMF_MGPU_OK,
              "intermediate-partition-bounds");
        CHECK(partition_bound_report.min_owned_ray_segment_work ==
                  partition_report.min_owned_ray_segment_work &&
                  partition_bound_report.max_computed_ray_segment_work ==
                      partition_report.max_computed_ray_segment_work,
              "intermediate-bound-weighted-partition-shared");
        for (size_t i = 0; i < cells; ++i) {
            double value = relative(partition[i], j_gpu[i]);
            if (value > max_partition_relative)
                max_partition_relative = value;
            value = relative(positive_partition[i], j_gpu_positive[i]);
            if (value > max_positive_partition_relative)
                max_positive_partition_relative = value;
            value = relative(partition_nearest[i], gpu_bound_nearest[i]);
            if (value > max_bound_partition_relative)
                max_bound_partition_relative = value;
            CHECK(partition_lower[i] <= partition_nearest[i] &&
                      partition_nearest[i] <= partition_upper[i],
                  "intermediate-partition-bound-order");
        }
    }
    CHECK(max_partition_relative < 2.0e-10,
          "all-partition-counts-agree");
    CHECK(max_positive_partition_relative < 2.0e-10,
          "all-positive-partition-counts-agree");
    CHECK(max_bound_partition_relative < 2.0e-10,
          "all-bound-partition-counts-agree");

    /* A physical-input failure is fail-closed and preserves J byte-for-byte. */
    double bad_chi[cells], rejected[cells], rejected_before[cells];
    std::memcpy(bad_chi, chi_tot, sizeof(bad_chi));
    std::memcpy(rejected, initial, sizeof(rejected));
    std::memcpy(rejected_before, rejected, sizeof(rejected_before));
    bad_chi[7] = -1.0;
    CHECK(cmf_exact_multigpu_direct_solve(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              bad_chi, chi_es, fixed, rejected, requested, 20, 1.0e-8,
              nullptr) == CMF_MGPU_NONFINITE,
          "negative-opacity-rejected");
    CHECK(std::memcmp(rejected, rejected_before, sizeof(rejected)) == 0,
          "negative-reject-preserves-j");
    std::memcpy(rejected, initial, sizeof(rejected));
    CHECK(cmf_exact_multigpu_positive_solve(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              bad_chi, chi_es, fixed, rejected, requested, 20, 1.0e-8,
              nullptr) == CMF_MGPU_NONFINITE,
          "positive-negative-opacity-rejected");
    CHECK(std::memcmp(rejected, initial, sizeof(rejected)) == 0,
          "positive-negative-reject-preserves-j");
    double rejected_lower[cells], rejected_nearest[cells];
    double rejected_upper[cells], rejected_lower_before[cells];
    double rejected_nearest_before[cells], rejected_upper_before[cells];
    for (size_t i = 0; i < cells; ++i) {
        rejected_lower[i] = 2.0 + (double)i;
        rejected_nearest[i] = 3.0 + (double)i;
        rejected_upper[i] = 4.0 + (double)i;
    }
    std::memcpy(rejected_lower_before, rejected_lower,
                sizeof(rejected_lower));
    std::memcpy(rejected_nearest_before, rejected_nearest,
                sizeof(rejected_nearest));
    std::memcpy(rejected_upper_before, rejected_upper,
                sizeof(rejected_upper));
    CHECK(cmf_exact_multigpu_apply_positive_bounds(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              bad_chi, chi_es, fixed, initial,
              rejected_lower, rejected_nearest, rejected_upper,
              requested, nullptr) == CMF_MGPU_NONFINITE,
          "bound-negative-opacity-rejected");
    CHECK(std::memcmp(rejected_lower, rejected_lower_before,
                      sizeof(rejected_lower)) == 0 &&
              std::memcmp(rejected_nearest, rejected_nearest_before,
                          sizeof(rejected_nearest)) == 0 &&
              std::memcmp(rejected_upper, rejected_upper_before,
                          sizeof(rejected_upper)) == 0,
          "bound-reject-preserves-all-outputs");
    double rejected_envelope[cells], rejected_envelope_before[cells];
    std::memcpy(rejected, initial, sizeof(rejected));
    for (size_t i = 0; i < cells; ++i)
        rejected_envelope[i] = 5.0 + (double)i;
    std::memcpy(rejected_envelope_before, rejected_envelope,
                sizeof(rejected_envelope));
    CHECK(cmf_exact_multigpu_positive_solve_envelope(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              bad_chi, chi_es, fixed, rejected, rejected_envelope,
              2U, requested, 20, 1.0e-8, nullptr) == CMF_MGPU_NONFINITE,
          "envelope-negative-opacity-rejected");
    CHECK(std::memcmp(rejected, initial, sizeof(rejected)) == 0 &&
              std::memcmp(rejected_envelope, rejected_envelope_before,
                          sizeof(rejected_envelope)) == 0,
          "envelope-reject-preserves-j-and-bound");

    std::memcpy(rejected, initial, sizeof(rejected));
    CHECK(cmf_exact_multigpu_direct_solve(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, rejected, visible + 1, 20, 1.0e-8,
              nullptr) == CMF_MGPU_INSUFFICIENT_DEVICES,
          "insufficient-device-rejected");
    CHECK(std::memcmp(rejected, initial, sizeof(rejected)) == 0,
          "device-reject-preserves-j");

    const CMFMultiGPUEpochSchedule invalid_epoch_schedule = {48, 0, 0};
    std::memcpy(rejected, initial, sizeof(rejected));
    CHECK(cmf_exact_multigpu_positive_solve_epoch(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, rejected, &invalid_epoch_schedule,
              requested, 20, 1.0e-8, nullptr) == CMF_MGPU_INVALID_INPUT,
          "invalid-epoch-schedule-rejected");
    CHECK(std::memcmp(rejected, initial, sizeof(rejected)) == 0,
          "invalid-epoch-schedule-preserves-j");

    std::memcpy(rejected, initial, sizeof(rejected));
    CMFMultiGPUReport short_report;
    CHECK(cmf_exact_multigpu_direct_solve(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, rejected, requested, 2, 1.0e-30,
              &short_report) == CMF_MGPU_NOT_CONVERGED,
          "cap-fails-closed");
    CHECK(short_report.iterations_used == 2, "cap-count");
    CHECK(std::memcmp(rejected, initial, sizeof(rejected)) == 0,
          "cap-preserves-j");
    std::memcpy(rejected, initial, sizeof(rejected));
    CMFMultiGPUReport positive_short_report;
    CHECK(cmf_exact_multigpu_positive_solve(
              NS, NB, dlog, nu, r_inner, r_outer, texp, 10000.0, 1.0,
              chi_tot, chi_es, fixed, rejected, requested, 2, 1.0e-30,
              &positive_short_report) == CMF_MGPU_NOT_CONVERGED,
          "positive-cap-fails-closed");
    CHECK(positive_short_report.iterations_used == 2,
          "positive-cap-count");
    CHECK(std::memcmp(rejected, initial, sizeof(rejected)) == 0,
          "positive-cap-preserves-j");

    if (failures) return 1;
    std::printf(
        "CMF_EXACT_MULTIGPU_SELFTEST PASS devices=%d visible=%d rays=%zu "
        "computed_with_halos=%zu iterations_cpu/multi/one=%d/%d/%d "
        "max_rel_cpu=%.3e max_rel_one=%.3e max_rel_partitions=%.3e "
        "positive_iterations_cpu/multi/one=%d/%d/%d "
        "max_rel_positive_cpu=%.3e max_rel_positive_one=%.3e "
        "max_rel_positive_partitions=%.3e max_rel_positive_direct=%.3e "
        "max_rel_bound_cpu=%.3e max_rel_bound_one=%.3e "
        "max_rel_bound_partitions=%.3e bound_rel_width=%.3e "
        "envelope_residual_max=%.3e envelope_min/max=%.3e/%.3e "
        "envelope_observed_ratio=%.3e "
        "persistent_contexts/bounds/upper=%zu/%zu/%zu "
        "ray_work_owned_min/max=%zu/%zu "
        "ray_work_computed_min/max=%zu/%zu "
        "positive_window=%zu "
        "max_device_MiB=%.3f one_device_MiB=%.3f "
        "positive_max_device_MiB=%.3f total_device_MiB=%.3f "
        "drift_bins=%.6f repair=0\n",
        requested, visible, gpu_report.n_rays,
        gpu_report.computed_rays_with_halos, cpu_report.iterations_used,
        gpu_report.iterations_used, one_report.iterations_used,
        max_cpu_multi, max_one_multi, max_partition_relative,
        cpu_positive_report.iterations_used,
        gpu_positive_report.iterations_used,
        one_positive_report.iterations_used,
        max_cpu_positive_multi, max_one_positive_multi,
        max_positive_partition_relative, max_positive_direct,
        max_cpu_bound_nearest, max_one_bound_nearest,
        max_bound_partition_relative, max_bound_relative_width,
        gpu_envelope_report.componentwise_residual_upper_max,
        gpu_envelope_report.componentwise_error_upper_min,
        gpu_envelope_report.componentwise_error_upper_max,
        max_envelope_observed_ratio,
        gpu_envelope_report.persistent_context_initializations,
        gpu_envelope_report.persistent_bound_applications,
        gpu_envelope_report.persistent_upper_operator_applications,
        gpu_report.min_owned_ray_segment_work,
        gpu_report.max_owned_ray_segment_work,
        gpu_report.min_computed_ray_segment_work,
        gpu_report.max_computed_ray_segment_work,
        gpu_positive_report.max_positive_window_bins,
        gpu_report.max_device_allocated_bytes / 1048576.0,
        one_report.max_device_allocated_bytes / 1048576.0,
        gpu_positive_report.max_device_allocated_bytes / 1048576.0,
        gpu_report.total_device_allocated_bytes / 1048576.0,
        gpu_report.max_characteristic_drift_bins);
    return 0;
}
