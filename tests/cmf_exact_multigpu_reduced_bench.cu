#include "cmf_exact_multigpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr double kC = 2.99792458e10;
constexpr double kTimeExplosion = 1683072.0;
constexpr double kTInner = 10020.0;
constexpr int kProductionShells = 50;
constexpr int kDefaultBins = 8192;
constexpr int kDefaultDevices = 4;
constexpr int kIterationLimit = 64;
constexpr double kTolerance = 1.0e-8;
constexpr size_t kProductionRaySegmentWork = 2025U;
constexpr size_t kFourMinOwnedRaySegmentWork = 490U;
constexpr size_t kFourMaxOwnedRaySegmentWork = 539U;
constexpr size_t kFourMinComputedRaySegmentWork = 496U;
constexpr size_t kFourMaxComputedRaySegmentWork = 570U;
constexpr size_t kEqualFourMinOwnedRaySegmentWork = 136U;
constexpr size_t kEqualFourMaxOwnedRaySegmentWork = 800U;
constexpr size_t kEqualFourMinComputedRaySegmentWork = 136U;
constexpr size_t kEqualFourMaxComputedRaySegmentWork = 849U;

struct Geometry {
    std::vector<double> r_inner;
    std::vector<double> r_outer;
};

[[noreturn]] void fail(const char *message)
{
    std::fprintf(stderr, "CMF_MGPU_REDUCED_FAIL %s\n", message);
    std::exit(1);
}

int positive_env_int(const char *name, int fallback, int maximum)
{
    const char *text = std::getenv(name);
    if (!text || !*text) return fallback;
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > maximum)
        fail("invalid-positive-integer-environment");
    return (int)value;
}

Geometry load_geometry(const char *path)
{
    FILE *stream = std::fopen(path, "r");
    if (!stream) fail("geometry-open");
    char line[512];
    if (!std::fgets(line, sizeof(line), stream)) {
        std::fclose(stream);
        fail("geometry-header");
    }
    Geometry geometry;
    int expected_shell = 0;
    while (std::fgets(line, sizeof(line), stream)) {
        int shell = -1;
        double r_inner = 0.0, r_outer = 0.0;
        double v_inner = 0.0, v_outer = 0.0;
        if (std::sscanf(line, "%d,%lf,%lf,%lf,%lf", &shell, &r_inner,
                        &r_outer, &v_inner, &v_outer) != 5 ||
            shell != expected_shell || !(r_inner > 0.0) ||
            !(r_outer > r_inner) ||
            (shell > 0 &&
             (!(r_inner >= geometry.r_inner.back()) ||
              !(r_outer > geometry.r_outer.back()))) ||
            !std::isfinite(v_inner) || !std::isfinite(v_outer)) {
            std::fclose(stream);
            fail("geometry-row");
        }
        geometry.r_inner.push_back(r_inner);
        geometry.r_outer.push_back(r_outer);
        ++expected_shell;
    }
    if (std::fclose(stream) != 0) fail("geometry-close");
    if (expected_shell != kProductionShells)
        fail("geometry-shell-count");
    return geometry;
}

void geometry_drift_contract(const Geometry &geometry, double dlognu,
                             double *max_drift, size_t *max_window)
{
    const int ns = (int)geometry.r_inner.size();
    std::vector<double> rmid((size_t)ns);
    for (int shell = 0; shell < ns; ++shell)
        rmid[(size_t)shell] = 0.5 *
            (geometry.r_inner[(size_t)shell] +
             geometry.r_outer[(size_t)shell]);
    double drift_max = 0.0;
    size_t window_max = 0;
    for (int ray = 0; ray < ns + 16; ++ray) {
        double impact = ray < 16
            ? rmid[0] * (double)ray / 16.0
            : rmid[(size_t)(ray - 16)];
        std::vector<double> z;
        for (int shell = ns - 1; shell >= 0; --shell) {
            if (!(rmid[(size_t)shell] > impact)) break;
            z.push_back(std::sqrt(rmid[(size_t)shell] *
                                  rmid[(size_t)shell] - impact * impact));
        }
        double z_inner = impact < rmid[0]
            ? std::sqrt(rmid[0] * rmid[0] - impact * impact) : 0.0;
        double drift = 0.0;
        for (size_t segment = 0; segment < z.size(); ++segment) {
            double downstream = segment + 1U < z.size()
                ? z[segment + 1U] : z_inner;
            double beta = (z[segment] - downstream) /
                          (kTimeExplosion * kC * dlognu);
            if (!(beta >= 0.0) || !std::isfinite(beta))
                fail("geometry-beta");
            drift += beta;
            int q = (int)std::floor(beta);
            double phi = beta - (double)q;
            int qtop = phi < 0.5 ? q : q + 1;
            size_t window = qtop >= 2 ? (size_t)(qtop - 1) : 0U;
            window_max = std::max(window_max, window);
        }
        drift_max = std::max(drift_max, drift);
    }
    *max_drift = drift_max;
    *max_window = window_max;
}

uint64_t fnv1a_update(uint64_t hash, const void *data, size_t bytes)
{
    const unsigned char *cursor = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= (uint64_t)cursor[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool write_result(const char *path, int ns, int nb,
                  const std::vector<double> &one_j,
                  const std::vector<double> &one_error,
                  const std::vector<double> &four_j,
                  const std::vector<double> &four_error)
{
    if (!path || !*path) return true;
    FILE *stream = std::fopen(path, "wb");
    if (!stream) return false;
    const unsigned char magic[16] = {
        'L','U','M','I','N','A','_','M','G','P','U','_','R','1',0,0
    };
    uint64_t shape[2] = {(uint64_t)ns, (uint64_t)nb};
    const std::vector<double> *fields[4] = {
        &one_j, &one_error, &four_j, &four_error
    };
    bool ok = std::fwrite(magic, 1, sizeof(magic), stream) == sizeof(magic) &&
              std::fwrite(shape, sizeof(shape[0]), 2, stream) == 2;
    for (int field = 0; field < 4 && ok; ++field)
        ok = std::fwrite(fields[field]->data(), sizeof(double),
                         fields[field]->size(), stream) ==
             fields[field]->size();
    if (std::fclose(stream) != 0) ok = false;
    return ok;
}

bool write_single_result(const char *path, int ns, int nb, int devices,
                         const std::vector<double> &j,
                         const std::vector<double> &error_upper)
{
    if (!path || !*path) return true;
    FILE *stream = std::fopen(path, "wb");
    if (!stream) return false;
    const unsigned char magic[16] = {
        'L','U','M','I','N','A','_','M','G','P','U','_','S','2',0,0
    };
    uint64_t header[3] = {
        (uint64_t)ns, (uint64_t)nb, (uint64_t)devices
    };
    bool ok = std::fwrite(magic, 1, sizeof(magic), stream) == sizeof(magic) &&
              std::fwrite(header, sizeof(header[0]), 3, stream) == 3 &&
              std::fwrite(j.data(), sizeof(double), j.size(), stream) ==
                  j.size() &&
              std::fwrite(error_upper.data(), sizeof(double),
                          error_upper.size(), stream) == error_upper.size();
    if (std::fclose(stream) != 0) ok = false;
    return ok;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s GEOMETRY_CSV\n", argv[0]);
        return 2;
    }
    Geometry geometry = load_geometry(argv[1]);
    const int ns = (int)geometry.r_inner.size();
    const int nb = positive_env_int(
        "CMF_MGPU_REDUCED_BINS", kDefaultBins, 2013113);
    const int requested_devices = positive_env_int(
        "CMF_MGPU_REDUCED_DEVICES", kDefaultDevices, ns + 16);
    const int refinements = positive_env_int(
        "CMF_MGPU_REDUCED_REFINEMENTS", 1, 16);
    const char *epoch_text = std::getenv("CMF_MGPU_REDUCED_EPOCH");
    const bool use_epoch = epoch_text && std::strcmp(epoch_text, "1") == 0;
    if (epoch_text && *epoch_text && std::strcmp(epoch_text, "0") != 0 &&
        std::strcmp(epoch_text, "1") != 0)
        fail("invalid-epoch-mode");
    const CMFMultiGPUEpochSchedule epoch_schedule = {
        positive_env_int("CMF_MGPU_EPOCH_BLOCK", 128, 256),
        positive_env_int("CMF_MGPU_EPOCH_BATCH", 64, 2013113),
        positive_env_int("CMF_MGPU_EPOCH_REPLAY", 32, 2013113)
    };
    if (epoch_schedule.block_size != 32 &&
        epoch_schedule.block_size != 64 &&
        epoch_schedule.block_size != 128 &&
        epoch_schedule.block_size != 256)
        fail("invalid-epoch-block-size");
    const char *partition_name =
        std::getenv("CMF_MGPU_REDUCED_PARTITION");
    if (!partition_name || !*partition_name) partition_name = "weighted";
    CMFMultiGPUPartitionMode partition_mode;
    if (std::strcmp(partition_name, "weighted") == 0) {
        partition_mode = CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS;
    } else if (std::strcmp(partition_name, "equal") == 0) {
        partition_mode = CMF_MGPU_PARTITION_EQUAL_RAYS;
    } else {
        fail("invalid-ray-partition");
    }
    const char *mode = std::getenv("CMF_MGPU_REDUCED_MODE");
    if (!mode || !*mode) mode = "both";
    const bool split_one = std::strcmp(mode, "one") == 0;
    const bool split_four = std::strcmp(mode, "four") == 0;
    if (!split_one && !split_four && std::strcmp(mode, "both") != 0)
        fail("invalid-run-mode");
    const double dlognu = (1.0e6 / kC) / 12.0;
    double expected_drift = 0.0;
    size_t expected_window = 0;
    geometry_drift_contract(
        geometry, dlognu, &expected_drift, &expected_window);
    if (!(expected_drift > 47649.0 && expected_drift < 47650.0) ||
        expected_window != 9108U)
        fail("production-geometry-drift-contract");
    const char *validate = std::getenv("CMF_MGPU_REDUCED_VALIDATE_ONLY");
    if (validate && std::strcmp(validate, "1") == 0) {
        std::printf(
            "CMF_MGPU_REDUCED_CONTRACT PASS shells=%d bins=%d rays=%d "
            "dlognu=%.17g drift=%.17g max_window=%zu partition=%s\n",
            ns, nb, ns + 16, dlognu, expected_drift, expected_window,
            partition_name);
        return 0;
    }

    int visible = 0;
    const int visible_required = split_one ? 1 : requested_devices;
    if (cudaGetDeviceCount(&visible) != cudaSuccess ||
        visible < visible_required || requested_devices < 2)
        fail("insufficient-visible-devices");
    size_t cells = (size_t)ns * (size_t)nb;
    if (cells / (size_t)ns != (size_t)nb)
        fail("cell-count-overflow");
    std::vector<double> nu(cells / (size_t)ns);
    std::vector<double> chi_tot(cells), chi_es(cells), fixed(cells);
    std::vector<double> one_j(cells), one_error(cells, -1.0);
    std::vector<double> four_j(cells), four_error(cells, -1.0);
    const double profile_width = 4.0e6 / kC;
    const double line_nu_lo = kC / (20000.0e-8);
    const double nu_min = line_nu_lo * (1.0 - profile_width) *
                          std::exp(-0.5 * dlognu);
    for (int bin = 0; bin < nb; ++bin)
        nu[(size_t)bin] = nu_min *
                          std::exp(((double)bin + 0.5) * dlognu);
    for (int shell = 0; shell < ns; ++shell) {
        for (int bin = 0; bin < nb; ++bin) {
            size_t index = (size_t)shell * (size_t)nb + (size_t)bin;
            double ripple = 1.0 + 0.20 *
                std::sin(0.00091 * (double)bin + 0.17 * (double)shell);
            double ratio = 0.18 + 0.04 *
                (0.5 + 0.5 *
                 std::cos(0.00037 * (double)bin + 0.11 * (double)shell));
            chi_tot[index] = (1.6e-15 + 1.5e-17 * (double)shell) * ripple;
            chi_es[index] = ratio * chi_tot[index];
            fixed[index] = (1.0e-7 + 2.0e-10 * (double)shell) *
                (1.0 + 0.12 *
                 std::cos(0.00073 * (double)bin + 0.07 * (double)shell));
            one_j[index] = four_j[index] = 0.8e-7;
        }
    }

    auto apply_bounds = [&](const double *input, double *lower,
                            double *nearest, double *upper, int devices,
                            CMFMultiGPUReport *report) {
        if (use_epoch)
            return cmf_exact_multigpu_apply_positive_bounds_epoch(
                ns, nb, dlognu, nu.data(), geometry.r_inner.data(),
                geometry.r_outer.data(), kTimeExplosion, kTInner, 1.0,
                chi_tot.data(), chi_es.data(), fixed.data(), input,
                lower, nearest, upper, &epoch_schedule, devices, report);
        return cmf_exact_multigpu_apply_positive_bounds(
            ns, nb, dlognu, nu.data(), geometry.r_inner.data(),
            geometry.r_outer.data(), kTimeExplosion, kTInner, 1.0,
            chi_tot.data(), chi_es.data(), fixed.data(), input,
            lower, nearest, upper, devices, report);
    };
    auto solve_envelope = [&](double *j, double *error, int devices,
                              CMFMultiGPUReport *report) {
        if (use_epoch)
            return cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(
                ns, nb, dlognu, nu.data(), geometry.r_inner.data(),
                geometry.r_outer.data(), kTimeExplosion, kTInner, 1.0,
                chi_tot.data(), chi_es.data(), fixed.data(), j, error,
                (size_t)refinements, partition_mode, &epoch_schedule,
                devices, kIterationLimit, kTolerance, report);
        return cmf_exact_multigpu_positive_solve_envelope_partitioned(
            ns, nb, dlognu, nu.data(), geometry.r_inner.data(),
            geometry.r_outer.data(), kTimeExplosion, kTInner, 1.0,
            chi_tot.data(), chi_es.data(), fixed.data(), j, error,
            (size_t)refinements, partition_mode, devices,
            kIterationLimit, kTolerance, report);
    };

    const char *apply_diagnostic =
        std::getenv("CMF_MGPU_REDUCED_APPLY_DIAGNOSTIC");
    if (apply_diagnostic && std::strcmp(apply_diagnostic, "1") == 0) {
        if (partition_mode != CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS)
            fail("apply-diagnostic-requires-weighted-partition");
        std::vector<double> one_lower(cells), one_nearest(cells);
        std::vector<double> one_upper(cells), four_lower(cells);
        std::vector<double> four_nearest(cells), four_upper(cells);
        CMFMultiGPUReport one_apply_report{}, four_apply_report{};
        CMFMultiGPUStatus one_apply_status = apply_bounds(
            one_j.data(), one_lower.data(), one_nearest.data(),
            one_upper.data(), 1, &one_apply_report);
        CMFMultiGPUStatus four_apply_status = apply_bounds(
            one_j.data(), four_lower.data(), four_nearest.data(),
            four_upper.data(), requested_devices, &four_apply_report);
        if (one_apply_status != CMF_MGPU_OK ||
            four_apply_status != CMF_MGPU_OK) {
            double one_min = std::numeric_limits<double>::infinity();
            double one_max = 0.0;
            for (double value : one_nearest) {
                one_min = std::min(one_min, value);
                one_max = std::max(one_max, value);
            }
            std::fprintf(
                stderr,
                "CMF_MGPU_REDUCED_APPLY_DIAG_FAIL status_one/four=%s/%s "
                "phase_one/four=%d/%d round_one/four=%d/%d "
                "failure_cell=%zu failure_shell/bin=%zu/%zu "
                "failure_device=%d rays=[%d,%d) "
                "sweep_stage=%d segment=%d local_ray=%d bin=%d "
                "outward=%d "
                "active/positive_rays=%zu/%zu max_ray=%d "
                "ray_in/out=%.17g/%.17g host_partial=%.17g "
                "device_geometry_partial/rmid/impact_diff="
                "%.17g/%.17g/%.17g "
                "failure_lower/nearest/upper=%.17g/%.17g/%.17g "
                "one_nearest_min/max=%.17g/%.17g\n",
                cmf_multigpu_status_name(one_apply_status),
                cmf_multigpu_status_name(four_apply_status),
                one_apply_report.failure_phase,
                four_apply_report.failure_phase,
                one_apply_report.failure_iteration,
                four_apply_report.failure_iteration,
                four_apply_report.failure_cell_index,
                four_apply_report.failure_cell_index / (size_t)nb,
                four_apply_report.failure_cell_index % (size_t)nb,
                four_apply_report.failure_device_index,
                four_apply_report.failure_ray_begin,
                four_apply_report.failure_ray_end,
                four_apply_report.failure_sweep_stage,
                four_apply_report.failure_segment_index,
                four_apply_report.failure_local_ray_index,
                four_apply_report.failure_bin_index,
                four_apply_report.failure_outward,
                four_apply_report.failure_active_ray_count,
                four_apply_report.failure_positive_intensity_count,
                four_apply_report.failure_global_ray_index,
                four_apply_report.failure_ray_in,
                four_apply_report.failure_ray_out,
                four_apply_report.failure_host_recomputed_partial,
                four_apply_report.failure_device_geometry_partial,
                four_apply_report.failure_device_rmid,
                four_apply_report.failure_max_impact_absolute_difference,
                four_apply_report.failure_lower,
                four_apply_report.failure_nearest,
                four_apply_report.failure_upper,
                one_min, one_max);
            return 1;
        }
        double max_relative = 0.0;
        double max_absolute = 0.0;
        size_t max_index = 0U;
        size_t ordering_failures = 0U;
        for (size_t index = 0; index < cells; ++index) {
            double difference = std::fabs(
                one_nearest[index] - four_nearest[index]);
            double denominator = std::max(
                std::fabs(one_nearest[index]),
                std::fabs(four_nearest[index]));
            double relative = denominator > 0.0
                            ? difference / denominator : 0.0;
            if (relative > max_relative) {
                max_relative = relative;
                max_index = index;
            }
            max_absolute = std::max(max_absolute, difference);
            if (!(one_lower[index] <= one_nearest[index] &&
                  one_nearest[index] <= one_upper[index] &&
                  four_lower[index] <= four_nearest[index] &&
                  four_nearest[index] <= four_upper[index]))
                ++ordering_failures;
        }
        std::printf(
            "CMF_MGPU_REDUCED_APPLY_DIAG PASS shells=%d bins=%d cells=%zu "
            "max_rel_one_four=%.17g max_abs_one_four=%.17g "
            "max_shell/bin=%zu/%zu one/four=%.17g/%.17g "
            "ordering_failures=%zu "
            "ray_work_one_owned_min/max=%zu/%zu "
            "ray_work_four_owned_min/max=%zu/%zu "
            "ray_work_four_computed_min/max=%zu/%zu "
            "numerical_repairs=0\n",
            ns, nb, cells, max_relative, max_absolute,
            max_index / (size_t)nb, max_index % (size_t)nb,
            one_nearest[max_index], four_nearest[max_index],
            ordering_failures,
            one_apply_report.min_owned_ray_segment_work,
            one_apply_report.max_owned_ray_segment_work,
            four_apply_report.min_owned_ray_segment_work,
            four_apply_report.max_owned_ray_segment_work,
            four_apply_report.min_computed_ray_segment_work,
            four_apply_report.max_computed_ray_segment_work);
        return ordering_failures == 0U ? 0 : 1;
    }

    if (split_one || split_four) {
        const int devices = split_one ? 1 : requested_devices;
        std::vector<double> &j = split_one ? one_j : four_j;
        std::vector<double> &error = split_one ? one_error : four_error;
        CMFMultiGPUReport report{};
        auto start = std::chrono::steady_clock::now();
        CMFMultiGPUStatus status = solve_envelope(
            j.data(), error.data(), devices, &report);
        auto end = std::chrono::steady_clock::now();
        if (status != CMF_MGPU_OK) {
            std::fprintf(stderr,
                         "CMF_MGPU_REDUCED_FAIL mode=%s status=%s "
                         "iterations=%d failure_phase=%d "
                         "failure_iteration=%d final_rel=%.17g "
                         "ray_work_owned_min/max=%zu/%zu "
                         "ray_work_computed_min/max=%zu/%zu\n",
                         mode, cmf_multigpu_status_name(status),
                         report.iterations_used, report.failure_phase,
                         report.failure_iteration,
                         report.final_max_relative_change,
                         report.min_owned_ray_segment_work,
                         report.max_owned_ray_segment_work,
                         report.min_computed_ray_segment_work,
                         report.max_computed_ray_segment_work);
            return 1;
        }
        if (report.n_rays != (size_t)(ns + 16) ||
            report.max_positive_window_bins != expected_window ||
            std::fabs(report.max_characteristic_drift_bins - expected_drift) >
                1.0e-12 * expected_drift ||
            report.persistent_context_initializations != 1U ||
            report.componentwise_error_envelope_verified != 1 ||
            report.devices_used != devices ||
            report.epoch_frequency_parallel != (use_epoch ? 1 : 0) ||
            report.weighted_contiguous_ray_partition !=
                (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS ?
                 1 : 0))
            fail("split-solver-report-contract");
        if (devices == 1) {
            if (report.min_owned_ray_segment_work !=
                    kProductionRaySegmentWork ||
                report.max_owned_ray_segment_work !=
                    kProductionRaySegmentWork ||
                report.min_computed_ray_segment_work !=
                    kProductionRaySegmentWork ||
                report.max_computed_ray_segment_work !=
                    kProductionRaySegmentWork)
                fail("split-one-ray-work-contract");
        } else if (devices == 4) {
            const bool weighted =
                partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS;
            const size_t expected_owned_min = weighted
                ? kFourMinOwnedRaySegmentWork
                : kEqualFourMinOwnedRaySegmentWork;
            const size_t expected_owned_max = weighted
                ? kFourMaxOwnedRaySegmentWork
                : kEqualFourMaxOwnedRaySegmentWork;
            const size_t expected_computed_min = weighted
                ? kFourMinComputedRaySegmentWork
                : kEqualFourMinComputedRaySegmentWork;
            const size_t expected_computed_max = weighted
                ? kFourMaxComputedRaySegmentWork
                : kEqualFourMaxComputedRaySegmentWork;
            if (report.min_owned_ray_segment_work != expected_owned_min ||
                report.max_owned_ray_segment_work != expected_owned_max ||
                report.min_computed_ray_segment_work !=
                    expected_computed_min ||
                report.max_computed_ray_segment_work !=
                    expected_computed_max)
                fail("split-four-ray-work-contract");
        }
        double j_min = std::numeric_limits<double>::infinity();
        double j_max = 0.0;
        double error_min = std::numeric_limits<double>::infinity();
        double error_max = 0.0;
        for (size_t index = 0; index < cells; ++index) {
            double value = j[index], bound = error[index];
            if (!(value >= 0.0) || !(bound >= 0.0) ||
                !std::isfinite(value) || !std::isfinite(bound))
                fail("split-nonfinite-or-negative-result");
            j_min = std::min(j_min, value);
            j_max = std::max(j_max, value);
            error_min = std::min(error_min, bound);
            error_max = std::max(error_max, bound);
        }
        uint64_t hash = UINT64_C(14695981039346656037);
        hash = fnv1a_update(hash, j.data(), cells * sizeof(double));
        hash = fnv1a_update(hash, error.data(), cells * sizeof(double));
        const char *output_path = std::getenv("CMF_MGPU_REDUCED_OUTPUT");
        if (!write_single_result(output_path, ns, nb, devices, j, error))
            fail("split-result-write");
        double seconds = std::chrono::duration<double>(end - start).count();
        std::printf(
            "CMF_MGPU_REDUCED_SPLIT_RESULT PASS mode=%s partition=%s epoch=%d "
            "shells=%d bins=%d "
            "cells=%zu rays=%zu dlognu=%.17g drift=%.17g max_window=%zu "
            "devices=%d iterations=%d contexts=%zu bounds=%zu upper=%zu "
            "ray_work_owned_min/max=%zu/%zu "
            "ray_work_computed_min/max=%zu/%zu "
            "J_min/max=%.17g/%.17g error_min/max=%.17g/%.17g "
            "alloc_max_bytes=%zu result_fnv64=%016llx "
            "numerical_repairs=0\n",
            mode, partition_name, use_epoch ? 1 : 0,
            ns, nb, cells, report.n_rays,
            dlognu, expected_drift,
            expected_window, devices, report.iterations_used,
            report.persistent_context_initializations,
            report.persistent_bound_applications,
            report.persistent_upper_operator_applications,
            report.min_owned_ray_segment_work,
            report.max_owned_ray_segment_work,
            report.min_computed_ray_segment_work,
            report.max_computed_ray_segment_work,
            j_min, j_max, error_min, error_max,
            report.max_device_allocated_bytes,
            (unsigned long long)hash);
        std::printf(
            "CMF_MGPU_REDUCED_SPLIT_TIMING mode=%s partition=%s "
            "seconds=%.9f\n",
            mode, partition_name, seconds);
        return 0;
    }

    CMFMultiGPUReport one_report{};
    auto one_start = std::chrono::steady_clock::now();
    CMFMultiGPUStatus one_status = solve_envelope(
        one_j.data(), one_error.data(), 1, &one_report);
    auto one_end = std::chrono::steady_clock::now();
    if (one_status != CMF_MGPU_OK) {
        std::fprintf(stderr,
                     "CMF_MGPU_REDUCED_FAIL one-device status=%s\n",
                     cmf_multigpu_status_name(one_status));
        return 1;
    }

    CMFMultiGPUReport four_report{};
    auto four_start = std::chrono::steady_clock::now();
    CMFMultiGPUStatus four_status = solve_envelope(
        four_j.data(), four_error.data(), requested_devices, &four_report);
    auto four_end = std::chrono::steady_clock::now();
    if (four_status != CMF_MGPU_OK) {
        std::fprintf(stderr,
                     "CMF_MGPU_REDUCED_FAIL multi-device status=%s\n",
                     cmf_multigpu_status_name(four_status));
        return 1;
    }

    if (one_report.n_rays != (size_t)(ns + 16) ||
        four_report.n_rays != one_report.n_rays ||
        one_report.max_positive_window_bins != expected_window ||
        four_report.max_positive_window_bins != expected_window ||
        std::fabs(one_report.max_characteristic_drift_bins - expected_drift) >
            1.0e-12 * expected_drift ||
        std::fabs(four_report.max_characteristic_drift_bins - expected_drift) >
            1.0e-12 * expected_drift ||
        one_report.persistent_context_initializations != 1U ||
        four_report.persistent_context_initializations != 1U ||
        one_report.componentwise_error_envelope_verified != 1 ||
        four_report.componentwise_error_envelope_verified != 1 ||
        one_report.epoch_frequency_parallel != (use_epoch ? 1 : 0) ||
        four_report.epoch_frequency_parallel != (use_epoch ? 1 : 0) ||
        one_report.devices_used != 1 ||
        four_report.devices_used != requested_devices ||
        one_report.weighted_contiguous_ray_partition !=
            (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS ? 1 : 0) ||
        four_report.weighted_contiguous_ray_partition !=
            (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS ? 1 : 0) ||
        one_report.min_owned_ray_segment_work !=
            kProductionRaySegmentWork ||
        one_report.max_owned_ray_segment_work !=
            kProductionRaySegmentWork ||
        one_report.min_computed_ray_segment_work !=
            kProductionRaySegmentWork ||
        one_report.max_computed_ray_segment_work !=
            kProductionRaySegmentWork ||
        four_report.min_owned_ray_segment_work !=
            (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS
             ? kFourMinOwnedRaySegmentWork
             : kEqualFourMinOwnedRaySegmentWork) ||
        four_report.max_owned_ray_segment_work !=
            (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS
             ? kFourMaxOwnedRaySegmentWork
             : kEqualFourMaxOwnedRaySegmentWork) ||
        four_report.min_computed_ray_segment_work !=
            (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS
             ? kFourMinComputedRaySegmentWork
             : kEqualFourMinComputedRaySegmentWork) ||
        four_report.max_computed_ray_segment_work !=
            (partition_mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS
             ? kFourMaxComputedRaySegmentWork
             : kEqualFourMaxComputedRaySegmentWork))
        fail("solver-report-contract");

    double max_relative = 0.0;
    double max_difference_envelope_ratio = 0.0;
    double j_min = std::numeric_limits<double>::infinity();
    double j_max = 0.0;
    double error_min = std::numeric_limits<double>::infinity();
    double error_max = 0.0;
    size_t covered = 0;
    for (size_t index = 0; index < cells; ++index) {
        double a = one_j[index], b = four_j[index];
        double ua = one_error[index], ub = four_error[index];
        if (!(a >= 0.0) || !(b >= 0.0) || !(ua >= 0.0) || !(ub >= 0.0) ||
            !std::isfinite(a) || !std::isfinite(b) ||
            !std::isfinite(ua) || !std::isfinite(ub))
            fail("nonfinite-or-negative-result");
        double difference = std::fabs(a - b);
        double denominator = std::max(std::fabs(a), std::fabs(b));
        double relative = denominator > 0.0 ? difference / denominator : 0.0;
        max_relative = std::max(max_relative, relative);
        double combined = std::nextafter(ua + ub,
                                         std::numeric_limits<double>::infinity());
        if (difference <= combined) ++covered;
        if (combined > 0.0)
            max_difference_envelope_ratio = std::max(
                max_difference_envelope_ratio, difference / combined);
        j_min = std::min(j_min, b);
        j_max = std::max(j_max, b);
        error_min = std::min(error_min, ub);
        error_max = std::max(error_max, ub);
    }
    if (covered != cells) fail("one-four-envelope-coverage");

    uint64_t hash = UINT64_C(14695981039346656037);
    hash = fnv1a_update(hash, one_j.data(), cells * sizeof(double));
    hash = fnv1a_update(hash, one_error.data(), cells * sizeof(double));
    hash = fnv1a_update(hash, four_j.data(), cells * sizeof(double));
    hash = fnv1a_update(hash, four_error.data(), cells * sizeof(double));
    const char *output_path = std::getenv("CMF_MGPU_REDUCED_OUTPUT");
    if (!write_result(output_path, ns, nb, one_j, one_error,
                      four_j, four_error))
        fail("result-write");

    double one_seconds = std::chrono::duration<double>(
        one_end - one_start).count();
    double four_seconds = std::chrono::duration<double>(
        four_end - four_start).count();
    std::printf(
        "CMF_MGPU_REDUCED_RESULT PASS epoch=%d shells=%d bins=%d cells=%zu rays=%zu "
        "dlognu=%.17g drift=%.17g max_window=%zu devices=%d "
        "iterations_one/four=%d/%d contexts_one/four=%zu/%zu "
        "bounds_one/four=%zu/%zu upper_one/four=%zu/%zu "
        "ray_work_owned_one_min/max=%zu/%zu "
        "ray_work_owned_four_min/max=%zu/%zu "
        "ray_work_computed_one_min/max=%zu/%zu "
        "ray_work_computed_four_min/max=%zu/%zu "
        "max_rel_one_four=%.17g envelope_ratio=%.17g covered=%zu/%zu "
        "J_min/max=%.17g/%.17g error_min/max=%.17g/%.17g "
        "alloc_one/four_max_bytes=%zu/%zu result_fnv64=%016llx "
        "numerical_repairs=0\n",
        use_epoch ? 1 : 0, ns, nb, cells, four_report.n_rays,
        dlognu, expected_drift,
        expected_window, requested_devices, one_report.iterations_used,
        four_report.iterations_used,
        one_report.persistent_context_initializations,
        four_report.persistent_context_initializations,
        one_report.persistent_bound_applications,
        four_report.persistent_bound_applications,
        one_report.persistent_upper_operator_applications,
        four_report.persistent_upper_operator_applications,
        one_report.min_owned_ray_segment_work,
        one_report.max_owned_ray_segment_work,
        four_report.min_owned_ray_segment_work,
        four_report.max_owned_ray_segment_work,
        one_report.min_computed_ray_segment_work,
        one_report.max_computed_ray_segment_work,
        four_report.min_computed_ray_segment_work,
        four_report.max_computed_ray_segment_work,
        max_relative, max_difference_envelope_ratio, covered, cells,
        j_min, j_max, error_min, error_max,
        one_report.max_device_allocated_bytes,
        four_report.max_device_allocated_bytes,
        (unsigned long long)hash);
    std::printf(
        "CMF_MGPU_REDUCED_TIMING one_seconds=%.9f four_seconds=%.9f "
        "speedup=%.9f\n",
        one_seconds, four_seconds,
        four_seconds > 0.0 ? one_seconds / four_seconds : 0.0);
    return 0;
}
