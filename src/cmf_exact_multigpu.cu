#include "cmf_exact_multigpu.h"
#include "cmf_error_envelope.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr double kC = 2.99792458e10;
constexpr double kH = 6.62607015e-27;
constexpr double kKB = 1.380649e-16;

static double monotonic_seconds()
{
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct HostGeometry {
    int ns = 0;
    int nb = 0;
    int nr = 0;
    int stride = 0;
    double a_lam = 0.0;
    double a_drift = 0.0;
    std::vector<double> rmid;
    std::vector<double> impact;
    std::vector<int> rn;
    std::vector<int> shell;
    std::vector<double> beta;
    std::vector<int> core;
};

/* The layout mirrors the CPU PositiveWindowNode, but is kept private so the
 * prototype does not expose or silently become the production owner. */
struct PositiveWindowNode {
    double value_transmission;
    double value_emission;
    double aggregate_transmission;
    double aggregate_emission;
};

struct PositiveTransformPair {
    double transmission;
    double emission;
};

struct DeviceShard {
    int device = -1;
    int ray_begin = 0;
    int ray_end = 0;
    int compute_end = 0;
    int local_rays = 0;
    size_t allocated_bytes = 0;
    int *rn = nullptr;
    int *segment_offset = nullptr;
    std::vector<int> host_segment_offset;
    int *shell = nullptr;
    int *core = nullptr;
    double *beta = nullptr;
    double *impact = nullptr;
    double *rmid = nullptr;
    double *dt1 = nullptr;
    double *t1 = nullptr;
    double *source = nullptr;
    double *source_cell = nullptr;
    double *inner = nullptr;
    double *in = nullptr;
    double *out = nullptr;
    double *partial = nullptr;
    PositiveWindowNode *positive_window = nullptr;
    PositiveTransformPair *epoch_front = nullptr;
    PositiveTransformPair *epoch_back = nullptr;
    size_t epoch_workspace_span = 0;
    int *failure = nullptr;
};

static bool valid_epoch_schedule(const CMFMultiGPUEpochSchedule *schedule)
{
    if (!schedule) return false;
    int block = schedule->block_size;
    return (block == 32 || block == 64 || block == 128 || block == 256) &&
           schedule->epoch_batch_cardinality > 0 &&
           schedule->direct_replay_max_window >= 1;
}

static bool checked_mul(size_t a, size_t b, size_t *out)
{
    if (!out || (a != 0 && b > std::numeric_limits<size_t>::max() / a))
        return false;
    *out = a * b;
    return true;
}

static double planck(double nu, double temperature)
{
    if (!(nu > 0.0) || !(temperature > 0.0) ||
        !std::isfinite(nu) || !std::isfinite(temperature)) return NAN;
    double x = kH * nu / (kKB * temperature);
    double prefactor = 2.0 * kH * nu * nu * nu / (kC * kC);
    if (!(x > 0.0) || !std::isfinite(x) || !std::isfinite(prefactor))
        return NAN;
    double value;
    if (x > 50.0) {
        double e = std::exp(-x);
        value = prefactor * e / (-std::expm1(-x));
    } else {
        value = prefactor / std::expm1(x);
    }
    return value >= 0.0 && std::isfinite(value) ? value : NAN;
}

static bool build_geometry(HostGeometry *g, int ns, int nb, double dlognu,
                           const double *r_inner, const double *r_outer,
                           double time_explosion)
{
    constexpr int ncore = 16;
    if (!g || ns <= 0 || nb < 2 || !(dlognu > 0.0) ||
        !(time_explosion > 0.0)) return false;
    g->ns = ns;
    g->nb = nb;
    g->nr = ns + ncore;
    g->stride = ns + 1;
    g->a_lam = 1.0 / (time_explosion * kC);
    g->a_drift = g->a_lam / dlognu;
    size_t slots;
    if (!checked_mul((size_t)g->nr, (size_t)g->stride, &slots)) return false;
    g->rmid.resize((size_t)ns);
    g->impact.resize((size_t)g->nr);
    g->rn.assign((size_t)g->nr, 0);
    g->shell.assign(slots, -1);
    g->beta.assign(slots, 0.0);
    g->core.assign((size_t)g->nr, 0);
    for (int s = 0; s < ns; ++s) {
        if (!(r_inner[s] > 0.0) || !(r_outer[s] > r_inner[s]) ||
            !std::isfinite(r_inner[s]) || !std::isfinite(r_outer[s]) ||
            (s > 0 && (r_inner[s] < r_inner[s - 1] ||
                       r_outer[s] <= r_outer[s - 1]))) return false;
        g->rmid[(size_t)s] = 0.5 * (r_inner[s] + r_outer[s]);
    }
    for (int k = 0; k < ncore; ++k)
        g->impact[(size_t)k] = g->rmid[0] * (double)k / (double)ncore;
    for (int s = 0; s < ns; ++s)
        g->impact[(size_t)(ncore + s)] = g->rmid[(size_t)s];

    double max_drift = 0.0;
    for (int k = 0; k < g->nr; ++k) {
        double p = g->impact[(size_t)k];
        std::vector<double> z;
        std::vector<int> shells;
        for (int s = ns - 1; s >= 0; --s) {
            if (g->rmid[(size_t)s] <= p) break;
            shells.push_back(s);
            z.push_back(std::sqrt(g->rmid[(size_t)s] * g->rmid[(size_t)s] - p * p));
        }
        int n = (int)shells.size();
        g->rn[(size_t)k] = n;
        g->core[(size_t)k] = p < g->rmid[0] ? 1 : 0;
        double zin = g->core[(size_t)k]
                   ? std::sqrt(g->rmid[0] * g->rmid[0] - p * p) : 0.0;
        double drift = 0.0;
        for (int i = 0; i < n; ++i) {
            size_t at = (size_t)k * (size_t)g->stride + (size_t)i;
            double ds = i + 1 < n ? z[(size_t)i] - z[(size_t)i + 1]
                                  : z[(size_t)i] - zin;
            double b = g->a_drift * ds;
            if (!(b >= 0.0) || !std::isfinite(b)) return false;
            g->shell[at] = shells[(size_t)i];
            g->beta[at] = b;
            drift += b;
        }
        if (drift > max_drift) max_drift = drift;
    }
    (void)max_drift;
    return true;
}

struct RayPartition {
    std::vector<int> boundary;
    std::vector<size_t> owned_segment_work;
    std::vector<size_t> computed_segment_work;
};

static size_t absolute_size_difference(size_t a, size_t b)
{
    return a >= b ? a - b : b - a;
}

/* Keep every shard contiguous, but place boundaries at cumulative active
 * ray/segment quantiles instead of equal ray counts.  The right boundary ray
 * remains the recomputed halo on every non-final shard. */
static bool build_ray_partition(
    const HostGeometry &geometry, int devices,
    CMFMultiGPUPartitionMode mode, RayPartition *partition)
{
    if (!partition || devices <= 0 || devices > geometry.nr ||
        geometry.nr <= 0 || geometry.rn.size() != (size_t)geometry.nr ||
        (mode != CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS &&
         mode != CMF_MGPU_PARTITION_EQUAL_RAYS))
        return false;
    try {
        std::vector<size_t> prefix((size_t)geometry.nr + 1U, 0U);
        for (int ray = 0; ray < geometry.nr; ++ray) {
            int work = geometry.rn[(size_t)ray];
            if (work < 0 || prefix[(size_t)ray] >
                    std::numeric_limits<size_t>::max() - (size_t)work)
                return false;
            prefix[(size_t)ray + 1U] =
                prefix[(size_t)ray] + (size_t)work;
        }
        size_t total = prefix[(size_t)geometry.nr];
        if (total == 0U) return false;

        partition->boundary.assign((size_t)devices + 1U, 0);
        partition->owned_segment_work.assign((size_t)devices, 0U);
        partition->computed_segment_work.assign((size_t)devices, 0U);
        partition->boundary[0] = 0;
        partition->boundary[(size_t)devices] = geometry.nr;
        if (mode == CMF_MGPU_PARTITION_EQUAL_RAYS) {
            for (int device = 1; device < devices; ++device) {
                partition->boundary[(size_t)device] =
                    (geometry.nr * device) / devices;
            }
        } else {
            size_t quotient = total / (size_t)devices;
            size_t remainder = total % (size_t)devices;
            for (int device = 1; device < devices; ++device) {
                int minimum = partition->boundary[(size_t)device - 1U] + 1;
                int maximum = geometry.nr - (devices - device);
                size_t target = quotient * (size_t)device +
                    (remainder * (size_t)device) / (size_t)devices;
                int upper = minimum;
                while (upper < maximum && prefix[(size_t)upper] < target)
                    ++upper;
                int chosen = upper;
                if (upper > minimum) {
                    int lower = upper - 1;
                    size_t lower_distance = absolute_size_difference(
                        prefix[(size_t)lower], target);
                    size_t upper_distance = absolute_size_difference(
                        prefix[(size_t)upper], target);
                    if (lower_distance <= upper_distance) chosen = lower;
                }
                partition->boundary[(size_t)device] = chosen;
            }
        }

        for (int device = 0; device < devices; ++device) {
            int begin = partition->boundary[(size_t)device];
            int end = partition->boundary[(size_t)device + 1U];
            if (begin < 0 || end <= begin || end > geometry.nr)
                return false;
            size_t owned = prefix[(size_t)end] - prefix[(size_t)begin];
            size_t computed = owned;
            if (end < geometry.nr) {
                size_t halo = (size_t)geometry.rn[(size_t)end];
                if (computed >
                    std::numeric_limits<size_t>::max() - halo)
                    return false;
                computed += halo;
            }
            partition->owned_segment_work[(size_t)device] = owned;
            partition->computed_segment_work[(size_t)device] = computed;
        }
    } catch (const std::bad_alloc &) {
        return false;
    }
    return true;
}

static void record_ray_partition(
    CMFMultiGPUReport *report, const RayPartition &partition,
    CMFMultiGPUPartitionMode mode)
{
    report->weighted_contiguous_ray_partition =
        mode == CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS ? 1 : 0;
    report->min_owned_ray_segment_work =
        std::numeric_limits<size_t>::max();
    report->min_computed_ray_segment_work =
        std::numeric_limits<size_t>::max();
    for (size_t work : partition.owned_segment_work) {
        report->min_owned_ray_segment_work = std::min(
            report->min_owned_ray_segment_work, work);
        report->max_owned_ray_segment_work = std::max(
            report->max_owned_ray_segment_work, work);
    }
    for (size_t work : partition.computed_segment_work) {
        report->min_computed_ray_segment_work = std::min(
            report->min_computed_ray_segment_work, work);
        report->max_computed_ray_segment_work = std::max(
            report->max_computed_ray_segment_work, work);
    }
    report->device_partition_count = (int)std::min(
        partition.owned_segment_work.size(),
        (size_t)CMF_MGPU_REPORT_MAX_DEVICES);
    for (int device = 0; device < report->device_partition_count; ++device) {
        report->device_ray_begin[device] =
            partition.boundary[(size_t)device];
        report->device_ray_end[device] =
            partition.boundary[(size_t)device + 1U];
        report->device_owned_segment_work[device] =
            partition.owned_segment_work[(size_t)device];
        report->device_computed_segment_work[device] =
            partition.computed_segment_work[(size_t)device];
    }
}

static void cleanup_shard(DeviceShard *s)
{
    if (!s || s->device < 0) return;
    (void)cudaSetDevice(s->device);
    cudaFree(s->rn);
    cudaFree(s->segment_offset);
    cudaFree(s->shell);
    cudaFree(s->core);
    cudaFree(s->beta);
    cudaFree(s->impact);
    cudaFree(s->rmid);
    cudaFree(s->dt1);
    cudaFree(s->t1);
    cudaFree(s->source);
    cudaFree(s->source_cell);
    cudaFree(s->inner);
    cudaFree(s->in);
    cudaFree(s->out);
    cudaFree(s->partial);
    cudaFree(s->positive_window);
    cudaFree(s->epoch_front);
    cudaFree(s->epoch_back);
    cudaFree(s->failure);
    *s = DeviceShard{};
}

static bool build_compact_shard_geometry(
    const HostGeometry &geometry, DeviceShard *shard,
    std::vector<int> *compact_shell, std::vector<double> *compact_beta)
{
    if (!shard || !compact_shell || !compact_beta ||
        shard->local_rays <= 0 || shard->ray_begin < 0 ||
        shard->ray_begin + shard->local_rays > geometry.nr)
        return false;
    try {
        shard->host_segment_offset.assign(
            (size_t)shard->local_rays + 1U, 0);
        size_t total = 0U;
        for (int local_ray = 0; local_ray < shard->local_rays; ++local_ray) {
            int ray = shard->ray_begin + local_ray;
            int active = geometry.rn[(size_t)ray];
            if (active < 0 || total > (size_t)INT_MAX - (size_t)active)
                return false;
            total += (size_t)active;
            shard->host_segment_offset[(size_t)local_ray + 1U] =
                (int)total;
        }
        compact_shell->resize(total);
        compact_beta->resize(total);
        for (int local_ray = 0; local_ray < shard->local_rays; ++local_ray) {
            int ray = shard->ray_begin + local_ray;
            size_t target = (size_t)shard->host_segment_offset[
                (size_t)local_ray];
            for (int segment = 0;
                 segment < geometry.rn[(size_t)ray]; ++segment) {
                size_t source = (size_t)ray * (size_t)geometry.stride +
                                (size_t)segment;
                compact_shell->at(target + (size_t)segment) =
                    geometry.shell[source];
                compact_beta->at(target + (size_t)segment) =
                    geometry.beta[source];
            }
        }
    } catch (const std::bad_alloc &) {
        return false;
    }
    return true;
}

template <typename T>
static cudaError_t device_allocate(T **pointer, size_t count, DeviceShard *s)
{
    size_t bytes;
    if (!pointer || !s || !checked_mul(count, sizeof(T), &bytes))
        return cudaErrorInvalidValue;
    cudaError_t rc = cudaMalloc((void **)pointer, bytes);
    if (rc == cudaSuccess) s->allocated_bytes += bytes;
    return rc;
}

static cudaError_t allocate_positive_sweep_workspace(
    DeviceShard *shard, int n_bins, int max_positive_window,
    const CMFMultiGPUEpochSchedule *epoch_schedule, size_t *workspace_bytes)
{
    if (!shard || n_bins < 2 || max_positive_window < 0)
        return cudaErrorInvalidValue;
    size_t before = shard->allocated_bytes;
    if (!epoch_schedule) {
        size_t nodes;
        if (!checked_mul((size_t)shard->local_rays, 2U, &nodes) ||
            !checked_mul(nodes, (size_t)max_positive_window, &nodes))
            return cudaErrorInvalidValue;
        if (nodes != 0 &&
            device_allocate(&shard->positive_window, nodes, shard) !=
                cudaSuccess)
            return cudaErrorMemoryAllocation;
    } else if (max_positive_window >
               epoch_schedule->direct_replay_max_window) {
        if ((size_t)n_bins > std::numeric_limits<size_t>::max() -
                                 (size_t)max_positive_window)
            return cudaErrorInvalidValue;
        shard->epoch_workspace_span =
            (size_t)n_bins + (size_t)max_positive_window;
        size_t nodes;
        if (!checked_mul((size_t)shard->local_rays,
                         shard->epoch_workspace_span, &nodes))
            return cudaErrorInvalidValue;
        if (device_allocate(&shard->epoch_front, nodes, shard) != cudaSuccess ||
            device_allocate(&shard->epoch_back, nodes, shard) != cudaSuccess)
            return cudaErrorMemoryAllocation;
    }
    if (workspace_bytes) *workspace_bytes = shard->allocated_bytes - before;
    return cudaSuccess;
}

__device__ __forceinline__ int clip_bin(int b, int nb)
{
    return b < 0 ? 0 : (b >= nb ? nb - 1 : b);
}

__device__ double direct_segment_value(
    int b, int nb, double beta, const double *dt1, const double *source,
    const double *upstream, bool upstream_zero)
{
    int q = (int)floor(beta);
    double phi = beta - (double)q;
    double intensity = 0.0;
    if (!upstream_zero) {
        int i0 = clip_bin(b + q, nb);
        int i1 = clip_bin(b + q + 1, nb);
        intensity = (1.0 - phi) * upstream[i0] + phi * upstream[i1];
    }
    if (beta <= 0.5) {
        double transmission = exp(-dt1[b] * beta);
        return intensity * transmission +
               (1.0 - transmission) * source[b];
    }
    double x = (double)b + beta;
    int m = (int)floor(x + 0.5);
    double cursor = x;
    for (;;) {
        double lower = fmax((double)m - 0.5, (double)b);
        double length = cursor - lower;
        if (length > 0.0) {
            int mm = clip_bin(m, nb);
            double transmission = exp(-dt1[mm] * length);
            intensity = intensity * transmission +
                        (1.0 - transmission) * source[mm];
        }
        if (lower <= (double)b + 1.0e-12) break;
        cursor = lower;
        --m;
    }
    return intensity;
}

__device__ __forceinline__ bool positive_finite(double value)
{
    return value >= 0.0 && isfinite(value);
}

constexpr int kFailureWords = 6;

__device__ __forceinline__ void record_positive_failure(
    int *failure, int sweep_stage, int first_index, int second_index,
    int outward, int rounding)
{
    if (!failure) return;
    if (atomicCAS(failure, 0, 2) == 0) {
        failure[1] = sweep_stage;
        failure[2] = first_index;
        failure[3] = second_index;
        failure[4] = outward;
        failure[5] = rounding;
    }
}

/* For a finite nonnegative IEEE binary64 value, increasing/decreasing the
 * unsigned payload by one is exactly nextafter(value,+inf/0).  Keeping this
 * device-only avoids CUDA 13 selecting C++23's host-only constexpr overload. */
__device__ __forceinline__ double positive_next_up(double value)
{
    if (value == 0.0) return __longlong_as_double(1LL);
    unsigned long long bits =
        (unsigned long long)__double_as_longlong(value);
    return __longlong_as_double((long long)(bits + 1ULL));
}

__device__ __forceinline__ double positive_next_down(double value)
{
    if (value == 0.0) return 0.0;
    unsigned long long bits =
        (unsigned long long)__double_as_longlong(value);
    return __longlong_as_double((long long)(bits - 1ULL));
}

__device__ bool positive_add_bound_device(
    double a, double b, int rounding, double *result)
{
    if (!result || !positive_finite(a) || !positive_finite(b)) return false;
    double sum = a + b;
    if (!positive_finite(sum)) return false;
    if (rounding != 0 && a != 0.0 && b != 0.0) {
        double b_virtual = sum - a;
        double error = (a - (sum - b_virtual)) + (b - b_virtual);
        if (rounding > 0 && error > 0.0)
            sum = positive_next_up(sum);
        else if (rounding < 0 && error < 0.0)
            sum = positive_next_down(sum);
    }
    if (!positive_finite(sum)) return false;
    *result = sum;
    return true;
}

__device__ bool positive_multiply_bound_device(
    double a, double b, int rounding, double *result)
{
    if (!result || !positive_finite(a) || !positive_finite(b)) return false;
    if (a == 0.0 || b == 0.0) {
        *result = 0.0;
        return true;
    }
    double product = a * b;
    if (!isfinite(product)) return false;
    if (rounding > 0)
        product = positive_next_up(product);
    else if (rounding < 0 && product != 0.0)
        product = positive_next_down(product);
    if (!positive_finite(product)) return false;
    *result = product;
    return true;
}

__device__ bool positive_two_product_sum_device(
    double a, double x, double b, double y, int rounding, double *result)
{
    if (rounding == 0) {
        double value = a * x + b * y;
        if (!positive_finite(value)) return false;
        *result = value;
        return true;
    }
    double ax;
    double by;
    return positive_multiply_bound_device(a, x, rounding, &ax) &&
           positive_multiply_bound_device(b, y, rounding, &by) &&
           positive_add_bound_device(ax, by, rounding, result);
}

__device__ bool positive_reverse_compose(
    double a_transmission, double a_emission,
    double b_transmission, double b_emission,
    int rounding, double *transmission, double *emission)
{
    if (!positive_finite(a_transmission) ||
        !positive_finite(a_emission) ||
        !positive_finite(b_transmission) ||
        !positive_finite(b_emission) || !transmission || !emission)
        return false;
    double attenuated;
    return positive_multiply_bound_device(
               b_transmission, a_transmission, rounding,
               transmission) &&
           positive_multiply_bound_device(
               b_transmission, a_emission, rounding, &attenuated) &&
           positive_add_bound_device(
               b_emission, attenuated, rounding, emission);
}

__device__ bool positive_window_push_back(
    PositiveWindowNode *back, int capacity, int *back_size,
    double transmission, double emission, int rounding)
{
    if (!back_size || *back_size < 0 || *back_size >= capacity ||
        !positive_finite(transmission) || !positive_finite(emission))
        return false;
    PositiveWindowNode &node = back[*back_size];
    node.value_transmission = transmission;
    node.value_emission = emission;
    if (*back_size == 0) {
        node.aggregate_transmission = transmission;
        node.aggregate_emission = emission;
    } else {
        PositiveWindowNode &previous = back[*back_size - 1];
        if (!positive_reverse_compose(
                previous.aggregate_transmission,
                previous.aggregate_emission, transmission, emission, rounding,
                &node.aggregate_transmission,
                &node.aggregate_emission)) return false;
    }
    ++*back_size;
    return true;
}

__device__ bool positive_window_transfer(
    PositiveWindowNode *front, PositiveWindowNode *back, int capacity,
    int *front_size, int *back_size, int rounding)
{
    if (!front_size || !back_size || *front_size != 0) return false;
    while (*back_size != 0) {
        PositiveWindowNode value = back[--*back_size];
        if (*front_size >= capacity) return false;
        PositiveWindowNode &node = front[*front_size];
        node.value_transmission = value.value_transmission;
        node.value_emission = value.value_emission;
        if (*front_size == 0) {
            node.aggregate_transmission = value.value_transmission;
            node.aggregate_emission = value.value_emission;
        } else {
            PositiveWindowNode &previous = front[*front_size - 1];
            if (!positive_reverse_compose(
                    value.value_transmission, value.value_emission,
                    previous.aggregate_transmission,
                    previous.aggregate_emission, rounding,
                    &node.aggregate_transmission,
                    &node.aggregate_emission)) return false;
        }
        ++*front_size;
    }
    return true;
}

__device__ bool positive_window_pop_front(
    PositiveWindowNode *front, PositiveWindowNode *back, int capacity,
    int *front_size, int *back_size, int rounding)
{
    if (*front_size == 0 && !positive_window_transfer(
            front, back, capacity, front_size, back_size, rounding))
        return false;
    if (*front_size == 0) return false;
    --*front_size;
    return true;
}

__device__ bool positive_window_aggregate(
    const PositiveWindowNode *front, const PositiveWindowNode *back,
    int front_size, int back_size,
    int rounding, double *transmission, double *emission)
{
    if (!transmission || !emission || front_size < 0 || back_size < 0)
        return false;
    if (front_size == 0 && back_size == 0) {
        *transmission = 1.0;
        *emission = 0.0;
        return true;
    }
    if (front_size == 0) {
        *transmission = back[back_size - 1].aggregate_transmission;
        *emission = back[back_size - 1].aggregate_emission;
        return positive_finite(*transmission) && positive_finite(*emission);
    }
    if (back_size == 0) {
        *transmission = front[front_size - 1].aggregate_transmission;
        *emission = front[front_size - 1].aggregate_emission;
        return positive_finite(*transmission) && positive_finite(*emission);
    }
    return positive_reverse_compose(
        front[front_size - 1].aggregate_transmission,
        front[front_size - 1].aggregate_emission,
        back[back_size - 1].aggregate_transmission,
        back[back_size - 1].aggregate_emission,
        rounding,
        transmission, emission);
}

__device__ bool positive_segment_values(
    int nb, double beta, const double *dt1, const double *t1,
    const double *source, const double *source_cell,
    const double *upstream, bool upstream_zero, double *output,
    PositiveWindowNode *front, PositiveWindowNode *back,
    int workspace_capacity, int rounding)
{
    int q = (int)floor(beta);
    double phi = beta - (double)q;
    if (!(q >= 0) || !(phi >= 0.0) || !(phi < 1.0) || !isfinite(phi))
        return false;
    if (beta <= 0.5) {
        for (int b = 0; b < nb; ++b) {
            double intensity = 0.0;
            if (!upstream_zero) {
                int i0 = clip_bin(b + q, nb);
                int i1 = clip_bin(b + q + 1, nb);
                if (!positive_two_product_sum_device(
                        1.0 - phi, upstream[i0], phi, upstream[i1],
                        rounding, &intensity)) return false;
            }
            double transmission = exp(-dt1[b] * beta);
            double value;
            if (!positive_two_product_sum_device(
                    transmission, intensity, 1.0 - transmission, source[b],
                    rounding, &value)) return false;
            output[b] = value;
        }
        return true;
    }

    int qtop;
    double psi;
    if (phi < 0.5) {
        qtop = q;
        psi = phi + 0.5;
    } else {
        qtop = q + 1;
        psi = phi - 0.5;
    }
    int window_count = qtop >= 2 ? qtop - 1 : 0;
    if (window_count > workspace_capacity ||
        (window_count != 0 && (!front || !back))) return false;
    int front_size = 0;
    int back_size = 0;
    int b = nb - 1;
    for (int m = b + qtop - 1; m >= b + 1; --m) {
        int mm = clip_bin(m, nb);
        if (!positive_window_push_back(
                back, window_count, &back_size,
                t1[mm], source_cell[mm], rounding)) return false;
    }
    for (; b >= 0; --b) {
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = clip_bin(b + q, nb);
            int i1 = clip_bin(b + q + 1, nb);
            if (!positive_two_product_sum_device(
                    1.0 - phi, upstream[i0], phi, upstream[i1],
                    rounding, &intensity)) return false;
        }
        int top = clip_bin(b + qtop, nb);
        double transmission = exp(-psi * dt1[top]);
        if (!positive_two_product_sum_device(
                transmission, intensity, 1.0 - transmission, source[top],
                rounding, &intensity)) return false;
        double aggregate_transmission;
        double aggregate_emission;
        if (!positive_finite(intensity) || !positive_window_aggregate(
                front, back, front_size, back_size, rounding,
                &aggregate_transmission, &aggregate_emission)) return false;
        if (!positive_two_product_sum_device(
                aggregate_transmission, intensity, 1.0,
                aggregate_emission, rounding, &intensity)) return false;
        double half = sqrt(t1[b]);
        double value;
        if (!positive_two_product_sum_device(
                half, intensity, 1.0 - half, source[b], rounding,
                &value)) return false;
        output[b] = value;
        if (b == 0 || window_count == 0) continue;
        if (!positive_window_pop_front(
                front, back, window_count,
                &front_size, &back_size, rounding) ||
            !positive_window_push_back(
                back, window_count, &back_size,
                t1[b], source_cell[b], rounding)) return false;
    }
    return true;
}

__global__ void positive_segment_kernel(
    int nb, int local_rays, int segment, bool outward,
    int max_window, const int *rn, const int *segment_offset,
    const int *shell, const int *core,
    const double *beta, const double *dt1, const double *t1,
    const double *source, const double *source_cell, const double *inner,
    double *in, double *out, PositiveWindowNode *window, int *failure,
    int rounding)
{
    int lr = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (lr >= local_rays) return;
    int n = rn[lr];
    if (segment >= n) return;
    size_t segment_slot =
        (size_t)segment_offset[lr] + (size_t)segment;
    int s = shell[segment_slot];
    const double *dt = dt1 + (size_t)s * (size_t)nb;
    const double *tr = t1 + (size_t)s * (size_t)nb;
    const double *src = source + (size_t)s * (size_t)nb;
    const double *src_cell = source_cell + (size_t)s * (size_t)nb;
    const double *upstream = nullptr;
    bool upstream_zero = false;
    if (!outward) {
        if (segment == 0) upstream_zero = true;
        else upstream = in + (segment_slot - 1U) * (size_t)nb;
    } else if (segment == n - 1) {
        upstream = core[lr] ? inner : in + segment_slot * (size_t)nb;
    } else {
        upstream = out + (segment_slot + 1U) * (size_t)nb;
    }
    PositiveWindowNode *front = nullptr;
    PositiveWindowNode *back = nullptr;
    if (max_window != 0) {
        size_t base = (size_t)lr * 2U * (size_t)max_window;
        front = window + base;
        back = front + max_window;
    }
    double *destination = outward ? out : in;
    if (!positive_segment_values(
            nb, beta[segment_slot], dt, tr, src, src_cell,
            upstream, upstream_zero,
            destination + segment_slot * (size_t)nb,
            front, back, max_window, rounding)) {
        record_positive_failure(
            failure, 1, segment, lr, outward ? 1 : 0, rounding);
    }
}

__device__ __forceinline__ PositiveTransformPair positive_pair_at(
    int index, int nb, const double *t1, const double *source_cell)
{
    int bounded = clip_bin(index, nb);
    PositiveTransformPair value = {t1[bounded], source_cell[bounded]};
    return value;
}

__device__ __forceinline__ bool positive_pair_reverse_compose(
    PositiveTransformPair a, PositiveTransformPair b, int rounding,
    PositiveTransformPair *result)
{
    return result && positive_reverse_compose(
        a.transmission, a.emission, b.transmission, b.emission,
        rounding, &result->transmission, &result->emission);
}

__device__ bool positive_epoch_pointwise(
    int b, int nb, double beta, const double *dt1, const double *t1,
    const double *source, const double *upstream, bool upstream_zero,
    PositiveTransformPair aggregate, int rounding, double *output)
{
    if (!output) return false;
    int q = (int)floor(beta);
    double phi = beta - (double)q;
    int qtop;
    double psi;
    if (phi < 0.5) {
        qtop = q;
        psi = phi + 0.5;
    } else {
        qtop = q + 1;
        psi = phi - 0.5;
    }
    double intensity = 0.0;
    if (!upstream_zero) {
        int i0 = clip_bin(b + q, nb);
        int i1 = clip_bin(b + q + 1, nb);
        if (!positive_two_product_sum_device(
                1.0 - phi, upstream[i0], phi, upstream[i1],
                rounding, &intensity)) return false;
    }
    int top = clip_bin(b + qtop, nb);
    double transmission = exp(-psi * dt1[top]);
    if (!positive_two_product_sum_device(
            transmission, intensity, 1.0 - transmission, source[top],
            rounding, &intensity) ||
        !positive_two_product_sum_device(
            aggregate.transmission, intensity, 1.0,
            aggregate.emission, rounding, &intensity)) return false;
    double half = sqrt(t1[b]);
    return positive_two_product_sum_device(
        half, intensity, 1.0 - half, source[b], rounding, output);
}

__global__ void positive_epoch_replay_segment_kernel(
    int nb, int local_rays, int segment, bool outward,
    int replay_max_window, const int *rn, const int *segment_offset,
    const int *shell,
    const int *core, const double *beta, const double *dt1,
    const double *t1, const double *source, const double *source_cell,
    const double *inner, double *in, double *out, int *failure,
    int rounding)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)local_rays * (size_t)nb;
    if (tid >= total) return;
    int lr = (int)(tid / (size_t)nb);
    int b = (int)(tid - (size_t)lr * (size_t)nb);
    int n = rn[lr];
    if (segment >= n) return;
    size_t segment_slot =
        (size_t)segment_offset[lr] + (size_t)segment;
    int s = shell[segment_slot];
    const double *dt = dt1 + (size_t)s * (size_t)nb;
    const double *tr = t1 + (size_t)s * (size_t)nb;
    const double *src = source + (size_t)s * (size_t)nb;
    const double *src_cell = source_cell + (size_t)s * (size_t)nb;
    const double *upstream = nullptr;
    bool upstream_zero = false;
    if (!outward) {
        if (segment == 0) upstream_zero = true;
        else upstream = in + (segment_slot - 1U) * (size_t)nb;
    } else if (segment == n - 1) {
        upstream = core[lr] ? inner : in + segment_slot * (size_t)nb;
    } else {
        upstream = out + (segment_slot + 1U) * (size_t)nb;
    }
    double ray_beta = beta[segment_slot];
    int q = (int)floor(ray_beta);
    double phi = ray_beta - (double)q;
    if (!(q >= 0) || !(phi >= 0.0) || !(phi < 1.0) || !isfinite(phi)) {
        record_positive_failure(
            failure, 30, segment, lr, outward ? 1 : 0, rounding);
        return;
    }
    double *destination = outward ? out : in;
    double *target = destination + segment_slot * (size_t)nb;
    if (ray_beta <= 0.5) {
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = clip_bin(b + q, nb);
            int i1 = clip_bin(b + q + 1, nb);
            if (!positive_two_product_sum_device(
                    1.0 - phi, upstream[i0], phi, upstream[i1],
                    rounding, &intensity)) {
                record_positive_failure(
                    failure, 31, segment, lr, outward ? 1 : 0, rounding);
                return;
            }
        }
        double transmission = exp(-dt[b] * ray_beta);
        if (!positive_two_product_sum_device(
                transmission, intensity, 1.0 - transmission, src[b],
                rounding, &target[b]))
            record_positive_failure(
                failure, 32, segment, lr, outward ? 1 : 0, rounding);
        return;
    }
    int qtop = phi < 0.5 ? q : q + 1;
    int window = qtop >= 2 ? qtop - 1 : 0;
    if (window > replay_max_window) return;
    PositiveTransformPair aggregate = {1.0, 0.0};
    if (window > 0) {
        int highest = nb - 1;
        int epoch = (highest - b) / window;
        int boundary_bin = highest - epoch * window;
        int offset = boundary_bin - b;
        if (offset == 0) {
            aggregate = positive_pair_at(
                boundary_bin + window, nb, tr, src_cell);
            for (int node = 1; node < window; ++node) {
                PositiveTransformPair next;
                if (!positive_pair_reverse_compose(
                        aggregate,
                        positive_pair_at(
                            boundary_bin + window - node,
                            nb, tr, src_cell),
                        rounding, &next)) {
                    record_positive_failure(
                        failure, 33, segment, lr,
                        outward ? 1 : 0, rounding);
                    return;
                }
                aggregate = next;
            }
        } else {
            int front_target = window - offset - 1;
            PositiveTransformPair front = positive_pair_at(
                boundary_bin + 1, nb, tr, src_cell);
            for (int node = 1; node <= front_target; ++node) {
                PositiveTransformPair next;
                if (!positive_pair_reverse_compose(
                        positive_pair_at(
                            boundary_bin + 1 + node, nb, tr, src_cell),
                        front, rounding, &next)) {
                    record_positive_failure(
                        failure, 34, segment, lr,
                        outward ? 1 : 0, rounding);
                    return;
                }
                front = next;
            }
            PositiveTransformPair back = positive_pair_at(
                boundary_bin, nb, tr, src_cell);
            for (int node = 1; node < offset; ++node) {
                PositiveTransformPair next;
                if (!positive_pair_reverse_compose(
                        back,
                        positive_pair_at(
                            boundary_bin - node, nb, tr, src_cell),
                        rounding, &next)) {
                    record_positive_failure(
                        failure, 35, segment, lr,
                        outward ? 1 : 0, rounding);
                    return;
                }
                back = next;
            }
            if (!positive_pair_reverse_compose(
                    front, back, rounding, &aggregate)) {
                record_positive_failure(
                    failure, 36, segment, lr,
                    outward ? 1 : 0, rounding);
                return;
            }
        }
    }
    if (!positive_epoch_pointwise(
            b, nb, ray_beta, dt, tr, src, upstream, upstream_zero,
            aggregate, rounding, &target[b]))
        record_positive_failure(
            failure, 37, segment, lr, outward ? 1 : 0, rounding);
}

__global__ void positive_epoch_large_segment_kernel(
    int nb, int local_rays, int segment, bool outward,
    int replay_max_window, int epoch_begin, size_t workspace_span,
    const int *rn, const int *segment_offset, const int *shell,
    const int *core, const double *beta,
    const double *dt1, const double *t1, const double *source,
    const double *source_cell, const double *inner,
    double *in, double *out, PositiveTransformPair *front_workspace,
    PositiveTransformPair *back_workspace, int *failure, int rounding)
{
    extern __shared__ double shared_values[];
    PositiveTransformPair *boundary_value =
        reinterpret_cast<PositiveTransformPair *>(shared_values);
    int lr = (int)blockIdx.y;
    if (lr >= local_rays) return;
    int n = rn[lr];
    if (segment >= n) return;
    size_t segment_slot =
        (size_t)segment_offset[lr] + (size_t)segment;
    double ray_beta = beta[segment_slot];
    int q = (int)floor(ray_beta);
    double phi = ray_beta - (double)q;
    int qtop = phi < 0.5 ? q : q + 1;
    int window = qtop >= 2 ? qtop - 1 : 0;
    if (window <= replay_max_window) return;
    int epoch = epoch_begin + (int)blockIdx.x;
    int boundary_bin = nb - 1 - epoch * window;
    if (boundary_bin < 0) return;
    int epoch_outputs = window;
    if (boundary_bin + 1 < epoch_outputs) epoch_outputs = boundary_bin + 1;
    size_t epoch_offset = (size_t)epoch * (size_t)window;
    if (epoch_offset + (size_t)window > workspace_span) {
        if (threadIdx.x == 0)
            record_positive_failure(
                failure, 40, segment, lr, outward ? 1 : 0, rounding);
        return;
    }
    PositiveTransformPair *front = front_workspace +
        (size_t)lr * workspace_span + epoch_offset;
    PositiveTransformPair *new_back = back_workspace +
        (size_t)lr * workspace_span + epoch_offset;
    int s = shell[segment_slot];
    const double *dt = dt1 + (size_t)s * (size_t)nb;
    const double *tr = t1 + (size_t)s * (size_t)nb;
    const double *src = source + (size_t)s * (size_t)nb;
    const double *src_cell = source_cell + (size_t)s * (size_t)nb;
    bool okay = true;
    if (threadIdx.x == 0) {
        PositiveTransformPair aggregate = positive_pair_at(
            boundary_bin + window, nb, tr, src_cell);
        for (int node = 1; node < window; ++node) {
            PositiveTransformPair next;
            okay = okay && positive_pair_reverse_compose(
                aggregate,
                positive_pair_at(
                    boundary_bin + window - node, nb, tr, src_cell),
                rounding, &next);
            aggregate = next;
        }
        *boundary_value = aggregate;
    } else if (threadIdx.x == 1 && epoch_outputs > 1) {
        PositiveTransformPair aggregate = positive_pair_at(
            boundary_bin + 1, nb, tr, src_cell);
        front[0] = aggregate;
        for (int node = 1; node < window; ++node) {
            PositiveTransformPair next;
            okay = okay && positive_pair_reverse_compose(
                positive_pair_at(
                    boundary_bin + 1 + node, nb, tr, src_cell),
                aggregate, rounding, &next);
            aggregate = next;
            front[node] = aggregate;
        }
    } else if (threadIdx.x == 2 && epoch_outputs > 1) {
        PositiveTransformPair aggregate = positive_pair_at(
            boundary_bin, nb, tr, src_cell);
        new_back[0] = aggregate;
        for (int node = 1; node < epoch_outputs - 1; ++node) {
            PositiveTransformPair next;
            okay = okay && positive_pair_reverse_compose(
                aggregate,
                positive_pair_at(
                    boundary_bin - node, nb, tr, src_cell),
                rounding, &next);
            aggregate = next;
            new_back[node] = aggregate;
        }
    }
    if (!okay)
        record_positive_failure(
            failure, 41, segment, lr, outward ? 1 : 0, rounding);
    __syncthreads();
    if (*failure != 0) return;

    const double *upstream = nullptr;
    bool upstream_zero = false;
    if (!outward) {
        if (segment == 0) upstream_zero = true;
        else upstream = in + (segment_slot - 1U) * (size_t)nb;
    } else if (segment == n - 1) {
        upstream = core[lr] ? inner : in + segment_slot * (size_t)nb;
    } else {
        upstream = out + (segment_slot + 1U) * (size_t)nb;
    }
    double *destination = outward ? out : in;
    double *target = destination + segment_slot * (size_t)nb;
    for (int offset = (int)threadIdx.x;
         offset < epoch_outputs; offset += (int)blockDim.x) {
        int b = boundary_bin - offset;
        PositiveTransformPair aggregate;
        if (offset == 0)
            aggregate = *boundary_value;
        else if (!positive_pair_reverse_compose(
                     front[window - offset - 1], new_back[offset - 1],
                     rounding, &aggregate)) {
            record_positive_failure(
                failure, 42, segment, lr, outward ? 1 : 0, rounding);
            continue;
        }
        if (!positive_epoch_pointwise(
                b, nb, ray_beta, dt, tr, src, upstream, upstream_zero,
                aggregate, rounding, &target[b]))
            record_positive_failure(
                failure, 43, segment, lr, outward ? 1 : 0, rounding);
    }
}

__global__ void segment_kernel(
    int nb, int local_rays, int segment, bool outward,
    const int *rn, const int *segment_offset, const int *shell,
    const int *core, const double *beta,
    const double *dt1, const double *source, const double *inner,
    double *in, double *out)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)local_rays * (size_t)nb;
    if (tid >= total) return;
    int lr = (int)(tid / (size_t)nb);
    int b = (int)(tid - (size_t)lr * (size_t)nb);
    int n = rn[lr];
    if (segment >= n) return;
    size_t segment_slot =
        (size_t)segment_offset[lr] + (size_t)segment;
    int s = shell[segment_slot];
    const double *dt = dt1 + (size_t)s * (size_t)nb;
    const double *src = source + (size_t)s * (size_t)nb;
    const double *upstream = nullptr;
    bool upstream_zero = false;
    if (!outward) {
        if (segment == 0) {
            upstream_zero = true;
        } else {
            upstream = in + (segment_slot - 1U) * (size_t)nb;
        }
    } else if (segment == n - 1) {
        upstream = core[lr] ? inner : in + segment_slot * (size_t)nb;
    } else {
        upstream = out + (segment_slot + 1U) * (size_t)nb;
    }
    double value = direct_segment_value(
        b, nb, beta[segment_slot], dt, src, upstream, upstream_zero);
    double *destination = outward ? out : in;
    destination[segment_slot * (size_t)nb + (size_t)b] = value;
}

__global__ void partial_j_kernel(
    int ns, int nb, int nr, int ray_begin, int ray_end,
    const int *rn, const int *segment_offset,
    const double *impact, const double *rmid,
    const double *in, const double *out, double *partial)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t cells = (size_t)ns * (size_t)nb;
    if (tid >= cells) return;
    int s = (int)(tid / (size_t)nb);
    int b = (int)(tid - (size_t)s * (size_t)nb);
    int segment = ns - 1 - s;
    double sum = 0.0;
    for (int ray = ray_end - 1; ray >= ray_begin; --ray) {
        int lr = ray - ray_begin;
        if (segment >= rn[lr]) continue;
        size_t at = ((size_t)segment_offset[lr] + (size_t)segment) *
                    (size_t)nb + (size_t)b;
        double current_j = 0.5 * (out[at] + in[at]);
        double current_mu_squared =
            1.0 - impact[ray] * impact[ray] / (rmid[s] * rmid[s]);
        if (!(current_mu_squared >= 0.0) ||
            !isfinite(current_mu_squared)) {
            partial[tid] = NAN;
            return;
        }
        double current_mu = sqrt(current_mu_squared);
        double previous_j = 0.0;
        double previous_mu = 0.0;
        int previous_ray = ray + 1;
        bool has_previous_sample = false;
        if (previous_ray < nr) {
            int previous_local = previous_ray - ray_begin;
            if (previous_local >= 0 && segment < rn[previous_local]) {
                size_t previous_at =
                    ((size_t)segment_offset[previous_local] +
                     (size_t)segment) * (size_t)nb + (size_t)b;
                previous_j = 0.5 * (out[previous_at] + in[previous_at]);
                double previous_mu_squared =
                    1.0 - impact[previous_ray] * impact[previous_ray] /
                          (rmid[s] * rmid[s]);
                if (!(previous_mu_squared >= 0.0) ||
                    !isfinite(previous_mu_squared)) {
                    partial[tid] = NAN;
                    return;
                }
                previous_mu = sqrt(previous_mu_squared);
                has_previous_sample = true;
            }
        }
        /* Match the canonical angular reconstruction exactly at the tangent
         * boundary: its first sampled intensity is held constant from mu=0
         * to the first ray.  This is a quadrature boundary condition, not a
         * floor or a replacement of a computed physical value. */
        if (!has_previous_sample) previous_j = current_j;
        sum += 0.5 * (previous_j + current_j) *
               (current_mu - previous_mu);
    }
    partial[tid] = sum;
}

__global__ void partial_j_bound_kernel(
    int ns, int nb, int nr, int ray_begin, int ray_end,
    const int *rn, const int *segment_offset,
    const double *impact, const double *rmid,
    const double *in, const double *out, double *partial,
    int *failure, int rounding)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t cells = (size_t)ns * (size_t)nb;
    if (tid >= cells) return;
    int s = (int)(tid / (size_t)nb);
    int b = (int)(tid - (size_t)s * (size_t)nb);
    int segment = ns - 1 - s;
    double sum = 0.0;
    for (int ray = ray_end - 1; ray >= ray_begin; --ray) {
        int lr = ray - ray_begin;
        if (segment >= rn[lr]) continue;
        size_t at = ((size_t)segment_offset[lr] + (size_t)segment) *
                    (size_t)nb + (size_t)b;
        double current_j;
        if (!positive_two_product_sum_device(
                0.5, out[at], 0.5, in[at], rounding, &current_j)) {
            record_positive_failure(failure, 21, s, b, 0, rounding);
            return;
        }
        double current_mu_squared =
            1.0 - impact[ray] * impact[ray] / (rmid[s] * rmid[s]);
        if (!(current_mu_squared >= 0.0) ||
            !isfinite(current_mu_squared)) {
            record_positive_failure(failure, 26, s, b, 0, rounding);
            return;
        }
        double current_mu = sqrt(current_mu_squared);
        double previous_j = current_j;
        double previous_mu = 0.0;
        int previous_ray = ray + 1;
        if (previous_ray < nr) {
            int previous_local = previous_ray - ray_begin;
            if (previous_local >= 0 && segment < rn[previous_local]) {
                size_t previous_at =
                    ((size_t)segment_offset[previous_local] +
                     (size_t)segment) * (size_t)nb + (size_t)b;
                if (!positive_two_product_sum_device(
                        0.5, out[previous_at], 0.5, in[previous_at],
                        rounding, &previous_j)) {
                    record_positive_failure(
                        failure, 22, s, b, 0, rounding);
                    return;
                }
                double previous_mu_squared =
                    1.0 - impact[previous_ray] * impact[previous_ray] /
                          (rmid[s] * rmid[s]);
                if (!(previous_mu_squared >= 0.0) ||
                    !isfinite(previous_mu_squared)) {
                    record_positive_failure(
                        failure, 27, s, b, 0, rounding);
                    return;
                }
                previous_mu = sqrt(previous_mu_squared);
            }
        }
        double average;
        double weighted;
        double next_sum;
        if (!positive_add_bound_device(
                previous_j, current_j, rounding, &average)) {
            record_positive_failure(failure, 23, s, b, 0, rounding);
            return;
        }
        if (!positive_multiply_bound_device(
                0.5 * (current_mu - previous_mu), average,
                rounding, &weighted)) {
            record_positive_failure(failure, 24, s, b, 0, rounding);
            return;
        }
        if (!positive_add_bound_device(
                sum, weighted, rounding, &next_sum)) {
            record_positive_failure(failure, 25, s, b, 0, rounding);
            return;
        }
        sum = next_sum;
    }
    partial[tid] = sum;
}

/* 0=success, 1=CUDA/runtime failure, 2=invalid positive recurrence. */
struct SweepFailureDetail {
    int device = -1;
    int sweep_stage = 0;
    int first_index = -1;
    int second_index = -1;
    int outward = -1;
    int rounding = 0;
};

static int max_large_epochs_for_segment(
    const HostGeometry &geometry, const DeviceShard &shard, int segment,
    int replay_max_window)
{
    int maximum = 0;
    for (int local_ray = 0; local_ray < shard.local_rays; ++local_ray) {
        int ray = shard.ray_begin + local_ray;
        if (segment >= geometry.rn[(size_t)ray]) continue;
        double beta = geometry.beta[
            (size_t)ray * (size_t)geometry.stride + (size_t)segment];
        int q = (int)std::floor(beta);
        double phi = beta - (double)q;
        int qtop = phi < 0.5 ? q : q + 1;
        int window = qtop >= 2 ? qtop - 1 : 0;
        if (window <= replay_max_window) continue;
        int epochs = (geometry.nb + window - 1) / window;
        if (epochs > maximum) maximum = epochs;
    }
    return maximum;
}

static int launch_positive_segment(
    DeviceShard &shard, const HostGeometry &geometry, int segment,
    bool outward, int max_positive_window, int rounding,
    const CMFMultiGPUEpochSchedule *epoch_schedule)
{
    if (!epoch_schedule) {
        constexpr int threads = 256;
        int blocks = (shard.local_rays + threads - 1) / threads;
        positive_segment_kernel<<<blocks, threads>>>(
            geometry.nb, shard.local_rays, segment, outward,
            max_positive_window, shard.rn, shard.segment_offset,
            shard.shell, shard.core,
            shard.beta, shard.dt1, shard.t1, shard.source,
            shard.source_cell, shard.inner, shard.in, shard.out,
            shard.positive_window, shard.failure, rounding);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    int threads = epoch_schedule->block_size;
    size_t total = (size_t)shard.local_rays * (size_t)geometry.nb;
    int blocks = (int)((total + (size_t)threads - 1U) / (size_t)threads);
    positive_epoch_replay_segment_kernel<<<blocks, threads>>>(
        geometry.nb, shard.local_rays, segment, outward,
        epoch_schedule->direct_replay_max_window,
        shard.rn, shard.segment_offset, shard.shell, shard.core,
        shard.beta, shard.dt1, shard.t1,
        shard.source, shard.source_cell, shard.inner, shard.in, shard.out,
        shard.failure, rounding);
    if (cudaGetLastError() != cudaSuccess) return 1;

    int maximum_epochs = max_large_epochs_for_segment(
        geometry, shard, segment,
        epoch_schedule->direct_replay_max_window);
    for (int epoch_begin = 0; epoch_begin < maximum_epochs;
         epoch_begin += epoch_schedule->epoch_batch_cardinality) {
        int count = std::min(
            epoch_schedule->epoch_batch_cardinality,
            maximum_epochs - epoch_begin);
        dim3 grid((unsigned int)count, (unsigned int)shard.local_rays, 1U);
        positive_epoch_large_segment_kernel<<<
            grid, threads, sizeof(PositiveTransformPair)>>>(
            geometry.nb, shard.local_rays, segment,
            outward, epoch_schedule->direct_replay_max_window, epoch_begin,
            shard.epoch_workspace_span, shard.rn, shard.segment_offset,
            shard.shell, shard.core, shard.beta, shard.dt1, shard.t1,
            shard.source,
            shard.source_cell, shard.inner, shard.in, shard.out,
            shard.epoch_front, shard.epoch_back, shard.failure, rounding);
        if (cudaGetLastError() != cudaSuccess) return 1;
    }
    return 0;
}

static int launch_sweep(DeviceShard *shards, int ndev,
                        const HostGeometry &g, size_t cells,
                        bool positive_sliding, int max_positive_window,
                        int rounding, SweepFailureDetail *detail = nullptr,
                        const CMFMultiGPUEpochSchedule *epoch_schedule = nullptr)
{
    constexpr int threads = 256;
    if (positive_sliding) {
        for (int d = 0; d < ndev; ++d) {
            DeviceShard &s = shards[d];
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemset(s.failure, 0,
                           (size_t)kFailureWords * sizeof(int)) != cudaSuccess)
                return 1;
        }
    }
    for (int segment = 0; segment < g.ns; ++segment) {
        for (int d = 0; d < ndev; ++d) {
            DeviceShard &s = shards[d];
            if (cudaSetDevice(s.device) != cudaSuccess) return 1;
            if (positive_sliding) {
                if (launch_positive_segment(
                        s, g, segment, false, max_positive_window,
                        rounding, epoch_schedule) != 0) return 1;
            } else {
                size_t total = (size_t)s.local_rays * (size_t)g.nb;
                int blocks = (int)((total + threads - 1U) / threads);
                segment_kernel<<<blocks, threads>>>(
                    g.nb, s.local_rays, segment, false,
                    s.rn, s.segment_offset, s.shell, s.core, s.beta,
                    s.dt1, s.source, s.inner,
                    s.in, s.out);
            }
            if (!positive_sliding && cudaGetLastError() != cudaSuccess)
                return 1;
        }
    }
    for (int segment = g.ns - 1; segment >= 0; --segment) {
        for (int d = 0; d < ndev; ++d) {
            DeviceShard &s = shards[d];
            if (cudaSetDevice(s.device) != cudaSuccess) return 1;
            if (positive_sliding) {
                if (launch_positive_segment(
                        s, g, segment, true, max_positive_window,
                        rounding, epoch_schedule) != 0) return 1;
            } else {
                size_t total = (size_t)s.local_rays * (size_t)g.nb;
                int blocks = (int)((total + threads - 1U) / threads);
                segment_kernel<<<blocks, threads>>>(
                    g.nb, s.local_rays, segment, true,
                    s.rn, s.segment_offset, s.shell, s.core, s.beta,
                    s.dt1, s.source, s.inner,
                    s.in, s.out);
            }
            if (!positive_sliding && cudaGetLastError() != cudaSuccess)
                return 1;
        }
    }
    if (positive_sliding) {
        for (int d = 0; d < ndev; ++d) {
            DeviceShard &s = shards[d];
            int failure[kFailureWords] = {0, 0, 0, 0, 0, 0};
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemcpy(failure, s.failure, sizeof(failure),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return 1;
            if (failure[0] != 0) {
                if (detail) {
                    detail->device = d;
                    detail->sweep_stage = failure[1];
                    detail->first_index = failure[2];
                    detail->second_index = failure[3];
                    detail->outward = failure[4];
                    detail->rounding = failure[5];
                }
                return 2;
            }
        }
    }
    for (int d = 0; d < ndev; ++d) {
        DeviceShard &s = shards[d];
        if (cudaSetDevice(s.device) != cudaSuccess) return 1;
        int blocks = (int)((cells + threads - 1U) / threads);
        if (positive_sliding) {
            partial_j_bound_kernel<<<blocks, threads>>>(
                g.ns, g.nb, g.nr, s.ray_begin, s.ray_end,
                s.rn, s.segment_offset, s.impact, s.rmid,
                s.in, s.out, s.partial,
                s.failure, rounding);
        } else {
            partial_j_kernel<<<blocks, threads>>>(
                g.ns, g.nb, g.nr, s.ray_begin, s.ray_end,
                s.rn, s.segment_offset, s.impact, s.rmid,
                s.in, s.out, s.partial);
        }
        if (cudaGetLastError() != cudaSuccess) return 1;
    }
    if (positive_sliding) {
        for (int d = 0; d < ndev; ++d) {
            DeviceShard &s = shards[d];
            int failure[kFailureWords] = {0, 0, 0, 0, 0, 0};
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemcpy(failure, s.failure, sizeof(failure),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return 1;
            if (failure[0] != 0) {
                if (detail) {
                    detail->device = d;
                    detail->sweep_stage = failure[1];
                    detail->first_index = failure[2];
                    detail->second_index = failure[3];
                    detail->outward = failure[4];
                    detail->rounding = failure[5];
                }
                return 2;
            }
        }
    }
    return 0;
}

static void record_sweep_failure_detail(
    CMFMultiGPUReport *report, const DeviceShard *shards, int ndev,
    int n_bins, const SweepFailureDetail &detail)
{
    if (!report) return;
    report->failure_sweep_stage = detail.sweep_stage;
    report->failure_outward = detail.outward;
    if (detail.device >= 0 && detail.device < ndev) {
        report->failure_device_index = detail.device;
        report->failure_ray_begin = shards[detail.device].ray_begin;
        report->failure_ray_end = shards[detail.device].ray_end;
    }
    if (detail.sweep_stage == 1 ||
        (detail.sweep_stage >= 30 && detail.sweep_stage <= 43)) {
        report->failure_segment_index = detail.first_index;
        report->failure_local_ray_index = detail.second_index;
    } else if (detail.sweep_stage >= 20 &&
               detail.sweep_stage <= 27) {
        report->failure_bin_index = detail.second_index;
        if (detail.first_index >= 0 && detail.second_index >= 0)
            report->failure_cell_index =
                (size_t)detail.first_index * (size_t)n_bins +
                (size_t)detail.second_index;
    }
}

static bool host_positive_add_bound(
    double a, double b, int rounding, double *result)
{
    if (!result || !(a >= 0.0) || !(b >= 0.0) ||
        !std::isfinite(a) || !std::isfinite(b)) return false;
    double sum = a + b;
    if (!std::isfinite(sum)) return false;
    if (rounding != 0 && a != 0.0 && b != 0.0) {
        double b_virtual = sum - a;
        double error = (a - (sum - b_virtual)) + (b - b_virtual);
        if (rounding > 0 && error > 0.0)
            sum = std::nextafter(sum, INFINITY);
        else if (rounding < 0 && error < 0.0)
            sum = std::nextafter(sum, 0.0);
    }
    if (!(sum >= 0.0) || !std::isfinite(sum)) return false;
    *result = sum;
    return true;
}

static bool host_positive_multiply_bound(
    double a, double b, int rounding, double *result)
{
    if (!result || !(a >= 0.0) || !(b >= 0.0) ||
        !std::isfinite(a) || !std::isfinite(b)) return false;
    if (a == 0.0 || b == 0.0) {
        *result = 0.0;
        return true;
    }
    double product = a * b;
    if (!std::isfinite(product)) return false;
    if (rounding > 0)
        product = std::nextafter(product, INFINITY);
    else if (rounding < 0)
        product = std::nextafter(product, 0.0);
    if (!(product >= 0.0) || !std::isfinite(product)) return false;
    *result = product;
    return true;
}

static bool host_absolute_difference_upper(double a, double b, double *upper)
{
    if (!upper || !std::isfinite(a) || !std::isfinite(b)) return false;
    double negative_b = -b;
    double difference = a + negative_b;
    if (!std::isfinite(difference)) return false;
    double b_virtual = difference - a;
    double error = (a - (difference - b_virtual)) +
                   (negative_b - b_virtual);
    return host_positive_add_bound(
        std::fabs(difference), std::fabs(error), 1, upper);
}

}  // namespace

extern "C" const char *cmf_multigpu_status_name(CMFMultiGPUStatus status)
{
    switch (status) {
    case CMF_MGPU_OK: return "OK";
    case CMF_MGPU_INVALID_INPUT: return "INVALID_INPUT";
    case CMF_MGPU_CUDA_UNAVAILABLE: return "CUDA_UNAVAILABLE";
    case CMF_MGPU_INSUFFICIENT_DEVICES: return "INSUFFICIENT_DEVICES";
    case CMF_MGPU_ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case CMF_MGPU_CUDA_FAILURE: return "CUDA_FAILURE";
    case CMF_MGPU_NONFINITE: return "NONFINITE";
    case CMF_MGPU_NOT_CONVERGED: return "NOT_CONVERGED";
    case CMF_MGPU_NEGATIVE_RECURRENCE: return "NEGATIVE_RECURRENCE";
    case CMF_MGPU_ERROR_ENVELOPE_FAILED: return "ERROR_ENVELOPE_FAILED";
    default: return "UNKNOWN";
    }
}

static CMFMultiGPUStatus solve_impl(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report, bool positive_sliding,
    CMFMultiGPUPartitionMode partition_mode,
    const CMFMultiGPUEpochSchedule *epoch_schedule)
{
    const double total_start = monotonic_seconds();
    CMFMultiGPUReport local{};
    local.status = CMF_MGPU_INVALID_INPUT;
    local.iteration_cap = iteration_cap;
    local.tolerance = tolerance;
    local.final_max_relative_change = INFINITY;
    local.final_max_absolute_change = INFINITY;
    local.max_scattering_ratio = INFINITY;
    local.fixed_point_absolute_error_bound = INFINITY;
    local.positive_sliding = positive_sliding ? 1 : 0;
    local.epoch_frequency_parallel = epoch_schedule ? 1 : 0;
    if (epoch_schedule) {
        local.epoch_block_size = epoch_schedule->block_size;
        local.epoch_batch_cardinality =
            epoch_schedule->epoch_batch_cardinality;
        local.epoch_direct_replay_max_window =
            epoch_schedule->direct_replay_max_window;
    }
    local.failure_iteration = -1;
    local.failure_cell_index = std::numeric_limits<size_t>::max();
    local.failure_device_index = -1;
    local.failure_ray_begin = -1;
    local.failure_ray_end = -1;
    local.failure_segment_index = -1;
    local.failure_local_ray_index = -1;
    local.failure_bin_index = -1;
    local.failure_outward = -1;
    local.failure_global_ray_index = -1;
    if (report) *report = local;
    if ((epoch_schedule &&
         (!positive_sliding || !valid_epoch_schedule(epoch_schedule))) ||
        (partition_mode != CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS &&
         partition_mode != CMF_MGPU_PARTITION_EQUAL_RAYS) ||
        n_shells <= 0 || n_bins < 2 || !(dlognu > 0.0) ||
        !std::isfinite(dlognu) || !nu || !r_inner || !r_outer ||
        !(time_explosion > 0.0) || !std::isfinite(time_explosion) ||
        !(T_inner > 0.0) || !std::isfinite(T_inner) ||
        !(inner_boundary_scale >= 0.0) ||
        !std::isfinite(inner_boundary_scale) || !chi_tot || !chi_es ||
        !S_fixed || !J || requested_devices <= 0 ||
        iteration_cap < 2 || !(tolerance > 0.0) ||
        !std::isfinite(tolerance)) return local.status;

    size_t cells;
    if (!checked_mul((size_t)n_shells, (size_t)n_bins, &cells) ||
        cells > std::numeric_limits<size_t>::max() / sizeof(double))
        return local.status;
    for (int b = 0; b < n_bins; ++b) {
        if (!(nu[b] > 0.0) || !std::isfinite(nu[b]) ||
            (b > 0 && !(nu[b] > nu[b - 1]))) return local.status;
    }

    int visible = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&visible);
    local.visible_devices = visible;
    if (cuda_status != cudaSuccess || visible <= 0) {
        local.status = CMF_MGPU_CUDA_UNAVAILABLE;
        if (report) *report = local;
        return local.status;
    }
    if (requested_devices > visible || requested_devices > n_shells + 16) {
        local.status = CMF_MGPU_INSUFFICIENT_DEVICES;
        if (report) *report = local;
        return local.status;
    }

    CMFMultiGPUStatus status = CMF_MGPU_INVALID_INPUT;
    std::vector<double> dt1, t1, source, source_cell, inner;
    std::vector<double> jwork, jnew, partial;
    std::vector<DeviceShard> shards;
    HostGeometry geometry;
    RayPartition partition;
    try {
        if (!build_geometry(&geometry, n_shells, n_bins, dlognu,
                            r_inner, r_outer, time_explosion)) {
            local.status = CMF_MGPU_INVALID_INPUT;
            if (report) *report = local;
            return local.status;
        }
        dt1.resize(cells);
        if (positive_sliding) {
            t1.resize(cells);
            source_cell.resize(cells);
        }
        source.resize(cells);
        inner.resize((size_t)n_bins);
        jwork.assign(J, J + cells);
        jnew.resize(cells);
        partial.resize((size_t)requested_devices * cells);
        shards.resize((size_t)requested_devices);
    } catch (const std::bad_alloc &) {
        local.status = CMF_MGPU_ALLOCATION_FAILED;
        if (report) *report = local;
        return local.status;
    }

    double max_ratio = 0.0;
    for (size_t idx = 0; idx < cells; ++idx) {
        double ct = chi_tot[idx], ce = chi_es[idx];
        double sf = S_fixed[idx], j = J[idx];
        if (!(ct >= 0.0) || !(ce >= 0.0) ||
            (ct == 0.0 && ce != 0.0) || !(sf >= 0.0) || !(j >= 0.0) ||
            !std::isfinite(ct) || !std::isfinite(ce) ||
            !std::isfinite(sf) || !std::isfinite(j)) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
        double ratio = ct > 0.0 ? ce / ct : 0.0;
        if (ratio > max_ratio) max_ratio = ratio;
        double depth = (ct + geometry.a_lam * 4.0) / geometry.a_drift;
        if (!(depth >= 0.0) || !std::isfinite(depth)) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
        dt1[idx] = depth;
        if (positive_sliding) {
            t1[idx] = std::exp(-depth);
            if (!(t1[idx] >= 0.0) || !(t1[idx] <= 1.0) ||
                !std::isfinite(t1[idx])) {
                local.status = CMF_MGPU_NONFINITE;
                if (report) *report = local;
                return local.status;
            }
        }
    }
    local.max_scattering_ratio = max_ratio;
    for (int b = 0; b < n_bins; ++b) {
        inner[(size_t)b] = inner_boundary_scale * planck(nu[b], T_inner);
        if (!(inner[(size_t)b] >= 0.0) || !std::isfinite(inner[(size_t)b])) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
    }

    double max_drift = 0.0;
    int max_positive_window = 0;
    for (int k = 0; k < geometry.nr; ++k) {
        double drift = 0.0;
        for (int i = 0; i < geometry.rn[(size_t)k]; ++i) {
            double beta = geometry.beta[
                (size_t)k * (size_t)geometry.stride + (size_t)i];
            drift += beta;
            if (positive_sliding) {
                if (!(beta <= (double)INT_MAX - 2.0)) {
                    local.status = CMF_MGPU_INVALID_INPUT;
                    if (report) *report = local;
                    return local.status;
                }
                int q = (int)std::floor(beta);
                double phi = beta - (double)q;
                int qtop = phi < 0.5 ? q : q + 1;
                int window = qtop >= 2 ? qtop - 1 : 0;
                if (window > max_positive_window)
                    max_positive_window = window;
            }
        }
        if (drift > max_drift) max_drift = drift;
    }
    local.max_characteristic_drift_bins = max_drift;
    local.max_positive_window_bins = (size_t)max_positive_window;

    status = CMF_MGPU_ALLOCATION_FAILED;
    if (!build_ray_partition(
            geometry, requested_devices, partition_mode, &partition))
        goto cleanup;
    record_ray_partition(&local, partition, partition_mode);
    for (int d = 0; d < requested_devices; ++d) {
        DeviceShard &s = shards[(size_t)d];
        s.device = d;
        s.ray_begin = partition.boundary[(size_t)d];
        s.ray_end = partition.boundary[(size_t)d + 1U];
        s.compute_end = std::min(geometry.nr, s.ray_end + 1);
        s.local_rays = s.compute_end - s.ray_begin;
        if (s.ray_end <= s.ray_begin || s.local_rays <= 0 ||
            cudaSetDevice(s.device) != cudaSuccess) {
            status = CMF_MGPU_CUDA_FAILURE;
            goto cleanup;
        }
        std::vector<int> compact_shell;
        std::vector<double> compact_beta;
        if (!build_compact_shard_geometry(
                geometry, &s, &compact_shell, &compact_beta)) goto cleanup;
        size_t local_slots = compact_shell.size();
        size_t segment_cells;
        if (!checked_mul(local_slots, (size_t)n_bins, &segment_cells))
            goto cleanup;
        if (device_allocate(&s.rn, (size_t)s.local_rays, &s) != cudaSuccess ||
            device_allocate(&s.segment_offset,
                            (size_t)s.local_rays + 1U, &s) != cudaSuccess ||
            device_allocate(&s.shell, local_slots, &s) != cudaSuccess ||
            device_allocate(&s.core, (size_t)s.local_rays, &s) != cudaSuccess ||
            device_allocate(&s.beta, local_slots, &s) != cudaSuccess ||
            device_allocate(&s.impact, (size_t)geometry.nr, &s) != cudaSuccess ||
            device_allocate(&s.rmid, (size_t)n_shells, &s) != cudaSuccess ||
            device_allocate(&s.dt1, cells, &s) != cudaSuccess ||
            device_allocate(&s.source, cells, &s) != cudaSuccess ||
            device_allocate(&s.inner, (size_t)n_bins, &s) != cudaSuccess ||
            device_allocate(&s.in, segment_cells, &s) != cudaSuccess ||
            device_allocate(&s.out, segment_cells, &s) != cudaSuccess ||
            device_allocate(&s.partial, cells, &s) != cudaSuccess)
            goto cleanup;
        if (positive_sliding) {
            size_t epoch_workspace_bytes = 0;
            if (device_allocate(&s.t1, cells, &s) != cudaSuccess ||
                device_allocate(&s.source_cell, cells, &s) != cudaSuccess ||
                allocate_positive_sweep_workspace(
                    &s, n_bins, max_positive_window, epoch_schedule,
                    &epoch_workspace_bytes) != cudaSuccess ||
                device_allocate(&s.failure, (size_t)kFailureWords, &s) !=
                    cudaSuccess)
                goto cleanup;
            if (epoch_schedule)
                local.epoch_workspace_bytes_per_device_max = std::max(
                    local.epoch_workspace_bytes_per_device_max,
                    epoch_workspace_bytes);
        }

        if (cudaMemcpy(s.rn, geometry.rn.data() + s.ray_begin,
                       (size_t)s.local_rays * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.segment_offset, s.host_segment_offset.data(),
                       ((size_t)s.local_rays + 1U) * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.shell, compact_shell.data(),
                       local_slots * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.core, geometry.core.data() + s.ray_begin,
                       (size_t)s.local_rays * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.beta, compact_beta.data(),
                       local_slots * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.impact, geometry.impact.data(),
                       (size_t)geometry.nr * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.rmid, geometry.rmid.data(),
                       (size_t)n_shells * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.dt1, dt1.data(), cells * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            (positive_sliding &&
             cudaMemcpy(s.t1, t1.data(), cells * sizeof(double),
                        cudaMemcpyHostToDevice) != cudaSuccess) ||
            cudaMemcpy(s.inner, inner.data(), (size_t)n_bins * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            status = CMF_MGPU_CUDA_FAILURE;
            goto cleanup;
        }
        local.owned_rays += (size_t)(s.ray_end - s.ray_begin);
        local.computed_rays_with_halos += (size_t)s.local_rays;
        local.total_device_allocated_bytes += s.allocated_bytes;
        local.max_device_allocated_bytes = std::max(
            local.max_device_allocated_bytes, s.allocated_bytes);
        if (d < CMF_MGPU_REPORT_MAX_DEVICES)
            local.device_allocated_bytes[d] = s.allocated_bytes;
    }

    local.devices_used = requested_devices;
    local.n_rays = (size_t)geometry.nr;
    local.deterministic_host_reduction = 1;
    local.initialization_seconds = monotonic_seconds() - total_start;
    status = CMF_MGPU_NOT_CONVERGED;
    for (int iteration = 0; iteration < iteration_cap; ++iteration) {
        double phase_start = monotonic_seconds();
        for (size_t idx = 0; idx < cells; ++idx) {
            double ratio = chi_tot[idx] > 0.0
                         ? chi_es[idx] / chi_tot[idx] : 0.0;
            source[idx] = S_fixed[idx] + ratio * jwork[idx];
            if (!(source[idx] >= 0.0) || !std::isfinite(source[idx])) {
                local.failure_phase = 1;
                local.failure_iteration = iteration;
                status = CMF_MGPU_NONFINITE;
                goto cleanup;
            }
            if (positive_sliding) {
                source_cell[idx] = (1.0 - t1[idx]) * source[idx];
                if (!(source_cell[idx] >= 0.0) ||
                    !std::isfinite(source_cell[idx])) {
                    local.failure_phase = 1;
                    local.failure_iteration = iteration;
                    status = CMF_MGPU_NONFINITE;
                    goto cleanup;
                }
            }
        }
        local.source_assembly_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        for (int d = 0; d < requested_devices; ++d) {
            DeviceShard &s = shards[(size_t)d];
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemcpy(s.source, source.data(), cells * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                (positive_sliding &&
                 cudaMemcpy(s.source_cell, source_cell.data(),
                            cells * sizeof(double),
                            cudaMemcpyHostToDevice) != cudaSuccess)) {
                status = CMF_MGPU_CUDA_FAILURE;
                goto cleanup;
            }
        }
        local.host_to_device_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        SweepFailureDetail sweep_detail;
        int sweep_status = launch_sweep(
            shards.data(), requested_devices, geometry, cells,
            positive_sliding, max_positive_window, 0, &sweep_detail,
            epoch_schedule);
        if (sweep_status != 0) {
            if (sweep_status != 1) {
                local.failure_phase = 2;
                local.failure_iteration = iteration;
                record_sweep_failure_detail(
                    &local, shards.data(), requested_devices, n_bins,
                    sweep_detail);
            }
            status = sweep_status == 1 ? CMF_MGPU_CUDA_FAILURE
                                       : CMF_MGPU_NONFINITE;
            goto cleanup;
        }
        local.device_sweep_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        for (int d = 0; d < requested_devices; ++d) {
            DeviceShard &s = shards[(size_t)d];
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemcpy(partial.data() + (size_t)d * cells, s.partial,
                           cells * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                status = CMF_MGPU_CUDA_FAILURE;
                goto cleanup;
            }
        }
        local.device_to_host_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        for (size_t idx = 0; idx < cells; ++idx) {
            double sum = 0.0;
            for (int d = requested_devices - 1; d >= 0; --d)
                sum += partial[(size_t)d * cells + idx];
            if (!(sum >= 0.0) || !std::isfinite(sum)) {
                local.failure_phase = 4;
                local.failure_iteration = iteration;
                local.failure_cell_index = idx;
                local.failure_nearest = sum;
                status = CMF_MGPU_NONFINITE;
                goto cleanup;
            }
            jnew[idx] = sum;
        }
        local.host_reduction_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        double max_relative = 0.0;
        double max_absolute = 0.0;
        for (size_t idx = 0; idx < cells; ++idx) {
            double absolute = std::fabs(jnew[idx] - jwork[idx]);
            double scale = std::fabs(jwork[idx]);
            double relative = scale > 0.0 ? absolute / scale
                : (absolute == 0.0 ? 0.0 : INFINITY);
            if (relative > max_relative) max_relative = relative;
            if (absolute > max_absolute) max_absolute = absolute;
        }
        jwork.swap(jnew);
        local.iterations_used = iteration + 1;
        local.final_max_relative_change = max_relative;
        local.final_max_absolute_change = max_absolute;
        if (max_ratio < 1.0) {
            double factor = max_ratio == 0.0 ? 0.0
                : max_ratio / (1.0 - max_ratio);
            local.fixed_point_absolute_error_bound = factor * max_absolute;
        } else {
            local.fixed_point_absolute_error_bound = INFINITY;
        }
        local.convergence_check_seconds += monotonic_seconds() - phase_start;
        if (iteration > 0 && max_relative < tolerance) {
            status = CMF_MGPU_OK;
            break;
        }
    }

cleanup:
    {
        double cleanup_start = monotonic_seconds();
        for (DeviceShard &s : shards) cleanup_shard(&s);
        local.cleanup_seconds += monotonic_seconds() - cleanup_start;
    }
    if (status == CMF_MGPU_OK)
        std::memcpy(J, jwork.data(), cells * sizeof(double));
    local.total_seconds = monotonic_seconds() - total_start;
    local.fixed_point_solve_seconds = local.total_seconds;
    local.status = status;
    if (report) *report = local;
    return status;
}

extern "C" CMFMultiGPUStatus cmf_exact_multigpu_direct_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report)
{
    return solve_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        requested_devices, iteration_cap, tolerance, report, false,
        CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, nullptr);
}

extern "C" CMFMultiGPUStatus cmf_exact_multigpu_positive_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report)
{
    return solve_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        requested_devices, iteration_cap, tolerance, report, true,
        CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, nullptr);
}

extern "C" CMFMultiGPUStatus cmf_exact_multigpu_positive_solve_epoch(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J,
    const CMFMultiGPUEpochSchedule *schedule,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report)
{
    return solve_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        requested_devices, iteration_cap, tolerance, report, true,
        CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, schedule);
}

static CMFMultiGPUStatus apply_positive_bounds_impl(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper,
    int requested_devices, CMFMultiGPUReport *report,
    const CMFMultiGPUEpochSchedule *epoch_schedule)
{
    CMFMultiGPUReport local{};
    local.status = CMF_MGPU_INVALID_INPUT;
    local.iteration_cap = 1;
    local.final_max_relative_change = INFINITY;
    local.final_max_absolute_change = INFINITY;
    local.max_scattering_ratio = INFINITY;
    local.fixed_point_absolute_error_bound = INFINITY;
    local.positive_sliding = 1;
    local.epoch_frequency_parallel = epoch_schedule ? 1 : 0;
    if (epoch_schedule) {
        local.epoch_block_size = epoch_schedule->block_size;
        local.epoch_batch_cardinality =
            epoch_schedule->epoch_batch_cardinality;
        local.epoch_direct_replay_max_window =
            epoch_schedule->direct_replay_max_window;
    }
    local.failure_iteration = -1;
    local.failure_cell_index = std::numeric_limits<size_t>::max();
    local.failure_device_index = -1;
    local.failure_ray_begin = -1;
    local.failure_ray_end = -1;
    local.failure_segment_index = -1;
    local.failure_local_ray_index = -1;
    local.failure_bin_index = -1;
    local.failure_outward = -1;
    local.failure_global_ray_index = -1;
    if (report) *report = local;
    if ((epoch_schedule && !valid_epoch_schedule(epoch_schedule)) ||
        n_shells <= 0 || n_bins < 2 || !(dlognu > 0.0) ||
        !std::isfinite(dlognu) || !nu || !r_inner || !r_outer ||
        !(time_explosion > 0.0) || !std::isfinite(time_explosion) ||
        !(T_inner > 0.0) || !std::isfinite(T_inner) ||
        !(inner_boundary_scale >= 0.0) ||
        !std::isfinite(inner_boundary_scale) || !chi_tot || !chi_es ||
        !S_fixed || !input_J || !lower || !nearest || !upper ||
        lower == nearest || lower == upper || nearest == upper ||
        requested_devices <= 0) return local.status;

    size_t cells;
    size_t partial_cells;
    if (!checked_mul((size_t)n_shells, (size_t)n_bins, &cells) ||
        !checked_mul((size_t)requested_devices, cells, &partial_cells) ||
        cells > std::numeric_limits<size_t>::max() / sizeof(double))
        return local.status;
    for (int b = 0; b < n_bins; ++b) {
        if (!(nu[b] > 0.0) || !std::isfinite(nu[b]) ||
            (b > 0 && !(nu[b] > nu[b - 1]))) return local.status;
    }

    int visible = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&visible);
    local.visible_devices = visible;
    if (cuda_status != cudaSuccess || visible <= 0) {
        local.status = CMF_MGPU_CUDA_UNAVAILABLE;
        if (report) *report = local;
        return local.status;
    }
    if (requested_devices > visible || requested_devices > n_shells + 16) {
        local.status = CMF_MGPU_INSUFFICIENT_DEVICES;
        if (report) *report = local;
        return local.status;
    }

    CMFMultiGPUStatus status = CMF_MGPU_INVALID_INPUT;
    HostGeometry geometry;
    RayPartition partition;
    std::vector<double> dt1, t1, source, source_cell, inner, partial;
    std::vector<double> lower_work, nearest_work, upper_work;
    std::vector<double> previous_partial;
    std::vector<DeviceShard> shards;
    try {
        if (!build_geometry(&geometry, n_shells, n_bins, dlognu,
                            r_inner, r_outer, time_explosion)) {
            local.status = CMF_MGPU_INVALID_INPUT;
            if (report) *report = local;
            return local.status;
        }
        dt1.resize(cells);
        t1.resize(cells);
        source.resize(cells);
        source_cell.resize(cells);
        inner.resize((size_t)n_bins);
        partial.resize(partial_cells);
        lower_work.resize(cells);
        nearest_work.resize(cells);
        upper_work.resize(cells);
        previous_partial.resize(partial_cells);
        shards.resize((size_t)requested_devices);
    } catch (const std::bad_alloc &) {
        local.status = CMF_MGPU_ALLOCATION_FAILED;
        if (report) *report = local;
        return local.status;
    }

    for (size_t idx = 0; idx < cells; ++idx) {
        double ct = chi_tot[idx];
        double ce = chi_es[idx];
        double sf = S_fixed[idx];
        double j = input_J[idx];
        if (!(ct >= 0.0) || !(ce >= 0.0) ||
            (ct == 0.0 && ce != 0.0) || !(sf >= 0.0) || !(j >= 0.0) ||
            !std::isfinite(ct) || !std::isfinite(ce) ||
            !std::isfinite(sf) || !std::isfinite(j)) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
        double depth = (ct + geometry.a_lam * 4.0) / geometry.a_drift;
        if (!(depth >= 0.0) || !std::isfinite(depth)) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
        dt1[idx] = depth;
        t1[idx] = std::exp(-depth);
        if (!(t1[idx] >= 0.0) || !(t1[idx] <= 1.0) ||
            !std::isfinite(t1[idx])) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
    }
    for (int b = 0; b < n_bins; ++b) {
        inner[(size_t)b] = inner_boundary_scale * planck(nu[b], T_inner);
        if (!(inner[(size_t)b] >= 0.0) ||
            !std::isfinite(inner[(size_t)b])) {
            local.status = CMF_MGPU_NONFINITE;
            if (report) *report = local;
            return local.status;
        }
    }

    double max_drift = 0.0;
    int max_positive_window = 0;
    for (int k = 0; k < geometry.nr; ++k) {
        double drift = 0.0;
        for (int i = 0; i < geometry.rn[(size_t)k]; ++i) {
            double beta = geometry.beta[
                (size_t)k * (size_t)geometry.stride + (size_t)i];
            if (!(beta <= (double)INT_MAX - 2.0)) {
                local.status = CMF_MGPU_INVALID_INPUT;
                if (report) *report = local;
                return local.status;
            }
            drift += beta;
            int q = (int)std::floor(beta);
            double phi = beta - (double)q;
            int qtop = phi < 0.5 ? q : q + 1;
            int window = qtop >= 2 ? qtop - 1 : 0;
            if (window > max_positive_window)
                max_positive_window = window;
        }
        if (drift > max_drift) max_drift = drift;
    }
    local.max_characteristic_drift_bins = max_drift;
    local.max_positive_window_bins = (size_t)max_positive_window;

    status = CMF_MGPU_ALLOCATION_FAILED;
    if (!build_ray_partition(
            geometry, requested_devices,
            CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, &partition))
        goto bounds_cleanup;
    record_ray_partition(
        &local, partition, CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS);
    for (int d = 0; d < requested_devices; ++d) {
        DeviceShard &s = shards[(size_t)d];
        s.device = d;
        s.ray_begin = partition.boundary[(size_t)d];
        s.ray_end = partition.boundary[(size_t)d + 1U];
        s.compute_end = std::min(geometry.nr, s.ray_end + 1);
        s.local_rays = s.compute_end - s.ray_begin;
        if (s.ray_end <= s.ray_begin || s.local_rays <= 0 ||
            cudaSetDevice(s.device) != cudaSuccess) {
            status = CMF_MGPU_CUDA_FAILURE;
            goto bounds_cleanup;
        }
        std::vector<int> compact_shell;
        std::vector<double> compact_beta;
        if (!build_compact_shard_geometry(
                geometry, &s, &compact_shell, &compact_beta))
            goto bounds_cleanup;
        size_t local_slots = compact_shell.size();
        size_t segment_cells;
        if (!checked_mul(local_slots, (size_t)n_bins, &segment_cells))
            goto bounds_cleanup;
        if (device_allocate(&s.rn, (size_t)s.local_rays, &s) != cudaSuccess ||
            device_allocate(&s.segment_offset,
                            (size_t)s.local_rays + 1U, &s) != cudaSuccess ||
            device_allocate(&s.shell, local_slots, &s) != cudaSuccess ||
            device_allocate(&s.core, (size_t)s.local_rays, &s) != cudaSuccess ||
            device_allocate(&s.beta, local_slots, &s) != cudaSuccess ||
            device_allocate(&s.impact, (size_t)geometry.nr, &s) != cudaSuccess ||
            device_allocate(&s.rmid, (size_t)n_shells, &s) != cudaSuccess ||
            device_allocate(&s.dt1, cells, &s) != cudaSuccess ||
            device_allocate(&s.t1, cells, &s) != cudaSuccess ||
            device_allocate(&s.source, cells, &s) != cudaSuccess ||
            device_allocate(&s.source_cell, cells, &s) != cudaSuccess ||
            device_allocate(&s.inner, (size_t)n_bins, &s) != cudaSuccess ||
            device_allocate(&s.in, segment_cells, &s) != cudaSuccess ||
            device_allocate(&s.out, segment_cells, &s) != cudaSuccess ||
            device_allocate(&s.partial, cells, &s) != cudaSuccess ||
            device_allocate(&s.failure, (size_t)kFailureWords, &s) !=
                cudaSuccess)
            goto bounds_cleanup;
        size_t epoch_workspace_bytes = 0;
        if (allocate_positive_sweep_workspace(
                &s, n_bins, max_positive_window, epoch_schedule,
                &epoch_workspace_bytes) != cudaSuccess)
            goto bounds_cleanup;
        if (epoch_schedule)
            local.epoch_workspace_bytes_per_device_max = std::max(
                local.epoch_workspace_bytes_per_device_max,
                epoch_workspace_bytes);

        if (cudaMemcpy(s.rn, geometry.rn.data() + s.ray_begin,
                       (size_t)s.local_rays * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.segment_offset, s.host_segment_offset.data(),
                       ((size_t)s.local_rays + 1U) * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.shell, compact_shell.data(),
                       local_slots * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.core, geometry.core.data() + s.ray_begin,
                       (size_t)s.local_rays * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.beta, compact_beta.data(),
                       local_slots * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.impact, geometry.impact.data(),
                       (size_t)geometry.nr * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.rmid, geometry.rmid.data(),
                       (size_t)n_shells * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.dt1, dt1.data(), cells * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.t1, t1.data(), cells * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s.inner, inner.data(), (size_t)n_bins * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            status = CMF_MGPU_CUDA_FAILURE;
            goto bounds_cleanup;
        }
        local.owned_rays += (size_t)(s.ray_end - s.ray_begin);
        local.computed_rays_with_halos += (size_t)s.local_rays;
        local.total_device_allocated_bytes += s.allocated_bytes;
        local.max_device_allocated_bytes = std::max(
            local.max_device_allocated_bytes, s.allocated_bytes);
    }

    local.devices_used = requested_devices;
    local.n_rays = (size_t)geometry.nr;
    local.deterministic_host_reduction = 1;
    for (int rounding = -1; rounding <= 1; ++rounding) {
        for (size_t idx = 0; idx < cells; ++idx) {
            double ratio = chi_tot[idx] > 0.0
                         ? chi_es[idx] / chi_tot[idx] : 0.0;
            if (rounding == 0) {
                source[idx] = S_fixed[idx] + ratio * input_J[idx];
                source_cell[idx] = (1.0 - t1[idx]) * source[idx];
            } else {
                double scattering;
                if (!host_positive_multiply_bound(
                        ratio, input_J[idx], rounding, &scattering) ||
                    !host_positive_add_bound(
                        S_fixed[idx], scattering, rounding, &source[idx]) ||
                    !host_positive_multiply_bound(
                        1.0 - t1[idx], source[idx], rounding,
                        &source_cell[idx])) {
                    local.failure_phase = 1;
                    local.failure_iteration = rounding;
                    status = CMF_MGPU_NONFINITE;
                    goto bounds_cleanup;
                }
            }
            if (!(source[idx] >= 0.0) || !(source_cell[idx] >= 0.0) ||
                !std::isfinite(source[idx]) ||
                !std::isfinite(source_cell[idx])) {
                local.failure_phase = 1;
                local.failure_iteration = rounding;
                status = CMF_MGPU_NONFINITE;
                goto bounds_cleanup;
            }
        }
        for (int d = 0; d < requested_devices; ++d) {
            DeviceShard &s = shards[(size_t)d];
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemcpy(s.source, source.data(), cells * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(s.source_cell, source_cell.data(),
                           cells * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                status = CMF_MGPU_CUDA_FAILURE;
                goto bounds_cleanup;
            }
        }
        SweepFailureDetail sweep_detail;
        int sweep_status = launch_sweep(
            shards.data(), requested_devices, geometry, cells,
            true, max_positive_window, rounding, &sweep_detail,
            epoch_schedule);
        if (sweep_status != 0) {
            if (sweep_status != 1) {
                local.failure_phase = 2;
                local.failure_iteration = rounding;
                record_sweep_failure_detail(
                    &local, shards.data(), requested_devices, n_bins,
                    sweep_detail);
            }
            status = sweep_status == 1 ? CMF_MGPU_CUDA_FAILURE
                                       : CMF_MGPU_NONFINITE;
            goto bounds_cleanup;
        }
        for (int d = 0; d < requested_devices; ++d) {
            DeviceShard &s = shards[(size_t)d];
            if (cudaSetDevice(s.device) != cudaSuccess ||
                cudaMemcpy(partial.data() + (size_t)d * cells, s.partial,
                           cells * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                status = CMF_MGPU_CUDA_FAILURE;
                goto bounds_cleanup;
            }
        }
        if (rounding >= 0) {
            for (int d = 0; d < requested_devices; ++d) {
                const DeviceShard &s = shards[(size_t)d];
                for (size_t idx = 0; idx < cells; ++idx) {
                    double prior = previous_partial[
                        (size_t)d * cells + idx];
                    double current = partial[(size_t)d * cells + idx];
                    bool ordered = current >= prior;
                    if (!ordered || !std::isfinite(current)) {
                        local.failure_phase = 5;
                        local.failure_iteration = rounding;
                        local.failure_cell_index = idx;
                        local.failure_device_index = d;
                        local.failure_ray_begin = s.ray_begin;
                        local.failure_ray_end = s.ray_end;
                        if (rounding == 0) {
                            local.failure_lower = prior;
                            local.failure_nearest = current;
                        } else {
                            local.failure_nearest = prior;
                            local.failure_upper = current;
                        }
                        int shell = (int)(idx / (size_t)n_bins);
                        int bin = (int)(idx % (size_t)n_bins);
                        int segment = n_shells - 1 - shell;
                        double host_sum = 0.0;
                        double device_geometry_sum = 0.0;
                        double max_intensity = -1.0;
                        if (cudaSetDevice(s.device) == cudaSuccess) {
                            double device_rmid = NAN;
                            (void)cudaMemcpy(
                                &device_rmid, s.rmid + shell,
                                sizeof(double), cudaMemcpyDeviceToHost);
                            local.failure_device_rmid = device_rmid;
                            for (int ray = s.ray_end - 1;
                                 ray >= s.ray_begin; --ray) {
                                int lr = ray - s.ray_begin;
                                if (segment >= geometry.rn[(size_t)ray])
                                    continue;
                                ++local.failure_active_ray_count;
                                size_t at =
                                    ((size_t)s.host_segment_offset[
                                         (size_t)lr] +
                                     (size_t)segment) * (size_t)n_bins +
                                    (size_t)bin;
                                double ray_in = NAN;
                                double ray_out = NAN;
                                if (cudaMemcpy(&ray_in, s.in + at,
                                               sizeof(double),
                                               cudaMemcpyDeviceToHost) !=
                                        cudaSuccess ||
                                    cudaMemcpy(&ray_out, s.out + at,
                                               sizeof(double),
                                               cudaMemcpyDeviceToHost) !=
                                        cudaSuccess)
                                    break;
                                double intensity = 0.5 * (ray_in + ray_out);
                                if (intensity > 0.0)
                                    ++local.failure_positive_intensity_count;
                                if (intensity > max_intensity) {
                                    max_intensity = intensity;
                                    local.failure_global_ray_index = ray;
                                    local.failure_ray_in = ray_in;
                                    local.failure_ray_out = ray_out;
                                }
                                double current_mu_squared =
                                    1.0 - geometry.impact[(size_t)ray] *
                                          geometry.impact[(size_t)ray] /
                                          (geometry.rmid[(size_t)shell] *
                                           geometry.rmid[(size_t)shell]);
                                double current_mu =
                                    current_mu_squared >= 0.0 &&
                                    std::isfinite(current_mu_squared)
                                        ? std::sqrt(current_mu_squared)
                                        : NAN;
                                double device_impact = NAN;
                                (void)cudaMemcpy(
                                    &device_impact, s.impact + ray,
                                    sizeof(double), cudaMemcpyDeviceToHost);
                                local.failure_max_impact_absolute_difference =
                                    std::max(
                                        local.failure_max_impact_absolute_difference,
                                        std::fabs(
                                            device_impact -
                                            geometry.impact[(size_t)ray]));
                                double current_mu_device_squared =
                                    1.0 - device_impact * device_impact /
                                          (device_rmid * device_rmid);
                                double current_mu_device =
                                    current_mu_device_squared >= 0.0 &&
                                    std::isfinite(current_mu_device_squared)
                                        ? std::sqrt(
                                              current_mu_device_squared)
                                        : NAN;
                                double previous_intensity = intensity;
                                double previous_mu = 0.0;
                                double previous_mu_device = 0.0;
                                int previous_ray = ray + 1;
                                if (previous_ray < geometry.nr &&
                                    segment < geometry.rn[
                                        (size_t)previous_ray]) {
                                    int previous_local =
                                        previous_ray - s.ray_begin;
                                    size_t previous_at =
                                        ((size_t)s.host_segment_offset[
                                             (size_t)previous_local] +
                                         (size_t)segment) *
                                        (size_t)n_bins + (size_t)bin;
                                    double previous_in = NAN;
                                    double previous_out = NAN;
                                    if (cudaMemcpy(
                                            &previous_in,
                                            s.in + previous_at,
                                            sizeof(double),
                                            cudaMemcpyDeviceToHost) !=
                                            cudaSuccess ||
                                        cudaMemcpy(
                                            &previous_out,
                                            s.out + previous_at,
                                            sizeof(double),
                                            cudaMemcpyDeviceToHost) !=
                                            cudaSuccess)
                                        break;
                                    previous_intensity =
                                        0.5 * (previous_in + previous_out);
                                    double previous_mu_squared =
                                        1.0 - geometry.impact[
                                                  (size_t)previous_ray] *
                                              geometry.impact[
                                                  (size_t)previous_ray] /
                                              (geometry.rmid[(size_t)shell] *
                                               geometry.rmid[(size_t)shell]);
                                    previous_mu =
                                        previous_mu_squared >= 0.0 &&
                                        std::isfinite(previous_mu_squared)
                                            ? std::sqrt(previous_mu_squared)
                                            : NAN;
                                    double previous_device_impact = NAN;
                                    (void)cudaMemcpy(
                                        &previous_device_impact,
                                        s.impact + previous_ray,
                                        sizeof(double),
                                        cudaMemcpyDeviceToHost);
                                    local.failure_max_impact_absolute_difference =
                                        std::max(
                                            local.failure_max_impact_absolute_difference,
                                            std::fabs(
                                                previous_device_impact -
                                                geometry.impact[
                                                    (size_t)previous_ray]));
                                    double previous_mu_device_squared =
                                        1.0 - previous_device_impact *
                                                  previous_device_impact /
                                              (device_rmid * device_rmid);
                                    previous_mu_device =
                                        previous_mu_device_squared >= 0.0 &&
                                        std::isfinite(
                                            previous_mu_device_squared)
                                            ? std::sqrt(
                                                  previous_mu_device_squared)
                                            : NAN;
                                }
                                host_sum +=
                                    0.5 * (previous_intensity + intensity) *
                                    (current_mu - previous_mu);
                                device_geometry_sum +=
                                    0.5 * (previous_intensity + intensity) *
                                    (current_mu_device - previous_mu_device);
                            }
                        }
                        local.failure_host_recomputed_partial = host_sum;
                        local.failure_device_geometry_partial =
                            device_geometry_sum;
                        status = CMF_MGPU_NONFINITE;
                        goto bounds_cleanup;
                    }
                }
            }
        }
        previous_partial = partial;
        std::vector<double> *destination = rounding < 0 ? &lower_work
            : (rounding > 0 ? &upper_work : &nearest_work);
        for (size_t idx = 0; idx < cells; ++idx) {
            double sum = 0.0;
            for (int d = requested_devices - 1; d >= 0; --d) {
                double value = partial[(size_t)d * cells + idx];
                if (rounding == 0) sum += value;
                else if (!host_positive_add_bound(
                             sum, value, rounding, &sum)) {
                    local.failure_phase = 4;
                    local.failure_iteration = rounding;
                    local.failure_cell_index = idx;
                    local.failure_nearest = value;
                    status = CMF_MGPU_NONFINITE;
                    goto bounds_cleanup;
                }
            }
            if (!(sum >= 0.0) || !std::isfinite(sum)) {
                local.failure_phase = 4;
                local.failure_iteration = rounding;
                local.failure_cell_index = idx;
                local.failure_nearest = sum;
                status = CMF_MGPU_NONFINITE;
                goto bounds_cleanup;
            }
            (*destination)[idx] = sum;
        }
    }

    for (size_t idx = 0; idx < cells; ++idx) {
        if (!(nearest_work[idx] >= lower_work[idx]) ||
            !(upper_work[idx] >= nearest_work[idx]) ||
            !std::isfinite(upper_work[idx])) {
            local.failure_phase = 6;
            local.failure_cell_index = idx;
            local.failure_lower = lower_work[idx];
            local.failure_nearest = nearest_work[idx];
            local.failure_upper = upper_work[idx];
            status = CMF_MGPU_NONFINITE;
            goto bounds_cleanup;
        }
    }
    status = CMF_MGPU_OK;

bounds_cleanup:
    for (DeviceShard &s : shards) cleanup_shard(&s);
    if (status == CMF_MGPU_OK) {
        std::memcpy(lower, lower_work.data(), cells * sizeof(double));
        std::memcpy(nearest, nearest_work.data(), cells * sizeof(double));
        std::memcpy(upper, upper_work.data(), cells * sizeof(double));
        local.iterations_used = 1;
        local.final_max_relative_change = 0.0;
        local.final_max_absolute_change = 0.0;
    }
    local.status = status;
    if (report) *report = local;
    return status;
}

extern "C" CMFMultiGPUStatus cmf_exact_multigpu_apply_positive_bounds(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper,
    int requested_devices, CMFMultiGPUReport *report)
{
    return apply_positive_bounds_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, input_J,
        lower, nearest, upper, requested_devices, report, nullptr);
}

extern "C" CMFMultiGPUStatus
cmf_exact_multigpu_apply_positive_bounds_epoch(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, const double *input_J,
    double *lower, double *nearest, double *upper,
    const CMFMultiGPUEpochSchedule *schedule,
    int requested_devices, CMFMultiGPUReport *report)
{
    return apply_positive_bounds_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, input_J,
        lower, nearest, upper, requested_devices, report, schedule);
}

namespace {

struct PersistentBoundContext {
    int n_shells = 0;
    int n_bins = 0;
    int requested_devices = 0;
    int max_positive_window = 0;
    size_t cells = 0;
    const double *chi_tot = nullptr;
    const double *chi_es = nullptr;
    HostGeometry geometry;
    std::vector<double> dt1;
    std::vector<double> t1;
    std::vector<double> source;
    std::vector<double> source_cell;
    std::vector<double> inner_physical;
    std::vector<double> inner_zero;
    std::vector<double> partial;
    std::vector<DeviceShard> shards;
    RayPartition partition;
    CMFMultiGPUReport allocation_report{};
    size_t bound_applications = 0;
    size_t upper_operator_applications = 0;
    CMFMultiGPUEpochSchedule epoch_schedule{};
    bool epoch_enabled = false;
    bool ready = false;
    double initialization_seconds = 0.0;
    double source_assembly_seconds = 0.0;
    double host_to_device_seconds = 0.0;
    double device_sweep_seconds = 0.0;
    double device_to_host_seconds = 0.0;
    double host_reduction_seconds = 0.0;
    double cleanup_seconds = 0.0;

    ~PersistentBoundContext()
    {
        release();
    }

    void release()
    {
        if (shards.empty()) return;
        double cleanup_start = monotonic_seconds();
        for (DeviceShard &shard : shards) cleanup_shard(&shard);
        shards.clear();
        ready = false;
        cleanup_seconds += monotonic_seconds() - cleanup_start;
    }

    CMFMultiGPUStatus initialize(
        int ns, int nb, double dlognu, const double *nu,
        const double *r_inner, const double *r_outer,
        double time_explosion, double T_inner, double inner_boundary_scale,
        const double *ct, const double *ce, int devices,
        CMFMultiGPUPartitionMode partition_mode,
        const CMFMultiGPUEpochSchedule *requested_epoch_schedule)
    {
        const double initialize_start = monotonic_seconds();
        allocation_report = CMFMultiGPUReport{};
        allocation_report.status = CMF_MGPU_INVALID_INPUT;
        allocation_report.positive_sliding = 1;
        allocation_report.epoch_frequency_parallel =
            requested_epoch_schedule ? 1 : 0;
        if (requested_epoch_schedule) {
            allocation_report.epoch_block_size =
                requested_epoch_schedule->block_size;
            allocation_report.epoch_batch_cardinality =
                requested_epoch_schedule->epoch_batch_cardinality;
            allocation_report.epoch_direct_replay_max_window =
                requested_epoch_schedule->direct_replay_max_window;
        }
        allocation_report.failure_iteration = -1;
        allocation_report.failure_cell_index =
            std::numeric_limits<size_t>::max();
        allocation_report.failure_device_index = -1;
        allocation_report.failure_ray_begin = -1;
        allocation_report.failure_ray_end = -1;
        allocation_report.failure_segment_index = -1;
        allocation_report.failure_local_ray_index = -1;
        allocation_report.failure_bin_index = -1;
        allocation_report.failure_outward = -1;
        allocation_report.failure_global_ray_index = -1;
        if ((requested_epoch_schedule &&
             !valid_epoch_schedule(requested_epoch_schedule)) ||
            (partition_mode != CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS &&
             partition_mode != CMF_MGPU_PARTITION_EQUAL_RAYS) ||
            ns <= 0 || nb < 2 || !(dlognu > 0.0) ||
            !std::isfinite(dlognu) || !nu || !r_inner || !r_outer ||
            !(time_explosion > 0.0) || !std::isfinite(time_explosion) ||
            !(T_inner > 0.0) || !std::isfinite(T_inner) ||
            !(inner_boundary_scale >= 0.0) ||
            !std::isfinite(inner_boundary_scale) || !ct || !ce ||
            devices <= 0) return allocation_report.status;
        for (int b = 0; b < nb; ++b) {
            if (!(nu[b] > 0.0) || !std::isfinite(nu[b]) ||
                (b > 0 && !(nu[b] > nu[b - 1])))
                return allocation_report.status;
        }
        int visible = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&visible);
        allocation_report.visible_devices = visible;
        if (cuda_status != cudaSuccess || visible <= 0) {
            allocation_report.status = CMF_MGPU_CUDA_UNAVAILABLE;
            return allocation_report.status;
        }
        if (devices > visible || devices > ns + 16) {
            allocation_report.status = CMF_MGPU_INSUFFICIENT_DEVICES;
            return allocation_report.status;
        }
        size_t cell_count;
        size_t partial_cells;
        if (!checked_mul((size_t)ns, (size_t)nb, &cell_count) ||
            !checked_mul((size_t)devices, cell_count, &partial_cells) ||
            cell_count > std::numeric_limits<size_t>::max() / sizeof(double))
            return allocation_report.status;
        try {
            if (!build_geometry(&geometry, ns, nb, dlognu,
                                r_inner, r_outer, time_explosion))
                return allocation_report.status;
            dt1.resize(cell_count);
            t1.resize(cell_count);
            source.resize(cell_count);
            source_cell.resize(cell_count);
            inner_physical.resize((size_t)nb);
            inner_zero.assign((size_t)nb, 0.0);
            partial.resize(partial_cells);
            shards.resize((size_t)devices);
        } catch (const std::bad_alloc &) {
            allocation_report.status = CMF_MGPU_ALLOCATION_FAILED;
            return allocation_report.status;
        }
        for (size_t idx = 0; idx < cell_count; ++idx) {
            if (!(ct[idx] >= 0.0) || !(ce[idx] >= 0.0) ||
                (ct[idx] == 0.0 && ce[idx] != 0.0) ||
                !std::isfinite(ct[idx]) || !std::isfinite(ce[idx])) {
                allocation_report.status = CMF_MGPU_NONFINITE;
                return allocation_report.status;
            }
            double depth = (ct[idx] + geometry.a_lam * 4.0) /
                           geometry.a_drift;
            if (!(depth >= 0.0) || !std::isfinite(depth)) {
                allocation_report.status = CMF_MGPU_NONFINITE;
                return allocation_report.status;
            }
            dt1[idx] = depth;
            t1[idx] = std::exp(-depth);
            if (!(t1[idx] >= 0.0) || !(t1[idx] <= 1.0) ||
                !std::isfinite(t1[idx])) {
                allocation_report.status = CMF_MGPU_NONFINITE;
                return allocation_report.status;
            }
        }
        for (int b = 0; b < nb; ++b) {
            inner_physical[(size_t)b] =
                inner_boundary_scale * planck(nu[b], T_inner);
            if (!(inner_physical[(size_t)b] >= 0.0) ||
                !std::isfinite(inner_physical[(size_t)b])) {
                allocation_report.status = CMF_MGPU_NONFINITE;
                return allocation_report.status;
            }
        }
        double max_drift = 0.0;
        for (int k = 0; k < geometry.nr; ++k) {
            double drift = 0.0;
            for (int i = 0; i < geometry.rn[(size_t)k]; ++i) {
                double beta = geometry.beta[
                    (size_t)k * (size_t)geometry.stride + (size_t)i];
                if (!(beta <= (double)INT_MAX - 2.0))
                    return allocation_report.status;
                drift += beta;
                int q = (int)std::floor(beta);
                double phi = beta - (double)q;
                int qtop = phi < 0.5 ? q : q + 1;
                int window = qtop >= 2 ? qtop - 1 : 0;
                if (window > max_positive_window)
                    max_positive_window = window;
            }
            if (drift > max_drift) max_drift = drift;
        }
        n_shells = ns;
        n_bins = nb;
        requested_devices = devices;
        cells = cell_count;
        chi_tot = ct;
        chi_es = ce;
        epoch_enabled = requested_epoch_schedule != nullptr;
        if (requested_epoch_schedule)
            epoch_schedule = *requested_epoch_schedule;
        allocation_report.max_characteristic_drift_bins = max_drift;
        allocation_report.max_positive_window_bins =
            (size_t)max_positive_window;

        if (!build_ray_partition(
                geometry, devices, partition_mode, &partition)) {
            allocation_report.status = CMF_MGPU_ALLOCATION_FAILED;
            return allocation_report.status;
        }
        record_ray_partition(&allocation_report, partition, partition_mode);

        for (int d = 0; d < devices; ++d) {
            DeviceShard &shard = shards[(size_t)d];
            shard.device = d;
            shard.ray_begin = partition.boundary[(size_t)d];
            shard.ray_end = partition.boundary[(size_t)d + 1U];
            shard.compute_end = std::min(geometry.nr, shard.ray_end + 1);
            shard.local_rays = shard.compute_end - shard.ray_begin;
            if (shard.ray_end <= shard.ray_begin || shard.local_rays <= 0 ||
                cudaSetDevice(shard.device) != cudaSuccess) {
                allocation_report.status = CMF_MGPU_CUDA_FAILURE;
                return allocation_report.status;
            }
            std::vector<int> compact_shell;
            std::vector<double> compact_beta;
            if (!build_compact_shard_geometry(
                    geometry, &shard, &compact_shell, &compact_beta)) {
                allocation_report.status = CMF_MGPU_ALLOCATION_FAILED;
                return allocation_report.status;
            }
            size_t local_slots = compact_shell.size();
            size_t segment_cells;
            if (!checked_mul(local_slots, (size_t)nb, &segment_cells)) {
                allocation_report.status = CMF_MGPU_ALLOCATION_FAILED;
                return allocation_report.status;
            }
            if (device_allocate(&shard.rn, (size_t)shard.local_rays,
                                &shard) != cudaSuccess ||
                device_allocate(&shard.segment_offset,
                                (size_t)shard.local_rays + 1U,
                                &shard) != cudaSuccess ||
                device_allocate(&shard.shell, local_slots, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.core, (size_t)shard.local_rays,
                                &shard) != cudaSuccess ||
                device_allocate(&shard.beta, local_slots, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.impact, (size_t)geometry.nr,
                                &shard) != cudaSuccess ||
                device_allocate(&shard.rmid, (size_t)ns, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.dt1, cell_count, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.t1, cell_count, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.source, cell_count, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.source_cell, cell_count, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.inner, (size_t)nb, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.in, segment_cells, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.out, segment_cells, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.partial, cell_count, &shard) !=
                    cudaSuccess ||
                device_allocate(&shard.failure, (size_t)kFailureWords,
                                &shard) != cudaSuccess) {
                allocation_report.status = CMF_MGPU_ALLOCATION_FAILED;
                return allocation_report.status;
            }
            size_t epoch_workspace_bytes = 0;
            if (allocate_positive_sweep_workspace(
                    &shard, nb, max_positive_window,
                    epoch_enabled ? &epoch_schedule : nullptr,
                    &epoch_workspace_bytes) != cudaSuccess) {
                allocation_report.status = CMF_MGPU_ALLOCATION_FAILED;
                return allocation_report.status;
            }
            if (epoch_enabled)
                allocation_report.epoch_workspace_bytes_per_device_max =
                    std::max(
                        allocation_report.
                            epoch_workspace_bytes_per_device_max,
                        epoch_workspace_bytes);
            if (cudaMemcpy(shard.rn,
                           geometry.rn.data() + shard.ray_begin,
                           (size_t)shard.local_rays * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.segment_offset,
                           shard.host_segment_offset.data(),
                           ((size_t)shard.local_rays + 1U) * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.shell,
                           compact_shell.data(),
                           local_slots * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.core,
                           geometry.core.data() + shard.ray_begin,
                           (size_t)shard.local_rays * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.beta,
                           compact_beta.data(),
                           local_slots * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.impact, geometry.impact.data(),
                           (size_t)geometry.nr * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.rmid, geometry.rmid.data(),
                           (size_t)ns * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.dt1, dt1.data(),
                           cell_count * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.t1, t1.data(),
                           cell_count * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                allocation_report.status = CMF_MGPU_CUDA_FAILURE;
                return allocation_report.status;
            }
            allocation_report.owned_rays +=
                (size_t)(shard.ray_end - shard.ray_begin);
            allocation_report.computed_rays_with_halos +=
                (size_t)shard.local_rays;
            allocation_report.total_device_allocated_bytes +=
                shard.allocated_bytes;
            allocation_report.max_device_allocated_bytes = std::max(
                allocation_report.max_device_allocated_bytes,
                shard.allocated_bytes);
            if (d < CMF_MGPU_REPORT_MAX_DEVICES)
                allocation_report.device_allocated_bytes[d] =
                    shard.allocated_bytes;
        }
        allocation_report.devices_used = devices;
        allocation_report.n_rays = (size_t)geometry.nr;
        allocation_report.deterministic_host_reduction = 1;
        allocation_report.persistent_context_initializations = 1U;
        initialization_seconds = monotonic_seconds() - initialize_start;
        allocation_report.initialization_seconds = initialization_seconds;
        allocation_report.status = CMF_MGPU_OK;
        ready = true;
        return CMF_MGPU_OK;
    }

    CMFMultiGPUStatus apply_round(
        const double *fixed, const double *input, bool zero_inner,
        int rounding, double *destination)
    {
        if (!ready || !fixed || !input || !destination ||
            rounding < -1 || rounding > 1)
            return CMF_MGPU_INVALID_INPUT;
        double phase_start = monotonic_seconds();
        for (size_t idx = 0; idx < cells; ++idx) {
            if (!(fixed[idx] >= 0.0) || !(input[idx] >= 0.0) ||
                !std::isfinite(fixed[idx]) || !std::isfinite(input[idx]))
                return CMF_MGPU_NONFINITE;
            double ratio = chi_tot[idx] > 0.0
                         ? chi_es[idx] / chi_tot[idx] : 0.0;
            if (rounding == 0) {
                source[idx] = fixed[idx] + ratio * input[idx];
                source_cell[idx] = (1.0 - t1[idx]) * source[idx];
            } else {
                double scattering;
                if (!host_positive_multiply_bound(
                        ratio, input[idx], rounding, &scattering) ||
                    !host_positive_add_bound(
                        fixed[idx], scattering, rounding, &source[idx]) ||
                    !host_positive_multiply_bound(
                        1.0 - t1[idx], source[idx], rounding,
                        &source_cell[idx]))
                    return CMF_MGPU_NONFINITE;
            }
            if (!(source[idx] >= 0.0) || !(source_cell[idx] >= 0.0) ||
                !std::isfinite(source[idx]) ||
                !std::isfinite(source_cell[idx]))
                return CMF_MGPU_NONFINITE;
        }
        source_assembly_seconds += monotonic_seconds() - phase_start;
        const std::vector<double> &inner =
            zero_inner ? inner_zero : inner_physical;
        phase_start = monotonic_seconds();
        for (DeviceShard &shard : shards) {
            if (cudaSetDevice(shard.device) != cudaSuccess ||
                cudaMemcpy(shard.source, source.data(),
                           cells * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.source_cell, source_cell.data(),
                           cells * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(shard.inner, inner.data(),
                           (size_t)n_bins * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                return CMF_MGPU_CUDA_FAILURE;
        }
        host_to_device_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        SweepFailureDetail sweep_detail;
        int sweep_status = launch_sweep(
            shards.data(), requested_devices, geometry, cells,
            true, max_positive_window, rounding, &sweep_detail,
            epoch_enabled ? &epoch_schedule : nullptr);
        if (sweep_status != 0) {
            if (sweep_status != 1)
                record_sweep_failure_detail(
                    &allocation_report, shards.data(), requested_devices,
                    n_bins, sweep_detail);
            return sweep_status == 1 ? CMF_MGPU_CUDA_FAILURE
                                     : CMF_MGPU_NONFINITE;
        }
        device_sweep_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        for (int d = 0; d < requested_devices; ++d) {
            DeviceShard &shard = shards[(size_t)d];
            if (cudaSetDevice(shard.device) != cudaSuccess ||
                cudaMemcpy(partial.data() + (size_t)d * cells,
                           shard.partial, cells * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return CMF_MGPU_CUDA_FAILURE;
        }
        device_to_host_seconds += monotonic_seconds() - phase_start;
        phase_start = monotonic_seconds();
        for (size_t idx = 0; idx < cells; ++idx) {
            double sum = 0.0;
            for (int d = requested_devices - 1; d >= 0; --d) {
                double value = partial[(size_t)d * cells + idx];
                if (rounding == 0) sum += value;
                else if (!host_positive_add_bound(
                             sum, value, rounding, &sum))
                    return CMF_MGPU_NONFINITE;
            }
            if (!(sum >= 0.0) || !std::isfinite(sum))
                return CMF_MGPU_NONFINITE;
            destination[idx] = sum;
        }
        host_reduction_seconds += monotonic_seconds() - phase_start;
        ++bound_applications;
        return CMF_MGPU_OK;
    }

    CMFMultiGPUStatus apply_bounds(
        const double *fixed, const double *input, bool zero_inner,
        double *lower, double *nearest, double *upper)
    {
        CMFMultiGPUStatus status = apply_round(
            fixed, input, zero_inner, -1, lower);
        if (status == CMF_MGPU_OK)
            status = apply_round(fixed, input, zero_inner, 0, nearest);
        if (status == CMF_MGPU_OK)
            status = apply_round(fixed, input, zero_inner, 1, upper);
        if (status != CMF_MGPU_OK) return status;
        for (size_t idx = 0; idx < cells; ++idx) {
            if (!(nearest[idx] >= lower[idx]) ||
                !(upper[idx] >= nearest[idx]) ||
                !std::isfinite(upper[idx])) return CMF_MGPU_NONFINITE;
        }
        return CMF_MGPU_OK;
    }
};

struct MultiGPUEnvelopeContext {
    PersistentBoundContext *persistent;
    const double *zero_fixed;
    size_t cells;
};

static int multigpu_apply_scattering_upper(
    const double *input, double *upper_output, size_t n, void *opaque)
{
    MultiGPUEnvelopeContext *context =
        static_cast<MultiGPUEnvelopeContext *>(opaque);
    if (!context || !context->persistent || !input || !upper_output ||
        n != context->cells)
        return -1;
    CMFMultiGPUStatus status = context->persistent->apply_round(
        context->zero_fixed, input, true, 1, upper_output);
    if (status == CMF_MGPU_OK)
        ++context->persistent->upper_operator_applications;
    return status == CMF_MGPU_OK ? 0 : -1;
}

}  // namespace

static CMFMultiGPUStatus positive_solve_envelope_impl(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations, CMFMultiGPUPartitionMode partition_mode,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report,
    const CMFMultiGPUEpochSchedule *epoch_schedule)
{
    const double total_start = monotonic_seconds();
    CMFMultiGPUReport local{};
    local.status = CMF_MGPU_INVALID_INPUT;
    local.iteration_cap = iteration_cap;
    local.tolerance = tolerance;
    local.final_max_relative_change = INFINITY;
    local.final_max_absolute_change = INFINITY;
    local.max_scattering_ratio = INFINITY;
    local.fixed_point_absolute_error_bound = INFINITY;
    local.componentwise_residual_upper_max = INFINITY;
    local.componentwise_error_upper_min = INFINITY;
    local.componentwise_error_upper_max = INFINITY;
    local.positive_sliding = 1;
    local.epoch_frequency_parallel = epoch_schedule ? 1 : 0;
    if (epoch_schedule) {
        local.epoch_block_size = epoch_schedule->block_size;
        local.epoch_batch_cardinality =
            epoch_schedule->epoch_batch_cardinality;
        local.epoch_direct_replay_max_window =
            epoch_schedule->direct_replay_max_window;
    }
    local.failure_iteration = -1;
    local.failure_cell_index = std::numeric_limits<size_t>::max();
    local.failure_device_index = -1;
    local.failure_ray_begin = -1;
    local.failure_ray_end = -1;
    local.failure_segment_index = -1;
    local.failure_local_ray_index = -1;
    local.failure_bin_index = -1;
    local.failure_outward = -1;
    local.failure_global_ray_index = -1;
    if (report) *report = local;
    if ((epoch_schedule && !valid_epoch_schedule(epoch_schedule)) ||
        (partition_mode != CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS &&
         partition_mode != CMF_MGPU_PARTITION_EQUAL_RAYS) ||
        n_shells <= 0 || n_bins < 2 || !J || !error_upper ||
        !chi_tot || !chi_es || !S_fixed || requested_devices <= 0)
        return local.status;
    size_t cells;
    if (!checked_mul((size_t)n_shells, (size_t)n_bins, &cells) ||
        cells > std::numeric_limits<size_t>::max() / sizeof(double))
        return local.status;

    std::vector<double> j_work;
    std::vector<double> lower;
    std::vector<double> nearest;
    std::vector<double> upper;
    std::vector<double> residual;
    std::vector<double> candidate;
    std::vector<double> zero_fixed;
    try {
        j_work.assign(J, J + cells);
        lower.resize(cells);
        nearest.resize(cells);
        upper.resize(cells);
        residual.resize(cells);
        candidate.resize(cells);
        zero_fixed.assign(cells, 0.0);
    } catch (const std::bad_alloc &) {
        local.status = CMF_MGPU_ALLOCATION_FAILED;
        if (report) *report = local;
        return local.status;
    }

    CMFMultiGPUReport solve_report;
    CMFMultiGPUStatus status = solve_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed,
        j_work.data(), requested_devices, iteration_cap, tolerance,
        &solve_report, true, partition_mode, epoch_schedule);
    local = solve_report;
    local.componentwise_residual_upper_max = INFINITY;
    local.componentwise_error_upper_min = INFINITY;
    local.componentwise_error_upper_max = INFINITY;
    if (status != CMF_MGPU_OK) {
        if (report) *report = local;
        return status;
    }

    PersistentBoundContext persistent;
    double phase_start = monotonic_seconds();
    status = persistent.initialize(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, requested_devices,
        partition_mode, epoch_schedule);
    local.envelope_context_setup_seconds =
        monotonic_seconds() - phase_start;
    CMFMultiGPUReport bound_report = persistent.allocation_report;
    local.initialization_seconds += bound_report.initialization_seconds;
    local.max_device_allocated_bytes = std::max(
        local.max_device_allocated_bytes,
        bound_report.max_device_allocated_bytes);
    local.total_device_allocated_bytes = std::max(
        local.total_device_allocated_bytes,
        bound_report.total_device_allocated_bytes);
    local.epoch_workspace_bytes_per_device_max = std::max(
        local.epoch_workspace_bytes_per_device_max,
        bound_report.epoch_workspace_bytes_per_device_max);
    if (status != CMF_MGPU_OK) {
        local.status = status;
        if (report) *report = local;
        return status;
    }
    phase_start = monotonic_seconds();
    status = persistent.apply_bounds(
        S_fixed, j_work.data(), false,
        lower.data(), nearest.data(), upper.data());
    if (status != CMF_MGPU_OK) {
        local.status = status;
        if (report) *report = local;
        return status;
    }
    local.bounds_seconds = monotonic_seconds() - phase_start;

    phase_start = monotonic_seconds();
    double max_residual = 0.0;
    double max_scattering_ratio = 0.0;
    for (size_t idx = 0; idx < cells; ++idx) {
        double lower_distance;
        double upper_distance;
        if (!(lower[idx] >= 0.0) || !(upper[idx] >= lower[idx]) ||
            !host_absolute_difference_upper(
                lower[idx], j_work[idx], &lower_distance) ||
            !host_absolute_difference_upper(
                upper[idx], j_work[idx], &upper_distance)) {
            local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
            if (report) *report = local;
            return local.status;
        }
        residual[idx] = std::max(lower_distance, upper_distance);
        if (residual[idx] > max_residual) max_residual = residual[idx];
        double ratio = chi_tot[idx] > 0.0
                     ? chi_es[idx] / chi_tot[idx] : 0.0;
        if (ratio > max_scattering_ratio) max_scattering_ratio = ratio;
    }
    local.envelope_residual_seconds = monotonic_seconds() - phase_start;
    local.componentwise_residual_upper_max = max_residual;
    local.max_scattering_ratio = max_scattering_ratio;
    if (max_scattering_ratio < 1.0) {
        double factor = max_scattering_ratio == 0.0 ? 0.0
            : max_scattering_ratio / (1.0 - max_scattering_ratio);
        local.fixed_point_absolute_error_bound =
            factor * local.final_max_absolute_change;
    } else {
        local.fixed_point_absolute_error_bound = INFINITY;
    }
    if (!(max_scattering_ratio >= 0.0) ||
        !(max_scattering_ratio < 1.0) || !std::isfinite(max_residual)) {
        local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
        if (report) *report = local;
        return local.status;
    }

    double seed = 0.0;
    if (max_residual != 0.0) {
        double denominator = std::nextafter(
            1.0 - max_scattering_ratio, 0.0);
        if (!(denominator > 0.0)) {
            local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
            if (report) *report = local;
            return local.status;
        }
        seed = std::nextafter(max_residual / denominator, INFINITY);
        if (!(seed > 0.0) || !std::isfinite(seed)) {
            local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
            if (report) *report = local;
            return local.status;
        }
    }

    MultiGPUEnvelopeContext context = {
        &persistent, zero_fixed.data(), cells
    };
    CMFEnvelopeReport envelope_report;
    CMFEnvelopeStatus envelope_status = CMF_ENVELOPE_INVALID_INPUT;
    size_t seed_attempts = 0;
    phase_start = monotonic_seconds();
    for (; seed_attempts < 64U; ++seed_attempts) {
        std::fill(candidate.begin(), candidate.end(), seed);
        envelope_status = cmf_error_envelope_verify(
            cells, residual.data(), candidate.data(),
            multigpu_apply_scattering_upper, &context, &envelope_report);
        if (envelope_status == CMF_ENVELOPE_OK) break;
        if (envelope_status != CMF_ENVELOPE_NOT_SUPERSOLUTION ||
            !(seed > 0.0) || seed > 0.5 * DBL_MAX) {
            local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
            if (report) *report = local;
            return local.status;
        }
        seed *= 2.0;
    }
    local.envelope_verify_seconds = monotonic_seconds() - phase_start;
    if (envelope_status != CMF_ENVELOPE_OK) {
        local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
        if (report) *report = local;
        return local.status;
    }
    phase_start = monotonic_seconds();
    envelope_status = cmf_error_envelope_refine(
        cells, residual.data(), candidate.data(), refinement_iterations,
        multigpu_apply_scattering_upper, &context, &envelope_report);
    local.envelope_refine_seconds = monotonic_seconds() - phase_start;
    if (envelope_status != CMF_ENVELOPE_OK) {
        local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
        if (report) *report = local;
        return local.status;
    }

    double bound_min = INFINITY;
    double bound_max = 0.0;
    for (double value : candidate) {
        if (!(value >= 0.0) || !std::isfinite(value)) {
            local.status = CMF_MGPU_ERROR_ENVELOPE_FAILED;
            if (report) *report = local;
            return local.status;
        }
        if (value < bound_min) bound_min = value;
        if (value > bound_max) bound_max = value;
    }

    phase_start = monotonic_seconds();
    std::memcpy(J, j_work.data(), cells * sizeof(double));
    std::memcpy(error_upper, candidate.data(), cells * sizeof(double));
    local.publication_seconds = monotonic_seconds() - phase_start;
    local.status = CMF_MGPU_OK;
    local.componentwise_error_envelope_verified = 1;
    local.componentwise_error_seed_attempts = seed_attempts + 1U;
    local.componentwise_error_refinement_iterations =
        envelope_report.iterations_completed;
    local.componentwise_error_upper_min = bound_min;
    local.componentwise_error_upper_max = bound_max;
    local.persistent_context_initializations =
        persistent.allocation_report.persistent_context_initializations;
    local.persistent_bound_applications = persistent.bound_applications;
    local.persistent_upper_operator_applications =
        persistent.upper_operator_applications;
    local.source_assembly_seconds += persistent.source_assembly_seconds;
    local.host_to_device_seconds += persistent.host_to_device_seconds;
    local.device_sweep_seconds += persistent.device_sweep_seconds;
    local.device_to_host_seconds += persistent.device_to_host_seconds;
    local.host_reduction_seconds += persistent.host_reduction_seconds;
    persistent.release();
    local.cleanup_seconds += persistent.cleanup_seconds;
    local.total_seconds = monotonic_seconds() - total_start;
    if (report) *report = local;
    return local.status;
}

extern "C" CMFMultiGPUStatus cmf_exact_multigpu_positive_solve_envelope(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report)
{
    return positive_solve_envelope_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        error_upper, refinement_iterations,
        CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, requested_devices,
        iteration_cap, tolerance, report, nullptr);
}

extern "C" CMFMultiGPUStatus
cmf_exact_multigpu_positive_solve_envelope_partitioned(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations, CMFMultiGPUPartitionMode partition_mode,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report)
{
    return positive_solve_envelope_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        error_upper, refinement_iterations, partition_mode,
        requested_devices, iteration_cap, tolerance, report, nullptr);
}

extern "C" CMFMultiGPUStatus
cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t refinement_iterations, CMFMultiGPUPartitionMode partition_mode,
    const CMFMultiGPUEpochSchedule *schedule,
    int requested_devices, int iteration_cap, double tolerance,
    CMFMultiGPUReport *report)
{
    return positive_solve_envelope_impl(
        n_shells, n_bins, dlognu, nu, r_inner, r_outer, time_explosion,
        T_inner, inner_boundary_scale, chi_tot, chi_es, S_fixed, J,
        error_upper, refinement_iterations, partition_mode,
        requested_devices, iteration_cap, tolerance, report, schedule);
}
