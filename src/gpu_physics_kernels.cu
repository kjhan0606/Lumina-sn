#include "gpu_physics_kernels.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#define GPU_BF_H_CGS 6.62607015e-27
#define GPU_FOUR_PI 12.56637061435917295385

__device__ static double sigma_over_nu(double n0, double s0, double n1,
                                       double s1, double a, double b)
{
    if (!(b > a) || !(n1 > n0)) return 0.0;
    double m = (s1 - s0) / (n1 - n0);
    return (s0 - m * n0) * log(b / a) + m * (b - a);
}

__global__ static void bf_rate_kernel(GpuRadiationFieldDeviceView view,
    size_t shell, const double *sigma_nu, const double *sigma_value,
    size_t n_points, double threshold, GpuBfRateCell *out)
{
    if (blockIdx.x || threadIdx.x) return;
    double integral = 0.0;
    int exact_zero = 1;
    for (size_t b = 0; b < view.n_bins; ++b) {
        double lo = view.frequency_bin_edges[b] > threshold
            ? view.frequency_bin_edges[b] : threshold;
        double hi = view.frequency_bin_edges[b + 1];
        if (!(hi > lo)) continue;
        double kernel = 0.0;
        for (size_t k = 0; k + 1 < n_points; ++k) {
            double a = sigma_nu[k] > lo ? sigma_nu[k] : lo;
            double c = sigma_nu[k + 1] < hi ? sigma_nu[k + 1] : hi;
            kernel += sigma_over_nu(sigma_nu[k], sigma_value[k],
                                    sigma_nu[k + 1], sigma_value[k + 1], a, c);
        }
        if (!(kernel > 0.0)) continue;
        size_t i = shell * view.n_bins + b;
        RadiationFieldValidityState validity = view.field_validity[i];
        if (validity == RADIATION_FIELD_VALID) {
            integral += view.J_nu[i] * kernel / GPU_BF_H_CGS;
            exact_zero = 0;
        } else if (validity != RADIATION_FIELD_EXACT_ZERO) {
            out->gamma = 0.0;
            out->validity = validity == RADIATION_FIELD_UNSAMPLED
                ? GPU_PHYSICS_UNSAMPLED
                : validity == RADIATION_FIELD_STALE ? GPU_PHYSICS_STALE
                                                    : GPU_PHYSICS_OOG;
            return;
        }
    }
    out->gamma = GPU_FOUR_PI * integral;
    out->validity = exact_zero ? GPU_PHYSICS_EXACT_ZERO : GPU_PHYSICS_VALID;
}

__global__ static void bb_rate_kernel(GpuRadiationFieldDeviceView view,
    size_t shell, uint64_t wanted, double B_lu, double B_ul, double A_ul,
    GpuBbRateCell *out)
{
    if (blockIdx.x || threadIdx.x) return;
    size_t lo = 0, hi = view.n_lines;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (view.line_id[mid] < wanted) lo = mid + 1; else hi = mid;
    }
    if (lo >= view.n_lines || view.line_id[lo] != wanted) {
        out->validity = GPU_PHYSICS_MISS;
        return;
    }
    size_t i = lo * view.n_shells + shell;
    LineJbarValidityState validity = view.line_validity[i];
    if (validity != LINE_JBAR_VALID && validity != LINE_JBAR_EXACT_ZERO) {
        out->validity = validity == LINE_JBAR_UNSAMPLED
            ? GPU_PHYSICS_UNSAMPLED : GPU_PHYSICS_OOG;
        return;
    }
    double j = view.line_jbar[i];
    out->jbar = j;
    out->upward = B_lu * j;
    out->stimulated_downward = B_ul * j;
    out->spontaneous_downward = A_ul;
    out->se = view.line_se[i];
    out->validity = validity == LINE_JBAR_EXACT_ZERO
        ? GPU_PHYSICS_EXACT_ZERO : GPU_PHYSICS_VALID;
}

static int checked_view(const GpuRadiationFieldDeviceView *view, size_t shell,
                        GpuPhysicsCounters *counters)
{
    if (!view || !counters || !view->frequency_bin_edges || !view->J_nu ||
        !view->field_validity || !view->line_id || !view->line_jbar ||
        !view->line_validity || view->generation == 0 ||
        shell >= view->n_shells) return -1;
    if (counters->required_generation != view->generation) {
        counters->blocked_generation++;
        counters->blocked_launches++;
        return -1;
    }
    counters->gpu_generation = view->generation;
    counters->line_generation = view->generation;
    return 0;
}

int gpu_physics_bf_rate(const GpuRadiationFieldDeviceView *view, size_t shell,
    const double *sigma_nu, const double *sigma_value, size_t n_points,
    double threshold, GpuBfRateCell *out, GpuPhysicsCounters *counters,
    void *cuda_stream)
{
    if (checked_view(view, shell, counters) || !sigma_nu || !sigma_value ||
        n_points < 2 || !(threshold > 0.0) || !out) return -1;
    counters->bf_cells_attempted++;
    double *d_nu = NULL, *d_sigma = NULL;
    GpuBfRateCell *d_out = NULL;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    size_t bytes = n_points * sizeof(double);
    if (cudaMalloc(&d_nu, bytes) != cudaSuccess ||
        cudaMalloc(&d_sigma, bytes) != cudaSuccess ||
        cudaMalloc(&d_out, sizeof(*out)) != cudaSuccess ||
        cudaMemcpyAsync(d_nu, sigma_nu, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_sigma, sigma_value, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemsetAsync(d_out, 0, sizeof(*out), stream) != cudaSuccess) goto fail;
    counters->physical_launches++;
    bf_rate_kernel<<<1, 1, 0, stream>>>(*view, shell, d_nu, d_sigma,
                                       n_points, threshold, d_out);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(out, d_out, sizeof(*out), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) goto fail;
    cudaFree(d_nu); cudaFree(d_sigma); cudaFree(d_out);
    if (out->validity == GPU_PHYSICS_VALID ||
        out->validity == GPU_PHYSICS_EXACT_ZERO) counters->bf_cells_published++;
    else gpu_physics_record_validity(counters, out->validity);
    return 0;
fail:
    cudaFree(d_nu); cudaFree(d_sigma); cudaFree(d_out);
    counters->blocked_launches++;
    return -1;
}

int gpu_physics_bb_rate(const GpuRadiationFieldDeviceView *view, size_t shell,
    uint64_t line_id, double B_lu, double B_ul, double A_ul,
    GpuBbRateCell *out, GpuPhysicsCounters *counters, void *cuda_stream)
{
    if (checked_view(view, shell, counters) || !out || !isfinite(B_lu) ||
        !isfinite(B_ul) || !isfinite(A_ul) || B_lu < 0.0 || B_ul < 0.0 ||
        A_ul < 0.0) return -1;
    counters->bb_cells_attempted++;
    counters->line_cache_lookups++;
    GpuBbRateCell *d_out = NULL;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    if (cudaMalloc(&d_out, sizeof(*out)) != cudaSuccess ||
        cudaMemsetAsync(d_out, 0, sizeof(*out), stream) != cudaSuccess) goto fail;
    counters->physical_launches++;
    bb_rate_kernel<<<1, 1, 0, stream>>>(*view, shell, line_id,
                                        B_lu, B_ul, A_ul, d_out);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(out, d_out, sizeof(*out), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) goto fail;
    cudaFree(d_out);
    if (out->validity == GPU_PHYSICS_VALID ||
        out->validity == GPU_PHYSICS_EXACT_ZERO) {
        counters->bb_cells_published++;
        counters->line_cache_hits++;
    } else gpu_physics_record_validity(counters, out->validity);
    return 0;
fail:
    cudaFree(d_out);
    counters->blocked_launches++;
    return -1;
}
