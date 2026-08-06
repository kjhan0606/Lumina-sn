#ifndef LUMINA_GPU_EMISSIVITY_KERNELS_H
#define LUMINA_GPU_EMISSIVITY_KERNELS_H

#include "gpu_physics_kernels.h"
#include "emissivity_publication.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* component is [5][n_shells*n_bins], matching the A2-09 owner ordering. */
typedef struct {
    const double *frequency_bin_edges;
    const double *eta_reemit;
    const double *reemit_cdf;
    uint64_t emissivity_generation, opacity_generation;
    uint64_t radiation_generation, line_generation;
    size_t n_shells, n_bins;
} GpuEmissivityDeviceView;

int gpu_physics_emissivity_cdf(
    const GpuRadiationFieldDeviceView *view, uint64_t expected_generation,
    const double *component, const double *sample_u,
    double *eta_total, double *cdf, double *sample_nu,
    GpuPhysicsCounters *counters, void *cuda_stream);
int gpu_emissivity_production_bind(
    const CpuEmissivityPublication *publication,
    const GpuRadiationFieldDeviceView *radiation_view,
    GpuEmissivityDeviceView *out, GpuPhysicsCounters *counters,
    void *cuda_stream);
int gpu_emissivity_production_view(
    const CpuEmissivityPublication *publication,
    const GpuRadiationFieldDeviceView *radiation_view,
    GpuEmissivityDeviceView *out, GpuPhysicsCounters *counters);
void gpu_emissivity_production_release(void);

#ifdef __CUDACC__
static __device__ __forceinline__ int gpu_emissivity_sample_device(
    GpuEmissivityDeviceView view, int shell, double u, double *frequency)
{
    if (!frequency || shell < 0 || (size_t)shell >= view.n_shells ||
        !view.frequency_bin_edges || !view.reemit_cdf || !view.eta_reemit ||
        !view.n_bins || !view.emissivity_generation ||
        view.emissivity_generation != view.opacity_generation ||
        view.radiation_generation != view.line_generation ||
        !(u >= 0.0 && u < 1.0)) return -1;
    const double *cdf = view.reemit_cdf + (size_t)shell * view.n_bins;
    size_t lo = 0, hi = view.n_bins - 1;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    double c0 = lo ? cdf[lo - 1] : 0.0, c1 = cdf[lo];
    if (!(c1 >= c0) || !(c1 > 0.0)) return -1;
    double f = c1 > c0 ? (u - c0) / (c1 - c0) : 0.0;
    *frequency = view.frequency_bin_edges[lo] + f *
        (view.frequency_bin_edges[lo + 1] - view.frequency_bin_edges[lo]);
    return isfinite(*frequency) && *frequency > 0.0 ? 0 : -1;
}
#endif

#ifdef __cplusplus
}
#endif
#endif
