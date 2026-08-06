#ifndef LUMINA_GPU_EMISSIVITY_KERNELS_H
#define LUMINA_GPU_EMISSIVITY_KERNELS_H

#include "gpu_physics_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* component is [5][n_shells*n_bins], matching the A2-09 owner ordering. */
int gpu_physics_emissivity_cdf(
    const GpuRadiationFieldDeviceView *view, uint64_t expected_generation,
    const double *component, const double *sample_u,
    double *eta_total, double *cdf, double *sample_nu,
    GpuPhysicsCounters *counters, void *cuda_stream);

#ifdef __cplusplus
}
#endif
#endif
