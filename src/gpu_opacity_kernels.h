#ifndef LUMINA_GPU_OPACITY_KERNELS_H
#define LUMINA_GPU_OPACITY_KERNELS_H

#include "gpu_physics_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double es, bb, bf, ff, total;
    double bf_event_measure;
    GpuPhysicsValidity validity;
} GpuOpacityCell;

int gpu_physics_signed_opacity(
    const GpuRadiationFieldDeviceView *view, uint64_t expected_generation,
    const double *es, const double *bb, const double *bf_net,
    const double *ff, const double *bf_event_measure, size_t n_cells,
    GpuOpacityCell *out, GpuPhysicsCounters *counters, void *cuda_stream);

#ifdef __cplusplus
}
#endif
#endif
