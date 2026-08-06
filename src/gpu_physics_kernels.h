#ifndef LUMINA_GPU_PHYSICS_KERNELS_H
#define LUMINA_GPU_PHYSICS_KERNELS_H

#include "gpu_physics_contract.h"
#include "gpu_radiation_field.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double gamma;
    GpuPhysicsValidity validity;
} GpuBfRateCell;

typedef struct {
    double jbar, upward, stimulated_downward, spontaneous_downward, se;
    GpuPhysicsValidity validity;
} GpuBbRateCell;

/* Cross-section points are one ascending piecewise-linear table. */
int gpu_physics_bf_rate(const GpuRadiationFieldDeviceView *view, size_t shell,
                        const double *sigma_nu, const double *sigma_value,
                        size_t n_points, double threshold,
                        GpuBfRateCell *out, GpuPhysicsCounters *counters,
                        void *cuda_stream);
int gpu_physics_bb_rate(const GpuRadiationFieldDeviceView *view, size_t shell,
                        uint64_t line_id, double B_lu, double B_ul, double A_ul,
                        GpuBbRateCell *out, GpuPhysicsCounters *counters,
                        void *cuda_stream);

#ifdef __cplusplus
}
#endif
#endif
