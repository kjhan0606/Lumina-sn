#ifndef LUMINA_GPU_OPACITY_KERNELS_H
#define LUMINA_GPU_OPACITY_KERNELS_H

#include "gpu_physics_kernels.h"
#include "opacity_publication.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double es, bb, bf, ff, total;
    double bf_event_measure;
    GpuPhysicsValidity validity;
} GpuOpacityCell;

/* A2-14 production publication.  The signed channels and the nonnegative
 * packet-event measure are distinct device fields and are valid only while
 * both the A2-08 publication and the A2-12 radiation view remain at the
 * generations recorded here. */
typedef struct {
    const double *chi_es, *chi_bb, *chi_bf, *chi_ff, *chi_total;
    const double *bf_event_measure;
    int bf_event_measure_provenance;
    uint64_t opacity_generation, radiation_generation, line_generation;
    size_t n_shells, n_bins;
} GpuOpacityDeviceView;

int gpu_physics_signed_opacity(
    const GpuRadiationFieldDeviceView *view, uint64_t expected_generation,
    const double *es, const double *bb, const double *bf_net,
    const double *ff, const double *bf_event_measure, size_t n_cells,
    GpuOpacityCell *out, GpuPhysicsCounters *counters, void *cuda_stream);
int gpu_opacity_production_bind(
    const CpuOpacityPublication *publication,
    const GpuRadiationFieldDeviceView *radiation_view,
    GpuOpacityDeviceView *out, GpuPhysicsCounters *counters,
    void *cuda_stream);
int gpu_opacity_production_view(
    const CpuOpacityPublication *publication,
    const GpuRadiationFieldDeviceView *radiation_view,
    GpuOpacityDeviceView *out, GpuPhysicsCounters *counters);
void gpu_opacity_production_release(void);

#ifdef __cplusplus
}
#endif
#endif
