#include "gpu_opacity_kernels.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

typedef struct {
    double *component[6];
    GpuOpacityDeviceView view;
} GpuOpacityProductionMirror;

static GpuOpacityProductionMirror production_opacity;

__global__ static void production_opacity_total_kernel(
    const double *es, const double *bb, const double *bf, const double *ff,
    size_t n, double *total)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) total[i] = ((es[i] + bb[i]) + bf[i]) + ff[i];
}

static void production_opacity_clear(void)
{
    for (int k = 0; k < 6; ++k) {
        if (production_opacity.component[k])
            cudaFree(production_opacity.component[k]);
        production_opacity.component[k] = NULL;
    }
    memset(&production_opacity.view, 0, sizeof(production_opacity.view));
}

__global__ static void signed_opacity_kernel(const double *es, const double *bb,
    const double *bf, const double *ff, const double *event, size_t n,
    GpuOpacityCell *out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    GpuOpacityCell v;
    v.es = es[i]; v.bb = bb[i]; v.bf = bf[i]; v.ff = ff[i];
    v.total = ((v.es + v.bb) + v.bf) + v.ff;
    v.bf_event_measure = event[i];
    v.validity = isfinite(v.total) && isfinite(v.bf_event_measure) &&
                 v.bf_event_measure >= 0.0
        ? (v.total == 0.0 ? GPU_PHYSICS_EXACT_ZERO : GPU_PHYSICS_VALID)
        : GPU_PHYSICS_NONFINITE;
    out[i] = v;
}

int gpu_physics_signed_opacity(const GpuRadiationFieldDeviceView *view,
    uint64_t expected_generation, const double *es, const double *bb,
    const double *bf_net, const double *ff, const double *bf_event_measure,
    size_t n, GpuOpacityCell *out, GpuPhysicsCounters *c, void *cuda_stream)
{
    if (!view || !c || !out || !es || !bb || !bf_net || !ff ||
        !bf_event_measure || n == 0 || view->generation == 0 ||
        expected_generation != view->generation ||
        c->required_generation != view->generation) {
        if (c) { c->blocked_generation++; c->blocked_launches++; }
        return -1;
    }
    c->opacity_cells_attempted += n;
    double *d_es = NULL, *d_bb = NULL, *d_bf = NULL, *d_ff = NULL, *d_event = NULL;
    GpuOpacityCell *d_out = NULL;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    size_t bytes = n * sizeof(double), out_bytes = n * sizeof(*out);
#define ALLOC_COPY(dst, src) \
    if (cudaMalloc(&(dst), bytes) != cudaSuccess || \
        cudaMemcpyAsync((dst), (src), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) goto fail
    ALLOC_COPY(d_es, es); ALLOC_COPY(d_bb, bb); ALLOC_COPY(d_bf, bf_net);
    ALLOC_COPY(d_ff, ff); ALLOC_COPY(d_event, bf_event_measure);
#undef ALLOC_COPY
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) goto fail;
    c->physical_launches++;
    signed_opacity_kernel<<<(unsigned)((n + 127) / 128), 128, 0, stream>>>(
        d_es, d_bb, d_bf, d_ff, d_event, n, d_out);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) goto fail;
    cudaFree(d_es); cudaFree(d_bb); cudaFree(d_bf); cudaFree(d_ff);
    cudaFree(d_event); cudaFree(d_out);
    for (size_t i = 0; i < n; ++i) {
        if (out[i].validity != GPU_PHYSICS_VALID &&
            out[i].validity != GPU_PHYSICS_EXACT_ZERO) {
            gpu_physics_record_validity(c, out[i].validity);
            return -1;
        }
        c->opacity_cells_published++;
    }
    c->cpu_gpu_opacity_compared += n;
    return 0;
fail:
    cudaFree(d_es); cudaFree(d_bb); cudaFree(d_bf); cudaFree(d_ff);
    cudaFree(d_event); cudaFree(d_out); c->blocked_launches++;
    return -1;
}

int gpu_opacity_production_bind(const CpuOpacityPublication *p,
    const GpuRadiationFieldDeviceView *rf, GpuOpacityDeviceView *out,
    GpuPhysicsCounters *c, void *cuda_stream)
{
    if (!p || !rf || !out || !c || !p->generation_committed ||
        p->radiation_generation != rf->generation ||
        p->line_jbar_generation != rf->generation ||
        p->n_shells != rf->n_shells || p->n_bins != rf->n_bins ||
        !p->chi_es || !p->chi_bb || !p->chi_bf || !p->chi_ff ||
        !p->chi_total || !p->chi_validity || !p->bf_event_measure ||
        p->n_routes != 1) {
        c->blocked_generation++; c->blocked_launches++;
        return -1;
    }
    const size_t n = p->n_shells * p->n_bins;
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            A208Validity v = p->chi_validity[k * n + i];
            if (v != A208_VALID && v != A208_EXACT_ZERO) {
                c->blocked_nonfinite++; c->blocked_launches++; return -1;
            }
        }
        if (!isfinite(p->bf_event_measure[i]) || p->bf_event_measure[i] < 0.0) {
            c->blocked_nonfinite++; c->blocked_launches++; return -1;
        }
    }
    production_opacity_clear();
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    const double *src[5] = {p->chi_es, p->chi_bb, p->chi_bf, p->chi_ff,
                            p->bf_event_measure};
    size_t bytes = n * sizeof(double);
    for (int k = 0; k < 6; ++k)
        if (cudaMalloc(&production_opacity.component[k], bytes) != cudaSuccess)
            goto fail;
    for (int k = 0; k < 5; ++k)
        if (cudaMemcpyAsync(production_opacity.component[k], src[k], bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
            goto fail;
    c->opacity_cells_attempted += n;
    c->physical_launches++;
    production_opacity_total_kernel<<<(unsigned)((n + 127) / 128), 128, 0,
                                      stream>>>(
        production_opacity.component[0], production_opacity.component[1],
        production_opacity.component[2], production_opacity.component[3], n,
        production_opacity.component[5]);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) goto fail;
    production_opacity.view.chi_es = production_opacity.component[0];
    production_opacity.view.chi_bb = production_opacity.component[1];
    production_opacity.view.chi_bf = production_opacity.component[2];
    production_opacity.view.chi_ff = production_opacity.component[3];
    production_opacity.view.bf_event_measure = production_opacity.component[4];
    production_opacity.view.chi_total = production_opacity.component[5];
    production_opacity.view.opacity_generation = p->generation_committed;
    production_opacity.view.radiation_generation = p->radiation_generation;
    production_opacity.view.line_generation = p->line_jbar_generation;
    production_opacity.view.n_shells = p->n_shells;
    production_opacity.view.n_bins = p->n_bins;
    c->gpu_generation = rf->generation;
    c->line_generation = rf->generation;
    c->opacity_cells_published += n;
    *out = production_opacity.view;
    return 0;
fail:
    production_opacity_clear(); c->blocked_launches++; return -1;
}

int gpu_opacity_production_view(const CpuOpacityPublication *p,
    const GpuRadiationFieldDeviceView *rf, GpuOpacityDeviceView *out,
    GpuPhysicsCounters *c)
{
    if (!p || !rf || !out || !c ||
        production_opacity.view.opacity_generation != p->generation_committed ||
        production_opacity.view.radiation_generation != rf->generation ||
        production_opacity.view.line_generation != rf->generation) {
        if (c) { c->blocked_generation++; c->blocked_launches++; }
        return -1;
    }
    *out = production_opacity.view;
    return 0;
}

void gpu_opacity_production_release(void) { production_opacity_clear(); }
