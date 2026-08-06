#include "gpu_opacity_kernels.h"

#include <cuda_runtime.h>
#include <math.h>

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
