#include "gpu_emissivity_kernels.h"

#include <cuda_runtime.h>
#include <math.h>

__global__ static void emissivity_cdf_kernel(GpuRadiationFieldDeviceView view,
    const double *component, const double *u, double *total, double *cdf,
    double *sample_nu, int *status)
{
    size_t s = blockIdx.x;
    if (threadIdx.x || s >= view.n_shells) return;
    size_t cells = view.n_shells * view.n_bins;
    double norm = 0.0;
    for (size_t b = 0; b < view.n_bins; ++b) {
        size_t i = s * view.n_bins + b;
        double sum = 0.0;
        for (size_t k = 0; k < 5; ++k) {
            double v = component[k * cells + i];
            if (!isfinite(v) || v < 0.0) { status[s] = 1; return; }
            sum += v;
        }
        total[i] = sum;
        norm += sum * (view.frequency_bin_edges[b + 1] -
                       view.frequency_bin_edges[b]);
        cdf[i] = norm;
    }
    if (!(norm > 0.0) || !isfinite(norm) || !isfinite(u[s]) ||
        u[s] < 0.0 || u[s] >= 1.0) { status[s] = 1; return; }
    size_t selected = view.n_bins - 1;
    double previous = 0.0;
    for (size_t b = 0; b < view.n_bins; ++b) {
        size_t i = s * view.n_bins + b;
        cdf[i] = b + 1 == view.n_bins ? 1.0 : cdf[i] / norm;
        if (cdf[i] < previous) { status[s] = 1; return; }
        if (u[s] <= cdf[i] && selected == view.n_bins - 1) selected = b;
        previous = cdf[i];
    }
    size_t i = s * view.n_bins + selected;
    double c0 = selected ? cdf[i - 1] : 0.0, c1 = cdf[i];
    double f = c1 > c0 ? (u[s] - c0) / (c1 - c0) : 0.0;
    sample_nu[s] = view.frequency_bin_edges[selected] + f *
        (view.frequency_bin_edges[selected + 1] -
         view.frequency_bin_edges[selected]);
}

int gpu_physics_emissivity_cdf(const GpuRadiationFieldDeviceView *view,
    uint64_t expected_generation, const double *component,
    const double *sample_u, double *eta_total, double *cdf,
    double *sample_nu, GpuPhysicsCounters *c, void *cuda_stream)
{
    if (!view || !c || !component || !sample_u || !eta_total || !cdf ||
        !sample_nu || view->generation == 0 ||
        expected_generation != view->generation ||
        c->required_generation != view->generation) {
        if (c) { c->blocked_generation++; c->blocked_launches++; }
        return -1;
    }
    size_t cells = view->n_shells * view->n_bins;
    if (!cells || cells > SIZE_MAX / (5 * sizeof(double))) return -1;
    double *d_component = NULL, *d_u = NULL, *d_total = NULL, *d_cdf = NULL;
    double *d_nu = NULL;
    int *d_status = NULL;
    int *status = (int *)calloc(view->n_shells, sizeof(int));
    if (!status) return -1;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    size_t cell_bytes = cells * sizeof(double);
#define CMALLOC(p, n) if (cudaMalloc(&(p), (n)) != cudaSuccess) goto fail
    CMALLOC(d_component, 5 * cell_bytes); CMALLOC(d_u, view->n_shells * sizeof(double));
    CMALLOC(d_total, cell_bytes); CMALLOC(d_cdf, cell_bytes);
    CMALLOC(d_nu, view->n_shells * sizeof(double));
    CMALLOC(d_status, view->n_shells * sizeof(int));
#undef CMALLOC
    if (cudaMemcpyAsync(d_component, component, 5 * cell_bytes,
            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_u, sample_u, view->n_shells * sizeof(double),
            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemsetAsync(d_status, 0, view->n_shells * sizeof(int), stream) != cudaSuccess)
        goto fail;
    c->emissivity_cells_attempted += cells;
    c->physical_launches++;
    emissivity_cdf_kernel<<<(unsigned)view->n_shells, 1, 0, stream>>>(
        *view, d_component, d_u, d_total, d_cdf, d_nu, d_status);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(eta_total, d_total, cell_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(cdf, d_cdf, cell_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(sample_nu, d_nu, view->n_shells * sizeof(double), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(status, d_status, view->n_shells * sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) goto fail;
    for (size_t s = 0; s < view->n_shells; ++s)
        if (status[s]) { gpu_physics_record_validity(c, GPU_PHYSICS_NONFINITE); goto fail; }
    c->emissivity_cells_published += cells;
    c->cpu_gpu_emissivity_compared += cells;
    c->rng_draws_gpu += view->n_shells;
    cudaFree(d_component); cudaFree(d_u); cudaFree(d_total); cudaFree(d_cdf);
    cudaFree(d_nu); cudaFree(d_status); free(status);
    return 0;
fail:
    cudaFree(d_component); cudaFree(d_u); cudaFree(d_total); cudaFree(d_cdf);
    cudaFree(d_nu); cudaFree(d_status); free(status); c->blocked_launches++;
    return -1;
}
