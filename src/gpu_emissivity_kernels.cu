#include "gpu_emissivity_kernels.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

typedef struct {
    double *edges, *eta, *cdf;
    GpuEmissivityDeviceView view;
} GpuEmissivityProductionMirror;

static GpuEmissivityProductionMirror production_emissivity;

static void production_emissivity_clear(void)
{
    if (production_emissivity.edges) cudaFree(production_emissivity.edges);
    if (production_emissivity.eta) cudaFree(production_emissivity.eta);
    if (production_emissivity.cdf) cudaFree(production_emissivity.cdf);
    memset(&production_emissivity, 0, sizeof(production_emissivity));
}

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

int gpu_emissivity_production_bind(const CpuEmissivityPublication *p,
    const GpuRadiationFieldDeviceView *rf, GpuEmissivityDeviceView *out,
    GpuPhysicsCounters *c, void *cuda_stream)
{
    if (!p || !rf || !out || !c || !p->committed_emissivity_generation ||
        p->cdf_generation != p->committed_emissivity_generation ||
        p->opacity_generation != p->committed_emissivity_generation ||
        p->radfield_generation != rf->generation ||
        p->line_view_generation != rf->generation ||
        p->n_shells != rf->n_shells || p->n_bins != rf->n_bins ||
        p->redistribution_status != EMISS_OK || !p->nu_edge ||
        !p->eta_reemit || !p->reemit_cdf) {
        c->blocked_generation++; c->blocked_launches++; return -1;
    }
    size_t cells = p->n_shells * p->n_bins;
    for (size_t i = 0; i < cells; ++i) {
        if ((p->cell_status[i] != EMISS_OK &&
             p->cell_status[i] != EMISS_EXACT_ZERO) ||
            !isfinite(p->eta_reemit[i]) || p->eta_reemit[i] < 0.0 ||
            !isfinite(p->reemit_cdf[i]) || p->reemit_cdf[i] < 0.0 ||
            p->reemit_cdf[i] > 1.0) {
            c->blocked_nonfinite++; c->blocked_launches++; return -1;
        }
    }
    production_emissivity_clear();
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    size_t edge_bytes = (p->n_bins + 1) * sizeof(double);
    size_t cell_bytes = cells * sizeof(double);
    if (cudaMalloc(&production_emissivity.edges, edge_bytes) != cudaSuccess ||
        cudaMalloc(&production_emissivity.eta, cell_bytes) != cudaSuccess ||
        cudaMalloc(&production_emissivity.cdf, cell_bytes) != cudaSuccess ||
        cudaMemcpyAsync(production_emissivity.edges, p->nu_edge, edge_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(production_emissivity.eta, p->eta_reemit, cell_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(production_emissivity.cdf, p->reemit_cdf, cell_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) goto fail;
    production_emissivity.view.frequency_bin_edges = production_emissivity.edges;
    production_emissivity.view.eta_reemit = production_emissivity.eta;
    production_emissivity.view.reemit_cdf = production_emissivity.cdf;
    production_emissivity.view.emissivity_generation =
        p->committed_emissivity_generation;
    production_emissivity.view.opacity_generation = p->opacity_generation;
    production_emissivity.view.radiation_generation = p->radfield_generation;
    production_emissivity.view.line_generation = p->line_view_generation;
    production_emissivity.view.n_shells = p->n_shells;
    production_emissivity.view.n_bins = p->n_bins;
    c->emissivity_cells_attempted += cells;
    c->emissivity_cells_published += cells;
    c->gpu_generation = rf->generation;
    c->line_generation = rf->generation;
    *out = production_emissivity.view;
    return 0;
fail:
    production_emissivity_clear(); c->blocked_launches++; return -1;
}

int gpu_emissivity_production_view(const CpuEmissivityPublication *p,
    const GpuRadiationFieldDeviceView *rf, GpuEmissivityDeviceView *out,
    GpuPhysicsCounters *c)
{
    if (!p || !rf || !out || !c ||
        production_emissivity.view.emissivity_generation !=
            p->committed_emissivity_generation ||
        production_emissivity.view.radiation_generation != rf->generation ||
        production_emissivity.view.line_generation != rf->generation) {
        if (c) { c->blocked_generation++; c->blocked_launches++; }
        return -1;
    }
    *out = production_emissivity.view;
    return 0;
}

void gpu_emissivity_production_release(void)
{ production_emissivity_clear(); }
