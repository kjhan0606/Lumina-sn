/* lumina_bf_gemm.cu — Task #39: GPU bf opacity via cuBLAS GEMM (TF32).
 * Extracted from lumina_cuda.cu so the validation harness (bench_bf_gemm)
 * can link against it without pulling in the transport main(). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
#include "lumina.h"
}

#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while(0)

/* ============================================================ */
/* Task #39: GPU bf opacity via cuBLAS GEMM (TF32 tensor cores) */
/* ============================================================ */
/* Reformulates per-level loop as: chi_bf[s,f] = n_level[s,l] @ sigma_bf[l,f]
 * with sigma_bf pre-baked on the LUMINA bf grid (CMFGEN cs_type 1/20-22 +
 * Kramers approx for 2,3,7,8,9). Uses CUBLAS_COMPUTE_32F_FAST_TF32 to engage
 * Ampere+ tensor cores (FP32 inputs, TF32 accumulate). On A40 (sm_86) this
 * gives ~70 TFLOPS vs <1 TFLOPS for naive FP32. Levels with sigma=0 below
 * their photo-edge contribute zero to the GEMM, so per-level edge cutoff is
 * folded into the pre-baked grid. */

typedef struct {
    int initialized;
    int n_levels, n_freq_bins, n_shells, n_ion_pops;
    /* Persistent device arrays */
    float  *d_sigma_bf;       /* [n_freq_bins x n_levels] col-major (= [l x nf] row-major) */
    float  *d_n_level;        /* [n_levels x n_shells]   col-major (= [s x nl] row-major) */
    float  *d_chi_bf;         /* [n_freq_bins x n_shells] col-major (= [s x nf] row-major) */
    double *d_T_rad;          /* [n_shells] */
    double *d_W;              /* [n_shells] */
    double *d_n_ion;          /* [n_ion_pops x n_shells] row-major */
    double *d_Z_part;         /* [n_ion_pops x n_shells] row-major */
    double *d_level_E_eV;     /* [n_levels] */
    int    *d_level_g;        /* [n_levels] */
    int    *d_level_metastable;/* [n_levels] */
    int    *d_level_to_ip;    /* [n_levels] -> ion_pop index */
    int    *d_level_stage;    /* [n_levels] */
    /* Host staging */
    float  *h_chi_bf;         /* pinned/pageable [n_freq_bins x n_shells] */
    cublasHandle_t cublas_handle;
} BFGemmState;

static BFGemmState g_bf_gemm = {0};

__global__ void bf_compute_n_level_kernel(
    float *n_level,                                /* col-major [n_levels x n_shells] */
    const double *T_rad, const double *W,
    const double *n_ion, const double *Z_part,     /* row-major [n_ion_pops x n_shells] */
    const double *level_E_eV,
    const int *level_g, const int *level_metastable,
    const int *level_to_ip, const int *level_stage,
    int n_shells, int n_levels)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (l >= n_levels || s >= n_shells) return;

    int stage = level_stage[l];
    /* Match CPU compute_bf_opacity: skip neutrals (stage<1) for bf ionization. */
    if (stage < 1) { n_level[s * n_levels + l] = 0.0f; return; }

    int ip = level_to_ip[l];
    double T_rad_s = T_rad[s];
    double W_s     = W[s];
    double n_ion_s = n_ion[ip * n_shells + s];
    double Z_part_s= Z_part[ip * n_shells + s];

    if (n_ion_s < 1e-30 || Z_part_s < 1e-300 || T_rad_s <= 0.0) {
        n_level[s * n_levels + l] = 0.0f; return;
    }

    double E_eV = level_E_eV[l];
    int g = level_g[l];
    int is_meta = level_metastable[l];

    double beta_rad = 1.0 / (K_BOLTZMANN * T_rad_s);
    double boltz = E_eV * EV_TO_ERG * beta_rad;
    if (boltz > 50.0) { n_level[s * n_levels + l] = 0.0f; return; }

    double weight = is_meta ? 1.0 : W_s;
    double n_lvl = n_ion_s * weight * (double)g * exp(-boltz) / Z_part_s;

    /* col-major store: n_level[l + s*n_levels] */
    n_level[s * n_levels + l] = (float)n_lvl;
}

extern "C" int bf_gemm_init(AtomicData *atom, int n_shells)
{
    if (g_bf_gemm.initialized) return 0;
    if (!atom->cmfgen_loaded) return -1;

    int n_levels = atom->n_levels;
    int n_freq   = atom->cmfgen_n_freq_bins;
    int n_ip     = atom->n_ion_pops;

    g_bf_gemm.n_levels    = n_levels;
    g_bf_gemm.n_freq_bins = n_freq;
    g_bf_gemm.n_shells    = n_shells;
    g_bf_gemm.n_ion_pops  = n_ip;

    /* Pack sigma_bf as col-major [n_freq x n_levels] (transpose of row-major
     * [n_levels x n_freq]). Same byte order, different stride convention. */
    size_t sig_bytes = (size_t)n_levels * (size_t)n_freq * sizeof(float);
    float *h_sigma_T = (float *)malloc(sig_bytes);
    for (int l = 0; l < n_levels; l++)
        for (int f = 0; f < n_freq; f++)
            h_sigma_T[l * n_freq + f] = (float)atom->cmfgen_sigma_bf[(size_t)l * n_freq + f];
    /* Note: above is the SAME row-major byte layout. cuBLAS interprets
     * d_sigma_bf as col-major [n_freq x n_levels] with lda=n_freq,
     * which is the transpose view we want. */

    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_sigma_bf, sig_bytes));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_sigma_bf, h_sigma_T, sig_bytes,
                          cudaMemcpyHostToDevice));
    free(h_sigma_T);

    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_n_level,
                          (size_t)n_levels * n_shells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_chi_bf,
                          (size_t)n_freq * n_shells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_T_rad, n_shells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_W, n_shells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_n_ion,
                          (size_t)n_ip * n_shells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_Z_part,
                          (size_t)n_ip * n_shells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_level_E_eV, n_levels * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_level_g, n_levels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_level_metastable, n_levels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_level_to_ip, n_levels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_bf_gemm.d_level_stage, n_levels * sizeof(int)));

    /* Build level_to_ip and level_stage on host, upload */
    int *h_level_to_ip = (int *)malloc(n_levels * sizeof(int));
    int *h_level_stage = (int *)malloc(n_levels * sizeof(int));
    for (int ip = 0; ip < n_ip; ip++) {
        int s = atom->level_offset[ip];
        int e = atom->level_offset[ip + 1];
        for (int l = s; l < e; l++) {
            h_level_to_ip[l] = ip;
            h_level_stage[l] = atom->ion_pop_stage[ip];
        }
    }
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_level_to_ip, h_level_to_ip,
                          n_levels * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_level_stage, h_level_stage,
                          n_levels * sizeof(int), cudaMemcpyHostToDevice));
    free(h_level_to_ip); free(h_level_stage);

    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_level_E_eV, atom->level_energy_eV,
                          n_levels * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_level_g, atom->level_g,
                          n_levels * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_level_metastable, atom->level_metastable,
                          n_levels * sizeof(int), cudaMemcpyHostToDevice));

    g_bf_gemm.h_chi_bf = (float *)malloc((size_t)n_freq * n_shells * sizeof(float));

    cublasCreate(&g_bf_gemm.cublas_handle);
    /* Engage TF32 tensor cores on Ampere+ for FP32 GEMMs */
    cublasSetMathMode(g_bf_gemm.cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    g_bf_gemm.initialized = 1;
    size_t mb = (sig_bytes
                 + (size_t)n_levels * n_shells * sizeof(float)
                 + (size_t)n_freq * n_shells * sizeof(float)) / (1024 * 1024);
    printf("  [BF-GEMM] init: %d levels x %d freq x %d shells, %zu MB GPU mem (TF32)\n",
           n_levels, n_freq, n_shells, mb);
    return 0;
}

extern "C" int bf_gemm_compute(BFOpacity *bf, AtomicData *atom,
                               PlasmaState *plasma, int n_shells)
{
    if (!g_bf_gemm.initialized) {
        if (bf_gemm_init(atom, n_shells) != 0) return -1;
    }
    if (bf->n_freq_bins != g_bf_gemm.n_freq_bins ||
        n_shells       != g_bf_gemm.n_shells) {
        fprintf(stderr, "[BF-GEMM] dimension mismatch\n");
        return -1;
    }

    int n_levels = g_bf_gemm.n_levels;
    int n_freq   = g_bf_gemm.n_freq_bins;
    int n_ip     = g_bf_gemm.n_ion_pops;

    /* Upload current plasma state */
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_T_rad, plasma->T_rad,
                          n_shells * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_W, plasma->W,
                          n_shells * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_n_ion, atom->ion_number_density,
                          (size_t)n_ip * n_shells * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_Z_part, atom->partition_functions,
                          (size_t)n_ip * n_shells * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* Step 1: launch n_level kernel */
    dim3 block(64, 4);
    dim3 grid((n_levels + block.x - 1) / block.x,
              (n_shells + block.y - 1) / block.y);
    bf_compute_n_level_kernel<<<grid, block>>>(
        g_bf_gemm.d_n_level,
        g_bf_gemm.d_T_rad, g_bf_gemm.d_W,
        g_bf_gemm.d_n_ion, g_bf_gemm.d_Z_part,
        g_bf_gemm.d_level_E_eV,
        g_bf_gemm.d_level_g, g_bf_gemm.d_level_metastable,
        g_bf_gemm.d_level_to_ip, g_bf_gemm.d_level_stage,
        n_shells, n_levels);
    CUDA_CHECK(cudaGetLastError());

    /* Step 2: cuBLAS GEMM (TF32 tensor cores)
     *   col-major view: A=sigma_bf [n_freq x n_levels], B=n_level [n_levels x n_shells]
     *   C = alpha * A @ B + beta * C, shape [n_freq x n_shells]
     *   Row-major equivalent: chi_bf[s,f] = sum_l n_level[s,l] * sigma_bf[l,f] */
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t stat = cublasGemmEx(
        g_bf_gemm.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n_freq, n_shells, n_levels,
        &alpha,
        g_bf_gemm.d_sigma_bf, CUDA_R_32F, n_freq,
        g_bf_gemm.d_n_level,  CUDA_R_32F, n_levels,
        &beta,
        g_bf_gemm.d_chi_bf,   CUDA_R_32F, n_freq,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[BF-GEMM] cublasGemmEx failed: %d\n", stat);
        return -1;
    }

    /* Step 3: copy chi_bf back, promote FP32 -> FP64 into bf->chi_bf */
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.h_chi_bf, g_bf_gemm.d_chi_bf,
                          (size_t)n_freq * n_shells * sizeof(float),
                          cudaMemcpyDeviceToHost));
    /* col-major [n_freq x n_shells] -> row-major [n_shells x n_freq] */
    for (int s = 0; s < n_shells; s++)
        for (int f = 0; f < n_freq; f++)
            bf->chi_bf[s * n_freq + f] = (double)g_bf_gemm.h_chi_bf[s * n_freq + f];

    /* Activation level: GEMM mode loses per-level dominance info. Set all to
     * -1 (thermal fallback for macro-atom on bf events). A future kernel can
     * argmax over levels to recover dominant absorber identity. */
    memset(bf->activation_level, -1,
           (size_t)n_shells * n_freq * sizeof(int));

    return 0;
}

extern "C" void bf_gemm_free(void)
{
    if (!g_bf_gemm.initialized) return;
    if (g_bf_gemm.cublas_handle) cublasDestroy(g_bf_gemm.cublas_handle);
    cudaFree(g_bf_gemm.d_sigma_bf);
    cudaFree(g_bf_gemm.d_n_level);
    cudaFree(g_bf_gemm.d_chi_bf);
    cudaFree(g_bf_gemm.d_T_rad);
    cudaFree(g_bf_gemm.d_W);
    cudaFree(g_bf_gemm.d_n_ion);
    cudaFree(g_bf_gemm.d_Z_part);
    cudaFree(g_bf_gemm.d_level_E_eV);
    cudaFree(g_bf_gemm.d_level_g);
    cudaFree(g_bf_gemm.d_level_metastable);
    cudaFree(g_bf_gemm.d_level_to_ip);
    cudaFree(g_bf_gemm.d_level_stage);
    free(g_bf_gemm.h_chi_bf);
    memset(&g_bf_gemm, 0, sizeof(g_bf_gemm));
}
