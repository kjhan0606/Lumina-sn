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
#include "gpu_radiation_field_contract.h"
}

static GpuRadiationFieldCounters g_a2_12_bf_counters;

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
    int n_shells, int n_levels, int include_neutrals)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (l >= n_levels || s >= n_shells) return;

    int stage = level_stage[l];
    /* [Wave-1 neutral-bf] Zero charge excludes free-free, not photoionization:
     * X I + hnu -> X II + e is physical. Preserve the historical zero exactly
     * unless LUMINA_FIX_BF_NEUTRAL is armed on the host. */
    if (stage < 1 && !include_neutrals) {
        n_level[s * n_levels + l] = 0.0f; return;
    }

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
    /* A2-12 GL01/GL02: no independent scalar owners survive. */
    g_bf_gemm.d_T_rad = NULL;
    g_bf_gemm.d_W = NULL;
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
    if (gpu_rf_block_unmigrated(&g_a2_12_bf_counters,
            GPU_OPACITY_NOT_MIGRATED, "lumina_bf_gemm.compute") != 0)
        return -(int)GPU_OPACITY_NOT_MIGRATED;
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
        n_shells, n_levels, lumina_fix_bf_neutral_enabled());
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

/* Fine-ν bf opacity: chi_bf_fine[s,i] = Σ_l n_level[s,l]·σ_bf,l(ν_i) with the bf EDGES
 * resolved at the exact photoion threshold on the fine grid (sharp onset at thr_l +
 * nearest-bin σ magnitude). This is the CMFGEN-method fix: it makes the producer's fine
 * continuum field develop the across-edge frequency structure that the binned grid (and
 * the log-ν interpolation in cmfgen_fine_jbar) averages away. Same dilute-LTE n_level and
 * TF32 GEMM as bf_gemm_compute, just on the fine ν grid, frequency-TILED + memory-aware.
 * chi_bf_fine_out is [n_shells * n_fine] row-major. Returns 0 on success, -1 on failure
 * (caller keeps the interpolated continuum). */
extern "C" int bf_gemm_compute_fine(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
        int n_shells, const double *nu_fine, int n_fine,
        double nu_min_bin, double dlognu_bin, double *chi_bf_fine_out)
{
    if (gpu_rf_block_unmigrated(&g_a2_12_bf_counters,
            GPU_OPACITY_NOT_MIGRATED, "lumina_bf_gemm.compute_fine") != 0)
        return -(int)GPU_OPACITY_NOT_MIGRATED;
    (void)bf;
    /* D-3's level/frequency-dependent stimulated-recombination corrfactor is
     * not a separable GEMM. Return to the caller's corrected coarse-grid
     * interpolation rather than silently replacing it with uncorrected fine chi. */
    if (lumina_fix_bf_stim_recomb_enabled()) return -1;
    if (!g_bf_gemm.initialized) { if (bf_gemm_init(atom, n_shells) != 0) return -1; }
    if (n_shells != g_bf_gemm.n_shells || !atom->cmfgen_sigma_bf || n_fine < 2) return -1;
    int n_levels = g_bf_gemm.n_levels;
    int n_freq   = g_bf_gemm.n_freq_bins;
    int n_ip     = g_bf_gemm.n_ion_pops;

    /* recompute dilute-LTE n_level for the current plasma (== bf_gemm_compute step 1) */
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_T_rad, plasma->T_rad, n_shells*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_W, plasma->W, n_shells*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_n_ion, atom->ion_number_density, (size_t)n_ip*n_shells*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_bf_gemm.d_Z_part, atom->partition_functions, (size_t)n_ip*n_shells*sizeof(double), cudaMemcpyHostToDevice));
    {
        dim3 block(64,4);
        dim3 grid((n_levels+block.x-1)/block.x, (n_shells+block.y-1)/block.y);
        bf_compute_n_level_kernel<<<grid,block>>>(g_bf_gemm.d_n_level, g_bf_gemm.d_T_rad,
            g_bf_gemm.d_W, g_bf_gemm.d_n_ion, g_bf_gemm.d_Z_part, g_bf_gemm.d_level_E_eV,
            g_bf_gemm.d_level_g, g_bf_gemm.d_level_metastable, g_bf_gemm.d_level_to_ip,
            g_bf_gemm.d_level_stage, n_shells, n_levels,
            lumina_fix_bf_neutral_enabled());
        CUDA_CHECK(cudaGetLastError());
    }

    /* per-level photoion threshold nu=(chi_ion-E_level)/h. Stage 0 is retained
     * under the same neutral-bf repair gate as the population kernel. */
    static double *thr = NULL; static int thr_n = 0;
    if (!thr || thr_n != n_levels) {
        free(thr); thr = (double*)malloc((size_t)n_levels*sizeof(double)); thr_n = n_levels;
        for (int l = 0; l < n_levels; l++) {
            thr[l] = -1.0;
            int ip = -1;
            for (int p = 0; p < n_ip; p++)
                if (l >= atom->level_offset[p] && l < atom->level_offset[p+1]) { ip = p; break; }
            if (ip < 0 ||
                (atom->ion_pop_stage[ip] < 1 &&
                 !lumina_fix_bf_neutral_enabled())) continue;
            int Z = atom->ion_pop_Z[ip], st = atom->ion_pop_stage[ip];
            double chi_eV = -1.0;
            for (int k = 0; k < atom->n_ionization; k++)
                if (atom->ioniz_Z[k]==Z && atom->ioniz_ion[k]==st) { chi_eV = atom->ioniz_energy_eV[k]; break; }
            if (chi_eV <= 0.0) continue;
            double e_th = (chi_eV - atom->level_energy_eV[l]) * EV_TO_ERG;
            if (e_th > 0.0) thr[l] = e_th / H_PLANCK;
        }
    }

    /* frequency-tiled GEMM: C[tn×n_shells] = σ_fine[tn×n_levels] · n_level[n_levels×n_shells] */
    double lnmin = log(nu_min_bin);
    size_t freeb=0, totb=0; cudaMemGetInfo(&freeb, &totb);
    long tile = (long)((freeb/2) / ((size_t)(n_levels + n_shells) * sizeof(float)));
    if (tile > n_fine) tile = n_fine;
    if (tile > 16384) tile = 16384;
    if (tile < 256)   tile = 256;
    float *h_sig = (float*)malloc((size_t)tile*n_levels*sizeof(float));
    float *h_chi = (float*)malloc((size_t)tile*n_shells*sizeof(float));
    float *d_sig = NULL, *d_chi = NULL;
    if (!h_sig || !h_chi ||
        cudaMalloc(&d_sig, (size_t)tile*n_levels*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_chi, (size_t)tile*n_shells*sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[BF-GEMM-FINE] alloc failed (tile=%ld) -> keep interpolated\n", tile);
        free(h_sig); free(h_chi); if (d_sig) cudaFree(d_sig); if (d_chi) cudaFree(d_chi);
        return -1;
    }
    const double *sig_tab = atom->cmfgen_sigma_bf;   /* [n_levels × n_freq] row-major */
    for (long t0 = 0; t0 < n_fine; t0 += tile) {
        long tn = (t0+tile <= n_fine) ? tile : (n_fine - t0);
        memset(h_sig, 0, (size_t)tn*n_levels*sizeof(float));
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,64)
        #endif
        for (int l = 0; l < n_levels; l++) {
            double tl = thr[l]; if (tl <= 0.0) continue;
            const double *srow = &sig_tab[(size_t)l*n_freq];
            for (long i = 0; i < tn; i++) {
                double nu = nu_fine[t0+i]; if (nu < tl) continue;        /* sharp edge */
                int bb = (int)((log(nu)-lnmin)/dlognu_bin); if (bb<0||bb>=n_freq) continue;
                double sg = srow[bb]; if (sg <= 0.0) continue;
                h_sig[(size_t)l*tn + i] = (float)sg;     /* col-major [tn×n_levels]: (i,l)=l*tn+i */
            }
        }
        if (cudaMemcpy(d_sig, h_sig, (size_t)tn*n_levels*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "[BF-GEMM-FINE] tile copy failed -> abort\n"); break; }
        float a=1.0f, b=0.0f;
        cublasGemmEx(g_bf_gemm.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)tn, n_shells, n_levels,
            &a, d_sig, CUDA_R_32F, (int)tn, g_bf_gemm.d_n_level, CUDA_R_32F, n_levels,
            &b, d_chi, CUDA_R_32F, (int)tn, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaMemcpy(h_chi, d_chi, (size_t)tn*n_shells*sizeof(float), cudaMemcpyDeviceToHost);
        for (int s = 0; s < n_shells; s++)
            for (long i = 0; i < tn; i++)
                chi_bf_fine_out[(size_t)s*n_fine + (t0+i)] = (double)h_chi[(size_t)s*tn + i];
    }
    free(h_sig); free(h_chi); cudaFree(d_sig); cudaFree(d_chi);
    return 0;
}

extern "C" void bf_gemm_free(void)
{
    if (!g_bf_gemm.initialized) return;
    if (g_bf_gemm.cublas_handle) cublasDestroy(g_bf_gemm.cublas_handle);
    cudaFree(g_bf_gemm.d_sigma_bf);
    cudaFree(g_bf_gemm.d_n_level);
    cudaFree(g_bf_gemm.d_chi_bf);
    /* A2-12 GL03/GL04: no scalar allocation remains to free. */
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

/* [FB-MILNE C2] device sigma_bf handle for the sigma-weighted fb emission draw.
 * col-major [n_freq x n_levels]; sigma for level l is contiguous at +l*n_freq. */
extern "C" const float *bf_gemm_get_d_sigma_bf(void) {
    return g_bf_gemm.d_sigma_bf;
}
