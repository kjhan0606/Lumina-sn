/* lumina_nlte_gemm.cu — Task #40 (A)+(B) + Task #138 Heavy.1: GPU NLTE photoionization rates.
 *
 * Reformulates the per-(pair, shell, level, freq_bin) inner loop in
 * nlte_assemble_rate_matrix as a single TF32 GEMM:
 *
 *     R_bf[lev_idx, shell] = sum_bb K[bb, lev_idx] * J_nu[bb, shell]
 *
 * where K[bb, lev_idx] = sigma(lev,bb) * 4π/(h·ν_bb) * Δν_bb is pre-baked once
 * during init.
 *
 * σ_bf source (Task #138): when atom->cmfgen_loaded and atom->cmfgen_has_sigma[lev]
 * are set, the K column for that level is built from the CMFGEN-tabulated
 * σ_bf row (atom->cmfgen_sigma_bf[lev * n_freq + bb], same grid as J_ν).
 * Levels without CMFGEN coverage fall back to the Kramers shape σ_0(Z,ion) ×
 * (ν_thresh/ν)^3, with σ_0 from the Verner-CMFGEN ground-state table when
 * available, else 7.91e-18/Z_eff^2.
 *
 * Recombination R_rec = R_bf * n_star_ratio is left on CPU because n_star_ratio
 * needs per-shell T_e/n_e (cheap scalar work). The CPU path mirrors this σ
 * source selection per level in nlte_assemble_rate_matrix. */

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

typedef struct {
    int initialized;
    int n_shells;
    int n_freq;
    int n_pairs;
    int L_phot;            /* total photoionization columns = sum n_lo_levels */
    int *phot_offset;      /* [n_pairs+1] CPU offsets into K columns / R_bf rows */

    float  *d_K;           /* [n_freq * L_phot] col-major */
    float  *d_J_nu;        /* [n_freq * n_shells] col-major */
    float  *d_R_bf;        /* [L_phot * n_shells] col-major */

    float  *h_J_nu;        /* [n_freq * n_shells] host staging */
    float  *h_R_bf_f32;    /* [L_phot * n_shells] host download buffer */
    double *h_R_bf;        /* [L_phot * n_shells] FP32→FP64 promoted */

    cublasHandle_t handle;
} NLTERatesGemmState;

static NLTERatesGemmState g_nlte_gemm = {0};

/* The NLTE GPU caller currently uses fixed pair tuples (0,1),(2,3),...,
 * (28,29),(29,30) in lumina_cuda.cu. Mirror that here; if the layout changes,
 * update both. #281: pair 15 starts at slot 29 (O II), giving (O II, O III)
 * overlap on top of pair 14 (O I, O II) for full CMFGEN O triplet fidelity. */
static const int NLTE_PAIR_LO[NLTE_PAIR_COUNT] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29
};
static const int NLTE_PAIR_N    = NLTE_PAIR_COUNT;

extern "C" int nlte_rates_gpu_init(NLTEConfig *nlte, AtomicData *atom, int n_shells)
{
    if (g_nlte_gemm.initialized) return 0;
    if (!nlte || !nlte->enabled || n_shells <= 0) return -1;

    int n_freq = nlte->n_freq_bins;
    /* #281: pair count is now NLTE_PAIR_COUNT (16) due to O triplet overlap pair —
     * n_nlte_ions/2 would undercount (31/2=15). Use explicit constant. */
    int n_pairs = NLTE_PAIR_N;
    if (nlte->n_nlte_ions < NLTE_MAX_IONS) {
        /* Defensive: if slot count came out smaller than expected, fall back */
        n_pairs = nlte->n_nlte_ions / 2;
        if (n_pairs > NLTE_PAIR_N) n_pairs = NLTE_PAIR_N;
    }

    /* Pass 1: count L_phot and build phot_offset[] */
    int *phot_offset = (int *)malloc((n_pairs + 1) * sizeof(int));
    phot_offset[0] = 0;
    for (int p = 0; p < n_pairs; p++) {
        int lo = NLTE_PAIR_LO[p];
        int n_lo_levels = nlte->nlte_ion_level_offset[lo + 1] -
                          nlte->nlte_ion_level_offset[lo];
        if (n_lo_levels < 0) n_lo_levels = 0;
        phot_offset[p + 1] = phot_offset[p] + n_lo_levels;
    }
    int L_phot = phot_offset[n_pairs];
    if (L_phot <= 0) {
        free(phot_offset);
        return -1;
    }

    g_nlte_gemm.n_shells = n_shells;
    g_nlte_gemm.n_freq   = n_freq;
    g_nlte_gemm.n_pairs  = n_pairs;
    g_nlte_gemm.L_phot   = L_phot;
    g_nlte_gemm.phot_offset = phot_offset;

    /* Pass 2: build K[n_freq * L_phot] FP32 col-major.
     *
     *   K[bb, idx] = sigma_0 * (ν_thresh/ν_bb)^3 * 4π / (h·ν_bb) * Δν_bb
     *               (zero where ν_bb < ν_thresh or ν_thresh <= 0)
     *
     * Outer guard `nu_edge < nu_max && ground_hi < N` from the assembler is NOT
     * reproduced here — instead, levels in pairs that fail the guard naturally
     * yield K=0 because either every ν_bb < ν_thresh or ν_thresh <= 0.
     * Belt-and-suspenders: assembler still checks `R_bf > 0` before applying. */
    size_t K_bytes = (size_t)n_freq * L_phot * sizeof(float);
    float *h_K = (float *)calloc((size_t)n_freq * L_phot, sizeof(float));

    /* Pre-compute per-bin ν_bin (geometric mid) and Δν_bin (linear width). */
    double *nu_bin   = (double *)malloc(n_freq * sizeof(double));
    double *dnu_bin  = (double *)malloc(n_freq * sizeof(double));
    for (int bb = 0; bb < n_freq; bb++) {
        double log_lo = log(nlte->nu_min) + bb * nlte->d_log_nu;
        double lo  = exp(log_lo);
        double hi  = exp(log_lo + nlte->d_log_nu);
        nu_bin[bb]  = exp(log_lo + 0.5 * nlte->d_log_nu);
        dnu_bin[bb] = hi - lo;
    }

    /* Task #138: per-level σ_bf source. CMFGEN tabulated grid shares the
     * NLTE J_ν grid (NLTE_N_FREQ_BINS, log-spaced); direct index, no interp. */
    const int use_cmfgen = atom->cmfgen_loaded &&
                           atom->cmfgen_n_freq_bins == n_freq;
    int n_active_levels = 0, n_cmfgen_levels = 0, n_kramers_levels = 0;
    for (int p = 0; p < n_pairs; p++) {
        int ion_idx_lo = NLTE_PAIR_LO[p];
        int Z_elem = nlte->nlte_Z[ion_idx_lo];
        int ion_lo = nlte->nlte_ion[ion_idx_lo];
        double chi_eV = -1.0;
        for (int i = 0; i < atom->n_ionization; i++) {
            if (atom->ioniz_Z[i] == Z_elem && atom->ioniz_ion[i] == ion_lo) {
                chi_eV = atom->ioniz_energy_eV[i];
                break;
            }
        }
        if (chi_eV < 0.0) chi_eV = 1e10; /* impossibly high */
        double chi_erg = chi_eV * EV_TO_ERG;
        double nu_edge = chi_erg / H_PLANCK;

        /* Kramers fallback σ_0: prefer Verner-CMFGEN ground-state table, else
         * generic 7.91e-18/Z_eff^2. */
        double sigma_0 = get_bf_sigma0(Z_elem, ion_lo);
        if (sigma_0 <= 0.0) {
            int Z_eff_int = Z_elem - ion_lo;
            if (Z_eff_int < 1) Z_eff_int = 1;
            sigma_0 = 7.91e-18 / ((double)Z_eff_int * (double)Z_eff_int);
        }

        int n_lo_levels = phot_offset[p + 1] - phot_offset[p];
        int lev_start = nlte->nlte_ion_level_offset[ion_idx_lo];

        /* Outer guard: if nu_edge ≥ nu_max, the entire pair contributes 0 (K stays 0). */
        if (nu_edge <= 0.0 || nu_edge >= nlte->nu_max) continue;

        for (int lev = 0; lev < n_lo_levels; lev++) {
            int global_lev = nlte->nlte_to_global_level[lev_start + lev];
            double E_lev_erg = atom->level_energy_eV[global_lev] * EV_TO_ERG;
            double nu_thresh = (chi_erg - E_lev_erg) / H_PLANCK;
            if (nu_thresh <= 0.0) continue;

            int level_has_cmfgen = use_cmfgen &&
                                   atom->cmfgen_has_sigma[global_lev];
            const double *sigma_row = level_has_cmfgen ?
                &atom->cmfgen_sigma_bf[(size_t)global_lev * (size_t)n_freq] : NULL;

            int idx = phot_offset[p] + lev;
            float *K_col = h_K + (size_t)idx * n_freq;
            for (int bb = 0; bb < n_freq; bb++) {
                double sigma;
                if (sigma_row) {
                    sigma = sigma_row[bb];
                    if (sigma <= 0.0) continue;
                } else {
                    if (nu_bin[bb] < nu_thresh) continue;
                    sigma = sigma_0 * pow(nu_thresh / nu_bin[bb], 3.0);
                }
                double k_val = sigma * 4.0 * M_PI_VAL /
                               (H_PLANCK * nu_bin[bb]) * dnu_bin[bb];
                K_col[bb] = (float)k_val;
            }
            if (level_has_cmfgen) n_cmfgen_levels++;
            else                  n_kramers_levels++;
            n_active_levels++;
        }
    }
    free(nu_bin); free(dnu_bin);

    /* Allocate device + host buffers */
    CUDA_CHECK(cudaMalloc(&g_nlte_gemm.d_K, K_bytes));
    CUDA_CHECK(cudaMemcpy(g_nlte_gemm.d_K, h_K, K_bytes, cudaMemcpyHostToDevice));
    free(h_K);

    CUDA_CHECK(cudaMalloc(&g_nlte_gemm.d_J_nu,
                          (size_t)n_freq * n_shells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_nlte_gemm.d_R_bf,
                          (size_t)L_phot * n_shells * sizeof(float)));

    g_nlte_gemm.h_J_nu     = (float  *)malloc((size_t)n_freq * n_shells * sizeof(float));
    g_nlte_gemm.h_R_bf_f32 = (float  *)malloc((size_t)L_phot * n_shells * sizeof(float));
    g_nlte_gemm.h_R_bf     = (double *)malloc((size_t)L_phot * n_shells * sizeof(double));

    cublasCreate(&g_nlte_gemm.handle);
    cublasSetMathMode(g_nlte_gemm.handle, CUBLAS_TF32_TENSOR_OP_MATH);

    g_nlte_gemm.initialized = 1;
    size_t mb = (K_bytes
                 + (size_t)n_freq * n_shells * sizeof(float)
                 + (size_t)L_phot * n_shells * sizeof(float)) / (1024 * 1024);
    printf("  [NLTE-GEMM] init: %d pairs, %d phot levels (%d active: %d CMFGEN + %d Kramers) x %d freq x %d shells, %zu MB GPU mem (TF32)\n",
           n_pairs, L_phot, n_active_levels, n_cmfgen_levels, n_kramers_levels,
           n_freq, n_shells, mb);
    return 0;
}

/* Compute R_bf[L_phot × n_shells] = K^T · J_nu using TF32 GEMM.
 * Returns 0 on success and fills out_lookup with pointers into internal
 * persistent buffers. The buffers stay valid until nlte_rates_gpu_free(). */
extern "C" int nlte_rates_gpu_compute(NLTEConfig *nlte, NLTERateLookup *out_lookup)
{
    if (!g_nlte_gemm.initialized) return -1;
    int n_freq = g_nlte_gemm.n_freq;
    int n_shells = g_nlte_gemm.n_shells;
    int L_phot = g_nlte_gemm.L_phot;

    /* Stage J_nu as col-major [n_freq × n_shells] FP32.
     * NLTEConfig stores it row-major [n_shells × n_freq] doubles. */
    for (int bb = 0; bb < n_freq; bb++)
        for (int s = 0; s < n_shells; s++)
            g_nlte_gemm.h_J_nu[s * n_freq + bb] =
                (float)nlte->J_nu[s * n_freq + bb];
    /* Note: above lays out as col-major view automatically because
     * h_J_nu[s*n_freq + bb] is the (bb, s) element of [n_freq × n_shells]
     * col-major (lda=n_freq). */
    CUDA_CHECK(cudaMemcpy(g_nlte_gemm.d_J_nu, g_nlte_gemm.h_J_nu,
                          (size_t)n_freq * n_shells * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* GEMM: R_bf = K^T · J_nu
     *   A = K     [n_freq × L_phot]  col-major, op=T → effective [L_phot × n_freq]
     *   B = J_nu  [n_freq × n_shells] col-major, op=N
     *   C = R_bf  [L_phot × n_shells] col-major
     *   m=L_phot, n=n_shells, k=n_freq, lda=n_freq, ldb=n_freq, ldc=L_phot */
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t stat = cublasGemmEx(
        g_nlte_gemm.handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        L_phot, n_shells, n_freq,
        &alpha,
        g_nlte_gemm.d_K,    CUDA_R_32F, n_freq,
        g_nlte_gemm.d_J_nu, CUDA_R_32F, n_freq,
        &beta,
        g_nlte_gemm.d_R_bf, CUDA_R_32F, L_phot,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NLTE-GEMM] cublasGemmEx failed: %d\n", stat);
        return -1;
    }

    CUDA_CHECK(cudaMemcpy(g_nlte_gemm.h_R_bf_f32, g_nlte_gemm.d_R_bf,
                          (size_t)L_phot * n_shells * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Promote FP32 → FP64. Layout col-major: element (idx, s) at [s*L_phot + idx]. */
    size_t total = (size_t)L_phot * n_shells;
    for (size_t i = 0; i < total; i++)
        g_nlte_gemm.h_R_bf[i] = (double)g_nlte_gemm.h_R_bf_f32[i];

    if (out_lookup) {
        out_lookup->R_bf_table    = g_nlte_gemm.h_R_bf;
        out_lookup->phot_offset   = g_nlte_gemm.phot_offset;
        out_lookup->L_phot_total  = L_phot;
    }
    return 0;
}

extern "C" void nlte_rates_gpu_free(void)
{
    if (!g_nlte_gemm.initialized) return;
    if (g_nlte_gemm.handle) cublasDestroy(g_nlte_gemm.handle);
    cudaFree(g_nlte_gemm.d_K);
    cudaFree(g_nlte_gemm.d_J_nu);
    cudaFree(g_nlte_gemm.d_R_bf);
    free(g_nlte_gemm.phot_offset);
    free(g_nlte_gemm.h_J_nu);
    free(g_nlte_gemm.h_R_bf_f32);
    free(g_nlte_gemm.h_R_bf);
    memset(&g_nlte_gemm, 0, sizeof(g_nlte_gemm));
}
