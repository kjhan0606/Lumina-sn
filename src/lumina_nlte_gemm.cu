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
#include "gpu_radiation_field.h"
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

    float  *h_R_bf_f32;    /* [L_phot * n_shells] host download buffer */
    double *h_R_bf;        /* [L_phot * n_shells] FP32→FP64 promoted */

    /* per-K-column metadata for the fine-ν correction (filled in init) */
    int    *col_glev;      /* [L_phot] global level idx per K col (−1 = Kramers σ) */
    double *col_nu_th;     /* [L_phot] photoion threshold ν per K column */
    double *col_sig0;      /* [L_phot] Kramers σ_0 (used when col_glev<0); −1 if cmfgen */
    double  nu_min_b, dlognu_b;  /* binned log grid (for in-window bin range) */

    cublasHandle_t handle;
} NLTERatesGemmState;

static NLTERatesGemmState g_nlte_gemm = {0};

__global__ static void canonical_jnu_to_float(
    GpuRadiationFieldDeviceView view, float *out)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = view.n_shells * view.n_bins;
    if (i >= n) return;
    RadiationFieldValidityState state = view.field_validity[i];
    out[i] = state == RADIATION_FIELD_EXACT_ZERO ? 0.0f
           : state == RADIATION_FIELD_VALID ? (float)view.J_nu[i] : NAN;
}

static double legacy_step_sigma(int bb, double nu_min, double dlog,
    const double *sigma_row, double sigma0, double threshold)
{
    double lo = exp(log(nu_min) + bb * dlog);
    double hi = exp(log(nu_min) + (bb + 1) * dlog);
    if (sigma_row) {
        double s = sigma_row[bb];
        if (!(s > 0.0) || !isfinite(s)) return 0.0;
        if (lo < threshold && hi > threshold)
            s *= (hi - lo) / (hi - threshold);
        return s;
    }
    if (lo >= threshold)
        return sigma0 * pow(threshold / sqrt(lo * hi), 3.0);
    if (hi > threshold) {
        double x = threshold / hi;
        return sigma0 * (1.0 - x * x * x) /
               (3.0 * log(hi / threshold));
    }
    return 0.0;
}

static double canonical_bf_kernel(double edge_lo, double edge_hi, int nfb,
    double nu_min, double dlog, const double *sigma_row, double sigma0,
    double threshold)
{
    /* CMFGEN rows contain full-bin averages.  legacy_step_sigma relocates the
     * first-bin mass onto its active support, so both stored and Kramers lanes
     * use the same explicit physical-edge clamp here. */
    double lo = edge_lo > threshold ? edge_lo : threshold;
    if (!(edge_hi > lo)) return 0.0;
    int first = (int)floor(log(lo / nu_min) / dlog);
    int last = (int)floor(log(nextafter(edge_hi, lo) / nu_min) / dlog);
    if (first < 0) first = 0;
    if (last >= nfb) last = nfb - 1;
    if (last < first) return 0.0;
    double sum = 0.0;
    for (int bb = first; bb <= last; ++bb) {
        double blo = exp(log(nu_min) + bb * dlog);
        double bhi = exp(log(nu_min) + (bb + 1) * dlog);
        double a = lo > blo ? lo : blo;
        double b = edge_hi < bhi ? edge_hi : bhi;
        double s = legacy_step_sigma(bb, nu_min, dlog, sigma_row,
                                     sigma0, threshold);
        if (b > a && s > 0.0) sum += s * log(b / a);
    }
    return sum * 4.0 * M_PI_VAL / H_PLANCK;
}

/* Fine-ν local field registered by the producer (cmfgen_fine_jbar) so the binned
 * R_bf GEMM can be corrected over the fine window (LUMINA_CMF_FINE_PHOTOION).
 * NULL → no correction (binned only). */
static const double *g_fgemm_jnu = NULL, *g_fgemm_nu = NULL;
static int g_fgemm_nf = 0, g_fgemm_ns = 0;
static double g_fgemm_nulo = 0.0, g_fgemm_dlognu = 0.0;
static AtomicData *g_fgemm_atom = NULL;
extern "C" void nlte_rates_gpu_set_fine(const double *jnu, const double *nu,
        int n_fine, double nu_lo, double dlognu, int n_shells, AtomicData *atom) {
    g_fgemm_jnu = jnu; g_fgemm_nu = nu; g_fgemm_nf = n_fine;
    g_fgemm_nulo = nu_lo; g_fgemm_dlognu = dlognu; g_fgemm_ns = n_shells;
    g_fgemm_atom = atom;
}

extern "C" int nlte_rates_gpu_init(NLTEConfig *nlte, AtomicData *atom, int n_shells)
{
    if (g_nlte_gemm.initialized) return 0;
    if (!nlte || !nlte->enabled || n_shells <= 0) return -1;

    if (nlte->radfield_view_status != RADIATION_FIELD_VIEW_OK ||
        !nlte->radfield_view.frequency_bin_edges ||
        nlte->radfield_view.n_bins != LUMINA_RADFIELD_N_BINS)
        return -1;
    int legacy_n_freq = nlte->n_freq_bins;
    int n_freq = (int)nlte->radfield_view.n_bins;
    /* Pair (lo,hi)+names from the centralized builder (single source of truth;
     * mirrors the GPU/CPU solve exactly). 16 base pairs, or 23 with the O triplet
     * + stage-IV (III,IV) pairs under LUMINA_NLTE_STAGE4. Only lo is needed here:
     * the R_bf table dimensions on the lower ion's photoionizing levels. */
    int gemm_pairs[NLTE_PAIR_COUNT][2];
    const char *gemm_names[NLTE_PAIR_COUNT];
    int n_pairs = nlte_get_pairs(gemm_pairs, gemm_names);

    /* Pass 1: count L_phot and build phot_offset[] */
    int *phot_offset = (int *)malloc((n_pairs + 1) * sizeof(int));
    phot_offset[0] = 0;
    for (int p = 0; p < n_pairs; p++) {
        int lo = gemm_pairs[p][0];
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
    g_nlte_gemm.nu_min_b  = nlte->nu_min;
    g_nlte_gemm.dlognu_b  = nlte->d_log_nu;
    g_nlte_gemm.col_glev  = (int *)malloc((size_t)L_phot * sizeof(int));
    g_nlte_gemm.col_nu_th = (double *)malloc((size_t)L_phot * sizeof(double));
    g_nlte_gemm.col_sig0  = (double *)malloc((size_t)L_phot * sizeof(double));
    for (int c = 0; c < L_phot; c++) { g_nlte_gemm.col_glev[c] = -1;
                                       g_nlte_gemm.col_nu_th[c] = 0.0;
                                       g_nlte_gemm.col_sig0[c] = -1.0; }

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
    for (int bb = 0; bb < n_freq; bb++) {
        double lo = nlte->radfield_view.frequency_bin_edges[bb];
        double hi = nlte->radfield_view.frequency_bin_edges[bb + 1];
        nu_bin[bb] = sqrt(lo * hi);
    }

    /* Task #138: per-level σ_bf source. CMFGEN tabulated grid shares the
     * NLTE J_ν grid (NLTE_N_FREQ_BINS, log-spaced); direct index, no interp. */
    const int use_cmfgen = atom->cmfgen_loaded &&
                           atom->cmfgen_n_freq_bins == legacy_n_freq;
    int n_active_levels = 0, n_cmfgen_levels = 0, n_kramers_levels = 0;

    /* FINE-PHOTOION reach diagnostic (LUMINA_CMF_FINE_PHOTOION_DIAG): per ion stage,
     * what fraction of each edge's photoion kernel weight Σσ_bf·4π/(hν)·Δν lies in the
     * fine window [LAMLO,LAMHI] Å — i.e. how much of the rate a fine-ν field would
     * resolve. If the high-ion edges (stage≥2, which ionize the thin outer) carry
     * negligible in-window weight, the fine-photoion approach cannot help and a full
     * tiled-GEMM build is not worth it. Read-only; no effect on K. */
    int fph_diag = 0;
    { const char *e=getenv("LUMINA_CMF_FINE_PHOTOION_DIAG"); if(e) fph_diag=atoi(e); }
    double fph_lamlo=228.0, fph_lamhi=4000.0;
    { const char *e=getenv("LUMINA_CMF_FINE_LAMLO"); if(e) fph_lamlo=atof(e);
      const char *h=getenv("LUMINA_CMF_FINE_LAMHI"); if(h) fph_lamhi=atof(h); }
    double fph_nu_hi = 2.99792458e10/(fph_lamlo*1e-8);   /* blue edge of window [Hz] */
    double fph_nu_lo = 2.99792458e10/(fph_lamhi*1e-8);   /* red edge */
    long fph_cnt[8]={0}; double fph_fsum[8]={0}; long fph_cmf[8]={0};
    for (int p = 0; p < n_pairs; p++) {
        int ion_idx_lo = gemm_pairs[p][0];
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
                &atom->cmfgen_sigma_bf[(size_t)global_lev *
                                        (size_t)legacy_n_freq] : NULL;

            int idx = phot_offset[p] + lev;
            g_nlte_gemm.col_glev[idx]  = level_has_cmfgen ? global_lev : -1;
            g_nlte_gemm.col_nu_th[idx] = nu_thresh;
            g_nlte_gemm.col_sig0[idx]  = level_has_cmfgen ? -1.0 : sigma_0;
            float *K_col = h_K + (size_t)idx * n_freq;
            for (int bb = 0; bb < n_freq; bb++)
                K_col[bb] = (float)canonical_bf_kernel(
                    nlte->radfield_view.frequency_bin_edges[bb],
                    nlte->radfield_view.frequency_bin_edges[bb + 1],
                    legacy_n_freq, nlte->nu_min, nlte->d_log_nu, sigma_row,
                    sigma_0, nu_thresh);
            if (fph_diag) {   /* in-window photoion-weight fraction for this edge */
                double wtot=0.0, win=0.0;
                for (int bb=0; bb<n_freq; bb++) {
                    double k=K_col[bb]; if (k<=0.0) continue;
                    wtot += k;
                    if (nu_bin[bb] >= fph_nu_lo && nu_bin[bb] <= fph_nu_hi) win += k;
                }
                if (wtot > 0.0) {
                    int st = ion_lo; if (st<0) st=0; if (st>7) st=7;
                    fph_cnt[st]++; fph_fsum[st]+= win/wtot;
                    if (level_has_cmfgen) fph_cmf[st]++;
                }
            }
            if (level_has_cmfgen) n_cmfgen_levels++;
            else                  n_kramers_levels++;
            n_active_levels++;
        }
    }
    if (fph_diag) {
        fprintf(stderr, "[FINE-PHOTOION-DIAG] window %.0f-%.0f A: per-ion-stage "
                "edges and mean in-window photoion-weight fraction\n",
                fph_lamlo, fph_lamhi);
        for (int st=0; st<8; st++) if (fph_cnt[st]>0)
            fprintf(stderr, "  ion stage %d (%s): %ld edges (%ld cmfgen-sigma), "
                    "mean in-window weight frac = %.3f\n", st,
                    st==0?"neutral":st==1?"I+":st==2?"II+":st>=3?"III+":"",
                    fph_cnt[st], fph_cmf[st], fph_fsum[st]/(double)fph_cnt[st]);
    }
    free(nu_bin);

    /* Allocate device + host buffers */
    CUDA_CHECK(cudaMalloc(&g_nlte_gemm.d_K, K_bytes));
    CUDA_CHECK(cudaMemcpy(g_nlte_gemm.d_K, h_K, K_bytes, cudaMemcpyHostToDevice));
    free(h_K);

    CUDA_CHECK(cudaMalloc(&g_nlte_gemm.d_J_nu,
                          (size_t)n_freq * n_shells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_nlte_gemm.d_R_bf,
                          (size_t)L_phot * n_shells * sizeof(float)));

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

/* Correct d_R_bf over the fine-ν window: subtract the coarse in-window contribution
 * (offset GEMM over the contiguous in-window binned bins) and add the fine-grid one
 * (tiled K_fine^T·J_fine). Frequency-tiled + memory-aware; on ANY cuBLAS/CUDA/alloc
 * failure it leaves R_bf as the binned result (graceful fallback — the fine field is
 * an enhancement). Returns 1 if it ran, 0 if skipped (no fine field / nothing in
 * window / fallback). This is the deterministic analog of ARTIS's exact-frequency bf
 * estimator: the photoion rate is integrated against the frequency-RESOLVED field so
 * the hard UV that the coarse bins average away reaches the thin outer. */
static int fine_correct_R_bf(void)
{
    if (!g_fgemm_jnu || g_fgemm_nf < 2 || !g_fgemm_atom) return 0;
    int L = g_nlte_gemm.L_phot, S = g_nlte_gemm.n_shells, NFb = g_nlte_gemm.n_freq;
    int NF = g_fgemm_nf;
    if (g_fgemm_ns != S || !g_nlte_gemm.col_glev) return 0;
    const double *nuf = g_fgemm_nu, *jf = g_fgemm_jnu;
    double nu_lo_w = nuf[0], nu_hi_w = nuf[NF - 1];

    int fine_cols = 0;
    for (int c = 0; c < L; c++)
        if ((g_nlte_gemm.col_glev[c] >= 0 || g_nlte_gemm.col_sig0[c] > 0.0)
            && g_nlte_gemm.col_nu_th[c] < nu_hi_w) fine_cols++;
    fprintf(stderr, "[FINE-GEMM] fine photoion correction: %d/%d cols in window "
            "%.0f-%.0f A (NF=%d)\n", fine_cols, L,
            2.99792458e18/nu_hi_w, 2.99792458e18/nu_lo_w, NF);
    if (fine_cols == 0) return 0;

    /* --- A. subtract the coarse in-window contribution (contiguous bin range) --- */
    double lnmin = log(g_nlte_gemm.nu_min_b), dln = g_nlte_gemm.dlognu_b;
    int bb_lo = (int)ceil ((log(nu_lo_w) - lnmin) / dln - 0.5);
    int bb_hi = (int)floor((log(nu_hi_w) - lnmin) / dln - 0.5);
    if (bb_lo < 0) bb_lo = 0;
    if (bb_hi > NFb - 1) bb_hi = NFb - 1;
    if (bb_hi >= bb_lo) {
        int kw = bb_hi - bb_lo + 1;
        float am = -1.0f, bt = 1.0f;
        cublasStatus_t st = cublasGemmEx(g_nlte_gemm.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            L, S, kw, &am,
            g_nlte_gemm.d_K   + bb_lo, CUDA_R_32F, NFb,
            g_nlte_gemm.d_J_nu+ bb_lo, CUDA_R_32F, NFb,
            &bt, g_nlte_gemm.d_R_bf,   CUDA_R_32F, L,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[FINE-GEMM] subtract GEMM failed %d -> keep binned\n", st);
            return 0;
        }
    }

    /* --- B. add the fine in-window contribution, frequency-tiled --- */
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    size_t per_col = ((size_t)L + S) * sizeof(float);
    long tile = (per_col > 0) ? (long)((freeb / 2) / per_col) : 4096;
    if (tile > NF) tile = NF;
    if (tile > 65536) tile = 65536;       /* cap host K-build cost per tile */
    if (tile < 256)  tile = 256;
    float *h_Kt = (float *)malloc((size_t)tile * L * sizeof(float));
    float *h_Jt = (float *)malloc((size_t)tile * S * sizeof(float));
    float *d_Kt = NULL, *d_Jt = NULL;
    if (!h_Kt || !h_Jt ||
        cudaMalloc(&d_Kt, (size_t)tile * L * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_Jt, (size_t)tile * S * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[FINE-GEMM] tile alloc failed (tile=%ld) -> keep binned "
                "(note: subtract already applied; re-add coarse)\n", tile);
        /* re-add the coarse in-window we subtracted, to stay consistent */
        if (bb_hi >= bb_lo) { int kw=bb_hi-bb_lo+1; float a=1.0f,b=1.0f;
            cublasGemmEx(g_nlte_gemm.handle,CUBLAS_OP_T,CUBLAS_OP_N,L,S,kw,&a,
                g_nlte_gemm.d_K+bb_lo,CUDA_R_32F,NFb, g_nlte_gemm.d_J_nu+bb_lo,CUDA_R_32F,NFb,
                &b,g_nlte_gemm.d_R_bf,CUDA_R_32F,L,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT); }
        free(h_Kt); free(h_Jt); if (d_Kt) cudaFree(d_Kt); if (d_Jt) cudaFree(d_Jt);
        return 0;
    }
    const double hpl = 6.62607015e-27;
    AtomicData *atom = g_fgemm_atom;
    for (long t0 = 0; t0 < NF; t0 += tile) {
        long tn = (t0 + tile <= NF) ? tile : (NF - t0);
        memset(h_Kt, 0, (size_t)tn * L * sizeof(float));
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
        #endif
        for (int c = 0; c < L; c++) {
            int gl = g_nlte_gemm.col_glev[c];
            double thr = g_nlte_gemm.col_nu_th[c]; if (thr <= 0.0) continue;
            double s0 = g_nlte_gemm.col_sig0[c];
            if (gl < 0 && s0 <= 0.0) continue;          /* inactive column */
            const double *srow = (gl >= 0) ?
                &atom->cmfgen_sigma_bf[(size_t)gl * NFb] : NULL;
            for (long it = 0; it < tn; it++) {
                double nu = nuf[t0 + it]; if (nu < thr) continue;
                double sig;
                if (srow) {  /* CMFGEN tabulated σ_bf, nearest coarse bin */
                    int bb = (int)((log(nu) - lnmin) / dln);
                    if (bb < 0 || bb >= NFb) continue;
                    sig = srow[bb]; if (sig <= 0.0) continue;
                } else {     /* Kramers fallback σ_0·(ν_thr/ν)^3 — matches binned K-build */
                    sig = s0 * (thr/nu) * (thr/nu) * (thr/nu);
                }
                double dnu = nu * g_fgemm_dlognu;
                h_Kt[(size_t)c * tn + it] = (float)(sig * 4.0 * M_PI_VAL / (hpl * nu) * dnu);
            }
        }
        for (long it = 0; it < tn; it++)
            for (int s = 0; s < S; s++)
                h_Jt[(size_t)s * tn + it] = (float)jf[(size_t)s * NF + (t0 + it)];
        if (cudaMemcpy(d_Kt, h_Kt, (size_t)tn * L * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_Jt, h_Jt, (size_t)tn * S * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "[FINE-GEMM] tile copy failed -> abort fine add\n"); break;
        }
        float a1 = 1.0f, b1 = 1.0f;
        cublasGemmEx(g_nlte_gemm.handle, CUBLAS_OP_T, CUBLAS_OP_N, L, S, (int)tn, &a1,
            d_Kt, CUDA_R_32F, (int)tn, d_Jt, CUDA_R_32F, (int)tn, &b1,
            g_nlte_gemm.d_R_bf, CUDA_R_32F, L,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    }
    cudaDeviceSynchronize();
    free(h_Kt); free(h_Jt); cudaFree(d_Kt); cudaFree(d_Jt);
    return 1;
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

    GpuRadiationFieldDeviceView view;
    GpuRadiationFieldReport report;
    if (gpu_radiation_field_production_view(&nlte->radiation_field, &view,
            &report) != GPU_RF_OK || view.n_bins != (size_t)n_freq ||
        view.n_shells != (size_t)n_shells)
        return -(int)GPU_RATE_NOT_MIGRATED;
    size_t cells = view.n_bins * view.n_shells;
    canonical_jnu_to_float<<<(unsigned)((cells + 255) / 256), 256>>>(
        view, g_nlte_gemm.d_J_nu);
    if (cudaGetLastError() != cudaSuccess) return -1;

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

    /* A2-13 forbids the diagnostic fine grid as a production rate input. */

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
    free(g_nlte_gemm.h_R_bf_f32);
    free(g_nlte_gemm.h_R_bf);
    free(g_nlte_gemm.col_glev);
    free(g_nlte_gemm.col_nu_th);
    free(g_nlte_gemm.col_sig0);
    memset(&g_nlte_gemm, 0, sizeof(g_nlte_gemm));
}
