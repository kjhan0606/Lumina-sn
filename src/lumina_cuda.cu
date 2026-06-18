/* lumina_cuda.cu — Phase 6: CUDA GPU Transport Kernel
 * Direct port of lumina_transport.c (CPU) to CUDA device code.
 * Every line annotated with Phase 6 - Step N for traceability.
 * Source: lumina_transport.c, lumina_main.c (CPU reference) */

#include <stdio.h>      /* Phase 6 - Step 1 */
#include <stdlib.h>     /* Phase 6 - Step 1 */
#include <string.h>     /* Phase 6 - Step 1 */
#include <math.h>       /* Phase 6 - Step 1 */
#include <stdint.h>     /* Phase 6 - Step 1 */
#include <time.h>       /* clock_gettime for NLTE profiling */
#include <cuda_runtime.h> /* Phase 6 - Step 1 */
#include <cublas_v2.h>    /* cuBLAS batched NLTE solver */

/* Phase 6 - Step 1: Include shared header for struct definitions */
extern "C" {             /* Phase 6 - Step 1 */
#include "lumina.h"      /* Phase 6 - Step 1 */
#include "lumina_cmfgen.h"  /* pure-CMFGEN parallel radiation path */
}                        /* Phase 6 - Step 1 */

/* ============================================================ */
/* cuBLAS batched NLTE solver data structure                    */
/* ============================================================ */
typedef struct {
    cublasHandle_t handle;
    double  *d_matrices;     /* [batch * max_N * max_N] device */
    double  *d_rhs;          /* [batch * max_N] device */
    double **d_Aarray;       /* [batch] device pointer array */
    double **d_Barray;       /* [batch] device pointer array */
    int     *d_pivot;        /* [batch * max_N] device */
    int     *d_info;         /* [batch] device */
    int      max_N;          /* largest matrix dim (Fe = 1362) */
    int      batch_size;     /* n_shells (30) */
    /* Host staging */
    double  *h_matrices;
    double  *h_rhs;
    double **h_Aarray;       /* device pointers, assembled on host */
    double **h_Barray;
    int     *h_info;
} CudaNLTESolver;

/* ============================================================ */
/* Phase 6 - Step 1: CUDA error checking macro                  */
/* ============================================================ */
#define CUDA_CHECK(call) do {                                    /* Phase 6 - Step 1 */ \
    cudaError_t err = (call);                                    /* Phase 6 - Step 1 */ \
    if (err != cudaSuccess) {                                    /* Phase 6 - Step 1 */ \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",            /* Phase 6 - Step 1 */ \
                __FILE__, __LINE__, cudaGetErrorString(err));    /* Phase 6 - Step 1 */ \
        exit(EXIT_FAILURE);                                      /* Phase 6 - Step 1 */ \
    }                                                            /* Phase 6 - Step 1 */ \
} while(0)                                                       /* Phase 6 - Step 1 */

/* ============================================================ */
/* Phase 6 - Step 1: Device data structure                      */
/* ============================================================ */
typedef struct {                           /* Phase 6 - Step 1 */
    /* Phase 6 - Step 1: Read-only opacity data */
    double *d_line_list_nu;                /* Phase 6 - Step 1: [n_lines] */
    double *d_tau_sobolev;                 /* Phase 6 - Step 1: [n_lines * n_shells] */
    double *d_electron_density;            /* Phase 6 - Step 1: [n_shells] */
    double *d_transition_probabilities;    /* Phase 6 - Step 1: [n_transitions * n_shells] */
    int    *d_macro_block_references;      /* Phase 6 - Step 1: [n_levels + 1] */
    int    *d_transition_type;             /* Phase 6 - Step 1: [n_transitions] */
    int    *d_destination_level_id;        /* Phase 6 - Step 1: [n_transitions] */
    int    *d_transition_line_id;          /* Phase 6 - Step 1: [n_transitions] */
    int    *d_line2macro_level_upper;      /* Phase 6 - Step 1: [n_lines] */
    double *d_p_kpacket;                    /* [n_macro_levels * n_shells] k-packet deactivation prob (NULL if off) */
    double *d_kpacket_cdf;                  /* [n_shells * n_macro_levels] per-shell re-excitation CDF (NULL if off) */
    int    *d_line_atomic_number;          /* [n_lines] Z for Fe two-level scatter */
    int    *d_line_ion_number;             /* [n_lines] ion stage for Fe II selective scatter */

    /* Phase 6 - Step 1: Geometry arrays */
    double *d_r_inner;                     /* Phase 6 - Step 1: [n_shells] */
    double *d_r_outer;                     /* Phase 6 - Step 1: [n_shells] */

    /* Phase 6 - Step 1: Estimators (atomic writes) */
    double *d_j_estimator;                 /* Phase 6 - Step 1: [n_shells] */
    double *d_nu_bar_estimator;            /* Phase 6 - Step 1: [n_shells] */

    /* Phase 6 - Step 1: RNG + output */
    uint64_t *d_rng_states;                /* Phase 6 - Step 1: [n_packets * 4] xoshiro */
    double *d_escaped_nu;                  /* Phase 6 - Step 1: [n_packets] */
    double *d_escaped_energy;              /* Phase 6 - Step 1: [n_packets] */
    int    *d_escaped_flag;                /* Phase 6 - Step 1: [n_packets] */
    int64_t *d_n_escaped;                  /* Phase 6 - Step 1: scalar counter */
    int64_t *d_n_reabsorbed;               /* Phase 6 - Step 1: scalar counter */

    /* Rotation packet mode: store r, mu at escape for Doppler weighting */
    double *d_escaped_r;                   /* [n_packets] */
    double *d_escaped_mu;                  /* [n_packets] */

    /* Virtual packet spectrum (GPU-side atomicAdd target) */
    double *d_virtual_spectrum;            /* [VSPEC_N_BINS] */

    /* NLTE: J_nu frequency histogram (atomicAdd target) */
    double *d_j_nu_estimator;              /* [n_shells * NLTE_N_FREQ_BINS] or NULL */
    int     nlte_n_freq_bins;              /* 0 if NLTE disabled */
    double  nlte_nu_min;
    double  nlte_d_log_nu;

    /* BF opacity: chi_bf grid + T_rad for BF absorption re-emission */
    double *d_chi_bf;                      /* [n_shells * BF_N_FREQ_BINS] or NULL */
    double *d_T_rad;                       /* [n_shells] for BF Planck re-emission */
    int    *d_bf_activation_level;         /* [n_shells * BF_N_FREQ_BINS] macro-atom level or -1 */
    int     bf_enabled;
    int     bf_n_freq_bins;
    double  bf_nu_min;
    double  bf_nu_max;
    double  bf_d_log_nu;
} CudaDeviceData;                          /* Phase 6 - Step 1 */

/* Virtual spectrum parameters (match real spectrum) */
#define VSPEC_LAMBDA_MIN  500.0
#define VSPEC_LAMBDA_MAX  20000.0
#define VSPEC_N_BINS      2000
#define N_VPACKETS        10     /* virtual packets per interaction (TARDIS default) */

/* ============================================================ */
/* Phase 6 - Step 1: cuda_allocate — allocate GPU memory        */
/* ============================================================ */
static void cuda_allocate(CudaDeviceData *dev, Geometry *geo,
                           OpacityState *opacity, int n_packets) {
    int ns = geo->n_shells;                /* Phase 6 - Step 1 */
    int nl = opacity->n_lines;             /* Phase 6 - Step 1 */
    int nt = opacity->n_macro_transitions; /* Phase 6 - Step 1 */
    int nlev = opacity->n_macro_levels;    /* Phase 6 - Step 1 */

    /* Phase 6 - Step 1: Read-only arrays */
    CUDA_CHECK(cudaMalloc(&dev->d_line_list_nu, nl * sizeof(double)));              /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_tau_sobolev, (size_t)nl * ns * sizeof(double)));  /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_electron_density, ns * sizeof(double)));          /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_transition_probabilities,                         /* Phase 6 - Step 1 */
               (size_t)nt * ns * sizeof(double)));                                  /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_macro_block_references, (nlev + 1) * sizeof(int))); /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_transition_type, nt * sizeof(int)));              /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_destination_level_id, nt * sizeof(int)));         /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_transition_line_id, nt * sizeof(int)));           /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_line2macro_level_upper, nl * sizeof(int)));       /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_line_atomic_number, nl * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev->d_line_ion_number, nl * sizeof(int)));
    /* k-packet tables: allocated only when LUMINA_KPACKET is enabled (host side
     * builds them in compute_transition_probabilities). NULL → selector skips. */
    dev->d_p_kpacket  = NULL;
    dev->d_kpacket_cdf = NULL;
    if (getenv("LUMINA_KPACKET") && atoi(getenv("LUMINA_KPACKET")) != 0) {
        CUDA_CHECK(cudaMalloc(&dev->d_p_kpacket,  (size_t)nlev * ns * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dev->d_kpacket_cdf, (size_t)ns * nlev * sizeof(double)));
        /* zero-init so the first transport iter (before compute_transition_
         * probabilities populates the tables) makes no k-packet rolls (p=0). */
        CUDA_CHECK(cudaMemset(dev->d_p_kpacket,  0, (size_t)nlev * ns * sizeof(double)));
        CUDA_CHECK(cudaMemset(dev->d_kpacket_cdf, 0, (size_t)ns * nlev * sizeof(double)));
    }

    /* Phase 6 - Step 1: Geometry */
    CUDA_CHECK(cudaMalloc(&dev->d_r_inner, ns * sizeof(double)));                   /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_r_outer, ns * sizeof(double)));                   /* Phase 6 - Step 1 */

    /* d_T_rad always-allocated: needed by EPS_UV / EPS_IR macro-atom thermalization
     * even when BF opacity is disabled. (cuda_allocate_bf will skip its own malloc
     * if d_T_rad is already non-NULL.) */
    CUDA_CHECK(cudaMalloc(&dev->d_T_rad, ns * sizeof(double)));

    /* Phase 6 - Step 1: Estimators */
    CUDA_CHECK(cudaMalloc(&dev->d_j_estimator, ns * sizeof(double)));               /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_nu_bar_estimator, ns * sizeof(double)));          /* Phase 6 - Step 1 */

    /* Phase 6 - Step 1: RNG (4 uint64 per packet for xoshiro256**) */
    CUDA_CHECK(cudaMalloc(&dev->d_rng_states, (size_t)n_packets * 4 * sizeof(uint64_t))); /* Phase 6 - Step 1 */

    /* Phase 6 - Step 1: Output arrays */
    CUDA_CHECK(cudaMalloc(&dev->d_escaped_nu, n_packets * sizeof(double)));         /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_escaped_energy, n_packets * sizeof(double)));     /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_escaped_flag, n_packets * sizeof(int)));          /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_n_escaped, sizeof(int64_t)));                     /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_n_reabsorbed, sizeof(int64_t)));                  /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_escaped_r, n_packets * sizeof(double)));          /* Rotation mode */
    CUDA_CHECK(cudaMalloc(&dev->d_escaped_mu, n_packets * sizeof(double)));         /* Rotation mode */
    CUDA_CHECK(cudaMalloc(&dev->d_virtual_spectrum, VSPEC_N_BINS * sizeof(double)));

    /* NLTE: J_nu estimator (allocated but NULL-checked in kernel) */
    dev->d_j_nu_estimator = NULL;
    dev->nlte_n_freq_bins = 0;
}

/* NLTE: allocate J_nu estimator on GPU */
static void cuda_allocate_nlte(CudaDeviceData *dev, NLTEConfig *nlte,
                                int n_shells) {
    size_t size = (size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dev->d_j_nu_estimator, size));
    CUDA_CHECK(cudaMemset(dev->d_j_nu_estimator, 0, size));
    dev->nlte_n_freq_bins = nlte->n_freq_bins;
    dev->nlte_nu_min = nlte->nu_min;
    dev->nlte_d_log_nu = nlte->d_log_nu;
    printf("  [NLTE] GPU J_nu estimator: %.1f KB\n", size / 1024.0);
}

/* BF opacity: allocate GPU arrays */
static void cuda_allocate_bf(CudaDeviceData *dev, BFOpacity *bf, int n_shells) {
    size_t chi_size = (size_t)n_shells * bf->n_freq_bins * sizeof(double);
    size_t act_size = (size_t)n_shells * bf->n_freq_bins * sizeof(int);
    CUDA_CHECK(cudaMalloc(&dev->d_chi_bf, chi_size));
    /* d_T_rad is now always-allocated in cuda_allocate(); skip if present. */
    if (!dev->d_T_rad)
        CUDA_CHECK(cudaMalloc(&dev->d_T_rad, n_shells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev->d_bf_activation_level, act_size));
    dev->bf_enabled = 1;
    dev->bf_n_freq_bins = bf->n_freq_bins;
    dev->bf_nu_min = bf->nu_min;
    dev->bf_nu_max = bf->nu_max;
    dev->bf_d_log_nu = bf->d_log_nu;
    printf("  [BF] GPU arrays allocated: %.1f KB\n", (chi_size + act_size) / 1024.0);
}

/* Upload T_rad to GPU. Always-callable (independent of BF status) — needed
 * by EPS_UV / EPS_IR macro-atom thermalization paths. */
static void cuda_upload_T_rad(CudaDeviceData *dev, PlasmaState *plasma,
                               int n_shells) {
    if (!dev->d_T_rad) return;
    CUDA_CHECK(cudaMemcpy(dev->d_T_rad, plasma->T_rad,
               n_shells * sizeof(double), cudaMemcpyHostToDevice));
}

/* BF opacity: upload chi_bf grid + T_rad + activation_level to GPU */
static void cuda_upload_bf(CudaDeviceData *dev, BFOpacity *bf,
                            PlasmaState *plasma, int n_shells) {
    if (!dev->d_chi_bf) return;
    size_t chi_size = (size_t)n_shells * bf->n_freq_bins * sizeof(double);
    size_t act_size = (size_t)n_shells * bf->n_freq_bins * sizeof(int);
    CUDA_CHECK(cudaMemcpy(dev->d_chi_bf, bf->chi_bf, chi_size,
               cudaMemcpyHostToDevice));
    cuda_upload_T_rad(dev, plasma, n_shells);
    CUDA_CHECK(cudaMemcpy(dev->d_bf_activation_level, bf->activation_level,
               act_size, cudaMemcpyHostToDevice));
}

/* NLTE: download J_nu from GPU to CPU NLTEConfig */
static void cuda_download_j_nu(CudaDeviceData *dev, NLTEConfig *nlte,
                                int n_shells) {
    if (!dev->d_j_nu_estimator) return;
    size_t size = (size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(double);
    CUDA_CHECK(cudaMemcpy(nlte->j_nu_estimator, dev->d_j_nu_estimator,
               size, cudaMemcpyDeviceToHost));
}

/* ============================================================ */
/* cuBLAS batched NLTE solver functions                         */
/* ============================================================ */

static void cuda_nlte_solver_init(CudaNLTESolver *sol, int max_N, int batch_size) {
    memset(sol, 0, sizeof(*sol));
    cublasCreate(&sol->handle);
    sol->max_N = max_N;
    sol->batch_size = batch_size;

    size_t mat_bytes = (size_t)batch_size * max_N * max_N * sizeof(double);
    size_t rhs_bytes = (size_t)batch_size * max_N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&sol->d_matrices, mat_bytes));
    CUDA_CHECK(cudaMalloc(&sol->d_rhs, rhs_bytes));
    CUDA_CHECK(cudaMalloc(&sol->d_Aarray, batch_size * sizeof(double *)));
    CUDA_CHECK(cudaMalloc(&sol->d_Barray, batch_size * sizeof(double *)));
    CUDA_CHECK(cudaMalloc(&sol->d_pivot, (size_t)batch_size * max_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sol->d_info, batch_size * sizeof(int)));

    sol->h_matrices = (double *)malloc(mat_bytes);
    sol->h_rhs      = (double *)malloc(rhs_bytes);
    sol->h_Aarray   = (double **)malloc(batch_size * sizeof(double *));
    sol->h_Barray   = (double **)malloc(batch_size * sizeof(double *));
    sol->h_info     = (int *)malloc(batch_size * sizeof(int));

    printf("  [NLTE-GPU] cuBLAS solver: max_N=%d, batch=%d, GPU mem=%.1f MB\n",
           max_N, batch_size, (mat_bytes + rhs_bytes) / (1024.0 * 1024.0));
}

static void cuda_nlte_solver_free(CudaNLTESolver *sol) {
    if (sol->handle) cublasDestroy(sol->handle);
    if (sol->d_matrices) cudaFree(sol->d_matrices);
    if (sol->d_rhs)      cudaFree(sol->d_rhs);
    if (sol->d_Aarray)   cudaFree(sol->d_Aarray);
    if (sol->d_Barray)   cudaFree(sol->d_Barray);
    if (sol->d_pivot)    cudaFree(sol->d_pivot);
    if (sol->d_info)     cudaFree(sol->d_info);
    free(sol->h_matrices);
    free(sol->h_rhs);
    free(sol->h_Aarray);
    free(sol->h_Barray);
    free(sol->h_info);
    memset(sol, 0, sizeof(*sol));
}

/* Two-sided inf-norm equilibration of a column-major NxN system A x = b
 * (A_cm[j*N+i] = A(i,j)). Scales rows then columns to O(1), which is an EXACT
 * transform (does not change the solution): A' = Dr A Dc, b' = Dr b, solve
 * A' y = b', recover x[j] = Dc[j]*y[j]. Returns the column scales in cscale[].
 *
 * Rationale (2026-06-18, 3-way verified audit): the NLTE rate matrix mixes a
 * conservation row of O(1) ones (RHS = number density ~1e8) with rate rows of
 * O(1e6-1e8) s^-1, fed RAW to cuBLAS getrf. The huge resulting condition number
 * makes getrf's partial pivoting unable to see the true rank at low n_e -> it
 * returns finite GARBAGE excited pops (~1e-26, alternating sign) with info=0
 * (not flagged singular). Equilibration collapses the condition number so the
 * solve returns the true populations where the system is well-posed, and lets
 * getrf honestly flag info!=0 for the genuinely-singular (missing-physics) case
 * -> routes to the CPU Boltzmann fallback instead of silent garbage. This is a
 * faithful numerical fix, NOT a clamp: it changes only the conditioning. */
static void nlte_equilibrate_system(double *A_cm, double *b, int N, double *cscale) {
    for (int i = 0; i < N; i++) {            /* row scaling: r[i] = 1/max_j|A(i,j)| */
        double rmax = 0.0;
        for (int j = 0; j < N; j++) {
            double v = fabs(A_cm[(size_t)j * N + i]);
            if (v > rmax) rmax = v;
        }
        double r = (rmax > 0.0) ? 1.0 / rmax : 1.0;
        for (int j = 0; j < N; j++) A_cm[(size_t)j * N + i] *= r;
        b[i] *= r;
    }
    for (int j = 0; j < N; j++) {            /* col scaling: c[j] = 1/max_i|A(i,j)| */
        double *col = A_cm + (size_t)j * N;
        double cmax = 0.0;
        for (int i = 0; i < N; i++) {
            double v = fabs(col[i]);
            if (v > cmax) cmax = v;
        }
        double c = (cmax > 0.0) ? 1.0 / cmax : 1.0;
        for (int i = 0; i < N; i++) col[i] *= c;
        cscale[j] = c;
    }
}

static int nlte_equilibrate_enabled(void) {
    static int init = 0, on = 0;
    if (!init) { const char *e = getenv("LUMINA_NLTE_EQUILIBRATE");
                 on = (e && atoi(e) != 0) ? 1 : 0; init = 1; }
    return on;
}

/* Singular-matrix Boltzmann fallback temperature (2026-06-18). The rate matrix
 * goes singular precisely when collisions dominate (high n_e, inner shells) ->
 * the level distribution there is the LTE limit at T_e, NOT T_rad. The legacy
 * fallback used Boltzmann@T_rad; at inner shells T_rad is 1-4% hotter than T_e,
 * which injects a spurious super-thermal S_l/B(T_e)>1 on every fallback line.
 * LUMINA_NLTE_FALLBACK_TE=1 falls back to Boltzmann@T_e instead. */
static int nlte_fallback_te_enabled(void) {
    static int init = 0, on = 0;
    if (!init) { const char *e = getenv("LUMINA_NLTE_FALLBACK_TE");
                 on = (e && atoi(e) != 0) ? 1 : 0; init = 1; }
    return on;
}

/* Residual check (2026-06-18, codex 019ed80e). cublasDgetrf reports info!=0 only
 * for an EXACT-zero pivot; a NEAR-singular matrix factorizes with info=0 yet the
 * triangular solve returns finite GARBAGE (the info=0 super-thermal tail, Sl/B up
 * to 1e55). LUMINA_NLTE_RESID_CHECK=1 computes ||A x - b|| / ||b|| on the ORIGINAL
 * (pre-equilibration) system per shell and, if it exceeds LUMINA_NLTE_RESID_TOL
 * (default 1e-3), flags that shell singular (h_info=sentinel) so the existing
 * caller routes it to the Boltzmann fallback. Catches what getrf info=0 misses. */
static int nlte_resid_check_enabled(void) {
    static int init = 0, on = 0;
    if (!init) { const char *e = getenv("LUMINA_NLTE_RESID_CHECK");
                 on = (e && atoi(e) != 0) ? 1 : 0; init = 1; }
    return on;
}
static double nlte_resid_tol(void) {
    static int init = 0; static double tol = 1e-3;
    if (!init) { const char *e = getenv("LUMINA_NLTE_RESID_TOL");
                 if (e) tol = atof(e); init = 1; }
    return tol;
}

/* Preemptive LTE@T_e zone (2026-06-18). The rate matrix goes near-singular at
 * inner shells BECAUSE collisions dominate (the LTE limit), and the LU then
 * lands on a low-residual but null-space-contaminated solution (the Sl/B up to
 * 1e55 garbage tail — provably NOT detectable from the matrix: residual ~1e-22,
 * cond identical to good solves). The physically-correct answer at a collision-
 * dominated shell is LTE@T_e. LUMINA_NLTE_LTE_NCRIT=<n_e threshold> forces every
 * shell with n_e above it to the Boltzmann@T_e fallback (requires FALLBACK_TE),
 * skipping the fragile solve. n_crit for forbidden metastables (the garbage
 * source) ~ A_ul/q ~ 1e7-1e9; allowed UV lines have n_crit >> 1e12, so a moderate
 * threshold leaves the emergent-UV-shaping NLTE intact (validated by A/B). 0=off. */
static double nlte_lte_zone_ncrit(void) {
    static int init = 0; static double ncrit = 0.0;
    if (!init) { const char *e = getenv("LUMINA_NLTE_LTE_NCRIT");
                 if (e) ncrit = atof(e); init = 1; }
    return ncrit;
}

/* Batched LU solve: factorize + triangular solve on GPU for all shells at once.
 * h_matrices[batch * N * N] and h_rhs[batch * N] must be pre-filled (column-major). */
static int cuda_nlte_batched_solve(CudaNLTESolver *sol, int N, int batch) {
    size_t mat_bytes = (size_t)batch * N * N * sizeof(double);
    size_t rhs_bytes = (size_t)batch * N * sizeof(double);

    /* Residual check (LUMINA_NLTE_RESID_CHECK=1): snapshot the ORIGINAL assembled
     * A and b BEFORE equilibration/getrf overwrite them, so ||A x - b|| can be
     * formed after the solve to detect near-singular (info=0) garbage. */
    int rc_on = nlte_resid_check_enabled();
    double *rc_A = NULL, *rc_b = NULL;
    if (rc_on) {
        rc_A = (double *)malloc(mat_bytes);
        rc_b = (double *)malloc(rhs_bytes);
        if (rc_A && rc_b) {
            memcpy(rc_A, sol->h_matrices, mat_bytes);
            memcpy(rc_b, sol->h_rhs, rhs_bytes);
        } else { free(rc_A); free(rc_b); rc_A = rc_b = NULL; rc_on = 0; }
    }

    /* Optional pre-solve equilibration (LUMINA_NLTE_EQUILIBRATE=1). Scale each
     * matrix+RHS to O(1) and remember per-shell column scales to un-scale the
     * downloaded solution. Faithful (exact) conditioning fix; see helper above. */
    int eq_on = nlte_equilibrate_enabled();
    double *eq_cscale = NULL;
    if (eq_on) {
        eq_cscale = (double *)malloc((size_t)batch * N * sizeof(double));
        if (eq_cscale) {
            for (int i = 0; i < batch; i++)
                nlte_equilibrate_system(sol->h_matrices + (size_t)i * N * N,
                                        sol->h_rhs + (size_t)i * N, N,
                                        eq_cscale + (size_t)i * N);
        } else {
            eq_on = 0;  /* OOM: skip equilibration, solve raw */
        }
    }

    /* Set up device pointer arrays (each points to contiguous slice) */
    for (int i = 0; i < batch; i++) {
        sol->h_Aarray[i] = sol->d_matrices + (size_t)i * N * N;
        sol->h_Barray[i] = sol->d_rhs      + (size_t)i * N;
    }

    /* Upload matrices, RHS, and pointer arrays */
    CUDA_CHECK(cudaMemcpy(sol->d_matrices, sol->h_matrices, mat_bytes,
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol->d_rhs, sol->h_rhs, rhs_bytes,
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol->d_Aarray, sol->h_Aarray,
               batch * sizeof(double *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sol->d_Barray, sol->h_Barray,
               batch * sizeof(double *), cudaMemcpyHostToDevice));

    /* Batched LU factorization */
    cublasStatus_t stat = cublasDgetrfBatched(sol->handle, N,
        sol->d_Aarray, N, sol->d_pivot, sol->d_info, batch);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NLTE-GPU] cublasDgetrfBatched failed: %d\n", stat);
        if (eq_cscale) free(eq_cscale);
        free(rc_A); free(rc_b);
        return -1;
    }

    /* A8: Inspect per-matrix info BEFORE the solve. getrf writes info[i]>0 when
     * matrix i is singular (pivot exactly 0). getrsBatched on a singular matrix
     * produces NaN/garbage in d_rhs without surfacing an error — masking the
     * problem as bad downstream populations rather than a numerical failure. */
    CUDA_CHECK(cudaMemcpy(sol->h_info, sol->d_info, batch * sizeof(int),
               cudaMemcpyDeviceToHost));
    int n_singular = 0, first_sing = -1;
    for (int i = 0; i < batch; i++) {
        if (sol->h_info[i] != 0) {
            if (first_sing < 0) first_sing = i;
            n_singular++;
        }
    }
    if (n_singular > 0) {
        fprintf(stderr,
                "[NLTE-GPU] WARNING: %d/%d batched matrices singular after getrf "
                "(first at i=%d, info=%d). Solves on these will yield NaN; "
                "caller should fall back to CPU per-matrix solve.\n",
                n_singular, batch, first_sing, sol->h_info[first_sing]);
        /* Continue — caller (nlte_solve_all_gpu) inspects sol->h_info[] and
         * routes singular slots to cusolver fallback. */
    }

    /* Batched triangular solve */
    int info_host = 0;
    stat = cublasDgetrsBatched(sol->handle, CUBLAS_OP_N, N, 1,
        (const double **)sol->d_Aarray, N, sol->d_pivot,
        sol->d_Barray, N, &info_host, batch);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NLTE-GPU] cublasDgetrsBatched failed: %d\n", stat);
        if (eq_cscale) free(eq_cscale);
        free(rc_A); free(rc_b);
        return -1;
    }

    /* Download solutions */
    CUDA_CHECK(cudaMemcpy(sol->h_rhs, sol->d_rhs, rhs_bytes,
               cudaMemcpyDeviceToHost));

    /* Un-scale: the GPU solved A' y = b' (column-equilibrated), so recover the
     * true population x[j] = cscale[j] * y[j]. Output semantics identical to the
     * un-equilibrated path -> no downstream change. */
    if (eq_on && eq_cscale) {
        for (int i = 0; i < batch; i++) {
            double *y = sol->h_rhs + (size_t)i * N;
            double *cs = eq_cscale + (size_t)i * N;
            for (int j = 0; j < N; j++) y[j] *= cs[j];
        }
    }
    if (eq_cscale) free(eq_cscale);

    /* Residual check: form ||A x - b|| / ||b|| per shell on the ORIGINAL system
     * (rc_A, rc_b snapshotted pre-equilibration; x = solved h_rhs). Flag a shell
     * singular (h_info sentinel) when the relative residual exceeds the tolerance,
     * so the caller routes it to the Boltzmann fallback. Catches the info=0
     * near-singular garbage that getrf and the inv_ceil gate both miss. */
    if (rc_on) {
        double tol = nlte_resid_tol();
        int n_bad = 0;
        for (int i = 0; i < batch; i++) {
            if (sol->h_info[i] != 0) continue;          /* already flagged */
            const double *A = rc_A + (size_t)i * N * N;  /* column-major */
            const double *b = rc_b + (size_t)i * N;
            const double *x = sol->h_rhs + (size_t)i * N;
            double rn = 0.0, bn = 0.0;
            for (int k = 0; k < N; k++) {
                double ax = 0.0;
                for (int j = 0; j < N; j++) ax += A[(size_t)j * N + k] * x[j];
                double d = ax - b[k];
                rn += d * d; bn += b[k] * b[k];
            }
            double rel = (bn > 0.0) ? sqrt(rn / bn) : (rn > 0.0 ? 1e30 : 0.0);
            if (rel > tol) { sol->h_info[i] = 88888; n_bad++; }  /* sentinel */
        }
        if (n_bad > 0) {
            static int rc_warn = 0;
            if (rc_warn < 8) {
                fprintf(stderr, "[NLTE-RESID] %d/%d shells exceed ||Ax-b||/||b|| > "
                        "%.1e -> routed to fallback (info=0 near-singular)\n",
                        n_bad, batch, tol);
                rc_warn++;
            }
        }
    }
    free(rc_A); free(rc_b);
    return 0;
}


/* GPU NLTE master solver: assemble on CPU (OpenMP), solve on GPU (cuBLAS batched).
 * Step 1.5: Iterative CE convergence wrapper — same logic as CPU nlte_solve_all. */
static void nlte_solve_all_gpu(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                double time_explosion, int n_shells,
                                CudaNLTESolver *sol,
                                GammaDeposition *gamma_dep) {
    printf("  [NLTE-GPU] Solving rate equations (cuBLAS batched, with CE)...\n");

    /* #281: 16 pairs (last two overlap on slot 29 = O II) for full O triplet. */
    int n_pairs = NLTE_PAIR_COUNT;
    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11},
                       {12,13}, {14,15}, {16,17}, {18,19},
                       {20,21}, {22,23}, {24,25}, {26,27},
                       {28,29}, {29,30} };
    const char *names[] = { "Si", "Ca", "Fe", "S", "Co", "Ni",
                            "C", "Mg", "Ti", "Cr",
                            "Al", "Sc", "V", "Mn",
                            "O(I-II)", "O(II-III)" };

    int ce_max_iter = 5;
    double ce_threshold = 1e-2;  /* 1% relative convergence on ion totals */
    double ce_damping = 0.5;     /* 50% damping */

    /* Save old ion totals for convergence check (n_nlte_ions * n_shells) */
    int n_ion_totals = nlte->n_nlte_ions * n_shells;
    double *old_ion_totals = (double *)calloc(n_ion_totals, sizeof(double));
    size_t pop_size = (size_t)nlte->n_nlte_levels_total * n_shells;
    double *old_pops = (double *)malloc(pop_size * sizeof(double));

    /* Task #40 (A)+(B): pre-bake photoionization rates on GPU.
     * J_nu is constant across the CE iterations, so we compute R_bf once
     * (covers all pairs × shells × levels) and pass the lookup table to the
     * rate-matrix assembler. Falls back to per-call CPU loop if init/compute
     * fails or env var disables it. */
    NLTERateLookup nlte_lookup = {0};
    NLTERateLookup *nlte_lookup_ptr = NULL;
    int nlte_gemm_ok = 0;
    if (getenv("LUMINA_NLTE_RATES_GEMM") == NULL || atoi(getenv("LUMINA_NLTE_RATES_GEMM")) != 0) {
        if (nlte_rates_gpu_init(nlte, atom, n_shells) == 0 &&
            nlte_rates_gpu_compute(nlte, &nlte_lookup) == 0) {
            nlte_lookup_ptr = &nlte_lookup;
            nlte_gemm_ok = 1;
        }
    }
    if (nlte_gemm_ok)
        printf("  [NLTE-GEMM] R_bf table ready (%d levels × %d shells)\n",
               nlte_lookup.L_phot_total, n_shells);

    /* Profiling: split wall-clock into assemble vs GPU solve. */
    double t_assemble_total = 0.0, t_solve_total = 0.0;

    for (int ce_iter = 0; ce_iter < ce_max_iter; ce_iter++) {
        /* Save current populations + compute old ion totals */
        memcpy(old_pops, nlte->nlte_level_populations, pop_size * sizeof(double));
        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int lev_s = nlte->nlte_ion_level_offset[ii];
            int lev_e = nlte->nlte_ion_level_offset[ii + 1];
            for (int s = 0; s < n_shells; s++) {
                double sum = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    sum += nlte->nlte_level_populations[l * n_shells + s];
                old_ion_totals[ii * n_shells + s] = sum;
            }
        }

        /* Solve all ion pairs */
        for (int p = 0; p < n_pairs; p++) {
            int lo = pairs[p][0], hi = pairs[p][1];
            int lev_start = nlte->nlte_ion_level_offset[lo];
            int super_start = nlte->nlte_ion_super_offset[lo];
            /* Solve on super-levels (N), redistribute to full levels (N_fl).
             * Identity mode: N==N_fl and the expansion is a no-op. */
            int N    = nlte->nlte_ion_super_offset[hi + 1] - super_start;
            int N_fl = nlte->nlte_ion_level_offset[hi + 1] - lev_start;
            int n_lo_super = nlte->nlte_ion_super_offset[lo + 1] - super_start;
            if (N <= 0 || N_fl <= 0) continue;

            /* C1 (GPU port of plasma.c:3357-3396): the O triplet is solved as
             * two overlapping pairs (14={28,29}=O(I-II), 15={29,30}=O(II-III))
             * that share slot 29 = O II. Two defects without protection:
             *  (1) pair 15 overwrites the O II block pair 14 set, breaking the
             *      O I<->O II conservation (observed O I ~500x over-pop), and
             *  (2) the combined renorm of pair 15 scales the O III levels by
             *      [n(OII)+n(OIII)]/Sigma_x; with O nearly neutral n(OIII)~2e-9
             *      while the scale is n(OII)-dominated -> O III ~1e13x over-pop.
             * Fix: snapshot the shared lo-ion (O II) block before the solve and
             * restore it after extraction (cures 1); and force PER-ION rescale
             * for every pair that shares a slot with another pair (both O pairs)
             * so each O ion is pinned to its own nebular total (cures 2 AND the
             * O I over-pop: pair 14's combined renorm dumps n(OI)+n(OII), which
             * is n(OII)-dominated, onto the O I levels -> O I ~500x over-pop).
             * lo_overlaps_prior gates the save/restore; pair_shares_slot gates
             * the per-ion force (see the rescale branches below). */
            int lo_overlaps_prior = 0;
            int pair_shares_slot = 0;
            for (int pp = 0; pp < n_pairs; pp++) {
                if (pp == p) continue;
                if (pairs[pp][0] == lo || pairs[pp][1] == lo) {
                    pair_shares_slot = 1;
                    if (pp < p) lo_overlaps_prior = 1;
                }
                if (pairs[pp][0] == hi || pairs[pp][1] == hi)
                    pair_shares_slot = 1;
            }
            double *saved_lo = NULL;
            int saved_lev_s = 0, saved_lev_e = 0;
            if (lo_overlaps_prior) {
                saved_lev_s = nlte->nlte_ion_level_offset[lo];
                saved_lev_e = nlte->nlte_ion_level_offset[lo + 1];
                size_t n_save = (size_t)(saved_lev_e - saved_lev_s) * n_shells;
                saved_lo = (double *)malloc(n_save * sizeof(double));
                memcpy(saved_lo,
                       &nlte->nlte_level_populations[(size_t)saved_lev_s * n_shells],
                       n_save * sizeof(double));
            }

            /* Zero staging buffers */
            size_t mat_bytes = (size_t)n_shells * N * N * sizeof(double);
            size_t rhs_bytes = (size_t)n_shells * N * sizeof(double);
            memset(sol->h_matrices, 0, mat_bytes);
            memset(sol->h_rhs, 0, rhs_bytes);

            /* Assemble rate matrices for all shells (CPU, OpenMP parallel) */
            struct timespec ts_a0, ts_a1, ts_s1;
            clock_gettime(CLOCK_MONOTONIC, &ts_a0);
            int skip_dead = nlte_skip_dead_pairs();
            int Z_pair = nlte->nlte_Z[lo];
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int s = 0; s < n_shells; s++) {
                double *A_cm = sol->h_matrices + (size_t)s * N * N;
                double *b    = sol->h_rhs      + (size_t)s * N;
                if (skip_dead) {
                    double n_tot = nlte_pair_total_density(nlte, atom, plasma,
                                                           Z_pair, lo, hi, s);
                    /* Dead pair (no atoms of this element here): leave the
                     * zeroed matrix -> getrf singular -> inline Boltzmann
                     * fallback writes ~0 pops. Skip the costly assembly. */
                    if (n_tot < 1e-10) continue;
                }
                nlte_assemble_rate_matrix(nlte, atom, plasma, opacity,
                                          lo, hi, s, time_explosion,
                                          A_cm, b, N, gamma_dep,
                                          nlte_lookup_ptr, p);
            }
            clock_gettime(CLOCK_MONOTONIC, &ts_a1);
            t_assemble_total += (ts_a1.tv_sec - ts_a0.tv_sec) +
                                1e-9 * (ts_a1.tv_nsec - ts_a0.tv_nsec);

            /* FALSIFIER (LUMINA_NLTE_MATDUMP=1): dump the ORIGINAL (pre-LU,
             * pre-equilibration) rate matrix A (col-major) + RHS b for one
             * target pair/shell so SVD/cond/null-space can be computed offline.
             * Confirms whether the negative-pops-with-info=0 are a structural
             * near-singularity (smallest right-singular vector on the top-ion
             * excited manifold) vs ill-scaling or an assembly bug. File is
             * overwritten each iter -> holds the last (converged) iter. */
            if (getenv("LUMINA_NLTE_MATDUMP") &&
                nlte->nlte_Z[lo] == (getenv("LUMINA_POP_Z") ? atoi(getenv("LUMINA_POP_Z")) : 8) &&
                nlte->nlte_ion[lo] == (getenv("LUMINA_POP_ION") ? atoi(getenv("LUMINA_POP_ION")) : 1)) {
                int sdump = getenv("LUMINA_POP_SHELL") ? atoi(getenv("LUMINA_POP_SHELL")) : 24;
                if (sdump >= 0 && sdump < n_shells) {
                    const char *path = getenv("LUMINA_NLTE_MATDUMP_PATH");
                    if (!path) path = "lumina_nlte_matrix.bin";
                    FILE *mf = fopen(path, "wb");
                    if (mf) {
                        int n_lo_lev = nlte->nlte_ion_level_offset[lo + 1] -
                                       nlte->nlte_ion_level_offset[lo];
                        int hdr[5] = { N, n_lo_lev, nlte->nlte_Z[lo],
                                       nlte->nlte_ion[lo], sdump };
                        fwrite(hdr, sizeof(int), 5, mf);
                        fwrite(sol->h_matrices + (size_t)sdump * N * N,
                               sizeof(double), (size_t)N * N, mf);   /* col-major */
                        fwrite(sol->h_rhs + (size_t)sdump * N,
                               sizeof(double), (size_t)N, mf);
                        fclose(mf);
                        fprintf(stderr, "[MATDUMP] wrote %s N=%d n_lo=%d Z=%d ion=%d s=%d\n",
                                path, N, n_lo_lev, nlte->nlte_Z[lo],
                                nlte->nlte_ion[lo], sdump);
                    }
                }
            }

            /* GPU batched solve */
            int ret = cuda_nlte_batched_solve(sol, N, n_shells);
            clock_gettime(CLOCK_MONOTONIC, &ts_s1);
            t_solve_total += (ts_s1.tv_sec - ts_a1.tv_sec) +
                             1e-9 * (ts_s1.tv_nsec - ts_a1.tv_nsec);

            /* Extract populations, handle singular matrices.
             * LUMINA_NLTE_FORCE_LTE_LEVELS=1: bypass the rate-solve result for
             * every shell and use Boltzmann@T_rad (i.e. only the ion-lock
             * matrix conservation row is honored — level distribution is LTE).
             * Tests whether the spectrum degradation under ion-lock comes from
             * the NLTE level-pop solve rather than the ion totals themselves. */
            static int force_lte_init = 0;
            static int force_lte_mode = 0;
            if (!force_lte_init) {
                const char *e = getenv("LUMINA_NLTE_FORCE_LTE_LEVELS");
                if (e && atoi(e) != 0) force_lte_mode = 1;
                force_lte_init = 1;
            }
            double lte_ncrit = nlte_lte_zone_ncrit();
            for (int s = 0; s < n_shells; s++) {
                int need_fallback = (ret != 0 || sol->h_info[s] != 0 || force_lte_mode);
                /* Preemptive LTE@T_e zone: at collision-dominated shells (n_e above
                 * the critical density) the rate solve is near-singular and the LU
                 * lands on null-space garbage; the physical limit is LTE@T_e, so
                 * route the whole shell to the (T_e) Boltzmann fallback. */
                if (lte_ncrit > 0.0 && plasma->n_electron &&
                    plasma->n_electron[s] > lte_ncrit)
                    need_fallback = 1;
                if (!need_fallback) {
                    /* Scan GPU solution for NaN/Inf — cuBLAS LU can succeed
                     * (info==0) but produce non-finite output when the rate
                     * matrix is ill-conditioned at high T_e. */
                    double *x = sol->h_rhs + (size_t)s * N;
                    for (int i = 0; i < N; i++) {
                        if (!isfinite(x[i])) { need_fallback = 1; break; }
                    }
                    /* Boltzmann-ceiling sanity gate: a near-singular matrix can
                     * yield a FINITE but inverted solution (excited levels
                     * 1e9-1e11x ground). Reject when any level exceeds its ion
                     * ground pop by more than (g_i/g_ground)*margin. */
                    double inv_ceil = nlte_inv_ceiling();
                    if (!need_fallback && inv_ceil > 0.0) {
                        int n_lo = n_lo_super;   /* SL-space ground/excited split */
                        int g0_lo = atom->level_g[nlte->super_anchor_global[super_start]];
                        int g0_hi = (n_lo < N) ?
                            atom->level_g[nlte->super_anchor_global[super_start + n_lo]] : 1;
                        double x0_lo = x[0];
                        double x0_hi = (n_lo < N) ? x[n_lo] : 1.0;
                        for (int i = 0; i < N; i++) {
                            double xg = (i < n_lo) ? x0_lo : x0_hi;
                            int gg = (i < n_lo) ? g0_lo : g0_hi;
                            if (xg <= 0.0) {
                                /* Empty/negative ground with a populated excited
                                 * level is itself an inversion — the garbage solve
                                 * drained the ground state. */
                                if (x[i] > 0.0) { need_fallback = 1; break; }
                                continue;
                            }
                            int gi = atom->level_g[nlte->super_anchor_global[super_start + i]];
                            double ceil_ratio = ((double)gi / (double)(gg > 0 ? gg : 1)) * inv_ceil;
                            if (x[i] / xg > ceil_ratio) { need_fallback = 1; break; }
                        }
                    }
                }
                /* Flat-pop diagnostic (LUMINA_POP_TRACE): dump the RAW GPU solve
                 * output x[] for a target (Z, lo-ion, shell) to locate the source
                 * of uniform/flat level populations. */
                if (getenv("LUMINA_POP_TRACE") &&
                    nlte->nlte_Z[lo] == (getenv("LUMINA_POP_Z") ? atoi(getenv("LUMINA_POP_Z")) : 8) &&
                    nlte->nlte_ion[lo] == (getenv("LUMINA_POP_ION") ? atoi(getenv("LUMINA_POP_ION")) : 1) &&
                    s == (getenv("LUMINA_POP_SHELL") ? atoi(getenv("LUMINA_POP_SHELL")) : 24)) {
                    double *xr = sol->h_rhs + (size_t)s * N;
                    fprintf(stderr, "[POPTRACE] Z=%d ion=%d s=%d N=%d info=%d fallback=%d "
                            "x[0]=%.4e x[%d]=%.4e x[%d]=%.4e x[%d]=%.4e\n",
                            nlte->nlte_Z[lo], nlte->nlte_ion[lo], s, N, sol->h_info[s],
                            need_fallback, xr[0], N/4, xr[N/4], N/2, xr[N/2],
                            (3*N)/4, xr[(3*N)/4]);
                }
                if (need_fallback) {
                    /* Singular or non-finite: fall back to Boltzmann.
                     * Temperature: T_e when LUMINA_NLTE_FALLBACK_TE=1 (faithful
                     * for the collision-dominated singular case; removes the
                     * spurious inner super-thermal S_l the T_rad fallback injects),
                     * else legacy T_rad. In ion-lock mode, rescale each ion
                     * separately to its nebular total — otherwise the combined
                     * rescale dumps the upper-ion nebular into the lower-ion level
                     * range when over-ionization is severe (Ni II/III bug). */
                    double T_rad = (nlte_fallback_te_enabled() && plasma->T_e &&
                                    plasma->T_e[s] > 0.0)
                                   ? plasma->T_e[s] : plasma->T_rad[s];
                    int Z_nl = nlte->nlte_Z[lo];
                    int gpu_fb_lock_mode = nlte_ion_lock_active(nlte->current_iter) ||
                                            nlte_per_ion_rescale_active() ||
                                            pair_shares_slot;
                    int n_lo_levels = nlte->nlte_ion_level_offset[lo + 1] -
                                      nlte->nlte_ion_level_offset[lo];

                    static int gpu_fb_warn = 0;
                    if (gpu_fb_warn < 16) {
                        fprintf(stderr,
                            "[NLTE-FALLBACK] GPU pair (Z=%d, ions %d/%d, N=%d) shell=%d "
                            "ret=%d info=%d -> Boltzmann@T_rad lock=%d\n",
                            Z_nl, nlte->nlte_ion[lo], nlte->nlte_ion[hi],
                            N, s, ret, sol->h_info[s], gpu_fb_lock_mode);
                        gpu_fb_warn++;
                    }

                    auto find_ip2 = [&](int stage) {
                        for (int j = 0; j < atom->n_ion_pops; j++) {
                            if (atom->ion_pop_Z[j] == Z_nl &&
                                atom->ion_pop_stage[j] == stage) return j;
                        }
                        return -1;
                    };

                    double sum_lo = 0.0, sum_hi = 0.0;
                    for (int i = 0; i < N_fl; i++) {
                        int global = nlte->nlte_to_global_level[lev_start + i];
                        double E = atom->level_energy_eV[global] * EV_TO_ERG;
                        int g = atom->level_g[global];
                        double pop = (double)g * exp(-E / (K_BOLTZMANN * T_rad));
                        nlte->nlte_level_populations[(lev_start + i) * n_shells + s] = pop;
                        if (i < n_lo_levels) sum_lo += pop;
                        else sum_hi += pop;
                    }

                    if (gpu_fb_lock_mode && n_lo_levels > 0 && n_lo_levels < N_fl) {
                        double n_lo_total = 0.0, n_hi_total = 0.0;
                        int ip_lo = find_ip2(nlte->nlte_ion[lo]);
                        int ip_hi = find_ip2(nlte->nlte_ion[hi]);
                        if (ip_lo >= 0) n_lo_total = atom->ion_number_density[ip_lo * n_shells + s];
                        if (ip_hi >= 0) n_hi_total = atom->ion_number_density[ip_hi * n_shells + s];
                        double scale_lo = (sum_lo > 0.0 && n_lo_total > 0.0) ? n_lo_total / sum_lo : 1.0;
                        double scale_hi = (sum_hi > 0.0 && n_hi_total > 0.0) ? n_hi_total / sum_hi : 1.0;
                        for (int i = 0; i < n_lo_levels; i++)
                            nlte->nlte_level_populations[(lev_start + i) * n_shells + s] *= scale_lo;
                        for (int i = n_lo_levels; i < N_fl; i++)
                            nlte->nlte_level_populations[(lev_start + i) * n_shells + s] *= scale_hi;
                    } else {
                        double n_total = nlte_pair_total_density(nlte, atom, plasma,
                                                                  Z_nl, lo, hi, s);
                        double sum = sum_lo + sum_hi;
                        if (sum > 0.0 && n_total > 0.0) {
                            double scale = n_total / sum;
                            for (int i = 0; i < N_fl; i++)
                                nlte->nlte_level_populations[(lev_start + i) * n_shells + s] *= scale;
                        }
                    }
                } else {
                    /* Clamp negatives, then rescale.
                     * Default: combined Σ x_i = n_pair_total.
                     * LUMINA_NLTE_ION_LOCK=1: per-ion rescale (Mihalas-Lucy)
                     * to break the Milne-T_e-vs-T_rad over-ionization trap. */
                    double *x = sol->h_rhs + (size_t)s * N;
                    int Z_nl = nlte->nlte_Z[lo];
                    int gpu_lock_mode = nlte_ion_lock_active(nlte->current_iter) ||
                                         nlte_per_ion_rescale_active() ||
                                         pair_shares_slot;
                    int n_lo_levels = nlte->nlte_ion_level_offset[lo + 1] -
                                      nlte->nlte_ion_level_offset[lo];

                    /* Clamp negatives on the SL solution, then redistribute
                     * each SL population down to its full levels by the local
                     * within-SL Boltzmann fraction. Identity mode: sl==i,
                     * frac==1 => xfl[i]==x[i] (byte-identical to baseline).
                     *
                     * b_k CEILING (LUMINA_NLTE_BK_CEIL=C, default 0=off): at low
                     * n_e the rate matrix is ill-conditioned (bf=0 for low levels
                     * below the cold-field cutoff, bb radiative near-cancels at
                     * J~=B, collisions ∝ n_e -> 0), so the LU solve returns GARBAGE
                     * excited pops (~1e-26, uniform magnitude, alternating sign).
                     * Optical lines are excited->excited, so a uniform-garbage
                     * excited block gives n_u/n_l ~= 1 -> S_l/B = exp(dE/kTe)
                     * super-thermal -> deterministic spectrum too blue. The bare
                     * clamp (x<0 -> 1e-30) leaves the POSITIVE garbage uniform and
                     * the negatives at a uniform floor -> still super-thermal. The
                     * inv_ceil gate only catches absolute inversion (excited>ground),
                     * not a large DEPARTURE at tiny absolute pop. Fix: cap each
                     * level at C * Boltzmann@T_e relative to its ion ground (b_k<=C).
                     * Literature bounds physical SN O II/Fe II departures at b~1-50,
                     * so C~100 removes the 1e20+ garbage while preserving real NLTE
                     * departures; capped excited follow C*Boltzmann (declining) ->
                     * excited->excited n_u/n_l -> Boltzmann ratio -> S_l -> B. */
                    {
                        static int bkc_init = 0; static double bk_ceil = 0.0;
                        if (!bkc_init) {
                            const char *e = getenv("LUMINA_NLTE_BK_CEIL");
                            bk_ceil = e ? atof(e) : 0.0;
                            if (bk_ceil < 0.0) bk_ceil = 0.0;
                            bkc_init = 1;
                        }
                        for (int i = 0; i < N; i++) {
                            if (x[i] < 0.0) x[i] = 1e-30;
                        }
                        if (bk_ceil > 0.0) {
                            double T_e_s = plasma->T_e ? plasma->T_e[s] : plasma->T_rad[s];
                            double kTe = K_BOLTZMANN * (T_e_s > 0.0 ? T_e_s : 1.0);
                            double xg_lo = x[0];
                            double xg_hi = (n_lo_super < N) ? x[n_lo_super] : 0.0;
                            int gg_lo = atom->level_g[nlte->super_anchor_global[super_start]];
                            double Eg_lo = atom->level_energy_eV[nlte->super_anchor_global[super_start]] * EV_TO_ERG;
                            int gg_hi = (n_lo_super < N) ? atom->level_g[nlte->super_anchor_global[super_start + n_lo_super]] : 1;
                            double Eg_hi = (n_lo_super < N) ? atom->level_energy_eV[nlte->super_anchor_global[super_start + n_lo_super]] * EV_TO_ERG : 0.0;
                            for (int i = 0; i < N; i++) {
                                int is_lo = (i < n_lo_super);
                                double xg = is_lo ? xg_lo : xg_hi;
                                if (xg <= 0.0) continue;
                                int gi = atom->level_g[nlte->super_anchor_global[super_start + i]];
                                double Ei = atom->level_energy_eV[nlte->super_anchor_global[super_start + i]] * EV_TO_ERG;
                                int gg = is_lo ? gg_lo : gg_hi;
                                double Eg = is_lo ? Eg_lo : Eg_hi;
                                double boltz = xg * ((double)gi / (double)(gg > 0 ? gg : 1)) *
                                               exp(-(Ei - Eg) / kTe);
                                double cap = bk_ceil * boltz;
                                if (cap > 0.0 && x[i] > cap) x[i] = cap;
                            }
                        }
                    }
                    double *xfl = (double *)malloc((size_t)N_fl * sizeof(double));
                    for (int i = 0; i < N_fl; i++) {
                        int sl = nlte->fl_to_super[lev_start + i] - super_start;
                        double frac = nlte->within_sl_frac[(size_t)(lev_start + i) * n_shells + s];
                        xfl[i] = x[sl] * frac;
                    }

                    auto find_ip = [&](int stage) {
                        for (int j = 0; j < atom->n_ion_pops; j++) {
                            if (atom->ion_pop_Z[j] == Z_nl &&
                                atom->ion_pop_stage[j] == stage) return j;
                        }
                        return -1;
                    };

                    if (gpu_lock_mode && n_lo_levels > 0 && n_lo_levels < N_fl) {
                        double n_lo_total = 0.0, n_hi_total = 0.0;
                        int ip_lo = find_ip(nlte->nlte_ion[lo]);
                        int ip_hi = find_ip(nlte->nlte_ion[hi]);
                        if (ip_lo >= 0)
                            n_lo_total = atom->ion_number_density[ip_lo * n_shells + s];
                        if (ip_hi >= 0)
                            n_hi_total = atom->ion_number_density[ip_hi * n_shells + s];
                        double sum_lo = 0.0, sum_hi = 0.0;
                        for (int i = 0; i < n_lo_levels; i++) sum_lo += xfl[i];
                        for (int i = n_lo_levels; i < N_fl; i++) sum_hi += xfl[i];
                        double scale_lo = (sum_lo > 0.0 && n_lo_total > 0.0) ? n_lo_total / sum_lo : 1.0;
                        double scale_hi = (sum_hi > 0.0 && n_hi_total > 0.0) ? n_hi_total / sum_hi : 1.0;
                        for (int i = 0; i < n_lo_levels; i++)
                            nlte->nlte_level_populations[(lev_start + i) * n_shells + s] = xfl[i] * scale_lo;
                        for (int i = n_lo_levels; i < N_fl; i++)
                            nlte->nlte_level_populations[(lev_start + i) * n_shells + s] = xfl[i] * scale_hi;
                    } else {
                        double n_total = nlte_pair_total_density(nlte, atom, plasma,
                                                                  Z_nl, lo, hi, s);
                        double sum = 0.0;
                        for (int i = 0; i < N_fl; i++) sum += xfl[i];
                        double scale = (sum > 0.0 && n_total > 0.0) ? n_total / sum : 1.0;
                        for (int i = 0; i < N_fl; i++) {
                            nlte->nlte_level_populations[(lev_start + i) * n_shells + s] =
                                xfl[i] * scale;
                        }
                    }
                    if (getenv("LUMINA_POP_TRACE") && nlte->nlte_Z[lo] == 8 &&
                        nlte->nlte_ion[lo] == 1 && s == 24) {
                        fprintf(stderr, "[POPSTORED] s=%d STORED ground=%.4e mid170=%.4e high300=%.4e\n",
                            s, nlte->nlte_level_populations[(lev_start + 0) * n_shells + s],
                            nlte->nlte_level_populations[(lev_start + 170) * n_shells + s],
                            nlte->nlte_level_populations[(lev_start + 300) * n_shells + s]);
                    }
                    free(xfl);
                }
            }

            /* C1 restore: put the shared lo-ion (O II) block back to the value
             * the prior overlapping pair set, so this pair's solve only updates
             * the hi-ion (O III) and the O I<->O II conservation survives. */
            if (saved_lo) {
                size_t n_save = (size_t)(saved_lev_e - saved_lev_s) * n_shells;
                memcpy(&nlte->nlte_level_populations[(size_t)saved_lev_s * n_shells],
                       saved_lo, n_save * sizeof(double));
                free(saved_lo);
            }
        }

        /* Apply damping for iter >= 1 */
        if (ce_iter > 0) {
            for (size_t i = 0; i < pop_size; i++) {
                double n_new = nlte->nlte_level_populations[i];
                double n_old = old_pops[i];
                nlte->nlte_level_populations[i] = n_old +
                    ce_damping * (n_new - n_old);
            }
        }

        /* Convergence: max relative change of ion totals */
        double max_rel_change = 0.0;
        if (ce_iter == 0) {
            int has_prior = 0;
            for (int k = 0; k < n_ion_totals; k++) {
                if (old_ion_totals[k] > 1.0) { has_prior = 1; break; }
            }
            if (!has_prior) {
                printf("    CE iter %d: first solve (no prior populations)\n",
                       ce_iter + 1);
                continue;
            }
        }

        for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
            int lev_s = nlte->nlte_ion_level_offset[ii];
            int lev_e = nlte->nlte_ion_level_offset[ii + 1];
            for (int s = 0; s < n_shells; s++) {
                double new_total = 0.0;
                for (int l = lev_s; l < lev_e; l++)
                    new_total += nlte->nlte_level_populations[l * n_shells + s];
                double old_total = old_ion_totals[ii * n_shells + s];
                if (old_total > 1.0) {
                    double rel = fabs(new_total - old_total) / old_total;
                    if (rel > max_rel_change) max_rel_change = rel;
                }
            }
        }

        printf("    CE iter %d: max_ion_rel_change = %.2e\n",
               ce_iter + 1, max_rel_change);

        if (max_rel_change < ce_threshold) {
            printf("    CE converged in %d iterations\n", ce_iter + 1);
            break;
        }
    }
    free(old_pops);
    free(old_ion_totals);

    printf("  [NLTE-PROF] assemble: %.2f s (%.1f%%)   GPU solve: %.2f s (%.1f%%)\n",
           t_assemble_total,
           100.0 * t_assemble_total / (t_assemble_total + t_solve_total + 1e-9),
           t_solve_total,
           100.0 * t_solve_total / (t_assemble_total + t_solve_total + 1e-9));

    /* Print ion pair summaries */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0], hi = pairs[p][1];
        int N = nlte->nlte_ion_level_offset[hi + 1] - nlte->nlte_ion_level_offset[lo];
        printf("    %s (%d levels): done [GPU]\n", names[p], N);
    }

    /* Probe-B fix (task #29): feed the NLTE-solved ion stage back into opacity
     * (rebuilds bulk tau from corrected ion_number_density) before the per-line
     * NLTE override below. No-op unless LUMINA_NLTE_OPACITY_IONSTAGE=1. */
    nlte_writeback_ion_stage(nlte, atom, plasma, opacity, time_explosion,
                             n_shells, pairs, n_pairs);

    /* Update tau_sobolev for NLTE lines (same as CPU path) */
    printf("  [NLTE-GPU] Updating tau_sobolev from NLTE populations...\n");
    {
        /* Per-Z skip mask via LUMINA_NLTE_SKIP_Z (comma list). */
        static int gpu_skip_z[100];
        static int gpu_skip_init = 0;
        if (!gpu_skip_init) {
            gpu_skip_init = 1;
            const char *e = getenv("LUMINA_NLTE_SKIP_Z");
            if (e && *e) {
                char buf[256]; strncpy(buf, e, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
                char *tok = strtok(buf, ", \t");
                while (tok) { int z = atoi(tok); if (z>0 && z<100) gpu_skip_z[z]=1; tok = strtok(NULL, ", \t"); }
                printf("  [NLTE-GPU] LUMINA_NLTE_SKIP_Z active (nebular tau kept): ");
                for (int i=1;i<100;i++) if (gpu_skip_z[i]) printf("%d ", i);
                printf("\n");
            }
        }
        int n_lines = opacity->n_lines;
        for (int line = 0; line < n_lines; line++) {
            int ion_idx = nlte->nlte_line_map[line];
            if (ion_idx < 0) continue;

            int Z     = atom->line_atomic_number[line];
            if (Z > 0 && Z < 100 && gpu_skip_z[Z]) continue; /* keep nebular tau */
            int ion_s = atom->line_ion_number[line];
            double f_lu   = atom->line_f_lu[line];
            double lam_cm = atom->line_wavelength_cm[line];

            int ip = -1;
            for (int j = 0; j < atom->n_ion_pops; j++) {
                if (atom->ion_pop_Z[j] == Z && atom->ion_pop_stage[j] == ion_s) {
                    ip = j; break;
                }
            }
            if (ip < 0) continue;
            int lev_base = atom->level_offset[ip];
            int lev_top  = atom->level_offset[ip + 1];

            int lower_global = -1, upper_global = -1;
            for (int l = lev_base; l < lev_top; l++) {
                if (atom->level_num[l] == atom->line_level_lower[line]) lower_global = l;
                if (atom->level_num[l] == atom->line_level_upper[line]) upper_global = l;
                if (lower_global >= 0 && upper_global >= 0) break;
            }
            if (lower_global < 0 || upper_global < 0) continue;

            int nlte_lo = nlte->global_to_nlte_level[lower_global];
            int nlte_up = nlte->global_to_nlte_level[upper_global];
            if (nlte_lo < 0 || nlte_up < 0) continue;

            int g_lo = atom->level_g[lower_global];
            int g_up = atom->level_g[upper_global];

            /* CMF NLTE line source S_l = (2hv^3/c^2)/(g_u n_l/(g_l n_u) - 1):
             * MUST be written here too — this GPU tau update was the only
             * writer-path omission (audit 2026-06-12); line_source_S stayed
             * empty so cmfgen_assemble's B(T_e) fallback fired for 100% of
             * the forest in every GPU run (maximal local thermalization). */
            double nu_l = C_SPEED_OF_LIGHT / lam_cm;
            double src_prefac = 2.0 * H_PLANCK * nu_l * nu_l * nu_l /
                                (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
            for (int s = 0; s < n_shells; s++) {
                double n_lower = nlte->nlte_level_populations[nlte_lo * n_shells + s];
                double n_upper = nlte->nlte_level_populations[nlte_up * n_shells + s];
                double stim_corr = 1.0;
                if (n_lower > 0.0 && n_upper > 0.0 && g_lo > 0 && g_up > 0) {
                    stim_corr = 1.0 - ((double)g_lo * n_upper) / ((double)g_up * n_lower);
                    if (stim_corr < 0.0) stim_corr = 0.0;
                }
                double tau_nlte = SOBOLEV_COEFF * f_lu * lam_cm * time_explosion *
                                  n_lower * stim_corr;
                if (!(tau_nlte > 1e-100)) tau_nlte = 1e-100;  /* NaN-catching */
                opacity->tau_sobolev[line * n_shells + s] = tau_nlte;

                double S_l = 0.0;
                if (n_lower > 0.0 && n_upper > 0.0 && g_lo > 0 && g_up > 0) {
                    double ratio = ((double)g_up * n_lower) /
                                   ((double)g_lo * n_upper);
                    double denom = ratio - 1.0;
                    if (denom > 1e-30) S_l = src_prefac / denom;
                }
                if (opacity->line_source_S)
                    opacity->line_source_S[line * n_shells + s] = S_l;
            }
        }
    }

    /* Print diagnostics */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0];
        int lev_s = nlte->nlte_ion_level_offset[lo];
        int lev_e = nlte->nlte_ion_level_offset[lo + 1];
        double sum_nlte = 0.0;
        for (int l = lev_s; l < lev_e; l++)
            sum_nlte += nlte->nlte_level_populations[l * n_shells + 0];

        int Z_nl = nlte->nlte_Z[lo];
        int ip = -1;
        for (int j = 0; j < atom->n_ion_pops; j++) {
            if (atom->ion_pop_Z[j] == Z_nl && atom->ion_pop_stage[j] == nlte->nlte_ion[lo]) {
                ip = j; break;
            }
        }
        double n_neb = (ip >= 0) ? atom->ion_number_density[ip * n_shells + 0] : 0.0;
        printf("    %s II shell 0: NLTE n_total=%.3e, nebular n_ion=%.3e\n",
               names[p], sum_nlte, n_neb);
    }

    /* [NLTE-DUMP] mirror of the CPU-path dump in nlte_solve_all() */
    {
        const char *env = getenv("LUMINA_NLTE_LEVEL_DUMP");
        if (env && env[0] == '1') {
            static int dump_counter = 0;
            char path[256];
            snprintf(path, sizeof(path),
                     "nlte_levels_iter%03d.csv", dump_counter++);
            FILE *fp = fopen(path, "w");
            if (!fp) {
                fprintf(stderr, "[NLTE-DUMP] failed to open %s\n", path);
            } else {
                fprintf(fp, "Z,ion,shell,level_idx,global_idx,E_eV,g,n_pop,T_e,T_rad,W,n_ion_total\n");
                for (int ii = 0; ii < nlte->n_nlte_ions; ii++) {
                    int Zv  = nlte->nlte_Z[ii];
                    int ion = nlte->nlte_ion[ii];
                    int lev_s = nlte->nlte_ion_level_offset[ii];
                    int lev_e = nlte->nlte_ion_level_offset[ii + 1];
                    int ip = -1;
                    for (int j = 0; j < atom->n_ion_pops; j++) {
                        if (atom->ion_pop_Z[j] == Zv && atom->ion_pop_stage[j] == ion) {
                            ip = j; break;
                        }
                    }
                    for (int l = lev_s; l < lev_e; l++) {
                        int gi = nlte->nlte_to_global_level[l];
                        double E_eV = atom->level_energy_eV[gi];
                        int gw = atom->level_g[gi];
                        int local_l = l - lev_s;
                        for (int s = 0; s < n_shells; s++) {
                            double n_pop = nlte->nlte_level_populations[
                                (size_t)l * n_shells + s];
                            double T_e   = plasma->T_e ? plasma->T_e[s] :
                                           plasma->T_e_T_rad_ratio * plasma->T_rad[s];
                            double T_rad = plasma->T_rad[s];
                            double W     = plasma->W[s];
                            double n_ion = (ip >= 0) ?
                                atom->ion_number_density[ip * n_shells + s] : 0.0;
                            fprintf(fp, "%d,%d,%d,%d,%d,%.6f,%d,%.6e,%.2f,%.2f,%.6e,%.6e\n",
                                    Zv, ion, s, local_l, gi, E_eV, gw, n_pop,
                                    T_e, T_rad, W, n_ion);
                        }
                    }
                }
                fclose(fp);
                printf("  [NLTE-DUMP] wrote %s\n", path);
            }
        }
    }
}

/* ============================================================ */
/* Phase 6 - Step 1: cuda_upload — copy data to GPU             */
/* ============================================================ */
static void cuda_upload(CudaDeviceData *dev, Geometry *geo,
                         OpacityState *opacity) {
    int ns = geo->n_shells;                /* Phase 6 - Step 1 */
    int nl = opacity->n_lines;             /* Phase 6 - Step 1 */
    int nt = opacity->n_macro_transitions; /* Phase 6 - Step 1 */
    int nlev = opacity->n_macro_levels;    /* Phase 6 - Step 1 */

    /* Phase 6 - Step 1: Upload read-only arrays */
    CUDA_CHECK(cudaMemcpy(dev->d_line_list_nu, opacity->line_list_nu,               /* Phase 6 - Step 1 */
               nl * sizeof(double), cudaMemcpyHostToDevice));                        /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_tau_sobolev, opacity->tau_sobolev,                  /* Phase 6 - Step 1 */
               (size_t)nl * ns * sizeof(double), cudaMemcpyHostToDevice));           /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_electron_density, opacity->electron_density,        /* Phase 6 - Step 1 */
               ns * sizeof(double), cudaMemcpyHostToDevice));                        /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_transition_probabilities,                           /* Phase 6 - Step 1 */
               opacity->transition_probabilities,                                    /* Phase 6 - Step 1 */
               (size_t)nt * ns * sizeof(double), cudaMemcpyHostToDevice));           /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_macro_block_references,                             /* Phase 6 - Step 1 */
               opacity->macro_block_references,                                      /* Phase 6 - Step 1 */
               (nlev + 1) * sizeof(int), cudaMemcpyHostToDevice));                   /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_transition_type, opacity->transition_type,          /* Phase 6 - Step 1 */
               nt * sizeof(int), cudaMemcpyHostToDevice));                           /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_destination_level_id,                               /* Phase 6 - Step 1 */
               opacity->destination_level_id,                                        /* Phase 6 - Step 1 */
               nt * sizeof(int), cudaMemcpyHostToDevice));                           /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_transition_line_id, opacity->transition_line_id,    /* Phase 6 - Step 1 */
               nt * sizeof(int), cudaMemcpyHostToDevice));                           /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_line2macro_level_upper,                             /* Phase 6 - Step 1 */
               opacity->line2macro_level_upper,                                      /* Phase 6 - Step 1 */
               nl * sizeof(int), cudaMemcpyHostToDevice));                           /* Phase 6 - Step 1 */

    /* k-packet: upload the collisional deactivation prob + per-shell re-excitation
     * CDF (host builds them in compute_transition_probabilities when LUMINA_KPACKET
     * is on). Guarded on the device arrays being allocated AND the host tables
     * being present. */
    if (dev->d_p_kpacket && opacity->p_kpacket) {
        CUDA_CHECK(cudaMemcpy(dev->d_p_kpacket, opacity->p_kpacket,
                   (size_t)nlev * ns * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (dev->d_kpacket_cdf && opacity->kpacket_cdf) {
        CUDA_CHECK(cudaMemcpy(dev->d_kpacket_cdf, opacity->kpacket_cdf,
                   (size_t)ns * nlev * sizeof(double), cudaMemcpyHostToDevice));
    }

    /* Phase 6 - Step 1: Upload geometry */
    CUDA_CHECK(cudaMemcpy(dev->d_r_inner, geo->r_inner,                              /* Phase 6 - Step 1 */
               ns * sizeof(double), cudaMemcpyHostToDevice));                        /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(dev->d_r_outer, geo->r_outer,                              /* Phase 6 - Step 1 */
               ns * sizeof(double), cudaMemcpyHostToDevice));                        /* Phase 6 - Step 1 */
}

/* ============================================================ */
/* Phase 6 - Step 1: cuda_reset_estimators — zero GPU estimators */
/* ============================================================ */
static void cuda_reset_estimators(CudaDeviceData *dev, int n_shells) {
    CUDA_CHECK(cudaMemset(dev->d_j_estimator, 0, n_shells * sizeof(double)));        /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemset(dev->d_nu_bar_estimator, 0, n_shells * sizeof(double)));   /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemset(dev->d_n_escaped, 0, sizeof(int64_t)));                    /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemset(dev->d_n_reabsorbed, 0, sizeof(int64_t)));                 /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemset(dev->d_virtual_spectrum, 0, VSPEC_N_BINS * sizeof(double)));
    if (dev->d_j_nu_estimator) {
        CUDA_CHECK(cudaMemset(dev->d_j_nu_estimator, 0,
                   (size_t)n_shells * NLTE_N_FREQ_BINS * sizeof(double)));
    }
}

/* ============================================================ */
/* [MA-FATE] Device-side fate histogram (declared here so host   */
/* helpers can reference it; device-side helpers below).         */
/* ============================================================ */
__device__ unsigned long long d_ma_fate_hist[MA_FATE_NBANDS * MA_FATE_NBANDS];

/* [MA-CYCLE] Device-side cycle histogram (mirrors host g_ma_cycle_hist). */
__device__ unsigned long long d_ma_cycle_hist[MA_CYCLE_BINS];

/* [EPS-UV] Probability that a UV-entry macro-atom call is replaced by
 * Planck(T_rad) thermalization (BF-style). Set from LUMINA_EPS_UV. */
__device__ double d_eps_uv = 0.0;

void cuda_set_eps_uv(double v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_eps_uv, &v, sizeof(double)));
}

/* [KPACKET] Collisional / k-packet thermal pool. When enabled, at each macro-
 * atom level the cascade visits, with probability d_kpacket_p[lev*ns+shell] the
 * activation deactivates collisionally into a k-packet (energy → free-electron
 * thermal pool) and immediately re-excites a macro-atom at a level drawn from
 * the per-shell thermal re-excitation CDF d_kpacket_cdf_g[shell*nlev+lev]. This
 * couples the line cascade to T_e and re-emits near the local Planck peak,
 * curing the purely-radiative down-cascade over-redshift. Tables built host-side
 * (compute_transition_probabilities) from van-Regemorter/Axelrod collisional
 * rates; device pointers + enable flag set via cuda_set_kpacket(). */
__device__ const double *d_kpacket_p     = NULL;   /* [nlev*ns] P(coll. deact.) */
__device__ const double *d_kpacket_cdf_g = NULL;   /* [ns*nlev] re-excite CDF    */
__device__ int    d_kpacket_enabled = 0;
__device__ int    d_kpacket_n_levels = 0;
__device__ unsigned long long d_kpacket_count = 0; /* diagnostic event counter   */

void cuda_set_kpacket(const double *d_p, const double *d_cdf, int n_levels, int enabled) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_kpacket_p,     &d_p,      sizeof(const double *)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kpacket_cdf_g, &d_cdf,    sizeof(const double *)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kpacket_n_levels, &n_levels, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kpacket_enabled,  &enabled,  sizeof(int)));
}

/* [EPS-IR] Probability that a NIR-entry (lam > 7000 A) macro-atom call is
 * replaced by Planck(T_rad) thermalization. Counterpart to LUMINA_EPS_UV;
 * intended to break the IR over-thermalization loop that emerges with the
 * 3.4M-transition carsus atomic data, where weak NIR forbidden lines vastly
 * outnumber strong UV resonance lines and trap energy in the IR cascade. */
__device__ double d_eps_ir = 0.0;

void cuda_set_eps_ir(double v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_eps_ir, &v, sizeof(double)));
}

/* [H2 EPS-UV red-only] If set, move the EPS_UV gate to post-cascade and
 * only fire when the cascade exit lands in [5500,10000)Å (red+NIR1). This
 * preserves the cascade's useful UV→UV/blue downbranching paths and only
 * suppresses the harmful UV→red conversion. */
__device__ int d_eps_uv_red_only = 0;

void cuda_set_eps_uv_red_only(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_eps_uv_red_only, &v, sizeof(int)));
}

/* [EPS-UV 2STEP] True 2-step UV→opt→red cascade. When the UV gate fires,
 * instead of full Planck(T_rad) thermalization the packet is re-emitted into
 * an *optical band* [LO,HI]Å (rejection-sampled Planck(T_rad)). Normal
 * transport then resumes — red flux emerges from optical line cascades and
 * carries proper P-Cygni shape (not a thermal continuum). Closes the
 * "flux winner ≠ shape winner" gap observed in H1/H2. */
__device__ int    d_eps_uv_2step = 0;
__device__ double d_eps_uv_2step_nu_lo = 0.0;  /* low-nu  edge (= high-λ) */
__device__ double d_eps_uv_2step_nu_hi = 0.0;  /* high-nu edge (= low-λ)  */

void cuda_set_eps_uv_2step(int on, double lo_A, double hi_A) {
    double nu_hi_local = C_SPEED_OF_LIGHT / (lo_A * 1.0e-8);
    double nu_lo_local = C_SPEED_OF_LIGHT / (hi_A * 1.0e-8);
    CUDA_CHECK(cudaMemcpyToSymbol(d_eps_uv_2step,       &on,           sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_eps_uv_2step_nu_lo, &nu_lo_local,  sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_eps_uv_2step_nu_hi, &nu_hi_local,  sizeof(double)));
}

/* [CAP] Configurable per-packet interaction cap + cap-hit counter. The macro-
 * atom (fluorescence) line-interaction mode is non-monotonic (a packet may be
 * re-emitted at a bluer line) and therefore UNBOUNDED in interaction count,
 * unlike coherent scatter which is monotonic-redward and bounded by ~N_lines.
 * Through DDC15's dense iron-curtain nebular opacity (~2.7e5 lines with tau>1)
 * this drives the per-packet transport loop toward the hard 100000 cap and
 * stalls the iteration ("iter3 hang"). LUMINA_MAX_INTERACTIONS lets us bound
 * the loop and count how many packets hit it, so the run completes and the
 * cap-hit fraction quantifies the truncation. */
__constant__ int d_max_interactions = 100000;
__device__ unsigned long long d_n_capped_dev = 0;

/* [CAP-SHELL] diagnostic: histogram of cap-hits indexed by the packet's current
 * shell when it hit the cap. Packets are born at shell 0 (inner boundary); a
 * higher index = closer to the surface. If scissored packets pile up at low
 * shell indices they are still trapped DEEP (survivorship bias: only shallow/
 * low-tau packets escape). Fixed ceiling 256 >> n_shells(49). Pure diagnostic,
 * no effect on transport/RNG/spectrum. */
#define CAP_SHELL_MAX 256
__device__ unsigned long long d_capped_by_shell[CAP_SHELL_MAX];

/* #5 fix (2026-06-04): the interaction cap was counting shell-boundary crossings
 * and diffuse-BC returns as "interactions", and DROPPED the packet's energy on
 * cap-hit (removing it from the spectrum -> L_emitted deficit -> T_inner over-
 * boost). Two independent, gated corrections:
 *  d_cap_real_only=1   : only LINE/CONTINUUM events count toward d_max_interactions;
 *                        boundary crossings/returns are bounded by d_max_total_steps.
 *  d_cap_force_escape=1 : on cap-hit, conserve energy by binning the packet at its
 *                        current (nu,energy) instead of dropping it.
 * Defaults 0 reproduce the legacy drop behaviour for a clean A/B. */
__constant__ int d_cap_real_only   = 0;
__constant__ int d_cap_force_escape = 0;
__constant__ int d_max_total_steps = 2000000; /* absolute safety ceiling on while-iterations */

/* Macro-atom INTERNAL cascade cap: max internal (up/down) jumps within a single
 * macro-atom activation before forced deactivation. Default 5000 (the old hard
 * constant). Diagnostic LUMINA_MA_INTERNAL_CAP lets us shorten the cascade to
 * test whether cumulative cascade depth (not up-pump magnitude) drives the
 * over-redshift. NOTE: a small cap is a non-physical truncation, useful only as
 * a discriminating probe, not a fix. */
__constant__ int d_ma_internal_cap = 5000;

/* Task #27 Phase 0: energy-weighted fate accounting. Reabsorbed (fell back
 * through the photosphere) and truncated (hit the interaction cap) carry the
 * deficit L_req - L_emitted; splitting it by energy localizes the T_inner
 * overshoot's cause (back-scatter trapping vs packet loss). Normalized units
 * (multiply by L_inner for physical erg/s). */
__device__ double d_E_reabsorbed_dev = 0.0;
__device__ double d_E_truncated_dev = 0.0;

/* Task #27 Phase 2a: luminosity-conserving diffusive inner boundary. When 1,
 * a packet that falls back through the photosphere (next_shell < 0) is NOT
 * killed; the core (a thermal reservoir at T_inner) re-emits it outward with
 * its energy bundle preserved and frequency re-thermalized to Planck(T_inner).
 * This returns the back-scatter-trapped energy that the absorbing hard sphere
 * would otherwise lose, so the self-pin no longer over-heats T_inner to close
 * the deficit. d_n_returned_dev counts re-emission events for diagnostics. */
__device__ int d_diffuse_inner_bc = 0;
__device__ unsigned long long d_n_returned_dev = 0;

void cuda_set_diffuse_inner_bc(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_diffuse_inner_bc, &v, sizeof(int)));
}

unsigned long long cuda_n_returned_get(void) {
    unsigned long long v = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&v, d_n_returned_dev, sizeof(unsigned long long)));
    return v;
}

void cuda_set_max_interactions(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_interactions, &v, sizeof(int)));
}

void cuda_set_cap_real_only(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_cap_real_only, &v, sizeof(int)));
}

void cuda_set_cap_force_escape(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_cap_force_escape, &v, sizeof(int)));
}

void cuda_set_max_total_steps(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_total_steps, &v, sizeof(int)));
}

void cuda_set_ma_internal_cap(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_ma_internal_cap, &v, sizeof(int)));
}

void cuda_n_capped_reset(void) {
    unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_capped_dev, &zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_returned_dev, &zero, sizeof(unsigned long long)));
    double dzero = 0.0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_E_reabsorbed_dev, &dzero, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_E_truncated_dev, &dzero, sizeof(double)));
    unsigned long long zhist[CAP_SHELL_MAX] = {0};
    CUDA_CHECK(cudaMemcpyToSymbol(d_capped_by_shell, zhist, sizeof(zhist)));
}

unsigned long long cuda_n_capped_get(void) {
    unsigned long long v = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&v, d_n_capped_dev, sizeof(unsigned long long)));
    return v;
}

/* [CAP-SHELL] download the per-shell cap-hit histogram (n entries) to host. */
void cuda_capped_by_shell_get(unsigned long long *host, int n) {
    unsigned long long all[CAP_SHELL_MAX];
    CUDA_CHECK(cudaMemcpyFromSymbol(all, d_capped_by_shell, sizeof(all)));
    for (int i = 0; i < n && i < CAP_SHELL_MAX; i++) host[i] = all[i];
}

double cuda_E_reabsorbed_get(void) {
    double v = 0.0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&v, d_E_reabsorbed_dev, sizeof(double)));
    return v;
}

double cuda_E_truncated_get(void) {
    double v = 0.0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&v, d_E_truncated_dev, sizeof(double)));
    return v;
}

void cuda_ma_fate_reset(void) {
    unsigned long long zero[MA_FATE_NBANDS * MA_FATE_NBANDS] = {0};
    CUDA_CHECK(cudaMemcpyToSymbol(d_ma_fate_hist, zero, sizeof(zero)));
}

void cuda_ma_fate_download_and_aggregate(void) {
    unsigned long long host_hist[MA_FATE_NBANDS * MA_FATE_NBANDS] = {0};
    CUDA_CHECK(cudaMemcpyFromSymbol(host_hist, d_ma_fate_hist, sizeof(host_hist)));
    macro_atom_fate_add_counts(host_hist);
}

/* [H3] forward decl needed: d_ma_fate_band_from_nu is defined later. */
__device__ __forceinline__ int d_ma_fate_band_from_nu(double nu_comov);

/* [H3] Per-(Z, ion, entry_band, exit_band) attribution histogram. */
__device__ unsigned long long d_ma_fate_zihist[MA_FATE_ZI_LEN];
__device__ int d_ma_fate_zi_enabled = 0;

void cuda_set_ma_fate_zi_enabled(int v) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_ma_fate_zi_enabled, &v, sizeof(int)));
}
void cuda_ma_fate_zi_reset(void) {
    unsigned long long *zero = (unsigned long long*)calloc(MA_FATE_ZI_LEN, sizeof(unsigned long long));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ma_fate_zihist, zero,
        sizeof(unsigned long long) * MA_FATE_ZI_LEN));
    free(zero);
}
void cuda_ma_fate_zi_download_and_aggregate(void) {
    unsigned long long *host = (unsigned long long*)calloc(MA_FATE_ZI_LEN, sizeof(unsigned long long));
    CUDA_CHECK(cudaMemcpyFromSymbol(host, d_ma_fate_zihist,
        sizeof(unsigned long long) * MA_FATE_ZI_LEN));
    macro_atom_fate_zi_add_counts(host);
    free(host);
}

__device__ __forceinline__ int d_ma_zi_z_index(int Z) {
    switch (Z) {
        case 6:  return 0;  case 8:  return 1;  case 12: return 2;
        case 13: return 3;  case 14: return 4;  case 16: return 5;
        case 20: return 6;  case 21: return 7;  case 22: return 8;
        case 23: return 9;  case 24: return 10; case 25: return 11;
        case 26: return 12; case 27: return 13; case 28: return 14;
        default: return -1;
    }
}

__device__ __forceinline__
void d_ma_fate_record_zi(double entry_nu, double exit_nu, int eb_hint,
                          int Z, int ion) {
    int eb = (eb_hint >= 0) ? eb_hint : d_ma_fate_band_from_nu(entry_nu);
    int xb = d_ma_fate_band_from_nu(exit_nu);
    atomicAdd(&d_ma_fate_hist[eb * MA_FATE_NBANDS + xb], 1ULL);
    if (d_ma_fate_zi_enabled) {
        int zi = d_ma_zi_z_index(Z);
        if (zi >= 0) {
            int io = (ion < 0) ? 0 : (ion >= MA_FATE_NION ? MA_FATE_NION - 1 : ion);
            int idx = ((zi * MA_FATE_NION + io) * MA_FATE_NBANDS + eb) *
                      MA_FATE_NBANDS + xb;
            atomicAdd(&d_ma_fate_zihist[idx], 1ULL);
        }
    }
}

void cuda_ma_cycle_reset(void) {
    unsigned long long zero[MA_CYCLE_BINS] = {0};
    CUDA_CHECK(cudaMemcpyToSymbol(d_ma_cycle_hist, zero, sizeof(zero)));
}

void cuda_ma_cycle_download_and_aggregate(void) {
    unsigned long long host_hist[MA_CYCLE_BINS] = {0};
    CUDA_CHECK(cudaMemcpyFromSymbol(host_hist, d_ma_cycle_hist, sizeof(host_hist)));
    macro_atom_cycle_add_counts(host_hist);
}

/* [KPACKET] per-iteration collisional-deactivation event counter. */
void cuda_kpacket_count_reset(void) {
    unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_kpacket_count, &zero, sizeof(zero)));
}

unsigned long long cuda_kpacket_count_get(void) {
    unsigned long long n = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&n, d_kpacket_count, sizeof(n)));
    return n;
}

/* ============================================================ */
/* Phase 6 - Step 1: cuda_download — download results from GPU  */
/* ============================================================ */
static void cuda_download_estimators(CudaDeviceData *dev, double *j_est,
                                      double *nu_bar_est, int n_shells) {
    CUDA_CHECK(cudaMemcpy(j_est, dev->d_j_estimator,                                 /* Phase 6 - Step 1 */
               n_shells * sizeof(double), cudaMemcpyDeviceToHost));                  /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMemcpy(nu_bar_est, dev->d_nu_bar_estimator,                       /* Phase 6 - Step 1 */
               n_shells * sizeof(double), cudaMemcpyDeviceToHost));                  /* Phase 6 - Step 1 */
}

/* ============================================================ */
/* Phase 6 - Step 1: cuda_free — release GPU memory             */
/* ============================================================ */
static void cuda_free(CudaDeviceData *dev) {
    cudaFree(dev->d_line_list_nu);              /* Phase 6 - Step 1 */
    cudaFree(dev->d_tau_sobolev);               /* Phase 6 - Step 1 */
    cudaFree(dev->d_electron_density);          /* Phase 6 - Step 1 */
    cudaFree(dev->d_transition_probabilities);  /* Phase 6 - Step 1 */
    cudaFree(dev->d_macro_block_references);    /* Phase 6 - Step 1 */
    cudaFree(dev->d_transition_type);           /* Phase 6 - Step 1 */
    cudaFree(dev->d_destination_level_id);      /* Phase 6 - Step 1 */
    cudaFree(dev->d_transition_line_id);        /* Phase 6 - Step 1 */
    cudaFree(dev->d_line2macro_level_upper);    /* Phase 6 - Step 1 */
    cudaFree(dev->d_line_atomic_number);
    cudaFree(dev->d_line_ion_number);
    cudaFree(dev->d_p_kpacket);                  /* [KPACKET] (NULL-safe)      */
    cudaFree(dev->d_kpacket_cdf);               /* [KPACKET] (NULL-safe)      */
    cudaFree(dev->d_r_inner);                   /* Phase 6 - Step 1 */
    cudaFree(dev->d_r_outer);                   /* Phase 6 - Step 1 */
    cudaFree(dev->d_j_estimator);               /* Phase 6 - Step 1 */
    cudaFree(dev->d_nu_bar_estimator);          /* Phase 6 - Step 1 */
    cudaFree(dev->d_rng_states);                /* Phase 6 - Step 1 */
    cudaFree(dev->d_escaped_nu);                /* Phase 6 - Step 1 */
    cudaFree(dev->d_escaped_energy);            /* Phase 6 - Step 1 */
    cudaFree(dev->d_escaped_flag);              /* Phase 6 - Step 1 */
    cudaFree(dev->d_n_escaped);                 /* Phase 6 - Step 1 */
    cudaFree(dev->d_n_reabsorbed);              /* Phase 6 - Step 1 */
    cudaFree(dev->d_escaped_r);                 /* Rotation mode */
    cudaFree(dev->d_escaped_mu);                /* Rotation mode */
    cudaFree(dev->d_virtual_spectrum);
    if (dev->d_j_nu_estimator) cudaFree(dev->d_j_nu_estimator);
    if (dev->d_chi_bf) cudaFree(dev->d_chi_bf);
    if (dev->d_T_rad)  cudaFree(dev->d_T_rad);
    if (dev->d_bf_activation_level) cudaFree(dev->d_bf_activation_level);
}

/* ============================================================ */
/* Phase 6 - Step 2: RNG device functions (xoshiro256**)        */
/* Matches CPU rng_init/rng_uniform/rng_mu exactly.             */
/* ============================================================ */

/* Phase 6 - Step 2: SplitMix64 for seeding (device) */
__device__ __forceinline__
uint64_t d_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL); /* Phase 6 - Step 2 */
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;    /* Phase 6 - Step 2 */
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;     /* Phase 6 - Step 2 */
    return z ^ (z >> 31);                             /* Phase 6 - Step 2 */
}

/* Phase 6 - Step 2: Initialize xoshiro256** state from seed */
__device__ __forceinline__
void d_rng_init(uint64_t *s, uint64_t seed) {
    uint64_t st = seed;      /* Phase 6 - Step 2 */
    s[0] = d_splitmix64(&st); /* Phase 6 - Step 2 */
    s[1] = d_splitmix64(&st); /* Phase 6 - Step 2 */
    s[2] = d_splitmix64(&st); /* Phase 6 - Step 2 */
    s[3] = d_splitmix64(&st); /* Phase 6 - Step 2 */
}

/* Phase 6 - Step 2: Rotate left helper */
__device__ __forceinline__
uint64_t d_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k)); /* Phase 6 - Step 2 */
}

/* Phase 6 - Step 2: xoshiro256** — uniform [0, 1) */
__device__ __forceinline__
double d_rng_uniform(uint64_t *s) {
    const uint64_t result = d_rotl(s[1] * 5, 7) * 9; /* Phase 6 - Step 2 */
    const uint64_t t = s[1] << 17;                    /* Phase 6 - Step 2 */
    s[2] ^= s[0]; /* Phase 6 - Step 2 */
    s[3] ^= s[1]; /* Phase 6 - Step 2 */
    s[1] ^= s[2]; /* Phase 6 - Step 2 */
    s[0] ^= s[3]; /* Phase 6 - Step 2 */
    s[2] ^= t;    /* Phase 6 - Step 2 */
    s[3] = d_rotl(s[3], 45); /* Phase 6 - Step 2 */
    return (result >> 11) * 0x1.0p-53; /* Phase 6 - Step 2: [0, 1) */
}

/* Phase 6 - Step 2: Uniform [-1, 1) for mu sampling */
__device__ __forceinline__
double d_rng_mu(uint64_t *s) {
    return 2.0 * d_rng_uniform(s) - 1.0; /* Phase 6 - Step 2 */
}

/* ============================================================ */
/* Phase 6 - Step 3: Physics device functions                   */
/* Direct port from lumina_transport.c                          */
/* ============================================================ */

/* Phase 6 - Step 3: Doppler factor (lab → comoving) */
__device__ __forceinline__
double d_get_doppler_factor(double r, double mu, double t_exp) {
    double beta = r / (C_SPEED_OF_LIGHT * t_exp); /* Phase 6 - Step 3 */
    return 1.0 - mu * beta;                        /* Phase 6 - Step 3 */
}

/* Phase 6 - Step 3: Inverse Doppler factor (comoving → lab) */
__device__ __forceinline__
double d_get_inverse_doppler_factor(double r, double mu, double t_exp) {
    double beta = r / (C_SPEED_OF_LIGHT * t_exp); /* Phase 6 - Step 3 */
    return 1.0 / (1.0 - mu * beta);                /* Phase 6 - Step 3 */
}

/* Phase 6 - Step 3: Distance to shell boundary */
__device__
void d_calculate_distance_boundary(double r, double mu,
                                    double r_inner, double r_outer,
                                    double *out_distance, int *out_delta_shell) {
    if (mu > 0.0) { /* Phase 6 - Step 3: outward-moving packet */
        *out_distance = sqrt(r_outer * r_outer + (mu * mu - 1.0) * r * r) /* Phase 6 - Step 3 */
                        - r * mu;              /* Phase 6 - Step 3 */
        *out_delta_shell = 1;                  /* Phase 6 - Step 3 */
    } else { /* Phase 6 - Step 3: inward-moving packet */
        double check = r_inner * r_inner + r * r * (mu * mu - 1.0); /* Phase 6 - Step 3 */
        if (check >= 0.0) { /* Phase 6 - Step 3: hits inner boundary */
            *out_distance = -r * mu - sqrt(check); /* Phase 6 - Step 3 */
            *out_delta_shell = -1;                  /* Phase 6 - Step 3 */
        } else { /* Phase 6 - Step 3: misses inner, bounces to outer */
            *out_distance = sqrt(r_outer * r_outer + /* Phase 6 - Step 3 */
                                 (mu * mu - 1.0) * r * r) - r * mu; /* Phase 6 - Step 3 */
            *out_delta_shell = 1;                    /* Phase 6 - Step 3 */
        }
    }
}

/* Phase 6 - Step 3: Distance to line resonance */
__device__ __forceinline__
double d_calculate_distance_line(double comov_nu, double nu_lab,
                                  int is_last_line, double nu_line,
                                  double t_exp) {
    if (is_last_line) { /* Phase 6 - Step 3 */
        return MISS_DISTANCE; /* Phase 6 - Step 3 */
    }
    double nu_diff = comov_nu - nu_line; /* Phase 6 - Step 3 */
    if (fabs(nu_diff / nu_lab) < CLOSE_LINE_THRESHOLD) { /* Phase 6 - Step 3 */
        nu_diff = 0.0; /* Phase 6 - Step 3 */
    }
    if (nu_diff >= 0.0) { /* Phase 6 - Step 3 */
        return (nu_diff / nu_lab) * C_SPEED_OF_LIGHT * t_exp; /* Phase 6 - Step 3 */
    }
    return MISS_DISTANCE; /* Phase 6 - Step 3 */
}

/* Phase 6 - Step 3: Calc packet energy at distance along path */
__device__ __forceinline__
double d_calc_packet_energy(double pkt_energy, double pkt_r, double pkt_mu,
                             double distance_trace, double t_exp) {
    double doppler = 1.0 - (distance_trace + pkt_mu * pkt_r) / /* Phase 6 - Step 3 */
                     (t_exp * C_SPEED_OF_LIGHT);                /* Phase 6 - Step 3 */
    return pkt_energy * doppler;                                 /* Phase 6 - Step 3 */
}

/* ============================================================ */
/* Phase 6 - Step 4: Estimator update device functions          */
/* ============================================================ */

/* Phase 6 - Step 4: Base J and nu_bar estimators (atomicAdd) */
__device__ __forceinline__
void d_update_base_estimators(double *d_j_est, double *d_nu_bar_est,
                               double *d_j_nu_est, int nlte_n_freq_bins,
                               double nlte_nu_min, double nlte_d_log_nu,
                               int shell_id, int n_shells, double distance,
                               double comov_nu, double comov_energy) {
    atomicAdd(&d_j_est[shell_id], comov_energy * distance);             /* Phase 6 - Step 4 */
    atomicAdd(&d_nu_bar_est[shell_id], comov_energy * distance * comov_nu); /* Phase 6 - Step 4 */

    /* NLTE: bin into J_nu frequency histogram */
    if (d_j_nu_est != NULL && nlte_n_freq_bins > 0 &&
        comov_nu > nlte_nu_min) {
        int freq_bin = (int)(log(comov_nu / nlte_nu_min) / nlte_d_log_nu);
        if (freq_bin >= 0 && freq_bin < nlte_n_freq_bins) {
            atomicAdd(&d_j_nu_est[shell_id * nlte_n_freq_bins + freq_bin],
                      comov_energy * distance);
        }
    }
}

/* Phase 6 - Step 4: Line estimators — skipped on GPU */
/* j_blue and Edotlu are too large for atomic writes (137252 * 30 doubles) */
/* CPU handles these in plasma solve; GPU only needs j/nu_bar for W,T_rad */

/* ============================================================ */
/* BF opacity device functions                                  */
/* ============================================================ */

/* Lookup chi_bf from precomputed grid (linear interpolation in log-nu) */
__device__ __forceinline__
double d_bf_get_chi(const double *d_chi_bf, int bf_n_freq_bins,
                     double bf_nu_min, double bf_nu_max, double bf_d_log_nu,
                     int shell, double nu) {
    if (d_chi_bf == NULL || nu < bf_nu_min || nu >= bf_nu_max) return 0.0;
    double log_ratio = log(nu / bf_nu_min);
    int bin = (int)(log_ratio / bf_d_log_nu);
    if (bin < 0) return 0.0;
    if (bin >= bf_n_freq_bins - 1)
        return d_chi_bf[shell * bf_n_freq_bins + bf_n_freq_bins - 1];
    double frac = log_ratio / bf_d_log_nu - (double)bin;
    double chi0 = d_chi_bf[shell * bf_n_freq_bins + bin];
    double chi1 = d_chi_bf[shell * bf_n_freq_bins + bin + 1];
    return chi0 + frac * (chi1 - chi0);
}

/* Lookup macro-atom activation level for BF absorption at given frequency */
__device__ __forceinline__
int d_bf_get_activation_level(const int *d_bf_act, int bf_n_freq_bins,
                               double bf_nu_min, double bf_nu_max, double bf_d_log_nu,
                               int shell, double nu) {
    if (d_bf_act == NULL || nu < bf_nu_min || nu >= bf_nu_max) return -1;
    int bin = (int)(log(nu / bf_nu_min) / bf_d_log_nu);
    if (bin < 0 || bin >= bf_n_freq_bins) return -1;
    return d_bf_act[shell * bf_n_freq_bins + bin];
}

/* Sample Planck frequency using Bjorkman-Wood method */
__device__ __forceinline__
double d_sample_planck_frequency(double T, uint64_t *rng) {
    double kT_h = K_BOLTZMANN * T / H_PLANCK;
    double xi0 = d_rng_uniform(rng);
    double l_coef = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 90.0;
    double target = xi0 * l_coef;
    double cumsum = 0.0;
    double l_min = 1.0;
    for (int l = 1; l <= 1000; l++) {
        double ld = (double)l;
        double l_inv4 = 1.0 / (ld * ld * ld * ld);
        cumsum += l_inv4;
        if (cumsum >= target) { l_min = ld; break; }
    }
    double r1 = d_rng_uniform(rng), r2 = d_rng_uniform(rng);
    double r3 = d_rng_uniform(rng), r4 = d_rng_uniform(rng);
    if (r1 < 1e-300) r1 = 1e-300;
    if (r2 < 1e-300) r2 = 1e-300;
    if (r3 < 1e-300) r3 = 1e-300;
    if (r4 < 1e-300) r4 = 1e-300;
    return -log(r1 * r2 * r3 * r4) / l_min * kT_h;
}

/* BF absorption: thermalize packet — re-emit as Planck(T_rad), reset line ID */
__device__
void d_bf_absorption_event(double *pkt_r, double *pkt_mu, double *pkt_nu,
                            int *pkt_next_line_id, double t_exp,
                            const double *d_T_rad, int shell_id,
                            const double *d_line_list_nu, int n_lines,
                            uint64_t *rng) {
    /* 1. New isotropic direction */
    *pkt_mu = d_rng_mu(rng);

    /* 2. Sample comoving frequency from Planck(T_rad) */
    double T_rad = d_T_rad[shell_id];
    double comov_nu = d_sample_planck_frequency(T_rad, rng);

    /* 3. Transform to lab frame */
    double inv_doppler = d_get_inverse_doppler_factor(*pkt_r, *pkt_mu, t_exp);
    *pkt_nu = comov_nu * inv_doppler;

    /* 4. Reinitialize next_line_id (binary search in descending list) */
    double comov_check = *pkt_nu * d_get_doppler_factor(*pkt_r, *pkt_mu, t_exp);
    int lo = 0, hi = n_lines;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_line_list_nu[mid] > comov_check) lo = mid + 1;
        else hi = mid;
    }
    if (lo == n_lines) lo = n_lines - 1;
    *pkt_next_line_id = lo;
}

/* [EPS-UV 2STEP] Band-constrained variant: Planck(T_rad) re-emission rejection-
 * sampled into [nu_lo, nu_hi]. Used by the 2-step UV→opt→red cascade so the
 * packet lands in the optical band and red flux can emerge from natural line
 * physics afterward. Falls back to a uniform draw within the band if the
 * rejection loop fails to find a sample within 256 attempts (very rare at
 * SN T_rad ~ 8000 K when band straddles the Wien peak). */
__device__
void d_bf_absorption_event_band(double *pkt_r, double *pkt_mu, double *pkt_nu,
                                 int *pkt_next_line_id, double t_exp,
                                 const double *d_T_rad, int shell_id,
                                 const double *d_line_list_nu, int n_lines,
                                 double nu_lo, double nu_hi,
                                 uint64_t *rng) {
    *pkt_mu = d_rng_mu(rng);
    double T_rad = d_T_rad[shell_id];
    double comov_nu = -1.0;
    for (int attempt = 0; attempt < 256; attempt++) {
        double cand = d_sample_planck_frequency(T_rad, rng);
        if (cand >= nu_lo && cand <= nu_hi) { comov_nu = cand; break; }
    }
    if (comov_nu < 0.0) {
        comov_nu = nu_lo + (nu_hi - nu_lo) * d_rng_uniform(rng);
    }
    double inv_doppler = d_get_inverse_doppler_factor(*pkt_r, *pkt_mu, t_exp);
    *pkt_nu = comov_nu * inv_doppler;
    double comov_check = *pkt_nu * d_get_doppler_factor(*pkt_r, *pkt_mu, t_exp);
    int lo = 0, hi = n_lines;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_line_list_nu[mid] > comov_check) lo = mid + 1;
        else hi = mid;
    }
    if (lo == n_lines) lo = n_lines - 1;
    *pkt_next_line_id = lo;
}

/* ============================================================ */
/* Phase 6 - Step 5: trace_packet device function               */
/* Direct port of lumina_transport.c trace_packet()             */
/* ============================================================ */

__device__
void d_trace_packet(
    /* Phase 6 - Step 5: Packet state (in/out) */
    double pkt_r, double pkt_mu, double pkt_nu, double pkt_energy,
    int pkt_shell_id, int pkt_next_line_id,
    /* Phase 6 - Step 5: Geometry */
    const double *d_r_inner, const double *d_r_outer,
    /* Phase 6 - Step 5: Opacity */
    const double *d_line_list_nu, const double *d_tau_sobolev,
    int n_lines, int n_shells,
    /* Phase 6 - Step 5: Continuum opacity */
    double chi_continuum,
    /* Phase 6 - Step 5: Estimators */
    double *d_j_est, double *d_nu_bar_est,
    /* Phase 6 - Step 5: RNG */
    uint64_t *rng,
    /* Phase 6 - Step 5: Config */
    double t_exp,
    /* Phase 6 - Step 5: Output */
    double *out_distance, int *out_type, int *out_delta_shell,
    int *out_next_line_id)
{
    int shell = pkt_shell_id;                     /* Phase 6 - Step 5 */
    double r_inner = d_r_inner[shell];            /* Phase 6 - Step 5 */
    double r_outer = d_r_outer[shell];            /* Phase 6 - Step 5 */

    /* Phase 6 - Step 5: Distance to shell boundary */
    double distance_boundary;                     /* Phase 6 - Step 5 */
    int delta_shell;                              /* Phase 6 - Step 5 */
    d_calculate_distance_boundary(pkt_r, pkt_mu, r_inner, r_outer, /* Phase 6 - Step 5 */
                                   &distance_boundary, &delta_shell); /* Phase 6 - Step 5 */

    /* Phase 6 - Step 5: Sample optical depth */
    double tau_event = -log(d_rng_uniform(rng));  /* Phase 6 - Step 5 */
    double tau_trace_line_combined = 0.0;         /* Phase 6 - Step 5 */

    /* Phase 6 - Step 5: Doppler factor at current position */
    double doppler_factor = d_get_doppler_factor(pkt_r, pkt_mu, t_exp); /* Phase 6 - Step 5 */
    double comov_nu = pkt_nu * doppler_factor;    /* Phase 6 - Step 5 */

    /* Phase 6 - Step 5: Continuum distance */
    double distance_continuum = tau_event / chi_continuum; /* Phase 6 - Step 5 */

    int start_line_id = pkt_next_line_id;         /* Phase 6 - Step 5 */
    int last_line_id = n_lines - 1;               /* Phase 6 - Step 5 */
    int cur_line_id = start_line_id;              /* Phase 6 - Step 5 */

    /* Phase 6 - Step 5: Main line-tracing loop */
    bool broke_out = false;                       /* Phase 6 - Step 5 */

    for (cur_line_id = start_line_id; cur_line_id < n_lines; cur_line_id++) { /* Phase 6 - Step 5 */
        double nu_line = d_line_list_nu[cur_line_id]; /* Phase 6 - Step 5 */
        double tau_sobolev = d_tau_sobolev[         /* Phase 6 - Step 5 */
            cur_line_id * n_shells + shell];        /* Phase 6 - Step 5 */

        /* Phase 6 - Step 5: Accumulate line tau */
        tau_trace_line_combined += tau_sobolev;    /* Phase 6 - Step 5 */

        /* Phase 6 - Step 5: Distance to this line */
        int is_last_line = (cur_line_id == last_line_id); /* Phase 6 - Step 5 */
        double distance_trace = d_calculate_distance_line( /* Phase 6 - Step 5 */
            comov_nu, pkt_nu, is_last_line, nu_line, t_exp); /* Phase 6 - Step 5 */

        /* Phase 6 - Step 5: Combined tau at trace distance */
        double tau_trace_continuum = chi_continuum * distance_trace; /* Phase 6 - Step 5 */
        double tau_trace_combined = tau_trace_line_combined +       /* Phase 6 - Step 5 */
                                     tau_trace_continuum;            /* Phase 6 - Step 5 */

        /* Phase 6 - Step 5: Find minimum distance */
        double distance = distance_trace;         /* Phase 6 - Step 5 */
        if (distance_boundary < distance) distance = distance_boundary; /* Phase 6 - Step 5 */
        if (distance_continuum < distance) distance = distance_continuum; /* Phase 6 - Step 5 */

        /* Phase 6 - Step 5: TARDIS: if distance_trace != 0 */
        if (distance_trace != 0.0) { /* Phase 6 - Step 5 */
            if (distance == distance_boundary) { /* Phase 6 - Step 5 */
                *out_type = 0; /* Phase 6 - Step 5: INTERACTION_BOUNDARY */
                *out_distance = distance_boundary; /* Phase 6 - Step 5 */
                *out_delta_shell = delta_shell;    /* Phase 6 - Step 5 */
                *out_next_line_id = cur_line_id;   /* Phase 6 - Step 5 */
                broke_out = true;                  /* Phase 6 - Step 5 */
                break;                             /* Phase 6 - Step 5 */
            } else if (distance == distance_continuum) { /* Phase 6 - Step 5 */
                *out_type = 2; /* Phase 6 - Step 5: INTERACTION_ESCATTERING */
                *out_distance = distance_continuum; /* Phase 6 - Step 5 */
                *out_delta_shell = delta_shell;     /* Phase 6 - Step 5 */
                *out_next_line_id = cur_line_id;    /* Phase 6 - Step 5 */
                broke_out = true;                   /* Phase 6 - Step 5 */
                break;                              /* Phase 6 - Step 5 */
            }
        }

        /* Phase 6 - Step 5: Update line estimators (j_blue) - GPU skips */
        /* Line estimators too large for atomicAdd on GPU; skip for now */

        /* Phase 6 - Step 5: Check if combined tau exceeds tau_event */
        if (tau_trace_combined > tau_event) { /* Phase 6 - Step 5 */
            *out_type = 1; /* Phase 6 - Step 5: INTERACTION_LINE */
            *out_distance = distance_trace;    /* Phase 6 - Step 5 */
            *out_delta_shell = delta_shell;    /* Phase 6 - Step 5 */
            *out_next_line_id = cur_line_id;   /* Phase 6 - Step 5 */
            broke_out = true;                  /* Phase 6 - Step 5 */
            break;                             /* Phase 6 - Step 5 */
        }

        /* Phase 6 - Step 5: Recalculate distance_continuum */
        distance_continuum = (tau_event - tau_trace_line_combined) / /* Phase 6 - Step 5 */
                              chi_continuum;                         /* Phase 6 - Step 5 */
    }

    /* Phase 6 - Step 5: for...else clause */
    if (!broke_out) { /* Phase 6 - Step 5 */
        *out_next_line_id = cur_line_id; /* Phase 6 - Step 5 */
        if (distance_continuum < distance_boundary) { /* Phase 6 - Step 5 */
            *out_type = 2; /* Phase 6 - Step 5: INTERACTION_ESCATTERING */
            *out_distance = distance_continuum; /* Phase 6 - Step 5 */
            *out_delta_shell = delta_shell;     /* Phase 6 - Step 5 */
        } else { /* Phase 6 - Step 5 */
            *out_type = 0; /* Phase 6 - Step 5: INTERACTION_BOUNDARY */
            *out_distance = distance_boundary;  /* Phase 6 - Step 5 */
            *out_delta_shell = delta_shell;     /* Phase 6 - Step 5 */
        }
    }
}

/* ============================================================ */
/* Phase 6 - Step 6: Interaction handler device functions       */
/* ============================================================ */

/* Phase 6 - Step 6: Thomson scatter */
__device__
void d_thomson_scatter(double *r, double *mu, double *nu, double *energy,
                        double t_exp, uint64_t *rng) {
    double old_doppler = d_get_doppler_factor(*r, *mu, t_exp); /* Phase 6 - Step 6 */
    double comov_nu = *nu * old_doppler;                       /* Phase 6 - Step 6 */
    double comov_energy = *energy * old_doppler;               /* Phase 6 - Step 6 */

    *mu = d_rng_mu(rng); /* Phase 6 - Step 6: new isotropic direction */

    double inv_new_doppler = d_get_inverse_doppler_factor(*r, *mu, t_exp); /* Phase 6 - Step 6 */
    *nu = comov_nu * inv_new_doppler;           /* Phase 6 - Step 6 */
    *energy = comov_energy * inv_new_doppler;   /* Phase 6 - Step 6 */
}

/* Phase 6 - Step 6: Line emission */
__device__ __forceinline__
void d_line_emission(double *nu, int *next_line_id,
                      int emission_line_id, double r, double mu,
                      double t_exp, const double *d_line_list_nu) {
    double inv_doppler = d_get_inverse_doppler_factor(r, mu, t_exp); /* Phase 6 - Step 6 */
    *nu = d_line_list_nu[emission_line_id] * inv_doppler;            /* Phase 6 - Step 6 */
    *next_line_id = emission_line_id + 1;                            /* Phase 6 - Step 6 */
}

/* ============================================================ */
/* [MA-FATE] Device-side helpers (storage declared earlier).     */
/* 4×4 (entry_band, exit_band) counts; aggregated to host after  */
/* each iteration via cuda_ma_fate_download_and_aggregate().     */
/* ============================================================ */
__device__ __forceinline__
int d_ma_fate_band_from_nu(double nu_comov) {
    if (nu_comov <= 0.0) return 7;
    double lam_A = (C_SPEED_OF_LIGHT / nu_comov) * 1.0e8;
    if (lam_A >= 1700.0  && lam_A <  3000.0) return 0;  /* UV-blanket  */
    if (lam_A >= 3000.0  && lam_A <  3300.0) return 1;  /* CaIIK-blue  */
    if (lam_A >= 3300.0  && lam_A <  3700.0) return 2;  /* UV-target   */
    if (lam_A >= 3700.0  && lam_A <  4400.0) return 3;  /* blue+fluor  */
    if (lam_A >= 4400.0  && lam_A <  5500.0) return 4;  /* green       */
    if (lam_A >= 5500.0  && lam_A <  7000.0) return 5;  /* red         */
    if (lam_A >= 7000.0  && lam_A < 10000.0) return 6;  /* NIR1        */
    return 7;                                            /* NIR2/far    */
}

__device__ __forceinline__
void d_ma_fate_record(double entry_nu_comov, double exit_nu_comov) {
    int eb = d_ma_fate_band_from_nu(entry_nu_comov);
    int xb = d_ma_fate_band_from_nu(exit_nu_comov);
    atomicAdd(&d_ma_fate_hist[eb * MA_FATE_NBANDS + xb], 1ULL);
}

/* Phase 6 - Step 6: Macro-atom interaction */
__device__
void d_macro_atom_interaction(int activation_level_id, int current_shell_id,
                               int n_shells, int n_macro_levels,
                               const int *d_macro_block_references,
                               const double *d_transition_probabilities,
                               const int *d_destination_level_id,
                               const int *d_transition_type,
                               const int *d_transition_line_id,
                               uint64_t *rng,
                               int *out_transition_id,
                               int *out_transition_type) {
    int current_type = 0;  /* Phase 6 - Step 6: start as internal */
    int ma_iter = 0;       /* Phase 6 - Step 6: safety counter */
    *out_transition_id = -1;  /* P8: safe default for orphaned levels */
    *out_transition_type = -1; /* safe default: emission */

    while (current_type >= 0 && ma_iter < d_ma_internal_cap) { /* Phase 6 - Step 6 */
        ma_iter++; /* Phase 6 - Step 6 */

        /* Phase 6 - Step 6: Bounds check */
        if (activation_level_id < 0 || activation_level_id >= n_macro_levels) { /* Phase 6 - Step 6 */
            current_type = -1; /* Phase 6 - Step 6: MA_BB_EMISSION */
            *out_transition_type = current_type; /* Phase 6 - Step 6 */
            break; /* Phase 6 - Step 6 */
        }

        /* [KPACKET] Collisional deactivation roll. With prob d_kpacket_p[lev,shell]
         * the activation thermalizes into a k-packet (energy → electron pool) and
         * re-excites at a level drawn from the per-shell thermal re-excitation CDF.
         * This breaks the deterministic redward radiative walk and re-emits near
         * the local Planck(T_e) peak. Bounded by d_ma_internal_cap (each roll is a
         * while-loop iteration). */
        if (d_kpacket_enabled && d_kpacket_p && d_kpacket_cdf_g) {
            double pk = d_kpacket_p[(size_t)activation_level_id * n_shells
                                    + current_shell_id];
            if (pk > 0.0 && d_rng_uniform(rng) < pk) {
                const double *cdf = d_kpacket_cdf_g
                                    + (size_t)current_shell_id * n_macro_levels;
                double xi = d_rng_uniform(rng);
                int lo = 0, hi = n_macro_levels - 1;       /* binary-search CDF */
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (cdf[mid] < xi) lo = mid + 1; else hi = mid;
                }
                activation_level_id = lo;     /* thermally re-excited upper level */
                current_type = 0;             /* stay internal; continue cascade  */
                atomicAdd(&d_kpacket_count, 1ULL);
                continue;
            }
        }

        double probability = 0.0;                              /* Phase 6 - Step 6 */
        double probability_event = d_rng_uniform(rng);          /* Phase 6 - Step 6 */

        int block_start = d_macro_block_references[activation_level_id];     /* Phase 6 - Step 6 */
        int block_end = d_macro_block_references[activation_level_id + 1];   /* Phase 6 - Step 6 */

        bool found = false; /* Phase 6 - Step 6 */
        for (int tid = block_start; tid < block_end; tid++) { /* Phase 6 - Step 6 */
            double tp = d_transition_probabilities[     /* Phase 6 - Step 6 */
                tid * n_shells + current_shell_id];     /* Phase 6 - Step 6 */
            probability += tp;                          /* Phase 6 - Step 6 */

            if (probability > probability_event) { /* Phase 6 - Step 6 */
                activation_level_id = d_destination_level_id[tid]; /* Phase 6 - Step 6 */
                current_type = d_transition_type[tid];             /* Phase 6 - Step 6 */
                *out_transition_id = tid;                          /* Phase 6 - Step 6 */
                *out_transition_type = current_type;               /* Phase 6 - Step 6 */
                found = true;                                      /* Phase 6 - Step 6 */
                break;                                             /* Phase 6 - Step 6 */
            }
        }

        if (!found) { /* Phase 6 - Step 6 */
            if (block_start >= block_end) { /* Phase 6 - Step 6: empty block */
                current_type = -1; /* Phase 6 - Step 6: MA_BB_EMISSION */
                *out_transition_type = current_type; /* Phase 6 - Step 6 */
                break; /* Phase 6 - Step 6 */
            }
            /* Phase 6 - Step 6: Pick last transition */
            int tid = block_end - 1;                               /* Phase 6 - Step 6 */
            activation_level_id = d_destination_level_id[tid];     /* Phase 6 - Step 6 */
            current_type = d_transition_type[tid];                 /* Phase 6 - Step 6 */
            *out_transition_id = tid;                              /* Phase 6 - Step 6 */
            *out_transition_type = current_type;                   /* Phase 6 - Step 6 */
        }
    }

    /* [MA-CYCLE] Record cycle count before line_id conversion */
    {
        int cb = ma_iter;
        if (cb < 0) cb = 0;
        if (cb >= MA_CYCLE_BINS) cb = MA_CYCLE_BINS - 1;
        atomicAdd(&d_ma_cycle_hist[cb], 1ULL);
    }

    /* Phase 6 - Step 6: Convert transition_id to line_id for emission */
    if (*out_transition_id >= 0)
        *out_transition_id = d_transition_line_id[*out_transition_id]; /* Phase 6 - Step 6 */
}

/* Phase 6 - Step 6: Line scatter event (resonant, downbranch, or macro-atom) */
__device__
void d_line_scatter_event(double *r, double *mu, double *nu, double *energy,
                           int *next_line_id, int shell_id,
                           double t_exp, int line_interaction_type,
                           /* Phase 6 - Step 6: Opacity data pointers */
                           const double *d_line_list_nu,
                           int n_shells, int n_macro_levels,
                           const int *d_macro_block_references,
                           const double *d_transition_probabilities,
                           const int *d_destination_level_id,
                           const int *d_transition_type,
                           const int *d_transition_line_id,
                           const int *d_line2macro_level_upper,
                           const int *d_line_atomic_number,
                           const int *d_line_ion_number,
                           int fe_scatter_mode,
                           const double *d_T_rad, int n_lines,
                           uint64_t *rng) {
    /* Phase 6 - Step 6: Get comoving frame at OLD angle */
    double old_doppler = d_get_doppler_factor(*r, *mu, t_exp); /* Phase 6 - Step 6 */

    /* Phase 6 - Step 6: Sample new isotropic direction */
    *mu = d_rng_mu(rng); /* Phase 6 - Step 6 */

    /* Phase 6 - Step 6: Transform energy to lab with NEW angle */
    double inv_new_doppler = d_get_inverse_doppler_factor(*r, *mu, t_exp); /* Phase 6 - Step 6 */
    double comov_energy = *energy * old_doppler;  /* Phase 6 - Step 6 */
    *energy = comov_energy * inv_new_doppler;     /* Phase 6 - Step 6 */

    if (line_interaction_type == 0) { /* Phase 6 - Step 6: LINE_SCATTER */
        d_line_emission(nu, next_line_id, *next_line_id, /* Phase 6 - Step 6 */
                         *r, *mu, t_exp, d_line_list_nu); /* Phase 6 - Step 6 */
    } else { /* Phase 6 - Step 6: macro-atom */
        double comov_nu = *nu * old_doppler;  /* Phase 6 - Step 6 */
        *nu = comov_nu * inv_new_doppler;     /* Phase 6 - Step 6 */

        /* Fe two-level atom: resonance scatter instead of macro-atom cascade
         * fe_scatter_mode: 0=off, 1=Fe II only, 2=all Fe (II+III+...) */
        int is_fe_scatter = 0;
        if (fe_scatter_mode && d_line_atomic_number[*next_line_id] == 26) {
            if (fe_scatter_mode == 2) is_fe_scatter = 1;           /* all Fe */
            else if (d_line_ion_number[*next_line_id] == 1) is_fe_scatter = 1; /* Fe II only */
        }
        if (is_fe_scatter) {
            d_line_emission(nu, next_line_id, *next_line_id,
                             *r, *mu, t_exp, d_line_list_nu);
        } else {
            /* [MA-FATE] entry comov nu = activation line frequency in atomic frame */
            double ma_entry_comov_nu = *nu * d_get_doppler_factor(*r, *mu, t_exp);

            /* [H3] Capture activating species (Z, ion) BEFORE cascade or
             * bypass overwrites *next_line_id. Used for per-(Z,ion) fate. */
            int act_Z   = d_line_atomic_number[*next_line_id];
            int act_ion = d_line_ion_number[*next_line_id];

            /* [EPS-UV] If UV-entry, with probability d_eps_uv replace the
             * macro-atom cascade with a Planck(T_rad) thermalization. This
             * emulates wavelength redistribution that the atomic data is
             * missing (no UV→optical downward radiative paths).
             * [H2] In red-only mode the gate is moved post-cascade. */
            int entry_band = d_ma_fate_band_from_nu(ma_entry_comov_nu);
            if (!d_eps_uv_red_only &&
                entry_band == 0 && d_eps_uv > 0.0 &&
                d_rng_uniform(rng) < d_eps_uv) {
                if (d_eps_uv_2step) {
                    d_bf_absorption_event_band(r, mu, nu, next_line_id, t_exp,
                                                d_T_rad, shell_id,
                                                d_line_list_nu, n_lines,
                                                d_eps_uv_2step_nu_lo,
                                                d_eps_uv_2step_nu_hi, rng);
                } else {
                    d_bf_absorption_event(r, mu, nu, next_line_id, t_exp,
                                           d_T_rad, shell_id,
                                           d_line_list_nu, n_lines, rng);
                }
                double ma_exit_comov_nu_th = *nu *
                    d_get_doppler_factor(*r, *mu, t_exp);
                d_ma_fate_record_zi(ma_entry_comov_nu, ma_exit_comov_nu_th,
                                     entry_band, act_Z, act_ion);
                return;
            }

            /* [EPS-IR] If NIR-entry (lam > 7000 A), with probability d_eps_ir
             * thermalize back to Planck(T_rad) instead of running the cascade.
             * Counters the IR-trapping inflation under dense atomic data. */
            if (d_eps_ir > 0.0) {
                double lam_A_entry = (C_SPEED_OF_LIGHT / ma_entry_comov_nu) * 1.0e8;
                if (lam_A_entry > 7000.0 && d_rng_uniform(rng) < d_eps_ir) {
                    d_bf_absorption_event(r, mu, nu, next_line_id, t_exp,
                                           d_T_rad, shell_id,
                                           d_line_list_nu, n_lines, rng);
                    double ma_exit_comov_nu_th = *nu *
                        d_get_doppler_factor(*r, *mu, t_exp);
                    d_ma_fate_record_zi(ma_entry_comov_nu, ma_exit_comov_nu_th,
                                         entry_band, act_Z, act_ion);
                    return;
                }
            }

            /* Phase 6 - Step 6: Activate macro-atom */
            int activation_level = d_line2macro_level_upper[*next_line_id]; /* Phase 6 - Step 6 */

            int transition_id;   /* Phase 6 - Step 6 */
            int transition_type; /* Phase 6 - Step 6 */
            d_macro_atom_interaction(activation_level, shell_id,        /* Phase 6 - Step 6 */
                                      n_shells, n_macro_levels,         /* Phase 6 - Step 6 */
                                      d_macro_block_references,         /* Phase 6 - Step 6 */
                                      d_transition_probabilities,       /* Phase 6 - Step 6 */
                                      d_destination_level_id,           /* Phase 6 - Step 6 */
                                      d_transition_type,                /* Phase 6 - Step 6 */
                                      d_transition_line_id,             /* Phase 6 - Step 6 */
                                      rng, &transition_id,              /* Phase 6 - Step 6 */
                                      &transition_type);                /* Phase 6 - Step 6 */

            if (transition_type == -1) { /* Phase 6 - Step 6: MA_BB_EMISSION */
                /* P8: Orphaned level (no transitions) → resonance scatter */
                int emit_line = (transition_id >= 0) ? transition_id : *next_line_id;
                d_line_emission(nu, next_line_id, emit_line, /* Phase 6 - Step 6 */
                                 *r, *mu, t_exp, d_line_list_nu); /* Phase 6 - Step 6 */
            }
            /* [MA-FATE] exit comov nu after cascade + line_emission */
            double ma_exit_comov_nu = *nu * d_get_doppler_factor(*r, *mu, t_exp);

            /* [H2] Post-cascade red-only gate: if UV-entry AND exit landed
             * in [5500,10000)Å (bands 5+6), with prob d_eps_uv re-sample
             * as Planck(T_rad). Kills UV→red conversion only; keeps
             * UV→UV/blue downbranching paths intact. */
            if (d_eps_uv_red_only && entry_band == 0 && d_eps_uv > 0.0) {
                int exit_band = d_ma_fate_band_from_nu(ma_exit_comov_nu);
                if ((exit_band == 5 || exit_band == 6) &&
                    d_rng_uniform(rng) < d_eps_uv) {
                    if (d_eps_uv_2step) {
                        d_bf_absorption_event_band(r, mu, nu, next_line_id, t_exp,
                                                    d_T_rad, shell_id,
                                                    d_line_list_nu, n_lines,
                                                    d_eps_uv_2step_nu_lo,
                                                    d_eps_uv_2step_nu_hi, rng);
                    } else {
                        d_bf_absorption_event(r, mu, nu, next_line_id, t_exp,
                                               d_T_rad, shell_id,
                                               d_line_list_nu, n_lines, rng);
                    }
                    ma_exit_comov_nu = *nu *
                        d_get_doppler_factor(*r, *mu, t_exp);
                }
            }
            d_ma_fate_record_zi(ma_entry_comov_nu, ma_exit_comov_nu,
                                 entry_band, act_Z, act_ion);
        }
    }
}

/* ============================================================ */
/* Virtual packet tracer: at each interaction, emit a v-packet  */
/* in a random direction, trace it through remaining shells,    */
/* accumulate tau, and atomicAdd to virtual spectrum.            */
/* ============================================================ */
__device__
void d_trace_virtual_packet(
    double r, int shell_id, double nu_cmf_emit, double pkt_energy,
    double t_exp, double L_inner,
    const double *d_r_inner, const double *d_r_outer,
    const double *d_line_list_nu, const double *d_tau_sobolev,
    const double *d_electron_density,
    const double *d_chi_bf, int bf_enabled, int bf_n_freq_bins,
    double bf_nu_min, double bf_nu_max, double bf_d_log_nu,
    int n_lines, int n_shells,
    int n_vpackets,
    double *d_virtual_spectrum, uint64_t *rng)
{
    /* Draw random emission direction for v-packet */
    double mu_v = 2.0 * d_rng_uniform(rng) - 1.0;

    /* z-coordinate along ray (impact parameter stays constant) */
    double inv_ct = 1.0 / (C_SPEED_OF_LIGHT * t_exp);
    double z0 = r * mu_v;
    double doppler = 1.0 - z0 * inv_ct;
    if (doppler <= 0.0) return;
    double nu_lab = nu_cmf_emit / doppler;

    /* Impact parameter squared: p^2 = r^2 - z0^2 = r^2(1-mu^2) */
    double p2 = r * r * (1.0 - mu_v * mu_v);
    if (p2 < 0.0) p2 = 0.0;

    /* Check if ray hits photosphere (p < r_inner[0] and going inward) */
    if (p2 < d_r_inner[0] * d_r_inner[0] && mu_v < 0.0)
        return;

    double tau_total = 0.0;
    double z_cur = z0;
    int s = shell_id;

    /* Phase 1: Inward propagation (mu_v < 0 → z_cur < 0, z increasing) */
    while (mu_v < 0.0 && s >= 0) {
        double r_inner_s = d_r_inner[s];
        if (p2 >= r_inner_s * r_inner_s) {
            /* Turning point in this shell: closest approach at z=0, r=sqrt(p2) */
            double nu_high = nu_lab * (1.0 - z_cur * inv_ct);
            double nu_low  = nu_lab;  /* at z=0: nu_cmf = nu_lab */
            if (nu_high > nu_low) {
                int lo = 0, hi = n_lines;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (d_line_list_nu[mid] > nu_high) lo = mid + 1;
                    else hi = mid;
                }
                for (int i = lo; i < n_lines && d_line_list_nu[i] >= nu_low; i++)
                    tau_total += d_tau_sobolev[(size_t)i * n_shells + s];
            }
            /* Continuum tau for turning-point segment (e-scatter + BF) */
            {
                double seg_len = fabs(0.0 - z_cur);
                double chi_cont_s = SIGMA_THOMSON * d_electron_density[s];
                if (bf_enabled && d_chi_bf != NULL) {
                    double nu_mid = nu_lab * (1.0 - 0.5 * z_cur * inv_ct);
                    chi_cont_s += d_bf_get_chi(d_chi_bf, bf_n_freq_bins,
                                                bf_nu_min, bf_nu_max, bf_d_log_nu,
                                                s, nu_mid);
                }
                tau_total += chi_cont_s * seg_len;
            }
            z_cur = 0.0;
            break;
        } else {
            /* Crosses inner boundary into shell s-1 */
            double arg_in = r_inner_s * r_inner_s - p2;
            if (arg_in < 0.0) arg_in = 0.0;
            double z_bnd = -sqrt(arg_in);
            double nu_high = nu_lab * (1.0 - z_cur * inv_ct);
            double nu_low  = nu_lab * (1.0 - z_bnd * inv_ct);
            if (nu_high > nu_low) {
                int lo = 0, hi = n_lines;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (d_line_list_nu[mid] > nu_high) lo = mid + 1;
                    else hi = mid;
                }
                for (int i = lo; i < n_lines && d_line_list_nu[i] >= nu_low; i++)
                    tau_total += d_tau_sobolev[(size_t)i * n_shells + s];
            }
            /* Continuum tau for inward shell crossing (e-scatter + BF) */
            {
                double seg_len = fabs(z_bnd - z_cur);
                double chi_cont_s = SIGMA_THOMSON * d_electron_density[s];
                if (bf_enabled && d_chi_bf != NULL) {
                    double nu_mid = nu_lab * (1.0 - 0.5 * (z_cur + z_bnd) * inv_ct);
                    chi_cont_s += d_bf_get_chi(d_chi_bf, bf_n_freq_bins,
                                                bf_nu_min, bf_nu_max, bf_d_log_nu,
                                                s, nu_mid);
                }
                tau_total += chi_cont_s * seg_len;
            }
            z_cur = z_bnd;
            s--;
        }
    }
    if (s < 0) return;  /* reabsorbed by photosphere */

    /* Phase 2: Outward propagation */
    while (s < n_shells) {
        double r_outer_s = d_r_outer[s];
        double arg = r_outer_s * r_outer_s - p2;
        if (arg < 0.0) arg = 0.0;  /* guard against float rounding */
        double z_bnd = sqrt(arg);
        double nu_high = nu_lab * (1.0 - z_cur * inv_ct);
        double nu_low  = nu_lab * (1.0 - z_bnd * inv_ct);
        if (nu_high > nu_low) {
            int lo = 0, hi = n_lines;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (d_line_list_nu[mid] > nu_high) lo = mid + 1;
                else hi = mid;
            }
            for (int i = lo; i < n_lines && d_line_list_nu[i] >= nu_low; i++)
                tau_total += d_tau_sobolev[(size_t)i * n_shells + s];
        }
        /* Continuum tau for outward shell crossing (e-scatter + BF) */
        {
            double seg_len = fabs(z_bnd - z_cur);
            double chi_cont_s = SIGMA_THOMSON * d_electron_density[s];
            if (bf_enabled && d_chi_bf != NULL) {
                double nu_mid = nu_lab * (1.0 - 0.5 * (z_cur + z_bnd) * inv_ct);
                chi_cont_s += d_bf_get_chi(d_chi_bf, bf_n_freq_bins,
                                            bf_nu_min, bf_nu_max, bf_d_log_nu,
                                            s, nu_mid);
            }
            tau_total += chi_cont_s * seg_len;
        }
        z_cur = z_bnd;
        s++;
    }

    /* Escape probability */
    if (tau_total > 50.0) return;
    double P_escape = exp(-tau_total);

    /* Bin into virtual spectrum [erg/s/cm] */
    double lambda_A = C_SPEED_OF_LIGHT / nu_lab * 1.0e8;
    if (lambda_A < VSPEC_LAMBDA_MIN || lambda_A >= VSPEC_LAMBDA_MAX) return;
    double dlambda_A = (VSPEC_LAMBDA_MAX - VSPEC_LAMBDA_MIN) / (double)VSPEC_N_BINS;
    int bin = (int)((lambda_A - VSPEC_LAMBDA_MIN) / dlambda_A);
    if (bin >= 0 && bin < VSPEC_N_BINS) {
        double dlambda_cm = dlambda_A * 1.0e-8;
        double weight = pkt_energy * L_inner * P_escape / dlambda_cm / (double)n_vpackets;
        atomicAdd(&d_virtual_spectrum[bin], weight);
    }
}

/* ============================================================ */
/* Bjorkman-Wood Planck frequency sampler (comoving nu).        */
/* Factored out of transport_kernel init so the Task #27        */
/* Phase 2a diffusive inner boundary can re-thermalize a        */
/* returned packet to T_inner with the same sampler.            */
/* ============================================================ */
__device__ __forceinline__
double d_sample_bw_planck_nu(uint64_t *rng, double kT_h)
{
    double xi0 = d_rng_uniform(rng);
    double l_coef = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 90.0;
    double target = xi0 * l_coef;
    double cumsum = 0.0;
    double l_min = 1.0;
    for (int l = 1; l <= 1000; l++) {
        double ld = (double)l;
        cumsum += 1.0 / (ld * ld * ld * ld);
        if (cumsum >= target) { l_min = ld; break; }
    }
    double r1 = d_rng_uniform(rng);
    double r2 = d_rng_uniform(rng);
    double r3 = d_rng_uniform(rng);
    double r4 = d_rng_uniform(rng);
    if (r1 < 1e-300) r1 = 1e-300;
    if (r2 < 1e-300) r2 = 1e-300;
    if (r3 < 1e-300) r3 = 1e-300;
    if (r4 < 1e-300) r4 = 1e-300;
    double x = -log(r1 * r2 * r3 * r4) / l_min;
    return x * kT_h;
}

/* ============================================================ */
/* Phase 6 - Step 7: Main transport kernel                      */
/* One thread = one packet. No grid-stride loop.                */
/* ============================================================ */

__global__
void transport_kernel(
    /* Phase 6 - Step 7: Geometry arrays */
    const double *d_r_inner, const double *d_r_outer,
    /* Phase 6 - Step 7: Opacity arrays */
    const double *d_line_list_nu, const double *d_tau_sobolev,
    const double *d_electron_density,
    const double *d_transition_probabilities,
    const int *d_macro_block_references,
    const int *d_transition_type,
    const int *d_destination_level_id,
    const int *d_transition_line_id,
    const int *d_line2macro_level_upper,
    const int *d_line_atomic_number,
    const int *d_line_ion_number,
    int fe_scatter_mode,
    /* Phase 6 - Step 7: Estimators */
    double *d_j_estimator, double *d_nu_bar_estimator,
    /* NLTE: J_nu frequency histogram */
    double *d_j_nu_estimator, int nlte_n_freq_bins,
    double nlte_nu_min, double nlte_d_log_nu,
    /* Phase 6 - Step 7: RNG */
    uint64_t *d_rng_states,
    /* Phase 6 - Step 7: Output */
    double *d_escaped_nu, double *d_escaped_energy,
    int *d_escaped_flag,
    double *d_escaped_r, double *d_escaped_mu,
    int64_t *d_n_escaped, int64_t *d_n_reabsorbed,
    /* Virtual packet spectrum */
    double *d_virtual_spectrum, double L_inner,
    /* BF opacity arrays */
    const double *d_chi_bf, const double *d_T_rad,
    const int *d_bf_activation_level,
    int bf_enabled, int bf_n_freq_bins,
    double bf_nu_min, double bf_nu_max, double bf_d_log_nu,
    /* Phase 6 - Step 7: Scalars */
    int n_packets, int n_shells, int n_lines, int n_macro_levels,
    double t_exp, double T_inner, double packet_energy,
    int line_interaction_type, uint64_t base_seed)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x; /* Phase 6 - Step 7 */
    if (p >= n_packets) return;                     /* Phase 6 - Step 7 */

    /* A10: Decorrelate per-packet seed via SplitMix64 mix instead of additive
     * (base_seed + p). With iter_seed = config.seed + iter*1e6 and n_packets
     * >= 1e6, additive scheme collides packet (iter, p) with (iter-1, p+1e6).
     * SplitMix64 mixing also bit-shuffles seeds so adjacent packets don't
     * land on near-correlated RNG trajectories. */
    uint64_t rng[4]; /* Phase 6 - Step 7 */
    uint64_t _sm = base_seed ^ ((uint64_t)p * 0x9e3779b97f4a7c15ULL);
    uint64_t _seed_p = d_splitmix64(&_sm);
    d_rng_init(rng, _seed_p);

    /* Phase 6 - Step 7: Initialize packet at inner boundary */
    double pkt_r = d_r_inner[0];                    /* Phase 6 - Step 7 */
    double pkt_mu = sqrt(d_rng_uniform(rng));       /* Phase 6 - Step 7 */
    int pkt_shell_id = 0;                           /* Phase 6 - Step 7 */
    int pkt_status = 0; /* Phase 6 - Step 7: PACKET_IN_PROCESS */

    /* Phase 6 - Step 7: Sample frequency from Bjorkman-Wood Planck */
    double kT_h = K_BOLTZMANN * T_inner / H_PLANCK; /* Phase 6 - Step 7 */
    double pkt_nu = d_sample_bw_planck_nu(rng, kT_h); /* Phase 6 - Step 7 */
    double pkt_energy = packet_energy;            /* Phase 6 - Step 7 */

    /* Phase 6 - Step 7: set_packet_props_partial_relativity */
    double inv_doppler = d_get_inverse_doppler_factor(pkt_r, pkt_mu, t_exp); /* Phase 6 - Step 7 */
    pkt_nu *= inv_doppler;     /* Phase 6 - Step 7 */
    pkt_energy *= inv_doppler; /* Phase 6 - Step 7 */

    /* Phase 6 - Step 7: Initialize line ID via binary search */
    double comov_nu_init = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp); /* Phase 6 - Step 7 */
    int lo = 0, hi = n_lines; /* Phase 6 - Step 7 */
    while (lo < hi) { /* Phase 6 - Step 7 */
        int mid = (lo + hi) / 2;                     /* Phase 6 - Step 7 */
        if (d_line_list_nu[mid] > comov_nu_init) {   /* Phase 6 - Step 7 */
            lo = mid + 1;                             /* Phase 6 - Step 7 */
        } else {                                      /* Phase 6 - Step 7 */
            hi = mid;                                 /* Phase 6 - Step 7 */
        }
    }
    if (lo == n_lines) lo = n_lines - 1; /* Phase 6 - Step 7 */
    int pkt_next_line_id = lo;            /* Phase 6 - Step 7 */

    /* Virtual packet from initial photosphere emission */
    if (d_virtual_spectrum != NULL) {
        double nu_cmf_init = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
        for (int vp = 0; vp < N_VPACKETS; vp++) {
            d_trace_virtual_packet(pkt_r, pkt_shell_id, nu_cmf_init, pkt_energy,
                                    t_exp, L_inner, d_r_inner, d_r_outer,
                                    d_line_list_nu, d_tau_sobolev,
                                    d_electron_density,
                                    d_chi_bf, bf_enabled, bf_n_freq_bins,
                                    bf_nu_min, bf_nu_max, bf_d_log_nu,
                                    n_lines, n_shells, N_VPACKETS,
                                    d_virtual_spectrum, rng);
        }
    }

    /* Phase 6 - Step 7: Main transport loop */
    int loop_count = 0;  /* counts interactions toward d_max_interactions */
    int total_steps = 0; /* counts ALL while-iterations (absolute safety ceiling) */
    while (pkt_status == 0 && loop_count < d_max_interactions
           && total_steps < d_max_total_steps) { /* Phase 6 - Step 7 */
        total_steps++;
        /* #5 fix: in legacy mode every step counts; in real-only mode the count is
         * advanced below only for LINE/CONTINUUM events (boundary crossings/returns
         * are bounded by d_max_total_steps instead). */
        if (!d_cap_real_only) loop_count++;

        /* Phase 6 - Step 7: Continuum opacity (e-scattering + BF) */
        int shell = pkt_shell_id;                        /* Phase 6 - Step 7 */
        double chi_e = d_electron_density[shell] * SIGMA_THOMSON; /* Phase 6 - Step 7 */
        double doppler_bf = d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
        double comov_nu_bf = pkt_nu * doppler_bf;
        double chi_bf_val = 0.0;
        if (bf_enabled && d_chi_bf != NULL) {
            chi_bf_val = d_bf_get_chi(d_chi_bf, bf_n_freq_bins,
                                       bf_nu_min, bf_nu_max, bf_d_log_nu,
                                       shell, comov_nu_bf);
        }
        double chi_continuum = chi_e + chi_bf_val;

        /* Phase 6 - Step 7: Trace packet */
        double distance;      /* Phase 6 - Step 7 */
        int interaction_type; /* Phase 6 - Step 7 */
        int delta_shell;      /* Phase 6 - Step 7 */
        int new_next_line_id; /* Phase 6 - Step 7 */

        d_trace_packet(pkt_r, pkt_mu, pkt_nu, pkt_energy,   /* Phase 6 - Step 7 */
                        pkt_shell_id, pkt_next_line_id,       /* Phase 6 - Step 7 */
                        d_r_inner, d_r_outer,                 /* Phase 6 - Step 7 */
                        d_line_list_nu, d_tau_sobolev,        /* Phase 6 - Step 7 */
                        n_lines, n_shells,                    /* Phase 6 - Step 7 */
                        chi_continuum,                        /* Phase 6 - Step 7 */
                        d_j_estimator, d_nu_bar_estimator,    /* Phase 6 - Step 7 */
                        rng, t_exp,                           /* Phase 6 - Step 7 */
                        &distance, &interaction_type,         /* Phase 6 - Step 7 */
                        &delta_shell, &new_next_line_id);     /* Phase 6 - Step 7 */
        pkt_next_line_id = new_next_line_id;                  /* Phase 6 - Step 7 */

        /* Phase 6 - Step 7: move_r_packet */
        if (distance > 0.0) { /* Phase 6 - Step 7 */
            double doppler_factor = d_get_doppler_factor(pkt_r, pkt_mu, t_exp); /* Phase 6 - Step 7 */
            double new_r = sqrt(pkt_r * pkt_r + distance * distance + /* Phase 6 - Step 7 */
                                2.0 * pkt_r * distance * pkt_mu);     /* Phase 6 - Step 7 */
            pkt_mu = (pkt_mu * pkt_r + distance) / new_r;             /* Phase 6 - Step 7 */
            pkt_r = new_r;                                             /* Phase 6 - Step 7 */

            double comov_nu = pkt_nu * doppler_factor;                 /* Phase 6 - Step 7 */
            double comov_energy = pkt_energy * doppler_factor;         /* Phase 6 - Step 7 */

            /* Phase 6 - Step 7: Update estimators (atomicAdd) */
            d_update_base_estimators(d_j_estimator, d_nu_bar_estimator, /* Phase 6 - Step 7 */
                                      d_j_nu_estimator, nlte_n_freq_bins,
                                      nlte_nu_min, nlte_d_log_nu,
                                      shell, n_shells, distance, comov_nu,
                                      comov_energy);                     /* Phase 6 - Step 7 */
        }

        /* Phase 6 - Step 7: Handle interaction */
        if (interaction_type == 0) { /* Phase 6 - Step 7: BOUNDARY */
            int next_shell = pkt_shell_id + delta_shell; /* Phase 6 - Step 7 */
            if (next_shell >= n_shells) { /* Phase 6 - Step 7: escaped */
                pkt_status = 1; /* Phase 6 - Step 7: PACKET_EMITTED */
            } else if (next_shell < 0) { /* Phase 6 - Step 7: reabsorbed */
                if (d_diffuse_inner_bc) {
                    /* Task #27 Phase 2a: luminosity-conserving diffusive lower
                     * boundary. Re-emit from the photosphere instead of killing
                     * the packet: preserve its lab-frame energy bundle, place it
                     * back at r_inner in shell 0 with a fresh outward direction,
                     * and re-thermalize its comoving frequency to Planck(T_inner)
                     * (the same Bjorkman-Wood sampler used at launch). pkt_status
                     * stays 0 so transport continues; loop_count is NOT reset, so
                     * the interaction cap still bounds total work. */
                    pkt_r = d_r_inner[0];
                    pkt_shell_id = 0;
                    pkt_mu = sqrt(d_rng_uniform(rng));
                    double comov_nu_re = d_sample_bw_planck_nu(rng, kT_h);
                    double inv_dopp_re = d_get_inverse_doppler_factor(pkt_r, pkt_mu, t_exp);
                    pkt_nu = comov_nu_re * inv_dopp_re;
                    /* re-init next line id at the new comoving frequency */
                    double comov_nu_re2 = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                    int lo_re = 0, hi_re = n_lines;
                    while (lo_re < hi_re) {
                        int mid_re = (lo_re + hi_re) / 2;
                        if (d_line_list_nu[mid_re] > comov_nu_re2) lo_re = mid_re + 1;
                        else hi_re = mid_re;
                    }
                    if (lo_re == n_lines) lo_re = n_lines - 1;
                    pkt_next_line_id = lo_re;
                    atomicAdd(&d_n_returned_dev, 1ULL);
                } else {
                    pkt_status = 2; /* Phase 6 - Step 7: PACKET_REABSORBED */
                }
            } else { /* Phase 6 - Step 7 */
                pkt_shell_id = next_shell; /* Phase 6 - Step 7 */
            }
        } else if (interaction_type == 1) { /* Phase 6 - Step 7: LINE */
            if (d_cap_real_only) loop_count++; /* #5: count real interactions only */
            d_line_scatter_event(&pkt_r, &pkt_mu, &pkt_nu, &pkt_energy, /* Phase 6 - Step 7 */
                                  &pkt_next_line_id, pkt_shell_id,       /* Phase 6 - Step 7 */
                                  t_exp, line_interaction_type,           /* Phase 6 - Step 7 */
                                  d_line_list_nu,                         /* Phase 6 - Step 7 */
                                  n_shells, n_macro_levels,               /* Phase 6 - Step 7 */
                                  d_macro_block_references,               /* Phase 6 - Step 7 */
                                  d_transition_probabilities,             /* Phase 6 - Step 7 */
                                  d_destination_level_id,                 /* Phase 6 - Step 7 */
                                  d_transition_type,                      /* Phase 6 - Step 7 */
                                  d_transition_line_id,                   /* Phase 6 - Step 7 */
                                  d_line2macro_level_upper,               /* Phase 6 - Step 7 */
                                  d_line_atomic_number,
                                  d_line_ion_number,
                                  fe_scatter_mode,
                                  d_T_rad, n_lines,
                                  rng);                                   /* Phase 6 - Step 7 */
            /* Virtual packet: trace from interaction point */
            if (d_virtual_spectrum != NULL) {
                double nu_cmf_v = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                for (int vp = 0; vp < N_VPACKETS; vp++) {
                    d_trace_virtual_packet(pkt_r, pkt_shell_id, nu_cmf_v, pkt_energy,
                                            t_exp, L_inner,
                                            d_r_inner, d_r_outer,
                                            d_line_list_nu, d_tau_sobolev,
                                            d_electron_density,
                                            d_chi_bf, bf_enabled, bf_n_freq_bins,
                                            bf_nu_min, bf_nu_max, bf_d_log_nu,
                                            n_lines, n_shells, N_VPACKETS,
                                            d_virtual_spectrum, rng);
                }
            }
        } else if (interaction_type == 2) { /* Phase 6 - Step 7: CONTINUUM (e-scatter or BF) */
            if (d_cap_real_only) loop_count++; /* #5: count real interactions only */
            /* Branch: Thomson scattering vs BF absorption */
            if (chi_bf_val > 0.0 && d_rng_uniform(rng) > chi_e / chi_continuum) {
                /* BF macro-atom channel: route through macro-atom if activation level available */
                double comov_nu_bf2 = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                int act_level = d_bf_get_activation_level(d_bf_activation_level,
                    bf_n_freq_bins, bf_nu_min, bf_nu_max, bf_d_log_nu,
                    pkt_shell_id, comov_nu_bf2);
                if (act_level >= 0) {
                    /* Isotropic re-emission + macro-atom cascade.
                     * Bug #7 fix: old_doppler must be evaluated with the OLD mu
                     * (before resample) so the lab→comov boost uses the
                     * incoming direction and the comov→lab boost uses the
                     * outgoing direction. Prior order computed old_doppler
                     * AFTER resample → both factors used NEW mu, leaving
                     * pkt_energy unchanged regardless of direction. */
                    double old_doppler = d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                    pkt_mu = d_rng_mu(rng);
                    double inv_new_doppler = d_get_inverse_doppler_factor(pkt_r, pkt_mu, t_exp);
                    pkt_energy *= old_doppler;       /* lab → comov, OLD mu */
                    pkt_energy *= inv_new_doppler;   /* comov → lab, NEW mu */
                    pkt_nu = comov_nu_bf2 * inv_new_doppler;
                    /* [MA-FATE] entry comov nu for BF activation = pre-absorption photon nu */
                    double ma_entry_comov_nu = comov_nu_bf2;
                    /* Run macro-atom cascade */
                    int transition_id, transition_type_ma;
                    d_macro_atom_interaction(act_level, pkt_shell_id,
                        n_shells, n_macro_levels,
                        d_macro_block_references,
                        d_transition_probabilities,
                        d_destination_level_id,
                        d_transition_type,
                        d_transition_line_id,
                        rng, &transition_id, &transition_type_ma);
                    if (transition_type_ma == -1) { /* MA_BB_EMISSION */
                        /* P8: Orphaned level → fall back to nearest line */
                        int emit_line = (transition_id >= 0) ? transition_id : pkt_next_line_id;
                        d_line_emission(&pkt_nu, &pkt_next_line_id, emit_line,
                                         pkt_r, pkt_mu, t_exp, d_line_list_nu);
                    }
                    /* [MA-FATE] exit comov nu after cascade */
                    double ma_exit_comov_nu = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                    d_ma_fate_record(ma_entry_comov_nu, ma_exit_comov_nu);
                } else {
                    d_bf_absorption_event(&pkt_r, &pkt_mu, &pkt_nu,
                                           &pkt_next_line_id, t_exp,
                                           d_T_rad, pkt_shell_id,
                                           d_line_list_nu, n_lines, rng);
                }
            } else {
                d_thomson_scatter(&pkt_r, &pkt_mu, &pkt_nu, &pkt_energy,
                                   t_exp, rng);
            }
            /* Virtual packet from continuum interaction */
            if (d_virtual_spectrum != NULL) {
                double nu_cmf_v = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                for (int vp = 0; vp < N_VPACKETS; vp++) {
                    d_trace_virtual_packet(pkt_r, pkt_shell_id, nu_cmf_v, pkt_energy,
                                            t_exp, L_inner,
                                            d_r_inner, d_r_outer,
                                            d_line_list_nu, d_tau_sobolev,
                                            d_electron_density,
                                            d_chi_bf, bf_enabled, bf_n_freq_bins,
                                            bf_nu_min, bf_nu_max, bf_d_log_nu,
                                            n_lines, n_shells, N_VPACKETS,
                                            d_virtual_spectrum, rng);
                }
            }
        }
    }

    /* Phase 6 - Step 7: Store results */
    if (pkt_status == 1) { /* Phase 6 - Step 7: EMITTED */
        d_escaped_flag[p] = 1;        /* Phase 6 - Step 7 */
        d_escaped_nu[p] = pkt_nu;     /* Phase 6 - Step 7 */
        d_escaped_energy[p] = pkt_energy; /* Phase 6 - Step 7 */
        d_escaped_r[p] = pkt_r;       /* Rotation mode: store escape r */
        d_escaped_mu[p] = pkt_mu;     /* Rotation mode: store escape mu */
        atomicAdd((unsigned long long *)d_n_escaped, 1ULL); /* Phase 6 - Step 7 */
    } else if (pkt_status == 2) { /* Phase 6 - Step 7: REABSORBED */
        d_escaped_flag[p] = 0;        /* Phase 6 - Step 7 */
        atomicAdd((unsigned long long *)d_n_reabsorbed, 1ULL); /* Phase 6 - Step 7 */
        atomicAdd(&d_E_reabsorbed_dev, pkt_energy); /* Task #27 Phase 0 */
    } else { /* Phase 6 - Step 7: still in process (loop limit) */
        atomicAdd(&d_n_capped_dev, 1ULL); /* [CAP] packet hit max_interactions */
        if (pkt_shell_id >= 0 && pkt_shell_id < n_shells && pkt_shell_id < CAP_SHELL_MAX)
            atomicAdd(&d_capped_by_shell[pkt_shell_id], 1ULL); /* [CAP-SHELL] depth of scissored packet */
        if (d_cap_force_escape) {
            /* #5 fix: conserve energy — bin the capped packet at its current
             * (nu,energy) instead of deleting it. Prevents the L_emitted deficit
             * that over-heats T_inner via update_t_inner's (L_em/L_req)^-0.5.
             * Energy goes to esc (NOT d_E_truncated_dev) so E-BUDGET stays single-counted. */
            d_escaped_flag[p] = 1;
            d_escaped_nu[p] = pkt_nu;
            d_escaped_energy[p] = pkt_energy;
            d_escaped_r[p] = pkt_r;
            d_escaped_mu[p] = pkt_mu;
            atomicAdd((unsigned long long *)d_n_escaped, 1ULL);
        } else {
            atomicAdd(&d_E_truncated_dev, pkt_energy); /* legacy: drop (energy lost) */
            d_escaped_flag[p] = 0;
        }
    }
}

/* ============================================================ */
/* Phase 6 - Step 7: RNG init kernel                            */
/* ============================================================ */
__global__
void rng_init_kernel(uint64_t *d_rng_states, int n_packets,
                      uint64_t base_seed) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; /* Phase 6 - Step 7 */
    if (p >= n_packets) return;                     /* Phase 6 - Step 7 */
    uint64_t *s = &d_rng_states[p * 4];            /* Phase 6 - Step 7 */
    /* A10: SplitMix64 mix — see transport_kernel for rationale. */
    uint64_t _sm = base_seed ^ ((uint64_t)p * 0x9e3779b97f4a7c15ULL);
    uint64_t _seed_p = d_splitmix64(&_sm);
    d_rng_init(s, _seed_p);
}

/* ============================================================ */
/* Phase 6 - Step 8: Host driver (main function)                */
/* ============================================================ */

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL); /* Phase 6 - Step 8: unbuffered output */
    printf("============================================================\n"); /* Phase 6 - Step 8 */
    printf("LUMINA-SN v2.0 CUDA — Phase 6 GPU Transport\n");                 /* Phase 6 - Step 8 */
    printf("============================================================\n"); /* Phase 6 - Step 8 */

    /* Phase 6 - Step 8: Print GPU info */
    int device; /* Phase 6 - Step 8 */
    CUDA_CHECK(cudaGetDevice(&device)); /* Phase 6 - Step 8 */
    cudaDeviceProp prop; /* Phase 6 - Step 8 */
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device)); /* Phase 6 - Step 8 */
    printf("GPU: %s (SM %d.%d, %.1f GB VRAM)\n", /* Phase 6 - Step 8 */
           prop.name, prop.major, prop.minor,     /* Phase 6 - Step 8 */
           prop.totalGlobalMem / 1073741824.0);   /* Phase 6 - Step 8 */

    /* Phase 6 - Step 8: Load TARDIS reference data (reuse lumina_atomic.c) */
    Geometry geo;        /* Phase 6 - Step 8 */
    OpacityState opacity; /* Phase 6 - Step 8 */
    PlasmaState plasma;  /* Phase 6 - Step 8 */
    MCConfig config;     /* Phase 6 - Step 8 */
    AtomicData atom_data; /* Task #072 */
    memset(&config, 0, sizeof(config)); /* Phase 6 - Step 8 */

    config.enable_full_relativity = false;       /* Phase 6 - Step 8 */
    config.disable_line_scattering = false;      /* Phase 6 - Step 8 */
    config.line_interaction_type = LINE_MACROATOM; /* Phase 6 - Step 8 */
    {
        const char *li_env = getenv("LUMINA_LINE_INTERACTION");
        if (li_env) {
            if      (strcmp(li_env, "scatter")    == 0 || strcmp(li_env, "0") == 0) config.line_interaction_type = LINE_SCATTER;
            else if (strcmp(li_env, "downbranch") == 0 || strcmp(li_env, "1") == 0) config.line_interaction_type = LINE_DOWNBRANCH;
            else if (strcmp(li_env, "macroatom")  == 0 || strcmp(li_env, "macro") == 0 || strcmp(li_env, "2") == 0) config.line_interaction_type = LINE_MACROATOM;
            else fprintf(stderr, "[WARN] unknown LUMINA_LINE_INTERACTION=%s, keeping macroatom\n", li_env);
        }
    }
    config.damping_constant = 0.5;               /* Phase 6 - Step 8 */
    config.hold_iterations = 3;                  /* Phase 6 - Step 8 */

    const char *ref_dir = "data/tardis_reference";    /* Phase 6 - Step 8 */
    if (argc > 1) ref_dir = argv[1];             /* Phase 6 - Step 8 */

    if (load_tardis_reference_data(ref_dir, &geo, &opacity, &plasma, &config) != 0) { /* Phase 6 - Step 8 */
        fprintf(stderr, "Failed to load reference data\n"); /* Phase 6 - Step 8 */
        return 1; /* Phase 6 - Step 8 */
    }

    /* Task #072: Load atomic data for plasma solver */
    if (load_atomic_data(&atom_data, ref_dir, geo.n_shells) != 0) {
        fprintf(stderr, "Failed to load atomic data\n");
        return 1;
    }
    /* Task #38: Optional pre-baked CMFGEN sigma_bf grid (per-level ν-dependent
     * photoionization). LUMINA_CMFGEN_SIGMA_BF semantics:
     *   unset / "1" / "on" / "yes" → load default path
     *   "0" / "off" / "no"          → skip load (Kramers fallback)
     *   anything containing '/'     → explicit path override */
    {
        const char *cmf_env = getenv("LUMINA_CMFGEN_SIGMA_BF");
        const char *cmf_path = "data/atomic/cmfgen_sigma_bf.bin";
        int cmf_enable = 1;
        if (cmf_env) {
            if (!strcmp(cmf_env, "0") || !strcmp(cmf_env, "off") || !strcmp(cmf_env, "no"))
                cmf_enable = 0;
            else if (strchr(cmf_env, '/'))
                cmf_path = cmf_env;
        }
        if (cmf_enable) load_cmfgen_sigma_bf(&atom_data, cmf_path);
    }
    /* Top-stage continuum anchor: inject synthetic IV ground levels (gated,
     * default off). MUST run after cmfgen_sigma_bf load (extends it) and before
     * nlte_init / GPU upload (they pick up the extended n_levels). */
    inject_topstage_continuum_levels(&atom_data, &opacity);
    /* Task #072: Initialize n_electron from TARDIS reference */
    plasma.n_electron = (double *)malloc(geo.n_shells * sizeof(double));
    for (int i = 0; i < geo.n_shells; i++)
        plasma.n_electron[i] = opacity.electron_density[i];

    /* P6: Initialize per-shell electron temperature */
    plasma.T_e = (double *)malloc(geo.n_shells * sizeof(double));
    compute_electron_temperature(&plasma, NULL, geo.time_explosion, geo.n_shells, 0);

    int n_packets = config.n_packets;
    if (argc > 2) n_packets = atoi(argv[2]);
    int n_iterations = config.n_iterations;
    if (argc > 3) n_iterations = atoi(argv[3]);

    /* Spectrum mode: "real" (default), "virtual", "rotation", "all" */
    int enable_virtual = 0, enable_rotation = 0;
    if (argc > 4) {
        if (strcmp(argv[4], "virtual") == 0) enable_virtual = 1;
        else if (strcmp(argv[4], "rotation") == 0) enable_rotation = 1;
        else if (strcmp(argv[4], "both") == 0) enable_virtual = 1;
        else if (strcmp(argv[4], "all") == 0) { enable_virtual = 1; enable_rotation = 1; }
    }

    /* NLTE mode: argv[5] == "nlte" or env LUMINA_NLTE=1 */
    int enable_nlte = 0;
    if (argc > 5 && strcmp(argv[5], "nlte") == 0) enable_nlte = 1;
    if (getenv("LUMINA_NLTE") && atoi(getenv("LUMINA_NLTE")) > 0) enable_nlte = 1;
    config.enable_nlte = enable_nlte;

    /* NLTE start iteration: default 0 (all iters), env LUMINA_NLTE_START_ITER=N */
    int nlte_start_iter = 0;
    if (getenv("LUMINA_NLTE_START_ITER"))
        nlte_start_iter = atoi(getenv("LUMINA_NLTE_START_ITER"));

    /* Dynamic transition probability update: default OFF, enable with LUMINA_DYNAMIC_TRANSPROB=1.
     * Parse the VALUE (not mere presence) so =0 truly disables — the old existence
     * check turned the dynamic B_lu·J up-pump ON for every run that set =0 (#257). */
    int enable_transprob_update = 0;
    if (getenv("LUMINA_DYNAMIC_TRANSPROB"))
        enable_transprob_update = atoi(getenv("LUMINA_DYNAMIC_TRANSPROB"));

    /* Fe two-level atom scatter: LUMINA_FE_SCATTER=1 (Fe II only) or =2 (all Fe) */
    int fe_scatter = 0;
    if (getenv("LUMINA_FE_SCATTER"))
        fe_scatter = atoi(getenv("LUMINA_FE_SCATTER"));

    /* [CAP] per-packet interaction cap (default 100000). Bounds the unbounded
     * macro-atom transport through DDC15's dense iron-curtain so iterations
     * complete; cap-hit count is printed per iteration. */
    int max_interactions = 100000;
    if (getenv("LUMINA_MAX_INTERACTIONS"))
        max_interactions = atoi(getenv("LUMINA_MAX_INTERACTIONS"));
    if (max_interactions < 1) max_interactions = 1;
    cuda_set_max_interactions(max_interactions);
    printf("  [CAP] LUMINA_MAX_INTERACTIONS=%d\n", max_interactions);
    fflush(stdout);

    /* #5 cap-semantics knobs (default OFF = legacy behaviour for clean A/B).
     *  LUMINA_CAP_REAL_ONLY=1   : count only LINE/CONTINUUM events toward the cap;
     *                             boundary crossings bounded by LUMINA_MAX_TOTAL_STEPS.
     *  LUMINA_CAP_FORCE_ESCAPE=1: on cap-hit, bin the packet (conserve energy) instead
     *                             of dropping it. */
    int cap_real_only = 0;
    if (getenv("LUMINA_CAP_REAL_ONLY"))
        cap_real_only = atoi(getenv("LUMINA_CAP_REAL_ONLY"));
    cuda_set_cap_real_only(cap_real_only);
    printf("  [CAP] LUMINA_CAP_REAL_ONLY=%d\n", cap_real_only);

    int cap_force_escape = 0;
    if (getenv("LUMINA_CAP_FORCE_ESCAPE"))
        cap_force_escape = atoi(getenv("LUMINA_CAP_FORCE_ESCAPE"));
    cuda_set_cap_force_escape(cap_force_escape);
    printf("  [CAP] LUMINA_CAP_FORCE_ESCAPE=%d\n", cap_force_escape);

    int max_total_steps = 2000000;
    if (getenv("LUMINA_MAX_TOTAL_STEPS"))
        max_total_steps = atoi(getenv("LUMINA_MAX_TOTAL_STEPS"));
    if (max_total_steps < 1) max_total_steps = 1;
    cuda_set_max_total_steps(max_total_steps);
    printf("  [CAP] LUMINA_MAX_TOTAL_STEPS=%d\n", max_total_steps);
    fflush(stdout);

    /* Macro-atom internal cascade cap (default 5000). Diagnostic probe for the
     * cascade-depth hypothesis: shortening it forces early deactivation. */
    int ma_internal_cap = 5000;
    if (getenv("LUMINA_MA_INTERNAL_CAP"))
        ma_internal_cap = atoi(getenv("LUMINA_MA_INTERNAL_CAP"));
    if (ma_internal_cap < 1) ma_internal_cap = 1;
    cuda_set_ma_internal_cap(ma_internal_cap);
    printf("  [CAP] LUMINA_MA_INTERNAL_CAP=%d\n", ma_internal_cap);
    fflush(stdout);

    /* Task #27 Phase 2a: diffusive (luminosity-conserving) inner boundary. */
    int diffuse_inner_bc = 0;
    if (getenv("LUMINA_DIFFUSE_INNER_BC"))
        diffuse_inner_bc = atoi(getenv("LUMINA_DIFFUSE_INNER_BC"));
    cuda_set_diffuse_inner_bc(diffuse_inner_bc);
    printf("  [INNER-BC] LUMINA_DIFFUSE_INNER_BC=%d (%s)\n", diffuse_inner_bc,
           diffuse_inner_bc ? "diffusive: re-emit returned packets at T_inner"
                            : "absorbing hard sphere (default)");
    fflush(stdout);

    /* [EPS-UV] thermalization knob: probability that a UV-entry macro-atom
     * call is replaced by Planck(T_rad) re-emission. Set with LUMINA_EPS_UV. */
    {
        double eps_uv = 0.0;
        if (getenv("LUMINA_EPS_UV"))
            eps_uv = atof(getenv("LUMINA_EPS_UV"));
        if (eps_uv < 0.0) eps_uv = 0.0;
        if (eps_uv > 1.0) eps_uv = 1.0;
        cuda_set_eps_uv(eps_uv);
        if (eps_uv > 0.0)
            printf("[EPS-UV] UV-entry macro-atom thermalization probability = %.3f\n",
                   eps_uv);
    }

    /* [EPS-IR] NIR-entry counterpart: probability of replacing a macro-atom
     * call with Planck(T_rad) thermalization for activations at lam>7000 A.
     * Set with LUMINA_EPS_IR. */
    {
        double eps_ir = 0.0;
        if (getenv("LUMINA_EPS_IR"))
            eps_ir = atof(getenv("LUMINA_EPS_IR"));
        if (eps_ir < 0.0) eps_ir = 0.0;
        if (eps_ir > 1.0) eps_ir = 1.0;
        cuda_set_eps_ir(eps_ir);
        if (eps_ir > 0.0)
            printf("[EPS-IR] NIR-entry macro-atom thermalization probability = %.3f\n",
                   eps_ir);
    }

    /* [H2 EPS-UV red-only] When set, the EPS_UV gate fires post-cascade only
     * when the cascade exit lands in [5500,10000)Å — preserves UV→UV/blue
     * downbranching, suppresses only UV→red. */
    {
        int red_only = 0;
        if (getenv("LUMINA_EPS_UV_RED_ONLY"))
            red_only = atoi(getenv("LUMINA_EPS_UV_RED_ONLY")) > 0 ? 1 : 0;
        cuda_set_eps_uv_red_only(red_only);
        if (red_only)
            printf("[EPS-UV] red-only mode ON: gate fires post-cascade, exit band 5/6 only\n");
    }

    /* [EPS-UV 2STEP] True 2-step UV→opt→red cascade. Re-emits into an
     * optical band [LO,HI]Å (Planck(T_rad) rejection sample) instead of full
     * thermalization, so red flux later emerges from natural optical line
     * physics carrying proper P-Cygni shape. Defaults to [3500,5500]Å. */
    {
        int on = 0;
        double lo_A = 3500.0, hi_A = 5500.0;
        if (getenv("LUMINA_EPS_UV_2STEP"))
            on = atoi(getenv("LUMINA_EPS_UV_2STEP")) > 0 ? 1 : 0;
        if (getenv("LUMINA_EPS_UV_2STEP_BAND_LO"))
            lo_A = atof(getenv("LUMINA_EPS_UV_2STEP_BAND_LO"));
        if (getenv("LUMINA_EPS_UV_2STEP_BAND_HI"))
            hi_A = atof(getenv("LUMINA_EPS_UV_2STEP_BAND_HI"));
        if (hi_A <= lo_A) { hi_A = lo_A + 1.0; }
        cuda_set_eps_uv_2step(on, lo_A, hi_A);
        if (on)
            printf("[EPS-UV-2STEP] band-constrained re-emit: λ∈[%.0f,%.0f]Å (Planck(T_rad) rejection)\n",
                   lo_A, hi_A);
    }

    /* [H3] per-(Z, ion, entry_band, exit_band) attribution histogram. */
    int ma_fate_zi_enabled = 0;
    if (getenv("LUMINA_MA_FATE_ZIHIST") &&
        atoi(getenv("LUMINA_MA_FATE_ZIHIST")) > 0)
        ma_fate_zi_enabled = 1;
    cuda_set_ma_fate_zi_enabled(ma_fate_zi_enabled);
    if (ma_fate_zi_enabled)
        printf("[MA-FATE] per-(Z,ion,band,band) attribution histogram ENABLED\n");

    /* Gamma-ray deposition: LUMINA_GAMMA_DEP=1 */
    int gamma_dep_enabled = 0;
    if (getenv("LUMINA_GAMMA_DEP") && atoi(getenv("LUMINA_GAMMA_DEP")) > 0)
        gamma_dep_enabled = 1;

    /* Line overlap correction: LUMINA_OVERLAP_CORR=1 (handled inside compute_plasma_state) */
    int overlap_corr_enabled = (getenv("LUMINA_OVERLAP_CORR") &&
                                 atoi(getenv("LUMINA_OVERLAP_CORR")) > 0);

    /* Bound-free opacity: LUMINA_BF_OPACITY=1 */
    int bf_opacity_enabled = (getenv("LUMINA_BF_OPACITY") &&
                               atoi(getenv("LUMINA_BF_OPACITY")) > 0);

    /* P6: Self-consistent T_e: LUMINA_SELF_CONSISTENT_TE=1 */
    int self_consistent_te = (getenv("LUMINA_SELF_CONSISTENT_TE") &&
                               atoi(getenv("LUMINA_SELF_CONSISTENT_TE")) > 0);
    /* Task #20: real radiative-equilibrium T_e (heating=cooling): LUMINA_RADEQ_TE=1 */
    int cmfgen_then_mc = (getenv("LUMINA_CMFGEN_THEN_MC") &&
                          atoi(getenv("LUMINA_CMFGEN_THEN_MC")));
    int radeq_te = (getenv("LUMINA_RADEQ_TE") &&
                     atoi(getenv("LUMINA_RADEQ_TE")) > 0);

    printf("\nSimulation parameters:\n");
    printf("  Packets: %d, Iterations: %d\n", n_packets, n_iterations);
    printf("  Line interaction: %s\n",
        config.line_interaction_type == LINE_SCATTER    ? "SCATTER" :
        config.line_interaction_type == LINE_DOWNBRANCH ? "DOWNBRANCH" : "MACROATOM");
    const char *mode_str = "real only";
    if (enable_virtual && enable_rotation) mode_str = "real + virtual + rotation";
    else if (enable_virtual) mode_str = "real + virtual";
    else if (enable_rotation) mode_str = "real + rotation";
    printf("  Spectrum mode: %s\n", mode_str);
    if (enable_nlte && nlte_start_iter > 0)
        printf("  NLTE: ENABLED from iter %d (first %d non-NLTE)\n",
               nlte_start_iter + 1, nlte_start_iter);
    else
        printf("  NLTE: %s\n", enable_nlte ? "ENABLED (all iters)" : "disabled");
    {
        const char *t_pin_env = getenv("LUMINA_T_INNER_FIX");
        if (t_pin_env) {
            double t_pin = atof(t_pin_env);
            if (t_pin > 0.0) {
                printf("  T_inner: %.2f K (overridden by LUMINA_T_INNER_FIX, was %.2f)\n",
                       t_pin, config.T_inner);
                config.T_inner = t_pin;
            }
        }
    }
    printf("  T_inner: %.2f K\n", config.T_inner);
    printf("  Transition probs: %s\n", enable_transprob_update ? "DYNAMIC" : "FROZEN");
    printf("  Fe scatter: %s\n", fe_scatter == 2 ? "ALL Fe TWO-LEVEL" :
                                  fe_scatter == 1 ? "Fe II TWO-LEVEL" : "MACRO-ATOM");
    printf("  Gamma-ray deposition: %s\n", gamma_dep_enabled ? "ENABLED" : "disabled");
    printf("  Line overlap correction: %s\n", overlap_corr_enabled ? "ENABLED" : "disabled");
    printf("  BF+FF opacity: %s\n", bf_opacity_enabled ? "ENABLED" : "disabled");
    if (radeq_te || self_consistent_te)
        printf("  T_e: RADIATIVE EQUILIBRIUM full balance (heating=cooling, no free params)\n");
    else
        printf("  Self-consistent T_e: disabled (ratio=%.2f)\n", plasma.T_e_T_rad_ratio);
    double spec_min = 500.0, spec_max = 20000.0;
    int spec_bins = 2000;
    if (getenv("LUMINA_SPEC_RANGE")) {
        sscanf(getenv("LUMINA_SPEC_RANGE"), "%lf,%lf,%d", &spec_min, &spec_max, &spec_bins);
        printf("  Spectrum range: %.0f-%.0f A, %d bins\n", spec_min, spec_max, spec_bins);
    }

    /* Multi-epoch rescaling: override t_exp via environment variable */
    const char *time_exp_env = getenv("LUMINA_TIME_EXPLOSION");
    if (time_exp_env) {
        double t_new_days = atof(time_exp_env);
        double t_new = t_new_days * 86400.0;
        printf("  Epoch rescale: t_exp %.2f -> %.2f days (ratio %.4f)\n",
               geo.time_explosion / 86400.0, t_new_days, t_new / geo.time_explosion);
        rescale_epoch(&geo, &plasma, t_new);
    }

    /* Phase 6 - Step 8: Compute shell volumes */
    double *volume = (double *)malloc(geo.n_shells * sizeof(double)); /* Phase 6 - Step 8 */
    for (int i = 0; i < geo.n_shells; i++) { /* Phase 6 - Step 8 */
        volume[i] = (4.0 / 3.0) * M_PI_VAL * /* Phase 6 - Step 8 */
            (geo.r_outer[i] * geo.r_outer[i] * geo.r_outer[i] - /* Phase 6 - Step 8 */
             geo.r_inner[i] * geo.r_inner[i] * geo.r_inner[i]); /* Phase 6 - Step 8 */
    }

    /* Phase 6 - Step 8: Create CPU estimators and spectrum */
    Estimators *est = create_estimators(geo.n_shells, opacity.n_lines); /* Phase 6 - Step 8 */
    Spectrum *spec = create_spectrum(spec_min, spec_max, spec_bins);

    /* Phase 6 - Step 8: Allocate and upload GPU data */
    CudaDeviceData dev; /* Phase 6 - Step 8 */
    memset(&dev, 0, sizeof(dev)); /* Phase 6 - Step 8 */
    cuda_allocate(&dev, &geo, &opacity, n_packets); /* Phase 6 - Step 8 */
    cuda_upload(&dev, &geo, &opacity);               /* Phase 6 - Step 8 */
    CUDA_CHECK(cudaMemcpy(dev.d_line_atomic_number, atom_data.line_atomic_number,
               opacity.n_lines * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_line_ion_number, atom_data.line_ion_number,
               opacity.n_lines * sizeof(int), cudaMemcpyHostToDevice));
    printf("  GPU memory allocated and uploaded.\n"); /* Phase 6 - Step 8 */

    /* [KPACKET] Bind device pointers + enable flag to the selector's global
     * symbols. Tables themselves are populated by compute_transition_probabilities
     * (needs LUMINA_DYNAMIC_TRANSPROB=1) and re-uploaded each iteration. */
    {
        int kpacket_on = (dev.d_p_kpacket != NULL) ? 1 : 0;
        cuda_set_kpacket(dev.d_p_kpacket, dev.d_kpacket_cdf,
                         opacity.n_macro_levels, kpacket_on);
        if (kpacket_on) {
            printf("  [KPACKET] collisional/k-packet thermal pool ENABLED "
                   "(%d levels x %d shells)\n", opacity.n_macro_levels, geo.n_shells);
            /* The tables are built ONLY inside compute_transition_probabilities,
             * which runs only on the dynamic-transprob path. Without it the
             * device tables stay all-zero and k-packet silently does nothing. */
            if (!enable_transprob_update)
                printf("  [KPACKET][WARN] LUMINA_KPACKET=1 but LUMINA_DYNAMIC_TRANSPROB"
                       " is OFF — k-packet tables will NEVER be populated and the "
                       "feature is a NO-OP. Set LUMINA_DYNAMIC_TRANSPROB=1.\n");
        }
    }

    /* NLTE: Initialize if enabled */
    NLTEConfig nlte;
    memset(&nlte, 0, sizeof(nlte));
    CudaNLTESolver nlte_solver;
    memset(&nlte_solver, 0, sizeof(nlte_solver));
    if (enable_nlte) {
        printf("\n--- NLTE Initialization ---\n");
        nlte_init(&nlte, &atom_data, &opacity, geo.n_shells);
        cuda_allocate_nlte(&dev, &nlte, geo.n_shells);
        /* Find max level count across all ion pairs for cuBLAS allocation.
         * #281: use explicit pair table (overlap pair 15 lo=29 not 2*15=30). */
        int max_N = 0;
        const int pair_lo_init[NLTE_PAIR_COUNT] = {
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29
        };
        for (int p = 0; p < NLTE_PAIR_COUNT; p++) {
            int lo = pair_lo_init[p], hi = lo + 1;
            int N = nlte.nlte_ion_level_offset[hi + 1] -
                    nlte.nlte_ion_level_offset[lo];
            if (N > max_N) max_N = N;
        }
        cuda_nlte_solver_init(&nlte_solver, max_N, geo.n_shells);
    }

    /* Gamma-ray deposition: initialize if enabled */
    GammaDeposition gamma_dep;
    memset(&gamma_dep, 0, sizeof(gamma_dep));
    if (gamma_dep_enabled) {
        gamma_deposition_init(&gamma_dep, geo.n_shells);
        printf("\n--- Gamma-ray Deposition Initialized ---\n");
    }

    /* BF opacity: initialize if enabled */
    BFOpacity bf;
    memset(&bf, 0, sizeof(bf));
    if (bf_opacity_enabled) {
        bf_opacity_init(&bf, geo.n_shells);
        compute_bf_opacity(&bf, &atom_data, &plasma, geo.n_shells);
        cuda_allocate_bf(&dev, &bf, geo.n_shells);
        cuda_upload_bf(&dev, &bf, &plasma, geo.n_shells);
        printf("--- BF+FF Opacity Initialized (%d freq bins) ---\n", bf.n_freq_bins);
    }

    /* T_rad upload (always; needed by EPS_UV / EPS_IR macro-atom paths even
     * when BF is disabled). cuda_upload_bf already covers BF-on case. */
    if (!bf_opacity_enabled)
        cuda_upload_T_rad(&dev, &plasma, geo.n_shells);

    /* Phase 6 - Step 8: Host-side escaped packet buffers */
    double *h_escaped_nu = (double *)malloc(n_packets * sizeof(double));     /* Phase 6 - Step 8 */
    double *h_escaped_energy = (double *)malloc(n_packets * sizeof(double)); /* Phase 6 - Step 8 */
    int *h_escaped_flag = (int *)malloc(n_packets * sizeof(int));            /* Phase 6 - Step 8 */
    double *h_escaped_r = (double *)malloc(n_packets * sizeof(double));      /* Rotation mode */
    double *h_escaped_mu = (double *)malloc(n_packets * sizeof(double));     /* Rotation mode */

    /* Phase 6 - Step 8: Kernel launch config */
    int threads_per_block = 256; /* Phase 6 - Step 8 */
    int blocks = (n_packets + threads_per_block - 1) / threads_per_block; /* Phase 6 - Step 8 */
    printf("  Kernel launch: %d blocks x %d threads\n", blocks, threads_per_block); /* Phase 6 - Step 8 */

    /* ============================================================ */
    /* PURE-CMFGEN parallel path (LUMINA_PURE_CMFGEN=1): bypass MC,  */
    /* deterministic J_nu + downstream solvers (CPU), dump plasma.   */
    /* ============================================================ */
    {
        const char *_pure = getenv("LUMINA_PURE_CMFGEN");
        if (_pure && atoi(_pure)) {
            const char *_ni = getenv("LUMINA_PURE_CMFGEN_ITER");
            int pc_iter = _ni ? atoi(_ni) : n_iterations;
            if (pc_iter < 1) pc_iter = 1;
            const char *_ali = getenv("LUMINA_CMFGEN_ALI_ITER");
            int n_ali = _ali ? atoi(_ali) : 8;
            if (n_ali < 1) n_ali = 1;
            printf("\n=== PURE-CMFGEN deterministic radiation path "
                   "(GPU MC transport bypassed; NLTE on GPU) ===\n");

            /* Orchestrate the CMFGEN loop here so the NLTE step uses the GPU
             * (GEMM) solver — cmfgen_run() internally uses the slow CPU NLTE. */
            CMFGENState cs;
            if (cmfgen_init(&cs, &geo) == 0) {
                for (int it = 0; it < pc_iter; it++) {
                    nlte.current_iter = it;
                    if (bf_opacity_enabled)
                        compute_bf_opacity(&bf, &atom_data, &plasma, geo.n_shells);
                    cmfgen_assemble(&cs, &geo, &opacity,
                                    bf_opacity_enabled ? &bf : NULL, &plasma);
                    cmfgen_solve_J(&cs, &geo, config.T_inner, n_ali);
                    cmfgen_window_color(&cs);
                    radeq_set_tail_color(cs.t_color, cs.n_shells);
                    radeq_set_tri_response(cs.tri_lo, cs.tri_up, cs.tri_r,
                                           cs.n_shells, cs.n_bins);
                    if (cs.diag && it == pc_iter - 1)
                        cmfgen_validate(&cs, &geo, &plasma);
                    cmfgen_write_jnu(&cs, &nlte);
                    /* Option-2 integral RE: register the CMFGEN line opacity/
                     * source so the Newton T_e solve can add the radiative line
                     * term (LUMINA_RADEQ_LINE_RE=1). */
                    /* RE/Newton line channel: chi_line_th except transfer-only
                     * eps_uv mode (FULL chi_line, cooling-only closure). */
                    radeq_set_line_re_source(cs.chi_line_re, cs.chi_abs, cs.chi_tot,
                                             cs.S_fixed, cs.J, cs.nu, cs.dnu,
                                             cs.lambda_star, plasma.T_e,
                                             cs.chi_line, cs.chi_line_cls,
                                             cs.n_shells, cs.n_bins);

                    compute_radiative_equilibrium_te(&plasma,
                        gamma_dep_enabled ? &gamma_dep : NULL,
                        enable_nlte ? &nlte : NULL, &atom_data, &opacity,
                        geo.time_explosion, geo.n_shells);
                    compute_plasma_state(&atom_data, &plasma, &opacity,
                                         geo.time_explosion);
                    if (getenv("LUMINA_COUPLED_NEWTON") &&
                        atoi(getenv("LUMINA_COUPLED_NEWTON")) && enable_nlte)
                        coupled_newton_solve_all(&plasma,
                            gamma_dep_enabled ? &gamma_dep : NULL,
                            &nlte, &atom_data, &opacity, &geo,
                            geo.time_explosion, geo.n_shells);
                    if (enable_nlte && it >= nlte_start_iter) {
                        nlte_apply_uv_jnu_cap(&nlte, &plasma, geo.n_shells);
                        nlte_solve_all_gpu(&nlte, &atom_data, &plasma, &opacity,
                                           geo.time_explosion, geo.n_shells,
                                           &nlte_solver,
                                           gamma_dep_enabled ? &gamma_dep : NULL);
                    }
                    printf("[CMFGEN] iter %2d: T_e[0]=%.0fK T_e[%d]=%.0fK "
                           "T_e[%d]=%.0fK J[mid,500]=%.3e\n", it,
                           plasma.T_e[0], geo.n_shells/2,
                           plasma.T_e[geo.n_shells/2], geo.n_shells-1,
                           plasma.T_e[geo.n_shells-1],
                           cs.J[(size_t)(geo.n_shells/2)*cs.n_bins + 500]);
                }
                cmfgen_write_spectrum(&cs, &geo, config.T_inner,
                                      "lumina_spectrum.csv");
                cmfgen_free(&cs);
            }

            /* DIAGNOSTIC (LUMINA_SL_DUMP=1): per-line two-level source S_l vs
             * the LTE thermal value B_nu(T_e) at the SAME converged T_e, for
             * every line with tau>cutoff. Proves the spectrum defect is the
             * line SOURCE FUNCTION (level populations), not T_e/n_e: if S_l/B
             * >> 1 in the UV while T_e/n_e match gold, the intermediate check
             * (T_e,n_e) under-determined the spectrum. */
            if (getenv("LUMINA_SL_DUMP")) {
                FILE *sf = fopen("lumina_sl_vs_B.csv", "w");
                if (sf) {
                    /* line_id + n_e + Jline (exact rate-matrix field) added so
                     * the FAITHFUL in-run S_l/J_line test (codex decider) and
                     * the two-level-with-collisions prediction (from line_list
                     * A/B + T_e,n_e) can be reconstructed offline. */
                    fprintf(sf, "shell,line_id,lambda_A,Te,ne,Jline,Sl,B_Te,Sl_over_B,tau\n");
                    double hpl = 6.62607015e-27, kb = 1.380649e-16,
                           cc = 2.99792458e10;
                    for (int l = 0; l < opacity.n_lines; l++) {
                        double nu_l = opacity.line_list_nu[l];
                        double lamA = cc / nu_l * 1e8;
                        for (int s = 0; s < geo.n_shells; s++) {
                            double tau = opacity.tau_sobolev[(size_t)l*geo.n_shells+s];
                            if (tau < 1e-3) continue;
                            double Sl = opacity.line_source_S
                                ? opacity.line_source_S[(size_t)l*geo.n_shells+s] : 0.0;
                            if (Sl <= 0.0) continue;
                            double Te = plasma.T_e[s];
                            double x = hpl*nu_l/(kb*Te);
                            double B = (x<500.0)? 2.0*hpl*nu_l*nu_l*nu_l/(cc*cc)/(exp(x)-1.0) : 0.0;
                            if (B <= 0.0) continue;
                            double Jline = nlte_get_J_at_nu(&nlte, s, nu_l);
                            double ne = plasma.n_electron[s];
                            fprintf(sf, "%d,%d,%.2f,%.1f,%.4e,%.4e,%.4e,%.4e,%.4e,%.3e\n",
                                    s, l, lamA, Te, ne, Jline, Sl, B, Sl/B, tau);
                        }
                    }
                    fclose(sf);
                    printf("S_l vs B(T_e) dump -> lumina_sl_vs_B.csv\n");
                }
            }

            /* DIAGNOSTIC (LUMINA_LEVELPOP_DUMP=1): per-level departure coefficient
             * b_k relative to its OWN-ion ground at the local T_e:
             *   b_k = (n_k/n_ground) / ((g_k/g_ground) exp(-(E_k-E_ground)/kTe))
             * b_k=1 -> thermal (Boltzmann); b_k>>1 -> super-thermal (the actual
             * population overpopulation that drives S_l/B>>1). This dumps the
             * quantity the S_l/B proxy only INFERRED: which levels, of which ion,
             * at which shell, are over/under populated and by how much. */
            if (getenv("LUMINA_LEVELPOP_DUMP")) {
                FILE *lp = fopen("lumina_levelpop.csv", "w");
                if (lp) {
                    fprintf(lp, "shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k,has_sigma,n_sig_pos\n");
                    const double kB_eV = 8.617333262e-5; /* eV/K */
                    int nfb = atom_data.cmfgen_n_freq_bins;
                    int n_sh = geo.n_shells;
                    for (int i = 0; i < nlte.n_nlte_ions; i++) {
                        int Z = nlte.nlte_Z[i], ion = nlte.nlte_ion[i];
                        int l0 = nlte.nlte_ion_level_offset[i];
                        int l1 = nlte.nlte_ion_level_offset[i + 1];
                        int g_glo = nlte.nlte_to_global_level[l0];
                        double Eg = atom_data.level_energy_eV[g_glo];
                        int gg = atom_data.level_g[g_glo]; if (gg < 1) gg = 1;
                        for (int s = 0; s < n_sh; s++) {
                            double Te = plasma.T_e[s];
                            double ng = nlte.nlte_level_populations[(size_t)l0 * n_sh + s];
                            for (int l = l0; l < l1; l++) {
                                int gl = nlte.nlte_to_global_level[l];
                                double Ek = atom_data.level_energy_eV[gl];
                                int gk = atom_data.level_g[gl]; if (gk < 1) gk = 1;
                                double nk = nlte.nlte_level_populations[(size_t)l * n_sh + s];
                                double bk = -1.0;
                                if (ng > 0.0 && nk > 0.0 && Te > 0.0) {
                                    double boltz = ((double)gk / (double)gg) *
                                        exp(-(Ek - Eg) / (kB_eV * Te));
                                    if (boltz > 0.0) bk = (nk / ng) / boltz;
                                }
                                int hs = atom_data.cmfgen_has_sigma ?
                                    atom_data.cmfgen_has_sigma[gl] : 0;
                                int nsp = 0;
                                if (hs && atom_data.cmfgen_sigma_bf) {
                                    const double *sr = &atom_data.cmfgen_sigma_bf[(size_t)gl * nfb];
                                    for (int b = 0; b < nfb; b++) if (sr[b] > 0.0) nsp++;
                                }
                                fprintf(lp, "%d,%d,%d,%d,%.4f,%d,%.6e,%.6e,%.4e,%d,%d\n",
                                        s, Z, ion, atom_data.level_num[gl], Ek, gk, nk, ng, bk, hs, nsp);
                            }
                        }
                    }
                    fclose(lp);
                    printf("Per-level departure b_k dump -> lumina_levelpop.csv\n");
                }
            }

            /* Validated OBSERVER-FRAME spectra on the converged pure-CMFGEN
             * state. The pure path used to exit with only the comoving Path-5
             * cmfgen_write_spectrum (no inter-shell Doppler -> no P-Cygni).
             * Path 3 (Lucy-1999 tangent-ray formal integral, per-line Sobolev
             * Doppler) always; Path 4 (CMF finite-profile + line overlap,
             * Blondin+2013) gated by LUMINA_TRANSPORT=cmf. Both consume
             * plasma+opacity(tau_sobolev)+nlte, all converged here. */
            {
                Spectrum *spec_fi = create_spectrum(spec_min, spec_max, spec_bins);
                compute_formal_integral_spectrum(
                    &geo, &plasma, &opacity, &atom_data,
                    nlte.enabled ? &nlte : NULL, config.T_inner, spec_fi, 100);
                FILE *ff = fopen("lumina_spectrum_formal.csv", "w");
                if (ff) {
                    fprintf(ff, "wavelength_angstrom,flux\n");
                    for (int i = 0; i < spec_fi->n_bins; i++)
                        fprintf(ff, "%.6f,%.6e\n",
                                spec_fi->wavelength[i], spec_fi->flux[i]);
                    fclose(ff);
                    printf("Formal integral spectrum -> lumina_spectrum_formal.csv\n");
                }
                free_spectrum(spec_fi);

                const char *_tr = getenv("LUMINA_TRANSPORT");
                if (_tr && strcmp(_tr, "cmf") == 0) {
                    const char *_nz = getenv("LUMINA_CMF_NZ");
                    const char *_ni = getenv("LUMINA_CMF_NIMPACT");
                    const char *_vt = getenv("LUMINA_CMF_VTURB_KMS");
                    int cmf_nz = _nz ? atoi(_nz) : 2000;
                    int cmf_ni = _ni ? atoi(_ni) : 50;
                    double vturb = (_vt ? atof(_vt) : 0.0) * 1.0e5;
                    if (cmf_nz < 1) cmf_nz = 2000;
                    if (cmf_ni < 1) cmf_ni = 50;
                    Spectrum *spec_cmf = create_spectrum(spec_min, spec_max, spec_bins);
                    compute_cmf_formal_spectrum(
                        &geo, &plasma, &opacity, &atom_data,
                        nlte.enabled ? &nlte : NULL,
                        bf_opacity_enabled ? &bf : NULL,
                        config.T_inner, spec_cmf, cmf_ni, cmf_nz, vturb);
                    FILE *cf = fopen("lumina_spectrum_cmf.csv", "w");
                    if (cf) {
                        fprintf(cf, "wavelength_angstrom,flux\n");
                        for (int i = 0; i < spec_cmf->n_bins; i++)
                            fprintf(cf, "%.6f,%.6e\n",
                                    spec_cmf->wavelength[i], spec_cmf->flux[i]);
                        fclose(cf);
                        printf("CMF formal spectrum -> lumina_spectrum_cmf.csv\n");
                    }
                    free_spectrum(spec_cmf);
                }
            }

            FILE *pf = fopen("lumina_plasma_state.csv", "w");
            if (pf) {
                fprintf(pf, "shell_id,W,T_rad,n_e,T_e\n");
                for (int i = 0; i < geo.n_shells; i++)
                    fprintf(pf, "%d,%.10f,%.6f,%.6e,%.6f\n", i,
                            plasma.W[i], plasma.T_rad[i],
                            plasma.n_electron[i], plasma.T_e[i]);
                fclose(pf);
                printf("Pure-CMFGEN plasma state written to "
                       "lumina_plasma_state.csv\n");
            }
            printf("\nDone (pure-CMFGEN).\n");
            /* LUMINA_CMFGEN_THEN_MC=1: do NOT return — fall through to the MC
             * transport loop with the plasma FROZEN at the converged
             * pure-CMFGEN state, to synthesize a macro-atom (fluorescence)
             * observer-frame spectrum on the GOOD plasma. The clean diagonal
             * test: good T_e/n_e (pure-CMFGEN) + multi-level fluorescence
             * (MC macro-atom). MC-loop count = argv N_ITER (independent of
             * LUMINA_PURE_CMFGEN_ITER); plasma solve skipped (cmfgen_then_mc). */
            if (!cmfgen_then_mc) return 0;
            printf("[THEN-MC] pure-CMFGEN converged; entering FROZEN-plasma MC "
                   "macro-atom spectrum pass (%d transport iters)\n", n_iterations);
        }
    }

    /* Phase 6 - Step 8: Iteration loop */
    for (int iter = 0; iter < n_iterations; iter++) { /* Phase 6 - Step 8 */
        printf("\n--- Iteration %d/%d ---\n", iter + 1, n_iterations); /* Phase 6 - Step 8 */

        /* Phase 6 - Step 8: Reset estimators */
        reset_estimators(est);   /* Phase 6 - Step 8: CPU estimators */
        reset_spectrum(spec);    /* Phase 6 - Step 8 */
        cuda_reset_estimators(&dev, geo.n_shells); /* Phase 6 - Step 8: GPU estimators */
        CUDA_CHECK(cudaMemset(dev.d_escaped_flag, 0, n_packets * sizeof(int))); /* Phase 6 - Step 8 */
        /* [MA-FATE] reset device-side macro-atom fate hist; aggregate
         * only on the final iteration so the printout reflects the
         * converged radiation field. */
        cuda_ma_fate_reset();
        cuda_ma_fate_zi_reset();
        cuda_ma_cycle_reset();
        cuda_kpacket_count_reset();
        cuda_n_capped_reset();
        if (iter == n_iterations - 1) {
            macro_atom_fate_reset();
            macro_atom_fate_zi_reset();
            macro_atom_cycle_reset();
        }

        /* Phase 6 - Step 8: Recompute L_inner, time_simulation, packet_energy */
        double L_inner = 4.0 * M_PI_VAL * geo.r_inner[0] * geo.r_inner[0] * /* Phase 6 - Step 8 */
                         SIGMA_SB * pow(config.T_inner, 4);                   /* Phase 6 - Step 8 */
        double time_simulation = 1.0 / L_inner;                               /* Phase 6 - Step 8 */
        double packet_energy = 1.0 / (double)n_packets;                       /* Phase 6 - Step 8 */

        /* Phase 6 - Step 8: RNG seed for this iteration */
        uint64_t iter_seed = config.seed + (uint64_t)iter * 1000000ULL; /* Phase 6 - Step 8 */

        /* Phase 6 - Step 8: Launch transport kernel */
        cudaEvent_t start_ev, stop_ev; /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaEventCreate(&start_ev)); /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaEventCreate(&stop_ev));  /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaEventRecord(start_ev));  /* Phase 6 - Step 8 */

        transport_kernel<<<blocks, threads_per_block>>>(
            dev.d_r_inner, dev.d_r_outer,
            dev.d_line_list_nu, dev.d_tau_sobolev,
            dev.d_electron_density,
            dev.d_transition_probabilities,
            dev.d_macro_block_references,
            dev.d_transition_type,
            dev.d_destination_level_id,
            dev.d_transition_line_id,
            dev.d_line2macro_level_upper,
            dev.d_line_atomic_number,
            dev.d_line_ion_number,
            fe_scatter,
            dev.d_j_estimator, dev.d_nu_bar_estimator,
            dev.d_j_nu_estimator,
            dev.nlte_n_freq_bins, dev.nlte_nu_min, dev.nlte_d_log_nu,
            dev.d_rng_states,
            dev.d_escaped_nu, dev.d_escaped_energy,
            dev.d_escaped_flag,
            dev.d_escaped_r, dev.d_escaped_mu,
            dev.d_n_escaped, dev.d_n_reabsorbed,
            enable_virtual ? dev.d_virtual_spectrum : (double *)NULL, L_inner,
            dev.d_chi_bf, dev.d_T_rad, dev.d_bf_activation_level,
            dev.bf_enabled, dev.bf_n_freq_bins,
            dev.bf_nu_min, dev.bf_nu_max, dev.bf_d_log_nu,
            n_packets, geo.n_shells, opacity.n_lines,
            opacity.n_macro_levels,
            geo.time_explosion, config.T_inner,
            packet_energy, config.line_interaction_type,
            iter_seed);

        CUDA_CHECK(cudaEventRecord(stop_ev));  /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaEventSynchronize(stop_ev)); /* Phase 6 - Step 8 */
        float elapsed_ms; /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev)); /* Phase 6 - Step 8 */

        /* Phase 6 - Step 8: Check for kernel errors */
        CUDA_CHECK(cudaGetLastError()); /* Phase 6 - Step 8 */

        /* [CAP] report packets that hit the interaction cap this iteration */
        {
            unsigned long long n_capped = cuda_n_capped_get();
            printf("  [CAP] iter%d: %llu / %d packets hit max_interactions=%d (%.3f%%)  transport %.1f ms\n",
                   iter, n_capped, n_packets, max_interactions,
                   100.0 * (double)n_capped / (double)n_packets, elapsed_ms);
            if (n_capped > 0) {
                unsigned long long cbs[256];
                cuda_capped_by_shell_get(cbs, geo.n_shells);
                printf("  [CAP-SHELL] iter%d cap-hit by shell (0=inner/photosphere): ", iter);
                for (int s = 0; s < geo.n_shells; s++)
                    if (cbs[s] > 0) printf("s%d=%llu ", s, cbs[s]);
                printf("\n");
            }
            if (diffuse_inner_bc) {
                unsigned long long n_returned = cuda_n_returned_get();
                printf("  [INNER-BC] iter%d: %llu packet re-emissions at inner boundary (%.2f per packet)\n",
                       iter, n_returned, (double)n_returned / (double)n_packets);
            }
            fflush(stdout);
        }

        /* [MA-FATE] aggregate device-side counts on the final iteration */
        if (iter == n_iterations - 1) {
            cuda_ma_fate_download_and_aggregate();
            cuda_ma_fate_zi_download_and_aggregate();
            cuda_ma_cycle_download_and_aggregate();
        }

        /* [KPACKET] per-iteration collisional-deactivation event count. */
        if (dev.d_p_kpacket) {
            unsigned long long nkp = cuda_kpacket_count_get();
            printf("  [KPACKET] collisional deactivations this iter: %llu\n", nkp);
        }

        /* Phase 6 - Step 8: Download results */
        cuda_download_estimators(&dev, est->j_estimator, est->nu_bar_estimator, /* Phase 6 - Step 8 */
                                  geo.n_shells); /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaMemcpy(h_escaped_nu, dev.d_escaped_nu,       /* Phase 6 - Step 8 */
                   n_packets * sizeof(double), cudaMemcpyDeviceToHost)); /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaMemcpy(h_escaped_energy, dev.d_escaped_energy, /* Phase 6 - Step 8 */
                   n_packets * sizeof(double), cudaMemcpyDeviceToHost)); /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaMemcpy(h_escaped_flag, dev.d_escaped_flag,    /* Phase 6 - Step 8 */
                   n_packets * sizeof(int), cudaMemcpyDeviceToHost)); /* Phase 6 - Step 8 */
        if (enable_rotation) {
            CUDA_CHECK(cudaMemcpy(h_escaped_r, dev.d_escaped_r,
                       n_packets * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_escaped_mu, dev.d_escaped_mu,
                       n_packets * sizeof(double), cudaMemcpyDeviceToHost));
        }

        int64_t n_escaped = 0, n_reabsorbed = 0; /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaMemcpy(&n_escaped, dev.d_n_escaped,           /* Phase 6 - Step 8 */
                   sizeof(int64_t), cudaMemcpyDeviceToHost));         /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaMemcpy(&n_reabsorbed, dev.d_n_reabsorbed,     /* Phase 6 - Step 8 */
                   sizeof(int64_t), cudaMemcpyDeviceToHost));         /* Phase 6 - Step 8 */

        double escape_fraction = (double)n_escaped / n_packets; /* Phase 6 - Step 8 */
        printf("  GPU kernel: %.1f ms (%.1f us/packet)\n",      /* Phase 6 - Step 8 */
               elapsed_ms, elapsed_ms * 1000.0 / n_packets);    /* Phase 6 - Step 8 */
        printf("  Escaped: %ld (%.2f%%), Reabsorbed: %ld (%.2f%%)\n", /* Phase 6 - Step 8 */
               (long)n_escaped, 100.0 * escape_fraction,         /* Phase 6 - Step 8 */
               (long)n_reabsorbed, 100.0 * n_reabsorbed / n_packets); /* Phase 6 - Step 8 */

        /* Phase 6 - Step 8: Spectrum binning + L_emitted (CPU) */
        double L_emitted = 0.0;
        for (int i = 0; i < n_packets; i++) { /* Phase 6 - Step 8 */
            if (h_escaped_flag[i]) { /* Phase 6 - Step 8 */
                bin_escaped_packet(spec, h_escaped_nu[i],
                                    h_escaped_energy[i] * L_inner);
                L_emitted += h_escaped_energy[i] * L_inner;
            }
        }

        /* Task #27 Phase 0: energy-budget decomposition. Localizes the
         * L_req - L_emitted deficit (which the T_inner self-pin compensates by
         * over-heating) into reabsorbed (back-scatter trapping) vs truncated
         * (interaction-cap packet loss). Gated by LUMINA_ENERGY_BUDGET=1. */
        if (getenv("LUMINA_ENERGY_BUDGET") &&
            atoi(getenv("LUMINA_ENERGY_BUDGET")) != 0) {
            double L_reabs = cuda_E_reabsorbed_get() * L_inner;
            double L_trunc = cuda_E_truncated_get() * L_inner;
            double L_req = config.luminosity_requested;
            double L_acc = L_emitted + L_reabs + L_trunc;
            printf("  [E-BUDGET iter%d] L_req=%.4e  esc=%.2f%%  reabs=%.2f%%  "
                   "trunc=%.2f%%  (accounted=%.2f%% of L_req)\n",
                   iter, L_req,
                   100.0 * L_emitted / L_req,
                   100.0 * L_reabs   / L_req,
                   100.0 * L_trunc   / L_req,
                   100.0 * L_acc     / L_req);
        }

        /* Binned-J estimator: download the raw J_nu histogram and expose it on
         * est so solve_radiation_field can fit a dilute Planck to the
         * frequency-resolved field (instead of the redshift-fragile nu_bar/j
         * moment). Gated by LUMINA_BINNED_J_ESTIMATOR to avoid an extra D2H
         * copy on the moment-baseline default. */
        if (enable_nlte && getenv("LUMINA_BINNED_J_ESTIMATOR") &&
            atoi(getenv("LUMINA_BINNED_J_ESTIMATOR")) != 0) {
            cuda_download_j_nu(&dev, &nlte, geo.n_shells);
            est->j_nu_estimator   = nlte.j_nu_estimator;
            est->nlte_n_freq_bins = nlte.n_freq_bins;
            est->nlte_nu_min      = nlte.nu_min;
            est->nlte_d_log_nu    = nlte.d_log_nu;
        }

        /* Option (8): freeze W/T_rad too once ion-lock activates — true
         * transport-only iteration; plasma state from converged free-NLTE iter. */
        if (!(iter > 0 && nlte_ion_lock_active(iter))) {
            /* Phase 6 - Step 8: Solve radiation field (CPU, reuse lumina_plasma.c) */
            solve_radiation_field(est, geo.time_explosion, time_simulation,
                                   volume, &opacity, &plasma,
                                   config.damping_constant);
        }

        /* Task #072: Recompute tau_sobolev and re-upload to GPU.
         * Option (8): skip ALL plasma updates once ion-lock activates — freeze
         * plasma at the converged free-NLTE state, transport packets only. */
        if (iter > 0 && nlte_ion_lock_active(iter)) {
            printf("  [plasma frozen by ion-lock; transport-only iter %d]\n", iter);
        } else if (iter > 0) {
            /* Gamma-ray deposition: compute heating/ionization rates */
            if (gamma_dep_enabled) {
                compute_gamma_deposition(&gamma_dep, &atom_data, &plasma, &geo);
                printf("  [Gamma] heating_rate[0]=%.2e, [%d]=%.2e erg/s/cm3\n",
                       gamma_dep.heating_rate[0], geo.n_shells - 1,
                       gamma_dep.heating_rate[geo.n_shells - 1]);
            }

            /* P6: Update per-shell T_e before plasma state.
             * Both LUMINA_RADEQ_TE and LUMINA_SELF_CONSISTENT_TE route to the
             * complete radiative-equilibrium balance (photoionization + Compton +
             * gamma heating vs. recombination + free-free + collisional bound-bound
             * + adiabatic cooling, no free parameters). The old Compton-only +
             * f_coll_boost path is retired for the self-consistent flag. */
            /* THEN-MC: the plasma is FROZEN at the converged pure-CMFGEN
             * state — skip the entire T_e / ionization / coupled-Newton /
             * NLTE re-solve. Keep BF re-upload + transition-probability build
             * + device uploads below so the macro-atom transport sees the
             * good frozen state. */
            if (cmfgen_then_mc) goto frozen_skip_plasma_solve;
            if (radeq_te || self_consistent_te) {
                /* Radiative-equilibrium T_e needs the CURRENT iteration's
                 * radiation field for photoionization heating. The MC pass
                 * for this iter is already complete, so download+normalize
                 * J_nu now (the later NLTE block re-normalizes harmlessly,
                 * since normalize recomputes J_nu from the raw estimator). */
                if (enable_nlte && iter >= nlte_start_iter) {
                    cuda_download_j_nu(&dev, &nlte, geo.n_shells);
                    nlte_normalize_j_nu(&nlte, time_simulation, volume, geo.n_shells);
                }
                compute_radiative_equilibrium_te(&plasma,
                    gamma_dep_enabled ? &gamma_dep : NULL,
                    &nlte, &atom_data, &opacity,
                    geo.time_explosion, geo.n_shells);
            } else {
                compute_electron_temperature(&plasma,
                    gamma_dep_enabled ? &gamma_dep : NULL,
                    geo.time_explosion, geo.n_shells, self_consistent_te);
            }

            compute_plasma_state(&atom_data, &plasma, &opacity, geo.time_explosion);

            /* PATH-A / A2: replace the operator-split RADEQ→ionization fixed point
             * on non-frozen inner shells with the simultaneous coupled-Newton
             * {n_e, T_e} solve (gated LUMINA_COUPLED_NEWTON=1). */
            if (getenv("LUMINA_COUPLED_NEWTON") &&
                atoi(getenv("LUMINA_COUPLED_NEWTON")) && enable_nlte)
                coupled_newton_solve_all(&plasma,
                    gamma_dep_enabled ? &gamma_dep : NULL,
                    &nlte, &atom_data, &opacity, &geo,
                    geo.time_explosion, geo.n_shells);

        frozen_skip_plasma_solve:
            /* Recompute BF opacity with updated plasma and re-upload to GPU */
            if (bf_opacity_enabled) {
                compute_bf_opacity(&bf, &atom_data, &plasma, geo.n_shells);
                cuda_upload_bf(&dev, &bf, &plasma, geo.n_shells);
            } else {
                /* EPS_UV / EPS_IR need fresh T_rad even when BF is disabled. */
                cuda_upload_T_rad(&dev, &plasma, geo.n_shells);
            }

            /* NLTE: solve rate equations and update tau for NLTE lines.
             * THEN-MC freezes the converged pure-CMFGEN level populations —
             * skip the re-solve (it would over-write the good frozen state). */
            if (!cmfgen_then_mc && enable_nlte && iter >= nlte_start_iter) {
                nlte.current_iter = iter;
                cuda_download_j_nu(&dev, &nlte, geo.n_shells);
                nlte_normalize_j_nu(&nlte, time_simulation, volume, geo.n_shells);
                nlte_apply_uv_jnu_cap(&nlte, &plasma, geo.n_shells);
                nlte_solve_all_gpu(&nlte, &atom_data, &plasma, &opacity,
                                    geo.time_explosion, geo.n_shells,
                                    &nlte_solver,
                                    gamma_dep_enabled ? &gamma_dep : NULL);
                /* tau_sobolev already updated inside nlte_solve_all_gpu */

                /* Re-apply overlap corrections after NLTE tau update */
                if (overlap_corr_enabled)
                    apply_overlap_corrections(&atom_data, &opacity, &plasma);
            }

            /* [TAU-DIAG] iter3-hang forensics: the macro-atom transport cost is
             * driven by the NUMBER of lines with tau > tau_event (~O(1)), i.e.
             * the iron-curtain density — not by any single line's magnitude.
             * NLTE first solves at end of iter (nlte_start_iter); if it inflates
             * the high-tau line count, macro-atom re-emission re-traverses the
             * forest and interaction counts explode. Also sanitize NaN/Inf -> floor
             * (NaN/Inf in tau are bugs; they would poison the tau-event compare). */
            {
                size_t ntau = (size_t)opacity.n_lines * geo.n_shells;
                long c1 = 0, c10 = 0, c100 = 0, c1e3 = 0, cbad = 0;
                double tmax = 0.0;
                for (size_t k = 0; k < ntau; k++) {
                    double t = opacity.tau_sobolev[k];
                    if (!isfinite(t)) { opacity.tau_sobolev[k] = 1e-100; cbad++; continue; }
                    if (t > tmax) tmax = t;
                    if (t > 1.0)   c1++;
                    if (t > 10.0)  c10++;
                    if (t > 100.0) c100++;
                    if (t > 1e3)   c1e3++;
                }
                printf("  [TAU-DIAG] post-NLTE iter%d: tau_max=%.3e  N(tau>1)=%ld  >10=%ld  >100=%ld  >1e3=%ld  NaN/Inf=%ld (sanitized)\n",
                       iter, tmax, c1, c10, c100, c1e3, cbad);
                fflush(stdout);
            }

            /* Dynamic transition probability recomputation. THEN-MC: ALWAYS
             * rebuild from the frozen converged NLTE level populations so the
             * macro-atom fluorescence branching reflects the good plasma. */
            if (cmfgen_then_mc) {
                compute_transition_probabilities(&atom_data, &plasma, &opacity,
                    enable_nlte ? &nlte : NULL, config.damping_constant, 1);
            } else if (enable_transprob_update && iter >= config.hold_iterations) {
                compute_transition_probabilities(&atom_data, &plasma, &opacity,
                    (enable_nlte && iter >= nlte_start_iter) ? &nlte : NULL,
                    config.damping_constant,
                    (iter > config.hold_iterations) ? 1 : 0);
            } else if (iter == 1) {
                /* Task #49: one-shot effective-branching dump on as-loaded
                 * transition_probabilities (frozen carsus values) using fresh
                 * iter-1 tau_sobolev.
                 * Sample shells 0, 3, n_shells/3 to expose inner-Fe vs outer
                 * physics (parallels compute_transition_probabilities). */
                diag_macro_branch(&atom_data, &plasma, &opacity, 0);
                if (geo.n_shells > 3)
                    diag_macro_branch(&atom_data, &plasma, &opacity, 3);
                if (geo.n_shells > 10)
                    diag_macro_branch(&atom_data, &plasma, &opacity,
                                      geo.n_shells / 3);
            }

            CUDA_CHECK(cudaMemcpy(dev.d_tau_sobolev, opacity.tau_sobolev,
                       (size_t)opacity.n_lines * geo.n_shells * sizeof(double),
                       cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev.d_electron_density, opacity.electron_density,
                       geo.n_shells * sizeof(double), cudaMemcpyHostToDevice));
            /* Re-upload updated transition probabilities to GPU */
            if (cmfgen_then_mc ||
                (enable_transprob_update && iter >= config.hold_iterations)) {
                CUDA_CHECK(cudaMemcpy(dev.d_transition_probabilities,
                           opacity.transition_probabilities,
                           (size_t)opacity.n_macro_transitions * geo.n_shells * sizeof(double),
                           cudaMemcpyHostToDevice));
                /* k-packet tables are rebuilt inside compute_transition_probabilities
                 * (they depend on T_e/n_e/J like the radiative rates); re-upload. */
                if (dev.d_p_kpacket && opacity.p_kpacket) {
                    CUDA_CHECK(cudaMemcpy(dev.d_p_kpacket, opacity.p_kpacket,
                               (size_t)opacity.n_macro_levels * geo.n_shells * sizeof(double),
                               cudaMemcpyHostToDevice));
                }
                if (dev.d_kpacket_cdf && opacity.kpacket_cdf) {
                    CUDA_CHECK(cudaMemcpy(dev.d_kpacket_cdf, opacity.kpacket_cdf,
                               (size_t)geo.n_shells * opacity.n_macro_levels * sizeof(double),
                               cudaMemcpyHostToDevice));
                }
            }
        }

        /* Phase 6 - Step 8: Print plasma state */
        printf("  Shell  W_LUMINA   T_rad_LUM   T_e_LUM    T_e/T_r   nubar/j\n");
        for (int i = 0; i < geo.n_shells; i += 5) {
            double ratio = est->nu_bar_estimator[i] / est->j_estimator[i];
            double te_ratio = plasma.T_rad[i] > 0 ? plasma.T_e[i] / plasma.T_rad[i] : 0.0;
            printf("  %3d    %.6f   %.2f K   %.2f K   %.4f   %.4e\n",
                   i, plasma.W[i], plasma.T_rad[i], plasma.T_e[i], te_ratio, ratio);
        }

        /* Phase 6 - Step 8: Update T_inner (after hold iterations) */
        if (iter >= config.hold_iterations) { /* Phase 6 - Step 8 */
            double old_T = config.T_inner; /* Phase 6 - Step 8 */
            int t_inner_frozen = nlte_ion_lock_active(iter);
            const char *t_pin_env = getenv("LUMINA_T_INNER_FIX");
            double t_pin = t_pin_env ? atof(t_pin_env) : 0.0;
            const char *diff_bc_env = getenv("LUMINA_DIFFUSION_INNER_BC");
            int diff_bc = diff_bc_env ? atoi(diff_bc_env) : 0;
            if (diff_bc) {
                /* A1 (path-A, 2-agent verified): fixed-L diffusion inner BC.
                 * CMFGEN fixes the base luminosity and lets T_inner follow the
                 * diffusion relation; it does NOT run a feedback controller that
                 * chases the (shot-noisy, reabsorption-biased) emergent L_em.
                 * The TARDIS-style update_t_inner = T_inner*(L_em/L_req)^-0.5 is
                 * an operator-split feedback loop: when ionization shifts and
                 * L_em transiently collapses it overshoots (4430->91163 K) and
                 * ping-pongs, pinning the inner T_e hot. Here T_inner is the
                 * Stefan-Boltzmann value consistent with the fixed L_req through
                 * the inner face, held constant (HD2012 sec.3.2.2). */
                double R_in = geo.r_inner[0];
                config.T_inner = pow(config.luminosity_requested /
                                     (4.0 * M_PI_VAL * R_in * R_in * SIGMA_SB),
                                     0.25);
                printf("  T_inner: %.2f K (fixed-L diffusion BC, L_req=%.3e, "
                       "L_em=%.3e)\n",
                       config.T_inner, config.luminosity_requested, L_emitted);
            } else if (t_pin > 0.0) {
                config.T_inner = t_pin;
                printf("  T_inner: %.2f K (pinned LUMINA_T_INNER_FIX, L_em=%.3e, L_req=%.3e)\n",
                       config.T_inner, L_emitted, config.luminosity_requested);
            } else if (!t_inner_frozen) {
                update_t_inner(&config, L_emitted);
                printf("  T_inner: %.2f K -> %.2f K (L_em=%.3e, L_req=%.3e)\n",
                       old_T, config.T_inner, L_emitted, config.luminosity_requested);
            } else {
                printf("  T_inner: %.2f K [frozen-by-lock] (L_em=%.3e, L_req=%.3e)\n",
                       config.T_inner, L_emitted, config.luminosity_requested);
            }
        } else { /* Phase 6 - Step 8 */
            printf("  T_inner: %.2f K (hold iteration %d/%d)\n", /* Phase 6 - Step 8 */
                   config.T_inner, iter + 1, config.hold_iterations); /* Phase 6 - Step 8 */
        }

        CUDA_CHECK(cudaEventDestroy(start_ev)); /* Phase 6 - Step 8 */
        CUDA_CHECK(cudaEventDestroy(stop_ev));  /* Phase 6 - Step 8 */
    }

    /* Phase 6 - Step 8: Final results comparison */
    printf("\n============================================================\n"); /* Phase 6 - Step 8 */
    printf("Final Results (CUDA)\n");                                           /* Phase 6 - Step 8 */
    printf("============================================================\n"); /* Phase 6 - Step 8 */

    /* [MA-FATE] Macro-atom packet fate histogram (final iteration) */
    macro_atom_fate_print("final iteration, GPU transport");
    macro_atom_cycle_print("final iteration, GPU transport");

    /* [H3] Dump per-(Z,ion,band,band) attribution CSV if enabled */
    if (ma_fate_zi_enabled) {
        macro_atom_fate_zi_dump_csv("ma_fate_zihist.csv",
            "final iteration, GPU transport");
    }

    char path[512]; /* Phase 6 - Step 8 */
    snprintf(path, sizeof(path), "%s/plasma_state.csv", ref_dir); /* Phase 6 - Step 8 */
    FILE *ref_fp = fopen(path, "r"); /* Phase 6 - Step 8 */
    double tardis_W[30], tardis_T_rad[30]; /* Phase 6 - Step 8 */
    if (ref_fp) { /* Phase 6 - Step 8 */
        char buf[1024]; /* Phase 6 - Step 8 */
        fgets(buf, sizeof(buf), ref_fp); /* Phase 6 - Step 8: skip header */
        int i = 0; /* Phase 6 - Step 8 */
        while (fgets(buf, sizeof(buf), ref_fp) && i < 30) { /* Phase 6 - Step 8 */
            int sid; /* Phase 6 - Step 8 */
            sscanf(buf, "%d,%lf,%lf", &sid, &tardis_W[i], &tardis_T_rad[i]); /* Phase 6 - Step 8 */
            i++; /* Phase 6 - Step 8 */
        }
        fclose(ref_fp); /* Phase 6 - Step 8 */

        printf("\nShell  W_LUMINA   W_TARDIS   W_err%%   T_rad_LUM  T_rad_TAR  T_err%%\n"); /* Phase 6 - Step 8 */
        printf("-----  --------   --------   ------   ---------  ---------  ------\n");      /* Phase 6 - Step 8 */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 6 - Step 8 */
            double w_err = (plasma.W[i] - tardis_W[i]) / tardis_W[i] * 100.0;    /* Phase 6 - Step 8 */
            double t_err = (plasma.T_rad[i] - tardis_T_rad[i]) / tardis_T_rad[i] * 100.0; /* Phase 6 - Step 8 */
            printf("  %3d  %8.6f   %8.6f   %+6.1f   %9.2f  %9.2f  %+6.1f\n",   /* Phase 6 - Step 8 */
                   i, plasma.W[i], tardis_W[i], w_err,                            /* Phase 6 - Step 8 */
                   plasma.T_rad[i], tardis_T_rad[i], t_err);                      /* Phase 6 - Step 8 */
        }

        double sum_w_err = 0.0, sum_t_err = 0.0; /* Phase 6 - Step 8 */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 6 - Step 8 */
            sum_w_err += fabs((plasma.W[i] - tardis_W[i]) / tardis_W[i]); /* Phase 6 - Step 8 */
            sum_t_err += fabs((plasma.T_rad[i] - tardis_T_rad[i]) / tardis_T_rad[i]); /* Phase 6 - Step 8 */
        }
        printf("\nMean |W error|: %.2f%%\n", sum_w_err / geo.n_shells * 100.0);     /* Phase 6 - Step 8 */
        printf("Mean |T_rad error|: %.2f%%\n", sum_t_err / geo.n_shells * 100.0);   /* Phase 6 - Step 8 */
        printf("T_inner final: %.2f K (TARDIS: 10521.52 K, err: %.2f%%)\n",         /* Phase 6 - Step 8 */
               config.T_inner,                                                        /* Phase 6 - Step 8 */
               (config.T_inner - 10521.52) / 10521.52 * 100.0);                      /* Phase 6 - Step 8 */
    }

    /* Write real spectrum to CSV */
    const char *output_file = "lumina_spectrum.csv";
    FILE *out = fopen(output_file, "w");
    if (out) {
        fprintf(out, "wavelength_angstrom,flux\n");
        for (int i = 0; i < spec->n_bins; i++) {
            fprintf(out, "%.6f,%.6e\n", spec->wavelength[i], spec->flux[i]);
        }
        fclose(out);
        printf("\nReal spectrum written to %s\n", output_file);
    }

    /* Download and write virtual spectrum */
    if (enable_virtual) {
        double *h_virtual_spectrum = (double *)calloc(VSPEC_N_BINS, sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_virtual_spectrum, dev.d_virtual_spectrum,
                   VSPEC_N_BINS * sizeof(double), cudaMemcpyDeviceToHost));
        FILE *vf = fopen("lumina_spectrum_virtual.csv", "w");
        if (vf) {
            fprintf(vf, "wavelength_angstrom,flux\n");
            double dlambda = (VSPEC_LAMBDA_MAX - VSPEC_LAMBDA_MIN) / VSPEC_N_BINS;
            for (int i = 0; i < VSPEC_N_BINS; i++) {
                double wl = VSPEC_LAMBDA_MIN + (i + 0.5) * dlambda;
                fprintf(vf, "%.6f,%.6e\n", wl, h_virtual_spectrum[i]);
            }
            fclose(vf);
            printf("Virtual spectrum written to lumina_spectrum_virtual.csv\n");
        }
        free(h_virtual_spectrum);
    }

    /* P5: Formal integral spectrum (noise-free) */
    {
        Spectrum *spec_fi = create_spectrum(spec_min, spec_max, spec_bins);
        compute_formal_integral_spectrum(
            &geo, &plasma, &opacity, &atom_data,
            nlte.enabled ? &nlte : NULL, config.T_inner,
            spec_fi, 100);
        FILE *ff = fopen("lumina_spectrum_formal.csv", "w");
        if (ff) {
            fprintf(ff, "wavelength_angstrom,flux\n");
            for (int i = 0; i < spec_fi->n_bins; i++)
                fprintf(ff, "%.6f,%.6e\n", spec_fi->wavelength[i], spec_fi->flux[i]);
            fclose(ff);
            printf("Formal integral spectrum written to lumina_spectrum_formal.csv\n");
        }
        free_spectrum(spec_fi);
    }

    /* CMF formal solver (paper-method line transfer), gated by LUMINA_TRANSPORT=cmf */
    {
        const char *_transport = getenv("LUMINA_TRANSPORT");
        if (_transport && strcmp(_transport, "cmf") == 0) {
            const char *_nz   = getenv("LUMINA_CMF_NZ");
            const char *_nimp = getenv("LUMINA_CMF_NIMPACT");
            const char *_vt   = getenv("LUMINA_CMF_VTURB_KMS");
            int cmf_nz   = _nz   ? atoi(_nz)   : 2000;
            int cmf_nimp = _nimp ? atoi(_nimp) : 50;
            double v_turb_cms = (_vt ? atof(_vt) : 0.0) * 1.0e5;
            if (cmf_nz < 1) cmf_nz = 2000;
            if (cmf_nimp < 1) cmf_nimp = 50;

            Spectrum *spec_cmf = create_spectrum(spec_min, spec_max, spec_bins);
            compute_cmf_formal_spectrum(
                &geo, &plasma, &opacity, &atom_data,
                nlte.enabled ? &nlte : NULL,
                bf_opacity_enabled ? &bf : NULL,
                config.T_inner, spec_cmf, cmf_nimp, cmf_nz, v_turb_cms);
            FILE *cf = fopen("lumina_spectrum_cmf.csv", "w");
            if (cf) {
                fprintf(cf, "wavelength_angstrom,flux\n");
                for (int i = 0; i < spec_cmf->n_bins; i++)
                    fprintf(cf, "%.6f,%.6e\n", spec_cmf->wavelength[i], spec_cmf->flux[i]);
                fclose(cf);
                printf("CMF formal spectrum written to lumina_spectrum_cmf.csv\n");
            }
            free_spectrum(spec_cmf);
        }
    }

    /* Rotation spectrum: Doppler-weight escaped packets (post-processing) */
    if (enable_rotation) {
        double L_inner_final = 4.0 * M_PI_VAL * geo.r_inner[0] * geo.r_inner[0] *
                               SIGMA_SB * pow(config.T_inner, 4);
        Spectrum *spec_rot = create_spectrum(spec_min, spec_max, spec_bins);
        double weight_sum = 0.0;
        int n_rot = 0;
        for (int i = 0; i < n_packets; i++) {
            if (h_escaped_flag[i]) {
                double beta = h_escaped_r[i] / (C_SPEED_OF_LIGHT * geo.time_explosion);
                double D_pkt = 1.0 - beta * h_escaped_mu[i];
                double D_obs = 1.0 - beta * 1.0; /* mu_obs = 1 (face-on) */
                double w = (D_obs / D_pkt) * (D_obs / D_pkt);
                bin_escaped_packet(spec_rot, h_escaped_nu[i],
                                    h_escaped_energy[i] * L_inner_final * w);
                weight_sum += w;
                n_rot++;
            }
        }
        FILE *rf = fopen("lumina_spectrum_rotation.csv", "w");
        if (rf) {
            fprintf(rf, "wavelength_angstrom,flux\n");
            for (int i = 0; i < spec_rot->n_bins; i++) {
                fprintf(rf, "%.6f,%.6e\n", spec_rot->wavelength[i], spec_rot->flux[i]);
            }
            fclose(rf);
            printf("Rotation spectrum written to lumina_spectrum_rotation.csv\n");
            printf("  Mean rotation weight: %.6f (N=%d)\n",
                   n_rot > 0 ? weight_sum / n_rot : 0.0, n_rot);
        }
        free_spectrum(spec_rot);
    }

    /* Phase 6 - Step 8: Write final plasma state */
    out = fopen("lumina_plasma_state.csv", "w"); /* Phase 6 - Step 8 */
    if (out) { /* Phase 6 - Step 8 */
        fprintf(out, "shell_id,W,T_rad,n_e,T_e\n"); /* +T_e: RADEQ solver target, vs CMFGEN gas temperature */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 6 - Step 8 */
            fprintf(out, "%d,%.10f,%.6f,%.6e,%.6f\n", i, plasma.W[i], plasma.T_rad[i], plasma.n_electron[i], plasma.T_e[i]); /* Phase 6 - Step 8 */
        }
        fclose(out); /* Phase 6 - Step 8 */
        printf("Plasma state written to lumina_plasma_state.csv\n"); /* Phase 6 - Step 8 */
    }

    /* Per-stage ion-population dump: true ionization state incl. Saha-treated
     * neutral stages (the NLTE level dump is blind to non-NLTE ion stages). */
    if (getenv("LUMINA_ION_POP_DUMP") && atoi(getenv("LUMINA_ION_POP_DUMP"))) {
        FILE *ip = fopen("lumina_ion_pops.csv", "w");
        if (ip) {
            fprintf(ip, "shell_id,Z,stage,n_ion\n");
            for (int s = 0; s < geo.n_shells; s++)
                for (int j = 0; j < atom_data.n_ion_pops; j++)
                    fprintf(ip, "%d,%d,%d,%.6e\n", s,
                            atom_data.ion_pop_Z[j], atom_data.ion_pop_stage[j],
                            atom_data.ion_number_density[(size_t)j * geo.n_shells + s]);
            fclose(ip);
            printf("Ion populations written to lumina_ion_pops.csv\n");
        }
    }

    /* Phase 6 - Step 8: Cleanup */
    cuda_free(&dev);              /* Phase 6 - Step 8 */
    free(h_escaped_nu);           /* Phase 6 - Step 8 */
    free(h_escaped_energy);       /* Phase 6 - Step 8 */
    free(h_escaped_flag);         /* Phase 6 - Step 8 */
    free(h_escaped_r);            /* Rotation mode */
    free(h_escaped_mu);           /* Rotation mode */
    free_geometry(&geo);          /* Phase 6 - Step 8 */
    free_opacity_state(&opacity); /* Phase 6 - Step 8 */
    free_plasma_state(&plasma);   /* Phase 6 - Step 8 */
    free_estimators(est);         /* Phase 6 - Step 8 */
    free_spectrum(spec);          /* Phase 6 - Step 8 */
    free(volume);                 /* Phase 6 - Step 8 */
    free_atomic_data(&atom_data); /* Task #072 */
    if (enable_nlte) {
        cuda_nlte_solver_free(&nlte_solver);
        nlte_free(&nlte);
    }
    if (gamma_dep_enabled) gamma_deposition_free(&gamma_dep);
    if (bf_opacity_enabled) bf_opacity_free(&bf);

    printf("\nDone.\n"); /* Phase 6 - Step 8 */
    return 0; /* Phase 6 - Step 8 */
}
