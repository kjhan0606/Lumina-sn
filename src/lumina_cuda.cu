/* lumina_cuda.cu — Phase 6: CUDA GPU Transport Kernel
 * Direct port of lumina_transport.c (CPU) to CUDA device code.
 * Every line annotated with Phase 6 - Step N for traceability.
 * Source: lumina_transport.c, lumina_main.c (CPU reference) */

#include <stdio.h>      /* Phase 6 - Step 1 */
#include <stdlib.h>     /* Phase 6 - Step 1 */
#include <string.h>     /* Phase 6 - Step 1 */
#include <math.h>       /* Phase 6 - Step 1 */
#include <stdint.h>     /* Phase 6 - Step 1 */
#include <cuda_runtime.h> /* Phase 6 - Step 1 */
#include <cublas_v2.h>    /* cuBLAS batched NLTE solver */

/* Phase 6 - Step 1: Include shared header for struct definitions */
extern "C" {             /* Phase 6 - Step 1 */
#include "lumina.h"      /* Phase 6 - Step 1 */
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

    /* Phase 6 - Step 1: Geometry */
    CUDA_CHECK(cudaMalloc(&dev->d_r_inner, ns * sizeof(double)));                   /* Phase 6 - Step 1 */
    CUDA_CHECK(cudaMalloc(&dev->d_r_outer, ns * sizeof(double)));                   /* Phase 6 - Step 1 */

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

/* Batched LU solve: factorize + triangular solve on GPU for all shells at once.
 * h_matrices[batch * N * N] and h_rhs[batch * N] must be pre-filled (column-major). */
static int cuda_nlte_batched_solve(CudaNLTESolver *sol, int N, int batch) {
    size_t mat_bytes = (size_t)batch * N * N * sizeof(double);
    size_t rhs_bytes = (size_t)batch * N * sizeof(double);

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
        return -1;
    }

    /* Batched triangular solve */
    int info_host = 0;
    stat = cublasDgetrsBatched(sol->handle, CUBLAS_OP_N, N, 1,
        (const double **)sol->d_Aarray, N, sol->d_pivot,
        sol->d_Barray, N, &info_host, batch);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NLTE-GPU] cublasDgetrsBatched failed: %d\n", stat);
        return -1;
    }

    /* Download solutions and info */
    CUDA_CHECK(cudaMemcpy(sol->h_rhs, sol->d_rhs, rhs_bytes,
               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sol->h_info, sol->d_info, batch * sizeof(int),
               cudaMemcpyDeviceToHost));
    return 0;
}

/* GPU NLTE master solver: assemble on CPU (OpenMP), solve on GPU (cuBLAS batched).
 * Step 1.5: Iterative CE convergence wrapper — same logic as CPU nlte_solve_all. */
static void nlte_solve_all_gpu(NLTEConfig *nlte, AtomicData *atom,
                                PlasmaState *plasma, OpacityState *opacity,
                                double time_explosion, int n_shells,
                                CudaNLTESolver *sol) {
    printf("  [NLTE-GPU] Solving rate equations (cuBLAS batched, with CE)...\n");

    int n_pairs = nlte->n_nlte_ions / 2;
    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11} };
    const char *names[] = { "Si", "Ca", "Fe", "S", "Co", "Ni" };

    int ce_max_iter = 5;
    double ce_threshold = 1e-2;  /* 1% relative convergence on ion totals */
    double ce_damping = 0.5;     /* 50% damping */

    /* Save old ion totals for convergence check (n_nlte_ions * n_shells) */
    int n_ion_totals = nlte->n_nlte_ions * n_shells;
    double *old_ion_totals = (double *)calloc(n_ion_totals, sizeof(double));
    size_t pop_size = (size_t)nlte->n_nlte_levels_total * n_shells;
    double *old_pops = (double *)malloc(pop_size * sizeof(double));

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

        /* Solve all 6 ion pairs */
        for (int p = 0; p < n_pairs; p++) {
            int lo = pairs[p][0], hi = pairs[p][1];
            int lev_start = nlte->nlte_ion_level_offset[lo];
            int N = nlte->nlte_ion_level_offset[hi + 1] - lev_start;
            if (N <= 0) continue;

            /* Zero staging buffers */
            size_t mat_bytes = (size_t)n_shells * N * N * sizeof(double);
            size_t rhs_bytes = (size_t)n_shells * N * sizeof(double);
            memset(sol->h_matrices, 0, mat_bytes);
            memset(sol->h_rhs, 0, rhs_bytes);

            /* Assemble rate matrices for all shells (CPU, OpenMP parallel) */
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int s = 0; s < n_shells; s++) {
                double *A_cm = sol->h_matrices + (size_t)s * N * N;
                double *b    = sol->h_rhs      + (size_t)s * N;
                nlte_assemble_rate_matrix(nlte, atom, plasma, opacity,
                                          lo, hi, s, time_explosion,
                                          A_cm, b, N);
            }

            /* GPU batched solve */
            int ret = cuda_nlte_batched_solve(sol, N, n_shells);

            /* Extract populations, handle singular matrices */
            for (int s = 0; s < n_shells; s++) {
                if (ret != 0 || sol->h_info[s] != 0) {
                    /* Singular: fall back to Boltzmann at T_rad */
                    double T_rad = plasma->T_rad[s];
                    int Z_nl = nlte->nlte_Z[lo];
                    double n_total = 0.0;
                    for (int i = lo; i <= hi; i++) {
                        int ip = -1;
                        for (int j = 0; j < atom->n_ion_pops; j++) {
                            if (atom->ion_pop_Z[j] == Z_nl &&
                                atom->ion_pop_stage[j] == nlte->nlte_ion[i]) {
                                ip = j; break;
                            }
                        }
                        if (ip >= 0)
                            n_total += atom->ion_number_density[ip * n_shells + s];
                    }
                    double sum = 0.0;
                    for (int i = 0; i < N; i++) {
                        int global = nlte->nlte_to_global_level[lev_start + i];
                        double E = atom->level_energy_eV[global] * EV_TO_ERG;
                        int g = atom->level_g[global];
                        double pop = (double)g * exp(-E / (K_BOLTZMANN * T_rad));
                        nlte->nlte_level_populations[(lev_start + i) * n_shells + s] = pop;
                        sum += pop;
                    }
                    if (sum > 0.0) {
                        double scale = n_total / sum;
                        for (int i = 0; i < N; i++)
                            nlte->nlte_level_populations[(lev_start + i) * n_shells + s] *= scale;
                    }
                } else {
                    double *x = sol->h_rhs + (size_t)s * N;
                    for (int i = 0; i < N; i++) {
                        double pop = x[i];
                        if (pop < 0.0) pop = 1e-30;
                        nlte->nlte_level_populations[(lev_start + i) * n_shells + s] = pop;
                    }
                }
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

    /* Print ion pair summaries */
    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0], hi = pairs[p][1];
        int N = nlte->nlte_ion_level_offset[hi + 1] - nlte->nlte_ion_level_offset[lo];
        printf("    %s (%d levels): done [GPU]\n", names[p], N);
    }

    /* Update tau_sobolev for NLTE lines (same as CPU path) */
    printf("  [NLTE-GPU] Updating tau_sobolev from NLTE populations...\n");
    {
        int n_lines = opacity->n_lines;
        for (int line = 0; line < n_lines; line++) {
            int ion_idx = nlte->nlte_line_map[line];
            if (ion_idx < 0) continue;

            int Z     = atom->line_atomic_number[line];
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

            for (int s = 0; s < n_shells; s++) {
                double n_lower = nlte->nlte_level_populations[nlte_lo * n_shells + s];
                double n_upper = nlte->nlte_level_populations[nlte_up * n_shells + s];
                double stim_corr = 1.0;
                if (n_lower > 0.0 && n_upper > 0.0 && g_lo > 0 && g_up > 0) {
                    stim_corr = 1.0 - ((double)g_lo * n_upper) / ((double)g_up * n_lower);
                    if (stim_corr < 0.0) stim_corr = 0.0;
                }
                double tau = SOBOLEV_COEFF * f_lu * lam_cm * time_explosion *
                             n_lower * stim_corr;
                if (tau < 1e-100) tau = 1e-100;
                opacity->tau_sobolev[line * n_shells + s] = tau;
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

    while (current_type >= 0 && ma_iter < 500) { /* Phase 6 - Step 6 */
        ma_iter++; /* Phase 6 - Step 6 */

        /* Phase 6 - Step 6: Bounds check */
        if (activation_level_id < 0 || activation_level_id >= n_macro_levels) { /* Phase 6 - Step 6 */
            current_type = -1; /* Phase 6 - Step 6: MA_BB_EMISSION */
            *out_transition_type = current_type; /* Phase 6 - Step 6 */
            break; /* Phase 6 - Step 6 */
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

    /* Phase 6 - Step 6: Convert transition_id to line_id for emission */
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
            d_line_emission(nu, next_line_id, transition_id, /* Phase 6 - Step 6 */
                             *r, *mu, t_exp, d_line_list_nu); /* Phase 6 - Step 6 */
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
            /* Electron scattering tau for turning-point segment */
            tau_total += SIGMA_THOMSON * d_electron_density[s] * fabs(0.0 - z_cur);
            z_cur = 0.0;
            break;
        } else {
            /* Crosses inner boundary into shell s-1 */
            double z_bnd = -sqrt(r_inner_s * r_inner_s - p2);
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
            /* Electron scattering tau for inward shell crossing */
            tau_total += SIGMA_THOMSON * d_electron_density[s] * fabs(z_bnd - z_cur);
            z_cur = z_bnd;
            s--;
        }
    }
    if (s < 0) return;  /* reabsorbed by photosphere */

    /* Phase 2: Outward propagation */
    while (s < n_shells) {
        double r_outer_s = d_r_outer[s];
        double z_bnd = sqrt(r_outer_s * r_outer_s - p2);
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
        /* Electron scattering tau for outward shell crossing */
        tau_total += SIGMA_THOMSON * d_electron_density[s] * fabs(z_bnd - z_cur);
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
    /* Phase 6 - Step 7: Scalars */
    int n_packets, int n_shells, int n_lines, int n_macro_levels,
    double t_exp, double T_inner, double packet_energy,
    int line_interaction_type, uint64_t base_seed)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x; /* Phase 6 - Step 7 */
    if (p >= n_packets) return;                     /* Phase 6 - Step 7 */

    /* Phase 6 - Step 7: Load RNG state into registers */
    uint64_t rng[4]; /* Phase 6 - Step 7 */
    d_rng_init(rng, base_seed + (uint64_t)p);      /* Phase 6 - Step 7 */

    /* Phase 6 - Step 7: Initialize packet at inner boundary */
    double pkt_r = d_r_inner[0];                    /* Phase 6 - Step 7 */
    double pkt_mu = sqrt(d_rng_uniform(rng));       /* Phase 6 - Step 7 */
    int pkt_shell_id = 0;                           /* Phase 6 - Step 7 */
    int pkt_status = 0; /* Phase 6 - Step 7: PACKET_IN_PROCESS */

    /* Phase 6 - Step 7: Sample frequency from Bjorkman-Wood Planck */
    double kT_h = K_BOLTZMANN * T_inner / H_PLANCK; /* Phase 6 - Step 7 */
    double xi0 = d_rng_uniform(rng);                 /* Phase 6 - Step 7 */
    double l_coef = M_PI_VAL * M_PI_VAL * M_PI_VAL * M_PI_VAL / 90.0; /* Phase 6 - Step 7 */
    double target = xi0 * l_coef;                    /* Phase 6 - Step 7 */
    double cumsum = 0.0;                             /* Phase 6 - Step 7 */
    double l_min = 1.0;                              /* Phase 6 - Step 7 */
    for (int l = 1; l <= 1000; l++) { /* Phase 6 - Step 7 */
        double ld = (double)l;                       /* Phase 6 - Step 7 */
        double l_inv4 = 1.0 / (ld * ld * ld * ld);  /* Phase 6 - Step 7 */
        cumsum += l_inv4;                            /* Phase 6 - Step 7 */
        if (cumsum >= target) { /* Phase 6 - Step 7 */
            l_min = ld;         /* Phase 6 - Step 7 */
            break;              /* Phase 6 - Step 7 */
        }
    }
    double r1 = d_rng_uniform(rng); /* Phase 6 - Step 7 */
    double r2 = d_rng_uniform(rng); /* Phase 6 - Step 7 */
    double r3 = d_rng_uniform(rng); /* Phase 6 - Step 7 */
    double r4 = d_rng_uniform(rng); /* Phase 6 - Step 7 */
    if (r1 < 1e-300) r1 = 1e-300;  /* Phase 6 - Step 7 */
    if (r2 < 1e-300) r2 = 1e-300;  /* Phase 6 - Step 7 */
    if (r3 < 1e-300) r3 = 1e-300;  /* Phase 6 - Step 7 */
    if (r4 < 1e-300) r4 = 1e-300;  /* Phase 6 - Step 7 */
    double x = -log(r1 * r2 * r3 * r4) / l_min; /* Phase 6 - Step 7 */
    double pkt_nu = x * kT_h;                    /* Phase 6 - Step 7 */
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
                                    n_lines, n_shells, N_VPACKETS,
                                    d_virtual_spectrum, rng);
        }
    }

    /* Phase 6 - Step 7: Main transport loop */
    int loop_count = 0; /* Phase 6 - Step 7 */
    while (pkt_status == 0 && loop_count < 100000) { /* Phase 6 - Step 7 */
        loop_count++; /* Phase 6 - Step 7 */

        /* Phase 6 - Step 7: Electron scattering opacity */
        int shell = pkt_shell_id;                        /* Phase 6 - Step 7 */
        double chi_e = d_electron_density[shell] * SIGMA_THOMSON; /* Phase 6 - Step 7 */
        double chi_continuum = chi_e;                    /* Phase 6 - Step 7 */

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
                pkt_status = 2; /* Phase 6 - Step 7: PACKET_REABSORBED */
            } else { /* Phase 6 - Step 7 */
                pkt_shell_id = next_shell; /* Phase 6 - Step 7 */
            }
        } else if (interaction_type == 1) { /* Phase 6 - Step 7: LINE */
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
                                            n_lines, n_shells, N_VPACKETS,
                                            d_virtual_spectrum, rng);
                }
            }
        } else if (interaction_type == 2) { /* Phase 6 - Step 7: ESCATTERING */
            d_thomson_scatter(&pkt_r, &pkt_mu, &pkt_nu, &pkt_energy, /* Phase 6 - Step 7 */
                               t_exp, rng);                            /* Phase 6 - Step 7 */
            /* Virtual packet from e-scatter */
            if (d_virtual_spectrum != NULL) {
                double nu_cmf_v = pkt_nu * d_get_doppler_factor(pkt_r, pkt_mu, t_exp);
                for (int vp = 0; vp < N_VPACKETS; vp++) {
                    d_trace_virtual_packet(pkt_r, pkt_shell_id, nu_cmf_v, pkt_energy,
                                            t_exp, L_inner,
                                            d_r_inner, d_r_outer,
                                            d_line_list_nu, d_tau_sobolev,
                                            d_electron_density,
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
    } else { /* Phase 6 - Step 7: still in process (loop limit) */
        d_escaped_flag[p] = 0;        /* Phase 6 - Step 7 */
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
    d_rng_init(s, base_seed + (uint64_t)p);         /* Phase 6 - Step 7 */
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
    /* Task #072: Initialize n_electron from TARDIS reference */
    plasma.n_electron = (double *)malloc(geo.n_shells * sizeof(double));
    for (int i = 0; i < geo.n_shells; i++)
        plasma.n_electron[i] = opacity.electron_density[i];

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

    /* Dynamic transition probability update: default OFF, enable with LUMINA_DYNAMIC_TRANSPROB=1 */
    int enable_transprob_update = 0;
    if (getenv("LUMINA_DYNAMIC_TRANSPROB"))
        enable_transprob_update = 1;

    printf("\nSimulation parameters:\n");
    printf("  Packets: %d, Iterations: %d\n", n_packets, n_iterations);
    printf("  Line interaction: MACROATOM\n");
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
    printf("  T_inner: %.2f K\n", config.T_inner);
    printf("  Transition probs: %s\n", enable_transprob_update ? "DYNAMIC" : "FROZEN");

    /* Phase 6 - Step 8: Compute shell volumes */
    double *volume = (double *)malloc(geo.n_shells * sizeof(double)); /* Phase 6 - Step 8 */
    for (int i = 0; i < geo.n_shells; i++) { /* Phase 6 - Step 8 */
        volume[i] = (4.0 / 3.0) * M_PI_VAL * /* Phase 6 - Step 8 */
            (geo.r_outer[i] * geo.r_outer[i] * geo.r_outer[i] - /* Phase 6 - Step 8 */
             geo.r_inner[i] * geo.r_inner[i] * geo.r_inner[i]); /* Phase 6 - Step 8 */
    }

    /* Phase 6 - Step 8: Create CPU estimators and spectrum */
    Estimators *est = create_estimators(geo.n_shells, opacity.n_lines); /* Phase 6 - Step 8 */
    Spectrum *spec = create_spectrum(500.0, 20000.0, 2000);            /* Phase 6 - Step 8 */

    /* Phase 6 - Step 8: Allocate and upload GPU data */
    CudaDeviceData dev; /* Phase 6 - Step 8 */
    memset(&dev, 0, sizeof(dev)); /* Phase 6 - Step 8 */
    cuda_allocate(&dev, &geo, &opacity, n_packets); /* Phase 6 - Step 8 */
    cuda_upload(&dev, &geo, &opacity);               /* Phase 6 - Step 8 */
    printf("  GPU memory allocated and uploaded.\n"); /* Phase 6 - Step 8 */

    /* NLTE: Initialize if enabled */
    NLTEConfig nlte;
    memset(&nlte, 0, sizeof(nlte));
    CudaNLTESolver nlte_solver;
    memset(&nlte_solver, 0, sizeof(nlte_solver));
    if (enable_nlte) {
        printf("\n--- NLTE Initialization ---\n");
        nlte_init(&nlte, &atom_data, &opacity, geo.n_shells);
        cuda_allocate_nlte(&dev, &nlte, geo.n_shells);
        /* Find max level count across all ion pairs for cuBLAS allocation */
        int max_N = 0;
        int n_pairs_init = nlte.n_nlte_ions / 2;
        for (int p = 0; p < n_pairs_init; p++) {
            int lo = 2 * p, hi = 2 * p + 1;
            int N = nlte.nlte_ion_level_offset[hi + 1] -
                    nlte.nlte_ion_level_offset[lo];
            if (N > max_N) max_N = N;
        }
        cuda_nlte_solver_init(&nlte_solver, max_N, geo.n_shells);
    }

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

    /* Phase 6 - Step 8: Iteration loop */
    for (int iter = 0; iter < n_iterations; iter++) { /* Phase 6 - Step 8 */
        printf("\n--- Iteration %d/%d ---\n", iter + 1, n_iterations); /* Phase 6 - Step 8 */

        /* Phase 6 - Step 8: Reset estimators */
        reset_estimators(est);   /* Phase 6 - Step 8: CPU estimators */
        reset_spectrum(spec);    /* Phase 6 - Step 8 */
        cuda_reset_estimators(&dev, geo.n_shells); /* Phase 6 - Step 8: GPU estimators */
        CUDA_CHECK(cudaMemset(dev.d_escaped_flag, 0, n_packets * sizeof(int))); /* Phase 6 - Step 8 */

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
            dev.d_j_estimator, dev.d_nu_bar_estimator,
            dev.d_j_nu_estimator,
            dev.nlte_n_freq_bins, dev.nlte_nu_min, dev.nlte_d_log_nu,
            dev.d_rng_states,
            dev.d_escaped_nu, dev.d_escaped_energy,
            dev.d_escaped_flag,
            dev.d_escaped_r, dev.d_escaped_mu,
            dev.d_n_escaped, dev.d_n_reabsorbed,
            enable_virtual ? dev.d_virtual_spectrum : (double *)NULL, L_inner,
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

        /* Phase 6 - Step 8: Solve radiation field (CPU, reuse lumina_plasma.c) */
        solve_radiation_field(est, geo.time_explosion, time_simulation, /* Phase 6 - Step 8 */
                               volume, &opacity, &plasma,               /* Phase 6 - Step 8 */
                               config.damping_constant);                /* Task #072: W/T_rad damping */

        /* Task #072: Recompute tau_sobolev and re-upload to GPU */
        if (iter > 0) {
            compute_plasma_state(&atom_data, &plasma, &opacity, geo.time_explosion);

            /* NLTE: solve rate equations and update tau for NLTE lines */
            if (enable_nlte && iter >= nlte_start_iter) {
                cuda_download_j_nu(&dev, &nlte, geo.n_shells);
                nlte_normalize_j_nu(&nlte, time_simulation, volume, geo.n_shells);
                nlte_solve_all_gpu(&nlte, &atom_data, &plasma, &opacity,
                                    geo.time_explosion, geo.n_shells,
                                    &nlte_solver);
                /* tau_sobolev already updated inside nlte_solve_all_gpu */
            }

            /* Dynamic transition probability recomputation */
            if (enable_transprob_update && iter >= config.hold_iterations) {
                compute_transition_probabilities(&atom_data, &plasma, &opacity,
                    config.damping_constant,
                    (iter > config.hold_iterations) ? 1 : 0);
            }

            CUDA_CHECK(cudaMemcpy(dev.d_tau_sobolev, opacity.tau_sobolev,
                       (size_t)opacity.n_lines * geo.n_shells * sizeof(double),
                       cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev.d_electron_density, opacity.electron_density,
                       geo.n_shells * sizeof(double), cudaMemcpyHostToDevice));
            /* Re-upload updated transition probabilities to GPU */
            if (enable_transprob_update && iter >= config.hold_iterations) {
                CUDA_CHECK(cudaMemcpy(dev.d_transition_probabilities,
                           opacity.transition_probabilities,
                           (size_t)opacity.n_macro_transitions * geo.n_shells * sizeof(double),
                           cudaMemcpyHostToDevice));
            }
        }

        /* Phase 6 - Step 8: Print plasma state */
        printf("  Shell  W_LUMINA   T_rad_LUM  nubar/j\n"); /* Phase 6 - Step 8 */
        for (int i = 0; i < geo.n_shells; i += 5) { /* Phase 6 - Step 8 */
            double ratio = est->nu_bar_estimator[i] / est->j_estimator[i]; /* Phase 6 - Step 8 */
            printf("  %3d    %.6f   %.2f K   %.4e\n", /* Phase 6 - Step 8 */
                   i, plasma.W[i], plasma.T_rad[i], ratio); /* Phase 6 - Step 8 */
        }

        /* Phase 6 - Step 8: Update T_inner (after hold iterations) */
        if (iter >= config.hold_iterations) { /* Phase 6 - Step 8 */
            double old_T = config.T_inner; /* Phase 6 - Step 8 */
            update_t_inner(&config, L_emitted); /* Phase 6 - Step 8 */
            printf("  T_inner: %.2f K -> %.2f K (L_em=%.3e, L_req=%.3e)\n",
                   old_T, config.T_inner, L_emitted, config.luminosity_requested);
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

    /* Rotation spectrum: Doppler-weight escaped packets (post-processing) */
    if (enable_rotation) {
        double L_inner_final = 4.0 * M_PI_VAL * geo.r_inner[0] * geo.r_inner[0] *
                               SIGMA_SB * pow(config.T_inner, 4);
        Spectrum *spec_rot = create_spectrum(500.0, 20000.0, 2000);
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
        fprintf(out, "shell_id,W,T_rad\n"); /* Phase 6 - Step 8 */
        for (int i = 0; i < geo.n_shells; i++) { /* Phase 6 - Step 8 */
            fprintf(out, "%d,%.10f,%.6f\n", i, plasma.W[i], plasma.T_rad[i]); /* Phase 6 - Step 8 */
        }
        fclose(out); /* Phase 6 - Step 8 */
        printf("Plasma state written to lumina_plasma_state.csv\n"); /* Phase 6 - Step 8 */
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

    printf("\nDone.\n"); /* Phase 6 - Step 8 */
    return 0; /* Phase 6 - Step 8 */
}
