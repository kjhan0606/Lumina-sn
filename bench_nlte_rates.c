/* bench_nlte_rates.c — Validate Task #40 (A)+(B) GPU photoionization rates.
 *
 * Calls nlte_assemble_rate_matrix twice for every pair × shell:
 *   path A — lookup=NULL (inline CPU per-bin sum)
 *   path B — lookup=GPU pre-baked R_bf table (cuBLAS TF32 GEMM)
 * Compares the resulting matrix entries and times both paths. */

#include "src/lumina.h"
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Rough Planck B_nu(T) [erg/s/cm²/Hz/sr] */
static double planck(double nu, double T) {
    double h = H_PLANCK, k = K_BOLTZMANN, c = 2.99792458e10;
    double x = h * nu / (k * T);
    if (x > 50.0) return 0.0;
    double prefac = 2.0 * h * nu * nu * nu / (c * c);
    return prefac / (exp(x) - 1.0);
}

int main(int argc, char **argv) {
    const char *ref_dir = (argc > 1) ? argv[1] : "data/tardis_reference";
    int n_shells = 30;
    int n_iter   = (argc > 2) ? atoi(argv[2]) : 3;

    AtomicData atom;
    PlasmaState plasma; memset(&plasma, 0, sizeof(plasma));
    OpacityState opacity; memset(&opacity, 0, sizeof(opacity));
    NLTEConfig nlte; memset(&nlte, 0, sizeof(nlte));

    if (load_atomic_data(&atom, ref_dir, n_shells) != 0) {
        fprintf(stderr, "load_atomic_data failed\n"); return 1;
    }

    /* Synthesize plasma: T_rad ramps 12000→6000 K, W 0.5→0.05, T_e=0.9 T_rad */
    plasma.n_shells = n_shells;
    plasma.T_rad      = (double *)malloc(n_shells * sizeof(double));
    plasma.W          = (double *)malloc(n_shells * sizeof(double));
    plasma.T_e        = (double *)malloc(n_shells * sizeof(double));
    plasma.n_electron = (double *)malloc(n_shells * sizeof(double));
    plasma.rho        = (double *)calloc(n_shells, sizeof(double));
    for (int s = 0; s < n_shells; s++) {
        double f = (double)s / (n_shells - 1);
        plasma.T_rad[s] = 12000.0 - 6000.0 * f;
        plasma.W[s]     = 0.5 * exp(-3.0 * f);
        plasma.T_e[s]   = 0.9 * plasma.T_rad[s];
        plasma.n_electron[s] = 1e9 * exp(-2.0 * f);
        plasma.rho[s]   = 1e-12;
    }

    /* Synthesize ion populations: 1e9 cm^-3 in stage>=1 */
    for (int ip = 0; ip < atom.n_ion_pops; ip++) {
        for (int s = 0; s < n_shells; s++) {
            atom.ion_number_density[ip * n_shells + s] =
                (atom.ion_pop_stage[ip] >= 1) ? 1e9 : 0.0;
            atom.partition_functions[ip * n_shells + s] = 1.0;
        }
    }

    /* Tiny opacity state with no lines (so radiative b-b is a no-op) */
    opacity.n_lines = 0;
    opacity.n_shells = n_shells;

    /* NLTE init from real atomic data */
    if (nlte_init(&nlte, &atom, &opacity, n_shells) != 0) {
        fprintf(stderr, "nlte_init failed\n"); return 1;
    }

    /* Synthesize J_nu = W * B_nu(T_rad) on the NLTE grid */
    for (int s = 0; s < n_shells; s++) {
        for (int bb = 0; bb < nlte.n_freq_bins; bb++) {
            double log_lo = log(nlte.nu_min) + bb * nlte.d_log_nu;
            double nu_mid = exp(log_lo + 0.5 * nlte.d_log_nu);
            nlte.J_nu[s * nlte.n_freq_bins + bb] =
                plasma.W[s] * planck(nu_mid, plasma.T_rad[s]);
        }
    }

    /* GPU init + compute */
    if (nlte_rates_gpu_init(&nlte, &atom, n_shells) != 0) {
        fprintf(stderr, "nlte_rates_gpu_init failed\n"); return 1;
    }
    NLTERateLookup lookup;
    if (nlte_rates_gpu_compute(&nlte, &lookup) != 0) {
        fprintf(stderr, "nlte_rates_gpu_compute failed\n"); return 1;
    }

    /* For each NLTE pair, allocate two NxN matrix buffers, assemble both ways,
     * compute max |diff| / max |entry|. */
    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11},
                       {12,13}, {14,15}, {16,17}, {18,19},
                       {20,21}, {22,23}, {24,25}, {26,27} };
    int n_pairs_avail = nlte.n_nlte_ions / 2;

    double t_cpu_total = 0, t_gpu_total = 0;
    double max_abs_diff = 0, max_abs_entry = 0;
    double sum_sq = 0; int n_entries = 0;
    long long n_dropped = 0;

    for (int rep = 0; rep < n_iter; rep++) {
        /* Re-time GPU compute every rep (J_nu is fixed but we want timing). */
        double tg0 = now_s();
        nlte_rates_gpu_compute(&nlte, &lookup);
        double tg1 = now_s();
        t_gpu_total += (tg1 - tg0);

        for (int p = 0; p < n_pairs_avail; p++) {
            int lo = pairs[p][0], hi = pairs[p][1];
            int lev_start = nlte.nlte_ion_level_offset[lo];
            int N = nlte.nlte_ion_level_offset[hi + 1] - lev_start;
            if (N <= 0) continue;

            double *A_cpu = (double *)calloc((size_t)N * N, sizeof(double));
            double *A_gpu = (double *)calloc((size_t)N * N, sizeof(double));
            double *b     = (double *)calloc((size_t)N, sizeof(double));

            for (int s = 0; s < n_shells; s++) {
                memset(A_cpu, 0, (size_t)N * N * sizeof(double));
                memset(A_gpu, 0, (size_t)N * N * sizeof(double));

                double tc0 = now_s();
                nlte_assemble_rate_matrix(&nlte, &atom, &plasma, &opacity,
                                          lo, hi, s, 1.0,
                                          A_cpu, b, N, NULL,
                                          NULL, -1);
                t_cpu_total += (now_s() - tc0);

                memset(b, 0, (size_t)N * sizeof(double));
                double tgp0 = now_s();
                nlte_assemble_rate_matrix(&nlte, &atom, &plasma, &opacity,
                                          lo, hi, s, 1.0,
                                          A_gpu, b, N, NULL,
                                          &lookup, p);
                t_gpu_total += (now_s() - tgp0);

                /* Diff */
                for (size_t i = 0; i < (size_t)N * N; i++) {
                    double a = A_cpu[i], g = A_gpu[i];
                    double d = a - g;
                    double m = fabs(a) > fabs(g) ? fabs(a) : fabs(g);
                    if (m > max_abs_entry) max_abs_entry = m;
                    if (fabs(d) > max_abs_diff) max_abs_diff = fabs(d);
                    if (m > 1e-30) {
                        sum_sq += d * d;
                        n_entries++;
                    }
                    if (m == 0.0) n_dropped++;
                }
            }
            free(A_cpu); free(A_gpu); free(b);
        }
    }
    double rms = (n_entries > 0) ? sqrt(sum_sq / n_entries) : 0.0;

    int n_pairs_processed = (n_pairs_avail < 14) ? n_pairs_avail : 14;
    printf("\n=== Task #40 NLTE photoionization GEMM validation ===\n");
    printf("  pairs processed: %d, n_shells: %d, n_freq: %d\n",
           n_pairs_processed, n_shells, nlte.n_freq_bins);
    printf("  L_phot_total   : %d (GPU table rows)\n", lookup.L_phot_total);
    printf("  max |entry|    : %.4e\n", max_abs_entry);
    printf("  max |CPU - GPU|: %.4e   (%.4f%% of max entry)\n",
           max_abs_diff,
           100.0 * max_abs_diff / (max_abs_entry > 0 ? max_abs_entry : 1));
    printf("  RMS over nonzero entries: %.4e\n", rms);
    printf("  CPU total assemble time: %8.2f ms (%d iter)\n",
           t_cpu_total * 1000.0, n_iter);
    printf("  GPU total assemble time: %8.2f ms (incl GEMM call %d×)\n",
           t_gpu_total * 1000.0, n_iter);
    if (t_gpu_total > 0)
        printf("  speedup        : %.2fx\n", t_cpu_total / t_gpu_total);

    nlte_rates_gpu_free();
    nlte_free(&nlte);
    free(plasma.T_rad); free(plasma.W); free(plasma.T_e);
    free(plasma.n_electron); free(plasma.rho);
    free_atomic_data(&atom);
    return (max_abs_entry > 0 && max_abs_diff / max_abs_entry < 0.01) ? 0 : 2;
}
