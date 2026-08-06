/* selftest_nlte_assemble.c — verify the GPU bound-bound NLTE assembly kernel
 * against the CPU nlte_assemble_rate_matrix on REAL atomic + NLTE data.
 *
 * For each ion pair, per shell:
 *   ref  = full CPU assembly (bb loop included)
 *   test = CPU assembly with bb skipped + GPU bb kernel added on top
 * and reports the max relative element difference. The bb kernel mirrors the
 * CPU arithmetic in FP64, so the only difference is atomicAdd summation order;
 * the match should be at round-off (<< 1e-5).
 *
 * Modeled on bench_nlte_rates.c but with opacity.n_lines = atom.n_lines so the
 * per-line bb loop actually runs. */

#include "src/lumina.h"
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double planck(double nu, double T) {
    double h = H_PLANCK, k = K_BOLTZMANN, c = 2.99792458e10;
    double x = h * nu / (k * T);
    if (x > 50.0) return 0.0;
    double prefac = 2.0 * h * nu * nu * nu / (c * c);
    return prefac / (exp(x) - 1.0);
}

int main(int argc, char **argv) {
    const char *ref_dir = (argc > 1) ? argv[1] : "data/tardis_reference";
    int n_shells = (argc > 2) ? atoi(argv[2]) : 4;

    AtomicData atom;
    PlasmaState plasma; memset(&plasma, 0, sizeof(plasma));
    OpacityState opacity; memset(&opacity, 0, sizeof(opacity));
    NLTEConfig nlte; memset(&nlte, 0, sizeof(nlte));

    if (load_atomic_data(&atom, ref_dir, n_shells) != 0) {
        fprintf(stderr, "load_atomic_data failed\n"); return 1;
    }
    printf("loaded %d lines, %d levels\n", atom.n_lines, atom.n_levels);

    plasma.n_shells   = n_shells;
    plasma.T_e        = (double *)malloc(n_shells * sizeof(double));
    plasma.n_electron = (double *)malloc(n_shells * sizeof(double));
    plasma.rho        = (double *)calloc(n_shells, sizeof(double));
    double *seed_temperature = (double *)malloc(n_shells * sizeof(double));
    double *seed_dilution = (double *)malloc(n_shells * sizeof(double));
    for (int s = 0; s < n_shells; s++) {
        double f = (double)s / (n_shells - 1);
        seed_temperature[s] = 12000.0 - 6000.0 * f;
        seed_dilution[s] = 0.5 * exp(-3.0 * f);
        plasma.T_e[s] = 0.9 * seed_temperature[s];
        plasma.n_electron[s] = 1e9 * exp(-2.0 * f);
        plasma.rho[s]   = 1e-12;
    }
    for (int ip = 0; ip < atom.n_ion_pops; ip++)
        for (int s = 0; s < n_shells; s++) {
            atom.ion_number_density[ip * n_shells + s] =
                (atom.ion_pop_stage[ip] >= 1) ? 1e9 : 0.0;
            atom.partition_functions[ip * n_shells + s] = 1.0;
        }

    /* Enable the per-line bb loop: the assembler reads opacity->n_lines. */
    opacity.n_lines  = atom.n_lines;
    opacity.n_shells = n_shells;

    if (nlte_init(&nlte, &atom, &opacity, n_shells) != 0) {
        fprintf(stderr, "nlte_init failed\n"); return 1;
    }

    for (int s = 0; s < n_shells; s++)
        for (int bb = 0; bb < nlte.n_freq_bins; bb++) {
            double log_lo = log(nlte.nu_min) + bb * nlte.d_log_nu;
            double nu_mid = exp(log_lo + 0.5 * nlte.d_log_nu);
            nlte.J_nu[s * nlte.n_freq_bins + bb] =
                seed_dilution[s] * planck(nu_mid, seed_temperature[s]);
        }

    /* Realistic within-SL Boltzmann fractions (the driver does this too). */
    if (nlte_precompute_within_sl_frac(&nlte, &atom, &plasma, n_shells) != 0) {
        fprintf(stderr, "within-SL projection failed\n");
        return 1;
    }

    if (nlte_assemble_gpu_init(&nlte, &atom, &opacity, n_shells) != 0) {
        fprintf(stderr, "nlte_assemble_gpu_init failed\n"); return 1;
    }
    nlte_assemble_gpu_refresh(&nlte, &plasma);

    int pairs[][2] = { {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11},
                       {12,13}, {14,15}, {16,17}, {18,19},
                       {20,21}, {22,23}, {24,25}, {26,27},
                       {28,29}, {29,30} };
    int n_pairs = NLTE_PAIR_COUNT;

    int *active = (int *)malloc(n_shells * sizeof(int));
    for (int s = 0; s < n_shells; s++) active[s] = 1;

    double global_max_rel = 0.0, global_max_abs = 0.0;
    double t_cpu_bb = 0.0, t_gpu_bb = 0.0;

    for (int p = 0; p < n_pairs; p++) {
        int lo = pairs[p][0], hi = pairs[p][1];
        int super_start = nlte.nlte_ion_super_offset[lo];
        int N    = nlte.nlte_ion_super_offset[hi + 1] - super_start;
        int n_lo_super = nlte.nlte_ion_super_offset[lo + 1] - super_start;
        if (N <= 0) continue;

        size_t mat_elems = (size_t)n_shells * N * N;
        double *A_ref  = (double *)calloc(mat_elems, sizeof(double));
        double *A_test = (double *)calloc(mat_elems, sizeof(double));
        double *b_ref  = (double *)calloc((size_t)n_shells * N, sizeof(double));
        double *b_test = (double *)calloc((size_t)n_shells * N, sizeof(double));

        /* Reference: full CPU assembly (bb included). */
        double tc0 = now_s();
        nlte_assemble_set_skip_bb(0);
        for (int s = 0; s < n_shells; s++)
            nlte_assemble_rate_matrix(&nlte, &atom, &plasma, &opacity,
                                      lo, hi, s, 1.0,
                                      A_ref + (size_t)s * N * N,
                                      b_ref + (size_t)s * N, N, NULL, NULL, p);
        t_cpu_bb += now_s() - tc0;

        /* Test: CPU remainder (bb skipped) + GPU bb kernel. */
        nlte_assemble_set_skip_bb(1);
        for (int s = 0; s < n_shells; s++)
            nlte_assemble_rate_matrix(&nlte, &atom, &plasma, &opacity,
                                      lo, hi, s, 1.0,
                                      A_test + (size_t)s * N * N,
                                      b_test + (size_t)s * N, N, NULL, NULL, p);
        nlte_assemble_set_skip_bb(0);
        double tg0 = now_s();
        nlte_assemble_bb_gpu_pair(A_test, N, n_shells, lo, hi,
                                  super_start, n_lo_super, active);
        t_gpu_bb += now_s() - tg0;

        double max_rel = 0.0, max_abs = 0.0;
        for (size_t k = 0; k < mat_elems; k++) {
            double r = A_ref[k], g = A_test[k];
            double d = fabs(g - r);
            if (d > max_abs) max_abs = d;
            if (fabs(r) > 1e-30) {
                double rel = d / fabs(r);
                if (rel > max_rel) max_rel = rel;
            }
        }
        if (max_rel > global_max_rel) global_max_rel = max_rel;
        if (max_abs > global_max_abs) global_max_abs = max_abs;
        printf("  pair %2d (lo=%d hi=%d N=%d): max rel %.3e  max abs %.3e\n",
               p, lo, hi, N, max_rel, max_abs);

        free(A_ref); free(A_test); free(b_ref); free(b_test);
    }

    printf("\n=== GPU bb-assembly self-check ===\n");
    printf("  GLOBAL max rel diff: %.4e\n", global_max_rel);
    printf("  GLOBAL max abs diff: %.4e\n", global_max_abs);
    printf("  CPU bb assemble time: %.2f ms\n", t_cpu_bb * 1000.0);
    printf("  GPU bb assemble time: %.2f ms (incl up/down copies)\n", t_gpu_bb * 1000.0);
    printf("  verdict: %s (tol 1e-5)\n",
           global_max_rel < 1e-5 ? "PASS" : "FAIL");

    free(active);
    nlte_assemble_gpu_free();
    nlte_free(&nlte);
    free(seed_temperature); free(seed_dilution); free(plasma.T_e);
    free(plasma.n_electron); free(plasma.rho);
    free_atomic_data(&atom);
    return (global_max_rel < 1e-5) ? 0 : 2;
}
