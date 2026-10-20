/* bench_bf_gemm.c — Validate Task #39 GEMM path against CPU reference.
 * Loads atomic data + CMFGEN sigma_bf, synthesizes plasma state, calls
 * compute_bf_opacity twice (CPU vs GEMM), reports max|delta|/max_chi and timing. */

#include "src/lumina.h"
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
    const char *ref_dir = (argc > 1) ? argv[1] : "data/tardis_reference_cmfgen";
    int n_shells = 30;
    int n_iter   = (argc > 2) ? atoi(argv[2]) : 3;

    AtomicData atom;
    PlasmaState plasma; memset(&plasma, 0, sizeof(plasma));
    BFOpacity bf_cpu, bf_gpu;

    if (load_atomic_data(&atom, ref_dir, n_shells) != 0) {
        fprintf(stderr, "load_atomic_data failed\n"); return 1;
    }
    char path[512];
    snprintf(path, sizeof(path), "data/atomic/cmfgen_sigma_bf.bin");
    load_cmfgen_sigma_bf(&atom, path);
    if (!atom.cmfgen_loaded) {
        fprintf(stderr, "CMFGEN sigma_bf not loaded — bench requires it\n");
        return 1;
    }

    /* Synthesize plasma state: T_rad ramps 12000 -> 6000 K, W ramps 0.5 -> 0.05 */
    plasma.T_rad = (double *)malloc(n_shells * sizeof(double));
    plasma.W     = (double *)malloc(n_shells * sizeof(double));
    plasma.T_e   = (double *)malloc(n_shells * sizeof(double));
    plasma.n_electron = (double *)malloc(n_shells * sizeof(double));
    for (int s = 0; s < n_shells; s++) {
        double f = (double)s / (n_shells - 1);
        plasma.T_rad[s] = 12000.0 - 6000.0 * f;
        plasma.W[s]     = 0.5 * exp(-3.0 * f);
        plasma.T_e[s]   = plasma.T_rad[s] * 0.9;
        plasma.n_electron[s] = 1e9 * exp(-2.0 * f);
    }

    /* Synthesize ion populations: 1e10 cm^-3 in stage>=1, partition=1.0 */
    for (int ip = 0; ip < atom.n_ion_pops; ip++) {
        for (int s = 0; s < n_shells; s++) {
            atom.ion_number_density[ip * n_shells + s] =
                (atom.ion_pop_stage[ip] >= 1) ? 1e9 : 0.0;
            atom.partition_functions[ip * n_shells + s] = 1.0;
        }
    }

    bf_opacity_init(&bf_cpu, n_shells);
    bf_opacity_init(&bf_gpu, n_shells);

    /* CPU reference */
    unsetenv("LUMINA_BF_GEMM");
    double t0 = now_s();
    for (int it = 0; it < n_iter; it++)
        compute_bf_opacity(&bf_cpu, &atom, &plasma, n_shells);
    double t_cpu = (now_s() - t0) / n_iter;

    /* GEMM path */
    setenv("LUMINA_BF_GEMM", "1", 1);
    double t1 = now_s();
    for (int it = 0; it < n_iter; it++)
        compute_bf_opacity(&bf_gpu, &atom, &plasma, n_shells);
    double t_gpu = (now_s() - t1) / n_iter;

    /* Compare bf-only contribution (skip free-free which is identical).
     * Free-free adds the same per-shell quantity to both, so subtracting
     * shell-0-bin-0 free-free won't cleanly isolate; instead compare full
     * chi_bf and rely on the fact that ff is computed with identical inputs.
     */
    size_t N = (size_t)n_shells * bf_cpu.n_freq_bins;
    double max_cpu = 0, max_gpu = 0, max_abs_diff = 0, sum_sq = 0;
    int n_nonzero = 0;
    for (size_t i = 0; i < N; i++) {
        double a = bf_cpu.chi_bf[i], b = bf_gpu.chi_bf[i];
        if (a > max_cpu) max_cpu = a;
        if (b > max_gpu) max_gpu = b;
        double d = a - b;
        if (fabs(d) > max_abs_diff) max_abs_diff = fabs(d);
        if (a > 1e-30 || b > 1e-30) { sum_sq += d * d; n_nonzero++; }
    }
    double rms = (n_nonzero > 0) ? sqrt(sum_sq / n_nonzero) : 0.0;

    printf("\n=== Task #39 BF GEMM validation ===\n");
    printf("  shells x freq = %d x %d  (%zu bins)\n",
           n_shells, bf_cpu.n_freq_bins, N);
    printf("  CPU max chi_bf : %.4e\n", max_cpu);
    printf("  GPU max chi_bf : %.4e\n", max_gpu);
    printf("  max |CPU - GPU|: %.4e   (%.3f%% of CPU max)\n",
           max_abs_diff, 100.0 * max_abs_diff / (max_cpu > 0 ? max_cpu : 1));
    printf("  RMS over nonzero bins: %.4e\n", rms);
    printf("  CPU time/iter: %8.2f ms\n", t_cpu * 1000);
    printf("  GPU time/iter: %8.2f ms   speedup: %.1fx\n",
           t_gpu * 1000, t_cpu / t_gpu);

    bf_opacity_free(&bf_cpu);
    bf_opacity_free(&bf_gpu);
    bf_gemm_free();
    free_atomic_data(&atom);
    free(plasma.T_rad); free(plasma.W); free(plasma.T_e); free(plasma.n_electron);
    return (max_abs_diff / (max_cpu > 0 ? max_cpu : 1) < 0.01) ? 0 : 2;
}
