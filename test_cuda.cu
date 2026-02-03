/**
 * LUMINA-SN CUDA Test Driver
 * test_cuda.cu - Integrated GPU-accelerated Monte Carlo simulation
 *
 * Usage:
 *   ./test_cuda [atomic_data.h5] [n_packets] [output.csv]
 *
 * This driver:
 *   1. Loads atomic data from HDF5
 *   2. Sets up simulation parameters (matching SN 2011fe)
 *   3. Computes plasma state and Sobolev opacities on CPU
 *   4. Uploads everything to GPU
 *   5. Runs transport on GPU
 *   6. Downloads spectrum and saves to CSV
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Include C headers (compiled with C linkage) */
extern "C" {
#include "atomic_data.h"
#include "plasma_physics.h"
#include "simulation_state.h"
}

#include "cuda_atomic.h"

/* ============================================================================
 * CONFIGURATION (matching CPU precision fit parameters)
 * ============================================================================ */

typedef struct {
    char atomic_file[256];
    char output_file[256];

    int    n_shells;
    double t_exp;
    double v_inner;
    double v_outer;
    double T_inner;
    double T_outer;
    double rho_inner;
    double rho_profile;

    int    stratified;
    int64_t n_packets;
    int    n_bins;

    double nu_min;
    double nu_max;
} Config;

static void config_defaults(Config *cfg)
{
    strcpy(cfg->atomic_file, "atomic/kurucz_cd23_chianti_H_He.h5");
    strcpy(cfg->output_file, "spectrum_cuda.csv");

    cfg->n_shells = 30;
    cfg->t_exp = 86400.0 * 19.0;     /* 19 days */
    cfg->v_inner = 1.0e9;            /* 10,000 km/s (precision fit) */
    cfg->v_outer = 2.5e9;            /* 25,000 km/s */
    cfg->T_inner = 13500.0;          /* 13,500 K (precision fit) */
    cfg->T_outer = 5500.0;
    cfg->rho_inner = 8e-14;
    cfg->rho_profile = -7.0;

    cfg->stratified = 1;
    cfg->n_packets = 1000000;        /* 1M packets for GPU */
    cfg->n_bins = CUDA_SPECTRUM_N_BINS;

    /* Optical wavelength range: 3000-10000 Å */
    cfg->nu_min = 2.998e10 / (10000.0e-8);
    cfg->nu_max = 2.998e10 / (3000.0e-8);
}

/* ============================================================================
 * CPU SETUP (reuse existing C code)
 * ============================================================================ */

static int setup_simulation(const Config *cfg, const AtomicData *atomic,
                             SimulationState *state)
{
    printf("\n[CPU] Setting up simulation state...\n");

    /* Initialize state */
    int status = simulation_state_init(state, atomic, cfg->n_shells, cfg->t_exp);
    if (status != 0) {
        fprintf(stderr, "Failed to initialize simulation state\n");
        return -1;
    }

    /* Set up shell geometry (velocity-based) */
    double log_v_inner = log(cfg->v_inner);
    double log_v_outer = log(cfg->v_outer);
    double d_log_v = (log_v_outer - log_v_inner) / cfg->n_shells;

    double log_T_inner = log(cfg->T_inner);
    double log_T_outer = log(cfg->T_outer);
    double d_log_T = (log_T_outer - log_T_inner) / cfg->n_shells;

    for (int i = 0; i < cfg->n_shells; i++) {
        double v_in = exp(log_v_inner + i * d_log_v);
        double v_out = exp(log_v_inner + (i + 1) * d_log_v);
        double v_mid = 0.5 * (v_in + v_out);

        double r_in = v_in * cfg->t_exp;
        double r_out = v_out * cfg->t_exp;

        simulation_set_shell_geometry(state, i, r_in, r_out);

        double T = exp(log_T_inner + i * d_log_T);
        simulation_set_shell_temperature(state, i, T);

        double rho = cfg->rho_inner * pow(v_mid / cfg->v_inner, cfg->rho_profile);
        simulation_set_shell_density(state, i, rho);
    }

    /* Set abundances */
    if (cfg->stratified) {
        simulation_set_stratified_abundances(state);
    } else {
        Abundances ab;
        abundances_set_type_ia_w7(&ab);
        simulation_set_abundances(state, &ab);
    }

    /* Compute plasma state */
    printf("[CPU] Computing plasma state...\n");
    simulation_compute_plasma(state);

    /* Compute Sobolev opacities */
    printf("[CPU] Computing Sobolev opacities...\n");
    simulation_compute_opacities(state);

    printf("[CPU] Setup complete: %ld active lines\n",
           (long)state->total_active_lines);

    return 0;
}

/* ============================================================================
 * SPECTRUM OUTPUT
 * ============================================================================ */

static void write_spectrum_csv(const char *filename,
                                const double *spectrum,
                                const double *spectrum_lumina,
                                const int64_t *counts,
                                int n_bins,
                                double nu_min, double nu_max,
                                double t_exp,
                                int64_t n_escaped, int64_t n_absorbed)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", filename);
        return;
    }

    double d_nu = (nu_max - nu_min) / n_bins;

    fprintf(fp, "# LUMINA-SN CUDA Spectrum\n");
    fprintf(fp, "# t_exp = %.2f days\n", t_exp / 86400.0);
    fprintf(fp, "# n_escaped = %ld, n_absorbed = %ld\n",
            (long)n_escaped, (long)n_absorbed);
    fprintf(fp, "wavelength_A,frequency_Hz,L_nu_standard,L_nu_lumina,counts\n");

    for (int i = 0; i < n_bins; i++) {
        double nu = nu_min + (i + 0.5) * d_nu;
        double wl_A = 2.998e10 / nu / 1e-8;  /* cm/s / Hz / (cm/Å) */

        fprintf(fp, "%.4f,%.6e,%.6e,%.6e,%ld\n",
                wl_A, nu, spectrum[i], spectrum_lumina[i], (long)counts[i]);
    }

    fclose(fp);
    printf("[OUTPUT] Written to %s\n", filename);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[])
{
    Config cfg;
    config_defaults(&cfg);

    /* Parse arguments */
    int pos_arg = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            switch (pos_arg) {
                case 0: strcpy(cfg.atomic_file, argv[i]); break;
                case 1: cfg.n_packets = atoll(argv[i]); break;
                case 2: strcpy(cfg.output_file, argv[i]); break;
            }
            pos_arg++;
        }
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          LUMINA-SN CUDA MONTE CARLO TRANSPORT                 ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Atomic data:   %-45s ║\n", cfg.atomic_file);
    printf("║  N_packets:     %-45ld ║\n", (long)cfg.n_packets);
    printf("║  N_shells:      %-45d ║\n", cfg.n_shells);
    printf("║  T_inner:       %-42.0f K ║\n", cfg.T_inner);
    printf("║  v_inner:       %-39.0f km/s ║\n", cfg.v_inner / 1e5);
    printf("║  Abundances:    %-45s ║\n", cfg.stratified ? "STRATIFIED" : "UNIFORM");
    printf("║  Output:        %-45s ║\n", cfg.output_file);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    /* Initialize CUDA */
    if (cuda_init_device(0) != 0) {
        return 1;
    }

    /* Load atomic data on CPU */
    printf("\n[CPU] Loading atomic data...\n");
    AtomicData atomic;
    int status = atomic_data_load_hdf5(cfg.atomic_file, &atomic);
    if (status != 0) {
        fprintf(stderr, "Failed to load atomic data\n");
        return 1;
    }
    printf("[CPU] Loaded: %d ions, %d levels, %ld lines\n",
           atomic.n_ions, atomic.n_levels, (long)atomic.n_lines);

    /* [FORCE FIX - Task Order #038-Revised] Explicitly build Macro-Atom tables */
    printf("\n[Override] Building Macro-Atom Downbranch Tables...\n");
    if (atomic_build_downbranch_table(&atomic) < 0) {
        fprintf(stderr, "CRITICAL ERROR: Failed to build downbranch table.\n");
        return 1;
    }
    printf("[Override] Table built. Total emission entries: %ld\n",
           (long)atomic.downbranch.total_emission_entries);

    if (atomic.downbranch.total_emission_entries == 0) {
        fprintf(stderr, "WARNING: Downbranch table is empty! Simulation will act as Scatter mode.\n");
    }

    /* Set up simulation on CPU */
    SimulationState state;
    if (setup_simulation(&cfg, &atomic, &state) != 0) {
        atomic_data_free(&atomic);
        return 1;
    }

    /* Upload to GPU */
    CudaDeviceMemory mem;
    if (cuda_allocate_atomic_data(&mem, &atomic, &state) != 0) {
        fprintf(stderr, "Failed to upload atomic data to GPU\n");
        atomic_data_free(&atomic);
        simulation_state_free(&state);
        return 1;
    }

    /* [FORCE FIX - Task Order #038-Revised] Upload Macro-Atom downbranch data */
    if (cuda_upload_downbranch_data(&mem, &atomic) != 0) {
        fprintf(stderr, "Failed to upload downbranch data to GPU\n");
        cuda_free_memory(&mem);
        atomic_data_free(&atomic);
        simulation_state_free(&state);
        return 1;
    }
    printf("[Override] Downbranch data uploaded to GPU: %ld entries\n",
           (long)mem.total_emission_entries);

    /* Set up CUDA simulation config */
    CudaSimConfig cuda_cfg;
    memset(&cuda_cfg, 0, sizeof(cuda_cfg));
    cuda_cfg.n_shells = cfg.n_shells;
    cuda_cfg.t_explosion = cfg.t_exp;
    cuda_cfg.nu_min = cfg.nu_min;
    cuda_cfg.nu_max = cfg.nu_max;
    cuda_cfg.n_bins = cfg.n_bins;
    cuda_cfg.d_nu = (cfg.nu_max - cfg.nu_min) / cfg.n_bins;
    cuda_cfg.d_log_nu = (log(cfg.nu_max) - log(cfg.nu_min)) / cfg.n_bins;
    cuda_cfg.n_packets = cfg.n_packets;
    cuda_cfg.packet_energy = 1.0 / cfg.n_packets;
    cuda_cfg.max_interactions = 1000;
    cuda_cfg.max_steps = 10000;
    cuda_cfg.tau_min_active = 0.01;
    cuda_cfg.mu_observer = 1.0;
    cuda_cfg.enable_lumina = 1;

    /* [FORCE FIX - Task Order #038-Revised] Hard-code Physics Flags */
    cuda_cfg.line_interaction_type = 2;  /* 2 = MACROATOM (Force P-Cygni) */
    cuda_cfg.enable_full_relativity = 1;
    printf("[Override] Config forced: line_interaction_type = %d (MACROATOM)\n",
           cuda_cfg.line_interaction_type);

    if (cuda_upload_sim_config(&cuda_cfg) != 0) {
        fprintf(stderr, "Failed to upload simulation config\n");
        cuda_free_memory(&mem);
        atomic_data_free(&atomic);
        simulation_state_free(&state);
        return 1;
    }

    /* Allocate output arrays */
    if (cuda_allocate_output(&mem, cfg.n_bins, cfg.n_packets) != 0) {
        fprintf(stderr, "Failed to allocate output arrays\n");
        cuda_free_memory(&mem);
        atomic_data_free(&atomic);
        simulation_state_free(&state);
        return 1;
    }

    /* Initialize RNG */
    uint64_t seed = (uint64_t)time(NULL);
    if (cuda_init_rng(&mem, cfg.n_packets, seed) != 0) {
        fprintf(stderr, "Failed to initialize RNG\n");
        cuda_free_memory(&mem);
        atomic_data_free(&atomic);
        simulation_state_free(&state);
        return 1;
    }

    /* Run transport */
    printf("\n[CUDA] Running transport with %ld packets...\n", (long)cfg.n_packets);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (cuda_launch_transport(&mem, cfg.n_packets) != 0) {
        fprintf(stderr, "Transport kernel failed\n");
        cuda_free_memory(&mem);
        atomic_data_free(&atomic);
        simulation_state_free(&state);
        return 1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    /* Download results */
    double *h_spectrum = (double *)calloc(cfg.n_bins, sizeof(double));
    double *h_spectrum_lumina = (double *)calloc(cfg.n_bins, sizeof(double));
    int64_t *h_counts = (int64_t *)calloc(cfg.n_bins, sizeof(int64_t));
    int64_t stats[3];

    if (cuda_download_results(&mem, h_spectrum, h_spectrum_lumina, h_counts, stats) != 0) {
        fprintf(stderr, "Failed to download results\n");
    } else {
        /* Print statistics */
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║                    CUDA TRANSPORT RESULTS                     ║\n");
        printf("╠═══════════════════════════════════════════════════════════════╣\n");
        printf("║  Runtime:        %-40.2f ms ║\n", elapsed_ms);
        printf("║  Throughput:     %-38.0f packets/s ║\n",
               cfg.n_packets / (elapsed_ms / 1000.0));
        printf("║  Escaped:        %-38ld (%.1f%%) ║\n",
               (long)stats[0], 100.0 * stats[0] / cfg.n_packets);
        printf("║  Absorbed:       %-38ld (%.1f%%) ║\n",
               (long)stats[1], 100.0 * stats[1] / cfg.n_packets);
        printf("║  Scattered:      %-45ld ║\n", (long)stats[2]);
        printf("╚═══════════════════════════════════════════════════════════════╝\n");

        /* Write spectrum */
        write_spectrum_csv(cfg.output_file, h_spectrum, h_spectrum_lumina,
                           h_counts, cfg.n_bins, cfg.nu_min, cfg.nu_max,
                           cfg.t_exp, stats[0], stats[1]);
    }

    /* Cleanup */
    free(h_spectrum);
    free(h_spectrum_lumina);
    free(h_counts);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cuda_free_memory(&mem);
    atomic_data_free(&atomic);
    simulation_state_free(&state);

    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                   SIMULATION COMPLETE                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
