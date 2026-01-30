/**
 * LUMINA-SN Full Transport Test and Validation
 * test_transport.c - Main driver for Monte Carlo transport
 *
 * Features:
 *   1. Single packet validation against Python traces
 *   2. Full ensemble simulation with OpenMP parallelization
 *   3. LUMINA rotation post-processing for spectrum synthesis
 *
 * Compile:
 *   gcc -O3 -fopenmp test_transport.c rpacket.c validation.c lumina_rotation.c -lm -o test_transport
 *
 * Usage:
 *   ./test_transport --validate trace_python.bin    # Validate against Python
 *   ./test_transport --simulate 1000000             # Run N packets
 *   ./test_transport --spectrum output.csv          # Generate spectrum
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "physics_kernels.h"
#include "rpacket.h"
#include "validation.h"
#include "lumina_rotation.h"

/* ============================================================================
 * SIMULATION PARAMETERS (can be overridden by command line)
 * ============================================================================ */

#define DEFAULT_N_PACKETS    100000
#define DEFAULT_N_SHELLS     20
#define DEFAULT_N_LINES      1000
#define DEFAULT_T_EXPLOSION  86400.0   /* 1 day in seconds */
#define DEFAULT_R_INNER      1.0e14    /* cm */
#define DEFAULT_R_OUTER      3.0e15    /* cm */
#define DEFAULT_WAVELENGTH_MIN 3000.0  /* Angstrom */
#define DEFAULT_WAVELENGTH_MAX 10000.0 /* Angstrom */
#define DEFAULT_N_WAVELENGTH_BINS 1000

/* ============================================================================
 * MINIMAL MODEL SETUP (for testing without full TARDIS plasma)
 * ============================================================================ */

typedef struct {
    NumbaModel model;
    NumbaPlasma plasma;
    double *r_inner_arr;
    double *r_outer_arr;
    double *line_list_nu_arr;
    double *tau_sobolev_arr;
    double *electron_density_arr;
} SimulationSetup;

static void setup_minimal_simulation(SimulationSetup *setup,
                                      int64_t n_shells, int64_t n_lines,
                                      double t_explosion,
                                      double r_inner, double r_outer) {
    /*
     * Create a minimal but physically reasonable simulation setup.
     * This is used for testing; production would load real TARDIS data.
     */

    /* Allocate arrays */
    setup->r_inner_arr = (double *)malloc(n_shells * sizeof(double));
    setup->r_outer_arr = (double *)malloc(n_shells * sizeof(double));
    setup->line_list_nu_arr = (double *)malloc(n_lines * sizeof(double));
    setup->tau_sobolev_arr = (double *)calloc(n_lines * n_shells, sizeof(double));
    setup->electron_density_arr = (double *)malloc(n_shells * sizeof(double));

    /* Shell boundaries: logarithmic spacing */
    double log_r_inner = log10(r_inner);
    double log_r_outer = log10(r_outer);
    double d_log_r = (log_r_outer - log_r_inner) / n_shells;

    for (int64_t i = 0; i < n_shells; i++) {
        setup->r_inner_arr[i] = pow(10, log_r_inner + i * d_log_r);
        setup->r_outer_arr[i] = pow(10, log_r_inner + (i + 1) * d_log_r);
    }

    /* Line frequencies: optical range (3000-10000 Å → ~3e14 to 1e15 Hz) */
    double nu_min = C_SPEED_OF_LIGHT / (10000.0e-8);  /* 3e14 Hz */
    double nu_max = C_SPEED_OF_LIGHT / (3000.0e-8);   /* 1e15 Hz */

    for (int64_t i = 0; i < n_lines; i++) {
        /* Linear spacing in frequency */
        setup->line_list_nu_arr[i] = nu_min +
            (double)i / (n_lines - 1) * (nu_max - nu_min);
    }

    /* Sobolev optical depths: exponentially decaying with radius */
    /* Typical values: 0.1 to 10 */
    srand(12345);  /* Reproducible */
    for (int64_t line = 0; line < n_lines; line++) {
        for (int64_t shell = 0; shell < n_shells; shell++) {
            /* Higher τ for inner shells, lower for outer */
            double tau_base = 5.0 * exp(-2.0 * shell / n_shells);
            double tau_variation = 0.5 + (double)rand() / RAND_MAX;
            setup->tau_sobolev_arr[line * n_shells + shell] =
                tau_base * tau_variation;
        }
    }

    /* Electron density: power-law decrease with radius */
    /* n_e ∝ r^(-3) for homologous expansion */
    double n_e_inner = 1e9;  /* cm^-3 at inner boundary */
    for (int64_t i = 0; i < n_shells; i++) {
        double r_mid = 0.5 * (setup->r_inner_arr[i] + setup->r_outer_arr[i]);
        setup->electron_density_arr[i] = n_e_inner *
            pow(r_inner / r_mid, 3);
    }

    /* Setup model structure */
    setup->model.r_inner = setup->r_inner_arr;
    setup->model.r_outer = setup->r_outer_arr;
    setup->model.time_explosion = t_explosion;
    setup->model.n_shells = n_shells;

    /* Setup plasma structure */
    setup->plasma.line_list_nu = setup->line_list_nu_arr;
    setup->plasma.tau_sobolev = setup->tau_sobolev_arr;
    setup->plasma.electron_density = setup->electron_density_arr;
    setup->plasma.n_lines = n_lines;
    setup->plasma.n_shells = n_shells;
}

static void free_simulation_setup(SimulationSetup *setup) {
    free(setup->r_inner_arr);
    free(setup->r_outer_arr);
    free(setup->line_list_nu_arr);
    free(setup->tau_sobolev_arr);
    free(setup->electron_density_arr);
}

/* ============================================================================
 * PACKET INITIALIZATION
 * ============================================================================ */

static void initialize_packet_from_photosphere(RPacket *pkt, int64_t index,
                                                const SimulationSetup *setup) {
    /*
     * Initialize a packet at the inner boundary (photosphere).
     *
     * Starting conditions:
     *   - r: just inside first shell
     *   - μ: isotropic outward (limb-darkened in real case)
     *   - ν: blackbody sampling (simplified: uniform in log)
     *   - energy: 1.0 (normalized per packet)
     */

    double r_start = setup->r_inner_arr[0] * 1.001;  /* Just above photosphere */

    /* Isotropic outward emission: μ = sqrt(ξ) for ξ ∈ [0,1] */
    double xi = (double)rand() / RAND_MAX;
    double mu_start = sqrt(xi);  /* Range [0, 1], biased outward */

    /* Frequency: sample from optical range */
    double nu_min = C_SPEED_OF_LIGHT / (10000.0e-8);
    double nu_max = C_SPEED_OF_LIGHT / (3000.0e-8);
    double xi_nu = (double)rand() / RAND_MAX;
    double nu_start = nu_min * pow(nu_max / nu_min, xi_nu);

    double energy_start = 1.0;  /* Normalized */

    rpacket_init(pkt, r_start, mu_start, nu_start, energy_start,
                 (int64_t)rand(), index);
    pkt->current_shell_id = 0;
}

/* ============================================================================
 * VALIDATION MODE: Compare single packet against Python trace
 * ============================================================================ */

static int run_validation(const char *python_trace_file,
                          const char *c_trace_file,
                          const char *c_csv_file) {
    printf("\n=== LUMINA-SN Validation Mode ===\n\n");

    /* Load Python trace */
    printf("Loading Python trace: %s\n", python_trace_file);
    ValidationTrace *trace_py = validation_trace_load_binary(python_trace_file);
    if (!trace_py) {
        fprintf(stderr, "Error: Cannot load Python trace file\n");
        return 1;
    }
    printf("  Loaded %ld steps\n", (long)trace_py->n_snapshots);

    /* Get initial conditions from Python trace */
    if (trace_py->n_snapshots == 0) {
        fprintf(stderr, "Error: Empty Python trace\n");
        validation_trace_free(trace_py);
        return 1;
    }

    /* Setup minimal simulation */
    SimulationSetup setup;
    setup_minimal_simulation(&setup, DEFAULT_N_SHELLS, DEFAULT_N_LINES,
                             DEFAULT_T_EXPLOSION, DEFAULT_R_INNER, DEFAULT_R_OUTER);

    /* Create packet with same initial conditions */
    /* Note: In production, these would come from the trace file header */
    RPacket pkt;
    rpacket_init(&pkt, setup.r_inner_arr[0] * 1.001, 0.5, 5.5e14, 1.0, 42, 0);
    pkt.current_shell_id = 0;

    /* Run C transport with tracing */
    printf("Running C transport...\n");
    ValidationTrace *trace_c = validation_trace_create(0, 10000);
    MonteCarloConfig mc_cfg = mc_config_default();
    single_packet_loop_traced(&pkt, &setup.model, &setup.plasma, &mc_cfg, NULL, trace_c);
    printf("  Generated %ld steps\n", (long)trace_c->n_snapshots);
    printf("  Final status: %s\n",
           pkt.status == PACKET_EMITTED ? "EMITTED" : "REABSORBED");

    /* Write C trace */
    if (c_trace_file) {
        validation_trace_write_binary(trace_c, c_trace_file);
        printf("  Wrote binary trace: %s\n", c_trace_file);
    }
    if (c_csv_file) {
        validation_trace_write_csv(trace_c, c_csv_file);
        printf("  Wrote CSV trace: %s\n", c_csv_file);
    }

    /* Compare traces */
    printf("\nComparing traces (tolerance: 1e-10)...\n");
    int64_t mismatches = validation_compare_traces(trace_c, trace_py, 1e-10);

    if (mismatches == 0) {
        printf("  \033[32mVALIDATION PASSED\033[0m: All steps match\n");
    } else {
        printf("  \033[31mVALIDATION FAILED\033[0m: %ld mismatches\n",
               (long)mismatches);
    }

    /* Cleanup */
    validation_trace_free(trace_py);
    validation_trace_free(trace_c);
    free_simulation_setup(&setup);

    return (mismatches == 0) ? 0 : 1;
}

/* ============================================================================
 * SIMULATION MODE: Run full Monte Carlo with OpenMP
 * ============================================================================ */

static int run_simulation(int64_t n_packets, const char *spectrum_file) {
    printf("\n=== LUMINA-SN Simulation Mode ===\n\n");
    printf("Configuration:\n");
    printf("  N_packets: %ld\n", (long)n_packets);
    printf("  N_shells: %d\n", DEFAULT_N_SHELLS);
    printf("  N_lines: %d\n", DEFAULT_N_LINES);
    printf("  t_explosion: %.0f s (%.2f days)\n",
           DEFAULT_T_EXPLOSION, DEFAULT_T_EXPLOSION / 86400.0);

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    printf("  OpenMP threads: %d\n", n_threads);
#else
    printf("  OpenMP: disabled\n");
#endif

    /* Setup simulation */
    SimulationSetup setup;
    setup_minimal_simulation(&setup, DEFAULT_N_SHELLS, DEFAULT_N_LINES,
                             DEFAULT_T_EXPLOSION, DEFAULT_R_INNER, DEFAULT_R_OUTER);

    /* Observer configuration for LUMINA rotation */
    ObserverConfig obs_config = {
        .mu_observer = 1.0,  /* Face-on */
        .time_explosion = DEFAULT_T_EXPLOSION,
        .r_outer = setup.r_outer_arr[DEFAULT_N_SHELLS - 1],
        .wavelength_min = DEFAULT_WAVELENGTH_MIN,
        .wavelength_max = DEFAULT_WAVELENGTH_MAX,
        .n_wavelength_bins = DEFAULT_N_WAVELENGTH_BINS
    };

    /* Create spectrum accumulator */
    Spectrum *spectrum = spectrum_create(obs_config.wavelength_min,
                                         obs_config.wavelength_max,
                                         obs_config.n_wavelength_bins);

    /* Statistics */
    int64_t n_emitted = 0;
    int64_t n_reabsorbed = 0;

    printf("\nRunning Monte Carlo transport...\n");
    double t_start = (double)clock() / CLOCKS_PER_SEC;

    /* Monte Carlo configuration */
    MonteCarloConfig mc_config = mc_config_default();

    /*
     * OPENMP PARALLELIZATION
     * ----------------------
     * Each thread processes packets independently.
     * Spectrum accumulation uses atomic updates.
     */
    #ifdef _OPENMP
    #pragma omp parallel reduction(+:n_emitted,n_reabsorbed)
    #endif
    {
        /* Thread-local RNG seed */
        #ifdef _OPENMP
        unsigned int thread_seed = (unsigned int)(time(NULL) ^ omp_get_thread_num());
        #else
        unsigned int thread_seed = (unsigned int)time(NULL);
        #endif

        /* Thread-local spectrum (to avoid contention) */
        Spectrum *local_spectrum = spectrum_create(obs_config.wavelength_min,
                                                   obs_config.wavelength_max,
                                                   obs_config.n_wavelength_bins);

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 100)
        #endif
        for (int64_t i = 0; i < n_packets; i++) {
            /* Initialize packet */
            RPacket pkt;

            /* Thread-safe random number generation */
            double xi = (double)rand_r(&thread_seed) / RAND_MAX;
            double mu = sqrt(xi);
            double xi_nu = (double)rand_r(&thread_seed) / RAND_MAX;
            double nu_min = C_SPEED_OF_LIGHT / (10000.0e-8);
            double nu_max = C_SPEED_OF_LIGHT / (3000.0e-8);
            double nu = nu_min * pow(nu_max / nu_min, xi_nu);

            rpacket_init(&pkt, setup.r_inner_arr[0] * 1.001, mu, nu, 1.0,
                         (int64_t)rand_r(&thread_seed), i);
            pkt.current_shell_id = 0;

            /* Run transport */
            single_packet_loop(&pkt, &setup.model, &setup.plasma, &mc_config, NULL);

            /* Process escaped packets with LUMINA rotation */
            if (pkt.status == PACKET_EMITTED) {
                n_emitted++;

                RotatedPacket rotated;
                lumina_rotate_packet(pkt.r, pkt.mu, pkt.nu, pkt.energy,
                                     &obs_config, &rotated);

                spectrum_add_packet(local_spectrum, &rotated,
                                    DEFAULT_T_EXPLOSION);
            } else {
                n_reabsorbed++;
            }

            /* Progress indicator (only thread 0) */
            #ifdef _OPENMP
            if (omp_get_thread_num() == 0 && i % 10000 == 0) {
                printf("\r  Progress: %.1f%%", 100.0 * i / n_packets);
                fflush(stdout);
            }
            #else
            if (i % 10000 == 0) {
                printf("\r  Progress: %.1f%%", 100.0 * i / n_packets);
                fflush(stdout);
            }
            #endif
        }

        /* Merge thread-local spectra (critical section) */
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            for (int64_t b = 0; b < spectrum->n_bins; b++) {
                spectrum->flux[b] += local_spectrum->flux[b];
            }
            spectrum->total_luminosity += local_spectrum->total_luminosity;
            spectrum->n_packets_used += local_spectrum->n_packets_used;
        }

        spectrum_free(local_spectrum);
    }

    double t_end = (double)clock() / CLOCKS_PER_SEC;
    printf("\r  Progress: 100.0%%\n");

    /* Report statistics */
    printf("\nResults:\n");
    printf("  Packets emitted:    %ld (%.1f%%)\n",
           (long)n_emitted, 100.0 * n_emitted / n_packets);
    printf("  Packets reabsorbed: %ld (%.1f%%)\n",
           (long)n_reabsorbed, 100.0 * n_reabsorbed / n_packets);
    printf("  Total luminosity:   %.3e erg/s\n", spectrum->total_luminosity);
    printf("  Runtime:            %.2f s\n", t_end - t_start);
    printf("  Packets/sec:        %.0f\n", n_packets / (t_end - t_start));

    /* Write spectrum */
    if (spectrum_file) {
        spectrum_normalize(spectrum, n_packets, DEFAULT_T_EXPLOSION);
        spectrum_write_csv(spectrum, spectrum_file);
        printf("  Spectrum written:   %s\n", spectrum_file);
    }

    /* Cleanup */
    spectrum_free(spectrum);
    free_simulation_setup(&setup);

    return 0;
}

/* ============================================================================
 * MAIN: Command-line interface
 * ============================================================================ */

static void print_usage(const char *prog) {
    printf("LUMINA-SN: Monte Carlo Radiative Transfer for Supernovae\n\n");
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Modes:\n");
    printf("  --validate <python.bin>  Validate against Python trace\n");
    printf("  --simulate <N>           Run N packet simulation\n");
    printf("\n");
    printf("Options:\n");
    printf("  --trace <file.bin>       Write C trace to binary file\n");
    printf("  --csv <file.csv>         Write C trace/spectrum to CSV\n");
    printf("  --spectrum <file.csv>    Write spectrum to CSV\n");
    printf("  --help                   Show this help\n");
}

int main(int argc, char *argv[]) {
    /* Parse command line */
    const char *python_trace = NULL;
    const char *c_trace = NULL;
    const char *c_csv = NULL;
    const char *spectrum_file = NULL;
    int64_t n_packets = 0;
    int mode = 0;  /* 0=none, 1=validate, 2=simulate */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate") == 0 && i + 1 < argc) {
            python_trace = argv[++i];
            mode = 1;
        } else if (strcmp(argv[i], "--simulate") == 0 && i + 1 < argc) {
            n_packets = atol(argv[++i]);
            mode = 2;
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            c_trace = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            c_csv = argv[++i];
        } else if (strcmp(argv[i], "--spectrum") == 0 && i + 1 < argc) {
            spectrum_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Default: run simulation with default packets */
    if (mode == 0) {
        mode = 2;
        n_packets = DEFAULT_N_PACKETS;
        spectrum_file = "spectrum.csv";
    }

    /* Execute mode */
    switch (mode) {
        case 1:
            return run_validation(python_trace, c_trace, c_csv);
        case 2:
            return run_simulation(n_packets, spectrum_file);
        default:
            print_usage(argv[0]);
            return 1;
    }
}
