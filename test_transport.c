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
#include "debug_rng.h"

/* Flag for injected RNG mode */
static int g_use_injected_rng = 0;

/* File path for injected electron densities */
static const char *g_inject_ne_file = NULL;

/* RNG skip count for alignment with TARDIS */
static int g_rng_skip_count = 0;

/* ============================================================================
 * SIMULATION PARAMETERS (can be overridden by command line)
 * ============================================================================ */

#define DEFAULT_N_PACKETS    100000
#define DEFAULT_N_SHELLS     20
#define DEFAULT_N_LINES      1000
#define DEFAULT_T_EXPLOSION  (13.0 * 86400.0)   /* 13 days in seconds (matches TARDIS) */
#define DEFAULT_V_INNER      11000.0  /* km/s (matches TARDIS) */
#define DEFAULT_V_OUTER      20000.0  /* km/s (matches TARDIS) */
#define DEFAULT_R_INNER      1.0e14    /* cm (fallback, computed from v*t) */
#define DEFAULT_R_OUTER      3.0e15    /* cm (fallback, computed from v*t) */
#define DEFAULT_WAVELENGTH_MIN 3000.0  /* Angstrom */
#define DEFAULT_WAVELENGTH_MAX 10000.0 /* Angstrom */
#define DEFAULT_N_WAVELENGTH_BINS 1000
#define DEFAULT_SEED         23111963  /* TARDIS validation seed */
#define DEFAULT_T_INNER      10000.0   /* K photosphere temperature */

/* ============================================================================
 * TRACE MODE GLOBALS
 * ============================================================================ */
static FILE *g_trace_output = NULL;
static int g_trace_step = 0;

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
 * TARDIS ELECTRON DENSITY LOADER
 * Load n_e profile from TARDIS export file
 * ============================================================================ */

static int load_electron_densities(const char *filename, SimulationSetup *setup) {
    /*
     * Load electron densities from TARDIS export file.
     * File format (tardis_electron_densities.txt):
     *   # Comment lines start with #
     *   shell_id r_inner r_outer n_e
     *
     * Or simple format (tardis_ne_only.txt):
     *   # Header comment
     *   n_e_value (one per line)
     */

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[LOAD_NE] Error: Cannot open %s\n", filename);
        return -1;
    }

    char line[512];
    int n_loaded = 0;
    int n_shells = (int)setup->plasma.n_shells;

    fprintf(stderr, "[LOAD_NE] Loading electron densities from %s\n", filename);
    fprintf(stderr, "[LOAD_NE] Expected n_shells = %d\n", n_shells);

    /* Detect file format by checking first data line */
    int is_detailed_format = 0;

    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n') continue;

        /* Try to parse detailed format: shell_id r_inner r_outer n_e */
        int shell_id;
        double r_inner, r_outer, n_e;
        if (sscanf(line, "%d %lf %lf %lf", &shell_id, &r_inner, &r_outer, &n_e) == 4) {
            is_detailed_format = 1;
            if (shell_id < n_shells) {
                setup->electron_density_arr[shell_id] = n_e;
                n_loaded++;
            }
        } else if (sscanf(line, "%lf", &n_e) == 1) {
            /* Simple format: just n_e values */
            if (n_loaded < n_shells) {
                setup->electron_density_arr[n_loaded] = n_e;
                n_loaded++;
            }
        }
    }

    fclose(fp);

    if (n_loaded != n_shells) {
        fprintf(stderr, "[LOAD_NE] Warning: Loaded %d values, expected %d\n",
                n_loaded, n_shells);
    }

    fprintf(stderr, "[LOAD_NE] Loaded %d electron densities (%s format)\n",
            n_loaded, is_detailed_format ? "detailed" : "simple");

    /* Print summary */
    fprintf(stderr, "[LOAD_NE] Electron density profile:\n");
    for (int i = 0; i < n_shells && i < 5; i++) {
        fprintf(stderr, "[LOAD_NE]   Shell %d: n_e = %.6e cm^-3\n",
                i, setup->electron_density_arr[i]);
    }
    if (n_shells > 5) {
        fprintf(stderr, "[LOAD_NE]   ... (%d more shells)\n", n_shells - 5);
    }

    return n_loaded;
}

/* ============================================================================
 * TARDIS-MATCHING SIMULATION SETUP
 * Uses velocities and time to compute radii (homologous expansion)
 * ============================================================================ */

static void setup_tardis_matching_simulation(SimulationSetup *setup,
                                              int64_t n_shells, int64_t n_lines,
                                              double t_explosion,
                                              double v_inner_kms, double v_outer_kms) {
    /*
     * Setup simulation matching TARDIS parameters.
     * Homologous expansion: r = v * t
     */

    /* Allocate arrays */
    setup->r_inner_arr = (double *)malloc(n_shells * sizeof(double));
    setup->r_outer_arr = (double *)malloc(n_shells * sizeof(double));
    setup->line_list_nu_arr = (double *)malloc(n_lines * sizeof(double));
    setup->tau_sobolev_arr = (double *)calloc(n_lines * n_shells, sizeof(double));
    setup->electron_density_arr = (double *)malloc(n_shells * sizeof(double));

    /* Convert velocities to cm/s */
    double v_inner = v_inner_kms * 1e5;
    double v_outer = v_outer_kms * 1e5;

    /* Shell boundaries: linear spacing in velocity */
    double dv = (v_outer - v_inner) / n_shells;
    for (int64_t i = 0; i < n_shells; i++) {
        double v_in = v_inner + i * dv;
        double v_out = v_inner + (i + 1) * dv;
        setup->r_inner_arr[i] = v_in * t_explosion;
        setup->r_outer_arr[i] = v_out * t_explosion;
    }

    /* Line frequencies: optical range (3000-10000 Å → ~3e14 to 1e15 Hz) */
    double nu_min = C_SPEED_OF_LIGHT / (10000.0e-8);  /* 3e14 Hz */
    double nu_max = C_SPEED_OF_LIGHT / (3000.0e-8);   /* 1e15 Hz */

    for (int64_t i = 0; i < n_lines; i++) {
        setup->line_list_nu_arr[i] = nu_min +
            (double)i / (n_lines - 1) * (nu_max - nu_min);
    }

    /* Sobolev optical depths: exponentially decaying with radius */
    srand(12345);  /* Reproducible */
    for (int64_t line = 0; line < n_lines; line++) {
        for (int64_t shell = 0; shell < n_shells; shell++) {
            double tau_base = 5.0 * exp(-2.0 * shell / n_shells);
            double tau_variation = 0.5 + (double)rand() / RAND_MAX;
            setup->tau_sobolev_arr[line * n_shells + shell] =
                tau_base * tau_variation;
        }
    }

    /* Electron density: power-law decrease with radius */
    double n_e_inner = 1e9;  /* cm^-3 at inner boundary */
    double r_inner_cm = setup->r_inner_arr[0];
    for (int64_t i = 0; i < n_shells; i++) {
        double r_mid = 0.5 * (setup->r_inner_arr[i] + setup->r_outer_arr[i]);
        setup->electron_density_arr[i] = n_e_inner * pow(r_inner_cm / r_mid, 3);
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

/* ============================================================================
 * TRACE LOGGING FUNCTIONS
 * ============================================================================ */

static void trace_log_header(void) {
    if (g_trace_output == NULL) return;
    fprintf(g_trace_output,
            "packet_id,step,r,mu,nu,energy,shell_id,status,interaction_type,distance\n");
}

static void trace_log_state(int64_t packet_id, const RPacket *pkt,
                            const char *interaction, double distance) {
    if (g_trace_output == NULL) return;
    fprintf(g_trace_output, "%ld,%d,%.16e,%.16e,%.16e,%.16e,%ld,%ld,%s,%.16e\n",
            (long)packet_id,
            g_trace_step,
            pkt->r,
            pkt->mu,
            pkt->nu,
            pkt->energy,
            (long)pkt->current_shell_id,
            (long)pkt->status,
            interaction,
            distance);
    g_trace_step++;
}

/* ============================================================================
 * INSTRUMENTED SINGLE PACKET LOOP FOR TRACING
 * ============================================================================ */

static void single_packet_loop_with_trace(RPacket *pkt, const NumbaModel *model,
                                           const NumbaPlasma *plasma,
                                           const MonteCarloConfig *config,
                                           Estimators *estimators,
                                           int64_t packet_id) {
    /*
     * Instrumented version of single_packet_loop that logs each step.
     * Matches TARDIS trace format for validation.
     */

    /* Reset step counter */
    g_trace_step = 0;

    /* Initialize line search position */
    rpacket_initialize_line_id(pkt, plasma, model);

    /* Log initial state BEFORE Doppler correction */
    trace_log_state(packet_id, pkt, "INITIAL_RAW", 0.0);

    /* Apply Doppler correction (partial relativity) */
    double inv_doppler = get_inverse_doppler_factor(
        pkt->r, pkt->mu, model->time_explosion);
    pkt->nu *= inv_doppler;
    pkt->energy *= inv_doppler;

    if (config->enable_full_relativity) {
        double beta = pkt->r / (model->time_explosion * C_SPEED_OF_LIGHT);
        pkt->mu = (pkt->mu + beta) / (1.0 + beta * pkt->mu);
    }

    /* Log state AFTER Doppler correction */
    trace_log_state(packet_id, pkt, "DOPPLER_INIT", 0.0);

    /* Main transport loop */
    int max_steps = 10000;  /* Safety limit */

    while (pkt->status == PACKET_IN_PROCESS && g_trace_step < max_steps) {

        /* Find next interaction */
        double distance;
        int delta_shell;
        InteractionType itype = trace_packet(
            pkt, model, plasma, config, estimators, &distance, &delta_shell);

        /* Process based on interaction type */
        switch (itype) {

            case INTERACTION_BOUNDARY:
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                move_packet_across_shell_boundary(pkt, delta_shell, model->n_shells);
                trace_log_state(packet_id, pkt, "BOUNDARY", distance);
                break;

            case INTERACTION_LINE:
                pkt->last_interaction_type = 2;
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                line_scatter(pkt, model->time_explosion,
                            config->line_interaction_type, plasma,
                            config->atomic_data);
                trace_log_state(packet_id, pkt, "LINE", distance);
                break;

            case INTERACTION_ESCATTERING:
                pkt->last_interaction_type = 1;
                move_r_packet(pkt, distance, model->time_explosion, estimators);
                thomson_scatter(pkt, model->time_explosion);
                trace_log_state(packet_id, pkt, "ESCATTERING", distance);
                break;
        }
    }

    /* Log final state */
    const char *final_str = (pkt->status == PACKET_EMITTED) ? "FINAL_EMITTED" :
                            (pkt->status == PACKET_REABSORBED) ? "FINAL_REABSORBED" :
                            "FINAL_UNKNOWN";
    trace_log_state(packet_id, pkt, final_str, 0.0);
}

/* ============================================================================
 * BLACKBODY PACKET SAMPLING (Matches TARDIS)
 * ============================================================================ */

/**
 * Get random number - uses injected RNG if enabled, otherwise native Xorshift
 */
static inline double get_rng(RNGState *rng) {
    if (g_use_injected_rng) {
        return debug_rng_next();
    } else {
        return rng_uniform(rng);
    }
}

static double sample_blackbody_nu(double T, RNGState *rng) {
    /*
     * Sample frequency from Planck distribution using rejection sampling.
     * Planck function: B(x) ∝ x^3 / (e^x - 1) where x = h*nu / (k*T)
     */
    const double h = 6.62607015e-27;  /* Planck constant [erg s] */
    const double k = 1.380649e-16;    /* Boltzmann constant [erg/K] */

    double x, y, f_x;
    do {
        x = get_rng(rng) * 20.0;  /* x = h*nu / (k*T), range [0, 20] */
        y = get_rng(rng) * 1.5;   /* Envelope */
        f_x = x * x * x / (exp(x) - 1.0 + 1e-300);  /* Planck function */
    } while (y > f_x / 1.42);  /* 1.42 ≈ max of x^3/(e^x-1) */

    return x * k * T / h;
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
 * TRACE MODE: Generate TARDIS-format trace for validation
 * ============================================================================ */

static int run_trace_mode(uint64_t seed, int n_packets, const char *output_file) {
    /*
     * Run packets with detailed tracing for TARDIS comparison.
     * Output CSV to stdout (or file) matching TARDIS format.
     */

    /* Setup output */
    if (output_file) {
        g_trace_output = fopen(output_file, "w");
        if (!g_trace_output) {
            fprintf(stderr, "Error: Cannot open %s for writing\n", output_file);
            return 1;
        }
    } else {
        g_trace_output = stdout;
    }

    /* Print config to stderr */
    fprintf(stderr, "======================================================================\n");
    fprintf(stderr, "LUMINA-SN TARDIS TRACE MODE\n");
    fprintf(stderr, "======================================================================\n");
    fprintf(stderr, "  Seed:       %lu\n", (unsigned long)seed);
    fprintf(stderr, "  Packets:    %d\n", n_packets);
    fprintf(stderr, "  Output:     %s\n", output_file ? output_file : "stdout");
    fprintf(stderr, "======================================================================\n");

    /* Setup simulation matching TARDIS config */
    double t_exp = DEFAULT_T_EXPLOSION;
    double v_inner = DEFAULT_V_INNER;
    double v_outer = DEFAULT_V_OUTER;
    int n_shells = DEFAULT_N_SHELLS;

    SimulationSetup setup;
    setup_tardis_matching_simulation(&setup, n_shells, DEFAULT_N_LINES,
                                      t_exp, v_inner, v_outer);

    /* Load TARDIS electron densities if specified */
    if (g_inject_ne_file) {
        int n_loaded = load_electron_densities(g_inject_ne_file, &setup);
        if (n_loaded <= 0) {
            fprintf(stderr, "Error: Failed to load electron densities\n");
            return 1;
        }
    }

    double r_inner = setup.r_inner_arr[0];
    double r_outer = setup.r_outer_arr[n_shells - 1];

    fprintf(stderr, "  t_exp:      %.6e s (%.1f days)\n", t_exp, t_exp / 86400.0);
    fprintf(stderr, "  v_inner:    %.0f km/s\n", v_inner);
    fprintf(stderr, "  v_outer:    %.0f km/s\n", v_outer);
    fprintf(stderr, "  r_inner:    %.6e cm\n", r_inner);
    fprintf(stderr, "  r_outer:    %.6e cm\n", r_outer);
    fprintf(stderr, "  n_shells:   %d\n", n_shells);
    fprintf(stderr, "  T_inner:    %.0f K\n", DEFAULT_T_INNER);
    fprintf(stderr, "======================================================================\n\n");

    /* Monte Carlo config */
    MonteCarloConfig mc_config = mc_config_default();
    mc_config.enable_full_relativity = 0;
    mc_config.disable_line_scattering = 0;
    mc_config.line_interaction_type = LINE_SCATTER;  /* Simple scatter for comparison */

    /* Enable debug mode for transport diagnostics */
    rpacket_set_debug_mode(1, g_use_injected_rng);

    /* Write CSV header */
    trace_log_header();

    /* Run packets */
    for (int i = 0; i < n_packets; i++) {
        RPacket pkt;

        /* Initialize RNG with seed + packet index */
        RNGState rng;
        rng_init(&rng, seed + i);

        /* ================================================================
         * HARDCODED PACKET #0 STATE FOR TARDIS VALIDATION
         * ================================================================
         * These values are taken directly from TARDIS trace Step 0.
         * Purpose: Bypass initialization divergence to validate transport.
         * ================================================================ */
        if (i == 0) {
            /* TARDIS Packet #0 exact initial state */
            pkt.r      = 1.2355200000000000e+15;
            pkt.mu     = 0.8648151880391006;
            pkt.nu     = 1.4056101158989945e+15;  /* Note: 1.4e15 Hz, not 1.4e14 */
            pkt.energy = 0.1032771750556338;

            fprintf(stderr, "[Packet %d] *** HARDCODED TARDIS STATE ***\n", i);
            fprintf(stderr, "           r  = %.16e\n", pkt.r);
            fprintf(stderr, "           mu = %.16e\n", pkt.mu);
            fprintf(stderr, "           nu = %.16e\n", pkt.nu);
            fprintf(stderr, "           E  = %.16e\n", pkt.energy);

            /* Skip RNG calls that would have been used for initialization */
            if (g_use_injected_rng) {
                /* Advance RNG index to align with where TARDIS would be */
                /* TARDIS uses ~20 RNG calls for blackbody sampling */
                fprintf(stderr, "           RNG index before skip: %d\n",
                        debug_rng_get_index());
            }
        } else {
            /* Normal initialization for other packets */
            pkt.r = r_inner;

            double xi = get_rng(&rng);
            pkt.mu = sqrt(xi);

            fprintf(stderr, "[Packet %d] RNG xi[0]=%.18e -> mu=%.18e\n",
                    i, xi, pkt.mu);

            pkt.nu = sample_blackbody_nu(DEFAULT_T_INNER, &rng);

            fprintf(stderr, "[Packet %d] After BB sampling, RNG index=%d\n",
                    i, g_use_injected_rng ? debug_rng_get_index() : -1);

            pkt.energy = 1.0;
        }

        /* Initialize other fields (common to all packets) */
        pkt.current_shell_id = 0;
        pkt.status = PACKET_IN_PROCESS;
        pkt.index = i;
        pkt.next_line_id = 0;
        pkt.last_interaction_type = -1;
        pkt.last_interaction_in_nu = 0.0;
        pkt.last_line_interaction_in_id = -1;
        pkt.last_line_interaction_out_id = -1;
        rng_init(&pkt.rng_state, seed + i);

        fprintf(stderr, "[Packet %d] Init: r=%.6e, mu=%.6f, nu=%.6e, E=%.6e\n",
                i, pkt.r, pkt.mu, pkt.nu, pkt.energy);

        /* Run traced transport */
        single_packet_loop_with_trace(&pkt, &setup.model, &setup.plasma,
                                       &mc_config, NULL, i);

        fprintf(stderr, "           -> %s after %d steps\n",
                (pkt.status == PACKET_EMITTED) ? "EMITTED" : "REABSORBED",
                g_trace_step);
    }

    /* Cleanup */
    free_simulation_setup(&setup);

    if (output_file && g_trace_output != stdout) {
        fclose(g_trace_output);
    }
    g_trace_output = NULL;

    fprintf(stderr, "\n[DONE] Trace generation complete.\n");
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
    printf("  --trace                  Generate TARDIS-format trace (CSV to stdout)\n");
    printf("\n");
    printf("Options:\n");
    printf("  --seed <N>               Random seed (default: %d)\n", DEFAULT_SEED);
    printf("  --n_packets <N>          Number of packets (default: 1 for trace mode)\n");
    printf("  --output <file.csv>      Write trace to file instead of stdout\n");
    printf("  --inject-rng <file.txt>  Use pre-generated RNG stream (for TARDIS sync)\n");
    printf("  --inject-ne <file.txt>   Use TARDIS electron density profile\n");
    printf("  --rng-skip <N>           Skip first N RNG values (align with TARDIS)\n");
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
    const char *output_file = NULL;
    const char *inject_rng_file = NULL;
    int64_t n_packets = 0;
    uint64_t seed = DEFAULT_SEED;
    int mode = 0;  /* 0=none, 1=validate, 2=simulate, 3=trace */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate") == 0 && i + 1 < argc) {
            python_trace = argv[++i];
            mode = 1;
        } else if (strcmp(argv[i], "--simulate") == 0 && i + 1 < argc) {
            n_packets = atol(argv[++i]);
            mode = 2;
        } else if (strcmp(argv[i], "--trace") == 0) {
            mode = 3;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--n_packets") == 0 && i + 1 < argc) {
            n_packets = atol(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--inject-rng") == 0 && i + 1 < argc) {
            inject_rng_file = argv[++i];
        } else if (strcmp(argv[i], "--inject-ne") == 0 && i + 1 < argc) {
            g_inject_ne_file = argv[++i];
        } else if (strcmp(argv[i], "--rng-skip") == 0 && i + 1 < argc) {
            g_rng_skip_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            c_csv = argv[++i];
        } else if (strcmp(argv[i], "--spectrum") == 0 && i + 1 < argc) {
            spectrum_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Load injected RNG if specified */
    if (inject_rng_file) {
        int count = debug_rng_load(inject_rng_file);
        if (count < 0) {
            fprintf(stderr, "Error: Failed to load RNG stream\n");
            return 1;
        }
        g_use_injected_rng = 1;
        fprintf(stderr, "[MAIN] RNG injection ENABLED from: %s\n", inject_rng_file);

        /* Skip RNG values to align with TARDIS transport start */
        if (g_rng_skip_count > 0) {
            debug_rng_skip(g_rng_skip_count);
            fprintf(stderr, "[MAIN] Skipped %d RNG values to sync with TARDIS transport\n",
                    g_rng_skip_count);
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
        case 3:
            /* Trace mode: default to 1 packet if not specified */
            if (n_packets <= 0) n_packets = 1;
            return run_trace_mode(seed, (int)n_packets, output_file);
        default:
            print_usage(argv[0]);
            return 1;
    }
}
