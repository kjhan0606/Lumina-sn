/**
 * LUMINA-SN Integrated Plasma-Transport Simulation
 * test_integrated.c - Full MC simulation with Saha-Boltzmann opacities
 *
 * This test combines:
 *   1. Atomic data loading (HDF5)
 *   2. Saha-Boltzmann ionization calculation
 *   3. Sobolev line opacity computation
 *   4. Monte Carlo packet transport
 *   5. LUMINA rotation for spectrum synthesis
 *
 * Usage:
 *   ./test_integrated [atomic_data.h5] [n_packets] [output.csv]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomic_data.h"
#include "plasma_physics.h"
#include "simulation_state.h"
#include "physics_kernels.h"
#include "rpacket.h"
#include "lumina_rotation.h"
#include "macro_atom.h"
#include "virtual_packet.h"

/* Required by physics_kernels.h - set to 0 for first-order Doppler */
int ENABLE_FULL_RELATIVITY = 0;

/* ============================================================================
 * DIAGNOSTIC COUNTERS (Line Interaction Statistics)
 * ============================================================================ */

/* Global counters for line interaction diagnostics */
static struct {
    /* Interaction counts by wavelength band */
    long n_uv_interact;       /* λ < 3000 Å */
    long n_blue_interact;     /* 3000-5000 Å */
    long n_red_interact;      /* 5000-7000 Å */
    long n_ir_interact;       /* λ > 7000 Å */

    /* Outcome counts */
    long n_scatter;           /* Pure resonance scatter */
    long n_thermalize;        /* Thermal re-emission */
    long n_uv_to_blue;        /* UV → blue fluorescence */
    long n_blue_scatter;      /* Blue preserved scatter */
    long n_red_scatter;       /* Red preserved scatter */
    long n_ir_absorbed;       /* IR true absorption */

    /* Frequency redistribution */
    double sum_input_wl;      /* Sum of input wavelengths */
    double sum_output_wl;     /* Sum of output wavelengths */
    long n_freq_samples;      /* Number of frequency samples */

    /* Macro-atom statistics */
    long n_macro_atom_calls;  /* Total macro-atom process calls */
    long n_macro_emit;        /* Radiative de-activations */
    long n_macro_absorb;      /* Collisional de-activations (absorbed) */

    /* Debug: line selection tracking */
    long n_line_checks;       /* Times we checked for line interaction */
    long n_line_no_idx;       /* Times line_idx was -1 */
    long n_line_low_tau;      /* Times tau_line <= 0.1 */
    long n_line_selected;     /* Times line was selected as d_min */
} g_line_stats = {0};

static void reset_line_stats(void) {
    memset(&g_line_stats, 0, sizeof(g_line_stats));
}

static void print_line_stats(void) {
    long total = g_line_stats.n_uv_interact + g_line_stats.n_blue_interact +
                 g_line_stats.n_red_interact + g_line_stats.n_ir_interact;

    printf("\n[LINE INTERACTION DIAGNOSTICS]\n");
    printf("  Line selection debug:\n");
    printf("    Total checks:     %ld\n", g_line_stats.n_line_checks);
    printf("    No line found:    %ld (%.1f%%)\n", g_line_stats.n_line_no_idx,
           g_line_stats.n_line_checks > 0 ? 100.0 * g_line_stats.n_line_no_idx / g_line_stats.n_line_checks : 0.0);
    printf("    Low tau (<0.1):   %ld (%.1f%%)\n", g_line_stats.n_line_low_tau,
           g_line_stats.n_line_checks > 0 ? 100.0 * g_line_stats.n_line_low_tau / g_line_stats.n_line_checks : 0.0);
    printf("    Line selected:    %ld (%.1f%%)\n", g_line_stats.n_line_selected,
           g_line_stats.n_line_checks > 0 ? 100.0 * g_line_stats.n_line_selected / g_line_stats.n_line_checks : 0.0);
    printf("  Total line interactions: %ld\n", total);
    printf("  Macro-atom calls: %ld\n", g_line_stats.n_macro_atom_calls);

    if (total == 0) return;

    printf("\n[LINE INTERACTION DIAGNOSTICS]\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    printf("  Wavelength Band Statistics:\n");
    printf("    UV   (< 3000 Å):    %8ld (%5.1f%%)\n",
           g_line_stats.n_uv_interact, 100.0 * g_line_stats.n_uv_interact / total);
    printf("    Blue (3000-5000 Å): %8ld (%5.1f%%)\n",
           g_line_stats.n_blue_interact, 100.0 * g_line_stats.n_blue_interact / total);
    printf("    Red  (5000-7000 Å): %8ld (%5.1f%%)\n",
           g_line_stats.n_red_interact, 100.0 * g_line_stats.n_red_interact / total);
    printf("    IR   (> 7000 Å):    %8ld (%5.1f%%)\n",
           g_line_stats.n_ir_interact, 100.0 * g_line_stats.n_ir_interact / total);
    printf("  ─────────────────────────────────────────────────────────\n");
    printf("  Interaction Outcomes:\n");
    printf("    Pure scatter:       %8ld\n", g_line_stats.n_scatter);
    printf("    Thermalize:         %8ld\n", g_line_stats.n_thermalize);
    printf("    UV → Blue fluor:    %8ld\n", g_line_stats.n_uv_to_blue);
    printf("    Blue scatter:       %8ld\n", g_line_stats.n_blue_scatter);
    printf("    Red scatter:        %8ld\n", g_line_stats.n_red_scatter);
    printf("    IR absorbed:        %8ld\n", g_line_stats.n_ir_absorbed);
    printf("  ─────────────────────────────────────────────────────────\n");

    if (g_line_stats.n_freq_samples > 0) {
        double avg_in = g_line_stats.sum_input_wl / g_line_stats.n_freq_samples;
        double avg_out = g_line_stats.sum_output_wl / g_line_stats.n_freq_samples;
        printf("  Frequency Redistribution:\n");
        printf("    Mean input λ:  %.1f Å\n", avg_in);
        printf("    Mean output λ: %.1f Å\n", avg_out);
        printf("    Shift ratio:   %.3f (%.1f%% %s)\n",
               avg_out / avg_in,
               fabs(avg_out / avg_in - 1.0) * 100.0,
               avg_out > avg_in ? "RED-shift" : "BLUE-shift");
    }
    printf("  ─────────────────────────────────────────────────────────\n");

    /* Macro-atom statistics (if macro-atom mode was used) */
    if (g_line_stats.n_macro_atom_calls > 0) {
        printf("  Macro-Atom Statistics:\n");
        printf("    Total calls:        %8ld\n", g_line_stats.n_macro_atom_calls);
        printf("    Radiative emit:     %8ld (%.1f%%)\n",
               g_line_stats.n_macro_emit,
               100.0 * g_line_stats.n_macro_emit / g_line_stats.n_macro_atom_calls);
        printf("    Absorbed:           %8ld (%.1f%%)\n",
               g_line_stats.n_macro_absorb,
               100.0 * g_line_stats.n_macro_absorb / g_line_stats.n_macro_atom_calls);
        printf("  ─────────────────────────────────────────────────────────\n");
    }
}

/* ============================================================================
 * TAU_SOBOLEV ARRAY BUILDER FOR MACRO-ATOM
 * ============================================================================
 * The macro-atom code needs tau_sobolev as an array indexed by global line_id.
 * This function builds that array from the shell's active_lines, which contains
 * tau only for lines with significant opacity.
 *
 * Lines not in active_lines are set to tau=0 (optically thin, β=1).
 */

static double *g_tau_sobolev_cache = NULL;
static int64_t g_tau_sobolev_cache_size = 0;
static int     g_tau_sobolev_cache_shell = -1;

/**
 * Build tau_sobolev array for macro-atom from shell's active_lines.
 *
 * @param shell      ShellState with active_lines
 * @param n_lines    Total number of lines in atomic data
 * @param shell_idx  Shell index (for caching)
 * @return           Pointer to tau array (cached, do not free)
 */
static double *build_tau_sobolev_array(
    const ShellState *shell,
    int64_t n_lines,
    int shell_idx)
{
    /* Return cached array if same shell */
    if (g_tau_sobolev_cache != NULL &&
        g_tau_sobolev_cache_shell == shell_idx &&
        g_tau_sobolev_cache_size == n_lines) {
        return g_tau_sobolev_cache;
    }

    /* Allocate or reallocate if needed */
    if (g_tau_sobolev_cache == NULL || g_tau_sobolev_cache_size < n_lines) {
        free(g_tau_sobolev_cache);
        g_tau_sobolev_cache = (double *)calloc(n_lines, sizeof(double));
        g_tau_sobolev_cache_size = n_lines;
    }

    /* Clear to zero (optically thin default) */
    memset(g_tau_sobolev_cache, 0, n_lines * sizeof(double));

    /* Fill in tau values from active_lines */
    for (int64_t i = 0; i < shell->n_active_lines; i++) {
        int64_t line_idx = shell->active_lines[i].line_idx;
        if (line_idx >= 0 && line_idx < n_lines) {
            g_tau_sobolev_cache[line_idx] = shell->active_lines[i].tau_sobolev;
        }
    }

    g_tau_sobolev_cache_shell = shell_idx;
    return g_tau_sobolev_cache;
}

/**
 * Free the tau_sobolev cache (call at end of simulation)
 */
static void free_tau_sobolev_cache(void) {
    free(g_tau_sobolev_cache);
    g_tau_sobolev_cache = NULL;
    g_tau_sobolev_cache_size = 0;
    g_tau_sobolev_cache_shell = -1;
}

/* ============================================================================
 * SIMULATION CONFIGURATION
 * ============================================================================ */

typedef struct {
    /* Input files */
    char atomic_file[256];

    /* Model parameters */
    int    n_shells;
    double t_exp;               /* Expansion time [s] */
    double v_inner;             /* Inner (photospheric) velocity [cm/s] */
    double v_outer;             /* Outer velocity [cm/s] */
    double T_inner;             /* Inner temperature [K] */
    double T_outer;             /* Outer temperature [K] */
    double rho_inner;           /* Inner density [g/cm³] */
    double rho_profile;         /* Density power law exponent */

    /* Abundances */
    double X_H;
    double X_He;
    int    stratified;          /* Use velocity-stratified abundances */

    /* Abundance scaling factors (for optimizer control) */
    double Si_scale;            /* Si mass fraction multiplier */
    double Fe_scale;            /* Fe mass fraction multiplier */
    double Ca_scale;            /* Ca mass fraction multiplier */
    double S_scale;             /* S mass fraction multiplier */

    /* NLTE excitation correction */
    double t_alpha;             /* T_exc / T_eff ratio (1.0 = LTE) */

    /* Temperature iteration (radiative equilibrium) */
    int    enable_t_iteration;  /* Enable TARDIS-style T iteration */
    int    t_iter_max;          /* Maximum iterations (default: 12) */
    double t_converge;          /* Convergence threshold (default: 0.05 = 5%) */
    double t_damping;           /* Damping factor (default: 0.7) */
    int    t_hold;              /* Hold iterations: skip convergence check for first N */
    double t_fraction;          /* Luminosity fraction for T_inner update */

    /* Task Order #30: Multi-Target Portability Parameters */
    double T_boundary;          /* Planck weighting temperature [K] */
    double epsilon_default;     /* Default thermalization probability */
    double epsilon_ir;          /* IR (λ>7000Å) thermalization probability */
    double fe_blue_scale;       /* Fe-group opacity scale in blue (3500-4500Å) */

    /* Simulation parameters */
    int    n_packets;
    double nu_min;              /* Minimum frequency [Hz] */
    double nu_max;              /* Maximum frequency [Hz] */

    /* Output */
    char   output_file[256];
    int    n_bins;

} SimConfig;

/* Global config pointer for opacity functions */
static SimConfig *g_cfg = NULL;

/* Default configuration - optimized for SN 2011fe at B-maximum */
static void config_set_defaults(SimConfig *cfg)
{
    strcpy(cfg->atomic_file, "atomic/kurucz_cd23_chianti_H_He.h5");

    cfg->n_shells = 30;             /* More shells for better resolution */
    cfg->t_exp = 86400.0 * 19.0;    /* ~19 days (rise time for SN 2011fe) */

    /*
     * Velocity structure for SN 2011fe at maximum light:
     *   - Photosphere at ~10,500 km/s (Pereira et al. 2013)
     *   - Si II 6355 forms at 10,000-11,000 km/s
     *   - Outer ejecta extends to ~25,000 km/s
     *
     * PRECISION FIT (Task Order #12):
     *   v_inner = 10,000 km/s to match Si II velocity at λ=6139 Å
     */
    cfg->v_inner = 1.0e9;           /* 10,000 km/s (photospheric velocity) */
    cfg->v_outer = 2.5e9;           /* 25,000 km/s (outer boundary) */

    /*
     * Temperature structure:
     *   - T_eff ~13,500 K for SN 2011fe (blue-shifted to fix red excess)
     *   - Bump from 12,500 K to shift spectral peak blueward
     *
     * PRECISION FIT (Task Order #12):
     *   T_inner = 13,500 K to correct continuum slope
     */
    cfg->T_inner = 13500.0;         /* 13,500 K - boosted for blue continuum */
    cfg->T_outer = 5500.0;          /* 5,500 K - outer boundary */

    /*
     * Density structure:
     *   - Central density ~1e-13 g/cm³
     *   - Power law ρ ∝ v^-7 (steeper than r^-3 for W7)
     */
    cfg->rho_inner = 8e-14;         /* 8×10^-14 g/cm³ */
    cfg->rho_profile = -7.0;        /* ρ ∝ v^-7 (Mazzali convention) */

    cfg->X_H = 0.0;                 /* Type Ia: no hydrogen */
    cfg->X_He = 0.0;                /* No helium */
    cfg->stratified = 1;            /* Use stratified abundances by default */

    /* Abundance scaling factors (default: no scaling) */
    cfg->Si_scale = 1.0;
    cfg->Fe_scale = 1.0;
    cfg->Ca_scale = 1.0;
    cfg->S_scale = 1.0;

    /* Excitation temperature ratio (NLTE correction) */
    /* t_alpha = T_exc / T_eff, default 1.0 = LTE */
    cfg->t_alpha = 1.0;

    /* Temperature iteration (radiative equilibrium) - ENABLED by default (TARDIS-style) */
    cfg->enable_t_iteration = 1;  /* Enable by default, disable via LUMINA_T_ITERATION=0 */
    cfg->t_iter_max = 12;         /* TARDIS default: 12 iterations */
    cfg->t_converge = 0.05;       /* 5% convergence threshold (TARDIS default) */
    cfg->t_damping = 0.7;         /* TARDIS damping_constant default */
    cfg->t_hold = 3;              /* Hold iterations: skip convergence check for first N */
    cfg->t_fraction = 0.67;       /* Target escape fraction (~67% with current opacity) */

    /* Task Order #30 v2: Physical Engine Defaults (with continuum opacity)
     *
     * With continuum opacity ENABLED, we can use PHYSICAL values:
     *   - T_boundary = 13,000 K (no 60,000 K hack needed)
     *   - IR thermalization = 0.50 (continuum handles IR absorption)
     *
     * The old "hack" values (T=60000K, eps_ir=0.95) are now deprecated
     * but available via LUMINA_LEGACY_MODE=1 environment variable.
     */
    cfg->T_boundary = 13000.0;      /* PHYSICAL value (continuum handles rest) */
    cfg->epsilon_default = 0.35;    /* Physical thermalization */
    cfg->epsilon_ir = 0.50;         /* Physical (bf/ff absorbs IR) */
    cfg->fe_blue_scale = 0.50;      /* Moderate Fe-group reduction */

    cfg->n_packets = 100000;        /* 100k packets for good S/N */

    /* Optical wavelength range: 3000-10000 Å */
    cfg->nu_min = wavelength_angstrom_to_nu(10000.0);
    cfg->nu_max = wavelength_angstrom_to_nu(3000.0);

    strcpy(cfg->output_file, "spectrum_sn2011fe.csv");
    cfg->n_bins = 500;
}

/* ============================================================================
 * SIMPLE SPECTRUM ACCUMULATOR (for this test)
 * ============================================================================ */

typedef struct {
    int     n_bins;
    double  nu_min;
    double  nu_max;
    double  d_nu;

    double *luminosity;          /* Accumulated L_nu [erg/s/Hz] */
    double *luminosity_lumina;   /* LUMINA-rotated L_nu */
    int64_t *counts;             /* Packet counts per bin */
    int64_t *counts_lumina;

    /* Statistics */
    int64_t n_escaped;
    int64_t n_absorbed;
    int64_t n_scattered;
    double  total_energy;

} SimpleSpectrum;

static void simple_spectrum_init(SimpleSpectrum *spec, int n_bins, double nu_min, double nu_max)
{
    spec->n_bins = n_bins;
    spec->nu_min = nu_min;
    spec->nu_max = nu_max;
    spec->d_nu = (nu_max - nu_min) / n_bins;

    spec->luminosity = (double *)calloc(n_bins, sizeof(double));
    spec->luminosity_lumina = (double *)calloc(n_bins, sizeof(double));
    spec->counts = (int64_t *)calloc(n_bins, sizeof(int64_t));
    spec->counts_lumina = (int64_t *)calloc(n_bins, sizeof(int64_t));

    spec->n_escaped = 0;
    spec->n_absorbed = 0;
    spec->n_scattered = 0;
    spec->total_energy = 0.0;
}

static void simple_spectrum_free(SimpleSpectrum *spec)
{
    free(spec->luminosity);
    free(spec->luminosity_lumina);
    free(spec->counts);
    free(spec->counts_lumina);
}

/* Task Order #28: Weight tracking for energy conservation audit */
static double g_weight_sum = 0.0;
static double g_energy_weighted_sum = 0.0;
static double g_energy_unweighted_sum = 0.0;
static int64_t g_lumina_packet_count = 0;

static void report_weight_audit(void) {
    if (g_lumina_packet_count > 0) {
        double mean_weight = g_weight_sum / g_lumina_packet_count;
        double energy_ratio = g_energy_weighted_sum / (g_energy_unweighted_sum + 1e-30);
        printf("\n[TASK ORDER #28: WEIGHT AUDIT]\n");
        printf("  N_LUMINA_packets: %ld\n", (long)g_lumina_packet_count);
        printf("  Mean weight:      %.6f\n", mean_weight);
        printf("  Energy ratio:     %.6f (weighted/unweighted)\n", energy_ratio);
        if (mean_weight < 0.9 || mean_weight > 1.1) {
            printf("  *** WARNING: Mean weight deviates from 1.0! ***\n");
            printf("  *** This indicates energy non-conservation! ***\n");
        }
    }
}

static void simple_spectrum_add(SimpleSpectrum *spec, double nu, double energy, int is_lumina)
{
    if (nu < spec->nu_min || nu > spec->nu_max) return;

    int bin = (int)((nu - spec->nu_min) / spec->d_nu);
    if (bin < 0) bin = 0;
    if (bin >= spec->n_bins) bin = spec->n_bins - 1;

    if (is_lumina) {
        spec->luminosity_lumina[bin] += energy;
        spec->counts_lumina[bin]++;
    } else {
        spec->luminosity[bin] += energy;
        spec->counts[bin]++;
    }
}

static void simple_spectrum_write_csv(const SimpleSpectrum *spec, const char *filename, double t_exp)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", filename);
        return;
    }

    fprintf(fp, "# LUMINA-SN Integrated Spectrum\n");
    fprintf(fp, "# t_exp = %.2f days\n", t_exp / 86400.0);
    fprintf(fp, "# n_escaped = %ld, n_absorbed = %ld\n",
            (long)spec->n_escaped, (long)spec->n_absorbed);
    fprintf(fp, "wavelength_A,frequency_Hz,L_nu_standard,L_nu_lumina,counts_std,counts_lumina\n");

    for (int i = 0; i < spec->n_bins; i++) {
        double nu = spec->nu_min + (i + 0.5) * spec->d_nu;
        double wl_A = nu_to_wavelength_angstrom(nu);

        fprintf(fp, "%.4f,%.6e,%.6e,%.6e,%ld,%ld\n",
                wl_A, nu,
                spec->luminosity[i],
                spec->luminosity_lumina[i],
                (long)spec->counts[i],
                (long)spec->counts_lumina[i]);
    }

    fclose(fp);
    printf("[SPECTRUM] Written to %s\n", filename);
}

/* ============================================================================
 * PLANCK FREQUENCY SAMPLING (for continuum re-emission)
 * ============================================================================ */

/**
 * Sample a frequency from the Planck distribution B_ν(T)
 *
 * Uses rejection sampling: sample uniformly in (nu_min, nu_max),
 * accept with probability proportional to B_ν(T).
 *
 * @param T       Temperature [K]
 * @param nu_min  Minimum frequency [Hz]
 * @param nu_max  Maximum frequency [Hz]
 * @return Sampled frequency [Hz]
 */
static double sample_planck_frequency(double T, double nu_min, double nu_max)
{
    /*
     * Planck function: B_ν(T) = (2hν³/c²) / (exp(hν/kT) - 1)
     *
     * We use rejection sampling:
     *   1. Sample ν uniformly in [nu_min, nu_max]
     *   2. Accept with probability B_ν / B_max
     *
     * B_max occurs at ν_peak ≈ 2.82 kT/h (Wien's law)
     */
    const double h = 6.62607015e-27;   /* Planck constant [erg·s] */
    const double k = 1.380649e-16;     /* Boltzmann constant [erg/K] */
    const double c = 2.99792458e10;    /* Speed of light [cm/s] */

    /* Wien peak frequency */
    double nu_peak = 2.821 * k * T / h;

    /* Clamp nu_peak to our range */
    if (nu_peak < nu_min) nu_peak = nu_min;
    if (nu_peak > nu_max) nu_peak = nu_max;

    /* Calculate B at peak for normalization */
    double x_peak = h * nu_peak / (k * T);
    double B_peak = (2.0 * h * nu_peak * nu_peak * nu_peak / (c * c)) /
                    (exp(x_peak) - 1.0 + 1e-30);

    /* Rejection sampling */
    int max_attempts = 1000;
    for (int i = 0; i < max_attempts; i++) {
        /* Sample uniform frequency */
        double nu = nu_min + drand48() * (nu_max - nu_min);

        /* Calculate Planck function at this frequency */
        double x = h * nu / (k * T);
        double B_nu;
        if (x > 30.0) {
            /* Wien approximation to avoid overflow */
            B_nu = (2.0 * h * nu * nu * nu / (c * c)) * exp(-x);
        } else {
            B_nu = (2.0 * h * nu * nu * nu / (c * c)) / (exp(x) - 1.0 + 1e-30);
        }

        /* Accept/reject */
        if (drand48() * B_peak < B_nu) {
            return nu;
        }
    }

    /* Fallback: return peak frequency */
    return nu_peak;
}

/* ============================================================================
 * PACKET TRANSPORT WITH REALISTIC OPACITIES
 * ============================================================================ */

typedef struct {
    double r;
    double mu;
    double nu;
    double energy;
    int    shell;
    int    status;  /* 0=in_process, 1=escaped, 2=absorbed */
    int    n_interactions;
    int64_t next_line_id;  /* TARDIS-style: track position in line list */
    double  tau_event;     /* Random τ threshold for next interaction */
    double  tau_accumulated; /* Cumulative τ across line searches */
} TransportPkt;

static void init_transport_pkt(TransportPkt *pkt, double r_inner, double nu, double energy)
{
    /* Start slightly inside the first shell to avoid boundary issues */
    pkt->r = r_inner * 1.001;
    /* Only emit outward (positive mu) from inner boundary */
    pkt->mu = sqrt(drand48());  /* Limb darkening: I ∝ μ, always positive */
    if (pkt->mu < 0.01) pkt->mu = 0.01;  /* Ensure outward motion */
    pkt->nu = nu;
    pkt->energy = energy;
    pkt->shell = 0;
    pkt->status = 0;  /* in process */
    pkt->n_interactions = 0;
    pkt->next_line_id = 0;  /* Start from first line */
    pkt->tau_event = -log(drand48() + 1e-30);  /* Random τ threshold */
    pkt->tau_accumulated = 0.0;  /* No τ accumulated yet */
}

static int find_shell_idx(const SimulationState *state, double r)
{
    for (int i = 0; i < state->n_shells; i++) {
        if (r >= state->shells[i].r_inner && r < state->shells[i].r_outer) {
            return i;
        }
    }
    return -1;  /* Outside grid */
}

static void transport_single_packet(TransportPkt *pkt,
                                     const SimulationState *state,
                                     SimpleSpectrum *spec,
                                     int max_interactions)
{
    int step_count = 0;
    int max_steps = 10000;  /* Prevent infinite loops */

    while (pkt->status == 0 && pkt->n_interactions < max_interactions && step_count < max_steps) {
        step_count++;
        /* Find current shell */
        int shell_id = find_shell_idx(state, pkt->r);

        if (shell_id < 0) {
            /* Escaped or absorbed */
            if (pkt->r >= state->shells[state->n_shells - 1].r_outer) {
                pkt->status = 1;  /* Escaped */

                /* Standard escape: only if going outward in observer direction */
                if (pkt->mu > 0.99) {
                    simple_spectrum_add(spec, pkt->nu, pkt->energy, 0);
                }

                /* LUMINA rotation: all packets contribute */
                ObserverConfig obs;
                memset(&obs, 0, sizeof(obs));
                obs.mu_observer = 1.0;
                obs.time_explosion = state->t_explosion;

                RotatedPacket rotated;
                lumina_rotate_packet(pkt->r, pkt->mu, pkt->nu, pkt->energy,
                                     &obs, &rotated);

                /* Task Order #28: Track weight statistics */
                g_weight_sum += rotated.weight;
                g_energy_weighted_sum += pkt->energy * rotated.weight;
                g_energy_unweighted_sum += pkt->energy;
                g_lumina_packet_count++;

                simple_spectrum_add(spec, rotated.nu_observer,
                                    pkt->energy * rotated.weight, 1);
            } else {
                /* Fell inside inner boundary - absorbed */
                pkt->status = 2;  /* Absorbed */
            }
            return;
        }

        const ShellState *shell = &state->shells[shell_id];

        /* Calculate distances */
        double d_boundary, d_electron, d_line, d_continuum;
        double tau_line = 0.0;
        int64_t line_idx = -1;

        /* Distance to shell boundary */
        int delta_shell = 0;
        d_boundary = calculate_distance_boundary(pkt->r, pkt->mu,
                                                  shell->r_inner, shell->r_outer,
                                                  &delta_shell);

        /* Distance to electron scattering */
        double tau_e = -log(drand48() + 1e-30);
        d_electron = tau_e / (shell->sigma_thomson_ne + 1e-30);

        /* Distance to line interaction - TARDIS-style cumulative τ */
        /* Get comoving frame frequency */
        double beta = pkt->r / (state->t_explosion * CONST_C);
        double doppler = 1.0 - beta * pkt->mu;
        double nu_cmf = pkt->nu * doppler;

        /* Maximum distance before other interactions occur */
        double d_max_search = (d_electron < d_boundary) ? d_electron : d_boundary;

        int64_t next_line_id_out;
        d_line = get_line_interaction_tardis_style(
            shell, nu_cmf, pkt->r, pkt->mu,
            state->t_explosion,
            pkt->next_line_id,   /* Start from where we left off */
            pkt->tau_event,      /* Random τ threshold */
            d_max_search,        /* Don't search beyond this distance */
            &line_idx, &tau_line, &next_line_id_out);

        /* Debug: track line selection statistics */
        g_line_stats.n_line_checks++;
        if (line_idx < 0) g_line_stats.n_line_no_idx++;
        else if (tau_line <= 0.1) g_line_stats.n_line_low_tau++;

        /*
         * NEW: Distance to continuum absorption (bf + ff)
         * Task Order #30 v2: This is the KEY physics that replaces the 60,000K hack!
         *
         * Continuum opacity provides natural IR absorption through:
         *   - Bound-free (photoionization) in UV/blue
         *   - Free-free (bremsstrahlung) across all wavelengths, especially IR
         */
        d_continuum = 1e99;  /* Default: no continuum interaction */
        if (g_physics_overrides.enable_continuum_opacity) {
            double kappa_cont = calculate_continuum_opacity(nu_cmf, shell);
            if (kappa_cont > 1e-30) {
                double tau_cont = -log(drand48() + 1e-30);
                d_continuum = tau_cont / kappa_cont;
            }
        }

        /* Take the minimum distance */
        double d_min = d_boundary;
        int interaction_type = 0;  /* 0=boundary, 1=electron, 2=line, 3=continuum */

        if (d_electron < d_min) {
            d_min = d_electron;
            interaction_type = 1;
        }

        /* Line interaction: cumulative τ approach already decided. */
        if (line_idx >= 0 && d_line < d_min && tau_line > 0.001) {
            d_min = d_line;
            interaction_type = 2;
            g_line_stats.n_line_selected++;
        }

        /* Continuum absorption - KEY for removing IR hack */
        if (d_continuum < d_min) {
            d_min = d_continuum;
            interaction_type = 3;
        }

        /* Sanity check */
        if (d_min <= 0.0 || d_min > 1e50 || !isfinite(d_min)) {
            d_min = (shell->r_outer - pkt->r) * 0.1;  /* Small step */
            if (d_min <= 0.0) d_min = shell->r_outer * 0.001;
        }

        /* Move packet */
        double r_new = sqrt(pkt->r * pkt->r + d_min * d_min +
                            2.0 * pkt->r * d_min * pkt->mu);
        double mu_new = (pkt->r * pkt->mu + d_min) / r_new;

        pkt->r = r_new;
        pkt->mu = mu_new;

        /* Process interaction */
        switch (interaction_type) {
            case 0:  /* Boundary crossing */
                /* Reset line tracking when entering new shell */
                pkt->next_line_id = 0;
                pkt->tau_event = -log(drand48() + 1e-30);
                break;

            case 1:  /* Electron scattering */
                /* Isotropic scattering in comoving frame */
                pkt->mu = 2.0 * drand48() - 1.0;
                spec->n_scattered++;
                pkt->n_interactions++;
                /* After scattering, frequency changes - restart line search */
                pkt->next_line_id = 0;
                pkt->tau_event = -log(drand48() + 1e-30);
                break;

            case 2:  /* Line interaction (Sobolev approximation) */
                if (line_idx >= 0 && tau_line > 0.1) {
                    /*
                     * Sobolev line interaction:
                     *   P_interact = 1 - exp(-τ)
                     *
                     * For τ >> 1 (e.g., Si II with τ~10^5), P_interact ≈ 1.
                     *
                     * After interaction, two outcomes:
                     *   1. Pure scattering: packet re-emitted isotropically
                     *   2. Thermal destruction: packet absorbed (contributes to
                     *      diffuse emission at thermal wavelengths)
                     *
                     * For resonance lines like Si II, assume mostly scattering
                     * but with small destruction probability (epsilon ~ 0.01-0.1)
                     * to represent collisional de-excitation or branching.
                     */
                    double p_interact = 1.0 - exp(-tau_line);

                    /*
                     * Task Order #30: PhysicsOverrides-based thermalization
                     * Uses g_physics_overrides for wavelength-dependent thermalization.
                     */
                    double wavelength_A = CONST_C / pkt->nu * 1e8;  /* Convert to Angstrom */
                    double epsilon = (wavelength_A > g_physics_overrides.ir_wavelength_min)
                        ? g_physics_overrides.ir_thermalization_frac
                        : g_physics_overrides.base_thermalization_frac;

                    if (drand48() < p_interact) {
                        /* Track wavelength band statistics */
                        if (wavelength_A < 3000.0) {
                            g_line_stats.n_uv_interact++;
                        } else if (wavelength_A < 5000.0) {
                            g_line_stats.n_blue_interact++;
                        } else if (wavelength_A < 7000.0) {
                            g_line_stats.n_red_interact++;
                        } else {
                            g_line_stats.n_ir_interact++;
                        }

                        /* =============================================================
                         * LINE INTERACTION HANDLING (TARDIS-style)
                         * =============================================================
                         * Three modes matching TARDIS exactly:
                         *   0 = SCATTER:    Pure resonance scattering (same line freq)
                         *   1 = DOWNBRANCH: Simplified cascade (downward only)
                         *   2 = MACROATOM:  Full macro-atom cascade
                         */
                        int line_mode = g_physics_overrides.line_interaction_type;

                        if (line_mode == LINE_SCATTER) {
                            /* ============================================================
                             * SCATTER MODE: Pure resonance scattering (TARDIS SCATTER)
                             * ============================================================
                             * The packet is re-emitted at the SAME LINE FREQUENCY (in CMF).
                             * Only the Doppler shift changes due to new random direction.
                             *
                             * This is the simplest mode with NO frequency redistribution.
                             * TARDIS uses this in scatter mode: line_emission(next_line_id)
                             *
                             * Algorithm (matching TARDIS line_scatter_event + line_emission):
                             *   1. Get old Doppler factor (for energy conservation)
                             *   2. Draw new random direction μ_new
                             *   3. Get inverse Doppler factor for new direction
                             *   4. nu_lab = nu_line * inverse_doppler_factor
                             *   5. energy_lab = energy_cmf * inverse_doppler_factor
                             */
                            g_line_stats.n_scatter++;

                            const Line *line = &state->atomic_data->lines[line_idx];
                            double nu_line = line->nu;  /* Rest-frame line frequency */

                            /* Old Doppler factor (before scatter) */
                            double beta = pkt->r / (state->t_explosion * CONST_C);
                            double old_doppler = 1.0 - beta * pkt->mu;

                            /* New random direction (isotropic in CMF) */
                            double mu_cmf_new = 2.0 * drand48() - 1.0;

                            /* Transform to lab frame (angle aberration) */
                            double mu_lab_new = (mu_cmf_new + beta) / (1.0 + beta * mu_cmf_new);
                            pkt->mu = mu_lab_new;

                            /* Inverse Doppler factor for new direction */
                            double inv_doppler_new = 1.0 / (1.0 - beta * mu_lab_new);

                            /* Energy conservation: transform energy through CMF */
                            double energy_cmf = pkt->energy * old_doppler;
                            pkt->energy = energy_cmf * inv_doppler_new;

                            /* KEY: Emit at the SAME line frequency (in CMF), Doppler-shifted */
                            pkt->nu = nu_line * inv_doppler_new;

                            pkt->n_interactions++;
                            pkt->next_line_id = line_idx + 1;  /* Move past this line */
                            pkt->tau_event = -log(drand48() + 1e-30);

                        } else if (line_mode == LINE_MACROATOM) {
                            /* ============================================================
                             * MACRO-ATOM MODE: Full multi-step cascade fluorescence
                             * ============================================================
                             * Uses macro_atom_process_line_interaction() which:
                             *   1. Initializes macro-atom state from absorbed line
                             *   2. Runs transition loop (up to 100 jumps)
                             *   3. Selects emission line based on transition probabilities
                             *   4. Returns new direction and frequency
                             */
                            g_line_stats.n_macro_atom_calls++;

                            RPacket rpkt;
                            rpkt.r = pkt->r;
                            rpkt.mu = pkt->mu;
                            rpkt.nu = pkt->nu;
                            rpkt.energy = pkt->energy;
                            rpkt.status = PACKET_IN_PROCESS;
                            rpkt.last_line_interaction_in_id = line_idx;
                            rpkt.last_line_interaction_out_id = -1;
                            rng_init(&rpkt.rng_state, (uint64_t)(time(NULL) ^ (pkt->n_interactions * 12345)));

                            /* Calculate dilution factor for this shell */
                            double W_shell = calculate_dilution_factor(
                                0.5 * (shell->r_inner + shell->r_outer),
                                g_physics_overrides.R_photosphere);

                            /* Build tau_sobolev array for macro-atom β factor */
                            double *tau_arr = build_tau_sobolev_array(
                                shell, state->atomic_data->n_lines, shell_id);

                            int survives = macro_atom_process_line_interaction(
                                &rpkt, state->atomic_data, line_idx,
                                shell->plasma.T, shell->plasma.n_e,
                                W_shell, tau_arr,
                                state->t_explosion);

                            if (survives) {
                                g_line_stats.n_macro_emit++;
                                pkt->mu = rpkt.mu;
                                pkt->nu = rpkt.nu;
                                pkt->n_interactions++;

                                pkt->next_line_id = next_line_id_out;
                                pkt->tau_event = -log(drand48() + 1e-30);

                                /* Track cascade statistics */
                                double new_wl = CONST_C / pkt->nu * 1e8;
                                if (new_wl > wavelength_A * 1.05) {
                                    g_line_stats.n_thermalize++;
                                }

                                static int macro_debug_count = 0;
                                if (macro_debug_count < 5 && fabs(new_wl - wavelength_A) > 100.0) {
                                    printf("  [MACRO-ATOM] λ=%.0f → %.0f Å (cascade)\n",
                                           wavelength_A, new_wl);
                                    macro_debug_count++;
                                }
                            } else {
                                g_line_stats.n_macro_absorb++;
                                pkt->status = 2;
                                pkt->n_interactions++;
                                g_line_stats.n_ir_absorbed++;
                                return;
                            }
                        } else {
                            /* DOWNBRANCH MODE: Simplified cascade (downward only)
                             * For now, treat same as SCATTER - TODO: implement downbranch */
                            g_line_stats.n_scatter++;

                            const Line *line = &state->atomic_data->lines[line_idx];
                            double nu_line = line->nu;
                            double beta = pkt->r / (state->t_explosion * CONST_C);
                            double old_doppler = 1.0 - beta * pkt->mu;
                            double mu_cmf_new = 2.0 * drand48() - 1.0;
                            double mu_lab_new = (mu_cmf_new + beta) / (1.0 + beta * mu_cmf_new);
                            pkt->mu = mu_lab_new;
                            double inv_doppler_new = 1.0 / (1.0 - beta * mu_lab_new);
                            double energy_cmf = pkt->energy * old_doppler;
                            pkt->energy = energy_cmf * inv_doppler_new;
                            pkt->nu = nu_line * inv_doppler_new;
                            pkt->n_interactions++;
                            pkt->next_line_id = line_idx + 1;
                            pkt->tau_event = -log(drand48() + 1e-30);
                        }

                        /* Track wavelength shift for diagnostics */
                        double new_wl = CONST_C / pkt->nu * 1e8;
                        g_line_stats.sum_input_wl += wavelength_A;
                        g_line_stats.sum_output_wl += new_wl;
                        g_line_stats.n_freq_samples++;

                    } /* End of line interaction if block */
                } /* End of tau > 0.1 if block */
                break;  /* End of case 2 (line interaction) */

            /* Placeholder for case 3 and higher - handle outside the legacy code path */
            case 3:  /* Continuum interaction */
            case 4:  /* Other */
            default:
                break;
        } /* End of switch */
    } /* End of while (step < MAX_STEPS) */
}

/**
 * Transport a single packet with MC estimator updates for temperature iteration.
 *
 * This version calls mc_estimators_update() for each path segment to accumulate
 * the J-estimator needed for radiative equilibrium temperature updates.
 *
 * @param pkt            Packet to transport
 * @param state          Simulation state
 * @param spec           Spectrum accumulator
 * @param est            MC estimators (can be NULL to skip updates)
 * @param max_interactions Maximum interactions allowed
 */
static void transport_single_packet_with_estimators(TransportPkt *pkt,
                                                      const SimulationState *state,
                                                      SimpleSpectrum *spec,
                                                      MCEstimators *est,
                                                      VPacketCollection *vpacket_coll,
                                                      int max_interactions)
{
    int step_count = 0;
    int max_steps = 10000;  /* Prevent infinite loops */

    /* TARDIS-style: Spawn virtual packets at INITIAL position (before transport) */
    if (g_physics_overrides.enable_virtual_packets && vpacket_coll) {
        int shell_id = find_shell_idx(state, pkt->r);
        if (shell_id >= 0) {
            spawn_vpacket_volley(pkt->r, pkt->mu, pkt->nu, pkt->energy,
                                  shell_id, pkt->next_line_id,
                                  state, vpacket_coll,
                                  g_physics_overrides.n_virtual_packets);
        }
    }

    while (pkt->status == 0 && pkt->n_interactions < max_interactions && step_count < max_steps) {
        step_count++;
        /* Find current shell */
        int shell_id = find_shell_idx(state, pkt->r);

        if (shell_id < 0) {
            /* Escaped or absorbed */
            if (pkt->r >= state->shells[state->n_shells - 1].r_outer) {
                pkt->status = 1;  /* Escaped */

                /* Standard escape: only if going outward in observer direction */
                if (pkt->mu > 0.99) {
                    simple_spectrum_add(spec, pkt->nu, pkt->energy, 0);
                }

                /* LUMINA rotation: all packets contribute */
                ObserverConfig obs;
                memset(&obs, 0, sizeof(obs));
                obs.mu_observer = 1.0;
                obs.time_explosion = state->t_explosion;

                RotatedPacket rotated;
                lumina_rotate_packet(pkt->r, pkt->mu, pkt->nu, pkt->energy,
                                     &obs, &rotated);

                /* Task Order #28: Track weight statistics */
                g_weight_sum += rotated.weight;
                g_energy_weighted_sum += pkt->energy * rotated.weight;
                g_energy_unweighted_sum += pkt->energy;
                g_lumina_packet_count++;

                simple_spectrum_add(spec, rotated.nu_observer,
                                    pkt->energy * rotated.weight, 1);
            } else {
                /* Fell inside inner boundary - absorbed */
                pkt->status = 2;  /* Absorbed */
            }
            return;
        }

        const ShellState *shell = &state->shells[shell_id];

        /* ================================================================
         * TARDIS-STYLE TRACE_PACKET
         * Distance comparison happens INSIDE the line loop for efficiency.
         * Exit immediately when boundary/continuum is closer than current line.
         * ================================================================ */

        /* Calculate distance to boundary */
        int delta_shell = 0;
        double d_boundary = calculate_distance_boundary(pkt->r, pkt->mu,
                                                         shell->r_inner, shell->r_outer,
                                                         &delta_shell);

        /* Sample tau_event for this trace (TARDIS samples once per trace) */
        double tau_event = -log(drand48() + 1e-30);
        double tau_trace_combined = 0.0;

        /* Calculate comoving frequency */
        double beta = pkt->r / (state->t_explosion * CONST_C);
        double doppler = 1.0 - beta * pkt->mu;
        double nu_cmf = pkt->nu * doppler;

        /* Distance to electron scattering (separate from continuum in TARDIS) */
        double tau_electron = -log(drand48() + 1e-30);
        double d_electron = tau_electron / (shell->sigma_thomson_ne + 1e-30);

        /* Distance to continuum (only if enabled) */
        double chi_continuum = 0.0;
        double d_continuum = 1e99;
        if (g_physics_overrides.enable_continuum_opacity) {
            chi_continuum = calculate_continuum_opacity(nu_cmf, shell);
            if (chi_continuum > 1e-30) {
                double tau_cont = -log(drand48() + 1e-30);
                d_continuum = tau_cont / chi_continuum;
            }
        }

        /* Line loop variables */
        const ActiveLine *lines = shell->active_lines;
        int64_t n_lines = shell->n_active_lines;
        int64_t start_line_id = pkt->next_line_id;

        /* Result variables */
        double d_min = d_boundary;
        int interaction_type = 0;  /* 0=boundary, 1=electron/continuum, 2=line */
        int64_t line_idx = -1;
        double tau_line = 0.0;

        g_line_stats.n_line_checks++;

        /* === TARDIS-STYLE LINE LOOP === */
        /* Loop through lines one at a time, comparing distances each step */
        for (int64_t cur_line = start_line_id; cur_line < n_lines; cur_line++) {
            double nu_line = lines[cur_line].nu;
            double tau_this_line = lines[cur_line].tau_sobolev;

            /* Calculate distance to this line FIRST */
            double nu_diff = nu_cmf - nu_line;
            double d_line;

            if (fabs(nu_diff / pkt->nu) < 1e-14) {
                nu_diff = 0.0;  /* Numerical threshold */
            }

            if (nu_diff >= 0) {
                d_line = (nu_diff / pkt->nu) * CONST_C * state->t_explosion;
            } else {
                /* Line is above current frequency - skip WITHOUT adding tau */
                continue;
            }

            /* Accumulate tau ONLY for lines at or below current frequency
             * This is the TARDIS algorithm: only count tau for lines the photon passes */
            tau_trace_combined += tau_this_line;

            /* In TARDIS mode: only line tau matters (no continuum in tau calc) */
            double tau_total = tau_trace_combined;

            /* CRITICAL: Check if boundary/electron is closer than THIS line
             * Must check BEFORE testing tau for line interaction.
             * Use direct < comparison to avoid floating point equality issues. */

            /* If boundary is closer than this line, packet hits boundary first */
            if (d_boundary < d_line) {
                interaction_type = 0;
                d_min = d_boundary;
                pkt->next_line_id = cur_line;  /* Stay at this line */
                break;
            }

            /* If electron scattering is closer than this line */
            if (d_electron < d_line) {
                interaction_type = 1;
                d_min = d_electron;
                pkt->next_line_id = cur_line;
                break;
            }

            /* If continuum is closer than this line (only when enabled) */
            if (d_continuum < d_line && g_physics_overrides.enable_continuum_opacity) {
                interaction_type = 3;
                d_min = d_continuum;
                pkt->next_line_id = cur_line;
                break;
            }

            /* Check for line interaction - only if we didn't exit above */
            if (tau_total > tau_event) {
                /* LINE INTERACTION! */
                interaction_type = 2;
                d_min = d_line;
                line_idx = lines[cur_line].line_idx;
                tau_line = tau_this_line;
                pkt->next_line_id = cur_line;
                g_line_stats.n_line_selected++;
                break;
            }

            /* Stop if too far in frequency (efficiency) */
            if (nu_line < nu_cmf * 0.5) {
                /* No line interaction - check boundary vs electron vs continuum */
                d_min = d_boundary;
                interaction_type = 0;
                if (d_electron < d_min) {
                    d_min = d_electron;
                    interaction_type = 1;
                }
                if (d_continuum < d_min && g_physics_overrides.enable_continuum_opacity) {
                    d_min = d_continuum;
                    interaction_type = 3;
                }
                pkt->next_line_id = n_lines;
                break;
            }
        }

        /* Track statistics */
        if (line_idx < 0) g_line_stats.n_line_no_idx++;
        else if (tau_line <= 0.1) g_line_stats.n_line_low_tau++;

        /* Sanity check */
        if (d_min <= 0.0 || d_min > 1e50 || !isfinite(d_min)) {
            d_min = (shell->r_outer - pkt->r) * 0.1;  /* Small step */
            if (d_min <= 0.0) d_min = shell->r_outer * 0.001;
        }

        /* ================================================================
         * UPDATE MC ESTIMATORS (for temperature iteration)
         * ================================================================
         * The J-estimator accumulates energy × distance for each path segment.
         * This gives us the mean intensity in each shell after normalization.
         */
        if (est != NULL) {
            /* Compute comoving-frame energy for this path segment */
            double energy_cmf = pkt->energy * doppler;

            /* Update J-estimator and frequency-weighted estimator */
            mc_estimators_update(est, shell_id, energy_cmf, d_min, nu_cmf);
        }

        /* Move packet */
        double r_new = sqrt(pkt->r * pkt->r + d_min * d_min +
                            2.0 * pkt->r * d_min * pkt->mu);
        double mu_new = (pkt->r * pkt->mu + d_min) / r_new;

        pkt->r = r_new;
        pkt->mu = mu_new;

        /* Process interaction - TARDIS style */
        switch (interaction_type) {
            case 0:  /* Boundary crossing */
                /* In TARDIS, next_line_id is set inside the loop.
                 * When entering a NEW shell, reset to 0 (different line list).
                 * next_line_id was already set in the loop above. */
                /* Reset only when actually changing shells */
                if (delta_shell != 0) {
                    pkt->next_line_id = 0;  /* New shell = new line list */
                }
                break;

            case 1:  /* Electron/continuum scattering */
                /* Isotropic scattering in comoving frame */
                pkt->mu = 2.0 * drand48() - 1.0;
                spec->n_scattered++;
                pkt->n_interactions++;
                /* Reset line tracking after direction change */
                pkt->next_line_id = 0;

                /* TARDIS-style: Spawn virtual packets AFTER electron scattering */
                if (g_physics_overrides.enable_virtual_packets && vpacket_coll) {
                    spawn_vpacket_volley(pkt->r, pkt->mu, pkt->nu, pkt->energy,
                                          shell_id, pkt->next_line_id,
                                          state, vpacket_coll,
                                          g_physics_overrides.n_virtual_packets);
                }
                break;

            case 2:  /* Line interaction (Sobolev approximation) */
                /* TARDIS: cumulative tau already decided interaction - NO extra dice roll! */
                if (line_idx >= 0) {
                    double wavelength_A = CONST_C / pkt->nu * 1e8;

                    /* Track wavelength band statistics */
                    if (wavelength_A < 3000.0) {
                        g_line_stats.n_uv_interact++;
                    } else if (wavelength_A < 5000.0) {
                        g_line_stats.n_blue_interact++;
                    } else if (wavelength_A < 7000.0) {
                        g_line_stats.n_red_interact++;
                    } else {
                        g_line_stats.n_ir_interact++;
                    }

                    /* =============================================================
                     * LINE INTERACTION HANDLING (TARDIS-style)
                     * =============================================================
                     * Three modes matching TARDIS exactly:
                     *   0 = SCATTER:    Pure resonance scattering (same line freq)
                     *   1 = DOWNBRANCH: Simplified cascade (downward only)
                     *   2 = MACROATOM:  Full macro-atom cascade
                     */
                    int line_mode = g_physics_overrides.line_interaction_type;

                    if (line_mode == LINE_SCATTER) {
                        /* ============================================================
                         * SCATTER MODE: Pure resonance scattering (TARDIS SCATTER)
                         * ============================================================
                         * The packet is re-emitted at the SAME LINE FREQUENCY (in CMF).
                         * Only the Doppler shift changes due to new random direction.
                         */
                        g_line_stats.n_scatter++;

                        const Line *line = &state->atomic_data->lines[line_idx];
                        double nu_line = line->nu;  /* Rest-frame line frequency */

                        /* Old Doppler factor (before scatter) */
                        double beta = pkt->r / (state->t_explosion * CONST_C);
                        double old_doppler = 1.0 - beta * pkt->mu;

                        /* New random direction (isotropic in CMF) */
                        double mu_cmf_new = 2.0 * drand48() - 1.0;

                        /* Transform to lab frame (angle aberration) */
                        double mu_lab_new = (mu_cmf_new + beta) / (1.0 + beta * mu_cmf_new);
                        pkt->mu = mu_lab_new;

                        /* Inverse Doppler factor for new direction */
                        double inv_doppler_new = 1.0 / (1.0 - beta * mu_lab_new);

                        /* Energy conservation: transform energy through CMF */
                        double energy_cmf = pkt->energy * old_doppler;
                        pkt->energy = energy_cmf * inv_doppler_new;

                        /* KEY: Emit at the SAME line frequency (in CMF), Doppler-shifted */
                        pkt->nu = nu_line * inv_doppler_new;

                        pkt->n_interactions++;
                        pkt->next_line_id = line_idx + 1;  /* Move past this line */

                    } else if (line_mode == LINE_MACROATOM) {
                        /* ============================================================
                         * MACRO-ATOM MODE: Full multi-step cascade fluorescence
                         * ============================================================ */
                        g_line_stats.n_macro_atom_calls++;

                        RPacket rpkt;
                        rpkt.r = pkt->r;
                        rpkt.mu = pkt->mu;
                        rpkt.nu = pkt->nu;
                        rpkt.energy = pkt->energy;
                        rpkt.status = PACKET_IN_PROCESS;
                        rpkt.last_line_interaction_in_id = line_idx;
                        rpkt.last_line_interaction_out_id = -1;
                        rng_init(&rpkt.rng_state, (uint64_t)(time(NULL) ^ (pkt->n_interactions * 12345)));

                        /* Calculate dilution factor for this shell */
                        double W_shell = calculate_dilution_factor(
                            0.5 * (shell->r_inner + shell->r_outer),
                            g_physics_overrides.R_photosphere);

                        /* Build tau_sobolev array for macro-atom β factor */
                        double *tau_arr = build_tau_sobolev_array(
                            shell, state->atomic_data->n_lines, shell_id);

                        int survives = macro_atom_process_line_interaction(
                            &rpkt, state->atomic_data, line_idx,
                            shell->plasma.T, shell->plasma.n_e,
                            W_shell, tau_arr,
                            state->t_explosion);

                        if (survives) {
                            g_line_stats.n_macro_emit++;
                            pkt->mu = rpkt.mu;
                            pkt->nu = rpkt.nu;
                            pkt->n_interactions++;

                            /* Track cascade statistics */
                            double new_wl = CONST_C / pkt->nu * 1e8;
                            if (new_wl > wavelength_A * 1.05) {
                                g_line_stats.n_thermalize++;
                            }
                            pkt->next_line_id = 0;
                        } else {
                            g_line_stats.n_macro_absorb++;
                            pkt->status = 2;
                            pkt->n_interactions++;
                            g_line_stats.n_ir_absorbed++;
                            return;
                        }
                    } else {
                        /* DOWNBRANCH MODE: Treat same as SCATTER for now */
                        g_line_stats.n_scatter++;

                        const Line *line = &state->atomic_data->lines[line_idx];
                        double nu_line = line->nu;
                        double beta = pkt->r / (state->t_explosion * CONST_C);
                        double old_doppler = 1.0 - beta * pkt->mu;
                        double mu_cmf_new = 2.0 * drand48() - 1.0;
                        double mu_lab_new = (mu_cmf_new + beta) / (1.0 + beta * mu_cmf_new);
                        pkt->mu = mu_lab_new;
                        double inv_doppler_new = 1.0 / (1.0 - beta * mu_lab_new);
                        double energy_cmf = pkt->energy * old_doppler;
                        pkt->energy = energy_cmf * inv_doppler_new;
                        pkt->nu = nu_line * inv_doppler_new;
                        pkt->n_interactions++;
                        pkt->next_line_id = line_idx + 1;
                    }

                    /* Track wavelength shift for diagnostics */
                    double new_wl = CONST_C / pkt->nu * 1e8;
                    g_line_stats.sum_input_wl += wavelength_A;
                    g_line_stats.sum_output_wl += new_wl;
                    g_line_stats.n_freq_samples++;

                    /* TARDIS-style: Spawn virtual packets AFTER line interaction */
                    if (g_physics_overrides.enable_virtual_packets && vpacket_coll) {
                        spawn_vpacket_volley(pkt->r, pkt->mu, pkt->nu, pkt->energy,
                                              shell_id, pkt->next_line_id,
                                              state, vpacket_coll,
                                              g_physics_overrides.n_virtual_packets);
                    }
                } /* End if (line_idx >= 0) */
                break;

            case 3:  /* Continuum absorption */
                {
                    double wavelength_A_cont = CONST_C / pkt->nu * 1e8;
                    double T_emit;

                    if (wavelength_A_cont < 5000.0) {
                        T_emit = g_physics_overrides.t_boundary;
                    } else {
                        T_emit = shell->plasma.T;
                        if (drand48() < 0.3) {
                            pkt->status = 2;
                            pkt->n_interactions++;
                            return;
                        }
                    }

                    double nu_new = sample_planck_frequency(T_emit,
                        wavelength_angstrom_to_nu(10000.0),
                        wavelength_angstrom_to_nu(3000.0));
                    pkt->nu = nu_new;
                    pkt->mu = 2.0 * drand48() - 1.0;
                    pkt->n_interactions++;
                    spec->n_scattered++;

                    /* TARDIS-style: Spawn virtual packets AFTER continuum interaction */
                    if (g_physics_overrides.enable_virtual_packets && vpacket_coll) {
                        spawn_vpacket_volley(pkt->r, pkt->mu, pkt->nu, pkt->energy,
                                              shell_id, pkt->next_line_id,
                                              state, vpacket_coll,
                                              g_physics_overrides.n_virtual_packets);
                    }
                }
                break;

            default:
                break;
        }
    }

    if (pkt->n_interactions >= max_interactions || step_count >= max_steps) {
        pkt->status = 2;
    }
}

/* ============================================================================
 * MAIN SIMULATION
 * ============================================================================ */

static void run_simulation(const SimConfig *cfg, const AtomicData *atomic)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║       LUMINA-SN INTEGRATED PLASMA-TRANSPORT SIMULATION        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    /*
     * DIAGNOSTIC: Print received parameters to verify optimizer injection
     * If these are identical across trials, the C engine is not receiving
     * the environment variables correctly.
     */
    printf("[PARAM CHECK] Received simulation parameters:\n");
    printf("  T_eff:        %.0f K\n", cfg->T_inner);
    printf("  v_inner:      %.0f km/s\n", cfg->v_inner / 1e5);
    printf("  rho_exponent: %.1f\n", -cfg->rho_profile);
    printf("  t_alpha:      %.3f\n", cfg->t_alpha);
    printf("  Si_scale:     %.3f\n", cfg->Si_scale);
    printf("  Fe_scale:     %.3f\n", cfg->Fe_scale);
    printf("  Ca_scale:     %.3f\n", cfg->Ca_scale);
    printf("  S_scale:      %.3f\n", cfg->S_scale);
    printf("\n");

    /* Initialize simulation state */
    SimulationState state;
    int status = simulation_state_init(&state, atomic, cfg->n_shells, cfg->t_exp);
    if (status != 0) {
        fprintf(stderr, "Failed to initialize simulation state\n");
        return;
    }

    /* Set up shell geometry and properties (velocity-based) */
    printf("[SETUP] Configuring %d shells (v = %.0f - %.0f km/s)...\n",
           cfg->n_shells, cfg->v_inner / 1e5, cfg->v_outer / 1e5);

    /*
     * Velocity-based shell setup (homologous expansion: r = v * t)
     * This is more physical for SNe where v is constant per mass shell.
     */
    double log_v_inner = log(cfg->v_inner);
    double log_v_outer = log(cfg->v_outer);
    double d_log_v = (log_v_outer - log_v_inner) / cfg->n_shells;

    /* Temperature gradient (linear in log-velocity) */
    double log_T_inner = log(cfg->T_inner);
    double log_T_outer = log(cfg->T_outer);
    double d_log_T = (log_T_outer - log_T_inner) / cfg->n_shells;

    for (int i = 0; i < cfg->n_shells; i++) {
        /* Velocity boundaries (logarithmic spacing) */
        double v_in = exp(log_v_inner + i * d_log_v);
        double v_out = exp(log_v_inner + (i + 1) * d_log_v);
        double v_mid = 0.5 * (v_in + v_out);

        /* Convert to radius via homologous expansion: r = v * t_exp */
        double r_in = v_in * cfg->t_exp;
        double r_out = v_out * cfg->t_exp;

        simulation_set_shell_geometry(&state, i, r_in, r_out);

        /* Temperature profile: power-law (linear in log-space) */
        double T = exp(log_T_inner + i * d_log_T);
        simulation_set_shell_temperature(&state, i, T);

        /* Density profile: power law in velocity */
        double rho = cfg->rho_inner * pow(v_mid / cfg->v_inner, cfg->rho_profile);
        simulation_set_shell_density(&state, i, rho);
    }

    /* Set abundances */
    if (cfg->X_H > 0 || cfg->X_He > 0) {
        /* H/He mode */
        Abundances ab;
        abundances_set_h_he(&ab, cfg->X_H, cfg->X_He);
        simulation_set_abundances(&state, &ab);
    } else if (cfg->stratified) {
        /* Stratified Type Ia abundances (velocity-dependent) */
        /* Check if scaling factors are non-default */
        int has_scaling = (cfg->Si_scale != 1.0 || cfg->Fe_scale != 1.0 ||
                           cfg->Ca_scale != 1.0 || cfg->S_scale != 1.0);
        if (has_scaling) {
            /* Use scaled abundances for optimizer */
            simulation_set_scaled_abundances(&state,
                                              cfg->Si_scale, cfg->Fe_scale,
                                              cfg->Ca_scale, cfg->S_scale);
        } else {
            /* Standard stratified abundances */
            simulation_set_stratified_abundances(&state);
        }
    } else {
        /* Uniform Type Ia W7-like composition */
        Abundances ab;
        abundances_set_type_ia_w7(&ab);
        simulation_set_abundances(&state, &ab);
    }

    /* Compute plasma state */
    simulation_compute_plasma(&state);

    /* Compute line opacities */
    simulation_compute_opacities(&state);

    /* Print summary */
    simulation_print_summary(&state);

    /* Print a few shell details */
    printf("\n[SHELL DETAILS]\n");
    simulation_print_shell(&state, 0);
    simulation_print_shell(&state, cfg->n_shells / 2);
    simulation_print_shell(&state, cfg->n_shells - 1);

    /* Initialize spectrum accumulator */
    SimpleSpectrum spectrum;
    simple_spectrum_init(&spectrum, cfg->n_bins, cfg->nu_min, cfg->nu_max);

    /* Initialize virtual packet collection (TARDIS-style spectrum synthesis) */
    VPacketCollection vpacket_coll;
    if (g_physics_overrides.enable_virtual_packets) {
        vpacket_collection_init(&vpacket_coll, cfg->n_packets * 10,
                                 cfg->nu_min, cfg->nu_max, cfg->n_bins);
        printf("[VPACKET] Initialized collection: capacity=%ld, n_bins=%d\n",
               (long)(cfg->n_packets * 10), cfg->n_bins);
    }

    /* Calculate inner radius from velocity */
    double r_inner = cfg->v_inner * cfg->t_exp;

    /*
     * PLANCK WEIGHTING FIX (Task Order #26)
     * ------------------------------------
     * Packets are sampled uniform in log(ν), but energy must be weighted
     * by the Planck function B_ν(T) to produce a thermal spectrum.
     *
     * B_ν(T) = (2hν³/c²) / (exp(hν/kT) - 1)
     *
     * For uniform log(ν) sampling, weight each packet by:
     *   w = B_ν(T) × ν  (the ν factor accounts for d(log ν) = dν/ν)
     *
     * We pre-compute a normalization factor so total energy sums to 1.
     */
    #define H_PLANCK_CGS 6.62607015e-27   /* Planck constant [erg·s] */
    #define K_BOLTZ_CGS  1.380649e-16     /* Boltzmann constant [erg/K] */
    #define C_CGS        2.99792458e10    /* Speed of light [cm/s] */

    /*
     * Task Order #30: PhysicsOverrides-based Planck boundary temperature
     * Uses g_physics_overrides.t_boundary (default 60000 K)
     */
    double T_photosphere = g_physics_overrides.t_boundary;
    printf("[PLANCK] Using T_boundary = %.0f K for packet weighting\n", T_photosphere);

    /* ========================================================================
     * TEMPERATURE ITERATION LOOP (TARDIS-style Radiative Equilibrium)
     * ========================================================================
     *
     * Reference: Lucy 2005, A&A 429, 19; TARDIS documentation
     *
     * Algorithm:
     * ----------
     * For iter = 1 to t_iter_max:
     *   1. Reset MC estimators
     *   2. Run MC transport, accumulating J-estimators
     *   3. Normalize estimators
     *   4. If iter > t_hold: check convergence
     *   5. If not converged: update temperatures using T_rad = (π J / σ)^{1/4}
     *   6. Recompute plasma state and opacities
     *
     * The iteration continues until:
     *   - Temperature converges (max relative change < t_converge)
     *   - Maximum iterations reached (t_iter_max)
     */

    /* Initialize MC estimators for temperature iteration */
    MCEstimators estimators;
    int use_estimators = cfg->enable_t_iteration;

    /*
     * LUMINOSITY TRACKING (TARDIS-style)
     * ----------------------------------
     * Compute the requested luminosity from T_inner using Stefan-Boltzmann:
     *   L = 4π R² σ T⁴
     *
     * This is used to:
     *   1. Normalize packet energies to physical luminosity
     *   2. Update T_inner based on L_emitted / L_requested ratio
     */
    #define CONST_SIGMA_SB 5.670374419e-5  /* Stefan-Boltzmann [erg/cm²/s/K⁴] */

    double L_requested = 4.0 * CONST_PI * r_inner * r_inner *
                         CONST_SIGMA_SB * pow(cfg->T_inner, 4.0);

    printf("\n[LUMINOSITY] Computing from Stefan-Boltzmann:\n");
    printf("  R_inner = %.3e cm (v_inner × t_exp)\n", r_inner);
    printf("  T_inner = %.0f K\n", cfg->T_inner);
    printf("  L_requested = %.3e erg/s = %.2f log(L_sun)\n",
           L_requested, log10(L_requested / 3.828e33));

    LuminosityEstimators lum_est;
    luminosity_estimators_init(&lum_est, L_requested, cfg->T_inner, cfg->t_fraction);

    if (use_estimators) {
        mc_estimators_init(&estimators, cfg->n_shells, 0);  /* No j_blue for now */
        mc_estimators_compute_volumes(&estimators, &state);
        printf("\n[T-ITERATION] TARDIS-style temperature iteration ENABLED\n");
        printf("  Max iterations: %d, Convergence: %.1f%%, Damping: %.2f\n",
               cfg->t_iter_max, cfg->t_converge * 100.0, cfg->t_damping);
        printf("  Hold iterations: %d (skip convergence check)\n", cfg->t_hold);
        printf("  L_requested: %.3e erg/s\n", L_requested);
    }

    int iter_max = cfg->enable_t_iteration ? cfg->t_iter_max : 1;
    int converged = 0;

    clock_t start_time = clock();

    for (int iter = 0; iter < iter_max && !converged; iter++) {

        if (use_estimators) {
            printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
            printf("║               TEMPERATURE ITERATION %2d / %2d                   ║\n",
                   iter + 1, iter_max);
            printf("╚═══════════════════════════════════════════════════════════════╝\n");

            /* Reset estimators for this iteration */
            mc_estimators_reset(&estimators);
            luminosity_estimators_reset(&lum_est);
        }

        /* Reset spectrum for this iteration */
        memset(spectrum.luminosity, 0, spectrum.n_bins * sizeof(double));
        memset(spectrum.luminosity_lumina, 0, spectrum.n_bins * sizeof(double));
        memset(spectrum.counts, 0, spectrum.n_bins * sizeof(int64_t));
        memset(spectrum.counts_lumina, 0, spectrum.n_bins * sizeof(int64_t));
        spectrum.n_escaped = 0;
        spectrum.n_absorbed = 0;
        spectrum.n_scattered = 0;
        spectrum.total_energy = 0.0;

        /* Re-compute Planck normalization with current T_inner (may have changed) */
        double T_current = cfg->enable_t_iteration ? state.shells[0].plasma.T : T_photosphere;
        double kT = K_BOLTZ_CGS * T_current;

        double planck_sum = 0.0;
        int n_samples = 1000;
        for (int j = 0; j < n_samples; j++) {
            double log_nu_j = log(cfg->nu_min) + (j + 0.5) / n_samples * (log(cfg->nu_max) - log(cfg->nu_min));
            double nu_j = exp(log_nu_j);
            double x = H_PLANCK_CGS * nu_j / kT;
            double B_nu = (x < 100.0) ? (2.0 * H_PLANCK_CGS * nu_j * nu_j * nu_j / (C_CGS * C_CGS)) / (exp(x) - 1.0) : 0.0;
            planck_sum += B_nu * nu_j;  /* Weight by ν for log-uniform sampling */
        }
        planck_sum /= n_samples;  /* Average B_ν × ν over the range */

        /*
         * PHYSICAL PACKET ENERGY (TARDIS-style)
         * ------------------------------------
         * Each packet carries a fraction of the total luminosity:
         *   packet_energy = L_requested / n_packets
         *
         * This gives proper physical units (erg/s) for the J-estimator.
         */
        double packet_energy_base = L_requested / cfg->n_packets;  /* Physical luminosity per packet */
        int progress_step = cfg->n_packets / 10;
        if (progress_step < 1) progress_step = 1;

        /* Run Monte Carlo transport */
        printf("\n[TRANSPORT] Running %d packets (iteration %d)...\n", cfg->n_packets, iter + 1);

        for (int i = 0; i < cfg->n_packets; i++) {
            /* Initialize packet at inner boundary */
            TransportPkt pkt;

            /* Random frequency (uniform in log space) */
            double log_nu = log(cfg->nu_min) +
                            drand48() * (log(cfg->nu_max) - log(cfg->nu_min));
            double nu = exp(log_nu);

            /* Weight packet energy by Planck function */
            double x = H_PLANCK_CGS * nu / kT;
            double B_nu = (x < 100.0) ? (2.0 * H_PLANCK_CGS * nu * nu * nu / (C_CGS * C_CGS)) / (exp(x) - 1.0) : 0.0;
            double planck_weight = (planck_sum > 0.0) ? (B_nu * nu) / planck_sum : 1.0;
            double packet_energy = packet_energy_base * planck_weight;

            init_transport_pkt(&pkt, r_inner, nu, packet_energy);

            /* Transport with estimator updates */
            transport_single_packet_with_estimators(&pkt, &state, &spectrum,
                                                     use_estimators ? &estimators : NULL,
                                                     g_physics_overrides.enable_virtual_packets ? &vpacket_coll : NULL,
                                                     1000);

            /* Update statistics */
            if (pkt.status == 1) {
                spectrum.n_escaped++;
                /* Track escaped luminosity for T_inner update */
                if (use_estimators) {
                    luminosity_estimators_add_emitted(&lum_est, pkt.energy);
                }
            } else {
                spectrum.n_absorbed++;
                /* Track absorbed luminosity */
                if (use_estimators) {
                    luminosity_estimators_add_absorbed(&lum_est, pkt.energy);
                }
            }
            spectrum.total_energy += pkt.energy;

            /* Progress */
            if ((i + 1) % progress_step == 0 && !use_estimators) {
                printf("  Progress: %d/%d packets (%.1f%%)\n",
                       i + 1, cfg->n_packets, 100.0 * (i + 1) / cfg->n_packets);
            }
        }

        printf("  Escaped: %ld (L=%.3e erg/s), Absorbed: %ld (L=%.3e erg/s)\n",
               (long)spectrum.n_escaped, lum_est.L_emitted,
               (long)spectrum.n_absorbed, lum_est.L_absorbed);

        /* Temperature update if iteration is enabled */
        if (use_estimators) {
            /*
             * TARDIS-STYLE J-ESTIMATOR NORMALIZATION
             * --------------------------------------
             * The J-estimator needs to be normalized by the ACTUAL luminosity
             * flowing through the simulation, not the normalized packet sum.
             *
             * total_energy now contains physical luminosity (erg/s) since we
             * set packet_energy = L_requested / n_packets.
             */
            mc_estimators_normalize(&estimators, spectrum.total_energy);

            /*
             * TARDIS-STYLE T_INNER UPDATE
             * ---------------------------
             * Update the inner boundary temperature based on luminosity ratio:
             *   T_new = T_old × (L_emitted / L_requested)^0.25
             *
             * If L_emitted > L_requested, we're too hot → cool down
             * If L_emitted < L_requested, we're too cold → heat up
             */
            double T_inner_old = state.shells[0].plasma.T;
            double T_inner_new = luminosity_update_T_inner(&lum_est, cfg->t_damping);

            /* Update inner shell temperature */
            state.shells[0].plasma.T = T_inner_new;

            /* Check convergence (skip hold iterations) */
            if (iter >= cfg->t_hold) {
                double max_delta_T = simulation_update_temperatures(&state, &estimators,
                                                                     cfg->t_damping);

                /* Also check luminosity convergence */
                int lum_converged = luminosity_converged(&lum_est, cfg->t_converge);

                if (temperature_converged(max_delta_T, cfg->t_converge) && lum_converged) {
                    printf("\n[T-ITERATION] *** CONVERGED *** after %d iterations\n", iter + 1);
                    printf("  Max ΔT = %.2f%%, L_ratio = %.3f\n",
                           max_delta_T * 100.0, lum_est.L_emitted / lum_est.L_requested);
                    converged = 1;
                } else {
                    printf("[T-ITERATION] Not converged (max ΔT = %.2f%%, L_ratio = %.3f)\n",
                           max_delta_T * 100.0, lum_est.L_emitted / lum_est.L_requested);

                    /* Recompute plasma state and opacities with new temperatures */
                    if (iter < iter_max - 1) {
                        printf("[T-ITERATION] Recomputing plasma state...\n");
                        simulation_compute_plasma(&state);
                        simulation_compute_opacities(&state);
                    }
                }
            } else {
                printf("[T-ITERATION] Hold iteration %d/%d (skipping convergence check)\n",
                       iter + 1, cfg->t_hold);

                /* Update temperatures without checking convergence */
                simulation_update_temperatures(&state, &estimators, cfg->t_damping);

                /* Recompute plasma state and opacities */
                if (iter < iter_max - 1) {
                    printf("[T-ITERATION] Recomputing plasma state...\n");
                    simulation_compute_plasma(&state);
                    simulation_compute_opacities(&state);
                }
            }
        }
    }

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    /* Statistics */
    printf("\n[STATISTICS]\n");
    printf("  Runtime:     %.2f s\n", elapsed);
    printf("  Packets/sec: %.0f\n", cfg->n_packets / elapsed);
    printf("  Escaped:     %ld (%.1f%%)\n",
           (long)spectrum.n_escaped, 100.0 * spectrum.n_escaped / cfg->n_packets);
    printf("  Absorbed:    %ld (%.1f%%)\n",
           (long)spectrum.n_absorbed, 100.0 * spectrum.n_absorbed / cfg->n_packets);
    printf("  Scattered:   %ld\n", (long)spectrum.n_scattered);

    /* Print line interaction diagnostics */
    print_line_stats();

    if (use_estimators) {
        double L_target = lum_est.L_requested * lum_est.fraction;
        printf("  T_inner:     %.0f K (final)\n", state.shells[0].plasma.T);
        printf("  L_emitted:   %.3e erg/s\n", lum_est.L_emitted);
        printf("  L_target:    %.3e erg/s (frac=%.2f)\n", L_target, lum_est.fraction);
        printf("  L_ratio:     %.4f (vs target)\n", lum_est.L_emitted / L_target);
        printf("  Iterations:  %d%s\n", converged ? iter_max : cfg->t_iter_max,
               converged ? " (converged)" : " (max reached)");
    }

    /* Write spectrum */
    simple_spectrum_write_csv(&spectrum, cfg->output_file, cfg->t_exp);

    /* Write virtual packet spectrum if enabled */
    if (g_physics_overrides.enable_virtual_packets) {
        char vpacket_filename[256];
        snprintf(vpacket_filename, sizeof(vpacket_filename), "lumina_vpacket_spectrum.dat");
        vpacket_spectrum_write_csv(&vpacket_coll, vpacket_filename, cfg->t_exp);
        printf("[VPACKET] Wrote virtual packet spectrum: %s\n", vpacket_filename);
        printf("[VPACKET] Collected %ld virtual packets\n", (long)vpacket_coll.n_packets);
    }

    /* Cleanup */
    if (use_estimators) {
        mc_estimators_free(&estimators);
    }
    simple_spectrum_free(&spectrum);
    if (g_physics_overrides.enable_virtual_packets) {
        vpacket_collection_free(&vpacket_coll);
    }
    simulation_state_free(&state);
    free_tau_sobolev_cache();  /* Free macro-atom tau cache */

    /* Task Order #28: Report weight audit */
    report_weight_audit();

    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                   SIMULATION COMPLETE                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[])
{
    /* Seed random number generator */
    srand48(time(NULL));

    /* Parse command line */
    SimConfig cfg;
    config_set_defaults(&cfg);

    /* Parse command line arguments */
    int type_ia_mode = 1;  /* Default: Type Ia (no H/He) */
    int pos_arg = 0;       /* Positional argument counter */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--h-he") == 0) {
            type_ia_mode = 0;
            cfg.X_H = 0.7;
            cfg.X_He = 0.3;
            cfg.stratified = 0;
        } else if (strcmp(argv[i], "--type-ia") == 0) {
            type_ia_mode = 1;
        } else if (strcmp(argv[i], "--stratified") == 0) {
            cfg.stratified = 1;
        } else if (strcmp(argv[i], "--uniform") == 0) {
            cfg.stratified = 0;
        } else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
            cfg.T_inner = atof(argv[++i]);
        } else if (strcmp(argv[i], "--v-inner") == 0 && i + 1 < argc) {
            cfg.v_inner = atof(argv[++i]) * 1e5;  /* km/s -> cm/s */
        } else if (strcmp(argv[i], "--v-outer") == 0 && i + 1 < argc) {
            cfg.v_outer = atof(argv[++i]) * 1e5;  /* km/s -> cm/s */
        } else if (strcmp(argv[i], "--rho-exp") == 0 && i + 1 < argc) {
            cfg.rho_profile = -atof(argv[++i]);  /* Store as negative for ρ ∝ v^-n */
        } else if (strcmp(argv[i], "--tardis-mode") == 0) {
            /* TARDIS-compatible mode: disable ALL extra LUMINA physics layers */
            setenv("LUMINA_TARDIS_MODE", "1", 1);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options] [atomic_file] [n_packets] [output_file]\n\n", argv[0]);
            printf("Options:\n");
            printf("  --h-he            Use H/He abundances\n");
            printf("  --type-ia         Use Type Ia composition (default)\n");
            printf("  --stratified      Use velocity-stratified abundances (default)\n");
            printf("  --uniform         Use uniform abundances\n");
            printf("  --T <K>           Set photospheric temperature (default: 13500 K)\n");
            printf("  --v-inner <km/s>  Set inner velocity (default: 10000 km/s)\n");
            printf("  --v-outer <km/s>  Set outer velocity (default: 25000 km/s)\n");
            printf("  --rho-exp <n>     Set density exponent ρ ∝ v^-n (default: 7)\n");
            printf("  --tardis-mode     TARDIS-compatible mode: disable ALL extra LUMINA layers\n");
            printf("\nEnvironment variables:\n");
            printf("  LUMINA_T_ALPHA      Excitation T ratio T_exc/T_eff (default: 1.0)\n");
            printf("  LUMINA_SI_SCALE     Silicon abundance multiplier (default: 1.0)\n");
            printf("  LUMINA_FE_SCALE     Iron abundance multiplier (default: 1.0)\n");
            printf("  LUMINA_CA_SCALE     Calcium abundance multiplier (default: 1.0)\n");
            printf("  LUMINA_S_SCALE      Sulfur abundance multiplier (default: 1.0)\n");
            printf("\nTask Order #30 v2: Physics Engine Controls:\n");
            printf("  LUMINA_T_BOUNDARY   Planck weighting temperature (default: 13000 K)\n");
            printf("  LUMINA_EPSILON      Base thermalization probability (default: 0.35)\n");
            printf("  LUMINA_EPSILON_IR   IR (λ>7000Å) thermalization (default: 0.50)\n");
            printf("  LUMINA_FE_BLUE_SCALE  Fe-group blue opacity scale (default: 0.50)\n");
            printf("\nContinuum Opacity:\n");
            printf("  LUMINA_CONTINUUM    Enable continuum opacity (default: 1)\n");
            printf("  LUMINA_BF_SCALE     Bound-free opacity multiplier (default: 1.0)\n");
            printf("  LUMINA_FF_SCALE     Free-free opacity multiplier (default: 1.0)\n");
            printf("  LUMINA_DILUTION     Enable NLTE dilution factor (default: 1)\n");
            printf("  LUMINA_LEGACY_MODE  Use old 60000K hack mode (default: 0)\n");
            printf("\nWavelength-Dependent Fluorescence (Task Order #30 v2.1):\n");
            printf("  LUMINA_WL_FLUOR      Enable wavelength-dep fluorescence (default: 1)\n");
            printf("  LUMINA_UV_BLUE_PROB  UV→Blue fluorescence prob (default: 0.85)\n");
            printf("  LUMINA_BLUE_SCATTER  Blue photon scatter prob (default: 0.70)\n");
            printf("\nTemperature Iteration (Radiative Equilibrium) - ENABLED BY DEFAULT:\n");
            printf("  LUMINA_T_ITERATION   Enable T iteration (default: 1, enabled)\n");
            printf("  LUMINA_T_ITER_MAX    Maximum iterations (default: 12)\n");
            printf("  LUMINA_T_CONVERGE    Convergence threshold (default: 0.05 = 5%%)\n");
            printf("  LUMINA_T_DAMPING     Damping factor (default: 0.7)\n");
            printf("  LUMINA_T_HOLD        Hold iterations (default: 3)\n");
            printf("  LUMINA_T_FRACTION    Luminosity fraction (default: 0.8)\n");
            printf("\nMacro-Atom Tuning (TARDIS-style thermalization):\n");
            printf("  LUMINA_MACRO_ATOM    Enable macro-atom mode (default: 1)\n");
            printf("  MACRO_EPSILON        Base thermalization probability (default: 0.35)\n");
            printf("  MACRO_IR_THERM       IR (λ>7000Å) thermalization (default: 0.80)\n");
            printf("  MACRO_COLLISIONAL_BOOST  Collisional rate multiplier (default: 10.0)\n");
            printf("  MACRO_GAUNT_SCALE    Gaunt factor scale (default: 5.0)\n");
            printf("  MACRO_UV_SCATTER     UV scatter boost (default: 0.5)\n");
            return 0;
        } else if (argv[i][0] != '-') {
            /* Positional arguments: atomic_file, n_packets, output_file */
            switch (pos_arg) {
                case 0: strcpy(cfg.atomic_file, argv[i]); break;
                case 1: cfg.n_packets = atoi(argv[i]); break;
                case 2: strcpy(cfg.output_file, argv[i]); break;
            }
            pos_arg++;
        }
    }

    /* Read physical parameters from environment (for optimizer control) */
    const char *env_t_alpha = getenv("LUMINA_T_ALPHA");
    const char *env_si = getenv("LUMINA_SI_SCALE");
    const char *env_fe = getenv("LUMINA_FE_SCALE");
    const char *env_ca = getenv("LUMINA_CA_SCALE");
    const char *env_s  = getenv("LUMINA_S_SCALE");

    /* Task Order #30 v2: Physics Engine Parameters */
    const char *env_t_boundary = getenv("LUMINA_T_BOUNDARY");
    const char *env_epsilon = getenv("LUMINA_EPSILON");
    const char *env_epsilon_ir = getenv("LUMINA_EPSILON_IR");
    const char *env_fe_blue = getenv("LUMINA_FE_BLUE_SCALE");

    /* NEW: Continuum opacity controls */
    const char *env_continuum = getenv("LUMINA_CONTINUUM");
    const char *env_bf_scale = getenv("LUMINA_BF_SCALE");
    const char *env_ff_scale = getenv("LUMINA_FF_SCALE");
    const char *env_dilution = getenv("LUMINA_DILUTION");
    const char *env_legacy = getenv("LUMINA_LEGACY_MODE");

    /* NEW: Wavelength-dependent fluorescence controls (Task Order #30 v2.1) */
    const char *env_wl_fluor = getenv("LUMINA_WL_FLUOR");       /* Enable wavelength-dependent model */
    const char *env_uv_blue_prob = getenv("LUMINA_UV_BLUE_PROB");/* UV→blue fluorescence probability */
    const char *env_blue_scatter = getenv("LUMINA_BLUE_SCATTER");/* Blue photon scatter probability */

    /* Line interaction type control (TARDIS-style) */
    const char *env_line_interaction = getenv("LUMINA_LINE_INTERACTION_TYPE");  /* 0=SCATTER, 1=DOWNBRANCH, 2=MACROATOM */
    /* Legacy: macro-atom control (deprecated - use LINE_INTERACTION_TYPE instead) */
    const char *env_macro_atom = getenv("LUMINA_MACRO_ATOM");   /* Enable/disable macro-atom mode */

    /* Virtual packet controls (TARDIS-style spectrum synthesis) */
    const char *env_vpacket = getenv("LUMINA_VPACKET");         /* Enable virtual packet spawning */
    const char *env_n_vpacket = getenv("LUMINA_N_VPACKET");     /* Number of v-packets per spawn */

    /* Temperature iteration controls */
    const char *env_t_iteration = getenv("LUMINA_T_ITERATION");
    const char *env_t_iter_max = getenv("LUMINA_T_ITER_MAX");
    const char *env_t_converge = getenv("LUMINA_T_CONVERGE");
    const char *env_t_damping = getenv("LUMINA_T_DAMPING");
    const char *env_t_hold = getenv("LUMINA_T_HOLD");
    const char *env_t_fraction = getenv("LUMINA_T_FRACTION");

    if (env_t_alpha) cfg.t_alpha = atof(env_t_alpha);
    if (env_si) cfg.Si_scale = atof(env_si);
    if (env_fe) cfg.Fe_scale = atof(env_fe);
    if (env_ca) cfg.Ca_scale = atof(env_ca);
    if (env_s)  cfg.S_scale = atof(env_s);

    /* Task Order #30: Apply new parameters to SimConfig */
    if (env_t_boundary) cfg.T_boundary = atof(env_t_boundary);
    if (env_epsilon) cfg.epsilon_default = atof(env_epsilon);
    if (env_epsilon_ir) cfg.epsilon_ir = atof(env_epsilon_ir);
    if (env_fe_blue) cfg.fe_blue_scale = atof(env_fe_blue);

    /* Set global config pointer for opacity functions */
    g_cfg = &cfg;

    /* Task Order #30 v2: Initialize PhysicsOverrides with new physics engine */
    PhysicsOverrides overrides;

    /* Check for TARDIS mode FIRST - disables ALL extra LUMINA physics */
    const char *env_tardis_mode = getenv("LUMINA_TARDIS_MODE");
    int use_tardis_mode = env_tardis_mode ? atoi(env_tardis_mode) : 0;

    /* Check for legacy mode (backwards compatibility with 60,000K hack) */
    int use_legacy_mode = env_legacy ? atoi(env_legacy) : 0;

    if (use_tardis_mode) {
        printf("[PHYSICS] TARDIS MODE: All extra LUMINA physics layers DISABLED\n");
        printf("[PHYSICS] Only macro-atom cascade determines photon fate\n");
        overrides = physics_overrides_tardis_mode();
        /* Also disable macro-atom tuning extras */
        setenv("MACRO_GAUNT_SCALE", "1.0", 1);
        setenv("MACRO_COLLISIONAL_BOOST", "1.0", 1);
        setenv("MACRO_EPSILON", "0.0", 1);
        setenv("MACRO_IR_THERM", "0.0", 1);
        setenv("MACRO_UV_SCATTER", "1.0", 1);
    } else if (use_legacy_mode) {
        printf("[PHYSICS] LEGACY MODE: Using 60,000K hack (no continuum opacity)\n");
        overrides = physics_overrides_legacy_hack();
    } else {
        /* Default: NEW PHYSICAL MODE with continuum opacity */
        printf("[PHYSICS] PHYSICAL MODE: Continuum opacity enabled\n");
        overrides = physics_overrides_default();
    }

    /* Apply environment overrides */
    if (env_t_boundary) overrides.t_boundary = atof(env_t_boundary);
    if (env_epsilon) overrides.base_thermalization_frac = atof(env_epsilon);
    if (env_epsilon_ir) overrides.ir_thermalization_frac = atof(env_epsilon_ir);
    if (env_fe_blue) overrides.blue_opacity_scalar = atof(env_fe_blue);
    if (env_continuum) overrides.enable_continuum_opacity = (atoi(env_continuum) != 0);
    if (env_bf_scale) overrides.bf_opacity_scale = atof(env_bf_scale);
    if (env_ff_scale) overrides.ff_opacity_scale = atof(env_ff_scale);
    if (env_dilution) overrides.enable_dilution_factor = (atoi(env_dilution) != 0);

    /* Apply wavelength fluorescence overrides */
    if (env_wl_fluor) overrides.enable_wavelength_fluorescence = (atoi(env_wl_fluor) != 0);
    if (env_uv_blue_prob) overrides.uv_to_blue_probability = atof(env_uv_blue_prob);
    if (env_blue_scatter) overrides.blue_scatter_probability = atof(env_blue_scatter);

    /* Apply line interaction type override (TARDIS-style) */
    if (env_line_interaction) {
        overrides.line_interaction_type = atoi(env_line_interaction);
        /* Sync with legacy use_macro_atom for compatibility */
        overrides.use_macro_atom = (overrides.line_interaction_type == LINE_MACROATOM) ? 1 : 0;
    }
    /* Legacy: macro-atom override (deprecated) */
    if (env_macro_atom) {
        overrides.use_macro_atom = atoi(env_macro_atom);
        /* If legacy macro-atom is enabled, set line_interaction_type to MACROATOM */
        if (overrides.use_macro_atom) {
            overrides.line_interaction_type = LINE_MACROATOM;
        }
    }

    /* Apply temperature iteration overrides */
    if (env_t_iteration) cfg.enable_t_iteration = atoi(env_t_iteration);
    if (env_t_iter_max) cfg.t_iter_max = atoi(env_t_iter_max);
    if (env_t_converge) cfg.t_converge = atof(env_t_converge);
    if (env_t_damping) cfg.t_damping = atof(env_t_damping);
    if (env_t_hold) cfg.t_hold = atoi(env_t_hold);
    if (env_t_fraction) cfg.t_fraction = atof(env_t_fraction);

    /* Apply virtual packet overrides */
    if (env_vpacket) overrides.enable_virtual_packets = atoi(env_vpacket);
    if (env_n_vpacket) overrides.n_virtual_packets = atoi(env_n_vpacket);

    physics_overrides_set(&overrides);

    /* Initialize macro-atom tuning parameters from environment */
    macro_atom_tuning_from_env();
    printf("[MACRO-ATOM TUNING] epsilon=%.2f, ir_therm=%.2f, collision_boost=%.1fx, gaunt=%.1fx\n",
           g_macro_atom_tuning.thermalization_epsilon,
           g_macro_atom_tuning.ir_thermalization_boost,
           g_macro_atom_tuning.collisional_boost,
           g_macro_atom_tuning.gaunt_factor_scale);
    if (g_macro_atom_tuning.downbranch_only) {
        printf("[MACRO-ATOM TUNING] DOWNBRANCH MODE: Only radiative de-excitation (no cascade)\n");
    } else {
        printf("[MACRO-ATOM TUNING] FULL MODE: Multi-step cascade with up/down transitions\n");
    }

    printf("[PHYSICS OVERRIDES] T_boundary=%.0fK, eps=%.2f, eps_ir=%.2f, blue_scale=%.2f\n",
           g_physics_overrides.t_boundary,
           g_physics_overrides.base_thermalization_frac,
           g_physics_overrides.ir_thermalization_frac,
           g_physics_overrides.blue_opacity_scalar);
    printf("[PHYSICS OVERRIDES] Continuum=%s, bf_scale=%.2f, ff_scale=%.2f, dilution=%s\n",
           g_physics_overrides.enable_continuum_opacity ? "ON" : "OFF",
           g_physics_overrides.bf_opacity_scale,
           g_physics_overrides.ff_opacity_scale,
           g_physics_overrides.enable_dilution_factor ? "ON" : "OFF");
    printf("[PHYSICS OVERRIDES] WL_Fluor=%s, UV→Blue=%.0f%%, Blue_Scatter=%.0f%%\n",
           g_physics_overrides.enable_wavelength_fluorescence ? "ON" : "OFF",
           g_physics_overrides.uv_to_blue_probability * 100.0,
           g_physics_overrides.blue_scatter_probability * 100.0);
    /* Line interaction type (TARDIS-style) */
    const char *line_type_names[] = {"SCATTER", "DOWNBRANCH", "MACROATOM"};
    int line_type = g_physics_overrides.line_interaction_type;
    if (line_type < 0 || line_type > 2) line_type = 0;
    printf("[LINE INTERACTION] Type=%d (%s)\n", line_type, line_type_names[line_type]);
    if (line_type == LINE_SCATTER) {
        printf("[LINE INTERACTION]   SCATTER: Pure resonance scattering (emit at SAME line frequency)\n");
    } else if (line_type == LINE_DOWNBRANCH) {
        printf("[LINE INTERACTION]   DOWNBRANCH: Simplified cascade (downward radiative only)\n");
    } else {
        printf("[LINE INTERACTION]   MACROATOM: Full cascade with up/down transitions\n");
    }
    /* Virtual packet status */
    printf("[VIRTUAL PACKETS] %s", g_physics_overrides.enable_virtual_packets ? "ENABLED" : "DISABLED");
    if (g_physics_overrides.enable_virtual_packets) {
        printf(" (n_vpackets=%d per spawn)\n", g_physics_overrides.n_virtual_packets);
        printf("[VIRTUAL PACKETS]   Spawning at: initial, e-scatter, line, continuum interactions\n");
    } else {
        printf("\n");
    }
    if (cfg.enable_t_iteration) {
        printf("[T-ITERATION] Radiative equilibrium iteration ENABLED:\n");
        printf("  Max iterations: %d, Convergence: %.1f%%, Damping: %.2f\n",
               cfg.t_iter_max, cfg.t_converge * 100.0, cfg.t_damping);
    }

    /* Set abundances based on mode */
    if (type_ia_mode) {
        cfg.X_H = 0.0;
        cfg.X_He = 0.0;
        if (cfg.stratified) {
            printf("[CONFIG] Type Ia mode: STRATIFIED composition\n");
        } else {
            printf("[CONFIG] Type Ia mode: UNIFORM W7 composition\n");
        }
        /* Report non-default parameters */
        if (cfg.t_alpha != 1.0) {
            printf("[CONFIG] NLTE excitation: T_alpha = %.3f (T_exc = %.0f K)\n",
                   cfg.t_alpha, cfg.T_inner * cfg.t_alpha);
        }
        if (cfg.Si_scale != 1.0 || cfg.Fe_scale != 1.0 ||
            cfg.Ca_scale != 1.0 || cfg.S_scale != 1.0) {
            printf("[CONFIG] Abundance scaling: Si=%.2fx, Fe=%.2fx, Ca=%.2fx, S=%.2fx\n",
                   cfg.Si_scale, cfg.Fe_scale, cfg.Ca_scale, cfg.S_scale);
        }
    } else {
        printf("[CONFIG] H/He mode: X_H=0.7, X_He=0.3\n");
    }

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║             LUMINA-SN Configuration (SN 2011fe)               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Atomic data:   %-45s ║\n", cfg.atomic_file);
    printf("║  N_shells:      %-45d ║\n", cfg.n_shells);
    printf("║  N_packets:     %-45d ║\n", cfg.n_packets);
    printf("║  t_exp:         %-42.1f days ║\n", cfg.t_exp / 86400.0);
    printf("║  v_inner:       %-39.0f km/s ║\n", cfg.v_inner / 1e5);
    printf("║  v_outer:       %-39.0f km/s ║\n", cfg.v_outer / 1e5);
    printf("║  T_eff:         %-42.0f K ║\n", cfg.T_inner);
    printf("║  ρ-exponent:    %-42.1f      ║\n", -cfg.rho_profile);
    printf("║  T_alpha:       %-42.2f      ║\n", cfg.t_alpha);
    printf("║  Abundances:    %-45s ║\n", cfg.stratified ? "STRATIFIED" : "UNIFORM");
    printf("║  Output:        %-45s ║\n", cfg.output_file);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    /* Load atomic data */
    printf("\n[INIT] Loading atomic data...\n");
    AtomicData atomic;
    int status = atomic_data_load_hdf5(cfg.atomic_file, &atomic);

    if (status != 0) {
        fprintf(stderr, "Failed to load atomic data from %s\n", cfg.atomic_file);
        return 1;
    }

    printf("[INIT] Loaded: %d ions, %d levels, %ld lines\n",
           atomic.n_ions, atomic.n_levels, (long)atomic.n_lines);

    /* Build downbranch (fluorescence cascade) table for proper line redistribution */
    printf("[INIT] Building downbranch table for fluorescence cascade...\n");
    if (atomic_build_downbranch_table(&atomic) != 0) {
        fprintf(stderr, "Warning: Failed to build downbranch table\n");
    } else {
        printf("[INIT] Downbranch table ready (%ld emission entries)\n",
               atomic.downbranch.total_emission_entries);
    }

    /* Run simulation */
    run_simulation(&cfg, &atomic);

    /* Cleanup */
    atomic_data_free(&atomic);

    return 0;
}
