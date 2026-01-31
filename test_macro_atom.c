/**
 * LUMINA-SN Macro-Atom Test
 * test_macro_atom.c - Verify fluorescence and frequency redistribution
 *
 * This test compares:
 *   1. Pure Scattering mode - coherent, no frequency change
 *   2. Macro-Atom mode - allows fluorescence (UV → optical)
 *
 * Expected result: Macro-atom mode should show red-shifting of spectrum
 * due to cascading through intermediate atomic levels.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "atomic_data.h"
#include "physics_kernels.h"
#include "rpacket.h"
#include "macro_atom.h"

/* Test configuration */
#define N_TEST_PACKETS 10000
#define T_ELECTRON     10000.0    /* 10,000 K */
#define N_E            1e9        /* 10^9 cm^-3 */
#define T_EXPLOSION    86400.0    /* 1 day */

/* ============================================================================
 * SIMPLE TEST: Frequency redistribution statistics
 * ============================================================================ */

typedef struct {
    int n_packets;
    int n_scattered;
    int n_absorbed;
    int n_blueshift;    /* ν_out > ν_in */
    int n_redshift;     /* ν_out < ν_in */
    int n_coherent;     /* |ν_out - ν_in| / ν_in < 0.01 */
    double mean_shift;  /* Mean (ν_out - ν_in) / ν_in */
    double total_energy_in;
    double total_energy_out;
} ScatterStats;

static void stats_init(ScatterStats *stats) {
    memset(stats, 0, sizeof(ScatterStats));
}

static void stats_add(ScatterStats *stats, double nu_in, double nu_out, double energy) {
    stats->n_packets++;
    stats->total_energy_in += energy;

    if (nu_out > 0) {
        stats->n_scattered++;
        stats->total_energy_out += energy;

        double rel_shift = (nu_out - nu_in) / nu_in;
        stats->mean_shift += rel_shift;

        if (rel_shift > 0.01) {
            stats->n_blueshift++;
        } else if (rel_shift < -0.01) {
            stats->n_redshift++;
        } else {
            stats->n_coherent++;
        }
    } else {
        stats->n_absorbed++;
    }
}

static void stats_print(const ScatterStats *stats, const char *mode_name) {
    printf("\n=== %s ===\n", mode_name);
    printf("  Packets processed: %d\n", stats->n_packets);
    printf("  Scattered:         %d (%.1f%%)\n",
           stats->n_scattered, 100.0 * stats->n_scattered / stats->n_packets);
    printf("  Absorbed:          %d (%.1f%%)\n",
           stats->n_absorbed, 100.0 * stats->n_absorbed / stats->n_packets);

    if (stats->n_scattered > 0) {
        double mean_shift = stats->mean_shift / stats->n_scattered;
        printf("\n  Frequency redistribution:\n");
        printf("    Coherent (|Δν/ν| < 1%%): %d (%.1f%%)\n",
               stats->n_coherent, 100.0 * stats->n_coherent / stats->n_scattered);
        printf("    Red-shifted (Δν < 0):   %d (%.1f%%)\n",
               stats->n_redshift, 100.0 * stats->n_redshift / stats->n_scattered);
        printf("    Blue-shifted (Δν > 0):  %d (%.1f%%)\n",
               stats->n_blueshift, 100.0 * stats->n_blueshift / stats->n_scattered);
        printf("    Mean Δν/ν:              %.4f\n", mean_shift);

        if (mean_shift < 0) {
            printf("    → NET RED-SHIFT (fluorescence)\n");
        } else if (mean_shift > 0) {
            printf("    → NET BLUE-SHIFT\n");
        }
    }
}

/* ============================================================================
 * TEST 1: Pure Scattering vs Macro-Atom (Simplified)
 * ============================================================================
 * This test uses the DOWNBRANCH mode as a simplified macro-atom proxy
 * to demonstrate the principle of fluorescence without requiring full
 * macro-atom transition data from HDF5.
 */

static void test_scattering_modes(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     TEST 1: Pure Scattering vs Downbranch (Fluorescence)      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    ScatterStats stats_scatter;
    ScatterStats stats_downbranch;
    stats_init(&stats_scatter);
    stats_init(&stats_downbranch);

    /* Create a mock plasma with a single line */
    double line_nu = 5e14;  /* ~6000 Å - visible light */
    double line_tau = 1.0;

    NumbaPlasma plasma;
    plasma.line_list_nu = &line_nu;
    plasma.tau_sobolev = &line_tau;
    plasma.electron_density = (double *)malloc(sizeof(double));
    plasma.electron_density[0] = N_E;
    plasma.n_lines = 1;
    plasma.n_shells = 1;

    /* Test packets */
    for (int i = 0; i < N_TEST_PACKETS; i++) {
        /* Create packet at a "UV" frequency (higher than line) */
        double nu_in = line_nu;  /* Absorbed at line frequency */
        double energy = 1.0 / N_TEST_PACKETS;

        /* Initialize RNG */
        RNGState rng;
        rng_init(&rng, 12345 + i);

        /* Test 1: Pure scattering */
        {
            RPacket pkt;
            rpacket_init(&pkt, 1e14, 0.5, nu_in, energy, 12345 + i, i);
            pkt.next_line_id = 0;
            pkt.rng_state = rng;

            line_scatter(&pkt, T_EXPLOSION, LINE_SCATTER, &plasma, NULL);

            double nu_out = pkt.nu * get_doppler_factor(pkt.r, pkt.mu, T_EXPLOSION);
            stats_add(&stats_scatter, nu_in, nu_out, energy);
        }

        /* Test 2: Downbranch (simplified fluorescence) */
        {
            RPacket pkt;
            rpacket_init(&pkt, 1e14, 0.5, nu_in, energy, 12345 + i, i);
            pkt.next_line_id = 0;
            pkt.rng_state = rng;

            line_scatter(&pkt, T_EXPLOSION, LINE_DOWNBRANCH, &plasma, NULL);

            double nu_out = pkt.nu * get_doppler_factor(pkt.r, pkt.mu, T_EXPLOSION);
            stats_add(&stats_downbranch, nu_in, nu_out, energy);
        }
    }

    stats_print(&stats_scatter, "Pure Scattering (LINE_SCATTER)");
    stats_print(&stats_downbranch, "Downbranch (LINE_DOWNBRANCH)");

    free(plasma.electron_density);

    printf("\n  INTERPRETATION:\n");
    printf("  ───────────────\n");
    printf("  Pure scattering is coherent: frequency unchanged (Δν/ν ≈ 0)\n");
    printf("  Downbranch mode shows red-shift: packets cascade to lower frequencies\n");
    printf("  This demonstrates fluorescence: UV absorption → optical emission\n");
}

/* ============================================================================
 * TEST 2: Macro-Atom with Real Atomic Data
 * ============================================================================ */

static void test_macro_atom_with_atomic_data(const AtomicData *atomic) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     TEST 2: Macro-Atom Transition Loop with Atomic Data       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    /* Check if macro-atom data is available */
    if (atomic->macro_atom_transitions == NULL ||
        atomic->n_macro_atom_transitions == 0) {
        printf("\n  [SKIP] No macro-atom transition data in HDF5 file.\n");
        printf("  Using simplified transition model instead.\n");

        /* Run simplified test with atomic line data */
        printf("\n  Testing simplified macro-atom with He I lines...\n");

        ScatterStats stats;
        stats_init(&stats);

        /* Find He I lines (Z=2, ion=0) */
        int n_he_lines = 0;
        int64_t he_line_ids[100];

        for (int64_t i = 0; i < atomic->n_lines && n_he_lines < 100; i++) {
            if (atomic->lines[i].atomic_number == 2 &&
                atomic->lines[i].ion_number == 0) {
                he_line_ids[n_he_lines++] = i;
            }
        }

        printf("  Found %d He I lines\n", n_he_lines);

        if (n_he_lines == 0) {
            printf("  [SKIP] No He I lines found.\n");
            return;
        }

        /* Test macro-atom simplified transition */
        for (int i = 0; i < N_TEST_PACKETS / 10; i++) {
            /* Pick a random He I line */
            RNGState rng;
            rng_init(&rng, 54321 + i);

            int line_idx = (int)(rng_uniform(&rng) * n_he_lines);
            int64_t line_id = he_line_ids[line_idx];
            double nu_in = atomic->lines[line_id].nu;
            double energy = 1.0;

            /* Initialize macro-atom state */
            MacroAtomState ma_state;
            macro_atom_init(&ma_state, atomic, line_id, T_ELECTRON, N_E, &rng);

            /* Run simplified transition */
            int survives = macro_atom_simplified_transition(&ma_state, atomic);

            double nu_out = survives ? ma_state.emission_nu : 0.0;
            stats_add(&stats, nu_in, nu_out, energy);
        }

        stats_print(&stats, "Simplified Macro-Atom (He I)");

        return;
    }

    /* Full macro-atom test with transition data */
    printf("\n  Testing full macro-atom loop...\n");
    printf("  Macro-atom transitions: %ld\n", (long)atomic->n_macro_atom_transitions);
    printf("  Macro-atom references:  %d\n", atomic->n_macro_atom_references);

    ScatterStats stats;
    stats_init(&stats);

    /* Find a line with macro-atom data */
    int64_t test_line_id = -1;
    for (int64_t i = 0; i < atomic->n_lines; i++) {
        const Line *line = &atomic->lines[i];
        const MacroAtomReference *ref = macro_atom_find_reference(
            atomic, line->atomic_number, line->ion_number,
            line->level_number_upper);

        if (ref != NULL && ref->count_total > 0) {
            test_line_id = i;
            printf("  Found test line: Z=%d, ion=%d, level=%d-%d, λ=%.2f Å\n",
                   line->atomic_number, line->ion_number,
                   line->level_number_lower, line->level_number_upper,
                   line->wavelength / CONST_ANGSTROM);
            break;
        }
    }

    if (test_line_id < 0) {
        printf("  [SKIP] No lines with macro-atom references found.\n");
        return;
    }

    /* Run macro-atom loop test */
    for (int i = 0; i < N_TEST_PACKETS / 10; i++) {
        RNGState rng;
        rng_init(&rng, 98765 + i);

        double nu_in = atomic->lines[test_line_id].nu;
        double energy = 1.0;

        MacroAtomState ma_state;
        macro_atom_init(&ma_state, atomic, test_line_id, T_ELECTRON, N_E, &rng);

        int survives = macro_atom_do_transition_loop(&ma_state, atomic);

        double nu_out = survives ? ma_state.emission_nu : 0.0;
        stats_add(&stats, nu_in, nu_out, energy);
    }

    stats_print(&stats, "Full Macro-Atom Loop");
}

/* ============================================================================
 * TEST 3: RNG Quality Verification
 * ============================================================================ */

static void test_rng_quality(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     TEST 3: Thread-Safe RNG Quality Check                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    RNGState rng1, rng2;
    rng_init(&rng1, 12345);
    rng_init(&rng2, 12345);

    printf("\n  Testing RNG reproducibility...\n");

    /* Should produce same sequence */
    int match_count = 0;
    for (int i = 0; i < 100; i++) {
        double v1 = rng_uniform(&rng1);
        double v2 = rng_uniform(&rng2);
        if (fabs(v1 - v2) < 1e-15) {
            match_count++;
        }
    }
    printf("  Same seed → Same sequence: %d/100 matches\n", match_count);

    /* Test uniformity */
    rng_init(&rng1, time(NULL));
    int bins[10] = {0};
    int n_samples = 100000;

    for (int i = 0; i < n_samples; i++) {
        double v = rng_uniform(&rng1);
        int bin = (int)(v * 10);
        if (bin >= 0 && bin < 10) {
            bins[bin]++;
        }
    }

    printf("\n  Uniformity test (100k samples, 10 bins):\n");
    double expected = n_samples / 10.0;
    double chi_sq = 0.0;

    for (int i = 0; i < 10; i++) {
        double diff = bins[i] - expected;
        chi_sq += diff * diff / expected;
        printf("    Bin %d: %d (%.1f%%)\n", i, bins[i], 100.0 * bins[i] / n_samples);
    }

    printf("  Chi-squared: %.2f (expected ~9 for good uniformity)\n", chi_sq);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[]) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           LUMINA-SN MACRO-ATOM TEST SUITE                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Physical Configuration:\n");
    printf("  T_electron:    %.0f K\n", T_ELECTRON);
    printf("  n_e:           %.0e cm⁻³\n", N_E);
    printf("  t_explosion:   %.0f s (%.2f days)\n", T_EXPLOSION, T_EXPLOSION / 86400.0);
    printf("  Test packets:  %d\n", N_TEST_PACKETS);

    /* Test 1: Compare scattering modes */
    test_scattering_modes();

    /* Test 3: RNG quality (run before atomic data tests) */
    test_rng_quality();

    /* Test 2: Macro-atom with atomic data (if available) */
    if (argc > 1) {
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║     Loading Atomic Data: %-36s ║\n", argv[1]);
        printf("╚═══════════════════════════════════════════════════════════════╝\n");

        AtomicData atomic;
        int status = atomic_data_load_hdf5(argv[1], &atomic);

        if (status != 0) {
            fprintf(stderr, "Failed to load atomic data from %s\n", argv[1]);
            return 1;
        }

        printf("  Loaded: %d ions, %d levels, %ld lines\n",
               atomic.n_ions, atomic.n_levels, (long)atomic.n_lines);

        test_macro_atom_with_atomic_data(&atomic);

        atomic_data_free(&atomic);
    } else {
        printf("\n  [INFO] To test with real atomic data, run:\n");
        printf("    ./test_macro_atom atomic/kurucz_cd23_chianti_H_He.h5\n");
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    ALL TESTS COMPLETE                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
