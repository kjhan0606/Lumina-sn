/**
 * LUMINA-SN Plasma Solver Debug (Task Order #040)
 * test_plasma_debug.c - Compare LUMINA ionization against TARDIS golden data
 *
 * Purpose:
 *   Load "Golden Inputs" (T_rad, n_e) from tardis_plasma_state.h5
 *   Run LUMINA's ionization solver
 *   Compare Si II fraction against TARDIS "Golden Output"
 *
 * Usage:
 *   ./test_plasma_debug atomic/kurucz_cd23_chianti_H_He.h5 tardis_plasma_state.h5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "atomic_data.h"
#include "plasma_physics.h"

/* TARDIS ion indexing for Type Ia elements */
#define TARDIS_SI_START 31  /* Si I at idx 31, Si II at idx 32, etc. */
#define TARDIS_FE_START 96  /* Fe I at idx 96, Fe II at idx 97, etc. */

/* Physical constants */
/* Use CONST_AMU from plasma_physics.h */

typedef struct {
    double T;           /* Temperature [K] */
    double n_e;         /* Electron density [cm^-3] */
    double Si_I;        /* Si I density [cm^-3] */
    double Si_II;       /* Si II density [cm^-3] */
    double Si_III;      /* Si III density [cm^-3] */
    double Si_IV;       /* Si IV density [cm^-3] */
    double Si_II_frac;  /* Si II fraction */
} TardisShellData;

static void print_header(void)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     TASK ORDER #040: PLASMA SOLVER DEBUG (TARDIS COMPARISON)      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Comparing LUMINA ionization solver against TARDIS golden data    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/**
 * Load TARDIS plasma state from HDF5 file
 */
static int load_tardis_data(const char *filename, TardisShellData *shells, int *n_shells)
{
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1;
    }

    /* Get dimensions */
    hsize_t dims[2];
    H5LTget_dataset_info(file_id, "/t_electrons", dims, NULL, NULL);
    *n_shells = (int)dims[0];

    if (*n_shells > 30) *n_shells = 30;

    /* Load temperature */
    double *T_buffer = (double *)malloc(*n_shells * sizeof(double));
    H5LTread_dataset_double(file_id, "/t_electrons", T_buffer);

    /* Load electron density */
    double *n_e_buffer = (double *)malloc(*n_shells * sizeof(double));
    H5LTread_dataset_double(file_id, "/electron_density", n_e_buffer);

    /* Load ion densities */
    H5LTget_dataset_info(file_id, "/ion_density", dims, NULL, NULL);
    int n_ions = (int)dims[0];
    int n_shells_ion = (int)dims[1];

    double *ion_buffer = (double *)malloc(n_ions * n_shells_ion * sizeof(double));
    H5LTread_dataset_double(file_id, "/ion_density", ion_buffer);

    /* Extract data for each shell */
    for (int s = 0; s < *n_shells; s++) {
        shells[s].T = T_buffer[s];
        shells[s].n_e = n_e_buffer[s];

        /* Si ions at indices 31-35 (Si I to Si V) */
        shells[s].Si_I   = ion_buffer[(TARDIS_SI_START + 0) * n_shells_ion + s];
        shells[s].Si_II  = ion_buffer[(TARDIS_SI_START + 1) * n_shells_ion + s];
        shells[s].Si_III = ion_buffer[(TARDIS_SI_START + 2) * n_shells_ion + s];
        shells[s].Si_IV  = ion_buffer[(TARDIS_SI_START + 3) * n_shells_ion + s];

        double Si_total = shells[s].Si_I + shells[s].Si_II + shells[s].Si_III + shells[s].Si_IV;
        shells[s].Si_II_frac = (Si_total > 0) ? shells[s].Si_II / Si_total : 0;
    }

    free(T_buffer);
    free(n_e_buffer);
    free(ion_buffer);
    H5Fclose(file_id);

    return 0;
}

/**
 * Run LUMINA ionization solver and compare against TARDIS
 */
static void compare_ionization(const AtomicData *atomic, const TardisShellData *tardis,
                                int shell_idx)
{
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("SHELL %d: T = %.1f K, n_e = %.3e cm^-3\n", shell_idx, tardis->T, tardis->n_e);
    printf("═══════════════════════════════════════════════════════════════════\n");

    /* Set up abundances (Type Ia - Si dominated) */
    Abundances ab;
    abundances_set_type_ia_w7(&ab);

    /* TARDIS golden output */
    printf("\n[TARDIS GOLDEN OUTPUT]\n");
    printf("  Si I:   %.4e\n", tardis->Si_I);
    printf("  Si II:  %.4e (%.2f%%)\n", tardis->Si_II, tardis->Si_II_frac * 100);
    printf("  Si III: %.4e\n", tardis->Si_III);
    printf("  Si IV:  %.4e\n", tardis->Si_IV);

    /* Run LUMINA standard Saha solver */
    PlasmaState plasma_saha;
    solve_ionization_balance(atomic, &ab, tardis->T, 1e-13, &plasma_saha);

    /* Renormalize using TARDIS n_e (fixed n_e mode) */
    /* For proper comparison, calculate with TARDIS's n_e */
    PlasmaState plasma;
    plasma_state_init(&plasma);
    plasma.T = tardis->T;
    plasma.n_e = tardis->n_e;

    /* Calculate partition functions */
    calculate_all_partition_functions(atomic, tardis->T, &plasma);

    /* Calculate Si number density from TARDIS total */
    double Si_total_tardis = tardis->Si_I + tardis->Si_II + tardis->Si_III + tardis->Si_IV;
    double n_Si = Si_total_tardis;

    printf("\n[LUMINA STANDARD SAHA] (using TARDIS n_e = %.3e)\n", tardis->n_e);

    /* Calculate Si ion fractions using Saha with TARDIS n_e */
    int Z = 14;  /* Silicon */
    double R[MAX_ATOMIC_NUMBER + 2];
    double sum_R = 0.0;
    R[0] = 1.0;
    sum_R = 1.0;
    double cumulative_ratio = 1.0;

    for (int i = 0; i < Z && i < 5; i++) {
        double U_i = plasma.partition_function[Z][i];
        double U_i1 = plasma.partition_function[Z][i + 1];
        double chi = atomic_get_ionization_energy(atomic, Z, i);

        double kT = 1.380649e-16 * tardis->T;  /* erg */
        double T32 = tardis->T * sqrt(tardis->T);

        /* Saha factor */
        /* Use SAHA_CONST from plasma_physics.h */
        double ratio_U = (U_i > 0.0) ? (2.0 * U_i1 / U_i) : 2.0;
        double phi = ratio_U * SAHA_CONST * T32 * exp(-chi / kT);

        if (tardis->n_e > 0.0 && phi > 0.0) {
            cumulative_ratio *= (phi / tardis->n_e);
        } else {
            cumulative_ratio = 0.0;
        }

        R[i + 1] = cumulative_ratio;
        sum_R += cumulative_ratio;

        printf("  Stage %d→%d: U_i=%.4f, U_i+1=%.4f, chi=%.4f eV, phi=%.3e, ratio=%.3e\n",
               i, i+1, U_i, U_i1, chi / 1.602e-12, phi, cumulative_ratio);
    }

    /* Calculate fractions */
    double lumina_Si_I = R[0] / sum_R;
    double lumina_Si_II = R[1] / sum_R;
    double lumina_Si_III = R[2] / sum_R;
    double lumina_Si_IV = (sum_R > 0 && R[3] > 0) ? R[3] / sum_R : 0;

    printf("\n  LUMINA Si fractions:\n");
    printf("  Si I:   %.4e (%.4f%%)\n", lumina_Si_I * n_Si, lumina_Si_I * 100);
    printf("  Si II:  %.4e (%.4f%%)\n", lumina_Si_II * n_Si, lumina_Si_II * 100);
    printf("  Si III: %.4e (%.4f%%)\n", lumina_Si_III * n_Si, lumina_Si_III * 100);
    printf("  Si IV:  %.4e (%.4f%%)\n", lumina_Si_IV * n_Si, lumina_Si_IV * 100);

    /* Discrepancy */
    double discrepancy = fabs(lumina_Si_II - tardis->Si_II_frac) * 100;
    printf("\n[DISCREPANCY]\n");
    printf("  TARDIS Si II: %.2f%%\n", tardis->Si_II_frac * 100);
    printf("  LUMINA Si II: %.2f%%\n", lumina_Si_II * 100);
    printf("  Difference:   %.2f percentage points\n", discrepancy);

    /* Diagnose: What would be needed to match? */
    printf("\n[DIAGNOSTIC] Checking physics factors:\n");

    /* Check ionization energy */
    double chi_Si_I = atomic_get_ionization_energy(atomic, 14, 0) / 1.602e-12;
    double chi_Si_II = atomic_get_ionization_energy(atomic, 14, 1) / 1.602e-12;
    printf("  χ(Si I → Si II) = %.4f eV (expected: 8.15 eV)\n", chi_Si_I);
    printf("  χ(Si II → Si III) = %.4f eV (expected: 16.35 eV)\n", chi_Si_II);

    /* Check partition functions */
    printf("  U(Si I, T=%.0f K) = %.4f\n", tardis->T, plasma.partition_function[14][0]);
    printf("  U(Si II, T=%.0f K) = %.4f\n", tardis->T, plasma.partition_function[14][1]);
    printf("  U(Si III, T=%.0f K) = %.4f\n", tardis->T, plasma.partition_function[14][2]);

    /* Check zeta factor (from TARDIS nebular approximation) */
    double zeta = atomic_get_zeta(atomic, 14, 2, tardis->T);  /* Si III recombination */
    printf("  ζ(Si III, T=%.0f K) = %.4f\n", tardis->T, zeta);

    /* If zeta < 1 and W < 1, the nebular formula changes things */
    /* Let's compute what W would need to be to match TARDIS */
    printf("\n[NEBULAR ANALYSIS]\n");

    /* TARDIS formula: n_{j+1}/n_j = (Φ/n_e) × W × δ where δ = 1/(ζ + W(1-ζ)) */
    /* Standard Saha: n_Si_III/n_Si_II = Φ_{II}/n_e */
    /* We know TARDIS ratio: R_tardis = Si_III / Si_II */
    double R_tardis = tardis->Si_III / tardis->Si_II;
    double R_saha = R[2] / R[1];  /* LUMINA Saha ratio */

    printf("  TARDIS Si III / Si II = %.4f\n", R_tardis);
    printf("  LUMINA Saha Si III / Si II = %.4f\n", R_saha);

    /* W × δ = R_tardis / R_saha */
    double W_delta = R_tardis / R_saha;
    printf("  Implied W × δ factor = %.4f\n", W_delta);

    /* If δ = 1/(ζ + W(1-ζ)) and we know ζ, solve for W */
    /* W_delta = W / (ζ + W(1-ζ)) */
    /* W_delta × (ζ + W - W×ζ) = W */
    /* W_delta × ζ + W_delta × W - W_delta × W × ζ = W */
    /* W_delta × ζ = W - W_delta × W + W_delta × W × ζ */
    /* W_delta × ζ = W × (1 - W_delta + W_delta × ζ) */
    /* W = W_delta × ζ / (1 - W_delta + W_delta × ζ) */
    double W_implied = W_delta * zeta / (1.0 - W_delta + W_delta * zeta);
    printf("  Implied dilution factor W = %.4f\n", W_implied);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <atomic_data.h5> <tardis_plasma_state.h5>\n", argv[0]);
        return 1;
    }

    print_header();

    /* Load atomic data */
    printf("[LOAD] Atomic data: %s\n", argv[1]);
    AtomicData atomic;
    if (atomic_data_load_hdf5(argv[1], &atomic) != 0) {
        fprintf(stderr, "Failed to load atomic data\n");
        return 1;
    }
    printf("[LOAD] Loaded %d ions, %d levels, %ld lines\n",
           atomic.n_ions, atomic.n_levels, (long)atomic.n_lines);

    /* Load TARDIS data */
    printf("[LOAD] TARDIS plasma state: %s\n", argv[2]);
    TardisShellData tardis_shells[30];
    int n_shells = 0;
    if (load_tardis_data(argv[2], tardis_shells, &n_shells) != 0) {
        atomic_data_free(&atomic);
        return 1;
    }
    printf("[LOAD] Loaded %d shells from TARDIS\n", n_shells);

    /* Compare specific shells */
    int test_shells[] = {0, 5, 9, 15, 20, 29};
    int n_test = sizeof(test_shells) / sizeof(test_shells[0]);

    for (int i = 0; i < n_test; i++) {
        int shell = test_shells[i];
        if (shell < n_shells) {
            compare_ionization(&atomic, &tardis_shells[shell], shell);
        }
    }

    /* Summary */
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                          SUMMARY                                  ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Shell    T [K]      TARDIS Si II    LUMINA Si II    Discrepancy  ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");

    for (int s = 0; s < n_shells; s += 5) {
        /* Quick recalculation of LUMINA Si II */
        PlasmaState p;
        plasma_state_init(&p);
        calculate_all_partition_functions(&atomic, tardis_shells[s].T, &p);

        double R[6];
        double sum_R = 1.0;
        R[0] = 1.0;
        double cr = 1.0;

        for (int i = 0; i < 4; i++) {
            double U_i = p.partition_function[14][i];
            double U_i1 = p.partition_function[14][i + 1];
            double chi = atomic_get_ionization_energy(&atomic, 14, i);
            double kT = 1.380649e-16 * tardis_shells[s].T;
            double T32 = tardis_shells[s].T * sqrt(tardis_shells[s].T);
            /* Use SAHA_CONST from plasma_physics.h */
            double ratio_U = (U_i > 0.0) ? (2.0 * U_i1 / U_i) : 2.0;
            double phi = ratio_U * SAHA_CONST * T32 * exp(-chi / kT);
            cr *= (phi / tardis_shells[s].n_e);
            R[i + 1] = cr;
            sum_R += cr;
        }

        double lumina_Si_II = (R[1] / sum_R) * 100;
        double tardis_Si_II = tardis_shells[s].Si_II_frac * 100;
        double disc = fabs(lumina_Si_II - tardis_Si_II);

        printf("║  %3d     %7.0f      %6.2f%%          %6.2f%%        %6.2f%%    ║\n",
               s, tardis_shells[s].T, tardis_Si_II, lumina_Si_II, disc);
    }

    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    atomic_data_free(&atomic);
    return 0;
}
