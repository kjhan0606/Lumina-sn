#include "lumina.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int find_ion(const AtomicData *atom, int Z, int stage) {
    for (int ip = 0; ip < atom->n_ion_pops; ip++)
        if (atom->ion_pop_Z[ip] == Z && atom->ion_pop_stage[ip] == stage)
            return ip;
    return -1;
}

static double legacy_active_tau(const AtomicData *atom, int line,
                                double T_rad, double W,
                                double time_explosion) {
    int ip = find_ion(atom, atom->line_atomic_number[line],
                      atom->line_ion_number[line]);
    if (ip < 0) return 0.0;

    int lower = -1, upper = -1;
    for (int l = atom->level_offset[ip]; l < atom->level_offset[ip + 1]; l++) {
        if (atom->level_num[l] == atom->line_level_lower[line]) lower = l;
        if (atom->level_num[l] == atom->line_level_upper[line]) upper = l;
        if (lower >= 0 && upper >= 0) break;
    }
    if (lower < 0 || upper < 0) return 0.0;

    double beta = 1.0 / (K_BOLTZMANN * T_rad);
    double n_ion = atom->ion_number_density[ip];
    double Z_part = atom->partition_functions[ip];
    double lower_boltz = atom->level_energy_eV[lower] * EV_TO_ERG * beta;
    double upper_boltz = atom->level_energy_eV[upper] * EV_TO_ERG * beta;
    double n_lower = 0.0, n_upper = 0.0;
    if (lower_boltz < 500.0) {
        double weight = atom->level_metastable[lower] ? 1.0 : W;
        n_lower = n_ion * weight * atom->level_g[lower] *
                  exp(-lower_boltz) / Z_part;
    }
    if (upper_boltz < 500.0) {
        double weight = atom->level_metastable[upper] ? 1.0 : W;
        n_upper = n_ion * weight * atom->level_g[upper] *
                  exp(-upper_boltz) / Z_part;
    }
    double stim = 1.0;
    if (n_lower > 0.0 && n_upper > 0.0) {
        stim = 1.0 - (atom->level_g[lower] * n_upper) /
                     (atom->level_g[upper] * n_lower);
        if (stim < 0.0) stim = 0.0;
    }
    double tau = SOBOLEV_COEFF * atom->line_f_lu[line] *
                 atom->line_wavelength_cm[line] * time_explosion *
                 n_lower * stim;
    if (tau < 1e-100) tau = 1e-100;
    return tau;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s REFERENCE_DECK\n", argv[0]);
        return 64;
    }

    enum { DECK_SHELLS = 50 };
    AtomicData atom;
    if (load_atomic_data(&atom, argv[1], DECK_SHELLS) != 0) return 2;

    unsigned char inactive_Z[100] = {0};
    for (int e = 0; e < atom.n_elements; e++) {
        int Z = atom.element_Z[e];
        if (Z > 0 && Z < 100)
            inactive_Z[Z] = (unsigned char)
                lumina_zinert_element_inactive(&atom, e, DECK_SHELLS);
    }

    double *original_abundances = atom.abundances;
    double *original_ions = atom.ion_number_density;
    double *original_partition = atom.partition_functions;
    double *original_partition_Te = atom.partition_functions_Te;
    atom.abundances = calloc((size_t)atom.n_elements, sizeof(double));
    atom.ion_number_density = calloc((size_t)atom.n_ion_pops, sizeof(double));
    atom.partition_functions = calloc((size_t)atom.n_ion_pops, sizeof(double));
    atom.partition_functions_Te = calloc((size_t)atom.n_ion_pops, sizeof(double));
    double *tau = calloc((size_t)atom.n_lines, sizeof(double));
    double *source = calloc((size_t)atom.n_lines, sizeof(double));
    if (!atom.abundances || !atom.ion_number_density ||
        !atom.partition_functions || !atom.partition_functions_Te ||
        !tau || !source) {
        fprintf(stderr, "[Z-INERT-CANONICAL-TAU][FATAL] allocation failed\n");
        return 3;
    }

    /* This is a one-shell execution view of the whole-deck classification.
     * An element which is active in any canonical shell must remain active in
     * the view even if its first physical shell happens to be exactly zero. */
    for (int e = 0; e < atom.n_elements; e++) {
        int Z = atom.element_Z[e];
        atom.abundances[e] =
            (Z > 0 && Z < 100 && inactive_Z[Z]) ? 0.0 : 1.0;
    }
    for (int ip = 0; ip < atom.n_ion_pops; ip++) {
        int Z = atom.ion_pop_Z[ip];
        atom.ion_number_density[ip] =
            (Z > 0 && Z < 100 && inactive_Z[Z]) ? 0.0 : (double)(17 * (ip + 1));
        atom.partition_functions[ip] = 1.0 + 0.125 * (double)(ip % 7);
        atom.partition_functions_Te[ip] = atom.partition_functions[ip];
    }

    double T_rad[1] = {10000.0};
    double W[1] = {0.5};
    PlasmaState plasma;
    OpacityState opacity;
    memset(&plasma, 0, sizeof plasma);
    memset(&opacity, 0, sizeof opacity);
    plasma.n_shells = 1;
    plasma.T_rad = T_rad;
    plasma.W = W;
    opacity.n_lines = atom.n_lines;
    opacity.n_shells = 1;
    opacity.tau_sobolev = tau;
    opacity.line_source_S = source;

    const double time_explosion = 86400.0;
    lumina_oracle_compute_tau_sobolev(&atom, &plasma, &opacity,
                                       time_explosion);

    long inactive_lines = 0, inactive_nonzero = 0;
    long active_lines = 0, active_bit_differences = 0;
    uint64_t active_hash = UINT64_C(14695981039346656037);
    for (int line = 0; line < atom.n_lines; line++) {
        int Z = atom.line_atomic_number[line];
        if (Z > 0 && Z < 100 && inactive_Z[Z]) {
            inactive_lines++;
            if (tau[line] != 0.0) inactive_nonzero++;
            continue;
        }
        double expected = legacy_active_tau(&atom, line, T_rad[0], W[0],
                                            time_explosion);
        active_lines++;
        if (memcmp(&tau[line], &expected, sizeof expected) != 0)
            active_bit_differences++;
        const unsigned char *bytes = (const unsigned char *)&tau[line];
        for (size_t b = 0; b < sizeof tau[line]; b++) {
            active_hash ^= bytes[b];
            active_hash *= UINT64_C(1099511628211);
        }
    }

    int audit_rc = lumina_zinert_validate(&atom, NULL, &opacity, 1,
                                          "canonical-tau-transport");
    printf("[Z-INERT-CANONICAL-TAU] inactive_lines=%ld inactive_nonzero=%ld "
           "active_lines=%ld active_tau_bit_differences=%ld "
           "active_tau_fnv64=%016llx audit_rc=%d verdict=%s\n",
           inactive_lines, inactive_nonzero, active_lines,
           active_bit_differences, (unsigned long long)active_hash, audit_rc,
           (!audit_rc && inactive_nonzero == 0 &&
            active_bit_differences == 0) ? "PASS" : "FAIL");

    free(atom.abundances);
    free(atom.ion_number_density);
    free(atom.partition_functions);
    free(atom.partition_functions_Te);
    free(tau);
    free(source);
    atom.abundances = original_abundances;
    atom.ion_number_density = original_ions;
    atom.partition_functions = original_partition;
    atom.partition_functions_Te = original_partition_Te;
    free_atomic_data(&atom);

    return (audit_rc || inactive_nonzero != 0 ||
            active_bit_differences != 0) ? 1 : 0;
}
