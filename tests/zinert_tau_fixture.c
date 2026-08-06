#include "lumina.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    int element_z[2] = {6, 14};
    double abundances[2] = {0.0, 1.0};
    int elem_ion_offset[3] = {0, 1, 2};
    int ion_z[2] = {6, 14};
    int ion_stage[2] = {0, 0};
    double ion_density[2] = {0.0, 123.0};
    double partition[2] = {1.0, 1.0};
    int level_offset[3] = {0, 2, 4};
    int level_num[4] = {0, 1, 0, 1};
    double level_energy[4] = {0.0, 1.0, 0.0, 1.0};
    int level_g[4] = {1, 3, 1, 3};
    int level_meta[4] = {0, 0, 0, 0};
    int line_z[4] = {6, 14, 99, 14};
    int line_ion[4] = {0, 0, 0, 0};
    int line_lower[4] = {0, 0, 0, 0};
    int line_upper[4] = {1, 1, 1, 99};
    double line_f[4] = {0.25, 0.25, 0.25, 0.25};
    double line_lam[4] = {5e-5, 5e-5, 5e-5, 5e-5};
    double T_e[1] = {9000.0};
    double tau[4] = {-1.0, -1.0, -1.0, -1.0};

    memset(&atom, 0, sizeof atom);
    memset(&plasma, 0, sizeof plasma);
    memset(&opacity, 0, sizeof opacity);
    atom.n_elements = 2;
    atom.element_Z = element_z;
    atom.abundances = abundances;
    atom.elem_ion_offset = elem_ion_offset;
    atom.n_ion_pops = 2;
    atom.ion_pop_Z = ion_z;
    atom.ion_pop_stage = ion_stage;
    atom.ion_number_density = ion_density;
    atom.partition_functions = partition;
    atom.n_levels = 4;
    atom.level_offset = level_offset;
    atom.level_num = level_num;
    atom.level_energy_eV = level_energy;
    atom.level_g = level_g;
    atom.level_metastable = level_meta;
    atom.line_atomic_number = line_z;
    atom.line_ion_number = line_ion;
    atom.line_level_lower = line_lower;
    atom.line_level_upper = line_upper;
    atom.line_f_lu = line_f;
    atom.line_wavelength_cm = line_lam;
    plasma.n_shells = 1;
    plasma.T_e = T_e;
    plasma.T_e_generation = 1;
    opacity.n_lines = 4;
    opacity.n_shells = 1;
    opacity.tau_sobolev = tau;

    /* Missing T_e must fail closed for the active line without touching the
     * exact-zero inert/missing-map outputs or crashing. */
    plasma.T_e = NULL;
    lumina_oracle_compute_tau_sobolev(&atom, &plasma, &opacity, 86400.0);
    if (tau[0] != 0.0 || !isnan(tau[1]) || tau[2] != 0.0 || tau[3] != 0.0) {
        fprintf(stderr,
                "[Z-INERT-TAU][FATAL] missing-T_e fail-closed outputs "
                "%.17g %.17g %.17g %.17g\n",
                tau[0], tau[1], tau[2], tau[3]);
        return 3;
    }

    plasma.T_e = T_e;
    for (int line = 0; line < 4; line++) tau[line] = -1.0;
    lumina_oracle_compute_tau_sobolev(&atom, &plasma, &opacity, 86400.0);

    double beta = 1.0 / (K_BOLTZMANN * T_e[0]);
    double n_lower = ion_density[1] * level_g[2];
    double n_upper = ion_density[1] * level_g[3] *
                     exp(-level_energy[3] * EV_TO_ERG * beta);
    double stim = 1.0 - (level_g[2] * n_upper) / (level_g[3] * n_lower);
    if (stim < 0.0) stim = 0.0;
    double active_expected = SOBOLEV_COEFF * line_f[1] * line_lam[1] *
                             86400.0 * n_lower * stim;
    if (active_expected < 1e-100) active_expected = 1e-100;

    if (tau[0] != 0.0 || tau[2] != 0.0 || tau[3] != 0.0) {
        fprintf(stderr,
                "[Z-INERT-TAU][FATAL] exact-zero outputs %.17g %.17g %.17g\n",
                tau[0], tau[2], tau[3]);
        return 1;
    }
    if (memcmp(&tau[1], &active_expected, sizeof(double)) != 0) {
        fprintf(stderr,
                "[Z-INERT-TAU][FATAL] active tau differs got=%.17g expected=%.17g\n",
                tau[1], active_expected);
        return 2;
    }
    printf("[Z-INERT-TAU] missing_te=FAIL_CLOSED inactive_valid=0 "
           "missing_ion=0 missing_level=0 "
           "active_tau_bits=IDENTICAL value=%.17g PASS\n", tau[1]);
    return 0;
}
