#include "lumina.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    memset(&atom, 0, sizeof atom);
    memset(&plasma, 0, sizeof plasma);
    memset(&opacity, 0, sizeof opacity);

    int element_Z[2] = {6, 14};
    double element_mass[2] = {12.0, 28.0};
    double abundance[2] = {0.0, 1.0};
    int elem_ion_offset[3] = {0, 2, 3};
    int ion_Z[3] = {6, 6, 14};
    int ion_stage[3] = {0, 1, 1};
    double ion_density[3] = {-1.0, -1.0, -1.0};
    double partition[3] = {0.0, 0.0, 0.0};
    int level_offset[4] = {0, 1, 2, 3};
    int level_Z[3] = {6, 6, 14};
    int level_ion[3] = {0, 1, 1};
    int level_num[3] = {0, 0, 0};
    double level_energy[3] = {0.0, 0.0, 0.0};
    int level_g[3] = {1, 1, 1};
    int level_meta[3] = {1, 1, 1};

    double W[1] = {0.5};
    double T_rad[1] = {10000.0};
    double rho[1] = {28.0 * AMU * 123.0};
    double n_e[1] = {1e8};
    double T_e[1] = {9000.0};
    double opacity_ne[1] = {0.0};
    double tau_dummy[1] = {0.0};

    atom.n_elements = 2;
    atom.element_Z = element_Z;
    atom.element_mass_amu = element_mass;
    atom.abundances = abundance;
    atom.elem_ion_offset = elem_ion_offset;
    atom.n_ion_pops = 3;
    atom.ion_pop_Z = ion_Z;
    atom.ion_pop_stage = ion_stage;
    atom.ion_number_density = ion_density;
    atom.partition_functions = partition;
    atom.n_levels = 3;
    atom.level_offset = level_offset;
    atom.level_Z = level_Z;
    atom.level_ion = level_ion;
    atom.level_num = level_num;
    atom.level_energy_eV = level_energy;
    atom.level_g = level_g;
    atom.level_metastable = level_meta;
    plasma.n_shells = 1;
    plasma.W = W;
    plasma.T_rad = T_rad;
    plasma.rho = rho;
    plasma.n_electron = n_e;
    plasma.T_e = T_e;
    plasma.T_e_generation = 1;
    opacity.n_lines = 0;
    opacity.n_shells = 1;
    opacity.tau_sobolev = tau_dummy;
    opacity.electron_density = opacity_ne;
    opacity.tau_required_generation = 1;

    if (lumina_prepare_solver_owned_tau(&atom, &plasma, &opacity, 86400.0,
                                        "Z-INERT population fixture") != 0)
        return 1;

    double active_ground_legacy = (abundance[1] * rho[0]) /
                                  (element_mass[1] * AMU);
    int active_identical =
        memcmp(&ion_density[2], &active_ground_legacy, sizeof(double)) == 0;
    int inactive_zero = ion_density[0] == 0.0 && ion_density[1] == 0.0;
    int audit_rc = lumina_zinert_validate(&atom, NULL, &opacity, 1,
                                          "population-fixture");

    printf("[Z-INERT-POP] inactive_ground=%.17g inactive_upper=%.17g "
           "active_ground_bits=%s active_ground=%.17g audit_rc=%d verdict=%s\n",
           ion_density[0], ion_density[1],
           memcmp(&ion_density[2], &active_ground_legacy,
                  sizeof(double)) == 0 ? "IDENTICAL" : "DIFFERENT",
           ion_density[2], audit_rc,
           (inactive_zero && active_identical && audit_rc == 0) ?
               "PASS" : "FAIL");
    return (inactive_zero && active_identical && audit_rc == 0) ? 0 : 1;
}
