#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const double sentinel = 12345.6789;
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));
    memset(&opacity, 0, sizeof(opacity));

    int element_z[1] = {1};
    double element_mass[1] = {1.0};
    double abundance[1] = {1.0};
    int elem_ion_offset[2] = {0, 1};
    int ion_z[1] = {1};
    int ion_stage[1] = {1};
    int level_offset[2] = {0, 2};
    int level_num[2] = {0, 1};
    double level_energy[2] = {0.0, 1.0};
    int level_g[2] = {2, 4};
    int level_meta[2] = {1, 0};
    double partition[1] = {0.0};
    double ion_density[1] = {0.0};
    int line_z[1] = {1};
    int line_ion[1] = {1};
    int line_lower[1] = {0};
    int line_upper[1] = {1};
    double line_f[1] = {0.5};
    double line_lambda[1] = {1.0e-5};
    double line_nu[1] = {2.99792458e15};

    atom.n_elements = 1;
    atom.element_Z = element_z;
    atom.element_mass_amu = element_mass;
    atom.abundances = abundance;
    atom.elem_ion_offset = elem_ion_offset;
    atom.n_ion_pops = 1;
    atom.ion_pop_Z = ion_z;
    atom.ion_pop_stage = ion_stage;
    atom.level_offset = level_offset;
    atom.level_num = level_num;
    atom.level_energy_eV = level_energy;
    atom.level_g = level_g;
    atom.level_metastable = level_meta;
    atom.partition_functions = partition;
    atom.ion_number_density = ion_density;
    atom.n_lines = 1;
    atom.line_atomic_number = line_z;
    atom.line_ion_number = line_ion;
    atom.line_level_lower = line_lower;
    atom.line_level_upper = line_upper;
    atom.line_f_lu = line_f;
    atom.line_wavelength_cm = line_lambda;
    atom.line_nu = line_nu;

    double W[1] = {0.5};
    double T_rad[1] = {10000.0};
    double rho[1] = {1.0e-14};
    double n_e[1] = {1.0e8};
    double T_e[1] = {9000.0};
    plasma.n_shells = 1;
    plasma.W = W;
    plasma.T_rad = T_rad;
    plasma.rho = rho;
    plasma.n_electron = n_e;
    plasma.T_e = T_e;

    double tau[1] = {sentinel};
    double opacity_ne[1] = {1.0e8};
    opacity.n_lines = 1;
    opacity.n_shells = 1;
    opacity.tau_sobolev = tau;
    opacity.electron_density = opacity_ne;
    opacity.tau_required_generation = 1;

    printf("KFRESH_BEFORE sentinel=%.17g computed_generation=%llu "
           "required_generation=%llu\n", tau[0],
           (unsigned long long)opacity.tau_computed_generation,
           (unsigned long long)opacity.tau_required_generation);
    if (lumina_prepare_solver_owned_tau(&atom, &plasma, &opacity, 86400.0,
                                        "CPU harness first consumer") != 0)
        return 1;
    printf("KFRESH_AFTER tau=%.17g sentinel_reached=%s computed_generation=%llu "
           "required_generation=%llu first_consumer_generation=%llu\n",
           tau[0], tau[0] == sentinel ? "YES" : "NO",
           (unsigned long long)opacity.tau_computed_generation,
           (unsigned long long)opacity.tau_required_generation,
           (unsigned long long)opacity.tau_first_consumer_generation);
    return tau[0] == sentinel ? 1 : 0;
}
