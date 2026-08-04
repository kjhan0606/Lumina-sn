#include "lumina.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    enum { NSHELL = 2 };
    AtomicData atom;
    NLTEConfig nlte;
    OpacityState opacity;
    int element_z[2] = {6, 14};
    double abundance[4] = {0.0, 0.0, 1.0, 1.0};
    int element_ion_offset[3] = {0, 2, 3};
    double ion_population[6] = {0.0, 0.0, 0.0, 0.0, 7.0, 9.0};
    double level_population[6] = {0.0, 0.0, 0.0, 0.0, 11.0, 13.0};
    int line_z[2] = {6, 14};
    double tau[4] = {0.0, 0.0, 2.0, 3.0};
    double source[4] = {0.0, 0.0, 5.0, 6.0};
    int inject = argc == 2 && strcmp(argv[1], "--inject-phantom") == 0;

    memset(&atom, 0, sizeof atom);
    memset(&nlte, 0, sizeof nlte);
    memset(&opacity, 0, sizeof opacity);
    atom.n_elements = 2;
    atom.element_Z = element_z;
    atom.abundances = abundance;
    atom.elem_ion_offset = element_ion_offset;
    atom.ion_number_density = ion_population;
    atom.line_atomic_number = line_z;
    nlte.n_nlte_ions = 2;
    nlte.nlte_Z[0] = 6;
    nlte.nlte_Z[1] = 14;
    nlte.nlte_ion_level_offset[0] = 0;
    nlte.nlte_ion_level_offset[1] = 2;
    nlte.nlte_ion_level_offset[2] = 3;
    nlte.nlte_level_populations = level_population;
    opacity.n_lines = 2;
    opacity.n_shells = NSHELL;
    opacity.tau_sobolev = tau;
    opacity.line_source_S = source;

    if (inject) {
        ion_population[1] = 1e-77;
        level_population[2] = 1e-88;
        tau[0] = 1e-100;
        source[1] = 1e-99;
    }

    double active_before[6] = {
        ion_population[4], ion_population[5], level_population[4],
        level_population[5], tau[2], source[3]
    };
    int rc = lumina_zinert_validate(&atom, &nlte, &opacity, NSHELL,
                                    inject ? "phantom-negative" : "exact-zero-positive");
    double active_after[6] = {
        ion_population[4], ion_population[5], level_population[4],
        level_population[5], tau[2], source[3]
    };
    if (memcmp(active_before, active_after, sizeof(active_before)) != 0) {
        fputs("[Z-INERT-SELFTEST][FATAL] validator changed active bytes\n", stderr);
        return 2;
    }
    printf("[Z-INERT-SELFTEST] case=%s validator_rc=%d active_bytes=IDENTICAL\n",
           inject ? "phantom-negative" : "exact-zero-positive", rc);
    return rc ? 1 : 0;
}
