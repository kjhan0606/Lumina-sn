#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__real_malloc(size_t);
int nlte_precompute_within_sl_frac_checked(
    NLTEConfig *, AtomicData *, PlasmaState *, int);
static int fail_matching_allocation;
static int matching_allocation_count;
void *__wrap_malloc(size_t n) {
    if (fail_matching_allocation && n == sizeof(double) &&
        ++matching_allocation_count == fail_matching_allocation)
        return NULL;
    return __real_malloc(n);
}

int main(void) {
    NLTEConfig nlte;
    AtomicData atom;
    PlasmaState plasma;
    memset(&nlte, 0, sizeof(nlte));
    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));

    int level_offset[2] = {0, 1};
    double energy_eV[1] = {0.0};
    int level_g[1] = {2};
    double partition[1] = {NAN};
    double temperature[1] = {10000.0};
    double electron_density[1] = {1.0};
    double fraction[1] = {0.25};
    int nlte_to_global[1] = {0};
    int full_to_super[1] = {0};
    int super_anchor[1] = {0};

    atom.n_ion_pops = 1;
    atom.n_levels = 1;
    atom.level_offset = level_offset;
    atom.level_energy_eV = energy_eV;
    atom.level_g = level_g;
    atom.partition_functions = partition;
    plasma.n_shells = 1;
    plasma.T_e = temperature;
    plasma.T_e_generation = 1;
    plasma.n_electron = electron_density;
    nlte.n_nlte_levels_total = 1;
    nlte.super_mode = 1;
    nlte.n_super_total = 1;
    nlte.nlte_to_global_level = nlte_to_global;
    nlte.fl_to_super = full_to_super;
    nlte.super_anchor_global = super_anchor;
    nlte.within_sl_frac = fraction;

    PopulationAtomicView view = {
        .n_ions = 1,
        .n_levels = 1,
        .level_offset = level_offset,
        .energy_eV = energy_eV,
        .g = level_g
    };
    if (population_partition_build(
            &view, temperature, 1, 1, 1, partition,
            &atom.partition_stamp) != POP_OK)
        return 2;

    PopulationDerivedStamp stamp_sentinel;
    memset(&stamp_sentinel, 0xA5, sizeof(stamp_sentinel));
    double fraction_sentinel = 0.25;
    int failure_rc[2] = {0, 0};
    int byte_preserved[2] = {0, 0};
    for (int ordinal = 1; ordinal <= 2; ++ordinal) {
        fraction[0] = fraction_sentinel;
        nlte.within_sl_stamp = stamp_sentinel;
        matching_allocation_count = 0;
        fail_matching_allocation = ordinal;
        failure_rc[ordinal - 1] = nlte_precompute_within_sl_frac_checked(
            &nlte, &atom, &plasma, 1);
        byte_preserved[ordinal - 1] =
            fraction[0] == fraction_sentinel &&
            memcmp(&nlte.within_sl_stamp, &stamp_sentinel,
                   sizeof(stamp_sentinel)) == 0;
    }

    fail_matching_allocation = 0;
    matching_allocation_count = 0;
    nlte.super_mode = 0;
    int normal_rc = nlte_precompute_within_sl_frac(
        &nlte, &atom, &plasma, 1);
    printf("work_oom_rc=%d super_partition_oom_rc=%d "
           "failure_bytes=%d/%d normal_legacy_rc=%d normal_fraction=%.1f\n",
           failure_rc[0], failure_rc[1], byte_preserved[0],
           byte_preserved[1], normal_rc, fraction[0]);
    return failure_rc[0] == -1 && failure_rc[1] == -1 &&
           byte_preserved[0] && byte_preserved[1] && normal_rc == 0 &&
           fraction[0] == 1.0 ? 0 : 1;
}
