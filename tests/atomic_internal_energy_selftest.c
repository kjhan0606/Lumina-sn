#include "atomic_internal_energy.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message)
{
    fprintf(stderr, "[ATOMIC-INTERNAL-ENERGY][FAIL] %s\n", message);
    return 1;
}

int main(void)
{
    enum { NS = 2, NI = 2, NL = 2 };
    AtomicData atom;
    NLTEConfig nlte;
    PlasmaState plasma;
    memset(&atom, 0, sizeof(atom));
    memset(&nlte, 0, sizeof(nlte));
    memset(&plasma, 0, sizeof(plasma));

    int ion_Z[NI] = {1, 1};
    int ion_stage[NI] = {0, 1};
    int level_offset[NI + 1] = {0, 2, 2};
    int level_Z[NL] = {1, 1};
    int level_ion[NL] = {0, 0};
    int level_g[NL] = {2, 4};
    double level_energy_eV[NL] = {5.0, 7.0};
    int topion_index[1] = {1};
    double topion_energy_cm[1] = {100.0};
    double topion_g[1] = {1.0};
    int ioniz_Z[1] = {1};
    int ioniz_stage[1] = {0};
    double ioniz_energy[1] = {13.6};
    int ioniz_ref_Z[1] = {1};
    int ioniz_ref_stage[1] = {0};
    double ioniz_ref_energy[1] = {99.0};
    double te[NS] = {10000.0, 12000.0};
    double ion_population[NI * NS] = {3.0, 2.0, 1.0, 2.0};
    double partition[NI * NS] = {0};
    double nlte_population[NL * NS] = {2.0, 1.0, 1.0, 1.0};
    int global_to_nlte[NL] = {0, 1};

    atom.n_ion_pops = NI;
    atom.n_levels = NL;
    atom.ion_pop_Z = ion_Z;
    atom.ion_pop_stage = ion_stage;
    atom.level_offset = level_offset;
    atom.level_Z = level_Z;
    atom.level_ion = level_ion;
    atom.level_g = level_g;
    atom.level_energy_eV = level_energy_eV;
    atom.topion_n = 1;
    atom.topion_ion_index = topion_index;
    atom.topion_E_cm = topion_energy_cm;
    atom.topion_g = topion_g;
    atom.n_ionization = 1;
    atom.ioniz_Z = ioniz_Z;
    atom.ioniz_ion = ioniz_stage;
    atom.ioniz_energy_eV = ioniz_energy;
    atom.n_ionization_reference = 1;
    atom.ioniz_ref_Z = ioniz_ref_Z;
    atom.ioniz_ref_ion = ioniz_ref_stage;
    atom.ioniz_ref_energy_eV = ioniz_ref_energy;
    atom.ion_number_density = ion_population;
    atom.partition_functions = partition;
    atom.population_committed_generation = 5;

    nlte.n_nlte_levels_total = NL;
    nlte.global_to_nlte_level = global_to_nlte;
    nlte.nlte_level_populations = nlte_population;
    nlte.population_committed_generation = 5;
    plasma.n_shells = NS;
    plasma.T_e = te;
    plasma.T_e_generation = 4;

    PopulationAtomicView view = {
        NI, NL, level_offset, level_energy_eV, level_g, NULL,
        level_Z, level_ion, 1, topion_index, topion_energy_cm, topion_g
    };
    if (population_partition_build(
            &view, te, NS, 5, 4, partition, &atom.partition_stamp) != POP_OK)
        return fail("partition fixture build");

    AtomicInternalEnergyCell output[NS];
    memset(output, 0xA5, sizeof(output));
    if (atomic_internal_energy_build(
            &atom, &nlte, &plasma, NS, 5, 4, output) !=
        ATOMIC_INTERNAL_ENERGY_OK)
        return fail("valid state rejected");
    const double ev_to_erg = 1.602176634e-12;
    double expected_e0 = 15.6 * ev_to_erg;
    double expected_e1 = 29.2 * ev_to_erg;
    if (output[0].n_atom_cm3 != 4.0 || output[1].n_atom_cm3 != 4.0 ||
        fabs(output[0].energy_density_erg_cm3 - expected_e0) >
            1.0e-14 * expected_e0 ||
        fabs(output[1].energy_density_erg_cm3 - expected_e1) >
            1.0e-14 * expected_e1 ||
        fabs(output[0].internal_energy_atom_erg - expected_e0 / 4.0) >
            1.0e-14 * expected_e0 ||
        fabs(output[1].internal_energy_atom_erg - expected_e1 / 4.0) >
            1.0e-14 * expected_e1)
        return fail("regular ionization did not override reference duplicate");

    /* Active-only ladders may omit q=0.  The separate energy-zero catalog
     * must reproduce the same known answer without restoring that row to the
     * regular rate/population table. */
    ioniz_ref_energy[0] = 13.6;
    atom.n_ionization = 0;
    memset(output, 0xA5, sizeof(output));
    if (atomic_internal_energy_build(
            &atom, &nlte, &plasma, NS, 5, 4, output) !=
            ATOMIC_INTERNAL_ENERGY_OK ||
        fabs(output[0].energy_density_erg_cm3 - expected_e0) >
            1.0e-14 * expected_e0 ||
        fabs(output[1].energy_density_erg_cm3 - expected_e1) >
            1.0e-14 * expected_e1)
        return fail("energy-zero reference fallback known answer");

    AtomicInternalEnergyCell before[NS];
    memcpy(before, output, sizeof(before));
    atom.n_ionization_reference = 0;
    if (atomic_internal_energy_build(
            &atom, &nlte, &plasma, NS, 5, 4, output) !=
            ATOMIC_INTERNAL_ENERGY_MISSING_IONIZATION ||
        memcmp(output, before, sizeof(before)) != 0)
        return fail("missing ionization did not fail atomically");
    atom.n_ionization = 1;
    atom.n_ionization_reference = 1;
    nlte_population[0] = 1.0;
    if (atomic_internal_energy_build(
            &atom, &nlte, &plasma, NS, 5, 4, output) !=
            ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE ||
        memcmp(output, before, sizeof(before)) != 0)
        return fail("population nonclosure did not fail atomically");

    /* Single-total SE owns the partition between two fully mapped stages.
     * Their individual sums may differ from the upstream ion estimates, but
     * the element total must remain exact. */
    int pair_level_offset[NI + 1] = {0, 1, 2};
    int pair_level_Z[NL] = {1, 1};
    int pair_level_ion[NL] = {0, 1};
    int pair_level_g[NL] = {1, 1};
    double pair_level_energy[NL] = {0.0, 0.0};
    int pair_global_to_nlte[NL] = {0, 1};
    double pair_level_population[NL * NS] = {2.5, 2.0, 1.5, 2.0};
    atom.level_offset = pair_level_offset;
    atom.level_Z = pair_level_Z;
    atom.level_ion = pair_level_ion;
    atom.level_g = pair_level_g;
    atom.level_energy_eV = pair_level_energy;
    atom.topion_n = 0;
    nlte.global_to_nlte_level = pair_global_to_nlte;
    nlte.nlte_level_populations = pair_level_population;
    /* The upstream estimate may call a stage exactly absent; the fully
     * mapped SE level population is still the stage owner. */
    ion_population[0] = 0.0;
    ion_population[2] = 4.0;
    PopulationAtomicView pair_view = {
        NI, NL, pair_level_offset, pair_level_energy, pair_level_g, NULL,
        pair_level_Z, pair_level_ion, 0, NULL, NULL, NULL
    };
    if (population_partition_build(
            &pair_view, te, NS, 5, 4, partition,
            &atom.partition_stamp) != POP_OK)
        return fail("single-total pair partition fixture build");
    memset(output, 0xA5, sizeof(output));
    if (atomic_internal_energy_build(
            &atom, &nlte, &plasma, NS, 5, 4, output) !=
            ATOMIC_INTERNAL_ENERGY_OK ||
        output[0].n_atom_cm3 != 4.0 || output[1].n_atom_cm3 != 4.0 ||
        fabs(output[0].energy_density_erg_cm3 - 20.4 * ev_to_erg) >
            1.0e-14 * 20.4 * ev_to_erg ||
        fabs(output[1].energy_density_erg_cm3 - 27.2 * ev_to_erg) >
            1.0e-14 * 27.2 * ev_to_erg)
        return fail("single-total pair element closure known answer");
    memcpy(before, output, sizeof(before));
    pair_level_population[2] = 1.0;
    if (atomic_internal_energy_build(
            &atom, &nlte, &plasma, NS, 5, 4, output) !=
            ATOMIC_INTERNAL_ENERGY_POPULATION_CLOSURE ||
        memcmp(output, before, sizeof(before)) != 0)
        return fail("single-total element nonclosure did not roll back");

    printf("[ATOMIC-INTERNAL-ENERGY][SELFTEST] status=PASS "
           "reference=NEUTRAL_GROUND excitation=PASS ionization=PASS "
           "topion=PASS stage_partition=LEVEL_SE "
           "upstream_zero_stage=PASS element_closure=PASS "
           "closure=FAIL_CLOSED output_rollback=PASS\n");
    return 0;
}
