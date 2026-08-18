#include "nlte_population_candidate.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message)
{
    fprintf(stderr, "[NLTE-CANDIDATE-ADIABATIC][FAIL] %s\n", message);
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
    double te[NS] = {10000.0, 12000.0};
    double ne[NS] = {1.0, 2.0};
    double ion_population[NI * NS] = {3.0, 2.0, 1.0, 2.0};
    double partition[NI * NS] = {0};
    double nlte_population[NL * NS] = {2.0, 1.0, 1.0, 1.0};
    double within[NL * NS] = {1.0, 1.0, 1.0, 1.0};
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
    atom.ion_number_density = ion_population;
    atom.partition_functions = partition;
    atom.population_committed_generation = 5;

    nlte.n_nlte_levels_total = NL;
    nlte.global_to_nlte_level = global_to_nlte;
    nlte.nlte_level_populations = nlte_population;
    nlte.within_sl_frac = within;
    nlte.population_committed_generation = 5;
    plasma.n_shells = NS;
    plasma.T_e = te;
    plasma.n_electron = ne;
    plasma.T_e_generation = 4;

    PopulationAtomicView view = {
        NI, NL, level_offset, level_energy_eV, level_g, NULL,
        level_Z, level_ion, 1, topion_index, topion_energy_cm, topion_g
    };
    if (population_partition_build(
            &view, te, NS, 5, 4, partition, &atom.partition_stamp) != POP_OK)
        return fail("partition fixture build");

    AtomicData atom_before = atom;
    NLTEConfig nlte_before = nlte;
    PlasmaState plasma_before = plasma;
    double ion_before[NI * NS], level_before[NL * NS], ne_before[NS];
    memcpy(ion_before, ion_population, sizeof(ion_before));
    memcpy(level_before, nlte_population, sizeof(level_before));
    memcpy(ne_before, ne, sizeof(ne_before));

    NLTEPopulationCandidate candidate;
    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, te, NS, 4, 5) !=
        NLTE_CANDIDATE_OK)
        return fail("candidate begin");
    double radius[NS] = {1.0e14, 2.0e14};
    double velocity[NS] = {1.0e8, 2.0e8};
    if (nlte_population_candidate_prepare_adiabatic(
            &candidate, radius, velocity, 1.0e6) != NLTE_CANDIDATE_OK ||
        !candidate.adiabatic_active)
        return fail("combined internal-energy/adiabatic producer");

    const double ev_to_erg = 1.602176634e-12;
    double expected_u0 = 15.6 * ev_to_erg / 4.0;
    double expected_u1 = 29.2 * ev_to_erg / 4.0;
    if (candidate.n_atom[0] != 4.0 || candidate.n_atom[1] != 4.0 ||
        fabs(candidate.internal_energy_atom[0] - expected_u0) >
            1.0e-14 * expected_u0 ||
        fabs(candidate.internal_energy_atom[1] - expected_u1) >
            1.0e-14 * expected_u1 ||
        !isfinite(candidate.adiabatic[0].signed_total) ||
        !isfinite(candidate.adiabatic[1].signed_total) ||
        candidate.adiabatic[0].signed_total !=
            candidate.adiabatic[0].cooling -
            candidate.adiabatic[0].heating)
        return fail("combined known answer/sign split");

    if (memcmp(&atom, &atom_before, sizeof(atom)) != 0 ||
        memcmp(&nlte, &nlte_before, sizeof(nlte)) != 0 ||
        memcmp(&plasma, &plasma_before, sizeof(plasma)) != 0 ||
        memcmp(ion_population, ion_before, sizeof(ion_before)) != 0 ||
        memcmp(nlte_population, level_before, sizeof(level_before)) != 0 ||
        memcmp(ne, ne_before, sizeof(ne_before)) != 0)
        return fail("combined candidate mutated public state");
    nlte_population_candidate_free(&candidate);

    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, te, NS, 4, 5) !=
            NLTE_CANDIDATE_OK)
        return fail("negative candidate begin");
    double bad_velocity[NS] = {1.0e8, 2.1e8};
    if (nlte_population_candidate_prepare_adiabatic(
            &candidate, radius, bad_velocity, 1.0e6) !=
            NLTE_CANDIDATE_ADIABATIC_FAILED || candidate.active ||
        candidate.n_atom || candidate.internal_energy_atom ||
        candidate.adiabatic)
        return fail("nonhomology did not fail atomically");
    nlte_population_candidate_free(&candidate);

    printf("[NLTE-CANDIDATE-ADIABATIC][SELFTEST] status=PASS "
           "internal_energy=NEUTRAL_GROUND vector=CMFGEN_COMPLETE "
           "sign=SIGNED nonhomology=FAIL_CLOSED public_mutations=0\n");
    return 0;
}
