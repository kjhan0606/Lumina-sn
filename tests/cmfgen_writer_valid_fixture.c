#define _POSIX_C_SOURCE 200809L
#include "cmfgen_final_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void stamp(PopulationDerivedStamp *s, const char *hash, size_t items) {
    memset(s, 0, sizeof(*s));
    s->required_population_generation = s->computed_population_generation = 7;
    s->te_generation = 11; s->n_shells = 2; s->n_items = items; s->status = POP_OK;
    memcpy(s->te_manifest_sha256, hash, 65);
}

int main(int argc, char **argv) {
    static const int zs[6] = {20, 27, 26, 28, 14, 16};
    static const double masses[6] = {56, 59, 58, 60, 56, 56};
    static const char hash[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    double te[2] = {10000, 12000}, ne[2] = {1.1, 1.1};
    int iz[18], stage[18], to_global[12], level_g[12];
    double ion[36], levels[24], energy[12];
    PlasmaState p = {0}; AtomicData a = {0}; NLTEConfig n = {0};
    if (argc != 2) return 2;
    for (int e = 0; e < 6; ++e) {
        double total = 5.6 / masses[e];
        for (int q = 0; q < 3; ++q) {
            int i = 3 * e + q; double fraction = q == 0 ? .2 : q == 1 ? .3 : .5;
            iz[i] = zs[e]; stage[i] = q + 1;
            ion[2 * i] = ion[2 * i + 1] = total * fraction;
        }
        for (int q = 0; q < 2; ++q) {
            int i = 2 * e + q, ip = 3 * e + q;
            n.nlte_Z[i] = zs[e]; n.nlte_ion[i] = q + 1;
            n.nlte_ion_level_offset[i] = i; to_global[i] = i; level_g[i] = 2; energy[i] = q * 10;
            levels[2 * i] = ion[2 * ip]; levels[2 * i + 1] = ion[2 * ip + 1];
        }
    }
    n.nlte_ion_level_offset[12] = 12;
    p.n_shells = 2; p.T_e = te; p.n_electron = ne; p.T_e_generation = 11;
    p.te_publication.required_te_generation = p.te_publication.committed_te_generation = 11;
    memcpy(p.te_publication.te_manifest_sha256, hash, 65);
    a.n_ion_pops = 18; a.n_levels = 12; a.ion_pop_Z = iz; a.ion_pop_stage = stage;
    a.ion_number_density = ion; a.level_energy_eV = energy; a.level_g = level_g;
    a.population_committed_generation = 7; stamp(&a.partition_stamp, hash, 18);
    n.n_nlte_ions = 12; n.n_nlte_levels_total = 12; n.nlte_to_global_level = to_global;
    n.nlte_level_populations = levels; n.population_required_generation = n.population_committed_generation = 7;
    stamp(&n.within_sl_stamp, hash, 12);
    setenv("LUMINA_ION_POP_DUMP", "1", 1); setenv("LUMINA_NLTE_LEVEL_DUMP", "1", 1);
    return cmfgen_final_writer(argv[1], &p, &a, &n, 2) == 0 ? 0 : 1;
}
