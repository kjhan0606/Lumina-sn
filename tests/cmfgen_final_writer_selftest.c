#define _POSIX_C_SOURCE 200809L
#include "cmfgen_final_writer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FINAL_WRITER_SELFTEST_FAIL line=%d expr=%s\n", __LINE__, #x); return 1; } } while (0)

static int contains(const char *path, const char *needle) {
    FILE *fp = fopen(path, "rb"); char buf[16384]; size_t n;
    if (!fp) return 0;
    n = fread(buf, 1, sizeof(buf) - 1, fp); fclose(fp); buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}
static void stamp(PopulationDerivedStamp *s, uint64_t pg, uint64_t tg,
                  const char hash[65], size_t items) {
    memset(s, 0, sizeof(*s)); s->required_population_generation = pg;
    s->computed_population_generation = pg; s->te_generation = tg;
    memcpy(s->te_manifest_sha256, hash, 65); s->n_shells = 2;
    s->n_items = items; s->status = POP_OK;
}

int main(int argc, char **argv) {
    static const char hash[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    char root[] = "/tmp/cmfgen-final-writer.XXXXXX", path[2048], disabled[256], stale[256];
    double te[] = {12345.25, 23456.5}, ne[] = {2.5, 3.5};
    int iz[] = {26, 26}, istage[] = {1, 2}; double ion[] = {-1.25, NAN, 3.5, 4.5};
    int to_global[] = {0, 1}, level_g[] = {2, 4}; double energy[] = {0.0, 1.25};
    double levels[] = {-0.0, -7.25, INFINITY, NAN};
    PlasmaState p = {0}, p0; AtomicData a = {0}, a0; NLTEConfig n = {0}, n0;
    double te0[2], ne0[2], ion0[4], levels0[4];
    const char *output_root;
    if (argc > 2) return 2;
    if (argc == 2) { output_root = argv[1]; CHECK(access(output_root, F_OK) == 0); }
    else { CHECK(mkdtemp(root) != NULL); output_root = root; }
    snprintf(disabled, sizeof(disabled), "%s/disabled", output_root); CHECK(mkdir(disabled, 0700) == 0);
    snprintf(stale, sizeof(stale), "%s/stale", output_root); CHECK(mkdir(stale, 0700) == 0);
    p.n_shells = 2; p.T_e = te; p.n_electron = ne; p.T_e_generation = 11;
    p.te_publication.required_te_generation = 11; p.te_publication.committed_te_generation = 11;
    memcpy(p.te_publication.te_manifest_sha256, hash, 65);
    a.n_ion_pops = 2; a.n_levels = 2; a.ion_pop_Z = iz; a.ion_pop_stage = istage;
    a.ion_number_density = ion; a.population_committed_generation = 7;
    a.level_energy_eV = energy; a.level_g = level_g; stamp(&a.partition_stamp, 7, 11, hash, 2);
    n.n_nlte_ions = 1; n.nlte_Z[0] = 26; n.nlte_ion[0] = 2;
    n.nlte_ion_level_offset[0] = 0; n.nlte_ion_level_offset[1] = 2;
    n.n_nlte_levels_total = 2; n.nlte_to_global_level = to_global;
    n.nlte_level_populations = levels; n.population_required_generation = 7;
    n.population_committed_generation = 7; stamp(&n.within_sl_stamp, 7, 11, hash, 2);
    p0 = p; a0 = a; n0 = n; memcpy(te0, te, sizeof te); memcpy(ne0, ne, sizeof ne);
    memcpy(ion0, ion, sizeof ion); memcpy(levels0, levels, sizeof levels);

    unsetenv("LUMINA_ION_POP_DUMP"); unsetenv("LUMINA_NLTE_LEVEL_DUMP");
    CHECK(cmfgen_final_writer(disabled, &p, &a, &n, 2) == 0);
    snprintf(path, sizeof(path), "%s/cmfgen_final_publication.json", disabled); CHECK(access(path, F_OK) != 0);
    snprintf(path, sizeof(path), "%s/lumina_plasma_state.csv", disabled);
    CHECK(contains(path, "shell_id,T_e,n_e\n") &&
          contains(path, "0,12345.250000,2.500000e+00\n") &&
          contains(path, "1,23456.500000,3.500000e+00\n"));
    setenv("LUMINA_ION_POP_DUMP", "1", 1); setenv("LUMINA_NLTE_LEVEL_DUMP", "1", 1);
    CHECK(cmfgen_final_writer(output_root, &p, &a, &n, 2) == 0);
    snprintf(path, sizeof(path), "%s/cmfgen_final_publication.json", output_root);
    CHECK(contains(path, "\"complete\": true") && contains(path, "\"population_generation\": 7") &&
          contains(path, "\"te_generation\": 11") && contains(path, "\"rows\": 4"));
    snprintf(path, sizeof(path), "%s/nlte_levels_final.csv", output_root);
    CHECK(contains(path, "-7.25") && contains(path, "inf") && contains(path, "nan") &&
          contains(path, ",7,7,11,"));
    snprintf(path, sizeof(path), "%s/lumina_ion_pops.csv", output_root); CHECK(contains(path, "-1.25"));
    CHECK(memcmp(&p, &p0, sizeof p) == 0 && memcmp(&a, &a0, sizeof a) == 0 && memcmp(&n, &n0, sizeof n) == 0);
    CHECK(memcmp(te, te0, sizeof te) == 0 && memcmp(ne, ne0, sizeof ne) == 0 &&
          memcmp(ion, ion0, sizeof ion) == 0 && memcmp(levels, levels0, sizeof levels) == 0);
    CHECK(cmfgen_final_writer(output_root, &p, &a, &n, 2) != 0); /* collision */
    n.population_required_generation = 8;
    CHECK(cmfgen_final_writer(stale, &p, &a, &n, 2) != 0);
    n.population_required_generation = 7;
    CHECK(cmfgen_final_writer("/path/that/does/not/exist", &p, &a, &n, 2) != 0);
    CHECK(memcmp(te, te0, sizeof te) == 0 && memcmp(ne, ne0, sizeof ne) == 0 &&
          memcmp(ion, ion0, sizeof ion) == 0 && memcmp(levels, levels0, sizeof levels) == 0);
    printf("CMFGEN_FINAL_WRITER_SELFTEST_PASS population_generation=7 te_generation=11 plasma_rows=2 saha_rows=4 nlte_rows=4 bytes_unchanged=1 negative_nan_inf_preserved=1 disabled_legacy_plasma=1\n");
    return 0;
}
