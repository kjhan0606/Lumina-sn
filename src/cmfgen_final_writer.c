#define _POSIX_C_SOURCE 200809L
#include "cmfgen_final_writer.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { FINAL_PATH_CAP = 1024 };

static int enabled(void) {
    const char *ion = getenv("LUMINA_ION_POP_DUMP");
    const char *level = getenv("LUMINA_NLTE_LEVEL_DUMP");
    return ion && atoi(ion) != 0 && level && atoi(level) != 0;
}
static int hex64(const char value[65]) {
    if (!value || value[64] != '\0') return 0;
    for (int i = 0; i < 64; ++i)
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f'))) return 0;
    return 1;
}
static int path_make(char out[FINAL_PATH_CAP], const char *dir, const char *name) {
    int n = snprintf(out, FINAL_PATH_CAP, "%s/%s", dir, name);
    return n > 0 && n < FINAL_PATH_CAP ? 0 : -1;
}
static FILE *exclusive_file(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0640);
    if (fd < 0) return NULL;
    FILE *fp = fdopen(fd, "w");
    if (!fp) close(fd);
    return fp;
}
static int finish(FILE *fp) {
    if (fflush(fp) != 0) { fclose(fp); return -1; }
    if (fsync(fileno(fp)) != 0) { fclose(fp); return -1; }
    return fclose(fp) == 0 ? 0 : -1;
}
static int write_legacy_plasma(const char *dir, const PlasmaState *p, int ns) {
    char path[FINAL_PATH_CAP];
    FILE *fp;
    if (!dir || !p || ns <= 0 || p->n_shells != ns || !p->T_e ||
        !p->n_electron || path_make(path, dir, "lumina_plasma_state.csv")) return -1;
    fp = fopen(path, "w");
    if (!fp) return -1;
    fprintf(fp, "shell_id,T_e,n_e\n");
    for (int s = 0; s < ns; ++s)
        fprintf(fp, "%d,%.6f,%.6e\n", s, p->T_e[s], p->n_electron[s]);
    return finish(fp);
}
static int validate(const PlasmaState *p, const AtomicData *a,
                    const NLTEConfig *n, int ns) {
    uint64_t pg, tg;
    if (!p || !a || !n || ns <= 0 || p->n_shells != ns || !p->T_e ||
        !p->n_electron || a->n_ion_pops <= 0 || !a->ion_pop_Z ||
        !a->ion_pop_stage || !a->ion_number_density || n->n_nlte_ions <= 0 ||
        n->n_nlte_ions > NLTE_MAX_IONS || n->n_nlte_levels_total <= 0 ||
        !n->nlte_to_global_level || !n->nlte_level_populations ||
        !a->level_energy_eV || !a->level_g) return -1;
    pg = a->population_committed_generation;
    tg = p->te_publication.committed_te_generation;
    if (!pg || !tg || n->population_required_generation != pg ||
        n->population_committed_generation != pg ||
        a->partition_stamp.status != POP_OK ||
        a->partition_stamp.required_population_generation != pg ||
        a->partition_stamp.computed_population_generation != pg ||
        n->within_sl_stamp.status != POP_OK ||
        n->within_sl_stamp.required_population_generation != pg ||
        n->within_sl_stamp.computed_population_generation != pg ||
        p->T_e_generation != tg || p->te_publication.required_te_generation != tg ||
        a->partition_stamp.te_generation != tg || n->within_sl_stamp.te_generation != tg ||
        !hex64(p->te_publication.te_manifest_sha256) ||
        strcmp(a->partition_stamp.te_manifest_sha256, p->te_publication.te_manifest_sha256) ||
        strcmp(n->within_sl_stamp.te_manifest_sha256, p->te_publication.te_manifest_sha256)) return -1;
    for (int i = 0; i < n->n_nlte_ions; ++i) {
        int b = n->nlte_ion_level_offset[i], e = n->nlte_ion_level_offset[i + 1];
        if (b < 0 || e < b || e > n->n_nlte_levels_total) return -1;
    }
    return 0;
}

int cmfgen_final_writer(const char *dir, const PlasmaState *p,
                        const AtomicData *a, const NLTEConfig *n, int ns) {
    char pp[FINAL_PATH_CAP], ipath[FINAL_PATH_CAP], lp[FINAL_PATH_CAP], mp[FINAL_PATH_CAP];
    FILE *fp; size_t pr = 0, ir = 0, lr = 0;
    if (!enabled()) return write_legacy_plasma(dir, p, ns);
    if (!dir || validate(p, a, n, ns) ||
        path_make(pp, dir, "lumina_plasma_state.csv") ||
        path_make(ipath, dir, "lumina_ion_pops.csv") ||
        path_make(lp, dir, "nlte_levels_final.csv") ||
        path_make(mp, dir, "cmfgen_final_publication.json")) return -1;
    if (access(pp, F_OK) == 0 || access(ipath, F_OK) == 0 ||
        access(lp, F_OK) == 0 || access(mp, F_OK) == 0) return -1;

    if (!(fp = exclusive_file(pp))) return -1;
    fprintf(fp, "shell_id,T_e,n_e,te_generation,te_manifest_sha256\n");
    for (int s = 0; s < ns; ++s, ++pr)
        fprintf(fp, "%d,%.17g,%.6e,%llu,%s\n", s, p->T_e[s], p->n_electron[s],
                (unsigned long long)p->T_e_generation, p->te_publication.te_manifest_sha256);
    { int bad = ferror(fp); if (finish(fp)) bad = 1; if (bad) return -1; }

    if (!(fp = exclusive_file(ipath))) return -1;
    fprintf(fp, "shell_id,Z,stage,n_ion,population_generation,partition_te_generation,te_manifest_sha256\n");
    for (int s = 0; s < ns; ++s) for (int i = 0; i < a->n_ion_pops; ++i, ++ir)
        fprintf(fp, "%d,%d,%d,%.6e,%llu,%llu,%s\n", s, a->ion_pop_Z[i],
                a->ion_pop_stage[i], a->ion_number_density[(size_t)i * ns + s],
                (unsigned long long)a->population_committed_generation,
                (unsigned long long)a->partition_stamp.te_generation,
                a->partition_stamp.te_manifest_sha256);
    { int bad = ferror(fp); if (finish(fp)) bad = 1; if (bad) return -1; }

    if (!(fp = exclusive_file(lp))) return -1;
    fprintf(fp, "Z,ion,shell,level_idx,global_idx,E_eV,g,n_pop,T_e,n_ion_total,population_generation,required_population_generation,partition_te_generation,te_manifest_sha256\n");
    for (int ii = 0; ii < n->n_nlte_ions; ++ii) {
        int ip = -1, b = n->nlte_ion_level_offset[ii], e = n->nlte_ion_level_offset[ii + 1];
        if (b == e) continue;
        for (int i = 0; i < a->n_ion_pops; ++i)
            if (a->ion_pop_Z[i] == n->nlte_Z[ii] && a->ion_pop_stage[i] == n->nlte_ion[ii]) { ip = i; break; }
        if (ip < 0) { fclose(fp); return -1; }
        for (int l = b; l < e; ++l) {
            int gi = n->nlte_to_global_level[l];
            if (gi < 0 || gi >= a->n_levels) { fclose(fp); return -1; }
            for (int s = 0; s < ns; ++s, ++lr)
                fprintf(fp, "%d,%d,%d,%d,%d,%.17g,%d,%.6e,%.17g,%.6e,%llu,%llu,%llu,%s\n",
                    n->nlte_Z[ii], n->nlte_ion[ii], s, l - b, gi,
                    a->level_energy_eV[gi], a->level_g[gi],
                    n->nlte_level_populations[(size_t)l * ns + s], p->T_e[s],
                    a->ion_number_density[(size_t)ip * ns + s],
                    (unsigned long long)n->population_committed_generation,
                    (unsigned long long)n->population_required_generation,
                    (unsigned long long)n->within_sl_stamp.te_generation,
                    n->within_sl_stamp.te_manifest_sha256);
        }
    }
    { int bad = ferror(fp); if (finish(fp)) bad = 1; if (bad) return -1; }

    if (!(fp = exclusive_file(mp))) return -1;
    fprintf(fp,
        "{\n  \"schema\": \"CMFGEN_FINAL_PUBLICATION_V1\",\n  \"complete\": true,\n"
        "  \"source_callsite\": \"lumina_cuda.cu:DET_TRANSACTIONAL_POST_CMFGEN_RUN\",\n"
        "  \"population_generation\": %llu,\n  \"required_population_generation\": %llu,\n"
        "  \"te_generation\": %llu,\n  \"partition_te_generation\": %llu,\n"
        "  \"te_manifest_sha256\": \"%s\",\n  \"files\": {\n"
        "    \"plasma\": {\"path\": \"lumina_plasma_state.csv\", \"authority\": \"PUBLIC_NE_TE\", \"rows\": %zu},\n"
        "    \"saha\": {\"path\": \"lumina_ion_pops.csv\", \"authority\": \"SAHA_STAGE\", \"rows\": %zu},\n"
        "    \"nlte\": {\"path\": \"nlte_levels_final.csv\", \"authority\": \"NLTE_LEVEL\", \"rows\": %zu}\n  }\n}\n",
        (unsigned long long)n->population_committed_generation,
        (unsigned long long)n->population_required_generation,
        (unsigned long long)p->T_e_generation,
        (unsigned long long)a->partition_stamp.te_generation,
        p->te_publication.te_manifest_sha256, pr, ir, lr);
    { int bad = ferror(fp); if (finish(fp)) bad = 1; return bad ? -1 : 0; }
}
