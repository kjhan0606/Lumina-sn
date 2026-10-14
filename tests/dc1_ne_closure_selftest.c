/* D-C-1 electron-density closure selftest (candidate API).
 * Build (verified 2026-09-06, --gc-sections drops the unreferenced CUDA-host
 * callers so lumina_plasma.c links with population_contract.c alone):
 *   gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections \
 *       -fdata-sections -Isrc -Wl,--gc-sections -o /tmp/dc1 \
 *       tests/dc1_ne_closure_selftest.c src/lumina_plasma.c \
 *       src/population_contract.c -lm -fopenmp && /tmp/dc1 */
#include "lumina.h"
#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "DC1_NE_CLOSURE_SELFTEST_FAIL reason=%s\n",       \
                (message));                                                 \
        return 1;                                                           \
    }                                                                       \
} while (0)

enum { TEST_SHELLS = 2, TEST_IONS = 6, TEST_LEVELS = 8, TEST_NLTE_LEVELS = 4 };

/* Synthetic deck: Si (target Z=14) stages 0,1,2 with levels {0},{1,2},{3,4};
 * C (non-target Z=6) stages 0,1,2 with one level each.  NLTE set = Si II, III
 * (nlte levels 0..3 -> global 1..4). */
typedef struct {
    NLTEPopulationCandidate c;
    int ion_Z[TEST_IONS], ion_stage[TEST_IONS], level_offset[TEST_IONS + 1];
    int level_Z[TEST_LEVELS], level_ion[TEST_LEVELS];
    int nlte_Z[2], nlte_ion[2], nlte_ion_level_offset[3], nlte_to_global[TEST_NLTE_LEVELS];
    double ion_population[TEST_IONS * TEST_SHELLS];
    double level_population[TEST_NLTE_LEVELS * TEST_SHELLS];
    double electron_density[TEST_SHELLS];
} Fixture;

static void fixture_init(Fixture *f) {
    memset(f, 0, sizeof(*f));
    const int ion_Z[] = {14, 14, 14, 6, 6, 6};
    const int ion_stage[] = {0, 1, 2, 0, 1, 2};
    const int level_offset[] = {0, 1, 3, 5, 6, 7, 8};
    const int level_Z[] = {14, 14, 14, 14, 14, 6, 6, 6};
    const int level_ion[] = {0, 1, 1, 2, 2, 0, 1, 2};
    const int nlte_Z[] = {14, 14};
    const int nlte_ion[] = {1, 2};
    const int nlte_off[] = {0, 2, 4};
    const int nlte_to_global[] = {1, 2, 3, 4};
    /* ion-major: ion0 (5,6) ion1 (20,40) ion2 (30,10) ion3 (7,8) ion4 (50,30) ion5 (60,20) */
    const double ion_population[] = {5, 6, 20, 40, 30, 10, 7, 8, 50, 30, 60, 20};
    /* nlte-level-major: Si II = nl0+nl1 = (30,30); Si III = nl2+nl3 = (20,15) */
    const double level_population[] = {18, 20, 12, 10, 12, 12, 8, 3};
    memcpy(f->ion_Z, ion_Z, sizeof ion_Z); memcpy(f->ion_stage, ion_stage, sizeof ion_stage);
    memcpy(f->level_offset, level_offset, sizeof level_offset);
    memcpy(f->level_Z, level_Z, sizeof level_Z); memcpy(f->level_ion, level_ion, sizeof level_ion);
    memcpy(f->nlte_Z, nlte_Z, sizeof nlte_Z); memcpy(f->nlte_ion, nlte_ion, sizeof nlte_ion);
    memcpy(f->nlte_ion_level_offset, nlte_off, sizeof nlte_off);
    memcpy(f->nlte_to_global, nlte_to_global, sizeof nlte_to_global);
    memcpy(f->ion_population, ion_population, sizeof ion_population);
    memcpy(f->level_population, level_population, sizeof level_population);
    /* q_saha: shell0 = 1*20+2*30 + 1*50+2*60 = 250 ; shell1 = 40+20+30+40 = 130 */
    f->electron_density[0] = 250.0; f->electron_density[1] = 130.0;
    f->c.atom.n_ion_pops = TEST_IONS; f->c.atom.n_levels = TEST_LEVELS;
    f->c.atom.ion_pop_Z = f->ion_Z; f->c.atom.ion_pop_stage = f->ion_stage;
    f->c.atom.level_offset = f->level_offset; f->c.atom.level_Z = f->level_Z;
    f->c.atom.level_ion = f->level_ion;
    f->c.nlte.n_nlte_ions = 2; f->c.nlte.n_nlte_levels_total = TEST_NLTE_LEVELS;
    memcpy(f->c.nlte.nlte_Z, f->nlte_Z, sizeof f->nlte_Z);
    memcpy(f->c.nlte.nlte_ion, f->nlte_ion, sizeof f->nlte_ion);
    memcpy(f->c.nlte.nlte_ion_level_offset, f->nlte_ion_level_offset, sizeof f->nlte_ion_level_offset);
    f->c.nlte.nlte_to_global_level = f->nlte_to_global;
    f->c.ion_population = f->ion_population; f->c.level_population = f->level_population;
    f->c.electron_density = f->electron_density; f->c.n_shells = TEST_SHELLS;
    f->c.required_population_generation = 7;
}

static int count_lines(FILE *s, const char *needle) {
    char line[1024]; int n = 0; rewind(s);
    while (fgets(line, sizeof line, s)) if (strstr(line, needle)) n++;
    return n;
}
static int grep_field(FILE *s, const char *key, double *out) {
    char line[1024]; rewind(s);
    while (fgets(line, sizeof line, s)) {
        if (!strstr(line, "[A2-07][NE-CLOSURE] ")) continue;
        char spaced[64]; snprintf(spaced, sizeof spaced, " %s", key);
        char *p = strstr(line, spaced); if (!p) return 0;
        *out = strtod(p + strlen(spaced), NULL); return 1;
    }
    return 0;
}

int main(void) {
    Fixture f; FILE *rec; double v, next[TEST_SHELLS];
    int worst = -1;
    int positive = 0, negative = 0;

    /* OFF: byte-identical, zero records */
    fixture_init(&f); unsetenv("LUMINA_NLTE_NE_CLOSURE");
    rec = tmpfile(); CHECK(rec, "STREAM");
    double before[TEST_SHELLS]; memcpy(before, f.electron_density, sizeof before);
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) == 0, "OFF_RETURN");
    CHECK(memcmp(before, f.electron_density, sizeof before) == 0, "OFF_BYTE_IDENTICAL");
    CHECK(count_lines(rec, "NE-CLOSURE") == 0, "OFF_NO_RECORDS"); fclose(rec); positive++;
    /* env=0 must also be silent */
    fixture_init(&f); setenv("LUMINA_NLTE_NE_CLOSURE", "0", 1); rec = tmpfile();
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) == 0 && count_lines(rec, "NE-CLOSURE") == 0 &&
          f.electron_density[0] == 250.0, "ENV_ZERO_EMISSION_REJECTED"); fclose(rec); negative++;

    /* The production controller's actual read-only step: shell0 q_closed is
     * 240, so 0.5 damping maps trial 250 -> 245.  This synthetic population
     * fixture exercises the real controller helper, not a real pair solve. */
    fixture_init(&f); setenv("LUMINA_NLTE_NE_CLOSURE", "1", 1); rec = tmpfile();
    CHECK(nlte_ne_closure_trial_step(&f.c, next, &v, &worst) == POP_OK,
          "STEP_RETURN");
    CHECK(next[0] == 245.0 && next[1] == 130.0, "STEP_DAMPING");
    CHECK(fabs(v - 0.04) < 1e-15 && worst == 0, "STEP_RESIDUAL");
    CHECK(f.electron_density[0] == 250.0 && f.electron_density[1] == 130.0,
          "STEP_READ_ONLY");
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0,
          "MISMATCH_REFUSED");
    CHECK(f.electron_density[0] == 250.0,
          "MISMATCH_NOT_POSTSOLVE_REPLACED");
    CHECK(count_lines(rec, "reason=NOT_CONVERGED") == 1,
          "MISMATCH_REASON");
    fclose(rec); positive++;

    /* Repeat the actual 0.5 step to the production tolerance while holding
     * this toy population response fixed.  Acceptance retains the n_e used by
     * the last synthetic state; it does not replace it with q_closed. */
    fixture_init(&f); rec = tmpfile(); int iterations = 0;
    for (iterations = 1; iterations <= 32; ++iterations) {
        CHECK(nlte_ne_closure_trial_step(&f.c, next, &v, &worst) == POP_OK,
              "ITERATED_STEP_RETURN");
        if (v <= 1.0e-8) break;
        memcpy(f.electron_density, next, sizeof(next));
    }
    CHECK(iterations == 23, "ITERATED_STEP_COUNT");
    double consumed_ne = f.electron_density[0];
    f.c.ne_closure_active = 1; f.c.ne_closure_iterations = iterations;
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) == 0,
          "ITERATED_ACCEPTED");
    CHECK(f.electron_density[0] == consumed_ne && consumed_ne != 240.0,
          "ITERATED_CONSUMED_NE_RETAINED");
    fclose(rec); positive++;

    /* Accepted state retains the exact consumed trial n_e and emits one small
     * summary.  The old 1e-12 duplicate-charge identity is distinct from this
     * n_e-to-hybrid-charge 1e-8 fixed-point residual. */
    fixture_init(&f); f.electron_density[0] = 240.0;
    f.c.ne_closure_active = 1; f.c.ne_closure_iterations = 19;
    rec = tmpfile();
    CHECK(nlte_ne_closure_candidate(&f.c, "SEED", rec) == 0,
          "ACCEPTED_RETURN");
    CHECK(f.electron_density[0] == 240.0 && f.electron_density[1] == 130.0,
          "ACCEPTED_NE_PRESERVED");
    CHECK(count_lines(rec, "[A2-07][NE-CLOSURE] commit=SEED") == 1,
          "ACCEPTED_ONE_SUMMARY");
    CHECK(grep_field(rec, "max_relative_residual=", &v) && v == 0.0,
          "ACCEPTED_FIXEDPOINT_RESIDUAL");
    CHECK(grep_field(rec, "iterations=", &v) && v == 19.0,
          "ACCEPTED_ITERATIONS");
    fclose(rec); positive++;

    /* Negatives: refusal cases (return != 0, FATAL, arrays untouched) and one sum-sensitivity case */
    fixture_init(&f); f.nlte_to_global[3] = 2; rec = tmpfile();   /* NLTE level map broken */
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0 && count_lines(rec, "[FATAL]") == 1 &&
          f.electron_density[0] == 250.0, "MISSING_NLTE_LEVEL_REJECTED"); fclose(rec); negative++;
    fixture_init(&f); f.ion_stage[1] = 2; f.ion_stage[2] = 1; rec = tmpfile(); /* charge convention reversed */
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0 && count_lines(rec, "[FATAL]") == 1,
          "REVERSED_CHARGE_REJECTED"); fclose(rec); negative++;
    fixture_init(&f); f.c.atom.n_ion_pops = 2; rec = tmpfile(); /* target exists, q2 stage absent */
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0 && count_lines(rec, "[FATAL]") == 1,
          "TARGET_Q2_STAGE_MISSING_REJECTED"); fclose(rec); negative++;
    fixture_init(&f); f.c.atom.n_ion_pops = 5; f.level_offset[5] = TEST_LEVELS; rec = tmpfile(); /* drop C III */
    CHECK(nlte_ne_closure_trial_step(&f.c, next, &v, &worst) == POP_OK &&
          next[0] == 185.0 && f.electron_density[0] == 250.0,
          "NON_NLTE_STAGE_DROP_CHANGES_SUM"); fclose(rec); negative++;
    fixture_init(&f); for (int i = 0; i < TEST_IONS * TEST_SHELLS; i++) f.ion_population[i] = 0.0;
    for (int i = 0; i < TEST_NLTE_LEVELS * TEST_SHELLS; i++) f.level_population[i] = 0.0;
    rec = tmpfile();
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0 && count_lines(rec, "POP_NONFINITE") == 1 &&
          f.electron_density[0] == 250.0, "NONPOSITIVE_REFUSED_NO_FLOOR"); fclose(rec); negative++;
    fixture_init(&f); f.level_population[0] = NAN; rec = tmpfile();
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0 && count_lines(rec, "POP_NONFINITE") == 1,
          "NONFINITE_REFUSED"); fclose(rec); negative++;
    fixture_init(&f); f.electron_density[1] = 0.0; rec = tmpfile();
    CHECK(nlte_ne_closure_candidate(&f.c, "R7", rec) != 0 && count_lines(rec, "POP_NONFINITE") == 1,
          "SAHA_NE_NONPOSITIVE_REFUSED"); fclose(rec); negative++;

    printf("DC1_NE_CLOSURE_SELFTEST_PASS positive=%d negative=%d\n", positive, negative);
    return 0;
}
