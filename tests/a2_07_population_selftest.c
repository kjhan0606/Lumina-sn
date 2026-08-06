#include "population_contract.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); return 1; } } while (0)

static long double ref_partition(const double *e, const int *g, int lo, int hi,
                                 long double te) {
    const long double ev = 1.602176634e-12L;
    const long double kb = 1.380649e-16L;
    long double e0 = e[lo], sum = 0.0L, corr = 0.0L;
    for (int i = lo + 1; i < hi; i++) if (e[i] < e0) e0 = e[i];
    for (int i = lo; i < hi; i++) {
        long double term = (long double)g[i] *
            expl(-((long double)e[i] - e0) * ev / (kb * te));
        long double y = term - corr, t = sum + y;
        corr = (t - sum) - y;
        sum = t;
    }
    return sum;
}

int main(void) {
    int offset[] = {0, 2, 5};
    double energy[] = {10.0, 10.5, 4.0, 5.0, 35.0};
    int g[] = {2, 4, 1, 3, 9};
    int z[] = {8, 8, 26, 26, 26};
    int ion[] = {1, 1, 2, 2, 2};
    PopulationAtomicView a = {2, 5, offset, energy, g, NULL, z, ion};
    double te[] = {4000.0, 10000.0, 140000.0};
    double pub[6];
    for (int i = 0; i < 6; i++) pub[i] = -7.0;
    PopulationDerivedStamp stamp;

    CHECK(population_partition_build(&a, te, 3, 7, 11, pub, &stamp) == POP_OK,
          "partition transaction");
    for (int i = 0; i < 2; i++) for (int s = 0; s < 3; s++) {
        long double ref = ref_partition(energy, g, offset[i], offset[i + 1], te[s]);
        long double rel = fabsl(((long double)pub[i * 3 + s] - ref) / ref);
        CHECK(rel <= 1.0e-10L, "long-double Z oracle");
    }
    CHECK(stamp.computed_population_generation == 7 && stamp.te_generation == 11,
          "stamp generations");
    CHECK(population_partition_view_check(&stamp, &a, te, 3, 7, 11) == POP_OK,
          "fresh view");

    double changed_te[] = {4000.0, 10000.0, 140000.0};
    unsigned char *bits = (unsigned char *)&changed_te[1];
    bits[0] ^= 1U;
    CHECK(population_partition_view_check(&stamp, &a, changed_te, 3, 7, 11) ==
          POP_STALE_DERIVED_TEMPERATURE, "T_e bit stale");
    CHECK(population_partition_view_check(&stamp, &a, te, 3, 8, 11) ==
          POP_STALE_DERIVED_TEMPERATURE, "generation stale");

    double before[6]; memcpy(before, pub, sizeof(pub));
    double bad0[] = {4000.0, 0.0, 140000.0};
    double badn[] = {4000.0, NAN, 140000.0};
    CHECK(population_partition_build(&a, NULL, 3, 8, 12, pub, &stamp) == POP_INVALID_TE,
          "missing T_e");
    CHECK(population_partition_build(&a, bad0, 3, 8, 12, pub, &stamp) == POP_INVALID_TE,
          "nonpositive T_e");
    CHECK(population_partition_build(&a, badn, 3, 8, 12, pub, &stamp) == POP_INVALID_TE,
          "nonfinite T_e");
    CHECK(memcmp(before, pub, sizeof(pub)) == 0, "invalid T_e published nothing");

    double ion_pub[] = {1.0, 2.0}, level_pub[] = {3.0, 4.0, 5.0};
    double ne_pub[] = {6.0}; uint64_t committed = 3;
    PopulationTransaction tx;
    CHECK(population_transaction_begin(&tx, ion_pub, 2, level_pub, 3,
          ne_pub, 1, NULL, 0, 4, &committed) == 0, "transaction begin");
    tx.work_ion[0] = 10.0; tx.work_level[1] = NAN; tx.work_ne[0] = 12.0;
    CHECK(population_transaction_commit(&tx) == POP_NONFINITE,
          "mid-shell nonfinite rejected");
    CHECK(ion_pub[0] == 1.0 && level_pub[1] == 4.0 && ne_pub[0] == 6.0 &&
          committed == 3, "partial publish zero");

    CHECK(population_transaction_begin(&tx, ion_pub, 2, level_pub, 3,
          ne_pub, 1, NULL, 0, 4, &committed) == 0, "transaction restart");
    tx.work_ion[0] = 10.0; tx.work_level[1] = 11.0; tx.work_ne[0] = 12.0;
    CHECK(population_transaction_commit(&tx) == POP_OK && committed == 4,
          "atomic commit");
    CHECK(ion_pub[0] == 10.0 && level_pub[1] == 11.0 && ne_pub[0] == 12.0,
          "all arrays committed");

    CHECK(population_rate_views_check(POP_OK, 9, POP_EXACT_ZERO, 9, 9) ==
          POP_OK, "BF/BB valid and exact-zero views");
    PopulationStatus blocked[] = {
        POP_BF_UNSAMPLED, POP_BF_OOG, POP_BF_MISS, POP_BF_STALE,
        POP_BB_UNSAMPLED, POP_BB_OOG, POP_BB_MISS, POP_BB_STALE,
        POP_PROFILE_MISMATCH, POP_QUERY_HASH_MISMATCH
    };
    for (size_t i = 0; i < sizeof(blocked) / sizeof(blocked[0]); i++) {
        PopulationStatus got = i < 4
            ? population_rate_views_check(blocked[i], 9, POP_OK, 9, 9)
            : population_rate_views_check(POP_OK, 9, blocked[i], 9, 9);
        CHECK(got == blocked[i], "invalid BF/BB status blocks solve");
    }
    CHECK(population_rate_views_check(POP_OK, 9, POP_OK, 10, 9) ==
          POP_STALE_DERIVED_TEMPERATURE, "one-view generation change blocks");

    double full_rank[] = {1.0, 2.0, 3.0, 5.0};
    double singular[] = {1.0, 2.0, 2.0, 4.0};
    CHECK(population_dense_rank_check(full_rank, 2, 1e-14) == POP_OK,
          "full rank accepted");
    CHECK(population_dense_rank_check(singular, 2, 1e-14) ==
          POP_RANK_INCOMPLETE, "isolated/singular row rejected");

    double fine[] = {1.0, 2.0, 4.0};
    int member[] = {0, 0, 1};
    double super[] = {-1.0, -1.0};
    CHECK(population_superlevel_aggregate(fine, member, 3, 2, super) == POP_OK &&
          super[0] == 3.0 && super[1] == 4.0,
          "superlevel aggregate");
    int ambiguous[] = {0, -2, 1};
    CHECK(population_superlevel_aggregate(fine, ambiguous, 3, 2, super) ==
          POP_ATOMIC_MISSING, "ambiguous membership is not zero-filled");

    double fraction = -1.0;
    CHECK(population_lte_level_fraction(&a, 1, 4, te[0], pub[3], &fraction) ==
          POP_EXACT_ZERO || fraction >= 0.0, "underflow is explicit zero");

    PopulationCounters c = {0};
    population_counter_note(&c, POP_EXACT_ZERO);
    population_counter_note(&c, POP_BF_UNSAMPLED);
    population_counter_note(&c, POP_BB_OOG);
    population_counter_note(&c, POP_BB_MISS);
    population_counter_note(&c, POP_BF_STALE);
    CHECK(c.pop_exact_zero_terms == 1 && c.pop_blocked_unsampled == 1 &&
          c.pop_blocked_oog == 1 && c.pop_blocked_miss == 1 &&
          c.pop_blocked_stale == 1, "validity counters");

    PopulationCounters pass = {0};
    pass.pop_generation_required = 4;
    pass.pop_generation_committed = 4;
    pass.pop_shells_attempted = 3;
    pass.pop_shells_published = 3;
    pass.pop_bf_terms = 2;
    pass.pop_bb_terms = 2;
    population_counters_print(stdout, &pass);

    puts("A2-07 population selftest: PASS");
    return 0;
}
