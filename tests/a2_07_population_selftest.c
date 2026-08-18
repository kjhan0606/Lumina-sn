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
    PopulationAtomicView a = {
        .n_ions = 2, .n_levels = 5, .level_offset = offset,
        .energy_eV = energy, .g = g, .runtime_membership = NULL,
        .level_Z = z, .level_ion = ion
    };
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

    /* R0 provenance negative control: the thermodynamic-only top-ion catalog
     * changes Z(T_e), so adding or mutating it must stale the atomic identity. */
    int topion_index[] = {1};
    double topion_energy_cm[] = {0.0};
    double topion_g[] = {2.0};
    PopulationAtomicView with_topion = a;
    with_topion.topion_n = 1;
    with_topion.topion_ion_index = topion_index;
    with_topion.topion_E_cm = topion_energy_cm;
    with_topion.topion_g = topion_g;
    double top_pub[6];
    PopulationDerivedStamp top_stamp;
    CHECK(population_partition_build(&with_topion, te, 3, 17, 11,
                                     top_pub, &top_stamp) == POP_OK,
          "top-ion atomic identity build");
    CHECK(population_partition_view_check(&top_stamp, &a, te, 3, 17, 11) ==
          POP_STALE_DERIVED_TEMPERATURE, "top-ion addition changes atomic hash");
    topion_g[0] = 3.0;
    CHECK(population_partition_view_check(&top_stamp, &with_topion, te, 3,
                                          17, 11) ==
          POP_STALE_DERIVED_TEMPERATURE, "top-ion value changes atomic hash");
    topion_g[0] = 2.0;
    with_topion.topion_E_cm = NULL;
    CHECK(population_atomic_model_sha256(&with_topion,
                                         top_stamp.atomic_model_sha256) ==
          POP_ATOMIC_MISSING, "partial top-ion catalog rejected");

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

    /* Detailed-balance SE known answer with a 14-decade population span.
     * The production solver must recover the raw positive solution through
     * algebraic equilibration/refinement; no negative tolerance or clamp is
     * present in this test or in the solver contract. */
    enum { SE_N = 10 };
    double se_matrix[SE_N * SE_N] = {0.0};
    double se_rhs[SE_N] = {0.0};
    double se_true[SE_N], se_solution[SE_N];
    const double per_level = 14.0 * log(10.0) / (SE_N - 1);
    double se_total = 0.0;
    for (int i = 0; i < SE_N; ++i) {
        se_true[i] = exp(-per_level * i);
        se_total += se_true[i];
    }
    for (int row = 0; row < SE_N; ++row) {
        double outflow = 0.0;
        for (int col = 0; col < SE_N; ++col) {
            if (row == col) continue;
            double rate_col_to_row = col < row
                ? 1.0 : se_true[row] / se_true[col];
            se_matrix[col * SE_N + row] = rate_col_to_row;
            double rate_row_to_col = row < col
                ? 1.0 : se_true[col] / se_true[row];
            outflow += rate_row_to_col;
        }
        se_matrix[row * SE_N + row] = -outflow;
    }
    for (int col = 0; col < SE_N; ++col)
        se_matrix[col * SE_N] = 1.0;
    se_rhs[0] = se_total;
    double se_matrix_before[SE_N * SE_N], se_rhs_before[SE_N];
    memcpy(se_matrix_before, se_matrix, sizeof(se_matrix));
    memcpy(se_rhs_before, se_rhs, sizeof(se_rhs));
    PopulationLinearSolveDiagnostic se_diagnostic;
    CHECK(population_dense_solve_equilibrated(
              se_matrix, se_rhs, SE_N, se_solution, &se_diagnostic) == POP_OK,
          "equilibrated detailed-balance solve accepted");
    double se_max_relative = 0.0;
    for (int i = 0; i < SE_N; ++i) {
        double relative = fabs(se_solution[i] - se_true[i]) / se_true[i];
        if (relative > se_max_relative) se_max_relative = relative;
        CHECK(isfinite(se_solution[i]) && se_solution[i] > 0.0,
              "equilibrated raw solution is strictly positive");
    }
    CHECK(se_max_relative <= 1.0e-7,
          "equilibrated detailed-balance known answer recovered");
    CHECK(se_diagnostic.rank == SE_N &&
          se_diagnostic.equilibration_iterations > 0 &&
          se_diagnostic.refinement_iterations >= 2 &&
          isfinite(se_diagnostic.pivot_growth) &&
          se_diagnostic.final_backward_error <=
              POP_DENSE_BACKWARD_ERROR_LIMIT,
          "equilibrated solve diagnostics satisfy contract");
    CHECK(memcmp(se_matrix, se_matrix_before, sizeof(se_matrix)) == 0 &&
          memcmp(se_rhs, se_rhs_before, sizeof(se_rhs)) == 0,
          "equilibrated solve preserves matrix and RHS bytes");
    double se_bad[SE_N * SE_N];
    memcpy(se_bad, se_matrix, sizeof(se_bad));
    for (int row = 0; row < SE_N; ++row)
        se_bad[1 * SE_N + row] = se_bad[0 * SE_N + row];
    CHECK(population_dense_solve_equilibrated(
              se_bad, se_rhs, SE_N, se_solution, &se_diagnostic) ==
              POP_RANK_INCOMPLETE,
          "equilibrated singular system fails closed");

    /* Irreducible generator with a 40-decade stationary span.  GTH consumes
     * only positive transition rates, so rounded large outflow diagonals
     * cannot create a forward negative population. */
    enum { GTH_N = 9 };
    double gth_generator[GTH_N * GTH_N] = {0.0};
    double gth_true[GTH_N], gth_solution[GTH_N];
    long double gth_weight = 1.0L, gth_weight_sum = 0.0L;
    for (int i = 0; i < GTH_N; ++i) {
        gth_true[i] = (double)gth_weight;
        gth_weight_sum += gth_weight;
        gth_weight *= 1.0e-5L;
        if (i + 1 < GTH_N) {
            gth_generator[i * GTH_N + (i + 1)] = 1.0;
            gth_generator[i * GTH_N + i] -= 1.0;
            gth_generator[(i + 1) * GTH_N + i] = 1.0e5;
            gth_generator[(i + 1) * GTH_N + (i + 1)] -= 1.0e5;
        }
    }
    const double gth_total = 7.25e7;
    for (int i = 0; i < GTH_N; ++i)
        gth_true[i] = (double)((long double)gth_true[i] *
                              (long double)gth_total / gth_weight_sum);
    double gth_before[GTH_N * GTH_N];
    memcpy(gth_before, gth_generator, sizeof(gth_before));
    PopulationGeneratorSolveDiagnostic gth_diagnostic;
    CHECK(population_generator_stationary_gth(
              gth_generator, GTH_N, gth_total, gth_solution,
              &gth_diagnostic) == POP_OK,
          "GTH irreducible generator solve accepted");
    for (int i = 0; i < GTH_N; ++i) {
        double relative = fabs(gth_solution[i] - gth_true[i]) / gth_true[i];
        CHECK(isfinite(gth_solution[i]) && gth_solution[i] > 0.0 &&
              relative <= 1.0e-12,
              "GTH known stationary distribution recovered positive");
    }
    CHECK(gth_diagnostic.generator_recognized == 1 &&
          gth_diagnostic.input_column_relative_error <=
              POP_GENERATOR_COLUMN_ERROR_LIMIT &&
          gth_diagnostic.exact_generator_componentwise_residual <=
              POP_GENERATOR_RESIDUAL_LIMIT &&
          gth_diagnostic.minimum_population > 0.0 &&
          gth_diagnostic.maximum_population <= gth_total,
          "GTH diagnostics satisfy generator contract");
    CHECK(memcmp(gth_before, gth_generator, sizeof(gth_before)) == 0,
          "GTH preserves generator bytes");

    double non_generator[GTH_N * GTH_N];
    memcpy(non_generator, gth_generator, sizeof(non_generator));
    non_generator[0 * GTH_N + 1] = -1.0;
    CHECK(population_generator_stationary_gth(
              non_generator, GTH_N, gth_total, gth_solution,
              &gth_diagnostic) == POP_SOLVE_FAILED &&
          gth_diagnostic.generator_recognized == 0,
          "negative off-diagonal is ineligible, not clamped");

    /* Two closed communicating classes are a valid reducible generator but
     * have no unique stationary vector from one total; fail closed. */
    double reducible[] = {
        -1.0, 1.0, 0.0, 0.0,
         1.0,-1.0, 0.0, 0.0,
         0.0, 0.0,-2.0, 2.0,
         0.0, 0.0, 2.0,-2.0
    };
    CHECK(population_generator_stationary_gth(
              reducible, 4, 1.0, gth_solution, &gth_diagnostic) ==
              POP_RANK_INCOMPLETE &&
          gth_diagnostic.generator_recognized == 1,
          "reducible generator fails closed");

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

    /* The bulk tau writer and A2-09 use this same routine.  Exercise both
     * source branches so a future reader-only reimplementation cannot hide
     * behind formula tests. */
    double line_lte = -1.0, line_nlte = -1.0;
    PopulationStatus line_lte_status = population_line_level_number_density(
        POP_LINE_VIEW_LTE_TE, &a, 1, 3, te[0], pub[3],
        2.5e8, NAN, &line_lte);
    CHECK((line_lte_status == POP_OK || line_lte_status == POP_EXACT_ZERO) &&
          isfinite(line_lte) && line_lte >= 0.0,
          "shared LTE line population branch");
    CHECK(population_line_level_number_density(
              POP_LINE_VIEW_NLTE_COMMITTED, &a, 1, 3, NAN, NAN,
              NAN, 7.25e6, &line_nlte) == POP_OK && line_nlte == 7.25e6,
          "committed NLTE line population branch");
    CHECK(population_line_level_number_density(
              POP_LINE_VIEW_NLTE_COMMITTED, &a, 1, 3, NAN, NAN,
              NAN, -1.0, &line_nlte) == POP_NONFINITE,
          "negative NLTE line population rejected");

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
