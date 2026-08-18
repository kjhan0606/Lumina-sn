#include "lumina.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void nlte_ew_note_save_restore_call(void);
void nlte_ew_note_per_ion_pin_call(void);
void nlte_ew_note_topstage_IV_call(void);
void nlte_ew_runtime_counts_snapshot(unsigned long out[3]);
void nlte_ew_note_save_restore_call_for(const NLTEConfig *nlte);
void nlte_ew_note_per_ion_pin_call_for(const NLTEConfig *nlte);
void nlte_ew_note_topstage_IV_call_for(const NLTEConfig *nlte);
void nlte_ew_runtime_counts_snapshot_for(
    const NLTEConfig *nlte, unsigned long out[3]);
void nlte_ew_test_capture_counters_reset(void);
void nlte_ew_test_capture_counters_bump(void);
void nlte_ew_test_capture_counters_snapshot(int out[19]);
void nlte_ew_test_expected_outflow_reset(void);
void nlte_ew_test_expected_outflow_bump(void);
void nlte_ew_test_expected_outflow_seed_invalid(void);
void nlte_ew_test_expected_outflow_snapshot(double out[3], int *bad_rate);

int main(void) {
    enum { N = 200000 };
    unsigned long before[3], after[3];
    int capture[19];
    /* Production initializes the gate before entering the shell-parallel
     * region; reproduce that ownership ordering rather than racing env parse. */
    (void)nlte_element_wide_enabled();
    nlte_ew_runtime_counts_snapshot(before);
    nlte_ew_test_capture_counters_reset();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        nlte_ew_note_save_restore_call();
        nlte_ew_note_per_ion_pin_call();
        nlte_ew_note_topstage_IV_call();
        nlte_ew_test_capture_counters_bump();
    }
    nlte_ew_runtime_counts_snapshot(after);
    nlte_ew_test_capture_counters_snapshot(capture);
    unsigned long expected = getenv("W32_EXPECT_COUNTER_DISABLED") ? 0UL : N;
    printf("expected=%lu save_restore=%lu per_ion_pin=%lu topstage_IV=%lu\n",
           expected, after[0] - before[0], after[1] - before[1],
           after[2] - before[2]);
    int capture_ok = 1;
    for (int k = 0; k < 19; k++)
        if (capture[k] != N) capture_ok = 0;
    printf("capture_counters=19 target_fail=%d all_exact=%d\n",
           capture[2], capture_ok);

    nlte_ew_test_expected_outflow_reset();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        nlte_ew_test_expected_outflow_bump();
    double ledger_matrix[3], before_invalid[3];
    int bad_rate = 0;
    nlte_ew_test_expected_outflow_snapshot(ledger_matrix, &bad_rate);
    for (int k = 0; k < 3; k++) before_invalid[k] = ledger_matrix[k];
    nlte_ew_test_expected_outflow_seed_invalid();
    nlte_ew_test_expected_outflow_snapshot(ledger_matrix, &bad_rate);
    int ledger_exact = ledger_matrix[0] == (double)N &&
                       ledger_matrix[1] == (double)N &&
                       ledger_matrix[2] == -(double)N;
    int invalid_unchanged = bad_rate == 1;
    for (int k = 0; k < 3; k++)
        if (ledger_matrix[k] != before_invalid[k]) invalid_unchanged = 0;
    printf("expected_outflow=%.17g matrix_inflow=%.17g matrix_debit=%.17g "
           "all_exact=%d\n", ledger_matrix[0], ledger_matrix[1],
           ledger_matrix[2], ledger_exact);
    printf("invalid_rate_bad_rate=%d arrays_unchanged=%d\n",
           bad_rate, invalid_unchanged);

    NLTEConfig private_nlte = {0};
    NLTEEWRuntimeCounts private_counts = {0};
    unsigned long global_before_private[3], global_after_private[3];
    unsigned long private_after[3];
    private_nlte.ew_runtime_counts_sink = &private_counts;
    nlte_ew_runtime_counts_snapshot(global_before_private);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        nlte_ew_note_save_restore_call_for(&private_nlte);
        nlte_ew_note_per_ion_pin_call_for(&private_nlte);
        nlte_ew_note_topstage_IV_call_for(&private_nlte);
    }
    nlte_ew_runtime_counts_snapshot_for(&private_nlte, private_after);
    nlte_ew_runtime_counts_snapshot(global_after_private);
    int private_exact = private_after[0] == N && private_after[1] == N &&
                        private_after[2] == N;
    int global_unchanged =
        global_after_private[0] == global_before_private[0] &&
        global_after_private[1] == global_before_private[1] &&
        global_after_private[2] == global_before_private[2];
    printf("private_sink_exact=%d process_global_unchanged=%d\n",
           private_exact, global_unchanged);
    return capture_ok && ledger_exact && invalid_unchanged &&
           private_exact && global_unchanged &&
           after[0] - before[0] == expected &&
           after[1] - before[1] == expected &&
           after[2] - before[2] == expected ? 0 : 1;
}
