#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>

void nlte_ew_runtime_counts_snapshot(unsigned long out[3]);

int nlte_element_wide_run_labeled(NLTEConfig *, AtomicData *, PlasmaState *,
                                  OpacityState *, int, int, int, double,
                                  GammaDeposition *);

/* Exercise the real CPU pair-owner paths before the frozen observer.  The
 * frozen state has one local shell (labelled s8 by the observer), so the
 * production EW selector does not recursively match shell index zero. */
int wave32_counter_wrapped_nlte_element_wide_run_labeled(
    NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
    OpacityState *opacity, int Z, int shell_index, int shell_label,
    double time_explosion, GammaDeposition *gamma_dep) {
    static int exercised = 0;
    if (!exercised) {
        unsigned long before[3], after[3];
        double *owned_line_source = NULL;
        exercised = 1;
        /* The frozen harness does not allocate the production line-source
         * writeback buffer because it normally assembles matrices only.  The
         * real nlte_solve_all owner path does, so provide that exact state. */
        if (!opacity->line_source_S) {
            owned_line_source = (double *)calloc(
                (size_t)opacity->n_lines * plasma->n_shells, sizeof(double));
            if (!owned_line_source) return -1;
            opacity->line_source_S = owned_line_source;
        }
        if (getenv("W32_RUNTIME_DISABLE_PIN_TOPSTAGE")) {
            setenv("LUMINA_NLTE_ION_LOCK", "0", 1);
            setenv("LUMINA_TOPSTAGE_IV", "0", 1);
        }
        nlte_ew_runtime_counts_snapshot(before);
        if (nlte_solve_all(nlte, atom, plasma, opacity, time_explosion,
                           plasma->n_shells, gamma_dep) != 0) {
            free(owned_line_source);
            if (owned_line_source) opacity->line_source_S = NULL;
            return -1;
        }
        nlte_ew_runtime_counts_snapshot(after);
        fprintf(stderr,
                "[W32-RUNTIME-COUNTERS] save_restore=%lu per_ion_pin=%lu "
                "topstage_IV=%lu\n",
                after[0] - before[0], after[1] - before[1],
                after[2] - before[2]);
        if (owned_line_source) {
            free(owned_line_source);
            opacity->line_source_S = NULL;
        }
    }
    if (getenv("W32_RUNTIME_COUNTER_ONLY")) return 0;
    return nlte_element_wide_run_labeled(
        nlte, atom, plasma, opacity, Z, shell_index, shell_label,
        time_explosion, gamma_dep);
}
