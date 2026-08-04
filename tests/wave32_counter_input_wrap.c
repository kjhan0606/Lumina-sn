#include "lumina.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nlte_element_wide_run_labeled(NLTEConfig *, AtomicData *, PlasmaState *,
                                  OpacityState *, int, int, int, double,
                                  GammaDeposition *);

/* Counter-specific negative inputs.  Exactly one mode is selected per process:
 *   no_kramers: present every sigma row (target-map failures remain fail-closed)
 *   no_pref: make every estimator bin positive without changing its scale
 *   grid_mismatch: make the CMFGEN sigma grid length differ by one
 *   nstar_overflow: force an inverse-Saha value outside double range.
 * These are test inputs, not production repairs. */
int wave32_input_wrapped_nlte_element_wide_run_labeled(
    NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
    OpacityState *opacity, int Z, int shell_index, int shell_label,
    double time_explosion, GammaDeposition *gamma_dep) {
    static int prepared = 0;
    if (!prepared) {
        const char *mode = getenv("W32_COUNTER_INPUT_MODE");
        prepared = 1;
        if (mode && strcmp(mode, "no_kramers") == 0) {
            for (int gl = 0; gl < atom->n_levels; gl++)
                atom->cmfgen_has_sigma[gl] = 1;
            fprintf(stderr, "[W32-COUNTER-INPUT] mode=no_kramers\n");
        } else if (mode && strcmp(mode, "no_pref") == 0) {
            size_t n = (size_t)plasma->n_shells * nlte->n_freq_bins;
            for (size_t q = 0; q < n; q++)
                if (!(nlte->bf_rate_estimator[q] > 0.0))
                    nlte->bf_rate_estimator[q] = DBL_MIN;
            fprintf(stderr, "[W32-COUNTER-INPUT] mode=no_pref\n");
        } else if (mode && strcmp(mode, "grid_mismatch") == 0) {
            atom->cmfgen_n_freq_bins = nlte->n_freq_bins - 1;
            fprintf(stderr, "[W32-COUNTER-INPUT] mode=grid_mismatch\n");
        } else if (mode && strcmp(mode, "nstar_overflow") == 0) {
            for (int s = 0; s < plasma->n_shells; s++) plasma->T_e[s] = 1.0;
            fprintf(stderr, "[W32-COUNTER-INPUT] mode=nstar_overflow\n");
        }
    }
    return nlte_element_wide_run_labeled(
        nlte, atom, plasma, opacity, Z, shell_index, shell_label,
        time_explosion, gamma_dep);
}
