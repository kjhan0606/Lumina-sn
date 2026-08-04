#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>

int nlte_element_wide_run_labeled(NLTEConfig *, AtomicData *, PlasmaState *,
                                  OpacityState *, int, int, int, double,
                                  GammaDeposition *);

int wave32_rc_wrapped_nlte_element_wide_run_labeled(
    NLTEConfig *nlte, AtomicData *atom, PlasmaState *plasma,
    OpacityState *opacity, int Z, int shell_index, int shell_label,
    double time_explosion, GammaDeposition *gamma_dep) {
    if (getenv("W32_FORCE_EW_OOM")) {
        fprintf(stderr,
                "[EW][OOM] forced fixture allocation failure Z=%d s=%d\n",
                Z, shell_label);
        return -1;
    }
    if (getenv("W32_FORCE_EW_IO")) {
        fprintf(stderr,
                "[EW][I/O-FAIL] forced fixture artifact failure Z=%d s=%d\n",
                Z, shell_label);
        return -1;
    }
    return nlte_element_wide_run_labeled(
        nlte, atom, plasma, opacity, Z, shell_index, shell_label,
        time_explosion, gamma_dep);
}
