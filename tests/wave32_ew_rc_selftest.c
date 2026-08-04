#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>

int wave32_rc_wrapped_nlte_element_wide_run_labeled(
    NLTEConfig *, AtomicData *, PlasmaState *, OpacityState *, int, int, int,
    double, GammaDeposition *);

int main(void) {
    setenv("LUMINA_NLTE_ELEMENT_WIDE", "2", 1);
    int bad_env = nlte_element_wide_run_labeled(
        NULL, NULL, NULL, NULL, 26, 0, 0, 1.0, NULL);

    setenv("W32_FORCE_EW_OOM", "1", 1);
    int forced_oom = wave32_rc_wrapped_nlte_element_wide_run_labeled(
        NULL, NULL, NULL, NULL, 26, 0, 0, 1.0, NULL);
    unsetenv("W32_FORCE_EW_OOM");

    setenv("W32_FORCE_EW_IO", "1", 1);
    int forced_io = wave32_rc_wrapped_nlte_element_wide_run_labeled(
        NULL, NULL, NULL, NULL, 26, 0, 0, 1.0, NULL);

    printf("bad_env_rc=%d forced_oom_rc=%d forced_io_rc=%d\n",
           bad_env, forced_oom, forced_io);
    return bad_env == -1 && forced_oom == -1 && forced_io == -1 ? 0 : 1;
}
