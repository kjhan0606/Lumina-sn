#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__real_malloc(size_t);
int nlte_precompute_within_sl_frac_checked(
    NLTEConfig *, AtomicData *, PlasmaState *, int);
static int force_oom;
void *__wrap_malloc(size_t n) {
    if (force_oom && n == sizeof(double)) return NULL;
    return __real_malloc(n);
}

int main(void) {
    NLTEConfig nlte;
    AtomicData atom;
    PlasmaState plasma;
    memset(&nlte, 0, sizeof(nlte));
    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));
    nlte.super_mode = 1;
    nlte.n_super_total = 1;
    force_oom = 1;
    int checked_rc = nlte_precompute_within_sl_frac_checked(
        &nlte, &atom, &plasma, 1);
    int legacy_rc = nlte_precompute_within_sl_frac(
        &nlte, &atom, &plasma, 1);
    int solve_rc = nlte_solve_all(
        &nlte, &atom, &plasma, NULL, 1.0, 1, NULL);
    force_oom = 0;
    nlte.super_mode = 0;
    int normal_rc = nlte_precompute_within_sl_frac(
        &nlte, &atom, &plasma, 1);
    printf("checked_oom_rc=%d legacy_oom_rc=%d solve_oom_rc=%d "
           "normal_legacy_rc=%d\n",
           checked_rc, legacy_rc, solve_rc, normal_rc);
    return checked_rc == -1 && legacy_rc == -1 && solve_rc == -1 &&
           normal_rc == 0 ? 0 : 1;
}
