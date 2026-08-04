#include "lumina.h"

#include <stdio.h>
#include <string.h>

int nlte_bf_field_source(const NLTEConfig *, double, double, double, int,
                         int *, int *, double *);
int nlte_bf_collisional_enabled(void);
unsigned long nlte_bf_gpu_field_bypass_count(void);

int main(void) {
    NLTEConfig n;
    double estimator = 1.0;
    int use = -1, bypass = -1;
    double selected = -1.0;
    memset(&n, 0, sizeof(n));
    n.bf_rate_estimator = &estimator;
    int source = nlte_bf_field_source(&n, 10000.0, 2.0e15, 3.0,
                                      1, &use, &bypass, &selected);
    printf("source=%d use_gpu=%d gpu_bypass=%d production_bypass=%lu "
           "selected=%.17g collisional=%d\n",
           source, use, bypass, nlte_bf_gpu_field_bypass_count(), selected,
           nlte_bf_collisional_enabled());
    return 0;
}
