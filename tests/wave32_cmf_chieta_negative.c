#include "lumina_cmfgen.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void init_fixture(CMFGENState *cs, Geometry *geo,
                         double r_inner[2], double r_outer[2],
                         double nu[3], double dnu[3], double chi[6],
                         double chic[6], double source[6], double J[6],
                         double eta_total[6]) {
    memset(cs, 0, sizeof(*cs));
    memset(geo, 0, sizeof(*geo));
    geo->n_shells = 2;
    geo->time_explosion = 1683072.0;
    geo->r_inner = r_inner;
    geo->r_outer = r_outer;
    cs->n_shells = 2;
    cs->n_bins = 3;
    cs->nu = nu;
    cs->dnu = dnu;
    cs->chi_tot = chi;
    cs->chi_es = chic;
    cs->S_fixed = source;
    cs->J = J;
    cs->eta_total_audit = eta_total;
    for (int q = 0; q < 6; q++)
        eta_total[q] = chi[q] * source[q] + chic[q] * J[q];
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    double ri[2] = {1e14, 2e14}, ro[2] = {2e14, 4e14};
    double nu[3] = {1e14, 2e14, 4e14}, dnu[3] = {5e13, 1e14, 2e14};
    double chi[6] = {1,2,3,4,5,6}, chic[6] = {.1,.2,.3,.4,.5,.6};
    double source[6] = {7,8,9,10,11,12}, J[6] = {13,14,15,16,17,18};
    double eta_total[6];
    CMFGENState cs;
    Geometry geo;
    int failures = 0;

#define EXPECT_REJECT(tag, mutation, iter, generation, post_damp) do {       \
    init_fixture(&cs, &geo, ri, ro, nu, dnu, chi, chic, source, J,           \
                 eta_total);                                                 \
    mutation;                                                                 \
    unlink(argv[1]);                                                          \
    int rc = cmfgen_dump_frozen_chieta(                                      \
        &cs, &geo, iter, generation, post_damp, argv[1]);                    \
    int exists = access(argv[1], F_OK) == 0;                                 \
    printf("%s rc=%d output_exists=%d\n", tag, rc, exists);                 \
    if (rc == 0 || exists) failures++;                                        \
} while (0)

    EXPECT_REJECT("negative_chi", chi[2] = -1.0, 10, 10, 1);
    chi[2] = 3.0;
    EXPECT_REJECT("noncontiguous_radius", ri[1] += 1e9, 10, 10, 1);
    ri[1] = 2e14;
    EXPECT_REJECT("nonascending_frequency", nu[1] = nu[0], 10, 10, 1);
    nu[1] = 2e14;
    EXPECT_REJECT("nonfinite_J", J[4] = NAN, 10, 10, 1);
    J[4] = 17.0;
    EXPECT_REJECT("negative_iteration", (void)0, -1, 10, 1);
    EXPECT_REJECT("negative_generation", (void)0, 10, -1, 1);
    EXPECT_REJECT("invalid_post_damp", (void)0, 10, 10, 2);
#undef EXPECT_REJECT
    return failures ? 1 : 0;
}
