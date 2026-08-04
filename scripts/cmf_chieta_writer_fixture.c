#include "lumina_cmfgen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 2;
    }
    double r_inner[2] = {1.0e14, 2.0e14};
    double r_outer[2] = {2.0e14, 4.0e14};
    double nu[3] = {1.0e14, 2.0e14, 4.0e14};
    double dnu[3] = {0.5e14, 1.0e14, 2.0e14};
    double chi_tot[6] = {1,2,3,4,5,6};
    double chi_es[6] = {0.1,0.2,0.3,0.4,0.5,0.6};
    double S_fixed[6] = {7,8,9,10,11,12};
    double J[6] = {13,14,15,16,17,18};
    double eta_total[6];
    Geometry geo;
    CMFGENState cs;
    memset(&geo,0,sizeof(geo));
    memset(&cs,0,sizeof(cs));
    geo.n_shells=2; geo.r_inner=r_inner; geo.r_outer=r_outer;
    geo.time_explosion=1683072.0;
    cs.n_shells=2; cs.n_bins=3; cs.nu=nu; cs.dnu=dnu;
    cs.chi_tot=chi_tot; cs.chi_es=chi_es;
    cs.S_fixed=S_fixed; cs.J=J; cs.eta_total_audit=eta_total;
    for (int q=0;q<6;q++)
        eta_total[q]=chi_tot[q]*S_fixed[q]+chi_es[q]*J[q];
    if (getenv("W32_SEED_BAD_ETA")) eta_total[0] += 1.0;
    const char *iter_env = getenv("W32_FIXTURE_ITER");
    const char *generation_env = getenv("W32_FIXTURE_GENERATION");
    const char *post_env = getenv("W32_FIXTURE_POST_DAMP");
    int iter = iter_env ? atoi(iter_env) : 10;
    int generation = generation_env ? atoi(generation_env) : iter;
    int post_damp = post_env ? atoi(post_env) : 1;
    return cmfgen_dump_frozen_chieta(
        &cs,&geo,iter,generation,post_damp,argv[1]) == 0 ? 0 : 1;
}
