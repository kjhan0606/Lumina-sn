#include "lumina_cmfgen.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition, message) do { \
    if (!(condition)) { fprintf(stderr,"FAIL: %s\n",message); return 1; } \
} while (0)

/* The fixture links only the CMF assembler.  These production providers are
 * irrelevant with BF and eps-phys disabled, but remain link-time references. */
double radeq_line_eps_phys(int line_id, double ne, double Te, double tau) {
    (void)line_id; (void)ne; (void)Te; (void)tau;
    return 1.0;
}
double bf_get_chi(BFOpacity *bf, int shell, double nu) {
    (void)bf; (void)shell; (void)nu;
    return 0.0;
}
double bf_get_eta(BFOpacity *bf, int shell, double nu) {
    (void)bf; (void)shell; (void)nu;
    return 0.0;
}

static int init_state(CMFGENState *cs, int ns, int nb) {
    memset(cs,0,sizeof(*cs));
    cs->n_shells=ns; cs->n_bins=nb;
    cs->nu_min=1.0e14; cs->nu_max=8.0e14;
    cs->d_log_nu=log(cs->nu_max/cs->nu_min)/(double)nb;
#define ALLOC(member,count) do { \
    cs->member=(double *)calloc((count),sizeof(double)); \
    if (!cs->member) return -1; \
} while (0)
    ALLOC(nu,nb); ALLOC(dnu,nb);
    size_t cells=(size_t)ns*nb;
    ALLOC(chi_es,cells); ALLOC(chi_abs,cells); ALLOC(chi_line,cells);
    ALLOC(chi_line_th,cells); ALLOC(chi_line_cls,cells);
    ALLOC(chi_tot,cells); ALLOC(S_fixed,cells); ALLOC(J,cells);
    ALLOC(eta_total_audit,cells);
#undef ALLOC
    for (int b=0;b<nb;b++) {
        cs->nu[b]=cs->nu_min*exp((b+0.5)*cs->d_log_nu);
        cs->dnu[b]=cs->nu[b]*cs->d_log_nu;
    }
    for (size_t q=0;q<cells;q++) cs->J[q]=1.0e-8*(double)(q+1);
    cs->frozen_morph_eps=-1.0;
    return 0;
}

static void audit(CMFGENState *cs) {
    size_t cells=(size_t)cs->n_shells*cs->n_bins;
    for (size_t q=0;q<cells;q++)
        cs->eta_total_audit[q]=cs->chi_tot[q]*cs->S_fixed[q]+
                               cs->chi_es[q]*cs->J[q];
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr,"usage: %s OUTPUT_BASE\n",argv[0]);
        return 2;
    }
    enum { NS=2, NB=3, NL=3 };
    double ri[NS]={1.0e14,2.0e14}, ro[NS]={2.0e14,4.0e14};
    Geometry geo={0}; geo.n_shells=NS; geo.r_inner=ri; geo.r_outer=ro;
    geo.time_explosion=1.0e6;

    double line_nu[NL]={1.5e14,3.0e14,6.0e14};
    double tau[NL*NS]={0.2,0.3, 0.4,0.5, 0.6,0.7};
    double line_source[NL*NS]={0};
    /* Keep continuum emissivity exactly zero so the two undefined-line bins
     * are an unambiguous no-fallback oracle. */
    double ne[NS]={0.0,0.0}, Te[NS]={8000.0,9000.0};
    OpacityState op={0}; op.n_lines=NL; op.n_shells=NS;
    op.line_list_nu=line_nu; op.tau_sobolev=tau;
    op.line_source_S=line_source; op.electron_density=ne;

    int line_Z[NL]={26,14,26}, line_ion[NL]={2,1,2};
    int lower[NL]={0,0,0}, upper_num[NL]={1,1,1};
    double Aul[NL]={1.0e8,2.0e8,0.0};
    int level_num[2]={0,1}, level_Z[2]={26,26}, level_ion[2]={2,2};
    AtomicData atom={0}; atom.n_lines=NL; atom.n_levels=2;
    atom.line_atomic_number=line_Z; atom.line_ion_number=line_ion;
    atom.line_level_lower=lower; atom.line_level_upper=upper_num;
    atom.line_A_ul=Aul; atom.level_num=level_num;
    atom.level_Z=level_Z; atom.level_ion=level_ion;

    int to_global[2]={0,1}, global_to[2]={0,1}, line_map[NL]={0,-1,0};
    double pop[2*NS]={10.0,20.0, 3.0,4.0};
    NLTEConfig nlte={0}; nlte.n_nlte_ions=1; nlte.n_nlte_levels_total=2;
    nlte.nlte_ion_level_offset[0]=0; nlte.nlte_ion_level_offset[1]=2;
    nlte.nlte_to_global_level=to_global; nlte.global_to_nlte_level=global_to;
    nlte.nlte_line_map=line_map; nlte.nlte_level_populations=pop;

    PlasmaState plasma={0}; plasma.n_shells=NS; plasma.T_e=Te;
    plasma.n_electron=ne;
    CMFGENState a,b,b2,seeded;
    if (init_state(&a,NS,NB) || init_state(&b,NS,NB) ||
        init_state(&b2,NS,NB) ||
        init_state(&seeded,NS,NB)) return 1;
    memcpy(b.J,a.J,sizeof(double)*NS*NB);
    memcpy(b2.J,a.J,sizeof(double)*NS*NB);
    memcpy(seeded.J,a.J,sizeof(double)*NS*NB);
    cmfgen_set_deposition(NULL,NS);
    char state_hash[65];
    if (cmfgen_emiss_ab_state_sha256(&a,&geo,&op,NULL,&plasma,&nlte,
                                      &atom,state_hash)) return 1;
    cmfgen_assemble(&a,&geo,&op,NULL,&plasma);
    CMFGENEmissABStats clean={0}, controlled={0}, bad={0};
    if (cmfgen_assemble_aulnu(&b,&geo,&op,NULL,&plasma,&nlte,&atom,
                              -1,-1,1.0,0,&clean)) return 1;
    if (cmfgen_assemble_aulnu(&b2,&geo,&op,NULL,&plasma,&nlte,&atom,
                              -1,-1,1.0,1,&controlled)) return 1;
    if (cmfgen_assemble_aulnu(&seeded,&geo,&op,NULL,&plasma,&nlte,&atom,
                              0,0,2.0,0,&bad)) return 1;
    REQUIRE(!(clean.active_transition_count != 3 ||
        clean.defined_transition_count != 1 ||
        clean.undefined_transition_count != 2 ||
        clean.active_line_shell_count != 6 ||
        clean.defined_line_shell_count != 2 ||
        clean.undefined_line_shell_count != 4 || bad.seed_hits != 1),
        "coverage/seed census");
    REQUIRE(controlled.retained_transition_count == 2 &&
            controlled.retained_line_shell_count == 4 &&
            controlled.a_reference_retained_contribution_fraction > 0.0,
            "B2 controlled-retention census");
    REQUIRE(memcmp(b.S_fixed,seeded.S_fixed,sizeof(double)*NS*NB) != 0,
            "seeded n_u corruption was not detected");
    for (int s=0;s<NS;s++)
        REQUIRE(b.S_fixed[(size_t)s*NB+1] == 0.0 &&
                b.S_fixed[(size_t)s*NB+2] == 0.0,
                "undefined line acquired an emissivity fallback");
    for (int s=0;s<NS;s++) {
        REQUIRE(b2.S_fixed[(size_t)s*NB+1] == a.S_fixed[(size_t)s*NB+1] &&
                b2.S_fixed[(size_t)s*NB+2] == a.S_fixed[(size_t)s*NB+2],
                "B2 did not retain undefined A emissivity");
        REQUIRE(b2.S_fixed[(size_t)s*NB] == b.S_fixed[(size_t)s*NB],
                "B2 changed the covered direct-Aul contribution");
    }
    double diag_band=0.0, diag_shell=0.0;
    for (int k=0;k<NB;k++) diag_band+=controlled.undefined_a_emissivity_by_band[k];
    for (int k=0;k<NS;k++) diag_shell+=controlled.undefined_a_emissivity_by_shell[k];
    REQUIRE(fabs(diag_band-controlled.a_reference_undefined_line_power) <=
            1e-14*fabs(diag_band) &&
            fabs(diag_shell-controlled.a_reference_undefined_line_power) <=
            1e-14*fabs(diag_shell), "undefined diagnostic arrays do not close");
    REQUIRE(!(memcmp(a.chi_tot,b.chi_tot,sizeof(double)*NS*NB) ||
        memcmp(a.chi_es,b.chi_es,sizeof(double)*NS*NB) ||
        memcmp(a.chi_line,b.chi_line,sizeof(double)*NS*NB)),
        "non-emissivity coordinate changed");
    audit(&a); audit(&b); audit(&b2); audit(&seeded);

    char ap[4096],bp[4096],b2p[4096],sp[4096],up[4096],b2up[4096];
    if (snprintf(ap,sizeof(ap),"%s.A",argv[1]) >= (int)sizeof(ap) ||
        snprintf(bp,sizeof(bp),"%s.B",argv[1]) >= (int)sizeof(bp) ||
        snprintf(b2p,sizeof(b2p),"%s.B2",argv[1]) >= (int)sizeof(b2p) ||
        snprintf(sp,sizeof(sp),"%s.B.seeded",argv[1]) >= (int)sizeof(sp) ||
        snprintf(up,sizeof(up),"%s.undefined.csv",bp) >= (int)sizeof(up) ||
        snprintf(b2up,sizeof(b2up),"%s.undefined.csv",b2p) >= (int)sizeof(b2up)) return 1;
    CMFGENChietaLaneMeta am={"A-production",state_hash,&clean};
    CMFGENChietaLaneMeta bm={"B-Aul-nu",state_hash,&clean};
    CMFGENChietaLaneMeta b2m={"B2-Aul-nu-retain-A-undefined",state_hash,
                              &controlled};
    CMFGENChietaLaneMeta sm={"B-Aul-nu",state_hash,&bad};
    if (cmfgen_write_emiss_ab_undefined(&clean,&atom,up) ||
        cmfgen_write_emiss_ab_undefined(&controlled,&atom,b2up) ||
        cmfgen_dump_frozen_chieta_lane(&a,&geo,10,10,1,ap,&am) ||
        cmfgen_dump_frozen_chieta_lane(&b,&geo,10,10,1,bp,&bm) ||
        cmfgen_dump_frozen_chieta_lane(&b2,&geo,10,10,1,b2p,&b2m) ||
        cmfgen_dump_frozen_chieta_lane(&seeded,&geo,10,10,1,sp,&sm)) return 1;
    printf("PASS state=%s clean_defined=%llu/%llu contribution=%.17g "
           "seed_line=0 seed_shell=0 factor=2 seed_hits=%llu\n",state_hash,
           (unsigned long long)clean.defined_transition_count,
           (unsigned long long)clean.active_transition_count,
           clean.a_reference_contribution_fraction,
           (unsigned long long)bad.seed_hits);
    cmfgen_emiss_ab_stats_free(&clean);
    cmfgen_emiss_ab_stats_free(&controlled);
    cmfgen_emiss_ab_stats_free(&bad);
    cmfgen_free(&a); cmfgen_free(&b); cmfgen_free(&b2); cmfgen_free(&seeded);
    return 0;
}
