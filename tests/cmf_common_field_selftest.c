/* 공통장 구현의 직접시험. 실제 별 모형 또는 CMFGEN 일치 시험은 아니다. */
#include "lumina_cmfgen.h"
#include "opacity_publication.h"
#include "line_jbar.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "COMMON_FIELD_FAIL line=%d: %s\n", __LINE__, #x); return 1; \
} } while (0)

static int rebin_test(void) {
    double fine[2*7], coarse[2*4], before[2*4];
    CMFGENState a = {0}, b = {0};
    a.n_shells = b.n_shells = 2;
    a.n_bins = 7; b.n_bins = 4;
    a.nu_min = b.nu_min = 1e14; a.nu_max = b.nu_max = 9e14;
    a.d_log_nu = log(9.0)/7; b.d_log_nu = log(9.0)/4;
    a.J = fine; b.J = coarse;
    for (int i=0; i<14; ++i) fine[i] = 1+i%7;
    REQUIRE(cmfgen_rebin_log_field(&a, &b) == 0);
    long double ia=0, ib=0;
    for (int i=0; i<7; ++i)
        ia += fine[i]*(a.nu_min*exp((i+1)*a.d_log_nu)-a.nu_min*exp(i*a.d_log_nu));
    for (int i=0; i<4; ++i)
        ib += coarse[i]*(b.nu_min*exp((i+1)*b.d_log_nu)-b.nu_min*exp(i*b.d_log_nu));
    REQUIRE(fabsl(ia-ib)/ia < 2e-14L);
    REQUIRE(memcmp(coarse, coarse+4, 4*sizeof(double)) == 0);
    memcpy(before, coarse, sizeof before);
    a.nu_min *= 1.01;
    REQUIRE(cmfgen_rebin_log_field(&a, &b) != 0);
    REQUIRE(memcmp(before, coarse, sizeof before) == 0);
    a.nu_min = b.nu_min;
    fine[13] = NAN;
    REQUIRE(cmfgen_rebin_log_field(&a, &b) != 0);
    REQUIRE(memcmp(before, coarse, sizeof before) == 0);
    fine[13] = -1;
    REQUIRE(cmfgen_rebin_log_field(&a, &b) != 0);
    REQUIRE(memcmp(before, coarse, sizeof before) == 0);
    for (int i=0; i<14; ++i) fine[i] = 0;
    REQUIRE(cmfgen_rebin_log_field(&a, &b) == 0);
    for (int i=0; i<8; ++i) REQUIRE(coarse[i] == 0);
    printf("rebin_conservation_relative=%.17Lg invalid_preserves_target=1\n", fabsl(ia-ib)/ia);
    return 0;
}

static int removed_line = -1;
static int fill_upper(const AtomicData *a, const PlasmaState *p,
        const NLTEConfig *n, size_t nl, size_t ns, double *out) {
    (void)a; (void)p; (void)n;
    for (size_t i=0; i<nl*ns; ++i) out[i]=(int)(i/ns)==removed_line?0:0.01;
    return 0;
}

static double planck(double nu, double T) {
    return (2*6.62607015e-27*nu*nu*nu/(2.99792458e10*2.99792458e10)) /
        expm1(6.62607015e-27*nu/(1.380649e-16*T));
}

static int producer_test(void) {
    /* 넓은 인공 선폭은 CPU 비용을 줄이기 위한 fixture 선택이다.
     * 생산 선폭(10km/s)의 수렴 또는 분해능을 인증하지 않는다. */
    enum { NS=3, NL=4 };
    double ri[NS]={3e14,4e14,5e14}, ro[NS]={4e14,5e14,6e14};
    double te[NS]={10000,9000,8000}, ne[NS]={1e8,1e8,1e8};
    Geometry geo={0}; geo.n_shells=NS;geo.r_inner=ri;geo.r_outer=ro;geo.time_explosion=1e6;
    PlasmaState plasma={0}; plasma.n_shells=NS;plasma.T_e=te;plasma.n_electron=ne;
    double lam[NL]={80,1200,1201,25000}, nu[NL], tau[NL*NS], jbar[NL*NS], A[NL];
    A208Validity valid[NL*NS];
    int Z[NL]={14,14,16,16}, q[NL]={1,1,1,1}, mapped[NL]={0,0,0,0}, dummy_eset=1;
    for(int l=0;l<NL;++l) {
        nu[l]=2.99792458e18/lam[l]; A[l]=1e8;
        for(int s=0;s<NS;++s) { tau[l*NS+s]=2;valid[l*NS+s]=A208_VALID;jbar[l*NS+s]=-1; }
    }
    OpacityState op={0}; op.n_shells=NS;op.n_lines=NL;op.line_list_nu=nu;
    op.tau_sobolev=tau;op.tau_validity=valid;op.jbar_line_det=jbar;
    AtomicData atom={0}; atom.n_lines=NL;atom.line_A_ul=A;atom.line_atomic_number=Z;atom.line_ion_number=q;
    NLTEConfig nlte={0};nlte.line_eset=&dummy_eset;nlte.nlte_line_map=mapped;
    CMFGENState cs={0}; REQUIRE(cmfgen_init(&cs,&geo)==0);
    cs.common_obs_requested=1;cs.common_obs_n=9;
    cs.common_obs_nu=malloc(9*sizeof(double));REQUIRE(cs.common_obs_nu);
    for(int f=0;f<9;++f)cs.common_obs_nu[f]=2.99792458e18/(1220.0-4*f);
    for(int s=0;s<NS;++s) for(int b=0;b<cs.n_bins;++b) {
        size_t i=(size_t)s*cs.n_bins+b;
        cs.chi_abs[i]=2e-14;cs.chi_es[i]=1e-15;cs.chi_tot[i]=2.1e-14;
        cs.S_fixed[i]=cs.chi_abs[i]/cs.chi_tot[i]*planck(cs.nu[b],te[s]);
        cs.J[i]=1e-99;
    }
    cmfgen_fine_set_bf_atom(NULL,&atom,&nlte,fill_upper);
    setenv("LUMINA_CMF_FINE_VDOP","1e8",1);
    setenv("LUMINA_CMF_FINE_PPD","1.3",1);
    setenv("LUMINA_CMF_FINE_ALI","64",1);
    setenv("LUMINA_CMF_FINE_TOL","1e-10",1);
    setenv("LUMINA_CMF_FINE_PHOTOION","1",1);
    REQUIRE(cmfgen_fine_jbar(&cs,&geo,&op,12000,&plasma,
                CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN)==0);
    REQUIRE(op.jbar_line_det_operator==CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN);
    REQUIRE(op.n_fine>0 && op.jnu_fine && op.nu_fine);
    REQUIRE(cs.common_obs_ready && cs.common_obs_lnu);
    double observer_before[9];memcpy(observer_before,cs.common_obs_lnu,sizeof observer_before);
    for(int f=0;f<9;++f)REQUIRE(isfinite(observer_before[f]) && observer_before[f]>0);
    REQUIRE(cmfgen_write_common_observer(&cs,"common_observer.csv","common_observer.json",4,3,2)==0);
    REQUIRE(cmfgen_write_common_observer(&cs,"common_observer.csv","common_observer.json",4,3,2)!=0);
    REQUIRE(!op.line_producer_terms_captured);
    double maxdiff=0;
    /* 독립적인 전체 교집합 적분. 생산 two-pointer 구현을 재사용하지 않는다. */
    for(int s=0;s<NS;++s) for(int b=0;b<cs.n_bins;++b) {
        double lo=cs.nu_min*exp(b*cs.d_log_nu);
        double hi=b+1==cs.n_bins?cs.nu_max:cs.nu_min*exp((b+1)*cs.d_log_nu);
        long double integral=0;
        for(int f=0;f<op.n_fine;++f) {
            double fl=op.nu_lo_fine*exp(f*op.dlognu_fine);
            double fh=op.nu_lo_fine*exp((f+1)*op.dlognu_fine);
            double overlap=fmin(hi,fh)-fmax(lo,fl);
            if(overlap>0) integral+=op.jnu_fine[(size_t)s*op.n_fine+f]*overlap;
        }
        double expected=(double)(integral/(hi-lo));
        double error=fabs(cs.J[(size_t)s*cs.n_bins+b]-expected)/fmax(expected,1e-300);
        if(error>maxdiff)maxdiff=error;
    }
    REQUIRE(maxdiff<2e-13);
    double *width=malloc((size_t)op.n_fine*sizeof(double));
    double *zero=calloc((size_t)NS*op.n_fine,sizeof(double));
    REQUIRE(width && zero);
    for(int f=0;f<op.n_fine;++f)width[f]=op.nu_fine[f]*op.dlognu_fine;
    double avg[NS],err[NS]; LineJbarProfileReport report;
    REQUIRE(line_jbar_gaussian_discrete_shells(NS,op.n_fine,op.nu_fine,width,
        op.jnu_fine,zero,nu[1],1e8,4,avg,err,&report)==LINE_JBAR_PROFILE_OK);
    for(int s=0;s<NS;++s) REQUIRE(avg[s]==jbar[NS+s]);
    printf("production_common_field fine_bins=%d BF_reference_max_relative=%.17g BB_same_field_exact=1\n",op.n_fine,maxdiff);
    free(width);free(zero);
    double old_jbar=jbar[NS+1];
    double *with_neighbor=malloc((size_t)NS*cs.n_bins*sizeof(double));
    REQUIRE(with_neighbor);
    memcpy(with_neighbor,cs.J,(size_t)NS*cs.n_bins*sizeof(double));
    removed_line=2;
    for(int s=0;s<NS;++s){tau[2*NS+s]=0;valid[2*NS+s]=A208_EXACT_ZERO;}
    REQUIRE(cmfgen_fine_jbar(&cs,&geo,&op,12000,&plasma,
                CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN)==0);
    double bb_change=fabs(jbar[NS+1]/old_jbar-1), bf_change=0;
    double obs_change=0;
    REQUIRE(cs.common_obs_ready);
    for(int f=0;f<9;++f) {
        double change=fabs(cs.common_obs_lnu[f]/observer_before[f]-1);
        if(change>obs_change)obs_change=change;
    }
    REQUIRE(obs_change>1e-3);
    printf("observer_actual_fine_producer neighbor_removal_max_relative=%.17g writer_no_overwrite=1\n",obs_change);
    for(int b=0;b<cs.n_bins;++b) {
        double lambda=2.99792458e18/cs.nu[b];
        if(lambda<1100 || lambda>1400)continue;
        double r=fabs(cs.J[cs.n_bins+b]/with_neighbor[cs.n_bins+b]-1);
        if(r>bf_change)bf_change=r;
    }
    REQUIRE(bb_change>1e-3 && bf_change>1e-3);
    printf("neighbor_removal BB_relative_change=%.17g BF_band_max_relative_change=%.17g\n",bb_change,bf_change);
    free(with_neighbor);
    removed_line=-1;
    for(int s=0;s<NS;++s){tau[2*NS+s]=2;valid[2*NS+s]=A208_VALID;}
    /* 기준 초기화 경로는 계속 BF 쪽 장을 바꾸지 않아야 한다. */
    for(int i=0;i<NS*cs.n_bins;++i)cs.J[i]=1e-99;
    REQUIRE(cmfgen_fine_jbar(&cs,&geo,&op,12000,&plasma,
                CMF_FINE_LINE_OPERATOR_INIT_SHARED_GAUSSIAN)==0);
    REQUIRE(!cs.common_obs_ready);
    for(int i=0;i<NS*cs.n_bins;++i)REQUIRE(cs.J[i]==1e-99);
    REQUIRE(cmfgen_fine_jbar(&cs,&geo,&op,12000,&plasma,
                CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV)==0);
    for(int i=0;i<NS*cs.n_bins;++i)REQUIRE(cs.J[i]==1e-99);
    free(op.jnu_fine);free(op.nu_fine);free(op.jbar_line_det_error_upper);
    cmfgen_free(&cs);
    return 0;
}

int main(void) {
    if(rebin_test() || producer_test())return 1;
    puts("COMMON_FIELD_SELFTEST_PASS");
    return 0;
}
