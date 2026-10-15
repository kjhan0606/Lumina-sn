/* 실제 fine 조립의 signed 선 소광 대조. 합성 한 선이며 SN 분포 시험은 아니다. */
#include "lumina_cmfgen.h"
#include "opacity_publication.h"
#include "line_net_rate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void w5_cont_selftest_arm_capture(double *,double *,double *,double *,double *,double *,size_t);
extern int w5_cont_selftest_capture_shape(int *,int *);
#define CHECK(x) do{if(!(x)){fprintf(stderr,"COMMON_SIGNED_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)

static int fill_upper(const AtomicData *a,const PlasmaState *p,const NLTEConfig *n,
                      size_t lines,size_t shells,double *out){
    (void)a;(void)p;
    for(size_t i=0;i<lines*shells;i++)out[i]=n->nlte_level_populations[shells+i%shells];
    return 0;
}

int main(void){
    enum {NS=2,CAP=65536};
    const double c=2.99792458e10,h=6.62607015e-27,pi=3.14159265358979323846;
    const double time=1e6,vdop=1e8;
    double ri[NS]={3e14,4e14},ro[NS]={4e14,5e14},T[NS]={1e4,1e4},ne[NS]={0,0};
    double nu=5e14,f=.5,lambda=c/nu;
    double A=8*pi*SOBOLEV_COEFF*pow(nu/c,2)*f*(2./4.);
    double tau[NS],jbar[NS],population[2*NS];
    A208Validity valid[NS]={A208_VALID,A208_VALID};
    int Z=14,stage=1,mapped=0,dummy_eset=1;
    int lz[2]={14,14},lq[2]={1,1},ln[2]={0,1},lg[2]={2,4},lower=0,upper=1;
    Geometry g={0};g.n_shells=NS;g.r_inner=ri;g.r_outer=ro;g.time_explosion=time;
    PlasmaState p={0};p.n_shells=NS;p.T_e=T;p.n_electron=ne;
    OpacityState o={0};o.n_shells=NS;o.n_lines=1;o.line_list_nu=&nu;
    o.tau_sobolev=tau;o.tau_validity=valid;o.jbar_line_det=jbar;
    AtomicData a={0};a.n_lines=1;a.line_A_ul=&A;a.line_atomic_number=&Z;a.line_ion_number=&stage;
    a.n_levels=2;a.level_Z=lz;a.level_ion=lq;a.level_num=ln;a.level_g=lg;
    a.line_level_lower=&lower;a.line_level_upper=&upper;a.line_f_lu=&f;a.line_wavelength_cm=&lambda;
    NLTEConfig n={0};n.line_eset=&dummy_eset;n.nlte_line_map=&mapped;n.nlte_level_populations=population;
    CMFGENState cs={0};CHECK(!cmfgen_init(&cs,&g));
    for(int s=0;s<NS;s++)for(int b=0;b<cs.n_bins;b++){
        size_t i=(size_t)s*cs.n_bins+b;
        cs.chi_abs[i]=cs.chi_tot[i]=2e-14;cs.chi_es[i]=0;
        cs.S_fixed[i]=2*h*pow(cs.nu[b],3)/(c*c)/expm1(h*cs.nu[b]/(1.380649e-16*T[s]));
    }
    double *capture=calloc(6*CAP,sizeof(double));CHECK(capture);
    w5_cont_selftest_arm_capture(capture,capture+CAP,capture+2*CAP,
                                capture+3*CAP,capture+4*CAP,capture+5*CAP,CAP);
    cmfgen_fine_set_bf_atom(NULL,&a,&n,fill_upper);
    setenv("LUMINA_CMF_FINE_VDOP","1e8",1);
    setenv("LUMINA_CMF_FINE_PPD","1.3",1);
    setenv("LUMINA_CMF_FINE_ALI","16",1);
    setenv("LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS","1",1);
    setenv("LUMINA_CMF_FINE_TOL","1e-10",1);
    setenv("LUMINA_CMF_FINE_PHOTOION","1",1);
    const double cases[]={2,0,-.25,-.5,-.500001,-2};
    for(int legacy=0;legacy<2;legacy++)for(size_t k=0;k<sizeof(cases)/sizeof(cases[0]);k++){
        for(int s=0;s<NS;s++){
            tau[s]=cases[k];population[NS+s]=200;
            population[s]=100+tau[s]/(SOBOLEV_COEFF*f*lambda*time);
            CHECK(population[s]>0);
        }
        CMFFineLineOperator mode=legacy?CMF_FINE_LINE_OPERATOR_INIT_SHARED_GAUSSIAN:
                                        CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN;
        CHECK(!cmfgen_fine_jbar(&cs,&g,&o,12000,&p,mode));
        int ns=0,nf=0;CHECK(w5_cont_selftest_capture_shape(&ns,&nf));
        CHECK(ns==NS&&nf>0&&(size_t)ns*nf<=CAP);
        double dnu=nu*vdop/c,integrated=cases[k]*nu/(c*time);
        if(legacy){
            LineNetSobolevMaterial old;
            CHECK(!line_net_sobolev_material(200,A,nu,cases[k],time,
                      LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK,1,&old));
            integrated=old.effective_integrated_opacity;
        }
        double chi0=integrated/(sqrt(pi)*dnu);
        double eta0=200*A*h*nu/(4*pi*sqrt(pi)*dnu),max_chi=0,max_eta=0;
        int support=0;
        for(int s=0;s<ns;s++)for(int b=0;b<nf;b++){
            size_t i=(size_t)s*nf+b;double x=(capture[5*CAP+b]-nu)/dnu;
            double profile=fabs(x)<=4?exp(-x*x):0;
            double want=chi0*profile,got=capture[4*CAP+i];
            double want_eta=eta0*profile,got_eta=capture[2*CAP+i];
            if(profile>0)support++;
            if(want==0)CHECK(got==0);else max_chi=fmax(max_chi,fabs(got/want-1));
            if(want_eta==0)CHECK(got_eta==0);else max_eta=fmax(max_eta,fabs(got_eta/want_eta-1));
        }
        printf("SIGNED_ACTUAL_ASSEMBLY mode=%s tau=%.17g n_lower=%.17g n_upper=200 support=%d chi_relative=%.17g eta_relative=%.17g\n",
               legacy?"INIT_SHARED":"COMMON",cases[k],population[0],support,max_chi,max_eta);
        fflush(stdout);
        CHECK(support>0&&max_chi<2e-13&&max_eta<2e-13);
    }
    free(o.jnu_fine);free(o.nu_fine);free(o.jbar_line_det_error_upper);
    free(capture);cmfgen_free(&cs);
    puts("CMF_COMMON_SIGNED_MATERIAL_SELFTEST_PASS");return 0;
}
