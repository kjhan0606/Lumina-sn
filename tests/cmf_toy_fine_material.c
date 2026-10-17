/* 실제 공통장 계수 조립만 실행한다. 형식해 진입 직전 의도적으로 중단하며
 * 새 복사장이나 수렴 결과를 게시하지 않는다. 생산 객체는 변경하지 않는다. */
#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include "cmf_exact_sliding.h"
#include "line_net_rate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static CMFToyModel *model;
static int captured;
#ifndef LUMINA_FINE_EXPECTED_ITERATION
#define LUMINA_FINE_EXPECTED_ITERATION 4
#endif
#ifdef LUMINA_FINE_ORIGIN
#include "cmf_continuum_origin.h"
#endif
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FINE_MATERIAL_FAIL line=%d %s\n",__LINE__,#x);goto done;}}while(0)
CMFExactStatus __wrap_cmf_exact_emissivity_solve(
 int ns,int nb,double dlog,const double *nu,const double *ri,const double *ro,
 double time,double temperature,double boundary_scale,const double *chi,
 const double *scatter,const double *eta,double *J,double *error_upper,
 size_t refinements,int iterations,double tolerance,CMFExactMode mode,CMFExactReport *report){
    (void)time;(void)temperature;(void)boundary_scale;(void)error_upper;
    (void)refinements;(void)iterations;(void)tolerance;(void)mode;
    if(report)report->status=CMF_EXACT_INVALID_INPUT;
    if(!model||captured||ns!=113||
       (model->opacity.n_fine && dlog!=model->opacity.dlognu_fine))
        return CMF_EXACT_INVALID_INPUT;
    const int s=39;size_t off=(size_t)s*nb;
#ifdef LUMINA_FULL_DOMAIN_MATERIAL
    /* 새 격자의 실제 계수 조립 확인. 과거 J는 초기 추정일 뿐이며 이
     * 중단 시험은 새 수송해를 만들지 않는다. 모든 전이의 지지범위 확인. */
    if(nb<=model->opacity.n_fine)return CMF_EXACT_INVALID_INPUT;
    double lo=nu[0]*exp(-.5*dlog),hi=nu[nb-1]*exp(.5*dlog);
    size_t bad;
    if(line_jbar_qset_bind_profile(&model->eset,5e6)||
       line_jbar_qset_profile_support_covered(&model->eset,lo,hi,&bad))
        return CMF_EXACT_INVALID_INPUT;
    for(size_t k=0;k<(size_t)ns*nb;k++)
        if(!isfinite(chi[k])||!isfinite(scatter[k])||scatter[k]<0||
           !isfinite(eta[k])||eta[k]<0||!isfinite(J[k])||J[k]<0)
            return CMF_EXACT_INVALID_INPUT;
    FILE *ext=fopen("material_extrema.csv","wx");
    if(!ext)return CMF_EXACT_INVALID_INPUT;
    int ext_ok=fprintf(ext,"shell,T,ne,negative_total_bins,min_nu_Hz,min_chi_total,min_chi_es,min_eta,chi_times_dr,chi_times_ct_dlog\n")>0;
    size_t global_min=0;
    for(int sh=0;sh<ns;sh++){
        size_t at=(size_t)sh*nb;int negatives=0;
        for(int b=0;b<nb;b++){
            size_t ix=(size_t)sh*nb+b;
            if(chi[ix]<0)negatives++;
            if(chi[ix]<chi[at])at=ix;
        }
        if(chi[at]<chi[global_min])global_min=at;
        if(fprintf(ext,"%d,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",sh,
            model->plasma.T_e[sh],model->plasma.n_electron[sh],negatives,nu[at%nb],chi[at],scatter[at],eta[at],
            chi[at]*(ro[sh]-ri[sh]),chi[at]*C_SPEED_OF_LIGHT*time*dlog)<0)ext_ok=0;
    }
    if(fclose(ext))ext_ok=0;
    if(!ext_ok)return CMF_EXACT_INVALID_INPUT;
    int sh=(int)(global_min/nb),worst=-1;double sum_line=0,most_negative=0;
    double freq=nu[global_min%nb];
    for(int l=0;l<model->atom.n_lines;l++){
        double f=model->atom.line_nu[l],x=(freq-f)/(f*5e6/C_SPEED_OF_LIGHT);
        if(fabs(x)>4)continue;
        double part=model->opacity.tau_sobolev[(size_t)l*ns+sh]/(time*5e6*sqrt(M_PI_VAL))*exp(-x*x);
        sum_line+=part;if(part<most_negative){most_negative=part;worst=l;}
    }
    printf("FULL_DOMAIN_MINIMUM shell=%d nu=%.17g total_chi=%.17g line_sum=%.17g continuum_plus_es=%.17g strongest_negative_line=%d contribution=%.17g\n",
        sh,freq,chi[global_min],sum_line,chi[global_min]-sum_line,worst,most_negative);
    printf("FULL_DOMAIN_MATERIAL old_bins=%d new_bins=%d all_profiles_covered=1 finite_coefficients=1 formal_solve=0\n",model->opacity.n_fine,nb);
#else
    if(nb!=model->opacity.n_fine)return CMF_EXACT_INVALID_INPUT;
    if(memcmp(J+off,model->opacity.jnu_fine+off,(size_t)nb*sizeof(double)))return CMF_EXACT_INVALID_INPUT;
#endif
#ifdef LUMINA_FINE_ORIGIN
    if(origin_capture(model,nb,nu,chi+off,eta+off))return CMF_EXACT_INVALID_INPUT;
#endif
    FILE *f=fopen("fine_shell39.bin","wx");if(!f)return CMF_EXACT_INVALID_INPUT;
    uint64_t n=(uint64_t)nb;
    int ok=fwrite("FNMAT01",1,8,f)==8&&fwrite(&n,8,1,f)==1;
    const double *arrays[]={nu,chi+off,scatter+off,eta+off,J+off};
    for(int k=0;k<5;k++)if(fwrite(arrays[k],8,n,f)!=n)ok=0;
    if(fclose(f))ok=0;
    if(ok){captured=1;printf("FINE_MATERIAL_CAPTURE shell=%d bins=%d T=%.17g dr=%.17g dlog=%.17g formal_solve=0\n",s,nb,model->plasma.T_e[s],ro[s]-ri[s],dlog);}
    return CMF_EXACT_INVALID_INPUT;
}
int main(int argc,char **argv){
    if(argc!=5)return 2;
    setvbuf(stdout,NULL,_IOLBF,0);
    CMFToyModel m={0};CMFGENState cs={0};FILE *f=NULL;int rc=1,iteration=-1;
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    int bootstrap=getenv("LUMINA_DIAG_BOOTSTRAP_LTE")!=NULL;
    if(!bootstrap)CHECK(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&iteration));
    CHECK(m.plasma.n_shells==113&&(bootstrap?iteration==-1:iteration==LUMINA_FINE_EXPECTED_ITERATION)&&m.plasma.initial_steady_nlte&&!m.plasma.physical_history);
    printf("FINE_MATERIAL_INPUT mode=%s prior_iterate_read=%d initial_guess_only=1\n",bootstrap?"BOOTSTRAP_LTE_SAHA":"NATIVE_ITERATE",!bootstrap);
    AtomicData *a=&m.atom;NLTEConfig *n=&m.nlte;const int s=39,ns=m.plasma.n_shells;
    const int lines[]={644532,432305,536032,642035,267406,430299,269281,405266};
    f=fopen("lines.csv","wx");CHECK(f);
    fprintf(f,"line,Z,nu,tau,n_upper_used,n_upper_NLTE,n_lower_NLTE,integrated_chi,integrated_eta,S_line,B_nu\n");
    for(size_t q=0;q<sizeof(lines)/sizeof(lines[0]);q++){
        int l=lines[q],gl=n->nlte_line_global[2*(size_t)l],gu=n->nlte_line_global[2*(size_t)l+1];
        CHECK(gl>=0&&gu>=0);int fl=n->global_to_nlte_level[gl],fu=n->global_to_nlte_level[gu];CHECK(fl>=0&&fu>=0);
        double upper=NAN;CHECK(lumina_line_upper_population_for_tau(a,&m.plasma,n,l,s,ns,&upper)==EMISS_OK);
        double direct=n->nlte_level_populations[(size_t)fu*ns+s];
        CHECK(upper==direct);
        double nu=a->line_nu[l],tau=m.opacity.tau_sobolev[(size_t)l*ns+s];
        LineNetSobolevMaterial material;
        CHECK(!line_net_sobolev_material(upper,a->line_A_ul[l],nu,tau,m.geometry.time_explosion,LINE_NET_NEGATIVE_OPACITY_NONE,1,&material));
        double B=2*H_PLANCK*nu*nu*nu/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(H_PLANCK*nu/(K_BOLTZMANN*m.plasma.T_e[s]));
        fprintf(f,"%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",l,a->line_atomic_number[l],nu,tau,upper,direct,
            n->nlte_level_populations[(size_t)fl*ns+s],material.effective_integrated_opacity,material.emission_per_sr,
            material.emission_per_sr/material.effective_integrated_opacity,B);
    }
    {int closed=fclose(f);f=NULL;CHECK(!closed);}
    CHECK(!cmfgen_init(&cs,&m.geometry));model=&m;
    uint64_t old=n->radiation_field.field.generation.computed_generation;
    int publish=cmfgen_initial_radiation_publish(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,n,a,m.plasma.T_e[0],0);
    CHECK(captured&&publish!=0&&n->radiation_field.field.generation.computed_generation==old);
    puts("FINE_MATERIAL_DONE capture=1 intentional_formal_stop=1 new_radiation_published=0 production_input_changed=0");rc=0;
done:
    if(f)fclose(f);
    model=NULL;cmfgen_free(&cs);cmf_toy_model_free(&m);return rc;
}
