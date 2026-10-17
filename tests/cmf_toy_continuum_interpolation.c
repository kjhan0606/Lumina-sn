/* 저장된 같은 J에서 구간 상수와 현행 수송의 계수 보간만 비교한다.
 * 실제 BF+FF C producer를 사용한다. 새 RTE/SE나 물리 이력을 만들지 않는다. */
#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include <float.h>
#include <stdio.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"CONT_INTERP_FAIL line=%d %s\n",__LINE__,#x);goto done;}}while(0)

/* lumina_cmfgen.c의 common fine continuum 소비식과 같은 중심 보간.
 * 실제 producer 재실행이 아니라 저장장에 대한 동일 식의 판독이다. */
static double interp(const double *a,int nb,double nu,double lo,double dl){
    double x=log(nu/lo)/dl-.5;
    int b=(int)floor(x);double f=x-b;
    if(b<0){b=0;f=0;}
    if(b>=nb-1){b=nb-2;f=1;}
    return (1-f)*a[b]+f*a[b+1];
}

int main(int argc,char **argv){
    if(argc!=6)return 2;
    int rc=1,iteration=-1;
    CMFToyModel m={0};FILE *fp=NULL,*spectral=NULL;double *coarse=NULL;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    double a[3]={2,4,6};
    CHECK(fabs(interp(a,3,exp(1.),1,1)-3)<1e-14);
    CHECK(interp(a,3,.5,1,1)==2&&interp(a,3,exp(4.),1,1)==6);
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],atoi(argv[3]),mass));
    CHECK(!cmf_toy_initial_iterate_read(&m,argv[4],argv[5],&iteration));
    CHECK(iteration==4&&m.plasma.n_shells==113);
    bf_set_nlte_pops(&m.nlte);
    compute_bf_opacity(&m.bf,&m.atom,&m.plasma,m.plasma.n_shells);
    const BFOpacity *bf=&m.bf;
    int ns=m.plasma.n_shells,nb=bf->n_freq_bins,nf=m.opacity.n_fine;
    CHECK(nb==1234&&nf==117606);
    coarse=malloc((size_t)ns*nb*sizeof(*coarse));CHECK(coarse);
    CMFGENState source={0},target={0};
    source.n_shells=ns;source.n_bins=nf;source.J=m.opacity.jnu_fine;
    source.nu_min=m.opacity.nu_lo_fine;source.d_log_nu=m.opacity.dlognu_fine;
    source.nu_max=source.nu_min*exp(nf*source.d_log_nu);
    target.n_shells=ns;target.n_bins=nb;target.J=coarse;
    target.nu_min=bf->nu_min;target.nu_max=bf->nu_max;target.d_log_nu=bf->d_log_nu;
    CHECK(!cmfgen_rebin_log_field(&source,&target));
    CHECK(source.nu_min<=bf->nu_min&&source.nu_max>=bf->nu_max);
    if(getenv("LUMINA_TEST_CONTINUUM_SPECTRAL")){
        spectral=fopen("continuum_spectral.csv","wx");CHECK(spectral);
        CHECK(fprintf(spectral,"shell,bin,energy_eV,T_K,dr_cm,chi_cont_cm_inv,eta_cont,chi_es_cm_inv,J_nu,B_nu_center\n")>0);
        for(int s=37;s<=44;s++){
            CHECK(bf->eta_milne_ok[s]);
            double dr=m.geometry.r_outer[s]-m.geometry.r_inner[s];CHECK(dr>0);
            for(int b=0;b<nb;b++){
                size_t k=(size_t)s*nb+b;
                double nu=bf->nu_min*exp((b+.5)*bf->d_log_nu);
                double B=2*H_PLANCK*nu*nu*nu/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(H_PLANCK*nu/(K_BOLTZMANN*m.plasma.T_e[s]));
                CHECK(fprintf(spectral,"%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    s,b,H_PLANCK*nu/EV_TO_ERG,m.plasma.T_e[s],dr,bf->chi_bf[k],bf->eta_bf[k],
                    m.plasma.n_electron[s]*SIGMA_THOMSON,coarse[k],B)>0);
            }
        }
        {int closed=fclose(spectral);spectral=NULL;CHECK(!closed);}
    }
    fp=fopen("continuum_interpolation.csv","wx");CHECK(fp);
    CHECK(fprintf(fp,"shell,velocity_km_s,T_K,absorption_constant,emission_constant,absorption_interpolated,emission_interpolated,absorption_relative_change,emission_relative_change,net_difference_over_scale,L1_absorption_difference_over_scale,L1_emission_difference_over_scale,rebin_relative_error\n")>0);
    double max_rebin=0,max_net=0,max_abs=0,max_eta=0;int worst_net=-1;
    for(int s=0;s<ns;s++){
        CHECK(bf->eta_milne_ok[s]);
        const double *chi=bf->chi_bf+(size_t)s*nb,*eta=bf->eta_bf+(size_t)s*nb;
        long double A=0,E=0,AI=0,EI=0,AD=0,ED=0,AR=0,abs_scale=0;
        int f=0;
        for(int b=0;b<nb;b++){
            double lo=bf->nu_min*exp(b*bf->d_log_nu);
            double hi=bf->nu_min*exp((b+1)*bf->d_log_nu);
            CHECK(isfinite(chi[b])&&isfinite(eta[b])&&eta[b]>=0);
            AR+=(long double)chi[b]*coarse[(size_t)s*nb+b]*(hi-lo);
            E+=(long double)eta[b]*(hi-lo);
            while(f<nf&&source.nu_min*exp((f+1)*source.d_log_nu)<=lo)f++;
            long double covered=0;
            for(int k=f;k<nf;k++){
                double fl=source.nu_min*exp(k*source.d_log_nu);
                if(fl>=hi)break;
                double fh=source.nu_min*exp((k+1)*source.d_log_nu);
                double w=fmin(hi,fh)-fmax(lo,fl);
                if(w<=0)continue;
                double nu=source.nu_min*exp((k+.5)*source.d_log_nu);
                double j=source.J[(size_t)s*nf+k];
                double ci=interp(chi,nb,nu,bf->nu_min,bf->d_log_nu);
                double ei=interp(eta,nb,nu,bf->nu_min,bf->d_log_nu);
                CHECK(isfinite(j)&&j>=0&&isfinite(ci)&&isfinite(ei)&&ei>=0);
                A+=(long double)chi[b]*j*w;AI+=(long double)ci*j*w;
                EI+=(long double)ei*w;
                AD+=fabsl((long double)(ci-chi[b])*j*w);
                ED+=fabsl((long double)(ei-eta[b])*w);
                abs_scale+=fabsl((long double)chi[b]*j*w);covered+=w;
            }
            CHECK(fabsl(covered-(hi-lo))<=1e-10L*(hi-lo));
        }
        long double scale=fmaxl(abs_scale,E);CHECK(scale>0);
        double er=(double)(fabsl(A-AR)/scale);CHECK(er<1e-10);
        double da=(double)((AI-A)/scale),de=(double)((EI-E)/scale);
        double dn=(double)(((AI-EI)-(A-E))/scale);
        /* 여기의 상대 흡수/방출은 각각 자신의 크기를 분모로 쓰고,
         * 순수지는 상쇄 전 흡수 절대적분/방출 중 큰 값으로 정규화한다. */
        CHECK(A!=0&&E>0);
        CHECK(fprintf(fp,"%d,%.17g,%.17g,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17g,%.17Lg,%.17Lg,%.17g\n",
            s,(m.geometry.v_inner[s]+m.geometry.v_outer[s])/2e5,m.plasma.T_e[s],
            4*M_PI_VAL*A,4*M_PI_VAL*E,4*M_PI_VAL*AI,4*M_PI_VAL*EI,(AI-A)/A,(EI-E)/E,dn,AD/scale,ED/scale,er)>0);
        max_rebin=fmax(max_rebin,er);max_abs=fmax(max_abs,fabs(da));max_eta=fmax(max_eta,fabs(de));
        if(fabs(dn)>max_net){max_net=fabs(dn);worst_net=s;}
    }
    {int closed=fclose(fp);fp=NULL;CHECK(!closed);}
    printf("CONTINUUM_INTERPOLATION_DONE shells=%d iteration=%d rebin_max=%.17g max_absorption_change_over_scale=%.17g max_emission_change_over_scale=%.17g max_net_change_over_scale=%.17g worst_net_shell=%d RTE_solved=0 SE_solved=0 physical_history=0 goal=NOT_JUDGED\n",ns,iteration,max_rebin,max_abs,max_eta,max_net,worst_net);
    rc=0;
done:
    if(fp)fclose(fp);
    if(spectral)fclose(spectral);
    free(coarse);bf_set_nlte_pops(NULL);cmf_toy_model_free(&m);return rc;
}
