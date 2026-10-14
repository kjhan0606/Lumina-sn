/* 직접시험: 초신성/CMFGEN 일치 판정이 아니다. */
#include "cmf_observer_ray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"OBSERVER_FAIL line=%d: %s\n",__LINE__,#x);return 1;}}while(0)

/* 생산 광선 분할/역변환 없이 균일한 공간 보폭으로 적분한다. */
static double reference_segment(const CMFObserverInput *in, int s, double p,
        double za, double zb, double nu, double I) {
    const int steps=100000;
    double dz=(zb-za)/steps;
    for(int j=0;j<steps;++j) {
        double z=za+(j+0.5)*dz;
        double bz=in->t_exp?z/(2.99792458e10*in->t_exp):0;
        double bp=in->t_exp?p/(2.99792458e10*in->t_exp):0;
        double q=(1-bz)/sqrt(1-bz*bz-bp*bp);
        int b=(int)floor(log(nu*q/in->nu_min)/in->dlognu);
        if(b<0 || b>=in->nf)return NAN;
        size_t k=(size_t)s*in->nf+b;
        double opacity=q*in->chi[k];
        double emission=(in->chi[k]*in->source[k]+in->es[k]*in->J[k])/(q*q);
        double e=exp(-opacity*dz);
        I=opacity?I*e+emission/opacity*(-expm1(-opacity*dz)):I+emission*dz;
    }
    return I;
}
static double reference_ray(const CMFObserverInput *in,double p,double nu) {
    double I=0;
    if(p<in->ri[0]) {
        double z=sqrt(in->ri[0]*in->ri[0]-p*p);
        double beta=in->t_exp?in->ri[0]/(2.99792458e10*in->t_exp):0;
        double mu=z/in->ri[0], q=(1-beta*mu)/sqrt(1-beta*beta);
        double x=6.62607015e-27*q*nu/(1.380649e-16*in->T_inner);
        I=in->T_inner?in->inner_scale*2*6.62607015e-27*nu*nu*nu/
          (2.99792458e10*2.99792458e10)/expm1(x):0;
    } else for(int s=in->ns-1;s>=0;--s) {
        if(p>=in->ro[s])continue;
        double zo=sqrt(in->ro[s]*in->ro[s]-p*p);
        double zi=p<in->ri[s]?sqrt(in->ri[s]*in->ri[s]-p*p):0;
        I=reference_segment(in,s,p,-zo,-zi,nu,I);
    }
    for(int s=0;s<in->ns;++s) {
        if(p>=in->ro[s])continue;
        double zo=sqrt(in->ro[s]*in->ro[s]-p*p);
        double zi=p<in->ri[s]?sqrt(in->ri[s]*in->ri[s]-p*p):0;
        I=reference_segment(in,s,p,zi,zo,nu,I);
    }
    return I;
}

#ifdef OBSERVER_TEST_GPU
static int benchmark(void) {
    enum {NS=50,NF=524288,NO=64};
    size_t cells=(size_t)NS*NF;
    double *data=(double*)malloc(4*cells*sizeof(double));
    double ri[NS],ro[NS],nu[NO],L[NO];
    CHECK(data);
    CMFObserverInput in={0};
    in.ns=NS;in.nf=NF;in.nu_min=5e13;in.dlognu=log(1000.0)/NF;
    in.t_exp=19.48*86400;in.T_inner=10000;in.inner_scale=1;
    in.ri=ri;in.ro=ro;in.chi=data;in.es=data+cells;
    in.source=data+2*cells;in.J=data+3*cells;
    in.nobs=NO;in.nu_obs=nu;in.ncore=32;in.nouter=64;in.substeps=1;
    for(int s=0;s<NS;++s){ri[s]=3e14+2.7e15*s/NS;ro[s]=3e14+2.7e15*(s+1)/NS;}
    for(size_t k=0;k<cells;++k) {
        data[k]=1e-14*(1+0.1*(k%37));data[cells+k]=2e-15;
        data[2*cells+k]=1e-5;data[3*cells+k]=2e-5;
    }
    for(int f=0;f<NO;++f)nu[f]=(2.99792458e18/25500.0)*exp(log(25500.0/2500.0)*f/(NO-1));
    struct timeval a,b;gettimeofday(&a,NULL);
    CHECK(cmf_observer_gpu(&in,L)==0);
    gettimeofday(&b,NULL);
    double sec=b.tv_sec-a.tv_sec+1e-6*(b.tv_usec-a.tv_usec);
    for(int f=0;f<NO;++f)CHECK(isfinite(L[f]) && L[f]>0);
    printf("OBSERVER_GPU_COST shells=%d fine_bins=%d observer_bins=%d rays=96 "
        "material_bytes=%zu wall_seconds=%.6f per_frequency_seconds=%.9f "
        "synthetic_cost_only=1\n",NS,NF,NO,4*cells*sizeof(double),sec,sec/NO);
    free(data);return 0;
}
#endif

int main(int argc,char **argv) {
#ifdef OBSERVER_TEST_GPU
    if(argc==2 && !strcmp(argv[1],"--bench"))return benchmark();
#else
    (void)argc;(void)argv;
#endif
    enum {NS=3,NF=2048,NO=7};
    double ri[NS]={1e14,1.3e14,1.6e14},ro[NS]={1.3e14,1.6e14,2e14};
    double chi[NS*NF],es[NS*NF],src[NS*NF],J[NS*NF];
    double nu[NO],L[NO],save[NO];
    CMFObserverInput in={0};
    in.ns=NS;in.nf=NF;in.nu_min=1e14;in.dlognu=log(10.0)/NF;
    in.T_inner=10000;in.inner_scale=0.7;
    in.ri=ri;in.ro=ro;in.chi=chi;in.es=es;in.source=src;in.J=J;
    in.nobs=NO;in.nu_obs=nu;in.ncore=32;in.nouter=64;in.substeps=1;
    const double pi=acos(-1.0);
    for(int f=0;f<NO;++f)nu[f]=3e14*exp(0.08*f);
    for(int i=0;i<NS*NF;++i){chi[i]=0;es[i]=0;src[i]=0;J[i]=1e-5;}
    CHECK(cmf_observer_cpu(&in,L)==0);
    double worst=0;
    for(int f=0;f<NO;++f) {
        double bb=2*6.62607015e-27*nu[f]*nu[f]*nu[f]/
            (2.99792458e10*2.99792458e10)/expm1(6.62607015e-27*nu[f]/1.380649e-16/10000);
        double expected=4*pi*pi*ri[0]*ri[0]*bb*in.inner_scale;
        double err=fabs(L[f]/expected-1);if(err>worst)worst=err;
    }
    printf("static_core_absolute_luminosity_relative=%.17g\n",worst);
    CHECK(worst<1e-11);
    /* 정적 균질 외피: 독립적인 연속 각도 적분과 비교한다. */
    for(int i=0;i<NS*NF;++i){chi[i]=2e-14;es[i]=5e-15;src[i]=3e-5;}
    CHECK(cmf_observer_cpu(&in,L)==0);
    double integ=0, umax=sqrt(1-ri[0]*ri[0]/ro[NS-1]/ro[NS-1]);
    for(int j=0;j<200000;++j) {
        double u=(j+0.5)/200000, p=ri[0]*sqrt(1-u*u);
        double ds=sqrt(ro[NS-1]*ro[NS-1]-p*p)-ri[0]*u;
        double I=obs_bb(nu[0],in.T_inner)*in.inner_scale*exp(-2e-14*ds)+
                 3.25e-5*(-expm1(-2e-14*ds));
        integ+=ri[0]*ri[0]*u*I/200000;
        u=(j+0.5)*umax/200000;
        I=3.25e-5*(-expm1(-2e-14*2*ro[NS-1]*u));
        integ+=ro[NS-1]*ro[NS-1]*u*I*umax/200000;
    }
    double err=fabs(L[0]/(8*pi*pi*integ)-1);
    printf("static_homogeneous_continuous_integral_relative=%.17g\n",err);
    CHECK(err<1e-3);
    /* 팽창 연속장과 주파수 가변 Gaussian 선. 같은 J의 산란 방출을 포함. */
    in.t_exp=2e14/(0.12*2.99792458e10);
    for(int s=0;s<NS;++s)for(int b=0;b<NF;++b) {
        size_t k=(size_t)s*NF+b;
        double x=(log(in.nu_min)+(b+0.5)*in.dlognu-log(3.6e14))/0.012;
        chi[k]=1e-14+2e-13*exp(-x*x)*(s+1);
        es[k]=3e-15;src[k]=(1+s)*1e-5;J[k]=2e-5*(1+0.3*sin(b*0.02));
    }
    worst=0;
    for(int r=0;r<3;++r)for(int f=0;f<NO;f+=2) {
        double p=(0.4+0.6*r)*1e14;
        double a=cmf_observer_ray(&in,p,nu[f]);
        double b=reference_ray(&in,p,nu[f]);
        err=fabs(a/b-1);if(err>worst)worst=err;
    }
    printf("expanding_gaussian_independent_space_integral_relative=%.17g\n",worst);
    CHECK(worst<1e-4);
    CHECK(cmf_observer_cpu(&in,L)==0);
#ifdef OBSERVER_TEST_GPU
    CHECK(cmf_observer_gpu(&in,save)==0);
    worst=0;
    for(int f=0;f<NO;++f){err=fabs(save[f]/L[f]-1);if(err>worst)worst=err;}
    printf("cpu_gpu_relative=%.17g\n",worst);CHECK(worst<1e-10);
#endif
    memcpy(save,L,sizeof L);
    chi[0]=-1;
    CHECK(cmf_observer_cpu(&in,L)!=0 && memcmp(save,L,sizeof L)==0);
    chi[0]=1e-14;nu[0]=1.01e14;
    CHECK(cmf_observer_cpu(&in,L)!=0 && memcmp(save,L,sizeof L)==0);
    nu[0]=3e14;J[NS*NF-1]=NAN;
    CHECK(cmf_observer_cpu(&in,L)!=0 && memcmp(save,L,sizeof L)==0);
    puts("CMF_OBSERVER_SELFTEST_PASS invalid_preserves_output=1");return 0;
}
