/* 같은 실제 FL 인구에서 국소 Gaussian 선 χ/η를 생산한다.
 * Sobolev 탈출확률이나 파장 마스크를 적용하지 않는다. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static size_t lower_bound(const double *x, size_t n, double y) {
    size_t a=0,b=n;
    while(a<b){size_t m=a+(b-a)/2;if(x[m]<y)a=m+1;else b=m;}
    return a;
}
static size_t upper_bound(const double *x, size_t n, double y) {
    size_t a=0,b=n;
    while(a<b){size_t m=a+(b-a)/2;if(x[m]<=y)a=m+1;else b=m;}
    return a;
}

int lumina_model_gaussian_sources(size_t nl,size_t ns,size_t nf,size_t nt,
    const double *nu,const double *population,const int32_t *lower,
    const int32_t *upper,const double *nu0,const double *f,
    const double *A,const double *g_ratio,double width_fraction,
    double support_fraction,double opacity_constant,double emission_constant,
    double *chi,double *eta) {
    if(!nl||!ns||!nf||!nt||ns>SIZE_MAX/nf||nl>SIZE_MAX/ns||
       nt>SIZE_MAX/(2*sizeof(size_t))||!nu||!population||!lower||!upper||
       !nu0||!f||!A||!g_ratio||!chi||!eta||
       !isfinite(width_fraction)||width_fraction<=0||
       !isfinite(support_fraction)||support_fraction<=0||support_fraction>=1||
       !isfinite(opacity_constant)||opacity_constant<=0||
       !isfinite(emission_constant)||emission_constant<=0) return 1;
    for(size_t b=0;b<nf;b++)
        if(!isfinite(nu[b])||nu[b]<=0||(b&&nu[b]<=nu[b-1]))return 1;
    for(size_t i=0;i<nl*ns;i++)
        if(!isfinite(population[i])||population[i]<0)return 1;
    size_t *bounds=malloc(2*nt*sizeof(size_t));if(!bounds)return 2;
    for(size_t l=0;l<nt;l++){
        if(lower[l]<0||upper[l]<0||(size_t)lower[l]>=nl||(size_t)upper[l]>=nl||
           lower[l]==upper[l]||!isfinite(nu0[l])||nu0[l]<=0||
           !isfinite(f[l])||f[l]<0||!isfinite(A[l])||A[l]<0||
           !isfinite(g_ratio[l])||g_ratio[l]<=0){free(bounds);return 1;}
        bounds[2*l]=lower_bound(nu,nf,nu0[l]*(1-support_fraction));
        bounds[2*l+1]=upper_bound(nu,nf,nu0[l]*(1+support_fraction));
    }
    if(ns*nf>SIZE_MAX/sizeof(double)){free(bounds);return 1;}
    memset(chi,0,ns*nf*sizeof(double));memset(eta,0,ns*nf*sizeof(double));
    /* 쉘별 독립 합: 스레드 수에 따라 선 합산 순서가 바뀌지 않는다. */
    #pragma omp parallel for schedule(static)
    for(size_t s=0;s<ns;s++){
        for(size_t l=0;l<nt;l++){
            double nlo=population[(size_t)lower[l]*ns+s];
            double nup=population[(size_t)upper[l]*ns+s];
            double integrated_chi=opacity_constant*f[l]*(nlo-g_ratio[l]*nup);
            double integrated_eta=emission_constant*nu0[l]*A[l]*nup;
            double width=nu0[l]*width_fraction;
            for(size_t b=bounds[2*l];b<bounds[2*l+1];b++){
                double x=(nu[b]-nu0[l])/width;
                double phi=exp(-x*x)/(1.77245385095516*width);
                chi[s*nf+b]+=integrated_chi*phi;
                eta[s*nf+b]+=integrated_eta*phi;
            }
        }
    }
    free(bounds);
    for(size_t i=0;i<ns*nf;i++)
        if(!isfinite(chi[i])||!isfinite(eta[i])||eta[i]<0)return 3;
    return 0;
}
