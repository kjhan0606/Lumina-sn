/* 같은 실제 FL 인구에서 국소 Gaussian 선 χ/η를 생산한다.
 * Sobolev 탈출확률이나 파장 마스크를 적용하지 않는다. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "model_gaussian_profile.h"

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
            for(size_t b=bounds[2*l];b<bounds[2*l+1];b++){
                double phi=model_gaussian_phi(nu[b],nu0[l],width_fraction);
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

/* 원래점 하나의 source-profile 적분. status: 1=VALID,2=EXACT_ZERO,
 * 3=OUT_OF_GRID,4=UNSAMPLED. 후자의 Jbar는 NaN이며 0으로 대체하지 않는다.
 * profile_mass는 관측값일 뿐 Jbar를 나누는 보정 인자가 아니다. */
int lumina_model_point_gaussian_jbar(size_t nf,size_t nt,const double *nu,
    const double *dnu,const double *j,const double *nu0,double width_fraction,
    double support_fraction,double *jbar,double *profile_mass,uint8_t *status)
{
    if(nf<2 || !nt || !nu || !dnu || !j || !nu0 || !jbar || !profile_mass || !status ||
       nt>SIZE_MAX/sizeof(double) || !isfinite(width_fraction) || width_fraction<=0. ||
       !isfinite(support_fraction) || support_fraction<=0. || support_fraction>=1.)return 1;
    for(size_t k=0;k<nf;k++)if(!isfinite(nu[k]) || nu[k]<=0. || (k && nu[k]<=nu[k-1]) ||
        !isfinite(dnu[k]) || dnu[k]<0. || !isfinite(j[k]) || j[k]<0.)return 1;
    for(size_t l=0;l<nt;l++)if(!isfinite(nu0[l]) || nu0[l]<=0. ||
        !isfinite(nu0[l]*(1.+support_fraction)) ||
        !isfinite(nu0[l]*width_fraction) || nu0[l]*width_fraction<=0.)return 1;
    double *values=malloc(nt*sizeof(double)),*masses=malloc(nt*sizeof(double));
    uint8_t *states=malloc(nt);
    if(!values || !masses || !states){free(values);free(masses);free(states);return 2;}
    for(size_t l=0;l<nt;l++){
        double lo=nu0[l]*(1.-support_fraction),hi=nu0[l]*(1.+support_fraction);
        if(lo<nu[0] || hi>nu[nf-1]){values[l]=NAN;masses[l]=NAN;states[l]=3;continue;}
        size_t first=lower_bound(nu,nf,lo),last=upper_bound(nu,nf,hi);
        if(first==last){values[l]=NAN;masses[l]=0.;states[l]=4;continue;}
        long double value=0.,mass=0.;
        for(size_t k=first;k<last;k++){
            double phi=model_gaussian_phi(nu[k],nu0[l],width_fraction);
            long double weight=(long double)phi*dnu[k];
            mass+=weight;value+=weight*j[k];
        }
        values[l]=(double)value;masses[l]=(double)mass;
        if(!isfinite(values[l]) || values[l]<0. || !isfinite(masses[l]) || masses[l]<=0.){
            free(values);free(masses);free(states);return 3;
        }
        states[l]=values[l]>0. ? 1 : 2;
    }
    memcpy(jbar,values,nt*sizeof(double));memcpy(profile_mass,masses,nt*sizeof(double));
    memcpy(status,states,nt);free(values);free(masses);free(states);return 0;
}

/* 비교 전용: native 정수 공명구간의 원 전역 가중치 적분.
 * status5는 native 격자 미등록 선이며 물리적인 0율 판정이 아니다. */
int lumina_model_point_gaussian_native_jbar(size_t nf,size_t nt,const double *nu,
    const double *dnu,const double *j,const double *nu0,const size_t *first,
    const size_t *last,const uint8_t *active,double width_fraction,
    double *jbar,double *profile_mass,uint8_t *status)
{
    if(nf<2 || !nt || !nu || !dnu || !j || !nu0 || !first || !last || !active ||
       !jbar || !profile_mass || !status || !isfinite(width_fraction) ||
       width_fraction<=0. || nt>SIZE_MAX/sizeof(double))return 1;
    for(size_t k=0;k<nf;k++)if(!isfinite(nu[k]) || nu[k]<=0. ||
       (k&&nu[k]<=nu[k-1]) || !isfinite(dnu[k]) || dnu[k]<0. ||
       !isfinite(j[k]) || j[k]<0.)return 1;
    for(size_t l=0;l<nt;l++)if(!isfinite(nu0[l]) || nu0[l]<=0. || active[l]>1 ||
       (active[l] && (first[l]>=last[l] || last[l]>nf)) ||
       (!active[l] && (first[l] || last[l])))return 1;
    double *values=malloc(nt*sizeof(double)),*masses=malloc(nt*sizeof(double));
    uint8_t *states=malloc(nt);
    if(!values||!masses||!states){free(values);free(masses);free(states);return 2;}
    for(size_t l=0;l<nt;l++){
        long double value=0.,mass=0.;
        if(active[l])for(size_t k=first[l];k<last[l];k++){
            long double weight=(long double)model_gaussian_phi(nu[k],nu0[l],width_fraction)*dnu[k];
            mass+=weight;value+=weight*j[k];
        }
        values[l]=(double)value;masses[l]=(double)mass;
        states[l]=active[l]?(values[l]>0.?1:2):5;
        if(!isfinite(values[l])||values[l]<0.||!isfinite(masses[l])||masses[l]<0.){
            free(values);free(masses);free(states);return 3;
        }
    }
    memcpy(jbar,values,nt*sizeof(double));memcpy(profile_mass,masses,nt*sizeof(double));
    memcpy(status,states,nt);free(values);free(masses);free(states);return 0;
}
