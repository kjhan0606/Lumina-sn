/* nodal J의 ν축 선형 표현과 기존 Gaussian의 적분.
 * 내부 표본 부재와 실제 장 범위 결손을 구분하며 외삽/재정규화를 하지 않는다. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "model_gaussian_profile.h"

/* model_gaussian_profile.h와 같은 분모. source의 기존 정규화를 바꾸지 않는다. */
#define PROFILE_DENOM 1.77245385095516L

static size_t upper_index(const double *x,size_t n,double value)
{
    size_t lo=0,hi=n;
    while(lo<hi){size_t m=lo+(hi-lo)/2;if(x[m]<=value)lo=m+1;else hi=m;}
    return lo;
}

static int interval_weights(long double a,long double b,long double left,long double right,
                            long double *wl,long double *wr)
{
    if(!(left<=a && a<b && b<=right && left<right))return -1;
    long double width=right-left;
    *wl=*wr=0.;
    if(b-a<.05L){
        /* 좁은 구간의 모멘트 차감은 같은 적분의 양의 GL8 표현으로 계산한다.
         * 길이/오차의 수치 분할이며 작은 광율을0으로 자르는 조건이 아니다. */
        static const long double x[4]={.183434642495649804939476142360184L,
            .525532409916328985817739049189246L,.796666477413626739591553936475831L,
            .960289856497536231683560868569473L};
        static const long double w[4]={.362683783378361982965150449277195L,
            .313706645877887287337962201986601L,.222381034453374470544355994426240L,
            .101228536290376259152531354309962L};
        long double half=(b-a)/2,mid=(a+b)/2;
        for(int k=0;k<4;k++)for(int sign=-1;sign<=1;sign+=2){
            long double node=sign*x[k],u=mid+half*node;
            long double q=half*w[k]*expl(-u*u)/PROFILE_DENOM;
            *wl+=q*((right-b)+half*(1-node))/width;
            *wr+=q*((a-left)+half*(1+node))/width;
        }
    }else{
        long double mass;
        if(a>=0)mass=erfcl(a)-erfcl(b);
        else if(b<=0)mass=erfcl(-b)-erfcl(-a);
        else mass=erfl(b)-erfl(a);
        mass*=sqrtl(acosl(-1.L))/(2*PROFILE_DENOM);
        long double first;
        if(a*a<=b*b)first=-expl(-a*a)*expm1l(a*a-b*b)/(2*PROFILE_DENOM);
        else first=expl(-b*b)*expm1l(b*b-a*a)/(2*PROFILE_DENOM);
        *wl=(right*mass-first)/width;*wr=(first-left*mass)/width;
    }
    if(!isfinite(*wl) || !isfinite(*wr) || *wl<0. || *wr<0.)return -1;
    return 0;
}

/* J는 frequency-major [nf*np], Jbar는 line-major [nt*np].
 * status1=VALID,2=EXACT_ZERO(전점),3=OUT_OF_GRID; profile mass로 나누지 않는다.
 * 실패 시 출력 전체는 보존된다. 범위 밖 선은 NaN+status3으로 명시한다. */
int lumina_model_linear_gaussian_jbar(size_t nf,size_t np,size_t nt,
    const double *nu,const double *j,const double *nu0,double width_fraction,double support_fraction,
    double *jbar,double *mass,uint8_t *status)
{
    if(nf<2 || !np || !nt || !nu || !j || !nu0 || !jbar || !mass || !status ||
       np>SIZE_MAX/nf || np>SIZE_MAX/nt || nt*np>SIZE_MAX/sizeof(long double) ||
       nt>SIZE_MAX/sizeof(double) || !isfinite(width_fraction) || width_fraction<=0. ||
       !isfinite(support_fraction) || support_fraction<=0. || support_fraction>=1.)return 1;
    for(size_t k=0;k<nf;k++)if(!isfinite(nu[k]) || nu[k]<=0. || (k && nu[k]<=nu[k-1]))return 1;
    for(size_t i=0;i<nf*np;i++)if(!isfinite(j[i]) || j[i]<0.)return 1;
    for(size_t l=0;l<nt;l++)if(!isfinite(nu0[l]) || nu0[l]<=0. ||
        !isfinite(nu0[l]*width_fraction) || nu0[l]*width_fraction<=0. ||
        !isfinite(nu0[l]*(1+support_fraction)))return 1;
    long double *work=calloc(nt*np,sizeof(*work));
    double *masses=malloc(nt*sizeof(*masses));uint8_t *states=malloc(nt);
    if(!work || !masses || !states){free(work);free(masses);free(states);return 2;}
    int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for(size_t l=0;l<nt;l++){
        double low=nu0[l]*(1-support_fraction),high=nu0[l]*(1+support_fraction);
        if(low<nu[0] || high>nu[nf-1]){
            for(size_t p=0;p<np;p++)work[l*np+p]=NAN;
            masses[l]=NAN;states[l]=3;continue;
        }
        size_t k=upper_index(nu,nf,low);
        if(k)k--;
        long double width=(long double)nu0[l]*width_fraction,integrated_mass=0.;
        for(;k+1<nf && nu[k]<high;k++){
            long double a=fmaxl(low,nu[k]),b=fminl(high,nu[k+1]);
            if(a>=b)continue;
            long double left=((long double)nu[k]-nu0[l])/width;
            long double right=((long double)nu[k+1]-nu0[l])/width;
            a=(a-nu0[l])/width;b=(b-nu0[l])/width;
            long double wl,wr;
            if(interval_weights(a,b,left,right,&wl,&wr)){failed=1;break;}
            integrated_mass+=wl+wr;
            for(size_t p=0;p<np;p++)work[l*np+p]+=wl*j[k*np+p]+wr*j[(k+1)*np+p];
        }
        masses[l]=(double)integrated_mass;states[l]=2;
        if(!isfinite(masses[l]) || masses[l]<=0.)failed=1;
        for(size_t p=0;p<np;p++){
            double value=(double)work[l*np+p];
            if(!isfinite(value) || value<0.)failed=1;
            if(value>0.)states[l]=1;
        }
    }
    if(!failed){
        for(size_t i=0;i<nt*np;i++)jbar[i]=(double)work[i];
        memcpy(mass,masses,nt*sizeof(*mass));memcpy(status,states,nt);
    }
    free(work);free(masses);free(states);return failed?3:0;
}

/* 실제 Gaussian 점표본 χ/η와 같은 사다리꼴 내적.
 * source의 선/주파수 지지와 Gaussian 평가를 공유한다. 질량을 나누거나
 * 미해상/범위 밖 선을0으로 채우지 않는다. spontaneous에는 A*mass를 쓴다.
 * J는 frequency-major, 출력은 line-major이며 실패 시 출력은 보존한다. */
int lumina_model_sampled_gaussian_jbar(size_t nf,size_t np,size_t nt,
    const double *nu,const double *j,const double *nu0,double width_fraction,double support_fraction,
    double *jbar,double *mass,uint8_t *status)
{
    if(nf<2 || !np || !nt || !nu || !j || !nu0 || !jbar || !mass || !status ||
       np>SIZE_MAX/nf || np>SIZE_MAX/nt || nt*np>SIZE_MAX/sizeof(long double) ||
       nt>SIZE_MAX/sizeof(double) || !isfinite(width_fraction) || width_fraction<=0. ||
       !isfinite(support_fraction) || support_fraction<=0. || support_fraction>=1.)return 1;
    for(size_t k=0;k<nf;k++)if(!isfinite(nu[k]) || nu[k]<=0. || (k && nu[k]<=nu[k-1]))return 1;
    for(size_t i=0;i<nf*np;i++)if(!isfinite(j[i]) || j[i]<0.)return 1;
    for(size_t l=0;l<nt;l++)if(!isfinite(nu0[l]) || nu0[l]<=0. ||
        !isfinite(nu0[l]*width_fraction) || nu0[l]*width_fraction<=0. ||
        !isfinite(nu0[l]*(1+support_fraction)))return 1;
    long double *work=calloc(nt*np,sizeof(*work));
    double *masses=malloc(nt*sizeof(*masses));uint8_t *states=malloc(nt);
    if(!work || !masses || !states){free(work);free(masses);free(states);return 2;}
    int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for(size_t l=0;l<nt;l++){
        double low=nu0[l]*(1-support_fraction),high=nu0[l]*(1+support_fraction);
        if(low<nu[0] || high>nu[nf-1]){
            for(size_t p=0;p<np;p++)work[l*np+p]=NAN;
            masses[l]=NAN;states[l]=3;continue;
        }
        size_t k=upper_index(nu,nf,low);
        if(k && nu[k-1]==low)k--;
        long double integrated_mass=0.;
        for(;k<nf && nu[k]<=high;k++){
            double dnu=0.;
            if(k)dnu+=.5*(nu[k]-nu[k-1]);
            if(k+1<nf)dnu+=.5*(nu[k+1]-nu[k]);
            long double weight=(long double)model_gaussian_phi(nu[k],nu0[l],width_fraction)*dnu;
            integrated_mass+=weight;
            for(size_t p=0;p<np;p++)work[l*np+p]+=weight*j[k*np+p];
        }
        masses[l]=(double)integrated_mass;states[l]=2;
        if(integrated_mass==0.){
            for(size_t p=0;p<np;p++)work[l*np+p]=NAN;
            states[l]=4;continue;
        }
        if(!isfinite(masses[l]) || masses[l]<=0.)failed=1;
        for(size_t p=0;p<np;p++){
            double value=(double)work[l*np+p];
            if(!isfinite(value) || value<0.)failed=1;
            if(value>0.)states[l]=1;
        }
    }
    if(!failed){
        for(size_t i=0;i<nt*np;i++)jbar[i]=(double)work[i];
        memcpy(mass,masses,nt*sizeof(*mass));memcpy(status,states,nt);
    }
    free(work);free(masses);free(states);return failed?3:0;
}
