#ifndef LUMINA_BF_CHANNELS_H
#define LUMINA_BF_CHANNELS_H
#include <math.h>
#include <stddef.h>
#include <string.h>

/* 불변 원자 입력. sigma는 채널별 전 bin 평균이며 파일 내부 분기 가중을
 * 이미 포함한다. A/B 파일 사이에 확률합1을 강제하지 않는다. */
typedef struct {
    int n_levels,n_channels,n_bins;
    double nu_min,nu_max;
    int *offset,*upper;
    double *threshold,*sigma;
} BfChannels;

typedef struct {
    double chi_absorption,chi_net,eta;
    double ionization,recombination;
    double energy_exchange;
} BfChannelBin;

/* 한 물리 bin의 동일 적분으로 율과 chi/eta를 만든다. 저장된 전체-bin
 * 평균 sigma를 threshold 위의 부분구간에 배치한다. J는 그 bin의 평균이다.
 * planck_field는 직접 LTE 시험용이며 생산에서는0이다.
 * ionization/recombination은 상/하위 입자당 s^-1, chi/eta는 bin 평균,
 * energy_exchange는 n_lower R_up - n_upper R_down의 광자 에너지 적분이다.
 * sigma=0은 명시적0, n_lower=0이어도 recombination/eta는 남는다. */
static inline int bf_channel_bin(double lo,double hi,double threshold,double sigma,
    double T,double ne,double gl,double gu,double nl,double nu,double J,
    int planck_field,BfChannelBin *out)
{
    if(!out||!(lo>0)||!(hi>lo)||!isfinite(hi)||!(threshold>0)||!isfinite(threshold)||
       !(sigma>=0)||!isfinite(sigma)||!(T>0)||!isfinite(T)||!(ne>=0)||!isfinite(ne)||
       !(gl>0)||!isfinite(gl)||!(gu>0)||!isfinite(gu)||!(nl>=0)||!isfinite(nl)||
       !(nu>=0)||!isfinite(nu)||!(J>=0)||!isfinite(J))return -1;
    BfChannelBin result={0};
    if(sigma==0){*out=result;return 0;}
    if(hi<=threshold)return -1;
    const double h=6.62607015e-27,k=1.380649e-16,c=2.99792458e10;
    const double me=9.1093837015e-28,pi=3.14159265358979323846;
    const double x[4]={.18343464249564980494,.52553240991632898582,
                       .79666647741362673959,.96028985649753623168};
    const double w[4]={.36268378337836198297,.31370664587788728734,
                       .22238103445337447054,.10122853629037625915};
    double left=fmax(lo,threshold),mid=.5*(left+hi),half=.5*(hi-left);
    double saha=ne*pow(h*h/(2*pi*me*k*T),1.5)*gl/(2*gu);
    if(!isfinite(saha))return -1;
    for(int q=0;q<4;q++)for(int sign=-1;sign<=1;sign+=2){
        double f=mid+sign*half*x[q];
        double photon=h*f,spont=2*h*f*f*f/(c*c);
        double inv=saha*exp(-h*(f-threshold)/(k*T));
        double j=J;
        if(planck_field){double z=photon/(k*T);j=z>700?spont*exp(-z):spont/expm1(z);}
        double weight=.5*w[q]*sigma;
        double absorption=nl*weight,stim=nu*weight*inv;
        double eta=stim*spont;
        result.chi_absorption+=absorption;
        result.chi_net+=absorption-stim;
        result.eta+=eta;
        result.ionization+=4*pi*weight*j/photon*(hi-lo);
        result.recombination+=4*pi*weight*inv*(spont+j)/photon*(hi-lo);
        result.energy_exchange+=4*pi*((absorption-stim)*j-eta)*(hi-lo);
    }
    if(!isfinite(result.chi_absorption)||!isfinite(result.chi_net)||!isfinite(result.eta)||
       !isfinite(result.ionization)||!isfinite(result.recombination)||
       !isfinite(result.energy_exchange))return -1;
    *out=result;return 0;
}
#endif
