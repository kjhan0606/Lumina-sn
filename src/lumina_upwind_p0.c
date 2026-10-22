/* 공통 P0 물질 구간의 CMF 주파수 1차 청색 상풍 비교기.
 * CMFGEN의 3점 공간 SC는 재현하지 않는다. 생산 수송에 연결되지 않는다. */
#include <math.h>
#include <stddef.h>

static int linear_weights(double tau,double length,double *attenuation,
    double *upstream,double *downstream)
{
    *attenuation=exp(-tau);
    if(fabs(tau)<1e-3){
        double term=1.,a=0.,b=0.;
        for(int k=0;k<16;k++){
            a+=term/(k+2.);
            b+=term/((k+1.)*(k+2.));
            term*=-tau/(k+1.);
        }
        *upstream=length*a;
        *downstream=length*b;
    }else{
        double mean=-expm1(-tau)/tau;
        *downstream=length*(1.-mean)/tau;
        *upstream=length*mean-*downstream;
    }
    return !isfinite(*attenuation)||!isfinite(*upstream)||!isfinite(*downstream)||
        *upstream<0.||*downstream<0. ? 2 : 0;
}

int lumina_upwind_p0_segment(int nf,double dlognu,double length,double drift,
    const double *chi,const double *eta,const double *incident,double *output)
{
    if(nf<2||!isfinite(dlognu)||dlognu<=0.||!isfinite(length)||length<0.||
       !isfinite(drift)||drift<0.||(length==0.&&drift!=0.)||
       !chi||!eta||!incident||!output||incident==output)return 1;
    for(int k=0;k<nf;k++)
        if(!isfinite(chi[k])||!isfinite(eta[k])||eta[k]<0.||
           !isfinite(incident[k])||incident[k]<0.)return 1;
    if(length==0.){
        for(int k=0;k<nf;k++)output[k]=incident[k];
        return 0;
    }
    double b=drift/length,coupling=b/dlognu;
    if(!isfinite(b)||!isfinite(coupling))return 2;
    for(int k=nf-1;k>=0;k--){
        double alpha_b=k==nf-1 ? 0. : coupling;
        double tau=(chi[k]+3.*b+alpha_b)*length;
        double t,w0,w1;
        if(!isfinite(tau)||linear_weights(tau,length,&t,&w0,&w1))return 2;
        double first=eta[k],last=eta[k];
        if(k<nf-1){
            first+=alpha_b*incident[k+1];
            last+=alpha_b*output[k+1];
        }
        output[k]=t*incident[k]+w0*first+w1*last;
        if(!isfinite(output[k])||output[k]<0.)return 2;
    }
    return 0;
}
