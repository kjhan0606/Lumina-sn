#ifndef CMF_POSITIVE_CUBIC_HPP
#define CMF_POSITIVE_CUBIC_HPP
#include "cmf_ordered_maps.hpp"
#include <cfloat>
#include <cmath>
#ifdef __CUDACC__
#define LUMINA_CUBIC __host__ __device__ inline
#else
#define LUMINA_CUBIC inline
#endif
namespace lumina_cubic {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::product;
struct Integral {Positive value,error;int degree;};

/* η(t)는 비음수 quadratic Bernstein, phase(t)는 cubic Bernstein다.
 * 공간선형χ×주파수선형χ가 만드는 cubic exponent의 양의 원천 적분이다.
 * 큰 phase 범위는 상위 적분기가 나눠야 한다. 물리 τ를 잘라 통과시키지 않는다.
 * relative는 이 다항 급수의 예산이며 외부 기하/주파수 근사 오차와 구별한다. */
LUMINA_CUBIC bool integrate(double length,const double *source,const double *phase,double log_scale,
    double relative,Integral &out)
{
    if(!source||!phase||!finite(length)||length<0||!finite(log_scale)||!finite(relative)||relative<=0||relative>.01)return false;
    double maximum=0.,low=phase[0],high=phase[0];
    for(int k=0;k<3;k++){
        if(!finite(source[k])||source[k]<0)return false;
        maximum=::fmax(maximum,source[k]);
    }
    for(int k=0;k<4;k++){
        if(!finite(phase[k]))return false;
        low=::fmin(low,phase[k]);high=::fmax(high,phase[k]);
    }
    if(maximum==0||length==0){out={Positive::zero(),Positive::zero(),0};return true;}
    double range=high-low;
    if(!finite(range)||range>.5)return false;
    double e[3]={source[0]/maximum,source[1]/maximum,source[2]/maximum};
    double z[4]={phase[0]-low,phase[1]-low,phase[2]-low,phase[3]-low};
    double mass=(e[0]+e[1]+e[2])/3.;
    double coefficients[73]={1.},next[73],total=0.,tail=range,exp_range=::exp(range);
    for(int n=0;n<=24;n++){
        int m=3*n;double denominator=(m+1.)*(m+2.)*(m+3.),term=0.;
        for(int i=0;i<=m;i++){
            double w0=(m-i+1.)*(m-i+2.)/denominator;
            double w1=2.*(i+1.)*(m-i+1.)/denominator;
            double w2=(i+1.)*(i+2.)/denominator;
            term+=coefficients[i]*(w0*e[0]+w1*e[1]+w2*e[2]);
        }
        total+=term;
        double operations=64.*(n+1)*(n+1)+128.;
        double gamma=operations*DBL_EPSILON/(1.-operations*DBL_EPSILON);
        double truncation=2.*mass*exp_range*tail;
        double error=truncation+gamma*total;
        if(error<=relative*total){
            double shift=log_scale+low;
            double delta=DBL_EPSILON*(::fabs(log_scale)+::fabs(low));
            if(!finite(shift)||!finite(delta)||delta>.01)return false;
            // log scale의 덧셈/exp 오차도 반환한다. 외부 적분기가 이를 소비한다.
            error+=total*(2.*delta+8.*DBL_EPSILON);
            Positive a,b,c,scale,value,uncertainty;
            if(!Positive::from_double(length,a)||!Positive::from_double(maximum,b)||!product(a,b,c)||
               !Positive::from_log(shift,a)||!product(c,a,scale)||!Positive::from_double(total,b)||
               !product(scale,b,value)||!Positive::from_double(error,b)||!product(scale,b,uncertainty))return false;
            out={value,uncertainty,n};return true;
        }
        if(n==24)break;
        int old=m;denominator=(old+1.)*(old+2.)*(old+3.);
        for(int i=0;i<=old+3;i++)next[i]=0.;
        for(int i=0;i<=old;i++){
            double weights[4]={
                (old-i+1.)*(old-i+2.)*(old-i+3.)/denominator,
                3.*(i+1.)*(old-i+1.)*(old-i+2.)/denominator,
                3.*(i+1.)*(i+2.)*(old-i+1.)/denominator,
                (i+1.)*(i+2.)*(i+3.)/denominator};
            for(int j=0;j<4;j++)next[i+j]+=coefficients[i]*(z[j]/(n+1.))*weights[j];
        }
        for(int i=0;i<=old+3;i++)coefficients[i]=next[i];
        tail*=range/(n+2.);
    }
    return false;
}
}
#undef LUMINA_CUBIC
#endif
