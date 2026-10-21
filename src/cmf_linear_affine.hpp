#ifndef CMF_LINEAR_AFFINE_HPP
#define CMF_LINEAR_AFFINE_HPP
#include "cmf_linear_native.h"
#include "cmf_ordered_maps.hpp"
#ifdef __CUDACC__
#define LUMINA_AFFINE __host__ __device__ inline
#else
#define LUMINA_AFFINE inline
#endif
namespace lumina_linear_affine {
using lumina_ordered::Positive;
using lumina_ordered::product;
using lumina_ordered::sum;
using lumina_ordered::finite;
LUMINA_AFFINE bool scale(Positive a,double v,Positive &out){Positive b;return Positive::from_double(v,b)&&product(a,b,out);}
LUMINA_AFFINE bool decode(CMFLinearPositive a,Positive &out){
    if(!finite(a.mantissa)||a.mantissa<0||(a.mantissa!=0&&(a.mantissa<.5||a.mantissa>=1.))||
       a.exponent<=-(INT64_C(1)<<60)||a.exponent>=(INT64_C(1)<<60))return false;
    out={a.mantissa,a.exponent};return true;
}
LUMINA_AFFINE CMFLinearPositive encode(Positive a){return {a.mantissa,a.exponent};}
LUMINA_AFFINE bool difference(Positive a,Positive b,Positive &out){
    if(a.mantissa==0){out=b;return true;}if(b.mantissa==0){out=a;return true;}
    if(a.exponent<b.exponent||(a.exponent==b.exponent&&a.mantissa<b.mantissa)){Positive t=a;a=b;b=t;}
    int64_t d=b.exponent-a.exponent;double m=a.mantissa-(d < -1074?0.: ::ldexp(b.mantissa,(int)d));
    Positive p;if(!Positive::from_double(m,p))return false;
    if(p.mantissa==0){out=Positive::zero();return true;}
    out={p.mantissa,p.exponent+a.exponent};return true;
}
/* 비율만 마지막 double로 바꾼다. intensity floor 또는 다른ν norm은 없다. */
LUMINA_AFFINE double ratio(Positive a,Positive b){
    if(a.mantissa==0)return 0.;
    if(b.mantissa==0)return INFINITY;
    int64_t d=a.exponent-b.exponent;
    if(d>1024)return INFINITY;
    if(d < -1074)return 0.;
    return ::ldexp(a.mantissa/b.mantissa,(int)d);
}
struct Map {
    Positive t,source,error;double tau_error;
    LUMINA_AFFINE bool upper(Positive &out) const {
        return finite(tau_error)&&tau_error>=0&&tau_error<=.25&&scale(t,::exp(tau_error),out);
    }
    LUMINA_AFFINE bool apply(Positive input,Positive &value,Positive &uncertainty) const {
        Positive response,a,b,rounding;
        if(!finite(tau_error)||tau_error<0||tau_error>.25||!product(t,input,response)||!sum(response,source,value)||
           !scale(response,::expm1(tau_error),a)||!sum(error,a,b)||!scale(value,32.*DBL_EPSILON,rounding))return false;
        return sum(b,rounding,uncertainty);
    }
};
LUMINA_AFFINE bool decode(const CMFLinearNativeMap &raw,Map &out){
    Map m;if(!decode(raw.transmission,m.t)||!decode(raw.source,m.source)||!decode(raw.source_error,m.error)||
       !finite(raw.tau_error)||raw.tau_error<0||raw.tau_error>.25)return false;
    m.tau_error=raw.tau_error;out=m;return true;
}
LUMINA_AFFINE bool compose(const Map &left,const Map &right,Map &out){
    Map m;Positive a,b,c,d,upper;
    m.tau_error=left.tau_error+right.tau_error+8.*DBL_EPSILON;
    if(!finite(m.tau_error)||m.tau_error>.25||!product(left.t,right.t,m.t)||!product(right.t,left.source,a)||
       !sum(a,right.source,m.source)||!right.upper(upper)||!product(upper,left.error,b)||!sum(b,right.error,b)||
       !scale(a,::expm1(right.tau_error),c)||!sum(b,c,d)||!scale(m.source,32.*DBL_EPSILON,a)||!sum(d,a,m.error))return false;
    out=m;return true;
}
struct Local {Positive value,geometry,numerical,response;};
LUMINA_AFFINE bool leaf(const CMFLinearNativeMap *raw,int stride,int column,Positive incident,Local &out){
    if(!raw||stride<1||column<0||column>=stride)return false;
    Map coarse,left,right,fine;Positive cv,ce,fv,fe,a,b,delta;
    if(!decode(raw[column],coarse)||!decode(raw[stride+column],left)||!decode(raw[2*stride+column],right)||
       !compose(left,right,fine)||!coarse.apply(incident,cv,ce)||!fine.apply(incident,fv,fe)||
       !difference(cv,fv,delta)||!sum(delta,ce,a)||!sum(a,fe,b)||!sum(cv,fv,a)||
       !scale(a,16.*DBL_EPSILON,a)||!sum(b,a,delta)||!fine.upper(a))return false;
    out={fv,delta,fe,a};return true;
}
}
#undef LUMINA_AFFINE
#endif
