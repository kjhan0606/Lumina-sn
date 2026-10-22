#ifndef CMF_MONOTONE_RECONSTRUCTION_HPP
#define CMF_MONOTONE_RECONSTRUCTION_HPP
#include "cmf_ordered_maps.hpp"
#ifdef __CUDACC__
#define LUMINA_MONO __host__ __device__ inline
#else
#define LUMINA_MONO inline
#endif
// 선택 준비 후보다. 기존7차/GPU 실행 경로에 자동 연결하지 않는다.
namespace lumina_monotone_reconstruction {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::sum;
using lumina_ordered::product;
LUMINA_MONO bool less(Positive a,Positive b){
    return b.mantissa!=0&&(a.mantissa==0||a.exponent<b.exponent||
        (a.exponent==b.exponent&&a.mantissa<b.mantissa));
}
LUMINA_MONO bool scale(Positive a,double weight,Positive &out){
    Positive w;return Positive::from_double(weight,w)&&product(a,w,out);
}
LUMINA_MONO bool difference(Positive a,Positive b,Positive &out){
    if(less(a,b))return false;
    if(b.mantissa==0){out=a;return true;}
    int64_t shift=b.exponent-a.exponent;
    double v=a.mantissa-(shift < -1074?0.: ::ldexp(b.mantissa,(int)shift));
    Positive result;if(!Positive::from_double(v,result))return false;
    if(result.mantissa==0){out=Positive::zero();return true;}
    result.exponent+=a.exponent;out=result;return true;
}
LUMINA_MONO bool mix(Positive a,Positive b,double t,Positive &out){
    if(!finite(t)||t<0||t>1)return false;
    if(t==0){out=a;return true;}if(t==1){out=b;return true;}
    // 작은값을기준으로더해같은극소값의곱underflow와큰값끼리상쇄를피한다.
    Positive delta,part;
    if(less(b,a))return difference(a,b,delta)&&scale(delta,1.-t,part)&&sum(b,part,out);
    return difference(b,a,delta)&&scale(delta,t,part)&&sum(a,part,out);
}
LUMINA_MONO bool harmonic(double a,double b,double &out){
    if(!finite(a)||!finite(b))return false;
    if(!((a>0&&b>0)||(a<0&&b<0))){out=0.;return true;}
    double lo=::fmin(::fabs(a),::fabs(b)),hi=::fmax(::fabs(a),::fabs(b));
    double value=lo/(.5+.5*(lo/hi));
    if(!finite(value))return false;
    out=a<0?-value:value;return true;
}
LUMINA_MONO bool control(Positive base,double signed_slope,Positive &out){
    Positive slope,part;
    if(!Positive::from_double(::fabs(signed_slope),slope)||!scale(slope,1./3.,part))return false;
    return signed_slope<0?difference(base,part,out):sum(base,part,out);
}
LUMINA_MONO bool read(const double *input,int at,double &out){
    double value=input[at];if(!finite(value)||value<0)return false;out=value;return true;
}
LUMINA_MONO bool one_scaled(int nf,double shift,const double *input,int i,Positive &out){
    if(nf<4||nf>2000000||i<0||i>=nf||!input||!finite(shift)||shift<0)return false;
    double a,b;
    if(shift>=nf-1){if(!read(input,nf-1,a))return false;return Positive::from_double(a,out);}
    int whole=(int)::floor(shift),left=i+whole;double t=shift-whole;
    if(left>=nf-1){if(!read(input,nf-1,a))return false;return Positive::from_double(a,out);}
    if(!read(input,left,a))return false;
    if(t==0)return Positive::from_double(a,out);
    if(!read(input,left+1,b))return false;
    if(a==b)return Positive::from_double(a,out);
    double delta=b-a,ml=delta,mr=delta,v;
    if(left>0&&(!read(input,left-1,v)||!harmonic(a-v,delta,ml)))return false;
    if(left+2<nf&&(!read(input,left+2,v)||!harmonic(delta,v-b,mr)))return false;
    Positive c[4],low,high;
    if(!Positive::from_double(a,c[0])||!Positive::from_double(b,c[3])||
       !control(c[0],ml,c[1])||!control(c[3],-mr,c[2]))return false;
    low=less(c[0],c[3])?c[0]:c[3];high=less(c[0],c[3])?c[3]:c[0];
    for(int degree=3;degree>0;degree--)for(int k=0;k<degree;k++)
        if(!mix(c[k],c[k+1],t,c[k]))return false;
    // 값clamp가아니다. 후보의수학적범위를벗어나는산술은실패로남긴다.
    if(less(c[0],low)||less(high,c[0]))return false;
    out=c[0];return true;
}
LUMINA_MONO bool one(int nf,double shift,const double *input,int i,double &out){
    Positive p;if(!one_scaled(nf,shift,input,i,p))return false;
    double value=p.value();
    // 표현못하는양수를물리적0으로바꾸지않는다. scaled API는그값을보존한다.
    if(!finite(value)||(p.mantissa!=0&&value==0))return false;
    out=value;return true;
}
}
#undef LUMINA_MONO
#endif
