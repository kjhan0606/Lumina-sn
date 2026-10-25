#ifndef CMF_FULLSR_ENTRY_HPP
#define CMF_FULLSR_ENTRY_HPP
#include "cmf_linear_affine.hpp"
#ifdef __CUDACC__
#define LUMINA_ENTRY __host__ __device__ inline
#else
#define LUMINA_ENTRY inline
#endif
namespace lumina_fullsr_entry {
using lumina_ordered::Positive;
using lumina_ordered::product;
using lumina_ordered::sum;
using lumina_linear_affine::scale;
LUMINA_ENTRY int order(Positive a,Positive b){
    if(!a.mantissa)return b.mantissa?-1:0;
    if(!b.mantissa)return 1;
    if(a.exponent!=b.exponent)return a.exponent>b.exponent?1:-1;
    return a.mantissa==b.mantissa?0:(a.mantissa>b.mantissa?1:-1);
}
LUMINA_ENTRY bool subtract(Positive a,Positive b,Positive &out){
    return order(a,b)>=0&&lumina_linear_affine::difference(a,b,out);
}
LUMINA_ENTRY bool relative_u(const CMFLinearPositive *values,int i,int base,double h,Positive &out){
    Positive a,factor;
    return lumina_linear_affine::decode(values[i],a)&&Positive::from_log(-3.*(i-base)*h,factor)&&product(a,factor,out);
}
/* 기존 CPU PCHIP의 h/3 배수와 양의 Bernstein 제어점을 그대로 공유한다. */
LUMINA_ENTRY bool harmonic(Positive a,Positive b,Positive &out){
    if(!a.mantissa||!b.mantissa){out=Positive::zero();return true;}
    if(order(a,b)>0){Positive t=a;a=b;b=t;}
    int64_t difference=a.exponent-b.exponent;
    double ratio=difference < -1074?0.: ::ldexp(a.mantissa/b.mantissa,(int)difference);
    return scale(a,(2./3.)/(1.+ratio),out);
}
LUMINA_ENTRY bool interpolate(const CMFLinearPositive *values,int nf,int k,double f,double h,int reconstruction,Positive &out){
    Positive a,b;
    if(!relative_u(values,k,k,h,a))return false;
    if(!f){out=a;return true;}
    if(k+1>=nf||!relative_u(values,k+1,k,h,b))return false;
    Positive controls[4]={a,a,b,b};int direction=order(b,a);
    if(reconstruction&&direction){
        Positive middle,left,right,previous,next,alpha,beta;
        if(!lumina_linear_affine::difference(a,b,middle))return false;
        if(k==0){if(!scale(middle,1./3.,alpha))return false;}
        else{
            if(!relative_u(values,k-1,k,h,previous)||!lumina_linear_affine::difference(a,previous,left))return false;
            alpha=Positive::zero();
            if(order(a,previous)==direction&&!harmonic(left,middle,alpha))return false;
        }
        if(k+2==nf){if(!scale(middle,1./3.,beta))return false;}
        else{
            if(!relative_u(values,k+2,k,h,next)||!lumina_linear_affine::difference(b,next,right))return false;
            beta=Positive::zero();
            if(order(next,b)==direction&&!harmonic(middle,right,beta))return false;
        }
        if(direction>0){if(!sum(a,alpha,controls[1])||!subtract(b,beta,controls[2]))return false;}
        else if(!subtract(a,alpha,controls[1])||!sum(b,beta,controls[2]))return false;
    }
    Positive result=Positive::zero(),part,total,factor;
    if(!reconstruction){
        if(!scale(a,1.-f,part)||!scale(b,f,total)||!sum(part,total,result))return false;
    }else{
        double g=1.-f,w[4]={g*g*g,3.*g*g*f,3.*g*f*f,f*f*f};
        for(int i=0;i<4;i++){
            if(!scale(controls[i],w[i],part)||!sum(result,part,total))return false;
            result=total;
        }
    }
    return Positive::from_log(3.*f*h,factor)&&product(result,factor,out);
}
/* comoving 입사 I다. Q의 exp(-3delta) 보정을 여기서 이중 적용하지 않는다. */
LUMINA_ENTRY bool incoming(const CMFLinearPositive *values,int nf,int column,double shift,double h,Positive &out){
    if(!values||nf<2||column<0||column>=nf||!lumina_ordered::finite(shift)||shift<0||
       !lumina_ordered::finite(h)||h<=0)return false;
    if(shift>=nf-1)return lumina_linear_affine::decode(values[nf-1],out);
    int whole=(int)::floor(shift),k=column+whole;double fraction=shift-whole;
    if(k>=nf-1)return lumina_linear_affine::decode(values[nf-1],out);
    return interpolate(values,nf,k,fraction,h,1,out);
}
}
#undef LUMINA_ENTRY
#endif
