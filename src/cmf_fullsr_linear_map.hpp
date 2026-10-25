#ifndef CMF_FULLSR_LINEAR_MAP_HPP
#define CMF_FULLSR_LINEAR_MAP_HPP
#include "cmf_ordered_maps.hpp"
#include "cmf_positive_moments.h"
#ifdef __CUDACC__
#define LUMINA_FULLSR_MAP __device__ inline
#else
#define LUMINA_FULLSR_MAP inline
#endif
namespace lumina_fullsr_map {
using lumina_ordered::Positive;
using lumina_ordered::product;
using lumina_ordered::sum;
struct Affine {Positive t,s;};
LUMINA_FULLSR_MAP bool scale(Positive a,double b,Positive &out){
    Positive p;return Positive::from_double(b,p)&&product(a,p,out);
}
LUMINA_FULLSR_MAP bool compose(Affine a,Affine b,Affine &out){
    Positive t,x,y;
    if(!product(b.t,a.t,t)||!product(b.t,a.s,x)||!sum(x,b.s,y))return false;
    out={t,y};return true;
}
/* CPU와 GPU가 같은 선형 chi/eta 양의 모멘트 코어를 사용한다. */
LUMINA_FULLSR_MAP bool one(double q0,double q1,double length,double eb,double er,
                           Affine &out,uint64_t &primitives){
    double tau=.5*q0+.5*q1;Affine result;
    if(!Positive::from_log(-tau,result.t))return false;
    result.s=Positive::zero();primitives++;
    if(eb!=0||er!=0){
        bool reflected=tau<0;double a=reflected?-q1:q0,b=reflected?-q0:q1;
        double w[3];CMFHistoryStats work={};
        if(::basis(a,b,::exp(-(.5*a+.5*b)),w,&work))return false;
        for(int k=0;k<3;k++)if(!lumina_ordered::finite(w[k])||w[k]<=0)return false;
        double blue=reflected?w[1]+w[2]:w[0]+w[1],red=reflected?w[0]+w[1]:w[1]+w[2];
        Positive p,q,r;
        if(!Positive::from_double(blue,p)||!scale(p,length,q)||!scale(q,eb,p)||
           !Positive::from_double(red,q)||!scale(q,length,r)||!scale(r,er,q)||!sum(p,q,result.s))return false;
        if(reflected){if(!product(result.s,result.t,p))return false;result.s=p;}
    }
    out=result;return true;
}
/* 내부 gain 봉우리는 정확한 effective chi=0에서 두 번만 분할한다.
 * 재귀를 제거하되 기존 CPU 연산 순서·순 gain의 scaled T를 보존한다. */
LUMINA_FULLSR_MAP bool source(double q0,double q1,double length,double eb,double er,
                              Affine &out,uint64_t &primitives,uint64_t &compositions){
    if(!lumina_ordered::finite(q0)||!lumina_ordered::finite(q1))return false;
    if(q0>0&&q1<0){
        double denominator=q0-q1;if(!lumina_ordered::finite(denominator))return false;
        double f=q0/denominator;if(!(f>0&&f<1))return false;
        double middle=(1.-f)*eb+f*er;Affine a,b;
        if(!one(q0*f,0.,length*f,eb,middle,a,primitives)||
           !one(0.,q1*(1.-f),length*(1.-f),middle,er,b,primitives)||!compose(a,b,out))return false;
        compositions++;return true;
    }
    return one(q0,q1,length,eb,er,out,primitives);
}
}
#undef LUMINA_FULLSR_MAP
#endif
