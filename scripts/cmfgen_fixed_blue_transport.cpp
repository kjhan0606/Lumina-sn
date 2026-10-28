// 저장된 고정-blue 식에 생산 CPU/GPU 공통 양의 적분 코어만 적용한다.
// 전주파수 특성선이나 CMFGEN 생산 solver의 대체 API가 아니다.
#include "cmf_fullsr_linear_map.hpp"
#include <cmath>

extern "C" int lumina_fixed_blue_leg(int n,const double *length,const double *chi,
    const double *eta,const double *b,const double *blue,double alpha,double incident,
    int pieces,double *output){
    using lumina_ordered::Positive;
    if(n<2||n>512||pieces<1||pieces>64||!length||!chi||!eta||!b||!blue||!output||
       !std::isfinite(alpha)||alpha<0||!std::isfinite(incident)||incident<0)return 1;
    for(int i=0;i<n;i++){
        if(!std::isfinite(chi[i])||!std::isfinite(eta[i])||eta[i]<0||
           !std::isfinite(b[i])||b[i]<0||!std::isfinite(blue[i])||blue[i]<0)return 1;
        if(i<n-1&&(!std::isfinite(length[i])||length[i]<=0))return 1;
    }
    Positive value;
    if(!Positive::from_double(incident,value))return 2;
    uint64_t primitive=0,composition=0;
    output[0]=incident;
    for(int i=0;i<n-1;i++){
        for(int k=0;k<pieces;k++){
            double q[2],source[2],dl=length[i]/pieces;
            for(int j=0;j<2;j++){
                double x=double(k+j)/pieces;
                double c=(1-x)*chi[i]+x*chi[i+1];
                double e=(1-x)*eta[i]+x*eta[i+1];
                double drift=(1-x)*b[i]+x*b[i+1];
                double previous=(1-x)*blue[i]+x*blue[i+1];
                q[j]=std::fma(alpha+3.,drift,c)*dl;
                source[j]=std::fma(alpha*drift,previous,e);
            }
            lumina_fullsr_map::Affine map;
            Positive response,next;
            if(!lumina_fullsr_map::source(q[0],q[1],dl,source[0],source[1],map,primitive,composition)||
               !lumina_ordered::product(map.t,value,response)||
               !lumina_ordered::sum(response,map.s,next))return 2;
            value=next;
        }
        // 이 진단은 유한 double로 저장된 기존 packet과만 비교한다.
        if(value.exponent>1024||value.exponent<-1074){
            if(value.mantissa!=0)return 3;
        }
        output[i+1]=std::ldexp(value.mantissa,int(value.exponent));
        if(!std::isfinite(output[i+1]))return 3;
    }
    return 0;
}

// 같은 s-선형 chi/eta/B/blue에서는 유효 방출항이 양의 이차식이다.
// 생산 공통 basis의 세 모멘트로 직접 적분한다. 기존 1..64분할 경로는 불변이다.
// 이 진단 API는 effective chi의 두 끝점이 비음수인 셀만 지원한다.
extern "C" int lumina_fixed_blue_quadratic_leg(int n,const double *length,const double *chi,
    const double *eta,const double *b,const double *blue,double alpha,double incident,
    int pieces,double *output){
    using lumina_ordered::Positive;
    if(n<2||n>512||pieces!=1||!length||!chi||!eta||!b||!blue||!output||
       !std::isfinite(alpha)||alpha<0||!std::isfinite(incident)||incident<0)return 1;
    for(int i=0;i<n;i++){
        if(!std::isfinite(chi[i])||!std::isfinite(eta[i])||eta[i]<0||
           !std::isfinite(b[i])||b[i]<0||!std::isfinite(blue[i])||blue[i]<0)return 1;
        if(i<n-1&&(!std::isfinite(length[i])||length[i]<=0))return 1;
    }
    Positive value;
    if(!Positive::from_double(incident,value))return 2;
    output[0]=incident;
    for(int i=0;i<n-1;i++){
        double q0=std::fma(alpha+3.,b[i],chi[i])*length[i];
        double q1=std::fma(alpha+3.,b[i+1],chi[i+1])*length[i];
        if(!std::isfinite(q0)||!std::isfinite(q1)||q0<0||q1<0)return 4;
        double tau=.5*q0+.5*q1,w[3];CMFHistoryStats stats={};
        if(::basis(q0,q1,std::exp(-tau),w,&stats))return 2;
        // basis는 u², u(1-u), (1-u)² 순서다. 중간 계수는 Bernstein의 2배다.
        double coefficient[3]={std::fma(alpha*b[i],blue[i],eta[i]),
            eta[i]+eta[i+1]+alpha*(b[i]*blue[i+1]+b[i+1]*blue[i]),
            std::fma(alpha*b[i+1],blue[i+1],eta[i+1])};
        Positive transmission,source=Positive::zero(),p,q,next;
        if(!Positive::from_log(-tau,transmission))return 2;
        for(int j=0;j<3;j++){
            if(!std::isfinite(w[j])||w[j]<0||!Positive::from_double(w[j],p)||
               !lumina_fullsr_map::scale(p,length[i],q)||
               !lumina_fullsr_map::scale(q,coefficient[j],p)||
               !lumina_ordered::sum(source,p,next))return 2;
            source=next;
        }
        if(!lumina_ordered::product(transmission,value,p)||!lumina_ordered::sum(p,source,next))return 2;
        value=next;
        if(value.exponent>1024||value.exponent<-1074){if(value.mantissa!=0)return 3;}
        output[i+1]=std::ldexp(value.mantissa,int(value.exponent));
        if(!std::isfinite(output[i+1]))return 3;
    }
    return 0;
}
