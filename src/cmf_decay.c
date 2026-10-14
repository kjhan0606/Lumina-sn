#include "cmf_decay.h"
#include <math.h>
#include <string.h>

int cmf_decay_chain_step(double l1,double l2,double dt,const double old[3],
                         double out[3],double decays[2]){
    if(!old||!out||!decays||!(l1>0)||!(l2>0)||!isfinite(l1)||!isfinite(l2)||
       !(dt>=0)||!isfinite(dt))return -1;
    for(int i=0;i<3;i++)if(!(old[i]>=0)||!isfinite(old[i]))return -1;
    long double a=(long double)l1*dt,b=(long double)l2*dt;
    long double d=fabsl(a-b),p1=expl(-a),p2=expl(-b);
    long double transfer=a*expl(-fminl(a,b))*(d==0?1:-expm1l(-d)/d);
    long double both=0;
    if(fmaxl(a,b)<.5L){
        /* 1-exp(-a)-transfer의 작은 시간 간격 차감 손실을 피한다.
         * ab sum_k (-1)^k h_k(a,b)/(k+2)! 이며 h_k=sum_j a^j b^(k-j). */
        long double h=1,bpower=1,invfact=.5L,sum=.5L;
        for(int k=1;k<32;k++){
            bpower*=b;h=a*h+bpower;invfact/=k+2;
            sum+=(k%2?-1:1)*h*invfact;
        }
        both=a*b*sum;
    }else if(d>.01L*fmaxl(a,b)){
        both=(a*(-expm1l(-b))-b*(-expm1l(-a)))/(a-b);
    }else{
        both=-expm1l(-a)-transfer;
    }
    long double d1=(long double)old[0]*(-expm1l(-a));
    long double d2=(long double)old[1]*(-expm1l(-b))+old[0]*both;
    double next[3]={(double)(old[0]*p1),(double)(old[1]*p2+old[0]*transfer),
                    (double)(old[2]+d2)};
    double number[2]={(double)d1,(double)d2};
    for(int i=0;i<3;i++)if(!(next[i]>=0)||!isfinite(next[i]))return -1;
    for(int i=0;i<2;i++)if(!(number[i]>=0)||!isfinite(number[i]))return -1;
    memcpy(out,next,sizeof next);memcpy(decays,number,sizeof number);return 0;
}
