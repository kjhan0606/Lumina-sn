/* 실제 안/밖 경계와 셸 중심에서의 J를 독립적인 구형 해와 비교한다.
 * static-limit 극한으로 주파수 이산화와 기하학 오차를 분리한다. */
#include "cmf_exact_sliding.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static double planck(double nu) {
    return 2*6.62607015e-27*nu*nu*nu/(2.99792458e10*2.99792458e10)
        /expm1(6.62607015e-27*nu/(1.380649e-16*10000));
}

/* 각도 적분은 생산 광선·셸 인덱스와 무관한 균등 mu 중점 적분.
 * 중심 광선의 불연속을 두 구간으로 분리한다. */
static double homogeneous_J(double r,double ri,double ro,double chi) {
    const int n=100000;
    double limb=sqrt(1-ri*ri/(r*r)),sum=0;
    for(int side=0;side<2;++side) {
        double low=side?limb:0,high=side?1:limb;
        double width=(high-low)/n;
        for(int q=0;q<n;++q) {
            double mu=low+(q+0.5)*width;
            double p2=r*r*(1-mu*mu),zs=r*mu;
            double zout=sqrt(ro*ro-p2);
            double outward=side?zs-sqrt(ri*ri-p2):zs+zout;
            double inward=zout-zs;
            sum+=0.5*(-expm1(-chi*inward)-expm1(-chi*outward))*width;
        }
    }
    return sum;
}

int main(int argc,char **argv) {
    enum {NS=3,NB=64};
    int require_physical=argc==2 && !strcmp(argv[1],"require-physical");
    const double ri[NS]={3e14,4e14,5e14},ro[NS]={4e14,5e14,6e14};
    double nu[NB],chi[NS*NB],es[NS*NB],sf[NS*NB],j[NS*NB];
    const double dlog=1e-3,t=1e24;
    for(int b=0;b<NB;++b)nu[b]=1e15*exp((b+0.5)*dlog);
    double vacuum_error=0,uniform_error=0;
    for(int trial=0;trial<2;++trial) {
        for(int i=0;i<NS*NB;++i){chi[i]=trial?1e-15:0;es[i]=0;sf[i]=trial?1:0;j[i]=0;}
        CMFExactReport report;
        if(cmf_exact_characteristic_solve(NS,NB,dlog,nu,ri,ro,t,10000,trial?0:1,
            chi,es,sf,j,8,1e-13,CMF_EXACT_MODE_POSITIVE_SLIDING,&report)!=CMF_EXACT_OK)return 2;
        for(int s=0;s<NS;++s) {
            double r=0.5*(ri[s]+ro[s]);
            double expected=trial?homogeneous_J(r,ri[0],ro[NS-1],1e-15)
                :0.5*(1-sqrt(1-ri[0]*ri[0]/(r*r)))*planck(nu[NB/2]);
            double ratio=j[s*NB+NB/2]/expected,error=fabs(ratio-1);
            if(trial && error>uniform_error)uniform_error=error;
            if(!trial && error>vacuum_error)vacuum_error=error;
            printf("case=%s shell=%d actual=%.17g physical=%.17g ratio=%.17g\n",
                trial?"uniform-emission":"vacuum-dilution",s,j[s*NB+NB/2],expected,ratio);
        }
    }
    printf("vacuum_max_relative_error=%.17g uniform_max_relative_error=%.17g\n",
            vacuum_error,uniform_error);
    if(require_physical && (vacuum_error>1e-10 || uniform_error>5e-3))return 1;
    return 0;
}
