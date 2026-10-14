#include "cmf_exact_sliding.h"
#include "cmf_cell_geometry.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#x);failures++;}}while(0)
enum {NS=4,NB=32,N=NS*NB};
static double planck(double nu,double T){return 2*6.62607015e-27*nu*nu*nu/(2.99792458e10*2.99792458e10)/expm1(6.62607015e-27*nu/(1.380649e-16*T));}
static double error(double a,double b){return fabs(a-b)/fmax(fabs(a),fabs(b));}

int main(void)
{
    double ri[NS],ro[NS],nu[NB],ct[N]={0},ce[N]={0},sf[N]={0},input[N]={0};
    double j[N],h[N],k[N],oj[NB],oh[NB],ok[NB],lo[N],nearest[N],up[N];
    CMFFormalMoments out={j,h,k,oj,oh,ok};const double T=10000,dlog=log(3e15/1e13)/NB;
    for(int s=0;s<NS;s++){ri[s]=(s+1)*1e10;ro[s]=(s+2)*1e10;}
    for(int b=0;b<NB;b++)nu[b]=1e13*exp((b+.5)*dlog);
    CHECK(cmf_exact_formal_moments(NS,NB,dlog,nu,ri,ro,1e30,T,1,ct,ce,sf,input,&out)==CMF_EXACT_OK);
    double vacuum_error=0;
    for(int s=0;s<NS;s++)for(int b=0;b<NB;b++){
        double r=.5*(ri[s]+ro[s]),mu=sqrt(1-ri[0]*ri[0]/(r*r)),B=planck(nu[b],T);
        double exact[]={.5*B*(1-mu),.25*B*(1-mu*mu),B*(1-mu*mu*mu)/6};
        double got[]={j[s*NB+b],h[s*NB+b],k[s*NB+b]};
        for(int m=0;m<3;m++)vacuum_error=fmax(vacuum_error,error(got[m],exact[m]));
    }
    for(int b=0;b<NB;b++){
        double mu=sqrt(1-ri[0]*ri[0]/(ro[NS-1]*ro[NS-1])),B=planck(nu[b],T);
        CHECK(error(oj[b],.5*B*(1-mu))<2e-12);
        CHECK(error(oh[b],.25*B*(1-mu*mu))<2e-12);
        CHECK(error(ok[b],B*(1-mu*mu*mu)/6)<2e-12);
    }
    CHECK(vacuum_error<2e-12);
    CHECK(cmf_exact_characteristic_apply_positive_bounds(NS,NB,dlog,nu,ri,ro,1e30,T,1,
        ct,ce,sf,input,lo,nearest,up)==CMF_EXACT_OK);
    CHECK(memcmp(nearest,j,sizeof(j))==0);
    for(int i=0;i<N;i++){ct[i]=1e-5;sf[i]=2;}
    CHECK(cmf_exact_formal_moments(NS,NB,dlog,nu,ri,ro,1e30,T,1,ct,ce,sf,input,&out)==CMF_EXACT_OK);
    for(int i=0;i<N;i++){
        CHECK(error(j[i],2)<2e-12);CHECK(fabs(h[i])<2e-12);CHECK(error(k[i],2.0/3)<2e-12);
    }
    for(int b=0;b<NB;b++){
        CHECK(error(oj[b],1)<2e-12);CHECK(error(oh[b],.5)<2e-12);CHECK(error(ok[b],1.0/3)<2e-12);
    }
    /* Full sphere: zero-radius inner face introduces no artificial core light. */
    for(int s=0;s<NS;s++){ri[s]=s*1e10;ro[s]=(s+1)*1e10;}
    for(int i=0;i<N;i++){ct[i]=ce[i]=sf[i]=input[i]=0;}
    CHECK(cmf_exact_formal_moments(NS,NB,dlog,nu,ri,ro,1e30,T,1,ct,ce,sf,input,&out)==CMF_EXACT_OK);
    for(int i=0;i<N;i++)CHECK(j[i]==0&&h[i]==0&&k[i]==0);
    for(int b=0;b<NB;b++)CHECK(oj[b]==0&&oh[b]==0&&ok[b]==0);
    /* Invalid input must not publish any partial result. */
    j[0]=101;h[0]=102;k[0]=103;oj[0]=104;oh[0]=105;ok[0]=106;input[N-1]=NAN;
    CHECK(cmf_exact_formal_moments(NS,NB,dlog,nu,ri,ro,1e30,T,1,ct,ce,sf,input,&out)!=CMF_EXACT_OK);
    CHECK(j[0]==101&&h[0]==102&&k[0]==103&&oj[0]==104&&oh[0]==105&&ok[0]==106);
    printf("FORMAL_MOMENTS vacuum_max_relative=%.17g legacy_J_bitwise=checked thick_isotropy=checked\n",vacuum_error);
    printf("CMF_FORMAL_MOMENTS_SELFTEST %s failures=%d\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
