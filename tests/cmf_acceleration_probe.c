/* 실제 CPU 형식해를 작은 가속법 대조에 제공한다. 생산 소스는 변경하지 않는다. */
#include "cmf_exact_sliding.h"
#include "cmf_anderson.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
enum {NS=3,NB=96,NC=NS*NB};
static double pg[NC],pr[NC];
static CMFAnderson1 aa;
static void material(int kind,double *nu,double *chi,double *es,double *fixed){
    for(int b=0;b<NB;b++)nu[b]=1e15*exp((b+.5)*1e-3);
    for(int s=0;s<NS;s++)for(int b=0;b<NB;b++){
        int i=s*NB+b;
        chi[i]=(1+.2*s)*1e-11;es[i]=.99*chi[i];
        fixed[i]=(1+.1*cos(.1*b))*1e-9;
        if(kind==1)es[i]=.999*chi[i];
        if(kind==2)es[i]=(s==1?.999:.8)*chi[i];
        if(kind==3){fixed[i]=0;es[i]=0;}
        if(kind==4){chi[i]=(2+s)*1e-15;es[i]=.27*chi[i];}
        if(kind==5){chi[i]=-1e-17;es[i]=1e-17;fixed[i]=-1e-7;}
    }
}
int acceleration_apply(int kind,const double *x,double *g){
    double ri[NS]={3e14,4e14,5e14},ro[NS]={4e14,5e14,6e14};
    double nu[NB],chi[NC],es[NC],fixed[NC],lo[NC],hi[NC];
    if(kind<0||kind>5)return -1;
    material(kind,nu,chi,es,fixed);
    return cmf_exact_characteristic_apply_positive_bounds(NS,NB,1e-3,nu,ri,ro,
        1e6,10000,kind==3?0:1,chi,es,fixed,x,lo,g,hi);
}
int acceleration_solve(int kind,int depth,double tolerance,double *x,double *bound){
    double ri[NS]={3e14,4e14,5e14},ro[NS]={4e14,5e14,6e14};
    double nu[NB],chi[NC],es[NC],fixed[NC];CMFExactReport report;
    material(kind,nu,chi,es,fixed);
    for(int i=0;i<NC;i++)x[i]=kind==3?0:.8e-7;
    setenv("LUMINA_CMF_ANDERSON1","1",1);
    char value[16];snprintf(value,sizeof value,"%d",depth);
    setenv("LUMINA_CMF_ANDERSON_DEPTH",value,1);
    int rc=cmf_exact_characteristic_solve_with_envelope(NS,NB,1e-3,nu,ri,ro,
        1e6,10000,kind==3?0:1,chi,es,fixed,x,bound,5,8192,tolerance,
        CMF_EXACT_MODE_POSITIVE_SLIDING,&report);
    return rc?-(int)rc:report.iterations_used;
}
int acceleration_reference(int kind,double *x,double *bound){
    return acceleration_solve(kind,1,1e-13,x,bound);
}
void acceleration_old_reset(void){
    memset(&aa,0,sizeof aa);aa.previous_g=pg;aa.previous_r=pr;
}
void acceleration_old_next(double *x,const double *g,double residual){
    cmf_anderson1_next(&aa,NC,x,g,residual);
}
size_t acceleration_old_count(int rejected){return rejected?aa.rejected:aa.proposals;}
