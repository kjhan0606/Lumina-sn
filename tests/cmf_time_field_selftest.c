#include "cmf_time_field.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"TIME_FIELD_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
enum {NS=3,NB=32,NC=NS*NB,NF=(NS+1)*NB};
typedef struct {
    double ri[NS],ro[NS],nu[NB],chi[NC],scat[NC],eta[NC];
    double dlog,epoch;
    int fail;
} Context;
static int formal(void *arg,const double *J,CMFFormalMoments *m){
    Context *c=arg;if(c->fail)return -1;
    return cmf_exact_formal_moments_emissivity(NS,NB,c->dlog,c->nu,c->ri,c->ro,
        c->epoch,10000,1,c->chi,c->scat,c->eta,J,m)!=CMF_EXACT_OK;
}
static double planck(double nu){return 2*6.62607015e-27*nu*nu*nu/
    (2.99792458e10*2.99792458e10)/expm1(6.62607015e-27*nu/(1.380649e-16*10000));}
int main(void){
    Context c={0};c.dlog=log(300)/NB;c.epoch=1001000;
    double edges[NB+1],radius[NS+1],oldj[NC],oldh[NF]={0},inner[NB]={0};
    double j[NC],h[NF],delta[NC],keepj[NC],keeph[NF],initial[NC],oldcopy[NC];
    for(int b=0;b<=NB;b++)edges[b]=1e13*exp(b*c.dlog);
    for(int b=0;b<NB;b++)c.nu[b]=sqrt(edges[b]*edges[b+1]);
    for(int s=0;s<=NS;s++)radius[s]=(3e8+s*1e8)*c.epoch;
    for(int s=0;s<NS;s++){
        c.ri[s]=radius[s];c.ro[s]=radius[s+1];
        for(int b=0;b<NB;b++){
            int i=s*NB+b;c.chi[i]=3e-14;c.scat[i]=2e-14;
            c.eta[i]=1e-14*planck(c.nu[b]);oldj[i]=.5*planck(c.nu[b]);initial[i]=oldj[i];
        }
    }
    memcpy(oldcopy,oldj,sizeof oldj);
    CMFTimeMomentInput p={0};p.n_shells=NS;p.n_bins=NB;p.time_old=1e6;p.time_new=c.epoch;
    p.radius_edges=radius;p.frequency_edges=edges;p.chi_total=c.chi;p.chi_scattering=c.scat;
    p.eta_fixed=c.eta;p.old_J=oldj;p.old_H=oldh;p.inner_H=inner;
    CMFTimeFieldReport report;
    REQUIRE(cmf_time_field_solve(&p,initial,formal,&c,128,1e-9,j,h,delta,&report)==CMF_TIME_OK);
    REQUIRE(report.converged&&report.iterations>2&&report.equation_residual<1e-10);
    REQUIRE(memcmp(oldcopy,oldj,sizeof oldj)==0);
    for(int i=0;i<NF;i++)REQUIRE(oldh[i]==0);
    for(int i=0;i<NC;i++)REQUIRE(j[i]>0&&isfinite(j[i])&&delta[i]>=0);
    printf("TIME_FORMAL_SCATTER iterations=%d J_update=%.17g closure_update=%.17g equation_residual=%.17g\n",
        report.iterations,report.relative_update,report.closure_update,report.equation_residual);
    memcpy(keepj,j,sizeof j);memcpy(keeph,h,sizeof h);
    c.fail=1;
    REQUIRE(cmf_time_field_solve(&p,initial,formal,&c,128,1e-9,j,h,delta,&report)!=CMF_TIME_OK);
    REQUIRE(memcmp(j,keepj,sizeof j)==0&&memcmp(h,keeph,sizeof h)==0&&!report.converged);
    c.fail=0;
    /* 실제 과거를 바꾸면 같은 현재 물질에서도 해가 달라진다. */
    for(int i=0;i<NC;i++)oldj[i]*=2;
    REQUIRE(cmf_time_field_solve(&p,initial,formal,&c,128,1e-9,j,h,delta,&report)==CMF_TIME_OK);
    double history_effect=0;for(int i=0;i<NC;i++)history_effect=fmax(history_effect,fabs(j[i]/keepj[i]-1));
    REQUIRE(history_effect>.1);
    /* 음의 실제 소광과 양의 독립 방출을 보존하며 허용 dt로 시간식을 푼다. */
    for(int i=0;i<NC;i++){c.chi[i]=-1e-16;c.scat[i]=0;}
    REQUIRE(cmf_time_field_solve(&p,initial,formal,&c,128,1e-9,j,h,delta,&report)==CMF_TIME_OK);
    REQUIRE(report.equation_residual<1e-10);
    for(int i=0;i<NC;i++)REQUIRE(c.chi[i]==-1e-16&&j[i]>0);
    printf("TIME_SIGNED_EXTINCTION equation_residual=%.17g history_effect=%.17g old_immutable=1 failure_preserves=1\n",
        report.equation_residual,history_effect);
    puts("CMF_TIME_FIELD_SELFTEST_PASS");return 0;
}
