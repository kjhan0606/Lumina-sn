#include "cmfgen_adiabatic.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define KB 1.380649e-16
static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#x);++failures;}}while(0)
static int near(double a,double b){return fabs(a-b)<=2e-12*fmax(fabs(a),fabs(b));}

static double expansion(int steps)
{
    double r0=1e-11,n0=1e10,e0=2e10,T0=1e4,u0=1e-11;
    CmfgenMaterialEnergyState old={1,1e5,&r0,&T0,&n0,&e0,&u0};
    double r,n,e,T,u=u0;CmfgenMaterialEnergyState now={1,0,&r,&T,&n,&e,&u};
    for(int step=1;step<=steps;++step){
        now.epoch_s=1e5*(1+(double)step/steps);
        double q=pow(old.epoch_s/now.epoch_s,3);
        r=r0*q;n=n0*q;e=e0*q;T=T0/(1+2*log(now.epoch_s/old.epoch_s));
        CmfgenMaterialTimeWork work;
        CHECK(cmfgen_material_time_work(&now,&old,&work)==CMFGEN_ADIABATIC_OK);
        CHECK(fabs(work.signed_total)/(fabs(work.thermal_time)+work.expansion)<3e-12);
        r0=r;n0=n;e0=e;T0=T;old.epoch_s=now.epoch_s;
    }
    double error=fabs(T/2500-1);
    printf("MATERIAL_ADIABATIC steps=%d T_final=%.17g relative_error=%.17g\n",steps,T,error);
    return error;
}

int main(void)
{
    double error20=expansion(20),error40=expansion(40),error80=expansion(80);
    CHECK(error40<.55*error20&&error80<.55*error40);
    double rold[2]={1e-11,2e-11},Told[2]={1e4,2e4},nold[2]={1e10,2e10};
    double eold[2]={1e10,2e10},uold[2]={1e-11,2e-11};
    double r[2],T[2],n[2],e[2],u[2];
    CmfgenMaterialEnergyState old={2,1e5,rold,Told,nold,eold,uold};
    CmfgenMaterialEnergyState now={2,1.01e5,r,T,n,e,u};
    const double q=pow(old.epoch_s/now.epoch_s,3),dt=now.epoch_s-old.epoch_s,chi=2e-11;
    CmfgenMaterialTimeWork work[2];
    for(int s=0;s<2;++s){
        r[s]=q*rold[s];n[s]=q*nold[s];e[s]=q*eold[s]+n[s];T[s]=Told[s];u[s]=uold[s]+chi;
    }
    CHECK(cmfgen_material_time_work(&now,&old,work)==CMFGEN_ADIABATIC_OK);
    for(int s=0;s<2;++s){
        CHECK(near(work[s].thermal_time,1.5*n[s]*KB*T[s]/dt));
        CHECK(near(work[s].atomic_time,n[s]*chi/dt));
        CHECK(near(work[s].expansion,3*(n[s]+e[s])*KB*T[s]*log(now.epoch_s/old.epoch_s)/dt));
        CHECK(near(work[s].signed_total,1.5*n[s]*KB*T[s]/dt+n[s]*chi/dt+
            3*(n[s]+e[s])*KB*T[s]*log(now.epoch_s/old.epoch_s)/dt));
        CHECK(work[s].cooling==work[s].signed_total&&work[s].heating==0);
    }
    printf("MATERIAL_IONIZATION thermal=%.17g atomic=%.17g expansion=%.17g total=%.17g\n",
        work[0].thermal_time,work[0].atomic_time,work[0].expansion,work[0].signed_total);
    for(int s=0;s<2;++s){e[s]=0;u[s]=0;}
    CHECK(cmfgen_material_time_work(&now,&old,work)==CMFGEN_ADIABATIC_OK);
    for(int s=0;s<2;++s){
        CHECK(work[s].signed_total<0&&work[s].heating==-work[s].signed_total&&work[s].cooling==0);
    }
    printf("MATERIAL_RECOMBINATION signed_total=%.17g heating=%.17g\n",work[0].signed_total,work[0].heating);
    CmfgenMaterialTimeWork saved[2];memcpy(saved,work,sizeof(saved));
    r[1]*=1.1;
    CHECK(cmfgen_material_time_work(&now,&old,work)==CMFGEN_ADIABATIC_NON_HOMOLOGOUS);
    CHECK(memcmp(saved,work,sizeof(saved))==0);r[1]/=1.1;
    u[1]=NAN;
    CHECK(cmfgen_material_time_work(&now,&old,work)!=CMFGEN_ADIABATIC_OK);
    CHECK(memcmp(saved,work,sizeof(saved))==0);u[1]=0;
    now.epoch_s=old.epoch_s;
    CHECK(cmfgen_material_time_work(&now,&old,work)==CMFGEN_ADIABATIC_INVALID_ARGUMENT);
    CHECK(memcmp(saved,work,sizeof(saved))==0);
    printf("CMF_MATERIAL_TIME_SELFTEST %s failures=%d\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
