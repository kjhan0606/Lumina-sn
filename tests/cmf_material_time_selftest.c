#include "cmfgen_adiabatic.h"
#ifdef CMF_MATERIAL_ROOT_SELFTEST
#include "radeq_publication.h"
#include "a210_coupled_newton.h"
#endif
#include <math.h>
#include <stdio.h>
#include <string.h>

#define KB 1.380649e-16
static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#x);++failures;}}while(0)
static int near(double a,double b){return fabs(a-b)<=2e-12*fmax(fabs(a),fabs(b));}

#ifdef CMF_MATERIAL_ROOT_SELFTEST
/* 실제 물질 게시/온도 root를 링크하는 history runner에서만 추가 실행한다. */
typedef struct {
    CmfgenMaterialEnergyState old,now;
    double electron;
} SmallStep;
static int small_step_residual(const double *T,const double *ne,size_t n,
    double *fT,double *fe,void *out,void *ctx){
    SmallStep *s=ctx;
    if(n!=2)return 1;
    for(size_t i=0;i<n;i++){
        CmfgenMaterialTimeWork w;A210TermLedger *l=(A210TermLedger*)out+i;
        CmfgenMaterialEnergyState now=s->now;now.temperature_K=T+i;now.n_electron_cm3=ne+i;
        if(cmfgen_material_time_work(&now,&s->old,&w)!=CMFGEN_ADIABATIC_OK)return 1;
        memset(l,0,sizeof(*l));l->equation_kind=A210_RE_INTEGRAL;l->m_line=1;l->radiative_line_included=1;
        for(int k=0;k<A210_NHEAT;k++)l->heating_status[k]=A210_EXACT_ZERO;
        for(int k=0;k<A210_NCOOL;k++)l->cooling_status[k]=A210_EXACT_ZERO;
        if(a210_apply_material_time_work(l,&w)!=RADEQ_OK)return 1;
        RadeqStatus status=a210_line_owner_finalize(l);
        if(status!=RADEQ_OK&&status!=RADEQ_EXACT_ZERO_BALANCE)return 1;
        fT[i]=a210_temperature_residual(l);fe[i]=log(s->electron/ne[i]);
        if(!isfinite(fT[i])||!isfinite(fe[i]))return 1;
    }
    return 0;
}

static void small_step_root(double dt){
    double rho0=1.66e-16,n0=1e8,ne0=344755,T0=1e4,u0=0,t0=1e6;
    double q=pow(t0/(t0+dt),3),rho=rho0*q,n=n0*q,ne[2]={ne0*q,ne0*q},T[2]={T0,T0};
    SmallStep s={{1,t0,&rho0,&T0,&n0,&ne0,&u0},{1,t0+dt,&rho,T,&n,ne,&u0},ne[0]};
    A210CoupledNewtonConfig cfg;A210CoupledNewtonReport report;A210TermLedger ledger[2]={{0},{0}};
    a210_coupled_newton_default_config(&cfg,2);cfg.te_min=3500;cfg.te_max=140000;
    cfg.eval_out_bytes=sizeof ledger;
    A210CoupledNewtonStatus status=a210_coupled_newton_solve(&cfg,small_step_residual,&s,T,ne,ledger,&report);
    CHECK(status==A210_CN_CONVERGED);
    if(status!=A210_CN_CONVERGED){fprintf(stderr,"SMALL_STEP_ROOT status=%s\n",a210_coupled_newton_status_name(status));return;}
    double analytic=T0/(1+2*log1p(dt/t0));
    for(int i=0;i<2;i++)CHECK(fabs(T[i]/analytic-1)<3e-12&&a210_energy_balance(&ledger[i])<1e-12);
    /* 단열식의 순가열 지표가1이어도 저장항의 backward error는 작다.
     * 작은 dt에서 입력 T 그대로를 수렴해로 받으면 안 된다. */
    A210TermLedger unchanged[2];double fT[2],fe[2],initial_T[2]={T0,T0};
    CHECK(!small_step_residual(initial_T,ne,2,fT,fe,unchanged,&s)&&fabs(fT[0])>1e-10);
    ElectronTemperaturePublication pub={0};char hash[65];memset(hash,'a',64);hash[64]=0;
    double tout[2]={-1,-1},nout[2]={-1,-1};
    CHECK(!a210_publication_candidate_from_root(T,ne,ledger,2,1,1,hash,report.evaluations,&pub,tout,nout));
    CHECK(tout[0]==T[0]&&nout[0]==ne[0]&&tout[1]==T[1]&&nout[1]==ne[1]);
    a210_publication_free(&pub);
    printf("MATERIAL_SMALL_STEP dt=%.17g T=%.17g analytic=%.17g residual=%.17g backward_error=%.17g net_heat_balance=%.17g norm=%.17g\n",
        dt,T[0],analytic,ledger[0].residual,a210_energy_balance(&ledger[0]),ledger[0].e_balance,ledger[0].material_time_norm);
}
#endif

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
#ifdef CMF_MATERIAL_ROOT_SELFTEST
    small_step_root(100);small_step_root(10);small_step_root(.01);
#endif
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
