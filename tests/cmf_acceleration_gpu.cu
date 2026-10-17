/* 동일 이산 형식해의 CPU 참조와 단일 GPU 다중 이력 가속을 대조한다. */
#include "cmf_exact_multigpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

enum {NS=3,NB=96,NC=NS*NB};
static int failures;
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"ACCELERATION_GPU_FAIL line=%d\n",__LINE__);failures++;}}while(0)
static void material(int kind,double *nu,double *chi,double *es,double *fixed){
    for(int b=0;b<NB;b++)nu[b]=1e15*std::exp((b+.5)*1e-3);
    for(int s=0;s<NS;s++)for(int b=0;b<NB;b++){
        int i=s*NB+b;chi[i]=(1+.2*s)*1e-11;es[i]=.99*chi[i];
        fixed[i]=(1+.1*std::cos(.1*b))*1e-9;
        if(kind==1)es[i]=.999*chi[i];
        if(kind==2)es[i]=(s==1?.999:.8)*chi[i];
        if(kind==3){fixed[i]=0;es[i]=0;}
        if(kind==4){chi[i]=(2+s)*1e-15;es[i]=.27*chi[i];}
        if(kind==5){chi[i]=-1e-17;es[i]=1e-17;fixed[i]=-1e-7;}
    }
}
int main(){
    double ri[NS]={3e14,4e14,5e14},ro[NS]={4e14,5e14,6e14};
    double nu[NB],chi[NC],es[NC],fixed[NC],eta[NC];
    double ref[NC],ref_bound[NC],x[NC],bound[NC],old[NC],default_x[NC];
    const CMFMultiGPUEpochSchedule schedule={64,2,1};
    setenv("LUMINA_CMF_ANDERSON1","1",1);
    for(int kind=0;kind<6;kind++){
        material(kind,nu,chi,es,fixed);
        for(int i=0;i<NC;i++){ref[i]=kind==3?0:.8e-7;eta[i]=chi[i]*fixed[i];}
        setenv("LUMINA_CMF_ANDERSON_DEPTH","1",1);
        CMFExactReport cpu;
        CHECK(cmf_exact_characteristic_solve_with_envelope(NS,NB,1e-3,nu,ri,ro,
            1e6,10000,kind==3?0:1,chi,es,fixed,ref,ref_bound,5,8192,1e-13,
            CMF_EXACT_MODE_POSITIVE_SLIDING,&cpu)==CMF_EXACT_OK);
        for(int depth: {0,1,2,4}){
            if(depth==0)unsetenv("LUMINA_CMF_ANDERSON_DEPTH");
            else{char text[8];std::snprintf(text,sizeof text,"%d",depth);setenv("LUMINA_CMF_ANDERSON_DEPTH",text,1);}
            for(int i=0;i<NC;i++)x[i]=kind==3?0:.8e-7;
            CMFMultiGPUReport report;
            int rc=cmf_exact_multigpu_emissivity_solve_epoch(NS,NB,1e-3,nu,ri,ro,
                1e6,10000,kind==3?0:1,chi,es,eta,x,bound,5,&schedule,1,8192,1e-10,&report);
            CHECK(rc==CMF_MGPU_OK);
            if(rc!=CMF_MGPU_OK){std::fprintf(stderr,"kind=%d depth=%d status=%d\n",kind,depth,rc);continue;}
            CHECK(report.componentwise_error_envelope_verified==1);
            CHECK(report.final_max_relative_change<1e-10);
            double largest=0;
            for(int i=0;i<NC;i++){
                CHECK(std::isfinite(x[i])&&x[i]>=0&&std::isfinite(bound[i])&&bound[i]>=0);
                const double error=std::fabs(x[i]-ref[i]);
                CHECK(error<=bound[i]+ref_bound[i]);
                if(ref[i]>0)largest=std::fmax(largest,error/ref[i]);else CHECK(x[i]==0);
            }
            CHECK(largest<1e-7);
            if(depth==0)std::memcpy(default_x,x,sizeof x);
            if(depth==1)CHECK(std::memcmp(default_x,x,sizeof x)==0);
            std::printf("GPU_ACCELERATION kind=%d depth=%d iterations=%d relative=%.17g bounds_verified=%d\n",
                        kind,depth,report.iterations_used,largest,report.componentwise_error_envelope_verified);
        }
        std::memcpy(old,x,sizeof x);
        setenv("LUMINA_CMF_ANDERSON_DEPTH","3",1);
        CMFMultiGPUReport invalid;
        CHECK(cmf_exact_multigpu_emissivity_solve_epoch(NS,NB,1e-3,nu,ri,ro,
            1e6,10000,kind==3?0:1,chi,es,eta,x,bound,5,&schedule,1,8192,1e-10,&invalid)==CMF_MGPU_INVALID_INPUT);
        CHECK(std::memcmp(old,x,sizeof x)==0);
    }
    if(failures)return 1;
    std::puts("GPU_ACCELERATION_PASS cases=24 invalid_depth_no_publication=6 default_depth1_byte_equal=6");
    return 0;
}
