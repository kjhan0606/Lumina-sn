/* 팽창 특성선의 물리 방정식 시험. 실제 경계의 균질 매질 해를
 * 생산 기하학 함수를 호출하지 않고 현재 광선 격자에 독립적으로 적분한다. */
#include "cmf_exact_sliding.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#ifdef CMF_TEST_CUDA
#include "cmf_exact_multigpu.h"
#endif

static double uniform_J(int shell, double chi, double a, int legacy) {
    const double rm[3]={3.5e14,4.5e14,5.5e14};
    double impact[35];
    for(int k=0;k<16;++k){impact[k]=3e14*k/16.;impact[16+k]=3e14+3e14*k/16.;}
    for(int s=0;s<3;++s)impact[32+s]=rm[s];
    for(int k=1;k<35;++k){double p=impact[k];int q=k;
        while(q>0 && impact[q-1]>p){impact[q]=impact[q-1];--q;}impact[q]=p;}
    double kappa=chi+(legacy?4:3)*a;
    double source=legacy?1:chi/kappa;
    double sum=0,r=rm[shell],limb=sqrt(1-9e28/(r*r));
    for(int ray=34;ray>=0;--ray) {
        double p=impact[ray];
        if(p>r)continue;
        double zout=sqrt(36e28-p*p);
        double zs=sqrt(r*r-p*p);
        int core=p<3e14;
        double zcore=core?sqrt(9e28-p*p):0;
        double length_in=zout-zs;
        double length_out=core?zs-zcore:zout+zs;
        double mu=zs/r,high=core?1:limb,low=core?limb:0;
        if(ray>0 && (impact[ray-1]<3e14)==core)
            high=0.5*(mu+sqrt(1-impact[ray-1]*impact[ray-1]/(r*r)));
        if(ray+1<35 && impact[ray+1]<=r && (impact[ray+1]<3e14)==core)
            low=0.5*(mu+sqrt(1-impact[ray+1]*impact[ray+1]/(r*r)));
        sum+=(high-low)*0.5*source*(-expm1(-kappa*length_in)-expm1(-kappa*length_out));
    }
    return sum;
}

int main(int argc,char **argv) {
    enum {NS=3,NB=128};
    int require_physical=argc==2 && !strcmp(argv[1],"require-physical");
    const double t=1e6,a=1/(t*2.99792458e10),dlog=1e-3;
    const double ri[NS]={3e14,4e14,5e14},ro[NS]={4e14,5e14,6e14};
    double nu[NB],chi[NS*NB],es[NS*NB],sf[NS*NB],j[NS*NB];
    for(int b=0;b<NB;++b)nu[b]=1e15*exp((b+0.5)*dlog);
    double largest_physical_error=0,largest_legacy_error=0;
#ifdef CMF_TEST_CUDA
    double largest_gpu_error=0;
    CMFMultiGPUReport gr;
    const CMFMultiGPUEpochSchedule schedule={64,2,1};
#endif
    const double opacities[3]={1e-18,1e-16,1e-13};
    for(int trial=0;trial<3;++trial) {
        for(int i=0;i<NS*NB;++i){chi[i]=opacities[trial];es[i]=0;sf[i]=1;j[i]=0;}
        CMFExactReport report;
        if(cmf_exact_characteristic_solve(NS,NB,dlog,nu,ri,ro,t,10000,0,
                chi,es,sf,j,8,1e-13,CMF_EXACT_MODE_POSITIVE_SLIDING,&report)!=CMF_EXACT_OK)return 2;
        for(int s=0;s<NS;++s) {
            double physical=uniform_J(s,opacities[trial],a,0);
            double legacy=uniform_J(s,opacities[trial],a,1);
            double err=fabs(j[s*NB+NB/2]/physical-1);
            double olderr=fabs(j[s*NB+NB/2]/legacy-1);
            if(err>largest_physical_error)largest_physical_error=err;
            if(olderr>largest_legacy_error)largest_legacy_error=olderr;
            printf("chi=%.17g shell=%d actual=%.17g physical=%.17g ratio=%.17g legacy=%.17g\n",
                    opacities[trial],s,j[s*NB+NB/2],physical,j[s*NB+NB/2]/physical,legacy);
        }
#ifdef CMF_TEST_CUDA
        for(int mode=0;mode<3;++mode) {
            double gj[NS*NB]={0};
            CMFMultiGPUStatus status;
            if(mode==0)status=cmf_exact_multigpu_direct_solve(
                NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,gj,1,8,1e-13,&gr);
            else if(mode==1)status=cmf_exact_multigpu_positive_solve(
                NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,gj,1,8,1e-13,&gr);
            else status=cmf_exact_multigpu_positive_solve_epoch(
                NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,gj,&schedule,1,8,1e-13,&gr);
            if(status!=CMF_MGPU_OK){fprintf(stderr,"gpu mode=%d status=%d\n",mode,status);return 3;}
            for(int s=0;s<NS;++s) {
                double err=fabs(gj[s*NB+NB/2]/uniform_J(s,opacities[trial],a,0)-1);
                if(err>largest_gpu_error)largest_gpu_error=err;
            }
        }
#endif
    }
    printf("physical_max_relative_error=%.17g legacy_max_relative_error=%.17g\n",
            largest_physical_error,largest_legacy_error);
    if(require_physical && largest_physical_error>1e-10)return 1;

    /* 입력 J를 고정한 1회 수송으로 산란항의 정규화도 독립적으로 확인. */
    double lo[NS*NB],near[NS*NB],hi[NS*NB];
    double max_scattering_error=0;
    for(int i=0;i<NS*NB;++i){chi[i]=1e-16;es[i]=0.25*chi[i];sf[i]=1;j[i]=0.7;}
    if(cmf_exact_characteristic_apply_positive_bounds(NS,NB,dlog,nu,ri,ro,t,
            10000,0,chi,es,sf,j,lo,near,hi)!=CMF_EXACT_OK)return 4;
    for(int s=0;s<NS;++s) {
        double expected=1.175*uniform_J(s,1e-16,a,0);
        double err=fabs(near[s*NB+NB/2]/expected-1);
        if(err>max_scattering_error)max_scattering_error=err;
    }
    for(int i=0;i<NS*NB;++i)if(!(lo[i]<=near[i] && near[i]<=hi[i]))return 5;
#ifdef CMF_TEST_CUDA
    double gl[NS*NB],gn[NS*NB],gu[NS*NB];
    for(int mode=0;mode<2;++mode) {
        CMFMultiGPUStatus status=mode==0
            ? cmf_exact_multigpu_apply_positive_bounds(NS,NB,dlog,nu,ri,ro,t,
                10000,0,chi,es,sf,j,gl,gn,gu,1,&gr)
            : cmf_exact_multigpu_apply_positive_bounds_epoch(NS,NB,dlog,nu,ri,ro,t,
                10000,0,chi,es,sf,j,gl,gn,gu,&schedule,1,&gr);
        if(status!=CMF_MGPU_OK)return 6;
        for(int i=0;i<NS*NB;++i)
            if(!(gl[i]<=gn[i] && gn[i]<=gu[i]) || fabs(gn[i]/near[i]-1)>1e-10)return 7;
    }
    /* 실제 생산 경로의 persistent 상한·산란 K*u도 통과해야 한다. */
    for(int i=0;i<NS*NB;++i)j[i]=0;
    CMFExactReport cr;
    if(cmf_exact_characteristic_solve(NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,
            j,120,1e-14,CMF_EXACT_MODE_DIRECT_REFERENCE,&cr)!=CMF_EXACT_OK)return 8;
    double max_envelope_ratio=0;
    for(int mode=0;mode<2;++mode) {
        double gj[NS*NB]={0},error[NS*NB];
        CMFMultiGPUStatus status=mode==0
            ? cmf_exact_multigpu_positive_solve_envelope(NS,NB,dlog,nu,ri,ro,t,
                10000,0,chi,es,sf,gj,error,5,1,120,1e-13,&gr)
            : cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(
                NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,gj,error,5,
                CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS,&schedule,1,120,1e-13,&gr);
        if(status!=CMF_MGPU_OK || !gr.componentwise_error_envelope_verified)return 9;
        for(int i=0;i<NS*NB;++i) {
            double diff=fabs(gj[i]-j[i]);
            if(!(diff<=error[i]))return 10;
            if(error[i]>0 && diff/error[i]>max_envelope_ratio)max_envelope_ratio=diff/error[i];
        }
    }
    printf("gpu_analytic_max_relative_error=%.17g gpu_envelope_max_ratio=%.17g\n",
            largest_gpu_error,max_envelope_ratio);
    if(largest_gpu_error>1e-10)return 11;
#endif
    /* 진공의 유한 S는 eta=chi*S=0: 복사를 생성해서는 안 된다. */
    for(int i=0;i<NS*NB;++i){chi[i]=es[i]=j[i]=0;sf[i]=1;}
    CMFExactReport vr;
    if(cmf_exact_characteristic_solve(NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,j,
            8,1e-13,CMF_EXACT_MODE_POSITIVE_SLIDING,&vr)!=CMF_EXACT_OK)return 12;
    double vacuum_max=0;
    for(int i=0;i<NS*NB;++i)if(j[i]>vacuum_max)vacuum_max=j[i];
#ifdef CMF_TEST_CUDA
    if(cmf_exact_multigpu_positive_solve(NS,NB,dlog,nu,ri,ro,t,10000,0,chi,es,sf,
            j,1,8,1e-13,&gr)!=CMF_MGPU_OK)return 13;
    for(int i=0;i<NS*NB;++i)if(j[i]>vacuum_max)vacuum_max=j[i];
#endif
    printf("scattering_max_relative_error=%.17g vacuum_max=%.17g\n",max_scattering_error,vacuum_max);
    if(require_physical && (max_scattering_error>1e-10 || vacuum_max!=0))return 14;
    return 0;
}
