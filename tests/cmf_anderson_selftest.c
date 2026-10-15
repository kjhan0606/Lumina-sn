#include "cmf_anderson.h"
#include "cmf_exact_sliding.h"
#include <assert.h>
#include <stdio.h>
#ifdef CMF_ANDERSON_GPU_TEST
#include "cmf_exact_multigpu.h"
#endif

static void helper_tests(void)
{
    double x=0,g,pg=0,pr=0;CMFAnderson1 a={0};
    a.previous_g=&pg;a.previous_r=&pr;
    for(int k=0;k<3;k++) {
        g=1+0.99*x;
        double residual=x?fabs(g-x)/fabs(x):INFINITY;
        if(k==2)assert(fabs(g-100)<1e-10 && residual<1e-12);
        else cmf_anderson1_next(&a,1,&x,&g,residual);
    }
    assert(a.proposals==1);
    /* 인위적인 나쁜 다음 residual이면 보존한 일반 다음값으로 복귀한다. */
    double saved=pg;g=x+100;
    cmf_anderson1_next(&a,1,&x,&g,100);
    assert(x==saved && !a.have_previous && a.rejected==1);
    /* 음수 외삽 후보를 0으로 자르지 않고 정상 G(x)로 되돌린다. */
    x=1;g=2;pg=1;pr=0.9;
    a.have_previous=1;a.proposed=0;a.rejected=0;
    cmf_anderson1_next(&a,1,&x,&g,1);
    assert(x==2 && !a.proposed && a.rejected==1);
    /* 영공간은 floor 없이 그대로0이다. */
    x=g=pg=pr=0;a.have_previous=1;a.proposed=0;
    cmf_anderson1_next(&a,1,&x,&g,0);assert(x==0 && !a.proposed);
}

int main(void)
{
    helper_tests();
    enum {NS=3,NB=96,NC=NS*NB};
    double ri[NS]={3e14,4e14,5e14},ro[NS]={4e14,5e14,6e14};
    double nu[NB],chi[NC],es[NC],fixed[NC],plain[NC],fast[NC],bound[NC];
    for(int b=0;b<NB;b++)nu[b]=1e15*exp((b+.5)*1e-3);
    for(int s=0;s<NS;s++)for(int b=0;b<NB;b++) {
        int i=s*NB+b;chi[i]=(1+.2*s)*1e-11;es[i]=.99*chi[i];
        fixed[i]=(1+.1*cos(.1*b))*1e-9;plain[i]=fast[i]=.8e-7;
    }
    CMFExactReport p,f;unsetenv("LUMINA_CMF_ANDERSON1");
    assert(cmf_exact_characteristic_solve(NS,NB,1e-3,nu,ri,ro,1e6,10000,1,
        chi,es,fixed,plain,4096,1e-13,CMF_EXACT_MODE_POSITIVE_SLIDING,&p)==CMF_EXACT_OK);
    setenv("LUMINA_CMF_ANDERSON1","1",1);
    assert(cmf_exact_characteristic_solve_with_envelope(NS,NB,1e-3,nu,ri,ro,1e6,10000,1,
        chi,es,fixed,fast,bound,3,4096,1e-11,CMF_EXACT_MODE_POSITIVE_SLIDING,&f)==CMF_EXACT_OK);
    double difference=0;
    for(int i=0;i<NC;i++) {
        double e=fabs(fast[i]-plain[i]);
        assert(e<=bound[i]);difference=fmax(difference,e/fabs(plain[i]));
    }
    assert(difference<1e-8 && f.iterations_used<p.iterations_used);
    printf("CMF_ANDERSON_CPU PASS plain=%d accelerated=%d relative_difference=%.17g envelope_verified=%d\n",
        p.iterations_used,f.iterations_used,difference,f.componentwise_error_envelope_verified);
#ifdef CMF_ANDERSON_GPU_TEST
    double gpu[NC];for(int i=0;i<NC;i++)gpu[i]=.8e-7;
    CMFMultiGPUReport report;
    assert(cmf_exact_multigpu_positive_solve(NS,NB,1e-3,nu,ri,ro,1e6,10000,1,
        chi,es,fixed,gpu,1,4096,1e-11,&report)==CMF_MGPU_OK);
    double gpu_difference=0;
    for(int i=0;i<NC;i++)gpu_difference=fmax(gpu_difference,fabs(gpu[i]-plain[i])/fabs(plain[i]));
    assert(gpu_difference<1e-8 && report.iterations_used<p.iterations_used);
    printf("CMF_ANDERSON_GPU PASS iterations=%d relative_difference=%.17g\n",report.iterations_used,gpu_difference);
#endif
    return 0;
}
