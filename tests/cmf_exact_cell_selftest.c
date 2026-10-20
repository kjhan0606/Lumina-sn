#include "cmf_exact_sliding.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(c,label) do { if (!(c)) { \
    fprintf(stderr,"CMF_CELL_FAIL %s line=%d\n",label,__LINE__); ++failures; \
} } while (0)

/* 한 균일 구각의 직접 해석해. ray 간격만 생산 각도 적분과 같게 하고
 * segment/주파수 sweep이나 그 버퍼를 사용하지 않는다. */
static double analytic_shell(double ri,double ro,double chi,double source,
                             double core,double time)
{
    const double mid=.5*(ri+ro), effective=chi+3.0/(2.99792458e10*time);
    double previous_mu=0.0,previous_j=0.0,sum=0.0;
    for (int k=17;k>=0;--k) {
        double p=k==17 ? mid : (k==16 ? ri : ri*k/16.0);
        double z=sqrt((mid-p)*(mid+p));
        double zo=sqrt((ro-p)*(ro+p));
        double si=chi/effective*source;
        double inward=si*(-expm1(-effective*(zo-z)));
        double outward;
        if (p<ri) {
            double zi=sqrt((ri-p)*(ri+p));
            outward=core*exp(-effective*(z-zi))+
                si*(-expm1(-effective*(z-zi)));
        } else outward=si*(-expm1(-effective*(zo+z)));
        double j=.5*(inward+outward),mu=z/mid;
        if (k!=17) sum+=.5*(previous_j+j)*(mu-previous_mu);
        previous_mu=mu;previous_j=j;
    }
    return sum;
}

static void analytic_tests(void)
{
    enum { NB=96 };
    double nu[NB],ct[NB],ce[NB]={0},sf[NB],j[NB],error[NB];
    double inner[NB],ratio[NB]={0},gradient[NB]={0};
    double ri=2e14,ro=6e14;
    for (int b=0;b<NB;++b) { nu[b]=1e14*exp((b+.5)*.03);inner[b]=1.0; }
    CMFExactInnerBoundary boundary={CMF_EXACT_INNER_DIFFUSION,NAN,NAN,
        inner,ratio,gradient};
    for (int scenario=0;scenario<2;++scenario) {
        double time=scenario ? 2e5 : 1e30;
        for (int b=0;b<NB;++b) {
            ct[b]=scenario ? 0.0 : 2e-15;
            sf[b]=scenario ? 0.0 : 5.0;
            j[b]=0.0;
        }
        CMFExactReport report;
        CHECK(cmf_exact_cell_solve(1,NB,.03,nu,&ri,&ro,time,&boundary,
            ct,ce,sf,j,error,2,8,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,
            &report)==CMF_EXACT_OK,"analytic-status");
        CHECK(report.componentwise_error_envelope_verified,"analytic-envelope");
        double expected=analytic_shell(ri,ro,ct[0],sf[0],1.0,time);
        for (int b=0;b<NB;++b)
            CHECK(fabs(j[b]-expected)<=2e-13*expected,"actual-boundary-and-full-path");
        printf("CMF_CELL_ANALYTIC case=%s J=%.17g exact_quadrature=%.17g rel=%.6g\n",
            scenario ? "vacuum-I_nu-3a" : "absorption-full-cell",
            j[NB/2],expected,fabs(j[NB/2]/expected-1));
    }
}

static void diffusion_tests(void)
{
    enum { NB=256 };
    double nu[NB],dn[NB],chi[NB],gradient[NB],old[NB];
    const double radius=1.7251e14,luminosity=9.9476e40;
    const double dlog=log(1e5)/NB;
    for (int b=0;b<NB;++b) {
        nu[b]=1e12*exp((b+.5)*dlog);
        dn[b]=2*nu[b]*sinh(.5*dlog);
        chi[b]=1e-12*(1.0+.2*sin(.2*b));
        gradient[b]=-7.0;
    }
    CHECK(cmf_exact_diffusion_gradient(NB,nu,dn,23021.0,luminosity,radius,
        chi,gradient)==CMF_EXACT_OK,"diffusion-gradient");
    long double integral=0;
    for (int b=0;b<NB;++b) integral+=(long double)gradient[b]*dn[b]/3;
    long double pi=acosl(-1.0L);
    double recovered=(double)(16*pi*pi*radius*radius*integral);
    CHECK(fabs(recovered/luminosity-1)<5e-15,"prescribed-luminosity");
    memcpy(old,gradient,sizeof(old));chi[NB/2]=0;
    CHECK(cmf_exact_diffusion_gradient(NB,nu,dn,23021.0,luminosity,radius,
        chi,gradient)==CMF_EXACT_INVALID_INPUT,"no-opacity-floor");
    CHECK(!memcmp(old,gradient,sizeof(old)),"failed-gradient-preserved");
    printf("CMF_CELL_DIFFUSION luminosity=%.17g recovered=%.17g rel=%.6g\n",
        luminosity,recovered,fabs(recovered/luminosity-1));
}

static double raw_path(double incident,double opacity,double eta,double length)
{
    double tau=opacity*length;
    return incident*exp(-tau)+eta*length*(tau==0 ? 1 : -expm1(-tau)/tau);
}

static double analytic_raw_shell(double ri,double ro,double chi,double eta,
                                  double core,double time)
{
    double mid=.5*(ri+ro),effective=chi+3*(1/(2.99792458e10*time));
    double previous_mu=0,previous_j=0,sum=0;
    for(int k=17;k>=0;k--){
        double impact=k==17?mid:(k==16?ri:ri*k/16.0);
        double z=sqrt((mid-impact)*(mid+impact));
        double zo=sqrt((ro-impact)*(ro+impact));
        double inward=raw_path(0,effective,eta,zo-z),outward;
        if(impact<ri){double zi=sqrt((ri-impact)*(ri+impact));
            outward=raw_path(core,effective,eta,z-zi);
        }else outward=raw_path(0,effective,eta,zo+z);
        double j=.5*(inward+outward),mu=z/mid;
        if(k!=17)sum+=.5*(previous_j+j)*(mu-previous_mu);
        previous_mu=mu;previous_j=j;
    }
    return sum;
}

static void signed_raw_tests(void)
{
    enum{NB=96};
    double nu[NB],ct[NB],ce[NB]={0},eta[NB],j[NB],error[NB];
    double source[NB],ratio[NB]={0},gradient[NB]={0};
    double ri=2e14,ro=6e14,time=2e5;
    CMFExactInnerBoundary bc={CMF_EXACT_INNER_DIFFUSION,NAN,NAN,source,ratio,gradient};
    double opacities[]={2e-15,0,-3*(1/(2.99792458e10*time)),-8e-15};
    for(int b=0;b<NB;b++){nu[b]=1e14*exp((b+.5)*.03);source[b]=1e-7;eta[b]=1e-22;}
    for(size_t k=0;k<sizeof(opacities)/sizeof(opacities[0]);k++){
        for(int b=0;b<NB;b++){ct[b]=opacities[k];j[b]=0;}
        CMFExactReport report;
        CHECK(cmf_exact_cell_solve_raw_sources(1,NB,.03,nu,&ri,&ro,time,&bc,
            ct,ce,eta,j,error,2,8,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,&report)==CMF_EXACT_OK,
            "raw-signed-analytic-status");
        double expected=analytic_raw_shell(ri,ro,ct[0],eta[0],source[0],time);
        CHECK(report.componentwise_error_envelope_verified,"raw-signed-envelope");
        CHECK(report.max_scattering_ratio==0,"raw-no-scattering-K-norm");
        for(int b=0;b<NB;b++)CHECK(fabs(j[b]/expected-1)<2e-13,"raw-signed-analytic");
        printf("CMF_CELL_RAW chi=%.17g J=%.17g analytic=%.17g rel=%.6g K_norm=%.6g\n",
            ct[0],j[NB/2],expected,fabs(j[NB/2]/expected-1),report.max_scattering_ratio);
    }
    double before[NB];memcpy(before,j,sizeof(before));
    for(int b=0;b<NB;b++)ct[b]=-1e-8;
    CHECK(cmf_exact_cell_solve_raw_sources(1,NB,.03,nu,&ri,&ro,time,&bc,
        ct,ce,eta,j,error,2,8,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,NULL)==CMF_EXACT_NONFINITE,
        "gain-overflow-not-clamped");
    CHECK(!memcmp(before,j,sizeof(before)),"gain-overflow-no-publish");
}

static void scattering_tests(void)
{
    enum { NS=3,NB=96,N=NS*NB };
    double ri[NS]={2e14,3e14,4e14},ro[NS]={3e14,4e14,6e14};
    double nu[NB],ct[N],ce[N],sf[N],jp[N],jd[N],error[N];
    double bcsource[NB],bcratio[NB],gradient[NB];
    for (int b=0;b<NB;++b) {
        nu[b]=1e14*exp((b+.5)*1e-3);
        bcsource[b]=1e-7*(1+.1*cos(.3*b));bcratio[b]=.4;
        gradient[b]=1e-8*(1+.2*sin(.2*b));
    }
    for (int s=0;s<NS;++s) for (int b=0;b<NB;++b) {
        int i=s*NB+b;
        ct[i]=2e-15*(1+.2*sin(.17*b+.4*s));ce[i]=.25*ct[i];
        sf[i]=1e-7*(1+.2*cos(.1*b+.2*s));jp[i]=jd[i]=0.0;
    }
    CMFExactInnerBoundary boundary={CMF_EXACT_INNER_DIFFUSION,NAN,NAN,
        bcsource,bcratio,gradient};
    CMFExactReport rp,rd;
    CHECK(cmf_exact_cell_solve(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jp,error,4,100,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,
        &rp)==CMF_EXACT_OK,"scattering-positive");
    CHECK(cmf_exact_cell_solve(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jd,NULL,0,100,1e-14,CMF_EXACT_MODE_DIRECT_REFERENCE,
        &rd)==CMF_EXACT_OK,"scattering-direct");
    CHECK(rp.componentwise_error_envelope_verified,"scattering-envelope");
    CHECK(rp.max_scattering_ratio==.4,"boundary-scattering-in-bound");
    double maxrel=0,maxratio=0;
    for (int i=0;i<N;++i) {
        double difference=fabs(jp[i]-jd[i]);
        double rel=difference/jd[i];if (rel>maxrel) maxrel=rel;
        CHECK(difference<=error[i],"envelope-includes-boundary-scattering");
        if (error[i]>0 && difference/error[i]>maxratio) maxratio=difference/error[i];
    }
    CHECK(maxrel<2e-11,"direct-positive-agree");
    printf("CMF_CELL_SCATTER direct_positive_rel=%.6g envelope_ratio=%.6g iterations=%d\n",
        maxrel,maxratio,rp.iterations_used);
    /* 같은 주파수 이동·경계 산란에 부호 있는 소광과 raw η를 공급한다. */
    for(int i=0;i<N;i++){
        sf[i]*=ct[i];
        if(i%7==0)ct[i]=-4e-17;
        jp[i]=jd[i]=0;
    }
    CHECK(cmf_exact_cell_solve_raw_sources(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jp,error,4,100,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,&rp)==CMF_EXACT_OK,
        "raw-scattering-positive");
    CHECK(cmf_exact_cell_solve_raw_sources(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jd,NULL,0,100,1e-14,CMF_EXACT_MODE_DIRECT_REFERENCE,&rd)==CMF_EXACT_OK,
        "raw-scattering-direct");
    CHECK(rp.max_scattering_ratio>0&&rp.max_scattering_ratio<1,"raw-actual-K-norm");
    CHECK(rp.componentwise_error_envelope_verified,"raw-scattering-envelope");
    maxrel=0;
    for(int i=0;i<N;i++){
        double difference=fabs(jp[i]-jd[i]);
        if(difference/jd[i]>maxrel)maxrel=difference/jd[i];
        CHECK(difference<=error[i],"raw-gain-envelope-includes-direct");
    }
    printf("CMF_CELL_RAW_SCATTER direct_positive_rel=%.6g K_norm=%.6g envelope=VERIFIED\n",
        maxrel,rp.max_scattering_ratio);
    for(int s=0;s<NS;s++)for(int b=0;b<NB;b++){
        int i=s*NB+b;ct[i]=2e-15*(1+.2*sin(.17*b+.4*s));
        ce[i]=.25*ct[i];sf[i]=1e-7*(1+.2*cos(.1*b+.2*s));
    }
    double before[N];memcpy(before,jp,sizeof(before));
    bcratio[17]=NAN;
    CHECK(cmf_exact_cell_solve(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jp,NULL,0,100,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,
        &rp)==CMF_EXACT_INVALID_INPUT,"bad-boundary-rejected");
    CHECK(!memcmp(before,jp,sizeof(before)),"bad-boundary-no-publish");
    bcratio[17]=.4;ri[1]+=1e10;
    CHECK(cmf_exact_cell_solve(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jp,NULL,0,100,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,
        &rp)==CMF_EXACT_INVALID_INPUT,"gap-rejected");
    CHECK(!memcmp(before,jp,sizeof(before)),"gap-no-publish");
    ri[1]=3e14;nu[20]*=1.0001;
    CHECK(cmf_exact_cell_solve(NS,NB,1e-3,nu,ri,ro,1e6,&boundary,
        ct,ce,sf,jp,NULL,0,100,1e-12,CMF_EXACT_MODE_POSITIVE_SLIDING,
        &rp)==CMF_EXACT_INVALID_INPUT,"nonlog-grid-rejected");
    CHECK(!memcmp(before,jp,sizeof(before)),"bad-grid-no-publish");
}

int main(void)
{
    analytic_tests();diffusion_tests();signed_raw_tests();scattering_tests();
    if (failures) return 1;
    puts("CMF_CELL_SELFTEST PASS; discretization and physical-equivalence NOT_JUDGED");
    return 0;
}
