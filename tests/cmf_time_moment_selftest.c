#include "cmf_time_moment.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define C_LIGHT 2.99792458e10
static int failures;
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#x);failures++;}}while(0)

typedef struct {
    CMFTimeMomentInput p;
    double *r,*nu,*ct,*cs,*eta,*f,*jo,*ho,*inner,*h,*fb,*jn,*hn;
} Fixture;

static double seed(double nu) {double x=log(nu/1e15)/0.3;return exp(-0.5*x*x);}

static void init(Fixture *a,int ns,int nb)
{
    memset(a,0,sizeof(*a));
    size_t n=(size_t)ns*nb, nh=(size_t)(ns+1)*nb;
    a->r=calloc(ns+1,sizeof(double));a->nu=calloc(nb+1,sizeof(double));
    a->ct=calloc(n,sizeof(double));a->cs=calloc(n,sizeof(double));
    a->eta=calloc(n,sizeof(double));a->f=calloc(n,sizeof(double));
    a->jo=calloc(n,sizeof(double));a->ho=calloc(nh,sizeof(double));
    a->jn=calloc(n,sizeof(double));a->hn=calloc(nh,sizeof(double));
    a->inner=calloc(nb,sizeof(double));a->h=calloc(nb,sizeof(double));a->fb=calloc(nb,sizeof(double));
    if(!a->r||!a->nu||!a->ct||!a->cs||!a->eta||!a->f||!a->jo||!a->ho||!a->jn||!a->hn||
       !a->inner||!a->h||!a->fb)abort();
    for(int s=0;s<=ns;s++)a->r[s]=1e13*s/ns;
    for(int b=0;b<=nb;b++)a->nu[b]=1e12*exp(log(1e5)*b/nb);
    for(int b=0;b<nb;b++)a->fb[b]=1.0/3;
    for(int s=0;s<ns;s++)for(int b=0;b<nb;b++){
        size_t i=(size_t)s*nb+b;a->f[i]=1.0/3;a->jo[i]=seed(sqrt(a->nu[b]*a->nu[b+1]));
    }
    a->p=(CMFTimeMomentInput){ns,nb,1e5,1.1e5,a->r,a->nu,a->ct,a->cs,a->eta,a->f,
        a->jo,a->ho,a->inner,a->h,a->fb};
}

static void destroy(Fixture *a)
{
    free(a->r);free(a->nu);free(a->ct);free(a->cs);free(a->eta);free(a->f);
    free(a->jo);free(a->ho);free(a->jn);free(a->hn);free(a->inner);free(a->h);free(a->fb);
}

/* Independent sum of the bin-integrated zeroth equation, using face fluxes
 * and shell volumes. Common 16*pi^2/c normalization cancels. */
static double balance(Fixture *a)
{
    int ns=a->p.n_shells,nb=a->p.n_bins;
    long double time=0, expansion=0, red=0, source=0, face=0;
    double dilution=pow(a->p.time_old/a->p.time_new,3);
    for(int s=0;s<ns;s++){
        long double vol=(powl(a->r[s+1],3)-powl(a->r[s],3))/3;
        red+=vol*a->nu[0]*a->jn[s*nb]/(C_LIGHT*a->p.time_new);
        for(int b=0;b<nb;b++){
            size_t i=(size_t)s*nb+b;long double w=vol*(a->nu[b+1]-a->nu[b]);
            time+=w*(a->jn[i]-dilution*a->jo[i])/(C_LIGHT*(a->p.time_new-a->p.time_old));
            expansion+=w*a->jn[i]/(C_LIGHT*a->p.time_new);
            source+=w*(a->eta[i]-(a->ct[i]-a->cs[i])*a->jn[i]);
        }
    }
    for(int b=0;b<nb;b++)face+=(a->nu[b+1]-a->nu[b])*
        ((long double)a->r[ns]*a->r[ns]*a->hn[ns*nb+b]-(long double)a->r[0]*a->r[0]*a->hn[b]);
    double relative=fabsl(time+expansion+red+face-source)/
        (fabsl(time)+fabsl(expansion)+fabsl(red)+fabsl(face)+fabsl(source));
    CHECK(isfinite(relative)&&relative<2e-11);
    return relative;
}

static double evolve(int nb,int steps,double *shape_error)
{
    Fixture a;init(&a,3,nb);double integral0=0;
    for(int b=0;b<nb;b++)integral0+=a.jo[b]*(a.nu[b+1]-a.nu[b]);
    for(int k=0;k<steps;k++){
        a.p.time_old=1e5*(1.0+(double)k/steps);
        a.p.time_new=1e5*(1.0+(double)(k+1)/steps);
        for(int s=0;s<=3;s++)a.r[s]=1e13*s/3*(a.p.time_new/1e5);
        CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_OK);
        balance(&a);
        memcpy(a.jo,a.jn,3*(size_t)nb*sizeof(double));
        memcpy(a.ho,a.hn,4*(size_t)nb*sizeof(double));
    }
    double integral=0,err=0,norm=0;
    for(int b=0;b<nb;b++){
        double dnu=a.nu[b+1]-a.nu[b], exact=0.125*seed(2*sqrt(a.nu[b+1]*a.nu[b]));
        integral+=a.jn[b]*dnu;err+=fabs(a.jn[b]-exact)*dnu;norm+=exact*dnu;
    }
    double E_ratio=8*integral/integral0;
    /* BE energy recurrence telescopes for equal steps and negligible edge loss. */
    CHECK(fabs(E_ratio-(double)(steps+1)/(2*steps+1))<2e-11);
    *shape_error=err/norm;destroy(&a);return fabs(E_ratio/0.5-1);
}

static void test_open_and_memory(void)
{
    Fixture a;init(&a,8,64);const int nb=64,probe=38; /* Near the injected spectral peak. */
    for(int s=0;s<8;s++)for(int b=0;b<nb;b++)a.ct[s*nb+b]=a.cs[s*nb+b]=1e-12;
    for(int b=0;b<nb;b++)a.h[b]=0.5;
    for(int s=1;s<=8;s++)for(int b=0;b<nb;b++)a.ho[s*nb+b]=0.01*s/8*a.jo[b];
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_OK);
    double residual=balance(&a),with_memory=a.jn[probe];
    CHECK(a.hn[8*nb+probe]>0);
    memset(a.ho,0,9*nb*sizeof(double));
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_OK);
    balance(&a);CHECK(fabs(with_memory-a.jn[probe])>1e-15);
    printf("OPEN_SCATTER balance=%.17g memory_delta=%.17g\n",residual,with_memory-a.jn[probe]);
    for(int s=0;s<8;s++)for(int b=0;b<nb;b++){
        a.ct[s*nb+b]=2e-12;a.eta[s*nb+b]=1e-12*a.jo[b];
    }
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_OK);balance(&a);
    /* Failure after some high-frequency bins were already solved must not publish. */
    double *savedJ=malloc(8*nb*sizeof(double)),*savedH=malloc(9*nb*sizeof(double));
    memcpy(savedJ,a.jn,8*nb*sizeof(double));memcpy(savedH,a.hn,9*nb*sizeof(double));
    a.ho[nb]=1e30;
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_NEGATIVE);
    CHECK(memcmp(savedJ,a.jn,8*nb*sizeof(double))==0&&memcmp(savedH,a.hn,9*nb*sizeof(double))==0);
    a.ho[nb]=0;a.p.time_old=a.p.time_new;
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_INVALID);
    a.p.time_old=1e5;a.f[0]=0;
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_INVALID);
    a.f[0]=1.0/3;a.ct[0]=0;
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_INVALID);
    CHECK(memcmp(savedJ,a.jn,8*nb*sizeof(double))==0&&memcmp(savedH,a.hn,9*nb*sizeof(double))==0);
    a.ct[0]=2e-12;
    CHECK(cmf_time_moment_step(&a.p,a.jo,a.ho)==CMF_TIME_OK);
    CHECK(memcmp(savedJ,a.jo,8*nb*sizeof(double))==0&&memcmp(savedH,a.ho,9*nb*sizeof(double))==0);
    free(savedJ);free(savedH);destroy(&a);
}

static double volume_monomial(double lo,double hi,int power)
{
    return 3.0/(power+3)*(pow(hi,power+3)-pow(lo,power+3))/(hi*hi*hi-lo*lo*lo);
}

/* Manufactured spherical solution, not a second implementation of the
 * discrete stencil: J=1+.15x^2, H=.05x, f=1/3+.2x^2, x=r/R.
 * The center is regular: H=0 and dJ/dr=df/dr=0 there.
 * Sources/history are obtained from continuous spatial derivatives. */
static double manufactured_spatial(int ns)
{
    Fixture a;init(&a,ns,8);const int nb=8;const double R=1e13, dt=10, opacity=2e-13;
    a.p.time_new=a.p.time_old+dt;
    double dilution=pow(a.p.time_old/a.p.time_new,3),error=0,norm=0;
    double *target=malloc(ns*sizeof(double));
    for(int s=0;s<ns;s++){
        double xl=a.r[s]/R,xr=a.r[s+1]/R;
        double x2=volume_monomial(xl,xr,2),x4=volume_monomial(xl,xr,4);
        target[s]=1+.15*x2;
        double K=1.0/3+(.15/3+.2)*x2+.03*x4;
        for(int b=0;b<nb;b++){
            size_t i=(size_t)s*nb+b;
            double drift=a.nu[b+1]/(a.nu[b+1]-a.nu[b])/(C_LIGHT*a.p.time_new);
            a.jo[i]=target[s]/dilution;a.ct[i]=opacity;a.f[i]=K/target[s];
            a.eta[i]=.15/R+opacity*target[s]+(b==nb-1 ? drift*target[s] : 0);
        }
    }
    for(int s=0;s<=ns;s++){
        double x=a.r[s]/R,H=.05*x;
        double spatial=((2*.15/3+5*.2)*x+7*.2*.15*x*x*x)/R;
        for(int b=0;b<nb;b++){
            double drift=a.nu[b+1]/(a.nu[b+1]-a.nu[b])/(C_LIGHT*a.p.time_new);
            a.ho[s*nb+b]=(H+C_LIGHT*dt*(opacity*H+spatial+(b==nb-1 ? drift*H : 0)))/dilution;
        }
    }
    for(int b=0;b<nb;b++){a.h[b]=.05/1.15;a.fb[b]=1.0/3+.2;}
    CHECK(cmf_time_moment_step(&a.p,a.jn,a.hn)==CMF_TIME_OK);
    balance(&a);
    for(int s=0;s<ns;s++){
        double w=pow(a.r[s+1]/R,3)-pow(a.r[s]/R,3);
        double delta=a.jn[s*nb]-target[s];error+=w*delta*delta;norm+=w*target[s]*target[s];
    }
    double relative=sqrt(error/norm);
    printf("SPHERICAL_MMS shells=%d rms=%.17g outer_H_error=%.17g\n",
           ns,relative,fabs(a.hn[ns*nb]-.05));
    free(target);destroy(&a);return relative;
}

int main(void)
{
    double shape,e10=evolve(128,10,&shape),e20=evolve(128,20,&shape),e40=evolve(128,40,&shape);
    CHECK(e20<0.55*e10&&e40<0.55*e20);
    double coarse,fine; evolve(128,512,&coarse);evolve(512,512,&fine);
    CHECK(fine<0.6*coarse);
    printf("HOMOLOGOUS time_errors=%.9g,%.9g,%.9g shape_coarse=%.9g shape_fine=%.9g\n",
           e10,e20,e40,coarse,fine);
    test_open_and_memory();
    double s20=manufactured_spatial(20),s40=manufactured_spatial(40),s80=manufactured_spatial(80);
    CHECK(s40<.55*s20&&s80<.55*s40);
    printf("CMF_TIME_MOMENT_SELFTEST %s failures=%d\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
