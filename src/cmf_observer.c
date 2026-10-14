#include "cmf_observer_ray.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int cmf_observer_validate(const CMFObserverInput *in) {
    if(!in || in->ns<1 || in->nf<1 || in->nobs<1 ||
       in->ncore<1 || in->ncore>1024 || in->nouter<1 || in->nouter>1024 ||
       in->substeps<1 || in->substeps>1024 || !in->ri || !in->ro ||
       !in->chi || !in->es || !in->source || !in->J || !in->nu_obs ||
       !isfinite(in->nu_min) || in->nu_min<=0 ||
       !isfinite(in->dlognu) || in->dlognu<=0 ||
       !isfinite(in->t_exp) || in->t_exp<0 ||
       !isfinite(in->T_inner) || in->T_inner<0 ||
       !isfinite(in->inner_scale) || in->inner_scale<0) return -1;
    for(int s=0;s<in->ns;++s)
        if(!isfinite(in->ri[s]) || !isfinite(in->ro[s]) || in->ri[s]<=0 ||
           in->ro[s]<=in->ri[s] || (s && in->ri[s]!=in->ro[s-1]))return -1;
    const double beta=in->t_exp==0?0:in->ro[in->ns-1]/(OBS_C*in->t_exp);
    if(!isfinite(beta) || beta>=1) return -1;
    const double margin=atanh(beta), span=in->nf*in->dlognu;
    for(int i=0;i<in->nobs;++i) {
        if(!isfinite(in->nu_obs[i]) || in->nu_obs[i]<=0) return -1;
        const double x=log(in->nu_obs[i]/in->nu_min);
        if(!isfinite(x) || x-margin<0 || !(x+margin<span)) return -1;
    }
    if((size_t)in->ns>SIZE_MAX/sizeof(double)/(size_t)in->nf ||
       (size_t)in->nobs>SIZE_MAX/sizeof(double)/(size_t)(in->ncore+in->nouter))return -1;
    const size_t n=(size_t)in->ns*in->nf;
    for(size_t k=0;k<n;++k)
        if(!isfinite(in->chi[k]) ||
           !isfinite(in->es[k]) || in->es[k]<0 ||
           !isfinite(in->source[k]) ||
           (in->chi[k]<0 ? in->source[k]>0 : in->source[k]<0) ||
           !isfinite(in->J[k]) || in->J[k]<0)return -1;
    return 0;
}

static int gauss(int n, double hi, double radius, double *p, double *w) {
    const double pi=acos(-1.0);
    for(int i=0;i<(n+1)/2;++i) {
        double x=cos(pi*(i+0.75)/(n+0.5)), dp=0;
        int converged=0;
        for(int k=0;k<64;++k) {
            double pn=1, prev=0;
            for(int j=1;j<=n;++j) {
                const double old=prev; prev=pn;
                pn=((2*j-1)*x*prev-(j-1)*old)/j;
            }
            dp=n*(x*pn-prev)/(x*x-1);
            const double dx=pn/dp; x-=dx;
            if(fabs(dx)<3e-15){converged=1;break;}
        }
        if(!converged)return -1;
        const double weight=hi/((1-x*x)*dp*dp);
        const double mu0=hi*(1-x)*0.5, mu1=hi*(1+x)*0.5;
        p[i]=radius*sqrt((1-mu0)*(1+mu0));
        w[i]=radius*radius*mu0*weight;
        p[n-1-i]=radius*sqrt((1-mu1)*(1+mu1));
        w[n-1-i]=radius*radius*mu1*weight;
    }
    return 0;
}

int cmf_observer_quadrature(const CMFObserverInput *in, double *p, double *w) {
    double ratio=in->ri[0]/in->ro[in->ns-1];
    return gauss(in->ncore,1,in->ri[0],p,w) ||
        gauss(in->nouter,sqrt((1-ratio)*(1+ratio)),in->ro[in->ns-1],
              p+in->ncore,w+in->ncore) ? -1 : 0;
}

int cmf_observer_reduce(const CMFObserverInput *in, const double *intensity,
                        const double *w, double *Lnu) {
    const int nr=in->ncore+in->nouter;
    double *result=(double*)malloc((size_t)in->nobs*sizeof(double));
    if(!result)return -1;
    const double pi=acos(-1.0);
    for(int f=0;f<in->nobs;++f) {
        long double sum=0;
        for(int r=0;r<nr;++r) {
            double v=intensity[(size_t)f*nr+r];
            if(!isfinite(v) || v<0){free(result);return -1;}
            sum+=(long double)w[r]*v;
        }
        result[f]=(double)(8*pi*pi*sum);
        if(!isfinite(result[f])){free(result);return -1;}
    }
    memcpy(Lnu,result,(size_t)in->nobs*sizeof(double));
    free(result);return 0;
}

int cmf_observer_cpu(const CMFObserverInput *in, double *Lnu) {
    if(!Lnu || cmf_observer_validate(in))return -1;
    const int nr=in->ncore+in->nouter;
    const size_t n=(size_t)in->nobs*nr;
    double *p=(double*)malloc((size_t)nr*sizeof(double));
    double *w=(double*)malloc((size_t)nr*sizeof(double));
    double *v=(double*)malloc(n*sizeof(double));
    int rc=-1;
    if(p && w && v && !cmf_observer_quadrature(in,p,w)) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,1)
        #endif
        for(int f=0;f<in->nobs;++f)
            for(int r=0;r<nr;++r)
                v[(size_t)f*nr+r]=cmf_observer_ray(in,p[r],in->nu_obs[f]);
        rc=cmf_observer_reduce(in,v,w,Lnu);
    }
    free(p);free(w);free(v);return rc;
}
