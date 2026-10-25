#include "cmf_outer_transport.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int workers,moments,approved_cpus,observed_threads;
    size_t block;
    double *data,*weights;
    int status[16];
} CMFOuterWeighted;

struct CMFOuterContext {
    CMFOuterProblem p;
    size_t n,m,e,samples,work_bytes;
    CMFOuterStep *steps;
    CMFOuterRay *rays;
    CMFOuterObservation *observations;
    double *data,*cell_eta,*log_chi,*log_eta,*previous_m,*previous_p,*moments;
    double *intensity,*alternate,*chi,*eta,*scratch,*mu_m,*mu_p,*turning,*ratio;
    int *seen_m,*seen_p;
    CMFOuterWeighted *weighted;
};

static int outer_add(size_t *value,size_t n)
{
    if(n>SIZE_MAX-*value)return -1;
    *value+=n;return 0;
}

static int outer_product(size_t a,size_t b,size_t *out)
{
    if(a&&b>SIZE_MAX/a)return -1;
    *out=a*b;return 0;
}

static int outer_span(int32_t begin,int32_t count,int32_t total)
{
    return begin>=0&&count>=0&&begin<=total&&count<=total-begin;
}

void cmf_outer_destroy(CMFOuterContext *c)
{
    if(!c)return;
    free(c->steps);free(c->rays);free(c->observations);free(c->data);
    free(c->seen_m);free(c->weighted);free(c);
}

static CMFExactStatus outer_fixed_opacity(CMFOuterContext *c)
{
    const CMFOuterProblem *p=&c->p;
    size_t nf=(size_t)p->frequencies;
    for(size_t b=0;b<nf;b++){
        double ca=p->node_chi[b],cb=p->node_chi[nf+b];
        double sa=p->node_es[b],sb=p->node_es[nf+b];
        double power=fmax(2.,log(sb/sa)/p->log_reference_radius_ratio);
        int special=ca<=sa||cb<=sb;
        double absorption=special ? 0. : fmax(2.,log((cb-sb)/(ca-sa))/p->log_reference_radius_ratio);
        if(!isfinite(power)||!isfinite(absorption))return CMF_EXACT_NONFINITE;
        for(int i=0;i<p->exterior_nodes;i++){
            double ratio=c->ratio[i];
            double chi=i==0 ? ca : special ? ca*pow(ratio,power) :
                (ca-sa)*pow(ratio,absorption)+sa*pow(ratio,power);
            if(!isfinite(chi))return CMF_EXACT_NONFINITE;
            c->log_chi[(size_t)i*nf+b]=chi==0 ? 0. : log(fabs(chi));
        }
    }
    return CMF_EXACT_OK;
}

static CMFExactStatus outer_storage_mode(const CMFOuterProblem *p,int workers,int moments,int cpus,
    size_t *stored_doubles,size_t *work_bytes)
{
    if(work_bytes)*work_bytes=0;
    if(!p||p->cells<1||p->cells>INT_MAX-2||p->frequencies<2||
       p->frequencies>INT_MAX-2||p->rays<1||p->steps<1||p->observations<1||
       p->exterior_nodes!=25||!(p->dlognu>0)||!isfinite(p->dlognu)||
       !(p->log_reference_radius_ratio>0)||!isfinite(p->log_reference_radius_ratio)||
       !p->step||workers<0||workers>16||
       (workers&&(moments!=1&&moments!=3))||(workers&&(cpus<1||cpus>16)))
        return CMF_EXACT_INVALID_INPUT;
    size_t nf=(size_t)p->frequencies,samples=(size_t)p->cells+2,n,m,e;
    if(outer_product((size_t)p->cells,nf,&n)||outer_product(samples,nf,&m)||
       outer_product((size_t)p->exterior_nodes,nf,&e))return CMF_EXACT_INVALID_INPUT;
    size_t doubles=n,part,bytes=sizeof(CMFOuterContext);
    if(outer_product(2,e,&part)||outer_add(&doubles,part))return CMF_EXACT_INVALID_INPUT;
    if(workers){
        size_t block,linear;
        if(outer_product((size_t)moments,m,&block)||outer_product(8,nf,&linear)||
           outer_add(&block,linear)||outer_product((size_t)workers,block,&part)||
           outer_add(&doubles,part)||outer_add(&doubles,samples)||
           outer_product((size_t)moments,(size_t)p->observations,&part)||outer_add(&doubles,part)||
           outer_add(&bytes,sizeof(CMFOuterWeighted))||
           outer_product(2*samples,sizeof(double),&part)||outer_add(&bytes,part)||
           outer_product((size_t)p->observations,sizeof(int),&part)||outer_add(&bytes,part))
            return CMF_EXACT_INVALID_INPUT;
    }else if(outer_product(5,m,&part)||outer_add(&doubles,part)||
             outer_product(8,nf,&part)||outer_add(&doubles,part)||
             outer_product(3,samples,&part)||outer_add(&doubles,part))
        return CMF_EXACT_INVALID_INPUT;
    if(outer_add(&doubles,(size_t)p->exterior_nodes)||
       outer_product(doubles,sizeof(double),&part)||outer_add(&bytes,part)||
       outer_product((size_t)p->steps,sizeof(CMFOuterStep),&part)||outer_add(&bytes,part)||
       outer_product((size_t)p->rays,sizeof(CMFOuterRay),&part)||outer_add(&bytes,part)||
       outer_product((size_t)p->observations,sizeof(CMFOuterObservation),&part)||outer_add(&bytes,part)||
       outer_product(2*samples,sizeof(int),&part)||outer_add(&bytes,part))
        return CMF_EXACT_INVALID_INPUT;
    size_t maximum_window=0;
    for(int i=0;i<p->steps;i++){
        const CMFOuterStep *s=p->step+i;
        if(!isfinite(s->length)||s->length<0||!isfinite(s->drift)||s->drift<0||
           (s->length==0&&s->drift!=0)||!isfinite(s->weight)||s->weight<0||s->weight>1||
           (s->exterior!=0&&s->exterior!=1)||s->index<0||
           s->index>=(s->exterior ? p->exterior_nodes-1 : p->cells)||
           !outer_span(s->observation_begin,s->observation_count,p->observations))
            return CMF_EXACT_INVALID_INPUT;
        double beta=s->drift/p->dlognu;
        if(!isfinite(beta)||beta>(double)(INT_MAX-p->frequencies-2))
            return CMF_EXACT_INVALID_INPUT;
        size_t window=(size_t)ceil(beta)+1;
        if(window>maximum_window)maximum_window=window;
    }
    /* 기존 positive window는 double4개 node의 front/back 두 stack을 쓴다.
     * 입력/외부 출력은 제외하며 최대 임시 window와1MiB 런타임 여유는 포함한다. */
    if(outer_product(maximum_window,8*sizeof(double)*(size_t)(workers ? workers : 1),&part)||outer_add(&bytes,part)||
       outer_add(&bytes,1024*1024))return CMF_EXACT_INVALID_INPUT;
    if(work_bytes)*work_bytes=bytes;
    if(stored_doubles)*stored_doubles=doubles;
    return CMF_EXACT_OK;
}

CMFExactStatus cmf_outer_workspace_bytes(const CMFOuterProblem *p,size_t *work_bytes)
{
    if(!work_bytes)return CMF_EXACT_INVALID_INPUT;
    return outer_storage_mode(p,0,3,1,NULL,work_bytes);
}

CMFExactStatus cmf_outer_weighted_workspace_bytes(const CMFOuterProblem *p,
    int workers,int moments,int approved_cpus,size_t *work_bytes)
{
    if(work_bytes)*work_bytes=0;
    if(!work_bytes||workers<1)return CMF_EXACT_INVALID_INPUT;
    return outer_storage_mode(p,workers,moments,approved_cpus,NULL,work_bytes);
}

static CMFExactStatus outer_build_weights(CMFOuterContext *c);

static CMFExactStatus outer_create_mode(const CMFOuterProblem *p,int workers,int moments,int cpus,
    CMFOuterContext **result,size_t *work_bytes)
{
    if(result)*result=NULL;
    if(work_bytes)*work_bytes=0;
    if(!result)return CMF_EXACT_INVALID_INPUT;
    size_t doubles,bytes;
    CMFExactStatus sizing=outer_storage_mode(p,workers,moments,cpus,&doubles,&bytes);
    if(sizing!=CMF_EXACT_OK)return sizing;
    if(!p->ray||!p->observation||!p->turning_mu||!p->exterior_ratio||
       !p->cell_chi||!p->cell_es||!p->cell_eta||!p->node_chi||!p->node_es||
       !p->node_eta||!p->inner_planck||!p->inner_gradient||!p->work_budget_bytes)
        return CMF_EXACT_INVALID_INPUT;
    size_t nf=(size_t)p->frequencies,samples=(size_t)p->cells+2;
    size_t n=(size_t)p->cells*nf,m=samples*nf,e=(size_t)p->exterior_nodes*nf;
    if(work_bytes)*work_bytes=bytes;
    if(bytes>p->work_budget_bytes)return CMF_EXACT_ALLOCATION_FAILED;
    for(int i=0;i<p->rays;i++){
        const CMFOuterRay *r=p->ray+i;
        if(!outer_span(r->minus_begin,r->minus_count,p->steps)||
           !outer_span(r->plus_begin,r->plus_count,p->steps)||
           !outer_span(r->turn_begin,r->turn_count,p->observations)||
           (r->core!=0&&r->core!=1)||!isfinite(r->inner_mu)||fabs(r->inner_mu)>1)
            return CMF_EXACT_INVALID_INPUT;
    }
    for(int i=0;i<p->observations;i++)
        if(p->observation[i].sample<0||(size_t)p->observation[i].sample>=samples||
           !isfinite(p->observation[i].mu)||fabs(p->observation[i].mu)>1)
            return CMF_EXACT_INVALID_INPUT;
    for(size_t i=0;i<samples;i++)
        if(!isfinite(p->turning_mu[i])||fabs(p->turning_mu[i])>=1)return CMF_EXACT_INVALID_INPUT;
    for(int i=0;i<p->exterior_nodes;i++)
        if(!(p->exterior_ratio[i]>0)||!isfinite(p->exterior_ratio[i])||
           (i==0 ? p->exterior_ratio[i]!=1. : p->exterior_ratio[i]>=p->exterior_ratio[i-1]))
            return CMF_EXACT_INVALID_INPUT;
    for(size_t i=0;i<n;i++)
        if(!isfinite(p->cell_chi[i])||!isfinite(p->cell_es[i])||p->cell_es[i]<0||
           !isfinite(p->cell_eta[i])||p->cell_eta[i]<0)return CMF_EXACT_INVALID_INPUT;
    for(size_t i=0;i<2*nf;i++)
        if(!isfinite(p->node_chi[i])||!(p->node_es[i]>0)||
           !isfinite(p->node_es[i])||p->node_eta[i]<0||!isfinite(p->node_eta[i]))
            return CMF_EXACT_INVALID_INPUT;
    for(size_t i=0;i<nf;i++)
        if(!isfinite(p->inner_planck[i])||p->inner_planck[i]<0||!isfinite(p->inner_gradient[i]))
            return CMF_EXACT_INVALID_INPUT;
    CMFOuterContext *c=calloc(1,sizeof(*c));
    if(!c)return CMF_EXACT_ALLOCATION_FAILED;
    c->p=*p;c->n=n;c->m=m;c->e=e;c->samples=samples;c->work_bytes=bytes;
    c->steps=malloc((size_t)p->steps*sizeof(*c->steps));
    c->rays=malloc((size_t)p->rays*sizeof(*c->rays));
    c->observations=malloc((size_t)p->observations*sizeof(*c->observations));
    c->data=calloc(doubles,sizeof(double));
    if(workers)c->weighted=calloc(1,sizeof(*c->weighted));
    else c->seen_m=calloc(2*samples,sizeof(int));
    if(!c->steps||!c->rays||!c->observations||!c->data||
       (workers ? !c->weighted : !c->seen_m)){
        cmf_outer_destroy(c);return CMF_EXACT_ALLOCATION_FAILED;
    }
    memcpy(c->steps,p->step,(size_t)p->steps*sizeof(*c->steps));
    memcpy(c->rays,p->ray,(size_t)p->rays*sizeof(*c->rays));
    memcpy(c->observations,p->observation,(size_t)p->observations*sizeof(*c->observations));
    c->p.step=c->steps;c->p.ray=c->rays;c->p.observation=c->observations;
    c->cell_eta=c->data;c->log_chi=c->cell_eta+n;c->log_eta=c->log_chi+e;
    if(workers){
        CMFOuterWeighted *w=c->weighted;
        w->workers=workers;w->moments=moments;w->approved_cpus=cpus;
        w->block=(size_t)moments*m+8*nf;w->data=c->log_eta+e;
        c->moments=w->data;c->turning=w->data+(size_t)workers*w->block;
        c->ratio=c->turning+samples;w->weights=c->ratio+p->exterior_nodes;
    }else{
        c->previous_m=c->log_eta+e;c->previous_p=c->previous_m+m;
        c->moments=c->previous_p+m;c->intensity=c->moments+3*m;
        c->alternate=c->intensity+nf;c->chi=c->alternate+nf;c->eta=c->chi+nf;
        c->scratch=c->eta+nf;c->mu_m=c->scratch+4*nf;c->mu_p=c->mu_m+samples;
        c->turning=c->mu_p+samples;c->ratio=c->turning+samples;
        c->seen_p=c->seen_m+samples;
    }
    memcpy(c->turning,p->turning_mu,samples*sizeof(double));
    memcpy(c->ratio,p->exterior_ratio,(size_t)p->exterior_nodes*sizeof(double));
    c->p.turning_mu=c->turning;c->p.exterior_ratio=c->ratio;
    CMFExactStatus status=outer_fixed_opacity(c);
    if(status==CMF_EXACT_OK&&workers)status=outer_build_weights(c);
    if(status!=CMF_EXACT_OK){cmf_outer_destroy(c);return status;}
    *result=c;return CMF_EXACT_OK;
}

CMFExactStatus cmf_outer_create(const CMFOuterProblem *p,
    CMFOuterContext **result,size_t *work_bytes)
{
    return outer_create_mode(p,0,3,1,result,work_bytes);
}

CMFExactStatus cmf_outer_create_weighted(const CMFOuterProblem *p,
    int workers,int moments,int approved_cpus,CMFOuterContext **result,size_t *work_bytes)
{
    if(workers<1){if(result)*result=NULL;if(work_bytes)*work_bytes=0;return CMF_EXACT_INVALID_INPUT;}
    return outer_create_mode(p,workers,moments,approved_cpus,result,work_bytes);
}

static CMFExactStatus outer_weight_observations(CMFOuterContext *c,int32_t begin,int32_t count,
    int outward,int *last,double *mu,int *used)
{
    size_t observations=(size_t)c->p.observations;
    for(int32_t q=begin;q<begin+count;q++){
        const CMFOuterObservation *o=c->observations+q;
        size_t s=(size_t)o->sample;
        size_t at=(size_t)outward*c->samples+s;
        if(used[q]++)return CMF_EXACT_INVALID_INPUT;
        if(last[at]<0){
            if(o->mu!=(outward ? 1. : -1.))return CMF_EXACT_INVALID_INPUT;
        }else{
            if(outward ? o->mu>mu[at]||o->mu<c->turning[s] :
                         o->mu<mu[at]||o->mu>c->turning[s])return CMF_EXACT_INVALID_INPUT;
            double x=outward ? o->mu : mu[at],y=outward ? mu[at] : o->mu;
            size_t left=outward ? (size_t)q : (size_t)last[at];
            size_t right=outward ? (size_t)last[at] : (size_t)q;
            double width=y-x,*w=c->weighted->weights;
            w[left]+=width/4;w[right]+=width/4;
            if(c->weighted->moments==3){
                w[observations+left]+=width*(2*x+y)/12;
                w[observations+right]+=width*(x+2*y)/12;
                w[2*observations+left]+=width*(3*x*x+2*x*y+y*y)/24;
                w[2*observations+right]+=width*(x*x+2*x*y+3*y*y)/24;
            }
        }
        last[at]=q;mu[at]=o->mu;
    }
    return CMF_EXACT_OK;
}

static CMFExactStatus outer_build_weights(CMFOuterContext *c)
{
    size_t count=2*c->samples;
    int *last=malloc(count*sizeof(int)),*used=calloc((size_t)c->p.observations,sizeof(int));
    double *mu=calloc(count,sizeof(double));
    if(!last||!used||!mu){free(last);free(used);free(mu);return CMF_EXACT_ALLOCATION_FAILED;}
    for(size_t i=0;i<count;i++)last[i]=-1;
    CMFExactStatus status=CMF_EXACT_OK;
    for(int32_t r=0;r<c->p.rays&&status==CMF_EXACT_OK;r++){
        const CMFOuterRay *ray=c->rays+r;
        for(int direction=0;direction<2&&status==CMF_EXACT_OK;direction++){
            if(direction)status=outer_weight_observations(c,ray->turn_begin,ray->turn_count,
                                                         1,last,mu,used);
            int32_t begin=direction ? ray->plus_begin : ray->minus_begin;
            int32_t steps=direction ? ray->plus_count : ray->minus_count;
            for(int32_t k=begin;k<begin+steps&&status==CMF_EXACT_OK;k++){
                const CMFOuterStep *s=c->steps+k;
                status=outer_weight_observations(c,s->observation_begin,s->observation_count,
                                                 direction,last,mu,used);
            }
        }
    }
    for(size_t i=0;i<c->samples;i++)
        if(last[i]<0||last[c->samples+i]<0||mu[i]!=c->turning[i]||
           mu[c->samples+i]!=c->turning[i])status=CMF_EXACT_INVALID_INPUT;
    for(int32_t i=0;i<c->p.observations;i++)if(used[i]!=1)status=CMF_EXACT_INVALID_INPUT;
    free(last);free(used);free(mu);return status;
}

static CMFExactStatus outer_sources(CMFOuterContext *c,const double *j)
{
    const CMFOuterProblem *p=&c->p;
    size_t nf=(size_t)p->frequencies;
    for(size_t i=0;i<c->m;i++)
        if(j[i]<0||!isfinite(j[i]))return CMF_EXACT_INVALID_INPUT;
    for(size_t i=0;i<c->n;i++){
        c->cell_eta[i]=p->cell_eta[i]+p->cell_es[i]*j[i];
        if(!isfinite(c->cell_eta[i]))return CMF_EXACT_NONFINITE;
    }
    for(size_t b=0;b<nf;b++){
        double a=p->node_eta[b]+p->node_es[b]*j[c->n+b];
        double v=p->node_eta[nf+b]+p->node_es[nf+b]*j[c->n+nf+b];
        if(a<0||v<0||!isfinite(a)||!isfinite(v))return CMF_EXACT_NONFINITE;
        double power=3.5;
        if(a>0&&v>0){
            double ratio=v/a;
            double logarithm=ratio>0&&isfinite(ratio) ? log(ratio) : log(v)-log(a);
            power=fmax(3.5,logarithm/p->log_reference_radius_ratio);
        }
        if(!isfinite(power))return CMF_EXACT_NONFINITE;
        for(int i=0;i<p->exterior_nodes;i++){
            double eta=i==0 ? a : fmax(CMF_OUTER_REFERENCE_ETA_MIN_CGS,a*pow(c->ratio[i],power));
            /* 기존 CMFGEN 외부 폐쇄관계의 처리이며 새 강도 floor가 아니다. */
            if(eta<0||!isfinite(eta))return CMF_EXACT_NONFINITE;
            c->log_eta[(size_t)i*nf+b]=eta>0 ? log(eta) : -INFINITY;
        }
    }
    return CMF_EXACT_OK;
}

static CMFExactStatus outer_observe(CMFOuterContext *c,int32_t begin,int32_t count,
    int outward,const double *intensity)
{
    size_t nf=(size_t)c->p.frequencies;
    if(c->weighted){
        const CMFOuterWeighted *w=c->weighted;
        for(int32_t q=begin;q<begin+count;q++){
            size_t base=(size_t)c->observations[q].sample*nf;
            for(int k=0;k<w->moments;k++){
                double weight=w->weights[(size_t)k*c->p.observations+(size_t)q];
                for(size_t b=0;b<nf;b++)c->moments[(size_t)k*c->m+base+b]+=weight*intensity[b];
            }
        }
        return CMF_EXACT_OK;
    }
    double *previous=outward ? c->previous_p : c->previous_m;
    double *last_mu=outward ? c->mu_p : c->mu_m;
    int *seen=outward ? c->seen_p : c->seen_m;
    for(int32_t at=begin;at<begin+count;at++){
        const CMFOuterObservation *o=c->observations+at;
        size_t s=(size_t)o->sample,base=s*nf;
        double mu=o->mu;
        if(!seen[s]){
            if(mu!=(outward ? 1. : -1.))return CMF_EXACT_INVALID_INPUT;
        }else{
            if(outward ? mu>last_mu[s]||mu<c->turning[s] :
                         mu<last_mu[s]||mu>c->turning[s])return CMF_EXACT_INVALID_INPUT;
            double x=outward ? mu : last_mu[s],y=outward ? last_mu[s] : mu;
            double width=y-x;
            for(size_t b=0;b<nf;b++){
                size_t i=base+b;
                double a=outward ? intensity[b] : previous[i];
                double v=outward ? previous[i] : intensity[b];
                c->moments[i]+=width*(a+v)/4;
                c->moments[c->m+i]+=width*((2*x+y)*a+(x+2*y)*v)/12;
                c->moments[2*c->m+i]+=width*((3*x*x+2*x*y+y*y)*a+
                                                        (x*x+2*x*y+3*y*y)*v)/24;
            }
        }
        memcpy(previous+base,intensity,nf*sizeof(double));
        last_mu[s]=mu;seen[s]=1;
    }
    return CMF_EXACT_OK;
}

static CMFExactStatus outer_sequence(CMFOuterContext *c,int32_t begin,int32_t count,
    int outward,double **intensity,double **alternate)
{
    const CMFOuterProblem *p=&c->p;
    size_t nf=(size_t)p->frequencies;
    for(int32_t at=begin;at<begin+count;at++){
        const CMFOuterStep *s=c->steps+at;
        const double *chi,*eta;
        if(s->exterior){
            size_t a=(size_t)s->index*nf,b=a+nf;
            double w=s->weight;
            for(size_t k=0;k<nf;k++){
                c->chi[k]=p->node_chi[k]==0 ? 0. : copysign(
                    exp((1-w)*c->log_chi[a+k]+w*c->log_chi[b+k]),p->node_chi[k]);
                c->eta[k]=exp(w==0 ? c->log_eta[a+k] : w==1 ? c->log_eta[b+k] :
                    (1-w)*c->log_eta[a+k]+w*c->log_eta[b+k]);
            }
            chi=c->chi;eta=c->eta;
        }else{
            chi=p->cell_chi+(size_t)s->index*nf;
            eta=c->cell_eta+(size_t)s->index*nf;
        }
        CMFExactStatus status=cmf_exact_positive_raw_segment(p->frequencies,p->dlognu,
            s->length,s->drift,chi,eta,*intensity,*alternate,c->scratch,4*nf);
        if(status!=CMF_EXACT_OK)return status;
        double *swap=*intensity;*intensity=*alternate;*alternate=swap;
        status=outer_observe(c,s->observation_begin,s->observation_count,outward,*intensity);
        if(status!=CMF_EXACT_OK)return status;
    }
    return CMF_EXACT_OK;
}

static CMFExactStatus outer_one_ray(CMFOuterContext *c,int32_t r)
{
    const CMFOuterProblem *p=&c->p;
    size_t nf=(size_t)p->frequencies;
    const CMFOuterRay *ray=c->rays+r;
    double *intensity=c->intensity,*alternate=c->alternate;
    memset(intensity,0,nf*sizeof(double));
    CMFExactStatus status=outer_sequence(c,ray->minus_begin,ray->minus_count,0,&intensity,&alternate);
    if(status!=CMF_EXACT_OK)return status;
    if(ray->core){
        for(size_t b=0;b<nf;b++){
            intensity[b]=p->inner_planck[b]+ray->inner_mu*p->inner_gradient[b];
            if(intensity[b]<0||!isfinite(intensity[b]))return CMF_EXACT_NONFINITE;
        }
    }
    /* 비core 접선의 동일 강도 표본은 두 방향의 mu=-beta 접합점이다. */
    status=outer_observe(c,ray->turn_begin,ray->turn_count,1,intensity);
    if(status!=CMF_EXACT_OK)return status;
    return outer_sequence(c,ray->plus_begin,ray->plus_count,1,&intensity,&alternate);
}

static CMFExactStatus outer_weighted_rays(CMFOuterContext *c)
{
    CMFOuterWeighted *w=c->weighted;
    size_t nf=(size_t)c->p.frequencies;
    int threads=1;
#ifdef _OPENMP
    threads=omp_get_max_threads();
    if(threads>w->workers)threads=w->workers;
    if(threads>w->approved_cpus)threads=w->approved_cpus;
#pragma omp parallel for num_threads(threads) schedule(static)
#else
    (void)threads;
#endif
    for(int worker=0;worker<w->workers;worker++){
        if(worker==0){
#ifdef _OPENMP
            w->observed_threads=omp_get_num_threads();
#else
            w->observed_threads=1;
#endif
        }
        CMFOuterContext local=*c;
        local.moments=w->data+(size_t)worker*w->block;
        memset(local.moments,0,(size_t)w->moments*c->m*sizeof(double));
        local.intensity=local.moments+(size_t)w->moments*c->m;
        local.alternate=local.intensity+nf;local.chi=local.alternate+nf;
        local.eta=local.chi+nf;local.scratch=local.eta+nf;
        CMFExactStatus status=CMF_EXACT_OK;
        for(int32_t r=worker;r<c->p.rays&&status==CMF_EXACT_OK;r+=w->workers)
            status=outer_one_ray(&local,r);
        w->status[worker]=(int)status;
    }
    for(int worker=0;worker<w->workers;worker++)
        if(w->status[worker]!=CMF_EXACT_OK)return (CMFExactStatus)w->status[worker];
    /* worker0를 최종 저장소로 사용하고 고정 worker 순서로 더한다. */
    for(size_t i=0;i<(size_t)w->moments*c->m;i++){
        for(int worker=1;worker<w->workers;worker++)
            c->moments[i]+=w->data[(size_t)worker*w->block+i];
        if(!isfinite(c->moments[i])||((i<c->m||i>=2*c->m)&&c->moments[i]<0))
            return CMF_EXACT_NONFINITE;
    }
    return CMF_EXACT_OK;
}

static CMFExactStatus outer_compute(CMFOuterContext *c,const double *own_j)
{
    if(c->weighted)c->weighted->observed_threads=0;
    CMFExactStatus status=outer_sources(c,own_j);
    if(status!=CMF_EXACT_OK)return status;
    if(c->weighted)return outer_weighted_rays(c);
    const CMFOuterProblem *p=&c->p;
    memset(c->moments,0,3*c->m*sizeof(double));
    memset(c->seen_m,0,2*c->samples*sizeof(int));
    for(int32_t r=0;r<p->rays;r++){
        status=outer_one_ray(c,r);
        if(status!=CMF_EXACT_OK)return status;
    }
    for(size_t i=0;i<c->samples;i++)
        if(!c->seen_m[i]||!c->seen_p[i]||c->mu_m[i]!=c->turning[i]||c->mu_p[i]!=c->turning[i])
            return CMF_EXACT_INVALID_INPUT;
    for(size_t i=0;i<3*c->m;i++)
        if(!isfinite(c->moments[i])||(i<c->m&&c->moments[i]<0)||
           (i>=2*c->m&&c->moments[i]<0))return CMF_EXACT_NONFINITE;
    return CMF_EXACT_OK;
}

int cmf_outer_parallel_threads(const CMFOuterContext *c)
{
    return c&&c->weighted ? c->weighted->observed_threads : 0;
}

CMFExactStatus cmf_outer_apply(CMFOuterContext *c,const double *own_j,
    double *new_j,double *new_h,double *new_k)
{
    if(!c||!own_j||!new_j||(c->weighted&&c->weighted->moments==1&&(new_h||new_k)))
        return CMF_EXACT_INVALID_INPUT;
    CMFExactStatus status=outer_compute(c,own_j);
    if(status!=CMF_EXACT_OK)return status;
    memcpy(new_j,c->moments,c->m*sizeof(double));
    if(new_h)memcpy(new_h,c->moments+c->m,c->m*sizeof(double));
    if(new_k)memcpy(new_k,c->moments+2*c->m,c->m*sizeof(double));
    return CMF_EXACT_OK;
}

static double outer_difference(size_t n,const double *a,const double *b,double *absolute)
{
    double maximum=0.,difference=0.;
    for(size_t i=0;i<n;i++){
        if(a[i]>maximum)maximum=a[i];
        double d=fabs(a[i]-b[i]);if(d>difference)difference=d;
    }
    if(absolute)*absolute=difference;
    return maximum>0 ? difference/maximum : difference==0 ? 0. : INFINITY;
}

CMFExactStatus cmf_outer_solve(CMFOuterContext *c,double *own_j,
    int iteration_cap,double tolerance,CMFOuterReport *report)
{
    if(!report)return CMF_EXACT_INVALID_INPUT;
    memset(report,0,sizeof(*report));report->status=CMF_EXACT_INVALID_INPUT;
    report->change_max_norm_ratio=INFINITY;report->residual_max=INFINITY;
    report->residual_max_norm_ratio=INFINITY;
    if(!c||!own_j||iteration_cap<1||iteration_cap>INT_MAX/2||
       !(tolerance>0)||!isfinite(tolerance))return report->status;
    report->work_bytes=c->work_bytes;
    for(int step=1;step<=iteration_cap;step++){
        report->formal_calls++;
        CMFExactStatus status=outer_compute(c,own_j);
        if(status!=CMF_EXACT_OK){report->status=status;return status;}
        report->change_max_norm_ratio=outer_difference(c->m,c->moments,own_j,NULL);
        memcpy(own_j,c->moments,c->m*sizeof(double));
        report->iterations=step;
        if(report->change_max_norm_ratio<=tolerance){
            report->formal_calls++;
            status=outer_compute(c,own_j);
            if(status!=CMF_EXACT_OK){report->status=status;return status;}
            /* 실제 반환 후보를 분모로 쓴다. 마지막 변화량을 해오차로 재해석하지 않는다. */
            report->residual_max_norm_ratio=outer_difference(c->m,own_j,c->moments,&report->residual_max);
            if(report->residual_max_norm_ratio<=tolerance){
                report->status=CMF_EXACT_OK;return CMF_EXACT_OK;
            }
        }
    }
    report->status=CMF_EXACT_NOT_CONVERGED;return report->status;
}
