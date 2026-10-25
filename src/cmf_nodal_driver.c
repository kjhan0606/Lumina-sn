/* 광선/적응의 C 외곽. 계수/J/모형 갱신은 하지 않는 동결 장 연산자다. */
#define _POSIX_C_SOURCE 200809L
#include "cmf_nodal_driver.h"
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LIGHT 2.99792458e10
typedef struct {double a,b,score;int depth;double *maps;unsigned char *safe;} Leaf;
typedef struct {int begin,width,count,dirty;double error,maximum;Leaf *leaves;} Slab;
typedef struct {size_t bytes;} Allocation;
typedef struct {
    CMFNodalInput *input;int ns,nf,nr,panels,block_width,max_leaves,max_depth;
    const double *r,*v,*gx,*gw;double h,rtol,atol,started,max_seconds;
    uint64_t max_work,max_interval_work,live,cache_bytes;int cached;CMFNodalDriverStats *stats;CMFCachedDriverStats *cache_stats;
} Driver;
typedef struct {double z,r,delta;} Point;
static int timed_out(const Driver *d,double interval_started);

static double now(void)
{
    struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return NAN;
    return t.tv_sec+1e-9*t.tv_nsec;
}

static void *allocate(Driver *d,size_t bytes)
{
    if(bytes>SIZE_MAX-sizeof(Allocation))return NULL;
    Allocation *p=malloc(sizeof(*p)+bytes);if(!p)return NULL;p->bytes=sizeof(*p)+bytes;
    d->live+=p->bytes;
    if(d->live>d->stats->driver_heap_peak_bytes)d->stats->driver_heap_peak_bytes=d->live;
    return p+1;
}

static void release(Driver *d,void *pointer)
{
    if(!pointer)return;
    Allocation *p=(Allocation *)pointer-1;d->live-=p->bytes;free(p);
}

static void drop_leaf_cache(Driver *d,Leaf *leaf)
{
    if(leaf->maps){d->cache_bytes-=((Allocation *)leaf->maps-1)->bytes;release(d,leaf->maps);leaf->maps=NULL;}
    if(leaf->safe){d->cache_bytes-=((Allocation *)leaf->safe-1)->bytes;release(d,leaf->safe);leaf->safe=NULL;}
}

static void drop_slab_cache(Driver *d,Slab *s)
{
    if(!s->leaves)return;
    int present=0;
    for(int i=0;i<s->count;i++){if(s->leaves[i].maps)present=1;drop_leaf_cache(d,s->leaves+i);}
    if(present)d->cache_stats->cache_evictions++;
}

static uint64_t remaining_work(const Driver *d,uint64_t interval_work)
{
    uint64_t current=d->stats->scalar_formals;
    uint64_t a=d->max_work-current,b=d->max_interval_work-(current-interval_work);
    return a<b ? a : b;
}

static uint64_t leaf_calls(const double *d,double h)
{
    double q[3]={(d[0]-d[2])/h,(d[0]-d[1])/h,(d[1]-d[2])/h};uint64_t count=3;
    for(int k=0;k<3;k++)if(q[k]>=.75&&q[k]<=1.25)count++;
    return count;
}

static int cached_compare(Driver *d,Slab *s,const double *lengths,const double *deltas,const double *radii,
    const double *initial,double *out,double *errors,double *responses,double started,uint64_t interval_work)
{
    int n=s->count,w=s->width,code=0;size_t stride=((size_t)w+7)/8;
    uint64_t projected=d->cache_bytes;double maximum_span=0.;
    for(int row=0;row<n;row++){
        if(!s->leaves[row].maps){
            int arrays=6+(int)(leaf_calls(deltas+3*row,d->h)-3)*2;
            projected+=(uint64_t)arrays*w*sizeof(double)+stride+2*sizeof(Allocation);
        }
        double q=(deltas[3*row]-deltas[3*row+2])/d->h;
        if(q>maximum_span)maximum_span=q;
    }
    /* 이 함수의 임시 배열, evaluate의 geometry/error/scratch, kernel의 제한 cache를 포함한다. */
    uint64_t kernel_bound=4*1024*1024+(uint64_t)fmin(d->nf-1.,ceil(maximum_span)+1.)*4*sizeof(CMFHistoryMap);
    uint64_t workspace=projected+((uint64_t)n*11+2*(uint64_t)n*w+2*w+12*w)*sizeof(double)+
        (uint64_t)n*(sizeof(double *)+stride+sizeof(Leaf))+w+kernel_bound;
    if(workspace>128*1024*1024)return 6;
    if(workspace>d->cache_stats->cache_workspace_peak_bytes)d->cache_stats->cache_workspace_peak_bytes=workspace;
    if(kernel_bound>d->stats->kernel_heap_peak_bound_bytes)d->stats->kernel_heap_peak_bound_bytes=kernel_bound;
    double *raw=allocate(d,(size_t)12*w*sizeof(double));
    unsigned char *safe=allocate(d,(size_t)w),*masks=allocate(d,(size_t)n*stride);
    const double **maps=allocate(d,(size_t)n*sizeof(double *));
    if(!raw||!safe||!masks||!maps){code=3;goto done;}
    for(int row=0;row<n;row++){
        Leaf *leaf=s->leaves+row;
        if(!leaf->maps){
            if(timed_out(d,started)){code=4;goto done;}
            uint64_t expected=leaf_calls(deltas+3*row,d->h);
            if(expected*(uint64_t)w>remaining_work(d,interval_work)){code=5;goto done;}
            CMFHistoryStats stats;uint64_t calls=0,checks=0;
            code=cmf_nodal_history_maps(d->input,d->nf,s->begin,w,1,d->h,lengths+3*row,deltas+3*row,radii+5*row,
                raw,safe,&stats,&calls,&checks);
            d->stats->scalar_formals+=calls*(uint64_t)w;d->stats->kernel_entries+=calls;
            d->stats->branch_scalar_formals+=checks*(uint64_t)w;d->cache_stats->cache_build_scalar_formals+=calls*(uint64_t)w;
            if(code)goto done;
            if(calls!=expected){code=2;goto done;}
            int arrays=6+(int)(expected-3)*2;
            leaf->maps=allocate(d,(size_t)arrays*w*sizeof(double));
            if(!leaf->maps){code=3;goto done;}
            d->cache_bytes+=((Allocation *)leaf->maps-1)->bytes;
            leaf->safe=allocate(d,stride);
            if(!leaf->safe){code=3;goto done;}
            d->cache_bytes+=((Allocation *)leaf->safe-1)->bytes;
            int at=0;
            for(int piece=0;piece<3;piece++)for(int k=0;k<2;k++){
                memcpy(leaf->maps+(size_t)at*w,raw+(size_t)(4*piece+k)*w,(size_t)w*sizeof(double));at++;
            }
            const double *v=deltas+3*row;
            double q[3]={(v[0]-v[2])/d->h,(v[0]-v[1])/d->h,(v[1]-v[2])/d->h};
            for(int piece=0;piece<3;piece++)if(q[piece]>=.75&&q[piece]<=1.25)for(int k=2;k<4;k++){
                memcpy(leaf->maps+(size_t)at*w,raw+(size_t)(4*piece+k)*w,(size_t)w*sizeof(double));at++;
            }
            memset(leaf->safe,0,stride);
            for(int f=0;f<w;f++)if(safe[f])leaf->safe[f/8]|=(unsigned char)(1u<<(f%8));
            d->cache_stats->cache_leaf_builds++;
        }
        maps[row]=leaf->maps;memcpy(masks+(size_t)row*stride,leaf->safe,stride);
    }
    if(timed_out(d,started)){code=4;goto done;}
    uint64_t fallback[3]={0,0,0};
    code=cmf_nodal_history_replay(d->input,d->nf,s->begin,w,n,d->h,lengths,deltas,radii,maps,masks,
        initial,out,errors,responses,remaining_work(d,interval_work),fallback);
    d->stats->scalar_formals+=fallback[0];d->stats->kernel_entries+=fallback[1];
    d->stats->branch_scalar_formals+=fallback[2];d->cache_stats->cache_fallback_scalar_formals+=fallback[0];
    d->cache_stats->cache_replay_leaf_columns+=(uint64_t)n*w;
done:
    release(d,maps);release(d,masks);release(d,safe);release(d,raw);return code;
}

static int timed_out(const Driver *d,double interval_started)
{
    double elapsed=now();return !isfinite(elapsed)||elapsed-d->started>=d->max_seconds||elapsed-interval_started>=60.;
}

static int doppler(const Driver *d,double impact,double z,double native_radius,double *value)
{
    double r;
    if(native_radius>0){
        r=native_radius;
        if(r<impact||fabs(z)!=sqrt((r-impact)*(r+impact)))return 8;
    }else r=hypot(impact,z);
    if(!isfinite(r)||r<d->r[d->nr-1]||r>d->r[0])return 7;
    int lo=0,hi=d->nr-1;
    while(hi-lo>1){int mid=lo+(hi-lo)/2;if(r<d->r[mid])lo=mid;else hi=mid;}
    double v=r==d->r[lo] ? d->v[lo] : r==d->r[hi] ? d->v[hi] :
        (d->v[lo]-d->v[hi])/(d->r[lo]-d->r[hi])*(r-d->r[hi])+d->v[hi];
    double beta=v/LIGHT;
    *value=(1.-beta*z/r)/sqrt((1.-beta)*(1.+beta));
    return !isfinite(*value)||*value<=0 ? 7 : 0;
}

static int point(const Driver *d,const double *step,double observer,double x,Point *out)
{
    double z=x==0. ? step[1] : x==1. ? step[2] : step[1]+(step[2]-step[1])*x;
    double native=x==0. ? step[3] : x==1. ? step[4] : 0.;
    out->z=z;out->r=native>0 ? native : hypot(step[0],z);
    double value;int code=doppler(d,step[0],z,native,&value);if(code)return code;
    out->delta=log(value/observer);
    return !isfinite(out->delta)||out->delta<0 ? 7 : 0;
}

static int length(const Driver *d,const double *step,Point a,Point b,double *out)
{
    double half=.5*(b.z-a.z),center=.5*(a.z+b.z),sum=0.;
    if(half<0)return 7;
    for(int i=0;i<8;i++){
        double value;int code=doppler(d,step[0],center+half*d->gx[i],0.,&value);if(code)return code;
        sum+=d->gw[i]*value;
    }
    *out=half*sum;return !isfinite(*out)||*out<0 ? 7 : 0;
}

int cmf_nodal_driver_geometry(CMFNodalInput *input,int nr,const double *r,const double *v,
    const double *gx,const double *gw,const double *step,double h,int n,const double *edges,
    double *lengths,double *deltas,double *radii,CMFNodalLocation *locations,double *shift)
{
    int ns,nf;
    if(!input||cmf_nodal_input_shape(input,&ns,&nf)||nr<ns||nr>576||!r||!v||!gx||!gw||!step||
       !isfinite(h)||h<=0||n<1||n>4096||!edges||!lengths||!deltas||!radii||!locations||!shift)return 1;
    for(int i=0;i<nr;i++)if(!isfinite(r[i])||r[i]<=0||(i&&r[i]>=r[i-1])||!isfinite(v[i])||v[i]<0||v[i]>=LIGHT)return 1;
    double sum=0.;for(int k=0;k<8;k++){
        if(!isfinite(gx[k])||fabs(gx[k])>=1||!isfinite(gw[k])||gw[k]<=0)return 1;
        sum+=gw[k];
    }
    if(fabs(sum-2.)>1e-12)return 1;
    for(int k=0;k<5;k++)if(!isfinite(step[k]))return 1;
    if(step[0]<0||step[2]<step[1]||step[3]<step[0]||step[4]<step[0]||
       step[3]<r[nr-1]||step[4]<r[nr-1]||step[3]>r[0]||step[4]>r[0]||
       fabs(step[1])!=sqrt((step[3]-step[0])*(step[3]+step[0]))||
       fabs(step[2])!=sqrt((step[4]-step[0])*(step[4]+step[0])))return 8;
    Driver d={.input=input,.ns=ns,.nf=nf,.nr=nr,.r=r,.v=v,.gx=gx,.gw=gw};
    double observer;int code=doppler(&d,step[0],step[2],step[4],&observer);if(code)return code;
    Point entrance;code=point(&d,step,observer,0.,&entrance);if(code)return code;
    *shift=entrance.delta/h;if(!isfinite(*shift))return 7;
    for(int row=0;row<n;row++){
        double a=edges[2*row],b=edges[2*row+1],m=.5*(a+b);
        if(!isfinite(a)||!isfinite(b)||a<0||b>1||a>=b)return 1;
        double x[5]={a,.5*(a+m),m,.5*(m+b),b};Point p[5];
        for(int k=0;k<5;k++){
            code=point(&d,step,observer,x[k],p+k);if(code)return code;
            radii[5*row+k]=p[k].r;
            code=cmf_nodal_input_prepare(input,p[k].r,locations+5*row+k);if(code)return code;
        }
        for(int k=0;k<3;k++)deltas[3*row+k]=p[2*k].delta;
        if(p[0].delta<p[2].delta||p[2].delta<p[4].delta)return 7;
        code=length(&d,step,p[0],p[4],lengths+3*row);if(code)return code;
        code=length(&d,step,p[0],p[2],lengths+3*row+1);if(code)return code;
        code=length(&d,step,p[2],p[4],lengths+3*row+2);if(code)return code;
    }
    return 0;
}

int cmf_nodal_driver_phase_grid(CMFNodalInput *input,int nr,const double *r,const double *v,
    const double *gx,const double *gw,const double *step,double h,int subdivisions,int capacity,
    int *steps,double *x,double *radii,double *lengths,double *drifts,double *phase,
    CMFNodalLocation *locations,double *maximum_phase_roundoff)
{
    if(!steps||subdivisions<1||subdivisions>16||capacity<0||capacity>65537)return 1;
    double edges[2]={0.,1.},old_length[3],old_delta[3],old_radii[5],span;
    CMFNodalLocation old_locations[5];
    int code=cmf_nodal_driver_geometry(input,nr,r,v,gx,gw,step,h,1,edges,
        old_length,old_delta,old_radii,old_locations,&span);
    if(code)return code;
    double count=ceil(span*subdivisions);if(!isfinite(count)||count>65536)return 3;
    int n=count<1?1:(int)count;*steps=n;
    if(capacity==0)return 0;
    if(capacity<n+1)return 3;
    if(!x||!radii||!lengths||!drifts||!phase||!locations||!maximum_phase_roundoff)return 1;
    int ns,nf;if(cmf_nodal_input_shape(input,&ns,&nf))return 1;
    Driver d={.input=input,.ns=ns,.nf=nf,.nr=nr,.r=r,.v=v,.gx=gx,.gw=gw};
    double observer;code=doppler(&d,step[0],step[2],step[4],&observer);if(code)return code;
    Point previous;code=point(&d,step,observer,0.,&previous);if(code)return code;
    x[0]=0.;radii[0]=previous.r;phase[0]=previous.delta/h;*maximum_phase_roundoff=0.;
    code=cmf_nodal_input_prepare(input,previous.r,locations);if(code)return code;
    for(int k=1;k<=n;k++){
        double target=(double)(n-k)/subdivisions,coordinate=1.;Point value;
        if(k==n){code=point(&d,step,observer,1.,&value);if(code)return code;}
        else{
            double lo=x[k-1],hi=1.;
            for(int iteration=0;iteration<64;iteration++){
                coordinate=lo+.5*(hi-lo);
                code=point(&d,step,observer,coordinate,&value);if(code)return code;
                if(value.delta==target*h||coordinate==lo||coordinate==hi)break;
                if(value.delta>target*h)lo=coordinate;else hi=coordinate;
                if(hi-lo<=4*DBL_EPSILON)break;
            }
        }
        if(coordinate<=x[k-1]||coordinate>1.||value.delta>previous.delta)return 7;
        x[k]=coordinate;radii[k]=value.r;phase[k]=value.delta/h;
        *maximum_phase_roundoff=fmax(*maximum_phase_roundoff,fabs(phase[k]-target));
        code=cmf_nodal_input_prepare(input,value.r,locations+k);if(code)return code;
        code=length(&d,step,previous,value,lengths+k-1);if(code)return code;
        if(lengths[k-1]<=0)return 7;
        drifts[k-1]=previous.delta-value.delta;previous=value;
    }
    return 0;
}

static int evaluate(Driver *d,const double *step,double observer,const double *incident,
    double *candidate,Slab *s,double started,uint64_t interval_work,int reuse)
{
    if(timed_out(d,started))return 4;
    int n=s->count,w=s->width,code=0;
    if((uint64_t)n*w>2097152)return 6;
    double *geometry=allocate(d,(size_t)n*11*sizeof(double));
    double *errors=allocate(d,(size_t)2*n*w*sizeof(double));
    double *scratch=allocate(d,(size_t)2*w*sizeof(double));
    if(!geometry||!errors||!scratch){code=3;goto done;}
    double *lengths=geometry,*deltas=geometry+3*n,*radii=geometry+6*n;
    double *responses=errors+(size_t)n*w,*tail=scratch,*total=scratch+w;
    uint64_t expected=3*(uint64_t)n;double max_span=0.;
    for(int row=0;row<n;row++){
        Leaf leaf=s->leaves[row];double m=.5*(leaf.a+leaf.b);
        double x[5]={leaf.a,.5*(leaf.a+m),m,.5*(m+leaf.b),leaf.b};Point p[5];
        for(int k=0;k<5;k++){code=point(d,step,observer,x[k],p+k);if(code)goto done;radii[5*row+k]=p[k].r;}
        for(int k=0;k<3;k++)deltas[3*row+k]=p[2*k].delta;
        if(p[0].delta<p[2].delta||p[2].delta<p[4].delta){code=7;goto done;}
        code=length(d,step,p[0],p[4],lengths+3*row);if(code)goto done;
        code=length(d,step,p[0],p[2],lengths+3*row+1);if(code)goto done;
        code=length(d,step,p[2],p[4],lengths+3*row+2);if(code)goto done;
        double spans[3]={(p[0].delta-p[4].delta)/d->h,(p[0].delta-p[2].delta)/d->h,(p[2].delta-p[4].delta)/d->h};
        for(int k=0;k<3;k++){if(spans[k]>=.75&&spans[k]<=1.25)expected++;if(spans[k]>max_span)max_span=spans[k];}
    }
    CMFHistoryStats stats={0};
    if(reuse){
        code=cached_compare(d,s,lengths,deltas,radii,incident+s->begin,candidate+s->begin,errors,responses,started,interval_work);
    }else{
        uint64_t work=expected*(uint64_t)w;
        if(work>remaining_work(d,interval_work)){code=5;goto done;}
        uint64_t calls=0,checks=0;
        code=cmf_nodal_history_compare(d->input,d->nf,s->begin,w,n,d->h,lengths,deltas,radii,incident+s->begin,
            candidate+s->begin,errors,responses,&stats,&calls,&checks);
        d->stats->scalar_formals+=calls*(uint64_t)w;d->stats->kernel_entries+=calls;
        d->stats->branch_scalar_formals+=checks*(uint64_t)w;
        if(!code&&calls!=expected)code=2;
    }
    d->stats->slab_evaluations++;
    if(code)goto done;
    if(timed_out(d,started)){code=4;goto done;}
    int capacity=max_span>=8188. ? 8192 : (int)ceil(max_span)+4;
    if(capacity<256)capacity=256;
    if(capacity>d->nf)capacity=d->nf;
    uint64_t kernel=(uint64_t)5*n*sizeof(CMFNodalLocation)+(uint64_t)9*w*8+
        (uint64_t)capacity*(sizeof(int)+4*8)+stats.queue_bytes;
    if(kernel>d->stats->kernel_heap_peak_bound_bytes)d->stats->kernel_heap_peak_bound_bytes=kernel;
    for(int f=0;f<w;f++){tail[f]=1.;total[f]=0.;}
    for(int row=n-1;row>=0;row--){
        double score=0.;
        for(int f=0;f<w;f++){
            size_t at=(size_t)row*w+f;double local=errors[at]*tail[f];total[f]+=local;tail[f]*=responses[at];
            if(!isfinite(local)||!isfinite(total[f])||!isfinite(tail[f])){code=2;goto done;}
            if(local>score)score=local;
        }
        s->leaves[row].score=score;
    }
    s->error=s->maximum=0.;
    for(int f=0;f<w;f++){
        double v=candidate[s->begin+f];if(!isfinite(v)||v<0){code=2;goto done;}
        if(v>s->maximum)s->maximum=v;
        if(total[f]>s->error)s->error=total[f];
    }
    s->dirty=0;if((uint64_t)n>d->stats->max_mesh_leaves)d->stats->max_mesh_leaves=n;
done:
    release(d,scratch);release(d,errors);release(d,geometry);return code;
}

static int refine(Driver *d,Slab *s)
{
    double maximum=0.;for(int i=0;i<s->count;i++)if(s->leaves[i].score>maximum)maximum=s->leaves[i].score;
    int selected=0;for(int i=0;i<s->count;i++)if(s->leaves[i].score>=.5*maximum)selected++;
    if(s->count+selected>d->max_leaves)return 6;
    for(int i=0;i<s->count;i++)if(s->leaves[i].score>=.5*maximum){
        Leaf leaf=s->leaves[i];double m=.5*(leaf.a+leaf.b);
        if(leaf.depth>=d->max_depth||m==leaf.a||m==leaf.b)return 6;
    }
    Leaf *next=allocate(d,(size_t)(s->count+selected)*sizeof(Leaf));if(!next)return 3;
    int at=0;
    for(int i=0;i<s->count;i++){
        Leaf leaf=s->leaves[i];
        if(leaf.score<.5*maximum){next[at++]=leaf;continue;}
        double m=.5*(leaf.a+leaf.b);
        next[at++]=(Leaf){leaf.a,m,0.,leaf.depth+1,NULL,NULL};
        next[at++]=(Leaf){m,leaf.b,0.,leaf.depth+1,NULL,NULL};
    }
    for(int i=0;i<s->count;i++)if(s->leaves[i].score>=.5*maximum)drop_leaf_cache(d,s->leaves+i);
    release(d,s->leaves);s->leaves=next;s->count=at;s->dirty=1;return 0;
}

static int summarize(const Slab *slabs,int count,double *error,double *maximum,int *active)
{
    *error=*maximum=0.;*active=0;
    for(int i=0;i<count;i++){
        if(slabs[i].dirty||!isfinite(slabs[i].error)||!isfinite(slabs[i].maximum))return 2;
        if(slabs[i].error>*error){*error=slabs[i].error;*active=i;}
        if(slabs[i].maximum>*maximum)*maximum=slabs[i].maximum;
    }
    return 0;
}

static int propagate(Driver *d,const double *step,double *ray,double *characteristic)
{
    double started=now(),observer;uint64_t work=d->stats->scalar_formals;
    int code=doppler(d,step[0],step[2],step[4],&observer);if(code)return code;
    Point first;code=point(d,step,observer,0.,&first);if(code)return code;
    uint64_t limited=0;code=cmf_history_reconstruct(d->nf,first.delta/d->h,ray,characteristic,&limited);if(code)return code;
    d->stats->reconstructions++;d->stats->limited_polynomials+=limited;
    int count=(d->nf+d->block_width-1)/d->block_width;
    Slab *slabs=allocate(d,(size_t)count*sizeof(Slab));if(!slabs)return 3;
    memset(slabs,0,(size_t)count*sizeof(Slab));
    for(int i=0;i<count;i++){
        Slab *s=slabs+i;s->begin=i*d->block_width;s->width=d->nf-s->begin;if(s->width>d->block_width)s->width=d->block_width;
        s->count=d->panels;s->dirty=1;s->leaves=allocate(d,(size_t)s->count*sizeof(Leaf));
        if(!s->leaves){code=3;goto done;}
        for(int k=0;k<s->count;k++)s->leaves[k]=(Leaf){(double)k/s->count,(double)(k+1)/s->count,0.,0,NULL,NULL};
    }
    uint64_t rounds=0;
    while(!code){
        if(timed_out(d,started)){code=4;break;}
        for(int i=0;i<count;i++)if(slabs[i].dirty){code=evaluate(d,step,observer,characteristic,ray,slabs+i,started,work,0);if(code)break;}
        if(code)break;
        double error,maximum;int active;
        code=summarize(slabs,count,&error,&maximum,&active);if(code)break;
        double tolerance=d->atol+d->rtol*maximum;
        d->stats->last_global_error=error;d->stats->last_global_tolerance=tolerance;
        rounds++;if(rounds>d->stats->max_rounds)d->stats->max_rounds=rounds;
        if(error<=tolerance)break;
        if(!d->cached){
            for(int i=0;i<count;i++)if(slabs[i].error>tolerance){code=refine(d,slabs+i);if(code)break;}
        }else{
            Slab *s=slabs+active;d->cache_stats->cache_peak_active_slabs=1;
            while(s->error>tolerance){
                if(timed_out(d,started)){code=4;break;}
                code=refine(d,s);if(code)break;
                code=evaluate(d,step,observer,characteristic,ray,s,started,work,1);if(code)break;
                /* 과거 최대값을 누적하지 않는다. 감소하면 이전 통과 slab도 다시 판정한다. */
                int next_active;
                code=summarize(slabs,count,&error,&maximum,&next_active);
                if(code)break;
                tolerance=d->atol+d->rtol*maximum;d->cache_stats->global_rechecks++;
                d->stats->last_global_error=error;d->stats->last_global_tolerance=tolerance;
                rounds++;if(rounds>d->stats->max_rounds)d->stats->max_rounds=rounds;
            }
            drop_slab_cache(d,s);
            if(d->cache_bytes){code=2;break;}
        }
    }
done:
    for(int i=0;i<count;i++){drop_slab_cache(d,slabs+i);release(d,slabs[i].leaves);}
    release(d,slabs);return code;
}

static int accumulate(Driver *d,double *output,const double *ray,int node,const double *weights)
{
    for(int m=0;m<3;m++)for(int f=0;f<d->nf;f++){
        size_t at=((size_t)m*d->ns+node)*d->nf+f;
        output[at]+=weights[m]*ray[f];if(!isfinite(output[at]))return 2;
    }
    return 0;
}

int cmf_nodal_driver_interval(CMFNodalInput *input,int nr,
    const double *r,const double *v,const double *gx,const double *gw,const double *step,
    double h,int panels,int block_width,double seconds,uint64_t max_vectors,
    const double *initial,double *output,CMFNodalDriverStats *stats,CMFCachedDriverStats *cache_stats)
{
    if(!stats||!cache_stats)return 1;
    memset(stats,0,sizeof(*stats));memset(cache_stats,0,sizeof(*cache_stats));int ns,nf;
    if(!input||cmf_nodal_input_shape(input,&ns,&nf)||nr<ns||nr>576||!r||!v||!gx||!gw||!step||
       !initial||!output||initial==output||!isfinite(h)||h<=0||panels<1||panels>16||
       block_width<1||block_width>65536||!isfinite(seconds)||seconds<=0||seconds>60.||
       max_vectors<1||max_vectors>2048){stats->status=1;return 1;}
    for(int i=0;i<nr;i++)if(!isfinite(r[i])||r[i]<=0||(i&&r[i]>=r[i-1])||!isfinite(v[i])||v[i]<0||v[i]>=LIGHT){
        stats->status=1;return 1;
    }
    double sum=0.;for(int i=0;i<8;i++){
        if(!isfinite(gx[i])||fabs(gx[i])>=1||!isfinite(gw[i])||gw[i]<=0){stats->status=1;return 1;}sum+=gw[i];
    }
    if(fabs(sum-2.)>1e-12){stats->status=1;return 1;}
    for(int k=0;k<5;k++)if(!isfinite(step[k])){stats->status=1;return 1;}
    if(step[0]<0||step[2]<step[1]||step[3]<step[0]||step[4]<step[0]||
       step[3]<r[nr-1]||step[4]<r[nr-1]||step[3]>r[0]||step[4]>r[0]||
       fabs(step[1])!=sqrt((step[3]-step[0])*(step[3]+step[0]))||
       fabs(step[2])!=sqrt((step[4]-step[0])*(step[4]+step[0]))){stats->status=8;return 8;}
    for(int f=0;f<nf;f++)if(!isfinite(initial[f])||initial[f]<0){stats->status=1;return 1;}
    Driver d={.input=input,.ns=ns,.nf=nf,.nr=nr,.panels=panels,.block_width=block_width,.max_leaves=4096,.max_depth=32,
        .r=r,.v=v,.gx=gx,.gw=gw,.h=h,.rtol=1e-6,.atol=0.,.started=now(),.max_seconds=seconds,
        .max_work=max_vectors*(uint64_t)nf,.max_interval_work=max_vectors*(uint64_t)nf,.stats=stats,
        .cached=1,.cache_stats=cache_stats};
    double *characteristic=allocate(&d,(size_t)nf*8);if(!characteristic){stats->status=3;return 3;}
    memcpy(output,initial,(size_t)nf*8);
    int code=propagate(&d,step,output,characteristic);release(&d,characteristic);
    stats->elapsed_seconds=now()-d.started;
    if(!code&&stats->elapsed_seconds>=seconds)code=4;
    stats->physical_intervals=code ? 0 : 1;stats->status=code;return code;
}

static int driver_impl(CMFNodalInput *input,int ns,int nf,int nr,const double *r,const double *v,const double *gx,const double *gw,
    int npath,const int *paths,const double *core_mu,const double *core_weights,int nstep,const double *steps,
    int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double h,int panels,int block_width,double rtol,double atol,double max_seconds,
    uint64_t max_vectors,uint64_t max_interval_vectors,int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats,
    CMFCachedDriverStats *cache_stats,double seconds_cap)
{
    if(!stats)return 1;
    CMFCachedDriverStats unused_cache={0};
    if(cache_stats)memset(cache_stats,0,sizeof(*cache_stats));
    memset(stats,0,sizeof(*stats));int actual_ns,actual_nf;
    if(!input||cmf_nodal_input_shape(input,&actual_ns,&actual_nf)||ns!=actual_ns||nf!=actual_nf||
       ns<24||nf<8||nr<ns||nr>576||!r||!v||!gx||!gw||npath<1||npath>1024||!paths||!core_mu||!core_weights||
       nstep<1||nstep>2000000||!steps||nevent<1||nevent>2000000||!event_ptr||!event_node||!event_weights||!planck||!gradient||!output||
       !isfinite(h)||h<=0||panels<1||panels>16||block_width<1||block_width>65536||
       !isfinite(rtol)||rtol<0||!isfinite(atol)||atol<0||rtol+atol<=0||!isfinite(max_seconds)||max_seconds<=0||max_seconds>seconds_cap||
       max_vectors<1||max_vectors>2000000||max_interval_vectors<1||max_interval_vectors>200000||
       max_leaves<panels||max_leaves>4096||max_depth<0||max_depth>32){stats->status=1;return 1;}
    for(int i=0;i<nr;i++)if(!isfinite(r[i])||r[i]<=0||(i&&r[i]>=r[i-1])||!isfinite(v[i])||v[i]<0||v[i]>=LIGHT){stats->status=1;return 1;}
    double weight_sum=0.;for(int i=0;i<8;i++){
        if(!isfinite(gx[i])||fabs(gx[i])>=1||!isfinite(gw[i])||gw[i]<=0){stats->status=1;return 1;}weight_sum+=gw[i];
    }
    if(fabs(weight_sum-2.)>1e-12||event_ptr[0]!=0||event_ptr[nstep]!=nevent){stats->status=1;return 1;}
    for(int i=0;i<nstep;i++)if(event_ptr[i]<0||event_ptr[i+1]<event_ptr[i]||event_ptr[i+1]>nevent){stats->status=1;return 1;}
    for(int i=0;i<nevent;i++){
        if(event_node[i]<0||event_node[i]>=ns){stats->status=1;return 1;}
        for(int m=0;m<3;m++)if(!isfinite(event_weights[3*i+m])||((m==0||m==2)&&event_weights[3*i+m]<0)){stats->status=1;return 1;}
    }
    for(int f=0;f<nf;f++)if(!isfinite(planck[f])||planck[f]<0||!isfinite(gradient[f])){stats->status=1;return 1;}
    int cursor=0;
    for(int i=0;i<npath;i++){
        const int *p=paths+5*i;
        if(p[0]!=cursor||p[1]<0||p[1]>nstep-cursor||p[2]!=p[0]+p[1]||p[3]<0||p[3]>nstep-p[2]||
           (p[4]!=0&&p[4]!=1)||!isfinite(core_mu[i])||fabs(core_mu[i])>1){stats->status=1;return 1;}
        for(int m=0;m<3;m++)if(!isfinite(core_weights[3*i+m])||((m==0||m==2)&&core_weights[3*i+m]<0)){stats->status=1;return 1;}
        cursor=p[2]+p[3];
    }
    if(cursor!=nstep){stats->status=1;return 1;}
    for(int i=0;i<nstep;i++){
        const double *s=steps+5*i;
        for(int k=0;k<5;k++)if(!isfinite(s[k])){stats->status=1;return 1;}
        if(s[0]<0||s[2]<s[1]||s[3]<s[0]||s[4]<s[0]||s[3]<r[nr-1]||s[4]<r[nr-1]||s[3]>r[0]||s[4]>r[0]||
           fabs(s[1])!=sqrt((s[3]-s[0])*(s[3]+s[0]))||fabs(s[2])!=sqrt((s[4]-s[0])*(s[4]+s[0]))){stats->status=8;return 8;}
    }
    Driver d={.input=input,.ns=ns,.nf=nf,.nr=nr,.panels=panels,.block_width=block_width,.max_leaves=max_leaves,.max_depth=max_depth,
        .r=r,.v=v,.gx=gx,.gw=gw,.h=h,.rtol=rtol,.atol=atol,.started=now(),.max_seconds=max_seconds,
        .max_work=max_vectors*(uint64_t)nf,.max_interval_work=max_interval_vectors*(uint64_t)nf,.stats=stats,
        .cached=cache_stats!=NULL,.cache_stats=cache_stats ? cache_stats : &unused_cache};
    double *ray=allocate(&d,(size_t)2*nf*sizeof(double));if(!ray){stats->status=3;return 3;}
    double *characteristic=ray+nf;int code=0;
    for(int path=0;path<npath&&!code;path++){
        stats->failed_path=path;const int *p=paths+5*path;
        if(p[0]<0||p[1]<0||p[2]<0||p[3]<0||p[0]>nstep-p[1]||p[2]>nstep-p[3]||(p[4]!=0&&p[4]!=1)||
           !isfinite(core_mu[path])||fabs(core_mu[path])>1){code=1;break;}
        memset(ray,0,(size_t)nf*sizeof(double));
        for(int leg=0;leg<2&&!code;leg++){
            if(leg==1&&p[4]){
                for(int f=0;f<nf;f++){ray[f]=planck[f]+core_mu[path]*gradient[f];if(!isfinite(ray[f])||ray[f]<0){code=2;break;}}
                if(code)break;
                code=accumulate(&d,output,ray,ns-1,core_weights+3*path);if(code)break;
            }
            int begin=p[2*leg],count=p[2*leg+1];
            for(int j=begin;j<begin+count;j++){
                stats->failed_interval=j;const double *step=steps+5*j;
                for(int k=0;k<5;k++)if(!isfinite(step[k]))code=1;
                if(code||step[0]<0||step[2]<step[1]||step[3]<=0||step[4]<=0){code=1;break;}
                code=propagate(&d,step,ray,characteristic);if(code)break;stats->physical_intervals++;
                for(int e=event_ptr[j];e<event_ptr[j+1];e++){code=accumulate(&d,output,ray,event_node[e],event_weights+3*e);if(code)break;}
                if(!code&&now()-d.started>=d.max_seconds)code=4;
                if(code)break;
            }
        }
    }
    release(&d,ray);stats->elapsed_seconds=now()-d.started;
    if(!code&&(!isfinite(stats->elapsed_seconds)||stats->elapsed_seconds>=d.max_seconds))code=4;
    stats->status=code;return code;
}

int cmf_nodal_driver(CMFNodalInput *input,int ns,int nf,int nr,const double *r,const double *v,const double *gx,const double *gw,
    int npath,const int *paths,const double *core_mu,const double *core_weights,int nstep,const double *steps,
    int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double h,int panels,int block_width,double rtol,double atol,double max_seconds,
    uint64_t max_vectors,uint64_t max_interval_vectors,int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats)
{
    return driver_impl(input,ns,nf,nr,r,v,gx,gw,npath,paths,core_mu,core_weights,nstep,steps,nevent,event_ptr,event_node,
        event_weights,planck,gradient,h,panels,block_width,rtol,atol,max_seconds,max_vectors,max_interval_vectors,max_leaves,max_depth,output,stats,NULL,120.);
}

int cmf_nodal_driver_cached(CMFNodalInput *input,int ns,int nf,int nr,const double *r,const double *v,const double *gx,const double *gw,
    int npath,const int *paths,const double *core_mu,const double *core_weights,int nstep,const double *steps,
    int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double h,int panels,int block_width,double rtol,double atol,double max_seconds,
    uint64_t max_vectors,uint64_t max_interval_vectors,int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats,CMFCachedDriverStats *cache_stats)
{
    if(!cache_stats)return 1;
    return driver_impl(input,ns,nf,nr,r,v,gx,gw,npath,paths,core_mu,core_weights,nstep,steps,nevent,event_ptr,event_node,
        event_weights,planck,gradient,h,panels,block_width,rtol,atol,max_seconds,max_vectors,max_interval_vectors,max_leaves,max_depth,output,stats,cache_stats,120.);
}

int cmf_nodal_driver_native90(CMFNodalInput *input,int ns,int nf,int nr,const double *r,const double *v,const double *gx,const double *gw,
    int npath,const int *paths,const double *core_mu,const double *core_weights,int nstep,const double *steps,
    int nevent,const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double h,int panels,int block_width,double rtol,double atol,double max_seconds,
    uint64_t max_vectors,uint64_t max_interval_vectors,int max_leaves,int max_depth,double *output,CMFNodalDriverStats *stats,CMFCachedDriverStats *cache_stats)
{
    if(ns!=90||nf!=1115020||!cache_stats){
        if(stats){memset(stats,0,sizeof(*stats));stats->status=1;}
        if(cache_stats)memset(cache_stats,0,sizeof(*cache_stats));
        return 1;
    }
    return driver_impl(input,ns,nf,nr,r,v,gx,gw,npath,paths,core_mu,core_weights,nstep,steps,nevent,event_ptr,event_node,
        event_weights,planck,gradient,h,panels,block_width,rtol,atol,max_seconds,max_vectors,max_interval_vectors,max_leaves,max_depth,
        output,stats,cache_stats,1500.);
}
