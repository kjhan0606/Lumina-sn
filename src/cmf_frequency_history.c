/* 주파수 이력의 양의 거리 모멘트 합성 후보. 기존 raw/P0 코드는 보존한다. */
#include "cmf_frequency_history.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#ifdef CMF_EXPQUAD_ENGINE
#include "cmf_expquad_engine.h"
#endif

/* 성능 경로의 보수적인 선택일 뿐 물리값의 floor/clamp가 아니다.
 * 범위 밖이면 압축 선형맵을 쓰지 않고 실제 입사의 원래 kernel로 돌아간다. */
static int affine_moderate(double x)
{
    return isfinite(x)&&(x==0.||(x>=1e-80&&x<=1e80));
}

static CMFHistoryMap identity(void)
{
    CMFHistoryMap m={0.,1.,0.,0.,0.,0.};return m;
}

static int choose(int n,int k)
{
    int value=1;for(int i=1;i<=k;i++)value=value*(n+1-i)/i;return value;
}

int cmf_history_reconstruct(int nf,double shift,const double *input,double *out,uint64_t *limited)
{
    if(nf<8||nf>2000000||!isfinite(shift)||shift<0||!input||!out||out==input||!limited)return 1;
    *limited=0;
    for(int i=0;i<nf;i++)if(!isfinite(input[i])||input[i]<0)return 1;
    if(shift>=nf-1){for(int i=0;i<nf;i++)out[i]=input[nf-1];return 0;}
    int whole=(int)floor(shift);double fraction=shift-whole;
    if(fraction==0){for(int i=0;i<nf;i++)out[i]=input[i+whole<nf ? i+whole : nf-1];return 0;}
    double bank[7][8][8];
    for(int offset=-6;offset<=0;offset++)for(int j=0;j<8;j++){
        double power[8]={1.,0.,0.,0.,0.,0.,0.,0.},denominator=1.;int degree=0;
        for(int k=0;k<8;k++)if(k!=j){
            double next[8]={0.};double root=offset+k;
            for(int m=0;m<=degree;m++){next[m]-=root*power[m];next[m+1]+=power[m];}
            memcpy(power,next,sizeof(power));degree++;denominator*=j-k;
        }
        for(int m=0;m<8;m++)power[m]/=denominator;
        for(int k=0;k<8;k++){
            double value=0.;for(int m=0;m<=k;m++)value+=(double)choose(k,m)/choose(7,m)*power[m];
            bank[offset+6][k][j]=value;
        }
    }
    for(int i=0;i<nf;i++){
        int left=i+whole;
        if(left>=nf-1){out[i]=input[nf-1];continue;}
        int first=left-3;if(first<0)first=0;if(first>nf-8)first=nf-8;
        double b[8],low[8],base=input[left],difference=input[left+1]-base,theta=1.;
        for(int k=0;k<8;k++){
            double value=0.;for(int j=0;j<8;j++)value+=bank[first-left+6][k][j]*(input[first+j]-base);
            b[k]=base+value;low[k]=base+(double)k/7*difference;
        }
        b[0]=input[left];b[7]=input[left+1];
        for(int k=0;k<8;k++){
            if(!isfinite(b[k]))return 2;
            if(b[k]<0){double ratio=low[k]/(low[k]-b[k]);if(ratio<theta)theta=ratio;}
        }
        if(theta<1){
            int fallback=0;theta=nextafter(theta,0.);(*limited)++;
            for(int k=0;k<8;k++){b[k]=theta*b[k]+(1-theta)*low[k];if(b[k]<0)fallback=1;}
            if(fallback)memcpy(b,low,sizeof(b));
        }
        for(int degree=7;degree>0;degree--)for(int k=0;k<degree;k++)b[k]=(1-fraction)*b[k]+fraction*b[k+1];
        out[i]=b[0];if(!isfinite(out[i])||out[i]<0)return 2;
    }
    return 0;
}

#include "cmf_positive_moments.h"

int cmf_history_compose(CMFHistoryMap l,CMFHistoryMap r,CMFHistoryMap *out)
{
    if(!out||!valid(l)||!valid(r))return 1;
    CMFHistoryMap m;
    m.d=l.d+r.d;m.t=r.t*l.t;
    m.aa=r.t*l.aa+r.aa;m.ab=r.t*l.ab+r.ab;
    m.pa=r.t*(l.pa+r.d*l.aa)+r.pa;
    m.mb=r.t*l.mb+r.mb+l.d*r.ab;
    if(!valid(m))return 2;
    *out=m;return 0;
}

int cmf_history_map(double length,double normalization,double k0,double k1,
    double ea0,double ea1,double eb0,double eb1,CMFHistoryMap *out,CMFHistoryStats *stats)
{
    return lumina_positive_map(length,normalization,k0,k1,ea0,ea1,eb0,eb1,out,stats);
}

typedef struct {CMFHistoryMap value,aggregate;} Node;
typedef struct {Node *front,*back;size_t f,b,capacity;CMFHistoryStats *stats;} Window;

static int compose(CMFHistoryMap a,CMFHistoryMap b,CMFHistoryMap *out,CMFHistoryStats *s)
{
    s->compositions++;return cmf_history_compose(a,b,out);
}

static int push(Window *w,CMFHistoryMap value)
{
    if(w->f+w->b>=w->capacity)return 3;
    Node *n=&w->back[w->b];n->value=value;n->aggregate=value;
    if(w->b&&compose(w->back[w->b-1].aggregate,value,&n->aggregate,w->stats))return 2;
    w->b++;w->stats->pushes++;
    if(w->f+w->b>w->stats->queue_peak)w->stats->queue_peak=w->f+w->b;
    return 0;
}

static int pop(Window *w)
{
    if(!w->f){
        while(w->b){
            CMFHistoryMap value=w->back[--w->b].value;Node *n=&w->front[w->f];
            n->value=value;n->aggregate=value;
            if(w->f&&compose(value,w->front[w->f-1].aggregate,&n->aggregate,w->stats))return 2;
            w->f++;
        }
        w->stats->transfers++;
    }
    if(!w->f)return 2;
    w->f--;return 0;
}

static int aggregate(Window *w,CMFHistoryMap *out)
{
    if(!w->f&&!w->b)*out=identity();
    else if(!w->f)*out=w->back[w->b-1].aggregate;
    else if(!w->b)*out=w->front[w->f-1].aggregate;
    else return compose(w->front[w->f-1].aggregate,w->back[w->b-1].aggregate,out,w->stats);
    return 0;
}

static double interpolate(const double *v,int j,double x)
{
    return x==0 ? v[j] : x==1 ? v[j+1] : v[j]+x*(v[j+1]-v[j]);
}

typedef struct {
    int nf;double length,span,a;const double *chi,*chi_b,*ea,*eb;CMFHistoryStats *stats;
    CMFCoefficientProvider provider;void *user;int chi_point,eta_a_point,eta_b_point;
    int cache_capacity,*cache_columns;double *cache_values;unsigned char *affine_safe;
} Context;

static int start_cache(Context *c)
{
    int size=c->span>=8188. ? 8192 : (int)ceil(c->span)+4;
    if(size<256)size=256;
    if(size>c->nf)size=c->nf;
    c->cache_capacity=size;c->cache_columns=malloc((size_t)size*sizeof(int));
    c->cache_values=malloc((size_t)size*4*sizeof(double));
    if(!c->cache_columns||!c->cache_values){free(c->cache_columns);free(c->cache_values);return 3;}
    for(int i=0;i<size;i++)c->cache_columns[i]=-1;
    return 0;
}

static void stop_cache(Context *c)
{
    free(c->cache_columns);free(c->cache_values);
}

static int cached_coefficients(Context *c,int column,int small,double *out)
{
    int slot=column%c->cache_capacity;double *value=c->cache_values+4*slot;
    if(c->cache_columns[slot]!=column){
        double next[4]={0.},ignored;
        if(small){
            if(c->provider(c->user,c->eta_a_point,column,next,next+1)||
               c->provider(c->user,c->eta_b_point,column,next+2,next+3))return 2;
        }else{
            if(c->provider(c->user,c->chi_point,column,next,&ignored)||
               c->provider(c->user,c->eta_a_point,column,&ignored,next+1)||
               c->provider(c->user,c->eta_b_point,column,&ignored,next+2))return 2;
        }
        for(int i=0;i<4;i++)if(!isfinite(next[i]))return 2;
        if(next[1]<0||(small ? next[3] : next[2])<0)return 2;
        memcpy(value,next,sizeof(next));c->cache_columns[slot]=column;
    }
    memcpy(out,value,4*sizeof(double));return 0;
}

static int provided_coefficients(Context *c,int j,double fraction,int small,double *out)
{
    if(fraction==1.)return cached_coefficients(c,j+1,small,out);
    if(cached_coefficients(c,j,small,out))return 2;
    if(fraction!=0.){
        double last[4];if(cached_coefficients(c,j+1,small,last))return 2;
        for(int i=0;i<4;i++)out[i]=out[i]+fraction*(last[i]-out[i]);
    }
    return 0;
}

static int coefficients(Context *c,int j,double fraction,double *chi,double *ea,double *eb)
{
    if(!c->provider){
        *chi=interpolate(c->chi,j,fraction);*ea=interpolate(c->ea,j,fraction);*eb=interpolate(c->eb,j,fraction);return 0;
    }
    double values[4];if(provided_coefficients(c,j,fraction,0,values))return 2;
    *chi=values[0];*ea=values[1];*eb=values[2];
    return !isfinite(*chi)||!isfinite(*ea)||*ea<0||!isfinite(*eb)||*eb<0 ? 2 : 0;
}

static int bin_map(Context *c,int j,double upper,double lower,double fraction,CMFHistoryMap *out)
{
    double c0,c1,a0,a1,b0,b1;
    if(coefficients(c,j,upper,&c0,&a0,&b0)||coefficients(c,j,lower,&c1,&a1,&b1))return 2;
    return cmf_history_map(c->length*fraction,c->length,
        c0+3*c->a,c1+3*c->a,a0,a1,b0,b1,out,c->stats);
}

static int blue_map(Context *c,double fraction,CMFHistoryMap *out)
{
    int i=c->nf-1;c->stats->blue_maps++;
    double chi,ea,eb;
    if(c->provider){if(coefficients(c,i,0.,&chi,&ea,&eb))return 2;}
    else{chi=c->chi[i];ea=c->ea[i];eb=c->eb[i];}
    return cmf_history_map(c->length*fraction,c->length,chi+3*c->a,chi+3*c->a,ea,ea,eb,eb,out,c->stats);
}

static int history_sweep(Context c,int begin,int count,double offset,const double *intensity,
    double *out,int slab_width,int naive,double *response)
{
    int nf=c.nf;double span=c.span;CMFHistoryStats *stats=c.stats;
    if(c.length==0){
        memcpy(out,intensity,(size_t)count*sizeof(double));
        if(response)for(int i=0;i<count;i++)response[i]=1.;
        return 0;
    }
    if(!isfinite(c.a))return 2;
    if(span==0){
        for(int local=0;local<count;local++){
            int i=begin+local;
            double q=i+offset;CMFHistoryMap m;
            if(q>=nf-1){if(blue_map(&c,1.,&m))return 2;}
            else{
                int j=(int)floor(q);double f=q-j;
                if(bin_map(&c,j,f,f,1.,&m))return 2;
                stats->partial_bin_maps++;
            }
            out[local]=m.t*intensity[local]+m.pa+m.mb;
            if(response)response[local]=m.t;
            if(!isfinite(out[local])||out[local]<0)return 2;
        }
        stats->slabs=((uint64_t)count+slab_width-1)/slab_width;return 0;
    }
    size_t capacity=span>=nf-1 ? (size_t)(nf-1) : (size_t)ceil(span)+1;
    Window w={0};w.capacity=capacity;w.stats=stats;
    if(capacity&&!naive){
        w.front=malloc(2*capacity*sizeof(Node));if(!w.front)return 3;
        w.back=w.front+capacity;stats->queue_bytes=2*capacity*sizeof(Node);
    }
    int previous_low=0,previous_high=-1,status=0;
    for(int local=count-1;local>=0;local--){
        int i=begin+local;
        if((count-1-local)%slab_width==0)stats->slabs++;
        double end=i+offset,start=end+span,blue=(double)(nf-1);
        CMFHistoryMap whole=identity(),part;
        if(end>=blue){if(blue_map(&c,1.,&whole)){status=2;break;}}
        else{
            double top=fmin(start,blue),blue_fraction=start>blue ? (start-blue)/span : 0.;
            if(blue_fraction>0){if(blue_map(&c,blue_fraction,&whole)){status=2;break;}}
            int end_bin=(int)floor(end),top_bin=(int)floor(top);
            double fb=end-end_bin,fa=top-top_bin;
            if(end_bin==top_bin){
                if(bin_map(&c,end_bin,fa,fb,1.-blue_fraction,&part)||
                   compose(whole,part,&whole,stats)){status=2;break;}
                stats->partial_bin_maps++;
            }else{
                if(fa>0){
                    if(bin_map(&c,top_bin,fa,0.,fa/span,&part)||
                       compose(whole,part,&whole,stats)){status=2;break;}
                    stats->partial_bin_maps++;
                }
                int low=end_bin+(fb>0),high=top_bin-1;
                if(low<=high){
                    if(naive){
                        part=identity();
                        for(int j=high;j>=low;j--){
                            CMFHistoryMap item;
                            if(bin_map(&c,j,1.,0.,1./span,&item)||compose(part,item,&part,stats)){status=2;break;}
                            stats->full_bin_maps++;
                        }
                    }else{
                        while(w.f+w.b&&previous_high>high){if(pop(&w)){status=2;break;}previous_high--;}
                        if(!w.f&&!w.b){previous_high=high;previous_low=high+1;}
                        while(!status&&previous_low>low){
                            CMFHistoryMap item;previous_low--;
                            if(bin_map(&c,previous_low,1.,0.,1./span,&item)||push(&w,item)){status=2;break;}
                            stats->full_bin_maps++;
                        }
                        if(!status&&aggregate(&w,&part))status=2;
                    }
                    if(status||compose(whole,part,&whole,stats)){status=2;break;}
                }
                if(fb>0){
                    if(bin_map(&c,end_bin,1.,fb,(1.-fb)/span,&part)||
                       compose(whole,part,&whole,stats)){status=2;break;}
                    stats->partial_bin_maps++;
                }
            }
        }
        out[local]=whole.t*intensity[local]+whole.pa+whole.mb;
        if(response)response[local]=whole.t;
        if(!isfinite(out[local])||out[local]<0){status=2;break;}
    }
    free(w.front);return status;
}

static int history_impl(int nf,double h,double length,double delta_a,double delta_b,
    const double *chi,const double *ea,const double *eb,const double *intensity,
    double *out,int slab_width,int naive,CMFHistoryStats *stats,double *response)
{
    if(!stats)return 1;
    memset(stats,0,sizeof(*stats));
    if(nf<1||nf>2000000||!isfinite(h)||h<=0||!isfinite(length)||length<0||
       !isfinite(delta_a)||!isfinite(delta_b)||delta_b<0||delta_a<delta_b||
       !chi||!ea||!eb||!intensity||!out||out==intensity||slab_width<1||slab_width>65536||
       (naive!=0&&naive!=1)||(naive&&nf>4096)||(length==0&&delta_a!=delta_b))return 1;
    for(int i=0;i<nf;i++)if(!isfinite(chi[i])||!isfinite(ea[i])||ea[i]<0||
        !isfinite(eb[i])||eb[i]<0||!isfinite(intensity[i])||intensity[i]<0)return 1;
    double span=(delta_a-delta_b)/h,offset=delta_b/h;
    if(!isfinite(span)||!isfinite(offset)||(naive&&span>4096))return 1;
    Context c={.nf=nf,.length=length,.span=span,.a=length ? (delta_a-delta_b)/length : 0.,
        .chi=chi,.ea=ea,.eb=eb,.stats=stats};
    return history_sweep(c,0,nf,offset,intensity,out,slab_width,naive,response);
}

int cmf_frequency_history_provider(int nf,int begin,int count,double h,double length,double da,double db,
    CMFCoefficientProvider provider,void *user,int cp,int ap,int bp,
    const double *intensity,double *out,double *response,CMFHistoryStats *stats)
{
    if(!stats)return 1;
    memset(stats,0,sizeof(*stats));
    if(nf<1||nf>2000000||count<1||count>nf||begin<0||begin>nf-count||!isfinite(h)||h<=0||
        !isfinite(length)||length<0||!isfinite(da)||!isfinite(db)||db<0||da<db||
        (length==0&&da!=db)||!provider||!user||cp<0||ap<0||bp<0||!intensity||!out||out==intensity||!response)return 1;
    for(int i=0;i<count;i++)if(!isfinite(intensity[i])||intensity[i]<0)return 1;
    double span=(da-db)/h,offset=db/h;if(!isfinite(span)||!isfinite(offset))return 1;
    Context c={.nf=nf,.length=length,.span=span,.a=length ? (da-db)/length : 0.,.stats=stats,
        .provider=provider,.user=user,.chi_point=cp,.eta_a_point=ap,.eta_b_point=bp};
    int code=start_cache(&c);if(code)return code;
    code=history_sweep(c,begin,count,offset,intensity,out,256,0,response);stop_cache(&c);return code;
}

int cmf_frequency_history(int nf,double h,double length,double delta_a,double delta_b,
    const double *chi,const double *ea,const double *eb,const double *intensity,
    double *out,int slab_width,int naive,CMFHistoryStats *stats)
{
    return history_impl(nf,h,length,delta_a,delta_b,chi,ea,eb,intensity,out,slab_width,naive,stats,NULL);
}

int cmf_frequency_history_response(int nf,double h,double length,double delta_a,double delta_b,
    const double *chi,const double *ea,const double *eb,const double *intensity,
    double *out,int slab_width,int naive,double *response,CMFHistoryStats *stats)
{
    if(!response)return 1;
    return history_impl(nf,h,length,delta_a,delta_b,chi,ea,eb,intensity,out,slab_width,naive,stats,response);
}

static void add_stats(CMFHistoryStats *total,const CMFHistoryStats *part)
{
    total->full_bin_maps+=part->full_bin_maps;total->partial_bin_maps+=part->partial_bin_maps;
    total->blue_maps+=part->blue_maps;total->compositions+=part->compositions;
    total->pushes+=part->pushes;total->transfers+=part->transfers;
    if(part->queue_peak>total->queue_peak)total->queue_peak=part->queue_peak;
    if(part->queue_bytes>total->queue_bytes)total->queue_bytes=part->queue_bytes;
    total->quadrature_nodes+=part->quadrature_nodes;total->quadrature_panels+=part->quadrature_panels;
    total->slabs+=part->slabs;
}

int cmf_frequency_history_compare(int count,int nf,double h,const double *lengths,
    const double *deltas,const double *chi,const double *eta,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats)
{
    if(count<1||count>4096||nf<1||nf>2000000||(uint64_t)count*nf>2097152||
       !lengths||!deltas||!chi||!eta||!initial||!out||out==initial||!errors||!responses||!stats)return 1;
    double *work=malloc(3*(size_t)nf*sizeof(double));if(!work)return 3;
    double *coarse=work,*first=work+nf,*left_response=work+2*nf;
    memcpy(out,initial,(size_t)nf*sizeof(double));memset(stats,0,sizeof(*stats));int code=0;
    for(int row=0;row<count;row++){
        size_t offset=(size_t)row*3*nf;CMFHistoryStats part;
        code=history_impl(nf,h,lengths[3*row],deltas[3*row],deltas[3*row+2],
            chi+offset,eta+offset,eta+offset+2*nf,out,coarse,256,0,&part,NULL);
        if(code)break;
        add_stats(stats,&part);
        code=history_impl(nf,h,lengths[3*row+1],deltas[3*row],deltas[3*row+1],
            chi+offset+nf,eta+offset,eta+offset+nf,out,first,256,0,&part,left_response);
        if(code)break;
        add_stats(stats,&part);
        code=history_impl(nf,h,lengths[3*row+2],deltas[3*row+1],deltas[3*row+2],
            chi+offset+2*nf,eta+offset+nf,eta+offset+2*nf,first,out,256,0,&part,responses+(size_t)row*nf);
        if(code)break;
        add_stats(stats,&part);
        for(int f=0;f<nf;f++){
            errors[(size_t)row*nf+f]=fabs(out[f]-coarse[f]);
            responses[(size_t)row*nf+f]*=left_response[f];
            if(!isfinite(responses[(size_t)row*nf+f])){code=2;break;}
        }
        if(code)break;
    }
    free(work);return code;
}

static double sample(const double *v,int nf,double x)
{
    if(x>=nf-1)return v[nf-1];
    int j=(int)floor(x);return interpolate(v,j,x-j);
}

static int small_coefficients(Context *c,int last,double column,double *chi,double *eta)
{
    if(c->provider){
        int j=column>=c->nf-1 ? c->nf-1 : (int)floor(column);
        double fraction=column>=c->nf-1 ? 0. : column-j,values[4];
        if(provided_coefficients(c,j,fraction,1,values))return 2;
        *chi=values[last ? 2 : 0];*eta=values[last ? 3 : 1];
        return !isfinite(*chi)||!isfinite(*eta)||*eta<0 ? 2 : 0;
    }
    *chi=sample(last ? c->chi_b : c->chi,c->nf,column);
    *eta=sample(last ? c->eb : c->ea,c->nf,column);return 0;
}

static int small_sweep(Context c,int begin,int count,double offset,const double *initial,double *out,double *response)
{
    int nf=c.nf;double length=c.length,span=c.span;CMFHistoryStats *stats=c.stats;
    if(length==0){
        memcpy(out,initial,(size_t)count*sizeof(double));
        for(int i=0;i<count;i++)response[i]=1.;
        return 0;
    }
    double gamma=3.*c.a;
    if(!isfinite(gamma))return 2;
    for(int local=0;local<count;local++){
        int i=begin+local;
        double end=i+offset,start=end+span,top=start,t0=0.,value=initial[local],transmission=1.;
        int pieces=0;
        do{
            double bottom=span==0 ? end : top>nf-1 ? fmax(end,nf-1.) : fmax(end,ceil(top)-1.);
            double t1=span==0||bottom==end ? 1. : (start-bottom)/span;
            if(++pieces>4||t1<=t0)return 2;
            double ca0,cb0,ea0,eb0,ca1,cb1,ea1,eb1;
            if(small_coefficients(&c,0,top,&ca0,&ea0)||small_coefficients(&c,1,top,&cb0,&eb0)||
               small_coefficients(&c,0,bottom,&ca1,&ea1)||small_coefficients(&c,1,bottom,&cb1,&eb1))return 2;
            double c0=(1-t0)*ca0+t0*cb0+gamma,c1=(1-t1)*ca1+t1*cb1+gamma;
            double e0=(1-t0)*ea0+t0*eb0,e1=(1-t1)*ea1+t1*eb1;
            double piece_length=length*(t1-t0);CMFHistoryMap m;
            if(cmf_history_map(piece_length,piece_length,c0,c1,e0,e1,e0,e1,&m,stats))return 2;
            value=m.t*value+m.pa+m.mb;transmission*=m.t;stats->partial_bin_maps++;
            if(!isfinite(value)||value<0||!isfinite(transmission))return 2;
            if(c.affine_safe&&(!affine_moderate(m.t)||m.t==0.||
               !affine_moderate(m.pa)||!affine_moderate(m.mb)||
               !affine_moderate(value)||!affine_moderate(transmission)||transmission==0.||
               ((e0>0.||e1>0.)&&m.pa==0.&&m.mb==0.)))c.affine_safe[local]=0;
            top=bottom;t0=t1;
        }while(top>end);
        out[local]=value;response[local]=transmission;
    }
    return 0;
}

int cmf_frequency_small_spatial(int nf,double h,double length,double da,double db,
    const double *ca,const double *cb,const double *ea,const double *eb,
    const double *initial,double *out,double *response,CMFHistoryStats *stats)
{
    if(!stats)return 1;
    memset(stats,0,sizeof(*stats));
    if(nf<1||nf>2000000||!isfinite(h)||h<=0||!isfinite(length)||length<0||
       !isfinite(da)||!isfinite(db)||db<0||da<db||!ca||!cb||!ea||!eb||
       !initial||!out||out==initial||!response||(length==0&&da!=db))return 1;
    double span=(da-db)/h,offset=db/h;
    if(!isfinite(span)||!isfinite(offset)||span>2.)return 1;
    for(int i=0;i<nf;i++)if(!isfinite(ca[i])||!isfinite(cb[i])||
        !isfinite(ea[i])||ea[i]<0||!isfinite(eb[i])||eb[i]<0||
        !isfinite(initial[i])||initial[i]<0)return 1;
    Context c={.nf=nf,.length=length,.span=span,.a=length ? (da-db)/length : 0.,
        .chi=ca,.chi_b=cb,.ea=ea,.eb=eb,.stats=stats};
    return small_sweep(c,0,nf,offset,initial,out,response);
}

static int small_provider_impl(int nf,int begin,int count,double h,double length,double da,double db,
    CMFCoefficientProvider provider,void *user,int first,int last,
    const double *initial,double *out,double *response,CMFHistoryStats *stats,unsigned char *affine_safe)
{
    if(!stats)return 1;
    memset(stats,0,sizeof(*stats));
    if(nf<1||nf>2000000||count<1||count>nf||begin<0||begin>nf-count||!isfinite(h)||h<=0||
        !isfinite(length)||length<0||!isfinite(da)||!isfinite(db)||db<0||da<db||
        (length==0&&da!=db)||!provider||!user||first<0||last<0||!initial||!out||out==initial||!response)return 1;
    for(int i=0;i<count;i++)if(!isfinite(initial[i])||initial[i]<0)return 1;
    double span=(da-db)/h,offset=db/h;if(!isfinite(span)||!isfinite(offset)||span>2.)return 1;
    Context c={.nf=nf,.length=length,.span=span,.a=length ? (da-db)/length : 0.,.stats=stats,
        .provider=provider,.user=user,.eta_a_point=first,.eta_b_point=last,.affine_safe=affine_safe};
    int code=start_cache(&c);if(code)return code;
    code=small_sweep(c,begin,count,offset,initial,out,response);stop_cache(&c);return code;
}

int cmf_frequency_small_provider(int nf,int begin,int count,double h,double length,double da,double db,
    CMFCoefficientProvider provider,void *user,int first,int last,
    const double *initial,double *out,double *response,CMFHistoryStats *stats)
{
    return small_provider_impl(nf,begin,count,h,length,da,db,provider,user,first,last,
        initial,out,response,stats,NULL);
}

typedef struct {
    int nf,begin,count;const double *ca,*cm,*cb,*ea,*eb;
    CMFCoefficientProvider provider;void *user;int first,middle,last;unsigned char *affine_safe;
} HybridSource;

static int hybrid_call(HybridSource s,int small,double h,double length,double da,double db,
    const double *initial,double *out,double *response,CMFHistoryStats *stats)
{
    if(s.provider){
        if(small)return small_provider_impl(s.nf,s.begin,s.count,h,length,da,db,s.provider,s.user,
            s.first,s.last,initial,out,response,stats,s.affine_safe);
        return cmf_frequency_history_provider(s.nf,s.begin,s.count,h,length,da,db,s.provider,s.user,
            s.middle,s.first,s.last,initial,out,response,stats);
    }
    if(small)return cmf_frequency_small_spatial(s.nf,h,length,da,db,s.ca,s.cb,s.ea,s.eb,initial,out,response,stats);
    return history_impl(s.nf,h,length,da,db,s.cm,s.ea,s.eb,initial,out,256,0,stats,response);
}

static int hybrid_piece(HybridSource source,double h,double length,double da,double db,
    const double *initial,double *out,double *response,double *difference,
    double *alternate,double *alternate_response,CMFHistoryStats *total,uint64_t *calls,uint64_t *checks)
{
    int nf=source.count;double span=(da-db)/h;CMFHistoryStats part;
    int code=hybrid_call(source,span<=1.,h,length,da,db,initial,out,response,&part);
    (*calls)++;if(code)return code;add_stats(total,&part);
    memset(difference,0,(size_t)nf*sizeof(double));
    if(span>=.75&&span<=1.25){
        code=hybrid_call(source,span>1.,h,length,da,db,initial,alternate,alternate_response,&part);
        (*calls)++;(*checks)++;if(code)return code;add_stats(total,&part);
        for(int f=0;f<nf;f++)difference[f]=fabs(out[f]-alternate[f]);
    }
    return 0;
}

static int hybrid_compare(int count,HybridSource base,double h,const double *lengths,
    const double *deltas,const double *chi,const double *eta,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks)
{
    int nf=base.count;
    double *work=malloc(9*(size_t)nf*sizeof(double));if(!work)return 3;
    double *coarse=work,*first=work+nf,*left_response=work+2*nf,*coarse_response=work+3*nf;
    double *coarse_diff=work+4*nf,*left_diff=work+5*nf,*right_diff=work+6*nf;
    double *alternate=work+7*nf,*alternate_response=work+8*nf;
    memcpy(out,initial,(size_t)nf*sizeof(double));memset(stats,0,sizeof(*stats));*calls=*checks=0;int code=0;
    for(int row=0;row<count;row++){
        HybridSource whole=base,left=base,right=base;
        if(base.provider){
            whole.first=left.first=5*row;whole.middle=left.last=right.first=5*row+2;
            whole.last=right.last=5*row+4;left.middle=5*row+1;right.middle=5*row+3;
        }else{
            const double *c=chi+(size_t)row*5*nf,*e=eta+(size_t)row*3*nf;
            whole.ca=left.ca=c;whole.cm=left.cb=right.ca=c+2*nf;whole.cb=right.cb=c+4*nf;
            left.cm=c+nf;right.cm=c+3*nf;
            whole.ea=left.ea=e;whole.eb=right.eb=e+2*nf;left.eb=right.ea=e+nf;
        }
        const double *l=lengths+3*row,*d=deltas+3*row;
        code=hybrid_piece(whole,h,l[0],d[0],d[2],out,coarse,
            coarse_response,coarse_diff,alternate,alternate_response,stats,calls,checks);
        if(code)break;
        code=hybrid_piece(left,h,l[1],d[0],d[1],out,first,
            left_response,left_diff,alternate,alternate_response,stats,calls,checks);
        if(code)break;
        double *response=responses+(size_t)row*nf;
        code=hybrid_piece(right,h,l[2],d[1],d[2],first,out,
            response,right_diff,alternate,alternate_response,stats,calls,checks);
        if(code)break;
        for(int f=0;f<nf;f++){
            errors[(size_t)row*nf+f]=fabs(out[f]-coarse[f])+coarse_diff[f]+response[f]*left_diff[f]+right_diff[f];
            response[f]*=left_response[f];
            if(!isfinite(response[f])||!isfinite(errors[(size_t)row*nf+f])){code=2;break;}
        }
        if(code)break;
    }
    free(work);return code;
}

int cmf_frequency_hybrid_compare(int count,int nf,double h,const double *lengths,
    const double *deltas,const double *chi,const double *eta,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks)
{
    if(count<1||count>4096||nf<1||nf>2000000||(uint64_t)count*nf>2097152||
       !isfinite(h)||h<=0||!lengths||!deltas||!chi||!eta||!initial||!out||out==initial||
       !errors||!responses||!stats||!calls||!checks)return 1;
    HybridSource source={.nf=nf,.count=nf};
    return hybrid_compare(count,source,h,lengths,deltas,chi,eta,initial,out,errors,responses,stats,calls,checks);
}

int cmf_frequency_hybrid_provider_compare(int nf,int begin,int width,int count,double h,
    const double *lengths,const double *deltas,CMFCoefficientProvider provider,void *user,
    const double *initial,double *out,double *errors,double *responses,CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks)
{
    if(count<1||count>4096||nf<1||nf>2000000||width<1||width>65536||width>nf||begin<0||begin>nf-width||
       (uint64_t)count*width>2097152||!isfinite(h)||h<=0||!lengths||!deltas||!provider||!user||
       !initial||!out||out==initial||!errors||!responses||!stats||!calls||!checks)return 1;
    HybridSource source={.nf=nf,.begin=begin,.count=width,.provider=provider,.user=user};
    return hybrid_compare(count,source,h,lengths,deltas,NULL,NULL,initial,out,errors,responses,stats,calls,checks);
}

int cmf_frequency_hybrid_provider_maps(int nf,int begin,int width,int count,double h,
    const double *lengths,const double *deltas,CMFCoefficientProvider provider,void *user,
    double *maps,unsigned char *safe,CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks)
{
    if(count<1||count>4096||nf<1||nf>2000000||width<1||width>65536||width>nf||begin<0||begin>nf-width||
       (uint64_t)count*width>2097152||!isfinite(h)||h<=0||!lengths||!deltas||!provider||!user||
       !maps||!safe||!stats||!calls||!checks)return 1;
    double *zero=calloc((size_t)width,sizeof(double));if(!zero)return 3;
    memset(stats,0,sizeof(*stats));*calls=*checks=0;int code=0;
    for(int row=0;row<count&&!code;row++){
        memset(safe+(size_t)row*width,1,(size_t)width);
        const double *l=lengths+3*row,*d=deltas+3*row;
        for(int piece=0;piece<3&&!code;piece++){
            HybridSource s={.nf=nf,.begin=begin,.count=width,.provider=provider,.user=user,.affine_safe=safe+(size_t)row*width};
            s.first=5*row+(piece==2 ? 2 : 0);s.middle=5*row+(piece==0 ? 2 : piece==1 ? 1 : 3);
            s.last=5*row+(piece==1 ? 2 : 4);
            double da=d[piece==2 ? 1 : 0],db=d[piece==1 ? 1 : 2],span=(da-db)/h;
            double *t=maps+((size_t)row*12+piece*4)*width,*b=t+width,*ta=b+width,*ba=ta+width;
            CMFHistoryStats part;
            code=hybrid_call(s,span<=1.,h,l[piece],da,db,zero,b,t,&part);(*calls)++;
            if(code)break;
            add_stats(stats,&part);
            if(span>=.75&&span<=1.25){
                code=hybrid_call(s,span>1.,h,l[piece],da,db,zero,ba,ta,&part);(*calls)++;(*checks)++;
                if(code)break;
                add_stats(stats,&part);
            }else{memcpy(ta,t,(size_t)width*sizeof(double));memcpy(ba,b,(size_t)width*sizeof(double));}
            for(int f=0;f<width;f++)if(!affine_moderate(t[f])||t[f]==0.||!affine_moderate(b[f])||
                !affine_moderate(ta[f])||ta[f]==0.||!affine_moderate(ba[f]))safe[(size_t)row*width+f]=0;
        }
    }
    free(zero);return code;
}
