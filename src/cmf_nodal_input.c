/* 읽기 전용 원래점 계수 cache. 생산 수송 실행기는 별도다. */
#define _POSIX_C_SOURCE 200809L
#include "cmf_nodal_input.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {int row,slab,next;uint64_t age;double *values;} Row;
typedef struct {int column;uint64_t age;double sign,*values;} Profile;
struct CMFNodalInput {
    int fd,ns,nf,added,reference_b,width,capacity,np,nbuckets,start,count;
    uint64_t base,clock;double minimum,*r,*ext,*scratch,*storage,*profile_storage,*j;
    const double *jview;int zero_j;
    int *buckets;Row *rows;Profile *profiles;CMFNodalInputStats stats;
    double *dense,*dense_profile;CMFNodalDenseStats dense_stats;
};

static int exact_read(int fd,void *target,size_t bytes,uint64_t offset,CMFNodalInputStats *s)
{
    unsigned char *p=target;size_t done=0;
    if(offset>INT64_MAX||bytes>INT64_MAX-offset)return 1;
    while(done<bytes){
        ssize_t n=pread(fd,p+done,bytes-done,(off_t)(offset+done));
        if(s)s->source_preads++;
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return 1;
        done+=(size_t)n;if(s)s->source_bytes+=(uint64_t)n;
    }
    return 0;
}

void cmf_nodal_input_destroy(CMFNodalInput *c)
{
    if(!c)return;
    if(c->fd>=0)close(c->fd);
    free(c->r);free(c->ext);free(c->scratch);free(c->storage);free(c->profile_storage);
    free(c->buckets);free(c->rows);free(c->profiles);free(c->j);
    free(c->dense);free(c->dense_profile);free(c);
}

CMFNodalInput *cmf_nodal_input_create(int fd,int ns,int nf,int added,int reference_b,
    const double *r,const double *ext,double minimum,int width,int capacity,int np,int *code)
{
    if(!code)return NULL;
    *code=1;uint16_t endian=1;
    if(sizeof(double)!=8||DBL_MANT_DIG!=53||*(unsigned char *)&endian!=1||
       fd<0||ns<24||ns>512||nf<2||nf>2000000||added<1||added>64||reference_b<1||reference_b>=ns||
       !r||!ext||!isfinite(minimum)||minimum<=0||width<1||width>8192||capacity<2||capacity>512||
       np<1||np>4096||(uint64_t)capacity*3*width*8>67108864||
       (fcntl(fd,F_GETFL)&O_ACCMODE)!=O_RDONLY)return NULL;
    for(int i=0;i<ns;i++)if(!isfinite(r[i])||r[i]<=0||(i&&r[i]>=r[i-1]))return NULL;
    for(int i=0;i<=added;i++)if(!isfinite(ext[i])||ext[i]<=0||(i&&ext[i]>=ext[i-1]))return NULL;
    if(ext[added]!=r[0])return NULL;
    unsigned char header[16];struct stat info;
    if(fstat(fd,&info)||exact_read(fd,header,16,0,NULL))return NULL;
    uint32_t hn,hf;memcpy(&hn,header+8,4);memcpy(&hf,header+12,4);
    uint64_t base=16+8*((uint64_t)nf+2*(uint64_t)ns);
    if(memcmp(header,"LMPHYS01",8)||hn!=(uint32_t)ns||hf!=(uint32_t)nf||
       info.st_size<0||(uint64_t)info.st_size!=base+7*(uint64_t)ns*nf*8)return NULL;
    CMFNodalInput *c=calloc(1,sizeof(*c));if(!c){*code=3;return NULL;}
    c->fd=-1;c->ns=ns;c->nf=nf;c->added=added;c->reference_b=reference_b;c->width=width;
    c->capacity=capacity;c->np=np;c->base=base;c->minimum=minimum;c->nbuckets=2*capacity+1;
    c->fd=dup(fd);c->r=malloc((size_t)ns*8);c->ext=malloc((size_t)(added+1)*8);
    c->scratch=malloc((size_t)7*width*8);c->storage=malloc((size_t)capacity*3*width*8);
    c->profile_storage=malloc((size_t)np*2*(added+1)*8);
    c->rows=calloc(capacity,sizeof(Row));c->profiles=calloc(np,sizeof(Profile));c->buckets=malloc((size_t)c->nbuckets*sizeof(int));
    if(c->fd<0||!c->r||!c->ext||!c->scratch||!c->storage||!c->profile_storage||
       !c->rows||!c->profiles||!c->buckets){*code=3;cmf_nodal_input_destroy(c);return NULL;}
    memcpy(c->r,r,(size_t)ns*8);memcpy(c->ext,ext,(size_t)(added+1)*8);
    for(int i=0;i<c->nbuckets;i++)c->buckets[i]=-1;
    for(int i=0;i<capacity;i++){c->rows[i].row=-1;c->rows[i].next=-1;c->rows[i].values=c->storage+(size_t)i*3*width;}
    for(int i=0;i<np;i++){c->profiles[i].column=-1;c->profiles[i].values=c->profile_storage+(size_t)i*2*(added+1);}
    c->stats.static_cache_bytes=(uint64_t)capacity*(3*width*8+sizeof(Row))+(uint64_t)c->nbuckets*sizeof(int)+7*(uint64_t)width*8;
    c->stats.profile_cache_bytes=(uint64_t)np*(2*(added+1)*8+sizeof(Profile));
    *code=0;return c;
}

static unsigned bucket(const CMFNodalInput *c,int row,int slab)
{
    uint64_t key=((uint64_t)(uint32_t)row<<32)|(uint32_t)slab;
    key^=key>>33;key*=UINT64_C(0xff51afd7ed558ccd);key^=key>>33;
    return (unsigned)(key%(unsigned)c->nbuckets);
}

static int cached_row(CMFNodalInput *c,int row,int column,Row **result)
{
    if(row<0||row>=c->ns||column<0||column>=c->nf)return 1;
    int slab=column/c->width;unsigned b=bucket(c,row,slab);
    for(int i=c->buckets[b];i>=0;i=c->rows[i].next)if(c->rows[i].row==row&&c->rows[i].slab==slab){
        c->rows[i].age=++c->clock;c->stats.cache_hits++;*result=c->rows+i;return 0;
    }
    int victim=0;
    for(int i=0;i<c->capacity;i++){
        if(c->rows[i].row<0){victim=i;break;}
        if(c->rows[i].age<c->rows[victim].age)victim=i;
    }
    Row *entry=c->rows+victim;
    if(entry->row>=0){
        int *link=c->buckets+bucket(c,entry->row,entry->slab);
        while(*link!=victim){if(*link<0)return 2;link=&c->rows[*link].next;}
        *link=entry->next;
    }
    entry->row=-1;entry->next=-1;c->stats.cache_misses++;
    int first=slab*c->width,n=c->nf-first;if(n>c->width)n=c->width;
    for(int plane=0;plane<7;plane++){
        uint64_t offset=c->base+8*(((uint64_t)plane*c->ns+row)*c->nf+first);
        if(exact_read(c->fd,c->scratch+(size_t)plane*c->width,(size_t)n*8,offset,&c->stats))return 2;
    }
    for(int i=0;i<n;i++){
        double p[7];for(int plane=0;plane<7;plane++){
            p[plane]=c->scratch[(size_t)plane*c->width+i];
            if(!isfinite(p[plane])||((plane==1||plane==2||plane==3||plane==5||plane==6)&&p[plane]<0))return 2;
        }
        double chi=(p[6]+(p[0]+p[2]))+p[4],eta=(p[1]+p[3])+p[5];
        if(!isfinite(chi)||!isfinite(eta))return 2;
        entry->values[i]=chi;entry->values[c->width+i]=p[6];entry->values[2*c->width+i]=eta;
    }
    entry->row=row;entry->slab=slab;entry->age=++c->clock;entry->next=c->buckets[b];c->buckets[b]=victim;
    uint64_t live=0;for(int i=0;i<c->capacity;i++)if(c->rows[i].row>=0)live++;
    if(live>c->stats.cache_peak_rows)c->stats.cache_peak_rows=live;
    *result=entry;return 0;
}

static int native(CMFNodalInput *c,int row,int column,double *chi,double *es,double *eta)
{
    if(row<0||row>=c->ns||column<0||column>=c->nf)return 1;
    if(c->dense){
        size_t at=(size_t)row*c->nf+column,n=(size_t)c->ns*c->nf;
        *chi=c->dense[at];*es=c->dense[n+at];*eta=c->dense[2*n+at];
        c->dense_stats.native_accesses++;return 0;
    }
    Row *entry;int code=cached_row(c,row,column,&entry);if(code)return code;
    int i=column%c->width;*chi=entry->values[i];*es=entry->values[c->width+i];*eta=entry->values[2*c->width+i];return 0;
}

static void invalidate_dense_profile(CMFNodalInput *c)
{
    free(c->dense_profile);c->dense_profile=NULL;c->dense_stats.profile_bytes=0;c->dense_stats.ready=0;
}

int cmf_nodal_input_bind(CMFNodalInput *c,int first,int count,const double *j)
{
    if(!c||!j||first<0||count<1||first>c->nf-count||(uint64_t)c->ns*count>131072)return 1;
    size_t n=(size_t)c->ns*count;for(size_t i=0;i<n;i++)if(!isfinite(j[i])||j[i]<0)return 1;
    double *copy=malloc(n*8);if(!copy)return 3;
    memcpy(copy,j,n*8);free(c->j);c->j=copy;c->jview=NULL;c->zero_j=0;c->start=first;c->count=count;
    invalidate_dense_profile(c);
    for(int i=0;i<c->np;i++)c->profiles[i].column=-1;
    c->stats.bindings++;c->stats.field_bytes_copied+=n*8;c->stats.field_buffer_bytes=n*8;return 0;
}

int cmf_nodal_input_bind_view(CMFNodalInput *c,const double *j)
{
    if(!c||!j)return 1;
    free(c->j);c->j=NULL;c->jview=j;c->zero_j=0;c->start=0;c->count=c->nf;
    invalidate_dense_profile(c);
    for(int i=0;i<c->np;i++)c->profiles[i].column=-1;
    c->stats.bindings++;c->stats.field_buffer_bytes=0;return 0;
}

int cmf_nodal_input_bind_zero_trial(CMFNodalInput *c)
{
    if(!c)return 1;
    free(c->j);c->j=NULL;c->jview=NULL;c->zero_j=1;c->start=0;c->count=c->nf;
    invalidate_dense_profile(c);
    for(int i=0;i<c->np;i++)c->profiles[i].column=-1;
    c->stats.bindings++;c->stats.field_buffer_bytes=0;return 0;
}

int cmf_nodal_input_raw(CMFNodalInput *c,int nr,const int *rows,int nc,const int *columns,
    double *chi,double *es,double *eta)
{
    if(!c||nr<1||nc<1||(uint64_t)nr*nc>131072||!rows||!columns||!chi||!es||!eta)return 1;
    for(int r=0;r<nr;r++)for(int f=0;f<nc;f++){
        size_t at=(size_t)r*nc+f;int code=native(c,rows[r],columns[f],chi+at,es+at,eta+at);
        if(code)return code;
    }
    return 0;
}

static int jvalue(const CMFNodalInput *c,int row,int column,double *value)
{
    if(c->zero_j){*value=0.;return 0;}
    if(c->jview){
        *value=c->jview[(size_t)row*c->nf+column];
        return !isfinite(*value)||*value<0 ? 1 : 0;
    }
    if(!c->j||column<c->start||column>=c->start+c->count)return 1;
    *value=c->j[(size_t)row*c->count+column-c->start];return 0;
}

static int profile(CMFNodalInput *c,int column,Profile **result)
{
    int victim=column%c->np;
    if(c->profiles[victim].column==column){c->profiles[victim].age=++c->clock;c->stats.profile_hits++;
        *result=c->profiles+victim;return 0;}
    double chi0,es0,eta0,chib,esb,etab,j0,jb;
    if(native(c,0,column,&chi0,&es0,&eta0)||native(c,c->reference_b,column,&chib,&esb,&etab)||
       jvalue(c,0,column,&j0)||jvalue(c,c->reference_b,column,&jb)||es0<=0||esb<=0)return 2;
    double total0=eta0+es0*j0,totalb=etab+esb*jb;
    if(!isfinite(total0)||!isfinite(totalb))return 2;
    double log_radius=log(c->r[0]/c->r[c->reference_b]);
    if(!isfinite(log_radius)||log_radius<=0)return 2;
    double es_power=fmax(2.,log(esb/es0)/log_radius),absorption_power=0.,eta_power=3.5;
    int special=chi0<=es0||chib<=esb;
    if(!special)absorption_power=fmax(2.,log((chib-esb)/(chi0-es0))/log_radius);
    if(total0>0&&totalb>0){
        double value=log(totalb/total0);if(!isfinite(value))value=log(totalb)-log(total0);
        eta_power=fmax(3.5,value/log_radius);
    }
    Profile *p=c->profiles+victim;p->column=-1;p->sign=chi0>0 ? 1. : chi0<0 ? -1. : 0.;
    int ne=c->added+1;
    for(int i=0;i<ne;i++){
        double ratio=c->r[0]/c->ext[i];
        double chi=i==c->added ? chi0 : special ? chi0*pow(ratio,es_power) :
            (chi0-es0)*pow(ratio,absorption_power)+es0*pow(ratio,es_power);
        double eta=i==c->added ? total0 : fmax(c->minimum,total0*pow(ratio,eta_power));
        if(!isfinite(chi)||!isfinite(eta))return 2;
        p->values[i]=chi==0 ? 0. : log(fabs(chi));
        p->values[ne+i]=eta==0 ? -INFINITY : log(eta);
    }
    p->column=column;p->age=++c->clock;c->stats.profile_misses++;*result=p;return 0;
}

static double dense_now(void)
{
    struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return NAN;
    return t.tv_sec+1e-9*t.tv_nsec;
}

int cmf_nodal_input_prepare_dense(CMFNodalInput *c,double seconds)
{
    if(!c||!isfinite(seconds)||seconds<=0||seconds>300.||
       (!c->zero_j&&!c->jview&&(!c->j||c->start||c->count!=c->nf)))return 1;
    double started=dense_now();if(!isfinite(started))return 4;
    uint64_t canonical=3*(uint64_t)c->ns*c->nf*8,profiles=(uint64_t)(2*(c->added+1)+1)*c->nf*8;
    if(canonical+profiles>UINT64_C(3221225472))return 3;
    int code=0;double *values=NULL,*scratch=NULL,*outer=NULL;size_t n=(size_t)c->ns*c->nf;
    if(!c->dense){
        values=malloc((size_t)canonical);scratch=malloc(7*65536*sizeof(double));
        if(!values||!scratch){code=3;goto done;}
        for(int row=0;row<c->ns;row++)for(int begin=0;begin<c->nf;begin+=65536){
            if(dense_now()-started>=seconds){code=4;goto done;}
            int width=c->nf-begin;if(width>65536)width=65536;
            for(int plane=0;plane<7;plane++){
                uint64_t offset=c->base+8*(((uint64_t)plane*c->ns+row)*c->nf+begin);
                if(exact_read(c->fd,scratch+(size_t)plane*65536,(size_t)width*8,offset,&c->stats)){code=2;goto done;}
            }
            for(int f=0;f<width;f++){
                double p[7];for(int plane=0;plane<7;plane++){
                    p[plane]=scratch[(size_t)plane*65536+f];
                    if(!isfinite(p[plane])||((plane==1||plane==2||plane==3||plane==5||plane==6)&&p[plane]<0)){
                        code=2;goto done;
                    }
                }
                size_t at=(size_t)row*c->nf+begin+f;
                values[at]=(p[6]+(p[0]+p[2]))+p[4];values[n+at]=p[6];values[2*n+at]=(p[1]+p[3])+p[5];
                if(!isfinite(values[at])||!isfinite(values[2*n+at])){code=2;goto done;}
            }
        }
        c->dense=values;values=NULL;c->dense_stats.canonical_bytes=canonical;
    }
    if(!c->dense_profile){
        outer=malloc((size_t)profiles);if(!outer){code=3;goto done;}
        int stride=2*(c->added+1)+1;
        for(int column=0;column<c->nf;column++){
            if(column%4096==0&&dense_now()-started>=seconds){code=4;goto done;}
            Profile *p;if(profile(c,column,&p)){code=2;goto done;}
            memcpy(outer+(size_t)column*stride,p->values,(size_t)(stride-1)*8);
            outer[(size_t)column*stride+stride-1]=p->sign;
        }
        c->dense_profile=outer;outer=NULL;c->dense_stats.profile_bytes=profiles;
    }
    c->dense_stats.ready=1;
done:
    free(outer);free(values);free(scratch);c->dense_stats.preparation_seconds+=dense_now()-started;return code;
}

int cmf_nodal_input_dense_stats(const CMFNodalInput *c,CMFNodalDenseStats *stats)
{
    if(!c||!stats)return 1;
    *stats=c->dense_stats;return 0;
}

int cmf_nodal_input_dense_view(const CMFNodalInput *c,CMFNodalDenseView *view)
{
    if(!c||!view||!c->dense||!c->dense_profile||!c->dense_stats.ready)return 1;
    const double *j=NULL;
    if(!c->zero_j){
        j=c->jview ? c->jview : c->start==0&&c->count==c->nf ? c->j : NULL;
        if(!j)return 1;
    }
    CMFNodalDenseView value={c->ns,c->nf,c->added+1,!c->zero_j,c->stats.bindings,
        c->dense,c->dense_profile,j,c->r,c->ext};
    *view=value;return 0;
}

static double mix(double first,double last,double weight)
{
    return (1.-weight)*first+weight*last;
}

int cmf_nodal_input_prepare(const CMFNodalInput *c,double radius,CMFNodalLocation *location)
{
    if(!c||!location||!isfinite(radius)||radius<c->r[c->ns-1]||radius>c->ext[0])return 1;
    if(radius>c->r[0]){
        int i=0;while(i<c->added-1&&radius<c->ext[i+1])i++;
        *location=(CMFNodalLocation){2,i+1,i,log(radius/c->ext[i+1])/log(c->ext[i]/c->ext[i+1])};
    }else if(radius==c->r[c->ns-1]){
        *location=(CMFNodalLocation){0,c->ns-1,c->ns-1,0.};
    }else{
        int outer=0;while(outer<c->ns-2&&radius<c->r[outer+1])outer++;
        int inner=outer+1;
        *location=(CMFNodalLocation){1,inner,outer,(radius-c->r[inner])/(c->r[outer]-c->r[inner])};
    }
    return !isfinite(location->weight)||location->weight<0||location->weight>1 ? 2 : 0;
}

static int at_column(CMFNodalInput *c,const CMFNodalLocation *location,int column,double *chi,double *eta)
{
    int first=location->first,last=location->last;double w=location->weight;
    if(location->mode==2){
        int ne=c->added+1;
        const double *values;double sign;
        if(c->dense_profile){
            values=c->dense_profile+(size_t)column*(2*ne+1);sign=values[2*ne];c->dense_stats.profile_accesses++;
        }else{Profile *p;if(profile(c,column,&p))return 2;values=p->values;sign=p->sign;}
        *chi=sign*exp(mix(values[first],values[last],w));
        double e=w==0 ? values[ne+first] : w==1 ? values[ne+last] : mix(values[ne+first],values[ne+last],w);
        *eta=exp(e);
    }else if(location->mode==0){
        double es,j;if(native(c,first,column,chi,&es,eta)||jvalue(c,first,column,&j)||es<=0)return 2;
        *eta+=es*j;
    }else{
        double c0,s0,e0,c1,s1,e1,j0,j1;
        if(native(c,first,column,&c0,&s0,&e0)||native(c,last,column,&c1,&s1,&e1)||
           jvalue(c,first,column,&j0)||jvalue(c,last,column,&j1)||s0<=0||s1<=0)return 2;
        *chi=mix(c0,c1,w);*eta=mix(e0,e1,w)+mix(s0,s1,w)*mix(j0,j1,w);
    }
    return !isfinite(*chi)||!isfinite(*eta)||*eta<0 ? 2 : 0;
}

int cmf_nodal_input_sample_prepared(CMFNodalInput *c,const CMFNodalLocation *location,double column,double *chi,double *eta)
{
    if(!c||!location||!chi||!eta||!isfinite(column)||column<0||!isfinite(location->weight)||
       location->weight<0||location->weight>1||location->last<0||
       (location->mode==0&&(location->first!=c->ns-1||location->last!=c->ns-1))||
       ((location->mode==1||location->mode==2)&&(location->last>=(location->mode==1 ? c->ns : c->added+1)-1||
         location->first!=location->last+1))||location->mode<0||location->mode>2)return 1;
    c->stats.queries++;int first=column>=c->nf-1 ? c->nf-1 : (int)floor(column);
    double fraction=column>=c->nf-1 ? 0. : column-first;
    int code=at_column(c,location,first,chi,eta);if(code)return code;
    if(fraction>0){double last_chi,last_eta;code=at_column(c,location,first+1,&last_chi,&last_eta);if(code)return code;
        *chi=*chi+fraction*(last_chi-*chi);*eta=*eta+fraction*(last_eta-*eta);}
    return !isfinite(*chi)||!isfinite(*eta)||*eta<0 ? 2 : 0;
}

int cmf_nodal_input_sample(CMFNodalInput *c,double radius,double column,double *chi,double *eta)
{
    CMFNodalLocation location;int code=cmf_nodal_input_prepare(c,radius,&location);if(code)return code;
    return cmf_nodal_input_sample_prepared(c,&location,column,chi,eta);
}

int cmf_nodal_input_eval(CMFNodalInput *c,int nr,const double *radii,int nc,const double *columns,double *chi,double *eta)
{
    if(!c||nr<1||nc<1||(uint64_t)nr*nc>131072||!radii||!columns||!chi||!eta)return 1;
    for(int r=0;r<nr;r++){
        CMFNodalLocation location;int code=cmf_nodal_input_prepare(c,radii[r],&location);if(code)return code;
        for(int f=0;f<nc;f++){
            size_t at=(size_t)r*nc+f;code=cmf_nodal_input_sample_prepared(c,&location,columns[f],chi+at,eta+at);if(code)return code;
        }
    }
    return 0;
}

int cmf_nodal_input_stats(const CMFNodalInput *c,CMFNodalInputStats *stats)
{
    if(!c||!stats)return 1;
    *stats=c->stats;return 0;
}

int cmf_nodal_input_shape(const CMFNodalInput *c,int *ns,int *nf)
{
    if(!c||!ns||!nf)return 1;
    *ns=c->ns;*nf=c->nf;return 0;
}
