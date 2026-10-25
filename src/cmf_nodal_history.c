/* 같은 수송 수식과 원래점 cache 사이의 C 직접 연결. Python callback은 없다. */
#include "cmf_nodal_history.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {CMFNodalInput *input;const CMFNodalLocation *locations;int points;} Samples;

static int coefficients(void *user,int point,double column,double *chi,double *eta)
{
    Samples *samples=user;
    if(!samples||point<0||point>=samples->points)return 1;
    return cmf_nodal_input_sample_prepared(samples->input,samples->locations+point,column,chi,eta);
}

int cmf_nodal_history_range(CMFNodalInput *input,int nf,int begin,int count,int small,
    double h,double length,double da,double db,const double *radii,
    const double *intensity,double *out,double *response,CMFHistoryStats *stats)
{
    int ns,actual_nf;
    if(!input||!radii||(small!=0&&small!=1)||cmf_nodal_input_shape(input,&ns,&actual_nf)||nf!=actual_nf)return 1;
    for(int i=0;i<3;i++)if(!isfinite(radii[i])||radii[i]<=0)return 1;
    CMFNodalLocation locations[3];
    for(int i=0;i<3;i++)if(cmf_nodal_input_prepare(input,radii[i],locations+i))return 1;
    Samples samples={input,locations,3};
    if(small)return cmf_frequency_small_provider(nf,begin,count,h,length,da,db,coefficients,&samples,0,2,
        intensity,out,response,stats);
    return cmf_frequency_history_provider(nf,begin,count,h,length,da,db,coefficients,&samples,1,0,2,
        intensity,out,response,stats);
}

int cmf_nodal_history_compare(CMFNodalInput *input,int nf,int begin,int width,int leaves,double h,
    const double *lengths,const double *deltas,const double *radii,const double *initial,
    double *out,double *errors,double *responses,CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks)
{
    int ns,actual_nf;
    if(!input||!radii||leaves<1||leaves>4096||cmf_nodal_input_shape(input,&ns,&actual_nf)||nf!=actual_nf)return 1;
    for(int i=0;i<5*leaves;i++)if(!isfinite(radii[i])||radii[i]<=0)return 1;
    CMFNodalLocation *locations=malloc((size_t)5*leaves*sizeof(*locations));if(!locations)return 3;
    for(int i=0;i<5*leaves;i++)if(cmf_nodal_input_prepare(input,radii[i],locations+i)){free(locations);return 1;}
    Samples samples={input,locations,5*leaves};
    int code=cmf_frequency_hybrid_provider_compare(nf,begin,width,leaves,h,lengths,deltas,coefficients,&samples,
        initial,out,errors,responses,stats,calls,checks);
    free(locations);return code;
}

int cmf_nodal_history_maps(CMFNodalInput *input,int nf,int begin,int width,int leaves,double h,
    const double *lengths,const double *deltas,const double *radii,double *maps,unsigned char *safe,
    CMFHistoryStats *stats,uint64_t *calls,uint64_t *checks)
{
    int ns,actual_nf;
    if(!input||!radii||leaves<1||leaves>4096||cmf_nodal_input_shape(input,&ns,&actual_nf)||nf!=actual_nf)return 1;
    CMFNodalLocation *locations=malloc((size_t)5*leaves*sizeof(*locations));if(!locations)return 3;
    for(int i=0;i<5*leaves;i++)if(cmf_nodal_input_prepare(input,radii[i],locations+i)){free(locations);return 1;}
    Samples samples={input,locations,5*leaves};
    int code=cmf_frequency_hybrid_provider_maps(nf,begin,width,leaves,h,lengths,deltas,coefficients,&samples,
        maps,safe,stats,calls,checks);
    free(locations);return code;
}

static int moderate(double x){return isfinite(x)&&(x==0.||(x>=1e-80&&x<=1e80));}

int cmf_nodal_history_replay(CMFNodalInput *input,int nf,int begin,int width,int leaves,double h,
    const double *lengths,const double *deltas,const double *radii,const double *const *maps,const unsigned char *safe,
    const double *initial,double *out,double *errors,double *responses,uint64_t max_fallback_scalar,uint64_t *fallback_stats)
{
    int ns,actual_nf;
    if(!input||cmf_nodal_input_shape(input,&ns,&actual_nf)||nf!=actual_nf||begin<0||width<1||width>65536||
       begin>nf-width||leaves<1||leaves>4096||(uint64_t)leaves*width>2097152||!isfinite(h)||h<=0||
       !lengths||!deltas||!radii||!maps||!safe||!initial||!out||out==initial||!errors||!responses||!fallback_stats)return 1;
    memset(fallback_stats,0,3*sizeof(uint64_t));
    for(int f=0;f<width;f++)if(!isfinite(initial[f])||initial[f]<0)return 1;
    memcpy(out,initial,(size_t)width*sizeof(double));
    size_t mask_stride=((size_t)width+7)/8;
    for(int row=0;row<leaves;row++){
        if(!maps[row])return 1;
        double *e=errors+(size_t)row*width,*response=responses+(size_t)row*width;
        const double *m=maps[row];
        const double *d=deltas+3*row,*a[3];
        double spans[3]={(d[0]-d[2])/h,(d[0]-d[1])/h,(d[1]-d[2])/h};int offset=6;
        for(int k=0;k<3;k++){
            a[k]=m+(size_t)2*k*width;
            if(spans[k]>=.75&&spans[k]<=1.25){a[k]=m+(size_t)offset*width;offset+=2;}
        }
        for(int f=0;f<width;){
            if(!(safe[(size_t)row*mask_stride+f/8]&(1u<<(f%8)))||!moderate(out[f])){
                int end=f+1;
                while(end<width&&(!(safe[(size_t)row*mask_stride+end/8]&(1u<<(end%8)))||!moderate(out[end])))end++;
                int count=end-f;uint64_t expected=3;
                for(int k=0;k<3;k++)if(spans[k]>=.75&&spans[k]<=1.25)expected++;
                if(fallback_stats[0]+expected*(uint64_t)count>max_fallback_scalar)return 5;
                double *value=malloc((size_t)count*sizeof(double));if(!value)return 3;
                CMFHistoryStats stats;uint64_t calls=0,checks=0;
                int code=cmf_nodal_history_compare(input,nf,begin+f,count,1,h,lengths+3*row,deltas+3*row,radii+5*row,
                    out+f,value,e+f,response+f,&stats,&calls,&checks);
                fallback_stats[0]+=calls*(uint64_t)count;fallback_stats[1]+=calls;fallback_stats[2]+=checks*(uint64_t)count;
                if(!code)memcpy(out+f,value,(size_t)count*sizeof(double));
                free(value);if(code)return code;
                f=end;continue;
            }
            double incident=out[f],coarse=m[f]*incident+m[width+f],alternate=a[0][f]*incident+a[0][width+f];
            double first=m[2*width+f]*incident+m[3*width+f],left_alt=a[1][f]*incident+a[1][width+f];
            double value=m[4*width+f]*first+m[5*width+f],right_alt=a[2][f]*first+a[2][width+f];
            /* 먼저 Tleft*Tright를 만들지 않는다. 원래 left→right 입사 순서를 유지한다. */
            e[f]=fabs(value-coarse)+fabs(coarse-alternate)+m[4*width+f]*fabs(first-left_alt)+fabs(value-right_alt);
            response[f]=m[4*width+f]*m[2*width+f];out[f]=value;
            if(!isfinite(value)||value<0||!isfinite(e[f])||!isfinite(response[f]))return 2;
            f++;
        }
    }
    return 0;
}
