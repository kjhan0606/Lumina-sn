#ifndef CMF_ANDERSON_MULTI_H
#define CMF_ANDERSON_MULTI_H
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* 시험된 다중 이력 가속 후보. 실제 G(x)-x/오차상계는 호출자가 판정한다.
 * QR의 수치 rank 절단은 최소제곱에만 적용하며 물리값을 자르지 않는다. */
#define CMF_ANDERSON_MULTI_MAX 4
typedef struct {
    size_t n, proposals, rejected, rank_losses;
    int depth, count, next, proposed;
    double previous_norm;
    double *storage, *g, *r, *work, *rhs;
} CMFAndersonMulti;

static inline int cmf_anderson_depth(void){
    const char *v=getenv("LUMINA_CMF_ANDERSON_DEPTH");
    if(!v||!strcmp(v,"1"))return 1;
    if(!strcmp(v,"2"))return 2;
    if(!strcmp(v,"4"))return 4;
    return 0;
}
static inline int cmf_anderson_multi_init(CMFAndersonMulti *a,size_t n,int depth){
    if(!a||!n||depth<2||depth>CMF_ANDERSON_MULTI_MAX||
       n>SIZE_MAX/sizeof(double)/(3*(size_t)depth+1))return -1;
    memset(a,0,sizeof(*a));
    a->storage=(double *)malloc(n*(3*(size_t)depth+1)*sizeof(double));
    if(!a->storage)return -1;
    a->n=n;a->depth=depth;a->g=a->storage;a->r=a->g+n*depth;
    a->work=a->r+n*depth;a->rhs=a->work+n*depth;
    return 0;
}
static inline void cmf_anderson_multi_free(CMFAndersonMulti *a){
    if(a){free(a->storage);memset(a,0,sizeof(*a));}
}
static inline double cmf_anderson_dot(size_t n,const double *a,const double *b){
    long double sum=0;
    for(size_t i=0;i<n;i++)sum+=(long double)a[i]*b[i];
    return (double)sum;
}
static inline void cmf_anderson_multi_next(CMFAndersonMulti *a,double *x,
                                         const double *g,double residual){
    const size_t n=a->n;
    if(a->proposed&&(!isfinite(residual)||residual>a->previous_norm)){
        int last=(a->next+a->depth-1)%a->depth;
        memcpy(x,a->g+(size_t)last*n,n*sizeof(double));
        a->count=a->next=a->proposed=0;a->rejected++;
        return;
    }
    int m=a->count,rank=0,perm[CMF_ANDERSON_MULTI_MAX];
    double *col[CMF_ANDERSON_MULTI_MAX];
    double R[CMF_ANDERSON_MULTI_MAX][CMF_ANDERSON_MULTI_MAX]={{0}};
    double beta[CMF_ANDERSON_MULTI_MAX]={0},coef[CMF_ANDERSON_MULTI_MAX]={0};
    for(int j=0;j<m;j++){perm[j]=j;col[j]=a->work+(size_t)j*n;}
    if(m)for(size_t i=0;i<n;i++){
        double scale=fmax(fabs(x[i]),fabs(g[i]));
        for(int j=0;j<m;j++)scale=fmax(scale,fmax(fabs(a->g[(size_t)j*n+i]),fabs(a->r[(size_t)j*n+i])));
        double r=scale>0?g[i]/scale-x[i]/scale:0;
        a->rhs[i]=r;
        for(int j=0;j<m;j++)col[j][i]=scale>0?r-a->r[(size_t)j*n+i]/scale:0;
    }
    double first_norm=0;
    /* 열 피벗 MGS와 재직교화. 정규방정식의 조건수 제곱을 피한다. */
    for(int k=0;k<m;k++){
        int pivot=k;double largest=-1;
        for(int j=k;j<m;j++){
            double norm=cmf_anderson_dot(n,col[j],col[j]);
            if(norm>largest){largest=norm;pivot=j;}
        }
        double norm=sqrt(largest);
        if(k==0)first_norm=norm;
        double threshold=DBL_EPSILON*(double)(n>(size_t)m?n:(size_t)m)*first_norm;
        if(!isfinite(norm)||!(norm>threshold))break;
        if(pivot!=k){
            double *p=col[k];col[k]=col[pivot];col[pivot]=p;
            int v=perm[k];perm[k]=perm[pivot];perm[pivot]=v;
            for(int i=0;i<k;i++){double v0=R[i][k];R[i][k]=R[i][pivot];R[i][pivot]=v0;}
        }
        R[k][k]=norm;
        for(size_t i=0;i<n;i++)col[k][i]/=norm;
        beta[k]=cmf_anderson_dot(n,col[k],a->rhs);
        for(size_t i=0;i<n;i++)a->rhs[i]-=beta[k]*col[k][i];
        for(int j=k+1;j<m;j++)for(int pass=0;pass<2;pass++){
            double dot=cmf_anderson_dot(n,col[k],col[j]);R[k][j]+=dot;
            for(size_t i=0;i<n;i++)col[j][i]-=dot*col[k][i];
        }
        rank++;
    }
    if(rank<m)a->rank_losses++;
    double ordered[CMF_ANDERSON_MULTI_MAX]={0};
    int use=rank>0;
    for(int k=rank-1;k>=0;k--){
        double rhs=beta[k];
        for(int j=k+1;j<rank;j++)rhs-=R[k][j]*ordered[j];
        ordered[k]=rhs/R[k][k];coef[perm[k]]=ordered[k];
        if(!isfinite(ordered[k]))use=0;
    }
    if(use)for(size_t i=0;i<n;i++){
        double candidate=g[i];
        for(int j=0;j<m;j++)candidate-=coef[j]*(g[i]-a->g[(size_t)j*n+i]);
        if(!(candidate>=0)||!isfinite(candidate)){use=0;break;}
    }
    if(m&&!use)a->rejected++;
    size_t offset=(size_t)a->next*n;
    for(size_t i=0;i<n;i++){
        double candidate=g[i];
        if(use)for(int j=0;j<m;j++)candidate-=coef[j]*(g[i]-a->g[(size_t)j*n+i]);
        a->r[offset+i]=g[i]-x[i];a->g[offset+i]=g[i];x[i]=candidate;
    }
    if(a->count<a->depth)a->count++;
    a->next=(a->next+1)%a->depth;a->previous_norm=residual;a->proposed=use;
    if(use)a->proposals++;
}
#endif
