/* 기존 기본 양의 모멘트 코어. CPU/GPU는 같은 식/허용오차/순서를 사용한다. */
#ifndef CMF_POSITIVE_MOMENTS_H
#define CMF_POSITIVE_MOMENTS_H
#include "cmf_frequency_history.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#ifdef __CUDACC__
#define LUMINA_POSITIVE static __device__ inline
#define LUMINA_QUAD_CONSTANT static __device__ __constant__
#define LUMINA_CUT_CAPACITY 512
#else
#define LUMINA_POSITIVE static inline
#define LUMINA_QUAD_CONSTANT static const
#define LUMINA_CUT_CAPACITY 4096
#endif
/* 장치 cut 용량 초과는 명시 오류다. cut/물리값을 잘라 통과시키지 않는다. */
LUMINA_QUAD_CONSTANT double LUMINA_QUAD_X[8]={.0950125098376374402,.281603550779258913,
    .458016777657227386,.617876244402643748,.755404408355003034,
    .865631202387831744,.944575023073232576,.989400934991649933};
LUMINA_QUAD_CONSTANT double LUMINA_QUAD_WEIGHT[8]={.189450610455068496,.182603415044923589,
    .169156519395002538,.149595988816576732,.124628971255533872,
    .095158511682492785,.062253523938647893,.027152459411754095};

LUMINA_POSITIVE int valid(CMFHistoryMap m)
{
    return isfinite(m.d)&&isfinite(m.t)&&isfinite(m.aa)&&isfinite(m.pa)&&
        isfinite(m.ab)&&isfinite(m.mb)&&m.d>=0&&m.t>=0&&m.aa>=0&&m.pa>=0&&m.ab>=0&&m.mb>=0;
}

LUMINA_POSITIVE int quadrature(double q0,double q1,double a,double b,double w[3],CMFHistoryStats *s)
{
    double half=.5*(b-a),mid=.5*(a+b);w[0]=w[1]=w[2]=0.;
    s->quadrature_nodes+=16;s->quadrature_panels++;
    for(int k=0;k<8;k++)for(int sign=-1;sign<=1;sign+=2){
        double u=mid+sign*half*LUMINA_QUAD_X[k],v=1.-u;
        double phase=-u*(q1+.5*(q0-q1)*u);
        double value=half*LUMINA_QUAD_WEIGHT[k]*exp(phase);
        if(!isfinite(phase)||!isfinite(value))return 2;
        w[0]+=value*u*u;w[1]+=value*u*v;w[2]+=value*v*v;
    }
    return 0;
}

LUMINA_POSITIVE int panel(double q0,double q1,double a,double b,int depth,
                 const double tol[3],double out[3],CMFHistoryStats *s)
{
    /* 이전 재귀의 left→right 평가/합산 순서를 고정 길이 DFS stack으로 보존한다. */
    typedef struct {double a,b,tol[3],left[3];int depth,state;} Frame;
    Frame stack[13],root={a,b,{tol[0],tol[1],tol[2]},{0.,0.,0.},depth,0};
    stack[0]=root;int at=0,returning=0;double value[3];
    for(;;){
        if(returning){
            if(at<0){for(int k=0;k<3;k++)out[k]=value[k];return 0;}
            Frame *f=stack+at;
            if(f->state==1){
                for(int k=0;k<3;k++)f->left[k]=value[k];
                f->state=2;double mid=.5*(f->a+f->b);
                Frame child={mid,f->b,{.5*f->tol[0],.5*f->tol[1],.5*f->tol[2]},
                    {0.,0.,0.},f->depth+1,0};
                if(at>=12)return 2;
                stack[++at]=child;returning=0;
            }else{
                for(int k=0;k<3;k++)value[k]=f->left[k]+value[k];
                at--;
            }
            continue;
        }
        Frame *f=stack+at;double mid=.5*(f->a+f->b),coarse[3],first[3],last[3];int met=1;
        if(quadrature(q0,q1,f->a,f->b,coarse,s)||quadrature(q0,q1,f->a,mid,first,s)||
           quadrature(q0,q1,mid,f->b,last,s))return 2;
        for(int k=0;k<3;k++){value[k]=first[k]+last[k];if(fabs(value[k]-coarse[k])>f->tol[k])met=0;}
        if(met){at--;returning=1;continue;}
        if(f->depth>=12||mid==f->a||mid==f->b||at>=12)return 2;
        f->state=1;
        Frame child={f->a,mid,{.5*f->tol[0],.5*f->tol[1],.5*f->tol[2]},
            {0.,0.,0.},f->depth+1,0};
        stack[++at]=child;
    }
}

#ifndef __CUDACC__
LUMINA_POSITIVE int compare_double(const void *a,const void *b)
{
    double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
#endif

/* 광학적으로 얇은 지수만 다항식으로 적분한다. 값의 floor/clamp가 아니다.
 * |phase(u)|<=Q이므로 상대 절단오차는 exp(2Q) Q^(N+1)/(N+1)! 이하다.
 * 계수 생성/세 모멘트 누적의 보수적 연산수 gamma 상계도 예산에 포함한다.
 * 범위를 벗어나거나 양의 모멘트/예산을 충족하지 못하면 기존 구적으로 돌아간다. */
LUMINA_POSITIVE int thin_basis(double q0,double q1,double out[3])
{
    double q=fmax(fabs(q0),fabs(q1));if(!(q<.1))return 1;
    int degree=0;double remainder=q,scale=exp(2*q);
    while(degree<12&&scale*remainder>4*DBL_EPSILON){
        degree++;remainder*=q/(degree+1.);
    }
    double operations=16.*(degree+1)*(degree+1)+128.;
    double gamma=operations*DBL_EPSILON/(1.-operations*DBL_EPSILON);
    double rho=fabs(q1)+.5*fabs(q0-q1);
    if(scale*remainder+gamma*exp(rho+q)>1e-12)return 1;
    double coefficients[25]={1.},a=-q1,b=-.5*(q0-q1);
    out[0]=out[2]=1./3;out[1]=1./6;
    for(int n=1;n<=degree;n++){
        double next[25]={0.};
        for(int k=0;k<=2*(n-1);k++){
            next[k+1]+=coefficients[k]*(a/n);
            next[k+2]+=coefficients[k]*(b/n);
        }
        for(int k=1;k<=2*n;k++){
            out[0]+=next[k]/(k+3.);
            out[1]+=next[k]/((k+2.)*(k+3.));
            out[2]+=2*next[k]/((k+1.)*(k+2.)*(k+3.));
        }
        memcpy(coefficients,next,sizeof(coefficients));
    }
    for(int k=0;k<3;k++)if(!isfinite(out[k])||out[k]<=0)return 1;
    return 0;
}

LUMINA_POSITIVE int basis(double q0,double q1,double t,double out[3],CMFHistoryStats *s)
{
    if(q0==q1&&fabs(q0)<.5){
        double term=1.;out[0]=out[1]=out[2]=0.;
        for(int k=0;k<24;k++){
            out[0]+=term/(k+3.);out[1]+=term/((k+2.)*(k+3.));
            out[2]+=2.*term/((k+1.)*(k+2.)*(k+3.));term*=-q0/(k+1.);
        }
        return 0;
    }
    if(q0==q1&&q0>=.5){
        double m0=-expm1(-q0)/q0,m1=(m0-t)/q0,m2=(2.*m1-t)/q0;
        out[0]=m2;out[1]=m1-m2;out[2]=m0-2.*m1+m2;
        return out[0]>=0&&out[1]>=0&&out[2]>=0 ? 0 : 2;
    }
#ifdef CMF_EXPQUAD_ENGINE
    if(!cmf_history_moments_engine(q0,q1,out))return 0;
#endif
    if(!thin_basis(q0,q1,out))return 0;
    if(!isfinite(q0-q1))return 2;
    /* 읽는 범위는 [0,count)뿐이고 추가 절점은 모두 쓰고 count를 늘린다.
     * 매 모멘트마다 사용하지 않는32KiB 전체를 초기화하지 않는다. */
    double cuts[LUMINA_CUT_CAPACITY],centers[3]={0.,1.,0.};int count=2,nc=2;
    cuts[0]=0.;cuts[1]=1.;
    if(q0!=q1){double vertex=-q1/(q0-q1);if(vertex>0&&vertex<1)centers[nc++]=vertex;}
    for(int k=0;k<nc;k++){
        double center=centers[k],scale=1./(1.+fabs(q1+(q0-q1)*center)+sqrt(fabs(q0-q1)));
        if(!isfinite(scale)||scale<=0)return 2;
        if(center>0&&center<1){if(count>=LUMINA_CUT_CAPACITY)return 2;cuts[count++]=center;}
        for(double step=scale;step<1.;step*=2.)for(int sign=-1;sign<=1;sign+=2){
            double value=center+sign*step;
            if(value>0&&value<1){if(count>=LUMINA_CUT_CAPACITY)return 2;cuts[count++]=value;}
        }
    }
#ifdef __CUDACC__
    for(int i=1;i<count;i++){double value=cuts[i];int j=i;while(j>0&&cuts[j-1]>value){cuts[j]=cuts[j-1];j--;}cuts[j]=value;}
#else
    qsort(cuts,count,sizeof(double),compare_double);
#endif
    double rough[3]={0.,0.,0.};int panels=0;
    for(int i=1;i<count;i++)if(cuts[i]>cuts[i-1]){
        double w[3];if(quadrature(q0,q1,cuts[i-1],cuts[i],w,s))return 2;
        for(int k=0;k<3;k++)rough[k]+=w[k];
        panels++;
    }
    if(!panels)return 2;
    double tol[3]={2e-12*rough[0]/panels,2e-12*rough[1]/panels,2e-12*rough[2]/panels};
    out[0]=out[1]=out[2]=0.;
    for(int i=1;i<count;i++)if(cuts[i]>cuts[i-1]){
        double w[3];if(panel(q0,q1,cuts[i-1],cuts[i],0,tol,w,s))return 2;
        for(int k=0;k<3;k++)out[k]+=w[k];
    }
    return 0;
}

LUMINA_POSITIVE int lumina_positive_map(double length,double normalization,double k0,double k1,
    double ea0,double ea1,double eb0,double eb1,CMFHistoryMap *out,CMFHistoryStats *stats)
{
    if(!out||!stats||!isfinite(length)||length<0||!isfinite(normalization)||normalization<=0||
       !isfinite(k0)||!isfinite(k1)||!isfinite(ea0)||ea0<0||!isfinite(ea1)||ea1<0||
       !isfinite(eb0)||eb0<0||!isfinite(eb1)||eb1<0)return 1;
    CMFHistoryMap m={0.,1.,0.,0.,0.,0.};if(length==0){*out=m;return 0;}
    double q0=k0*length,q1=k1*length;
    if(!isfinite(q0)||!isfinite(q1))return 2;
    m.d=length/normalization;m.t=exp(-(.5*q0+.5*q1));
    if(!isfinite(m.d)||!isfinite(m.t))return 2;
    if(ea0==0&&ea1==0&&eb0==0&&eb1==0){*out=m;return 0;}
    double w[3];if(basis(q0,q1,m.t,w,stats))return 2;
    double a0=length*ea0,a1=length*ea1,b0=length*eb0,b1=length*eb1;
    m.aa=a0*(w[0]+w[1])+a1*(w[1]+w[2]);
    m.ab=b0*(w[0]+w[1])+b1*(w[1]+w[2]);
    m.pa=m.d*(a0*w[0]+a1*w[1]);m.mb=m.d*(b0*w[1]+b1*w[2]);
    if(!valid(m))return 2;
    *out=m;return 0;
}


#undef LUMINA_POSITIVE
#undef LUMINA_QUAD_CONSTANT
#undef LUMINA_CUT_CAPACITY
#endif
