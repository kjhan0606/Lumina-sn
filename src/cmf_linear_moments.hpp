#ifndef CMF_LINEAR_MOMENTS_HPP
#define CMF_LINEAR_MOMENTS_HPP
#include "cmf_ordered_maps.hpp"
#include <cmath>
#include <cstdint>
#include <cfloat>
#ifdef __CUDACC__
#define LUMINA_LINEAR __host__ __device__ inline
#define LUMINA_LINEAR_READER __device__ inline
#else
#define LUMINA_LINEAR inline
#define LUMINA_LINEAR_READER inline
#endif
namespace lumina_linear {
using lumina_ordered::finite;

/* p/q는 구간 폭으로 나눈 좌/우 barycentric 적분이다. x의 큰 원점을
 * 보관하거나 prefix끼리 빼지 않는다. mag는 실제 signed 상쇄의 오차 상계다. */
struct Trace {double p,q,mag_p,mag_q,lo,hi,slo,shi;};
struct Node {double width;Trace trace[4];int depth;}; // χ입구/χ출구/η입구/η출구

LUMINA_LINEAR Node empty(){Node n={};return n;}
LUMINA_LINEAR bool leaf(double width,const double *left,const double *right,const double *slopes,Node &out)
{
    if(!finite(width)||width<=0||!left||!right||!slopes)return false;
    Node n={};n.width=width;
    for(int k=0;k<4;k++){
        double a=left[k],b=right[k],s=slopes[k];
        if(!finite(a)||!finite(b)||!finite(s)||(k>=2&&(a<0||b<0)))return false;
        Trace t={a/3.+b/6.,a/6.+b/3.,::fabs(a)/3.+::fabs(b)/6.,::fabs(a)/6.+::fabs(b)/3.,
            ::fmin(a,b),::fmax(a,b),s,s};
        n.trace[k]=t;
    }
    out=n;return true;
}
LUMINA_LINEAR bool combine(const Node &a,const Node &b,Node &out)
{
    if(a.width==0){out=b;return true;}if(b.width==0){out=a;return true;}
    double width=a.width+b.width;
    if(!finite(width)||a.width<0||b.width<0)return false;
    double l=a.width/width,r=b.width/width;Node n={};n.width=width;n.depth=1+(a.depth>b.depth?a.depth:b.depth);
    for(int k=0;k<4;k++){
        const Trace &x=a.trace[k],&y=b.trace[k];Trace t;
        double middle=l*x.q+r*y.p,mag=l*x.mag_q+r*y.mag_p;
        t.p=l*x.p+r*middle;t.q=l*middle+r*y.q;
        t.mag_p=l*x.mag_p+r*mag;t.mag_q=l*mag+r*y.mag_q;
        t.lo=::fmin(x.lo,y.lo);t.hi=::fmax(x.hi,y.hi);t.slo=::fmin(x.slo,y.slo);t.shi=::fmax(x.shi,y.shi);
        if(!finite(t.p)||!finite(t.q)||!finite(t.mag_p)||!finite(t.mag_q))return false;
        n.trace[k]=t;
    }
    out=n;return true;
}

/* 출력 column의 큰 정수부와 작은 fraction을 분리해 유지한다. */
struct Position {int bin;double fraction;};
LUMINA_LINEAR bool position(int column,double relative,Position &out)
{
    if(column<0||!finite(relative)||relative<0||relative>2000000000.)return false;
    double whole=::floor(relative);if(whole>INT32_MAX-column)return false;
    out={column+static_cast<int>(whole),relative-whole};return true;
}
LUMINA_LINEAR double difference(Position a,Position b){return (double(b.bin)-double(a.bin))+(b.fraction-a.fraction);}

template<class Reader> LUMINA_LINEAR_READER bool at(const Reader &r,int nf,Position x,double *out)
{
    if(!out||x.bin<0||!finite(x.fraction)||x.fraction<0||x.fraction>=1)return false;
    int j=x.bin>=nf-1?nf-1:x.bin;double f=x.bin>=nf-1?0.:x.fraction;
    double a[4],b[4];
    if(!r.at(0,j,a[0],a[2])||!r.at(2,j,a[1],a[3]))return false;
    if(f>0){if(!r.at(0,j+1,b[0],b[2])||!r.at(2,j+1,b[1],b[3]))return false;}
    for(int k=0;k<4;k++){
        out[k]=f==0||a[k]==b[k]?a[k]:(1.-f)*a[k]+f*b[k];
        if(!finite(out[k])||(k>=2&&out[k]<0))return false;
    }
    return true;
}
template<class Reader> LUMINA_LINEAR_READER bool partial(const Reader &r,int nf,int bin,double first,double last,Node &out)
{
    if(bin<0||bin>=nf-1||!finite(first)||!finite(last)||first<0||last>1||first>=last)return false;
    double a[4],b[4],v0[4],v1[4],slopes[4];
    if(!r.at(0,bin,a[0],a[2])||!r.at(2,bin,a[1],a[3])||
       !r.at(0,bin+1,b[0],b[2])||!r.at(2,bin+1,b[1],b[3]))return false;
    for(int k=0;k<4;k++){
        slopes[k]=b[k]-a[k];
        v0[k]=a[k]==b[k]?a[k]:(1.-first)*a[k]+first*b[k];
        v1[k]=a[k]==b[k]?a[k]:(1.-last)*a[k]+last*b[k];
    }
    return leaf(last-first,v0,v1,slopes,out);
}
template<class Reader> LUMINA_LINEAR_READER bool blue(const Reader &r,int nf,double width,Node &out)
{
    double v[4],slopes[4]={};if(!at(r,nf,{nf-1,0.},v))return false;return leaf(width,v,v,slopes,out);
}

/* 각 잎은 block개의 연속 원래 bin이다. 경계 조각만 직접 계산한다. */
struct Tree {const Node *nodes;int leaves,block,nf;};
LUMINA_LINEAR bool node_range(const Tree &t,int low,int high,Node &out)
{
    if(!t.nodes||t.leaves<1||(t.leaves&(t.leaves-1))||low<0||high<low||high>t.leaves)return false;
    Node left=empty(),right=empty(),next;int a=low+t.leaves,b=high+t.leaves;
    while(a<b){
        if(a&1){if(!combine(left,t.nodes[a++],next))return false;left=next;}
        if(b&1){if(!combine(t.nodes[--b],right,next))return false;right=next;}
        a/=2;b/=2;
    }
    return combine(left,right,out);
}
template<class Reader> LUMINA_LINEAR_READER bool query(const Reader &r,const Tree &t,Position begin,Position end,Node &out)
{
    double width=difference(begin,end);
    if(t.nf<2||t.nf>2000000||t.block<1||t.block>256||!finite(width)||width<=0||begin.bin<0||
       begin.fraction<0||begin.fraction>=1||end.fraction<0||end.fraction>=1)return false;
    Node whole=empty(),part,next;Position blue_edge={t.nf-1,0.};
    if(begin.bin>=t.nf-1)return blue(r,t.nf,width,out);
    Position top=end.bin>=t.nf-1?blue_edge:end;
    int low=begin.bin,high=top.bin;
    if(low==high){
        if(top.fraction>begin.fraction){if(!partial(r,t.nf,low,begin.fraction,top.fraction,whole))return false;}
    }else{
        if(begin.fraction>0){
            if(!partial(r,t.nf,low,begin.fraction,1.,whole))return false;
            low++;
        }
        while(low<high&&low%t.block){
            if(!partial(r,t.nf,low,0.,1.,part)||!combine(whole,part,next))return false;
            whole=next;low++;
        }
        int stop=high-high%t.block;
        if(stop>low){
            if(!node_range(t,low/t.block,stop/t.block,part)||!combine(whole,part,next))return false;
            whole=next;low=stop;
        }
        while(low<high){
            if(!partial(r,t.nf,low,0.,1.,part)||!combine(whole,part,next))return false;
            whole=next;low++;
        }
        if(top.fraction>0){
            if(!partial(r,t.nf,top.bin,0.,top.fraction,part)||!combine(whole,part,next))return false;
            whole=next;
        }
    }
    double extra=difference(blue_edge,end);
    if(extra>0){if(!blue(r,t.nf,extra,part)||!combine(whole,part,next))return false;whole=next;}
    out=whole;return whole.width>0;
}

struct Integral {double value,error;};
LUMINA_LINEAR bool path_integral(const Node &node,int first,double length,double u0,double u1,Integral &out)
{
    if(first!=0&&first!=2)return false;
    if(!finite(length)||length<0||!finite(u0)||!finite(u1)||u0<0||u1>1||u0>u1)return false;
    const Trace &a=node.trace[first],&b=node.trace[first+1];
    double value=u0*a.p+u1*a.q+(1.-u0)*b.p+(1.-u1)*b.q;
    double mag=u0*a.mag_p+u1*a.mag_q+(1.-u0)*b.mag_p+(1.-u1)*b.mag_q;
    double scale=length*(u1-u0),v=scale*value;
    // subnormal에서 상대오차식만 적용하지 않는다. 주어진0 입력은 정확한0이다.
    bool nonzero=a.lo!=0||a.hi!=0||b.lo!=0||b.hi!=0;
    double operations=128.*(node.depth+2);
    double e=scale*(mag*(operations*DBL_EPSILON)+(nonzero?operations*DBL_TRUE_MIN:0.));
    if(!finite(v)||!finite(e))return false;
    out={v,e};return true;
}

/* 선형 secant와 원래 piecewise-linear 함수 사이의 보수적인 상계다.
 * 기울기 범위를 사용하므로 두 끝점이0인 좁은 방출선도 숨지 않는다. */
LUMINA_LINEAR double secant_error(const Trace &t,double width)
{
    double range=t.shi-t.slo;if(!finite(range))return INFINITY;
    return .25*width*range+64*DBL_EPSILON*(::fabs(t.lo)+::fabs(t.hi));
}
}
#undef LUMINA_LINEAR
#undef LUMINA_LINEAR_READER
#endif
