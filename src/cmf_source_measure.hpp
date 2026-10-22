#ifndef CMF_SOURCE_MEASURE_HPP
#define CMF_SOURCE_MEASURE_HPP
#include "cmf_characteristic_linear.hpp"
#ifdef __CUDACC__
#define LUMINA_MEASURE __device__ inline
#else
#define LUMINA_MEASURE inline
#endif

namespace lumina_source_measure {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::sum;
using lumina_ordered::product;
using lumina_characteristic::scaled;
using lumina_linear::Position;
constexpr int degree=25;

/* 모멘트마다 독립 scale을 둔다. 작은 η를 큰 η의 공통 double scale로
 * 나눈 뒤0으로 잃지 않는다. width로 정규화한 Bernstein 적분이다. */
struct Node {Positive m[2][degree+1];double width;int first,last,depth;};
struct Tree {const Node *nodes;int leaves,block,nf;};
LUMINA_MEASURE Node empty(){Node n={};return n;}
LUMINA_MEASURE double choose(int n,int k){
    if(k<0||k>n)return 0.;
    if(k>n-k)k=n-k;
    double v=1.;for(int i=1;i<=k;i++)v=v*(n-k+i)/i;
    return v;
}
LUMINA_MEASURE bool leaf(double width,const Positive *a,const Positive *b,Node &out){
    if(!finite(width)||width<=0)return false;
    Node n={};n.width=width;
    for(int c=0;c<2;c++)for(int k=0;k<=degree;k++){
        Positive x,y;
        if(!scaled(a[c],double(degree-k+1)/((degree+1)*(degree+2)),x)||
           !scaled(b[c],double(k+1)/((degree+1)*(degree+2)),y)||!sum(x,y,n.m[c][k]))return false;
    }
    out=n;return true;
}
LUMINA_MEASURE bool combine(const Node &a,const Node &b,Node &out){
    if(a.width==0){out=b;return true;}if(b.width==0){out=a;return true;}
    double width=a.width+b.width;
    if(!finite(width)||a.width<0||b.width<0||a.last!=b.first)return false;
    double l=a.width/width,r=b.width/width,lp[degree+2],rp[degree+2];lp[0]=rp[0]=1.;
    for(int k=1;k<=degree+1;k++){lp[k]=lp[k-1]*l;rp[k]=rp[k-1]*r;}
    Node n={};n.width=width;n.first=a.first;n.last=b.last;n.depth=1+(a.depth>b.depth?a.depth:b.depth);
    for(int k=0;k<=degree;k++){
        for(int j=k;j<=degree;j++){
            double weight=choose(j,k)*lp[k+1]*rp[j-k];
            for(int c=0;c<2;c++){Positive v;if(!scaled(a.m[c][j],weight,v)||!sum(n.m[c][k],v,n.m[c][k]))return false;}
        }
        for(int j=0;j<=k;j++){
            double weight=choose(degree-j,k-j)*lp[k-j]*rp[degree-k+1];
            for(int c=0;c<2;c++){Positive v;if(!scaled(b.m[c][j],weight,v)||!sum(n.m[c][k],v,n.m[c][k]))return false;}
        }
    }
    out=n;return true;
}
template<class Reader> LUMINA_MEASURE bool sample(const Reader &r,int nf,Position p,Positive *out){
    if(nf<2||p.bin<0||!finite(p.fraction)||p.fraction<0||p.fraction>=1)return false;
    int j=p.bin>=nf-1?nf-1:p.bin;double f=p.bin>=nf-1?0.:p.fraction;
    for(int c=0;c<2;c++){
        double chi,eta;Positive a,b,x,y;
        if(!r.at(2*c,j,chi,eta)||!Positive::from_double(eta,a))return false;
        if(f==0){out[c]=a;continue;}
        if(!r.at(2*c,j+1,chi,eta)||!Positive::from_double(eta,b)||
           !scaled(a,1.-f,x)||!scaled(b,f,y)||!sum(x,y,out[c]))return false;
    }
    return true;
}
template<class Reader> LUMINA_MEASURE bool partial(const Reader &r,int nf,int bin,double first,double last,Node &out){
    if(bin<0||bin>=nf-1||first<0||last>1||first>=last)return false;
    Positive a[2],b[2];
    Position end=last==1.?Position{bin+1,0.}:Position{bin,last};
    if(!sample(r,nf,{bin,first},a)||!sample(r,nf,end,b)||!leaf(last-first,a,b,out))return false;
    out.first=bin;out.last=bin+1;return true;
}

/* cubic을 부분 구간으로 제한하는 양의 de Casteljau 조합이다.
 * 입력 계수는 signed일 수 있어 별도의 phase 산술 상계를 소비한다. */
LUMINA_MEASURE void split_cubic(const double *p,double t,double *left,double *right){
    double a[4]={p[0],p[1],p[2],p[3]};left[0]=a[0];right[3]=a[3];
    for(int n=3;n>0;n--){for(int k=0;k<n;k++)a[k]=(1.-t)*a[k]+t*a[k+1];left[4-n]=a[0];right[n-1]=a[n-1];}
}
LUMINA_MEASURE bool restrict_cubic(const double *p,double a,double b,double *out){
    if(!finite(a)||!finite(b)||a<0||b>1||a>=b)return false;
    double l[4],r[4],x[4];split_cubic(p,b,l,r);split_cubic(l,a/b,x,out);return true;
}
/* 1+H*(1+H/2*(...*(1+H/8)))는24차 Bernstein 다항식이다.
 * H의 계수도 비음수다. 함수 최솟값만으로 양성을 추정하지 않는다. */
LUMINA_MEASURE bool exponential_polynomial(const double *phase,double *coeff,double &shift,double &tail){
    double low=phase[0],high=phase[0];
    for(int k=0;k<4;k++){
        if(!finite(phase[k]))return false;
        low=::fmin(low,phase[k]);high=::fmax(high,phase[k]);
    }
    double width=high-low;if(width>.125)return false;
    double h[4]={phase[0]-low,phase[1]-low,phase[2]-low,phase[3]-low};
    double a[25]={1.},b[25];int m=0;
    for(int order=8;order>=1;order--){
        for(int k=0;k<=m+3;k++)b[k]=1.;
        double denominator=(m+1.)*(m+2.)*(m+3.);
        for(int k=0;k<=m;k++){
            double weights[4]={
                (m-k+1.)*(m-k+2.)*(m-k+3.)/denominator,
                3.*(k+1.)*(m-k+1.)*(m-k+2.)/denominator,
                3.*(k+1.)*(k+2.)*(m-k+1.)/denominator,
                (k+1.)*(k+2.)*(k+3.)/denominator};
            for(int j=0;j<4;j++)b[k+j]+=a[k]*(h[j]/order)*weights[j];
        }
        m+=3;for(int k=0;k<=m;k++)a[k]=b[k];
    }
    for(int k=0;k<25;k++)coeff[k]=a[k];
    shift=low;tail=::exp(width)*::pow(width,9)/362880.;return finite(tail);
}
struct Integral {Positive value,error;int nodes;};

/* pruning 전용 점별 상계다. 실제 감쇠 적분/원천 모멘트는 바꾸지 않는다.
 * 비음수 선형 보간의 hi와 공간 혼합을 쓴다. 보간/양의 산술의 경로는
 * 32회 미만이며256eps와 절대256*TRUE_MIN으로 보수적으로 감싼다.
 * 위치 불확실성의 slope 항과 감쇠 상계는 기존 Evaluate가 별도로 더한다. */
LUMINA_MEASURE bool envelope_upper(double length,double u0,double u1,double hi0,double hi1,Positive &out){
    if(!finite(length)||length<=0||!finite(u0)||!finite(u1)||u0<0||u1>1||u0>=u1||
       !finite(hi0)||!finite(hi1)||hi0<0||hi1<0)return false;
    Positive h[2],ends[2],x,y,l,du,scale,value,error,absolute;
    if(!Positive::from_double(hi0,h[0])||!Positive::from_double(hi1,h[1]))return false;
    double u[2]={u0,u1};
    for(int k=0;k<2;k++)
        if(!scaled(h[0],u[k],x)||!scaled(h[1],1.-u[k],y)||!sum(x,y,ends[k]))return false;
    Positive upper=lumina_characteristic::less_equal(ends[0],ends[1])?ends[1]:ends[0];
    if(!Positive::from_double(length,l)||!Positive::from_double(u1-u0,du)||!product(l,du,scale)||
       !product(scale,upper,value)||!scaled(value,256.*DBL_EPSILON/(1.-256.*DBL_EPSILON),error)||
       !Positive::from_double(DBL_TRUE_MIN,absolute)||!scaled(absolute,256.,absolute)||
       !product(scale,absolute,absolute)||!sum(value,error,x)||!sum(x,absolute,y))return false;
    out=y;return true;
}

LUMINA_MEASURE bool finish(Positive total,int depth,double fraction,double du,double length,const double *phase,
    double log_scale,bool constant,double shift,double tail,Integral &out){
    Positive l,h,s,scale,value,error;
    // length*du의 중간 underflow/overflow도 만들지 않는다.
    if(!Positive::from_double(length,l)||!Positive::from_double(du*fraction,h)||!product(l,h,scale)||
       !Positive::from_log(log_scale+shift,s)||!product(scale,s,scale)||!product(scale,total,value))return false;
    // 한 병합에서 출력 모멘트당 최대52개의 양의 항을 합한다. 계수의
    // 거듭제곱·이항계수·scale 산술을 포함한 경로당 상계다. prefix 상쇄는 없다.
    double operations=8192.*(depth+3),epsilon=operations*DBL_EPSILON;
    double phase_size=0.;if(!constant)for(int k=0;k<4;k++)phase_size+=::fabs(phase[k]);
    double exponent_error=64.*DBL_EPSILON*(::fabs(log_scale)+::fabs(shift)+phase_size+1.);
    if(!finite(exponent_error)||exponent_error>.01||epsilon>=.01)return false;
    double relative=::expm1(exponent_error)+epsilon/(1.-epsilon)+tail/(1.-tail);
    if(!scaled(value,relative,error)||!sum(out.value,value,out.value)||!sum(out.error,error,out.error))return false;
    out.nodes++;return true;
}

LUMINA_MEASURE bool consume(const Node &n,double t0,double t1,double u0,double u1,double length,
    const double *phase,double log_scale,bool constant,Integral &out){
    double du=u1-u0,ua=::fma(du,t0,u0),ub=::fma(du,t1,u0),p[25],local[4],shift=0.,tail=0.;
    if(ua<0||ub>1||ua>=ub)return false;
    if(constant){for(int k=0;k<25;k++)p[k]=1.;}
    else if(!restrict_cubic(phase,t0,t1,local)||!exponential_polynomial(local,p,shift,tail))return false;
    Positive total=Positive::zero(),term;
    for(int k=0;k<=degree;k++){
        double a=k<degree?(degree-k)*p[k]/degree:0.,b=k>0?k*p[k-1]/degree:0.;
        double wa=a*ua+b*ub,wb=a*(1.-ua)+b*(1.-ub);
        if(!scaled(n.m[0][k],wa,term)||!sum(total,term,total)||
           !scaled(n.m[1][k],wb,term)||!sum(total,term,total))return false;
    }
    return finish(total,n.depth,t1-t0,du,length,phase,log_scale,constant,shift,tail,out);
}

LUMINA_MEASURE bool positive_mix(double u,Positive a,Positive b,Positive &out){
    Positive x,y;return scaled(a,u,x)&&scaled(b,1.-u,y)&&sum(x,y,out);
}
/* 원래 한 bin 안의 η만 선형이다. 여러 꺾임을 하나의 secant로 만드는
 * 근사가 아니다. 정확한 quadratic 공간 원천과 감쇠 다항식을 직접 곱한다. */
LUMINA_MEASURE bool consume_linear(const Positive *a,const Positive *b,double t0,double t1,double u0,double u1,
    double length,const double *phase,double log_scale,bool constant,Integral &out){
    double du=u1-u0,ua=::fma(du,t0,u0),ub=::fma(du,t1,u0),p[25],local[4],shift=0.,tail=0.;
    if(ua<0||ub>1||ua>=ub)return false;
    Positive e[3],x,y;
    if(!positive_mix(ua,a[0],a[1],e[0])||!positive_mix(ub,b[0],b[1],e[2])||
       !positive_mix(ua,b[0],b[1],x)||!positive_mix(ub,a[0],a[1],y)||
       !scaled(x,.5,x)||!scaled(y,.5,y)||!sum(x,y,e[1]))return false;
    double w[3]={1./3.,1./3.,1./3.};
    if(!constant){
        if(!restrict_cubic(phase,t0,t1,local)||!exponential_polynomial(local,p,shift,tail))return false;
        w[0]=w[1]=w[2]=0.;
        constexpr double denominator=25.*26.*27.;
        for(int k=0;k<25;k++){
            w[0]+=p[k]*(25.-k)*(26.-k)/denominator;
            w[1]+=p[k]*2.*(k+1.)*(25.-k)/denominator;
            w[2]+=p[k]*(k+1.)*(k+2.)/denominator;
        }
    }
    Positive total=Positive::zero(),term;
    for(int k=0;k<3;k++)if(!scaled(e[k],w[k],term)||!sum(total,term,total))return false;
    return finish(total,0,t1-t0,du,length,phase,log_scale,constant,shift,tail,out);
}

/* query를 덮는 노드에서 위상만 제한한다. query마다25차 모멘트 전체를
 * 부모 좌표로 재병합하거나 원천의 native 꺾임으로 패널을 쪼개지 않는다. */
template<class Reader> LUMINA_MEASURE bool integrate_range(const lumina_characteristic::Profile<Reader> &p,
    const Tree &tree,double u0,double u1,const double *phase,double log_scale,bool constant,Integral &out){
    if(!tree.nodes||tree.nf!=p.tree.nf||tree.leaves<1||(tree.leaves&(tree.leaves-1))||tree.block<1||tree.block>256)return false;
    Integral total={Positive::zero(),Positive::zero(),0};
    Position begin,end;
    if(!p.position(u0,begin)||!p.position(u1,end))return false;
    if(p.span==0){
        Positive a[2];if(!sample(p.reader,tree.nf,begin,a)||
           !consume_linear(a,a,0.,1.,u0,u1,p.length,phase,log_scale,constant,total))return false;
    }else{
        double width=lumina_linear::difference(begin,end);if(width<=0)return false;
        auto part=[&](const Node &node,Position a,Position b)->bool {
            double t0=lumina_linear::difference(begin,a)/width,t1=lumina_linear::difference(begin,b)/width;
            return consume(node,t0,t1,u0,u1,p.length,phase,log_scale,constant,total);
        };
        auto original_bin=[&](Position a,Position b)->bool {
            Positive va[2],vb[2];
            double t0=lumina_linear::difference(begin,a)/width,t1=lumina_linear::difference(begin,b)/width;
            return sample(p.reader,tree.nf,a,va)&&sample(p.reader,tree.nf,b,vb)&&
                consume_linear(va,vb,t0,t1,u0,u1,p.length,phase,log_scale,constant,total);
        };
        Position top=end.bin>=tree.nf-1?Position{tree.nf-1,0.}:end;
        int low=begin.bin,high=top.bin;
        if(begin.bin<tree.nf-1){
            if(low==high){
                if(!original_bin(begin,top))return false;
            }else{
                if(begin.fraction>0){
                    if(!original_bin(begin,{low+1,0.}))return false;
                    low++;
                }
                while(low<high&&low%tree.block){
                    if(!original_bin({low,0.},{low+1,0.}))return false;
                    low++;
                }
                int stop=high-high%tree.block;
                if(stop>low){
                    int a=low/tree.block+tree.leaves,b=stop/tree.block+tree.leaves;
                    while(a<b){
                        if(a&1){const Node &x=tree.nodes[a++];if(!part(x,{x.first,0.},{x.last,0.}))return false;}
                        if(b&1){const Node &x=tree.nodes[--b];if(!part(x,{x.first,0.},{x.last,0.}))return false;}
                        a/=2;b/=2;
                    }
                    low=stop;
                }
                while(low<high){
                    if(!original_bin({low,0.},{low+1,0.}))return false;
                    low++;
                }
                if(top.fraction>0&&!original_bin({top.bin,0.},top))return false;
            }
        }
        if(lumina_linear::difference({tree.nf-1,0.},end)>0){
            Position a=begin.bin>=tree.nf-1?begin:Position{tree.nf-1,0.};Positive v[2];
            double t0=lumina_linear::difference(begin,a)/width;
            if(!sample(p.reader,tree.nf,a,v)||!consume_linear(v,v,t0,1.,u0,u1,p.length,phase,log_scale,constant,total))return false;
        }
    }
    Positive accumulation;
    if(!scaled(total.value,64.*(total.nodes+1)*DBL_EPSILON,accumulation)||
       !sum(total.error,accumulation,total.error))return false;
    out=total;return true;
}

struct Evaluate {
    Tree source;
    template<class Reader> LUMINA_MEASURE lumina_characteristic::PanelStatus operator()(
        const lumina_characteristic::Profile<Reader> &p,double u0,double u1,double series_relative,
        lumina_characteristic::Panel &out) const {
        using namespace lumina_characteristic;
        lumina_linear::Node n;lumina_linear::Integral tau;double a[4],b[4];
        if(!p.range(u0,u1,n)||!p.tau(u0,tau)||!p.values(u0,a)||!p.values(u1,b))return INVALID;
        double du=u1-u0,H=p.length*du,d3=3.*p.delta*du;
        if(!finite(H)||H<=0||!finite(d3))return INVALID;
        Panel result={Positive::zero(),Positive::zero(),Positive::zero(),Positive::zero()};
        double coordinate=p.span==0?0.:p.coordinate_error(u1),ce[2],cl[2];
        for(int k=0;k<2;k++){
            const auto &c=n.trace[k];double slope=::fmax(::fabs(c.slo),::fabs(c.shi));
            double cr=64.*DBL_EPSILON*::fmax(::fabs(c.lo),::fabs(c.hi))+slope*coordinate;
            ce[k]=lumina_linear::secant_error(c,p.span==0?0.:n.width)+cr;cl[k]=c.lo-cr;
        }
        double lower=H*::fmin(mix(u0,cl[0],cl[1]),mix(u1,cl[0],cl[1]))+d3;
        double low_error=128.*DBL_EPSILON*(::fabs(lower)+::fabs(d3)+1.);
        double log_upper=-tau.value+tau.error+::fmax(0.,-lower+low_error);
        log_upper+=128.*DBL_EPSILON*(::fabs(log_upper)+1.);
        Positive base;
#ifdef CMF_GPU_ENVELOPE_UPPER
        // tiny/정확한0 후보에서는 원 양의 모멘트 적분으로 되돌아간다.
        // double 보간 뒤0이 된 subnormal을 물리적0으로 오인하지 않는다.
        if(::fmax(n.trace[2].hi,n.trace[3].hi)>=1024.*DBL_MIN){
            if(!envelope_upper(p.length,u0,u1,n.trace[2].hi,n.trace[3].hi,base))return INVALID;
        }else
#endif
        {
            Integral mass_integral;double zero_phase[4]={};
            if(!integrate_range(p,source,u0,u1,zero_phase,0.,true,mass_integral)||
               !sum(mass_integral.value,mass_integral.error,base))return INVALID;
        }
        double slope=::fmax(::fmax(::fabs(n.trace[2].slo),::fabs(n.trace[2].shi)),
                            ::fmax(::fabs(n.trace[3].slo),::fabs(n.trace[3].shi)));
        Positive source_coordinate,attenuation;
        if(!mass(p.length,slope,source_coordinate)||!scaled(source_coordinate,du*coordinate,source_coordinate)||
           !sum(base,source_coordinate,base)||
           !Positive::from_log(log_upper,attenuation)||!product(base,attenuation,result.upper))return INVALID;
        out=result;
        if(result.upper.mantissa==0)return READY;
        double K[3]={mix(u0,a[0],a[1]),.5*mix(u0,b[0],b[1])+.5*mix(u1,a[0],a[1]),mix(u1,b[0],b[1])};
        double phase[4]={0.,-(H*(K[0]/3.)+d3/3.),-(H*(K[0]/3.+K[1]/3.)+2.*d3/3.),
            -(H*(K[0]/3.+K[1]/3.+K[2]/3.)+d3)};
        double low=0.,high=0.,construction=::fabs(d3);
        for(int k=0;k<3;k++)construction+=H*::fabs(K[k]/3.);
        for(int k=0;k<4;k++){if(!finite(phase[k]))return INVALID;low=::fmin(low,phase[k]);high=::fmax(high,phase[k]);}
        double spectral=H*::fmax(mix(u0,ce[0],ce[1]),mix(u1,ce[0],ce[1]));
        double arithmetic=tau.error+128.*DBL_EPSILON*construction;
        if(!finite(spectral)||!finite(arithmetic))return INVALID;
        if(high-low>.125||spectral+arithmetic>.25)return SPLIT;
        Integral integral;
        if(!integrate_range(p,source,u0,u1,phase,-tau.value,false,integral))return INVALID;
        Positive enclosed,first,second;
        if(!sum(integral.value,integral.error,enclosed)||!scaled(enclosed,series_relative,first))return INVALID;
        if(!less_equal(integral.error,first))return SPLIT;
        if(!scaled(enclosed,::expm1(arithmetic),first)||!sum(integral.error,first,result.roundoff)||
           !Positive::from_log(-tau.value+high+spectral+arithmetic,attenuation)||
           !product(source_coordinate,attenuation,second)||!sum(result.roundoff,second,result.roundoff)||
           !scaled(enclosed,::exp(arithmetic)*::expm1(spectral),result.frequency))return INVALID;
        result.value=integral.value;out=result;return READY;
    }
};
template<class Reader> LUMINA_MEASURE lumina_characteristic::Status integrate(
    const lumina_characteristic::Profile<Reader> &p,const Tree &source,Positive incident,double relative,
    int max_panels,lumina_characteristic::Result &out){
    return lumina_characteristic::integrate_using(p,incident,relative,max_panels,out,Evaluate{source});
}
}
#undef LUMINA_MEASURE
#endif
