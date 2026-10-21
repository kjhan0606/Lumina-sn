#ifndef CMF_CHARACTERISTIC_LINEAR_HPP
#define CMF_CHARACTERISTIC_LINEAR_HPP
#include "cmf_linear_moments.hpp"
#include "cmf_positive_cubic.hpp"
#ifdef __CUDACC__
#define LUMINA_CHAR __device__ inline
#else
#define LUMINA_CHAR inline
#endif

namespace lumina_characteristic {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::product;
using lumina_ordered::sum;
using lumina_linear::Node;

LUMINA_CHAR bool less_equal(Positive a,Positive b){
    if(a.mantissa==0)return true;
    if(b.mantissa==0)return false;
    return a.exponent<b.exponent||(a.exponent==b.exponent&&a.mantissa<=b.mantissa);
}
LUMINA_CHAR bool scaled(Positive a,double b,Positive &out){
    Positive p;return Positive::from_double(b,p)&&product(a,p,out);
}
LUMINA_CHAR bool mass(double length,double eta,Positive &out){
    Positive a,b;return Positive::from_double(length,a)&&Positive::from_double(eta,b)&&product(a,b,out);
}
LUMINA_CHAR double mix(double u,double a,double b){return u*a+(1.-u)*b;}

template<class Reader> struct NormalizedSource {
    const Reader *reader;double scale;
    LUMINA_CHAR bool at(int point,int column,double &chi,double &eta) const {
        if(!reader->at(point,column,chi,eta))return false;
        eta/=scale;return finite(eta);
    }
};

/* 공간선형 재구성 한 조각에만 적용한다. 비선형 반경/도플러 기하의
 * 조각화 오차와 생산 PHOT 오차 소비는 호출자의 별도 책임이다. */
template<class Reader> struct Profile {
    Reader reader;lumina_linear::Tree tree;
    int column;double exit_shift,span,length,delta;
    LUMINA_CHAR bool position(double u,lumina_linear::Position &p) const {
        return lumina_linear::position(column,::fma(span,u,exit_shift),p);
    }
    LUMINA_CHAR double coordinate_error(double u) const {
        return 8.*DBL_EPSILON*(1.+::fabs(exit_shift)+::fabs(span*u));
    }
    LUMINA_CHAR bool values(double u,double *v) const {
        lumina_linear::Position p;return position(u,p)&&lumina_linear::at(reader,tree.nf,p,v);
    }
    LUMINA_CHAR bool source_values(double u,double scale,double *v) const {
        lumina_linear::Position p;NormalizedSource<Reader> r={&reader,scale};
        return position(u,p)&&lumina_linear::at(r,tree.nf,p,v);
    }
    LUMINA_CHAR bool range(double u0,double u1,Node &out) const {
        if(span==0){double v[4],s[4]={};return values(0.,v)&&lumina_linear::leaf(1.,v,v,s,out);}
        lumina_linear::Position a,b;
        return position(u0,a)&&position(u1,b)&&lumina_linear::query(reader,tree,a,b,out);
    }
    LUMINA_CHAR bool tau(double u,lumina_linear::Integral &out) const {
        if(u==0){out={0.,0.};return true;}
        Node n;lumina_linear::Integral a;
        if(!range(0.,u,n)||!lumina_linear::path_integral(n,0,length,0.,u,a))return false;
        double slope=::fmax(::fmax(::fabs(n.trace[0].slo),::fabs(n.trace[0].shi)),
            ::fmax(::fabs(n.trace[1].slo),::fabs(n.trace[1].shi)));
        double relativistic=3.*delta*u,value=a.value+relativistic;
        double error=a.error+length*u*slope*coordinate_error(u)+
            16.*DBL_EPSILON*(::fabs(a.value)+::fabs(relativistic));
        if(!finite(value)||!finite(error))return false;
        out={value,error};return true;
    }
};

struct Panel {
    Positive value,roundoff,frequency,upper;
};
enum PanelStatus { INVALID=-1, SPLIT=0, READY=1 };

template<class Reader> LUMINA_CHAR PanelStatus panel(const Profile<Reader> &p,double u0,double u1,
    double series_relative,Panel &out)
{
    if(!finite(u0)||!finite(u1)||u0<0||u1>1||u0>=u1)return INVALID;
    Node n;lumina_linear::Integral tau;double a[4],b[4];
    if(!p.range(u0,u1,n)||!p.tau(u0,tau))return INVALID;
    double du=u1-u0,H=p.length*du,d3=3.*p.delta*du;
    if(!finite(H)||H<=0||!finite(d3))return INVALID;
    Panel result={Positive::zero(),Positive::zero(),Positive::zero(),Positive::zero()};
    double eta_scale=::fmax(n.trace[2].hi,n.trace[3].hi);
    bool varying=n.trace[2].slo!=0||n.trace[2].shi!=0||n.trace[3].slo!=0||n.trace[3].shi!=0;
    // subnormal의 변화량을 double 범위정보로 보증할 수 없으면 실패한다.
    // 일정한 subnormal은 먼저 정규화해 Lη 또는 gain으로 회복할 수 있다.
    if(eta_scale<1024.*DBL_MIN&&varying)return INVALID;
    if(eta_scale==0){out=result;return READY;}
    if(!p.source_values(u0,eta_scale,a)||!p.source_values(u1,eta_scale,b))return INVALID;
    double position_error=p.span==0?0.:p.coordinate_error(u1);
    double chi_error[2],eta_error[2],chi_lo[2],eta_hi[2];
    for(int k=0;k<2;k++){
        const auto &c=n.trace[k];auto e=n.trace[k+2];
        e.lo/=eta_scale;e.hi/=eta_scale;e.slo/=eta_scale;e.shi/=eta_scale;
        double cs=::fmax(::fabs(c.slo),::fabs(c.shi)),es=::fmax(::fabs(e.slo),::fabs(e.shi));
        double cr=64.*DBL_EPSILON*::fmax(::fabs(c.lo),::fabs(c.hi))+cs*position_error;
        double er=64.*DBL_EPSILON*e.hi+es*position_error;
        chi_error[k]=lumina_linear::secant_error(c,p.span==0?0.:n.width)+cr;
        eta_error[k]=lumina_linear::secant_error(e,p.span==0?0.:n.width)+er;
        chi_lo[k]=c.lo-cr;eta_hi[k]=e.hi+er;
    }
    double chi_min=::fmin(mix(u0,chi_lo[0],chi_lo[1]),mix(u1,chi_lo[0],chi_lo[1]));
    double eta_max=::fmax(mix(u0,eta_hi[0],eta_hi[1]),mix(u1,eta_hi[0],eta_hi[1]));
    double lower=H*chi_min+d3;
    double lower_error=64.*DBL_EPSILON*(::fabs(H*chi_min)+::fabs(d3));
    double log_upper=-tau.value+tau.error+::fmax(0.,-lower+lower_error);
    log_upper+=64.*DBL_EPSILON*(::fabs(tau.value)+tau.error+::fabs(lower)+lower_error+1.);
    Positive m,attenuation;
    if(!finite(eta_max)||!finite(log_upper)||!mass(H,eta_scale,m)||!scaled(m,eta_max,m)||
       !Positive::from_log(log_upper,attenuation)||!product(m,attenuation,result.upper))return INVALID;
    // 먼저 상계를 반환한다. caller는 원천을 버릴 때 이 전량을 오차로 센다.
    out=result;
    double K[3],E[3];
    K[0]=mix(u0,a[0],a[1]);K[2]=mix(u1,b[0],b[1]);
    K[1]=.5*mix(u0,b[0],b[1])+.5*mix(u1,a[0],a[1]);
    E[0]=mix(u0,a[2],a[3]);E[2]=mix(u1,b[2],b[3]);
    E[1]=.5*mix(u0,b[2],b[3])+.5*mix(u1,a[2],a[3]);
    double phase[4]={0.,-(H*(K[0]/3.)+d3/3.),
        -(H*(K[0]/3.+K[1]/3.)+2.*(d3/3.)),
        -(H*(K[0]/3.+K[1]/3.+K[2]/3.)+d3)};
    double low=0.,high=0.,construction=::fabs(d3);
    for(int k=0;k<3;k++)construction+=H*::fabs(K[k]/3.);
    for(int k=0;k<4;k++){
        if(!finite(phase[k]))return INVALID;
        low=::fmin(low,phase[k]);high=::fmax(high,phase[k]);
    }
    double dc=::fmax(mix(u0,chi_error[0],chi_error[1]),mix(u1,chi_error[0],chi_error[1]));
    double de=::fmax(mix(u0,eta_error[0],eta_error[1]),mix(u1,eta_error[0],eta_error[1]));
    double spectral=H*dc,arithmetic=tau.error+128.*DBL_EPSILON*construction;
    if(!finite(spectral)||!finite(arithmetic))return INVALID;
    if(high-low>.5||spectral+arithmetic>.25)return SPLIT;
    lumina_cubic::Integral integral;
    if(!lumina_cubic::integrate(H,E,phase,-tau.value,series_relative,integral))return SPLIT;
    if(!scaled(integral.value,eta_scale,integral.value)||!scaled(integral.error,eta_scale,integral.error))return INVALID;
    Positive first,second,third;
    Positive enclosed;
    if(!sum(integral.value,integral.error,enclosed)||
       !scaled(enclosed,::expm1(arithmetic),first)||!sum(integral.error,first,result.roundoff)||
       !scaled(enclosed,::exp(arithmetic)*::expm1(spectral),first)||
       !mass(H,eta_scale,second)||!scaled(second,de,second)||
       !Positive::from_log(-tau.value+high+arithmetic+spectral,third)||
       !product(second,third,second)||!sum(first,second,result.frequency))return INVALID;
    result.value=integral.value;out=result;return READY;
}

struct Result {
    Positive value,transmitted,source,error,roundoff,frequency,omitted;
    int panels,accepted,split,pruned,max_depth;
};
enum Status { OK=0, BAD_INPUT=1, NUMERIC_RANGE=2, WORK_LIMIT=3, PRECISION_LIMIT=4,
    SUBDIVISION_PRECISION_LIMIT=5, ACCUMULATION_PRECISION_LIMIT=6, TRANSMISSION_PRECISION_LIMIT=7 };

/* 물리 원천을 자르지 않고 오차 배분에 쓸 하한만 계산한다. 깊은 흡수선의
 * 최대χ를 전체 길이에 곱해 출구 쪽 방출선 예산까지 없애지 않는다. */
template<class Reader> LUMINA_CHAR bool source_lower_panel(const Profile<Reader> &p,double u0,double u1,Positive &out)
{
    Node n;lumina_linear::Integral start;
    if(!p.range(u0,u1,n)||!p.tau(u0,start))return false;
    Positive a,b,m,error,quarter,absolute;
    double du=u1-u0,weights[4]={u0,u1,1.-u0,1.-u1};m=Positive::zero();
    for(int k=0;k<4;k++){
        const auto &t=n.trace[2+k/2];double value=k%2?t.q:t.p;
        if(!mass(p.length,value,a)||!scaled(a,du*weights[k],b)||!sum(m,b,m))return false;
    }
    double operations=128.*(n.depth+2),source_slope=::fmax(
        ::fmax(::fabs(n.trace[2].slo),::fabs(n.trace[2].shi)),
        ::fmax(::fabs(n.trace[3].slo),::fabs(n.trace[3].shi)));
    double coordinate=p.span==0?0.:p.coordinate_error(u1);
    if(!scaled(m,operations*DBL_EPSILON,error)||!mass(p.length,du*operations*DBL_TRUE_MIN,absolute)||
       !sum(error,absolute,error)||!scaled(m,.25,quarter))return false;
    out=Positive::zero();
    if(!finite(source_slope*coordinate))return true;
    if(!mass(p.length,du*source_slope*coordinate,absolute)||!sum(error,absolute,error))return false;
    if(m.mantissa==0||!less_equal(error,quarter))return true;
    double slope=::fmax(::fmax(::fabs(n.trace[0].slo),::fabs(n.trace[0].shi)),
        ::fmax(::fabs(n.trace[1].slo),::fabs(n.trace[1].shi)));
    double mag=u0*n.trace[0].mag_p+u1*n.trace[0].mag_q+
        (1.-u0)*n.trace[1].mag_p+(1.-u1)*n.trace[1].mag_q;
    double depth=p.length*du*(mag*(1.+operations*DBL_EPSILON)+slope*coordinate+operations*DBL_TRUE_MIN);
    double relative=::fmax(0.,3.*p.delta*du),upper=start.value+start.error+depth+relative;
    upper+=128.*DBL_EPSILON*(::fabs(start.value)+start.error+depth+relative+1.);
    // 0은 여기서 안전한 하한이다. 실제 T/원천의 표현 범위 오류를 숨기는0은 아니다.
    if(!finite(upper)||!Positive::from_log(-upper,a))return true;
    if(!scaled(m,.5,b)||!product(a,b,out))return false;
    return true;
}

template<class Reader> LUMINA_CHAR bool source_lower(const Profile<Reader> &p,Positive &out)
{
    Positive total=Positive::zero(),part;
    // 공통 주파수 crossing mesh가 아닌, 하한만을 위한8개 서로 겹치지 않는 u 구간.
    for(int k=0;k<8;k++){
        if(!source_lower_panel(p,k/8.,(k+1)/8.,part)||!sum(total,part,total))return false;
    }
    out=total;return true;
}

/* 깊이 우선으로 출구 쪽부터 진행한다. η의 양수성으로 이미 얻은 원천을
 * 나머지 꼬리의 절대오차 예산에 쓸 수 있다. 전ν 공통 crossing mesh는 없다.
 * 성공 때만 out을 쓴다. 오차는 이 공간선형 문제에 대한 주파수별 상계다. */
template<class Reader,class PanelFunction> LUMINA_CHAR Status integrate_using(const Profile<Reader> &p,Positive incident,double relative,
    int max_panels,Result &out,const PanelFunction &evaluate)
{
    if(!finite(relative)||relative<=0||relative>.01||max_panels<1||p.tree.nf<2||p.tree.nf>2000000||
       p.column<0||p.column>=p.tree.nf||
       !finite(p.exit_shift)||p.exit_shift<0||!finite(p.span)||p.span<0||!finite(p.length)||p.length<=0||
       !finite(p.delta)||!finite(incident.mantissa)||incident.mantissa<0||
       (incident.mantissa!=0&&(incident.mantissa<.5||incident.mantissa>=1.)))return BAD_INPUT;
    lumina_linear::Integral tau;Positive transmission,first,second;
    if(!p.tau(1.,tau)||!Positive::from_log(-tau.value,transmission))return NUMERIC_RANGE;
    Result r={};
    if(!product(transmission,incident,r.transmitted))return NUMERIC_RANGE;
    if(r.transmitted.mantissa!=0){
        if(tau.error>.25||!scaled(r.transmitted,::expm1(tau.error),r.roundoff))return TRANSMISSION_PRECISION_LIMIT;
    }
    Positive seed;if(!source_lower(p,seed))return NUMERIC_RANGE;
    struct Pending {double u0,u1;int depth;};
    Pending stack[64];int count=1;stack[0]={0.,1.,0};
    const double local_relative=relative/8.;
    while(count){
        Pending q=stack[--count];
        if(r.panels>=max_panels)return WORK_LIMIT;
        ++r.panels;
        if(q.depth>r.max_depth)r.max_depth=q.depth;
        Panel a;PanelStatus status=evaluate(p,q.u0,q.u1,relative/32.,a);
        if(status==INVALID)return NUMERIC_RANGE;
        Positive scale,budget,errors;
        if(!sum(r.transmitted,r.source,scale)||!sum(scale,seed,scale)||
           !scaled(scale,local_relative*(q.u1-q.u0),budget))return NUMERIC_RANGE;
        if(less_equal(a.upper,budget)){
            if(!sum(r.omitted,a.upper,r.omitted))return NUMERIC_RANGE;
            r.pruned++;continue;
        }
        if(status==READY){
            // 줄일 수 있는 주파수 근사는 국소 예산으로 제어한다. 누적 τ의
            // 반올림은 du를 줄여도 상대적으로 줄지 않으므로 같은 출력ν의
            // 전체 수치 예산에서 소비한다. 다른ν의 밝기는 사용하지 않는다.
            if(!scaled(scale,q.u1-q.u0,first)||!sum(a.value,first,first)||
               !scaled(first,local_relative,budget)||!sum(r.roundoff,a.roundoff,errors)||
               !sum(a.value,scale,first)||!scaled(first,relative/4.,second))return NUMERIC_RANGE;
            if(less_equal(a.frequency,budget)&&less_equal(errors,second)){
                if(!sum(r.source,a.value,r.source)||!sum(r.roundoff,a.roundoff,r.roundoff)||
                   !sum(r.frequency,a.frequency,r.frequency))return NUMERIC_RANGE;
                r.accepted++;continue;
            }
        }
        double mid=.5*(q.u0+q.u1);
        if(q.depth>=60||count+2>64||mid<=q.u0||mid>=q.u1)return SUBDIVISION_PRECISION_LIMIT;
        stack[count++]={mid,q.u1,q.depth+1};stack[count++]={q.u0,mid,q.depth+1};r.split++;
    }
    if(!sum(r.transmitted,r.source,r.value))return NUMERIC_RANGE;
    // 양의 합의 누적 반올림도 따로 반환한다. 오차 예산 미달이면 성공하지 않는다.
    double operations=32.*(double(r.panels)+2.),gamma=operations*DBL_EPSILON;
    if(gamma>=.01)return ACCUMULATION_PRECISION_LIMIT;
    if(!scaled(r.value,gamma/(1.-gamma),first)||!sum(r.roundoff,first,r.roundoff)||
       !sum(r.roundoff,r.frequency,first)||!sum(first,r.omitted,r.error)||
       !scaled(r.value,relative,second))return NUMERIC_RANGE;
    if(!less_equal(r.error,second))return ACCUMULATION_PRECISION_LIMIT;
    out=r;return OK;
}

struct SecantPanel {
    template<class Reader> LUMINA_CHAR PanelStatus operator()(const Profile<Reader> &p,double u0,double u1,
        double relative,Panel &out) const {return panel(p,u0,u1,relative,out);}
};
// 기존 B의 기본 경로와 판정 기준은 그대로 보존한다.
template<class Reader> LUMINA_CHAR Status integrate(const Profile<Reader> &p,Positive incident,double relative,
    int max_panels,Result &out)
{
    return integrate_using(p,incident,relative,max_panels,out,SecantPanel{});
}
}
#undef LUMINA_CHAR
#endif
