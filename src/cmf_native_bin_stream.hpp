#ifndef CMF_NATIVE_BIN_STREAM_HPP
#define CMF_NATIVE_BIN_STREAM_HPP
#include "cmf_source_measure.hpp"
#include "cmf_linear_affine.hpp"
#ifdef __CUDACC__
#define LUMINA_BIN __device__ inline
#else
#define LUMINA_BIN inline
#endif
namespace lumina_native_bin {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::sum;
using lumina_ordered::product;
using lumina_characteristic::scaled;
using lumina_characteristic::less_equal;
using lumina_linear::Position;
using lumina_linear_affine::Map;

struct Model {
    double phase[4],length,du,tau_error;
    Positive eta[3],coordinate;
};
LUMINA_BIN bool positive_lerp(Positive a,Positive b,double t,Positive &out){
    return lumina_source_measure::positive_mix(t,b,a,out);
}
LUMINA_BIN bool split_source(const Positive *p,double t,Positive *left,Positive *right){
    Positive a,b,c;
    if(!positive_lerp(p[0],p[1],t,a)||!positive_lerp(p[1],p[2],t,b)||!positive_lerp(a,b,t,c))return false;
    left[0]=p[0];left[1]=a;left[2]=c;right[0]=c;right[1]=b;right[2]=p[2];return true;
}
LUMINA_BIN bool restrict_source(const Positive *p,double a,double b,Positive *out){
    Positive left[3],right[3],unused[3];
    return b>0&&a>=0&&a<b&&b<=1&&split_source(p,b,left,right)&&split_source(left,a/b,unused,out);
}
LUMINA_BIN bool scale_length(Positive a,double length,double du,Positive &out){
    Positive l,h,v;
    return Positive::from_double(length,l)&&Positive::from_double(du,h)&&product(l,h,v)&&product(v,a,out);
}
// 국소 원천의 절대오차 예산만 나눈다. 물리 T/원천을 double로 변환하거나
// 다른 주파수의 밝기를 끌어오지 않는다. 반으로 줄여 나눗셈 반올림도 감싼다.
LUMINA_BIN bool divide_budget(Positive budget,Positive response,Positive &out){
    constexpr int64_t limit=INT64_C(1)<<60;
    if(response.mantissa<.5||response.mantissa>=1.||budget.mantissa<0||
       budget.mantissa>=1.||budget.exponent<=-limit||budget.exponent>=limit||
       response.exponent<=-limit||response.exponent>=limit)return false;
    if(budget.mantissa==0){out=Positive::zero();return true;}
    Positive quotient;
    if(!Positive::from_double(.5*(budget.mantissa/response.mantissa),quotient))return false;
    quotient.exponent+=budget.exponent-response.exponent;
    if(quotient.exponent<=-limit||quotient.exponent>=limit)return false;
    out=quotient;return true;
}
template<class Reader> LUMINA_BIN bool model(const lumina_characteristic::Profile<Reader> &p,
    Position a,Position b,double u0,double u1,Model &out){
    if(u0<0||u1>1||u0>=u1)return false;
    double va[4],vb[4];Positive ea[2],eb[2];lumina_linear::Node n;
    if(!lumina_linear::at(p.reader,p.tree.nf,a,va)||!lumina_linear::at(p.reader,p.tree.nf,b,vb)||
       !lumina_source_measure::sample(p.reader,p.tree.nf,a,ea)||
       !lumina_source_measure::sample(p.reader,p.tree.nf,b,eb))return false;
    if(p.span==0){double slopes[4]={};if(!lumina_linear::leaf(1.,va,va,slopes,n))return false;}
    else if(a.bin>=p.tree.nf-1){if(!lumina_linear::blue(p.reader,p.tree.nf,1.,n))return false;}
    else{
        if(b.bin!=a.bin&&!(b.bin==a.bin+1&&b.fraction==0))return false;
        if(!lumina_linear::partial(p.reader,p.tree.nf,a.bin,a.fraction,b.bin==a.bin?b.fraction:1.,n))return false;
    }
    double du=u1-u0,H=p.length*du,d3=3.*p.delta*du;
    if(!finite(H)||H<=0||!finite(d3))return false;
    using lumina_characteristic::mix;
    double K[3]={mix(u0,va[0],va[1]),.5*mix(u0,vb[0],vb[1])+.5*mix(u1,va[0],va[1]),mix(u1,vb[0],vb[1])};
    Model m={};m.length=p.length;m.du=du;
    m.phase[0]=0.;m.phase[1]=-(H*(K[0]/3.)+d3/3.);
    m.phase[2]=-(H*(K[0]/3.+K[1]/3.)+2.*d3/3.);
    m.phase[3]=-(H*(K[0]/3.+K[1]/3.+K[2]/3.)+d3);
    double coordinate=p.span==0?0.:p.coordinate_error(u1),bound=0.,construction=::fabs(d3);
    for(int k=0;k<2;k++){
        const auto &t=n.trace[k];double slope=::fmax(::fabs(t.slo),::fabs(t.shi));
        double error=64.*DBL_EPSILON*::fmax(::fabs(t.lo),::fabs(t.hi))+slope*coordinate;
        bound=::fmax(bound,error);
    }
    for(int k=0;k<3;k++)construction+=H*::fabs(K[k]/3.);
    m.tau_error=H*bound+256.*DBL_EPSILON*(construction+::fabs(d3));
    if(!finite(m.tau_error)||m.tau_error>.25)return false;
    for(int k=0;k<4;k++)if(!finite(m.phase[k]))return false;
    Positive x,y;
    using lumina_source_measure::positive_mix;
    if(!positive_mix(u0,ea[0],ea[1],m.eta[0])||!positive_mix(u1,eb[0],eb[1],m.eta[2])||
       !positive_mix(u0,eb[0],eb[1],x)||!positive_mix(u1,ea[0],ea[1],y)||
       !scaled(x,.5,x)||!scaled(y,.5,y)||!sum(x,y,m.eta[1]))return false;
    double slope=::fmax(::fmax(::fabs(n.trace[2].slo),::fabs(n.trace[2].shi)),
                        ::fmax(::fabs(n.trace[3].slo),::fabs(n.trace[3].shi)));
    if(!Positive::from_double(slope,x)||!scaled(x,coordinate,x)||!scale_length(x,p.length,du,m.coordinate))return false;
    out=m;return true;
}

LUMINA_BIN bool polynomial_integral(const Model &m,const Positive *eta,const double *phase,
    double fraction,Positive &value,Positive &error){
    double p[25],shift,tail;int degree;
    if(!lumina_source_measure::exponential_polynomial_adaptive(phase,p,degree,shift,tail))return false;
    double w[3]={},denominator=(degree+1.)*(degree+2.)*(degree+3.);
    for(int k=0;k<=degree;k++){
        w[0]+=p[k]*(degree-k+1.)*(degree-k+2.)/denominator;
        w[1]+=p[k]*2.*(k+1.)*(degree-k+1.)/denominator;
        w[2]+=p[k]*(k+1.)*(k+2.)/denominator;
    }
    Positive total=Positive::zero(),term;
    for(int k=0;k<3;k++)if(!scaled(eta[k],w[k],term)||!sum(total,term,total))return false;
    lumina_source_measure::Integral result={Positive::zero(),Positive::zero(),0};
    if(!lumina_source_measure::finish(total,0,fraction,m.du,m.length,phase,0.,false,shift,tail,result))return false;
    value=result.value;error=result.error;return true;
}

// 한 native bin 안에서도 두꺼운 흡수/이득은 나눈다. 위상은 이미 정확한
// cubic이므로 누적 광학깊이/범위 트리를 패널마다 다시 질의하지 않는다.
LUMINA_BIN lumina_characteristic::Status one_bin(const Model &m,double relative,Positive absolute_budget,int limit,
    lumina_characteristic::Result &out,Map &map){
    using namespace lumina_characteristic;
    Result r={};Positive mass=Positive::zero(),part,seed=Positive::zero(),transmission;
    double minimum=m.phase[0];
    for(int k=0;k<3;k++)if(!scaled(m.eta[k],1./3.,part)||!sum(mass,part,mass))return NUMERIC_RANGE;
    if(!scale_length(mass,m.length,m.du,mass)||!Positive::from_log(m.phase[3],transmission))return NUMERIC_RANGE;
    for(int k=1;k<4;k++)minimum=::fmin(minimum,m.phase[k]);
    // 표현 밖의 이 값은 물리 원천이 아니라 예산 하한이다. 안전한0 하한만 사용한다.
    Positive attenuation,quarter,lower;
    if(!scaled(mass,.25,quarter)||!scaled(mass,.5,lower))return NUMERIC_RANGE;
    if(less_equal(m.coordinate,quarter)&&Positive::from_log(minimum-m.tau_error,attenuation)&&
       !product(lower,attenuation,seed))return NUMERIC_RANGE;
    struct Pending {double a,b;int depth;};Pending stack[64];int count=1;stack[0]={0.,1.,0};
    while(count){
        Pending q=stack[--count];if(r.panels>=limit)return WORK_LIMIT;r.panels++;
        if(r.max_depth<q.depth)r.max_depth=q.depth;
        double phase[4],width=q.b-q.a,high;Positive eta[3];
        if(!lumina_source_measure::restrict_cubic(m.phase,q.a,q.b,phase)||
           !restrict_source(m.eta,q.a,q.b,eta))return NUMERIC_RANGE;
        high=phase[0];double low=high;
        for(int k=1;k<4;k++){high=::fmax(high,phase[k]);low=::fmin(low,phase[k]);}
        Positive total=Positive::zero(),upper,coordinate,enclosed,scale,budget,error;
        for(int k=0;k<3;k++)if(!scaled(eta[k],1./3.,part)||!sum(total,part,total))return NUMERIC_RANGE;
        if(!scale_length(total,m.length,m.du*width,total)||!scaled(total,1.+512.*DBL_EPSILON,total)||
           !scaled(m.coordinate,width,coordinate)||!sum(total,coordinate,upper)||
           !Positive::from_log(high+m.tau_error+256.*DBL_EPSILON*(::fabs(high)+1.),attenuation)||
           !product(upper,attenuation,upper)||!sum(r.source,seed,scale)||
           !scaled(scale,relative/8.*width,budget)||!scaled(absolute_budget,width/8.,part)||
           !sum(budget,part,budget))return NUMERIC_RANGE;
        if(less_equal(upper,budget)){if(!sum(r.omitted,upper,r.omitted))return NUMERIC_RANGE;r.pruned++;continue;}
        bool ready=high-low<=.125;
        Positive value,roundoff;
        if(ready){
            if(!polynomial_integral(m,eta,phase,width,value,roundoff)||!sum(value,roundoff,enclosed)||
               !scaled(enclosed,::expm1(m.tau_error),part)||!sum(roundoff,part,roundoff)||
               !product(coordinate,attenuation,part)||!sum(roundoff,part,roundoff)||
               !sum(r.roundoff,roundoff,error)||!sum(scale,value,part)||!scaled(part,relative/4.,budget)||
               !scaled(absolute_budget,.25,part)||!sum(budget,part,budget))return NUMERIC_RANGE;
            if(less_equal(error,budget)){
                if(!sum(r.source,value,r.source)||!sum(r.roundoff,roundoff,r.roundoff))return NUMERIC_RANGE;
                r.accepted++;continue;
            }
        }
        double middle=.5*(q.a+q.b);
        if(q.depth>=60||count+2>64||middle<=q.a||middle>=q.b)return SUBDIVISION_PRECISION_LIMIT;
        stack[count++]={middle,q.b,q.depth+1};stack[count++]={q.a,middle,q.depth+1};r.split++;
    }
    double gamma=32.*(r.panels+2.)*DBL_EPSILON;
    if(gamma>=.01||!scaled(r.source,gamma/(1.-gamma),part)||!sum(r.roundoff,part,r.roundoff)||
       !sum(r.roundoff,r.omitted,r.error))return ACCUMULATION_PRECISION_LIMIT;
    r.value=r.source;
    map={transmission,r.source,r.error,m.tau_error};out=r;return OK;
}

template<class Reader> LUMINA_BIN lumina_characteristic::Status integrate(
    const lumina_characteristic::Profile<Reader> &p,Positive incident,double relative,int max_panels,
    lumina_characteristic::Result &out){
    using namespace lumina_characteristic;
    if(!finite(relative)||relative<=0||relative>.01||max_panels<1||p.tree.nf<2||p.tree.nf>2000000||
       p.column<0||p.column>=p.tree.nf||!finite(p.exit_shift)||p.exit_shift<0||!finite(p.span)||p.span<0||
       !finite(p.length)||p.length<=0||!finite(p.delta)||!finite(incident.mantissa)||incident.mantissa<0||
       (incident.mantissa!=0&&(incident.mantissa<.5||incident.mantissa>=1.)))return BAD_INPUT;
    Position first,last,a;if(!p.position(0.,first)||!p.position(1.,last))return NUMERIC_RANGE;a=first;
    Positive global_seed;
    if(!source_lower(p,global_seed))return NUMERIC_RANGE;
    Result r={};Map accumulated={Positive::one(),Positive::zero(),Positive::zero(),0.};double u0=0.;
    while(u0<1.){
        Position b=last;double u1=1.;
        if(p.span>0&&a.bin<p.tree.nf-1&&lumina_linear::difference({a.bin+1,0.},last)>0){
            b={a.bin+1,0.};u1=lumina_linear::difference(first,b)/p.span;
        }
        if(!finite(u1)||u1<=u0||u1>1.)return SUBDIVISION_PRECISION_LIMIT;
        Model m;if(!model(p,a,b,u0,u1,m))return NUMERIC_RANGE;
        Positive scale,response,output_budget,absolute_budget;
        if(!sum(accumulated.source,global_seed,scale)||!accumulated.upper(response)||
           !scaled(scale,relative/64.*m.du,output_budget)||
           !divide_budget(output_budget,response,absolute_budget))return NUMERIC_RANGE;
        Result local;Map upstream,next;
        int code=one_bin(m,relative/16.,absolute_budget,max_panels-r.panels,local,upstream);
        if(code)return static_cast<Status>(code);
        // u 증가순으로 만나지만 photon 진행순서는 상류→하류다.
        if(!lumina_linear_affine::compose(upstream,accumulated,next))return NUMERIC_RANGE;
        Positive upper,x,y,z,roundoff,omitted;
        if(!accumulated.upper(upper)||!product(upper,local.roundoff,x)||!sum(x,r.roundoff,x)||
           !product(accumulated.t,upstream.source,y)||!scaled(y,::expm1(accumulated.tau_error),y)||
           !sum(x,y,z)||!scaled(next.source,32.*DBL_EPSILON,x)||!sum(z,x,roundoff)||
           !product(upper,local.omitted,x)||!sum(x,r.omitted,omitted)||!sum(roundoff,omitted,next.error))return NUMERIC_RANGE;
        r.roundoff=roundoff;r.omitted=omitted;accumulated=next;
        r.panels+=local.panels;r.accepted+=local.accepted;r.split+=local.split;r.pruned+=local.pruned;
        if(r.max_depth<local.max_depth)r.max_depth=local.max_depth;
        a=b;u0=u1;
    }
    // 한 번의 원 τ 질의는 T의 원 표현/오차와 그대로 대조된다.
    lumina_linear::Integral tau;Positive transmission,part,tolerance;
    if(!p.tau(1.,tau)||!Positive::from_log(-tau.value,transmission)||
       !product(transmission,incident,r.transmitted)||!sum(r.transmitted,accumulated.source,r.value))return NUMERIC_RANGE;
    r.source=accumulated.source;
    if(tau.error>.25||!scaled(r.transmitted,::expm1(tau.error),part)||!sum(r.roundoff,part,r.roundoff)||
       !scaled(r.value,32.*DBL_EPSILON,part)||!sum(r.roundoff,part,r.roundoff)||
       !sum(r.roundoff,r.omitted,r.error)||!scaled(r.value,relative,tolerance))return NUMERIC_RANGE;
    if(!less_equal(r.error,tolerance))return ACCUMULATION_PRECISION_LIMIT;
    out=r;return OK;
}
}
#undef LUMINA_BIN
#endif
