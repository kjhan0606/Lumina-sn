#ifndef CMF_WARP_MEASURE_CUH
#define CMF_WARP_MEASURE_CUH
#include "cmf_source_measure.hpp"

// 이 경로의 모든 집단 연산에는 같은 출력의32 lane 전부가 참여한다.
// 출력별 DFS/오차 예산을 섞지 않고, 큰 모멘트/다항식 배열만 분산한다.
namespace lumina_warp_measure {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::sum;
using lumina_ordered::product;
using lumina_characteristic::scaled;
using lumina_characteristic::mass;
using lumina_characteristic::mix;
using lumina_characteristic::less_equal;
using lumina_linear::Position;
constexpr unsigned mask=0xffffffffu;
static_assert(sizeof(lumina_characteristic::Pending)==24,"warp DFS shared budget ABI");
__device__ inline int lane(){return threadIdx.x&31;}
__device__ inline bool all(bool v){return __all_sync(mask,v);}
__device__ inline double shuffle(double v,int at){return __shfl_sync(mask,v,at);}
__device__ inline Positive shuffle(Positive p,int at){
    return {shuffle(p.mantissa,at),(std::int64_t)__shfl_sync(mask,(long long)p.exponent,at)};
}
__device__ inline bool reduce(Positive value,Positive &out){
    bool valid=true;
    for(int offset=16;offset;offset/=2){
        Positive other={__shfl_down_sync(mask,value.mantissa,offset),
            (std::int64_t)__shfl_down_sync(mask,(long long)value.exponent,offset)};
        if(lane()+offset<32&&!sum(value,other,value))valid=false;
    }
    out=shuffle(value,0);return all(valid);
}
__device__ inline double reduce(double value){
    for(int offset=16;offset;offset/=2){
        double other=__shfl_down_sync(mask,value,offset);
        if(lane()+offset<32)value+=other;
    }
    return shuffle(value,0);
}
struct Node {
    double width,field;int depth;
    __device__ double get(int trace,int component) const {return shuffle(field,8*trace+component);}
};
__device__ inline Node load(const lumina_linear::Node &n,int traces){
    int k=lane()/8,c=lane()%8;
    const char *at=reinterpret_cast<const char *>(&n.trace[k])+c*sizeof(double);
    return {n.width,k<traces?*reinterpret_cast<const double *>(at):0.,n.depth};
}
__device__ inline bool combine(const Node &a,const Node &b,int traces,Node &out){
    if(a.width==0){out=b;return true;}if(b.width==0){out=a;return true;}
    double width=a.width+b.width;
    if(!finite(width)||a.width<0||b.width<0)return false;
    int c=lane()%8,base=lane()/8*8;double l=a.width/width,r=b.width/width;
    double xq=shuffle(a.field,base+1),yp=shuffle(b.field,base),
        xm=shuffle(a.field,base+3),ym=shuffle(b.field,base+2);
    double middle=l*xq+r*yp,mag=l*xm+r*ym,v=0.;
    if(c==0)v=l*a.field+r*middle;
    else if(c==1)v=l*middle+r*b.field;
    else if(c==2)v=l*a.field+r*mag;
    else if(c==3)v=l*mag+r*b.field;
    else if(c==4||c==6)v=::fmin(a.field,b.field);
    else v=::fmax(a.field,b.field);
    if(!all(lane()>=8*traces||c>=4||finite(v)))return false;
    out={width,lane()<8*traces?v:0.,1+(a.depth>b.depth?a.depth:b.depth)};return true;
}
template<class Reader> __device__ bool leaf(const Reader &reader,int nf,int bin,double first,double last,
    double width,bool blue,int traces,Node &out,bool stationary=false){
    int k=lane()/8,c=lane()%8,point=2*(k%2);double xa,ea,xb,eb;
    bool valid=reader.at(point,bin,xa,ea);
    if(!blue&&(!stationary||first>0))valid=valid&&reader.at(point,bin+1,xb,eb);
    else {xb=xa;eb=ea;}
    if(!all(valid))return false;
    double a=k<2?xa:ea,b=k<2?xb:eb,s=blue||stationary?0.:b-a;
    double v0=a==b?a:(1.-first)*a+first*b,v1=a==b?a:(1.-last)*a+last*b;
    if(!all(finite(v0)&&finite(v1)&&finite(s)&&(k<2||(v0>=0&&v1>=0)))||!finite(width)||width<=0)return false;
    double v=0.;
    if(c==0)v=v0/3.+v1/6.;else if(c==1)v=v0/6.+v1/3.;
    else if(c==2)v=::fabs(v0)/3.+::fabs(v1)/6.;else if(c==3)v=::fabs(v0)/6.+::fabs(v1)/3.;
    else if(c==4)v=::fmin(v0,v1);else if(c==5)v=::fmax(v0,v1);else v=s;
    (void)nf;out={width,k<traces?v:0.,0};return true;
}
template<class Reader> __device__ bool query(const Reader &reader,const lumina_linear::Tree &tree,
    Position begin,Position end,int traces,Node &out){
    double width=lumina_linear::difference(begin,end);
    if(!finite(width)||width<=0)return false;
    Node whole={0.,0.,0},part,next;Position edge={tree.nf-1,0.};
    if(begin.bin>=tree.nf-1)return leaf(reader,tree.nf,tree.nf-1,0.,1.,width,true,traces,out);
    Position top=end.bin>=tree.nf-1?edge:end;
    int low=begin.bin,high=top.bin;
    if(low==high){
        if(top.fraction>begin.fraction&&!leaf(reader,tree.nf,low,begin.fraction,top.fraction,
            top.fraction-begin.fraction,false,traces,whole))return false;
    }else{
        if(begin.fraction>0){
            if(!leaf(reader,tree.nf,low,begin.fraction,1.,1.-begin.fraction,false,traces,whole))return false;
            low++;
        }
        while(low<high&&low%tree.block){
            if(!leaf(reader,tree.nf,low,0.,1.,1.,false,traces,part)||!combine(whole,part,traces,next))return false;
            whole=next;low++;
        }
        int stop=high-high%tree.block;
        if(stop>low){
            Node left={0.,0.,0},right={0.,0.,0};int a=low/tree.block+tree.leaves,b=stop/tree.block+tree.leaves;
            while(a<b){
                if(a&1){part=load(tree.nodes[a++],traces);if(!combine(left,part,traces,next))return false;left=next;}
                if(b&1){part=load(tree.nodes[--b],traces);if(!combine(part,right,traces,next))return false;right=next;}
                a/=2;b/=2;
            }
            if(!combine(left,right,traces,part)||!combine(whole,part,traces,next))return false;
            whole=next;low=stop;
        }
        while(low<high){
            if(!leaf(reader,tree.nf,low,0.,1.,1.,false,traces,part)||!combine(whole,part,traces,next))return false;
            whole=next;low++;
        }
        if(top.fraction>0){
            if(!leaf(reader,tree.nf,top.bin,0.,top.fraction,top.fraction,false,traces,part)||
               !combine(whole,part,traces,next))return false;
            whole=next;
        }
    }
    double extra=lumina_linear::difference(edge,end);
    if(extra>0){
        if(!leaf(reader,tree.nf,tree.nf-1,0.,1.,extra,true,traces,part)||!combine(whole,part,traces,next))return false;
        whole=next;
    }
    out=whole;return whole.width>0;
}
template<class Reader> struct Profile {
    Reader reader;lumina_linear::Tree tree;int column;double exit_shift,span,length,delta;
    __device__ bool position(double u,Position &p) const {return lumina_linear::position(column,::fma(span,u,exit_shift),p);}
    __device__ double coordinate_error(double u) const {return 8.*DBL_EPSILON*(1.+::fabs(exit_shift)+::fabs(span*u));}
    __device__ bool values(double u,double *v) const {Position p;return position(u,p)&&lumina_linear::at(reader,tree.nf,p,v);}
    __device__ bool range(double u0,double u1,Node &out,int traces=4) const {
        if(span==0){
            Position p;if(!position(0.,p))return false;
            int j=p.bin>=tree.nf-1?tree.nf-1:p.bin;
            double f=p.bin>=tree.nf-1?0.:p.fraction;
            return leaf(reader,tree.nf,j,f,f,1.,j==tree.nf-1,traces,out,true);
        }
        Position a,b;return position(u0,a)&&position(u1,b)&&query(reader,tree,a,b,traces,out);
    }
    __device__ bool tau(double u,lumina_linear::Integral &out) const {
        if(u==0){out={0.,0.};return true;}
        Node n;if(!range(0.,u,n,2))return false;
        double ap=n.get(0,0),aq=n.get(0,1),bp=n.get(1,0),bq=n.get(1,1),
            amp=n.get(0,2),amq=n.get(0,3),bmp=n.get(1,2),bmq=n.get(1,3);
        double value=0.*ap+u*aq+1.*bp+(1.-u)*bq;
        double mag=0.*amp+u*amq+1.*bmp+(1.-u)*bmq,scale=length*u,a=scale*value;
        bool nonzero=n.get(0,4)!=0||n.get(0,5)!=0||n.get(1,4)!=0||n.get(1,5)!=0;
        double operations=128.*(n.depth+2),e=scale*(mag*(operations*DBL_EPSILON)+(nonzero?operations*DBL_TRUE_MIN:0.));
        double slo0=n.get(0,6),shi0=n.get(0,7),slo1=n.get(1,6),shi1=n.get(1,7);
        double slope=::fmax(::fmax(::fabs(slo0),::fabs(shi0)),::fmax(::fabs(slo1),::fabs(shi1)));
        double relativistic=3.*delta*u,result=a+relativistic,
            error=e+length*u*slope*coordinate_error(u)+16.*DBL_EPSILON*(::fabs(a)+::fabs(relativistic));
        if(!finite(result)||!finite(error))return false;
        out={result,error};return true;
    }
};

template<class Reader> __device__ bool lower_panel(const Profile<Reader> &p,double u0,double u1,Positive &out){
    Node n;lumina_linear::Integral start;
    if(!p.range(u0,u1,n)||!p.tau(u0,start))return false;
    Positive a,b,m=Positive::zero(),error,quarter,absolute;
    double du=u1-u0,weights[4]={u0,u1,1.-u0,1.-u1};
    for(int k=0;k<4;k++){
        double value=n.get(2+k/2,k%2);
        if(!mass(p.length,value,a)||!scaled(a,du*weights[k],b)||!sum(m,b,m))return false;
    }
    double s0=n.get(2,6),s1=n.get(2,7),s2=n.get(3,6),s3=n.get(3,7);
    double operations=128.*(n.depth+2),source_slope=::fmax(::fmax(::fabs(s0),::fabs(s1)),::fmax(::fabs(s2),::fabs(s3)));
    double coordinate=p.span==0?0.:p.coordinate_error(u1);
    if(!scaled(m,operations*DBL_EPSILON,error)||!mass(p.length,du*operations*DBL_TRUE_MIN,absolute)||
       !sum(error,absolute,error)||!scaled(m,.25,quarter))return false;
    out=Positive::zero();if(!finite(source_slope*coordinate))return true;
    if(!mass(p.length,du*source_slope*coordinate,absolute)||!sum(error,absolute,error))return false;
    if(m.mantissa==0||!less_equal(error,quarter))return true;
    s0=n.get(0,6);s1=n.get(0,7);s2=n.get(1,6);s3=n.get(1,7);
    double slope=::fmax(::fmax(::fabs(s0),::fabs(s1)),::fmax(::fabs(s2),::fabs(s3))),
        m0=n.get(0,2),m1=n.get(0,3),m2=n.get(1,2),m3=n.get(1,3);
    double mag=u0*m0+u1*m1+(1.-u0)*m2+(1.-u1)*m3;
    double depth=p.length*du*(mag*(1.+operations*DBL_EPSILON)+slope*coordinate+operations*DBL_TRUE_MIN);
    double relative=::fmax(0.,3.*p.delta*du),upper=start.value+start.error+depth+relative;
    upper+=128.*DBL_EPSILON*(::fabs(start.value)+start.error+depth+relative+1.);
    if(!finite(upper)||!Positive::from_log(-upper,a))return true;
    return scaled(m,.5,b)&&product(a,b,out);
}
template<class Reader> __device__ bool source_lower(const Profile<Reader> &p,Positive &out){
    Positive total=Positive::zero(),part;
    for(int k=0;k<8;k++)if(!lower_panel(p,k/8.,(k+1)/8.,part)||!sum(total,part,total))return false;
    out=total;return true;
}

struct Polynomial {double coefficient,shift,tail;int degree;};
__device__ inline bool exponential(const double *phase,bool adaptive,Polynomial &out){
    double low=phase[0],high=phase[0];
    for(int k=0;k<4;k++){
        if(!finite(phase[k]))return false;
        low=::fmin(low,phase[k]);high=::fmax(high,phase[k]);
    }
    double width=high-low;if(!finite(width)||width>.125)return false;
    int order=8;double tail;
    if(adaptive){
        double e=::exp(width),remainder=width;order=0;
        while(order<8&&e*remainder>32.*DBL_EPSILON){++order;remainder*=width/(order+1.);}
        tail=e*remainder;
    }else tail=::exp(width)*::pow(width,9)/362880.;
    double h[4]={phase[0]-low,phase[1]-low,phase[2]-low,phase[3]-low};
    double a=lane()==0?1.:0.;int m=0;
    for(int level=order;level>=1;level--){
        double b=lane()<=m+3?1.:0.,denominator=(m+1.)*(m+2.)*(m+3.);
        // 한 대상 계수에 원래 추가되는 순서(k 오름차순)를 그대로 따른다.
        for(int j=3;j>=0;j--){
            int k=lane()-j;double x=shuffle(a,k<0?0:k);
            if(k<0||k>m)continue;
            double weight;
            if(j==0)weight=(m-k+1.)*(m-k+2.)*(m-k+3.)/denominator;
            else if(j==1)weight=3.*(k+1.)*(m-k+1.)*(m-k+2.)/denominator;
            else if(j==2)weight=3.*(k+1.)*(k+2.)*(m-k+1.)/denominator;
            else weight=(k+1.)*(k+2.)*(k+3.)/denominator;
            b+=x*(h[j]/level)*weight;
        }
        a=b;m+=3;
    }
    out={a,low,tail,m};return finite(tail);
}

template<class Reader> __device__ bool sample(const Profile<Reader> &p,Position at,Positive *out){
    return lumina_source_measure::sample(p.reader,p.tree.nf,at,out);
}
__device__ inline bool consume(const lumina_source_measure::Node &node,double t0,double t1,double u0,double u1,
    double length,const double *phase,double log_scale,bool constant,lumina_source_measure::Integral &out){
    double du=u1-u0,ua=::fma(du,t0,u0),ub=::fma(du,t1,u0),local[4];
    if(ua<0||ub>1||ua>=ub)return false;
    Polynomial p={lane()<25?1.:0.,0.,0.,24};
    if(!constant&&(!lumina_source_measure::restrict_cubic(phase,t0,t1,local)||!exponential(local,false,p)))return false;
    double previous=shuffle(p.coefficient,lane()>0?lane()-1:0);
    Positive contribution=Positive::zero(),x,y,total;bool valid=true;
    if(lane()<=25){
        int k=lane();double a=k<25?(25-k)*p.coefficient/25.:0.,b=k>0?k*previous/25.:0.;
        double wa=a*ua+b*ub,wb=a*(1.-ua)+b*(1.-ub);
        valid=scaled(node.m[0][k],wa,x)&&scaled(node.m[1][k],wb,y)&&sum(x,y,contribution);
    }
    if(!all(valid)||!reduce(contribution,total))return false;
    // 52항의 양의 균형 합산은 기존 finish의 경로당8192*(depth+3)eps
    // 보수 예산 안이다. signed χ의 합산 순서는 바꾸지 않는다.
    return lumina_source_measure::finish(total,node.depth,t1-t0,du,length,phase,log_scale,
        constant,p.shift,p.tail,out);
}
__device__ inline bool consume_linear(const Positive *a,const Positive *b,double t0,double t1,double u0,double u1,
    double length,const double *phase,double log_scale,bool constant,lumina_source_measure::Integral &out){
    double du=u1-u0,ua=::fma(du,t0,u0),ub=::fma(du,t1,u0),local[4],shift=0.,tail=0.;
    if(ua<0||ub>1||ua>=ub)return false;
    Positive e[3],x,y;
    if(!lumina_source_measure::positive_mix(ua,a[0],a[1],e[0])||
       !lumina_source_measure::positive_mix(ub,b[0],b[1],e[2])||
       !lumina_source_measure::positive_mix(ua,b[0],b[1],x)||
       !lumina_source_measure::positive_mix(ub,a[0],a[1],y)||
       !scaled(x,.5,x)||!scaled(y,.5,y)||!sum(x,y,e[1]))return false;
    double w[3]={1./3.,1./3.,1./3.};
    if(!constant){
        Polynomial p;
        if(!lumina_source_measure::restrict_cubic(phase,t0,t1,local)||!exponential(local,
#ifdef CMF_GPU_ADAPTIVE_LINEAR_EXP
            true,
#else
            false,
#endif
            p))return false;
        double m=p.degree,denominator=(m+1.)*(m+2.)*(m+3.),k=lane(),v=lane()<=p.degree?p.coefficient:0.;
        w[0]=reduce(v*(m-k+1.)*(m-k+2.)/denominator);
        w[1]=reduce(v*2.*(k+1.)*(m-k+1.)/denominator);
        w[2]=reduce(v*(k+1.)*(k+2.)/denominator);shift=p.shift;tail=p.tail;
    }
    Positive total=Positive::zero(),term;
    for(int k=0;k<3;k++)if(!scaled(e[k],w[k],term)||!sum(total,term,total))return false;
    return lumina_source_measure::finish(total,0,t1-t0,du,length,phase,log_scale,constant,shift,tail,out);
}

template<class Reader> __device__ bool integrate_range(const Profile<Reader> &p,const lumina_source_measure::Tree &tree,
    double u0,double u1,const double *phase,double log_scale,bool constant,lumina_source_measure::Integral &out){
    using lumina_linear::difference;
    if(!tree.nodes||tree.nf!=p.tree.nf||tree.leaves<1||(tree.leaves&(tree.leaves-1))||tree.block<1||tree.block>256)return false;
    lumina_source_measure::Integral total={Positive::zero(),Positive::zero(),0};Position begin,end;
    if(!p.position(u0,begin)||!p.position(u1,end))return false;
    if(p.span==0){
        Positive a[2];if(!sample(p,begin,a)||!lumina_warp_measure::consume_linear(a,a,0.,1.,u0,u1,p.length,phase,log_scale,constant,total))return false;
    }else{
        double width=difference(begin,end);if(width<=0)return false;
        auto part=[&](const lumina_source_measure::Node &node,Position a,Position b)->bool{
            return lumina_warp_measure::consume(node,difference(begin,a)/width,difference(begin,b)/width,u0,u1,p.length,phase,log_scale,constant,total);
        };
        auto bin=[&](Position a,Position b)->bool{
            Positive va[2],vb[2];
            return sample(p,a,va)&&sample(p,b,vb)&&lumina_warp_measure::consume_linear(va,vb,difference(begin,a)/width,
                difference(begin,b)/width,u0,u1,p.length,phase,log_scale,constant,total);
        };
        Position edge={tree.nf-1,0.},top=end.bin>=tree.nf-1?edge:end;
        int low=begin.bin,high=top.bin;
        if(begin.bin<tree.nf-1){
            if(low==high){if(!bin(begin,top))return false;}
            else{
                if(begin.fraction>0){if(!bin(begin,{low+1,0.}))return false;low++;}
                while(low<high&&low%tree.block){if(!bin({low,0.},{low+1,0.}))return false;low++;}
                int stop=high-high%tree.block;
                if(stop>low){
                    int a=low/tree.block+tree.leaves,b=stop/tree.block+tree.leaves;
                    while(a<b){
                        if(a&1){const auto &x=tree.nodes[a++];if(!part(x,{x.first,0.},{x.last,0.}))return false;}
                        if(b&1){const auto &x=tree.nodes[--b];if(!part(x,{x.first,0.},{x.last,0.}))return false;}
                        a/=2;b/=2;
                    }
                    low=stop;
                }
                while(low<high){if(!bin({low,0.},{low+1,0.}))return false;low++;}
                if(top.fraction>0&&!bin({top.bin,0.},top))return false;
            }
        }
        if(difference(edge,end)>0){
            Position a=begin.bin>=tree.nf-1?begin:edge;Positive v[2];
            if(!sample(p,a,v)||!lumina_warp_measure::consume_linear(v,v,difference(begin,a)/width,1.,u0,u1,p.length,phase,log_scale,constant,total))return false;
        }
    }
    Positive accumulation;
    if(!scaled(total.value,64.*(total.nodes+1)*DBL_EPSILON,accumulation)||!sum(total.error,accumulation,total.error))return false;
    out=total;return true;
}

struct Evaluate {
    lumina_source_measure::Tree source;
    template<class Reader> __device__ lumina_characteristic::PanelStatus operator()(const Profile<Reader> &p,
        double u0,double u1,double series_relative,lumina_characteristic::Panel &out) const {
        using namespace lumina_characteristic;
        lumina_warp_measure::Node n;lumina_linear::Integral tau;double a[4],b[4];
        if(!p.range(u0,u1,n)||!p.tau(u0,tau)||!p.values(u0,a)||!p.values(u1,b))return INVALID;
        double du=u1-u0,H=p.length*du,d3=3.*p.delta*du;
        if(!finite(H)||H<=0||!finite(d3))return INVALID;
        Panel result={Positive::zero(),Positive::zero(),Positive::zero(),Positive::zero()};
        double coordinate=p.span==0?0.:p.coordinate_error(u1),ce[2],cl[2];
        for(int k=0;k<2;k++){
            double lo=n.get(k,4),hi=n.get(k,5),slo=n.get(k,6),shi=n.get(k,7),slope=::fmax(::fabs(slo),::fabs(shi));
            double cr=64.*DBL_EPSILON*::fmax(::fabs(lo),::fabs(hi))+slope*coordinate;
            lumina_linear::Trace t={0.,0.,0.,0.,lo,hi,slo,shi};
            ce[k]=lumina_linear::secant_error(t,p.span==0?0.:n.width)+cr;cl[k]=lo-cr;
        }
        double lower=H*::fmin(mix(u0,cl[0],cl[1]),mix(u1,cl[0],cl[1]))+d3;
        double low_error=128.*DBL_EPSILON*(::fabs(lower)+::fabs(d3)+1.);
        double log_upper=-tau.value+tau.error+::fmax(0.,-lower+low_error);
        log_upper+=128.*DBL_EPSILON*(::fabs(log_upper)+1.);
        Positive base;double hi0=n.get(2,5),hi1=n.get(3,5);
#ifdef CMF_GPU_ENVELOPE_UPPER
        if(::fmax(hi0,hi1)>=1024.*DBL_MIN){
            if(!lumina_source_measure::envelope_upper(p.length,u0,u1,hi0,hi1,base))return INVALID;
        }else
#endif
        {
            lumina_source_measure::Integral mass_integral;double zero[4]={};
            if(!integrate_range(p,source,u0,u1,zero,0.,true,mass_integral)||!sum(mass_integral.value,mass_integral.error,base))return INVALID;
        }
        double s0=n.get(2,6),s1=n.get(2,7),s2=n.get(3,6),s3=n.get(3,7),
            slope=::fmax(::fmax(::fabs(s0),::fabs(s1)),::fmax(::fabs(s2),::fabs(s3)));
        Positive source_coordinate,attenuation;
        if(!mass(p.length,slope,source_coordinate)||!scaled(source_coordinate,du*coordinate,source_coordinate)||
           !sum(base,source_coordinate,base)||!Positive::from_log(log_upper,attenuation)||
           !product(base,attenuation,result.upper))return INVALID;
        out=result;if(result.upper.mantissa==0)return READY;
        double K[3]={mix(u0,a[0],a[1]),.5*mix(u0,b[0],b[1])+.5*mix(u1,a[0],a[1]),mix(u1,b[0],b[1])};
        double phase[4]={0.,-(H*(K[0]/3.)+d3/3.),-(H*(K[0]/3.+K[1]/3.)+2.*d3/3.),
            -(H*(K[0]/3.+K[1]/3.+K[2]/3.)+d3)};
        double low=0.,high=0.,construction=::fabs(d3);
        for(int k=0;k<3;k++)construction+=H*::fabs(K[k]/3.);
        for(int k=0;k<4;k++){if(!finite(phase[k]))return INVALID;low=::fmin(low,phase[k]);high=::fmax(high,phase[k]);}
        double spectral=H*::fmax(mix(u0,ce[0],ce[1]),mix(u1,ce[0],ce[1])),arithmetic=tau.error+128.*DBL_EPSILON*construction;
        if(!finite(spectral)||!finite(arithmetic))return INVALID;
        if(high-low>.125||spectral+arithmetic>.25)return SPLIT;
        lumina_source_measure::Integral integral;
        if(!integrate_range(p,source,u0,u1,phase,-tau.value,false,integral))return INVALID;
        Positive enclosed,first,second;
        if(!sum(integral.value,integral.error,enclosed)||!scaled(enclosed,series_relative,first))return INVALID;
        if(!less_equal(integral.error,first))return SPLIT;
        if(!scaled(enclosed,::expm1(arithmetic),first)||!sum(integral.error,first,result.roundoff)||
           !Positive::from_log(-tau.value+high+spectral+arithmetic,attenuation)||!product(source_coordinate,attenuation,second)||
           !sum(result.roundoff,second,result.roundoff)||!scaled(enclosed,::exp(arithmetic)*::expm1(spectral),result.frequency))return INVALID;
        result.value=integral.value;out=result;return READY;
    }
};

struct Stack {
    lumina_characteristic::Pending *values;
    struct Entry {
        lumina_characteristic::Pending *at;
        __device__ void operator=(lumina_characteristic::Pending value){
            if(lane()==0)*at=value;__syncwarp(mask);
        }
        __device__ operator lumina_characteristic::Pending() const {
            __syncwarp(mask);lumina_characteristic::Pending value=*at;__syncwarp(mask);return value;
        }
    };
    __device__ Entry operator[](int at) const {return {values+at};}
};
}
#endif
