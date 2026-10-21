#ifndef CMF_ORDERED_MAPS_HPP
#define CMF_ORDERED_MAPS_HPP
#include <cmath>
#include <cfloat>
#include <cstdint>
#ifdef __CUDACC__
#define LUMINA_HD __host__ __device__
#else
#define LUMINA_HD
#endif

namespace lumina_ordered {
LUMINA_HD inline bool finite(double x){return x==x&&x<=DBL_MAX&&x>=-DBL_MAX;}

/* 서로 다른 크기의 양의 계수를 공통 scale 하나로 뭉개지 않는다.
 * mantissa는 [0.5,1), exponent는 정확한 정수다. 0은 실제0만 표시한다. */
struct Positive {
    double mantissa;
    std::int64_t exponent;
    LUMINA_HD static Positive zero(){return {0.,0};}
    LUMINA_HD static Positive one(){return {.5,1};}
    LUMINA_HD static bool from_double(double value,Positive &out){
        if(!finite(value)||value<0)return false;
        int exponent=0;double mantissa=::frexp(value,&exponent);
        out={mantissa,exponent};return true;
    }
    LUMINA_HD static bool from_log(double value,Positive &out){
        if(!finite(value))return false;
        const double hi=0x1.62e42fefa39efp-1,lo=2.31904681384629955842e-17;
        double power=::floor(value/hi)+1.;
        /* 표현 범위 밖은 거부한다. 물리값이나 logT를 잘라내지 않는다. */
        if(power<=-0x1p52||power>=0x1p52)return false;
        double remainder=::fma(-power,hi,value)-power*lo;
        double mantissa=::exp(remainder);int adjustment=0;
        mantissa=::frexp(mantissa,&adjustment);
        if(!finite(mantissa)||mantissa<=0)return false;
        out={mantissa,static_cast<std::int64_t>(power)+adjustment};return true;
    }
    LUMINA_HD double value() const {
        if(mantissa==0)return 0.;
        /* IEEE binary64로의 마지막 변환이다. 중간 계수에는 적용하지 않는다. */
        if(exponent < -1074)return 0.;
        if(exponent > 1024)return INFINITY;
        return ::ldexp(mantissa,static_cast<int>(exponent));
    }
};

LUMINA_HD inline bool product(Positive a,Positive b,Positive &out){
    if(a.mantissa==0||b.mantissa==0){out=Positive::zero();return true;}
    constexpr std::int64_t limit=INT64_C(1)<<60;
    if(a.exponent<=-limit||a.exponent>=limit||b.exponent<=-limit||b.exponent>=limit)return false;
    double value=a.mantissa*b.mantissa;std::int64_t exponent=a.exponent+b.exponent;
    if(value<.5){value*=2;exponent--;}
    if(exponent<=-limit||exponent>=limit)return false;
    out={value,exponent};return true;
}

LUMINA_HD inline bool sum(Positive a,Positive b,Positive &out){
    if(a.mantissa==0){out=b;return true;}if(b.mantissa==0){out=a;return true;}
    if(a.exponent<b.exponent){Positive t=a;a=b;b=t;}
    std::int64_t difference=b.exponent-a.exponent;
    double small=difference < -1074 ? 0. : ::ldexp(b.mantissa,static_cast<int>(difference));
    double value=a.mantissa+small;std::int64_t exponent=a.exponent;
    if(value>=1){value*=.5;exponent++;}
    constexpr std::int64_t limit=INT64_C(1)<<60;
    if(exponent<=-limit||exponent>=limit)return false;
    out={value,exponent};return true;
}

struct Map {
    double d;
    Positive t,aa,pa,ab,mb;
    LUMINA_HD static Map identity(){
        return {0.,Positive::one(),Positive::zero(),Positive::zero(),Positive::zero(),Positive::zero()};
    }
    LUMINA_HD static bool from_values(const double *v,double log_t,Map &out){
        if(!v||!finite(v[0])||v[0]<0||!finite(v[1])||v[1]<0)return false;
        out.d=v[0];
        return Positive::from_log(log_t,out.t)&&Positive::from_double(v[2],out.aa)&&
            Positive::from_double(v[3],out.pa)&&Positive::from_double(v[4],out.ab)&&Positive::from_double(v[5],out.mb);
    }
};

/* left가 먼저, right가 나중이다. 이 비가환 순서를 트리에서도 유지한다. */
LUMINA_HD inline bool compose(const Map &left,const Map &right,Map &out){
    Map result=Map::identity();result.d=left.d+right.d;
    if(!finite(result.d))return false;
    Positive a,b,c,dl,dr;
    if(!Positive::from_double(left.d,dl)||!Positive::from_double(right.d,dr)||
       !product(left.t,right.t,result.t))return false;
    if(!product(right.t,left.aa,a)||!sum(a,right.aa,result.aa))return false;
    if(!product(right.t,left.ab,a)||!sum(a,right.ab,result.ab))return false;
    if(!product(dr,left.aa,a)||!sum(left.pa,a,b)||!product(right.t,b,c)||!sum(c,right.pa,result.pa))return false;
    if(!product(right.t,left.mb,a)||!product(dl,right.ab,b)||!sum(right.mb,b,c)||!sum(a,c,result.mb))return false;
    out=result;return true;
}

LUMINA_HD inline bool apply_scaled(const Map &map,Positive incident,Positive &total){
    Positive source,response;
    return product(map.t,incident,response)&&sum(map.pa,map.mb,source)&&sum(response,source,total);
}

LUMINA_HD inline bool apply(const Map &map,double incident,double &value){
    Positive initial,total;
    if(!Positive::from_double(incident,initial)||!apply_scaled(map,initial,total))return false;
    value=total.value();return finite(value);
}

/* 트리 잎은 전파 시간순이다. 주파수 오름차순을 그대로 넣지 않는다. */
LUMINA_HD inline bool query(const Map *tree,int leaves,int begin,int end,Map &out){
    if(!tree||leaves<1||(leaves&(leaves-1))||begin<0||end<begin||end>leaves)return false;
    Map left=Map::identity(),right=Map::identity(),next;
    int lo=begin+leaves,hi=end+leaves;
    while(lo<hi){
        if(lo&1){if(!compose(left,tree[lo++],next))return false;left=next;}
        if(hi&1){if(!compose(tree[--hi],right,next))return false;right=next;}
        lo/=2;hi/=2;
    }
    return compose(left,right,out);
}
}
#undef LUMINA_HD
#endif
