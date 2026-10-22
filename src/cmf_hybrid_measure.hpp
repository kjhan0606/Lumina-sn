#ifndef CMF_HYBRID_MEASURE_HPP
#define CMF_HYBRID_MEASURE_HPP
#include "cmf_characteristic_linear.hpp"
#include "cmf_linear_compact.hpp"
#ifdef __CUDACC__
#define LUMINA_HYBRID __host__ __device__ inline
#else
#define LUMINA_HYBRID inline
#endif
namespace lumina_hybrid_measure {
// 두 독립 시도의 최대 합도 원 호출 예산을 넘지 않는다.
LUMINA_HYBRID int primary_budget(int total){return total>=1&&total<=100000?(total+1)/2:0;}
LUMINA_HYBRID int secondary_budget(int total){return primary_budget(total)?total-primary_budget(total):0;}
LUMINA_HYBRID bool retryable(int code){
    using namespace lumina_characteristic;
    return code==WORK_LIMIT||code==PRECISION_LIMIT||code==SUBDIVISION_PRECISION_LIMIT||
        code==ACCUMULATION_PRECISION_LIMIT||code==TRANSMISSION_PRECISION_LIMIT;
}
// kernel 사이의 미완 저장 태그다. 물리적 음의 τ오차가 아니며 소비 전에 교체한다.
// 실제 맵의 tau_error는 비음수이므로 정상 맵과 겹치지 않는다.
LUMINA_HYBRID bool mark(lumina_linear_compact::Map &out,int code){
    if(!retryable(code))return false;
    out={{0.,0},{0.,0},{0.,0},-double(code)};return true;
}
LUMINA_HYBRID int pending(const lumina_linear_compact::Map &m){
    if(m.tau_error>=0)return 0;
    for(int code=3;code<=7;code++)if(m.tau_error==-double(code)&&
        m.transmission.mantissa==0&&m.transmission.exponent==0&&m.source.mantissa==0&&m.source.exponent==0&&
        m.source_error.mantissa==0&&m.source_error.exponent==0)return code;
    return -1;
}
}
#undef LUMINA_HYBRID
#endif
