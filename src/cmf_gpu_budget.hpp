#ifndef CMF_GPU_BUDGET_HPP
#define CMF_GPU_BUDGET_HPP
#include <cstdint>
#include <limits>

namespace lumina_gpu_budget {
enum Reason { OK=0, BAD_INPUT=1, OVERFLOW=2, LEGACY_RESERVE_LIMIT=3, TOTAL_CAP=4, AVAILABLE=5 };
struct Plan {std::uint64_t reserved;Reason reason;};
inline bool fits(std::uint64_t cap,std::uint64_t reserved,std::uint64_t live,std::uint64_t bytes){
    return reserved<cap&&live<=cap-reserved&&bytes<=cap-reserved-live;
}
/* 순수한 host 예약 계산이다. 실제 할당/관측16GiB 검사를 대신하지 않는다.
 * 기본8GiB 예약 상한과 명시적 총한도 선택을 구별하고 모든 합/곱을 검사한다. */
inline Plan plan(std::uint64_t cap,std::uint64_t baseline,std::uint64_t total,std::uint64_t available,
    std::uint64_t local,std::uint64_t stack,std::uint64_t threads,bool total_cap){
    constexpr std::uint64_t gib=UINT64_C(1)<<30,padding=UINT64_C(256)<<20;
    const auto maximum=std::numeric_limits<std::uint64_t>::max();
    // NVML free와 CUDA total은 같은 회계 기준이 아니다. 총한도 선택에서는
    // 차이를 context에 보수적으로 포함한다. 큰 잘못된 값은 아래 overflow/
    // 총한도 검사로 거부하며, 기본 경로의 기존 입력 거부는 유지한다.
    if(cap<gib||cap>16*gib||!baseline||(!total_cap&&baseline>total)||available>total||!threads)return {0,BAD_INPUT};
    if(local>maximum-stack)return {0,OVERFLOW};
    std::uint64_t frame=local+stack,context=baseline>available?baseline-available:0;
    if(frame&&threads>maximum/frame)return {0,OVERFLOW};
    std::uint64_t work=frame*threads;
    if(context>maximum-padding||work>maximum-context-padding)return {0,OVERFLOW};
    std::uint64_t reserved=context+work+padding;
    if(!total_cap&&reserved>8*gib)return {reserved,LEGACY_RESERVE_LIMIT};
    if(reserved>=cap)return {reserved,TOTAL_CAP};
    if(available<cap-reserved)return {reserved,AVAILABLE};
    return {reserved,OK};
}
}
#endif
