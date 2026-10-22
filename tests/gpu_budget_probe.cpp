#include "cmf_gpu_budget.hpp"
#include "cmf_gpu_runtime.h"
extern "C" int reservation_probe(uint64_t cap,uint64_t baseline,uint64_t total,uint64_t available,
    uint64_t local,uint64_t stack,uint64_t threads,int selected,uint64_t *reserved){
    if(!reserved)return 99;
    auto p=lumina_gpu_budget::plan(cap,baseline,total,available,local,stack,threads,selected!=0);
    *reserved=p.reserved;return p.reason;
}
extern "C" int reservation_fits_probe(uint64_t cap,uint64_t reserved,uint64_t live,uint64_t bytes){
    return lumina_gpu_budget::fits(cap,reserved,live,bytes);
}
/* 실패 숫자 reader의 CPU ABI 시험이다. GPU/문맥은 만들지 않는다. */
extern "C" int cmf_gpu_last_preflight(CMFGpuMemory *out,uint64_t *available,uint64_t *cap,int *reason){
    if(!out||!available||!cap||!reason)return 1;
    *out={0,0,UINT64_C(9)<<30,0,UINT64_C(79)<<30,UINT64_C(80)<<30,9168,16384,221184};
    *available=UINT64_C(75)<<30;*cap=UINT64_C(16)<<30;*reason=3;return 0;
}
