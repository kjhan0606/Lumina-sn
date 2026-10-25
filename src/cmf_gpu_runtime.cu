#include <cuda_runtime.h>
#include "cmf_gpu_runtime.h"
#include "cmf_gpu_budget.hpp"
#if defined(CMF_GPU_FULLSR_CHAIN)
#include "cmf_fullsr_chain.h"
#elif defined(CMF_GPU_LINEAR)
#include "cmf_linear_cuda.h"
#else
#include "cmf_history_cuda.h"
#endif
#ifdef CMF_GPU_FOREST
#include "cmf_forest_cuda.h"
#endif
#include <algorithm>
#include <new>
#include <vector>
struct Block {void *pointer;uint64_t bytes;};
struct CMFGpuRuntime {
    uint64_t cap;CMFGpuMemory memory;int *status;std::vector<Block> blocks;
};
static thread_local CMFGpuMemory last_preflight={};
static thread_local uint64_t last_available=0,last_cap=0;
static thread_local int last_reason=0;
static thread_local bool preflight_valid=false;
extern "C" int cmf_gpu_last_preflight(CMFGpuMemory *out,uint64_t *available,uint64_t *cap,int *reason){
    if(!preflight_valid||!out||!available||!cap||!reason)return 1;
    *out=last_preflight;*available=last_available;*cap=last_cap;*reason=last_reason;return 0;
}
static bool contains(CMFGpuRuntime *c,const void *pointer,uint64_t bytes)
{
    if(!c||!pointer||!bytes)return false;
    uintptr_t p=reinterpret_cast<uintptr_t>(pointer);
    for(const auto &b:c->blocks){
        uintptr_t a=reinterpret_cast<uintptr_t>(b.pointer);
        if(p>=a&&p-a<=b.bytes&&bytes<=b.bytes-(p-a))return true;
    }
    return false;
}
static int observed(CMFGpuRuntime *c)
{
    size_t available,total;
    if(cudaMemGetInfo(&available,&total)!=cudaSuccess)return 2;
    uint64_t used=c->memory.baseline_free>available?c->memory.baseline_free-available:0;
    c->memory.observed_peak=std::max(c->memory.observed_peak,used);
    return used>c->cap?3:0;
}
extern "C" CMFGpuRuntime *cmf_gpu_create(int device,uint64_t cap,uint64_t baseline,int *code)
{
    preflight_valid=false;
    if(!code)return nullptr;
    *code=1;
    if(device!=0||cap<(UINT64_C(1)<<30)||cap>(UINT64_C(16)<<30)||!baseline)return nullptr;
    CMFGpuRuntime *c=new(std::nothrow) CMFGpuRuntime{};if(!c){*code=3;return nullptr;}
    c->cap=cap;c->memory.baseline_free=baseline;
    cudaDeviceProp prop;size_t available,total,stack;
    size_t kernel_local=0;
    if(cudaSetDevice(device)!=cudaSuccess){delete c;*code=2;return nullptr;}
#if defined(CMF_GPU_FULLSR_CHAIN)
    if(lumina_fullsr_chain_workspace(&kernel_local)){delete c;*code=2;return nullptr;}
#elif defined(CMF_GPU_LINEAR)
    if(lumina_linear_cuda_workspace(&kernel_local)){delete c;*code=2;return nullptr;}
#else
    size_t a,b,d,e,f,g;
    if(lumina_history_cuda_workspace(&a,&b,&d)||lumina_history_cuda_replay_workspace(&e,&f)||
       lumina_history_cuda_aux_workspace(&g)){delete c;*code=2;return nullptr;}
    kernel_local=std::max({a,b,d,e,f,g});
#endif
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess||
       cudaDeviceSetLimit(cudaLimitStackSize,16384)!=cudaSuccess||
       cudaDeviceGetLimit(&stack,cudaLimitStackSize)!=cudaSuccess||cudaMemGetInfo(&available,&total)!=cudaSuccess){
        delete c;*code=2;return nullptr;
    }
    size_t forest=0;
#ifdef CMF_GPU_FOREST
    if(lumina_forest_workspace(&forest)){delete c;*code=2;return nullptr;}
#endif
    c->memory.total=total;c->memory.kernel_local=std::max(kernel_local,forest);c->memory.stack=stack;
    c->memory.threads=std::max<uint64_t>(65536,(uint64_t)prop.multiProcessorCount*prop.maxThreadsPerMultiProcessor);
    constexpr bool total_cap=
#ifdef CMF_GPU_TOTAL_CAP_RESERVATION
        true;
#else
        false;
#endif
    auto reservation=lumina_gpu_budget::plan(cap,baseline,total,available,c->memory.kernel_local,stack,c->memory.threads,total_cap);
    c->memory.reserved=reservation.reserved;
    last_preflight=c->memory;last_available=available;last_cap=cap;last_reason=reservation.reason;preflight_valid=true;
    /* stack도 resident 전 스레드로 보수적으로 예약한다. 이미 할당된 stack과
     * context를 중복 상계할 수 있지만 총16GiB를 넓히지 않는다. */
    if(reservation.reason!=lumina_gpu_budget::OK){delete c;*code=3;return nullptr;}
    void *status=nullptr;int rc=cmf_gpu_allocate(c,sizeof(int),&status);
    if(rc){cmf_gpu_destroy(c);*code=rc;return nullptr;}
    c->status=static_cast<int *>(status);
    if(cudaMemset(c->status,0,sizeof(int))!=cudaSuccess){cmf_gpu_destroy(c);*code=2;return nullptr;}
    *code=0;return c;
}
extern "C" void cmf_gpu_destroy(CMFGpuRuntime *c)
{
    if(!c)return;
    cudaDeviceSynchronize();
    for(const auto &b:c->blocks)cudaFree(b.pointer);
    delete c;
}
extern "C" int cmf_gpu_allocate(CMFGpuRuntime *c,uint64_t bytes,void **out)
{
    if(!c||!out||!bytes)return 1;
    if(!lumina_gpu_budget::fits(c->cap,c->memory.reserved,c->memory.live,bytes))return 3;
    void *p=nullptr;if(cudaMalloc(&p,bytes)!=cudaSuccess)return 2;
    try{c->blocks.push_back({p,bytes});}catch(...){cudaFree(p);return 3;}
    c->memory.live+=bytes;c->memory.peak=std::max(c->memory.peak,c->memory.live);
    int rc=observed(c);
    if(rc){cmf_gpu_release(c,p);return rc;}
    *out=p;return 0;
}
extern "C" int cmf_gpu_release(CMFGpuRuntime *c,void *p)
{
    if(!c||!p||p==c->status)return 1;
    for(auto it=c->blocks.begin();it!=c->blocks.end();++it)if(it->pointer==p){
        if(cudaFree(p)!=cudaSuccess)return 2;
        c->memory.live-=it->bytes;c->blocks.erase(it);return 0;
    }
    return 1;
}
extern "C" int cmf_gpu_sync(CMFGpuRuntime *c)
{
    if(!c)return 1;
    if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
    int status=0;
    if(cudaMemcpy(&status,c->status,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
    int rc=observed(c);return rc?rc:status?100+status:0;
}
extern "C" int cmf_gpu_upload(CMFGpuRuntime *c,void *device,const void *host,uint64_t bytes)
{
    if(!host||!contains(c,device,bytes)||device==c->status)return 1;
    return cudaMemcpy(device,host,bytes,cudaMemcpyHostToDevice)==cudaSuccess?observed(c):2;
}
extern "C" int cmf_gpu_download(CMFGpuRuntime *c,void *host,const void *device,uint64_t bytes)
{
    if(!host||!contains(c,device,bytes))return 1;
    int rc=cmf_gpu_sync(c);if(rc)return rc;
    return cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost)==cudaSuccess?0:2;
}
extern "C" int cmf_gpu_zero(CMFGpuRuntime *c,void *device,uint64_t bytes)
{
    if(!contains(c,device,bytes)||device==c->status)return 1;
    return cudaMemsetAsync(device,0,bytes)==cudaSuccess?0:2;
}
extern "C" int cmf_gpu_memory(CMFGpuRuntime *c,CMFGpuMemory *out)
{
    if(!c||!out)return 1;
    int rc=observed(c);*out=c->memory;return rc;
}
extern "C" int *cmf_gpu_status(CMFGpuRuntime *c){return c?c->status:nullptr;}
