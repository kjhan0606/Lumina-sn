/* 실패한 런타임의 수치 계산 없는 재현. 기존 보호 조건을 그대로 보고한다. */
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
int main(int argc,char **argv){
    if(argc!=3)return 1;
    uint64_t baseline=std::strtoull(argv[2],nullptr,10),cap=UINT64_C(16)<<30;
    void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library)return 2;
    using Workspace=int (*)(size_t *);
    Workspace workspace=reinterpret_cast<Workspace>(dlsym(library,"lumina_linear_cuda_workspace"));
    size_t local=0,available=0,total=0,stack=0;cudaDeviceProp prop;
    if(!workspace||cudaSetDevice(0)!=cudaSuccess||workspace(&local)||
       cudaGetDeviceProperties(&prop,0)!=cudaSuccess||cudaDeviceSetLimit(cudaLimitStackSize,16384)!=cudaSuccess||
       cudaDeviceGetLimit(&stack,cudaLimitStackSize)!=cudaSuccess||cudaMemGetInfo(&available,&total)!=cudaSuccess)return 2;
    uint64_t threads=std::max<uint64_t>(65536,uint64_t(prop.multiProcessorCount)*prop.maxThreadsPerMultiProcessor);
    uint64_t context=baseline>available?baseline-available:0;
    uint64_t reserved=context+(local+stack)*threads+(UINT64_C(256)<<20);
    std::printf("{\"baseline\":%llu,\"cuda_total\":%llu,\"cuda_free\":%llu,\"context\":%llu,"
        "\"kernel_local\":%llu,\"stack\":%llu,\"threads\":%llu,\"reserved\":%llu,"
        "\"baseline_gt_total\":%s,\"reserved_gt_8GiB\":%s,\"reserved_ge_16GiB\":%s,"
        "\"free_lt_remaining\":%s,\"observed_gt_16GiB\":%s}\n",
        (unsigned long long)baseline,(unsigned long long)total,(unsigned long long)available,
        (unsigned long long)context,(unsigned long long)local,(unsigned long long)stack,
        (unsigned long long)threads,(unsigned long long)reserved,baseline>total?"true":"false",
        reserved>(UINT64_C(8)<<30)?"true":"false",reserved>=cap?"true":"false",
        reserved<cap&&available<cap-reserved?"true":"false",context>cap?"true":"false");
    return context>cap?3:0;
}
