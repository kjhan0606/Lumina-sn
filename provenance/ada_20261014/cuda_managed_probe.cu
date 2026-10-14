#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void page_write(uint64_t *p, size_t pages) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<pages;
         i+=(size_t)blockDim.x*gridDim.x) p[i*512]=i ^ 0x713a9bd5ULL;
}
__global__ void page_check(const uint64_t *p, size_t pages, unsigned *bad) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<pages;
         i+=(size_t)blockDim.x*gridDim.x)
        if (p[i*512]!=(i ^ 0x713a9bd5ULL)) atomicAdd(bad,1U);
}
#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"FAIL %s: %s\n",#call,cudaGetErrorString(e)); return 1; } } while(0)
int main() {
    int count=0, managed=0, concurrent=0;
    CUDA_OK(cudaGetDeviceCount(&count));
    if(count!=1) return 2;
    CUDA_OK(cudaDeviceGetAttribute(&managed,cudaDevAttrManagedMemory,0));
    CUDA_OK(cudaDeviceGetAttribute(&concurrent,cudaDevAttrConcurrentManagedAccess,0));
    std::printf("visible=%d managed=%d concurrent_managed=%d\n",count,managed,concurrent);
    if(!managed||!concurrent) return 3;
    size_t free_bytes=0,total=0;
    CUDA_OK(cudaMemGetInfo(&free_bytes,&total));
    const size_t bytes=total+(2ULL<<30),pages=bytes/4096;
    uint64_t *p=nullptr; unsigned *bad=nullptr;
    CUDA_OK(cudaMalloc(&bad,sizeof(unsigned)));
    CUDA_OK(cudaMemset(bad,0,sizeof(unsigned)));
    CUDA_OK(cudaMallocManaged(&p,bytes,cudaMemAttachGlobal));
    std::printf("device_total_bytes=%zu managed_bytes=%zu pages=%zu\n",total,bytes,pages);
    std::fflush(stdout);
    page_write<<<256,256>>>(p,pages);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    page_check<<<256,256>>>(p,pages,bad);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    unsigned errors=0; CUDA_OK(cudaMemcpy(&errors,bad,sizeof(errors),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(p)); CUDA_OK(cudaFree(bad));
    std::printf("OVERSUBSCRIPTION_%s mismatched_pages=%u\n",errors?"FAIL":"PASS",errors);
    return errors?4:0;
}
