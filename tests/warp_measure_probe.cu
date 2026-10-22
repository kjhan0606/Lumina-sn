#include <cuda_runtime.h>
#include "cmf_warp_measure.cuh"
__global__ void polynomial(const double *phase,int adaptive,double *out,int *status){
    lumina_warp_measure::Polynomial p;
    if(!lumina_warp_measure::exponential(phase,adaptive!=0,p)){
        if(threadIdx.x==0)atomicCAS(status,0,80);return;
    }
    if(threadIdx.x<25)out[threadIdx.x]=p.coefficient;
    if(threadIdx.x==0){out[25]=p.shift;out[26]=p.tail;out[27]=p.degree;}
}
__global__ void positive_sum(const lumina_ordered::Positive *values,lumina_ordered::Positive *out,int *status){
    lumina_ordered::Positive result;
    if(!lumina_warp_measure::reduce(values[threadIdx.x],result)){
        if(threadIdx.x==0)atomicCAS(status,0,81);return;
    }
    if(threadIdx.x==0)*out=result;
}
extern "C" int lumina_warp_polynomial_probe(const double *phase,int adaptive,double *out,int *status){
    if(!phase||!out||!status||(adaptive!=0&&adaptive!=1))return 1;
    polynomial<<<1,32>>>(phase,adaptive,out,status);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_warp_positive_sum_probe(const void *values,void *out,int *status){
    if(!values||!out||!status)return 1;
    positive_sum<<<1,32>>>(static_cast<const lumina_ordered::Positive *>(values),
        static_cast<lumina_ordered::Positive *>(out),status);return cudaGetLastError()==cudaSuccess?0:2;
}
