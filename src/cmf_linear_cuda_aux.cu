#include <cuda_runtime.h>
#include "cmf_linear_cuda.h"
#include "cmf_gpu_reconstruction.hpp"
/* 원래8점 재구성·경계·각도 누적만 재사용한다. C 수송 맵은 포함하지 않는다. */
__global__ static void reconstruct(int nf,double shift,lumina_reconstruction::Bank bank,const double *initial,
    double *out,unsigned long long *limited,int *status)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf||*status)return;
    double value;bool flag;
    if(!lumina_reconstruction::one(nf,shift,initial,bank,i,value,flag)){atomicCAS(status,0,31);return;}
    out[i]=value;if(flag)atomicAdd(limited,1ULL);
}
extern "C" int lumina_history_cuda_reconstruct(int nf,double shift,const double *initial,double *output,uint64_t *limited,int *status)
{
    if(nf<8||nf>2000000||!lumina_ordered::finite(shift)||shift<0||!initial||!output||initial==output||!limited||!status)return 1;
    if(cudaMemsetAsync(limited,0,sizeof(uint64_t))!=cudaSuccess)return 2;
    static const auto bank=lumina_reconstruction::build_bank();
    reconstruct<<<(nf+127)/128,128>>>(nf,shift,bank,initial,output,reinterpret_cast<unsigned long long *>(limited),status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
__global__ static void boundary(int nf,const double *planck,const double *gradient,double mu,double *ray,int *status)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf||*status)return;
    double value=planck[i]+mu*gradient[i];
    if(!lumina_ordered::finite(planck[i])||planck[i]<0||!lumina_ordered::finite(gradient[i])||
       !lumina_ordered::finite(value)||value<0){atomicCAS(status,0,32);return;}
    ray[i]=value;
}
extern "C" int lumina_history_cuda_boundary(int nf,const double *planck,const double *gradient,double mu,double *ray,int *status)
{
    if(nf<8||nf>2000000||!planck||!gradient||!ray||!status||!lumina_ordered::finite(mu)||::fabs(mu)>1)return 1;
    boundary<<<(nf+255)/256,256>>>(nf,planck,gradient,mu,ray,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
__global__ static void accumulate(int ns,int nf,int n,const int *nodes,const double *weights,
    const double *ray,double *moments,int *status)
{
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf||*status)return;
    double value=ray[f];if(!lumina_ordered::finite(value)||value<0){atomicCAS(status,0,33);return;}
    for(int e=0;e<n;e++){
        int node=nodes[e];if(node<0||node>=ns){atomicCAS(status,0,33);return;}
        for(int m=0;m<3;m++){
            double w=weights[3*e+m];size_t at=((size_t)m*ns+node)*nf+f;
            double next=moments[at]+w*value;
            if(!lumina_ordered::finite(w)||((m==0||m==2)&&w<0)||!lumina_ordered::finite(next)||
               ((m==0||m==2)&&next<0)){atomicCAS(status,0,33);return;}
            moments[at]=next;
        }
    }
}
extern "C" int lumina_history_cuda_accumulate(int ns,int nf,int n,const int *nodes,const double *weights,
    const double *ray,double *moments,int *status)
{
    if(ns<24||ns>512||nf<8||nf>2000000||n<1||n>1024||!nodes||!weights||!ray||!moments||!status)return 1;
    accumulate<<<(nf+255)/256,256>>>(ns,nf,n,nodes,weights,ray,moments,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_linear_cuda_aux_workspace(size_t *out){
    if(!out)return 1;
    const void *kernels[]={(const void *)reconstruct,(const void *)boundary,(const void *)accumulate};
    size_t maximum=0;
    for(const void *kernel:kernels){
        cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,kernel)!=cudaSuccess)return 2;
        if(a.localSizeBytes>maximum)maximum=a.localSizeBytes;
    }
    *out=maximum;return 0;
}
