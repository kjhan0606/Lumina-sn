#include <cuda_runtime.h>
#include "cmf_fullsr_linear_map.hpp"
#include "cmf_linear_native.h"
#include "cmf_linear_provider.hpp"
using lumina_fullsr_map::Affine;
using lumina_ordered::Positive;
static constexpr int threads=64;
static_assert(sizeof(Affine)==32&&sizeof(Positive)==sizeof(CMFLinearPositive),"양의 맵 ABI");
__device__ static bool sample(lumina_native_linear::Reader &reader,int point,double coordinate,double &c,double &e){
    if(!lumina_ordered::finite(coordinate)||coordinate<0)return false;
    int low=coordinate>=reader.view.nf-1?reader.view.nf-1:(int)::floor(coordinate);
    double fraction=coordinate>=reader.view.nf-1?0.:coordinate-low,c0,e0,c1,e1;
    if(!reader.at(point,low,c0,e0))return false;
    if(fraction==0){c=c0;e=e0;return true;}
    if(!reader.at(point,low+1,c1,e1))return false;
    c=c0+fraction*(c1-c0);e=e0+fraction*(e1-e0);
    return lumina_ordered::finite(c)&&lumina_ordered::finite(e)&&e>=0;
}
__global__ static void build(int steps,int rays,const double *length,const double *delta,
    const double *chi,const double *eta,Affine *maps,int *status){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=steps*rays||*status)return;
    int k=index/rays;double l=length[k],d=delta[k],c0=chi[index],c1=chi[index+rays];
    double e0=eta[index],e1=eta[index+rays];uint64_t a=0,b=0;Affine result;
    if(!lumina_ordered::finite(l)||l<=0||!lumina_ordered::finite(d)||d<0||
       !lumina_ordered::finite(c0)||!lumina_ordered::finite(c1)||
       !lumina_ordered::finite(e0)||e0<0||!lumina_ordered::finite(e1)||e1<0||
       !lumina_fullsr_map::source(::fma(c0,l,3.*d),::fma(c1,l,3.*d),l,e0,e1,result,a,b)){
        atomicCAS(status,0,71);return;
    }
    maps[index]=result;
}
__global__ static void build_provider(CMFNodalDenseView view,int steps,int rays,const double *length,
    const double *delta,const CMFNodalLocation *location,const double *phase,const int *columns,Affine *maps,int *status){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=steps*rays||*status)return;
    int k=index/rays,j=index%rays;double c0,e0,c1,e1,l=length[k],d=delta[k];uint64_t a=0,b=0;Affine result;
    lumina_native_linear::Reader reader={view,{location[k],location[k+1]},0};
    if(!lumina_native_linear::valid(view,reader.location)||columns[j]<0||columns[j]>=view.nf||
       !lumina_ordered::finite(l)||l<=0||!lumina_ordered::finite(d)||d<0||
       !sample(reader,0,columns[j]+phase[k],c0,e0)||!sample(reader,2,columns[j]+phase[k+1],c1,e1)||
       !lumina_fullsr_map::source(::fma(c0,l,3.*d),::fma(c1,l,3.*d),l,e0,e1,result,a,b)){
        atomicCAS(status,0,74);return;
    }
    maps[index]=result;
}
__global__ static void reduce(int rows,int rays,const Affine *input,Affine *output,int *status){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=((rows+1)/2)*rays||*status)return;
    int row=index/rays,j=index%rays,a=2*row*rays+j,b=a+rays;Affine result=input[a];
    if(2*row+1<rows&&!lumina_fullsr_map::compose(input[a],input[b],result)){
        atomicCAS(status,0,72);return;
    }
    output[index]=result;
}
__global__ static void apply(int rays,const Affine *maps,const CMFLinearPositive *initial,
                             CMFLinearPositive *output,int *status){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=rays||*status)return;
    Positive in={initial[j].mantissa,initial[j].exponent},a,result;
    if(!lumina_ordered::finite(in.mantissa)||in.mantissa<0||
       (in.mantissa!=0&&(in.mantissa<.5||in.mantissa>=1.))||
       in.exponent<=-(INT64_C(1)<<60)||in.exponent>=(INT64_C(1)<<60)||
       !lumina_ordered::product(maps[j].t,in,a)||!lumina_ordered::sum(a,maps[j].s,result)){
        atomicCAS(status,0,73);return;
    }
    output[j]={result.mantissa,result.exponent};
}
/* 주소는 전부 기존 runtime 소유 장치 버퍼다. enqueue 성공만으로 소비하지 않는다. */
extern "C" int lumina_fullsr_chain_build(int steps,int rays,const double *length,const double *delta,
    const double *chi,const double *eta,void *maps,int *status){
    if(steps<1||steps>65536||rays<1||rays>512||(uint64_t)steps*rays>2000000||
       !length||!delta||!chi||!eta||!maps||!status)return 1;
    build<<<(steps*rays+threads-1)/threads,threads>>>(steps,rays,length,delta,chi,eta,(Affine *)maps,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_provider_build(const CMFNodalDenseView *view,int steps,int rays,
    const double *length,const double *delta,const CMFNodalLocation *location,const double *phase,
    const int *columns,void *maps,int *status){
    if(!view||steps<1||steps>65536||rays<1||rays>512||(uint64_t)steps*rays>2000000||
       !length||!delta||!location||!phase||!columns||!maps||!status)return 1;
    build_provider<<<(steps*rays+threads-1)/threads,threads>>>(*view,steps,rays,length,delta,location,phase,columns,(Affine *)maps,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_reduce(int rows,int rays,const void *input,void *output,int *status){
    if(rows<2||rows>65536||rays<1||rays>512||(uint64_t)rows*rays>2000000||
       !input||!output||input==output||!status)return 1;
    reduce<<<(((rows+1)/2)*rays+threads-1)/threads,threads>>>(rows,rays,(const Affine *)input,(Affine *)output,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_apply(int rays,const void *maps,const CMFLinearPositive *initial,
    CMFLinearPositive *output,int *status){
    if(rays<1||rays>512||!maps||!initial||!output||!status)return 1;
    apply<<<(rays+threads-1)/threads,threads>>>(rays,(const Affine *)maps,initial,output,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_workspace(size_t *local){
    if(!local)return 1;
    cudaFuncAttributes a,b,c,d;
    if(cudaFuncGetAttributes(&a,build)!=cudaSuccess||cudaFuncGetAttributes(&b,reduce)!=cudaSuccess||
       cudaFuncGetAttributes(&c,apply)!=cudaSuccess||cudaFuncGetAttributes(&d,build_provider)!=cudaSuccess)return 2;
    *local=a.localSizeBytes>b.localSizeBytes?a.localSizeBytes:b.localSizeBytes;
    if(c.localSizeBytes>*local)*local=c.localSizeBytes;
    if(d.localSizeBytes>*local)*local=d.localSizeBytes;
    return 0;
}
extern "C" int lumina_fullsr_chain_device_uuid(char *bytes){
    if(!bytes)return 1;
    cudaDeviceProp p;if(cudaGetDeviceProperties(&p,0)!=cudaSuccess)return 2;
    for(int k=0;k<16;k++){bytes[k]=p.uuid.bytes[k];}
    return 0;
}
