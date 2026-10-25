#include <cuda_runtime.h>
#include "cmf_fullsr_linear_map.hpp"
#include "cmf_fullsr_chain.h"
#include "cmf_linear_provider.hpp"
#include "cmf_fullsr_entry.hpp"
extern "C" int lumina_linear_cuda_aux_workspace(size_t *out);
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
__global__ static void prepare_entry(int nf,int first,int count,const double *phase,double h,
    const CMFLinearPositive *incident,CMFLinearPositive *prepared,int *status){
    int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=count||*status)return;int column=first+at;Positive value;
    if(!lumina_fullsr_entry::incoming(incident,nf,column,phase[0],h,value)){atomicCAS(status,0,75);return;}
    prepared[column]={value.mantissa,value.exponent};
}
__global__ static void stream(CMFNodalDenseView view,int steps,int first,int count,
    const double *length,const double *delta,const CMFNodalLocation *location,const double *phase,
    const CMFLinearPositive *incident,CMFLinearPositive *output,int *status){
    int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=count||*status)return;int column=first+at;
    Positive current;
    if(!lumina_linear_affine::decode(incident[column],current)){atomicCAS(status,0,75);return;}
    for(int k=0;k<steps;k++){
        double c0,c1,e0,e1,l=length[k],d=delta[k];uint64_t a=0,b=0;Affine map;Positive response,next;
        lumina_native_linear::Reader reader={view,{location[k],location[k+1]},0};
        if(!lumina_native_linear::valid(view,reader.location)||!lumina_ordered::finite(l)||l<=0||
           !lumina_ordered::finite(d)||d<0||!sample(reader,0,column+phase[k],c0,e0)||
           !sample(reader,2,column+phase[k+1],c1,e1)||
           !lumina_fullsr_map::source(::fma(c0,l,3.*d),::fma(c1,l,3.*d),l,e0,e1,map,a,b)||
           !lumina_ordered::product(map.t,current,response)||!lumina_ordered::sum(response,map.s,next)){
            atomicCAS(status,0,76);return;
        }
        current=next;
    }
    output[column]={current.mantissa,current.exponent};
}
__global__ static void encode(int nf,const double *input,CMFLinearPositive *output,int *status){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nf||*status)return;Positive p;
    if(!Positive::from_double(input[j],p)){atomicCAS(status,0,77);return;}
    output[j]={p.mantissa,p.exponent};
}
__global__ static void decode(int nf,const CMFLinearPositive *input,double *output,
    unsigned long long *counts,int *status){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nf||*status)return;Positive p;
    if(!lumina_linear_affine::decode(input[j],p)){atomicCAS(status,0,78);return;}
    double value=p.value();
    if(!lumina_ordered::finite(value)){atomicAdd(counts+2,1ULL);atomicCAS(status,0,79);return;}
    if(p.mantissa==0)atomicAdd(counts,1ULL);
    else if(value==0)atomicAdd(counts+1,1ULL);
    output[j]=value;
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
extern "C" int lumina_fullsr_chain_stream(const CMFNodalDenseView *view,int steps,int first,int count,
    const double *length,const double *delta,const CMFNodalLocation *location,const double *phase,
    double h,const CMFLinearPositive *incident,CMFLinearPositive *output,int *status){
    if(!view||steps<1||steps>65536||first<0||count<1||count>view->nf-first||
       !length||!delta||!location||!phase||!incident||!output||incident==output||!status||
       !lumina_ordered::finite(h)||h<=0)return 1;
    prepare_entry<<<(count+threads-1)/threads,threads>>>(view->nf,first,count,phase,h,incident,output,status);
    if(cudaGetLastError()!=cudaSuccess)return 2;
    // shadow 안의 각 열은 이 커널에서 독립적이다. 원 native 입사는 바꾸지 않는다.
    stream<<<(count+threads-1)/threads,threads>>>(*view,steps,first,count,length,delta,location,phase,output,output,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_encode(int nf,const double *input,CMFLinearPositive *output,int *status){
    if(nf<2||nf>2000000||!input||!output||!status)return 1;
    encode<<<(nf+255)/256,256>>>(nf,input,output,status);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_decode(int nf,const CMFLinearPositive *input,double *output,uint64_t *counts,int *status){
    if(nf<2||nf>2000000||!input||!output||!counts||!status)return 1;
    decode<<<(nf+255)/256,256>>>(nf,input,output,(unsigned long long *)counts,status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_fullsr_chain_workspace(size_t *local){
    if(!local)return 1;
    cudaFuncAttributes a,b,c,d,e,f,g,entry;
    if(cudaFuncGetAttributes(&a,build)!=cudaSuccess||cudaFuncGetAttributes(&b,reduce)!=cudaSuccess||
       cudaFuncGetAttributes(&c,apply)!=cudaSuccess||cudaFuncGetAttributes(&d,build_provider)!=cudaSuccess||
       cudaFuncGetAttributes(&e,stream)!=cudaSuccess||cudaFuncGetAttributes(&f,encode)!=cudaSuccess||
       cudaFuncGetAttributes(&g,decode)!=cudaSuccess||cudaFuncGetAttributes(&entry,prepare_entry)!=cudaSuccess)return 2;
    *local=a.localSizeBytes>b.localSizeBytes?a.localSizeBytes:b.localSizeBytes;
    if(c.localSizeBytes>*local)*local=c.localSizeBytes;
    if(d.localSizeBytes>*local)*local=d.localSizeBytes;
    if(e.localSizeBytes>*local)*local=e.localSizeBytes;
    if(f.localSizeBytes>*local)*local=f.localSizeBytes;
    if(g.localSizeBytes>*local)*local=g.localSizeBytes;
    if(entry.localSizeBytes>*local)*local=entry.localSizeBytes;
    size_t aux=0;if(lumina_linear_cuda_aux_workspace(&aux))return 2;
    if(aux>*local)*local=aux;
    return 0;
}
extern "C" int lumina_fullsr_chain_device_uuid(char *bytes){
    if(!bytes)return 1;
    cudaDeviceProp p;if(cudaGetDeviceProperties(&p,0)!=cudaSuccess)return 2;
    for(int k=0;k<16;k++){bytes[k]=p.uuid.bytes[k];}
    return 0;
}
