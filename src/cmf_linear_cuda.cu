#include <cuda_runtime.h>
#include "cmf_linear_cuda.h"
#include "cmf_linear_provider.hpp"
#include "cmf_linear_endpoint_cache.hpp"
#include "cmf_characteristic_linear.hpp"
#include "cmf_linear_affine.hpp"
#ifdef CMF_GPU_COMPACT_MEASURE
#ifndef CMF_GPU_SOURCE_MEASURE
#error CMF_GPU_COMPACT_MEASURE_requires_CMF_GPU_SOURCE_MEASURE
#endif
#include "cmf_linear_compact.hpp"
using StoredMap=lumina_linear_compact::Map;
#else
using StoredMap=CMFLinearNativeMap;
#endif
#ifdef CMF_GPU_SOURCE_MEASURE
#include "cmf_source_projection.hpp"
#endif
#include <algorithm>
using lumina_linear::Node;
using lumina_ordered::Positive;
using lumina_ordered::product;
using lumina_ordered::sum;
using lumina_linear_affine::Local;
static constexpr int threads=128,chunk=32768;
static constexpr int maximum_block=
#ifdef CMF_GPU_SOURCE_MEASURE
    32;
static constexpr size_t node_bytes=sizeof(Node)+sizeof(lumina_source_measure::Node);
#else
    256;
static constexpr size_t node_bytes=sizeof(Node);
#endif
__host__ __device__ static int source_block(const CMFLinearCudaPiece &p){
#ifdef CMF_GPU_COMPACT_MEASURE
    (void)p;return 32;
#else
    return p.block;
#endif
}
__host__ __device__ static uint64_t moment_tree_bytes(const CMFLinearCudaPiece &p){
    uint64_t bytes=(uint64_t)2*p.leaves*sizeof(Node);
#ifdef CMF_GPU_SOURCE_MEASURE
    bytes+=(uint64_t)2*p.source_leaves*sizeof(lumina_source_measure::Node);
#endif
    return bytes;
}
#ifdef CMF_GPU_ENDPOINT_CACHE
__host__ __device__ static int endpoint_prefix(const CMFLinearCudaPiece &p){
    return lumina_endpoint_cache::required_prefix(p.nf,p.bins,p.block,source_block(p),p.offset,p.span,p.count);
}
#endif
static uint64_t tree_bytes(const CMFLinearCudaPiece &p){
    uint64_t bytes=moment_tree_bytes(p);
#ifdef CMF_GPU_ENDPOINT_CACHE
    bytes+=(uint64_t)4*p.nf*sizeof(double);
#endif
    return bytes;
}
#ifdef CMF_GPU_ENDPOINT_CACHE
using Reader=lumina_endpoint_cache::Reader;
__device__ static Reader make_reader(const CMFLinearCudaPiece &p,const Node *nodes){
    const double *values=reinterpret_cast<const double *>(reinterpret_cast<const char *>(nodes)+moment_tree_bytes(p));
    return {values,p.nf,endpoint_prefix(p)};
}
__global__ static void fill_endpoints(CMFLinearCudaPiece p,Node *nodes,int *status){
    int prefix=endpoint_prefix(p),count=prefix+(prefix<p.nf?1:0);
    int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=count||*status)return;
    int column=at<prefix?at:p.nf-1;
    double *values=reinterpret_cast<double *>(reinterpret_cast<char *>(nodes)+moment_tree_bytes(p));
    lumina_native_linear::Reader original={p.view,{p.locations[0],p.locations[1]},p.base};
    if(!lumina_endpoint_cache::fill(original,p.nf,column,values))atomicCAS(status,0,47);
}
#else
using Reader=lumina_native_linear::Reader;
__device__ static Reader make_reader(const CMFLinearCudaPiece &p,const Node *){
    return {p.view,{p.locations[0],p.locations[1]},p.base};
}
#endif
extern "C" size_t lumina_linear_cuda_endpoint_bytes(int nf){
#ifdef CMF_GPU_ENDPOINT_CACHE
    return nf>=2&&nf<=2000000?(size_t)4*nf*sizeof(double):0;
#else
    (void)nf;return 0;
#endif
}
static bool valid_piece(const CMFLinearCudaPiece *p){
    using lumina_ordered::finite;
    return p&&lumina_native_linear::valid(p->view,p->locations)&&finite(p->length)&&p->length>0&&
        finite(p->h)&&p->h>0&&finite(p->da)&&finite(p->db)&&p->db>=0&&p->da>=p->db&&
        p->begin>=0&&p->count>=1&&p->begin<=p->view.nf-p->count&&p->base>=0&&p->base<=p->view.nf-2&&
        p->nf==p->view.nf-p->base&&p->block>=1&&p->block<=maximum_block&&p->leaves>=1&&p->leaves<=2097152&&
        !(p->leaves&(p->leaves-1))&&p->source_leaves>=1&&p->source_leaves<=2097152&&
        !(p->source_leaves&(p->source_leaves-1))&&
        (uint64_t)p->bins<=(uint64_t)p->source_leaves*source_block(*p)&&
        p->bins>=0&&p->bins<=p->nf-1&&
        (uint64_t)p->bins<=(uint64_t)p->leaves*p->block&&finite(p->offset)&&p->offset>=0&&
        finite(p->span)&&p->span>=0&&p->span<=2000000000.&&p->offset<=2000000000.;
}
extern "C" size_t lumina_linear_cuda_node_bytes(void){return node_bytes;}
extern "C" size_t lumina_linear_cuda_map_bytes(void){return sizeof(StoredMap);}
extern "C" size_t lumina_linear_cuda_local_bytes(void){return sizeof(Local);}
extern "C" int lumina_linear_cuda_plan(const CMFNodalDenseView *view,const CMFNodalLocation *locations,
    double length,double h,double da,double db,int begin,int count,int block,CMFLinearCudaPiece *out)
{
    using lumina_ordered::finite;
    if(!view||!out||!lumina_native_linear::valid(*view,locations)||!finite(length)||length<=0||
       !finite(h)||h<=0||!finite(da)||!finite(db)||db<0||da<db||begin<0||count<1||
       begin>view->nf-count||block<1||block>maximum_block)return 1;
    double shift=db/h,span=(da-db)/h;
    if(!finite(shift)||!finite(span)||shift>2000000000.||span>2000000000.)return 1;
    int whole=(int)::floor(shift),base=std::min(view->nf-2,begin+whole),nf=view->nf-base;
    double offset=shift-whole+double(begin+whole-base);
    int bins=span==0||offset>=nf-1?0:(int)::ceil(::fmin(nf-1.,offset+(count-1)+span));
    int blocks=bins?(bins-1)/block+1:0,leaves=1;while(leaves<blocks)leaves*=2;
    CMFLinearCudaPiece p={*view,{locations[0],locations[1]},length,h,da,db,offset,span,begin,count,base,nf,block,leaves,bins,leaves};
    int sb=source_block(p),source_blocks=bins?(bins-1)/sb+1:0;
    p.source_leaves=1;while(p.source_leaves<source_blocks)p.source_leaves*=2;
    if(tree_bytes(p)>(UINT64_C(1)<<30))return 3;
    if(!valid_piece(&p))return 1;
    *out=p;return 0;
}
__global__ static void build_blocks(CMFLinearCudaPiece p,Node *nodes,int *status){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=p.leaves||*status)return;
    Node n=lumina_linear::empty(),part,next;
    Reader reader=make_reader(p,nodes);
    int blocks=p.bins?(p.bins-1)/p.block+1:0;
    if(i<blocks){
        int end=(i+1)*p.block;if(end>p.nf-1)end=p.nf-1;
        for(int j=i*p.block;j<end;j++){
            if(!lumina_linear::partial(reader,p.nf,j,0.,1.,part)||!lumina_linear::combine(n,part,next)){
                atomicCAS(status,0,41);return;
            }
            n=next;
        }
    }
    nodes[p.leaves+i]=n;
}
__global__ static void tree_level(int base,Node *nodes,int *status){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=base||*status)return;
    int at=base+i;Node n;
    if(!lumina_linear::combine(nodes[2*at],nodes[2*at+1],n)){atomicCAS(status,0,42);return;}
    nodes[at]=n;
}
#ifdef CMF_GPU_SOURCE_MEASURE
using SourceNode=lumina_source_measure::Node;
static constexpr int source_degree=lumina_source_measure::degree;
static constexpr int projection_capacity=(maximum_block+1)*(source_degree+1);
// k축 스레드가 서로 다른 계수를 읽으므로 broadcast 상수 메모리가 아닌
// 연속 global 메모리를 쓴다. 빌드 중에만 읽고 같은 default stream에서 교체한다.
__device__ static double projection_coefficients[2*projection_capacity];
static int prepare_projection(const CMFLinearCudaPiece &p){
    try{
        int block=source_block(p),tail=(p.nf-1)%block;if(tail==0)tail=block;
        const auto &full=lumina_source_projection::matrix(block);
        const auto &last=lumina_source_projection::matrix(tail);
        if(cudaMemcpyToSymbol(projection_coefficients,full.weights.data(),full.weights.size()*sizeof(double),0,
            cudaMemcpyHostToDevice)!=cudaSuccess||
           cudaMemcpyToSymbol(projection_coefficients,last.weights.data(),last.weights.size()*sizeof(double),
            projection_capacity*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
        return 0;
    }catch(...){return 3;}
}
__global__ static void source_blocks(CMFLinearCudaPiece p,SourceNode *nodes,const Node *chi_nodes,int *status){
    int i=blockIdx.x,lane=threadIdx.x;
    __shared__ Positive eta[2][maximum_block+1];__shared__ int failed;
    if(lane==0)failed=*status;__syncthreads();if(failed)return;
#ifdef CMF_GPU_COMPACT_MEASURE
    constexpr int block=32;
#else
    int block=p.block;
#endif
    int blocks=p.bins?(p.bins-1)/block+1:0;
    int first=i*block,width=i<blocks?min(block,p.nf-1-first):0;
    if(width>0&&lane<=width){
        Reader reader=make_reader(p,chi_nodes);
        for(int c=0;c<2;c++){
            double chi,value;
            if(!reader.at(2*c,first+lane,chi,value)||!Positive::from_double(value,eta[c][lane])){
                atomicExch(&failed,1);atomicCAS(status,0,48);
            }
        }
    }
    __syncthreads();if(failed)return;
    if(lane<=source_degree){
        Positive total[2]={Positive::zero(),Positive::zero()},term;
        int offset=width==block?0:projection_capacity;
        for(int j=0;width>0&&j<=width;j++){
            double weight=projection_coefficients[offset+j*(source_degree+1)+lane];
            for(int c=0;c<2;c++)if(!lumina_characteristic::scaled(eta[c][j],weight,term)||!sum(total[c],term,total[c])){
                atomicCAS(status,0,48);return;
            }
        }
        nodes[p.source_leaves+i].m[0][lane]=total[0];nodes[p.source_leaves+i].m[1][lane]=total[1];
    }
    if(lane==0){auto &n=nodes[p.source_leaves+i];n.width=width;n.first=width?first:0;
        n.last=width?first+width:0;n.depth=width?width-1:0;}
}
__global__ static void source_tree_level(int base,SourceNode *nodes,int *status){
    int at=base+blockIdx.x,k=threadIdx.x;if(k>source_degree||*status)return;
    const SourceNode &a=nodes[2*at],&b=nodes[2*at+1];SourceNode &out=nodes[at];
    if(a.width==0||b.width==0){
        const SourceNode &n=a.width==0?b:a;
        out.m[0][k]=n.m[0][k];out.m[1][k]=n.m[1][k];
        if(k==0){out.width=n.width;out.first=n.first;out.last=n.last;out.depth=n.depth;}
        return;
    }
    double width=a.width+b.width,l=a.width/width,r=b.width/width,lp[source_degree+2],rp[source_degree+2];
    lp[0]=rp[0]=1.;for(int j=1;j<=source_degree+1;j++){lp[j]=lp[j-1]*l;rp[j]=rp[j-1]*r;}
    Positive total[2]={Positive::zero(),Positive::zero()},term;
    for(int j=k;j<=source_degree;j++){
        double weight=lumina_source_measure::choose(j,k)*lp[k+1]*rp[j-k];
        for(int c=0;c<2;c++)if(!lumina_characteristic::scaled(a.m[c][j],weight,term)||!sum(total[c],term,total[c])){
            atomicCAS(status,0,49);return;
        }
    }
    for(int j=0;j<=k;j++){
        double weight=lumina_source_measure::choose(source_degree-j,k-j)*lp[k-j]*rp[source_degree-k+1];
        for(int c=0;c<2;c++)if(!lumina_characteristic::scaled(b.m[c][j],weight,term)||!sum(total[c],term,total[c])){
            atomicCAS(status,0,49);return;
        }
    }
    out.m[0][k]=total[0];out.m[1][k]=total[1];
    if(k==0){out.width=width;out.first=a.first;out.last=b.last;out.depth=1+max(a.depth,b.depth);}
}
#endif
extern "C" int lumina_linear_cuda_build(const CMFLinearCudaPiece *p,void *raw,int *status){
    if(!valid_piece(p)||!raw||!status)return 1;
    Node *nodes=static_cast<Node *>(raw);
#ifdef CMF_GPU_ENDPOINT_CACHE
    int prefix=endpoint_prefix(*p),count=prefix+(prefix<p->nf?1:0);
    fill_endpoints<<<(count+threads-1)/threads,threads>>>(*p,nodes,status);
    if(cudaGetLastError()!=cudaSuccess)return 2;
#endif
#ifdef CMF_GPU_SOURCE_MEASURE
    int rc=prepare_projection(*p);if(rc)return rc;
    SourceNode *source=reinterpret_cast<SourceNode *>(nodes+2*p->leaves);
    source_blocks<<<p->source_leaves,64>>>(*p,source,nodes,status);
    if(cudaGetLastError()!=cudaSuccess)return 2;
#endif
    build_blocks<<<(p->leaves+threads-1)/threads,threads>>>(*p,nodes,status);
    if(cudaGetLastError()!=cudaSuccess)return 2;
    for(int base=p->leaves/2;base;base/=2){
        tree_level<<<(base+threads-1)/threads,threads>>>(base,nodes,status);
        if(cudaGetLastError()!=cudaSuccess)return 2;
    }
#ifdef CMF_GPU_SOURCE_MEASURE
    for(int base=p->source_leaves/2;base;base/=2){
        source_tree_level<<<base,32>>>(base,source,status);
        if(cudaGetLastError()!=cudaSuccess)return 2;
    }
#endif
    return 0;
}
__global__ static void source_maps(CMFLinearCudaPiece p,const Node *nodes,double relative,int maximum,
    int first,int count,StoredMap *maps,int *status)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||*status)return;i+=first;
    Reader reader=make_reader(p,nodes);
    lumina_linear::Tree tree={nodes,p.leaves,p.block,p.nf};
    lumina_characteristic::Profile<Reader> profile={reader,tree,i,p.offset,p.span,p.length,p.da-p.db};
    // 처음부터 청색 끝을 넘는 열도 원래 마지막ν의 상수 연장이다.
    if(i>=p.nf){profile.column=0;profile.exit_shift=p.offset+i;}
    lumina_characteristic::Result result;
#ifdef CMF_GPU_SOURCE_MEASURE
    const SourceNode *source=reinterpret_cast<const SourceNode *>(nodes+2*p.leaves);
#ifdef CMF_GPU_COMPACT_MEASURE
    lumina_source_measure::Tree measure={source,p.source_leaves,32,p.nf};
#else
    lumina_source_measure::Tree measure={source,p.source_leaves,p.block,p.nf};
#endif
    int code=lumina_source_measure::integrate(profile,measure,Positive::zero(),relative,maximum,result);
#else
    int code=lumina_characteristic::integrate(profile,Positive::zero(),relative,maximum,result);
#endif
    if(code){atomicCAS(status,0,100+code);return;}
    lumina_linear::Integral tau;Positive t;
    if(!profile.tau(1.,tau)||!Positive::from_log(-tau.value,t)){atomicCAS(status,0,43);return;}
    using lumina_linear_affine::encode;
#ifdef CMF_GPU_COMPACT_MEASURE
    StoredMap m={encode(t),encode(result.source),encode(result.error),tau.error};
#else
    StoredMap m={encode(t),encode(result.source),encode(result.error),encode(result.roundoff),
        encode(result.frequency),encode(result.omitted),tau.error,0,0,
        result.panels,result.accepted,result.split,result.pruned,result.max_depth};
#endif
    maps[i]=m;
}
extern "C" int lumina_linear_cuda_query(const CMFLinearCudaPiece *p,const void *nodes,double relative,
    int maximum,void *maps,int *status)
{
    using lumina_ordered::finite;
    if(!valid_piece(p)||!nodes||!maps||!status||!finite(relative)||relative<=0||relative>.01||maximum<1||maximum>100000)return 1;
    for(int first=0;first<p->count;first+=chunk){
        int count=std::min(chunk,p->count-first);
        source_maps<<<(count+threads-1)/threads,threads>>>(*p,static_cast<const Node *>(nodes),relative,maximum,
            first,count,static_cast<StoredMap *>(maps),status);
        if(cudaGetLastError()!=cudaSuccess)return 2;
    }
    return 0;
}
#ifdef CMF_GPU_DIAGNOSTIC_RANGE
extern "C" int lumina_linear_cuda_query_range(const CMFLinearCudaPiece *p,const void *nodes,double relative,
    int maximum,int first,int count,void *maps,int *status)
{
    using lumina_ordered::finite;
    if(!valid_piece(p)||!nodes||!maps||!status||!finite(relative)||relative<=0||relative>.01||
       maximum<1||maximum>100000||first<0||count<1||count>1024||first>p->count-count)return 1;
    source_maps<<<(count+threads-1)/threads,threads>>>(*p,static_cast<const Node *>(nodes),relative,maximum,
        first,count,static_cast<StoredMap *>(maps),status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
#endif
__device__ static void maximum(double *address,double value){
    atomicMax(reinterpret_cast<unsigned long long *>(address),__double_as_longlong(value));
}
__global__ static void forward(int n,int width,int stride,int first,int count,const void *const *maps,
    const double *initial,Local *results,int *status)
{
    int f=first+blockIdx.x*blockDim.x+threadIdx.x;if(f>=first+count||*status)return;
    Positive current,numerical=Positive::zero();
    if(!Positive::from_double(initial[f],current)){atomicCAS(status,0,44);return;}
    for(int row=0;row<n;row++){
        Local r;Positive a;
        if(!lumina_linear_affine::leaf(static_cast<const StoredMap *>(maps[row]),stride,f,current,r)||
           !product(r.response,numerical,a)||!sum(a,r.numerical,numerical)){atomicCAS(status,0,45);return;}
        r.numerical=numerical;results[(size_t)row*width+f]=r;current=r.value;
    }
}
__global__ static void backward(int n,int width,int block_width,double relative,int first,int count,
    const Local *results,double *output,double *scores,double *ratios,Positive *frequency_errors,int *status)
{
    int f=first+blockIdx.x*blockDim.x+threadIdx.x;if(f>=first+count||*status)return;
    Local last=results[(size_t)(n-1)*width+f];Positive tolerance,tail=Positive::one(),geometry=Positive::zero(),a,b;
    if(!lumina_linear_affine::scale(last.value,relative,tolerance)){atomicCAS(status,0,46);return;}
    for(int row=n-1;row>=0;row--){
        const Local &r=results[(size_t)row*width+f];
        if(!product(r.geometry,tail,a)||!sum(geometry,a,geometry)||!product(tail,r.response,tail)){
            atomicCAS(status,0,46);return;
        }
        maximum(scores+row,lumina_linear_affine::ratio(a,tolerance));
    }
    if(!sum(geometry,last.numerical,a)||!lumina_linear_affine::scale(last.value,64.*(n+1)*DBL_EPSILON,b)||!sum(a,b,a)){
        atomicCAS(status,0,46);return;
    }
    double value=last.value.value();if(!lumina_ordered::finite(value)){atomicCAS(status,0,46);return;}
    output[f]=value;frequency_errors[f]=a;
    maximum(ratios+f/block_width,lumina_linear_affine::ratio(a,tolerance));
}
extern "C" int lumina_linear_cuda_replay(int n,int width,int stride,int block_width,double relative,
    const void *const *maps,const double *initial,double *output,void *raw_results,double *scores,
    double *ratios,void *errors,int *status)
{
    using lumina_ordered::finite;
    if(n<1||n>4096||width<1||width>2000000||stride<width||stride>2000000||
       (uint64_t)n*width>2097152||block_width<1||block_width>65536||!finite(relative)||relative<=0||relative>.01||
       !maps||!initial||!output||initial==output||!raw_results||!scores||!ratios||!errors||!status)return 1;
    int slabs=(width+block_width-1)/block_width;
    if(cudaMemsetAsync(scores,0,(size_t)n*sizeof(double))!=cudaSuccess||
       cudaMemsetAsync(ratios,0,(size_t)slabs*sizeof(double))!=cudaSuccess)return 2;
    Local *results=static_cast<Local *>(raw_results);
    for(int first=0;first<width;first+=chunk){
        int count=std::min(chunk,width-first);
        forward<<<(count+threads-1)/threads,threads>>>(n,width,stride,first,count,maps,initial,results,status);
        if(cudaGetLastError()!=cudaSuccess)return 2;
    }
    for(int first=0;first<width;first+=chunk){
        int count=std::min(chunk,width-first);
        backward<<<(count+threads-1)/threads,threads>>>(n,width,block_width,relative,first,count,results,output,scores,
            ratios,static_cast<Positive *>(errors),status);
        if(cudaGetLastError()!=cudaSuccess)return 2;
    }
    return 0;
}
__global__ static void weighted(int nf,const double *weights,const double *intensity,const Positive *errors,
    Positive *pairs,int *status)
{
    __shared__ Positive values[threads],bounds[threads];
    int lane=threadIdx.x,first=blockIdx.x*4096,last=first+4096;if(last>nf)last=nf;
    Positive v=Positive::zero(),e=Positive::zero(),w,a,b;
    for(int f=first+lane;f<last;f+=threads){
        Positive bound;
        CMFLinearPositive raw={errors[f].mantissa,errors[f].exponent};
        if(!Positive::from_double(weights[f],w)||!Positive::from_double(intensity[f],a)||!product(w,a,b)||
           !sum(v,b,v)||!lumina_linear_affine::decode(raw,bound)||
           !product(w,bound,b)||!sum(e,b,e)){atomicCAS(status,0,47);}
    }
    values[lane]=v;bounds[lane]=e;__syncthreads();
    for(int offset=threads/2;offset;offset/=2){
        if(lane<offset&&(!sum(values[lane],values[lane+offset],values[lane])||
                        !sum(bounds[lane],bounds[lane+offset],bounds[lane])))atomicCAS(status,0,47);
        __syncthreads();
    }
    if(lane==0){
        Positive rounding,total;
        if(!lumina_linear_affine::scale(values[0],64.*4096*DBL_EPSILON,rounding)||!sum(bounds[0],rounding,total)){
            atomicCAS(status,0,47);return;
        }
        pairs[2*blockIdx.x]=values[0];pairs[2*blockIdx.x+1]=total;
    }
}
extern "C" int lumina_linear_cuda_weighted(int nf,const double *weights,const double *intensity,const void *errors,
    void *pairs,int *status)
{
    if(nf<1||nf>2000000||!weights||!intensity||!errors||!pairs||!status)return 1;
    weighted<<<(nf+4095)/4096,threads>>>(nf,weights,intensity,static_cast<const Positive *>(errors),
        static_cast<Positive *>(pairs),status);
    return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int lumina_linear_cuda_aux_workspace(size_t *local);
extern "C" int lumina_linear_cuda_workspace(size_t *local){
    if(!local)return 1;
    const void *kernels[]={(const void *)build_blocks,(const void *)tree_level,(const void *)source_maps,
        (const void *)forward,(const void *)backward,(const void *)weighted
#ifdef CMF_GPU_SOURCE_MEASURE
        ,(const void *)source_blocks,(const void *)source_tree_level
#endif
#ifdef CMF_GPU_ENDPOINT_CACHE
        ,(const void *)fill_endpoints
#endif
    };
    size_t maximum=0,aux;
    for(const void *kernel:kernels){
        cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,kernel)!=cudaSuccess)return 2;
        maximum=std::max(maximum,a.localSizeBytes);
    }
    if(lumina_linear_cuda_aux_workspace(&aux))return 2;
    *local=std::max(maximum,aux);return 0;
}
