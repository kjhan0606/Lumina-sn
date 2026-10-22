#include "cmf_linear_cuda.h"
#include "cmf_linear_provider.hpp"
#include "cmf_source_measure.hpp"
#include "cmf_source_projection.hpp"
#include "cmf_linear_endpoint_cache.hpp"
#include <algorithm>
#include <chrono>
#include <vector>

namespace {
struct Reader {
    lumina_native_linear::Reader original;uint64_t *reads;
    const double *endpoints=nullptr;int nf=0;
    bool at(int point,int column,double &chi,double &eta) const {
        ++*reads;
        if(endpoints)return lumina_endpoint_cache::Reader{endpoints,nf,nf}.at(point,column,chi,eta);
        return original.at(point,column,chi,eta);
    }
};
}
/* 같은 전역 트리 좌표의 덮개만 CPU에서 만든다. p.offset에 원 column을 더하지 않는다. */
static int execute(const CMFLinearCudaPiece *p,int column,CMFLinearNativeMap *out,
    uint64_t *stats,double *times,bool global)
{
    using Clock=std::chrono::steady_clock;
    using namespace lumina_linear;
    using lumina_ordered::Positive;
    using lumina_ordered::finite;
    if(!p||!out||!stats||!times||!lumina_native_linear::valid(p->view,p->locations)||
       column<0||column>=p->count||p->count<1||p->count>p->view.nf||p->block!=2||p->base<0||p->nf!=p->view.nf-p->base||
       p->nf<2||p->base>p->view.nf-2||!finite(p->offset)||p->offset<0||p->offset>2000000000.||
       !finite(p->span)||p->span<0||p->span>2000000000.||!finite(p->length)||p->length<=0||
       !finite(p->da)||!finite(p->db)||p->db<0||p->da<p->db)return 1;
    auto start=Clock::now();uint64_t reads=0;
    try{
        bool blue=column>=p->nf;
        int anchor=blue||global?0:std::min(column/32*32,(p->nf-2)/32*32),nf=p->nf-anchor;
        int local_column=blue?0:column-anchor;
        double offset=blue?p->offset+column:p->offset;
        Reader reader={{p->view,{p->locations[0],p->locations[1]},p->base+anchor},&reads};
#ifdef CMF_GPU_ENDPOINT_CACHE
        std::vector<double> endpoints((size_t)4*nf);
        for(int column=0;column<nf;column++)
            if(!lumina_endpoint_cache::fill(reader.original,nf,column,endpoints.data()))return 3;
        reader.endpoints=endpoints.data();reader.nf=nf;
#endif
        int bins=p->span==0||local_column+offset>=nf-1?0:
            (int)::ceil(::fmin(nf-1.,local_column+offset+p->span));
        if(global)bins=p->bins;
        int blocks=(bins+1)/2,leaves=1;while(leaves<blocks)leaves*=2;
        int source_blocks=(bins+31)/32,source_leaves=1;while(source_leaves<source_blocks)source_leaves*=2;
        uint64_t bytes=uint64_t(2)*leaves*sizeof(Node)+uint64_t(2)*source_leaves*sizeof(lumina_source_measure::Node);
#ifdef CMF_GPU_ENDPOINT_CACHE
        bytes+=(uint64_t)4*nf*sizeof(double);
#endif
        if(bytes>(UINT64_C(1)<<30))return 2;
        std::vector<Node> nodes((size_t)2*leaves,empty());
        std::vector<lumina_source_measure::Node> sources((size_t)2*source_leaves,lumina_source_measure::empty());
        uint64_t built=0,combinations=0;
        for(int i=0;i<blocks;i++){
            Node n=empty();
            for(int j=2*i;j<std::min(nf-1,2*i+2);j++){
                Node part,next;if(!partial(reader,nf,j,0.,1.,part)||!combine(n,part,next))return 3;
                n=next;built++;combinations++;
            }
            nodes[leaves+i]=n;
        }
        for(int i=leaves-1;i>0;i--){
            if(!combine(nodes[2*i],nodes[2*i+1],nodes[i]))return 3;
            combinations++;
        }
        for(int i=0;i<source_blocks;i++)
            if(!lumina_source_projection::block(reader,nf,32*i,std::min(nf-1,32*i+32),sources[source_leaves+i]))return 3;
        for(int i=source_leaves-1;i>0;i--)
            if(!lumina_source_measure::combine(sources[2*i],sources[2*i+1],sources[i]))return 3;
        auto ready=Clock::now();uint64_t build_reads=reads;
        Tree tree={nodes.data(),leaves,2,nf};
        lumina_source_measure::Tree source={sources.data(),source_leaves,32,nf};
        lumina_characteristic::Profile<Reader> profile={reader,tree,local_column,offset,p->span,p->length,p->da-p->db};
        lumina_characteristic::Result r={};
        int status=lumina_source_measure::integrate(profile,source,Positive::zero(),1e-8,100000,r);
        auto finish=[&](int code){
            auto done=Clock::now();stats[0]=reads;stats[1]=build_reads;stats[2]=bytes;stats[3]=anchor;
            times[0]=std::chrono::duration<double>(ready-start).count();times[1]=std::chrono::duration<double>(done-ready).count();
            return code;
        };
        if(status)return finish(100+status);
        Integral depth;Positive t;
        if(!profile.tau(1.,depth)||!Positive::from_log(-depth.value,t))return finish(4);
        auto put=[](Positive v)->CMFLinearPositive{return {v.mantissa,v.exponent};};
        CMFLinearNativeMap result={put(t),put(r.source),put(r.error),put(r.roundoff),put(r.frequency),put(r.omitted),
            depth.error,built,combinations,r.panels,r.accepted,r.split,r.pruned,r.max_depth};
        *out=result;return finish(0);
    }catch(...){return 9;}
}
extern "C" int native_cost_probe(const CMFLinearCudaPiece *p,int column,CMFLinearNativeMap *out,uint64_t *stats,double *times){
    return execute(p,column,out,stats,times,false);
}
extern "C" int native_cost_reference(const CMFLinearCudaPiece *p,int column,CMFLinearNativeMap *out,uint64_t *stats,double *times){
    return execute(p,column,out,stats,times,true);
}
