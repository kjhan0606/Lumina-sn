#include "cmf_source_measure.hpp"
#include "cmf_source_projection.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

namespace {
struct Reader {
    int nf;const double *v;uint64_t *reads;
    bool at(int point,int column,double &chi,double &eta) const {
        ++*reads;
        if((point!=0&&point!=2)||column<0||column>=nf)return false;
        int k=point/2;chi=v[(size_t)k*nf+column];eta=v[(size_t)(k+2)*nf+column];
        return lumina_ordered::finite(chi)&&lumina_ordered::finite(eta)&&eta>=0;
    }
};
int execute(int nf,int block,const double *values,const double *params,int limit,
    double *mantissa,int64_t *exponent,int *counts,double *timings,uint64_t *stats,int eta_block=0){
    using Clock=std::chrono::steady_clock;
    auto start=Clock::now();
    if(eta_block==0)eta_block=block;
    if(nf<2||nf>2000000||block<1||block>256||eta_block<1||eta_block>256||!values||!params||!mantissa||!exponent||!counts)return 1;
    try{
        int blocks=(nf-2)/block+1,leaves=1;while(leaves<blocks)leaves*=2;
        int eta_blocks=(nf-2)/eta_block+1,eta_leaves=1;while(eta_leaves<eta_blocks)eta_leaves*=2;
        uint64_t bytes=uint64_t(2)*leaves*sizeof(lumina_linear::Node)+uint64_t(2)*eta_leaves*sizeof(lumina_source_measure::Node);
        if(bytes>(UINT64_C(1)<<30))return 1;
        uint64_t reads=0;Reader reader={nf,values,&reads};
        std::vector<lumina_linear::Node> nodes((size_t)2*leaves,lumina_linear::empty());
        std::vector<lumina_source_measure::Node> sources((size_t)2*eta_leaves,lumina_source_measure::empty());
        for(int i=0;i<blocks;i++){
            auto chi=lumina_linear::empty();
            int end=std::min(nf-1,(i+1)*block);
            for(int bin=i*block;bin<end;bin++){
                lumina_linear::Node p,n;
                if(!lumina_linear::partial(reader,nf,bin,0.,1.,p)||!lumina_linear::combine(chi,p,n))return 2;
                chi=n;
            }
            nodes[leaves+i]=chi;
        }
        for(int i=leaves-1;i>0;i--){
            if(!lumina_linear::combine(nodes[2*i],nodes[2*i+1],nodes[i]))return 2;
        }
        for(int i=0;i<eta_blocks;i++)
            if(!lumina_source_projection::block(reader,nf,i*eta_block,std::min(nf-1,(i+1)*eta_block),sources[eta_leaves+i]))return 2;
        for(int i=eta_leaves-1;i>0;i--)
            if(!lumina_source_measure::combine(sources[2*i],sources[2*i+1],sources[i]))return 2;
        uint64_t build_reads=reads;auto built=Clock::now();
        lumina_linear::Tree tree={nodes.data(),leaves,block,nf};
        lumina_source_measure::Tree source={sources.data(),eta_leaves,eta_block,nf};
        lumina_characteristic::Profile<Reader> p={reader,tree,0,params[1],params[2],params[0],params[3]};
        lumina_ordered::Positive incident;
        if(!lumina_ordered::Positive::from_double(params[4],incident))return 1;
        lumina_characteristic::Result result;
        int status=lumina_source_measure::integrate(p,source,incident,params[5],limit,result);
        if(status)return 100+status;
        auto done=Clock::now();
        lumina_ordered::Positive output[7]={result.value,result.transmitted,result.source,result.error,
            result.roundoff,result.frequency,result.omitted};
        for(int k=0;k<7;k++){mantissa[k]=output[k].mantissa;exponent[k]=output[k].exponent;}
        counts[0]=result.panels;counts[1]=result.accepted;counts[2]=result.split;
        counts[3]=result.pruned;counts[4]=result.max_depth;
        if(timings){timings[0]=std::chrono::duration<double>(built-start).count();
            timings[1]=std::chrono::duration<double>(done-built).count();}
        if(stats){stats[0]=reads;stats[1]=build_reads;stats[2]=bytes;
            stats[3]=sizeof(lumina_source_measure::Node);}
        return 0;
    }catch(...){return 9;}
}
}
extern "C" int source_measure_probe(int nf,int block,const double *values,const double *params,int limit,
    double *mantissa,int64_t *exponent,int *counts){
    return execute(nf,block,values,params,limit,mantissa,exponent,counts,nullptr,nullptr);
}
extern "C" int source_measure_split_probe(int nf,int block,int eta_block,const double *values,const double *params,int limit,
    double *mantissa,int64_t *exponent,int *counts){
    return execute(nf,block,values,params,limit,mantissa,exponent,counts,nullptr,nullptr,eta_block);
}
extern "C" int source_measure_cost(int nf,int block,const double *values,const double *params,int limit,
    double *mantissa,int64_t *exponent,int *counts,double *timings,uint64_t *stats){
    return execute(nf,block,values,params,limit,mantissa,exponent,counts,timings,stats);
}
extern "C" int source_measure_split_cost(int nf,int block,int eta_block,const double *values,const double *params,int limit,
    double *mantissa,int64_t *exponent,int *counts,double *timings,uint64_t *stats){
    return execute(nf,block,values,params,limit,mantissa,exponent,counts,timings,stats,eta_block);
}
extern "C" int source_projection_weights(int width,double *out,int count){
    if(width<1||width>256||!out||count!=(width+1)*(lumina_source_measure::degree+1))return 1;
    try{const auto &m=lumina_source_projection::matrix(width);std::copy(m.weights.begin(),m.weights.end(),out);return 0;}
    catch(...){return 2;}
}
extern "C" int source_projection_moments(int width,const double *values,int direct,double *mantissa,int64_t *exponent){
    if(width<1||width>256||!values||!mantissa||!exponent)return 1;
    try{
        uint64_t reads=0;Reader r={width+1,values,&reads};auto n=lumina_source_measure::empty();
        if(direct){if(!lumina_source_projection::block(r,width+1,0,width,n))return 2;}
        else for(int j=0;j<width;j++){
            lumina_source_measure::Node p,next;
            if(!lumina_source_measure::partial(r,width+1,j,0.,1.,p)||!lumina_source_measure::combine(n,p,next))return 2;
            n=next;
        }
        for(int c=0;c<2;c++)for(int k=0;k<=lumina_source_measure::degree;k++){
            int i=c*(lumina_source_measure::degree+1)+k;
            mantissa[i]=n.m[c][k].mantissa;exponent[i]=n.m[c][k].exponent;
        }
        return 0;
    }catch(...){return 3;}
}
