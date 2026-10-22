#include "cmf_native_bin_stream.hpp"
#include <vector>
#include <cstring>
namespace {
struct Reader {
    int nf;const double *v;
    bool at(int point,int column,double &chi,double &eta) const {
        if((point!=0&&point!=2)||column<0||column>=nf)return false;
        int k=point/2;chi=v[(size_t)k*nf+column];eta=v[(size_t)(k+2)*nf+column];
        return lumina_ordered::finite(chi)&&lumina_ordered::finite(eta)&&eta>=0;
    }
};
}
extern "C" int native_bin_stream_probe(int nf,int block,const double *values,const double *params,int limit,
    double *m,int64_t *e,int *counts){
    using namespace lumina_linear;
    if(nf<2||nf>2049||block<1||block>256||!values||!params||!m||!e||!counts)return 99;
    Reader reader={nf,values};int blocks=(nf-2)/block+1,leaves=1;while(leaves<blocks)leaves*=2;
    std::vector<Node> nodes((size_t)2*leaves,empty());
    for(int i=0;i<blocks;i++){
        Node n=empty();
        for(int j=i*block;j<std::min(nf-1,(i+1)*block);j++){
            Node part,next;if(!partial(reader,nf,j,0.,1.,part)||!combine(n,part,next))return 98;n=next;
        }
        nodes[leaves+i]=n;
    }
    for(int i=leaves-1;i>0;i--)if(!combine(nodes[2*i],nodes[2*i+1],nodes[i]))return 98;
    Tree tree={nodes.data(),leaves,block,nf};
    lumina_characteristic::Profile<Reader> p={reader,tree,0,params[1],params[2],params[0],params[3]};
    lumina_ordered::Positive initial;if(!lumina_ordered::Positive::from_double(params[4],initial))return 97;
    lumina_characteristic::Result r;int code=lumina_native_bin::integrate(p,initial,params[5],limit,r);
    if(code)return code;
    lumina_ordered::Positive fields[7]={r.value,r.transmitted,r.source,r.error,r.roundoff,r.frequency,r.omitted};
    for(int i=0;i<7;i++){m[i]=fields[i].mantissa;e[i]=fields[i].exponent;}
    int c[5]={r.panels,r.accepted,r.split,r.pruned,r.max_depth};std::memcpy(counts,c,sizeof(c));return 0;
}
