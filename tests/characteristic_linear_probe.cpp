#include "cmf_characteristic_linear.hpp"
#include <vector>
#include <cstring>
struct Reader {
    int nf;const double *values;
    bool at(int point,int column,double &chi,double &eta) const {
        if((point!=0&&point!=2)||column<0||column>=nf)return false;
        int k=point==0?0:1;chi=values[(size_t)k*nf+column];eta=values[(size_t)(k+2)*nf+column];
        return lumina_ordered::finite(chi)&&lumina_ordered::finite(eta)&&eta>=0;
    }
};
struct RefineUntilCoordinateLimit {
    template<class R> lumina_characteristic::PanelStatus operator()(
        const lumina_characteristic::Profile<R> &p,double a,double b,double,
        lumina_characteristic::Panel &out) const {
        lumina_linear::Node node;
        if(!p.range(a,b,node))return lumina_characteristic::INVALID;
        out={};out.upper=lumina_ordered::Positive::one();
        return lumina_characteristic::SPLIT;
    }
};
extern "C" int lumina_characteristic_resolution_probe(int invalid_reader){
    double values[8]={.1,.1,.1,.1,1.,1.,1.,1.};
    if(invalid_reader)values[5]=-1.;
    Reader reader={2,values};lumina_linear::Node nodes[2]={};
    lumina_linear::Tree tree={nodes,1,1,2};
    lumina_characteristic::Profile<Reader> p={reader,tree,0,1000.,1.,1.,0.};
    lumina_characteristic::Result result={};result.source=lumina_ordered::Positive::one();
    auto status=lumina_characteristic::integrate_using(p,lumina_ordered::Positive::zero(),
        1e-7,10000,result,RefineUntilCoordinateLimit{});
    if(result.source.mantissa!=.5||result.source.exponent!=1)return 99;
    return status;
}
extern "C" int lumina_characteristic_linear_probe(int nf,int block,const double *values,const double *parameters,
    int max_panels,double *mantissas,std::int64_t *exponents,int *counts)
{
    using namespace lumina_linear;
    if(nf<2||nf>2049||block<1||block>256||!values||!parameters||!mantissas||!exponents||!counts)return 99;
    try{
        Reader reader={nf,values};int blocks=(nf-2)/block+1,leaves=1;while(leaves<blocks)leaves*=2;
        std::vector<Node> nodes((size_t)2*leaves,empty());
        for(int i=0;i<blocks;i++){
            Node whole=empty(),part,next;int end=(i+1)*block;if(end>nf-1)end=nf-1;
            for(int bin=i*block;bin<end;bin++){
                if(!partial(reader,nf,bin,0.,1.,part)||!combine(whole,part,next))return 98;
                whole=next;
            }
            nodes[leaves+i]=whole;
        }
        for(int i=leaves-1;i>0;i--)if(!combine(nodes[2*i],nodes[2*i+1],nodes[i]))return 98;
        Tree tree={nodes.data(),leaves,block,nf};
        lumina_characteristic::Profile<Reader> profile={reader,tree,0,parameters[1],parameters[2],parameters[0],parameters[3]};
        lumina_ordered::Positive incident;
        if(!lumina_ordered::Positive::from_double(parameters[4],incident))return 97;
        lumina_characteristic::Result result={};
        int status=lumina_characteristic::integrate(profile,incident,parameters[5],max_panels,result);
        if(status)return status;
        lumina_ordered::Positive fields[7]={result.value,result.transmitted,result.source,result.error,
            result.roundoff,result.frequency,result.omitted};
        for(int i=0;i<7;i++){mantissas[i]=fields[i].mantissa;exponents[i]=fields[i].exponent;}
        int c[5]={result.panels,result.accepted,result.split,result.pruned,result.max_depth};
        std::memcpy(counts,c,sizeof(c));return 0;
    }catch(...){return 96;}
}
