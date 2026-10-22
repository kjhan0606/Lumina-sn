#include "cmf_characteristic_linear.hpp"
#include <vector>
#include <cstring>
#include <algorithm>
struct Reader {
    int nf;const double *values;
    bool at(int point,int column,double &chi,double &eta) const {
        if((point!=0&&point!=2)||column<0||column>=nf)return false;
        int k=point/2;chi=values[(size_t)k*nf+column];eta=values[(size_t)(k+2)*nf+column];
        return lumina_ordered::finite(chi)&&lumina_ordered::finite(eta)&&eta>=0;
    }
};
extern "C" int chi_tau_node_bytes(){return sizeof(lumina_linear::ChiNode);}
extern "C" int chi_tau_probe(int nf,int block,const double *values,double shift,double span,
    double length,double delta,double u,double *out){
    using namespace lumina_linear;
    using lumina_ordered::finite;
    if(!values||!out||nf<2||nf>2049||block<1||block>256||!finite(u)||u<=0||u>1)return 1;
    try{
        Reader r={nf,values};int blocks=(nf-2)/block+1,leaves=1;while(leaves<blocks)leaves*=2;
        std::vector<Node> nodes((size_t)2*leaves,empty());
        for(int i=0;i<blocks;i++){
            Node whole=empty(),part,next;int end=std::min(nf-1,(i+1)*block);
            for(int j=i*block;j<end;j++){
                if(!partial(r,nf,j,0.,1.,part)||!combine(whole,part,next))return 2;
                whole=next;
            }
            nodes[leaves+i]=whole;
        }
        for(int i=leaves-1;i>0;i--)if(!combine(nodes[2*i],nodes[2*i+1],nodes[i]))return 2;
        Tree tree={nodes.data(),leaves,block,nf};
        lumina_characteristic::Profile<Reader> p={r,tree,0,shift,span,length,delta};
#ifdef CMF_GPU_CHI_ONLY_TAU
        ChiNode n;
#else
        Node n;
#endif
        Integral tau;
        if(!p.range(0.,u,n)||!p.tau(u,tau))return 3;
        double result[20]={n.width,double(n.depth)};
        for(int k=0;k<2;k++){
            const Trace &t=n.trace[k];double fields[8]={t.p,t.q,t.mag_p,t.mag_q,t.lo,t.hi,t.slo,t.shi};
            std::memcpy(result+2+8*k,fields,sizeof(fields));
        }
        result[18]=tau.value;result[19]=tau.error;
        std::memcpy(out,result,sizeof(result));return 0;
    }catch(...){return 9;}
}
