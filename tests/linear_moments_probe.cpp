#include "cmf_linear_moments.hpp"
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
extern "C" int lumina_linear_moments_probe(int nf,int block,const double *values,int nq,const int *bins,
    const double *fractions,const double *paths,double *output)
{
    using namespace lumina_linear;
    if(nf<2||nf>2049||block<1||block>256||nq<1||nq>4096||!values||!bins||!fractions||!paths||!output)return 1;
    try{
        Reader reader={nf,values};int blocks=(nf-2)/block+1,leaves=1;while(leaves<blocks)leaves*=2;
        std::vector<Node> nodes((size_t)2*leaves,empty());
        for(int i=0;i<blocks;i++){
            Node whole=empty(),part,next;int end=(i+1)*block;if(end>nf-1)end=nf-1;
            for(int bin=i*block;bin<end;bin++){
                if(!partial(reader,nf,bin,0.,1.,part)||!combine(whole,part,next))return 2;
                whole=next;
            }
            nodes[leaves+i]=whole;
        }
        for(int i=leaves-1;i>0;i--)if(!combine(nodes[2*i],nodes[2*i+1],nodes[i]))return 2;
        Tree tree={nodes.data(),leaves,block,nf};std::vector<double> results((size_t)nq*34);
        for(int i=0;i<nq;i++){
            Node n;Integral depth,mass;
            if(!query(reader,tree,{bins[2*i],fractions[2*i]},{bins[2*i+1],fractions[2*i+1]},n)||
               !path_integral(n,0,paths[3*i],paths[3*i+1],paths[3*i+2],depth)||
               !path_integral(n,2,paths[3*i],paths[3*i+1],paths[3*i+2],mass))return 3;
            double *out=results.data()+34*i;out[0]=depth.value;out[1]=depth.error;out[2]=mass.value;
            out[3]=mass.error;out[4]=n.width;out[5]=n.depth;
            for(int k=0;k<4;k++){
                const Trace &t=n.trace[k];double *v=out+6+7*k;
                v[0]=t.p;v[1]=t.q;v[2]=t.lo;v[3]=t.hi;v[4]=t.slo;v[5]=t.shi;v[6]=secant_error(t,n.width);
            }
        }
        std::memcpy(output,results.data(),results.size()*sizeof(double));return 0;
    }catch(...){return 4;}
}
