#include "cmf_linear_endpoint_cache.hpp"
#include "cmf_linear_provider.hpp"
#include <vector>
#include <algorithm>
extern "C" int endpoint_cache_prefix_probe(int nf,int bins,int chi_block,int source_block,
    double offset,double span,int count){
    return lumina_endpoint_cache::required_prefix(nf,bins,chi_block,source_block,offset,span,count);
}
#include <cstring>
extern "C" int endpoint_cache_probe(const CMFNodalDenseView *view,const CMFNodalLocation *locations,
    double *out,int count){
    if(!view||!locations||!out||!lumina_native_linear::valid(*view,locations)||count!=4*view->nf)return 1;
    try{
        int nf=view->nf;std::vector<double> values((size_t)4*nf);
        lumina_native_linear::Reader original={*view,{locations[0],locations[1]},0};
        for(int j=0;j<nf;j++)if(!lumina_endpoint_cache::fill(original,nf,j,values.data()))return 2;
        lumina_endpoint_cache::Reader cached={values.data(),nf,nf};
        for(int j=0;j<nf;j++)for(int c=0;c<2;c++){
            double x,y,a,b;
            if(!original.at(2*c,j,x,y)||!cached.at(2*c,j,a,b)||
               std::memcmp(&x,&a,sizeof(double))||std::memcmp(&y,&b,sizeof(double)))return 3;
        }
        std::copy(values.begin(),values.end(),out);return 0;
    }catch(...){return 9;}
}
extern "C" int endpoint_cache_partial_probe(const double *values,int nf,int filled,int point,int column,double *out){
    if(!out)return 1;
    double chi,eta;lumina_endpoint_cache::Reader reader={values,nf,filled};
    if(!reader.at(point,column,chi,eta))return 2;
    out[0]=chi;out[1]=eta;return 0;
}
