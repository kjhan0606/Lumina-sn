#include "cmf_linear_provider.hpp"
extern "C" int lumina_linear_provider_probe(const CMFNodalDenseView *view,const CMFNodalLocation *locations,
    int point,int column,double *out)
{
    if(!view||!out||!lumina_native_linear::valid(*view,locations))return 1;
    lumina_native_linear::Reader reader={*view,{locations[0],locations[1]},0};
    double chi,eta;if(!reader.at(point,column,chi,eta))return 2;
    out[0]=chi;out[1]=eta;return 0;
}
