#include "cmf_linear_provider.hpp"
#include "cmf_characteristic_linear.hpp"
/* 실제 GPU 수치 실행 없는 native provider+B 적응 적분의 device 연결 확인. */
__global__ void lumina_native_linear_compile(CMFNodalDenseView view,CMFNodalLocation entrance,
    CMFNodalLocation exit,const lumina_linear::Node *nodes,lumina_characteristic::Result *out,int *status)
{
    if(blockIdx.x||threadIdx.x)return;
    CMFNodalLocation locations[2]={entrance,exit};
    if(!lumina_native_linear::valid(view,locations)){*status=1;return;}
    lumina_native_linear::Reader reader={view,{entrance,exit},0};
    lumina_linear::Tree tree={nodes,1,32,view.nf};
    lumina_characteristic::Profile<lumina_native_linear::Reader> profile={reader,tree,0,.25,1.5,2.,.01};
    *status=lumina_characteristic::integrate(profile,lumina_ordered::Positive::zero(),1e-6,10000,*out);
}
