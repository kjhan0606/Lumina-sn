// 국소 비교도 생산 Reader의 반경 보간과 ES*J 조립을 그대로 소비한다.
#include "cmf_linear_provider.hpp"
extern "C" int lumina_radial_packet_sample(int n,const double *weight,
    const double *input,double *chi,double *eta){
    if(n<1||n>65536||!weight||!input||!chi||!eta)return 1;
    double canonical[12],outer[10]={},j[4],radius[2]={2.,1.};
    for(int point=0;point<2;point++)for(int frequency=0;frequency<2;frequency++){
        int at=(1-point)*2+frequency;
        canonical[at]=input[point];
        canonical[4+at]=input[4+point];
        canonical[8+at]=input[2+point];
        j[at]=input[6+point];
    }
    CMFNodalDenseView view={2,2,2,1,1,canonical,outer,j,radius,radius};
    for(int k=0;k<n;k++){
        CMFNodalLocation location[2]={{1,1,0,weight[k]},{1,1,0,weight[k]}};
        if(!lumina_native_linear::valid(view,location))return 1;
        lumina_native_linear::Reader reader={view,{location[0],location[1]},0};
        if(!reader.at(0,0,chi[k],eta[k]))return 2;
    }
    return 0;
}
