// CPU 비교도 GPU provider와 같은 Reader 및 끝점 주파수 선형식을 호출한다.
#include "cmf_linear_provider.hpp"
extern "C" int lumina_characteristic_samples(int n,const CMFNodalDenseView *view,
    const double *weights,const double *columns,double *chi,double *eta){
    if(n<1||n>65536||!view||!weights||!columns||!chi||!eta)return 1;
    for(int k=0;k<n;k++){
        double coordinate=columns[k];
        if(!lumina_ordered::finite(coordinate)||coordinate<0||coordinate>view->nf-1)return 1;
        CMFNodalLocation p={1,1,0,weights[k]};
        lumina_native_linear::Reader reader={*view,{p,p},0};
        if(!lumina_native_linear::valid(*view,reader.location))return 1;
        int low=(int)::floor(coordinate);double f=coordinate-low,c0,e0,c1,e1;
        if(!reader.at(0,low,c0,e0))return 2;
        if(f==0){chi[k]=c0;eta[k]=e0;}
        else{
            if(!reader.at(0,low+1,c1,e1))return 2;
            chi[k]=c0+f*(c1-c0);eta[k]=e0+f*(e1-e0);
        }
        if(!lumina_ordered::finite(chi[k])||!lumina_ordered::finite(eta[k])||eta[k]<0)return 2;
    }
    return 0;
}
