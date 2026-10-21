#ifndef CMF_LINEAR_PROVIDER_HPP
#define CMF_LINEAR_PROVIDER_HPP
#include "cmf_nodal_input.h"
#include "cmf_ordered_maps.hpp"
#include <cstddef>
#ifdef __CUDACC__
#define LUMINA_PROVIDER __host__ __device__ inline
#else
#define LUMINA_PROVIDER inline
#endif
namespace lumina_native_linear {
using lumina_ordered::finite;
LUMINA_PROVIDER bool valid(const CMFNodalDenseView &v,const CMFNodalLocation *p){
    if(v.ns<2||v.ns>512||v.nf<2||v.nf>2000000||v.ne<2||v.ne>65||!v.generation||
       !v.canonical||!v.outer||(v.jmode!=0&&v.jmode!=1)||(v.jmode==1&&!v.j)||!p)return false;
    for(int k=0;k<2;k++){
        auto a=p[k];
        if(!finite(a.weight)||a.weight<0||a.weight>1||a.first<0||a.last<0||a.mode<0||a.mode>2||
           (a.mode==0&&(a.first!=v.ns-1||a.last!=a.first))||
           (a.mode==1&&(a.first!=a.last+1||a.first>=v.ns))||
           (a.mode==2&&(a.first!=a.last+1||a.first>=v.ne)))return false;
    }
    return true;
}
/* 기존 native C consumer와 같은 순서다. native 반경에서 진η/ES/J를
 * 각각 보간한 뒤 ES*J를 조립하며 주파수 보간은 상위 코어가 나중에 한다.
 * offset은 CPU 소수 열 시험의 국소 트리만 위한 원래 ν index 이동이다. */
struct Reader {
    CMFNodalDenseView view;CMFNodalLocation location[2];int offset;
    LUMINA_PROVIDER bool at(int point,int column,double &chi,double &eta) const {
        if((point!=0&&point!=2)||offset<0||offset>view.nf-2||column<0||column>=view.nf-offset)return false;
        column+=offset;auto p=location[point==0?0:1];int f=p.first,l=p.last;double w=p.weight;
        if(p.mode==2){
            const double *v=view.outer+(size_t)column*(2*view.ne+1);
            chi=v[2*view.ne]*::exp((1.-w)*v[f]+w*v[l]);
            double e=w==0?v[view.ne+f]:w==1?v[view.ne+l]:(1.-w)*v[view.ne+f]+w*v[view.ne+l];
            eta=::exp(e);
        }else{
            size_t n=(size_t)view.ns*view.nf,a=(size_t)f*view.nf+column,b=(size_t)l*view.nf+column;
            double s0=view.canonical[n+a],s1=view.canonical[n+b];
            double j0=view.jmode?view.j[a]:0.,j1=view.jmode?view.j[b]:0.;
            if(!finite(j0)||j0<0||!finite(j1)||j1<0||s0<=0||s1<=0)return false;
            if(p.mode==0){chi=view.canonical[a];eta=view.canonical[2*n+a]+s0*j0;}
            else{
                chi=(1.-w)*view.canonical[a]+w*view.canonical[b];
                eta=(1.-w)*view.canonical[2*n+a]+w*view.canonical[2*n+b]+
                    ((1.-w)*s0+w*s1)*((1.-w)*j0+w*j1);
            }
        }
        return finite(chi)&&finite(eta)&&eta>=0;
    }
};
}
#undef LUMINA_PROVIDER
#endif
