#include "cmf_linear_native.h"
#include "cmf_linear_provider.hpp"
#include "cmf_source_measure.hpp"
#include "cmf_source_projection.hpp"
#include <algorithm>
#include <vector>

extern "C" size_t cmf_measure_native_map_bytes(void){return sizeof(CMFLinearNativeMap);}
/* 원래 native 반경/provider와 B 기하를 보존하고 원천 적분만 선택한다. */
extern "C" int cmf_measure_native_map(const CMFNodalDenseView *view,uint64_t generation,const CMFNodalLocation *locations,
    double length,double h,double da,double db,int column,int block,double relative,int max_panels,CMFLinearNativeMap *out){
    using namespace lumina_linear;
    using lumina_ordered::Positive;
    using lumina_ordered::finite;
    if(!view||!out||!lumina_native_linear::valid(*view,locations)||view->generation!=generation||
       !finite(length)||length<=0||!finite(h)||h<=0||!finite(da)||!finite(db)||db<0||da<db||
       column<0||column>=view->nf||block<1||block>256)return 1;
    double shift=db/h,span=(da-db)/h;
    if(!finite(shift)||!finite(span)||shift>2000000000.||span>2000000000.)return 1;
    try{
        int whole=(int)::floor(shift),base=std::min(view->nf-2,column+whole);
        double offset=shift-whole+double(column+whole-base);
        lumina_native_linear::Reader reader={*view,{locations[0],locations[1]},base};
        int nf=view->nf-base;
        int bins=span==0||offset>=nf-1?0:(int)::ceil(::fmin(nf-1.,offset+span));
        int blocks=bins?(bins-1)/block+1:0,leaves=1;while(leaves<blocks)leaves*=2;
        if((uint64_t)2*leaves*(sizeof(Node)+sizeof(lumina_source_measure::Node))>(UINT64_C(1)<<30))return 6;
        std::vector<Node> nodes((size_t)2*leaves,empty());
        std::vector<lumina_source_measure::Node> sources((size_t)2*leaves,lumina_source_measure::empty());
        uint64_t built=0,combinations=0;
        for(int i=0;i<blocks;i++){
            Node whole_node=empty();lumina_source_measure::Node eta;
            int end=std::min(nf-1,(i+1)*block);
            for(int bin=i*block;bin<end;bin++){
                Node part,next;
                if(!partial(reader,nf,bin,0.,1.,part)||!combine(whole_node,part,next))return 2;
                whole_node=next;built++;combinations++;
            }
            if(!lumina_source_projection::block(reader,nf,i*block,end,eta))return 2;
            nodes[leaves+i]=whole_node;sources[leaves+i]=eta;
        }
        for(int i=leaves-1;i>0;i--){
            if(!combine(nodes[2*i],nodes[2*i+1],nodes[i])||
               !lumina_source_measure::combine(sources[2*i],sources[2*i+1],sources[i]))return 2;
            combinations++;
        }
        Tree tree={nodes.data(),leaves,block,nf};
        lumina_source_measure::Tree source={sources.data(),leaves,block,nf};
        lumina_characteristic::Profile<lumina_native_linear::Reader> profile={reader,tree,0,offset,span,length,da-db};
        lumina_characteristic::Result integral;
        int status=lumina_source_measure::integrate(profile,source,Positive::zero(),relative,max_panels,integral);
        if(status)return 100+status;
        Integral depth;Positive t;
        if(!profile.tau(1.,depth)||!Positive::from_log(-depth.value,t))return 3;
        auto put=[](Positive a)->CMFLinearPositive{return {a.mantissa,a.exponent};};
        CMFLinearNativeMap result={put(t),put(integral.source),put(integral.error),put(integral.roundoff),
            put(integral.frequency),put(integral.omitted),depth.error,built,combinations,
            integral.panels,integral.accepted,integral.split,integral.pruned,integral.max_depth};
        *out=result;return 0;
    }catch(...){return 6;}
}
