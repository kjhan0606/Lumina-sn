#include "cmf_ordered_maps.hpp"
#include <vector>
#include <cstring>

/* 작은 host 직접시험 전용. 실제 전격자나 GPU를 실행하지 않는다. */
extern "C" int lumina_ordered_probe(int n,const double *values,const double *log_t,
    int nq,const int *begin,const int *end,const double *incident,double *output)
{
    using namespace lumina_ordered;
    if(n<1||n>65536||nq<1||nq>65536||!values||!log_t||!begin||!end||!incident||!output)return 1;
    for(int q=0;q<nq;q++)if(begin[q]<0||end[q]<begin[q]||end[q]>n||!lumina_ordered::finite(incident[q])||incident[q]<0)return 1;
    try {
        int leaves=1;while(leaves<n)leaves*=2;
        std::vector<Map> tree(static_cast<std::size_t>(2)*leaves,Map::identity());
        for(int i=0;i<n;i++)if(!Map::from_values(values+6*i,log_t[i],tree[leaves+i]))return 2;
        for(int i=leaves-1;i>0;i--)if(!compose(tree[2*i],tree[2*i+1],tree[i]))return 3;
        std::vector<double> result(static_cast<std::size_t>(7)*nq);
        for(int q=0;q<nq;q++){
            Map value;
            if(!query(tree.data(),leaves,begin[q],end[q],value))return 3;
            double intensity;if(!apply(value,incident[q],intensity))return 4;
            double *v=result.data()+7*q;
            v[0]=value.d;v[1]=value.t.value();v[2]=value.aa.value();v[3]=value.pa.value();
            v[4]=value.ab.value();v[5]=value.mb.value();v[6]=intensity;
            for(int k=0;k<7;k++)if(!lumina_ordered::finite(v[k]))return 4;
        }
        std::memcpy(output,result.data(),result.size()*sizeof(double));return 0;
    }catch(...){return 5;}
}
