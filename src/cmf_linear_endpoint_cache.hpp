#ifndef CMF_LINEAR_ENDPOINT_CACHE_HPP
#define CMF_LINEAR_ENDPOINT_CACHE_HPP
#include "cmf_ordered_maps.hpp"
#include <cstddef>
#ifdef __CUDACC__
#define LUMINA_ENDPOINT __host__ __device__ inline
#else
#define LUMINA_ENDPOINT inline
#endif
namespace lumina_endpoint_cache {
/* query 끝뿐 아니라 트리 build가 block 단위로 읽는 마지막 vertex까지
 * 채워야 한다. 부분 query의 η32 잎도 마지막 block 전체를 만든다. */
LUMINA_ENDPOINT int required_prefix(int nf,int bins,int chi_block,int source_block,
    double offset,double span,int count){
    if(nf<2||nf>2000000||bins<0||bins>=nf||chi_block<1||chi_block>256||source_block<1||source_block>256||
       count<1||count>2000000||!lumina_ordered::finite(offset)||offset<0||
       !lumina_ordered::finite(span)||span<0)return -1;
    if(offset>=nf-1)return 0;
    int last=(int)::ceil(::fmin(double(nf-1),offset+(count-1)+span));
    if(bins){
        int a=((bins+chi_block-1)/chi_block)*chi_block;
        int b=((bins+source_block-1)/source_block)*source_block;
        int end=a>b?a:b;if(end>=nf)end=nf-1;
        if(last<end)last=end;
    }
    return last+1;
}
// 각 공간 조각/J 세대의 원 Reader 결과만 저장한다. 보간된 총원천을
// 반경에서 다시 보간하거나 주파수/공간 근사를 추가하지 않는다.
struct Reader {
    const double *values;int nf,filled;
    LUMINA_ENDPOINT bool at(int point,int column,double &chi,double &eta) const {
        if(!values||(point!=0&&point!=2)||nf<2||filled<0||filled>nf||column<0||column>=nf||
           (column>=filled&&column!=nf-1))return false;
        int c=point/2;double x=values[(size_t)c*nf+column],e=values[(size_t)(c+2)*nf+column];
        if(!lumina_ordered::finite(x)||!lumina_ordered::finite(e)||e<0)return false;
        chi=x;eta=e;return true;
    }
};
template<class Original> LUMINA_ENDPOINT bool fill(const Original &r,int nf,int column,double *values){
    if(!values||nf<2||column<0||column>=nf)return false;
    double chi[2],eta[2];
    for(int c=0;c<2;c++)if(!r.at(2*c,column,chi[c],eta[c]))return false;
    for(int c=0;c<2;c++){values[(size_t)c*nf+column]=chi[c];values[(size_t)(c+2)*nf+column]=eta[c];}
    return true;
}
}
#undef LUMINA_ENDPOINT
#endif
