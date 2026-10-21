/* GPU 수치 실행 없이 B 코어의 device reader/정적 수학 인스턴스를 확인한다. */
#include "cmf_linear_moments.hpp"
#include "cmf_positive_cubic.hpp"
struct DeviceReader {
    const double *values;int nf;
    __device__ bool at(int point,int column,double &chi,double &eta) const {
        if((point!=0&&point!=2)||column<0||column>=nf)return false;
        int row=point==0?0:1;chi=values[(size_t)row*nf+column];eta=values[(size_t)(row+2)*nf+column];
        return lumina_ordered::finite(chi)&&lumina_ordered::finite(eta)&&eta>=0;
    }
};
__global__ void lumina_linear_core_compile(const double *values,int nf,const lumina_linear::Node *nodes,
    double *output,int *status)
{
    if(blockIdx.x||threadIdx.x)return;
    DeviceReader r={values,nf};lumina_linear::Node n;lumina_linear::Integral depth;
    lumina_linear::Tree tree={nodes,1,32,nf};
    if(!lumina_linear::query(r,tree,{0,.25},{1,.75},n)||!lumina_linear::path_integral(n,0,1.,0.,1.,depth)){
        *status=1;return;
    }
    double source[3]={1.,2.,1.},phase[4]={0.,-.1,-.2,-.25};lumina_cubic::Integral integral;
    if(!lumina_cubic::integrate(1.,source,phase,-depth.value,1e-10,integral)){*status=2;return;}
    output[0]=depth.value;output[1]=integral.value.value();*status=0;
}
