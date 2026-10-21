/* 장치 경로 인스턴스의 컴파일 확인용이다. 이 파일만으로 GPU 실행을 하지 않는다. */
#include "cmf_source_measure.hpp"
struct SourceMeasureCompileReader {
    const double *values;int nf;
    __device__ bool at(int point,int column,double &chi,double &eta) const {
        if((point!=0&&point!=2)||column<0||column>=nf)return false;
        chi=values[(point/2)*nf+column];eta=values[(point/2+2)*nf+column];
        return lumina_ordered::finite(chi)&&lumina_ordered::finite(eta)&&eta>=0;
    }
};
__global__ void source_measure_compile_only(const double *values,int nf,const lumina_linear::Node *nodes,
    const lumina_source_measure::Node *source,int leaves,int block,int *status,lumina_characteristic::Result *output){
    lumina_linear::Tree tree={nodes,leaves,block,nf};
    lumina_source_measure::Tree measure={source,leaves,block,nf};
    SourceMeasureCompileReader reader={values,nf};
    lumina_characteristic::Profile<SourceMeasureCompileReader> p={reader,tree,0,.25,3.7,1.,.001};
    lumina_characteristic::Result result;
    int code=lumina_source_measure::integrate(p,measure,lumina_ordered::Positive::zero(),1e-8,100000,result);
    *status=code;if(!code)*output=result;
}
