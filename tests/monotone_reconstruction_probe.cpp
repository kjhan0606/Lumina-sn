#include "cmf_monotone_reconstruction.hpp"
#include "cmf_gpu_reconstruction.hpp"
#include <vector>
#include <cstring>
extern "C" int monotone_reconstruction_probe(int n,double shift,const double *input,double *output){
    if(n<4||n>16384||!input||!output)return 1;
    std::vector<double> values(n);
    for(int i=0;i<n;i++)if(!lumina_monotone_reconstruction::one(n,shift,input,i,values[i]))return 2;
    std::memcpy(output,values.data(),n*sizeof(double));return 0;
}
extern "C" int monotone_reconstruction_scaled_probe(int n,double shift,const double *input,int i,double *m,int64_t *e){
    if(!m||!e)return 1;
    lumina_ordered::Positive result;
    if(!lumina_monotone_reconstruction::one_scaled(n,shift,input,i,result))return 2;
    *m=result.mantissa;*e=result.exponent;return 0;
}
/* 실제 native 격자의 고정된 일부 열만 반 bin 이동해 비교한다. 수송은 없다. */
extern "C" int reconstruction_native_sample_probe(int n,int count,const int *columns,const double *input,
    double *original,double *candidate,int *flags){
    if(n<8||n>2000000||count<1||count>16384||!columns||!input||!original||!candidate||!flags)return 1;
    for(int k=0;k<count;k++)if(columns[k]<0||columns[k]>=n)return 1;
    const auto bank=lumina_reconstruction::build_bank();
    for(int k=0;k<count;k++){
        bool limited=false;double a=0.,b=0.;
        bool old_ok=lumina_reconstruction::one(n,.5,input,bank,columns[k],a,limited);
        bool new_ok=lumina_monotone_reconstruction::one(n,.5,input,columns[k],b);
        flags[k]=(old_ok?0:1)|(new_ok?0:2)|(limited?4:0);
        original[k]=a;candidate[k]=b;
    }
    return 0;
}
