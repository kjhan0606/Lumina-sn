#include "cmf_monotone_reconstruction.hpp"
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
