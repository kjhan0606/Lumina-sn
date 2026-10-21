#include "cmf_positive_cubic.hpp"
#include <vector>
#include <cstring>
extern "C" int lumina_positive_cubic_probe(int n,const double *length,const double *source,const double *phase,
    const double *logs,double relative,double *mantissas,int64_t *exponents,int *degrees)
{
    if(n<1||n>512||!length||!source||!phase||!logs||!mantissas||!exponents||!degrees)return 1;
    std::vector<double> m((size_t)n*2);std::vector<int64_t> e((size_t)n*2);std::vector<int> d(n);
    for(int i=0;i<n;i++){
        lumina_cubic::Integral value;
        if(!lumina_cubic::integrate(length[i],source+3*i,phase+4*i,logs[i],relative,value))return 2;
        m[2*i]=value.value.mantissa;m[2*i+1]=value.error.mantissa;
        e[2*i]=value.value.exponent;e[2*i+1]=value.error.exponent;d[i]=value.degree;
    }
    std::memcpy(mantissas,m.data(),m.size()*sizeof(double));std::memcpy(exponents,e.data(),e.size()*sizeof(int64_t));
    std::memcpy(degrees,d.data(),d.size()*sizeof(int));return 0;
}
