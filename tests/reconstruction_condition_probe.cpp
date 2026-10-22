#include "cmf_gpu_reconstruction.hpp"
extern "C" int reconstruction_condition_probe(int n,const double *input,double *out,double *controls,
    double *theta,int *flags){
    if(n!=8||!input||!out||!controls||!theta||!flags)return 1;
    const auto bank=lumina_reconstruction::build_bank();double low[8],t=1.,value;bool limited;
    for(int k=0;k<8;k++){
        double v=0.;
        for(int j=0;j<8;j++)v+=bank.v[3][k][j]*(input[j]-input[3]);
        controls[k]=input[3]+v;low[k]=input[3]+(double)k/7.*(input[4]-input[3]);
    }
    controls[0]=input[3];controls[7]=input[4];
    for(int k=0;k<8;k++)if(controls[k]<0){double r=low[k]/(low[k]-controls[k]);if(r<t)t=r;}
    int fallback=0;
    if(t<1){
        double adjusted=::nextafter(t,0.);
        for(int k=0;k<8;k++)if(adjusted*controls[k]+(1.-adjusted)*low[k]<0)fallback=1;
    }
    if(!lumina_reconstruction::one(n,.5,input,bank,3,value,limited))return 2;
    *out=value;*theta=t;flags[0]=limited;flags[1]=fallback;return 0;
}
