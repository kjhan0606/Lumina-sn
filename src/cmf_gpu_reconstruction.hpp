#ifndef CMF_GPU_RECONSTRUCTION_HPP
#define CMF_GPU_RECONSTRUCTION_HPP
#include "cmf_ordered_maps.hpp"
#ifdef __CUDACC__
#define LUMINA_RECON __host__ __device__ inline
#else
#define LUMINA_RECON inline
#endif
namespace lumina_reconstruction {
struct Bank {double v[7][8][8];};
inline int choose(int n,int k){int v=1;for(int i=1;i<=k;i++)v=v*(n+1-i)/i;return v;}
inline Bank build_bank()
{
    Bank bank;
    for(int offset=-6;offset<=0;offset++)for(int j=0;j<8;j++){
        double power[8]={1.,0.,0.,0.,0.,0.,0.,0.},denominator=1.;int degree=0;
        for(int k=0;k<8;k++)if(k!=j){
            double next[8]={0.};double root=offset+k;
            for(int m=0;m<=degree;m++){next[m]-=root*power[m];next[m+1]+=power[m];}
            for(int m=0;m<8;m++)power[m]=next[m];
            degree++;denominator*=j-k;
        }
        for(int m=0;m<8;m++)power[m]/=denominator;
        for(int k=0;k<8;k++){
            double value=0.;for(int m=0;m<=k;m++)value+=(double)choose(k,m)/choose(7,m)*power[m];
            bank.v[offset+6][k][j]=value;
        }
    }
    return bank;
}
LUMINA_RECON bool one(int nf,double shift,const double *input,const Bank &bank,int i,double &out,bool &limited)
{
    using lumina_ordered::finite;
    if(nf<8||i<0||i>=nf||!input||!finite(shift)||shift<0||!finite(input[i])||input[i]<0)return false;
    limited=false;
    if(shift>=nf-1){out=input[nf-1];return finite(out)&&out>=0;}
    int whole=(int)::floor(shift);double fraction=shift-whole;
    int left=i+whole;
    if(fraction==0||left>=nf-1){out=input[left<nf?left:nf-1];return finite(out)&&out>=0;}
    int first=left-3;if(first<0)first=0;if(first>nf-8)first=nf-8;
    double b[8],low[8],base=input[left],difference=input[left+1]-base,theta=1.;
    for(int k=0;k<8;k++){
        double value=0.;
        for(int j=0;j<8;j++){
            if(!finite(input[first+j])||input[first+j]<0)return false;
            value+=bank.v[first-left+6][k][j]*(input[first+j]-base);
        }
        b[k]=base+value;low[k]=base+(double)k/7*difference;
    }
    b[0]=input[left];b[7]=input[left+1];
    for(int k=0;k<8;k++){
        if(!finite(b[k]))return false;
        if(b[k]<0){double ratio=low[k]/(low[k]-b[k]);if(ratio<theta)theta=ratio;}
    }
    if(theta<1){
        int fallback=0;theta=::nextafter(theta,0.);limited=true;
        for(int k=0;k<8;k++){b[k]=theta*b[k]+(1-theta)*low[k];if(b[k]<0)fallback=1;}
        if(fallback)for(int k=0;k<8;k++)b[k]=low[k];
    }
    for(int degree=7;degree>0;degree--)for(int k=0;k<degree;k++)b[k]=(1-fraction)*b[k]+fraction*b[k+1];
    out=b[0];return finite(out)&&out>=0;
}
}
#undef LUMINA_RECON
#endif
