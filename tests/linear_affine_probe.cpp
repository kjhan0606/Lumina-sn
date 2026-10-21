#include "cmf_linear_affine.hpp"
#include <vector>
#include <cstring>
extern "C" int lumina_linear_affine_probe(int n,int width,const CMFLinearNativeMap *maps,const double *initial,
    double relative,double *output,double *errors,double *ratios,double *scores)
{
    using namespace lumina_linear_affine;
    using lumina_ordered::finite;
    if(n<1||n>32||width<1||width>128||!maps||!initial||!output||!errors||!ratios||!scores||
       !finite(relative)||relative<=0||relative>.01)return 1;
    try{
        std::vector<Local> local((size_t)n*width);std::vector<double> o(width),e(width),r(width),s((size_t)n*width);
        for(int f=0;f<width;f++){
            Positive value,numerical=Positive::zero(),a,b;
            if(!Positive::from_double(initial[f],value))return 2;
            for(int row=0;row<n;row++){
                Local &x=local[(size_t)row*width+f];
                if(!leaf(maps+(size_t)row*3*width,width,f,value,x)||!product(x.response,numerical,a)||
                   !sum(a,x.numerical,numerical))return 2;
                x.numerical=numerical;value=x.value;
            }
            Positive tail=Positive::one(),geometry=Positive::zero(),tol;
            if(!scale(value,relative,tol))return 2;
            for(int row=n-1;row>=0;row--){
                Local &x=local[(size_t)row*width+f];
                if(!product(x.geometry,tail,a)||!sum(geometry,a,geometry)||!product(tail,x.response,tail))return 2;
                s[(size_t)row*width+f]=ratio(a,tol);
            }
            if(!sum(geometry,numerical,a)||!scale(value,64.*(n+1)*DBL_EPSILON,b)||!sum(a,b,a))return 2;
            o[f]=value.value();e[f]=a.value();r[f]=ratio(a,tol);
        }
        std::memcpy(output,o.data(),width*sizeof(double));std::memcpy(errors,e.data(),width*sizeof(double));
        std::memcpy(ratios,r.data(),width*sizeof(double));std::memcpy(scores,s.data(),s.size()*sizeof(double));return 0;
    }catch(...){return 3;}
}
