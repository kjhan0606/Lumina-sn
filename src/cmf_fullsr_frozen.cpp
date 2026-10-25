#include "cmf_fullsr_frozen.h"
#include "cmf_linear_provider.hpp"
#include "cmf_linear_affine.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <new>
#include <vector>
#include "cmf_positive_moments.h"

namespace {
using lumina_ordered::Positive;
using lumina_ordered::product;
using lumina_ordered::sum;
struct Affine {Positive t,s;};
struct Node {Affine value,aggregate;};
double clock_now(){timespec t{};return clock_gettime(CLOCK_MONOTONIC,&t)?NAN:t.tv_sec+1e-9*t.tv_nsec;}
bool valid(Positive p){return std::isfinite(p.mantissa)&&p.mantissa>=0&&
    (p.mantissa==0||(p.mantissa>=.5&&p.mantissa<1.))&&
    p.exponent>-(INT64_C(1)<<60)&&p.exponent<(INT64_C(1)<<60);}
Positive decode(CMFLinearPositive p){return {p.mantissa,p.exponent};}
bool scale(Positive a,double b,Positive &out){Positive p;return Positive::from_double(b,p)&&product(a,p,out);}
int order(Positive a,Positive b){
    if(!a.mantissa)return b.mantissa?-1:0;
    if(!b.mantissa)return 1;
    if(a.exponent!=b.exponent)return a.exponent>b.exponent?1:-1;
    return a.mantissa==b.mantissa?0:(a.mantissa>b.mantissa?1:-1);
}
bool subtract(Positive a,Positive b,Positive &out){
    return order(a,b)>=0&&lumina_linear_affine::difference(a,b,out);
}
Affine identity(){return {Positive::one(),Positive::zero()};}
bool compose(Affine a,Affine b,Affine &out,CMFFrozenStats &s){
    Positive t,x,y;
    if(!product(b.t,a.t,t)||!product(b.t,a.s,x)||!sum(x,b.s,y))return false;
    out={t,y};s.compositions++;return true;
}
bool apply(Affine a,Positive incoming,Positive &out){
    Positive p;return valid(incoming)&&product(a.t,incoming,p)&&sum(p,a.s,out);
}
/* q0/q1은 양 끝 effective chi*length다. 내부 gain 봉우리에서는 0점을
 * 나누고, 순 gain은 반사+scaled T로 보존해 모멘트의 double overflow를 피한다. */
bool linear_source(double q0,double q1,double length,double eb,double er,Affine &out,CMFFrozenStats &stats){
    if(!std::isfinite(q0)||!std::isfinite(q1))return false;
    if(q0>0&&q1<0){
        double denominator=q0-q1;if(!std::isfinite(denominator))return false;
        double f=q0/denominator;if(!(f>0&&f<1))return false;
        double middle=(1.-f)*eb+f*er;Affine a,b;
        return linear_source(q0*f,0.,length*f,eb,middle,a,stats)&&
            linear_source(0.,q1*(1.-f),length*(1.-f),middle,er,b,stats)&&compose(a,b,out,stats);
    }
    double tau=.5*q0+.5*q1;
    Affine result;if(!Positive::from_log(-tau,result.t))return false;
    result.s=Positive::zero();stats.primitive_maps++;
    if(eb!=0||er!=0){
        bool reflected=tau<0;double a=reflected?-q1:q0,b=reflected?-q0:q1;
        double w[3];CMFHistoryStats work={};
        if(::basis(a,b,std::exp(-(.5*a+.5*b)),w,&work))return false;
        for(double x:w)if(!std::isfinite(x)||x<=0)return false;
        double blue=reflected?w[1]+w[2]:w[0]+w[1],red=reflected?w[0]+w[1]:w[1]+w[2];
        Positive p,q,r;
        if(!Positive::from_double(blue,p)||!scale(p,length,q)||!scale(q,eb,p)||
           !Positive::from_double(red,q)||!scale(q,length,r)||!scale(r,er,q)||!sum(p,q,result.s))return false;
        if(reflected){if(!product(result.s,result.t,p))return false;result.s=p;}
    }
    out=result;return true;
}
/* 각 native bin의 주파수 방향은 청색 -> 적색이다. */
bool bin_map(double cb,double cr,double eb,double er,double length,double delta,int n,
             Affine &out,CMFFrozenStats &stats,bool linear){
    Affine total=identity();
    for(int k=0;k<n;k++){
        double a=(double)k/n,b=(double)(k+1)/n;
        double c0=(1.-a)*cb+a*cr,c1=(1.-b)*cb+b*cr;
        double e0=(1.-a)*eb+a*er,e1=(1.-b)*eb+b*er;
        double dl=length/n,d=delta/n;
        Affine part;
        if(linear){
            if(!linear_source(std::fma(c0,dl,3.*d),std::fma(c1,dl,3.*d),dl,e0,e1,part,stats))return false;
        }else{
            CMFLinearPositive zero[3]={},source;
            if(cmf_fullsr_sc_step(dl,d,1.,1,c0,c1,e0,e1,zero,&source)||
               !Positive::from_log(-std::fma(.5*c0+.5*c1,dl,3.*d),part.t))return false;
            part.s=decode(source);stats.primitive_maps++;
        }
        Affine next;if(!compose(total,part,next,stats))return false;
        total=next;
    }
    out=total;return true;
}

class Window {
    std::vector<Node> front,back;
    CMFFrozenStats &stats;
public:
    explicit Window(size_t maximum,CMFFrozenStats &s):stats(s){front.reserve(maximum);back.reserve(maximum);}
    size_t size()const{return front.size()+back.size();}
    bool push(Affine value){
        Affine combined=value;
        if(!back.empty()&&!compose(back.back().aggregate,value,combined,stats))return false;
        back.push_back({value,combined});return true;
    }
    bool pop(){
        if(front.empty())while(!back.empty()){
            Affine value=back.back().value,combined=value;back.pop_back();
            if(!front.empty()&&!compose(value,front.back().aggregate,combined,stats))return false;
            front.push_back({value,combined});
        }
        if(front.empty())return false;
        front.pop_back();return true;
    }
    bool aggregate(Affine &out){
        if(front.empty()){out=back.empty()?identity():back.back().aggregate;return true;}
        if(back.empty()){out=front.back().aggregate;return true;}
        return compose(front.back().aggregate,back.back().aggregate,out,stats);
    }
};

bool relative_u(const CMFLinearPositive *values,int i,int base,double h,Positive &out){
    Positive a=decode(values[i]),factor;
    return valid(a)&&Positive::from_log(-3.*(i-base)*h,factor)&&product(a,factor,out);
}
/* PCHIP 내부 기울기의 h/3 배수다. 양 끝 기울기는 secant를 쓴다.
 * harmonic <= 2*min/3이므로 Bernstein 제어점은 이웃 값 범위 안이다. */
bool harmonic(Positive a,Positive b,Positive &out){
    if(!a.mantissa||!b.mantissa){out=Positive::zero();return true;}
    if(order(a,b)>0)std::swap(a,b);
    int64_t difference=a.exponent-b.exponent;
    double ratio=difference < -1074?0.:std::ldexp(a.mantissa/b.mantissa,(int)difference);
    return scale(a,(2./3.)/(1.+ratio),out);
}
bool interpolate(const CMFLinearPositive *values,int nf,int k,double f,double h,int reconstruction,Positive &out){
    Positive a,b;
    if(!relative_u(values,k,k,h,a))return false;
    if(!f){out=a;return true;}
    if(k+1>=nf||!relative_u(values,k+1,k,h,b))return false;
    Positive controls[4]={a,a,b,b};int direction=order(b,a);
    if(reconstruction&&direction){
        Positive middle,left,right,previous,next,alpha,beta;
        if(!lumina_linear_affine::difference(a,b,middle))return false;
        if(k==0){if(!scale(middle,1./3.,alpha))return false;}
        else{
            if(!relative_u(values,k-1,k,h,previous)||!lumina_linear_affine::difference(a,previous,left))return false;
            alpha=Positive::zero();
            if(order(a,previous)==direction&&!harmonic(left,middle,alpha))return false;
        }
        if(k+2==nf){if(!scale(middle,1./3.,beta))return false;}
        else{
            if(!relative_u(values,k+2,k,h,next)||!lumina_linear_affine::difference(b,next,right))return false;
            beta=Positive::zero();
            if(order(next,b)==direction&&!harmonic(middle,right,beta))return false;
        }
        if(direction>0){if(!sum(a,alpha,controls[1])||!subtract(b,beta,controls[2]))return false;}
        else if(!subtract(a,alpha,controls[1])||!sum(b,beta,controls[2]))return false;
    }
    Positive result=Positive::zero(),part,total,factor;
    if(!reconstruction){
        if(!scale(a,1.-f,part)||!scale(b,f,total)||!sum(part,total,result))return false;
    }else{
        double g=1.-f,w[4]={g*g*g,3.*g*g*f,3.*g*f*f,f*f*f};
        for(int i=0;i<4;i++){
            if(!scale(controls[i],w[i],part)||!sum(result,part,total))return false;
            result=total;
        }
    }
    return Positive::from_log(3.*f*h,factor)&&product(result,factor,out);
}
}

static int cell(const CMFNodalDenseView *view,const CMFNodalLocation *location,
    double length,double delta,double h,int subdivisions,int reconstruction,int last,
    const CMFLinearPositive *incident,CMFLinearPositive *output,double seconds,CMFFrozenStats *stats,bool linear){
    if(!stats)return 1;
    *stats={};stats->failed_column=-1;
    if(!view||!location||!incident||!output||!std::isfinite(length)||length<=0||
       !std::isfinite(delta)||delta<0||!std::isfinite(h)||h<=0||
       subdivisions<1||subdivisions>16||(reconstruction!=0&&reconstruction!=1)||
       !std::isfinite(seconds)||seconds<=0||seconds>160.||last<0||last>=view->nf){stats->status=1;return 1;}
    CMFNodalLocation pair[2]={*location,*location};
    if(!lumina_native_linear::valid(*view,pair)){stats->status=1;return 1;}
    double span=delta/h;
    if(!std::isfinite(span)){stats->status=2;return 2;}
    int nf=view->nf,maximum=nf-1;
    int width=span>=maximum?maximum:(int)std::floor(span);
    double fraction=span>=maximum?0.:span-width;
    uint64_t count=nf-last;
    stats->heap_bytes=count*(sizeof(Positive)+2*sizeof(double))+2*(uint64_t)(width+1)*sizeof(Node);
    if(stats->heap_bytes>UINT64_C(536870912)){stats->status=3;return 3;}
    double started=clock_now();
    try{
        std::vector<double> chi(count),eta(count);
        std::vector<Positive> candidate(count);
        Window window(width+1,*stats);
        lumina_native_linear::Reader reader={*view,{*location,*location},0};
        for(int j=maximum;j>=last;j--){
            if((maximum-j)%2048==0&&clock_now()-started>=seconds){stats->status=4;break;}
            if(!reader.at(0,j,chi[j-last],eta[j-last])){stats->status=2;stats->failed_column=j;break;}
        }
        for(int j=maximum;j>=last&&!stats->status;j--){
            stats->failed_column=j;
            if((maximum-j)%2048==0&&clock_now()-started>=seconds){stats->status=4;break;}
            Affine total=identity(),prefix=identity();Positive incoming,result;
            if(delta==0){
                incoming=decode(incident[j]);
                if(!bin_map(chi[j-last],chi[j-last],eta[j-last],eta[j-last],length,0.,1,total,*stats,linear)||
                   !apply(total,incoming,result)){stats->status=2;break;}
            }else{
                if(width&&j<maximum){
                    Affine full;
                    if(!bin_map(chi[j+1-last],chi[j-last],eta[j+1-last],eta[j-last],
                        length/span,h,subdivisions,full,*stats,linear)||!window.push(full)||
                        (window.size()>(size_t)width&&!window.pop())){stats->status=2;break;}
                }
                int k=j+width;
                if(k>=maximum){
                    incoming=decode(incident[maximum]);
                    double portion=(span-(maximum-j))/span;
                    if(portion>0&&!bin_map(chi[maximum-last],chi[maximum-last],eta[maximum-last],eta[maximum-last],
                        length*portion,delta*portion,1,prefix,*stats,linear)){stats->status=2;break;}
                }else{
                    if(!interpolate(incident,nf,k,fraction,h,reconstruction,incoming)){stats->status=2;break;}
                    if(fraction){
                        double cr=chi[k-last],cb=chi[k+1-last],er=eta[k-last],eb=eta[k+1-last];
                        double portion=fraction/span;
                        if(!bin_map((1.-fraction)*cr+fraction*cb,cr,(1.-fraction)*er+fraction*eb,er,
                            length*portion,delta*portion,subdivisions,prefix,*stats,linear)){stats->status=2;break;}
                    }
                }
                Affine whole;
                if(!window.aggregate(whole)||!compose(prefix,whole,total,*stats)||!apply(total,incoming,result)){
                    stats->status=2;break;
                }
            }
            candidate[j-last]=result;stats->completed_outputs++;
        }
        if(!stats->status){
            for(int j=last;j<nf;j++)output[j]={candidate[j-last].mantissa,candidate[j-last].exponent};
            stats->failed_column=-1;
        }
    }catch(const std::bad_alloc &){stats->status=3;}
    stats->elapsed_seconds=clock_now()-started;return stats->status;
}

extern "C" int cmf_fullsr_frozen_cell(const CMFNodalDenseView *v,const CMFNodalLocation *p,
    double length,double delta,double h,int n,int reconstruction,int last,
    const CMFLinearPositive *input,CMFLinearPositive *output,double seconds,CMFFrozenStats *s){
    return cell(v,p,length,delta,h,n,reconstruction,last,input,output,seconds,s,false);
}
extern "C" int cmf_fullsr_frozen_linear_cell(const CMFNodalDenseView *v,const CMFNodalLocation *p,
    double length,double delta,double h,int n,int reconstruction,int last,
    const CMFLinearPositive *input,CMFLinearPositive *output,double seconds,CMFFrozenStats *s){
    return cell(v,p,length,delta,h,n,reconstruction,last,input,output,seconds,s,true);
}

extern "C" int cmf_fullsr_frozen_begin(int nf,double h,double delta,int reconstruction,int last,
    const CMFLinearPositive *input,CMFLinearPositive *output,double seconds,CMFFrozenStats *stats){
    if(!stats)return 1;
    *stats={};stats->failed_column=-1;
    if(nf<2||nf>2000000||!input||!output||!std::isfinite(h)||h<=0||
       !std::isfinite(delta)||delta<0||(reconstruction!=0&&reconstruction!=1)||
       last<0||last>=nf||!std::isfinite(seconds)||seconds<=0||seconds>160.){stats->status=1;return 1;}
    double span=delta/h;
    if(!std::isfinite(span)||span>1e9){stats->status=2;return 2;}
    int whole=(int)std::floor(span);double fraction=span-whole;
    Positive frame;if(!Positive::from_log(-3.*delta,frame)){stats->status=2;return 2;}
    double started=clock_now();stats->heap_bytes=(uint64_t)(nf-last)*sizeof(Positive);
    try{
        std::vector<Positive> candidate(nf-last);
        for(int j=nf-1;j>=last;j--){
            stats->failed_column=j;
            if((nf-1-j)%2048==0&&clock_now()-started>=seconds){stats->status=4;break;}
            Positive incoming,result;int k=j+whole;
            if(k>=nf-1)incoming=decode(input[nf-1]);
            else if(!interpolate(input,nf,k,fraction,h,reconstruction,incoming)){stats->status=2;break;}
            if(!valid(incoming)||!product(incoming,frame,result)){stats->status=2;break;}
            candidate[j-last]=result;stats->completed_outputs++;
        }
        if(!stats->status){
            for(int j=last;j<nf;j++)output[j]={candidate[j-last].mantissa,candidate[j-last].exponent};
            stats->failed_column=-1;
        }
    }catch(const std::bad_alloc &){stats->status=3;}
    stats->elapsed_seconds=clock_now()-started;return stats->status;
}

extern "C" int cmf_fullsr_frozen_fixed_cell(const CMFNodalDenseView *view,const CMFNodalLocation *location,
    double length,double da,double db,double h,int subdivisions,int last,
    const CMFLinearPositive *input,CMFLinearPositive *output,double seconds,CMFFrozenStats *stats){
    if(!stats)return 1;
    *stats={};stats->failed_column=-1;
    if(!view||!location||!input||!output||!std::isfinite(length)||length<=0||
       !std::isfinite(da)||!std::isfinite(db)||db<0||da<db||!std::isfinite(h)||h<=0||
       subdivisions<1||subdivisions>16||last<0||last>=view->nf||
       !std::isfinite(seconds)||seconds<=0||seconds>160.){stats->status=1;return 1;}
    CMFNodalLocation pair[2]={*location,*location};
    if(!lumina_native_linear::valid(*view,pair)){stats->status=1;return 1;}
    double delta=da-db,span=delta/h,red_offset=db/h;
    if(!std::isfinite(span)||!std::isfinite(red_offset)||span>1e9||red_offset>1e9){stats->status=2;return 2;}
    int rb=(int)std::floor(red_offset),shift=(int)std::floor(span);
    double rf=red_offset-rb,temporary=rf+(span-shift);
    int carry=temporary>=1.?1:0,bb=rb+shift+carry;
    double bf=temporary-carry;
    if(delta>0&&bb==rb&&bf==rf){stats->status=2;return 2;}
    int nf=view->nf,maximum=nf-1,low_offset=rb+(rf>0?1:0);
    int width=std::min(maximum,std::max(0,bb-low_offset));
    uint64_t count=nf-last;
    stats->heap_bytes=count*(sizeof(Positive)+2*sizeof(double))+2*(uint64_t)(width+1)*sizeof(Node);
    if(stats->heap_bytes>UINT64_C(536870912)){stats->status=3;return 3;}
    Positive transfer_frame,source_frame;
    if(!Positive::from_log(3.*delta,transfer_frame)||!Positive::from_log(-3.*db,source_frame)){
        stats->status=2;return 2;
    }
    double started=clock_now();
    try{
        std::vector<double> chi(count),eta(count);
        std::vector<Positive> candidate(count);
        Window window(width+1,*stats);
        lumina_native_linear::Reader reader={*view,{*location,*location},0};
        for(int j=maximum;j>=last;j--){
            if((maximum-j)%2048==0&&clock_now()-started>=seconds){stats->status=4;break;}
            if(!reader.at(0,j,chi[j-last],eta[j-last])){stats->status=2;stats->failed_column=j;break;}
        }
        auto sample=[&](int k,double f,double &c,double &e){
            if(k>=maximum){c=chi[maximum-last];e=eta[maximum-last];}
            else{c=(1.-f)*chi[k-last]+f*chi[k+1-last];e=(1.-f)*eta[k-last]+f*eta[k+1-last];}
        };
        auto part=[&](int blue,double blue_fraction,int red,double red_fraction,
                      double portion,Affine &out){
            double cb,cr,eb,er;sample(blue,blue_fraction,cb,eb);sample(red,red_fraction,cr,er);
            return portion>0&&std::isfinite(portion)&&bin_map(cb,cr,eb,er,length*portion,
                delta*portion,subdivisions,out,*stats,true);
        };
        for(int j=maximum;j>=last&&!stats->status;j--){
            stats->failed_column=j;
            if((maximum-j)%2048==0&&clock_now()-started>=seconds){stats->status=4;break;}
            int red=j+rb,blue=j+bb,low=j+low_offset;
            Affine total=identity();
            if(width&&low<maximum){
                Affine full;
                if(!part(low+1,0.,low,0.,1./span,full)||!window.push(full)||
                   (window.size()>(size_t)width&&!window.pop())){stats->status=2;break;}
            }
            if(delta==0){
                double c,e;sample(red,rf,c,e);
                if(!linear_source(c*length,c*length,length,e,e,total,*stats)){stats->status=2;break;}
            }else if(red>=maximum){
                if(!part(maximum,0.,maximum,0.,1.,total)){stats->status=2;break;}
            }else if(red==blue){
                if(!part(blue,bf,red,rf,1.,total)){stats->status=2;break;}
            }else{
                Affine prefix=identity(),whole,suffix=identity(),intermediate;
                if(blue>=maximum){
                    double portion=((blue-maximum)+bf)/span;
                    if(portion>0&&!part(maximum,0.,maximum,0.,portion,prefix)){stats->status=2;break;}
                }else if(bf>0&&!part(blue,bf,blue,0.,bf/span,prefix)){stats->status=2;break;}
                if(rf>0&&!part(red+1,0.,red,rf,(1.-rf)/span,suffix)){stats->status=2;break;}
                if(!window.aggregate(whole)||!compose(prefix,whole,intermediate,*stats)||
                   !compose(intermediate,suffix,total,*stats)){stats->status=2;break;}
            }
            Affine fixed;Positive result;
            if(!product(total.t,transfer_frame,fixed.t)||!product(total.s,source_frame,fixed.s)||
               !apply(fixed,decode(input[j]),result)){stats->status=2;break;}
            candidate[j-last]=result;stats->completed_outputs++;
        }
        if(!stats->status){
            for(int j=last;j<nf;j++)output[j]={candidate[j-last].mantissa,candidate[j-last].exponent};
            stats->failed_column=-1;
        }
    }catch(const std::bad_alloc &){stats->status=3;}
    stats->elapsed_seconds=clock_now()-started;return stats->status;
}

extern "C" int cmf_fullsr_linear_chain(int nstep,int nray,const double *length,const double *delta,
    const double *chi,const double *eta,const CMFLinearPositive *initial,CMFLinearPositive *output,
    CMFFrozenAffine *maps,double seconds,CMFFrozenStats *stats,int *failed_step){
    if(!stats)return 1;
    *stats={};stats->failed_column=-1;if(failed_step)*failed_step=-1;
    if(nstep<1||nstep>65536||nray<1||nray>512||(uint64_t)nstep*nray>2000000||
       !length||!delta||!chi||!eta||!initial||!output||
       !std::isfinite(seconds)||seconds<=0||seconds>160.){stats->status=1;return 1;}
    for(int k=0;k<nstep;k++)if(!std::isfinite(length[k])||length[k]<=0||
        !std::isfinite(delta[k])||delta[k]<0){stats->status=1;if(failed_step)*failed_step=k;return 1;}
    for(size_t i=0;i<(size_t)(nstep+1)*nray;i++)if(!std::isfinite(chi[i])||!std::isfinite(eta[i])||eta[i]<0){
        stats->status=1;if(failed_step)*failed_step=(int)(i/nray);return 1;
    }
    double started=clock_now();
    stats->heap_bytes=(uint64_t)nray*sizeof(Positive)+(maps?(uint64_t)nstep*nray*sizeof(Affine):0);
    try{
        std::vector<Positive> current(nray);
        std::vector<Affine> saved(maps?(size_t)nstep*nray:0);
        for(int j=0;j<nray;j++){
            current[j]=decode(initial[j]);if(!valid(current[j])){stats->status=1;return 1;}
        }
        for(int k=0;k<nstep&&!stats->status;k++){
            if(clock_now()-started>=seconds){stats->status=4;if(failed_step)*failed_step=k;break;}
            for(int j=0;j<nray;j++){
                size_t a=(size_t)k*nray+j,b=a+nray;Affine map;Positive next;
                if(!linear_source(std::fma(chi[a],length[k],3.*delta[k]),
                    std::fma(chi[b],length[k],3.*delta[k]),length[k],eta[a],eta[b],map,*stats)||
                   !apply(map,current[j],next)){
                    stats->status=2;stats->failed_column=j;if(failed_step)*failed_step=k;break;
                }
                current[j]=next;if(maps)saved[a]=map;
            }
        }
        if(!stats->status){
            for(int j=0;j<nray;j++)output[j]={current[j].mantissa,current[j].exponent};
            if(maps)for(size_t i=0;i<saved.size();i++)maps[i]={{saved[i].t.mantissa,saved[i].t.exponent},
                {saved[i].s.mantissa,saved[i].s.exponent}};
            stats->completed_outputs=nray;
        }
    }catch(const std::bad_alloc &){stats->status=3;}
    stats->elapsed_seconds=clock_now()-started;return stats->status;
}
