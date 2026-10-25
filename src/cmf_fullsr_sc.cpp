#include "cmf_fullsr_sc.h"
#include "cmf_ordered_maps.hpp"
#include "cmf_linear_affine.hpp"
#include "cmf_linear_provider.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <new>
#include <vector>
#include <omp.h>

namespace {
using lumina_ordered::Positive;
using lumina_ordered::finite;
using lumina_ordered::product;
using lumina_ordered::sum;

double now(){timespec t{};return clock_gettime(CLOCK_MONOTONIC,&t)?NAN:t.tv_sec+1e-9*t.tv_nsec;}
bool valid(Positive a){return finite(a.mantissa)&&a.mantissa>=0&&
    (a.mantissa==0||(a.mantissa>=.5&&a.mantissa<1.))&&
    a.exponent>-(INT64_C(1)<<60)&&a.exponent<(INT64_C(1)<<60);}
bool scale(Positive a,double b,Positive &out){Positive p;return Positive::from_double(b,p)&&product(a,p,out);}

/* 양의 적분 가중치를 scaled로 구성한다. 작은 eta와 큰 gain을 따로 보존한다. */
bool weights(double tau,Positive &t,Positive &w0,Positive &w1){
    if(!finite(tau)||!Positive::from_log(-tau,t))return false;
    if(std::abs(tau)<.1){
        double term=1.,a=.5,b=.5;
        for(int n=1;n<=14;n++){
            term*=-tau/n;a+=term/(n+2);b+=term/((n+1.)*(n+2.));
        }
        return Positive::from_double(a,w0)&&Positive::from_double(b,w1);
    }
    double u=1./std::abs(tau);Positive inverse,denominator,a,b,c;
    if(!Positive::from_double(u,inverse)||!product(inverse,inverse,denominator))return false;
    if(tau>0){
        double e=std::exp(-tau),x=-std::expm1(-tau)-tau*e;
        return Positive::from_double(x,a)&&product(a,denominator,w0)&&
            Positive::from_double(1.-u+e*u,b)&&product(b,inverse,w1);
    }
    double gain=-tau;
    if(gain<1.){
        double e=std::exp(gain),x=gain*e-std::expm1(gain),y=std::expm1(gain)-gain;
        return Positive::from_double(x,a)&&product(a,denominator,w0)&&
            Positive::from_double(y,b)&&product(b,denominator,w1);
    }
    if(!scale(t,gain-1.,a)||!sum(a,Positive::one(),b)||!product(b,denominator,w0)||
       !Positive::from_double(gain+1.,a))return false;
    if(t.exponent<a.exponent||(t.exponent==a.exponent&&t.mantissa<a.mantissa))return false;
    return lumina_linear_affine::difference(t,a,c)&&product(c,denominator,w1);
}

bool step(double length,double drift,double h,bool top,double chi0,double chi1,
          double eta0,double eta1,Positive incident,Positive blue0,Positive blue1,Positive &out){
    if(!finite(length)||length<=0||!finite(drift)||drift<0||!finite(h)||h<=0||
       !finite(chi0)||!finite(chi1)||!finite(eta0)||eta0<0||!finite(eta1)||eta1<0||
       !valid(incident)||!valid(blue0)||!valid(blue1))return false;
    double shift=top?3.*drift:drift/h;
    double tau=std::fma(.5*chi0+.5*chi1,length,shift);
    Positive t,w0,w1,a,b,c,result;
    if(!finite(shift)||!weights(tau,t,w0,w1)||!product(incident,t,result))return false;
    if(!scale(w0,length,a)||!scale(a,eta0,b)||!sum(result,b,c))return false;
    result=c;
    if(!scale(w1,length,a)||!scale(a,eta1,b)||!sum(result,b,c))return false;
    result=c;
    if(!top&&shift>0){
        Positive spectral;
        if(!Positive::from_log(-3.*h,spectral)||!scale(spectral,shift,a))return false;
        if(!product(a,w0,b)||!product(b,blue0,c)||!sum(result,c,b))return false;
        result=b;
        if(!product(a,w1,b)||!product(b,blue1,c)||!sum(result,c,b))return false;
        result=b;
    }
    out=result;return valid(out);
}

bool sample(const CMFNodalDenseView &v,CMFNodalLocation location,int f,double fraction,double &chi,double &eta){
    lumina_native_linear::Reader r={v,{location,location},0};
    if(!r.at(0,f,chi,eta))return false;
    if(fraction){
        double cb,eb;if(f+1>=v.nf||!r.at(0,f+1,cb,eb))return false;
        chi=(1.-fraction)*chi+fraction*cb;eta=(1.-fraction)*eta+fraction*eb;
    }
    return finite(chi)&&finite(eta)&&eta>=0;
}
}

extern "C" int cmf_fullsr_sc_step(double length,double drift,double h,int top,
    double chi0,double chi1,double eta0,double eta1,const CMFLinearPositive *state,CMFLinearPositive *output){
    if(!state||!output||(top!=0&&top!=1))return 1;
    Positive result;
    if(!step(length,drift,h,top,chi0,chi1,eta0,eta1,
        {state[0].mantissa,state[0].exponent},{state[1].mantissa,state[1].exponent},
        {state[2].mantissa,state[2].exponent},result))return 2;
    *output={result.mantissa,result.exponent};return 0;
}

extern "C" int cmf_fullsr_sc_advance(const CMFNodalDenseView *view,
    int npath,int nvertex,const int *path_ptr,const int *kind,const CMFNodalLocation *location,
    const double *length,const double *drift,const double *core_mu,int nevent,
    const int *event_ptr,const int *event_node,const double *event_weights,
    const double *planck,const double *gradient,double h,int subdivisions,
    int first,int last,int tile_width,int threads,double seconds,
    CMFLinearPositive *raw_blue,double *moments,double *escape,CMFFullSRSCStats *stats){
    if(!stats)return 1;
    *stats={};stats->next_frequency=first;stats->failed_path=stats->failed_frequency=stats->failed_vertex=-1;
    if(!view||!path_ptr||!kind||!location||!length||!drift||!core_mu||!event_ptr||
       !event_node||!event_weights||!planck||!gradient||!raw_blue||(!moments&&!escape)||
       npath<1||npath>512||nvertex<npath||nvertex>4000000||nevent<0||
       first<last||last<0||first>=view->nf||tile_width<1||tile_width>4096||
       threads<1||threads>16||subdivisions<1||subdivisions>16||
       !std::isfinite(h)||h<=0||!std::isfinite(seconds)||seconds<=0||seconds>2700.||
       path_ptr[0]!=0||path_ptr[npath]!=nvertex||event_ptr[0]!=0||event_ptr[nvertex]!=nevent){
        stats->status=1;return 1;
    }
    for(int p=0;p<npath;p++)if(path_ptr[p]>=path_ptr[p+1]||path_ptr[p]<0||path_ptr[p+1]>nvertex||
        kind[path_ptr[p]]!=1){stats->status=1;return 1;}
    for(int k=0;k<nvertex;k++){
        CMFNodalLocation endpoints[2]={location[k],location[k]};
        if(!lumina_native_linear::valid(*view,endpoints)||kind[k]<0||kind[k]>2||
           !std::isfinite(core_mu[k])||std::abs(core_mu[k])>1||
           !std::isfinite(length[k])||!std::isfinite(drift[k])||drift[k]<0||
           (kind[k]==0?length[k]<=0:length[k]!=0||drift[k]!=0)||
           event_ptr[k]>event_ptr[k+1]||event_ptr[k]<0||event_ptr[k+1]>nevent){stats->status=1;return 1;}
    }
    for(int e=0;e<nevent;e++)for(int m=0;m<3;m++)if(event_node[e]<0||event_node[e]>=view->ns||
        !std::isfinite(event_weights[3*e+m])||(m!=1&&event_weights[3*e+m]<0)){stats->status=1;return 1;}
    static_assert(sizeof(Positive)==sizeof(CMFLinearPositive),"scaled state ABI");
    for(int k=0;k<nvertex;k++)if(!valid({raw_blue[k].mantissa,raw_blue[k].exponent})){stats->status=1;return 1;}
    uint64_t per_ray=(uint64_t)3*view->ns*tile_width;
    uint64_t heap=(uint64_t)nvertex*sizeof(Positive)+(moments?(uint64_t)npath*per_ray*8:0)+
        (escape?(uint64_t)npath*tile_width*8:0);
    if(heap>UINT64_C(4294967296)){stats->status=3;return 3;}
    stats->heap_bytes=heap;
    const double started=now();double reported=started;
    try{
        std::vector<Positive> blue(nvertex);
        for(int k=0;k<nvertex;k++)blue[k]={raw_blue[k].mantissa,raw_blue[k].exponent};
        std::vector<double> tile(moments?(size_t)npath*per_ray:0);
        std::vector<double> outgoing(escape?(size_t)npath*tile_width:0);
        for(int hi=first;hi>=last;){
            int lo=std::max(last,hi-tile_width+1);
            std::fill(tile.begin(),tile.end(),0.);
            std::atomic<int> error(0);
            #pragma omp parallel for schedule(static) num_threads(threads)
            for(int p=0;p<npath;p++){
                auto fail=[&](int code,int f,int k){
                    int zero=0;if(error.compare_exchange_strong(zero,code)){
                        stats->failed_path=p;stats->failed_frequency=f;stats->failed_vertex=k;
                    }
                };
                int begin=path_ptr[p],end=path_ptr[p+1];
                for(int f=hi;f>=lo&&!error.load();f--){
                    if((hi-f)%16==0&&now()-started>=seconds){fail(4,f,begin);break;}
                    int qtop=f==view->nf-1?0:subdivisions-1;
                    for(int q=qtop;q>=0&&!error.load();q--){
                        bool top=f==view->nf-1;double fraction=(double)q/subdivisions;
                        double bc=planck[f],bg=gradient[f];
                        if(fraction){bc=(1.-fraction)*bc+fraction*planck[f+1];bg=(1.-fraction)*bg+fraction*gradient[f+1];}
                        if(!std::isfinite(bc)||bc<0||!std::isfinite(bg)){fail(2,f,begin);break;}
                        Positive current=Positive::zero(),previous_blue=Positive::zero();
                        double previous_chi=0.,previous_eta=0.;
                        for(int k=begin;k<end;k++){
                            double chi,eta;Positive old=blue[k],next;
                            if(!sample(*view,location[k],f,fraction,chi,eta)){fail(2,f,k);break;}
                            if(kind[k]){
                                double value=kind[k]==1?0.:bc+core_mu[k]*bg;
                                if(!Positive::from_double(value,current)){fail(2,f,k);break;}
                            }else{
                                if(!step(length[k],drift[k],h/subdivisions,top,previous_chi,chi,
                                    previous_eta,eta,current,previous_blue,old,next)){fail(2,f,k);break;}
                                current=next;
                            }
                            blue[k]=current;previous_blue=old;previous_chi=chi;previous_eta=eta;
                            if(q==0){
                                if(moments)for(int e=event_ptr[k];e<event_ptr[k+1];e++)for(int m=0;m<3;m++){
                                    double weight=event_weights[3*e+m];Positive contribution;
                                    if(!scale(current,std::abs(weight),contribution)){fail(2,f,k);break;}
                                    double value=contribution.value();if(weight<0)value=-value;
                                    size_t at=(size_t)p*per_ray+((size_t)m*view->ns+event_node[e])*tile_width+hi-f;
                                    double total=tile[at]+value;
                                    if(!std::isfinite(value)||!std::isfinite(total)){fail(2,f,k);break;}
                                    tile[at]=total;
                                }
                                if(escape&&k==end-1){
                                    double value=current.value();if(!std::isfinite(value)){fail(2,f,k);break;}
                                    outgoing[(size_t)p*tile_width+hi-f]=value;
                                }
                            }
                            if(error.load())break;
                        }
                    }
                }
            }
            if(error.load()){
                stats->status=error.load();break;
            }
            if(moments)for(int m=0;m<3&&!stats->status;m++)for(int row=0;row<view->ns&&!stats->status;row++)
                for(int f=hi;f>=lo;f--){
                    double total=0.;
                    for(int p=0;p<npath;p++)total+=tile[(size_t)p*per_ray+((size_t)m*view->ns+row)*tile_width+hi-f];
                    if(!std::isfinite(total)){stats->status=2;stats->failed_frequency=f;break;}
                    tile[((size_t)m*view->ns+row)*tile_width+hi-f]=total;
                }
            if(stats->status)break;
            if(moments)for(int m=0;m<3;m++)for(int row=0;row<view->ns;row++)for(int f=hi;f>=lo;f--)
                moments[((size_t)m*view->ns+row)*view->nf+f]=tile[((size_t)m*view->ns+row)*tile_width+hi-f];
            if(escape)for(int p=0;p<npath;p++)for(int f=hi;f>=lo;f--)
                escape[(size_t)p*view->nf+f]=outgoing[(size_t)p*tile_width+hi-f];
            for(int k=0;k<nvertex;k++)raw_blue[k]={blue[k].mantissa,blue[k].exponent};
            int frequencies=hi-lo+1;
            stats->completed_frequencies+=frequencies;
            uint64_t inner=(uint64_t)frequencies*subdivisions;
            if(hi==view->nf-1)inner-=subdivisions-1;
            for(int k=0;k<nvertex;k++)if(!kind[k])stats->space_updates+=inner;
            stats->observation_updates+=(uint64_t)nevent*frequencies;
            stats->next_frequency=lo-1;hi=lo-1;
            if(now()-reported>=60.){
                std::fprintf(stderr,"SC 진행: 다음 주파수 %d, 이번 호출 완성 %llu, 경과 %.1f초\n",
                    hi,(unsigned long long)stats->completed_frequencies,now()-started);std::fflush(stderr);reported=now();
            }
        }
    }catch(const std::bad_alloc &){stats->status=3;}
    stats->elapsed_seconds=now()-started;return stats->status;
}
