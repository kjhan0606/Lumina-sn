// 보존된 한 대역의 고정 총eta 수송. 전역 물질/산란 반복이나 생산 API가 아니다.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <vector>

extern "C" int lumina_fixed_blue_quadratic_leg(int,const double*,const double*,
    const double*,const double*,const double*,double,double,int,double*);

struct LuminaBandStats {
    int completed,failed_ray,kernel_status;
    uint64_t directional_cells;
    double elapsed_seconds;
};
static double band_clock(){timespec t{};return clock_gettime(CLOCK_MONOTONIC,&t)?NAN:t.tv_sec+1.e-9*t.tv_nsec;}

extern "C" int lumina_quadratic_band(int nf,int nr,int nc,int ne,
    const int *ptr,const int *obs,const double *lm,const double *lp,
    const double *bm,const double *bp,const double *wp,const double *wm,
    const double *chi,const double *eta,const double *h,const double *inner,const double *outer,
    const double *initial_m,const double *initial_p,double seconds,double *output,LuminaBandStats *stats){
    if(!stats)return 1;
    *stats={0,-1,0,0,0.};
    if(nf<1||nf>20000||nr<1||nr>512||nc<0||nc>nr||ne<2||ne>512||
       !ptr||!obs||!lm||!lp||!bm||!bp||!wp||!wm||!chi||!eta||!h||!inner||!outer||
       !initial_m||!initial_p||!output||!std::isfinite(seconds)||seconds<=0||seconds>160||ptr[0]!=0)return 1;
    for(int r=0;r<nr;r++){
        int n=ptr[r+1]-ptr[r];
        if(n<2||n>ne||obs[r]<0||obs[r]>=n||!std::isfinite(wp[r])||wp[r]<0||
           !std::isfinite(wm[r])||wm[r]<0)return 1;
    }
    if(ptr[nr]>nr*ne)return 1;
    double started=band_clock();
    auto finish=[&](int code){stats->elapsed_seconds=band_clock()-started;return code;};
    try{
        int count=ptr[nr];
        std::vector<double> old_m(initial_m,initial_m+count),old_p(initial_p,initial_p+count);
        std::vector<double> new_m(count),new_p(count),reverse_chi(ne),reverse_eta(ne);
        for(int f=0;f<nf;f++){
            if(!std::isfinite(h[f])||h[f]<=0)return finish(1);
            double intensity=0.;
            for(int r=0;r<nr;r++){
                stats->failed_ray=r;
                if(band_clock()-started>=seconds)return finish(4);
                int off=ptr[r],n=ptr[r+1]-off;
                const double *c=chi+f*ne,*e=eta+f*ne;
                int code=lumina_fixed_blue_quadratic_leg(n,lm+off,c,e,bm+off,old_m.data()+off,
                    1/h[f],outer[f*nr+r],1,new_m.data()+off);
                if(code){stats->kernel_status=code;return finish(8);}
                std::reverse_copy(c,c+n,reverse_chi.begin());
                std::reverse_copy(e,e+n,reverse_eta.begin());
                double incoming=r<nc?inner[f*nr+r]:new_m[off+n-1];
                code=lumina_fixed_blue_quadratic_leg(n,lp+off,reverse_chi.data(),reverse_eta.data(),
                    bp+off,old_p.data()+off,1/h[f],incoming,1,new_p.data()+off);
                if(code){stats->kernel_status=code;return finish(8);}
                intensity+=wm[r]*new_m[off+obs[r]]+wp[r]*new_p[off+n-1-obs[r]];
                stats->directional_cells+=2*(n-1);
            }
            if(!std::isfinite(intensity)||intensity<0)return finish(8);
            output[f]=intensity;
            old_m.swap(new_m);old_p.swap(new_p);
            stats->completed++;stats->failed_ray=-1;
        }
    }catch(...){return finish(3);}
    return finish(0);
}
