#include "cmf_exact_multigpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures;
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#x);++failures;}}while(0)

struct Moments {
    std::vector<double> data;
    CMFFormalMoments m;
    Moments(size_t cells,int nb):data(3*cells+3*nb){
        m={data.data(),data.data()+cells,data.data()+2*cells,
           data.data()+3*cells,data.data()+3*cells+nb,data.data()+3*cells+2*nb};
    }
};

static void run_case(int kind)
{
    const int ns=5,nb=256;const size_t cells=(size_t)ns*nb;
    const double dlog=kind==2?1e-4:0.02,time=kind==2?1e5:1e30;
    std::vector<double> ri(ns),ro(ns),nu(nb),ct(cells),ce(cells),sf(cells),input(cells);
    for(int s=0;s<ns;++s){ri[s]=(s+(kind==3?0:1))*1e13;ro[s]=ri[s]+1e13;}
    for(int b=0;b<nb;++b)nu[b]=1e14*std::exp((b+.5)*dlog);
    for(int s=0;s<ns;++s)for(int b=0;b<nb;++b){
        size_t i=(size_t)s*nb+b;
        if(kind==1){ct[i]=1e-7;sf[i]=2;}
        if(kind==2){
            ct[i]=(1+.4*std::sin(b*.13+s))*(s+1)*1e-13;
            ce[i]=.75*ct[i];sf[i]=1e-6*(1+.6*std::cos(b*.07+s));
            input[i]=1e-6*(1+.3*std::sin(b*.2));
        }
    }
    Moments cpu(cells,nb),gpu(cells,nb),other(cells,nb);
    CHECK(cmf_exact_formal_moments(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
        10000,1,ct.data(),ce.data(),sf.data(),input.data(),&cpu.m)==CMF_EXACT_OK);
    CMFMultiGPUEpochSchedule schedule={128,64,32};CMFMultiGPUReport report;
    CHECK(cmf_exact_multigpu_formal_moments_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
        10000,1,ct.data(),ce.data(),sf.data(),input.data(),&gpu.m,&schedule,1,&report)==CMF_MGPU_OK);
    double maximum=0;
    for(size_t i=0;i<gpu.data.size();++i){
        /* H can be exactly zero. Normalize all three moments by their J,
         * not by an artificial absolute floor or nearly cancelling H. */
        double scale=i<3*cells?cpu.m.J[i%cells]:cpu.m.outer_J[(i-3*cells)%nb];
        double delta=std::fabs(gpu.data[i]-cpu.data[i]);
        if(scale==0)CHECK(delta==0);else maximum=std::max(maximum,delta/scale);
    }
    CHECK(maximum<3e-11);
    if(kind==0){
        double radius=ro[ns-1],mu=std::sqrt(1-ri[0]*ri[0]/(radius*radius));
        for(int b=0;b<nb;++b){
            double B=2*6.62607015e-27*nu[b]*nu[b]*nu[b]/(2.99792458e10*2.99792458e10)/
                     std::expm1(6.62607015e-27*nu[b]/(1.380649e-16*10000));
            CHECK(std::fabs(gpu.m.outer_J[b]/(.5*B*(1-mu))-1)<3e-12);
            CHECK(std::fabs(gpu.m.outer_H[b]/(.25*B*(1-mu*mu))-1)<3e-12);
            CHECK(std::fabs(gpu.m.outer_K[b]/(B*(1-mu*mu*mu)/6)-1)<3e-12);
        }
    }
    if(kind==1)for(size_t i=0;i<cells;++i){
        CHECK(std::fabs(gpu.m.J[i]-2)<2e-12);CHECK(std::fabs(gpu.m.H[i])<2e-12);
        CHECK(std::fabs(gpu.m.K[i]-2.0/3)<2e-12);
    }
    if(kind==3)for(double v:gpu.data)CHECK(v==0);
    std::vector<double> lo(cells),nearest(cells),up(cells);
    CHECK(cmf_exact_multigpu_apply_positive_bounds_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
        10000,1,ct.data(),ce.data(),sf.data(),input.data(),lo.data(),nearest.data(),up.data(),
        &schedule,1,nullptr)==CMF_MGPU_OK);
    CHECK(std::memcmp(nearest.data(),gpu.m.J,cells*sizeof(double))==0);
    schedule={64,7,32};
    CHECK(cmf_exact_multigpu_formal_moments_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
        10000,1,ct.data(),ce.data(),sf.data(),input.data(),&other.m,&schedule,1,nullptr)==CMF_MGPU_OK);
    CHECK(std::memcmp(other.data.data(),gpu.data.data(),gpu.data.size()*sizeof(double))==0);
    auto saved=gpu.data;input[cells-1]=NAN;
    CHECK(cmf_exact_multigpu_formal_moments_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
        10000,1,ct.data(),ce.data(),sf.data(),input.data(),&gpu.m,&schedule,1,nullptr)!=CMF_MGPU_OK);
    CHECK(std::memcmp(saved.data(),gpu.data.data(),saved.size()*sizeof(double))==0);
    std::printf("FORMAL_GPU case=%d max_delta_over_J=%.17g memory_bytes=%zu drift_bins=%.9g legacy_J_bitwise=checked schedule_bitwise=checked\n",
        kind,maximum,report.max_device_allocated_bytes,report.max_characteristic_drift_bins);
}

int main(void)
{
    for(int kind=0;kind<4;++kind)run_case(kind);
    std::printf("CMF_FORMAL_MOMENTS_GPU_SELFTEST %s failures=%d\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
