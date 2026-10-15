/* 실제100셸 기하/격자, 합성 계수의 단일 형식해 일정 비교. SN 수렴 시험 아님. */
#include "cmf_exact_multigpu.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <ctime>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"SCHEDULE_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static double seconds(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char **argv){
    if(argc!=2)return 2;
    const int ns=100,nb=117606;const size_t cells=(size_t)ns*nb,faces=(size_t)(ns+1)*nb;
    const double time=172800,lo=58412785919616.055,hi=40363009529215944.;
    const double dlog=std::log(hi/lo)/nb;
    std::vector<double> ri(ns),ro(ns),nu(nb),chi(cells),es(cells),eta(cells),input(cells);
    std::vector<int> seen(ns,0);char line[2048];FILE *fp=std::fopen(argv[1],"r");CHECK(fp);
    while(std::fgets(line,sizeof line,fp)){
        int s;double r0,r1;
        if(std::sscanf(line,"%d,%lf,%lf",&s,&r0,&r1)!=3)continue;
        CHECK(s>=0&&s<ns&&r1>r0&&r0>=0);
        if(seen[s])CHECK(ri[s]==r0&&ro[s]==r1);
        else{ri[s]=r0;ro[s]=r1;seen[s]=1;}
    }
    CHECK(!std::fclose(fp));
    for(int s=0;s<ns;s++)CHECK(seen[s]&&(!s||ri[s]==ro[s-1]));
    for(int b=0;b<nb;b++)nu[b]=lo*std::exp((b+.5)*dlog);
    for(int s=0;s<ns;s++)for(int b=0;b<nb;b++){
        size_t i=(size_t)s*nb+b;
        chi[i]=(1+.1*std::sin(b*.001))*1e-11*std::exp(-.06*s);
        es[i]=.8*chi[i];eta[i]=.2*chi[i]*(1+.1*std::cos(b*.0003))*1e-7;
        input[i]=(1+.1*std::sin(b*.0002))*1e-7;
    }
    const size_t count=3*cells+3*nb+faces;
    std::vector<double> baseline(count),out(count);
    const int batches[3]={64,4096,64};
    for(int k=0;k<3;k++){
        CMFFormalMoments moments={out.data(),out.data()+cells,out.data()+2*cells,
            out.data()+3*cells,out.data()+3*cells+nb,out.data()+3*cells+2*nb,out.data()+3*cells+3*nb};
        CMFMultiGPUEpochSchedule schedule={128,batches[k],32};CMFMultiGPUReport report;
        double started=seconds();
        CMFMultiGPUStatus status=cmf_exact_multigpu_formal_moments_emissivity_epoch(
            ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,10000,0,
            chi.data(),es.data(),eta.data(),input.data(),&moments,&schedule,1,&report);
        CHECK(status==CMF_MGPU_OK);
        for(double v:out)CHECK(std::isfinite(v));
        if(k==0)baseline=out;
        else CHECK(!std::memcmp(out.data(),baseline.data(),count*sizeof(double)));
        std::printf("SCHEDULE_FORMAL_PASS batch=%d seconds=%.9g sweep_seconds=%.9g device_bytes=%zu values=%zu bitwise=1 actual_geometry=1 synthetic_material=1\n",
                    batches[k],seconds()-started,report.device_sweep_seconds,report.max_device_allocated_bytes,count);
        std::fflush(stdout);
    }
    return 0;
}
