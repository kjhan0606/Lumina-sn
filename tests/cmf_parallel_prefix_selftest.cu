/* 긴 구간의64그룹 누적을 독립 CPU와 직렬GPU에 비교한다. */
#include "cmf_exact_multigpu.h"
#include "cmf_exact_sliding.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"PREFIX_FAIL line=%d: %s\n",__LINE__,#x);return 1;}}while(0)
int main() {
    const int ns=3,nb=4096;const size_t cells=(size_t)ns*nb;
    double ri[3]={3e14,4e14,5e14},ro[3]={4e14,5e14,6e14};
    std::vector<double> nu(nb),chi(cells),es(cells),src(cells),J(cells);
    std::vector<double> cl(cells),cn(cells),cu(cells),sl(cells),sn(cells),su(cells);
    std::vector<double> lo(cells),ne(cells),up(cells),reference(3*cells);
    const double dlog=1e-5;
    for(int b=0;b<nb;++b)nu[b]=2e14*exp(b*dlog);
    const CMFMultiGPUEpochSchedule schedules[]={{32,1,1},{128,64,32},{256,100,2048}};
    for(int test=0;test<3;++test) {
        for(size_t k=0;k<cells;++k) {
            double shape=1+0.3*sin(k*0.01)+0.15*cos(k*0.03);
            chi[k]=(test==0?1e-13:test==1?1e-9:0)*shape;
            es[k]=0.05*chi[k];src[k]=1e-5*shape;J[k]=2e-5*(1+0.2*sin(k*0.07));
        }
        REQUIRE(cmf_exact_characteristic_apply_positive_bounds(ns,nb,dlog,nu.data(),ri,ro,
            1e6,10000,1,chi.data(),es.data(),src.data(),J.data(),cl.data(),cn.data(),cu.data())==CMF_EXACT_OK);
        CMFMultiGPUReport report;
        REQUIRE(cmf_exact_multigpu_apply_positive_bounds(ns,nb,dlog,nu.data(),ri,ro,
            1e6,10000,1,chi.data(),es.data(),src.data(),J.data(),sl.data(),sn.data(),su.data(),1,&report)==CMF_MGPU_OK);
        double worst=0,width=0;
        for(int schedule=0;schedule<3;++schedule) {
            REQUIRE(cmf_exact_multigpu_apply_positive_bounds_epoch(ns,nb,dlog,nu.data(),ri,ro,
                1e6,10000,1,chi.data(),es.data(),src.data(),J.data(),lo.data(),ne.data(),up.data(),
                &schedules[schedule],1,&report)==CMF_MGPU_OK);
            REQUIRE(report.max_positive_window_bins>=64);
            for(size_t k=0;k<cells;++k) {
                REQUIRE(0<=lo[k] && lo[k]<=ne[k] && ne[k]<=up[k]);
                REQUIRE(lo[k]<=cu[k] && up[k]>=cl[k]);
                REQUIRE(lo[k]<=su[k] && up[k]>=sl[k]);
                double scale=std::max(cn[k],1e-300);
                worst=std::max(worst,fabs(ne[k]-cn[k])/scale);
                worst=std::max(worst,fabs(ne[k]-sn[k])/scale);
                width=std::max(width,(up[k]-lo[k])/scale);
            }
            if(schedule==0) {
                memcpy(reference.data(),lo.data(),cells*sizeof(double));
                memcpy(reference.data()+cells,ne.data(),cells*sizeof(double));
                memcpy(reference.data()+2*cells,up.data(),cells*sizeof(double));
            } else {
                REQUIRE(memcmp(reference.data(),lo.data(),cells*sizeof(double))==0);
                REQUIRE(memcmp(reference.data()+cells,ne.data(),cells*sizeof(double))==0);
                REQUIRE(memcmp(reference.data()+2*cells,up.data(),cells*sizeof(double))==0);
            }
        }
        printf("PREFIX case=%d max_window=%zu CPU_and_serial_GPU_relative=%.17g "
               "bound_relative_width=%.17g schedule_bitwise=1\n",test,
               report.max_positive_window_bins,worst,width);
        REQUIRE(worst<1e-10);
        if(test==0) {
            std::vector<double> cj=J,gj=J,ce(cells),ge(cells);
            CMFExactReport cr;
            REQUIRE(cmf_exact_characteristic_solve_with_envelope(ns,nb,dlog,nu.data(),ri,ro,
                1e6,10000,1,chi.data(),es.data(),src.data(),cj.data(),ce.data(),12,80,1e-12,
                CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)==CMF_EXACT_OK);
            REQUIRE(cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(ns,nb,dlog,
                nu.data(),ri,ro,1e6,10000,1,chi.data(),es.data(),src.data(),gj.data(),ge.data(),
                12,CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS,&schedules[1],1,80,1e-12,&report)==CMF_MGPU_OK);
            double ratio=0;
            for(size_t k=0;k<cells;++k) {
                double diff=fabs(gj[k]-cj[k]), bound=ce[k]+ge[k];
                REQUIRE(diff<=bound);
                if(bound>0)ratio=std::max(ratio,diff/bound);
            }
            printf("PREFIX_FIXED_POINT component_envelope_ratio=%.17g\n",ratio);
        }
    }
    puts("PARALLEL_PREFIX_SELFTEST_PASS");return 0;
}
