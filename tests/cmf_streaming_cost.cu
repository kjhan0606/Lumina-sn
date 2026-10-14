/* 실제50셸 기하학/10km/s/PPD12의 비용 시험. 재료는 인공 배열이며 물리 결과가 아니다. */
#include "cmf_exact_multigpu.h"
#include "cmf_cell_geometry.h"
#include "lumina_frequency_grid.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>

int main(int argc,char **argv) {
    if(argc!=2)return 2;
    FILE *fp=fopen(argv[1],"r");if(!fp)return 2;
    char row[512];if(!fgets(row,sizeof row,fp))return 2;
    std::vector<double> ri,ro;
    while(fgets(row,sizeof row,fp)) {
        int s;double a,b,va,vb;
        if(sscanf(row,"%d,%lf,%lf,%lf,%lf",&s,&a,&b,&va,&vb)!=5 ||
           s!=(int)ri.size())return 2;
        ri.push_back(a);ro.push_back(b);
    }
    fclose(fp);
    const int ns=(int)ri.size(), nb=131072;
    if(ns!=50 || !cmf_cell_grid_valid(ns,ri.data(),ro.data()))return 2;
    const double t=1683072.0, c=2.99792458e10, dlog=(1e6/c)/12;
    const int full_nf=(int)ceil(log(NLTE_NU_MAX/NLTE_NU_MIN)/dlog+1);
    const int nr=cmf_cell_ray_count(ns);
    std::vector<double> p(nr);
    cmf_cell_fill_rays(ns,ri.data(),ro.data(),p.data());
    size_t slots=0,maxwindow=0;
    for(int r=0;r<nr;++r)for(int s=ns-1;s>=0;--s) {
        if(ro[s]<=p[r])break;
        ++slots;double a,b;
        cmf_cell_half_lengths(ri[s],ro[s],p[r],&a,&b);
        double lens[2]={a,b};
        for(double length:lens) {
            double beta=length/(t*c*dlog);
            int q=(int)floor(beta), qt=(beta-q)<0.5?q:q+1;
            size_t window=qt>=2?(size_t)qt-1:0;
            if(window>maxwindow)maxwindow=window;
        }
    }
    auto device_bytes=[&](size_t nf) {
        size_t doubles=3U*nr*nf+5U*ns*nf+nf+2U*slots+(size_t)nr*ns+nr+ns;
        size_t ints=3U*nr+1+slots+6;
        return 8U*doubles+4U*ints+32U*nr*(nf+maxwindow);
    };
    const size_t full_new=device_bytes(full_nf);
    const size_t full_old=full_new+16U*(slots-nr)*(size_t)full_nf;
    cudaDeviceProp prop;if(cudaGetDeviceProperties(&prop,0)!=cudaSuccess)return 2;
    printf("STREAMING_RESOURCE shells=%d rays=%d ray_cells=%zu ppd=12 vdop_cms=1000000 "
        "full_fine_bins=%d max_window=%zu old_device_bytes=%zu new_device_bytes=%zu "
        "new_GiB=%.6f GPU=%s total_device_GiB=%.3f\n",ns,nr,slots,full_nf,maxwindow,
        full_old,full_new,full_new/1073741824.0,prop.name,prop.totalGlobalMem/1073741824.0);
    fflush(stdout);
    size_t cells=(size_t)ns*nb;
    std::vector<double> nu(nb),chi(cells),es(cells),src(cells),J(cells);
    std::vector<double> lower(cells),near(cells),upper(cells);
    for(int b=0;b<nb;++b)nu[b]=NLTE_NU_MIN*exp((b+0.5)*dlog);
    for(size_t k=0;k<cells;++k) {
        chi[k]=1e-14*(1+0.02*(k%23));es[k]=1e-15;
        src[k]=1e-5*(1+0.01*(k%7));J[k]=2e-5;
    }
    const CMFMultiGPUEpochSchedule schedule={128,64,32};
    CMFMultiGPUReport report;
    auto start=std::chrono::steady_clock::now();
    CMFMultiGPUStatus status=cmf_exact_multigpu_apply_positive_bounds_epoch(
        ns,nb,dlog,nu.data(),ri.data(),ro.data(),t,10020,1,
        chi.data(),es.data(),src.data(),J.data(),lower.data(),near.data(),upper.data(),
        &schedule,1,&report);
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    printf("STREAMING_COST status=%s test_bins=%d directed_applications=3 "
        "seconds=%.6f allocated_bytes=%zu formula_bytes=%zu synthetic_material=1\n",
        cmf_multigpu_status_name(status),nb,sec,report.max_device_allocated_bytes,device_bytes(nb));
    if(status!=CMF_MGPU_OK || report.max_device_allocated_bytes!=device_bytes(nb))return 1;
    for(size_t k=0;k<cells;++k)
        if(!(0<=lower[k] && lower[k]<=near[k] && near[k]<=upper[k] && std::isfinite(upper[k])))return 1;
    puts("STREAMING_COST_PASS");return 0;
}
