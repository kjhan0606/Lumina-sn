#include "cmf_exact_multigpu.h"
#include "cmf_cell_geometry.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures;
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"EMISSIVITY_FAIL line=%d %s\n",__LINE__,#x);++failures;}}while(0)
struct MomentData{
    std::vector<double> v;CMFFormalMoments m;
    MomentData(int cells,int nb):v(3*cells+3*nb){m={v.data(),v.data()+cells,v.data()+2*cells,
        v.data()+3*cells,v.data()+3*cells+nb,v.data()+3*cells+2*nb};}
};
static void run(int kind){
    const int ns=3,nb=192,cells=ns*nb;
    const double time=kind==5?1e30:1e6,dlog=(kind==3||kind==4)?1e-5:.02,a=1/(2.99792458e10*time);
    const double value=kind==0?0:(kind==1?-3*a:(kind==2?-2e-14:(kind==3?1e-13:-2e-14)));
    double ri[ns]={1e13,1.3e13,1.6e13},ro[ns]={1.3e13,1.6e13,2e13};
    std::vector<double> nu(nb),ct(cells,value),ce(cells,1e-16),eta(cells,1e-20),input(cells,1e-6);
    for(int b=0;b<nb;b++)nu[b]=1e14*std::exp((b+.5)*dlog);
    MomentData cpu(cells,nb),gpu(cells,nb),alternate(cells,nb);
    CMFMultiGPUEpochSchedule schedule={128,64,32};CMFMultiGPUReport mr;
    CHECK(cmf_exact_formal_moments_emissivity(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
        ct.data(),ce.data(),eta.data(),input.data(),&cpu.m)==CMF_EXACT_OK);
    CHECK(cmf_exact_multigpu_formal_moments_emissivity_epoch(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
        ct.data(),ce.data(),eta.data(),input.data(),&gpu.m,&schedule,1,&mr)==CMF_MGPU_OK);
    double max_delta=0;
    for(size_t i=0;i<cpu.v.size();i++){
        double norm=i<(size_t)3*cells?cpu.m.J[i%cells]:cpu.m.outer_J[(i-3*cells)%nb];
        if(norm>0)max_delta=std::max(max_delta,std::fabs(gpu.v[i]-cpu.v[i])/norm);
        else CHECK(gpu.v[i]==cpu.v[i]);
    }
    CHECK(max_delta<3e-11);
    /* 수송을 셸별로 걷지 않는 균질 광선 해석해. 각도 격자는 생산과 동일하며
     * 따라서 이는 각도 이산화 오차를 제외한 전파/방출 식 검증이다. */
    int nr=cmf_cell_ray_count(ns);std::vector<double> rays(nr);cmf_cell_fill_rays(ns,ri,ro,rays.data());
    double expected[3]={0,0,0},effective=value+3*a,emission=eta[0]+ce[0]*input[0];
    for(int r=0;r<nr;r++){
        double p=rays[r],length=p<ri[0]?std::sqrt(ro[2]*ro[2]-p*p)-std::sqrt(ri[0]*ri[0]-p*p):
                                                    2*std::sqrt(ro[2]*ro[2]-p*p);
        double I=effective==0?emission*length:emission/effective*(-std::expm1(-effective*length));
        for(int m=0;m<3;m++)expected[m]+=.5*I*cmf_cell_angle_moment_weight(r,nr,rays.data(),ro[2],ri[0],m);
    }
    double analytic_delta=0;
    const double *outer[3]={gpu.m.outer_J,gpu.m.outer_H,gpu.m.outer_K};
    for(int m=0;m<3;m++)for(int b=0;b<nb;b++)
        analytic_delta=std::max(analytic_delta,std::fabs(outer[m][b]/expected[m]-1));
    CHECK(analytic_delta<3e-11);
    auto jc=input,jg=input,jref=input;std::vector<double> ec(cells),eg(cells);
    CMFExactReport cr;CMFMultiGPUReport gr;
    CHECK(cmf_exact_emissivity_solve(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,ct.data(),ce.data(),eta.data(),
        jc.data(),ec.data(),8,128,1e-11,CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)==CMF_EXACT_OK);
    CHECK(cmf_exact_multigpu_emissivity_solve_epoch(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
        ct.data(),ce.data(),eta.data(),jg.data(),eg.data(),8,&schedule,1,128,1e-11,&gr)==CMF_MGPU_OK);
    CHECK(cmf_exact_emissivity_solve(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,ct.data(),ce.data(),eta.data(),
        jref.data(),nullptr,0,256,1e-14,CMF_EXACT_MODE_DIRECT_REFERENCE,nullptr)==CMF_EXACT_OK);
    CHECK(cr.componentwise_error_envelope_verified && gr.componentwise_error_envelope_verified);
    if(kind==0){
        CMFExactReport cr;CMFMultiGPUReport gr;
        auto rejected=input;std::vector<double> bound(cells,123);
        auto original=rejected,original_bound=bound;
        CHECK(cmf_exact_emissivity_solve(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
            ct.data(),ce.data(),eta.data(),rejected.data(),bound.data(),8,2,1e-14,
            CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)==CMF_EXACT_NOT_CONVERGED);
        CHECK(cr.status==CMF_EXACT_NOT_CONVERGED);
        CHECK(std::memcmp(rejected.data(),original.data(),cells*sizeof(double))==0);
        CHECK(std::memcmp(bound.data(),original_bound.data(),cells*sizeof(double))==0);
        CHECK(cmf_exact_multigpu_emissivity_solve_epoch(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
            ct.data(),ce.data(),eta.data(),rejected.data(),bound.data(),8,&schedule,1,2,1e-14,
            &gr)==CMF_MGPU_NOT_CONVERGED);
        CHECK(std::memcmp(rejected.data(),original.data(),cells*sizeof(double))==0);
        CHECK(std::memcmp(bound.data(),original_bound.data(),cells*sizeof(double))==0);
        auto bad_eta=eta;bad_eta[0]=-eta[0];
        CHECK(cmf_exact_emissivity_solve(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
            ct.data(),ce.data(),bad_eta.data(),rejected.data(),bound.data(),8,128,1e-11,
            CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)!=CMF_EXACT_OK);
        CHECK(cmf_exact_multigpu_emissivity_solve_epoch(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
            ct.data(),ce.data(),bad_eta.data(),rejected.data(),bound.data(),8,&schedule,1,128,1e-11,
            &gr)!=CMF_MGPU_OK);
        CHECK(std::memcmp(rejected.data(),original.data(),cells*sizeof(double))==0);
        CHECK(std::memcmp(bound.data(),original_bound.data(),cells*sizeof(double))==0);
        CHECK(cmf_exact_emissivity_solve(0,nb,dlog,nu.data(),ri,ro,time,1e4,0,
            ct.data(),ce.data(),eta.data(),rejected.data(),bound.data(),8,128,1e-11,
            CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)==CMF_EXACT_INVALID_INPUT);
        CHECK(cr.status==CMF_EXACT_INVALID_INPUT);
        if(!failures)std::printf("EMISSIVITY_REJECTION_PASS outputs_preserved=1\n");
    }
    for(int i=0;i<cells;i++){
        CHECK(std::fabs(jg[i]-jc[i])<=eg[i]+ec[i]);
        CHECK(std::fabs(jg[i]-jref[i])<=eg[i]+64*2.2204460492503131e-16*std::fabs(jref[i]));
    }
    schedule={64,7,1};
    CHECK(cmf_exact_multigpu_formal_moments_emissivity_epoch(ns,nb,dlog,nu.data(),ri,ro,time,1e4,0,
        ct.data(),ce.data(),eta.data(),input.data(),&alternate.m,&schedule,1,nullptr)==CMF_MGPU_OK);
    for(size_t i=0;i<gpu.v.size();i++)CHECK(std::fabs(alternate.v[i]-gpu.v[i])<3e-11*gpu.m.outer_J[0]);
    std::printf("EMISSIVITY case=%d chi=%.17g effective=%.17g analytic_relative=%.9g cpu_gpu=%.9g "
        "iterations=%d/%d norm=%.9g drift=%.9g\n",kind,value,effective,analytic_delta,max_delta,
        cr.iterations_used,gr.iterations_used,gr.scattering_operator_norm_upper,mr.max_characteristic_drift_bins);
}
int main(){for(int k=0;k<6;k++)run(k);std::printf("CMF_EMISSIVITY_%s failures=%d\n",failures?"FAIL":"PASS",failures);return failures?1:0;}
