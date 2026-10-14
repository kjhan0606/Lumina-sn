#include "cmf_exact_multigpu.h"
#include "cmf_time_field.h"
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
    Moments(size_t cells,int nb):data(4*cells+4*nb){
        m={data.data(),data.data()+cells,data.data()+2*cells,
           data.data()+3*cells,data.data()+3*cells+nb,data.data()+3*cells+2*nb,
           data.data()+3*cells+3*nb};
    }
};

static void run_case(int kind)
{
    const int ns=5,nb=256;const size_t cells=(size_t)ns*nb;
    const double dlog=kind==2?1e-4:0.02;
    const double time=kind==2?1e5:(kind==4?1683072:1e30);
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
        if(kind==4){
            /* 423030 첫 실패 셀의 실제 계수. 기하만 작은 합성 외피다. */
            ct[i]=-3.6516509489148456e-17;
            ce[i]=7.2222128090546361e-16;
            sf[i]=1.1096205597289692e-20/ct[i];
            input[i]=1e-6;
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
        if(i>=3*cells+3*nb){
            size_t face=i-3*cells-3*nb,s=std::min(face/nb,(size_t)ns-1);
            scale=cpu.m.J[s*nb+face%nb];
        }
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
            for(int s=0;s<=ns;s++){
                double r=s<ns?ri[s]:ro[ns-1];
                CHECK(std::fabs(gpu.m.face_H[(size_t)s*nb+b]/(.25*B*ri[0]*ri[0]/(r*r))-1)<3e-12);
            }
        }
    }
    if(kind==1)for(size_t i=0;i<cells;++i){
        CHECK(std::fabs(gpu.m.J[i]-2)<2e-12);CHECK(std::fabs(gpu.m.H[i])<2e-12);
        CHECK(std::fabs(gpu.m.K[i]-2.0/3)<2e-12);
    }
    if(kind==1)for(int s=1;s<ns;s++)for(int b=0;b<nb;b++)
        CHECK(std::fabs(gpu.m.face_H[(size_t)s*nb+b])<2e-12);
    for(int b=0;b<nb;b++)CHECK(std::fabs(gpu.m.face_H[cells+b]-gpu.m.outer_H[b])<=
        3e-12*gpu.m.outer_J[b]);
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
    if(kind==4){
        auto jc=input,jg=input,jref=input;
        std::vector<double> ec(cells),eg(cells);
        CMFExactReport cr;CMFMultiGPUReport gr;
        CHECK(cmf_exact_characteristic_solve_with_envelope(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
            10000,1,ct.data(),ce.data(),sf.data(),jc.data(),ec.data(),8,128,1e-11,
            CMF_EXACT_MODE_POSITIVE_SLIDING,&cr)==CMF_EXACT_OK);
        CHECK(cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(ns,nb,dlog,nu.data(),
            ri.data(),ro.data(),time,10000,1,ct.data(),ce.data(),sf.data(),jg.data(),eg.data(),8,
            CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS,&schedule,1,128,1e-11,&gr)==CMF_MGPU_OK);
        CHECK(cr.max_scattering_ratio>1 && gr.max_scattering_ratio>1);
        CHECK(cr.scattering_operator_norm_upper<1 && gr.scattering_operator_norm_upper<1);
        CHECK(cr.componentwise_error_envelope_verified && gr.componentwise_error_envelope_verified);
        CHECK(std::isfinite(cr.fixed_point_absolute_error_bound));
        CHECK(std::isfinite(gr.fixed_point_absolute_error_bound));
        CHECK(cmf_exact_characteristic_solve(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
            10000,1,ct.data(),ce.data(),sf.data(),jref.data(),256,1e-14,
            CMF_EXACT_MODE_DIRECT_REFERENCE,nullptr)==CMF_EXACT_OK);
        double max_error_fraction=0;
        for(size_t i=0;i<cells;++i){
            double error=std::fabs(jg[i]-jref[i]);
            CHECK(std::fabs(jc[i]-jg[i])<=ec[i]+eg[i]);
            CHECK(error<=eg[i]+32*2.2204460492503131e-16*std::fabs(jref[i]));
            if(eg[i]>0)max_error_fraction=std::max(max_error_fraction,error/eg[i]);
        }
        std::printf("SIGNED_EXTINCTION_PASS local_ratio=%.17g norm_cpu=%.17g norm_gpu=%.17g "
            "iterations=%d/%d direct_error_over_bound=%.9g floor=0 clamp=0\n",
            gr.max_scattering_ratio,cr.scattering_operator_norm_upper,gr.scattering_operator_norm_upper,
            cr.iterations_used,gr.iterations_used,max_error_fraction);
        auto saved=gpu.data;
        sf[0]=-sf[0]; /* 실제 음의 방출률은 허용하지 않는다. */
        CHECK(cmf_exact_multigpu_formal_moments_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
            10000,1,ct.data(),ce.data(),sf.data(),input.data(),&gpu.m,&schedule,1,nullptr)!=CMF_MGPU_OK);
        CHECK(std::memcmp(saved.data(),gpu.data.data(),saved.size()*sizeof(double))==0);
        sf[0]=-sf[0];ct[0]=NAN;
        CHECK(cmf_exact_multigpu_formal_moments_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
            10000,1,ct.data(),ce.data(),sf.data(),input.data(),&gpu.m,&schedule,1,nullptr)!=CMF_MGPU_OK);
        CHECK(std::memcmp(saved.data(),gpu.data.data(),saved.size()*sizeof(double))==0);
    }
    auto saved=gpu.data;input[cells-1]=NAN;
    CHECK(cmf_exact_multigpu_formal_moments_epoch(ns,nb,dlog,nu.data(),ri.data(),ro.data(),time,
        10000,1,ct.data(),ce.data(),sf.data(),input.data(),&gpu.m,&schedule,1,nullptr)!=CMF_MGPU_OK);
    CHECK(std::memcmp(saved.data(),gpu.data.data(),saved.size()*sizeof(double))==0);
    std::printf("FORMAL_GPU case=%d max_delta_over_J=%.17g memory_bytes=%zu drift_bins=%.9g legacy_J_bitwise=checked schedule_bitwise=checked face_H=checked\n",
        kind,maximum,report.max_device_allocated_bytes,report.max_characteristic_drift_bins);
}

struct TimeContext {
    int ns,nb,gpu;
    double dlog,epoch;
    const double *nu,*ri,*ro,*chi,*scat,*eta;
};
static int time_formal(void *arg,const double *J,CMFFormalMoments *m){
    TimeContext *c=(TimeContext*)arg;
    if(!c->gpu)return cmf_exact_formal_moments_emissivity(c->ns,c->nb,c->dlog,c->nu,
        c->ri,c->ro,c->epoch,10000,1,c->chi,c->scat,c->eta,J,m)!=CMF_EXACT_OK;
    CMFMultiGPUEpochSchedule schedule={128,64,32};
    return cmf_exact_multigpu_formal_moments_emissivity_epoch(c->ns,c->nb,c->dlog,c->nu,
        c->ri,c->ro,c->epoch,10000,1,c->chi,c->scat,c->eta,J,m,&schedule,1,nullptr)!=CMF_MGPU_OK;
}
static void time_case(){
    const int ns=3,nb=64,nc=ns*nb,nf=(ns+1)*nb;
    double epoch=1001000,dlog=std::log(300.)/nb;
    std::vector<double> nu(nb),edges(nb+1),radius(ns+1),ri(ns),ro(ns),chi(nc),scat(nc),eta(nc);
    std::vector<double> oldj(nc),oldh(nf),inner(nb),cpu(nc),gpu(nc),hc(nf),hg(nf),delta(nc);
    for(int b=0;b<=nb;b++)edges[b]=1e13*std::exp(b*dlog);
    for(int b=0;b<nb;b++)nu[b]=std::sqrt(edges[b]*edges[b+1]);
    for(int s=0;s<=ns;s++)radius[s]=(3e8+s*1e8)*epoch;
    for(int s=0;s<ns;s++){
        ri[s]=radius[s];ro[s]=radius[s+1];
        for(int b=0;b<nb;b++){
            double B=2*6.62607015e-27*nu[b]*nu[b]*nu[b]/(2.99792458e10*2.99792458e10)/
                     std::expm1(6.62607015e-27*nu[b]/(1.380649e-16*10000));
            int i=s*nb+b;chi[i]=3e-14;scat[i]=2e-14;eta[i]=1e-14*B;oldj[i]=.5*B;
        }
    }
    CMFTimeMomentInput p={};p.n_shells=ns;p.n_bins=nb;p.time_old=1e6;p.time_new=epoch;
    p.radius_edges=radius.data();p.frequency_edges=edges.data();p.chi_total=chi.data();
    p.chi_scattering=scat.data();p.eta_fixed=eta.data();p.old_J=oldj.data();
    p.old_H=oldh.data();p.inner_H=inner.data();
    TimeContext ctx={ns,nb,0,dlog,epoch,nu.data(),ri.data(),ro.data(),chi.data(),scat.data(),eta.data()};
    CMFTimeFieldReport cr,gr;
    CHECK(cmf_time_field_solve(&p,oldj.data(),time_formal,&ctx,128,1e-9,
        cpu.data(),hc.data(),delta.data(),&cr)==CMF_TIME_OK);
    ctx.gpu=1;
    CHECK(cmf_time_field_solve(&p,oldj.data(),time_formal,&ctx,128,1e-9,
        gpu.data(),hg.data(),delta.data(),&gr)==CMF_TIME_OK);
    double error=0;
    for(int i=0;i<nc;i++)error=std::max(error,std::fabs(cpu[i]-gpu[i])/cpu[i]);
    for(int i=0;i<nf;i++){
        double scale=cpu[std::min(i,nc-1)];error=std::max(error,std::fabs(hc[i]-hg[i])/scale);
    }
    CHECK(cr.converged&&gr.converged&&cr.iterations==gr.iterations&&error<3e-11);
    CHECK(cr.equation_residual<1e-10&&gr.equation_residual<1e-10);
    std::printf("TIME_FIELD_GPU max_delta_over_J=%.17g iterations=%d/%d equation_residual=%.17g/%.17g\n",
        error,cr.iterations,gr.iterations,cr.equation_residual,gr.equation_residual);
}

int main(void)
{
    time_case();
    for(int kind=0;kind<5;++kind)run_case(kind);
    std::printf("CMF_FORMAL_MOMENTS_GPU_SELFTEST %s failures=%d\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
