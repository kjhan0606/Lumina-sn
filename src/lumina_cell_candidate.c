/* 실제 raw 계수와 확산 경계를 기존 셀 solver에 연결하는 후보 실행기.
 * 상대론 프레임 선택/입력 신원은 Python 실행기에서 검증한다. */
#include "lumina_cmfgen.h"
#include "cmf_exact_sliding.h"

static int cell_candidate(int ns,int nf,double dlog,const double *nu,
    const double *ri,const double *ro,double epoch,double boundary_temperature,
    double luminosity,const double *boundary_chi,const double *boundary_eta,
    const double *boundary_es,const double *bf_chi,const double *bf_eta,
    const double *ff_chi,const double *ff_eta,const double *bb_chi,
    const double *bb_eta,const double *es,double *j,double *error,
    int iteration_cap,double tolerance,uint64_t *counts,double *diagnostics,int relativistic)
{
    if(ns<1||nf<2||!nu||!ri||!ro||!boundary_chi||!boundary_eta||!boundary_es||
       !j||!error||!counts||!diagnostics||
       (size_t)ns>SIZE_MAX/(size_t)nf/sizeof(double)/2||
       (size_t)nf>SIZE_MAX/sizeof(double)/4)return CMF_EXACT_INVALID_INPUT;
    size_t n=(size_t)ns*nf;
    double *material=malloc(2*n*sizeof(double));
    double *boundary=malloc(4*(size_t)nf*sizeof(double));
    if(!material||!boundary){free(material);free(boundary);return CMF_EXACT_ALLOCATION_FAILED;}
    double *chi=material,*eta=material+n;
    double *sf=boundary,*ratio=sf+nf,*gradient=ratio+nf,*dnu=gradient+nf;
    CMFExactStatus status=CMF_EXACT_INVALID_INPUT;
    if(cmfgen_sum_raw_coefficients(n,es,bf_chi,bf_eta,ff_chi,ff_eta,bb_chi,bb_eta,chi,eta))goto done;
    for(int b=0;b<nf;b++){
        if(!(boundary_chi[b]>0)||!isfinite(boundary_chi[b])||
           !(boundary_eta[b]>=0)||!isfinite(boundary_eta[b])||
           !(boundary_es[b]>=0)||!isfinite(boundary_es[b]))goto done;
        sf[b]=boundary_eta[b]/boundary_chi[b];ratio[b]=boundary_es[b]/boundary_chi[b];
        dnu[b]=2*nu[b]*sinh(.5*dlog);
    }
    status=cmf_exact_diffusion_gradient(nf,nu,dnu,boundary_temperature,luminosity,
        ri[0],boundary_chi,gradient);
    if(status!=CMF_EXACT_OK)goto done;
    CMFExactInnerBoundary bc={CMF_EXACT_INNER_DIFFUSION,NAN,NAN,sf,ratio,gradient};
    if(relativistic){bc.kind=CMF_EXACT_INNER_PLANCK_DIFFUSION;bc.temperature=boundary_temperature;}
    CMFExactReport report;
    status=(relativistic ? cmf_exact_cell_solve_relativistic_raw_sources :
        cmf_exact_cell_solve_raw_sources)(ns,nf,dlog,nu,ri,ro,epoch,&bc,
        chi,es,eta,j,error,3,iteration_cap,tolerance,CMF_EXACT_MODE_POSITIVE_SLIDING,&report);
    counts[0]=(uint64_t)report.iterations_used;counts[1]=(uint64_t)report.n_rays;
    counts[2]=(uint64_t)report.ray_buffer_slots;
    counts[3]=(uint64_t)report.componentwise_error_envelope_verified;
    diagnostics[0]=report.final_max_relative_change;
    diagnostics[1]=report.max_scattering_ratio;
    diagnostics[2]=report.componentwise_residual_upper_max;
    diagnostics[3]=report.componentwise_error_upper_max;
done:
    free(material);free(boundary);return status;
}

/* 기존 ABI와 이산화를 보존한다. full-SR 선택은 별도 심볼로 명시한다. */
__attribute__((visibility("default")))
int lumina_cell_stationary_candidate(int ns,int nf,double dlog,const double *nu,
    const double *ri,const double *ro,double epoch,double temperature,double luminosity,
    const double *bchi,const double *beta,const double *bes,const double *bfchi,const double *bfeta,
    const double *ffchi,const double *ffeta,const double *bbchi,const double *bbeta,const double *es,
    double *j,double *error,int cap,double tol,uint64_t *counts,double *diagnostics)
{
    return cell_candidate(ns,nf,dlog,nu,ri,ro,epoch,temperature,luminosity,bchi,beta,bes,
        bfchi,bfeta,ffchi,ffeta,bbchi,bbeta,es,j,error,cap,tol,counts,diagnostics,0);
}

__attribute__((visibility("default")))
int lumina_cell_relativistic_candidate(int ns,int nf,double dlog,const double *nu,
    const double *ri,const double *ro,double epoch,double temperature,double luminosity,
    const double *bchi,const double *beta,const double *bes,const double *bfchi,const double *bfeta,
    const double *ffchi,const double *ffeta,const double *bbchi,const double *bbeta,const double *es,
    double *j,double *error,int cap,double tol,uint64_t *counts,double *diagnostics)
{
    return cell_candidate(ns,nf,dlog,nu,ri,ro,epoch,temperature,luminosity,bchi,beta,bes,
        bfchi,bfeta,ffchi,ffeta,bbchi,bbeta,es,j,error,cap,tol,counts,diagnostics,1);
}
