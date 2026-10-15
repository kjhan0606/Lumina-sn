/* 실제 toy06/원자 자료의 시간 연결 시험. J/H는 명시적 합성 fixture다.
 * 수렴 초기해·초신성 스펙트럼으로 해석하거나 생산 이력으로 쓰지 않는다. */
#include "cmf_toy_model.h"
#include "cmf_gamma_source.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TOY_EPOCH_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
int main(int argc,char **argv){
    if(argc==10&&!strcmp(argv[1],"--cmf-toy-step"))return cmf_toy_step_main(argc,argv);
    if(argc!=4)return 2;
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};CMFPhysicalTimeState past={0},updated={0},restored={0};
    NLTEPhysicalHistory projected={0};CMFGammaNuclearData nuclear={0};CMFToyGammaSource source={0};
    double *radiation=NULL,*saved=NULL,*initial_abundances=NULL;int ns=atoi(argv[3]),rc=0;
    CHECK(!cmf_toy_model_prepare(&m,argv[1],argv[2],ns,masses));
    initial_abundances=malloc((size_t)6*ns*sizeof(double));CHECK(initial_abundances);
    memcpy(initial_abundances,m.atom.abundances,(size_t)6*ns*sizeof(double));
    radiation=calloc((size_t)(4*ns+2),sizeof(double));CHECK(radiation);
    for(int i=0;i<2*ns;i++)radiation[i]=1e-7;
    for(int i=2;i<2*(ns+1);i++)radiation[2*ns+i]=1e-10;
    CMFTimeRadiationHistory rh={ns,2,m.geometry.time_explosion,1e14,.1,m.ejecta.velocity_edges,radiation,radiation+2*ns};
    CHECK(!cmf_physical_time_state_capture(&past,&m.nlte,&m.atom,&m.plasma,&rh));
    size_t cells=(size_t)(5+m.atom.n_ion_pops+m.nlte.n_nlte_levels_total)*ns;
    saved=malloc(cells*sizeof(double));CHECK(saved);memcpy(saved,past.material.storage,cells*sizeof(double));
    CHECK(!cmf_gamma_nuclear_read(&nuclear,"input/NUC_DECAY_DATA_ni56_co56_fe56"));
    CHECK(!cmf_toy_gamma_source_build_at(&source,&m.ejecta,&nuclear,172800,2.02*86400,CMF_DECAY_INTERVAL_AVERAGE));
    double density=source.epoch.density[0];source.epoch.density[0]*=2;
    CHECK(cmf_toy_model_begin_epoch(&m,&past,&source.epoch,&projected)!=0);
    CHECK(m.prepared&&!projected.storage&&!m.plasma.physical_history&&m.geometry.time_explosion==172800);
    source.epoch.density[0]=density;
    CHECK(!cmf_toy_model_begin_epoch(&m,&past,&source.epoch,&projected));
    CHECK(m.prepared&&m.plasma.physical_history==&projected&&!m.plasma.initial_steady_nlte);
    CHECK(projected.decay_epoch_s==source.epoch.epoch_s&&projected.energy.epoch_s==172800);
    CHECK(m.plasma.te_publication.te_lane==A210_TE_LANE_FREE_T&&m.plasma.te_publication.re_root_required);
    CHECK(!memcmp(m.plasma.T_e,past.material.energy.temperature_K,(size_t)ns*sizeof(double)));
    CHECK(!memcmp(saved,past.material.storage,cells*sizeof(double)));
    CHECK(!memcmp(past.radiation.J,radiation,(size_t)2*ns*sizeof(double)));
    CHECK(!memcmp(past.radiation.H,radiation+2*ns,(size_t)2*(ns+1)*sizeof(double)));
    const int Z[6]={28,27,26,20,16,14};
    double dilution=pow(172800/source.epoch.epoch_s,3),nuclei_error=0,charge_error=0;
    double ni_before=0,ni_after=0;
    for(int s=0;s<ns;s++){
        double charge=0;
        for(int k=0;k<6;k++){
            double sum=0,before=0;
            for(int ip=0;ip<m.atom.n_ion_pops;ip++)if(m.atom.ion_pop_Z[ip]==Z[k]){
                double value=m.atom.ion_number_density[(size_t)ip*ns+s];
                CHECK(isfinite(value)&&value>=0);sum+=value;charge+=value*m.atom.ion_pop_stage[ip];
                before+=past.material.ion_density[(size_t)ip*ns+s];
            }
            double expected=source.epoch.number_density[(size_t)k*ns+s];
            if(expected>0)nuclei_error=fmax(nuclei_error,fabs(sum/expected-1));else CHECK(sum==0);
            if(k==0){ni_before+=before;ni_after+=sum/dilution;}
        }
        CHECK(charge>0);charge_error=fmax(charge_error,fabs(m.plasma.n_electron[s]/charge-1));
    }
    CHECK(nuclei_error<1e-10&&charge_error<1e-10&&ni_after<ni_before);
    CHECK(fabs(ni_after/ni_before-exp(-nuclear.decay.lambda[0]*(source.epoch.epoch_s-172800)))<1e-12);
    for(size_t i=0;i<(size_t)m.nlte.n_nlte_levels_total*ns;i++)
        CHECK(m.nlte.nlte_level_populations[i]==projected.level_density[i]*dilution);
    CHECK(!tau_sobolev_assert_fresh(&m.opacity,"toy epoch selftest"));
    CHECK(cmf_toy_model_begin_epoch(&m,&past,&source.epoch,&projected)!=0);
    /* 이력 reader는 현재 원자객체의2일 조성으로 과거 에너지를 검증하면 안 된다. */
    rh.epoch_s=source.epoch.epoch_s;
    CHECK(!cmf_physical_time_state_capture(&updated,&m.nlte,&m.atom,&m.plasma,&rh));
    CHECK(!cmf_physical_time_state_write(&updated,"fixture_nonconverged_epoch.bin"));
    memcpy(m.atom.abundances,initial_abundances,(size_t)6*ns*sizeof(double));
    CHECK(!cmf_physical_time_state_read(&restored,"fixture_nonconverged_epoch.bin",&m.nlte,&m.atom,&m.geometry));
    CHECK(!memcmp(updated.material.storage,restored.material.storage,cells*sizeof(double)));
    printf("TOY_EPOCH_SELFTEST_PASS shells=%d nuclei_error=%.17g charge_error=%.17g Ni_fraction=%.17g past_preserved=1 checkpoint_exact=1 synthetic_JH=1 NLTE_converged=0 spectrum_compared=0\n",
        ns,nuclei_error,charge_error,ni_after/ni_before);
done:
    cmf_toy_model_free(&m);nlte_physical_history_free(&projected);
    cmf_physical_time_state_free(&past);cmf_physical_time_state_free(&updated);cmf_physical_time_state_free(&restored);
    cmf_toy_gamma_source_free(&source);free(radiation);free(saved);free(initial_abundances);return rc;
}
