#include "cmf_toy_model.h"
#include "cmf_time_epoch.h"
#include "cmf_gamma_source.h"
#include "cmf_nonthermal_provider.h"
#include <errno.h>

/* 성공한 초기 물리 이력만 저장한다. 고정 너비 header 뒤에는 int32 IDs,
 * double 배열이 순서대로 온다. footer가 없으면 완결된 이력이 아니다.
 * native little-endian binary, 이 실행의 source/input hash는 runner가 보존. */
static int write_initial_state(const CMFPhysicalTimeState *s){
    if(!s||!s->material.storage||!s->radiation_storage||sizeof(int)!=4||sizeof(double)!=8)return -1;
    uint64_t ns=s->radiation.n_shells,nb=s->radiation.n_bins,ni=s->material.n_ions,nl=s->material.n_levels;
    uint64_t head[]={1,ns,nb,ni,nl,s->material.population_generation,s->material.temperature_generation};
    double parameters[]={s->radiation.epoch_s,s->radiation.nu_min,s->radiation.dlognu,s->material.decay_epoch_s};
    uint64_t material=(5+ni+nl)*ns,radiation=ns+1+(2*ns+1)*nb;
    uint64_t bytes=8+sizeof(head)+sizeof(parameters)+4*(nl+2*ni)+8*(material+radiation)+8;
    if(bytes>UINT64_C(2147483648))return -1;
    FILE *fp=fopen("initial_state.bin","wbx");if(!fp)return -1;
    int rc=0;
#define PUT(ptr,size,count) do{if(fwrite((ptr),(size),(count),fp)!=(count))rc=-1;}while(0)
    PUT("CMFINIT1",1,8);PUT(head,sizeof(uint64_t),7);PUT(parameters,sizeof(double),4);
    PUT(s->material.level_global,4,nl);PUT(s->material.ion_Z,4,ni);PUT(s->material.ion_stage,4,ni);
    PUT(s->material.storage,8,material);PUT(s->radiation_storage,8,radiation);PUT("CMFDONE1",1,8);
#undef PUT
    if(fclose(fp))rc=-1;
    if(!rc)fprintf(stderr,"[TOY-INITIAL] saved_state_bytes=%llu epoch=%.17g bins=%llu\n",
                   (unsigned long long)bytes,parameters[0],(unsigned long long)nb);
    return rc;
}

int cmf_toy_initial_main(int argc,char **argv){
    if(argc!=8){fprintf(stderr,"usage: %s --cmf-toy-initial ejecta atomic_dir nuclear_data shells packets iterations\n",argv[0]);return 2;}
    long input[3];
    for(int i=0;i<3;i++){char *end;errno=0;input[i]=strtol(argv[5+i],&end,10);if(errno||*end)return 2;}
    int ns=(int)input[0],cap=(int)input[2];uint64_t packets=(uint64_t)input[1];
    if(input[0]<2||input[0]>256||input[1]<2||input[1]>2000000||input[2]<0||input[2]>512)return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};CMFGammaNuclearData nuclear={0};CMFToyGammaSource source={0};
    CMFGammaResult deposition={0};GammaDeposition gamma={0};CMFGENState cs={0};
    CMFPhysicalTimeState initial={0};CMFInitialNLTEReport report={0};
    int rc=1;const char *phase="material";FILE *fp=NULL;
    if(cmf_toy_model_prepare(&m,argv[2],argv[3],ns,mass))goto done;
    if(cmf_toy_model_write_seed(&m,"initial_predictor.csv"))goto done;
    phase="gamma";
    if(cmf_gamma_nuclear_read(&nuclear,argv[4])||
       cmf_toy_gamma_source_build(&source,&m.ejecta,&nuclear,m.geometry.time_explosion)||
       cmf_gamma_transport(&source.input,packets,UINT64_C(2026101502),&deposition))goto done;
    gamma_deposition_init(&gamma,ns);
    if(!gamma.heating_rate||!gamma.nonthermal_ioniz_rate)goto done;
    fp=fopen("initial_gamma.csv","wx");if(!fp)goto done;
    if(fprintf(fp,"shell,gamma_erg_cm3_s,gamma_MCSE,positive_packets,positron_erg_cm3_s,total_erg_cm3_s\n")<0)goto done;
    int zeros=0;
    for(int s=0;s<ns;s++){
        gamma.heating_rate[s]=deposition.deposition_erg_cm3_s[s]+source.epoch.positron_emission[s];
        zeros+=deposition.positive_packet_count[s]==0;
        if(fprintf(fp,"%d,%.17g,%.17g,%.0f,%.17g,%.17g\n",s,deposition.deposition_erg_cm3_s[s],
            deposition.deposition_standard_error[s],deposition.positive_packet_count[s],
            source.epoch.positron_emission[s],gamma.heating_rate[s])<0)goto done;
    }
    {int closed=fclose(fp);fp=NULL;if(closed)goto done;}
    if(gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_NONLOCAL_MC,m.geometry.time_explosion,NULL,NULL,NULL,gamma.heating_rate))goto done;
    /* 명시적인 all-thermal 대조다. 비열이 물리적으로0이라고 주장하지 않는다.
     * 빈 채널 provider로 기존 ETA/W 경로가 개입하지 않게 한다. */
    double energy[]={1,2000};
    CMFNonthermalBoundModel thermal={m.atom.n_levels,m.atom.n_ion_pops,2,0,energy,NULL,NULL};
    CMFNonthermalProvider provider={&thermal,cmf_nonthermal_bound_evaluate};gamma.nonthermal_provider=&provider;
    fprintf(stderr,"[TOY-INITIAL] epoch=%.17g gamma_packets=%llu gamma_emitted=%.17g gamma_deposited=%.17g gamma_MCSE=%.17g zero_sample_shells=%d nonthermal=ALL_THERMAL_CONTROL nuclear_Q_completion=0 initial_NLTE_converged=0\n",
        m.geometry.time_explosion,(unsigned long long)packets,deposition.emitted_comoving_erg_s,
        deposition.deposited_comoving_erg_s,deposition.deposited_comoving_standard_error,zeros);
    phase="hydro temperature";
    fp=fopen("initial_T.csv","wx");if(!fp)goto done;
    if(fprintf(fp,"# shell T_e (hydro input at 2 days only)\n")<0)goto done;
    for(int s=0;s<ns;s++)if(fprintf(fp,"%d %.17g\n",s,m.plasma.T_e[s])<0)goto done;
    {int closed=fclose(fp);fp=NULL;if(closed)goto done;}
    if(setenv("LUMINA_FIXED_TE_PROFILE","initial_T.csv",1))goto done;
    if(!cap){printf("TOY_INITIAL_PREPARED gamma=ACTUAL_MC nonthermal=ALL_THERMAL_CONTROL radiation_solved=0\n");rc=0;goto done;}
    phase="initial coupled solve";
    if(cmfgen_init(&cs,&m.geometry))goto done;
    cs.common_obs_n=3000;cs.common_obs_nu=malloc((size_t)cs.common_obs_n*sizeof(double));
    if(!cs.common_obs_nu)goto done;
    for(int b=0;b<cs.common_obs_n;b++)cs.common_obs_nu[b]=C_SPEED_OF_LIGHT/(25500e-8)*exp(log(25500./2500)*b/(cs.common_obs_n-1));
    CMFTimeEpochStatus status=cmfgen_solve_initial_nlte(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,
        &m.nlte,&m.atom,&gamma,m.plasma.T_e[0],cap,1e-5,&initial,&report);
    fprintf(stderr,"[TOY-INITIAL] status=%d iterations=%d calls=%d ne_change=%.17g population_change=%.17g J_change=%.17g converged=%d\n",
        status,report.iterations,report.radiation_calls,report.electron_update,report.population_update,report.radiation_update,report.converged);
    if(status!=CMF_EPOCH_OK||!report.converged)goto done;
    phase="initial history output";
    if(write_initial_state(&initial)||cmf_toy_model_write_seed(&m,"initial_converged_material.csv")||
       cmfgen_write_common_observer(&cs,"initial_spectrum.csv","initial_spectrum.json",m.nlte.radfield_view.generation,
          m.atom.population_committed_generation,m.plasma.T_e_generation,m.opacity.tau_computed_generation))goto done;
    printf("TOY_INITIAL_CONVERGED epoch=%.17g nonthermal=ALL_THERMAL_CONTROL final_epoch_19_48d=0 spectrum_compared=0\n",m.geometry.time_explosion);
    rc=0;
done:
    if(fp)fclose(fp);
    fprintf(stderr,"[TOY-INITIAL] exit=%d phase=%s\n",rc,phase);
    cmf_physical_time_state_free(&initial);cmfgen_free(&cs);gamma_deposition_free(&gamma);
    cmf_gamma_result_free(&deposition);cmf_toy_gamma_source_free(&source);cmf_toy_model_free(&m);
    return rc;
}
