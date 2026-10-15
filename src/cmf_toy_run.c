#include "cmf_toy_model.h"
#include "cmf_time_epoch.h"
#include "cmf_gamma_source.h"
#include "cmf_nonthermal_binding.h"
#include <errno.h>
#include <unistd.h>

typedef struct {CMFToyModel *model;const char *identity;int offset;} ToyInitialSave;
static int toy_initial_save(void *context,int iteration){
    ToyInitialSave *s=context;
    if(iteration>INT32_MAX-s->offset||
       cmf_toy_initial_iterate_write(s->model,"initial_iterate.next",s->identity,s->offset+iteration)||
       rename("initial_iterate.next","initial_iterate.bin"))return -1;
    fprintf(stderr,"[TOY-ITERATE] saved=initial_iterate.bin completed_iterations=%d physical_history=0 converged=0\n",
            s->offset+iteration);
    return 0;
}

/* 운전 전체에서 불변 채널을 소유한다. 후보별 인구/ne는 provider가 다시 읽는다.
 * 부분 모형을 완전한 CMFGEN 비열 처리로 표시하지 않는다. */
typedef struct {
    double energy[256];
    CMFNonthermalBoundModel thermal;
    CMFNonthermalExcitations excitation;
    CMFNonthermalResolvedModel resolved;
    CMFNonthermalProvider provider;
    const char *label;
} ToyNonthermal;
static void toy_nonthermal_free(ToyNonthermal *nt){
    cmf_nonthermal_binding_free(&nt->resolved);
    cmf_nonthermal_excitation_free(&nt->excitation);
}
static int toy_nonthermal_prepare(ToyNonthermal *nt,const AtomicData *a,
                                  const char *raw_path,const char *binding_path){
    if(!raw_path&&!binding_path){
        nt->energy[0]=1;nt->energy[1]=2000;
        nt->thermal=(CMFNonthermalBoundModel){a->n_levels,a->n_ion_pops,2,0,nt->energy,NULL,NULL};
        nt->provider=(CMFNonthermalProvider){&nt->thermal,cmf_nonthermal_bound_evaluate};
        nt->label="ALL_THERMAL_CONTROL";return 0;
    }
    if(!raw_path||!binding_path)return -1;
    CMFNonthermalAtomicData raw={0};
    for(int i=0;i<256;i++)nt->energy[i]=exp((double)i/255*log(2000));
    int failed=cmf_nonthermal_atomic_read(&raw,raw_path)||
        cmf_nonthermal_excitation_build(a,nt->energy,256,20,&nt->excitation)||
        cmf_nonthermal_binding_build_policy(a,&raw,&nt->excitation,binding_path,
                                           CMF_NT_SIGMA_SOURCE_KINETIC,&nt->resolved);
    cmf_nonthermal_atomic_free(&raw);
    if(failed)return -1;
    nt->provider=(CMFNonthermalProvider){&nt->resolved.view,cmf_nonthermal_bound_evaluate};
    nt->label="RESOLVED_NONTHERMAL_SUBSET";
    fprintf(stderr,"[TOY-NONTHERMAL] mode=%s resolved_routes=%d unresolved_routes=%d missing_ions=%d ion_channels=%d excitation_channels=%d energy_points=256 lower_limit=20 sigma=SOURCE_KINETIC complete_atomic_model=0 public_CMFGEN_equivalence=0\n",
        nt->label,nt->resolved.n_resolved_routes,nt->resolved.n_unresolved_routes,
        nt->resolved.n_missing_ions,nt->resolved.n_ion_channels,nt->excitation.view.n_channels);
    return 0;
}

int cmf_toy_initial_main(int argc,char **argv){
    if(argc!=8&&argc!=10){fprintf(stderr,"usage: %s --cmf-toy-initial ejecta atomic_dir nuclear_data shells packets iterations [nonthermal_atomic resolved_bindings]\n",argv[0]);return 2;}
    long input[3];
    for(int i=0;i<3;i++){char *end;errno=0;input[i]=strtol(argv[5+i],&end,10);if(errno||*end)return 2;}
    int ns=(int)input[0],cap=(int)input[2];uint64_t packets=(uint64_t)input[1];
    if(input[0]<2||input[0]>256||input[1]<2||input[1]>2000000||input[2]<0||input[2]>512)return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};CMFGammaNuclearData nuclear={0};CMFToyGammaSource source={0};
    CMFGammaResult deposition={0};GammaDeposition gamma={0};CMFGENState cs={0};
    ToyNonthermal nt={0};
    ToyInitialSave save={&m,getenv("LUMINA_INITIAL_ITERATE_ID"),0};
    const char *resume=getenv("LUMINA_INITIAL_ITERATE_INPUT");
    CMFPhysicalTimeState initial={0};CMFInitialNLTEReport report={0};
    int rc=1;const char *phase="material";FILE *fp=NULL;
    if(cmf_toy_model_prepare(&m,argv[2],argv[3],ns,mass))goto done;
    phase="initial numerical restart";
    if(save.identity&&(access("initial_iterate.bin",F_OK)==0||access("initial_iterate.next",F_OK)==0))goto done;
    if(resume&&(!save.identity||cmf_toy_initial_iterate_read(&m,resume,save.identity,&save.offset)))goto done;
    if(resume){if(save.offset>=cap)goto done;cap-=save.offset;}
    if(cmf_toy_model_write_material(&m,"initial_predictor.csv",resume?
       "NUMERICAL_RESTART; not converged NLTE; J predictor only":
       "BOOTSTRAP_LTE_SAHA; not converged NLTE; radiation/gamma unpublished"))goto done;
    phase="nonthermal model";
    if(toy_nonthermal_prepare(&nt,&m.atom,argc==10?argv[8]:NULL,argc==10?argv[9]:NULL))goto done;
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
    gamma.nonthermal_provider=&nt.provider;
    fprintf(stderr,"[TOY-INITIAL] epoch=%.17g gamma_packets=%llu gamma_emitted=%.17g gamma_deposited=%.17g gamma_MCSE=%.17g zero_sample_shells=%d nonthermal=%s nuclear_Q_completion=0 initial_NLTE_converged=0\n",
        m.geometry.time_explosion,(unsigned long long)packets,deposition.emitted_comoving_erg_s,
        deposition.deposited_comoving_erg_s,deposition.deposited_comoving_standard_error,zeros,nt.label);
    phase="hydro temperature";
    fp=fopen("initial_T.csv","wx");if(!fp)goto done;
    if(fprintf(fp,"# shell T_e (hydro input at 2 days only)\n")<0)goto done;
    for(int s=0;s<ns;s++)if(fprintf(fp,"%d %.17g\n",s,m.plasma.T_e[s])<0)goto done;
    {int closed=fclose(fp);fp=NULL;if(closed)goto done;}
    if(setenv("LUMINA_FIXED_TE_PROFILE","initial_T.csv",1))goto done;
    if(!cap){printf("TOY_INITIAL_PREPARED gamma=ACTUAL_MC nonthermal=%s radiation_solved=0\n",nt.label);rc=0;goto done;}
    phase="initial coupled solve";
    if(cmfgen_init(&cs,&m.geometry))goto done;
    cs.common_obs_n=3000;cs.common_obs_nu=malloc((size_t)cs.common_obs_n*sizeof(double));
    if(!cs.common_obs_nu)goto done;
    for(int b=0;b<cs.common_obs_n;b++)cs.common_obs_nu[b]=C_SPEED_OF_LIGHT/(25500e-8)*exp(log(25500./2500)*b/(cs.common_obs_n-1));
    CMFTimeEpochStatus status=cmfgen_solve_initial_nlte_saved(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,
        &m.nlte,&m.atom,&gamma,m.plasma.T_e[0],cap,1e-5,&initial,&report,
        save.identity?toy_initial_save:NULL,&save);
    fprintf(stderr,"[TOY-INITIAL] status=%d iterations=%d calls=%d ne_change=%.17g population_change=%.17g J_change=%.17g converged=%d\n",
        status,report.iterations,report.radiation_calls,report.electron_update,report.population_update,report.radiation_update,report.converged);
    if(status!=CMF_EPOCH_OK||!report.converged)goto done;
    phase="initial history output";
    if(cmf_physical_time_state_write(&initial,"initial_state.bin")||
       cmf_toy_model_write_material(&m,"initial_converged_material.csv",argc==10?
           "CONVERGED_INITIAL_NLTE; fixed hydro T; RESOLVED_NONTHERMAL_SUBSET":
           "CONVERGED_INITIAL_NLTE; fixed hydro T; ALL_THERMAL_CONTROL")||
       cmfgen_write_common_observer(&cs,"initial_spectrum.csv","initial_spectrum.json",m.nlte.radfield_view.generation,
          m.atom.population_committed_generation,m.plasma.T_e_generation,m.opacity.tau_computed_generation))goto done;
    printf("TOY_INITIAL_CONVERGED epoch=%.17g nonthermal=%s final_epoch_19_48d=0 spectrum_compared=0\n",m.geometry.time_explosion,nt.label);
    rc=0;
done:
    if(fp)fclose(fp);
    fprintf(stderr,"[TOY-INITIAL] exit=%d phase=%s\n",rc,phase);
    cmf_physical_time_state_free(&initial);cmfgen_free(&cs);gamma_deposition_free(&gamma);
    toy_nonthermal_free(&nt);
    cmf_gamma_result_free(&deposition);cmf_toy_gamma_source_free(&source);cmf_toy_model_free(&m);
    return rc;
}

int cmf_toy_step_main(int argc,char **argv){
    if(argc!=10&&argc!=12){fprintf(stderr,"usage: %s --cmf-toy-step ejecta atomic_dir nuclear_data previous_state new_epoch_s shells packets iterations [nonthermal_atomic resolved_bindings]\n",argv[0]);return 2;}
    char *end;errno=0;double epoch=strtod(argv[6],&end);
    if(errno||*end||!(epoch>0)||!isfinite(epoch)||getenv("LUMINA_FIXED_TE_PROFILE"))return 2;
    long values[3];
    for(int i=0;i<3;i++){errno=0;values[i]=strtol(argv[7+i],&end,10);if(errno||*end)return 2;}
    if(values[0]<2||values[0]>256||values[1]<2||values[1]>2000000||values[2]<0||values[2]>512)return 2;
    int ns=(int)values[0],cap=(int)values[2];uint64_t packets=(uint64_t)values[1];
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};CMFPhysicalTimeState past={0},next={0};NLTEPhysicalHistory projected={0};
    CMFGammaNuclearData nuclear={0};CMFToyGammaSource source={0};CMFGammaResult deposition={0};
    GammaDeposition gamma={0};CMFGENState cs={0};CMFTimeRadiationTrial trial={0};CMFTimeEpochReport report={0};
    ToyNonthermal nt={0};
    double *inner=NULL;FILE *fp=NULL;int rc=1;const char *phase="material and physical history";
    if(cmf_toy_model_prepare(&m,argv[2],argv[3],ns,mass)||m.ejecta.velocity_edges[0]!=0||
       cmf_physical_time_state_read(&past,argv[5],&m.nlte,&m.atom,&m.geometry)||
       !(epoch>past.radiation.epoch_s))goto done;
    phase="nonthermal model";
    if(toy_nonthermal_prepare(&nt,&m.atom,argc==12?argv[10]:NULL,argc==12?argv[11]:NULL))goto done;
    phase="decay and expansion";
    if(cmf_gamma_nuclear_read(&nuclear,argv[4])||
       cmf_toy_gamma_source_build_at(&source,&m.ejecta,&nuclear,past.radiation.epoch_s,epoch,CMF_DECAY_INTERVAL_AVERAGE)||
       cmf_toy_model_begin_epoch(&m,&past,&source.epoch,&projected)||
       cmf_toy_model_write_material(&m,"step_predictor.csv","TIME_PREDICTOR_FROM_DECAYED_HISTORY; not converged"))goto done;
    phase="gamma";
    if(cmf_gamma_transport(&source.input,packets,UINT64_C(2026101502)^(uint64_t)llround(epoch),&deposition))goto done;
    gamma_deposition_init(&gamma,ns);
    if(!gamma.heating_rate||!gamma.nonthermal_ioniz_rate)goto done;
    fp=fopen("step_gamma.csv","wx");if(!fp)goto done;
    if(fprintf(fp,"shell,gamma_erg_cm3_s,gamma_MCSE,positive_packets,positron_erg_cm3_s,total_erg_cm3_s\n")<0)goto done;
    for(int s=0;s<ns;s++){
        gamma.heating_rate[s]=deposition.deposition_erg_cm3_s[s]+source.epoch.positron_emission[s];
        if(fprintf(fp,"%d,%.17g,%.17g,%.0f,%.17g,%.17g\n",s,deposition.deposition_erg_cm3_s[s],
            deposition.deposition_standard_error[s],deposition.positive_packet_count[s],
            source.epoch.positron_emission[s],gamma.heating_rate[s])<0)goto done;
    }
    {int closed=fclose(fp);fp=NULL;if(closed)goto done;}
    if(gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_NONLOCAL_MC,epoch,NULL,NULL,NULL,gamma.heating_rate))goto done;
    gamma.nonthermal_provider=&nt.provider;
    fprintf(stderr,"[TOY-STEP] old=%.17g new=%.17g packets=%llu gamma_deposited=%.17g gamma_MCSE=%.17g source=INTERVAL_AVERAGE nonthermal=%s nuclear_Q_completion=0 T=FREE\n",
        past.radiation.epoch_s,epoch,(unsigned long long)packets,deposition.deposited_comoving_erg_s,deposition.deposited_comoving_standard_error,nt.label);
    if(!cap){puts("TOY_STEP_PREPARED physical_history=FILE radiation_solved=0 new_epoch_converged=0");rc=0;goto done;}
    phase="time coupled solve";
    inner=calloc((size_t)past.radiation.n_bins,sizeof(double));if(!inner||cmfgen_init(&cs,&m.geometry))goto done;
    trial.old=&past.radiation;trial.epoch_s=epoch;trial.inner_H=inner;cs.time_trial=&trial;
    cs.common_obs_n=3000;cs.common_obs_nu=malloc((size_t)cs.common_obs_n*sizeof(double));
    if(!cs.common_obs_nu)goto done;
    for(int b=0;b<cs.common_obs_n;b++)cs.common_obs_nu[b]=C_SPEED_OF_LIGHT/(25500e-8)*exp(log(25500./2500)*b/(cs.common_obs_n-1));
    CMFTimeEpochStatus status=cmfgen_solve_time_epoch(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,
        &m.nlte,&m.atom,&gamma,m.plasma.T_e[0],cap,1e-5,&next,&report);
    fprintf(stderr,"[TOY-STEP] status=%d iterations=%d calls=%d T_change=%.17g ne_change=%.17g population_change=%.17g J_change=%.17g H_change=%.17g converged=%d\n",
        status,report.iterations,report.radiation_calls,report.temperature_update,report.electron_update,
        report.population_update,report.radiation_update,report.flux_update,report.converged);
    if(status!=CMF_EPOCH_OK||!report.converged)goto done;
    phase="time history output";
    if(cmf_physical_time_state_write(&next,"step_state.bin")||
       cmf_toy_model_write_material(&m,"step_material.csv",argc==12?
           "CONVERGED_TIME_NLTE; FREE_T; RESOLVED_NONTHERMAL_SUBSET":
           "CONVERGED_TIME_NLTE; FREE_T; ALL_THERMAL_CONTROL")||
       cmfgen_write_common_observer(&cs,"step_spectrum.csv","step_spectrum.json",m.nlte.radfield_view.generation,
          m.atom.population_committed_generation,m.plasma.T_e_generation,m.opacity.tau_computed_generation))goto done;
    printf("TOY_STEP_CONVERGED epoch=%.17g nonthermal=%s spectrum_compared=0\n",epoch,nt.label);rc=0;
done:
    if(fp)fclose(fp);
    fprintf(stderr,"[TOY-STEP] exit=%d phase=%s\n",rc,phase);
    cmfgen_free(&cs);cmf_time_radiation_trial_free(&trial);free(inner);gamma_deposition_free(&gamma);
    toy_nonthermal_free(&nt);
    cmf_gamma_result_free(&deposition);cmf_toy_gamma_source_free(&source);cmf_toy_model_free(&m);
    nlte_physical_history_free(&projected);cmf_physical_time_state_free(&past);cmf_physical_time_state_free(&next);
    return rc;
}
