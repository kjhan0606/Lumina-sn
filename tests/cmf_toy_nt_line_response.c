/* 생산 저장 침적을 재사용한다. MC를 다시 돌리지 않으며 현재 고정 물질의
 * 실제 생산 비열 공급자로 전이율을 재평가한다. 공간 수송/전하해는 없다. */
#include "cmf_toy_model.h"
#include "cmf_nonthermal_binding.h"
static int response_selected_Z(int z);
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
static double nt_energy[256];
static CMFNonthermalExcitations nt_excitation;
static CMFNonthermalResolvedModel nt_resolved;
static CMFNonthermalProvider nt_provider;
static CMFNonthermalRates nt_rates;
static void response_nonthermal_free(void){
    cmf_nonthermal_rates_free(&nt_rates);
    cmf_nonthermal_binding_free(&nt_resolved);
    cmf_nonthermal_excitation_free(&nt_excitation);
}
static int response_nonthermal_prepare(CMFToyModel *m,GammaDeposition *g,int shell){
    const char *gamma_path=getenv("LUMINA_DIAG_GAMMA_CSV");
    const char *raw_path=getenv("LUMINA_DIAG_NT_ATOMIC");
    const char *binding_path=getenv("LUMINA_DIAG_NT_BINDINGS");
    if(!gamma_path||!raw_path||!binding_path)return -1;
    FILE *f=fopen(gamma_path,"r");if(!f)return -1;
    char line[1024];int ns=m->plasma.n_shells,rows=0,rc=-1;
    double *heat=calloc((size_t)ns,sizeof(double));if(!heat){fclose(f);return -1;}
    if(!fgets(line,sizeof(line),f)||strcmp(line,"shell,gamma_erg_cm3_s,gamma_MCSE,positive_packets,positron_erg_cm3_s,total_erg_cm3_s\n"))goto done;
    while(fgets(line,sizeof(line),f)){
        int s;unsigned long long packets;double dep,se,positron,total;char extra;
        if(sscanf(line,"%d,%lf,%lf,%llu,%lf,%lf %c",&s,&dep,&se,&packets,&positron,&total,&extra)!=6||
           s!=rows||s>=ns||!(total>=0)||!isfinite(total)||fabs(dep+positron-total)>1e-14*fmax(1.,total))goto done;
        heat[s]=total;rows++;
    }
    if(rows!=ns)goto done;
    if(gamma_deposition_publish(g,GAMMA_PROVENANCE_NONLOCAL_MC,m->geometry.time_explosion,NULL,NULL,NULL,heat))goto done;
    for(int i=0;i<256;i++)nt_energy[i]=exp((double)i/255*log(2000));
    CMFNonthermalAtomicData raw={0};
    int failed=cmf_nonthermal_atomic_read(&raw,raw_path)||
        cmf_nonthermal_excitation_build(&m->atom,nt_energy,256,20,&nt_excitation)||
        cmf_nonthermal_binding_build_policy(&m->atom,&raw,&nt_excitation,binding_path,CMF_NT_SIGMA_SOURCE_KINETIC,&nt_resolved);
    cmf_nonthermal_atomic_free(&raw);if(failed)goto done;
    nt_provider=(CMFNonthermalProvider){&nt_resolved.view,cmf_nonthermal_bound_evaluate};
    g->nonthermal_provider=&nt_provider;
    if(nt_provider.evaluate(nt_provider.context,&m->atom,&m->nlte,&m->plasma,g,&nt_rates))goto done;
    g->nonthermal_rates=&nt_rates;
    fclose(f);f=fopen("nonthermal_rates.csv","wx");if(!f)goto done;
    fprintf(f,"Z,lower_global,upper_global,upper_ion,rate_per_particle\n");
    for(int i=0;i<nt_rates.n_transitions;i++){
        const CMFNonthermalTransition *q=&nt_rates.transitions[i];int Z=m->atom.level_Z[q->lower_global];
        if(!response_selected_Z(Z))continue;
        fprintf(f,"%d,%d,%d,%d,%.17g\n",Z,q->lower_global,q->upper_global,q->upper_ion,
                nt_rates.rate_per_particle[(size_t)i*ns+shell]);
    }
    printf("NT_RESPONSE_SOURCE shell=%d deposition=%.17g replayed_MC=1 new_MC=0 resolved=%d unresolved=%d missing_ions=%d ion_channels=%d excitation_channels=%d frozen_candidate_rates=1\n",
        shell,heat[shell],nt_resolved.n_resolved_routes,nt_resolved.n_unresolved_routes,
        nt_resolved.n_missing_ions,nt_resolved.n_ion_channels,nt_excitation.view.n_channels);
    rc=0;
done:
    if(f&&fclose(f))rc=-1;
    free(heat);return rc;
}
#define LUMINA_RESPONSE_NONTHERMAL 1
#include "cmf_toy_line_response.c"
