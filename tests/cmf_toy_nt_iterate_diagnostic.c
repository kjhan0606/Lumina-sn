/* 저장된 실제 인구/감마 침적의 부분 비열 응답. 공동 물리해를 만들지 않는다. */
#include "cmf_toy_model.h"
#include "cmf_nonthermal_binding.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define REQUIRE(x) do {if(!(x)){fprintf(stderr,"TOY_NT_ITERATE_FAIL line=%d %s\n",__LINE__,#x);goto done;}} while(0)

/* 같은 실제 인구에서 격자 세분화와 경계 확장을 분리한다. NLTE를 다시 풀지 않는다. */
static int grid_comparison(const CMFToyModel *m,const CMFNonthermalAtomicData *raw,
                           const char *binding,int iteration,int source_mode){
    enum {NS=4,CASES=5};const int selected[NS]={30,39,41,43};
    const int points[CASES]={256,511,1021,320,280};
    const char *names[CASES]={"base256","refine511","refine1021","lower0p1","upper4000"};
    const char *source_names[CASES]={"base256","upper4000","upper8000","upper16000","upper1000"};
    const AtomicData *a=&m->atom;int ns=m->plasma.n_shells,rc=-1,rows=0;
    double *pop=NULL,*energy=NULL,electron[NS],power[NS];
    long double baseline[NS][4][2]={0},fine[NS][4][2]={0};
    CMFNonthermalExcitations exc={0};CMFNonthermalResolvedModel bound={0};CMFNonthermalRates rates={0};
    NLTEConfig n=m->nlte;PlasmaState p={0};GammaDeposition g={0};FILE *fp=NULL;
    REQUIRE(ns==113);
    pop=malloc((size_t)n.n_nlte_levels_total*NS*sizeof(double));REQUIRE(pop);
    for(int b=0;b<NS;b++){
        int s=selected[b];electron[b]=m->plasma.n_electron[s];power[b]=1.602176634e-12;
        for(int k=0;k<n.n_nlte_levels_total;k++)pop[(size_t)k*NS+b]=m->nlte.nlte_level_populations[(size_t)k*ns+s];
    }
    n.nlte_level_populations=pop;p.n_shells=NS;p.n_electron=electron;
    g.n_shells=NS;g.heating_rate=power;g.generation=1;g.epoch=m->geometry.time_explosion;
    g.provenance=GAMMA_PROVENANCE_NONLOCAL_MC;
    fp=fopen("iterate_nonthermal_grid.csv","wx");REQUIRE(fp);
    REQUIRE(fprintf(fp,"case,iteration,shell,Z,charge,n_energy,Emin_eV,Emax_eV,single_events_per_eV,auger_events_per_eV,single_relative_to_base,auger_relative_to_base,single_per_ion_per_unit_eV,heat_fraction,ion_fraction,excitation_fraction\n")>0);
    for(int c=0;c<CASES;c++){
        int ne=points[c];
        if(source_mode){
            ne=256+24*c;
            if(c==4){ne=0;while(exp((double)ne/255*log(2000))<1000)ne++;ne++;}
        }
        energy=malloc((size_t)ne*sizeof(double));REQUIRE(energy);
        if(source_mode){
            if(c==4){
                for(int e=0;e<ne-1;e++)energy[e]=exp((double)e/255*log(2000));
                energy[ne-1]=1000;
            }else{
                for(int e=0;e<256;e++)energy[e]=exp((double)e/255*log(2000));
                for(int e=1;e<=24*c;e++)energy[255+e]=2000*exp((double)e/24*log(2));
            }
        }else if(c<3)for(int e=0;e<ne;e++)energy[e]=exp((double)e/(ne-1)*log(2000));
        else if(c==3){
            for(int e=0;e<64;e++)energy[e]=exp(log(.1)+(double)e/64*log(10));
            for(int e=0;e<256;e++)energy[64+e]=exp((double)e/255*log(2000));
        }else{
            for(int e=0;e<256;e++)energy[e]=exp((double)e/255*log(2000));
            for(int e=1;e<=24;e++)energy[255+e]=2000*exp((double)e/24*log(2));
        }
        struct timespec start,finish;REQUIRE(!clock_gettime(CLOCK_MONOTONIC,&start));
        REQUIRE(!cmf_nonthermal_excitation_build(a,energy,ne,20,&exc));
        REQUIRE(!cmf_nonthermal_binding_build_policy(a,raw,&exc,binding,CMF_NT_SIGMA_SOURCE_KINETIC,&bound));
        REQUIRE(bound.n_resolved_routes==41&&bound.n_unresolved_routes==10&&bound.n_missing_ions==2);
        REQUIRE(!cmf_nonthermal_bound_evaluate(&bound.view,a,&n,&p,&g,&rates));
        REQUIRE(!clock_gettime(CLOCK_MONOTONIC,&finish));
        long double max_base=0,max_refinement=0;
        for(int b=0;b<NS;b++){
            long double event[4][2]={0},ion=0,excitation=0;
            for(int k=0;k<rates.n_transitions;k++){
                const CMFNonthermalTransition *t=&rates.transitions[k];int lo=t->lower_global,electrons;double delta;
                REQUIRE(!cmf_nonthermal_transition_energy(a,t,&delta,&electrons));
                int lower=n.global_to_nlte_level[lo];REQUIRE(lower>=0&&lower<n.n_nlte_levels_total);
                long double count=(long double)rates.rate_per_particle[(size_t)k*NS+b]*pop[(size_t)lower*NS+b];
                REQUIRE(count>=0&&isfinite((double)count));
                if(electrons)ion+=count*delta;else excitation+=count*delta;
                int z=a->level_Z[lo],q=a->level_ion[lo];
                if(electrons&&(z==14||z==16)&&(q==1||q==2))event[(z==16?2:0)+q-1][electrons-1]+=count;
            }
            REQUIRE(ion>=0&&excitation>=0&&ion+excitation<=1+1e-10L);
            for(int j=0;j<4;j++){
                int z=j<2?14:16,q=j%2+1,ip=-1;
                for(int k=0;k<a->n_ion_pops;k++)if(a->ion_pop_Z[k]==z&&a->ion_pop_stage[k]==q)ip=k;
                REQUIRE(ip>=0);double population=a->ion_number_density[(size_t)ip*ns+selected[b]];REQUIRE(population>0);
                long double difference[2];
                for(int k=0;k<2;k++){
                    if(c==0)baseline[b][j][k]=event[j][k];
                    if(c==1&&!source_mode)fine[b][j][k]=event[j][k];
                    /* 사건이 전혀 없는 채널은 두 값이 모두0일 때만 차이0이다. */
                    REQUIRE(baseline[b][j][k]>0||event[j][k]==0);
                    difference[k]=baseline[b][j][k]>0?event[j][k]/baseline[b][j][k]-1:0;
                    max_base=fmaxl(max_base,fabsl(difference[k]));
                    if(c==2&&!source_mode){
                        REQUIRE(fine[b][j][k]>0||event[j][k]==0);
                        long double d=fine[b][j][k]>0?event[j][k]/fine[b][j][k]-1:0;
                        max_refinement=fmaxl(max_refinement,fabsl(d));
                    }
                }
                REQUIRE(fprintf(fp,"%s,%d,%d,%d,%d,%d,%.17g,%.17g,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg\n",
                    source_mode?source_names[c]:names[c],iteration,selected[b],z,q,ne,energy[0],energy[ne-1],event[j][0],event[j][1],
                    difference[0],difference[1],event[j][0]/population,1-ion-excitation,ion,excitation)>0);rows++;
            }
        }
        printf("TOY_NT_GRID case=%s points=%d shells=%d seconds=%.9g max_rate_relative_to_base=%.9Lg max_rate_1021_vs_511=%.9Lg physical_equivalence=NOT_JUDGED\n",
            source_mode?source_names[c]:names[c],ne,NS,(double)(finish.tv_sec-start.tv_sec)+(finish.tv_nsec-start.tv_nsec)*1e-9,max_base,max_refinement);fflush(stdout);
        cmf_nonthermal_rates_free(&rates);cmf_nonthermal_binding_free(&bound);cmf_nonthermal_excitation_free(&exc);
        free(energy);energy=NULL;
    }
    REQUIRE(rows==CASES*NS*4);
    puts("TOY_NT_GRID_COMPLETE fixed_population=1 transport_solved=0 NLTE_solved=0 physical_gamma_used=0");rc=0;
done:
    if(fp&&fclose(fp))rc=-1;
    cmf_nonthermal_rates_free(&rates);cmf_nonthermal_binding_free(&bound);cmf_nonthermal_excitation_free(&exc);
    free(pop);free(energy);return rc;
}

static int read_gamma(const char *path,int ns,double *v){
    FILE *fp=fopen(path,"r");if(!fp)return -1;
    char line[512];int rc=-1;
    if(!fgets(line,sizeof line,fp)||strcmp(line,"shell,gamma_erg_cm3_s,gamma_MCSE,positive_packets,positron_erg_cm3_s,total_erg_cm3_s\n"))goto done;
    for(int s=0;s<ns;s++){
        int index=-1,used=0;
        if(!fgets(line,sizeof line,fp)||sscanf(line,"%d,%lf,%lf,%lf,%lf,%lf%n",&index,
            &v[s],&v[ns+s],&v[2*ns+s],&v[3*ns+s],&v[4*ns+s],&used)!=6||
            index!=s||strcmp(line+used,"\n"))goto done;
        for(int k=0;k<5;k++)if(!(v[k*ns+s]>=0)||!isfinite(v[k*ns+s]))goto done;
        if(floor(v[2*ns+s])!=v[2*ns+s]||(v[2*ns+s]==0&&(v[s]!=0||v[ns+s]!=0)))goto done;
        double sum=v[s]+v[3*ns+s];
        if(sum==0?v[4*ns+s]!=0:fabs(v[4*ns+s]/sum-1)>1e-12)goto done;
    }
    if(fgetc(fp)!=EOF||ferror(fp))goto done;
    rc=0;
done:
    fclose(fp);return rc;
}

/* 이미 검증된 연결표의 순서와 실제 bound 채널을 대조한다. 물리율을 바꾸지 않는다. */
static int *binding_sources(const char *path,const CMFNonthermalAtomicData *raw,
                            const CMFNonthermalResolvedModel *bound,int offset){
    FILE *fp=fopen(path,"r");if(!fp)return NULL;
    int *index=malloc((size_t)bound->n_ion_channels*sizeof(*index));char line[512];
    if(!index||!fgets(line,sizeof line,fp)||!fgets(line,sizeof line,fp))goto fail;
    for(int k=0;k<bound->n_ion_channels;k++){
        int source,lo,found=-1;
        if(!fgets(line,sizeof line,fp)||sscanf(line,"%d,%d,",&source,&lo)!=2||
           lo!=bound->transitions[offset+k].lower_global)goto fail;
        for(int j=0;j<raw->n_routes;j++)if(raw->routes[j].source_line==source){if(found>=0)goto fail;found=j;}
        if(found<0)goto fail;index[k]=found;
    }
    if(fgetc(fp)!=EOF||ferror(fp))goto fail;
    fclose(fp);return index;
fail:
    fclose(fp);free(index);return NULL;
}

static int missing_bounds(const CMFToyModel *m,int s,int iteration,const CMFNonthermalAtomicData *raw,
    const CMFNonthermalResolvedModel *bound,int offset,const int *source,
    const CMFNonthermalRates *unit,FILE *fp){
    int represented[CMF_NT_MAX_ROUTES]={0},missing=0,absent=0,ns=m->plasma.n_shells;
    const AtomicData *a=&m->atom;
    for(int k=0;k<bound->n_ion_channels;k++)represented[source[k]]=1;
    for(int ip=0;ip<a->n_ion_pops;ip++){
        int ground=a->level_offset[ip],z=a->ion_pop_Z[ip],q=a->ion_pop_stage[ip];
        if(ground==a->level_offset[ip+1])continue;
        if(a->level_num[ground]!=0||a->level_energy_eV[ground]!=0)return -1;
        double population=a->ion_number_density[(size_t)ip*ns+s],total=0;
        for(int j=0;j<a->n_ion_pops;j++)if(a->ion_pop_Z[j]==z)total+=a->ion_number_density[(size_t)j*ns+s];
        if(!(population>=0)||!(total>=0)||!isfinite(population+total))return -1;
        /* 내부 셸에는 Si/S 자체가 없을 수 있다. 0/0 이온 분율은 빈 칸이다. */
        char fraction[32]="";
        if(total>0)snprintf(fraction,sizeof fraction,"%.17g",population/total);
        int found=0,base=-1;
        for(int r=0;r<raw->n_routes;r++)if(raw->routes[r].Z==z&&raw->routes[r].charge==q){
            found++;
            if(!represented[r])continue;
            for(int k=0;k<bound->n_ion_channels;k++)if(source[k]==r&&bound->transitions[offset+k].lower_global==ground){
                if(base<0||raw->routes[r].threshold_eV<raw->routes[base].threshold_eV)base=r;
                break;
            }
        }
        if(!found){
            absent++;
            if(fprintf(fp,"%d,%d,%d,%d,0,%.17g,%s,,,,,,NO_ATOMIC_CROSS_SECTION\n",
                       iteration,s,z,q,population,fraction)<0)return -1;
        }
        for(int r=0;r<raw->n_routes;r++)if(raw->routes[r].Z==z&&raw->routes[r].charge==q&&!represented[r]){
            missing++;
            const CMFNonthermalAtomicRoute *route=&raw->routes[r];const char *term=route->terms;
            while(isspace((unsigned char)*term))term++;
            const char *status=base<0?"NO_DOMINATING_REFERENCE":"FIXED_DEGRADATION_UPPER_BOUND";
            if(*term=='*')status="UNSUPPORTED_EXCITED_LOWER";
            double ratio=0;long double coefficient=0;
            if(base>=0&&*term!='*'){
                for(int k=0;k<bound->n_ion_channels;k++)if(source[k]==base&&bound->transitions[offset+k].lower_global==ground)
                    coefficient+=unit->rate_per_particle[offset+k];
                for(int e=0;e<bound->view.n_energy;e++){
                    double sig=cmf_nonthermal_atomic_sigma(route,bound->view.energy_eV[e]),reference=0;
                    if(!(sig>=0)||!isfinite(sig))return -1;
                    for(int k=0;k<bound->n_ion_channels;k++)if(source[k]==base&&bound->transitions[offset+k].lower_global==ground)
                        reference+=bound->channels[offset+k].cross_section_cm2[e];
                    if(sig>0){
                        if(!(reference>0)){status="NO_DOMINATING_REFERENCE";break;}
                        ratio=fmax(ratio,sig/reference);
                    }
                }
            }
            int bounded=!strcmp(status,"FIXED_DEGRADATION_UPPER_BOUND");
            if(bounded){
                long double events=ratio*coefficient*population;
                if(!isfinite(ratio)||!isfinite((double)events)||events<0)return -1;
                if(fprintf(fp,"%d,%d,%d,%d,%d,%.17g,%s,%.17g,%d,%.17g,%.17Lg,%.17Lg,%s\n",
                    iteration,s,z,q,route->source_line,population,fraction,route->threshold_eV,
                    raw->routes[base].source_line,ratio,events,events*route->threshold_eV,status)<0)return -1;
            }else if(fprintf(fp,"%d,%d,%d,%d,%d,%.17g,%s,%.17g,,,,,%s\n",
                            iteration,s,z,q,route->source_line,population,fraction,route->threshold_eV,status)<0)return -1;
        }
    }
    if(missing!=bound->n_unresolved_routes||absent!=bound->n_missing_ions)return -1;
    printf("TOY_NT_COVERAGE shell=%d missing_routes=%d missing_ions=%d fixed_degradation_only=1 coupled_feedback_bounded=0\n",s,missing,absent);
    return 0;
}

int main(int argc,char **argv){
    if(argc!=9)return 2;
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    const double ev=1.602176634e-12;
    CMFToyModel m={0};CMFNonthermalAtomicData raw={0};CMFNonthermalExcitations exc={0};
    CMFNonthermalResolvedModel bound={0};CMFNonthermalRates unit={0},actual={0},batch={0};
    double energy[256],*deposition=NULL,*pop=NULL;FILE *fp=NULL,*coverage=NULL;
    double *batch_pop=NULL,*batch_ne=NULL,*batch_power=NULL;
    int *source=NULL,rc=-1,iteration=-1,rows=0;
    REQUIRE(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],atoi(argv[3]),masses));
    REQUIRE(!cmf_toy_initial_iterate_read(&m,argv[4],argv[5],&iteration));
    const AtomicData *a=&m.atom;int ns=m.plasma.n_shells;
    deposition=calloc((size_t)5*ns,sizeof(*deposition));
    pop=malloc((size_t)m.nlte.n_nlte_levels_total*sizeof(*pop));REQUIRE(deposition&&pop);
    REQUIRE(!read_gamma(argv[8],ns,deposition));
    REQUIRE(!cmf_nonthermal_atomic_read(&raw,argv[6]));
    if(getenv("CMF_TEST_NT_GRID")||getenv("CMF_TEST_NT_SOURCE")){
        rc=grid_comparison(&m,&raw,argv[7],iteration,getenv("CMF_TEST_NT_SOURCE")!=NULL);goto done;
    }
    for(int k=0;k<256;k++)energy[k]=exp((double)k/255*log(2000));
    REQUIRE(!cmf_nonthermal_excitation_build(a,energy,256,20,&exc));
    REQUIRE(!cmf_nonthermal_binding_build_policy(a,&raw,&exc,argv[7],CMF_NT_SIGMA_SOURCE_KINETIC,&bound));
    if(getenv("CMF_TEST_NT_COVERAGE")){
        source=binding_sources(argv[7],&raw,&bound,exc.view.n_channels);REQUIRE(source);
        coverage=fopen("iterate_nonthermal_missing.csv","wx");REQUIRE(coverage);
        REQUIRE(fprintf(coverage,"iteration,shell,Z,charge,source_line,ion_cm3,element_fraction,raw_threshold_eV,base_source_line,max_sigma_ratio,upper_events_per_eV,upper_raw_loss_fraction,status\n")>0);
    }
    printf("TOY_NT_MODEL ion_channels=%d excitation_channels=%d resolved=%d unresolved=%d missing_ions=%d complete_atomic_model=0\n",
           bound.n_ion_channels,exc.view.n_channels,bound.n_resolved_routes,bound.n_unresolved_routes,bound.n_missing_ions);
    NLTEConfig n=m.nlte;n.nlte_level_populations=pop;
    double electron=0,power=ev;PlasmaState p={0};p.n_shells=1;p.n_electron=&electron;
    GammaDeposition g={0};g.n_shells=1;g.generation=1;g.epoch=m.geometry.time_explosion;
    g.provenance=GAMMA_PROVENANCE_NONLOCAL_MC;g.heating_rate=&power;
    int selected[256],nselected=0;REQUIRE(ns<=256);
    for(int s=0;s<ns;s++){
        double velocity=(m.geometry.v_inner[s]+m.geometry.v_outer[s])/2e5;
        if((velocity>=12100&&velocity<=12900)||(coverage&&(s==0||s==20||s==30)))selected[nselected++]=s;
    }
    if(coverage){
        REQUIRE(nselected==9);
        batch_pop=malloc((size_t)n.n_nlte_levels_total*nselected*sizeof(double));
        batch_ne=malloc((size_t)nselected*sizeof(double));batch_power=malloc((size_t)nselected*sizeof(double));
        REQUIRE(batch_pop&&batch_ne&&batch_power);
        for(int b=0;b<nselected;b++){
            int s=selected[b];batch_ne[b]=m.plasma.n_electron[s];batch_power[b]=ev;
            for(int k=0;k<n.n_nlte_levels_total;k++)batch_pop[(size_t)k*nselected+b]=m.nlte.nlte_level_populations[(size_t)k*ns+s];
        }
        NLTEConfig bn=n;bn.nlte_level_populations=batch_pop;
        PlasmaState bp=p;bp.n_shells=nselected;bp.n_electron=batch_ne;
        GammaDeposition bg=g;bg.n_shells=nselected;bg.heating_rate=batch_power;
        struct timespec start,finish;REQUIRE(!clock_gettime(CLOCK_MONOTONIC,&start));
        REQUIRE(!cmf_nonthermal_bound_evaluate(&bound.view,a,&bn,&bp,&bg,&batch));
        REQUIRE(!clock_gettime(CLOCK_MONOTONIC,&finish));
        printf("TOY_NT_BATCH shells=%d channels=%d seconds=%.9g threads=%d\n",nselected,batch.n_transitions,
            (double)(finish.tv_sec-start.tv_sec)+(finish.tv_nsec-start.tv_nsec)*1e-9,
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );
        free(batch_pop);batch_pop=NULL;free(batch_ne);batch_ne=NULL;free(batch_power);batch_power=NULL;
    }
    fp=fopen("iterate_nonthermal.csv","wx");REQUIRE(fp);
    REQUIRE(fprintf(fp,"iteration,shell,Z,charge,velocity_km_s,T_K,ne_cm3,gamma_erg_cm3_s,gamma_MCSE,positive_packets,positron_erg_cm3_s,total_erg_cm3_s,single_events_per_eV,auger_events_per_eV,single_events_cm3_s,auger_events_cm3_s,single_per_ion_s,single_MCSE_cm3_s,heat_fraction,ion_fraction,excitation_fraction\n")>0);
    double max_linear_error=0;
    for(int b=0;b<nselected;b++){
        int s=selected[b];
        double velocity=(m.geometry.v_inner[s]+m.geometry.v_outer[s])/2e5;
        electron=m.plasma.n_electron[s];
        for(int k=0;k<n.n_nlte_levels_total;k++)pop[k]=m.nlte.nlte_level_populations[(size_t)k*ns+s];
        power=ev;REQUIRE(!cmf_nonthermal_bound_evaluate(&bound.view,a,&n,&p,&g,&unit));
        if(coverage){
            REQUIRE(batch.n_transitions==unit.n_transitions);
            REQUIRE(!memcmp(batch.transitions,unit.transitions,(size_t)unit.n_transitions*sizeof(*unit.transitions)));
            for(int k=0;k<unit.n_transitions;k++)REQUIRE(!memcmp(&unit.rate_per_particle[k],&batch.rate_per_particle[(size_t)k*nselected+b],sizeof(double)));
        }
        if(coverage)REQUIRE(!missing_bounds(&m,s,iteration,&raw,&bound,exc.view.n_channels,source,&unit,coverage));
        power=deposition[4*ns+s];REQUIRE(!cmf_nonthermal_bound_evaluate(&bound.view,a,&n,&p,&g,&actual));
        REQUIRE(actual.n_transitions==unit.n_transitions);
        long double single[4]={0},auger[4]={0},ions=0,excitation=0;
        for(int k=0;k<unit.n_transitions;k++){
            const CMFNonthermalTransition *t=&unit.transitions[k];int lo=t->lower_global,electrons;
            double delta,expected=unit.rate_per_particle[k]*(power/ev),got=actual.rate_per_particle[k];
            REQUIRE(!cmf_nonthermal_transition_energy(a,t,&delta,&electrons));
            REQUIRE(memcmp(t,&actual.transitions[k],sizeof(*t))==0);
            REQUIRE(isfinite(expected)&&isfinite(got)&&expected>=0&&got>=0);
            double error=expected==0?(got==0?0:INFINITY):fabs(got/expected-1);
            max_linear_error=fmax(max_linear_error,error);REQUIRE(error<1e-12);
            int fl=n.global_to_nlte_level[lo];REQUIRE(fl>=0&&fl<n.n_nlte_levels_total);
            long double events=(long double)unit.rate_per_particle[k]*pop[fl];
            if(electrons)ions+=events*delta;else excitation+=events*delta;
            int slot=-1,z=a->level_Z[lo],q=a->level_ion[lo];
            if((z==14||z==16)&&(q==1||q==2))slot=(z==16?2:0)+q-1;
            if(slot>=0){if(electrons==1)single[slot]+=events;if(electrons==2)auger[slot]+=events;}
        }
        REQUIRE(ions>=0&&excitation>=0&&ions+excitation<=1+1e-10L);
        for(int slot=0;slot<4;slot++){
            int z=slot<2?14:16,q=slot%2+1,ip=-1;
            for(int k=0;k<a->n_ion_pops;k++)if(a->ion_pop_Z[k]==z&&a->ion_pop_stage[k]==q)ip=k;
            REQUIRE(ip>=0);
            double population=a->ion_number_density[(size_t)ip*ns+s];REQUIRE(population>=0);
            long double event=single[slot]*(power/ev);
            if(population==0){REQUIRE(single[slot]==0&&auger[slot]==0);continue;}
            REQUIRE(fprintf(fp,"%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.0f,%.17g,%.17g,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg\n",
                iteration,s,z,q,velocity,m.plasma.T_e[s],electron,deposition[s],deposition[ns+s],deposition[2*ns+s],deposition[3*ns+s],power,
                single[slot],auger[slot],event,auger[slot]*(power/ev),event/population,
                single[slot]*(deposition[ns+s]/ev),1-ions-excitation,ions,excitation)>0);
            rows++;
        }
        cmf_nonthermal_rates_free(&unit);cmf_nonthermal_rates_free(&actual);
    }
    REQUIRE(rows>0&&!m.nlte.radiation_field.field.generation.computed_generation);
    printf("TOY_NT_ITERATE_DIAGNOSTIC iteration=%d rows=%d max_deposition_linearity_error=%.17g transport_solved=0 NLTE_solved=0 zero_sample_is_not_physical_zero=1\n",iteration,rows,max_linear_error);
    if(coverage)printf("TOY_NT_BATCH_BYTE_EQUAL coefficients=%zu\n",(size_t)nselected*batch.n_transitions);
    rc=0;
done:
    if(fp&&fclose(fp))rc=-1;
    if(coverage&&fclose(coverage))rc=-1;
    cmf_nonthermal_rates_free(&unit);cmf_nonthermal_rates_free(&actual);cmf_nonthermal_rates_free(&batch);
    cmf_nonthermal_binding_free(&bound);cmf_nonthermal_excitation_free(&exc);cmf_nonthermal_atomic_free(&raw);
    cmf_toy_model_free(&m);free(deposition);free(pop);free(source);
    free(batch_pop);free(batch_ne);free(batch_power);return rc?1:0;
}
