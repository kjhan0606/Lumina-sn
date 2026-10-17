/* 실제 시간 수송/R6/R7/온도·전자·인구 공동 실행. 합성2셸·3준위 원자이며
 * SN 초기조건 또는 CMFGEN 분포 시험이 아니다. matrix-cache 모드의 wrapper는
 * 실제 함수를 그대로 호출하고 재사용 횟수만 읽는다. 물리 풀이를 대체하지 않는다. */
#include "cmf_time_epoch.h"
#include "cmf_toy_model.h"
#include "line_jbar.h"
#include "cmf_nonthermal_provider.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef CMF_MATERIAL_GUESS_PRODUCER_TEST
/* 실제 원자적 저장 callback을 사용한다. 사용하지 않는 CLI 함수는 링크 GC한다. */
#include "cmf_toy_run.c"
#endif
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TIME_COUPLED_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static double Bnu(double nu,double T){return 2*H_PLANCK*nu*nu*nu/
    (C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(H_PLANCK*nu/(K_BOLTZMANN*T));}
static int nt_calls;
#ifdef CMF_MATRIX_CACHE_INTEGRATION
static size_t matrix_cache_hits;
extern PopulationStatus __real_nlte_element_initial_steady_solve_cached(NLTEConfig*,AtomicData*,
    PlasmaState*,OpacityState*,int,int,double,GammaDeposition*,struct NLTEElementBaseCache*);
PopulationStatus __wrap_nlte_element_initial_steady_solve_cached(NLTEConfig *n,AtomicData *a,
    PlasmaState *p,OpacityState *o,int e,int s,double epoch,GammaDeposition *g,
    struct NLTEElementBaseCache *cache){
    size_t before=nlte_element_base_cache_hits(cache);
    PopulationStatus result=__real_nlte_element_initial_steady_solve_cached(n,a,p,o,e,s,epoch,g,cache);
    matrix_cache_hits+=nlte_element_base_cache_hits(cache)-before;return result;
}
#endif
static int saved_iterations;
static int material_guesses;
static CMFToyModel save_view;
static Geometry *save_geo;
static OpacityState *save_opac;
static PlasmaState *save_plasma;
static NLTEConfig *save_nlte;
static AtomicData *save_atom;
static const char *iterate_id="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
static void sync_save_view(void){
    save_view.prepared=1;save_view.geometry=*save_geo;save_view.opacity=*save_opac;
    save_view.plasma=*save_plasma;save_view.nlte=*save_nlte;save_view.atom=*save_atom;
}
static int write_test_material(const char *path,const PlasmaState *p,
                               const AtomicData *a,const NLTEConfig *n){
    FILE *f=fopen(path,"wx");if(!f)return -1;
    int bad=fprintf(f,"field,index,shell,value\n")<0;
    for(int s=0;s<p->n_shells;s++){
        bad|=fprintf(f,"T,0,%d,%.17g\nne,0,%d,%.17g\n",s,p->T_e[s],s,p->n_electron[s])<0;
        for(int i=0;i<a->n_ion_pops;i++)
            bad|=fprintf(f,"ion,%d,%d,%.17g\n",i,s,a->ion_number_density[(size_t)i*p->n_shells+s])<0;
        for(int i=0;i<n->n_nlte_levels_total;i++)
            bad|=fprintf(f,"level,%d,%d,%.17g\n",i,s,n->nlte_level_populations[(size_t)i*p->n_shells+s])<0;
    }
    return fclose(f)||bad?-1:0;
}
static int save_iteration(void *context,int iteration){
    CMFGENState *c=context;
    if(c->common_obs_ready||iteration!=saved_iterations)return -1;
    saved_iterations++;
#ifdef CMF_MATERIAL_GUESS_PRODUCER_TEST
    sync_save_view();ToyInitialSave producer={&save_view,iterate_id,0};
    if(toy_initial_save(&producer,iteration))return -1;
#endif
    if(iteration==3&&getenv("CMF_TEST_ITERATE")&&!getenv("CMF_TEST_ITERATE_INPUT")){
        sync_save_view();
        return cmf_toy_initial_iterate_write(&save_view,"initial_iterate_test.bin",iterate_id,iteration);
    }
    return 0;
}
static int save_material_guess(void *context,int completed){
    CMFGENState *c=context;
    if(c->common_obs_ready||completed+1!=saved_iterations||completed!=material_guesses||
       save_plasma->physical_history||!save_plasma->initial_steady_nlte)return -1;
    material_guesses++;
    sync_save_view();
#ifdef CMF_MATERIAL_GUESS_PRODUCER_TEST
    ToyInitialSave producer={&save_view,iterate_id,0};
    if(toy_material_guess_save(&producer,completed))return -1;
#endif
    if(completed==3&&!getenv("CMF_TEST_ITERATE_INPUT")){
        if(cmf_toy_initial_iterate_write(&save_view,"initial_material_guess_test.bin",iterate_id,completed))return -1;
        if(getenv("CMF_TEST_MATERIAL_STOP")){
            puts("INITIAL_MATERIAL_INTERRUPTION completed_iterations=3 pending_material_update=1 physical_history=0 converged=0");
            fflush(NULL);_Exit(86); /* 저장 직후 강제 종료: 이후 수송/성공 출력 없음. */
        }
    }
    return 0;
}
static double nt_max_rate;
static int actual_nt(const void *context,const AtomicData *a,const NLTEConfig *n,
                     const PlasmaState *p,const GammaDeposition *g,CMFNonthermalRates *r){
    int status=cmf_nonthermal_bound_evaluate(context,a,n,p,g,r);
    if(!status){
        nt_calls++;
        for(size_t k=0;k<(size_t)r->n_transitions*r->n_shells;k++)
            nt_max_rate=fmax(nt_max_rate,r->rate_per_particle[k]);
    }
    return status;
}
int main(void){
    enum{NS=2,NI=2,NL=3,NB=NLTE_N_FREQ_BINS};
    setenv("LUMINA_PURE_CMFGEN","1",1);setenv("LUMINA_CMF_COMMON_FIELD","1",1);
    setenv("LUMINA_CMF_BF_MILNE","2",1);setenv("LUMINA_CMF_FINE_VDOP","5000000",1);
    setenv("LUMINA_CMF_FINE_PPD","1.3",1);setenv("LUMINA_CMF_FINE_ALI","32",1);
    setenv("LUMINA_CMF_FINE_TOL","1e-9",1);setenv("LUMINA_CMF_FINE_PHOTOION","1",1);
    setenv("LUMINA_NLTE_NE_CLOSURE","1",1);
    int iz[NI]={1,1},st[NI]={0,1},off[NI+1]={0,2,3},map[NL]={0,1,2};
    int lz[NL]={1,1,1},ls[NL]={0,0,1},weight[NL]={2,4,1},num[NL]={0,1,0},sl[NL]={0,1,0};
    int meta[NL]={1,0,1},element=1,eloff[2]={0,2},ionstage=0;
    double mass=1,abund[NS]={1,1},chi=13.6,E[NL]={0,2,0};
    double T[NS]={1e4,1e4},rho[NS]={1e8*AMU,1e8*AMU},ne[NS]={9e7,9e7};
    double pops[NL*NS],ions[NI*NS]={1e7,1e7,9e7,9e7},part[NI*NS],within[NL*NS];
    double ratio=2*exp(-2*EV_TO_ERG/(K_BOLTZMANN*T[0]));
    for(int s=0;s<NS;s++){pops[s]=1e7/(1+ratio);pops[NS+s]=1e7-pops[s];pops[2*NS+s]=9e7;}
    for(int i=0;i<NL*NS;i++)within[i]=1;
    double nu=2*EV_TO_ERG/H_PLANCK,lam=C_SPEED_OF_LIGHT/nu,f=1e-8;
    double A=8*M_PI_VAL*SOBOLEV_COEFF*.5*f/(lam*lam),Bl=4*M_PI_VAL*SOBOLEV_COEFF*f/(H_PLANCK*nu);
    double Bu=.5*Bl;
    int lineZ=1,lineion=0,lower=0,upper=1,linemap=0;uint8_t mask=1;
    double tau[NS]={0},source[NS]={0};A208Validity tv[NS],sv[NS];
    AtomicData a={0};a.n_ion_pops=NI;a.n_levels=NL;a.n_lines=1;
    a.ion_pop_Z=iz;a.ion_pop_stage=st;a.level_offset=off;a.level_Z=lz;a.level_ion=ls;
    a.level_g=weight;a.level_num=num;a.level_super=sl;a.level_metastable=meta;a.level_energy_eV=E;
    a.ion_number_density=ions;a.partition_functions=part;a.population_committed_generation=1;
    a.n_elements=1;a.element_Z=&element;a.elem_ion_offset=eloff;a.element_mass_amu=&mass;a.abundances=abund;
    a.n_ionization=1;a.ioniz_Z=&element;a.ioniz_ion=&ionstage;a.ioniz_energy_eV=&chi;
    a.line_atomic_number=&lineZ;a.line_ion_number=&lineion;a.line_level_lower=&lower;a.line_level_upper=&upper;
    a.line_nu=&nu;a.line_wavelength_cm=&lam;a.line_f_lu=&f;a.line_A_ul=&A;a.line_B_lu=&Bl;a.line_B_ul=&Bu;
    int coff[NL+1]={0,1,2,2},target[2]={2,2},uion[2]={1,1};double gu[2]={1,1};
    double threshold[2]={chi*EV_TO_ERG/H_PLANCK,(chi-2)*EV_TO_ERG/H_PLANCK};
    double sigma[2*NB],dlog=log(NLTE_NU_MAX/NLTE_NU_MIN)/NB;
    for(int r=0;r<2;r++)for(int b=0;b<NB;b++)sigma[r*NB+b]=
        NLTE_NU_MIN*exp((b+1)*dlog)>threshold[r]?1e-18:0;
    BfChannels channels={NL,2,NB,NLTE_NU_MIN,NLTE_NU_MAX,coff,target,threshold,sigma,2,uion,gu};a.bf_channels=&channels;
    NLTEConfig n={0};n.enabled=1;n.n_nlte_ions=NI;n.n_nlte_levels_total=n.n_super_total=NL;
    if(getenv("CMF_TEST_INITIAL_MATRIX_CACHE")){
        n.channel_owned_recombination=1;n.exclude_empirical_heavy_ce=1;
    }
    for(int i=0;i<NI;i++){n.nlte_Z[i]=iz[i];n.nlte_ion[i]=st[i];}
    for(int i=0;i<=NI;i++)n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=off[i];
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.nlte_level_populations=pops;n.within_sl_frac=within;n.nlte_line_map=&linemap;n.population_committed_generation=1;
    int line_global[2]={0,1};n.nlte_line_global=line_global;n.nlte_line_global_count=1;
    int ion_line_offset[3]={0,1,1},ion_line_id[1]={0};
    n.nlte_ion_line_offsets=ion_line_offset;n.nlte_ion_lines=ion_line_id;n.nlte_line_index_count=1;
    n.n_freq_bins=NB;n.nu_min=NLTE_NU_MIN;n.nu_max=NLTE_NU_MAX;n.d_log_nu=dlog;
    n.J_nu=calloc((size_t)NS*NB,sizeof(double));CHECK(n.J_nu);
    PlasmaState p={0};p.n_shells=NS;p.T_e=T;p.rho=rho;p.n_electron=ne;p.T_e_generation=1;
    p.te_publication.required_te_generation=p.te_publication.committed_te_generation=1;
    CHECK(population_te_manifest_sha256(T,NS,p.te_publication.te_manifest_sha256)==POP_OK);
    PopulationAtomicView av={0};av.n_ions=NI;av.n_levels=NL;av.level_offset=off;
    av.energy_eV=E;av.g=weight;av.level_Z=lz;av.level_ion=ls;
    CHECK(population_partition_build(&av,T,NS,1,1,part,&a.partition_stamp)==POP_OK);
    CHECK(!nlte_precompute_within_sl_frac(&n,&a,&p,NS));
    OpacityState o={0};o.n_shells=NS;o.n_lines=1;o.line_list_nu=&nu;o.tau_sobolev=tau;o.line_source_S=source;
    o.tau_validity=tv;o.line_source_validity=sv;o.electron_density=ne;o.t_electrons=T;
    o.tau_required_generation=o.tau_computed_generation=1;
    LineJbarQSet q={0};LineJbarESet e={0};
    CHECK(!line_jbar_qset_build(&q,1,&nu,&linemap,&mask)&&!line_jbar_eset_build(&e,1,&nu));
    n.line_qset=&q;n.line_eset=&e;CHECK(!radiation_field_owner_init(&n.radiation_field,NS));
    CHECK(!cmfgen_bind_common_profile(&n,5e6));
    double oldtime=1e6,newtime=oldtime+100,ve[NS+1]={3e8,4e8,5e8},ri[NS],ro[NS];
    if(getenv("CMF_TEST_CENTER"))ve[0]=0;
    for(int s=0;s<NS;s++){ri[s]=ve[s]*newtime;ro[s]=ve[s+1]*newtime;}
    Geometry g={0};g.n_shells=NS;g.time_explosion=newtime;g.r_inner=ri;g.r_outer=ro;g.v_inner=ve;g.v_outer=ve+1;
    CMFGENState c={0};CHECK(!cmfgen_init(&c,&g));
    if(getenv("CMF_TEST_INITIAL")){
        /* 같은 실제3준위 원자로 정상 초기 NLTE/복사장 공동해를 실행한다.
         * 과거를 만들거나 시간항을 큰 dt로 줄이지 않는다. */
        p.initial_steady_nlte=1;
        unsetenv("LUMINA_NLTE_NE_CLOSURE"); /* 초기화에서는 전하 폐합이 선택 사항이 아니다. */
        /* 연결 시험의 약한 합성 선이다. 실제50km/s 선의 해상도 검증이 아니다. */
        setenv("LUMINA_CMF_FINE_PPD","0.13",1);
        /* 내부/외부 전리에너지원이 없는 순수 재결합 fixture 대신 고정 T의
         * 열적 충돌 전리와 그 상세균형 역과정도 실제로 푼다. */
        setenv("LUMINA_NLTE_COLL_ION","1",1);
        FILE *profile=fopen("initial_T.csv","wx");CHECK(profile);
        CHECK(fprintf(profile,"# shell T_e\n0 10000\n1 10000\n")>0&&!fclose(profile));
        profile=fopen("wrong_initial_T.csv","wx");CHECK(profile);
        CHECK(fprintf(profile,"# shell T_e\n0 10001\n1 10000\n")>0&&!fclose(profile));
        nlte_update_tau_sobolev(&n,&a,&o,newtime,NS);
        GammaDeposition gamma={0};gamma_deposition_init(&gamma,NS);double heating[NS]={0};
        int nonthermal=getenv("CMF_TEST_INITIAL_NT")!=NULL;
        double nt_energy[64],nt_sigma[128],abcd[4]={1,.5,.2,.1};
        for(int i=0;i<64;i++){
            nt_energy[i]=exp((double)i/63*log(2000));
            nt_sigma[i]=cmf_nonthermal_bethe(nt_energy[i],2,.1,0);
            nt_sigma[64+i]=cmf_nonthermal_arnaud(nt_energy[i],13.6,abcd);
        }
        CMFNonthermalTransition nt_transitions[2]={{0,1,-1},{0,2,-1}};
        CMFNonthermalRoute nt_channels[2]={{CMF_NT_EXCITATION,2,0,nt_sigma,0},
                                         {CMF_NT_IONIZATION,13.6,0,nt_sigma+64,0}};
        CMFNonthermalBoundModel nt_model={NL,NI,64,2,nt_energy,nt_transitions,nt_channels};
        CMFNonthermalProvider provider={&nt_model,actual_nt};
        if(nonthermal){heating[0]=1e-10;heating[1]=2e-10;}
        CHECK(!gamma_deposition_publish(&gamma,nonthermal?GAMMA_PROVENANCE_NONLOCAL_MC:GAMMA_PROVENANCE_EXTERNAL_FILE,newtime,NULL,NULL,NULL,heating));
        if(nonthermal)gamma.nonthermal_provider=&provider;
        BFOpacity bf={0};CHECK(!bf_opacity_init_checked(&bf,NS));
        c.common_obs_n=5;c.common_obs_nu=malloc(5*sizeof(double));CHECK(c.common_obs_nu);
        for(int i=0;i<5;i++)c.common_obs_nu[i]=nu*(.99+.005*i);
        CMFInitialNLTEReport report;
        CMFPhysicalTimeState initial={0},evolved={0};
        setenv("LUMINA_FIXED_TE_PROFILE","wrong_initial_T.csv",1);
        CHECK(cmfgen_solve_initial_nlte(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-7,&initial,&report)==CMF_EPOCH_INVALID);
        CHECK(!initial.radiation_storage&&!initial.material.storage);
        CHECK(T[0]==1e4&&T[1]==1e4&&a.population_committed_generation==1);
        setenv("LUMINA_FIXED_TE_PROFILE","initial_T.csv",1);
        save_geo=&g;save_opac=&o;save_plasma=&p;save_nlte=&n;save_atom=&a;
        if(getenv("CMF_TEST_ITERATE_INPUT")){
            int completed=-1;save_view.prepared=1;save_view.geometry=g;save_view.opacity=o;
            save_view.plasma=p;save_view.nlte=n;save_view.atom=a;
            const char *guess=getenv("CMF_TEST_INITIAL_GUESS");
            const char *run_id=guess?"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef":iterate_id;
            CHECK(!cmf_toy_initial_iterate_start(&save_view,getenv("CMF_TEST_ITERATE_INPUT"),run_id,guess?iterate_id:NULL,&completed));
            CHECK(completed==(guess?0:3));
            CHECK(!save_view.plasma.physical_history&&!save_view.nlte.radiation_field.field.generation.computed_generation);
            g=save_view.geometry;o=save_view.opacity;p=save_view.plasma;n=save_view.nlte;a=save_view.atom;
        }
        CMFTimeEpochStatus status=cmfgen_solve_initial_nlte_checkpointed(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-7,&initial,&report,
            save_iteration,getenv("CMF_TEST_MATERIAL_GUESS")?save_material_guess:NULL,&c);
        fprintf(stderr,"INITIAL_COUPLED_RESULT status=%d iterations=%d ne=%.17g/%.17g T=%.17g/%.17g\n",
                status,report.iterations,ne[0],ne[1],T[0],T[1]);
        CHECK(status==CMF_EPOCH_OK&&report.converged&&c.common_obs_ready&&report.iterations>1);
        CHECK(saved_iterations==report.iterations+1);
        if(getenv("CMF_TEST_MATERIAL_GUESS")){
            CHECK(material_guesses==report.iterations);
            printf("INITIAL_MATERIAL_GUESS_CALLBACK_PASS calls=%d completed_count_unchanged=1\n",material_guesses);
        }
        printf("INITIAL_SAVE_CALLBACK_PASS calls=%d physical_history_created_by_callback=0\n",saved_iterations);
#ifdef CMF_MATRIX_CACHE_INTEGRATION
        CHECK(matrix_cache_hits>0);
        printf("INITIAL_MATRIX_CACHE_USED hits=%zu actual_SF_provider=1 actual_checked_field=1 synthetic_atom=1\n",matrix_cache_hits);
#endif
        CHECK(T[0]==1e4&&T[1]==1e4&&!p.physical_history&&!c.time_trial);
        CHECK(initial.material.energy.epoch_s==newtime&&initial.radiation.epoch_s==newtime);
        CHECK(initial.radiation.J!=o.jnu_fine&&initial.radiation.H!=c.initial_face_H);
        CHECK(initial.radiation.n_bins==o.n_fine&&c.initial_face_J_residual<1e-7);
        CHECK(!memcmp(initial.radiation.J,o.jnu_fine,(size_t)NS*o.n_fine*sizeof(double)));
        CHECK(!memcmp(initial.radiation.H,c.initial_face_H,(size_t)(NS+1)*o.n_fine*sizeof(double)));
        for(int b=0;b<o.n_fine;b++)CHECK(initial.radiation.H[b]==0);
        double up,down,ci,three;
        CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&up,&down,&ci,&three)&&ci>0&&three>0);
        /* 동일한 실제 게시 J/물질에서 cache 유무의 행렬/RHS는 byte-identical이다. */
        double cached_A[9]={0},cached_b[3]={0},searched_A[9]={0},searched_b[3]={0};
        CMFNonthermalRates matrix_nt={0};
        if(nonthermal){
            CHECK(!actual_nt(&nt_model,&a,&n,&p,&gamma,&matrix_nt));
            gamma.nonthermal_rates=&matrix_nt;
        }
        CHECK(!nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,newtime,cached_A,cached_b,3,&gamma,NULL,-1));
        n.nlte_line_global=NULL;n.nlte_ion_lines=NULL;
        CHECK(!nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,newtime,searched_A,searched_b,3,&gamma,NULL,-1));
        n.nlte_line_global=line_global;n.nlte_ion_lines=ion_line_id;
        CHECK(!memcmp(cached_A,searched_A,sizeof cached_A)&&!memcmp(cached_b,searched_b,sizeof cached_b));
        gamma.nonthermal_rates=NULL;cmf_nonthermal_rates_free(&matrix_nt);
        puts("LINE_GLOBAL_CACHE_MATRIX_PARITY_PASS exact_matrix_and_rhs=1 ion_index=1");
        for(int s=0;s<NS;s++){
            CHECK(fabs(ne[s]/ions[NS+s]-1)<1e-8);
            CHECK(fabs((ions[s]+ions[NS+s])/1e8-1)<1e-12);
            CHECK(fabs((pops[s]+pops[NS+s])/ions[s]-1)<1e-12);
        }
        CHECK(!cmfgen_write_common_observer(&c,"initial_spectrum.csv","initial_spectrum.json",
            n.radfield_view.generation,a.population_committed_generation,p.T_e_generation,o.tau_computed_generation));
        CHECK(!write_test_material("initial_material.csv",&p,&a,&n));
        fprintf(stderr,"INITIAL_HISTORY epoch=%.17g face_J_residual=%.17g bins=%d center_H=0\n",
            initial.radiation.epoch_s,c.initial_face_J_residual,initial.radiation.n_bins);
        CHECK(n.bf_rate_cache==NULL); /* 요청 수명의 차용 cache를 public에 남기지 않는다. */
        /* 실제 수렴해를 파일에서 복원한 뒤 다음 시간해가 그 J/H를 소비한다. */
        CMFPhysicalTimeState restored={0};
        CHECK(!cmf_physical_time_state_write(&initial,"initial_state.bin"));
        CHECK(cmf_physical_time_state_write(&initial,"initial_state.bin")!=0);
        abund[0]=.5;abund[1]=.8; /* 현재 조성과 과거의 조성은 다를 수 있다. */
        CHECK(!cmf_physical_time_state_read(&restored,"initial_state.bin",&n,&a,&g));
        CHECK(abund[0]==.5&&abund[1]==.8);abund[0]=abund[1]=1;
        CHECK(!memcmp(initial.material.storage,restored.material.storage,(size_t)(5+NI+NL)*NS*sizeof(double)));
        CHECK(!memcmp(initial.radiation_storage,restored.radiation_storage,
            ((size_t)NS+1+(size_t)(2*NS+1)*initial.radiation.n_bins)*sizeof(double)));
        double *preserved=restored.material.storage;
        st[0]=1;CHECK(cmf_physical_time_state_read(&restored,"initial_state.bin",&n,&a,&g)!=0);st[0]=0;
        CHECK(restored.material.storage==preserved);
        FILE *bad=fopen("bad_history.bin","wbx");CHECK(bad);
        CHECK(fwrite("CMFINIT1",1,8,bad)==8&&!fclose(bad));
        CHECK(cmf_physical_time_state_read(&restored,"bad_history.bin",&n,&a,&g)!=0&&restored.material.storage==preserved);
        CHECK(!cmf_physical_time_state_write(&initial,"bad_energy.bin"));
        bad=fopen("bad_energy.bin","r+b");CHECK(bad);
        double wrong_energy=2*initial.material.energy.internal_energy_atom_erg[0];
        CHECK(!fseek(bad,96+4*(NL+2*NI)+8*4*NS,SEEK_SET));
        CHECK(fwrite(&wrong_energy,8,1,bad)==1&&!fclose(bad));
        CHECK(cmf_physical_time_state_read(&restored,"bad_energy.bin",&n,&a,&g)!=0&&restored.material.storage==preserved);
        cmf_physical_time_state_free(&initial);initial=restored;memset(&restored,0,sizeof(restored));
        fprintf(stderr,"INITIAL_CHECKPOINT_ROUNDTRIP exact_material=1 exact_J_H=1 overwrite_rejected=1 invalid_history_preserved=1\n");
        /* 방금 구한 실제 초기해에서 다음 물리 시점으로 진행한다. 임의 J=B나
         * 면 H=0으로 과거를 교체하지 않고, cs를 해제해 깊은복사도 확인한다. */
        double *old_radiation=malloc((size_t)(2*NS+1)*o.n_fine*sizeof(double));CHECK(old_radiation);
        memcpy(old_radiation,initial.radiation.J,(size_t)NS*o.n_fine*sizeof(double));
        memcpy(old_radiation+(size_t)NS*o.n_fine,initial.radiation.H,(size_t)(NS+1)*o.n_fine*sizeof(double));
        double initial_material[(5+NI+NL)*NS];memcpy(initial_material,initial.material.storage,sizeof initial_material);
        double evolved_time=0,dilution=0,step=100;
        p.initial_steady_nlte=0;p.physical_history=&initial.material;
        unsetenv("LUMINA_FIXED_TE_PROFILE");setenv("LUMINA_NLTE_NE_CLOSURE","1",1);
        double *inner=calloc((size_t)o.n_fine,sizeof(double));CHECK(inner);
        CMFTimeRadiationTrial trial={0};
        CMFTimeEpochReport time_report;
        /* 복사 첫 시도에서만 dt 축소를 검사한다. 물질이 게시된 뒤의 실패를
         * 불완전한 rollback으로 재시도하지 않는다. 모든 시도는 같은 과거다. */
        for(int attempt=0;attempt<5;attempt++,step*=.1){
            evolved_time=newtime+step;dilution=pow(newtime/evolved_time,3);
            cmfgen_free(&c);cmf_time_radiation_trial_free(&trial);g.time_explosion=evolved_time;
            for(int s=0;s<NS;s++){
                ri[s]=ve[s]*evolved_time;ro[s]=ve[s+1]*evolved_time;
                rho[s]=initial.material.energy.density_g_cm3[s]*dilution;
                ne[s]=initial.material.energy.n_electron_cm3[s]*dilution;
            }
            for(int i=0;i<NI*NS;i++)ions[i]=initial.material.ion_density[i]*dilution;
            for(int i=0;i<NL*NS;i++)pops[i]=initial.material.level_density[i]*dilution;
            CHECK(!cmfgen_init(&c,&g));nlte_update_tau_sobolev(&n,&a,&o,evolved_time,NS);
            CHECK(!gamma_deposition_publish(&gamma,nonthermal?GAMMA_PROVENANCE_NONLOCAL_MC:GAMMA_PROVENANCE_EXTERNAL_FILE,evolved_time,NULL,NULL,NULL,heating));
            if(nonthermal)gamma.nonthermal_provider=&provider;
            trial.old=&initial.radiation;trial.epoch_s=evolved_time;trial.inner_H=inner;c.time_trial=&trial;
            c.common_obs_n=5;c.common_obs_nu=malloc(5*sizeof(double));CHECK(c.common_obs_nu);
            for(int i=0;i<5;i++)c.common_obs_nu[i]=nu*(.99+.005*i);
            status=cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,32,1e-7,&evolved,&time_report);
            fprintf(stderr,"INITIAL_TIME_ATTEMPT dt=%.17g status=%d field_status=%d iterations=%d\n",
                step,status,trial.report.status,time_report.iterations);
            if(status==CMF_EPOCH_OK)break;
            CHECK(status==CMF_EPOCH_RADIATION&&time_report.iterations==0&&
                (trial.report.status==CMF_TIME_NEGATIVE||trial.report.status==CMF_TIME_STEP_TOO_LARGE));
            CHECK(T[0]==10000&&T[1]==10000&&a.population_committed_generation==initial.material.population_generation);
        }
        fprintf(stderr,"INITIAL_TIME_BRIDGE_RESULT status=%d iterations=%d T=%.17g/%.17g ne=%.17g/%.17g\n",
            status,time_report.iterations,T[0],T[1],ne[0],ne[1]);
        CHECK(status==CMF_EPOCH_OK&&time_report.converged&&evolved.radiation.epoch_s==evolved_time);
        CHECK(T[0]!=10000&&T[1]!=10000);
        CHECK(!memcmp(initial.material.storage,initial_material,sizeof initial_material));
        CHECK(!memcmp(initial.radiation.J,old_radiation,(size_t)NS*o.n_fine*sizeof(double)));
        CHECK(!memcmp(initial.radiation.H,old_radiation+(size_t)NS*o.n_fine,(size_t)(NS+1)*o.n_fine*sizeof(double)));
        for(int s=0;s<NS;s++){
            CHECK(fabs(ne[s]/ions[NS+s]-1)<1e-8);
            CHECK(fabs((ions[s]+ions[NS+s])/(1e8*dilution)-1)<1e-12);
            CHECK(p.te_publication.ledger[s].adiabatic_model==A210_MATERIAL_TIME_HISTORY);
            const A210TermLedger *ledger=&p.te_publication.ledger[s];
            CHECK(a210_energy_balance(ledger)<1e-12);
            fprintf(stderr,"INITIAL_TIME_ENERGY shell=%d residual=%.17g time_backward_error=%.17g net_heat_balance=%.17g time_norm=%.17g\n",
                s,ledger->residual,a210_energy_balance(ledger),ledger->e_balance,ledger->material_time_norm);
        }
        CHECK(!cmfgen_write_common_observer(&c,"initial_evolved_spectrum.csv","initial_evolved_spectrum.json",
            n.radfield_view.generation,a.population_committed_generation,p.T_e_generation,o.tau_computed_generation));
        CHECK(!write_test_material("initial_evolved_material.csv",&p,&a,&n));
        if(nonthermal){
            CHECK(nt_calls>report.iterations+time_report.iterations&&nt_max_rate>0&&isfinite(nt_max_rate));
            CHECK(gamma.nonthermal_rates==NULL);
            printf("CMF_INITIAL_NONTHERMAL_PASS provider=ACTUAL_SPENCER_FANO calls=%d maximum_rate=%.17g refreshed_material=1 synthetic_atom=1 physical_gamma_transport=0 SN_comparison=0\n",nt_calls,nt_max_rate);
        }
        puts("CMF_INITIAL_COUPLED_SELFTEST_PASS actual_radiation=1 actual_R7=1 initial_fixed_T=1 evolved_free_T=1 synthetic_atom=1 time_history=ACTUAL_INITIAL_FACE_H SN_comparison=0");
        free(inner);free(old_radiation);cmf_time_radiation_trial_free(&trial);
        cmf_physical_time_state_free(&initial);cmf_physical_time_state_free(&evolved);
        cmfgen_fine_set_bf_atom(NULL,NULL,NULL,NULL);bf_set_nlte_pops(NULL);
        cmfgen_free(&c);bf_opacity_free(&bf);gamma_deposition_free(&gamma);
        a208_publication_free(&o.cpu_opacity);a209_publication_free(&o.cpu_emissivity);
        a210_publication_free(&p.te_publication);radiation_field_owner_free(&n.radiation_field);
        line_jbar_qset_free(&q);line_jbar_qset_free(&e);
        free(n.J_nu);free(o.jbar_line_det);free(o.jbar_line_det_error_upper);free(o.jnu_fine);free(o.nu_fine);
        return 0;
    }
    double fine_dlog=(5e6/C_SPEED_OF_LIGHT)/1.3;
    double fine_min=NLTE_NU_MIN*exp(-.5*fine_dlog);
    int fine_bins=(int)ceil(log(LUMINA_RADFIELD_NU_MAX_HZ*exp(.5*fine_dlog)/fine_min)/fine_dlog);
    double *oldJ=malloc((size_t)NS*fine_bins*sizeof(double)),*oldH=calloc((size_t)(NS+1)*fine_bins,sizeof(double));
    double *inner=calloc((size_t)fine_bins,sizeof(double));CHECK(oldJ&&oldH&&inner);
    for(int s=0;s<NS;s++)for(int b=0;b<fine_bins;b++)oldJ[(size_t)s*fine_bins+b]=Bnu(fine_min*exp((b+.5)*fine_dlog),T[s]);
    CMFTimeRadiationHistory rh={NS,fine_bins,oldtime,fine_min,fine_dlog,ve,oldJ,oldH};
    CMFPhysicalTimeState past={0},next={0};CHECK(!cmf_physical_time_state_capture(&past,&n,&a,&p,&rh));
    double past_material[(5+NI+NL)*NS];memcpy(past_material,past.material.storage,sizeof past_material);
    p.physical_history=&past.material;
    double expansion=pow(oldtime/newtime,3);
    for(int s=0;s<NS;s++){rho[s]*=expansion;ne[s]*=expansion;}
    for(int i=0;i<NI*NS;i++)ions[i]*=expansion;
    for(int i=0;i<NL*NS;i++)pops[i]*=expansion;
    nlte_update_tau_sobolev(&n,&a,&o,newtime,NS);
    GammaDeposition gamma={0};gamma_deposition_init(&gamma,NS);double heating[NS]={0};
    CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,newtime,NULL,NULL,NULL,heating));
    BFOpacity bf={0};CHECK(!bf_opacity_init_checked(&bf,NS));
    CMFTimeRadiationTrial trial={0};trial.old=&past.radiation;trial.epoch_s=newtime;trial.inner_H=inner;c.time_trial=&trial;
    /* 합성 선 주변의 소형 관측 격자를 사용한다. */
    c.common_obs_n=5;c.common_obs_nu=malloc(5*sizeof(double));CHECK(c.common_obs_nu);
    for(int i=0;i<5;i++)c.common_obs_nu[i]=nu*(.99+.005*i);
    CMFTimeEpochReport rep;
    CMFTimeEpochStatus status=cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-7,&next,&rep);
    fprintf(stderr,"TIME_COUPLED_RESULT status=%d iterations=%d T=%.17g/%.17g ne=%.17g/%.17g\n",status,rep.iterations,T[0],T[1],ne[0],ne[1]);
    CHECK(status==CMF_EPOCH_OK&&rep.converged&&c.common_obs_ready);
    fprintf(stderr,"TIME_COUPLED_BOUNDARY inner_velocity=%.17g inner_H=%.17g\n",ve[0],trial.H[0]);
    if(ve[0]==0)for(int b=0;b<fine_bins;b++)CHECK(trial.H[b]==0);
    CHECK(T[0]!=1e4&&ne[0]!=9e7*expansion&&next.material.energy.epoch_s==newtime);
    CHECK(!memcmp(past.radiation.J,oldJ,(size_t)NS*fine_bins*sizeof(double)));
    CHECK(!memcmp(past_material,past.material.storage,sizeof past_material));
    for(int s=0;s<NS;s++){
        const A210TermLedger *ledger=&p.te_publication.ledger[s];
        CHECK(ledger->adiabatic_model==A210_MATERIAL_TIME_HISTORY);
        CHECK(fabs((ions[s]+ions[NS+s])/(1e8*expansion)-1)<1e-12);
        CHECK(fabs(ne[s]/ions[NS+s]-1)<1e-8);
        CHECK(fabs((pops[s]+pops[NS+s])/ions[s]-1)<1e-12);
        CHECK(pops[2*NS+s]==ions[NS+s]);
        CHECK(a210_energy_balance(ledger)<1e-12);
        fprintf(stderr,"TIME_COUPLED_CLOSURE shell=%d time_backward_error=%.17g net_heat_balance=%.17g residual=%.17g charge_relative=%.17g neutral_fraction=%.17g\n",
            s,a210_energy_balance(ledger),ledger->e_balance,ledger->residual,ne[s]/ions[NS+s]-1,ions[s]/(ions[s]+ions[NS+s]));
    }
    CHECK(!cmfgen_write_common_observer(&c,"coupled_spectrum.csv","coupled_spectrum.json",
        n.radfield_view.generation,a.population_committed_generation,p.T_e_generation,o.tau_computed_generation));
    puts("CMF_TIME_COUPLED_SELFTEST_PASS actual_radiation=1 actual_R7=1 synthetic_atom=1 SN_comparison=0");
    cmfgen_fine_set_bf_atom(NULL,NULL,NULL,NULL);bf_set_nlte_pops(NULL);
    cmf_time_radiation_trial_free(&trial);cmf_physical_time_state_free(&past);cmf_physical_time_state_free(&next);
    cmfgen_free(&c);bf_opacity_free(&bf);gamma_deposition_free(&gamma);
    a208_publication_free(&o.cpu_opacity);a209_publication_free(&o.cpu_emissivity);
    a210_publication_free(&p.te_publication);radiation_field_owner_free(&n.radiation_field);
    line_jbar_qset_free(&q);line_jbar_qset_free(&e);
    free(n.J_nu);free(o.jbar_line_det);free(o.jbar_line_det_error_upper);
    free(o.jnu_fine);free(o.nu_fine);free(oldJ);free(oldH);free(inner);
    return 0;
}
