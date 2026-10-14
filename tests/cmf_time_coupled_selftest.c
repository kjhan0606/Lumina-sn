/* 실제 시간 수송/R6/R7/온도·전자·인구 공동 실행. 합성2셸·3준위 원자이며
 * SN 초기조건 또는 CMFGEN 분포 시험이 아니다. 생산 함수를 wrap하지 않는다. */
#include "cmf_time_epoch.h"
#include "line_jbar.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TIME_COUPLED_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static double Bnu(double nu,double T){return 2*H_PLANCK*nu*nu*nu/
    (C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(H_PLANCK*nu/(K_BOLTZMANN*T));}
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
    for(int i=0;i<NI;i++){n.nlte_Z[i]=iz[i];n.nlte_ion[i]=st[i];}
    for(int i=0;i<=NI;i++)n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=off[i];
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.nlte_level_populations=pops;n.within_sl_frac=within;n.nlte_line_map=&linemap;n.population_committed_generation=1;
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
        CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,newtime,NULL,NULL,NULL,heating));
        BFOpacity bf={0};CHECK(!bf_opacity_init_checked(&bf,NS));
        c.common_obs_n=5;c.common_obs_nu=malloc(5*sizeof(double));CHECK(c.common_obs_nu);
        for(int i=0;i<5;i++)c.common_obs_nu[i]=nu*(.99+.005*i);
        CMFInitialNLTEReport report;
        setenv("LUMINA_FIXED_TE_PROFILE","wrong_initial_T.csv",1);
        CHECK(cmfgen_solve_initial_nlte(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-7,&report)==CMF_EPOCH_INVALID);
        CHECK(T[0]==1e4&&T[1]==1e4&&a.population_committed_generation==1);
        setenv("LUMINA_FIXED_TE_PROFILE","initial_T.csv",1);
        CMFTimeEpochStatus status=cmfgen_solve_initial_nlte(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-7,&report);
        fprintf(stderr,"INITIAL_COUPLED_RESULT status=%d iterations=%d ne=%.17g/%.17g T=%.17g/%.17g\n",
                status,report.iterations,ne[0],ne[1],T[0],T[1]);
        CHECK(status==CMF_EPOCH_OK&&report.converged&&c.common_obs_ready&&report.iterations>1);
        CHECK(T[0]==1e4&&T[1]==1e4&&!p.physical_history&&!c.time_trial);
        double up,down,ci,three;
        CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&up,&down,&ci,&three)&&ci>0&&three>0);
        for(int s=0;s<NS;s++){
            CHECK(fabs(ne[s]/ions[NS+s]-1)<1e-8);
            CHECK(fabs((ions[s]+ions[NS+s])/1e8-1)<1e-12);
            CHECK(fabs((pops[s]+pops[NS+s])/ions[s]-1)<1e-12);
        }
        CHECK(!cmfgen_write_common_observer(&c,"initial_spectrum.csv","initial_spectrum.json",
            n.radfield_view.generation,a.population_committed_generation,p.T_e_generation,o.tau_computed_generation));
        puts("CMF_INITIAL_COUPLED_SELFTEST_PASS actual_radiation=1 actual_R7=1 fixed_T=1 synthetic_atom=1 time_history=NOT_CAPTURED SN_comparison=0");
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
        CHECK(fabs(ledger->e_balance)<1e-12);
        fprintf(stderr,"TIME_COUPLED_CLOSURE shell=%d material_energy_balance=%.17g charge_relative=%.17g neutral_fraction=%.17g\n",
            s,ledger->e_balance,ne[s]/ions[NS+s]-1,ions[s]/(ions[s]+ions[NS+s]));
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
