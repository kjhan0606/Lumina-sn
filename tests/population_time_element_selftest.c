#include "nlte_population_candidate.h"
#include "cmf_nonthermal_rates.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c,m) do{if(!(c)){fprintf(stderr,"[ELEMENT-TIME][FAIL] %s line=%d\n",m,__LINE__);return 1;}}while(0)
extern int nlte_time_core_selftest_solve(NLTEConfig*,AtomicData*,PlasmaState*,OpacityState*,double,GammaDeposition*);

static int provider_calls,provider_fail;
static int fixture_provider(const void *ctx,const AtomicData *a,const NLTEConfig *n,
    const PlasmaState *p,const GammaDeposition *g,CMFNonthermalRates *out){
    const CMFNonthermalRates *in=ctx;
    if(provider_fail||!p->n_electron||!n->nlte_level_populations||a->n_levels!=in->n_levels)return -1;
    *out=*in;out->epoch_s=g->epoch;out->deposition_generation=g->generation;
    out->transitions=malloc((size_t)in->n_transitions*sizeof(*out->transitions));
    out->rate_per_particle=malloc((size_t)in->n_transitions*in->n_shells*sizeof(double));
    if(!out->transitions||!out->rate_per_particle)return -1;
    memcpy(out->transitions,in->transitions,(size_t)in->n_transitions*sizeof(*out->transitions));
    memcpy(out->rate_per_particle,in->rate_per_particle,(size_t)in->n_transitions*in->n_shells*sizeof(double));
    provider_calls++;return 0;
}
static int chain_check(int count,int Zvalue,int resolved_nt)
{
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState op={0};
    int Z[4],stage[4],offset[5]={0,1,2,3,4},map[4]={0,1,2,3};
    int g[4]={1,1,1,1},num[4]={0,0,0,0},sl[4]={0,0,0,0};
    int el[1]={Zvalue},eloff[2]={0,count},iz[3]={Zvalue,Zvalue,Zvalue},ist[3]={1,2,3};
    int routes[5]={0},target[3]={1,2,3};double probability[3]={1,1,1};
    double mass[1]={32},abundance[1]={1},chi[3]={16.35,33.49,45.14},energy[4]={0,0,0,0};
    double rho[1]={32*AMU},oldrho[1]={8*32*AMU},T[1]={1e4},ne[1]={1e-100};
    double population[4]={1,0,0,0},ion_population[4]={1,0,0,0},past[4]={8,0,0,0};
    double within[4]={1,1,1,1},partition[4]={1,1,1,1},J[1]={0};
    n.enabled=1;n.n_nlte_ions=count;n.n_nlte_levels_total=count;n.n_super_total=count;
    for(int i=0;i<count;i++){
        Z[i]=Zvalue;stage[i]=i+1;n.nlte_Z[i]=Zvalue;n.nlte_ion[i]=stage[i];
        n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=i;routes[i]=i;
    }
    routes[count-1]=routes[count]=count-1;
    n.nlte_ion_level_offset[count]=n.nlte_ion_super_offset[count]=count;
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.within_sl_frac=within;n.nlte_level_populations=population;n.nu_max=1e15;n.nu_min=1e14;
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;n.radfield_view.J_nu=J;n.radfield_view.generation=3;
    n.line_view_status=LINE_JBAR_VIEW_OK;n.line_view.jbar=J;n.line_view.generation=3;
    n.population_committed_generation=1;
    a.n_ion_pops=count;a.n_levels=count;a.ion_pop_Z=Z;a.ion_pop_stage=stage;
    a.level_offset=offset;a.level_energy_eV=energy;a.level_g=g;a.level_num=num;a.level_super=sl;
    a.level_Z=Z;a.level_ion=stage;a.ion_number_density=ion_population;a.partition_functions=partition;
    a.n_elements=1;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abundance;
    a.n_ionization=count-1;a.ioniz_Z=iz;a.ioniz_ion=ist;a.ioniz_energy_eV=chi;
    a.ma_rr_loaded=1;a.ma_rr_n_routes=count-1;a.ma_rr_target_offset=routes;
    a.ma_rr_targets=target;a.ma_rr_probability=probability;a.population_committed_generation=1;
    p.n_shells=1;p.rho=rho;p.T_e=T;p.n_electron=ne;p.T_e_generation=1;
    NLTEPhysicalHistory h={0};h.energy.n_shells=1;h.energy.epoch_s=1;
    h.energy.density_g_cm3=oldrho;h.n_levels=count;h.n_ions=count;
    h.level_density=past;h.level_global=map;h.ion_Z=Z;h.ion_stage=stage;p.physical_history=&h;
    GammaDeposition gd={0};gamma_deposition_init(&gd,1);
    double heating[1]={2*35*EV_TO_ERG/.05};
    CHECK(!gamma_deposition_publish(&gd,resolved_nt?GAMMA_PROVENANCE_NONLOCAL_MC:GAMMA_PROVENANCE_EXTERNAL_FILE,
                                   2,NULL,NULL,NULL,heating),"deposition");
    CMFNonthermalTransition nt_route[2]={{0,2,-1},{2,3,-1}};double nt_rate[2]={2,1};
    CMFNonthermalRates nt={.n_shells=1,.n_levels=count,.n_ions=count,.n_transitions=count==4?2:1,
        .epoch_s=2,.deposition_generation=gd.generation,.transitions=nt_route,.rate_per_particle=nt_rate};
    CMFNonthermalProvider provider={&nt,fixture_provider};
    if(resolved_nt){
        CHECK(gd.nonthermal_ioniz_rate[0]==0&&gd.heating_rate[0]==heating[0],"MC did not invoke ETA/W or discard deposition");
        GammaDeposition published=gd;double invalid[1]={-1};
        CHECK(gamma_deposition_publish(&gd,GAMMA_PROVENANCE_NONLOCAL_MC,3,NULL,NULL,NULL,invalid)!=0&&
              !memcmp(&gd,&published,sizeof gd)&&gd.heating_rate[0]==heating[0],"invalid MC publication preservation");
        gd.nonthermal_rates=&nt;gd.nonthermal_provider=&provider;
    }
    /* 고정 계수 시험: 전자밀도는 극소로 주어 DR을 무시 가능한 수준으로
     * 둔다. 전하/온도 자기일관성이나 실제 Si/S 원자료 시험으로 해석하지 않는다. */
    CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_OK,"whole-element assembly and solve");
    double expected[4]={0},left=1;
    for(int i=0;i<count-1;i++){expected[i]=left/3;left*=2./3;}expected[count-1]=left;
    if(resolved_nt){expected[0]=1./3;expected[1]=0;expected[2]=count==3?2./3:1./3;expected[3]=count==4?1./3:0;}
    double sum=0;
    for(int i=0;i<count;i++){
        CHECK(fabs(population[i]-expected[i])<1e-14 && population[i]==ion_population[i],"analytic multistage population");
        sum+=population[i];
    }
    CHECK(fabs(sum-1)<1e-14 && past[0]==8 && past[1]==0,"mass and past");
    int calls_before=provider_calls;
    if(resolved_nt)gd.nonthermal_rates=NULL; /* 실제 core가 현재 후보에서 생성해야 한다. */
    CHECK(!nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd),"actual private core whole-element route");
    if(resolved_nt)CHECK(provider_calls>calls_before&&gd.nonthermal_rates==NULL,"current-candidate provider or lifetime");
    CHECK(a.population_committed_generation==2 && n.population_committed_generation==2,"atomic private commit");
    for(int i=0;i<count;i++)CHECK(fabs(population[i]-expected[i])<1e-14,"core preserved multistage solution");
    double saved[4],savedion[4];memcpy(saved,population,sizeof(saved));memcpy(savedion,ion_population,sizeof(savedion));
    if(resolved_nt){
        gd.nonthermal_rates=&nt;nt_route[0].upper_global=count;
        CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK&&
              !memcmp(saved,population,sizeof(saved))&&!memcmp(savedion,ion_population,sizeof(savedion)),"resolved target rejection");
        nt_route[0].upper_global=2;gd.nonthermal_rates=NULL;provider_fail=1;
        CHECK(nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd)!=0&&a.population_committed_generation==2&&
              !memcmp(saved,population,sizeof(saved))&&!memcmp(savedion,ion_population,sizeof(savedion)),"provider failure rollback");
        provider_fail=0;gd.nonthermal_provider=NULL;
        CHECK(nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd)!=0&&
              !memcmp(saved,population,sizeof(saved)),"MC without SF provider refused");
        printf("[ELEMENT-TIME] RESOLVED_NT_PASS Z=%d stages=%d n0=%.17g n1=%.17g n2=%.17g top=%.17g provider_calls=%d ETA_W=0 total_deposition_preserved=1\n",
               Zvalue,count,saved[0],saved[1],saved[2],saved[count-1],provider_calls-calls_before);
        gamma_deposition_free(&gd);return 0;
    }
    target[0]=count;
    CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK &&
          !memcmp(saved,population,sizeof(saved)) && !memcmp(savedion,ion_population,sizeof(savedion)),"missing target atomic rejection");
    CHECK(nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd)!=0 &&
          a.population_committed_generation==2 && !memcmp(saved,population,sizeof(saved)) &&
          !memcmp(savedion,ion_population,sizeof(savedion)),"private core failed step rollback");
    target[0]=1;n.n_nlte_ions=count-1;
    CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_ATOMIC_MISSING,"omitted ion stage rejection");
    printf("[ELEMENT-TIME] PASS Z=%d stages=%d n0=%.17g n1=%.17g top=%.17g total=%.17g core=PASS rollback=PASS\n",
           Zvalue,count,saved[0],saved[1],saved[count-1],sum);
    if(count==3){
        /* 광전리도 같은 원소 행렬에 들어가는지 확인한다. 여기의 J와 단면적은
         * 합성값이며 CMFGEN 원자료/분포를 대신하지 않는다. */
        enum{NB=64};double edges[NB+1],field[NB],sigma[4*NB];
        RadiationFieldValidityState valid[NB];uint64_t samples[NB];
        int has[4]={1,1,1,1};
        n.n_nlte_ions=count;n.n_freq_bins=NB;n.nu_min=1e14;n.nu_max=1e17;
        n.d_log_nu=log(n.nu_max/n.nu_min)/NB;n.J_nu=field;
        for(int i=0;i<=NB;i++)edges[i]=n.nu_min*exp(i*n.d_log_nu);
        for(int i=0;i<NB;i++){field[i]=1e-8;valid[i]=RADIATION_FIELD_VALID;samples[i]=1;}
        for(int i=0;i<4*NB;i++)sigma[i]=1e-18;
        a.cmfgen_loaded=1;a.cmfgen_n_freq_bins=NB;a.cmfgen_has_sigma=has;a.cmfgen_sigma_bf=sigma;
        n.radfield_view.n_shells=1;n.radfield_view.n_bins=NB;
        n.radfield_view.frequency_bin_edges=edges;n.radfield_view.J_nu=field;
        n.radfield_view.validity=valid;n.radfield_view.count=samples;
        CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_OK,"actual canonical BF into time matrix");
        CHECK(population[0]<.1 && population[2]>saved[2],"BF field did not change ion distribution");
        double bf_saved[4];memcpy(bf_saved,population,sizeof(bf_saved));
        n.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
        CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK &&
              !memcmp(bf_saved,population,sizeof(bf_saved)),"stale BF silently treated as zero");
        n.radfield_view_status=RADIATION_FIELD_VIEW_OK;
        for(int i=0;i<NB;i++)valid[i]=RADIATION_FIELD_UNSAMPLED;
        CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK &&
              !memcmp(bf_saved,population,sizeof(bf_saved)),"unsampled BF silently treated as zero");
        printf("[ELEMENT-TIME] BF_PASS lower_before=%.17g lower_after=%.17g top_after=%.17g stale_rejected=1 unsampled_rejected=1\n",
            saved[0],bf_saved[0],bf_saved[2]);
    }
    gamma_deposition_free(&gd);return 0;
}

static int charge_exchange_check(void)
{
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState op={0};
    int Z[4]={14,14,20,20},stage[4]={1,2,1,2},offset[5]={0,1,2,3,4};
    int map[4]={0,1,2,3},g[4]={1,1,1,1},num[4]={0,0,0,0},sl[4]={0,0,0,0};
    int el[2]={14,20},eloff[3]={0,2,4},ist[2]={1,1};
    double mass[2]={28,40},abundance[2]={56./176,120./176},chi[2]={16.35,11.87},energy[4]={0};
    double rho[1]={176*AMU},oldrho[1]={8*176*AMU},T[1]={1e4},ne[1]={1e-100};
    double population[4]={0,2,3,0},ions[4]={0,2,3,0},past[4]={0,16,24,0};
    double within[4]={1,1,1,1},partition[4]={1,1,1,1},J[1]={0};
    n.enabled=1;n.n_nlte_ions=n.n_nlte_levels_total=n.n_super_total=4;
    for(int i=0;i<4;i++){
        n.nlte_Z[i]=Z[i];n.nlte_ion[i]=stage[i];
        n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=i;
    }
    n.nlte_ion_level_offset[4]=n.nlte_ion_super_offset[4]=4;
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.within_sl_frac=within;n.nlte_level_populations=population;n.nu_max=1e15;n.nu_min=1e14;
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;n.radfield_view.J_nu=J;n.radfield_view.generation=3;
    n.line_view_status=LINE_JBAR_VIEW_OK;n.line_view.jbar=J;n.line_view.generation=3;
    n.population_committed_generation=1;
    a.n_ion_pops=a.n_levels=4;a.ion_pop_Z=Z;a.ion_pop_stage=stage;
    a.level_offset=offset;a.level_energy_eV=energy;a.level_g=g;a.level_num=num;a.level_super=sl;
    a.level_Z=Z;a.level_ion=stage;a.ion_number_density=ions;a.partition_functions=partition;
    a.n_elements=2;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abundance;
    a.n_ionization=2;a.ioniz_Z=el;a.ioniz_ion=ist;a.ioniz_energy_eV=chi;a.population_committed_generation=1;
    p.n_shells=1;p.rho=rho;p.T_e=T;p.n_electron=ne;p.T_e_generation=1;
    NLTEPhysicalHistory h={0};h.energy.n_shells=1;h.energy.epoch_s=1e9;
    h.energy.density_g_cm3=oldrho;h.n_levels=h.n_ions=4;
    h.level_density=past;h.level_global=map;h.ion_Z=Z;h.ion_stage=stage;p.physical_history=&h;
    GammaDeposition gd={0};gamma_deposition_init(&gd,1);double heating[1]={0};
    CHECK(!gamma_deposition_publish(&gd,GAMMA_PROVENANCE_EXTERNAL_FILE,2e9,NULL,NULL,NULL,heating),"zero NT publication");
    CHECK(!nlte_time_core_selftest_solve(&n,&a,&p,&op,2e9,&gd),"coupled CE core");
    double reverse=exp(-4.48*EV_TO_ERG/(K_BOLTZMANN*T[0]));
    double extent=12/(6+sqrt(36-24*(1-reverse)));
    double charge=ions[0]+2*ions[1]+ions[2]+2*ions[3];
    CHECK(fabs(ions[0]-extent)<2e-8 && fabs(ions[3]-extent)<2e-8 &&
          fabs(ions[0]+ions[1]-2)<1e-14 && fabs(ions[2]+ions[3]-3)<1e-14 &&
          fabs(charge-7)<2e-8,"actual CE extent or conserved charge");
    CHECK(past[0]==0 && past[1]==16 && past[2]==24 && past[3]==0,"CE did not overwrite physical past");
    printf("[ELEMENT-TIME] CE_PASS SiII=%.17g CaIII=%.17g analytic_extent=%.17g charge=%.17g\n",
        ions[0],ions[3],extent,charge);
    gamma_deposition_free(&gd);return 0;
}

int main(void)
{
    if(chain_check(3,14,0)||chain_check(4,16,0)||charge_exchange_check()||
       chain_check(3,14,1)||chain_check(4,16,1))return 1;
    puts("[ELEMENT-TIME] PASS overlapping_pair_restore=0 steady_ion_renormalization=0");
    return 0;
}
