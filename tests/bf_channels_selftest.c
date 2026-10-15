#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c,m) do{if(!(c)){fprintf(stderr,"[BF-CHANNELS][FAIL] %s line=%d\n",m,__LINE__);return 1;}}while(0)
static int near(double x,double y,double tol){return fabs(x-y)<=tol*fmax(fmax(fabs(x),fabs(y)),1e-300);}

static int kernel_check(void){
    double edge=20*EV_TO_ERG/H_PLANCK,T=1e4,ne=1e8;
    double ratio=ne*pow(H_PLANCK*H_PLANCK/(2*M_PI_VAL*M_ELECTRON*K_BOLTZMANN*T),1.5)*
                 2./6*exp(H_PLANCK*edge/(K_BOLTZMANN*T));
    BfChannelBin v={0};
    CHECK(!bf_channel_bin(.999*edge,1.004*edge,edge,1e-18,T,ne,2,3,ratio,1,0,1,&v),"Planck bin");
    CHECK(near(ratio*v.ionization,v.recombination,3e-14),"Milne detailed balance");
    CHECK(fabs(v.energy_exchange)<1e-13*4*M_PI_VAL*v.eta*.005*edge,"LTE energy balance");
    CHECK(!bf_channel_bin(.999*edge,1.004*edge,edge,1e-18,T,ne,2,3,0,1,1e-8,0,&v),"empty lower");
    CHECK(v.chi_net<0&&v.eta>0&&v.recombination>0&&v.chi_absorption==0,"inverse survives zero lower");
    CHECK(near(v.energy_exchange,4*M_PI_VAL*(v.chi_net*1e-8-v.eta)*.005*edge,2e-13),"same radiative energy");
    BfChannelBin saved=v;
    CHECK(bf_channel_bin(edge,edge,edge,1e-18,T,ne,2,3,0,1,1e-8,0,&v)!=0&&
          !memcmp(&saved,&v,sizeof(v)),"invalid bin transactional");
    CHECK(!bf_channel_bin(.9*edge,1.1*edge,edge,0,T,ne,2,3,1,1,1e-8,0,&v)&&
          v.ionization==0&&v.recombination==0&&v.eta==0,"explicit zero is not fallback");
    puts("[BF-CHANNELS] KERNEL_PASS detailed_balance=1 zero_lower_emission=1 energy_identity=1");
    return 0;
}

static int write_channel_file(const char *path,const BfChannels *c,int bad){
    FILE *fp=fopen(path,"wbx");if(!fp)return -1;
    uint32_t version=c->version?c->version:1;
    uint32_t header[5]={0x43424643,version,(uint32_t)c->n_levels,(uint32_t)c->n_channels,(uint32_t)c->n_bins};
    double edges[2]={c->nu_min,c->nu_max},threshold[2]={c->threshold[0],c->threshold[1]*(bad?1.01:1)};
    int ok=fwrite(header,4,5,fp)==5&&fwrite(edges,8,2,fp)==2&&
           fwrite(c->offset,4,c->n_levels+1,fp)==(size_t)c->n_levels+1&&
           fwrite(c->upper,4,2,fp)==2&&fwrite(threshold,8,2,fp)==2&&
           fwrite(c->sigma,8,2*c->n_bins,fp)==(size_t)2*c->n_bins;
    if(version==2)ok=ok&&fwrite(c->upper_g,8,2,fp)==2;
    return fclose(fp)||!ok?-1:0;
}

static int cache_check(NLTEConfig *n,AtomicData *a,double *J,RadiationFieldValidityState *valid){
    struct NLTEBFRateCache *cache=nlte_bf_rate_cache_create(a,1);
    CHECK(cache,"rate cache allocation");
    const double density[]={0,1e-100,1,1e8,1e13,1e50};
    const double temperature[]={1e4,8000,2e4};
    double worst=0;
    for(int field=0;field<2;field++){
      if(field){for(size_t b=0;b<n->radfield_view.n_bins;b++)J[b]*=3;n->radfield_view.generation++;}
      for(int t=0;t<3;t++)for(int e=0;e<6;e++)for(int r=0;r<2;r++){
        double direct[4],cached[4];n->bf_rate_cache=NULL;
        CHECK(!nlte_bf_channel_rates(n,a,0,r,0,temperature[t],density[e],direct,direct+1,direct+2,direct+3),"uncached rates");
        n->bf_rate_cache=cache;
        CHECK(!nlte_bf_channel_rates(n,a,0,r,0,temperature[t],density[e],cached,cached+1,cached+2,cached+3),"cached rates");
        for(int k=0;k<4;k++){
            CHECK(near(direct[k],cached[k],1e-12),"density/T/field cache parity");
            if(direct[k]||cached[k])worst=fmax(worst,fabs(direct[k]-cached[k])/fmax(fabs(direct[k]),fabs(cached[k])));
        }
        if(r==1&&density[e]>0)CHECK(cached[2]>0&&cached[3]>0,"collisional and three body terms exercised");
      }
    }
    double out[4]={-1,-2,-3,-4},saved[4];memcpy(saved,out,sizeof out);
    n->radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    CHECK(nlte_bf_channel_rates(n,a,0,1,0,1e4,1e8,out,out+1,out+2,out+3)&&!memcmp(saved,out,sizeof out),"disabled view rejected with cache");
    n->radfield_view_status=RADIATION_FIELD_VIEW_OK;
    valid[n->radfield_view.n_bins-1]=RADIATION_FIELD_UNSAMPLED;n->radfield_view.generation++;
    CHECK(nlte_bf_channel_rates(n,a,0,1,0,1e4,1e8,out,out+1,out+2,out+3)&&!memcmp(saved,out,sizeof out),"new invalid field not hidden by cache");
    n->bf_rate_cache=NULL;nlte_bf_rate_cache_free(cache);
    printf("[BF-CACHE] PASS max_relative=%.17g density_cases=6 temperatures=3 changed_field=1 collisions=1 three_body=1 SN_comparison=0\n",worst);
    return 0;
}

static int integration_check(void){
    enum{NB=NLTE_N_FREQ_BINS};
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState op={0};
    int ionZ[2]={1,1},st[2]={0,1},off[3]={0,1,3},map[3]={0,1,2};
    int levZ[3]={1,1,1},levst[3]={0,1,1},g[3]={2,1,3},num[3]={0,0,1},super[3]={0,0,1},meta[3]={1,1,1};
    int el[1]={1},eloff[2]={0,2},iz[1]={1},ist[1]={0},coff[4]={0,2,2,2},target[2]={1,2};
    int nt_offsets[4]={0,1,1,1},nt_target[1]={1};double nt_probability[1]={1};
    double mass[1]={1},abund[1]={1},chi[1]={20},E[3]={0,0,6};
    double rho[1]={AMU},T[1]={1e4},ne[1]={1e8},pops[3]={1,0,0},ions[2]={1,0};
    double within[3]={1,1,1},partition[2]={2,1+3*exp(-6*EV_TO_ERG/(K_BOLTZMANN*1e4))};
    double edges[NB+1],J[NB],sigma[2*NB],threshold[2]={20*EV_TO_ERG/H_PLANCK,26*EV_TO_ERG/H_PLANCK};
    RadiationFieldValidityState valid[NB];uint64_t samples[NB];
    double dlog=log(NLTE_NU_MAX/NLTE_NU_MIN)/NB;
    for(int b=0;b<=NB;b++)edges[b]=NLTE_NU_MIN*exp(b*dlog);
    for(int b=0;b<NB;b++){
        J[b]=1e-8;valid[b]=RADIATION_FIELD_VALID;samples[b]=1;
        sigma[b]=0;sigma[NB+b]=edges[b+1]>threshold[1]?1e-18:0;
    }
    BfChannels c={3,2,NB,NLTE_NU_MIN,NLTE_NU_MAX,coff,target,threshold,sigma};
    n.enabled=1;n.n_nlte_ions=2;n.n_nlte_levels_total=n.n_super_total=3;
    n.nlte_Z[0]=n.nlte_Z[1]=1;n.nlte_ion[0]=0;n.nlte_ion[1]=1;
    for(int i=0;i<3;i++)n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=off[i];
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.within_sl_frac=within;n.nlte_level_populations=pops;n.n_freq_bins=NB;
    n.nu_min=NLTE_NU_MIN;n.nu_max=NLTE_NU_MAX;n.d_log_nu=dlog;n.J_nu=J;
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;n.radfield_view.n_shells=1;n.radfield_view.n_bins=NB;
    n.radfield_view.frequency_bin_edges=edges;n.radfield_view.J_nu=J;
    n.radfield_view.validity=valid;n.radfield_view.count=samples;n.radfield_view.generation=3;
    n.line_view_status=LINE_JBAR_VIEW_OK;n.line_view.jbar=J;n.line_view.generation=3;
    n.population_committed_generation=1;
    a.n_ion_pops=2;a.n_levels=3;a.ion_pop_Z=ionZ;a.ion_pop_stage=st;a.level_offset=off;
    a.level_Z=levZ;a.level_ion=levst;a.level_g=g;a.level_num=num;a.level_super=super;
    a.level_energy_eV=E;a.level_metastable=meta;a.ion_number_density=ions;a.partition_functions=partition;
    a.n_elements=1;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abund;
    a.n_ionization=1;a.ioniz_Z=iz;a.ioniz_ion=ist;a.ioniz_energy_eV=chi;a.population_committed_generation=1;
    a.ma_rr_loaded=1;a.ma_rr_n_routes=1;a.ma_rr_target_offset=nt_offsets;
    a.ma_rr_targets=nt_target;a.ma_rr_probability=nt_probability;
    p.n_shells=1;p.rho=rho;p.T_e=T;p.n_electron=ne;p.T_e_generation=1;
    CHECK(!write_channel_file("channels.bin",&c,0)&&!load_bf_channels(&a,"channels.bin"),"channel loader");
    BfChannels *loaded=a.bf_channels;
    if(getenv("CMF_TEST_BF_CACHE_ONLY")){
        int status=cache_check(&n,&a,J,valid);free_bf_channels(loaded);return status;
    }
    CHECK(!write_channel_file("channels-invalid.bin",&c,1)&&load_bf_channels(&a,"channels-invalid.bin")&&
          a.bf_channels==loaded,"loader rejects wrong threshold without replacing prior data");
    double up,down,cu,cd;
    CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&up,&down,&cu,&cd)&&up==0&&down==0,"A zero channel");
    CHECK(!nlte_bf_channel_rates(&n,&a,0,1,0,T[0],ne[0],&up,&down,&cu,&cd)&&up>0&&down>0&&cu==0&&cd==0,"B excited parent rates");
    /* 실제 게시 격자는3866칸이고 원자 pack은1234칸이다. 같은 보존 평균의
     * 율, 누락/비표본 거부를 확인한다. 아래 whole-element 시험 전에 복원한다. */
    RadiationFieldView saved_view=n.radfield_view;
    double canonical_edges[LUMINA_RADFIELD_N_BINS+1],canonical_J[LUMINA_RADFIELD_N_BINS];
    RadiationFieldValidityState canonical_valid[LUMINA_RADFIELD_N_BINS];
    for(int i=0;i<=LUMINA_RADFIELD_N_BINS;i++)canonical_edges[i]=radiation_field_canonical_frequency_edge(i);
    for(int i=0;i<LUMINA_RADFIELD_N_BINS;i++){canonical_J[i]=1e-8;canonical_valid[i]=RADIATION_FIELD_VALID;}
    n.radfield_view.n_bins=LUMINA_RADFIELD_N_BINS;n.radfield_view.frequency_bin_edges=canonical_edges;
    n.radfield_view.J_nu=canonical_J;n.radfield_view.validity=canonical_valid;
    double up_fine,down_fine;
    CHECK(!nlte_bf_channel_rates(&n,&a,0,1,0,T[0],ne[0],&up_fine,&down_fine,&cu,&cd)&&
          near(up,up_fine,1e-13)&&near(down,down_fine,1e-13),"canonical grid conservation");
    for(int i=0;i<LUMINA_RADFIELD_N_BINS;i++)canonical_J[i]=(i%2?3:1)*1e-8;
    double expected_up=0,expected_down=0;
    for(int b=0;b<NB;b++){
        int q0=-LUMINA_RADFIELD_J_LO+2*b;
        double jm=(canonical_J[q0]*(canonical_edges[q0+1]-edges[b])+
                   canonical_J[q0+1]*(edges[b+1]-canonical_edges[q0+1]))/(edges[b+1]-edges[b]);
        BfChannelBin v;
        CHECK(!bf_channel_bin(edges[b],edges[b+1],threshold[1],sigma[NB+b],T[0],ne[0],2,3,1,1,jm,0,&v),
              "independent rebin coefficients");
        expected_up+=v.ionization;expected_down+=v.recombination;
    }
    CHECK(!nlte_bf_channel_rates(&n,&a,0,1,0,T[0],ne[0],&up_fine,&down_fine,&cu,&cd)&&
          near(expected_up,up_fine,1e-13)&&near(expected_down,down_fine,1e-13)&&up_fine>up,
          "canonical varying field not stale coarse copy");
    canonical_valid[LUMINA_RADFIELD_N_BINS-1]=RADIATION_FIELD_UNSAMPLED;
    CHECK(nlte_bf_channel_rates(&n,&a,0,1,0,T[0],ne[0],&up_fine,&down_fine,&cu,&cd)!=0,
          "canonical missing cell rejected");
    canonical_valid[LUMINA_RADFIELD_N_BINS-1]=RADIATION_FIELD_VALID;
    n.radfield_view.n_bins--;
    CHECK(nlte_bf_channel_rates(&n,&a,0,1,0,T[0],ne[0],&up_fine,&down_fine,&cu,&cd)!=0,
          "canonical missing coverage rejected");
    n.radfield_view=saved_view;
    NLTEPhysicalHistory h={0};double past[3]={8,0,0},oldrho[1]={8*AMU};
    h.energy.n_shells=1;h.energy.epoch_s=1;h.energy.density_g_cm3=oldrho;
    h.n_levels=3;h.n_ions=2;h.level_density=past;h.level_global=map;h.ion_Z=ionZ;h.ion_stage=st;p.physical_history=&h;
    GammaDeposition gd={0};gamma_deposition_init(&gd,1);double heating[1]={0};
    CHECK(!gamma_deposition_publish(&gd,GAMMA_PROVENANCE_EXTERNAL_FILE,2,NULL,NULL,NULL,heating),"zero NT");
    double A[9]={0},b[3]={0};
    CHECK(!nlte_assemble_rate_matrix(&n,&a,&p,&op,0,1,0,2,A,b,3,&gd,NULL,-1),"actual pair assembler");
    CHECK(near(A[2],up,1e-13)&&near(A[6],down,1e-13)&&A[1]==0&&A[3]==0,"rates reach distinct upper level once");
    for(int col=0;col<3;col++)CHECK(fabs(A[col*3]+A[col*3+1]+A[col*3+2])<1e-13,"particle conservation");
    PopulationStatus stime=nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd);
    if(stime!=POP_OK)fprintf(stderr,"element status=%s\n",population_status_name(stime));
    CHECK(stime==POP_OK,"whole-element channel solve");
    CHECK(near(pops[0],(1+down)/(1+up+down),1e-12)&&pops[1]==0&&
          near(pops[2],up/(1+up+down),1e-12),"analytic channel evolution");
    double saved[3];memcpy(saved,pops,sizeof(saved));
    n.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK&&
          !memcmp(saved,pops,sizeof(saved)),"invalid field preserves populations");
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;
    /* 실제 연속 producer: n_lower=0, n_upper_excited=1. FF는 따로 빼서 비교한다. */
    pops[0]=pops[1]=0;pops[2]=1;ions[0]=0;ions[1]=1;
    BFOpacity bf={0};CHECK(!bf_opacity_init_checked(&bf,1),"BF allocation");
    bf_set_nlte_pops(&n);compute_bf_opacity(&bf,&a,&p,1);
    int checked=0;
    for(int ib=0;ib<NB;ib++)if(sigma[NB+ib]>0){
        BfChannelBin v;
        CHECK(!bf_channel_bin(edges[ib],edges[ib+1],threshold[1],sigma[NB+ib],T[0],ne[0],2,3,0,1,0,0,&v),"reference coefficients");
        double nu=NLTE_NU_MIN*exp((ib+.5)*dlog),x=H_PLANCK*nu/(K_BOLTZMANN*T[0]);
        double ff=C_FF_OPACITY/sqrt(T[0])*ne[0]/(nu*nu*nu)*(1-exp(-x));
        double B=x>700?0:2*H_PLANCK*nu*nu*nu/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(x);
        CHECK(near(bf.chi_bf[ib],ff+v.chi_net,1e-12)&&near(bf.eta_bf[ib],ff*B+v.eta,1e-12),"production chi/eta match channel");
        CHECK(bf.event_chi_bf[ib]==0,"no fictitious absorption from empty lower");checked++;
    }
    CHECK(checked>0&&bf.eta_milne_ok[0],"inverse emissivity survived continuum loop");
    /* 동적 이온 목록의 끝 슬롯도 실제 선 방출 인구를 사용해야 한다.
     * 구 고정 pair 표에서 소유하지 않는 slot 37을 의도적으로 사용한다. */
    int lineZ[1]={1},linest[1]={1},linelo[1]={0},lineup[1]={1},linemap[1]={37};
    a.n_lines=1;a.line_atomic_number=lineZ;a.line_ion_number=linest;
    a.line_level_lower=linelo;a.line_level_upper=lineup;
    n.n_nlte_ions=38;n.nlte_Z[37]=1;n.nlte_ion[37]=1;n.nlte_line_map=linemap;
    a.bf_channels->version=2;double nupper=-1;
    CHECK(lumina_line_upper_population_for_tau(&a,&p,&n,0,0,1,&nupper)==EMISS_OK&&
          nupper==1,"dynamic high-slot line population authority");
    printf("[BF-CHANNELS] INTEGRATION_PASS up=%.17g down=%.17g excited_after=%.17g continuum_bins=%d\n",up,down,saved[2],checked);
    bf_set_nlte_pops(NULL);bf_opacity_free(&bf);gamma_deposition_free(&gd);free_bf_channels(a.bf_channels);
    return 0;
}

extern int nlte_time_core_selftest_solve(NLTEConfig*,AtomicData*,PlasmaState*,OpacityState*,double,GammaDeposition*);

static int terminal_check(void){
    enum{NB=NLTE_N_FREQ_BINS};
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState op={0};
    int Z[3]={14,14,14},st[3]={1,2,3},off[4]={0,1,2,2},map[2]={0,1};
    int g[2]={2,1},num[2]={0,0},super[2]={0,0},meta[2]={1,1};
    int el[1]={14},eloff[2]={0,3},coff[3]={0,1,2},upper[2]={1,-1},uion[2]={1,2};
    int top_ip[1]={2};
    int refZ[1]={14},refstage[1]={0};double refchi[1]={8};
    double top_E[1]={0},top_g[1]={6},gu[2]={1,6};
    double mass[1]={28},abund[1]={1},chi[2]={16.35,33.49},E[2]={0,0};
    double rho[1]={28*AMU},oldrho[1]={8*28*AMU},T[1]={1e4},ne[1]={1e-100};
    double pops[2]={1,0},ions[3]={1,0,0},past[2]={8,0},pastion[3]={8,0,0};
    double within[2]={1,1},partition[3]={2,1,6};
    double edges[NB+1],J[NB],sigma[2*NB],threshold[2];
    RadiationFieldValidityState valid[NB];uint64_t samples[NB];
    double dlog=log(NLTE_NU_MAX/NLTE_NU_MIN)/NB;
    for(int r=0;r<2;r++)threshold[r]=chi[r]*EV_TO_ERG/H_PLANCK;
    for(int b=0;b<=NB;b++)edges[b]=NLTE_NU_MIN*exp(b*dlog);
    for(int b=0;b<NB;b++){
        J[b]=1e-8;valid[b]=RADIATION_FIELD_VALID;samples[b]=1;
        for(int r=0;r<2;r++)sigma[r*NB+b]=edges[b+1]>threshold[r]?1e-18:0;
    }
    BfChannels c={2,2,NB,NLTE_NU_MIN,NLTE_NU_MAX,coff,upper,threshold,sigma,2,uion,gu};
    n.enabled=1;n.n_nlte_ions=3;n.n_nlte_levels_total=n.n_super_total=2;
    for(int i=0;i<3;i++){n.nlte_Z[i]=Z[i];n.nlte_ion[i]=st[i];}
    for(int i=0;i<4;i++)n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=off[i];
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.within_sl_frac=within;n.nlte_level_populations=pops;n.n_freq_bins=NB;
    n.nu_min=NLTE_NU_MIN;n.nu_max=NLTE_NU_MAX;n.d_log_nu=dlog;n.J_nu=J;
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;n.radfield_view.n_shells=1;n.radfield_view.n_bins=NB;
    n.radfield_view.frequency_bin_edges=edges;n.radfield_view.J_nu=J;
    n.radfield_view.validity=valid;n.radfield_view.count=samples;n.radfield_view.generation=3;
    n.line_view_status=LINE_JBAR_VIEW_OK;n.line_view.jbar=J;n.line_view.generation=3;
    n.population_committed_generation=1;
    a.n_ion_pops=3;a.n_levels=2;a.ion_pop_Z=Z;a.ion_pop_stage=st;a.level_offset=off;
    a.level_Z=Z;a.level_ion=st;a.level_g=g;a.level_num=num;a.level_super=super;
    a.level_energy_eV=E;a.level_metastable=meta;a.ion_number_density=ions;a.partition_functions=partition;
    a.n_elements=1;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abund;
    a.n_ionization=2;a.ioniz_Z=Z;a.ioniz_ion=st;a.ioniz_energy_eV=chi;a.population_committed_generation=1;
    a.n_ionization_reference=1;a.ioniz_ref_Z=refZ;a.ioniz_ref_ion=refstage;a.ioniz_ref_energy_eV=refchi;
    a.topion_n=1;a.topion_ion_index=top_ip;a.topion_E_cm=top_E;a.topion_g=top_g;
    p.n_shells=1;p.rho=rho;p.T_e=T;p.n_electron=ne;p.T_e_generation=1;
    CHECK(!write_channel_file("terminal.bin",&c,0)&&!load_bf_channels(&a,"terminal.bin"),"terminal channel loader");
    CHECK(a.bf_channels->upper[1]==-1&&a.bf_channels->upper_ion[1]==2&&a.bf_channels->upper_g[1]==6,"terminal identity");
    NLTEConfig projection={0};
    CHECK(!nlte_init(&projection,&a,&op,1)&&projection.n_nlte_ions==3&&
          projection.n_nlte_levels_total==2&&projection.nlte_ion_level_offset[2]==2&&
          projection.nlte_ion_level_offset[3]==2,"actual dynamic projection no fake terminal level");
    nlte_free(&projection);
    BfChannels *loaded=a.bf_channels;
    top_g[0]=4;
    CHECK(load_bf_channels(&a,"terminal.bin")&&a.bf_channels==loaded,"inconsistent terminal partition rejected");
    top_g[0]=6;
    NLTEPhysicalHistory h={0};h.energy.n_shells=1;h.energy.epoch_s=1;h.energy.density_g_cm3=oldrho;
    h.n_levels=2;h.n_ions=3;h.level_density=past;h.ion_density=pastion;
    h.level_global=map;h.ion_Z=Z;h.ion_stage=st;p.physical_history=&h;
    GammaDeposition gd={0};gamma_deposition_init(&gd,1);double heating[1]={2*35*EV_TO_ERG/.05};
    CHECK(!gamma_deposition_publish(&gd,GAMMA_PROVENANCE_EXTERNAL_FILE,2,NULL,NULL,NULL,heating),"terminal NT");
    double up[2],down[2],cu,cd;
    for(int r=0;r<2;r++)CHECK(!nlte_bf_channel_rates(&n,&a,r,r,0,T[0],ne[0],up+r,down+r,&cu,&cd),"terminal rates");
    double q0=up[0]+2,q1=up[1]+2;
    double expected[3]={1/(1+q0),q0/((1+q0)*(1+q1)),0};
    expected[2]=1-expected[0]-expected[1];
    PopulationStatus ps=nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd);
    if(ps!=POP_OK)fprintf(stderr,"terminal status=%s\n",population_status_name(ps));
    CHECK(ps==POP_OK,"terminal actual whole-element solve");
    for(int i=0;i<3;i++)CHECK(near(ions[i],expected[i],2e-13),"analytic terminal chain including NT");
    CHECK(!nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd),"terminal private core solve");
    for(int i=0;i<3;i++)CHECK(near(ions[i],expected[i],2e-13),"terminal core preserved solution");
    CHECK(past[0]==8&&pastion[0]==8&&pastion[2]==0,"immutable physical past");
    double saved[2],savedion[3];memcpy(saved,pops,sizeof(saved));memcpy(savedion,ions,sizeof(savedion));
    n.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    CHECK(nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd)!=0&&
          !memcmp(saved,pops,sizeof(saved))&&!memcmp(savedion,ions,sizeof(savedion)),"terminal failed step rollback");
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;
    num[1]=1;
    CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK&&
          !memcmp(saved,pops,sizeof(saved))&&!memcmp(savedion,ions,sizeof(savedion)),"missing NT ground rejected");
    num[1]=0;
    /* 과거 최상위 이온이 실제 RHS에 남는지: J=0, 극소 ne에서 거의 모두 잔존. */
    past[0]=pastion[0]=0;pastion[2]=8;
    for(int b=0;b<NB;b++){J[b]=0;valid[b]=RADIATION_FIELD_EXACT_ZERO;}
    CHECK(nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_OK&&near(ions[2],1,1e-13),"terminal physical history retained");
    /* 단일 경계 상태 g=6의 역과정이 continuum에도 같은 계수로 전달된다. */
    ne[0]=1e8;pops[0]=pops[1]=0;ions[0]=ions[1]=0;ions[2]=1;
    BFOpacity bf={0};CHECK(!bf_opacity_init_checked(&bf,1),"terminal BF allocation");
    bf_set_nlte_pops(&n);compute_bf_opacity(&bf,&a,&p,1);
    int checked=0;
    for(int ib=0;ib<NB;ib++)if(sigma[NB+ib]>0){
        BfChannelBin v;
        CHECK(!bf_channel_bin(edges[ib],edges[ib+1],threshold[1],sigma[NB+ib],T[0],ne[0],1,6,0,1,0,0,&v),"terminal reference coefficients");
        double nu=NLTE_NU_MIN*exp((ib+.5)*dlog),x=H_PLANCK*nu/(K_BOLTZMANN*T[0]);
        double ff=C_FF_OPACITY/sqrt(T[0])*ne[0]*9/(nu*nu*nu)*(1-exp(-x));
        double B=x>700?0:2*H_PLANCK*nu*nu*nu/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)/expm1(x);
        CHECK(near(bf.chi_bf[ib],ff+v.chi_net,1e-12)&&near(bf.eta_bf[ib],ff*B+v.eta,1e-12),"terminal production inverse coefficients");
        CHECK(bf.event_chi_bf[ib]==0,"terminal zero lower no absorption");checked++;
    }
    CHECK(checked>0&&bf.eta_milne_ok[0],"terminal inverse emission");
    NLTEPhysicalHistory captured={0};
    CHECK(nlte_physical_history_capture(&captured,&n,&a,&p,2)==NLTE_CANDIDATE_OK&&
          captured.n_ions==3&&captured.n_levels==2&&captured.ion_density[2]==1&&
          near(captured.energy.internal_energy_atom_erg[0],(8+16.35+33.49)*EV_TO_ERG,1e-13),
          "actual history capture retains terminal and internal energy");
    nlte_physical_history_free(&captured);
    /* 초기 정상해는 실제 과거 없이 동일한 raw 전이율을 쓴다. 시간해의 큰
     * dt 근사가 아니다. terminal까지 3단계 해석적 birth/death 균형과 비교한다. */
    p.physical_history=NULL;p.initial_steady_nlte=1;
    for(int b=0;b<NB;b++){J[b]=1e-8;valid[b]=RADIATION_FIELD_VALID;}
    for(int r=0;r<2;r++){
        CHECK(!nlte_bf_channel_rates(&n,&a,r,r,0,T[0],ne[0],up+r,down+r,&cu,&cd),"steady channel rates");
        const DRCoefficient *dr=dr_lookup(14,r+2);
        down[r]+=dr?dr_alpha_eval(dr,T[0])*ne[0]:0;
    }
    double r0=(up[0]+2)/down[0],r1=(up[1]+2)/down[1];
    double norm=1+r0+r0*r1,eq[3]={1/norm,r0/norm,r0*r1/norm};
    CHECK(nlte_element_initial_steady_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_OK,"terminal steady actual solve");
    for(int i=0;i<3;i++)CHECK(near(ions[i],eq[i],2e-12),"stationary terminal analytic balance");
    CHECK(!nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd),"steady actual private core");
    for(int i=0;i<3;i++)CHECK(near(ions[i],eq[i],2e-12),"steady core did not pair rescale");
    double keep[3];memcpy(keep,ions,sizeof keep);
    p.physical_history=&h;
    CHECK(nlte_element_initial_steady_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK&&
          nlte_element_time_solve(&n,&a,&p,&op,0,0,2,&gd)!=POP_OK&&
          !memcmp(keep,ions,sizeof keep),"conflicting steady/history preserved");
    p.physical_history=NULL;
    n.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    CHECK(nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd)!=0&&
          !memcmp(keep,ions,sizeof keep),"steady invalid radiation rollback");
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;
    p.initial_steady_nlte=0;
    CHECK(nlte_time_core_selftest_solve(&n,&a,&p,&op,2,&gd)!=0&&
          !memcmp(keep,ions,sizeof keep),"v2 legacy pair route still refused");
    p.initial_steady_nlte=1;abund[0]=0;
    CHECK(nlte_element_initial_steady_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_OK&&
          ions[0]==0&&ions[1]==0&&ions[2]==0&&pops[0]==0&&pops[1]==0,"exact zero abundance");
    /* 반복된 정확한 영해는 행렬 조립 없이 유지한다. 시간해는 별도 경로다. */
    CHECK(nlte_element_initial_steady_solve(&n,&a,&p,&op,0,0,2,&gd)==POP_OK&&
          ions[0]==0&&ions[1]==0&&ions[2]==0&&pops[0]==0&&pops[1]==0,"already zero steady element");
    CHECK(T[0]==1e4,"initial prescribed T preserved");
    printf("[BF-CHANNELS] INITIAL_STEADY_PASS n0=%.17g n1=%.17g terminal=%.17g actual_core=1 history=NONE\n",keep[0],keep[1],keep[2]);
    printf("[BF-CHANNELS] TERMINAL_PASS n0=%.17g n1=%.17g terminal=%.17g history=PASS core=PASS continuum_bins=%d\n",savedion[0],savedion[1],savedion[2],checked);
    bf_set_nlte_pops(NULL);
    /* 실제 catalog g=6을 가진 고온 terminal은 미지 g=1 reservoir가 아니다.
     * 첫 물질 predictor의 Saha 경로를 실제로 호출한다. */
    T[0]=1e5;rho[0]=28*AMU*1e8;ne[0]=3e8;abund[0]=1;p.initial_steady_nlte=0;
    double bootstrap_tau=0;
    op.n_shells=1;op.electron_density=ne;op.t_electrons=T;op.tau_sobolev=&bootstrap_tau;
    CHECK(!lumina_bootstrap_window_open("terminal catalog bootstrap selftest"),"bootstrap window");
    int bootstrap=compute_plasma_state(&a,&p,&op,2);
    lumina_bootstrap_window_close();
    fprintf(stderr,"TERMINAL_BOOTSTRAP result=%d status=%s terminal=%.17g ne=%.17g partition=%.17g\n",
        bootstrap,population_status_name(p.population_last_error),ions[2],ne[0],partition[2]);
    CHECK(!bootstrap&&ions[2]/1e8>.99&&partition[2]==6,"known terminal catalog allowed at high ion fraction");
    memcpy(savedion,ions,sizeof savedion);double savedne=ne[0];
    a.topion_n=0;
    CHECK(compute_plasma_state(&a,&p,&op,2)!=0&&p.population_last_error==POP_ATOMIC_MISSING&&
        !memcmp(savedion,ions,sizeof savedion)&&ne[0]==savedne,"unknown terminal still rejected without mutation");
    a.topion_n=1;
    puts("[BF-CHANNELS] TERMINAL_BOOTSTRAP_PASS known_g=6 unknown=REJECTED");
    bf_opacity_free(&bf);gamma_deposition_free(&gd);free_bf_channels(a.bf_channels);
    return 0;
}

int main(void){
    setenv("LUMINA_PURE_CMFGEN","1",1);setenv("LUMINA_CMF_BF_MILNE","2",1);
    if(getenv("CMF_TEST_BF_CACHE_ONLY")){setenv("LUMINA_NLTE_COLL_ION","1",1);return integration_check();}
    if(kernel_check()||integration_check()||terminal_check())return 1;
    puts("[BF-CHANNELS] PASS synthetic_physics_only=1 actual_CMFGEN_distribution=NOT_TESTED");return 0;
}
