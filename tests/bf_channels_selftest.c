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
    uint32_t header[5]={0x43424643,1,(uint32_t)c->n_levels,(uint32_t)c->n_channels,(uint32_t)c->n_bins};
    double edges[2]={c->nu_min,c->nu_max},threshold[2]={c->threshold[0],c->threshold[1]*(bad?1.01:1)};
    int ok=fwrite(header,4,5,fp)==5&&fwrite(edges,8,2,fp)==2&&
           fwrite(c->offset,4,c->n_levels+1,fp)==(size_t)c->n_levels+1&&
           fwrite(c->upper,4,2,fp)==2&&fwrite(threshold,8,2,fp)==2&&
           fwrite(c->sigma,8,2*c->n_bins,fp)==(size_t)2*c->n_bins;
    return fclose(fp)||!ok?-1:0;
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
    CHECK(!write_channel_file("channels-invalid.bin",&c,1)&&load_bf_channels(&a,"channels-invalid.bin")&&
          a.bf_channels==loaded,"loader rejects wrong threshold without replacing prior data");
    double up,down,cu,cd;
    CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&up,&down,&cu,&cd)&&up==0&&down==0,"A zero channel");
    CHECK(!nlte_bf_channel_rates(&n,&a,0,1,0,T[0],ne[0],&up,&down,&cu,&cd)&&up>0&&down>0&&cu==0&&cd==0,"B excited parent rates");
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
    printf("[BF-CHANNELS] INTEGRATION_PASS up=%.17g down=%.17g excited_after=%.17g continuum_bins=%d\n",up,down,saved[2],checked);
    bf_set_nlte_pops(NULL);bf_opacity_free(&bf);gamma_deposition_free(&gd);free_bf_channels(a.bf_channels);
    return 0;
}

int main(void){
    setenv("LUMINA_PURE_CMFGEN","1",1);setenv("LUMINA_CMF_BF_MILNE","2",1);
    if(kernel_check()||integration_check())return 1;
    puts("[BF-CHANNELS] PASS synthetic_physics_only=1 actual_CMFGEN_distribution=NOT_TESTED");return 0;
}
