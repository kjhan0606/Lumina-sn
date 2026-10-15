/* 실제 Si II/III 행렬: PHOT 공명 보존과 단방향 총계수 제외를 분리한다.
 * 합성 단면적이며 실제 Si 공명의 완전성/공개 스펙트럼 시험은 아니다. */
#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"RECOMB_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static int near(double a,double b,double tol){return fabs(a-b)<=tol*fmax(fabs(a),fabs(b));}
int main(void){
    enum{NB=NLTE_N_FREQ_BINS};
    int thermal=getenv("LUMINA_NLTE_BF_JEQB")&&atoi(getenv("LUMINA_NLTE_BF_JEQB"));
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState o={0};
    int Z[2]={14,14},stage[2]={1,2},offset[3]={0,1,2},map[2]={0,1};
    int num[2]={0,0},g[2]={2,1},el[1]={14},eloff[2]={0,2};
    int coff[3]={0,1,1},upper[1]={1},uion[1]={1};double gu[1]={1};
    double mass[1]={28},abund[1]={1},chi[1]={16.35},E[2]={0,0};
    double T[1]={1e4},ne[1]={1e8},rho[1]={28*AMU};
    double ratio=ne[0]*pow(H_PLANCK*H_PLANCK/(2*M_PI_VAL*M_ELECTRON*K_BOLTZMANN*T[0]),1.5)*
        exp(chi[0]*EV_TO_ERG/(K_BOLTZMANN*T[0]));
    double pop[2]={ratio/(1+ratio),1/(1+ratio)},part[2]={2,1},within[2]={1,1};
    double edges[NB+1],J[NB],sigma[NB],threshold[1]={chi[0]*EV_TO_ERG/H_PLANCK};
    RadiationFieldValidityState valid[NB];
    double dlog=log(NLTE_NU_MAX/NLTE_NU_MIN)/NB;
    for(int b=0;b<=NB;b++)edges[b]=NLTE_NU_MIN*exp(b*dlog);
    for(int b=0;b<NB;b++){
        J[b]=1e-8;valid[b]=RADIATION_FIELD_VALID;
        double x=sqrt(edges[b]*edges[b+1])/threshold[0];
        sigma[b]=edges[b+1]>threshold[0]?1e-19+2e-17*exp(-pow((x-1.08)/.015,2)):0;
    }
    BfChannels c={2,1,NB,NLTE_NU_MIN,NLTE_NU_MAX,coff,upper,threshold,sigma,2,uion,gu};
    n.enabled=1;n.n_nlte_ions=2;n.n_nlte_levels_total=n.n_super_total=2;
    n.nlte_Z[0]=n.nlte_Z[1]=14;n.nlte_ion[0]=1;n.nlte_ion[1]=2;
    for(int i=0;i<3;i++)n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=offset[i];
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.within_sl_frac=within;n.nlte_level_populations=pop;n.n_freq_bins=NB;
    n.nu_min=NLTE_NU_MIN;n.nu_max=NLTE_NU_MAX;n.d_log_nu=dlog;n.J_nu=J;
    n.radfield_view=(RadiationFieldView){.n_shells=1,.n_bins=NB,
        .frequency_bin_edges=edges,.J_nu=J,.validity=valid,.generation=1};
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;
    n.exclude_empirical_heavy_ce=1;n.solve_effect_policy_explicit=1;
    a.n_ion_pops=2;a.n_levels=2;a.ion_pop_Z=Z;a.ion_pop_stage=stage;a.level_offset=offset;
    a.level_Z=Z;a.level_ion=stage;a.level_num=num;a.level_g=g;a.level_energy_eV=E;
    a.ion_number_density=pop;a.partition_functions=part;a.bf_channels=&c;
    a.n_elements=1;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abund;
    a.n_ionization=1;a.ioniz_Z=el;a.ioniz_ion=stage;a.ioniz_energy_eV=chi;
    p.n_shells=1;p.T_e=T;p.n_electron=ne;p.rho=rho;
    NLTEPhysicalHistory history={0};p.physical_history=&history;
    GammaDeposition gamma={0};double heat[1]={0};gamma_deposition_init(&gamma,1);
    CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,1,NULL,NULL,NULL,heat));
    double up,down,cu,cd;
    CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&up,&down,&cu,&cd));
    CHECK(up>0&&down>0&&cu==0&&cd==0);
    double A[2][4]={{0}},b[2]={0};
    for(int owned=0;owned<=1;owned++){
        n.channel_owned_recombination=owned;
        CHECK(!nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1,A[owned],b,2,&gamma,NULL,-1));
        CHECK(near(A[owned][1],up,1e-13));
        for(int col=0;col<2;col++)CHECK(near(A[owned][2*col],-A[owned][2*col+1],1e-13));
    }
    CHECK(A[0][2]>A[1][2]&&near(A[1][2],down,1e-13)&&A[0][1]==A[1][1]);
    double residual=A[1][0]*pop[0]+A[1][2]*pop[1];
    double scale=up*pop[0]+down*pop[1];
    double old_residual=A[0][0]*pop[0]+A[0][2]*pop[1];
    if(thermal){CHECK(fabs(residual)<1e-12*scale&&old_residual>1e-3*scale);}
    else{
        CHECK(fabs(residual)>1e-3*scale);
        for(int ib=0;ib<NB;ib++)J[ib]*=2;
        double up2,down2;
        CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&up2,&down2,&cu,&cd));
        CHECK(near(up2,2*up,1e-13)&&down2>down&&down2<2*down);
    }
    /* 공명을 없앤 반사실은 율을 낮춰야 한다. 생산 단면적은 변경하지 않는다. */
    for(int ib=0;ib<NB;ib++)sigma[ib]=edges[ib+1]>threshold[0]?1e-19:0;
    double flatup,flatdown;
    CHECK(!nlte_bf_channel_rates(&n,&a,0,0,0,T[0],ne[0],&flatup,&flatdown,&cu,&cd));
    CHECK(flatdown<down);
    double saved[4];memcpy(saved,A[1],sizeof(saved));a.bf_channels=NULL;
    CHECK(nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1,A[1],b,2,&gamma,NULL,-1)!=0&&
        !memcmp(saved,A[1],sizeof(saved)));
    printf("RECOMB_CHANNEL_PASS thermal=%d up=%.17g retained_recomb=%.17g legacy_addition=%.17g thermal_residual=%.17g legacy_residual=%.17g normalized_residual=%.17g resonance_retained=1 missing_pack_rejected=1 SN_comparison=0\n",
        thermal,up,down,A[0][2]-down,residual,old_residual,fabs(residual)/scale);
    gamma_deposition_free(&gamma);return 0;
}
