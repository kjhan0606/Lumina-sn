#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c,m) do{if(!(c)){fprintf(stderr,"[POP-TIME-ASSEMBLY][FAIL] %s line=%d\n",m,__LINE__);return 1;}}while(0)
extern int nlte_time_pair_selftest_solve(NLTEConfig*,AtomicData*,PlasmaState*,OpacityState*,double,GammaDeposition*);

int main(void)
{
    /* 실제 전이율 조립의 비열 이온화 단일채널 확인. 광자는 H edge 아래,
     * 선과 충돌은 없는 합성 원자다. 실제 Si/S 복합 원자의 검증은 아니다. */
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState op={0};
    int Z[2]={1,1},stage[2]={0,1},offset[3]={0,1,2},map[2]={0,1},g[2]={2,1},num[2]={0,0};
    int el[1]={1},eloff[2]={0,2},iz[1]={1},ist[1]={0};
    double mass[1]={1},abundance[1]={1},chi[1]={13.6},energy[2]={0,0};
    double rho[1]={AMU},oldrho[1]={8*AMU},T[1]={1e4},ne[1]={.5};
    double population[2]={1,0},ion_population[2]={1,0},past[2]={8,0},within[2]={1,1};
    n.n_nlte_ions=2;n.n_nlte_levels_total=2;n.n_super_total=2;
    n.nlte_Z[0]=n.nlte_Z[1]=1;n.nlte_ion[0]=0;n.nlte_ion[1]=1;
    n.nlte_ion_level_offset[0]=n.nlte_ion_super_offset[0]=0;
    n.nlte_ion_level_offset[1]=n.nlte_ion_super_offset[1]=1;
    n.nlte_ion_level_offset[2]=n.nlte_ion_super_offset[2]=2;
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.within_sl_frac=within;n.nlte_level_populations=population;n.nu_max=1e15;n.nu_min=1e14;
    a.n_ion_pops=2;a.n_levels=2;a.ion_pop_Z=Z;a.ion_pop_stage=stage;
    a.level_offset=offset;a.level_energy_eV=energy;a.level_g=g;a.level_num=num;
    a.level_Z=Z;a.level_ion=stage;a.ion_number_density=ion_population;
    a.n_elements=1;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abundance;
    a.n_ionization=1;a.ioniz_Z=iz;a.ioniz_ion=ist;a.ioniz_energy_eV=chi;
    p.n_shells=1;p.rho=rho;p.T_e=T;p.n_electron=ne;
    NLTEPhysicalHistory h={0};h.energy.n_shells=1;h.energy.epoch_s=1;
    h.energy.density_g_cm3=oldrho;h.n_levels=2;h.n_ions=2;
    h.level_density=past;h.level_global=map;h.ion_Z=Z;h.ion_stage=stage;p.physical_history=&h;
    GammaDeposition gd={0};gamma_deposition_init(&gd,1);
    double heating[1]={2*35*EV_TO_ERG/.05};
    CHECK(!gamma_deposition_publish(&gd,GAMMA_PROVENANCE_EXTERNAL_FILE,2,NULL,NULL,NULL,heating),"deposition producer");
    double A[4]={0},b[2]={0},x[2]={0};PopulationGeneratorSolveDiagnostic d;
    CHECK(!nlte_assemble_rate_matrix(&n,&a,&p,&op,0,1,0,2,A,b,2,&gd,NULL,-1),"actual raw rate assembly");
    CHECK(fabs(A[0]+2)<1e-14 && fabs(A[1]-2)<1e-14 && A[2]==0 && A[3]==0 && b[0]==0 && b[1]==0,
          "raw channel was replaced by steady closure or synthetic age source");
    CHECK(nlte_physical_history_solve_levels(&n,&a,&p,0,2,0,2,A,x,&d)==POP_OK,"assembled rate time step");
    CHECK(fabs(x[0]-1./3)<1e-14 && fabs(x[1]-2./3)<1e-14,"nonthermal evolution analytic solution");
    CHECK(past[0]==8 && past[1]==0 && population[0]==1 && population[1]==0,"trial preserves public and past");
    CHECK(!nlte_time_pair_selftest_solve(&n,&a,&p,&op,2,&gd),"actual pair owner time solve");
    CHECK(fabs(population[0]-x[0])<1e-14 && fabs(population[1]-x[1])<1e-14 &&
          ion_population[0]==1 && ion_population[1]==0 && past[0]==8 && past[1]==0,
          "pair owner reset time solution to steady ion totals");
    printf("[POP-TIME-ASSEMBLY] PASS actual_NT_rate=%.17g neutral=%.17g ionized=%.17g steady_solution_used=0 pair_owner=PASS\n",A[1],x[0],x[1]);
    gamma_deposition_free(&gd);return 0;
}
