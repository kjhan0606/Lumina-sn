/* 한 CE 반응의 두 원소 전하 변화는 상쇄돼야 한다.
 * 실제 행렬 조립을 사용하고 외부 전리/재결합은 영으로 둔다. */
#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"CE_CHARGE_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
int main(void){
    setenv("LUMINA_DR_BOOST","0",1);
    enum{NI=4,NL=6,NS=1};
    int iz[NI]={14,14,20,20},ist[NI]={1,2,1,2},off[NI+1]={0,1,3,4,6};
    int lz[NL]={14,14,14,20,20,20},ls[NL]={1,2,2,1,2,2};
    int map[NL]={0,1,2,3,4,5},num[NL]={0,0,1,0,0,1},weight[NL]={1,1,1,1,1,1};
    int el[2]={14,20},eloff[3]={0,2,4},qi[2]={1,1};
    double E[NL]={0,0,2,0,0,3},mass[2]={28,40},abund[2]={.5,.5},chi[2]={16.35,11.87};
    double T[NS]={1e4},ne[NS]={0},rho[NS]={AMU};
    double pop[NL]={2,3,5,7,11,13},ions[NI]={2,8,7,24},part[NI]={1,1,1,1},within[NL]={1,1,1,1,1,1};
    NLTEConfig n={0};AtomicData a={0};PlasmaState p={0};OpacityState o={0};
    n.enabled=1;n.n_nlte_ions=NI;n.n_nlte_levels_total=n.n_super_total=NL;
    for(int i=0;i<NI;i++){n.nlte_Z[i]=iz[i];n.nlte_ion[i]=ist[i];}
    for(int i=0;i<=NI;i++)n.nlte_ion_level_offset[i]=n.nlte_ion_super_offset[i]=off[i];
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=n.super_anchor_global=map;
    n.nlte_level_populations=pop;n.within_sl_frac=within;n.nu_min=1e14;n.nu_max=1e15;
    n.solve_effect_policy_explicit=1;
    a.n_ion_pops=NI;a.n_levels=NL;a.ion_pop_Z=iz;a.ion_pop_stage=ist;a.level_offset=off;
    a.level_Z=lz;a.level_ion=ls;a.level_num=num;a.level_g=weight;a.level_energy_eV=E;
    a.ion_number_density=ions;a.partition_functions=part;
    a.n_elements=2;a.element_Z=el;a.elem_ion_offset=eloff;a.element_mass_amu=mass;a.abundances=abund;
    a.n_ionization=2;a.ioniz_Z=el;a.ioniz_ion=qi;a.ioniz_energy_eV=chi;
    p.n_shells=NS;p.T_e=T;p.n_electron=ne;p.rho=rho;
    /* 실제 과거 소유 모드는 조립의 정상 정규화행/나이 근사를 배제한다.
     * 이 시험은 시간해를 만들지 않고 raw generator만 읽는다. */
    NLTEPhysicalHistory history={0};p.physical_history=&history;
    double flux[2]={0};
    for(int e=0;e<2;e++){
        double A[9]={0},b[3]={0};int lo=2*e,base=off[lo];
        CHECK(!nlte_assemble_rate_matrix(&n,&a,&p,&o,lo,lo+1,0,1,A,b,3,NULL,NULL,-1));
        double atoms=0;
        for(int row=0;row<3;row++){
            double rate=0;
            for(int col=0;col<3;col++)rate+=A[3*col+row]*pop[base+col];
            atoms+=rate;flux[e]+=rate*ls[base+row];
        }
        CHECK(fabs(atoms)<1e-22);
    }
    double scale=fabs(flux[0])+fabs(flux[1]);
    printf("CE_CHARGE_RESULT Si_charge_rate=%.17g Ca_charge_rate=%.17g net=%.17g relative=%.17g excited_upper=1 actual_assembly=1\n",
        flux[0],flux[1],flux[0]+flux[1],fabs(flux[0]+flux[1])/scale);
    CHECK(scale>0&&isfinite(scale));
    CHECK(fabs(flux[0]+flux[1])<1e-12*scale);
    puts("CE_CHARGE_PASS");return 0;
}
