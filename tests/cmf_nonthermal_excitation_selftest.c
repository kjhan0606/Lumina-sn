#include "cmf_nonthermal_excitation.h"
#include <assert.h>
#include <stdio.h>
int main(void){
    AtomicData a={0};int offsets[]={0,3,5,5},Z[]={14,14,14,14,14},q[]={1,1,1,2,2};
    int numbers[]={0,1,2,0,1},g[]={2,4,6,1,3},ionz[]={14,14,14},ionq[]={1,2,3};
    double energy[]={0,.01,4,0,2};
    int linez[]={14,14,14},lineq[]={1,1,2},lo[]={0,0,0},hi[]={1,2,1};
    double f[]={.01,.2,.3},A[]={10,2000,1e4};
    int clo[]={0,0},chi[]={1,2};double temps[]={1000,1e5},omega[]={1,3,2,4};
    a.n_levels=5;a.n_ion_pops=3;a.n_lines=3;
    a.level_offset=offsets;a.level_Z=Z;a.level_ion=q;a.level_num=numbers;
    a.level_energy_eV=energy;a.level_g=g;a.ion_pop_Z=ionz;a.ion_pop_stage=ionq;
    a.line_atomic_number=linez;a.line_ion_number=lineq;a.line_level_lower=lo;
    a.line_level_upper=hi;a.line_f_lu=f;a.line_A_ul=A;
    a.ncol_ions=1;a.col_ion_Z[0]=14;a.col_ion_stage[0]=1;
    a.col_ion_n_trans[0]=2;a.col_ion_n_temp[0]=2;a.col_ion_tgrid[0]=temps;
    a.col_ion_lo[0]=clo;a.col_ion_hi[0]=chi;a.col_ion_omega[0]=omega;
    double E[]={.1,1,10,100};CMFNonthermalExcitations x={0};
    assert(!cmf_nonthermal_excitation_build(&a,E,4,1,&x));
    assert(x.view.n_channels==3&&x.n_omega==1&&x.n_bethe==2&&x.n_subthermal==1);
    assert(x.transitions[0].lower_global==0&&x.transitions[0].upper_global==1);
    assert(x.transitions[2].lower_global==3&&x.transitions[2].upper_global==4);
    assert(x.sigma[3]==cmf_nonthermal_omega(100,.01,3,2));
    assert(x.sigma[7]==cmf_nonthermal_bethe(100,4,.2,1));
    assert(x.sigma[11]==cmf_nonthermal_bethe(100,2,.3,2));
    CMFNonthermalExcitations saved=x;
    omega[3]=-1;assert(cmf_nonthermal_excitation_build(&a,E,4,1,&x)!=0);
    assert(!memcmp(&saved,&x,sizeof x));omega[3]=4;
    numbers[4]=3;assert(cmf_nonthermal_excitation_build(&a,E,4,1,&x)!=0);
    assert(!memcmp(&saved,&x,sizeof x));numbers[4]=1;
    cmf_nonthermal_excitation_free(&x);
    puts("CMF_NONTHERMAL_EXCITATION_SELFTEST_OK local_IDs=1 Omega_highest_T=1 A_selection=1 rollback=1");
    return 0;
}
