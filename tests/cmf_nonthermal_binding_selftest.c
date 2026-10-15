#include "cmf_nonthermal_binding.h"
#include <assert.h>
int main(int argc,char **argv){
    if(argc!=2)return 2;
    AtomicData a={0};int offsets[]={0,1,2,2},Z[]={14,14},q[]={1,2},ionz[]={14,14,14},ionq[]={1,2,3};
    double energy[]={0,0},gaps[]={5,7};
    a.n_levels=2;a.n_ion_pops=3;a.level_offset=offsets;a.level_Z=Z;a.level_ion=q;a.level_energy_eV=energy;
    a.ion_pop_Z=ionz;a.ion_pop_stage=ionq;a.n_ionization=2;a.ioniz_Z=Z;a.ioniz_ion=q;a.ioniz_energy_eV=gaps;
    double E[]={1,10,100};CMFNonthermalRoute empty={0};CMFNonthermalTransition target={0};
    CMFNonthermalExcitations ex={0};ex.energy=E;ex.channels=&empty;ex.transitions=&target;
    ex.view=(CMFNonthermalBoundModel){2,3,3,0,E,&target,&empty};
    CMFNonthermalAtomicRoute rows[2]={{0}};
    rows[0].Z=rows[1].Z=14;rows[0].charge=rows[1].charge=1;
    rows[0].source_line=1;rows[1].source_line=2;
    rows[0].threshold_eV=5;rows[1].threshold_eV=20;rows[0].abcd[0]=rows[1].abcd[0]=1;
    CMFNonthermalAtomicData raw={2,rows};CMFNonthermalResolvedModel m={0};
    assert(!cmf_nonthermal_binding_build(&a,&raw,&ex,argv[1],&m));
    assert(m.n_ion_channels==2&&m.n_resolved_routes==2&&m.n_unresolved_routes==1&&m.n_missing_ions==1);
    assert(m.transitions[1].upper_global==-1&&m.transitions[1].upper_ion==2);
    assert(m.ion_sigma[1]==cmf_nonthermal_atomic_sigma(&rows[0],10));
    assert(m.channels[1].auger_electron_eV==8);
    CMFNonthermalResolvedModel saved=m;
    gaps[1]=8;assert(cmf_nonthermal_binding_build(&a,&raw,&ex,argv[1],&m)!=0);
    assert(!memcmp(&saved,&m,sizeof m));gaps[1]=7;
    rows[1].threshold_eV=NAN;assert(cmf_nonthermal_binding_build(&a,&raw,&ex,argv[1],&m)!=0);
    assert(!memcmp(&saved,&m,sizeof m));rows[1].threshold_eV=20;
    cmf_nonthermal_binding_free(&m);
    puts("CMF_NONTHERMAL_BINDING_SELFTEST_OK single=1 Auger_terminal=1 partial_model_explicit=1 rollback=1");return 0;
}
