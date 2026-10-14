/* 실제 private A210 선 에너지 경로. 합성 한 선이며 전체 물질 root는 아니다. */
#include "cmf_time_epoch.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
extern RadeqStatus cmf_time_line_energy_selftest(const NLTEPopulationCandidate*,double,
    A210LineNetPublication*,A210LineNetShell*);
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"TIME_LINE_ENERGY_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
int main(void){
    NLTEPopulationCandidate c={0};c.active=c.bundle_complete=1;c.n_shells=1;
    double T=1e4,ne=1,natom=1,nu=2*EV_TO_ERG/H_PLANCK,tau=-2,err=0,J=1e-6;
    double edges[2]={nu/2,nu*2},A=1e8,B=1e10,f=.5,lambda=C_SPEED_OF_LIGHT/nu;
    int Z=14,stage=1,lower=0,upper=1,offset[2]={0,2},elemoff[2]={0,1},lz[2]={14,14},li[2]={1,1};
    int weights[2]={2,4},num[2]={0,1},sl[2]={0,1},map[2]={0,1},line_map=0;
    double E[2]={0,2},pops[2]={.75,.25},ions=1,partition,abundance=1,mass=28;
    uint64_t line_id=0,count=0;LineJbarValidityState valid=LINE_JBAR_VALID;
    A208Validity tv=A208_VALID;
    c.plasma.n_shells=1;c.plasma.T_e=&T;c.plasma.n_electron=&ne;
    NLTEPhysicalHistory past={0};past.energy.epoch_s=1e6;c.plasma.physical_history=&past;
    c.trial_te=&T;c.n_atom=&natom;
    AtomicData *a=&c.atom;NLTEConfig *n=&c.nlte;OpacityState *o=&c.opacity;
    a->n_lines=1;a->line_atomic_number=&Z;a->line_ion_number=&stage;
    a->line_level_lower=&lower;a->line_level_upper=&upper;a->line_A_ul=&A;a->line_B_lu=&B;
    a->line_f_lu=&f;a->line_wavelength_cm=&lambda;a->line_nu=&nu;
    a->n_ion_pops=1;a->ion_pop_Z=&Z;a->ion_pop_stage=&stage;a->level_offset=offset;
    a->n_levels=2;a->level_Z=lz;a->level_ion=li;a->level_g=weights;a->level_num=num;
    a->level_super=sl;a->level_energy_eV=E;a->ion_number_density=&ions;a->partition_functions=&partition;
    a->n_elements=1;a->element_Z=&Z;a->elem_ion_offset=elemoff;a->abundances=&abundance;a->element_mass_amu=&mass;
    a->population_committed_generation=1;
    BfChannels channels={0};channels.version=2;a->bf_channels=&channels;
    PopulationAtomicView av={0};av.n_ions=1;av.n_levels=2;av.level_offset=offset;av.energy_eV=E;
    av.g=weights;av.level_Z=lz;av.level_ion=li;
    REQUIRE(population_partition_build(&av,&T,1,1,1,&partition,&a->partition_stamp)==POP_OK);
    n->n_nlte_ions=1;n->n_nlte_levels_total=2;n->nlte_line_map=&line_map;
    n->global_to_nlte_level=map;n->nlte_level_populations=pops;n->population_committed_generation=1;
    n->line_energy_view_status=LINE_JBAR_VIEW_OK;n->radfield_view.generation=4;
    LineJbarView *v=&n->line_energy_view;v->n_lines=v->cache_n_lines=v->n_shells=1;
    v->line_id=&line_id;v->jbar=&J;v->se=&err;v->count=&count;v->validity=&valid;
    v->cache_set_kind=LINE_JBAR_SET_ENERGY_DOMAIN;v->generation=4;
    v->statistic_kind=RADIATION_FIELD_DETERMINISTIC;v->uncertainty_kind=LINE_JBAR_UNCERTAINTY_ITERATION_DELTA;
    o->n_lines=o->n_shells=1;o->line_list_nu=&nu;o->tau_sobolev=&tau;o->tau_validity=&tv;
    o->cpu_opacity.n_shells=o->cpu_opacity.n_bins=1;o->cpu_opacity.frequency_edges=edges;
    o->cpu_opacity.radiation_generation=4;o->jbar_line_det_error_upper=&err;
    o->jbar_line_det_operator=CMF_FINE_LINE_OPERATOR_COMMON_GAUSSIAN;
    o->jbar_line_det_time_old=1e6;o->jbar_line_det_time_new=1.001e6;
    o->jbar_line_det_time_equation_residual=1e-15;o->jbar_line_det_time_closure_update=1e-10;
    o->jbar_line_det_exact_iterations=3;o->jbar_line_det_exact_iteration_cap=32;
    o->jbar_line_det_exact_tolerance=1e-8;o->jbar_line_det_exact_residual=1e-10;
    o->jbar_line_det_grid_n_bins=2;o->jbar_line_det_grid_nu_min=edges[0];o->jbar_line_det_grid_nu_max=edges[1];
    A210LineNetPublication pub={0};A210LineNetShell result={0};
    RadeqStatus status=cmf_time_line_energy_selftest(&c,1.001e6,&pub,&result);
    REQUIRE(status==RADEQ_OK&&result.eligible_cells==1&&result.srce_chk_cells==0);
    REQUIRE(pub.uncertainty_kind==LINE_JBAR_UNCERTAINTY_ITERATION_DELTA);
    LineNetSobolevMaterial raw;double scale;
    REQUIRE(!line_net_sobolev_material(.25,A,nu,tau,1.001e6,LINE_NET_NEGATIVE_OPACITY_NONE,1,&raw));
    REQUIRE(!line_net_cmfgen_scl_ln(0,2,nu,1,.5,1e30,&scale));
    double expected=4*M_PI_VAL*scale*(raw.emission_per_sr-raw.raw_integrated_opacity*J);
    REQUIRE(fabs(result.signed_rate/expected-1)<1e-13);
    printf("TIME_ACTUAL_LINE_ENERGY signed_rate=%.17g expected=%.17g raw_negative_tau=%.17g srce_chk_cells=%llu\n",
        result.signed_rate,expected,tau,(unsigned long long)result.srce_chk_cells);
    o->jbar_line_det_time_equation_residual=1e-2;
    REQUIRE(cmf_time_line_energy_selftest(&c,1.001e6,&pub,&result)==RADEQ_STALE_LINE);
    o->jbar_line_det_time_equation_residual=1e-15;past.energy.epoch_s-=1;
    REQUIRE(cmf_time_line_energy_selftest(&c,1.001e6,&pub,&result)==RADEQ_STALE_LINE);
    past.energy.epoch_s+=1;v->uncertainty_kind=LINE_JBAR_UNCERTAINTY_LEGACY;
    REQUIRE(cmf_time_line_energy_selftest(&c,1.001e6,&pub,&result)==RADEQ_STALE_LINE);
    puts("CMF_TIME_LINE_ENERGY_SELFTEST_PASS");return 0;
}
