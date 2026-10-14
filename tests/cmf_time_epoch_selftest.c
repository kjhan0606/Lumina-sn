/* 시간 운전 순서/종료/과거 보존 시험. 수송·R7는 해석 가능한 affine 모형,
 * 침적의 epoch 확인은 stub이다. history/원자 내부에너지/partition은 실제
 * 코드다. SN 물리 시험 아님. */
#include "cmf_time_epoch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"TIME_EPOCH_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static int radiation_calls,material_calls,observer_calls,fault;
static const NLTEPhysicalHistory *expected_old;
static int rebuild(AtomicData *a,NLTEConfig *n,PlasmaState *p){
    PopulationAtomicView v={0};v.n_ions=a->n_ion_pops;v.n_levels=a->n_levels;
    v.level_offset=a->level_offset;v.energy_eV=a->level_energy_eV;v.g=a->level_g;
    v.level_Z=a->level_Z;v.level_ion=a->level_ion;
    n->population_committed_generation=a->population_committed_generation;
    return population_partition_build(&v,p->T_e,p->n_shells,a->population_committed_generation,
        p->T_e_generation,a->partition_functions,&a->partition_stamp)!=POP_OK;
}
GammaDepositionPublicationStatus __wrap_gamma_deposition_require(const GammaDeposition *g,double epoch){
    return g&&g->generation&&g->epoch==epoch?GAMMA_PUBLICATION_OK:GAMMA_PUBLICATION_STALE_EPOCH;
}
int __wrap_cmfgen_time_radiation_publish(CMFGENState *c,Geometry *g,OpacityState *o,BFOpacity *bf,
    PlasmaState *p,NLTEConfig *n,AtomicData *a,double T,int observer){
    (void)g;(void)o;(void)bf;(void)a;(void)T;
    radiation_calls++;
    if(p->physical_history!=expected_old||(fault==1&&radiation_calls==2)||(fault==4&&observer))return -1;
    CMFTimeRadiationTrial *t=c->time_trial;int ns=t->old->n_shells,nb=t->old->n_bins;
    if(!t->J)t->J=malloc((size_t)ns*nb*sizeof(double));
    if(!t->H)t->H=malloc((size_t)(ns+1)*nb*sizeof(double));
    if(!t->J||!t->H)return -1;
    for(int s=0;s<ns;s++)for(int b=0;b<nb;b++)t->J[s*nb+b]=(1+.1*b)*p->T_e[s]/1e4;
    for(int s=0;s<=ns;s++)for(int b=0;b<nb;b++)t->H[s*nb+b]=s?.1*t->J[(s-1)*nb+b]:t->inner_H[b];
    t->report.converged=1;n->radiation_field.field.generation.computed_generation++;
    c->common_obs_ready=observer;observer_calls+=observer;return 0;
}
int __wrap_lumina_r7_publish_and_solve_te(OpacityState *o,BFOpacity *bf,AtomicData *a,
    PlasmaState *p,NLTEConfig *n,GammaDeposition *gd,double epoch,int ns,int solve,const char *lane,int iter){
    (void)o;(void)bf;(void)gd;(void)epoch;(void)lane;(void)iter;
    material_calls++;if(!solve||p->physical_history!=expected_old||fault==2)return -1;
    for(int s=0;s<ns;s++){
        p->T_e[s]=fault==3?(p->T_e[s]<10000?11000:9000):.6*p->T_e[s]+4000;
        p->n_electron[s]=p->T_e[s]/20000;
        double upper=p->T_e[s]/40000;
        n->nlte_level_populations[s]=4-upper;n->nlte_level_populations[ns+s]=upper;
    }
    a->population_committed_generation++;p->T_e_generation++;
    p->te_publication.te_lane=A210_TE_LANE_FREE_T;p->te_publication.re_root_required=1;
    p->te_publication.committed_te_generation=p->T_e_generation;
    p->te_publication.population_generation=a->population_committed_generation;
    for(int s=0;s<ns;s++)p->te_publication.ledger[s].adiabatic_model=A210_MATERIAL_TIME_HISTORY;
    if(fault==5)p->te_publication.te_lane=A210_TE_LANE_FIXED_T;
    if(fault==6)p->te_publication.ledger[0].adiabatic_model=A210_ADIABATIC_CMFGEN_COMPLETE;
    return rebuild(a,n,p);
}
int main(void){
    enum{NS=2,NB=3};
    int Z[1]={1},stage[1]={0},offset[2]={0,2},lz[2]={1,1},li[2]={0,0},weights[2]={2,4},map[2]={0,1};
    double E[2]={0,2},rho[NS]={1e-12,2e-12},T[NS]={8000,12000},ne[NS]={.4,.6};
    double ions[NS]={4,4},levels[2*NS]={3.8,3.7,.2,.3},part[NS],ve[NS+1]={3e8,4e8,5e8};
    double oldj[NS*NB]={1,2,3,4,5,6},oldh[(NS+1)*NB]={0},inner[NB]={0};
    AtomicData a={0};a.n_ion_pops=1;a.n_levels=2;a.ion_pop_Z=Z;a.ion_pop_stage=stage;
    a.level_offset=offset;a.level_Z=lz;a.level_ion=li;a.level_g=weights;a.level_energy_eV=E;
    a.ion_number_density=ions;a.partition_functions=part;a.population_committed_generation=1;
    NLTEConfig n={0};n.n_nlte_levels_total=2;n.nlte_to_global_level=n.global_to_nlte_level=map;
    n.nlte_level_populations=levels;n.radiation_field.field.generation.computed_generation=9;
    PlasmaState p={0};p.n_shells=NS;p.rho=rho;p.T_e=T;p.n_electron=ne;p.T_e_generation=1;
    A210TermLedger ledger[NS]={0};p.te_publication.n_shells=NS;p.te_publication.ledger=ledger;
    REQUIRE(!rebuild(&a,&n,&p));
    CMFTimeRadiationHistory r={NS,NB,1e6,1e14,.1,ve,oldj,oldh};
    CMFPhysicalTimeState past={0},next={0};
    REQUIRE(!cmf_physical_time_state_capture(&past,&n,&a,&p,&r));
    REQUIRE(past.radiation.J!=oldj&&past.material.level_density!=levels);
    double past_levels[2*NS];memcpy(past_levels,past.material.level_density,sizeof past_levels);
    p.physical_history=&past.material;expected_old=&past.material;
    Geometry g={0};g.n_shells=NS;g.time_explosion=1.001e6;
    for(int s=0;s<NS;s++)rho[s]*=pow(1e6/g.time_explosion,3);
    OpacityState o={0};o.n_shells=NS;BFOpacity bf={0};GammaDeposition gamma={0};
    gamma.epoch=g.time_explosion;gamma.generation=1;
    CMFTimeRadiationTrial trial={0};trial.old=&past.radiation;trial.epoch_s=g.time_explosion;trial.inner_H=inner;
    CMFGENState c={0};c.n_shells=NS;c.time_trial=&trial;
    CMFTimeEpochReport report;
    REQUIRE(cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-9,&next,&report)==CMF_EPOCH_OK);
    REQUIRE(report.converged&&report.iterations>2&&material_calls==report.iterations);
    REQUIRE(observer_calls==1&&radiation_calls==material_calls+2&&report.radiation_calls==radiation_calls);
    REQUIRE(report.radiation_generation==9+(uint64_t)radiation_calls&&c.common_obs_ready);
    REQUIRE(fabs(T[0]/1e4-1)<2e-9&&fabs(T[1]/1e4-1)<2e-9);
    REQUIRE(next.material.energy.epoch_s==g.time_explosion&&next.radiation.epoch_s==g.time_explosion);
    REQUIRE(!memcmp(past.material.level_density,past_levels,sizeof past_levels));
    REQUIRE(!memcmp(past.radiation.J,oldj,sizeof oldj)&&p.physical_history==expected_old);
    printf("TIME_EPOCH_COUPLED_SEQUENCE iterations=%d T=%.17g/%.17g generation=%llu snapshot_deep=1 past_immutable=1\n",
        report.iterations,T[0],T[1],(unsigned long long)report.radiation_generation);
    double *saved=next.radiation_storage;double nextj[NS*NB];memcpy(nextj,next.radiation.J,sizeof nextj);
    for(fault=1;fault<=6;fault++){
        radiation_calls=material_calls=observer_calls=0;
        CMFTimeEpochStatus status=cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,2,1e-9,&next,&report);
        REQUIRE(status==((fault==2||fault>=5)?CMF_EPOCH_MATERIAL:fault==3?CMF_EPOCH_NOT_CONVERGED:CMF_EPOCH_RADIATION));
        REQUIRE(!report.converged&&!c.common_obs_ready&&saved==next.radiation_storage);
        REQUIRE(!memcmp(nextj,next.radiation.J,sizeof nextj)&&!memcmp(oldj,past.radiation.J,sizeof oldj));
        T[0]=T[1]=1e4;ne[0]=ne[1]=.5;levels[0]=levels[1]=3.75;levels[2]=levels[3]=.25;
        REQUIRE(!rebuild(&a,&n,&p));
    }
    fault=0;
    REQUIRE(cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-9,&past,&report)==CMF_EPOCH_INVALID);
    past.material.energy.epoch_s-=1;
    REQUIRE(cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-9,&next,&report)==CMF_EPOCH_INVALID);
    past.material.energy.epoch_s+=1;rho[0]*=1.01;
    REQUIRE(cmfgen_solve_time_epoch(&c,&g,&o,&bf,&p,&n,&a,&gamma,1e4,64,1e-9,&next,&report)==CMF_EPOCH_INVALID);
    cmf_time_radiation_trial_free(&trial);cmf_physical_time_state_free(&past);cmf_physical_time_state_free(&next);
    puts("CMF_TIME_EPOCH_SELFTEST_PASS synthetic_phase_callbacks=1 actual_history_energy=1");return 0;
}
