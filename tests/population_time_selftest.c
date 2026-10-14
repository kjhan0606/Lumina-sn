#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c,m) do{if(!(c)){fprintf(stderr,"[POP-TIME][FAIL] %s line=%d\n",m,__LINE__);return 1;}}while(0)

static int core_check(void)
{
    double A[4]={-2,2,3,-3},old[2]={7,3},out[2]={-1,-1};
    PopulationGeneratorSolveDiagnostic d;
    CHECK(population_generator_implicit_step(A,2,old,.8,.4,out,&d)==POP_OK,"two-state step");
    double expected=.8*(old[1]+.4*2*(old[0]+old[1]))/(1+.4*5);
    CHECK(fabs(out[1]-expected)<1e-14 && fabs(out[0]+out[1]-8)<1e-14,"analytic two-state");
    printf("[POP-TIME] analytic_n_upper=%.17g computed=%.17g residual=%.17g\n",expected,out[1],d.exact_generator_componentwise_residual);
    double error_prev=INFINITY;
    for(int steps=20;steps<=80;steps*=2){
        double n[2]={10,0};
        for(int k=0;k<steps;k++)
            CHECK(population_generator_implicit_step(A,2,n,1,1.0/steps,n,&d)==POP_OK,"time convergence");
        double exact=4*(1-exp(-5.)),error=fabs(n[1]-exact);
        CHECK(error<error_prev && (steps==20 || error_prev/error>1.8),"first-order convergence");
        error_prev=error;
        printf("[POP-TIME] steps=%d upper=%.17g analytic=%.17g error=%.17g\n",steps,n[1],exact,error);
    }
    double zero[9]={0},isolated[3]={0,7,2},x[3]={-1,-1,-1};
    CHECK(population_generator_implicit_step(zero,3,isolated,.125,1e9,x,&d)==POP_OK &&
          x[0]==0 && x[1]==.875 && x[2]==.25,"zero rates freeze plus dilution");
    double one_way[4]={-1,1,0,0},start[2]={8,0};
    CHECK(population_generator_implicit_step(one_way,2,start,1,2,out,&d)==POP_OK &&
          fabs(out[0]-8./3)<1e-15 && fabs(out[1]-16./3)<1e-15,"reducible generator");
    double stiff[9]={-1e30,1e30,0,2e30,-2e30,0,0,0,0},mem[3]={9,0,9};
    CHECK(population_generator_implicit_step(stiff,3,mem,.5,1e12,x,&d)==POP_OK &&
          fabs(x[0]-3)<1e-14 && fabs(x[1]-1.5)<1e-14 && x[2]==4.5,"stiff rate retains finite memory leak");
    double saved[2];memcpy(saved,out,sizeof(out));
    A[0]=-1;
    CHECK(population_generator_implicit_step(A,2,old,1,1,out,&d)!=POP_OK &&
          !memcmp(saved,out,sizeof(out)),"bad generator atomic failure");
    A[0]=-2;A[1]=-2;
    CHECK(population_generator_implicit_step(A,2,old,1,1,out,&d)!=POP_OK,"negative rate");
    A[1]=NAN;
    CHECK(population_generator_implicit_step(A,2,old,1,1,out,&d)==POP_NONFINITE,"nonfinite rate");
    CHECK(population_generator_implicit_step(zero,2,old,1,0,out,&d)!=POP_OK,"zero dt");
    return 0;
}

static int history_check(void)
{
    /* 과거 full-level 인구를 현재 super-level에 합산한다. 현재 반복 인구와
     * 현재 SL 내부 Boltzmann 분율을 과거 source로 재사용하지 않는다. */
    double rho_old[1]={8},rho[1]={1},levels[3]={2,4,2};
    int globals[3]={0,1,2},mapping[3]={0,0,1},Z[2]={1,1},stage[2]={0,1};
    NLTEPhysicalHistory h={0};h.energy.n_shells=1;h.energy.epoch_s=1;
    h.energy.density_g_cm3=rho_old;h.n_levels=3;h.n_ions=2;
    h.level_density=levels;h.level_global=globals;h.ion_Z=Z;h.ion_stage=stage;
    NLTEConfig n={0};n.n_nlte_levels_total=3;n.nlte_to_global_level=globals;n.fl_to_super=mapping;
    double live[3]={100,200,300},fraction[3]={.01,.99,1};
    n.nlte_level_populations=live;n.within_sl_frac=fraction;
    AtomicData a={0};a.n_ion_pops=2;a.ion_pop_Z=Z;a.ion_pop_stage=stage;
    PlasmaState p={0};p.n_shells=1;p.rho=rho;p.physical_history=&h;
    double A[4]={-2,2,3,-3},x[2],saved[2];PopulationGeneratorSolveDiagnostic d;
    for(int repeat=0;repeat<2;repeat++){
        CHECK(nlte_physical_history_solve_levels(&n,&a,&p,0,2,0,2,A,x,&d)==POP_OK,"history source solve");
        CHECK(fabs(x[1]-.375)<1e-15 && fabs(x[0]-.625)<1e-15,"history aggregate and density dilution");
        live[0]*=10;fraction[0]*=.1;
    }
    memcpy(saved,x,sizeof(x));rho[0]=2;
    CHECK(nlte_physical_history_solve_levels(&n,&a,&p,0,2,0,2,A,x,&d)!=POP_OK &&
          !memcmp(saved,x,sizeof(x)),"nonhomology atomic rejection");
    rho[0]=1;int changed[3]={0,2,1};n.nlte_to_global_level=changed;
    CHECK(nlte_physical_history_solve_levels(&n,&a,&p,0,2,0,2,A,x,&d)!=POP_OK,"changed level identity");
    CHECK(levels[0]==2 && levels[1]==4 && levels[2]==2,"past unchanged");
    printf("[POP-TIME] history_aggregate=PASS same_epoch_repeats=2 current_iterate_source=0\n");
    return 0;
}

int main(void)
{
    if(core_check()||history_check())return 1;
    puts("[POP-TIME] PASS positivity=NO_CLAMP conservation=NO_RENORMALIZATION");
    return 0;
}
