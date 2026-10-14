#include "cmf_decay.h"
#include "nlte_population_candidate.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"DECAY_FAIL line=%d %s\n",__LINE__,#x);return 1;}}while(0)
static int near(double x,double y){return fabs(x-y)<3e-13*fmax(fmax(fabs(x),fabs(y)),1e-300);}

static int chain(void){
    double old[3]={100,30,20},out[3],count[2],half[3],halfcount[2],twice[3];
    double l1=log(2)/(6.0749*86400),l2=log(2)/(77.233*86400),dt=8*86400;
    CHECK(!cmf_decay_chain_step(l1,l2,dt,old,out,count));
    CHECK(near(out[0]+out[1]+out[2],150)&&near(count[0],old[0]-out[0])&&near(count[1],out[2]-old[2]));
    CHECK(!cmf_decay_chain_step(l1,l2,dt/2,old,half,halfcount));
    CHECK(!cmf_decay_chain_step(l1,l2,dt/2,half,twice,halfcount));
    for(int i=0;i<3;i++)CHECK(near(out[i],twice[i]));
    double unit[3]={1,0,0};
    CHECK(!cmf_decay_chain_step(1,1,2,unit,out,count));
    CHECK(near(out[0],exp(-2))&&near(out[1],2*exp(-2))&&near(out[2],1-3*exp(-2)));
    CHECK(!cmf_decay_chain_step(1,1,1e-10,unit,out,count));
    CHECK(near(out[2],.5e-20*(1-2e-10/3))&&out[2]>0);
    CHECK(!cmf_decay_chain_step(1,1e-100,10,unit,out,count));
    CHECK(near(out[2],1e-100*(10-1+exp(-10))));
    CHECK(!cmf_decay_chain_step(1,2,0,old,out,count)&&!memcmp(out,old,sizeof old)&&count[0]==0&&count[1]==0);
    double saved[3];memcpy(saved,out,sizeof out);
    CHECK(cmf_decay_chain_step(-1,2,1,old,out,count)&&!memcmp(saved,out,sizeof out));
    puts("CMF_DECAY_CHAIN_PASS semigroup=1 equal_rate=1 tiny_step=1 slow_daughter=1");return 0;
}

static int projection(void){
    enum{NS=2,NI=6,NE=3};
    int Z[NI]={28,28,27,27,26,26},stage[NI]={0,1,0,1,0,1},g[NI]={1,1,1,1,1,1};
    int off[NI+1]={0,1,2,3,4,5,6},map[NI]={0,1,2,3,4,5},el[NE]={28,27,26},eo[NE+1]={0,2,4,6};
    int istage[NE]={0,0,0};double chi[NE]={10,20,30},E[NI]={0};
    double T[NS]={10000,12000},rho[NS]={1e-12,2e-12},ne[NS]={66,110};
    double populations[NI*NS]={70,140,30,60,6,8,24,32,8,12,12,18};
    double part[NI*NS],within[NI*NS],after[NE*NS],l1=log(2)/(6.0749*86400),l2=log(2)/(77.233*86400);
    for(int i=0;i<NI*NS;i++)within[i]=1;
    const double epoch=10*86400,old_epoch=2*86400;
    AtomicData a={0};NLTEConfig n={0};PlasmaState p={0};
    a.n_elements=NE;a.element_Z=el;a.elem_ion_offset=eo;a.n_ion_pops=a.n_levels=NI;
    a.ion_pop_Z=Z;a.ion_pop_stage=stage;a.level_Z=Z;a.level_ion=stage;a.level_g=g;
    a.level_offset=off;a.level_energy_eV=E;a.ion_number_density=populations;a.partition_functions=part;
    a.n_ionization=NE;a.ioniz_Z=el;a.ioniz_ion=istage;a.ioniz_energy_eV=chi;a.population_committed_generation=1;
    n.n_nlte_levels_total=n.n_super_total=NI;n.n_nlte_ions=NI;
    n.nlte_to_global_level=n.global_to_nlte_level=n.fl_to_super=map;n.nlte_level_populations=populations;
    n.within_sl_frac=within;
    n.population_committed_generation=1;
    p.n_shells=NS;p.rho=rho;p.T_e=T;p.n_electron=ne;p.T_e_generation=1;
    PopulationAtomicView v={0};v.n_ions=v.n_levels=NI;v.level_offset=off;
    v.energy_eV=E;v.g=g;v.level_Z=Z;v.level_ion=stage;
    CHECK(population_partition_build(&v,T,NS,1,1,part,&a.partition_stamp)==POP_OK);
    NLTEPhysicalHistory old={0},projected={0};
    CHECK(nlte_physical_history_capture(&old,&n,&a,&p,old_epoch)==NLTE_CANDIDATE_OK);
    double saved[(5+2*NI)*NS];memcpy(saved,old.storage,sizeof saved);
    double reference[NS][13];FILE *fp=fopen("cmf_decay_oracle.dat","r");CHECK(fp);
    for(int s=0;s<NS;s++){
        for(int k=0;k<13;k++)CHECK(fscanf(fp,"%lf",&reference[s][k])==1);
        double pop[3]={populations[s]+populations[NS+s],populations[2*NS+s]+populations[3*NS+s],populations[4*NS+s]+populations[5*NS+s]};
        double out[3],count[2];CHECK(!cmf_decay_chain_step(l1,l2,epoch-old_epoch,pop,out,count));
        for(int e=0;e<NE;e++){after[e*NS+s]=out[e];CHECK(near(out[e],reference[s][e]));}
        CHECK(near(count[0],reference[s][3])&&near(count[1],reference[s][4]));
        CHECK(near((2*count[0]+4*count[1])/(epoch-old_epoch),reference[s][5]));
        CHECK(near(.2*count[1]/(epoch-old_epoch),reference[s][6]));
    }fclose(fp);
    CHECK(nlte_physical_history_decay_projection(&projected,&old,&n,&a,epoch,after)==NLTE_CANDIDATE_OK);
    CHECK(!memcmp(saved,old.storage,sizeof saved)&&projected.decay_epoch_s==epoch);
    for(int s=0;s<NS;s++){
        double energy=0,electron=0,total=0;
        for(int i=0;i<NI;i++){
            CHECK(near(projected.level_density[i*NS+s],reference[s][7+i]));
            CHECK(projected.level_density[i*NS+s]==projected.ion_density[i*NS+s]);
            electron+=stage[i]*reference[s][7+i];total+=reference[s][7+i];
            energy+=stage[i]*chi[i/2]*reference[s][7+i]*EV_TO_ERG;
        }
        CHECK(near(projected.energy.n_electron_cm3[s],electron));
        CHECK(near(projected.energy.internal_energy_atom_erg[s],energy/total));
    }
    /* 실제 시간 인구/에너지 소비부: 시간 단계에 맞는 투영만 허용한다. */
    p.physical_history=&projected;for(int s=0;s<NS;s++)rho[s]*=pow(old_epoch/epoch,3);
    double zero[4]={0},solution[2];
    for(int e=0;e<NE;e++)for(int s=0;s<NS;s++){
        CHECK(nlte_physical_history_solve_element(&n,&a,&p,s,epoch,2*e,2,-1,zero,solution)==POP_OK);
        for(int k=0;k<2;k++)CHECK(near(solution[k],reference[s][7+2*e+k]*pow(old_epoch/epoch,3)));
    }
    CHECK(nlte_physical_history_solve_element(&n,&a,&p,0,epoch+1,0,2,-1,zero,solution)!=POP_OK);
    for(int s=0;s<NS;s++){
        ne[s]=projected.energy.n_electron_cm3[s]*pow(old_epoch/epoch,3);
        for(int i=0;i<NI;i++)populations[i*NS+s]=projected.level_density[i*NS+s]*pow(old_epoch/epoch,3);
    }
    double velocity[NS]={1e8,2e8},radius[NS]={epoch*1e8,epoch*2e8};
    NLTEPopulationCandidate candidate={0};
    CHECK(nlte_population_candidate_begin(&candidate,&n,&a,&p,T,NS,1,1)==NLTE_CANDIDATE_OK);
    CHECK(nlte_population_candidate_prepare_adiabatic(&candidate,radius,velocity,epoch)==NLTE_CANDIDATE_OK);
    for(int s=0;s<NS;s++){
        double expected=(candidate.n_atom[s]+ne[s])*K_BOLTZMANN*T[s]*3*log(epoch/old_epoch)/(epoch-old_epoch);
        CHECK(near(candidate.time_work[s].signed_total,expected));
    }
    nlte_population_candidate_free(&candidate);
    projected.decay_epoch_s+=1;
    CHECK(nlte_population_candidate_begin(&candidate,&n,&a,&p,T,NS,1,1)==NLTE_CANDIDATE_OK);
    CHECK(nlte_population_candidate_prepare_adiabatic(&candidate,radius,velocity,epoch)==NLTE_CANDIDATE_ADIABATIC_FAILED);
    nlte_population_candidate_free(&candidate);projected.decay_epoch_s-=1;
    NLTEPhysicalHistory rejected={0};
    CHECK(nlte_physical_history_decay_projection(&rejected,&projected,&n,&a,epoch+1,after)!=NLTE_CANDIDATE_OK&&rejected.storage==NULL);
    double *preserved=projected.storage;
    CHECK(nlte_physical_history_decay_projection(&projected,&projected,&n,&a,epoch,after)!=NLTE_CANDIDATE_OK);
    CHECK(nlte_physical_history_decay_projection(&old,&old,&n,&a,epoch,after)!=NLTE_CANDIDATE_OK);
    after[0]+=1;
    CHECK(nlte_physical_history_decay_projection(&projected,&old,&n,&a,epoch,after)!=NLTE_CANDIDATE_OK&&projected.storage==preserved);
    CHECK(!memcmp(saved,old.storage,sizeof saved));
    nlte_physical_history_free(&old);nlte_physical_history_free(&projected);
    puts("CMF_DECAY_PROJECTION_PASS actual_original_Fortran=1 charge_energy_rebuilt=1 actual_time_population=1 past_preserved=1");
    return 0;
}
int main(void){return chain()||projection();}
