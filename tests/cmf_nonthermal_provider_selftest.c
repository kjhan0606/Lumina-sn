#include "cmf_nonthermal_provider.h"
#include <assert.h>
#include <stdio.h>

int main(void){
    enum {NS=2,NL=4,NI=4,NE=128,NC=4};
    AtomicData a={0};NLTEConfig n={0};PlasmaState p={0};
    int lz[NL]={14,14,14,14},li[NL]={1,1,2,3},map[NL]={0,1,2,3};
    int iz[NI]={14,14,14,14},iq[NI]={1,2,3,4},offset[NI+1]={0,2,3,4,4};
    double en[NL]={0,2,0,0},ip[3]={16,30,45};
    double population[NL*NS]={4e7,3e7,1e7,1e7,3e7,4e7,2e7,2e7};
    double ne[NS]={1.7e8,1.8e8},within[NL*NS];
    for(int i=0;i<NL*NS;i++)within[i]=1;
    a.n_levels=NL;a.n_ion_pops=NI;a.level_Z=lz;a.level_ion=li;a.level_energy_eV=en;
    a.ion_pop_Z=iz;a.ion_pop_stage=iq;a.level_offset=offset;
    a.n_ionization=3;a.ioniz_Z=iz;a.ioniz_ion=iq;a.ioniz_energy_eV=ip;
    n.n_nlte_levels_total=NL;n.global_to_nlte_level=map;n.fl_to_super=map;
    n.nlte_level_populations=population;n.within_sl_frac=within;
    p.n_shells=NS;p.n_electron=ne;
    double heating[NS]={1e-10,2e-10},E[NE],sigma[NC*NE];
    GammaDeposition g={0};g.n_shells=NS;g.heating_rate=heating;g.generation=3;g.epoch=172800;
    g.provenance=GAMMA_PROVENANCE_NONLOCAL_MC;
    CMFNonthermalTransition tr[NC]={{0,1,-1},{0,2,-1},{0,3,-1},{3,-1,3}};
    CMFNonthermalRoute channel[NC]={
        {CMF_NT_EXCITATION,2,0,sigma,0},{CMF_NT_IONIZATION,16,0,sigma+NE,0},
        {CMF_NT_IONIZATION,80,0,sigma+2*NE,34},{CMF_NT_IONIZATION,45,0,sigma+3*NE,0}};
    for(int i=0;i<NE;i++){
        E[i]=exp(log(.1)+log(20000)*i/(NE-1));
        sigma[i]=cmf_nonthermal_bethe(E[i],2,.1,1);
        const double abcd[4]={22.8,-12,1.9,-22.6};
        sigma[NE+i]=cmf_nonthermal_arnaud(E[i],16,abcd);
        sigma[2*NE+i]=cmf_nonthermal_arnaud(E[i],80,abcd);
        sigma[3*NE+i]=cmf_nonthermal_arnaud(E[i],45,abcd);
    }
    CMFNonthermalBoundModel m={NL,NI,NE,NC,E,tr,channel};CMFNonthermalRates rates={0};
    assert(!cmf_nonthermal_bound_evaluate(&m,&a,&n,&p,&g,&rates));
    const double absenergy[5]={0,2,16,46,91};
    for(int s=0;s<NS;s++){
        double A[25]={0},outflow[5]={0},stored=0;
        assert(!cmf_nonthermal_rates_add(&rates,&n,&a,14,s,NS,g.epoch,g.generation,0,5,3,A,outflow));
        assert(A[3]>0&&A[3*5+4]>0); /* 두 단계 전리 및 실제 terminal */
        for(int j=0;j<5;j++){
            double sum=0;for(int i=0;i<5;i++)sum+=A[j*5+i];assert(fabs(sum)<1e-16);
            if(j<NL)for(int i=0;i<5;i++)stored+=(absenergy[i]-absenergy[j])*A[j*5+i]*population[j*NS+s]*1.602176634e-12;
        }
        assert(stored>0&&stored<heating[s]);
        printf("NT_PROVIDER shell=%d input=%.17g internal_gain=%.17g remaining_heat=%.17g excitation=%.17g single=%.17g double=%.17g terminal=%.17g\n",
               s,heating[s],stored,heating[s]-stored,A[1],A[2],A[3],A[3*5+4]);
        double before[25];memcpy(before,A,sizeof A);
        assert(cmf_nonthermal_rates_add(&rates,&n,&a,14,s,NS,g.epoch+1,g.generation,0,5,3,A,outflow));
        assert(!memcmp(before,A,sizeof A));rates.transitions[0].upper_global=99;
        assert(cmf_nonthermal_rates_add(&rates,&n,&a,14,s,NS,g.epoch,g.generation,0,5,3,A,outflow));
        assert(!memcmp(before,A,sizeof A));rates.transitions[0].upper_global=1;
    }
    double initial=rates.rate_per_particle[1*NS];ne[0]*=10;
    assert(!cmf_nonthermal_bound_evaluate(&m,&a,&n,&p,&g,&rates));
    assert(rates.rate_per_particle[1*NS]<initial); /* 현재 후보 ne 실제 소비 */
    CMFNonthermalRates saved=rates;channel[2].auger_electron_eV=33;
    assert(cmf_nonthermal_bound_evaluate(&m,&a,&n,&p,&g,&rates));
    assert(!memcmp(&saved,&rates,sizeof rates));channel[2].auger_electron_eV=34;
    heating[0]=0;
    assert(!cmf_nonthermal_bound_evaluate(&m,&a,&n,&p,&g,&rates));
    for(int k=0;k<NC;k++)assert(rates.rate_per_particle[k*NS]==0);
    heating[1]=0;saved=rates;channel[0].threshold_eV=NAN;
    assert(cmf_nonthermal_bound_evaluate(&m,&a,&n,&p,&g,&rates));
    assert(!memcmp(&saved,&rates,sizeof rates));
    cmf_nonthermal_rates_free(&rates);
    puts("CMF_NONTHERMAL_PROVIDER_SELFTEST_OK actual_NLTE_matrix_adapter=1 SN_atomic_model=0");
    return 0;
}
