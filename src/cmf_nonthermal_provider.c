#include "cmf_nonthermal_provider.h"

/* 공급자가 사용하는 원자 내부에너지 차이. 원형 단면적 문턱을 실제 준위의
 * 차이와 몰래 동일시하지 않는다. 둘이 맞지 않는 미완성 연결은 거부한다. */
int cmf_nonthermal_transition_energy(const AtomicData *a,const CMFNonthermalTransition *t,
                             double *change,int *electrons){
    if(!a||!t||!change||!electrons)return -1;
    int lo=t->lower_global,up=t->upper_global,qi,Z;
    if(lo<0||lo>=a->n_levels||!a->level_energy_eV||!a->level_Z||!a->level_ion)return -1;
    Z=a->level_Z[lo];qi=a->level_ion[lo];double end;
    int qf;
    if(up>=0){
        if(up>=a->n_levels||t->upper_ion!=-1||a->level_Z[up]!=Z)return -1;
        qf=a->level_ion[up];end=a->level_energy_eV[up];
    }else{
        int ip=t->upper_ion;
        if(up!=-1||ip<0||ip>=a->n_ion_pops||!a->ion_pop_Z||!a->ion_pop_stage||!a->level_offset||
           a->ion_pop_Z[ip]!=Z||a->level_offset[ip]!=a->level_offset[ip+1])return -1;
        qf=a->ion_pop_stage[ip];end=0;
    }
    if(qf<qi||qf>qi+2||!isfinite(end)||!isfinite(a->level_energy_eV[lo]))return -1;
    double energy=end-a->level_energy_eV[lo];
    if(qf>qi&&(!a->ioniz_Z||!a->ioniz_ion||!a->ioniz_energy_eV))return -1;
    for(int q=qi;q<qf;q++){
        int found=0;
        for(int k=0;k<a->n_ionization;k++)if(a->ioniz_Z[k]==Z&&a->ioniz_ion[k]==q){
            if(found||!(a->ioniz_energy_eV[k]>0)||!isfinite(a->ioniz_energy_eV[k]))return -1;
            energy+=a->ioniz_energy_eV[k];found=1;
        }
        if(!found)return -1;
    }
    if(!(energy>0)||!isfinite(energy))return -1;
    *change=energy;*electrons=qf-qi;return 0;
}

/* 입력은 공통 read-only, 출력은 한 셸만 소유한다. 셸 내부 합산 순서는 불변이다. */
static int evaluate_shell(const CMFNonthermalBoundModel *m,const NLTEConfig *n,
    const PlasmaState *p,const GammaDeposition *g,size_t s,double *loss,
    CMFNonthermalRoute *channels,CMFNonthermalRates *r){
    size_t ns=p->n_shells,nc=m->n_channels,ne=m->n_energy;
    double deposition=g->heating_rate[s]/1.602176634e-12;
    if(!(deposition>=0)||!isfinite(deposition)||!(p->n_electron[s]>0)||!isfinite(p->n_electron[s]))return -1;
    for(size_t k=0;k<nc;k++){
        int lower=n->global_to_nlte_level[m->transitions[k].lower_global];
        if(lower<0||lower>=n->n_nlte_levels_total)return -1;
        channels[k]=m->channels[k];
        channels[k].lower_population_cm3=n->nlte_level_populations[(size_t)lower*ns+s];
        if(!(channels[k].lower_population_cm3>=0)||!isfinite(channels[k].lower_population_cm3))return -1;
    }
    if(deposition==0)return 0;
    for(size_t i=0;i<ne;i++)loss[i]=cmf_nonthermal_coulomb(m->energy_eV[i],p->n_electron[s]);
    CMFNonthermalInput input={m->n_energy,m->n_channels,m->energy_eV,loss,channels};
    CMFNonthermalResult solved={0};
    if(cmf_nonthermal_solve(&input,&solved))return -1;
    for(size_t k=0;k<nc;k++){
        double rate=deposition*solved.coefficient_cm3_per_eV[k];
        if(!(rate>=0)||!isfinite(rate)){cmf_nonthermal_free(&solved);return -1;}
        r->rate_per_particle[k*ns+s]=rate;
    }
    cmf_nonthermal_free(&solved);return 0;
}

int cmf_nonthermal_bound_evaluate(const void *context,const AtomicData *a,const NLTEConfig *n,
    const PlasmaState *p,const GammaDeposition *g,CMFNonthermalRates *out){
    const CMFNonthermalBoundModel *m=context;
    if(!m||!a||!n||!p||!g||!out||m->n_levels!=a->n_levels||m->n_ions!=a->n_ion_pops||
       m->n_channels<0||m->n_energy<2||m->n_energy>100000||!m->energy_eV||
       (m->n_channels&&(!m->transitions||!m->channels))||g->provenance!=GAMMA_PROVENANCE_NONLOCAL_MC||
       !g->generation||!g->heating_rate||!(g->epoch>0)||!isfinite(g->epoch)||
       p->n_shells<1||p->n_shells!=g->n_shells||!p->n_electron||!n->nlte_level_populations||
       !n->global_to_nlte_level)return -1;
    size_t ns=p->n_shells,nc=m->n_channels,ne=m->n_energy;
    if(nc>SIZE_MAX/ns/sizeof(double))return -1;
    for(size_t i=0;i<ne;i++)if(!(m->energy_eV[i]>0)||!isfinite(m->energy_eV[i])||
        (i&&!(m->energy_eV[i]>m->energy_eV[i-1])))return -1;
    CMFNonthermalRates r={.n_shells=p->n_shells,.n_levels=a->n_levels,.n_ions=a->n_ion_pops,
        .n_transitions=m->n_channels,.epoch_s=g->epoch,.deposition_generation=g->generation};
    r.transitions=malloc((nc?nc:1)*sizeof(*r.transitions));
    r.rate_per_particle=calloc(nc?nc*ns:1,sizeof(double));
    if(!r.transitions||!r.rate_per_particle)goto fail;
    if(nc)memcpy(r.transitions,m->transitions,nc*sizeof(*r.transitions));
    for(size_t k=0;k<nc;k++){
        double delta;int electrons;
        if(cmf_nonthermal_transition_energy(a,&m->transitions[k],&delta,&electrons))goto fail;
        const CMFNonthermalRoute *q=&m->channels[k];
        if(!q->cross_section_cm2||!(q->threshold_eV>0)||!isfinite(q->threshold_eV)||
           !(q->auger_electron_eV>=0&&q->auger_electron_eV<q->threshold_eV)||!isfinite(q->auger_electron_eV)||
           (electrons==0&&(q->kind!=CMF_NT_EXCITATION||q->auger_electron_eV!=0))||
           (electrons>0&&(q->kind!=CMF_NT_IONIZATION||electrons!=1+(q->auger_electron_eV>0)))||
           fabs((q->threshold_eV-q->auger_electron_eV)/delta-1)>1e-10)goto fail;
        for(size_t i=0;i<ne;i++)if(!(q->cross_section_cm2[i]>=0)||!isfinite(q->cross_section_cm2[i])||
            (m->energy_eV[i]<q->threshold_eV&&q->cross_section_cm2[i]!=0)||
            (q->kind==CMF_NT_IONIZATION&&m->energy_eV[i]==q->threshold_eV&&q->cross_section_cm2[i]!=0))goto fail;
    }
    {
        int failed=0;
        /* 각 스레드가 작은 임시 작업공간을 재사용한다. 외부 결과는 실패 시 불변이다. */
#ifdef _OPENMP
#pragma omp parallel reduction(|:failed) if(ns>1)
#endif
        {
            double *loss=malloc(ne*sizeof(double));
            CMFNonthermalRoute *channels=malloc((nc?nc:1)*sizeof(*channels));
            if(!loss||!channels)failed=1;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
            for(size_t s=0;s<ns;s++){
                if(!failed&&evaluate_shell(m,n,p,g,s,loss,channels,&r))failed=1;
            }
            free(loss);free(channels);
        }
        if(failed)goto fail;
    }
    cmf_nonthermal_rates_free(out);*out=r;return 0;
fail:
    cmf_nonthermal_rates_free(&r);return -1;
}
