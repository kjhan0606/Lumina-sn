#ifndef CMF_NONTHERMAL_RATES_H
#define CMF_NONTHERMAL_RATES_H
#include "lumina.h"
/* 침적은 GammaDeposition 소유, 아래는 한 물질 반복의 full-level 전이율이다.
 * upper_global=-1이면 upper_ion은 실제 terminal ion-pop 인덱스이다.
 * 각 전이율은 lower particle당 s^-1이며 분기 가중치를 이미 포함한다. */
typedef struct {
    int lower_global,upper_global,upper_ion;
} CMFNonthermalTransition;
typedef struct CMFNonthermalRates {
    int n_shells,n_levels,n_ions,n_transitions;
    double epoch_s;
    uint64_t deposition_generation;
    CMFNonthermalTransition *transitions;
    double *rate_per_particle; /* transition-major */
} CMFNonthermalRates;
typedef struct CMFNonthermalProvider {
    const void *context; /* 불변 원자 단면적/경로; 후보 인구를 보관하지 않는다. */
    int (*evaluate)(const void*,const AtomicData*,const NLTEConfig*,const PlasmaState*,
                    const GammaDeposition*,CMFNonthermalRates*);
} CMFNonthermalProvider;
static inline void cmf_nonthermal_rates_free(CMFNonthermalRates *r){
    if(r){free(r->transitions);free(r->rate_per_particle);memset(r,0,sizeof(*r));}
}
/* 전체 원소의 private generator에1회 합산한다. 원자/투영 오류는 적용 전에
 * 거부하며, 인접 이온 pair로 경로를 잘라내지 않는다. */
static inline int cmf_nonthermal_rates_add(const CMFNonthermalRates *r,
    const NLTEConfig *n,const AtomicData *a,int Z,int shell,int ns,double epoch,
    uint64_t deposition_generation,int sl0,int N,int terminal,double *A,double *outflow){
    if(!r||!n||!a||!A||!outflow||ns<1||shell<0||shell>=ns||N<1||sl0<0||
       r->n_shells!=ns||r->n_levels!=a->n_levels||r->n_ions!=a->n_ion_pops||
       r->n_transitions<0||r->epoch_s!=epoch||r->deposition_generation!=deposition_generation||
       !deposition_generation||(r->n_transitions&&(!r->transitions||!r->rate_per_particle))||
       !a->level_Z||!a->level_ion||!a->level_offset||!a->ion_pop_Z||!a->ion_pop_stage||
       !n->global_to_nlte_level||!n->fl_to_super||!n->within_sl_frac)return -1;
    for(int pass=0;pass<2;pass++)for(int k=0;k<r->n_transitions;k++){
        const CMFNonthermalTransition *q=&r->transitions[k];
        if(q->lower_global<0||q->lower_global>=a->n_levels)return -1;
        if(a->level_Z[q->lower_global]!=Z)continue;
        int lower=n->global_to_nlte_level[q->lower_global],target=-1;
        if(lower<0||lower>=n->n_nlte_levels_total)return -1;
        int source=n->fl_to_super[lower]-sl0;
        if(q->upper_global>=0){
            if(q->upper_global>=a->n_levels||q->upper_ion!=-1||
               a->level_Z[q->upper_global]!=Z||a->level_ion[q->upper_global]<a->level_ion[q->lower_global])return -1;
            if(a->level_ion[q->upper_global]==a->level_ion[q->lower_global]&&
               (!a->level_energy_eV||!isfinite(a->level_energy_eV[q->upper_global])||
                !isfinite(a->level_energy_eV[q->lower_global])||
                a->level_energy_eV[q->upper_global]<a->level_energy_eV[q->lower_global]))return -1;
            int upper=n->global_to_nlte_level[q->upper_global];
            if(upper<0||upper>=n->n_nlte_levels_total)return -1;
            target=n->fl_to_super[upper]-sl0;
        }else{
            if(q->upper_global!=-1||terminal<0||terminal>=a->n_ion_pops||q->upper_ion!=terminal||
               a->ion_pop_Z[terminal]!=Z||a->ion_pop_stage[terminal]<=a->level_ion[q->lower_global]||
               a->level_offset[terminal]!=a->level_offset[terminal+1])return -1;
            target=N-1;
        }
        double f=n->within_sl_frac[(size_t)lower*ns+shell];
        double rate=r->rate_per_particle[(size_t)k*ns+shell];
        if(source<0||source>=N||target<0||target>=N||!(f>=0&&f<=1)||!isfinite(f)||
           !(rate>=0)||!isfinite(rate)||!isfinite(rate*f))return -1;
        if(pass&&source!=target){
            rate*=f;A[(size_t)source*N+target]+=rate;
            A[(size_t)source*N+source]-=rate;outflow[source]+=rate;
        }
    }
    return 0;
}
#endif
