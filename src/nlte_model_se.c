#include "nlte_model_se.h"

#include <limits.h>

void nlte_model_element_free(NLTEModelElement *e)
{
    if(!e)return;
    free(e->super_to_row);free(e->terminal_to_row);free(e->initial_population);
    memset(e,0,sizeof(*e));
}

int nlte_model_element_init(NLTEModelElement *out,const NLTEConfig *n,
    const AtomicData *a,const PlasmaState *p,int Z,int shell)
{
    if(!out || out->n_variables || out->super_to_row || out->terminal_to_row ||
       out->initial_population || !n || !a || !p || !n->model_atom_projection ||
       a->n_levels<=0 || a->n_ion_pops<=0 || n->n_super_total<=0 ||
       n->n_nlte_levels_total!=a->n_levels || p->n_shells<=0 ||
       shell<0 || shell>=p->n_shells || Z<=0 || !n->nlte_to_global_level ||
       !n->global_to_nlte_level || !n->fl_to_super || !n->super_anchor_global || !n->nlte_level_populations ||
       !a->level_Z || !a->level_ion || !a->level_offset ||
       !a->ion_pop_Z || !a->ion_pop_stage || !a->ion_number_density ||
       !a->element_Z || !a->population_committed_generation ||
       n->population_committed_generation!=a->population_committed_generation)return -1;
    NLTEModelElement e={.Z=Z,.shell=shell,.n_super_total=(size_t)n->n_super_total,
        .n_ion_pops=(size_t)a->n_ion_pops,.material_generation=a->population_committed_generation};
    e.super_to_row=malloc(e.n_super_total*sizeof(int));
    e.terminal_to_row=malloc(e.n_ion_pops*sizeof(int));
    if(!e.super_to_row || !e.terminal_to_row)goto fail;
    for(size_t i=0;i<e.n_super_total;i++)e.super_to_row[i]=-1;
    for(size_t i=0;i<e.n_ion_pops;i++)e.terminal_to_row[i]=-1;
    for(int gl=0;gl<a->n_levels;gl++){
        if(a->level_Z[gl]!=Z)continue;
        int ng=n->global_to_nlte_level[gl];
        if(ng<0 || ng>=n->n_nlte_levels_total || n->nlte_to_global_level[ng]!=gl)goto fail;
        int sl=n->fl_to_super[ng];
        if(sl<0 || (size_t)sl>=e.n_super_total)goto fail;
        int anchor=n->super_anchor_global[sl];
        if(anchor<0 || anchor>=a->n_levels || a->level_Z[anchor]!=Z ||
           a->level_ion[anchor]!=a->level_ion[gl])goto fail;
        if(e.super_to_row[sl]<0)e.super_to_row[sl]=(int)e.n_bound++;
    }
    e.n_variables=e.n_bound;
    for(int ip=0;ip<a->n_ion_pops;ip++){
        if(a->ion_pop_Z[ip]!=Z)continue;
        if(a->level_offset[ip]<0 || a->level_offset[ip+1]<a->level_offset[ip] ||
           a->level_offset[ip+1]>a->n_levels)goto fail;
        if(a->level_offset[ip]==a->level_offset[ip+1]){
            if(e.n_variables>=INT_MAX)goto fail;
            e.terminal_to_row[ip]=(int)e.n_variables++;
            e.n_terminals++;
        }
    }
    if(!e.n_bound || !e.n_variables || e.n_variables>SIZE_MAX/sizeof(double))goto fail;
    e.initial_population=calloc(e.n_variables,sizeof(double));
    if(!e.initial_population)goto fail;
    for(int gl=0;gl<a->n_levels;gl++){
        if(a->level_Z[gl]!=Z)continue;
        int ng=n->global_to_nlte_level[gl],row=e.super_to_row[n->fl_to_super[ng]];
        double value=n->nlte_level_populations[(size_t)ng*p->n_shells+shell];
        if(!isfinite(value) || value<0.)goto fail;
        e.initial_population[row]+=value;
    }
    for(int ip=0;ip<a->n_ion_pops;ip++)if(e.terminal_to_row[ip]>=0){
        double value=a->ion_number_density[(size_t)ip*p->n_shells+shell];
        if(!isfinite(value) || value<0.)goto fail;
        e.initial_population[e.terminal_to_row[ip]]=value;
    }
    int element=-1;
    for(int i=0;i<a->n_elements;i++)if(a->element_Z[i]==Z){if(element>=0)goto fail;element=i;}
    if(element<0)goto fail;
    e.total_nuclei=lumina_element_number_density(a,p,element,shell,p->n_shells);
    long double sum=0.;
    for(size_t i=0;i<e.n_variables;i++){
        if(!isfinite(e.initial_population[i]))goto fail;
        sum+=e.initial_population[i];
    }
    if(!isfinite(e.total_nuclei) || e.total_nuclei<0. ||
       fabsl(sum-e.total_nuclei)>1.e-12L*fmaxl(fabsl(sum),fabsl(e.total_nuclei)))goto fail;
    *out=e;return 0;
fail:
    nlte_model_element_free(&e);return -1;
}

void nlte_model_bf_block_free(NLTEModelBFBlock *b)
{
    if(!b)return;
    free(b->radiative);free(b->collisional);free(b->radiative_heat);memset(b,0,sizeof(*b));
}

static int add_route(double *a,size_t n,size_t lower,size_t upper,double forward,double reverse)
{
    if(lower>=n || upper>=n || lower==upper || !isfinite(forward) || forward<0. ||
       !isfinite(reverse) || reverse<0.)return -1;
    a[lower*n+upper]+=forward;
    a[upper*n+lower]+=reverse;
    return 0;
}

/* 대각은 최종 비대각 유출의 합이다. 작은 양의 율을 자르지 않는다. */
static int close_columns(double *a,size_t n)
{
    for(size_t j=0;j<n;j++){
        long double outflow=0.;
        for(size_t i=0;i<n;i++)if(i!=j){
            if(!isfinite(a[j*n+i]) || a[j*n+i]<0.)return -1;
            outflow+=a[j*n+i];
        }
        a[j*n+j]=-(double)outflow;
        if(!isfinite(a[j*n+j]))return -1;
    }
    return 0;
}

int nlte_model_element_bf_assemble(NLTEModelBFBlock *out,const NLTEModelElement *e,
    const NLTEConfig *n,const AtomicData *a,const PlasmaState *p)
{
    if(!out || out->n_variables || out->radiative || out->collisional || out->radiative_heat ||
       !e || !n || !a || !p || !e->n_variables || !e->super_to_row || !e->terminal_to_row ||
       e->material_generation!=a->population_committed_generation ||
       e->n_super_total!=(size_t)n->n_super_total || e->n_ion_pops!=(size_t)a->n_ion_pops ||
       !n->within_sl_frac || nlte_point_bf_view_check(n,a,p,e->shell) ||
       e->n_variables>SIZE_MAX/e->n_variables ||
       e->n_variables*e->n_variables>SIZE_MAX/sizeof(double))return -1;
    size_t nv=e->n_variables,ns=(size_t)p->n_shells;
    NLTEModelBFBlock b={.n_variables=nv};
    b.radiative=calloc(nv*nv,sizeof(double));b.collisional=calloc(nv*nv,sizeof(double));
    b.radiative_heat=calloc(nv,sizeof(double));
    long double *heat=calloc(nv,sizeof(long double));
    if(!b.radiative || !b.collisional || !b.radiative_heat || !heat)goto fail;
    for(int gl=0;gl<a->n_levels;gl++){
        if(a->level_Z[gl]!=e->Z)continue;
        int ng=n->global_to_nlte_level[gl];
        if(ng<0 || ng>=n->n_nlte_levels_total)goto fail;
        int sl=n->fl_to_super[ng];
        if(sl<0 || (size_t)sl>=e->n_super_total)goto fail;
        int lower=e->super_to_row[sl];
        double f=n->within_sl_frac[(size_t)ng*ns+e->shell];
        if(lower<0 || !isfinite(f) || f<0. || f>1.)goto fail;
        size_t begin,end;
        if(bf_rate_point_row(n->point_bf_rates,(size_t)a->n_levels,ns,(size_t)gl,
                            a->population_committed_generation,&begin,&end))goto fail;
        for(size_t r=begin;r<end;r++){
            BfPointTransition tr;
            if(nlte_point_bf_transition(n,a,p,gl,r,e->shell,&tr))goto fail;
            int upper=-1;
            if(tr.target_level>=0){
                int up=n->global_to_nlte_level[tr.target_level];
                if(up<0 || up>=n->n_nlte_levels_total)goto fail;
                int us=n->fl_to_super[up];
                if(us<0 || (size_t)us>=e->n_super_total)goto fail;
                upper=e->super_to_row[us];
            }else{
                if(tr.target_ion<0 || (size_t)tr.target_ion>=e->n_ion_pops)goto fail;
                upper=e->terminal_to_row[tr.target_ion];b.terminal_routes++;
            }
            if(upper<0 || add_route(b.radiative,nv,(size_t)lower,(size_t)upper,
                tr.radiative.photo*f,tr.radiative.spontaneous+tr.radiative.stimulated) ||
               add_route(b.collisional,nv,(size_t)lower,(size_t)upper,
                tr.collisional_ionization*f,tr.three_body_recombination))goto fail;
            heat[lower]+=(long double)tr.radiative.heat_photo*f;
            heat[upper]-=(long double)tr.radiative.heat_spontaneous+tr.radiative.heat_stimulated;
            b.routes++;
        }
    }
    if(close_columns(b.radiative,nv) || close_columns(b.collisional,nv))goto fail;
    for(size_t i=0;i<nv;i++){b.radiative_heat[i]=(double)heat[i];if(!isfinite(b.radiative_heat[i]))goto fail;}
    free(heat);*out=b;return 0;
fail:
    free(heat);nlte_model_bf_block_free(&b);return -1;
}
