#include "nlte_model_se.h"
#include "nlte_model_sum.h"

#include <float.h>
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
    free(b->radiative);free(b->collisional);free(b->radiative_heat);
    free(b->collisional_heat);memset(b,0,sizeof(*b));
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
       out->collisional_heat ||
       !e || !n || !a || !p || !e->n_variables || !e->super_to_row || !e->terminal_to_row ||
       e->material_generation!=a->population_committed_generation ||
       e->n_super_total!=(size_t)n->n_super_total || e->n_ion_pops!=(size_t)a->n_ion_pops ||
       !n->within_sl_frac || nlte_point_bf_view_check(n,a,p,e->shell) ||
       e->n_variables>SIZE_MAX/e->n_variables ||
       e->n_variables*e->n_variables>SIZE_MAX/sizeof(double))return -1;
    size_t nv=e->n_variables,ns=(size_t)p->n_shells;
    NLTEModelBFBlock b={.Z=e->Z,.shell=e->shell,
        .material_generation=e->material_generation,
        .radiation_generation=n->point_bf_rates->radiation_generation,
        .n_variables=nv};
    b.radiative=calloc(nv*nv,sizeof(double));b.collisional=calloc(nv*nv,sizeof(double));
    b.radiative_heat=calloc(nv,sizeof(double));
    b.collisional_heat=calloc(nv,sizeof(double));
    long double *heat=calloc(nv,sizeof(long double));
    long double *collision_heat=calloc(nv,sizeof(long double));
    if(!b.radiative || !b.collisional || !b.radiative_heat || !b.collisional_heat ||
       !heat || !collision_heat)goto fail;
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
            /* SUBCOL_MULTI_V6: 직접 CI/C3는 전체 이온화 문턱을 전자와 교환한다.
             * 광자의 운동에너지 적분 또는 3kT/2로 대체하지 않는다. */
            long double threshold_energy=6.62607015e-27L*n->point_bf_rates->threshold_Hz[r];
            collision_heat[lower]+=threshold_energy*tr.collisional_ionization*f;
            collision_heat[upper]-=threshold_energy*tr.three_body_recombination;
            b.routes++;
        }
    }
    if(close_columns(b.radiative,nv) || close_columns(b.collisional,nv))goto fail;
    for(size_t i=0;i<nv;i++){
        b.radiative_heat[i]=(double)heat[i];b.collisional_heat[i]=(double)collision_heat[i];
        if(!isfinite(b.radiative_heat[i]) || !isfinite(b.collisional_heat[i]))goto fail;
    }
    free(heat);free(collision_heat);*out=b;return 0;
fail:
    free(heat);free(collision_heat);nlte_model_bf_block_free(&b);return -1;
}

void nlte_model_combined_generator_free(NLTEModelCombinedGenerator *b)
{
    if(!b)return;
    free(b->generator);free(b->fixed_residual);memset(b,0,sizeof(*b));
}

static int model_generator_closed(const double *a,size_t n)
{
    if(!a)return -1;
    for(size_t j=0;j<n;j++){
        long double outgoing=0.;
        for(size_t i=0;i<n;i++)if(i!=j){
            double rate=a[j*n+i];
            if(!isfinite(rate) || rate<0.)return -1;
            outgoing+=rate;
        }
        double diagonal=a[j*n+j];
        if(!isfinite(diagonal) || diagonal>0.)return -1;
        long double scale=fmaxl(fabsl(diagonal),outgoing);
        if((scale==0. && diagonal!=0.) ||
           (scale>0. && fabsl((long double)diagonal+outgoing)>
                          64.L*DBL_EPSILON*scale))return -1;
    }
    return 0;
}

int nlte_model_element_generator_combine(NLTEModelCombinedGenerator *out,
    const NLTEModelElement *e,const NLTEModelBFBlock *bf,
    const NLTEModelBBBlock *bb,const NLTEModelCollisionBBBlock *col)
{
    if(!out || out->n_variables || out->generator || out->fixed_residual ||
       !e || !bf || !bb || !col || !e->n_variables || !e->initial_population ||
       !e->material_generation || !bf->radiation_generation ||
       bf->radiation_generation!=bb->radiation_generation ||
       e->n_variables>SIZE_MAX/e->n_variables ||
       e->n_variables*e->n_variables>SIZE_MAX/sizeof(double) ||
       e->n_variables>SIZE_MAX/sizeof(double) ||
       bf->n_variables!=e->n_variables || bb->n_variables!=e->n_variables ||
       col->n_variables!=e->n_variables ||
       bf->Z!=e->Z || bb->Z!=e->Z || col->Z!=e->Z ||
       bf->shell!=e->shell || bb->shell!=e->shell || col->shell!=e->shell ||
       bf->material_generation!=e->material_generation ||
       bb->material_generation!=e->material_generation ||
       col->material_generation!=e->material_generation ||
       !isfinite(e->total_nuclei) || e->total_nuclei<0.)return -1;
    size_t n=e->n_variables;
    if(model_generator_closed(bf->radiative,n) ||
       model_generator_closed(bf->collisional,n) ||
       model_generator_closed(bb->radiative,n) ||
       model_generator_closed(col->collisional,n))return -1;
    long double population_total=0.;
    for(size_t j=0;j<n;j++){
        if(!isfinite(e->initial_population[j]) || e->initial_population[j]<0.)return -1;
        population_total+=e->initial_population[j];
    }
    if(fabsl(population_total-e->total_nuclei)>
       1.e-12L*fmaxl(population_total,(long double)e->total_nuclei))return -1;
    NLTEModelCombinedGenerator b={.Z=e->Z,.shell=e->shell,
        .material_generation=e->material_generation,
        .radiation_generation=bf->radiation_generation,.n_variables=n};
    b.generator=calloc(n*n,sizeof(double));b.fixed_residual=calloc(n,sizeof(double));
    if(!b.generator || !b.fixed_residual)goto fail;
    for(size_t j=0;j<n;j++){
        long double outgoing=0.;
        for(size_t i=0;i<n;i++)if(i!=j){
            long double rate=(long double)bf->radiative[j*n+i]+
                bf->collisional[j*n+i]+bb->radiative[j*n+i]+col->collisional[j*n+i];
            b.generator[j*n+i]=(double)rate;
            if(!isfinite(b.generator[j*n+i]))goto fail;
            outgoing+=rate;
        }
        b.generator[j*n+j]=-(double)outgoing;
        if(!isfinite(b.generator[j*n+j]))goto fail;
    }
    for(size_t i=0;i<n;i++){
        long double residual=0.;
        for(size_t j=0;j<n;j++)
            residual+=(long double)b.generator[j*n+i]*e->initial_population[j];
        b.fixed_residual[i]=(double)residual;
        if(!isfinite(b.fixed_residual[i]))goto fail;
    }
    *out=b;return 0;
fail:
    nlte_model_combined_generator_free(&b);return -1;
}

PopulationStatus nlte_model_element_fixed_field_candidate(
    const NLTEModelCombinedGenerator *combined,const NLTEModelElement *element,
    double *candidate,PopulationGeneratorSolveDiagnostic *diagnostic)
{
    if(!combined || !element || !candidate || !diagnostic ||
       !combined->generator || !element->initial_population ||
       !combined->n_variables || combined->n_variables!=element->n_variables ||
       combined->Z!=element->Z || combined->shell!=element->shell ||
       !combined->material_generation || !combined->radiation_generation ||
       combined->material_generation!=element->material_generation ||
       combined->n_variables>SIZE_MAX/combined->n_variables ||
       combined->n_variables*combined->n_variables>SIZE_MAX/sizeof(double) ||
       combined->n_variables>SIZE_MAX/sizeof(double) ||
       !isfinite(element->total_nuclei) || !(element->total_nuclei>0.) ||
       candidate==element->initial_population || candidate==combined->generator ||
       model_generator_closed(combined->generator,combined->n_variables))
        return POP_SOLVE_FAILED;
    size_t n=combined->n_variables;
    double *work=malloc(n*sizeof(*work));
    if(!work)return POP_SOLVE_FAILED;
    PopulationStatus status=population_generator_stationary_gth(
        combined->generator,n,element->total_nuclei,work,diagnostic);
    if(status==POP_OK){
        long double total=0.;
        for(size_t i=0;i<n;i++){
            if(!isfinite(work[i]) || !(work[i]>0.)){status=POP_NONFINITE;break;}
            total+=work[i];
        }
        if(status==POP_OK &&
           fabsl(total-element->total_nuclei)>
           1.e-12L*fmaxl(total,(long double)element->total_nuclei))
            status=POP_SOLVE_FAILED;
    }
    if(status==POP_OK)memcpy(candidate,work,n*sizeof(*candidate));
    free(work);
    return status;
}
