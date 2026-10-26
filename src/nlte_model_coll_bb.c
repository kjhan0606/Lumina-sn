#include "nlte_model_coll_bb.h"

void nlte_model_dissolution_free(NLTEModelDissolutionContext *c)
{
    if(!c)return;
    free(c->ground_row);free(c->saha);free(c->threshold_Hz);memset(c,0,sizeof(*c));
}

int nlte_model_dissolution_init(NLTEModelDissolutionContext *out,
    const NLTEModelElement *e,const NLTEConfig *n,
    const AtomicData *a,const PlasmaState *p)
{
    if(!out || out->n_levels || out->ground_row || out->saha || out->threshold_Hz ||
       !e || !n || !a || !p || !e->terminal_to_row || !e->super_to_row ||
       e->material_generation!=a->population_committed_generation ||
       !a->ion_pop_stage || !a->ion_pop_Z || !a->level_offset || !a->level_Z ||
       !a->level_ion || !n->global_to_nlte_level || !n->fl_to_super ||
       nlte_point_bf_view_check(n,a,p,e->shell))return -1;
    NLTEModelDissolutionContext c={.Z=e->Z,.shell=e->shell,
        .n_levels=(size_t)a->n_levels,.n_variables=e->n_variables,
        .material_generation=e->material_generation,
        .temperature_K=p->T_e[e->shell],.electron_density_cm3=p->n_electron[e->shell]};
    memcpy(c.material_model_sha256,a->partition_stamp.atomic_model_sha256,65);
    c.ground_row=malloc(c.n_levels*sizeof(*c.ground_row));
    c.saha=calloc(c.n_levels,sizeof(*c.saha));
    c.threshold_Hz=calloc(c.n_levels,sizeof(*c.threshold_Hz));
    if(!c.ground_row || !c.saha || !c.threshold_Hz)goto fail;
    for(size_t gl=0;gl<c.n_levels;gl++)c.ground_row[gl]=-1;
    const BfPointRateView *v=n->point_bf_rates;
    for(int gl=0;gl<a->n_levels;gl++){
        if(a->level_Z[gl]!=e->Z)continue;
        size_t begin,end;
        if(bf_rate_point_row(v,c.n_levels,v->n_shells,(size_t)gl,
                            c.material_generation,&begin,&end))goto fail;
        for(size_t r=begin;r<end;r++){
            if(v->phot_grid[r]!=0)continue;
            /* 현재 입력의 primary는 FL당 하나다. 중복을 묵시 합산하지 않는다. */
            if(c.ground_row[gl]>=0)goto fail;
            BfPointTransition transition;
            if(nlte_point_bf_transition(n,a,p,gl,r,e->shell,&transition))goto fail;
            int ip=-1,row=-1;
            for(int k=0;k<a->n_ion_pops;k++)if(a->ion_pop_Z[k]==e->Z &&
                a->ion_pop_stage[k]==a->level_ion[gl]+1){if(ip>=0)goto fail;ip=k;}
            if(ip<0)goto fail;
            if(transition.target_level>=0){
                int target=transition.target_level;
                if(target!=a->level_offset[ip] || target>=a->level_offset[ip+1])goto fail;
                int ng=n->global_to_nlte_level[target];
                if(ng<0 || ng>=n->n_nlte_levels_total)goto fail;
                int sl=n->fl_to_super[ng];
                if(sl<0 || (size_t)sl>=e->n_super_total)goto fail;
                row=e->super_to_row[sl];
            }else{
                if(transition.target_ion!=ip)goto fail;
                row=e->terminal_to_row[ip];
            }
            double saha=v->saha_lower_per_upper[r*v->n_shells+e->shell];
            if(row<0 || (size_t)row>=e->n_variables || !isfinite(saha) || saha<0. ||
               !isfinite(v->threshold_Hz[r]) || v->threshold_Hz[r]<=0.)goto fail;
            c.ground_row[gl]=row;c.saha[gl]=saha;c.threshold_Hz[gl]=v->threshold_Hz[r];
        }
        if(c.ground_row[gl]<0)goto fail;
    }
    *out=c;return 0;
fail:
    nlte_model_dissolution_free(&c);return -1;
}

void nlte_model_collision_bb_block_free(NLTEModelCollisionBBBlock *b)
{
    if(!b)return;
    free(b->collisional);free(b->collisional_heat);
    free(b->dissolution);free(b->dissolution_heat);memset(b,0,sizeof(*b));
}

static int collision_metadata_check(
    const NLTEModelElement *e,const NLTEConfig *n,
    const AtomicData *a,const PlasmaState *p,const NLTEModelCollisionBBView *v)
{
    if(!e || !n || !a || !p || !v || !e->n_variables || !e->super_to_row ||
       !n->model_atom_projection || !n->global_to_nlte_level || !n->fl_to_super ||
       !n->within_sl_frac || !a->level_offset || !a->ion_pop_Z || !a->level_Z ||
       !a->level_ion || !a->level_energy_eV || !a->level_g ||
       !a->model_level_occupation || !p->T_e || !p->n_electron ||
       e->Z!=v->Z || e->shell<0 || p->n_shells<=0 || e->shell>=p->n_shells ||
       v->native_shell!=(size_t)e->shell || v->n_shells!=(size_t)p->n_shells ||
       a->model_occupation_n_shells!=v->n_shells ||
       v->n_levels!=(size_t)a->n_levels ||
       !v->material_generation ||
       v->material_generation!=e->material_generation ||
       v->material_generation!=a->population_committed_generation ||
       n->population_committed_generation!=v->material_generation ||
       (a->partition_stamp.status!=POP_OK ||
        v->material_model_sha256[64]!='\0' ||
        a->partition_stamp.atomic_model_sha256[64]!='\0' ||
        !v->material_model_sha256[0] ||
        strcmp(v->material_model_sha256,a->partition_stamp.atomic_model_sha256)) ||
       v->temperature_K!=p->T_e[e->shell] ||
       v->electron_density_cm3!=p->n_electron[e->shell] ||
       !isfinite(v->temperature_K) || v->temperature_K<=0. ||
       !isfinite(v->electron_density_cm3) || v->electron_density_cm3<=0. ||
       e->n_variables>SIZE_MAX/e->n_variables ||
       e->n_variables*e->n_variables>SIZE_MAX/sizeof(double))return -1;

    const size_t nv=e->n_variables;
    const NLTEModelDissolutionContext *d=v->dissolution;
    if(d && (d->Z!=e->Z || d->shell!=e->shell || d->n_levels!=v->n_levels ||
       d->n_variables!=nv || d->material_generation!=v->material_generation ||
       d->temperature_K!=v->temperature_K || d->electron_density_cm3!=v->electron_density_cm3 ||
       d->material_model_sha256[64]!='\0' ||
       strcmp(d->material_model_sha256,v->material_model_sha256) ||
       !d->ground_row || !d->saha || !d->threshold_Hz))return -1;
    return 0;
}

struct NLTEModelCollisionBBStream {
    const NLTEModelElement *element;
    const NLTEConfig *nlte;
    const AtomicData *atom;
    const PlasmaState *plasma;
    NLTEModelCollisionBBView metadata;
    NLTEModelCollisionBBBlock block;
    long double *heat;
    uint64_t trial_id;
    double scale,kT;
    int ion,next_lower,next_upper,failed,finished;
};

static int stream_seek(NLTEModelCollisionBBStream *s)
{
    const AtomicData *a=s->atom;
    while(s->ion<a->n_ion_pops){
        int ip=s->ion,begin=a->level_offset[ip],end=a->level_offset[ip+1];
        if(begin<0 || end<begin || end>a->n_levels)return -1;
        if(a->ion_pop_Z[ip]==s->element->Z && end-begin>=2){
            s->next_lower=begin;s->next_upper=begin+1;return 0;
        }
        s->ion++;
    }
    s->next_lower=s->next_upper=-1;return 0;
}

void nlte_model_collision_bb_stream_free(NLTEModelCollisionBBStream *s)
{
    if(!s)return;
    nlte_model_collision_bb_block_free(&s->block);free(s->heat);free(s);
}

int nlte_model_collision_bb_stream_begin(NLTEModelCollisionBBStream **out,
    const NLTEModelElement *e,const NLTEConfig *n,const AtomicData *a,
    const PlasmaState *p,const NLTEModelCollisionBBView *v,uint64_t trial_id)
{
    if(!out || *out || !trial_id || collision_metadata_check(e,n,a,p,v))return -1;
    size_t expected=0;
    for(int ip=0;ip<a->n_ion_pops;ip++){
        int begin=a->level_offset[ip],end=a->level_offset[ip+1];
        if(begin<0 || end<begin || end>a->n_levels)return -1;
        if(a->ion_pop_Z[ip]!=e->Z)continue;
        size_t count=(size_t)(end-begin);
        if(count && count-1>SIZE_MAX/count)return -1;
        size_t pairs=count ? count*(count-1)/2:0;
        if(pairs>SIZE_MAX-expected)return -1;
        expected+=pairs;
    }
    if(expected!=v->n_pairs)return -1;
    NLTEModelCollisionBBStream *s=calloc(1,sizeof(*s));
    if(!s)return -1;
    s->element=e;s->nlte=n;s->atom=a;s->plasma=p;s->metadata=*v;s->trial_id=trial_id;
    s->scale=8.63e-6*v->electron_density_cm3/sqrt(v->temperature_K);
    s->kT=8.617333262145e-5*v->temperature_K;
    if(!isfinite(s->scale) || !isfinite(s->kT) || s->kT<=0.)goto fail;
    s->metadata.lower=NULL;s->metadata.upper=NULL;s->metadata.omega=NULL;
    size_t nv=e->n_variables;
    s->block=(NLTEModelCollisionBBBlock){.Z=e->Z,.shell=e->shell,
        .material_generation=e->material_generation,.n_variables=nv};
    s->heat=calloc(nv,sizeof(*s->heat));
    s->block.collisional=calloc(nv*nv,sizeof(double));
    s->block.collisional_heat=calloc(nv,sizeof(double));
    if(!s->heat || !s->block.collisional || !s->block.collisional_heat)goto fail;
    if(v->dissolution){
        s->block.dissolution=calloc(nv*nv,sizeof(double));
        s->block.dissolution_heat=calloc(nv,sizeof(double));
        if(!s->block.dissolution || !s->block.dissolution_heat)goto fail;
    }
    if(stream_seek(s))goto fail;
    *out=s;return 0;
fail:
    nlte_model_collision_bb_stream_free(s);return -1;
}

static int collision_pair(NLTEModelCollisionBBStream *s,int lo,int up,double omega)
{
    const NLTEModelElement *e=s->element;
    const NLTEConfig *n=s->nlte;
    const AtomicData *a=s->atom;
    const NLTEModelCollisionBBView *v=&s->metadata;
    const NLTEModelDissolutionContext *d=v->dissolution;
    NLTEModelCollisionBBBlock *b=&s->block;
    const size_t nv=e->n_variables,ns=v->n_shells;
    long double *heat=s->heat;
    const double scale=s->scale,kT=s->kT;
            if(lo<0 || up<=lo || up>=a->n_levels ||
               a->level_Z[lo]!=e->Z || a->level_Z[up]!=e->Z ||
               a->level_ion[lo]!=a->level_ion[up] ||
               !isfinite(omega) || omega<=0. ||
               a->level_g[lo]<=0 || a->level_g[up]<=0)return -1;
            const double gap=a->level_energy_eV[up]-a->level_energy_eV[lo];
            if(!isfinite(gap) || gap<0.)return -1;
            const int nl=n->global_to_nlte_level[lo],nu=n->global_to_nlte_level[up];
            if(nl<0 || nu<0 || nl>=n->n_nlte_levels_total || nu>=n->n_nlte_levels_total)return -1;
            const int sl=n->fl_to_super[nl],su=n->fl_to_super[nu];
            if(sl<0 || su<0 || (size_t)sl>=e->n_super_total ||
               (size_t)su>=e->n_super_total)return -1;
            const int rl=e->super_to_row[sl],ru=e->super_to_row[su];
            if(rl<0 || ru<0 || (size_t)rl>=nv || (size_t)ru>=nv)return -1;
            const double wl=a->model_level_occupation[(size_t)lo*ns+e->shell];
            const double wu=a->model_level_occupation[(size_t)up*ns+e->shell];
            const double fl=n->within_sl_frac[(size_t)nl*ns+e->shell];
            const double fu=n->within_sl_frac[(size_t)nu*ns+e->shell];
            if(!isfinite(wl) || !isfinite(wu) || wl<=0. || wu<=0. || wl>1. || wu>1. ||
               !isfinite(fl) || !isfinite(fu) || fl<0. || fu<0. || fl>1. || fu>1.)return -1;
            const double base=scale*omega;
            if(d){
                int ground=d->ground_row[lo];
                double lost=1.-wu/wl;
                if(ground<0 || (size_t)ground>=nv || ground==rl ||
                   !isfinite(lost) || lost<0. || !isfinite(d->saha[lo]) || d->saha[lo]<0. ||
                   !isfinite(d->threshold_Hz[lo]) || d->threshold_Hz[lo]<=0.)return -1;
                /* within_sl_frac는 점유확률을 포함한 LTE FL/SL 비다.
                 * BF saha는 이미 상위 ground FL / ground SL 투영을 포함한다. */
                double rate=base*exp(-gap/kT)/a->level_g[lo]*lost;
                double forward=fl*rate,reverse=d->saha[lo]*rate;
                if(!isfinite(forward) || forward<0. || !isfinite(reverse) || reverse<0.)return -1;
                b->dissolution[(size_t)rl*nv+ground]+=forward;
                b->dissolution[(size_t)ground*nv+rl]+=reverse;
                const double threshold_energy=6.62607015e-27*d->threshold_Hz[lo];
                b->dissolution_heat[rl]+=threshold_energy*forward;
                b->dissolution_heat[ground]-=threshold_energy*reverse;
                if(rate>0.){
                    b->dissolution_pairs++;
                    if(rl==ru)b->intra_super_dissolution_pairs++;
                }
            }
            if(rl==ru){b->intra_super_pairs++;return 0;}
            /* SUBCOL_MULTI_V6의 HN_F/HN_S 투영·점유확률 비·상세균형. */
            const double forward=fl*base*exp(-gap/kT)/a->level_g[lo]*(wu/wl);
            const double reverse=fu*base/a->level_g[up];
            if(!isfinite(forward) || forward<0. ||
               !isfinite(reverse) || reverse<0.)return -1;
            b->collisional[(size_t)rl*nv+ru]+=forward;
            b->collisional[(size_t)ru*nv+rl]+=reverse;
            const long double energy=(long double)gap*1.602176634e-12L;
            heat[rl]+=energy*forward;heat[ru]-=energy*reverse;
            b->cross_super_pairs++;return 0;
}

int nlte_model_collision_bb_stream_push(NLTEModelCollisionBBStream *s,uint64_t trial_id,
    const int32_t *lower,const int32_t *upper,const double *omega,size_t count)
{
    if(!s)return -1;
    if(s->failed || s->finished || !trial_id || trial_id!=s->trial_id ||
       (count && (!lower || !upper || !omega)) ||
       collision_metadata_check(s->element,s->nlte,s->atom,s->plasma,&s->metadata))goto fail;
    for(size_t k=0;k<count;k++){
        if(s->next_lower<0 || lower[k]!=s->next_lower || upper[k]!=s->next_upper ||
           collision_pair(s,lower[k],upper[k],omega[k]))goto fail;
        s->block.pairs++;
        int end=s->atom->level_offset[s->ion+1];
        s->next_upper++;
        if(s->next_upper>=end){
            s->next_lower++;s->next_upper=s->next_lower+1;
            if(s->next_upper>=end){s->ion++;if(stream_seek(s))goto fail;}
        }
    }
    return 0;
fail:
    s->failed=1;return -1;
}

int nlte_model_collision_bb_stream_finish(NLTEModelCollisionBBStream *s,
    uint64_t trial_id,NLTEModelCollisionBBBlock *out)
{
    if(!s)return -1;
    if(s->failed || s->finished || trial_id!=s->trial_id || !out || out->n_variables ||
       out->collisional || out->collisional_heat || out->dissolution || out->dissolution_heat ||
       s->next_lower>=0 || s->block.pairs!=s->metadata.n_pairs ||
       collision_metadata_check(s->element,s->nlte,s->atom,s->plasma,&s->metadata))goto fail;
    NLTEModelCollisionBBBlock *b=&s->block;
    const size_t nv=b->n_variables;
    const int d=s->metadata.dissolution!=NULL;
    if(b->cross_super_pairs+b->intra_super_pairs!=b->pairs)goto fail;
    for(size_t j=0;j<nv;j++){
        long double outflow=0.,dissolution_outflow=0.;
        for(size_t i=0;i<nv;i++)if(i!=j){
            if(d){
                double value=b->dissolution[j*nv+i];
                if(!isfinite(value) || value<0.)goto fail;
                dissolution_outflow+=value;b->collisional[j*nv+i]+=value;
            }
            if(!isfinite(b->collisional[j*nv+i]) || b->collisional[j*nv+i]<0.)goto fail;
            outflow+=b->collisional[j*nv+i];
        }
        b->collisional[j*nv+j]=-(double)outflow;
        b->collisional_heat[j]=(double)s->heat[j];
        if(d){
            b->dissolution[j*nv+j]=-(double)dissolution_outflow;
            b->collisional_heat[j]+=b->dissolution_heat[j];
            if(!isfinite(b->dissolution[j*nv+j]) || !isfinite(b->dissolution_heat[j]))goto fail;
        }
        if(!isfinite(b->collisional[j*nv+j]) || !isfinite(b->collisional_heat[j]))goto fail;
    }
    *out=*b;memset(b,0,sizeof(*b));s->finished=1;return 0;
fail:
    s->failed=1;return -1;
}

int nlte_model_element_collision_bb_assemble(NLTEModelCollisionBBBlock *out,
    const NLTEModelElement *e,const NLTEConfig *n,const AtomicData *a,
    const PlasmaState *p,const NLTEModelCollisionBBView *v)
{
    NLTEModelCollisionBBStream *stream=NULL;
    if(!out || out->n_variables || out->collisional || out->collisional_heat ||
       out->dissolution || out->dissolution_heat ||
       nlte_model_collision_bb_stream_begin(&stream,e,n,a,p,v,1))return -1;
    int rc=nlte_model_collision_bb_stream_push(stream,1,v->lower,v->upper,v->omega,v->n_pairs);
    if(!rc)rc=nlte_model_collision_bb_stream_finish(stream,1,out);
    nlte_model_collision_bb_stream_free(stream);return rc;
}
