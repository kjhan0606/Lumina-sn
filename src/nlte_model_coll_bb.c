#include "nlte_model_coll_bb.h"

void nlte_model_collision_bb_block_free(NLTEModelCollisionBBBlock *b)
{
    if(!b)return;
    free(b->collisional);free(b->collisional_heat);memset(b,0,sizeof(*b));
}

int nlte_model_element_collision_bb_assemble(NLTEModelCollisionBBBlock *out,
    const NLTEModelElement *e,const NLTEConfig *n,
    const AtomicData *a,const PlasmaState *p,const NLTEModelCollisionBBView *v)
{
    if(!out || out->n_variables || out->collisional || out->collisional_heat ||
       !e || !n || !a || !p || !v || !e->n_variables || !e->super_to_row ||
       !n->model_atom_projection || !n->global_to_nlte_level || !n->fl_to_super ||
       !n->within_sl_frac || !a->level_offset || !a->ion_pop_Z || !a->level_Z ||
       !a->level_ion || !a->level_energy_eV || !a->level_g ||
       !a->model_level_occupation || !p->T_e || !p->n_electron ||
       e->Z!=v->Z || e->shell<0 || p->n_shells<=0 || e->shell>=p->n_shells ||
       v->native_shell!=(size_t)e->shell || v->n_shells!=(size_t)p->n_shells ||
       a->model_occupation_n_shells!=v->n_shells ||
       v->n_levels!=(size_t)a->n_levels ||
       (v->n_pairs && (!v->lower || !v->upper || !v->omega)) ||
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

    const size_t nv=e->n_variables,ns=v->n_shells;
    NLTEModelCollisionBBBlock b={.Z=e->Z,.shell=e->shell,
        .material_generation=e->material_generation,.n_variables=nv};
    long double *heat=calloc(nv,sizeof(*heat));
    b.collisional=calloc(nv*nv,sizeof(double));
    b.collisional_heat=calloc(nv,sizeof(double));
    if(!heat || !b.collisional || !b.collisional_heat)goto fail;
    const double scale=8.63e-6*v->electron_density_cm3/sqrt(v->temperature_K);
    const double kT=8.617333262145e-5*v->temperature_K;
    if(!isfinite(scale) || !isfinite(kT) || kT<=0.)goto fail;
    size_t pair=0;
    for(int ip=0;ip<a->n_ion_pops;ip++){
        if(a->ion_pop_Z[ip]!=e->Z)continue;
        const int begin=a->level_offset[ip],end=a->level_offset[ip+1];
        if(begin<0 || end<begin || end>a->n_levels)goto fail;
        for(int lo=begin;lo<end;lo++)for(int up=lo+1;up<end;up++){
            if(pair>=v->n_pairs || v->lower[pair]!=lo || v->upper[pair]!=up ||
               a->level_Z[lo]!=e->Z || a->level_Z[up]!=e->Z ||
               a->level_ion[lo]!=a->level_ion[up] ||
               !isfinite(v->omega[pair]) || v->omega[pair]<=0. ||
               a->level_g[lo]<=0 || a->level_g[up]<=0)goto fail;
            const double gap=a->level_energy_eV[up]-a->level_energy_eV[lo];
            if(!isfinite(gap) || gap<0.)goto fail;
            const int nl=n->global_to_nlte_level[lo],nu=n->global_to_nlte_level[up];
            if(nl<0 || nu<0 || nl>=n->n_nlte_levels_total || nu>=n->n_nlte_levels_total)goto fail;
            const int sl=n->fl_to_super[nl],su=n->fl_to_super[nu];
            if(sl<0 || su<0 || (size_t)sl>=e->n_super_total ||
               (size_t)su>=e->n_super_total)goto fail;
            const int rl=e->super_to_row[sl],ru=e->super_to_row[su];
            if(rl<0 || ru<0 || (size_t)rl>=nv || (size_t)ru>=nv)goto fail;
            const double wl=a->model_level_occupation[(size_t)lo*ns+e->shell];
            const double wu=a->model_level_occupation[(size_t)up*ns+e->shell];
            const double fl=n->within_sl_frac[(size_t)nl*ns+e->shell];
            const double fu=n->within_sl_frac[(size_t)nu*ns+e->shell];
            if(!isfinite(wl) || !isfinite(wu) || wl<=0. || wu<=0. || wl>1. || wu>1. ||
               !isfinite(fl) || !isfinite(fu) || fl<0. || fu<0. || fl>1. || fu>1.)goto fail;
            if(rl==ru){b.intra_super_pairs++;pair++;continue;}
            /* SUBCOL_MULTI_V6의 HN_F/HN_S 투영·점유확률 비·상세균형. */
            const double base=scale*v->omega[pair];
            const double forward=fl*base*exp(-gap/kT)/a->level_g[lo]*(wu/wl);
            const double reverse=fu*base/a->level_g[up];
            if(!isfinite(forward) || forward<0. ||
               !isfinite(reverse) || reverse<0.)goto fail;
            b.collisional[(size_t)rl*nv+ru]+=forward;
            b.collisional[(size_t)ru*nv+rl]+=reverse;
            const long double energy=(long double)gap*1.602176634e-12L;
            heat[rl]+=energy*forward;heat[ru]-=energy*reverse;
            b.cross_super_pairs++;pair++;
        }
    }
    if(pair!=v->n_pairs || b.cross_super_pairs+b.intra_super_pairs!=pair)goto fail;
    for(size_t j=0;j<nv;j++){
        long double outflow=0.;
        for(size_t i=0;i<nv;i++)if(i!=j){
            if(!isfinite(b.collisional[j*nv+i]) || b.collisional[j*nv+i]<0.)goto fail;
            outflow+=b.collisional[j*nv+i];
        }
        b.collisional[j*nv+j]=-(double)outflow;
        b.collisional_heat[j]=(double)heat[j];
        if(!isfinite(b.collisional[j*nv+j]) || !isfinite(b.collisional_heat[j]))goto fail;
    }
    free(heat);b.pairs=pair;*out=b;return 0;
fail:
    free(heat);nlte_model_collision_bb_block_free(&b);return -1;
}
