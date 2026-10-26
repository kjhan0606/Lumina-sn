#include "nlte_model_bb.h"

void nlte_model_bb_block_free(NLTEModelBBBlock *b)
{
    if(!b)return;
    free(b->radiative);free(b->radiative_heat);free(b->native_thermal_heat);memset(b,0,sizeof(*b));
}

int nlte_model_element_bb_assemble_with_native_thermal(NLTEModelBBBlock *out,
    const NLTEModelElement *e,const NLTEConfig *n,
    const AtomicData *a,const PlasmaState *p,const NLTEModelBBView *v,
    const double *native_thermal_factor)
{
    if(!out || out->n_variables || out->radiative || out->radiative_heat || out->native_thermal_heat ||
       !e || !n || !a || !p || !v || !e->n_variables || !e->super_to_row ||
       !n->model_atom_projection || !n->global_to_nlte_level || !n->fl_to_super ||
       !n->within_sl_frac || !a->level_Z || !a->level_ion ||
       n->n_nlte_levels_total!=a->n_levels ||
       e->n_super_total!=(size_t)n->n_super_total || e->n_ion_pops!=(size_t)a->n_ion_pops ||
       e->shell<0 || p->n_shells<=0 || e->shell>=p->n_shells ||
       v->native_shell!=(size_t)e->shell || v->n_shells!=(size_t)p->n_shells ||
       v->n_levels!=(size_t)a->n_levels || !v->n_lines ||
       !v->lower || !v->upper || !v->frequency_Hz || !v->absorption ||
       !v->stimulated || !v->spontaneous || !v->status ||
       !v->material_generation || v->material_generation!=e->material_generation ||
       v->material_generation!=a->population_committed_generation ||
       n->population_committed_generation!=a->population_committed_generation ||
       !v->radiation_generation || n->radfield_view_status!=RADIATION_FIELD_VIEW_OK ||
       n->radfield_view.generation!=v->radiation_generation ||
       (a->model_level_occupation &&
        (a->partition_stamp.status!=POP_OK || v->material_model_sha256[64]!='\0' ||
         a->partition_stamp.atomic_model_sha256[64]!='\0' ||
         !v->material_model_sha256[0] ||
         strcmp(v->material_model_sha256,a->partition_stamp.atomic_model_sha256))) ||
       e->n_variables>SIZE_MAX/e->n_variables ||
       e->n_variables*e->n_variables>SIZE_MAX/sizeof(double) ||
       e->n_variables>SIZE_MAX/sizeof(long double))return -1;
    const size_t nv=e->n_variables,ns=(size_t)p->n_shells;
    NLTEModelBBBlock b={.Z=e->Z,.shell=e->shell,
        .material_generation=e->material_generation,
        .radiation_generation=v->radiation_generation,
        .n_variables=nv};
    long double *heat=calloc(nv,sizeof(*heat));
    long double *native_heat=NULL;
    b.radiative=calloc(nv*nv,sizeof(double));b.radiative_heat=calloc(nv,sizeof(double));
    if(!heat || !b.radiative || !b.radiative_heat)goto fail;
    if(native_thermal_factor){
        native_heat=calloc(nv,sizeof(*native_heat));
        b.native_thermal_heat=calloc(nv,sizeof(*b.native_thermal_heat));
        if(!native_heat || !b.native_thermal_heat)goto fail;
    }
    for(size_t i=0;i<v->n_lines;i++){
        int lo=v->lower[i],up=v->upper[i];
        if(lo<0 || up<0 || lo>=a->n_levels || up>=a->n_levels || lo==up ||
           a->level_Z[lo]!=a->level_Z[up] || a->level_ion[lo]!=a->level_ion[up] ||
           !isfinite(v->frequency_Hz[i]) || v->frequency_Hz[i]<=0. ||
           !isfinite(v->absorption[i]) || v->absorption[i]<0. ||
           !isfinite(v->stimulated[i]) || v->stimulated[i]<0. ||
           !isfinite(v->spontaneous[i]) || v->spontaneous[i]<0. ||
           (native_thermal_factor && (!isfinite(native_thermal_factor[i]) ||
                                     native_thermal_factor[i]<0.)) ||
           (v->status[i]!=1 && v->status[i]!=2) ||
           (v->status[i]==2 && (v->absorption[i]!=0. || v->stimulated[i]!=0.)))goto fail;
        if(a->level_Z[lo]!=e->Z)continue;
        int nl=n->global_to_nlte_level[lo],nu=n->global_to_nlte_level[up];
        if(nl<0 || nu<0 || nl>=n->n_nlte_levels_total || nu>=n->n_nlte_levels_total)goto fail;
        int sl=n->fl_to_super[nl],su=n->fl_to_super[nu];
        if(sl<0 || su<0 || (size_t)sl>=e->n_super_total || (size_t)su>=e->n_super_total)goto fail;
        int lower=e->super_to_row[sl],upper=e->super_to_row[su];
        double fl=n->within_sl_frac[(size_t)nl*ns+e->shell];
        double fu=n->within_sl_frac[(size_t)nu*ns+e->shell];
        if(lower<0 || upper<0 || !isfinite(fl) || fl<0. || fl>1. ||
           !isfinite(fu) || fu<0. || fu>1.)goto fail;
        double forward=fl*v->absorption[i];
        double reverse=fu*(v->spontaneous[i]+v->stimulated[i]);
        if(!isfinite(forward) || forward<0. || !isfinite(reverse) || reverse<0.)goto fail;
        if(lower==upper)b.intra_super_lines++;
        else{
            b.radiative[(size_t)lower*nv+upper]+=forward;
            b.radiative[(size_t)upper*nv+lower]+=reverse;
        }
        long double energy=6.62607015e-27L*v->frequency_Hz[i];
        heat[lower]+=energy*forward;heat[upper]-=energy*reverse;
        if(native_heat){
            long double scaled_energy=energy*native_thermal_factor[i];
            native_heat[lower]+=scaled_energy*forward;native_heat[upper]-=scaled_energy*reverse;
        }
        b.lines++;
    }
    for(size_t j=0;j<nv;j++){
        long double outflow=0.;
        for(size_t i=0;i<nv;i++)if(i!=j){
            if(!isfinite(b.radiative[j*nv+i]) || b.radiative[j*nv+i]<0.)goto fail;
            outflow+=b.radiative[j*nv+i];
        }
        b.radiative[j*nv+j]=-(double)outflow;
        b.radiative_heat[j]=(double)heat[j];
        if(native_heat){
            b.native_thermal_heat[j]=(double)native_heat[j];
            if(!isfinite(b.native_thermal_heat[j]))goto fail;
        }
        if(!isfinite(b.radiative[j*nv+j]) || !isfinite(b.radiative_heat[j]))goto fail;
    }
    free(heat);free(native_heat);*out=b;return 0;
fail:
    free(heat);free(native_heat);nlte_model_bb_block_free(&b);return -1;
}

int nlte_model_element_bb_assemble(NLTEModelBBBlock *out,
    const NLTEModelElement *element,const NLTEConfig *nlte,
    const AtomicData *atom,const PlasmaState *plasma,const NLTEModelBBView *view)
{
    return nlte_model_element_bb_assemble_with_native_thermal(out,element,nlte,atom,plasma,view,NULL);
}
