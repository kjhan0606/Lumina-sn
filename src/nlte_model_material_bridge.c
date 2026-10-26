#include "nlte_model_material_bridge.h"
#include <stdatomic.h>

struct NLTEModelMaterialBridge {
    AtomicData atom;
    NLTEConfig nlte;
    PlasmaState plasma;
    PopulationOccupationModel occupation_model;
    double *occupation_parameters,*occupation,*response;
    int32_t *columns;
    int *column_ion,*column_super,*super_column;
    size_t n_columns;
    NLTEPopulationCandidate trial;
    BfPointRateView bf;
    uint64_t trial_id;
    size_t active_elements;
    uint8_t *finished;
    _Atomic int failed;
    int closed;
};

struct NLTEModelMaterialElement {
    NLTEModelMaterialBridge *owner;
    uint64_t trial_id;
    NLTEModelElement element;
    NLTEModelBFBlock bf;
    NLTEModelBBBlock bb;
    NLTEModelDissolutionContext dissolution;
    NLTEModelCollisionBBStream *stream;
    int finished;
};

void nlte_model_material_bridge_free(NLTEModelMaterialBridge *b)
{
    if(!b || b->active_elements)return;
    nlte_population_candidate_free(&b->trial);
    nlte_free(&b->nlte);free_atomic_data(&b->atom);
    free(b->plasma.T_e);free(b->plasma.n_electron);free(b->plasma.rho);
    free(b->occupation);free(b->occupation_parameters);free(b->response);free(b->columns);
    free(b->column_ion);free(b->column_super);free(b->super_column);free(b->finished);free(b);
}

static int read_occupation(NLTEModelMaterialBridge *b,const char *path,const char *parameters)
{
    FILE *f=fopen(path,"rb");char tag[8];uint32_t shape[2];
    if(!f)return -1;
    int failed=fread(tag,1,8,f)!=8 || memcmp(tag,"LMATOCC1",8) ||
        fread(shape,4,2,f)!=2 || shape[0]!=(uint32_t)b->atom.n_levels ||
        shape[1]!=(uint32_t)b->plasma.n_shells;
    size_t count=(size_t)b->atom.n_levels*b->plasma.n_shells;
    if(!failed){
        b->occupation=malloc(count*sizeof(double));
        failed=!b->occupation || fread(b->occupation,8,count,f)!=count || fgetc(f)!=EOF;
    }
    fclose(f);if(failed)return -1;
    f=fopen(parameters,"rb");uint32_t nl;
    if(!f)return -1;
    failed=fread(tag,1,8,f)!=8 || memcmp(tag,"LMOCPAR1",8) ||
        fread(&nl,4,1,f)!=1 || nl!=(uint32_t)b->atom.n_levels;
    if(!failed){
        b->occupation_parameters=malloc(2*(size_t)nl*sizeof(double));
        failed=!b->occupation_parameters ||
            fread(b->occupation_parameters,8,2*(size_t)nl,f)!=2*(size_t)nl || fgetc(f)!=EOF;
    }
    fclose(f);if(failed)return -1;
    b->occupation_model=(PopulationOccupationModel){nl,b->occupation_parameters,b->occupation_parameters+nl};
    return 0;
}

NLTEModelMaterialBridge *nlte_model_material_bridge_open(const char *directory,
    const char *nodes,const char *occupation,const char *parameters)
{
    if(!directory || !nodes || !occupation || !parameters)return NULL;
    NLTEModelMaterialBridge *b=calloc(1,sizeof(*b));
    FILE *f=NULL;double *physical=NULL,*values=NULL;
    if(!b)return NULL;
    atomic_init(&b->failed,0);
    f=fopen(nodes,"rb");char tag[8];uint32_t shape[3];double epoch;
    if(!f || fread(tag,1,8,f)!=8 || memcmp(tag,"LUMNOD01",8) ||
       fread(shape,4,3,f)!=3 || fread(&epoch,8,1,f)!=1 || !isfinite(epoch) || epoch<=0. ||
       !shape[0] || shape[0]>90 || !shape[1] || shape[1]>10000 || !shape[2])goto fail;
    size_t ns=shape[0],nc=shape[1];b->n_columns=nc;b->plasma.n_shells=(int)ns;
    if(load_atomic_data(&b->atom,directory,(int)ns) || shape[2]!=(uint32_t)b->atom.n_levels)goto fail;
    OpacityState opacity={.n_lines=b->atom.n_lines};
    if(nlte_init_model_atom(&b->nlte,&b->atom,&opacity,(int)ns))goto fail;
    b->super_column=malloc((size_t)b->nlte.n_super_total*sizeof(int));
    if(!b->super_column)goto fail;
    for(int sl=0;sl<b->nlte.n_super_total;sl++)b->super_column[sl]=-1;
    b->columns=malloc(nc*3*sizeof(int32_t));physical=malloc(ns*4*sizeof(double));
    values=malloc(ns*nc*sizeof(double));
    b->plasma.T_e=malloc(ns*sizeof(double));b->plasma.n_electron=malloc(ns*sizeof(double));
    b->plasma.rho=malloc(ns*sizeof(double));
    b->column_ion=malloc(nc*sizeof(int));b->column_super=malloc(nc*sizeof(int));
    b->finished=calloc((size_t)b->atom.n_elements*ns,1);
    b->response=malloc(nc*ns*sizeof(double));
    if(!b->columns || !physical || !values || !b->plasma.T_e || !b->plasma.n_electron ||
       !b->plasma.rho || !b->column_ion || !b->column_super || !b->finished || !b->response ||
       fread(b->columns,4,nc*3,f)!=nc*3 || fread(physical,8,ns*4,f)!=ns*4 ||
       fread(values,8,ns*nc,f)!=ns*nc || fgetc(f)!=EOF)goto fail;
    fclose(f);f=NULL;
    for(size_t s=0;s<ns;s++){
        b->plasma.T_e[s]=physical[4*s+1];b->plasma.n_electron[s]=physical[4*s+2];
        b->plasma.rho[s]=physical[4*s+3];
    }
    if(read_occupation(b,occupation,parameters) ||
       lumina_publish_seed_te(&b->plasma,"private model material bridge seed") ||
       lumina_import_superlevel_material_with_occupation(&b->nlte,&b->atom,&b->plasma,
           (int)nc,b->columns,values,b->occupation)!=POP_OK)goto fail;
    for(size_t c=0;c<nc;c++){
        int Z=b->columns[3*c],q=b->columns[3*c+1],sl=b->columns[3*c+2];
        int ip=-1,super=-1;
        if(q<=0)goto fail; /* 현재 선형 제거 계약은 전 이온이 양이온인 원 모형이다. */
        for(int k=0;k<b->atom.n_ion_pops;k++)if(b->atom.ion_pop_Z[k]==Z && b->atom.ion_pop_stage[k]==q){
            if(ip>=0)goto fail;
            ip=k;
        }
        if(ip<0)goto fail;
        if(sl<0){if(sl!=-1 || b->atom.level_offset[ip]!=b->atom.level_offset[ip+1])goto fail;}
        else{
            for(int gl=b->atom.level_offset[ip];gl<b->atom.level_offset[ip+1];gl++)
                if(b->atom.level_super[gl]==sl){
                    int ng=b->nlte.global_to_nlte_level[gl];
                    if(ng<0)goto fail;
                    int current=b->nlte.fl_to_super[ng];
                    if(super>=0 && super!=current)goto fail;
                    super=current;
                }
            if(super<0)goto fail;
        }
        b->column_ion[c]=ip;b->column_super[c]=super;
        if(super>=0){
            if(b->super_column[super]>=0)goto fail;
            b->super_column[super]=(int)c;
        }
    }
    for(int sl=0;sl<b->nlte.n_super_total;sl++)if(b->super_column[sl]<0)goto fail;
    free(physical);free(values);return b;
fail:
    if(f)fclose(f);
    free(physical);free(values);nlte_model_material_bridge_free(b);return NULL;
}

int nlte_model_material_bridge_ion_index(const NLTEModelMaterialBridge *b,int Z,int charge)
{
    if(!b)return -1;
    for(int i=0;i<b->atom.n_ion_pops;i++)
        if(b->atom.ion_pop_Z[i]==Z && b->atom.ion_pop_stage[i]==charge)return i;
    return -1;
}

int nlte_model_material_bridge_prepare(NLTEModelMaterialBridge *b,uint64_t id,
    const double *te,const double *ne,const int32_t *columns,size_t n_columns,
    const double *pop,const double *occupation,
    const double *fraction,const BfPointRateView *bf)
{
    if(!b || b->active_elements || !te || !ne || !columns || n_columns!=b->n_columns ||
       memcmp(columns,b->columns,3*n_columns*sizeof(int32_t)) ||
       !pop || !occupation || !fraction || !bf ||
       id<=b->trial_id || id<=b->atom.population_committed_generation ||
       bf->material_generation!=id || !bf->radiation_generation ||
       bf->n_levels!=(size_t)b->atom.n_levels || bf->n_shells!=(size_t)b->plasma.n_shells)return -1;
    b->trial_id=id;b->failed=1;
    nlte_population_candidate_free(&b->trial);
    size_t ns=(size_t)b->plasma.n_shells;
    for(size_t s=0;s<ns;s++)if(!isfinite(te[s]) || te[s]<=0. || !isfinite(ne[s]) || ne[s]<=0.)return -1;
    if(nlte_population_candidate_begin_model(&b->trial,&b->nlte,&b->atom,&b->plasma,
            te,ns,id,id,&b->occupation_model)!=NLTE_CANDIDATE_OK)return -1;
    NLTEPopulationCandidate *c=&b->trial;
    memcpy(c->electron_density,ne,ns*sizeof(double));
    memset(c->ion_population,0,c->n_ion_values*sizeof(double));
    for(size_t k=0;k<b->n_columns;k++)for(size_t s=0;s<ns;s++){
        double value=pop[k*ns+s];
        if(!isfinite(value) || value<0.)return -1;
        c->ion_population[(size_t)b->column_ion[k]*ns+s]+=value;
    }
    for(int e=0;e<b->atom.n_elements;e++)for(size_t s=0;s<ns;s++){
        long double total=0.;
        for(int ip=0;ip<b->atom.n_ion_pops;ip++)if(b->atom.ion_pop_Z[ip]==b->atom.element_Z[e])
            total+=c->ion_population[(size_t)ip*ns+s];
        double expected=lumina_element_number_density(&b->atom,&b->plasma,e,(int)s,(int)ns);
        if(!isfinite(expected) || expected<0. || fabsl(total-expected)>1.e-12L*fmaxl(total,expected))return -1;
    }
    if(nlte_population_candidate_prepare_thermodynamics(c)!=NLTE_CANDIDATE_OK)return -1;
    for(int gl=0;gl<b->atom.n_levels;gl++){
        int ng=c->nlte.global_to_nlte_level[gl];
        if(ng<0 || ng>=c->nlte.n_nlte_levels_total)return -1;
        int sl=c->nlte.fl_to_super[ng];
        if(sl<0 || sl>=c->nlte.n_super_total || b->super_column[sl]<0)return -1;
        for(size_t s=0;s<ns;s++){
            size_t i=(size_t)gl*ns+s,j=(size_t)ng*ns+s;
            double w=c->level_occupation[i],f=c->within_sl_fraction[j];
            if(!isfinite(occupation[i]) || !isfinite(fraction[i]) ||
               fabs(w-occupation[i])>5.e-12*fmax(fabs(w),fabs(occupation[i])) ||
               fabs(f-fraction[i])>5.e-12*fmax(fabs(f),fabs(fraction[i])))return -1;
            c->level_population[j]=pop[(size_t)b->super_column[sl]*ns+s]*f;
        }
    }
    /* 객체 자체와 모든 mutable 배열이 private다. 공개 commit API를 호출하지
     * 않고 이 trial 소유 파생 상태에만 고유 token을 부여한다. 전하 잔차는
     * off-root 그대로 남기며 legacy ionization/pair solve로 바꾸지 않는다. */
    c->atom.population_committed_generation=id;c->nlte.population_committed_generation=id;
    c->nlte.full_model_population_generation=id;
    b->bf=*bf;
    memcpy(b->bf.material_model_sha256,c->atom.partition_stamp.atomic_model_sha256,65);
    c->nlte.point_bf_rates=&b->bf;c->nlte.radfield_view.generation=bf->radiation_generation;
    c->nlte.radfield_view_status=RADIATION_FIELD_VIEW_OK;
    for(size_t s=0;s<ns;s++)if(nlte_point_bf_view_check(&c->nlte,&c->atom,&c->plasma,(int)s))return -1;
    memset(b->finished,0,(size_t)b->atom.n_elements*ns);
    for(size_t i=0;i<b->n_columns*ns;i++)b->response[i]=NAN;
    b->failed=0;b->closed=0;return 0;
}

static int element_slot(const NLTEModelMaterialBridge *b,int Z,int node)
{
    if(node<0 || node>=b->plasma.n_shells)return -1;
    for(int e=0;e<b->atom.n_elements;e++)if(b->atom.element_Z[e]==Z)return e*b->plasma.n_shells+node;
    return -1;
}

void nlte_model_material_bridge_element_free(NLTEModelMaterialElement *e)
{
    if(!e)return;
    if(!e->finished)e->owner->failed=1;
    nlte_model_collision_bb_stream_free(e->stream);
    nlte_model_dissolution_free(&e->dissolution);nlte_model_bb_block_free(&e->bb);
    nlte_model_bf_block_free(&e->bf);nlte_model_element_free(&e->element);
    e->owner->active_elements--;free(e);
}

NLTEModelMaterialElement *nlte_model_material_bridge_element_begin(
    NLTEModelMaterialBridge *b,uint64_t id,int Z,int node,const NLTEModelBBView *view)
{
    if(!b || b->failed || b->closed || id!=b->trial_id || !id || !view)return NULL;
    int slot=element_slot(b,Z,node);
    if(slot<0 || b->finished[slot])return NULL;
    NLTEModelMaterialElement *e=calloc(1,sizeof(*e));
    if(!e)return NULL;
    e->owner=b;e->trial_id=id;b->active_elements++;b->finished[slot]=1;
    NLTEPopulationCandidate *c=&b->trial;
    NLTEModelBBView bb=*view;
    memcpy(bb.material_model_sha256,c->atom.partition_stamp.atomic_model_sha256,65);
    if(nlte_model_element_init(&e->element,&c->nlte,&c->atom,&c->plasma,Z,node) ||
       nlte_model_element_bf_assemble(&e->bf,&e->element,&c->nlte,&c->atom,&c->plasma) ||
       nlte_model_element_bb_assemble(&e->bb,&e->element,&c->nlte,&c->atom,&c->plasma,&bb) ||
       nlte_model_dissolution_init(&e->dissolution,&e->element,&c->nlte,&c->atom,&c->plasma))goto fail;
    size_t pairs=0;
    for(int ip=0;ip<c->atom.n_ion_pops;ip++)if(c->atom.ion_pop_Z[ip]==Z){
        size_t count=(size_t)(c->atom.level_offset[ip+1]-c->atom.level_offset[ip]);
        if(count>1)pairs+=count*(count-1)/2;
    }
    NLTEModelCollisionBBView col={.Z=Z,.n_levels=(size_t)c->atom.n_levels,
        .n_shells=(size_t)c->plasma.n_shells,.native_shell=(size_t)node,.n_pairs=pairs,
        .material_generation=id,.temperature_K=c->trial_te[node],
        .electron_density_cm3=c->electron_density[node],.dissolution=&e->dissolution};
    memcpy(col.material_model_sha256,c->atom.partition_stamp.atomic_model_sha256,65);
    if(nlte_model_collision_bb_stream_begin(&e->stream,&e->element,&c->nlte,&c->atom,&c->plasma,&col,id))goto fail;
    return e;
fail:
    nlte_model_material_bridge_element_free(e);return NULL;
}

int nlte_model_material_bridge_element_push(NLTEModelMaterialElement *e,uint64_t id,
    const int32_t *lo,const int32_t *up,const double *omega,size_t count)
{
    if(!e || e->finished || e->owner->failed)return -1;
    if(id!=e->trial_id){e->owner->failed=1;return -1;}
    if(nlte_model_collision_bb_stream_push(e->stream,id,lo,up,omega,count)){
        e->owner->failed=1;return -1;
    }
    return 0;
}

int nlte_model_material_bridge_element_finish(NLTEModelMaterialElement *e,uint64_t id,
    double *output,size_t count,double ledger[7])
{
    if(!e || e->finished || !output || !ledger || e->owner->failed)return -1;
    if(id!=e->trial_id){e->owner->failed=1;return -1;}
    NLTEModelMaterialBridge *b=e->owner;size_t ns=(size_t)b->plasma.n_shells,nv=e->element.n_variables;
    if(count!=b->n_columns*ns || output==b->trial.level_population || output==b->nlte.nlte_level_populations)return -1;
    NLTEModelCollisionBBBlock col={0};NLTEModelCombinedGenerator combined={0};
    double *solved=malloc(nv*sizeof(double));int failed=1;
    PopulationGeneratorSolveDiagnostic diag={0};long double terms[7]={0.};
    if(!solved || nlte_model_collision_bb_stream_finish(e->stream,id,&col) ||
       nlte_model_element_generator_combine(&combined,&e->element,&e->bf,&e->bb,&col) ||
       nlte_model_element_fixed_field_candidate(&combined,&e->element,solved,&diag)!=POP_OK)goto done;
    for(size_t row=0;row<nv;row++){
        double p=solved[row];
        terms[0]+=(long double)p*e->bf.radiative_photon_heat[row];
        terms[1]+=(long double)p*e->bb.radiative_heat[row];
        terms[2]+=(long double)p*e->bf.radiative_heat[row];
        terms[3]+=(long double)p*e->bf.collisional_heat[row];
        terms[4]+=(long double)p*col.collisional_heat[row];
        long double gross=0.;
        for(size_t j=0;j<nv;j++)gross+=fabsl((long double)combined.generator[j*nv+row]*e->element.initial_population[j]);
        if(gross>0.)terms[5]=fmaxl(terms[5],fabsl(combined.fixed_residual[row])/gross);
    }
    terms[6]=diag.exact_generator_componentwise_residual;
    for(int k=0;k<7;k++)if(!isfinite((double)terms[k]))goto done;
    for(size_t c=0;c<b->n_columns;c++)if(b->columns[3*c]==e->element.Z){
        int row=b->column_super[c]>=0 ? e->element.super_to_row[b->column_super[c]] :
            e->element.terminal_to_row[b->column_ion[c]];
        if(row<0 || (size_t)row>=nv)goto done;
    }
    for(size_t c=0;c<b->n_columns;c++)if(b->columns[3*c]==e->element.Z){
        int row=b->column_super[c]>=0 ? e->element.super_to_row[b->column_super[c]] :
            e->element.terminal_to_row[b->column_ion[c]];
        output[c*ns+e->element.shell]=solved[row];
        b->response[c*ns+e->element.shell]=solved[row];
    }
    for(int k=0;k<7;k++)ledger[k]=(double)terms[k];
    b->finished[element_slot(b,e->element.Z,e->element.shell)]=2;e->finished=1;failed=0;
done:
    if(failed)b->failed=1;
    free(solved);nlte_model_combined_generator_free(&combined);nlte_model_collision_bb_block_free(&col);
    return failed ? -1 : 0;
}

int nlte_model_material_bridge_push_points(NLTEModelMaterialElement *const *elements,
    uint64_t id,const int32_t *lo,const int32_t *up,const double *omega,size_t np,size_t pairs)
{
    if(!elements || !id || !lo || !up || !omega || !np || np>90 || !pairs || pairs>65536)return -1;
    NLTEModelMaterialBridge *b=elements[0] ? elements[0]->owner : NULL;
    if(!b || b->failed || id!=b->trial_id)return -1;
    for(size_t p=0;p<np;p++){
        if(!elements[p] || elements[p]->owner!=b || elements[p]->trial_id!=id || elements[p]->finished)return -1;
        for(size_t q=0;q<p;q++)if(elements[p]==elements[q])return -1;
    }
    int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for(size_t p=0;p<np;p++)
        if(nlte_model_collision_bb_stream_push(elements[p]->stream,id,lo,up,omega+p*pairs,pairs))failed=1;
    if(failed)b->failed=1;
    return failed ? -1 : 0;
}

int nlte_model_material_bridge_finalize(NLTEModelMaterialBridge *b,uint64_t id,
    const double *pop,size_t count,double *energy,double *charge)
{
    if(!b || b->failed || b->closed || b->active_elements || id!=b->trial_id ||
       !pop || !energy || !charge || count!=b->n_columns*(size_t)b->plasma.n_shells)return -1;
    size_t ns=(size_t)b->plasma.n_shells;
    for(size_t i=0;i<(size_t)b->atom.n_elements*ns;i++)if(b->finished[i]!=2)return -1;
    if(memcmp(pop,b->response,count*sizeof(double)))return -1;
    for(size_t i=0;i<count;i++)if(!isfinite(pop[i]) || pop[i]<0.)return -1;
    NLTEPopulationCandidate *c=&b->trial;
    memset(c->ion_population,0,c->n_ion_values*sizeof(double));
    for(size_t k=0;k<b->n_columns;k++)for(size_t s=0;s<ns;s++)
        c->ion_population[(size_t)b->column_ion[k]*ns+s]+=pop[k*ns+s];
    /* 양이온 총수는 수학적으로 보존돼도 stage 합산의 최하위 bit는 달라질 수
     * 있다. exact stale 검사를 완화하지 않고 새 인구의 파생 상태를 재생산한다.
     * 이후 phase는 닫으므로 이전 BF/BB stamp로 다시 율을 소비하지 않는다. */
    if(nlte_population_candidate_prepare_thermodynamics(c)!=NLTE_CANDIDATE_OK){b->failed=1;return -1;}
    for(int gl=0;gl<c->atom.n_levels;gl++){
        int ng=c->nlte.global_to_nlte_level[gl],sl=c->nlte.fl_to_super[ng];
        for(size_t s=0;s<ns;s++)c->level_population[(size_t)ng*ns+s]=
            pop[(size_t)b->super_column[sl]*ns+s]*c->within_sl_fraction[(size_t)ng*ns+s];
    }
    AtomicInternalEnergyCell *cells=calloc(ns,sizeof(*cells));
    double *charges=calloc(ns,sizeof(double));int failed=1;
    if(!cells || !charges || nlte_population_candidate_occupation_check(c)!=POP_OK ||
       atomic_internal_energy_build(&c->atom,&c->nlte,&c->plasma,ns,id,id,cells)!=ATOMIC_INTERNAL_ENERGY_OK)goto done;
    for(size_t s=0;s<ns;s++){
        long double q=0.;
        for(int ip=0;ip<c->atom.n_ion_pops;ip++)q+=(long double)c->atom.ion_pop_stage[ip]*c->ion_population[(size_t)ip*ns+s];
        charges[s]=(double)(q-c->electron_density[s]);
        if(!isfinite(charges[s]))goto done;
    }
    for(size_t s=0;s<ns;s++){
        energy[3*s]=cells[s].n_atom_cm3;energy[3*s+1]=cells[s].energy_density_erg_cm3;
        energy[3*s+2]=cells[s].internal_energy_atom_erg;charge[s]=charges[s];
    }
    b->closed=1;failed=0;
done:
    free(cells);free(charges);if(failed)b->failed=1;
    return failed ? -1 : 0;
}

int nlte_model_material_bridge_finish_points(NLTEModelMaterialElement *const *elements,
    uint64_t id,size_t np,double *output,size_t count,double *ledger)
{
    if(!elements || !np || np>90 || !output || !ledger || !elements[0])return -1;
    NLTEModelMaterialBridge *b=elements[0]->owner;
    if(b->failed || b->closed || id!=b->trial_id || count!=b->n_columns*(size_t)b->plasma.n_shells)return -1;
    for(size_t p=0;p<np;p++){
        if(!elements[p] || elements[p]->owner!=b || elements[p]->trial_id!=id || elements[p]->finished)return -1;
        for(size_t q=0;q<p;q++)if(elements[p]==elements[q])return -1;
    }
    double *work=malloc(count*sizeof(double)),*terms=malloc(np*7*sizeof(double));
    if(!work || !terms){free(work);free(terms);return -1;}
    memcpy(work,output,count*sizeof(double));int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for(size_t p=0;p<np;p++)
        if(nlte_model_material_bridge_element_finish(elements[p],id,work,count,terms+7*p))failed=1;
    if(!failed){memcpy(output,work,count*sizeof(double));memcpy(ledger,terms,np*7*sizeof(double));}
    else b->failed=1;
    free(work);free(terms);return failed ? -1 : 0;
}
