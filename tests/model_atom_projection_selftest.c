/* 실제 원자모형 등록에서 상위 이온 누락과 terminal 혼입을 확인한다. */
#include "nlte_population_candidate.h"
#include "nlte_model_se.h"
#include "nlte_model_bb.h"
#include "nlte_model_coll_bb.h"
#include "nlte_model_sum.h"

static int check(int yes, const char *message) {
    if (!yes) fprintf(stderr, "[MODEL-PROJECTION][FAIL] %s\n", message);
    return yes ? 0 : 1;
}

typedef struct {
    int full_frequency, native_node;
    NLTEModelElement element[2];
    NLTEModelBFBlock bf[2];
    NLTEModelBBBlock bb[2];
    NLTEModelCollisionBBBlock collision[2];
    NLTEModelDissolutionContext dissolution[2];
} NativeFixedFieldBlocks;

static int fixed_field_slot(int Z) {return Z==14 ? 0 : (Z==16 ? 1 : -1);}

/* 원 물질의 항별 잔차: 정상해 후보의 작은 잔차와 혼동하지 않는다. */
static void print_input_residual(int Z,int stage,int sl,size_t row,
    const NLTEModelElement *e,const NLTEModelBFBlock *bf,
    const NLTEModelBBBlock *bb,const NLTEModelCollisionBBBlock *col,
    const NLTEModelCombinedGenerator *combined)
{
    const double *matrix[4]={bf->radiative,bf->collisional,bb->radiative,col->collisional};
    long double term[4]={0.},absolute_flow=0.;
    for(int k=0;k<4;k++)for(size_t j=0;j<e->n_variables;j++){
        long double flow=(long double)matrix[k][j*e->n_variables+row]*e->initial_population[j];
        term[k]+=flow;absolute_flow+=fabsl(flow);
    }
    printf("[MODEL-SE][INPUT-RESIDUAL-ROW] Z=%d stage0=%d super0=%d bf_rad=%.17g bf_col=%.17g bb_rad=%.17g bb_col=%.17g total=%.17g absolute_flow=%.17g\n",
        Z,stage,sl,(double)term[0],(double)term[1],(double)term[2],(double)term[3],
        combined->fixed_residual[row],(double)absolute_flow);
    if(col->dissolution){
        long double value=0.,flow=0.;
        for(size_t j=0;j<e->n_variables;j++){
            long double current=(long double)col->dissolution[j*e->n_variables+row]*e->initial_population[j];
            value+=current;flow+=fabsl(current);
        }
        printf("[MODEL-SE][DISSOLUTION-ROW] Z=%d stage0=%d super0=%d total=%.17g absolute_flow=%.17g\n",
               Z,stage,sl,(double)value,(double)flow);
    }
}

static void native_fixed_field_free(NativeFixedFieldBlocks *b)
{
    if(!b)return;
    for(int i=0;i<2;i++){
        nlte_model_element_free(&b->element[i]);
        nlte_model_bf_block_free(&b->bf[i]);
        nlte_model_bb_block_free(&b->bb[i]);
        nlte_model_collision_bb_block_free(&b->collision[i]);
        nlte_model_dissolution_free(&b->dissolution[i]);
    }
    memset(b,0,sizeof(*b));
}

/* 선택 또는 완성 ν 적분의 전 경로 handoff. SE/RE 해가 아니다. */
static int import_point_contributions(const char *path,AtomicData *a,NLTEConfig *n,PlasmaState *p,
                                      NativeFixedFieldBlocks *cache)
{
    FILE *fp=fopen(path,"rb");char tag[8];uint32_t shape[4];double snapshot[2];
    if(!fp || fread(tag,1,8,fp)!=8 ||
       (memcmp(tag,"LMBFPTS1",8) && memcmp(tag,"LMBFFUL1",8) && memcmp(tag,"LMBF90V1",8)) ||
       fread(shape,4,4,fp)!=4 ||
       fread(snapshot,8,2,fp)!=2 || shape[0]!=(uint32_t)a->n_levels ||
       shape[2]!=(uint32_t)p->n_shells || !shape[1])return 1;
    int all_nodes=!memcmp(tag,"LMBF90V1",8);
    if(all_nodes ? (shape[2]!=90 || shape[3]!=UINT32_MAX) : shape[3]>=shape[2])return 1;
    int full_frequency=!memcmp(tag,"LMBFFUL1",8);
    if(full_frequency && (!cache || cache->full_frequency))return 1;
    size_t nr=shape[1],ns=shape[2],node=all_nodes ? 0 : shape[3];
    if(nr>SIZE_MAX/ns || nr*ns>(512*1024*1024)/sizeof(BfPointRates) ||
       (!all_nodes && (snapshot[0]!=p->T_e[node] || snapshot[1]!=p->n_electron[node])))return 1;
    int *offset=calloc((size_t)a->n_levels+1,sizeof(int));
    int *target=malloc(nr*sizeof(int)),*terminal=malloc(nr*sizeof(int)),*grid=malloc(nr*sizeof(int));
    double *prob=malloc(nr*sizeof(double)),*edge=malloc(nr*sizeof(double)),*sigma=malloc(nr*sizeof(double));
    double *saha=calloc(nr*ns,sizeof(double)),*te=calloc(ns,sizeof(double)),*ne=calloc(ns,sizeof(double));
    BfPointRates *rates=calloc(nr*ns,sizeof(*rates));uint8_t *ready=calloc(ns,1);
    if(!offset || !target || !terminal || !grid || !prob || !edge || !sigma ||
       !saha || !te || !ne || !rates || !ready)return 1;
    if(all_nodes){
        if(fread(te,8,ns,fp)!=ns || fread(ne,8,ns,fp)!=ns)return 1;
        for(size_t s=0;s<ns;s++){
            if(te[s]!=p->T_e[s] || ne[s]!=p->n_electron[s])return 1;
            ready[s]=1;
        }
    }
    int previous=-1;
    for(size_t r=0;r<nr;r++){
        int32_t ids[5];double values[10];
        if(fread(ids,4,5,fp)!=5 || fread(values,8,all_nodes ? 3 : 10,fp)!=(all_nodes ? 3 : 10) || ids[0]<previous ||
           ids[0]<0 || ids[0]>=a->n_levels)return 1;
        previous=ids[0];offset[ids[0]+1]++;
        target[r]=ids[1];terminal[r]=-1;grid[r]=ids[4];
        if(target[r]<0){
            for(int ip=0;ip<a->n_ion_pops;ip++)if(a->ion_pop_Z[ip]==ids[2] && a->ion_pop_stage[ip]==ids[3]){
                if(terminal[r]>=0)return 1;terminal[r]=ip;
            }
            if(terminal[r]<0)return 1;
        }
        prob[r]=values[0];edge[r]=values[1];sigma[r]=values[2];
        if(all_nodes){
            for(size_t s=0;s<ns;s++){
                double v[7];if(fread(v,8,7,fp)!=7)return 1;
                saha[r*ns+s]=v[0];
                rates[r*ns+s]=(BfPointRates){v[1],v[2],v[3],v[4],v[5],v[6]};
            }
        }else{
            saha[r*ns+node]=values[3];
            rates[r*ns+node]=(BfPointRates){values[4],values[5],values[6],values[7],values[8],values[9]};
        }
    }
    if(fgetc(fp)!=EOF)return 1;fclose(fp);
    for(int gl=0;gl<a->n_levels;gl++)offset[gl+1]+=offset[gl];
    if(!all_nodes){te[node]=snapshot[0];ne[node]=snapshot[1];ready[node]=1;}
    BfPointRateView view={.n_levels=(size_t)a->n_levels,.n_routes=nr,.n_shells=ns,
        .radiation_generation=1,.material_generation=a->population_committed_generation,
        .offset=offset,.target=target,.terminal_ion=terminal,.phot_grid=grid,
        .probability=prob,.threshold_Hz=edge,.sigma_edge=sigma,.saha_lower_per_upper=saha,
        .temperature=te,.electron_density=ne,.shell_ready=ready,.rates=rates};
    memcpy(view.material_model_sha256,a->partition_stamp.atomic_model_sha256,65);
    const BfPointRateView *old=n->point_bf_rates;RadiationFieldView saved=n->radfield_view;
    int saved_status=n->radfield_view_status;
    n->point_bf_rates=&view;n->radfield_view.generation=1;n->radfield_view_status=RADIATION_FIELD_VIEW_OK;
    size_t used=0,terminal_used=0;
    size_t first_node=node,last_node=all_nodes ? ns : node+1;
    for(node=first_node;node<last_node;node++)for(int e=0;e<a->n_elements;e++){
        NLTEModelElement layout={0};NLTEModelBFBlock block={0};
        int Z=a->element_Z[e],slot=fixed_field_slot(Z);
        if(nlte_model_element_init(&layout,n,a,p,Z,(int)node) ||
           nlte_model_element_bf_assemble(&block,&layout,n,a,p))return 1;
        long double collision_heat=0.,electron_heat=0.,photon_heat=0.;
        for(size_t row=0;row<layout.n_variables;row++){
            collision_heat+=(long double)layout.initial_population[row]*block.collisional_heat[row];
            electron_heat+=(long double)layout.initial_population[row]*block.radiative_heat[row];
            photon_heat+=(long double)layout.initial_population[row]*block.radiative_photon_heat[row];
        }
        printf("[MODEL-SE][BF-COLLISION-HEAT] Z=%d native_node=%zu cooling_erg_cm3_s=%.17g\n",
               Z,node,(double)collision_heat);
        printf("[MODEL-SE][BF-ENERGY] Z=%d native_node=%zu electron_erg_cm3_s=%.17g photon_erg_cm3_s=%.17g\n",
               Z,node,(double)electron_heat,(double)photon_heat);
        used+=block.routes;terminal_used+=block.terminal_routes;
        if(full_frequency && slot>=0){
            const char *dissolution=getenv("LUMINA_MODEL_COLLISION_DISSOLUTION");
            if(dissolution && atoi(dissolution)!=0 &&
               nlte_model_dissolution_init(&cache->dissolution[slot],&layout,n,a,p))return 1;
            cache->element[slot]=layout;cache->bf[slot]=block;
            memset(&layout,0,sizeof(layout));memset(&block,0,sizeof(block));
        }
        nlte_model_bf_block_free(&block);nlte_model_element_free(&layout);
    }
    node=first_node;
    if((!all_nodes && nlte_point_bf_view_check(n,a,p,node ? 0 : 1)==0) ||
       used!=nr*(last_node-first_node))return 1;
    n->point_bf_rates=old;n->radfield_view=saved;n->radfield_view_status=saved_status;
    if(full_frequency){cache->full_frequency=1;cache->native_node=(int)node;}
    printf("[MODEL-SE][ACTUAL-BF-CONTRIBUTIONS] native_node=%zu routes=%zu terminal_routes=%zu uncomputed_nodes=REJECTED full_frequency=%d SE_RE_solved=0\n",node,used,terminal_used,full_frequency || all_nodes);
    free(offset);free(target);free(terminal);free(grid);free(prob);free(edge);free(sigma);
    free(saha);free(te);free(ne);free(rates);free(ready);return 0;
}

/* 전체 native 복사 선의 한 원래점 handoff. 충돌/RE/완성 SE를 가장하지 않는다. */
static int import_point_lines(const char *path,AtomicData *a,NLTEConfig *n,PlasmaState *p,
                              NativeFixedFieldBlocks *cache)
{
    FILE *fp=fopen(path,"rb");char tag[8];uint32_t shape[4];
    if(!fp || fread(tag,1,8,fp)!=8 || memcmp(tag,"LMBBPTS1",8) ||
       fread(shape,4,4,fp)!=4 || shape[0]!=(uint32_t)a->n_levels ||
       shape[2]!=(uint32_t)p->n_shells || shape[3]>=shape[2] ||
       !shape[1] || shape[1]>2000000)return 1;
    size_t count=shape[1],node=shape[3];
    if(cache && cache->full_frequency && cache->native_node!=(int)node)return 1;
    int32_t *lo=malloc(count*sizeof(*lo)),*up=malloc(count*sizeof(*up));
    double *nu=malloc(count*sizeof(*nu)),*abs=malloc(count*sizeof(*abs));
    double *stim=malloc(count*sizeof(*stim)),*spont=malloc(count*sizeof(*spont));
    uint8_t *status=malloc(count);
    if(!lo || !up || !nu || !abs || !stim || !spont || !status)return 1;
    for(size_t i=0;i<count;i++){
        int32_t ids[2];double values[4];
        if(fread(ids,4,2,fp)!=2 || fread(values,8,4,fp)!=4)return 1;
        lo[i]=ids[0];up[i]=ids[1];
        nu[i]=values[0];abs[i]=values[1];stim[i]=values[2];spont[i]=values[3];status[i]=1;
    }
    if(fgetc(fp)!=EOF)return 1;
    fclose(fp);
    NLTEModelBBView view={.n_levels=(size_t)a->n_levels,.n_lines=count,
        .n_shells=(size_t)p->n_shells,.native_shell=node,
        .radiation_generation=1,.material_generation=a->population_committed_generation,
        .lower=lo,.upper=up,.frequency_Hz=nu,.absorption=abs,
        .stimulated=stim,.spontaneous=spont,.status=status};
    memcpy(view.material_model_sha256,a->partition_stamp.atomic_model_sha256,65);
    RadiationFieldView saved=n->radfield_view;int saved_status=n->radfield_view_status;
    n->radfield_view.generation=1;n->radfield_view_status=RADIATION_FIELD_VIEW_OK;
    size_t used=0,intra=0;
    for(int i=0;i<a->n_elements;i++){
        NLTEModelElement layout={0};NLTEModelBBBlock block={0};
        int Z=a->element_Z[i];
        if(nlte_model_element_init(&layout,n,a,p,Z,(int)node) ||
           nlte_model_element_bb_assemble(&block,&layout,n,a,p,&view))return 1;
        used+=block.lines;intra+=block.intra_super_lines;
        long double heat=0.;
        for(size_t k=0;k<layout.n_variables;k++){
            heat+=(long double)layout.initial_population[k]*block.radiative_heat[k];
            long double flow=0.;
            for(size_t r=0;r<layout.n_variables;r++)flow+=block.radiative[k*layout.n_variables+r];
            if(fabsl(flow)>1.e-10L*fmaxl(1.,fabsl(block.radiative[k*layout.n_variables+k])))return 1;
        }
        printf("[MODEL-BB][ACTUAL] Z=%d lines=%zu intra=%zu heat_erg_cm3_s=%.17g\n",
               Z,block.lines,block.intra_super_lines,(double)heat);
        int slot=fixed_field_slot(Z);
        if(cache && cache->full_frequency && slot>=0){
            if(cache->bb[slot].radiative)return 1;
            cache->bb[slot]=block;memset(&block,0,sizeof(block));
        }
        nlte_model_bb_block_free(&block);nlte_model_element_free(&layout);
    }
    if(used!=count || !intra)return 1;
    NLTEModelElement layout={0};NLTEModelBBBlock blocked={0};
    if(nlte_model_element_init(&layout,n,a,p,a->element_Z[0],(int)node))return 1;
    view.native_shell=node?0:1;
    if(nlte_model_element_bb_assemble(&blocked,&layout,n,a,p,&view)==0 ||
       blocked.radiative || blocked.n_variables)return 1;
    nlte_model_element_free(&layout);
    n->radfield_view=saved;n->radfield_view_status=saved_status;
    free(lo);free(up);free(nu);free(abs);free(stim);free(spont);free(status);
    printf("[MODEL-BB][HANDOFF] native_node=%zu lines=%zu intra=%zu other_nodes=REJECTED collision=ABSENT SE_RE_solved=0\n",
           node,used,intra);
    return 0;
}

/* 원래점의 Si/S 모든 bound 이온 전 FL 쌍 Ω를 실제 private generator에 연결한다. */
static int import_point_collisions(const char *path,AtomicData *a,NLTEConfig *n,PlasmaState *p,
                                   NativeFixedFieldBlocks *cache)
{
    FILE *fp=fopen(path,"rb");char tag[8];uint32_t shape[4];double snapshot[2];
    if(!fp || fread(tag,1,8,fp)!=8 || memcmp(tag,"LMCBPTS1",8) ||
       fread(shape,4,4,fp)!=4 || fread(snapshot,8,2,fp)!=2 ||
       shape[0]!=(uint32_t)a->n_levels || shape[1]!=(uint32_t)p->n_shells ||
       shape[2]>=shape[1] || shape[3]==0 || shape[3]>400000 ||
       snapshot[0]!=p->T_e[shape[2]] || snapshot[1]!=p->n_electron[shape[2]])return 1;
    size_t count=shape[3],node=shape[2];
    if(cache && cache->full_frequency && cache->native_node!=(int)node)return 1;
    int32_t *lower=malloc(count*sizeof(*lower)),*upper=malloc(count*sizeof(*upper));
    double *omega=malloc(count*sizeof(*omega));
    if(!lower || !upper || !omega)return 1;
    for(size_t i=0;i<count;i++){
        if(fread(&lower[i],4,1,fp)!=1 || fread(&upper[i],4,1,fp)!=1 ||
           fread(&omega[i],8,1,fp)!=1 || lower[i]<0 || upper[i]<=lower[i] ||
           upper[i]>=a->n_levels || a->level_Z[lower[i]]!=a->level_Z[upper[i]] ||
           a->level_ion[lower[i]]!=a->level_ion[upper[i]] ||
           !isfinite(omega[i]) || omega[i]<=0. ||
           (i && (lower[i]<lower[i-1] ||
                  (lower[i]==lower[i-1] && upper[i]<=upper[i-1]))))return 1;
    }
    if(fgetc(fp)!=EOF)return 1;
    fclose(fp);
    size_t cursor=0;
    for(int element=0;element<2;element++){
        int Z=element ? 16:14;
        size_t start=cursor;
        while(cursor<count && a->level_Z[lower[cursor]]==Z)cursor++;
        if(cursor==start)return 1;
        NLTEModelElement layout={0};NLTEModelCollisionBBBlock block={0};
        NLTEModelCollisionBBView view={.Z=Z,.n_levels=(size_t)a->n_levels,
            .n_shells=(size_t)p->n_shells,.native_shell=node,.n_pairs=cursor-start,
            .material_generation=a->population_committed_generation,
            .temperature_K=snapshot[0],.electron_density_cm3=snapshot[1],
            .lower=lower+start,.upper=upper+start,.omega=omega+start};
        if(cache && cache->dissolution[element].n_levels)
            view.dissolution=&cache->dissolution[element];
        memcpy(view.material_model_sha256,a->partition_stamp.atomic_model_sha256,65);
        NLTEModelCollisionBBStream *stream=NULL;
        if(nlte_model_element_init(&layout,n,a,p,Z,(int)node) ||
           nlte_model_collision_bb_stream_begin(&stream,&layout,n,a,p,&view,
               a->population_committed_generation))return 1;
        size_t chunks=0;
        for(size_t at=0;at<view.n_pairs;at+=16384){
            size_t used=view.n_pairs-at;if(used>16384)used=16384;
            if(nlte_model_collision_bb_stream_push(stream,a->population_committed_generation,
                view.lower+at,view.upper+at,view.omega+at,used)){
                nlte_model_collision_bb_stream_free(stream);return 1;
            }
            chunks++;
        }
        if(nlte_model_collision_bb_stream_finish(stream,a->population_committed_generation,&block) ||
           block.pairs!=cursor-start){nlte_model_collision_bb_stream_free(stream);return 1;}
        nlte_model_collision_bb_stream_free(stream);
        printf("[MODEL-COLL-BB][STREAM] Z=%d chunks=%zu pairs=%zu trial_id=%llu\n",
            Z,chunks,block.pairs,(unsigned long long)a->population_committed_generation);
        long double heat=0.;
        for(size_t row=0;row<layout.n_variables;row++)
            heat+=(long double)layout.initial_population[row]*block.collisional_heat[row];
        printf("[MODEL-COLL-BB][ACTUAL] Z=%d pairs=%zu cross=%zu intra=%zu heat_erg_cm3_s=%.17g\n",
               Z,block.pairs,block.cross_super_pairs,block.intra_super_pairs,(double)heat);
        if(block.dissolution){
            long double dissolution_heat=0.;
            for(size_t row=0;row<layout.n_variables;row++)
                dissolution_heat+=(long double)layout.initial_population[row]*block.dissolution_heat[row];
            printf("[MODEL-COLL-BB][DISSOLUTION] Z=%d pairs=%zu intra=%zu heat_erg_cm3_s=%.17g\n",
                   Z,block.dissolution_pairs,block.intra_super_dissolution_pairs,(double)dissolution_heat);
        }
        if(cache && cache->full_frequency){
            if(cache->collision[element].collisional)return 1;
            cache->collision[element]=block;memset(&block,0,sizeof(block));
        }
        nlte_model_collision_bb_block_free(&block);nlte_model_element_free(&layout);
    }
    if(cursor!=count)return 1;
    free(lower);free(upper);free(omega);
    printf("[MODEL-COLL-BB][HANDOFF] native_node=%zu pairs=%zu SE_RE_solved=0\n",node,count);
    return 0;
}

/* 고정된 F(0), T/ne, 점유확률의 누락채널 표시가 붙은 private 원소별 SE 후보.
 * 전하/RE 또는 수송 자기일관 해로 게시하지 않는다. */
static int solve_native_fixed_field(NativeFixedFieldBlocks *cache,
    const AtomicData *a,const NLTEConfig *n)
{
    if(!cache || !cache->full_frequency)return 0;
    for(int slot=0;slot<2;slot++){
        int Z=slot ? 16:14;
        const NLTEModelElement *e=&cache->element[slot];
        NLTEModelCombinedGenerator combined={0};
        double *candidate=NULL;
        int *row_stage=NULL;
        long double *initial_stage=NULL,*candidate_stage=NULL;
        int max_stage=-1,failed=1;
        if(e->Z!=Z || e->shell!=cache->native_node || !cache->bf[slot].radiative ||
           !cache->bb[slot].radiative || !cache->collision[slot].collisional ||
           nlte_model_element_generator_combine(&combined,e,&cache->bf[slot],
               &cache->bb[slot],&cache->collision[slot]))goto done;
        for(int ip=0;ip<a->n_ion_pops;ip++)
            if(a->ion_pop_Z[ip]==Z && a->ion_pop_stage[ip]>max_stage)
                max_stage=a->ion_pop_stage[ip];
        if(max_stage<0 || max_stage>=Z+1)goto done;
        size_t nv=e->n_variables,ns=(size_t)max_stage+1;
        candidate=malloc(nv*sizeof(*candidate));
        row_stage=malloc(nv*sizeof(*row_stage));
        initial_stage=calloc(ns,sizeof(*initial_stage));
        candidate_stage=calloc(ns,sizeof(*candidate_stage));
        if(!candidate || !row_stage || !initial_stage || !candidate_stage)goto done;
        for(size_t row=0;row<nv;row++)row_stage[row]=-1;
        for(int gl=0;gl<a->n_levels;gl++)if(a->level_Z[gl]==Z){
            int ng=n->global_to_nlte_level[gl];
            if(ng<0 || ng>=n->n_nlte_levels_total)goto done;
            int sl=n->fl_to_super[ng];
            if(sl<0 || (size_t)sl>=e->n_super_total)goto done;
            int row=e->super_to_row[sl],stage=a->level_ion[gl];
            if(row<0 || (size_t)row>=nv || stage<0 || (size_t)stage>=ns ||
               (row_stage[row]>=0 && row_stage[row]!=stage))goto done;
            row_stage[row]=stage;
        }
        for(int ip=0;ip<a->n_ion_pops;ip++)if(a->ion_pop_Z[ip]==Z){
            int row=e->terminal_to_row[ip],stage=a->ion_pop_stage[ip];
            if(row<0)continue;
            if((size_t)row>=nv || stage<0 || (size_t)stage>=ns ||
               (row_stage[row]>=0 && row_stage[row]!=stage))goto done;
            row_stage[row]=stage;
        }
        PopulationGeneratorSolveDiagnostic diagnostic={0};
        PopulationStatus status=nlte_model_element_fixed_field_candidate(
            &combined,e,candidate,&diagnostic);
        if(status!=POP_OK){
            fprintf(stderr,"[MODEL-SE][FIXED-FIELD-FAIL] Z=%d native_node=%d status=%s recognized=%d residual=%.17g\n",
                Z,cache->native_node,population_status_name(status),
                diagnostic.generator_recognized,
                diagnostic.exact_generator_componentwise_residual);
            goto done;
        }
        for(size_t row=0;row<nv;row++){
            if(row_stage[row]<0)goto done;
            initial_stage[row_stage[row]]+=e->initial_population[row];
            candidate_stage[row_stage[row]]+=candidate[row];
        }
        long double old_total=0.,new_total=0.;
        for(size_t stage=0;stage<ns;stage++){
            old_total+=initial_stage[stage];new_total+=candidate_stage[stage];
        }
        if(fabsl(old_total-e->total_nuclei)>1.e-12L*e->total_nuclei ||
           fabsl(new_total-e->total_nuclei)>1.e-12L*e->total_nuclei)goto done;
        printf("[MODEL-SE][FIXED-FIELD] Z=%d native_node=%d variables=%zu state=PRIVATE_PARTIAL_CHANNEL_CANDIDATE residual=%.17g physical_equivalence=NOT_PROVEN\n",
            Z,cache->native_node,nv,
            diagnostic.exact_generator_componentwise_residual);
        for(size_t stage=0;stage<ns;stage++)
            printf("[MODEL-SE][FIXED-FIELD-STAGE] Z=%d stage0=%zu original=%.17g candidate=%.17g\n",
                Z,stage,(double)initial_stage[stage],(double)candidate_stage[stage]);
        size_t emitted=0;
        for(size_t sl=0;sl<e->n_super_total;sl++)if(e->super_to_row[sl]>=0){
            int row=e->super_to_row[sl];
            int anchor=n->super_anchor_global[sl];
            if(anchor<0 || anchor>=a->n_levels || !a->level_super ||
               a->level_Z[anchor]!=Z || a->level_ion[anchor]!=row_stage[row] ||
               a->level_super[anchor]<0)goto done;
            printf("[MODEL-SE][FIXED-FIELD-ROW] Z=%d stage0=%d super0=%d original=%.17g candidate=%.17g\n",
                Z,row_stage[row],a->level_super[anchor],
                e->initial_population[row],candidate[row]);
            print_input_residual(Z,row_stage[row],a->level_super[anchor],(size_t)row,e,
                &cache->bf[slot],&cache->bb[slot],&cache->collision[slot],&combined);
            emitted++;
        }
        for(int ip=0;ip<a->n_ion_pops;ip++)if(e->terminal_to_row[ip]>=0){
            int row=e->terminal_to_row[ip];
            if(a->ion_pop_Z[ip]!=Z || a->ion_pop_stage[ip]!=row_stage[row])goto done;
            printf("[MODEL-SE][FIXED-FIELD-ROW] Z=%d stage0=%d super0=-1 original=%.17g candidate=%.17g\n",
                Z,row_stage[row],e->initial_population[row],candidate[row]);
            print_input_residual(Z,row_stage[row],-1,(size_t)row,e,
                &cache->bf[slot],&cache->bb[slot],&cache->collision[slot],&combined);
            emitted++;
        }
        if(emitted!=nv)goto done;
        failed=0;
done:
        free(candidate);free(row_stage);free(initial_stage);free(candidate_stage);
        nlte_model_combined_generator_free(&combined);
        if(failed)return 1;
    }
    return 0;
}

static int import_native_nodes(const char *path, AtomicData *atom,
                               NLTEConfig *nlte, int ns, const char *export_path,
                               const double *occupation,
                               const PopulationOccupationModel *occupation_model,
                               const char *point_contributions,
                               const char *bb_contributions,
                               const char *collision_contributions) {
    FILE *fp=fopen(path,"rb");if (!fp) return 1;
    char magic[8];uint32_t nd,nc,nl;double epoch;
    if (fread(magic,1,8,fp)!=8 || (memcmp(magic,"LUMNOD01",8) && memcmp(magic,"LUMCEL01",8)) ||
        fread(&nd,4,1,fp)!=1 || fread(&nc,4,1,fp)!=1 || fread(&nl,4,1,fp)!=1 ||
        fread(&epoch,8,1,fp)!=1 || nd!=(uint32_t)ns || nl!=(uint32_t)atom->n_levels ||
        nc==0 || nc>10000 || epoch<=0) return 1;
    int32_t *cols=malloc((size_t)nc*3*sizeof(*cols));
    int stride=memcmp(magic,"LUMCEL01",8)==0?6:4;
    int t_at=stride==6?2:1,ne_at=t_at+1,rho_at=t_at+2;
    double *physical=malloc((size_t)ns*(size_t)stride*sizeof(double));
    double *values=malloc((size_t)ns*nc*sizeof(double));
    double *te=malloc((size_t)ns*sizeof(double)),*ne=malloc((size_t)ns*sizeof(double));
    double *rho=malloc((size_t)ns*sizeof(double));
    if (!cols || !physical || !values || !te || !ne || !rho ||
        fread(cols,4,(size_t)nc*3,fp)!=(size_t)nc*3 ||
        fread(physical,8,(size_t)ns*(size_t)stride,fp)!=(size_t)ns*(size_t)stride ||
        fread(values,8,(size_t)ns*nc,fp)!=(size_t)ns*nc || fgetc(fp)!=EOF) return 1;
    fclose(fp);
    for (int s=0;s<ns;s++) { te[s]=physical[stride*s+t_at];ne[s]=physical[stride*s+ne_at];rho[s]=physical[stride*s+rho_at]; }
    PlasmaState p={0};p.n_shells=ns;p.T_e=te;p.n_electron=ne;p.rho=rho;
    if (lumina_publish_seed_te(&p,"native material restart direct test")) return 1;
    int32_t saved_z=cols[0];cols[0]=999;
    if (lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation)!=POP_ATOMIC_MISSING ||
        atom->population_committed_generation || nlte->population_committed_generation) return 1;
    cols[0]=saved_z;
    double *fixed=atom->element_density.number_density;double saved_n=fixed[0];
    fixed[0]*=1.01;
    if (lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation)!=POP_SOLVE_FAILED ||
        atom->population_committed_generation || nlte->population_committed_generation) return 1;
    fixed[0]=saved_n;
    double saved_ne=ne[0];ne[0]*=1.001;
    if (lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation)!=POP_NE_NOT_CONVERGED ||
        atom->population_committed_generation || nlte->population_committed_generation) return 1;
    ne[0]=saved_ne;
    PopulationStatus status=lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation);
    if (check(status==POP_OK,population_status_name(status))) return 1;
    if (check(nlte->full_model_population_generation==1,"full material ownership")) return 1;
    double max_total=0,max_charge=0,max_full_level_total=0;
    for (int i=0;i<nlte->n_nlte_ions;i++) {
        int ip=-1;
        for (int j=0;j<atom->n_ion_pops;j++)
            if (atom->ion_pop_Z[j]==nlte->nlte_Z[i] &&
                atom->ion_pop_stage[j]==nlte->nlte_ion[i]) ip=j;
        if (ip<0) return 1;
        for (int s=0;s<ns;s++) {
            long double full=0;
            for (int g=nlte->nlte_ion_level_offset[i];g<nlte->nlte_ion_level_offset[i+1];g++)
                full+=nlte->nlte_level_populations[(size_t)g*ns+s];
            double reference=atom->ion_number_density[(size_t)ip*ns+s];
            double rel=reference>0 ? (double)fabsl(full/reference-1) : (double)fabsl(full);
            if (rel>max_full_level_total) max_full_level_total=rel;
        }
    }
    if (check(max_full_level_total<=1e-12,"full-level reconstruction changed ion totals")) return 1;
    for (int s=0;s<ns;s++) {
        double charge=0;
        for (int e=0;e<atom->n_elements;e++) {
            double sum=0;
            for (int ip=atom->elem_ion_offset[e];ip<atom->elem_ion_offset[e+1];ip++) {
                double n=atom->ion_number_density[(size_t)ip*ns+s];
                sum+=n;charge+=n*atom->ion_pop_stage[ip];
            }
            double expected=fixed[(size_t)e*ns+s];
            double error=fabs(sum/expected-1);if (error>max_total) max_total=error;
        }
        double error=fabs(charge/ne[s]-1);if (error>max_charge) max_charge=error;
        if (te[s]!=physical[stride*s+t_at] || ne[s]!=physical[stride*s+ne_at] || rho[s]!=physical[stride*s+rho_at]) return 1;
    }
    printf("[MATERIAL-IMPORT][ACTUAL] input=%.8s samples=%d SL_and_terminal=%u total_error=%.17g charge_error=%.17g Te_ne_rho=UNCHANGED negative_controls=3 RE_solved=0\n",
           magic,ns,nc,max_total,max_charge);
    printf("[MATERIAL-IMPORT][FULL-LEVEL] ion_total_relative_error=%.17g\n",max_full_level_total);
    int checked=0;
    for (int e=0;e<atom->n_elements;e++) {
        int chosen=-1;
        for (int l=0;l<atom->n_lines;l++)
            if (atom->line_atomic_number[l]==atom->element_Z[e]) { chosen=l;break; }
        if (chosen<0) return 1;
        int ip=-1;
        for (int i=0;i<atom->n_ion_pops;i++)
            if (atom->ion_pop_Z[i]==atom->line_atomic_number[chosen] &&
                atom->ion_pop_stage[i]==atom->line_ion_number[chosen]) ip=i;
        if (ip<0) return 1;
        int gl=atom->level_offset[ip]+atom->line_level_upper[chosen];
        int ng=nlte->global_to_nlte_level[gl];
        double upper=NAN;
        EmissivityStatus st=lumina_line_upper_population_for_tau(
            atom,&p,nlte,chosen,0,(size_t)ns,&upper);
        if (check((st==EMISS_OK || st==EMISS_EXACT_ZERO) && ng>=0 &&
                  upper==nlte->nlte_level_populations[(size_t)ng*ns],
                  "all-element line consumer did not use imported population")) return 1;
        nlte->full_model_population_generation=0;
        st=lumina_line_upper_population_for_tau(atom,&p,nlte,chosen,0,(size_t)ns,&upper);
        nlte->full_model_population_generation=1;
        if (check(st==EMISS_STALE_POP,"unowned full model fell back to LTE")) return 1;
        checked++;
    }
    printf("[MATERIAL-IMPORT][LINE-OWNER] elements=%d imported_NLTE=EXACT unowned_generation=REJECTED\n",checked);
    size_t bound_rows=0,terminal_rows=0,max_rows=0;
    for(int element=0;element<atom->n_elements;element++){
        NLTEModelElement layout={0};int node=ns>49 ? 49 : 0;
        if(check(nlte_model_element_init(&layout,nlte,atom,&p,
             atom->element_Z[element],node)==0,"actual full-model element layout"))return 1;
        bound_rows+=layout.n_bound;terminal_rows+=layout.n_terminals;
        if(layout.n_variables>max_rows)max_rows=layout.n_variables;
        nlte_model_element_free(&layout);
    }
    if(check(bound_rows==(size_t)nlte->n_super_total && terminal_rows==6,
             "actual full-model SL/terminal coverage"))return 1;
    printf("[MODEL-SE][ACTUAL-LAYOUT] elements=%d SL=%zu terminal=%zu maximum_element_rows=%zu field_or_SE_solved=0\n",
        atom->n_elements,bound_rows,terminal_rows,max_rows);
    NativeFixedFieldBlocks fixed_field={0};
    if(point_contributions && import_point_contributions(point_contributions,atom,nlte,&p,&fixed_field))return 1;
    if (occupation_model) {
        NLTEPopulationCandidate candidate={0};
        size_t count=(size_t)atom->n_levels*ns;
        double *before=malloc(count*sizeof(double));
        if(!before || !atom->model_level_occupation)return 1;
        memcpy(before,atom->model_level_occupation,count*sizeof(double));
        if(check(nlte_population_candidate_begin_model(&candidate,nlte,atom,&p,
            te,(size_t)ns,p.T_e_generation+1,atom->population_committed_generation+1,
            occupation_model)==NLTE_CANDIDATE_OK,"actual model candidate begin"))return 1;
        double max_w=0.;
        for(size_t i=0;i<count;i++){
            double expected=before[i],got=candidate.level_occupation[i];
            double error=expected>0. ? fabs(got/expected-1.) : fabs(got);
            if(error>max_w)max_w=error;
        }
        if(check(max_w<5.e-12,"actual private occupation differs from native fit") ||
           check(candidate.level_occupation!=atom->model_level_occupation,"occupation alias") ||
           check(nlte_population_candidate_occupation_check(&candidate)==POP_OK,"actual fresh occupation"))return 1;
        int node=ns>49 ? 49 : 0;
        candidate.trial_te[node]*=1.07;candidate.electron_density[node]*=1.11;
        if(check(nlte_population_candidate_occupation_check(&candidate)==POP_STALE_DERIVED_TEMPERATURE,
                 "changed T/ne did not stale occupation") ||
           check(nlte_population_candidate_prepare_thermodynamics(&candidate)==NLTE_CANDIDATE_OK,
                 "actual T/ne occupation refresh") ||
           check(nlte_population_candidate_occupation_check(&candidate)==POP_OK,"refreshed occupation not consumable") ||
           check(!memcmp(before,atom->model_level_occupation,count*sizeof(double)),"private w changed public profile"))return 1;
        printf("[MODEL-OCCUPATION][ACTUAL-PRIVATE] values=%zu max_relative=%.17g T_ne_refresh=PASS public_w=UNCHANGED SE_RE_solved=0\n",count,max_w);
        nlte_population_candidate_free(&candidate);free(before);
    }
    if (export_path) {
        FILE *out=fopen(export_path,"wbx");if (!out) return 1;
        uint32_t shape[2]={(uint32_t)atom->n_levels,(uint32_t)ns};
        int failed=fwrite("LMATFL01",1,8,out)!=8 || fwrite(shape,4,2,out)!=2;
        for (int gl=0;gl<atom->n_levels && !failed;gl++) {
            int at=nlte->global_to_nlte_level[gl];
            if (at<0 || fwrite(nlte->nlte_level_populations+(size_t)at*ns,
                    sizeof(double),(size_t)ns,out)!=(size_t)ns) failed=1;
        }
        if (fclose(out)!=0 || failed) return 1;
        printf("[MATERIAL-IMPORT][FULL-EXPORT] original_FL_order=%d samples=%d\n",atom->n_levels,ns);
    }
    if(bb_contributions && import_point_lines(bb_contributions,atom,nlte,&p,&fixed_field))return 1;
    if(collision_contributions && import_point_collisions(collision_contributions,atom,nlte,&p,&fixed_field))return 1;
    if(solve_native_fixed_field(&fixed_field,atom,nlte))return 1;
    native_fixed_field_free(&fixed_field);
    free(cols);free(physical);free(values);free(te);free(ne);free(rho);
    return 0;
}

int main(int argc, char **argv) {
    if (argc>=2 && argc<=10) {
        AtomicData actual = {0}; OpacityState op = {0}; NLTEConfig model = {0};
        int ns=argc>=3 ? atoi(argv[2]):1;
        if (ns<=0 || ns>1000) return 2;
        /* 이 원자 view의 조성은 명시적으로 한 점만 저장한 조건부 입력이다.
         * 물리 쉘 격자 템플릿으로 쓰지 않고 원자 등록만 검사한다. */
        if (check(load_atomic_data(&actual,argv[1],ns)==0,"actual atomic load")) return 1;
        op.n_lines=actual.n_lines;
        if (check(nlte_init_model_atom(&model,&actual,&op,ns)==0,
                  "actual model projection")) return 1;
        if (check(model.n_nlte_levels_total==actual.n_levels,
                  "actual full-level coverage")) return 1;
        for (int l=0;l<actual.n_levels;l++)
            if (check(model.global_to_nlte_level[l]>=0,"actual full-level missing")) return 1;
        for (int l=0;l<actual.n_lines;l++)
            if (check(model.nlte_line_map[l]>=0,"actual line missing")) return 1;
        printf("[MODEL-PROJECTION][ACTUAL] bound_ions=%d full_levels=%d superlevels=%d lines=%d\n",
               model.n_nlte_ions,model.n_nlte_levels_total,model.n_super_total,actual.n_lines);
        double *occupation=NULL;
        if (argc>=6) {
            FILE *input=fopen(argv[5],"rb");char magic[8];uint32_t shape[2];
            if (!input || fread(magic,1,8,input)!=8 || memcmp(magic,"LMATOCC1",8) ||
                fread(shape,4,2,input)!=2 || shape[0]!=(uint32_t)actual.n_levels || shape[1]!=(uint32_t)ns) return 1;
            size_t count=(size_t)shape[0]*shape[1];occupation=malloc(count*sizeof(double));
            if (!occupation || fread(occupation,sizeof(double),count,input)!=count || fgetc(input)!=EOF) return 1;
            fclose(input);
        }
        PopulationOccupationModel occupation_model={0};double *occupation_parameters=NULL;
        if(argc>=7){
            FILE *input=fopen(argv[6],"rb");char tag[8];uint32_t count;
            if(!input || fread(tag,1,8,input)!=8 || memcmp(tag,"LMOCPAR1",8) ||
               fread(&count,4,1,input)!=1 || count!=(uint32_t)actual.n_levels)return 1;
            occupation_parameters=malloc(2*(size_t)count*sizeof(double));
            if(!occupation_parameters || fread(occupation_parameters,sizeof(double),2*(size_t)count,input)!=2*(size_t)count ||
               fgetc(input)!=EOF)return 1;
            fclose(input);
            occupation_model=(PopulationOccupationModel){count,occupation_parameters,occupation_parameters+count};
        }
        if (argc>=4 && import_native_nodes(argv[3],&actual,&model,ns,argc>=5?argv[4]:NULL,occupation,
                                          argc>=7 ? &occupation_model : NULL,
                                          argc>=8 ? argv[7] : NULL,
                                          argc>=9 ? argv[8] : NULL,
                                          argc==10 ? argv[9] : NULL)) return 1;
        free(occupation_parameters);
        free(occupation);
        if (actual.element_density.source==POP_ELEMENT_TOTAL_EXPLICIT_NUMBER) {
            PlasmaState p={0};p.n_shells=ns;
            double *rho=calloc((size_t)ns,sizeof(*rho));p.rho=rho;
            if (!rho) return 1;
            for (int e=0;e<actual.n_elements;e++)
                for (int s=0;s<ns;s++) {
                    size_t at=(size_t)e*ns+s;
                    double expected=actual.element_density.number_density[at];
                    actual.element_mass_amu[e]=NAN;
                    actual.abundances[at]=NAN;rho[s]=NAN;
                    if (check(lumina_element_number_density(&actual,&p,e,s,ns)==expected,
                              "actual inventory changed by mass/X/rho")) return 1;
                }
            free(rho);
            printf("[MODEL-PROJECTION][ACTUAL-NUMBER] elements=%d points=%d; poisoned mass/X/rho ignored\n",
                   actual.n_elements,ns);
        }
        nlte_free(&model); free_atomic_data(&actual);
        return 0;
    }
    enum { NI = 12, NL = 8, NS = 2, NLINE = 4 };
    int ion_z[NI] = {28,28,27,27,26,26,20,20,16,16,14,14};
    int stages[NI] = {5,6,5,6,5,6,4,5,4,5,4,5};
    int offsets[NI+1] = {0,1,1,2,2,3,3,4,4,5,5,8,8};
    int level_z[NL] = {28,27,26,20,16,14,14,14};
    int level_stage[NL] = {5,5,5,4,4,4,4,4};
    int level_num[NL] = {0,0,0,0,0,0,1,2};
    int super[NL] = {0,0,0,0,0,0,0,1};
    int metastable[NL] = {0};
    double energy[NL] = {0,0,0,0,0,0,1,2};
    int line_z[NLINE] = {14,20,26,28};
    int line_stage[NLINE] = {4,4,5,5};
    int line_upper[NLINE] = {1,0,0,0};
    AtomicData atom = {0}; OpacityState opacity = {0}; NLTEConfig nlte = {0};
    atom.n_ion_pops=NI; atom.n_levels=NL; atom.n_lines=NLINE;
    atom.ion_pop_Z=ion_z; atom.ion_pop_stage=stages; atom.level_offset=offsets;
    atom.level_Z=level_z; atom.level_ion=level_stage; atom.level_num=level_num;
    atom.level_super=super; atom.level_metastable=metastable;
    atom.level_energy_eV=energy;
    atom.line_atomic_number=line_z; atom.line_ion_number=line_stage;
    atom.line_level_upper=line_upper; opacity.n_lines=NLINE;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)==0,"initialization")) return 1;
    if (check(nlte.n_nlte_ions==6,"every bound ion, no terminal") ||
        check(nlte.n_nlte_levels_total==NL,"all full levels") ||
        check(nlte.n_super_total==7,"all superlevels") ||
        check(nlte.super_mode,"superlevel reconstruction active")) return 1;
    const int expected_z[6]={14,16,20,26,27,28};
    for (int i=0;i<6;i++)
        if (check(nlte.nlte_Z[i]==expected_z[i],"contiguous sorted projection")) return 1;
    for (int l=0;l<NL;l++)
        if (check(nlte.global_to_nlte_level[l]>=0,"unmapped full level")) return 1;
    for (int l=0;l<NLINE;l++)
        if (check(nlte.nlte_line_map[l]>=0,"unmapped high-ion line")) return 1;
    nlte_free(&nlte);
    offsets[NI]=NL-1;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "inconsistent level ownership accepted")) return 1;
    offsets[NI]=NL; level_stage[0]=4;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "wrong ion identity accepted")) return 1;
    level_stage[0]=5; super[0]=-1;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "negative superlevel accepted")) return 1;
    puts("[MODEL-PROJECTION][PASS] full-level=8 bound-ion=6 terminal=6 excluded; identity failures rejected");
    return 0;
}
