#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cmf_toy_model_free(CMFToyModel *m){
    if(!m)return;
    bf_set_nlte_pops(NULL);
    cmfgen_fine_set_bf_atom(NULL,NULL,NULL,NULL);
    radiation_field_owner_free(&m->nlte.radiation_field);
    nlte_free(&m->nlte);
    line_jbar_qset_free(&m->qset);line_jbar_qset_free(&m->eset);
    bf_opacity_free(&m->bf);
    /* 이3개 배열은 원자/물질 소유자의 차용 포인터다. */
    m->opacity.line_list_nu=m->opacity.electron_density=m->opacity.t_electrons=NULL;
    free(m->opacity.jnu_fine);free(m->opacity.nu_fine);
    free_opacity_state(&m->opacity);
    a210_publication_free(&m->plasma.te_publication);
    free(m->shell_storage);free_atomic_data(&m->atom);cmf_toy_ejecta_free(&m->ejecta);
    memset(m,0,sizeof(*m));
}

int cmf_toy_model_prepare(CMFToyModel *m,const char *path,const char *atomic_dir,
                         int ns,const double mass[CMF_TOY_ELEMENTS]){
    return cmf_toy_model_prepare_refined(m,path,atomic_dir,ns,mass,0);
}
int cmf_toy_model_prepare_runtime(CMFToyModel *m,const char *path,const char *atomic_dir,
                         int ns,const double mass[CMF_TOY_ELEMENTS]){
    const char *setting=getenv("LUMINA_TOY_GRID_T_RATIO");double ratio=0;
    if(setting){
        char *end;errno=0;ratio=strtod(setting,&end);
        if(errno||!*setting||*end||!isfinite(ratio)||ratio<=1)return -1;
    }
    if(cmf_toy_model_prepare_refined(m,path,atomic_dir,ns,mass,ratio))return -1;
    fprintf(stderr,"[TOY-GRID] base_shells=%d actual_shells=%d T_ratio=%.17g policy=%s\n",
            ns,m->ejecta.n_shells,ratio,setting?"SOURCE_T_REFINEMENT":"UNIFORM_BASE");
    return 0;
}
int cmf_toy_model_prepare_refined(CMFToyModel *m,const char *path,const char *atomic_dir,
                         int ns,const double mass[CMF_TOY_ELEMENTS],double temperature_ratio){
    if(!m||m->prepared||m->shell_storage||!path||!atomic_dir||ns<2||ns>256)return -1;
    const char *phase="ejecta";
    if(cmf_toy_ejecta_read_refined(&m->ejecta,path,ns,mass,temperature_ratio,256))goto fail;
    ns=m->ejecta.n_shells;
    phase="atomic structure";
    if(load_atomic_structure(&m->atom,atomic_dir,ns))goto fail;
    if(load_ion_coldata_manifest(&m->atom,atomic_dir)<=0)goto fail;
    /* toy CLI는 기존 main의 원자 초기화보다 먼저 분기한다. 표를 읽기만 하면
     * 전이별 dE/g/실표 맵이 비어 모든 선이 OMEGA_SET으로 잘못 분류된다. */
    omega_cmfgen_arm(&m->atom);
    AtomicData *a=&m->atom;Geometry *g=&m->geometry;PlasmaState *p=&m->plasma;
    OpacityState *o=&m->opacity;NLTEConfig *n=&m->nlte;
    if(!a->bf_channels||a->bf_channels->version!=2||a->n_lines<1||a->n_levels<1||
       a->n_ion_pops<1||!a->abundances||!a->ion_number_density||!a->partition_functions)goto fail;
    phase="initial material";
    m->shell_storage=calloc((size_t)7*ns,sizeof(double));if(!m->shell_storage)goto fail;
    g->n_shells=p->n_shells=o->n_shells=ns;
    g->r_inner=m->shell_storage;g->r_outer=g->r_inner+ns;
    g->v_inner=g->r_outer+ns;g->v_outer=g->v_inner+ns;
    p->rho=g->v_outer+ns;p->T_e=p->rho+ns;p->n_electron=p->T_e+ns;
    CMFEjectaEpoch init={0};init.n_shells=ns;init.epoch_s=m->ejecta.epoch_s;
    init.previous_epoch_s=init.epoch_s;
    /* 초기 시점에는 붕괴를 다시 적용하지 않는다. 방출/침적은 여기서 생성하지 않는다. */
    init.storage=calloc((size_t)7*ns,sizeof(double));if(!init.storage)goto fail;
    init.density=init.storage;init.number_density=init.storage+ns;
    for(int s=0;s<ns;s++){
        double lo=m->ejecta.velocity_edges[s]*init.epoch_s,hi=m->ejecta.velocity_edges[s+1]*init.epoch_s;
        double volume=(4*M_PI_VAL/3)*(hi-lo)*(hi*hi+lo*hi+lo*lo);
        init.density[s]=m->ejecta.mass_g[s]/volume;
        for(int k=0;k<CMF_TOY_ELEMENTS;k++)init.number_density[k*ns+s]=
            init.density[s]*m->ejecta.number_per_g[k*ns+s];
    }
    int applied=cmf_ejecta_apply_epoch(&m->ejecta,&init,g,p,a);
    cmf_ejecta_epoch_free(&init);if(applied)goto fail;
    memcpy(p->T_e,m->ejecta.initial_temperature,(size_t)ns*sizeof(double));
    for(int s=0;s<ns;s++)for(int e=0;e<a->n_elements;e++)p->n_electron[s]+=
        a->element_Z[e]*a->abundances[e*ns+s]*p->rho[s]/(a->element_mass_amu[e]*AMU);
    size_t cells=(size_t)a->n_lines*ns;
    if(cells>SIZE_MAX/sizeof(double))goto fail;
    o->n_lines=a->n_lines;o->line_list_nu=a->line_nu;
    o->electron_density=p->n_electron;o->t_electrons=p->T_e;
    o->tau_sobolev=calloc(cells,sizeof(double));o->line_source_S=calloc(cells,sizeof(double));
    o->tau_validity=calloc(cells,sizeof(*o->tau_validity));
    o->line_source_validity=calloc(cells,sizeof(*o->line_source_validity));
    if(!o->tau_sobolev||!o->line_source_S||!o->tau_validity||!o->line_source_validity)goto fail;
    phase="Saha predictor";
    p->initial_steady_nlte=1;
    if(lumina_publish_seed_te(p,"toy06 hydro T; not converged NLTE"))goto fail;
    if(lumina_bootstrap_window_open("full-ejecta initial material"))goto fail;
    int prepared=lumina_prepare_solver_owned_tau(a,p,o,g->time_explosion,"toy06 initial NLTE");
    lumina_bootstrap_window_close();if(prepared)goto fail;
    phase="NLTE projection and LTE level predictor";
    if(nlte_init(n,a,o,ns)||!n->super_mode)goto fail;
    /* StaNdaRT(2022) §3.2: toy 모델에는 H/He와 그 CE가 없다.
     * 별도의 경험적 Ca-Si/철족 CE 표를 CMFGEN 물리로 추가하지 않는다. */
    n->exclude_empirical_heavy_ce=1;
    /* Hillier(2023) §4–5: 공명은 PHOT 또는 명시적 양방향 상태로 처리한다.
     * 원자료 PHOT의 공명을 유지하며 역과정 없는 총 DR fit을 중첩하지 않는다. */
    n->channel_owned_recombination=1;
    n->population_committed_generation=a->population_committed_generation;
    PopulationAtomicView av={0};av.n_ions=a->n_ion_pops;av.n_levels=a->n_levels;
    av.level_offset=a->level_offset;av.energy_eV=a->level_energy_eV;av.g=a->level_g;
    av.level_Z=a->level_Z;av.level_ion=a->level_ion;
    for(int ip=0;ip<a->n_ion_pops;ip++)for(int gl=a->level_offset[ip];gl<a->level_offset[ip+1];gl++){
        int mapped=n->global_to_nlte_level[gl];if(mapped<0)goto fail;
        for(int s=0;s<ns;s++){
            double f;
            PopulationStatus status=population_lte_level_fraction(&av,ip,gl,p->T_e[s],a->partition_functions[ip*ns+s],&f);
            if(status!=POP_OK&&status!=POP_EXACT_ZERO)goto fail;
            n->nlte_level_populations[(size_t)mapped*ns+s]=a->ion_number_density[ip*ns+s]*f;
        }
    }
    if(nlte_precompute_within_sl_frac(n,a,p,ns))goto fail;
    phase="radiation owners";
    if(radiation_field_owner_init(&n->radiation_field,ns))goto fail;
    uint8_t *mask=malloc((size_t)a->n_lines);if(!mask)goto fail;
    size_t inside=0,outside=0;
    int qrc=line_jbar_bb_domain_mask_build(mask,a->n_lines,a->line_nu,n->nlte_line_map,&inside,&outside);
    if(!qrc)qrc=line_jbar_qset_build(&m->qset,a->n_lines,a->line_nu,n->nlte_line_map,mask);
    free(mask);if(qrc||line_jbar_eset_build(&m->eset,a->n_lines,a->line_nu))goto fail;
    n->line_qset=&m->qset;n->line_eset=&m->eset;
    if(cmfgen_bind_common_profile(n,5e6)||bf_opacity_init_checked(&m->bf,ns))goto fail;
    bf_set_nlte_pops(n);
    nlte_update_tau_sobolev(n,a,o,g->time_explosion,ns);
    if(tau_sobolev_assert_fresh(o,"full-ejecta prepared material"))goto fail;
    m->prepared=1;
    fprintf(stderr,"[TOY-MODEL] prepared shells=%d source_shells=%d epoch=%.17g levels=%d super=%d ions=%d lines=%d Q=%zu E=%zu radiation_published=0 gamma_published=0 initial_NLTE_converged=0\n",
        ns,m->ejecta.n_source_shells,g->time_explosion,a->n_levels,n->n_super_total,a->n_ion_pops,a->n_lines,m->qset.n_q,m->eset.n_q);
    return 0;
fail:
    fprintf(stderr,"[TOY-MODEL] preparation failed phase=%s\n",phase);
    cmf_toy_model_free(m);return -1;
}

int cmf_toy_model_write_material(const CMFToyModel *m,const char *path,const char *status){
    if(!m||!m->prepared||!path||!status||strchr(status,'\n')||strchr(status,'\r'))return -1;
    FILE *fp=fopen(path,"wx");if(!fp)return -1;
    const AtomicData *a=&m->atom;const PlasmaState *p=&m->plasma;int ns=p->n_shells;
    int rc=fprintf(fp,"# %s\n"
        "shell,r_inner_cm,r_outer_cm,rho_g_cm3,T_K,n_e_cm3,Z,stage,n_ion_cm3,mass_fraction\n",status)<0;
    for(int s=0;s<ns;s++)for(int e=0;e<a->n_elements;e++)
      for(int ip=a->elem_ion_offset[e];ip<a->elem_ion_offset[e+1];ip++)
        if(fprintf(fp,"%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g\n",s,
            m->geometry.r_inner[s],m->geometry.r_outer[s],p->rho[s],p->T_e[s],p->n_electron[s],
            a->ion_pop_Z[ip],a->ion_pop_stage[ip],a->ion_number_density[ip*ns+s],a->abundances[e*ns+s])<0)rc=1;
    if(fclose(fp))rc=1;return rc?-1:0;
}

int cmf_toy_model_write_seed(const CMFToyModel *m,const char *path){
    return cmf_toy_model_write_material(m,path,"BOOTSTRAP_LTE_SAHA; not converged NLTE; radiation/gamma unpublished");
}

static int iterate_identity(const char *s){
    if(!s||strlen(s)!=64)return -1;
    for(int i=0;i<64;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return -1;
    return 0;
}

int cmf_toy_initial_iterate_write(const CMFToyModel *m,const char *path,const char *id,int iteration){
    if(!m||!m->prepared||!path||iterate_identity(id)||iteration<0||
       !m->plasma.initial_steady_nlte||m->plasma.physical_history||
       m->opacity.n_fine<2||!m->opacity.jnu_fine)return -1;
    const AtomicData *a=&m->atom;const NLTEConfig *n=&m->nlte;
    const PlasmaState *p=&m->plasma;const OpacityState *o=&m->opacity;
    uint64_t ns=p->n_shells,ni=a->n_ion_pops,nl=n->n_nlte_levels_total,ne=a->n_elements,nb=o->n_fine;
    uint64_t h[]={ns,ni,nl,ne,nb,a->population_committed_generation,(uint64_t)iteration};
    double par[]={m->geometry.time_explosion,o->nu_lo_fine,o->dlognu_fine};
    uint32_t endian=1;
    if(sizeof(int)!=4||sizeof(double)!=8||*(unsigned char*)&endian!=1||!ns||ns>256||
       !ni||!nl||!ne||!h[5]||h[5]==UINT64_MAX||
       (5+ne+ni+nl+nb)*ns>UINT64_C(250000000))return -1;
    FILE *fp=fopen(path,"wbx");if(!fp)return -1;int rc=-1;
#define ITERATE_PUT(x,z,c) do{if(fwrite((x),(z),(c),fp)!=(c))goto iterate_write_done;}while(0)
    ITERATE_PUT("CMFITER1",1,8);ITERATE_PUT(id,1,64);ITERATE_PUT(h,8,7);ITERATE_PUT(par,8,3);
    ITERATE_PUT(n->nlte_to_global_level,4,nl);ITERATE_PUT(a->ion_pop_Z,4,ni);ITERATE_PUT(a->ion_pop_stage,4,ni);
    ITERATE_PUT(m->geometry.r_inner,8,ns);ITERATE_PUT(m->geometry.r_outer,8,ns);
    ITERATE_PUT(p->rho,8,ns);ITERATE_PUT(p->T_e,8,ns);ITERATE_PUT(a->abundances,8,ne*ns);
    ITERATE_PUT(p->n_electron,8,ns);ITERATE_PUT(a->ion_number_density,8,ni*ns);
    ITERATE_PUT(n->nlte_level_populations,8,nl*ns);ITERATE_PUT(o->jnu_fine,8,nb*ns);
    ITERATE_PUT("ITEREND1",1,8);rc=0;
#undef ITERATE_PUT
iterate_write_done:
    if(fclose(fp))rc=-1;
    return rc;
}

int cmf_toy_initial_iterate_read(CMFToyModel *m,const char *path,const char *id,int *iteration){
    if(!m||!m->prepared||!path||!iteration||iterate_identity(id)||
       !m->plasma.initial_steady_nlte||m->plasma.physical_history||m->opacity.jnu_fine||
       m->nlte.radiation_field.field.generation.computed_generation)return -1;
    AtomicData *a=&m->atom;NLTEConfig *n=&m->nlte;PlasmaState *p=&m->plasma;OpacityState *o=&m->opacity;
    uint32_t endian=1;uint64_t h[7],ns=p->n_shells,ni=a->n_ion_pops,nl=n->n_nlte_levels_total,ne=a->n_elements;
    uint64_t values=0,bytes=0,nb=0;double par[3],*data=NULL,*frequency=NULL,*j=NULL,*mutable_data=NULL;
    int *maps=NULL,rc=-1;char marker[8],stored_id[64];long length;
    if(sizeof(int)!=4||sizeof(double)!=8||*(unsigned char*)&endian!=1)return -1;
    FILE *fp=fopen(path,"rb");if(!fp)return -1;
#define ITERATE_GET(x,z,c) do{if(fread((x),(z),(c),fp)!=(c))goto iterate_read_done;}while(0)
    ITERATE_GET(marker,1,8);if(memcmp(marker,"CMFITER1",8))goto iterate_read_done;
    ITERATE_GET(stored_id,1,64);if(memcmp(stored_id,id,64))goto iterate_read_done;
    ITERATE_GET(h,8,7);ITERATE_GET(par,8,3);
    if(h[0]!=ns||h[1]!=ni||h[2]!=nl||h[3]!=ne||!ns||ns>256||
       h[4]<2||h[4]>INT32_MAX||!h[5]||h[5]>=UINT64_MAX-1||h[6]>INT32_MAX||
       par[0]!=m->geometry.time_explosion||!(par[1]>0)||!isfinite(par[1])||
       !(par[2]>0)||!isfinite(par[2]))goto iterate_read_done;
    nb=h[4];values=(5+ne+ni+nl+nb)*ns;
    if(values>UINT64_C(250000000))goto iterate_read_done;
    bytes=160+4*(nl+2*ni)+8*values;
    if(fseek(fp,0,SEEK_END)||(length=ftell(fp))<0||(uint64_t)length!=bytes||
       fseek(fp,152,SEEK_SET))goto iterate_read_done;
    maps=malloc((nl+2*ni)*sizeof(int));data=malloc(values*sizeof(double));
    frequency=malloc(nb*sizeof(double));j=malloc(ns*nb*sizeof(double));
    if(!maps||!data||!frequency||!j)goto iterate_read_done;
    ITERATE_GET(maps,4,nl+2*ni);ITERATE_GET(data,8,values);ITERATE_GET(marker,1,8);
    if(memcmp(marker,"ITEREND1",8)||fgetc(fp)!=EOF||
       memcmp(maps,n->nlte_to_global_level,nl*4)||memcmp(maps+nl,a->ion_pop_Z,ni*4)||
       memcmp(maps+nl+ni,a->ion_pop_stage,ni*4)||
       memcmp(data,m->geometry.r_inner,ns*8)||memcmp(data+ns,m->geometry.r_outer,ns*8)||
       memcmp(data+2*ns,p->rho,ns*8)||memcmp(data+3*ns,p->T_e,ns*8)||
       memcmp(data+4*ns,a->abundances,ne*ns*8))goto iterate_read_done;
    for(uint64_t k=0;k<values;k++)if(!(data[k]>=0)||!isfinite(data[k]))goto iterate_read_done;
    mutable_data=data+(4+ne)*ns;
    for(uint64_t s=0;s<ns;s++){
        double charge=0;
        for(uint64_t ip=0;ip<ni;ip++){
            double sum=0,ion=mutable_data[(1+ip)*ns+s];
            charge+=a->ion_pop_stage[ip]*ion;
            for(int gl=a->level_offset[ip];gl<a->level_offset[ip+1];gl++){
                int mapped=n->global_to_nlte_level[gl];if(mapped<0||(uint64_t)mapped>=nl)goto iterate_read_done;
                sum+=mutable_data[(1+ni+(uint64_t)mapped)*ns+s];
            }
            if(a->level_offset[ip]<a->level_offset[ip+1]&&
               (ion==0?sum!=0:fabs(sum/ion-1)>1e-10))goto iterate_read_done;
        }
        if(!(charge>0)||!isfinite(charge)||fabs(mutable_data[s]/charge-1)>1e-8)goto iterate_read_done;
        for(uint64_t e=0;e<ne;e++){
            double total=0,original=0;
            for(int ip=a->elem_ion_offset[e];ip<a->elem_ion_offset[e+1];ip++){
                total+=mutable_data[(1+ip)*ns+s];original+=a->ion_number_density[ip*ns+s];
            }
            if(original==0?total!=0:fabs(total/original-1)>1e-10)goto iterate_read_done;
        }
    }
    for(uint64_t b=0;b<nb;b++){
        frequency[b]=par[1]*exp(((double)b+.5)*par[2]);
        if(!(frequency[b]>0)||!isfinite(frequency[b])||(b&&frequency[b]<=frequency[b-1]))goto iterate_read_done;
    }
    memcpy(j,mutable_data+(1+ni+nl)*ns,ns*nb*8);
    /* 여기부터 실패하면 소유자를 폐기한다. 수렴/물리 과거는 게시하지 않는다. */
    m->prepared=0;
    memcpy(p->n_electron,mutable_data,ns*8);memcpy(a->ion_number_density,mutable_data+ns,ni*ns*8);
    memcpy(n->nlte_level_populations,mutable_data+(1+ni)*ns,nl*ns*8);
    a->population_committed_generation=h[5]+1;
    n->population_required_generation=n->population_committed_generation=a->population_committed_generation;
    {
        PopulationAtomicView view={0};view.n_ions=ni;view.n_levels=a->n_levels;
        view.level_offset=a->level_offset;view.energy_eV=a->level_energy_eV;view.g=a->level_g;
        view.level_Z=a->level_Z;view.level_ion=a->level_ion;view.topion_n=a->topion_n;
        view.topion_ion_index=a->topion_ion_index;view.topion_E_cm=a->topion_E_cm;view.topion_g=a->topion_g;
        if(population_partition_build(&view,p->T_e,ns,a->population_committed_generation,p->T_e_generation,
           a->partition_functions,&a->partition_stamp)!=POP_OK||nlte_precompute_within_sl_frac(n,a,p,(int)ns))goto iterate_read_done;
    }
    nlte_update_tau_sobolev(n,a,o,m->geometry.time_explosion,(int)ns);
    if(tau_sobolev_assert_fresh(o,"initial numerical restart"))goto iterate_read_done;
    o->n_fine=(int)nb;o->nu_lo_fine=par[1];o->dlognu_fine=par[2];
    free(o->nu_fine);o->nu_fine=frequency;frequency=NULL;o->jnu_fine=j;j=NULL;
    *iteration=(int)h[6];m->prepared=1;rc=0;
    fprintf(stderr,"[TOY-ITERATE] resumed=%s completed_iterations=%d physical_history=0 converged=0 J_predictor_only=1\n",path,*iteration);
#undef ITERATE_GET
iterate_read_done:
    fclose(fp);free(maps);free(data);free(frequency);free(j);return rc;
}

int cmf_toy_initial_iterate_start(CMFToyModel *m,const char *path,const char *run_identity,
                                const char *guess_identity,int *offset){
    if(!offset||iterate_identity(run_identity)||
       (guess_identity&&(iterate_identity(guess_identity)||!strcmp(guess_identity,run_identity))))return -1;
    int source_iteration=-1;
    if(cmf_toy_initial_iterate_read(m,path,guess_identity?guess_identity:run_identity,&source_iteration))return -1;
    *offset=guess_identity?0:source_iteration;
    if(guess_identity)fprintf(stderr,"[TOY-INITIAL-GUESS] source_id=%s run_id=%s source_iterations=%d new_completed_iterations=0 physical_history=0 converged=0 J_predictor_only=1 radiation_recompute_required=1\n",
                              guess_identity,run_identity,source_iteration);
    return 0;
}

int cmf_toy_model_begin_epoch(CMFToyModel *m,const CMFPhysicalTimeState *past,
    const CMFEjectaEpoch *epoch,NLTEPhysicalHistory *projected){
    if(!m||!m->prepared||!past||!epoch||!projected||projected==&past->material||
       m->plasma.physical_history||m->nlte.radiation_field.field.generation.computed_generation||
       !past->material.storage||!past->material.energy.density_g_cm3||past->material.decay_epoch_s!=0||
       past->material.energy.n_shells!=(size_t)m->plasma.n_shells||
       !past->material.population_generation||past->material.population_generation==UINT64_MAX||!past->material.temperature_generation||
       past->material.energy.epoch_s!=past->radiation.epoch_s||
       epoch->previous_epoch_s!=past->radiation.epoch_s||!(epoch->epoch_s>epoch->previous_epoch_s)||
       !isfinite(epoch->epoch_s)||!(epoch->previous_epoch_s>0)||
       epoch->n_shells!=m->plasma.n_shells||past->radiation.n_shells!=m->plasma.n_shells||
       !past->radiation.velocity_edges||!epoch->number_density||!epoch->density||
       getenv("LUMINA_FIXED_TE_PROFILE"))return -1;
    AtomicData *a=&m->atom;NLTEConfig *n=&m->nlte;PlasmaState *p=&m->plasma;
    size_t ns=p->n_shells;double dilution=pow(epoch->previous_epoch_s/epoch->epoch_s,3);
    if(!(dilution>0)||!isfinite(dilution)||a->n_elements!=CMF_TOY_ELEMENTS)return -1;
    for(size_t s=0;s<=ns;s++){
        double v=m->ejecta.velocity_edges[s],old=past->radiation.velocity_edges[s];
        if(!isfinite(old)||(v==0?old!=0:fabs(old/v-1)>1e-12))return -1;
        if(s<ns&&(!(past->material.energy.density_g_cm3[s]>0)||
           !isfinite(past->material.energy.density_g_cm3[s])||!(epoch->density[s]>0)||!isfinite(epoch->density[s])||
           fabs(epoch->density[s]/past->material.energy.density_g_cm3[s]/dilution-1)>1e-10))return -1;
    }
    const int source_Z[6]={28,27,26,20,16,14};
    double *after=malloc((size_t)a->n_elements*ns*sizeof(double));if(!after)return -1;
    for(int e=0;e<a->n_elements;e++){
        int k=0;while(k<6&&source_Z[k]!=a->element_Z[e])k++;
        if(k==6){free(after);return -1;}
        for(size_t s=0;s<ns;s++)after[e*ns+s]=epoch->number_density[k*ns+s]/dilution;
    }
    NLTEPhysicalHistory trial={0};
    int status=nlte_physical_history_decay_projection(&trial,&past->material,n,a,epoch->epoch_s,after);
    free(after);if(status!=NLTE_CANDIDATE_OK)return -1;
    /* 여기부터 실패한 모델을 다시 소비하지 못하게 한다. 과거는 건드리지 않는다. */
    m->prepared=0;
    if(cmf_ejecta_apply_epoch(&m->ejecta,epoch,&m->geometry,p,a))goto failed;
    memcpy(p->T_e,trial.energy.temperature_K,ns*sizeof(double));
    for(size_t s=0;s<ns;s++)p->n_electron[s]=trial.energy.n_electron_cm3[s]*dilution;
    for(size_t i=0;i<(size_t)a->n_ion_pops*ns;i++)a->ion_number_density[i]=trial.ion_density[i]*dilution;
    for(size_t i=0;i<(size_t)n->n_nlte_levels_total*ns;i++)n->nlte_level_populations[i]=trial.level_density[i]*dilution;
    a->population_committed_generation=trial.population_generation+1;
    n->population_required_generation=n->population_committed_generation=a->population_committed_generation;
    p->T_e_generation=trial.temperature_generation;
    a210_publication_free(&p->te_publication);
    p->te_publication.te_lane=A210_TE_LANE_FREE_T;p->te_publication.re_root_required=1;
    p->te_publication.required_te_generation=p->te_publication.committed_te_generation=p->T_e_generation;
    if(population_te_manifest_sha256(p->T_e,ns,p->te_publication.te_manifest_sha256)!=POP_OK)goto failed;
    PopulationAtomicView view={0};view.n_ions=a->n_ion_pops;view.n_levels=a->n_levels;
    view.level_offset=a->level_offset;view.energy_eV=a->level_energy_eV;view.g=a->level_g;
    view.level_Z=a->level_Z;view.level_ion=a->level_ion;view.topion_n=a->topion_n;
    view.topion_ion_index=a->topion_ion_index;view.topion_E_cm=a->topion_E_cm;view.topion_g=a->topion_g;
    if(population_partition_build(&view,p->T_e,ns,a->population_committed_generation,
        p->T_e_generation,a->partition_functions,&a->partition_stamp)!=POP_OK||
       nlte_precompute_within_sl_frac(n,a,p,(int)ns))goto failed;
    p->initial_steady_nlte=0;
    nlte_update_tau_sobolev(n,a,&m->opacity,epoch->epoch_s,(int)ns);
    if(tau_sobolev_assert_fresh(&m->opacity,"time predictor from physical history"))goto failed;
    nlte_physical_history_free(projected);*projected=trial;
    p->physical_history=projected;m->prepared=1;
    fprintf(stderr,"[TOY-EPOCH] old=%.17g new=%.17g dilution=%.17g source_mode=%d predictor=DECAYED_HISTORY T=FREE new_epoch_converged=0\n",
        epoch->previous_epoch_s,epoch->epoch_s,dilution,epoch->source_mode);
    return 0;
failed:
    nlte_physical_history_free(&trial);p->physical_history=NULL;return -1;
}

int cmf_toy_bootstrap_main(int argc,char **argv){
    if(argc!=6){fprintf(stderr,"usage: %s --cmf-toy-bootstrap ejecta atomic_dir shells output.csv\n",argv[0]);return 2;}
    char *end;errno=0;long ns=strtol(argv[4],&end,10);
    if(errno||*end||ns<2||ns>256)return 2;
    /* Ni56/Co56/Fe56 동위원소 질량; Ca/S/Si는 원자표의 원소 평균 질량.
     * 공개 계산의 원자/핵자료 정체성 일치까지 주장하지 않는다. */
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel model={0};int rc=cmf_toy_model_prepare(&model,argv[2],argv[3],(int)ns,mass);
    if(!rc)rc=cmf_toy_model_write_seed(&model,argv[5]);
    cmf_toy_model_free(&model);return rc?1:0;
}
