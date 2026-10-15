#include "cmf_time_epoch.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* native little-endian, fixed-width header. 총2GiB 이내의 정상 이력만 읽는다. */
static int checkpoint_counts(uint64_t ns,uint64_t nb,uint64_t ni,uint64_t nl,
    uint64_t *material,uint64_t *radiation,uint64_t *bytes){
    uint32_t endian=1;
    if(sizeof(int)!=4||sizeof(double)!=8||*(unsigned char*)&endian!=1||
       !ns||ns>256||!nb||nb>INT32_MAX||!ni||ni>INT32_MAX||!nl||nl>INT32_MAX)return -1;
    *material=(5+ni+nl)*ns;*radiation=ns+1+(2*ns+1)*nb;
    *bytes=8+7*8+4*8+4*(nl+2*ni)+8*(*material+*radiation)+8;
    return *bytes<=UINT64_C(2147483648)?0:-1;
}

int cmf_physical_time_state_write(const CMFPhysicalTimeState *s,const char *path){
    if(!s||!path||!s->material.storage||!s->radiation_storage||
       !s->material.level_global||!s->material.ion_Z||!s->material.ion_stage||
       s->material.decay_epoch_s!=0||s->radiation.epoch_s!=s->material.energy.epoch_s||
       s->radiation.n_shells!=(int)s->material.energy.n_shells)return -1;
    uint64_t ns=s->radiation.n_shells,nb=s->radiation.n_bins,ni=s->material.n_ions,nl=s->material.n_levels;
    uint64_t material,radiation,bytes;
    if(checkpoint_counts(ns,nb,ni,nl,&material,&radiation,&bytes))return -1;
    uint64_t head[]={1,ns,nb,ni,nl,s->material.population_generation,s->material.temperature_generation};
    double parameters[]={s->radiation.epoch_s,s->radiation.nu_min,s->radiation.dlognu,s->material.decay_epoch_s};
    FILE *fp=fopen(path,"wbx");if(!fp)return -1;
    int rc=-1;
#define CHECKPOINT_PUT(ptr,size,count) do{if(fwrite((ptr),(size),(count),fp)!=(count))goto write_done;}while(0)
    CHECKPOINT_PUT("CMFINIT1",1,8);CHECKPOINT_PUT(head,8,7);CHECKPOINT_PUT(parameters,8,4);
    CHECKPOINT_PUT(s->material.level_global,4,nl);CHECKPOINT_PUT(s->material.ion_Z,4,ni);CHECKPOINT_PUT(s->material.ion_stage,4,ni);
    CHECKPOINT_PUT(s->material.storage,8,material);CHECKPOINT_PUT(s->radiation_storage,8,radiation);
    CHECKPOINT_PUT("CMFDONE1",1,8);rc=0;
#undef CHECKPOINT_PUT
write_done:
    if(fclose(fp))rc=-1;
    if(!rc)fprintf(stderr,"[CMF-HISTORY] wrote=%s bytes=%llu epoch=%.17g\n",path,
                   (unsigned long long)bytes,parameters[0]);
    return rc;
}

int cmf_physical_time_state_read(CMFPhysicalTimeState *out,const char *path,
    const NLTEConfig *n,const AtomicData *a,const Geometry *g){
    if(!out||!path||!n||!a||!g||g->n_shells<1||!g->v_inner||!g->v_outer||
       a->n_ion_pops<1||n->n_nlte_levels_total<1||!n->nlte_to_global_level||
       !a->ion_pop_Z||!a->ion_pop_stage)return -1;
    FILE *fp=fopen(path,"rb");if(!fp)return -1;
    CMFPhysicalTimeState raw={0},verified={0};int rc=-1;
    uint64_t h[7],material,radiation,bytes;double par[4],*partition=NULL;
    char marker[8];long length;
#define CHECKPOINT_GET(ptr,size,count) do{if(fread((ptr),(size),(count),fp)!=(count))goto read_done;}while(0)
    CHECKPOINT_GET(marker,1,8);if(memcmp(marker,"CMFINIT1",8))goto read_done;
    CHECKPOINT_GET(h,8,7);CHECKPOINT_GET(par,8,4);
    if(h[0]!=1||h[1]!=(uint64_t)g->n_shells||h[3]!=(uint64_t)a->n_ion_pops||
       h[4]!=(uint64_t)n->n_nlte_levels_total||!h[5]||!h[6]||par[3]!=0||
       checkpoint_counts(h[1],h[2],h[3],h[4],&material,&radiation,&bytes)||
       !(par[0]>0)||!isfinite(par[0])||!(par[1]>0)||!isfinite(par[1])||
       !(par[2]>0)||!isfinite(par[2]))goto read_done;
    if(fseek(fp,0,SEEK_END)||(length=ftell(fp))<0||(uint64_t)length!=bytes||fseek(fp,96,SEEK_SET))goto read_done;
    raw.material.level_global=malloc(h[4]*4);raw.material.ion_Z=malloc(h[3]*4);raw.material.ion_stage=malloc(h[3]*4);
    raw.material.storage=malloc(material*8);raw.radiation_storage=malloc(radiation*8);
    partition=malloc(h[3]*h[1]*8);
    if(!raw.material.level_global||!raw.material.ion_Z||!raw.material.ion_stage||
       !raw.material.storage||!raw.radiation_storage||!partition)goto read_done;
    CHECKPOINT_GET(raw.material.level_global,4,h[4]);CHECKPOINT_GET(raw.material.ion_Z,4,h[3]);
    CHECKPOINT_GET(raw.material.ion_stage,4,h[3]);CHECKPOINT_GET(raw.material.storage,8,material);
    CHECKPOINT_GET(raw.radiation_storage,8,radiation);CHECKPOINT_GET(marker,1,8);
    if(memcmp(marker,"CMFDONE1",8)||fgetc(fp)!=EOF||
       memcmp(raw.material.level_global,n->nlte_to_global_level,h[4]*4)||
       memcmp(raw.material.ion_Z,a->ion_pop_Z,h[3]*4)||
       memcmp(raw.material.ion_stage,a->ion_pop_stage,h[3]*4))goto read_done;
    for(uint64_t i=0;i<material;i++)if(!isfinite(raw.material.storage[i])||raw.material.storage[i]<0)goto read_done;
    {
        size_t ns=h[1],ni=h[3],nl=h[4],nb=h[2];double *v=raw.radiation_storage,*m=raw.material.storage;
        for(size_t s=0;s<=ns;s++){
            double expect=s<ns?g->v_inner[s]:g->v_outer[ns-1];
            if(!isfinite(v[s])||!isfinite(expect)||v[s]<0||expect<0||
               (expect==0?v[s]!=0:fabs(v[s]/expect-1)>1e-12))goto read_done;
        }
        raw.radiation=(CMFTimeRadiationHistory){(int)ns,(int)nb,par[0],par[1],par[2],v,v+ns+1,v+ns+1+ns*nb};
        AtomicData av=*a;NLTEConfig nv=*n;PlasmaState pv={0};PopulationAtomicView view={0};
        av.ion_number_density=m+5*ns;av.partition_functions=partition;av.population_committed_generation=h[5];
        /* 내부에너지 계산은 저장 이온/준위 핵수를 사용한다. 현재 실행의
         * 원소 질량분율을 과거에 덮어쓰거나 다시 정규화하지 않는다. */
        nv.nlte_level_populations=m+(5+ni)*ns;nv.population_committed_generation=h[5];
        pv.n_shells=(int)ns;pv.rho=m;pv.T_e=m+ns;pv.n_electron=m+3*ns;pv.T_e_generation=h[6];
        view.n_ions=ni;view.n_levels=a->n_levels;view.level_offset=a->level_offset;
        view.energy_eV=a->level_energy_eV;view.g=a->level_g;view.level_Z=a->level_Z;view.level_ion=a->level_ion;
        view.topion_n=a->topion_n;view.topion_ion_index=a->topion_ion_index;
        view.topion_E_cm=a->topion_E_cm;view.topion_g=a->topion_g;
        if(population_partition_build(&view,pv.T_e,ns,h[5],h[6],partition,&av.partition_stamp)!=POP_OK||
           cmf_physical_time_state_capture(&verified,&nv,&av,&pv,&raw.radiation))goto read_done;
        for(size_t s=0;s<ns;s++){
            double charge=0;
            for(size_t i=0;i<ni;i++)charge+=a->ion_pop_stage[i]*av.ion_number_density[i*ns+s];
            if(!isfinite(charge)||(charge==0?pv.n_electron[s]!=0:fabs(pv.n_electron[s]/charge-1)>1e-8))goto read_done;
            for(size_t k=2;k<=4;k+=2){
                double saved=m[k*ns+s],actual=verified.material.storage[k*ns+s];
                if(actual==0?saved!=0:fabs(saved/actual-1)>1e-10)goto read_done;
            }
        }
        /* 재계산은 검증용이다. 저장된 과거 값을 반올림조차 바꾸지 않는다. */
        memcpy(verified.material.storage,m,material*8);
        (void)nl;
    }
    cmf_physical_time_state_free(out);*out=verified;memset(&verified,0,sizeof(verified));rc=0;
#undef CHECKPOINT_GET
read_done:
    fclose(fp);free(partition);cmf_physical_time_state_free(&raw);cmf_physical_time_state_free(&verified);
    return rc;
}

void cmf_physical_time_state_free(CMFPhysicalTimeState *s){
    if(s){nlte_physical_history_free(&s->material);free(s->radiation_storage);memset(s,0,sizeof(*s));}
}
int cmf_physical_time_state_capture(CMFPhysicalTimeState *out,const NLTEConfig *n,
    const AtomicData *a,const PlasmaState *p,const CMFTimeRadiationHistory *r){
    if(!out||!n||!a||!p||!r||r->n_shells<=0||r->n_bins<=0||
       r->n_shells!=p->n_shells||!(r->epoch_s>0)||!isfinite(r->epoch_s)||
       !(r->nu_min>0)||!isfinite(r->nu_min)||!(r->dlognu>0)||!isfinite(r->dlognu)||
       !r->velocity_edges||!r->J||!r->H)return -1;
    size_t ns=r->n_shells,nb=r->n_bins;
    if(ns+1>SIZE_MAX/nb || (ns+1)*nb>SIZE_MAX/(3*sizeof(double)))return -1;
    size_t cells=ns*nb,faces=(ns+1)*nb;
    for(size_t s=0;s<=ns;s++)if(!isfinite(r->velocity_edges[s])||r->velocity_edges[s]<0||
        (s&&!(r->velocity_edges[s]>r->velocity_edges[s-1])))return -1;
    for(size_t i=0;i<cells;i++)if(!(r->J[i]>=0)||!isfinite(r->J[i]))return -1;
    for(size_t i=0;i<faces;i++)if(!isfinite(r->H[i]))return -1;
    CMFPhysicalTimeState trial={0};
    trial.radiation_storage=malloc((ns+1+cells+faces)*sizeof(double));
    if(!trial.radiation_storage)return -1;
    if(nlte_physical_history_capture(&trial.material,n,a,p,r->epoch_s)!=NLTE_CANDIDATE_OK){
        cmf_physical_time_state_free(&trial);return -1;
    }
    trial.radiation=*r;
    trial.radiation.velocity_edges=trial.radiation_storage;
    trial.radiation.J=trial.radiation_storage+ns+1;
    trial.radiation.H=trial.radiation_storage+ns+1+cells;
    memcpy(trial.radiation_storage,r->velocity_edges,(ns+1)*sizeof(double));
    memcpy(trial.radiation_storage+ns+1,r->J,cells*sizeof(double));
    memcpy(trial.radiation_storage+ns+1+cells,r->H,faces*sizeof(double));
    cmf_physical_time_state_free(out);*out=trial;return 0;
}

static double update(const double *a,const double *b,size_t n){
    double result=0;
    for(size_t i=0;i<n;i++){
        if(!(a[i]>=0)||!(b[i]>=0)||!isfinite(a[i])||!isfinite(b[i]))return INFINITY;
        double scale=fmax(a[i],b[i]);if(scale)result=fmax(result,fabs(a[i]-b[i])/scale);
    }
    return result;
}
/* H는 부호가 있으므로 영 교차를 임의 floor로 정규화하지 않고 인접 J를 쓴다. */
static double flux_update(const double *ja,const double *jb,const double *ha,
                          const double *hb,size_t ns,size_t nb){
    double result=0;
    for(size_t s=0;s<=ns;s++)for(size_t b=0;b<nb;b++){
        size_t left=(s?s-1:0)*nb+b,right=(s<ns?s:ns-1)*nb+b,i=s*nb+b;
        if(!isfinite(ha[i])||!isfinite(hb[i]))return INFINITY;
        double scale=fmax(fmax(ja[left],jb[left]),fmax(ja[right],jb[right]));
        double diff=fabs(ha[i]-hb[i]);
        if(scale>0)result=fmax(result,diff/scale);else if(diff)return INFINITY;
    }
    return result;
}

CMFTimeEpochStatus cmfgen_solve_initial_nlte_saved(CMFGENState *cs,Geometry *g,OpacityState *o,
    BFOpacity *bf,PlasmaState *p,NLTEConfig *n,AtomicData *a,GammaDeposition *gamma,
    double T_inner,int cap,double tol,CMFPhysicalTimeState *initial,CMFInitialNLTEReport *out,
    CMFInitialIterationSave save,void *context){
    CMFInitialNLTEReport rep={0};rep.tolerance=tol;
    rep.electron_update=rep.population_update=rep.radiation_update=INFINITY;
    if(out)*out=rep;
    if(cs){cs->common_obs_ready=0;cs->initial_face_ready=0;cs->initial_face_requested=0;}
    if(!cs||!g||!o||!bf||!p||!n||!a||!gamma||!initial||!out||cap<1||!(tol>0&&tol<1)||
       !isfinite(tol)||!(T_inner>0)||!isfinite(T_inner)||cs->time_trial||p->physical_history||
       p->initial_steady_nlte!=1||!getenv("LUMINA_FIXED_TE_PROFILE")||
       !p->T_e||!p->n_electron||!a->ion_number_density||!a->ion_pop_stage||!n->nlte_level_populations||
       p->n_shells<2||p->n_shells!=g->n_shells||cs->n_shells!=p->n_shells||o->n_shells!=p->n_shells||
       a->n_ion_pops<1||n->n_nlte_levels_total<1||
       gamma_deposition_require(gamma,g->time_explosion)!=GAMMA_PUBLICATION_OK)return CMF_EPOCH_INVALID;
    size_t ns=p->n_shells,ni=a->n_ion_pops,nl=n->n_nlte_levels_total;
    /* 잘못된 파일이 첫 material 게시에서 입력 온도를 바꾸지 않도록 선확인한다. */
    double *profile=NULL,tmin=0,tmax=0;char hash[65];
    const char *reason=a210_fixed_te_profile_load(getenv("LUMINA_FIXED_TE_PROFILE"),ns,
                                                &profile,hash,&tmin,&tmax);
    if(reason||!profile){free(profile);return CMF_EPOCH_INVALID;}
    int same=!memcmp(profile,p->T_e,ns*sizeof(double));free(profile);
    if(!same)return CMF_EPOCH_INVALID;
    if(ni>SIZE_MAX-nl-2||(ni+nl+2)>SIZE_MAX/ns/sizeof(double))return CMF_EPOCH_ALLOCATION;
    double *saved=malloc((ni+nl+2)*ns*sizeof(double)),*j=NULL;
    if(!saved)return CMF_EPOCH_ALLOCATION;
    double *te=saved,*ne=te+ns,*ions=ne+ns,*levels=ions+ni*ns;
    memcpy(te,p->T_e,ns*sizeof(double));
    CMFTimeEpochStatus status=CMF_EPOCH_RADIATION;
    rep.radiation_calls++;
    if(cmfgen_initial_radiation_publish(cs,g,o,bf,p,n,a,T_inner,0)||
       o->n_fine<2||!o->jnu_fine||!o->nu_fine)goto initial_done;
    size_t nb=o->n_fine;
    if(nb>SIZE_MAX/ns/sizeof(double)){status=CMF_EPOCH_ALLOCATION;goto initial_done;}
    j=malloc(ns*nb*sizeof(double));
    if(!j){status=CMF_EPOCH_ALLOCATION;goto initial_done;}
    if(save&&save(context,0)){status=CMF_EPOCH_HISTORY;goto initial_done;}
    for(int it=0;it<cap;it++){
        memcpy(ne,p->n_electron,ns*sizeof(double));memcpy(ions,a->ion_number_density,ni*ns*sizeof(double));
        memcpy(levels,n->nlte_level_populations,nl*ns*sizeof(double));memcpy(j,o->jnu_fine,ns*nb*sizeof(double));
        n->current_iter=it;
        if(lumina_r7_publish_and_solve_te(o,bf,a,p,n,gamma,g->time_explosion,(int)ns,1,"DET-INITIAL",it)||
           p->te_publication.te_lane!=A210_TE_LANE_FIXED_T||p->te_publication.re_root_required||
           memcmp(te,p->T_e,ns*sizeof(double))||p->physical_history||p->initial_steady_nlte!=1){
            status=CMF_EPOCH_MATERIAL;goto initial_done;
        }
        rep.iterations=it+1;rep.electron_update=update(ne,p->n_electron,ns);
        rep.population_update=fmax(update(ions,a->ion_number_density,ni*ns),update(levels,n->nlte_level_populations,nl*ns));
        rep.radiation_calls++;
        if(cmfgen_initial_radiation_publish(cs,g,o,bf,p,n,a,T_inner,0)||
           o->n_fine!=(int)nb||!o->jnu_fine){status=CMF_EPOCH_RADIATION;goto initial_done;}
        rep.radiation_update=update(j,o->jnu_fine,ns*nb);
        if(save&&save(context,it+1)){status=CMF_EPOCH_HISTORY;goto initial_done;}
        fprintf(stderr,"[CMF-INITIAL] epoch=%.17g iteration=%d ne=%.9g population=%.9g J=%.9g tolerance=%.9g T=INPUT_FIXED\n",
            g->time_explosion,it+1,rep.electron_update,rep.population_update,rep.radiation_update,tol);
        if(rep.electron_update<tol&&rep.population_update<tol&&rep.radiation_update<tol){
            for(size_t s=0;s<ns;s++){
                double charge=0;
                for(size_t i=0;i<ni;i++)charge+=a->ion_pop_stage[i]*a->ion_number_density[i*ns+s];
                if(!(charge>0)||!isfinite(charge)||fabs(p->n_electron[s]/charge-1)>1e-8){
                    status=CMF_EPOCH_MATERIAL;goto initial_done;
                }
            }
            memcpy(j,o->jnu_fine,ns*nb*sizeof(double));rep.radiation_calls++;
            cs->initial_face_requested=1;
            if(cmfgen_initial_radiation_publish(cs,g,o,bf,p,n,a,T_inner,1)||!cs->common_obs_ready||
               o->n_fine!=(int)nb||!o->jnu_fine){status=CMF_EPOCH_RADIATION;goto initial_done;}
            if(update(j,o->jnu_fine,ns*nb)>=tol){status=CMF_EPOCH_NOT_CONVERGED;goto initial_done;}
            if(!cs->initial_face_ready||!cs->initial_face_H||cs->initial_face_bins!=(int)nb||
               cs->initial_face_nu_min!=o->nu_lo_fine||cs->initial_face_dlog!=o->dlognu_fine||
               !(cs->initial_face_J_residual<tol)){status=CMF_EPOCH_HISTORY;goto initial_done;}
            double *velocity=malloc((ns+1)*sizeof(double));
            if(!velocity){status=CMF_EPOCH_ALLOCATION;goto initial_done;}
            for(size_t s=0;s<=ns;s++)velocity[s]=(s<ns?g->r_inner[s]:g->r_outer[ns-1])/g->time_explosion;
            CMFTimeRadiationHistory radiation={(int)ns,(int)nb,g->time_explosion,
                cs->initial_face_nu_min,cs->initial_face_dlog,velocity,o->jnu_fine,cs->initial_face_H};
            int captured=cmf_physical_time_state_capture(initial,n,a,p,&radiation);
            free(velocity);
            if(captured){status=CMF_EPOCH_HISTORY;goto initial_done;}
            rep.converged=1;status=CMF_EPOCH_OK;goto initial_done;
        }
    }
    status=CMF_EPOCH_NOT_CONVERGED;
initial_done:
    cs->initial_face_requested=0;
    if(status!=CMF_EPOCH_OK){cs->common_obs_ready=0;cs->initial_face_ready=0;}
    *out=rep;free(saved);free(j);return status;
}

CMFTimeEpochStatus cmfgen_solve_initial_nlte(CMFGENState *cs,Geometry *g,OpacityState *o,
    BFOpacity *bf,PlasmaState *p,NLTEConfig *n,AtomicData *a,GammaDeposition *gamma,
    double T_inner,int cap,double tol,CMFPhysicalTimeState *initial,CMFInitialNLTEReport *out){
    return cmfgen_solve_initial_nlte_saved(cs,g,o,bf,p,n,a,gamma,T_inner,cap,tol,
                                         initial,out,NULL,NULL);
}

CMFTimeEpochStatus cmfgen_solve_time_epoch(CMFGENState *cs,Geometry *g,OpacityState *o,
    BFOpacity *bf,PlasmaState *p,NLTEConfig *n,AtomicData *a,GammaDeposition *gamma,
    double T_inner,int cap,double tol,CMFPhysicalTimeState *next,CMFTimeEpochReport *out){
    CMFTimeEpochReport rep={0};rep.tolerance=tol;
    rep.temperature_update=rep.electron_update=rep.population_update=
        rep.radiation_update=rep.flux_update=INFINITY;
    if(out)*out=rep;
    if(cs)cs->common_obs_ready=0;
    if(!cs||!g||!o||!bf||!p||!n||!a||!gamma||!next||!out||cap<1||
       !(tol>0&&tol<1)||!isfinite(tol)||!(T_inner>0)||!isfinite(T_inner)||
       !cs->time_trial||!cs->time_trial->old||!cs->time_trial->inner_H||
       !p->physical_history||p->initial_steady_nlte||!p->rho||!p->T_e||!p->n_electron||
       !a->ion_number_density||!a->ion_pop_Z||!a->ion_pop_stage||
       !n->nlte_level_populations||!n->nlte_to_global_level||
       a->n_ion_pops<=0||n->n_nlte_levels_total<=0||p->n_shells<2||getenv("LUMINA_FIXED_TE_PROFILE")||
       p->n_shells!=g->n_shells||p->n_shells!=cs->n_shells||o->n_shells!=p->n_shells||
       gamma_deposition_require(gamma,g->time_explosion)!=GAMMA_PUBLICATION_OK)return CMF_EPOCH_INVALID;
    CMFTimeRadiationTrial *trial=cs->time_trial;
    const CMFTimeRadiationHistory *oldr=trial->old;
    const NLTEPhysicalHistory *oldm=p->physical_history;
    size_t ns=p->n_shells,ni=a->n_ion_pops,nl=n->n_nlte_levels_total;
    if(&next->material==oldm||&next->radiation==oldr||oldr->n_shells!=(int)ns||oldr->n_bins<=0||
       oldm->energy.n_shells!=ns||oldm->n_ions!=ni||oldm->n_levels!=nl||
       oldm->energy.epoch_s!=oldr->epoch_s||trial->epoch_s!=g->time_explosion||
       (oldm->decay_epoch_s!=0&&oldm->decay_epoch_s!=trial->epoch_s)||
       !(oldr->epoch_s>0)||!(trial->epoch_s>oldr->epoch_s)||
       !oldm->energy.density_g_cm3||!oldm->level_global||!oldm->ion_Z||!oldm->ion_stage||
       memcmp(oldm->level_global,n->nlte_to_global_level,nl*sizeof(int))||
       memcmp(oldm->ion_Z,a->ion_pop_Z,ni*sizeof(int))||
       memcmp(oldm->ion_stage,a->ion_pop_stage,ni*sizeof(int)))return CMF_EPOCH_INVALID;
    double dilution=pow(oldr->epoch_s/trial->epoch_s,3);
    for(size_t s=0;s<ns;s++)if(!(oldm->energy.density_g_cm3[s]>0)||!(p->rho[s]>0)||
        !isfinite(p->rho[s])||!isfinite(oldm->energy.density_g_cm3[s])||!(dilution>0)||
        fabs((p->rho[s]/oldm->energy.density_g_cm3[s])/dilution-1)>1e-10)return CMF_EPOCH_INVALID;
    size_t nb=oldr->n_bins;
    if(ni>SIZE_MAX-nl-2||(ni+nl+2)>SIZE_MAX/ns||ns+1>SIZE_MAX/nb)return CMF_EPOCH_ALLOCATION;
    size_t material=(ni+nl+2)*ns,cells=ns*nb,faces=(ns+1)*nb;
    if(material>SIZE_MAX/sizeof(double)||faces>SIZE_MAX/sizeof(double)||cells>SIZE_MAX/sizeof(double)-faces||
       material>SIZE_MAX/sizeof(double)-cells-faces)return CMF_EPOCH_ALLOCATION;
    double *saved=malloc((material+cells+faces)*sizeof(double));
    if(!saved)return CMF_EPOCH_ALLOCATION;
    double *te=saved,*ne=te+ns,*ions=ne+ns,*levels=ions+ni*ns;
    double *j=levels+nl*ns,*h=j+cells;
    CMFTimeEpochStatus status=CMF_EPOCH_RADIATION;
    cs->common_obs_ready=0;
    rep.radiation_calls++;
    if(cmfgen_time_radiation_publish(cs,g,o,bf,p,n,a,T_inner,0))goto done;
    status=CMF_EPOCH_NOT_CONVERGED;
    for(int it=0;it<cap;it++){
        if(!trial->report.converged||!trial->J||!trial->H||
           p->physical_history!=oldm||trial->old!=oldr){status=CMF_EPOCH_INVALID;goto done;}
        memcpy(te,p->T_e,ns*sizeof(double));memcpy(ne,p->n_electron,ns*sizeof(double));
        memcpy(ions,a->ion_number_density,ni*ns*sizeof(double));
        memcpy(levels,n->nlte_level_populations,nl*ns*sizeof(double));
        memcpy(j,trial->J,cells*sizeof(double));memcpy(h,trial->H,faces*sizeof(double));
        n->current_iter=it;
        if(lumina_r7_publish_and_solve_te(o,bf,a,p,n,gamma,g->time_explosion,(int)ns,1,"DET-TIME",it)){
            status=CMF_EPOCH_MATERIAL;goto done;
        }
        const ElectronTemperaturePublication *tp=&p->te_publication;
        if(tp->te_lane!=A210_TE_LANE_FREE_T||!tp->re_root_required||tp->n_shells!=ns||!tp->ledger||
           tp->committed_te_generation!=p->T_e_generation||
           tp->population_generation!=a->population_committed_generation||
           p->physical_history!=oldm||trial->old!=oldr){status=CMF_EPOCH_MATERIAL;goto done;}
        for(size_t s=0;s<ns;s++)if(tp->ledger[s].adiabatic_model!=A210_MATERIAL_TIME_HISTORY){
            status=CMF_EPOCH_MATERIAL;goto done;
        }
        rep.iterations=it+1;
        rep.temperature_update=update(te,p->T_e,ns);
        rep.electron_update=update(ne,p->n_electron,ns);
        rep.population_update=fmax(update(ions,a->ion_number_density,ni*ns),
                                   update(levels,n->nlte_level_populations,nl*ns));
        rep.radiation_calls++;
        if(cmfgen_time_radiation_publish(cs,g,o,bf,p,n,a,T_inner,0)){
            status=CMF_EPOCH_RADIATION;goto done;
        }
        rep.radiation_update=update(j,trial->J,cells);
        rep.flux_update=flux_update(j,trial->J,h,trial->H,ns,nb);
        fprintf(stderr,"[CMF-EPOCH] old=%.17g new=%.17g iteration=%d "
            "Te=%.9g ne=%.9g population=%.9g J=%.9g H_over_J=%.9g tolerance=%.9g\n",
            oldr->epoch_s,trial->epoch_s,it+1,rep.temperature_update,rep.electron_update,
            rep.population_update,rep.radiation_update,rep.flux_update,tol);
        if(rep.temperature_update<tol&&rep.electron_update<tol&&rep.population_update<tol&&
           rep.radiation_update<tol&&rep.flux_update<tol){
            /* 마지막 물질에서 관측 형식해. 추가 물질 갱신은 없다. */
            memcpy(j,trial->J,cells*sizeof(double));memcpy(h,trial->H,faces*sizeof(double));
            rep.radiation_calls++;
            if(cmfgen_time_radiation_publish(cs,g,o,bf,p,n,a,T_inner,1)||!cs->common_obs_ready){
                status=CMF_EPOCH_RADIATION;goto done;
            }
            if(update(j,trial->J,cells)>=tol||flux_update(j,trial->J,h,trial->H,ns,nb)>=tol){
                status=CMF_EPOCH_NOT_CONVERGED;goto done;
            }
            CMFTimeRadiationHistory current=*oldr;current.epoch_s=trial->epoch_s;
            current.J=trial->J;current.H=trial->H;
            if(cmf_physical_time_state_capture(next,n,a,p,&current)){
                status=CMF_EPOCH_HISTORY;goto done;
            }
            rep.converged=1;status=CMF_EPOCH_OK;break;
        }
    }
done:
    if(status!=CMF_EPOCH_OK)cs->common_obs_ready=0;
    rep.radiation_generation=n->radiation_field.field.generation.computed_generation;
    rep.population_generation=a->population_committed_generation;
    *out=rep;free(saved);return status;
}
