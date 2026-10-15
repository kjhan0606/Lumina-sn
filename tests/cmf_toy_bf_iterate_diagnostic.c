/* 불변 초기 수치 반복의 BF 전리 균형. 생산/RTE/NLTE를 갱신하지 않는다. */
#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) do {if(!(x)){fprintf(stderr,"TOY_BF_ITERATE_FAIL line=%d %s\n",__LINE__,#x);goto done;}} while(0)

/* 이전 물질은 재사용하지 않는다. 기존 C reader로 이미 검증한 불변 사본에서
 * 같은 신원/격자/크기/끝표지의 J 배열만 읽는다. runner가 전체 파일 hash도 확인한다. */
static double *previous_field(const CMFToyModel *m,const char *path,const char *id,int *iteration){
    FILE *fp=fopen(path,"rb");if(!fp)return NULL;
    double *j=NULL,par[3];uint64_t h[7];char marker[8],stored_id[64];
    size_t ns=m->plasma.n_shells,ni=m->atom.n_ion_pops,nl=m->nlte.n_nlte_levels_total;
    size_t ne=m->atom.n_elements,nb=m->opacity.n_fine;
    uint64_t offset=152+4*(nl+2*ni)+8*ns*(5+ne+ni+nl),bytes=offset+8*ns*nb+8;
    if(fread(marker,1,8,fp)!=8||memcmp(marker,"CMFITER1",8)||fread(stored_id,1,64,fp)!=64||
       memcmp(stored_id,id,64)||fread(h,8,7,fp)!=7||fread(par,8,3,fp)!=3||
       h[0]!=ns||h[1]!=ni||h[2]!=nl||h[3]!=ne||h[4]!=nb||h[6]>INT32_MAX||
       par[0]!=m->geometry.time_explosion||par[1]!=m->opacity.nu_lo_fine||
       par[2]!=m->opacity.dlognu_fine||fseek(fp,0,SEEK_END)||ftell(fp)!=(long)bytes||
       fseek(fp,(long)offset,SEEK_SET))goto done;
    j=malloc(ns*nb*sizeof(*j));if(!j)goto done;
    if(fread(j,8,ns*nb,fp)!=ns*nb||fread(marker,1,8,fp)!=8||memcmp(marker,"ITEREND1",8)||fgetc(fp)!=EOF){
        free(j);j=NULL;goto done;
    }
    for(size_t k=0;k<ns*nb;k++)if(j[k]<0||!isfinite(j[k])){free(j);j=NULL;goto done;}
    *iteration=(int)h[6];
done:
    fclose(fp);return j;
}

/* 조건부 율 분해 전용. 이온 총량과 실제 소유자 인구는 보존하고 Si/S II/III의
 * 이온 내 준위 분포만 private 배열에서 LTE로 바꾼다. 새 SE 해가 아니다. */
static double *lte_levels(const CMFToyModel *m){
    const AtomicData *a=&m->atom;const NLTEConfig *n=&m->nlte;
    int ns=m->plasma.n_shells;
    size_t count=(size_t)n->n_nlte_levels_total*ns;
    double *pop=malloc(count*sizeof(*pop));if(!pop)return NULL;
    memcpy(pop,n->nlte_level_populations,count*sizeof(*pop));
    PopulationAtomicView av={0};av.n_ions=a->n_ion_pops;av.n_levels=a->n_levels;
    av.level_offset=a->level_offset;av.energy_eV=a->level_energy_eV;av.g=a->level_g;
    av.level_Z=a->level_Z;av.level_ion=a->level_ion;
    for(int ip=0;ip<a->n_ion_pops;ip++){
        int z=a->ion_pop_Z[ip],q=a->ion_pop_stage[ip];
        if((z!=14&&z!=16)||(q!=1&&q!=2))continue;
        for(int s=0;s<ns;s++){
            long double sum=0;
            for(int gl=a->level_offset[ip];gl<a->level_offset[ip+1];gl++){
                int fl=n->global_to_nlte_level[gl];double f=0;
                if(fl<0||fl>=n->n_nlte_levels_total)goto fail;
                PopulationStatus status=population_lte_level_fraction(&av,ip,gl,m->plasma.T_e[s],
                    a->partition_functions[(size_t)ip*ns+s],&f);
                if(status!=POP_OK&&status!=POP_EXACT_ZERO)goto fail;
                sum+=f;pop[(size_t)fl*ns+s]=a->ion_number_density[(size_t)ip*ns+s]*f;
            }
            if(fabsl(sum-1)>1e-10L)goto fail;
        }
    }
    return pop;
fail:
    free(pop);return NULL;
}

static int evaluate(CMFToyModel *m,double *fine_J,const double *pop,const double *ions,
    const double *ne,const double *temperature,int field_iteration,int population_iteration,FILE *fp){
    int rc=-1,ns=m->plasma.n_shells,nb=NLTE_N_FREQ_BINS;
    RadiationFieldOwner owner={0};RadiationFieldView view={0};
    double *edges=NULL,*J=NULL;RadiationFieldValidityState *valid=NULL;
    const AtomicData *a=&m->atom;const NLTEConfig *n=&m->nlte;
    const BfChannels *channels=a->bf_channels;
    REQUIRE(channels&&n->radfield_view_status==RADIATION_FIELD_VIEW_DISABLED);
    J=malloc((size_t)ns*nb*sizeof(*J));edges=malloc((nb+1)*sizeof(*edges));
    valid=malloc((size_t)ns*nb*sizeof(*valid));REQUIRE(J&&edges&&valid);
    CMFGENState source={0},target={0};
    source.n_shells=ns;source.n_bins=m->opacity.n_fine;source.J=fine_J;
    source.nu_min=m->opacity.nu_lo_fine;source.d_log_nu=m->opacity.dlognu_fine;
    source.nu_max=source.nu_min*exp(source.n_bins*source.d_log_nu);
    target.n_shells=ns;target.n_bins=nb;target.J=J;
    target.nu_min=NLTE_NU_MIN;target.nu_max=NLTE_NU_MAX;
    target.d_log_nu=log(NLTE_NU_MAX/NLTE_NU_MIN)/nb;
    REQUIRE(!cmfgen_rebin_log_field(&source,&target));
    for(int b=0;b<=nb;b++)edges[b]=NLTE_NU_MIN*exp(b*target.d_log_nu);
    edges[0]=NLTE_NU_MIN;edges[nb]=NLTE_NU_MAX;
    for(size_t k=0;k<(size_t)ns*nb;k++)valid[k]=J[k]==0?RADIATION_FIELD_EXACT_ZERO:RADIATION_FIELD_VALID;
    REQUIRE(!radiation_field_owner_init(&owner,(size_t)ns));
    RadiationFieldCommitRequest request={0};
    request.provenance_kind=RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer="PURE_CMFGEN_COMOVING_CONSERVATIVE_REBIN";
    request.generation=1;request.epoch=m->geometry.time_explosion;
    request.n_shells=(size_t)ns;request.v_inner=m->geometry.v_inner;request.v_outer=m->geometry.v_outer;
    request.source_n_bins=(size_t)nb;request.source_frequency_bin_edges=edges;
    request.source_J_nu=J;request.source_validity=valid;request.statistic_kind=RADIATION_FIELD_DETERMINISTIC;
    REQUIRE(!radiation_field_commit(&owner,&request));
    REQUIRE(radiation_field_read_view(&owner,request.epoch,ns,1,&view)==RADIATION_FIELD_VIEW_OK);
    NLTEConfig diagnostic=*n;diagnostic.radfield_view=view;
    diagnostic.radfield_view_status=RADIATION_FIELD_VIEW_OK;diagnostic.bf_rate_cache=NULL;
    int rows=0;
    for(int s=0;s<ns;s++){
        double velocity=(m->geometry.v_inner[s]+m->geometry.v_outer[s])/2e5;
        if(velocity<12100||velocity>12900)continue;
        for(int z=14;z<=16;z+=2){
            int low=-1,high=-1;
            for(int ip=0;ip<a->n_ion_pops;ip++)if(a->ion_pop_Z[ip]==z){
                if(a->ion_pop_stage[ip]==1)low=ip;
                if(a->ion_pop_stage[ip]==2)high=ip;
            }
            REQUIRE(low>=0&&high>=0);
            double nlow=ions[(size_t)low*ns+s],nhigh=ions[(size_t)high*ns+s];
            REQUIRE(nlow>0&&nhigh>0);
            long double photo=0,recomb=0,coll=0,three=0;int used=0;
            for(int gl=a->level_offset[low];gl<a->level_offset[low+1];gl++){
                int fl=n->global_to_nlte_level[gl];REQUIRE(fl>=0);
                for(int r=channels->offset[gl];r<channels->offset[gl+1];r++){
                    int upper=channels->upper[r];REQUIRE(upper>=0&&upper<a->n_levels);
                    REQUIRE(a->level_Z[upper]==z&&a->level_ion[upper]==2);
                    int uf=n->global_to_nlte_level[upper];REQUIRE(uf>=0);
                    double up,down,cu,cd;
                    REQUIRE(!nlte_bf_channel_rates(&diagnostic,a,gl,r,s,temperature[s],ne[s],&up,&down,&cu,&cd));
                    double nl=pop[(size_t)fl*ns+s],nu=pop[(size_t)uf*ns+s];
                    photo+=(long double)nl*up;recomb+=(long double)nu*down;
                    coll+=(long double)nl*cu;three+=(long double)nu*cd;used++;
                }
            }
            long double forward=photo+coll,reverse=recomb+three;
            REQUIRE(used>0&&forward>0&&reverse>0&&isfinite((double)(forward+reverse)));
            if(getenv("CMF_TEST_BF_LTE_LEVELS")&&getenv("LUMINA_NLTE_BF_JEQB")){
                double chi=NAN;
                for(int k=0;k<a->n_ionization;k++)
                    if(a->ioniz_Z[k]==z&&a->ioniz_ion[k]==1)chi=a->ioniz_energy_eV[k];
                REQUIRE(isfinite(chi)&&chi>0&&ne[s]>0);
                double saha=2*pow(2*M_PI_VAL*M_ELECTRON*K_BOLTZMANN*temperature[s]/
                    (H_PLANCK*H_PLANCK),1.5)/ne[s]*
                    a->partition_functions[(size_t)high*ns+s]/a->partition_functions[(size_t)low*ns+s]*
                    exp(-chi*EV_TO_ERG/(K_BOLTZMANN*temperature[s]));
                long double equilibrium=(forward/nlow)/(reverse/nhigh);
                REQUIRE(saha>0&&isfinite(saha));
                printf("TOY_BF_LTE_BALANCE field=%d shell=%d Z=%d thermal_III_II=%.17Lg saha_III_II=%.17g relative=%.17Lg\n",
                    field_iteration,s,z,equilibrium,saha,fabsl(equilibrium/saha-1));
            }
            REQUIRE(fprintf(fp,"%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg,%.17Lg\n",
                field_iteration,population_iteration,s,z,velocity,temperature[s],ne[s],nhigh/nlow,used,
                photo/nlow,recomb/nhigh,coll/nlow,three/nhigh,forward,reverse,
                (forward-reverse)/fmaxl(forward,reverse))>0);
            rows++;
        }
    }
    REQUIRE(rows>0&&!n->radiation_field.field.generation.computed_generation);
    REQUIRE(n->radfield_view_status==RADIATION_FIELD_VIEW_DISABLED);
    printf("TOY_BF_ITERATE_DIAGNOSTIC field_iteration=%d population_iteration=%d rows=%d production_owner_unchanged=1 transport_solved=0 NLTE_solved=0\n",field_iteration,population_iteration,rows);
    rc=0;
done:
    radiation_field_owner_free(&owner);free(edges);free(J);free(valid);return rc;
}

int main(int argc,char **argv){
    if(argc!=7)return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};double *old_J=NULL,*private_pop=NULL;FILE *fp=NULL;int rc=-1,new_iteration=-1,old_iteration=-1;
    REQUIRE(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],atoi(argv[3]),mass));
    REQUIRE(!cmf_toy_initial_iterate_read(&m,argv[4],argv[5],&new_iteration));
    const double *pop=m.nlte.nlte_level_populations,*ions=m.atom.ion_number_density;
    const double *ne=m.plasma.n_electron,*temperature=m.plasma.T_e;
    old_J=previous_field(&m,argv[6],argv[5],&old_iteration);REQUIRE(old_J);
    REQUIRE(old_iteration+1==new_iteration);
    if(getenv("CMF_TEST_BF_LTE_LEVELS")){
        private_pop=lte_levels(&m);REQUIRE(private_pop);pop=private_pop;
    }
    fp=fopen("iterate_bf_balance.csv","wx");REQUIRE(fp);
    REQUIRE(fprintf(fp,"field_iteration,population_iteration,shell,Z,velocity_km_s,T_K,ne_cm3,III_II,channels,photo_per_II_s,recomb_per_III_s,collision_per_II_s,threebody_per_III_s,forward_cm3_s,reverse_cm3_s,signed_imbalance\n")>0);
    REQUIRE(!evaluate(&m,m.opacity.jnu_fine,pop,ions,ne,temperature,new_iteration,new_iteration,fp));
    REQUIRE(!evaluate(&m,old_J,pop,ions,ne,temperature,old_iteration,new_iteration,fp));
    rc=0;
done:
    if(fp&&fclose(fp))rc=-1;
    cmf_toy_model_free(&m);free(old_J);free(private_pop);return rc?1:0;
}
