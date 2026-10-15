/* 실제 원자로 검색 최적화 전후 정상 인구를 비교한다.
 * 명시적 열평형 falsifier이며 SN 복사장/초기해/시간 이력이 아니다. */
#include "cmf_toy_model.h"
#include "cmf_nonthermal_rates.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TOY_RATES_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
static double seconds(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;
}
static double relative_difference(const double *a,const double *b,size_t count){
    double worst=0;
    for(size_t i=0;i<count;i++){
        if(!isfinite(a[i])||!isfinite(b[i])||a[i]<0||b[i]<0)return INFINITY;
        double den=fmax(a[i],b[i]);if(den)worst=fmax(worst,fabs(a[i]-b[i])/den);
    }
    return worst;
}
static void remove_partners(CMFToyModel *m,int Z,int shell){
    if(!getenv("CMF_TEST_ISOLATE_CE")||!atoi(getenv("CMF_TEST_ISOLATE_CE")))return;
    /* 시험 대상 밖 원소만 영 인구로 둔다. 고정 ne인 성분 분리 반사실이며
     * 생산 조성/전하 폐합/시간 상태가 아니다. 생산 CE 코드는 바꾸지 않는다. */
    int ns=m->plasma.n_shells;
    for(int ip=0;ip<m->atom.n_ion_pops;ip++)if(m->atom.ion_pop_Z[ip]!=Z){
        m->atom.ion_number_density[(size_t)ip*ns+shell]=0;
        for(int gl=m->atom.level_offset[ip];gl<m->atom.level_offset[ip+1];gl++){
            int fl=m->nlte.global_to_nlte_level[gl];
            m->nlte.nlte_level_populations[(size_t)fl*ns+shell]=0;
        }
    }
}
/* 실제 원자 행렬/합성 Planck 장의 캐시 대조. NT 계수는 새로 바꾸는 합성값이며
 * SF/실제 SN 이온분포 검증으로 표시하지 않는다. */
static int matrix_cache_check(CMFToyModel *m,GammaDeposition *gamma,int e,int s,
    const double *seed_pop,const double *seed_ion){
    NLTEConfig *n=&m->nlte;AtomicData *a=&m->atom;PlasmaState *p=&m->plasma;
    size_t np=(size_t)n->n_nlte_levels_total*p->n_shells,ni=(size_t)a->n_ion_pops*p->n_shells;
    double *cached_pop=malloc(np*sizeof(double)),*cached_ion=malloc(ni*sizeof(double));
    double *nt_value=calloc((size_t)p->n_shells,sizeof(double));
    struct NLTEElementBaseCache *cache=NULL;int rc=0;
    int ip=a->elem_ion_offset[e];
    CMFNonthermalTransition transition={a->level_offset[ip],a->level_offset[ip+1],-1};
    CMFNonthermalRates rates={.n_shells=p->n_shells,.n_levels=a->n_levels,
        .n_ions=a->n_ion_pops,.n_transitions=1,.epoch_s=gamma->epoch,
        .deposition_generation=gamma->generation,.transitions=&transition,.rate_per_particle=nt_value};
    const CMFNonthermalRates *original_rates=gamma->nonthermal_rates;
    double original_ne=p->n_electron[s],cached_seconds=0,plain_seconds=0;
    uint64_t generation=n->line_view.generation;
    CHECK(cached_pop&&cached_ion&&nt_value&&a->level_offset[ip+1]<a->level_offset[ip+2]);
    cache=nlte_element_base_cache_create(n,a,p,&m->opacity,gamma->epoch,gamma);CHECK(cache);
    gamma->nonthermal_rates=&rates;
    for(int trial=0;trial<6;trial++){
        p->n_electron[s]=original_ne*(trial==3?1.2:1);
        n->line_view.generation=generation+(trial==4);
        n->exclude_empirical_heavy_ce=trial==5?0:1;
        nt_value[s]=trial==0?0:(trial==1?1e4:1e3);
        size_t before=nlte_element_base_cache_hits(cache);
        memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
        memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
        double start=seconds();
        CHECK(nlte_element_initial_steady_solve_cached(n,a,p,&m->opacity,e,s,gamma->epoch,gamma,cache)==POP_OK);
        double cached_s=seconds()-start;
        memcpy(cached_pop,n->nlte_level_populations,np*sizeof(double));
        memcpy(cached_ion,a->ion_number_density,ni*sizeof(double));
        CHECK(nlte_element_base_cache_hits(cache)-before==(size_t)(trial==1||trial==2));
        memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
        memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
        start=seconds();
        CHECK(nlte_element_initial_steady_solve(n,a,p,&m->opacity,e,s,gamma->epoch,gamma)==POP_OK);
        double plain_s=seconds()-start;
        CHECK(!memcmp(cached_pop,n->nlte_level_populations,np*sizeof(double))&&
              !memcmp(cached_ion,a->ion_number_density,ni*sizeof(double)));
        if(trial==1||trial==2){cached_seconds+=cached_s;plain_seconds+=plain_s;}
        printf("TOY_MATRIX_CACHE shell=%d Z=%d trial=%d byte_exact=1 hit=%zu cached_s=%.9g plain_s=%.9g\n",
            s,a->element_Z[e],trial,nlte_element_base_cache_hits(cache)-before,cached_s,plain_s);
    }
    n->line_view.generation=generation;n->exclude_empirical_heavy_ce=1;p->n_electron[s]=original_ne;
    memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
    memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
    nt_value[s]=-1;
    CHECK(nlte_element_initial_steady_solve_cached(n,a,p,&m->opacity,e,s,gamma->epoch,gamma,cache)!=POP_OK);
    CHECK(!memcmp(seed_pop,n->nlte_level_populations,np*sizeof(double))&&
          !memcmp(seed_ion,a->ion_number_density,ni*sizeof(double)));
    n->line_view_status=LINE_JBAR_VIEW_DISABLED;
    CHECK(!nlte_element_base_cache_create(n,a,p,&m->opacity,gamma->epoch,gamma));
    n->line_view_status=LINE_JBAR_VIEW_OK;
    printf("TOY_MATRIX_CACHE_PASS shell=%d Z=%d warm_cached_s=%.9g warm_plain_s=%.9g changed_NT_applied=1 ne_invalidated=1 field_invalidated=1 CE_not_cached=1 invalid_NT_preserved=1\n",
        s,a->element_Z[e],cached_seconds,plain_seconds);
done:
    n->line_view_status=LINE_JBAR_VIEW_OK;n->line_view.generation=generation;
    n->exclude_empirical_heavy_ce=1;p->n_electron[s]=original_ne;gamma->nonthermal_rates=original_rates;
    nlte_element_base_cache_free(cache);free(cached_pop);free(cached_ion);free(nt_value);return rc;
}
int main(int argc,char **argv){
    if(argc!=4)return 2;
    setvbuf(stdout,NULL,_IOLBF,0);
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};GammaDeposition gamma={0};int ns=atoi(argv[3]),rc=0;
    double *edges=NULL,*J=NULL,*heat=NULL,*seed_pop=NULL,*seed_ion=NULL;
    double *cached_pop=NULL,*cached_ion=NULL;
    RadiationFieldValidityState *valid=NULL;
    int *saved_global=NULL,*saved_lines=NULL;
    struct NLTEBFRateCache *rate_cache=NULL;
    CHECK(getenv("LUMINA_NLTE_JEQB")&&atoi(getenv("LUMINA_NLTE_JEQB"))==1);
    CHECK(getenv("LUMINA_NLTE_BF_JEQB")&&atoi(getenv("LUMINA_NLTE_BF_JEQB"))==1);
    CHECK(!cmf_toy_model_prepare(&m,argv[1],argv[2],ns,masses));
    NLTEConfig *n=&m.nlte;AtomicData *a=&m.atom;PlasmaState *p=&m.plasma;
    saved_global=n->nlte_line_global;saved_lines=n->nlte_ion_lines;
    CHECK(saved_global&&saved_lines&&a->bf_channels&&p->initial_steady_nlte&&!p->physical_history);
    CHECK(n->exclude_empirical_heavy_ce==1);
    CHECK(n->channel_owned_recombination==1);
    const int profile_cached=getenv("CMF_TEST_PROFILE_CACHED")&&atoi(getenv("CMF_TEST_PROFILE_CACHED"));
    if(profile_cached)puts("TOY_RATES_PROFILE current_cached_path_repeated=1 old_search_path_tested=0");
    const int cache_test=getenv("CMF_TEST_RATE_CACHE")&&atoi(getenv("CMF_TEST_RATE_CACHE"));
    const int matrix_test=getenv("CMF_TEST_MATRIX_CACHE")&&atoi(getenv("CMF_TEST_MATRIX_CACHE"));
    CMFNonthermalRates empty_rates={0};
    const int gth_parallel=getenv("CMF_TEST_GTH_PARALLEL")&&atoi(getenv("CMF_TEST_GTH_PARALLEL"));
    if(gth_parallel){omp_set_dynamic(0);puts("TOY_GTH_PARALLEL same_rates=1 threads=1/2");}
    if(cache_test){rate_cache=nlte_bf_rate_cache_create(a,(size_t)ns);CHECK(rate_cache);}
    n->solve_effect_policy_explicit=1;n->solve_effects_allowed=0;
    size_t np=(size_t)n->n_nlte_levels_total*ns,ni=(size_t)a->n_ion_pops*ns;
    size_t nb=(size_t)a->bf_channels->n_bins;
    seed_pop=malloc(np*sizeof(double));cached_pop=malloc(np*sizeof(double));
    seed_ion=malloc(ni*sizeof(double));cached_ion=malloc(ni*sizeof(double));
    edges=malloc((nb+1)*sizeof(double));J=calloc((size_t)ns*nb,sizeof(double));
    valid=malloc((size_t)ns*nb*sizeof(*valid));heat=calloc((size_t)ns,sizeof(double));
    CHECK(seed_pop&&cached_pop&&seed_ion&&cached_ion&&edges&&J&&valid&&heat);
    memcpy(seed_pop,n->nlte_level_populations,np*sizeof(double));
    memcpy(seed_ion,a->ion_number_density,ni*sizeof(double));
    double dlog=log(a->bf_channels->nu_max/a->bf_channels->nu_min)/nb;
    for(size_t b=0;b<=nb;b++)edges[b]=a->bf_channels->nu_min*exp(b*dlog);
    for(int s=0;s<ns;s++)for(size_t b=0;b<nb;b++){
        double nu=sqrt(edges[b]*edges[b+1]),x=H_PLANCK*nu/(K_BOLTZMANN*p->T_e[s]);
        double em=exp(-x);
        J[(size_t)s*nb+b]=2*H_PLANCK*nu*nu*nu/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)*em/(-expm1(-x));
        valid[(size_t)s*nb+b]=J[(size_t)s*nb+b]>0?RADIATION_FIELD_VALID:RADIATION_FIELD_EXACT_ZERO;
    }
    /* 이 view는 시험 안에서만 존재한다. 생산 owner 게시/시점 저장은 하지 않는다.
     * BF falsifier가 bin 내 Planck 적분을 직접 계산하며 위 중앙값은 사용하지 않는다. */
    n->radfield_view=(RadiationFieldView){.n_shells=(size_t)ns,.n_bins=nb,
        .frequency_bin_edges=edges,.J_nu=J,.validity=valid,.generation=1};
    n->radfield_view_status=RADIATION_FIELD_VIEW_OK;
    gamma_deposition_init(&gamma,ns);
    CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,
        m.geometry.time_explosion,NULL,NULL,NULL,heat));
    if(matrix_test){
        CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_NONLOCAL_MC,
            m.geometry.time_explosion,NULL,NULL,NULL,heat));
        empty_rates=(CMFNonthermalRates){.n_shells=ns,.n_levels=a->n_levels,.n_ions=a->n_ion_pops,
            .epoch_s=gamma.epoch,.deposition_generation=gamma.generation};
        gamma.nonthermal_rates=&empty_rates;
        /* 이 모드는 명시적 JEQB이므로 line view 수치는 읽지 않는다. */
        n->line_view=(LineJbarView){.n_shells=(size_t)ns,.jbar=J,.generation=1};
        n->line_view_status=LINE_JBAR_VIEW_OK;
    }
    printf("TOY_RATES_CONTRACT DR_multiplier=%s CE_partners_removed=%s exclude_empirical_heavy_ce=%d channel_owned_recombination=%d fixed_ne=1 transport=0\n",
        getenv("LUMINA_DR_BOOST")?getenv("LUMINA_DR_BOOST"):"1",
        getenv("CMF_TEST_ISOLATE_CE")?getenv("CMF_TEST_ISOLATE_CE"):"0",n->exclude_empirical_heavy_ce,n->channel_owned_recombination);
    const int samples[2]={0,ns/3};int tested=0,empty=0;
    for(int sample=0;sample<2;sample++)for(int e=0;e<a->n_elements;e++){
        int s=samples[sample],Z=a->element_Z[e];
        double total=a->abundances[(size_t)e*ns+s]*p->rho[s]/(a->element_mass_amu[e]*AMU);
        memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
        memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
        n->bf_rate_cache=rate_cache;
        if(total==0){
            CHECK(nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma)==POP_OK);
            CHECK(!memcmp(seed_pop,n->nlte_level_populations,np*sizeof(double))&&
                !memcmp(seed_ion,a->ion_number_density,ni*sizeof(double)));
            empty++;continue;
        }
        remove_partners(&m,Z,s);
        printf("TOY_RATES_BEGIN shell=%d Z=%d T=%.17g ne=%.17g nuclei=%.17g\n",s,Z,p->T_e[s],p->n_electron[s],total);
        double start=seconds();
        if(gth_parallel)omp_set_num_threads(1);
        PopulationStatus cached=nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma);
        double cached_s=seconds()-start;
        printf("TOY_RATES_CACHED shell=%d Z=%d status=%d wall_s=%.9g\n",s,Z,cached,cached_s);
        CHECK(cached==POP_OK);
        memcpy(cached_pop,n->nlte_level_populations,np*sizeof(double));
        memcpy(cached_ion,a->ion_number_density,ni*sizeof(double));
        memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
        memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
        remove_partners(&m,Z,s);
        n->bf_rate_cache=NULL;
        if(!profile_cached&&!cache_test){n->nlte_line_global=NULL;n->nlte_ion_lines=NULL;}
        start=seconds();
        if(gth_parallel)omp_set_num_threads(2);
        PopulationStatus searched=nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma);
        double searched_s=seconds()-start;
        n->nlte_line_global=saved_global;n->nlte_ion_lines=saved_lines;
        printf("TOY_RATES_SEARCHED shell=%d Z=%d status=%d wall_s=%.9g\n",s,Z,searched,searched_s);
        CHECK(searched==POP_OK);
        double difference=fmax(relative_difference(cached_pop,n->nlte_level_populations,np),
            relative_difference(cached_ion,a->ion_number_density,ni));
        if(cache_test){CHECK(difference<1e-10);}
        else{CHECK(!memcmp(cached_pop,n->nlte_level_populations,np*sizeof(double)));
             CHECK(!memcmp(cached_ion,a->ion_number_density,ni*sizeof(double)));}
        if(cache_test){
            if(gth_parallel){
                memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
                memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
                n->bf_rate_cache=rate_cache;omp_set_num_threads(1);start=seconds();
                CHECK(nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma)==POP_OK);
                printf("TOY_GTH_WARM1 shell=%d Z=%d seconds=%.9g\n",s,Z,seconds()-start);
                CHECK(!memcmp(cached_pop,n->nlte_level_populations,np*sizeof(double))&&
                    !memcmp(cached_ion,a->ion_number_density,ni*sizeof(double)));
                omp_set_num_threads(2);
            }
            memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
            memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
            n->bf_rate_cache=rate_cache;start=seconds();
            CHECK(nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma)==POP_OK);
            double warm_s=seconds()-start;
            CHECK(!memcmp(cached_pop,n->nlte_level_populations,np*sizeof(double))&&
                !memcmp(cached_ion,a->ion_number_density,ni*sizeof(double)));
            printf("TOY_RATE_CACHE shell=%d Z=%d max_relative=%.17g uncached_wall_s=%.9g warm_wall_s=%.9g warm_repeat_exact=1\n",s,Z,difference,searched_s,warm_s);
        }
        double sum=0,ion_l1=0,level_l1=0;
        for(int ip=a->elem_ion_offset[e];ip<a->elem_ion_offset[e+1];ip++){
            size_t cell=(size_t)ip*ns+s;double value=cached_ion[cell];
            CHECK(isfinite(value)&&value>=0);sum+=value;ion_l1+=fabs(value-seed_ion[cell]);
            printf("TOY_RATES_ION shell=%d Z=%d stage=%d saha_fraction=%.17g steady_fraction=%.17g\n",
                s,Z,a->ion_pop_stage[ip],seed_ion[cell]/total,value/total);
            for(int gl=a->level_offset[ip];gl<a->level_offset[ip+1];gl++){
                int fl=n->global_to_nlte_level[gl];CHECK(fl>=0);
                size_t k=(size_t)fl*ns+s;
                CHECK(isfinite(cached_pop[k])&&cached_pop[k]>=0);
                level_l1+=fabs(cached_pop[k]-seed_pop[k]);
            }
        }
        CHECK(fabs(sum/total-1)<1e-10);
        printf("TOY_RATES_PARITY shell=%d Z=%d exact_all_populations=%d cached_wall_s=%.9g searched_wall_s=%.9g ion_L1_per_nucleus=%.17g level_L1_per_nucleus=%.17g\n",
            s,Z,!cache_test,cached_s,searched_s,ion_l1/total,level_l1/total);
        if(matrix_test)CHECK(!matrix_cache_check(&m,&gamma,e,s,seed_pop,seed_ion));
        tested++;
    }
    CHECK(tested==6);
    CHECK(empty==6);puts("TOY_EMPTY_ELEMENT_PASS cases=6 exact_population_preserved=1 steady_only=1");
    puts("TOY_RATES_PARITY_PASS elements=6 synthetic_radiation=PLANCK_FALSIFIER actual_atom=1 SN_initial=0 SN_comparison=0");
done:
    /* 실패 경로에서도 소유자가 cache를 해제할 수 있도록 원래 포인터를 복원한다. */
    if(saved_global)m.nlte.nlte_line_global=saved_global;
    if(saved_lines)m.nlte.nlte_ion_lines=saved_lines;
    m.nlte.bf_rate_cache=NULL;nlte_bf_rate_cache_free(rate_cache);
    memset(&m.nlte.radfield_view,0,sizeof(m.nlte.radfield_view));
    m.nlte.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    memset(&m.nlte.line_view,0,sizeof(m.nlte.line_view));
    m.nlte.line_view_status=LINE_JBAR_VIEW_DISABLED;
    gamma_deposition_free(&gamma);cmf_toy_model_free(&m);
    free(edges);free(J);free(valid);free(heat);free(seed_pop);free(seed_ion);free(cached_pop);free(cached_ion);
    return rc;
}
