/* 실제 원자로 검색 최적화 전후 정상 인구를 비교한다.
 * 명시적 열평형 falsifier이며 SN 복사장/초기해/시간 이력이 아니다. */
#include "cmf_toy_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TOY_RATES_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
static double seconds(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;
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
    CHECK(getenv("LUMINA_NLTE_JEQB")&&atoi(getenv("LUMINA_NLTE_JEQB"))==1);
    CHECK(getenv("LUMINA_NLTE_BF_JEQB")&&atoi(getenv("LUMINA_NLTE_BF_JEQB"))==1);
    CHECK(!cmf_toy_model_prepare(&m,argv[1],argv[2],ns,masses));
    NLTEConfig *n=&m.nlte;AtomicData *a=&m.atom;PlasmaState *p=&m.plasma;
    saved_global=n->nlte_line_global;saved_lines=n->nlte_ion_lines;
    CHECK(saved_global&&saved_lines&&a->bf_channels&&p->initial_steady_nlte&&!p->physical_history);
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
    const int samples[2]={0,ns/3};int tested=0;
    for(int sample=0;sample<2;sample++)for(int e=0;e<a->n_elements;e++){
        int s=samples[sample],Z=a->element_Z[e];
        double total=a->abundances[(size_t)e*ns+s]*p->rho[s]/(a->element_mass_amu[e]*AMU);
        if(total==0)continue;
        memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
        memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
        printf("TOY_RATES_BEGIN shell=%d Z=%d T=%.17g ne=%.17g nuclei=%.17g\n",s,Z,p->T_e[s],p->n_electron[s],total);
        double start=seconds();
        PopulationStatus cached=nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma);
        double cached_s=seconds()-start;
        printf("TOY_RATES_CACHED shell=%d Z=%d status=%d wall_s=%.9g\n",s,Z,cached,cached_s);
        CHECK(cached==POP_OK);
        memcpy(cached_pop,n->nlte_level_populations,np*sizeof(double));
        memcpy(cached_ion,a->ion_number_density,ni*sizeof(double));
        memcpy(n->nlte_level_populations,seed_pop,np*sizeof(double));
        memcpy(a->ion_number_density,seed_ion,ni*sizeof(double));
        n->nlte_line_global=NULL;n->nlte_ion_lines=NULL;
        start=seconds();
        PopulationStatus searched=nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,m.geometry.time_explosion,&gamma);
        double searched_s=seconds()-start;
        n->nlte_line_global=saved_global;n->nlte_ion_lines=saved_lines;
        printf("TOY_RATES_SEARCHED shell=%d Z=%d status=%d wall_s=%.9g\n",s,Z,searched,searched_s);
        CHECK(searched==POP_OK);
        CHECK(!memcmp(cached_pop,n->nlte_level_populations,np*sizeof(double)));
        CHECK(!memcmp(cached_ion,a->ion_number_density,ni*sizeof(double)));
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
        printf("TOY_RATES_PARITY shell=%d Z=%d exact_all_populations=1 cached_wall_s=%.9g searched_wall_s=%.9g ion_L1_per_nucleus=%.17g level_L1_per_nucleus=%.17g\n",
            s,Z,cached_s,searched_s,ion_l1/total,level_l1/total);
        tested++;
    }
    CHECK(tested==6);
    puts("TOY_RATES_PARITY_PASS elements=6 synthetic_radiation=PLANCK_FALSIFIER actual_atom=1 SN_initial=0 SN_comparison=0");
done:
    /* 실패 경로에서도 소유자가 cache를 해제할 수 있도록 원래 포인터를 복원한다. */
    if(saved_global)m.nlte.nlte_line_global=saved_global;
    if(saved_lines)m.nlte.nlte_ion_lines=saved_lines;
    memset(&m.nlte.radfield_view,0,sizeof(m.nlte.radfield_view));
    m.nlte.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    gamma_deposition_free(&gamma);cmf_toy_model_free(&m);
    free(edges);free(J);free(valid);free(heat);free(seed_pop);free(seed_ion);free(cached_pop);free(cached_ion);
    return rc;
}
