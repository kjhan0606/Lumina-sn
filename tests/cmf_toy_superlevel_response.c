/* 같은 자체 J/T/ne에서 Si/S 준위 묶음의 조건부 인구 응답.
 * 생산 입력 불변. 비열은 두 경우 모두 제외하며 새 RTE/전하해/물리이력은 없다. */
#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"SL_RESPONSE_FAIL line=%d %s\n",__LINE__,#x);goto done;} } while(0)

/* 진단 전용 선택. 기본 Si/S 시험과 생산 원자망은 바꾸지 않는다. */
static int response_selected_Z(int z){
#ifdef LUMINA_RESPONSE_TARGET_Z
    return z==LUMINA_RESPONSE_TARGET_Z;
#else
    return z==14||z==16;
#endif
}

/* 모든 투영 인덱스를 함께 갱신한다. 다른 원소의 묶음은 보존한다. */
static int separate_silicon_sulfur(CMFToyModel *m){
    NLTEConfig *n=&m->nlte;AtomicData *a=&m->atom;
    int old[NLTE_MAX_IONS+1];
    memcpy(old,n->nlte_ion_super_offset,sizeof(old));
    int *anchors=malloc((size_t)n->n_nlte_levels_total*sizeof(*anchors));
    if(!anchors)return -1;
    for(int i=0;i<n->n_nlte_levels_total;i++)anchors[i]=-1;
    int next=0;
    for(int ip=0;ip<n->n_nlte_ions;ip++){
        int first=n->nlte_ion_level_offset[ip],end=n->nlte_ion_level_offset[ip+1];
        int split=response_selected_Z(n->nlte_Z[ip]);
        n->nlte_ion_super_offset[ip]=next;
        for(int fl=first;fl<end;fl++){
            int gl=n->nlte_to_global_level[fl];
            int local=split?fl-first:n->fl_to_super[fl]-old[ip];
            if(local<0||local>=end-first){free(anchors);return -1;}
            n->fl_to_super[fl]=next+local;
            if(split)a->level_super[gl]=local;
            int prior=anchors[next+local];
            if(prior<0||a->level_energy_eV[gl]<a->level_energy_eV[prior])anchors[next+local]=gl;
        }
        next+=split?end-first:old[ip+1]-old[ip];
    }
    n->nlte_ion_super_offset[n->n_nlte_ions]=next;n->n_super_total=next;
    free(n->super_anchor_global);n->super_anchor_global=anchors;
    return nlte_precompute_within_sl_frac(n,a,&m->plasma,m->plasma.n_shells);
}

int main(int argc,char **argv){
    if(argc!=6&&argc!=9&&argc!=10)return 2;
    /* 선택적인 국소 주파수 응답. 저장장을 진단 메모리에서만 변경한다.
     * BB/BF가 같은 변경장을 소비하며, FULL 비교 대신 현행 묶음만 푼다. */
    const int spectral_probe=argc>=9;
    const char *probe_consumer=argc==10?argv[9]:"BOTH";
    if(strcmp(probe_consumer,"BOTH")&&strcmp(probe_consumer,"BB")&&strcmp(probe_consumer,"BF"))return 2;
    double energy_lo=0,energy_hi=0,field_factor=1;
    if(spectral_probe){
        energy_lo=strtod(argv[6],NULL);energy_hi=strtod(argv[7],NULL);
        field_factor=strtod(argv[8],NULL);
        if(!isfinite(energy_lo)||!isfinite(energy_hi)||!isfinite(field_factor)||
           energy_lo<0||energy_hi<=energy_lo||field_factor<=0)return 2;
    }
    setvbuf(stdout,NULL,_IOLBF,0);
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};RadiationFieldOwner owner={0};GammaDeposition gamma={0};
    double *nu=NULL,*width=NULL,*zero=NULL,*jbar=NULL,*se=NULL,*edges=NULL,*coarse=NULL;
    double *saved_pop=NULL,*saved_ion=NULL,*zero_heat=NULL,*saved_J=NULL;
    uint64_t *ids=NULL,*count=NULL;LineJbarValidityState *lv=NULL;
    RadiationFieldValidityState *valid=NULL;
    int rc=1,iteration=-1,s=atoi(argv[5]);
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    CHECK(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&iteration));
    NLTEConfig *n=&m.nlte;AtomicData *a=&m.atom;PlasmaState *p=&m.plasma;
    int ns=p->n_shells,nf=m.opacity.n_fine,nb=NLTE_N_FREQ_BINS;
#ifndef LUMINA_RESPONSE_EXPECTED_ITERATION
#define LUMINA_RESPONSE_EXPECTED_ITERATION 4
#endif
    CHECK(ns==113&&s>=0&&s<ns&&iteration==LUMINA_RESPONSE_EXPECTED_ITERATION&&p->initial_steady_nlte&&!p->physical_history);
    printf("SL_RESPONSE_INPUT completed_iterations=%d fine_bins=%d shell=%d\n",iteration,nf,s);
#ifdef LUMINA_RESPONSE_PLANCK_FINE
    /* 율의 JEQB 우회 없이 실제 Gaussian 평균/coarse 재분배를 통과시킨다. */
    CHECK(!getenv("LUMINA_NLTE_JEQB")&&!getenv("LUMINA_NLTE_BF_JEQB"));
#ifdef LUMINA_RESPONSE_PLANCK_FULL_DOMAIN
    /* 기존 저장점은 물질 초기값이다. 새 범위의 합성 Planck 시험장을
     * 명시적으로 만들며, 이를 새 RTE 해나 과거 저장장으로 표시하지 않는다. */
    double lo=m.opacity.nu_lo_fine;
    double hi=lo*exp(nf*m.opacity.dlognu_fine);
    for(int l=0;l<a->n_lines;l++) {
        double lower=a->line_nu[l]*(1-4*5e6/C_SPEED_OF_LIGHT)*exp(-.5*m.opacity.dlognu_fine);
        double upper=a->line_nu[l]*(1+4*5e6/C_SPEED_OF_LIGHT)*exp(.5*m.opacity.dlognu_fine);
        if(lower<lo)lo=lower;
        if(upper>hi)hi=upper;
    }
    nf=(int)ceil(log(hi/lo)/m.opacity.dlognu_fine);
    double *new_J=calloc((size_t)ns*nf,sizeof(double));
    double *new_nu=malloc((size_t)nf*sizeof(double));
    CHECK(new_J&&new_nu);
    for(int b=0;b<nf;b++)new_nu[b]=lo*exp((b+.5)*m.opacity.dlognu_fine);
    free(m.opacity.jnu_fine);free(m.opacity.nu_fine);
    m.opacity.jnu_fine=new_J;m.opacity.nu_fine=new_nu;
    m.opacity.n_fine=nf;m.opacity.nu_lo_fine=lo;
    printf("PLANCK_FULL_DOMAIN bins=%d lower_Hz=%.17g q_lines=%zu e_lines=%zu\n",nf,lo,m.qset.n_q,m.eset.n_q);
#endif
    for(int b=0;b<nf;b++){
        double f=m.opacity.nu_lo_fine*exp((b+.5)*m.opacity.dlognu_fine);
        double y=H_PLANCK*f/(K_BOLTZMANN*p->T_e[s]);
        m.opacity.jnu_fine[(size_t)s*nf+b]=2*H_PLANCK*f*f*f/(C_SPEED_OF_LIGHT*C_SPEED_OF_LIGHT)*exp(-y)/(-expm1(-y));
    }
    printf("PLANCK_PIPELINE shell=%d T=%.17g synthetic_fine_J=1 rate_JEQB_override=0 RTE_solved=0\n",s,p->T_e[s]);
#endif
    size_t np=(size_t)n->n_nlte_levels_total*ns,ni=(size_t)a->n_ion_pops*ns,nq=0;
    for(size_t q=0;q<m.qset.n_q;q++){
        int z=a->line_atomic_number[m.qset.line_id[q]];if(response_selected_Z(z))nq++;
    }
    ids=malloc(nq*sizeof(*ids));jbar=calloc(nq*ns,sizeof(*jbar));se=calloc(nq*ns,sizeof(*se));
    count=calloc(nq*ns,sizeof(*count));lv=calloc(nq*ns,sizeof(*lv));
    nu=malloc((size_t)nf*sizeof(double));width=malloc((size_t)nf*sizeof(double));zero=calloc((size_t)nf,sizeof(double));
    edges=malloc((nb+1)*sizeof(double));coarse=malloc((size_t)ns*nb*sizeof(double));valid=malloc((size_t)ns*nb*sizeof(*valid));
    saved_pop=malloc(np*sizeof(double));saved_ion=malloc(ni*sizeof(double));
    CHECK(ids&&jbar&&se&&count&&lv&&nu&&width&&zero&&edges&&coarse&&valid&&saved_pop&&saved_ion);
    memcpy(saved_pop,n->nlte_level_populations,np*sizeof(double));memcpy(saved_ion,a->ion_number_density,ni*sizeof(double));
    for(int b=0;b<nf;b++){nu[b]=m.opacity.nu_lo_fine*exp((b+.5)*m.opacity.dlognu_fine);width[b]=nu[b]*m.opacity.dlognu_fine;}
    if(spectral_probe){
        saved_J=malloc((size_t)nf*sizeof(*saved_J));CHECK(saved_J);
        memcpy(saved_J,m.opacity.jnu_fine+(size_t)s*nf,(size_t)nf*sizeof(*saved_J));
        int changed=0;
        for(int b=0;b<nf;b++){
            double energy=H_PLANCK*nu[b]/EV_TO_ERG;
            if(energy>=energy_lo&&energy<energy_hi){
                if(strcmp(probe_consumer,"BF"))m.opacity.jnu_fine[(size_t)s*nf+b]*=field_factor;
                changed++;
            }
        }
        CHECK(changed>0);
        printf("SPECTRAL_PROBE shell=%d energy_lo_eV=%.17g energy_hi_eV=%.17g factor=%.17g bins=%d consumer=%s frozen_T_ne=1 synthetic_J=1 production_input_changed=0\n",
               s,energy_lo,energy_hi,field_factor,changed,probe_consumer);
    }
    size_t q=0;
    for(size_t k=0;k<m.qset.n_q;k++){
        int line=m.qset.line_id[k],z=a->line_atomic_number[line];if(!response_selected_Z(z))continue;
        double v,error;LineJbarProfileReport report;
        CHECK(line_jbar_gaussian_discrete_shells(1,nf,nu,width,m.opacity.jnu_fine+(size_t)s*nf,zero,
              m.qset.line_nu[k],5e6,4,&v,&error,&report)==LINE_JBAR_PROFILE_OK);
        ids[q]=line;jbar[q*ns+s]=v;lv[q*ns+s]=v>0?LINE_JBAR_VALID:LINE_JBAR_EXACT_ZERO;q++;
    }
    CHECK(q==nq);
    if(spectral_probe&&!strcmp(probe_consumer,"BB"))
        memcpy(m.opacity.jnu_fine+(size_t)s*nf,saved_J,(size_t)nf*sizeof(*saved_J));
    if(spectral_probe&&!strcmp(probe_consumer,"BF")){
        for(int b=0;b<nf;b++){
            double energy=H_PLANCK*nu[b]/EV_TO_ERG;
            if(energy>=energy_lo&&energy<energy_hi)m.opacity.jnu_fine[(size_t)s*nf+b]*=field_factor;
        }
    }
    n->line_view=(LineJbarView){.n_lines=nq,.cache_n_lines=nq,.n_shells=(size_t)ns,.line_id=ids,
        .jbar=jbar,.validity=lv,.count=count,.se=se,.generation=1,.statistic_kind=RADIATION_FIELD_DETERMINISTIC};
    n->line_view_status=LINE_JBAR_VIEW_OK;
    CMFGENState source={.n_shells=ns,.n_bins=nf,.J=m.opacity.jnu_fine,.nu_min=m.opacity.nu_lo_fine,.d_log_nu=m.opacity.dlognu_fine};
    source.nu_max=source.nu_min*exp(nf*source.d_log_nu);
    CMFGENState target={.n_shells=ns,.n_bins=nb,.J=coarse,.nu_min=NLTE_NU_MIN,.nu_max=NLTE_NU_MAX,
        .d_log_nu=log(NLTE_NU_MAX/NLTE_NU_MIN)/nb};
    CHECK(!cmfgen_rebin_log_field(&source,&target));
    for(int b=0;b<=nb;b++)edges[b]=NLTE_NU_MIN*exp(b*target.d_log_nu);
    edges[0]=NLTE_NU_MIN;edges[nb]=NLTE_NU_MAX;
    for(size_t k=0;k<(size_t)ns*nb;k++)valid[k]=coarse[k]>0?RADIATION_FIELD_VALID:RADIATION_FIELD_EXACT_ZERO;
    CHECK(!radiation_field_owner_init(&owner,ns));
    RadiationFieldCommitRequest request={.provenance_kind=RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY,
        .producer="PURE_CMFGEN_COMOVING_CONSERVATIVE_REBIN",.generation=1,.epoch=m.geometry.time_explosion,
        .n_shells=(size_t)ns,.v_inner=m.geometry.v_inner,.v_outer=m.geometry.v_outer,.source_n_bins=(size_t)nb,
        .source_frequency_bin_edges=edges,.source_J_nu=coarse,.source_validity=valid,.statistic_kind=RADIATION_FIELD_DETERMINISTIC};
    CHECK(!radiation_field_commit(&owner,&request));
    CHECK(radiation_field_read_view(&owner,request.epoch,ns,1,&n->radfield_view)==RADIATION_FIELD_VIEW_OK);
    n->radfield_view_status=RADIATION_FIELD_VIEW_OK;n->solve_effect_policy_explicit=1;n->solve_effects_allowed=0;
    /* NULL은 무침적이 아니라 미게시 입력이다. 양쪽에 같은 명시적 영입력을 준다. */
    zero_heat=calloc((size_t)ns,sizeof(*zero_heat));CHECK(zero_heat);
    gamma_deposition_init(&gamma,ns);
#ifdef LUMINA_RESPONSE_NONTHERMAL
    CHECK(!response_nonthermal_prepare(&m,&gamma,s));
#else
    CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,request.epoch,NULL,NULL,NULL,zero_heat));
    puts("SL_RESPONSE_COUNTERFACTUAL synthetic_zero_nonthermal=1 production_input_changed=0");
#endif
    int mode_first=0,mode_end=spectral_probe?1:2;
#ifdef LUMINA_RESPONSE_GROUPED_ONLY
    CHECK(!spectral_probe);mode_end=1;
#endif
#ifdef LUMINA_RESPONSE_FULL_ONLY
    /* 원자모형 진단용 개별 준위 행렬만 저장한다. 기본 시험 동작은 불변이다. */
    CHECK(!spectral_probe);mode_first=1;mode_end=2;
#endif
    for(int mode=mode_first;mode<mode_end;mode++){
        if(mode)CHECK(!separate_silicon_sulfur(&m));
        for(int e=0;e<a->n_elements;e++){
            int z=a->element_Z[e];if(!response_selected_Z(z))continue;
            memcpy(n->nlte_level_populations,saved_pop,np*sizeof(double));memcpy(a->ion_number_density,saved_ion,ni*sizeof(double));
            double total=a->abundances[(size_t)e*ns+s]*p->rho[s]/(a->element_mass_amu[e]*AMU);
            printf("SL_RESPONSE_BEGIN mode=%s shell=%d Z=%d super_total=%d\n",mode?"FULL":"GROUPED",s,z,n->n_super_total);
#ifdef LUMINA_LINE_RESPONSE_CAPTURE
            CHECK(!line_response_capture_prepare(&m,e,s));
#endif
            CHECK(nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,request.epoch,&gamma)==POP_OK);
            long double sum=0;double low=NAN,high=NAN;
            for(int ip=a->elem_ion_offset[e];ip<a->elem_ion_offset[e+1];ip++){
                double value=a->ion_number_density[(size_t)ip*ns+s];CHECK(isfinite(value)&&value>=0);sum+=value;
                if(a->ion_pop_stage[ip]==1)low=value;
                if(a->ion_pop_stage[ip]==2)high=value;
                printf("SL_RESPONSE_ION mode=%s shell=%d Z=%d charge=%d fraction=%.17g before_fraction=%.17g\n",
                    mode?"FULL":"GROUPED",s,z,a->ion_pop_stage[ip],value/total,saved_ion[(size_t)ip*ns+s]/total);
            }
            CHECK(total>0&&fabsl(sum/total-1)<1e-10L&&low>0&&high>=0);
            printf("SL_RESPONSE_RESULT mode=%s shell=%d Z=%d III_II=%.17g conservation=%.17Lg T=%.17g ne=%.17g\n",
                   mode?"FULL":"GROUPED",s,z,high/low,fabsl(sum/total-1),p->T_e[s],p->n_electron[s]);
        }
    }
    puts("SL_RESPONSE_DONE fixed_J_T_ne=1 "
#ifdef LUMINA_RESPONSE_NONTHERMAL
         "nonthermal=RECORDED_MC_RESOLVED_SUBSET "
#else
         "nonthermal=0 "
#endif
         "RTE_solved=0 charge_solved=0 physical_history=0 goal_judged=0");rc=0;
done:
#ifdef LUMINA_RESPONSE_NONTHERMAL
    response_nonthermal_free();
#endif
    gamma_deposition_free(&gamma);radiation_field_owner_free(&owner);cmf_toy_model_free(&m);
    free(nu);free(width);free(zero);free(jbar);free(se);free(ids);free(count);free(lv);
    free(edges);free(coarse);free(valid);free(saved_pop);free(saved_ion);free(zero_heat);free(saved_J);return rc;
}
