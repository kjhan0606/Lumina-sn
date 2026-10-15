/* 실제 저장 J/T/ne/비열 계수를 고정한 누락 충돌의 인구 응답.
 * 사설 진단 view이며 생산 owner, 장 수렴 인증, 물리 시간 이력을 만들지 않는다.
 * 비열 계수는 저장 인구로 한 번 계산한다. 수정 인구의 SF/전하/RTE 피드백은 미포함. */
#include "cmf_toy_model.h"
#include "lumina_cmfgen.h"
#include "cmf_nonthermal_binding.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"COLLISION_POP_FAIL line=%d %s\n",__LINE__,#x);goto done;}}while(0)

/* 기존 reader로 검증한 인접 반복 사본의 J만 읽는다. 전체 사본 hash는 runner가 확인. */
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

static int read_gamma(const char *path,int ns,double *v){
    FILE *fp=fopen(path,"r");if(!fp)return -1;char line[512];int rc=-1;
    if(!fgets(line,sizeof line,fp)||strcmp(line,"shell,gamma_erg_cm3_s,gamma_MCSE,positive_packets,positron_erg_cm3_s,total_erg_cm3_s\n"))goto done;
    for(int s=0;s<ns;s++){
        int index=-1,used=0;double a,b,c,d,e;
        if(!fgets(line,sizeof line,fp)||sscanf(line,"%d,%lf,%lf,%lf,%lf,%lf%n",&index,&a,&b,&c,&d,&e,&used)!=6||
           index!=s||strcmp(line+used,"\n")||!isfinite(a+b+c+d+e)||a<0||b<0||c<0||d<0||e<0||
           floor(c)!=c||((a+d)==0?e!=0:fabs(e/(a+d)-1)>1e-12))goto done;
        v[s]=e;
    }
    if(fgetc(fp)!=EOF||ferror(fp))goto done;rc=0;
done:
    fclose(fp);return rc;
}

int main(int argc,char **argv){
    if(argc!=9)return 2;
    setvbuf(stdout,NULL,_IOLBF,0);
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};GammaDeposition gamma={0};RadiationFieldOwner owner={0};
    CMFNonthermalAtomicData raw={0};CMFNonthermalExcitations exc={0};
    CMFNonthermalResolvedModel bound={0};CMFNonthermalRates one={0},rates={0};
    double *fine=NULL,*nu=NULL,*dnu=NULL,*zero=NULL,*jbar=NULL,*se=NULL,*edges=NULL,*coarse=NULL;
    double *saved_pop=NULL,*saved_ion=NULL,*one_pop=NULL,*heat=NULL;
    uint64_t *ids=NULL,*count=NULL;LineJbarValidityState *line_valid=NULL;
    RadiationFieldValidityState *valid=NULL;FILE *ions=NULL,*levels=NULL;
    int rc=1,iteration=-1,field_iteration=-1,s=43;
    CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    CHECK(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&iteration));
    NLTEConfig *n=&m.nlte;AtomicData *a=&m.atom;PlasmaState *p=&m.plasma;
    int ns=p->n_shells,nf=m.opacity.n_fine,nb=NLTE_N_FREQ_BINS;
    CHECK(ns==113&&p->initial_steady_nlte&&!p->physical_history);
    fine=previous_field(&m,argv[5],argv[4],&field_iteration);CHECK(fine&&field_iteration+1==iteration);
    size_t np=(size_t)n->n_nlte_levels_total*ns,ni=(size_t)a->n_ion_pops*ns,nq=0;
    for(size_t q=0;q<m.qset.n_q;q++){
        int z=a->line_atomic_number[m.qset.line_id[q]];if(z==14||z==16||z==20||z==27)nq++;
    }
    ids=malloc(nq*sizeof(*ids));jbar=calloc(nq*ns,sizeof(*jbar));se=calloc(nq*ns,sizeof(*se));
    count=calloc(nq*ns,sizeof(*count));line_valid=calloc(nq*ns,sizeof(*line_valid));
    nu=malloc((size_t)nf*sizeof(double));dnu=malloc((size_t)nf*sizeof(double));zero=calloc((size_t)nf,sizeof(double));
    edges=malloc((nb+1)*sizeof(double));coarse=malloc((size_t)ns*nb*sizeof(double));valid=malloc((size_t)ns*nb*sizeof(*valid));
    saved_pop=malloc(np*sizeof(double));saved_ion=malloc(ni*sizeof(double));
    one_pop=malloc((size_t)n->n_nlte_levels_total*sizeof(double));heat=malloc((size_t)ns*sizeof(double));
    CHECK(ids&&jbar&&se&&count&&line_valid&&nu&&dnu&&zero&&edges&&coarse&&valid&&saved_pop&&saved_ion&&one_pop&&heat);
    memcpy(saved_pop,n->nlte_level_populations,np*sizeof(double));memcpy(saved_ion,a->ion_number_density,ni*sizeof(double));
    for(int b=0;b<nf;b++){nu[b]=m.opacity.nu_lo_fine*exp((b+.5)*m.opacity.dlognu_fine);dnu[b]=nu[b]*m.opacity.dlognu_fine;}
    size_t q=0;
    for(size_t k=0;k<m.qset.n_q;k++){
        int line=m.qset.line_id[k],z=a->line_atomic_number[line];if(z!=14&&z!=16&&z!=20&&z!=27)continue;
        double v,error;LineJbarProfileReport report;
        CHECK(line_jbar_gaussian_discrete_shells(1,nf,nu,dnu,fine+(size_t)s*nf,zero,
              m.qset.line_nu[k],5e6,4,&v,&error,&report)==LINE_JBAR_PROFILE_OK);
        ids[q]=line;jbar[q*ns+s]=v;line_valid[q*ns+s]=v>0?LINE_JBAR_VALID:LINE_JBAR_EXACT_ZERO;q++;
    }
    CHECK(q==nq);
    n->line_view=(LineJbarView){.n_lines=nq,.cache_n_lines=nq,.n_shells=(size_t)ns,.line_id=ids,
        .jbar=jbar,.validity=line_valid,.count=count,.se=se,.generation=1,.statistic_kind=RADIATION_FIELD_DETERMINISTIC};
    n->line_view_status=LINE_JBAR_VIEW_OK;
    CMFGENState source={.n_shells=ns,.n_bins=nf,.J=fine,.nu_min=m.opacity.nu_lo_fine,.d_log_nu=m.opacity.dlognu_fine};
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
    CHECK(!read_gamma(argv[8],ns,heat));gamma_deposition_init(&gamma,ns);
    CHECK(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_NONLOCAL_MC,request.epoch,NULL,NULL,NULL,heat));
    double energy[256];for(int k=0;k<256;k++)energy[k]=exp((double)k/255*log(2000));
    CHECK(!cmf_nonthermal_atomic_read(&raw,argv[6]));
    CHECK(!cmf_nonthermal_excitation_build(a,energy,256,20,&exc));
    CHECK(!cmf_nonthermal_binding_build_policy(a,&raw,&exc,argv[7],CMF_NT_SIGMA_SOURCE_KINETIC,&bound));
    CHECK(bound.n_resolved_routes==41&&bound.n_unresolved_routes==10&&bound.n_missing_ions==2);
    for(int fl=0;fl<n->n_nlte_levels_total;fl++)one_pop[fl]=saved_pop[(size_t)fl*ns+s];
    NLTEConfig single=*n;single.nlte_level_populations=one_pop;
    PlasmaState single_p={.n_shells=1,.n_electron=p->n_electron+s};
    GammaDeposition single_g=gamma;single_g.n_shells=1;single_g.heating_rate=heat+s;
    CHECK(!cmf_nonthermal_bound_evaluate(&bound.view,a,&single,&single_p,&single_g,&one));
    rates=one;rates.n_shells=ns;rates.rate_per_particle=calloc((size_t)rates.n_transitions*ns,sizeof(double));
    CHECK(rates.rate_per_particle);
    for(int k=0;k<rates.n_transitions;k++)rates.rate_per_particle[(size_t)k*ns+s]=one.rate_per_particle[k];
    gamma.nonthermal_rates=&rates;
    ions=fopen("ions.csv","wx");levels=fopen("levels.csv","wx");CHECK(ions&&levels);
    CHECK(fprintf(ions,"iteration,field_iteration,shell,Z,charge,T_K,ne_cm3,before_cm3,after_cm3,nuclei_cm3\n")>0);
    CHECK(fprintf(levels,"iteration,shell,Z,charge,global_level,local_level,energy_eV,before_cm3,after_cm3,nuclei_cm3\n")>0);
    for(int e=0;e<a->n_elements;e++){
        int z=a->element_Z[e];if(z!=14&&z!=16&&z!=20&&z!=27)continue;
        memcpy(n->nlte_level_populations,saved_pop,np*sizeof(double));memcpy(a->ion_number_density,saved_ion,ni*sizeof(double));
        double total=a->abundances[(size_t)e*ns+s]*p->rho[s]/(a->element_mass_amu[e]*AMU);
        printf("COLLISION_POP_BEGIN iteration=%d field=%d shell=%d Z=%d fixed_NT=1\n",iteration,field_iteration,s,z);
        CHECK(nlte_element_initial_steady_solve(n,a,p,&m.opacity,e,s,request.epoch,&gamma)==POP_OK);
        long double sum=0,l1=0;
        for(int ip=a->elem_ion_offset[e];ip<a->elem_ion_offset[e+1];ip++){
            double value=a->ion_number_density[(size_t)ip*ns+s];CHECK(value>=0&&isfinite(value));sum+=value;
            CHECK(fprintf(ions,"%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",iteration,field_iteration,s,z,
                a->ion_pop_stage[ip],p->T_e[s],p->n_electron[s],saved_ion[(size_t)ip*ns+s],value,total)>0);
            for(int gl=a->level_offset[ip];gl<a->level_offset[ip+1];gl++){
                int fl=n->global_to_nlte_level[gl];CHECK(fl>=0);
                double before=saved_pop[(size_t)fl*ns+s],after=n->nlte_level_populations[(size_t)fl*ns+s];
                CHECK(after>=0&&isfinite(after));l1+=fabs(after-before);
                CHECK(fprintf(levels,"%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n",iteration,s,z,
                    a->ion_pop_stage[ip],gl,a->level_num[gl],a->level_energy_eV[gl],before,after,total)>0);
            }
        }
        CHECK(total>0&&fabsl(sum/total-1)<1e-10L);
        printf("COLLISION_POP_SOLVED iteration=%d Z=%d full_level_L1_per_nucleus=%.17Lg\n",iteration,z,l1/total);
    }
    puts("COLLISION_POP_PASS fixed_field_T_ne_NT=1 SF_feedback=0 charge_solved=0 RTE_solved=0 physical_history=0");rc=0;
done:
    if(ions&&fclose(ions))rc=1;if(levels&&fclose(levels))rc=1;
    gamma.nonthermal_rates=NULL;gamma_deposition_free(&gamma);
    /* rates.transitions는 one의 불변 배열을 빌렸다. */
    if(rates.rate_per_particle!=one.rate_per_particle)free(rates.rate_per_particle);
    cmf_nonthermal_rates_free(&one);cmf_nonthermal_binding_free(&bound);
    cmf_nonthermal_excitation_free(&exc);cmf_nonthermal_atomic_free(&raw);
    radiation_field_owner_free(&owner);cmf_toy_model_free(&m);
    free(fine);free(nu);free(dnu);free(zero);free(jbar);free(se);free(ids);free(count);free(line_valid);
    free(edges);free(coarse);free(valid);free(saved_pop);free(saved_ion);free(one_pop);free(heat);
    return rc;
}
