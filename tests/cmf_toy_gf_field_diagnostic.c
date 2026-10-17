/* 고정 물질에서 CMFGEN형 약한 선 제외의 복사장 효과만 평가한다.
 * 생산 원자/SE는 변경하지 않는다. BF 율은 기존 검증된 진단 함수를 재사용한다. */
#define main unused_bf_diagnostic_main
#include "cmf_toy_bf_iterate_diagnostic.c"
#undef main

/* GENOSC_V9: 낮은 준위와 상위준위별 최소 하향 A(동률 포함)를 보호한다.
 * 번호는 입력에서는0-based, LEV_CUT은 CMFGEN의1-based다. */
static int weak_mask(const AtomicData *a,const NLTEConfig *n,double cut,
                     int lev_cut,int minimum,int atomic_cut,unsigned char *mask){
    if(!a||!n||!mask||!isfinite(cut)||cut<0||lev_cut<0||minimum<1||minimum>64)return -1;
    int *head=malloc((size_t)a->n_levels*sizeof(int));
    int *next=malloc((size_t)a->n_lines*sizeof(int));
    int *seen=malloc((size_t)a->n_levels*sizeof(int));int rc=-1;
    if(!head||!next||!seen)goto done;
    memset(mask,0,(size_t)a->n_lines);
    for(int g=0;g<a->n_levels;g++)head[g]=seen[g]=-1;
    for(int l=0;l<a->n_lines;l++){
        int lo=n->nlte_line_global[2*(size_t)l],up=n->nlte_line_global[2*(size_t)l+1];
        if(lo<0||lo>=a->n_levels||up<0||up>=a->n_levels||
           a->level_num[lo]>=a->level_num[up]||a->level_Z[lo]!=a->level_Z[up]||
           a->level_ion[lo]!=a->level_ion[up]||a->level_g[lo]<=0||
           !isfinite(a->line_A_ul[l])||a->line_A_ul[l]<0||
           !isfinite(a->line_f_lu[l])||a->line_f_lu[l]<0)goto done;
        next[l]=head[up];head[up]=l;
    }
    for(int up=0;up<a->n_levels;up++){
        double strongest[64]={0};
        for(int l=head[up];l>=0;l=next[l]){
            int lo=n->nlte_line_global[2*(size_t)l];
            if(seen[lo]==up)goto done;seen[lo]=up; /* 행렬 원소 중복은 조용히 합치지 않는다. */
            double value=a->line_A_ul[l];
            for(int k=0;k<minimum;k++)if(value>strongest[k]){
                for(int j=minimum-1;j>k;j--)strongest[j]=strongest[j-1];
                strongest[k]=value;break;
            }
        }
        if(!cut||a->level_Z[up]<atomic_cut||a->level_num[up]+1<=lev_cut||
           a->level_num[up]+1<=minimum)continue;
        double limit=strongest[minimum-1];
        for(int l=head[up];l>=0;l=next[l]){
            int lo=n->nlte_line_global[2*(size_t)l];
            mask[l]=a->level_num[lo]+1>lev_cut&&
                a->level_g[lo]*a->line_f_lu[l]<cut&&a->line_A_ul[l]<limit;
        }
    }
    rc=0;
done:
    free(head);free(next);free(seen);return rc;
}

static int selection_selftest(void){
    int Z[12],q[12],num[12],g[12];
    for(int i=0;i<12;i++){Z[i]=16;q[i]=1;num[i]=i;g[i]=2;}
    int maps[]={0,10,2,10,3,10,4,10,5,10,0,1,6,8};
    double f[]={.0001,.0001,.0001,.0001,.02,.0001,.0001};
    double A[]={.1,.2,5,5,1,.1,.1};unsigned char mask[7];
    AtomicData a={0};NLTEConfig n={0};a.n_levels=12;a.n_lines=7;
    a.level_Z=Z;a.level_ion=q;a.level_num=num;a.level_g=g;a.line_f_lu=f;a.line_A_ul=A;
    n.nlte_line_global=maps;
    if(weak_mask(&a,&n,.01,2,2,10,mask))return -1;
    for(int l=0;l<7;l++)if(mask[l]!=(l==1))return -1;
    if(weak_mask(&a,&n,0,2,2,10,mask))return -1;
    for(int l=0;l<7;l++)if(mask[l])return -1;
    if(weak_mask(&a,&n,.01,2,2,20,mask))return -1;
    for(int l=0;l<7;l++)if(mask[l])return -1;
    maps[12]=2;maps[13]=10;
    if(!weak_mask(&a,&n,.01,2,2,10,mask))return -1;
    puts("GF_SELECTION_SELFTEST_PASS low_levels=1 minimum=1 ties=1 strong=1 zero_cut=1 atomic_cut=1 duplicate_rejected=1");
    return 0;
}

static int bf_snapshot(const CMFToyModel *m,const char *path,int iteration){
    /* 새 소유자를 만들거나 복사본을 해제하지 않는다. 기존 함수에 읽기용 view만 준다. */
    CMFToyModel view=*m;
    view.nlte.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    view.nlte.radiation_field.field.generation.computed_generation=0;
    FILE *f=fopen(path,"wx");if(!f)return -1;
    int rc=fprintf(f,"field_iteration,population_iteration,shell,Z,velocity_km_s,T_K,ne_cm3,III_II,channels,photo_per_II_s,recomb_per_III_s,collision_per_II_s,threebody_per_III_s,forward_cm3_s,reverse_cm3_s,signed_imbalance\n")<0;
    if(!rc)rc=evaluate(&view,view.opacity.jnu_fine,view.nlte.nlte_level_populations,
        view.atom.ion_number_density,view.plasma.n_electron,view.plasma.T_e,iteration,iteration,f);
    if(fclose(f))rc=-1;return rc;
}

int main(int argc,char **argv){
    if(selection_selftest())return 1;
    if(argc==2&&!strcmp(argv[1],"--selftest"))return 0;
    if(argc!=7)return 2;
    char *end=NULL;double cut=strtod(argv[6],&end);
    if(!*argv[6]||*end||cut!=1e-3||!strcmp(argv[4],argv[5]))return 2;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};CMFGENState cs={0};unsigned char *mask=NULL;
    double *saved=NULL,*oldJ=NULL;int rc=-1,iteration=-1;FILE *out=NULL;
    REQUIRE(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    REQUIRE(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&iteration));
    AtomicData *a=&m.atom;NLTEConfig *n=&m.nlte;int ns=m.geometry.n_shells;
    REQUIRE(ns==113&&!m.plasma.physical_history);
    size_t ni=a->n_ion_pops,nl=n->n_nlte_levels_total,nb=m.opacity.n_fine;
    size_t nm=(2+ni+nl)*(size_t)ns;
    saved=malloc(nm*sizeof(double));oldJ=malloc((size_t)ns*nb*sizeof(double));
    mask=malloc((size_t)a->n_lines);REQUIRE(saved&&oldJ&&mask);
    memcpy(saved,m.plasma.T_e,ns*8);memcpy(saved+ns,m.plasma.n_electron,ns*8);
    memcpy(saved+2*ns,a->ion_number_density,ni*ns*8);
    memcpy(saved+(2+ni)*ns,n->nlte_level_populations,nl*ns*8);
    memcpy(oldJ,m.opacity.jnu_fine,(size_t)ns*nb*8);
    REQUIRE(!bf_snapshot(&m,"baseline_bf.csv",iteration));
    REQUIRE(!weak_mask(a,n,cut,9,10,10,mask));
    int removed=0,by_Z[29]={0};
    for(int l=0;l<a->n_lines;l++)if(mask[l]){
        REQUIRE(a->line_atomic_number[l]>=10&&a->line_atomic_number[l]<29);
        removed++;by_Z[a->line_atomic_number[l]]++;
        a->line_f_lu[l]=a->line_A_ul[l]=a->line_B_lu[l]=a->line_B_ul[l]=0;
    }
    REQUIRE(removed>0&&removed<a->n_lines);
    printf("GF_SELECTION cutoff=%.17g lev_cut=9 minimum=10 atomic_cut=10 removed=%d total=%d public_exact_parameters_known=0\n",cut,removed,a->n_lines);
    for(int z=10;z<29;z++)if(by_Z[z])printf("GF_SELECTION_ELEMENT Z=%d removed=%d\n",z,by_Z[z]);
    nlte_update_tau_sobolev(n,a,&m.opacity,m.geometry.time_explosion,ns);
    REQUIRE(!tau_sobolev_assert_fresh(&m.opacity,"private gf selection diagnostic"));
    for(int l=0;l<a->n_lines;l++)if(mask[l])for(int s=0;s<ns;s++)
        REQUIRE(m.opacity.tau_sobolev[(size_t)l*ns+s]==0);
    REQUIRE(!cmfgen_init(&cs,&m.geometry));
    REQUIRE(!cmfgen_initial_radiation_publish(&cs,&m.geometry,&m.opacity,&m.bf,
        &m.plasma,n,a,m.plasma.T_e[0],0));
    REQUIRE(nb==(size_t)m.opacity.n_fine);
    REQUIRE(!memcmp(saved,m.plasma.T_e,ns*8)&&!memcmp(saved+ns,m.plasma.n_electron,ns*8));
    REQUIRE(!memcmp(saved+2*ns,a->ion_number_density,ni*ns*8));
    REQUIRE(!memcmp(saved+(2+ni)*ns,n->nlte_level_populations,nl*ns*8));
    REQUIRE(!bf_snapshot(&m,"selected_bf.csv",iteration));
    REQUIRE(!cmf_toy_initial_iterate_write(&m,"selected_iterate.bin",argv[5],iteration));
    out=fopen("field_change.csv","wx");REQUIRE(out);
    REQUIRE(fprintf(out,"shell,J_integral_relative,J_integral_L1_relative\n")>0);
    for(int s=0;s<ns;s++){
        long double before=0,after=0,difference=0;
        for(size_t b=0;b<nb;b++){
            double width=m.opacity.nu_fine[b]*2*sinh(.5*m.opacity.dlognu_fine);
            double u=oldJ[(size_t)s*nb+b],v=m.opacity.jnu_fine[(size_t)s*nb+b];
            before+=(long double)u*width;after+=(long double)v*width;
            difference+=(long double)fabs(v-u)*width;
        }
        REQUIRE(before>0&&after>0);
        REQUIRE(fprintf(out,"%d,%.17Lg,%.17Lg\n",s,(after-before)/before,difference/fmaxl(before,after))>0);
    }
    puts("GF_FIELD_DONE frozen_material_byte_identical=1 field_recomputed=1 SE_solved=0 physical_history=0 final_goal=NOT_JUDGED");rc=0;
done:
    if(out&&fclose(out))rc=-1;
    free(mask);free(saved);free(oldJ);cmfgen_free(&cs);cmf_toy_model_free(&m);return rc?1:0;
}
