/* S III 인구 -> 새 장 -> 열적 전리비의 국소 민감도. 새 SE 해가 아니다. */
#define main unused_feedback_bf_main
#include "cmf_toy_bf_iterate_diagnostic.c"
#undef main

static int sulfur_shift(AtomicData *a,NLTEConfig *n,PlasmaState *p,int shell,double delta,
                        double *old_ratio,double *new_ratio){
    if(!a||!n||!p||!old_ratio||!new_ratio||!isfinite(delta)||p->n_shells<1||
       shell<0||shell>=p->n_shells)return -1;
    int ns=p->n_shells,ip[2]={-1,-1};
    for(int k=0;k<a->n_ion_pops;k++)if(a->ion_pop_Z[k]==16){
        if(a->ion_pop_stage[k]==1)ip[0]=k;
        if(a->ion_pop_stage[k]==2)ip[1]=k;
    }
    if(ip[0]<0||ip[1]<0)return -1;
    double lower=a->ion_number_density[(size_t)ip[0]*ns+shell];
    double upper=a->ion_number_density[(size_t)ip[1]*ns+shell];
    double change=upper*delta,lo=lower-change,hi=upper+change;
    double density=p->n_electron[shell]+change;
    if(!(lower>0&&upper>0&&lo>0&&hi>0&&density>0)||
       !isfinite(lo+hi+density))return -1;
    double factor[2]={lo/lower,hi/upper};
    /* 전건 확인 뒤에만 private 상태를 수정한다. 잘못된 후보는 무변경 거부. */
    for(int k=0;k<2;k++){
        long double sum=0;
        for(int gl=a->level_offset[ip[k]];gl<a->level_offset[ip[k]+1];gl++){
            int fl=n->global_to_nlte_level[gl];
            if(fl<0||fl>=n->n_nlte_levels_total)return -1;
            double v=n->nlte_level_populations[(size_t)fl*ns+shell];
            if(!(v>=0)||!isfinite(v)||!isfinite(v*factor[k]))return -1;
            sum+=v;
        }
        double ion=k?upper:lower;
        if(fabsl(sum/ion-1)>1e-10L)return -1;
    }
    *old_ratio=upper/lower;*new_ratio=hi/lo;
    for(int k=0;k<2;k++)for(int gl=a->level_offset[ip[k]];gl<a->level_offset[ip[k]+1];gl++)
        n->nlte_level_populations[(size_t)n->global_to_nlte_level[gl]*ns+shell]*=factor[k];
    a->ion_number_density[(size_t)ip[0]*ns+shell]=lo;
    a->ion_number_density[(size_t)ip[1]*ns+shell]=hi;
    p->n_electron[shell]=density;
    return 0;
}

static int shift_test(void){
    int z[]={16,16},q[]={1,2},off[]={0,2,4},map[]={0,1,2,3};
    const double bad[]={-1,5,NAN};
    for(int sign=-1;sign<=1;sign++){
        double ions[]={80,160,20,40},levels[]={20,40,60,120,8,16,12,24},ne[]={120,240};
        AtomicData a={0};NLTEConfig n={0};PlasmaState p={0};
        a.n_ion_pops=2;a.ion_pop_Z=z;a.ion_pop_stage=q;a.level_offset=off;a.ion_number_density=ions;
        n.n_nlte_levels_total=4;n.global_to_nlte_level=map;n.nlte_level_populations=levels;
        p.n_shells=2;p.n_electron=ne;double before,after;
        for(int k=0;k<3;k++){
            double si[4],sl[8],sn[2];memcpy(si,ions,sizeof si);memcpy(sl,levels,sizeof sl);memcpy(sn,ne,sizeof sn);
            if(!sulfur_shift(&a,&n,&p,0,bad[k],&before,&after)||memcmp(si,ions,sizeof si)||
               memcmp(sl,levels,sizeof sl)||memcmp(sn,ne,sizeof sn))return -1;
        }
        if(sulfur_shift(&a,&n,&p,0,sign*.01,&before,&after))return -1;
        if(fabs(ions[0]+ions[2]-100)>1e-12||fabs(ne[0]-ions[0]-2*ions[2])>1e-12||
           ions[1]!=160||ions[3]!=40||ne[1]!=240||
           levels[1]!=40||levels[3]!=120||levels[5]!=16||levels[7]!=24||
           fabs(levels[0]/ions[0]-.25)>1e-14||fabs(levels[4]/ions[2]-.4)>1e-14||
           fabs((levels[0]+levels[2])/ions[0]-1)>1e-14||
           fabs((levels[4]+levels[6])/ions[2]-1)>1e-14||before!=.25||after!=ions[2]/ions[0])return -1;
    }
    puts("FEEDBACK_SHIFT_SELFTEST_PASS both_signs=1 nuclei_charge=1 within_ion_ratio=1 untouched_shell=1 invalid_transactional=1");
    return 0;
}

static int rates(const CMFToyModel *m,double *j,const char *path){
    CMFToyModel v=*m;
    v.nlte.radfield_view_status=RADIATION_FIELD_VIEW_DISABLED;
    v.nlte.radiation_field.field.generation.computed_generation=0;
    FILE *fp=fopen(path,"wx");if(!fp)return -1;
    int rc=fprintf(fp,"field_iteration,population_iteration,shell,Z,velocity_km_s,T_K,ne_cm3,III_II,channels,photo_per_II_s,recomb_per_III_s,collision_per_II_s,threebody_per_III_s,forward_cm3_s,reverse_cm3_s,signed_imbalance\n")<0;
    if(!rc)rc=evaluate(&v,j,v.nlte.nlte_level_populations,v.atom.ion_number_density,
        v.plasma.n_electron,v.plasma.T_e,0,0,fp);
    if(fclose(fp))rc=-1;return rc;
}

int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--selftest"))return shift_test()?1:0;
    if(argc!=7)return 2;
    const double mass[]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    char *end=NULL;double delta=strtod(argv[5],&end);
    if(!end||*end||!(delta==0||delta==.01||delta==-.01))return 2;
    CMFToyModel m={0};CMFGENState cs={0};double *old=NULL,*material=NULL;FILE *fp=NULL;
    int rc=-1,it=-1;
    REQUIRE(!getenv("LUMINA_CMF_FINE_LINE_EPS")&&!getenv("LUMINA_INNER_BB_SCALE"));
    REQUIRE(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    REQUIRE(!cmf_toy_initial_iterate_read(&m,argv[3],argv[4],&it));
    int ns=m.plasma.n_shells,nb=m.opacity.n_fine,ni=m.atom.n_ion_pops,nl=m.nlte.n_nlte_levels_total;
    REQUIRE(ns==113&&nb==117606&&it==4&&!m.plasma.physical_history);
    double ratio0,ratio1,ne0=m.plasma.n_electron[39];
    REQUIRE(!sulfur_shift(&m.atom,&m.nlte,&m.plasma,39,delta,&ratio0,&ratio1));
    double charge=0;
    for(int ip=0;ip<ni;ip++)charge+=m.atom.ion_pop_stage[ip]*m.atom.ion_number_density[(size_t)ip*ns+39];
    REQUIRE(charge>0&&fabs(m.plasma.n_electron[39]/charge-1)<1e-8);
    m.atom.population_committed_generation++;
    m.nlte.population_required_generation=m.nlte.population_committed_generation=m.atom.population_committed_generation;
    PopulationAtomicView av={0};AtomicData *a=&m.atom;
    av.n_ions=ni;av.n_levels=a->n_levels;av.level_offset=a->level_offset;av.energy_eV=a->level_energy_eV;
    av.g=a->level_g;av.level_Z=a->level_Z;av.level_ion=a->level_ion;av.topion_n=a->topion_n;
    av.topion_ion_index=a->topion_ion_index;av.topion_E_cm=a->topion_E_cm;av.topion_g=a->topion_g;
    REQUIRE(population_partition_build(&av,m.plasma.T_e,ns,a->population_committed_generation,
        m.plasma.T_e_generation,a->partition_functions,&a->partition_stamp)==POP_OK);
    REQUIRE(!nlte_precompute_within_sl_frac(&m.nlte,a,&m.plasma,ns));
    nlte_update_tau_sobolev(&m.nlte,a,&m.opacity,m.geometry.time_explosion,ns);
    REQUIRE(!tau_sobolev_assert_fresh(&m.opacity,"private S III sensitivity"));
    size_t cells=(size_t)ns*nb,nm=(size_t)(2+ni+nl)*ns;
    old=malloc(cells*sizeof(double));material=malloc(nm*sizeof(double));REQUIRE(old&&material);
    memcpy(old,m.opacity.jnu_fine,cells*8);
    memcpy(material,m.plasma.T_e,ns*8);memcpy(material+ns,m.plasma.n_electron,ns*8);
    memcpy(material+2*ns,a->ion_number_density,(size_t)ni*ns*8);
    memcpy(material+(2+ni)*ns,m.nlte.nlte_level_populations,(size_t)nl*ns*8);
    REQUIRE(!rates(&m,old,"stored_field_bf.csv"));
    REQUIRE(!cmfgen_init(&cs,&m.geometry));
    REQUIRE(!cmfgen_initial_radiation_publish(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,
        &m.nlte,a,m.plasma.T_e[0],0));
    REQUIRE(!rates(&m,m.opacity.jnu_fine,"new_field_bf.csv"));
    REQUIRE(!memcmp(material,m.plasma.T_e,ns*8)&&!memcmp(material+ns,m.plasma.n_electron,ns*8)&&
        !memcmp(material+2*ns,a->ion_number_density,(size_t)ni*ns*8)&&
        !memcmp(material+(2+ni)*ns,m.nlte.nlte_level_populations,(size_t)nl*ns*8));
    fp=fopen("field_change.csv","wx");REQUIRE(fp);
    REQUIRE(fprintf(fp,"shell,relative_J_integral\n")>0);double worst=0;
    for(int s=0;s<ns;s++){
        long double before=0,after=0;
        for(int b=0;b<nb;b++){
            double w=m.opacity.nu_fine[b]*2*sinh(.5*m.opacity.dlognu_fine);
            before+=(long double)old[(size_t)s*nb+b]*w;
            after+=(long double)m.opacity.jnu_fine[(size_t)s*nb+b]*w;
        }
        REQUIRE(before>0&&after>0);double change=(double)(after/before-1);
        worst=fmax(worst,fabs(change));REQUIRE(fprintf(fp,"%d,%.17g\n",s,change)>0);
    }
    {int closed=fclose(fp);fp=NULL;REQUIRE(!closed);}
    REQUIRE(delta!=0||worst<1e-6);
    REQUIRE(!cmf_toy_model_write_material(&m,"diagnostic_material.csv","DIAGNOSTIC_PERTURBED_MATERIAL; NOT_CONVERGED; NOT_PHYSICAL_HISTORY"));
    REQUIRE(!cmf_toy_initial_iterate_write(&m,"diagnostic_state.bin",argv[6],0));
    fp=fopen("perturbation.json","wx");REQUIRE(fp);
    REQUIRE(fprintf(fp,"{\"source_iteration\":4,\"full_SE_updates\":0,\"physical_history\":false,\"shell\":39,\"delta\":%.17g,\"old_ratio\":%.17g,\"new_ratio\":%.17g,\"old_ne\":%.17g,\"new_ne\":%.17g,\"charge_relative\":%.17g,\"maximum_J_integral_change\":%.17g}\n",
        delta,ratio0,ratio1,ne0,m.plasma.n_electron[39],fabs(m.plasma.n_electron[39]/charge-1),worst)>0);
    {int closed=fclose(fp);fp=NULL;REQUIRE(!closed);}
    printf("FEEDBACK_FIELD_DONE delta=%.17g shell=39 full_SE_updates=0 physical_history=0 final_goal=NOT_JUDGED\n",delta);rc=0;
done:
    if(fp&&fclose(fp))rc=-1;
    free(old);free(material);cmfgen_free(&cs);cmf_toy_model_free(&m);return rc?1:0;
}
