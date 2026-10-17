/* 원본 온도 표현만 다른 두 격자의 실제 공동 갱신. 생산 옵션은 추가하지 않는다. */
#define cmf_toy_initial_main unused_temperature_initial_main
#define cmf_toy_step_main unused_temperature_step_main
#include "../src/cmf_toy_run.c"
#undef cmf_toy_initial_main
#undef cmf_toy_step_main
#include <float.h>

extern void mon_interp_(double*,int*,int*,double*,int*,double*,int*,double*,int*);
static int parent_shell[256];
static double initial_energy_relative;
static double cube_volume(double a,double b){return (b-a)*(b*b+b*a+a*a);}
static int temperature_at(const CMFToyEjecta *raw,double v,double *t){
    int n=raw->n_shells,one=1;double x[8192],y[8192],q=log(v),value;
    if(n<3||n>8192||!(v>0))return -1;
    for(int s=0;s<n;s++){
        x[s]=log(.5*(raw->velocity_edges[s]+raw->velocity_edges[s+1]));
        y[s]=log(raw->initial_temperature[s]);
    }
    if(q<x[0]||q>x[n-1])return -1;
    mon_interp_(&value,&one,&one,&q,&one,y,&n,x,&n);
    *t=exp(value);return !isfinite(*t)||!(*t>0)?-1:0;
}

/* 시험 빌드에서 cmf_toy_model.c의 reader 호출만 이 함수로 연결한다. */
int temperature_ejecta_read_refined(CMFToyEjecta *out,const char *path,int requested,
    const double masses[6],double ratio,int max_shells){
    const char *mode=getenv("CMF_TEST_T_RECON_MODE");
    if(!mode)return cmf_toy_ejecta_read_refined(out,path,requested,masses,ratio,max_shells);
    if(strcmp(mode,"constant")&&strcmp(mode,"reconstructed"))return -1;
    CMFToyEjecta old={0},raw={0},next={0};int rc=-1,count=0;
    if(cmf_toy_ejecta_read_refined(&old,path,requested,masses,ratio,max_shells)||old.n_shells!=113||
       cmf_toy_ejecta_read(&raw,path,0,masses)||raw.n_shells!=807)goto done;
    for(int s=0;s<old.n_shells;s++){
        double a=old.velocity_edges[s],b=old.velocity_edges[s+1];
        int split=a>=12100e5&&b<=12500e5;
        int parts=split?3*(int)llround((b-a)/50e5):1;
        if(parts<1||(split&&fabs((b-a)/50e5-parts/3)>1e-12))goto done;
        count+=parts;
    }
    if(count!=132||count>max_shells||count>256)goto done;
    next=old;next.n_shells=count;next.storage=calloc(9*(size_t)count+1,sizeof(double));
    if(!next.storage){memset(&next,0,sizeof next);goto done;}
    next.velocity_edges=next.storage;next.mass_g=next.storage+count+1;
    next.initial_temperature=next.mass_g+count;next.number_per_g=next.initial_temperature+count;
    int at=0;long double old_energy=0,new_energy=0;
    for(int s=0;s<old.n_shells;s++){
        double a=old.velocity_edges[s],b=old.velocity_edges[s+1],vol=cube_volume(a,b);
        int split=a>=12100e5&&b<=12500e5,parts=split?3*(int)llround((b-a)/50e5):1;
        long double total=0;
        old_energy+=(long double)vol*powl(old.initial_temperature[s],4);
        for(int k=0;k<parts;k++,at++){
            double lo=k?next.velocity_edges[at]:a,hi=k+1==parts?b:a+(b-a)*(k+1)/parts;
            double v=cube_volume(lo,hi),T=old.initial_temperature[s];
            if(split&&!strcmp(mode,"reconstructed")&&temperature_at(&raw,.5*(lo+hi),&T))goto done;
            next.velocity_edges[at]=lo;next.velocity_edges[at+1]=hi;
            next.mass_g[at]=old.mass_g[s]*(v/vol);next.initial_temperature[at]=T;
            for(int e=0;e<6;e++)next.number_per_g[e*count+at]=old.number_per_g[e*old.n_shells+s];
            parent_shell[at]=s;total+=next.mass_g[at];new_energy+=(long double)v*powl(T,4);
            if(!(hi>lo)||!(next.mass_g[at]>0)||!isfinite(next.mass_g[at]))goto done;
        }
        if(fabsl(total/old.mass_g[s]-1)>1e-12L)goto done;
    }
    if(at!=count)goto done;
    initial_energy_relative=(double)(new_energy/old_energy-1);
    cmf_toy_ejecta_free(out);*out=next;memset(&next,0,sizeof next);rc=0;
done:
    cmf_toy_ejecta_free(&old);cmf_toy_ejecta_free(&raw);cmf_toy_ejecta_free(&next);return rc;
}

typedef struct {uint64_t h[7];double par[3],*data;int *maps;} Donor;
static int donor_read(Donor *d,const char *path,const char *identity){
    /* 이전 프로세스의 정식 reader 검증 뒤 같은 hash의 파일을 다시 읽는다.
     * 새 프로세스에서 bootstrap 1회 규칙을 우회하거나 재설정하지 않는다. */
    FILE *fp=fopen(path,"rb");if(!fp)return -1;char mark[8],id[64];int rc=-1;
#define DGET(v,z,c) do{if(fread(v,z,c,fp)!=(c))goto done;}while(0)
    DGET(mark,1,8);DGET(id,1,64);DGET(d->h,8,7);DGET(d->par,8,3);
    if(memcmp(mark,"CMFITER1",8)||strlen(identity)!=64||memcmp(id,identity,64)||
       d->h[0]!=113||d->h[1]!=27||d->h[2]!=19627||d->h[3]!=6||d->h[4]!=117606||
       !d->h[5]||d->h[5]==UINT64_MAX||d->h[6]!=4||d->par[0]!=172800)goto done;
    size_t ns=d->h[0],ni=d->h[1],nl=d->h[2],ne=d->h[3],nb=d->h[4],n=(5+ne+ni+nl+nb)*ns;
    d->maps=malloc((nl+2*ni)*sizeof(int));d->data=malloc(n*sizeof(double));
    if(!d->maps||!d->data)goto done;
    DGET(d->maps,4,nl+2*ni);DGET(d->data,8,n);DGET(mark,1,8);
    if(memcmp(mark,"ITEREND1",8)||fgetc(fp)!=EOF)goto done;
    for(size_t k=0;k<n;k++)if(d->data[k]<0||!isfinite(d->data[k]))goto done;
    rc=0;
done:
    fclose(fp);return rc;
#undef DGET
}
static int remap_predictor(CMFToyModel *m,const Donor *old){
    int ns=m->plasma.n_shells,os=old->h[0],nb=old->h[4];
    AtomicData *a=&m->atom;NLTEConfig *n=&m->nlte;PlasmaState *p=&m->plasma;OpacityState *o=&m->opacity;
    int ni=a->n_ion_pops,nl=n->n_nlte_levels_total;
    if(ns!=132||os!=113||nb!=117606||ni!=old->h[1]||nl!=old->h[2]||a->n_elements!=old->h[3]||
       memcmp(n->nlte_to_global_level,old->maps,nl*sizeof(int))||
       memcmp(a->ion_pop_Z,old->maps+nl,ni*sizeof(int))||
       memcmp(a->ion_pop_stage,old->maps+nl+ni,ni*sizeof(int))||o->jnu_fine)return -1;
    const double *ri=old->data,*ro=ri+os,*rho=ro+os,*T=rho+os,*ab=T+os;
    const double *ne=ab+a->n_elements*os,*ion=ne+os,*level=ion+ni*os,*J=level+(size_t)nl*os;
    double *j=malloc((size_t)ns*nb*sizeof(double)),*nu=malloc(nb*sizeof(double));
    if(!j||!nu){free(j);free(nu);return -1;}
    for(int s=0;s<ns;s++){
        int q=parent_shell[s];double charge=0;
        if(q<0||q>=os||fabs(p->rho[s]/rho[q]-1)>1e-12||
           m->geometry.r_inner[s]<ri[q]-4*DBL_EPSILON*ri[q]||
           m->geometry.r_outer[s]>ro[q]+4*DBL_EPSILON*ro[q])goto fail;
        if(!strcmp(getenv("CMF_TEST_T_RECON_MODE"),"constant")&&p->T_e[s]!=T[q])goto fail;
        for(int e=0;e<a->n_elements;e++)if(fabs(a->abundances[e*ns+s]-ab[e*os+q])>1e-14)goto fail;
        p->n_electron[s]=ne[q];
        for(int ip=0;ip<ni;ip++){
            double v=ion[ip*os+q],sum=0;
            a->ion_number_density[ip*ns+s]=v;charge+=a->ion_pop_stage[ip]*v;
            for(int gl=a->level_offset[ip];gl<a->level_offset[ip+1];gl++){
                int fl=n->global_to_nlte_level[gl];if(fl<0||fl>=nl)goto fail;
                double pop=level[fl*os+q];
                n->nlte_level_populations[fl*ns+s]=pop;sum+=pop;
            }
            if(a->level_offset[ip]<a->level_offset[ip+1]&&(v==0?sum!=0:fabs(sum/v-1)>1e-10))goto fail;
        }
        if(!(charge>0)||fabs(p->n_electron[s]/charge-1)>1e-8)goto fail;
        memcpy(j+(size_t)s*nb,J+(size_t)q*nb,nb*sizeof(double));
    }
    a->population_committed_generation=old->h[5]+1;
    n->population_required_generation=n->population_committed_generation=a->population_committed_generation;
    PopulationAtomicView v={0};v.n_ions=ni;v.n_levels=a->n_levels;v.level_offset=a->level_offset;
    v.energy_eV=a->level_energy_eV;v.g=a->level_g;v.level_Z=a->level_Z;v.level_ion=a->level_ion;
    v.topion_n=a->topion_n;v.topion_ion_index=a->topion_ion_index;v.topion_E_cm=a->topion_E_cm;v.topion_g=a->topion_g;
    if(population_partition_build(&v,p->T_e,ns,a->population_committed_generation,p->T_e_generation,
       a->partition_functions,&a->partition_stamp)!=POP_OK||nlte_precompute_within_sl_frac(n,a,p,ns))goto fail;
    nlte_update_tau_sobolev(n,a,o,m->geometry.time_explosion,ns);
    if(tau_sobolev_assert_fresh(o,"temperature representation predictor"))goto fail;
    for(int b=0;b<nb;b++)nu[b]=old->par[1]*exp((b+.5)*old->par[2]);
    free(o->nu_fine);
    o->nu_fine=nu;o->jnu_fine=j;o->n_fine=nb;
    o->nu_lo_fine=old->par[1];o->dlognu_fine=old->par[2];
    return 0;
fail:
    free(j);free(nu);return -1;
}

static int save_response(void *ctx,int iteration){
    CMFToyModel *m=ctx;char path[80];
    snprintf(path,sizeof path,"response_material_%d.csv",iteration);
    if(cmf_toy_model_write_material(m,path,"DIAGNOSTIC_INITIAL_ITERATE; NOT_CONVERGED; NOT_PHYSICAL_HISTORY"))return -1;
    snprintf(path,sizeof path,"response_field_%d.csv",iteration);FILE *fp=fopen(path,"wx");if(!fp)return -1;
    int rc=fprintf(fp,"shell,parent,J_integral\n")<0;
    for(int s=0;s<m->plasma.n_shells&&!rc;s++){
        long double sum=0;
        for(int b=0;b<m->opacity.n_fine;b++)sum+=(long double)m->opacity.jnu_fine[(size_t)s*m->opacity.n_fine+b]*
            m->opacity.nu_fine[b]*2*sinh(.5*m->opacity.dlognu_fine);
        rc=fprintf(fp,"%d,%d,%.17Lg\n",s,parent_shell[s],sum)<0;
    }
    if(fclose(fp))rc=1;
    printf("TEMPERATURE_RESPONSE_SAVED iteration=%d physical_history=0\n",iteration);fflush(stdout);
    return rc?-1:0;
}
static int selftest(const char *path){
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyEjecta raw={0},a={0},b={0};int rc=-1;double worst=0;
    if(cmf_toy_ejecta_read(&raw,path,0,mass))goto done;
    for(int s=0;s<raw.n_shells;s++){
        double t;if(temperature_at(&raw,.5*(raw.velocity_edges[s]+raw.velocity_edges[s+1]),&t))goto done;
        worst=fmax(worst,fabs(t/raw.initial_temperature[s]-1));
    }
    if(worst>1e-12||setenv("CMF_TEST_T_RECON_MODE","constant",1)||
       temperature_ejecta_read_refined(&a,path,100,mass,1.1,256)||fabs(initial_energy_relative)>1e-12)goto done;
    if(setenv("CMF_TEST_T_RECON_MODE","reconstructed",1)||temperature_ejecta_read_refined(&b,path,100,mass,1.1,256)||
       memcmp(a.mass_g,b.mass_g,a.n_shells*sizeof(double))||
       memcmp(a.number_per_g,b.number_per_g,6*a.n_shells*sizeof(double))||
       memcmp(a.velocity_edges,b.velocity_edges,(a.n_shells+1)*sizeof(double)))goto done;
    {
        int n=4,q=3,one=1;double x[]={0,1,2,3},y[]={1,3,5,7},xx[]={.2,1.2,2.8},out[3];
        mon_interp_(out,&q,&one,xx,&q,y,&n,x,&n);
        for(int i=0;i<q;i++)if(fabs(out[i]-(2*xx[i]+1))>2e-15)goto done;
        for(int i=0;i<n;i++)y[i]=7;
        mon_interp_(out,&q,&one,xx,&q,y,&n,x,&n);
        for(int i=0;i<q;i++)if(out[i]!=7)goto done;
    }
    printf("TEMPERATURE_MAPPING_PASS shells=%d node_relative=%.17g mass_nuclei_preserved=1 paired_geometry_identical=1 radiation_T4_relative=%.17g\n",b.n_shells,worst,initial_energy_relative);rc=0;
done:
    cmf_toy_ejecta_free(&raw);cmf_toy_ejecta_free(&a);cmf_toy_ejecta_free(&b);return rc;
}

int main(int argc,char **argv){
    if(argc==3&&!strcmp(argv[1],"--selftest"))return selftest(argv[2])?1:0;
    const double mass[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    if(argc==6&&!strcmp(argv[1],"--verify-donor")){
        CMFToyModel donor={0};int iteration=-1;
        int rc=unsetenv("CMF_TEST_T_RECON_MODE")||cmf_toy_model_prepare_runtime(&donor,argv[2],argv[3],100,mass)||
            cmf_toy_initial_iterate_read(&donor,argv[4],argv[5],&iteration)||iteration!=4;
        cmf_toy_model_free(&donor);
        if(!rc)puts("TEMPERATURE_DONOR_VALIDATED source_iteration=4 physical_history=0");
        return rc?1:0;
    }
    if(argc!=9||(strcmp(argv[5],"constant")&&strcmp(argv[5],"reconstructed")))return 2;
    Donor old={0};CMFToyModel m={0};CMFGammaNuclearData nuclear={0};CMFToyGammaSource source={0};
    CMFGammaResult dep={0};GammaDeposition gamma={0};ToyNonthermal nt={0};CMFGENState cs={0};
    CMFPhysicalTimeState initial={0};CMFInitialNLTEReport report={0};FILE *fp=NULL;int rc=1;
#define TREQ(x) do{if(!(x)){fprintf(stderr,"TEMPERATURE_RESPONSE_FAIL line=%d %s\n",__LINE__,#x);goto done;}}while(0)
    TREQ(!donor_read(&old,argv[3],argv[4]));
    TREQ(!setenv("CMF_TEST_T_RECON_MODE",argv[5],1));
    TREQ(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],100,mass));
    TREQ(!remap_predictor(&m,&old));
    free(old.data);free(old.maps);memset(&old,0,sizeof old);
    TREQ(!cmf_toy_model_write_material(&m,"mapped_predictor.csv","DIAGNOSTIC_REMAPPED_GUESS; J_PREDICTOR_ONLY; NO_PHYSICAL_HISTORY"));
    TREQ(!toy_nonthermal_prepare(&nt,&m.atom,argv[7],argv[8]));
    TREQ(!cmf_gamma_nuclear_read(&nuclear,argv[6])&&
        !cmf_toy_gamma_source_build(&source,&m.ejecta,&nuclear,m.geometry.time_explosion)&&
        !cmf_gamma_transport_shell_mixture(&source.input,200000,UINT64_C(2026101502),.5,&dep));
    int ns=m.plasma.n_shells;gamma_deposition_init(&gamma,ns);TREQ(gamma.heating_rate&&gamma.nonthermal_ioniz_rate);
    fp=fopen("gamma.csv","wx");TREQ(fp&&fprintf(fp,"shell,heating_rate\n")>0);
    for(int s=0;s<ns;s++){
        gamma.heating_rate[s]=dep.deposition_erg_cm3_s[s]+source.epoch.positron_emission[s];
        TREQ(fprintf(fp,"%d,%.17g\n",s,gamma.heating_rate[s])>0);
    }
    {int e=fclose(fp);fp=NULL;TREQ(!e);}
    TREQ(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_NONLOCAL_MC,m.geometry.time_explosion,NULL,NULL,NULL,gamma.heating_rate));
    gamma.nonthermal_provider=&nt.provider;
    fp=fopen("temperature.csv","wx");TREQ(fp);
    for(int s=0;s<ns;s++)TREQ(fprintf(fp,"%d %.17g\n",s,m.plasma.T_e[s])>0);
    {int e=fclose(fp);fp=NULL;TREQ(!e);}
    TREQ(!setenv("LUMINA_FIXED_TE_PROFILE","temperature.csv",1)&&!cmfgen_init(&cs,&m.geometry));
    printf("TEMPERATURE_RESPONSE_START mode=%s shells=%d radiation_T4_relative=%.17g source_iteration=4 iterations_requested=1\n",argv[5],ns,initial_energy_relative);fflush(stdout);
    CMFTimeEpochStatus status=cmfgen_solve_initial_nlte_saved(&cs,&m.geometry,&m.opacity,&m.bf,&m.plasma,
        &m.nlte,&m.atom,&gamma,m.plasma.T_e[0],1,1e-5,&initial,&report,save_response,&m);
    TREQ(status==CMF_EPOCH_NOT_CONVERGED&&report.iterations==1&&report.radiation_calls==2&&!report.converged);
    TREQ(!m.plasma.physical_history);
    printf("TEMPERATURE_RESPONSE_DONE mode=%s ne_change=%.17g population_change=%.17g J_change=%.17g full_SE_updates=1 converged=0 physical_history=0 final_goal=NOT_JUDGED\n",
        argv[5],report.electron_update,report.population_update,report.radiation_update);rc=0;
done:
    if(fp)fclose(fp);
    cmf_physical_time_state_free(&initial);cmfgen_free(&cs);gamma_deposition_free(&gamma);toy_nonthermal_free(&nt);
    cmf_gamma_result_free(&dep);cmf_toy_gamma_source_free(&source);free(old.data);free(old.maps);cmf_toy_model_free(&m);return rc;
}
